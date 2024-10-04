import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.Ring
import Mathlib.Analysis.Calculus.Angle
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Real
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Finset
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Ceva
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Triangle
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.Coprime.Basic
import Mathlib.NumberTheory.ModularArithmetic
import Mathlib.NumberTheory.Prime
import Mathlib.Probability
import Mathlib.SetTheory.Cardinal
import Mathlib.Tactic
import Mathlib.Tactic.Positivity
import Mathlib.Topology.Basic
import data.real.basic

namespace staircase_markup_199_cells_l24_24354

theorem staircase_markup_199_cells : ∀ n, (n = 199) → (∃ L : ℕ → ℕ, L 1 = 2 ∧ (∀ k, k ≥ 2 → L k = L (k - 1) + 1) ∧ L 199 = 200) :=
by {
  intros n hn,
  use (λ k, k + 1),
  split,
  { refl, },
  split,
  { intros k hk,
    rw nat.sub_add_cancel (le_of_lt_succ hk),
    refl, },
  { rwa hn, },
  sorry
}

end staircase_markup_199_cells_l24_24354


namespace total_spent_by_pete_and_raymond_l24_24277

def initial_money_in_cents : ℕ := 250
def pete_spent_in_nickels : ℕ := 4
def nickel_value_in_cents : ℕ := 5
def raymond_dimes_left : ℕ := 7
def dime_value_in_cents : ℕ := 10

theorem total_spent_by_pete_and_raymond : 
  (pete_spent_in_nickels * nickel_value_in_cents) 
  + (initial_money_in_cents - (raymond_dimes_left * dime_value_in_cents)) = 200 := sorry

end total_spent_by_pete_and_raymond_l24_24277


namespace sum_of_possible_values_l24_24233

def f (x : ℝ) : ℝ :=
  if x < 3 then 5 * x + 20 else 3 * x - 9

theorem sum_of_possible_values :
  (∃ x : ℝ, f x = 1 ∧ x < 3 ∨ f x = 1 ∧ x ≥ 3) →
  (∑ x in {nat.to_real (-19/5), nat.to_real (10/3)}, id x) = -7/15 :=
by
  -- This is only the statement. The proof is not required.
  sorry

end sum_of_possible_values_l24_24233


namespace sqrt_equality_l24_24472

theorem sqrt_equality :
  ∃ a b : ℕ, (0 < a ∧ 0 < b) ∧ a < b ∧ (real.sqrt (1 + real.sqrt (21 + 12 * real.sqrt 3)) = real.sqrt a + real.sqrt b) :=
by
  use 1, 3
  split
  { split
    { exact nat.pos_of_ne_zero (ne_of_gt zero_lt_one) }
    { exact nat.pos_of_ne_zero (ne_of_gt (by norm_num)) }
  }
  split
  { exact nat.lt_of_lt_of_le zero_lt_one (by norm_num) }
  { sorry }

end sqrt_equality_l24_24472


namespace range_of_m_l24_24151

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - 2 * x - m > 0) → m < -1 :=
begin
  sorry
end

end range_of_m_l24_24151


namespace tan_product_l24_24424

noncomputable def tan : ℝ → ℝ := sorry

theorem tan_product :
  (tan (Real.pi / 8)) * (tan (3 * Real.pi / 8)) * (tan (5 * Real.pi / 8)) = 2 * Real.sqrt 7 :=
by
  sorry

end tan_product_l24_24424


namespace crystal_run_final_segment_length_l24_24018

theorem crystal_run_final_segment_length :
  let north_distance := 2
  let southeast_leg := 1 / Real.sqrt 2
  let southeast_movement_north := -southeast_leg
  let southeast_movement_east := southeast_leg
  let northeast_leg := 2 / Real.sqrt 2
  let northeast_movement_north := northeast_leg
  let northeast_movement_east := northeast_leg
  let total_north_movement := north_distance + northeast_movement_north + southeast_movement_north
  let total_east_movement := southeast_movement_east + northeast_movement_east
  total_north_movement = 2.5 ∧ 
  total_east_movement = 3 * Real.sqrt 2 / 2 ∧ 
  Real.sqrt (total_north_movement^2 + total_east_movement^2) = Real.sqrt 10.75 :=
by
  sorry

end crystal_run_final_segment_length_l24_24018


namespace coefficient_x2_term_l24_24468

def polynomial1 : ℚ[X] := 3 * X^3 + 4 * X^2 + 5 * X
def polynomial2 : ℚ[X] := 2 * X^2 - 9 * X + 1

theorem coefficient_x2_term :
  polynomial1 * polynomial2.coeff 2 = -41 :=
sorry

end coefficient_x2_term_l24_24468


namespace prob_four_ones_in_five_rolls_l24_24574

open ProbabilityTheory

theorem prob_four_ones_in_five_rolls :
  let p_one := (1 : ℝ) / 6;
      p_not_one := 5 / 6;
      single_sequence_prob := p_one^4 * p_not_one;
      total_prob := (5 * single_sequence_prob)
  in
  total_prob = (25 / 7776) := 
by 
  sorry

end prob_four_ones_in_five_rolls_l24_24574


namespace repeating_decimal_sum_as_fraction_l24_24039

theorem repeating_decimal_sum_as_fraction : (0.3333... : ℚ) + (0.5656... : ℚ) = 89/99 :=
by
  let x := (0.3333... : ℚ)
  let y := (0.5656... : ℚ)
  have hx : x = 1/3 := sorry
  have hy : y = 56/99 := sorry
  calc
    x + y = 1/3 + 56/99 : by rw [hx, hy]
    ... = 267/297 : sorry
    ... = 89/99 : sorry


end repeating_decimal_sum_as_fraction_l24_24039


namespace cassie_nails_l24_24867

def num_dogs : ℕ := 4
def nails_per_dog_leg : ℕ := 4
def legs_per_dog : ℕ := 4
def num_parrots : ℕ := 8
def claws_per_parrot_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def extra_claws : ℕ := 1

def total_nails_to_cut : ℕ :=
  num_dogs * nails_per_dog_leg * legs_per_dog +
  num_parrots * claws_per_parrot_leg * legs_per_parrot + extra_claws

theorem cassie_nails : total_nails_to_cut = 113 :=
  by sorry

end cassie_nails_l24_24867


namespace train_passes_man_in_correct_time_l24_24394

noncomputable def train_passing_man_time (train_length : ℝ) (train_speed_kmh : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh + man_speed_kmh
  let relative_speed_ms := relative_speed_kmh * (5 / 18)
  train_length / relative_speed_ms

theorem train_passes_man_in_correct_time :
  train_passing_man_time 180 55 7 ≈ 10.45 :=
by
  sorry

end train_passes_man_in_correct_time_l24_24394


namespace sum_of_squares_of_roots_l24_24457

theorem sum_of_squares_of_roots : 
  (∃ r1 r2 : ℝ, r1 + r2 = 11 ∧ r1 * r2 = 12 ∧ (r1 ^ 2 + r2 ^ 2) = 97) := 
sorry

end sum_of_squares_of_roots_l24_24457


namespace sum_m_n_eq_three_l24_24878

theorem sum_m_n_eq_three 
  (m n : ℕ) -- m and n are positive integers
  (h1 : 0 < m) -- m is positive
  (h2 : 0 < n) -- n is positive
  (h3 : m + 5 < n) -- m + 5 < n
  (h4 : (m + (m + 3) + (m + 5) + n + (n + 2) + (2 * n - 1)) / 6 = n + 1) -- mean condition
  (h5 : ((m + 5) + n) / 2 = n + 1) -- median condition
  : m + n = 3 := 
sorry

end sum_m_n_eq_three_l24_24878


namespace zero_coeff_implies_empty_set_l24_24946

theorem zero_coeff_implies_empty_set (a : ℝ) : 
  (∀ S : set ℝ, {x | a * x = 1} ⊆ S) → a = 0 :=
by
  intro h
  -- The proof steps will be here
  sorry

end zero_coeff_implies_empty_set_l24_24946


namespace sum_remainder_l24_24237

theorem sum_remainder (a b c : ℕ) (h1 : a % 30 = 14) (h2 : b % 30 = 5) (h3 : c % 30 = 18) : 
  (a + b + c) % 30 = 7 :=
by
  sorry

end sum_remainder_l24_24237


namespace set_theorems_l24_24657

def U := {x : ℝ | true}

def A := {x : ℝ | x ≥ 3}

def B := {x : ℝ | x < 7}

def C (s : set ℝ) : set ℝ := {x : ℝ | x ∉ s}

theorem set_theorems :
  (A ∩ B = {x | 3 ≤ x ∧ x < 7}) ∧
  (A ∪ B = set.univ) ∧
  (C A = {x | x < 3}) ∧
  (C A ∩ B = {x | x < 3}) ∧
  (C A ∪ B = {x | x < 7}) := 
by
  sorry

end set_theorems_l24_24657


namespace faster_train_length_l24_24368

noncomputable def length_of_faster_train (speed_faster_train speed_slower_train crossing_time : ℝ) : ℝ :=
  let relative_speed := (speed_faster_train - speed_slower_train) * (5 / 18) in
  relative_speed * crossing_time

theorem faster_train_length :
  length_of_faster_train 72 36 20 = 200 :=
by
  sorry

end faster_train_length_l24_24368


namespace find_angle_A_find_sin_B_plus_pi_six_l24_24162

section Geometry

variables {a b c : ℝ} {A B C : ℝ} (m n : ℝ × ℝ)

-- Assumptions
-- Angle A in a triangle
-- m is parallel to n
-- b + c = √3 a
def assumptions :=
  (m = (1, 2 * Real.sin A)) ∧
  (n = (Real.sin A, 1 + Real.cos A)) ∧
  (m.1 * n.2 = m.2 * n.1) ∧
  (b + c = Real.sqrt 3 * a)

-- Goal (I): Prove angle A = π/3
theorem find_angle_A (h : assumptions m n):
  A = Real.pi / 3 :=
sorry

-- Additional assumption for part II: Using Law of Sines and trigonometry
def law_of_sines :=
  b = a * Real.sin B / Real.sin A ∧
  c = a * Real.sin C / Real.sin A

-- Goal (II): Prove sin(B + π/6) = √3/2
theorem find_sin_B_plus_pi_six (h1 : assumptions m n) (h2 : law_of_sines):
  Real.sin (B + Real.pi / 6) = Real.sqrt 3 / 2 :=
sorry

end Geometry

end find_angle_A_find_sin_B_plus_pi_six_l24_24162


namespace part1_part2_l24_24132

noncomputable def f (x : ℝ) := Real.log (1 + x)

theorem part1 (x : ℝ) (h : x ∈ (-1 : ℝ) <|> x ∈ (0 : ℝ) <|> x ∈ ℝ) : f x < x :=
sorry

theorem part2 (n k : ℕ) (h1 : 2 ≤ n) (h2 : 1 ≤ k) (h3 : k ≤ n - 1) :
  (∑ i in Finset.range k, 1 / n * Real.log (1 + i / n)) <
  (1 + (k + 1) / n) * Real.log (1 + (k + 1) / n) - (k + 1) / n ∧
  (1 + (k + 1) / n) * Real.log (1 + (k + 1) / n) - (k + 1) / n ≤ 2 * Real.log 2 - 1 :=
sorry

end part1_part2_l24_24132


namespace H_iterate_five_times_l24_24810

noncomputable def H : ℤ → ℤ
| 2 => -1
| -1 => 3
| 3 => 3
| _ => 0 -- assuming other values are 0 for completeness.

theorem H_iterate_five_times (H_step : ∀ x, H (H (H (H (H x)))) = 3) : 
  H (H (H (H (H 2)))) = 3 :=
by
  rw H
  rw H
  rw H
  rw H
  rw H
  exact H_step 2

end H_iterate_five_times_l24_24810


namespace arithmetic_progression_a_eq_c_minus_b_geometric_progression_a_eq_zero_l24_24577

variable {a b c : ℝ} {n q : ℝ}

-- Conditions for arithmetic progression
def is_arithmetic_progression (x : ℕ → ℝ) :=
  ∀ n, x n = b + (c - b) * (n - 1)

def arithmetic_condition (x : ℕ → ℝ) (a : ℝ) :=
  ∀ n, x (n + 2) = 3 * x (n + 1) - 2 * x n + a

-- Proving the value of 'a' for arithmetic progression
theorem arithmetic_progression_a_eq_c_minus_b (x : ℕ → ℝ) (h1 : is_arithmetic_progression x) (h2 : arithmetic_condition x a) : 
  a = c - b :=
sorry

-- Conditions for geometric progression
def is_geometric_progression (y : ℕ → ℝ) :=
  ∀ n, y n = b * q^(n - 1)

def geometric_condition (y : ℕ → ℝ) (a : ℝ) :=
  ∀ n, y (n + 2) = 3 * y (n + 1) - 2 * y n + a

-- Proving the value of 'a' for geometric progression
theorem geometric_progression_a_eq_zero (y : ℕ → ℝ) (h1 : is_geometric_progression y) (h2 : geometric_condition y a) (h3 : q = 2) (h4 : b > 0) : 
  a = 0 :=
sorry

end arithmetic_progression_a_eq_c_minus_b_geometric_progression_a_eq_zero_l24_24577


namespace g_neither_even_nor_odd_l24_24458

def g (x : ℝ) : ℝ := log (x + sqrt (2 + x^2))

theorem g_neither_even_nor_odd : (∀ x : ℝ, g (-x) ≠ g x ∧ g (-x) ≠ - g x) :=
sorry

end g_neither_even_nor_odd_l24_24458


namespace no_solution_when_n_is_neg_one_l24_24907

theorem no_solution_when_n_is_neg_one : 
    ∀ (x y z : ℝ), n = -1 → ¬ (∃ (x y z : ℝ), n * x + y = 2 ∧ n * y + z = 2 ∧ x + n^2 * z = 2) := 
by
  intros x y z h hsol
  cases hsol with x' hxyz
  cases hxyz with y' h'xyz
  cases h'xyz with z' hxhyhz
  have h1 : n * x' + y' = 2 := hxhyhz.left
  have h2 : n * y' + z' = 2 := hxhyhz.right.left
  have h3 : x' + n^2 * z' = 2 := hxhyhz.right.right
  subst h

  -- Step to show the equation collapse to contradiction, here we add the equations
  have h_sum : (-1) * x' + y' + (-1) * y' + z' + x' + (-1)^2 * z' = 6 := ...
  rw [one_mul (-1), one_mul 1] at h_sum
  simp at h_sum
  -- Continue showing contradiction and concluding no solution exists

  sorry -- Proof completion needed

end no_solution_when_n_is_neg_one_l24_24907


namespace smallest_digit_not_in_units_place_of_number_divisible_by_5_l24_24360

def smallest_non_units_digit : Nat :=
  let possible_units_digits := {x : Nat | x < 10 ∧ (x = 0 ∨ x = 5)}
  let remaining_digits := {x : Nat | x < 10 ∧ x ∉ possible_units_digits}
  Set.min' remaining_digits sorry -- We assume this set {x : Nat | x > 0 ∧ x ≠ 5} is nonempty and finite.

theorem smallest_digit_not_in_units_place_of_number_divisible_by_5 :
  smallest_non_units_digit = 1 := 
sorry

end smallest_digit_not_in_units_place_of_number_divisible_by_5_l24_24360


namespace set_intersection_complement_l24_24373

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
noncomputable def M : Set ℕ := {2, 3, 4, 5}
noncomputable def N : Set ℕ := {1, 4, 5, 7}

theorem set_intersection_complement :
  M ∩ (U \ N) = {2, 3} :=
by
  sorry

end set_intersection_complement_l24_24373


namespace imaginary_part_z2_l24_24115

-- Complex numbers z1 and z2
def z1 : ℂ := Complex.mk 2 (-3)

noncomputable def z2 : ℂ := (1 + 2 * Complex.i) / (2 - 3 * Complex.i)

-- The statement to prove
theorem imaginary_part_z2 : z1 * z2 = 1 + 2 * Complex.i → z2.im = 7 / 13 :=
by
  intro h
  sorry

end imaginary_part_z2_l24_24115


namespace extreme_values_when_a_is_minus_3_intersection_with_x_axis_when_a_ge_1_l24_24128

-- (1) Extreme values when a = -3
theorem extreme_values_when_a_is_minus_3 :
  let f := λ x : ℝ, x^3 - x^2 - 3 * x + 3 in
  has_extrema f (-1) 5 3 (-6) :=
by sorry

-- (2) Intersection with x-axis when a ≥ 1
theorem intersection_with_x_axis_when_a_ge_1 (a : ℝ) (h : a ≥ 1) :
  let f := λ x : ℝ, x^3 - x^2 + a * x - a in
  (∃ x, f x = 0) ∧ (∀ x₁ x₂, f x₁ = 0 → f x₂ = 0 → x₁ = x₂) :=
by sorry

end extreme_values_when_a_is_minus_3_intersection_with_x_axis_when_a_ge_1_l24_24128


namespace exams_passed_in_fourth_year_l24_24814

-- Define the variables representing the number of exams each year
variables (x1 x2 x3 x4 x5 : ℕ)

-- Define the conditions
def condition_1 : Prop := x1 + x2 + x3 + x4 + x5 = 31
def condition_2 : Prop := x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < x5
def condition_3 : Prop := x5 = 3 * x1

-- The theorem we aim to prove
theorem exams_passed_in_fourth_year : 
  condition_1 x1 x2 x3 x4 x5 → condition_2 x1 x2 x3 x4 x5 → condition_3 x1 x2 x3 x4 x5 → x4 = 8 :=
by 
  sorry

end exams_passed_in_fourth_year_l24_24814


namespace div_cond_l24_24228

-- Definitions for the conditions
variable (a b m n : ℕ)
variable (h1 : 1 < a)
variable (h2 : Nat.coprime a b)
variable (h3 : (a^m + b^m) ∣ (a^n + b^n))

-- The statement of the proof problem
theorem div_cond (a b m n : ℕ) (h1 : 1 < a) (h2 : Nat.coprime a b) (h3 : (a^m + b^m) ∣ (a^n + b^n)) : m ∣ n := by
  sorry

end div_cond_l24_24228


namespace marty_paint_combinations_l24_24660

def colors : List String := ["blue", "green", "yellow", "black", "white"]
def methods : List String := ["brush", "roller", "sponge", "spray"]

def valid_combinations (color : String) : List String :=
  if color = "white" then ["brush", "roller", "sponge"] else methods

theorem marty_paint_combinations :
  ∑ color in colors, (valid_combinations color).length = 19 := 
by
  sorry

end marty_paint_combinations_l24_24660


namespace find_ellipse_l24_24475

noncomputable def standard_equation_ellipse (x y : ℝ) : Prop :=
  (x^2 / 9 + y^2 / 3 = 1)
  ∨ (x^2 / 18 + y^2 / 9 = 1)
  ∨ (y^2 / (45 / 2) + x^2 / (45 / 4) = 1)

variables 
  (P1 P2 : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (a b : ℝ)

def passes_through_points (P1 P2 : ℝ × ℝ) : Prop :=
  ∀ equation : (ℝ → ℝ → Prop), 
    equation P1.1 P1.2 ∧ equation P2.1 P2.2

def focus_conditions (focus : ℝ × ℝ) : Prop :=
  -- Condition indicating focus, relationship with the minor axis etc., will be precisely defined here
  true -- Placeholder, needs correct mathematical condition

theorem find_ellipse : 
  passes_through_points P1 P2 
  → focus_conditions focus 
  → standard_equation_ellipse x y :=
sorry

end find_ellipse_l24_24475


namespace find_a5_l24_24189

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then 1 else 2 * sequence (n - 1) / (sequence (n - 1) + 2)

theorem find_a5 : sequence 5 = 1 / 3 :=
sorry

end find_a5_l24_24189


namespace count_palindrome_five_digit_div_by_5_l24_24242

-- Define what it means for a number to be palindromic.
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Define what it means for a number to be a five-digit number.
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

-- Define what it means for a number to be divisible by 5.
def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Define the set of five-digit palindromic numbers divisible by 5.
def palindrome_five_digit_div_by_5_numbers (n : ℕ) : Prop :=
  is_five_digit n ∧ is_palindrome n ∧ is_divisible_by_5 n

-- Prove that the quantity of such numbers is 100.
theorem count_palindrome_five_digit_div_by_5 : 
  (finset.filter 
    (λ n, palindrome_five_digit_div_by_5_numbers n)
    (finset.range 100000)
  ).card = 100 :=
begin
  sorry
end

end count_palindrome_five_digit_div_by_5_l24_24242


namespace minimum_most_popular_book_buyers_l24_24375

theorem minimum_most_popular_book_buyers
  (people : Finset ℕ) (books : ℕ → Finset ℕ)
  (h1 : people.card = 10)
  (h2 : ∀ p ∈ people, (books p).card = 3)
  (h3 : ∀ p1 p2 ∈ people, p1 ≠ p2 → (books p1 ∩ books p2).nonempty) :
  ∃ b : ℕ, 5 ≤ (people.filter (λ p, b ∈ books p)).card :=
sorry

end minimum_most_popular_book_buyers_l24_24375


namespace force_required_at_18_inch_l24_24308

-- Defining the conditions given
def inverse_var_force {L F k : ℝ} : Prop := F * L = k

theorem force_required_at_18_inch (k : ℝ) (h_k : k = 3600) :
  (∃ F : ℝ, inverse_var_force F 18 k ∧ F = 200) :=
by
  -- We skip the proof using sorry
  sorry

end force_required_at_18_inch_l24_24308


namespace minimum_value_k_determine_k_l24_24020

theorem minimum_value_k (k : ℝ) :
  ∀ x y : ℝ, 0 ≤ 9*x^2 - 12*k*x*y + (4*k^2 + 3)*y^2 - 6*x - 6*y + 9 :=
  by sorry

theorem determine_k : minimum_value_k (4/3) := by sorry

end minimum_value_k_determine_k_l24_24020


namespace Camp_Cedar_number_of_counselors_l24_24005

theorem Camp_Cedar_number_of_counselors (boys : ℕ) (girls : ℕ) (total_children : ℕ) (counselors : ℕ)
  (h_boys : boys = 40)
  (h_girls : girls = 3 * boys)
  (h_total_children : total_children = boys + girls)
  (h_counselors : counselors = total_children / 8) :
  counselors = 20 :=
by
  -- this is a statement, so we conclude with sorry to skip the proof.
  sorry

end Camp_Cedar_number_of_counselors_l24_24005


namespace find_a_l24_24236

open Set

-- Define set A
def A : Set ℝ := {-1, 1, 3}

-- Define set B in terms of a
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

-- State the theorem
theorem find_a (a : ℝ) (h : A ∩ B a = {3}) : a = 1 :=
sorry

end find_a_l24_24236


namespace point_M_is_outside_segment_DC_l24_24616

-- Define the setup for trapezoid ABCD with the given conditions
variables {A B C D M : Type}
variables (AD : ℝ) (AB : ℝ) (CD : ℝ) (is_trapezoid : Prop)
variables (is_perpendicular : CD ⊥ AD)
variables (angle_bisector_intersects : Prop)

-- Assume the given values
def is_valid_trapezoid (A B C D : Type) (AD AB CD : ℝ) (is_trapezoid : Prop) : Prop :=
  AD = 19 ∧ AB = 13 ∧ CD = 12 ∧ is_trapezoid ∧ is_perpendicular ∧ angle_bisector_intersects

-- Formalize the question
theorem point_M_is_outside_segment_DC (A B C D M : Type)
  (AD AB CD : ℝ) (is_trapezoid : Prop) 
  (is_perpendicular : CD ⊥ AD)
  (angle_bisector_intersects : Prop)
  (h : is_valid_trapezoid A B C D AD AB CD is_trapezoid) :
  ¬ (M ∈ CD) :=
sorry

end point_M_is_outside_segment_DC_l24_24616


namespace find_A_l24_24699

theorem find_A :
  ∃ A B C : ℝ, 
  (1 : ℝ) / (x^3 - 7 * x^2 + 11 * x + 15) = 
  A / (x - 5) + B / (x + 3) + C / ((x + 3)^2) → 
  A = 1 / 64 := 
by 
  sorry

end find_A_l24_24699


namespace exists_unequal_sides_not_both_zero_some_angle_not_acute_none_is_zero_l24_24769

-- Condition for problem 1
def square (sides : ℝ) (equal : Prop) : Prop :=
  ∃ (a b c d : ℝ), a = b ∧ b = c ∧ c = d ∧ (equal ↔ (a ≠ b ∨ b ≠ c ∨ c ≠ d ∨ d ≠ a))

-- Problem 1
theorem exists_unequal_sides (equal : Prop) : 
  ¬(∀ (a b c d : ℝ), square sides equal → a = b ∧ b = c ∧ c = d ∧ d = a) :=
  sorry

-- Condition for problem 2
def sum_of_squares_zero (x y : ℝ) : Prop := x^2 + y^2 = 0

-- Problem 2
theorem not_both_zero (x y : ℝ) : 
  sum_of_squares_zero x y → ¬ (x = 0 ∧ y = 0) :=
  sorry

-- Condition for problem 3
def acute_triangle (A B C : Triangle) : Prop := 
  A.is_acute ∧ B.is_acute ∧ C.is_acute

-- Problem 3
theorem some_angle_not_acute (A B C : Triangle) : 
  acute_triangle A B C → (¬ (A.is_acute ∧ B.is_acute ∧ C.is_acute)) :=
  sorry

-- Condition for problem 4
def product_zero (a b c : ℝ) : Prop := a * b * c = 0

-- Problem 4
theorem none_is_zero (a b c : ℝ) : 
  product_zero a b c → ¬(a = 0 ∨ b = 0 ∨ c = 0) :=
  sorry

end exists_unequal_sides_not_both_zero_some_angle_not_acute_none_is_zero_l24_24769


namespace does_not_round_to_72_56_l24_24767

-- Definitions for the numbers in question
def numA := 72.558
def numB := 72.563
def numC := 72.55999
def numD := 72.564
def numE := 72.555

-- Function to round a number to the nearest hundredth
def round_nearest_hundredth (x : Float) : Float :=
  (Float.round (x * 100) / 100 : Float)

-- Lean statement for the equivalent proof problem
theorem does_not_round_to_72_56 :
  round_nearest_hundredth numA = 72.56 ∧
  round_nearest_hundredth numB = 72.56 ∧
  round_nearest_hundredth numC = 72.56 ∧
  round_nearest_hundredth numD = 72.56 ∧
  round_nearest_hundredth numE ≠ 72.56 :=
by
  sorry

end does_not_round_to_72_56_l24_24767


namespace range_of_t_l24_24211

noncomputable def M : set ℝ := {x | -2 < x ∧ x < 5}
noncomputable def N (t : ℝ) : set ℝ := {x | 2 - t < x ∧ x < 2 * t + 1}

theorem range_of_t (t : ℝ) : (M ∩ N t = N t) ↔ t ≤ 2 :=
sorry

end range_of_t_l24_24211


namespace tangent_product_l24_24437

theorem tangent_product (n θ : ℝ) 
  (h1 : ∀ n θ, tan (n * θ) = (sin (n * θ)) / (cos (n * θ)))
  (h2 : tan 8 θ = (8 * tan θ - 56 * (tan θ) ^ 3 + 56 * (tan θ) ^ 5 - 8 * (tan θ) ^ 7) / 
                  (1 - 28 * (tan θ) ^ 2 + 70 * (tan θ) ^ 4 - 28 * (tan θ) ^ 6))
  (h3 : tan (8 * (π / 8)) = 0)
  (h4 : tan (8 * (3 * π / 8)) = 0)
  (h5 : tan (8 * (5 * π / 8)) = 0) :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = 2 * sqrt 2 :=
sorry

end tangent_product_l24_24437


namespace min_inverse_sum_l24_24927

-- Let {a_n} be a geometric sequence with positive terms
-- Assume a_7 = a_6 + 2 * a_5 and sqrt(a_m * a_n) = 4 * a_1
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem min_inverse_sum (a : ℕ → ℝ) (q : ℝ) (m n : ℕ) :
  geometric_sequence a q →
  0 < a 1 →
  a 6 * q = a 5 * q + 2 * (a 4 * q) →  -- Condition a_7 = a_6 + 2 * a_5
  sqrt (a m * a n) = 4 * a 1 →         -- Condition sqrt(a_m * a_n) = 4 * a_1
  1 ≤ m ∧ 1 ≤ n → 
  m + n = 6 →                         -- Because a_7 represents a_6 * q and a_5 represents a_4 * q
  min_inv_sum = min (1/m + 1/n) = 2/3 :=   -- Minimum value of (1/m + 1/n) = 2/3
begin
  sorry
end

end min_inverse_sum_l24_24927


namespace probability_white_model_a_truck_l24_24410

structure VehicleCounts :=
  (trucks : ℕ)
  (cars : ℕ)
  (vans : ℕ)

structure TruckColorPercentages :=
  (red : ℕ)
  (black : ℕ)
  (white : ℕ)

structure ModelAColorPercentages :=
  (red : ℕ)
  (black : ℕ)
  (white : ℕ)

def total_vehicles (vc : VehicleCounts) : ℕ :=
  vc.trucks + vc.cars + vc.vans

def percentage_of_trucks (percentage : ℕ) (total_trucks : ℕ) : ℕ :=
  (percentage * total_trucks) / 100

theorem probability_white_model_a_truck
  (vc : VehicleCounts)
  (tcp : TruckColorPercentages)
  (model_a : ModelAColorPercentages) :
  vc = { trucks := 50, cars := 40, vans := 30 } →
  tcp = { red := 40, black := 20, white := 10 } →
  model_a = { red := 30, black := 25, white := 20 } →
  let white_trucks := percentage_of_trucks tcp.white vc.trucks in
  let white_model_a_trucks := percentage_of_trucks model_a.white white_trucks in
  (white_model_a_trucks * 100 / total_vehicles vc = 1) :=
begin
  intros h1 h2 h3,
  let vc_total := total_vehicles vc,
  let white_model_a_trucks := percentage_of_trucks model_a.white white_trucks,
  have white_trucks_calc : white_trucks = 5 := rfl,
  have white_model_a_trucks_calc : white_model_a_trucks = 1 := rfl,
  have vc_total_calc : vc_total = 120 := rfl,
  exact (1 * 100 / 120 = 0.83333333).floor_eq_zero_add_one,
end

end probability_white_model_a_truck_l24_24410


namespace binary_to_base5_conversion_l24_24880

theorem binary_to_base5_conversion : ∀ (b : ℕ), b = 1101 → (13 : ℕ) % 5 = 3 ∧ (13 / 5) % 5 = 2 → b = 1101 → (1101 : ℕ) = 13 → 13 = 23 :=
by
  sorry

end binary_to_base5_conversion_l24_24880


namespace triangle_side_correct_l24_24583

-- Define the conditions
def a : ℝ := 9
def b : ℝ := 2 * Real.sqrt 3
def C : ℝ := Real.pi * 5 / 6  -- 150 degrees in radians

-- Define the cosine of angle C
noncomputable def cos_C : ℝ := Real.cos C

-- Given the cosine rule c² = a² + b² - 2ab cos(C), find c
noncomputable def c : ℝ := Real.sqrt (a^2 + b^2 - 2 * a * b * cos_C)

theorem triangle_side_correct :
  c = 7 * Real.sqrt 3 := by
  sorry

end triangle_side_correct_l24_24583


namespace problem_irational_count_l24_24402

noncomputable def isRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

noncomputable def isIrrational (x : ℝ) : Prop := ¬isRational x

def countIrrationals (l : List ℝ) : ℕ :=
  l.countp isIrrational

theorem problem_irational_count : countIrrationals [-3, 22 / 7, 3.14, -3 * Real.pi, 3.030030003] = 2 := by
  sorry

end problem_irational_count_l24_24402


namespace perpendicular_lines_slope_l24_24545

theorem perpendicular_lines_slope {m : ℝ} : 
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0) → 
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0 → (m * (-1/2)) = -1) → 
  m = 2 :=
by 
  intros h_perpendicular h_slope
  sorry

end perpendicular_lines_slope_l24_24545


namespace parabolas_intersect_prob_l24_24452

theorem parabolas_intersect_prob :
  let outcomes := {1, 2, 3, 4, 5, 6, 7, 8}
  let P := outcomes.card
  let occurrences := P * P * P * P
  let eq_prob := occurrences - (P * (P-1))
  let required_prob := eq_prob.toRational
  let expected_prob := 63 / 64
  required_prob = expected_prob :=
by 
  sorry

end parabolas_intersect_prob_l24_24452


namespace rocco_piles_of_quarters_proof_l24_24289

-- Define the value of a pile of different types of coins
def pile_value (coin_value : ℕ) (num_coins_in_pile : ℕ) : ℕ :=
  coin_value * num_coins_in_pile

-- Define the number of piles for different coins
def num_piles_of_dimes : ℕ := 6
def num_piles_of_nickels : ℕ := 9
def num_piles_of_pennies : ℕ := 5
def num_coins_in_pile : ℕ := 10

-- Define the total value of each type of coin
def value_of_a_dime : ℕ := 10  -- in cents
def value_of_a_nickel : ℕ := 5  -- in cents
def value_of_a_penny : ℕ := 1  -- in cents
def value_of_a_quarter : ℕ := 25  -- in cents

-- Define the total money Rocco has in cents
def total_money : ℕ := 2100  -- since $21 = 2100 cents

-- Calculate the value of all piles of each type of coin
def total_dimes_value : ℕ := num_piles_of_dimes * (pile_value value_of_a_dime num_coins_in_pile)
def total_nickels_value : ℕ := num_piles_of_nickels * (pile_value value_of_a_nickel num_coins_in_pile)
def total_pennies_value : ℕ := num_piles_of_pennies * (pile_value value_of_a_penny num_coins_in_pile)

-- Calculate the value of the quarters
def value_of_quarters : ℕ := total_money - (total_dimes_value + total_nickels_value + total_pennies_value)
def num_piles_of_quarters : ℕ := value_of_quarters / 250 -- since each pile of quarters is worth 250 cents

-- Theorem to prove
theorem rocco_piles_of_quarters_proof : num_piles_of_quarters = 4 := by
  sorry

end rocco_piles_of_quarters_proof_l24_24289


namespace students_interested_both_l24_24027

/-- total students surveyed -/
def U : ℕ := 50

/-- students who liked watching table tennis matches -/
def A : ℕ := 35

/-- students who liked watching badminton matches -/
def B : ℕ := 30

/-- students not interested in either -/
def nU_not_interest : ℕ := 5

theorem students_interested_both : (A + B - (U - nU_not_interest)) = 20 :=
by sorry

end students_interested_both_l24_24027


namespace find_f_31_over_2_l24_24455

-- Define the necessary conditions as given in the problem
def f (x : ℝ) : ℝ := -- function is not completely defined; we are given pieces as conditions
  if 0 ≤ x ∧ x ≤ 1 then x * (3 - 2 * x)  -- piece defined in [0, 1]
  else if x + 1 = (-(x + 1)) then sorry -- even function condition f(x+1) = f(-(x+1))
  else if x = -x then sorry -- odd function condition f(x) = -f(x)
  else sorry -- general case we cannot completely define without more information

-- Ensuring that f(x) satisfies given conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x  -- f is an odd function
axiom f_even_shift : ∀ x : ℝ, f (x + 1) = f (-x + 1)  -- f(x + 1) is even
axiom f_in_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x * (3 - 2 * x)  -- f(x) for x ∈ [0, 1]

-- The theorem to prove based on the given problem
theorem find_f_31_over_2 : f (31 / 2) = -1 :=
begin
  sorry -- The proof is not provided
end

end find_f_31_over_2_l24_24455


namespace num_integer_four_tuples_count_l24_24054

def f (x y z u : ℤ) : ℚ :=
  (x - y) / (x + y) + (y - z) / (y + z) + (z - u) / (z + u) + (u - x) / (u + x)

theorem num_integer_four_tuples_count :
  (finset.univ.product
    (finset.univ.product
      (finset.univ.product
        (finset.univ.filter (λ x : ℤ, 1 ≤ x ∧ x ≤ 10))
        (finset.univ.filter (λ y : ℤ, 1 ≤ y ∧ y ≤ 10)))
      (finset.univ.filter (λ z : ℤ, 1 ≤ z ∧ z ≤ 10)))
    (finset.univ.filter (λ u : ℤ, 1 ≤ u ∧ u ≤ 10)))
  .card (λ (w : ℤ × (ℤ × (ℤ × ℤ))),
    let ⟨x, ⟨y, ⟨z, u⟩⟩⟩ := w in f x y z u > 0) = 3924 := sorry

end num_integer_four_tuples_count_l24_24054


namespace probability_correct_dial_l24_24265

theorem probability_correct_dial:
  (let prefixes := {296, 299, 298} in 
   let remaining_digits := {0, 1, 6, 7, 9} in 
   let total_possibilities := 3 * (Nat.factorial 5) in
   let p : ℚ := 1 / total_possibilities in
   p = 1 / 360) :=
  
by
  sorry

end probability_correct_dial_l24_24265


namespace distance_Bella_Galya_l24_24192

theorem distance_Bella_Galya 
    (AB BV VG GD : ℝ)
    (DB : AB + 3 * BV + 2 * VG + GD = 700)
    (DV : AB + 2 * BV + 2 * VG + GD = 600)
    (DG : AB + 2 * BV + 3 * VG + GD = 650) :
    BV + VG = 150 := 
by
  -- Proof goes here
  sorry

end distance_Bella_Galya_l24_24192


namespace complex_solutions_count_l24_24055

noncomputable def has_exactly_two_complex_solutions (f : ℂ → ℂ) : Prop :=
  ∃ z1 z2 : ℂ, z1 ≠ z2 ∧ f z1 = 0 ∧ f z2 = 0 ∧ ∀ z : ℂ, f z = 0 → (z = z1 ∨ z = z2)

theorem complex_solutions_count : has_exactly_two_complex_solutions (λ z, (z^4 - 1) / (z^3 + 2z^2 - z - 2)) :=
sorry

end complex_solutions_count_l24_24055


namespace Camp_Cedar_number_of_counselors_l24_24004

theorem Camp_Cedar_number_of_counselors (boys : ℕ) (girls : ℕ) (total_children : ℕ) (counselors : ℕ)
  (h_boys : boys = 40)
  (h_girls : girls = 3 * boys)
  (h_total_children : total_children = boys + girls)
  (h_counselors : counselors = total_children / 8) :
  counselors = 20 :=
by
  -- this is a statement, so we conclude with sorry to skip the proof.
  sorry

end Camp_Cedar_number_of_counselors_l24_24004


namespace reading_homework_is_4_l24_24286

-- Defining the conditions.
variables (R : ℕ)  -- Number of pages of reading homework
variables (M : ℕ)  -- Number of pages of math homework

-- Rachel has 7 pages of math homework.
def math_homework_equals_7 : Prop := M = 7

-- Rachel has 3 more pages of math homework than reading homework.
def math_minus_reads_is_3 : Prop := M = R + 3

-- Prove the number of pages of reading homework is 4.
theorem reading_homework_is_4 (M R : ℕ) 
  (h1 : math_homework_equals_7 M) -- M = 7
  (h2 : math_minus_reads_is_3 M R) -- M = R + 3
  : R = 4 :=
sorry

end reading_homework_is_4_l24_24286


namespace longest_shortest_chord_product_l24_24125

-- Definition of a circle with center M and radius r
def circle_eq (x y : ℝ) (h : (x - 1)^2 + (y - 1)^2 = 9) := true

-- Definition stating that point P(2,2) lies inside this circle
def point_P_in_circle := ∃ h : (2 - 1)^2 + (2 - 1)^2 < 9, true

-- Definition of the longest chord AC and it being the diameter
def longest_chord_ac (P : ℝ × ℝ) := ∃ A C : ℝ × ℝ, true

-- Definition of the shortest chord BD which is orthogonal to AC at P
def shortest_chord_bd (P : ℝ × ℝ) := ∃ B D : ℝ × ℝ, true

-- The product of lengths of AC and BD
theorem longest_shortest_chord_product : 
  (circle_eq 1 1)
  → point_P_in_circle
  → longest_chord_ac (2,2)
  → shortest_chord_bd (2,2)
  → (6 : ℝ) * (2 * real.sqrt 7) = 12 * real.sqrt 7 :=
by
  sorry

end longest_shortest_chord_product_l24_24125


namespace number_representation_fewer_sevens_exists_l24_24783

def representable_using_fewer_sevens (n : ℕ) : Prop :=
  ∃ (N : ℕ), let num := 7 * (10 ^ n - 1) / 9 in 
  N < n ∧ num = N

theorem number_representation_fewer_sevens_exists : ∃ (n : ℕ), representable_using_fewer_sevens n :=
sorry

end number_representation_fewer_sevens_exists_l24_24783


namespace rectangle_ratio_4_to_1_l24_24378

theorem rectangle_ratio_4_to_1 (x y : ℕ) : 
  let A := x * y in
  let x' := 2 * x in
  let y' := y / 2 in
  let x'' := x' + 1 in
  let y'' := y' - 4 in
  let x_final_case_a := 2 * x in
  let y_final_case_a := 8 * x in
  let ratio_final_case_a := y_final_case_a / x_final_case_a in
  let x''_case_b := x' - 4 in
  let y''_case_b := y' + 1 in
  let x_final_case_b := 2 * y in
  let y_final_case_b := y / 2 in
  let ratio_final_case_b := y_final_case_b / x_final_case_b in
  ratio_final_case_a = 4 ∧ ratio_final_case_b = 4 := 
sorry

end rectangle_ratio_4_to_1_l24_24378


namespace part1_part2_l24_24918

variable {α : ℝ}

def tan (x : ℝ) : ℝ := sin x / cos x

theorem part1 (h : tan α = 2) : (2 * sin α - 3 * cos α) / (4 * sin α - 9 * cos α) = -1 := 
  sorry

theorem part2 (h : tan α = 2) : 4 * (sin α)^2 - 3 * sin α * cos α - 5 * (cos α)^2 = 1 :=
  sorry

end part1_part2_l24_24918


namespace part_I_part_II_l24_24527

-- Define the function f
def f (x : ℝ) : ℝ := x / (2 * x + 1)

-- Define the sequence a_n recursively
def a : ℕ → ℝ
| 0     := 1  -- Note: In lean, we generally start sequences from 0.
| (n+1) := f (a n)

-- Part (I): Prove that the sequence {1/a_n} is an arithmetic sequence
theorem part_I : (∃ (d : ℝ), ∀ n, 1 / (a (n + 1)) - 1 / (a n) = d) :=
sorry

-- Define S_n as described
def S (n : ℕ) : ℝ :=
∑ i in Finset.range n, a i * a (i + 1)

-- Part (II): Prove that 2S_n < 1
theorem part_II (n : ℕ) : 2 * (S n) < 1 :=
sorry

end part_I_part_II_l24_24527


namespace number_of_intersections_l24_24013

-- Definitions for the problem conditions
def focus : (ℝ × ℝ) := (0, 0)

def directrix (a c : ℝ) : (ℝ → ℝ) := (λ x, a * x + c)

def parabola (a c : ℝ) : set (ℝ × ℝ) := 
  { p | let (x, y) := p in y^2 = 4 * (x + a * y + c)}

-- Set of allowable values for a and c 
def a_values : set ℝ := {-3, -2, -1, 0, 1, 2, 3}
def c_values : set ℝ := {-4, -3, -2, -1, 1, 2, 3, 4}

-- Total number of parabolas, focusing on the given a and c values
def parabolas_set : set (set (ℝ × ℝ)) := 
  {parabola a c | a ∈ a_values ∧ c ∈ c_values }

-- Intersection of two parabolas
def intersection_points (p1 p2 : set (ℝ × ℝ)) : set (ℝ × ℝ) := p1 ∩ p2

-- No three parabolas share a common point
axiom no_three_parabolas_common_point (p1 p2 p3 : set (ℝ × ℝ)) : ¬ (∃ x, x ∈ p1 ∧ x ∈ p2 ∧ x ∈ p3)

theorem number_of_intersections : 
  ∑ i j in (parabolas_set.product parabolas_set), i ≠ j 
  → set.size (intersection_points i j) = 2 → 1022 := 
sorry

end number_of_intersections_l24_24013


namespace unit_sales_price_decrease_ratio_l24_24772

noncomputable def calculate_ratio (P U : ℝ) (price_decrease_percent : ℝ) (new_price U_new : ℝ) :=
  (new_price = P * (1 - price_decrease_percent / 100)) ∧
  (P * U = new_price * U_new) →
  (U_new = U / (1 - price_decrease_percent / 100)) →
  let percent_increase = ((U_new - U) / U) * 100
  let percent_decrease = price_decrease_percent
  (percent_increase / percent_decrease) = 1.111

theorem unit_sales_price_decrease_ratio (P U : ℝ) (price_decrease_percent : ℝ) (U_new : ℝ) :
  price_decrease_percent = 10 →
  (P * U = (P * (1 - price_decrease_percent / 100)) * U_new) →
  (P * (1 - price_decrease_percent / 100) = 0.90 * P) →
  0 < P → 0 < U →
  calculate_ratio P U price_decrease_percent (0.90 * P) U_new :=
begin
  intros hd hr hp hp_gt_zero hu_gt_zero,
  -- proof goes here
  sorry
end

end unit_sales_price_decrease_ratio_l24_24772


namespace a_eq_zero_l24_24118

noncomputable def f (x a : ℝ) := x^2 - abs (x + a)

theorem a_eq_zero (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 0 :=
by
  sorry

end a_eq_zero_l24_24118


namespace arithmetic_sequence_proof_l24_24934

open Nat

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  (∀ n, a (n + 1) = a n + d) ∧ (a 1 = 1)

def b_sequence (a : ℕ → ℤ) (b : ℕ → ℚ) : Prop :=
  ∀ n, b n = 1 / (a n * a (n + 1))

def S_n_sum (b : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n, S n = (∑ i in range n, b (i + 1))

def arithmetic_sequence_conditions (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = a n + d ∧ a 1 + 3 < a 3 ∧ a 2 + 5 > a 4 ∧ d ∈ Int

def general_term_result (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = 2 * n - 1

def geometric_mean_condition (S : ℕ → ℚ) : Prop :=
  S 2 = Real.sqrt (S 1 * S m)

def final_value_m (m : ℕ) : Prop :=
  m = 12

theorem arithmetic_sequence_proof (a : ℕ → ℤ) (d : ℤ) (b : ℕ → ℚ) (S : ℕ → ℚ) (m : ℕ)
  (h1 : arithmetic_sequence_conditions a d)
  (h2 : general_term_result a)
  (h3 : b_sequence a b)
  (h4 : S_n_sum b S)
  (h5 : geometric_mean_condition S) :
  final_value_m m :=
sorry

end arithmetic_sequence_proof_l24_24934


namespace shaded_region_area_l24_24185

noncomputable def area_of_shaded_region (R: ℝ) (r: ℝ) : ℝ :=
  let larger_circle_area := π * R^2
  let smaller_circle_area := π * r^2
  let two_smaller_circles_area := 2 * smaller_circle_area
  larger_circle_area - two_smaller_circles_area

theorem shaded_region_area (R: ℝ) (h1: R = 9) (h2: r = R / 4) :
  area_of_shaded_region R r = 70.875 * π :=
by {
  -- proof would go here
  sorry
}

end shaded_region_area_l24_24185


namespace polygons_with_largest_area_l24_24908

def unit_square_area := 1
def right_triangle_area := 0.5

def polygon_P_area := 3 * unit_square_area + 2 * right_triangle_area
def polygon_Q_area := 4 * unit_square_area + 1 * right_triangle_area
def polygon_R_area := 6 * unit_square_area
def polygon_S_area := 2 * unit_square_area + 4 * right_triangle_area
def polygon_T_area := 5 * unit_square_area + 2 * right_triangle_area

theorem polygons_with_largest_area :
  (polygon_R_area = 6 ∧ polygon_T_area = 6) ∧
  ∀ (a b c d e : ℝ), a = polygon_P_area → b = polygon_Q_area → c = polygon_R_area →
                     d = polygon_S_area → e = polygon_T_area →
                     (c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e) ∧
                     (e ≥ a ∧ e ≥ b ∧ e ≥ d ∧ e ≥ c) := by
  sorry

end polygons_with_largest_area_l24_24908


namespace rocky_first_round_knockouts_l24_24684

theorem rocky_first_round_knockouts
  (total_fights : ℕ)
  (knockout_percentage : ℝ)
  (first_round_knockout_percentage : ℝ)
  (h1 : total_fights = 190)
  (h2 : knockout_percentage = 0.50)
  (h3 : first_round_knockout_percentage = 0.20) :
  (total_fights * knockout_percentage * first_round_knockout_percentage = 19) := 
by
  sorry

end rocky_first_round_knockouts_l24_24684


namespace problem1_problem2_problem3_problem4_l24_24689

-- Define the number of different arrangements for the first condition
def arrangements1 : ℕ := 5040

-- Define the number of different arrangements for the second condition
def arrangements2 : ℕ := 1440

-- Define the number of different arrangements for the third condition
def arrangements3 : ℕ := 720

-- Define the number of different arrangements for the fourth condition
def arrangements4 : ℕ := 1440

-- Prove the first statement
theorem problem1 (students : Fin 7 → Prop) (front_row : Fin 3 → Prop) (back_row : Fin 4 → Prop) :
  (is_permutation students front_row back_row) = arrangements1 := sorry

-- Prove the second statement
theorem problem2 (students : Fin 7 → Prop) (front_row : Fin 3 → Prop) (back_row : Fin 4 → Prop)
  (A_in_front : students A → front_row A) (B_in_back : students B → back_row B) :
  (is_permutation students front_row back_row A_in_front B_in_back) = arrangements2 := sorry

-- Prove the third statement
theorem problem3 (students : Fin 7 → Prop) (row : Fin 7 → Prop)
  (ABC_together : ∀ (A B C : students), is_next_to A B C row) :
  (is_permutation students row ABC_together) = arrangements3 := sorry

-- Prove the fourth statement
theorem problem4 (students : Fin 7 → Prop) (boys : Fin 4 → Prop) (girls : Fin 3 → Prop)
  (girls_not_adjacent : ∀ (G1 G2 : girls), ¬is_adjacent G1 G2) :
  (is_permutation students boys girls girls_not_adjacent) = arrangements4 := sorry

end problem1_problem2_problem3_problem4_l24_24689


namespace negation_of_proposition_l24_24316

theorem negation_of_proposition :
  (¬ (∃ x₀ : ℝ, x₀ > 2 ∧ x₀^3 - 2 * x₀^2 < 0)) ↔ (∀ x : ℝ, x > 2 → x^3 - 2 * x^2 ≥ 0) := by
  sorry

end negation_of_proposition_l24_24316


namespace quadratic_inequality_solution_set_empty_l24_24158

theorem quadratic_inequality_solution_set_empty
  (m : ℝ)
  (h : ∀ x : ℝ, mx^2 - mx - 1 < 0) :
  -4 < m ∧ m < 0 :=
sorry

end quadratic_inequality_solution_set_empty_l24_24158


namespace product_ineq_l24_24139

def a : ℕ → ℝ 
| 0       := 0
| (n + 1) := a n + 1 / (n + 1)!

def b (n : ℕ) : ℝ :=
(n + 1)! * a n

theorem product_ineq : 
  (∏ i in finset.range 2023, (1 + 1 / b i)) < 7 / 4 := 
sorry

end product_ineq_l24_24139


namespace tan_alpha_eq_neg2_complex_expression_eq_neg5_l24_24917

variable (α : ℝ)

-- Given conditions
def sin_alpha : sin α = -((2 * real.sqrt 5) / 5) := sorry
def tan_alpha_neg : tan α < 0 := sorry

-- Proof problem for the first question
theorem tan_alpha_eq_neg2 (h₁ : sin α = -((2 * real.sqrt 5) / 5)) (h₂ : tan α < 0) : tan α = -2 := sorry

-- Proof problem for the second question
theorem complex_expression_eq_neg5 (h₁ : sin α = -((2 * real.sqrt 5) / 5)) (h₂ : tan α < 0) : 
  (2 * sin (α + real.pi) + cos (2 * real.pi - α)) / 
  (cos (α - (real.pi / 2)) - sin (3 * real.pi / 2 + α)) = -5 := sorry

end tan_alpha_eq_neg2_complex_expression_eq_neg5_l24_24917


namespace find_a_b_find_A_l24_24217

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 2 * (Real.log x / Real.log 2) ^ 2 + 2 * a * (Real.log (1 / x) / Real.log 2) + b

theorem find_a_b : (∀ x : ℝ, 0 < x → f x a b = 2 * (Real.log x / Real.log 2)^2 + 2 * a * (Real.log (1 / x) / Real.log 2) + b) 
                     → f (1/2) a b = -8 
                     ∧ ∀ x : ℝ, 0 < x → x ≠ 1/2 → f x a b ≥ f (1 / 2) a b
                     → a = -2 ∧ b = -6 := 
sorry

theorem find_A (a b : ℝ) (h₁ : a = -2) (h₂ : b = -6) : 
  { x : ℝ | 0 < x ∧ f x a b > 0 } = {x | 0 < x ∧ (x < 1/8 ∨ x > 2)} :=
sorry

end find_a_b_find_A_l24_24217


namespace solution_set_l24_24935

-- Define the conditions
variable {f : ℝ → ℝ}

-- Condition 1: f is an even function
def even_function := ∀ x, f(-x) = f(x)

-- Condition 2: f is monotonically increasing on [0, +∞)
def monotonically_increasing := ∀ x y, 0 ≤ x → x ≤ y → f(x) ≤ f(y)

-- Condition 3: f(-3) = 0
def value_at_neg3 := f(-3) = 0

-- The theorem stating the solution set of x * f(x - 2) > 0
theorem solution_set (h1 : even_function f) (h2 : monotonically_increasing f) (h3 : value_at_neg3 f) :
  { x : ℝ | x * f(x - 2) > 0 } = { x : ℝ | -1 < x ∧ x < 0 ∨ x > 5 } :=
sorry

end solution_set_l24_24935


namespace circle_center_sum_is_one_l24_24064

def circle_center_sum (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 + 4 * x - 6 * y = 3) → ((h = -2) ∧ (k = 3))

theorem circle_center_sum_is_one :
  ∀ h k : ℝ, circle_center_sum h k → h + k = 1 :=
by
  intros h k hc
  sorry

end circle_center_sum_is_one_l24_24064


namespace red_to_blue_l24_24292

def is_red (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 2020

def is_blue (n : ℕ) : Prop :=
  ¬ is_red n ∧ ∃ m : ℕ, n = m ^ 2019

theorem red_to_blue (n : ℕ) (hn : n > 10^100000000) (hnred : is_red n) 
    (hn1red : is_red (n+1)) :
    ∃ (k : ℕ), 1 ≤ k ∧ k ≤ 2019 ∧ is_blue (n + k) :=
sorry

end red_to_blue_l24_24292


namespace avg_growth_rate_leq_half_sum_l24_24319

theorem avg_growth_rate_leq_half_sum (m n p : ℝ) (hm : 0 ≤ m) (hn : 0 ≤ n)
    (hp : (1 + p / 100)^2 = (1 + m / 100) * (1 + n / 100)) : 
    p ≤ (m + n) / 2 :=
by
  sorry

end avg_growth_rate_leq_half_sum_l24_24319


namespace family_of_four_children_has_at_least_one_boy_and_one_girl_l24_24830

noncomputable section

def probability_at_least_one_boy_one_girl : ℚ :=
  1 - (1 / 16 + 1 / 16)

theorem family_of_four_children_has_at_least_one_boy_and_one_girl :
  probability_at_least_one_boy_one_girl = 7 / 8 := by
  sorry

end family_of_four_children_has_at_least_one_boy_and_one_girl_l24_24830


namespace tan_product_l24_24428

theorem tan_product :
  (Real.tan (Real.pi / 8)) * (Real.tan (3 * Real.pi / 8)) * (Real.tan (5 * Real.pi / 8)) = 1 :=
sorry

end tan_product_l24_24428


namespace num_candidates_above_630_l24_24722

noncomputable def normal_distribution_candidates : Prop :=
  let μ := 530
  let σ := 50
  let total_candidates := 1000
  let probability_above_630 := (1 - 0.954) / 2  -- Probability of scoring above 630
  let expected_candidates_above_630 := total_candidates * probability_above_630
  expected_candidates_above_630 = 23

theorem num_candidates_above_630 : normal_distribution_candidates := by
  sorry

end num_candidates_above_630_l24_24722


namespace log_eq_to_x_inv_half_l24_24110

theorem log_eq_to_x_inv_half (x : ℝ) (h : log 7 (log 3 (log 2 x)) = 0) : x^(-1/2) = sqrt 2 / 4 :=
by sorry

end log_eq_to_x_inv_half_l24_24110


namespace sum_floor_values_arith_seq_l24_24876

def floor_sum_arith_seq (a d : ℝ) (n : ℕ) : ℕ :=
  (List.range n).sum (λ k => ⌊a + k * d⌋₊)

theorem sum_floor_values_arith_seq :
  floor_sum_arith_seq 0.5 0.6 167 = 8350 :=
by sorry

end sum_floor_values_arith_seq_l24_24876


namespace sum_placed_on_SI_l24_24778

theorem sum_placed_on_SI :
  let P₁ := 4000
  let r₁ := 0.10
  let t₁ := 2
  let CI := P₁ * ((1 + r₁)^t₁ - 1)

  let SI := (1 / 2 * CI : ℝ)
  let r₂ := 0.08
  let t₂ := 3
  let P₂ := SI / (r₂ * t₂)

  P₂ = 1750 :=
by
  sorry

end sum_placed_on_SI_l24_24778


namespace discount_rate_for_1000_min_price_for_1_3_discount_l24_24379

def discounted_price (original_price : ℕ) : ℕ := 
  original_price * 80 / 100

def voucher_amount (discounted_price : ℕ) : ℕ :=
  if discounted_price < 400 then 30
  else if discounted_price < 500 then 60
  else if discounted_price < 700 then 100
  else if discounted_price < 900 then 130
  else 0 -- Can extend the rule as needed

def discount_rate (original_price : ℕ) : ℚ := 
  let total_discount := original_price * 20 / 100 + voucher_amount (discounted_price original_price)
  (total_discount : ℚ) / (original_price : ℚ)

theorem discount_rate_for_1000 : 
  discount_rate 1000 = 0.33 := 
by
  sorry

theorem min_price_for_1_3_discount :
  ∀ (x : ℕ), 500 ≤ x ∧ x ≤ 800 → 0.33 ≤ discount_rate x ↔ (625 ≤ x ∧ x ≤ 750) :=
by
  sorry

end discount_rate_for_1000_min_price_for_1_3_discount_l24_24379


namespace exists_five_digit_palindromic_divisible_by_5_count_five_digit_palindromic_numbers_divisible_by_5_l24_24253

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem exists_five_digit_palindromic_divisible_by_5 :
  ∃ n, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n := by
  -- Proof is omitted
  sorry

theorem count_five_digit_palindromic_numbers_divisible_by_5 :
  (finset.filter (λ n, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n) (finset.range 100000)).card = 100 := by
  -- Proof is omitted
  sorry

end exists_five_digit_palindromic_divisible_by_5_count_five_digit_palindromic_numbers_divisible_by_5_l24_24253


namespace rational_quotient_of_arith_geo_subseq_l24_24675

theorem rational_quotient_of_arith_geo_subseq (A d : ℝ) (h_d_nonzero : d ≠ 0)
    (h_contains_geo : ∃ (q : ℝ) (k m n : ℕ), q ≠ 1 ∧ q ≠ 0 ∧ 
        A + k * d = (A + m * d) * q ∧ A + m * d = (A + n * d) * q)
    : ∃ (r : ℚ), A / d = r :=
  sorry

end rational_quotient_of_arith_geo_subseq_l24_24675


namespace sum_binomial_inequality_l24_24221

theorem sum_binomial_inequality (n : ℕ) (h : n ≥ 9) : 
  ∑ r in Finset.range (⌊ (n : ℚ) / 2 ⌋ + 1), (n + 1 - 2 * r) / (n + 1 - r) * (Nat.choose n r) < 2 ^ (n - 2) :=
by
  sorry

end sum_binomial_inequality_l24_24221


namespace number_of_valid_configurations_l24_24012

def square := Type
def L_shape := list square

def positions := fin 11

def valid_cube_configuration (p : positions) : Prop := 
  -- Dummy placeholder for the actual validity condition
  sorry 

theorem number_of_valid_configurations : 
  (finset.filter valid_cube_configuration (finset.univ : finset positions)).card = 7 :=
sorry

end number_of_valid_configurations_l24_24012


namespace evaluate_expression_eq_four_l24_24860

noncomputable def evaluate_expression : ℝ :=
  (1 / 2)⁻¹ - real.sqrt 3 * real.tan (real.pi / 6) + (real.pi - 2023) ^ 0 + abs (-2)

theorem evaluate_expression_eq_four : evaluate_expression = 4 :=
by
  -- Definitions based on the given problem conditions
  have h1 : (1 / 2)⁻¹ = 2 := by sorry
  have h2 : real.tan (real.pi / 6) = real.sqrt 3 / 3 := by sorry
  have h3 : (real.pi - 2023) ^ 0 = 1 := by sorry
  have h4 : abs (-2) = 2 := by sorry
  -- Calculation using the conditions
  rw [h1, h2, h3, h4]
  sorry

end evaluate_expression_eq_four_l24_24860


namespace gcd_of_ropes_l24_24873

theorem gcd_of_ropes : Nat.gcd (Nat.gcd 45 75) 90 = 15 := 
by
  sorry

end gcd_of_ropes_l24_24873


namespace total_amount_spent_l24_24278

noncomputable def value_of_nickel : ℕ := 5
noncomputable def value_of_dime : ℕ := 10
noncomputable def initial_amount : ℕ := 250

def amount_spent_by_Pete (nickels_spent : ℕ) : ℕ :=
  nickels_spent * value_of_nickel

def amount_remaining_with_Raymond (dimes_left : ℕ) : ℕ :=
  dimes_left * value_of_dime

theorem total_amount_spent (nickels_spent : ℕ) (dimes_left : ℕ) :
  (amount_spent_by_Pete nickels_spent + 
   (initial_amount - amount_remaining_with_Raymond dimes_left)) = 200 :=
by
  sorry

end total_amount_spent_l24_24278


namespace smallest_number_of_points_in_T_l24_24392

-- Define the initial point
def initial_point : ℝ × ℝ := (1, 2)

-- Define the set of symmetries
def symmetries : set (ℝ × ℝ) → set (ℝ × ℝ)
| S := 
  { p | let (a, b) := p in (a, b) ∈ S ∨ (-a, b) ∈ S ∨ (a, -b) ∈ S ∨ (-a, -b) ∈ S ∨ 
                 (b, a) ∈ S ∨ (-b, -a) ∈ S ∨ (b, -a) ∈ S ∨ (-b, a) ∈ S }

-- Define the symmetric set T
def T : set (ℝ × ℝ) := 
  { p | p ∈ symmetries {initial_point} }

-- Prove that T contains 8 unique points
theorem smallest_number_of_points_in_T : fintype.card T = 8 :=
sorry

end smallest_number_of_points_in_T_l24_24392


namespace tan_product_l24_24422

noncomputable def tan : ℝ → ℝ := sorry

theorem tan_product :
  (tan (Real.pi / 8)) * (tan (3 * Real.pi / 8)) * (tan (5 * Real.pi / 8)) = 2 * Real.sqrt 7 :=
by
  sorry

end tan_product_l24_24422


namespace complex_number_solution_l24_24109

theorem complex_number_solution (z : ℂ) (hz : 1 + complex.I * real.sqrt 3 = z * (1 - complex.I * real.sqrt 3)) :
  z = -1 / 2 + complex.I * real.sqrt 3 / 2 :=
sorry

end complex_number_solution_l24_24109


namespace no_unhappy_days_l24_24736

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
  sorry

end no_unhappy_days_l24_24736


namespace simplify_evaluate_at_minus_3_l24_24694

def simplify_and_evaluate (x : ℝ) :=
  (3 / (x + 1) - x + 1) / ((x ^ 2 - 4 * x + 4) / (x + 1))

theorem simplify_evaluate_at_minus_3 :
  simplify_and_evaluate (-3) = -1 / 5 :=
by
  -- Proof to be filled in
  sorry

end simplify_evaluate_at_minus_3_l24_24694


namespace gcd_3375_9180_l24_24469

-- Definition of gcd and the problem condition
theorem gcd_3375_9180 : Nat.gcd 3375 9180 = 135 := by
  sorry -- Proof can be filled in with the steps using the Euclidean algorithm

end gcd_3375_9180_l24_24469


namespace arcsin_neg_one_eq_neg_pi_div_two_l24_24413

theorem arcsin_neg_one_eq_neg_pi_div_two : arcsin (-1) = - (Real.pi / 2) := sorry

end arcsin_neg_one_eq_neg_pi_div_two_l24_24413


namespace num_factors_of_72_l24_24553

theorem num_factors_of_72 : ∃ n : ℕ, n = 72 ∧ ∃ f : ℕ, f = 12 ∧ (factors n).length = f :=
by
  sorry

end num_factors_of_72_l24_24553


namespace cassie_nail_cutting_l24_24871

structure AnimalCounts where
  dogs : ℕ
  dog_nails_per_foot : ℕ
  dog_feet : ℕ
  parrots : ℕ
  parrot_claws_per_leg : ℕ
  parrot_legs : ℕ
  extra_toe_parrots : ℕ
  extra_toe_claws : ℕ

def total_nails (counts : AnimalCounts) : ℕ :=
  counts.dogs * counts.dog_nails_per_foot * counts.dog_feet +
  counts.parrots * counts.parrot_claws_per_leg * counts.parrot_legs +
  counts.extra_toe_parrots * counts.extra_toe_claws

theorem cassie_nail_cutting :
  total_nails {
    dogs := 4,
    dog_nails_per_foot := 4,
    dog_feet := 4,
    parrots := 8,
    parrot_claws_per_leg := 3,
    parrot_legs := 2,
    extra_toe_parrots := 1,
    extra_toe_claws := 1
  } = 113 :=
by sorry

end cassie_nail_cutting_l24_24871


namespace ratio_a6_b6_l24_24234

variable (n : ℕ) (a_n b_n : ℕ → ℕ) (S_n T_n : ℕ → ℕ)

-- Define the sums of the first n terms of sequences {a_n} and {b_n}
def S_sum (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), a_n i
def T_sum (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), b_n i

-- Conditions:
axiom seq_sum_ratio : ∀ (n : ℕ+), S_sum n / T_sum n = n / (2 * n + 1)

-- Question to prove:
theorem ratio_a6_b6 : a_n 6 / b_n 6 = 11 / 23 := by
  sorry

end ratio_a6_b6_l24_24234


namespace pie_cutting_minimum_pieces_l24_24713

/-- 
If \( p \) and \( q \) are coprime, the minimum number of pieces needed such that the pie can be 
equally divided among either \( p \) people or \( q \) people is \( p + q - 1 \).
-/
theorem pie_cutting_minimum_pieces (p q : ℕ) (h_coprime : Nat.coprime p q) : 
    ∃ n : ℕ, n = p + q - 1 := 
sorry

end pie_cutting_minimum_pieces_l24_24713


namespace tool_cost_problem_l24_24339

theorem tool_cost_problem :
  ∀ (x : ℕ) (c_retail c_wholesale : ℕ),
  (c_retail * x = 3600) ∧ (c_wholesale * (x + 60) = 3600) ∧ (c_wholesale * 60 = c_retail * 50) →
  x = 300 :=
by
  intros x c_retail c_wholesale h,
  have h1 := h.1,
  have h2 := h.2.1,
  have h3 := h.2.2,
  -- Prove that x = 300
  sorry

end tool_cost_problem_l24_24339


namespace find_m_value_l24_24124

-- Definition
noncomputable def ellipse_eccentricity_condition : Prop :=
  ∃ m : ℝ, (m = -9/4 ∨ m = 3) ∧
           (∃ x y : ℝ, (x^2 / 9 + y^2 / (m + 9) = 1) ∧ (√(m) / √(m + 9) = 1/2) ∨ (√(-m) / 3 = 1/2))

-- The theorem statement
theorem find_m_value :
  ellipse_eccentricity_condition :=
sorry

end find_m_value_l24_24124


namespace minimum_abs_z_value_l24_24093

noncomputable def minimum_abs_z : ℂ × ℝ → ℝ :=
λ (z: ℂ) (x: ℝ), abs z

theorem minimum_abs_z_value (z : ℂ) (h : ∃ x : ℝ, 4 * x^2 - 8 * z * x + 4 * complex.I + 3 = 0) : minimum_abs_z z = 1 :=
begin
  sorry
end


end minimum_abs_z_value_l24_24093


namespace tan_product_l24_24446

theorem tan_product (t : ℝ) (h1 : 1 - 7 * t^2 + 7 * t^4 - t^6 = 0)
  (h2 : t = tan (Real.pi / 8) ∨ t = tan (3 * Real.pi / 8) ∨ t = tan (5 * Real.pi / 8)) :
  tan (Real.pi / 8) * tan (3 * Real.pi / 8) * tan (5 * Real.pi / 8) = 1 := 
sorry

end tan_product_l24_24446


namespace problem_statement_l24_24084

variable (m : ℝ) (a b : ℝ)

-- Given conditions
def condition1 : Prop := 9^m = 10
def condition2 : Prop := a = 10^m - 11
def condition3 : Prop := b = 8^m - 9

-- Problem statement to prove
theorem problem_statement (h1 : condition1 m) (h2 : condition2 m a) (h3 : condition3 m b) : a > 0 ∧ 0 > b := 
sorry

end problem_statement_l24_24084


namespace mean_proportional_example_l24_24471

theorem mean_proportional_example : 
  ∀ (A B M : ℝ), M = 56.5 ∧ A = 49 → M^2 = A * B → B = 64.9 :=
by
  intros A B M h1 h2
  cases h1 with hM hA
  rw [hM, hA] at h2
  linarith

end mean_proportional_example_l24_24471


namespace family_of_four_children_includes_one_boy_one_girl_l24_24837

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - ((1/2)^4 + (1/2)^4)

theorem family_of_four_children_includes_one_boy_one_girl :
  probability_at_least_one_boy_and_one_girl = 7 / 8 :=
by
  sorry

end family_of_four_children_includes_one_boy_one_girl_l24_24837


namespace no_unhappy_days_l24_24725

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end no_unhappy_days_l24_24725


namespace no_such_natural_number_l24_24026

theorem no_such_natural_number :
  ¬ ∃ n : ℕ, 
    let s := (to_digits 10 n).sum in
    n = 2011 * s + 2011 :=
begin
  sorry
end

end no_such_natural_number_l24_24026


namespace part1_part2_l24_24788

theorem part1 (n : ℕ) (hn : 0 < n) : (3^(2 * n) - 8 * n - 1) % 64 = 0 :=
sorry

theorem part2 : (2^30 - 3) % 7 = 5 :=
sorry

end part1_part2_l24_24788


namespace distance_from_point_to_directrix_l24_24978

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24978


namespace find_c_plus_d_l24_24716

theorem find_c_plus_d (c d : ℝ)
  (h1 : c + d = c + sqrt d + (c - sqrt d) = 0)
  (h2 : c * c - d = 9) :
  c + d = -9 :=
sorry

end find_c_plus_d_l24_24716


namespace family_of_four_children_includes_one_boy_one_girl_l24_24838

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - ((1/2)^4 + (1/2)^4)

theorem family_of_four_children_includes_one_boy_one_girl :
  probability_at_least_one_boy_and_one_girl = 7 / 8 :=
by
  sorry

end family_of_four_children_includes_one_boy_one_girl_l24_24838


namespace harry_investment_l24_24261

variable (H : ℕ) -- Harry's investment
variable (P : ℕ) -- Total Profit
variable (M_inv : ℕ) -- Mary's investment
variable (profit_m : ℕ) -- Mary's profit
variable (profit_h : ℕ) -- Harry's profit
variable (f : ℕ) -- divided profit

-- Conditions
def conditions 
  (H M P f profit_m profit_h : ℕ) :=
  M_inv = 700 ∧ 
  P = 3000 ∧
  f = P / 3 ∧
  profit_m = f / 2 + (700 * f) / (700 + H) + 800 ∧
  profit_h = f / 2 + (H * f) / (700 + H) ∧
  P = profit_m + profit_h

-- Question
theorem harry_investment (hys : conditions H 700 3000 (3000 / 3) profit_m profit_h) :
  H = 300 := 
  sorry

end harry_investment_l24_24261


namespace widescreen_tv_horizontal_length_50_inch_l24_24666

noncomputable def widescreen_horizontal_length (d : ℝ) (r_w : ℝ) (r_h : ℝ) : ℝ :=
let x := Real.sqrt (d^2 / (r_w^2 + r_h^2)) in r_w * x

theorem widescreen_tv_horizontal_length_50_inch : 
  widescreen_horizontal_length 50 16 9 ≈ 43.56 :=
by
  simp [widescreen_horizontal_length, Real.sqrt]
  sorry

end widescreen_tv_horizontal_length_50_inch_l24_24666


namespace sunlovers_happy_days_l24_24734

open Nat

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end sunlovers_happy_days_l24_24734


namespace smallest_C_exists_l24_24477

def sequence_property (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, x 1 + x 2 + ... + x n ≤ x (n + 1)

theorem smallest_C_exists (C : ℝ) :
  (∀ x : ℕ → ℝ, (∀ i, 0 < x i) → sequence_property x →
    ∀ n, (sqrt (x 1) + sqrt (x 2) + ... + sqrt (x n) ≤ C * sqrt (x 1 + x 2 + ... + x n))) ↔
    C = 1 + sqrt 2 := 
sorry

end smallest_C_exists_l24_24477


namespace parabola_tangent_intersection_l24_24271

noncomputable def parabola_eqn (p : ℝ) (h : p > 0) : Prop :=
  ∀ y x : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x

noncomputable def intersect_lengths (k : ℝ) (h1 : k ≠ 0) (h2 : k = 1/2) : ℝ :=
  2 * Real.sqrt 11

theorem parabola_tangent_intersection (k : ℝ) :
  ∀ (p : ℝ), p > 0 → (parabola_eqn p) → (k = 1/2) → (∃ y1 y2 : ℝ, y1 * y2 = 16 ∧ y1 + y2 = (4 / k)) →
  (intersect_lengths k (by norm_num) (by norm_num) = 2 * Real.sqrt 11) :=
by
  intros p hp h_eqn hk h_intersect
  sorry

end parabola_tangent_intersection_l24_24271


namespace exists_five_digit_palindromic_divisible_by_5_count_five_digit_palindromic_numbers_divisible_by_5_l24_24255

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem exists_five_digit_palindromic_divisible_by_5 :
  ∃ n, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n := by
  -- Proof is omitted
  sorry

theorem count_five_digit_palindromic_numbers_divisible_by_5 :
  (finset.filter (λ n, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n) (finset.range 100000)).card = 100 := by
  -- Proof is omitted
  sorry

end exists_five_digit_palindromic_divisible_by_5_count_five_digit_palindromic_numbers_divisible_by_5_l24_24255


namespace repeating_decimal_multiplication_l24_24038

theorem repeating_decimal_multiplication :
  (0.0808080808 : ℝ) * (0.3333333333 : ℝ) = (8 / 297) := by
  sorry

end repeating_decimal_multiplication_l24_24038


namespace darren_tshirts_total_l24_24775

theorem darren_tshirts_total :
  let white_packs := 5 in
  let tshirts_per_white_pack := 6 in
  let blue_packs := 3 in
  let tshirts_per_blue_pack := 9 in
  white_packs * tshirts_per_white_pack + blue_packs * tshirts_per_blue_pack = 57 :=
by {
  sorry
}

end darren_tshirts_total_l24_24775


namespace largest_n_for_divisibility_l24_24761

theorem largest_n_for_divisibility (n : ℕ) (h : (n + 20) ∣ (n^3 + 1000)) : n ≤ 180 := 
sorry

example : ∃ n : ℕ, (n + 20) ∣ (n^3 + 1000) ∧ n = 180 :=
by
  use 180
  sorry

end largest_n_for_divisibility_l24_24761


namespace can_pick_4038_students_in_circle_l24_24332

structure Group :=
  (boy girl : Type)
  (handshake : boy → girl → Prop)

constant groups : Fin 2020 → Group

axiom distinct_groups {g1 g2 : Fin 2020} (b1 : g1.boy) (g1_girl : g1.girl) (b2 : g2.boy) (g2_girl : g2.girl) :
  g1 ≠ g2 → 
  g1.handshake b1 g1_girl ∧ g2.handshake b2 g2_girl → 
  (g1.handshake b1 g2_girl → g2.handshake b2 g1_girl)

axiom no_same_gender_handshakes {g : Fin 2020} (b1 b2 : g.boy) (g1 g2 : g.girl) :
  ¬(g.handshake b1 b2 ∨ g.handshake g1 g2)

axiom handshake_once_per_group {g : Fin 2020} (b : g.boy) (g2 : g.girl) :
  g.handshake b g2

axiom at_least_three_handshakes_for_four_students {g1 g2 : Fin 2020} (b1 : g1.boy) (g1_girl : g1.girl) (b2 : g2.boy) (g2_girl : g2.girl) :
  g1 ≠ g2 → 
  3 ≤ (if g1.handshake b1 g1_girl then 1 else 0) + (if g1.handshake b1 g2_girl then 1 else 0) + (if g2.handshake b2 g1_girl then 1 else 0) + (if g2.handshake b2 g2_girl then 1 else 0)

theorem can_pick_4038_students_in_circle :
  ∃ (selected_students : Fin 4038 → Sum (Σ i, groups i.boy) (Σ i, groups i.girl)),
    (∀ i, ∃ j, selected_students j = Sum.inl i ∨ selected_students j = Sum.inr i) ∧
    (∀ i : Fin 4037, ∃ b g, selected_students i = Sum.inl b ∧ selected_students (i + 1) = Sum.inr g ∧ groups b.fst.handshake b.2 g.2) ∧
    (∃ b g, selected_students 4037 = Sum.inl b ∧ selected_students 0 = Sum.inr g ∧ groups b.fst.handshake b.2 g.2) :=
sorry

end can_pick_4038_students_in_circle_l24_24332


namespace tape_cover_cube_l24_24096

theorem tape_cover_cube (n : ℕ) : 
  ∃ k, k = 2 * n

end tape_cover_cube_l24_24096


namespace quadrilateral_WXYZ_WX_25_l24_24182

theorem quadrilateral_WXYZ_WX_25 (W X Y Z : Type) [metric_space W] 
  [metric_space X] [metric_space Y] [metric_space Z]
  (WZ XY YZ : ℝ)
  (angleX angleY : ℝ)
  (h1 : WZ = 7)
  (h2 : XY = 14)
  (h3 : YZ = 24)
  (h4 : angleX = 90)
  (h5 : angleY = 90) :
  ∃ WX : ℝ, WX = 25 :=
by {
  sorry
}

end quadrilateral_WXYZ_WX_25_l24_24182


namespace f_18_possibilities_sum_l24_24641

theorem f_18_possibilities_sum:
  (∃ f : ℕ → ℕ, (∀ a b : ℕ, 2 * f (a^2 + b^2) = f a ^ 2 + f b ^ 2) ∧ 
  (let vals := {f 18 | ∀ t : ℕ, t ∈ { f 18 | f 18 = (f 3)^2 } } 
  in vals.card = 3 ∧ vals.sum = 5)) →
  True := sorry

end f_18_possibilities_sum_l24_24641


namespace values_of_m_l24_24488

def A : Set ℝ := { -1, 2 }
def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

theorem values_of_m (m : ℝ) : (A ∪ B m = A) ↔ (m = -1/2 ∨ m = 0 ∨ m = 1) := by
  sorry

end values_of_m_l24_24488


namespace distance_from_point_to_directrix_l24_24953

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24953


namespace circles_externally_tangent_l24_24544

-- Define the circle equations as predicates
def circle_eq (x y real : ℝ) (a b c : ℝ) := x^2 + y^2 + a * x + b * y + c = 0

-- Define the circles
def c1 : ℝ → ℝ → Prop := circle_eq x y (-4) (-6) 9
def c2 : ℝ → ℝ → Prop := circle_eq x y 12 6 (-19)

-- Define the centers and radii
def center1 := (2, 3)
def radius1 := 2

def center2 := (-6, -3)
def radius2 := 8

-- Calculate the distance between centers
def distance_between_centers : ℝ :=
  Real.sqrt ((center1.fst - center2.fst)^2 + (center1.snd - center2.snd)^2)

-- Define the relative position predicate
def externally_tangent := distance_between_centers = radius1 + radius2

-- The theorem to be proved
theorem circles_externally_tangent : externally_tangent :=
by
  sorry

end circles_externally_tangent_l24_24544


namespace find_k_l24_24706

noncomputable def distance_x (x : ℝ) := 5
noncomputable def distance_y (x k : ℝ) := |x^2 - k|
noncomputable def total_distance (x k : ℝ) := distance_x x + distance_y x k

theorem find_k (x k : ℝ) (hk : distance_y x k = 2 * distance_x x) (htot : total_distance x k = 30) :
  k = x^2 - 10 :=
sorry

end find_k_l24_24706


namespace planks_from_friends_l24_24009

theorem planks_from_friends :
  let total_planks := 200
  let planks_from_storage := total_planks / 4
  let planks_from_parents := total_planks / 2
  let planks_from_store := 30
  let planks_from_friends := total_planks - (planks_from_storage + planks_from_parents + planks_from_store)
  planks_from_friends = 20 :=
by
  let total_planks := 200
  let planks_from_storage := total_planks / 4
  let planks_from_parents := total_planks / 2
  let planks_from_store := 30
  let planks_from_friends := total_planks - (planks_from_storage + planks_from_parents + planks_from_store)
  rfl

end planks_from_friends_l24_24009


namespace Melanie_dimes_and_coins_l24_24266

-- Define all given conditions
def d1 : Nat := 7
def d2 : Nat := 8
def d3 : Nat := 4
def r : Float := 2.5

-- State the theorem to prove
theorem Melanie_dimes_and_coins :
  let d_t := d1 + d2 + d3
  let c_t := Float.ofNat d_t * r
  d_t = 19 ∧ c_t = 47.5 :=
by
  sorry

end Melanie_dimes_and_coins_l24_24266


namespace factorial_fraction_value_l24_24849

theorem factorial_fraction_value :
  (15.factorial / (6.factorial * 9.factorial) = 5005) :=
by
  sorry

end factorial_fraction_value_l24_24849


namespace convert_base4_to_decimal_l24_24453

theorem convert_base4_to_decimal : 
  let b4 := 4
  let n := [1, 0, 1, 0] -- equivalent to 1010 in base-4
  let dec := n[3] * b4^3 + n[2] * b4^2 + n[1] * b4^1 + n[0] * b4^0 
  dec = 68
:= by
  let b4 := 4
  let n := [1, 0, 1, 0]
  have : dec = n[3] * b4^3 + n[2] * b4^2 + n[1] * b4^1 + n[0] * b4^0 := rfl
  sorry

end convert_base4_to_decimal_l24_24453


namespace evaluate_expression_l24_24031

theorem evaluate_expression : (2301 - 2222)^2 / 144 = 43 := 
by 
  sorry

end evaluate_expression_l24_24031


namespace sufficient_but_not_necessary_l24_24106

theorem sufficient_but_not_necessary (a b : ℝ) (hp : a > 1 ∧ b > 1) (hq : a + b > 2 ∧ a * b > 1) : 
  (a > 1 ∧ b > 1 → a + b > 2 ∧ a * b > 1) ∧ ¬(a + b > 2 ∧ a * b > 1 → a > 1 ∧ b > 1) :=
by
  sorry

end sufficient_but_not_necessary_l24_24106


namespace value_range_of_quadratic_l24_24747

theorem value_range_of_quadratic :
  let f : ℝ → ℝ := λ x => x^2 - 4 * x + 3 in
  ∀ x, x ∈ set.Icc (-1 : ℝ) 1 → f x ∈ set.Icc (0 : ℝ) 8 :=
by
  -- here goes the proof
  sorry

end value_range_of_quadratic_l24_24747


namespace power_multiplication_l24_24848

variable (a : ℝ)

theorem power_multiplication : (-a)^3 * a^2 = -a^5 := 
sorry

end power_multiplication_l24_24848


namespace find_AB_union_and_intersection_find_a_range_l24_24145

variables (a x : ℝ)

def setA := { x | 2 < x ∧ x < 4 }
def setB (a : ℝ) := { x | (x - 3) * (x - a) < 0 }

theorem find_AB_union_and_intersection : 
  let A := setA
  let B := setB 5 in
  (A ∩ B = { x | 3 < x ∧ x < 4 }) ∧ (A ∪ B = { x | 2 < x ∧ x < 5 }) :=
by sorry

theorem find_a_range (h : setA ∩ setB a = setB a) : 
  2 ≤ a ∧ a ≤ 4 :=
by sorry

end find_AB_union_and_intersection_find_a_range_l24_24145


namespace sub_base8_l24_24408

theorem sub_base8 : (1352 - 674) == 1456 :=
by sorry

end sub_base8_l24_24408


namespace distance_from_A_to_directrix_l24_24995

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l24_24995


namespace problem_part1_problem_part2_l24_24371

def M (f : ℤ → ℝ) : Prop :=
  f 0 ≠ 0 ∧ ∀ n m : ℤ, f n * f m = f (n + m) + f (n * m)

theorem problem_part1 (f : ℤ → ℝ) (h : M f) (h1 : f 1 = 5 / 2) :
  ∀ n : ℤ, f n = (2:ℝ) ^ n + 2 ^ (-n) :=
sorry

theorem problem_part2 (f : ℤ → ℝ) (h : M f) (h2 : f 1 = real.sqrt 3) :
  ∀ n : ℤ, f n = 2 * real.cos ((real.pi * n) / 6) :=
sorry

end problem_part1_problem_part2_l24_24371


namespace sufficient_cond_l24_24920

theorem sufficient_cond (x : ℝ) (h : 1/x > 2) : x < 1/2 := 
by {
  sorry 
}

end sufficient_cond_l24_24920


namespace white_beads_count_l24_24007

-- W representing the initial number of white beads.
-- Given conditions:
-- 1. The total number of black beads is 90.
-- 2. Charley pulls out 1/6 of the black beads and 1/3 of the white beads.
-- 3. Charley pulls out 32 beads in total.

theorem white_beads_count (W : ℕ) : 
  (W / 3) + 15 = 32 → W = 51 := 
by
  intro hw
  have h1 : W / 3 = 17 := by linarith
  have h2 : W = 51 := by linarith
  exact h2

end white_beads_count_l24_24007


namespace gemstone_necklaces_sold_correct_l24_24204

-- Define the conditions
def bead_necklaces_sold : Nat := 4
def necklace_cost : Nat := 3
def total_earnings : Nat := 21
def bead_necklaces_earnings : Nat := bead_necklaces_sold * necklace_cost
def gemstone_necklaces_earnings : Nat := total_earnings - bead_necklaces_earnings
def gemstone_necklaces_sold : Nat := gemstone_necklaces_earnings / necklace_cost

-- Theorem to prove the number of gem stone necklaces sold
theorem gemstone_necklaces_sold_correct :
  gemstone_necklaces_sold = 3 :=
by
  -- Proof omitted
  sorry

end gemstone_necklaces_sold_correct_l24_24204


namespace cos_alpha_plus_beta_l24_24489

-- Conditions
def condition1 (α β : ℝ) : Prop :=
  cos (α - β / 2) = -1 / 9

def condition2 (α β : ℝ) : Prop :=
  sin (α / 2 - β) = 2 / 3

def condition3 (α : ℝ) : Prop :=
  π / 2 < α ∧ α < π

def condition4 (β : ℝ) : Prop :=
  0 < β ∧ β < π / 2

-- Final statement to prove
theorem cos_alpha_plus_beta (α β : ℝ) :
  (condition1 α β) →
  (condition2 α β) →
  (condition3 α) →
  (condition4 β) →
  cos (α + β) = -239 / 729 :=
by
  intros h1 h2 h3 h4
  sorry

end cos_alpha_plus_beta_l24_24489


namespace isosceles_triangle_circle_tangent_property_l24_24936

theorem isosceles_triangle_circle_tangent_property
  (A B C M D F E : Point)
  (h_iso : CA = CB)
  (circle_tangent_to_CA_at_A : tangent (circle (A, M)) CA A)
  (circle_tangent_to_CB_at_B : tangent (circle (B, M)) CB B)
  (M_on_arc : lies_on_arc M (circle (A, B)))
  (M_perpendicular_AB : perpendicular M D AB)
  (M_perpendicular_CA : perpendicular M F CA)
  (M_perpendicular_CB : perpendicular M E CB) :
  MD^2 = ME * MF :=
sorry

end isosceles_triangle_circle_tangent_property_l24_24936


namespace women_count_l24_24594

theorem women_count (T : ℕ) (hw_ret : ℕ) (hm_ret : ℕ) (hw_nonret : ℕ) (hm_nonret : ℕ) : 
  3 * (hm_nonret + hm_ret) = 360 → 
  hw_nonret = 0.4 * (T / 3) → 
  hm_nonret = 0.6 * (T / 3) → 
  hw_ret = 0.6 * (2 * T / 3) → 
  hm_ret = 0.4 * (2 * T / 3) →
  (hw_nonret + hw_ret) = 107 :=
by
  sorry

end women_count_l24_24594


namespace ellipse_standard_equation_l24_24114

theorem ellipse_standard_equation :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
    (∀ (x y : ℝ), x = 1 ∧ y = (sqrt 3) / 2 → (x^2 / a^2 + y^2 / b^2 = 1)) ∧
    (a^2 - b^2 = 3) ∧
    (standard_equation : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1) :=
begin
  let a := 2,
  let b := 1,
  use [a, b],
  split,
  { linarith },
  split,
  { linarith },
  split,
  {
    intros x y hxy,
    cases hxy,
    simp,
  },
  split,
  {
    calc
      a^2 - b^2 = 4 - 1 : by simp [a, b]
      ... = 3 : by norm_num,
  },
  {
    exact sorry,
  }
end

end ellipse_standard_equation_l24_24114


namespace distance_from_point_to_directrix_l24_24954

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24954


namespace not_monotonic_implies_t_range_l24_24130

theorem not_monotonic_implies_t_range (t : ℝ) :
  (∃ x : ℝ, t < x ∧ x < t + 1 ∧ (-x + 4 - 3 / x = 0)) ↔ (0 < t ∧ t < 1) ∨ (2 < t ∧ t < 3) :=
begin
  sorry
end

end not_monotonic_implies_t_range_l24_24130


namespace central_angle_of_sector_l24_24704

noncomputable def sector_area (α r : ℝ) : ℝ := (1/2) * α * r^2

theorem central_angle_of_sector :
  sector_area 3 2 = 6 :=
by
  unfold sector_area
  norm_num
  done

end central_angle_of_sector_l24_24704


namespace number_representation_fewer_sevens_exists_l24_24782

def representable_using_fewer_sevens (n : ℕ) : Prop :=
  ∃ (N : ℕ), let num := 7 * (10 ^ n - 1) / 9 in 
  N < n ∧ num = N

theorem number_representation_fewer_sevens_exists : ∃ (n : ℕ), representable_using_fewer_sevens n :=
sorry

end number_representation_fewer_sevens_exists_l24_24782


namespace volume_of_center_octahedron_in_unit_cube_l24_24604

-- Definitions based on the problem conditions
structure UnitCube where
  vertices : Fin 8 → (ℝ × ℝ × ℝ)
  unit_length : ∀ (i j : Fin 8), i ≠ j → (vertices i).distance (vertices j) = 1

-- The octahedron formed by the intersecting planes at the center of a unit cube
def octahedral_volume_in_cube (C : UnitCube) : ℝ :=
  -- Given the unit cube, return volume of the octahedron at the center
  1 / 6

-- The proof problem
theorem volume_of_center_octahedron_in_unit_cube (C : UnitCube) : 
  octahedral_volume_in_cube C = 1 / 6 := 
  sorry

end volume_of_center_octahedron_in_unit_cube_l24_24604


namespace term_250_of_modified_sequence_l24_24760

-- Define the sequence that omits perfect squares and multiples of 10
def valid_integer (n : ℕ) : Prop :=
  ¬ (∃ k : ℕ, k^2 = n) ∧ n % 10 ≠ 0

-- Define the nth term of the modified sequence
noncomputable def nth_valid_integer (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter valid_integer).nth (n - 1) -- subtracting 1 for zero-based indexing by Lean

-- The theorem to prove
theorem term_250_of_modified_sequence : nth_valid_integer 250 = 330 :=
by
  sorry

end term_250_of_modified_sequence_l24_24760


namespace tan_product_l24_24426

noncomputable def tan : ℝ → ℝ := sorry

theorem tan_product :
  (tan (Real.pi / 8)) * (tan (3 * Real.pi / 8)) * (tan (5 * Real.pi / 8)) = 2 * Real.sqrt 7 :=
by
  sorry

end tan_product_l24_24426


namespace num_factors_of_72_l24_24552

theorem num_factors_of_72 : ∃ n : ℕ, n = 72 ∧ ∃ f : ℕ, f = 12 ∧ (factors n).length = f :=
by
  sorry

end num_factors_of_72_l24_24552


namespace area_enclosed_abs_eq_ten_l24_24356

theorem area_enclosed_abs_eq_ten :
  (area {p : ℝ × ℝ | |p.1| + |2 * p.2| = 10}) = 100 := 
by
  sorry

end area_enclosed_abs_eq_ten_l24_24356


namespace carlos_picks_24_integers_l24_24006

def is_divisor (n m : ℕ) : Prop := m % n = 0

theorem carlos_picks_24_integers :
  ∃ (s : Finset ℕ), s.card = 24 ∧ ∀ n ∈ s, is_divisor n 4500 ∧ 1 ≤ n ∧ n ≤ 4500 ∧ n % 3 = 0 :=
by
  sorry

end carlos_picks_24_integers_l24_24006


namespace distance_from_point_to_directrix_l24_24981

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24981


namespace determinant_value_l24_24121

-- Given definitions and conditions
def determinant (a b c d : ℤ) : ℤ := a * d - b * c
def special_determinant (m : ℤ) : ℤ := determinant (m^2) (m-3) (1-2*m) (m-2)

-- The proof problem
theorem determinant_value (m : ℤ) (h : m^2 - 2 * m - 3 = 0) : special_determinant m = 9 := sorry

end determinant_value_l24_24121


namespace factor_tree_X_value_l24_24169

-- Define the constants
def F : ℕ := 5 * 3
def G : ℕ := 7 * 3

-- Define the intermediate values
def Y : ℕ := 5 * F
def Z : ℕ := 7 * G

-- Final value of X
def X : ℕ := Y * Z

-- Prove the value of X
theorem factor_tree_X_value : X = 11025 := by
  sorry

end factor_tree_X_value_l24_24169


namespace area_of_triangle_AOB_l24_24136

/-- Given the parabola y² = 4x and a line passing through its focus,
    if the line intersects the parabola at points A and B such that A is in the first quadrant
    and vector BA equals 4 times vector BF, then the area of triangle AOB is 4√3/3. -/
theorem area_of_triangle_AOB :
  ∀ (A B F O : ℝ × ℝ),
    (O = (0, 0)) →
    (F = (1, 0)) →
    (A = (x₁, y₁)) →
    (B = (x₂, y₂)) →
    y₁^2 = 4 * x₁ →
    y₂^2 = 4 * x₂ →
    x₁ > 0 → y₁ > 0 → -- A is in first quadrant
    ∀ l : ℝ, l = (x, my + 1) →
    y₁ = -3 * y₂ →
    m^2 = 1/3 →
    A ∈ l ∧ B ∈ l →
    ∃ (S : ℝ), S = (1/2) * |4√3/3| := 
sorry

end area_of_triangle_AOB_l24_24136


namespace distance_from_point_to_directrix_l24_24962

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24962


namespace M_inter_N_empty_l24_24921

open Complex

def setM : Set ℂ :=
  {z : ℂ | ∃ (t : ℝ), z = (t / (1 + t)) + I * ((1 + t) / t) ∧ t ≠ -1 ∧ t ≠ 0}

def setN : Set ℂ :=
  {z : ℂ | ∃ (t : ℝ), z = √2 * (Real.cos (Real.arcsin t) + I * Real.cos (Real.arccos t)) ∧ abs t ≤ 1}

theorem M_inter_N_empty : setM ∩ setN = ∅ := 
by
  sorry

end M_inter_N_empty_l24_24921


namespace flowers_given_l24_24685

theorem flowers_given (initial_flowers total_flowers flowers_given : ℕ) 
  (h1 : initial_flowers = 67) 
  (h2 : total_flowers = 90) 
  (h3 : total_flowers = initial_flowers + flowers_given) : 
  flowers_given = 23 :=
by {
  sorry
}

end flowers_given_l24_24685


namespace sum_log_inv_eq_half_p_plus_q_eq_three_l24_24922

noncomputable def a (n : ℕ) : ℝ := Real.log (n + 1) / Real.log n

theorem sum_log_inv_eq_half : 
  ∑ n in Finset.range (1023 - 1), (1 : ℝ) / Real.log (a (n + 2)) 100 = 1 / 2 := sorry

theorem p_plus_q_eq_three (p q : ℕ) (hp : p ≠ 0) (hq : q ≠ 0) (h_coprime : Nat.coprime p q) (h_eq : (q : ℝ) / p = 1 / 2) :
  p + q = 3 := sorry

end sum_log_inv_eq_half_p_plus_q_eq_three_l24_24922


namespace min_value_when_a_is_negative_one_max_value_bounds_l24_24528

-- Conditions
def f (a x : ℝ) : ℝ := a * x^2 + x
def a1 : ℝ := -1
def a : ℝ := -2
def a_lower_bound : ℝ := -2
def a_upper_bound : ℝ := 0
def interval : Set ℝ := Set.Icc 0 2

-- Part I: Minimum value when a = -1
theorem min_value_when_a_is_negative_one : 
  ∃ x ∈ interval, f a1 x = -2 := 
by
  sorry

-- Part II: Maximum value criterions
theorem max_value_bounds (a : ℝ) (H : a ∈ Set.Icc a_lower_bound a_upper_bound) :
  (∀ x ∈ interval, 
    (a ≥ -1/4 → f a ( -1 / (2 * a) ) = -1 / (4 * a)) 
    ∧ (a < -1/4 → f a 2 = 4 * a + 2 )) :=
by
  sorry

end min_value_when_a_is_negative_one_max_value_bounds_l24_24528


namespace probability_of_one_exactly_four_times_l24_24573

def roll_probability := (1 : ℝ) / 6
def non_one_probability := (5 : ℝ) / 6

lemma prob_roll_one_four_times :
  ∑ x in {1, 2, 3, 4, 5}, 
      roll_probability^4 * non_one_probability = 
    5 * (roll_probability^4 * non_one_probability) :=
by
  sorry

theorem probability_of_one_exactly_four_times :
  (5 : ℝ) * roll_probability^4 * non_one_probability = (25 : ℝ) / 7776 :=
by
  have key := prob_roll_one_four_times
  sorry

end probability_of_one_exactly_four_times_l24_24573


namespace arcsin_neg_one_eq_neg_half_pi_l24_24421

theorem arcsin_neg_one_eq_neg_half_pi :
  arcsin (-1) = - (Float.pi / 2) :=
by
  sorry

end arcsin_neg_one_eq_neg_half_pi_l24_24421


namespace product_of_nonreal_roots_l24_24474

noncomputable def polynomial := fun (x : ℂ) => x^5 - 5 * x^4 + 10 * x^3 - 10 * x^2 + 5 * x

theorem product_of_nonreal_roots :
  (∏ (r in (polynomial.roots 2430).filter (λ x, ¬is_real x)), r) = 2432 :=
sorry

end product_of_nonreal_roots_l24_24474


namespace cubic_equation_roots_l24_24696

theorem cubic_equation_roots :
  (∀ x : ℝ, (x^3 - 7*x^2 + 36 = 0) → (x = -2 ∨ x = 3 ∨ x = 6)) ∧
  ∃ (x1 x2 x3 : ℝ), (x1 * x2 = 18) ∧ (x1 * x2 * x3 = -36) :=
by
  sorry

end cubic_equation_roots_l24_24696


namespace ratio_a_to_c_l24_24166

theorem ratio_a_to_c (a b c d : ℕ) 
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) : 
  a / c = 75 / 16 := 
  by 
    sorry

end ratio_a_to_c_l24_24166


namespace determinant_matrixA_l24_24033

variable (α β : ℝ)

def matrixA : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, Real.cos α, Real.sin α],
    ![Real.sin α, 0, Real.cos β],
    ![-Real.cos α, -Real.sin β, 0]]

theorem determinant_matrixA :
  Matrix.det (matrixA α β) = Real.cos (β - 2 * α) :=
  sorry

end determinant_matrixA_l24_24033


namespace rational_root_qe_even_coeff_l24_24281

theorem rational_root_qe_even_coeff 
    (a b c : ℤ) (ha : a ≠ 0) (h_rat_root : ∃ (p q: ℤ), q ≠ 0 ∧ gcd p q = 1 ∧ p * p - 2 * p * q + c * q = 0):
    ¬ (Odd a ∧ Odd b ∧ Odd c) :=
by
suffices ⟨a, b, c, ha, h_rat_root⟩ → ¬Odd a ∧ ¬Odd b ∧ ¬Odd c
suffices sorry

end rational_root_qe_even_coeff_l24_24281


namespace behavior_of_g_as_x_approaches_infinity_l24_24451

def g (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 1

theorem behavior_of_g_as_x_approaches_infinity :
  (∀ x : ℝ, x → ∞ → g(x) → ∞) ∧ (∀ x : ℝ, x → -∞ → g(x) → -∞) :=
by
  sorry

end behavior_of_g_as_x_approaches_infinity_l24_24451


namespace problem_statement_l24_24088

theorem problem_statement (m : ℝ) (a b : ℝ) 
  (h1 : 9^m = 10) (h2 : a = 10^m - 11) (h3 : b = 8^m - 9) : 
  a > 0 ∧ 0 > b :=
sorry

end problem_statement_l24_24088


namespace solution_set_of_quadratic_inequality_l24_24906

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x^2 - 2*x - 3 > 0) ↔ (x > 3 ∨ x < -1) := 
sorry

end solution_set_of_quadratic_inequality_l24_24906


namespace pete_and_ray_spent_200_cents_l24_24272

-- Define the basic units
def cents_in_a_dollar := 100
def value_of_a_nickel := 5
def value_of_a_dime := 10

-- Define the initial amounts and spending
def pete_initial_amount := 250  -- cents
def ray_initial_amount := 250  -- cents
def pete_spent_nickels := 4 * value_of_a_nickel
def ray_remaining_dimes := 7 * value_of_a_dime

-- Calculate total amounts spent
def total_spent_pete := pete_spent_nickels
def total_spent_ray := ray_initial_amount - ray_remaining_dimes
def total_spent := total_spent_pete + total_spent_ray

-- The proof problem statement
theorem pete_and_ray_spent_200_cents : total_spent = 200 := by {
 sorry
}

end pete_and_ray_spent_200_cents_l24_24272


namespace binomial_expansion_coefficient_l24_24186

theorem binomial_expansion_coefficient :
  let T (r : ℕ) := (7.choose r) * (1 / (3 * x))^(7 - r) * (2 * x * x^(1/2))^r 
  in T 4 = 560 :=
by 
  sorry

end binomial_expansion_coefficient_l24_24186


namespace binomial_integral_formula_l24_24764

theorem binomial_integral_formula (n : ℕ) : 
  ( ∑ i in range (n + 1), (C n i) * (1 / (i + 1)) * (1 / 2)^(i + 1) )
  = (1 / (n + 1)) * ((3 / 2)^(n + 1) - 1) :=
sorry

end binomial_integral_formula_l24_24764


namespace probability_of_absolute_difference_l24_24683

open MeasureTheory
open Probability

def fair_coin_flip : Measure Bool := Measure.of_fun (fun b => if b then 1/2 else 1/2)

def die_roll : Measure ℕ := Measure.of_fun (fun n => if n ∈ {1,2,3,4,5,6} then 1/6 else 0)

def random_number : Measure (ℝ × ℝ) := Measure.of_fun (fun (x, y) =>
  (if x ∈ ({0,1} : Set ℝ) then 1/4 else MeasureUniform (0, 1) x) *
  (if y ∈ ({0,1} : Set ℝ) then 1/4 else MeasureUniform (0, 1) y))

theorem probability_of_absolute_difference :
  ∀ (x y : ℝ), P(|x - y| > 1/2) = 19/32 := 
  by sorry

end probability_of_absolute_difference_l24_24683


namespace prime_or_prime_power_sequence_l24_24915

-- Defining the sequence properties
def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ k > 0, a k = Nat.find (λ n, n > a (k - 1) ∧ ∀ m < k, RelativelyPrime n (a m))

-- Main theorem statement
theorem prime_or_prime_power_sequence (a : ℕ → ℕ) (a0 : ℕ) (h1 : a 0 = a0) (h2 : a0 > 1)
  (h3 : sequence_property a)
  (h4 : ∀ n, Prime (a n) ∨ (∃ p m, a n = p ^ m ∧ Prime p)) :
  ∃ n, a0 = 2 ^ 2 ^ n :=
sorry

end prime_or_prime_power_sequence_l24_24915


namespace sets_are_equal_l24_24656

theorem sets_are_equal :
  let M := {x | ∃ k : ℤ, x = 2 * k + 1}
  let N := {x | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}
  M = N :=
by
  sorry

end sets_are_equal_l24_24656


namespace example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l24_24244

-- Definition of a five-digit palindromic number
def is_palindromic (n : ℕ) : Prop := let s := n.to_string in s = s.reverse

-- Definition of a five-digit number
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

-- Part (a): Prove that 51715 is a five-digit palindromic number and is divisible by 5
theorem example_palindromic_divisible_by_5 :
  is_five_digit 51715 ∧ is_palindromic 51715 ∧ 51715 % 5 = 0 :=
by sorry

-- Part (b): Prove that there are exactly 100 five-digit palindromic numbers divisible by 5
theorem count_palindromic_divisible_by_5 :
  (finset.filter (λ n, is_five_digit n ∧ is_palindromic n ∧ n % 5 = 0) 
    (finset.range 100000)).card = 100 :=
by sorry

end example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l24_24244


namespace domain_f_x_range_t_g_defined_range_t_f_le_g_l24_24133

-- Definition for the domain of f(x)
def domain_f (x : ℝ) : Prop := x > -1

-- Proof obligation that domain of f(x) is x > -1
theorem domain_f_x (x : ℝ) : domain_f x ↔ (2 * log (x + 1)).isDefined ∧ (x > -1) := 
sorry

-- Definition for the condition of g(x) defined in [0, 1]
def g_defined_in_unit_interval (x : ℝ) (t : ℝ) : Prop := x ∈ Icc 0 1 → (2 * x + t > 0)

-- Proof obligation for the range of t for g(x) to be defined in [0, 1]
theorem range_t_g_defined (t : ℝ) : (∀ x ∈ Icc (0 : ℝ) 1, g_defined_in_unit_interval x t) ↔ (t > -2) :=
sorry

-- Definition for the condition of f(x) <= g(x) in [0, 1]
def f_le_g_in_unit_interval (x : ℝ) (t : ℝ) : Prop := x ∈ Icc 0 1 → (2 * log (x + 1) ≤ log (2 * x + t))

-- Proof obligation for the range of t for (f(x) ≤ g(x)) in [0, 1]
theorem range_t_f_le_g (t : ℝ) : (∀ x ∈ Icc (0 : ℝ) 1, f_le_g_in_unit_interval x t) ↔ (t ≥ 2) :=
sorry

end domain_f_x_range_t_g_defined_range_t_f_le_g_l24_24133


namespace market_value_correct_l24_24313
noncomputable theory

-- Definitions for the conditions provided
def income := 756
def investment := 6000
def brokerage := 1/4 / 100
def interest_rate := 10.5 / 100

-- Definition of the face value of the stock using the given income and the interest rate
def face_value := (income * 100) / interest_rate

-- Definition of the brokerage fee
def brokerage_fee := (face_value / 100) * 0.25

-- Market value is calculated as face_value minus brokerage_fee
def market_value := face_value - brokerage_fee

-- The theorem that needs to be proved
theorem market_value_correct : market_value = 7182 :=
by
  -- Implementation of the proof goes here
  sorry

end market_value_correct_l24_24313


namespace comparison_of_exponential_and_power_l24_24492

theorem comparison_of_exponential_and_power :
  let a := 2 ^ 0.6
  let b := 0.6 ^ 2
  a > b :=
by
  let a := 2 ^ 0.6
  let b := 0.6 ^ 2
  sorry

end comparison_of_exponential_and_power_l24_24492


namespace distance_from_point_to_directrix_l24_24990

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24990


namespace ellipse_constant_product_l24_24479

def ellipse_equation (x y a b : ℝ) :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def distance_to_foci (x y c : ℝ) :=
  (sqrt ((x + c)^2 + y^2), sqrt ((x - c)^2 + y^2))

def distance_to_tangent_line (x y a b : ℝ) : ℝ :=
  (1 / sqrt ((x / a^2)^2 + (y / b^2)^2))

theorem ellipse_constant_product (a b : ℝ) (x y : ℝ)
  (h₁ : a > b) (h₂ : b > 0)
  (h₃ : ellipse_equation x y a b) :
  let c := sqrt (a^2 - b^2),
      (r1, r2) := distance_to_foci x y c,
      d := distance_to_tangent_line x y a b in
  r1 * r2 * d^2 = a^2 * b^2 :=
sorry

end ellipse_constant_product_l24_24479


namespace abcd_sum_is_12_l24_24483

theorem abcd_sum_is_12 (a b c d : ℤ) 
  (h1 : a + c = 2) 
  (h2 : a * c + b + d = -1) 
  (h3 : a * d + b * c = 18) 
  (h4 : b * d = 24) : 
  a + b + c + d = 12 :=
sorry

end abcd_sum_is_12_l24_24483


namespace cassie_nails_l24_24866

def num_dogs : ℕ := 4
def nails_per_dog_leg : ℕ := 4
def legs_per_dog : ℕ := 4
def num_parrots : ℕ := 8
def claws_per_parrot_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def extra_claws : ℕ := 1

def total_nails_to_cut : ℕ :=
  num_dogs * nails_per_dog_leg * legs_per_dog +
  num_parrots * claws_per_parrot_leg * legs_per_parrot + extra_claws

theorem cassie_nails : total_nails_to_cut = 113 :=
  by sorry

end cassie_nails_l24_24866


namespace raft_turn_impossible_l24_24389

theorem raft_turn_impossible (A : ℝ) (h : A ≥ 2 * real.sqrt 2) : 
  (∀ (width : ℝ), width = 1 →  
  ¬ (∃ (rotate : ℝ), rotate = 90 ∧ 
  ¬ (∃ L : ℝ, rectangle_at_angle A L width))) :=
by
  sorry

/-
  Conditions:
  - A ≥ 2 * sqrt(2)
  - Canal width is 1 unit
  - $90^\circ$ turn

  Prove:
  - A raft with area A cannot turn in the canal with a width of 1 unit.
-/

end raft_turn_impossible_l24_24389


namespace probability_at_least_one_boy_and_one_girl_l24_24834

theorem probability_at_least_one_boy_and_one_girl :
  let P := (1 - (1/16 + 1/16)) = 7 / 8,
  (∀ (N: ℕ), (N = 4) → 
    let prob_all_boys := (1 / N) ^ N,
    let prob_all_girls := (1 / N) ^ N,
    let prob_at_least_one_boy_and_one_girl := 1 - (prob_all_boys + prob_all_girls)
  in prob_at_least_one_boy_and_one_girl = P) :=
by
  sorry

end probability_at_least_one_boy_and_one_girl_l24_24834


namespace holiday_price_correct_l24_24293

-- Define the problem parameters
def original_price : ℝ := 250
def first_discount_rate : ℝ := 0.40
def second_discount_rate : ℝ := 0.10

-- Define the calculation for the first discount
def price_after_first_discount (original: ℝ) (rate: ℝ) : ℝ :=
  original * (1 - rate)

-- Define the calculation for the second discount
def price_after_second_discount (intermediate: ℝ) (rate: ℝ) : ℝ :=
  intermediate * (1 - rate)

-- The final Lean statement to prove
theorem holiday_price_correct : 
  price_after_second_discount (price_after_first_discount original_price first_discount_rate) second_discount_rate = 135 :=
by
  sorry

end holiday_price_correct_l24_24293


namespace internal_angle_A_area_triangle_l24_24618

-- Definitions for the given triangle and conditions
variables {A B C M : Type}
variables {a b c : ℝ}
variables {MA MB MC : ℝ → Type} -- Vectors can be more precisely defined
variables (h1 : M = centroid A B C) -- M is the centroid
variables (h2 : MA + MB + MC = 0) -- Given vector equation
variables (h3 : a * MA + b * MB + (sqrt 3 / 3) * c * MC = 0)

-- Prove the size of the internal angle A
theorem internal_angle_A :
  (a * MA + b * MB + (sqrt 3 / 3) * c * MC = 0) → 
  (a = b = sqrt 3 / 3 * c) → 
  cos (angle A) = sqrt 3 / 2 :=
sorry

-- Given a = 3, prove the area of the triangle
theorem area_triangle (ha : a = 3) :
  (b = a) → (c = 3 * sqrt 3) → 
  area A B C = 9 * sqrt 3 / 4 :=
sorry

end internal_angle_A_area_triangle_l24_24618


namespace diff_reading_math_homework_l24_24682

-- Define the conditions as given in the problem
def pages_math_homework : ℕ := 3
def pages_reading_homework : ℕ := 4

-- The statement to prove that Rachel had 1 more page of reading homework than math homework
theorem diff_reading_math_homework : pages_reading_homework - pages_math_homework = 1 := by
  sorry

end diff_reading_math_homework_l24_24682


namespace distance_from_point_to_directrix_l24_24960

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24960


namespace max_container_volume_l24_24768

noncomputable def max_height_and_volume (length_of_steel: ℝ) (length_extra: ℝ) : ℝ × ℝ :=
let x := 0.7 in  -- derived from solving V'(x) = 0
let h := 3.2 - 2 * x in
let V := (x + length_extra) * x * h in
(h, V)

theorem max_container_volume : max_height_and_volume 14.8 0.5 = (1.8, 1.512) := 
by
  sorry

end max_container_volume_l24_24768


namespace peter_class_students_l24_24586

def total_students (students_with_two_hands students_with_one_hand students_with_three_hands : ℕ) : ℕ :=
  students_with_two_hands + students_with_one_hand + students_with_three_hands + 1

theorem peter_class_students
  (students_with_two_hands students_with_one_hand students_with_three_hands : ℕ)
  (total_hands_without_peter : ℕ) :

  students_with_two_hands = 10 →
  students_with_one_hand = 3 →
  students_with_three_hands = 1 →
  total_hands_without_peter = 20 →
  total_students students_with_two_hands students_with_one_hand students_with_three_hands = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end peter_class_students_l24_24586


namespace f_10_half_l24_24091

noncomputable def f (x : ℝ) : ℝ := x^2 / (2 * x + 1)
noncomputable def fn (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0     => x
  | n + 1 => f (fn n x)

theorem f_10_half :
  fn 10 (1 / 2) = 1 / (3 ^ 1024 - 1) :=
sorry

end f_10_half_l24_24091


namespace parallelogram_area_l24_24806

theorem parallelogram_area (s : ℝ) : 
  let side1 := s
  let side2 := 3 * s
  let angle : Real := Real.pi / 3  -- 60 degrees in radians
  area_parallelogram side1 side2 angle = (3 * s^2 * Real.sqrt 3) / 2 := 
by
  sorry

noncomputable def area_parallelogram (a b : ℝ) (θ : ℝ) : ℝ :=
  a * b * Real.sin θ
  

end parallelogram_area_l24_24806


namespace inequality_ab_l24_24365

theorem inequality_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  1 / (a^2 + 1) + 1 / (b^2 + 1) ≥ 1 := 
sorry

end inequality_ab_l24_24365


namespace coefficient_binomial_expansion_l24_24898

theorem coefficient_binomial_expansion : 
  let n : ℕ := 5
  let k : ℕ := 3
  let binomial_coeff := Nat.choose n k
  binomial_coeff = 10 :=
by
  let n := 5
  let k := 3
  have binomial_coeff := Nat.choose n k
  show binomial_coeff = 10
  sorry

end coefficient_binomial_expansion_l24_24898


namespace intersection_complement_correct_l24_24942

open Set

def A : Set ℕ := {x | (x + 4) * (x - 5) ≤ 0}
def B : Set ℕ := {x | x < 2}
def U : Set ℕ := { x | True }

theorem intersection_complement_correct :
  (A ∩ (U \ B)) = {x | x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5} :=
by
  sorry

end intersection_complement_correct_l24_24942


namespace distance_from_point_to_directrix_l24_24992

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24992


namespace no_distinct_solution_l24_24105

noncomputable def no_solution (x y z : ℝ) : Prop :=
  distinct x y z ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x*(y+z) + y*(z+x) = y*(z+x) + z*(x+y)

theorem no_distinct_solution (x y z : ℝ) (h : no_solution x y z) : false :=
by
  sorry

end no_distinct_solution_l24_24105


namespace length_AD_l24_24220

open Real

-- Define the properties of the quadrilateral
variable (A B C D: Point)
variable (angle_ABC angle_BCD: ℝ)
variable (AB BC CD: ℝ)

-- Given conditions
axiom angle_ABC_eq_135 : angle_ABC = 135 * π / 180
axiom angle_BCD_eq_120 : angle_BCD = 120 * π / 180
axiom AB_eq_sqrt_6 : AB = sqrt 6
axiom BC_eq_5_minus_sqrt_3 : BC = 5 - sqrt 3
axiom CD_eq_6 : CD = 6

-- The theorem to prove
theorem length_AD {AD : ℝ} (h : True) :
  AD = 2 * sqrt 19 :=
sorry

end length_AD_l24_24220


namespace m2_def_rate_correct_l24_24598

variable (total_production : ℕ)
variable (m1_production_percent m2_production_percent m3_production_percent : ℝ)
variable (m1_def_rate m3_def_rate non_def_rate : ℝ)

-- Given conditions
def m1_production : ℝ := m1_production_percent * total_production
def m2_production : ℝ := m2_production_percent * total_production
def m3_production : ℝ := m3_production_percent * total_production
def total_production_units : ℝ := total_production

-- Defective products calculation
def def_products_m1 : ℝ := m1_def_rate * m1_production
def def_products_m3 : ℝ := m3_def_rate * m3_production
def total_def_rate : ℝ := 1 - non_def_rate
def total_def_products : ℝ := total_def_rate * total_production_units

-- The calculated defective percentage by m2
def def_products_m2 : ℝ := total_def_products - (def_products_m1 + def_products_m3)
def def_rate_m2 : ℝ := def_products_m2 / m2_production

theorem m2_def_rate_correct :
  m1_production_percent = 0.25 → 
  m2_production_percent = 0.35 →
  m3_production_percent = 0.40 →
  m1_def_rate = 0.02 →
  m3_def_rate = 0.05 →
  non_def_rate = 0.961 →
  def_rate_m2 = 0.04 :=
by
  intros
  sorry

end m2_def_rate_correct_l24_24598


namespace solution_set_of_inequality_l24_24328

theorem solution_set_of_inequality (x : ℝ) :
  (x - 1) * (x - 2) ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end solution_set_of_inequality_l24_24328


namespace solution_set_of_f_inequality_inequality_a_b_l24_24131
noncomputable theory

-- Definition of the piecewise function f
def f (x : ℝ) : ℝ :=
  if x < -3 then -5*x - 5
  else if x <= 2 then x + 13
  else 5*x + 5

-- Question (1) Statement
theorem solution_set_of_f_inequality :
  {x : ℝ | f x > 15} = {x : ℝ | x < -4 ∨ x > 2} :=
sorry

-- Question (2) Definitions
def m := (10 : ℝ)
def condition_4a_25b (a b : ℝ) : Prop := 4*a + 25*b = m

-- Question (2) Statement
theorem inequality_a_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : condition_4a_25b a b) :
  (1 / a) + (1 / b) ≥ 49 / 10 :=
sorry

end solution_set_of_f_inequality_inequality_a_b_l24_24131


namespace sum_of_j_values_l24_24509

theorem sum_of_j_values :
  let P := (2, 9)
  let Q := (14, 20)
  let slope := (Q.2 - P.2) / (Q.1 - P.1)
  let y_intercept := P.2 - slope * P.1
  ∃ j1 j2, 
    j1 ≥ j2 ∧ 
    let y := slope * 6 + y_intercept in
    (y - j1) ^ 2 ≤ (y - j2) ^ 2 ∧
    j1 + j2 = 13 := 
by {
  sorry
}

end sum_of_j_values_l24_24509


namespace ratio_area_MNP_to_ABCD_l24_24226

theorem ratio_area_MNP_to_ABCD
  (A B C D M N P : Point)
  (square_ABCD : is_square A B C D)
  (midpoint_M : midpoint M D C)
  (midpoint_N : midpoint N A C)
  (intersection_P : intersection P (line B M) (line A C)) :
  area_ratio (triangle M N P) (square A B C D) = 1 / 24 := 
  sorry

end ratio_area_MNP_to_ABCD_l24_24226


namespace area_of_shape_l24_24495

def greatest_integer_le (x : ℝ) : ℤ := Int.floor x
def satisfying_points (x y : ℝ) : Prop := (greatest_integer_le x) ^ 2 + (greatest_integer_le y) ^ 2 = 25
def shape_area : ℝ := 12

theorem area_of_shape :
  (∃ (x y: ℝ), satisfying_points x y) → shape_area = 12 := sorry

end area_of_shape_l24_24495


namespace circle_tangent_to_line_l24_24386

def problem_statement : Prop :=
  ∃ l : ℝ → ℝ, 
  (∀ x y : ℝ, (∃ x', (x'^2 = 4 * y) ∧ (l x' = y)) ∧ (l 0 = 1)) ∧
  (∀ r : ℝ, (r = 1) → l(r) = -1)

theorem circle_tangent_to_line :
  problem_statement :=
sorry

end circle_tangent_to_line_l24_24386


namespace largest_of_seven_consecutive_integers_l24_24331

theorem largest_of_seven_consecutive_integers (a : ℕ) (h : a > 0) (sum_eq_77 : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6) = 77)) :
  a + 6 = 14 :=
by
  sorry

end largest_of_seven_consecutive_integers_l24_24331


namespace proof_longer_leg_of_smallest_triangle_l24_24174

noncomputable def longer_leg_smallest_triangle : ℝ :=
  let hypotenuse_largest_triangle := 10 in
  let short_leg_1 := hypotenuse_largest_triangle / 2 in
  let long_leg_1 := short_leg_1 * (Real.sqrt 3) in

  let short_leg_2 := long_leg_1 / 2 in
  let long_leg_2 := short_leg_2 * (Real.sqrt 3) in

  let short_leg_3 := long_leg_2 / 2 in
  let long_leg_3 := short_leg_3 * (Real.sqrt 3) in

  let short_leg_4 := long_leg_3 / 2 in
  let long_leg_4 := short_leg_4 * (Real.sqrt 3) in
  
  long_leg_4

theorem proof_longer_leg_of_smallest_triangle :
  longer_leg_smallest_triangle = (45 : ℝ) / 8 :=
  by 
    sorry

end proof_longer_leg_of_smallest_triangle_l24_24174


namespace ellipse_properties_l24_24504

noncomputable def ellipse (a b : ℝ) := (a > b ∧ b > 0) ∧ ∃ x y : ℝ, x = 1 ∧ y = (sqrt 3) / 2 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def focal_length (c : ℝ) := c = sqrt 3

noncomputable def line_intersects (k m : ℝ) := ∀ P Q : ℝ×ℝ, ∃ x1 y1 x2 y2 : ℝ, (y1 = k * x1 + m ∧ y2 = k * x2 + m ∧ (y1 * y2 = k^2 * x1 * x2)) ∧ (y1 ≠ 0 ∨ y2 ≠ 0)

noncomputable def max_area_of_triangle (O : ℝ×ℝ) (P Q : ℝ×ℝ) := ∃ x1 x2 y1 y2 : ℝ, y1 = (1/2) * x1 + sqrt(3)/2 ∧ y2 = (1/2) * x2 + sqrt(3)/2 ∧ (sqrt((x2 - x1)^2 + (y2 - y1)^2)) = 1

theorem ellipse_properties (a b : ℝ) (c : ℝ) (k m : ℝ) (O : ℝ×ℝ) (P Q : ℝ×ℝ) :
  (ellipse a b ∧ focal_length c ∧ line_intersects k m ∧ max_area_of_triangle O P Q) →
  ((a^2 = 4 ∧ b^2 = 1) ∧ (k = 1/2 ∨ k = -1/2) ∧ (max_area_of_triangle O P Q = 1)) :=
by
  sorry

end ellipse_properties_l24_24504


namespace solution_l24_24045

open Nat

-- Define the function f and its properties
def f (n : ℕ) : ℕ := sorry

noncomputable def problem : Prop :=
  (∀ m, f(m) = 1 ↔ m = 1) ∧
  (∀ m n, let d := gcd m n in f(m * n) = f(m) * f(n) / f(d)) ∧
  (∀ m, f^[2000] m = f(m)) ∧
  (∀ n, f(n) = n)

theorem solution : ∃ f : ℕ → ℕ, ∀ n, f(n) = n :=
  sorry

end solution_l24_24045


namespace average_age_of_troupe_l24_24597

theorem average_age_of_troupe
  (number_females : ℕ) (number_males : ℕ) 
  (average_age_females : ℕ) (average_age_males : ℕ)
  (total_people : ℕ) (total_age : ℕ)
  (h1 : number_females = 12) 
  (h2 : number_males = 18) 
  (h3 : average_age_females = 25) 
  (h4 : average_age_males = 30)
  (h5 : total_people = 30)
  (h6 : total_age = (25 * 12 + 30 * 18)) :
  total_age / total_people = 28 :=
by
  -- Proof goes here
  sorry

end average_age_of_troupe_l24_24597


namespace number_of_zeros_in_decimal_l24_24380

theorem number_of_zeros_in_decimal (h : "0.0𝑲02021" = 2.021 * 10 ^ (-15)) : 
  (number_of_zeros_in "0.0𝑲02021") = 16 :=
sorry

end number_of_zeros_in_decimal_l24_24380


namespace count_palindrome_five_digit_div_by_5_l24_24238

-- Define what it means for a number to be palindromic.
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Define what it means for a number to be a five-digit number.
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

-- Define what it means for a number to be divisible by 5.
def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Define the set of five-digit palindromic numbers divisible by 5.
def palindrome_five_digit_div_by_5_numbers (n : ℕ) : Prop :=
  is_five_digit n ∧ is_palindrome n ∧ is_divisible_by_5 n

-- Prove that the quantity of such numbers is 100.
theorem count_palindrome_five_digit_div_by_5 : 
  (finset.filter 
    (λ n, palindrome_five_digit_div_by_5_numbers n)
    (finset.range 100000)
  ).card = 100 :=
begin
  sorry
end

end count_palindrome_five_digit_div_by_5_l24_24238


namespace polar_coordinates_correctness_l24_24612

theorem polar_coordinates_correctness :
  (¬ (∀ {C : ℝ → Prop} {P : ℝ × ℝ}, C 1 → ¬ C (-1) ∧ 0) ∧
  ¬ (∀ {θ : ℝ}, θ = Real.arctan 1 ↔ θ = π / 4 ∨ θ = 5 * π / 4) ∧
  (∀ {ρ : ℝ}, ρ = 3 ↔ ρ = -3)) :=
by {
  sorry
}

end polar_coordinates_correctness_l24_24612


namespace machine_precision_insufficient_l24_24302

noncomputable def sample_data := [(3.0, 2), (3.5, 6), (3.8, 9), (4.4, 7), (4.5, 1)]
def sigma0_sq := 0.1
def alpha := 0.05

theorem machine_precision_insufficient :
  let n := 25
  let x_bar := (2 * 3.0 + 6 * 3.5 + 9 * 3.8 + 7 * 4.4 + 1 * 4.5) / 25
  let u := λ x : ℝ, 10 * x - 39
  let u_list := sample_data.map (λ xi, (u xi.1, xi.2))
  let sum_ni_ui := u_list.sum (λ p, p.2 * p.1)
  let sum_ni_ui_sq := u_list.sum (λ p, p.2 * p.1^2)
  let su_sq := (sum_ni_ui_sq - (sum_ni_ui^2 / n)) / (n - 1)
  let sx_sq := su_sq / 100
  let chi_sq_obs := ((n - 1) * sx_sq) / sigma0_sq
  let chi_sq_crit := 36.4 -- from chi-square table for df=24, alpha=0.05
  chi_sq_obs > chi_sq_crit :=
by 
  sorry

end machine_precision_insufficient_l24_24302


namespace problem_statement_l24_24177

variables {A B C M O Q E F : Type}
variables [geometry A] [geometry B] [geometry C] [geometry M] [geometry O] [geometry Q] [geometry E] [geometry F]

def isosceles_triangle (A B C : Type) := (A B = A C)

def midpoint (M : Type) (B C : Type) := M = (B + C) / 2

def perpendicular (l1 l2 : Type) := ∃ O where l1 ∩ l2 = {O} and ∀ x ∈ l1, ∀ y ∈ l2, ⟨x - O, y - O⟩ = 0

def collinear (E Q F : Type) := ∀ l, E ∈ l → Q ∈ l → F ∈ l

def OQ_perpendicular_EF (OQ EF : Type) := perpendicular OQ EF

def QE_equals_QF (Q E F : Type) := dist Q E = dist Q F

theorem problem_statement :
  (isosceles_triangle A B C) →
  (midpoint M B C) →
  (O ∈ (line_through A M)) →
  (perpendicular (line_through O B) (line_through A B)) →
  (Q ∈ line_segment B C) →
  (Q ≠ B) ∧ (Q ≠ C) →
  (E ∈ (line_through A B)) →
  (F ∈ (line_through A C)) →
  (collinear E Q F) →
  (OQ_perpendicular_EF OQ EF) ↔ (QE_equals_QF Q E F) :=
sorry

end problem_statement_l24_24177


namespace tan_pi_by_eight_product_l24_24433

theorem tan_pi_by_eight_product :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = -real.sqrt 2 := by 
sorry

end tan_pi_by_eight_product_l24_24433


namespace market_value_of_stock_l24_24771

theorem market_value_of_stock 
  (yield : ℝ) 
  (dividend_percentage : ℝ) 
  (face_value : ℝ) 
  (market_value : ℝ) 
  (h1 : yield = 0.10) 
  (h2 : dividend_percentage = 0.07) 
  (h3 : face_value = 100) 
  (h4 : market_value = (dividend_percentage * face_value) / yield) :
  market_value = 70 := by
  sorry

end market_value_of_stock_l24_24771


namespace polynomial_division_quotient_l24_24058

noncomputable theory
open Polynomial

def dividend : Polynomial ℤ := 8 * X ^ 4 + 2 * X ^ 3 - 9 * X ^ 2 + 4 * X - 6
def divisor : Polynomial ℤ := X - 3
def quotient : Polynomial ℤ := 8 * X ^ 3 + 26 * X ^ 2 + 69 * X + 211

theorem polynomial_division_quotient :
  (dividend / divisor) = quotient :=
sorry

end polynomial_division_quotient_l24_24058


namespace find_range_of_m_l24_24510

def has_two_distinct_negative_real_roots (m : ℝ) : Prop := 
  let Δ := m^2 - 4
  Δ > 0 ∧ -m > 0

def inequality_holds_for_all_real (m : ℝ) : Prop :=
  let Δ := (4 * (m - 2))^2 - 16
  Δ < 0

def problem_statement (m : ℝ) : Prop :=
  (has_two_distinct_negative_real_roots m ∨ inequality_holds_for_all_real m) ∧ 
  ¬(has_two_distinct_negative_real_roots m ∧ inequality_holds_for_all_real m)

theorem find_range_of_m (m : ℝ) : problem_statement m ↔ ((1 < m ∧ m ≤ 2) ∨ (3 ≤ m)) :=
by
  sorry

end find_range_of_m_l24_24510


namespace no_unhappy_days_l24_24730

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end no_unhappy_days_l24_24730


namespace distance_from_point_to_directrix_l24_24988

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24988


namespace martin_speed_correct_l24_24659

noncomputable def martin_average_speed : ℝ :=
  let total_distance : ℝ := 12
  let time_first_third := (4 / 3) -- 4 miles at 3 mph
  let time_next_quarter := 2 -- 3 miles at 1.5 mph
  let rest_time := 0.5 -- rest time
  let time_remainder := (5 / 2.5) -- 5 miles at 2.5 mph
  let total_time := time_first_third + time_next_quarter + rest_time + time_remainder
  total_distance / total_time

theorem martin_speed_correct : martin_average_speed ≈ 2.057 :=
  by
  sorry

end martin_speed_correct_l24_24659


namespace find_odd_and_monotonic_function_l24_24820

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - (f x)

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → x < y → f x < f y

def candidates := [λ x : ℝ, Real.log x, λ x : ℝ, x + 1 / x, λ x : ℝ, x^2, λ x : ℝ, x^(1 / 3)]

def correct_function := λ x : ℝ, x^(1 / 3)

theorem find_odd_and_monotonic_function :
  ∃ f ∈ candidates, is_odd_function f ∧ is_monotonically_increasing f ∧ f = correct_function :=
by
  sorry

end find_odd_and_monotonic_function_l24_24820


namespace remainder_a3_mod_3n_eq_one_l24_24222

theorem remainder_a3_mod_3n_eq_one (n : ℕ) (a : ℤ) (h : ∃ k : ℤ, a * a ≡ 1 [MOD (3 * n)]) :
  (a ^ 3) % (3 * n) = 1 := 
  sorry

end remainder_a3_mod_3n_eq_one_l24_24222


namespace prob_four_ones_in_five_rolls_l24_24576

open ProbabilityTheory

theorem prob_four_ones_in_five_rolls :
  let p_one := (1 : ℝ) / 6;
      p_not_one := 5 / 6;
      single_sequence_prob := p_one^4 * p_not_one;
      total_prob := (5 * single_sequence_prob)
  in
  total_prob = (25 / 7776) := 
by 
  sorry

end prob_four_ones_in_five_rolls_l24_24576


namespace books_remaining_in_library_l24_24335

def initial_books : ℕ := 250
def books_taken_out_Tuesday : ℕ := 120
def books_returned_Wednesday : ℕ := 35
def books_withdrawn_Thursday : ℕ := 15

theorem books_remaining_in_library :
  initial_books
  - books_taken_out_Tuesday
  + books_returned_Wednesday
  - books_withdrawn_Thursday = 150 :=
by
  sorry

end books_remaining_in_library_l24_24335


namespace find_rate_of_interest_l24_24773

-- Define the conditions
variables (P_B P_C T_B T_C R SI_B SI_C : ℝ)
variable (total_interest : ℝ)

-- Define the specific numerical values given in the problem
def PB := 5000
def PC := 3000
def TB := 2
def TC := 4
def total := 2200

theorem find_rate_of_interest (P_B P_C T_B T_C : ℝ) (total_interest : ℝ) (h1 : P_B = 5000 ) (h2 : P_C = 3000 ) (h3 : T_B = 2) (h4 : T_C = 4) (h5 : total_interest = 2200) :
    let SI_B := P_B * T_B * R / 100
    let SI_C := P_C * T_C * R / 100
in SI_B + SI_C = total_interest → R = 10 :=
by 
  intros SI_B SI_C
  sorry -- Proof goes here

end find_rate_of_interest_l24_24773


namespace term_2020_is_4039_l24_24481

theorem term_2020_is_4039 (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, n > 0 → (∑ i in finset.range n, a (i + 1)) / n = n) : 
  a 2020 = 4039 :=
sorry

end term_2020_is_4039_l24_24481


namespace distance_from_point_to_directrix_l24_24952

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24952


namespace weight_of_second_triangle_l24_24351

theorem weight_of_second_triangle :
  let side_len1 := 4
  let density1 := 0.9
  let weight1 := 10.8
  let side_len2 := 6
  let density2 := 1.2
  let weight2 := 18.7
  let area1 := (side_len1 ^ 2 * Real.sqrt 3) / 4
  let area2 := (side_len2 ^ 2 * Real.sqrt 3) / 4
  let calc_weight1 := area1 * density1
  let calc_weight2 := area2 * density2
  calc_weight1 = weight1 → calc_weight2 = weight2 := 
by
  intros
  -- Proof logic goes here
  sorry

end weight_of_second_triangle_l24_24351


namespace necessary_but_not_sufficient_condition_l24_24938

-- Defining the basic objects: lines m, n and plane α
variables (m n : Line) (α : Plane)

-- Defining parallelism between two lines
def parallel_lines (m n : Line) : Prop := m ∥ n

-- Defining the relationship that lines m, n form equal angles with plane α
def form_equal_angles_with_plane (m n : Line) (α : Plane) : Prop :=
  ∀ θ, angle_between_line_and_plane m α = θ ↔ angle_between_line_and_plane n α = θ

-- The statement to be proven in Lean
theorem necessary_but_not_sufficient_condition 
  (h1 : parallel_lines m n) :
  form_equal_angles_with_plane m n α :=
sorry

end necessary_but_not_sufficient_condition_l24_24938


namespace distance_Bella_Galya_l24_24193

theorem distance_Bella_Galya 
    (AB BV VG GD : ℝ)
    (DB : AB + 3 * BV + 2 * VG + GD = 700)
    (DV : AB + 2 * BV + 2 * VG + GD = 600)
    (DG : AB + 2 * BV + 3 * VG + GD = 650) :
    BV + VG = 150 := 
by
  -- Proof goes here
  sorry

end distance_Bella_Galya_l24_24193


namespace cost_price_equal_l24_24815

theorem cost_price_equal (total_selling_price : ℝ) (profit_percent_first profit_percent_second : ℝ) (length_first_segment length_second_segment : ℝ) (C : ℝ) :
  total_selling_price = length_first_segment * (1 + profit_percent_first / 100) * C + length_second_segment * (1 + profit_percent_second / 100) * C →
  C = 15360 / (66 + 72) :=
by {
  sorry
}

end cost_price_equal_l24_24815


namespace solution_set_l24_24515

-- Given conditions
variables {f : ℝ → ℝ} (h_even : ∀ x, f (-x) = f x) (f_deriv : ∀ x, f' x = deriv f x)
variables (h_f_neg1 : f (-1) = 4) (ineq : ∀ x, 3 * f x + x * (f' x) > 3)

-- Prove the solution set of inequality
theorem solution_set : ∀ x, f x < 1 + 3 / x^3 ↔ x ∈ set.Ioo 0 1 :=
sorry

end solution_set_l24_24515


namespace interval_length_slope_l24_24637

def S : set (ℤ × ℤ) := {p | 1 ≤ p.1 ∧ p.1 ≤ 30 ∧ 1 ≤ p.2 ∧ p.2 ≤ 30}

theorem interval_length_slope :
  let m : ℝ := slope such that 
    (card { p ∈ S | p.2 ≤ m * p.1 } = 300)
  ∃ a b : ℕ, a.gcd b = 1 ∧ (interval_of_m).length = (a : ℚ) / b ∧ a + b = 85 :=
begin
  sorry
end

end interval_length_slope_l24_24637


namespace total_nails_to_cut_l24_24863

theorem total_nails_to_cut :
  let dogs := 4 
  let legs_per_dog := 4
  let nails_per_dog_leg := 4
  let parrots := 8
  let legs_per_parrot := 2
  let nails_per_parrot_leg := 3
  let extra_nail := 1
  let total_dog_nails := dogs * legs_per_dog * nails_per_dog_leg
  let total_parrot_nails := (parrots * legs_per_parrot * nails_per_parrot_leg) + extra_nail
  total_dog_nails + total_parrot_nails = 113 :=
sorry

end total_nails_to_cut_l24_24863


namespace complement_union_eq_l24_24090

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

end complement_union_eq_l24_24090


namespace minimum_value_inequality_l24_24941

theorem minimum_value_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x^2 + y^2 + z^2 = 1) :
  (x / (1 - x^2)) + (y / (1 - y^2)) + (z / (1 - z^2)) ≥ (3 * Real.sqrt 3 / 2) :=
sorry

end minimum_value_inequality_l24_24941


namespace bernardo_wins_smallest_N_l24_24846

theorem bernardo_wins_smallest_N :
  ∃ N: ℕ, N < 50 ∧ N ≥ 37.5 ∧ (∑ d in (N.digits 10), d) = 11 :=
by
  -- The proof remains to be constructed
  sorry

end bernardo_wins_smallest_N_l24_24846


namespace tan_product_l24_24443

theorem tan_product (t : ℝ) (h1 : 1 - 7 * t^2 + 7 * t^4 - t^6 = 0)
  (h2 : t = tan (Real.pi / 8) ∨ t = tan (3 * Real.pi / 8) ∨ t = tan (5 * Real.pi / 8)) :
  tan (Real.pi / 8) * tan (3 * Real.pi / 8) * tan (5 * Real.pi / 8) = 1 := 
sorry

end tan_product_l24_24443


namespace cameron_list_length_l24_24002

-- Definitions of multiples
def smallest_multiple_perfect_square := 900
def smallest_multiple_perfect_cube := 27000
def multiple_of_30 (n : ℕ) : Prop := n % 30 = 0

-- Problem statement
theorem cameron_list_length :
  ∀ n, 900 ≤ n ∧ n ≤ 27000 ∧ multiple_of_30 n ->
  (871 = (900 - 30 + 1)) :=
sorry

end cameron_list_length_l24_24002


namespace probability_two_boys_and_three_girls_l24_24779

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_boys_and_three_girls :
  binomial_probability 5 2 0.5 = 0.3125 :=
by
  sorry

end probability_two_boys_and_three_girls_l24_24779


namespace range_of_a_l24_24888

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a-1)*x^2 + a*x + 1 ≥ 0) : a ≥ 1 :=
by {
  sorry
}

end range_of_a_l24_24888


namespace quadratic_inequality_solution_l24_24739

theorem quadratic_inequality_solution
  (x : ℝ) :
  -2 * x^2 + x < -3 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi (3 / 2) := by
  sorry

end quadratic_inequality_solution_l24_24739


namespace triangle_area_ratio_l24_24341

noncomputable def area_ratio (a b c d e f : ℕ) : ℚ :=
  (a * b) / (d * e)

theorem triangle_area_ratio : area_ratio 6 8 10 9 12 15 = 4 / 9 :=
by
  sorry

end triangle_area_ratio_l24_24341


namespace segment_parallel_l24_24703

open EuclideanGeometry

variables {P I : Point} {A B C A1 C1 A0 C0 : Point} {ABC : Triangle}

-- The conditions
def angle_bisectors_conditions (ABC : Triangle) (A1 C1 A0 C0 P I : Point) :=
  ABC.is_angle_bisector A A0 ∧ ABC.is_angle_bisector C C0 ∧
  incidence_geometry.line_through (A1 : Point) (A0 : Point) ∧
  incidence_geometry.line_through (C1 : Point) (C0 : Point) ∧
  incidence_geometry.line (A1C1: Line) ∧ incidence_geometry.line (A0C0: Line) ∧
  A1C1.inter_Lines A0C0 = P ∧ ABC.incenter = I

-- The statement to prove
theorem segment_parallel (ABC : Triangle) (A B C A1 C1 A0 C0 P I : Point)
  (h_cond : angle_bisectors_conditions ABC A1 C1 A0 C0 P I) : P.I ∥ A.C := 
sorry

end segment_parallel_l24_24703


namespace f_is_odd_and_has_zero_point_l24_24821

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def has_zero_point (f : ℝ → ℝ) : Prop :=
  ∃ x, f(x) = 0

def f (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem f_is_odd_and_has_zero_point :
  is_odd_function f ∧ has_zero_point f :=
  by
    -- Proof of is_odd_function f
    have odd_proof: is_odd_function f :=
      λ x, by
        calc
          f(-x) = 2^(-x) - 2^x : rfl
               ... = -(2^x - 2^(-x)) : by ring
               ... = -f(x) : rfl
    -- Proof of has_zero_point f
    have zero_proof: has_zero_point f :=
      ⟨0, by {
         calc
          f(0) = 2^0 - 2^(-0) : rfl
              ... = 1 - 1 : by norm_num
              ... = 0 : rfl
      }⟩
    exact ⟨odd_proof, zero_proof⟩

end f_is_odd_and_has_zero_point_l24_24821


namespace cameron_list_length_l24_24001

-- Definitions of multiples
def smallest_multiple_perfect_square := 900
def smallest_multiple_perfect_cube := 27000
def multiple_of_30 (n : ℕ) : Prop := n % 30 = 0

-- Problem statement
theorem cameron_list_length :
  ∀ n, 900 ≤ n ∧ n ≤ 27000 ∧ multiple_of_30 n ->
  (871 = (900 - 30 + 1)) :=
sorry

end cameron_list_length_l24_24001


namespace maximum_points_with_unique_3_order_sequences_l24_24010

theorem maximum_points_with_unique_3_order_sequences (n : ℕ) :
  (∀ (i j : ℕ), i ≠ j → (i < n ∧ j < n) → 
    (∀ (colors : fin n → bool), (λ k, colors ((i + k) % n)) 
      ≠ (λ k, colors ((j + k) % n))) → n ≤ 8) :=
sorry

end maximum_points_with_unique_3_order_sequences_l24_24010


namespace sequence_difference_l24_24931

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1
  else sqrt n

noncomputable def z (n : ℕ) : ℂ :=
  (∏ k in Finset.range (n+1), (1 - Complex.I / (a (k+1))))

theorem sequence_difference:
  abs (z 2019 - z 2020) = 1 :=
by
  sorry

end sequence_difference_l24_24931


namespace intersection_set_union_subset_A_l24_24707

noncomputable def A := {x : ℝ | x > 3}
noncomputable def B (a : ℝ) := {x : ℝ | a - 1 ≤ x ∧ x ≤ a + 3}

-- Statement 1: If a = 2, then A ∩ B = {x | 3 < x ≤ 5}
theorem intersection_set (a : ℝ) (h : a = 2) : A ∩ B a = {x : ℝ | 3 < x ∧ x ≤ 5} := 
by
  rw [h],
  sorry

-- Statement 2: If A ∪ B = A, then a > 4
theorem union_subset_A (a : ℝ) (h : A ∪ B a = A) : a > 4 := 
by 
  sorry

end intersection_set_union_subset_A_l24_24707


namespace sum_a_eq_sum_b_l24_24104

theorem sum_a_eq_sum_b {n : ℕ} (a : Fin n → ℕ) (b : ℕ → ℕ) (H : ∀ k, b k = {i | a i ≥ k}.card) :
  (Finset.univ.sum a) = (Finset.range (a 0 + 1)).sum b := 
sorry

end sum_a_eq_sum_b_l24_24104


namespace arcsin_neg_one_eq_neg_pi_div_two_l24_24417

theorem arcsin_neg_one_eq_neg_pi_div_two : Real.arcsin (-1) = -Real.pi / 2 :=
by
  sorry

end arcsin_neg_one_eq_neg_pi_div_two_l24_24417


namespace no_rearrangement_to_positive_and_negative_roots_l24_24623

theorem no_rearrangement_to_positive_and_negative_roots (a b c : ℝ) :
  (∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ a ≠ 0 ∧ b = -a * (x1 + x2) ∧ c = a * x1 * x2) →
  (∃ y1 y2 : ℝ, y1 > 0 ∧ y2 > 0 ∧ a ≠ 0 ∧ b != 0 ∧ c != 0 ∧ 
    (∃ b' c' : ℝ, b' ≠ b ∧ c' ≠ c ∧ 
      b' = -a * (y1 + y2) ∧ c' = a * y1 * y2)) →
  False := by
  sorry

end no_rearrangement_to_positive_and_negative_roots_l24_24623


namespace cassie_nails_l24_24868

def num_dogs : ℕ := 4
def nails_per_dog_leg : ℕ := 4
def legs_per_dog : ℕ := 4
def num_parrots : ℕ := 8
def claws_per_parrot_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def extra_claws : ℕ := 1

def total_nails_to_cut : ℕ :=
  num_dogs * nails_per_dog_leg * legs_per_dog +
  num_parrots * claws_per_parrot_leg * legs_per_parrot + extra_claws

theorem cassie_nails : total_nails_to_cut = 113 :=
  by sorry

end cassie_nails_l24_24868


namespace train_length_is_129_96_l24_24396

-- Definitions for given conditions
def train_speed_kmph : ℝ := 52
def crossing_time_sec : ℝ := 9

-- Conversion factor from km/hr to m/s
def kmph_to_mps (v : ℝ) : ℝ := v * (1000 / 3600)

-- Length of the train in meters
def train_length_meters (speed_kmph : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_mps := kmph_to_mps speed_kmph
  speed_mps * time_sec

theorem train_length_is_129_96 :
  train_length_meters train_speed_kmph crossing_time_sec = 129.96 :=
sorry

end train_length_is_129_96_l24_24396


namespace train_speed_is_36_km_per_hr_l24_24385

def jogger_speed_km_per_hr : ℝ := 9
def distance_ahead_of_jogger_m : ℝ := 240
def length_of_train_m : ℝ := 130
def time_to_pass_jogger_s : ℝ := 37

theorem train_speed_is_36_km_per_hr :
  let total_distance_to_cover := distance_ahead_of_jogger_m + length_of_train_m,
      speed_m_per_s := total_distance_to_cover / time_to_pass_jogger_s,
      speed_km_per_hr := speed_m_per_s * 3.6
  in speed_km_per_hr = 36 :=
by
  let total_distance_to_cover := distance_ahead_of_jogger_m + length_of_train_m
  let speed_m_per_s := total_distance_to_cover / time_to_pass_jogger_s
  let speed_km_per_hr := speed_m_per_s * 3.6
  have h : speed_km_per_hr = 36 := sorry
  exact h

end train_speed_is_36_km_per_hr_l24_24385


namespace distance_from_point_to_directrix_l24_24986

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24986


namespace halfway_between_l24_24902

-- Definitions based on given conditions
def a : ℚ := 1 / 7
def b : ℚ := 1 / 9

-- Theorem that needs to be proved
theorem halfway_between (h : True) : (a + b) / 2 = 8 / 63 := by
  sorry

end halfway_between_l24_24902


namespace horner_method_example_l24_24352

noncomputable def polynomial := λ x : ℝ, 2 * x^4 - x^3 + 3 * x^2 + 7

theorem horner_method_example : polynomial 3 = 54 :=
by
  unfold polynomial
  norm_num

end horner_method_example_l24_24352


namespace solve_inequality_l24_24697

noncomputable def g (x : ℝ) := Real.arcsin x + x^3

theorem solve_inequality (x : ℝ) (h1 : -1 ≤ x ∧ x ≤ 1)
    (h2 : Real.arcsin (x^2) + Real.arcsin x + x^6 + x^3 > 0) :
    0 < x ∧ x ≤ 1 :=
by
  sorry

end solve_inequality_l24_24697


namespace find_ratio_d_e_f_u_v_w_l24_24635

variables {d e f u v w : ℝ}

-- Assume the given conditions
axiom origin_eq : true
axiom fixed_point : true
axiom plane_intersect_conditions : true
axiom center_of_sphere_condition : true

theorem find_ratio_d_e_f_u_v_w
  (h1: origin_eq)
  (h2: fixed_point)
  (h3: plane_intersect_conditions)
  (h4: center_of_sphere_condition) :
  d / u + e / v + f / w = 2 :=
sorry

end find_ratio_d_e_f_u_v_w_l24_24635


namespace solve_xyz_l24_24108

variable {x y z : ℝ}

theorem solve_xyz (h1 : (x + y + z) * (xy + xz + yz) = 35) (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) : x * y * z = 8 := 
by
  sorry

end solve_xyz_l24_24108


namespace train_cross_bridge_time_l24_24816

-- Define the lengths of the train and the bridge
def length_of_train : ℝ := 120
def length_of_bridge : ℝ := 255

-- Define the speed of the train in km/hr
def speed_of_train_kmph : ℝ := 45

-- Conversion factor to convert speed from km/hr to m/s
def speed_conversion_factor : ℝ := 1000 / 3600

-- Convert the speed of the train to m/s
def speed_of_train_mps : ℝ := speed_of_train_kmph * speed_conversion_factor

-- Calculate the total distance the train needs to travel
def total_distance : ℝ := length_of_train + length_of_bridge

-- Calculate the time to cross the bridge in seconds
def time_to_cross_bridge : ℝ := total_distance / speed_of_train_mps

-- Prove that the time to cross the bridge is 30 seconds
theorem train_cross_bridge_time : time_to_cross_bridge = 30 := sorry

end train_cross_bridge_time_l24_24816


namespace geometric_sequence_formula_arithmetic_sequence_sum_l24_24925

-- Geometric sequence conditions
def a (n : ℕ) : ℝ := 3 * (2 : ℝ)^(n-1)

-- Arithmetic sequence conditions
def b (n : ℕ) : ℝ := 6 * (n - 1)

-- Given the conditions, prove the general formula for the geometric sequence
theorem geometric_sequence_formula :
  ∀ (n : ℕ), a n = 3 * (2 : ℝ)^(n-1) := sorry

-- Given the conditions, prove the sum of the first n terms of the arithmetic sequence
theorem arithmetic_sequence_sum (n : ℕ) :
  let S (n : ℕ) : ℝ := n * 0 + (n * (n - 1) / 2) * 6 
  in S n = 3 * n^2 - 3 * n := sorry

end geometric_sequence_formula_arithmetic_sequence_sum_l24_24925


namespace distance_Bella_to_Galya_l24_24190

theorem distance_Bella_to_Galya (D_B D_V D_G : ℕ) (BV VG : ℕ)
  (hD_B : D_B = 700)
  (hD_V : D_V = 600)
  (hD_G : D_G = 650)
  (hBV : BV = 100)
  (hVG : VG = 50)
  : BV + VG = 150 := by
  sorry

end distance_Bella_to_Galya_l24_24190


namespace geom_seq_m_value_l24_24611

/-- Given a geometric sequence {a_n} with a1 = 1 and common ratio q ≠ 1,
    if a_m = a_1 * a_2 * a_3 * a_4 * a_5, then m = 11. -/
theorem geom_seq_m_value (q : ℝ) (h_q : q ≠ 1) :
  ∃ (m : ℕ), (m = 11) ∧ (∃ a : ℕ → ℝ, a 1 = 1 ∧ (∀ n, a (n + 1) = a n * q ) ∧ (a m = a 1 * a 2 * a 3 * a 4 * a 5)) :=
by
  sorry

end geom_seq_m_value_l24_24611


namespace nearest_integer_to_a_pow_5_l24_24649

noncomputable theory

-- Given conditions translated to Lean
variables {a b c : ℝ}
variables (h1 : a ≥ b) (h2 : b ≥ c)
variables (h3 : a^2 * b * c + a * b^2 * c + a * b * c^2 + 8 = a + b + c)
variables (h4 : a^2 * b + a^2 * c + b^2 * c + b^2 * a + c^2 * a + c^2 * b + 3 * a * b * c = -4)
variables (h5 : a^2 * b^2 * c + a * b^2 * c^2 + a^2 * b * c^2 = 2 + a * b + b * c + c * a)
variables (h6 : a + b + c > 0)

-- The theorem to prove
theorem nearest_integer_to_a_pow_5 : 
  ∃ n : ℤ, n = 1279 ∧ (abs ((a^5 : ℝ) - n) < 0.5) :=
sorry

end nearest_integer_to_a_pow_5_l24_24649


namespace find_twin_primes_l24_24647

-- Defining the function L for the least common multiple from 2 to n
def L (n : Nat) : Nat :=
  Nat.lcmList (List.range (n-1) ⟨n > 1, sorry⟩)

-- Main theorem statement about prime numbers p and q satisfying conditions
theorem find_twin_primes : ∃ (p q : Nat), q = p + 2 ∧ Nat.prime p ∧ Nat.prime q ∧ L q > q * (L p) ∧ p = 3 ∧ q = 5 :=
by
  sorry

end find_twin_primes_l24_24647


namespace probability_of_landing_on_G_l24_24397

theorem probability_of_landing_on_G :
  let p_G : ℝ := 1 / 8,
      p_E : ℝ := 1 / 5,
      p_F : ℝ := 3 / 10,
      p_H : ℝ := p_G,
      p_I : ℝ := 2 * p_G,
  p_E + p_F + p_G + p_H + p_I = 1 :=
by
  let p_G := 1 / 8
  let p_E := 1 / 5
  let p_F := 3 / 10
  let p_H := p_G
  let p_I := 2 * p_G
  calc
    p_E + p_F + p_G + p_H + p_I = (1 / 5) + (3 / 10) + (1 / 8) + (1 / 8) + (2 * (1 / 8)) : by rfl
                          ...   = (2 / 10) + (3 / 10) + (1 / 8) + (1 / 8) + (2 / 8)     : by norm_num
                          ...   = (1 / 2) + (1/2) : by sorry
                          ...   = 1 : by norm_num

end probability_of_landing_on_G_l24_24397


namespace junior_score_is_90_l24_24167

variables (total_students juniors seniors : ℕ)
variables (average_score overall_score senior_score junior_score : ℝ)

-- Conditions
def cond1 := juniors = 0.2 * total_students
def cond2 := seniors = 0.8 * total_students
def cond3 := overall_score = 86 * total_students
def cond4 := forall j, j ∈ juniors → junior_score
def cond5 := senior_score / seniors = 85

theorem junior_score_is_90 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) (h5 : cond5) :
  juniors * junior_score = 180 → junior_score = 90 :=
sorry

end junior_score_is_90_l24_24167


namespace parallel_chords_l24_24825

variables (α β γ : ℝ) (A B C O M C₁ A₁ B₁ A₂ B₂ : ℝ)

-- Assume some necessary conditions
axiom angles_of_triangle : ∀ (α β γ : ℝ), α + β + γ = 180
axiom incenter : O = (A + B + C) / 3  -- Simplified assumption of incenter
axiom angle_bisector_CO : ∀ (θ : ℝ), CO = θ / 2
axiom exterior_angle_theorem : 
  ∀ (α β γ : ℝ), (α + β = γ) ↔ γ > α ∧ γ > β

-- Main statement to be proven
theorem parallel_chords : A₁B₁ ∥ A₂B₂ :=
  sorry

end parallel_chords_l24_24825


namespace distance_from_point_to_directrix_l24_24982

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24982


namespace lattice_points_slope_interval_length_l24_24638

theorem lattice_points_slope_interval_length :
  let S : finset (ℤ × ℤ) := finset.product (finset.range 30) (finset.range 30) in
  (∃ a b : ℤ, (∀ n : ℤ, 1 ≤ n ∧ n ≤ 30) 
  ∧ gcd a b = 1 ∧ (exists m: ℚ, (m * 30).nat_abs * (m ≤ n) = 300) 
  ∧ ((b - a) = 1) 
  ∧ a + b = 85 := sorry

end lattice_points_slope_interval_length_l24_24638


namespace proof_problem_l24_24214

-- Let a be a real number such that a ≥ 2
def a_ge_2 (a : ℝ) : Prop := a ≥ 2

-- Define x1 and x2 as the roots of the equation x^2 - a*x + 1 = 0
def roots (a x1 x2 : ℝ) : Prop := x1 * x1 - a * x1 + 1 = 0 ∧ x2 * x2 - a * x2 + 1 = 0

-- Define Sn = x1^n + x2^n for n = 1, 2, ...
def S (x1 x2 : ℝ) (n : ℕ) : ℝ := x1^n + x2^n

-- The sequence {S_n / S_{n+1}} is non-increasing
def non_increasing_seq (x1 x2 : ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n → (S x1 x2 n / S x1 x2 (n + 1)) ≥ (S x1 x2 (n + 1) / S x1 x2 (n + 2))

-- For all positive integers n, the sum ∑_{i=1}^{n} (S_i / S_{i+1}) > n - 1
def sum_ineq (x1 x2 : ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n → ∑ i in Finset.range n, (S x1 x2 (i + 1) / S x1 x2 (i + 2)) > n - 1

-- Theorem to prove: given a ≥ 2 and the roots condition, 
-- the sequence is non-increasing and the only solution for the inequality is a = 2
theorem proof_problem (a x1 x2 : ℝ) (n : ℕ) (h1 : a_ge_2 a) (h2 : roots a x1 x2) :
    non_increasing_seq x1 x2 ∧ (sum_ineq x1 x2 → a = 2) :=
by 
  sorry
  
end proof_problem_l24_24214


namespace fair_die_probability_l24_24568

noncomputable def probability_of_rolling_four_ones_in_five_rolls 
  (prob_1 : ℚ) (prob_not_1 : ℚ) (n : ℕ) (k : ℕ) : ℚ :=
(binomial n k) * (prob_1 ^ k) * (prob_not_1 ^ (n - k))

theorem fair_die_probability :
  let n := 5
  let k := 4
  let prob_1 := 1 / 6
  let prob_not_1 := 5 / 6
  probability_of_rolling_four_ones_in_five_rolls prob_1 prob_not_1 n k = 25 / 7776 := by
  sorry

end fair_die_probability_l24_24568


namespace distance_from_A_to_directrix_of_C_l24_24976

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24976


namespace evaluate_expression_eq_four_l24_24859

noncomputable def evaluate_expression : ℝ :=
  (1 / 2)⁻¹ - real.sqrt 3 * real.tan (real.pi / 6) + (real.pi - 2023) ^ 0 + abs (-2)

theorem evaluate_expression_eq_four : evaluate_expression = 4 :=
by
  -- Definitions based on the given problem conditions
  have h1 : (1 / 2)⁻¹ = 2 := by sorry
  have h2 : real.tan (real.pi / 6) = real.sqrt 3 / 3 := by sorry
  have h3 : (real.pi - 2023) ^ 0 = 1 := by sorry
  have h4 : abs (-2) = 2 := by sorry
  -- Calculation using the conditions
  rw [h1, h2, h3, h4]
  sorry

end evaluate_expression_eq_four_l24_24859


namespace range_of_t_l24_24210

noncomputable def M : set ℝ := {x | -2 < x ∧ x < 5}
noncomputable def N (t : ℝ) : set ℝ := {x | 2 - t < x ∧ x < 2 * t + 1}

theorem range_of_t (t : ℝ) : (M ∩ N t = N t) ↔ t ≤ 2 :=
sorry

end range_of_t_l24_24210


namespace sum_of_roots_eq_2023_l24_24060

/-!
  ## Statement
  Given the equation
  √(2 * x^2 - 2024 * x + 1023131) + √(3 * x^2 - 2025 * x + 1023132) + √(4 * x^2 - 2026 * x + 1023133)
  = √(x^2 - x + 1) + √(2 * x^2 - 2 * x + 2) + √(3 * x^2 - 3 * x + 3)
  Prove that the sum of all roots of this equation is 2023.
-/
theorem sum_of_roots_eq_2023 
  (h : ∀ x, 
    (Real.sqrt (2 * x^2 - 2024 * x + 1023131) 
     + Real.sqrt (3 * x^2 - 2025 * x + 1023132)
     + Real.sqrt (4 * x^2 - 2026 * x + 1023133))
    = (Real.sqrt (x^2 - x + 1) 
     + Real.sqrt (2 * x^2 - 2 * x + 2) 
     + Real.sqrt (3 * x^2 - 3 * x + 3)))
  : ∃ roots, roots.sum = 2023 :=
by 
  sorry

end sum_of_roots_eq_2023_l24_24060


namespace sum_of_paired_cards_is_53_l24_24753

-- Definitions
def red_cards := {1, 2, 3, 4, 5, 6}
def blue_cards := {3, 4, 5, 6, 7, 8}

def divides (a b : ℕ) : Prop :=
  b % a = 0

-- Question
def question : Prop :=
  ∃ (pairing : red_cards → blue_cards), 
    (∀ r ∈ red_cards, divides r (pairing r)) ∧
    (Finset.sum (Finset.filter (λ x, x ∈ red_cards) red_cards) = Finset.sum (Finset.filter (λ x, x ∈ blue_cards) (Finset.image pairing red_cards))) ∧
    (Finset.sum (Finset.filter (λ x, x ∈ red_cards) red_cards) + Finset.sum (Finset.filter (λ x, x ∈ blue_cards) (Finset.image pairing red_cards)) = 53)

-- Theorem stating the total sum of the correctly paired cards is 53
theorem sum_of_paired_cards_is_53 : question :=
sorry

end sum_of_paired_cards_is_53_l24_24753


namespace total_spent_by_pete_and_raymond_l24_24276

def initial_money_in_cents : ℕ := 250
def pete_spent_in_nickels : ℕ := 4
def nickel_value_in_cents : ℕ := 5
def raymond_dimes_left : ℕ := 7
def dime_value_in_cents : ℕ := 10

theorem total_spent_by_pete_and_raymond : 
  (pete_spent_in_nickels * nickel_value_in_cents) 
  + (initial_money_in_cents - (raymond_dimes_left * dime_value_in_cents)) = 200 := sorry

end total_spent_by_pete_and_raymond_l24_24276


namespace books_sold_l24_24312

theorem books_sold (total_books : ℕ) (fraction_remaining : ℚ) (books_left : ℚ) (books_sold : ℕ) :
  total_books = 15750 →
  fraction_remaining = (7 / 23 : ℚ) →
  books_left = total_books * fraction_remaining →
  books_sold = total_books - books_left.toNat →
  books_sold = 10957 :=
by
  intro h1 h2 h3 h4
  sorry

end books_sold_l24_24312


namespace monotonic_intervals_and_extreme_values_tangent_lines_at_point_l24_24530

noncomputable def f (x : ℝ) : ℝ := x^3 - 6 * x + 5

theorem monotonic_intervals_and_extreme_values :
  (deriv f = λ x, 3 * x^2 - 6)
  ∧ (∀ x, x ∈ Iio (-√2) ∪ Ioi (√2) → deriv f x > 0)
  ∧ (∀ x, x ∈ Ioo (-√2) (√2)  → deriv f x < 0)
  ∧ f (-√2) = 5 + 4 * √2
  ∧ f (√2) = 5 - 4 * √2 := 
by
  sorry  

theorem tangent_lines_at_point :
  let m := 1 in let n := 0 in 
  (∀ x y, (y = -3 * x + 3) → f 1 = 0) ∨ (∀ x y, (y = - 21 / 4 * x + 21 / 4) → f 1 = 0) :=
by
  sorry

end monotonic_intervals_and_extreme_values_tangent_lines_at_point_l24_24530


namespace even_numbers_count_l24_24149

open Finset

-- Define the set of digits
def digits := {0, 1, 2, 5, 7, 8}

-- Define the range of numbers
def is_in_range (n : ℕ) : Prop := 300 ≤ n ∧ n < 800

-- Define what it means for a number to have all different digits from the set
def all_different_digits (n : ℕ) : Prop :=
  let ds := n.digits 10 in
  ds.to_finset ⊆ digits ∧ ds.length = ds.to_finset.card

-- Define what it means for a number to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- The main statement
theorem even_numbers_count :
  (card {n ∈ Ico 300 800 | is_even n ∧ all_different_digits n} = 36) :=
by
  -- Sorry is used to skip the proof
  sorry

end even_numbers_count_l24_24149


namespace simplify_and_evaluate_l24_24693

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) : 
  ( ( (x^2 - 4 * x + 4) / (x^2 - 4) ) / ( (x-2) / (x^2 + 2*x) ) ) + 3 = 6 :=
by
  sorry

end simplify_and_evaluate_l24_24693


namespace J_eval_l24_24484

def J (x y z : ℝ) : ℝ := x / y + y / z + z / x

theorem J_eval : J (-3) 18 12 = -8 / 3 :=
by sorry

end J_eval_l24_24484


namespace deriv_prob1_deriv_prob2_l24_24049

noncomputable def prob1 (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem deriv_prob1 : ∀ x, deriv prob1 x = -x * Real.sin x :=
by 
  sorry

noncomputable def prob2 (x : ℝ) : ℝ := x / (Real.exp x - 1)

theorem deriv_prob2 : ∀ x, x ≠ 0 → deriv prob2 x = (Real.exp x * (1 - x) - 1) / (Real.exp x - 1)^2 :=
by
  sorry

end deriv_prob1_deriv_prob2_l24_24049


namespace matthew_friends_l24_24661

theorem matthew_friends (total_crackers : ℕ) (crackers_per_person : ℕ) (h1 : total_crackers = 36) (h2 : crackers_per_person = 2) : total_crackers / crackers_per_person = 18 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end matthew_friends_l24_24661


namespace probability_at_least_one_boy_and_one_girl_l24_24832

theorem probability_at_least_one_boy_and_one_girl :
  let P := (1 - (1/16 + 1/16)) = 7 / 8,
  (∀ (N: ℕ), (N = 4) → 
    let prob_all_boys := (1 / N) ^ N,
    let prob_all_girls := (1 / N) ^ N,
    let prob_at_least_one_boy_and_one_girl := 1 - (prob_all_boys + prob_all_girls)
  in prob_at_least_one_boy_and_one_girl = P) :=
by
  sorry

end probability_at_least_one_boy_and_one_girl_l24_24832


namespace sin_210_eq_neg_1_div_2_l24_24787

theorem sin_210_eq_neg_1_div_2 :
  sin (210 * degree) = - (1 / 2) :=
by 
  sorry

end sin_210_eq_neg_1_div_2_l24_24787


namespace christine_bought_4_more_markers_than_lucas_l24_24260

theorem christine_bought_4_more_markers_than_lucas (p : ℝ) (n_l n_c : ℕ) 
  (h1 : p > 0.01)
  (h2 : p * n_l = 2.25)
  (h3 : p * n_c = 3.25) :
  n_c = n_l + 4 :=
begin
  sorry
end

end christine_bought_4_more_markers_than_lucas_l24_24260


namespace ellipse_proof_l24_24102

-- Define the given ellipse with a > b > 0
def ellipse_eq (x y a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the conditions of the problem
def conditions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (ha_eq : a = sqrt(3))
  (f1 f2 : ℝ × ℝ) (af2_line : ℝ → ℝ) (hperim : ∀ A B, (∀ F1, (A B, af2_line B)),
  4 * sqrt(3)) (hdist_f1_line : abs(fst(f1)) = 2 * sqrt(6) / 3) (he : a ≠ 0 → c / a > sqrt(3) / 3) : Prop :=
  ∃ a b c : ℝ, ellipse_eq a b c = sqrt(2) ∧ a = sqrt(3) ∧ b = 1 ∧ c = sqrt(2)

-- Define the final ellipse equation
def ellipse_sol (x y : ℝ) : Prop :=
  (x^2 / 3) + y^2 = 1

-- Determine if there exist a value of k such that line intersects ellipse
def find_k (k : ℝ) (E : ℝ × ℝ) (hE : E = (1, 0)) : Prop :=
  ∀ P Q : ℝ × ℝ, line_eq x = k * x + 2)
  → ellipse_eq P = P_y ∧ ellipse_eq Q Q_y
  → circle_eq_diameter P Q (1, 0)

theorem ellipse_proof : 
  ∀ (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hecc : c / a > sqrt(3) / 3)
    (hdist : sqrt(3) ∧ bc = sqrt(2)), ellipse_eq = (x^2 / 3) + (y^2 / b^2) = 1 → 
    ∃ k : ℝ, k = -7 / 6 :=
  
  sorry

end ellipse_proof_l24_24102


namespace store_profit_l24_24797

variable (m n : ℝ)
variable (h_mn : m > n)

theorem store_profit : 10 * (m - n) > 0 :=
by
  sorry

end store_profit_l24_24797


namespace triplet_solution_l24_24046

theorem triplet_solution (a b c : ℕ) (h1 : a^2 + b^2 + c^2 = 2005) (h2 : a ≤ b) (h3 : b ≤ c) :
  (a = 24 ∧ b = 30 ∧ c = 23) ∨ 
  (a = 12 ∧ b = 30 ∧ c = 31) ∨
  (a = 18 ∧ b = 40 ∧ c = 9) ∨
  (a = 15 ∧ b = 22 ∧ c = 36) ∨
  (a = 12 ∧ b = 30 ∧ c = 31) :=
sorry

end triplet_solution_l24_24046


namespace first_meet_at_starting_point_l24_24350

-- Definitions
def track_length := 300
def speed_A := 2
def speed_B := 4

-- Theorem: A and B will meet at the starting point for the first time after 400 seconds.
theorem first_meet_at_starting_point : 
  (∃ (t : ℕ), t = 400 ∧ (
    (∃ (n : ℕ), n * (track_length * (speed_B - speed_A)) = t * (speed_A + speed_B) * track_length) ∨
    (∃ (m : ℕ), m * (track_length * (speed_B + speed_A)) = t * (speed_A - speed_B) * track_length))) := 
    sorry

end first_meet_at_starting_point_l24_24350


namespace has_buried_correct_number_of_bones_l24_24406

def bones_received_per_month : ℕ := 10
def number_of_months : ℕ := 5
def bones_available : ℕ := 8

def total_bones_received : ℕ := bones_received_per_month * number_of_months
def bones_buried : ℕ := total_bones_received - bones_available

theorem has_buried_correct_number_of_bones : bones_buried = 42 := by
  sorry

end has_buried_correct_number_of_bones_l24_24406


namespace pond_width_l24_24607

theorem pond_width (length depth volume : ℝ) (h_length : length = 20) (h_depth : depth = 5) (h_volume : volume = 1000) :
  ∃ (width : ℝ), width = 10 :=
by
  -- condition given in the problem
  have h : 20 * 5 * 10 = volume,
  -- substitute the given values and solve 
  sorry

end pond_width_l24_24607


namespace volume_of_tetrahedron_sphere_l24_24304

def point (α : Type*) := prod (prod α α) α 

noncomputable def volume_of_circumscribed_sphere (A B C D : point ℝ) : ℝ :=
  let distance := λ (p q : point ℝ), Real.sqrt ((p.1.1 - q.1.1)^2 + (p.1.2 - q.1.2)^2 + (p.2 - q.2)^2)
  let d := distance A D / 2
  (4 / 3) * Real.pi * d^3

theorem volume_of_tetrahedron_sphere :
  volume_of_circumscribed_sphere (0,0,Real.sqrt 5) (Real.sqrt 3,0,0) (0,1,0) (Real.sqrt 3,1,Real.sqrt 5) = 
    9 * Real.pi / 2 := sorry

end volume_of_tetrahedron_sphere_l24_24304


namespace combinatorial_identity_l24_24853

theorem combinatorial_identity :
  (Nat.factorial 15) / ((Nat.factorial 6) * (Nat.factorial 9)) = 5005 :=
sorry

end combinatorial_identity_l24_24853


namespace min_value_expr_l24_24646

theorem min_value_expr (x y z w : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) (hz : -1 < z ∧ z < 1) (hw : -2 < w ∧ w < 2) :
  2 ≤ (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w / 2)) + 1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w / 2))) :=
sorry

end min_value_expr_l24_24646


namespace tan_product_l24_24427

theorem tan_product :
  (Real.tan (Real.pi / 8)) * (Real.tan (3 * Real.pi / 8)) * (Real.tan (5 * Real.pi / 8)) = 1 :=
sorry

end tan_product_l24_24427


namespace triangle_angle_bisector_cosine_l24_24195

noncomputable def cos_angle_BAD (A B C D : Type*) [normed_group A] [normed_group B] [normed_group C] (a b c : ℝ) (cos_A : ℝ) :=
  (cos_A = -2/7) → D ∈ segment ℝ B C → angle A B C = cos_A → ∥A∥ = 4 → ∥B∥ = 7 → ∥C∥ = 9 → ∥A - D∥ = sqrt 70 / 14

theorem triangle_angle_bisector_cosine {A B C D : Type*} [normed_group A] [normed_group B] [normed_group C]
  (a b c : ℝ) (cos_A : ℝ) :
  (cos_A = -2/7) → D ∈ segment ℝ B C → ∠ A B C = cos_A → ∥A∥ = 4 → ∥B∥ = 7 → ∥C∥ = 9 → ∥A - D∥ = sqrt 70 / 14 :=
sorry

end triangle_angle_bisector_cosine_l24_24195


namespace simplify_fraction_l24_24765

theorem simplify_fraction : (1 / (2 + (2/3))) = (3 / 8) :=
by
  sorry

end simplify_fraction_l24_24765


namespace voucher_placement_l24_24801

/-- A company wants to popularize the sweets they market by hiding prize vouchers in some of the boxes.
The management believes the promotion is effective and the cost is bearable if a customer who buys 10 boxes has approximately a 50% chance of finding at least one voucher.
We aim to determine how often vouchers should be placed in the boxes to meet this requirement. -/
theorem voucher_placement (n : ℕ) (h_positive : n > 0) :
  (1 - (1 - 1/n)^10) ≥ 1/2 → n ≤ 15 :=
sorry

end voucher_placement_l24_24801


namespace ellipse_equation_area_triangle_l24_24505

noncomputable def c := Real.sqrt 2
noncomputable def e := Real.sqrt 2 / 3
noncomputable def a := 3
noncomputable def b := Real.sqrt 7

theorem ellipse_equation : 
  a > b ∧ e = Real.sqrt 2 / 3 ∧ c * 2 = Real.sqrt 2 * 2 ∧ (a = 3) ∧ (b = Real.sqrt 7) →
  (∀ x y : ℝ, (x ^ 2) / 9 + (y ^ 2) / 7 = 1) := 
by
  intro h
  sorry

theorem area_triangle :
  ∀ (A F_1 F_2 : Point) (angle_F1AF2 : Angle) (h : angle_F1AF2 = 60) : 
  ∃ (area : ℝ), 
  area = 7 * Real.sqrt 3 / 3 :=
by
  intros
  sorry

end ellipse_equation_area_triangle_l24_24505


namespace prob_four_ones_in_five_rolls_l24_24575

open ProbabilityTheory

theorem prob_four_ones_in_five_rolls :
  let p_one := (1 : ℝ) / 6;
      p_not_one := 5 / 6;
      single_sequence_prob := p_one^4 * p_not_one;
      total_prob := (5 * single_sequence_prob)
  in
  total_prob = (25 / 7776) := 
by 
  sorry

end prob_four_ones_in_five_rolls_l24_24575


namespace books_in_library_final_l24_24334

variable (initial_books : ℕ) (books_taken_out_tuesday : ℕ) 
          (books_returned_wednesday : ℕ) (books_taken_out_thursday : ℕ)

def books_left_in_library (initial_books books_taken_out_tuesday 
                          books_returned_wednesday books_taken_out_thursday : ℕ) : ℕ :=
  initial_books - books_taken_out_tuesday + books_returned_wednesday - books_taken_out_thursday

theorem books_in_library_final 
  (initial_books := 250) 
  (books_taken_out_tuesday := 120) 
  (books_returned_wednesday := 35) 
  (books_taken_out_thursday := 15) :
  books_left_in_library initial_books books_taken_out_tuesday 
                        books_returned_wednesday books_taken_out_thursday = 150 :=
by 
  sorry

end books_in_library_final_l24_24334


namespace distance_downstream_l24_24805

def V_s : ℝ := 7
def time_upstream : ℝ := 2
def distance_upstream : ℝ := 50

def V_up (V_b : ℝ) : ℝ := V_b - V_s 
def V_down (V_b : ℝ) : ℝ := V_b + V_s

def V_b : ℝ := 32 -- Result derived from the conditions

theorem distance_downstream : 
  (V_down V_b) * time_down = 78 :=
by
  let V_down := V_b + V_s
  let time_down := 2
  have V_down_def : V_down = 39, by sorry
  have time_down_def : time_down = 2, by sorry
  have distance_def : V_down * time_down = 78, by sorry
  exact distance_def

end distance_downstream_l24_24805


namespace max_xyz_l24_24230

theorem max_xyz (x y z : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : 5 * x + 8 * y + 3 * z = 90) : xyz ≤ 225 :=
by
  sorry

end max_xyz_l24_24230


namespace angle_bisector_length_l24_24172

theorem angle_bisector_length {A B C M : Type} [triangle : Triangle ABC] 
  (right_angle : ∠ABC = 90) (bisects : is_angle_bisector_of M ∠CAB BC) (BM MC : ℝ)
  (h : BM = 1) (h2 : MC = 2) : length_AM = 2 := by
sorry

end angle_bisector_length_l24_24172


namespace count_palindrome_five_digit_div_by_5_l24_24243

-- Define what it means for a number to be palindromic.
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Define what it means for a number to be a five-digit number.
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

-- Define what it means for a number to be divisible by 5.
def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Define the set of five-digit palindromic numbers divisible by 5.
def palindrome_five_digit_div_by_5_numbers (n : ℕ) : Prop :=
  is_five_digit n ∧ is_palindrome n ∧ is_divisible_by_5 n

-- Prove that the quantity of such numbers is 100.
theorem count_palindrome_five_digit_div_by_5 : 
  (finset.filter 
    (λ n, palindrome_five_digit_div_by_5_numbers n)
    (finset.range 100000)
  ).card = 100 :=
begin
  sorry
end

end count_palindrome_five_digit_div_by_5_l24_24243


namespace triangle_area_ratio_l24_24347

theorem triangle_area_ratio (a b c : ℕ) (d e f : ℕ) 
  (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) 
  (h4 : d = 9) (h5 : e = 12) (h6 : f = 15) 
  (GHI_right : a^2 + b^2 = c^2)
  (JKL_right : d^2 + e^2 = f^2):
  (0.5 * a * b) / (0.5 * d * e) = 4 / 9 := 
by 
  sorry

end triangle_area_ratio_l24_24347


namespace lines_eventually_identical_l24_24671

noncomputable def stable_lines_after_steps (n : ℕ) (line : Fin n → ℕ) : ℕ :=
  sorry -- This definition should capture the transformation process

theorem lines_eventually_identical (n : ℕ) (line : Fin n → ℕ) :
  ∃ k, ∀ m ≥ k, stable_lines_after_steps n line = stable_lines_after_steps n (stable_lines_after_steps n line) :=
by
  assume n line
  -- Prove the existence of such a k where the lines become identical after step k
  sorry

end lines_eventually_identical_l24_24671


namespace ones_digit_of_11_pow_46_l24_24358

theorem ones_digit_of_11_pow_46 : (11 ^ 46) % 10 = 1 :=
by sorry

end ones_digit_of_11_pow_46_l24_24358


namespace triangle_ratio_l24_24585

theorem triangle_ratio (P Q R X Y Z : Point) 
  (h1 : lies_on_line_segment X QR) 
  (h2 : lies_on_line_segment Y PR) 
  (h3 : intersects_at PX QY Z) 
  (h4 : PZ / ZX = 2) 
  (h5 : QZ / ZY = 5) : RX / RY = 5 / 4 :=
by 
  sorry

end triangle_ratio_l24_24585


namespace monomial_coefficient_l24_24885

theorem monomial_coefficient : 
  ∀ (a b : ℝ), ∃ (c : ℝ), c * a * b^2 = -2 * real.pi * a * b^2 := 
by 
  intros a b
  use -2 * real.pi
  sorry

end monomial_coefficient_l24_24885


namespace product_of_coefficients_l24_24751

theorem product_of_coefficients (b c : ℤ)
  (H1 : ∀ r, r^2 - 2 * r - 1 = 0 → r^5 - b * r - c = 0):
  b * c = 348 :=
by
  -- Solution steps would go here
  sorry

end product_of_coefficients_l24_24751


namespace distance_from_point_to_directrix_l24_24987

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24987


namespace unit_digit_of_4137_pow_754_l24_24361

theorem unit_digit_of_4137_pow_754 : (4137 ^ 754) % 10 = 9 := by
  -- We only care about the unit digit, which corresponds to modular arithmetic modulo 10
  -- Since 4137 ≡ 7 (mod 10), we reduce the problem to finding (7 ^ 754) % 10
  -- The powers of 7 modulo 10 repeat every 4: 7, 9, 3, 1
  -- So, we compute the exponent modulo 4 to find the position in the cycle
  have exp_mod : (754 % 4) = 2 := by sorry
  -- Now, we know that (7 ^ 754) % 10 is the same as (7 ^ 2) % 10
  have pattern : (7 ^ 2) % 10 = 9 := by sorry
  -- Therefore, the unit digit of (4137 ^ 754) is the same as that of (7 ^ 2)
  exact Eq.trans (Nat.pow_mod 4137 754 10) (trans (congr_arg (Nat.pow 7) exp_mod) pattern)

end unit_digit_of_4137_pow_754_l24_24361


namespace percentage_A_is_22_l24_24295

noncomputable def percentage_A_in_mixture : ℝ :=
  (0.8 * 0.20 + 0.2 * 0.30) * 100

theorem percentage_A_is_22 :
  percentage_A_in_mixture = 22 := 
by
  sorry

end percentage_A_is_22_l24_24295


namespace fair_die_probability_l24_24569

noncomputable def probability_of_rolling_four_ones_in_five_rolls 
  (prob_1 : ℚ) (prob_not_1 : ℚ) (n : ℕ) (k : ℕ) : ℚ :=
(binomial n k) * (prob_1 ^ k) * (prob_not_1 ^ (n - k))

theorem fair_die_probability :
  let n := 5
  let k := 4
  let prob_1 := 1 / 6
  let prob_not_1 := 5 / 6
  probability_of_rolling_four_ones_in_five_rolls prob_1 prob_not_1 n k = 25 / 7776 := by
  sorry

end fair_die_probability_l24_24569


namespace number_of_valid_A_is_8_l24_24153
-- Import the necessary Lean libraries

open Set

-- Given definitions
variables {A B : Set.{0} ℕ}
variable {a : ℕ}

-- The conditions given in the problem
axiom h1 : {a} ⊆ (A ∪ B)
axiom h2 : (A ∪ B) ⊆ {a, b, c, d}
axiom h3 : a ∈ B
axiom h4 : A ∩ B = ∅

-- The statement to be proved
theorem number_of_valid_A_is_8 : (∃ (s : Finset (Set ℕ)), s.card = 8 ∧ ∀ t ∈ s, {a} ⊆ (t ∪ B) ∧ (t ∪ B) ⊆ {a, b, c, d} ∧ a ∈ B ∧ t ∩ B = ∅) :=
by sorry

end number_of_valid_A_is_8_l24_24153


namespace bead_arrangement_probability_l24_24486

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

noncomputable def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def total_orderings : ℕ :=
  factorial 9 / (factorial 4 * factorial 3 * factorial 2)

noncomputable def valid_arrangements : ℕ := 10

noncomputable def probability (numerator denominator : ℕ) : ℚ :=
  (numerator : ℚ) / (denominator : ℚ)

theorem bead_arrangement_probability :
  probability valid_arrangements total_orderings = 1 / 126 :=
by
  -- Proof skipped
  sorry

end bead_arrangement_probability_l24_24486


namespace sandwiches_bread_slices_l24_24627

theorem sandwiches_bread_slices :
  ∀ (sandwiches packs slices_per_pack : ℕ),
    sandwiches = 8 →
    packs = 4 →
    slices_per_pack = 4 →
    (packs * slices_per_pack) / sandwiches = 2 :=
by
  intros sandwiches packs slices_per_pack h_sandwiches h_packs h_slices_per_pack
  rw [h_sandwiches, h_packs, h_slices_per_pack]
  norm_num
  sorry

end sandwiches_bread_slices_l24_24627


namespace num_factors_72_l24_24551

-- Define the prime factorization of 72
def prime_factors_of_72 := (2 ^ 3) * (3 ^ 2)

-- Define a helper function to count the number of factors
def num_factors (n : ℕ) : ℕ := 
  (nat.factors n).to_finset.card + 1

-- Theorem statement
theorem num_factors_72 : num_factors 72 = 12 := 
  sorry

end num_factors_72_l24_24551


namespace no_unhappy_days_l24_24724

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end no_unhappy_days_l24_24724


namespace fraction_division_l24_24355

theorem fraction_division: 
  ((3 + 1 / 2) / 7) / (5 / 3) = 3 / 10 := 
by 
  sorry

end fraction_division_l24_24355


namespace problem_statement_l24_24082

variable (m : ℝ) (a b : ℝ)

-- Given conditions
def condition1 : Prop := 9^m = 10
def condition2 : Prop := a = 10^m - 11
def condition3 : Prop := b = 8^m - 9

-- Problem statement to prove
theorem problem_statement (h1 : condition1 m) (h2 : condition2 m a) (h3 : condition3 m b) : a > 0 ∧ 0 > b := 
sorry

end problem_statement_l24_24082


namespace ratio_of_area_l24_24345

-- Define the sides of the triangles
def sides_GHI := (6, 8, 10)
def sides_JKL := (9, 12, 15)

-- Define the function to calculate the area of a triangle given its sides as a Pythagorean triple
def area_of_right_triangle (a b : Nat) : Nat :=
  (a * b) / 2

-- Calculate the areas
def area_GHI := area_of_right_triangle 6 8
def area_JKL := area_of_right_triangle 9 12

-- Calculate the ratio of the areas
def ratio_of_areas := area_GHI * 2 / area_JKL * 3

-- Statement to prove
theorem ratio_of_area (sides_GHI sides_JKL : (Nat × Nat × Nat)) :
  (ratio_of_areas = (4 : ℚ) / 9) :=
sorry

end ratio_of_area_l24_24345


namespace solve_for_x_l24_24296

-- Let's define the components of the problem first
def numerator : ℝ := real.sqrt (8^2 + 15^2)
def denominator : ℝ := real.sqrt (49 + 64)
def x : ℝ := numerator / denominator

-- State the theorem
theorem solve_for_x : x = 17 / real.sqrt 113 :=
by 
  have h_num : numerator = 17 := by sorry
  have h_den : denominator = real.sqrt 113 := by sorry
  rw [h_num, h_den]
  sorry

end solve_for_x_l24_24296


namespace compute_five_fold_application_l24_24633

def f (x : ℤ) : ℤ :=
  if x ≥ 0 then -2 * x^2 else x^2 + 4 * x + 12

theorem compute_five_fold_application :
  f (f (f (f (f 2)))) = -449183247763232 :=
  by
    sorry

end compute_five_fold_application_l24_24633


namespace abs_function_le_two_l24_24283

theorem abs_function_le_two {x : ℝ} (h : |x| ≤ 2) : |3 * x - x^3| ≤ 2 :=
sorry

end abs_function_le_two_l24_24283


namespace no_values_of_b_l24_24914

def f (b x : ℝ) := x^2 + b * x - 1

theorem no_values_of_b : ∀ b : ℝ, ∃ x : ℝ, f b x = 3 :=
by
  intro b
  use 0  -- example, needs actual computation
  sorry

end no_values_of_b_l24_24914


namespace equal_angles_PAC_QAB_l24_24270

universe u
variables {α : Type u} [EuclideanGeometry α]

theorem equal_angles_PAC_QAB (A B C A1 B1 C1 : α) (X P Q : α)
(hX_on_bisector : IsOnBisector A A1 X)
(hBX_B1 : LineThrough B X ∩ LineThrough A C = {B1})
(hCX_C1 : LineThrough C X ∩ LineThrough A B = {C1})
(hA1B1_CC1_P : LineSegment A1 B1 ∩ LineSegment C C1 = {P})
(hA1C1_BB1_Q : LineSegment A1 C1 ∩ LineSegment B B1 = {Q}) :
Angle A P C = Angle Q A B :=
sorry

end equal_angles_PAC_QAB_l24_24270


namespace factorial_fraction_eq_l24_24856

theorem factorial_fraction_eq :
  (15.factorial / (6.factorial * 9.factorial) = 5005) := 
sorry

end factorial_fraction_eq_l24_24856


namespace problem_statement_l24_24085

theorem problem_statement (m : ℝ) (a b : ℝ) 
  (h1 : 9^m = 10) (h2 : a = 10^m - 11) (h3 : b = 8^m - 9) : 
  a > 0 ∧ 0 > b :=
sorry

end problem_statement_l24_24085


namespace area_of_LOM_is_approximately_11_l24_24602

variable (α β γ : ℝ)
variable (A B C L O M : Type)

-- Definitions for the properties of the triangle and bisectors
def is_scalene (t : Triangle) : Prop := t.A ≠ t.B ∧ t.B ≠ t.C ∧ t.C ≠ t.A

def angle_sum (α β γ : ℝ) : Prop := α + β + γ = 180

def circumcircle_bisectors_meet
  (t : Triangle) (L O M : Point) : Prop :=
  let A' := angle_bisector_meet_circumcircle t.A
  let B' := angle_bisector_meet_circumcircle t.B
  let C' := angle_bisector_meet_circumcircle t.C
  A' = L ∧ B' = O ∧ C' = M

noncomputable def area (t : Triangle) : ℝ := 8

-- Main theorem statement
theorem area_of_LOM_is_approximately_11
  (h_scalene : is_scalene TriangleABC)
  (h_angle_diff : α = β - γ)
  (h_angle_twice : β = 2 * γ)
  (h_angle_sum : angle_sum α β γ)
  (h_area_ABC : area TriangleABC = 8)
  (h_circ_meet : circumcircle_bisectors_meet TriangleABC L O M) :
  area (TriangleLOM) ≈ 11 := sorry

end area_of_LOM_is_approximately_11_l24_24602


namespace distance_from_point_to_directrix_l24_24961

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24961


namespace quadratic_has_two_distinct_real_roots_l24_24326

theorem quadratic_has_two_distinct_real_roots :
  ∀ (a b c : ℝ), a = 4 → b = -2 → c = -1/4 → (b^2 - 4 * a * c) > 0 →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (4*x₁^2 - 2*x₁ - 1/4 = 0) ∧ (4*x₂^2 - 2*x₂ - 1/4 = 0) := 
by {
  intros a b c ha hb hc hΔ,
  have : b^2 - 4 * a * c = 8, by {
    rw [ha, hb, hc],
    calc (-2)^2 - 4 * 4 * (-1/4)
        = 4 + 4 : by norm_num
        ... = 8 : by norm_num,
  },
  have h : 8 > 0 := by norm_num,
  sorry
}

end quadratic_has_two_distinct_real_roots_l24_24326


namespace parabola_focus_eq_directrix_minimum_area_triangle_l24_24534

-- Definition of the first problem part: equation and directrix of parabola C1
theorem parabola_focus_eq_directrix (p : ℝ) (P : ℝ × ℝ) :
  x^2 = 2 * p * y ∧ (∃ x y, x^2 = 2 * p * y ∧ y = x^2 + 1 ∧ P = (2 * t, t^2)) →
  (x^2 = 4 * y) ∧ (directrix = y = -1 ) :=
sorry

-- Definition of the second problem part: minimum area of triangle
theorem minimum_area_triangle (t : ℝ) (x1 x2 : ℝ) 
  (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  (P = (2 * t, t^2)) ∧ 
  (∃ y1 y2, y1 = x1^2 + 1 ∧ y2 = x2^2 + 1 ∧ 
             tangent_through_P_A ∧ tangent_through_P_B) →
  (area_triangle P A B has minimum 2) :=
sorry

end parabola_focus_eq_directrix_minimum_area_triangle_l24_24534


namespace veranda_area_l24_24307

/-- The width of the veranda on all sides of the room. -/
def width_of_veranda : ℝ := 2

/-- The length of the room. -/
def length_of_room : ℝ := 21

/-- The width of the room. -/
def width_of_room : ℝ := 12

/-- The area of the veranda given the conditions. -/
theorem veranda_area (length_of_room width_of_room width_of_veranda : ℝ) :
  (length_of_room + 2 * width_of_veranda) * (width_of_room + 2 * width_of_veranda) - length_of_room * width_of_room = 148 :=
by
  sorry

end veranda_area_l24_24307


namespace tan_product_l24_24444

theorem tan_product (t : ℝ) (h1 : 1 - 7 * t^2 + 7 * t^4 - t^6 = 0)
  (h2 : t = tan (Real.pi / 8) ∨ t = tan (3 * Real.pi / 8) ∨ t = tan (5 * Real.pi / 8)) :
  tan (Real.pi / 8) * tan (3 * Real.pi / 8) * tan (5 * Real.pi / 8) = 1 := 
sorry

end tan_product_l24_24444


namespace distance_from_A_to_directrix_of_C_l24_24969

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24969


namespace area_second_cross_section_l24_24601

variable {V : Type} [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Given conditions as variables and hypotheses
variables (a : ℝ) (A B C D A₁ B₁ C₁ D₁ O O₁ : V)
(h_prism : ∀ {X Y : V}, X ≠ Y → ∥X - Y∥ = a)
(h_axis : O ≠ O₁ → ∥O - O₁∥ = 2 * a)

-- Plane 1 and midpoints
variables (M N Q : V)
(h_M : M = midpoint ℝ A D)
(h_N : N = midpoint ℝ C D)
(h_Q : Q = midpoint ℝ O O₁)
(h_area1 : 12 = area (plane_section M N Q))

-- Plane 2 and specific point dividing the axis
variables (P : V)
(h_P : ∥P - O∥ = a / 2)

-- The second cross section we're interested in
variables (h_area2 : area (axis_division_section P) = 9)

-- The goal to prove is that the second cross section has the area 9
theorem area_second_cross_section :
  area (axis_division_section P) = 9 :=
sorry

end area_second_cross_section_l24_24601


namespace sum_of_reciprocals_l24_24645

-- Given conditions
variable (p q r : ℂ) 
-- p, q, r are roots of the given polynomial
variable (h₀ : (Polynomial.X ^ 3 - Polynomial.X - 2).is_root p)
variable (h₁ : (Polynomial.X ^ 3 - Polynomial.X - 2).is_root q)
variable (h₂ : (Polynomial.X ^ 3 - Polynomial.X - 2).is_root r)

-- The proof problem
theorem sum_of_reciprocals (h₀ : (Polynomial.X ^ 3 - Polynomial.X - 2).is_root p)
                          (h₁ : (Polynomial.X ^ 3 - Polynomial.X - 2).is_root q)
                          (h₂ : (Polynomial.X ^ 3 - Polynomial.X - 2).is_root r) :
  (1 / (p - 1)) + (1 / (q - 1)) + (1 / (r - 1)) = -2 :=
by
  sorry

end sum_of_reciprocals_l24_24645


namespace points_A_N_I_M_concyclic_l24_24101

-- Definition of the problem's geometric setup
structure TriangleIncenter (ABC : Type*) :=
(A B C I D N M : ABC)
(perpend_bis_AD_int_CI : N = intersection (perpendicular_bisector AD) CI)
(perpend_bis_AD_int_BI : M = intersection (perpendicular_bisector AD) BI)
(incenter_property : I = incenter A B C)
(tangency_point_D : D = incircle_tangent_point BC I)

-- Theorem statement in Lean
theorem points_A_N_I_M_concyclic
  (ABC : Type*)
  [h : TriangleIncenter ABC] :
  cyclic_quadrilateral h.A h.N h.I h.M :=
sorry

end points_A_N_I_M_concyclic_l24_24101


namespace tan_pi_by_eight_product_l24_24435

theorem tan_pi_by_eight_product :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = -real.sqrt 2 := by 
sorry

end tan_pi_by_eight_product_l24_24435


namespace quadrilateral_is_parallelogram_l24_24411

variables {A B C D O : Point}
variable [Quadrilateral ABCD]
variable (O : Point)
variable [InteriorPoint O ABCD]

-- Condition 1: O is a fixed interior point
def is_interior_point (O : Point) (ABCD : Quadrilateral) : Prop :=
  ∃ (a b c d : Point), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
  InteriorPoint O ABCD

-- Condition 2: Any line passing through O divides the perimeter into two equal lengths
def divides_perimeter_equal_lengths (O : Point) (ABCD : Quadrilateral) : Prop :=
  ∀ (EF : Line), (EF.contains O) → divides_perimeter EF ABCD

-- Question: Prove that ABCD is a parallelogram
theorem quadrilateral_is_parallelogram
  (h1 : is_interior_point O ABCD)
  (h2 : divides_perimeter_equal_lengths O ABCD) :
  IsParallelogram ABCD :=
sorry

end quadrilateral_is_parallelogram_l24_24411


namespace family_of_four_children_includes_one_boy_one_girl_l24_24836

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - ((1/2)^4 + (1/2)^4)

theorem family_of_four_children_includes_one_boy_one_girl :
  probability_at_least_one_boy_and_one_girl = 7 / 8 :=
by
  sorry

end family_of_four_children_includes_one_boy_one_girl_l24_24836


namespace tan_pi_by_eight_product_l24_24434

theorem tan_pi_by_eight_product :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = -real.sqrt 2 := by 
sorry

end tan_pi_by_eight_product_l24_24434


namespace minimum_value_is_25_div_6_l24_24235

noncomputable def minimum_value_of_expression (x y a b : ℝ) : ℝ :=
  if h : (3 * x - y - 6 ≤ 0) ∧ (x - y + 2 ≥ 0) ∧ (x ≥ 0) ∧ (y ≥ 0) ∧ (a > 0) ∧ (b > 0) ∧ (∀ x y, (3 * x - y - 6 ≤ 0) → (x - y + 2 ≥ 0) → (x ≥ 0) → (y ≥ 0) → (ax + by ≤ 12)) then
    (2 / a) + (3 / b)
  else
    0

theorem minimum_value_is_25_div_6 {x y a b : ℝ} 
  (h1 : 3 * x - y - 6 ≤ 0) 
  (h2 : x - y + 2 ≥ 0) 
  (h3 : x ≥ 0) 
  (h4 : y ≥ 0) 
  (h5 : a > 0) 
  (h6 : b > 0) 
  (h7 : ∀ x y, (3 * x - y - 6 ≤ 0) → (x - y + 2 ≥ 0) → (x ≥ 0) → (y ≥ 0) → (a * x + b * y ≤ 12))
  : minimum_value_of_expression x y a b = 25 / 6 := by
  sorry

end minimum_value_is_25_div_6_l24_24235


namespace max_sum_of_squares_of_real_roots_l24_24519

theorem max_sum_of_squares_of_real_roots :
  ∀ (k : ℝ) (x₁ x₂ : ℝ), 
  (x₁ = (k - 2 + real.sqrt ((k - 2) ^ 2 - 4 * (k ^ 2 + 3 * k + 5))) / 2 ∧ 
  x₂ = (k - 2 - real.sqrt ((k - 2) ^ 2 - 4 * (k ^ 2 + 3 * k + 5))) / 2) ∧ 
  (3 * k^2 + 16 * k + 16 ≤ 0) → 
  x₁^2 + x₂^2 ≤ 18 :=
begin
  sorry
end

end max_sum_of_squares_of_real_roots_l24_24519


namespace flagpole_break_distance_correct_l24_24383

-- Define the height of the flagpole
def flagpole_height : ℝ := 12

-- Define the height above the ground where the tip is dangling after breaking
def tip_height_above_ground : ℝ := 2

-- Define the height of the standing part of the flagpole
def standing_height := flagpole_height - tip_height_above_ground

-- Define the distance from the base where the flagpole broke
def break_distance : ℝ := 2 * real.sqrt(11)

-- Prove that the break distance is correct given the conditions
theorem flagpole_break_distance_correct :
  ∀ (flagpole_height tip_height_above_ground : ℝ),
    flagpole_height = 12 →
    tip_height_above_ground = 2 →
    real.sqrt (flagpole_height^2 - (flagpole_height - tip_height_above_ground)^2) = break_distance :=
by
  intros
  rw [← real.sqrt_mul_self (flagpole_height - tip_height_above_ground)]
  rw [add_comm, real.sqrt_inj]
  linarith
  sorry

end flagpole_break_distance_correct_l24_24383


namespace discount_percentage_l24_24146

-- Define given conditions
def price_per_kg : ℝ := 5
def total_kg : ℝ := 10
def total_price_after_discount : ℝ := 30

-- The problem statement to prove
theorem discount_percentage :
  let original_total_price := total_kg * price_per_kg in
  let discount_amount := original_total_price - total_price_after_discount in
  let discount_per_kg := discount_amount / total_kg in
  let discount_percent := (discount_per_kg / price_per_kg) * 100 in
  discount_percent = 40 := 
by 
  sorry

end discount_percentage_l24_24146


namespace abundant_numbers_less_than_50_count_l24_24403

-- Define an abundant number
def is_abundant (n : ℕ) : Prop := 
  (∑ i in (finset.filter (λ d, d < n ∧ n % d = 0) (finset.range n)).val, i) > n

-- Define the range we are interested in
def abundant_numbers_less_than (m : ℕ) : finset ℕ := 
  finset.filter is_abundant (finset.range m)

-- State the total number of abundant numbers less than 50
theorem abundant_numbers_less_than_50_count : (abundant_numbers_less_than 50).card = 9 :=
by 
  sorry

end abundant_numbers_less_than_50_count_l24_24403


namespace solution_count_l24_24518

/-- There are 91 solutions to the equation x + y + z = 15 given that x, y, z are all positive integers. -/
theorem solution_count (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 15) : 
  ∃! n, n = 91 := 
by sorry

end solution_count_l24_24518


namespace arithmetic_seq_sum_l24_24933

/-- Given an arithmetic sequence {a_n} such that a_5 + a_6 + a_7 = 15,
prove that the sum of the first 11 terms of the sequence S_11 is 55. -/
theorem arithmetic_seq_sum (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h₁ : a 5 + a 6 + a 7 = 15)
  (h₂ : ∀ n, S n = n * (a 1 + a n) / 2) :
  S 11 = 55 :=
sorry

end arithmetic_seq_sum_l24_24933


namespace max_f_in_interval_l24_24653

def f (x : ℝ) : ℝ :=
  if x.is_irrational then x
  else
    let p := x.numerator
    let q := x.denominator in
    if (p, q) = (p.gcd q, 1) ∧ 0 < p < q then (p+1)/q else 0

theorem max_f_in_interval : ∀ x, 
  (7/8 < x) ∧ (x < 8/9) → 
  (∀ x, f x ≤ 16/17) ∧ (∃ x, f x = 16/17) := 
begin
  sorry
end

end max_f_in_interval_l24_24653


namespace ellipse_properties_l24_24287

theorem ellipse_properties :
  let a_sq := 4
  let b_sq := 2
  let a := Real.sqrt a_sq  -- semi-major axis
  let b := Real.sqrt b_sq  -- semi-minor axis
  let major_axis := 2 * a
  let minor_axis := 2 * b
  let c := Real.sqrt (a_sq - b_sq)  -- semi-focal distance
  let focal_distance := 2 * c
  let e := c / a  -- eccentricity in an ellipse
  in (major_axis = 4) ∧ (focal_distance = 2 * Real.sqrt 2) ∧ (e = Real.sqrt 2 / 2) := 
by {
  sorry
}

end ellipse_properties_l24_24287


namespace determine_h_f_l24_24826

variable (e f g h : ℕ)
variable (u v : ℕ)
variable (h_e : e = u^4)
variable (h_f : f = u^5)
variable (h_g : g = v^2)
variable (h_h : h = v^3)
variable (h_cond1 : e^5 = f^4)
variable (h_cond2 : g^3 = h^2)
variable (h_cond3 : g - e = 31)

theorem determine_h_f : h - f = 971 := by
  sorry

end determine_h_f_l24_24826


namespace diamond_3_7_l24_24490

def star (a b : ℕ) : ℕ := a^2 + 2*a*b + b^2
def diamond (a b : ℕ) : ℕ := star a b - a * b

theorem diamond_3_7 : diamond 3 7 = 79 :=
by 
  sorry

end diamond_3_7_l24_24490


namespace recommended_student_for_competition_l24_24340

noncomputable def A_scores := [68, 80, 78, 92, 81, 77, 84, 83, 79]
noncomputable def B_scores := [86, 80, 75, 83, 75, 77, 79, 80, 80, 85]

def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

noncomputable def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort(· ≤ ·)
  if sorted.length % 2 = 0 then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

def variance (l : List ℕ) (mean : ℕ) : ℕ :=
  let sq_diff := l.map (λ x => (x - mean) * (x - mean))
  sq_diff.sum / l.length

theorem recommended_student_for_competition :
  mean A_scores = 80 ∧ median A_scores = 79 ∧
  mean B_scores = 80 ∧ median B_scores = 80 ∧ variance B_scores 80 < 33 ∧
  mode A_scores = 78 ∧ mode B_scores = 80 →
  "B should be selected" = "B should be selected" :=
by 
  sorry

end recommended_student_for_competition_l24_24340


namespace max_m_inequality_l24_24208

theorem max_m_inequality (n : ℕ) (hn : n > 1) (x : Fin n → ℝ)
  (hx : ∑ i, (x i) ^ 2 = 1) :
  (∃ m, m = min (λ (i j : Fin n), if i ≠ j then (abs (x i - x j)) else 1) ∧ 
         m ≤ Real.sqrt (12 / (n * (n - 1) * (n + 1)))) :=
sorry

end max_m_inequality_l24_24208


namespace sphere_surface_area_l24_24520

noncomputable def surface_area_of_sphere (r : ℝ) : ℝ :=
  4 * Real.pi * r^2

theorem sphere_surface_area (A B C D : ℝ → ℝ) (O : ℝ → ℝ) :
  (AB_perpendicular_plane_BCD : (∀ (B C D : ℝ → ℝ), plane_perpendicular_to (O A) 2)) →
  AB = 2 →
  angle B C D = Real.pi / 3 →
  dist C B = 1 →
  dist C D = 1 →
  surface_area_of_sphere (radius_of_sphere A B C D) = 8 * Real.pi :=
by
  sorry

end sphere_surface_area_l24_24520


namespace exists_acute_angle_triangle_l24_24676

theorem exists_acute_angle_triangle (a b c d e : ℝ) (h_triangle : ∀ (x y z : ℝ), x + y > z ∧ x + z > y ∧ y + z > x) :
  ∃ (x y z : ℝ), x^2 + y^2 > z^2 :=
begin
  sorry
end

end exists_acute_angle_triangle_l24_24676


namespace find_k_l24_24903

theorem find_k (x y z k : ℝ) (h1 : 5 / (x + y) = k / (x + z)) (h2 : k / (x + z) = 9 / (z - y)) : k = 14 :=
by
  sorry

end find_k_l24_24903


namespace edward_received_10_l24_24030

theorem edward_received_10 
  (initial_balance : ℤ) 
  (amount_spent : ℤ) 
  (final_balance : ℤ) 
  (money_received : ℤ)
  (h1 : initial_balance = 14) 
  (h2 : amount_spent = 17) 
  (h3 : final_balance = 7) 
  (h4 : initial_balance - amount_spent + money_received = final_balance) : 
  money_received = 10 :=
begin
  sorry
end

end edward_received_10_l24_24030


namespace increase_by_8_percent_l24_24155

def production_cost_change (x : ℝ) : Prop :=
if x < 0 then "decrease" else "increase"

def change_percentage := 0.08

theorem increase_by_8_percent (h : production_cost_change (-0.1) = "decrease") : 
  production_cost_change change_percentage = "increase" :=
sorry

end increase_by_8_percent_l24_24155


namespace max_percent_error_l24_24203

theorem max_percent_error (d : ℝ) (error_percentage : ℝ) (true_diameter : ℝ) (true_area : ℝ) : 
  true_diameter = 30 → error_percentage = 0.3 → true_area = (real.pi * (true_diameter / 2)^2) →
  ∃ max_percent_error : ℝ, 
  max_percent_error = 69 :=
by
  intros h1 h2 h3
  let d_min := true_diameter - true_diameter * error_percentage
  let d_max := true_diameter + true_diameter * error_percentage
  let area_min := real.pi * (d_min / 2)^2
  let area_max := real.pi * (d_max / 2)^2
  let percent_error_min := ((true_area - area_min) / true_area) * 100
  let percent_error_max := ((area_max - true_area) / true_area) * 100
  use percent_error_max
  sorry -- Placeholder for completing the proof

end max_percent_error_l24_24203


namespace runner_advantage_l24_24179

theorem runner_advantage (x y z : ℝ) (hx_y: y - x = 0.1) (hy_z: z - y = 0.11111111111111111) :
  z - x = 0.21111111111111111 :=
by
  sorry

end runner_advantage_l24_24179


namespace tangent_product_l24_24438

theorem tangent_product (n θ : ℝ) 
  (h1 : ∀ n θ, tan (n * θ) = (sin (n * θ)) / (cos (n * θ)))
  (h2 : tan 8 θ = (8 * tan θ - 56 * (tan θ) ^ 3 + 56 * (tan θ) ^ 5 - 8 * (tan θ) ^ 7) / 
                  (1 - 28 * (tan θ) ^ 2 + 70 * (tan θ) ^ 4 - 28 * (tan θ) ^ 6))
  (h3 : tan (8 * (π / 8)) = 0)
  (h4 : tan (8 * (3 * π / 8)) = 0)
  (h5 : tan (8 * (5 * π / 8)) = 0) :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = 2 * sqrt 2 :=
sorry

end tangent_product_l24_24438


namespace trader_profit_percent_l24_24393

-- Definitions based on the conditions
variables (P : ℝ) -- Original price of the car
def discount_price := 0.95 * P
def taxes := 0.03 * P
def maintenance := 0.02 * P
def total_cost := discount_price + taxes + maintenance 
def selling_price := 0.95 * P * 1.60
def profit := selling_price - total_cost

-- Theorem
theorem trader_profit_percent : (profit P / P) * 100 = 52 :=
by
  sorry

end trader_profit_percent_l24_24393


namespace distance_from_point_to_directrix_l24_24989

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24989


namespace pentagon_area_l24_24206

noncomputable def area_pentagon (BC BK DC DK AB DE : ℝ) (angle_ABC angle_CDE : ℝ) (BD : ℝ) : ℝ :=
  if (BC = BK ∧ DC = DK ∧ AB = BC ∧ DE = DC ∧ angle_ABC = (2 * Real.pi) / 3 ∧ angle_CDE = Real.pi / 3 ∧ BD = 2) then sqrt 3
  else 0

theorem pentagon_area
  (BC BK DC DK AB DE : ℝ) (angle_ABC angle_CDE : ℝ) (BD : ℝ)
  (h1 : BC = BK) (h2 : DC = DK) (h3 : AB = BC) (h4 : DE = DC) (h5 : angle_ABC = (2 * Real.pi) / 3) (h6 : angle_CDE = Real.pi / 3) (h7 : BD = 2) :
  area_pentagon BC BK DC DK AB DE angle_ABC angle_CDE BD = sqrt 3 :=
by 
  -- the proof would go here 
  sorry

end pentagon_area_l24_24206


namespace ellipse_standard_eq_max_area_triangle_l24_24122

-- Given conditions
variable (a b : ℝ) (x y : ℝ)
variable (A : ℝ × ℝ := (2, 0))
variable (P Q : ℝ × ℝ)
variable (e : ℝ := 3/sqrt 4 / 2)
variable (h : ℝ := 1)
variable (transverse : ∀a, a ∈ [h]) → point a :
format (format "transverse (mrProof a ) ") = simp only with smul_eq_mul

-- Condition definitions in Lean 4
def ellipse_eq (x y a b : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1

def point_on_ellipse (x y : ℝ) (a b : ℝ) : Prop := 
  ellipse_eq x y a b ∧ x = 1 ∧ y = sqrt 3 / 2

def is_eccentricity (a b : ℝ) (e : ℝ) : Prop := 
  let c := sqrt (a^2 - b^2) in 
  e = c / a

-- Making sure the intersection is valid
def valid_intersection (A P Q : ℝ × ℝ) : Prop := 
  let AP := P - A in 
  let AQ := Q - A in 
  inner AP AQ = 0

-- Problem Statements to Prove
theorem ellipse_standard_eq : 
  ∀ a b : ℝ, 
  (a > b) → 
  (∀ x y : ℝ, point_on_ellipse a b x y) → 
  ellipse_eq 2 0 a b :=
sorry

theorem max_area_triangle : 
  ∀ (A P Q : ℝ × ℝ), 
  valid_intersection A P Q →
  area_triangle O P Q = 24 / 25 :=
sorry

end ellipse_standard_eq_max_area_triangle_l24_24122


namespace total_nails_to_cut_l24_24865

theorem total_nails_to_cut :
  let dogs := 4 
  let legs_per_dog := 4
  let nails_per_dog_leg := 4
  let parrots := 8
  let legs_per_parrot := 2
  let nails_per_parrot_leg := 3
  let extra_nail := 1
  let total_dog_nails := dogs * legs_per_dog * nails_per_dog_leg
  let total_parrot_nails := (parrots * legs_per_parrot * nails_per_parrot_leg) + extra_nail
  total_dog_nails + total_parrot_nails = 113 :=
sorry

end total_nails_to_cut_l24_24865


namespace minimum_bottles_needed_l24_24812

theorem minimum_bottles_needed 
  (small : ℕ := 40) (medium : ℕ := 120) (large : ℕ := 480)
  (num_small num_medium : ℕ) :
  (num_medium * medium + num_small * small = large) →
  (num_medium + num_small = 6) :=
begin
  intros h,
  -- Since the proof is not required, we end with sorry
  sorry
end

end minimum_bottles_needed_l24_24812


namespace inequality_solution_l24_24330

theorem inequality_solution (x : ℝ) : 5 * x > 4 * x + 2 → x > 2 :=
by
  sorry

end inequality_solution_l24_24330


namespace sin_cos_angle_addition_l24_24746

theorem sin_cos_angle_addition :
  sin (20 * Real.pi / 180) * cos (40 * Real.pi / 180) +
  cos (20 * Real.pi / 180) * sin (40 * Real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end sin_cos_angle_addition_l24_24746


namespace toluene_production_l24_24048

def molar_mass_benzene : ℝ := 78.11 -- The molar mass of benzene in g/mol
def benzene_mass : ℝ := 156 -- The mass of benzene in grams
def methane_moles : ℝ := 2 -- The moles of methane

-- Define the balanced chemical reaction
def balanced_reaction (benzene methanol toluene hydrogen : ℝ) : Prop :=
  benzene + methanol = toluene + hydrogen

-- The main theorem statement
theorem toluene_production (h1 : balanced_reaction benzene_mass methane_moles 1 1)
  (h2 : benzene_mass / molar_mass_benzene = 2) :
  ∃ toluene_moles : ℝ, toluene_moles = 2 :=
by
  sorry

end toluene_production_l24_24048


namespace count_integer_solutions_l24_24912

theorem count_integer_solutions : 
  {x : ℤ | x^2 < 8 * x + 1}.to_finset.card = 8 :=
by
  sorry

end count_integer_solutions_l24_24912


namespace factorial_fraction_value_l24_24850

theorem factorial_fraction_value :
  (15.factorial / (6.factorial * 9.factorial) = 5005) :=
by
  sorry

end factorial_fraction_value_l24_24850


namespace product_of_t_for_quadratic_factorization_l24_24057

theorem product_of_t_for_quadratic_factorization :
  (∏ t in ({a + b | a * b = 6 ∧ a ≠ b} : set Int).to_finset) = -1225 := 
sorry

end product_of_t_for_quadratic_factorization_l24_24057


namespace geometric_identity_l24_24928

-- Given conditions from step a)
def hyperbola (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def P := (-real.sqrt 3, 0 : ℝ)
noncomputable def G := (-real.sqrt 3 / 3, 0 : ℝ)

-- Theorem we want to prove
theorem geometric_identity 
  (a b : ℝ)
  (h1 : hyperbola 1 1 a b)
  (E : ∀ x y : ℝ, ellipse x y 3 (real.sqrt 3/ sqrt 2)) 
  (h2 : 2*(1 / 3 + 1 / (real.sqrt 3 / sqrt 2)^2) = 2) :
  true :=
by
  sorry

end geometric_identity_l24_24928


namespace distance_from_point_to_directrix_l24_24955

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24955


namespace painting_problem_l24_24066

theorem painting_problem (n₁ n₂ : ℕ) (t₁ t₂ k : ℕ) (h₁ : n₁ * t₁ = k) (h₂ : n₂ * t₂ = k) (n₁_val : n₁ = 4) (t₁_val : t₁ = 12) (n₂_val : n₂ = 6) : t₂ = 8 :=
by
  have h_k : k = 4 * 12 := by rw [n₁_val, t₁_val]; exact h₁
  have h₂' : 6 * t₂ = 48 := by rw [←h_k, ←n₂_val]; exact h₂
  have t₂_val : t₂ = 48 / 6 := by exact (nat.eq_of_mul_eq_mul_right (by norm_num) h₂')
  exact t₂_val

end painting_problem_l24_24066


namespace find_g3_l24_24309

variable {g : ℝ → ℝ}

-- Defining the condition from the problem
def g_condition (x : ℝ) (h : x ≠ 0) : g x - 3 * g (1 / x) = 3^x + x^2 := sorry

-- The main statement to prove
theorem find_g3 : g 3 = - (3 * 3^(1/3) + 1/3 + 36) / 8 := sorry

end find_g3_l24_24309


namespace line_integral_value_l24_24877

-- Define the vector field in spherical coordinates
def vector_field (r θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  (exp r * sin θ, 3 * θ ^ 2 * sin φ, r * φ * θ)

-- Define the path L in parametric form
def path (θ : ℝ) : (ℝ × ℝ × ℝ) :=
  (1, θ, π / 2)

-- Define the line integral of the vector field along the path
noncomputable def line_integral : ℝ :=
  ∫ θ in 0..(π / 2), 3 * θ ^ 2

theorem line_integral_value : line_integral = (π ^ 3 / 8) :=
by
  sorry

end line_integral_value_l24_24877


namespace sequence_50th_term_is_44_l24_24687

/-- Defines the rules for the sequence. -/
def sequence_next (n : ℕ) : ℕ :=
  if n < 15 then n * 7
  else if n % 2 = 0 ∧ 15 ≤ n ∧ n ≤ 35 then n + 10
  else if n % 2 = 1 ∧ n > 35 then n - 7
  else 0 -- This case should never happen based on the problem's conditions.

/-- Defines the sequence recursively according to the rules. -/
def sequence : ℕ → ℕ
| 0 => 76
| (n + 1) => sequence_next (sequence n)

/-- Prove the 50th term of the sequence starting from 76 is 44. -/
theorem sequence_50th_term_is_44 : sequence 49 = 44 :=
  sorry

end sequence_50th_term_is_44_l24_24687


namespace car_price_difference_l24_24259

variable (original_paid old_car_proceeds : ℝ)
variable (new_car_price additional_amount : ℝ)

theorem car_price_difference :
  old_car_proceeds = new_car_price - additional_amount →
  old_car_proceeds = 0.8 * original_paid →
  additional_amount = 4000 →
  new_car_price = 30000 →
  (original_paid - new_car_price) = 2500 :=
by
  intro h1 h2 h3 h4
  sorry

end car_price_difference_l24_24259


namespace max_value_product_l24_24513

-- Definition of the given ellipse and its parameters
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 25) + (y^2 / 9) = 1

-- The coordinates of the foci of the ellipse
def F1 (x y : ℝ) : Prop := (x = -4) ∧ (y = 0)  -- This is an approximation for the focus
def F2 (x y : ℝ) : Prop := (x = 4) ∧ (y = 0)   -- This is an approximation for the focus

-- The distance between a point P and a focus
def dist (P F : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

-- The maximum value to be proven
theorem max_value_product (P : ℝ × ℝ) (on_ellipse : ellipse P.1 P.2) :
  (∃ (F1 F2 : ℝ × ℝ), dist P F1 * dist P F2 ≤ 25) :=
sorry

end max_value_product_l24_24513


namespace distance_from_A_to_directrix_of_C_l24_24971

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24971


namespace exists_sum_of_squares_form_l24_24447

theorem exists_sum_of_squares_form (n : ℕ) (h : n % 25 = 9) :
  ∃ (a b c : ℕ), n = (a * (a + 1)) / 2 + (b * (b + 1)) / 2 + (c * (c + 1)) / 2 := 
by 
  sorry

end exists_sum_of_squares_form_l24_24447


namespace distance_from_A_to_directrix_of_C_l24_24964

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24964


namespace market_cost_l24_24665

theorem market_cost (peach_pies apple_pies blueberry_pies : ℕ) (fruit_per_pie : ℕ) 
  (price_per_pound_apple price_per_pound_blueberry price_per_pound_peach : ℕ) :
  peach_pies = 5 ∧
  apple_pies = 4 ∧
  blueberry_pies = 3 ∧
  fruit_per_pie = 3 ∧
  price_per_pound_apple = 1 ∧
  price_per_pound_blueberry = 1 ∧
  price_per_pound_peach = 2 →
  let total_peaches := peach_pies * fruit_per_pie in
  let total_apples := apple_pies * fruit_per_pie in
  let total_blueberries := blueberry_pies * fruit_per_pie in
  let cost_apples := total_apples * price_per_pound_apple in
  let cost_blueberries := total_blueberries * price_per_pound_blueberry in
  let cost_peaches := total_peaches * price_per_pound_peach in
  (cost_apples + cost_blueberries + cost_peaches = 51) :=
by
  intros
  sorry

end market_cost_l24_24665


namespace largest_possible_difference_l24_24404

theorem largest_possible_difference (A B : ℤ) 
  (hA1 : 45000 ≤ A) (hA2 : A ≤ 55000)
  (hB1 : 54545 ≤ B) (hB2 : B ≤ 66667) :
  (max B - min A).round = 22000 :=
by
  sorry

end largest_possible_difference_l24_24404


namespace insurance_coverage_is_80_percent_l24_24200

-- Conditions
def num_appointments : ℕ := 3
def cost_per_appointment : ℝ := 400
def pet_insurance_cost : ℝ := 100
def total_paid : ℝ := 660
def num_subsequent_appointments : ℕ := 2

-- Total cost for the vet appointments
def total_vet_cost : ℝ := num_appointments * cost_per_appointment

-- Amount John paid for the vet appointments
def amount_paid_for_vet : ℝ := total_paid - pet_insurance_cost

-- Cost covered by insurance
def insurance_covered : ℝ := total_vet_cost - amount_paid_for_vet

-- Cost for the last two appointments
def subsequent_appointments_cost : ℝ := num_subsequent_appointments * cost_per_appointment

-- The percentage covered by insurance for the last two appointments
def insurance_coverage_percentage : ℝ := (insurance_covered / subsequent_appointments_cost) * 100

-- Theorem to prove
theorem insurance_coverage_is_80_percent : insurance_coverage_percentage = 80 := 
by
  sorry

end insurance_coverage_is_80_percent_l24_24200


namespace distance_from_point_to_directrix_l24_24951

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24951


namespace tan_product_l24_24445

theorem tan_product (t : ℝ) (h1 : 1 - 7 * t^2 + 7 * t^4 - t^6 = 0)
  (h2 : t = tan (Real.pi / 8) ∨ t = tan (3 * Real.pi / 8) ∨ t = tan (5 * Real.pi / 8)) :
  tan (Real.pi / 8) * tan (3 * Real.pi / 8) * tan (5 * Real.pi / 8) = 1 := 
sorry

end tan_product_l24_24445


namespace knights_count_l24_24672

-- Definitions for the problem
def Knight (x : Nat) : Prop := sorry
def Liar (x : Nat) : Prop := sorry

-- Condition: 15 natives arranged in a circle
def isCircle (n : Nat) : Prop := n = 15

-- Condition: Knights always tell the truth, and liars always lie
def alwaysTellTruth (x : Nat) : Prop :=
  Knight x → (Knight ((x + 7) % 15) ∧ Liar ((x + 8) % 15)) ∨ (Liar ((x + 7) % 15) ∧ Knight ((x + 8) % 15))
def alwaysLie (x : Nat) : Prop :=
  Liar x → ¬((Knight ((x + 7) % 15) ∧ Liar ((x + 8) % 15)) ∨ (Liar ((x + 7) % 15) ∧ Knight ((x + 8) % 15)))

-- Main theorem to be proved: There are 10 knights
theorem knights_count (n : Nat) (config: ∀ x, Knight x ∨ Liar x) : isCircle n → (∑ i in range n, @ite Prop (Knight i) 1 0 = 10) :=
by
  { sorry }

end knights_count_l24_24672


namespace fraction_arithmetic_l24_24555

theorem fraction_arithmetic : ( (4 / 5 - 1 / 10) / (2 / 5) ) = 7 / 4 :=
  sorry

end fraction_arithmetic_l24_24555


namespace average_speed_calc_l24_24794

def first_hour_speed := 90 -- km/h
def second_hour_speed := 75 -- km/h
def third_hour_speed := 110 -- km/h
def uphill_speed := 65 -- km/h
def uphill_duration := 2 -- hours
def tailwind_speed := 95 -- km/h
def tailwind_duration := 45 / 60 -- hours (0.75)
def fuel_efficiency_speed := 80 -- km/h
def fuel_efficiency_duration := 1.5 -- hours

def total_distance := (first_hour_speed * 1) +
                     (second_hour_speed * 1) +
                     (third_hour_speed * 1) +
                     (uphill_speed * uphill_duration) +
                     (tailwind_speed * tailwind_duration) +
                     (fuel_efficiency_speed * fuel_efficiency_duration)

def total_time := 1 + 1 + 1 + uphill_duration + tailwind_duration + fuel_efficiency_duration

theorem average_speed_calc : total_distance / total_time = 82.24 :=
by
  let d := total_distance
  let t := total_time
  have h : d = 596.25 := by sorry
  have ht : t = 7.25 := by sorry
  rw [h, ht]
  norm_num
  sorry

end average_speed_calc_l24_24794


namespace find_multiple_l24_24298

variable (P W : ℕ)
variable (h1 : ∀ P W, P * 16 * (W / (P * 16)) = W)
variable (h2 : ∀ P W m, (m * P) * 4 * (W / (16 * P)) = W / 2)

theorem find_multiple (P W : ℕ) (h1 : ∀ P W, P * 16 * (W / (P * 16)) = W)
                      (h2 : ∀ P W m, (m * P) * 4 * (W / (16 * P)) = W / 2) : m = 2 :=
by
  sorry

end find_multiple_l24_24298


namespace train_length_is_50_04_l24_24369

-- Conditions
def train_length (L : ℝ) : Prop :=
  Let relative_speed := (46 - 36) * (5/18) -- km/hr to m/s conversion
  ∧ Let Distance := relative_speed * 36 -- calculate total distance in meters
  ∧ (2 * L = Distance)

-- Proof Problem:
theorem train_length_is_50_04 : ∃ (L : ℝ), train_length L ∧ L = 50.04 :=
sorry

end train_length_is_50_04_l24_24369


namespace distance_from_A_to_directrix_of_C_l24_24968

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24968


namespace distance_from_A_to_directrix_of_C_l24_24974

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24974


namespace log_equation_solution_l24_24111

-- Define the primary condition
variable (x : ℝ)
hypothesis (h1 : x < 1)
hypothesis (h2 : (log 10 x)^2 - log 10 (x^4) = 72)

-- The theorem stating the problem in Lean
theorem log_equation_solution : (log 10 x)^4 - log 10 (x^4) = 1320 :=
by {
    assume h1,
    assume h2,
    sorry
}

end log_equation_solution_l24_24111


namespace problem_statement_l24_24567

theorem problem_statement (x y : ℤ) (h : |x + 2 * y| + (y - 3)^2 = 0) : x^y = -216 := by
  -- Proof to be added
  sorry

end problem_statement_l24_24567


namespace max_mass_difference_l24_24813

theorem max_mass_difference :
  let m₁ := (24.9, 25.1),
      m₂ := (24.8, 25.2),
      m₃ := (24.7, 25.3) in
  ∃ a b ∈ {m₁, m₂, m₃}, max a.2 b.2 - min a.1 b.1 = 0.6 :=
by
  sorry

end max_mass_difference_l24_24813


namespace length_of_AB_l24_24674

open_locale big_operators

variables {V : Type*} [add_comm_group V] [module ℝ V]
variables {a b : V}

-- Define the centroid condition and problem statement
def is_centroid (G A B C : V) : Prop :=
  G = (A + B + C) / 3

-- The goal statement to be proved in Lean
theorem length_of_AB {G A B C : V} (hG : is_centroid G A B C) (hGB : G - B = a) (hGC : G - C = b) :
  A - B = 2 * a + b :=
sorry

end length_of_AB_l24_24674


namespace bisection_by_homothety_l24_24067

variable {ABC : Type} [geometry_triple ABC]
variable {O : Point} {Ω : Circle}
variable {A B C M N D : Point}
variable {AB AC BC : Line}

-- Conditions
axiom center_of_circumcircle : is_center O Ω
axiom is_midpoint_BC : is_midpoint D B C
axiom is_midpoint_AB : is_midpoint M A B
axiom is_midpoint_AC : is_midpoint N A C
axiom perpendicular_from_O : ∀ (l : Line), is_angle_bisector l ∧ (is_interior_angle A l ∨ is_exterior_angle A l) → perpendicular_from O l

-- The statement to prove
theorem bisection_by_homothety :
  let D' := homothety(O, 2) D in
  let M' := homothety(O, 2) M in
  bisects (line_through D' M') (segment M N) :=
  sorry

end bisection_by_homothety_l24_24067


namespace probability_of_2_red_1_black_l24_24750

theorem probability_of_2_red_1_black :
  let P_red := 4 / 7
  let P_black := 3 / 7 
  let prob_RRB := P_red * P_red * P_black 
  let prob_RBR := P_red * P_black * P_red 
  let prob_BRR := P_black * P_red * P_red 
  let total_prob := 3 * prob_RRB
  total_prob = 144 / 343 :=
by
  sorry

end probability_of_2_red_1_black_l24_24750


namespace sum_S_ijk_l24_24227

open Nat

def S : Set (ℕ × ℕ × ℕ) := {t | ∃ (i j k : ℕ), (i, j, k) = t ∧ i > 0 ∧ j > 0 ∧ k > 0 ∧ i + j + k = 17}

noncomputable def sumS : ℕ := ∑ t in S, Prod.map (λ a b, a * b) t

theorem sum_S_ijk : sumS = 11628 := 
sorry

end sum_S_ijk_l24_24227


namespace thales_circle_power_of_point_l24_24648

/-- Let M and N be points on the Thales' circle of segment AB, distinct from A and B. 
Let C be the midpoint of segment NA, 
and D be the midpoint of segment NB. 
The circle is intersected at point E a second time by the line MC, 
and at point F by the line MD. 
Given AB = 2 units, prove that MC * CE + MD * DF = 1. --/
theorem thales_circle_power_of_point
  (A B M N C D E F : Type)
  [metric_space A]
  [metric_space B]
  [metric_space M]
  [metric_space N]
  [metric_space C]
  [metric_space D]
  [metric_space E]
  [metric_space F]
  (h_circle: M ∈ circle A B N)
  (h_thales: ∠A N B = 90)
  (h_AB: dist A B = 2)
  (h_midpoint_C: midpoint A N = C)
  (h_midpoint_D: midpoint B N = D)
  (h_intersect_C: second_intersection MC circle = E)
  (h_intersect_D: second_intersection MD circle = F) :
  dist M C * dist C E + dist M D * dist D F = 1 :=
sorry

end thales_circle_power_of_point_l24_24648


namespace monotonic_increasing_interval_sinx2x_minus_pi_over_3_l24_24715


theorem monotonic_increasing_interval_sinx2x_minus_pi_over_3 :
  monotonic_increasing_interval (λ x : ℝ, sin (2 * x - (π / 3)) - sin (2 * x)) =
    set.Icc (π / 12) (7 * π / 12) :=
sorry

end monotonic_increasing_interval_sinx2x_minus_pi_over_3_l24_24715


namespace tangent_lines_through_P_l24_24714

-- Define Circle, and the conditions of the problem
variable (k : Circle)
variable (A B C D P E F : Point)
variable (tangent_A tangent_B tangent_C tangent_D : Line)
variable (intersection_EF : Line)

-- Assuming the given conditions
variable (chord1 : is_chord_of k A B)
variable (chord2 : is_chord_of k C D)
variable (intersect_P : intersection_point (chord1, chord2) = P ∧ P ≠ center(k))
variable (tangents_intersect_E : intersection_point (tangents k A B) = E)
variable (tangents_intersect_F : intersection_point (tangents k C D) = F)

-- The construct for the proof statement
theorem tangent_lines_through_P : 
  ∀ (Q : Point), (Q ∈ intersection_EF) → (∃ T1 T2 : Point, is_tangent_to_circle_at k Q T1 T2 ∧ line_through T1 T2 P) :=
by
  sorry

end tangent_lines_through_P_l24_24714


namespace integer_triple_solution_l24_24896

theorem integer_triple_solution (x y z : ℤ) :
  x^2 * (y - z) + y^2 * (z - x) + z^2 * (x - y) = 2 → 
  ∃ k : ℤ, (x = k + 1 ∧ y = k ∧ z = k - 1) :=
begin
  sorry
end

end integer_triple_solution_l24_24896


namespace polynomial_not_divisible_l24_24374

noncomputable def is_root (p q : ℂ) (k : ℕ) : Bool := (f(ω^k) = 0 ∧ f(ω^(2*k)) = 0)

theorem polynomial_not_divisible (k : ℕ) : (k % 3 ≠ 0) ↔ ¬ is_root (λ x, x^(2*k) + 1 + (x + 1)^(2*k)) (ω) k := sorry

end polynomial_not_divisible_l24_24374


namespace geometric_sequence_common_ratio_l24_24926

variable (a_1 q : ℝ)
variable (a_pos : 0 < a_1) (q_pos : 0 < q)

theorem geometric_sequence_common_ratio :
  (a_1 * q ^ 4) ^ 2 = 2 * (a_1 * q ^ 2) * (a_1 * q ^ 8) → q = real.sqrt 2 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l24_24926


namespace set_membership_l24_24144

-- Defining the universal set U
def U := {1, 2, 3, 4, 5, 6, 7}

-- Defining sets M and N
def M := {3, 4, 5}
def N := {1, 3, 6}

-- Defining the complement function with respect to U
def complement (A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A}

-- Defining the target set {2, 7}
def target_set := {2, 7}

-- Stating the theorem
theorem set_membership : target_set = (complement M) ∩ (complement N) :=
by sorry

end set_membership_l24_24144


namespace distance_from_A_to_directrix_of_C_l24_24972

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24972


namespace subtract_fractions_correct_l24_24905

theorem subtract_fractions_correct :
  (3 / 8 + 5 / 12 - 1 / 6) = (5 / 8) := by
sorry

end subtract_fractions_correct_l24_24905


namespace exists_base_for_1994_no_base_for_1993_l24_24691

-- Problem 1: Existence of a base for 1994 with identical digits
theorem exists_base_for_1994 :
  ∃ b : ℕ, 1 < b ∧ b < 1993 ∧ (∃ a : ℕ, ∀ n : ℕ, 1994 = a * ((b ^ n - 1) / (b - 1)) ∧ a = 2) :=
sorry

-- Problem 2: Non-existence of a base for 1993 with identical digits
theorem no_base_for_1993 :
  ¬∃ b : ℕ, 1 < b ∧ b < 1992 ∧ (∃ a : ℕ, ∀ n : ℕ, 1993 = a * ((b ^ n - 1) / (b - 1))) :=
sorry

end exists_base_for_1994_no_base_for_1993_l24_24691


namespace distance_R_is_50sqrt6_l24_24460

-- Definitions from the conditions
def side_length : ℝ := 300
def distance_to_vertices (r : ℝ) := 
  ∀ (D E F X Y : ℝ), -- Points are real numbers for simplicity
  (X - D)^2 + (X - E)^2 + (X - F)^2 = 0 →  -- X is equidistant to D, E, F
  (Y - D)^2 + (Y - E)^2 + (Y - F)^2 = 0 →  -- Y is equidistant to D, E, F
  distance R = r                          -- the point R is equidistant to D, E, F, X, and Y

-- The theorem to prove
theorem distance_R_is_50sqrt6 (R : ℝ) :
  distance_to_vertices R 50 := 
sorry

end distance_R_is_50sqrt6_l24_24460


namespace salesman_afternoon_sales_l24_24809

noncomputable
def afternoon_sales (morning_afternoon_ratio : ℕ) (total_sales : ℕ) : ℕ :=
  let morning_sales := total_sales / (morning_afternoon_ratio + 1)
  let afternoon_sales := morning_afternoon_ratio * morning_sales
  afternoon_sales

theorem salesman_afternoon_sales
  (morning_afternoon_ratio : ℕ)
  (total_sales : ℕ)
  (h_ratio : morning_afternoon_ratio = 2)
  (h_total : total_sales = 360) :
  afternoon_sales morning_afternoon_ratio total_sales = 240 :=
by
  rw [h_ratio, h_total]
  dsimp [afternoon_sales]
  norm_num
  sorry

end salesman_afternoon_sales_l24_24809


namespace sequence_formula_l24_24919

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2^(n + 1)

theorem sequence_formula (n : ℕ) :
  n ≠ 0 → sequence n = 
    if n = 1 then 1 else 2^(n + 1) :=
by
  intro h
  unfold sequence
  rw if_neg h
  sorry

end sequence_formula_l24_24919


namespace quadrilateral_offset_l24_24465

theorem quadrilateral_offset (d A h₂ x : ℝ)
  (h_da: d = 40)
  (h_A: A = 400)
  (h_h2 : h₂ = 9)
  (h_area : A = 1/2 * d * (x + h₂)) : 
  x = 11 :=
by sorry

end quadrilateral_offset_l24_24465


namespace solution_set_inequality_l24_24740

theorem solution_set_inequality (x : ℝ) : (1 < x ∧ x < 3) ↔ (x^2 - 4*x + 3 < 0) :=
by sorry

end solution_set_inequality_l24_24740


namespace product_of_repeating_decimal_l24_24011

theorem product_of_repeating_decimal 
  (t : ℚ) 
  (h : t = 456 / 999) : 
  8 * t = 1216 / 333 :=
by
  sorry

end product_of_repeating_decimal_l24_24011


namespace total_market_cost_l24_24662

-- Defining the variables for the problem
def pounds_peaches : Nat := 5 * 3
def pounds_apples : Nat := 4 * 3
def pounds_blueberries : Nat := 3 * 3

def cost_per_pound_peach := 2
def cost_per_pound_apple := 1
def cost_per_pound_blueberry := 1

-- Defining the total costs
def cost_peaches : Nat := pounds_peaches * cost_per_pound_peach
def cost_apples : Nat := pounds_apples * cost_per_pound_apple
def cost_blueberries : Nat := pounds_blueberries * cost_per_pound_blueberry

-- Total cost
def total_cost : Nat := cost_peaches + cost_apples + cost_blueberries

-- Theorem to prove the total cost is $51.00
theorem total_market_cost : total_cost = 51 := by
  sorry

end total_market_cost_l24_24662


namespace problem_l24_24076

theorem problem (m a b : ℝ) (h₀ : 9 ^ m = 10) (h₁ : a = 10 ^ m - 11) (h₂ : b = 8 ^ m - 9) :
  a > 0 ∧ 0 > b := 
sorry

end problem_l24_24076


namespace find_C_find_S_l24_24582

variable {A B C a b c S : ℝ}

-- Given the conditions
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom angle_side_relation1 : A + B + C = π
axiom angle_side_relation2 : a = b * sin A
axiom angle_side_relation3 : 4 * S = sqrt 3 * (a^2 + b^2 - c^2)
axiom angle_side_relation4 : S = ½ * a * c * sin B

-- f(x) function definition
def f (x : ℝ) : ℝ := 4 * sin x * cos (x + π / 6) + 1

-- Statements to prove
theorem find_C : C = π / 3 := sorry
theorem find_S (hfx : ∀ (A : ℝ), f A ≤ f (π / 6)) : S = sqrt 3 / 2 := sorry

end find_C_find_S_l24_24582


namespace ellipse_eqn_and_max_area_l24_24123

noncomputable def ellipse_equation : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 8 + y^2 / 4 = 1))

noncomputable def maximum_triangle_area : Prop :=
  ∃ (a b p m : ℝ), 
    a > b ∧ b > 0 ∧ 
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ a^2 = b^2 + (b^2) ∧ b = 2 ∧ a^2 = 8) ∧
    (∃ F : ℝ × ℝ, (F = (2, 0)) ∧ (p = 4) ∧ (∀ x y : ℝ, y^2 = 2 * p * x) ∧ 
      0 ≤ m ∧ m ≤ 1 ∧ 
      (∃ A B : ℝ × ℝ, 
        (∀ y x : ℝ, y = x + m ∧ y^2 = 8 * x ∧
           (area_of_triangle F A B) =  32 * sqrt(6) / 9)))

theorem ellipse_eqn_and_max_area : ellipse_equation ∧ maximum_triangle_area :=
begin
  split;
  sorry -- proofs to be filled in
end

end ellipse_eqn_and_max_area_l24_24123


namespace complex_expr_simplify_l24_24692

noncomputable def complex_demo : Prop :=
  let i := Complex.I
  7 * (4 + 2 * i) - 2 * i * (7 + 3 * i) = (34 : ℂ)

theorem complex_expr_simplify : 
  complex_demo :=
by
  -- proof skipped
  sorry

end complex_expr_simplify_l24_24692


namespace jenny_cases_l24_24199

theorem jenny_cases (total_boxes cases_per_box : ℕ) (h1 : total_boxes = 24) (h2 : cases_per_box = 8) :
  total_boxes / cases_per_box = 3 := by
  sorry

end jenny_cases_l24_24199


namespace no_unhappy_days_l24_24738

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
  sorry

end no_unhappy_days_l24_24738


namespace unique_solution_l24_24019

noncomputable def f (x : ℝ) : ℝ := 2^x + 3^x + 6^x

theorem unique_solution : ∀ x : ℝ, f x = 7^x ↔ x = 2 :=
by
  sorry

end unique_solution_l24_24019


namespace min_abs_k1_k2_is_one_l24_24533

open Classical

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let e := Real.sqrt (a^2 + b^2) / a in
  e

theorem min_abs_k1_k2_is_one (a b p q s t : ℝ) (ha : a > 0) (hb : b > 0)
  (h_MN_on_hyperbola : (p^2 / a^2 - q^2 / b^2 = 1) ∧ (-p^2 / a^2 - q^2 / b^2 = -1))
  (h_P_on_hyperbola : s^2 / a^2 - t^2 / b^2 = 1)
  (k1 k2 : ℝ) (h_k_slope : k1 * k2 = b^2 / a^2)
  (min_abs_sum : |k1| + |k2| = 1) : eccentricity_of_hyperbola a b ha hb = Real.sqrt 5 / 2 :=
by
  sorry

end min_abs_k1_k2_is_one_l24_24533


namespace min_n_for_non_zero_constant_term_l24_24523

-- Define the condition of non-zero constant term in the expansion of (x^5 - 1/x)^n
theorem min_n_for_non_zero_constant_term (n : ℕ) (h : ∃ k, (x^5 - 1/x)^n = k ∧ k ≠ 0) :
  6 ≤ n :=
begin
  -- proof skipped
  sorry
end

end min_n_for_non_zero_constant_term_l24_24523


namespace price_of_cookies_l24_24758

def cupcakes_per_day := 20
def cupcakes_price := 1.5
def biscuits_per_day := 20
def biscuits_price := 1
def bakery_days := 5
def total_earnings := 350
def cookies_per_day := 10

theorem price_of_cookies :
  let cupcakes_earning := cupcakes_per_day * cupcakes_price in
  let biscuits_earning := biscuits_per_day * biscuits_price in
  let daily_earnings := cupcakes_earning + biscuits_earning in
  let earnings_from_cupcakes_and_biscuits := daily_earnings * bakery_days in
  let earnings_from_cookies := total_earnings - earnings_from_cupcakes_and_biscuits in
  let cookies_sold := cookies_per_day * bakery_days in
  let cookie_price := earnings_from_cookies / cookies_sold in
  cookie_price = 2 := 
by {
  sorry
}

end price_of_cookies_l24_24758


namespace max_value_of_expr_l24_24911

noncomputable def max_expr_value (x : ℝ) : ℝ :=
  x^6 / (x^12 + 4*x^9 - 6*x^6 + 16*x^3 + 64)

theorem max_value_of_expr : ∀ x : ℝ, max_expr_value x ≤ 1/26 :=
by
  sorry

end max_value_of_expr_l24_24911


namespace probability_second_year_not_science_l24_24173

def total_students := 2000

def first_year := 600
def first_year_science := 300
def first_year_arts := 200
def first_year_engineering := 100

def second_year := 450
def second_year_science := 250
def second_year_arts := 150
def second_year_engineering := 50

def third_year := 550
def third_year_science := 300
def third_year_arts := 200
def third_year_engineering := 50

def postgraduate := 400
def postgraduate_science := 200
def postgraduate_arts := 100
def postgraduate_engineering := 100

def not_third_year_not_science :=
  (first_year_arts + first_year_engineering) +
  (second_year_arts + second_year_engineering) +
  (postgraduate_arts + postgraduate_engineering)

def second_year_not_science := second_year_arts + second_year_engineering

theorem probability_second_year_not_science :
  (second_year_not_science / not_third_year_not_science : ℚ) = (2 / 7 : ℚ) :=
by
  let total := (first_year_arts + first_year_engineering) + (second_year_arts + second_year_engineering) + (postgraduate_arts + postgraduate_engineering)
  have not_third_year_not_science : total = 300 + 200 + 200 := by sorry
  have second_year_not_science_eq : second_year_not_science = 200 := by sorry
  sorry

end probability_second_year_not_science_l24_24173


namespace arcsin_neg_one_eq_neg_pi_div_two_l24_24416

theorem arcsin_neg_one_eq_neg_pi_div_two : Real.arcsin (-1) = -Real.pi / 2 :=
by
  sorry

end arcsin_neg_one_eq_neg_pi_div_two_l24_24416


namespace equal_segment_sums_exist_l24_24546

noncomputable def sum_range (l : List ℕ) (start end_incl : ℕ) : ℕ :=
  l.slice start (end_incl + 1).sum

theorem equal_segment_sums_exist 
  (a : List ℕ) (b : List ℕ) 
  (h₁ : a.length = 19) (h₂ : ∀ x ∈ a, x ≤ 88)
  (h₃ : b.length = 88) (h₄ : ∀ y ∈ b, y ≤ 19) :
  ∃ (m_start m_end n_start n_end : ℕ),
    m_start ≤ m_end ∧ m_end < a.length ∧
    n_start ≤ n_end ∧ n_end < b.length ∧
    sum_range a m_start m_end = sum_range b n_start n_end :=
begin
  sorry
end

end equal_segment_sums_exist_l24_24546


namespace find_base_l24_24476

theorem find_base (a : ℕ) :
  let digit_sum := (2 + 7 + 6)
  (185_a + 276_a = 46 * 12 ∧ digit_sum = 17) → 
  a = 12 :=
by
  sorry

end find_base_l24_24476


namespace distance_from_point_to_directrix_l24_24984

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24984


namespace no_unhappy_days_l24_24735

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
  sorry

end no_unhappy_days_l24_24735


namespace fraction_of_rotten_fruits_l24_24792

theorem fraction_of_rotten_fruits (a p : ℕ) (rotten_apples_eq_rotten_pears : (2 / 3) * a = (3 / 4) * p)
    (rotten_apples_fraction : 2 / 3 = 2 / 3)
    (rotten_pears_fraction : 3 / 4 = 3 / 4) :
    (4 * a) / (3 * (a + (4 / 3) * (2 * a) / 3)) = 12 / 17 :=
by
  sorry

end fraction_of_rotten_fruits_l24_24792


namespace total_spent_by_pete_and_raymond_l24_24275

def initial_money_in_cents : ℕ := 250
def pete_spent_in_nickels : ℕ := 4
def nickel_value_in_cents : ℕ := 5
def raymond_dimes_left : ℕ := 7
def dime_value_in_cents : ℕ := 10

theorem total_spent_by_pete_and_raymond : 
  (pete_spent_in_nickels * nickel_value_in_cents) 
  + (initial_money_in_cents - (raymond_dimes_left * dime_value_in_cents)) = 200 := sorry

end total_spent_by_pete_and_raymond_l24_24275


namespace no_unhappy_days_l24_24728

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end no_unhappy_days_l24_24728


namespace third_side_parallel_to_original_side_l24_24176

theorem third_side_parallel_to_original_side
  (A B C A1 B1 C1 : Point)
  (h1 : Altitude A A1 B B1 C C1)
  (h2 : Parallel A1 C1 A C)
  (h3 : Parallel A1 B1 A B)
: Parallel B1 C1 B C :=
sorry

end third_side_parallel_to_original_side_l24_24176


namespace number_of_perfect_apples_l24_24589

theorem number_of_perfect_apples (total_apples : ℕ) (too_small_ratio : ℚ) (not_ripe_ratio : ℚ)
  (h_total : total_apples = 30)
  (h_too_small_ratio : too_small_ratio = 1 / 6)
  (h_not_ripe_ratio : not_ripe_ratio = 1 / 3) :
  (total_apples - (too_small_ratio * total_apples).natAbs - (not_ripe_ratio * total_apples).natAbs) = 15 := by
  sorry

end number_of_perfect_apples_l24_24589


namespace students_who_like_both_l24_24595

theorem students_who_like_both (A B C : ℕ) : 
  A = 9 → B = 8 → C = 11 → A + B - C = 6 :=
begin
  intros hA hB hC,
  rw [hA, hB, hC],
  simp,
end

end students_who_like_both_l24_24595


namespace resulting_polygon_sides_l24_24015

/-- Given a sequence of shapes and their respective side counts, and specific construction conditions,
    we need to prove that the resulting polygon has 28 sides exposed to the outside. -/
theorem resulting_polygon_sides : 
  let pentagon_sides := 5 in
  let triangle_sides := 3 in
  let heptagon_sides := 7 in
  let nonagon_sides := 9 in
  let dodecagon_sides := 12 in
  let shared_sides := 2 in
  let pentagon_and_dodecagon := pentagon_sides + dodecagon_sides - shared_sides in
  let triangle_heptagon_nonagon_shared := ((triangle_sides + heptagon_sides + nonagon_sides) - (3 * shared_sides)) in
  pentagon_and_dodecagon + triangle_heptagon_nonagon_shared = 28 :=
by
  sorry

end resulting_polygon_sides_l24_24015


namespace distance_Bella_to_Galya_l24_24191

theorem distance_Bella_to_Galya (D_B D_V D_G : ℕ) (BV VG : ℕ)
  (hD_B : D_B = 700)
  (hD_V : D_V = 600)
  (hD_G : D_G = 650)
  (hBV : BV = 100)
  (hVG : VG = 50)
  : BV + VG = 150 := by
  sorry

end distance_Bella_to_Galya_l24_24191


namespace alices_favorite_number_l24_24817

theorem alices_favorite_number :
  ∃ n : ℕ, 80 < n ∧ n ≤ 130 ∧ n % 13 = 0 ∧ n % 3 ≠ 0 ∧ ((n / 100) + (n % 100 / 10) + (n % 10)) % 4 = 0 ∧ n = 130 :=
by
  sorry

end alices_favorite_number_l24_24817


namespace max_value_f_inequality_a2_b2_c2_l24_24097

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

-- Define the maximum value s as 3
noncomputable def s : ℝ := 3

-- Define conditions for a, b, and c
variables (a b c : ℝ)
axiom pos_a : a ∈ set.Ioi 0
axiom pos_b : b ∈ set.Ioi 0
axiom pos_c : c ∈ set.Ioi 0
axiom sum_abc : a + b + c = s

-- Statement to prove the maximum value of f(x) is 3
theorem max_value_f : (∃ x : ℝ, f x = s) := 
  sorry

-- Statement to prove a^2 + b^2 + c^2 >= 3
theorem inequality_a2_b2_c2 : a^2 + b^2 + c^2 ≥ 3 :=
  sorry

end max_value_f_inequality_a2_b2_c2_l24_24097


namespace minimum_cardinality_of_three_sets_l24_24700

open Set

noncomputable def min_intersection_cardinality (X Y Z : Type) [Fintype X] [Fintype Y] [Fintype Z]
  (h1 : Fintype.card X + Fintype.card Y + Fintype.card Z = Fintype.card (X ∪ Y ∪ Z))
  (h2 : Fintype.card X = 50)
  (h3 : Fintype.card Y = 50)
  (h4 : (X ∩ Y ∩ Z).Nonempty) : ℕ :=
  min |X ∩ Y ∩ Z|

theorem minimum_cardinality_of_three_sets (X Y Z : Type) [Fintype X] [Fintype Y] [Fintype Z]
  (h1 : Fintype.card X + Fintype.card Y + Fintype.card Z = Fintype.card (X ∪ Y ∪ Z))
  (h2 : Fintype.card X = 50)
  (h3 : Fintype.card Y = 50)
  (h4 : (X ∩ Y ∩ Z).Nonempty) : min_intersection_cardinality X Y Z h1 h2 h3 h4 = 1 := sorry

end minimum_cardinality_of_three_sets_l24_24700


namespace total_market_cost_l24_24663

-- Defining the variables for the problem
def pounds_peaches : Nat := 5 * 3
def pounds_apples : Nat := 4 * 3
def pounds_blueberries : Nat := 3 * 3

def cost_per_pound_peach := 2
def cost_per_pound_apple := 1
def cost_per_pound_blueberry := 1

-- Defining the total costs
def cost_peaches : Nat := pounds_peaches * cost_per_pound_peach
def cost_apples : Nat := pounds_apples * cost_per_pound_apple
def cost_blueberries : Nat := pounds_blueberries * cost_per_pound_blueberry

-- Total cost
def total_cost : Nat := cost_peaches + cost_apples + cost_blueberries

-- Theorem to prove the total cost is $51.00
theorem total_market_cost : total_cost = 51 := by
  sorry

end total_market_cost_l24_24663


namespace functional_equation_solution_l24_24095

-- Define the functional equation with given conditions
def func_eq (f : ℤ → ℝ) (N : ℕ) : Prop :=
  (∀ k : ℤ, f (2 * k) = 2 * f k) ∧
  (∀ k : ℤ, f (N - k) = f k)

-- State the mathematically equivalent proof problem
theorem functional_equation_solution (N : ℕ) (f : ℤ → ℝ) 
  (h1 : ∀ k : ℤ, f (2 * k) = 2 * f k)
  (h2 : ∀ k : ℤ, f (N - k) = f k) : 
  ∀ a : ℤ, f a = 0 := 
sorry

end functional_equation_solution_l24_24095


namespace jerry_total_miles_l24_24626

def monday : ℕ := 15
def tuesday : ℕ := 18
def wednesday : ℕ := 25
def thursday : ℕ := 12
def friday : ℕ := 10

def total : ℕ := monday + tuesday + wednesday + thursday + friday

theorem jerry_total_miles : total = 80 := by
  sorry

end jerry_total_miles_l24_24626


namespace find_values_general_formula_l24_24501

variable (a_n S_n : ℕ → ℝ)

-- Conditions
axiom sum_sequence (n : ℕ) (hn : n > 0) :  S_n n = (1 / 3) * (a_n n - 1)

-- Questions
theorem find_values :
  (a_n 1 = 2) ∧ (a_n 2 = 5) ∧ (a_n 3 = 8) := sorry

theorem general_formula (n : ℕ) :
  n > 0 → a_n n = n + 1 := sorry

end find_values_general_formula_l24_24501


namespace triangle_area_ratio_l24_24348

theorem triangle_area_ratio (a b c : ℕ) (d e f : ℕ) 
  (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) 
  (h4 : d = 9) (h5 : e = 12) (h6 : f = 15) 
  (GHI_right : a^2 + b^2 = c^2)
  (JKL_right : d^2 + e^2 = f^2):
  (0.5 * a * b) / (0.5 * d * e) = 4 / 9 := 
by 
  sorry

end triangle_area_ratio_l24_24348


namespace Sophie_needs_additional_coins_l24_24299

theorem Sophie_needs_additional_coins (n_friends : ℕ) (initial_coins : ℕ) (min_additional_coins : ℕ) :
    n_friends = 10 →
    initial_coins = 40 →
    min_additional_coins = 15 → 
    (∀ i : ℕ, i ≥ 1 → i ≤ n_friends → ∃ unique (dist : ℕ → ℕ), 
        (∀ j, 1 ≤ j ∧ j ≤ n_friends → dist j = j) ∧ 
        ((range (succ n_friends)).sum dist  - initial_coins = min_additional_coins)) :=
by
  sorry

end Sophie_needs_additional_coins_l24_24299


namespace distance_from_point_to_directrix_l24_24979

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24979


namespace no_unhappy_days_l24_24723

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end no_unhappy_days_l24_24723


namespace triangle_area_ratio_l24_24343

noncomputable def area_ratio (a b c d e f : ℕ) : ℚ :=
  (a * b) / (d * e)

theorem triangle_area_ratio : area_ratio 6 8 10 9 12 15 = 4 / 9 :=
by
  sorry

end triangle_area_ratio_l24_24343


namespace exactly_one_statement_correct_l24_24709

/-- The negation of the proposition "There exists an x_0 ∈ ℝ such that x_0^2 - x_0 > 0" 
is "For all x ∈ ℝ, x^2 - x ≤ 0". Thus, statement (1) is correct. -/
def statement1 : Prop :=
  ¬ (∃ (x₀ : ℝ), x₀^2 - x₀ > 0) ↔ ∀ (x : ℝ), x^2 - x ≤ 0

/-- Given that the proposition p ∧ q is false, then it means at least one of p or q must be false.
Thus, statement (2) is incorrect. -/
def statement2 (p q : Prop) : Prop :=
  ¬ (p ∧ q) → (¬ p ∧ ¬ q)

/-- The contrapositive of "If x^2 = 1, then x = 1" is "If x ≠ 1, then x^2 ≠ 1".
Thus, statement (3) is incorrect. -/
def statement3 : Prop :=
  (∀ x : ℝ, x^2 = 1 → x = 1) ↔ (∀ x : ℝ, x ≠ 1 → x^2 ≠ 1)

/-- "x = -1" is a necessary but not sufficient condition for "x^2 - 5x - 6 = 0".
Thus, statement (4) is incorrect. -/
def statement4 : Prop :=
  ∀ x : ℝ, x^2 - 5 * x - 6 = 0 → x = -1

theorem exactly_one_statement_correct :
  (statement1 ∧ ¬ statement2 ∧ ¬ statement3 ∧ ¬ statement4) :=
by sorry

end exactly_one_statement_correct_l24_24709


namespace values_of_cos_0_45_l24_24556

-- Define the interval and the condition for the cos function
def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 360
def cos_condition (x : ℝ) : Prop := Real.cos x = 0.45

-- Final theorem statement
theorem values_of_cos_0_45 : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (x : ℝ), in_interval x ∧ cos_condition x ↔ x = 1 ∨ x = 2 := 
sorry

end values_of_cos_0_45_l24_24556


namespace midpoint_polar_coordinates_l24_24171

theorem midpoint_polar_coordinates (r θ_A θ_B : ℝ) (h1 : θ_A = π / 3) (h2 : θ_B = 2 * π / 3) (h3 : r > 0) : 
  let θ_M := (θ_A + θ_B) / 2 in
  (r, θ_M) = (10, π / 2) :=
by
  sorry

end midpoint_polar_coordinates_l24_24171


namespace find_coordinates_and_area_l24_24522

noncomputable def triangleABC : Type :=
{A B C : ℝ × ℝ // 
(A = (3, 2)) ∧ 
(∃ a b : ℝ, B = (a, b) ∧ 2 * a - b - 9 = 0 ∧ (a + 3) / 2 - 3 * ((b + 2) / 2) + 8 = 0) ∧ 
(∃ m n : ℝ, C = (m, n) ∧ m - 3 * n + 8 = 0 ∧ (n - 2) = -1 / 2 * (m - 3))
}

theorem find_coordinates_and_area 
  (t : triangleABC) : 
  let B := (8, 7) in 
  let C := (1, 3) in 
  (t.1.1 = B) ∧ (t.1.2 = C) ∧ 
  (1 / 2) * (sqrt ((8 - 3)^2 + (7 - 2)^2)) * (abs ((1 - 3 - 1) / sqrt 2 )  / (1 + 1) / 2 ) = 15 / 2 := 
by
  sorry

end find_coordinates_and_area_l24_24522


namespace tobias_hours_per_day_correct_l24_24667

-- Define the conditions
def nathan_hours_per_day := 3
def days_per_week := 7
def nathan_weeks := 2
def total_hours := 77

-- Define the term to be solved
noncomputable def tobias_hours_per_day : ℕ :=
  let nathan_total_hours := nathan_hours_per_day * days_per_week * nathan_weeks in
  let tobias_total_hours := total_hours - nathan_total_hours in
  tobias_total_hours / days_per_week

-- The statement to be proven
theorem tobias_hours_per_day_correct : tobias_hours_per_day = 5 := by
  sorry

end tobias_hours_per_day_correct_l24_24667


namespace factorize_quadratic_l24_24042

theorem factorize_quadratic (x : ℝ) : x^2 - 2 * x = x * (x - 2) :=
sorry

end factorize_quadratic_l24_24042


namespace probability_of_one_exactly_four_times_l24_24572

def roll_probability := (1 : ℝ) / 6
def non_one_probability := (5 : ℝ) / 6

lemma prob_roll_one_four_times :
  ∑ x in {1, 2, 3, 4, 5}, 
      roll_probability^4 * non_one_probability = 
    5 * (roll_probability^4 * non_one_probability) :=
by
  sorry

theorem probability_of_one_exactly_four_times :
  (5 : ℝ) * roll_probability^4 * non_one_probability = (25 : ℝ) / 7776 :=
by
  have key := prob_roll_one_four_times
  sorry

end probability_of_one_exactly_four_times_l24_24572


namespace problem_statement_l24_24074

theorem problem_statement (m a b : ℝ) (h0 : 9^m = 10) (h1 : a = 10^m - 11) (h2 : b = 8^m - 9) : a > 0 ∧ 0 > b :=
by
  sorry

end problem_statement_l24_24074


namespace sum_of_digits_fibonacci_greater_than_c_l24_24456

noncomputable def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem sum_of_digits_fibonacci_greater_than_c {b c : ℕ} (hb : b > 1) (hc : c > 1) :
  ∃ n, digit_sum b (fibonacci n) > c := sorry

end sum_of_digits_fibonacci_greater_than_c_l24_24456


namespace octagon_area_is_pi_over_2_l24_24808

noncomputable def side_length : ℝ :=
  2 / Real.sqrt (2 + Real.sqrt 2)

def parameterized_side (t : ℝ) (ht : -side_length / 2 ≤ t ∧ t ≤ side_length / 2) : ℂ :=
  ⟨1, t⟩

def transformed_side (a b : ℝ) (ha : a = 1) (hb : b ∈ Icc (-side_length / 2) (side_length / 2)) : ℂ :=
  ⟨1 / (1 + b^2), -b / (1 + b^2)⟩

def curve_eqn (x y : ℝ) (hxy : (y / x)^2 + y^2 = x / (1 + (y / x)^2)) : bool :=
  (x - 0.5)^2 + y^2 = 0.25

noncomputable def octagon_area_proof : ℝ :=
  16 * (1 / 8) * Real.pi * ((1/2)^2)

theorem octagon_area_is_pi_over_2 :
  octagon_area_proof = Real.pi / 2 :=
sorry

end octagon_area_is_pi_over_2_l24_24808


namespace constant_term_in_expansion_l24_24900

theorem constant_term_in_expansion (x : ℂ) : 
  (2 - (3 / x)) * (x ^ 2 + 2 / x) ^ 5 = 0 := 
sorry

end constant_term_in_expansion_l24_24900


namespace obtuse_triangle_of_sin_cos_sum_l24_24507

theorem obtuse_triangle_of_sin_cos_sum
  (A : ℝ) (hA : 0 < A ∧ A < π) 
  (h_eq : Real.sin A + Real.cos A = 12 / 25) :
  π / 2 < A ∧ A < π :=
sorry

end obtuse_triangle_of_sin_cos_sum_l24_24507


namespace distance_from_point_to_directrix_l24_24985

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24985


namespace monotonically_increasing_interval_l24_24311
noncomputable def log_0_3 : ℝ → ℝ := λ x, Real.log x / Real.log 0.3

theorem monotonically_increasing_interval :
  ∀ x : ℝ, x < -2 → ∀ u : ℝ, u = x^2 + x - 2 → (x - 1) * (x + 2) > 0 → (∀ v : ℝ, v = (x^2 + x - 2) → log_0_3 v) = log_0_3 (x^2 + x - 2) :=
by
  sorry

end monotonically_increasing_interval_l24_24311


namespace solution_l24_24791

def problem_statement : Prop :=
  let number := 300 in
  let thirty_percent := 0.3 * number in
  thirty_percent - 70 = 20

theorem solution : problem_statement :=
by
  sorry

end solution_l24_24791


namespace series_converges_l24_24025

def series (k : ℚ) (n : ℕ) : ℚ := (n! * k^n) / (n + 1)^n

noncomputable def ratio_test_limit (an : ℕ → ℚ) : ℚ :=
  lim (λ n, |(an (n + 1)) / (an n)|)

theorem series_converges : 
  ratio_test_limit (series (19 / 7)) < 1 :=
  sorry

end series_converges_l24_24025


namespace angle_AXB_at_min_dot_product_l24_24094

open Real

variables (O P A B: ℝ × ℝ)
#eval (2 : ℝ)
def X (x y: ℝ) : Prop := 
  x = 2 * y

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := 
  (u.1 - v.1, u.2 - v.2)

def dot_product (u v : ℝ × ℝ) : ℝ := 
  u.1 * v.1 + u.2 * v.2

def norm (u : ℝ × ℝ) : ℝ := 
  sqrt (u.1 * u.1 + u.2 * u.2)

noncomputable def angle_between (u v : ℝ × ℝ) := 
  arccos ((dot_product u v) / (norm u * norm v))
 
theorem angle_AXB_at_min_dot_product :
  O = (0, 0) → P = (2, 1) → A = (1, 7) → B = (5, 1) →
  ∃ (y0 : ℝ), X P y0 →
  let X := (2 * y0, y0) in 
  let XA := vector_sub A X in 
  let XB := vector_sub B X in 
  angle_between XA XB = arccos (-4 * sqrt 17 / 17) := by
sorry

end angle_AXB_at_min_dot_product_l24_24094


namespace putnam_inequality_l24_24225

variable (a x : ℝ)

theorem putnam_inequality (h1 : 0 < x) (h2 : x < a) :
  (a - x)^6 - 3 * a * (a - x)^5 +
  5 / 2 * a^2 * (a - x)^4 -
  1 / 2 * a^4 * (a - x)^2 < 0 :=
by
  sorry

end putnam_inequality_l24_24225


namespace ronalds_egg_sharing_l24_24290

theorem ronalds_egg_sharing (total_eggs : ℕ) (eggs_per_friend : ℕ) (num_friends : ℕ) 
  (h1 : total_eggs = 16) (h2 : eggs_per_friend = 2) 
  (h3 : num_friends = total_eggs / eggs_per_friend) : 
  num_friends = 8 := 
by 
  sorry

end ronalds_egg_sharing_l24_24290


namespace angle_A_eq_pi_div_3_of_condition_l24_24161

-- Define the problem using Lean statements and conditions
theorem angle_A_eq_pi_div_3_of_condition
  (a b c : ℝ)
  (h : (a + b + c) * (b + c - a) = 3 * b * c) :
  ∠A = Real.pi / 3 := 
sorry

end angle_A_eq_pi_div_3_of_condition_l24_24161


namespace price_difference_l24_24256

noncomputable def original_price (P : ℝ) : Prop :=
  0.80 * P + 4000 = 30000

theorem price_difference (P : ℝ) (h : original_price P) : P - 30000 = 2500 := by
  unfold original_price at h
  linarith

#check price_difference -- to ensure that the theorem is correct

end price_difference_l24_24256


namespace f_neg_sqrt2_over_2_l24_24947

-- Define the conditions
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ :=
if h : x > 0 then log x / log 2 - 1 else 0 -- The definition outside of x > 0 is irrelevant for our proof

-- Define the theorem
theorem f_neg_sqrt2_over_2 :
  is_odd_function f →
  (∀ x, x > 0 → f x = log x / log 2 - 1) →
  f (- (real.sqrt 2) / 2) = 3 / 2 :=
by
  intros h_odd h_fpos
  sorry

end f_neg_sqrt2_over_2_l24_24947


namespace average_mb_per_hour_of_music_l24_24381

/--
Given a digital music library:
- It contains 14 days of music.
- The first 7 days use 10,000 megabytes of disk space.
- The next 7 days use 14,000 megabytes of disk space.
- Each day has 24 hours.

Prove that the average megabytes per hour of music in this library is 71 megabytes.
-/
theorem average_mb_per_hour_of_music
  (days_total : ℕ) 
  (days_first : ℕ) 
  (days_second : ℕ) 
  (mb_first : ℕ) 
  (mb_second : ℕ) 
  (hours_per_day : ℕ) 
  (total_mb : ℕ) 
  (total_hours : ℕ) :
  days_total = 14 →
  days_first = 7 →
  days_second = 7 →
  mb_first = 10000 →
  mb_second = 14000 →
  hours_per_day = 24 →
  total_mb = mb_first + mb_second →
  total_hours = days_total * hours_per_day →
  total_mb / total_hours = 71 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end average_mb_per_hour_of_music_l24_24381


namespace volume_of_box_with_ratio_l24_24390

theorem volume_of_box_with_ratio (x : ℕ) (hx : 0 < x) :
  ∃ (V : ℕ), V = 70 * x^3 ∧ V = 70 :=
by
  use 70
  have h1 : 2 * x * 5 * x * 7 * x = 70 * x^3 := by linarith
  have h2 : 70 * 1^3 = 70 := by norm_num
  rw ← h2
  exact ⟨rfl, rfl⟩
sory

end volume_of_box_with_ratio_l24_24390


namespace range_of_x_l24_24563

theorem range_of_x (x : ℝ) (h : (x + 1) ^ 0 = 1) : x ≠ -1 :=
sorry

end range_of_x_l24_24563


namespace quadrilateral_condition_l24_24391

theorem quadrilateral_condition (a b c d : ℝ) (h : a <= b ∧ b <= c ∧ c <= d) 
  (h1 : a + b + c + d = 2) 
  (h2 : ∀ x ∈ ({a, b, c, d} : set ℝ), 1/4 <= x ∧ x <= 1/2) :
  a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a :=
sorry

end quadrilateral_condition_l24_24391


namespace incorrect_statement_trajectory_of_P_l24_24803

noncomputable def midpoint_of_points (x1 x2 y1 y2 : ℝ) : ℝ × ℝ :=
((x1 + x2) / 2, (y1 + y2) / 2)

theorem incorrect_statement_trajectory_of_P (p k x0 y0 : ℝ) (hp : p > 0)
    (A B : ℝ × ℝ)
    (hA : A.1 * A.1 + 2 * p * A.2 = 0)
    (hB : B.1 * B.1 + 2 * p * B.2 = 0)
    (hMid : (x0, y0) = midpoint_of_points A.1 B.1 A.2 B.2)
    (hLine : A.2 = k * (A.1 - p / 2))
    (hLineIntersection : B.2 = k * (B.1 - p / 2)) : y0 ^ 2 ≠ 4 * p * (x0 - p / 2) :=
by
  sorry

end incorrect_statement_trajectory_of_P_l24_24803


namespace AB_gt_BC_l24_24786

variable {Point : Type}
variable (A B C D : Point)

-- Definitions of the conditions
def bisector_triangle (A B C D : Point) : Prop :=
  is_bisector_of_triangle A B C D

def AD_gt_CD (A D C : Point) : Prop :=
  dist A D > dist C D

-- The theorem statement
theorem AB_gt_BC
  (h₁ : bisector_triangle A B C D)
  (h₂ : AD_gt_CD A D C) :
  dist A B > dist B C :=
sorry

end AB_gt_BC_l24_24786


namespace prove_distance_uphill_l24_24156

noncomputable def distance_uphill := 
  let flat_speed := 20
  let uphill_speed := 12
  let extra_flat_distance := 30
  let uphill_time (D : ℝ) := D / uphill_speed
  let flat_time (D : ℝ) := (D + extra_flat_distance) / flat_speed
  ∃ D : ℝ, uphill_time D = flat_time D ∧ D = 45

theorem prove_distance_uphill : distance_uphill :=
sorry

end prove_distance_uphill_l24_24156


namespace exists_five_digit_palindromic_divisible_by_5_count_five_digit_palindromic_numbers_divisible_by_5_l24_24250

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem exists_five_digit_palindromic_divisible_by_5 :
  ∃ n, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n := by
  -- Proof is omitted
  sorry

theorem count_five_digit_palindromic_numbers_divisible_by_5 :
  (finset.filter (λ n, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n) (finset.range 100000)).card = 100 := by
  -- Proof is omitted
  sorry

end exists_five_digit_palindromic_divisible_by_5_count_five_digit_palindromic_numbers_divisible_by_5_l24_24250


namespace find_angle_A_l24_24584
-- Necessary import for mathematical framework

-- Problem statement in Lean 4
theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h_triangle : a^2 - b^2 - c^2 + sqrt 3 * b * c = 0) :
  A = π / 6 :=
sorry

end find_angle_A_l24_24584


namespace problem_statement_l24_24608

-- Definitions based on the conditions:

def curve_C (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.sin (θ + π / 4)

def line_l_param (P : ℝ×ℝ) (t : ℝ) : ℝ×ℝ :=
  (t * Real.cos (π / 3), P.2 + t * Real.sin (π / 3))

def curve_C_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * x + 2 * y

def param_eq_sub (t : ℝ) : Prop :=
  t^2 - t - 1 = 0

def absolute_value (x : ℝ) : ℝ := if x < 0 then -x else x

def PM (M P : ℝ×ℝ) : ℝ :=
  Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2)

def PN (N P : ℝ×ℝ) : ℝ :=
  Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2)

-- The proof statement:
theorem problem_statement
  (P : ℝ × ℝ := ⟨ 0, 1 ⟩)
  (M N : ℝ × ℝ)
  (a b : ℝ)
  (h1 : curve_C_cartesian M.1 M.2)
  (h2 : curve_C_cartesian N.1 N.2)
  (h3 : param_eq_sub a)
  (h4 : param_eq_sub b):
  (|PM(M, P)| := absolute_value a) ∧ (|PN(N, P)| := absolute_value b)
  → (1 / absolute_value a + 1 / absolute_value b = Real.sqrt 5) :=
begin
  sorry,
end

end problem_statement_l24_24608


namespace count_palindrome_five_digit_div_by_5_l24_24241

-- Define what it means for a number to be palindromic.
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Define what it means for a number to be a five-digit number.
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

-- Define what it means for a number to be divisible by 5.
def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Define the set of five-digit palindromic numbers divisible by 5.
def palindrome_five_digit_div_by_5_numbers (n : ℕ) : Prop :=
  is_five_digit n ∧ is_palindrome n ∧ is_divisible_by_5 n

-- Prove that the quantity of such numbers is 100.
theorem count_palindrome_five_digit_div_by_5 : 
  (finset.filter 
    (λ n, palindrome_five_digit_div_by_5_numbers n)
    (finset.range 100000)
  ).card = 100 :=
begin
  sorry
end

end count_palindrome_five_digit_div_by_5_l24_24241


namespace negation_of_prop_p_is_correct_l24_24137

-- Define the original proposition p
def prop_p (x y : ℝ) : Prop := x > 0 ∧ y > 0 → x * y > 0

-- Define the negation of the proposition p
def neg_prop_p (x y : ℝ) : Prop := x ≤ 0 ∨ y ≤ 0 → x * y ≤ 0

-- The theorem we need to prove
theorem negation_of_prop_p_is_correct : ∀ x y : ℝ, neg_prop_p x y := 
sorry

end negation_of_prop_p_is_correct_l24_24137


namespace pyramid_volume_proof_l24_24324

/-
The rectangle ABCD has dimensions AB = 14 and BC = 15.
Diagonals AC and BD intersect at P.
Triangle ABP is cut out and removed.
Edges AP and BP are joined.
The figure is creased along segments CP and DP to form a triangular pyramid with all faces being isosceles triangles.
Prove that the volume of the pyramid is 35 * 99 / sqrt(421).
-/

theorem pyramid_volume_proof :
  let A := (0 : ℝ, 15 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 0 : ℝ, 0 : ℝ)
  let C := (7 : ℝ, 0 : ℝ, 0 : ℝ)
  let D := (-7 : ℝ, 0 : ℝ, 0 : ℝ)
  let P := (0 : ℝ, 421 / 30 : ℝ, 99 / real.sqrt 421 : ℝ)
  volume_of_pyramid (A, B, C, D, P) = (35 : ℝ) * (99 : ℝ) / real.sqrt(421) :=
sorry

end pyramid_volume_proof_l24_24324


namespace abby_bridget_adjacent_probability_l24_24398

theorem abby_bridget_adjacent_probability :
  let students := 7
  let seats := 8
  let rows := 2
  let columns := 4
  let total_arrangements := fact 8
  let row_pairs := 2 * 3
  let column_pairs := 4
  let abby_bridget_permutations := 2
  let remaining_kids_arrangements := fact 6
  let favorable_outcomes := (row_pairs + column_pairs) * abby_bridget_permutations * remaining_kids_arrangements
  let probability := favorable_outcomes / total_arrangements
  probability = (5 : ℚ) / 14 :=
by {
  sorry
}

end abby_bridget_adjacent_probability_l24_24398


namespace arcsin_neg_one_eq_neg_half_pi_l24_24419

theorem arcsin_neg_one_eq_neg_half_pi :
  arcsin (-1) = - (Float.pi / 2) :=
by
  sorry

end arcsin_neg_one_eq_neg_half_pi_l24_24419


namespace determine_diamonds_in_F10_l24_24796

noncomputable def D : ℕ → ℕ
| 1     := 1
| 2     := 9
| 3     := 25
| (n+1) := D n + 8 * (n + 1)

theorem determine_diamonds_in_F10 : D 10 = 681 :=
by
  sorry

end determine_diamonds_in_F10_l24_24796


namespace exists_uv_l24_24632

open Set

def S (x y : ℝ) : Set ℕ := {s | ∃ n : ℕ, s = Int.floor (n * x + y)}

theorem exists_uv (r : ℝ) (hr : r > 1 ∧ ∃ p q : ℕ, r = 1 + p / q ∧ Nat.coprime p q) :
  ∃ u v : ℝ, S r 0 ∩ S u v = ∅ ∧ (S r 0 ∪ S u v = {k | True}) :=
by
  sorry

end exists_uv_l24_24632


namespace minimize_cost_l24_24800

def y1 (k1 x : ℝ) : ℝ := k1 / x
def y2 (k2 x : ℝ) : ℝ := k2 * x

theorem minimize_cost 
  (x y1 y2 : ℝ)
  (h1 : y1 = 20000) 
  (h2 : y2 = 80000)
  (hx : x = 10)
  (k1 k2 : ℝ)
  (hk1 : k1 = y1 * x) 
  (hk2 : k2 = y2 / x) :
  (∃ x_min, x_min = 5) := 
sorry

end minimize_cost_l24_24800


namespace even_function_phi_l24_24127

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + φ)

noncomputable def f' (x φ : ℝ) : ℝ := -Real.sqrt 3 * Real.sin (Real.sqrt 3 * x + φ)

noncomputable def y (x φ : ℝ) : ℝ := f x φ + f' x φ

def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = g (-x)

theorem even_function_phi :
  (∀ x : ℝ, y x φ = y (-x) φ) → ∃ k : ℤ, φ = -Real.pi / 3 + k * Real.pi :=
by
  sorry

end even_function_phi_l24_24127


namespace find_coordinates_C_l24_24901

def is_equidistant (A B C : ℝ × ℝ × ℝ) : Prop :=
  let dist (P Q : ℝ × ℝ × ℝ) : ℝ := real.dist P Q in
  dist A C = dist B C

def coordinates_A : ℝ × ℝ × ℝ := (-4, 1, 7)
def coordinates_B : ℝ × ℝ × ℝ := (3, 5, -2)

theorem find_coordinates_C :
  ∃ z : ℝ, is_equidistant coordinates_A coordinates_B (0, 0, z) ∧ z = 14 / 9 :=
by
  sorry

end find_coordinates_C_l24_24901


namespace sunlovers_happy_days_l24_24731

open Nat

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end sunlovers_happy_days_l24_24731


namespace AB_greater_than_BC_l24_24183

variables {A B C D M Q T : Point}

-- Given Conditions
def angle_A_eq_40_deg (A B C D : Point) : Prop :=
  angle A B C = 40

def angle_D_eq_45_deg (A B C D : Point) : Prop :=
  angle D A B = 45

def angle_B_bisector_divides_AD_in_half (A B C D M : Point) : Prop :=
  midpoint M A D ∧ ∃ B_bisector, (angle B A D = angle B D A) ∧ line_through B B_bisector ∧ segment_div B_bisector A D

theorem AB_greater_than_BC (A B C D : Point) (M : Point)
  (h1 : angle_A_eq_40_deg A B C D)
  (h2 : angle_D_eq_45_deg A B C D)
  (h3 : angle_B_bisector_divides_AD_in_half A B C D M) : 
  length (segment A B) > length (segment B C) :=
sorry

end AB_greater_than_BC_l24_24183


namespace total_balls_no_holes_is_81_l24_24264

-- Define the total number of soccer balls
def total_soccer_balls : ℕ := 180

-- Define the total number of basketballs
def total_basketballs : ℕ := 75

-- Define the number of soccer balls with holes
def soccer_balls_with_holes : ℕ := 125

-- Define the number of basketballs with holes
def basketballs_with_holes : ℕ := 49

-- Define the number of soccer balls without holes
def soccer_balls_without_holes : ℕ := total_soccer_balls - soccer_balls_with_holes

-- Define the number of basketballs without holes
def basketballs_without_holes : ℕ := total_basketballs - basketballs_with_holes

-- Define the total number of balls without holes
def total_balls_without_holes : ℕ := soccer_balls_without_holes + basketballs_without_holes

-- Prove that the total number of balls without holes is 81
theorem total_balls_no_holes_is_81 : total_balls_without_holes = 81 :=
by
  unfold total_soccer_balls total_basketballs soccer_balls_with_holes basketballs_with_holes
         soccer_balls_without_holes basketballs_without_holes total_balls_without_holes
  sorry

end total_balls_no_holes_is_81_l24_24264


namespace find_x_for_floor_eq_49_l24_24463

theorem find_x_for_floor_eq_49 (x : ℝ) (hx : ⌊x⌊x⌋⌋ = 49) : 7 ≤ x ∧ x < 50 / 7 := 
by sorry

end find_x_for_floor_eq_49_l24_24463


namespace jack_change_l24_24625

theorem jack_change :
  let discountedCost1 := 4.50
  let discountedCost2 := 4.50
  let discountedCost3 := 5.10
  let cost4 := 7.00
  let totalDiscountedCost := discountedCost1 + discountedCost2 + discountedCost3 + cost4
  let tax := totalDiscountedCost * 0.05
  let taxRounded := 1.06 -- Tax rounded to nearest cent
  let totalCostWithTax := totalDiscountedCost + taxRounded
  let totalCostWithServiceFee := totalCostWithTax + 2.00
  let totalPayment := 20 + 10 + 4 * 1
  let change := totalPayment - totalCostWithServiceFee
  change = 9.84 :=
by
  sorry

end jack_change_l24_24625


namespace confidence_level_unrelated_l24_24159

noncomputable def chi_squared_value : ℝ := 8.654

theorem confidence_level_unrelated :
  chi_squared_value > 6.635 →
  (100 - 99) = 1 :=
by
  sorry

end confidence_level_unrelated_l24_24159


namespace collinearity_and_concyclicity_l24_24930

variable {A B C P P_A P_B P_C : Point}
variable [IsQuadrilateral ABCP]
variable [OrthogonalProjection P_A P BC]
variable [OrthogonalProjection P_B P CA]
variable [OrthogonalProjection P_C P AB]

theorem collinearity_and_concyclicity :
  Collinear P_A P_B P_C ↔ Concyclic A B C P :=
sorry

end collinearity_and_concyclicity_l24_24930


namespace angle_Z_of_triangle_l24_24164

theorem angle_Z_of_triangle (X Y Z : ℝ) (h1 : X + Y = 90) (h2 : X + Y + Z = 180) : 
  Z = 90 := 
sorry

end angle_Z_of_triangle_l24_24164


namespace direct_proportion_function_decrease_no_first_quadrant_l24_24099

-- Part (1)
theorem direct_proportion_function (a b : ℝ) (h : y = (2*a-4)*x + (3-b)) : a ≠ 2 ∧ b = 3 :=
sorry

-- Part (2)
theorem decrease_no_first_quadrant (a b : ℝ) (h : y = (2*a-4)*x + (3-b)) : a < 2 ∧ b ≥ 3 :=
sorry

end direct_proportion_function_decrease_no_first_quadrant_l24_24099


namespace arctan_sum_property_l24_24547

open Real

theorem arctan_sum_property (x y z : ℝ) :
  arctan x + arctan y + arctan z = π / 2 → x * y + y * z + x * z = 1 :=
by
  sorry

end arctan_sum_property_l24_24547


namespace product_increase_2022_l24_24613

theorem product_increase_2022 (a b c : ℕ) (h1 : a = 1) (h2 : b = 1) (h3 : c = 678) :
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2022 :=
by {
  -- The proof would go here, but it's not required per the instructions.
  sorry
}

end product_increase_2022_l24_24613


namespace range_of_x_l24_24117

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_x (b: ℝ) (h_b_gt_1: b > 1) :
  (∀ x: ℝ, f x = f (-x)) →
  (∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂) →
  f 1 < f (Real.log x / Real.log b) →
  x ∈ (set.Ioo 0 (1 / b)) ∪ (set.Ioi b) :=
sorry

end range_of_x_l24_24117


namespace distance_from_A_to_directrix_l24_24997

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l24_24997


namespace polynomial_coefficient_identity_l24_24487

theorem polynomial_coefficient_identity :
  ∀ (a a_1 a_2 a_3 a_4 : ℝ), (3 - x)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 →
    a - a_1 + a_2 - a_3 + a_4 = 256 :=
begin
  sorry
end

end polynomial_coefficient_identity_l24_24487


namespace min_radius_circle_cover_l24_24103

-- Define that four points (A, B, C, D) exist on the plane
variable (A B C D : ℝ × ℝ)

-- Define the condition that distance between any two of these four points is at most 1
def distance_le_one (p q : ℝ × ℝ) : Prop :=
  Real.dist p q ≤ 1

-- Define the condition for all four points
def all_distances_le_one : Prop :=
  distance_le_one A B ∧ distance_le_one A C ∧ distance_le_one A D ∧
  distance_le_one B C ∧ distance_le_one B D ∧ distance_le_one C D

-- Statement to prove minimum radius of circle covering four points is sqrt(3)/3
theorem min_radius_circle_cover :
  all_distances_le_one A B C D → (∃ (R : ℝ), (∀ (p : ℝ × ℝ) (q : ℝ × ℝ), distance_le_one p q → ∃ (c : ℝ × ℝ), Real.dist c p ≤ R ∧ Real.dist c q ≤ R) ∧ R = Real.sqrt 3 / 3) :=
sorry

end min_radius_circle_cover_l24_24103


namespace find_angle_l24_24748

section VectorAngle

variables {ℝ : Type*} [InnerProductSpace ℝ ℝ] [NormedSpace ℝ ℝ]

-- Given conditions
variables (a b d : ℝ)
variables (norm_a : ∥a∥ = 1) (norm_b : ∥b∥ = 1) (norm_d : ∥d∥ = 3)
variables (vec_eq : a × (a × d) - 2 • b = 0)

noncomputable def angle_formulas := {φ : ℝ // φ = Real.acos (sqrt 5 / 3) ∨ φ = Real.acos (-sqrt 5 / 3)}

theorem find_angle (a b d : ℝ) (norm_a : ∥a∥ = 1) (norm_b : ∥b∥ = 1) (norm_d : ∥d∥ = 3)
  (vec_eq : a × (a × d) - 2 • b = 0) :
  ∃ (φ : ℝ), φ ∈ angle_formulas :=
sorry

end VectorAngle

end find_angle_l24_24748


namespace cubic_eq_one_complex_solution_l24_24160

theorem cubic_eq_one_complex_solution (k : ℂ) :
  (∃ (x : ℂ), 8 * x^3 + 12 * x^2 + k * x + 1 = 0) ∧
  (∀ (x y z : ℂ), 8 * x^3 + 12 * x^2 + k * x + 1 = 0 → 8 * y^3 + 12 * y^2 + k * y + 1 = 0
    → 8 * z^3 + 12 * z^2 + k * z + 1 = 0 → x = y ∧ y = z) →
  k = 6 :=
sorry

end cubic_eq_one_complex_solution_l24_24160


namespace quadratic_trinomial_proof_l24_24363

-- Definitions based on conditions
def term1 := -x * y^2 / 5
def monom_x := x
def expr1 := -2^2 * x * y * z^2
def expr2 := x * y + x - 1

-- Theorem statement
theorem quadratic_trinomial_proof : 
  (∃ (a b c : ℤ), expr2 = a * x^2 + b * x + c) ∧ 
  degree expr2 = 2 ∧ 
  ∃ (t1 t2 t3 : expr2.term), expr2.num_terms = 3 := 
sorry

end quadratic_trinomial_proof_l24_24363


namespace general_term_of_sequence_l24_24502

theorem general_term_of_sequence (a : ℕ → ℝ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = (a n) ^ 2) :
  ∀ n : ℕ, n > 0 → a n = 3 ^ (2 ^ (n - 1)) :=
by
  intros n hn
  sorry

end general_term_of_sequence_l24_24502


namespace exp_form_l24_24923

noncomputable def theta (θ : ℝ) := θ > 0 ∧ θ < π

noncomputable def condition (x θ : ℝ) : Prop :=
  x + 1/x = 2 * Real.cos θ

theorem exp_form (n : ℕ) (θ x : ℝ) (h_theta : theta θ) (h_cond : condition x θ) : 
  x^n + 1/x^n = 2 * Real.cos(n * θ) :=
by sorry

end exp_form_l24_24923


namespace distance_from_A_to_directrix_of_C_l24_24977

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24977


namespace sum_eq_1_alternating_sum_eq_neg243_even_index_sum_eq_neg121_l24_24068

variable (x : ℝ)

-- Define the polynomial expansion
def polynomial_expansion (coefficients : List ℝ) : ℝ := 
  coefficients.get! 5 * x^5 + coefficients.get! 4 * x^4 + coefficients.get! 3 * x^3 + coefficients.get! 2 * x^2 + coefficients.get! 1 * x + coefficients.get! 0

-- Assume the given identity
axiom identity : ∀ x, polynomial_expansion x [a_5, a_4, a_3, a_2, a_1, a_0] = (2 * x - 1)^5

-- Prove the required equalities
theorem sum_eq_1 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 1 :=
by
  sorry

theorem alternating_sum_eq_neg243 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = -243 :=
by
  sorry

theorem even_index_sum_eq_neg121 : a_0 + a_2 + a_4 = -121 :=
by
  sorry

end sum_eq_1_alternating_sum_eq_neg243_even_index_sum_eq_neg121_l24_24068


namespace smallest_h_l24_24480

noncomputable def h (r : ℕ) : ℕ :=
  2 * r

theorem smallest_h (r : ℕ) (hr : 1 ≤ r) : ∀ (P : finset ℕ → finset (finset ℕ) → Prop),
  (∀ (s : finset ℕ) (c : finset (finset ℕ)), 
    (c.card = r) → 
    (∀ (a x y : ℕ), (a ≥ 0) → (1 ≤ x) → (x ≤ y) →
      ((∃ (r : finset ℕ) (H : r ∈ c), a + x ∈ r ∧ a + y ∈ r ∧ a + x + y ∈ r) → s.card = 2 * r)) →
  ∃ (s : finset ℕ) (c : finset (finset ℕ)), 
    (s = finset.range (h r).succ) ∧
    (c.card = r) ∧
    (∀ (a x y : ℕ), (a ≥ 0) → (1 ≤ x) → (x ≤ y) →
      ((∃ (r : finset ℕ) (H : r ∈ c), a + x ∈ r ∧ a + y ∈ r ∧ a + x + y ∈ r)) :=
by sorry

end smallest_h_l24_24480


namespace range_of_m_l24_24655

variables {a c m : ℝ}

def f (x : ℝ) : ℝ := a * x^2 - 2 * a * x + c

theorem range_of_m 
  (h_decreasing : ∀ x y, 0 ≤ x → x ≤ 1 → x ≤ y → y ≤ 1 → f(x) ≥ f(y))
  (h_f_m_le_f_0 : f(m) ≤ f(0)) : 0 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l24_24655


namespace problem_statement_l24_24083

variable (m : ℝ) (a b : ℝ)

-- Given conditions
def condition1 : Prop := 9^m = 10
def condition2 : Prop := a = 10^m - 11
def condition3 : Prop := b = 8^m - 9

-- Problem statement to prove
theorem problem_statement (h1 : condition1 m) (h2 : condition2 m a) (h3 : condition3 m b) : a > 0 ∧ 0 > b := 
sorry

end problem_statement_l24_24083


namespace find_point_T_l24_24043

-- Definition of our main points
def A := (0, 0)
def C := (3, 4)
def B := (3, 0)
def D := (0, 4)
def areaRectangle := 3 * 4

-- Proof that the coordinates of point T yield the desired triangle area.
theorem find_point_T (T : ℝ × ℝ) (hx : T.1 = D.1) (hy : T.2 = D.2 + 6) :
  let areaBDT := (1 / 2) * 4 * (T.2 - D.2) in
  areaBDT = areaRectangle → T = (0, 10) := 
by
  intro h
  unfold let areaBDT
  rw [hx, D.1, hy, D.2] at *
  sorry

end find_point_T_l24_24043


namespace perimeter_of_triangle_AF1B_l24_24521

noncomputable def perimeter_triangle_AF1B : ℝ :=
  let a := 6 in
  4 * a

theorem perimeter_of_triangle_AF1B 
  (a b : ℝ)
  (ha : a = 6)
  (h_ellipse : ∀ x y : ℝ, (x^2 / 36) + (y^2 / 16) = 1 → |x| <= 6 ∧ |y| <= 4)
  (h_major_axis : ∀ p : ℝ × ℝ, (p.fst^2 / 36) + (p.snd^2 / 16) = 1 → 
    let F1 := (-sqrt 20, 0)
    let F2 := (sqrt 20, 0) in
    |p.fst - (-sqrt 20)| + |p.fst - (sqrt 20)| = 12) :
  perimeter_triangle_AF1B = 24 := 
sorry

end perimeter_of_triangle_AF1B_l24_24521


namespace triangle_area_l24_24184

theorem triangle_area (BC : ℝ) (h : ℝ) (A : ½ * BC * h = 40) : true :=
by
  -- Given
  have BC_length : BC = 8 := by exact sorry
  have height : h = 10 := by exact sorry
  -- Therefore
  have area : ½ * 8 * 10 = 40 := by exact sorry
  
  triv -- signifies that we have proven the required statement (true)

end triangle_area_l24_24184


namespace calc_expr_equals_4_l24_24862

def calc_expr : ℝ :=
  (1/2)⁻¹ - (Real.sqrt 3) * Real.tan (Real.pi / 6) + (Real.pi - 2023)^0 + |(-2)|

theorem calc_expr_equals_4 : calc_expr = 4 := by
  -- Proof code goes here
  sorry

end calc_expr_equals_4_l24_24862


namespace ten_faucets_fill_50_gallon_tub_in_100_seconds_l24_24065

theorem ten_faucets_fill_50_gallon_tub_in_100_seconds :
  (five_faucets_fill_150_gallon_in_10_minutes: ∀ (faucets: ℕ), faucets = 5 → ∀ (gallons: ℕ), gallons = 150 → ∀ (minutes: ℕ), minutes = 10 → (gallons / minutes = 15)) →
  (time_to_fill_tub : ∀ (faucets: ℕ), faucets = 10 → ∀ (gallons: ℕ), gallons = 50 → ∀ (minutes : ℕ), minutes = 5/3 → (10 * minutes = 100 seconds)) := 
by
  intro five_faucets_fill_150_gallon_in_10_minutes
  intro time_to_fill_tub
  sorry

end ten_faucets_fill_50_gallon_tub_in_100_seconds_l24_24065


namespace value_of_b_l24_24215

theorem value_of_b {a b : ℝ} (h : (1 + complex.I) / (1 - complex.I) = a + b * complex.I) : b = 1 :=
sorry

end value_of_b_l24_24215


namespace planar_area_solution_l24_24467

def planar_area_problem (inequalities : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem planar_area_solution :
  planar_area_problem {p | (some_system_of_inequalities p)} = 36 :=
sorry

end planar_area_solution_l24_24467


namespace family_of_four_children_has_at_least_one_boy_and_one_girl_l24_24829

noncomputable section

def probability_at_least_one_boy_one_girl : ℚ :=
  1 - (1 / 16 + 1 / 16)

theorem family_of_four_children_has_at_least_one_boy_and_one_girl :
  probability_at_least_one_boy_one_girl = 7 / 8 := by
  sorry

end family_of_four_children_has_at_least_one_boy_and_one_girl_l24_24829


namespace find_largest_value_l24_24766

theorem find_largest_value
  (h1: 0 < Real.sin 2) (h2: Real.sin 2 < 1)
  (h3: Real.log 2 / Real.log (1 / 3) < 0)
  (h4: Real.log (1 / 3) / Real.log (1 / 2) > 1) :
  Real.log (1 / 3) / Real.log (1 / 2) > Real.sin 2 ∧ 
  Real.log (1 / 3) / Real.log (1 / 2) > Real.log 2 / Real.log (1 / 3) := by
  sorry

end find_largest_value_l24_24766


namespace find_intersection_l24_24540

namespace SetIntersectionProof

def P : Set ℕ := {1, 2, 3}
def Q : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}
def PQ_intersection := {1, 2}

theorem find_intersection : P ∩ Q = PQ_intersection := by
  sorry

end SetIntersectionProof

end find_intersection_l24_24540


namespace fraction_to_decimal_l24_24881

theorem fraction_to_decimal : (5 / 50) = 0.10 := 
by
  sorry

end fraction_to_decimal_l24_24881


namespace distance_from_point_to_directrix_l24_24980

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24980


namespace find_Q_l24_24701

/-- Given conditions -/
variables (P B F R C Q : ℕ) (t : ℕ) -- Assuming thickness t is a natural number
variables (h1 : P ≠ B) (h2 : P ≠ F) (h3 : P ≠ R) (h4 : P ≠ C) 
variables (h5 : B ≠ F) (h6 : B ≠ R) (h7 : B ≠ C) 
variables (h8 : F ≠ R) (h9 : F ≠ C) (h10 : R ≠ C)
variables (hP_pos : 0 < P) (hB_pos : 0 < B) (hF_pos : 0 < F) 
variables (hR_pos : 0 < R) (hC_pos : 0 < C) 

/-- Given that a shelf can be fully filled with different combinations of books -/
theorem find_Q (h1 : P * t = R * t + 2 * C * t) (h2 : Q * t = R * t + 2 * C * t) : Q = R + 2 * C :=
by sorry

end find_Q_l24_24701


namespace john_sold_books_on_wednesday_l24_24201

theorem john_sold_books_on_wednesday :
  (total_stock = 620) →
  (books_sold_mon = 50) →
  (books_sold_tue = 82) →
  (books_sold_thu = 48) →
  (books_sold_fri = 40) →
  (percentage_unsold = 54.83870967741935) →
  let books_unsold := (percentage_unsold / 100) * total_stock in
  let total_books_sold_mon_to_fri_excl_wed := books_sold_mon + books_sold_tue + books_sold_thu + books_sold_fri in
  let books_sold_wed := total_stock - (total_books_sold_mon_to_fri_excl_wed + books_unsold) in
  books_sold_wed = 60 :=
begin
  intros,
  let books_unsold := (percentage_unsold / 100) * total_stock,
  let total_books_sold_mon_to_fri_excl_wed := books_sold_mon + books_sold_tue + books_sold_thu + books_sold_fri,
  let books_sold_wed := total_stock - (total_books_sold_mon_to_fri_excl_wed + books_unsold),
  sorry
end

end john_sold_books_on_wednesday_l24_24201


namespace total_amount_spent_l24_24279

noncomputable def value_of_nickel : ℕ := 5
noncomputable def value_of_dime : ℕ := 10
noncomputable def initial_amount : ℕ := 250

def amount_spent_by_Pete (nickels_spent : ℕ) : ℕ :=
  nickels_spent * value_of_nickel

def amount_remaining_with_Raymond (dimes_left : ℕ) : ℕ :=
  dimes_left * value_of_dime

theorem total_amount_spent (nickels_spent : ℕ) (dimes_left : ℕ) :
  (amount_spent_by_Pete nickels_spent + 
   (initial_amount - amount_remaining_with_Raymond dimes_left)) = 200 :=
by
  sorry

end total_amount_spent_l24_24279


namespace factorial_fraction_value_l24_24851

theorem factorial_fraction_value :
  (15.factorial / (6.factorial * 9.factorial) = 5005) :=
by
  sorry

end factorial_fraction_value_l24_24851


namespace probability_at_least_one_boy_and_one_girl_l24_24831

theorem probability_at_least_one_boy_and_one_girl :
  let P := (1 - (1/16 + 1/16)) = 7 / 8,
  (∀ (N: ℕ), (N = 4) → 
    let prob_all_boys := (1 / N) ^ N,
    let prob_all_girls := (1 / N) ^ N,
    let prob_at_least_one_boy_and_one_girl := 1 - (prob_all_boys + prob_all_girls)
  in prob_at_least_one_boy_and_one_girl = P) :=
by
  sorry

end probability_at_least_one_boy_and_one_girl_l24_24831


namespace exists_five_digit_palindromic_divisible_by_5_count_five_digit_palindromic_numbers_divisible_by_5_l24_24251

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem exists_five_digit_palindromic_divisible_by_5 :
  ∃ n, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n := by
  -- Proof is omitted
  sorry

theorem count_five_digit_palindromic_numbers_divisible_by_5 :
  (finset.filter (λ n, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n) (finset.range 100000)).card = 100 := by
  -- Proof is omitted
  sorry

end exists_five_digit_palindromic_divisible_by_5_count_five_digit_palindromic_numbers_divisible_by_5_l24_24251


namespace cos_solution_count_l24_24559

theorem cos_solution_count :
  ∃ n : ℕ, n = 2 ∧ 0 ≤ x ∧ x < 360 → cos x = 0.45 :=
by
  sorry

end cos_solution_count_l24_24559


namespace distance_from_A_to_directrix_of_C_l24_24975

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24975


namespace arcsin_neg_one_eq_neg_half_pi_l24_24420

theorem arcsin_neg_one_eq_neg_half_pi :
  arcsin (-1) = - (Float.pi / 2) :=
by
  sorry

end arcsin_neg_one_eq_neg_half_pi_l24_24420


namespace P_sufficient_for_Q_P_not_necessary_for_Q_l24_24092

variable (x : ℝ)
def P : Prop := x >= 0
def Q : Prop := 2 * x + 1 / (2 * x + 1) >= 1

theorem P_sufficient_for_Q : P x -> Q x := 
by sorry

theorem P_not_necessary_for_Q : ¬ (Q x -> P x) := 
by sorry

end P_sufficient_for_Q_P_not_necessary_for_Q_l24_24092


namespace hyperbola_problem_l24_24134

open Real

variable {x y a b : ℝ}

def is_asymptote (a b : ℝ) : Prop :=
  b = 2 * a

def is_focus_parabola (x y : ℝ) : Prop :=
  (x, y) = (5, 0)

def is_focus_hyperbola (a b : ℝ) : Prop :=
  Real.sqrt (a^2 + b^2) = 5

def hyperbola_equation (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

theorem hyperbola_problem 
  (a > 0) (b > 0)
  (asymptote : is_asymptote a b)
  (focus : is_focus_parabola 5 0)
  (focus_hyp : is_focus_hyperbola a b) :
  hyperbola_equation x y (sqrt 5) (2 * sqrt 5) :=
sorry

end hyperbola_problem_l24_24134


namespace barbara_candies_l24_24405

theorem barbara_candies : (9 + 18) = 27 :=
by
  sorry

end barbara_candies_l24_24405


namespace depletion_rate_proof_l24_24804

noncomputable def depletion_rate (initial_value final_value : ℝ) (time : ℝ) : ℝ := 
(1 : ℝ) - real.sqrt (final_value / initial_value)

theorem depletion_rate_proof :
  depletion_rate 400 225 2 = 0.25 :=
by
  unfold depletion_rate
  simp
  have h : 225 / 400 = 0.5625 := by norm_num
  rw h
  have h_sqrt : real.sqrt 0.5625 = 0.75 := by norm_num
  rw h_sqrt
  norm_num

end depletion_rate_proof_l24_24804


namespace steven_time_l24_24461

noncomputable def plowing_time (flat_acres hilly_acres : ℕ) (plow_rate_flat plow_rate_hilly : ℕ) : ℕ :=
  (flat_acres / plow_rate_flat) + (hilly_acres.to_real / plow_rate_hilly.to_real).ceil_nat

noncomputable def mowing_time (total_acres : ℕ) (mowing_rate_rain : ℕ) : ℕ :=
  (total_acres.to_real / mowing_rate_rain.to_real).ceil_nat

noncomputable def total_time_to_work (flat_acres hilly_acres grass_acres : ℕ) 
  (plow_rate_flat plow_rate_hilly mowing_rate_rain : ℕ) : ℕ :=
  let plow_days := plowing_time flat_acres hilly_acres plow_rate_flat plow_rate_hilly
  let mow_days := mowing_time grass_acres mowing_rate_rain
  plow_days + mow_days

theorem steven_time (flat_acres hilly_acres grass_acres : ℕ)
  (plow_rate_flat plow_rate_hilly mowing_rate_rain rainy_days : ℕ) : 
  flat_acres = 40 → hilly_acres = 15 → grass_acres = 30 → 
  plow_rate_flat = 10 → plow_rate_hilly = 7 → mowing_rate_rain = 7.2 → 
  rainy_days = 5 → total_time_to_work flat_acres hilly_acres grass_acres plow_rate_flat plow_rate_hilly mowing_rate_rain = 12 := 
by 
  intros 
  sorry

end steven_time_l24_24461


namespace range_of_f_l24_24924

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem range_of_f :
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), (f x) ∈ Set.Icc (-1 : ℝ) 2) :=
by
  sorry

end range_of_f_l24_24924


namespace cyclic_quadrilateral_parallelogram_diagonals_l24_24606

theorem cyclic_quadrilateral_parallelogram_diagonals
  (A B C D O : Point) 
  (h_cyclic : cyclic_quad A B C D)
  (h_diag_int : intersect AC BD O)
  (h_AB_CD_parallel : parallel AB CD)
  (h_AB_CD_len : length AB = 8 ∧ length CD = 8)
  (h_AD_BC_len : length AD = 5 ∧ length BC = 5) : 
  parallelogram A B C D ∧ length (diagonal A C) = 8 ∧ length (diagonal B D) = 8 :=
sorry

end cyclic_quadrilateral_parallelogram_diagonals_l24_24606


namespace books_in_library_final_l24_24333

variable (initial_books : ℕ) (books_taken_out_tuesday : ℕ) 
          (books_returned_wednesday : ℕ) (books_taken_out_thursday : ℕ)

def books_left_in_library (initial_books books_taken_out_tuesday 
                          books_returned_wednesday books_taken_out_thursday : ℕ) : ℕ :=
  initial_books - books_taken_out_tuesday + books_returned_wednesday - books_taken_out_thursday

theorem books_in_library_final 
  (initial_books := 250) 
  (books_taken_out_tuesday := 120) 
  (books_returned_wednesday := 35) 
  (books_taken_out_thursday := 15) :
  books_left_in_library initial_books books_taken_out_tuesday 
                        books_returned_wednesday books_taken_out_thursday = 150 :=
by 
  sorry

end books_in_library_final_l24_24333


namespace distance_from_A_to_directrix_l24_24999

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l24_24999


namespace count_marble_pairs_l24_24752

-- Define conditions:
structure Marbles :=
(red : ℕ) (green : ℕ) (blue : ℕ) (yellow : ℕ) (white : ℕ)

def tomsMarbles : Marbles :=
  { red := 1, green := 1, blue := 1, yellow := 3, white := 2 }

-- Define a function to count pairs of marbles:
def count_pairs (m : Marbles) : ℕ :=
  -- Count pairs of identical marbles:
  (if m.yellow >= 2 then 1 else 0) + 
  (if m.white >= 2 then 1 else 0) +
  -- Count pairs of different colored marbles:
  (Nat.choose 5 2)

-- Theorem statement:
theorem count_marble_pairs : count_pairs tomsMarbles = 12 :=
  by
    sorry

end count_marble_pairs_l24_24752


namespace average_apples_sold_per_day_l24_24314

theorem average_apples_sold_per_day (boxes_sold : ℕ) (days : ℕ) (apples_per_box : ℕ) (H1 : boxes_sold = 12) (H2 : days = 4) (H3 : apples_per_box = 25) : (boxes_sold * apples_per_box) / days = 75 :=
by {
  -- Based on given conditions, the total apples sold is 12 * 25 = 300.
  -- Dividing by the number of days, 300 / 4 gives us 75 apples/day.
  -- The proof is omitted as instructed.
  sorry
}

end average_apples_sold_per_day_l24_24314


namespace sandy_age_l24_24688

variable (S M : ℕ)

-- Conditions
def condition1 := M = S + 12
def condition2 := S * 9 = M * 7

theorem sandy_age : condition1 S M → condition2 S M → S = 42 := by
  intros h1 h2
  sorry

end sandy_age_l24_24688


namespace lightning_distance_l24_24891

/--
Linus observed a flash of lightning and then heard the thunder 15 seconds later.
Given:
- speed of sound: 1088 feet/second
- 1 mile = 5280 feet
Prove that the distance from Linus to the lightning strike is 3.25 miles.
-/
theorem lightning_distance (time_seconds : ℕ) (speed_sound : ℕ) (feet_per_mile : ℕ) (distance_miles : ℚ) :
  time_seconds = 15 →
  speed_sound = 1088 →
  feet_per_mile = 5280 →
  distance_miles = 3.25 :=
by
  sorry

end lightning_distance_l24_24891


namespace radian_measure_of_240_degrees_l24_24322

theorem radian_measure_of_240_degrees : (240 * (π / 180) = 4 * π / 3) := by
  sorry

end radian_measure_of_240_degrees_l24_24322


namespace domain_transformation_l24_24116

theorem domain_transformation (f : ℝ → ℝ) :
  set.Icc (-1 : ℝ) (3 : ℝ) ⊆ set.preimage (λ x, 3 * x - 2) (set.Icc (1 / 3 : ℝ) (5 / 3 : ℝ)) :=
sorry

end domain_transformation_l24_24116


namespace no_such_integers_exist_l24_24679

theorem no_such_integers_exist :
  ¬ ∃ (a b c d : ℤ), 
    let P := λ x : ℤ, a * x^3 + b * x^2 + c * x + d in 
    P 19 = 1 ∧ P 62 = 2 :=
by sorry

end no_such_integers_exist_l24_24679


namespace repeating_decimal_product_l24_24036

theorem repeating_decimal_product (x y : ℚ) (h₁ : x = 8 / 99) (h₂ : y = 1 / 3) :
    x * y = 8 / 297 := by
  sorry

end repeating_decimal_product_l24_24036


namespace range_of_3a_minus_b_l24_24561

theorem range_of_3a_minus_b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3)
                             (h3 : 2 < a - b) (h4 : a - b < 4) :
    ∃ (x : ℝ), 3 ≤ x ∧ x ≤ 11 ∧ x = 3 * a - b :=
sorry

end range_of_3a_minus_b_l24_24561


namespace minimum_value_a_zero_range_of_a_l24_24499

def f (x : ℝ) (a : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then x^2 + 2 * x + a - 1 
  else if 0 < x ∧ x ≤ 3 then -x^2 + 2 * x - a 
  else 0

theorem minimum_value_a_zero : 
  ∃ x : ℝ, x ∈ Set.Icc (-3 : ℝ) (0 : ℝ) ∪ Set.Ioc (0 : ℝ) (3 : ℝ) ∧ f x 0 = -3 :=
by
  sorry

theorem range_of_a : 
  ∀ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ) ∪ Set.Ioc (0 : ℝ) (3 : ℝ), f x a ≤ |x| ↔ (¼ : ℝ) ≤ a ∧ a ≤ 1 :=
by
  sorry

end minimum_value_a_zero_range_of_a_l24_24499


namespace ms_brown_expects_8100_tulips_l24_24268

def steps_length := 3
def width_steps := 18
def height_steps := 25
def tulips_per_sqft := 2

def width_feet := width_steps * steps_length
def height_feet := height_steps * steps_length
def area_feet := width_feet * height_feet
def expected_tulips := area_feet * tulips_per_sqft

theorem ms_brown_expects_8100_tulips :
  expected_tulips = 8100 := by
  sorry

end ms_brown_expects_8100_tulips_l24_24268


namespace triangle_count_in_figure_l24_24554

theorem triangle_count_in_figure :
  let rectangle_triangles (h w : ℕ) (v_divs h_divs : ℕ) :=
    let small_rects := v_divs * h_divs in
    let small_triangles := small_rects * 4 in
    let intermediate_vertical_pairs := (v_divs - 1) * h_divs * 2 in
    let intermediate_horizontal_pairs := (h_divs - 1) * v_divs * 2 in
    let large_triangles := 2 in
    small_triangles + intermediate_vertical_pairs + intermediate_horizontal_pairs + large_triangles 
  in
  rectangle_triangles 30 40 3 2 = 46 :=
by
  sorry

end triangle_count_in_figure_l24_24554


namespace problem_l24_24075

theorem problem (m a b : ℝ) (h₀ : 9 ^ m = 10) (h₁ : a = 10 ^ m - 11) (h₂ : b = 8 ^ m - 9) :
  a > 0 ∧ 0 > b := 
sorry

end problem_l24_24075


namespace probability_at_least_one_boy_and_girl_l24_24841
-- Necessary imports

-- Defining the probability problem in Lean 4
theorem probability_at_least_one_boy_and_girl (n : ℕ) (hn : n = 4)
    (p : ℚ) (hp : p = 1 / 2) :
    let prob_all_same := (p ^ n) + (p ^ n) in
    (1 - prob_all_same) = 7 / 8 := by
  -- Include the proof steps here
  sorry

end probability_at_least_one_boy_and_girl_l24_24841


namespace percentage_increase_second_year_is_20_l24_24718

noncomputable def find_percentage_increase_second_year : ℕ :=
  let P₀ := 1000
  let P₁ := P₀ + (10 * P₀) / 100
  let Pf := 1320
  let P := (Pf - P₁) * 100 / P₁
  P

theorem percentage_increase_second_year_is_20 :
  find_percentage_increase_second_year = 20 :=
by
  sorry

end percentage_increase_second_year_is_20_l24_24718


namespace probability_jack_queen_king_l24_24777

theorem probability_jack_queen_king (deck_size jacks queens kings : ℕ)
  (h1 : deck_size = 52) (h2 : jacks = 4) (h3 : queens = 4) (h4 : kings = 4) :
  probability (jacks + queens + kings) deck_size = 3 / 13 := by
sorry

end probability_jack_queen_king_l24_24777


namespace number_of_irrational_numbers_in_set_l24_24400

theorem number_of_irrational_numbers_in_set :
  let s := {-3 : ℝ, (22/7 : ℝ), 3.14, -3 * Real.pi, (Real.sqrt 10 - Real.sqrt 10 /3)} in
  (s.filter (λ x, ¬ ∃ a b : ℚ, (↑a : ℝ) = x * (↑b : ℝ))).card = 2 :=
by
  sorry

end number_of_irrational_numbers_in_set_l24_24400


namespace pounds_added_l24_24202

-- Definitions based on conditions
def initial_weight : ℝ := 5
def weight_increase_percent : ℝ := 1.5  -- 150% increase
def final_weight : ℝ := 28

-- Statement to prove
theorem pounds_added (w_initial w_final w_percent_added : ℝ) (h_initial: w_initial = 5) (h_final: w_final = 28)
(h_percent: w_percent_added = 1.5) :
  w_final - w_initial = 23 := 
by
  sorry

end pounds_added_l24_24202


namespace find_y_value_l24_24776

theorem find_y_value : (12 ^ 3 * 6 ^ 4) / 432 = 5184 := by
  sorry

end find_y_value_l24_24776


namespace num_divisors_of_M_l24_24634

theorem num_divisors_of_M (M : ℕ) (hM : M = ∑ n in (finset.filter (λ n, n ∣ 2016^2 ∧ 2016 ∣ n^2) (finset.range (2016^2 + 1))), n) :
  (nat.divisors_count M) = 360 :=
by
  -- We're provided the conditions and asked to prove the number of divisors.
  sorry

end num_divisors_of_M_l24_24634


namespace minimal_a4_l24_24932

-- Given conditions
variables {a_1 a_3 a_4 d : ℝ}
variable S : ℕ → ℝ

-- Define the sum of the first 'n' terms of the arithmetic sequence
def S (n : ℕ) : ℝ := n / 2 * (2 * a_1 + (n - 1) * d)

-- Provided conditions
axiom S4_condition : S 4 ≤ 4
axiom S5_condition : S 5 ≥ 15

-- The problem statement to prove
theorem minimal_a4 : a_4 ≥ 7 :=
sorry

end minimal_a4_l24_24932


namespace general_term_formula_l24_24050

-- Definition of the sequence that alternates between 1 and -1
def sequence (n : ℕ) : ℤ := (-1)^(n + 1)

-- Proof statement to show that this definition satisfies the given conditions
theorem general_term_formula (n : ℕ) : 
  ((n % 2 = 1) → (sequence n = 1)) ∧ ((n % 2 = 0) → (sequence n = -1)) :=
  sorry

end general_term_formula_l24_24050


namespace tan_product_l24_24429

theorem tan_product :
  (Real.tan (Real.pi / 8)) * (Real.tan (3 * Real.pi / 8)) * (Real.tan (5 * Real.pi / 8)) = 1 :=
sorry

end tan_product_l24_24429


namespace cos_double_angle_l24_24152

variable {α : ℝ}

theorem cos_double_angle (h1 : (Real.tan α - (1 / Real.tan α) = 3 / 2)) (h2 : (α > π / 4) ∧ (α < π / 2)) :
  Real.cos (2 * α) = -3 / 5 := 
sorry

end cos_double_angle_l24_24152


namespace all_n_geq_3_sum_distinct_fib_l24_24140

noncomputable def fibonacci : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

theorem all_n_geq_3_sum_distinct_fib (n : ℕ) (h : n ≥ 3) :
  ∃ (m : ℕ) (indices : Fin m → ℕ), (Function.Injective indices) ∧ (n = (Finset.range m).sum (λ x, fibonacci (indices x))) := 
sorry

end all_n_geq_3_sum_distinct_fib_l24_24140


namespace molecular_weight_of_CH3CH2CHO_HCl_reaction_l24_24887

def molecular_weight (compound : Array (ℕ × Float)) : Float :=
  compound.foldl (fun acc (num, weight) => acc + (num * weight)) 0

def CH3CH2CHO := #[⟨3, 12.01⟩, ⟨6, 1.008⟩, ⟨1, 16.00⟩] -- propanal
def CH3CH(OH)CH2Cl := #[⟨3, 12.01⟩, ⟨7, 1.008⟩, ⟨1, 16.00⟩, ⟨1, 35.45⟩] -- 2-chloropropanol

theorem molecular_weight_of_CH3CH2CHO_HCl_reaction 
    (MW_product : Float) 
    (h : MW_product = molecular_weight CH3CH(OH)CH2Cl) : 
    MW_product = 94.536 :=
by
    sorry

end molecular_weight_of_CH3CH2CHO_HCl_reaction_l24_24887


namespace factorial_fraction_eq_l24_24855

theorem factorial_fraction_eq :
  (15.factorial / (6.factorial * 9.factorial) = 5005) := 
sorry

end factorial_fraction_eq_l24_24855


namespace problem_statement_l24_24071

theorem problem_statement (m a b : ℝ) (h0 : 9^m = 10) (h1 : a = 10^m - 11) (h2 : b = 8^m - 9) : a > 0 ∧ 0 > b :=
by
  sorry

end problem_statement_l24_24071


namespace min_value_of_linear_function_l24_24654

theorem min_value_of_linear_function :
  ∃ (y_min : ℝ), y_min = Inf (set.image (λ x, -x + 3) (set.Icc (0 : ℝ) 3)) ∧ y_min = 0 :=
begin
  sorry
end

end min_value_of_linear_function_l24_24654


namespace perfect_apples_l24_24591

theorem perfect_apples (total_apples : ℕ) (fraction_small fraction_unripe : ℚ) 
  (h_total_apples : total_apples = 30) 
  (h_fraction_small : fraction_small = 1 / 6) 
  (h_fraction_unripe : fraction_unripe = 1 / 3) : 
  total_apples * (1 - fraction_small - fraction_unripe) = 15 :=
  by
  rw [h_total_apples, h_fraction_small, h_fraction_unripe]
  have h : 1 - (1/6 + 1/3) = 1/2 := by norm_num
  rw h
  norm_num

end perfect_apples_l24_24591


namespace circles_intersect_l24_24719

def circle1 := { x : ℝ × ℝ | (x.1 - 1)^2 + (x.2 + 2)^2 = 1 }
def circle2 := { x : ℝ × ℝ | (x.1 - 2)^2 + (x.2 + 1)^2 = 1 / 4 }

theorem circles_intersect :
  ∃ x : ℝ × ℝ, x ∈ circle1 ∧ x ∈ circle2 :=
sorry

end circles_intersect_l24_24719


namespace tangent_product_l24_24439

theorem tangent_product (n θ : ℝ) 
  (h1 : ∀ n θ, tan (n * θ) = (sin (n * θ)) / (cos (n * θ)))
  (h2 : tan 8 θ = (8 * tan θ - 56 * (tan θ) ^ 3 + 56 * (tan θ) ^ 5 - 8 * (tan θ) ^ 7) / 
                  (1 - 28 * (tan θ) ^ 2 + 70 * (tan θ) ^ 4 - 28 * (tan θ) ^ 6))
  (h3 : tan (8 * (π / 8)) = 0)
  (h4 : tan (8 * (3 * π / 8)) = 0)
  (h5 : tan (8 * (5 * π / 8)) = 0) :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = 2 * sqrt 2 :=
sorry

end tangent_product_l24_24439


namespace no_unhappy_days_l24_24726

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end no_unhappy_days_l24_24726


namespace categorize_numbers_l24_24044

namespace NumberSets

-- Define the numbers
def eight := 8 : ℤ
def neg_one := -1 : ℤ
def neg_four_tenths := -2 / 5 : ℚ
def three_fifths := 3 / 5 : ℚ
def zero := 0 : ℚ
def one_third := 1 / 3 : ℚ
def neg_one_three_sevenths := -10 / 7 : ℚ
def neg_neg_five := 5 : ℤ
def neg_abs_neg_twenty_sevenths := -20 / 7 : ℚ

-- Define predicates for the sets
def is_positive (x : ℚ) : Prop := x > 0
def is_negative (x : ℚ) : Prop := x < 0
def is_integer (x : ℚ) : Prop := ∃ (z : ℤ), x = z
def is_fraction (x : ℚ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_non_negative_rational (x : ℚ) : Prop := x ≥ 0

-- Theorem to prove
theorem categorize_numbers :
  {x | is_positive x} = {eight, three_fifths, one_third, neg_neg_five} ∧
  {x | is_negative x} = {neg_one, neg_four_tenths, neg_one_three_sevenths, neg_abs_neg_twenty_sevenths} ∧
  {x | is_integer x} = {eight, neg_one, zero, neg_neg_five} ∧
  {x | is_fraction x} = {neg_four_tenths, three_fifths, one_third, neg_one_three_sevenths, neg_abs_neg_twenty_sevenths} ∧
  {x | is_non_negative_rational x} = {eight, three_fifths, zero, one_third, neg_neg_five} :=
  by
    sorry

end NumberSets

end categorize_numbers_l24_24044


namespace part1_part2_l24_24650

-- Input data for the sequence and the condition imposed on pairs
variables {α : Type*} [linear_order α] [has_le α] [has_mul α] [has_add α] [has_sub α]
variables (a : ℕ → α) (n : ℕ)

-- Conditions
axiom strictly_increasing (h : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j)
axiom integer_fraction (h : ∀ i j, 1 ≤ i → i < j → j ≤ n → is_integer (a i / (a j - a i)))

-- Proof statements
theorem part1 (m : ℕ) (h₁ : 1 ≤ m) (h₂ : m ≤ n - 1): m * (a (m + 1)) ≤(m + 1) * (a m) :=
sorry

theorem part2 (i j : ℕ) (h₁ : 1 ≤ i) (h₂ : i < j) (h₃ : j ≤ n): i * (a j) ≤ j * (a i) :=
sorry

end part1_part2_l24_24650


namespace sufficient_wrapping_paper_l24_24624

theorem sufficient_wrapping_paper (a b c : ℝ) : 
  let side_length := (a + b + 2 * c) / Real.sqrt 2 in
  let surface_area := 2 * ((a * b) + (a * c) + (b * c)) in
  side_length^2 >= surface_area :=
by
  sorry

end sufficient_wrapping_paper_l24_24624


namespace number_of_sevens_l24_24780

theorem number_of_sevens (n : ℕ) : ∃ (k : ℕ), k < n ∧ ∃ (f : ℕ → ℕ), (∀ i, f i = 7) ∧ (7 * ((77 - 7) / 7) ^ 14 - 1) / (7 + (7 + 7)/7) = 7^(f k) :=
by sorry

end number_of_sevens_l24_24780


namespace distance_from_A_to_directrix_of_C_l24_24966

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24966


namespace determine_a_l24_24542
open Set

-- Given Condition Definitions
def U : Set ℕ := {1, 3, 5, 7}
def M (a : ℤ) : Set ℕ := {1, Int.natAbs (a - 5)} -- using ℤ for a and natAbs to get |a - 5|

-- Problem statement
theorem determine_a (a : ℤ) (hM_subset_U : M a ⊆ U) (h_complement : U \ M a = {5, 7}) : a = 2 ∨ a = 8 :=
by sorry

end determine_a_l24_24542


namespace locus_of_centers_l24_24497

-- Definitions
variables {O P : Point} {radius : ℝ}
variables (A B : Point) (AB : Line)
variable (circle_center : Point)
noncomputable def circle := { x : Point // dist x O = radius }
noncomputable def circumscribing_circle (A B P : Point) := { center : Point // ∀ C : Point, (dist C A = dist center A ∧ dist C B = dist center B ∧ dist C P = dist center P) }

-- Conditions
axiom CircleWithCenterO (circle : circle)
axiom PointOutsideCircle (P : Point) (h : dist P O > radius)
axiom DiameterOfCircle (A B : Point) (hAB : dist A O = dist B O = radius)
axiom VariableDiameter (AB : Line) : LineOnCircle A B circle
axiom ConnectionsToP (A B P : Point)

-- Conclusion / Theorem
theorem locus_of_centers (O P : Point) (radius : ℝ) (A B : Point) (circle : circle)
  (circum_circle : circumscribing_circle A B P) : 
  ∃ l : Line, (IsPerpendicular l (LineThroughPoints O P)) ∧ (∀ center : Point, center ∈ circum_circle → center ∈ l) :=
sorry


end locus_of_centers_l24_24497


namespace parametric_to_standard_l24_24454

theorem parametric_to_standard (α : ℝ) :
    let x := √3 * Real.cos α + 2
    let y := √3 * Real.sin α - 3
    (x - 2) ^ 2 + (y + 3) ^ 2 = 3 :=
by
  sorry

end parametric_to_standard_l24_24454


namespace exists_set_Sn_l24_24284

theorem exists_set_Sn (n : ℕ) (hn : n ≥ 2) : 
  ∃ S : Finset ℤ, S.card = n ∧ 
  (∀ (a b ∈ S), a ≠ b → (a - b)^2 ∣ a * b) :=
sorry

end exists_set_Sn_l24_24284


namespace question_true_if_pq_false_l24_24113

theorem question_true_if_pq_false (p q : Prop) (hp : p = False) (hq : q = False) : 
  (p ∨ ¬q) = True :=
by
  rw [hp, hq]
  simp only [false_or, eq_self_iff_true, not_false_iff]
  exact True.intro

end question_true_if_pq_false_l24_24113


namespace arcsin_neg_one_eq_neg_pi_div_two_l24_24414

theorem arcsin_neg_one_eq_neg_pi_div_two : arcsin (-1) = - (Real.pi / 2) := sorry

end arcsin_neg_one_eq_neg_pi_div_two_l24_24414


namespace problem_statement_l24_24089

theorem problem_statement (m : ℝ) (a b : ℝ) 
  (h1 : 9^m = 10) (h2 : a = 10^m - 11) (h3 : b = 8^m - 9) : 
  a > 0 ∧ 0 > b :=
sorry

end problem_statement_l24_24089


namespace example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l24_24249

-- Definition of a five-digit palindromic number
def is_palindromic (n : ℕ) : Prop := let s := n.to_string in s = s.reverse

-- Definition of a five-digit number
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

-- Part (a): Prove that 51715 is a five-digit palindromic number and is divisible by 5
theorem example_palindromic_divisible_by_5 :
  is_five_digit 51715 ∧ is_palindromic 51715 ∧ 51715 % 5 = 0 :=
by sorry

-- Part (b): Prove that there are exactly 100 five-digit palindromic numbers divisible by 5
theorem count_palindromic_divisible_by_5 :
  (finset.filter (λ n, is_five_digit n ∧ is_palindromic n ∧ n % 5 = 0) 
    (finset.range 100000)).card = 100 :=
by sorry

end example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l24_24249


namespace cookie_batches_l24_24843

theorem cookie_batches (total_students attendance_percentage cookies_per_student cookies_per_batch : ℕ) 
                       (total_students_eq : total_students = 150)
                       (attendance_percentage_eq : attendance_percentage = 60)
                       (cookies_per_student_eq : cookies_per_student = 3)
                       (cookies_per_batch_eq : cookies_per_batch = 20) :
  let attending_students := total_students * attendance_percentage / 100
  let total_cookies := attending_students * cookies_per_student
  let batches := (total_cookies + cookies_per_batch - 1) / cookies_per_batch -- Rounding up
  batches = 14 :=
by
  rw [total_students_eq, attendance_percentage_eq, cookies_per_student_eq, cookies_per_batch_eq]
  let attending_students := 150 * 60 / 100
  let total_cookies := attending_students * 3
  have h1 : attending_students = 90, by norm_num
  rw h1 at total_cookies
  have h2 : total_cookies = 270, by norm_num
  rw h2
  let batches := (270 + 20 - 1) / 20
  have h3 : batches = 14, by norm_num
  exact h3

end cookie_batches_l24_24843


namespace lattice_points_slope_interval_length_l24_24639

theorem lattice_points_slope_interval_length :
  let S : finset (ℤ × ℤ) := finset.product (finset.range 30) (finset.range 30) in
  (∃ a b : ℤ, (∀ n : ℤ, 1 ≤ n ∧ n ≤ 30) 
  ∧ gcd a b = 1 ∧ (exists m: ℚ, (m * 30).nat_abs * (m ≤ n) = 300) 
  ∧ ((b - a) = 1) 
  ∧ a + b = 85 := sorry

end lattice_points_slope_interval_length_l24_24639


namespace average_of_last_six_numbers_l24_24303

theorem average_of_last_six_numbers 
  (avg_11 : real)
  (avg_6_first : real)
  (sixth_number : real) 
  (total_nums : ℕ)
  (first_6 : ℕ) 
  (last_6 : ℕ) 
  (h1 : avg_11 = 10.7)
  (h2 : avg_6_first = 10.5)
  (h3 : sixth_number = 13.700000000000017)
  (h4 : total_nums = 11)
  (h5 : first_6 = 6)
  (h6 : last_6 = 6) :
  (10.7 * 11 - (10.5 * 6 - 13.7)) / 6 = 11.4 := 
by 
  sorry

end average_of_last_six_numbers_l24_24303


namespace chuck_play_area_l24_24008

theorem chuck_play_area :
  let shed_width : ℝ := 2
      shed_length : ℝ := 4
      leash_length : ℝ := 4
      larger_arc_area : ℝ := (3 / 4) * (Real.pi * (leash_length ^ 2))
      smaller_arc_area : ℝ := (1 / 4) * (Real.pi * ((leash_length - shed_length) ^ 2))
  in 
  larger_arc_area + smaller_arc_area = 13 * Real.pi := 
by
  sorry

end chuck_play_area_l24_24008


namespace repeating_decimal_multiplication_l24_24037

theorem repeating_decimal_multiplication :
  (0.0808080808 : ℝ) * (0.3333333333 : ℝ) = (8 / 297) := by
  sorry

end repeating_decimal_multiplication_l24_24037


namespace measure_angle_EHD_l24_24181

variable {EFGH : Type} [Parallelogram EFGH]

-- Assume angle EFG (α) and angle FGH (β)
variables (α β : ℝ)

-- Condition 1: EFGH is a parallelogram
class Parallelogram (EFGH : Type) := 
  (angle : EFGH → ℝ)
  (consecutiveAnglesSupplementary : ∀ (x y : EFGH), x ≠ y → angle x + angle y = 180)
  (oppositeAnglesEqual : ∀ (x y : EFGH), x ≠ y → angle x = angle y)

-- Condition 2: Measure of angle EFG is 2 times the measure of angle FGH
def measure_condition (α β : ℝ) : Prop :=
  α = 2 * β

-- Goal: Prove that the measure of angle EHD is 60 degrees
theorem measure_angle_EHD (h₁ : measure_condition α β) (h₂ : (Parallelogram.angle ⟨α, β⟩) = Parallelogram.angle ⟨α, β⟩)
        : Parallelogram.angle EFGH = 60 :=
  sorry

end measure_angle_EHD_l24_24181


namespace statement_B_correct_statement_C_correct_l24_24642

def f (x : Real) : Real := cos x

theorem statement_B_correct (x : Real) :
  (deriv (λ x, (f x) / x)) x = (-x * sin x - cos x) / (x^2) := sorry

theorem statement_C_correct :
  TangentLineCos (π / 2) :=
sorry

end statement_B_correct_statement_C_correct_l24_24642


namespace james_work_hours_l24_24198

def total_cost_without_tax : ℝ :=
  20 * 5 + 15 * 4 + 25 * 3.5 + 60 * 1.5 + 20 * 2 + 5 * 6

def tax : ℝ :=
  total_cost_without_tax * 0.07

def total_cost_with_tax : ℝ :=
  total_cost_without_tax + tax

def total_cost_with_cleaning : ℝ :=
  total_cost_with_tax + 15

def interest : ℝ :=
  total_cost_with_cleaning * 0.05

def total_cost_with_interest : ℝ :=
  total_cost_with_cleaning + interest

def total_cost_with_penalty : ℝ :=
  total_cost_with_interest + 50

def janitorial_overtime : ℝ :=
  10 * 10 * 1.5

def total_final_cost : ℝ :=
  total_cost_with_penalty + janitorial_overtime

def hours_to_work : ℕ :=
  Int.ceil (total_final_cost / 8 : ℝ).to_real

theorem james_work_hours :
  hours_to_work = 85 :=
by
  sorry

end james_work_hours_l24_24198


namespace smallest_positive_period_l24_24157

theorem smallest_positive_period (a b : ℝ) :
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f x = f (x + T) :=
begin
  let f := λ x, a * cos x ^ 2 + b * sin x + tan x,
  cases exists_or_eq_zero b with hb0,
  { existsi (2 * pi),
    split,
    linarith [pi_pos],
    intro x,
    sorry },
  { existsi pi,
    split,
    linarith [pi_pos],
    intro x,
    sorry
  }
end

end smallest_positive_period_l24_24157


namespace distance_from_point_to_directrix_l24_24983

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24983


namespace factorize1_factorize2_factorize3_l24_24894

theorem factorize1 (x : ℝ) : x^3 + 6 * x^2 + 9 * x = x * (x + 3)^2 := 
  sorry

theorem factorize2 (x y : ℝ) : 16 * x^2 - 9 * y^2 = (4 * x - 3 * y) * (4 * x + 3 * y) := 
  sorry

theorem factorize3 (x y : ℝ) : (3 * x + y)^2 - (x - 3 * y) * (3 * x + y) = 2 * (3 * x + y) * (x + 2 * y) := 
  sorry

end factorize1_factorize2_factorize3_l24_24894


namespace sum_of_roots_eq_2023_l24_24061

/-!
  ## Statement
  Given the equation
  √(2 * x^2 - 2024 * x + 1023131) + √(3 * x^2 - 2025 * x + 1023132) + √(4 * x^2 - 2026 * x + 1023133)
  = √(x^2 - x + 1) + √(2 * x^2 - 2 * x + 2) + √(3 * x^2 - 3 * x + 3)
  Prove that the sum of all roots of this equation is 2023.
-/
theorem sum_of_roots_eq_2023 
  (h : ∀ x, 
    (Real.sqrt (2 * x^2 - 2024 * x + 1023131) 
     + Real.sqrt (3 * x^2 - 2025 * x + 1023132)
     + Real.sqrt (4 * x^2 - 2026 * x + 1023133))
    = (Real.sqrt (x^2 - x + 1) 
     + Real.sqrt (2 * x^2 - 2 * x + 2) 
     + Real.sqrt (3 * x^2 - 3 * x + 3)))
  : ∃ roots, roots.sum = 2023 :=
by 
  sorry

end sum_of_roots_eq_2023_l24_24061


namespace family_of_four_children_has_at_least_one_boy_and_one_girl_l24_24827

noncomputable section

def probability_at_least_one_boy_one_girl : ℚ :=
  1 - (1 / 16 + 1 / 16)

theorem family_of_four_children_has_at_least_one_boy_and_one_girl :
  probability_at_least_one_boy_one_girl = 7 / 8 := by
  sorry

end family_of_four_children_has_at_least_one_boy_and_one_girl_l24_24827


namespace tan_identity_l24_24407

noncomputable def tan_add (a b : ℝ) : ℝ :=
  (Mathlib.Function.Real.Tangent a + Mathlib.Function.Real.Tangent b) /
  (1 - Mathlib.Function.Real.Tangent a * Mathlib.Function.Real.Tangent b)

theorem tan_identity :
  (∀ (a b : ℝ), tan_add a b = Mathlib.Function.Real.Tangent (a + b)) →
  Mathlib.Function.Real.Tangent 30 = (√3) / 3 →
  (Mathlib.Function.Real.Tangent 10 * Mathlib.Function.Real.Tangent 20 + √3 * (Mathlib.Function.Real.Tangent 10 + Mathlib.Function.Real.Tangent 20) = 1) :=
by
  intros h1 h2
  sorry

end tan_identity_l24_24407


namespace log_10_of_3_bounds_l24_24541

theorem log_10_of_3_bounds :
  2 / 5 < real.log 3 / real.log 10 ∧ real.log 3 / real.log 10 < 1 / 2 :=
by
  have h1 : 3^5 = 243 := by norm_num
  have h2 : 3^6 = 729 := by norm_num
  have h3 : 2^8 = 256 := by norm_num
  have h4 : 2^10 = 1024 := by norm_num
  have h5 : 10^2 = 100 := by norm_num
  have h6 : 10^3 = 1000 := by norm_num
  sorry

end log_10_of_3_bounds_l24_24541


namespace find_solutions_l24_24464

theorem find_solutions (x : ℝ) :
  (x^2 - 12 * (Int.floor x) + 20 = 0) →
  (x ∈ {2, 2 * Real.sqrt 19, 2 * Real.sqrt 22, 10}) :=
begin
  sorry
end

end find_solutions_l24_24464


namespace intersection_sets_l24_24142

noncomputable def P : set ℤ := { x | ∃ y : ℝ, y = real.sqrt (1 - (x^2)) }
noncomputable def Q : set ℝ := { y | ∃ x : ℝ, y = real.cos x }

theorem intersection_sets (P Q : set ℝ) : 
  P ∩ Q = P :=
sorry

end intersection_sets_l24_24142


namespace larger_number_of_hcf_23_lcm_factors_13_15_l24_24711

theorem larger_number_of_hcf_23_lcm_factors_13_15 :
  ∃ A B, (Nat.gcd A B = 23) ∧ (A * B = 23 * 13 * 15) ∧ (A = 345 ∨ B = 345) := sorry

end larger_number_of_hcf_23_lcm_factors_13_15_l24_24711


namespace solve_system_of_equations_l24_24297

theorem solve_system_of_equations :
  ∃ x : ℕ → ℝ,
  (∀ i : ℕ, i < 100 → x i > 0) ∧
  (x 0 + 1 / x 1 = 4) ∧
  (x 1 + 1 / x 2 = 1) ∧
  (x 2 + 1 / x 0 = 4) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i < 99 → x (2 * i + 1) + 1 / x (2 * i + 2) = 1) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i < 99 → x (2 * i + 2) + 1 / x (2 * i + 3) = 4) ∧
  (x 99 + 1 / x 0 = 1) ∧
  (∀ i : ℕ, i < 50 → x (2 * i) = 2) ∧
  (∀ i : ℕ, i < 50 → x (2 * i + 1) = 1 / 2) :=
sorry

end solve_system_of_equations_l24_24297


namespace greatest_root_of_g_is_sqrt15_over_5_l24_24051

noncomputable def g (x : ℝ) := 20 * x^4 - 18 * x^2 + 3

theorem greatest_root_of_g_is_sqrt15_over_5 :
  ∃ (x : ℝ), g x = 0 ∧ ∀ y : ℝ, g y = 0 → y ≤ x ∧ x = sqrt (15) / 5 :=
begin
  sorry
end

end greatest_root_of_g_is_sqrt15_over_5_l24_24051


namespace a5_eq_11_l24_24609

variable (a : ℕ → ℚ) (S : ℕ → ℚ)
variable (n : ℕ) (d : ℚ) (a1 : ℚ)

-- The definitions as given in the conditions
def arithmetic_sequence (a : ℕ → ℚ) (a1 : ℚ) (d : ℚ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

def sum_of_terms (S : ℕ → ℚ) (a1 : ℚ) (d : ℚ) : Prop :=
  ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)

-- Given conditions
def cond1 (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  a 3 + S 3 = 22

def cond2 (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  a 4 - S 4 = -15

-- The statement to prove
theorem a5_eq_11 (a : ℕ → ℚ) (S : ℕ → ℚ) (a1 : ℚ) (d : ℚ)
  (h_arith : arithmetic_sequence a a1 d)
  (h_sum : sum_of_terms S a1 d)
  (h1 : cond1 a S)
  (h2 : cond2 a S) : a 5 = 11 := by
  sorry

end a5_eq_11_l24_24609


namespace factorize_quadratic_l24_24041

theorem factorize_quadratic (x : ℝ) : x^2 - 2 * x = x * (x - 2) :=
sorry

end factorize_quadratic_l24_24041


namespace area_of_region_l24_24897

def equation (x y : ℝ) : Prop := |x - 50| + |y| = |x / 5|

theorem area_of_region :
  ∃ S : set (ℝ × ℝ),
  (∀ p ∈ S, equation p.1 p.2) ∧
  (measurable_set S) ∧
  (measure_theory.measure.region (set.univ : set S).prod ) = 208.3 := sorry

end area_of_region_l24_24897


namespace evaluate_expression_correct_l24_24034

noncomputable def evaluate_expression (x : ℝ) : ℝ :=
  3 - (-3) ^ (-2 / 3)

theorem evaluate_expression_correct :
  evaluate_expression (-3) = 3 - 1 / real.cbrt 9 :=
by
  sorry

end evaluate_expression_correct_l24_24034


namespace geometric_sequence_sum_of_sequence_l24_24538

variables {a₁ b₁ : ℝ} (a : ℕ → ℝ) (b : ℕ → ℝ)

noncomputable def a_seq (n : ℕ) : ℝ :=
if n = 0 then a₁ else 3/4 * a (n - 1) - b (n - 1) / 2

noncomputable def b_seq (n : ℕ) : ℝ :=
if n = 0 then b₁ else (3/2 * b (n - 1) - a (n - 1) / 4) / 2

-- Main goal 1: Prove that a_n + 2 * b_n is a geometric sequence
theorem geometric_sequence (h₁ : a₁ + 2 * b₁ = 1) (h₂ : ∀ n, a (n + 1) = a_seq a b n) (h₃ : ∀ n, b (n + 1) = b_seq a b n) :
  ∀ n, a n + 2 * b n = 1 / 2 ^ n := sorry

-- Main goal 2: Find sum of first n terms of a_n under given conditions
theorem sum_of_sequence (n : ℕ) (h₁ : a₁ + 2 * b₁ = 1)
  (cond : (a₁ - 2 * b₁ = 1) ∨ (b_seq 1 = -1 / 8) ∨ (a_seq 1 - 2 * b_seq 1 = 1))
  (h₂ : ∀ n, a (n + 1) = a_seq a b n) (h₃ : ∀ n, b (n + 1) = b_seq a b n) :
  ∑ k in Finset.range n, a k = (n + 2 : ℝ) / 2 - 1 / 2 ^ n := sorry

end geometric_sequence_sum_of_sequence_l24_24538


namespace collinear_points_vector_equality_l24_24939

-- Define points A, B, and C
def Point := (ℝ × ℝ)

def A : Point := (1, 1)
def B : Point := (3, -1)

variable (a b : ℝ)
def C : Point := (a, b)

-- Define vector subtraction
def vec_sub (p1 p2 : Point) : Point :=
  (p1.1 - p2.1, p1.2 - p2.2)

-- Define vector scaling
def vec_scale (c : ℝ) (p : Point) : Point :=
  (c * p.1, c * p.2)

-- First proof problem: Point A, B, C are collinear implies a = 2 - b
theorem collinear_points (h : ∃ k : ℝ, vec_sub B A = vec_scale k (vec_sub C A)) : a = 2 - b :=
sorry

-- Second proof problem: Vector AC = 2 * vector AB implies C is (5, -3)
theorem vector_equality (h : vec_sub C A = vec_scale 2 (vec_sub B A)) : C = (5, -3) :=
sorry

end collinear_points_vector_equality_l24_24939


namespace part_a_part_b_l24_24370

section
variables {n : ℕ} (a x : Fin n → ℝ)

def d_i (i : Fin n) : ℝ :=
  max (fun j => a j) {x | x ≤ i} - min (fun j => a j) {x | i ≤ x}

def d : ℝ :=
  max (fun i => d_i a i) {x | x < n}

-- Part (a)
theorem part_a (h : ∀ i, x i ≤ x (i + 1)) : 
  max (fun i => (|x i - a i|)) {x | x < n} ≥ d / 2 :=
sorry

-- Part (b)
theorem part_b : 
  ∃ x : Fin n → ℝ, (∀ i, x i ≤ x (i + 1)) ∧ max (fun i => (|x i - a i|)) {x | x < n} = d / 2 :=
sorry
end

end part_a_part_b_l24_24370


namespace find_angle_C_find_a_and_b_max_area_l24_24605

-- Problem 1: Prove that angle C = π / 3
theorem find_angle_C (A B C : ℝ) (a b c : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : a = 2) (h8 : c = 2)
  (h9 : sqrt 3 * a = 2 * c * Real.sin A) :
  C = π / 3 :=
sorry

-- Problem 2: Prove that a = 2 and b = 2
theorem find_a_and_b (A B C : ℝ) (a b c : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : c = 2) (h8 : sqrt 3 * a = 2 * c * Real.sin A)
  (h9 : (1 / 2) * a * b * Real.sin C = sqrt 3) :
  a = 2 ∧ b = 2 :=
sorry

-- Problem 3: Prove that the maximum area is sqrt 3
theorem max_area (A B C : ℝ) (a b c : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : c = 2) (h8 : sqrt 3 * a = 2 * c * Real.sin A)
  (h9 : (1 / 2) * a * b * Real.sin C = sqrt 3) :
  let area := (1 / 2) * a * b * Real.sin C in
  area = sqrt 3 :=
sorry

end find_angle_C_find_a_and_b_max_area_l24_24605


namespace cow_value_increase_l24_24629

theorem cow_value_increase :
  let starting_weight : ℝ := 732
  let increase_factor : ℝ := 1.35
  let price_per_pound : ℝ := 2.75
  let new_weight := starting_weight * increase_factor
  let value_at_new_weight := new_weight * price_per_pound
  let value_at_starting_weight := starting_weight * price_per_pound
  let increase_in_value := value_at_new_weight - value_at_starting_weight
  increase_in_value = 704.55 :=
by
  sorry

end cow_value_increase_l24_24629


namespace initial_weight_of_beef_l24_24811

theorem initial_weight_of_beef (W : ℝ) :
  (W * 0.75 = (120 / 0.85 + 144 / 0.80 + 180 / 0.75)) → W = 748.235 :=
by
  intro h,
  sorry

end initial_weight_of_beef_l24_24811


namespace repeating_decimal_product_l24_24035

theorem repeating_decimal_product (x y : ℚ) (h₁ : x = 8 / 99) (h₂ : y = 1 / 3) :
    x * y = 8 / 297 := by
  sorry

end repeating_decimal_product_l24_24035


namespace sum_of_roots_eq_2023_l24_24062

def lhs (x : ℝ) : ℝ :=
  Real.sqrt (2 * x^2 - 2024 * x + 1023131) + 
  Real.sqrt (3 * x^2 - 2025 * x + 1023132) + 
  Real.sqrt (4 * x^2 - 2026 * x + 1023133)

def rhs (x : ℝ) : ℝ :=
  Real.sqrt (x^2 - x + 1) + 
  Real.sqrt (2 * x^2 - 2 * x + 2) + 
  Real.sqrt (3 * x^2 - 3 * x + 3)

theorem sum_of_roots_eq_2023 :
  {x : ℝ | lhs x = rhs x} = {1010, 1013} → 1010 + 1013 = 2023 :=
sorry

end sum_of_roots_eq_2023_l24_24062


namespace min_value_of_sin_cos_sixth_l24_24053

theorem min_value_of_sin_cos_sixth (α : ℝ) (hα1 : 0 ≤ α) (hα2 : α ≤ π / 2) : 
  ∃ β : ℝ, β = sin α ^ 6 + cos α ^ 6 ∧ (∀ ε : ℝ, (sin ε ^ 6 + cos ε ^ 6) ≥ β) ∧ β = 1/4 :=
sorry

end min_value_of_sin_cos_sixth_l24_24053


namespace zeta_martingale_l24_24213

-- Definitions of the given conditions
variable {X : Type} [Fintype X]
variable (P : X → X → ℝ) -- Transition probability matrix
variable (ϕ : X → ℝ) -- Non-negative function

-- Homogeneous Markov chain
variable (ξ : ℕ → X)
variable {H_markov : ∀ k (x : X), ∑ y, P x y = 1}

-- One-step transition operator
def T (ϕ : X → ℝ) (x : X) : ℝ := ∑ y, ϕ y * P x y

-- Harmonic function condition
variable {H_harmonic : ∀ x, T ϕ x = ϕ x}

-- Filtration
variable (ℱ : ℕ → MeasureTheory.Measure X)

-- Sequence of random variables
def ζ (k : ℕ) : ℝ := ϕ (ξ k)

-- Martingale statement
theorem zeta_martingale : ∀ k, MeasureTheory.condexp ℱ k (ζ (k + 1)) = ζ k := by
  sorry

end zeta_martingale_l24_24213


namespace total_nails_to_cut_l24_24864

theorem total_nails_to_cut :
  let dogs := 4 
  let legs_per_dog := 4
  let nails_per_dog_leg := 4
  let parrots := 8
  let legs_per_parrot := 2
  let nails_per_parrot_leg := 3
  let extra_nail := 1
  let total_dog_nails := dogs * legs_per_dog * nails_per_dog_leg
  let total_parrot_nails := (parrots * legs_per_parrot * nails_per_parrot_leg) + extra_nail
  total_dog_nails + total_parrot_nails = 113 :=
sorry

end total_nails_to_cut_l24_24864


namespace remaining_if_stoss_half_l24_24028

-- Define the variables
variables (B M T G S : ℝ)

-- Define the conditions given in the problem statement
def condition_M := B - (M / 2) = (1 / 10) * B
def condition_T := B - (T / 2) = (1 / 8) * B
def condition_G := B - (G / 2) = (1 / 4) * B
def total_consumption := M + T + G + S = B

-- Define the resultant proof
theorem remaining_if_stoss_half : 
  condition_M ∧ condition_T ∧ condition_G ∧ total_consumption → 
  (B - (S / 2) = (1 / 40) * B) :=
by
  sorry

end remaining_if_stoss_half_l24_24028


namespace compare_log_values_l24_24491

noncomputable def a : ℝ := (Real.log 2) / 2
noncomputable def b : ℝ := (Real.log 3) / 3
noncomputable def c : ℝ := (Real.log 5) / 5

theorem compare_log_values : c < a ∧ a < b := by
  -- Proof is omitted
  sorry

end compare_log_values_l24_24491


namespace find_a_2016_l24_24514

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = (n + 1) / n * a n

theorem find_a_2016 (a : ℕ → ℝ) (h : seq a) : a 2016 = 4032 :=
by
  sorry

end find_a_2016_l24_24514


namespace seating_arrangements_l24_24180

open Nat

def factorial : ℕ → ℕ
| 0       => 1
| (n+1)   => (n + 1) * factorial n

def total_arrangements (n : ℕ) : ℕ := factorial n

def restricted_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  factorial (n - k + 1) * factorial k

theorem seating_arrangements :
  total_arrangements 8 - restricted_arrangements 8 4 = 37440 :=
by
  sorry

end seating_arrangements_l24_24180


namespace problem_statement_l24_24087

theorem problem_statement (m : ℝ) (a b : ℝ) 
  (h1 : 9^m = 10) (h2 : a = 10^m - 11) (h3 : b = 8^m - 9) : 
  a > 0 ∧ 0 > b :=
sorry

end problem_statement_l24_24087


namespace interval_length_slope_l24_24636

def S : set (ℤ × ℤ) := {p | 1 ≤ p.1 ∧ p.1 ≤ 30 ∧ 1 ≤ p.2 ∧ p.2 ≤ 30}

theorem interval_length_slope :
  let m : ℝ := slope such that 
    (card { p ∈ S | p.2 ≤ m * p.1 } = 300)
  ∃ a b : ℕ, a.gcd b = 1 ∧ (interval_of_m).length = (a : ℚ) / b ∧ a + b = 85 :=
begin
  sorry
end

end interval_length_slope_l24_24636


namespace smallest_n_of_sum_of_squares_of_divisors_l24_24205

theorem smallest_n_of_sum_of_squares_of_divisors :
  ∃ n x_3 x_4, 
    (1 = 1) ∧ (2 ∣ n) ∧ (x_3 ∣ n) ∧ (x_4 ∣ n) ∧ 
    (1 < 2) ∧ (2 < x_3) ∧ (x_3 < x_4) ∧ 
    (n = 1^2 + 2^2 + x_3^2 + x_4^2) ∧
    (∀ m, (m = 1^2 + 2^2 + y_3^2 + y_4^2) → (1 = 1) ∧ (2 ∣ m) ∧ (y_3 ∣ m) ∧ (y_4 ∣ m) ∧ 
           (1 < 2) ∧ (2 < y_3) ∧ (y_3 < y_4) ∧ (m ≥ n)) :=
begin
  sorry
end

end smallest_n_of_sum_of_squares_of_divisors_l24_24205


namespace find_f_f_ln2p1_l24_24493

noncomputable def f : ℝ → ℝ := 
λ x, if x < 2 then 3 * Real.exp (x - 1) else Real.logBase 7 (8 * x + 1)

theorem find_f_f_ln2p1 : f (f (Real.log 2 + 1)) = 2 := by
  sorry

end find_f_f_ln2p1_l24_24493


namespace no_unhappy_days_l24_24727

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end no_unhappy_days_l24_24727


namespace election_win_percentage_l24_24600

theorem election_win_percentage (total_votes : ℕ) (james_percentage : ℝ) (additional_votes_needed : ℕ) (votes_needed_to_win_percentage : ℝ) :
    total_votes = 2000 →
    james_percentage = 0.005 →
    additional_votes_needed = 991 →
    votes_needed_to_win_percentage = (1001 / 2000) * 100 →
    votes_needed_to_win_percentage > 50.05 :=
by
  intros h_total_votes h_james_percentage h_additional_votes_needed h_votes_needed_to_win_percentage
  sorry

end election_win_percentage_l24_24600


namespace seat_arrangement_count_is_20_l24_24338

def number_of_seat_arrangements : ℕ := 5

theorem seat_arrangement_count_is_20 :
  (∃ (s : finset (fin number_of_seat_arrangements)) (h : s.card = 2), 
     ∃ (t : list (fin (number_of_seat_arrangements - 2))), 
      t.nodup ∧ t.length = 3 ∧ 
      ∀ (perm : list (fin (number_of_seat_arrangements - 2))) 
          (_h : perm.nodup ∧ perm.length = 3), 
        disjoint s perm) 
  → (@finset.univ (fin number_of_seat_arrangements) _).card * 2 = 20 :=
by sorry

end seat_arrangement_count_is_20_l24_24338


namespace change_in_expression_l24_24300

theorem change_in_expression (x b : ℝ) (hb : b > 0):
  let expr := x^2 - 5 * x + 6 in
  (x + b)^2 - 5 * (x + b) + 6 - expr = 2 * b * x + b^2 - 5 * b ∧ 
  (x - b)^2 - 5 * (x - b) + 6 - expr = -2 * b * x + b^2 + 5 * b := 
by
  let expr := x^2 - 5 * x + 6
  have h1 : (x + b)^2 - 5 * (x + b) + 6 - expr = 2 * b * x + b^2 - 5 * b := sorry
  have h2 : (x - b)^2 - 5 * (x - b) + 6 - expr = -2 * b * x + b^2 + 5 * b := sorry
  exact ⟨h1, h2⟩

end change_in_expression_l24_24300


namespace total_seats_in_hall_l24_24599

-- Conditions
def total_seats_filled_percentage := 0.75
def vacant_seats : ℕ := 175

-- Prove total seats is 700 given the conditions
theorem total_seats_in_hall (S : ℕ) 
  (H1 : 175 = (1 - total_seats_filled_percentage) * S) : 
  S = 700 :=
by sorry

end total_seats_in_hall_l24_24599


namespace gasoline_price_increase_percent_l24_24720

theorem gasoline_price_increase_percent {P Q : ℝ}
  (h₁ : P > 0)
  (h₂: Q > 0)
  (x : ℝ)
  (condition : P * Q * 1.08 = P * (1 + x/100) * Q * 0.90) :
  x = 20 :=
by {
  sorry
}

end gasoline_price_increase_percent_l24_24720


namespace find_possible_values_of_a_l24_24317

theorem find_possible_values_of_a (a b c : ℝ) (h1 : a * b + a + b = c) (h2 : b * c + b + c = a) (h3 : c * a + c + a = b) :
  a = 0 ∨ a = -1 ∨ a = -2 :=
by
  sorry

end find_possible_values_of_a_l24_24317


namespace larger_sign_diameter_l24_24755

theorem larger_sign_diameter (d k : ℝ) 
  (h1 : ∀ d, d > 0) 
  (h2 : ∀ k, (π * (k * d / 2)^2 = 49 * π * (d / 2)^2)) : 
  k = 7 :=
by
sorry

end larger_sign_diameter_l24_24755


namespace measure_of_angle_A_is_60_degrees_l24_24631

-- Definitions for the conditions given:
variables {A B C : Type} [nonempty A] [nonempty B] [nonempty C]

-- Definitions of the conditions using Lean's structures:
def is_acute_angled (Δ : triangle A B C) : Prop := sorry  -- to be defined formally
def is_not_equilateral (Δ : triangle A B C) : Prop := sorry  -- to be defined formally
def lies_on_perpendicular_bisector (P : A) (X Y : A) : Prop := sorry  -- P lies on the perpendicular bisector of XY
def orthocenter (Δ : triangle A B C) : A := sorry  -- to be defined formally
def circumcenter (Δ : triangle A B C) : A := sorry  -- to be defined formally
def measure_angle_A (Δ : triangle A B C) : ℝ := sorry  -- to be defined formally

-- The theorem to prove:
theorem measure_of_angle_A_is_60_degrees (Δ : triangle A B C)
  (h1 : is_acute_angled Δ)
  (h2 : is_not_equilateral Δ)
  (h3 : lies_on_perpendicular_bisector (vertex A) (orthocenter Δ) (circumcenter Δ)) :
  measure_angle_A Δ = 60 :=
sorry

end measure_of_angle_A_is_60_degrees_l24_24631


namespace don_travel_time_to_hospital_l24_24367

noncomputable def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

noncomputable def time_to_travel (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

theorem don_travel_time_to_hospital :
  let speed_mary := 60
  let speed_don := 30
  let time_mary_minutes := 15
  let time_mary_hours := time_mary_minutes / 60
  let distance := distance_traveled speed_mary time_mary_hours
  let time_don_hours := time_to_travel distance speed_don
  time_don_hours * 60 = 30 :=
by
  sorry

end don_travel_time_to_hospital_l24_24367


namespace sunlovers_happy_days_l24_24733

open Nat

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end sunlovers_happy_days_l24_24733


namespace no_valid_m_factorization_l24_24022

theorem no_valid_m_factorization :
  ∀ (m : ℤ), ¬ ∃ (A B C D E F : ℤ),
  (λ (x y : ℤ), x^2 + 4 * x * y + 2 * x + m * y + m) = (λ (x y : ℤ), (A * x + B * y + C) * (D * x + E * y + F)) :=
by
  sorry

end no_valid_m_factorization_l24_24022


namespace equation_of_symmetric_line_l24_24670

theorem equation_of_symmetric_line
  (a b : ℝ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) :
  (∀ x : ℝ, ∃ y : ℝ, (x = a * y + b)) → (∀ x : ℝ, ∃ y : ℝ, (y = (1/a) * x - (b/a))) :=
by
  sorry

end equation_of_symmetric_line_l24_24670


namespace pete_and_ray_spent_200_cents_l24_24274

-- Define the basic units
def cents_in_a_dollar := 100
def value_of_a_nickel := 5
def value_of_a_dime := 10

-- Define the initial amounts and spending
def pete_initial_amount := 250  -- cents
def ray_initial_amount := 250  -- cents
def pete_spent_nickels := 4 * value_of_a_nickel
def ray_remaining_dimes := 7 * value_of_a_dime

-- Calculate total amounts spent
def total_spent_pete := pete_spent_nickels
def total_spent_ray := ray_initial_amount - ray_remaining_dimes
def total_spent := total_spent_pete + total_spent_ray

-- The proof problem statement
theorem pete_and_ray_spent_200_cents : total_spent = 200 := by {
 sorry
}

end pete_and_ray_spent_200_cents_l24_24274


namespace k_has_six_non_zero_digits_l24_24677

theorem k_has_six_non_zero_digits (k : ℕ) (A : ℕ) (hA : A = 10101010101) (hdiv : A ∣ k) : 
  (nat.digits 10 k).filter (λ x, x ≠ 0) >= 6 :=
begin
  sorry
end

end k_has_six_non_zero_digits_l24_24677


namespace length_of_other_parallel_side_l24_24466

noncomputable def is_area_of_trapezium_valid (x : ℕ) : Prop :=
  209 = (1 / 2 : ℝ) * (x + 18) * 11

theorem length_of_other_parallel_side : ∃ x : ℕ, is_area_of_trapezium_valid x ∧ x = 20 :=
  by {
    use 20,
    unfold is_area_of_trapezium_valid,
    norm_num,
    sorry,  -- Here we skip the proof
  }

end length_of_other_parallel_side_l24_24466


namespace find_constants_l24_24884

theorem find_constants (a1 a2 : ℝ) 
  (h : a1 • (3, 4) + a2 • (-3, 6) = (0 : ℝ, 5)) : 
  (a1, a2) = (1/2, 1/2) :=
sorry

end find_constants_l24_24884


namespace probability_last_two_seats_correct_l24_24702

theorem probability_last_two_seats_correct :
  let n := 100
  let first_student_sits_randomly := ∀ (prob : ℝ), 0 < prob ∧ prob ≤ 1
  let other_students_behave_correctly := ∀ (k : ℤ), 1 ≤ k ∧ k < n → k ≠ n - 1 ∧ k ≠ n
  true → -- Assuming conditions hold for other sequences (omitted detailed steps)
  let probability_correct_seats := 1 / (3 : ℝ)
  (∀ event : ℝ, first_student_sits_randomly event) →
  (other_students_behave_correctly → ∃ final_probability : ℝ, final_probability = probability_correct_seats) :=
begin
  intro h,
  sorry
end

end probability_last_two_seats_correct_l24_24702


namespace range_of_m_l24_24107

open_locale real

variables {m : ℝ}
def prop_p := (1, -1, m) • (1, 2, m) ≥ 0
def prop_q := 1 < (5 + m) / 5 ∧ (5 + m) / 5 < 2

theorem range_of_m (H₁ : ¬ prop_q) (H₂ : ¬ (prop_p ∧ prop_q)) : 0 < m ∧ m < 1 :=
sorry

end range_of_m_l24_24107


namespace factorial_fraction_eq_l24_24857

theorem factorial_fraction_eq :
  (15.factorial / (6.factorial * 9.factorial) = 5005) := 
sorry

end factorial_fraction_eq_l24_24857


namespace triangle_perpendicular_sum_l24_24194

theorem triangle_perpendicular_sum {A B C P M N : Type*} [EuclideanGeometry A B C P M N]
  (hABC : Triangle A B C) 
  (hAC_eq_AB : A.distance_to C = A.distance_to B)
  (hP_on_BC : P.on_line_segment B C)
  (hPM_perpendicular_BC : P.perpendicular_to_line_segment B C M)
  (hM_on_AB : M.on_line_segment A B) 
  (hN_on_AC : N.on_line_segment A C)
  (hHa : Altitude A hABC = hA) : 
  P.distance_to M + P.distance_to N = 2 * hA := 
sorry

end triangle_perpendicular_sum_l24_24194


namespace rosa_parks_food_drive_l24_24306

theorem rosa_parks_food_drive :
  ∀ (total_students students_collected_12_cans students_collected_none students_remaining total_cans cans_collected_first_group total_cans_first_group total_cans_last_group cans_per_student_last_group : ℕ),
    total_students = 30 →
    students_collected_12_cans = 15 →
    students_collected_none = 2 →
    students_remaining = total_students - students_collected_12_cans - students_collected_none →
    total_cans = 232 →
    cans_collected_first_group = 12 →
    total_cans_first_group = students_collected_12_cans * cans_collected_first_group →
    total_cans_last_group = total_cans - total_cans_first_group →
    cans_per_student_last_group = total_cans_last_group / students_remaining →
    cans_per_student_last_group = 4 :=
by
  intros total_students students_collected_12_cans students_collected_none students_remaining total_cans cans_collected_first_group total_cans_first_group total_cans_last_group cans_per_student_last_group
  sorry

end rosa_parks_food_drive_l24_24306


namespace bug_final_position_after_2023_jumps_l24_24294

open Nat

def bug_jump (pos : Nat) : Nat :=
  if pos % 2 = 1 then (pos + 2) % 6 else (pos + 1) % 6

noncomputable def final_position (n : Nat) : Nat :=
  (iterate bug_jump n 6) % 6

theorem bug_final_position_after_2023_jumps : final_position 2023 = 1 := by
  sorry

end bug_final_position_after_2023_jumps_l24_24294


namespace value_of_a2019_l24_24536

noncomputable def a : ℕ → ℝ
| 0 => 3
| (n + 1) => 1 / (1 - a n)

theorem value_of_a2019 : a 2019 = 2 / 3 :=
sorry

end value_of_a2019_l24_24536


namespace points_per_game_l24_24267

theorem points_per_game (total_points games : ℕ) (h1 : total_points = 91) (h2 : games = 13) :
  total_points / games = 7 :=
by
  sorry

end points_per_game_l24_24267


namespace distance_from_A_to_directrix_of_C_l24_24970

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24970


namespace f_prime_at_0_l24_24187

noncomputable def a (n : ℕ) : ℝ := 2 * (2/8)^(n-1)

def f (x : ℝ) : ℝ := x * (x - a 1) * (x - a 2) * (x - a 3) * (x - a 4) * (x - a 5) * (x - a 6) * (x - a 7) * (x - a 8)

theorem f_prime_at_0 : deriv f 0 = 4096 := 
  sorry

end f_prime_at_0_l24_24187


namespace claire_crafting_hours_l24_24412

variable (total_hours_in_day : ℕ := 24)
variable (hours_cleaning : ℕ := 4)
variable (hours_cooking : ℕ := 2)
variable (hours_sleeping : ℕ := 8)

theorem claire_crafting_hours : 
  ∀ (remaining_hours : ℕ), (remaining_hours = total_hours_in_day - (hours_cleaning + hours_cooking + hours_sleeping)) →
  remaining_hours / 2 = 5 :=
by
  assume remaining_hours,
  intro h,
  sorry

end claire_crafting_hours_l24_24412


namespace family_of_four_children_includes_one_boy_one_girl_l24_24835

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - ((1/2)^4 + (1/2)^4)

theorem family_of_four_children_includes_one_boy_one_girl :
  probability_at_least_one_boy_and_one_girl = 7 / 8 :=
by
  sorry

end family_of_four_children_includes_one_boy_one_girl_l24_24835


namespace alpha_perpendicular_beta_necessary_but_not_sufficient_l24_24937

open Set

variables {α β : Type*} [n : NormedSpace ℝ α] [m : NormedSpace ℝ β]
variables (a b c : α) (plane_α plane_β : AffineSubspace ℝ α)

-- Conditions
def a_parallel_b : Prop := ∥a - b∥ = 0
def b_perpendicular_to_plane_α : Prop := ∀ v ∈ plane_α.direction, inner v (b - a) = 0

-- Statement
theorem alpha_perpendicular_beta_necessary_but_not_sufficient
  (h1 : a_parallel_b a b)
  (h2 : b_perpendicular_to_plane_α b plane_α)
  (h3 : ∀ v ∈ plane_α.direction, inner v (b - a) = 0)
  (h4 : ∀ v ∈ plane_β.direction, inner v (plane_α.direction) = 0) :
  (∀ v ∈ plane_β.direction, inner v (a - c) = 0) → False :=
by
  sorry

end alpha_perpendicular_beta_necessary_but_not_sufficient_l24_24937


namespace triangle_area_ratio_l24_24349

theorem triangle_area_ratio (a b c : ℕ) (d e f : ℕ) 
  (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) 
  (h4 : d = 9) (h5 : e = 12) (h6 : f = 15) 
  (GHI_right : a^2 + b^2 = c^2)
  (JKL_right : d^2 + e^2 = f^2):
  (0.5 * a * b) / (0.5 * d * e) = 4 / 9 := 
by 
  sorry

end triangle_area_ratio_l24_24349


namespace calculate_total_bill_l24_24560

variables (orig_order : ℝ) (tomato_old tomato_new lettuce_old lettuce_new celery_old celery_new delivery_tip : ℝ)

def new_tomato_price := tomato_new - tomato_old
def new_lettuce_price := lettuce_new - lettuce_old
def new_celery_price := celery_new - celery_old

def total_increase := new_tomato_price + new_lettuce_price + new_celery_price
def new_food_cost := orig_order + total_increase
def new_total_bill := new_food_cost + delivery_tip

theorem calculate_total_bill :
  orig_order = 25 →
  tomato_old = 0.99 →
  tomato_new = 2.20 →
  lettuce_old = 1.00 →
  lettuce_new = 1.75 →
  celery_old = 1.96 →
  celery_new = 2.00 →
  delivery_tip = 8.00 →
  new_total_bill = 35.00 :=
by
  intros ho to_old to_new lo_old lo_new co_old co_new dt
  rw [←ho, ←to_old, ←to_new, ←lo_old, ←lo_new, ←co_old, ←co_new, ←dt]
  dsimp [new_tomato_price, new_lettuce_price, new_celery_price, total_increase, new_food_cost, new_total_bill]
  sorry

end calculate_total_bill_l24_24560


namespace part1_part2_part3_l24_24014

noncomputable def y (a x : ℝ) : ℝ := a * real.sqrt (1 - x^2) + real.sqrt (1 + x) + real.sqrt (1 - x)
noncomputable def t (x : ℝ) : ℝ := real.sqrt (1 + x) + real.sqrt (1 - x)
noncomputable def m (a t : ℝ) : ℝ := (1/2) * a * t^2 + t - a
noncomputable def g(a : ℝ) : ℝ := 
  if a > -1/2 then a + 2
  else if -real.sqrt(2)/2 < a ∧ a <= -1/2 then -a - 1/(2 * a)
  else real.sqrt(2)

theorem part1 (a x : ℝ) (h : t x ∈ set.Icc (real.sqrt 2) 2) : 
  y a x = m a (t x) := sorry

theorem part2 (a : ℝ) : 
  ∀ t ∈ set.Icc (real.sqrt 2) 2, 
  m a t ≤ g a := sorry

theorem part3 (a : ℝ) (h : a >= -real.sqrt 2) : 
  g a = g (1/a) ↔ (a ∈ set.Icc (-real.sqrt 2) (-real.sqrt 2 / 2) ∨ a = 1) := sorry

end part1_part2_part3_l24_24014


namespace problem_statement_l24_24072

theorem problem_statement (m a b : ℝ) (h0 : 9^m = 10) (h1 : a = 10^m - 11) (h2 : b = 8^m - 9) : a > 0 ∧ 0 > b :=
by
  sorry

end problem_statement_l24_24072


namespace not_divisible_by_10100_l24_24282

theorem not_divisible_by_10100 (n : ℕ) : (3^n + 1) % 10100 ≠ 0 := 
by 
  sorry

end not_divisible_by_10100_l24_24282


namespace curve_common_points_max_area_quadrilateral_l24_24535

noncomputable def curve_C1_parametric (a b : ℝ) (θ : ℝ) (h : a > b > 0) :=
  (a * Real.cos θ, b * Real.sin θ)

noncomputable def curve_C1_general (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def curve_C2_polar (r : ℝ) (θ : ℝ) (hr : r > 0) :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def curve_C2_cartesian (x y r : ℝ) : Prop :=
  x^2 + y^2 = r^2

theorem curve_common_points (a b r : ℝ) (h1 : a > b > 0) (h2 : r > 0) :
  (if r = a ∨ r = b then 
    ∃ p₁ p₂ : ℝ × ℝ, curve_C1_general p₁.1 p₁.2 a b ∧ curve_C2_cartesian p₁.1 p₁.2 r ∧ 
                      curve_C1_general p₂.1 p₂.2 a b ∧ curve_C2_cartesian p₂.1 p₂.2 r ∧ p₁ ≠ p₂
  else if b < r < a then 
    ∃ p₁ p₂ p₃ p₄ : ℝ × ℝ, 
    curve_C1_general p₁.1 p₁.2 a b ∧ curve_C2_cartesian p₁.1 p₁.2 r ∧ 
    curve_C1_general p₂.1 p₂.2 a b ∧ curve_C2_cartesian p₂.1 p₂.2 r ∧ 
    curve_C1_general p₃.1 p₃.2 a b ∧ curve_C2_cartesian p₃.1 p₃.2 r ∧ 
    curve_C1_general p₄.1 p₄.2 a b ∧ curve_C2_cartesian p₄.1 p₄.2 r ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄
  else 
    ∀ p : ℝ × ℝ, ¬ (curve_C1_general p.1 p.2 a b ∧ curve_C2_cartesian p.1 p.2 r)) :=
sorry

theorem max_area_quadrilateral (a b r : ℝ) (h1 : a > b > 0) (h2 : b < r < a) :
  ∃ θ₀ : ℝ, (curve_C1_parametric a b θ₀ h1).1 * (curve_C2_polar r θ₀ r).2 * 4 = 2 * a * b :=
sorry

end curve_common_points_max_area_quadrilateral_l24_24535


namespace convert_neg_900_deg_to_rad_l24_24016

theorem convert_neg_900_deg_to_rad : (-900 : ℝ) * (Real.pi / 180) = -5 * Real.pi :=
by
  sorry

end convert_neg_900_deg_to_rad_l24_24016


namespace mutually_perpendicular_planes_unnecessary_intersecting_word_l24_24785

theorem mutually_perpendicular_planes_unnecessary_intersecting_word (P Q : Plane) :
  (∃ H1 H2 : Line, H1 ∈ P ∧ H2 ∈ Q ∧ H1 ∩ H2 ⊆ P ∩ Q ∧ DihedralAngle H1 H2 = 90) →
  (word_used_for_intersection : String) = "пересекаясь" → 
  "пересекаясь" is unnecessary :=
sorry

end mutually_perpendicular_planes_unnecessary_intersecting_word_l24_24785


namespace problem1_problem2_l24_24532

-- Problem 1: Monotonically Increasing Intervals of F
theorem problem1 (a : ℝ) (x : ℝ) (h1 : -1 ≤ a) :
  (F : ℝ → ℝ) = (λ x, x - 1/x - 2 * a * log x) → 
  if h1 ∧ a ≤ 1 then (∀ x > 0, F' x ≥ 0) else 
  if h1 ∧ a > 1 then ∀ x > 0, x < a - sqrt(a^2 - 1) ∨ x > a + sqrt(a^2 - 1) → F' x ≥ 0 :=
sorry

-- Problem 2: Minimum Value of h(x1) - h(x2)
theorem problem2 (a : ℝ) (x1 : ℝ) (x2 : ℝ) (h1 : x1 ∈ set.Icc 0 (1/3)) :
  let h := λ x, x - 1/x + 2 * a * log x in
  (x1 * x2 = 1) ∧ (x1 + x2 = -2 * a) → 
  ∃ m, m = (function.min (λ x, 2 * (x - 1/x - (x + 1/x) * log x)) x1) 
  m = (20 * log 3 - 16) / 3 :=
sorry

end problem1_problem2_l24_24532


namespace percentage_people_taking_bus_l24_24165

-- Definitions
def population := 80
def car_pollution := 10 -- pounds of carbon per car per year
def bus_pollution := 100 -- pounds of carbon per bus per year
def bus_capacity := 40 -- people per bus
def carbon_reduction := 100 -- pounds of carbon reduced per year after the bus is introduced

-- Problem statement in Lean 4
theorem percentage_people_taking_bus :
  (10 / 80 : ℝ) = 0.125 :=
by
  sorry

end percentage_people_taking_bus_l24_24165


namespace inverse_of_g_l24_24710

variables {X Y Z W : Type} [Invertible X] [Invertible Y] [Invertible Z]

-- Define the functions u, v, and w as invertible
variable (u : X → Y) (v : Y → Z) (w : W → X)
variable [Invertible u] [Invertible v] [Invertible w]

-- Define g as the composition of v, u, and w
def g : W → Z := v ∘ u ∘ w

-- Prove that the inverse of g is the composition of the inverses of w, u, and v in the reverse order
theorem inverse_of_g :
  g⁻¹ = (w⁻¹ ∘ u⁻¹ ∘ v⁻¹) :=
sorry

end inverse_of_g_l24_24710


namespace solve_system_of_equations_l24_24698

theorem solve_system_of_equations (x y : ℝ) :
  (3 * x^2 + 4 * x * y + 12 * y^2 + 16 * y = -6) ∧
  (x^2 - 12 * x * y + 4 * y^2 - 10 * x + 12 * y = -7) →
  (x = 1 / 2) ∧ (y = -3 / 4) :=
by
  sorry

end solve_system_of_equations_l24_24698


namespace tan_product_l24_24425

noncomputable def tan : ℝ → ℝ := sorry

theorem tan_product :
  (tan (Real.pi / 8)) * (tan (3 * Real.pi / 8)) * (tan (5 * Real.pi / 8)) = 2 * Real.sqrt 7 :=
by
  sorry

end tan_product_l24_24425


namespace common_divisors_count_l24_24150

-- Definition of the three numbers
def a := 9240
def b := 7920
def c := 8800

-- Proof that the number of positive divisors common to a, b, and c is 32
theorem common_divisors_count : (λ n, n.num_divisors) (gcd a (gcd b c)) = 32 := sorry

end common_divisors_count_l24_24150


namespace rational_t_l24_24790

variable (A B t : ℚ)

theorem rational_t (A B : ℚ) (hA : A = 2 * t / (1 + t^2)) (hB : B = (1 - t^2) / (1 + t^2)) : ∃ t' : ℚ, t = t' :=
by
  sorry

end rational_t_l24_24790


namespace probability_valid_triangle_l24_24359

def vertices := {1, 2, 3, 4, 5, 6}

def is_valid_triangle (triangle: set ℕ) : Prop :=
  triangle = {1, 3, 5} ∨ triangle = {2, 4, 6}

def total_ways := (vertices.to_finset.powerset.filter (λ s, s.card = 3)).card

theorem probability_valid_triangle :
  total_ways = 20 ∧ (
  (vertices.to_finset.powerset.filter (λ s, s.card = 3 ∧ is_valid_triangle s)).card / total_ways
  ) = 1 / 10 := 
sorry

end probability_valid_triangle_l24_24359


namespace distance_from_point_to_directrix_l24_24948

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24948


namespace number_of_different_winning_scores_l24_24168

def sum_of_arithmetic_series (n : ℕ) : ℕ :=
  n * (n + 1) / 2

example : sum_of_arithmetic_series 12 = 78 := by
  unfold sum_of_arithmetic_series
  norm_num
  sorry

theorem number_of_different_winning_scores : 
  let total_sum := sum_of_arithmetic_series 12 in
  let sum_of_lowest_6_scores := sum_of_arithmetic_series 6 in
  let highest_winning_score := total_sum / 2 in
  ∀ possible_scores : finset ℕ, 
  (∀ score ∈ possible_scores, sum_of_lowest_6_scores ≤ score ∧ score < highest_winning_score) →
  (possible_scores = finset.range (36 - 21 + 1) + 21).card = 16 :=
by 
  sorry

end number_of_different_winning_scores_l24_24168


namespace angle_between_BC_and_EF_l24_24620

def triangle_ABC (A B C : Type) [metric_space ℕ] :=
triangle A B C

variables {A B C E F : ℝ}

theorem angle_between_BC_and_EF :
  is_isosceles A B C ∧
  (∡ A B C = 20) ∧
  (B C ∠ E = 50) ∧
  (C B ∠ F = 60) →
  angle_between BC EF = 30 :=
by sorry

end angle_between_BC_and_EF_l24_24620


namespace num_monic_quadratic_trinomials_l24_24056

theorem num_monic_quadratic_trinomials : 
  let N := 3363 in
  let max_coeff := 1331^(38:ℕ) in
  ∃! (f : ℕ × ℕ → Prop), 
  (∀ a b : ℕ, f (a, b) ↔ 
    11^a + 11^b ≤ max_coeff ∧
    a + b ≤ 114 ∧ 
    a ≥ b) ∧
  ∃! (count : ℕ), 
  count = N := sorry

end num_monic_quadratic_trinomials_l24_24056


namespace tan_pi_by_eight_product_l24_24436

theorem tan_pi_by_eight_product :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = -real.sqrt 2 := by 
sorry

end tan_pi_by_eight_product_l24_24436


namespace sum_of_products_of_subsets_l24_24141

-- Define the set M
def M : Set ℚ := {-2/3, 5/4, 1, 4}

-- Define the non-empty subsets of M and their product
def non_empty_subsets (s : Set ℚ) : List (Set ℚ) :=
  (s.toFinset.powerset.filter (λ t, t ≠ ∅)).toList.map (λ t, t : Set ℚ)

def product (s : Set ℚ) : ℚ :=
  s.fold (λ x y, x * y) 1

def M_subsets := non_empty_subsets M
def subset_products := M_subsets.map product

-- Define the proof statement
theorem sum_of_products_of_subsets : (List.sum subset_products) = 13 / 2 := 
by sorry

end sum_of_products_of_subsets_l24_24141


namespace book_total_pages_l24_24847

theorem book_total_pages (P : ℕ) (days_read : ℕ) (pages_per_day : ℕ) (fraction_read : ℚ) 
  (total_pages_read : ℕ) :
  (days_read = 15 ∧ pages_per_day = 12 ∧ fraction_read = 3 / 4 ∧ total_pages_read = 180 ∧ 
    total_pages_read = days_read * pages_per_day ∧ total_pages_read = fraction_read * P) → 
    P = 240 :=
by
  intros h
  sorry

end book_total_pages_l24_24847


namespace rectangle_area_l24_24500

theorem rectangle_area (a b : ℕ) 
  (h1 : 2 * (a + b) = 16)
  (h2 : a^2 + b^2 - 2 * a * b - 4 = 0) :
  a * b = 30 :=
by
  sorry

end rectangle_area_l24_24500


namespace regular_polygon_sides_l24_24742

theorem regular_polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 2 * 360) : 
  n = 6 :=
sorry

end regular_polygon_sides_l24_24742


namespace find_pointN_coordinates_l24_24929

noncomputable def pointN_on_z_axis_equidistant_from_AB : Prop :=
  ∃ (z : ℝ), (0, 0, z) = (0, 0, 2 / 5) ∧
    let dA := real.sqrt (1^2 + 0^2 + (z - 3)^2)
    let dB := real.sqrt ((-1)^2 + 1^2 + (-2 - z)^2)
    in dA = dB

theorem find_pointN_coordinates : pointN_on_z_axis_equidistant_from_AB :=
begin
  sorry
end

end find_pointN_coordinates_l24_24929


namespace sine_roots_polynomial_l24_24883

noncomputable def polynomial_with_given_sine_roots : Polynomial ℝ :=
  Polynomial.C 8 * Polynomial.X^3 -
  Polynomial.C 4 * Polynomial.X^2 -
  Polynomial.C 4 * Polynomial.X -
  Polynomial.C 1

theorem sine_roots_polynomial :
  is_root polynomial_with_given_sine_roots (Real.sin (Real.pi / 14)) ∧
  is_root polynomial_with_given_sine_roots (Real.sin (5 * Real.pi / 14)) ∧
  is_root polynomial_with_given_sine_roots (Real.sin (-3 * Real.pi / 14)) :=
by
  sorry

end sine_roots_polynomial_l24_24883


namespace exists_five_digit_palindromic_divisible_by_5_count_five_digit_palindromic_numbers_divisible_by_5_l24_24252

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem exists_five_digit_palindromic_divisible_by_5 :
  ∃ n, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n := by
  -- Proof is omitted
  sorry

theorem count_five_digit_palindromic_numbers_divisible_by_5 :
  (finset.filter (λ n, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n) (finset.range 100000)).card = 100 := by
  -- Proof is omitted
  sorry

end exists_five_digit_palindromic_divisible_by_5_count_five_digit_palindromic_numbers_divisible_by_5_l24_24252


namespace binomial_identity_l24_24285

open Nat

theorem binomial_identity (n k : ℕ) (h1 : 0 ≤ k) (h2 : k ≤ n - 1) :
  (∑ j in Finset.range (k + 1), Nat.choose n j) = 
  (∑ j in Finset.range (k + 1), Nat.choose (n - 1 - j) (k - j) * 2^j) :=
sorry

end binomial_identity_l24_24285


namespace example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l24_24247

-- Definition of a five-digit palindromic number
def is_palindromic (n : ℕ) : Prop := let s := n.to_string in s = s.reverse

-- Definition of a five-digit number
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

-- Part (a): Prove that 51715 is a five-digit palindromic number and is divisible by 5
theorem example_palindromic_divisible_by_5 :
  is_five_digit 51715 ∧ is_palindromic 51715 ∧ 51715 % 5 = 0 :=
by sorry

-- Part (b): Prove that there are exactly 100 five-digit palindromic numbers divisible by 5
theorem count_palindromic_divisible_by_5 :
  (finset.filter (λ n, is_five_digit n ∧ is_palindromic n ∧ n % 5 = 0) 
    (finset.range 100000)).card = 100 :=
by sorry

end example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l24_24247


namespace factorization_quad_l24_24305

theorem factorization_quad (c d : ℕ) (h_factor : (x^2 - 18 * x + 77 = (x - c) * (x - d)))
  (h_nonneg : c ≥ 0 ∧ d ≥ 0) (h_lt : c > d) : 4 * d - c = 17 := by
  sorry

end factorization_quad_l24_24305


namespace correct_statements_l24_24485

variable (a b : Line) (alpha beta : Plane)

-- Definitions for perpendicular and parallel relationship
constant perp : Line → Plane → Prop
constant para : Line → Plane → Prop
constant line_perp : Line → Line → Prop
constant para_planes : Plane → Plane → Prop
constant perp_planes : Plane → Plane → Prop

-- Conditions
axiom a_perp_alpha : perp a alpha
axiom b_perp_beta : perp b beta
axiom a_perp_b : line_perp a b
axiom alpha_para_beta : para_planes alpha beta
axiom a_para_b : parallel a b
axiom a_para_alpha : para a alpha
axiom b_para_beta : para b beta

-- Statement
theorem correct_statements :
  (∃ (alpha_perp_beta : Prop), alpha_perp_beta = true) ∧
  (∃ (b_perp_beta : Prop), b_perp_beta = true) ∧
  (¬ (∀ (a_para_b : Prop), a_para_b = true)) ∧
  (∃ (alpha_perp_beta : Prop), alpha_perp_beta = true) :=
by
  sorry

end correct_statements_l24_24485


namespace segment_length_l24_24178

-- Definitions from conditions
variables (r α : ℝ)
variable (ABC : Type) 
variables (A B C O : ABC)
variables [Isosceles α r ABC A B C O]

axiom incenter_radius : ∀ (B C : ABC), radius O B C = r
axiom base_angle : ∀ (B : ABC), angle A B C = α

-- The goal is to prove the given expression for the segment inside the triangle
theorem segment_length : 
  segment_length A B O = 4 * r * (Real.cos (α / 2)) ^ 2 / (Real.sin (3 * α / 2)) :=
sorry

end segment_length_l24_24178


namespace BI_eq_sqrt_131_l24_24619

noncomputable def length_BI {A B C I : Type*} 
  (h_AB : dist A B = 29) 
  (h_AC : dist A C = 28) 
  (h_BC : dist B C = 27) 
  (h_I_center : incenter I A B C) : ℝ :=
  dist B I

theorem BI_eq_sqrt_131 {A B C I : Type*} 
  (h_AB : dist A B = 29) 
  (h_AC : dist A C = 28) 
  (h_BC : dist B C = 27) 
  (h_I_center : incenter I A B C) : 
  length_BI h_AB h_AC h_BC h_I_center = √131 :=
sorry

end BI_eq_sqrt_131_l24_24619


namespace find_P_l24_24940

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -3⟩

noncomputable def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

theorem find_P :
  ∃ (P : Point), distance P A = (3 / 2) * distance P B ∧ P = ⟨16 / 5, 0⟩ :=
begin
  sorry,
end

end find_P_l24_24940


namespace max_distance_triangle_l24_24615

theorem max_distance_triangle (A B C: Point) (P : Point) 
  (AB BC AC : ℝ) : 
  ∠ABC = 90 ∧ AC = 10 ∧ AB = 8 ∧ BC = 6 ∧ d_parallel = 1 ∧ between P A'P ∧ between P C'P
  → max_distance_to_sides P ABC = 7 :=
by
  -- Definitions and conditions
  sorry

end max_distance_triangle_l24_24615


namespace find_natural_number_n_l24_24462

theorem find_natural_number_n (n x y : ℕ) (h1 : n + 195 = x^3) (h2 : n - 274 = y^3) : 
  n = 2002 :=
by
  sorry

end find_natural_number_n_l24_24462


namespace find_circumradius_l24_24163

theorem find_circumradius {A B C : ℝ} {a b c : ℝ} 
  (hB : B = π / 3)
  (hDotProduct : (a * b * Real.cos (π - B)) = -2)
  (hSin : Real.sin A + Real.sin C = 2 * Real.sin B) :
  let R := a / (2 * Real.sin A) in
  R = 2 * Real.sqrt 3 / 3 :=
sorry

end find_circumradius_l24_24163


namespace angle_bisector_ABK_altitude_BF_l24_24506

noncomputable def isosceles_trapezoid : Type :=
{ 
  ABCD : Type,
  AD : ℝ,
  BC : ℝ,
  AB : ℝ,
  CD : ℝ,
  AB_eq_CD : AB = CD,
  intersection_K : Point,
  angle_BAD_bisector_intersects_K : True,
}

theorem angle_bisector_ABK_altitude_BF (ABCD : isosceles_trapezoid) (AD BC AB CD : ℝ) (intersection_K : Point)
  (hAD : AD = 10) (hBC : BC = 2) (hAB : AB = 5) (hCD : CD = 5) : 
  exists (BF : ℝ), BF = sqrt(10) / 2 :=
by
  sorry

end angle_bisector_ABK_altitude_BF_l24_24506


namespace bake_sale_comparison_l24_24844

theorem bake_sale_comparison :
  let tamara_small_brownies := 4 * 2
  let tamara_large_brownies := 12 * 3
  let tamara_cookies := 36 * 1.5
  let tamara_total := tamara_small_brownies + tamara_large_brownies + tamara_cookies

  let sarah_muffins := 24 * 1.75
  let sarah_choco_cupcakes := 7 * 2.5
  let sarah_vanilla_cupcakes := 8 * 2
  let sarah_strawberry_cupcakes := 15 * 2.75
  let sarah_total := sarah_muffins + sarah_choco_cupcakes + sarah_vanilla_cupcakes + sarah_strawberry_cupcakes

  sarah_total - tamara_total = 18.75 := by
  sorry

end bake_sale_comparison_l24_24844


namespace hexagon_vertices_zero_l24_24337

theorem hexagon_vertices_zero (n : ℕ) (a0 a1 a2 a3 a4 a5 : ℕ) 
  (h_sum : a0 + a1 + a2 + a3 + a4 + a5 = n) 
  (h_pos : 0 < n) :
  (n = 2 ∨ n % 2 = 1) → 
  ∃ (b0 b1 b2 b3 b4 b5 : ℕ), b0 = 0 ∧ b1 = 0 ∧ b2 = 0 ∧ b3 = 0 ∧ b4 = 0 ∧ b5 = 0 := sorry

end hexagon_vertices_zero_l24_24337


namespace triangle_area_ratio_l24_24342

noncomputable def area_ratio (a b c d e f : ℕ) : ℚ :=
  (a * b) / (d * e)

theorem triangle_area_ratio : area_ratio 6 8 10 9 12 15 = 4 / 9 :=
by
  sorry

end triangle_area_ratio_l24_24342


namespace fair_die_probability_l24_24570

noncomputable def probability_of_rolling_four_ones_in_five_rolls 
  (prob_1 : ℚ) (prob_not_1 : ℚ) (n : ℕ) (k : ℕ) : ℚ :=
(binomial n k) * (prob_1 ^ k) * (prob_not_1 ^ (n - k))

theorem fair_die_probability :
  let n := 5
  let k := 4
  let prob_1 := 1 / 6
  let prob_not_1 := 5 / 6
  probability_of_rolling_four_ones_in_five_rolls prob_1 prob_not_1 n k = 25 / 7776 := by
  sorry

end fair_die_probability_l24_24570


namespace f_increasing_solve_inequality_l24_24498

variable {f : ℝ → ℝ}

-- Given Conditions
axiom f_property : ∀ (m n : ℝ), f(m + n) = f(m) + f(n) - 1
axiom f_positive : ∀ x > 0, f(x) > 1
axiom f_at_3 : f 3 = 4

-- (1) Prove f(x) is increasing
theorem f_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2) := sorry

-- (2) Solve inequality f(a^2 + a - 5) < 2 given f(3) = 4
theorem solve_inequality (a : ℝ) : f(a^2 + a - 5) < 2 ↔ -3 < a ∧ a < 2 := sorry

end f_increasing_solve_inequality_l24_24498


namespace simplify_expression_equals_l24_24695

noncomputable def simplify_expression : ℝ :=
  (√3 - 1)^(1 - √2) / (√3 + 1)^(1 + √2)

theorem simplify_expression_equals :
  simplify_expression = 4 - 2 * √3 :=
  sorry

end simplify_expression_equals_l24_24695


namespace cassie_nail_cutting_l24_24870

structure AnimalCounts where
  dogs : ℕ
  dog_nails_per_foot : ℕ
  dog_feet : ℕ
  parrots : ℕ
  parrot_claws_per_leg : ℕ
  parrot_legs : ℕ
  extra_toe_parrots : ℕ
  extra_toe_claws : ℕ

def total_nails (counts : AnimalCounts) : ℕ :=
  counts.dogs * counts.dog_nails_per_foot * counts.dog_feet +
  counts.parrots * counts.parrot_claws_per_leg * counts.parrot_legs +
  counts.extra_toe_parrots * counts.extra_toe_claws

theorem cassie_nail_cutting :
  total_nails {
    dogs := 4,
    dog_nails_per_foot := 4,
    dog_feet := 4,
    parrots := 8,
    parrot_claws_per_leg := 3,
    parrot_legs := 2,
    extra_toe_parrots := 1,
    extra_toe_claws := 1
  } = 113 :=
by sorry

end cassie_nail_cutting_l24_24870


namespace arcsin_neg_one_eq_neg_pi_div_two_l24_24415

theorem arcsin_neg_one_eq_neg_pi_div_two : arcsin (-1) = - (Real.pi / 2) := sorry

end arcsin_neg_one_eq_neg_pi_div_two_l24_24415


namespace dave_tiles_area_l24_24874

-- Definition of the problem conditions
def ratio_clara_to_dave : ℕ × ℕ := (4, 7)
def total_area : ℕ := 330

-- The proof goal to show the area tiled by Dave
theorem dave_tiles_area : 
  let (a, b) := ratio_clara_to_dave in
  let total_parts := a + b in
  let each_part_area := total_area / total_parts in
  let dave_parts := b in
  dave_parts * each_part_area = 210 := 
by
  -- We leave the proof as a sorry placeholder
  sorry

end dave_tiles_area_l24_24874


namespace distance_from_point_to_directrix_l24_24957

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24957


namespace range_f_l24_24129

noncomputable def f (x : ℝ) : ℝ := (2 * x^2 - 1) / (x^2 + 2)

theorem range_f :
  set.range f = set.Ico (-1/2 : ℝ) 2 :=
sorry

end range_f_l24_24129


namespace paint_o_circles_l24_24784

theorem paint_o_circles :
  let colors := 3
  let circles := 4
  colors ^ circles = 81 :=
by 
  cases colors
  cases circles
  sorry 

end paint_o_circles_l24_24784


namespace product_series_value_l24_24409

theorem product_series_value :
  let product := 
    ((1 * 3 : ℝ) / (2 * 3)) * ((2 * 4) / (3 * 4)) * ((3 * 5) / (4 * 5)) * 
    (∏ n in Finset.range 99, ((n + 1) * (n + 3)) / ((n + 2) * (n + 3))) * 
    ((100 * 102) / (101 * 102)) * (103 / 101)
  in product = 103 / 10201 := 
by
  sorry

end product_series_value_l24_24409


namespace extreme_values_for_f_when_a_is_one_number_of_zeros_of_h_l24_24508

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / Real.exp x + x^2 / 2 - x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := max (f a x) (g x)

theorem extreme_values_for_f_when_a_is_one :
  (∀ x : ℝ, (f 1 x) ≤ 0) ∧ f 1 0 = 0 ∧ f 1 1 = (1 / Real.exp 1) - 1 / 2 :=
sorry

theorem number_of_zeros_of_h (a : ℝ) :
  (0 ≤ a → 
   if 1 < a ∧ a < Real.exp 1 / 2 then
     ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧ h a x1 = 0 ∧ h a x2 = 0
   else if 0 ≤ a ∧ a ≤ 1 ∨ a = Real.exp 1 / 2 then
     ∃ x : ℝ, 0 < x ∧ x < 1 ∧ h a x = 0
   else
     ∀ x : ℝ, x > 0 → h a x ≠ 0) :=
sorry

end extreme_values_for_f_when_a_is_one_number_of_zeros_of_h_l24_24508


namespace surface_area_bound_l24_24325

theorem surface_area_bound
  (a b c d : ℝ)
  (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) (h4: 0 ≤ d) 
  (h_quad: a + b + c > d) : 
  2 * (a * b + b * c + c * a) ≤ (a + b + c) ^ 2 - (d ^ 2) / 3 :=
sorry

end surface_area_bound_l24_24325


namespace perpendicular_planes_line_condition_l24_24212

variables {α β : Type} [plane α] [plane β]
variables {m n l : line} 

-- Formalizing the given conditions as premises and the goal as the conclusion
theorem perpendicular_planes_line_condition (h_perp_planes : α ⊥ β)
                                          (h_line_intersect : α ∩ β = m)
                                          (h_n_in_alpha : n ⊂ α)
                                          (h_m_perp_n : m ⊥ n) :
  n ⊥ β :=
sorry

end perpendicular_planes_line_condition_l24_24212


namespace sum_div_minuend_eq_two_l24_24743

variable (Subtrahend Minuend Difference : ℝ)

theorem sum_div_minuend_eq_two
  (h : Subtrahend + Difference = Minuend) :
  (Subtrahend + Minuend + Difference) / Minuend = 2 :=
by
  sorry

end sum_div_minuend_eq_two_l24_24743


namespace maximize_S_minimize_S_l24_24473

open Finset

def permutation (n : ℕ) (a : Fin n → Fin n) : Prop :=
  (∀ i j : Fin n, i ≠ j → a i ≠ a j) ∧ (∀ i : Fin n, ∃ j : Fin n, a j = i)

def S (n : ℕ) (a : Fin n → Fin n) : ℕ :=
  (Finset.univ \.sum (λ i, a i * a (Finset.cycle n i)))

def max_permutation (n : ℕ) (a : Fin n → Fin n) : Prop :=
  (∀ b : Fin n → Fin n, permutation n b → S n a ≥ S n b)

def min_permutation (n : ℕ) (a : Fin n → Fin n) : Prop :=
  (∀ b : Fin n → Fin n, permutation n b → S n a ≤ S n b)

theorem maximize_S (n : ℕ) :
  ∃ a : Fin n → Fin n, permutation n a ∧ max_permutation n a :=
sorry

theorem minimize_S (n : ℕ) :
  ∃ a : Fin n → Fin n, permutation n a ∧ min_permutation n a :=
sorry

end maximize_S_minimize_S_l24_24473


namespace probability_at_least_one_boy_and_girl_l24_24839
-- Necessary imports

-- Defining the probability problem in Lean 4
theorem probability_at_least_one_boy_and_girl (n : ℕ) (hn : n = 4)
    (p : ℚ) (hp : p = 1 / 2) :
    let prob_all_same := (p ^ n) + (p ^ n) in
    (1 - prob_all_same) = 7 / 8 := by
  -- Include the proof steps here
  sorry

end probability_at_least_one_boy_and_girl_l24_24839


namespace seq_values_seq_general_formula_seq_limit_l24_24138

-- Define the sequence with conditions
def seq (a : ℕ → ℕ) :=
  a 2 = 6 ∧ ∀ n, (a (n + 1) + a n - 1) / (a (n + 1) - a n + 1) = n

-- Problem 1
theorem seq_values {a : ℕ → ℕ} (h : seq a) : 
  a 1 = 1 ∧ a 3 = 15 ∧ a 4 = 28 :=
sorry

-- Problem 2
theorem seq_general_formula {a : ℕ → ℕ} (h : seq a) : 
  ∀ n, a n = n * (2 * n - 1) :=
sorry

-- Problem 3
theorem seq_limit {a : ℕ → ℕ} (h : seq a) : 
  tendsto (λ n : ℕ, (∑ i in range n, 1 / (a (i + 2) - (i + 2)))) at_top (𝓝 (1 / 2)) :=
sorry

end seq_values_seq_general_formula_seq_limit_l24_24138


namespace cameron_list_length_l24_24003

-- Definitions of multiples
def smallest_multiple_perfect_square := 900
def smallest_multiple_perfect_cube := 27000
def multiple_of_30 (n : ℕ) : Prop := n % 30 = 0

-- Problem statement
theorem cameron_list_length :
  ∀ n, 900 ≤ n ∧ n ≤ 27000 ∧ multiple_of_30 n ->
  (871 = (900 - 30 + 1)) :=
sorry

end cameron_list_length_l24_24003


namespace find_min_of_S_l24_24478

noncomputable def S (k : ℝ) : ℝ :=
let x1 := Real.arcsin (k / 2) in
let x3 := π - Real.arcsin (k / 2) in
  (∫ x in 0..x1, k * Real.cos x - Real.sin (2 * x))
  + (∫ x in x1..(π / 2), Real.sin (2 * x) - k * Real.cos x)
  + |∫ x in (π / 2)..x3, k * Real.cos x - Real.sin (2 * x)|
  + |∫ x in x3..π, Real.sin (2 * x) - k * Real.cos x|

theorem find_min_of_S : ∃ k, 0 < k ∧ k < 2 ∧ S k = -1.0577 := 
sorry

end find_min_of_S_l24_24478


namespace market_cost_l24_24664

theorem market_cost (peach_pies apple_pies blueberry_pies : ℕ) (fruit_per_pie : ℕ) 
  (price_per_pound_apple price_per_pound_blueberry price_per_pound_peach : ℕ) :
  peach_pies = 5 ∧
  apple_pies = 4 ∧
  blueberry_pies = 3 ∧
  fruit_per_pie = 3 ∧
  price_per_pound_apple = 1 ∧
  price_per_pound_blueberry = 1 ∧
  price_per_pound_peach = 2 →
  let total_peaches := peach_pies * fruit_per_pie in
  let total_apples := apple_pies * fruit_per_pie in
  let total_blueberries := blueberry_pies * fruit_per_pie in
  let cost_apples := total_apples * price_per_pound_apple in
  let cost_blueberries := total_blueberries * price_per_pound_blueberry in
  let cost_peaches := total_peaches * price_per_pound_peach in
  (cost_apples + cost_blueberries + cost_peaches = 51) :=
by
  intros
  sorry

end market_cost_l24_24664


namespace polynomial_has_zero_l24_24807

theorem polynomial_has_zero (r s : ℤ) : 
    ∃ α β : ℤ, ∃ P : polynomial ℤ, 
    P = (polynomial.X - polynomial.C r) * (polynomial.X - polynomial.C s) * 
        (polynomial.X^2 + polynomial.C α * polynomial.X + polynomial.C β) ∧
    P.eval ((1 + complex.I * real.sqrt 7) / 2) = 0 :=
sorry

end polynomial_has_zero_l24_24807


namespace part1_parallel_part2_perpendicular_min_ab_l24_24135

-- Define the lines and conditions
def line1 (a : ℝ) (x y : ℝ) := x + a^2 * y + 1 = 0
def line2 (a b : ℝ) (x y : ℝ) := (a^2 + 1) * x - b * y + 3 = 0

-- Slopes of the lines
def slope_line1 (a : ℝ) : ℝ := -(1 / a^2)
def slope_line2 (a b : ℝ) : ℝ := (a^2 + 1) / b

-- Part 1: Prove that when b = -2 and l1 || l2, the value of a is ±1
theorem part1_parallel (a : ℝ) (h_parallel : slope_line1 a = slope_line2 a (-2)) :
  (a = 1 ∨ a = -1) := by {
  sorry
}

-- Part 2: Prove that if l1 ⊥ l2, the minimum value of |ab| is 2
theorem part2_perpendicular_min_ab (a b : ℝ) (h_perpendicular : slope_line1 a * slope_line2 a b = -1) :
  |a * b| = 2 := by {
  sorry
}

end part1_parallel_part2_perpendicular_min_ab_l24_24135


namespace num_domino_arrangements_l24_24263

theorem num_domino_arrangements :
  ∃ n : ℕ, n = 126 ∧ ∀ (grid : array 6 (array 5 (option (nat × nat)))),
  ∃ path : list (nat × nat),
  (∀ i, i < 4 → path[i].1 = path[i-1].1 + 1) ∧
  (∀ i, 4 ≤ i < 9 → path[i].2 = path[i-1].2 + 1) →
  path.length = 9  →
  ∀ i, (path[i].1, path[i].2) ∈ grid ∧
  (∀ i, path[i], path[i+1] touch_sides grid ∧
  ∀ i j, path[i] ≠ path[j] ∧ (∃ dominoes, dominoes.length = 5 ∧
  (∃ arrange, arrange = path.length / 2) →
  arrange = n :=
sorry

end num_domino_arrangements_l24_24263


namespace distance_from_A_to_directrix_l24_24996

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l24_24996


namespace distance_from_point_to_directrix_l24_24956

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24956


namespace problem_statement_l24_24073

theorem problem_statement (m a b : ℝ) (h0 : 9^m = 10) (h1 : a = 10^m - 11) (h2 : b = 8^m - 9) : a > 0 ∧ 0 > b :=
by
  sorry

end problem_statement_l24_24073


namespace car_price_difference_l24_24258

variable (original_paid old_car_proceeds : ℝ)
variable (new_car_price additional_amount : ℝ)

theorem car_price_difference :
  old_car_proceeds = new_car_price - additional_amount →
  old_car_proceeds = 0.8 * original_paid →
  additional_amount = 4000 →
  new_car_price = 30000 →
  (original_paid - new_car_price) = 2500 :=
by
  intro h1 h2 h3 h4
  sorry

end car_price_difference_l24_24258


namespace distance_from_A_to_directrix_l24_24998

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l24_24998


namespace find_y_l24_24154

theorem find_y (x y : ℝ) (h1 : x^2 - 4 * x = y + 5) (h2 : x = 7) : y = 16 := by
  sorry

end find_y_l24_24154


namespace problem_statement_l24_24086

theorem problem_statement (m : ℝ) (a b : ℝ) 
  (h1 : 9^m = 10) (h2 : a = 10^m - 11) (h3 : b = 8^m - 9) : 
  a > 0 ∧ 0 > b :=
sorry

end problem_statement_l24_24086


namespace max_length_self_avoiding_polygon_l24_24357

-- Definitions for the conditions in the problem

-- A grid of size 8x8
def grid : Type := fin 8 × fin 8

-- A closed self-avoiding polygon on a grid
structure SelfAvoidingPolygon :=
(vertices : list grid)
(is_self_avoiding : ∀ (p1 p2 : grid), p1 ≠ p2 → vertices.indexOf p1 ≠ vertices.indexOf p2)
(is_closed : vertices.head = vertices.last)

-- The maximum length of such a polygon
def maxLengthOfClosedSelfAvoidingPolygon (g : grid) : ℕ :=
  sorry

-- Theorem statement
theorem max_length_self_avoiding_polygon (boundary : grid) :
  maxLengthOfClosedSelfAvoidingPolygon boundary = 80 :=
by sorry

end max_length_self_avoiding_polygon_l24_24357


namespace find_m_of_transformed_point_eq_l24_24588

theorem find_m_of_transformed_point_eq (m : ℝ) (h : m + 1 = 5) : m = 4 :=
by
  sorry

end find_m_of_transformed_point_eq_l24_24588


namespace hexagon_area_is_32_l24_24450

noncomputable def area_of_hexagon : ℝ := 
  let p0 : ℝ × ℝ := (0, 0)
  let p1 : ℝ × ℝ := (2, 4)
  let p2 : ℝ × ℝ := (5, 4)
  let p3 : ℝ × ℝ := (7, 0)
  let p4 : ℝ × ℝ := (5, -4)
  let p5 : ℝ × ℝ := (2, -4)
  -- Triangle 1: p0, p1, p2
  let area_tri1 := 1 / 2 * (3 : ℝ) * (4 : ℝ)
  -- Triangle 2: p2, p3, p4
  let area_tri2 := 1 / 2 * (8 : ℝ) * (2 : ℝ)
  -- Triangle 3: p4, p5, p0
  let area_tri3 := 1 / 2 * (3 : ℝ) * (4 : ℝ)
  -- Triangle 4: p1, p2, p5
  let area_tri4 := 1 / 2 * (8 : ℝ) * (3 : ℝ)
  area_tri1 + area_tri2 + area_tri3 + area_tri4

theorem hexagon_area_is_32 : area_of_hexagon = 32 := 
by
  sorry

end hexagon_area_is_32_l24_24450


namespace complex_conjugate_of_fraction_l24_24899

noncomputable def conjugate_z (z : ℂ) := conj z

theorem complex_conjugate_of_fraction :
  let z := ((-3 + complex.I) / (2 + complex.I)) in
  conjugate_z z = (-1 - complex.I) := by
  sorry

end complex_conjugate_of_fraction_l24_24899


namespace OA_perp_BC_l24_24209

variables {ABC : Type*} [Triangle ABC] 
variables {A B C A' B' C' H O : ABC}

-- Assuming the necessary geometric properties are defined in the Triangle structure, 
-- such as feet of altitudes, orthocenter, and circumcenter

def is_feet_of_altitude (X Y Z : ABC) : Prop := -- definition for feet of altitude
sorry

def is_orthocenter (H : ABC) (Δ : Triangle ABC) : Prop := -- definition for orthocenter
sorry

def is_circumcenter (O : ABC) (Δ : Triangle ABC) : Prop := -- definition for circumcenter
sorry

-- Now the actual statement 
theorem OA_perp_BC' (ABC : Triangle ABC) (A B C A' B' C' H O : ABC)
  (hA' : is_feet_of_altitude A' A B C)
  (hB' : is_feet_of_altitude B' B A C)
  (hC' : is_feet_of_altitude C' C A B)
  (hH : is_orthocenter H ABC)
  (hO : is_circumcenter O ABC) : 
  is_perpendicular (line_through O A) (line_through B' C') :=
begin
  sorry  -- proof not required
end

end OA_perp_BC_l24_24209


namespace question_1_question_2_question_3_l24_24126

theorem question_1 (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + (m - 1) < 1) ↔ 
    m < (1 - 2 * Real.sqrt 7) / 3 := sorry

theorem question_2 (m : ℝ) : 
  ∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + (m - 1) ≥ (m + 1) * x := sorry

theorem question_3 (m : ℝ) : 
  (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2), (m + 1) * x^2 - (m - 1) * x + (m - 1) ≥ 0) ↔ 
    m ≥ 1 := sorry

end question_1_question_2_question_3_l24_24126


namespace martinez_family_combined_height_l24_24321

def chiquita_height := 5
def mr_martinez_height := chiquita_height + 2
def mrs_martinez_height := chiquita_height - 1
def son_height := chiquita_height + 3
def combined_height := chiquita_height + mr_martinez_height + mrs_martinez_height + son_height

theorem martinez_family_combined_height : combined_height = 24 :=
by
  sorry

end martinez_family_combined_height_l24_24321


namespace distance_from_point_to_directrix_l24_24991

noncomputable def parabola_point_distance_to_directrix (x y p : ℝ) : ℝ :=
  x + p / 2

theorem distance_from_point_to_directrix (x y p dist_to_directrix : ℝ)
  (hx : y^2 = 2 * p * x)
  (hdist_def : dist_to_directrix = parabola_point_distance_to_directrix x y p) :
  dist_to_directrix = 9 / 4 :=
by
  sorry

end distance_from_point_to_directrix_l24_24991


namespace general_formula_l24_24537

theorem general_formula (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n + 1) :
  ∀ n : ℕ, a (n + 1) = 2^(n + 1) - 1 :=
by
  sorry

end general_formula_l24_24537


namespace sum_of_angles_ADB_BEC_CFA_l24_24617

theorem sum_of_angles_ADB_BEC_CFA {A B C D E F : Type} 
  {AB BC AD AC BE BA CF CB : ℝ}
  (h_iso : AB = BC)
  (h_D_on_CA : AD = AC)
  (h_E_on_AB : BE = BA)
  (h_F_on_BC : CF = CB) :
  let α := ∠BAC in
  ∠ADB + ∠BEC + ∠CFA = 90° :=
sorry

end sum_of_angles_ADB_BEC_CFA_l24_24617


namespace jaime_average_speed_l24_24197

theorem jaime_average_speed :
  let start_time := 10.0 -- 10:00 AM
  let end_time := 15.5 -- 3:30 PM (in 24-hour format)
  let total_distance := 21.0 -- kilometers
  let total_time := end_time - start_time -- time in hours
  total_distance / total_time = 3.82 := 
sorry

end jaime_average_speed_l24_24197


namespace hotel_rolls_probability_l24_24384

theorem hotel_rolls_probability :
  let rolls := (finset.range 12).image (λ i, if i < 3 then 'nut' else if i < 6 then 'cheese' else if i < 9 then 'fruit' else 'seed')
  let guest1 := (finset.range 4).image (λ i, if i < 1 then 'nut' else if i < 2 then 'cheese' else if i < 3 then 'fruit' else 'seed')
  let guest2 := (finset.range 4).image (λ i, if i < 1 then 'nut' else if i < 2 then 'cheese' else if i < 3 then 'fruit' else 'seed')
  let guest3 := (finset.range 4).image (λ i, if i < 1 then 'nut' else if i < 2 then 'cheese' else if i < 3 then 'fruit' else 'seed')
  let all_guests := guest1 ∪ guest2 ∪ guest3
  finset.disjoint guest1 guest2 ∧ finset.disjoint guest2 guest3 ∧ finset.disjoint guest1 guest3 ∧ all_guests = rolls →
  (∃ m n: ℕ, let gcd := nat.gcd m n in gcd = 1 ∧ (2, 103950) = (m, n) ∧ m + n = 103952)
by
  sorry

end hotel_rolls_probability_l24_24384


namespace count_palindrome_five_digit_div_by_5_l24_24240

-- Define what it means for a number to be palindromic.
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Define what it means for a number to be a five-digit number.
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

-- Define what it means for a number to be divisible by 5.
def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Define the set of five-digit palindromic numbers divisible by 5.
def palindrome_five_digit_div_by_5_numbers (n : ℕ) : Prop :=
  is_five_digit n ∧ is_palindrome n ∧ is_divisible_by_5 n

-- Prove that the quantity of such numbers is 100.
theorem count_palindrome_five_digit_div_by_5 : 
  (finset.filter 
    (λ n, palindrome_five_digit_div_by_5_numbers n)
    (finset.range 100000)
  ).card = 100 :=
begin
  sorry
end

end count_palindrome_five_digit_div_by_5_l24_24240


namespace parallel_I₁I₂_O₁O₂_l24_24175

variables {A B C D E F I₁ I₂ O₁ O₂ : Type*}
variables [Triangle A B C]
variables [feet_of_altitudes D E F A B C]
variables [incenter_of_triangle I₁ A E F]
variables [incenter_of_triangle I₂ B D F]
variables [circumcenter_of_triangle O₁ A C I₁]
variables [circumcenter_of_triangle O₂ B C I₂]

theorem parallel_I₁I₂_O₁O₂ : are_parallel (line_through I₁ I₂) (line_through O₁ O₂) :=
sorry

end parallel_I₁I₂_O₁O₂_l24_24175


namespace distance_from_point_to_directrix_l24_24949

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24949


namespace cassie_nail_cutting_l24_24869

structure AnimalCounts where
  dogs : ℕ
  dog_nails_per_foot : ℕ
  dog_feet : ℕ
  parrots : ℕ
  parrot_claws_per_leg : ℕ
  parrot_legs : ℕ
  extra_toe_parrots : ℕ
  extra_toe_claws : ℕ

def total_nails (counts : AnimalCounts) : ℕ :=
  counts.dogs * counts.dog_nails_per_foot * counts.dog_feet +
  counts.parrots * counts.parrot_claws_per_leg * counts.parrot_legs +
  counts.extra_toe_parrots * counts.extra_toe_claws

theorem cassie_nail_cutting :
  total_nails {
    dogs := 4,
    dog_nails_per_foot := 4,
    dog_feet := 4,
    parrots := 8,
    parrot_claws_per_leg := 3,
    parrot_legs := 2,
    extra_toe_parrots := 1,
    extra_toe_claws := 1
  } = 113 :=
by sorry

end cassie_nail_cutting_l24_24869


namespace compute_fraction_l24_24232

theorem compute_fraction (a b : ℚ) (ha : a = 4/7) (hb : b = 5/6) : 
  a ^ (-3) * b ^ 2 = 8575 / 2304 := by 
  sorry

end compute_fraction_l24_24232


namespace problem_statement_l24_24080

variable (m : ℝ) (a b : ℝ)

-- Given conditions
def condition1 : Prop := 9^m = 10
def condition2 : Prop := a = 10^m - 11
def condition3 : Prop := b = 8^m - 9

-- Problem statement to prove
theorem problem_statement (h1 : condition1 m) (h2 : condition2 m a) (h3 : condition3 m b) : a > 0 ∧ 0 > b := 
sorry

end problem_statement_l24_24080


namespace soccer_players_selection_l24_24823

theorem soccer_players_selection (N : ℕ) (hN : N ≥ 2) 
  (players : Finset ℕ) (unique_heights : players.card = N * (N + 1)) : 
  ∃ selected_players : Finset ℕ, selected_players.card = 2 * N ∧
  (∀ i : ℕ, (1 ≤ i ∧ i ≤ N) → ∀ x y, x ≠ y → (x, y) ∈ selected_players.pairs ∧ 
  ((x = taller i ∧ y = taller (i + 1)) ∨ (x = taller (N - i + 1) ∧ y = taller (N - i + 2)))) := 
sorry

end soccer_players_selection_l24_24823


namespace correct_propositions_l24_24525

variable {a b : Type}
variable {α : Type}

def parallel (a b : Type) : Prop := sorry
def perpendicular (a b : Type) : Prop := sorry

axiom proposition1 (h1 : parallel a α) (h2 : perpendicular b α) : perpendicular a b
axiom proposition2 (h1 : perpendicular a b) (h2 : perpendicular b α) : parallel a α
axiom proposition3 (h1 : parallel a b) (h2 : perpendicular b α) : perpendicular a α
axiom proposition4 (h1 : perpendicular a b) (h2 : parallel b α) : perpendicular a α

theorem correct_propositions : 
  (proposition1 (sorry) (sorry)) ∧ (proposition3 (sorry) (sorry)) :=
by
  sorry

end correct_propositions_l24_24525


namespace combinatorial_identity_l24_24854

theorem combinatorial_identity :
  (Nat.factorial 15) / ((Nat.factorial 6) * (Nat.factorial 9)) = 5005 :=
sorry

end combinatorial_identity_l24_24854


namespace gcd_pens_pencils_l24_24315

theorem gcd_pens_pencils (pens : ℕ) (pencils : ℕ) (h1 : pens = 1048) (h2 : pencils = 828) : Nat.gcd pens pencils = 4 := 
by
  -- Given: pens = 1048 and pencils = 828
  have h : pens = 1048 := h1
  have h' : pencils = 828 := h2
  sorry

end gcd_pens_pencils_l24_24315


namespace distance_from_A_to_directrix_of_C_l24_24963

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24963


namespace eval_diff_of_squares_l24_24032

theorem eval_diff_of_squares : (81^2 - 49^2 = 4160) :=
by
  have a : Int := 81
  have b : Int := 49
  have h : a^2 - b^2 = (a + b) * (a - b) := by
    exact Int.sub_eq (a * a) (b * b)
  calc
    81^2 - 49^2
        = (81 + 49) * (81 - 49) : by rw h
    ... = 130 * 32 : by norm_num
    ... = 4160 : by norm_num

end eval_diff_of_squares_l24_24032


namespace tangent_product_l24_24440

theorem tangent_product (n θ : ℝ) 
  (h1 : ∀ n θ, tan (n * θ) = (sin (n * θ)) / (cos (n * θ)))
  (h2 : tan 8 θ = (8 * tan θ - 56 * (tan θ) ^ 3 + 56 * (tan θ) ^ 5 - 8 * (tan θ) ^ 7) / 
                  (1 - 28 * (tan θ) ^ 2 + 70 * (tan θ) ^ 4 - 28 * (tan θ) ^ 6))
  (h3 : tan (8 * (π / 8)) = 0)
  (h4 : tan (8 * (3 * π / 8)) = 0)
  (h5 : tan (8 * (5 * π / 8)) = 0) :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = 2 * sqrt 2 :=
sorry

end tangent_product_l24_24440


namespace value_of_MN_l24_24564

theorem value_of_MN
  (M N : ℝ)
  (h1 : log M (N ^ 2) = log N (M ^ 2))
  (h2 : M ≠ N)
  (h3 : M * N > 0)
  (h4 : M ≠ 1)
  (h5 : N ≠ 1) :
  M * N = 1 := 
  by sorry

end value_of_MN_l24_24564


namespace range_of_a_l24_24526

def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 0 then a * x + 2 - 3 * a else 2^x - 1

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) → a ∈ Set.Iio (2 / 3) :=
by
  intro h
  sorry

end range_of_a_l24_24526


namespace sum_induction_l24_24759

theorem sum_induction (n : ℕ) (h : n > 0) : 
  (∑ i in Finset.range n, 1 / ((2 * i + 1) * (2 * i + 3))) = n / (2 * n + 1) :=
sorry

end sum_induction_l24_24759


namespace probability_complement_A_l24_24119

variables {Ω : Type*} [MeasurableSpace Ω] {P : ProbabilityMeasure Ω}
variables (A B : Set Ω)

-- Conditions
def mutually_exclusive (A B : Set Ω) : Prop := ∀ ω, ω ∈ A → ω ∉ B

theorem probability_complement_A :
  mutually_exclusive A B →
  P.probability (A ∪ B) = 0.8 →
  P.probability B = 0.3 →
  P.probability (Aᶜ) = 0.5 :=
by
  sorry

end probability_complement_A_l24_24119


namespace gcd_polynomial_l24_24910

theorem gcd_polynomial (n : ℕ) (h : n > 2^5) :
  Nat.gcd (n^5 + 125) (n + 5) = if n % 5 = 0 then 5 else 1 :=
by
  sorry

end gcd_polynomial_l24_24910


namespace exists_five_digit_palindromic_divisible_by_5_count_five_digit_palindromic_numbers_divisible_by_5_l24_24254

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem exists_five_digit_palindromic_divisible_by_5 :
  ∃ n, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n := by
  -- Proof is omitted
  sorry

theorem count_five_digit_palindromic_numbers_divisible_by_5 :
  (finset.filter (λ n, is_five_digit n ∧ is_palindromic n ∧ is_divisible_by_5 n) (finset.range 100000)).card = 100 := by
  -- Proof is omitted
  sorry

end exists_five_digit_palindromic_divisible_by_5_count_five_digit_palindromic_numbers_divisible_by_5_l24_24254


namespace decreasing_functions_l24_24524

noncomputable def f1 (x : ℝ) : ℝ := -x^2 + 1
noncomputable def f2 (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def f4 (x : ℝ) : ℝ := 3 ^ x

theorem decreasing_functions :
  (∀ x y : ℝ, 0 < x → x < y → f1 y < f1 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f2 y > f2 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f3 y > f3 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f4 y > f4 x) :=
by {
  sorry
}

end decreasing_functions_l24_24524


namespace max_area_2017_2018_l24_24819

noncomputable def max_area_of_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

theorem max_area_2017_2018 :
  max_area_of_triangle 2017 2018 = 2035133 := by
  sorry

end max_area_2017_2018_l24_24819


namespace total_amount_spent_l24_24280

noncomputable def value_of_nickel : ℕ := 5
noncomputable def value_of_dime : ℕ := 10
noncomputable def initial_amount : ℕ := 250

def amount_spent_by_Pete (nickels_spent : ℕ) : ℕ :=
  nickels_spent * value_of_nickel

def amount_remaining_with_Raymond (dimes_left : ℕ) : ℕ :=
  dimes_left * value_of_dime

theorem total_amount_spent (nickels_spent : ℕ) (dimes_left : ℕ) :
  (amount_spent_by_Pete nickels_spent + 
   (initial_amount - amount_remaining_with_Raymond dimes_left)) = 200 :=
by
  sorry

end total_amount_spent_l24_24280


namespace boatsman_speed_l24_24376

-- Definitions for the problem conditions
def upstream_time (v : ℝ) := 40 / (v - 3)
def downstream_time (v : ℝ) := 40 / (v + 3)
def time_difference (v : ℝ) := upstream_time v - downstream_time v

-- Proof statement
theorem boatsman_speed (v : ℝ) (h : time_difference v = 6) : v = 7 := by
  sorry

end boatsman_speed_l24_24376


namespace intersection_A_B_eq_C_l24_24511

def A : Set ℝ := { x | 4 - x^2 ≥ 0 }
def B : Set ℝ := { x | x > -1 }
def C : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_A_B_eq_C : A ∩ B = C := 
by {
  sorry
}

end intersection_A_B_eq_C_l24_24511


namespace exists_equal_sum_subsequences_l24_24100

-- Define the sequences and their properties
def sequence_x (x : ℕ → ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 19 → 1 ≤ x i ∧ x i ≤ 93

def sequence_y (y : ℕ → ℕ) : Prop :=
  ∀ (j : ℕ), 1 ≤ j ∧ j ≤ 93 → 1 ≤ y j ∧ y j ≤ 19

-- The proof statement
theorem exists_equal_sum_subsequences (x y : ℕ → ℕ) :
  sequence_x x → sequence_y y →
  ∃ (S T : finset ℕ), (S.nonempty ∧ T.nonempty) ∧ (S.sum x = T.sum y) :=
by
  intros hx hy
  sorry

end exists_equal_sum_subsequences_l24_24100


namespace find_circle_C_eq_l24_24112

-- Definition of a circle with given center and radius
def Circle (center : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
  { p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 }

-- Conditions
def C1_center : ℝ × ℝ := (3, -4)
def O_radius : ℝ := 1
def O_center : ℝ × ℝ := (0, 0)
def O : set (ℝ × ℝ) := Circle O_center O_radius

-- Noncomputable distance between centers
noncomputable def OC_distance : ℝ :=
  Real.sqrt ((C1_center.1 - O_center.1)^2 + (C1_center.2 - O_center.2)^2)

-- External tangency radius of circle C
def Rc_extern : ℝ := OC_distance - O_radius
-- Internal tangency radius of circle C
def Rc_intern : ℝ := OC_distance + O_radius

-- Proof statement
theorem find_circle_C_eq : 
  (Circle C1_center Rc_extern = { p | (p.1 - 3)^2 + (p.2 + 4)^2 = 16 }) ∨ 
  (Circle C1_center Rc_intern = { p | (p.1 - 3)^2 + (p.2 + 4)^2 = 36 }) :=
sorry

end find_circle_C_eq_l24_24112


namespace sin_inequality_for_a_l24_24143

theorem sin_inequality_for_a (a : ℝ) : (∀ x : ℝ, sin x - 2 * a ≥ 0) → a ≤ -1 / 2 :=
sorry

end sin_inequality_for_a_l24_24143


namespace sum_of_integers_odd_then_product_even_product_expression_even_difference_sum_product_even_no_integer_solutions_l24_24789

-- Part 1(a)
theorem sum_of_integers_odd_then_product_even (a b : ℤ) (h : odd (a + b)) : even (a * b) :=
sorry

-- Part 1(b)
theorem product_expression_even (n : ℤ) : even ((3 * n - 1) * (5 * n + 2)) :=
sorry

-- Part 1(c)
theorem difference_sum_product_even (k n : ℤ) : even ((k - n) * (k + n + 1)) :=
sorry

-- Part 1(d)
theorem no_integer_solutions (x y : ℤ) : ¬ (315 = (x - y) * (x + y + 1)) :=
sorry

end sum_of_integers_odd_then_product_even_product_expression_even_difference_sum_product_even_no_integer_solutions_l24_24789


namespace correlation_examples_l24_24362

def correlation (X Y : Type) : Prop := sorry  -- Placeholder for a proper correlation definition.

-- Define the variables and relationships
def snowfall := Type
def traffic_accidents := Type
def brain_capacity := Type
def intelligence := Type
def age := Type
def weight := Type
def rainfall := Type
def crop_yield := Type

-- Define the relationships
def relationship1 := correlation snowfall traffic_accidents
def relationship2 := correlation brain_capacity intelligence
def relationship3 := correlation age weight
def relationship4 := correlation rainfall crop_yield

-- The theorem according to the given correct answer.
theorem correlation_examples : 
  relationship1 ∧ relationship2 ∧ relationship4 :=
  sorry

end correlation_examples_l24_24362


namespace tan_product_l24_24431

theorem tan_product :
  (Real.tan (Real.pi / 8)) * (Real.tan (3 * Real.pi / 8)) * (Real.tan (5 * Real.pi / 8)) = 1 :=
sorry

end tan_product_l24_24431


namespace hall_length_calculation_l24_24170

theorem hall_length_calculation
  (width height : ℕ)  -- defining width and height as natural numbers (in meters)
  (total_cost cost_per_sqm : ℕ) -- defining total_cost and cost_per_sqm as natural numbers (in Rs)
  (Hwidth : width = 15)
  (Hheight : height = 5)
  (Htotal_cost : total_cost = 38000)
  (Hcost_per_sqm : cost_per_sqm = 40) :
  ∃ (length : ℕ), length = 32 :=
begin
  -- translating the given conditions into the formal proof environment
  have h_total_area : total_cost / cost_per_sqm = 950,
  { rw [Htotal_cost, Hcost_per_sqm], exact rfl },

  -- expressing the total area covered with mat
  let floor_area := λ length, length * width,
  let wall_area := λ length, 2 * (length * height + width * height),

  have h_total_area_eqn : ∀ length : ℕ, floor_area length + wall_area length = 950,
  { intro length,
    simp [floor_area, wall_area, Hwidth, Hheight],
    ring,  -- simplifying the algebraic expression
    rw [Hwidth, Hheight],
    exact h_total_area },

  -- showing there exists a length such that the equation holds
  use 32,
  have h_length : 25 * 32 + 150 = 950,
  { ring },
  exact h_total_area_eqn 32,
end

end hall_length_calculation_l24_24170


namespace tan_product_l24_24430

theorem tan_product :
  (Real.tan (Real.pi / 8)) * (Real.tan (3 * Real.pi / 8)) * (Real.tan (5 * Real.pi / 8)) = 1 :=
sorry

end tan_product_l24_24430


namespace mary_regular_hourly_rate_l24_24262

-- Definitions based on the conditions
def max_regular_hours : ℕ := 20
def max_hours : ℕ := 60
def overtime_multiplier : ℝ := 1.25
def total_earnings : ℝ := 560

-- Let R be the regular hourly rate
variable (R : ℝ)

-- Sum of earnings from regular and overtime hours
def weekly_earnings (overtime_hours : ℕ) : ℝ :=
  max_regular_hours * R + (overtime_multiplier * R * overtime_hours)

-- Maximum overtime hours
def max_overtime_hours (max_hours - max_regular_hours) : ℕ := 40

-- The Hypothesis
hypothesis max_overtime_weekly_earnings : weekly_earnings R 40 = total_earnings

-- The Lean statement: Prove R = 8
theorem mary_regular_hourly_rate : R = 8 :=
  by
    sorry

end mary_regular_hourly_rate_l24_24262


namespace problem_l24_24079

theorem problem (m a b : ℝ) (h₀ : 9 ^ m = 10) (h₁ : a = 10 ^ m - 11) (h₂ : b = 8 ^ m - 9) :
  a > 0 ∧ 0 > b := 
sorry

end problem_l24_24079


namespace circles_intersect_l24_24320

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem circles_intersect :
  let center1 := (0, 2)
  let center2 := (-2, -1)
  let radius1 := 1
  let radius2 := 4
  let d := dist center1 center2
  3 < d ∧ d < 5 :=
by {
  let center1 := (0, 2)
  let center2 := (-2, -1)
  let radius1 := 1
  let radius2 := 4
  let d := dist center1 center2
  have h1 : dist center1 center2 = real.sqrt 13 := sorry,
  have h2 : 3 < real.sqrt 13 := sorry,
  have h3 : real.sqrt 13 < 5 := sorry,
  exact ⟨h2, h3⟩
}

end circles_intersect_l24_24320


namespace mod_inverse_13_1728_l24_24762

theorem mod_inverse_13_1728 :
  (13 * 133) % 1728 = 1 := by
  sorry

end mod_inverse_13_1728_l24_24762


namespace rectangle_perimeter_l24_24098

theorem rectangle_perimeter (t s : ℝ) (h : t ≥ s) : 2 * (t - s) + 2 * s = 2 * t := 
by 
  sorry

end rectangle_perimeter_l24_24098


namespace all_defective_is_impossible_l24_24818

def total_products : ℕ := 10
def defective_products : ℕ := 2
def selected_products : ℕ := 3

theorem all_defective_is_impossible :
  ∀ (products : Finset ℕ),
  products.card = selected_products →
  ∀ (product_ids : Finset ℕ),
  product_ids.card = defective_products →
  products ⊆ product_ids → False :=
by
  sorry

end all_defective_is_impossible_l24_24818


namespace min_games_required_l24_24593

-- Given condition: max_games ≤ 15
def max_games := 15

-- Theorem statement to prove: minimum number of games that must be played is 8
theorem min_games_required (n : ℕ) (h : n ≤ max_games) : n = 8 :=
sorry

end min_games_required_l24_24593


namespace work_done_by_force_l24_24377

def F (x : ℝ) : ℝ :=
  if x ≤ 2 then 10 else 3 * x + 4

def work_done (a b : ℝ) (F : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, F x

theorem work_done_by_force :
  work_done 0 4 F = 46 :=
by
  sorry

end work_done_by_force_l24_24377


namespace cos_solution_count_l24_24558

theorem cos_solution_count :
  ∃ n : ℕ, n = 2 ∧ 0 ≤ x ∧ x < 360 → cos x = 0.45 :=
by
  sorry

end cos_solution_count_l24_24558


namespace quadratic_polynomial_with_real_coeff_and_root_l24_24904

theorem quadratic_polynomial_with_real_coeff_and_root :
  ∃ (p : ℝ[X]), p.degree = 2 ∧ (p.coeff 1 = 6) ∧ (p.coeff 2 = 1) ∧ is_root p (-3 - 4i) :=
by
  have h : polynomial.map (algebra_map ℝ ℂ) (X^2 + 6*X + 25) = (X + C(Complex.mk (-3) (-4*i))) * (X + C(Complex.mk (-3) (4*i))),
  {
    -- This would be where the proof proceeds in a full manner
    sorry
  },
  use (X^2 + 6*X + 25),
  split,
  { exact polynomial.degree_eq_degree (by simp) (by simp),},
  split,
  { exact polynomial.coeff_X_pow },
  split,
  { exact polynomial.coeff_of_degree_eq two},
  { exact is_root_of_map_eq_root h}
  sorry

end quadratic_polynomial_with_real_coeff_and_root_l24_24904


namespace triangle_AC_length_l24_24757

theorem triangle_AC_length
  (AB BC DE EF DF : ℝ) 
  (h1 : AB = 6)
  (h2 : BC = 11)
  (h3 : DE = 6)
  (h4 : EF = 9)
  (h5 : DF = 10)
  (θ : ℝ)
  (h6 : θ = ( some angle ## deriving from these numbers, cos value ## )
  (h7 : x^2 = AB^2 + BC^2 - 2 * AB * BC * θ ))
  : AC ≈ 10.75 :=
sory -- proof goes here

end triangle_AC_length_l24_24757


namespace volume_OABC_l24_24754

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def volume_of_tetrahedron (O A B C : Point3D) :=
  1 / 6 * abs (O.x * (A.y * B.z + B.y * C.z + C.y * A.z - C.y * B.z - A.y * C.z - B.y * A.z))

theorem volume_OABC :
  let O := Point3D.mk 0 0 0
  let A := Point3D.mk a 0 0
  let B := Point3D.mk 0 b 0
  let C := Point3D.mk 0 0 c
  (a^2 + b^2 = 49) →
  (b^2 + c^2 = 64) →
  (c^2 + a^2 = 81) →
  volume_of_tetrahedron O A B C = 8 * √11 :=
by
  sorry

end volume_OABC_l24_24754


namespace distance_from_A_to_directrix_of_C_l24_24965

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24965


namespace a_share_calculation_l24_24774

noncomputable def investment_a : ℕ := 15000
noncomputable def investment_b : ℕ := 21000
noncomputable def investment_c : ℕ := 27000
noncomputable def total_investment : ℕ := investment_a + investment_b + investment_c -- 63000
noncomputable def b_share : ℕ := 1540
noncomputable def total_profit : ℕ := 4620  -- from the solution steps

theorem a_share_calculation :
  (investment_a * total_profit) / total_investment = 1100 := 
by
  sorry

end a_share_calculation_l24_24774


namespace Maria_workday_ends_at_4_30_l24_24658

-- Define the times
def time_of_day := ℕ -- measuring minutes since midnight

def eight_am : time_of_day := 8 * 60
def one_pm : time_of_day := 13 * 60
def one_thirty_pm : time_of_day := 13 * 60 + 30
def four_thirty_pm : time_of_day := 16 * 60 + 30

-- Define workday parameters
def work_hours := 8 * 60 -- total working time in minutes
def lunch_duration := 30 -- lunch break duration in minutes

-- Starting conditions
def start_time := eight_am
def lunch_start_time := one_pm

-- End time calculation function
def end_time (start_time lunch_start_time : time_of_day) (work_hours lunch_duration : ℕ) :=
  let working_time_before_lunch := lunch_start_time - start_time in
  let working_time_after_lunch := work_hours - working_time_before_lunch in
  lunch_start_time + lunch_duration + working_time_after_lunch

-- The statement to be proved
theorem Maria_workday_ends_at_4_30 :
  end_time start_time lunch_start_time work_hours lunch_duration = four_thirty_pm :=
by
  sorry

end Maria_workday_ends_at_4_30_l24_24658


namespace total_nails_needed_l24_24147

-- Define the conditions
def nails_already_have : ℕ := 247
def nails_found : ℕ := 144
def nails_to_buy : ℕ := 109

-- The statement to prove
theorem total_nails_needed : nails_already_have + nails_found + nails_to_buy = 500 := by
  -- The proof goes here
  sorry

end total_nails_needed_l24_24147


namespace friends_lunch_spending_l24_24364

-- Problem conditions and statement to prove
theorem friends_lunch_spending (x : ℝ) (h1 : x + (x + 15) + (x - 20) + 2 * x = 100) : 
  x = 21 :=
by sorry

end friends_lunch_spending_l24_24364


namespace count_ordered_pairs_harmonic_mean_l24_24712

theorem count_ordered_pairs_harmonic_mean :
  let H := λ (x y : ℕ), 2 * x * y / (x + y)
  in (H x y = 5^20) ∧ (x < y) → ∃! n, n = 20 :=
sorry

end count_ordered_pairs_harmonic_mean_l24_24712


namespace side_length_of_triangle_l24_24622

variables {x y : ℝ}
variables {A B C M : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space M]

noncomputable def distance (a b : Type) [metric_space a] [metric_space b] : ℝ := sorry
noncomputable def area (a b c : Type) [metric_space a] [metric_space b] [metric_space c] : ℝ := sorry

theorem side_length_of_triangle
  (h1 : distance A B ^ 2 + distance B C ^ 2 + distance C A ^ 2 + 3 * (distance A M ^ 2 + distance B M ^ 2 + distance C M ^ 2) ≤ 24 * x)
  (h2 : area A B C ≥ sqrt 3 * x) :
  y = 2 * sqrt x :=
sorry

end side_length_of_triangle_l24_24622


namespace probability_humanities_correct_l24_24459

-- Define the conditions:
def morning_classes := ["math", "chinese", "politics", "geography"]
def afternoon_classes := ["english", "history", "physical_education"]

def humanities_subjects := ["politics", "history", "geography"]

-- Define the total number of combinations:
def total_combinations : ℕ := list.length morning_classes * list.length afternoon_classes

-- Define the count of humanities in the morning and afternoon:
def humanities_morning := ["politics", "geography"]
def humanities_afternoon := ["history"]

-- Calculate the number of favorable outcomes:
def favorable_combinations : ℕ := 
  list.length humanities_morning * list.length afternoon_classes + (list.length morning_classes - list.length humanities_morning) * list.length humanities_afternoon

-- Define the probability calculation:
def probability_humanities : ℚ := favorable_combinations / total_combinations

-- The final proof statement:
theorem probability_humanities_correct :
  probability_humanities = 2 / 3 :=
by
  -- Add the necessary proof steps or sorry for now
  sorry

end probability_humanities_correct_l24_24459


namespace four_digit_number_count_l24_24549

theorem four_digit_number_count (digits : finset ℕ) :
  digits = {2, 3, 9, 9} →
  ∃ (count : ℕ), count = 12 :=
by
  intro h
  use 12
  sorry

end four_digit_number_count_l24_24549


namespace curve_passes_through_fixed_point_l24_24288

theorem curve_passes_through_fixed_point (m n : ℝ) :
  (2:ℝ)^2 + (-2:ℝ)^2 - 2 * m * (2:ℝ) - 2 * n * (-2:ℝ) + 4 * (m - n - 2) = 0 :=
by sorry

end curve_passes_through_fixed_point_l24_24288


namespace triangle_inequality_l24_24587

/-- Given a triangle ABC with side lengths a, b, c, and for an n-sided polygon A_1 A_2 ... A_n (n >= 3) with side lengths a_i,
prove the following inequality holds: 
a / (b + c - a) + b / (c + a - b) + c / (a + b - c) >= (b + c - a) / a + (c + a - b) / b + (a + b - c) / c >= 3. -/
theorem triangle_inequality (a b c : ℝ) (n : ℕ) (a_i : ℕ → ℝ) (s : ℝ) 
  (h : n ≥ 3)
  (h_sum : ∑ i in finset.range n, a_i i = s) 
  (h_triangle : ∀ i, a_i i > 0) :
  (a / (b + c - a) + b / (c + a - b) + c / (a + b - c) 
  ≥ (b + c - a) / a + (c + a - b) / b + (a + b - c) / c) ∧
  ((b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3) := 
sorry

end triangle_inequality_l24_24587


namespace Olivia_steps_l24_24669

def round_to_nearest_ten (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

theorem Olivia_steps :
  let x := 57 + 68
  let y := x - 15
  round_to_nearest_ten y = 110 := 
by
  sorry

end Olivia_steps_l24_24669


namespace max_cable_connections_l24_24824

theorem max_cable_connections :
  ∀ (A B : ℕ), A = 28 → B = 12 → (∀ a ∈ finset.range A, ∃ b1 b2 ∈ finset.range B, b1 ≠ b2) → 
    (∀ b ∈ finset.range B, ∃ a1 a2 ∈ finset.range A, a1 ≠ a2) → 
    (finset.range A.card * finset.range B.card = 336) :=
by
  intros A B hA hB hA_connections hB_connections
  have max_connections := A * B
  show max_connections = 336
  rw [hA, hB]
  have calc : 28 * 12 = 336 := by norm_num
  exact calc


end max_cable_connections_l24_24824


namespace distance_from_A_to_directrix_of_C_l24_24973

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24973


namespace perfect_apples_l24_24592

theorem perfect_apples (total_apples : ℕ) (fraction_small fraction_unripe : ℚ) 
  (h_total_apples : total_apples = 30) 
  (h_fraction_small : fraction_small = 1 / 6) 
  (h_fraction_unripe : fraction_unripe = 1 / 3) : 
  total_apples * (1 - fraction_small - fraction_unripe) = 15 :=
  by
  rw [h_total_apples, h_fraction_small, h_fraction_unripe]
  have h : 1 - (1/6 + 1/3) = 1/2 := by norm_num
  rw h
  norm_num

end perfect_apples_l24_24592


namespace determine_k_l24_24229

noncomputable def ari_seq (a1 d n : ℚ) : ℚ := a1 + (n - 1) * d

theorem determine_k :
  ∃ k : ℕ, (∃ a1 d : ℚ, 
    ari_seq a1 d 4 + ari_seq a1 d 7 + ari_seq a1 d 10 = 17 ∧ 
    ari_seq a1 d 4 + ari_seq a1 d 14 = 14 ∧ 
    ari_seq a1 d k = 13) ∧ k = 18 :=
begin
  sorry
end

end determine_k_l24_24229


namespace std_dev_of_ten_numbers_l24_24749

noncomputable def std_dev (s : Finset ℝ) : ℝ :=
  let n := s.card
  let mean := (s.sum id) / n
  real.sqrt ((s.sum (λ x, (x - mean)^2)) / n)

theorem std_dev_of_ten_numbers (x : Finset ℝ)
  (h_size : x.card = 10)
  (h_sum : x.sum id = 30)
  (h_sum_sq : x.sum (λ x, x^2) = 100) :
  std_dev x = 1 :=
by
  sorry

end std_dev_of_ten_numbers_l24_24749


namespace problem_statement_l24_24070

theorem problem_statement (m a b : ℝ) (h0 : 9^m = 10) (h1 : a = 10^m - 11) (h2 : b = 8^m - 9) : a > 0 ∧ 0 > b :=
by
  sorry

end problem_statement_l24_24070


namespace no_factor_l24_24024

noncomputable def polynomial := Polynomial (ℝ)

noncomputable def p : polynomial := Polynomial.X ^ 4 - 4 * Polynomial.X ^ 2 + 16

noncomputable def c1 : polynomial := Polynomial.X ^ 2 - 4
noncomputable def c2 : polynomial := Polynomial.X + 2
noncomputable def c3 : polynomial := Polynomial.X ^ 2 + 4 * Polynomial.X + 4
noncomputable def c4 : polynomial := Polynomial.X ^ 2 + 1

theorem no_factor : ¬(c1 ∣ p) ∧ ¬(c2 ∣ p) ∧ ¬(c3 ∣ p) ∧ ¬(c4 ∣ p) :=
by
  sorry

end no_factor_l24_24024


namespace find_f_neg2_l24_24644

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2^x + 3*x - 1 else -(2^(-x) + 3*(-x) - 1)

theorem find_f_neg2 : f (-2) = -9 :=
by sorry

end find_f_neg2_l24_24644


namespace problem_l24_24078

theorem problem (m a b : ℝ) (h₀ : 9 ^ m = 10) (h₁ : a = 10 ^ m - 11) (h₂ : b = 8 ^ m - 9) :
  a > 0 ∧ 0 > b := 
sorry

end problem_l24_24078


namespace example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l24_24248

-- Definition of a five-digit palindromic number
def is_palindromic (n : ℕ) : Prop := let s := n.to_string in s = s.reverse

-- Definition of a five-digit number
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

-- Part (a): Prove that 51715 is a five-digit palindromic number and is divisible by 5
theorem example_palindromic_divisible_by_5 :
  is_five_digit 51715 ∧ is_palindromic 51715 ∧ 51715 % 5 = 0 :=
by sorry

-- Part (b): Prove that there are exactly 100 five-digit palindromic numbers divisible by 5
theorem count_palindromic_divisible_by_5 :
  (finset.filter (λ n, is_five_digit n ∧ is_palindromic n ∧ n % 5 = 0) 
    (finset.range 100000)).card = 100 :=
by sorry

end example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l24_24248


namespace solve_systems_eq_l24_24581

theorem solve_systems_eq (a b : ℝ) (x y : ℝ)
  (h1 : x - y = 0) 
  (h2 : 2 * a * x + b * y = 4)
  (h3 : 2 * x + y = 3)
  (h4 : a * x + b * y = 3) : 
  a = 1 ∧ b = 2 := 
by{} 


end solve_systems_eq_l24_24581


namespace radius_of_shorter_tank_l24_24756

theorem radius_of_shorter_tank (h : ℝ) (r : ℝ) 
  (volume_eq : ∀ (π : ℝ), π * (10^2) * (2 * h) = π * (r^2) * h) : 
  r = 10 * Real.sqrt 2 := 
by 
  sorry

end radius_of_shorter_tank_l24_24756


namespace area_of_region_l24_24673

theorem area_of_region :
  let region := { (x, y) : ℝ × ℝ | abs(abs(x) - 2) + abs(y - 3) ≤ 3 } in
  let area := 34 in
  area_of_polygon region = area :=
sorry

end area_of_region_l24_24673


namespace unique_four_digit_numbers_divisible_by_4_once_multiple_four_digit_numbers_divisible_by_4_l24_24148

theorem unique_four_digit_numbers_divisible_by_4_once :
  ∃ n, n = 6 ∧
  (∀ x y z w : ℕ,
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
    x ∈ {1, 2, 3, 4} ∧ y ∈ {1, 2, 3, 4} ∧ z ∈ {1, 2, 3, 4} ∧ w ∈ {1, 2, 3, 4} ∧
    (z * 10 + w) % 4 = 0 → ∃! l, l = x * 1000 + y * 100 + z * 10 + w) := 
begin
  sorry
end

theorem multiple_four_digit_numbers_divisible_by_4 :
  ∃ n, n = 64 ∧
  (∀ x y z w : ℕ,
    x ∈ {1, 2, 3, 4} ∧ y ∈ {1, 2, 3, 4} ∧ z ∈ {1, 2, 3, 4} ∧ w ∈ {1, 2, 3, 4} ∧
    (z * 10 + w) % 4 = 0 → ∃! l, l = x * 1000 + y * 100 + z * 10 + w) := 
begin
  sorry
end

end unique_four_digit_numbers_divisible_by_4_once_multiple_four_digit_numbers_divisible_by_4_l24_24148


namespace distance_from_RS_l24_24621

-- Definitions for the problem
def square (t : ℝ) : Type := {P Q R S : (ℝ × ℝ) // 
  P = (0, t) ∧ Q = (t, t) ∧ R = (t, 0) ∧ S = (0, 0)}

def quarter_circle (center : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
  {pt : ℝ × ℝ | (pt.1 - center.1)^2 + (pt.2 - center.2)^2 = radius^2}

-- Points Y is the intersection of two arcs
def intersect_point (t : ℝ) : (ℝ × ℝ) :=
  let P := (0, t)
  let Q := (t, t)
  let arcP := quarter_circle P t
  let arcQ := quarter_circle Q t
  {pt : ℝ × ℝ // pt ∈ arcP ∧ pt ∈ arcQ ∧ 0 ≤ pt.1 ∧ pt.1 ≤ t ∧ 0 ≤ pt.2 ∧ pt.2 ≤ t}.val

-- The statement to prove
theorem distance_from_RS {t : ℝ} (ht : 0 < t) :
  let Y := intersect_point t
  Y.2 = t * (2 - Real.sqrt 3) / 2 :=
sorry

end distance_from_RS_l24_24621


namespace abs_lt_one_suff_but_not_necc_l24_24566

theorem abs_lt_one_suff_but_not_necc (x : ℝ) : (|x| < 1 → x^2 + x - 2 < 0) ∧ ¬(x^2 + x - 2 < 0 → |x| < 1) :=
by
  sorry

end abs_lt_one_suff_but_not_necc_l24_24566


namespace max_sequence_sum_l24_24909

variable {α : Type*} [LinearOrderedField α]

noncomputable def arithmeticSequence (a1 d : α) (n : ℕ) : α :=
  a1 + d * n

noncomputable def sequenceSum (a1 d : α) (n : ℕ) : α :=
  n * (a1 + (a1 + d * (n - 1))) / 2

theorem max_sequence_sum (a1 d : α) (n : ℕ) (hn : 5 ≤ n ∧ n ≤ 10)
    (h1 : d < 0) (h2 : sequenceSum a1 d 5 = sequenceSum a1 d 10) :
    n = 7 ∨ n = 8 :=
  sorry

end max_sequence_sum_l24_24909


namespace frustum_surface_area_l24_24323

theorem frustum_surface_area (r r' l : ℝ) (h_r : r = 1) (h_r' : r' = 4) (h_l : l = 5) :
  π * r^2 + π * r'^2 + π * (r + r') * l = 42 * π :=
by
  rw [h_r, h_r', h_l]
  norm_num
  sorry

end frustum_surface_area_l24_24323


namespace max_n_condition_permutation_l24_24231
-- Lean 4 statement

theorem max_n_condition_permutation
  (n : ℕ)
  {x : ℕ → ℕ} 
  (hx : ∀ i, 1 ≤ x i ∧ x i ≤ n)
  (hsum : ∑ i in Finset.range n, x i = n * (n + 1) / 2)
  (hprod : ∏ i in Finset.range n, x i = n.factorial) :
  n = 8 :=
by sorry

end max_n_condition_permutation_l24_24231


namespace exponent_fraction_law_l24_24890

theorem exponent_fraction_law :
  (2 ^ 2017 + 2 ^ 2013) / (2 ^ 2017 - 2 ^ 2013) = 17 / 15 :=
  sorry

end exponent_fraction_law_l24_24890


namespace distance_from_point_to_directrix_l24_24959

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24959


namespace tangent_line_and_point_l24_24120

theorem tangent_line_and_point (x0 y0 k: ℝ) (hx0 : x0 ≠ 0) 
  (hC : y0 = x0^3 - 3 * x0^2 + 2 * x0) (hl : y0 = k * x0) 
  (hk_tangent : k = 3 * x0^2 - 6 * x0 + 2) : 
  (k = -1/4) ∧ (x0 = 3/2) ∧ (y0 = -3/8) :=
by
  sorry

end tangent_line_and_point_l24_24120


namespace product_ab_l24_24893

theorem product_ab (a b : ℝ) (i : ℂ) (h1 : i = complex.I)
  (h2 : (1 + 7 * i) / (2 - i) = a + b * i) : a * b = -5 :=
by
  sorry

end product_ab_l24_24893


namespace num_factors_72_l24_24550

-- Define the prime factorization of 72
def prime_factors_of_72 := (2 ^ 3) * (3 ^ 2)

-- Define a helper function to count the number of factors
def num_factors (n : ℕ) : ℕ := 
  (nat.factors n).to_finset.card + 1

-- Theorem statement
theorem num_factors_72 : num_factors 72 = 12 := 
  sorry

end num_factors_72_l24_24550


namespace existence_of_constant_and_infinite_ns_l24_24680

/-- 
  Proposition: There exist a constant c > 0 and infinitely many positive integers n 
  such that there are infinitely many positive integers that cannot be expressed as the sum 
  of fewer than cn ln(n) n-th powers of pairwise relatively prime positive integers.
-/
theorem existence_of_constant_and_infinite_ns :
  ∃ c > 0, ∃ᶠ n in at_top, ∃ᶠ m in at_top,
    ∀ (x : ℕ), x ∉ {∑ i in finset.range (nat.ceil (c * n * real.log n)), a i | ∀ i, pairwise.coprime [a 0, .., a (nat.ceil (c * n * real.log n) - 1)]} :=
begin
  sorry
end

end existence_of_constant_and_infinite_ns_l24_24680


namespace intersection_of_sets_l24_24512

theorem intersection_of_sets:
  let A := {-2, -1, 0, 1}
  let B := {x : ℤ | x^3 + 1 ≤ 0 }
  A ∩ B = {-2, -1} :=
by
  sorry

end intersection_of_sets_l24_24512


namespace nancy_file_allocation_l24_24269

theorem nancy_file_allocation :
  ∀ (initial_pdfs initial_word_files initial_ppt_files total_files deleted_ppt_files deleted_pdfs remaining_pdfs remaining_files folders_with_word_files folders_with_mixed_files folders_with_only_pdfs total_folders : ℕ),
    initial_pdfs = 43 →
    initial_word_files = 30 →
    initial_ppt_files = 30 →
    total_files = 103 →
    deleted_ppt_files = 30 →
    deleted_pdfs = 33 →
    remaining_pdfs = initial_pdfs - deleted_pdfs →
    remaining_files = remaining_pdfs + initial_word_files →
    folders_with_word_files = initial_word_files / 7 →
    folders_with_mixed_files = if initial_word_files % 7 = 0 then 0 else 1 →
    folders_with_only_pdfs = (remaining_pdfs - (7 - (initial_word_files % 7))) / 7 →
    total_folders = folders_with_word_files + folders_with_mixed_files + if remaining_pdfs - (7 - (initial_word_files % 7)) ≤ 7 then 1 else folders_with_only_pdfs →
    total_folders = 6 :=
  sorry

end nancy_file_allocation_l24_24269


namespace arithmetic_sequence_sum_S20_l24_24503

noncomputable def S_n (a : ℕ → ℝ) : ℕ → ℝ := λ n, (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_sum_S20 (a : ℕ → ℝ) (h_seq : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_condition : a 5 + a 16 = 3) : 
  S_n a 20 = 30 :=
by sorry

end arithmetic_sequence_sum_S20_l24_24503


namespace max_ratio_l24_24482

-- Define conditions
def conditions (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ (x^3 + y^4 = x^2 * y)

-- Statement of the theorem
theorem max_ratio (A B : ℝ) 
  (hA : ∀ x y : ℝ, conditions x y → x ≤ A)
  (hB : ∀ x y : ℝ, conditions x y → y ≤ B) :
  A / B = 729 / 1024 :=
  sorry

end max_ratio_l24_24482


namespace smallest_possible_norm_l24_24224

noncomputable section

open Real

def min_norm_of_vector (v : ℝ × ℝ) (h : ‖⟨v.1 + 4, v.2 + 2⟩‖ = 10) : ℝ :=
  inf { ‖v‖ | ‖⟨v.1 + 4, v.2 + 2⟩‖ = 10 }

theorem smallest_possible_norm (v : ℝ × ℝ) (h : ‖⟨v.1 + 4, v.2 + 2⟩‖ = 10) :
  min_norm_of_vector v h = 10 - 2 * sqrt 5 :=
sorry

end smallest_possible_norm_l24_24224


namespace cameron_list_length_l24_24000

-- Definitions of multiples
def smallest_multiple_perfect_square := 900
def smallest_multiple_perfect_cube := 27000
def multiple_of_30 (n : ℕ) : Prop := n % 30 = 0

-- Problem statement
theorem cameron_list_length :
  ∀ n, 900 ≤ n ∧ n ≤ 27000 ∧ multiple_of_30 n ->
  (871 = (900 - 30 + 1)) :=
sorry

end cameron_list_length_l24_24000


namespace distance_from_A_to_directrix_of_C_l24_24967

def point (A : ℝ × ℝ) := A = (1, real.sqrt 5)
def parabola (y x p : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) := -p / 2
def distance_to_directrix (x p : ℝ) := x + p / 2

theorem distance_from_A_to_directrix_of_C :
  ∀ (p : ℝ), point (1, real.sqrt 5) → parabola (real.sqrt 5) 1 p → distance_to_directrix 1 p = 9 / 4 :=
by
  intros p h_point h_parabola
  sorry

end distance_from_A_to_directrix_of_C_l24_24967


namespace range_of_a_l24_24529

noncomputable def f (a b x : ℝ) : ℝ := log x - (1 / 2) * a * x^2 - b * x

theorem range_of_a (a b : ℝ) (h : 1 = 1 → ∀ x > 0, deriv (f a b) x = 0 → x = 1) :
  a > -1 :=
by
  have b_eq : b = 1 - a :=
    calc
      deriv (f a b) 1 = 0 : by sorry
      -- derive f(x) and set f'(1) = 0 to solve for b in terms of a
  show a > -1 from sorry
  -- show that a must be greater than -1 given the conditions

end range_of_a_l24_24529


namespace num_possible_values_n_l24_24387

theorem num_possible_values_n : 
  let S := {4, 7, 11, 13}
  let mean (set : Set ℝ) : ℝ := (set.sum id) / (set.card)
  let median (list : List ℝ) : ℝ := list.nth (list.length / 2).floor
  ∃ (n : ℝ), mean (insert n S) = median ((insert n S).toList) ∧ (insert n S).card = 5 :=
by
  sorry

end num_possible_values_n_l24_24387


namespace polar_to_parabola_l24_24886

theorem polar_to_parabola (r θ : ℝ) (h : r = 6 * tan θ * sec θ) : ∃ a b : ℝ, a = 6 ∧ (r * cos θ)^2 = a * (r * sin θ) :=
by
  sorry

end polar_to_parabola_l24_24886


namespace calculator_display_after_50_presses_l24_24301

noncomputable def special_key_transform (x : ℚ) : ℚ := 1 / (1 - x)

def initial_display : ℚ := 1 / 2

def repeated_transform (n : ℕ) : ℚ :=
  Function.iterate special_key_transform n initial_display

theorem calculator_display_after_50_presses : repeated_transform 50 = -1 :=
by
  sorry

end calculator_display_after_50_presses_l24_24301


namespace divisible_by_power_of_two_within_a_day_l24_24196

def card_initial_conition :
  ∃ (cards : Fin 100 → ℕ),
  (Finset.card {i | cards i % 2 = 1} = 43) := sorry

def card_transformation_condition (cards : Fin n → ℕ) :
  ∃ (new_card : ℕ),
  ∀ trio : Finset (Fin n), trio.card = 3 →
  new_card = ∑ i in trio, ∏ j in trio, cards j := sorry

theorem divisible_by_power_of_two_within_a_day :
  ∃ m ∈ {c | ∃ n : ℕ, c = some n ∧ n % 2^10000 = 0}, card_on_table_after_days m 1 :=
sorry

end divisible_by_power_of_two_within_a_day_l24_24196


namespace count_palindrome_five_digit_div_by_5_l24_24239

-- Define what it means for a number to be palindromic.
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Define what it means for a number to be a five-digit number.
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

-- Define what it means for a number to be divisible by 5.
def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Define the set of five-digit palindromic numbers divisible by 5.
def palindrome_five_digit_div_by_5_numbers (n : ℕ) : Prop :=
  is_five_digit n ∧ is_palindrome n ∧ is_divisible_by_5 n

-- Prove that the quantity of such numbers is 100.
theorem count_palindrome_five_digit_div_by_5 : 
  (finset.filter 
    (λ n, palindrome_five_digit_div_by_5_numbers n)
    (finset.range 100000)
  ).card = 100 :=
begin
  sorry
end

end count_palindrome_five_digit_div_by_5_l24_24239


namespace convert_to_polar_l24_24017

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  (Math.sqrt (x^2 + y^2), Real.arctan (y / x))

theorem convert_to_polar : polar_coordinates (2 * Real.sqrt 2) (2 * Real.sqrt 2) = (4, Real.pi / 4) :=
by
  sorry

end convert_to_polar_l24_24017


namespace abs_sum_inequality_solution_l24_24741

theorem abs_sum_inequality_solution (x : ℝ) : 
  (|x - 5| + |x + 1| < 8) ↔ (-2 < x ∧ x < 6) :=
sorry

end abs_sum_inequality_solution_l24_24741


namespace women_at_conference_l24_24596

theorem women_at_conference:
  ∃ (W : ℕ), 
    let men := 500 in
    let children := 500 in
    let total_people := men + W + children in
    let non_indian_men := 0.9 * men in
    let non_indian_women := 0.4 * W in
    let non_indian_children := 0.3 * children in
    let non_indian_people := non_indian_men + non_indian_women + non_indian_children in
    let non_indian_percentage := non_indian_people / total_people in
    non_indian_percentage = 0.5538461538461539 ∧ W = 300 :=
begin
  sorry
end

end women_at_conference_l24_24596


namespace sum_of_roots_eq_2023_l24_24063

def lhs (x : ℝ) : ℝ :=
  Real.sqrt (2 * x^2 - 2024 * x + 1023131) + 
  Real.sqrt (3 * x^2 - 2025 * x + 1023132) + 
  Real.sqrt (4 * x^2 - 2026 * x + 1023133)

def rhs (x : ℝ) : ℝ :=
  Real.sqrt (x^2 - x + 1) + 
  Real.sqrt (2 * x^2 - 2 * x + 2) + 
  Real.sqrt (3 * x^2 - 3 * x + 3)

theorem sum_of_roots_eq_2023 :
  {x : ℝ | lhs x = rhs x} = {1010, 1013} → 1010 + 1013 = 2023 :=
sorry

end sum_of_roots_eq_2023_l24_24063


namespace solution_set_for_inequality_l24_24329

theorem solution_set_for_inequality : {x : ℝ | (x - 2) / (x + 2) ≤ 0} = set.Icc (-2 : ℝ) (2 : ℝ) \ {x | x = -2} :=
by
  sorry

end solution_set_for_inequality_l24_24329


namespace percent_absent_is_20_l24_24686

noncomputable def total_students : ℕ := 105
noncomputable def boys : ℕ := 60
noncomputable def girls : ℕ := 45
noncomputable def boys_absent_fraction : ℚ := 1 / 10
noncomputable def girls_absent_fraction : ℚ := 1 / 3

theorem percent_absent_is_20 :
  let boys_absent := (boys_absent_fraction * boys : ℚ).natAbs
  let girls_absent := (girls_absent_fraction * girls : ℚ).natAbs
  let total_absent := boys_absent + girls_absent
  (total_absent : ℚ) / total_students * 100 = 20 :=
by
  sorry

end percent_absent_is_20_l24_24686


namespace calc_expr_equals_4_l24_24861

def calc_expr : ℝ :=
  (1/2)⁻¹ - (Real.sqrt 3) * Real.tan (Real.pi / 6) + (Real.pi - 2023)^0 + |(-2)|

theorem calc_expr_equals_4 : calc_expr = 4 := by
  -- Proof code goes here
  sorry

end calc_expr_equals_4_l24_24861


namespace complement_intersection_l24_24539

def A : set ℝ := { x | x < 3 }

def B : set ℕ := { x | x ≤ 5 }

theorem complement_intersection :
  ( { x : ℝ | x ≥ 3 } ∩ (B : set ℝ) ) = { 3, 4, 5 } :=
by
  -- Proof to be done here
  sorry

end complement_intersection_l24_24539


namespace no_real_numbers_x_l24_24913

theorem no_real_numbers_x (x : ℝ) : ¬ (-(x^2 + x + 1) ≥ 0) := sorry

end no_real_numbers_x_l24_24913


namespace max_value_m_l24_24021

/-- Proof that the inequality (a^2 + 4(b^2 + c^2))(b^2 + 4(a^2 + c^2))(c^2 + 4(a^2 + b^2)) 
    is greater than or equal to 729 for all a, b, c ∈ ℝ \ {0} with 
    |1/a| + |1/b| + |1/c| ≤ 3. -/
theorem max_value_m (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h_cond : |1 / a| + |1 / b| + |1 / c| ≤ 3) :
  (a^2 + 4 * (b^2 + c^2)) * (b^2 + 4 * (a^2 + c^2)) * (c^2 + 4 * (a^2 + b^2)) ≥ 729 :=
by {
  sorry
}

end max_value_m_l24_24021


namespace tiling_condition_l24_24448

def chessboard := fin 8 × fin 8

def is_black (square : chessboard) : Prop :=
  (square.1 + square.2) % 2 = 0

def is_white (square : chessboard) : Prop :=
  ¬ is_black square

theorem tiling_condition (B : chessboard) (s1 s2 : chessboard) :
  (s1 ≠ s2) → (s1 ∈ B) → (s2 ∈ B) →
  (∃ t : set (chessboard × chessboard), (∀ (d : chessboard × chessboard), d ∈ t → d.1 ≠ d.2) ∧
    ∀ s ∈ B, ∃ d ∈ t, s ∈ d) ↔ (is_black s1 ∧ is_white s2 ∨ is_white s1 ∧ is_black s2) :=
  sorry

end tiling_condition_l24_24448


namespace jogs_per_day_l24_24882

-- Definitions of conditions
def weekdays_per_week : ℕ := 5
def total_weeks : ℕ := 3
def total_miles : ℕ := 75

-- Define the number of weekdays in total weeks
def total_weekdays : ℕ := total_weeks * weekdays_per_week

-- Theorem to prove Damien jogs 5 miles per day on weekdays
theorem jogs_per_day : total_miles / total_weekdays = 5 := by
  sorry

end jogs_per_day_l24_24882


namespace number_of_perfect_apples_l24_24590

theorem number_of_perfect_apples (total_apples : ℕ) (too_small_ratio : ℚ) (not_ripe_ratio : ℚ)
  (h_total : total_apples = 30)
  (h_too_small_ratio : too_small_ratio = 1 / 6)
  (h_not_ripe_ratio : not_ripe_ratio = 1 / 3) :
  (total_apples - (too_small_ratio * total_apples).natAbs - (not_ripe_ratio * total_apples).natAbs) = 15 := by
  sorry

end number_of_perfect_apples_l24_24590


namespace find_all_triplets_l24_24047

theorem find_all_triplets (a b c : ℕ)
  (h₀_a : a > 0)
  (h₀_b : b > 0)
  (h₀_c : c > 0) :
  6^a = 1 + 2^b + 3^c ↔ 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 5 ∧ c = 1) :=
by
  sorry

end find_all_triplets_l24_24047


namespace value_of_f_neg1_plus_f_8_l24_24218

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (h : ∀ x, f(-x) = -f(x)) : true
axiom periodic_function (h : ∀ x, f(x + 4) = f(x)) : true
axiom f_one : f 1 = 1

theorem value_of_f_neg1_plus_f_8 : f(-1) + f(8) = -1 :=
by
  have h1 : f(-1) = -f(1),
  { apply odd_function, },
  have h2 : f(8) = f(8 - 4 * 2),
  { apply periodic_function, },
  have h3 : f(1) = 1,
  { apply f_one, },
  have h4 : f(0) = 0,
  {  sorry, }, -- Due to odd property assumed in lean context.
  rw [←h1, h3, h4],
  exact subtract_eq_of_eq_add'.1 sorry

end value_of_f_neg1_plus_f_8_l24_24218


namespace tangent_product_l24_24441

theorem tangent_product (n θ : ℝ) 
  (h1 : ∀ n θ, tan (n * θ) = (sin (n * θ)) / (cos (n * θ)))
  (h2 : tan 8 θ = (8 * tan θ - 56 * (tan θ) ^ 3 + 56 * (tan θ) ^ 5 - 8 * (tan θ) ^ 7) / 
                  (1 - 28 * (tan θ) ^ 2 + 70 * (tan θ) ^ 4 - 28 * (tan θ) ^ 6))
  (h3 : tan (8 * (π / 8)) = 0)
  (h4 : tan (8 * (3 * π / 8)) = 0)
  (h5 : tan (8 * (5 * π / 8)) = 0) :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = 2 * sqrt 2 :=
sorry

end tangent_product_l24_24441


namespace problem_l24_24077

theorem problem (m a b : ℝ) (h₀ : 9 ^ m = 10) (h₁ : a = 10 ^ m - 11) (h₂ : b = 8 ^ m - 9) :
  a > 0 ∧ 0 > b := 
sorry

end problem_l24_24077


namespace opposite_abs_reciprocal_neg_1_5_l24_24318

theorem opposite_abs_reciprocal_neg_1_5 : 
  let x := (-1.5 : ℝ)
  in -|(1 / x)| = (-2 / 3) :=
by
  sorry

end opposite_abs_reciprocal_neg_1_5_l24_24318


namespace inequality_correct_l24_24565

theorem inequality_correct (a b : ℝ) (ha : a < 0) (hb : b > 0) : (1/a) < (1/b) :=
sorry

end inequality_correct_l24_24565


namespace probability_at_least_one_boy_and_girl_l24_24842
-- Necessary imports

-- Defining the probability problem in Lean 4
theorem probability_at_least_one_boy_and_girl (n : ℕ) (hn : n = 4)
    (p : ℚ) (hp : p = 1 / 2) :
    let prob_all_same := (p ^ n) + (p ^ n) in
    (1 - prob_all_same) = 7 / 8 := by
  -- Include the proof steps here
  sorry

end probability_at_least_one_boy_and_girl_l24_24842


namespace A_inter_B_eq_l24_24652

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 > 1}

theorem A_inter_B_eq : A ∩ B = {-2, 2} := 
by
  sorry

end A_inter_B_eq_l24_24652


namespace problem_irational_count_l24_24401

noncomputable def isRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

noncomputable def isIrrational (x : ℝ) : Prop := ¬isRational x

def countIrrationals (l : List ℝ) : ℕ :=
  l.countp isIrrational

theorem problem_irational_count : countIrrationals [-3, 22 / 7, 3.14, -3 * Real.pi, 3.030030003] = 2 := by
  sorry

end problem_irational_count_l24_24401


namespace no_unhappy_days_l24_24729

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end no_unhappy_days_l24_24729


namespace total_cost_train_and_bus_l24_24395

noncomputable def trainFare := 3.75 + 2.35
noncomputable def busFare := 3.75
noncomputable def totalFare := trainFare + busFare

theorem total_cost_train_and_bus : totalFare = 9.85 :=
by
  -- We'll need a proof here if required.
  sorry

end total_cost_train_and_bus_l24_24395


namespace probability_at_least_one_boy_and_one_girl_l24_24833

theorem probability_at_least_one_boy_and_one_girl :
  let P := (1 - (1/16 + 1/16)) = 7 / 8,
  (∀ (N: ℕ), (N = 4) → 
    let prob_all_boys := (1 / N) ^ N,
    let prob_all_girls := (1 / N) ^ N,
    let prob_at_least_one_boy_and_one_girl := 1 - (prob_all_boys + prob_all_girls)
  in prob_at_least_one_boy_and_one_girl = P) :=
by
  sorry

end probability_at_least_one_boy_and_one_girl_l24_24833


namespace range_of_a_for_decreasing_f_l24_24643

theorem range_of_a_for_decreasing_f :
  {a : ℝ | ∀ x y : ℝ, x < y → f a x ≥ f a y } = set.Icc (1/6) (1/3 - 1 / 3) where
  f : ℝ → ℝ → ℝ
  | a, x =>
    if (x < 1) then (3 * a - 1) * x + 4 * a
    else a^x :=
sorry

end range_of_a_for_decreasing_f_l24_24643


namespace smallest_m_l24_24207

def f (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n+1), 1 / (Nat.lcm k n)^2

theorem smallest_m (m : ℕ) :
  (m > 0) ∧ (∃ (a b : ℚ), (m * f 10) * (b^2) = (a^2) * (π^2)) → m = 42 :=
by
  sorry

end smallest_m_l24_24207


namespace factor_expression_l24_24040

theorem factor_expression (a : ℝ) : 198 * a ^ 2 + 36 * a + 54 = 18 * (11 * a ^ 2 + 2 * a + 3) :=
by
  sorry

end factor_expression_l24_24040


namespace probability_monet_paintings_consecutive_l24_24548

theorem probability_monet_paintings_consecutive :
  let total_pieces := 12
  let monet_paintings := 4
  let favorable_arrangements := (9.factorial * 4.factorial)
  let total_arrangements := 12.factorial
  favorable_arrangements / total_arrangements = (1 : ℝ) / 55 :=
by
  sorry

end probability_monet_paintings_consecutive_l24_24548


namespace expression_eval_l24_24892

noncomputable def a : ℕ := 2001
noncomputable def b : ℕ := 2003

theorem expression_eval : 
  b^3 - a * b^2 - a^2 * b + a^3 = 8 :=
by sorry

end expression_eval_l24_24892


namespace distance_from_point_to_directrix_l24_24958

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24958


namespace find_a_l24_24516

theorem find_a (x a : ℝ) (h1 : 7^(2 * x) = 36) (h2 : 7^(-x) = 6^(-a / 2)) : a = 2 :=
sorry

end find_a_l24_24516


namespace smallest_abundant_number_up_to_20_l24_24889

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, d < n ∧ n % d = 0) (List.range n)

def is_abundant (n : ℕ) : Prop :=
  n < List.sum (proper_divisors n)

def smallest_abundant_up_to (k : ℕ) : ℕ :=
  (List.range (k + 1)).filter is_abundant |>.head!

theorem smallest_abundant_number_up_to_20 : smallest_abundant_up_to 20 = 12 :=
by
  sorry

end smallest_abundant_number_up_to_20_l24_24889


namespace pete_and_ray_spent_200_cents_l24_24273

-- Define the basic units
def cents_in_a_dollar := 100
def value_of_a_nickel := 5
def value_of_a_dime := 10

-- Define the initial amounts and spending
def pete_initial_amount := 250  -- cents
def ray_initial_amount := 250  -- cents
def pete_spent_nickels := 4 * value_of_a_nickel
def ray_remaining_dimes := 7 * value_of_a_dime

-- Calculate total amounts spent
def total_spent_pete := pete_spent_nickels
def total_spent_ray := ray_initial_amount - ray_remaining_dimes
def total_spent := total_spent_pete + total_spent_ray

-- The proof problem statement
theorem pete_and_ray_spent_200_cents : total_spent = 200 := by {
 sorry
}

end pete_and_ray_spent_200_cents_l24_24273


namespace arcsin_neg_one_eq_neg_pi_div_two_l24_24418

theorem arcsin_neg_one_eq_neg_pi_div_two : Real.arcsin (-1) = -Real.pi / 2 :=
by
  sorry

end arcsin_neg_one_eq_neg_pi_div_two_l24_24418


namespace exists_constant_c_l24_24681

def min_distance_to_line (S : set (ℝ × ℝ)) (l : set (ℝ × ℝ)) : ℝ :=
  Inf {(dist p q) | p ∈ S, q ∈ l}

theorem exists_constant_c
    (n : ℕ) (h₀ : 1 < n)
    (S : set (ℝ × ℝ)) (h₁ : S.size = n)
    (h₂ : ∀ p q ∈ S, p ≠ q → dist p q ≥ 1)
    :
    ∃ (c : ℝ), c = 0.1 ∧ ∀ l : set (ℝ × ℝ), (∃ p q ∈ S, seg_inter l p q) → min_distance_to_line S l ≥ c * (n : ℝ) ^ (-1 / 3) :=
by
  sorry

end exists_constant_c_l24_24681


namespace daily_shampoo_usage_l24_24291

theorem daily_shampoo_usage
  (S : ℝ)
  (h1 : ∀ t : ℝ, t = 14 → 14 * S + 14 * (S / 2) = 21) :
  S = 1 := by
  sorry

end daily_shampoo_usage_l24_24291


namespace exists_n_coprime_to_6_l24_24219

theorem exists_n_coprime_to_6 (k : ℕ) (hk : k.coprime 6) : ∃ n : ℕ, (2^n + 3^n + 6^n - 1) % k = 0 :=
by
  sorry

end exists_n_coprime_to_6_l24_24219


namespace distance_from_A_to_directrix_l24_24993

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l24_24993


namespace smallest_number_divisibility_condition_l24_24059

theorem smallest_number_divisibility_condition :
  ∃ n : ℕ, (∀ s : Finset ℕ, (s.card = n ∧ ∀ x ∈ s, x ≤ 1000) → ∃ a b ∈ s, a ≠ b ∧ (a < b ∧ ¬ b % a = 0)) ∧ n = 11 :=
begin
  sorry
end

end smallest_number_divisibility_condition_l24_24059


namespace probability_third_six_is_correct_l24_24872

open ProbabilityTheory

section DiceProblem

def fair_die (n : ℕ) : ℝ := if n = 6 then 1/6 else 1/6
def biased_die (n : ℕ) : ℝ := if n = 6 then 1/2 else 1/10

-- Charles selects a die at random
def choose_die : ℝ := 1/2

-- Charles rolls the dice three times
-- First two rolls are sixes
def roll_six_prob (die_prob : ℕ → ℝ) : ℝ := (die_prob 6) * (die_prob 6)

-- Given the first two rolls are sixes, calculate the probability of each die being used using Bayes' theorem
def fair_die_prob_given_six_six : ℝ :=
  (roll_six_prob fair_die) / ((roll_six_prob fair_die) + (roll_six_prob biased_die))

def biased_die_prob_given_six_six : ℝ :=
  (roll_six_prob biased_die) / ((roll_six_prob fair_die) + (roll_six_prob biased_die))

-- Calculate the probability of the third roll being a six
def prob_third_six : ℝ :=
  (fair_die_prob_given_six_six * fair_die 6) + (biased_die_prob_given_six_six * biased_die 6)

-- Prove that the computed probability is equal to 7/15
theorem probability_third_six_is_correct : prob_third_six = 7/15 := by
  sorry

end DiceProblem

end probability_third_six_is_correct_l24_24872


namespace max_value_inner_product_expression_l24_24496

variables (a b e : E) [inner_product_space ℝ E]
variables (ha : ∥a∥ = 2) (hb : ∥b∥ = 3) (he : ∥e∥ = 1) (hab : inner a b = -3)

theorem max_value_inner_product_expression : 
  ∃ (c : ℝ), c = ∥a • e + b • e∥ ∧ c = real.sqrt 7 :=
by sorry

end max_value_inner_product_expression_l24_24496


namespace distance_from_A_to_directrix_l24_24994

def point_A := (1 : ℝ, Real.sqrt 5)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

def distance_to_directrix (p x : ℝ) := x + p / 2

theorem distance_from_A_to_directrix :
  ∃ p : ℝ, parabola p 1 (Real.sqrt 5) ∧ distance_to_directrix p 1 = 9 / 4 :=
by
  sorry

end distance_from_A_to_directrix_l24_24994


namespace example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l24_24246

-- Definition of a five-digit palindromic number
def is_palindromic (n : ℕ) : Prop := let s := n.to_string in s = s.reverse

-- Definition of a five-digit number
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

-- Part (a): Prove that 51715 is a five-digit palindromic number and is divisible by 5
theorem example_palindromic_divisible_by_5 :
  is_five_digit 51715 ∧ is_palindromic 51715 ∧ 51715 % 5 = 0 :=
by sorry

-- Part (b): Prove that there are exactly 100 five-digit palindromic numbers divisible by 5
theorem count_palindromic_divisible_by_5 :
  (finset.filter (λ n, is_five_digit n ∧ is_palindromic n ∧ n % 5 = 0) 
    (finset.range 100000)).card = 100 :=
by sorry

end example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l24_24246


namespace problem_a_problem_b_problem_c_problem_d_l24_24858

noncomputable def A : ℝ := ∏ n in (finset.range 22).map (function.has_left_inverse.symm.1 $ ⟨(* 4), nat.mul_left_injective n.succ_pos⟩), real.cos (4 * n + 4)
theorem problem_a : A = (1 / 2) ^ 22 := by sorry

noncomputable def B : ℝ := ∏ n in (finset.range 7).map (function.has_left_inverse.symm.1 $ ⟨(* 12), nat.mul_left_injective n.succ_pos⟩), real.cos (12 * n + 12)
theorem problem_b : B = (1 / 2) ^ 7 := by sorry

noncomputable def C : ℝ := ∏ n in (finset.range 45).map (function.has_left_inverse.symm.1 $ ⟨(* 2), nat.mul_left_injective n.succ_pos⟩), real.cos (2 * n + 1)
theorem problem_c : C = (1 / 2) ^ 45 * real.sqrt 2 := by sorry

noncomputable def D : ℝ := ∏ n in (finset.range 9).map (function.has_left_inverse.symm.1 $ ⟨(* 10), nat.mul_left_injective n.succ_pos⟩), real.cos (10 * n + 5)
theorem problem_d : D = (1 / 2) ^ 9 * real.sqrt 2 := by sorry

end problem_a_problem_b_problem_c_problem_d_l24_24858


namespace probability_at_least_one_boy_and_girl_l24_24840
-- Necessary imports

-- Defining the probability problem in Lean 4
theorem probability_at_least_one_boy_and_girl (n : ℕ) (hn : n = 4)
    (p : ℚ) (hp : p = 1 / 2) :
    let prob_all_same := (p ^ n) + (p ^ n) in
    (1 - prob_all_same) = 7 / 8 := by
  -- Include the proof steps here
  sorry

end probability_at_least_one_boy_and_girl_l24_24840


namespace fourth_number_of_expression_l24_24795

theorem fourth_number_of_expression (x : ℝ) (h : 0.3 * 0.8 + 0.1 * x = 0.29) : x = 0.5 :=
by
  sorry

end fourth_number_of_expression_l24_24795


namespace candidate_percentage_l24_24793

theorem candidate_percentage (P : ℝ) (h : (P / 100) * 7800 + 2340 = 7800) : P = 70 :=
sorry

end candidate_percentage_l24_24793


namespace family_of_four_children_has_at_least_one_boy_and_one_girl_l24_24828

noncomputable section

def probability_at_least_one_boy_one_girl : ℚ :=
  1 - (1 / 16 + 1 / 16)

theorem family_of_four_children_has_at_least_one_boy_and_one_girl :
  probability_at_least_one_boy_one_girl = 7 / 8 := by
  sorry

end family_of_four_children_has_at_least_one_boy_and_one_girl_l24_24828


namespace surface_area_of_inscribed_sphere_l24_24744

theorem surface_area_of_inscribed_sphere (edge_length : ℝ) (r : ℝ) 
  (h_edge_length : edge_length = 4) 
  (h_radius : r = edge_length / 2) : 
  4 * real.pi * r^2 = 16 * real.pi := 
by 
  sorry

end surface_area_of_inscribed_sphere_l24_24744


namespace max_value_of_a_l24_24945

theorem max_value_of_a 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 - a * x) 
  (h2 : ∀ x ≥ 1, ∀ y ≥ 1, x ≤ y → f x ≤ f y) : 
  a ≤ 3 :=
sorry

end max_value_of_a_l24_24945


namespace tan_product_l24_24423

noncomputable def tan : ℝ → ℝ := sorry

theorem tan_product :
  (tan (Real.pi / 8)) * (tan (3 * Real.pi / 8)) * (tan (5 * Real.pi / 8)) = 2 * Real.sqrt 7 :=
by
  sorry

end tan_product_l24_24423


namespace days_required_for_C_l24_24372

noncomputable def work_rates {A B C : Type} (r_A r_B r_C : ℚ) :=
  (r_A = 1/9) ∧ 
  (r_A + r_B = 1/3) ∧ 
  (r_B + r_C = 1/5) ∧ 
  (r_A + r_C = 1/6)

theorem days_required_for_C {A B C : Type} (r_A r_B r_C : ℚ) (h : work_rates r_A r_B r_C) : 
  1/r_C = 18 := 
  by
    sorry

end days_required_for_C_l24_24372


namespace function_range_is_correct_l24_24721

noncomputable def function_range : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = Real.log (x^2 - 6 * x + 17) }

theorem function_range_is_correct : function_range = {x : ℝ | x ≤ Real.log 8} :=
by
  sorry

end function_range_is_correct_l24_24721


namespace max_value_of_f_l24_24069

noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x - 1/2

theorem max_value_of_f : ∃ x, 0 ≤ x ∧ x ≤ 2 ∧ (∀ y, (0 ≤ y ∧ y ≤ 2) → f y ≤ f x) ∧ f x = -3 :=
by
  sorry

end max_value_of_f_l24_24069


namespace number_of_sevens_l24_24781

theorem number_of_sevens (n : ℕ) : ∃ (k : ℕ), k < n ∧ ∃ (f : ℕ → ℕ), (∀ i, f i = 7) ∧ (7 * ((77 - 7) / 7) ^ 14 - 1) / (7 + (7 + 7)/7) = 7^(f k) :=
by sorry

end number_of_sevens_l24_24781


namespace sum_first_100_terms_l24_24216

-- Define the arithmetic progressions a_n and b_n
def a (n : ℕ) (d_a : ℕ) := 25 + (n - 1) * d_a
def b (n : ℕ) (d_b : ℕ) := 75 + (n - 1) * d_b

-- State the conditions
axiom a1 : ∀ (d_a : ℕ) (d_b : ℕ), a 1 d_a = 25
axiom b1 : ∀ (d_a : ℕ) (d_b : ℕ), b 1 d_b = 75
axiom sum_condition : ∀ (d_a : ℕ) (d_b : ℕ), a 100 d_a + b 100 d_b = 100

-- Prove the sum of the first hundred terms of the sequence a_n + b_n
theorem sum_first_100_terms (d_a d_b : ℕ) (h : d_a + d_b = 0) :
  (Finset.range 100).sum (λ n, a (n + 1) d_a + b (n + 1) d_b) = 10000 :=
by sorry

end sum_first_100_terms_l24_24216


namespace range_of_m_l24_24494

theorem range_of_m 
  (f : ℝ → ℝ := λ x, x^2)
  (g : ℝ → ℝ := λ x, (1/2)^x - m)
  (m : ℝ) 
  (h : ∀ x1 ∈ Icc 0 2, ∃ x2 ∈ Icc 1 2, f x1 ≥ g x2) : 
  m ≥ 1/4 := 
sorry

end range_of_m_l24_24494


namespace common_ratio_of_geometric_sequence_l24_24944

variable {a_n : ℕ → ℝ} -- assuming the sequence is indexed by natural numbers
variable {S_n : ℕ → ℝ} -- sum of first n terms function
variable {q : ℝ} -- common ratio

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q)

theorem common_ratio_of_geometric_sequence
  (h_geom : is_geometric_sequence a_n q)
  (h_sum : sum_of_first_n_terms a_n S_n)
  (h1 : 4 * S_n 3 = a_n 4 - 2)
  (h2 : 4 * S_n 2 = 5 * a_n 2 - 2)
  (hq_ne : q ≠ -1) :
  q = 5 :=
sorry

end common_ratio_of_geometric_sequence_l24_24944


namespace part_I_part_II_l24_24531

section
variable (x a : ℝ)

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (2 * x + 1)

theorem part_I :
  { x : ℝ | f x ≥ 4 * x + 3 } = set.Iic (-3 / 7) :=
sorry

theorem part_II :
  (∀ x : ℝ, 2 * f x ≥ 3 * a ^ 2 - a - 1) → -1 ≤ a ∧ a ≤ 4 / 3 :=
sorry
end

end part_I_part_II_l24_24531


namespace sphere_volume_diameter_l24_24578

theorem sphere_volume_diameter {D : ℝ} : 
  (D^3/2 + (1/21) * (D^3/2)) = (π * D^3 / 6) ↔ π = 22 / 7 := 
sorry

end sphere_volume_diameter_l24_24578


namespace find_x_l24_24651

variable (a b x : ℝ)
variable (hb : b ≠ 0)
variable (r : ℝ)

/- Given conditions -/
variable (hr1 : r = (3 * a)^(3 * b))
variable (hr2 : r = a^b * x^(2 * b))

/- The theorem to prove -/
theorem find_x : x = 3 * sqrt 3 * a :=
by
  sorry

end find_x_l24_24651


namespace area_of_quadrilateral_l24_24614

variables (A B C D : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [nonempty A] [nonempty B] [nonempty C] [nonempty D]
variables (AB : ℝ) (BC : ℝ) (CD : ℝ)
variable (circumscribed_inscribed_quadrilateral : Prop)

/-- The quadrilateral ABCD, where AB = 2, BC = 4, and CD = 5, is both circumscribed and inscribed.
    This implies the sum of the lengths of the opposite sides is equal. --/
def is_circumscribed_inscribed (AB : ℝ) (BC : ℝ) (CD : ℝ) (AD : ℝ) (circumscribed_inscribed_quadrilateral : Prop) : Prop :=
  circumscribed_inscribed_quadrilateral ∧ (AB = 2) ∧ (BC = 4) ∧ (CD = 5) ∧ (AB + CD = BC + AD)

theorem area_of_quadrilateral (AB BC CD : ℝ) (circumscribed_inscribed_quadrilateral : Prop) :
  is_circumscribed_inscribed AB BC CD 3 circumscribed_inscribed_quadrilateral → 
  ∃AD, AD = 3 → 
  ∃S, S = 2 * real.sqrt 30 :=
sorry

end area_of_quadrilateral_l24_24614


namespace complement_intersection_l24_24543

def U : Set ℝ := fun x => True
def A : Set ℝ := fun x => x < 0
def B : Set ℝ := fun x => x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2

theorem complement_intersection (hU : ∀ x : ℝ, U x) :
  ((compl A) ∩ B) = {0, 1, 2} :=
by {
  sorry
}

end complement_intersection_l24_24543


namespace sequence_sum_properties_l24_24449

noncomputable def a : ℕ → ℤ 
| 0       := x
| 1       := y
| (n + 2) := a (n + 1) - a n

variable (x y : ℤ)

theorem sequence_sum_properties (h1 : (list.range 1014).sum (fun n => a n) = 1643) 
                                (h2 : (list.range 1643).sum (fun n => a n) = 1014) :
  (list.range 2021).sum (fun n => a n) = 1643 := 
by
  sorry

end sequence_sum_properties_l24_24449


namespace fourth_intersection_point_l24_24610

noncomputable def curve (x y : ℝ) : Prop := x * y = 2

noncomputable def is_on_curve (p : ℝ × ℝ) : Prop := curve p.1 p.2

noncomputable def known_points := [(2, 1), (-4, -1/2 : ℝ), (1/2, 4)]

noncomputable def fourth_point : ℝ × ℝ := (-1, -2)

theorem fourth_intersection_point : 
  ∀ (a: ℝ × ℝ), 
  a ∉ known_points → is_on_curve a →
  (∃ (points : list (ℝ × ℝ)), 
  list.length points = 4 ∧ 
  (∀ p ∈ points, is_on_curve p) ∧ 
  known_points ⊆ points ∧ 
  a ∈ points) → 
  a = fourth_point :=
begin
  intros a hnotin hcurve hint,
  sorry
end

end fourth_intersection_point_l24_24610


namespace number_of_irrational_numbers_in_set_l24_24399

theorem number_of_irrational_numbers_in_set :
  let s := {-3 : ℝ, (22/7 : ℝ), 3.14, -3 * Real.pi, (Real.sqrt 10 - Real.sqrt 10 /3)} in
  (s.filter (λ x, ¬ ∃ a b : ℚ, (↑a : ℝ) = x * (↑b : ℝ))).card = 2 :=
by
  sorry

end number_of_irrational_numbers_in_set_l24_24399


namespace no_p_dependence_l24_24517

theorem no_p_dependence (m : ℕ) (p : ℕ) (hp : Prime p) (hm : m < p)
  (n : ℕ) (hn : 0 < n) (k : ℕ) 
  (h : m^2 + n^2 + p^2 - 2*m*n - 2*m*p - 2*n*p = k^2) : 
  ∀ q : ℕ, Prime q → m < q → (m^2 + n^2 + q^2 - 2*m*n - 2*m*q - 2*n*q = k^2) :=
by sorry

end no_p_dependence_l24_24517


namespace ratio_of_area_l24_24344

-- Define the sides of the triangles
def sides_GHI := (6, 8, 10)
def sides_JKL := (9, 12, 15)

-- Define the function to calculate the area of a triangle given its sides as a Pythagorean triple
def area_of_right_triangle (a b : Nat) : Nat :=
  (a * b) / 2

-- Calculate the areas
def area_GHI := area_of_right_triangle 6 8
def area_JKL := area_of_right_triangle 9 12

-- Calculate the ratio of the areas
def ratio_of_areas := area_GHI * 2 / area_JKL * 3

-- Statement to prove
theorem ratio_of_area (sides_GHI sides_JKL : (Nat × Nat × Nat)) :
  (ratio_of_areas = (4 : ℚ) / 9) :=
sorry

end ratio_of_area_l24_24344


namespace polar_to_cartesian_coordinates_l24_24580

theorem polar_to_cartesian_coordinates (ρ θ : ℝ) (hρ : ρ = 2) (hθ : θ = 5 * Real.pi / 6) :
  (ρ * Real.cos θ, ρ * Real.sin θ) = (-Real.sqrt 3, 1) :=
by
  sorry

end polar_to_cartesian_coordinates_l24_24580


namespace polynomial_inequality_l24_24717

theorem polynomial_inequality 
  (n : ℕ) 
  (a : Fin n → ℝ)
  (h_nonneg : ∀ i, 0 ≤ a i) 
  (P : ℝ → ℝ := λ x, x^n + ∑ i in Finset.range n, a i * x^(n - 1 - i))
  (h_roots : ∃ (roots : Fin n → ℝ), ∀ i, P (roots i) = 0) :
  P 2 ≥ 3^n :=
sorry

end polynomial_inequality_l24_24717


namespace find_fifth_term_l24_24745

def sequence (a : ℕ → ℕ) :=
  a 0 = 2 ∧ a 1 = 4 ∧ a 2 = 8 ∧ a 3 = 14 ∧ (∀ n, n > 0 → a (n + 1) - a n = 2 * (n + 1))

theorem find_fifth_term (a : ℕ → ℕ) (h : sequence a) : a 4 = 22 :=
  by
  intros
  sorry

end find_fifth_term_l24_24745


namespace buses_trips_product_l24_24690

theorem buses_trips_product :
  ∃ (n k : ℕ), n > 3 ∧ n * (n - 1) * (2 * k - 1) = 600 ∧ (n * k = 52 ∨ n * k = 40) := 
by
  sorry

end buses_trips_product_l24_24690


namespace equal_column_sums_l24_24678

theorem equal_column_sums (n : ℕ) (h_nonzero : n > 0) :
  ∃ (A : ℕ → ℕ → ℕ), (∀ i j, 1 ≤ A i j ∧ A i j ≤ n^2 ) ∧ 
    (∀ j, ∑ i in Finset.range n, A i j = n * (n * (n - 1) / 2 + j + 1)) :=
by
  sorry

end equal_column_sums_l24_24678


namespace ratio_of_area_l24_24346

-- Define the sides of the triangles
def sides_GHI := (6, 8, 10)
def sides_JKL := (9, 12, 15)

-- Define the function to calculate the area of a triangle given its sides as a Pythagorean triple
def area_of_right_triangle (a b : Nat) : Nat :=
  (a * b) / 2

-- Calculate the areas
def area_GHI := area_of_right_triangle 6 8
def area_JKL := area_of_right_triangle 9 12

-- Calculate the ratio of the areas
def ratio_of_areas := area_GHI * 2 / area_JKL * 3

-- Statement to prove
theorem ratio_of_area (sides_GHI sides_JKL : (Nat × Nat × Nat)) :
  (ratio_of_areas = (4 : ℚ) / 9) :=
sorry

end ratio_of_area_l24_24346


namespace books_remaining_in_library_l24_24336

def initial_books : ℕ := 250
def books_taken_out_Tuesday : ℕ := 120
def books_returned_Wednesday : ℕ := 35
def books_withdrawn_Thursday : ℕ := 15

theorem books_remaining_in_library :
  initial_books
  - books_taken_out_Tuesday
  + books_returned_Wednesday
  - books_withdrawn_Thursday = 150 :=
by
  sorry

end books_remaining_in_library_l24_24336


namespace equation_of_parallel_line_l24_24708

-- Define the given condition that line passes through (1, 0)
def passes_through (p : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  l p.1 = p.2

-- Define the line parallel condition
def parallel (l1 l2 : ℝ → ℝ) : Prop :=
  (∃m b1 b2, l1 = fun x => m * x + b1 ∧ l2 = fun x => m * x + b2)

-- The line equation
def line (m b : ℝ) : ℝ → ℝ :=
  fun x => m * x + b

-- The statement of the problem
theorem equation_of_parallel_line 
  (m : ℝ) (b1 : ℝ) (b2 : ℝ) :
  passes_through (1, 0) (line m b2) ∧ parallel (line m b1) (line m b2) →
  b2 = -0.5 :=
by
  sorry

end equation_of_parallel_line_l24_24708


namespace hexagon_coloring_l24_24029

open Finset Nat

def vertex := ℕ

structure Hexagon :=
(A B C D E F : vertex)

def adjacent (h : Hexagon) : vertex → vertex → Prop
| h.A, h.B | h.B, h.C | h.C, h.D | h.D, h.E | h.E, h.F | h.F, h.A := True
| h.B, h.A | h.C, h.B | h.D, h.C | h.E, h.D | h.F, h.E | h.A, h.F := True
| _, _ := False

def diagonal (h : Hexagon) : vertex → vertex → Prop
| h.A, h.C | h.A, h.D | h.B, h.D | h.B, h.E 
| h.C, h.E | h.C, h.F | h.D, h.F | h.D, h.A 
| h.E, h.A | h.E, h.B | h.F, h.B | h.F, h.C := True
| _, _ := False

def valid_coloring (h : Hexagon) (c : vertex → ℕ) :=
  ∀ v1 v2, (adjacent h v1 v2 ∨ diagonal h v1 v2) → c v1 ≠ c v2

def num_valid_colorings : ℕ :=
  8 * 7 * 6 * 6 * 6 * 5

theorem hexagon_coloring (h : Hexagon) :
  (∃ c : vertex → ℕ, valid_coloring h c) → num_valid_colorings = 75600 :=
begin
  sorry
end

end hexagon_coloring_l24_24029


namespace find_a_l24_24943

theorem find_a (x y a : ℕ) (h₁ : x = 2) (h₂ : y = 3) (h₃ : a * x + 3 * y = 13) : a = 2 :=
by 
  sorry

end find_a_l24_24943


namespace problem_statement_l24_24081

variable (m : ℝ) (a b : ℝ)

-- Given conditions
def condition1 : Prop := 9^m = 10
def condition2 : Prop := a = 10^m - 11
def condition3 : Prop := b = 8^m - 9

-- Problem statement to prove
theorem problem_statement (h1 : condition1 m) (h2 : condition2 m a) (h3 : condition3 m b) : a > 0 ∧ 0 > b := 
sorry

end problem_statement_l24_24081


namespace volumes_of_two_solids_l24_24353

-- Definitions filled with concrete knowledge from conditions
variables (a : ℝ)
variables (ABCDE M_1 M_2 M_3 M_4 M_5 N_1 N_2 N_3 : Type)

-- Representative definitions and assumptions based on given conditions
def square_based_pyramid : Prop := True
def is_midpoint (p1 p2 : Type) (p3 : Type) (ratio : ℝ) : Prop := True
def divides_edge_in_ratio (e1 e2 : Type) (p : Type) (ratio : ℝ) : Prop := True
def plane_through_points (ps : List Type) : Prop := True
def plane_parallel_to_edge (p : Type) (e : Type) : Prop := True
def volume_of_pyramid (p : Type) (v : ℝ) : Prop := True
def height_equal_base_edge (h b : ℝ) : Prop := True

-- Assumptions based on conditions
axiom condition1 : square_based_pyramid ABCDE
axiom condition2 : is_midpoint AE M_1 (2 : ℝ)
axiom condition3 : is_midpoint AB M_2 (1 : ℝ)
axiom condition4 : divides_edge_in_ratio BC M_3 (1 / 3)
axiom condition5 : plane_through_points [M_1, M_2, M_3]
axiom condition6 : divides_edge_in_ratio CE M_4 (2 / 3)
axiom condition7 : divides_edge_in_ratio DE M_5 (4 / 5)
axiom condition8 : is_midpoint DA N_1 (4 / 5)
axiom condition9 : is_midpoint DC N_2 (2 / 3)
axiom condition10 : divides_edge_in_ratio DB N_3 (4 / 5)
axiom condition11 : plane_parallel_to_edge (plane_through_points [M_1, M_2, M_3]) BE
axiom condition12 : height_equal_base_edge a a

-- Main proposition to prove
theorem volumes_of_two_solids : 
  volume_of_pyramid (M_5AM_2M_3CD) (a^3 / 45) ∧ volume_of_pyramid ((volume_of_pyramid.to_fit ABCDE) - M_5AM_2M_3CD) ((14 * a^3) / 45) :=
by 
  sorry

end volumes_of_two_solids_l24_24353


namespace values_of_cos_0_45_l24_24557

-- Define the interval and the condition for the cos function
def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 360
def cos_condition (x : ℝ) : Prop := Real.cos x = 0.45

-- Final theorem statement
theorem values_of_cos_0_45 : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (x : ℝ), in_interval x ∧ cos_condition x ↔ x = 1 ∨ x = 2 := 
sorry

end values_of_cos_0_45_l24_24557


namespace odd_prime_divisor_form_l24_24223

theorem odd_prime_divisor_form (x p : ℤ) (h : ℤ) 
  (h1 : p ∣ x^2 + 1)
  (h2 : Int.prime p)
  (h3 : p % 2 = 1) :
  ∃ h : ℤ, p = 4 * h + 1 :=
by
  sorry

end odd_prime_divisor_form_l24_24223


namespace find_a_l24_24916

def A (x : ℝ) : Set ℝ := {1, 2, x^2 - 5 * x + 9}
def B (x a : ℝ) : Set ℝ := {3, x^2 + a * x + a}

theorem find_a (a x : ℝ) (hxA : A x = {1, 2, 3}) (h2B : 2 ∈ B x a) :
  a = -2/3 ∨ a = -7/4 :=
by sorry

end find_a_l24_24916


namespace least_N_monochromatic_prism_l24_24470

theorem least_N_monochromatic_prism :
  ∃ N : ℕ, (∀ (coloring : (ℕ × ℕ × ℕ) → bool),
    (∃ (v1 v2 v3 v4 v5 v6 v7 v8 : ℕ × ℕ × ℕ),
      -- v1 to v8 are 8 vertices of a rectangular prism such that all have the same color
      ∀ (i j : ℕ) (h1 : 1 ≤ i ∧ i ≤ 3) (h2 : 1 ≤ j ∧ j ≤ 7) (h3 : 1 ≤ k ∧ k ≤ N), 
        (coloring v1 = coloring v2) ∧
        (coloring v1 = coloring v3) ∧
        (coloring v1 = coloring v4) ∧
        (coloring v1 = coloring v5) ∧
        (coloring v1 = coloring v6) ∧
        (coloring v1 = coloring v7) ∧
        (coloring v1 = coloring v8))) ∧
    N = 127 := sorry

end least_N_monochromatic_prism_l24_24470


namespace trigonometric_identity_l24_24875

theorem trigonometric_identity :
  (1 - Real.sin (Real.pi / 6)) * (1 - Real.sin (5 * Real.pi / 6)) = 1 / 4 :=
by
  have h1 : Real.sin (5 * Real.pi / 6) = Real.sin (Real.pi - Real.pi / 6) := by sorry
  have h2 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  sorry

end trigonometric_identity_l24_24875


namespace no_unhappy_days_l24_24737

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
  sorry

end no_unhappy_days_l24_24737


namespace find_x_value_l24_24603

theorem find_x_value (A B C x : ℝ) (hA : A = 40) (hB : B = 3 * x) (hC : C = 2 * x) (hSum : A + B + C = 180) : x = 28 :=
by
  sorry

end find_x_value_l24_24603


namespace interval_fraction_assignment_l24_24668

theorem interval_fraction_assignment (p q : ℕ) (hpq_coprime : Nat.coprime p q) 
  (hp_pos : 0 < p) (hq_pos : 0 < q) : 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ p + q - 2 →
  ∃ (i : ℕ), (1 ≤ i ∧ i < p) ∧ (i / p = k / (p + q)) ∨ 
  ∃ (j : ℕ), (1 ≤ j ∧ j < q) ∧ (j / q = k / (p + q)) :=
by
  sorry

end interval_fraction_assignment_l24_24668


namespace determinant_of_A_is_39_l24_24763

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![4, -5], ![3, 6]]

-- State the theorem to determine the determinant of A
theorem determinant_of_A_is_39 : A.det = 39 := by
  sorry

end determinant_of_A_is_39_l24_24763


namespace example_function_properties_l24_24879

noncomputable def f : ℝ → ℝ := λ x, -abs x

theorem example_function_properties :
  (∀ x y : ℝ, x < y → y < 0 → f x ≤ f y) ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ y : ℝ, y ≤ 0) ∧
  (∃ x : ℝ, f x = 0) :=
by
  -- Monotonically increasing on (-∞, 0)
  split
  · intros x y hxy hy
    change -abs x ≤ -abs y
    rw [neg_le_neg_iff]
    exact abs_le_abs hxy 

  -- Even function symmetry
  split
  · intro x
    rw [← abs_neg x]

  -- f(x) is less than or equal to 0
  split
  · intro y
    exact le_of_eq (neg_nonneg.mpr (abs_nonneg y))

  -- Maximum value occurs at x = 0
  · use 0
    simp

end example_function_properties_l24_24879


namespace probability_of_one_exactly_four_times_l24_24571

def roll_probability := (1 : ℝ) / 6
def non_one_probability := (5 : ℝ) / 6

lemma prob_roll_one_four_times :
  ∑ x in {1, 2, 3, 4, 5}, 
      roll_probability^4 * non_one_probability = 
    5 * (roll_probability^4 * non_one_probability) :=
by
  sorry

theorem probability_of_one_exactly_four_times :
  (5 : ℝ) * roll_probability^4 * non_one_probability = (25 : ℝ) / 7776 :=
by
  have key := prob_roll_one_four_times
  sorry

end probability_of_one_exactly_four_times_l24_24571


namespace magnitude_power_eq_l24_24895

-- Define the complex number 2 + i
def z : ℂ := (2 + 1 * Complex.I)

-- Define its magnitude
def magnitude_z : ℝ := Complex.abs z

-- Define the sixth power of its magnitude
def magnitude_z_6 : ℝ := magnitude_z ^ 6

-- State the theorem to prove
theorem magnitude_power_eq : magnitude_z_6 = 125 := by
  -- The proof is skipped; hence, we use sorry
  sorry

end magnitude_power_eq_l24_24895


namespace sunlovers_happy_days_l24_24732

open Nat

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end sunlovers_happy_days_l24_24732


namespace graph_is_centrally_symmetric_l24_24822

-- Define the function and its properties
def f (x : ℝ) : ℝ := (2 * x) / (x^2 - 1)

-- Property of the function given
def domain (x : ℝ) : Prop := x ∈ Set.univ \ (Set.Icc (-1 : ℝ) (1 : ℝ))

-- Proof goal: centrall symmetric graph
theorem graph_is_centrally_symmetric :
  ∀ x : ℝ, domain x → f (-x) = -f (x) :=
by
  intros x hx
  sorry

end graph_is_centrally_symmetric_l24_24822


namespace initial_capital_is_15000_l24_24366

noncomputable def initialCapital (profitIncrease: ℝ) (oldRate newRate: ℝ) (distributionRatio: ℝ) : ℝ :=
  (profitIncrease / ((newRate - oldRate) * distributionRatio))

theorem initial_capital_is_15000 :
  initialCapital 200 0.05 0.07 (2 / 3) = 15000 :=
by
  sorry

end initial_capital_is_15000_l24_24366


namespace books_sold_on_friday_l24_24628

theorem books_sold_on_friday
  (total_books : ℕ)
  (books_sold_mon : ℕ)
  (books_sold_tue : ℕ)
  (books_sold_wed : ℕ)
  (books_sold_thu : ℕ)
  (pct_unsold : ℚ)
  (initial_stock : total_books = 1400)
  (sold_mon : books_sold_mon = 62)
  (sold_tue : books_sold_tue = 62)
  (sold_wed : books_sold_wed = 60)
  (sold_thu : books_sold_thu = 48)
  (percentage_unsold : pct_unsold = 0.8057142857142857) :
  total_books - (books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + 40) = total_books * pct_unsold :=
by
  sorry

end books_sold_on_friday_l24_24628


namespace example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l24_24245

-- Definition of a five-digit palindromic number
def is_palindromic (n : ℕ) : Prop := let s := n.to_string in s = s.reverse

-- Definition of a five-digit number
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

-- Part (a): Prove that 51715 is a five-digit palindromic number and is divisible by 5
theorem example_palindromic_divisible_by_5 :
  is_five_digit 51715 ∧ is_palindromic 51715 ∧ 51715 % 5 = 0 :=
by sorry

-- Part (b): Prove that there are exactly 100 five-digit palindromic numbers divisible by 5
theorem count_palindromic_divisible_by_5 :
  (finset.filter (λ n, is_five_digit n ∧ is_palindromic n ∧ n % 5 = 0) 
    (finset.range 100000)).card = 100 :=
by sorry

end example_palindromic_divisible_by_5_count_palindromic_divisible_by_5_l24_24245


namespace price_difference_l24_24257

noncomputable def original_price (P : ℝ) : Prop :=
  0.80 * P + 4000 = 30000

theorem price_difference (P : ℝ) (h : original_price P) : P - 30000 = 2500 := by
  unfold original_price at h
  linarith

#check price_difference -- to ensure that the theorem is correct

end price_difference_l24_24257


namespace greatest_number_of_groups_l24_24798

theorem greatest_number_of_groups (s a t b n : ℕ) (hs : s = 10) (ha : a = 15) (ht : t = 12) (hb : b = 18) :
  (∀ n, n ≤ n ∧ n ∣ s ∧ n ∣ a ∧ n ∣ t ∧ n ∣ b ∧ n > 1 → 
  (s / n < (a / n) + (t / n) + (b / n))
  ∧ (∃ groups, groups = n)) → n = 3 :=
sorry

end greatest_number_of_groups_l24_24798


namespace symmetric_point_x_axis_l24_24705

theorem symmetric_point_x_axis (x y : ℝ) (h : (x, y) = (2, 3)) : (x, -y) = (2, -3) := by
  rw h
  simp
  sorry

end symmetric_point_x_axis_l24_24705


namespace find_income_l24_24310

def income_and_savings (x : ℕ) : ℕ := 10 * x
def expenditure (x : ℕ) : ℕ := 4 * x
def savings (x : ℕ) : ℕ := income_and_savings x - expenditure x

theorem find_income (savings_eq : 6 * 1900 = 11400) : income_and_savings 1900 = 19000 :=
by
  sorry

end find_income_l24_24310


namespace minimum_selling_price_l24_24845

def monthly_sales : ℕ := 50
def base_cost : ℕ := 1200
def shipping_cost : ℕ := 20
def store_fee : ℕ := 10000
def repair_fee : ℕ := 5000
def profit_margin : ℕ := 20

def total_monthly_expenses : ℕ := store_fee + repair_fee
def total_cost_per_machine : ℕ := base_cost + shipping_cost + total_monthly_expenses / monthly_sales
def min_selling_price : ℕ := total_cost_per_machine * (1 + profit_margin / 100)

theorem minimum_selling_price : min_selling_price = 1824 := 
by
  sorry 

end minimum_selling_price_l24_24845


namespace store_loss_l24_24799

noncomputable def cost_price_profiltable_set := 168 / 1.2
noncomputable def cost_price_loss_set := 168 / 0.8
def total_selling_price := 168 * 2
def total_cost_price := cost_price_profiltable_set + cost_price_loss_set

theorem store_loss :
  total_selling_price - total_cost_price = -14 := by
  sorry

end store_loss_l24_24799


namespace find_k_l24_24802

theorem find_k (k : ℝ) :
    (1 - 7) * (k - 3) = (3 - k) * (7 - 1) → k = 6.5 :=
by
sorry

end find_k_l24_24802


namespace combinatorial_identity_l24_24852

theorem combinatorial_identity :
  (Nat.factorial 15) / ((Nat.factorial 6) * (Nat.factorial 9)) = 5005 :=
sorry

end combinatorial_identity_l24_24852


namespace angle_determines_magnitude_l24_24640

-- Given two non-zero vectors a and b, and the angle θ between them
variables (a b : ℝ^3) (θ : ℝ) (t : ℝ)

-- Condition: For any real number t, the minimum value of |a + tb| is 1
axiom min_value_condition : ∀ t : ℝ, min (λ t, ∥ a + t • b ∥) = 1

-- Theorem: If θ is determined, then |a| is uniquely determined
theorem angle_determines_magnitude (hθ : θ) : ∥ a ∥ :=
sorry

end angle_determines_magnitude_l24_24640


namespace distance_from_point_to_directrix_l24_24950

def parabola_distance (x_A y_A p : ℝ) : ℝ :=
  if h : y_A^2 = 2 * p * x_A then x_A + p / 2 else 0

theorem distance_from_point_to_directrix :
  parabola_distance 1 (Real.sqrt 5) (5 / 2) = 9 / 4 := sorry

end distance_from_point_to_directrix_l24_24950


namespace segment_shorter_than_semi_sum_l24_24770

-- Definitions representing the conditions of the problem 
structure LineSegment (P Q : Type*) :=
(mk :: (start : P) (end : Q))

variable {P Q : Type*}

-- Representation of the midpoints of the line segments 
def is_midpoint (O : P) (A B : P) : Prop :=
dist O A = dist O B

-- The proof statement
theorem segment_shorter_than_semi_sum (AB A1B1 : LineSegment P) (O O1 : P)
  (hO : is_midpoint O AB.start AB.end)
  (hO1 : is_midpoint O1 A1B1.start A1B1.end)
  : dist O O1 < 1/2 * (dist AB.start A1B1.start + dist AB.end A1B1.end) :=
sorry

end segment_shorter_than_semi_sum_l24_24770


namespace find_largest_valid_number_l24_24052

-- Definitions and conditions
def starts_with_8 (n : ℕ) : Prop := n / 100 = 8

def has_property (n : ℕ) : Prop :=
  ∀ d ∈ [8, (n / 10) % 10, n % 10], d ≠ 0 → n % d = 0 ∧ n % (8 + (n / 10) % 10 + n % 10) = 0

noncomputable def largest_number_satisfying_conditions (n : ℕ) : ℕ :=
  if starts_with_8 n ∧ has_property n then n else 853

-- Theorem statement to be proved
theorem find_largest_valid_number : ∃ n : ℕ, starts_with_8 n ∧ has_property n ∧ n ≤ 899 ∧ n = 853 :=
by
  sorry

end find_largest_valid_number_l24_24052


namespace smallest_positive_integer_between_101_and_200_l24_24327

theorem smallest_positive_integer_between_101_and_200 :
  ∃ n : ℕ, n > 1 ∧ n % 6 = 1 ∧ n % 7 = 1 ∧ n % 8 = 1 ∧ 101 ≤ n ∧ n ≤ 200 :=
by
  sorry

end smallest_positive_integer_between_101_and_200_l24_24327


namespace determine_b_constant_remainder_l24_24023

-- Definitions of the polynomials
def numerator (b : ℝ) : ℝ[X] := 12 * X^3 - 9 * X^2 + b * X + 8
def denominator : ℝ[X] := 3 * X^2 - 4 * X + 2

-- Lean 4 statement for the proof problem
theorem determine_b_constant_remainder :
  ∃ b : ℝ, let quotient := euclidean_domain.div (numerator b) denominator,
               remainder := euclidean_domain.mod (numerator b) denominator
           in remainder.degree < denominator.degree ∧ remainder.coeff 1 = 0 ∧ remainder.coeff 0 = remainder :=
  ∃ b : ℝ, b = -4 / 3

end determine_b_constant_remainder_l24_24023


namespace complex_eq_solution_l24_24562

theorem complex_eq_solution (a b : ℂ) (h : 2 + a * complex.I = (b * complex.I - 1) * complex.I) : a + b * complex.I = -1 - 2 * complex.I :=
by {
  sorry
}

end complex_eq_solution_l24_24562


namespace math_problem_proof_l24_24188

-- Definitions of the conditions
def C1_parametric (α : ℝ) : ℝ × ℝ := (cos α, 1 + sin α)

def C2_polar (θ : ℝ) : ℝ := 2 * cos θ + 2 * sqrt 3 * sin θ

def line_l (θ : ℝ) : Prop := 
  θ = π / 3

-- Definitions of translated questions and correct answers
def C1_polar_equation (ρ θ : ℝ) : Prop :=
  ρ = 2 * sin θ

def C2_rectangular_equation (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - sqrt 3) ^ 2 = 4

def length_MN : ℝ := 4 - sqrt 3

-- The theorem we want to prove
theorem math_problem_proof :
  (∃ α θ ρ, C1_parametric α = (cos α, 1 + sin α) ∧ C2_polar θ = 2 * cos θ + 2 * sqrt 3 * sin θ ∧ line_l θ)
  →
  (∃ ρ θ, C1_polar_equation ρ θ)
  ∧ (∃ x y, C2_rectangular_equation x y)
  ∧ (length_MN = 4 - sqrt 3) :=
by
  sorry

end math_problem_proof_l24_24188


namespace largest_circle_area_rounded_nearest_whole_number_l24_24388

noncomputable def circle_area_from_rope (L : ℝ) : ℝ := 
  let r := L / (2 * Real.pi)
  ℝ := Real.pi * r^2

theorem largest_circle_area_rounded_nearest_whole_number :
  circle_area_from_rope (2 * 2 * (20 + 10)) = 1146 :=
by
  sorry

end largest_circle_area_rounded_nearest_whole_number_l24_24388


namespace sum_multiple_32_sum_multiples_l24_24630

theorem sum_multiple_32 : 
  let x := 64 + 96 + 128 + 160 + 288 + 352 + 3232 
  in x % 32 = 0 :=
by
  let x := 64 + 96 + 128 + 160 + 288 + 352 + 3232
  have h1 : 64 % 32 = 0 := by norm_num
  have h2 : 96 % 32 = 0 := by norm_num
  have h3 : 128 % 32 = 0 := by norm_num
  have h4 : 160 % 32 = 0 := by norm_num
  have h5 : 288 % 32 = 0 := by norm_num
  have h6 : 352 % 32 = 0 := by norm_num
  have h7 : 3232 % 32 = 0 := by norm_num
  exact (Nat.add_mod_eq_zero h1 h2 h3 h4 h5 h6 h7)

theorem sum_multiples : 
  let x := 64 + 96 + 128 + 160 + 288 + 352 + 3232 
  in x % 4 = 0 ∧ x % 8 = 0 ∧ x % 16 = 0 ∧ x % 32 = 0 :=
by
  let x := 64 + 96 + 128 + 160 + 288 + 352 + 3232
  have h32 : x % 32 = 0 := sum_multiple_32
  have h16 : x % 16 = 0 := Nat.dvd_trans (Nat.dvd_refl 32) (by norm_num : 32 % 16 = 0)
  have h8 : x % 8 = 0 := Nat.dvd_trans (Nat.dvd_refl 32) (by norm_num : 32 % 8 = 0)
  have h4 : x % 4 = 0 := Nat.dvd_trans (Nat.dvd_refl 32) (by norm_num : 32 % 4 = 0)
  exact ⟨h4, h8, h16, h32⟩

end sum_multiple_32_sum_multiples_l24_24630


namespace tan_product_l24_24442

theorem tan_product (t : ℝ) (h1 : 1 - 7 * t^2 + 7 * t^4 - t^6 = 0)
  (h2 : t = tan (Real.pi / 8) ∨ t = tan (3 * Real.pi / 8) ∨ t = tan (5 * Real.pi / 8)) :
  tan (Real.pi / 8) * tan (3 * Real.pi / 8) * tan (5 * Real.pi / 8) = 1 := 
sorry

end tan_product_l24_24442


namespace min_value_ineq_l24_24579

noncomputable def minimum_value (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (h : 4 * a + b = 1) : ℝ :=
  1 / a + 4 / b

theorem min_value_ineq (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (h : 4 * a + b = 1) :
  minimum_value a b ha hb h ≥ 16 :=
sorry

end min_value_ineq_l24_24579


namespace coal_consumption_rel_l24_24382

variables (Q a x y : ℝ)
variables (h₀ : 0 < x) (h₁ : x < a) (h₂ : Q ≠ 0) (h₃ : a ≠ 0) (h₄ : a - x ≠ 0)

theorem coal_consumption_rel :
  y = Q / (a - x) - Q / a :=
sorry

end coal_consumption_rel_l24_24382


namespace tan_pi_by_eight_product_l24_24432

theorem tan_pi_by_eight_product :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = -real.sqrt 2 := by 
sorry

end tan_pi_by_eight_product_l24_24432
