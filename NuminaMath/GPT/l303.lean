import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.Archimedean
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Conformalism
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.Logarithm
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Combinatorics.Category
import Mathlib.Combinatorics.Composition
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Enat
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Probability.Basic
import Mathlib.Tactic

namespace sequence_cycle_reappearance_l303_303254

theorem sequence_cycle_reappearance : ∀ (letters_digits: (List ℕ)) (n1 n2 : ℕ), 
  (∀ letters , letters = [1, 2, 3, 4, 5] → (∀ k, letters.length = k.succ → (letters.drop 1) ++ [letters.head] = [2, 3, 4, 5, 1])) → 
  (∀ digits , digits = [3, 5, 3, 1] → (∀ k, digits.length = k → (digits.drop 1) ++ [digits.head] = [5, 3, 1, 3])) →
  Nat.lcm n1 n2 = 20 :=
by
  intros letters_digits n1 n2 h_letters h_digits 
  have letters_cycle: letters_digits.head = [1, 2, 3, 4, 5] := by sorry
  have digits_cycle: letters_digits.last = [3, 5, 3, 1] := by sorry
  have h_lcm: Nat.lcm n1_lengths n2_lengths = 20 := by sorry
  exact h_lcm

end sequence_cycle_reappearance_l303_303254


namespace probability_even_and_greater_than_10_l303_303032

-- Define the set of numbers
def balls : Set ℕ := {1, 2, 3, 4, 5}

-- Define the condition of the product being even and greater than 10
def condition (x y : ℕ) : Prop := (x * y) % 2 = 0 ∧ x * y > 10

-- Calculate the probability
theorem probability_even_and_greater_than_10 :
  (∃ (favorable_outcomes : Finset (ℕ × ℕ)), 
     (∀ x y, (x ∈ balls ∧ y ∈ balls) → (x, y) ∈ favorable_outcomes ↔ condition x y) ∧
     favorable_outcomes.card / (balls.card * balls.card) = 1/5) :=
sorry

end probability_even_and_greater_than_10_l303_303032


namespace pet_store_cages_l303_303477

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) (h1 : initial_puppies = 56) (h2 : sold_puppies = 24) (h3 : puppies_per_cage = 4) : 
  (initial_puppies - sold_puppies) / puppies_per_cage = 8 := 
by 
  rw [h1, h2, h3]
  norm_num
  sorry

end pet_store_cages_l303_303477


namespace calculate_remainder_l303_303512

open Nat

theorem calculate_remainder :
  let ω := complex.exp (2 * complex.pi * complex.I / 4) in
  ω^4 = 1 ∧ ω ≠ 1 ∧ ω^2 = -1 ∧ ω^3 = -ω ∧
  let S := (1 + ω)^2011 + (1 + ω^2)^2011 + (1 + ω^3)^2011 + (2:ℂ)^2011 in
  S = 4 * ∑ k in range (503), nat.choose 2011 (4 * k) →
  (1 + ω^2)^2011 = 0 ∧ (1 + ω)^2011 + (1 + ω^3)^2011 = 0 →
  S = (2:ℂ)^2011 →
  (2^2011 : ℕ) % 8 = 0 ∧ (2^2011 : ℕ) % 125 = 48 →
  (2^2011 : ℕ) % 1000 = 48 →
  (4 * ∑ k in range (503), nat.choose 2011 (4 * k)) % 1000 = 48 →
  ((∑ k in range (503), nat.choose 2011 (4 * k)) % 1000 = 12) :=
by
  intros ω ω4 ω_ne ω2 ω3 S hS h1 h2 h3 h4 h5
  sorry

end calculate_remainder_l303_303512


namespace question1_question2_l303_303271

noncomputable def first_operation (x : ℕ) : ℕ := x / 2
noncomputable def second_operation (x : ℕ) : ℕ := 4 * x + 1

def operation_possible (start : ℕ) (target : ℕ) : Prop :=
  ∃ n : ℕ, ∃ f : Fin n → bool, 
    let iterate : ℕ → Fin n → ℕ
      | x, ⟨0, _⟩ => x
      | x, ⟨i+1, h⟩ => if f ⟨i, Nat.lt_of_lt_succ h⟩ then second_operation (iterate x ⟨i, Nat.lt_of_lt_succ h⟩)
                        else first_operation (iterate x ⟨i, Nat.lt_of_lt_succ h⟩) 
    in iterate start ⟨n, Nat.zero_lt_succ _⟩ = target

theorem question1 : ¬operation_possible 1 2000 := sorry

def under_2000 (x : ℕ) : Prop := x < 2000

def count_operations (start : ℕ) (cond : ℕ → Prop) : ℕ :=
  {x : ℕ // cond x}.elems.count (λ x, operation_possible start x.val)

theorem question2 : count_operations 1 under_2000 = 233 := sorry

end question1_question2_l303_303271


namespace problem_I_problem_II_l303_303662

noncomputable theory

def a (x : ℝ) : ℝ × ℝ := (5 * real.sqrt 3 * real.cos x, real.cos x)
def b (x : ℝ) : ℝ × ℝ := (real.sin x, 2 * real.cos x)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def magnitude_squared (v : ℝ × ℝ) : ℝ := v.1 ^ 2 + v.2 ^ 2

def f (x : ℝ) : ℝ :=
  let a := a x
  let b := b x
  dot_product a b + magnitude_squared b + 3 / 2

open interval real

theorem problem_I : ∀ (x : ℝ),
  x ∈ Icc (pi / 6) (pi / 2) -> f x ∈ Icc (5 / 2) 10 :=
sorry

theorem problem_II : ∀ (x : ℝ),
  x ∈ Icc (pi / 6) (pi / 2) ->
  f x = 8 ->
  f (x - pi / 12) = (3 * real.sqrt 3) / 2 + 7 :=
sorry

end problem_I_problem_II_l303_303662


namespace max_xyz_value_l303_303862

noncomputable def max_xyz (x y z : ℝ) := 
  if (x^2 + y^2 + z^2 + x + 2*y + 3*z = 13/4) then x + y + z else 0

theorem max_xyz_value (x y z : ℝ) (h : x^2 + y^2 + z^2 + x + 2*y + 3*z = 13/4): 
  0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z → (x + y + z ≤ sqrt 3) := 
sorry

end max_xyz_value_l303_303862


namespace find_coordinates_of_M_l303_303535

def ellipse_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : b < a) : Prop :=
  ∀ x y : ℝ,  x^2 / a^2 + y^2 / b^2 = 1 

def point_A (a : ℝ) : Prop := ∃ (x y : ℝ), x = a ∧ y = 0

def point_B (b : ℝ) : Prop := ∃ (x y : ℝ), x = 0 ∧ y = -b

def circle_center : ℝ × ℝ := (sqrt 3 / 2, -1 / 2)

def angle_BMN_60 (M : ℝ × ℝ) (N : ℝ × ℝ) (B : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧
  N = (6 * k / (1 + 3 * k^2), k * (6 * k / (1 + 3 * k^2)) - 1) ∧
  ∠B M N = 60

def coordinates_M: Prop :=
  ∃ (x y : ℝ), x = sqrt 3 / 3 ∧ y = 0

theorem find_coordinates_of_M (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : b < a)
  (ellipse_eq : ellipse_equation a b a_pos b_pos a_gt_b)
  (A : point_A a) (B : point_B b)
  (M : ℝ × ℝ) (N : ℝ × ℝ) (circle_center : ℝ × ℝ)
  (B_coord : B = (0, -1))
  (angle_60 : angle_BMN_60 M N (0, -1)) :
  coordinates_M :=
sorry

end find_coordinates_of_M_l303_303535


namespace scientific_notation_l303_303763

theorem scientific_notation : (10374 * 10^9 : Real) = 1.037 * 10^13 :=
by
  sorry

end scientific_notation_l303_303763


namespace correct_propositions_l303_303494

-- Proposition 1 Conditions: Inverse of the statement "If a^2 < b^2, then a < b" is "If a^2 > b^2, then a > b"
def prop1_inverse (a b : ℝ) : Prop := (a^2 > b^2) → (a > b)

-- Proposition 2 Conditions: Negation of "Congruent triangles have equal areas"
def prop2_negation : Prop := ¬(∀ {A B : Type} [metric_space A] [metric_space B] {P Q R : A} {U V W : B}, ((P, Q, R) ≃ᵤ (U, V, W)) → (area₀ P Q R = area₀ U V W))

-- Proposition 3 Conditions: Contrapositive of "If a > 1, then the solution set of ax^2 - 2ax + a + 3 > 0 is ℝ"
def prop3_contrapositive (a : ℝ) : Prop := (∀ {x : ℝ}, ax^2 - 2 * a * x + a + 3 > 0) ↔ (a ≤ 1)

-- Proposition 4 Conditions: "If √3 x (where x ≠ 0) is rational, then x is irrational"
def prop4 (x : ℝ) : Prop := (√3 * x).is_rational → (¬x.is_rational) ∧ x ≠ 0

-- Proof Problem: Prove that propositions (3) and (4) are correct.
theorem correct_propositions (a : ℝ) (x : ℝ) : prop3_contrapositive a ∧ prop4 x :=
by sorry

end correct_propositions_l303_303494


namespace f_eq_f_inv_iff_x_eq_3_5_l303_303544

def f (x : ℝ) : ℝ := 3 * x - 7
def f_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem f_eq_f_inv_iff_x_eq_3_5 (x : ℝ) : f(x) = f_inv(x) ↔ x = 3.5 := by
  sorry

end f_eq_f_inv_iff_x_eq_3_5_l303_303544


namespace third_derivative_l303_303462

noncomputable def y (x : ℝ) : ℝ := (5 * x - 1) * (Real.log x)^2

/-- Third derivative of the given function -/
theorem third_derivative (x : ℝ) (h : x > 0) : 
  (deriv^[3] (λ x, (5 * x - 1) * (Real.log x)^2)) x = (6 - 2 * (5 * x + 2) * Real.log x) / x^3 :=
sorry

end third_derivative_l303_303462


namespace product_of_divisors_of_18_l303_303344

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l303_303344


namespace calculate_expression_l303_303914

theorem calculate_expression :
  (Real.sqrt 3) ^ 0 + 2 ^ (-1 : ℤ) + Real.sqrt 2 * Real.cos (Real.pi / 4) - |(-1:ℝ) / 2| = 2 := 
by
  sorry

end calculate_expression_l303_303914


namespace coefficient_x2_in_expansion_l303_303023

theorem coefficient_x2_in_expansion (n k : ℕ) (h : n = 4 ∧ k = 2) :
  (nat.choose n k = 6) :=
by
  have h1 : nat.choose 4 2 = 6, by sorry,
  exact h1

end coefficient_x2_in_expansion_l303_303023


namespace product_of_divisors_of_18_l303_303373

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303373


namespace find_angle_B_l303_303659

noncomputable def triangle_sides_and_angles 
(a b c : ℝ) (A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

noncomputable def vectors_parallel 
(A B C a b c : ℝ) : Prop :=
  (Real.sin B - Real.sin A) / Real.sin C = (Real.sqrt 3 * a + c) / (a + b)

theorem find_angle_B (A B C a b c : ℝ)
  (h_triangle : triangle_sides_and_angles a b c A B C)
  (h_parallel : vectors_parallel A B C a b c) :
  B = 5 * Real.pi / 6 :=
sorry

end find_angle_B_l303_303659


namespace sin_product_eq_one_sixteenth_l303_303507

theorem sin_product_eq_one_sixteenth : 
  (Real.sin (12 * Real.pi / 180)) * 
  (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (54 * Real.pi / 180)) * 
  (Real.sin (78 * Real.pi / 180)) = 
  1 / 16 := 
sorry

end sin_product_eq_one_sixteenth_l303_303507


namespace find_c_l303_303620

variable (x y c : ℝ)

def condition1 : Prop := 2 * x + 5 * y = 3
def condition2 : Prop := c = Real.sqrt (4^(x + 1/2) * 32^y)

theorem find_c (h1 : condition1 x y) (h2 : condition2 x y c) : c = 4 := by
  sorry

end find_c_l303_303620


namespace tractor_planting_rate_l303_303563

theorem tractor_planting_rate
  (A : ℕ) (D : ℕ)
  (T1_days : ℕ) (T1 : ℕ)
  (T2_days : ℕ) (T2 : ℕ)
  (total_acres : A = 1700)
  (total_days : D = 5)
  (crew1_tractors : T1 = 2)
  (crew1_days : T1_days = 2)
  (crew2_tractors : T2 = 7)
  (crew2_days : T2_days = 3)
  : (A / (T1 * T1_days + T2 * T2_days)) = 68 := 
sorry

end tractor_planting_rate_l303_303563


namespace milesForHaircut_l303_303820

-- Definitions
def milesForGroceries : ℕ := 10
def milesForDoctor : ℕ := 5
def halfwayMiles : ℕ := 15
def totalMiles : ℕ := 2 * halfwayMiles

-- Theorem statement
theorem milesForHaircut : 
  milesForGroceries + milesForDoctor + ?milesForHaircut = totalMiles :=
sorry

end milesForHaircut_l303_303820


namespace arithmetic_sequence_sum_ratio_l303_303607

theorem arithmetic_sequence_sum_ratio 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (a : ℝ) 
  (d : ℝ) 
  (n : ℕ) 
  (a_n_def : ∀ n, a_n n = a + (n - 1) * d) 
  (S_n_def : ∀ n, S_n n = n * (2 * a + (n - 1) * d) / 2) 
  (h : 3 * (a + 4 * d) = 5 * (a + 2 * d)) : 
  S_n 5 / S_n 3 = 5 / 2 := 
by 
  sorry

end arithmetic_sequence_sum_ratio_l303_303607


namespace problem_complement_intersection_l303_303088

open Set

-- Define the universal set U
def U : Set ℕ := {0, 2, 4, 6, 8, 10}

-- Define set A
def A : Set ℕ := {0, 2, 4, 6}

-- Define set B based on A
def B : Set ℕ := {x | x ∈ A ∧ x < 4}

-- Define the complement of set A within U
def complement_A_U : Set ℕ := U \ A

-- Define the complement of set B within U
def complement_B_U : Set ℕ := U \ B

-- Prove the given equations
theorem problem_complement_intersection :
  (complement_A_U = {8, 10}) ∧ (A ∩ complement_B_U = {4, 6}) := 
by
  sorry

end problem_complement_intersection_l303_303088


namespace factor_quadratic_l303_303938

theorem factor_quadratic (x : ℝ) (m n : ℝ) 
  (hm : m^2 = 16) (hn : n^2 = 25) (hmn : 2 * m * n = 40) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := 
by sorry

end factor_quadratic_l303_303938


namespace a_minus_b_value_l303_303072

theorem a_minus_b_value (a b : ℤ) :
  (∀ x : ℝ, 9 * x^3 + y^2 + a * x - b * x^3 + x + 5 = y^2 + 5) → a - b = -10 :=
by
  sorry

end a_minus_b_value_l303_303072


namespace students_owning_both_pets_l303_303694

theorem students_owning_both_pets :
  ∃ n, (n = 25) ∧ (∀ u : Type, ∀ D C : set u,
  |D| = 35 → |C| = 40 → |D ∪ C| = 50 → ∀ x ∈ u, x ∈ D ∪ C) :=
begin
  sorry
end 

end students_owning_both_pets_l303_303694


namespace words_in_power_form_l303_303464

theorem words_in_power_form 
  (X Y : List Char) 
  (h : simplified_equation X Y) : 
  ∃ (Z : Char) (k l : ℕ), X = List.repeat Z k ∧ Y = List.repeat Z l :=
sorry

end words_in_power_form_l303_303464


namespace cost_of_each_math_book_l303_303839

-- Define the given conditions
def total_books : ℕ := 90
def math_books : ℕ := 53
def history_books : ℕ := total_books - math_books
def history_book_cost : ℕ := 5
def total_price : ℕ := 397

-- The required theorem
theorem cost_of_each_math_book (M : ℕ) (H : 53 * M + history_books * history_book_cost = total_price) : M = 4 :=
by
  sorry

end cost_of_each_math_book_l303_303839


namespace count_even_integers_with_four_different_digits_l303_303100

-- Define the conditions
def is_valid_integer (n : ℕ) : Prop :=
  n >= 4000 ∧ n < 8000 ∧ (n % 2 = 0) ∧ (nat.digits 10 n).nodup

-- Formalize the theorem statement
theorem count_even_integers_with_four_different_digits :
  {n : ℕ | is_valid_integer n}.to_finset.card = 336 :=
by
  sorry

end count_even_integers_with_four_different_digits_l303_303100


namespace simplify_fraction_l303_303228

/-
  Given the conditions that \(i^2 = -1\),
  prove that \(\displaystyle\frac{2-i}{1+4i} = -\frac{2}{17} - \frac{9}{17}i\).
-/
theorem simplify_fraction : 
  let i : ℂ := ⟨0, 1⟩ in
  i^2 = -1 → (2 - i) / (1 + 4 * i) = - (2 / 17) - (9 / 17) * i :=
by
  intro h
  sorry

end simplify_fraction_l303_303228


namespace distinct_positive_values_count_l303_303099

theorem distinct_positive_values_count : 
  ∃ (n : ℕ), n = 33 ∧ ∀ (x : ℕ), 
    (20 ≤ x ∧ x ≤ 99 ∧ 20 ≤ 2 * x ∧ 2 * x < 200 ∧ 3 * x ≥ 200) 
    ↔ (67 ≤ x ∧ x < 100) :=
  sorry

end distinct_positive_values_count_l303_303099


namespace minimum_rubles_to_reverse_order_l303_303710

theorem minimum_rubles_to_reverse_order (strip_length : ℕ) (initial_order reversed_order: Fin strip_length → ℕ) : strip_length = 100 ∧ 
(∀ i : Fin strip_length, initial_order (strip_length - 1 - i) = reversed_order i) ∧ 
(∀ i : Fin strip_length−1, ∃ n, 1 + n ≤ strip_length ∧ initial_order (i+suc n) = initial_order i) ∧
(∀ i : Fin strip_length−4, ∃ m, 3 + m < strip_length ∧ initial_order (i+3+m) = initial_order i) →
minimum_rubles required to rearrange initial_order into reversed_order = 50 :=
by
  sorry

end minimum_rubles_to_reverse_order_l303_303710


namespace coprime_quad_exists_l303_303614

theorem coprime_quad_exists
  (S : Finset ℕ)
  (hS : S.card = 133)
  (hCoprimePairs : ∃ N ≠ 799, ∃ (A B : Finset ℕ), A.card = 799 ∧ (∀ {a b : ℕ}, a ∈ A → b ∈ A → Nat.coprime a b))
  : ∃ a b c d ∈ S, Nat.coprime a b ∧ Nat.coprime b c ∧ Nat.coprime c d ∧ Nat.coprime d a := 
sorry

end coprime_quad_exists_l303_303614


namespace number_of_students_l303_303697

theorem number_of_students : 
    ∃ (n : ℕ), 
      (∃ (x : ℕ), 
        (∀ (k : ℕ), x = 4 * k ∧ 5 * x + 1 = n)
      ) ∧ 
      (∃ (y : ℕ), 
        (∀ (k : ℕ), y = 5 * k ∧ 4 * y + 1 = n)
      ) ∧
      n ≤ 30 ∧ 
      n = 21 :=
  sorry

end number_of_students_l303_303697


namespace product_of_divisors_of_18_l303_303368

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303368


namespace find_a_l303_303185

def f : ℝ → ℝ :=
λ x, if x ≤ 0 then -x else x^2

theorem find_a (a : ℝ) (h : f a = 4) : a = -4 ∨ a = 2 := 
sorry

end find_a_l303_303185


namespace total_number_of_digit_occurrences_is_559_l303_303956

-- Define PDF content
def PDF_Content := "extracted text from HMMTNovember2016GutsTest.pdf"

-- Define a function to count the occurrences of a digit in a string
def Digit_Count (d : Char) : Nat :=
  PDF_Content.count (λ c => c = d)

-- The sum of Digit_Count across all digits 0 to 9
def Total_Digit_Count : Nat :=
  (['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] : List Char).foldr (λ d acc => Digit_Count d + acc) 0

-- The main theorem to prove
theorem total_number_of_digit_occurrences_is_559 : Total_Digit_Count = 559 := sorry

end total_number_of_digit_occurrences_is_559_l303_303956


namespace product_of_divisors_of_18_l303_303315

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303315


namespace smallest_natural_number_l303_303256

theorem smallest_natural_number (a : ℕ) : 
  (∃ a, a % 3 = 0 ∧ (a - 1) % 4 = 0 ∧ (a - 2) % 5 = 0) → a = 57 :=
by
  sorry

end smallest_natural_number_l303_303256


namespace y_equals_px_div_5x_p_l303_303110

variable (p x y : ℝ)

theorem y_equals_px_div_5x_p (h : p = 5 * x * y / (x - y)) : y = p * x / (5 * x + p) :=
sorry

end y_equals_px_div_5x_p_l303_303110


namespace find_n_for_modulus_l303_303040

theorem find_n_for_modulus :
  ∃ (n : ℝ), 0 < n ∧ |5 + complex.I * n| = 5 * real.sqrt 13 ∧ n = 10 * real.sqrt 3 :=
begin
  -- sorry is used because the proof steps are not necessary
  sorry
end

end find_n_for_modulus_l303_303040


namespace popsicle_melting_faster_l303_303467

theorem popsicle_melting_faster (t : ℕ) :
  ∀ (n : ℕ), if n = 6 then (2 ^ (n - 1)) * t = 32 * t else true :=
by
  intro n
  cases n
  case zero => exact true.intro
  case succ n =>
    cases n
    case zero => exact true.intro
    case succ n =>
      cases n
      case zero => exact true.intro
      case succ n =>
        cases n
        case zero => exact true.intro
        case succ n =>
          cases n
          case zero => exact true.intro
          case succ n =>
            cases n
            case zero => exact true.intro
            case succ n =>
              case zero => exact true.intro
              sorry

end popsicle_melting_faster_l303_303467


namespace marcinkiewicz_theorem_l303_303771

theorem marcinkiewicz_theorem {P : Polynomial ℝ} (φ : ℝ → ℝ) (X : RandomVariable) :
  (∀ t, φ t = Real.exp (P.eval t)) → (∀ t, φ t = Expectation (Complex.exp (Complex.I * t * X))) → P.degree ≤ 2 := by
  sorry

end marcinkiewicz_theorem_l303_303771


namespace a_n_formula_b_n_formula_sum_c_n_T_n_formula_l303_303058

section Sequences

variable {a : ℕ → ℝ} {b : ℕ → ℝ} {c : ℕ → ℝ} {T : ℕ → ℝ} {S : ℕ → ℝ}

-- Conditions from the problem
axiom a_def : ∀ n, S n = 2 * a n - 2
axiom b1 : b 1 = 1
axiom b_line : ∀ n, b n - b(n+1) + 2 = 0

-- Prove that the sequences are as described in the solution

theorem a_n_formula : ∀ n, a n = 2^n := 
sorry

theorem b_n_formula : ∀ n, b n = 2 * n - 1 := 
sorry

-- Prove the sum of the sequence c_n
theorem sum_c_n : ∀ n, (∑ k in finset.range n, c k) = (2 * n - 3) * 2^(n + 1) + 6 :=
sorry

-- Define c_n as a_n * b_n
def c (n : ℕ) : ℕ → ℝ := λ n, (a n) * (b n)

-- Using the defined sequences prove that T_n is the sum of c_n
theorem T_n_formula : ∀ n, T n = (2 * n - 3) * 2^(n + 1) + 6 := 
sorry

end Sequences

end a_n_formula_b_n_formula_sum_c_n_T_n_formula_l303_303058


namespace num_permutations_with_prime_sums_l303_303920

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def valid_pi (pi : ℕ → ℕ) : Prop :=
  ∀ (m n : ℕ), m ∈ finset.range 11 → n ∈ finset.range 11 → is_prime (m + n) → is_prime (pi m + pi n)

noncomputable def num_valid_permutations : ℕ :=
  finset.univ.filter (λ pi, valid_pi (λ i, finset.univ {i ↦ i})).card

theorem num_permutations_with_prime_sums : num_valid_permutations = 4 := sorry

end num_permutations_with_prime_sums_l303_303920


namespace simplified_expression_l303_303457

def expression : ℝ := 
  ((0.2 * 0.4 - (0.3 / 0.5)) + ((0.6 * 0.8 + (0.1 / 0.2)) - (0.9 * (0.3 - 0.2 * 0.4))))^2 * (1 - (0.4^2 / (0.2 * 0.8)))

theorem simplified_expression : expression = 0 := 
by 
  -- Add the complete proof steps here in an interactive session or by using tactics.
  sorry

end simplified_expression_l303_303457


namespace minimum_triangle_area_l303_303593

theorem minimum_triangle_area : 
  ∀ (x1 y1 x2 y2 x3 y3 : ℝ), 
  (integer_point x1 y1) → (integer_point x2 y2) → (integer_point x3 y3) → 
  x1 ≠ x2 ∨ y1 ≠ y2 → x2 ≠ x3 ∨ y2 ≠ y3 → x1 ≠ x3 ∨ y1 ≠ y3 →
  0 < min_area ∧ min_area = (1 / 2) := by
  sorry

def integer_point (x y : ℝ) : Prop := ∃ k l : ℤ, x = k ∧ y = l

def min_area : ℝ := 
  let area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ := 
    (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
  let triangles := { area x1 y1 x2 y2 x3 y3 | integer_point x1 y1 ∧ integer_point x2 y2 ∧ integer_point x3 y3 ∧ 
                                        (x1 ≠ x2 ∨ y1 ≠ y2) ∧ (x2 ≠ x3 ∨ y2 ≠ y3) ∧ (x1 ≠ x3 ∨ y1 ≠ y3) }
  Inf triangles

end minimum_triangle_area_l303_303593


namespace exp_to_rect_form_l303_303538

theorem exp_to_rect_form : exp (11 * (Real.pi * Complex.I) / 2) = -Complex.I := by
  sorry

end exp_to_rect_form_l303_303538


namespace max_chord_length_a_2_max_chord_length_m_2_range_of_m_l303_303971

-- Define the circle equation
def circle_eq (x y a : ℝ) := x^2 + y^2 - 2*a*x - 6*a*y + 10*a^2 - 4*a = 0

-- Define the line equation
def line_eq (x y m : ℝ) := y = x + m

-- Proving the maximum length of the chord when a = 2
theorem max_chord_length_a_2 : ∃ ab_max, ab_max = 4 * sqrt 2 :=
by
  sorry

-- Proving the maximum length of the chord when m = 2
theorem max_chord_length_m_2 : ∃ ab_max, ab_max = 2 * sqrt 6 :=
by
  sorry

-- Proving the range of m when the line is tangent below the center of the circle
theorem range_of_m (m a : ℝ) (h : 0 < a ∧ a ≤ 4) : m = 2 * a - 2 * sqrt (2 * a) → 
  ∃ m_min m_max, m_min = -1 ∧ m_max = 8 - 4 * sqrt 2 :=
by
  sorry

end max_chord_length_a_2_max_chord_length_m_2_range_of_m_l303_303971


namespace qualified_products_count_l303_303867

-- Define the standard value and the acceptable range
def standard_value := 50
def lower_limit := 49.8
def upper_limit := 50.2

-- Define the errors for the six serial numbers
def errors := [-0.3, -0.5, 0, 0.1, -0.05, 0.12]

-- Function to calculate the actual diameter from the standard value and error
def calculate_diameter (error : Float) : Float :=
  standard_value + error

-- Function to check if a diameter is within the acceptable range
def is_within_tolerance (diameter : Float) : Bool :=
  lower_limit ≤ diameter ∧ diameter ≤ upper_limit

-- Counting the number of qualified products
def count_qualified_products : Nat :=
  List.length (List.filter is_within_tolerance (List.map calculate_diameter errors))

theorem qualified_products_count : count_qualified_products = 4 :=
  sorry

end qualified_products_count_l303_303867


namespace units_digit_sum_l303_303191

theorem units_digit_sum : 
  (3 : Nat) + 3^2 + 3^3 + ∑ i in Finset.range (2015 - 3), 3^i = 9 % 10 := by
  sorry

end units_digit_sum_l303_303191


namespace dodecagon_ratio_proof_l303_303221

noncomputable def dodecagon_division_ratio : Prop :=
  let S : ℝ := 1 -- Assume the area of small triangle ABM is 1 unit
  let small_triangle_area := S
  let large_triangle_area := 9 * S
  small_triangle_area / large_triangle_area = (1 : ℝ) / 9

theorem dodecagon_ratio_proof : dodecagon_division_ratio :=
by
  let S := (1 : ℝ)
  let small_triangle_area := S
  let large_triangle_area := 9 * S
  have h1 : small_triangle_area = S := rfl
  have h2 : large_triangle_area = 9 * S := rfl
  have h3 : small_triangle_area / large_triangle_area = (1 : ℝ) / 9 := by
    rw [h1, h2, div_eq_div_iff]
    simp
  exact h3

end dodecagon_ratio_proof_l303_303221


namespace least_n_possible_l303_303197

theorem least_n_possible : ∃ (n : ℕ), n > 2 ∧ n = 18 ∧ (∑ i in range (n + 1), i) / 2 = 97  := 
begin
  sorry
end

end least_n_possible_l303_303197


namespace sequence_of_arrows_from_425_to_427_l303_303126

theorem sequence_of_arrows_from_425_to_427 :
  ∀ (arrows : ℕ → ℕ), (∀ n, arrows (n + 4) = arrows n) →
  (arrows 425, arrows 426, arrows 427) = (arrows 1, arrows 2, arrows 3) :=
by
  intros arrows h_period
  have h1 : arrows 425 = arrows 1 := by 
    sorry
  have h2 : arrows 426 = arrows 2 := by 
    sorry
  have h3 : arrows 427 = arrows 3 := by 
    sorry
  sorry

end sequence_of_arrows_from_425_to_427_l303_303126


namespace prove_students_second_and_third_l303_303758

namespace MonicaClasses

def Monica := 
  let classes_per_day := 6
  let students_first_class := 20
  let students_fourth_class := students_first_class / 2
  let students_fifth_class := 28
  let students_sixth_class := 28
  let total_students := 136
  let known_students := students_first_class + students_fourth_class + students_fifth_class + students_sixth_class
  let students_second_and_third := total_students - known_students
  students_second_and_third = 50

theorem prove_students_second_and_third : Monica :=
  by
    sorry

end MonicaClasses

end prove_students_second_and_third_l303_303758


namespace range_of_g_l303_303002

def g (x : ℝ) := (Int.floor (2 * x) : ℝ) - 2 * x

theorem range_of_g :
  ∀ y : ℝ, y ∈ set.range g ↔ y ∈ set.Icc (-1 : ℝ) 0 :=
sorry

end range_of_g_l303_303002


namespace power_of_two_factorial_power_of_two_minus_one_divide_factorial_infinite_l303_303774

theorem power_of_two_factorial (n : ℕ) : ¬ (2^n ∣ n!) :=
by {
   sorry -- Proof to be provided
}

theorem power_of_two_minus_one_divide_factorial_infinite : 
  ∃ (f : ℕ → ℕ), (∀ p : ℕ, f p = 2^p) ∧ (∀ p : ℕ, 2^(f p - 1) ∣ (f p)!) :=
by {
   sorry -- Proof to be provided
}

end power_of_two_factorial_power_of_two_minus_one_divide_factorial_infinite_l303_303774


namespace larger_number_is_400_l303_303855

def problem_statement : Prop :=
  ∃ (a b hcf lcm num1 num2 : ℕ),
  hcf = 25 ∧
  a = 14 ∧
  b = 16 ∧
  lcm = hcf * a * b ∧
  num1 = hcf * a ∧
  num2 = hcf * b ∧
  num1 < num2 ∧
  num2 = 400

theorem larger_number_is_400 : problem_statement :=
  sorry

end larger_number_is_400_l303_303855


namespace midpoint_trajectory_extension_trajectory_l303_303869

-- Define the conditions explicitly

def is_midpoint (M A O : ℝ × ℝ) : Prop :=
  M = ((O.1 + A.1) / 2, (O.2 + A.2) / 2)

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 + P.2 ^ 2 - 8 * P.1 = 0

-- First problem: Trajectory equation of the midpoint M
theorem midpoint_trajectory (M O A : ℝ × ℝ) (hO : O = (0,0)) (hA : on_circle A) (hM : is_midpoint M A O) :
  M.1 ^ 2 + M.2 ^ 2 - 4 * M.1 = 0 :=
sorry

-- Define the condition for N
def extension_point (O A N : ℝ × ℝ) : Prop :=
  (A.1 - O.1) * 2 = N.1 - O.1 ∧ (A.2 - O.2) * 2 = N.2 - O.2

-- Second problem: Trajectory equation of the point N
theorem extension_trajectory (N O A : ℝ × ℝ) (hO : O = (0,0)) (hA : on_circle A) (hN : extension_point O A N) :
  N.1 ^ 2 + N.2 ^ 2 - 16 * N.1 = 0 :=
sorry

end midpoint_trajectory_extension_trajectory_l303_303869


namespace product_of_divisors_of_18_l303_303428

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303428


namespace overall_average_speed_l303_303490

-- Definition of given conditions
def time_skateboarding := 45 / 60 -- time in hours
def speed_skateboarding := 20 -- speed in mph
def time_jogging := 75 / 60 -- time in hours
def speed_jogging := 6 -- speed in mph

-- Solution steps and the resulting average speed proof
theorem overall_average_speed :
  let distance_skateboarding := speed_skateboarding * time_skateboarding in
  let distance_jogging := speed_jogging * time_jogging in
  let total_distance := distance_skateboarding + distance_jogging in
  let total_time := time_skateboarding + time_jogging in
  total_distance / total_time = 11.25 :=
by
  sorry

end overall_average_speed_l303_303490


namespace centroid_equidistant_l303_303717

variable {A B C : Point}
variable {a b c : ℝ}  -- lengths of sides opposite to vertices A, B, C
variable {S : Point}  -- centroid of triangle ABC

/-- A1, B1, C1 are points such that:
    AA1 = k * BC
    BB1 = k * CA
    CC1 = k * AB
    S is the centroid of triangle ABC
    k = 1/sqrt(3)
    Find the distance d such that S is equidistant from A1, B1, C1
-/
theorem centroid_equidistant (k : ℝ) (hA1 : dist A A1 = k * dist B C)
    (hB1 : dist B B1 = k * dist C A)
    (hC1 : dist C C1 = k * dist A B) (hcent : S = centroid A B C)
    (hk : k = 1 / real.sqrt 3) :
  (dist S A1 = dist S B1 ∧ dist S B1 = dist S C1) → 
  dist S A1 = 1 / 3 * real.sqrt (2 * (a^2 + b^2 + c^2)) := sorry

end centroid_equidistant_l303_303717


namespace percent_palindromes_contain_seven_l303_303882

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = String.reverse s

def is_four_digit_palindrome (n : ℕ) : Prop :=
  is_palindrome n ∧ 1000 ≤ n ∧ n < 5000

def contains_seven (n : ℕ) : Prop :=
  let s := n.toString in '7' ∈ s.toList

theorem percent_palindromes_contain_seven : 
  (∃ (count_all count_seven : ℕ), 
    (∀ n, is_four_digit_palindrome n → ∃ k, (count_all = k + 1) → count_all = 40)
    ∧ (∀ n, is_four_digit_palindrome n → contains_seven n → ∃ k, (count_seven = k + 1) → count_seven = 13)
    ∧ ((count_seven : ℚ) / (count_all : ℚ) * 100 = 32.5)) :=
by
  sorry

end percent_palindromes_contain_seven_l303_303882


namespace present_age_of_B_l303_303854

theorem present_age_of_B (A B : ℕ) (h1 : A + 20 = 2 * (B - 20)) (h2 : A = B + 10) : B = 70 :=
by
  sorry

end present_age_of_B_l303_303854


namespace exists_unique_plane_perpendicular_to_alpha_l303_303602

-- Define the entities and given conditions in Lean
open EuclideanGeometry

variables (m : Line) (α : Plane)

-- Conditions imposed by the problem
axiom m_intersects_alpha : m ∩ α ≠ ∅
axiom m_not_perpendicular_to_alpha : ¬ is_perpendicular m α

-- The Lean 4 statement for the problem
theorem exists_unique_plane_perpendicular_to_alpha : 
  ∃! β : Plane, is_perpendicular β α ∧ m ⊆ β :=
sorry

end exists_unique_plane_perpendicular_to_alpha_l303_303602


namespace modified_pyramid_volume_l303_303479

variable (s h : ℝ)
variable (h₀ : (1/3) * s^2 * h = 60)
variable (s' : ℝ := 3 * s)
variable (h' : ℝ := 0.75 * h)

def volume_pyramid (side_length : ℝ) (height : ℝ) : ℝ :=
  (1/3) * side_length^2 * height

theorem modified_pyramid_volume : volume_pyramid s' h' = 405 :=
by 
  have : volume_pyramid s h = 60 := h₀
  have h₁ : s^2 * h = 180 := by
    field_simp at this
    linarith
  have h₂ : (3 * s)^2 = 9 * s^2 := by ring
  have h₃ : 0.75 * h = 3/4 * h := by norm_num
  have : volume_pyramid s' h' = (1/3) * (9 * s^2) * (3/4 * h) := by
    simp [volume_pyramid, s', h']
    ring
  have : (9 * s^2) * (3/4 * h) = (27/4) * (s^2 * h) := by
    ring
  have : (1/3) * (27/4) * (s^2 * h) = (9/4) * (s^2 * h) := by
    ring
  exact calc
    volume_pyramid s' h' = (9/4) * 180 := by
      rw [this, h₁]
    ... = 405 := by norm_num

end modified_pyramid_volume_l303_303479


namespace product_of_divisors_of_18_l303_303380

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303380


namespace sum_of_distances_l303_303605

theorem sum_of_distances (A B C D M N O P : ℝ × ℝ) (side : ℝ):
  A = (0, 0) → B = (side, 0) → C = (side, side) → D = (0, side) →
  M = (side / 2, 0) → N = (side, side / 2) → O = (side / 2, side) → P = (0, side / 2) →
  side = 4 → 
  distance A M + distance A N + distance A O + distance A P + distance A C + distance A B = 10 + 4 * Real.sqrt 5 + 4 * Real.sqrt 2 :=
by
  intros
  sorry

end sum_of_distances_l303_303605


namespace perfect_square_of_sum_zero_l303_303212

-- Define the integers and the condition that their sum is zero
variables (a b c d : ℤ)
hypothesis h : a + b + c + d = 0

-- Prove that the given expression is a perfect square
theorem perfect_square_of_sum_zero (h : a + b + c + d = 0) : 
  2 * (a^4 + b^4 + c^4 + d^4) + 8 * a * b * c * d = (a^2 + b^2 + c^2 + d^2)^2 :=
by 
  sorry

end perfect_square_of_sum_zero_l303_303212


namespace right_angle_locus_l303_303982

noncomputable def P (x y : ℝ) : Prop :=
  let M : ℝ × ℝ := (-2, 0)
  let N : ℝ × ℝ := (2, 0)
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16

theorem right_angle_locus (x y : ℝ) : P x y → x^2 + y^2 = 4 ∧ x ≠ 2 ∧ x ≠ -2 :=
by
  sorry

end right_angle_locus_l303_303982


namespace calculate_remainder_l303_303509

open Nat

theorem calculate_remainder :
  let ω := complex.exp (2 * complex.pi * complex.I / 4) in
  ω^4 = 1 ∧ ω ≠ 1 ∧ ω^2 = -1 ∧ ω^3 = -ω ∧
  let S := (1 + ω)^2011 + (1 + ω^2)^2011 + (1 + ω^3)^2011 + (2:ℂ)^2011 in
  S = 4 * ∑ k in range (503), nat.choose 2011 (4 * k) →
  (1 + ω^2)^2011 = 0 ∧ (1 + ω)^2011 + (1 + ω^3)^2011 = 0 →
  S = (2:ℂ)^2011 →
  (2^2011 : ℕ) % 8 = 0 ∧ (2^2011 : ℕ) % 125 = 48 →
  (2^2011 : ℕ) % 1000 = 48 →
  (4 * ∑ k in range (503), nat.choose 2011 (4 * k)) % 1000 = 48 →
  ((∑ k in range (503), nat.choose 2011 (4 * k)) % 1000 = 12) :=
by
  intros ω ω4 ω_ne ω2 ω3 S hS h1 h2 h3 h4 h5
  sorry

end calculate_remainder_l303_303509


namespace number_of_Al_atoms_l303_303870

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_Br : ℝ := 79.90
def number_of_Br_atoms : ℕ := 3
def molecular_weight : ℝ := 267

theorem number_of_Al_atoms (x : ℝ) : 
  molecular_weight = (atomic_weight_Al * x) + (atomic_weight_Br * number_of_Br_atoms) → 
  x = 1 :=
by
  sorry

end number_of_Al_atoms_l303_303870


namespace unique_function_l303_303793

theorem unique_function {f : ℕ → ℕ} (h : ∀ n > 0, f(f(n)) + f(n) = 2n + 6) : ∀ n > 0, f(n) = n + 2 :=
by
  sorry

end unique_function_l303_303793


namespace product_of_all_positive_divisors_of_18_l303_303396

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l303_303396


namespace find_a_l303_303118

theorem find_a
  (x y a : ℝ)
  (h1 : x + y = 1)
  (h2 : 2 * x + y = 0)
  (h3 : a * x - 3 * y = 0) :
  a = -6 :=
sorry

end find_a_l303_303118


namespace emily_extra_distance_five_days_l303_303828

-- Define the distances
def distance_troy : ℕ := 75
def distance_emily : ℕ := 98

-- Emily's extra walking distance in one-way
def extra_one_way : ℕ := distance_emily - distance_troy

-- Emily's extra walking distance in a round trip
def extra_round_trip : ℕ := extra_one_way * 2

-- The extra distance Emily walks in five days
def extra_five_days : ℕ := extra_round_trip * 5

-- Theorem to be proven
theorem emily_extra_distance_five_days : extra_five_days = 230 := by
  -- Proof will go here
  sorry

end emily_extra_distance_five_days_l303_303828


namespace quadrilateral_area_l303_303276

noncomputable def area_pacq (AP PQ PC : ℕ) : ℝ :=
  let AQ := real.sqrt (PQ^2 - AP^2)
  let QC := real.sqrt (PC^2 - PQ^2)
  (1 / 2) * AP * AQ + (1 / 2) * PC * QC

theorem quadrilateral_area (hAP : AP = 9) (hPQ : PQ = 20) (hPC : PC = 21) :
  area_pacq AP PQ PC = (9 * real.sqrt 319 + 21 * real.sqrt 41) / 2 := by
  sorry

end quadrilateral_area_l303_303276


namespace solution_l303_303636

def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x^2) - 3 * x) + 1

theorem solution : f (Real.log 2 / Real.log 2) + f (-Real.log 2 / Real.log 2) = 2 := by
  sorry

end solution_l303_303636


namespace firetruck_area_reachable_l303_303879

-- Define the conditions as constants and assumptions
constant speed_on_road : ℝ := 60 -- miles per hour
constant speed_off_road : ℝ := 10 -- miles per hour
constant time_available : ℝ := 8 / 60 -- hours (converted from minutes)
constant pi : ℝ := Real.pi

-- Define distances the truck can travel
def distance_on_road := speed_on_road * time_available
def distance_off_road := speed_off_road * time_available

-- Calculate the areas
def area_on_road := distance_on_road * distance_on_road
def quarter_circle_area (r : ℝ) := (pi * r^2) / 4
def area_off_road := 4 * (quarter_circle_area distance_off_road)

-- Total area calculation
def total_area := area_on_road + area_off_road

-- Constants m and n for the final expression
constant m : ℕ := 4384
constant n : ℕ := 63

-- Final target: Prove that the firetruck can reach within 8 minutes the area which results in m + n = 4447
theorem firetruck_area_reachable :
  total_area = (m : ℝ) / (n : ℝ) ∧ Nat.gcd m n = 1 ∧ m + n = 4447 :=
by
  sorry

end firetruck_area_reachable_l303_303879


namespace union_is_correct_l303_303163

def A : set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : set ℝ := {x | 0 < x ∧ x < 2}
def union_AB : set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem union_is_correct : A ∪ B = union_AB := by sorry

end union_is_correct_l303_303163


namespace guy_age_l303_303469

-- Define the conditions
def age_hence (A : ℕ) : ℕ := A + 8
def age_ago (A : ℕ) : ℕ := A - 8
def expression (A : ℕ) : ℕ := (8 * age_hence A - 8 * age_ago A) / 2

-- State the theorem
theorem guy_age (A : ℕ) : expression A = A → A = 64 :=
begin
  sorry
end

end guy_age_l303_303469


namespace smallest_possible_value_eq_one_l303_303737

open Complex

noncomputable def ω : ℂ := exp (2 * π * I / 4)

lemma ω_prop : ω ^ 4 = 1 ∧ ω ≠ 1 :=
begin
  split,
  { simp [ω], field_simp, norm_num },
  { norm_num [ω] }
end

theorem smallest_possible_value_eq_one (a b c d : ℤ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  ∃ x : ℝ, x = |(↑a : ℂ) + (↑b : ℂ) * ω + (↑c : ℂ) * ω^2 + (↑d : ℂ) * ω^3| ∧ x = 1 :=
sorry

end smallest_possible_value_eq_one_l303_303737


namespace indeterminate_equation_solution_l303_303232

theorem indeterminate_equation_solution (a b : ℤ) :
  let x := a
  let y := b
  let z := a + b
  let u := a ^ 2 + a * b + b ^ 2
  let v := a * b
  let w := a * b * (a + b)
  let t := b * (a + b)
  in x ^ 4 + y ^ 4 + z ^ 4 = u ^ 2 + v ^ 2 + w ^ 2 + t ^ 2 := 
by
  sorry

end indeterminate_equation_solution_l303_303232


namespace digits_set_120_ways_l303_303273

theorem digits_set_120_ways : 
    (∃ (a : Finset ℕ), (∀ (x ∈ a), x ∈ (Finset.range 10)) ∧ a.card = 5 ∧ a.prod id = 120) ↔ 
    (∃ (a : Finset ℕ), a = {1, 2, 3, 4, 5}) :=
by sorry

end digits_set_120_ways_l303_303273


namespace product_of_all_positive_divisors_of_18_l303_303385

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l303_303385


namespace inequality_abc_l303_303201

theorem inequality_abc (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : a^2 + b^2 + c^2 = 1) : 
  (ab / c) + (bc / a) + (ca / b) ≥ real.sqrt 3 :=
begin
  sorry
end

end inequality_abc_l303_303201


namespace simplify_fraction_l303_303227

/-
  Given the conditions that \(i^2 = -1\),
  prove that \(\displaystyle\frac{2-i}{1+4i} = -\frac{2}{17} - \frac{9}{17}i\).
-/
theorem simplify_fraction : 
  let i : ℂ := ⟨0, 1⟩ in
  i^2 = -1 → (2 - i) / (1 + 4 * i) = - (2 / 17) - (9 / 17) * i :=
by
  intro h
  sorry

end simplify_fraction_l303_303227


namespace solid_is_cone_l303_303686

def is_isosceles_triangle (shape : Type) : Prop := sorry  -- Define what it means to be an isosceles triangle

def is_circle (shape : Type) : Prop := sorry  -- Define what it means to be a circle

variable (solid : Type)

-- Define the views of the solid
structure Views (solid : Type) :=
(front_view : solid)
(side_view : solid)
(top_view : solid)

variable (views : Views solid)

-- Assume conditions given in the problem
axiom front_view_is_isosceles_triangle : is_isosceles_triangle views.front_view
axiom side_view_is_isosceles_triangle : is_isosceles_triangle views.side_view
axiom top_view_is_circle : is_circle views.top_view

-- Define what it means to be a cone
def is_cone (solid : Type) : Prop := sorry

-- Theorem stating that under the given conditions, the solid is a cone
theorem solid_is_cone :
  front_view_is_isosceles_triangle →
  side_view_is_isosceles_triangle →
  top_view_is_circle →
  is_cone solid := by
  intros
  sorry

end solid_is_cone_l303_303686


namespace binomial_coefficient_sum_mod_l303_303532

theorem binomial_coefficient_sum_mod : 
  let S := ((1 + Complex.exp (Complex.I * Real.pi / 2))^2011) + 
           ((1 + Complex.exp (3 * Complex.I * Real.pi / 2))^2011) + 
           ((1 + -1)^2011) + 
           ((1 + 1)^2011)
  in 
  let desired_sum := (range 503).sum (λ j, Nat.choose 2011 (4 * j)) / 4
  in 
  (S % 1000 = 137) :
  nat.Mod 1000 S = 137 := 
begin
  sorry
end

end binomial_coefficient_sum_mod_l303_303532


namespace tractor_planting_rate_l303_303567

theorem tractor_planting_rate
  (acres : ℕ) (days : ℕ) (first_crew_tractors : ℕ) (first_crew_days : ℕ) 
  (second_crew_tractors : ℕ) (second_crew_days : ℕ) 
  (total_acres : ℕ) (total_days : ℕ) 
  (first_crew_days_calculated : ℕ) 
  (second_crew_days_calculated : ℕ) 
  (total_tractor_days : ℕ) 
  (acres_per_tractor_day : ℕ) :
  total_acres = acres → 
  total_days = days → 
  first_crew_tractors * first_crew_days = first_crew_days_calculated → 
  second_crew_tractors * second_crew_days = second_crew_days_calculated → 
  first_crew_days_calculated + second_crew_days_calculated = total_tractor_days → 
  total_acres / total_tractor_days = acres_per_tractor_day → 
  acres_per_tractor_day = 68 :=
by
  intros
  sorry

end tractor_planting_rate_l303_303567


namespace diffraction_slit_width_l303_303749

theorem diffraction_slit_width (λ : ℝ) (L : ℝ) (y : ℝ) : 
  475 * (10:ℝ)^(-9) = λ →
  2.013 = L →
  765 * (10:ℝ)^(-3) = y →
  let θ := y / L in 
  let d := λ / θ in 
  d = 1250 * (10:ℝ)^(-9) :=
by
  intros hλ hL hy
  dsimp
  rw [hλ, hL, hy]
  dsimp [θ, d]
  norm_num
  sorry

end diffraction_slit_width_l303_303749


namespace min_k_value_l303_303589

noncomputable def minimum_k_condition (x y z k : ℝ) : Prop :=
  k * (x^2 - x + 1) * (y^2 - y + 1) * (z^2 - z + 1) ≥ (x * y * z)^2 - (x * y * z) + 1

theorem min_k_value :
  ∀ x y z : ℝ, x ≤ 0 → y ≤ 0 → z ≤ 0 → minimum_k_condition x y z (16 / 9) :=
by
  sorry

end min_k_value_l303_303589


namespace sum_10_to_20_is_165_l303_303689

/-- The sum of the integers from 10 to 20 inclusive is 165. 
Given the number of even integers from 10 to 20 inclusive is 6, 
and their sum with the above result is 171.
-/
theorem sum_10_to_20_is_165 : (∑ n in Finset.range (21 - 10), (10 + n : ℕ)) = 165 ∧ (∑ n in Finset.range (21 - 10), (10 + n : ℕ)) + 6 = 171 := 
by
  sorry

end sum_10_to_20_is_165_l303_303689


namespace relationship_l303_303992

open Real

-- Given conditions
def f : ℝ → ℝ
def odd_f (f : ℝ → ℝ) := ∀ x, f x + f (-x) = 0
def condition (f : ℝ → ℝ) := ∀ x > 0, (f x / x) + (f' x) > 0

-- Definitions of a, b, c
def a : ℝ := f 1
def b : ℝ := ln 2 * f (ln 2)
def c : ℝ := log 2 (1 / 3) * f (log 2 (1 / 3))

-- The relationship to prove
theorem relationship (h1 : odd_f f) (h2: condition f) : c > a ∧ a > b := sorry

end relationship_l303_303992


namespace lattice_points_in_region_l303_303006

theorem lattice_points_in_region : ∃ n : ℕ, n = 1 ∧ ∀ p : ℤ × ℤ, 
  (p.snd = abs p.fst ∨ p.snd = -(p.fst ^ 3) + 6 * (p.fst)) → n = 1 :=
by
  sorry

end lattice_points_in_region_l303_303006


namespace product_of_divisors_of_18_is_5832_l303_303409

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l303_303409


namespace principal_amount_l303_303459

theorem principal_amount (
  P : ℝ)
  (R : ℝ := 25) -- interest rate per annum
  (T : ℝ := 5) -- time in years
  (CI : ℝ := P * ((1 + (R / 2) / 100) ^ (2 * T)) - P)
  (SI : ℝ := P * R * T / 100)
  (D : ℝ := CI - SI)
  (h : D = 4800) :
  P ≈ 5987.30 :=
by {
  sorry
}

end principal_amount_l303_303459


namespace painted_cubes_only_two_faces_l303_303874

theorem painted_cubes_only_two_faces :
  ∀ (n : ℕ), n = 3 →
  let total_small_cubes := n * n * n in
  total_small_cubes = 27 →
  let face_painted_cubes := 6 in
  let corner_cubes := 8 in
  let inner_cubes := 1 in
  let edge_cubes := (total_small_cubes - face_painted_cubes - corner_cubes - inner_cubes) in
  edge_cubes = 12 :=
by
  intros n h1 h2 face_painted_cubes corner_cubes inner_cubes
  have h : total_small_cubes = 27 := by rw h2
  have edge_cubes_def : edge_cubes = (total_small_cubes - face_painted_cubes - corner_cubes - inner_cubes) := rfl
  have edge_cubes_result : edge_cubes = 12 := by
    simp [face_painted_cubes, corner_cubes, inner_cubes, total_small_cubes] at edge_cubes_def
    rw [←h, edge_cubes_def]
    norm_num
  exact edge_cubes_result

end painted_cubes_only_two_faces_l303_303874


namespace sequence_property_l303_303078

noncomputable def f (x : ℝ) : ℝ := x / (3 * x + 1)

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = f (a n)

theorem sequence_property (a : ℕ → ℝ) (h_seq : sequence a) :
  a 2 = 1 / 4 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = 1 / (3 * (n + 1) - 2) :=
by
  sorry

end sequence_property_l303_303078


namespace digits_in_expression_l303_303929

noncomputable def numDigits (n : ℕ) : ℕ :=
(n.toString.length)

theorem digits_in_expression : numDigits (2^15 * 5^12 - 10^5) = 12 :=
by
  sorry

end digits_in_expression_l303_303929


namespace sales_not_books_magazines_stationery_l303_303781

variable (books_sales : ℕ := 45)
variable (magazines_sales : ℕ := 30)
variable (stationery_sales : ℕ := 10)
variable (total_sales : ℕ := 100)

theorem sales_not_books_magazines_stationery : 
  books_sales + magazines_sales + stationery_sales < total_sales → 
  total_sales - (books_sales + magazines_sales + stationery_sales) = 15 :=
by
  sorry

end sales_not_books_magazines_stationery_l303_303781


namespace simplify_fraction_l303_303226

/-
  Given the conditions that \(i^2 = -1\),
  prove that \(\displaystyle\frac{2-i}{1+4i} = -\frac{2}{17} - \frac{9}{17}i\).
-/
theorem simplify_fraction : 
  let i : ℂ := ⟨0, 1⟩ in
  i^2 = -1 → (2 - i) / (1 + 4 * i) = - (2 / 17) - (9 / 17) * i :=
by
  intro h
  sorry

end simplify_fraction_l303_303226


namespace doughnuts_left_l303_303553

theorem doughnuts_left (total_doughnuts : ℕ) (total_staff : ℕ) (staff3 : ℕ) (staff2 : ℕ) :
  total_doughnuts = 120 → total_staff = 35 →
  staff3 = 15 → staff2 = 10 →
  (let staff4 := total_staff - (staff3 + staff2) in
   let eaten3 := staff3 * 3 in
   let eaten2 := staff2 * 2 in
   let eaten4 := staff4 * 4 in
   let total_eaten := eaten3 + eaten2 + eaten4 in
   total_doughnuts - total_eaten = 15) :=
by
  intros total_doughnuts_eq total_staff_eq staff3_eq staff2_eq
  have staff4 := total_staff - (staff3 + staff2)
  have eaten3 := staff3 * 3
  have eaten2 := staff2 * 2
  have eaten4 := staff4 * 4
  have total_eaten := eaten3 + eaten2 + eaten4
  exact (total_doughnuts - total_eaten = 15)
  sorry

end doughnuts_left_l303_303553


namespace f_at_3_l303_303989

-- Define the function f and its conditions
variable (f : ℝ → ℝ)

-- The domain of the function f is ℝ, hence f : ℝ → ℝ
-- Also given:
axiom f_symm : ∀ x : ℝ, f (1 - x) = f (1 + x)
axiom f_add : f (-1) + f (3) = 12

-- Final proof statement
theorem f_at_3 : f 3 = 6 :=
by
  sorry

end f_at_3_l303_303989


namespace lateral_area_right_square_prism_volume_right_square_prism_l303_303784

noncomputable def base_edge_length := 3
noncomputable def height := 2

noncomputable def lateral_area (base_edge_length height : ℕ) : ℕ := 4 * height * base_edge_length
noncomputable def volume (base_edge_length height : ℕ) : ℕ := base_edge_length^2 * height

theorem lateral_area_right_square_prism :
  lateral_area base_edge_length height = 24 := by
  sorry

theorem volume_right_square_prism :
  volume base_edge_length height = 18 := by
  sorry

end lateral_area_right_square_prism_volume_right_square_prism_l303_303784


namespace distinct_floor_seq_count_l303_303016

def floor_seq (n : ℕ) : ℕ := Int.to_nat ⌊(n:ℚ) ^ 2 / 2000⌋

theorem distinct_floor_seq_count : 
  (Set.card (Set.image floor_seq { n : ℕ | 1 ≤ n ∧ n ≤ 2000 })) = 1501 :=
by
  sorry

end distinct_floor_seq_count_l303_303016


namespace soda_cost_l303_303492

-- Definitions based on conditions of the problem
variable (b s : ℤ)
variable (h1 : 4 * b + 3 * s = 540)
variable (h2 : 3 * b + 2 * s = 390)

-- The theorem to prove the cost of a soda
theorem soda_cost : s = 60 := by
  sorry

end soda_cost_l303_303492


namespace midpoints_of_AC_and_CD_l303_303140

open EuclideanGeometry

theorem midpoints_of_AC_and_CD 
  (ABCD : Quadrilateral)
  (area_ratio : ∀ (S_ABD S_BCD S_ABC : ℝ), S_ABC / S_ABD = 1 / 3 ∧ S_ABC / S_BCD = 1 / 4)
  (M N : Point)
  (H1 : M ∈ Seg AC)
  (H2 : N ∈ Seg CD)
  (H3 : dist A M / dist A C = dist C N / dist C D)
  (H4 : collinear [B, M, N])
  : M = midpoint A C ∧ N = midpoint C D  := 
sorry

end midpoints_of_AC_and_CD_l303_303140


namespace product_of_divisors_18_l303_303301

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l303_303301


namespace cookies_spent_in_april_l303_303034

theorem cookies_spent_in_april
  (cookies_per_day : ℕ)
  (cost_per_cookie : ℕ)
  (days_in_april : ℕ)
  (total_days_of_april : days_in_april = 30)
  (cookies_per_day_each_day : cookies_per_day = 3)
  (cost_per_cookie_each : cost_per_cookie = 18) :
  cookies_per_day * days_in_april * cost_per_cookie = 1620 :=
by
  rw [total_days_of_april, cookies_per_day_each_day, cost_per_cookie_each]
  show 3 * 30 * 18 = 1620
  calc
    3 * 30 * 18 = 90 * 18  : by rw mul_assoc
             ... = 1620    : by rfl

end cookies_spent_in_april_l303_303034


namespace kittens_born_next_spring_l303_303751

theorem kittens_born_next_spring (breeding_rabbits : ℕ) (total_rabbits : ℕ)
  (kittens_multiplier : ℕ) (kittens_first_adopted : ℕ) (returned_kittens : ℕ)
  (next_spring_kittens_adopted : ℕ) :
  breeding_rabbits = 10 →
  total_rabbits = 121 →
  kittens_multiplier = 10 →
  kittens_first_adopted = 50 →
  returned_kittens = 5 →
  next_spring_kittens_adopted = 4 →
  let kittens_first_spring := breeding_rabbits * kittens_multiplier in
  let remaining_kittens_first_spring := (kittens_first_spring - kittens_first_adopted) + returned_kittens in
  let total_non_breeding_rabbits := total_rabbits - breeding_rabbits in
  let kittens_next_spring := total_non_breeding_rabbits - remaining_kittens_first_spring in
  kittens_next_spring = 56 :=
by
  intros
  sorry

end kittens_born_next_spring_l303_303751


namespace roots_cubic_sum_l303_303998

theorem roots_cubic_sum :
  (∃ x1 x2 x3 x4 : ℂ, (x1^4 + 5*x1^3 + 6*x1^2 + 5*x1 + 1 = 0) ∧
                       (x2^4 + 5*x2^3 + 6*x2^2 + 5*x2 + 1 = 0) ∧
                       (x3^4 + 5*x3^3 + 6*x3^2 + 5*x3 + 1 = 0) ∧
                       (x4^4 + 5*x4^3 + 6*x4^2 + 5*x4 + 1 = 0)) →
  (x1^3 + x2^3 + x3^3 + x4^3 = -54) :=
sorry

end roots_cubic_sum_l303_303998


namespace max_subset_size_l303_303575

theorem max_subset_size :
  ∃ S : Finset ℕ, (∀ (x y : ℕ), x ∈ S → y ∈ S → y ≠ 2 * x) →
  S.card = 1335 :=
sorry

end max_subset_size_l303_303575


namespace product_of_all_positive_divisors_of_18_l303_303388

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l303_303388


namespace exists_rectangle_same_color_l303_303150

def point (x y : ℕ) := x ∈ Finset.range 12 ∧ y ∈ Finset.range 12
def M : Finset (ℕ × ℕ) := 
  Finset.filter (λ p, point p.1 p.2) (Finset.product (Finset.range 12) (Finset.range 12))

def color := {red, white, blue}

noncomputable def color_map : (ℕ × ℕ) → color := sorry

theorem exists_rectangle_same_color :
  ∃ (a b c d : ℕ × ℕ), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧
  a.1 ≠ b.1 ∧ a.2 = b.2 ∧ 
  c.1 ≠ d.1 ∧ c.2 = d.2 ∧
  a.1 = c.1 ∧ b.1 = d.1 ∧ 
  color_map a = color_map b ∧ 
  color_map a = color_map c ∧ 
  color_map a = color_map d :=
sorry

end exists_rectangle_same_color_l303_303150


namespace find_cosB_l303_303154

noncomputable def given_parameters (a c : ℝ) (sinA sinC sinB : ℝ) : Prop :=
  a = 4 ∧ c = 9 ∧ sinA * sinC = sinB^2

theorem find_cosB (a c : ℝ) (sinA sinC sinB : ℝ) 
  (h : given_parameters a c sinA sinC sinB) : 
  cos (B : ℝ) = 61 / 72 :=
by
  rcases h with ⟨ha, hc, hsin⟩
  sorry

end find_cosB_l303_303154


namespace product_of_all_positive_divisors_of_18_l303_303391

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l303_303391


namespace scientific_notation_l303_303761

def z := 10374 * 10^9

theorem scientific_notation (a : ℝ) (n : ℤ) (h₁ : 1 ≤ |a|) (h₂ : |a| < 10) (h₃ : a * 10^n = z) : a = 1.04 ∧ n = 13 := sorry

end scientific_notation_l303_303761


namespace soccer_ball_cost_l303_303765

theorem soccer_ball_cost :
  ∃ x y : ℝ, x + y = 100 ∧ 2 * x + 3 * y = 262 ∧ x = 38 :=
by
  sorry

end soccer_ball_cost_l303_303765


namespace inequality_proof_l303_303083

theorem inequality_proof (b c : ℝ) (hb : 0 < b) (hc : 0 < c) :
  (b - c) ^ 2011 * (b + c) ^ 2011 * (c - b) ^ 2011 ≥ 
  (b ^ 2011 - c ^ 2011) * (b ^ 2011 + c ^ 2011) * (c ^ 2011 - b ^ 2011) := 
by
  sorry

end inequality_proof_l303_303083


namespace length_of_train_is_approx_116_69_l303_303849

-- Define the given conditions
def speed_kmph := 60
def time_s := 7

-- Convert speed from km/hr to m/s
noncomputable def speed_mps := (speed_kmph * 1000) / 3600

-- Define the distance formula based on speed and time
noncomputable def distance := speed_mps * time_s

-- Theorem: Prove that the length of the train is approximately 116.69 meters
theorem length_of_train_is_approx_116_69 :
  abs (distance - 116.69) < 0.01 :=
sorry

end length_of_train_is_approx_116_69_l303_303849


namespace sin_double_angle_l303_303597

theorem sin_double_angle (α : ℝ) 
  (h1 : Real.cos (α + Real.pi / 4) = 3 / 5)
  (h2 : Real.pi / 2 ≤ α ∧ α ≤ 3 * Real.pi / 2) : 
  Real.sin (2 * α) = 7 / 25 := 
by sorry

end sin_double_angle_l303_303597


namespace find_alpha_l303_303258

open Real

/-- The arithmetic means of any three consecutive numbers in the list 
    [sin α, cos α, tan α, sin (2 * α), cos (2 * α)] are equal -/
def arithmetic_means_equal (α : ℝ) : Prop :=
  (sin α + cos α + tan α) / 3 = (cos α + tan α + sin (2 * α)) / 3 ∧
  (cos α + tan α + sin (2 * α)) / 3 = (tan α + sin (2 * α) + cos (2 * α)) / 3

/-- The set of all α for which the arithmetic means of any three consecutive numbers 
    in the list [sin α, cos α, tan α, sin (2 * α), cos (2 * α)] are equal -/
theorem find_alpha (k : ℤ) :
  arithmetic_means_equal α → 
  (α = k * π ∨ α = ± (π / 3) + 2 * k * π ∨ α = 2 * k * π ∨ α = ± (2 * π / 3) + 2 * k * π) :=
sorry

end find_alpha_l303_303258


namespace cost_equivalence_at_325_l303_303664

def cost_plan1 (x : ℕ) : ℝ := 65 + 0.40 * x
def cost_plan2 (x : ℕ) : ℝ := 0.60 * x

theorem cost_equivalence_at_325 : cost_plan1 325 = cost_plan2 325 :=
by sorry

end cost_equivalence_at_325_l303_303664


namespace teena_initial_distance_behind_l303_303243

theorem teena_initial_distance_behind :
  ∀ (teena_speed yoe_speed relative_time distance_ahead),
  teena_speed = 55 ∧ yoe_speed = 40 ∧ relative_time = 1.5 ∧ distance_ahead = 15 →
  (teena_speed - yoe_speed) * relative_time - distance_ahead = 7.5 :=
by
  intros teena_speed yoe_speed relative_time distance_ahead
  intro h
  cases h with h_teena_speed h
  cases h with h_yoe_speed h
  cases h with h_relative_time h_distance_ahead
  rw [h_teena_speed, h_yoe_speed, h_relative_time, h_distance_ahead]
  ring
  sorry

end teena_initial_distance_behind_l303_303243


namespace count_edge_cubes_l303_303875

/-- 
A cube is painted red on all faces and then cut into 27 equal smaller cubes.
Prove that the number of smaller cubes that are painted on only 2 faces is 12. 
-/
theorem count_edge_cubes (c : ℕ) (inner : ℕ)  (edge : ℕ) (face : ℕ) :
  (c = 27 ∧ inner = 1 ∧ edge = 12 ∧ face = 6) → edge = 12 :=
by
  -- Given the conditions from the problem statement
  sorry

end count_edge_cubes_l303_303875


namespace relationship_among_abc_l303_303738

noncomputable def a := 6^0.4
noncomputable def b := logBase 0.4 0.5
noncomputable def c := logBase 8 0.4

theorem relationship_among_abc : c < b ∧ b < a := by
  sorry

end relationship_among_abc_l303_303738


namespace product_of_divisors_of_18_l303_303417

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l303_303417


namespace inhabitants_wrong_l303_303809

theorem inhabitants_wrong (m n : ℕ) (h_gcd : Nat.gcd (m + 1) (n + 1) > 1) :
  ∃ (v : ℕ), ¬(∃ (S : set ℕ), (∀ ᶠ e ∈ S, (e ∈ (1..m+n+1)) ∧ S = {i | 1 ≤ i ∧ i ≤ m + n})):
  sorry

end inhabitants_wrong_l303_303809


namespace convert_to_polar_l303_303925

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2)
  let θ := if y = 0 then if x > 0 then 0 else π else if x = 0 then if y > 0 then π / 2 else 3 * π / 2
           else if x > 0 then real.arctan (y / x) else if y > 0 then real.arctan (y / x) + π else real.arctan (y / x) - π
  in (r, θ)

theorem convert_to_polar : rectangular_to_polar (real.sqrt 3) (-real.sqrt 3) = (real.sqrt 6, 7 * real.pi / 4) :=
by
    sorry

end convert_to_polar_l303_303925


namespace positive_n_value_l303_303037

noncomputable def isRealModulus (z : ℂ) : ℝ := complex.abs z

theorem positive_n_value (n : ℝ) (h : isRealModulus (5 + n * complex.i) = 5 * real.sqrt 13) : n = 10 * real.sqrt 3 :=
by
  sorry

end positive_n_value_l303_303037


namespace value_of_f_at_2_l303_303638

def f (x : ℝ) : ℝ := x^3 - x^2 - 1

theorem value_of_f_at_2 : f 2 = 3 := by
  sorry

end value_of_f_at_2_l303_303638


namespace trig_identity_l303_303105

-- Define the angle alpha with the given condition tan(alpha) = 2
variables (α : ℝ) (h : Real.tan α = 2)

-- State the theorem
theorem trig_identity : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
by
  sorry

end trig_identity_l303_303105


namespace common_sale_days_l303_303865

noncomputable def bookstoreSaleDaysInJuly : List ℕ :=
  [4, 8, 12, 16, 20, 24, 28]

noncomputable def shoeStoreSaleDaysInJuly : List ℕ :=
  [2, 9, 16, 23, 30]

theorem common_sale_days : 
  (List.intersect bookstoreSaleDaysInJuly shoeStoreSaleDaysInJuly).length = 1 := 
by
  sorry

end common_sale_days_l303_303865


namespace max_sum_of_arithmetic_seq_l303_303959

/-- An arithmetic sequence {a_n} satisfies the conditions:
    1. a_6 + a_7 + a_8 > 0
    2. a_6 + a_9 < 0
    To prove: The sum of the first n terms of {a_n} is maximized when n = 7.
-/
theorem max_sum_of_arithmetic_seq (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 5 + a 6 + a 7 > 0)
  (h2 : a 5 + a 8 < 0) :
  let sum_of_first_n (n : ℕ) := ∑ i in range n, a i in
  (argmax sum_of_first_n (range 100)) = 7 :=
sorry

end max_sum_of_arithmetic_seq_l303_303959


namespace maplewood_total_population_l303_303144

-- Define the number of cities
def num_cities : ℕ := 25

-- Define the bounds for the average population
def lower_bound : ℕ := 5200
def upper_bound : ℕ := 5700

-- Define the average population, calculated as the midpoint of the bounds
def average_population : ℕ := (lower_bound + upper_bound) / 2

-- Define the total population as the product of the number of cities and the average population
def total_population : ℕ := num_cities * average_population

-- Theorem statement to prove the total population is 136,250
theorem maplewood_total_population : total_population = 136250 := by
  -- Insert formal proof here
  sorry

end maplewood_total_population_l303_303144


namespace chebyshev_polynomial_sum_of_roots_l303_303734

noncomputable def Pn (n : ℕ) : ℝ → ℝ :=
  λ t, ∑ i in finset.range (n + 1), a i * t^i

def chebyshev (n : ℕ) (t : ℝ) : ℝ :=
  if n = 0 then 1
  else if n = 1 then t
  else 2 * t * chebyshev (n - 1) t - chebyshev (n - 2) t

theorem chebyshev_polynomial :
  ∀ θ : ℝ, Pn 3 (cos θ) = cos (3 * θ) :=
sorry

def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x - 1

theorem sum_of_roots :
  ∀ x₁ x₂ x₃ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0) ∧ (-1 < x₁ ∧ x₁ < 1) ∧ (-1 < x₂ ∧ x₂ < 1) ∧ (-1 < x₃ ∧ x₃ < 1) →
  x₁ + x₂ + x₃ = 0 :=
sorry

end chebyshev_polynomial_sum_of_roots_l303_303734


namespace binomial_coefficient_sum_mod_l303_303531

theorem binomial_coefficient_sum_mod : 
  let S := ((1 + Complex.exp (Complex.I * Real.pi / 2))^2011) + 
           ((1 + Complex.exp (3 * Complex.I * Real.pi / 2))^2011) + 
           ((1 + -1)^2011) + 
           ((1 + 1)^2011)
  in 
  let desired_sum := (range 503).sum (λ j, Nat.choose 2011 (4 * j)) / 4
  in 
  (S % 1000 = 137) :
  nat.Mod 1000 S = 137 := 
begin
  sorry
end

end binomial_coefficient_sum_mod_l303_303531


namespace sum_of_divisors_100_l303_303264

theorem sum_of_divisors_100 : ∑ d in (finset.divisors 100), d = 217 := 
by 
-- Proof would go here, but is not required
sorry

end sum_of_divisors_100_l303_303264


namespace percentage_increase_twice_l303_303799

theorem percentage_increase_twice (P : ℝ) (x : ℝ) :
  P * (1 + x)^2 = P * 1.3225 → x = 0.15 :=
by
  intro h
  have h1 : (1 + x)^2 = 1.3225 := by sorry
  have h2 : x^2 + 2 * x = 0.3225 := by sorry
  have h3 : x = (-2 + Real.sqrt 5.29) / 2 := by sorry
  have h4 : x = -2 / 2 + Real.sqrt 5.29 / 2 := by sorry
  have h5 : x = 0.15 := by sorry
  exact h5

end percentage_increase_twice_l303_303799


namespace clock_angle_at_1030_l303_303901

theorem clock_angle_at_1030 :
  let hour_hand_angle : ℝ := 10.5 * 30
  let minute_hand_angle : ℝ := 30 * 360 / 60
  abs (minute_hand_angle - hour_hand_angle) = 135 := 
by 
  have hour_hand_angle : ℝ := 10.5 * 30
  have minute_hand_angle : ℝ := 30 * 360 / 60
  calc abs (minute_hand_angle - hour_hand_angle) = abs (180 - 315) : by sorry
  ... = abs (-135) : by sorry
  ... = 135 : by sorry

end clock_angle_at_1030_l303_303901


namespace binom_sum_mod_1000_l303_303519

theorem binom_sum_mod_1000 : 
  (∑ i in (finset.range 2012).filter (λ i, i % 4 = 0), nat.choose 2011 i) % 1000 = 15 :=
sorry

end binom_sum_mod_1000_l303_303519


namespace product_of_divisors_of_18_l303_303345

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l303_303345


namespace product_of_divisors_18_l303_303349

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l303_303349


namespace find_a_l303_303120

theorem find_a (a x y : ℝ) (h1 : ax - 3y = 0) (h2 : x + y = 1) (h3 : 2x + y = 0) : a = -6 := 
by sorry

end find_a_l303_303120


namespace composition_points_value_l303_303242

theorem composition_points_value (f g : ℕ → ℕ) (ab cd : ℕ) 
  (h₁ : f 2 = 6) 
  (h₂ : f 3 = 4) 
  (h₃ : f 4 = 2)
  (h₄ : g 2 = 4) 
  (h₅ : g 3 = 2) 
  (h₆ : g 5 = 6) :
  let (a, b) := (2, 6)
  let (c, d) := (3, 4)
  ab + cd = (a * b) + (c * d) :=
by {
  sorry
}

end composition_points_value_l303_303242


namespace scientific_notation_l303_303760

def z := 10374 * 10^9

theorem scientific_notation (a : ℝ) (n : ℤ) (h₁ : 1 ≤ |a|) (h₂ : |a| < 10) (h₃ : a * 10^n = z) : a = 1.04 ∧ n = 13 := sorry

end scientific_notation_l303_303760


namespace decreasing_interval_l303_303576

noncomputable def function_y (x : ℝ) : ℝ := x^2 + x + 1

theorem decreasing_interval : 
  ∀ (x : ℝ), (x <= -1/2) → (function_y' x < 0) :=
sorry

end decreasing_interval_l303_303576


namespace marathon_times_total_l303_303753

theorem marathon_times_total 
  (runs_as_fast : ℝ → ℝ → Prop)
  (takes_more_time : ℝ → ℝ → ℝ → Prop)
  (dean_time : ℝ)
  (h_micah_speed : runs_as_fast 2/3 1)
  (h_jake_time : ∀ t, takes_more_time 1/3 t (t * 4/3))
  (h_dean_time : dean_time = 9) :
  let micah_time := dean_time * (2/3)
  let jake_time := micah_time + (1/3 * micah_time)
  dean_time + micah_time + jake_time = 23 :=
by
  sorry

end marathon_times_total_l303_303753


namespace num_three_digit_nums_with_prime_sum_l303_303672

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n: ℕ) : ℕ := 
  n / 100 + (n / 10) % 10 + n % 10

def is_three_digit (n: ℕ) : Prop := 
  100 ≤ n ∧ n ≤ 999

def three_digit_nums_with_prime_sum :=
  { n : ℕ | is_three_digit n ∧ is_prime (sum_of_digits n) }

theorem num_three_digit_nums_with_prime_sum : 
  (three_digit_nums_with_prime_sum.card) = 37 :=
sorry

end num_three_digit_nums_with_prime_sum_l303_303672


namespace polar_angle_pi_over_4_l303_303550

theorem polar_angle_pi_over_4 :
  { p : ℝ × ℝ // ∃ r : ℝ, p = (r * cos (π / 4), r * sin (π / 4)) } ⊆
  { p : ℝ × ℝ // ∃ k : ℝ, p = (k, k) } :=
sorry

end polar_angle_pi_over_4_l303_303550


namespace incircle_radius_of_isosceles_right_triangle_l303_303825

theorem incircle_radius_of_isosceles_right_triangle :
  ∀ (DF : ℝ), DF = 12 → ∀ (∠D : ℝ), ∠D = 45 → ∃ r : ℝ, r = 12 - 6 * Real.sqrt 2 :=
by
  intros DF hDF ∠D hD
  use 12 - 6 * Real.sqrt 2
  sorry

end incircle_radius_of_isosceles_right_triangle_l303_303825


namespace percent_of_total_cost_is_65_01_l303_303818

theorem percent_of_total_cost_is_65_01
    (cost_bread cost_ham_before_discount cost_cake_before_discount cost_cheese_per_pound : ℝ)
    (ham_discount_percent cake_discount_percent : ℝ)
    (cheese_weight : ℝ) :
    cost_bread = 50 →
    cost_ham_before_discount = 150 →
    cost_cake_before_discount = 200 →
    cost_cheese_per_pound = 75 →
    ham_discount_percent = 0.10 →
    cake_discount_percent = 0.20 →
    cheese_weight = 1.5 →
    ((cost_ham_before_discount * (1 - ham_discount_percent) + cost_bread + cost_cheese_per_pound * cheese_weight) /
    (cost_bread + cost_ham_before_discount * (1 - ham_discount_percent) + cost_cake_before_discount * (1 - cake_discount_percent) + cost_cheese_per_pound * cheese_weight) * 100) ≈ 65.01 := 
by 
  sorry

end percent_of_total_cost_is_65_01_l303_303818


namespace cylinder_surface_area_l303_303877

theorem cylinder_surface_area {d h : ℝ} (h1 : d = 4) (h2 : h = 3) : 
  let r := d / 2 in 
  let S := 2 * Real.pi * r^2 + 2 * Real.pi * r * h in
  S = 20 * Real.pi :=
by
  sorry

end cylinder_surface_area_l303_303877


namespace unpainted_cubes_count_l303_303864

/- Definitions of the conditions -/
def total_cubes : ℕ := 6 * 6 * 6
def painted_faces_per_face : ℕ := 4
def total_faces : ℕ := 6
def painted_faces : ℕ := painted_faces_per_face * total_faces
def overlapped_painted_faces : ℕ := 4 -- Each center four squares on one face corresponds to a center square on the opposite face.
def unique_painted_cubes : ℕ := painted_faces / 2

/- Lean Theorem statement that corresponds to proving the question asked in the problem -/
theorem unpainted_cubes_count : 
  total_cubes - unique_painted_cubes = 208 :=
  by
    sorry

end unpainted_cubes_count_l303_303864


namespace place_mat_side_length_l303_303885

theorem place_mat_side_length :
  ∀ (R : ℝ), R = 5 →
  ∀ (n : ℕ), n = 8 →
  ∀ (x : ℝ),
  let θ := (2 * Real.pi) / n in
  x = 2 * R * Real.sin (θ / 2) →
  x = 5 * Real.sqrt (2 - Real.sqrt 2) := by
  assume R R_eq n n_eq x h,
  rw [R_eq, n_eq, h],
  sorry

end place_mat_side_length_l303_303885


namespace domain_phi_inequality_fx_gx_l303_303615

noncomputable theory

def f (a x : ℝ) := log a (x - 1)
def g (a x : ℝ) := log a (6 - 2 * x)
def φ (a x : ℝ) := f a x + g a x

theorem domain_phi (a : ℝ) (ha : 0 < a ∧ a ≠ 1) :
  ∀ x, (1 < x ∧ x < 3) ↔ (∃ y, φ a y = φ a x) :=
by
  sorry

theorem inequality_fx_gx (a : ℝ) (ha : 0 < a ∧ a ≠ 1) :
  ∀ x, (if 1 < a then (1 < x ∧ x ≤ 7 / 3) else (7 / 3 ≤ x ∧ x < 3)) ↔ 
       (f a x ≤ g a x) :=
by
  sorry

end domain_phi_inequality_fx_gx_l303_303615


namespace relationship_among_a_b_c_l303_303966

theorem relationship_among_a_b_c :
  let a := log 3 / log 2
  let b := log (1 / 2) / log 3
  let c := (1 / 2) ^ 3
  b < c ∧ c < a := 
by
  sorry

end relationship_among_a_b_c_l303_303966


namespace bookPagesProof_l303_303465

-- Define the conditions
def isIdenticalNotebookStructure (n : ℕ) (x : ℕ) : Prop :=
  x = n * ((x / n))

-- Define the constraint on the sum of the pages
def pageSumConstraint (x : ℕ) : Prop :=
  let part1 := (x / 4) + 1 in
  let part2 := (x / 4) + 2 in
  let part3 := (x / 3) - 1 in
  let part4 := x / 3 in
  part1 + part2 + part3 + part4 = 338

-- Main statement to be proved
theorem bookPagesProof : ∀ (x : ℕ), isIdenticalNotebookStructure 12 x ∧ pageSumConstraint x → x = 288 :=
  sorry

end bookPagesProof_l303_303465


namespace total_weight_three_new_people_l303_303246

/-- Let w_new be the total weight of the three new people.
    Given:
    - avg_increase: The average weight of 25 persons increases by 1.8 kg when three people are replaced.
    - w1, w2, w3: The weights of the three replaced people are 70 kg, 80 kg, and 75 kg respectively.
    Prove: The total weight of the three new people (w_new) is 270 kg.
-/ 
theorem total_weight_three_new_people (avg_increase : 1.8) (num_persons : 25) (w1 : 70) (w2 : 80) (w3 : 75) :
  ∃ w_new, w_new = 270 :=
by
  sorry

end total_weight_three_new_people_l303_303246


namespace inequality_holds_l303_303619

-- Given conditions
variables {a b x y : ℝ}
variables (pos_a : 0 < a) (pos_b : 0 < b) (pos_x : 0 < x) (pos_y : 0 < y)
variable (h : a + b = 1)

-- Goal/Question
theorem inequality_holds : (a * x + b * y) * (b * x + a * y) ≥ x * y :=
by sorry

end inequality_holds_l303_303619


namespace initial_participants_l303_303698

theorem initial_participants (p : ℕ) (h1 : 0.6 * p = 0.6 * (p : ℝ)) (h2 : ∀ (n : ℕ), n = 4 * m → 30 = (2 / 5) * n * (1 / 4)) :
  p = 300 :=
by sorry

end initial_participants_l303_303698


namespace Marta_can_guess_correctly_l303_303189

section MartaGame

variable (digits : Finset ℕ) (round1 round2 round3 : Vector ℕ 5)

-- Conditions from the problem
def distinct_digits (n : ℕ) : Prop := (toDigits 10 n).Nodup

def correct_feedback (round : Vector ℕ 5) (num : ℕ) (feedback : Vector Portray 5) : Prop := 
  ∀ i : Fin 5, 
    match feedback[i] with
    | Portray.green  => (toDigits 10 num)[i] = round[i]
    | Portray.yellow => ∃ j : Fin 5, j ≠ i ∧ (toDigits 10 num)[j] = round[i]
    | Portray.gray   => ∀ j : Fin 5, (toDigits 10 num)[j] ≠ round[i]

-- Defining feedbacks from the problem
def feedback_round1 : Vector Portray 5 := ⟨[Portray.green, Portray.gray, Portray.green, Portray.gray, Portray.green], by simpa⟩
def feedback_round2 : Vector Portray 5 := ⟨[Portray.gray, Portray.green, Portray.gray, Portray.gray, Portray.green], by simpa⟩
def feedback_round3 : Vector Portray 5 := ⟨[Portray.green, Portray.yellow, Portray.gray, Portray.green, Portray.gray], by simpa⟩

-- Statement to prove
theorem Marta_can_guess_correctly (num : ℕ) :
  distinct_digits num →
  correct_feedback round1 num feedback_round1 →
  correct_feedback round2 num feedback_round2 →
  correct_feedback round3 num feedback_round3 →
  num = 71284 :=
sorry

end MartaGame

end Marta_can_guess_correctly_l303_303189


namespace distinct_gcd_values_252_l303_303447

noncomputable def number_of_distinct_gcd_values (a b : ℕ) : ℕ :=
  if h : a * b = 252 then
    let g := Nat.gcd a b in
    if (∀ p q r p' q' r', a = 2^p * 3^q * 7^r ∧ b = 2^p' * 3^q' * 7^r' ∧ p + p' = 2 ∧ q + q' = 2∧ r + r' = 1) then
      4
    else
      0
  else 0

theorem distinct_gcd_values_252 : number_of_distinct_gcd_values a b = 4 :=
  by
    sorry

end distinct_gcd_values_252_l303_303447


namespace triplet_unique_solution_l303_303019

theorem triplet_unique_solution {x y z : ℝ} :
  x^2 - 2*x - 4*z = 3 →
  y^2 - 2*y - 2*x = -14 →
  z^2 - 4*y - 4*z = -18 →
  (x = 2 ∧ y = 3 ∧ z = 4) :=
by
  sorry

end triplet_unique_solution_l303_303019


namespace math_problem_l303_303202

variable (a b c : ℝ)

theorem math_problem 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  (ab / c + bc / a + ca / b) ≥ Real.sqrt 3 := 
by
  sorry

end math_problem_l303_303202


namespace point_A_inside_circle_max_min_dist_square_on_circle_chord_through_origin_l303_303077

def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 4 * y - m = 0

def inside_circle (x y m : ℝ) : Prop :=
  (x-1)^2 + (y+2)^2 < 5 + m

theorem point_A_inside_circle (m : ℝ) : -1 < m ∧ m < 4 ↔ inside_circle m (-2) m :=
sorry

def circle_equation_m_4 (x y : ℝ) : Prop :=
  circle_equation x y 4

def dist_square_to_point_H (x y : ℝ) : ℝ :=
  (x - 4)^2 + (y - 2)^2

theorem max_min_dist_square_on_circle (P : ℝ × ℝ) :
  circle_equation_m_4 P.1 P.2 →
  4 ≤ dist_square_to_point_H P.1 P.2 ∧ dist_square_to_point_H P.1 P.2 ≤ 64 :=
sorry

def line_equation (m x y : ℝ) : Prop :=
  y = x + m

theorem chord_through_origin (m : ℝ) :
  ∃ m : ℝ, line_equation m (1 : ℝ) (-2 : ℝ) ∧ 
  (m = -4 ∨ m = 1) :=
sorry

end point_A_inside_circle_max_min_dist_square_on_circle_chord_through_origin_l303_303077


namespace product_of_divisors_of_18_l303_303343

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l303_303343


namespace f_eq_f_inv_iff_x_eq_3_5_l303_303543

def f (x : ℝ) : ℝ := 3 * x - 7
def f_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem f_eq_f_inv_iff_x_eq_3_5 (x : ℝ) : f(x) = f_inv(x) ↔ x = 3.5 := by
  sorry

end f_eq_f_inv_iff_x_eq_3_5_l303_303543


namespace product_of_divisors_of_18_l303_303422

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l303_303422


namespace product_of_all_positive_divisors_of_18_l303_303394

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l303_303394


namespace minimum_guests_l303_303252

-- Define the conditions as variables
def total_food : ℕ := 4875
def max_food_per_guest : ℕ := 3

-- Define the theorem we need to prove
theorem minimum_guests : ∃ g : ℕ, g * max_food_per_guest = total_food ∧ g >= 1625 := by
  sorry

end minimum_guests_l303_303252


namespace a_n_formula_b_n_formula_sum_a2n_b2nm1_l303_303064

variables (a b : ℕ → ℝ) (S : ℕ → ℝ)
variables (d q : ℝ)

-- Conditions
axiom arithmetic_sequence : ∀ n : ℕ, S n = n * (a 1 + a n) / 2
axiom geometric_sequence : ∀ n : ℕ, b n = 2 * q ^ (n - 1)
axiom b2_b3_sum : b 1 * (q + q ^ 2) = 12  -- since b1 = 2
axiom b3_a4_relation : b 3 = a 4 - 2 * a 1
axiom S11_b4_relation : S 11 = 11 * b 4

-- Targets to Prove
theorem a_n_formula : d > 0 → (∀ n: ℕ, a n = 3 * n - 2) :=
sorry

theorem b_n_formula : d > 0 → (∀ n: ℕ, b n = 2 ^ n) :=
sorry

theorem sum_a2n_b2nm1 :
  d > 0 → 
  (∀ n: ℕ, ∑ i in finset.range n, a (2 * i) * b (2 * i - 1) = (3 * n - 2) / 3 * 4 ^ (n + 1) + 8 / 3) :=
sorry

end a_n_formula_b_n_formula_sum_a2n_b2nm1_l303_303064


namespace binomial_coefficient_sum_mod_l303_303528

theorem binomial_coefficient_sum_mod : 
  let S := ((1 + Complex.exp (Complex.I * Real.pi / 2))^2011) + 
           ((1 + Complex.exp (3 * Complex.I * Real.pi / 2))^2011) + 
           ((1 + -1)^2011) + 
           ((1 + 1)^2011)
  in 
  let desired_sum := (range 503).sum (λ j, Nat.choose 2011 (4 * j)) / 4
  in 
  (S % 1000 = 137) :
  nat.Mod 1000 S = 137 := 
begin
  sorry
end

end binomial_coefficient_sum_mod_l303_303528


namespace ferry_speed_difference_l303_303594

-- Definitions of conditions
def time_P : ℝ := 3 -- Ferry P travels for 3 hours
def speed_P : ℝ := 8 -- Ferry P travels at 8 km/h
def time_different : ℝ := 1 -- Journey of ferry Q is 1 hour longer

-- Determine the distance traveled by ferry P
def distance_P : ℝ := speed_P * time_P

-- Determine the distance traveled by ferry Q
def distance_Q : ℝ := 2 * distance_P

-- Time taken by ferry Q
def time_Q : ℝ := time_P + time_different

-- Speed of ferry Q
def speed_Q : ℝ := distance_Q / time_Q

-- Speed difference
def speed_difference : ℝ := speed_Q - speed_P

-- Theorem statement with proof to be provided
theorem ferry_speed_difference : speed_difference = 4 := 
by
  sorry

end ferry_speed_difference_l303_303594


namespace units_digit_of_9_pow_8_pow_7_l303_303031

theorem units_digit_of_9_pow_8_pow_7 :
  let units_digit (n : ℕ) : ℕ := n % 10 in
  units_digit (9 ^ (8 ^ 7)) = 1 :=
by
  sorry

end units_digit_of_9_pow_8_pow_7_l303_303031


namespace product_equals_permutation_l303_303109

theorem product_equals_permutation (m : ℕ) (h : 0 < m) : m * (m + 1) * (m + 2) * ... * (m + 20) = finset.prod (finset.range 21) (λ k, m + k) := sorry

end product_equals_permutation_l303_303109


namespace abc_right_triangle_max_area_l303_303721

-- Define the initial conditions
variables {a b c : ℝ}
variables (Δ : Triangle ℝ) 

-- Conditions
axiom sin_cos_condition : Δ.sinA + Δ.sinB = (Δ.cosA + Δ.cosB) * Δ.sinC
axiom a_b_c_condition : a + b + c = 1 + √2
axiom pythagorean : a^2 + b^2 = c^2

-- Define the problem statements
theorem abc_right_triangle (Δ : Triangle ℝ) (h1 : sin_cos_condition Δ) (h2 : pythagorean a b c) : Δ.angleC = 90 :=
by
  sorry

theorem max_area (Δ : Triangle ℝ) (h1 : a_b_c_condition a b c) (h2 : pythagorean a b c) : 
  Δ.area <= 1 / 4 :=
by
  sorry

end abc_right_triangle_max_area_l303_303721


namespace min_sum_permutations_l303_303924

open Finset

theorem min_sum_permutations :
  let a := (Finset.perm 6).toFunctor;
      b := (Finset.perm 6).toFunctor;
      c := (Finset.perm 6).toFunctor;
      S := (∑ i in range 6, a i * b i * c i)
  in
  S = 162 := by
  sorry

end min_sum_permutations_l303_303924


namespace all_children_receive_candy_iff_power_of_two_l303_303552

theorem all_children_receive_candy_iff_power_of_two (n : ℕ) : 
  (∀ (k : ℕ), k < n → ∃ (m : ℕ), (m * (m + 1) / 2) % n = k) ↔ ∃ (k : ℕ), n = 2^k :=
by sorry

end all_children_receive_candy_iff_power_of_two_l303_303552


namespace product_of_divisors_of_18_l303_303424

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303424


namespace solve_equation_l303_303778

theorem solve_equation (x : ℝ) :
    (sin (2 * x) ≠ 0) →
    (1 / (tan (2 * x)) - cos (2 * x) ≠ 0) →
    (3 * (cos (2 * x) + 1 / (tan (2 * x))) / (1 / (tan (2 * x)) - cos (2 * x)) - 2 * (sin (2 * x) + 1) = 0) →
    (∃ (k : ℤ), x = ((-1) ^ (k + 1)) * (Real.pi / 12) + (k * (Real.pi / 2))) := sorry

end solve_equation_l303_303778


namespace inequality_sqrt_three_l303_303209

theorem inequality_sqrt_three (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h : a^2 + b^2 + c^2 = 1) : 
  (ab / c + bc / a + ca / b) ≥ √3 :=
by
  sorry

end inequality_sqrt_three_l303_303209


namespace intersection_points_circle_radius_squared_l303_303259

noncomputable def intersection_parabolas_radius_squared : ℝ :=
  let intersect_points := { p : ℝ × ℝ | p.snd = (p.fst - 2)^2 ∧ p.fst + 6 = (p.snd - 1)^2 };
  let radius_squared := (intersect_points.to_finset : finset (ℝ × ℝ)).to_list /- slightly informal, needs further formalization -/
  sorry -- exact solution of finding points will be proven here

theorem intersection_points_circle_radius_squared : intersection_parabolas_radius_squared = 4 :=
sorry -- detailed proof here

end intersection_points_circle_radius_squared_l303_303259


namespace average_non_holiday_visitors_l303_303897

theorem average_non_holiday_visitors :
  ∃ x : ℝ, 
    let total_days := 327
    let holiday_events := 14
    let total_visitors := 406
    let non_holiday_days := total_days - holiday_events
    let holiday_visitors := 2 * x * holiday_events
    let non_holiday_visitors := x * non_holiday_days
    non_holiday_visitors + holiday_visitors = total_visitors → x = 1 :=
begin
  sorry
end

end average_non_holiday_visitors_l303_303897


namespace pyramid_volume_l303_303921

-- Define the conditions
def base_length : ℝ := 4
def height : ℝ := 6
def base_area : ℝ := base_length^2
def volume (base_area height : ℝ) : ℝ := (1 / 3) * base_area * height

-- The theorem we need to prove
theorem pyramid_volume : volume base_area height = 32 :=
by
  -- We state the conditions explicitly here
  have base_length_val : base_length = 4 := rfl
  have height_val : height = 6 := rfl
  have base_area_val : base_area = 16 := by
    dsimp [base_area, base_length]
    rw base_length_val
    norm_num
  -- Calculate the volume
  dsimp [volume, base_area, height]
  rw [base_area_val, height_val]
  norm_num
  sorry

end pyramid_volume_l303_303921


namespace product_of_divisors_of_18_l303_303376

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303376


namespace simplify_complex_subtraction_l303_303225

-- Definition of the nested expression
def complex_subtraction (x : ℝ) : ℝ :=
  1 - (2 - (3 - (4 - (5 - (6 - x)))))

-- Statement of the theorem to be proven
theorem simplify_complex_subtraction (x : ℝ) : complex_subtraction x = x - 3 :=
by {
  -- This proof will need to be filled in to verify the statement
  sorry
}

end simplify_complex_subtraction_l303_303225


namespace complex_fraction_simplification_l303_303230

theorem complex_fraction_simplification (i : ℂ) (hi : i^2 = -1) : 
  ((2 - i) / (1 + 4 * i)) = (-2 / 17 - (9 / 17) * i) :=
  sorry

end complex_fraction_simplification_l303_303230


namespace meeting_time_and_location_l303_303883

/-- Define the initial conditions -/
def start_time : ℕ := 8 -- 8:00 AM
def city_distance : ℕ := 12 -- 12 kilometers
def pedestrian_speed : ℚ := 6 -- 6 km/h
def cyclist_speed : ℚ := 18 -- 18 km/h

/-- Define the conditions for meeting time and location -/
theorem meeting_time_and_location :
  ∃ (meet_time : ℕ) (meet_distance : ℚ),
    meet_time = 9 * 60 + 15 ∧   -- 9:15 AM in minutes
    meet_distance = 4.5 :=      -- 4.5 kilometers
sorry

end meeting_time_and_location_l303_303883


namespace minimum_area_of_triangle_ABC_is_2_l303_303091

noncomputable def minimum_area_of_triangle_ABC (A B : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  let center := (1, -1)
  let radius := Real.sqrt 2
  let distance := λ p q : ℝ × ℝ, Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)
  let triangle_area := λ p q r : ℝ × ℝ, 
    0.5 * Real.abs ((q.1 - p.1) * (r.2 - p.2) - (r.1 - p.1) * (q.2 - p.2))
  ∃ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 2 ∧ 
    triangle_area A B (x, y) = 2

-- Define points
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 2)

-- Prove the statement
theorem minimum_area_of_triangle_ABC_is_2 : minimum_area_of_triangle_ABC A B (1, -1) :=
sorry

end minimum_area_of_triangle_ABC_is_2_l303_303091


namespace angle_trisector_l303_303481

theorem angle_trisector (A B C D L A' B' C' : ℝ)
  (h_rectangular: ∃ P Q R S : ℝ, PQ = AD ∧ QR = AD ∧ RS = AD ∧ SP = AD ∧ ∠PAD = 90)
  (h_angle: ∃ x : ℝ, 0 < x ∧ x < 180 ∧ ∠LAD = x) 
  (h_AB_BC: AB = BC)
  (h_fold: reflect_fold C = C' ∧ reflect_fold A = A' ∧ reflect_fold B = B') :
  (∠BA'A = x/3 ∧ ∠AA'B = x/3) :=
sorry

end angle_trisector_l303_303481


namespace new_mean_proof_l303_303071

-- Define the condition of the original mean
def original_mean_condition (x1 x2 x3 x4 : ℝ) : Prop :=
  (x1 + x2 + x3 + x4) / 4 = 5

-- Define the function to calculate the mean of the new set
def new_mean (x1 x2 x3 x4 : ℝ) : ℝ :=
  (x1 + 1 + (x2 + 2) + (x3 + x4 + 4) + (5 + 5)) / 4

-- State the theorem
theorem new_mean_proof (x1 x2 x3 x4 : ℝ) :
  original_mean_condition x1 x2 x3 x4 → new_mean x1 x2 x3 x4 = 8 := by
sorry

end new_mean_proof_l303_303071


namespace EFGH_area_l303_303148

/-
We are given that:
1. $ABCD$ is a square with side $AB = 10$ units.
2. $BE = 2$ units.
3. $EFGH$ is a rectangle inside square $ABCD$ such that:
   - $E$ and $F$ are points on $AB$ and $CD$ respectively.
   - $G$ and $H$ are on $AD$ and $BC$, respectively.
   - $EH$ and $FG$ are parallel to $AB$.
4. $EH = 2x$.

We need to prove that the area of rectangle $EFGH$ is $\frac{16\sqrt{6}}{3}$ given $x = \frac{4\sqrt{6}}{3}$.
-/

theorem EFGH_area (ABCD : Type*) [square ABCD]
  (AB : segment ABCD) (E : point ABCD) (F : point ABCD)
  (G : point ABCD) (H : point ABCD)
  (BE : ℝ) (EH : ℝ) (x : ℝ) :
  side AB = 10 ∧ side BE = 2 ∧ side EH = 2 * x ∧
  parallel EH AB ∧ parallel FG AB →
  x = 4 * (sqrt 6) / 3 →
  area EFGH = 16 * (sqrt 6) / 3 :=
by
  intros h h1
  sorry

end EFGH_area_l303_303148


namespace product_of_divisors_of_18_l303_303370

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303370


namespace product_S_l303_303652

noncomputable def a (n : ℕ) : ℝ := 1 / 2^n

def S (n : ℕ) : ℝ := ∑ i in finset.range n, 1 / (real.logb 2 (a i) * real.logb 2 (a (i + 1)))

theorem product_S : (∏ i in finset.range 10, S (i + 1)) = 1 / 11 := by
  sorry

end product_S_l303_303652


namespace total_time_correct_l303_303756

-- Definitions for the conditions
def dean_time : ℕ := 9
def micah_time : ℕ := (2 * dean_time) / 3
def jake_time : ℕ := micah_time + micah_time / 3

-- Proof statement for the total time
theorem total_time_correct : micah_time + dean_time + jake_time = 23 := by
  sorry

end total_time_correct_l303_303756


namespace equal_cost_miles_l303_303666

   -- Conditions:
   def initial_fee_first_plan : ℝ := 65
   def cost_per_mile_first_plan : ℝ := 0.40
   def cost_per_mile_second_plan : ℝ := 0.60

   -- Proof problem:
   theorem equal_cost_miles : 
     let x := 325 in 
     initial_fee_first_plan + cost_per_mile_first_plan * x = cost_per_mile_second_plan * x :=
   by
     -- Placeholder for the proof
     sorry
   
end equal_cost_miles_l303_303666


namespace binomial_coeff_x3_in_1_plus_2x_to_5_l303_303714

theorem binomial_coeff_x3_in_1_plus_2x_to_5 :
  (∃ c, (1 + 2 * (x : ℝ))^5 = ∑ k in finset.range 6, (nat.choose 5 k) * (1 : ℝ)^(5 - k) * (2 * x)^k
  ∧ (∀ x, (c = 80) → ((nat.choose 5 3) * 2^3 = c))) :=
sorry

end binomial_coeff_x3_in_1_plus_2x_to_5_l303_303714


namespace marathon_times_total_l303_303754

theorem marathon_times_total 
  (runs_as_fast : ℝ → ℝ → Prop)
  (takes_more_time : ℝ → ℝ → ℝ → Prop)
  (dean_time : ℝ)
  (h_micah_speed : runs_as_fast 2/3 1)
  (h_jake_time : ∀ t, takes_more_time 1/3 t (t * 4/3))
  (h_dean_time : dean_time = 9) :
  let micah_time := dean_time * (2/3)
  let jake_time := micah_time + (1/3 * micah_time)
  dean_time + micah_time + jake_time = 23 :=
by
  sorry

end marathon_times_total_l303_303754


namespace length_of_train_l303_303894

/-- A train takes 6 seconds to cross a man walking at 5 kmph in the opposite direction. 
The speed of the train is 84.99280057595394 kmph. Prove that the length of the train is approximately 149.988 meters. -/
theorem length_of_train (speed_train : ℝ) (speed_man : ℝ) (time_crossing : ℝ) : 
  speed_train = 84.99280057595394 → 
  speed_man = 5 → 
  time_crossing = 6 → 
  let relative_speed := (speed_train + speed_man) * (5/18) in
  (relative_speed * time_crossing ≈ 149.98800095992656) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  let relative_speed := (84.99280057595394 + 5) * (5 / 18)
  let length_of_train := relative_speed * 6
  show length_of_train ≈ 149.98800095992656
  sorry

end length_of_train_l303_303894


namespace mod_computation_l303_303238

theorem mod_computation : ∃ m, (289 * 673 ≡ m [MOD 50]) ∧ (0 ≤ m ∧ m < 50) ∧ (m = 47) :=
by {
  use 47,
  split,
  { norm_num }, -- This simplifies the numerical computations internally and checks the congruence
  split,
  { norm_num }, -- This verifies that 47 is within the specified range
  refl, -- This asserts that m = 47 exactly
}

end mod_computation_l303_303238


namespace chandu_work_days_l303_303899

theorem chandu_work_days (W : ℝ) (c : ℝ) 
  (anand_rate : ℝ := W / 7) 
  (bittu_rate : ℝ := W / 8) 
  (chandu_rate : ℝ := W / c) 
  (completed_in_7_days : 3 * anand_rate + 2 * bittu_rate + 2 * chandu_rate = W) : 
  c = 7 :=
by
  sorry

end chandu_work_days_l303_303899


namespace a1_divides_a2_and_a2_divides_a3_probability_l303_303177

noncomputable def probability_a1_divides_a2_and_a2_divides_a3 : ℚ :=
  let n := 21
  let favorable := Nat.choose (n + 1) 3
  let total := n ^ 6
  (favorable * favorable) / total

theorem a1_divides_a2_and_a2_divides_a3_probability :
  probability_a1_divides_a2_and_a2_divides_a3 = 2371600 / 85766121 :=
by
  -- Calculation and combination of results are directly stated
  sorry

end a1_divides_a2_and_a2_divides_a3_probability_l303_303177


namespace product_of_divisors_of_18_l303_303425

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303425


namespace sum_of_consecutive_perfect_squares_l303_303030

theorem sum_of_consecutive_perfect_squares (k : ℕ) (h_pos : 0 < k)
  (h_eq : 2 * k^2 + 2 * k + 1 = 181) : k = 9 ∧ (k + 1) = 10 := by
  sorry

end sum_of_consecutive_perfect_squares_l303_303030


namespace triangle_area_l303_303020

theorem triangle_area (a b c : ℕ) (h1 : a = 14) (h2 : b = 48) (h3 : c = 50)
  (h4 : a^2 + b^2 = c^2) : 1/2 * (a : ℝ) * (b : ℝ) = 336 :=
by {
  have h: 14^2 + 48^2 = 50^2 := by norm_num,
  rw [h1, h2, h3],
  exact sorry,
}

end triangle_area_l303_303020


namespace find_ray_solutions_l303_303946

noncomputable def polynomial (a x : ℝ) : ℝ :=
  x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3

theorem find_ray_solutions (a : ℝ) :
  (∀ x : ℝ, polynomial a x ≥ 0 → ∃ b : ℝ, ∀ y ≥ b, polynomial a y ≥ 0) ↔ a = 1 ∨ a = -1 :=
sorry

end find_ray_solutions_l303_303946


namespace product_of_divisors_18_l303_303350

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l303_303350


namespace potential_damage_proportion_l303_303472

theorem potential_damage_proportion (length_log : ℝ) (length_bed width_bed : ℝ) :
  length_log = 2 ∧ length_bed = 3 ∧ width_bed = 2 →
  (1 - π / 6) = (1 - (π / (length_bed * width_bed))) :=
by
  intro h,
  cases h with h_length_log h_bed,
  cases h_bed with h_length_bed h_width_bed,
  rw [h_length_log, h_length_bed, h_width_bed],
  sorry

end potential_damage_proportion_l303_303472


namespace zero_function_is_uniq_l303_303018

theorem zero_function_is_uniq (f : ℝ → ℝ) :
  (∀ (x : ℝ) (hx : x ≠ 0) (y : ℝ), f (x^2 + y) ≥ (1/x + 1) * f y) → 
  (∀ x, f x = 0) :=
by
  sorry

end zero_function_is_uniq_l303_303018


namespace highway_length_on_map_l303_303806

theorem highway_length_on_map (total_length_km : ℕ) (scale : ℚ) (length_on_map_cm : ℚ) 
  (h1 : total_length_km = 155) (h2 : scale = 1 / 500000) :
  length_on_map_cm = 31 :=
by
  sorry

end highway_length_on_map_l303_303806


namespace factor_quadratic_l303_303939

theorem factor_quadratic (x : ℝ) (m n : ℝ) 
  (hm : m^2 = 16) (hn : n^2 = 25) (hmn : 2 * m * n = 40) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := 
by sorry

end factor_quadratic_l303_303939


namespace unit_vector_perpendicular_l303_303582

theorem unit_vector_perpendicular (x y : ℝ) (h : 3 * x + 4 * y = 0) (m : x^2 + y^2 = 1) : 
  (x = -4/5 ∧ y = 3/5) ∨ (x = 4/5 ∧ y = -3/5) :=
by
  sorry

end unit_vector_perpendicular_l303_303582


namespace cos_angle_B_l303_303988

variable {V : Type} [AddCommGroup V] [Module ℝ V]

structure Triangle (V : Type) [AddCommGroup V] [Module ℝ V] :=
  (A B C I : V)

open Triangle

def incenter_property (T : Triangle V) : Prop :=
  2 • (T.I - T.A) + 5 • (T.I - T.B) + 6 • (T.I - T.C) = 0

theorem cos_angle_B (T : Triangle V) (h : incenter_property T) : 
  ∃ k : ℝ, (cos_angle T.A T.B T.C = 5 / 8) :=
sorry

end cos_angle_B_l303_303988


namespace sum_of_angles_l303_303955

theorem sum_of_angles (x : ℝ) (h : 0 ≤ x ∧ x ≤ 360) :
  (∑ y in {x | 0 ≤ x ∧ x ≤ 360 ∧ sin x^3 - cos x^3 = 1 / cos x - 1 / sin x}.to_finset, y) = 270 :=
sorry

end sum_of_angles_l303_303955


namespace product_of_divisors_of_18_l303_303361

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303361


namespace blue_edges_exist_l303_303733

theorem blue_edges_exist (n : ℕ) (hn : Odd n) (h_edges : (∑ i in Finset.range n, 2 * (n + 1)) = n^2 + n ^ 2) 
    (h_red : ∃ r, r ≤ n^2) : 
    ∃ s : ℕ, s < n ∧ (∃ k, k >= 3) :=
by sorry

end blue_edges_exist_l303_303733


namespace inclination_angle_of_line_l303_303796

theorem inclination_angle_of_line (α : ℝ) : 
  (∀ x, y = sqrt 3 * x + 1 → (α ∈ Set.Ico 0 180) → tan α = sqrt 3 → α = 60) := sorry

end inclination_angle_of_line_l303_303796


namespace correct_frustum_proposition_l303_303493

-- Definitions of the propositions
def propA : Prop := ∀ (pyramid : Type) (plane : Type), (cut_pyramid_with_plane_formed_frustum pyramid plane)
def propB : Prop := ∀ (polyhedron : Type) (parallel_bases : Type), all_other_faces_trapezoids polyhedron parallel_bases -> is_frustum polyhedron
def propC : Prop := ∀ (frustum : Type), are_bases_similar_squares frustum
def propD : Prop := ∀ (frustum : Type), lateral_edges_intersect_at_point_if_extended frustum

-- The theorem to prove the correct proposition D about frustums
theorem correct_frustum_proposition : propD := sorry

end correct_frustum_proposition_l303_303493


namespace pentagon_equality_l303_303497

noncomputable def regular_pentagon (A B C D E : Point) : Prop :=
  is_regular_pentagon A B C D E

noncomputable def is_on_arc (P A E : Point) (O : Circle) : Prop :=
  lies_on_arc P A E O

theorem pentagon_equality (A B C D E P : Point) (O : Circle) 
  (h1 : regular_pentagon A B C D E)
  (h2 : is_on_arc P A E O) : 
  distance P A + distance P C + distance P E = distance P B + distance P D :=
sorry

end pentagon_equality_l303_303497


namespace sam_watermelons_second_batch_l303_303222

theorem sam_watermelons_second_batch
  (initial_watermelons : ℕ)
  (total_watermelons : ℕ)
  (second_batch_watermelons : ℕ) :
  initial_watermelons = 4 →
  total_watermelons = 7 →
  second_batch_watermelons = total_watermelons - initial_watermelons →
  second_batch_watermelons = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sam_watermelons_second_batch_l303_303222


namespace find_interest_rate_of_second_part_l303_303486

-- Definitions for the problem
def total_sum : ℚ := 2678
def P2 : ℚ := 1648
def P1 : ℚ := total_sum - P2
def r1 : ℚ := 0.03  -- 3% per annum
def t1 : ℚ := 8     -- 8 years
def I1 : ℚ := P1 * r1 * t1
def t2 : ℚ := 3     -- 3 years

-- Statement to prove
theorem find_interest_rate_of_second_part : ∃ r2 : ℚ, I1 = P2 * r2 * t2 ∧ r2 * 100 = 5 := by
  sorry

end find_interest_rate_of_second_part_l303_303486


namespace final_selling_price_l303_303816

noncomputable def calculateSellingPrice (cost price profit: ℕ) : ℕ := 
  price + profit

theorem final_selling_price :
  let cost_A := 240 in
  let profit_A := (20 * cost_A) / 100 in
  let selling_A := calculateSellingPrice cost_A profit_A in
  
  let cost_B := 150 in
  let profit_B := (15 * cost_B) / 100 in
  let selling_B := calculateSellingPrice cost_B profit_B in
  
  let cost_C := 350 in
  let profit_C := (25 * cost_C) / 100 in
  let selling_C := calculateSellingPrice cost_C profit_C in
  
  let total_selling_price := selling_A + selling_B + selling_C in
  let discount := (10 * total_selling_price) / 100 in
  let final_price := total_selling_price - discount in
  
  final_price = 808.20 := sorry

end final_selling_price_l303_303816


namespace abs_opposite_sign_eq_sum_l303_303113

theorem abs_opposite_sign_eq_sum (a b : ℤ) (h : (|a + 1| * |b + 2| < 0)) : a + b = -3 :=
sorry

end abs_opposite_sign_eq_sum_l303_303113


namespace complement_inter_proof_l303_303656

open Set

variable (U : Set ℕ) (A B : Set ℕ)

def complement_inter (U A B : Set ℕ) : Set ℕ :=
  compl (A ∩ B)

theorem complement_inter_proof (hU : U = {1, 2, 3, 4, 5, 6, 7, 8} )
  (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4, 5}) :
  complement_inter U A B = {1, 4, 5, 6, 7, 8} :=
by
  sorry

end complement_inter_proof_l303_303656


namespace perimeter_trapezoid_l303_303720

theorem perimeter_trapezoid 
(E F G H : Point)
(EF GH : ℝ)
(HJ EI FG EH : ℝ)
(h_eq1 : EF = GH)
(h_FG : FG = 10)
(h_EH : EH = 20)
(h_EI : EI = 5)
(h_HJ : HJ = 5)
(h_EF_HG : EF = Real.sqrt (EI^2 + ((EH - FG) / 2)^2)) :
  2 * EF + FG + EH = 30 + 10 * Real.sqrt 2 :=
by
  sorry

end perimeter_trapezoid_l303_303720


namespace sales_worth_l303_303453

variables (S : ℝ)
variables (old_scheme_remuneration new_scheme_remuneration : ℝ)

def old_scheme := 0.05 * S
def new_scheme := 1300 + 0.025 * (S - 4000)

theorem sales_worth :
  new_scheme S = old_scheme S + 600 →
  S = 24000 :=
by
  intro h
  sorry

end sales_worth_l303_303453


namespace find_line_l303_303470

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the circle equation
def circle (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the line l that passes through P and intersects the circle
def intersects_circle (line_eq : ℝ → ℝ) : Prop :=
  ∃ A B : ℝ × ℝ,
    (circle A.fst A.snd ∧ circle B.fst B.snd) ∧
    A.fst ≠ B.fst ∧ A.snd ≠ B.snd ∧
    ∥A - B∥ = 4 * Real.sqrt 2 ∧
    line_eq A.fst = A.snd ∧ line_eq B.fst = B.snd

-- Define the theorem that the line l is either x = 1 or 3x - 4y + 5 = 0
theorem find_line : ∃ l : ℝ → ℝ, 
  (intersects_circle l ∧ (l 1 = 2)) →
  l = (λ x, 2) ∨ l = (λ x, (3*x + 5) / 4) :=
by
  sorry

end find_line_l303_303470


namespace product_of_divisors_of_18_l303_303415

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l303_303415


namespace hyperbola_eccentricity_l303_303067

theorem hyperbola_eccentricity (a b : ℝ) :
  (∀ x, y = ± 2*x --> (y = 2*x ∨ y = -2*x)) →
  (sqrt (1 + (b/a)^2) = sqrt 5 ∨ sqrt (1 + (b/a)^2) = sqrt 5 / 2) := by
sorry

end hyperbola_eccentricity_l303_303067


namespace exists_point_P_l303_303775

def acute_angled_triangle (A B C : Point ℝ) : Prop :=
  ∠BCA < π / 2 ∧ ∠CAB < π / 2 ∧ ∠ABC < π / 2

theorem exists_point_P (A B C : Point ℝ) (h : acute_angled_triangle A B C) :
  ∃ P : Point ℝ, (∀ Q : Point ℝ, lies_on Q (line B C) → ∠AQP = π / 2) ∧
                 (∀ Q : Point ℝ, lies_on Q (line C A) → ∠BQP = π / 2) ∧
                 (∀ Q : Point ℝ, lies_on Q (line A B) → ∠CQP = π / 2) :=
sorry

end exists_point_P_l303_303775


namespace second_quadrant_points_l303_303056

theorem second_quadrant_points (x y : ℤ) (P : ℤ × ℤ) :
  P = (x, y) ∧ x < 0 ∧ y > 0 ∧ y ≤ x + 4 →
  P ∈ {(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-3, 1)} :=
by
  sorry

end second_quadrant_points_l303_303056


namespace five_pow_n_starts_with_1_l303_303983

theorem five_pow_n_starts_with_1 (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 2017) 
  (h1 : (floor (2018 * log 10 5) + 1 = 1411))
  (h2 : (floor (2018 * log 10 5) = 3 * 10 ^ (1411 - 1))) :
  ∃ k : ℕ, k = 607 ∧ 
    (∀ n, 1 ≤ n ∧ n ≤ 2017 → fractional_part (n * log 10 5) < log 10 2) := 
  sorry

end five_pow_n_starts_with_1_l303_303983


namespace solution_of_phi_l303_303993

theorem solution_of_phi 
    (φ : ℝ) 
    (H : ∃ k : ℤ, 2 * (π / 6) + φ = k * π) :
    φ = - (π / 3) := 
sorry

end solution_of_phi_l303_303993


namespace arithmetic_seq_geometric_seq_conditions_l303_303608

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 + n * d

-- Define the sum of the first n terms of the sequence
def sum_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n.succ * (a 0 + a n) / 2)

-- Define the geometric sequence condition for a1, a4, a6
def geometric_seq_cond (a : ℕ → ℝ) : Prop :=
  a 0 * a 5 = (a 3) ^ 2

-- The main goal translated into Lean 4
theorem arithmetic_seq_geometric_seq_conditions
  (a : ℕ → ℝ) (d : ℝ) (h_seq : arithmetic_seq a d)
  (h_geo : geometric_seq_cond a) (h_d : d ≠ 0) :
  a 0 = -9 * d ∧ sum_seq a 18 = 0 ∧
  (d < 0 → ∀ n, sum_seq a 8 = sum_seq a n → n ≤ 8) ∧
  (d > 0 → ∀ n, sum_seq a 9 = sum_seq a n → n ≥ 9) :=
sorry

end arithmetic_seq_geometric_seq_conditions_l303_303608


namespace doughnuts_remaining_l303_303556

theorem doughnuts_remaining 
  (total_doughnuts : ℕ)
  (total_staff : ℕ)
  (staff_3_doughnuts : ℕ)
  (doughnuts_eaten_by_3 : ℕ)
  (staff_2_doughnuts : ℕ)
  (doughnuts_eaten_by_2 : ℕ)
  (staff_4_doughnuts : ℕ)
  (doughnuts_eaten_by_4 : ℕ) :
  total_doughnuts = 120 →
  total_staff = 35 →
  staff_3_doughnuts = 15 →
  staff_2_doughnuts = 10 →
  doughnuts_eaten_by_3 = staff_3_doughnuts * 3 →
  doughnuts_eaten_by_2 = staff_2_doughnuts * 2 →
  staff_4_doughnuts = total_staff - (staff_3_doughnuts + staff_2_doughnuts) →
  doughnuts_eaten_by_4 = staff_4_doughnuts * 4 →
  total_doughnuts - (doughnuts_eaten_by_3 + doughnuts_eaten_by_2 + doughnuts_eaten_by_4) = 15 :=
by
  intros
  -- Proof goes here
  sorry

end doughnuts_remaining_l303_303556


namespace sequence_sum_formula_l303_303604

open Nat

-- Definitions from conditions
def a : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n + 2) => a (n + 2) - a (n + 1)

def S : ℕ → ℕ
| 0 => a 0
| 1 => a 0 + a 1
| (n + 1) => S n + a (n + 1)

-- Theorem statement to be proved
theorem sequence_sum_formula (n : ℕ) : S n = 2^n - 1 := by
  sorry

end sequence_sum_formula_l303_303604


namespace manage_committee_combination_l303_303887

theorem manage_committee_combination : (Nat.choose 20 3) = 1140 := by
  sorry

end manage_committee_combination_l303_303887


namespace shaded_fraction_is_half_l303_303135

-- Define the number of rows and columns in the grid
def num_rows : ℕ := 8
def num_columns : ℕ := 8

-- Define the number of shaded triangles based on the pattern explained
def shaded_rows : List ℕ := [1, 3, 5, 7]
def num_shaded_rows : ℕ := 4
def triangles_per_row : ℕ := num_columns
def num_shaded_triangles : ℕ := num_shaded_rows * triangles_per_row

-- Define the total number of triangles
def total_triangles : ℕ := num_rows * num_columns

-- Define the fraction of shaded triangles
def shaded_fraction : ℚ := num_shaded_triangles / total_triangles

-- Prove the shaded fraction is 1/2
theorem shaded_fraction_is_half : shaded_fraction = 1 / 2 :=
by
  -- Provide the calculations
  sorry

end shaded_fraction_is_half_l303_303135


namespace zah_to_bah_l303_303680

def bah_to_rah (bah : ℝ) : ℝ := 3 * bah
def rah_to_yah (rah : ℝ) : ℝ := 1.5 * rah
def yah_to_zah (yah : ℝ) : ℝ := 2.5 * yah

def zah_to_yah (zah : ℝ) : ℝ := zah / 2.5
def yah_to_rah (yah : ℝ) : ℝ := yah / 1.5
def rah_to_bah (rah : ℝ) : ℝ := rah / 3

theorem zah_to_bah (zah : ℝ) : bah_to_rah (rah_to_yah (yah_to_zah zah)) = 133.33 :=
by
  -- Applying conversions, skipping detailed steps
  sorry

end zah_to_bah_l303_303680


namespace translate_parabola_down_l303_303275

theorem translate_parabola_down (x : ℝ) : 
  let y := 3 * x^2,
  let y_translated := y - 2
  in y_translated = 3 * x^2 - 2 :=
by 
  sorry

end translate_parabola_down_l303_303275


namespace infinitely_many_primes_in_arithmetic_progression_l303_303851

/-- Theorem stating that there are infinitely many primes in the arithmetic progression 11 + 10n. -/
theorem infinitely_many_primes_in_arithmetic_progression : ∀ (p : ℕ), ∃ ∞ n, prime (11 + 10 * n) :=
begin
  sorry
end

end infinitely_many_primes_in_arithmetic_progression_l303_303851


namespace congruence_theorem_l303_303448

def triangle_congruent_SSA (a b : ℝ) (gamma : ℝ) :=
  b * b = a * a + (-2 * a * 5 * Real.cos gamma) + 25

theorem congruence_theorem : triangle_congruent_SSA 3 5 (150 * Real.pi / 180) :=
by
  -- Proof is omitted, based on the problem's instruction.
  sorry

end congruence_theorem_l303_303448


namespace product_of_divisors_of_18_l303_303342

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l303_303342


namespace tank_fill_time_l303_303768

theorem tank_fill_time :
  let fill_rate_A := 1 / 8
  let empty_rate_B := 1 / 24
  let combined_rate := fill_rate_A - empty_rate_B
  let time_with_both_pipes := 66
  let partial_fill := time_with_both_pipes * combined_rate
  let remaining_fill := 1 - (partial_fill % 1)
  let additional_time_A := remaining_fill / fill_rate_A
  time_with_both_pipes + additional_time_A = 70 :=
by
  let fill_rate_A := 1 / 8
  let empty_rate_B := 1 / 24
  let combined_rate := fill_rate_A - empty_rate_B
  let time_with_both_pipes := 66
  let partial_fill := time_with_both_pipes * combined_rate
  let remaining_fill := 1 - (partial_fill % 1)
  let additional_time_A := remaining_fill / fill_rate_A
  have h : time_with_both_pipes + additional_time_A = 70 := sorry
  exact h

end tank_fill_time_l303_303768


namespace trajectory_and_area_l303_303603

-- Definitions for the conditions
def distance_to_line (P : ℝ × ℝ) (x_val : ℝ) : ℝ :=
  abs (P.1 - x_val)

def distance_to_point (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def is_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 4) + (P.2^2 / 3) = 1

-- Problem statement
theorem trajectory_and_area :
  ∀ (P : ℝ × ℝ),
  (distance_to_line P 4 = 2 * distance_to_point P (1, 0)) →
  is_ellipse P ∧
  let A := (2, 0) in
  let line_f1 (x : ℝ) := x - 1 in
  let intersection_points : (ℝ × ℝ) × (ℝ × ℝ) := sorry in -- find C and D
  let |y1_y2| := abs (intersection_points.1.2 - intersection_points.2.2) in
  ∃ S : ℝ, S = (1 / 2) * 1 * |y1_y2| ∧ S = 6 * real.sqrt 2 / 7 :=
sorry

end trajectory_and_area_l303_303603


namespace prove_ratio_chickens_pigs_horses_sheep_l303_303848

noncomputable def ratio_chickens_pigs_horses_sheep (c p h s : ℕ) : Prop :=
  (∃ k : ℕ, c = 26*k ∧ p = 5*k) ∧
  (∃ l : ℕ, s = 25*l ∧ h = 9*l) ∧
  (∃ m : ℕ, p = 10*m ∧ h = 3*m) ∧
  c = 156 ∧ p = 30 ∧ h = 9 ∧ s = 25

theorem prove_ratio_chickens_pigs_horses_sheep (c p h s : ℕ) :
  ratio_chickens_pigs_horses_sheep c p h s :=
sorry

end prove_ratio_chickens_pigs_horses_sheep_l303_303848


namespace arithmetic_sequence_geometric_condition_l303_303977

theorem arithmetic_sequence_geometric_condition (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d = 3)
  (h3 : ∃ k, a (k+3) * a k = (a (k+1)) * (a (k+2))) :
  a 2 = -9 :=
by
  sorry

end arithmetic_sequence_geometric_condition_l303_303977


namespace radius_of_sphere_with_same_volume_as_frustum_l303_303871

noncomputable def conicalFrustumVolume (R r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * h * (R^2 + R * r + r^2)

noncomputable def sphereVolume (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem radius_of_sphere_with_same_volume_as_frustum :
  ∀ (h R r : ℝ), h = 5 ∧ R = 2 ∧ r = 3 →
    ∃ r_sphere : ℝ, sphereVolume r_sphere = conicalFrustumVolume R r h ∧ r_sphere = Real.cbrt 95 :=
by
  intro h R r
  assume ⟨h_eq, R_eq, r_eq⟩
  use Real.cbrt 95
  split
  · have H₁ : conicalFrustumVolume R r h = (95 / 3) * Real.pi := sorry
    have H₂ : sphereVolume (Real.cbrt 95) = (95 / 3) * Real.pi := sorry
    rw [H₁, H₂]

  · sorry

end radius_of_sphere_with_same_volume_as_frustum_l303_303871


namespace smallest_Y_l303_303165

-- Define the necessary conditions
def is_digits_0_1 (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

def is_divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

-- Define the main problem statement
theorem smallest_Y (S Y : ℕ) (hS_pos : S > 0) (hS_digits : is_digits_0_1 S) (hS_div_15 : is_divisible_by_15 S) (hY : Y = S / 15) :
  Y = 74 :=
sorry

end smallest_Y_l303_303165


namespace base_height_is_two_inches_l303_303917

noncomputable def height_sculpture_feet : ℝ := 2 + (10 / 12)
noncomputable def combined_height_feet : ℝ := 3
noncomputable def base_height_feet : ℝ := combined_height_feet - height_sculpture_feet
noncomputable def base_height_inches : ℝ := base_height_feet * 12

theorem base_height_is_two_inches :
  base_height_inches = 2 := by
  sorry

end base_height_is_two_inches_l303_303917


namespace product_of_divisors_18_l303_303331

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l303_303331


namespace quadratic_inequality_range_a_l303_303647

theorem quadratic_inequality_range_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + 2 * a > 0) ↔ (0 < a ∧ a < 8) :=
by
  sorry

end quadratic_inequality_range_a_l303_303647


namespace vector_dot_cross_product_l303_303183
-- Import the necessary library for vectors and algebra operations

-- Define the vectors 'a', 'b', and 'c'
def a : ℝ × ℝ × ℝ := (-4, 9, 2)
def b : ℝ × ℝ × ℝ := (7, e, -1)
def c : ℝ × ℝ × ℝ := (-1, -1, 8)

-- Define the problem statement to prove equivalence
theorem vector_dot_cross_product (e : ℝ) :
  let ab := (a.1 - b.1, a.2 - b.2, a.3 - b.3)
  let bc := (b.1 - c.1, b.2 - c.2, b.3 - c.3)
  let ca := (c.1 - a.1, c.2 - a.2, c.3 - a.3)
  let cross_product := ((bc.2 * ca.3 - bc.3 * ca.2), (bc.3 * ca.1 - bc.1 * ca.3), (bc.1 * ca.2 - bc.2 * ca.1))
  let dot_product := ab.1 * cross_product.1 + ab.2 * cross_product.2 + ab.3 * cross_product.3
  in dot_product = 117 - 12 * e := sorry

end vector_dot_cross_product_l303_303183


namespace product_of_divisors_of_18_l303_303379

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303379


namespace find_point_P_on_CD_l303_303145

-- Define the points and their properties
variables {A B C D P : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace P]

-- Define conditions
def convex_quadrilateral (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] : Prop :=
  true -- Detailed definition of convex quadrilateral can be extended

def right_angle_at_C_and_D (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] : Prop :=
  ∠DAB = 90 ∧ ∠DBC = 90

-- Problem statement
theorem find_point_P_on_CD (A B C D P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace P]
  (h1 : convex_quadrilateral A B C D)
  (h2 : right_angle_at_C_and_D A B C D) :
  ∃ P : Type, P ∈ ['CD] ∧ ∠APD = 2 * ∠BPC := sorry

end find_point_P_on_CD_l303_303145


namespace product_of_divisors_of_18_l303_303333

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l303_303333


namespace angle_between_cube_diagonals_l303_303947

def cube_vertices : list (ℝ × ℝ × ℝ) :=
  [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)]

def diag1 : ℝ × ℝ × ℝ := (1, 1, 0)
def diag2 : ℝ × ℝ × ℝ := (1, 0, 1)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

noncomputable def angle_between_vectors (v w : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos (dot_product v w / (magnitude v * magnitude w))

theorem angle_between_cube_diagonals :
  angle_between_vectors diag1 diag2 = real.pi / 3 :=
sorry

end angle_between_cube_diagonals_l303_303947


namespace perfect_square_trinomial_m_l303_303112

theorem perfect_square_trinomial_m (m : ℤ) (x : ℤ) : (∃ a : ℤ, x^2 - mx + 16 = (x - a)^2) ↔ (m = 8 ∨ m = -8) :=
by sorry

end perfect_square_trinomial_m_l303_303112


namespace fraction_termite_ridden_not_collapsing_l303_303456

theorem fraction_termite_ridden_not_collapsing (total_homes : ℕ) :
  (1 / 3) * (total_homes : ℚ) * (1 - (5 / 8)) = 1 / 8 * total_homes :=
by
  -- Using the given conditions and doing necessary calculations
  have h1 : (1 : ℚ) / 3 = 1 / 3 := rfl
  have h2 : (5 : ℚ) / 8 = 5 / 8 := rfl
  have collapsed_homes := ((1 / 3) * (5 / 8) : ℚ)
  have non_collapsed_homes := (1 / 3 - (1 / 3) * (5 / 8) : ℚ)
  calc
    (non_collapsed_homes * (total_homes : ℚ) : ℚ)
        = (1 / 3 - (1 / 3) * (5 / 8)) : by
    apply sorry
    ... = 1 / 8 : by
    apply sorry


end fraction_termite_ridden_not_collapsing_l303_303456


namespace no_squares_sharing_two_vertices_with_ABC_l303_303166

theorem no_squares_sharing_two_vertices_with_ABC
  (A B C : Type)
  [plane_geometry A B C]  -- Assuming the existence of a plane geometry structure
  (isosceles_triangle_ABC : is_isosceles_triangle A B C)
  (eq_AB_AC : AB = AC)
  (neq_BC_AB : BC ≠ AB) :
  count_squares_sharing_two_vertices A B C = 0 := 
sorry

end no_squares_sharing_two_vertices_with_ABC_l303_303166


namespace infinite_points_in_region_l303_303012

theorem infinite_points_in_region : 
  ∀ x y : ℚ, 0 < x → 0 < y → x + 2 * y ≤ 6 → ¬(∃ n : ℕ, ∀ x y : ℚ, 0 < x → 0 < y → x + 2 * y ≤ 6 → sorry) :=
sorry

end infinite_points_in_region_l303_303012


namespace x_minus_y_equals_two_l303_303964

-- Definitions of vectors and points
variable {Point : Type}
variable {Vector : Type}
variable (B M C A : Point) (AB AC AM BM BC : Vector)

-- Definition of scalar multiplication and vector addition
variable {ℝ : Type} [Field ℝ]
variable [AddCommGroup Vector]
variable [Module ℝ Vector]
variable (x y : ℝ)

-- Given conditions
axiom BM_eq_half_BC : BM = - (1 / 2 : ℝ) • BC
axiom AM_eq_xAB_yAC : AM = x • AB + y • AC

-- Prove that x - y = 2
theorem x_minus_y_equals_two (h1 : BM_eq_half_BC) (h2 : AM_eq_xAB_yAC) :
    x - y = 2 := sorry

end x_minus_y_equals_two_l303_303964


namespace series_sum_2022_l303_303803

def alternating_series (n : ℕ) : ℤ :=
  if n % 2 = 0 then -(2 * (n/2)) else 2 * ((n + 1)/2)

noncomputable def series_sum (n : ℕ) : ℤ := 
  ∑ i in Finset.range (n + 1), alternating_series i

theorem series_sum_2022 : series_sum 1011 = 1012 := by
  sorry

end series_sum_2022_l303_303803


namespace acute_triangle_statements_l303_303704

variable (A B C : ℝ)

-- Conditions for acute triangle
def acute_triangle := A + B + C = π ∧ 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2

-- Statement A: If A > B, then sin A > sin B.
def statement_A := ∀ h : A > B, Real.sin A > Real.sin B

-- Statement B: If A = π / 3, then the range of values for B is (0, π / 2).
def statement_B := ∀ h : A = π / 3, 0 < B ∧ B < π / 2

-- Statement C: sin A + sin B > cos A + cos B
def statement_C := Real.sin A + Real.sin B > Real.cos A + Real.cos B

-- Statement D: tan B tan C > 1
def statement_D := Real.tan B * Real.tan C > 1

-- The theorem to prove
theorem acute_triangle_statements (h : acute_triangle A B C) :
  statement_A A B C ∧ ¬statement_B A B C ∧ statement_C A B C ∧ statement_D A B C :=
by sorry

end acute_triangle_statements_l303_303704


namespace values_of_n_f100_eq_16_l303_303036

def num_divisors (n : ℕ) : ℕ :=
  if n = 0 then 0 else (List.range (n + 1) ).filter (λ d, n % d = 0).length

def f₁ (n : ℕ) : ℕ := 3 * num_divisors n

def f (n : ℕ) : ℕ → ℕ
| 1 := f₁ n
| (k+2) := f₁ (f n (k+1))

theorem values_of_n_f100_eq_16 : (Finset.filter (λ n, f n 100 = 16) (Finset.range 101)).card = 15 := sorry

end values_of_n_f100_eq_16_l303_303036


namespace sqrt2_not_rational_l303_303210

theorem sqrt2_not_rational : ¬ ∃ (p q : ℕ), coprime p q ∧ q ≠ 0 ∧ (↑p / ↑q) = Real.sqrt 2 :=
by
  sorry

end sqrt2_not_rational_l303_303210


namespace min_value_a_l303_303125

theorem min_value_a (a : ℝ) : (∃ x ∈ Icc (1 : ℝ) 3, x^2 - 2 ≤ a) ↔ (a ≥ -1) :=
by
  sorry

end min_value_a_l303_303125


namespace smallest_number_groups_l303_303439

theorem smallest_number_groups (x : ℕ) (h₁ : x % 18 = 0) (h₂ : x % 45 = 0) : x = 90 :=
sorry

end smallest_number_groups_l303_303439


namespace cos_14_pi_over_3_l303_303941

theorem cos_14_pi_over_3 : Real.cos (14 * Real.pi / 3) = -1 / 2 :=
by 
  -- Proof is omitted according to the instructions
  sorry

end cos_14_pi_over_3_l303_303941


namespace area_of_triangle_BCD_l303_303987

theorem area_of_triangle_BCD :
  let A := {p : ℝ × ℝ | (p.1 + 1)^2 + p.2^2 = 1}
  let B := {p : ℝ × ℝ | p.1^2 + (p.2 - 2)^2 = 4}
  let C D : ℝ × ℝ := (0, 0) -- Placeholders for the intersection points
  let intersection_points := A ∩ B -- Intersection points of circles A and B
  let triangle_area := 1 / 2 * (δ x_intersection1 x_intersection2) * (δ y_intersection1 y_intersection2) -- Placeholder area calculation
  triangle_area = 8 * sqrt 5 / 5
by
  -- Proof would go here, inserting 'sorry' to indicate it is not needed
  sorry

end area_of_triangle_BCD_l303_303987


namespace triangle_BC_length_l303_303161

noncomputable def length_of_BC (ABC : Triangle) (incircle_radius : ℝ) (altitude_A_to_BC : ℝ) 
    (BD_squared_plus_CD_squared : ℝ) : ℝ :=
  if incircle_radius = 3 ∧ altitude_A_to_BC = 15 ∧ BD_squared_plus_CD_squared = 33 then
    3 * Real.sqrt 7
  else
    0 -- This value is arbitrary, as the conditions above are specific

theorem triangle_BC_length {ABC : Triangle}
    (incircle_radius : ℝ) (altitude_A_to_BC : ℝ) (BD_squared_plus_CD_squared : ℝ) :
    incircle_radius = 3 →
    altitude_A_to_BC = 15 →
    BD_squared_plus_CD_squared = 33 →
    length_of_BC ABC incircle_radius altitude_A_to_BC BD_squared_plus_CD_squared = 3 * Real.sqrt 7 :=
by intros; sorry

end triangle_BC_length_l303_303161


namespace side_of_beef_original_weight_l303_303886

theorem side_of_beef_original_weight :
  ∃ W : ℝ, (∃ (lost_weight_percentage : ℝ) (after_processing_weight : ℝ), 
  lost_weight_percentage = 0.35 ∧ after_processing_weight = 580 ∧ 
  W * (1 - lost_weight_percentage) = after_processing_weight) ∧ W = 892 :=
begin
  sorry
end

end side_of_beef_original_weight_l303_303886


namespace product_of_divisors_18_l303_303321

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l303_303321


namespace find_b_l303_303795

noncomputable def gcd_of_factorials_is_5040 (b : ℤ) : Prop :=
  Int.gcd ((b - 2)!.to_nat) (Int.gcd ((b + 1)!.to_nat) ((b + 4)!.to_nat)) = 5040

theorem find_b (b : ℤ) (h : gcd_of_factorials_is_5040 b) : b = 9 := by
  sorry

end find_b_l303_303795


namespace number_of_even_numbers_from_124467_is_240_l303_303773

theorem number_of_even_numbers_from_124467_is_240 :
  let digits := [1, 2, 4, 4, 6, 7] in
  ∃ ns : list (list ℕ),
    (∀ n ∈ ns, (n.length = 6) ∧ (∀ x, (x ∈ n -> x ∈ digits)) ∧ (n.last.getD 0 % 2 = 0)) ∧
    (ns.length = 240) :=
by
  sorry

end number_of_even_numbers_from_124467_is_240_l303_303773


namespace students_per_group_l303_303811

-- Define the conditions:
def total_students : ℕ := 120
def not_picked_students : ℕ := 22
def groups : ℕ := 14

-- Calculate the picked students:
def picked_students : ℕ := total_students - not_picked_students

-- Statement of the problem:
theorem students_per_group : picked_students / groups = 7 :=
  by sorry

end students_per_group_l303_303811


namespace stefan_more_vail_l303_303235

/-- Aiguo had 20 seashells --/
def a : ℕ := 20

/-- Vail had 5 less seashells than Aiguo --/
def v : ℕ := a - 5

/-- The total number of seashells of Stefan, Vail, and Aiguo is 66 --/
def total_seashells (s v a : ℕ) : Prop := s + v + a = 66

theorem stefan_more_vail (s v a : ℕ)
  (h_a : a = 20)
  (h_v : v = a - 5)
  (h_total : total_seashells s v a) :
  s - v = 16 :=
by {
  -- proofs would go here
  sorry
}

end stefan_more_vail_l303_303235


namespace largest_k_log_3_B_l303_303807

def tower_of_threes : ℕ → ℕ 
| 1 := 3
| (n + 1) := 3^(tower_of_threes n)

def B : ℕ := (tower_of_threes 4) ^ (tower_of_threes 4) ^ (tower_of_threes 4)

theorem largest_k_log_3_B : ∃ k : ℕ, k = 5 ∧ 
  (∃ x : ℝ, x = (fin.iterate (λ y, Real.log 3 y) k B)) :=
begin
  sorry -- proof is omitted as required
end

end largest_k_log_3_B_l303_303807


namespace has_zero_in_interval_l303_303170

def f (x : ℝ) : ℝ := 3 * x - x ^ 2

theorem has_zero_in_interval : ∃ x ∈ set.Icc (-1 : ℝ) (0 : ℝ), f x = 0 :=
sorry

end has_zero_in_interval_l303_303170


namespace planting_rate_l303_303570

theorem planting_rate (total_acres : ℕ) (days : ℕ) (initial_tractors : ℕ) (initial_days : ℕ) (additional_tractors : ℕ) (additional_days : ℕ) :
  total_acres = 1700 →
  days = 5 →
  initial_tractors = 2 →
  initial_days = 2 →
  additional_tractors = 7 →
  additional_days = 3 →
  (total_acres / ((initial_tractors * initial_days) + (additional_tractors * additional_days))) = 68 :=
by
  sorry

end planting_rate_l303_303570


namespace lambda_mu_sum_l303_303097

variables {ℝ : Type} [Nontrivial ℝ]

noncomputable theory

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, 3)
def c : ℝ × ℝ := (3, 4)
def lambda : ℝ := -1
def mu : ℝ := 2

theorem lambda_mu_sum : lambda + mu = 1 :=
by sorry

end lambda_mu_sum_l303_303097


namespace product_of_divisors_of_18_l303_303382

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303382


namespace sector_to_cone_radius_l303_303627

theorem sector_to_cone_radius :
  ∀ (r : ℝ) (θ : ℝ), r = 3 ∧ θ = 120 → (arc_length θ r = circumference 1) :=
begin
  sorry
end

end sector_to_cone_radius_l303_303627


namespace probability_PBC_area_l303_303725

-- Define the right triangle ABC
axiom A : Type
axiom B : A
axiom C : A
axiom P : A

-- Conditions
axiom AB : Real
axiom BC : Real
axiom AC : Real

-- Values of the sides of the triangle
def AB_val : Real := 6
def BC_val : Real := 8
def AC_val : Real := 10

-- Conditions translated into Lean
axiom AB_cond : AB = AB_val
axiom BC_cond : BC = BC_val
axiom AC_cond : AC = AC_val

-- QUESTION: Calculate the probability that the area of triangle PBC is less than one-third the area of triangle ABC
theorem probability_PBC_area : (probability (area_of_triangle P B C < (1 / 3) * (area_of_triangle A B C)) = 5 / 9)
  assume AB_cond BC_cond AC_cond 
  -- No proof required as per instructions
  sorry

end probability_PBC_area_l303_303725


namespace tangent_lines_through_point_and_circle_l303_303025

noncomputable theory

-- Definitions for conditions
def point : ℝ × ℝ := (4, 0)
def circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1

-- Statements for the tangent line equations
def line1 (x y : ℝ) : Prop := x = 4
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y = 12

-- The main theorem to prove
theorem tangent_lines_through_point_and_circle :
  (∃ x y : ℝ, point = (x, y) ∧ (circle x y → line1 x y)) ∨
  (∃ x y : ℝ, point = (x, y) ∧ (circle x y → line2 x y)) :=
by sorry

end tangent_lines_through_point_and_circle_l303_303025


namespace substitution_ways_mod_1000_l303_303134

def a0 : ℕ := 1
def a1 : ℕ := 14 * 11
def a2 : ℕ := a1 * (13 * 12)
def a3 : ℕ := a2 * (12 * 13)
def a4 : ℕ := a3 * (11 * 14)

def total_ways : ℕ := a0 + a1 + a2 + a3 + a4
def result : ℕ := total_ways % 1000

theorem substitution_ways_mod_1000 : result = -- replace this with the computed result
by
  -- Proof steps skipped
  sorry

end substitution_ways_mod_1000_l303_303134


namespace line_PQ_parallel_bases_l303_303249

variables (A B C D E F P Q M : Type) [ordered_ring A] 

-- Vertices of trapezoid ABCD
variables (AB CD AD BC : A) -- The bases AB and CD, and non-parallel sides AD and BC

-- Condition definitions
variables (trapezoid_ABCD : AB ∥ CD)
variables (diagonal_intersection : ∃ M, M ∈ (diagonal_AD ∩ diagonal_BC))
variables (line_e : ∃ e, e ∩ M)
variables (intersection_E : E ∈ (e ∩ AD) ∧ E ≠ A ∧ E ≠ D)
variables (intersection_F : F ∈ (e ∩ BC))
variables (intersecting_diagonals_ABFE : ∃ P, P ∈ (diagonal_AF ∩ diagonal_BE))
variables (intersecting_diagonals_CDEF : ∃ Q, Q ∈ (diagonal_CF ∩ diagonal_DE))

-- The statement to be proven
theorem line_PQ_parallel_bases :
  PQ ∥ AD ↔ PQ ∥ BC :=
sorry

end line_PQ_parallel_bases_l303_303249


namespace tangent_line_at_point_l303_303644

theorem tangent_line_at_point :
  let f := λ x : ℝ, x^3 - (1/2) * x^2 - 2 * x,
      f' := λ x : ℝ, 3 * x^2 - x - 2,
      x₀ := 2,
      y₀ := f x₀,
      m := f' x₀ in
  y₀ = 2 ∧ m = 8 →
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) → (y = 8 * x - 14) :=
by
  intros f f' x₀ y₀ m h1 x y h2
  calc
    y = 8 * (x - 2) + 2 : by rw [h2, h1.2]
    ... = 8 * x - 16 + 2 : by ring
    ... = 8 * x - 14 : by ring

end tangent_line_at_point_l303_303644


namespace mean_of_solutions_of_cubic_l303_303578

noncomputable def mean_of_solutions (f : ℝ → ℝ) (roots : List ℝ) :=
  roots.sum / roots.length

theorem mean_of_solutions_of_cubic : 
  mean_of_solutions (λ x => x^3 + 5 * x^2 - 2 * x) [0, (-5 + Math.sqrt 29) / 2, (-5 - Math.sqrt 29) / 2] = -5 / 3 :=
by
  sorry

end mean_of_solutions_of_cubic_l303_303578


namespace product_of_divisors_of_18_l303_303365

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303365


namespace find_N_on_focal_radius_l303_303005

variables {N_1 N_2 F M : Point} {l : ℝ} (directrix : Line)

-- Defining points and properties specific to the problem
def is_on_focal_radius (N F M : Point) : Prop :=
  collinear N F M

def distance (A B : Point) : ℝ := sorry -- Define the Euclidean distance between points A and B.

def distance_to_directrix (N : Point) (directrix : Line) : ℝ := sorry -- Define the perpendicular distance from point N to the directrix.

-- Statement capturing the essence of the given problem
theorem find_N_on_focal_radius (h1 : is_on_focal_radius N_1 F M)
    (h2 : is_on_focal_radius N_2 F M)
    (h3 : abs (distance N_1 F - distance_to_directrix N_1 directrix) = l)
    (h4 : abs (distance N_2 F - distance_to_directrix N_2 directrix) = l) :
    ∃ (N_1 N_2 : Point), N_1 ≠ N_2 :=
sorry

end find_N_on_focal_radius_l303_303005


namespace modular_inverse_solution_l303_303173

theorem modular_inverse_solution (m : ℤ) (h1 : 0 ≤ m) (h2 : m < 31) (h3 : 4 * m ≡ 1 [MOD 31]) : (3 ^ m) ^ 4 - 3 ≡ 29 [MOD 31] := 
sorry

end modular_inverse_solution_l303_303173


namespace B_days_to_complete_work_l303_303463

theorem B_days_to_complete_work (A_days : ℕ) (efficiency_less_percent : ℕ) 
  (hA : A_days = 12) (hB_efficiency : efficiency_less_percent = 20) :
  let A_work_rate := 1 / 12
  let B_work_rate := (1 - (20 / 100)) * A_work_rate
  let B_days := 1 / B_work_rate
  B_days = 15 :=
by
  sorry

end B_days_to_complete_work_l303_303463


namespace factorize_expression_l303_303940

theorem factorize_expression (x y : ℝ) :
  x^2 + 4y^2 - 4xy - 1 = (x - 2y + 1) * (x - 2y - 1) :=
by
  sorry

end factorize_expression_l303_303940


namespace simplest_quadratic_radical_l303_303847
  
theorem simplest_quadratic_radical (A B C D: ℝ) 
  (hA : A = Real.sqrt 0.1) 
  (hB : B = Real.sqrt (-2)) 
  (hC : C = 3 * Real.sqrt 2) 
  (hD : D = -Real.sqrt 20) : C = 3 * Real.sqrt 2 :=
by
  have h1 : ∀ (x : ℝ), Real.sqrt x = Real.sqrt x := sorry
  sorry

end simplest_quadratic_radical_l303_303847


namespace product_of_divisors_18_l303_303325

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l303_303325


namespace product_of_divisors_18_l303_303300

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l303_303300


namespace measure_of_angle_l303_303788

theorem measure_of_angle (x : ℝ) (h1 : 90 = x + (3 * x + 10)) : x = 20 :=
by
  sorry

end measure_of_angle_l303_303788


namespace range_of_m_l303_303080

theorem range_of_m {x m : ℝ} (h : ∀ x, x^2 - 2*x + 2*m - 1 ≥ 0) : m ≥ 1 :=
sorry

end range_of_m_l303_303080


namespace line_intersects_circle_l303_303013

-- Define the circle equation in standard form
def circle (x y : ℝ) := (x + 2)^2 + (y - 1)^2 - 3 = 0

-- Define the line equation
def line (x y k : ℝ) := x - k * y + 1 = 0

-- Define the point inside the circle
def point_in_circle (x y : ℝ) := circle x y < 0

-- Prove that the line intersects the circle
theorem line_intersects_circle (k : ℝ) : ∃ x y : ℝ, circle x y = 0 ∧ line x y k = 0 :=
by
  sorry

end line_intersects_circle_l303_303013


namespace administrative_staff_drawn_in_stratified_sampling_l303_303766

theorem administrative_staff_drawn_in_stratified_sampling
  (total_staff : ℕ)
  (full_time_teachers : ℕ)
  (administrative_staff : ℕ)
  (logistics_personnel : ℕ)
  (sample_size : ℕ)
  (h_total : total_staff = 320)
  (h_teachers : full_time_teachers = 248)
  (h_admin : administrative_staff = 48)
  (h_logistics : logistics_personnel = 24)
  (h_sample : sample_size = 40)
  : (administrative_staff * (sample_size / total_staff) = 6) :=
by
  -- mathematical proof goes here
  sorry

end administrative_staff_drawn_in_stratified_sampling_l303_303766


namespace cost_per_litre_mixed_fruit_l303_303255

noncomputable def cost_superfruit : ℝ := 1399.45
noncomputable def cost_acai : ℝ := 3104.35
noncomputable def litres_mixed_fruit : ℕ := 33
noncomputable def litres_acai : ℕ := 22

theorem cost_per_litre_mixed_fruit :
  let cost_mixed_fruit := 256.79 in
  33 * cost_mixed_fruit + 22 * cost_acai = 
  (33 + 22) * cost_superfruit :=
by
  sorry

end cost_per_litre_mixed_fruit_l303_303255


namespace product_of_divisors_of_18_is_5832_l303_303404

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l303_303404


namespace exists_six_digit_number_l303_303551

theorem exists_six_digit_number : ∃ (n : ℕ), 100000 ≤ n ∧ n < 1000000 ∧ (∃ (x y : ℕ), n = 1000 * x + y ∧ 0 ≤ x ∧ x < 1000 ∧ 0 ≤ y ∧ y < 1000 ∧ 6 * n = 1000 * y + x) :=
by
  sorry

end exists_six_digit_number_l303_303551


namespace product_of_divisors_of_18_is_5832_l303_303410

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l303_303410


namespace find_smallest_beta_l303_303736

noncomputable def smallest_angle (a b c : V3) (beta : ℝ) : Prop :=
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥c∥ = 1 ∧
  (a ⬝ b) = Real.cos beta ∧
  (c ⬝ (a × b)) = Real.cos beta ∧
  (b ⬝ (c × a)) = 1 / 3

theorem find_smallest_beta (a b c : V3) (beta : ℝ) (cond : smallest_angle a b c beta) : beta ≈ Real.to_degrees (1/2 * Real.arcsin (2 / 3)) :=
by
  sorry

end find_smallest_beta_l303_303736


namespace bottles_sold_from_wed_to_sun_l303_303927

variable (beginning_inventory : ℕ)
variable (sold_monday : ℕ)
variable (sold_tuesday : ℕ)
variable (delivery_saturday : ℕ)
variable (end_inventory : ℕ)
variable (sold_wed_to_sun : ℕ)

def total_sold_monday_tuesday : ℕ := sold_monday + sold_tuesday

def remaining_inventory_after_tuesday : ℕ := beginning_inventory - total_sold_monday_tuesday

def total_inventory_after_delivery : ℕ := remaining_inventory_after_tuesday + delivery_saturday

def total_sold_wed_to_sun : ℕ := total_inventory_after_delivery - end_inventory

theorem bottles_sold_from_wed_to_sun :
    beginning_inventory = 4500 ->
    sold_monday = 2445 ->
    sold_tuesday = 900 ->
    delivery_saturday = 650 ->
    end_inventory = 1555 ->
    total_sold_wed_to_sun = 250 := by
  intros h1 h2 h3 h4 h5
  simp [total_sold_wed_to_sun, total_inventory_after_delivery, remaining_inventory_after_tuesday, total_sold_monday_tuesday]
  rw [h1, h2, h3, h4, h5]
  rfl

end bottles_sold_from_wed_to_sun_l303_303927


namespace lambda_value_l303_303094

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem lambda_value (a b : V) (λ : ℝ) (h₀ : ¬ (∀ k : ℝ, b = k • a)) (h₁ : ∃ k : ℝ, λ • a + b = k • (a - 2 • b)) : 
  λ = -1 / 2 := 
sorry

end lambda_value_l303_303094


namespace maximum_sum_y_squared_theorem_l303_303759

noncomputable def maximum_sum_y_squared (n : ℕ) : ℝ :=
  let x : ℝ → ℝ := λ x, if x % 1 > 0.5 then 1 - x % 1 else x % 1
  let y : ℝ → ℝ := λ x, x * x (x x)
  (Finset.sum (Finset.range n) (λ i, y (i.to_real))) / 4

theorem maximum_sum_y_squared_theorem (n : ℕ) (x : ℕ → ℝ) 
  (hx_nonneg : ∀ i, 0 ≤ x i) (hx_sum : (Finset.sum (Finset.range n) (λ i, x i.to_real)) = n) : 
  ∑ i in Finset.range n, (x i.to_real) * (if (x i.to_real) % 1 > 0.5 then 1 - (x i.to_real % 1) else x i.to_real % 1) ^ 2 = (n^2 - n + 0.5) / 4 :=
sorry

end maximum_sum_y_squared_theorem_l303_303759


namespace plastic_bottle_recycling_l303_303585

noncomputable def total_new_bottles (initial_bottles : ℕ) : ℕ :=
  if initial_bottles < 3 then
    0
  else
    let rec recycle (bottles : ℕ) (acc : ℕ) : ℕ :=
      if bottles < 3 then
        acc
      else
        let new_bottles := bottles / 5
        recycle (new_bottles + bottles % 5) (acc + new_bottles)
    recycle initial_bottles 0

theorem plastic_bottle_recycling (initial_bottles : ℕ) (h : initial_bottles = 625) :
  total_new_bottles initial_bottles = 156 := by
  rw [h]
  unfold total_new_bottles
  sorry

end plastic_bottle_recycling_l303_303585


namespace perimeter_of_triangle_ABF1_l303_303747

theorem perimeter_of_triangle_ABF1 :
  ∀ (x y : ℝ)
    (ellipse : x / 12 + y / 16 = 1)
    (F1 F2 A B : ℝ × ℝ)
    (line_through_F2 : ∃ k : ℝ, k ≠ 0 ∧ line_eq : (∀ t : ℝ, B = (F2.1 + k * t, F2.2 + (1 / k) * t) ∨ A = (F2.1 + k * t, F2.2 + (1 / k) * t)))
    (A_on_ellipse : ellipse A.1 A.2)
    (B_on_ellipse : ellipse B.1 B.2)
    (major_axis_length : 2 * 4 = 8),
    perimeter_of_triangle := 
      dist A F1 + dist A B + dist B F1 = 16
  :=
  sorry

end perimeter_of_triangle_ABF1_l303_303747


namespace chosen_numbers_divisibility_l303_303043

theorem chosen_numbers_divisibility (n : ℕ) (chosen : Finset ℕ) (h_chosen : chosen.card = n + 1) (h_range : ∀ x ∈ chosen, x ≤ 2 * n) :
  ∃ a b ∈ chosen, a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
by
  sorry

end chosen_numbers_divisibility_l303_303043


namespace probability_genuine_given_equal_weight_l303_303041

noncomputable def num_genuine : ℕ := 16
noncomputable def num_counterfeit : ℕ := 4
noncomputable def total_coins : ℕ := num_genuine + num_counterfeit
noncomputable def event_A : Event := {ω | all_four_selected_are_genuine ω}
noncomputable def event_B : Event := {ω | combined_weight_pairs_equal ω}

axiom coin_events_proba (A B : Event) : 
  (P(A ∩ B) = 1092 / 2907) ∧ (P(B) = 2907 / 5814)

theorem probability_genuine_given_equal_weight : 
  @Probability.event (coin ω) event_A ∩ event_B * (coin ω) event_B :=
begin
  rw [conditional_probability_def],
  rw [coin_events_proba],
  simp,
  norm_num [1092 / 2907, 2907 / 5814];
  sorry,
end

end probability_genuine_given_equal_weight_l303_303041


namespace cube_edge_length_l303_303270

theorem cube_edge_length (sum_of_edges : ℕ) (num_edges : ℕ) (h : sum_of_edges = 144) (num_edges_h : num_edges = 12) :
  sum_of_edges / num_edges = 12 :=
by
  -- The proof is skipped.
  sorry

end cube_edge_length_l303_303270


namespace incircle_radius_of_isosceles_right_triangle_l303_303826

theorem incircle_radius_of_isosceles_right_triangle :
  ∀ (DF : ℝ), DF = 12 → ∀ (∠D : ℝ), ∠D = 45 → ∃ r : ℝ, r = 12 - 6 * Real.sqrt 2 :=
by
  intros DF hDF ∠D hD
  use 12 - 6 * Real.sqrt 2
  sorry

end incircle_radius_of_isosceles_right_triangle_l303_303826


namespace binomial_sum_mod_1000_l303_303516

open BigOperators

theorem binomial_sum_mod_1000 :
  ((∑ k in finset.range 503 \ finset.range 3, nat.choose 2011 (4 * k)) % 1000) = 49 := 
sorry

end binomial_sum_mod_1000_l303_303516


namespace constant_term_max_term_l303_303634

theorem constant_term_max_term (x : ℝ) (n : ℕ) (h_n : n = 5) 
  (h_coefficient_max : ∀ k, (binomial 10 k) = (binomial 10 5 → k = 10 - k = 4)) :
  let expression := (sqrt x + (1 / (3 * x))) ^ (2 * n) in
  constant_term expression = 210 := 
by
  sorry

end constant_term_max_term_l303_303634


namespace repeating_decimal_subtraction_l303_303440

noncomputable def x := (0.246 : Real)
noncomputable def y := (0.135 : Real)
noncomputable def z := (0.579 : Real)

theorem repeating_decimal_subtraction :
  x - y - z = (-156 : ℚ) / 333 :=
by
  sorry

end repeating_decimal_subtraction_l303_303440


namespace variance_linear_transformation_variance_transformed_variable_l303_303073

variable (X : Type) [MeasureTheory.MeasureSpace X]

def variance (f : X → ℝ) : ℝ := sorry  -- Assume a definition for variance

variable (D : (X → ℝ) → ℝ)
hypothesis (D_eq_1 : D id = 1)

theorem variance_linear_transformation (a b : ℝ) (f : X → ℝ):
    D f = D id → D (fun x => a * f x + b) = a^2 * D f :=
sorry

theorem variance_transformed_variable :
    D id = 1 → D (fun x => 2 * id x + 1) = 4 :=
by
  intro D_eq_1
  apply variance_linear_transformation
  exact D_eq_1
  rfl

end variance_linear_transformation_variance_transformed_variable_l303_303073


namespace tan_double_angle_eq_l303_303997

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the tangent function in terms of point coordinates
def tan_alpha (P : ℝ × ℝ) : ℝ := P.2 / P.1

-- Theorem to prove the double angle formula for tangent given the conditions
theorem tan_double_angle_eq (α : ℝ) (P : ℝ × ℝ) (h : P = (2, 1)) :
  let tan_α := tan_alpha P
  in tan (2 * α) = 2 * tan_α / (1 - (tan_α ^ 2)) :=
by
  have h1 : tan_α = 1 / 2,
  { rw [h, tan_alpha], simp, }
  have h2 : tan (2 * α) = 4 / 3,
  { rw [h1], simp, }
  sorry

end tan_double_angle_eq_l303_303997


namespace max_volume_tetrahedron_l303_303151

theorem max_volume_tetrahedron
  (A B C D : Point)
  (h1 : ∠ A B C = 90°)
  (h2 : ∠ C D B = 90°)
  (h3 : dist B C = 2)
  (h4 : skew_angle A B C D = 60°)
  (h5 : circumradius A B C D = sqrt 5) :
  ∃ V, V = 2 * sqrt 3 ∧ maximal_volume A B C D V := sorry

end max_volume_tetrahedron_l303_303151


namespace sqrt_expression_value_l303_303842

theorem sqrt_expression_value :
  (real.sqrt (16 - 8 * real.sqrt 3) + real.sqrt (16 + 8 * real.sqrt 3))^3 = 24 := by
  sorry

end sqrt_expression_value_l303_303842


namespace jason_picked_2_pears_l303_303729

-- Defining the problem conditions
def total_pears : ℕ := 5
def keith_pears : ℕ := 3

-- Statement of the question: How many pears did Jason pick?
def jason_pears : ℕ := total_pears - keith_pears

-- Lean statement to prove the question
theorem jason_picked_2_pears : jason_pears = 2 := 
by 
  have h : jason_pears = 5 - 3,
  from rfl,
  exact h.trans (by rfl)

end jason_picked_2_pears_l303_303729


namespace angle_bisector_AD_l303_303182

/-- Let Γ₁ and Γ₂ be two circles such that Γ₁ is internally tangent to Γ₂ at A.
    Let D be a point on Γ₁. The tangent to Γ₁ at D intersects Γ₂ at points M and N.
    Show that AD is the angle bisector of ∠MAN. -/
theorem angle_bisector_AD (Γ₁ Γ₂ : Circle) (A D M N : Point) 
  (h_tangent : internally_tangent Γ₁ Γ₂ A) 
  (h_D_on_Γ₁ : on_circle D Γ₁)
  (h_tangent_D : tangent_at_point Γ₁ D M N Γ₂) :
  angle_bisector (segment AD) (angle MAN) :=
sorry

end angle_bisector_AD_l303_303182


namespace find_x_positive_sqrt_prod_eq_24_l303_303678

theorem find_x_positive_sqrt_prod_eq_24 (x : ℝ) (h : 0 < x) :
  (sqrt (12 * x) * sqrt (18 * x) * sqrt (6 * x) * sqrt (24 * x) = 24) ↔ x = sqrt (3 / 22) := 
sorry

end find_x_positive_sqrt_prod_eq_24_l303_303678


namespace arithmetic_sequence_sum_l303_303502

theorem arithmetic_sequence_sum :
  (a d l n S : ℕ) (h_a : a = 2) (h_d : d = 5) (h_l : l = 102) (h_n : n = 21) :
  S = (n / 2) * (a + l) ↔ S = 1092 :=
by
  sorry

end arithmetic_sequence_sum_l303_303502


namespace arithmetic_series_first_term_l303_303957

theorem arithmetic_series_first_term :
  ∃ (a d : ℝ), (25 * (2 * a + 49 * d) = 200) ∧ (25 * (2 * a + 149 * d) = 2700) ∧ (a = -20.5) :=
by
  sorry

end arithmetic_series_first_term_l303_303957


namespace product_of_divisors_18_l303_303303

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l303_303303


namespace product_of_divisors_of_18_l303_303308

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303308


namespace max_3x_4y_eq_73_l303_303621

theorem max_3x_4y_eq_73 :
  (∀ x y : ℝ, x ^ 2 + y ^ 2 = 14 * x + 6 * y + 6 → 3 * x + 4 * y ≤ 73) ∧
  (∃ x y : ℝ, x ^ 2 + y ^ 2 = 14 * x + 6 * y + 6 ∧ 3 * x + 4 * y = 73) :=
by sorry

end max_3x_4y_eq_73_l303_303621


namespace max_value_of_f_l303_303004

noncomputable def f (x : ℝ) : ℝ := 8 * sin x - tan x

theorem max_value_of_f :
  ∃ x ∈ Ioo 0 (π / 2), (∀ y ∈ Ioo 0 (π / 2), x ≠ y → f x > f y) ∧ f x = 3 * real.sqrt 3 :=
sorry

end max_value_of_f_l303_303004


namespace alice_has_ball_after_two_turns_l303_303491

def prob_alice_keeps_ball : ℚ := (2/3 * 1/2) + (1/3 * 1/3)

theorem alice_has_ball_after_two_turns :
  prob_alice_keeps_ball = 4 / 9 :=
by
  -- This line is just a placeholder for the actual proof
  sorry

end alice_has_ball_after_two_turns_l303_303491


namespace road_trip_time_l303_303159

theorem road_trip_time : 
  let distance_jenna := 200
  let speed_jenna := 50
  let time_jenna := distance_jenna / speed_jenna

  let distance_friend := 100
  let speed_friend := 20
  let time_friend := distance_friend / speed_friend

  let breaks := 2
  let break_time := 30 / 60 -- convert minutes to hours
  let total_break_time := breaks * break_time

  let total_time := time_jenna + time_friend + total_break_time 

  total_time = 10 := 
by
  unfold distance_jenna speed_jenna time_jenna distance_friend speed_friend time_friend breaks break_time total_break_time total_time
  -- Here, you can show step-by-step calculations as demonstrated above.
  sorry

end road_trip_time_l303_303159


namespace stone_volume_l303_303272

-- Statement: Given the conditions of the problem, prove the volume of the stone is 1120 cm³.
theorem stone_volume (width length height initial_water final_water : ℝ)
  (h₁ : width = 16)
  (h₂ : length = 14)
  (h₃ : initial_water = 4)
  (h₄ : final_water = 9) :
  (width * length * (final_water - initial_water) = 1120) :=
by
  -- The problem conditions entered as hypotheses
  rw [h₁, h₂, h₃, h₄] -- Replaces variables with their given values
  -- Computation of the volume
  have step1 : final_water - initial_water = 5 := by norm_num
  rw step1
  have step2 : 16 * 14 * 5 = 1120 := by norm_num
  exact step2

end stone_volume_l303_303272


namespace jackson_star_fish_count_l303_303157

def total_starfish_per_spiral_shell (hermit_crabs : ℕ) (shells_per_crab : ℕ) (total_souvenirs : ℕ) : ℕ :=
  (total_souvenirs - (hermit_crabs + hermit_crabs * shells_per_crab)) / (hermit_crabs * shells_per_crab)

theorem jackson_star_fish_count :
  total_starfish_per_spiral_shell 45 3 450 = 2 :=
by
  -- The proof will be filled in here
  sorry

end jackson_star_fish_count_l303_303157


namespace weight_of_quarter_l303_303916

variable (w : ℝ) -- weight of each quarter in ounces

-- Define the conditions as Lean hypotheses
def melted_value (w : ℝ) : ℝ := 100 * w -- value of each quarter when melted down
def store_value : ℝ := 0.25 -- value of each quarter when spent in a store

theorem weight_of_quarter :
  (melted_value w = 80 * store_value) → w = 0.2 :=
by
  intro h
  have h_eq : 100 * w = 100 * 0.2 := by
    linarith
  show w = 0.2 from (mul_right_inj' (ne_of_gt (by norm_num : 100 > 0))).mp h_eq
  sorry

end weight_of_quarter_l303_303916


namespace product_of_divisors_of_18_l303_303288

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l303_303288


namespace product_of_divisors_of_18_l303_303314

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303314


namespace mans_rate_in_still_water_l303_303881

theorem mans_rate_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 22) 
  (h2 : speed_against_stream = 10) : 
  (speed_with_stream + speed_against_stream) / 2 = 16 := 
by {
  rw [h1, h2],
  norm_num,
}

end mans_rate_in_still_water_l303_303881


namespace group_C_questions_l303_303706

theorem group_C_questions (a b c : ℕ) (total_questions : ℕ) (h1 : a + b + c = 100)
  (h2 : b = 23)
  (h3 : a ≥ (6 * (a + 2 * b + 3 * c)) / 10)
  (h4 : 2 * b ≤ (25 * (a + 2 * b + 3 * c)) / 100)
  (h5 : 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c) :
  c = 1 :=
sorry

end group_C_questions_l303_303706


namespace product_of_divisors_of_18_l303_303309

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303309


namespace medians_divide_EF_into_three_equal_parts_l303_303274

variables {A B C P E F K L O : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited E] [Inhabited F] [Inhabited K] [Inhabited L] [Inhabited O]

/-- Given a triangle ABC, point P on AC, and lines through P parallel to the medians AK and CL intersect BC and AB at points E and F respectively.
Also given that O is the centroid of the triangle, proving that the medians AK and CL divide the segment EF into three equal parts. -/
theorem medians_divide_EF_into_three_equal_parts : 
  ∀ (A B C P K L E F O : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited K] [Inhabited L] [Inhabited E] [Inhabited F] [Inhabited O],
    -- Conditions
    (∃ (P : Type), P ∈ AC) →
    (parallel_to_median (line_through P) AK ∧ parallel_to_median (line_through P) CL) →
    (intersection (line_through P parallel_to_median (line_through P) K) BC = E) →
    (intersection (line_through P parallel_to_median (line_through P) L) AB = F) →
    (centroid_of_triangle (triangle A B C) = O) →
  -- Conclusion
    divides_into_three_equal_parts AK CL EF :=
sorry

end medians_divide_EF_into_three_equal_parts_l303_303274


namespace number_line_4_units_away_l303_303952

theorem number_line_4_units_away (x : ℝ) : |x + 3.2| = 4 ↔ (x = 0.8 ∨ x = -7.2) :=
by
  sorry

end number_line_4_units_away_l303_303952


namespace f_eq_f_inv_l303_303546

noncomputable def f (x : ℝ) : ℝ := 3 * x - 7

noncomputable def f_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem f_eq_f_inv (x : ℝ) : f x = f_inv x ↔ x = 3.5 := by
  sorry

end f_eq_f_inv_l303_303546


namespace planting_rate_l303_303571

theorem planting_rate (total_acres : ℕ) (days : ℕ) (initial_tractors : ℕ) (initial_days : ℕ) (additional_tractors : ℕ) (additional_days : ℕ) :
  total_acres = 1700 →
  days = 5 →
  initial_tractors = 2 →
  initial_days = 2 →
  additional_tractors = 7 →
  additional_days = 3 →
  (total_acres / ((initial_tractors * initial_days) + (additional_tractors * additional_days))) = 68 :=
by
  sorry

end planting_rate_l303_303571


namespace product_of_divisors_of_18_is_5832_l303_303399

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l303_303399


namespace product_of_divisors_of_18_l303_303319

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303319


namespace magnitude_of_sum_find_k_l303_303098

-- Definitions
def vector_a : ℝ × ℝ := (2, 0)
def vector_b : ℝ × ℝ := (1, 4)

-- Tasks

-- (I) Prove |vector_a + vector_b| = 5
theorem magnitude_of_sum : (sqrt ((vector_a.1 + vector_b.1)^2 + (vector_a.2 + vector_b.2)^2)) = 5 := sorry

-- (II) Prove k = 1/2 if k vector_a + vector_b is parallel to vector_a + 2 vector_b
theorem find_k (k : ℝ) : (∀ k, 
  let vector_k := (2 * k + 1, 4) in 
  let vector_new_b := (4, 8) in 
  8 * vector_k.1 - 4 * vector_k.2 = 0 → k = 1/2) := sorry

end magnitude_of_sum_find_k_l303_303098


namespace product_of_divisors_of_18_l303_303435

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303435


namespace find_x_for_opposite_directions_l303_303663

-- Define the vectors and the opposite direction condition
def vector_a (x : ℝ) : ℝ × ℝ := (1, -x)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -16)

-- Define the condition that vectors are in opposite directions
def opp_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (-k) • b

-- The main theorem statement
theorem find_x_for_opposite_directions : ∃ x : ℝ, opp_directions (vector_a x) (vector_b x) ∧ x = -5 := 
sorry

end find_x_for_opposite_directions_l303_303663


namespace calculate_expression_l303_303908

theorem calculate_expression : (Real.sqrt 3)^0 + 2^(-1) + Real.sqrt 2 * Real.cos (Float.pi / 4) - abs (-1 / 2) = 2 := 
by
  sorry

end calculate_expression_l303_303908


namespace product_of_divisors_of_18_l303_303426

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303426


namespace factorize_expression_l303_303017

theorem factorize_expression (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3)^2 :=
by
  sorry

end factorize_expression_l303_303017


namespace complex_fraction_simplification_l303_303231

theorem complex_fraction_simplification (i : ℂ) (hi : i^2 = -1) : 
  ((2 - i) / (1 + 4 * i)) = (-2 / 17 - (9 / 17) * i) :=
  sorry

end complex_fraction_simplification_l303_303231


namespace minimum_value_expression_l303_303950

theorem minimum_value_expression (x y : ℝ) : ∃ (m : ℝ), ∀ x y : ℝ, x^2 + 3 * x * y + y^2 ≥ m ∧ m = 0 :=
by
  use 0
  sorry

end minimum_value_expression_l303_303950


namespace product_of_divisors_of_18_l303_303375

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303375


namespace angle_NMH_is_90_l303_303732

noncomputable theory
open_locale classical

variables (P Q R S A B C D T M N K H : Type) 
[quadPQRS : quadrilateral_with_incircle P Q R S]
[touches : incircle_touches_sides PQRS A B C D]
[RQTRP : ∃ T M, 
  intersects_line RP BA T /\ 
  intersects_line RP BC M]
[N_point : ∃ N,
  on_line TB N /\
  bisects_angle T M B N]
[intersection : ∃ K,
  intersection CN TM K]
[second_intersection : ∃ H,
  intersection BK CD H]

def angle_NMH_eq_90 : Prop := ∠ NM H = 90

theorem angle_NMH_is_90 : angle_NMH_eq_90 P Q R S A B C D T M N K H :=
sorry

end angle_NMH_is_90_l303_303732


namespace find_n_for_modulus_l303_303039

theorem find_n_for_modulus :
  ∃ (n : ℝ), 0 < n ∧ |5 + complex.I * n| = 5 * real.sqrt 13 ∧ n = 10 * real.sqrt 3 :=
begin
  -- sorry is used because the proof steps are not necessary
  sorry
end

end find_n_for_modulus_l303_303039


namespace length_EM_l303_303618

-- Given conditions
variables 
  (F : Point)
  (parabola : Line → Prop)
  (l : Line → Prop)
  (A B : Point)
  (M E : Point)
  (x1 x2 y1 y2 x0 y0 : ℝ)

-- Assumptions
axiom focus_F' : F = (1, 0) -- Focus of the parabola y^2 = 4x
axiom parabola_eq' : parabola = λ p, p.y^2 = 4 * p.x
axiom line_l' : l = λ x y, y = k * x + c -- general line equation
axiom AB_intersects' : A ∈ parabola ∧ B ∈ parabola ∧ l = line(thru F)
axiom AB_length' : dist(A, B) = 6 -- |AB| = 6
axiom midpoint_bisector' : midpoint(A, B) = E -- E is midpoint of AB
axiom E_foot' : dist_perpendicular(E, AB) = M -- foot of the perpendicular from M to AB is E

-- Goal
theorem length_EM : dist(E, M) = sqrt(6) := 
by sorry

end length_EM_l303_303618


namespace larger_cube_volume_is_512_l303_303872

def original_cube_volume := 64 -- volume in cubic feet
def scale_factor := 2 -- the factor by which the dimensions are scaled

def side_length (volume : ℕ) : ℕ := volume^(1/3) -- Assuming we have a function to compute cube root

def larger_cube_volume (original_volume : ℕ) (scale_factor : ℕ) : ℕ :=
  let original_side_length := side_length original_volume
  let larger_side_length := scale_factor * original_side_length
  larger_side_length ^ 3

theorem larger_cube_volume_is_512 :
  larger_cube_volume original_cube_volume scale_factor = 512 :=
sorry

end larger_cube_volume_is_512_l303_303872


namespace sequence_behavior_l303_303215

theorem sequence_behavior (b : ℕ → ℕ) :
  (∀ n, b n = n) ∨ ∃ N, ∀ n, n ≥ N → b n = b N :=
sorry

end sequence_behavior_l303_303215


namespace product_of_divisors_of_18_l303_303338

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l303_303338


namespace sum_of_ratios_eq_four_l303_303089

theorem sum_of_ratios_eq_four 
  (A B C D E : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E]
  (BD DC AE EB : ℝ)
  (h1 : BD = 2 * DC)
  (h2 : AE = 2 * EB) : 
  (BD / DC) + (AE / EB) = 4 :=
  sorry

end sum_of_ratios_eq_four_l303_303089


namespace average_math_score_l303_303192

theorem average_math_score (scores : Fin 4 → ℕ) (other_avg : ℕ) (num_students : ℕ) (num_other_students : ℕ)
  (h1 : scores 0 = 90) (h2 : scores 1 = 85) (h3 : scores 2 = 88) (h4 : scores 3 = 80)
  (h5 : other_avg = 82) (h6 : num_students = 30) (h7 : num_other_students = 26) :
  (90 + 85 + 88 + 80 + 26 * 82) / 30 = 82.5 :=
by
  sorry

end average_math_score_l303_303192


namespace parallel_lines_direction_vector_l303_303835

theorem parallel_lines_direction_vector (k : ℝ) :
  (∃ c : ℝ, (5, -3) = (c * -2, c * k)) ↔ k = 6 / 5 :=
by sorry

end parallel_lines_direction_vector_l303_303835


namespace popsicle_melting_faster_l303_303466

theorem popsicle_melting_faster (t : ℕ) :
  ∀ (n : ℕ), if n = 6 then (2 ^ (n - 1)) * t = 32 * t else true :=
by
  intro n
  cases n
  case zero => exact true.intro
  case succ n =>
    cases n
    case zero => exact true.intro
    case succ n =>
      cases n
      case zero => exact true.intro
      case succ n =>
        cases n
        case zero => exact true.intro
        case succ n =>
          cases n
          case zero => exact true.intro
          case succ n =>
            cases n
            case zero => exact true.intro
            case succ n =>
              case zero => exact true.intro
              sorry

end popsicle_melting_faster_l303_303466


namespace income_calculation_l303_303926

variable (income tax : ℝ)

-- Define the given conditions
def tax_rate_first := 0.1
def tax_rate_excess := 0.2
def threshold := 40000.0
def total_tax := 8000.0

-- The main statement to prove
theorem income_calculation (h1 : income > threshold) 
                          (h2 : tax = total_tax) 
                          (h3 : tax = tax_rate_first * threshold + tax_rate_excess * (income - threshold)) : 
  income = 60000.0 :=
by
  sorry

end income_calculation_l303_303926


namespace binom_sum_mod_1000_l303_303522

theorem binom_sum_mod_1000 : 
  (∑ i in (finset.range 2012).filter (λ i, i % 4 = 0), nat.choose 2011 i) % 1000 = 15 :=
sorry

end binom_sum_mod_1000_l303_303522


namespace integral_with_binomial_expansion_l303_303147

theorem integral_with_binomial_expansion :
  let a := (- choose 5 3 : ℤ) in
  a = -10 →
  ∫ x in ((a : ℝ)..-1), 2 * x = -99 := by
  intros a ha
  have h : a = -10 := ha
  rw [h]
  norm_num
  simp
  ring
  sorry

end integral_with_binomial_expansion_l303_303147


namespace shorter_diagonal_length_trapezoid_l303_303821

theorem shorter_diagonal_length_trapezoid 
  (EF GH : ℝ) (a b : ℝ) (h1 : EF = 25) 
  (h2 : GH = 15) (h3 : a = 13) 
  (h4 : b = 17) (acuteE : acute_angle E)
  (acuteF : acute_angle F) : 
  (shorter_diagonal EF GH a b) = 7 := 
sorry

end shorter_diagonal_length_trapezoid_l303_303821


namespace product_of_divisors_of_18_l303_303371

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303371


namespace rebecca_groups_of_eggs_l303_303218

/-- Given:
  - Rebecca wants to split a collection of eggs into groups of 3.
  - Rebecca has 99 bananas, 9 eggs, and 27 marbles.
  Prove that the number of groups of eggs that can be created is 3.
-/
theorem rebecca_groups_of_eggs:
  ∀ (bananas eggs marbles : ℕ), bananas = 99 → eggs = 9 → marbles = 27 → (eggs / 3) = 3 := by
  intros bananas eggs marbles h_bananas h_eggs h_marbles
  rw [h_eggs]
  norm_num
  rw [Nat.div_eq_of_eq_mul_left]
  norm_num -- this confirms 3 * 3 = 9
  sorry

end rebecca_groups_of_eggs_l303_303218


namespace right_triangle_hypotenuse_length_l303_303699

theorem right_triangle_hypotenuse_length
  {X Y Z : Type} [RightAngledTriangle X Y Z] 
  (cosX : ∀ X Y Z : ℝ, cos X = 3 / 5) 
  (YZ_len : YZ = 25) 
  : XZ = 125 / 3 := by
  sorry

end right_triangle_hypotenuse_length_l303_303699


namespace triangle_ABC_BC_length_l303_303624

-- Define the function f(x)
def f (x : ℝ) : ℝ :=
  2 * sin (2 / 3 * x + π / 6) - 1

-- Define triangle ABC with given conditions
variables (A B C AB BC : ℝ)
hypothesis h1 : f C = 1
hypothesis h2 : AB = 2
hypothesis h3 : 2 * sin B ^ 2 = cos B + cos (A - C)
  
-- We need to prove BC = sqrt 5 - 1
theorem triangle_ABC_BC_length : BC = real.sqrt 5 - 1 :=
sorry

end triangle_ABC_BC_length_l303_303624


namespace product_of_divisors_of_18_l303_303378

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303378


namespace total_seeds_planted_l303_303767

theorem total_seeds_planted 
    (seeds_per_bed : ℕ) 
    (seeds_grow_per_bed : ℕ) 
    (total_flowers : ℕ) 
    (h1 : seeds_per_bed = 15) 
    (h2 : seeds_grow_per_bed = 60) 
    (h3 : total_flowers = 220) : 
    ∃ (total_seeds : ℕ), total_seeds = 85 := 
by
    sorry

end total_seeds_planted_l303_303767


namespace product_of_divisors_of_18_l303_303381

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303381


namespace sine_cos_suffices_sine_cos_necessary_l303_303107

theorem sine_cos_suffices
  (a b c : ℝ)
  (h : ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) :
  c > Real.sqrt (a^2 + b^2) :=
sorry

theorem sine_cos_necessary
  (a b c : ℝ)
  (h : c > Real.sqrt (a^2 + b^2)) :
  ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0 :=
sorry

end sine_cos_suffices_sine_cos_necessary_l303_303107


namespace ellen_total_legos_l303_303015

-- Conditions
def ellen_original_legos : ℝ := 2080.0
def ellen_winning_legos : ℝ := 17.0

-- Theorem statement
theorem ellen_total_legos : ellen_original_legos + ellen_winning_legos = 2097.0 :=
by
  -- The proof would go here, but we will use sorry to indicate it is skipped.
  sorry

end ellen_total_legos_l303_303015


namespace product_of_divisors_of_18_l303_303341

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l303_303341


namespace line_intersects_y_axis_at_0_6_l303_303902

theorem line_intersects_y_axis_at_0_6 : ∃ y : ℝ, 4 * y + 3 * (0 : ℝ) = 24 ∧ (0, y) = (0, 6) :=
by
  use 6
  simp
  sorry

end line_intersects_y_axis_at_0_6_l303_303902


namespace kittens_more_than_twice_puppies_l303_303815

-- Define the number of puppies
def num_puppies : ℕ := 32

-- Define the number of kittens
def num_kittens : ℕ := 78

-- Define the problem statement
theorem kittens_more_than_twice_puppies :
  num_kittens = 2 * num_puppies + 14 :=
by sorry

end kittens_more_than_twice_puppies_l303_303815


namespace eventB_not_random_l303_303449

-- Definitions for the conditions of the problem
def eventA : Prop := (∃ t, (sun_rises_in_east_at t) ∧ (rains_in_west_at t))
def eventB : Prop := ¬ cold_when_snows ∧ cold_when_melts
def eventC : Prop := ∃ t, rains_continuously_during_qingming t
def eventD : Prop := ∀ t, sunny_every_day t → plums_yellow t

-- Definition for random event and inevitable event
def is_random_event (e : Prop) : Prop := ¬ ∃ t, e = true ∨ e = false
def is_inevitable_event (e : Prop) : Prop := e = true
def is_impossible_event (e : Prop) : Prop := e = false

-- The proof statement
theorem eventB_not_random :
  ¬ is_random_event eventB :=
sorry

end eventB_not_random_l303_303449


namespace sum_b_eq_T_l303_303631

noncomputable def S (n : ℕ) : ℕ := 2^n + 1

noncomputable def a (n : ℕ) : ℕ := if n = 1 then 3 else 2^(n - 1)

noncomputable def b (n : ℕ) : ℕ := (a n)^2 + n

noncomputable def T (n : ℕ) : ℕ :=
  if n = 1 then 10 else (8 / 3 : ℚ) + (4^n / 3 : ℚ) + ((n^2 + n - 2) / 2 : ℚ)

theorem sum_b_eq_T (n : ℕ) : 
  ∑ i in finset.range n, b (i + 1) = T n := 
sorry

end sum_b_eq_T_l303_303631


namespace pet_store_dogs_count_l303_303884

def initial_dogs : ℕ := 2
def sunday_received_dogs : ℕ := 5
def sunday_sold_dogs : ℕ := 2
def monday_received_dogs : ℕ := 3
def monday_returned_dogs : ℕ := 1
def tuesday_received_dogs : ℕ := 4
def tuesday_sold_dogs : ℕ := 3

theorem pet_store_dogs_count :
  initial_dogs 
  + sunday_received_dogs - sunday_sold_dogs
  + monday_received_dogs + monday_returned_dogs
  + tuesday_received_dogs - tuesday_sold_dogs = 10 := 
sorry

end pet_store_dogs_count_l303_303884


namespace ratio_DC_over_BS_is_2_l303_303606

-- Definitions of geometric entities
variables {A B C D S : Type} 
variables [inst : DecidableEq A] [inst : DecidableEq B] [inst : DecidableEq C] [inst : DecidableEq D] [inst : DecidableEq S]

-- Assuming the angle measures and geometric properties provided in the conditions
variables (triangle : A B C)
variables (midpointS_of_AD : S) (midpoint_property : midpointS_of_AD = midpoint S (line_segment A D))
variables (angle_BAC : measure_angle A B C = 60)
variables (angle_SBA : measure_angle S B A = 30)

-- The main theorem statement
theorem ratio_DC_over_BS_is_2 (h : is_angle_bisector A B D) : (DC / BS) = 2 :=
sorry

end ratio_DC_over_BS_is_2_l303_303606


namespace probability_fourth_roll_six_l303_303918

theorem probability_fourth_roll_six
  (fair_die : ℕ → ℝ)
  (biased_die : ℕ → ℝ)
  (prob_fair_die_six : fair_die 6 = 1 / 6)
  (prob_biased_die_six : biased_die 6 = 3 / 4)
  (prob_biased_die_other : ∀ i, i ≠ 6 → biased_die i = 1 / 20)
  (first_three_sixes : ℕ → ℝ) :
  first_three_sixes 6 = 774 / 1292 := 
sorry

end probability_fourth_roll_six_l303_303918


namespace inverse_exists_l303_303026

noncomputable def f (x : ℝ) : ℝ := 7 * x^3 - 2 * x^2 + 5 * x - 9

theorem inverse_exists :
  ∃ x : ℝ, 7 * x^3 - 2 * x^2 + 5 * x - 5.5 = 0 :=
sorry

end inverse_exists_l303_303026


namespace calculate_expression_l303_303913

theorem calculate_expression :
  (Real.sqrt 3) ^ 0 + 2 ^ (-1 : ℤ) + Real.sqrt 2 * Real.cos (Real.pi / 4) - |(-1:ℝ) / 2| = 2 := 
by
  sorry

end calculate_expression_l303_303913


namespace Emily_walks_more_distance_than_Troy_l303_303829

theorem Emily_walks_more_distance_than_Troy (Troy_distance Emily_distance : ℕ) (days : ℕ) 
  (hTroy : Troy_distance = 75) (hEmily : Emily_distance = 98) (hDays : days = 5) : 
  ((Emily_distance * 2 - Troy_distance * 2) * days) = 230 :=
by
  sorry

end Emily_walks_more_distance_than_Troy_l303_303829


namespace length_segment_midpoints_diagonals_trapezoid_l303_303253

theorem length_segment_midpoints_diagonals_trapezoid
  (a b c d : ℝ)
  (h_side_lengths : (2 = a ∨ 2 = b ∨ 2 = c ∨ 2 = d) ∧ 
                    (10 = a ∨ 10 = b ∨ 10 = c ∨ 10 = d) ∧ 
                    (10 = a ∨ 10 = b ∨ 10 = c ∨ 10 = d) ∧ 
                    (20 = a ∨ 20 = b ∨ 20 = c ∨ 20 = d))
  (h_parallel_sides : (a = 20 ∧ b = 2) ∨ (a = 2 ∧ b = 20)) :
  (1/2) * |a - b| = 9 :=
by
  sorry

end length_segment_midpoints_diagonals_trapezoid_l303_303253


namespace inequality_abc_l303_303198

theorem inequality_abc (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : a^2 + b^2 + c^2 = 1) : 
  (ab / c) + (bc / a) + (ca / b) ≥ real.sqrt 3 :=
begin
  sorry
end

end inequality_abc_l303_303198


namespace product_of_all_positive_divisors_of_18_l303_303390

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l303_303390


namespace solution_set_x2_gt_1_solution_set_neg_x2_plus_2x_plus_3_gt_0_l303_303029

theorem solution_set_x2_gt_1 : 
  {x : ℝ | x^2 > 1} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
sorry

theorem solution_set_neg_x2_plus_2x_plus_3_gt_0 : 
  {x : ℝ | -x^2 + 2x + 3 > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end solution_set_x2_gt_1_solution_set_neg_x2_plus_2x_plus_3_gt_0_l303_303029


namespace unique_number_encoding_l303_303797

-- Defining participants' score ranges 
def score_range := {x : ℕ // x ≤ 5}

-- Defining total score
def total_score (s1 s2 s3 s4 s5 s6 : score_range) : ℕ := 
  s1.val + s2.val + s3.val + s4.val + s5.val + s6.val

-- Main statement to encode participant's scores into a unique number
theorem unique_number_encoding (s1 s2 s3 s4 s5 s6 : score_range) :
  ∃ n : ℕ, ∃ s : ℕ, 
    s = total_score s1 s2 s3 s4 s5 s6 ∧ 
    n = s * 10^6 + s1.val * 10^5 + s2.val * 10^4 + s3.val * 10^3 + s4.val * 10^2 + s5.val * 10 + s6.val := 
sorry

end unique_number_encoding_l303_303797


namespace probability_closer_to_Z_l303_303723

noncomputable def triangle := {p : ℝ × ℝ // 
  (p.1 = 0 ∧ p.2 = 0) ∨                 -- Point Z
  (p.1 = 6 ∧ p.2 = 0) ∨                 -- Point Y
  (p.1 = 0 ∧ p.2 = 8)}                  -- Point X

def is_in_triangle (Q : ℝ × ℝ) : Prop :=
  Q.1 >= 0 ∧ Q.2 >= 0 ∧ Q.1 <= 6 ∧ Q.2 <= 8

def closer_to_Z (Q : ℝ × ℝ) : Prop :=
  (Q.1 - 0) ^ 2 + (Q.2 - 0) ^ 2 < (Q.1 - 6) ^ 2 + (Q.2 - 0) ^ 2 ∧
  (Q.1 - 0) ^ 2 + (Q.2 - 0) ^ 2 < (Q.1 - 0) ^ 2 + (Q.2 - 8) ^ 2

theorem probability_closer_to_Z
  (Q : {pt : ℝ × ℝ // is_in_triangle pt}) :
  (measure_theory.measure_space.volume {pt : ℝ × ℝ // closer_to_Z pt}) /
  (measure_theory.measure_space.volume {pt : ℝ × ℝ // is_in_triangle pt}) = 5 / 24 :=
sorry

end probability_closer_to_Z_l303_303723


namespace product_of_divisors_of_18_l303_303434

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303434


namespace max_determinant_l303_303184

noncomputable def v : ℝ × ℝ × ℝ := (3, 1, -2)
noncomputable def w : ℝ × ℝ × ℝ := (1, -1, 4)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.2.1 * b.2.2 - a.2.2 * b.2.1),
   (a.2.2 * b.1 - a.1 * b.2.2),
   (a.1 * b.2.1 - a.2.1 * b.1))

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (u.1^2 + u.2.1^2 + u.2.2^2)

theorem max_determinant : ∃ u : ℝ × ℝ × ℝ, (v ≠ 0 ∧ w ≠ 0 ∧ u ≠ 0 ∧ magnitude u = 1) → 
  matrix.determinant ![u, v, w] = 2 * real.sqrt 30 :=
  sorry

end max_determinant_l303_303184


namespace product_of_divisors_of_18_l303_303307

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303307


namespace tractor_planting_rate_l303_303565

theorem tractor_planting_rate
  (A : ℕ) (D : ℕ)
  (T1_days : ℕ) (T1 : ℕ)
  (T2_days : ℕ) (T2 : ℕ)
  (total_acres : A = 1700)
  (total_days : D = 5)
  (crew1_tractors : T1 = 2)
  (crew1_days : T1_days = 2)
  (crew2_tractors : T2 = 7)
  (crew2_days : T2_days = 3)
  : (A / (T1 * T1_days + T2 * T2_days)) = 68 := 
sorry

end tractor_planting_rate_l303_303565


namespace product_of_divisors_of_18_is_5832_l303_303398

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l303_303398


namespace max_roots_eq_49_l303_303279

noncomputable def f (x : ℝ) (a b: Fin 50 → ℝ) : ℝ :=
  ∑ i : Fin 50, | x - a i | - ∑ i : Fin 50, | x - b i |

theorem max_roots_eq_49 (a b: Fin 50 → ℝ) (h_distinct: (∀ i j : Fin 50, i ≠ j → a i ≠ a j ∧ b i ≠ b j ∧ a i ≠ b j ∧ a j ≠ b i)) :
  ∃ (roots : Finset ℝ), (|roots| ≤ 49) ∧ (∀ x : ℝ, f x a b = 0 → x ∈ roots) :=
by
  sorry

end max_roots_eq_49_l303_303279


namespace production_rate_l303_303681

variable (k : ℝ) -- production rate in items per (worker·hour)
variable (w1 w2 h1 h2 d1 d2 i1 : ℕ) -- variables for workers, hours, days, and items

-- Initial conditions
def condition1 (h1 h2 d1 d2 : ℕ) : Prop :=
  8 * h1 * d1 * k = 512

-- Target condition
def condition2 (h1 h2 d1 d2 w1 w2 i1 : ℕ) : Prop :=
  w2 * h2 * d2 * k = i1

theorem production_rate (h1 h2 d1 d2 w1 w2 i1 : ℕ) : condition1 h1 h2 d1 d2 → condition2 h1 h2 d1 d2 10 10 1000 :=
by
  intros h_cond
  sorry

end production_rate_l303_303681


namespace product_of_divisors_18_l303_303346

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l303_303346


namespace correct_inequality_l303_303542

variable {f : ℝ → ℝ}

-- Conditions in Lean
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop := 
∀ x1 x2 ∈ Set.Ici (0 : ℝ), x1 < x2 → f x2 < f x1

-- The proof problem
theorem correct_inequality
  (h_even : is_even f)
  (h_mono_dec : is_monotonically_decreasing f) :
  f (Real.log 0.7 6) < f (Real.sqrt 6) ∧ f (Real.sqrt 6) < f (0.7 ^ 6) := sorry

end correct_inequality_l303_303542


namespace gate_distance_probability_correct_l303_303928

-- Define the number of gates
def num_gates : ℕ := 15

-- Define the distance between adjacent gates
def distance_between_gates : ℕ := 80

-- Define the maximum distance Dave can walk
def max_distance : ℕ := 320

-- Define the function that calculates the probability
def calculate_probability (num_gates : ℕ) (distance_between_gates : ℕ) (max_distance : ℕ) : ℚ :=
  let total_pairs := num_gates * (num_gates - 1)
  let valid_pairs :=
    2 * (4 + 5 + 6 + 7) + 7 * 8
  valid_pairs / total_pairs

-- Assert the relevant result and stated answer
theorem gate_distance_probability_correct :
  let m := 10
  let n := 21
  let probability := calculate_probability num_gates distance_between_gates max_distance
  m + n = 31 ∧ probability = (10 / 21 : ℚ) :=
by
  sorry

end gate_distance_probability_correct_l303_303928


namespace find_c_l303_303746

theorem find_c (c : ℝ) (p q : ℝ → ℝ) (h1 : p = λ x, 4 * x - 3) 
  (h2 : q = λ x, 5 * x - c) (h3 : p (q 3) = 53) : c = 1 := 
sorry

end find_c_l303_303746


namespace product_of_divisors_18_l303_303297

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l303_303297


namespace problem1_problem2_problem3_l303_303860

-- Problem 1 Lean 4 Statement
theorem problem1 : ∀ n : ℕ, (n % 8 = 0) ∧ (n % 25 = 24) → ∃ k : ℕ, n = 200 * k + 24 :=
begin
  sorry,
end

-- Problem 2 Lean 4 Statement
theorem problem2 : ∀ n : ℕ, (n % 21 = 0) ∧ (n % 165 = 164) → false :=
begin
  sorry,
end

-- Problem 3 Lean 4 Statement
theorem problem3 : ∀ n : ℤ, (n % 9 = 0) ∧ (n % 25 = 24) ∧ (n % 4 = 2) → n % 900 = 774 :=
begin
  sorry,
end

end problem1_problem2_problem3_l303_303860


namespace prism_volume_eq_l303_303247

variable {a b : ℝ}
variable {α β : ℝ}

noncomputable def volume_prism (a b α β : ℝ) : ℝ :=
  (a^2 * b / 4) * sqrt (3 - 4 * (cos α ^ 2 - cos α * cos β + cos β ^2))

theorem prism_volume_eq : 
  volume_prism a b α β =
  (a^2 * b / 4) * sqrt (3 - 4 * (cos α ^ 2 - cos α * cos β + cos β ^2)) :=
by
  sorry

end prism_volume_eq_l303_303247


namespace angle_PMN_60_l303_303715

theorem angle_PMN_60 (P Q R M N : Type) (h1 : ∠ P Q R = 60) :
  ∠ P M N = 60 := 
sorry

end angle_PMN_60_l303_303715


namespace p_necessary_not_sufficient_q_l303_303967

def condition_p (x : ℝ) : Prop := abs x ≤ 2
def condition_q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

theorem p_necessary_not_sufficient_q (x : ℝ) :
  (condition_p x → condition_q x) = false ∧ (condition_q x → condition_p x) = true :=
by
  sorry

end p_necessary_not_sufficient_q_l303_303967


namespace product_of_divisors_18_l303_303332

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l303_303332


namespace product_of_divisors_of_18_l303_303383

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303383


namespace square_of_binomial_b_value_l303_303675

theorem square_of_binomial_b_value (b : ℤ) (h : ∃ c : ℤ, 16 * (x : ℤ) * x + 40 * x + b = (4 * x + c) ^ 2) : b = 25 :=
sorry

end square_of_binomial_b_value_l303_303675


namespace inequality_abc_l303_303200

theorem inequality_abc (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : a^2 + b^2 + c^2 = 1) : 
  (ab / c) + (bc / a) + (ca / b) ≥ real.sqrt 3 :=
begin
  sorry
end

end inequality_abc_l303_303200


namespace matrix_inverse_pairs_l303_303003

open Matrix

theorem matrix_inverse_pairs :
  let a d : ℝ 
  {M : Matrix (Fin 2) (Fin 2) ℝ} := 
  M = ![![a, 4], ![-12, d]] →
  M ⬝ M = 1 →
  (M ⬝ M = (1 : Matrix (Fin 2) (Fin 2) ℝ)) =
  2 :=
begin
  sorry
end

end matrix_inverse_pairs_l303_303003


namespace valid_path_count_10gon_l303_303695

def is_valid_path (vertices: List ℕ) : Prop :=
  (∀ i, 1 ≤ i ∧ i < 9 → 
    (vertices.nth (2 * i + 1) = vertices.nth (2 * i - 1) + 2 ∨ 
     vertices.nth (2 * i + 1) = vertices.nth (2 * i - 1) + 1)) ∧
  (vertices.nth 0 = 1) ∧ 
  (vertices.nth (List.length vertices - 1) = 10) ∧
  (vertices ≠ [1, 9, 10]) ∧
  (vertices ≠ [1, 10])

theorem valid_path_count_10gon : 
  {paths : List (List ℕ) // ∀ p ∈ paths, is_valid_path p}.length = 55 := 
sorry

end valid_path_count_10gon_l303_303695


namespace minimum_weights_needed_l303_303558

-- Define the conditions of the problem
def weights_balanced_with_non_integers (S : Set ℝ) : Prop :=
  (∀ w ∈ S, w ∉ ℤ) ∧ (∀ n : ℕ, n ≥ 1 ∧ n ≤ 40 → 
    ∃ subset ⊆ S, (∀ w ∈ subset, w ∉ ℤ) ∧ (∑ w in subset, w = n))

-- Define the problem statement
theorem minimum_weights_needed :
  ∃ S : Set ℝ, weights_balanced_with_non_integers S ∧ S.card = 7 :=
sorry

end minimum_weights_needed_l303_303558


namespace min_rubles_needed_to_reverse_strip_l303_303707

-- Definition of the problem
def strip_length := 100

-- Conditions
def can_swap_adjacent_for_ruble (i j : Nat) : Prop := 
  abs (i - j) = 1

def can_swap_three_apart_for_free (i j : Nat) : Prop := 
  abs (i - j) = 4

-- Minimum rubles required to reverse the tokens
def min_rubles_to_reverse (n : Nat) : Nat :=
  if n = 100 then 50 else sorry

-- The statement to be proven
theorem min_rubles_needed_to_reverse_strip : min_rubles_to_reverse strip_length = 50 := 
by sorry

end min_rubles_needed_to_reverse_strip_l303_303707


namespace equal_cost_miles_l303_303667

   -- Conditions:
   def initial_fee_first_plan : ℝ := 65
   def cost_per_mile_first_plan : ℝ := 0.40
   def cost_per_mile_second_plan : ℝ := 0.60

   -- Proof problem:
   theorem equal_cost_miles : 
     let x := 325 in 
     initial_fee_first_plan + cost_per_mile_first_plan * x = cost_per_mile_second_plan * x :=
   by
     -- Placeholder for the proof
     sorry
   
end equal_cost_miles_l303_303667


namespace sum_of_coordinates_proof_l303_303814

noncomputable def sum_of_coordinates : ℝ := 
let p1 := (10 + 4 * Real.sqrt 11, 22) in
let p2 := (10 - 4 * Real.sqrt 11, 22) in
let p3 := (10 + 4 * Real.sqrt 11, 8) in
let p4 := (10 - 4 * Real.sqrt 11, 8) in
(p1.1 + p2.1 + p3.1 + p4.1) + (p1.2 + p2.2 + p3.2 + p4.2)

theorem sum_of_coordinates_proof :
  let d := sum_of_coordinates 
  in d = 100 :=
by
  sorry

end sum_of_coordinates_proof_l303_303814


namespace sum_b_first_10_terms_l303_303802

/-- Sequence a_n defined by a_n = (2n + 3) / 5 -/
def a (n : ℕ) : ℚ := (2 * n + 3) / 5

/-- Sequence b_n defined by floor(a_n) -/
def b (n : ℕ) : ℤ := int.floor (a n)

/-- Sum of first 10 terms of sequence b_n -/
theorem sum_b_first_10_terms : (List.range 10).map b |>.sum = 24 := by
  sorry

end sum_b_first_10_terms_l303_303802


namespace acute_triangle_statements_l303_303703

variable (A B C : ℝ)

-- Conditions for acute triangle
def acute_triangle := A + B + C = π ∧ 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2

-- Statement A: If A > B, then sin A > sin B.
def statement_A := ∀ h : A > B, Real.sin A > Real.sin B

-- Statement B: If A = π / 3, then the range of values for B is (0, π / 2).
def statement_B := ∀ h : A = π / 3, 0 < B ∧ B < π / 2

-- Statement C: sin A + sin B > cos A + cos B
def statement_C := Real.sin A + Real.sin B > Real.cos A + Real.cos B

-- Statement D: tan B tan C > 1
def statement_D := Real.tan B * Real.tan C > 1

-- The theorem to prove
theorem acute_triangle_statements (h : acute_triangle A B C) :
  statement_A A B C ∧ ¬statement_B A B C ∧ statement_C A B C ∧ statement_D A B C :=
by sorry

end acute_triangle_statements_l303_303703


namespace alice_speed_is_6_5_l303_303833

-- Definitions based on the conditions.
def a : ℝ := sorry -- Alice's speed
def b : ℝ := a + 3 -- Bob's speed

-- Alice cycles towards the park 80 miles away and Bob meets her 15 miles away from the park
def d_alice : ℝ := 65 -- Alice's distance traveled (80 - 15)
def d_bob : ℝ := 95 -- Bob's distance traveled (80 + 15)

-- Equating the times
def time_eqn := d_alice / a = d_bob / b

-- Alice's speed is 6.5 mph
theorem alice_speed_is_6_5 : a = 6.5 :=
by
  have h1 : b = a + 3 := sorry
  have h2 : a * 65 = (a + 3) * 95 := sorry
  have h3 : 30 * a = 195 := sorry
  have h4 : a = 6.5 := sorry
  exact h4

end alice_speed_is_6_5_l303_303833


namespace second_catch_l303_303693

theorem second_catch (N x : ℕ) (hN : N = 1250) (h : 2 * N = 50 * x) : x = 50 :=
by
  subst hN
  calc
    x = 2 * 1250 / 50 := by sorry
    ... = 50 := by sorry

end second_catch_l303_303693


namespace sum_geometric_series_l303_303241

theorem sum_geometric_series (x : ℂ) (h₀ : x ≠ 1) (h₁ : x^10 - 3*x + 2 = 0) : 
  x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := 
by 
  sorry

end sum_geometric_series_l303_303241


namespace line_divides_triangle_l303_303534

noncomputable def area_triangle (D E F : (ℝ × ℝ)) : ℝ :=
1 / 2 * abs ((E.1 - D.1) * (F.2 - D.2) - (F.1 - D.1) * (E.2 - D.2))

theorem line_divides_triangle :
  ∃ θ : ℝ, ∀ (D E F : (ℝ × ℝ)),
  D = (0, 0) → E = (10, 0) → F = (0, 2) →
  ∃ line_eq : ℝ → ℝ, line_eq = λ x, (Real.tan θ) * x ∧
  area_triangle D E F / 2 = area_triangle D F (x, line_eq x) :=
begin
  sorry
end

end line_divides_triangle_l303_303534


namespace product_of_divisors_of_18_l303_303414

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l303_303414


namespace product_of_divisors_of_18_l303_303311

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303311


namespace ellipse_properties_l303_303979

theorem ellipse_properties (a b : ℝ) (h : a > b) (e : ℝ) (ecc : e = sqrt 3 / 2) (d : ℝ) (dist : d = 2) :
  (a = 2 ∧ b = 1 ∧ (∀ k : ℝ, k = ± sqrt 11 / 2 ↔ 
    let l := λ x, k * x + sqrt 3,
    let A := (x1, l x1),
    let B := (x2, l x2),
    let LHS := x1 * x2 + l x1 * l x2 in 
    LHS = 0)) → 
  (∃ c x y : ℝ, (x ^ 2 + (y ^ 2) / 4 = 1) ∧ 
    ((∃ k : ℝ, k = sqrt 11 / 2 ∨ k = - sqrt 11 / 2)
    ∧ ∃ x1 x2 : ℝ, (4 + k^2) * x^2 + 2 * sqrt 3 * k * x - 1 = 0 
    ∧ ((x1 + x2 = - 2 * sqrt 3 * k / (4 + k^2)) 
    ∧ (x1 * x2 = - 1 / (4 + k^2)) 
    ∧ (1 + k^2) * (- 1 / (4 + k^2)) + sqrt 3 * k * (- 2 * sqrt 3 * k / (4 + k^2)) + 3 = 0))) := 
sorry

end ellipse_properties_l303_303979


namespace path_length_of_B_l303_303900

variable (BC : ℝ) (AC : ℝ) (B : Point) (ABC : Region)

def arc_AC := semicircle AC B
def rolled := rolled_semicircle AC B (origin_orientation AC B)

theorem path_length_of_B 
(h1 : arc_AC)
(h2 : BC = 2)
(h3 : rolled):
  length_of_path B = 2 * Real.pi :=
sorry

end path_length_of_B_l303_303900


namespace volume_of_second_cube_l303_303445

noncomputable def cube_volume (s : ℝ) : ℝ := s^3
noncomputable def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

theorem volume_of_second_cube :
  (∃ s₁ s₂ : ℝ, cube_volume s₁ = 8 ∧
                 cube_surface_area s₂ = 3 * cube_surface_area s₁ ∧
                 cube_volume s₂ = 24 * real.sqrt 3) :=
by
  sorry

end volume_of_second_cube_l303_303445


namespace product_of_divisors_of_18_l303_303377

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303377


namespace product_of_divisors_of_18_l303_303292

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l303_303292


namespace product_of_divisors_of_18_l303_303286

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l303_303286


namespace paths_from_A_to_B_are_sixteen_l303_303923

def Vertex : Type := ℕ -- Representation for vertices

inductive Path : Type
| fromAtoB : list Vertex → Path

-- Define the vertices
def A : Vertex := 0
def B : Vertex := 1
def C : Vertex := 2
def D : Vertex := 3
def E : Vertex := 4
def F : Vertex := 5
def G : Vertex := 6

-- Define an edge between two vertices
def edge (v₁ v₂ : Vertex) : Prop :=
(v₁, v₂) ∈ [(A, G), (G, D), (A, C), (A, D), (A, F), (G, E),
             (C, B), (D, B), (D, E), (D, F), (E, F), (F, B)]

-- Define a path that visits each labeled point at most once
def valid_path (p : list Vertex) : Prop :=
(∀ v, v ∈ p → v ∈ [A, B, C, D, E, F, G]) ∧ (list.nodup p) ∧ ((p.head?) = some A) ∧ (p.last? = some B)

-- Define the total number of unique paths from A to B
def num_paths_from_A_to_B : ℕ :=
16

-- Statement: Prove that the number of continuous paths from A to B, visiting each labeled point at most once, is 16.
theorem paths_from_A_to_B_are_sixteen : 
  ∃ ps : list (list Vertex), (∀ p ∈ ps, valid_path p ∧ (p.head? = some A) ∧ (p.last? = some B)) ∧ ps.length = num_paths_from_A_to_B :=
sorry

end paths_from_A_to_B_are_sixteen_l303_303923


namespace find_FC_l303_303045

variable (DC : ℝ) (CB : ℝ) (AB AD ED : ℝ)
variable (FC : ℝ)
variable (h1 : DC = 9)
variable (h2 : CB = 6)
variable (h3 : AB = (1/3) * AD)
variable (h4 : ED = (2/3) * AD)

theorem find_FC : FC = 9 :=
by sorry

end find_FC_l303_303045


namespace f_eq_f_inv_l303_303545

noncomputable def f (x : ℝ) : ℝ := 3 * x - 7

noncomputable def f_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem f_eq_f_inv (x : ℝ) : f x = f_inv x ↔ x = 3.5 := by
  sorry

end f_eq_f_inv_l303_303545


namespace product_of_divisors_of_18_l303_303336

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l303_303336


namespace inequality_abc_l303_303214

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := 
by 
  sorry

end inequality_abc_l303_303214


namespace product_of_divisors_of_18_l303_303359

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303359


namespace prob_remainder_mod_1000_l303_303524

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ 
| n 0 := 1
| 0 k := 0
| n k := binom (n-1) (k-1) * n / k

-- Define the sum we are interested in, only including indices that are multiples of 4
def sum_binom_2011_multiple_4 : ℕ :=
  (Finset.range (2012 / 4 + 1)).sum (λ i, binom 2011 (4 * i))

-- The statement we want to prove
theorem prob_remainder_mod_1000 : 
  sum_binom_2011_multiple_4 % 1000 = 12 := 
sorry

end prob_remainder_mod_1000_l303_303524


namespace binomial_coefficient_sum_mod_l303_303530

theorem binomial_coefficient_sum_mod : 
  let S := ((1 + Complex.exp (Complex.I * Real.pi / 2))^2011) + 
           ((1 + Complex.exp (3 * Complex.I * Real.pi / 2))^2011) + 
           ((1 + -1)^2011) + 
           ((1 + 1)^2011)
  in 
  let desired_sum := (range 503).sum (λ j, Nat.choose 2011 (4 * j)) / 4
  in 
  (S % 1000 = 137) :
  nat.Mod 1000 S = 137 := 
begin
  sorry
end

end binomial_coefficient_sum_mod_l303_303530


namespace number_of_possible_values_r_l303_303505

theorem number_of_possible_values_r (r : ℝ) (C_C : ℝ := 300 * Real.pi) :
  (5 < r ∧ r < 150) ∧ ∃ k : ℕ, r = k * 50 → 
  (r ∈ {50, 100} ∧ ∃ n : ℕ, n = 2) :=
by
  sorry

end number_of_possible_values_r_l303_303505


namespace cos_identity_l303_303633

-- Define the given conditions 
variable (α : ℝ)
def terminal_point (α : ℝ) (x y : ℝ) := x = -4 ∧ y = 3

-- Define the radius (as given in the proof problem)
def radius (x y : ℝ) := real.sqrt (x ^ 2 + y ^ 2)

-- Define sine and cosine using given conditions
def sin_alpha (x y : ℝ) := y / radius x y

-- Prove the identity
theorem cos_identity (α : ℝ) (x y : ℝ) (h : terminal_point α x y) :
  cos (3 * real.pi / 2 - α) = -sin_alpha x y :=
by {
  -- The proof will be filled here.
  sorry
}

end cos_identity_l303_303633


namespace product_of_divisors_of_18_l303_303432

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303432


namespace incorrect_statement_C_l303_303047

noncomputable def f (a x : ℝ) : ℝ := x^2 * (Real.log x - a) + a

theorem incorrect_statement_C :
  ¬ (∀ a : ℝ, a > 0 → ∀ x : ℝ, x > 0 → f a x ≥ 0) := sorry

end incorrect_statement_C_l303_303047


namespace product_of_divisors_18_l303_303327

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l303_303327


namespace product_of_divisors_18_l303_303304

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l303_303304


namespace base7_to_dec_base4_eq_l303_303840

def base7_to_dec (s : List ℕ) : ℕ :=
  s.foldl (λ acc d => acc * 7 + d) 0

def dec_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  List.reverse $ List.unfoldl (λ x => if x = 0 then none else some (x % 4, x / 4)) n

theorem base7_to_dec_base4_eq :
  base7_to_dec [5, 4, 3, 2, 1, 0] = 94773 ∧ dec_to_base4 94773 = [1, 1, 3, 2, 3, 0, 1, 1] :=
by 
  sorry

end base7_to_dec_base4_eq_l303_303840


namespace find_b_l303_303001

-- Definitions for conditions
def eq1 (a : ℤ) : Prop := 2 * a + 1 = 1
def eq2 (a b : ℤ) : Prop := 2 * b - 3 * a = 2

-- The theorem statement
theorem find_b (a b : ℤ) (h1 : eq1 a) (h2 : eq2 a b) : b = 1 :=
  sorry  -- Proof to be filled in.

end find_b_l303_303001


namespace john_sells_cow_per_pound_correct_l303_303727

-- Define the initial conditions
def initial_weight : ℝ := 400
def weight_increase_factor : ℝ := 1.5
def value_increase : ℝ := 600

-- Calculate the final weight after the increase
def final_weight := initial_weight * weight_increase_factor

-- Calculate the weight gain
def weight_gain := final_weight - initial_weight

-- Define the price per pound variable
def price_per_pound := value_increase / weight_gain

-- The theorem statement to be proven
theorem john_sells_cow_per_pound_correct :
  price_per_pound = 3 := by sorry

end john_sells_cow_per_pound_correct_l303_303727


namespace second_number_is_180_l303_303265

theorem second_number_is_180 
  (x : ℝ) 
  (first : ℝ := 2 * x) 
  (third : ℝ := (1/3) * first)
  (h : first + x + third = 660) : 
  x = 180 :=
sorry

end second_number_is_180_l303_303265


namespace product_of_divisors_of_18_l303_303420

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l303_303420


namespace product_of_divisors_18_l303_303330

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l303_303330


namespace evaluate_expression_l303_303933

theorem evaluate_expression : 
  let i := real.sqrt (-1)
  in i^(14761) + i^(14762) + i^(14763) + i^(14764) = 0 :=
by
  sorry

end evaluate_expression_l303_303933


namespace sum_of_valid_candies_l303_303446

/-- 
  Define the problem conditions: 
  N % 6 = 5 and N % 8 = 7, N < 100.
-/
def valid_candies (N : ℕ) : Prop :=
  N % 6 = 5 ∧ N % 8 = 7 ∧ N < 100

/-- 
  The main theorem: 
  The sum of all valid candies which satisfy the conditions is 236.
-/
theorem sum_of_valid_candies :
  (Finset.filter valid_candies (Finset.range 100)).sum id = 236 :=
by
  sorry

end sum_of_valid_candies_l303_303446


namespace m_seq_non_decreasing_l303_303162

-- Define P and Q as polynomials with integer coefficients
variables {P Q : ℤ[X][X]} -- ℤ[X][X] represents polynomials with integer coefficients in two variables

-- Define the sequences a_n and b_n
def a_seq (a_0 : ℤ) (b_0 : ℤ) : ℕ → ℤ
| 0     := a_0
| (n+1) := P.eval₂ a_seq n b_seq n

def b_seq (a_0 : ℤ) (b_0 : ℤ) : ℕ → ℤ
| 0     := b_0
| (n+1) := Q.eval₂ a_seq n b_seq n

-- Define m_n
def m_seq (a_0 : ℤ) (b_0 : ℤ) (n : ℕ) : ℤ :=
  (Int.gcd (a_seq a_0 b_0 (n+1) - a_seq a_0 b_0 n) (b_seq a_0 b_0 (n+1) - b_seq a_0 b_0 n)) - 1

-- The main theorem
theorem m_seq_non_decreasing 
  (a_0 b_0 : ℤ) : ∀ n : ℕ, (m_seq a_0 b_0 (n+1)) ≥ (m_seq a_0 b_0 n) :=
sorry

end m_seq_non_decreasing_l303_303162


namespace enclosed_area_correct_l303_303069

-- Define the two lines
def line1 (x : ℝ) : ℝ := -x + 2
def line2 (x : ℝ) : ℝ := 3 * x + 2

-- Define the y-intercept of line1 and check its intersection with line2
def y_intercept_line1 : ℝ := line1 0
def y_intersect_line2_at_yaxis : Prop := line2 0 = y_intercept_line1

-- Define the x-intercept of line2
def x_intercept_line2 : ℝ := -2 / 3

-- Define the area of the triangle enclosed by line2 and the coordinate axes
def enclosed_area : ℝ := 1 / 2 * |x_intercept_line2| * y_intercept_line1

-- Prove the area calculation
theorem enclosed_area_correct : y_intersect_line2_at_yaxis → enclosed_area = 2 / 3 :=
by
  intro h
  rw [y_intercept_line2_at_yaxis] at h
  rw [enclosed_area, x_intercept_line2]
  norm_num
  sorry -- Proof goes here

end enclosed_area_correct_l303_303069


namespace marks_per_correct_answer_l303_303137

-- Definitions based on the conditions
def total_questions : ℕ := 60
def total_marks : ℕ := 160
def correct_questions : ℕ := 44
def wrong_mark_loss : ℕ := 1

-- The number of correct answers multiplies the marks per correct answer,
-- minus the loss from wrong answers, equals the total marks.
theorem marks_per_correct_answer (x : ℕ) :
  correct_questions * x - (total_questions - correct_questions) * wrong_mark_loss = total_marks → x = 4 := by
sorry

end marks_per_correct_answer_l303_303137


namespace perpendicular_ED_BL_l303_303719

variables {A B C M L D E : Type}

-- Defining the conditions
def is_triangle (A B C : Type) : Prop := true
def is_median (B M A C : Type) : Prop := true
def is_angle_bisector (B L A C : Type) : Prop := true
def parallel (x y : Type) : Prop := true
def AB_gt_BC (A B C : Type) : Prop := true

theorem perpendicular_ED_BL 
  (h_triangle : is_triangle A B C) 
  (h_median : is_median B M A C)
  (h_bisector : is_angle_bisector B L A C)
  (h_parallel_MD_AB : parallel M D)
  (h_parallel_LE_BC : parallel L E)
  (h_AB_gt_BC : AB_gt_BC A B C) : 
  perpendicular E D B L :=
by sorry

end perpendicular_ED_BL_l303_303719


namespace percentage_decrease_l303_303716

theorem percentage_decrease (x y : ℝ) : 
  (xy^2 - (0.7 * x) * (0.6 * y)^2) / xy^2 = 0.748 :=
by
  sorry

end percentage_decrease_l303_303716


namespace product_of_divisors_18_l303_303357

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l303_303357


namespace product_of_divisors_of_18_l303_303364

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303364


namespace average_score_correct_l303_303239

-- Define the set of numbers between 1 and 200
def numbers : finset ℕ := finset.range 201

-- Define the score function for a given subset of 100 elements
def score (s : finset ℕ) (h : s.card = 100) : ℤ :=
  let total_sum := 200 * 201 / 2 in
  (total_sum - 2 * s.sum id)^2

-- Calculate the average score
def average_score : ℚ :=
  let total_sum := 200 * 201 / 2 in
  let num_ways := nat.choose 200 100 in
  let left_term := (total_sum^2 * num_ways : ℤ) in
  let mid_term := (80400 * ∑ s in finset.powerset_len 100 numbers, s.sum id) in
  let right_term := (∑ s in finset.powerset_len 100 numbers, (s.sum id)^2) in
  (left_term.ediv num_ways - mid_term + 4 * right_term : ℚ) / num_ways

theorem average_score_correct :
  average_score = 134000000 / 199 :=
sorry

end average_score_correct_l303_303239


namespace product_of_divisors_18_l303_303351

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l303_303351


namespace A_and_D_mutually_exclusive_A_and_B_mutually_exclusive_A_and_C_independent_B_and_C_independent_l303_303586

open Set

variable {Ω : Type} [Fintype Ω] (A B C D : Set Ω)

noncomputable def P (s : Set Ω) : ℝ := (Fintype.card s : ℝ) / (Fintype.card Ω)

variable (h1 : Fintype.card Ω = 100)
variable (h2 : Fintype.card A = 60)
variable (h3 : Fintype.card B = 40)
variable (h4 : Fintype.card C = 20)
variable (h5 : Fintype.card D = 10)
variable (h6 : Fintype.card (A ∪ B) = 100)
variable (h7 : Fintype.card (A ∩ C) = 12)
variable (h8 : Fintype.card (A ∪ D) = 70)

-- Prove the necessary statements:

theorem A_and_D_mutually_exclusive : Disjoint A D :=
by
  sorry

theorem A_and_B_mutually_exclusive : Disjoint A B :=
by
  sorry

theorem A_and_C_independent : P (A ∩ C) = P A * P C :=
by
  sorry

theorem B_and_C_independent : P (B ∩ C) = P B * P C :=
by
  sorry

end A_and_D_mutually_exclusive_A_and_B_mutually_exclusive_A_and_C_independent_B_and_C_independent_l303_303586


namespace product_AD_BD_l303_303000

noncomputable def triangle_configuration (A B C P D : Point) :=
  dist P A = dist P C ∧
  angle A P C = 2 * angle A B C ∧
  intersects (line A B) (line C P) D ∧
  dist C P = 5 ∧
  dist C D = 3

theorem product_AD_BD (A B C P D : Point) (h : triangle_configuration A B C P D) :
  dist A D * dist B D = 14 :=
  sorry

end product_AD_BD_l303_303000


namespace expansion_contains_constant_term_l303_303990

theorem expansion_contains_constant_term (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 16) (h3 : ∃ r : ℕ, n = 4 * r) :
  ∃ c : ℕ, ∃ k : ℕ, (x - (1 / x^3)) ^ n = c ∧ c = (choose n k) * x ^ (n - 4 * k) :=
sorry

end expansion_contains_constant_term_l303_303990


namespace product_of_divisors_18_l303_303358

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l303_303358


namespace smallest_number_starting_with_2016_and_divisible_by_2017_l303_303580

theorem smallest_number_starting_with_2016_and_divisible_by_2017 :
  ∃ n : ℕ, (nat.digits 10 n).reverse.take 4 = [6, 1, 0, 2] ∧ (n % 2017 = 0) ∧ 
  ∀ m : ℕ, (nat.digits 10 m).reverse.take 4 = [6, 1, 0, 2] → (m % 2017 = 0) → n ≤ m :=
  ∃ n : ℕ, (nat.digits 10 n).reverse.take 4 = [6, 1, 0, 2] ∧ (n % 2017 = 0) ∧ n = 20162001
sorry

end smallest_number_starting_with_2016_and_divisible_by_2017_l303_303580


namespace abc_positive_l303_303046

theorem abc_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : a > 0 ∧ b > 0 ∧ c > 0 :=
by
  sorry

end abc_positive_l303_303046


namespace product_of_divisors_18_l303_303352

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l303_303352


namespace arnold_danny_age_l303_303496

theorem arnold_danny_age (x : ℕ) : (x + 1) * (x + 1) = x * x + 17 → x = 8 :=
by
  sorry

end arnold_danny_age_l303_303496


namespace solve_proof_problem_1_solve_proof_problem_2_l303_303919

noncomputable def proof_problem_1 (a b : ℝ) : Prop :=
  ((a^(3/2) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / ((1/3) * a * b^(5/6))) = -9 * a

noncomputable def proof_problem_2 : Prop :=
  (real.log 3 / (2 * real.log 2) * (real.log 2 / (2 * real.log 3)) - real.log (32^(1/4)) / real.log (1/2)) = 11 / 8

theorem solve_proof_problem_1 (a b : ℝ) : proof_problem_1 a b := by
  sorry

theorem solve_proof_problem_2 : proof_problem_2 := by
  sorry

end solve_proof_problem_1_solve_proof_problem_2_l303_303919


namespace triangle_length_sum_l303_303141

variables {A B C O P Q : Type*}
variables [EuclideanGeometry A B C O P Q]

-- Given conditions
variables (hBC : dist B C = 5)
variables (hBC_midpoint : midpoint O B C)
variables (hCircleCenter : center O P Q)
variables (hCircleRadius : radius O = 2)
variables (hOCo : dist O A = 2.5)
variables (hOP : dist O P = 2)
variables (hOQ : dist O Q = 2)

-- Question transformed into a proof statement
theorem triangle_length_sum :
  dist A P^2 + dist A Q^2 + dist P Q^2 = 73 / 2 :=
begin
  sorry
end

end triangle_length_sum_l303_303141


namespace compute_a_111_l303_303482

def repunit : ℕ → ℕ
| 0     := 1
| (n+1) := repunit n * 10 + 1

def sum_of_repunits (lst : List ℕ) : ℕ :=
lst.foldl (λ acc n => acc + repunit n) 0

def a (n : ℕ) : ℕ :=
(sum_of_repunits (List.range n).filter (λ m => (n >> m) % 2 = 1))

theorem compute_a_111 : a 111 = 1223456 := by
  sorry

end compute_a_111_l303_303482


namespace isosceles_triangle_area_l303_303021

theorem isosceles_triangle_area (a b c : ℝ) (h₁ : a = 10) (h₂ : b = 10) (h₃ : c = 12) (h₄ : a = b) :
  (1 / 2) * c * (sqrt (a ^ 2 - (c / 2) ^ 2)) = 48 :=
by
  -- proof omitted
  sorry

end isosceles_triangle_area_l303_303021


namespace james_owns_145_l303_303158

theorem james_owns_145 (total : ℝ) (diff : ℝ) (james_and_ali : total = 250) (james_more_than_ali : diff = 40):
  ∃ (james ali : ℝ), ali + diff = james ∧ ali + james = total ∧ james = 145 :=
by
  sorry

end james_owns_145_l303_303158


namespace union_eq_l303_303654

def setA : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def setB : Set ℝ := { x | 2 < x ∧ x ≤ 5 }

theorem union_eq : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end union_eq_l303_303654


namespace color_impossibility_l303_303155

def number_of_color_pairs : ℕ := Nat.choose 16 2
def number_of_adjacencies_in_8x8_grid : ℕ := (8 * 7) * 2

theorem color_impossibility
  (colors : Finset ℕ)
  (h_colors : colors.card = 16)
  (grid : Finset (Fin 64))
  (h_grid : grid.card = 64)
  : number_of_color_pairs > number_of_adjacencies_in_8x8_grid :=
begin
  -- Calculation details:
  -- number_of_color_pairs = Nat.choose 16 2 = 120
  -- number_of_adjacencies_in_8x8_grid = (8 * 7) * 2 = 112
  sorry
end

end color_impossibility_l303_303155


namespace calculate_remainder_l303_303511

open Nat

theorem calculate_remainder :
  let ω := complex.exp (2 * complex.pi * complex.I / 4) in
  ω^4 = 1 ∧ ω ≠ 1 ∧ ω^2 = -1 ∧ ω^3 = -ω ∧
  let S := (1 + ω)^2011 + (1 + ω^2)^2011 + (1 + ω^3)^2011 + (2:ℂ)^2011 in
  S = 4 * ∑ k in range (503), nat.choose 2011 (4 * k) →
  (1 + ω^2)^2011 = 0 ∧ (1 + ω)^2011 + (1 + ω^3)^2011 = 0 →
  S = (2:ℂ)^2011 →
  (2^2011 : ℕ) % 8 = 0 ∧ (2^2011 : ℕ) % 125 = 48 →
  (2^2011 : ℕ) % 1000 = 48 →
  (4 * ∑ k in range (503), nat.choose 2011 (4 * k)) % 1000 = 48 →
  ((∑ k in range (503), nat.choose 2011 (4 * k)) % 1000 = 12) :=
by
  intros ω ω4 ω_ne ω2 ω3 S hS h1 h2 h3 h4 h5
  sorry

end calculate_remainder_l303_303511


namespace sum_of_digits_of_square_mod_9_l303_303944

theorem sum_of_digits_of_square_mod_9 (n : ℤ) : 
  ∃ r, r ∈ {0, 1, 4, 7} ∧ (n^2 % 9 = r) :=
sorry

end sum_of_digits_of_square_mod_9_l303_303944


namespace non_factorial_trailing_numbers_less_than_1992_l303_303688

def factorial_trailing_number (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ (n = (∑ k in (Finset.range (Nat.log 5 m + 1)), m / 5^k))

theorem non_factorial_trailing_numbers_less_than_1992 :
  (Finset.range 1992).filter (λ n, ¬ factorial_trailing_number n)).card = 396 := 
sorry

end non_factorial_trailing_numbers_less_than_1992_l303_303688


namespace find_current_I_l303_303536

-- Definition of complex numbers V and Z based on the given problem
def V : ℂ := 2 + 2 * complex.i
def Z : ℂ := 2 - 2 * complex.i

-- The relationship V = I * Z in a circuit.
theorem find_current_I : ∃ I : ℂ, V = I * Z ∧ I = complex.i :=
by
    use complex.i
    split
    { -- Proof of V = I * Z
      sorry }
    { -- Proof of I = i
      exact rfl }

end find_current_I_l303_303536


namespace constant_term_of_f_comp_f_is_minus_20_l303_303079

def f (x : ℝ) : ℝ := 
  if x < 0 then (x - 1/x) ^ 6 else -Real.sqrt x 

theorem constant_term_of_f_comp_f_is_minus_20 (x : ℝ) (h : x > 0) : 
  ∃ c : ℝ, c = -20 ∧ 
  (∃ u : ℝ, x = u ∧ f (f u) = c) := sorry

end constant_term_of_f_comp_f_is_minus_20_l303_303079


namespace circle_center_is_neg4_2_l303_303022

noncomputable def circle_center (x y : ℝ) : Prop :=
  x^2 + 8 * x + y^2 - 4 * y = 16

theorem circle_center_is_neg4_2 :
  ∃ (h k : ℝ), (h = -4 ∧ k = 2) ∧
  ∀ (x y : ℝ), circle_center x y ↔ (x + 4)^2 + (y - 2)^2 = 36 :=
by
  sorry

end circle_center_is_neg4_2_l303_303022


namespace binom_sum_mod_1000_l303_303518

theorem binom_sum_mod_1000 : 
  (∑ i in (finset.range 2012).filter (λ i, i % 4 = 0), nat.choose 2011 i) % 1000 = 15 :=
sorry

end binom_sum_mod_1000_l303_303518


namespace sum_first_20_abs_arithmetic_seq_l303_303705

open Finset

/-- In an arithmetic sequence {a_n} with a common difference greater than 1, 
    it is known that a_1^2=64 and a_2+a_3+a_10=36. Prove that the sum of the 
    first 20 terms of the sequence {|a_n|} is 812. -/
theorem sum_first_20_abs_arithmetic_seq {a : ℕ → ℤ} (d : ℤ) (h1 : d > 1) 
  (h2 : a 1 = ±8) (h3 : a 2 + a 3 + a 10 = 36) : 
  (∑ n in range 20, |a (n + 1)|) = 812 :=
sorry

end sum_first_20_abs_arithmetic_seq_l303_303705


namespace portion_of_money_given_to_Blake_l303_303500

theorem portion_of_money_given_to_Blake
  (initial_amount : ℝ)
  (tripled_amount : ℝ)
  (sale_amount : ℝ)
  (amount_given_to_Blake : ℝ)
  (h1 : initial_amount = 20000)
  (h2 : tripled_amount = 3 * initial_amount)
  (h3 : sale_amount = tripled_amount)
  (h4 : amount_given_to_Blake = 30000) :
  amount_given_to_Blake / sale_amount = 1 / 2 :=
sorry

end portion_of_money_given_to_Blake_l303_303500


namespace negative_solution_range_l303_303623

theorem negative_solution_range (m : ℝ) : (∃ x : ℝ, 2 * x + 4 = m - x ∧ x < 0) → m < 4 := by
  sorry

end negative_solution_range_l303_303623


namespace perpendicular_midpoints_l303_303195

-- Define points A, B, O, M on the circle
variables (A B O M : Type*) [HasCenter O] [HasArc A B 60]
variables (C D E F : Type*) [MidpointOfAO C] [MidpointOfOB D] [MidpointOfBM E] [MidpointOfMA F]

-- Prove the perpendicularity condition
theorem perpendicular_midpoints :
  are_perpendicular (line_through F D) (line_through E C) :=
sorry

end perpendicular_midpoints_l303_303195


namespace prob_remainder_mod_1000_l303_303526

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ 
| n 0 := 1
| 0 k := 0
| n k := binom (n-1) (k-1) * n / k

-- Define the sum we are interested in, only including indices that are multiples of 4
def sum_binom_2011_multiple_4 : ℕ :=
  (Finset.range (2012 / 4 + 1)).sum (λ i, binom 2011 (4 * i))

-- The statement we want to prove
theorem prob_remainder_mod_1000 : 
  sum_binom_2011_multiple_4 % 1000 = 12 := 
sorry

end prob_remainder_mod_1000_l303_303526


namespace Emily_walks_more_distance_than_Troy_l303_303830

theorem Emily_walks_more_distance_than_Troy (Troy_distance Emily_distance : ℕ) (days : ℕ) 
  (hTroy : Troy_distance = 75) (hEmily : Emily_distance = 98) (hDays : days = 5) : 
  ((Emily_distance * 2 - Troy_distance * 2) * days) = 230 :=
by
  sorry

end Emily_walks_more_distance_than_Troy_l303_303830


namespace derivative_at_1_l303_303970

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x

theorem derivative_at_1 : Deriv f 1 = 2 * Real.log 2 + (1 / Real.log 2) := by
  -- The proof will go here but is omitted as per the instructions
  sorry

end derivative_at_1_l303_303970


namespace magnitude_of_difference_is_3sqrt5_l303_303096

noncomputable def vector_a : ℝ × ℝ := (1, -2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 4)

def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem magnitude_of_difference_is_3sqrt5 (x : ℝ) (h_parallel : parallel vector_a (vector_b x)) :
  (Real.sqrt ((vector_a.1 - (vector_b x).1) ^ 2 + (vector_a.2 - (vector_b x).2) ^ 2)) = 3 * Real.sqrt 5 :=
sorry

end magnitude_of_difference_is_3sqrt5_l303_303096


namespace lambda_value_l303_303095

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem lambda_value (a b : V) (λ : ℝ) (h₀ : ¬ (∀ k : ℝ, b = k • a)) (h₁ : ∃ k : ℝ, λ • a + b = k • (a - 2 • b)) : 
  λ = -1 / 2 := 
sorry

end lambda_value_l303_303095


namespace cosine_of_angle_l303_303632

theorem cosine_of_angle {α : ℝ} (P : ℝ × ℝ)
  (hP : P = (-3 / 5, 4 / 5))
  (h_unit_circle : ∀ (x y : ℝ), (x, y) = P → x^2 + y^2 = 1) :
  ∃ (cos_alpha : ℝ), cos_alpha = -3 / 5 := by
  use -3 / 5
  sorry

end cosine_of_angle_l303_303632


namespace product_of_divisors_18_l303_303347

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l303_303347


namespace coordinates_of_Q_l303_303770

theorem coordinates_of_Q (P : ℝ × ℝ) (θ : ℝ) (arc_length : ℝ)
  (h1 : P = (1, 0))
  (h2 : θ = -π / 3)
  (h3 : arc_length = π / 3) :
  let Q := (Real.cos θ, Real.sin θ) in
  Q = (1 / 2, -Real.sqrt 3 / 2) :=
by
  sorry

end coordinates_of_Q_l303_303770


namespace product_of_divisors_of_18_l303_303411

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l303_303411


namespace factor_quadratic_l303_303936

theorem factor_quadratic (x : ℝ) : (16 * x^2 - 40 * x + 25) = (4 * x - 5)^2 :=
by 
  sorry

end factor_quadratic_l303_303936


namespace find_a_of_quadratic_intercepts_l303_303994

noncomputable def quadratic_intercepts_distance (a : ℝ) (h : a ≠ 0) : ℝ :=
  let b := -4 * a in
  let c := 8 in
  real.sqrt ((-b / a)^2 - 4 * c / a)

theorem find_a_of_quadratic_intercepts (a : ℝ) (h : a ≠ 0) :
  quadratic_intercepts_distance a h = 6 → a = -8 / 5 := 
sorry

end find_a_of_quadratic_intercepts_l303_303994


namespace value_of_b_over_a_l303_303800

def rectangle_ratio (a b : ℝ) : Prop :=
  let d := Real.sqrt (a^2 + b^2)
  let P := 2 * (a + b)
  (b / d) = (d / (a + b))

theorem value_of_b_over_a (a b : ℝ) (h : rectangle_ratio a b) : b / a = 1 :=
by sorry

end value_of_b_over_a_l303_303800


namespace cos_over_sin_eq_ten_thirteen_l303_303062

theorem cos_over_sin_eq_ten_thirteen
  (α : ℝ)
  (h1 : cos (π / 4 - α) = 12 / 13)
  (h2 : 0 < α ∧ α < π / 4) :
  (cos (2 * α)) / (sin (π / 4 + α)) = 10 / 13 :=
by
  sorry

end cos_over_sin_eq_ten_thirteen_l303_303062


namespace sin_minus_cos_l303_303969

variable (α : ℝ)
variable h1 : -π/2 < α ∧ α < 0
variable h2 : sin α + cos α = 1/5

theorem sin_minus_cos (h1 : -π/2 < α ∧ α < 0) (h2 : sin α + cos α = 1/5) : sin α - cos α = -7/5 :=
sorry

end sin_minus_cos_l303_303969


namespace A_eq_B_l303_303960

def A (n m : ℕ) : ℕ :=
  finset.card { f : fin n → fin m | ∀ i j, i < j → f j - f i ≤ j - i }

def B (n m : ℕ) : ℕ :=
  finset.card { g : fin (2 * n + m + 1) → fin (m + 1) | 
                g 0 = 0 ∧ 
                g (2 * n + m) = m ∧ 
                ∀ i, 1 ≤ i → |g i - g (i - 1)| = 1 }

theorem A_eq_B (n m : ℕ) (hn : 1 ≤ n) (hm : 1 ≤ m) : A n m = B n m := 
  sorry

end A_eq_B_l303_303960


namespace product_of_divisors_of_18_l303_303316

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303316


namespace measure_opposite_angle_eq_30_l303_303764

noncomputable def triangle_opposite_angle (A B C : Type) [EuclideanGeometry A] (R : ℝ) (O : A → A → ℝ)
  (h : is_circumcircle_center O A B C) (side_eq_radius : O = R) : ℝ :=
  30 -- degrees

theorem measure_opposite_angle_eq_30 (A B C : Type) [EuclideanGeometry A] (R : ℝ) (O : A → A → ℝ)
  (h : is_circumcircle_center O A B C) (side_eq_radius : O = R):
  triangle_opposite_angle A B C R O h side_eq_radius = 30 := 
sorry

end measure_opposite_angle_eq_30_l303_303764


namespace always_possible_rotate_14_teeth_gears_l303_303850

theorem always_possible_rotate_14_teeth_gears :
  ∀ (teeth : ℕ), teeth = 14 → 
  ∃ (r : ℕ), r ≤ 13 ∧ ∀ i, (i + r) % 14 ∉ {0,1,2,3,5,6,7,8,9,10,11,12} :=
by sorry

end always_possible_rotate_14_teeth_gears_l303_303850


namespace value_standard_deviations_from_mean_l303_303783

-- Define the mean (µ)
def μ : ℝ := 15.5

-- Define the standard deviation (σ)
def σ : ℝ := 1.5

-- Define the value X
def X : ℝ := 12.5

-- Prove that the Z-score is -2
theorem value_standard_deviations_from_mean : (X - μ) / σ = -2 := by
  sorry

end value_standard_deviations_from_mean_l303_303783


namespace arithmetic_seq_geometric_seq_conditions_l303_303609

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 + n * d

-- Define the sum of the first n terms of the sequence
def sum_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n.succ * (a 0 + a n) / 2)

-- Define the geometric sequence condition for a1, a4, a6
def geometric_seq_cond (a : ℕ → ℝ) : Prop :=
  a 0 * a 5 = (a 3) ^ 2

-- The main goal translated into Lean 4
theorem arithmetic_seq_geometric_seq_conditions
  (a : ℕ → ℝ) (d : ℝ) (h_seq : arithmetic_seq a d)
  (h_geo : geometric_seq_cond a) (h_d : d ≠ 0) :
  a 0 = -9 * d ∧ sum_seq a 18 = 0 ∧
  (d < 0 → ∀ n, sum_seq a 8 = sum_seq a n → n ≤ 8) ∧
  (d > 0 → ∀ n, sum_seq a 9 = sum_seq a n → n ≥ 9) :=
sorry

end arithmetic_seq_geometric_seq_conditions_l303_303609


namespace min_rubles_needed_to_reverse_strip_l303_303708

-- Definition of the problem
def strip_length := 100

-- Conditions
def can_swap_adjacent_for_ruble (i j : Nat) : Prop := 
  abs (i - j) = 1

def can_swap_three_apart_for_free (i j : Nat) : Prop := 
  abs (i - j) = 4

-- Minimum rubles required to reverse the tokens
def min_rubles_to_reverse (n : Nat) : Nat :=
  if n = 100 then 50 else sorry

-- The statement to be proven
theorem min_rubles_needed_to_reverse_strip : min_rubles_to_reverse strip_length = 50 := 
by sorry

end min_rubles_needed_to_reverse_strip_l303_303708


namespace hyperbola_b_value_l303_303995

theorem hyperbola_b_value (b : ℝ) (h₁ : b > 0) 
  (h₂ : ∃ x y, x^2 - (y^2 / b^2) = 1 ∧ (∀ (c : ℝ), c = Real.sqrt (1 + b^2) → c / 1 = 2)) : b = Real.sqrt 3 :=
by { sorry }

end hyperbola_b_value_l303_303995


namespace count_edge_cubes_l303_303876

/-- 
A cube is painted red on all faces and then cut into 27 equal smaller cubes.
Prove that the number of smaller cubes that are painted on only 2 faces is 12. 
-/
theorem count_edge_cubes (c : ℕ) (inner : ℕ)  (edge : ℕ) (face : ℕ) :
  (c = 27 ∧ inner = 1 ∧ edge = 12 ∧ face = 6) → edge = 12 :=
by
  -- Given the conditions from the problem statement
  sorry

end count_edge_cubes_l303_303876


namespace number_that_divides_and_leaves_remainder_54_l303_303115

theorem number_that_divides_and_leaves_remainder_54 :
  ∃ n : ℕ, n > 0 ∧ (55 ^ 55 + 55) % n = 54 ∧ n = 56 :=
by
  sorry

end number_that_divides_and_leaves_remainder_54_l303_303115


namespace correct_statement_proof_l303_303451

noncomputable def ab_condition (a b : ℝ) : Prop := a * b - 8 = 12.25

def composite_number (n : ℕ) : Prop := n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem correct_statement_proof :
  ∃ (n : ℕ), composite_number (n) ∧ (n, 1) ∧ (n, d) ∧ (1, d) := sorry

end correct_statement_proof_l303_303451


namespace altitude_length_l303_303973

theorem altitude_length {s t : ℝ} 
  (A B C : ℝ × ℝ) 
  (hA : A = (-s, s^2))
  (hB : B = (s, s^2))
  (hC : C = (t, t^2))
  (h_parabola_A : A.snd = (A.fst)^2)
  (h_parabola_B : B.snd = (B.fst)^2)
  (h_parabola_C : C.snd = (C.fst)^2)
  (hyp_parallel : A.snd = B.snd)
  (right_triangle : (t + s) * (t - s) + (t^2 - s^2)^2 = 0) :
  (s^2 - (t^2)) = 1 :=
by
  sorry

end altitude_length_l303_303973


namespace ln_quadratic_decreasing_interval_l303_303278

noncomputable def decreasing_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

theorem ln_quadratic_decreasing_interval :
  decreasing_interval (λ x, Real.log (-x^2 - 2 * x + 8)) (-1) 2 :=
sorry

end ln_quadratic_decreasing_interval_l303_303278


namespace x_plus_q_in_terms_of_q_l303_303114

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x - 5| = q) (h2 : x > 5) : x + q = 2 * q + 5 :=
by
  sorry

end x_plus_q_in_terms_of_q_l303_303114


namespace find_c_l303_303143

theorem find_c (a c : ℝ) (ha : a > 1) (h_li_point : ∃ x : ℝ, (y = a * x^2 - 7 * x + c) ∧ (y = -x)) :
  0 < c ∧ c < 9 :=
by {
  -- Definitions and conditions are established here:
  -- "Li point" definition:
  let li_point := λ (x y : ℝ), x * y < 0,
  -- Parabola equation:
  let parabola := λ (x : ℝ), a * x^2 - 7 * x + c,
  -- "Li point" on the parabola condition:
  have h1 : (∃ x : ℝ, (li_point x (parabola x)) ∧ (parabola x = -x)),
  exact h_li_point,
  -- Given condition:
  have ha_positive : a > 1 := ha,
  -- Goal proof:
  sorry
}

end find_c_l303_303143


namespace product_of_divisors_18_l303_303355

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l303_303355


namespace prism_cut_similar_triples_l303_303483

theorem prism_cut_similar_triples :
  ∃! (s : ℕ × ℕ × ℕ), s.1.1 ≤ s.1.2 ∧ s.1.2 ≤ s.2 ∧ s.1.2 = 2023 ∧ 
  let a := s.1.1, b := s.1.2, c := s.2 in 
  (∃ (k : ℕ), k * a = 2023 ∧ 2023 * a = k * c)
  ∧ (a * c = 2023^2)
  ∧ 7 = ∏ (factors of 2023^2), s.1.1 < s.2) :=
sorry

end prism_cut_similar_triples_l303_303483


namespace cost_equivalence_at_325_l303_303665

def cost_plan1 (x : ℕ) : ℝ := 65 + 0.40 * x
def cost_plan2 (x : ℕ) : ℝ := 0.60 * x

theorem cost_equivalence_at_325 : cost_plan1 325 = cost_plan2 325 :=
by sorry

end cost_equivalence_at_325_l303_303665


namespace product_of_divisors_of_18_l303_303335

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l303_303335


namespace euler_four_i_in_third_quadrant_l303_303559

theorem euler_four_i_in_third_quadrant :
  let eix (x : ℝ) : ℂ := Complex.exp (x * Complex.I)
  in eix 4 = Complex.cos 4 + Complex.sin 4 * Complex.I →
  (π < 4 ∧ 4 < 3 * π / 2) →
  (Complex.cos 4 < 0 ∧ Complex.sin 4 < 0) →
  eix 4 ∈ {z : ℂ | z.re < 0 ∧ z.im < 0} :=
by
  intro eix_def exp_cond trig_cond
  sorry

end euler_four_i_in_third_quadrant_l303_303559


namespace range_of_a_min_g_interval_inequality_f_l303_303646

-- Define the functions f and g
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * log (x + 2)
def g (x : ℝ) : ℝ := x * exp x

-- Problem instances
-- 1. Range of a for two extreme points
theorem range_of_a (x1 x2 a : ℝ) (h1 : x1 < x2) (h2 : ∀ x, f'(x) = 2*x - a/(x + 2)) (h3 : 2*x1*x2 + 4*(x1 + x2) = a) :
  -2 < a ∧ a < 0 := sorry

-- 2. Minimum value of g in the interval (-2, 0)
theorem min_g_interval : ∃ x ∈ Ioo (-2 : ℝ) 0, ∀ y ∈ Ioo (-2 : ℝ) 0, g y ≥ g x ∧ g x = -1 / exp 1 := sorry

-- 3. Inequality for f(x1) / x2 < -1
theorem inequality_f (x1 x2 a : ℝ) (h1 : x1 < x2) (h2 : f'(x1) = 0 ∧ f'(x2) = 0) (h3 : 2*x1*x2 + 4*(x1 + x2) = a) :
  (f x1 a) / x2 < -1 := sorry

end range_of_a_min_g_interval_inequality_f_l303_303646


namespace smallest_k_for_coloring_l303_303744

theorem smallest_k_for_coloring (n : ℕ) (hn : n > 0) : 
  ∃ (k : ℕ), (∀ (coloring : (Σ (i j : fin (2 * n)), fin n) → fin n), 
  ∃ (c1 c2 r1 r2 : fin (2 * n)), 
  c1 ≠ c2 ∧ r1 ≠ r2 ∧ coloring ⟨c1, r1⟩ = coloring ⟨c1, r2⟩ ∧ 
  coloring ⟨c1, r1⟩ = coloring ⟨c2, r1⟩ ∧ 
  coloring ⟨c2, r2⟩ = coloring ⟨c2, r1⟩ ∧ 
  coloring ⟨c1, r2⟩ = coloring ⟨c2, r2⟩) 
  ∧ k = 2 * n^2 - n + 1 :=
begin
  sorry
end

end smallest_k_for_coloring_l303_303744


namespace find_speed_of_train_l303_303488

noncomputable def train_speed_given_conditions
  (length_of_train : ℝ)
  (time_to_cross: ℝ)
  (speed_of_man_kmh : ℝ)
  : ℝ :=
  let speed_of_man_ms := speed_of_man_kmh * (1000 / 3600) in
  let relative_speed_ms := length_of_train / time_to_cross in
  let speed_of_train_ms := relative_speed_ms - speed_of_man_ms in
  speed_of_train_ms * (3600 / 1000)

theorem find_speed_of_train :
  train_speed_given_conditions 200 8 8 ≈ 82 := 
begin
  sorry
end

end find_speed_of_train_l303_303488


namespace flag_43_is_red_flag_67_is_blue_l303_303801

def flag_color (n : ℕ) : string :=
  match n % 7 with
  | 0 => "yellow"
  | 1 => "red"
  | 2 => "red"
  | 3 => "red"
  | 4 => "blue"
  | 5 => "blue"
  | 6 => "yellow"
  | _ => "error" -- This case is impossible but needed to satisfy the exhaustiveness checker.

theorem flag_43_is_red : flag_color 43 = "red" :=
by
  sorry

theorem flag_67_is_blue : flag_color 67 = "blue" :=
by
  sorry

end flag_43_is_red_flag_67_is_blue_l303_303801


namespace cos_alpha_eq_neg_half_l303_303123

-- Given: α, β, γ form a geometric sequence with a common ratio of 2
--       sin(α), sin(β), sin(γ) also form a geometric sequence
-- Prove: cos(α) = -1/2

theorem cos_alpha_eq_neg_half (α β γ : ℝ) (h1 : β = 2 * α) (h2 : γ = 4 * α)
(h3 : sin β * sin β = sin α * sin γ) : cos α = -1 / 2 :=
begin
  sorry -- Proof to be provided
end

end cos_alpha_eq_neg_half_l303_303123


namespace trigonometric_identity_l303_303598

open Real

theorem trigonometric_identity (α : ℝ) (h1 : tan α = 4/3) (h2 : 0 < α ∧ α < π / 2) :
  sin (π + α) + cos (π - α) = -7/5 :=
by
  sorry

end trigonometric_identity_l303_303598


namespace product_of_divisors_of_18_l303_303310

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303310


namespace estimation_correct_l303_303691

-- Definitions corresponding to conditions.
def total_population : ℕ := 10000
def surveyed_population : ℕ := 200
def aware_surveyed : ℕ := 125

-- The proportion step: 125/200 = x/10000
def proportion (aware surveyed total_pop : ℕ) : ℕ :=
  (aware * total_pop) / surveyed

-- Using this to define our main proof goal
def estimated_aware := proportion aware_surveyed surveyed_population total_population

-- Final proof statement
theorem estimation_correct :
  estimated_aware = 6250 :=
sorry

end estimation_correct_l303_303691


namespace binom_sum_mod_1000_l303_303521

theorem binom_sum_mod_1000 : 
  (∑ i in (finset.range 2012).filter (λ i, i % 4 = 0), nat.choose 2011 i) % 1000 = 15 :=
sorry

end binom_sum_mod_1000_l303_303521


namespace countValidNumbers_l303_303670

def isValidDigit (d : ℕ) : Prop := d ≤ 3

def isValidNumber (N : ℕ) : Prop :=
  let digits := List.reverse (Nat.digits 10 N)
  digits.length = 5 ∧
  List.all digits isValidDigit ∧
  digits.head ≠ 0

def satisfiesConditions (N : ℕ) : Prop :=
  Nat.gcd N 15 = 1 ∧ Nat.gcd N 20 = 1

theorem countValidNumbers : 
  (List.filter (λ N, isValidNumber N ∧ satisfiesConditions N) (List.range 100000)).length = 256 :=
by
  sorry

end countValidNumbers_l303_303670


namespace winning_strategy_l303_303745

/-- 
For any integer n ≥ 3, we have the following result:
- If n is odd, Alice has a winning strategy.
- If n is even, Bob has a winning strategy.
-/ 
theorem winning_strategy (n : ℕ) (h : n ≥ 3) : 
  (n % 2 = 1 → ∃ strategy_for_alice : winning_strategy_for_alice, 
    strategy_for_alice) ∧ 
  (n % 2 = 0 → ∃ strategy_for_bob : winning_strategy_for_bob, 
    strategy_for_bob) :=
sorry

end winning_strategy_l303_303745


namespace trig_identity_l303_303562

theorem trig_identity (α : ℝ) (h : 2 * Real.cos α + 1 ≠ 0) :
  (Real.cos (3 * α) + Real.cos (4 * α) + Real.cos (5 * α)) /
  (Real.sin (3 * α) + Real.sin (4 * α) + Real.sin (5 * α)) =
  Real.cot (4 * α) :=
by
  sorry

end trig_identity_l303_303562


namespace determine_relationship_accurately_l303_303844

-- Conditions
variable (categorical_relationship_accurately_determined : String → Prop)
variable (3D_bar_chart_visual_reflection : Prop)
variable (3D_bar_chart_not_accurate : Prop)
variable (independence_test_stat_method : Prop)
variable (independence_test_accurate_determination : Prop)

-- Definitions
def method_options := ["3D_bar_chart", "independence_test"]
def accurate_method := "independence_test"

-- Main Statement
theorem determine_relationship_accurately :
  (3D_bar_chart_visual_reflection → 3D_bar_chart_not_accurate) →
  (independence_test_stat_method ∧ independence_test_accurate_determination) →
  categorical_relationship_accurately_determined accurate_method :=
by
  sorry

end determine_relationship_accurately_l303_303844


namespace math_problem_l303_303068

open Real

variables (a b b1 : ℝ) (F1 F2 M : ℝ × ℝ)
variables (C1 C2 : Set (ℝ × ℝ))

-- Defining the foci and intersection point
def F1 := (-1, 0)
def F2 := (1, 0)
def M := (2 * sqrt 3 / 3, sqrt 3 / 3)

-- Ellipse C1 and Hyperbola C2
def is_ellipse (C : Set (ℝ × ℝ)) := ∃ a b, a > 0 ∧ b > 0 ∧ a > b ∧
  (∀ p ∈ C, (p.1^2 / a^2) + (p.2^2 / b^2) = 1)
def is_hyperbola (C : Set (ℝ × ℝ)) := ∃ b1, b1 > 0 ∧
  (∀ p ∈ C, (p.1^2) - (p.2^2 / b1^2) = 1)

-- Conditions of the problem
def intersection_condition (p : ℝ × ℝ) :=
  p ∈ C1 ∧ p ∈ C2 ∧ p = M

def hyperbola_vertices := F1 ∈ C2 ∧ F2 ∈ C2

-- Fixed point condition
def fixed_point_condition := 
  ∃ P Q : ℝ × ℝ, 
    (P.1^2 - P.2^2 = 1) ∧ 
    (Q.1 = 0) ∧ 
    (F1.2 ≠ Q.2) ∧ 
    (P.2 / (P.1 + 1) = - (Q.1 + 1) / (Q.2)) ∧ 
    (∃ K : ℝ × ℝ, line_through R P Q = line_through R (1, 0))

theorem math_problem :
  is_ellipse C1 ∧ is_hyperbola C2 ∧ intersection_condition M ∧ hyperbola_vertices → fixed_point_condition := 
by 
  sorry

end math_problem_l303_303068


namespace tangent_line_eqn_l303_303577

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem tangent_line_eqn :
  let slope := (deriv f 2),
      point := (2, f 2)
  in
  slope = 1 / 2 ∧ point = (2, Real.log 2) ∧
  ∀ (x y : ℝ), y - f(2) = slope * (x - 2) ↔ x - 2 * y + 2 * Real.log 2 - 2 = 0 :=
by
  sorry

end tangent_line_eqn_l303_303577


namespace count_valid_numbers_l303_303838

open Finset

def digit_set : Finset ℕ := {0, 1, 2, 3, 4, 5}

def is_valid_number (n : ℕ) : Prop :=
  40000 < n ∧ n < 100000 ∧
  (∀ i, i < 5 → ∃ d ∈ digit_set, d = (n / 10^i) % 10) ∧
  (∀ i j, i < 5 → j < 5 → i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧
  (n % 2 = 0)

theorem count_valid_numbers :
  ∃! n, ∃ s : Finset ℕ, s.card = 120 ∧
  ∀ x ∈ s, x ∈ {x : ℕ | is_valid_number x}.to_finset :=
sorry

end count_valid_numbers_l303_303838


namespace second_quadrant_points_l303_303055

theorem second_quadrant_points (x y : ℤ) (P : ℤ × ℤ) :
  P = (x, y) ∧ x < 0 ∧ y > 0 ∧ y ≤ x + 4 →
  P ∈ {(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-3, 1)} :=
by
  sorry

end second_quadrant_points_l303_303055


namespace locus_of_E_is_hyperbola_center_l303_303049

def is_hyperbola_center (a b : ℝ) (d : ℝ) (α : ℝ) : Prop :=
  b = 0 ∧ a = (2 * d) / (Real.sin α)^2

theorem locus_of_E_is_hyperbola_center
  (a₁ b₁ : ℝ)
  (a b m n d : ℝ)
  (h_origin : a₁ = 0 ∧ b₁ = 0)
  (h_intersect_e : ∀ x, e x ↔ x = d)
  (h_axes_eq : ∀ x, (a x ↔ x = b * x) ∧ (b x ↔ x = m * x))
  (h_tanα : Real.tan α = (m - n) / (1 + m * n)) :
  is_hyperbola_center a b d α :=
by
  sorry

end locus_of_E_is_hyperbola_center_l303_303049


namespace evaluate_expression_l303_303561

theorem evaluate_expression : sqrt (18 - 8 * sqrt 2) + sqrt (18 + 8 * sqrt 2) = 8 := 
by
  sorry

end evaluate_expression_l303_303561


namespace felicity_gasoline_usage_l303_303572

def gallons_of_gasoline (G D: ℝ) :=
  G = 2 * D

def combined_volume (M D: ℝ) :=
  M = D - 5

def ethanol_consumption (E M: ℝ) :=
  E = 0.35 * M

def biodiesel_consumption (B M: ℝ) :=
  B = 0.65 * M

def distance_relationship_F_A (F A: ℕ) :=
  A = F + 150

def distance_relationship_F_Bn (F Bn: ℕ) :=
  F = Bn + 50

def total_distance (F A Bn: ℕ) :=
  F + A + Bn = 1750

def gasoline_mileage : ℕ := 35

def diesel_mileage : ℕ := 25

def ethanol_mileage : ℕ := 30

def biodiesel_mileage : ℕ := 20

theorem felicity_gasoline_usage : 
  ∀ (F A Bn: ℕ) (G D M E B: ℝ),
  gallons_of_gasoline G D →
  combined_volume M D →
  ethanol_consumption E M →
  biodiesel_consumption B M →
  distance_relationship_F_A F A →
  distance_relationship_F_Bn F Bn →
  total_distance F A Bn →
  G = 56
  := by
    intros
    sorry

end felicity_gasoline_usage_l303_303572


namespace find_DF_l303_303139

variable (ABCD : Type) [Parallelogram ABCD]
variable (AB BC CD DA : ℝ) (DE DF : ℝ)
variable (E F : ABCD)

-- Conditions
variable (h_DC : CD = 20)
variable (h_EB : EB = 5)
variable (h_DE : DE = 8)
-- \(DF\) is the altitude to the base \(BC\)
variable (h_altitude_DE : is_altitude DE AB)
variable (h_altitude_DF : is_altitude DF BC)

-- Prove DF = 8
theorem find_DF (h : Parallelogram ABCD)
  (h1: DC = 20)
  (h2: EB = 5)
  (h3: DE = 8)
  (altitude_DE : is_altitude DE AB)
  (altitude_DF : is_altitude DF BC) : DF = 8 :=
sorry

end find_DF_l303_303139


namespace repeating_decimal_as_fraction_l303_303935

/-- Define x as the repeating decimal 7.182182... -/
def x : ℚ := 
  7 + 182 / 999

/-- Define y as the fraction 7175/999 -/
def y : ℚ := 
  7175 / 999

/-- Theorem stating that the repeating decimal 7.182182... is equal to the fraction 7175/999 -/
theorem repeating_decimal_as_fraction : x = y :=
sorry

end repeating_decimal_as_fraction_l303_303935


namespace rectangle_angle_end_l303_303219

theorem rectangle_angle_end (EF FG : ℝ) (HD : FG = 4) (EF_eq : EF = 8) (N : Point) (HN : N ∈ segment E F) 
(END_equal_FND : ∠END = ∠FND) : ∠END = 45 :=
by
  sorry

end rectangle_angle_end_l303_303219


namespace polygon_sides_count_l303_303499

theorem polygon_sides_count :
    ∀ (n1 n2 n3 n4 n5 n6 : ℕ),
    n1 = 3 ∧ n2 = 4 ∧ n3 = 5 ∧ n4 = 6 ∧ n5 = 7 ∧ n6 = 8 →
    (n1 - 2) + (n2 - 2) + (n3 - 2) + (n4 - 2) + (n5 - 2) + (n6 - 1) + 3 = 24 :=
by
  intros n1 n2 n3 n4 n5 n6 h
  sorry

end polygon_sides_count_l303_303499


namespace product_of_divisors_18_l303_303323

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l303_303323


namespace product_of_divisors_18_l303_303320

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l303_303320


namespace no_discrepancy_l303_303819

-- Definitions based on the conditions
def t1_hours : ℝ := 1.5 -- time taken clockwise in hours
def t2_minutes : ℝ := 90 -- time taken counterclockwise in minutes

-- Lean statement to prove the equivalence
theorem no_discrepancy : t1_hours * 60 = t2_minutes :=
by sorry

end no_discrepancy_l303_303819


namespace planting_rate_l303_303569

theorem planting_rate (total_acres : ℕ) (days : ℕ) (initial_tractors : ℕ) (initial_days : ℕ) (additional_tractors : ℕ) (additional_days : ℕ) :
  total_acres = 1700 →
  days = 5 →
  initial_tractors = 2 →
  initial_days = 2 →
  additional_tractors = 7 →
  additional_days = 3 →
  (total_acres / ((initial_tractors * initial_days) + (additional_tractors * additional_days))) = 68 :=
by
  sorry

end planting_rate_l303_303569


namespace parallelogram_base_length_l303_303782

theorem parallelogram_base_length (A h : ℕ) (hA : A = 32) (hh : h = 8) : (A / h) = 4 := by
  sorry

end parallelogram_base_length_l303_303782


namespace product_of_divisors_18_l303_303353

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l303_303353


namespace calculate_expression_l303_303910

theorem calculate_expression : 
  (Real.sqrt 3) ^ 0 + 2 ^ (-1:ℤ) + Real.sqrt 2 * Real.cos (Float.pi / 4) - Real.abs (-1/2) = 2 := 
by
  sorry

end calculate_expression_l303_303910


namespace sum_of_powers_l303_303906

theorem sum_of_powers : 5^5 + 5^5 + 5^5 + 5^5 = 4 * 5^5 :=
by
  sorry

end sum_of_powers_l303_303906


namespace proposition_D_l303_303059

variables (α β : Plane) (m n : Line)

-- Assume the necessary conditions
axiom perp_planes (h1 : α ⊥ β) : true
axiom perp_line_plane (h2 : m ⊥ β) : true
axiom not_in_plane (h3 : ¬ (m ⊆ α)) : true

-- The proposition to be proven
theorem proposition_D (h1 : α ⊥ β) (h2 : m ⊥ β) (h3 : ¬ (m ⊆ α)) : m ∥ α :=
sorry -- The proof is not required

end proposition_D_l303_303059


namespace weight_of_new_person_l303_303245

theorem weight_of_new_person 
  (avg_increase : Real)
  (num_persons : Nat)
  (old_weight : Real)
  (new_avg_increase : avg_increase = 2.2)
  (number_of_persons : num_persons = 15)
  (weight_of_old_person : old_weight = 75)
  : (new_weight : Real) = old_weight + avg_increase * num_persons := 
  by sorry

end weight_of_new_person_l303_303245


namespace find_a_and_a_n_find_sum_b_n_l303_303630

variable (a : ℕ) (a_n S_n : ℕ → ℕ)

-- Conditions
def geometric_sum_condition (n : ℕ) : Prop :=
  6 * S_n n = 3^(n + 1) + a

noncomputable def sequence_formula : ℕ → ℕ
| n := if n = 1 then a / 6 else 3^(n - 1)

-- Statement for part (I)
theorem find_a_and_a_n (n : ℕ) (h : geometric_sum_condition n) :
  a = 9 ∧ (∀ n, sequence_formula n = 3^(n-1)) :=
sorry

-- Conditions for part (II)
def b_n (n : ℕ) : ℝ :=
  (-1)^(n-1) * (2 * n^2 + 2 * n + 1) / ((real.log 3 (a_n n) + 2)^2 * (real.log 3 (a_n n) + 1)^2)

def sum_b_condition (n : ℕ) (T_n : ℕ → ℝ) : Prop :=
  (∀ k, b_n a_n k = sequence_formula k) → 
  (∀ n, T_n n = ∑ i in finset.range (n+1), b_n a_n i)

-- Statement for part (II)
theorem find_sum_b_n (T_n : ℕ → ℝ) (h : sum_b_condition a_n T_n) :
  T_n = (λ n, 1 + (-1)^(n-1) / (n+1)^2) :=
sorry

end find_a_and_a_n_find_sum_b_n_l303_303630


namespace third_side_length_count_l303_303683

theorem third_side_length_count : ∃ n : ℕ, n = 15 ∧
  (∀ x : ℝ, (2 < x ∧ x < 18) ↔ (floor x = x ∧ 3 ≤ x ∧ x ≤ 17)) :=
by
  sorry

end third_side_length_count_l303_303683


namespace number_of_mappings_l303_303178

noncomputable def X : Set ℕ := {i | 1 ≤ i ∧ i ≤ 10}

def f (x : ℕ) : ℕ

axiom f_composition : ∀ i ∈ X, f (f i) = i
axiom f_constraint : ∀ i ∈ X, abs (f i - i) ≤ 2

theorem number_of_mappings : ∃ (numMappings : ℕ), numMappings = 401 :=
by sorry

end number_of_mappings_l303_303178


namespace product_of_divisors_of_18_l303_303367

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303367


namespace distance_between_planes_l303_303948

theorem distance_between_planes :
  let plane1 (x y z : ℝ) := 2*x - 3*y + z - 4
  let plane2 (x y z : ℝ) := 4*x - 6*y + 2*z + 3
  ∃ d : ℝ, d = 11 * real.sqrt 14 / 28 ∧ 
    ∀ x y z : ℝ, plane1 x y z = 0 → plane2 x y z = plane2 2 0 0 → d = abs (4 * 2 + 3) / real.sqrt (4^2 + (-6)^2 + 2^2) :=
sorry

end distance_between_planes_l303_303948


namespace product_of_divisors_of_18_l303_303374

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303374


namespace function_extreme_values_l303_303251

theorem function_extreme_values:
  let y : ℝ → ℝ := λ x, sqrt (x - 4) + sqrt (15 - 3 * x) in
  ∀ x : ℝ, 4 ≤ x ∧ x ≤ 5 → (∃ z : ℝ, z = 1 ∨ z = sqrt 3) :=
begin
  intro y,
  sorry
end

end function_extreme_values_l303_303251


namespace find_rope_costs_l303_303557

theorem find_rope_costs (x y : ℕ) (h1 : 10 * x + 5 * y = 175) (h2 : 15 * x + 10 * y = 300) : x = 10 ∧ y = 15 :=
    sorry

end find_rope_costs_l303_303557


namespace Avery_builds_in_4_hours_l303_303903

variable (A : ℝ) (TomTime : ℝ := 2) (TogetherTime : ℝ := 1) (RemainingTomTime : ℝ := 0.5)

-- Conditions:
axiom Tom_builds_in_2_hours : TomTime = 2
axiom Work_together_for_1_hour : TogetherTime = 1
axiom Tom_finishes_in_0_5_hours : RemainingTomTime = 0.5

-- Question:
theorem Avery_builds_in_4_hours : A = 4 :=
by
  sorry

end Avery_builds_in_4_hours_l303_303903


namespace solve_stamps_l303_303592

noncomputable def stamps_problem : Prop :=
  ∃ (A B C D : ℝ), 
    A + B + C + D = 251 ∧
    A = 2 * B + 2 ∧
    A = 3 * C + 6 ∧
    A = 4 * D - 16 ∧
    D = 32

theorem solve_stamps : stamps_problem :=
sorry

end solve_stamps_l303_303592


namespace tan_double_angle_l303_303687

theorem tan_double_angle (α : ℝ) (x y : ℝ) (hxy : y / x = -2) : 
  2 * y / (1 - (y / x)^2) = (4 : ℝ) / 3 :=
by sorry

end tan_double_angle_l303_303687


namespace product_of_divisors_of_18_l303_303284

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l303_303284


namespace product_of_divisors_18_l303_303326

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l303_303326


namespace problem_statement_l303_303103

theorem problem_statement (x y : ℝ) (log2_3 log5_3 : ℝ)
  (h1 : log2_3 > 1)
  (h2 : 0 < log5_3)
  (h3 : log5_3 < 1)
  (h4 : log2_3^x - log5_3^x ≥ log2_3^(-y) - log5_3^(-y)) :
  x + y ≥ 0 := 
sorry

end problem_statement_l303_303103


namespace min_people_in_photographs_l303_303131

-- Definitions based on conditions
def photographs := (List (Nat × Nat × Nat))
def menInCenter (photos : photographs) := photos.map (fun (c, _, _) => c)

-- Condition: there are 10 photographs each with a distinct man in the center
def valid_photographs (photos: photographs) :=
  photos.length = 10 ∧ photos.map (fun (c, _, _) => c) = List.range 10

-- Theorem to be proved: The minimum number of different people in the photographs is at least 16
theorem min_people_in_photographs (photos: photographs) (h : valid_photographs photos) : 
  ∃ people : Finset Nat, people.card ≥ 16 := 
sorry

end min_people_in_photographs_l303_303131


namespace KLMN_parallelogram_l303_303176

variables {A B C D E F K L M N : Type}
variables [add_comm_group E] [vector_space ℝ E]
variables (A B C D E F K L M N : E)

open_locale classical

-- Conditions
def midpoint (P Q X : E) : Prop := X = (P + Q) / 2

-- Prove KLMN is a parallelogram
theorem KLMN_parallelogram
  (hE: midpoint A B E)
  (hF: midpoint C D F)
  (hK: midpoint A F K)
  (hL: midpoint C E L)
  (hM: midpoint B F M)
  (hN: midpoint D E N) :
  ∃ p q r s: E, K = p + q / 2 ∧ L = q + r / 2 ∧ M = r + s / 2 ∧ N = s + p / 2 :=
sorry

end KLMN_parallelogram_l303_303176


namespace printers_ratio_l303_303856

theorem printers_ratio (Rate_X : ℝ := 1 / 16) (Rate_Y : ℝ := 1 / 10) (Rate_Z : ℝ := 1 / 20) :
  let Time_X := 1 / Rate_X
  let Time_YZ := 1 / (Rate_Y + Rate_Z)
  (Time_X / Time_YZ) = 12 / 5 := by
  sorry

end printers_ratio_l303_303856


namespace total_balls_5000th_step_l303_303188

/-- 
  Mandy has an infinite number of balls and empty boxes available.
  Each box can hold up to five balls.
  The boxes are arrayed in a line.
  At the start, a ball is placed in the first box on the left. 
  At every following step, a ball is placed into the first box which can still receive a ball, 
  and any boxes to its left are emptied.
  Determine the total number of balls in the boxes after Mandy's 5000th step.
-/
theorem total_balls_5000th_step : 
  let initial_ball_first_box := 1
  let boxes_ball_capacity := 5
  let steps := 5000
  sum_of_digits_6_base steps = 13 :=
by
  sorry

/-- Function to convert a number to base-6 and sum its digits -/
def sum_of_digits_6_base (n : ℕ) : ℕ := sorry

end total_balls_5000th_step_l303_303188


namespace find_interest_rate_l303_303495
noncomputable def annualInterestRate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  P * (1 + r / n)^(n * t) = A

theorem find_interest_rate :
  annualInterestRate 5000 6050.000000000001 1 2 0.1 :=
by
  -- The proof goes here
  sorry

end find_interest_rate_l303_303495


namespace convex_hull_not_triangle_l303_303596
noncomputable theory

open Set

theorem convex_hull_not_triangle {n : ℕ} (n_pos : n > 0) :
  ∀ (points : Finset (ℝ × ℝ)), 
  points.card = 3 * n - 1 → 
  (∀ p1 p2 p3 ∈ points, affineIndependent ℝ [p1, p2, p3]) →
  ∃ (sub_points : Finset (ℝ × ℝ)), 
  sub_points ⊆ points ∧
  sub_points.card = 2 * n ∧
  ¬(exists p1 p2 p3 ∈ sub_points, 
    sub_points ⊆ convexHull ℝ {p1, p2, p3}) :=
by
  intro points h_card h_independent
  sorry

end convex_hull_not_triangle_l303_303596


namespace find_value_of_T_l303_303741

theorem find_value_of_T (S : ℕ) (hS : S = 120) : 
    let F := (Finset.range (S + 1)).sum (λ n, 2^n)
    let T := Real.sqrt ((Real.log (1 + F)) / (Real.log 2))
    in T = 11 :=
by
    have hF : F = 2^(S + 1) - 1 := sorry
    rw hS at hF
    rw hF
    have hT : T = Real.sqrt (121 * (Real.log 2 / Real.log 2)) := sorry
    have hLog : Real.log 2 ≠ 0 := sorry
    have hSimpl : T = Real.sqrt 121 := sorry
    have hSqrt : Real.sqrt 121 = 11 := sorry
    exact hSqrt

end find_value_of_T_l303_303741


namespace range_of_x_l303_303035

theorem range_of_x (x a : ℝ) (h1 : a ∈ set.Icc (-1 : ℝ) 1) (h2 : x ∈ set.Ioo (-∞ : ℝ) 1 ∪ set.Ioo 3 (∞ : ℝ)) :
  x^2 + (a - 4)*x + 4 - 2*a > 0 :=
by 
  sorry

end range_of_x_l303_303035


namespace sum_of_squares_of_perpendicular_chords_constant_l303_303972

theorem sum_of_squares_of_perpendicular_chords_constant
  (R : ℝ) (K : Point)
  (AB CD : Chord)
  (h1 : AB ⊥ CD)
  (h2 : AB.intersect K)
  (h3 : CD.intersect K)
  (h4 : AB.on_circle)
  (h5 : CD.on_circle)
  : ∃ k : ℝ, AB.length ^ 2 + CD.length ^ 2 = k := 
sorry

end sum_of_squares_of_perpendicular_chords_constant_l303_303972


namespace exists_point_on_graph_of_quadratic_l303_303213

-- Define the condition for the discriminant to be zero
def is_single_root (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c = 0

-- Define a function representing a quadratic polynomial
def quadratic_poly (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- The main statement
theorem exists_point_on_graph_of_quadratic (b c : ℝ) 
  (h : is_single_root 1 b c) :
  ∃ (p q : ℝ), q = (p^2) / 4 ∧ is_single_root 1 p q :=
sorry

end exists_point_on_graph_of_quadratic_l303_303213


namespace least_subtract_divisible_l303_303460

theorem least_subtract_divisible:
  ∃ n : ℕ, n = 31 ∧ (13603 - n) % 87 = 0 :=
by
  sorry

end least_subtract_divisible_l303_303460


namespace product_of_all_positive_divisors_of_18_l303_303387

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l303_303387


namespace inequality_abc_l303_303199

theorem inequality_abc (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : a^2 + b^2 + c^2 = 1) : 
  (ab / c) + (bc / a) + (ca / b) ≥ real.sqrt 3 :=
begin
  sorry
end

end inequality_abc_l303_303199


namespace percentage_of_second_solution_l303_303101

theorem percentage_of_second_solution
  (x : ℕ) (a : ℤ) (b : ℕ)
  (h1 : x = 60) (h2 : a = 50) (h3 : b = 40)
  (h4 : 0.20 * ↑b + (a / 100) * ↑x = 0.50 * (↑b + ↑x)) :
  a = 50 :=
by sorry

end percentage_of_second_solution_l303_303101


namespace product_of_divisors_of_18_is_5832_l303_303402

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l303_303402


namespace segments_in_given_ratio_l303_303817

-- Definition of points and lines
section
variable (P : Point) (AM AN : Line)

-- Given condition: ratio k
variable (k : ℚ)

-- Statement of the proof problem
theorem segments_in_given_ratio (PB PC : Length) (B C : Point) :
  (PB = distance P B) → (PC = distance P C) → 
  (is_on_line B AM) → (is_on_line C AN) → 
  (is_line_through_points BC P) → 
  (∃ BC : Line, (PC / PB) = k) :=
by
  sorry
end

end segments_in_given_ratio_l303_303817


namespace odd_function_sufficient_but_not_necessary_l303_303124

def is_function_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_sufficient_but_not_necessary (f : ℝ → ℝ) (hdom : ∀ x : ℝ, x ∈ set.univ) :
  (is_function_odd f → f(0) = 0) ∧ (∃ g : ℝ → ℝ, (g 0 = 0) ∧ ¬ (is_function_odd g)) :=
by
  sorry

end odd_function_sufficient_but_not_necessary_l303_303124


namespace explicit_form_of_f_monotonic_increasing_intervals_range_of_a_l303_303642

noncomputable def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem explicit_form_of_f
  (h1 : f (1) = -5)
  (h2 : Df(1) = 0)
  (h3 : Df(-3) = 0)
  : f x = x^3 + 3*x^2 - 9*x
:= sorry

theorem monotonic_increasing_intervals
  (h1 : f (1) = -5)
  (h2 : Df(1) = 0)
  (h3 : Df(-3) = 0)
  (h4: ∀ x, Df (x) = 3 * x^2 + 6 * x - 9) 
  : ∀ (m : ℝ), [m, m + 1] ⊆ Ioi 1 ∨ [m, m + 1] ⊆ Ioi (-3) → (m ≤ -4 ∨ m ≥ 1)
:= sorry

theorem range_of_a
  (h1 : f (1) = -5)
  (h2 : Df(1) = 0)
  (h3 : Df(-3) = 0)
  : ∀ a, ∃ x y, x ≠ y ∧ f (x) = a ∧ f (y) = a ↔ -5 ≤ a ∧ a ≤ 27
:= sorry

end explicit_form_of_f_monotonic_increasing_intervals_range_of_a_l303_303642


namespace very_large_positive_number_evaluation_l303_303474

def sequence (i : ℕ) : ℕ :=
if i ≥ 1 ∧ i ≤ 6 then 2 * i
else if i > 6 then 
  let rec aux (n : ℕ) : ℕ :=
    if n < 1 then 1
    else sequence (n - 1) * aux (n - 1)
  in aux (i - 1) + 1
else 0

theorem very_large_positive_number_evaluation :
  let a := sequence in
  let product := List.foldl (*) 1 (List.map a (List.range 11)) in
  let sum_of_squares := List.sum (List.map (λ x => x ^ 2) (List.map a (List.range 11))) in
  product - sum_of_squares > 0 :=
by
  sorry

end very_large_positive_number_evaluation_l303_303474


namespace product_of_divisors_of_18_l303_303427

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303427


namespace min_value_of_f_l303_303250

noncomputable def f (a x : ℝ) : ℝ := a^(2 * x) + 3 * a^x - 2

theorem min_value_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) 
  (h₂ : ∃ x ∈ set.Icc (-1 : ℝ) 1, f a x = 8) :
  ∃ x ∈ set.Icc (-1 : ℝ) 1, f a x = -1 / 4 :=
sorry

end min_value_of_f_l303_303250


namespace product_of_divisors_of_18_l303_303423

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l303_303423


namespace rhombus_diagonal_l303_303790

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) 
  (h_d1 : d1 = 70) 
  (h_area : area = 5600): 
  (area = (d1 * d2) / 2) → d2 = 160 :=
by
  sorry

end rhombus_diagonal_l303_303790


namespace cube_triangle_area_sum_l303_303804

theorem cube_triangle_area_sum : 
  ∃ m n p : ℕ, 
    let sum_of_areas := 12 + 12 * Real.sqrt 2 + 4 * Real.sqrt 3 in
    m + Real.sqrt n + Real.sqrt p = sum_of_areas ∧ m + n + p = 348 :=
begin
  use [12, 288, 48],
  split,
  { simp only [Real.sqrt],
    norm_num },
  { norm_num }
end

end cube_triangle_area_sum_l303_303804


namespace max_value_a_plus_sqrt_ab_plus_cbrt_abc_l303_303168

theorem max_value_a_plus_sqrt_ab_plus_cbrt_abc (a b c : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) 
(h_nonneg_c : 0 ≤ c) (h_sum : a + b + c = 2) : 
  a + Real.sqrt (a * b) + Real.cbrt (a * b * c) ≤ 11 / 3 :=
sorry

end max_value_a_plus_sqrt_ab_plus_cbrt_abc_l303_303168


namespace product_of_divisors_of_18_l303_303436

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303436


namespace insect_eggs_base10_l303_303898

-- Defining the base-7 number 235 as a tuple of its digits
def base7_number : List ℕ := [2, 3, 5]

-- Function to convert a base-7 number to a base-10 number
def base7_to_base10 (digits : List ℕ) : ℕ :=
  digits.reverse.enum_from 0 |>.foldl (λ acc ⟨i, d⟩ => acc + d * 7^i) 0

-- The main theorem to prove
theorem insect_eggs_base10 : base7_to_base10 base7_number = 124 :=
by
  sorry

end insect_eggs_base10_l303_303898


namespace orthocenters_fixed_circle_l303_303057

variables (A B X Y Z : ℝ^3)
variable (Omega : ℝ^3 → Prop)

def equilateral_triangle (A B X : ℝ^3) : Prop :=
  dist A B = dist B X ∧ 
  dist B X = dist X A ∧ 
  ∃ θ, θ ≠ 0 ∧ θ ≠ π ∧ dist A B = dist B X * 2 * sin(θ / 2)

def square (A B Y Z : ℝ^3) : Prop :=
  dist A B = dist B Y ∧ dist B Y = dist Y Z ∧ dist Y Z = dist Z A ∧ 
  ∃ φ, φ = π / 2 ∧ dist A B = √(dist B A^2 + dist Y Z^2)

def orthocenter (X Y Z H : ℝ^3) : Prop :=
  ∀ H : ℝ^3, ∃ l₁ l₂ l₃ : ℝ^3, 
  is_line l₁ X Y ∧ 
  is_line l₂ Y Z ∧ 
  is_line l₃ Z X ∧ 
  H ∈ l₁ ∧ H ∈ l₂ ∧ H ∈ l₃

theorem orthocenters_fixed_circle :
  ∀ (A B X Y Z : ℝ^3), 
  (equilateral_triangle A B X) ∧ 
  (square A B Y Z) → 
  ∃ Omega, ∀ H, orthocenter X Y Z H → Omega H :=
sorry

end orthocenters_fixed_circle_l303_303057


namespace sum_of_series_l303_303629

theorem sum_of_series (n : ℕ) : 
  (finset.range n).sum (λ k, 1 / ((k + 1) * (k + 2) : ℝ)) = n / (n + 1) :=
by sorry

end sum_of_series_l303_303629


namespace find_a_l303_303121

theorem find_a (a x y : ℝ) (h1 : ax - 3y = 0) (h2 : x + y = 1) (h3 : 2x + y = 0) : a = -6 := 
by sorry

end find_a_l303_303121


namespace calculate_expression_l303_303915

theorem calculate_expression :
  (Real.sqrt 3) ^ 0 + 2 ^ (-1 : ℤ) + Real.sqrt 2 * Real.cos (Real.pi / 4) - |(-1:ℝ) / 2| = 2 := 
by
  sorry

end calculate_expression_l303_303915


namespace product_of_divisors_of_18_l303_303287

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l303_303287


namespace max_children_with_distinct_candies_l303_303812

theorem max_children_with_distinct_candies (n : ℕ) (candies : ℕ) :
  candies = 20 → (∀ i j : ℕ, i ≠ j → distinct_candies i j) → n ≤ 5 :=
by sorry

-- Auxiliary function to represent the distinct_candies condition
def distinct_candies (i j : ℕ) : Prop :=
  i ≠ j

end max_children_with_distinct_candies_l303_303812


namespace inequality_sqrt_three_l303_303207

theorem inequality_sqrt_three (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h : a^2 + b^2 + c^2 = 1) : 
  (ab / c + bc / a + ca / b) ≥ √3 :=
by
  sorry

end inequality_sqrt_three_l303_303207


namespace find_a_prove_interval_l303_303641

noncomputable def f (x : ℝ) (a : ℝ) := a + log (x^2)

def condition (x a : ℝ) : Prop := f x a ≤ a * |x|

theorem find_a (H : ∀ x : ℝ, condition x 2) : ∀ a : ℝ, a = 2 :=
by sorry

noncomputable def g (x : ℝ) (a : ℝ) := x * f x a / (x - a)

theorem prove_interval (H : ∀ x : ℝ, condition x 2) (hmin : ∃ m : ℝ, ∀ x > 2, g x 2 ≥ g m 2) : 6 < f (classical.some hmin) 2 ∧ f (classical.some hmin) 2 < 7 :=
by sorry

end find_a_prove_interval_l303_303641


namespace sin_sequence_convergence_l303_303742

theorem sin_sequence_convergence (a : ℝ) (a_seq : ℕ → ℝ) (h₁ : a_seq 1 = a) (h₂ : ∀ n, a_seq (n + 1) = (n + 1) * (a_seq n)) :
  ¬(∀ n, sinus (a_seq n) converges → ∃ N, ∀ m > N, sinus (a_seq m) = 0) :=
sorry

end sin_sequence_convergence_l303_303742


namespace product_of_divisors_18_l303_303296

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l303_303296


namespace problem_statement_l303_303599

def f (α : ℝ) : ℝ := 
  (Real.cos (π / 2 + α) * Real.sin (3 * π / 2 - α)) / 
  (Real.cos (-π - α) * Real.tan (π - α))

theorem problem_statement :
  f (-25 * π / 3) = 1 / 2 := by
  sorry

end problem_statement_l303_303599


namespace problem_proof_l303_303240

theorem problem_proof {n : ℕ} (h₀ : n ≥ 2) (x : Fin n → ℝ) 
  (h₁ : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  ∃ i : Fin (n-1), x i * (1 - x ⟨i.1 + 1, Nat.lt_of_lt_pred i.2⟩) ≥ (1 / 4) * x 0 * (1 - x ⟨n - 1, Nat.pred_lt (Nat.succ_le_of_lt h₀)⟩) := 
begin
  sorry
end

end problem_proof_l303_303240


namespace num_cubes_with_more_than_one_blue_face_l303_303895

-- Define the parameters of the problem
def block_length : ℕ := 5
def block_width : ℕ := 3
def block_height : ℕ := 1

def total_cubes : ℕ := 15
def corners : ℕ := 4
def edges : ℕ := 6
def middles : ℕ := 5

-- Define the condition that the total number of cubes painted on more than one face.
def cubes_more_than_one_blue_face : ℕ := corners + edges

-- Prove that the number of cubes painted on more than one face is 10
theorem num_cubes_with_more_than_one_blue_face :
  cubes_more_than_one_blue_face = 10 :=
by
  show (4 + 6) = 10
  sorry

end num_cubes_with_more_than_one_blue_face_l303_303895


namespace stockPrices_l303_303779

-- Given conditions: initial price and daily changes
def initialPrice : ℝ := 27
def changeMonday : ℝ := -1.5
def changeTuesday : ℝ := -1
def changeWednesday : ℝ := +1.5
def changeThursday : ℝ := +0.5
def changeFriday : ℝ := +1
def changeSaturday : ℝ := -0.5

-- Calculating the price each day
def priceMonday : ℝ := initialPrice + changeMonday
def priceTuesday : ℝ := priceMonday + changeTuesday
def priceWednesday : ℝ := priceTuesday + changeWednesday
def priceThursday : ℝ := priceWednesday + changeThursday
def priceFriday : ℝ := priceThursday + changeFriday
def priceSaturday : ℝ := priceFriday + changeSaturday

theorem stockPrices :
  priceWednesday = 26 ∧
  max priceMonday (max priceTuesday (max priceWednesday (max priceThursday (max priceFriday priceSaturday)))) = 27.5 ∧
  min priceMonday (min priceTuesday (min priceWednesday (min priceThursday (min priceFriday priceSaturday)))) = 24.5 :=
by
  -- Proof can be derived here
  sorry

end stockPrices_l303_303779


namespace original_weight_of_marble_l303_303889

variable (W: ℝ) 

theorem original_weight_of_marble (h: 0.80 * 0.82 * 0.72 * W = 85.0176): W = 144 := 
by
  sorry

end original_weight_of_marble_l303_303889


namespace incircle_radius_of_right_45_45_90_triangle_l303_303823

theorem incircle_radius_of_right_45_45_90_triangle (D E F : ℝ) (r : ℝ) 
  (h1 : ∠D = 45)
  (h2 : ∠E = 45)
  (h3 : ∠F = 90)
  (h4 : DF = 12)
  (h5 : EF = 12)
  (h6 : DE = 12 * Real.sqrt 2) :
  inradius (triangle DEF) = 6 - 3 * Real.sqrt 2 := 
sorry

end incircle_radius_of_right_45_45_90_triangle_l303_303823


namespace find_certain_number_l303_303682

theorem find_certain_number (n : ℕ) (h : 9823 + n = 13200) : n = 3377 :=
by
  sorry

end find_certain_number_l303_303682


namespace probability_zhang_watches_entire_news_l303_303786

noncomputable def broadcast_time_start := 12 * 60 -- 12:00 in minutes
noncomputable def broadcast_time_end := 12 * 60 + 30 -- 12:30 in minutes
noncomputable def news_report_duration := 5 -- 5 minutes
noncomputable def zhang_on_tv_time := 12 * 60 + 20 -- 12:20 in minutes
noncomputable def favorable_time_start := zhang_on_tv_time
noncomputable def favorable_time_end := zhang_on_tv_time + news_report_duration -- 12:20 to 12:25

theorem probability_zhang_watches_entire_news : 
  let total_broadcast_time := broadcast_time_end - broadcast_time_start
  let favorable_time_span := favorable_time_end - favorable_time_start
  favorable_time_span / total_broadcast_time = 1 / 6 :=
by
  sorry

end probability_zhang_watches_entire_news_l303_303786


namespace sphere_volume_l303_303996

theorem sphere_volume (l w h : ℝ) (hl : l = 1) (hw : w = 1) (hh : h = 2) : 
    let d := Real.sqrt (l^2 + w^2 + h^2)
    let r := d / 2
    let V := (4/3) * Real.pi * r^3
in V = Real.sqrt 6 * Real.pi :=
by
  sorry

end sphere_volume_l303_303996


namespace inequality_sqrt_three_l303_303208

theorem inequality_sqrt_three (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h : a^2 + b^2 + c^2 = 1) : 
  (ab / c + bc / a + ca / b) ≥ √3 :=
by
  sorry

end inequality_sqrt_three_l303_303208


namespace trig_identity_l303_303984

theorem trig_identity (x : ℝ) (h : sin x - 2 * cos x = 0) : 2 * sin x ^ 2 + cos x ^ 2 + 1 = 14 / 5 :=
by 
  sorry

end trig_identity_l303_303984


namespace prob_remainder_mod_1000_l303_303523

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ 
| n 0 := 1
| 0 k := 0
| n k := binom (n-1) (k-1) * n / k

-- Define the sum we are interested in, only including indices that are multiples of 4
def sum_binom_2011_multiple_4 : ℕ :=
  (Finset.range (2012 / 4 + 1)).sum (λ i, binom 2011 (4 * i))

-- The statement we want to prove
theorem prob_remainder_mod_1000 : 
  sum_binom_2011_multiple_4 % 1000 = 12 := 
sorry

end prob_remainder_mod_1000_l303_303523


namespace problem1_problem2_l303_303645

-- Define the function f(x)
def f (x a : ℝ) := abs (x + a) - abs (2 * x - 1)

-- Problem 1: Find the solution set of the inequality f(x) > 0 when a = 1.
theorem problem1 : {x : ℝ | f x 1 > 0} = set.Ioo 0 2 := 
by 
  sorry

-- Problem 2: If a > 0, find the range of values for 'a' such that the inequality f(x) < 1 holds true for all x ∈ R.
theorem problem2 : 
  (∀ x : ℝ, f x a < 1) → 0 < a ∧ a < 1 / 2 := 
by 
  sorry

end problem1_problem2_l303_303645


namespace repeating_decimal_subtraction_l303_303441

noncomputable def x := (0.246 : Real)
noncomputable def y := (0.135 : Real)
noncomputable def z := (0.579 : Real)

theorem repeating_decimal_subtraction :
  x - y - z = (-156 : ℚ) / 333 :=
by
  sorry

end repeating_decimal_subtraction_l303_303441


namespace product_of_divisors_of_18_l303_303285

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l303_303285


namespace trapezoid_XZ_length_l303_303152

theorem trapezoid_XZ_length (W X Y Z P Q : Point) :
  parallel WY XZ ∧ length WY = 7 ∧ length XY = 5 * sqrt 2 ∧ angle XYZ = 30 ∧ angle WZ = 60 →
  ∃ x : Real, length XZ = x ∧ x = 7 + 7 / 4 * sqrt 3 :=
by
  sorry

end trapezoid_XZ_length_l303_303152


namespace find_diameter_l303_303024

noncomputable def cost_per_meter : ℝ := 3.50
noncomputable def total_cost : ℝ := 395.84
noncomputable def circumference (C : ℝ) : Prop :=
  C = total_cost / cost_per_meter

noncomputable def pi_approx : ℝ := 3.14159

noncomputable def diameter (C : ℝ) (D : ℝ) : Prop :=
  C = pi_approx * D

theorem find_diameter (C : ℝ) (D : ℝ) (hC : circumference C) (hD : diameter C D) : D ≈ 36 := by
  sorry

end find_diameter_l303_303024


namespace probability_of_adjacent_abby_bridget_l303_303896

def students : Type := Fin 8

def rows : Type := Fin 2

def columns : Type := Fin 4

structure seating :=
(row : rows)
(column : columns)

def seating_positions (s : students) : seating := sorry

def random_assignment := sorry

def number_of_total_arrangements : Nat := 8.factorial

def adjacent_in_row_or_column (a b : students) : Prop :=
  (seating_positions a).row = (seating_positions b).row
  ∧ abs ((seating_positions a).column - (seating_positions b).column) = 1
  ∨
  (seating_positions a).column = (seating_positions b).column
  ∧ abs ((seating_positions a).row - (seating_positions b).row) = 1

def number_of_favorable_arrangements : Nat := 
  (2 * 3 + 4) * 2 * 6.factorial

def probability : ℚ :=
  number_of_favorable_arrangements / number_of_total_arrangements

theorem probability_of_adjacent_abby_bridget :
  probability = 5 / 14 := sorry

end probability_of_adjacent_abby_bridget_l303_303896


namespace complex_mul_conj_eq_two_l303_303050

def z : ℂ := (2 * complex.I) / (1 - complex.I)

theorem complex_mul_conj_eq_two : z * conj z = 2 := by
  sorry

end complex_mul_conj_eq_two_l303_303050


namespace cost_difference_is_90_l303_303852

namespace PrintShops

def cost_per_copy_x := 1.25
def cost_per_copy_y := 2.75
def num_copies := 60

def total_cost_x := num_copies * cost_per_copy_x
def total_cost_y := num_copies * cost_per_copy_y
def cost_difference := total_cost_y - total_cost_x

theorem cost_difference_is_90 : cost_difference = 90 := by sorry

end PrintShops

end cost_difference_is_90_l303_303852


namespace prove_G_eq_G_l303_303179

noncomputable def is_isosceles (A B C : Point) (α : ℝ) : Prop :=
  ∃ (β : ℝ), β = α / 2 ∧ ∠ B A C = α

noncomputable def is_similar_isosceles (A K L M N : Point) (α : ℝ) : Prop :=
  is_isosceles A K L α ∧ is_isosceles A M N α

noncomputable def is_rotationally_symmetric (G G' : Point) (A N L : Point) (α : ℝ) : Prop :=
  ∃ (f f' : Point → Point),
  (∀ (P : Point), f (f P) = P) ∧ (∀ (P : Point), f' (f' P) = P) ∧
  f (R_A α N) = L ∧ f' (R_A α L) = N ∧
  R_G (π - α) (R_A α P) = R_G' (π - α) (R_A α P)

theorem prove_G_eq_G' {A K L M N G G' : Point} (α : ℝ) :
  is_similar_isosceles A K L M N α →
  is_rotationally_symmetric G G' A N L α →
  G = G' :=
sorry

end prove_G_eq_G_l303_303179


namespace liquid_x_percentage_l303_303454

theorem liquid_x_percentage
  (percentage_A : ℝ) (weight_A : ℝ) (percentage_B : ℝ) (weight_B : ℝ)
  (hx : percentage_A = 0.8) (hy : weight_A = 600)
  (hz : percentage_B = 1.8) (hw : weight_B = 700) : 
  let amount_X_A := percentage_A / 100 * weight_A in
  let amount_X_B := percentage_B / 100 * weight_B in
  let total_amount_X := amount_X_A + amount_X_B in
  let total_weight := weight_A + weight_B in
  (total_amount_X / total_weight) * 100 = 1.34 := 
by
  sorry

end liquid_x_percentage_l303_303454


namespace min_area_triangle_fab_l303_303649

theorem min_area_triangle_fab (p : ℝ) (hp : p > 0) :
  ∃ A B : ℝ × ℝ, ∃ F : ℝ × ℝ,
    (parabola F) ∧ (focus_of_parabola F) ∧ (perpendicular_chords_through_focus A B F) ∧ 
    (area_of_triangle F A B = (3 - 2 * real.sqrt 2) * p^2) :=
sorry

end min_area_triangle_fab_l303_303649


namespace solve_diff_eqn_l303_303777

-- Define the differential equation
def diff_eqn (y : ℝ → ℝ) :=
  deriv (deriv (deriv (deriv y))) - 4 * deriv (deriv (deriv y)) + 8 * deriv (deriv y) - 8 * deriv y + 4 * y

-- Statement of the theorem
theorem solve_diff_eqn :
  ∀ (y : ℝ → ℝ),
  (∀ x, diff_eqn y x = 0) ↔
  ∃ (C₁ C₂ C₃ C₄ : ℝ),
    y = λ x, exp x * ((C₁ + C₃ * x) * cos x + (C₂ + C₄ * x) * sin x) :=
sorry

end solve_diff_eqn_l303_303777


namespace parabola_directrix_l303_303549

theorem parabola_directrix (x : ℝ) : 
  (∃ y : ℝ, y = (x^2 - 4 * x + 7) / 8) → 
  (∃ d : ℝ, d = -("1e19/8") := sorry

end parabola_directrix_l303_303549


namespace find_angle_B_l303_303658

noncomputable def triangle_sides_and_angles 
(a b c : ℝ) (A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

noncomputable def vectors_parallel 
(A B C a b c : ℝ) : Prop :=
  (Real.sin B - Real.sin A) / Real.sin C = (Real.sqrt 3 * a + c) / (a + b)

theorem find_angle_B (A B C a b c : ℝ)
  (h_triangle : triangle_sides_and_angles a b c A B C)
  (h_parallel : vectors_parallel A B C a b c) :
  B = 5 * Real.pi / 6 :=
sorry

end find_angle_B_l303_303658


namespace min_value_b_div_a_l303_303999

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := log x + (real.exp 1 - a) * x - 2 * b

theorem min_value_b_div_a (a b : ℝ) 
  (h : ∀ x : ℝ, 0 < x → f x a b ≤ 0) 
  : ∃ a b, (b / a) = -1 / (2 * real.exp 1) := 
sorry

end min_value_b_div_a_l303_303999


namespace failing_percentage_exceeds_35_percent_l303_303142

theorem failing_percentage_exceeds_35_percent:
  ∃ (n D A B failD failA : ℕ), 
  n = 25 ∧
  D + A - B = n ∧
  (failD * 100) / D = 30 ∧
  (failA * 100) / A = 30 ∧
  ((failD + failA) * 100) / n > 35 := 
by
  sorry

end failing_percentage_exceeds_35_percent_l303_303142


namespace min_value_expression_l303_303175

theorem min_value_expression (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 3/2) : 
  ∃ m, (∀ x y z, 2 * x^2 + 4 * x * y + 9 * y^2 + 10 * y * z + 3 * z^2 ≥ m) ∧
  m = 27 / 2^(4/9) * (90)^(1/9) :=
begin
  sorry
end

end min_value_expression_l303_303175


namespace max_days_for_process_C_l303_303478

theorem max_days_for_process_C (A B C D : ℕ) (total_duration : ℕ) (x : ℕ) : 
  A = 2 →
  B = 5 →
  D = 4 →
  total_duration = 9 →
  A + max B (C + D) = total_duration →
  C ≤ x →
  x = 3 :=
by
  intros hA hB hD htotal hduration hC
  have hC' : C ≤ 3, { sorry }
  have hCmax : x = 3, { sorry }
  exact hCmax

end max_days_for_process_C_l303_303478


namespace angle_equality_l303_303051

-- Definitions based on conditions
variable (A B C D P T : Type)
variable [convex_quadrilateral A B C D]
variable (h1 : angle A B C = angle B C D)
variable (h2 : ∃ P, (line_through A D).intersection (line_through B C) = P)
variable (h3 : ∃ T, (line_through P).parallel (line_through A B) ∧ (line_through T).intersection (line_through B D) = T)

-- Prove that angle ACB = angle PCT
theorem angle_equality
  (A B C D P T : Type)
  [convex_quadrilateral A B C D]
  (h1 : angle A B C = angle B C D)
  (h2 : ∃ P, (line_through A D).intersection (line_through B C) = P)
  (h3 : ∃ T, (line_through P).parallel (line_through A B) ∧ (line_through T).intersection (line_through B D) = T) :
  angle A C B = angle P C T := by
    sorry

end angle_equality_l303_303051


namespace product_of_all_positive_divisors_of_18_l303_303397

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l303_303397


namespace inequality_solution_l303_303234

theorem inequality_solution (x : ℝ) :
  -1 < (x^2 - 12 * x + 35) / (x^2 - 4 * x + 8) ∧
  (x^2 - 12 * x + 35) / (x^2 - 4 * x + 8) < 1 ↔
  x > (27 / 8) :=
by sorry

end inequality_solution_l303_303234


namespace range_of_a_l303_303655

noncomputable def setA : Set ℝ := {x | 3 + 2 * x - x^2 >= 0}
noncomputable def setB (a : ℝ) : Set ℝ := {x | x > a}

theorem range_of_a (a : ℝ) : (setA ∩ setB a).Nonempty → a < 3 :=
by
  sorry

end range_of_a_l303_303655


namespace painted_cubes_only_two_faces_l303_303873

theorem painted_cubes_only_two_faces :
  ∀ (n : ℕ), n = 3 →
  let total_small_cubes := n * n * n in
  total_small_cubes = 27 →
  let face_painted_cubes := 6 in
  let corner_cubes := 8 in
  let inner_cubes := 1 in
  let edge_cubes := (total_small_cubes - face_painted_cubes - corner_cubes - inner_cubes) in
  edge_cubes = 12 :=
by
  intros n h1 h2 face_painted_cubes corner_cubes inner_cubes
  have h : total_small_cubes = 27 := by rw h2
  have edge_cubes_def : edge_cubes = (total_small_cubes - face_painted_cubes - corner_cubes - inner_cubes) := rfl
  have edge_cubes_result : edge_cubes = 12 := by
    simp [face_painted_cubes, corner_cubes, inner_cubes, total_small_cubes] at edge_cubes_def
    rw [←h, edge_cubes_def]
    norm_num
  exact edge_cubes_result

end painted_cubes_only_two_faces_l303_303873


namespace find_lambda_l303_303093

variables {R : Type*} [Field R]
variables {V : Type*} [AddCommGroup V] [Module R V]
variables (a b : V)
variable (λ : R)

-- Defining collinearity
def not_collinear (a b : V) : Prop := ¬∃ k : R, b = k • a

-- Defining parallelism
def parallel (u v : V) : Prop := ∃ k : R, u = k • v

-- The problem statement
theorem find_lambda (h₁ : not_collinear a b)
                    (h₂ : parallel (λ • a + b) (a - 2 • b)) :
  λ = -1 / 2 :=
sorry

end find_lambda_l303_303093


namespace find_length_and_value_l303_303194

noncomputable def second_largest_length (a b : ℕ) [fact (nat.gcd a b = 1)] : ℝ :=
  real.sqrt (a / b)

theorem find_length_and_value :
  ∃ (a b : ℕ) [fact (nat.gcd a b = 1)], 
  let side_length := 10 in
  let P_inside_square := true in 
  let circumcenters_exist := true in
  let sum_lengths := (PA + PB + PC + PD = 23 * real.sqrt 2) in
  let area_quadrilateral := 50 in
  (second_largest_length a b = real.sqrt (50/1)) ∧ (100 * a + b = 5001) :=
begin
  sorry
end

end find_length_and_value_l303_303194


namespace additional_set_possible_l303_303963

open Set

variable (S : Set (Set ℕ)) (elements : Fin 5)
variable (sets : Fin 14 → Set ℕ)

-- Condition: Each set contains at least one element.
def NonEmptySets : Prop :=
  ∀ i : Fin 14, sets i ≠ ∅

-- Condition: Any two sets have at least one common element.
def CommonElementSets : Prop :=
  ∀ i j : Fin 14, i ≠ j → (sets i ∩ sets j) ≠ ∅

-- Condition: No two sets are identical.
def UniqueSets : Prop :=
  ∀ i j : Fin 14, i ≠ j → sets i ≠ sets j

-- Condition: Sets are formed from elements.
def SetsFromElements : Prop :=
  ∀ i : Fin 14, sets i ⊆ elements

theorem additional_set_possible {elements : Finset ℕ} (h : elements.card = 5) :
  ∃ (fifteenth_set : Set ℕ), fifteenth_set ⊆ elements ∧
  fifteenth_set ≠ ∅ ∧
  (∀ i : Fin 14, fifteenth_set ≠ sets i) ∧
  (∀ i : Fin 14, (fifteenth_set ∩ sets i) ≠ ∅) :=
by
  -- We skip the proof itself as required.
  sorry

end additional_set_possible_l303_303963


namespace intersection_A_B_l303_303617

open Set

-- Given definitions of sets A and B
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 - 2 * x ≥ 0}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {-1, 0, 2} :=
sorry

end intersection_A_B_l303_303617


namespace radius_semicircle_PR_l303_303153

-- Definitions based on given conditions
variable (P Q R : Type) [InnerProductSpace ℝ P] [AffineSpace P Q] [MetricSpace P]
variable (angleQ_right : ∠ Q = π / 2)
variable (area_PQ : real.pi / 2 * (PQ / 2) ^ 2 = 18 * real.pi)
variable (arc_length_QR : real.pi * (QR / 2) = 10 * real.pi)

-- Statement of the theorem
theorem radius_semicircle_PR :
  ((PQ / 2) ^ 2 + (QR / 2) ^ 2 = (PR / 2) ^ 2) →
  (PR / 2 = 4 * sqrt 17) :=
sorry

end radius_semicircle_PR_l303_303153


namespace find_t_l303_303014

variables (c o u n t s : ℕ)

theorem find_t (h1 : c + o = u) 
               (h2 : u + n = t)
               (h3 : t + c = s)
               (h4 : o + n + s = 18)
               (hz : c > 0) (ho : o > 0) (hu : u > 0) (hn : n > 0) (ht : t > 0) (hs : s > 0) : 
               t = 9 := 
by
  sorry

end find_t_l303_303014


namespace arithmetic_geometric_seq_summation_behavior_l303_303611

noncomputable def a_n (a_1 d : ℝ) (n : ℕ) : ℝ := a_1 + (n - 1) * d

noncomputable def S_n (a_1 d : ℝ) (n : ℕ) : ℝ := n / 2 * (2 * a_1 + (n - 1) * d)

theorem arithmetic_geometric_seq (a_1 d : ℝ) (n : ℕ) (h1 : d ≠ 0) (h2 : a_1 * (a_1 + 5 * d) = (a_1 + 3 * d) ^ 2) :
  S_n a_1 d 19 = 0 :=
by
  have a1_eq : a_1 = -9 * d,
  { sorry }, -- Details of the intermediate proofs are omitted as per instruction
  have a_10_eq : a_n a_1 d 10 = 0,
  { sorry }, -- Details of the intermediate proofs are omitted as per instruction
  have s_19_eq : S_n a_1 d 19 = 0,
  { sorry }, -- Details of the intermediate proofs are omitted as per instruction
  exact s_19_eq

theorem summation_behavior (a_1 d : ℝ) (h1 : d ≠ 0) (h2 : a_1 * (a_1 + 5 * d) = (a_1 + 3 * d) ^ 2) :
  (d < 0 → S_n a_1 d 9 = S_n a_1 d 10) ∧ (d > 0 → S_n a_1 d 10 = S_n a_1 d 9) :=
by
  have a1_eq : a_1 = -9 * d,
  { sorry }, -- Details of the intermediate proofs are omitted as per instruction
  sorry -- The detailed behavior proofs omitted as per instruction

end arithmetic_geometric_seq_summation_behavior_l303_303611


namespace product_of_divisors_of_18_l303_303431

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303431


namespace binomial_sum_mod_1000_l303_303513

open BigOperators

theorem binomial_sum_mod_1000 :
  ((∑ k in finset.range 503 \ finset.range 3, nat.choose 2011 (4 * k)) % 1000) = 49 := 
sorry

end binomial_sum_mod_1000_l303_303513


namespace tan_angle_sum_l303_303965

theorem tan_angle_sum
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 :=
by
  sorry

end tan_angle_sum_l303_303965


namespace exists_root_in_interval_l303_303009

open Real

/-- The functions involved: f and g -/
def f (x : ℝ) : ℝ := 2^x
def g (x : ℝ) : ℝ := 2 - x

/-- The proof problem: there exists an x in the interval (0, 1) such that f(x) = g(x) -/
theorem exists_root_in_interval : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = g x := 
sorry

end exists_root_in_interval_l303_303009


namespace calculate_remainder_l303_303508

open Nat

theorem calculate_remainder :
  let ω := complex.exp (2 * complex.pi * complex.I / 4) in
  ω^4 = 1 ∧ ω ≠ 1 ∧ ω^2 = -1 ∧ ω^3 = -ω ∧
  let S := (1 + ω)^2011 + (1 + ω^2)^2011 + (1 + ω^3)^2011 + (2:ℂ)^2011 in
  S = 4 * ∑ k in range (503), nat.choose 2011 (4 * k) →
  (1 + ω^2)^2011 = 0 ∧ (1 + ω)^2011 + (1 + ω^3)^2011 = 0 →
  S = (2:ℂ)^2011 →
  (2^2011 : ℕ) % 8 = 0 ∧ (2^2011 : ℕ) % 125 = 48 →
  (2^2011 : ℕ) % 1000 = 48 →
  (4 * ∑ k in range (503), nat.choose 2011 (4 * k)) % 1000 = 48 →
  ((∑ k in range (503), nat.choose 2011 (4 * k)) % 1000 = 12) :=
by
  intros ω ω4 ω_ne ω2 ω3 S hS h1 h2 h3 h4 h5
  sorry

end calculate_remainder_l303_303508


namespace minimum_x_plus_y_l303_303065

theorem minimum_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) 
    (h1 : x - y < 1) (h2 : 2 * x - y > 2) (h3 : x < 5) : 
    x + y ≥ 6 :=
sorry

end minimum_x_plus_y_l303_303065


namespace ratio_x_y_l303_303961

theorem ratio_x_y (x y : ℝ) (h : (1/x - 1/y) / (1/x + 1/y) = 2023) : (x + y) / (x - y) = -1 := 
by
  sorry

end ratio_x_y_l303_303961


namespace valid_points_in_second_quadrant_l303_303053

-- Define the conditions for P(x, y) being in the second quadrant and meeting the given inequality
def in_second_quadrant (x y : ℤ) : Prop := x < 0 ∧ y > 0
def satisfies_inequality (x y : ℤ) : Prop := y ≤ x + 4

-- Define the set of all coordinates that meet the conditions
def valid_coordinates : set (ℤ × ℤ) :=
  {(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-3, 1)}

-- Lean statement to prove that valid_coordinates contains all points (x, y) satisfying the conditions
theorem valid_points_in_second_quadrant :
  {P : ℤ × ℤ | in_second_quadrant P.1 P.2 ∧ satisfies_inequality P.1 P.2}
  = valid_coordinates :=
sorry

end valid_points_in_second_quadrant_l303_303053


namespace minimum_value_l303_303986

variable (x y : ℝ)

def expression (x y : ℝ) : ℝ :=
  1 / x - 4 * y / (y + 1)

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : expression x y = 1 / 2 :=
sorry

end minimum_value_l303_303986


namespace find_fraction_increase_l303_303934

noncomputable def present_value : ℝ := 64000
noncomputable def value_after_two_years : ℝ := 87111.11111111112

theorem find_fraction_increase (f : ℝ) :
  64000 * (1 + f) ^ 2 = 87111.11111111112 → f = 0.1666666666666667 := 
by
  intro h
  -- proof steps here
  sorry

end find_fraction_increase_l303_303934


namespace positive_n_value_l303_303038

noncomputable def isRealModulus (z : ℂ) : ℝ := complex.abs z

theorem positive_n_value (n : ℝ) (h : isRealModulus (5 + n * complex.i) = 5 * real.sqrt 13) : n = 10 * real.sqrt 3 :=
by
  sorry

end positive_n_value_l303_303038


namespace k_h_set_l303_303613

noncomputable def k_set (r : ℕ) : Set ℕ :=
  { k | ∃ (m n : ℕ), odd m ∧ m > 1 ∧ k ∣ (m^k - 1) ∧ m ∣ (n^( (m^k - 1) / k ) + 1) }

theorem k_h_set (r : ℕ) :
  k_set r = { k | ∃ s t : ℕ, 2 ∣ k ∧ ( k = 2^r * 2^s * t ∧ (2 ∣ t = false) ) } := sorry

end k_h_set_l303_303613


namespace an_is_zero_l303_303160

noncomputable def f (a : ℕ → ℝ) (x : ℝ) : ℝ := ∑' n, a n / (x + n^2)

theorem an_is_zero (a : ℕ → ℝ) (α β : ℝ) (hα : 2 < α) (hβ : β > 1 / α) :
  (∑' n, |a n| / n^α < ∞) →
  (∀ (x : ℝ), 0 ≤ x → ∃ C > 0, ∀ x > C, |f a x| ≤ C * exp (-x^β)) →
  ∀ n, a n = 0 :=
by
  sorry

end an_is_zero_l303_303160


namespace spiders_make_webs_l303_303116

theorem spiders_make_webs :
  (∀ (s d : ℕ), s = 7 ∧ d = 7 → (∃ w : ℕ, w = s)) ∧
  (∀ (d w : ℕ), w = 1 ∧ d = 7 → (∃ s : ℕ, s = w)) →
  (∀ (s : ℕ), s = 1) :=
by
  sorry

end spiders_make_webs_l303_303116


namespace product_of_divisors_of_18_is_5832_l303_303401

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l303_303401


namespace locus_of_point_Q_l303_303968

-- Given conditions
variables (λ : ℝ) [h : Fact (λ > 1)]
variables (P B A C : Point)
variables (triangleABC : Triangle B A C)
variables (U V : Point)
variables (BU BA CV CA : ℝ)
variables (UQ UV : ℝ)

-- Definitions based on given conditions
def BU_eq_lambda_BA : Prop := BU = λ * BA
def CV_eq_lambda_CA : Prop := CV = λ * CA
def UQ_eq_lambda_UV : Prop := UQ = λ * UV

-- Derivation of point D based on extension rule
noncomputable def D := extend BC (λ * BC)

/-- Proof problem to show Q's locus --/
theorem locus_of_point_Q :
  (λ > 1) →
  BU = λ * BA →
  CV = λ * CA →
  UQ = λ * UV →
  locus Q = {Q : Point | dist Q D = 2 * dist A D} := sorry

end locus_of_point_Q_l303_303968


namespace total_toll_is_205_79_l303_303267

-- Definitions based on conditions
def T (B A1 A2 : ℝ) (X1 X2 : ℕ) : ℝ :=
  B + A1 * (X1 - 2) + A2 * X2

def F (w : ℝ) : ℝ :=
  if w > 10000 then 0.1 * (w - 10000) else 0

def S (T F : ℝ) (is_peak : Bool) : ℝ :=
  if is_peak then 0.02 * (T + F) else 0

-- Given values
noncomputable def B := 0.50
noncomputable def A1 := 0.75
noncomputable def A2 := 0.50
noncomputable def w := 12000.0
noncomputable def is_peak := true
noncomputable def X1 := 1
noncomputable def X2 := 4

-- Final total toll calculation
noncomputable def total_toll : ℝ :=
  let T_val := T B A1 A2 X1 X2
  let F_val := F w
  let toll_without_surcharge := T_val + F_val
  let surcharge := S T_val F_val is_peak
  toll_without_surcharge + surcharge

theorem total_toll_is_205_79 : total_toll = 205.79 := by
  sorry

end total_toll_is_205_79_l303_303267


namespace f_g_evaluation_l303_303677

-- Definitions of the functions g and f
def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3 * x - 2

-- Goal: Prove that f(g(2)) = 22
theorem f_g_evaluation : f (g 2) = 22 :=
by
  sorry

end f_g_evaluation_l303_303677


namespace rice_and_wheat_grains_division_l303_303713

-- Definitions for the conditions in the problem
def total_grains : ℕ := 1534
def sample_size : ℕ := 254
def wheat_in_sample : ℕ := 28

-- Proving the approximate amount of wheat grains in the batch  
theorem rice_and_wheat_grains_division : total_grains * (wheat_in_sample / sample_size) = 169 := by 
  sorry

end rice_and_wheat_grains_division_l303_303713


namespace polynomial_coefficient_a5_l303_303104

theorem polynomial_coefficient_a5 : 
  (∃ (a0 a1 a2 a3 a4 a5 a6 : ℝ), 
    (∀ (x : ℝ), ((2 * x - 1)^5 * (x + 2) = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6)) ∧ 
    a5 = 176) := sorry

end polynomial_coefficient_a5_l303_303104


namespace problem_statement_l303_303117

variable {x y : ℝ}

theorem problem_statement (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : y - 2 / x ≠ 0) :
  (2 * x - 3 / y) / (3 * y - 2 / x) = (2 * x * y - 3) / (3 * x * y - 2) :=
sorry

end problem_statement_l303_303117


namespace calculate_remainder_l303_303510

open Nat

theorem calculate_remainder :
  let ω := complex.exp (2 * complex.pi * complex.I / 4) in
  ω^4 = 1 ∧ ω ≠ 1 ∧ ω^2 = -1 ∧ ω^3 = -ω ∧
  let S := (1 + ω)^2011 + (1 + ω^2)^2011 + (1 + ω^3)^2011 + (2:ℂ)^2011 in
  S = 4 * ∑ k in range (503), nat.choose 2011 (4 * k) →
  (1 + ω^2)^2011 = 0 ∧ (1 + ω)^2011 + (1 + ω^3)^2011 = 0 →
  S = (2:ℂ)^2011 →
  (2^2011 : ℕ) % 8 = 0 ∧ (2^2011 : ℕ) % 125 = 48 →
  (2^2011 : ℕ) % 1000 = 48 →
  (4 * ∑ k in range (503), nat.choose 2011 (4 * k)) % 1000 = 48 →
  ((∑ k in range (503), nat.choose 2011 (4 * k)) % 1000 = 12) :=
by
  intros ω ω4 ω_ne ω2 ω3 S hS h1 h2 h3 h4 h5
  sorry

end calculate_remainder_l303_303510


namespace smallest_positive_number_l303_303954

def smallest_positive_option : ℝ :=
  let a := 15 - 4 * Real.sqrt 14
  let b := 4 * Real.sqrt 14 - 15
  let c := 20 - 6 * Real.sqrt 15
  let d := 60 - 12 * Real.sqrt 31
  let e := 12 * Real.sqrt 31 - 60
  if a > 0 ∧ (a < e ∨ e ≤ 0) then a else if e > 0 then e else -1 /*dummy value*/

theorem smallest_positive_number : smallest_positive_option = 15 - 4 * Real.sqrt 14 :=
by
  sorry

end smallest_positive_number_l303_303954


namespace range_of_a_in_second_quadrant_l303_303684

theorem range_of_a_in_second_quadrant :
  (∀ (x y : ℝ), x^2 + y^2 + 6*x - 4*a*y + 3*a^2 + 9 = 0 → x < 0 ∧ y > 0) → (0 < a ∧ a < 3) :=
by
  sorry

end range_of_a_in_second_quadrant_l303_303684


namespace next_palindrome_after_2002_has_product_4_l303_303808

-- Definitions
def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def digits_product (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).prod

-- Problem Statement
theorem next_palindrome_after_2002_has_product_4 :
  (∀ n, n > 2002 ∧ is_palindrome n → n = 2112) ∧ digits_product 2112 = 4 :=
  by
    sorry

end next_palindrome_after_2002_has_product_4_l303_303808


namespace not_possible_l303_303149

/-- Define a 3x3 grid -/
/-- Define a "path of length n" as a sequence traversing n edges -/
structure Grid3x3 (V : Type) := 
  (vertices : fin 16 → V)
  (edges : fin 24 → V × V)
  (connected : ∀ e ∈ edges, Σ v w, (v, w) = e)

noncomputable def grid3x3 : Grid3x3 ℕ := {
  vertices := λ i, i.val,
  edges := λ i, (i.val, i.val + 1),
  connected := sorry
}

/-- Check for paths of given lengths that cover the grid under non-overlapping conditions -/
def valid_paths (G : Grid3x3 ℕ) (lengths : list ℕ) : Prop :=
  ∃ (paths : list (list (V × V))),
    (∀ p ∈ paths, length p = lengths.nth p) ∧
    (∀ p1 p2, p1 ≠ p2 → disjoint (p1.map prod.fst) (p2.map prod.fst))

theorem not_possible :
  ¬ valid_paths grid3x3 [8, 8, 8, 8, 8] :=
begin
  sorry
end

end not_possible_l303_303149


namespace complement_of_A_with_respect_to_U_l303_303087

def U : Set ℤ := {1, 2, 3, 4, 5}
def A : Set ℤ := {x | abs (x - 3) < 2}
def C_UA : Set ℤ := { x | x ∈ U ∧ x ∉ A }

theorem complement_of_A_with_respect_to_U :
  C_UA = {1, 5} :=
by
  sorry

end complement_of_A_with_respect_to_U_l303_303087


namespace product_of_divisors_of_18_is_5832_l303_303400

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l303_303400


namespace second_car_speed_l303_303831

open_locale big_operators

-- Condition definitions
def v1 : ℝ := 60 -- speed of the first car in km/h
def track_length : ℝ := 150 -- length of the circular track in km
def meet_time : ℝ := 2 -- time in hours after which they meet for the second time

-- Theorem statement 
theorem second_car_speed
  (v1 : ℝ)
  (track_length : ℝ)
  (meet_time : ℝ)
  (first_car_speed : v1 = 60)
  (two_circumferences : 2 * track_length = meet_time * (v1 + v2)) :
  v2 = 90 :=
sorry

end second_car_speed_l303_303831


namespace travel_cost_from_B_to_C_l303_303769

noncomputable def distance (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

noncomputable def travel_cost_by_air (distance : ℝ) (booking_fee : ℝ) (per_km_cost : ℝ) : ℝ :=
  booking_fee + (distance * per_km_cost)

theorem travel_cost_from_B_to_C :
  let AC := 4000
  let AB := 4500
  let BC := Real.sqrt (AB^2 - AC^2)
  let booking_fee := 120
  let per_km_cost := 0.12
  travel_cost_by_air BC booking_fee per_km_cost = 367.39 := by
  sorry

end travel_cost_from_B_to_C_l303_303769


namespace solution_set_of_inequality_l303_303171

theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_mono : ∀ x y, x ≤ y → f(x) ≤ f(y))
  (h_func : ∀ a b, f(a + b) = f(a) + f(b) - 1)
  (h_f4 : f 4 = 5) :
  {m : ℝ | f(3 * m^2 - m - 2) < 3} = set.Ioo (-1 : ℝ) (4 / 3 : ℝ) :=
sorry

end solution_set_of_inequality_l303_303171


namespace probability_three_correct_l303_303584

open Fintype

/-- 
The probability that exactly three out of five packages are delivered 
to the correct houses, given random delivery, is 1/6.
-/
theorem probability_three_correct : 
  (∃ (X : Finset (Fin 5)) (hX : X.card = 3) (Y : Finset (Fin 5)) (hY : Y.card = 2) (f : (Fin 5) → (Fin 5)),
    ∀ (x ∈ X, f x = x) ∧ (∀ y ∈ Y, f y ≠ y) ∧ HasDistribNeg.negOne Y y 
    ∧ ∑ y in Y, 1! = 2) 
  → (Real.ofRat (⅙)) := sorry

end probability_three_correct_l303_303584


namespace max_k_no_real_roots_l303_303685

theorem max_k_no_real_roots : ∀ k : ℤ, (∀ x : ℝ, x^2 - 2 * x - (k : ℝ) ≠ 0) → k ≤ -2 :=
by
  sorry

end max_k_no_real_roots_l303_303685


namespace smallest_vertical_segment_length_l303_303794

theorem smallest_vertical_segment_length :
  let f := λ x : ℝ, abs x
  let g := λ x : ℝ, -x^2 - 5*x - 4 in
  ∃ x : ℝ, ∀ y : ℝ, (f x - g x) ≥ 0 ∧ (f x - g x) ≤ (f y - g y) := 
sorry

end smallest_vertical_segment_length_l303_303794


namespace option_B_incorrect_l303_303070

def normal_dist_math_scores (x : ℝ) : ℝ :=
  (1 / (10 * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - 80)^2 / 200))

theorem option_B_incorrect :
  ¬(Real.abs ((Real.erf ((120 - 80) / (10 * 2))) - (Real.erf ((80 - 60) / (10 * 2)))) < ε) :=
sorry

end option_B_incorrect_l303_303070


namespace volume_of_inscribed_sphere_l303_303888

theorem volume_of_inscribed_sphere (s : ℝ) (h : s = 8) : 
    let r := s / (4 * Real.sqrt 6) in
    let V := (4 / 3) * Real.pi * r^3 in
    V = (8 * Real.sqrt 6 / 27) * Real.pi :=
by
  sorry

end volume_of_inscribed_sphere_l303_303888


namespace greatest_perimeter_isosceles_triangle_l303_303540

theorem greatest_perimeter_isosceles_triangle :
  let base := 12
  let height := 15
  let segments := 6
  let max_perimeter := 32.97
  -- Assuming division such that each of the 6 pieces is of equal area,
  -- the greatest perimeter among these pieces to the nearest hundredth is:
  (∀ (base height segments : ℝ), base = 12 ∧ height = 15 ∧ segments = 6 → 
   max_perimeter = 32.97) :=
by
  sorry

end greatest_perimeter_isosceles_triangle_l303_303540


namespace possible_values_a1_b1_l303_303537

noncomputable theory

open Nat

-- Define the sequences a_n and b_n and their properties
variable (a b : ℕ → ℕ)
variable (strictly_increasing : ∀ n, a n < a (n + 1) ∧ b n < b (n + 1))
variable (positive_integers : ∀ n, a n > 0 ∧ b n > 0)
variable (A10_eq_B10_lt_2017 : a 10 = b 10 ∧ a 10 < 2017)
variable (recursive_a : ∀ n, a (n + 2) = a (n + 1) + a n)
variable (recursive_b : ∀ n, b (n + 1) = 2 * b n)

-- Problem statement: Find possible values of a_1 + b_1
theorem possible_values_a1_b1 : 
  ∃ k ∈ ({9, 20, 23} : Finset ℕ), k = (a 1 + b 1) :=
by sorry

end possible_values_a1_b1_l303_303537


namespace sum_of_positive_integers_n_squared_minus_19n_plus_99_is_perfect_square_l303_303581

def is_perfect_square (k : ℤ) := ∃ m : ℤ, m * m = k

theorem sum_of_positive_integers_n_squared_minus_19n_plus_99_is_perfect_square :
  (∑ n in { n : ℕ | n^2 - 19 * n + 99 = x^2 ∧ is_perfect_square (n^2 - 19 * n + 99) }, (n : ℤ)) = 38 :=
begin
  sorry
end

end sum_of_positive_integers_n_squared_minus_19n_plus_99_is_perfect_square_l303_303581


namespace circle_and_arc_symmetry_l303_303780

-- Definitions based on conditions
def is_symmetric (shape : Type) : Prop := 
  ∃ center : shape, ∀ pt : shape, reflection(center, pt) = pt

def has_infinitely_many_axes_of_symmetry (shape : Type) : Prop :=
  ∀ line : shape, is_axis_of_symmetry(line, shape)

def has_one_axis_of_symmetry (arc : Type) : Prop :=
  ∃! line : arc, is_axis_of_symmetry(line, arc)

def no_center_of_symmetry (arc : Type) : Prop :=
  ¬ is_symmetric(arc)

-- Assuming we have necessary definitions for shape properties
axiom reflection : Π {shape : Type}, shape → shape → shape
axiom is_axis_of_symmetry : Π {shape : Type}, shape → shape → Prop

-- Lean statement
theorem circle_and_arc_symmetry (circle arc : Type) :
  (is_symmetric circle ∧ has_infinitely_many_axes_of_symmetry circle) ∧ 
  (has_one_axis_of_symmetry arc ∧ no_center_of_symmetry arc) :=
sorry

end circle_and_arc_symmetry_l303_303780


namespace product_of_divisors_18_l303_303305

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l303_303305


namespace problem_statement_l303_303712

noncomputable def line_l := { p : ℝ × ℝ | p.1 - p.2 + 4 = 0 }

def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

def curve_C (α : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 * Real.cos α, Real.sin α)

-- Define point P in Cartesian coordinates
def point_P : ℝ × ℝ := polar_to_cartesian 4 (Real.pi / 2)

-- Definition of the distance from a point to a line
def distance_from_point_to_line (Q : ℝ × ℝ) : ℝ :=
  |Q.1 - Q.2 + 4| / Real.sqrt 2

-- Definition of d(α)
def d_alpha (α : ℝ) : ℝ :=
  Real.sqrt 3 * Real.cos α - Real.sin α + 4

-- Problem statement
theorem problem_statement :
  (point_P ∈ line_l) ∧ (∃ α : ℝ, distance_from_point_to_line (curve_C α) = Real.sqrt 2) :=
by
  sorry

end problem_statement_l303_303712


namespace incircle_radius_of_right_45_45_90_triangle_l303_303824

theorem incircle_radius_of_right_45_45_90_triangle (D E F : ℝ) (r : ℝ) 
  (h1 : ∠D = 45)
  (h2 : ∠E = 45)
  (h3 : ∠F = 90)
  (h4 : DF = 12)
  (h5 : EF = 12)
  (h6 : DE = 12 * Real.sqrt 2) :
  inradius (triangle DEF) = 6 - 3 * Real.sqrt 2 := 
sorry

end incircle_radius_of_right_45_45_90_triangle_l303_303824


namespace incircle_circumcircle_tangent_l303_303090

-- Define the necessary geometric setup

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def incircle (A B C : Point) : Circle := sorry
noncomputable def tangent (c1 c2 : Circle) : Prop := sorry

theorem incircle_circumcircle_tangent
  (A B C D E F M N P Q X : Point)
  (h_incircle_touches : touches_in_circle A B C D E F)
  (hM : M = midpoint B F)
  (hN : N = midpoint B D)
  (hP : P = midpoint C E)
  (hQ : Q = midpoint C D)
  (hX : on_intersect MN PQ X) :
  tangent (circumcircle X B C) (incircle A B C) :=
begin
  sorry
end

end incircle_circumcircle_tangent_l303_303090


namespace a1_eq_3_gen_formula_an_sum_bn_l303_303262

-- Define the sequence {a_n} and its sum {S_n}
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
axiom ax1 : ∀ n, a n > 0
axiom ax2 : ∀ n, (a n) ^ 2 + 2 * (a n) = 4 * (S n) + 3

-- Problem 1: Prove a_1 = 3
theorem a1_eq_3 (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, a n > 0) (h2 : ∀ n, (a n) ^ 2 + 2 * (a n) = 4 * (S n) + 3) : a 1 = 3 :=
by sorry

-- Problem 2: Prove the general formula for a_n is 2n + 1
theorem gen_formula_an (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, a n > 0) (h2 : ∀ n, (a n) ^ 2 + 2 * (a n) = 4 * (S n) + 3) : ∀ n, a n = 2 * n +1 :=
by sorry

-- Problem 3: Prove the sum of the first n terms of the sequence {b_n} is n / (6n + 9)
theorem sum_bn (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, a n > 0) (h2 : ∀ n, (a n) ^ 2 + 2 * (a n) = 4 * (S n) + 3) :
  ∀ n, (finset.sum (finset.range n) (λ k, (1 : ℚ) / ((a k) * (a (k + 1))))) = n / (6 * n + 9) :=
by sorry

end a1_eq_3_gen_formula_an_sum_bn_l303_303262


namespace range_of_a_odd_not_even_l303_303635

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

def A : Set ℝ := Set.Ioo (-1 : ℝ) 1

def B (a : ℝ) : Set ℝ := Set.Ioo a (a + 1)

theorem range_of_a (a : ℝ) (h1 : B a ⊆ A) : -1 ≤ a ∧ a ≤ 0 := by
  sorry

theorem odd_not_even : (∀ x ∈ A, f (-x) = - f x) ∧ ¬ (∀ x ∈ A, f x = f (-x)) := by
  sorry

end range_of_a_odd_not_even_l303_303635


namespace large_square_area_l303_303268

theorem large_square_area (
  a b c : ℕ
  (h1 : 4 * a < b)
  (h2 : c^2 = a^2 + b^2 + 10)
  ) : c^2 = 36 :=
sorry

end large_square_area_l303_303268


namespace optionA_optionB_optionC_optionD_l303_303702

-- Given an acute triangle ABC with angles A, B, and C
variables {A B C : ℝ}
-- Assume the angles are between 0 and π/2
variable hacute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2
-- Assume the angles sum to π
variable hsum : A + B + C = π

-- Prove each statement
theorem optionA (hA_B : A > B) : sin A > sin B := sorry
theorem optionB (hA_eq : A = π / 3) : ¬(0 < B ∧ B < π / 2) := sorry
theorem optionC : sin A + sin B > cos A + cos B := sorry
theorem optionD : tan B * tan C > 1 := sorry

end optionA_optionB_optionC_optionD_l303_303702


namespace minimum_rubles_to_reverse_order_l303_303709

theorem minimum_rubles_to_reverse_order (strip_length : ℕ) (initial_order reversed_order: Fin strip_length → ℕ) : strip_length = 100 ∧ 
(∀ i : Fin strip_length, initial_order (strip_length - 1 - i) = reversed_order i) ∧ 
(∀ i : Fin strip_length−1, ∃ n, 1 + n ≤ strip_length ∧ initial_order (i+suc n) = initial_order i) ∧
(∀ i : Fin strip_length−4, ∃ m, 3 + m < strip_length ∧ initial_order (i+3+m) = initial_order i) →
minimum_rubles required to rearrange initial_order into reversed_order = 50 :=
by
  sorry

end minimum_rubles_to_reverse_order_l303_303709


namespace parabola_vertex_distance_l303_303547

theorem parabola_vertex_distance :
  let vertex1 := (0, 2.5)
  let vertex2 := (0, -1.5)
  let d := dist vertex1 vertex2
  d = 4 :=
by
  let vertex1 := (0 : ℝ, 2.5)
  let vertex2 := (0 : ℝ, -1.5)
  let d := dist vertex1 vertex2
  calc
    d = dist vertex1 vertex2 : rfl
    ... = |2.5 - (-1.5)| : by sorry -- calculation to be filled
    ... = 4 : by sorry -- calculation to be filled

end parabola_vertex_distance_l303_303547


namespace largest_three_digit_integer_divisible_by_each_digit_l303_303949

theorem largest_three_digit_integer_divisible_by_each_digit (n : ℕ) :
  (∃ h t u : ℕ, n = 100 * h + 10 * t + u ∧ h = 8 ∧ 
    n % h = 0 ∧ n % t = 0 ∧ n % u = 0 ∧ 
    (u ≠ 0) ∧ (u ≠ t) ∧ (t ≠ 0)) → n ≤ 864 :=
begin
  sorry
end


end largest_three_digit_integer_divisible_by_each_digit_l303_303949


namespace train_length_approx_280_l303_303893

namespace TrainLength

-- Given conditions
def speed_kmh : ℝ := 40
def time_seconds : ℝ := 25.2

-- Convert speed from km/hr to m/s
def speed_ms : ℝ := (speed_kmh * 1000) / 3600

-- Length of the train in meters
def length_of_train : ℝ := speed_ms * time_seconds

-- The theorem to prove
theorem train_length_approx_280 : length_of_train ≈ 280 :=
by
  -- Here '≈' denotes the approximate equality that we need to show
  sorry 

end TrainLength

end train_length_approx_280_l303_303893


namespace sum_black_equals_white_l303_303132

theorem sum_black_equals_white (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  let frame_squares := ((finset.range m).product (finset.range n)).filter (λ (i, j), 
                      i = 0 ∨ i = m-1 ∨ j = 0 ∨ j = n-1),
      black_squares := frame_squares.filter (λ (i, j), ((i + j) % 2 = 0)),
      white_squares := frame_squares.filter (λ (i, j), ((i + j) % 2 = 1)),
      sum_black := black_squares.sum (λ (i, j), i * j),
      sum_white := white_squares.sum (λ (i, j), i * j)
  in sum_black = sum_white :=
by sorry

end sum_black_equals_white_l303_303132


namespace find_f_f_neg1_l303_303640

noncomputable def f : ℤ → ℤ 
| x := if x ≥ 4 then x - 4 else f (x + 3)

theorem find_f_f_neg1 : f (f (-1)) = 0 := 
by
  sorry

end find_f_f_neg1_l303_303640


namespace sin_A_value_l303_303690

theorem sin_A_value (a b : ℝ) (B : ℝ) (h1 : a = 4) (h2 : b = 6) (h3 : B = real.pi / 3) : 
  real.sin A = real.sqrt 3 / 3 :=
by
  -- start with the assumptions from the conditions
  sorry

end sin_A_value_l303_303690


namespace tobacco_acres_difference_l303_303878

variable (total_land: ℝ) (initial_ratio new_ratio: ℝ × ℝ × ℝ)
variable (spring_ratio summer_ratio fall_ratio winter_ratio: ℝ × ℝ × ℝ)

def acres_per_ratio (total : ℝ) (ratio : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let total_parts := ratio.1 + ratio.2 + ratio.3
  (total * (ratio.1 / total_parts), total * (ratio.2 / total_parts), total * (ratio.3 / total_parts))

def total_acres (season_ratios: list (ℝ × ℝ × ℝ)) (total: ℝ) : ℝ :=
  season_ratios.map (λ r => (acres_per_ratio total r).2).sum

theorem tobacco_acres_difference:
  let initial_land := acres_per_ratio total_land initial_ratio;
  let new_land := acres_per_ratio total_land new_ratio;
  let seasonal_land := total_acres [spring_ratio, summer_ratio, fall_ratio, winter_ratio] total_land;
  (seasonal_land - initial_land.3 = 1545) :=
  sorry

-- Setting up variables for concrete values
def total_land := 1350
def initial_ratio := (5, 2, 2)
def new_ratio := (2, 2, 5)
def spring_ratio := (3, 1, 1)
def summer_ratio := (2, 3, 1)
def fall_ratio := (1, 2, 2)
def winter_ratio := (1, 1, 3)

end tobacco_acres_difference_l303_303878


namespace find_angle_A_find_area_l303_303700

-- Problem 1 setup
variables {A B C : ℝ}
variables {a b c : ℝ}
variables (acute_triangle : ∀ {A B C : ℝ}, 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 ∧ A + B + C = π)
variables (sin_eq : 2 * a * Real.sin B = Real.sqrt 3 * b)

theorem find_angle_A (h_acute : acute_triangle) (h_sin_eq : sin_eq) : A = π / 3 := sorry

-- Problem 2 setup
variables (a_eq : a = 6)
variables (sum_bc_eq : b + c = 8)
variables (sin_A_eq : Real.sin A = Real.sqrt 3 / 2)

theorem find_area (h_acute : acute_triangle) (h_sin_eq : sin_eq) (h_a_eq : a_eq) (h_bc_eq : sum_bc_eq) (h_sin_A : sin_A_eq) :
  let area := 1 / 2 * b * c * Real.sin A in
  area = 7 * Real.sqrt 3 / 3 := sorry

end find_angle_A_find_area_l303_303700


namespace find_values_of_a_and_b_l303_303601

theorem find_values_of_a_and_b
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (x : ℝ) (hx : x > 1)
  (h : 9 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 17)
  (h2 : (Real.log b / Real.log a) * (Real.log a / Real.log b) = 2) :
  a = 10 ^ Real.sqrt 2 ∧ b = 10 := by
sorry

end find_values_of_a_and_b_l303_303601


namespace product_of_divisors_18_l303_303322

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l303_303322


namespace proof_problem_l303_303789

-- Define Points and Quadrilaterals
variables (A B C D A' B' C' D' M M' : Type) 
variables [euclidean_space V : Type]

-- The condition that corresponding sides are equal
axiom sides_equal : 
  dist A B = dist A' B' ∧ dist B C = dist B' C' ∧ 
  dist C D = dist C' D' ∧ dist D A = dist D' A'

-- Midpoints
def midpoint (P Q : V) := (P + Q) / 2

-- Perpendiculars passing through midpoints
axiom midpoint_perpendicular (P Q R : V) : midpoint P Q ⊥ R

-- Define the perpendicular condition
def perp (X Y Z W : V) := 
  (X - Y) + (Z - W) = 0

-- Define product equality of segments
def prod_eq (M P Q R S T : V) := 
  dist M P * dist M R = dist Q M * dist S T

-- Conjecture in Lean 4
theorem proof_problem :
  sides_equal A B C D A' B' C' D' →
  (perp B D A C ∧ perp B' D' A' C') ∨ 
  (prod_eq M A C M' A' C' ∨ (dist A M + dist M C) ≠ (dist A' M' + dist M' C')) := sorry

end proof_problem_l303_303789


namespace fill_tank_in_18_minutes_l303_303193

-- Define the conditions
def rate_pipe_A := 1 / 9  -- tanks per minute
def rate_pipe_B := - (1 / 18) -- tanks per minute (negative because it's emptying)

-- Define the net rate of both pipes working together
def net_rate := rate_pipe_A + rate_pipe_B

-- Define the time to fill the tank when both pipes are working
def time_to_fill_tank := 1 / net_rate

theorem fill_tank_in_18_minutes : time_to_fill_tank = 18 := 
    by
    -- Sorry to skip the actual proof
    sorry

end fill_tank_in_18_minutes_l303_303193


namespace calculate_expression_l303_303912

theorem calculate_expression : 
  (Real.sqrt 3) ^ 0 + 2 ^ (-1:ℤ) + Real.sqrt 2 * Real.cos (Float.pi / 4) - Real.abs (-1/2) = 2 := 
by
  sorry

end calculate_expression_l303_303912


namespace binomial_sum_mod_1000_l303_303517

open BigOperators

theorem binomial_sum_mod_1000 :
  ((∑ k in finset.range 503 \ finset.range 3, nat.choose 2011 (4 * k)) % 1000) = 49 := 
sorry

end binomial_sum_mod_1000_l303_303517


namespace product_of_divisors_of_18_l303_303412

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l303_303412


namespace cos_beta_calculation_l303_303066

theorem cos_beta_calculation (α β : ℝ) 
  (h0 : 0 < α ∧ α < π / 2) 
  (h1 : 0 < β ∧ β < π / 2) 
  (h2 : cos α = sqrt 5 / 5) 
  (h3 : sin (α - β) = 3 * sqrt 10 / 10) : 
  cos β = 7 * sqrt 2 / 10 :=
sorry

end cos_beta_calculation_l303_303066


namespace math_problem_l303_303204

variable (a b c : ℝ)

theorem math_problem 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  (ab / c + bc / a + ca / b) ≥ Real.sqrt 3 := 
by
  sorry

end math_problem_l303_303204


namespace red_stripe_area_l303_303489

theorem red_stripe_area
(diameter height stripe_width : ℝ)
(diameter = 30)
(height = 80)
(stripe_width = 3)
: 2 * stripe_width * height = 240 := 
by 
  -- calculations based on the conditions
  sorry
  
end red_stripe_area_l303_303489


namespace _l303_303280

def unit_circle_center : ℝ × ℝ := (0, 0)

constant radius1_circles_intersecting_unit_circle : list (ℝ × ℝ)

noncomputable def is_valid_circle (center : ℝ × ℝ) : Prop := 
  let (x, y) := center in
  let distance_to_origin := Real.sqrt (x^2 + y^2) in
  -- The circle intersects the fixed unit circle
  (distance_to_origin ≤ 2 ∧ distance_to_origin ≥ 1) ∧ 
  -- The circle does not contain the center of the unit circle
  distance_to_origin ≠ 0 ∧
  -- The circle does not contain the center of another circle in the list  
  ∀ c' ∈ radius1_circles_intersecting_unit_circle, 
    let (x', y') := c' in
    c' ≠ center → Real.sqrt ((x - x')^2 + (y - y')^2) ≥ 1

constant valid_circles : list (ℝ × ℝ)
constant number_of_valid_circles : ℕ 

@[simp] theorem max_circles : number_of_valid_circles = 18 :=
by
  -- Assuming valid_circles contains all the valid circle centers
  have all_valid := ∀ c ∈ valid_circles, is_valid_circle c,
  -- Assuming length of valid_circles list is the number of valid circles
  exact sorry

end _l303_303280


namespace actual_distance_traveled_l303_303853

theorem actual_distance_traveled (D : ℕ) (h : (D:ℚ) / 12 = (D + 20) / 16) : D = 60 :=
sorry

end actual_distance_traveled_l303_303853


namespace valid_points_in_second_quadrant_l303_303054

-- Define the conditions for P(x, y) being in the second quadrant and meeting the given inequality
def in_second_quadrant (x y : ℤ) : Prop := x < 0 ∧ y > 0
def satisfies_inequality (x y : ℤ) : Prop := y ≤ x + 4

-- Define the set of all coordinates that meet the conditions
def valid_coordinates : set (ℤ × ℤ) :=
  {(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-3, 1)}

-- Lean statement to prove that valid_coordinates contains all points (x, y) satisfying the conditions
theorem valid_points_in_second_quadrant :
  {P : ℤ × ℤ | in_second_quadrant P.1 P.2 ∧ satisfies_inequality P.1 P.2}
  = valid_coordinates :=
sorry

end valid_points_in_second_quadrant_l303_303054


namespace solution_for_x_l303_303591

theorem solution_for_x (t : ℤ) :
  ∃ x : ℤ, (∃ (k1 k2 k3 : ℤ), 
    (2 * x + 1 = 3 * k1) ∧ (3 * x + 1 = 4 * k2) ∧ (4 * x + 1 = 5 * k3)) :=
  sorry

end solution_for_x_l303_303591


namespace phase_shift_of_sine_l303_303953

theorem phase_shift_of_sine (x : ℝ) : 
  phase_shift (2 * sin (x + π / 4)) = -π / 4 := 
sorry

end phase_shift_of_sine_l303_303953


namespace product_of_divisors_of_18_is_5832_l303_303403

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l303_303403


namespace conditioner_to_shampoo_ratio_l303_303224

theorem conditioner_to_shampoo_ratio (daily_shampoo : ℕ) (total_volume_two_weeks : ℕ) (ratio : ℚ) :
  daily_shampoo = 1 → 
  total_volume_two_weeks = 21 → 
  ratio = (1 / 2 : ℚ) → 
  (14 * daily_shampoo + 14 * (total_volume_two_weeks - 14 * daily_shampoo) / 14) = 21 :=
by
  intros h_shampoo h_total h_ratio
  have h1 : 14 * 1 = 14, from sorry,
  have h2 : 21 - 14 = 7, from sorry,
  have h3 : 7 / 14 = 0.5, from sorry,
  have h4 : 0.5 / 0.5 = 1, from sorry,
  have h5 : 1 / 0.5 = 2, from sorry,
  rw [h1, h2, h3, h4, h5] at *,
  sorry

end conditioner_to_shampoo_ratio_l303_303224


namespace math_problem_l303_303203

variable (a b c : ℝ)

theorem math_problem 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  (ab / c + bc / a + ca / b) ≥ Real.sqrt 3 := 
by
  sorry

end math_problem_l303_303203


namespace evaluate_v2_at_x_neg4_l303_303932

def f (x : ℤ) : ℤ := x^6 + 6 * x^4 + 9 * x^2 + 208

def horner_value_0 : ℤ := 208
def horner_value_1 (x : ℤ) : ℤ := x * horner_value_0
def horner_value_2 (x : ℤ) : ℤ := x * horner_value_1 x + 9

theorem evaluate_v2_at_x_neg4 : horner_value_2 (-4) = 3321 :=
by {
  -- Definitions revolving around horner's method and polynomial evaluation
  have v0: horner_value_0 = 208 := rfl,
  have v1: horner_value_1 (-4) = (-4) * 208 := rfl,
  have v2: horner_value_2 (-4) = (-4) * ((-4) * 208) + 9 := rfl,
  -- Evaluate expressions step-by-step
  rw [v0, v1, v2],
  simp only [neg_mul_eq_neg_mul_symm],
  norm_num,
  sorry -- this is the point where we need to prove the final evaluation step
}

end evaluate_v2_at_x_neg4_l303_303932


namespace no_valid_relation_l303_303174

variables (t x y : ℝ)
hypotheses (htpos : t > 0) (htneq : t ≠ -1)
(def_hx : x = t^(t / (t + 1)))
(def_hy : y = t^(1 / (t + 1)))

theorem no_valid_relation : ¬ (x^(x * y) = y^(y * x)) ∧
                           ¬ (x^(1 / y) = y^(1 / x)) ∧
                           ¬ (x^y = y^x) ∧
                           ¬ (x^x = y^y) :=
by sorry

end no_valid_relation_l303_303174


namespace candidates_count_l303_303890

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 90) : n = 10 :=
by
  sorry

end candidates_count_l303_303890


namespace product_of_divisors_of_18_l303_303418

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l303_303418


namespace product_of_divisors_of_18_l303_303419

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l303_303419


namespace product_of_divisors_of_18_l303_303337

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l303_303337


namespace proof_problem_l303_303048

noncomputable def problem : Prop :=
  ∃ (m n l : Type) (α β : Type) 
    (is_line : ∀ x, x = m ∨ x = n ∨ x = l)
    (is_plane : ∀ x, x = α ∨ x = β)
    (perpendicular : ∀ (l α : Type), Prop)
    (parallel : ∀ (l α : Type), Prop)
    (belongs_to : ∀ (l α : Type), Prop),
    (parallel l α → ∃ l', parallel l' α ∧ parallel l l') ∧
    (perpendicular m α ∧ perpendicular m β → parallel α β)

theorem proof_problem : problem :=
sorry

end proof_problem_l303_303048


namespace product_of_divisors_of_18_l303_303283

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l303_303283


namespace area_of_region_l303_303277

theorem area_of_region : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 + 6*x - 8*y = 16) → 
  (π * 41) = (π * 41) :=
by
  sorry

end area_of_region_l303_303277


namespace find_length_PF_l303_303626

noncomputable def length_PF (x y : ℝ) (P : ℝ×ℝ) (F : ℝ×ℝ) :=
  let PA_length := (P.1 - F.1)
  PA_length * PA_length

theorem find_length_PF : 
  ∃ P : ℝ×ℝ, 
    (let focus : ℝ×ℝ := (1.5, 0)) ∧ 
    (let directrix : ℝ → Prop := fun x => x = -1.5) ∧ 
    (P.2 * P.2 = 6 * P.1) ∧ -- P is on the parabola
    (let y := 3 * Real.sqrt 3) ∧ 
    (P.2 = y) ∧ -- vertical position of point A from directrix
    (P.1 = 4.5) ∧ -- solving for P coordinates
    (length_PF P focus = 6) :=
by
  sorry

end find_length_PF_l303_303626


namespace product_of_divisors_of_18_l303_303281

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l303_303281


namespace find_EQ_l303_303822

section TrapezoidProof

  variables (EF FG GH HE : ℝ)
  variables (Q : ℝ)
  
  def trapezoid_properties :
      EF = 150 ∧ FG = 65 ∧ GH = 35 ∧ HE = 90 ∧ EF = GH := sorry

  def circle_properties :
      Q ∈ set.Icc 0 150 ∧
      (∀ x with x ≤ Q * (FG / FG + HE / HE), EQ = 5 / 3) := sorry

  theorem find_EQ :
      ∀ (EQ QF : ℝ), 
      trapezoid_properties ∧
      circle_properties →
      EQ = Q ∧ 
      QF = 150 - Q →
      EQ = 375 / 4 := 
      
  sorry

end TrapezoidProof

end find_EQ_l303_303822


namespace doughnuts_left_l303_303554

theorem doughnuts_left (total_doughnuts : ℕ) (total_staff : ℕ) (staff3 : ℕ) (staff2 : ℕ) :
  total_doughnuts = 120 → total_staff = 35 →
  staff3 = 15 → staff2 = 10 →
  (let staff4 := total_staff - (staff3 + staff2) in
   let eaten3 := staff3 * 3 in
   let eaten2 := staff2 * 2 in
   let eaten4 := staff4 * 4 in
   let total_eaten := eaten3 + eaten2 + eaten4 in
   total_doughnuts - total_eaten = 15) :=
by
  intros total_doughnuts_eq total_staff_eq staff3_eq staff2_eq
  have staff4 := total_staff - (staff3 + staff2)
  have eaten3 := staff3 * 3
  have eaten2 := staff2 * 2
  have eaten4 := staff4 * 4
  have total_eaten := eaten3 + eaten2 + eaten4
  exact (total_doughnuts - total_eaten = 15)
  sorry

end doughnuts_left_l303_303554


namespace minimum_value_of_f_l303_303650

noncomputable def f (x : ℝ) : ℝ := sorry

theorem minimum_value_of_f :
  (∀ x : ℝ, f (x + 1) + f (x - 1) = 2 * x^2 - 4 * x) →
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = -2 :=
by
  sorry

end minimum_value_of_f_l303_303650


namespace cos_theta_EF_PBC_l303_303718

noncomputable def cos_angle_EF_PBC : ℝ := 
let A := (0, 0, 0) in
let P := (0, 0, 2) in
let B := (2, 0, 0) in
let C := (0, 2, 0) in
let E := ((0 + 2) / 2, (0 + 0) / 2, (0 + 0) / 2) in
let F := ((0 + 0) / 2, (0 + 2) / 2, (2 + 0) / 2) in
let EF := (F.1 - E.1, F.2 - E.2, F.3 - E.3) in
let PB := (B.1 - P.1, B.2 - P.2, B.3 - P.3) in
let PC := (C.1 - P.1, C.2 - P.2, C.3 - P.3) in
let n := (
  PB.2 * PC.3 - PB.3 * PC.2,
  PB.3 * PC.1 - PB.1 * PC.3,
  PB.1 * PC.2 - PB.2 * PC.1
) in
let norm_EF := real.sqrt (EF.1 * EF.1 + EF.2 * EF.2 + EF.3 * EF.3) in
let norm_n := real.sqrt (n.1 * n.1 + n.2 * n.2 + n.3 * n.3) in
let a := (EF.1 / norm_EF, EF.2 / norm_EF, EF.3 / norm_EF) in
let n' := (n.1 / norm_n, n.2 / norm_n, n.3 / norm_n) in
(a.1 * n'.1 + a.2 * n'.2 + a.3 * n'.3) / (1 * 1)

theorem cos_theta_EF_PBC : cos_angle_EF_PBC = 1 / 3 :=
sorry

end cos_theta_EF_PBC_l303_303718


namespace circumcircle_eq_l303_303616

noncomputable def A : (ℝ × ℝ) := (0, 0)
noncomputable def B : (ℝ × ℝ) := (4, 0)
noncomputable def C : (ℝ × ℝ) := (0, 6)

theorem circumcircle_eq :
  ∃ h k r, h = 2 ∧ k = 3 ∧ r = 13 ∧ (∀ x y, ((x - h)^2 + (y - k)^2 = r) ↔ (x - 2)^2 + (y - 3)^2 = 13) := sorry

end circumcircle_eq_l303_303616


namespace hair_growth_l303_303156

theorem hair_growth (initial final : ℝ) (h_init : initial = 18) (h_final : final = 24) : final - initial = 6 :=
by
  sorry

end hair_growth_l303_303156


namespace product_of_divisors_of_18_l303_303318

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303318


namespace sheets_in_stack_l303_303866

theorem sheets_in_stack (sheets : ℕ) (thickness : ℝ) (h1 : sheets = 400) (h2 : thickness = 4) :
    let thickness_per_sheet := thickness / sheets
    let stack_height := 6
    (stack_height / thickness_per_sheet = 600) :=
by
  sorry

end sheets_in_stack_l303_303866


namespace volume_of_second_cube_l303_303444

noncomputable def cube_volume (s : ℝ) : ℝ := s^3
noncomputable def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

theorem volume_of_second_cube :
  (∃ s₁ s₂ : ℝ, cube_volume s₁ = 8 ∧
                 cube_surface_area s₂ = 3 * cube_surface_area s₁ ∧
                 cube_volume s₂ = 24 * real.sqrt 3) :=
by
  sorry

end volume_of_second_cube_l303_303444


namespace no_integer_solutions_2_pow_2x_minus_5_pow_2y_eq_75_l303_303010

theorem no_integer_solutions_2_pow_2x_minus_5_pow_2y_eq_75 :
  ∀ x y : ℤ, 2^(2*x) - 5^(2*y) ≠ 75 :=
by
  intros x y
  sorry

end no_integer_solutions_2_pow_2x_minus_5_pow_2y_eq_75_l303_303010


namespace multiply_by_5_l303_303237

theorem multiply_by_5 (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 := by
  sorry

end multiply_by_5_l303_303237


namespace find_a_b_subtract_l303_303739

variables (a b : ℝ)

def f (x : ℝ) := a * x + b
def g (x : ℝ) := -4 * x + 3
def h (x : ℝ) := f(g x)
def h_inv (x : ℝ) := x - 9

theorem find_a_b_subtract (H1 : ∀ x, h x = x + 9) : a - b = -10 :=
by {
  sorry
}

end find_a_b_subtract_l303_303739


namespace sqrt_sum_squares_l303_303843

theorem sqrt_sum_squares :
  sqrt ((5 - 3 * sqrt 5) ^ 2) + sqrt ((5 + 3 * sqrt 5) ^ 2) = 6 * sqrt 5 :=
by
  sorry

end sqrt_sum_squares_l303_303843


namespace horner_value_is_v3_base_6_to_decimal_l303_303861

-- Polynomial evaluation using Horner's method

def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.reverse.foldl (λ acc c, acc * x + c) 0

theorem horner_value_is_v3 : 
  horner [3, 5, 6, 79, -8, 35, 12] (-4) = -57 :=
by
  sorry

-- Conversion from base 6 to decimal

def from_base (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.reverse.foldl (λ acc d, acc * b + d) 0

theorem base_6_to_decimal : 
  from_base 6 [2, 1, 0] = 78 :=
by
  sorry

end horner_value_is_v3_base_6_to_decimal_l303_303861


namespace product_of_divisors_18_l303_303356

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l303_303356


namespace problem1_problem2_l303_303186

-- Define the function f
def f (m x : ℝ) : ℝ := m * x - m / x - 2 * Real.log x

-- Problem 1
theorem problem1 (x : ℝ) (h : x > 1) : f 1 x > 0 :=
sorry

-- Problem 2
theorem problem2 : ∃ m : ℝ, (∀ x, 1 ≤ x ∧ x ≤ Real.sqrt 3 → f m x < 2) ∧ 
                (-∞ < m ∧ m < Real.sqrt 3 * (1 + Real.log (Real.sqrt 3))) :=
sorry

end problem1_problem2_l303_303186


namespace probability_three_correct_l303_303583

open Fintype

/-- 
The probability that exactly three out of five packages are delivered 
to the correct houses, given random delivery, is 1/6.
-/
theorem probability_three_correct : 
  (∃ (X : Finset (Fin 5)) (hX : X.card = 3) (Y : Finset (Fin 5)) (hY : Y.card = 2) (f : (Fin 5) → (Fin 5)),
    ∀ (x ∈ X, f x = x) ∧ (∀ y ∈ Y, f y ≠ y) ∧ HasDistribNeg.negOne Y y 
    ∧ ∑ y in Y, 1! = 2) 
  → (Real.ofRat (⅙)) := sorry

end probability_three_correct_l303_303583


namespace sum_of_subsets_is_power_of_two_l303_303587

theorem sum_of_subsets_is_power_of_two :
  let S := finset.Icc 1 1999 in
  let f (s: finset ℕ) : ℕ := s.sum id in
  let total_sum := f S in
  total_sum = 1999000 ∧
  (∑ E in S.powerset, (f E : ℚ) / (f S : ℚ)) = 2 ^ 1998 :=
by {
  let S := finset.Icc 1 1999,
  let f : finset ℕ → ℕ := fun s => s.sum id,
  let total_sum := f S,
  have h_total_sum : total_sum = 1999000 := sorry,
  have h_sum :
    ∑ E in S.powerset, (f E : ℚ) / (f S : ℚ) = 2 ^ 1998 := sorry,
  exact ⟨h_total_sum, h_sum⟩
}

end sum_of_subsets_is_power_of_two_l303_303587


namespace solve_cubic_equation_l303_303233

open Real

noncomputable def is_solution (x : ℝ) : Prop :=
  (cbrt (x^2 + 2 * x) + cbrt (3 * x^2 + 6 * x - 4) = cbrt (x^2 + 2 * x - 4))

theorem solve_cubic_equation (x : ℝ) :
  is_solution x ↔ (x = -2 ∨ x = 0) :=
by sorry

end solve_cubic_equation_l303_303233


namespace original_price_of_shoes_l303_303187

theorem original_price_of_shoes (P : ℝ) (h1 : 0.80 * P = 480) : P = 600 := 
by
  sorry

end original_price_of_shoes_l303_303187


namespace scientific_notation_l303_303762

theorem scientific_notation : (10374 * 10^9 : Real) = 1.037 * 10^13 :=
by
  sorry

end scientific_notation_l303_303762


namespace product_of_divisors_of_18_is_5832_l303_303407

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l303_303407


namespace circumcircle_fixed_point_l303_303731

open EuclideanGeometry

-- Define the given conditions and the theorem
theorem circumcircle_fixed_point
  (A B C E F : Point)
  (h_acute : acute_triangle A B C)
  (hE : E ∈ line A C)
  (hF : F ∈ line A B)
  (h_eq : dist B C ^ 2 = dist B A * dist B F + dist C E * dist C A) :
  ∃ P, P ≠ A ∧ Circumcircle A E F = Circumcircle A P :=
sorry

end circumcircle_fixed_point_l303_303731


namespace find_b_l303_303805

theorem find_b 
  (b : ℝ) 
  (h₁ : ∃ r : ℝ, r * 120 = b) 
  (h₂ : ∃ r : ℝ, b * r = 60 / 24) 
  (h₃ : b > 0) : 
  b = 10 * real.sqrt 3 := 
sorry

end find_b_l303_303805


namespace simplify_trig_expression_l303_303776

theorem simplify_trig_expression :
  (tan 20 + tan 30 + tan 80 + tan 70) / cos 40 = (4 + 2 * sec 40) / sqrt 3 :=
by sorry

end simplify_trig_expression_l303_303776


namespace value_of_f_at_2_l303_303108

def f (x : ℤ) : ℤ := x^3 - x^2 + 2*x - 1

theorem value_of_f_at_2 : f 2 = 7 := 
by
  have : f 2 = 2^3 - 2^2 + 2 * 2 - 1 := rfl
  calc
    f 2 = 2^3 - 2^2 + 2 * 2 - 1 : by rw this
    ... = 8 - 4 + 4 - 1 : by norm_num
    ... = 7 : by norm_num

end value_of_f_at_2_l303_303108


namespace apples_for_juice_l303_303244

theorem apples_for_juice (total_apples : ℝ) (mixed_percentage : ℝ) (juice_percentage : ℝ) :
  total_apples = 5.5 ∧ mixed_percentage = 0.2 ∧ juice_percentage = 0.5 → 
  total_apples * (1 - mixed_percentage) * juice_percentage = 2.2 :=
by
  intro h
  cases h with h_total_apples h_remaining
  cases h_remaining with h_mixed_percentage h_juice_percentage
  rw [h_total_apples, h_mixed_percentage, h_juice_percentage]
  norm_num
  sorry

end apples_for_juice_l303_303244


namespace product_of_divisors_of_18_l303_303421

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l303_303421


namespace product_of_divisors_18_l303_303294

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l303_303294


namespace BT_plus_CT_leq_2R_l303_303858

variables {A B C D M N T : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M] [MetricSpace N] [MetricSpace T]

-- Define the given conditions
-- condition 1: ∠A is the smallest angle in ∆ABC
def smallest_angle_A (A B C : Type) : Prop := sorry

-- condition 2: Point D is on the arc BC of the circumcircle of ∆ABC that does not contain A
def on_arc_BC_not_containing_A (A B C D : Type) : Prop := sorry

-- condition 3: The perpendicular bisectors of segments AB and AC intersect line AD at points M and N, respectively
def perpendicular_bisector_intersections (A B C D M N : Type) : Prop := sorry

-- condition 4: Point T is the intersection of lines BM and CN
def T_intersection_of_BM_and_CN {A B C D M N T : Type} [IntersectionPoint B M T] [IntersectionPoint C N T] : Prop :=
  sorry

-- condition 5: R is the radius of the circumcircle of ∆ABC
def radius_of_circumcircle_ABC (A B C : Type) (R : ℝ) : Prop := sorry

-- Result to be proved: BT + CT ≤ 2R
theorem BT_plus_CT_leq_2R (A B C D M N T : Type) (R : ℝ)
  [smallest_angle_A A B C] [on_arc_BC_not_containing_A A B C D]
  [perpendicular_bisector_intersections A B C D M N]
  [T_intersection_of_BM_and_CN A B C M N T]
  [radius_of_circumcircle_ABC A B C R] :
  BT + CT ≤ 2R :=
sorry

end BT_plus_CT_leq_2R_l303_303858


namespace inequality_neg_multiply_l303_303106

theorem inequality_neg_multiply {a b : ℝ} (h : a > b) : -2 * a < -2 * b :=
sorry

end inequality_neg_multiply_l303_303106


namespace sphere_circumscribed_around_pyramid_l303_303785

noncomputable def circumscribed_sphere_radius (r h : ℝ) : ℝ :=
  sqrt (r^2 + (h / 2)^2)

theorem sphere_circumscribed_around_pyramid (r h : ℝ) :
  ∃ R : ℝ, R = circumscribed_sphere_radius r h :=
by
  use circumscribed_sphere_radius r h
  sorry

end sphere_circumscribed_around_pyramid_l303_303785


namespace second_piece_weight_l303_303891

theorem second_piece_weight (w1 : ℝ) (s1 : ℝ) (s2 : ℝ) (w2 : ℝ) :
  (s1 = 4) → (w1 = 16) → (s2 = 6) → w2 = w1 * (s2^2 / s1^2) → w2 = 36 :=
by
  intro h_s1 h_w1 h_s2 h_w2
  rw [h_s1, h_w1, h_s2] at h_w2
  norm_num at h_w2
  exact h_w2

end second_piece_weight_l303_303891


namespace product_of_divisors_18_l303_303295

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l303_303295


namespace geoff_walk_probability_l303_303595

noncomputable def probability (n : ℕ) : ℚ :=
  if h : 1 ≤ n then (1 / n : ℚ) else 0

def walk_probability (p : ℚ) : Prop :=
  let N := (10^4 * p).floor in
  N = 8101

theorem geoff_walk_probability : ∃ p : ℚ, 
  (∀ n : ℕ, n < 40 → (p < (2 / 40)^n)) → walk_probability p :=
sorry

end geoff_walk_probability_l303_303595


namespace product_of_divisors_18_l303_303348

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l303_303348


namespace train_speed_l303_303487

def length_train : ℝ := 20
def time_to_cross_pole : ℝ := 0.49996000319974404
def speed_train_in_kmph (length : ℝ) (time : ℝ) : ℝ := (length / time) * 3.6

theorem train_speed :
  speed_train_in_kmph length_train time_to_cross_pole ≈ 144.01 :=
by
  sorry

end train_speed_l303_303487


namespace calculate_expression_l303_303911

theorem calculate_expression : 
  (Real.sqrt 3) ^ 0 + 2 ^ (-1:ℤ) + Real.sqrt 2 * Real.cos (Float.pi / 4) - Real.abs (-1/2) = 2 := 
by
  sorry

end calculate_expression_l303_303911


namespace line_equation_slope_point_l303_303471

theorem line_equation_slope_point (m b : ℝ) (h1 : m = -3) (h2 : ∀ x y : ℝ, (x, y) = (2, 4) → y = m * x + b) : m + b = 7 :=
by
  subst h1
  obtain ⟨x, y, h3⟩ := (2, 4)
  exact h2 x y h3
  sorry

end line_equation_slope_point_l303_303471


namespace sage_win_strategy_l303_303962

-- Define labels for hat colors
def HatColor := ℤ -- using integers modulo 3 for hat colors

-- Defining each hat color
def red : HatColor := 0
def blue : HatColor := 1
def green : HatColor := 2

-- Define the guesses of each sage in terms of their neighbors' hat colors
def guess_A (b d : HatColor) : HatColor := (b + d) % 3
def guess_B (a c : HatColor) : HatColor := (- (a + c)) % 3
def guess_C (b d : HatColor) : HatColor := (b - d) % 3
def guess_D (c a : HatColor) : HatColor := (c - a) % 3

-- State the main theorem
theorem sage_win_strategy (a b c d : HatColor) (h1 : guess_A b d ≠ a)
                          (h2 : guess_B a c ≠ b) (h3 : guess_C b d ≠ c)
                          (h4 : guess_D c a ≠ d) : false :=
by {
  -- The proof, ensuring that at least one sage guesses their hat color correctly
  -- This part would represent the contradiction derived from the given conditions
  sorry
}

end sage_win_strategy_l303_303962


namespace range_of_a_l303_303076

theorem range_of_a (a : ℝ) (h1 : a > 0) :
  let ellipse := (x : ℝ) (y : ℝ) (a : ℝ) := x^2 + (1/2) * y^2 - a^2
  let A := (2 : ℝ, 1 : ℝ)
  let B := (4 : ℝ, 3 : ℝ)
  (ellipse 2 1 a > 0 ∧ ellipse 4 3 a > 0) ∨ (ellipse 2 1 a < 0 ∧ ellipse 4 3 a < 0)
  ↔ (0 < a ∧ a < (3 * Real.sqrt 2 / 2)) ∨ (a > Real.sqrt 82 / 2) :=
by
  sorry


end range_of_a_l303_303076


namespace initial_mixture_volume_l303_303473

/--
Given:
1. A mixture initially contains 20% water.
2. When 13.333333333333334 liters of water is added, water becomes 25% of the new mixture.

Prove that the initial volume of the mixture is 200 liters.
-/
theorem initial_mixture_volume (V : ℝ) (h1 : V > 0) (h2 : 0.20 * V + 13.333333333333334 = 0.25 * (V + 13.333333333333334)) : V = 200 :=
sorry

end initial_mixture_volume_l303_303473


namespace product_of_divisors_of_18_l303_303317

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303317


namespace matrix_sum_property_l303_303857

variables {m n : ℕ}
variables (A B : matrix (fin m) (fin n) ℕ)

-- Conditions
def non_decreasing_rows (M : matrix (fin m) (fin n) ℕ) : Prop :=
  ∀ i : fin m, ∀ j1 j2 : fin n, j1 ≤ j2 → M i j1 ≤ M i j2

def non_decreasing_cols (M : matrix (fin m) (fin n) ℕ) : Prop :=
  ∀ j : fin n, ∀ i1 i2 : fin m, i1 ≤ i2 → M i1 j ≤ M i2 j

def sum_top_k_rows (M : matrix (fin m) (fin n) ℕ) (k : fin (m + 1)) : ℕ :=
  ∑ i in range k, ∑ j in range n, M i j

def sum_left_l_cols (M : matrix (fin m) (fin n) ℕ) (l : fin (n + 1)) : ℕ :=
  ∑ i in range m, ∑ j in range l, M i j

theorem matrix_sum_property
  (h_rows_A : non_decreasing_rows A)
  (h_rows_B : non_decreasing_rows B)
  (h_cols_A : non_decreasing_cols A)
  (h_cols_B : non_decreasing_cols B)
  (h_sum1 : ∀ k : fin (m + 1), sum_top_k_rows A k ≥ sum_top_k_rows B k)
  (h_sum2 : ∑ i in range m, ∑ j in range n, A i j = ∑ i in range m, ∑ j in range n, B i j) :
  ∀ l : fin (n + 1), sum_left_l_cols A l ≤ sum_left_l_cols B l :=
sorry

end matrix_sum_property_l303_303857


namespace train_speed_l303_303892

theorem train_speed
  (length : ℝ) (time : ℝ) (speed_in_kmph : ℝ) 
  (h1 : length = 100)
  (h2 : time = 3.9996800255979523)
  (h3 : speed_in_kmph = (length / time) * 3.6) :
  speed_in_kmph ≈ 90.003 :=
by
  sorry

end train_speed_l303_303892


namespace find_n_arithmetic_sequence_l303_303074

-- Given conditions
def a₁ : ℕ := 20
def aₙ : ℕ := 54
def Sₙ : ℕ := 999

-- Arithmetic sequence sum formula and proof statement of n = 27
theorem find_n_arithmetic_sequence
  (a₁ : ℕ)
  (aₙ : ℕ)
  (Sₙ : ℕ)
  (h₁ : a₁ = 20)
  (h₂ : aₙ = 54)
  (h₃ : Sₙ = 999) : ∃ n : ℕ, n = 27 := 
by
  sorry

end find_n_arithmetic_sequence_l303_303074


namespace find_y_l303_303146

/-- Representing the angles and the conditions -/
variables (AB CD BMN MND NMP MPN MNP : ℝ)
variables (parallel_ab_cd : AB = CD)
variables (angle_bmn : BMN = 2 * BMN)
variables (angle_mnd : MND = 70)
variables (angle_nmp : NMP = 70)
variables (angle_mpn : MPN = BMN / 2)

/-- Given that AB is parallel to CD and the angles are as defined, prove that y = 55 degrees. -/
theorem find_y (AB CD BMN MND NMP MPN MNP : ℝ) (parallel_ab_cd : AB = CD) (angle_bmn : BMN = 2 * MPN) (angle_mnd : MND = 70)
  (angle_nmp : NMP = 70) (angle_mpn : MPN = BMN / 2) : MNP = 55 :=
by {
  -- Use the property that angle BMN + angle MND must equal 180 degrees because they form a straight line
  have h1 : BMN + MND = 180, from sorry,
  -- Substitute the known values
  have h2 : 2 * BMN / 2 + 70 = 180, from sorry,
  -- Solve for BMN
  have h3 : BMN = 110 / 2, from sorry,
  -- Simplify to get BMN
  have h4 : BMN = 55, from sorry,
  -- Use the triangle sum property (sum of angles of triangle MNP is 180 degrees)
  have h5 : angle_bmn + angle_nmp + MNP = 180, from sorry,
  -- Substitute the known values into the triangle property
  show MNP = 55, from sorry
}

end find_y_l303_303146


namespace product_of_divisors_of_18_l303_303289

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l303_303289


namespace complex_fraction_simplification_l303_303229

theorem complex_fraction_simplification (i : ℂ) (hi : i^2 = -1) : 
  ((2 - i) / (1 + 4 * i)) = (-2 / 17 - (9 / 17) * i) :=
  sorry

end complex_fraction_simplification_l303_303229


namespace range_of_p_l303_303044

-- Definitions of A and B
def A (p : ℝ) := {x : ℝ | x^2 + (p + 2) * x + 1 = 0}
def B := {x : ℝ | x > 0}

-- Condition of the problem: A ∩ B = ∅
def condition (p : ℝ) := ∀ x ∈ A p, x ∉ B

-- The statement to prove: p > -4
theorem range_of_p (p : ℝ) : condition p → p > -4 :=
by
  intro h
  sorry

end range_of_p_l303_303044


namespace product_of_divisors_of_18_l303_303312

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303312


namespace calculate_expression_l303_303909

theorem calculate_expression : (Real.sqrt 3)^0 + 2^(-1) + Real.sqrt 2 * Real.cos (Float.pi / 4) - abs (-1 / 2) = 2 := 
by
  sorry

end calculate_expression_l303_303909


namespace prob_remainder_mod_1000_l303_303527

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ 
| n 0 := 1
| 0 k := 0
| n k := binom (n-1) (k-1) * n / k

-- Define the sum we are interested in, only including indices that are multiples of 4
def sum_binom_2011_multiple_4 : ℕ :=
  (Finset.range (2012 / 4 + 1)).sum (λ i, binom 2011 (4 * i))

-- The statement we want to prove
theorem prob_remainder_mod_1000 : 
  sum_binom_2011_multiple_4 % 1000 = 12 := 
sorry

end prob_remainder_mod_1000_l303_303527


namespace a5_plus_a6_l303_303978

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a n = a 0 + d * n

def sum_of_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n + 1) * a 0 + d * n * (n + 1) / 2 where d := (a 1 - a 0)

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
hypothesis h1 : arithmetic_sequence a
hypothesis h2 : sum_of_terms a S
hypothesis h3 : S 2 = 4
hypothesis h4 : S 4 = 16

theorem a5_plus_a6 :
  a 5 + a 6 = 20 :=
sorry

end a5_plus_a6_l303_303978


namespace henry_has_30_more_lollipops_than_alison_l303_303668

noncomputable def num_lollipops_alison : ℕ := 60
noncomputable def num_lollipops_diane : ℕ := 2 * num_lollipops_alison
noncomputable def total_num_days : ℕ := 6
noncomputable def num_lollipops_per_day : ℕ := 45
noncomputable def total_lollipops : ℕ := total_num_days * num_lollipops_per_day
noncomputable def num_lollipops_total_ad : ℕ := num_lollipops_alison + num_lollipops_diane
noncomputable def num_lollipops_henry : ℕ := total_lollipops - num_lollipops_total_ad
noncomputable def lollipops_diff_henry_alison : ℕ := num_lollipops_henry - num_lollipops_alison

theorem henry_has_30_more_lollipops_than_alison :
  lollipops_diff_henry_alison = 30 :=
by
  unfold lollipops_diff_henry_alison
  unfold num_lollipops_henry
  unfold num_lollipops_total_ad
  unfold total_lollipops
  sorry

end henry_has_30_more_lollipops_than_alison_l303_303668


namespace max_marks_calculation_l303_303757

theorem max_marks_calculation (marks_scored : ℕ) (marks_short : ℕ) (pass_percentage : ℝ) 
  (h1 : marks_scored = 212) (h2 : marks_short = 76) (h3 : pass_percentage = 0.50) : 
  let marks_needed_to_pass := marks_scored + marks_short in
  let maximum_marks := marks_needed_to_pass / pass_percentage in
  maximum_marks = 576 :=
by
  -- Given conditions
  have h_marks_pass : marks_needed_to_pass = 288 := by
    rw [h1, h2]
    exact rfl
  sorry

end max_marks_calculation_l303_303757


namespace find_x_values_l303_303945

theorem find_x_values (x : ℝ) : 
  x ≠ -1 ∧ x ≠ -5 →
  (1 / (x + 1) + 6 / (x + 5) ≥ 1 ↔ x ∈ Set.Icc (-5) (-2) ∪ Set.Icc (-1) 3) :=
by
  intro h
  split
  sorry  -- Split for completing the proof by two-way implication.

end find_x_values_l303_303945


namespace SammyFinishedProblems_l303_303223

def initial : ℕ := 9 -- number of initial math problems
def remaining : ℕ := 7 -- number of remaining math problems
def finished (init rem : ℕ) : ℕ := init - rem -- defining number of finished problems

theorem SammyFinishedProblems : finished initial remaining = 2 := by
  sorry -- placeholder for proof

end SammyFinishedProblems_l303_303223


namespace walnut_tree_logs_l303_303726

theorem walnut_tree_logs : 
  ∀ (logs_per_pine logs_per_maple logs_total logs_per_walnut_to_find : ℕ),
    logs_per_pine = 80 → 
    logs_per_maple = 60 → 
    logs_total = 1220 → 
    let total_pine := 8 * logs_per_pine in
    let total_maple := 3 * logs_per_maple in
    let total_walnut := logs_total - total_pine - total_maple in
    4 * logs_per_walnut_to_find = total_walnut →
    logs_per_walnut_to_find = 100 :=
by 
  intros logs_per_pine logs_per_maple logs_total logs_per_walnut_to_find h_pine h_maple h_total total_pine total_maple total_walnut h_walnut.
  sorry

end walnut_tree_logs_l303_303726


namespace product_of_divisors_of_18_l303_303290

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l303_303290


namespace geometric_sequence_count_l303_303671

def distinct_digits (a b c d : ℕ) : Prop :=
a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def nonzero_first_digit (a : ℕ) : Prop :=
a ≠ 0

def increasing_geometric_sequence (ab bc cd : ℕ) (r : ℕ) : Prop :=
bc = ab * r ∧ cd = bc * r

noncomputable def num_geometric_four_digit_integers : ℕ :=
∑ n in finset.range 10000, 
  let a := n / 1000,
      b := (n / 100) % 10,
      c := (n / 10) % 10,
      d := n % 10,
      ab := 10 * a + b,
      bc := 10 * b + c,
      cd := 10 * c + d,
      r := bc / ab in
  if nonzero_first_digit a ∧ distinct_digits a b c d ∧ increasing_geometric_sequence ab bc cd r then 1 else 0

theorem geometric_sequence_count : 
  num_geometric_four_digit_integers = 10 := 
sorry

end geometric_sequence_count_l303_303671


namespace maximal_subsets_upper_bound_l303_303588

noncomputable def length (U : Set (ℝ × ℝ)) : ℝ :=
  (U.toFinset.sum id).norm

def is_maximal (V : Set (ℝ × ℝ)) (B : Set (ℝ × ℝ)) : Prop :=
  ∀ (A : Set (ℝ × ℝ)), A ≠ ∅ ∧ A ⊆ V → length B ≥ length A

theorem maximal_subsets_upper_bound (V : Set (ℝ × ℝ)) (n : ℕ)
  (hn : n ≥ 1) (HV : V.finite) (hV_card : V.toFinset.card = n) :
  (∃ S : Set (Set (ℝ × ℝ)), ∀ B ∈ S, is_maximal V B ∧ S.finite ∧ S.toFinset.card ≤ 2 * n) :=
sorry

end maximal_subsets_upper_bound_l303_303588


namespace inequality_sqrt_three_l303_303206

theorem inequality_sqrt_three (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h : a^2 + b^2 + c^2 = 1) : 
  (ab / c + bc / a + ca / b) ≥ √3 :=
by
  sorry

end inequality_sqrt_three_l303_303206


namespace product_of_divisors_of_18_l303_303291

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l303_303291


namespace tractor_planting_rate_l303_303566

theorem tractor_planting_rate
  (acres : ℕ) (days : ℕ) (first_crew_tractors : ℕ) (first_crew_days : ℕ) 
  (second_crew_tractors : ℕ) (second_crew_days : ℕ) 
  (total_acres : ℕ) (total_days : ℕ) 
  (first_crew_days_calculated : ℕ) 
  (second_crew_days_calculated : ℕ) 
  (total_tractor_days : ℕ) 
  (acres_per_tractor_day : ℕ) :
  total_acres = acres → 
  total_days = days → 
  first_crew_tractors * first_crew_days = first_crew_days_calculated → 
  second_crew_tractors * second_crew_days = second_crew_days_calculated → 
  first_crew_days_calculated + second_crew_days_calculated = total_tractor_days → 
  total_acres / total_tractor_days = acres_per_tractor_day → 
  acres_per_tractor_day = 68 :=
by
  intros
  sorry

end tractor_planting_rate_l303_303566


namespace product_of_divisors_18_l303_303354

theorem product_of_divisors_18 : ∏ d in (finset.filter (∣ 18) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_18_l303_303354


namespace product_of_divisors_of_18_l303_303413

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l303_303413


namespace bernard_blue_notebooks_l303_303905

theorem bernard_blue_notebooks :
  ∀ (red blue white gave_left total_left : ℕ), 
    red = 15 →
    white = 19 →
    total_left = 5 →
    gave_left = 46 →
    total_left + gave_left = 51 →
    total_left + gave_left - (red + white) = 17 :=
by
  intros red blue white gave_left total_left
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  exact sorry

end bernard_blue_notebooks_l303_303905


namespace max_dot_product_l303_303612

-- Define ellipse and points F1, F2, A, and P
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

def F1 : (ℝ × ℝ) := (-1, 0)
def F2 : (ℝ × ℝ) := (1, 0)
def A : (ℝ × ℝ) := (1, 1.5)

def F1_vec (P : ℝ × ℝ) : (ℝ × ℝ) := (P.1 + 1, P.2)
def F2A_vec : (ℝ × ℝ) := (0, 1.5)

-- Dot product of vectors F1P and F2A
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define problem in Lean
theorem max_dot_product :
  ∀ P : ℝ × ℝ, ellipse P.1 P.2 → dot_product (F1_vec P) F2A_vec ≤ 3 * sqrt 3 / 2 := by
  sorry

end max_dot_product_l303_303612


namespace binomial_sum_mod_1000_l303_303515

open BigOperators

theorem binomial_sum_mod_1000 :
  ((∑ k in finset.range 503 \ finset.range 3, nat.choose 2011 (4 * k)) % 1000) = 49 := 
sorry

end binomial_sum_mod_1000_l303_303515


namespace product_of_divisors_18_l303_303298

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l303_303298


namespace solve_system_l303_303216

/-- Given the system of equations:
    3 * (x + y) - 4 * (x - y) = 5
    (x + y) / 2 + (x - y) / 6 = 0
  Prove that the solution is x = -1/3 and y = 2/3 
-/
theorem solve_system (x y : ℚ) 
  (h1 : 3 * (x + y) - 4 * (x - y) = 5)
  (h2 : (x + y) / 2 + (x - y) / 6 = 0) : 
  x = -1 / 3 ∧ y = 2 / 3 := 
sorry

end solve_system_l303_303216


namespace birds_meeting_distance_l303_303791

theorem birds_meeting_distance :
  ∀ (d distance speed1 speed2: ℕ),
  distance = 20 →
  speed1 = 4 →
  speed2 = 1 →
  (d / speed1) = ((distance - d) / speed2) →
  d = 16 :=
by
  intros d distance speed1 speed2 hdist hspeed1 hspeed2 htime
  sorry

end birds_meeting_distance_l303_303791


namespace tractor_planting_rate_l303_303568

theorem tractor_planting_rate
  (acres : ℕ) (days : ℕ) (first_crew_tractors : ℕ) (first_crew_days : ℕ) 
  (second_crew_tractors : ℕ) (second_crew_days : ℕ) 
  (total_acres : ℕ) (total_days : ℕ) 
  (first_crew_days_calculated : ℕ) 
  (second_crew_days_calculated : ℕ) 
  (total_tractor_days : ℕ) 
  (acres_per_tractor_day : ℕ) :
  total_acres = acres → 
  total_days = days → 
  first_crew_tractors * first_crew_days = first_crew_days_calculated → 
  second_crew_tractors * second_crew_days = second_crew_days_calculated → 
  first_crew_days_calculated + second_crew_days_calculated = total_tractor_days → 
  total_acres / total_tractor_days = acres_per_tractor_day → 
  acres_per_tractor_day = 68 :=
by
  intros
  sorry

end tractor_planting_rate_l303_303568


namespace product_of_divisors_18_l303_303329

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l303_303329


namespace product_of_divisors_of_18_l303_303372

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303372


namespace dihedral_angle_cosine_l303_303837

noncomputable def cos_dihedral_angle : ℚ := 1 / 3

theorem dihedral_angle_cosine (R r : ℝ) (θ : ℝ) : 
  R = 3 * r → 
  ∃ edge_angle : ℝ, edge_angle = π / 3 ∧ 
  cos θ = cos_dihedral_angle → 
  cos θ = 1 / 3 :=
by
  intros h1 h2.
  sorry

end dihedral_angle_cosine_l303_303837


namespace geometric_sequence_a6_l303_303975

theorem geometric_sequence_a6 : 
  ∀ (a : ℕ → ℚ), (∀ n, a n ≠ 0) → a 1 = 3 → (∀ n, 2 * a (n+1) - a n = 0) → a 6 = 3 / 32 :=
by
  intros a h1 h2 h3
  sorry

end geometric_sequence_a6_l303_303975


namespace product_of_divisors_18_l303_303302

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l303_303302


namespace ellipse_equation_range_of_m_l303_303651

-- Definitions based on conditions
noncomputable def a := Real.sqrt 3
noncomputable def b := 1
noncomputable def c := Real.sqrt 2
noncomputable def e := Real.sqrt 6 / 3

-- Definition of ellipse and conditions encapsulated
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def focus1 : ℝ × ℝ := (-c, 0)
def focus2 : ℝ × ℝ := (c, 0)
def perimeter_condition (xA yA xB yB : ℝ) : Prop := 
  dist (xA, yA) focus1 + dist (xA, yA) focus2 + dist (xB, yB) focus2 = 4 * Real.sqrt 3
def eccentricity_condition : Prop := e = Real.sqrt 6 / 3

-- First theorem: Equation of the ellipse
theorem ellipse_equation : 
  (∃ xA yA xB yB : ℝ, perimeter_condition xA yA xB yB) ∧ eccentricity_condition → ∀ x y, ellipse x y = ((x^2)/3 + y^2 = 1) :=
sorry

-- Definitions related to the second question
def line_intersects_ellipse (k m x y : ℝ) : Prop := y = k * x + m ∧ ellipse x y
def vertices : ℝ × ℝ := (0, -b)
def conditions_MN (k m xM yM xN yN : ℝ) : Prop := line_intersects_ellipse k m xM yM ∧ line_intersects_ellipse k m xN yN ∧ |dist (vertices) (xM, yM)| = |dist (vertices) (xN, yN)|

-- Second theorem: Range of m
theorem range_of_m (k : ℝ) :
  (∃ xM yM xN yN : ℝ, conditions_MN k m xM yM xN yN) → 1/2 ≤ m ∧ m < 2 :=
sorry

end ellipse_equation_range_of_m_l303_303651


namespace product_of_divisors_of_18_is_5832_l303_303406

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l303_303406


namespace product_of_divisors_of_18_is_5832_l303_303408

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l303_303408


namespace ethanol_to_acetic_acid_6_moles_l303_303951

-- We define our reaction and assumptions
def ethanol_to_acetic_acid_reaction (ethanol moles) (oxygen moles) : Nat :=
  2 * ethanol

-- Our statement to verify the problem above:
theorem ethanol_to_acetic_acid_6_moles (ethanol moles : Nat) (oxygen moles : Nat)
  (h1 : ethanol = 3) (h2 : oxygen = 3) :
  ethanol_to_acetic_acid_reaction ethanol oxygen = 6 :=
by
  rw [h1, h2]
  exact rfl

end ethanol_to_acetic_acid_6_moles_l303_303951


namespace product_of_divisors_18_l303_303324

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l303_303324


namespace total_time_correct_l303_303755

-- Definitions for the conditions
def dean_time : ℕ := 9
def micah_time : ℕ := (2 * dean_time) / 3
def jake_time : ℕ := micah_time + micah_time / 3

-- Proof statement for the total time
theorem total_time_correct : micah_time + dean_time + jake_time = 23 := by
  sorry

end total_time_correct_l303_303755


namespace polygon_permutation_possible_l303_303217

theorem polygon_permutation_possible (n : ℕ) : 
  (∃ a : ℕ, ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
    (a + k) % n ∈ finset.range (n + 1)) ↔ (n % 4 ≠ 2) :=
begin
  sorry
end

end polygon_permutation_possible_l303_303217


namespace find_angle_B_l303_303660

theorem find_angle_B (a b c : ℝ) (A B C : ℝ)
  (h1 : (sin B - sin A, sqrt 3 * a + c) = (sin C, a + b)) :
  B = 5 * Real.pi / 6 :=
sorry

end find_angle_B_l303_303660


namespace number_of_zeros_f_l303_303541

def f (x : ℝ) : ℝ := if x > -1 ∧ x ≤ 4 then x^2 - 2^x else sorry

theorem number_of_zeros_f :
  (∑ i in (Finset.range 2014).filter(λ i, f i = 0), 1) = 604 :=
by sorry

end number_of_zeros_f_l303_303541


namespace triangle_ABI_CDIE_bisector_area_equal_l303_303248

open Real
open EuclideanGeometry

/-- Given a triangle ABC with CA = 9, CB = 4, and the bisectors AD and BE intersecting at I,
along with the areas of triangle ABI and quadrilateral CDIE being equal, 
prove that AB must be equal to 6. -/
theorem triangle_ABI_CDIE_bisector_area_equal 
  (A B C D E I : Point)
  (h1 : dist A C = 9)
  (h2 : dist B C = 4)
  (h3 : is_angle_bisector A D B ∧ is_angle_bisector B E C)
  (h4 : ∃ S₁ S₂ : ℝ, S₁ = triangle_area A B I ∧ S₂ = quadrilateral_area C D I E ∧ S₁ = S₂) :
  dist A B = 6 :=
sorry

end triangle_ABI_CDIE_bisector_area_equal_l303_303248


namespace problem_conditions_l303_303985

variable (a b c d : ℝ)

theorem problem_conditions (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_eq : a^2 + b^2 = ab + 1) (h_cd : cd > 1) :
  a + b ≤ 2 ∧ ¬ (sqrt (a * c) + sqrt (b * d) = c + d) :=
by
  sorry

end problem_conditions_l303_303985


namespace no_stars_in_telescope_view_l303_303220

theorem no_stars_in_telescope_view (n k : ℕ) (α : ℕ → ℝ) 
  (h1 : ∑ j in Finset.range k, α j = (real.pi / n)) : 
  ∃ (θ : ℕ → ℝ), 
  (∀ i j, i ≠ j → θ i ≠ θ j ∧ θ i ≠ θ j + k * real.pi / n) → -- orientations are adjusted
  (∀ i, ¬ (∃ j, θ j ≤ i ∧ i ≤ θ j + α j)) := -- no stars fall into any telescope view
sorry

end no_stars_in_telescope_view_l303_303220


namespace count_digitally_balanced_numbers_l303_303475

def is_digitally_balanced (n : ℕ) : Prop :=
  if n < 1000 then
    let a := n / 100;
    let b := (n % 100) / 10;
    let c := n % 10;
    a + b = c
  else
    let a := n / 1000;
    let b := (n % 1000) / 100;
    let c := (n % 100) / 10;
    let d := n % 10;
    a + b = c + d

theorem count_digitally_balanced_numbers :
  {n : ℕ | 100 ≤ n ∧ n ≤ 9999 ∧ is_digitally_balanced n}.card = 660 :=
by sorry

end count_digitally_balanced_numbers_l303_303475


namespace add_fraction_to_series_eq_one_l303_303461

theorem add_fraction_to_series_eq_one :
  (∑ n in Finset.range 22, (1 : ℚ) / ((n + 2) * (n + 3))) + 13 / 24 = 1 :=
by {
  sorry
}

end add_fraction_to_series_eq_one_l303_303461


namespace probability_one_each_item_l303_303673

theorem probability_one_each_item :
  let num_items := 32
  let total_ways := Nat.choose num_items 4
  let favorable_outcomes := 8 * 8 * 8 * 8
  total_ways = 35960 →
  let probability := favorable_outcomes / total_ways
  probability = (128 : ℚ) / 1125 :=
by
  sorry

end probability_one_each_item_l303_303673


namespace arithmetic_problem_l303_303136

-- The initial sequence definition and sum
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

-- The definition of the sum of the first n terms of an arithmetic sequence
noncomputable def sum_arithmetic_seq (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 0 + a (n - 1))) / 2

-- Define the main problem
theorem arithmetic_problem (a : ℕ → ℤ) (d : ℤ)
  (h_seq : arithmetic_seq a d) (h_d : d = 4) (h_cond : a 1 + a 4 = 22):
  let S_n := λ n, (n * (a 0 + a (n - 1))) / 2 in
  S_n n = 2 * n^2 - n ∧ 
  let T_n := λ n, (∑ i in range n, i / ((2 * i + 1) * S_n i)) in
  T_14 = 14 / 29 :=
sorry

end arithmetic_problem_l303_303136


namespace min_value_of_inv_tan_sum_eq_l303_303127

noncomputable def min_value_of_inv_tan_sum (A B C : ℝ) (hB_bound : cos B = 1 / 4) : ℝ :=
  Inf {(1 / tan A + 1 / tan C) | (A + B + C = π ∧ sin B ≠ 0)}

theorem min_value_of_inv_tan_sum_eq :
  ∀ (A B C : ℝ) (hB_cos : cos B = 1 / 4), min_value_of_inv_tan_sum A B C hB_cos = 2 * sqrt 15 / 5 :=
by sorry

end min_value_of_inv_tan_sum_eq_l303_303127


namespace angle_DEF_eq_45_l303_303724

-- Given definitions used in the problem statement
variables {A B C D E F G : Type} -- Points in the plane
variables [T : Triangle ABC] -- Triangle ABC
variables (BD : Line) -- Bisector BD
variables (DE : Line) -- Bisector DE in triangle ABD
variables (DF : Line) -- Bisector DF in triangle CBD
variables (EF AC : Line) -- Lines EF and AC
variables (h_parallel : EF ∥ AC) -- EF is parallel to AC

-- Goal to prove:
theorem angle_DEF_eq_45
  (h_bisector_BD : IsBisector BD T)
  (h_bisector_DE : IsBisector DE (triangle_of_IsBisector h_bisector_BD))
  (h_bisector_DF : IsBisector DF (triangle_of_LeftOverTriangle h_bisector_BD))
  (h_parallel : IsParallel EF AC) :
  angle_DEF = 45 :=
sorry

end angle_DEF_eq_45_l303_303724


namespace third_term_expansion_l303_303266

theorem third_term_expansion (x : ℝ) : 
  ( (1 - x) * (1 + 2 * x) ^ 5 ).series_term 2 = 30 * x^2 :=
sorry

end third_term_expansion_l303_303266


namespace students_prob_red_light_l303_303485

noncomputable def probability_red_light_encountered (p1 p2 p3 : ℚ) : ℚ :=
  1 - ((1 - p1) * (1 - p2) * (1 - p3))

theorem students_prob_red_light :
  probability_red_light_encountered (1/2) (1/3) (1/4) = 3/4 :=
by
  sorry

end students_prob_red_light_l303_303485


namespace optionA_optionB_optionC_optionD_l303_303701

-- Given an acute triangle ABC with angles A, B, and C
variables {A B C : ℝ}
-- Assume the angles are between 0 and π/2
variable hacute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2
-- Assume the angles sum to π
variable hsum : A + B + C = π

-- Prove each statement
theorem optionA (hA_B : A > B) : sin A > sin B := sorry
theorem optionB (hA_eq : A = π / 3) : ¬(0 < B ∧ B < π / 2) := sorry
theorem optionC : sin A + sin B > cos A + cos B := sorry
theorem optionD : tan B * tan C > 1 := sorry

end optionA_optionB_optionC_optionD_l303_303701


namespace ellipse_problem_solution_l303_303075

noncomputable def ellipse_equation (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a > b) (hvertex : a = 2) (heccentricity : b^2 = a^2 - (a * real.sqrt 2 / 2)^2) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) = (x^2 / 4 + y^2 / 2 = 1)

noncomputable def find_k (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a > b) (area : ℝ) (harea : area = 4 * real.sqrt 2 / 5) : Prop :=
  ∃ k : ℝ, (area = |k| * real.sqrt (4 + 6 * k^2) / (1 + 2 * k^2)) ∧ (k = real.sqrt 2 ∨ k = -real.sqrt 2)

theorem ellipse_problem_solution :
  ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a > b ∧ a = 2 ∧ b^2 = 2 ∧ (ellipse_equation a b sorry sorry sorry sorry sorry) ∧
  (find_k a b sorry sorry sorry (4 * real.sqrt 2 / 5) sorry) :=
begin
  sorry
end

end ellipse_problem_solution_l303_303075


namespace sufficient_but_not_necessary_l303_303600

variable (m: ℝ)
variable p : m > 1
variable q : ∃ x : ℝ, x^2 + 2*m*x + 1 < 0

theorem sufficient_but_not_necessary (m: ℝ) (p : m > 1) (q : ∃ x : ℝ, x^2 + 2*m*x + 1 < 0) : 
  (m > 1 → (∃ x : ℝ, x^2 + 2*m*x + 1 < 0)) ∧ ¬((∃ x : ℝ, x^2 + 2*m*x + 1 < 0) → m > 1) :=
sorry

end sufficient_but_not_necessary_l303_303600


namespace shaded_area_trapezoid_eq_l303_303498

noncomputable def area_trapezoid_ABCD : ℝ := 117
noncomputable def EF_length : ℝ := 13
noncomputable def MN_length : ℝ := 4

theorem shaded_area_trapezoid_eq :
  ∃ (P : Type) (O : P) (ABCD EF MN : set P),
    (trapezoid ABCD) ∧
    (length EF = EF_length) ∧
    (length MN = MN_length) ∧
    (area ABCD = area_trapezoid_ABCD) →
    (let triangles_area := 2 * ((1 / 2) * MN_length * EF_length) in
    area ABCD - triangles_area = 65) :=
begin
  intros,
  sorry
end

end shaded_area_trapezoid_eq_l303_303498


namespace tractor_planting_rate_l303_303564

theorem tractor_planting_rate
  (A : ℕ) (D : ℕ)
  (T1_days : ℕ) (T1 : ℕ)
  (T2_days : ℕ) (T2 : ℕ)
  (total_acres : A = 1700)
  (total_days : D = 5)
  (crew1_tractors : T1 = 2)
  (crew1_days : T1_days = 2)
  (crew2_tractors : T2 = 7)
  (crew2_days : T2_days = 3)
  : (A / (T1 * T1_days + T2 * T2_days)) = 68 := 
sorry

end tractor_planting_rate_l303_303564


namespace sum_segments_not_equal_l303_303052

theorem sum_segments_not_equal
  (n : ℕ)
  (hn : n = 2022) 
  (points : Fin n)
  (adj_points_dist_eq : ∀ i j : Fin n, (j - i).NatAbs > 1 → (points j - points i) = (points (i + 1) - points i)) 
  (red_points : Fin n → Prop)
  (blue_points : Fin n → Prop)
  (half_red_half_blue : ∀ i, (red_points i ↔ i < (n / 2)) ∧ (blue_points i ↔ i ≥ (n / 2)))
  : ¬ (sum (λ (i j : Fin n), if red_points i ∧ blue_points j then (j - i).NatAbs else 0) = 
       sum (λ (i j : Fin n), if blue_points i ∧ red_points j then (j - i).NatAbs else 0)) := 
begin
  sorry,
end

end sum_segments_not_equal_l303_303052


namespace sequence_item_l303_303082

theorem sequence_item (n : ℕ) (a_n : ℕ → Rat) (h : a_n n = 2 / (n^2 + n)) : a_n n = 1 / 15 → n = 5 := by
  sorry

end sequence_item_l303_303082


namespace product_of_divisors_of_18_l303_303369

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303369


namespace solve_for_x_l303_303455

/-- Given condition that 0.75 : x :: 5 : 9 -/
def ratio_condition (x : ℝ) : Prop := 0.75 / x = 5 / 9

theorem solve_for_x (x : ℝ) (h : ratio_condition x) : x = 1.35 := by
  sorry

end solve_for_x_l303_303455


namespace range_t_l303_303261

def seq_a (n : ℕ) : ℝ := sorry -- positive sequence a_n
def seq_b (n : ℕ) : ℝ := sorry -- sequence b_n
def sum_seq_b (n : ℕ) : ℝ := sorry -- sum of first n terms of sequence b_n

axiom a1_pos : ∀ n : ℕ, 0 < seq_a n
axiom a1_init : seq_a 1 = 1
axiom a_recursion : ∀ n : ℕ, (sqrt ((1 / (seq_a n)^2) + 3) = sqrt (1 / (seq_a (n+1))^2))

theorem range_t (t : ℝ) :
  (∀ n : ℕ, sum_seq_b n < t) → t ≥ 1 / 3 :=
sorry

-- Definitions for seq_a and seq_b, and sum_seq_b are to be filled in
-- according to the conditions and their arithmetic properties.

end range_t_l303_303261


namespace conic_section_is_ellipse_l303_303452

def is_ellipse (eq : ∀ x y : ℝ, Real.sqrt ((x-2)^2 + (y+2)^2) + Real.sqrt ((x+3)^2 + (y-4)^2) = 14) : Prop :=
  ∀ x y : ℝ, eq x y

theorem conic_section_is_ellipse :
  is_ellipse (λ x y : ℝ, Real.sqrt ((x-2)^2 + (y+2)^2) + Real.sqrt ((x+3)^2 + (y-4)^2) = 14) :=
sorry

end conic_section_is_ellipse_l303_303452


namespace problem_statement_l303_303181

noncomputable def Fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := Fibonacci (n+1) + Fibonacci n

def polynomial_p (p : ℕ → ℕ) (deg_990 : Prop) (interpolates : ∀ k, 992 ≤ k ∧ k ≤ 1982 → p k = Fibonacci k) : Prop :=
  deg_990 ∧ interpolates ∧ p 1983 = Fibonacci 1983 - 1

theorem problem_statement (p : ℕ → ℕ) (deg_990 : Prop) (interpolates : ∀ k, 992 ≤ k ∧ k ≤ 1982 → p k = Fibonacci k) :
  polynomial_p p deg_990 interpolates :=
sorry

end problem_statement_l303_303181


namespace ratio_of_inscribed_circle_radii_l303_303133

-- Define the conditions for the right triangle and angle bisector.
variables {A B C D : Type*} [RightTriangle A B C] [IsAngleBisector A D (: α)]

-- Define the radii of the inscribed circles in triangles ABD and ADC.
variables {r R : ℝ}

-- The theorem to establish the ratio of the radii.
theorem ratio_of_inscribed_circle_radii (α : ℝ) :
    ∃ (r R : ℝ), 
    let α := α in 
    (r / R) = (sqrt 2 * tan (π / 4 + α / 4)) / (2 * sin (π / 4 + α / 2)) :=
sorry

end ratio_of_inscribed_circle_radii_l303_303133


namespace area_ratio_l303_303832

noncomputable def ratio_of_areas (r : ℝ) : ℝ :=
  let radius_Y : ℝ := r / 3
  let area_OY := real.pi * (radius_Y ^ 2)
  let area_OQ := real.pi * (r ^ 2)
  area_OY / area_OQ

theorem area_ratio (r : ℝ) : ratio_of_areas r = 1 / 9 := by
  sorry

end area_ratio_l303_303832


namespace find_directrix_l303_303086

-- Define the parabola equation
def parabola_eq (x y : ℝ) : Prop := x^2 = 8 * y

-- State the problem to find the directrix of the given parabola
theorem find_directrix (x y : ℝ) (h : parabola_eq x y) : y = -2 :=
sorry

end find_directrix_l303_303086


namespace number_of_correct_statements_l303_303257

theorem number_of_correct_statements (stmt1: Prop) (stmt2: Prop) (stmt3: Prop) :
  stmt1 ∧ stmt2 ∧ stmt3 → (∀ n, n = 3) :=
by
  sorry

end number_of_correct_statements_l303_303257


namespace find_a2015_l303_303085

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ n ≥ 2, a n + a (n - 1) = 4

theorem find_a2015 (a : ℕ → ℕ) (h : sequence a) : a 2015 = 3 :=
sorry

end find_a2015_l303_303085


namespace part1_part2_part3_l303_303081

-- Definitions of the functions f and g
def f (a : ℝ) (x : ℝ) := x^2 + a * x + 3
def g (a : ℝ) (x : ℝ) := (6 + a) * 2^(x - 1)

-- Part I
theorem part1 (a : ℝ) : f a 1 = f a 3 → a = -4 := by
  sorry

-- Part II
theorem part2 (x1 x2 : ℝ) (hx : x1 < x2) : ∀ x, g (-4) x = 2^x → F x := by
  let F := λ x, (2 / (1 + g (-4) x))
  have h1 : ∀ x, g (-4) x = 2^x := by
    intro x; exact rfl
  rw h1 at *
  show F x1 > F x2
  sorry

-- Part III
theorem part3 (x : ℝ) (a : ℝ) : (a < -4 ∨ a > 4) → ∀ x ∈ Icc (-2 : ℝ) 2, f a x ≥ a → a = -7 := by
  sorry

end part1_part2_part3_l303_303081


namespace product_of_divisors_of_18_l303_303293

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l303_303293


namespace emily_extra_distance_five_days_l303_303827

-- Define the distances
def distance_troy : ℕ := 75
def distance_emily : ℕ := 98

-- Emily's extra walking distance in one-way
def extra_one_way : ℕ := distance_emily - distance_troy

-- Emily's extra walking distance in a round trip
def extra_round_trip : ℕ := extra_one_way * 2

-- The extra distance Emily walks in five days
def extra_five_days : ℕ := extra_round_trip * 5

-- Theorem to be proven
theorem emily_extra_distance_five_days : extra_five_days = 230 := by
  -- Proof will go here
  sorry

end emily_extra_distance_five_days_l303_303827


namespace ellipse_eccentricity_l303_303548

open Complex

-- Define the roots of the polynomial equation.
def roots : List Complex := [2, -1 / 2 + (sqrt 7 : ℂ) / 2 * I, -1 / 2 - (sqrt 7 : ℂ) / 2 * I, -5 / 2 + (1 / 2) * I, -5 / 2 - (1 / 2) * I]

-- Define the points in the complex plane derived from the roots.
def points : List (ℝ × ℝ) := [ (2, 0), (-0.5, (real.sqrt 7) / 2), (-0.5, -(real.sqrt 7) / 2), (-2.5, 0.5), (-2.5, -0.5) ]

-- Define the eccentricity of the ellipse passing through these points.
def eccentricity_of_ellipse : ℝ :=
  let h := 0 in  -- Assume the center is (h, 0)
  let a :=  // semi-major axis length
  let b :=  // semi-minor axis length
  real.sqrt ((a ^ 2 - b ^ 2) / a ^ 2)

theorem ellipse_eccentricity (e : ℝ) :
  ∀ (a b : ℝ) (c : ℝ), e = c / a → c^2 = a^2 - b^2 → list_all points_on_ellipse points a b h →
  e = 1 / real.sqrt 5 :=
sorry

end ellipse_eccentricity_l303_303548


namespace min_students_in_band_l303_303484

theorem min_students_in_band : ∃ n : ℕ, n ≠ 0 ∧ (n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0) ∧ n = 168 := 
by
  use 168
  split
  {
    norm_num
  }
  split
  {
    norm_num
  }
  sorry

end min_students_in_band_l303_303484


namespace route_speeds_l303_303711

theorem route_speeds (x : ℝ) (hx : x > 0) :
  (25 / x) - (21 / (1.4 * x)) = (20 / 60) := by
  sorry

end route_speeds_l303_303711


namespace solution_to_equation_l303_303574

theorem solution_to_equation :
  ∀ (x y : ℝ), x ≥ 2 → y ≥ 1 → 
  (36 * real.sqrt (x - 2) + 4 * real.sqrt (y - 1) = 28 - 4 * real.sqrt (x - 2) - real.sqrt (y - 1))
  → x = 5 ∧ y = 3 :=
by
  intros x y x_ge_2 y_ge_1 equation
  sorry

end solution_to_equation_l303_303574


namespace limit_root_n_seq_eq_limit_ratio_seq_l303_303211

noncomputable def seq_limit (a : ℕ → ℝ) (l : ℝ) : Prop :=
∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l| < ε

theorem limit_root_n_seq_eq_limit_ratio_seq {a : ℕ → ℝ} {l : ℝ} 
  (h : seq_limit (λ n, a (n + 1) / a n) l) : 
  seq_limit (λ n, Real.sqrt (a n)) l :=
sorry

end limit_root_n_seq_eq_limit_ratio_seq_l303_303211


namespace product_of_divisors_18_l303_303306

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l303_303306


namespace power_modulo_l303_303437

theorem power_modulo (k : ℕ) : 7^32 % 19 = 1 → 7^2050 % 19 = 11 :=
by {
  sorry
}

end power_modulo_l303_303437


namespace area_of_rectangle_PQRS_l303_303480

variables (PQ PS XZ : ℝ)
variables (h1 : PS = 3 * PQ)
variables (h2 : XZ = 15)
variables (h3 : altitude Y XZ = 9)

theorem area_of_rectangle_PQRS :
  area_of_rectangle PQ PS :=
begin
  sorry
end

end area_of_rectangle_PQRS_l303_303480


namespace product_of_divisors_of_18_l303_303416

theorem product_of_divisors_of_18 : 
  let divisors := [1, 2, 3, 6, 9, 18] in divisors.prod = 5832 := 
by
  let divisors := [1, 2, 3, 6, 9, 18]
  have h : divisors.prod = 18^3 := sorry
  have h_calc : 18^3 = 5832 := by norm_num
  exact Eq.trans h h_calc

end product_of_divisors_of_18_l303_303416


namespace units_digit_of_7_pow_y_plus_6_is_9_l303_303841

theorem units_digit_of_7_pow_y_plus_6_is_9 (y : ℕ) (hy : 0 < y) : 
  (7^y + 6) % 10 = 9 ↔ ∃ k : ℕ, y = 4 * k + 3 := by
  sorry

end units_digit_of_7_pow_y_plus_6_is_9_l303_303841


namespace ob_le_oa1_div4_l303_303981

-- Defining the conditions
def intersect_at_single_point (m₁ m₂ m₃ m₄ : Line) (O : Point) : Prop :=
  ∀ A ∈ {m₁, m₂, m₃, m₄}, O ∈ A

def parallel_lines (l₁ l₂ : Line) : Prop :=
  ∀ (P Q : Point), P ∈ l₁ → Q ∈ l₁ → (λ P Q, P.x * Q.y - Q.x * P.y = 0) P Q

variables {m₁ m₂ m₃ m₄ : Line}
variables {O A₁ A₂ A₃ A₄ B : Point}

theorem ob_le_oa1_div4
  (h₀ : intersect_at_single_point m₁ m₂ m₃ m₄ O)
  (h₁ : A₁ ∈ m₁)
  (h₂ : parallel_lines (line_through (mk_line A₁ A₂)) m₄)
  (h₃ : A₂ ∈ m₂)
  (h₄ : parallel_lines (line_through (mk_line A₂ A₃)) m₁)
  (h₅ : A₃ ∈ m₃)
  (h₆ : parallel_lines (line_through (mk_line A₃ A₄)) m₂)
  (h₇ : A₄ ∈ m₄)
  (h₈ : parallel_lines (line_through (mk_line A₄ B)) m₃) :
  dist O B ≤ 1 / 4 * dist O A₁ := sorry

end ob_le_oa1_div4_l303_303981


namespace integer_a_values_l303_303942

theorem integer_a_values (a : ℤ) :
  (∃ x : ℤ, x^3 + 3 * x^2 + a * x - 7 = 0) ↔ a = -70 ∨ a = -29 ∨ a = -5 ∨ a = 3 :=
by
  sorry

end integer_a_values_l303_303942


namespace correct_regression_eq_l303_303260

-- Definitions related to the conditions
def negative_correlation (y x : ℝ) : Prop :=
  -- y is negatively correlated with x implies a negative slope in regression
  ∃ a b : ℝ, a < 0 ∧ ∀ x, y = a * x + b

-- The potential regression equations
def regression_eq1 (x : ℝ) : ℝ := -10 * x + 200
def regression_eq2 (x : ℝ) : ℝ := 10 * x + 200
def regression_eq3 (x : ℝ) : ℝ := -10 * x - 200
def regression_eq4 (x : ℝ) : ℝ := 10 * x - 200

-- Prove that the correct regression equation is selected given the conditions
theorem correct_regression_eq (y x : ℝ) (h : negative_correlation y x) : 
  (∀ x : ℝ, y = regression_eq1 x) ∨ (∀ x : ℝ, y = regression_eq2 x) ∨ 
  (∀ x : ℝ, y = regression_eq3 x) ∨ (∀ x : ℝ, y = regression_eq4 x) →
  ∀ x : ℝ, y = regression_eq1 x := by
  -- This theorem states that given negative correlation and the possible options, 
  -- the correct regression equation consistent with all conditions must be regression_eq1.
  sorry

end correct_regression_eq_l303_303260


namespace total_weight_of_plastic_rings_l303_303128

def weight_orange := 0.08333333333333333
def weight_purple := 0.3333333333333333
def weight_white := 0.4166666666666667
def weight_blue := 0.5416666666666666
def weight_red := 0.625
def conversion_factor := 28.35

theorem total_weight_of_plastic_rings :
  let total_weight_in_ounces := weight_orange + weight_purple + weight_white + weight_blue + weight_red in
  let total_weight_in_grams := total_weight_in_ounces * conversion_factor in
  total_weight_in_grams = 56.7 :=
by
  sorry

end total_weight_of_plastic_rings_l303_303128


namespace product_of_all_positive_divisors_of_18_l303_303386

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l303_303386


namespace find_a_l303_303119

theorem find_a
  (x y a : ℝ)
  (h1 : x + y = 1)
  (h2 : 2 * x + y = 0)
  (h3 : a * x - 3 * y = 0) :
  a = -6 :=
sorry

end find_a_l303_303119


namespace product_of_divisors_of_18_l303_303334

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l303_303334


namespace smallest_positive_period_max_min_values_l303_303625

noncomputable def f (x a : ℝ) : ℝ :=
  (Real.cos x) * (2 * Real.sqrt 3 * Real.sin x - Real.cos x) + a * Real.sin x ^ 2

theorem smallest_positive_period (a : ℝ) (h : f (Real.pi / 12) a = 0) : 
  ∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) a = f x a) ∧ (∀ ε > 0, ε < T → ∃ y, y < T ∧ f y a ≠ f 0 a) := 
sorry

theorem max_min_values (a : ℝ) (h : f (Real.pi / 12) a = 0) :
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x a ≤ Real.sqrt 3) ∧ 
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), -2 ≤ f x a) := 
sorry

end smallest_positive_period_max_min_values_l303_303625


namespace correct_statements_l303_303643

def f (x : ℝ) : ℝ := x * Real.exp x

def f_n (n : ℕ) : (ℝ → ℝ) :=
  if n = 0 then
    λ x, (f x)'
  else
    let rec aux : ℕ → (ℝ → ℝ) := λ m, if m = 0 then (λ x, (f x)') else (λ x, ((aux (m - 1))') )
    aux n

theorem correct_statements (x1 x2 : ℝ) (h : x2 > x1) :
  (f 1 = 0 ∧ (∀ x, f_n 2016 x = x * Real.exp x + 2017 * Real.exp x)) :=
by
  sorry

end correct_statements_l303_303643


namespace calculate_expression_l303_303907

theorem calculate_expression : (Real.sqrt 3)^0 + 2^(-1) + Real.sqrt 2 * Real.cos (Float.pi / 4) - abs (-1 / 2) = 2 := 
by
  sorry

end calculate_expression_l303_303907


namespace line_circle_intersection_l303_303798

theorem line_circle_intersection (a : ℝ) : 
  (∀ x y : ℝ, (4 * x + 3 * y + a = 0) → ((x - 1)^2 + (y - 2)^2 = 9)) ∧
  (∃ A B : ℝ, dist A B = 4 * Real.sqrt 2) →
  (a = -5 ∨ a = -15) :=
by 
  sorry

end line_circle_intersection_l303_303798


namespace binom_sum_mod_1000_l303_303520

theorem binom_sum_mod_1000 : 
  (∑ i in (finset.range 2012).filter (λ i, i % 4 = 0), nat.choose 2011 i) % 1000 = 15 :=
sorry

end binom_sum_mod_1000_l303_303520


namespace find_k_for_given_prime_l303_303180

theorem find_k_for_given_prime (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (k : ℕ) 
  (h : ∃ a : ℕ, k^2 - p * k = a^2) : 
  k = (p + 1)^2 / 4 :=
sorry

end find_k_for_given_prime_l303_303180


namespace sodium_acetate_formed_is_3_l303_303027

-- Definitions for chemicals involved in the reaction
def AceticAcid : Type := ℕ -- Number of moles of acetic acid
def SodiumHydroxide : Type := ℕ -- Number of moles of sodium hydroxide
def SodiumAcetate : Type := ℕ -- Number of moles of sodium acetate

-- Given conditions as definitions
def reaction (acetic_acid naoh : ℕ) : ℕ :=
  if acetic_acid = naoh then acetic_acid else min acetic_acid naoh

-- Lean theorem statement
theorem sodium_acetate_formed_is_3 
  (acetic_acid naoh : ℕ) 
  (h1 : acetic_acid = 3) 
  (h2 : naoh = 3) :
  reaction acetic_acid naoh = 3 :=
by
  -- Proof body (to be completed)
  sorry

end sodium_acetate_formed_is_3_l303_303027


namespace ball_box_count_l303_303810

noncomputable def f (n k : ℕ) : ℕ :=
  if k = 0 then 1
  else if k = n then Nat.factorial n
  else (2 * n - k) * f (n - 1) (k - 1) + f (n - 1) k

theorem ball_box_count (n k : ℕ) (h1 : k ≥ 0) (h2 : k ≤ n) : f n k = (Nat.choose n k) ^ 2 * Nat.factorial k :=
  sorry

end ball_box_count_l303_303810


namespace smallest_n_l303_303772

theorem smallest_n (n : ℕ) : 
  (n % 6 = 4) ∧ (n % 7 = 2) ∧ (n > 20) → n = 58 :=
by
  sorry

end smallest_n_l303_303772


namespace eval_expr_l303_303503

theorem eval_expr : (900 ^ 2) / (262 ^ 2 - 258 ^ 2) = 389.4 := 
by
  sorry

end eval_expr_l303_303503


namespace problem_l303_303167

noncomputable def M (x y z : ℝ) : ℝ :=
  (Real.sqrt (x^2 + x * y + y^2) * Real.sqrt (y^2 + y * z + z^2)) +
  (Real.sqrt (y^2 + y * z + z^2) * Real.sqrt (z^2 + z * x + x^2)) +
  (Real.sqrt (z^2 + z * x + x^2) * Real.sqrt (x^2 + x * y + y^2))

theorem problem (x y z : ℝ) (α β : ℝ) 
  (h1 : ∀ x y z, α * (x * y + y * z + z * x) ≤ M x y z)
  (h2 : ∀ x y z, M x y z ≤ β * (x^2 + y^2 + z^2)) :
  (∀ α, α ≤ 3) ∧ (∀ β, β ≥ 3) :=
sorry

end problem_l303_303167


namespace f_2018_eq_0_l303_303639

noncomputable def f : ℝ → ℝ := λ x, 
  if 0 ≤ x ∧ x ≤ 1 then 2 * (1 - x) 
  else if 1 < x ∧ x ≤ 2 then x - 1
  else 0 

def f_n (n : ℕ) (x : ℝ) : ℝ := nat.rec_on n x (λ n r, f r)

theorem f_2018_eq_0 : f_n 2018 2 = 0 :=
sorry

end f_2018_eq_0_l303_303639


namespace dihedral_angle_cosine_l303_303836

noncomputable def cos_dihedral_angle : ℚ := 1 / 3

theorem dihedral_angle_cosine (R r : ℝ) (θ : ℝ) : 
  R = 3 * r → 
  ∃ edge_angle : ℝ, edge_angle = π / 3 ∧ 
  cos θ = cos_dihedral_angle → 
  cos θ = 1 / 3 :=
by
  intros h1 h2.
  sorry

end dihedral_angle_cosine_l303_303836


namespace similar_extended_altitudes_l303_303164

variable {α : Type*} [Real α]

structure Triangle (α : Type*) [Real α] :=
  (A B C : α)

noncomputable def similar_triangles
  (T₁ T₂ : Triangle α) : Prop :=
  ∃(k : α), (T₂.A = T₁.A) ∧
            (T₂.B = T₁.B) ∧
            (T₂.C = T₁.C) ∧
            (T₂.B - T₂.A = k * (T₁.B - T₁.A)) ∧
            (T₂.C - T₂.A = k * (T₁.C - T₁.A)) ∧
            (T₂.C - T₂.B = k * (T₁.C - T₁.B))

theorem similar_extended_altitudes
  {T : Triangle α}
  (k k' : α)
  (h₁ : k > 0)
  (h₂ : k' > 0)
  (h3 : T.A' = T.A + k * i * (T.C - T.B))
  (h4 : T.B' = T.B + k' * i * (T.A - T.C))
  (h5 : T.C' = T.C + k * i * (T.B - T.A)) :
  similar_triangles T ⟨ T.A', T.B', T.C' ⟩ → k = 1 ∧ k' = 1 :=
  by sorry

end similar_extended_altitudes_l303_303164


namespace delta_ne_zero_for_all_k_and_n_l303_303033

def delta1 (u : ℕ → ℤ) (n : ℕ) : ℤ :=
  u (n + 1) - u n

def delta (u : ℕ → ℤ) : ℕ → ℕ → ℤ
| 1, n => delta1 u n
| k + 1, n => delta1 (delta u k) n

def u (n : ℕ) : ℤ := n^4 + n^2

theorem delta_ne_zero_for_all_k_and_n : ∀ (k : ℕ) (n : ℕ), delta u k n ≠ 0 :=
by
  assume k n
  have h : u_n := u n
  sorry

end delta_ne_zero_for_all_k_and_n_l303_303033


namespace product_of_divisors_of_18_l303_303360

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303360


namespace sum_expression_eq_2021_l303_303169

-- Define the polynomial with given roots
noncomputable def poly : Polynomial ℝ := 
  Polynomial.monomial 2020 1 + Polynomial.monomial 2019 1
  + ∑ i in Finset.range 2018, Polynomial.monomial i 1 - 2020

-- Define the problem conditions
def roots := (Polynomial.roots poly).toFinset

-- Define the expression to sum
noncomputable def expression : ℝ := 
  ∑ a in roots, (1 / (1 - a))

-- Prove the main statement
theorem sum_expression_eq_2021 : expression = 2021 :=
by
  sorry

end sum_expression_eq_2021_l303_303169


namespace three_collinear_points_same_color_l303_303196

theorem three_collinear_points_same_color
    (points : Type)
    [linear_ordered_field points]
    (color : points → bool) :
  (∀ x y : points, x ≠ y → color x ≠ color y)
  → ∃ (a b c : points), color a = color b ∧ color b = color c ∧ 2 * b = a + c :=
sorry

end three_collinear_points_same_color_l303_303196


namespace problem_l303_303792

def f : ℕ → ℕ → ℕ :=
sorry

axiom f_def1 : ∀ (x : ℕ), f x x = x
axiom f_def2 : ∀ (x y : ℕ), f x y = f y x
axiom f_def3 : ∀ (x y : ℕ), (x + y) * f x y = y * f x (x + y)
axiom f_def4 : ∀ (x : ℕ), f x (2 * x) = 3 * x

theorem problem (h : f_def3 16 64) : f 16 64 = 64 :=
sorry

end problem_l303_303792


namespace sqrt_lt_implies_lt_l303_303845

theorem sqrt_lt_implies_lt 
    (a b : ℝ) 
    (h : sqrt a < sqrt b) 
    : a < b := 
    sorry

end sqrt_lt_implies_lt_l303_303845


namespace problem_statement_l303_303504

def S : ℤ := (-2^2 - 2^3 - 2^4 - 2^5 - 2^6 - 2^7 - 2^8 - 2^9 - 2^10 - 2^11 - 2^12 - 2^13 - 2^14 - 2^15 - 2^16 - 2^17 - 2^18 - 2^19)

theorem problem_statement (hS : S = -2^20 + 4) : 2 - 2^2 - 2^3 - 2^4 - 2^5 - 2^6 - 2^7 - 2^8 - 2^9 - 2^10 - 2^11 - 2^12 - 2^13 - 2^14 - 2^15 - 2^16 - 2^17 - 2^18 - 2^19 + 2^20 = 6 :=
by
  sorry

end problem_statement_l303_303504


namespace domain_of_f_range_of_f_interval_of_increase_of_f_l303_303637

noncomputable def f (x : ℝ) : ℝ := Real.log (4 - x^2) / Real.log 2

theorem domain_of_f :
  (∀ x : ℝ, f x ∈ ℝ) ↔ -2 < x ∧ x < 2 ∧ 4 - x^2 > 0 :=
sorry

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≤ 2 :=
sorry

theorem interval_of_increase_of_f :
  ∀ x1 x2 : ℝ, (x1 < x2 ∧ -2 < x1 ∧ x1 < 0 ∧ -2 < x2 ∧ x2 < 0) → f x1 < f x2 :=
sorry

end domain_of_f_range_of_f_interval_of_increase_of_f_l303_303637


namespace red_ball_probability_l303_303269

noncomputable def Urn1_blue : ℕ := 5
noncomputable def Urn1_red : ℕ := 3
noncomputable def Urn2_blue : ℕ := 4
noncomputable def Urn2_red : ℕ := 4
noncomputable def Urn3_blue : ℕ := 8
noncomputable def Urn3_red : ℕ := 0

noncomputable def P_urn (n : ℕ) : ℝ := 1 / 3
noncomputable def P_red_urn1 : ℝ := (Urn1_red : ℝ) / (Urn1_blue + Urn1_red)
noncomputable def P_red_urn2 : ℝ := (Urn2_red : ℝ) / (Urn2_blue + Urn2_red)
noncomputable def P_red_urn3 : ℝ := (Urn3_red : ℝ) / (Urn3_blue + Urn3_red)

theorem red_ball_probability : 
  (P_urn 1 * P_red_urn1 + P_urn 2 * P_red_urn2 + P_urn 3 * P_red_urn3) = 7 / 24 :=
  by sorry

end red_ball_probability_l303_303269


namespace which_is_linear_in_two_vars_l303_303846

theorem which_is_linear_in_two_vars (x y : ℝ) :
  let A := x + x * y = 8
  let B := y - x = 1
  let C := x + 1/x = 2
  let D := x^2 - 2 * x + 1 = 0
  B := y - x = 1 :=
  -- To be proved
  sorry

end which_is_linear_in_two_vars_l303_303846


namespace product_of_divisors_of_18_l303_303384

theorem product_of_divisors_of_18 : 
  ∏ d in (finset.filter (λ d, 18 % d = 0) (finset.range 19)), d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303384


namespace product_of_divisors_of_18_l303_303433

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303433


namespace positive_solution_sqrt_eq_l303_303579

theorem positive_solution_sqrt_eq (x : ℝ) (hx : 0 < x) :
  (sqrt (x + sqrt (x + sqrt (x + ...))) = sqrt (x * sqrt (x^2 * sqrt (x ...)))) ↔ x = 2 :=
by
  sorry

end positive_solution_sqrt_eq_l303_303579


namespace negation_proposition_l303_303931

theorem negation_proposition (x : ℝ) (hx : 0 < x) : x + 4 / x ≥ 4 :=
sorry

end negation_proposition_l303_303931


namespace angle_E_measure_l303_303722

theorem angle_E_measure {D E F : Type} (angle_D angle_E angle_F : ℝ) 
  (h1 : angle_E = angle_F)
  (h2 : angle_F = 3 * angle_D)
  (h3 : angle_D = (1/2) * angle_E) 
  (h_sum : angle_D + angle_E + angle_F = 180) :
  angle_E = 540 / 7 := 
by
  sorry

end angle_E_measure_l303_303722


namespace perpendicular_planes_l303_303740

variables (m n : Line) (α β : Plane)

-- Conditions
axiom h1 : m ≠ n
axiom h2 : α ≠ β
axiom h3 : m ⊆ α
axiom h4 : n ⊆ β

-- Definition of perpendicular lines and planes
axiom perp_line : Line → Line → Prop -- m ⊥ n
axiom perp_plane : Plane → Plane → Prop -- α ⊥ β

-- Definition of a line lying in a plane
axiom line_in_plane : Line → Plane → Prop -- m ⊆ α

-- The statement to prove
theorem perpendicular_planes (h5 : perp_line n α) : perp_plane α β :=
by sorry

end perpendicular_planes_l303_303740


namespace inverse_of_B_squared_l303_303063

theorem inverse_of_B_squared (B_inv : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B_inv = ![![3, -2], ![0, 5]]) : 
  (B_inv * B_inv) = ![![9, -16], ![0, 25]] :=
by
  sorry

end inverse_of_B_squared_l303_303063


namespace value_of_7_star_3_l303_303674

def star (a b : ℕ) : ℕ := 4 * a + 3 * b - a * b

theorem value_of_7_star_3 : star 7 3 = 16 :=
by
  -- Proof would go here
  sorry

end value_of_7_star_3_l303_303674


namespace product_of_divisors_of_18_is_5832_l303_303405

theorem product_of_divisors_of_18_is_5832 :
  ∏ d in (finset.filter (λ d : ℕ, 18 % d = 0) (finset.range 19)), d = 5832 :=
sorry

end product_of_divisors_of_18_is_5832_l303_303405


namespace product_of_divisors_of_18_l303_303429

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303429


namespace find_number_l303_303028

theorem find_number (x : ℕ) (h : 15 * x = x + 196) : 15 * x = 210 :=
by
  sorry

end find_number_l303_303028


namespace volume_of_second_cube_l303_303443

theorem volume_of_second_cube (V1 : ℝ) (condition1 : V1 = 8) 
  (A2 : ℝ) (condition2 : A2 = 3 * 6 * (2 : ℝ) ^ 2) :
  ∃ V2 : ℝ, V2 = 24 * real.sqrt 3 :=
by
  use (24 * real.sqrt 3)
  sorry

end volume_of_second_cube_l303_303443


namespace point_B_in_fourth_quadrant_l303_303122

theorem point_B_in_fourth_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : (b > 0 ∧ a < 0) :=
by {
    sorry
}

end point_B_in_fourth_quadrant_l303_303122


namespace find_m_n_l303_303573

theorem find_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (hmn : m^n = n^(m - n)) : 
  (m = 9 ∧ n = 3) ∨ (m = 8 ∧ n = 2) :=
sorry

end find_m_n_l303_303573


namespace distance_swim_back_is_12_l303_303476

-- Define the conditions
def swimming_speed_still_water : ℝ := 4 -- km/h
def current_speed : ℝ := 2 -- km/h
def swimming_time : ℝ := 6 -- hours

-- Define the effective swimming speed against the current
def effective_speed := swimming_speed_still_water - current_speed

-- Define the distance swam back against the current
def distance_swim_back := effective_speed * swimming_time

-- Prove that the man swims back a distance of 12 kilometers
theorem distance_swim_back_is_12 : distance_swim_back = 12 := by
  simp [distance_swim_back, effective_speed, swimming_speed_still_water, current_speed, swimming_time]
  simp
  sorry

end distance_swim_back_is_12_l303_303476


namespace product_of_all_positive_divisors_of_18_l303_303393

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l303_303393


namespace domain_of_function_l303_303008

theorem domain_of_function {
  x : ℝ
} : 
  (x + 1 > 0) → (-x^2 - 3x + 4 ≥ 0) → x ∈ Ioo (-1) 1 :=
sorry

end domain_of_function_l303_303008


namespace florist_bouquets_l303_303880

def initial_roses : ℕ := 17
def quarter_rounded_down (n : ℕ) : ℕ := n / 4
def picked_roses : ℕ := 48
def bouquet_size : ℕ := 5

-- Function to compute the number of remaining roses after selling a quarter
def remaining_roses (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

-- The final number of roses after picking additional ones
def total_roses (remaining : ℕ) (picked : ℕ) : ℕ := remaining + picked

-- Function to compute the number of bouquets that can be made
def bouquets (total : ℕ) (size : ℕ) : ℕ := total / size

-- Theorem stating that the total number of bouquets is 12 given the conditions
theorem florist_bouquets : 
  let sold := quarter_rounded_down initial_roses in
  let remaining := remaining_roses initial_roses sold in
  let total := total_roses remaining picked_roses in
  bouquets total bouquet_size = 12 :=
by 
  sorry

end florist_bouquets_l303_303880


namespace steve_initial_boxes_l303_303236

theorem steve_initial_boxes :
  ∀ (total_pencils_per_box pencils_given_to_Lauren pencils_left: ℕ),
    total_pencils_per_box = 12 → 
    pencils_given_to_Lauren = 6 → 
    pencils_left = 9 →
    (pencils_given_to_Lauren + (pencils_given_to_Lauren + 3) + pencils_left) / total_pencils_per_box = 2 :=
by
  intros total_pencils_per_box pencils_given_to_Lauren pencils_left h1 h2 h3
  rw [h1, h2, h3]
  sorry

end steve_initial_boxes_l303_303236


namespace avg_children_in_fam_with_kids_l303_303190

-- Definitions and conditions
def total_families : Nat := 9
def average_children_per_family : ℕ → ℕ → ℕ := λ total_kids num_families, total_kids / num_families
def total_kids_per_family (avg_kids : Nat) (num_families : Nat) : Nat := avg_kids * num_families
def childless_families : ℕ := 3
def families_with_children (total_fam : ℕ) (childless_fam : ℕ) : ℕ := total_fam - childless_fam

-- Main theorem
theorem avg_children_in_fam_with_kids :
  average_children_per_family
    (total_kids_per_family 3 total_families)
    (families_with_children total_families childless_families) = 4.5 := by
  sorry

end avg_children_in_fam_with_kids_l303_303190


namespace lucas_avocado_purchase_l303_303752

theorem lucas_avocado_purchase (initial_money spent_change avocado_cost : ℕ) (h1 : initial_money = 20) (h2 : spent_change = 14) (h3 : avocado_cost = 2) : 
  (initial_money - spent_change) / avocado_cost = 3 :=
by
  -- The definitions and conditions are used here
  rw [h1, h2, h3]
  sorry

end lucas_avocado_purchase_l303_303752


namespace binomial_coefficient_sum_mod_l303_303529

theorem binomial_coefficient_sum_mod : 
  let S := ((1 + Complex.exp (Complex.I * Real.pi / 2))^2011) + 
           ((1 + Complex.exp (3 * Complex.I * Real.pi / 2))^2011) + 
           ((1 + -1)^2011) + 
           ((1 + 1)^2011)
  in 
  let desired_sum := (range 503).sum (λ j, Nat.choose 2011 (4 * j)) / 4
  in 
  (S % 1000 = 137) :
  nat.Mod 1000 S = 137 := 
begin
  sorry
end

end binomial_coefficient_sum_mod_l303_303529


namespace friendship_distribution_impossible_l303_303130

theorem friendship_distribution_impossible :
  ¬ ∃ (students : Finset ℕ) 
      (friends_per_student : ℕ → ℕ), 
      students.card = 25 ∧
      (∃ (s₁ s₂ s₃ : Finset ℕ),
        s₁.card = 6 ∧ s₂.card = 10 ∧ s₃.card = 9 ∧
        (∀ x ∈ s₁, friends_per_student x = 3) ∧
        (∀ x ∈ s₂, friends_per_student x = 4) ∧
        (∀ x ∈ s₃, friends_per_student x = 5) ∧
        s₁ ∪ s₂ ∪ s₃ = students) ∧ 
      ∑ x in students, friends_per_student x = 103 :=
by
  sorry

end friendship_distribution_impossible_l303_303130


namespace product_of_divisors_of_18_l303_303313

theorem product_of_divisors_of_18 : (finset.prod (finset.filter (λ n, 18 % n = 0) (finset.range 19)) id) = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303313


namespace length_of_AQ_l303_303506

theorem length_of_AQ (P: Type) (A Q: P) (r: ℝ) 
  (h1: (2 * real.pi * r = 18 * real.pi)) 
  (h2: ∠(A, Q) = 60) 
  (h3: P.dist A P = r ∧ P.dist P Q = r) 
  : P.dist A Q = 9 :=
by sorry

end length_of_AQ_l303_303506


namespace product_of_all_positive_divisors_of_18_l303_303395

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l303_303395


namespace isosceles_triangle_perimeter_l303_303980

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 4) (h2 : b = 8) (h3 : ∃ p q r, p = b ∧ q = b ∧ r = a ∧ p + q > r) : 
  a + b + b = 20 := 
by 
  sorry

end isosceles_triangle_perimeter_l303_303980


namespace seq_increasing_l303_303102

theorem seq_increasing (n : ℕ) (h : n > 0) : (↑n / (↑n + 2): ℝ) < (↑n + 1) / (↑n + 3) :=
by 
-- Converting ℕ to ℝ to make definitions correct
let an := (↑n / (↑n + 2): ℝ)
let an1 := (↑n + 1) / (↑n + 3)
-- Proof would go here
sorry

end seq_increasing_l303_303102


namespace find_c_plus_d_l303_303111

theorem find_c_plus_d (x c d : ℝ) (h_pos_c : c > 0) (h_pos_d : d > 0) 
  (h_eq : x = c + real.sqrt d)
  (h_x : x^2 + 4 * x + 4 / x + 1 / x^2 = 34) :
  c + d = 11 :=
sorry

end find_c_plus_d_l303_303111


namespace prob_remainder_mod_1000_l303_303525

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ 
| n 0 := 1
| 0 k := 0
| n k := binom (n-1) (k-1) * n / k

-- Define the sum we are interested in, only including indices that are multiples of 4
def sum_binom_2011_multiple_4 : ℕ :=
  (Finset.range (2012 / 4 + 1)).sum (λ i, binom 2011 (4 * i))

-- The statement we want to prove
theorem prob_remainder_mod_1000 : 
  sum_binom_2011_multiple_4 % 1000 = 12 := 
sorry

end prob_remainder_mod_1000_l303_303525


namespace problem1_l303_303974

noncomputable def a_seq : ℕ → ℝ
| 0     := 3
| (n+1) := (3 * a_seq n - 1) / (a_seq n + 1)

def one_over_a_minus_one_seq (n : ℕ) : ℝ :=
1 / (a_seq n - 1)

def is_arithmetic (seq : ℕ → ℝ) (d : ℝ) (a0 : ℝ) : Prop :=
∀ n, seq (n+1) = seq n + d ∧ seq 0 = a0

def b_seq : ℕ → ℝ
| 0     := a_seq 0
| (n+1) := b_seq n * a_seq (n+1)

def S_n (n : ℕ) : ℝ :=
∑ i in finset.range (n+1), 1 / b_seq i

theorem problem1 :
  is_arithmetic one_over_a_minus_one_seq (1/2) (1/2) ∧
  (∀ n, a_seq n = (n + 2) / n) ∧
  (∀ n, S_n n = n / (n + 2)) :=
sorry

end problem1_l303_303974


namespace keith_turnips_l303_303728

theorem keith_turnips (a t k : ℕ) (h1 : a = 9) (h2 : t = 15) : k = t - a := by
  sorry

end keith_turnips_l303_303728


namespace complement_intersection_subset_condition_l303_303061

-- Definition of sets A, B, and C
def A := { x : ℝ | 3 ≤ x ∧ x < 7 }
def B := { x : ℝ | 2 < x ∧ x < 10 }
def C (a : ℝ) := { x : ℝ | x < a }

-- Proof problem 1 statement
theorem complement_intersection :
  ( { x : ℝ | x < 3 ∨ x ≥ 7 } ∩ { x : ℝ | 2 < x ∧ x < 10 } ) = { x : ℝ | 2 < x ∧ x < 3 ∨ 7 ≤ x ∧ x < 10 } :=
by
  sorry

-- Proof problem 2 statement
theorem subset_condition (a : ℝ) :
  ( { x : ℝ | 3 ≤ x ∧ x < 7 } ⊆ { x : ℝ | x < a } ) → (a ≥ 7) :=
by
  sorry

end complement_intersection_subset_condition_l303_303061


namespace product_of_all_positive_divisors_of_18_l303_303392

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l303_303392


namespace relationship_abc_l303_303991

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ → (x₂ * f x₁ - x₁ * f x₂) / (x₁ - x₂) > 0
  
noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f x / x

-- a, b, and c definitions
def a (f : ℝ → ℝ) : ℝ :=
  3 * f (1 / 3)

def b (f : ℝ → ℝ) : ℝ :=
  -5 / 2 * f (-2 / 5)

def c (f : ℝ → ℝ) : ℝ :=
  f 1

theorem relationship_abc (f : ℝ → ℝ) 
  (hf_odd : odd_function f)
  (hf_incr_neg : increasing_on_neg f) :
  a f > b f ∧ b f > c f := 
by 
  sorry

end relationship_abc_l303_303991


namespace line_intersects_circle_at_two_points_trajectory_midpoint_l303_303958

variable (m : ℝ)
def circle_eq := ∀ x y : ℝ, x^2 + (y - 1)^2 = 5
def line_eq := ∀ x y : ℝ, m * x - y + 1 - m = 0

theorem line_intersects_circle_at_two_points (m : ℝ) : 
  ∃ x1 y1 x2 y2 : ℝ, circle_eq x1 y1 ∧ line_eq x1 y1 ∧ circle_eq x2 y2 ∧ line_eq x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2) := by sorry

theorem trajectory_midpoint : ∀ M : ℝ × ℝ, 
  (∃ x1 y1 x2 y2 : ℝ, circle_eq x1 y1 ∧ circle_eq x2 y2 ∧ 
    line_eq x1 y1 ∧ line_eq x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2) ∧ 
    M = ((x1 + x2) / 2, (y1 + y2) / 2)) → 
  (M.1 - 1 / 2)^2 + (M.2 - 1)^2 = 1 / 4 := by sorry

end line_intersects_circle_at_two_points_trajectory_midpoint_l303_303958


namespace original_ratio_l303_303834

theorem original_ratio (x y : ℤ) (h1 : y = 24) (h2 : (x + 6) / y = 1 / 2) : x / y = 1 / 4 := by
  sorry

end original_ratio_l303_303834


namespace largest_among_a_b_c_d_is_c_l303_303735

noncomputable def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋ 

def a : ℝ := fractional_part (⌊Real.pi⌋^2)
def b : ℝ := ⌊fractional_part Real.pi^2⌋
def c : ℝ := ⌊⌊Real.pi⌋^2⌋
def d : ℝ := fractional_part (fractional_part Real.pi^2)

theorem largest_among_a_b_c_d_is_c : max (max a b) (max c d) = c := by
  sorry

end largest_among_a_b_c_d_is_c_l303_303735


namespace part1_part2_l303_303084

variables (q x : ℝ)
def f (x : ℝ) (q : ℝ) : ℝ := x^2 - 16*x + q + 3
def g (x : ℝ) (q : ℝ) : ℝ := f x q + 51

theorem part1 (h1 : ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x q = 0):
  (-20 : ℝ) ≤ q ∧ q ≤ 12 := 
  sorry

theorem part2 (h2 : ∀ x ∈ Set.Icc (q : ℝ) 10, g x q ≥ 0) : 
  9 ≤ q ∧ q < 10 := 
  sorry

end part1_part2_l303_303084


namespace equation_of_C_fixed_point_of_AB_l303_303648

noncomputable theory

variable {p k : ℝ}
variable {x_M y_M x_N y_N x_A y_A x_B y_B : ℝ}

-- Conditions for Part 1:
def line_eq1 := ∀ x, y = (1 / 2) * x - 1
def parabola_eq1 := ∀ (p > 0), x^2 = -2 * p * y
def intersection_points1 := (x_M + 1) * (x_N + 1) = -8

-- Correct answer for Part 1:
theorem equation_of_C (h : parabola_eq1 p) (hx : intersection_points1) : parabola_eq1 3 :=
sorry

-- Conditions for Part 2:
def line_eq2 := ∀ x, y = k * x - 3 / 2
def parabola_eq2 := ∀ x, x^2 = -6 * y
def intersection_points2 := x_A = -x_B

-- Correct answer for Part 2:
theorem fixed_point_of_AB (h1 : parabola_eq2 x) (h2 : line_eq2 x) (h3 : intersection_points2) : (0, 3 / 2) ∈ line_eq2 :=
sorry

end equation_of_C_fixed_point_of_AB_l303_303648


namespace product_of_divisors_of_18_l303_303430

theorem product_of_divisors_of_18 : 
  ∏ i in (finset.filter (λ x : ℕ, x ∣ 18) (finset.range (18 + 1))), i = 5832 := 
by 
  sorry

end product_of_divisors_of_18_l303_303430


namespace minimal_d_for_invertibility_l303_303172

-- Define the function g(x)
def g (x : ℝ) : ℝ := (x - 3)^2 + 4

-- State the problem
theorem minimal_d_for_invertibility :
  ∃ d : ℝ, (∀ x, x ∈ set.Ici d → function.injective (g ∘ (λ x, x))) ∧ (∀ y : ℝ, (y < d) → ∃ a b : ℝ, g a = y ∧ g b = y ∧ a ≠ b) ∧ d = 3 :=
  sorry

end minimal_d_for_invertibility_l303_303172


namespace product_of_divisors_of_18_l303_303339

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l303_303339


namespace ellipse_properties_l303_303628

noncomputable def ellipse_focus_eq (a b c : ℝ) (e : ℝ) (h : c / a = e) : Prop :=
  a^2 = b^2 + c^2

theorem ellipse_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (hc : b < a)
  (hf : (1 : ℝ) = 1)
  (he : (1 : ℝ) / (2 : ℝ) = 1 / 2) :
  (k : ℝ) :=
  k = - 1 / 2 :=
begin
  -- Detailed proof can be constructed here
  sorry
end

end ellipse_properties_l303_303628


namespace rightmost_two_digits_of_100A_l303_303468

noncomputable def cube_edge_length : ℝ := 4
noncomputable def rope_length : ℝ := 5
noncomputable def reachable_area : ℝ := 
16 + 4 * (real.pi * 5^2 / 4 - 5 + 2 * real.sqrt (25 - 16) + 2 * real.sqrt (25 - 1))

theorem rightmost_two_digits_of_100A :
  let A := reachable_area in
  let integer_closest_to_100A := real.floor (100 * A + 0.5) in
  (integer_closest_to_100A % 100) = 81 :=
by
  let A := reachable_area
  let integer_closest_to_100A := real.floor (100 * A + 0.5)
  have h : integer_closest_to_100A = 6181 := sorry
  exact ⟨h, rfl⟩

end rightmost_two_digits_of_100A_l303_303468


namespace solution_quad_ineq_l303_303263

noncomputable def quadratic_inequality_solution_set :=
  {x : ℝ | (x > -1) ∧ (x < 3) ∧ (x ≠ 2)}

theorem solution_quad_ineq (x : ℝ) :
  ((x^2 - 2*x - 3)*(x^2 - 4*x + 4) < 0) ↔ x ∈ quadratic_inequality_solution_set :=
by sorry

end solution_quad_ineq_l303_303263


namespace intersection_of_A_and_B_l303_303748

def A : Set ℝ := {x | |x - 2| < 1}
def B : Set ℤ := Set.univ  -- ℤ is defined in Lean as the set of all integers

theorem intersection_of_A_and_B : A ∩ (B : Set ℝ) = {2} :=
by {
  sorry  -- We skip the proof here.
}

end intersection_of_A_and_B_l303_303748


namespace beta_series_converges_l303_303622

noncomputable def alpha_n (n : ℕ) : ℝ := sorry -- to be specified (sequence of positive reals)

noncomputable def beta_n (n : ℕ) : ℝ := (alpha_n n * n) / (n + 1)

def alpha_series_converges : Prop :=
  ∃ l : ℝ, (λ s, ∑ i in Finset.range s, alpha_n i) → l

theorem beta_series_converges
  (hpos : ∀ n, 0 < alpha_n n)
  (h_alpha_converges : alpha_series_converges) :
  ∃ l : ℝ, (λ s, ∑ i in Finset.range s, beta_n i) → l := 
by 
  sorry

end beta_series_converges_l303_303622


namespace inequality_holds_only_for_n_eq_three_l303_303943

theorem inequality_holds_only_for_n_eq_three :
  (n : ℕ) > 0 → (∀ (a : Fin n → ℝ), (∀ i, 0 < a i) →
    (∑ i, (a i)^2) * (∑ i, (a i)) - ∑ i, (a i)^3 ≥ 6 * ∏ i, (a i)) ↔ n = 3 :=
by sorry

end inequality_holds_only_for_n_eq_three_l303_303943


namespace calculate_value_l303_303679

theorem calculate_value :
  let X := (354 * 28) ^ 2
  let Y := (48 * 14) ^ 2
  (X * 9) / (Y * 2) = 2255688 :=
by
  sorry

end calculate_value_l303_303679


namespace evaluate_f_difference_l303_303743

def f (x : ℝ) : ℝ := x^6 + x^4 + 5 * sin x
def g (x : ℝ) : ℝ := x^6 + x^4
def h (x : ℝ) : ℝ := 5 * sin x

theorem evaluate_f_difference : f 3 - f (-3) = 10 * sin 3 :=
by
  have g_even : ∀ x, g x = g (-x) := by
    intro x
    rw [g, pow_six_eq_pow_six_of_neg, pow_four_eq_pow_four_of_neg]
  have h_odd : ∀ x, h (-x) = -h x := by
    intro x
    rw [h, sin_neg]
  rw [f, f]
  -- Following lines would include the necessary calculations
  sorry

end evaluate_f_difference_l303_303743


namespace product_of_divisors_of_18_l303_303282

def n : ℕ := 18

theorem product_of_divisors_of_18 : (∏ d in (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))), d) = 5832 := 
by 
  -- Proof of the theorem will go here
  sorry

end product_of_divisors_of_18_l303_303282


namespace product_of_all_positive_divisors_of_18_l303_303389

def product_divisors_18 : ℕ :=
  ∏ d in (Multiset.to_finset ([1, 2, 3, 6, 9, 18] : Multiset ℕ)), d

theorem product_of_all_positive_divisors_of_18 : product_divisors_18 = 5832 := by
  sorry

end product_of_all_positive_divisors_of_18_l303_303389


namespace product_of_divisors_18_l303_303328

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l303_303328


namespace num_pairs_in_arithmetic_progression_l303_303011

theorem num_pairs_in_arithmetic_progression : 
  { (a, b) : ℝ × ℝ | (a = (8 + b) / 2 ∧ a + a * b = 2 * b) }.to_finset.card = 2 := 
sorry

end num_pairs_in_arithmetic_progression_l303_303011


namespace evaluate_expression_l303_303560

theorem evaluate_expression : (2^4 + 2^5) / (2^(-2) + 2^(-1)) = 64 :=
by
  -- The detailed proof will go here.
  sorry

end evaluate_expression_l303_303560


namespace fishes_zero_l303_303904

def lakes (x y z : ℕ) : ℕ := x + y + z

theorem fishes_zero (x y z : ℕ) (hx : x = 23) (hy : y = 30) (hz : z = 44) (h : lakes x y z = 0) : lakes x y z = 0 :=
by
  rw [hx, hy, hz] at h
  exact h

end fishes_zero_l303_303904


namespace similar_triangles_l303_303976

noncomputable theory

variables {A B C G M X Y Q P : Type} [InnerProductSpace ℝ A]
variables {triangle_ABC : Triangle A B C}
variables {G : Point} (hG : centroid G triangle_ABC)
variables {M : Point} (hM : midpoint M B C)
variables {X Y : Point}
variables (hX : line_through G ∥ line_through B C ∧ intersects X (segment AB))
variables (hY : line_through G ∥ line_through B C ∧ intersects Y (segment AC))
variables {Q : Point} (hQ : line_through X C ∩ line_through G B = Q)
variables {P : Point} (hP : line_through Y B ∩ line_through G C = P)

theorem similar_triangles (hT : triangle_ABC) : similar (Triangle M P Q) (Triangle A B C) :=
sorry

end similar_triangles_l303_303976


namespace graph_is_hyperbola_l303_303007

theorem graph_is_hyperbola : ∀ (x y : ℝ), x^2 - 18 * y^2 - 6 * x + 4 * y + 9 = 0 → ∃ a b c d : ℝ, a * (x - b)^2 - c * (y - d)^2 = 1 :=
by
  -- Proof is omitted
  sorry

end graph_is_hyperbola_l303_303007


namespace kaleb_money_earned_l303_303859

-- Definitions based on the conditions
def total_games : ℕ := 10
def non_working_games : ℕ := 8
def price_per_game : ℕ := 6

-- Calculate the number of working games
def working_games : ℕ := total_games - non_working_games

-- Calculate the total money earned by Kaleb
def money_earned : ℕ := working_games * price_per_game

-- The theorem to prove
theorem kaleb_money_earned : money_earned = 12 := by sorry

end kaleb_money_earned_l303_303859


namespace total_tennis_balls_used_l303_303138

theorem total_tennis_balls_used :
  let rounds := [1028, 514, 257, 128, 64, 32, 16, 8, 4]
  let cans_per_game_A := 6
  let cans_per_game_B := 8
  let balls_per_can_A := 3
  let balls_per_can_B := 4
  let games_A_to_B := rounds.splitAt 4
  let total_A := games_A_to_B.1.sum * cans_per_game_A * balls_per_can_A
  let total_B := games_A_to_B.2.sum * cans_per_game_B * balls_per_can_B
  total_A + total_B = 37573 := 
by
  sorry

end total_tennis_balls_used_l303_303138


namespace math_problem_l303_303205

variable (a b c : ℝ)

theorem math_problem 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  (ab / c + bc / a + ca / b) ≥ Real.sqrt 3 := 
by
  sorry

end math_problem_l303_303205


namespace hilary_regular_toenails_in_jar_l303_303669

-- Conditions
def jar_capacity : Nat := 100
def big_toenail_size : Nat := 2
def num_big_toenails : Nat := 20
def remaining_regular_toenails_space : Nat := 20

-- Question & Answer
theorem hilary_regular_toenails_in_jar : 
  (jar_capacity - remaining_regular_toenails_space - (num_big_toenails * big_toenail_size)) = 40 :=
by
  sorry

end hilary_regular_toenails_in_jar_l303_303669


namespace product_of_divisors_of_18_l303_303362

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303362


namespace second_largest_is_D_l303_303813

noncomputable def A := 3 * 3
noncomputable def C := 4 * A
noncomputable def B := C - 15
noncomputable def D := A + 19

theorem second_largest_is_D : 
    ∀ (A B C D : ℕ), 
      A = 9 → 
      B = 21 →
      C = 36 →
      D = 28 →
      D = 28 :=
by
  intros A B C D hA hB hC hD
  have h1 : A = 9 := by assumption
  have h2 : B = 21 := by assumption
  have h3 : C = 36 := by assumption
  have h4 : D = 28 := by assumption
  exact h4

end second_largest_is_D_l303_303813


namespace product_of_divisors_of_18_l303_303363

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303363


namespace shaded_area_ratio_l303_303042

theorem shaded_area_ratio (side_length : ℝ) (A_square A_dodecagon : ℝ)
  (h_side : side_length = 2) (h_square : A_square = side_length * side_length)
  (h_dodecagon : A_dodecagon = 3) :
  let A_remaining := A_square - A_dodecagon
  let A_shaded := A_remaining / 4
  A_shaded / A_dodecagon = 1 / 12 :=
by
  -- use the given side length condition
  have h1: side_length = 2 := h_side,
  -- use the area calculation of the square
  have h2: A_square = 4 := by rw [h_side, h_square],
  -- use the given area of the dodecagon
  have h3: A_dodecagon = 3 := h_dodecagon,
  -- calculate the remaining area
  let A_remaining := A_square - A_dodecagon,
  have h4: A_remaining = 1 := by linarith,
  -- calculate the shaded area
  let A_shaded := A_remaining / 4,
  have h5: A_shaded = 1 / 4 := by norm_num,
  -- calculate the ratio
  have h6: A_shaded / A_dodecagon = (1 / 4) / 3,
  have h7: (1 / 4) / 3 = 1 / 12 := by norm_num,
  -- the statement we need to prove
  exact h7; sorry

end shaded_area_ratio_l303_303042


namespace problem_l303_303653

theorem problem (a b : ℝ) (h1 : {1, a, b / a} = {0, a^2, a + b}) (h2 : a ≠ 0) : a^2022 + b^2023 = 1 := 
sorry

end problem_l303_303653


namespace polynomial_C_can_be_factored_l303_303450

theorem polynomial_C_can_be_factored : (∃ (a b : ℝ), x^2 - 1 = (x - a) * (x - b)) :=
by {
  use [1, -1],
  sorry
}

end polynomial_C_can_be_factored_l303_303450


namespace cartesian_to_polar_l303_303539

-- Definitions based on the conditions:
def M_cartesian := (sqrt 3, -1 : ℝ)
def polar_coords := (2, 11 * Real.pi / 6 : ℝ)

-- The goal to prove:
theorem cartesian_to_polar (x y ρ θ : ℝ) 
  (hx : x = sqrt 3) 
  (hy : y = -1)
  (hρ : ρ = 2) 
  (hθ : θ = 11 * Real.pi / 6) 
  (hx_eq : x = ρ * Real.cos θ) 
  (hy_eq : y = ρ * Real.sin θ) 
  : (x, y) = (sqrt 3, -1) ∧ (ρ, θ) = (2, 11 * Real.pi / 6) := 
by 
  sorry

end cartesian_to_polar_l303_303539


namespace doughnuts_remaining_l303_303555

theorem doughnuts_remaining 
  (total_doughnuts : ℕ)
  (total_staff : ℕ)
  (staff_3_doughnuts : ℕ)
  (doughnuts_eaten_by_3 : ℕ)
  (staff_2_doughnuts : ℕ)
  (doughnuts_eaten_by_2 : ℕ)
  (staff_4_doughnuts : ℕ)
  (doughnuts_eaten_by_4 : ℕ) :
  total_doughnuts = 120 →
  total_staff = 35 →
  staff_3_doughnuts = 15 →
  staff_2_doughnuts = 10 →
  doughnuts_eaten_by_3 = staff_3_doughnuts * 3 →
  doughnuts_eaten_by_2 = staff_2_doughnuts * 2 →
  staff_4_doughnuts = total_staff - (staff_3_doughnuts + staff_2_doughnuts) →
  doughnuts_eaten_by_4 = staff_4_doughnuts * 4 →
  total_doughnuts - (doughnuts_eaten_by_3 + doughnuts_eaten_by_2 + doughnuts_eaten_by_4) = 15 :=
by
  intros
  -- Proof goes here
  sorry

end doughnuts_remaining_l303_303555


namespace fraction_of_sum_l303_303868

theorem fraction_of_sum (l : List ℝ) (hl : l.length = 51)
  (n : ℝ) (hn : n ∈ l)
  (h : n = 7 * (l.erase n).sum / 50) :
  n / l.sum = 7 / 57 := by
  sorry

end fraction_of_sum_l303_303868


namespace factor_quadratic_l303_303937

theorem factor_quadratic (x : ℝ) : (16 * x^2 - 40 * x + 25) = (4 * x - 5)^2 :=
by 
  sorry

end factor_quadratic_l303_303937


namespace complement_union_l303_303657

open Set

def U : Set ℕ := {x | x < 6}

def A : Set ℕ := {1, 3}

def B : Set ℕ := {3, 5}

theorem complement_union :
  (U \ (A ∪ B)) = {0, 2, 4} :=
by
  sorry

end complement_union_l303_303657


namespace smallest_palindrome_base2_base8_l303_303533

def is_palindrome (s : String) : Bool :=
  s == s.reverse

def to_binary (n : ℕ) : String :=
  if n = 0 then "0" else
  let rec loop (n : ℕ) (acc : String) : String :=
    if n = 0 then acc else loop (n / 2) (String.mk (repr (n % 2).toNat :: acc.toList))
  loop n ""

def to_octal (n : ℕ) : String :=
  if n = 0 then "0" else
  let rec loop (n : ℕ) (acc : String) : String :=
    if n = 0 then acc else loop (n / 8) (String.mk (repr (n % 8).toNat :: acc.toList))
  loop n ""

theorem smallest_palindrome_base2_base8 (m : ℕ) (h1 : 8 < m)
  (h2 : is_palindrome (to_binary m) = true)
  (h3 : is_palindrome (to_octal m) = true) : m = 63 := by
  sorry

end smallest_palindrome_base2_base8_l303_303533


namespace f_x_plus_two_l303_303676

def f (x : ℝ) : ℝ := x * (x - 1) / 2

theorem f_x_plus_two (x : ℝ) : f (x + 2) = (x + 2) * f (x + 1) / x :=
by
  sorry

end f_x_plus_two_l303_303676


namespace number_of_possible_passcodes_l303_303501

def is_hexadecimal_prime (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 11 ∨ d = 13

def is_hexadecimal_digit (d : ℕ) : Prop :=
  0 ≤ d ∧ d ≤ 15

noncomputable def possible_passcodes_count : ℕ :=
  4050

theorem number_of_possible_passcodes :
  (∀ passcode : vector ℕ 4, (∀ d, d ∈ passcode.to_list → is_hexadecimal_digit d) → 
   (count (λ d, is_hexadecimal_prime d) passcode.to_list = 2) → 
   true) :=
begin
  sorry
end

end number_of_possible_passcodes_l303_303501


namespace measure_minor_arc_PB_l303_303129

structure Circle (α : Type) :=
(O : α) -- Center of the circle

noncomputable def measure_angle (a b c : ℕ) := 
  2 * b -- Inscribed angle theorem: arc PC = 2 * ∠PBC

theorem measure_minor_arc_PB (α : Type) (O : Circle α) (P B C : α)
  (H1 : measure_angle P 36 C = 72)
  (H2 : ∃ O : α, arc_PBC_forms_semicircle : (180 : ℕ)) :
  arc PB = 108 :=
by
  sorry

end measure_minor_arc_PB_l303_303129


namespace volume_of_second_cube_l303_303442

theorem volume_of_second_cube (V1 : ℝ) (condition1 : V1 = 8) 
  (A2 : ℝ) (condition2 : A2 = 3 * 6 * (2 : ℝ) ^ 2) :
  ∃ V2 : ℝ, V2 = 24 * real.sqrt 3 :=
by
  use (24 * real.sqrt 3)
  sorry

end volume_of_second_cube_l303_303442


namespace product_of_divisors_18_l303_303299

theorem product_of_divisors_18 : (∏ d in (list.range 18).filter (λ n, 18 % n = 0), d) = 18 ^ (9 / 2) :=
begin
  sorry
end

end product_of_divisors_18_l303_303299


namespace product_of_divisors_of_18_l303_303340

theorem product_of_divisors_of_18 : ∏ d in (Finset.filter (λ d, 18 % d = 0) (Finset.range 19)), d = 104976 := by
    sorry

end product_of_divisors_of_18_l303_303340


namespace find_lambda_l303_303092

variables {R : Type*} [Field R]
variables {V : Type*} [AddCommGroup V] [Module R V]
variables (a b : V)
variable (λ : R)

-- Defining collinearity
def not_collinear (a b : V) : Prop := ¬∃ k : R, b = k • a

-- Defining parallelism
def parallel (u v : V) : Prop := ∃ k : R, u = k • v

-- The problem statement
theorem find_lambda (h₁ : not_collinear a b)
                    (h₂ : parallel (λ • a + b) (a - 2 • b)) :
  λ = -1 / 2 :=
sorry

end find_lambda_l303_303092


namespace each_friend_pays_6413_l303_303863

noncomputable def amount_each_friend_pays (total_bill : ℝ) (friends : ℕ) (first_discount : ℝ) (second_discount : ℝ) : ℝ :=
  let bill_after_first_coupon := total_bill * (1 - first_discount)
  let bill_after_second_coupon := bill_after_first_coupon * (1 - second_discount)
  bill_after_second_coupon / friends

theorem each_friend_pays_6413 :
  amount_each_friend_pays 600 8 0.10 0.05 = 64.13 :=
by
  sorry

end each_friend_pays_6413_l303_303863


namespace product_of_divisors_of_18_l303_303366

theorem product_of_divisors_of_18 : ∏ d in {1, 2, 3, 6, 9, 18}, d = 5832 := by
  sorry

end product_of_divisors_of_18_l303_303366


namespace find_angle_B_l303_303661

theorem find_angle_B (a b c : ℝ) (A B C : ℝ)
  (h1 : (sin B - sin A, sqrt 3 * a + c) = (sin C, a + b)) :
  B = 5 * Real.pi / 6 :=
sorry

end find_angle_B_l303_303661


namespace hyperbola_intersection_l303_303060

variable (a b c : ℝ) -- positive constants
variables (F1 F2 : (ℝ × ℝ)) -- foci of the hyperbola

-- The positive constants a and b
axiom a_pos : a > 0
axiom b_pos : b > 0

-- The foci are at (-c, 0) and (c, 0)
axiom F1_def : F1 = (-c, 0)
axiom F2_def : F2 = (c, 0)

-- We want to prove that the points (-c, b^2 / a) and (-c, -b^2 / a) are on the hyperbola
theorem hyperbola_intersection :
  (F1 = (-c, 0) ∧ F2 = (c, 0) ∧ a > 0 ∧ b > 0) →
  ∀ y : ℝ, ∃ y1 y2 : ℝ, (y1 = b^2 / a ∧ y2 = -b^2 / a ∧ 
  ( ( (-c)^2 / a^2) - (y1^2 / b^2) = 1 ∧  (-c)^2 / a^2 - y2^2 / b^2 = 1 ) ) :=
by
  intros h
  sorry

end hyperbola_intersection_l303_303060


namespace find_common_ratio_l303_303696

noncomputable def common_ratio_of_geometric_sequence {a : ℕ → ℝ} (q : ℝ) : Prop :=
  (∀ n, a n > 0) → 
  (a 2 * a 6 = 16) → 
  (a 4 + a 8 = 8) → 
  q = 1

theorem find_common_ratio :
  ∃ (q : ℝ), common_ratio_of_geometric_sequence a q :=
begin
  -- Initiating the proof
  sorry
end

end find_common_ratio_l303_303696


namespace perimeter_of_square_field_l303_303458

variable (s a p : ℕ)

-- Given conditions as definitions
def area_eq_side_squared (a s : ℕ) : Prop := a = s^2
def perimeter_eq_four_sides (p s : ℕ) : Prop := p = 4 * s
def given_equation (a p : ℕ) : Prop := 6 * a = 6 * (2 * p + 9)

-- The proof statement
theorem perimeter_of_square_field (s a p : ℕ) 
  (h1 : area_eq_side_squared a s)
  (h2 : perimeter_eq_four_sides p s)
  (h3 : given_equation a p) :
  p = 36 :=
by
  sorry

end perimeter_of_square_field_l303_303458


namespace sum_of_squares_primes_l303_303590

theorem sum_of_squares_primes (k : ℕ) (p : Fin k → ℕ) (prime_p : ∀ i, Prime (p i)) (distinct_p : Pairwise (≠) (Finset.univ.image p)) :
  (∃ (k = 10),
  ∑ i, (p i)^2 = 2010) :=
sorry

end sum_of_squares_primes_l303_303590


namespace arithmetic_sequence_50th_term_l303_303922

theorem arithmetic_sequence_50th_term (a d n : ℕ) (h_start : a = 2) (h_diff : d = 5) (h_n : n = 50) :
  a + (n - 1) * d = 247 :=
by {
  rw [h_start, h_diff, h_n],
  norm_num,
}

end arithmetic_sequence_50th_term_l303_303922


namespace smallest_positive_integer_is_53_l303_303930

theorem smallest_positive_integer_is_53 :
  ∃ a : ℕ, a > 0 ∧ a % 3 = 2 ∧ a % 4 = 1 ∧ a % 5 = 3 ∧ a = 53 :=
by
  sorry

end smallest_positive_integer_is_53_l303_303930


namespace intersection_point_of_AL_and_BC_constant_l303_303730

-- Definitions of the geometric objects
variables {A B C P Q S R L : Type} [geometry_space : geometry_space A]
variables 
  (triangle_ABC : is_triangle A B C) 
  (P_on_AB : is_between P A B) 
  (Q_on_AC : is_between Q A C) 
  (BPQ_is_cyclic : is_cyclic_quadrilateral B P Q C) 
  (circumcircle_ABQ : is_circle_through A B Q) 
  (circumcircle_APC : is_circle_through A P C)
  (S_on_BC : is_intersect S (line_of_circle circumcircle_ABQ) (line_of BC) S ≠ B)
  (R_on_BC : is_intersect R (line_of_circle circumcircle_APC) (line_of BC) R ≠ B)
  (L_intersection : is_intersect L (line_of PR) (line_of QS))

-- The theorem to prove
theorem intersection_point_of_AL_and_BC_constant :
  ∀ (P Q : Type) [is_between P A B] [is_between Q A C] [is_cyclic_quadrilateral B P Q C],
  let S' := is_intersect S' (line_of_circle (is_circle_through A B Q)) (line_of BC) (S' ≠ B),
  let R' := is_intersect R' (line_of_circle (is_circle_through A P C)) (line_of BC) (R' ≠ B),
  let L' := is_intersect L' (line_of PR') (line_of QS'),
  intersection_point AL BC = intersection_point A' L' BC :=
  sorry

end intersection_point_of_AL_and_BC_constant_l303_303730


namespace Linda_savings_l303_303750

theorem Linda_savings (S : ℝ) : (5 / 6 * S) + 500 = S → S = 3000 := by
  intro h
  calc
  S = S - (5 / 6 * S) + 500 : by rw [h]
   ... = 1 / 6 * S + 500    : by ring
   ... = 500 + 500 : by norm_num
   ... = 3000 / 6 + 3000 - 3000 / 6 - 500 : by kernel
   ... = (6 / 6 * 3000) - (5 / 6 * 3000) - 500
   end have the side avantages=>
   -- sorry

end Linda_savings_l303_303750


namespace arithmetic_geometric_seq_summation_behavior_l303_303610

noncomputable def a_n (a_1 d : ℝ) (n : ℕ) : ℝ := a_1 + (n - 1) * d

noncomputable def S_n (a_1 d : ℝ) (n : ℕ) : ℝ := n / 2 * (2 * a_1 + (n - 1) * d)

theorem arithmetic_geometric_seq (a_1 d : ℝ) (n : ℕ) (h1 : d ≠ 0) (h2 : a_1 * (a_1 + 5 * d) = (a_1 + 3 * d) ^ 2) :
  S_n a_1 d 19 = 0 :=
by
  have a1_eq : a_1 = -9 * d,
  { sorry }, -- Details of the intermediate proofs are omitted as per instruction
  have a_10_eq : a_n a_1 d 10 = 0,
  { sorry }, -- Details of the intermediate proofs are omitted as per instruction
  have s_19_eq : S_n a_1 d 19 = 0,
  { sorry }, -- Details of the intermediate proofs are omitted as per instruction
  exact s_19_eq

theorem summation_behavior (a_1 d : ℝ) (h1 : d ≠ 0) (h2 : a_1 * (a_1 + 5 * d) = (a_1 + 3 * d) ^ 2) :
  (d < 0 → S_n a_1 d 9 = S_n a_1 d 10) ∧ (d > 0 → S_n a_1 d 10 = S_n a_1 d 9) :=
by
  have a1_eq : a_1 = -9 * d,
  { sorry }, -- Details of the intermediate proofs are omitted as per instruction
  sorry -- The detailed behavior proofs omitted as per instruction

end arithmetic_geometric_seq_summation_behavior_l303_303610


namespace slope_of_line_through_points_l303_303438

def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (6, -7)

theorem slope_of_line_through_points : 
  ∀ (p1 p2 : ℝ × ℝ), p1 = point1 → p2 = point2 → 
  (p2.snd - p1.snd) / (p2.fst - p1.fst) = -2 :=
by
  intro p1 p2 h1 h2
  rw [h1, h2]
  calc
    (-7 - 3) / (6 - 1) = (-10) / 5 : by norm_num
    ... = -2 : by norm_num

end slope_of_line_through_points_l303_303438


namespace equal_cells_of_both_colors_l303_303787

theorem equal_cells_of_both_colors (m n : ℕ) (board : ℕ → ℕ → ℤ)
  (h : ∀ i j, 
    board i j = 1 ∨ board i j = -1 ∧
    board i j * (∑ x, board i x + ∑ y, board y j) ≤ 0) : 
  (∀ i, ∑ j, board i j = 0) ∧ (∀ j, ∑ i, board i j = 0) :=
sorry

end equal_cells_of_both_colors_l303_303787


namespace equilateral_iff_sum_altitudes_eq_9r_l303_303692

-- Let a, b, c be the sides of the triangle
variables (a b c : ℝ)

-- Let h_a, h_b, h_c be the altitudes of the triangle from vertices opposite sides a, b, c respectively
noncomputable def h_a (Δ : ℝ) (a : ℝ) : ℝ := 2 * Δ / a
noncomputable def h_b (Δ : ℝ) (b : ℝ) : ℝ := 2 * Δ / b
noncomputable def h_c (Δ : ℝ) (c : ℝ) : ℝ := 2 * Δ / c

-- Let r be the inradius of the triangle
variable (r : ℝ)

-- Let s be the semiperimeter of the triangle
noncomputable def s (a b c : ℝ) : ℝ := (a + b + c) / 2

-- The area of the triangle Δ is given by Δ = r * s
noncomputable def Δ (r : ℝ) (s : ℝ) : ℝ := r * s

-- Main theorem to be proven
theorem equilateral_iff_sum_altitudes_eq_9r :
  (h_a (Δ r (s a b c)) a + h_b (Δ r (s a b c)) b + h_c (Δ r (s a b c)) c = 9 * r) ↔ (a = b = c) :=
by
  sorry

end equilateral_iff_sum_altitudes_eq_9r_l303_303692


namespace binomial_sum_mod_1000_l303_303514

open BigOperators

theorem binomial_sum_mod_1000 :
  ((∑ k in finset.range 503 \ finset.range 3, nat.choose 2011 (4 * k)) % 1000) = 49 := 
sorry

end binomial_sum_mod_1000_l303_303514
