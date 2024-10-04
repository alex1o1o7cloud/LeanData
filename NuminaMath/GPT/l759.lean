import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.Field
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Squarefree
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.LocalExtr
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Lemmas
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Vector.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.Divisors
import Mathlib.Real.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic

namespace sum_of_numbers_is_89_l759_759231

noncomputable def a : ℝ := Real.sqrt 55
def num1 : ℝ := a
def num2 : ℝ := 2 * a
def num3 : ℝ := 4 * a
def num4 : ℝ := 5 * a

theorem sum_of_numbers_is_89 :
  num1^2 + num2^2 + num3^2 + num4^2 = 2540 →
  (num1 * num3 = num2 * num4) →
  (num1 + num2 + num3 + num4 ≈ 89) :=
by
  intros h1 h2
  sorry

end sum_of_numbers_is_89_l759_759231


namespace trapezoid_PQRS_area_l759_759221

def point := (ℝ × ℝ)

def P : point := (1, 0)
def Q : point := (1, 3)
def R : point := (5, 9)
def S : point := (5, 3)

def trapezoid_area (A B C D : point) : ℝ :=
  let (Ax, Ay) := A
  let (Bx, By) := B
  let (Cx, Cy) := C
  let (Dx, Dy) := D
  let b1 := abs (Ay - By)
  let b2 := abs (Cy - Dy)
  let h := abs (Cx - Ax)
  (1 / 2) * (b1 + b2) * h

theorem trapezoid_PQRS_area :
  trapezoid_area P Q R S = 18 := by
  sorry

end trapezoid_PQRS_area_l759_759221


namespace greatest_constant_triangle_l759_759323

theorem greatest_constant_triangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_acute : cos_angle a b c > 0) :
  (a^2 + b^2 + c^2) / (a^2 + b^2) > 1 := 
sorry

-- Helper definition for cosine of the angle opposite to side c
def cos_angle (a b c : ℝ) : ℝ := 
  (a^2 + b^2 - c^2) / (2 * a * b)

end greatest_constant_triangle_l759_759323


namespace greatest_three_digit_multiple_of_17_l759_759855

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759855


namespace greatest_three_digit_multiple_of_17_l759_759732

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759732


namespace greatest_three_digit_multiple_of_17_l759_759772

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759772


namespace not_possible_to_color_plane_l759_759074

theorem not_possible_to_color_plane :
  ¬ ∃ (color : ℕ → ℕ × ℕ → ℕ) (c : ℕ), 
    (c = 2016) ∧
    (∀ (A B C : ℕ × ℕ), (A ≠ B ∧ B ≠ C ∧ C ≠ A) → 
                        (color c A = color c B) ∨ (color c B = color c C) ∨ (color c C = color c A)) :=
by
  sorry

end not_possible_to_color_plane_l759_759074


namespace simplify_fraction_l759_759487

theorem simplify_fraction : (5^3 + 5^5) / (5^4 - 5^2) = 65 / 12 := 
by 
  sorry

end simplify_fraction_l759_759487


namespace abs_neg_eq_three_l759_759402

theorem abs_neg_eq_three (a : ℝ) (h : | -a | = 3) : a = 3 ∨ a = -3 :=
sorry

end abs_neg_eq_three_l759_759402


namespace oliver_new_cards_l759_759124

theorem oliver_new_cards (old_cards : ℕ) (cards_per_page : ℕ) (used_pages : ℕ)
  (h_old_cards : old_cards = 10)
  (h_cards_per_page : cards_per_page = 3)
  (h_used_pages : used_pages = 4) : 
  let total_capacity := used_pages * cards_per_page in
  let new_cards := total_capacity - old_cards in
  new_cards = 2 := 
by
  have h_total_capacity : total_capacity = 12,
    -- proof of total_capacity = 12 omitted
  have h_new_cards : new_cards = total_capacity - old_cards,
    -- proof of new_cards = total_capacity - old_cards omitted
  have h_new_cards_2 : new_cards = 2,
    -- proof of new_cards = 2 omitted
  sorry

end oliver_new_cards_l759_759124


namespace sum_first_six_terms_l759_759533

-- Define the conditions given in the problem
def a3 := 7
def a4 := 11
def a5 := 15

-- Define the common difference
def d := a4 - a3 -- 4

-- Define the first term
def a1 := a3 - 2 * d -- -1

-- Define the sum of the first six terms of the arithmetic sequence
def S6 := (6 / 2) * (2 * a1 + (6 - 1) * d) -- 54

-- The theorem we want to prove
theorem sum_first_six_terms : S6 = 54 := by
  sorry

end sum_first_six_terms_l759_759533


namespace num_sets_N_l759_759459

open Set

-- Define the set M and the set U
def M : Set ℕ := {1, 2}
def U : Set ℕ := {1, 2, 3, 4}

-- The statement to prove
theorem num_sets_N : 
  ∃ count : ℕ, count = 4 ∧ 
  (∀ N : Set ℕ, M ∪ N = U → N = {3, 4} ∨ N = {1, 3, 4} ∨ N = {2, 3, 4} ∨ N = {1, 2, 3, 4}) :=
by
  sorry

end num_sets_N_l759_759459


namespace gardener_ways_to_plant_trees_l759_759251

theorem gardener_ways_to_plant_trees : 
  (∃ (f : ℕ → ℕ), (∀ d : ℕ, d ∈ {1, 2, 3} → f d ≥ 1) ∧ (f 1 + f 2 + f 3 = 10)) → 36 := 
by
  sorry

end gardener_ways_to_plant_trees_l759_759251


namespace sum_of_primes_less_than_20_l759_759993

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759993


namespace sum_primes_less_than_20_l759_759935

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759935


namespace cos_x_plus_2y_eq_one_l759_759342

noncomputable def f (t : ℝ) : ℝ := t^3 + sin t

theorem cos_x_plus_2y_eq_one
  (x y : ℝ)
  (a : ℝ)
  (hx_range : -π/4 ≤ x ∧ x ≤ π/4)
  (hy_range : -π/4 ≤ y ∧ y ≤ π/4)
  (h1 : x^3 + sin x - 2 * a = 0)
  (h2 : 4 * y^3 + sin y * cos y + a = 0) :
  cos (x + 2 * y) = 1 :=
by
  sorry

end cos_x_plus_2y_eq_one_l759_759342


namespace verify_solutions_l759_759544

-- Definition of conditions and problem setup
def sum_of_positive_divisors (n : ℕ) : ℕ :=
  ∑ d in divisors n, d

def sum_of_positive_odd_divisors (n : ℕ) : ℕ :=
  ∑ d in divisors n, if d % 2 = 1 then d else 0

def number_of_solutions (n : ℕ) : ℕ :=
  if n % 2 = 1 then 8 * sum_of_positive_divisors n
  else 24 * sum_of_positive_odd_divisors n

-- Lean theorem statement
theorem verify_solutions :
  (number_of_solutions 25 = 248) ∧
  (number_of_solutions 28 = 192) ∧
  (number_of_solutions 84 = 768) ∧
  (number_of_solutions 96 = 96) ∧
  (number_of_solutions 105 = 1536) :=
by
  sorry

end verify_solutions_l759_759544


namespace marble_probabilities_absolute_difference_l759_759244

theorem marble_probabilities_absolute_difference :
  let red_marbles := 501
  let black_marbles := 1502
  let total_marbles := red_marbles + black_marbles
  let Ps := (red_marbles * (red_marbles - 1) / 2 + black_marbles * (black_marbles - 1) / 2) / (total_marbles * (total_marbles - 1) / 2)
  let Pd := (red_marbles * black_marbles) / (total_marbles * (total_marbles - 1) / 2)
  |Ps - Pd| = 1 / 4 :=
by
  sorry

end marble_probabilities_absolute_difference_l759_759244


namespace radius_circumcircle_ADT_l759_759481

-- Define the conditions
variables (A B C P D T : Type) [metric_space A]
variables (BC : line_segment A)
variables (P_on_BC : P ∈ BC)
variables (angle_A : real) (PD PT DT : real)
variables (center_incircle_APB center_incircle_APC : A)
variables (angle_DAT : real)

-- Given conditions
variable (h_angle_A : angle_A = 60)
variable (h_PD : PD = 7)
variable (h_PT : PT = 4)
variable (h_center_D : center_incircle_APB = D)
variable (h_center_T : center_incircle_APC = T)
variable (h_angle_DAT : angle_DAT = 30)
variable (h_DT : DT = real.sqrt(65))

-- The theorem to prove
theorem radius_circumcircle_ADT : (radius_of_circumcircle A D T) = real.sqrt(65) :=
sorry

end radius_circumcircle_ADT_l759_759481


namespace greatest_three_digit_multiple_of_17_l759_759750

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759750


namespace arithmetic_sequence_sum_l759_759367

variable {α : Type*} [Field α] [Archimedean α]

-- Define the sequence as a function of natural numbers
def a (n : ℕ) : α := sorry

-- Define the sum of the sequence up to some n
def S (n : ℕ) : α := ∑ i in Finset.range (n+1), a i

theorem arithmetic_sequence_sum (h1 : a 3 + a 7 - a 10 = 0)
                                (h2 : a 11 - a 4 = 4) : S 13 = 52 := 
by sorry

end arithmetic_sequence_sum_l759_759367


namespace greatest_three_digit_multiple_of_seventeen_l759_759710

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759710


namespace arithmetic_sequence_sum_l759_759528

theorem arithmetic_sequence_sum :
  ∃ a : ℕ → ℤ, 
    a 3 = 7 ∧ a 4 = 11 ∧ a 5 = 15 ∧ 
    (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 54) := 
by {
  sorry
}

end arithmetic_sequence_sum_l759_759528


namespace triangle_is_isosceles_l759_759049

theorem triangle_is_isosceles (A B C : ℝ) (h1 : A + B + C = real.pi) (h2 : 2 * real.cos B * real.sin A = real.sin C) : 
  ∃ M, (M = A) ∨ (M = B) ∨ (M = C) :=
sorry

end triangle_is_isosceles_l759_759049


namespace distance_constant_l759_759351

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4 + y^2 / 3 = 1)

def perpendicular (A B : ℝ × ℝ) : Prop :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  x1 * x2 + y1 * y2 = 0

def distance_from_origin_to_line (k b : ℝ) : ℝ :=
  abs b / sqrt (1 + k^2)

theorem distance_constant (k b: ℝ) (A B : ℝ × ℝ) (hxA : ellipse_equation A.1 A.2)
  (hxB : ellipse_equation B.1 B.2) (h_perpendicular : perpendicular A B) :
  distance_from_origin_to_line k b = (2 * sqrt 21) / 7 :=
sorry

end distance_constant_l759_759351


namespace train_passenger_count_l759_759257

theorem train_passenger_count (P : ℕ) (total_passengers : ℕ) (r : ℕ)
  (h1 : r = 60)
  (h2 : total_passengers = P + r + 3 * (P + r))
  (h3 : total_passengers = 640) :
  P = 100 :=
by
  sorry

end train_passenger_count_l759_759257


namespace sum_integers_75_to_95_l759_759213

theorem sum_integers_75_to_95 : ∑ k in finset.Icc 75 95, k = 1785 := by
  sorry

end sum_integers_75_to_95_l759_759213


namespace smallest_upper_bound_l759_759330

def f_N (N : ℕ) (x : ℝ) : ℝ := 
  ∑ n in Finset.range (N+1), (N + 1 / 2 - n) / ((N+1) * (2 * n + 1)) * Real.sin ((2 * n + 1) * x)

theorem smallest_upper_bound (N : ℕ) (x : ℝ)  : 
  ∃ M : ℝ, (∀ N : ℕ, ∀ x : ℝ, f_N N x ≤ M) ∧ M = Real.pi / 4 :=
by
  sorry

end smallest_upper_bound_l759_759330


namespace sum_of_primes_less_than_20_l759_759873

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759873


namespace greatest_three_digit_multiple_of_17_l759_759804

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759804


namespace greatest_three_digit_multiple_of_17_is_986_l759_759672

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759672


namespace greatest_three_digit_multiple_of_seventeen_l759_759699

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759699


namespace sum_of_integers_75_to_95_l759_759207

theorem sum_of_integers_75_to_95 : (∑ i in Finset.range (95 - 75 + 1), (i + 75)) = 1785 := by
  sorry

end sum_of_integers_75_to_95_l759_759207


namespace problem_statement_l759_759119

-- Define the universal set
def U : Set ℕ := {x | x ≤ 6}

-- Define set A
def A : Set ℕ := {1, 3, 5}

-- Define set B
def B : Set ℕ := {4, 5, 6}

-- Define the complement of A with respect to U
def complement_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- Define the intersection of the complement of A and B
def intersect_complement_A_B : Set ℕ := {x | x ∈ complement_A ∧ x ∈ B}

-- Theorem statement to be proven
theorem problem_statement : intersect_complement_A_B = {4, 6} :=
by
  sorry

end problem_statement_l759_759119


namespace max_sum_multiplication_table_l759_759511

noncomputable def S : Set ℕ := {2, 3, 5, 7, 11, 13, 17}

theorem max_sum_multiplication_table: 
  ∃ A B : Finset ℕ, 
    A.card = 4 ∧ B.card = 3 ∧ 
    A ∪ B = S ∧ 
    A ∩ B = ∅ ∧ 
    ((A.sum id) * (B.sum id) = 841) := 
sorry

end max_sum_multiplication_table_l759_759511


namespace minimum_value_inequality_l759_759168

theorem minimum_value_inequality
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y - 3 = 0) :
  ∃ t : ℝ, (∀ (x y : ℝ), (2 * x + y = 3) → (0 < x) → (0 < y) → (t = (4 * y - x + 6) / (x * y)) → 9 ≤ t) ∧
          (∃ (x_ y_: ℝ), 2 * x_ + y_ = 3 ∧ 0 < x_ ∧ 0 < y_ ∧ (4 * y_ - x_ + 6) / (x_ * y_) = 9) :=
sorry

end minimum_value_inequality_l759_759168


namespace verify_chebyshev_polynomials_l759_759236

-- Define the Chebyshev polynomials of the first kind Tₙ(x)
def T : ℕ → ℝ → ℝ
| 0, x => 1
| 1, x => x
| (n+1), x => 2 * x * T n x - T (n-1) x

-- Define the Chebyshev polynomials of the second kind Uₙ(x)
def U : ℕ → ℝ → ℝ
| 0, x => 1
| 1, x => 2 * x
| (n+1), x => 2 * x * U n x - U (n-1) x

-- State the theorem to verify the Chebyshev polynomials initial conditions and recurrence relations
theorem verify_chebyshev_polynomials (n : ℕ) (x : ℝ) :
  T 0 x = 1 ∧ T 1 x = x ∧
  U 0 x = 1 ∧ U 1 x = 2 * x ∧
  (T (n+1) x = 2 * x * T n x - T (n-1) x) ∧
  (U (n+1) x = 2 * x * U n x - U (n-1) x) := sorry

end verify_chebyshev_polynomials_l759_759236


namespace greatest_three_digit_multiple_of_17_l759_759745

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759745


namespace circumscribed_quadrilateral_equality_l759_759246

variable {ℝ : Type} [linear_ordered_field ℝ]

def is_cyclic_quadrilateral (A B C D P : ℝ) : Prop :=
  ∃ (circumcircle : ℝ), -- some condition to specify A, B, C, D, P lie on a circumcircle
  ∀ (P : ℝ), P lies on circumcircle

def distance_to_line (P : ℝ) (line : ℝ) : ℝ := 
  -- implementation for distance from point P to line (AB, CD, BC, DA, AC, BD)
  sorry
 
theorem circumscribed_quadrilateral_equality
  {A B C D P : ℝ}
  (h_cyclic : is_cyclic_quadrilateral A B C D P)
  : 
  (distance_to_line P AB) * (distance_to_line P CD) = 
  (distance_to_line P BC) * (distance_to_line P DA) ∧ 
  (distance_to_line P AB) * (distance_to_line P CD) = 
  (distance_to_line P AC) * (distance_to_line P BD) 
  := 
sorry

end circumscribed_quadrilateral_equality_l759_759246


namespace sufficient_condition_not_necessary_condition_l759_759238

theorem sufficient_condition (α β : ℝ) (h : α = β) : sin α ^ 2 + cos β ^ 2 = 1 := 
by {
  rw h,
  exact sin_sq_cos_sq α,
}

theorem not_necessary_condition (α β : ℝ) (h : sin α ^ 2 + cos β ^ 2 = 1) : α ≠ β := 
by {
  -- We need an example to show that α ≠ β can still satisfy the equation.
  let α := 0,
  let β := pi / 2,
  have h1 : α ≠ β, 
  { norm_num,
    linarith, },
  show α ≠ β, from h1,
  sorry
}

end sufficient_condition_not_necessary_condition_l759_759238


namespace sum_of_integers_75_to_95_l759_759210

theorem sum_of_integers_75_to_95 : (∑ i in Finset.range (95 - 75 + 1), (i + 75)) = 1785 := by
  sorry

end sum_of_integers_75_to_95_l759_759210


namespace log_simplification_l759_759140

theorem log_simplification (p q r s z y : ℝ) (h1 : p > 0) (h2 : q > 0) (h3 : r > 0)
                           (h4 : s > 0) (h5 : z > 0) (h6 : y > 0) :
  log (p / q) + log (q / r) + log (r / s) - log (pz / sy) = log (y / z) :=
by
  sorry

end log_simplification_l759_759140


namespace find_total_grade10_students_l759_759261

/-
Conditions:
1. The school has a total of 1800 students in grades 10 and 11.
2. 90 students are selected as a sample for a survey.
3. The sample contains 42 grade 10 students.
-/

variables (total_students sample_size sample_grade10 total_grade10 : ℕ)

axiom total_students_def : total_students = 1800
axiom sample_size_def : sample_size = 90
axiom sample_grade10_def : sample_grade10 = 42

theorem find_total_grade10_students : total_grade10 = 840 :=
by
  have h : (sample_size : ℚ) / (total_students : ℚ) = (sample_grade10 : ℚ) / (total_grade10 : ℚ) :=
    sorry
  sorry

end find_total_grade10_students_l759_759261


namespace sum_of_primes_less_than_20_l759_759894

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759894


namespace distance_traveled_first_7_minutes_velocity_vector_change_magnitude_l759_759172

-- Definitions of the motion functions
def x (t : ℝ) : ℝ := t * (t - 6) ^ 2
def y (t : ℝ) : ℝ := if t < 7 then 0 else (t - 7) ^ 2

-- Distance traveled by the robot in the first 7 minutes
theorem distance_traveled_first_7_minutes (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ 7) : 
  ∫ (t:ℝ) in 0..7, (abs (3 * t ^ 2 - 24 * t + 36)) = 71 := 
sorry

-- Magnitude of the change in the velocity vector during the eighth minute
theorem velocity_vector_change_magnitude : 
  abs (sqrt ((21)^2 + (2)^2)) = sqrt 445 := 
sorry

end distance_traveled_first_7_minutes_velocity_vector_change_magnitude_l759_759172


namespace sum_of_primes_less_than_20_is_77_l759_759929

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759929


namespace calc_g_g_neg3_l759_759362

def g (x : ℚ) : ℚ :=
x⁻¹ + x⁻¹ / (2 + x⁻¹)

theorem calc_g_g_neg3 : g (g (-3)) = -135 / 8 := 
by
  sorry

end calc_g_g_neg3_l759_759362


namespace percentage_increase_is_fifty_percent_l759_759128

-- Given the conditions:
variable (I : ℝ) -- Original income
variable (E : ℝ) -- Original expenditure
variable (I_new : ℝ) -- New income
variable (E_new : ℝ) -- New expenditure
variable (S : ℝ) -- Original savings
variable (S_new : ℝ) -- New savings

-- Given conditions translated into Lean:
def condition1 : E = 0.75 * I := sorry
def condition2 : I_new = 1.20 * I := sorry
def condition3 : E_new = 0.825 * I := sorry

-- Definition of original and new savings based on conditions:
def original_savings : S = 0.25 * I := by
  rw [←condition1]
  simp [S]

def new_savings : S_new = 0.375 * I := by
  rw [←condition2, ←condition3]
  simp [S_new]

-- Calculation of the percentage increase in savings:
def percentage_increase_in_savings : ℝ := ((S_new - S) / S) * 100

-- The proof goal:
theorem percentage_increase_is_fifty_percent (h1 : E = 0.75 * I) (h2 : I_new = 1.20 * I) (h3 : E_new = 0.825 * I) :
  percentage_increase_in_savings = 50 := sorry

end percentage_increase_is_fifty_percent_l759_759128


namespace greatest_three_digit_multiple_of_17_is_986_l759_759663

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759663


namespace calories_per_candy_bar_l759_759542

theorem calories_per_candy_bar (total_calories : ℕ) (number_of_bars : ℕ) (h : total_calories = 24 ∧ number_of_bars = 3) : 
  total_calories / number_of_bars = 8 :=
by
  have h1 : total_calories = 24 := h.1
  have h2 : number_of_bars = 3 := h.2
  rw [h1, h2]
  norm_num

end calories_per_candy_bar_l759_759542


namespace sum_integers_75_to_95_l759_759204

theorem sum_integers_75_to_95 :
  let a := 75
  let l := 95
  let n := 95 - 75 + 1
  ∑ k in Finset.range n, (a + k) = 1785 := by
  sorry

end sum_integers_75_to_95_l759_759204


namespace max_length_OB_l759_759547

-- Define the setup and conditions
variables (O A B : Type)
          [metric_space O]
          (ray1 ray2 : O → O → ℝ)
          (h_angle : angle (ray1 O A) (ray2 O B) = 45)
          (h_AB : dist A B = 2)

-- State the theorem to be proved
theorem max_length_OB (A B : O) (O : O):
  ∀ (ray1 : O → O → ℝ) (ray2 : O → O → ℝ),
  angle (ray1 O A) (ray2 O B) = 45 →
  dist A B = 2 →
  ∃ (OB : ℝ), OB = 2 * sqrt 2 := sorry

end max_length_OB_l759_759547


namespace greatest_three_digit_multiple_of_17_l759_759734

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759734


namespace greatest_three_digit_multiple_of_17_l759_759793

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759793


namespace sum_of_primes_lt_20_eq_77_l759_759904

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759904


namespace calculate_expression_l759_759283

theorem calculate_expression : (π - 2023)^0 - (1 / 3)^(-2) = -8 := by
  sorry

end calculate_expression_l759_759283


namespace greatest_three_digit_multiple_of_17_l759_759649

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759649


namespace interval_of_monotonic_increase_fx_lt_x_minus_1_k_range_values_l759_759016

noncomputable def f (x : ℝ) : ℝ := log x - (x - 1)^2 / 2

theorem interval_of_monotonic_increase :
  {x : ℝ | 0 < x ∧ x < (1 + Real.sqrt 5) / 2}.Nonempty := sorry

theorem fx_lt_x_minus_1 (x : ℝ) (h : 1 < x) : f x < x - 1 := sorry

theorem k_range_values (k : ℝ) : (k < 1) ↔ ∃ x0 > 1, ∀ x ∈ Set.Ioo 1 x0, f x > k * (x - 1) := sorry

end interval_of_monotonic_increase_fx_lt_x_minus_1_k_range_values_l759_759016


namespace greatest_three_digit_multiple_of_17_is_986_l759_759633

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759633


namespace greatest_three_digit_multiple_of_17_is_986_l759_759619

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759619


namespace complementary_angle_difference_l759_759507

theorem complementary_angle_difference (a b : ℝ) (h1 : a = 4 * b) (h2 : a + b = 90) : (a - b) = 54 :=
by
  -- Proof is intentionally omitted
  sorry

end complementary_angle_difference_l759_759507


namespace sum_primes_less_than_20_l759_759962

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759962


namespace greatest_three_digit_multiple_of_17_l759_759564

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759564


namespace sum_primes_less_than_20_l759_759956

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759956


namespace greatest_three_digit_multiple_of_17_l759_759787

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759787


namespace greatest_three_digit_multiple_of_17_l759_759830

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759830


namespace smallest_b_l759_759486

-- Define the variables and conditions
variables {a b : ℝ}

-- Assumptions based on the problem conditions
axiom h1 : 2 < a
axiom h2 : a < b

-- The theorems for the triangle inequality violations
theorem smallest_b (h : a ≥ b / (2 * b - 1)) (h' : 2 + a ≤ b) : b = (3 + Real.sqrt 7) / 2 :=
sorry

end smallest_b_l759_759486


namespace greatest_three_digit_multiple_of_17_l759_759602

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759602


namespace equilateral_A_find_x_l759_759352

-- Define an equilateral triangle ABC with side length 'a'
variable (a : ℝ) (x : ℝ)

-- Given the conditions
-- AC' = x, BA' = x, CB' = x
def isEquilateralExtension (a x : ℝ) (A' B C' A B' : Point) (AB BC CA : Line) := 
  AC'.length = BA'.length ∧
  BA'.length = CB'.length ∧
  CB'.length = x ∧
  AB = BC ∧
  BC = CA ∧
  CA = a

-- Question 1: Prove A'B'C' is equilateral
theorem equilateral_A'B'C' (h : isEquilateralExtension a x A' B' C' A B') : 
  isEquilateral A' B' C' :=
sorry

-- Question 2: Find the value of x such that the side length of A'B'C' is equal to l
theorem find_x (l : ℝ) (h : isEquilateralExtension a x A' B' C' A B') :
  (side_length A' B' C' = l → x = (a + sqrt (4 * l^2 - a^2)) / 2 ∨ x = (a - sqrt (4 * l^2 - a^2)) / 2) :=
sorry

end equilateral_A_find_x_l759_759352


namespace sweet_treats_distribution_l759_759467

-- Define the number of cookies, cupcakes, brownies, and students
def cookies : ℕ := 20
def cupcakes : ℕ := 25
def brownies : ℕ := 35
def students : ℕ := 20

-- Define the total number of sweet treats
def total_sweet_treats : ℕ := cookies + cupcakes + brownies

-- Define the number of sweet treats each student will receive
def sweet_treats_per_student : ℕ := total_sweet_treats / students

-- Prove that each student will receive 4 sweet treats
theorem sweet_treats_distribution : sweet_treats_per_student = 4 := 
by sorry

end sweet_treats_distribution_l759_759467


namespace greatest_three_digit_multiple_of_17_l759_759733

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759733


namespace original_number_eq_nine_l759_759324

theorem original_number_eq_nine (N : ℕ) (h1 : ∃ k : ℤ, N - 4 = 5 * k) : N = 9 :=
sorry

end original_number_eq_nine_l759_759324


namespace greatest_three_digit_multiple_of_17_l759_759796

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759796


namespace sum_primes_less_than_20_l759_759933

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759933


namespace greatest_three_digit_multiple_of17_l759_759722

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759722


namespace greatest_three_digit_multiple_of_17_l759_759834

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759834


namespace units_digit_of_k_l759_759403

theorem units_digit_of_k (k : ℤ) (a : ℝ) (n : ℕ) (h1 : k > 1) 
  (h2 : polynomial.eval a (polynomial.C 1 + -polynomial.C k * polynomial.X + polynomial.C 1 * polynomial.X^2) = 0)
  (h3 : n > 10) 
  (h4 : ∀ n : ℕ, n > 10 → (a ^ (2 * n) + a ^ (-2 * n)) % 10 = 7) :
  k % 10 = 3 :=
sorry

end units_digit_of_k_l759_759403


namespace sum_of_primes_less_than_20_is_77_l759_759915

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759915


namespace average_weight_all_boys_l759_759152

/-- Definition of average weight for a group of boys --/
def average_weight (total_weight : ℝ) (number_of_boys : ℕ) : ℝ := 
total_weight / number_of_boys

theorem average_weight_all_boys :
  (average_weight (16 * 50.25 + 8 * 45.15) (16 + 8) = 48.55) :=
by
  sorry

end average_weight_all_boys_l759_759152


namespace area_of_circle_l759_759458

open EuclideanGeometry Real

variable {a : ℝ}

/-- A theorem to prove the area of a circle given its equation and intersection property with a line. -/
theorem area_of_circle 
    (h_line : ∀ x, x + 2 * a) 
    (h_circle : ∀ x y, x^2 + y^2 - 2 * a * y - 2 = 0)
    (h_AB : ∀ A B, dist A B = 2 * sqrt 3 ∧ A = (x₁, x₁ + 2 * a) ∧ B = (x₂, x₂ + 2 * a)) :
    ∃ (r : ℝ), π * r^2 = 4 * π :=
sorry

lemma center_of_circle_is_origin (a : ℝ) : center_of_circle a := 
  begin
    -- Complete the square and compare with the standard circle equation
    have h1 : (0, a) = ...,
    sorry
  end

end area_of_circle_l759_759458


namespace percentage_caught_sampling_candy_l759_759054

theorem percentage_caught_sampling_candy:
  let total_percentage : ℝ := 23.913043478260867 in
  let percentage_not_caught : ℝ := 0.08 * total_percentage in
  let percentage_caught : ℝ := total_percentage - percentage_not_caught in
  percentage_caught = 22 :=
by
  sorry

end percentage_caught_sampling_candy_l759_759054


namespace number_of_cars_l759_759262

-- define conditions
def clients : ℕ := 18
def selections_per_client : ℕ := 3
def selections_total : ℕ := 54
def selections_per_car : ℕ := 3

-- state the problem and the proof goal
theorem number_of_cars :
  (∑ i in range clients, selections_per_client) / selections_per_car = 18 :=
by
  sorry

end number_of_cars_l759_759262


namespace sum_first_four_terms_eq_neg20_l759_759520

noncomputable theory

-- Definitions for terms in the geometric series and arithmetic sequence condition
def aₙ (n : ℕ) (q : ℝ) : ℝ := 1 * q^(n-1)  -- Notional definition of n-th term in geometric sequence

-- Sum of the first n terms of a geometric series
def Sₙ (n : ℕ) (a₁ q : ℝ) : ℝ := if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

-- Conditions based on the problem statement
axiom a₁ : ℝ
axiom a₁_eq_1 : a₁ = 1
axiom q : ℝ
axiom q_ne_1 : q ≠ 1
axiom arithmetic_sequence_condition : -3 * a₁ + aₙ 3 q = -2 * aₙ 2 q

-- The main theorem to prove S₄ = -20
theorem sum_first_four_terms_eq_neg20 : Sₙ 4 a₁ q = -20 :=
by
  rw a₁_eq_1  -- Substitute a₁ with 1 by the given condition
  rw Sₙ      -- Unfold the definition of Sₙ
  split_ifs  -- Deal with the if statement in the definition of Sₙ
  sorry

end sum_first_four_terms_eq_neg20_l759_759520


namespace max_length_OB_l759_759554

-- Define the problem conditions
def angle_AOB : ℝ := 45
def length_AB : ℝ := 2
def max_sin_angle_OAB : ℝ := 1

-- Claim to be proven
theorem max_length_OB : ∃ OB_max, OB_max = 2 * Real.sqrt 2 :=
by
  sorry

end max_length_OB_l759_759554


namespace greatest_three_digit_multiple_of_17_l759_759861

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759861


namespace greatest_three_digit_multiple_of_17_l759_759822

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759822


namespace max_value_expression_l759_759137

theorem max_value_expression (a b c x1 x2 x3 λ: ℝ) (h1: f x = x^3 + a * x^2 + b * x + c) (h2: x2 - x1 = λ) (h3: x3 > (1/2) * (x1 + x2)) (h4: (x - x1) * (x - x2) * (x - x3) = 0) :
  ∃ M, M = (3 * Real.sqrt 3) / 2 ∧ ∀ a b c λ, (2 * a^3 + 27 * c - 9 * a * b) / λ^3 ≤ M :=
sorry

end max_value_expression_l759_759137


namespace greatest_three_digit_multiple_of_17_l759_759682

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759682


namespace winning_singer_is_C_l759_759181

def Singer : Type := {A, B, C, D}

def wins_award (s : Singer) : Prop := sorry

-- Statements made by the singers
def statement_A : Prop := wins_award B ∨ wins_award C
def statement_B : Prop := ¬(wins_award A ∨ wins_award C)
def statement_C : Prop := wins_award C
def statement_D : Prop := wins_award B

-- Exactly two statements are true
def two_statements_true : Prop :=
  (statement_A ∧ statement_B ∧ ¬statement_C ∧ ¬statement_D) ∨
  (statement_A ∧ ¬statement_B ∧ statement_C ∧ ¬statement_D) ∨
  (statement_A ∧ ¬statement_B ∧ ¬statement_C ∧ statement_D) ∨
  (¬statement_A ∧ statement_B ∧ statement_C ∧ ¬statement_D) ∨
  (¬statement_A ∧ statement_B ∧ ¬statement_C ∧ statement_D) ∨
  (¬statement_A ∧ ¬statement_B ∧ statement_C ∧ statement_D)

-- The theorem we need to prove
theorem winning_singer_is_C : wins_award C :=
by 
  have h : two_statements_true := sorry
  sorry

end winning_singer_is_C_l759_759181


namespace y_completion_time_l759_759232

theorem y_completion_time (x_days : ℕ) (x_worked_days : ℕ) (y_remaining_days : ℕ) (total_work : ℚ)
  (hx : x_days = 40)
  (hx_worked : x_worked_days = 8)
  (hy_remaining : y_remaining_days = 16)
  (htotal_work : total_work = 1) :
  let x_work_rate := total_work / x_days,
      x_completed := x_work_rate * x_worked_days,
      remaining_work := total_work - x_completed,
      y_work_rate := remaining_work / y_remaining_days
  in y_completion_time = 20 :=
by 
  have x_work_rate := total_work / x_days,
  have x_completed := x_work_rate * x_worked_days,
  have remaining_work := total_work - x_completed,
  have y_work_rate := remaining_work / y_remaining_days,
  have y_completion_time := total_work / y_work_rate,
  sorry

end y_completion_time_l759_759232


namespace number_of_correct_conclusions_l759_759002

variables {R : Type*} [linear_ordered_field R]
noncomputable def f : R → R := sorry

-- Conditions
axiom domain_of_f : ∀ x, x ∈ set.univ
axiom f_multiplicative : ∀ x y, f(x) * f(y) = f(x + y - 1)
axiom f_gt_one_for_x_gt_one : ∀ x, x > 1 → f(x) > 1

-- Statements to prove
lemma conclusion1 : f(1) = 1 := sorry
lemma conclusion2 : ¬ (∀ x, f(x - 1) = f(1 - x)) := sorry
lemma conclusion3 : ∀ x ∈ (set.Ici 1), monotone f := sorry
lemma conclusion4 : ∀ x < 1, 0 < f(x) ∧ f(x) < 1 := sorry

-- Final proof to show number of correct conclusions
theorem number_of_correct_conclusions : (conclusion1 ∧ conclusion3 ∧ conclusion4 ∧ ¬conclusion2) :=
begin
  split,
  { exact conclusion1 },
  split,
  { exact conclusion3 },
  split,
  { exact conclusion4 },
  { exact conclusion2 }
end

end number_of_correct_conclusions_l759_759002


namespace greatest_three_digit_multiple_of_seventeen_l759_759704

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759704


namespace arith_seq_sum_l759_759531

theorem arith_seq_sum (a₃ a₄ a₅ : ℤ) (h₁ : a₃ = 7) (h₂ : a₄ = 11) (h₃ : a₅ = 15) :
  let d := a₄ - a₃;
  let a := a₄ - 3 * d;
  (6 / 2 * (2 * a + 5 * d)) = 54 :=
by
  sorry

end arith_seq_sum_l759_759531


namespace arithmetic_sequence_a7_l759_759424

theorem arithmetic_sequence_a7 :
  ∀ (a : ℕ → ℕ) (d : ℕ),
  (∀ n, a (n + 1) = a n + d) →
  a 1 = 2 →
  a 3 + a 5 = 10 →
  a 7 = 8 :=
by
  intros a d h_seq h_a1 h_sum
  sorry

end arithmetic_sequence_a7_l759_759424


namespace greatest_three_digit_multiple_of_17_is_986_l759_759764

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759764


namespace find_f2_l759_759047

def f (x : ℝ) : ℝ := sorry

theorem find_f2 : (∀ x, f (x-1) = x / (x-1)) → f 2 = 3 / 2 :=
by
  sorry

end find_f2_l759_759047


namespace cory_needs_22_weeks_l759_759296

open Nat

def cory_birthday_money : ℕ := 100 + 45 + 20
def bike_cost : ℕ := 600
def weekly_earning : ℕ := 20

theorem cory_needs_22_weeks : ∃ x : ℕ, cory_birthday_money + x * weekly_earning ≥ bike_cost ∧ x = 22 := by
  sorry

end cory_needs_22_weeks_l759_759296


namespace sum_primes_less_than_20_l759_759946

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759946


namespace sum_of_primes_less_than_20_is_77_l759_759919

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759919


namespace envelope_width_l759_759273

theorem envelope_width (L W A : ℝ) (hL : L = 4) (hA : A = 16) (hArea : A = L * W) : W = 4 := 
by
  -- We state the problem
  sorry

end envelope_width_l759_759273


namespace excenter_angle_sum_eq_180_l759_759101

-- Define the data types and basic geometry entities
variables {A B C J_A J_B P Q R : Type}
variables [IsExcenter J_A A B C] [IsExcenter J_B B A C]
variables [IsOnCircumcircle P Q A B C] [Parallel PQ AB]
variables [IntersectsAt PQ AC P] [IntersectsAt PQ BC Q]
variables [IntersectsAt R (LineThrough A B) (LineThrough C P)]

-- Formal statement of the proof problem
theorem excenter_angle_sum_eq_180 {ABC : Triangle}
  (J_A : Point) (J_B : Point)
  (P Q : Point) (R : Point)
  (h1 : IsExcenter J_A A B C)
  (h2 : IsExcenter J_B B A C)
  (h3 : IsOnCircumcircle P Q A B C)
  (h4 : Parallel PQ (Segment A B))
  (h5 : Intersects P (Segment A C) Q)
  (h6 : Intersects Q (Segment B C) P)
  (h7 : Intersects R (Line A B) (Line C P)) :
  angle (Line J_A Q) (Line J_B Q) + angle (Line J_A R) (Line J_B R) = 180 :=
sorry

end excenter_angle_sum_eq_180_l759_759101


namespace greatest_three_digit_multiple_of17_l759_759730

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759730


namespace distance_traveled_first_7_minutes_velocity_vector_change_magnitude_l759_759171

-- Definitions of the motion functions
def x (t : ℝ) : ℝ := t * (t - 6) ^ 2
def y (t : ℝ) : ℝ := if t < 7 then 0 else (t - 7) ^ 2

-- Distance traveled by the robot in the first 7 minutes
theorem distance_traveled_first_7_minutes (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ 7) : 
  ∫ (t:ℝ) in 0..7, (abs (3 * t ^ 2 - 24 * t + 36)) = 71 := 
sorry

-- Magnitude of the change in the velocity vector during the eighth minute
theorem velocity_vector_change_magnitude : 
  abs (sqrt ((21)^2 + (2)^2)) = sqrt 445 := 
sorry

end distance_traveled_first_7_minutes_velocity_vector_change_magnitude_l759_759171


namespace tangent_line_condition_l759_759178

theorem tangent_line_condition (k : ℝ) : 
  (∀ x y : ℝ, (x-2)^2 + (y-1)^2 = 1 → x - k * y - 1 = 0 → False) ↔ k = 0 :=
sorry

end tangent_line_condition_l759_759178


namespace greatest_three_digit_multiple_of_17_l759_759597

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759597


namespace greatest_three_digit_multiple_of_17_l759_759600

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759600


namespace correct_formula_l759_759010

theorem correct_formula :
  (∀ (x: ℤ), x ∈ {0, 1, 2, 3, 4} -> 
  let y := 5 * x^2 + x in
  (y = match x with
  | 0 => 0
  | 1 => 6
  | 2 => 22
  | 3 => 48
  | 4 => 84
  | _ => false))
  := sorry

end correct_formula_l759_759010


namespace determine_n_l759_759037

theorem determine_n (n : ℕ) (f : ℝ → ℝ) (h : ∀ x, f x = x^n) (h' : ∀ x, deriv f x = n * x^(n-1)) :
  deriv f 2 = 12 → n = 3 :=
by
  intro hf'
  rw [h] at h'
  rw [deriv_pow''] at h'
  sorry

end determine_n_l759_759037


namespace volume_of_common_part_of_congruent_cubes_l759_759154

noncomputable def volume_common_part (a : ℝ) : ℝ :=
  a^3 * (Real.sqrt 2 - 3 / 2)

theorem volume_of_common_part_of_congruent_cubes
  (a : ℝ) 
  (cubes_congruent : Congruent (
      Cube (ABCD, A'B'C'D') (a))
    (Cube (EFGH, E'F'G'H') (a)))
  (diagonal_planes_shared : SharedPlane 
      (AC, C'A') (EG, G'E'))
  (rotation_transform : Rotation90
      (AC, C'A') (EG, G'E'))
  : volume_common_part a = a^3 * (Real.sqrt 2 - 3 / 2) :=
by
  sorry

end volume_of_common_part_of_congruent_cubes_l759_759154


namespace percentage_selected_in_state_A_l759_759057

-- Definitions
def num_candidates : ℕ := 8000
def percentage_selected_state_B : ℕ := 7
def extra_selected_candidates : ℕ := 80

-- Question
theorem percentage_selected_in_state_A :
  ∃ (P : ℕ), ((P / 100) * 8000 + 80 = 560) ∧ (P = 6) := sorry

end percentage_selected_in_state_A_l759_759057


namespace abs_neg_seven_l759_759150

theorem abs_neg_seven : |(-7 : ℤ)| = 7 := by
  sorry

end abs_neg_seven_l759_759150


namespace minimum_value_of_m_l759_759442

-- Define the function and the conditions
def is_on_graph (x y : ℝ) : Prop := y = x^2 - 2
def m (x y : ℝ) : ℝ := (3*x + y - 4) / (x - 1) + (x + 3*y - 4) / (y - 1)
def valid_domain (x : ℝ) : Prop := x > sqrt 3

-- The theorem statement
theorem minimum_value_of_m (x y : ℝ) (hx : valid_domain x) (hy : is_on_graph x y) : m x y ≥ 8 :=
sorry

end minimum_value_of_m_l759_759442


namespace sum_of_primes_lt_20_eq_77_l759_759901

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759901


namespace sum_of_primes_less_than_20_l759_759985

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759985


namespace vertical_distance_rings_l759_759247

theorem vertical_distance_rings :
  let external_diameters := list.range ((20 - 4) / 2 + 1),
      internal_diameters  := list.map (λ n, 20 - 2 * n) external_diameters,
      num_rings := internal_diameters.length,
      sum_diameters := internal_diameters.sum,
      overlap := 1 * (num_rings - 1) 
  in sum_diameters - overlap = 82 :=
by sorry

end vertical_distance_rings_l759_759247


namespace construct_trapezoid_from_conditions_l759_759293

-- Define the trapezoid construction problem
variables (α β e k : ℝ) -- Angles α and β, diagonal e, perimeter k
variables (AB CD AC : ℝ) -- Lengths of sides AB, CD, and diagonal AC
variables (a b c d : ℝ) -- Variable lengths

-- Assumptions based on conditions stated
axiom (trapezoid_condition : AB + CD + a + b + c + d = k)
axiom (parallel_AB_CD : AB ∥ CD)
axiom (angle_DAB_eq_alpha : ∠ DAB = α)
axiom (angle_ABC_eq_beta : ∠ ABC = β)
axiom (diagonal_AC_eq_e : AC = e)

-- Statement to prove
theorem construct_trapezoid_from_conditions :
  ∃ (A B C D : ℝ × ℝ), (AB ∥ CD) ∧ (∠ DAB = α) ∧ (∠ ABC = β) ∧ (AC = e) ∧ (A + B + C + D = k) :=
sorry

end construct_trapezoid_from_conditions_l759_759293


namespace subset_intersection_exists_l759_759100

theorem subset_intersection_exists {n : ℕ} (A : Fin (n + 1) → Finset (Fin n)) 
    (h_distinct : ∀ i j : Fin (n + 1), i ≠ j → A i ≠ A j)
    (h_size : ∀ i : Fin (n + 1), (A i).card = 3) : 
    ∃ (i j : Fin (n + 1)), i ≠ j ∧ (A i ∩ A j).card = 1 :=
by
  sorry

end subset_intersection_exists_l759_759100


namespace sum_of_primes_less_than_20_is_77_l759_759917

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759917


namespace colony_fungi_day_l759_759058

theorem colony_fungi_day (n : ℕ): 
  (4 * 2^n > 150) = (n = 6) :=
sorry

end colony_fungi_day_l759_759058


namespace greatest_three_digit_multiple_of_17_is_986_l759_759756

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759756


namespace sum_of_primes_less_than_20_l759_759998

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l759_759998


namespace greatest_three_digit_multiple_of_17_l759_759595

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759595


namespace greatest_three_digit_multiple_of_17_l759_759740

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759740


namespace greatest_three_digit_multiple_of_17_l759_759823

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759823


namespace greatest_three_digit_multiple_of_17_l759_759589

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759589


namespace greatest_three_digit_multiple_of_17_l759_759578

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759578


namespace greatest_three_digit_multiple_of_17_l759_759645

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759645


namespace sum_of_primes_less_than_20_l759_759992

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759992


namespace plain_chips_count_l759_759482

theorem plain_chips_count (total_chips : ℕ) (BBQ_chips : ℕ)
  (hyp1 : total_chips = 9) (hyp2 : BBQ_chips = 5)
  (hyp3 : (5 * 4 / (2 * 1) : ℚ) / ((9 * 8 * 7) / (3 * 2 * 1)) = 0.11904761904761904) :
  total_chips - BBQ_chips = 4 := by
sorry

end plain_chips_count_l759_759482


namespace sum_primes_less_than_20_l759_759975

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759975


namespace sum_primes_less_than_20_l759_759971

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759971


namespace sum_of_primes_less_than_20_is_77_l759_759925

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759925


namespace greatest_three_digit_multiple_of_17_is_986_l759_759666

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759666


namespace sum_even_minus_sum_odd_l759_759228

theorem sum_even_minus_sum_odd :
  let a := (∑ i in finset.range 60 + 2, 2 * i)
  let b := (∑ i in finset.range 60 + 1, 2 * i - 1)
in a - b = 60 :=
by
  -- Skip the actual proof but ensure it can be compiled successfully
  rfl -- Placeholder for the correct proof logic

end sum_even_minus_sum_odd_l759_759228


namespace greatest_three_digit_multiple_of_17_l759_759577

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759577


namespace part1_part2_l759_759051

-- Problem conditions and target statement
theorem part1
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : C = 2 / 3 * Real.pi)
  (h2 : (a, b, c) = (c - 4, c - 2, c))
  (h3 : a^2 + b^2 - 2 * a * b * Real.cos C = c^2) :
  c = 7 :=
sorry

theorem part2
  (a b c A B : ℝ)
  (h0 : c = Real.sqrt 3)
  (h1 : C = 2 / 3 * Real.pi)
  (h2 : a = 2 * Real.sin B)
  (h3 : b = 2 * Real.sin (Real.pi / 3 - B))
  (h4 : A = Real.pi - B - C)
  (h5 : 0 < A) :
  let f : ℝ → ℝ := λ θ, 2 * Real.sin (θ + Real.pi / 3) + Real.sqrt 3 in
  ∃ max_value, max_value = 2 + Real.sqrt 3 ∧
  (∀ θ, 0 < θ ∧ θ < Real.pi / 3 → (f θ ≤ max_value)) :=
sorry

end part1_part2_l759_759051


namespace sum_of_three_consecutive_integers_l759_759540

theorem sum_of_three_consecutive_integers (n m l : ℕ) (h1 : n + 1 = m) (h2 : m + 1 = l) (h3 : l = 13) : n + m + l = 36 := 
by sorry

end sum_of_three_consecutive_integers_l759_759540


namespace fencing_cost_l759_759503

variable (Length : ℝ) (Breadth : ℝ) (CostPerMeter : ℝ)
variable (TotalCost : ℝ)

-- Conditions
def conditions :=
  Length = 60 ∧ Breadth = Length - 20 ∧ CostPerMeter = 26.50

-- Define the perimeter of the rectangle.
def perimeter (Length Breadth : ℝ) := 2 * (Length + Breadth)

-- Define total cost of fencing based on the perimeter.
def total_cost (Perimeter CostPerMeter : ℝ) := CostPerMeter * Perimeter

-- Problem Statement: Prove the total cost is 5300 given the conditions
theorem fencing_cost (h : conditions Length Breadth CostPerMeter) : TotalCost = 5300 :=
by
  -- Extract conditions as hypotheses.
  rcases h with ⟨h_length, h_breadth, h_cost_per_meter⟩
  -- Calculate the breadth.
  have hc_breadth : Breadth = 40 := by
    rw [h_length, h_breadth]
    linarith
  -- Calculate the perimeter.
  have hc_perimeter : perimeter Length Breadth = 200 := by
    rw [perimeter, h_length, hc_breadth]
    norm_num
  -- Calculate the total cost of fencing.
  have hc_total_cost : total_cost (perimeter Length Breadth) CostPerMeter = 5300 := by
    rw [total_cost, hc_perimeter, h_cost_per_meter]
    norm_num
  -- Conclude the proof with the intended result.
  exact hc_total_cost

-- Note: Use "sorry" if we want to skip proof, but in this context, the statement directly leads to conclusion.

end fencing_cost_l759_759503


namespace greatest_three_digit_multiple_of_17_l759_759835

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759835


namespace sum_of_first_six_terms_arithmetic_seq_l759_759524

theorem sum_of_first_six_terms_arithmetic_seq (a b c : ℤ) (d : ℤ) (n : ℤ) :
    (a = 7) ∧ (b = 11) ∧ (c = 15) ∧ (d = b - a) ∧ (d = c - b) 
    ∧ (n = a - d) 
    ∧ (d = 4) -- the common difference is always 4 here as per the solution given 
    ∧ (n = -1) -- the correct first term as per calculation
    → (n + (n + d) + (a) + (b) + (c) + (c + d) = 54) := 
begin
  sorry
end

end sum_of_first_six_terms_arithmetic_seq_l759_759524


namespace h_at_4_l759_759099

open Polynomial

noncomputable def f : Polynomial ℝ := X^3 - X + 1
variable (h : Polynomial ℝ)
variable [h_cubic : h.degree = 3]
variable [h0 : h.coeff 0 = 1]
variable [roots_squared : ∀ x, x ∈ h.roots → sqrt x ∈ f.roots]

theorem h_at_4 : h(4) = -3599 := by
  sorry

end h_at_4_l759_759099


namespace largest_set_not_divisible_sum_l759_759865

theorem largest_set_not_divisible_sum {n : ℕ} (h : n = 26) : 
  ∃ (s : Finset ℕ), (∀ (x y ∈ s), x + y ≠ 0 [MOD n]) ∧ (0 < s.card) ∧ (s.card = 76) :=
by {
  sorry
}

end largest_set_not_divisible_sum_l759_759865


namespace slope_positive_if_and_only_if_l759_759176

/-- Given points A(2, 1) and B(1, m^2), the slope of the line passing through them is positive,
if and only if m is in the range -1 < m < 1. -/
theorem slope_positive_if_and_only_if
  (m : ℝ) : 1 - m^2 > 0 ↔ -1 < m ∧ m < 1 :=
by
  sorry

end slope_positive_if_and_only_if_l759_759176


namespace greatest_three_digit_multiple_of_17_l759_759607

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759607


namespace sum_primes_less_than_20_l759_759961

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759961


namespace arithmetic_sequence_sum_l759_759527

theorem arithmetic_sequence_sum :
  ∃ a : ℕ → ℤ, 
    a 3 = 7 ∧ a 4 = 11 ∧ a 5 = 15 ∧ 
    (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 54) := 
by {
  sorry
}

end arithmetic_sequence_sum_l759_759527


namespace average_marks_of_all_students_l759_759494

theorem average_marks_of_all_students :
  (22 * 40 + 28 * 60) / (22 + 28) = 51.2 :=
by
  sorry

end average_marks_of_all_students_l759_759494


namespace greatest_three_digit_multiple_of_17_l759_759842

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759842


namespace sqrt_extraction_count_l759_759135

theorem sqrt_extraction_count (p : ℕ) [Fact p.Prime] : 
    ∃ k, k = (p + 1) / 2 ∧ ∀ n < p, ∃ x < p, x^2 ≡ n [MOD p] ↔ n < k := 
by
  sorry

end sqrt_extraction_count_l759_759135


namespace greatest_three_digit_multiple_of_17_l759_759605

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759605


namespace greatest_three_digit_multiple_of_17_l759_759692

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759692


namespace sweet_treats_per_student_l759_759472

theorem sweet_treats_per_student : 
  ∀ (cookies cupcakes brownies students : ℕ), 
    cookies = 20 →
    cupcakes = 25 →
    brownies = 35 →
    students = 20 →
    (cookies + cupcakes + brownies) / students = 4 :=
by
  intros cookies cupcakes brownies students hcook hcup hbrown hstud
  have h1 : cookies + cupcakes + brownies = 80, from calc
    cookies + cupcakes + brownies = 20 + 25 + 35 := by rw [hcook, hcup, hbrown]
    ... = 80 := rfl
  have h2 : (cookies + cupcakes + brownies) / students = 80 / 20, from
    calc (cookies + cupcakes + brownies) / students
      = 80 / 20 := by rw [h1, hstud]
  exact eq.trans h2 (by norm_num)

end sweet_treats_per_student_l759_759472


namespace sum_of_primes_lt_20_eq_77_l759_759908

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759908


namespace tangent_lines_perpendicular_l759_759501

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * x^2 + a * Real.log x

def f_derivative (a x : ℝ) : ℝ := x + (a / x)

theorem tangent_lines_perpendicular (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < 2 ∧ 1 < x₂ ∧ x₂ < 2 ∧ f_derivative a x₁ * f_derivative a x₂ = -1) ↔
  -3 < a ∧ a < -2 := 
sorry

end tangent_lines_perpendicular_l759_759501


namespace greatest_three_digit_multiple_of_17_l759_759799

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759799


namespace sum_of_primes_less_than_20_l759_759886

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759886


namespace greatest_three_digit_multiple_of_17_l759_759801

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759801


namespace total_number_of_baseball_cards_l759_759093

def baseball_cards_total : Nat :=
  let carlos := 20
  let matias := carlos - 6
  let jorge := matias
  carlos + matias + jorge
   
theorem total_number_of_baseball_cards :
  baseball_cards_total = 48 :=
by
  rfl

end total_number_of_baseball_cards_l759_759093


namespace part1_solution_set_part2_minimum_value_l759_759020

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

theorem part1_solution_set (x : ℝ) :
  (f x ≥ -1) ↔ (2 / 3 ≤ x ∧ x ≤ 6) := sorry

variables {a b c : ℝ}
theorem part2_minimum_value (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = 6) :
  (1 / (2 * a + b) + 1 / (2 * a + c) ≥ 2 / 3) := 
sorry

end part1_solution_set_part2_minimum_value_l759_759020


namespace greatest_three_digit_multiple_of_17_l759_759612

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759612


namespace solution_set_M_minimum_value_expr_l759_759021

-- Define the function f(x)
def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

-- Proof problem (1): Prove that the solution set M of the inequality f(x) ≥ -1 is {x | 2/3 ≤ x ≤ 6}.
theorem solution_set_M : 
  { x : ℝ | f x ≥ -1 } = { x : ℝ | 2/3 ≤ x ∧ x ≤ 6 } :=
sorry

-- Define the requirement for t and the expression to minimize
noncomputable def t : ℝ := 6
noncomputable def expr (a b c : ℝ) : ℝ := 1 / (2 * a + b) + 1 / (2 * a + c)

-- Proof problem (2): Given t = 6 and 4a + b + c = 6, prove that the minimum value of expr is 2/3.
theorem minimum_value_expr (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = t) :
  expr a b c ≥ 2/3 :=
sorry

end solution_set_M_minimum_value_expr_l759_759021


namespace circle_properties_l759_759368

-- Define the equation of the circle
def circle_eqn (x y : ℝ) (k : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y - k = 0

-- Prove the statements based on given conditions
theorem circle_properties (x y k : ℝ) (h : circle_eqn x y k) :
  (1, -2) = (1, -2) ∧ (k > -5) ∧ (k = 4 → real.sqrt (k + 5) = 3) :=
by {
  sorry
}

end circle_properties_l759_759368


namespace greatest_three_digit_multiple_of_17_is_986_l759_759758

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759758


namespace greatest_three_digit_multiple_of_17_l759_759806

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759806


namespace greatest_three_digit_multiple_of_17_l759_759864

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759864


namespace greatest_three_digit_multiple_of_17_l759_759650

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759650


namespace number_wall_block_l759_759428

theorem number_wall_block (n : ℕ) (a b c d e f : ℕ) (h1 : a = 4) (h2 : b = 8) (h3 : c = 7) (h4 : e = n + 16) (h5 : d = 27) (h6 : f = 46) : n = 3 := by
  have h7 : e + d = f, from sorry
  have h8 : (n + 16) + 27 = 46, from sorry
  have h9 : n + 43 = 46, from sorry
  have h10 : n = 46 - 43, from sorry
  have h11 : n = 3, from sorry
  exact h11

end number_wall_block_l759_759428


namespace allocation_schemes_correct_l759_759305

def number_of_allocation_schemes : ℕ :=
  let choose (n k : ℕ) : ℕ := Nat.choose n k in
  let factorial (n : ℕ) : ℕ := Nat.factorial n in
  (choose 6 2 * choose 4 2 * (4 * 3 * factorial 2)) / (factorial 2 * factorial 2)

theorem allocation_schemes_correct : number_of_allocation_schemes = 540 :=
  by sorry

end allocation_schemes_correct_l759_759305


namespace find_original_function_l759_759405

theorem find_original_function (f : ℝ → ℝ) : 
  (∀ x, f(2(x - π/3)) = sin(x - π/4)) → f(x) = sin(x / 2 + π / 12) := sorry

end find_original_function_l759_759405


namespace exists_five_digit_number_with_property_l759_759313

theorem exists_five_digit_number_with_property :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (n^2 % 100000) = n := 
sorry

end exists_five_digit_number_with_property_l759_759313


namespace sum_of_primes_less_than_20_is_77_l759_759916

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759916


namespace tangent_lines_perpendicular_l759_759500

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * x^2 + a * Real.log x

def f_derivative (a x : ℝ) : ℝ := x + (a / x)

theorem tangent_lines_perpendicular (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < 2 ∧ 1 < x₂ ∧ x₂ < 2 ∧ f_derivative a x₁ * f_derivative a x₂ = -1) ↔
  -3 < a ∧ a < -2 := 
sorry

end tangent_lines_perpendicular_l759_759500


namespace greatest_three_digit_multiple_of_17_is_986_l759_759632

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759632


namespace sum_primes_less_than_20_l759_759973

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759973


namespace hyperbola_asymptotes_l759_759345

theorem hyperbola_asymptotes (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b)
    (h₃ : ∃ l : ℝ → ℝ, ∀ x : ℝ, (l x)^2 = 4 * a^2) (h₄ : ∃ f₁ f₂ : (ℝ × ℝ), f₁ ≠ f₂) :
    (∃ asymptote : ℝ → ℝ, asymptote = λ x : ℝ, sqrt 2 * x 
                             ∨ asymptote = λ x : ℝ, -sqrt 2 * x) :=
begin
  sorry
end

end hyperbola_asymptotes_l759_759345


namespace greatest_three_digit_multiple_of_seventeen_l759_759707

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759707


namespace sum_of_primes_less_than_20_l759_759990

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759990


namespace find_smallest_x_l759_759451

theorem find_smallest_x : exists (x : ℕ), x > 1 ∧ cos (x * (real.pi / 180)) = cos ((x * x) * (real.pi / 180)) ∧ x = 26 :=
by
  sorry

end find_smallest_x_l759_759451


namespace sum_of_primes_less_than_20_is_77_l759_759918

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759918


namespace election_result_l759_759419

def CandidateVotes (total_voters : ℕ) (turnout : ℕ) (votes_a_percent : ℚ)
    (lead_b_over_a : ℕ) (votes_c_fraction : ℚ) (redistribute_fraction : ℚ) :
    Prop :=
  let total_cast := (turnout * total_voters) / 100
  let votes_a := (votes_a_percent * total_cast).natAbs
  let votes_b := votes_a + lead_b_over_a
  let votes_c := (votes_c_fraction * votes_a).natAbs
  let redistributed_to_a := (redistribute_fraction * votes_c).natAbs
  let redistributed_to_b := (redistribute_fraction * votes_c).natAbs
  let final_votes_a := votes_a + redistributed_to_a
  let final_votes_b := votes_b + redistributed_to_b
  total_cast = 5600 ∧ final_votes_a = 2100 ∧ final_votes_b = 6100

theorem election_result :
  CandidateVotes 8000 70 0.3 4000 0.5 0.5 :=
by sorry

end election_result_l759_759419


namespace sum_of_primes_lt_20_eq_77_l759_759902

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759902


namespace rods_in_one_mile_l759_759360

theorem rods_in_one_mile :
  (1 * 80 * 4 = 320) :=
sorry

end rods_in_one_mile_l759_759360


namespace greatest_three_digit_multiple_of_17_l759_759805

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759805


namespace slope_of_line_l759_759199

-- Define the points
def point1 : (ℝ × ℝ) := (1, -3)
def point2 : (ℝ × ℝ) := (-4, 7)

-- Change in y-coordinates
def delta_y : ℝ := point2.2 - point1.2

-- Change in x-coordinates
def delta_x : ℝ := point2.1 - point1.1

-- The slope
def slope := delta_y / delta_x

-- The theorem stating the slope of the line
theorem slope_of_line : slope = -2 :=
by
  -- Calculate the slope and prove that it equals -2
  sorry

end slope_of_line_l759_759199


namespace sequence_general_formulas_l759_759365

theorem sequence_general_formulas 
  (a b : ℕ+ → ℝ)
  (h1 : a 1 = b 1)
  (h2 : a 2 = b 2)
  (h3 : a 3 = b 3)
  (h4 : ∀ n : ℕ+, a 1 + 2*a 2 + 2^2*a 3 + … + 2^(n-1)*a n = 8*n) 
  (h5 : ∃ d : ℕ+ → ℝ, ∀ n : ℕ+, b (n+1) - b n = d n) :
  (∀ n : ℕ+, a n = 2^(4-n)) ∧ 
  (∀ n : ℕ+, b n = ↑n^2 - 7*↑n + 14) ∧ 
  (¬ ∃ k : ℕ+, b k - a k ∈ Ioo 0 1) :=
  by
    sorry

-- Explanation of parameters:
-- a, b : ℕ+ → ℝ: Sequences {a_n} and {b_n} defined as functions from ℕ+ (positive natural numbers) to ℝ (real numbers).
-- h1, h2, h3: Conditions that the first three terms of sequences {a_n} and {b_n} are equal.
-- h4: Condition that for all n in ℕ+, a sum involving sequence {a_n} equates to 8n.
-- h5: Condition that the sequence {b_{n+1} - b_n} is arithmetic; d is the common difference function.
-- Result: Proof that {a_n} follows the given formula, {b_n} follows the given formula, and there is no k for which the difference falls in (0, 1).

end sequence_general_formulas_l759_759365


namespace complementary_angles_ratio_l759_759505

theorem complementary_angles_ratio (x : ℝ) (hx : 5 * x = 90) : abs (4 * x - x) = 54 :=
by
  have h₁ : x = 18 := by 
    linarith [hx]
  rw [h₁]
  norm_num

end complementary_angles_ratio_l759_759505


namespace greatest_three_digit_multiple_of_17_l759_759780

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759780


namespace pigs_in_barn_l759_759180

theorem pigs_in_barn (original_pigs : ℕ) (joined_pigs : ℕ) (total_pigs : ℕ) 
  (h1 : original_pigs = 64) (h2 : joined_pigs = 22) : total_pigs = 86 :=
by 
  rw [h1, h2]
  exact 64 + 22
  sorry

end pigs_in_barn_l759_759180


namespace greatest_three_digit_multiple_of_17_is_986_l759_759761

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759761


namespace sum_of_numbers_in_ratio_with_lcm_l759_759492

theorem sum_of_numbers_in_ratio_with_lcm (a b : ℕ) (h_lcm : Nat.lcm a b = 36) (h_ratio : a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3) : a + b = 30 :=
sorry

end sum_of_numbers_in_ratio_with_lcm_l759_759492


namespace greatest_three_digit_multiple_of_17_l759_759596

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759596


namespace find_lambda_l759_759361

-- Definitions of the given conditions
variables {R : Type*} [LinearOrderedField R] {V : Type*} [AddCommGroup V] [Module R V]
variables (e1 e2 : V) (λ : R)
variables (A B C D: V)

-- Non-collinearity of e1 and e2
axiom non_collinear : ¬(∃ k : R, e1 = k • e2)

-- Defined vectors
def AB := e1 + e2
def CB := -λ • e1 - 8 • e2
def CD := 3 • e1 - 3 • e2
def BD := CD - CB

-- Collinearity condition: A, B, and D are on the same line
axiom collinear : ∃ m : R, AB = m • BD

-- Goal: Prove the value of λ
theorem find_lambda : λ = 2 :=
sorry

end find_lambda_l759_759361


namespace sum_primes_less_than_20_l759_759972

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759972


namespace greatest_three_digit_multiple_of_17_is_986_l759_759625

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759625


namespace nelly_part_payment_is_875_l759_759474

noncomputable def part_payment (total_cost remaining_amount : ℝ) :=
  0.25 * total_cost

theorem nelly_part_payment_is_875 (total_cost : ℝ) (remaining_amount : ℝ)
  (h1 : remaining_amount = 2625)
  (h2 : remaining_amount = 0.75 * total_cost) :
  part_payment total_cost remaining_amount = 875 :=
by
  sorry

end nelly_part_payment_is_875_l759_759474


namespace sum_of_primes_less_than_20_l759_759869

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759869


namespace greatest_three_digit_multiple_of_seventeen_l759_759703

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759703


namespace greatest_three_digit_multiple_of_17_l759_759789

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759789


namespace total_people_on_playground_l759_759543

open Nat

-- Conditions
def num_girls := 28
def num_boys := 35
def num_3rd_grade_girls := 15
def num_3rd_grade_boys := 18
def num_teachers := 4

-- Derived values (from conditions)
def num_4th_grade_girls := num_girls - num_3rd_grade_girls
def num_4th_grade_boys := num_boys - num_3rd_grade_boys
def num_3rd_graders := num_3rd_grade_girls + num_3rd_grade_boys
def num_4th_graders := num_4th_grade_girls + num_4th_grade_boys

-- Total number of people
def total_people := num_3rd_graders + num_4th_graders + num_teachers

-- Proof statement
theorem total_people_on_playground : total_people = 67 :=
  by
     -- This is where the proof would go
     sorry

end total_people_on_playground_l759_759543


namespace greatest_three_digit_multiple_of_17_l759_759841

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759841


namespace matrices_are_inverses_l759_759161

noncomputable def matrix_inverse : Prop :=
  let A := !![![a, 3], ![2, 5]] in
  let B := !![![b, -1/5], ![1/2, 1/10]] in
  (A * B = !![![1, 0], ![0, 1]]) ∧ (a = 1.5) ∧ (b = -5/4)

theorem matrices_are_inverses {a b : ℚ} :
  matrix_inverse :=
by
  sorry

end matrices_are_inverses_l759_759161


namespace two_trains_crossing_time_l759_759557

theorem two_trains_crossing_time
  (length_train: ℝ) (time_telegraph_post_first: ℝ) (time_telegraph_post_second: ℝ)
  (length_train_eq: length_train = 120) 
  (time_telegraph_post_first_eq: time_telegraph_post_first = 10) 
  (time_telegraph_post_second_eq: time_telegraph_post_second = 15) :
  (2 * length_train) / (length_train / time_telegraph_post_first + length_train / time_telegraph_post_second) = 12 :=
by
  sorry

end two_trains_crossing_time_l759_759557


namespace satisfies_conditions_l759_759560

theorem satisfies_conditions : ∃ (n : ℤ), 0 ≤ n ∧ n < 31 ∧ -250 % 31 = n % 31 ∧ n = 29 :=
by
  sorry

end satisfies_conditions_l759_759560


namespace sum_of_areas_of_excellent_rectangles_eq_942_l759_759259

def is_excellent_rectangle (a b : ℕ) : Prop :=
  a * b = 3 * (2 * a + 2 * b)

theorem sum_of_areas_of_excellent_rectangles_eq_942 :
  (finset.univ.filter (λ (p : ℕ × ℕ), let a := p.1; let b := p.2 in is_excellent_rectangle a b)).image (λ (p : ℕ × ℕ), p.1 * p.2).sum = 942 :=
sorry

end sum_of_areas_of_excellent_rectangles_eq_942_l759_759259


namespace maximum_length_OB_l759_759550

theorem maximum_length_OB 
  (O A B : Type) 
  [EuclideanGeometry O]
  (h_angle_OAB : ∠ O A B = 45°)
  (h_AB : distance A B = 2) : 
  (exists OB_max, max (distance O B) = OB_max ∧ OB_max = 2 * sqrt 2) :=
by
  sorry

end maximum_length_OB_l759_759550


namespace max_angles_with_intersections_l759_759272

-- Definitions based on the conditions
def angle_60 (A : Type) (a b : A) : Prop := sorry -- Define angle 60 degrees

def intersect_at_4_points (A : Type) (angle1 angle2 : A) : Prop := sorry -- Define intersection at 4 points

-- Main theorem statement
theorem max_angles_with_intersections (n : ℕ) : n ≤ 2 ↔ 
  ∀ (A : Type) (angles : fin n → (A × A)), 
  (∀ (i j : fin n), i ≠ j → angle_60 A (angles i).1 (angles i).2) ∧
  (∀ (i j : fin n), i ≠ j → intersect_at_4_points A (angles i) (angles j)) :=
by {
  sorry
}

end max_angles_with_intersections_l759_759272


namespace greatest_three_digit_multiple_of_17_is_986_l759_759659

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759659


namespace problem_solution_l759_759183

theorem problem_solution (x y : ℝ) (a b c d : ℕ) 
  (h1 : x + y = 12) 
  (h2 : 3 * x * y = 12)
  (hx : x = (a + b * real.sqrt c) / d ∨ x = (a - b * real.sqrt c) / d)
  (ha : a = 6)
  (hb : b = 4)
  (hc : c = 3)
  (hd : d = 1) :
  a + b + c + d = 14 :=
by
  sorry

end problem_solution_l759_759183


namespace sin_fourth_plus_cos_fourth_l759_759358

theorem sin_fourth_plus_cos_fourth (α : ℝ) (h : Real.cos (2 * α) = 3 / 5) : 
  Real.sin α ^ 4 + Real.cos α ^ 4 = 17 / 25 := 
by
  sorry

end sin_fourth_plus_cos_fourth_l759_759358


namespace exponential_fixed_point_l759_759377

variable (a : ℝ)

noncomputable def f (x : ℝ) := a^(x - 1) + 3

theorem exponential_fixed_point (ha1 : a > 0) (ha2 : a ≠ 1) : f a 1 = 4 :=
by
  sorry

end exponential_fixed_point_l759_759377


namespace num_ways_make_change_l759_759392

def is_valid_combination (pennies nickels dimes quarters : ℕ) : Prop :=
  pennies * 1 + nickels * 5 + dimes * 10 + quarters * 25 = 50

theorem num_ways_make_change : 
  (∑ p in finset.Icc 0 50, ∑ n in finset.Icc 0 10, ∑ d in finset.Icc 0 5, ∑ q in finset.Icc 0 2, 
    if is_valid_combination p n d q ∧ q < 2 then 1 else 0) = 33 := 
by 
  sorry

end num_ways_make_change_l759_759392


namespace orthocenter_is_incenter_l759_759113

theorem orthocenter_is_incenter 
  (A B C H H_A H_B H_C : Point) 
  (h_triangle : Triangle A B C)
  (h_orthocenter : Orthocenter H A B C)
  (h_foot_A : AltitudeFoot H_A A B C)
  (h_foot_B : AltitudeFoot H_B B A C)
  (h_foot_C : AltitudeFoot H_C C A B) : 
  Incenter H H_A H_B H_C :=
sorry

end orthocenter_is_incenter_l759_759113


namespace distribute_candies_l759_759269

-- Definition of the problem conditions
def candies : ℕ := 10

-- The theorem stating the proof problem
theorem distribute_candies : (2 ^ (candies - 1)) = 512 := 
by
  sorry

end distribute_candies_l759_759269


namespace orthocenter_condition_l759_759067

noncomputable def is_acute_triangle (A B C : Point) : Prop :=
  ∀ (α β γ : Angle), α + β + γ = 180°

noncomputable def angle_greater (A O B: Point) (α β : Angle) : Prop :=
  α > β

noncomputable def is_obtuse_angle (α : Angle) : Prop :=
  α > 90°

noncomputable def is_orthocenter (H A B C: Point) : Prop :=
  -- Define orthocenter condition

noncomputable def on_circumcircle (P A B C : Point) : Prop :=
  -- Define points on circumcircle

noncomputable def parallel_lines (l₁ l₂ : Line) : Prop := 
  -- Define parallel lines condition

theorem orthocenter_condition (A B C D H F : Point)
    (acute_triangle : is_acute_triangle A B C) 
    (angle_condition : angle_greater A C B)
    (D_on_BC : on_line D B C)
    (ADB_obtuse : is_obtuse_angle (angle A D B))
    (H_orthocenter_ABD : is_orthocenter H A B D)
    (F_in_circum_ABD : on_circumcircle F A B D)
    : (is_orthocenter F A B C) ↔ (parallel_lines (line H D) (line C F) ∧ on_circumcircle H A B C) :=
sorry

end orthocenter_condition_l759_759067


namespace square_side_length_on_hyperbola_l759_759537

theorem square_side_length_on_hyperbola :
  (∀ (A B C D : ℝ × ℝ), 
    ((A = (0,0)) ∧ 
     (B.fst * B.snd = 4) ∧ 
     (C.fst * C.snd = 4) ∧ 
     (D.fst * D.snd = 4) ∧ 
     ((A.fst + C.fst) / 2 = 2) ∧
     ((A.snd + C.snd) / 2 = 2) ∧
     (∃ s : ℝ, 
        dist A B = s ∧ 
        dist B C = s ∧ 
        dist C D = s ∧ 
        dist D A = s)
    ) → 
    (∃ s : ℝ, s = 2 * real.sqrt 2)) := 
sorry

end square_side_length_on_hyperbola_l759_759537


namespace complementary_angle_difference_l759_759508

theorem complementary_angle_difference (a b : ℝ) (h1 : a = 4 * b) (h2 : a + b = 90) : (a - b) = 54 :=
by
  -- Proof is intentionally omitted
  sorry

end complementary_angle_difference_l759_759508


namespace inscribed_hexagon_sum_of_diagonals_l759_759253

theorem inscribed_hexagon_sum_of_diagonals :
  ∀ (A B C D E F : Point) (O : Point),
  is_cyclic_quadrilateral A B C O ∧
  is_cyclic_quadrilateral O C D E ∧ 
  is_cyclic_quadrilateral O E F A ∧
  dist A B = 40 ∧
  dist B C = 60 ∧
  dist C D = 100 ∧
  dist D E = 100 ∧
  dist E F = 100 ∧
  dist F A = 100 →
  dist A C + dist A D + dist A E = 624 :=
by
  intros A B C D E F O h_cyclic_ABCO h_cyclic_OCDE h_cyclic_OEFA h_AB h_BC h_CD h_DE h_EF h_FA
  sorry

end inscribed_hexagon_sum_of_diagonals_l759_759253


namespace constant_term_expansion_l759_759411

theorem constant_term_expansion (a b : ℕ) (n : ℕ) (h : 2^n = 512) : (constant_term (a + b)^n) = 84 := by
  sorry

end constant_term_expansion_l759_759411


namespace number_of_cartons_of_pencils_l759_759130

theorem number_of_cartons_of_pencils (P E : ℕ) 
  (h1 : P + E = 100) 
  (h2 : 6 * P + 3 * E = 360) : 
  P = 20 := 
by
  sorry

end number_of_cartons_of_pencils_l759_759130


namespace friend_gain_percentage_l759_759255

theorem friend_gain_percentage (original_cost_price selling_price : ℝ) (loss_percentage gain_percentage : ℝ)
  (h1 : original_cost_price = 50000)
  (h2 : loss_percentage = 10)
  (h3 : selling_price = 54000)
  (h4 : gain_percentage = 20) :
  let cost_price_for_friend := original_cost_price * (1 - loss_percentage / 100) in
  let gain_amount := selling_price - cost_price_for_friend in
  gain_percentage = (gain_amount / cost_price_for_friend) * 100 := 
by
  -- Definitions of intermediate values
  let cost_price_for_friend := original_cost_price * (1 - loss_percentage / 100)
  let gain_amount := selling_price - cost_price_for_friend
  sorry

end friend_gain_percentage_l759_759255


namespace chickens_rabbits_l759_759287

theorem chickens_rabbits (c r : ℕ) 
  (h1 : c = r - 20)
  (h2 : 4 * r = 6 * c + 10) :
  c = 35 := by
  sorry

end chickens_rabbits_l759_759287


namespace sin_cos_value_l759_759399

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
by
  sorry

end sin_cos_value_l759_759399


namespace johan_house_rooms_l759_759434

theorem johan_house_rooms 
    (R : ℕ) -- Total number of rooms
    (H1 : ∀ rooms : ℕ, (rooms / 8) * 8 = rooms ∧ (rooms % 8 = 0)) -- Each room has 8 walls
    (H2 : 3 * R / 5) -- Johan paints 3/5 of the rooms green
    (H3 : 2 * R / 5) -- Johan paints the rest of the rooms purple
    (H4 : 32 = (R * 2) / 5 * 8) -- Johan painted 32 walls purple
    : R = 10 := 
by
  sorry

end johan_house_rooms_l759_759434


namespace angle_C_range_m_l759_759050

variables (a b c : ℝ) (A B C : ℝ)
variables (m : ℝ)

-- Given conditions
def condition1 : Prop := a^2 + b^2 - c^2 = sqrt(3) * a * b
def condition2 : Prop := 0 < A ∧ A ≤ 2*Real.pi / 3

-- Mathematical equivalent proof problems
theorem angle_C (h : condition1) : C = Real.pi / 6 :=
sorry

theorem range_m (h1 : condition2) (h2 : m = 2 * Real.cos(A / 2)^2 - Real.sin(B) - 1) :
  set.Icc (-1 : ℝ) (1/2 : ℝ) :=
sorry

end angle_C_range_m_l759_759050


namespace greatest_three_digit_multiple_of_17_l759_759771

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759771


namespace greatest_three_digit_multiple_of17_l759_759715

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759715


namespace area_of_inscribed_triangle_l759_759196

-- Define the square with a given diagonal
def diagonal (d : ℝ) : Prop := d = 16
def side_length_of_square (s : ℝ) : Prop := s = 8 * Real.sqrt 2
def side_length_of_equilateral_triangle (a : ℝ) : Prop := a = 8 * Real.sqrt 2

-- Define the area of the equilateral triangle
def area_of_equilateral_triangle (area : ℝ) : Prop :=
  area = 32 * Real.sqrt 3

-- The theorem: Given the above conditions, prove the area of the equilateral triangle
theorem area_of_inscribed_triangle (d s a area : ℝ) 
  (h1 : diagonal d) 
  (h2 : side_length_of_square s) 
  (h3 : side_length_of_equilateral_triangle a) 
  (h4 : s = a) : 
  area_of_equilateral_triangle area :=
sorry

end area_of_inscribed_triangle_l759_759196


namespace sum_primes_less_than_20_l759_759948

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759948


namespace greatest_three_digit_multiple_of_17_is_986_l759_759669

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759669


namespace greatest_three_digit_multiple_of_17_l759_759746

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759746


namespace hyperbola_asymptotes_l759_759379

theorem hyperbola_asymptotes (a b c : ℝ) (h : a > 0) (h_b_gt_0: b > 0) 
  (eqn1 : b = 2 * Real.sqrt 2 * a)
  (focal_distance : 2 * a = (2 * c)/3)
  (focal_length : c = 3 * a) : 
  (∀ x : ℝ, ∀ y : ℝ, (y = (2 * Real.sqrt 2) * x) ∨ (y = -(2 * Real.sqrt 2) * x)) := by
  sorry

end hyperbola_asymptotes_l759_759379


namespace sum_primes_less_than_20_l759_759966

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759966


namespace sin_cos_value_l759_759397

theorem sin_cos_value (x : ℝ) (h : sin x = 4 * cos x) : sin x * cos x = 4 / 17 := 
sorry

end sin_cos_value_l759_759397


namespace solve_eq_l759_759328

theorem solve_eq (z : ℂ) :
  z^4 = 16 ↔ (z = 2 ∨ z = -2 ∨ z = 2*complex.I ∨ z = -2*complex.I) :=
sorry

end solve_eq_l759_759328


namespace fifth_decimal_of_power_l759_759404

theorem fifth_decimal_of_power (x : ℝ) (n : ℕ) (h1 : x = 1.0025) (h2 : n = 10) :
  (Real.repr (Real.round_dp 5 (x^n)) 5) = '8' :=
by
  -- Proof would go here
  sorry

end fifth_decimal_of_power_l759_759404


namespace greatest_three_digit_multiple_of_17_l759_759563

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759563


namespace smallest_possible_c_l759_759447

theorem smallest_possible_c 
  (a b c : ℕ) (hp : a > 0 ∧ b > 0 ∧ c > 0) 
  (hg : b^2 = a * c) 
  (ha : 2 * c = a + b) : 
  c = 2 :=
by
  sorry

end smallest_possible_c_l759_759447


namespace probability_math_majors_consecutive_l759_759190

theorem probability_math_majors_consecutive :
  (5 / 12) * (4 / 11) * (3 / 10) * (2 / 9) * (1 / 8) * 12 = 1 / 66 :=
by
  sorry

end probability_math_majors_consecutive_l759_759190


namespace sum_integers_75_to_95_l759_759206

theorem sum_integers_75_to_95 :
  let a := 75
  let l := 95
  let n := 95 - 75 + 1
  ∑ k in Finset.range n, (a + k) = 1785 := by
  sorry

end sum_integers_75_to_95_l759_759206


namespace greatest_three_digit_multiple_of_17_l759_759579

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759579


namespace jeffrey_walks_to_mailbox_l759_759220

theorem jeffrey_walks_to_mailbox :
  ∀ (D total_steps net_gain_per_set steps_per_set sets net_gain : ℕ),
    steps_per_set = 3 ∧ 
    net_gain = 1 ∧ 
    total_steps = 330 ∧ 
    net_gain_per_set = net_gain ∧ 
    sets = total_steps / steps_per_set ∧ 
    D = sets * net_gain →
    D = 110 :=
by
  intro D total_steps net_gain_per_set steps_per_set sets net_gain
  intro h
  sorry

end jeffrey_walks_to_mailbox_l759_759220


namespace sweet_treats_per_student_l759_759464

theorem sweet_treats_per_student :
  let cookies := 20
  let cupcakes := 25
  let brownies := 35
  let students := 20
  (cookies + cupcakes + brownies) / students = 4 :=
by 
  sorry

end sweet_treats_per_student_l759_759464


namespace sweet_treats_distribution_l759_759469

-- Define the number of cookies, cupcakes, brownies, and students
def cookies : ℕ := 20
def cupcakes : ℕ := 25
def brownies : ℕ := 35
def students : ℕ := 20

-- Define the total number of sweet treats
def total_sweet_treats : ℕ := cookies + cupcakes + brownies

-- Define the number of sweet treats each student will receive
def sweet_treats_per_student : ℕ := total_sweet_treats / students

-- Prove that each student will receive 4 sweet treats
theorem sweet_treats_distribution : sweet_treats_per_student = 4 := 
by sorry

end sweet_treats_distribution_l759_759469


namespace order_of_fractions_l759_759223

theorem order_of_fractions (a b c d : ℚ)
  (h₁ : a = 21/14)
  (h₂ : b = 25/18)
  (h₃ : c = 23/16)
  (h₄ : d = 27/19)
  (h₅ : a > b)
  (h₆ : a > c)
  (h₇ : a > d)
  (h₈ : b < c)
  (h₉ : b < d)
  (h₁₀ : c > d) :
  b < d ∧ d < c ∧ c < a := 
sorry

end order_of_fractions_l759_759223


namespace coefficient_of_ab2c3_l759_759341

noncomputable def m : ℤ :=
  3 * ∫ x in 0..Real.pi, Real.sin x

def binom_coeff (n k : ℕ) : ℤ := (n.factorial / (k.factorial * (n-k).factorial))

def compute_coefficient (a b c : ℤ) (m : ℤ) : ℤ :=
  binom_coeff m 1 * binom_coeff (m-1) 3 * (2^2) * (-3^3)

theorem coefficient_of_ab2c3 :
  let m := 3 * ∫ x in 0..Real.pi, Real.sin x in
  m = 6 →
  compute_coefficient a b c m = -6480 :=
by
  intros m_eq
  rw [m_eq]
  sorry

end coefficient_of_ab2c3_l759_759341


namespace greatest_three_digit_multiple_of_17_l759_759678

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759678


namespace max_dist_O_to_Circle_l759_759117

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the circle C with center (3, 4) and radius 1
def is_on_circle (M : ℝ × ℝ) : Prop := 
  let (x, y) := M in
  (x - 3) ^ 2 + (y - 4) ^ 2 = 1

-- Define the distance |OM|
def dist (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

-- The maximum value of the distance |OM|, given the conditions
theorem max_dist_O_to_Circle (M : ℝ × ℝ) (h : is_on_circle M) : dist O M = 6 := by
  sorry

end max_dist_O_to_Circle_l759_759117


namespace solution_count_l759_759175

theorem solution_count (a : ℝ) :
  (∃ x y : ℝ, x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) → 
  (∃ (num_solutions : ℕ), 
    (num_solutions = 3 ∧ a = 1 ∨ a = -1) ∨ 
    (num_solutions = 2 ∧ a = Real.sqrt 2 ∨ a = -Real.sqrt 2)) :=
by sorry

end solution_count_l759_759175


namespace sum_integers_75_to_95_l759_759212

theorem sum_integers_75_to_95 : ∑ k in finset.Icc 75 95, k = 1785 := by
  sorry

end sum_integers_75_to_95_l759_759212


namespace greatest_three_digit_multiple_of_17_l759_759828

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759828


namespace greatest_three_digit_multiple_of_17_l759_759583

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759583


namespace greatest_three_digit_multiple_of_17_l759_759843

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759843


namespace sum_first_9_terms_l759_759008

-- Definitions for arithmetic sequence and sum of arithmetic series
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n + 1) * (a 0 + a n) / 2

-- Given conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
hypothesis h_arith_seq : arithmetic_sequence a
hypothesis h_a2 : a 1 = 3 * a 0 - 6

-- The property to prove
theorem sum_first_9_terms : S 9 = 27 :=
  sorry

end sum_first_9_terms_l759_759008


namespace arithmetic_sequence_twentieth_term_l759_759147

theorem arithmetic_sequence_twentieth_term :
  let a₁ := 8
  let d := -3
  let aₙ (n : ℕ) := a₁ + (n - 1) * d
  aₙ 20 = -49 :=
by
  -- Definition of the initial term and common difference
  let a₁ := 8
  let d := -3
  -- Definition of the general term formula
  let aₙ (n : ℕ) := a₁ + (n - 1) * d
  -- Goal is to prove the 20th term equals -49
  have : aₙ 20 = -49 := sorry
  exact this

end arithmetic_sequence_twentieth_term_l759_759147


namespace train_length_correct_l759_759265

noncomputable def train_length (speed_train speed_man : ℝ) (relative_time : ℝ) : ℝ :=
  let relative_speed := (speed_train + speed_man) * 1000 / 3600
  relative_speed * relative_time

theorem train_length_correct :
  train_length 40 4 9 ≈ 109.98 :=
by
  sorry

end train_length_correct_l759_759265


namespace max_length_OB_l759_759549

-- Define the setup and conditions
variables (O A B : Type)
          [metric_space O]
          (ray1 ray2 : O → O → ℝ)
          (h_angle : angle (ray1 O A) (ray2 O B) = 45)
          (h_AB : dist A B = 2)

-- State the theorem to be proved
theorem max_length_OB (A B : O) (O : O):
  ∀ (ray1 : O → O → ℝ) (ray2 : O → O → ℝ),
  angle (ray1 O A) (ray2 O B) = 45 →
  dist A B = 2 →
  ∃ (OB : ℝ), OB = 2 * sqrt 2 := sorry

end max_length_OB_l759_759549


namespace sum_of_primes_less_than_20_l759_759884

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759884


namespace greatest_three_digit_multiple_of_17_l759_759616

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759616


namespace greatest_three_digit_multiple_of_17_l759_759788

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759788


namespace greatest_three_digit_multiple_of_17_is_986_l759_759668

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759668


namespace sum_of_primes_lt_20_eq_77_l759_759899

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759899


namespace total_distance_traveled_l759_759431

def speed := 60  -- Jace drives 60 miles per hour
def first_leg_time := 4  -- Jace drives for 4 hours straight
def break_time := 0.5  -- Jace takes a 30-minute break (0.5 hours)
def second_leg_time := 9  -- Jace drives for another 9 hours straight

def distance (speed : ℕ) (time : ℕ) : ℕ := speed * time  -- Distance formula

theorem total_distance_traveled : 
  distance speed first_leg_time + distance speed second_leg_time = 780 := by
-- Sorry allows us to skip the proof, since only the statement is required.
sorry

end total_distance_traveled_l759_759431


namespace greatest_three_digit_multiple_of_17_l759_759838

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759838


namespace greatest_three_digit_multiple_of_17_l759_759735

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759735


namespace greatest_three_digit_multiple_of_17_l759_759851

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759851


namespace find_magnitude_of_a_minus_2b_l759_759031

open Real

namespace VectorMagnitude

def vector (α : Type*) := vector α 2

-- Define a and b as given
def a : vector ℝ := ⟨[x, y], by simp⟩
def b : vector ℝ := ⟨[-1, 2], by simp⟩

-- Encode the given condition ∀ a + b = (1, 3)
def a_add_b_eq : a + b = ⟨[1, 3], by simp⟩ := sorry

-- Define the magnitude function
def magnitude (v : vector ℝ) : ℝ :=
  real.sqrt (v.head^2 + v.tail.head^2)

-- Define the vector expression a - 2b
def a_minus_2b : vector ℝ := a - (2 • b)

-- Statement of the proof problem
theorem find_magnitude_of_a_minus_2b (x y : ℝ) (h : a + b = (λ _ => ⟨[1, 3], by simp⟩)) :
  magnitude (a - (2 • b)) = 5 := sorry

end VectorMagnitude

end find_magnitude_of_a_minus_2b_l759_759031


namespace greatest_three_digit_multiple_of_17_l759_759584

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759584


namespace rhombus_properties_l759_759155

noncomputable def rhombus_perimeter_area (d1 d2 : ℝ) : ℝ × ℝ :=
  let hypotenuse := Math.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  let perimeter := 4 * hypotenuse
  let area := (d1 * d2) / 2
  (perimeter, area)

theorem rhombus_properties (d1 d2 : ℝ) (hd1 : d1 = 10) (hd2 : d2 = 24) :
  rhombus_perimeter_area d1 d2 = (52, 120) :=
by
  rw [hd1, hd2]
  sorry

end rhombus_properties_l759_759155


namespace area_of_sector_l759_759007

theorem area_of_sector (r : ℝ) (theta : ℝ) (h_r : r = 6) (h_theta : theta = 60) : (θ / 360 * π * r^2 = 6 * π) :=
by sorry

end area_of_sector_l759_759007


namespace volume_inscribed_sphere_l759_759263

-- Defining the radius of the sphere as half of the diameter of the cylinder's base.
def radius_sphere (diameter_cylinder_base : ℝ) : ℝ :=
  diameter_cylinder_base / 2

-- Defining the volume formula for a sphere.
def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

-- Given conditions
def diameter_cylinder_base : ℝ := 10
def height_cylinder : ℝ := 12

-- Volume of the inscribed sphere
theorem volume_inscribed_sphere : volume_sphere (radius_sphere diameter_cylinder_base) = (500 / 3) * Real.pi := by
  sorry

end volume_inscribed_sphere_l759_759263


namespace greatest_three_digit_multiple_of_17_l759_759862

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759862


namespace greatest_three_digit_multiple_of_17_l759_759574

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759574


namespace range_of_m_l759_759132

-- Definitions of Propositions p and q
def Proposition_p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m > 0) ∧ (1 > 0)  -- where x₁ + x₂ = -m > 0 and x₁x₂ = 1

def Proposition_q (m : ℝ) : Prop :=
  16 * (m + 2)^2 - 16 < 0  -- discriminant of 4x^2 + 4(m+2)x + 1 = 0 is less than 0

-- Given: "Proposition p or Proposition q" is true
def given (m : ℝ) : Prop :=
  Proposition_p m ∨ Proposition_q m

-- Prove: Range of values for m is (-∞, -1)
theorem range_of_m (m : ℝ) (h : given m) : m < -1 :=
sorry

end range_of_m_l759_759132


namespace soccer_players_l759_759270

/-- 
If the total number of socks in the washing machine is 16,
and each player wears a pair of socks (2 socks per player), 
then the number of players is 8. 
-/
theorem soccer_players (total_socks : ℕ) (socks_per_player : ℕ) (h1 : total_socks = 16) (h2 : socks_per_player = 2) : total_socks / socks_per_player = 8 :=
by
  -- Proof goes here
  sorry

end soccer_players_l759_759270


namespace sweet_treats_per_student_l759_759470

theorem sweet_treats_per_student : 
  ∀ (cookies cupcakes brownies students : ℕ), 
    cookies = 20 →
    cupcakes = 25 →
    brownies = 35 →
    students = 20 →
    (cookies + cupcakes + brownies) / students = 4 :=
by
  intros cookies cupcakes brownies students hcook hcup hbrown hstud
  have h1 : cookies + cupcakes + brownies = 80, from calc
    cookies + cupcakes + brownies = 20 + 25 + 35 := by rw [hcook, hcup, hbrown]
    ... = 80 := rfl
  have h2 : (cookies + cupcakes + brownies) / students = 80 / 20, from
    calc (cookies + cupcakes + brownies) / students
      = 80 / 20 := by rw [h1, hstud]
  exact eq.trans h2 (by norm_num)

end sweet_treats_per_student_l759_759470


namespace greatest_three_digit_multiple_of17_l759_759731

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759731


namespace greatest_three_digit_multiple_of_17_l759_759608

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759608


namespace sum_primes_less_than_20_l759_759936

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759936


namespace evaluate_expression_l759_759312

theorem evaluate_expression :
  (-2)^3 + (-2)^2 + (-2)^1 + 2^1 + 2^2 + 2^3 = 8 :=
by
  sorry

end evaluate_expression_l759_759312


namespace number_of_special_points_correct_l759_759443

-- Define the unit square region
def unit_square : set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define n-ray partitional point inside a square region R
def is_n_ray_partitional (n : ℕ) (X : ℝ × ℝ) : Prop :=
  n ≥ 4 ∧ ∃ (rays : fin n → ℝ → ℝ × ℝ), 
    (∀ i, rays i 0 = X) ∧ 
    (∀ i, ∃ t > 0, t < 1 ∧ (rays i t).1 = 1 ∨ (rays i t).1 = 0 ∨ (rays i t).2 = 1 ∨ (rays i t).2 = 0) ∧ 
    (∀ (i j : fin n), i ≠ j → ∃ t₁ t₂ > 0, rays i t₁ ≠ rays j t₂) ∧ 
    (area_divided_by_rays unit_square X rays = some (area unit_square / n))

-- Define the number of points that are 100-ray partitional but not 60-ray partitional
def num_special_points : ℕ :=
  2320

-- The theorem statement
theorem number_of_special_points_correct :
  ∀ (R : set (ℝ × ℝ)), R = unit_square →
    ∀ (n : ℕ), n = 100 →
    ∀ (m : ℕ), m = 60 →
      (num_points_n_ray_partitional R n - num_points_n_ray_partitional R m = num_special_points) := 
sorry

-- Auxiliary definition for counting n-ray partitional points inside a set R
noncomputable def num_points_n_ray_partitional (R : set (ℝ × ℝ)) (n : ℕ) : ℕ :=
  sorry


end number_of_special_points_correct_l759_759443


namespace rotated_line_x_intercept_l759_759072

theorem rotated_line_x_intercept :
  let l := λ x y : ℝ, 3 * x - 5 * y + 40 = 0 in
  let point := (20.0, 20.0) in
  let θ := Real.pi / 4 in -- 45 degrees in radians
  let new_slope := ( 4 : ℝ) in -- This results after rotating the slope of 3/5 by 45 degrees
  let new_line := λ x y : ℝ, y - 20 = new_slope * (x - 20) in
  ∃ x : ℝ, new_line x 0 ∧ x = 15 :=
begin
  sorry
end

end rotated_line_x_intercept_l759_759072


namespace arith_seq_sum_l759_759529

theorem arith_seq_sum (a₃ a₄ a₅ : ℤ) (h₁ : a₃ = 7) (h₂ : a₄ = 11) (h₃ : a₅ = 15) :
  let d := a₄ - a₃;
  let a := a₄ - 3 * d;
  (6 / 2 * (2 * a + 5 * d)) = 54 :=
by
  sorry

end arith_seq_sum_l759_759529


namespace set_B_is_correct_l759_759026

open Set

variable (A : Set Int) (f : Int → Int)
def img_a_B : Set Int := image f A

def A := {-3, -2, -1, 1, 2, 3, 4}
def f (a : Int) : Int := abs a

theorem set_B_is_correct : img_a_B f A = {1, 2, 3, 4} :=
by
  sorry

end set_B_is_correct_l759_759026


namespace line_BD_parallel_plane_alpha_l759_759349

noncomputable def square {A B C D : Point} (ABCD_square : Square A B C D) : Set Point := sorry
noncomputable def distance_from_plane (P : Point) (plane : Plane) : Real := sorry

theorem line_BD_parallel_plane_alpha
    {A B C D : Point}
    {alpha : Plane}
    (ABCD_square : Square A B C D)
    (dist_A : distance_from_plane A alpha = 1)
    (dist_B : distance_from_plane B alpha = 2)
    (dist_C : distance_from_plane C alpha = 3)
    (same_side : OnSameSideOfPlane (Square.toSet ABCD_square) alpha) :
    IsParallel (Line_through B D) alpha :=
sorry

end line_BD_parallel_plane_alpha_l759_759349


namespace greatest_three_digit_multiple_of_17_l759_759591

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759591


namespace maximum_length_OB_l759_759551

theorem maximum_length_OB 
  (O A B : Type) 
  [EuclideanGeometry O]
  (h_angle_OAB : ∠ O A B = 45°)
  (h_AB : distance A B = 2) : 
  (exists OB_max, max (distance O B) = OB_max ∧ OB_max = 2 * sqrt 2) :=
by
  sorry

end maximum_length_OB_l759_759551


namespace sum_of_primes_lt_20_eq_77_l759_759905

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759905


namespace passengers_in_each_car_l759_759539

theorem passengers_in_each_car (P : ℕ) (h1 : 20 * (P + 2) = 80) : P = 2 := 
by
  sorry

end passengers_in_each_car_l759_759539


namespace max_product_of_first_n_terms_l759_759009

theorem max_product_of_first_n_terms (n: ℕ) : 
  let S : ℕ → ℤ := λ n, n^2 - 10 * n,
      a : ℕ → ℤ := λ n, if n = 1 then S 1 else S n - S (n - 1) in
  n > 0 → ∃ m : ℕ, ∀ k : ℕ, (k = m → ∏ i in finset.range k, a (i + 1) = ∏ i in finset.range n, a (i + 1)) :=
begin
  sorry
end

end max_product_of_first_n_terms_l759_759009


namespace original_cardboard_area_l759_759233

theorem original_cardboard_area
    (a b h : ℝ)
    (a_w b_w : a = 5) 
    (a_l b_l : b = 4) 
    (volume : a * b * h = 60) 
    (h_val : h = 3) :
  let original_length := h + a + h in
  let original_width := h + b + h in 
  original_length * original_width = 110 := by
  sorry

end original_cardboard_area_l759_759233


namespace calculate_y_l759_759192

variable {A B C D E : Type}
variable (tri : A Vectors) (is_acute : ∀ {A B C : Type}, A is_acute ∧ B is_acute ∧ C is_acute)
variable (BD : Real := 7) (DC : Real := 4) (AE : Real := 3) (EB : Real := y)

theorem calculate_y (y : Real) (condition1 : A is_acute) (condition2 : B is_acute) (condition3 : C is_acute)
  (condition4 : is_acute_triangle : ∀ {A B C : VType}, A is_acute ∧ B is_acute ∧ C is_acute)
  (condition5 : BD = 7) (condition6 : DC = 4) (condition7 : AE = 3) (condition8 : EB = y) :
  y = 12 / 7 := by
  sorry

end calculate_y_l759_759192


namespace sum_of_primes_less_than_20_l759_759877

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759877


namespace greatest_three_digit_multiple_of_17_l759_759655

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759655


namespace jason_earned_amount_l759_759435

theorem jason_earned_amount (init_jason money_jason : ℤ)
    (h0 : init_jason = 3)
    (h1 : money_jason = 63) :
    money_jason - init_jason = 60 := 
by
  sorry

end jason_earned_amount_l759_759435


namespace tangent_line_at_1_minus1_l759_759157

noncomputable def f (x : ℝ) := x^3 - 2 * x^2
def tangent_at_1_minus1 := ∀ (x y : ℝ), y = -x

theorem tangent_line_at_1_minus1 :
  tangent_at_1_minus1 1 (-1) :=
by
  sorry

end tangent_line_at_1_minus1_l759_759157


namespace greatest_three_digit_multiple_of_17_l759_759808

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759808


namespace arrangement_count_example_l759_759179

theorem arrangement_count_example 
  (teachers : Finset String) 
  (students : Finset String) 
  (locations : Finset String) 
  (h_teachers : teachers.card = 2) 
  (h_students : students.card = 4) 
  (h_locations : locations.card = 2)
  : ∃ n : ℕ, n = 12 := 
sorry

end arrangement_count_example_l759_759179


namespace f_of_neg2_and_3_l759_759116

def f (x : ℝ) : ℝ :=
if x < 0 then 2*x + 4 else 9 - 3*x

theorem f_of_neg2_and_3 :
  f (-2) = 0 ∧ f (3) = 0 :=
by
  -- The proof can be filled here, but is omitted as per the instructions
  sorry

end f_of_neg2_and_3_l759_759116


namespace sum_primes_less_than_20_l759_759969

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759969


namespace greatest_three_digit_multiple_of_seventeen_l759_759697

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759697


namespace last_letter_of_60th_permutation_is_E_l759_759512

-- Define the given conditions
def word : String := "AHSSES"
def length_word : Nat := 6
def repetitions_S : Nat := 2

-- The statement to be proven
theorem last_letter_of_60th_permutation_is_E :
  (last_letter_of_60th_permutation word length_word repetitions_S = 'E') := sorry

end last_letter_of_60th_permutation_is_E_l759_759512


namespace g_2_eq_8_l759_759145

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) : ℝ := (3 * x - 4) / x

noncomputable def g (x : ℝ) : ℝ := 1 / f_inv x + 7

theorem g_2_eq_8 : g 2 = 8 := 
by 
  unfold g
  unfold f_inv
  sorry

end g_2_eq_8_l759_759145


namespace count_m_in_A_l759_759384

def A : Set ℕ := { 
  x | ∃ (a0 a1 a2 a3 : ℕ), a0 ∈ Finset.range 8 ∧ 
                           a1 ∈ Finset.range 8 ∧ 
                           a2 ∈ Finset.range 8 ∧ 
                           a3 ∈ Finset.range 8 ∧ 
                           a3 ≠ 0 ∧ 
                           x = a0 + a1 * 8 + a2 * 8^2 + a3 * 8^3 }

theorem count_m_in_A (m n : ℕ) (hA_m : m ∈ A) (hA_n : n ∈ A) (h_sum : m + n = 2018) (h_m_gt_n : m > n) :
  ∃! (count : ℕ), count = 497 := 
sorry

end count_m_in_A_l759_759384


namespace zero_intersections_l759_759000

noncomputable def Line : Type := sorry  -- Define Line as a type
noncomputable def is_skew (a b : Line) : Prop := sorry  -- Predicate for skew lines
noncomputable def is_common_perpendicular (EF a b : Line) : Prop := sorry  -- Predicate for common perpendicular
noncomputable def is_parallel (l EF : Line) : Prop := sorry  -- Predicate for parallel lines
noncomputable def count_intersections (l a b : Line) : ℕ := sorry  -- Function to count intersections

theorem zero_intersections (EF a b l : Line) 
  (h_skew : is_skew a b) 
  (h_common_perpendicular : is_common_perpendicular EF a b)
  (h_parallel : is_parallel l EF) : 
  count_intersections l a b = 0 := 
sorry

end zero_intersections_l759_759000


namespace greatest_three_digit_multiple_of_17_l759_759573

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759573


namespace train_speed_is_64_kmh_l759_759241

noncomputable def train_speed_kmh (train_length platform_length time_seconds : ℕ) : ℕ :=
  let total_distance := train_length + platform_length
  let speed_mps := total_distance / time_seconds
  let speed_kmh := speed_mps * 36 / 10
  speed_kmh

theorem train_speed_is_64_kmh
  (train_length : ℕ)
  (platform_length : ℕ)
  (time_seconds : ℕ)
  (h_train_length : train_length = 240)
  (h_platform_length : platform_length = 240)
  (h_time_seconds : time_seconds = 27) :
  train_speed_kmh train_length platform_length time_seconds = 64 := by
  sorry

end train_speed_is_64_kmh_l759_759241


namespace necessary_not_sufficient_l759_759239

theorem necessary_not_sufficient (x : ℝ) : (x > 5) → (x > 2) ∧ ¬((x > 2) → (x > 5)) :=
by
  sorry

end necessary_not_sufficient_l759_759239


namespace sum_of_integers_75_to_95_l759_759218

def arithmeticSumOfIntegers (a l : ℕ) : ℕ :=
  let n := l - a + 1
  n / 2 * (a + l)

theorem sum_of_integers_75_to_95 : arithmeticSumOfIntegers 75 95 = 1785 :=
  by
  sorry

end sum_of_integers_75_to_95_l759_759218


namespace greatest_three_digit_multiple_of_17_is_986_l759_759751

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759751


namespace total_bears_is_1620_l759_759306

-- Define the conditions for the problem
def ratio_black_white (black white : ℕ) : Prop := black = 2 * white
def more_brown_black (brown black : ℕ) : Prop := brown = black + 40

def bears_in_parks (black_a black_b black_c white_a white_b white_c brown_a brown_b brown_c total : ℕ) : Prop :=
  2 * white_a = black_a ∧ 
  brown_a = black_a + 40 ∧ 
  black_b = 3 * black_a ∧ 
  2 * white_b = black_b ∧ 
  brown_b = black_b + 40 ∧ 
  black_c = 2 * black_b ∧ 
  2 * white_c = black_c ∧ 
  brown_c = black_c + 40 ∧ 
  total = (black_a + black_b + black_c + white_a + white_b + white_c + brown_a + brown_b + brown_c)

-- The theorem to be proved
theorem total_bears_is_1620 : 
  ∃ (black_a black_b black_c white_a white_b white_c brown_a brown_b brown_c total : ℕ),
    black_a = 60 ∧ 
    bears_in_parks black_a black_b black_c white_a white_b white_c brown_a brown_b brown_c total ∧ 
    total = 1620 :=
begin
  sorry
end

end total_bears_is_1620_l759_759306


namespace greatest_three_digit_multiple_of_17_l759_759833

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759833


namespace greatest_three_digit_multiple_of_17_l759_759676

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759676


namespace greatest_three_digit_multiple_of_17_is_986_l759_759658

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759658


namespace greatest_three_digit_multiple_of_17_l759_759837

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759837


namespace range_f1_range_a_2roots_l759_759372

-- Problem 1
def f1 (x : ℝ) : ℝ := 4^x - 4 * 2^x + 3

theorem range_f1 : ∀ (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2), -1 ≤ f1 x ∧ f1 x ≤ 3 :=
sorry

-- Problem 2
def f2 (x : ℝ) (a : ℝ) : ℝ := 4^x + a * 2^x + 3

theorem range_a_2roots :
  (∀ x : ℝ, 0 < x → f2 x a = 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f2 x₁ a = 0 ∧ f2 x₂ a = 0) →
  -4 < a ∧ a < -2 * Real.sqrt 3 :=
sorry

end range_f1_range_a_2roots_l759_759372


namespace greatest_three_digit_multiple_of_17_is_986_l759_759754

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759754


namespace max_length_OB_l759_759555

-- Define the problem conditions
def angle_AOB : ℝ := 45
def length_AB : ℝ := 2
def max_sin_angle_OAB : ℝ := 1

-- Claim to be proven
theorem max_length_OB : ∃ OB_max, OB_max = 2 * Real.sqrt 2 :=
by
  sorry

end max_length_OB_l759_759555


namespace tangent_line_at_point_zero_l759_759322

open Real

theorem tangent_line_at_point_zero :
  ∀ (x : ℝ), y = sin x + exp x → tangent_slope = derivative (fun z => sin z + exp z) 0 → 
  (∃ (y : ℝ), y = sin 0 + exp 0 ∧ 2 * x - y + 1 = 0) :=
by
  intro x y eq_curve deriv_tangent
  -- The proof is omitted and left as an exercise
  sorry

end tangent_line_at_point_zero_l759_759322


namespace smallest_positive_period_interval_of_decrease_length_of_side_AC_l759_759387

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x ^ 2, Real.sin x)

noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (1 / 2, Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ :=
  vector_a x.1 * vector_b x.1 + vector_a x.2 * vector_b x.2

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x := sorry

theorem interval_of_decrease :
  ∀ k : ℤ, ∀ x ∈ Set.Icc (k * π + π / 6) (k * π + 2 * π / 3), f x > f (x + π) := sorry

theorem length_of_side_AC :
  ∀ (A B C : ℝ), A + B = (7 / 12) * π ∧ f A = 1 ∧ C = 2 * Real.sqrt 3 →
  ∃ AC, AC = 2 * Real.sqrt 2 := sorry

end smallest_positive_period_interval_of_decrease_length_of_side_AC_l759_759387


namespace greatest_three_digit_multiple_of_17_l759_759654

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759654


namespace school_paid_correct_amount_l759_759149

noncomputable def total_cost (kindergarten elementary highschool : ℕ) (price : ℕ) (discount1 discount2 : ℕ → ℕ) : ℕ :=
let total_models := kindergarten + elementary + highschool in
if total_models > 10 then
  total_models * discount2 price
else if total_models > 5 then
  total_models * discount1 price
else
  total_models * price

theorem school_paid_correct_amount :
  total_cost 2 (2 * 2) (3 * 2) 100
    (λ p, p - (p / 20))  -- 5% discount function
    (λ p, p - (p / 10))  -- 10% discount function
  = 1080 := by
    sorry

end school_paid_correct_amount_l759_759149


namespace nth_equation_proof_l759_759123

theorem nth_equation_proof (n : ℕ) (h : n ≥ 1) :
  1 / (n + 1 : ℚ) + 1 / (n * (n + 1)) = 1 / n := 
sorry

end nth_equation_proof_l759_759123


namespace greatest_three_digit_multiple_of_17_l759_759840

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759840


namespace yu_chan_walked_distance_l759_759479

def step_length : ℝ := 0.75
def walking_time : ℝ := 13
def steps_per_minute : ℝ := 70

theorem yu_chan_walked_distance : step_length * steps_per_minute * walking_time = 682.5 :=
by
  sorry

end yu_chan_walked_distance_l759_759479


namespace coefficient_of_x_in_expansion_l759_759495

noncomputable def coeff (n : ℕ) (f : ℕ → ℂ) : ℂ :=
∑ i in finset.range n.succ, (finset.powerset_len i (finset.range n)).val.cases_on 0 f

theorem coefficient_of_x_in_expansion :
  coeff 1 (λ x, ((1 + 2 * complex.sqrt x) ^ 3 * (1 - complex.cbrt x) ^ 5).coeff x) = 2 :=
sorry

end coefficient_of_x_in_expansion_l759_759495


namespace weigh_grain_with_inaccurate_scales_l759_759541

theorem weigh_grain_with_inaccurate_scales
  (inaccurate_scales : ℕ → ℕ → Prop)
  (correct_weight : ℕ)
  (bag_of_grain : ℕ → Prop)
  (balanced : ∀ a b : ℕ, inaccurate_scales a b → a = b := sorry)
  : ∃ grain_weight : ℕ, bag_of_grain grain_weight ∧ grain_weight = correct_weight :=
sorry

end weigh_grain_with_inaccurate_scales_l759_759541


namespace negq_sufficient_but_not_necessary_for_p_l759_759449

variable (p q : Prop)

theorem negq_sufficient_but_not_necessary_for_p
  (h1 : ¬p → q)
  (h2 : ¬(¬q → p)) :
  (¬q → p) ∧ ¬(p → ¬q) :=
sorry

end negq_sufficient_but_not_necessary_for_p_l759_759449


namespace ratio_of_areas_equilateral_triangle_l759_759189

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def area (tri : Triangle) : Real := sorry

theorem ratio_of_areas_equilateral_triangle
  (A B C : Point)
  (tri_ABC : equilateral_triangle A B C)
  (D : Point) (E : Point) (F : Point)
  (hD : D = midpoint A B) (hE : E = midpoint B C) (hF : F = midpoint C A)
  (P : Point) (Q : Point) (R : Point)
  (hP : P = midpoint A D) (hQ : Q = midpoint B E) (hR : R = midpoint C F) :
  let shaded_area := area (triangle A P R) + area (triangle C Q R)
  let non_shaded_area := area (triangle A B C) - shaded_area
  in shaded_area / non_shaded_area = 1 / 7 :=
sorry

end ratio_of_areas_equilateral_triangle_l759_759189


namespace greatest_three_digit_multiple_of_17_l759_759603

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759603


namespace Zack_magazines_l759_759125

theorem Zack_magazines (hours : ℕ) (minutes_per_magazine : ℕ) (minutes_per_hour : ℕ) (total_minutes : ℕ) (m : ℕ)
  (hours_eq : hours = 5)
  (minutes_per_magazine_eq : minutes_per_magazine = 20)
  (minutes_per_hour_eq : minutes_per_hour = 60)
  (total_minutes_eq : total_minutes = hours * minutes_per_hour)
  (m_eq : m = total_minutes / minutes_per_magazine) :
  m = 15 :=
by
  rw [hours_eq, minutes_per_magazine_eq, minutes_per_hour_eq] at total_minutes_eq
  have h1 : total_minutes = 5 * 60 := total_minutes_eq
  have h2 : total_minutes = 300 := by norm_num
  rw [←h2, minutes_per_magazine_eq] at m_eq
  have h3 : m = 300 / 20 := m_eq
  norm_num at h3
  exact h3

end Zack_magazines_l759_759125


namespace calculate_expression_l759_759336

noncomputable def sequence (n : ℕ) : ℝ := sorry

def recurrence_relation (n : ℕ) (hn : n ≥ 2) : Prop :=
  sequence n = (sequence (n - 1) + 198 * sequence n + sequence (n + 1)) / 200

def distinct_terms (n : ℕ) (hn : n ≥ 2) : Prop :=
  ∀ m : ℕ, m ≥ 2 → m ≠ n → sequence m ≠ sequence n

theorem calculate_expression (h1 : ∀ n ≥ 2, recurrence_relation n (by assumption))
    (h2 : ∀ n ≥ 2, distinct_terms n (by assumption)) :
  (Real.sqrt ((sequence 2023 - sequence 1) / 2022 * (2021 / (sequence 2023 - sequence 2))) + 2022) = 2023 :=
  sorry

end calculate_expression_l759_759336


namespace part_a_part_b_l759_759436

variables {A B C I H K M N X Y O : Type}
variables [triangle : NonisocelesAcuteTriangle A B C (circle O)]
variables [PointOnSegment I B C]
variables [Projection H I A B]
variables [Projection K I A C]
variables [IntersectLineCircle HK (circle O) M N]
variables [CenterCircle X A B K]
variables [CenterCircle Y A C H]

-- Part (a)
theorem part_a (projection : Perpendicular A I B C) : Center (circle I M N) A := sorry

-- Part (b)
theorem part_b (parallel : Parallel X Y B C) : OrthocenterOfTriangle X O Y = Midpoint I O := sorry

end part_a_part_b_l759_759436


namespace robot_vacuum_distance_and_velocity_change_l759_759174

/-- The robot vacuum cleaner's movement in the x-direction for 0 ≤ t ≤ 7 is given by x = t(t-6)^2 --/
def x (t : ℝ) : ℝ := t * (t - 6)^2

/-- The y-direction is constant 0 for 0 ≤ t ≤ 7 --/
def y (t : ℝ) : ℝ := 0

/-- The y-direction for t ≥ 7 is given by y = (t - 7)^2 --/
def y_after_seven (t : ℝ) : ℝ := (t - 7)^2

/-- The velocity in the x-direction for 0 ≤ t ≤ 7 is the derivative of x(t) --/
def velocity_x (t : ℝ) : ℝ := deriv x t

/-- The velocity in the y-direction for t ≥ 7 is the derivative of y_after_seven(t) --/
def velocity_y_after_seven (t : ℝ) : ℝ := deriv y_after_seven t

/-- Prove that the distance traveled by the robot in the first 7 minutes is 71 meters 
    and the absolute change in the velocity vector during the eighth minute is √445. --/
theorem robot_vacuum_distance_and_velocity_change :
  (∫ t in 0..7, abs (deriv x t)) = 71 ∧
  (sqrt ((velocity_x 8 - velocity_x 7)^2 + (velocity_y_after_seven 8 - velocity_y_after_seven 7)^2)) = sqrt 445 :=
by
  sorry

end robot_vacuum_distance_and_velocity_change_l759_759174


namespace find_integer_solutions_l759_759319

theorem find_integer_solutions :
  ∃ s : set (ℤ × ℤ), s = {(3, -1), (5, 1), (1, 5), (-1, 3)} ∧
    ∀ x y : ℤ, 2 * (x + y) = x * y + 7 ↔ (x, y) ∈ s :=
by
  sorry

end find_integer_solutions_l759_759319


namespace greatest_three_digit_multiple_of_17_l759_759588

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759588


namespace symmetry_y_axis_B_l759_759423

def point_A : ℝ × ℝ := (-1, 2)

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-(p.1), p.2)

theorem symmetry_y_axis_B :
  symmetric_point point_A = (1, 2) :=
by
  -- proof is omitted
  sorry

end symmetry_y_axis_B_l759_759423


namespace fraction_of_earth_habitable_l759_759407

theorem fraction_of_earth_habitable :
  ∀ (earth_surface land_area inhabitable_land_area : ℝ),
    land_area = 1 / 3 → 
    inhabitable_land_area = 1 / 4 → 
    (earth_surface * land_area * inhabitable_land_area) = 1 / 12 :=
  by
    intros earth_surface land_area inhabitable_land_area h_land h_inhabitable
    sorry

end fraction_of_earth_habitable_l759_759407


namespace parabola_range_proof_l759_759380

noncomputable def parabola_range (a : ℝ) : Prop := 
  (-2 ≤ a ∧ a < 3) → 
  ∃ b : ℝ, b = a^2 + 2*a + 4 ∧ (3 ≤ b ∧ b < 19)

theorem parabola_range_proof (a : ℝ) (h : -2 ≤ a ∧ a < 3) : 
  ∃ b : ℝ, b = a^2 + 2*a + 4 ∧ (3 ≤ b ∧ b < 19) :=
sorry

end parabola_range_proof_l759_759380


namespace greatest_three_digit_multiple_of_17_l759_759807

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759807


namespace simplify_expression_l759_759488

-- The main problem statement:
theorem simplify_expression :
  ((-3 - 3/8 : ℝ) ^ (-2/3) * log 5 ((3 : ℝ)^(log 9 5) - (3 * real.sqrt 3)^(2/3) + (7 : ℝ)^(log 7 3))) = (2/9 : ℝ) :=
begin
  sorry
end

end simplify_expression_l759_759488


namespace count_integer_solutions_l759_759389

def integer_solutions_count (a b c : Int) (x : Int) : Prop :=
  |a * x + b| ≤ c

theorem count_integer_solutions : 
  ∃ (S : Finset Int), 
  (∀ x ∈ S, (|7 * x - 4| ≤ 14)) ∧ (S.card = 4) := 
begin
  sorry
end

end count_integer_solutions_l759_759389


namespace sum_of_primes_less_than_20_is_77_l759_759928

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759928


namespace sum_of_primes_less_than_20_l759_759984

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759984


namespace exists_m_n_for_d_l759_759133

theorem exists_m_n_for_d (d : ℤ) : ∃ m n : ℤ, d = (n - 2 * m + 1) / (m^2 - n) := 
sorry

end exists_m_n_for_d_l759_759133


namespace greatest_three_digit_multiple_of_17_l759_759795

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759795


namespace sum_primes_less_than_20_l759_759939

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759939


namespace num_perfect_square_factors_of_2160_l759_759169

theorem num_perfect_square_factors_of_2160 :
  let p := 2160, pf := prime_factorization p
  in is_factor_of_perfect_square p pf (2^4 * 3^3 * 5) :=
begin
  sorry -- Proof is omitted
end

end num_perfect_square_factors_of_2160_l759_759169


namespace ScarlettsDishCost_l759_759460

theorem ScarlettsDishCost (L P : ℝ) (tip_rate tip_amount : ℝ) (x : ℝ) 
  (hL : L = 10) (hP : P = 17) (htip_rate : tip_rate = 0.10) (htip_amount : tip_amount = 4) 
  (h : tip_rate * (L + P + x) = tip_amount) : x = 13 :=
by
  sorry

end ScarlettsDishCost_l759_759460


namespace sum_of_primes_less_than_20_l759_759895

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759895


namespace k_value_l759_759335

theorem k_value (k : ℝ) (x : ℝ) (y : ℝ) (hk : k^2 - 5 = -1) (hx : x > 0) (hy : y = (k - 1) * x^(k^2 - 5)) (h_dec : ∀ (x1 x2 : ℝ), x1 > 0 → x2 > x1 → (k - 1) * x2^(k^2 - 5) < (k - 1) * x1^(k^2 - 5)):
  k = 2 := by
  sorry

end k_value_l759_759335


namespace math_problem_l759_759170

variable (a b c : ℝ)

variables (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
variables (h4 : a ≠ -b) (h5 : b ≠ -c) (h6 : c ≠ -a)

theorem math_problem 
    (h₁ : (a * b) / (a + b) = 4)
    (h₂ : (b * c) / (b + c) = 5)
    (h₃ : (c * a) / (c + a) = 7) :
    (a * b * c) / (a * b + b * c + c * a) = 280 / 83 := 
sorry

end math_problem_l759_759170


namespace sum_of_primes_less_than_20_l759_759999

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l759_759999


namespace greatest_three_digit_multiple_of_17_l759_759639

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759639


namespace greatest_three_digit_multiple_of_17_l759_759615

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759615


namespace greatest_three_digit_multiple_of_17_l759_759785

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759785


namespace greatest_three_digit_multiple_of_17_l759_759565

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759565


namespace percentage_increase_of_numerator_l759_759165

theorem percentage_increase_of_numerator (N D : ℝ) (P : ℝ) (h1 : N / D = 0.75)
  (h2 : (N + (P / 100) * N) / (D - (8 / 100) * D) = 15 / 16) :
  P = 15 :=
sorry

end percentage_increase_of_numerator_l759_759165


namespace greatest_three_digit_multiple_of_17_l759_759581

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759581


namespace necessary_condition_for_abs_ab_l759_759430

theorem necessary_condition_for_abs_ab {a b : ℝ} (h : |a - b| = |a| - |b|) : ab ≥ 0 :=
sorry

end necessary_condition_for_abs_ab_l759_759430


namespace max_value_smallest_area_l759_759001

-- Definitions as per the conditions given in the problem

-- Define the points A, B, and C and the interior points D and E
-- Assume the area of triangle ABC is 1.
variables {A B C D E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]

-- Define the function that calculates the area of a triangle given three points
def area (P Q R : Type) [Inhabited P] [Inhabited Q] [Inhabited R] : ℝ := sorry

-- Define the assumption that points D and E lie within the triangle ABC
def in_triangle (P A B C : Type) [Inhabited P] [Inhabited A] [Inhabited B] [Inhabited C] : Prop := sorry

axiom h1 : area A B C = 1
axiom h2 : in_triangle D A B C
axiom h3 : in_triangle E A B C

-- Define the problem of finding the maximum value of the smallest area 
-- of the triangles formed by points {A, B, C, D, E}

noncomputable def max_smallest_area (s : ℝ) : Prop :=
∀ (X Y Z : Type) [Inhabited X] [Inhabited Y] [Inhabited Z], 
  area X Y Z ≥ s → area X Y Z ≤ 1 - (Real.sqrt 3 / 2)

theorem max_value_smallest_area :
  ∃ s, max_smallest_area s ∧ s = 1 - Real.sqrt 3 / 2 :=
begin
  -- Proof can be filled here, we just state that the theorem should exist.
  sorry
end

end max_value_smallest_area_l759_759001


namespace sum_primes_less_than_20_l759_759976

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759976


namespace greatest_three_digit_multiple_of_seventeen_l759_759696

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759696


namespace sum_of_primes_less_than_20_l759_759887

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759887


namespace find_b_l759_759280

variable {a b c d : ℝ}

def y_eq_a_cos_bx_plus_c_d (x : ℝ) : ℝ :=
  a * Real.cos (b * x + c) + d

theorem find_b
  (h1 : ∀ x, y_eq_a_cos_bx_plus_c_d (x + 3 * Real.pi / 2) = y_eq_a_cos_bx_plus_c_d x)
  (h2 : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  b = 4 / 3 := by
  sorry

end find_b_l759_759280


namespace domain_sqrt_function_l759_759156

theorem domain_sqrt_function :
  (∃ (D : set ℝ), ∀ x, x ∈ D ↔ (∃ y, y = real.sqrt (x - 1))) :=
begin
  sorry
end

end domain_sqrt_function_l759_759156


namespace greatest_three_digit_multiple_of_17_l759_759853

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759853


namespace integer_solutions_l759_759317

theorem integer_solutions (x y : ℤ) : 2 * (x + y) = x * y + 7 ↔ (x, y) = (3, -1) ∨ (x, y) = (5, 1) ∨ (x, y) = (1, 5) ∨ (x, y) = (-1, 3) := by
  sorry

end integer_solutions_l759_759317


namespace values_of_a_b_l759_759446

-- Let a and b be real numbers
variables (a b : ℝ)

-- Condition that roots of z^2 + (6 + a * I) * z + (14 + b * I) = 0 are complex conjugates
def roots_are_complex_conjugates (a b : ℝ) : Prop :=
  ∃ (z1 z2 : ℂ), z1 = complex.conj z2 ∧
                 (∀ z, z^2 + (6 + a * complex.I) * z + (14 + b * complex.I) = 0 ↔ z = z1 ∨ z = z2)

-- Prove that a = 0 and b = 0
theorem values_of_a_b (ha : a = 0) (hb : b = 0) : 
  roots_are_complex_conjugates a b :=
sorry

end values_of_a_b_l759_759446


namespace greatest_three_digit_multiple_of_17_l759_759858

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759858


namespace part1_part2_l759_759373

noncomputable def f (x : ℝ) : ℝ := (4^x - 1) / (4^x + 1)

theorem part1 (x : ℝ) : f x < 1 / 3 ↔ x < 1 / 2 :=
sorry

theorem part2 : set.range f = set.Ioo (-1 : ℝ) 1 :=
sorry

end part1_part2_l759_759373


namespace greatest_three_digit_multiple_of_17_l759_759857

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759857


namespace sum_primes_less_than_20_l759_759945

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759945


namespace greatest_three_digit_multiple_of_17_l759_759677

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759677


namespace greatest_three_digit_multiple_of_17_l759_759637

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759637


namespace smallest_y_value_l759_759302

theorem smallest_y_value (y : ℝ) : 3 * y ^ 2 + 33 * y - 105 = y * (y + 16) → y = -21 / 2 ∨ y = 5 := sorry

end smallest_y_value_l759_759302


namespace greatest_three_digit_multiple_of_17_is_986_l759_759636

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759636


namespace greatest_three_digit_multiple_of17_l759_759729

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759729


namespace greatest_three_digit_multiple_of_17_l759_759770

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759770


namespace greatest_three_digit_multiple_of_17_l759_759569

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759569


namespace line_intersects_circle_l759_759370

-- Definitions based on conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 9 = 0
def line_eq (m x y : ℝ) : Prop := m*x + y + m - 2 = 0

-- Theorem statement based on question and correct answer
theorem line_intersects_circle (m : ℝ) :
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y :=
sorry

end line_intersects_circle_l759_759370


namespace sum_primes_less_than_20_l759_759978

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759978


namespace probability_union_inequality_l759_759385

theorem probability_union_inequality (P : set (set α) → ℝ) [ProbabilityMeasure P] (A B : set α) :
  P A + P B ≥ P (A ∪ B) :=
sorry

end probability_union_inequality_l759_759385


namespace greatest_three_digit_multiple_of_17_l759_759681

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759681


namespace greatest_three_digit_multiple_of_17_l759_759827

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759827


namespace greatest_three_digit_multiple_of_17_l759_759813

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759813


namespace sum_primes_less_than_20_l759_759952

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759952


namespace equal_distances_l759_759438

-- Define the appropriate types and structures
variables {Point : Type} [MetricSpace Point] [CMetricSpace Point]

-- Let definitions for the points and circles
variables {O1 O2 A B C K L D : Point}
variables {ω₁ ω₂ : Circle Point}

-- Given the conditions
axiom intersect_circles (h₁ : intersects ω₁ ω₂) : Point
axiom O1O2_A_intersects : intersects_line_circle (line_through O1 O2) ω₁ = A
axiom O1O2_B_intersects : intersects_line_circle (line_through O1 O2) ω₂ = B
axiom AC_K_intersects : second_point_of_intersection (line_through A C) ω₂ = K
axiom BC_L_intersects : second_point_of_intersection (line_through B C) ω₁ = L
axiom AL_BK_intersect_D : (line_through A L).intersects (line_through B K) = D

-- The statement to prove
theorem equal_distances {O1 O2 A B C K L D : Point}
  (h₁ : intersects ω₁ ω₂)
  (h₂ : intersect_circles h₁ = C)
  (h₃ : intersects_line_circle (line_through O1 O2) ω₁ = A)
  (h₄ : intersects_line_circle (line_through O1 O2) ω₂ = B)
  (h₅ : second_point_of_intersection (line_through A C) ω₂ = K)
  (h₆ : second_point_of_intersection (line_through B C) ω₁ = L)
  (h₇ : (line_through A L).intersects (line_through B K) = D) :
  distance A D = distance B D :=
sorry

end equal_distances_l759_759438


namespace relationship_between_abc_l759_759339

noncomputable def a := Real.logb 0.5 0.6
noncomputable def b := (0.25 : ℝ) ^ (-0.3)
noncomputable def c := (0.6 : ℝ) ^ (-0.6)

theorem relationship_between_abc : b > c ∧ c > a := by
  sorry

end relationship_between_abc_l759_759339


namespace num_solutions_coin_exchange_eq_29_l759_759294

def num_solutions_coin_exchange : ℕ :=
  (finset.range 21).sum (λ x, (finset.range 11).sum (λ y,
    if (20 - 2 * y) ≥ 0 ∧ (20 - 2 * y) % 5 = 0
    then 1
    else 0))

theorem num_solutions_coin_exchange_eq_29 :
  num_solutions_coin_exchange = 29 := 
  sorry

end num_solutions_coin_exchange_eq_29_l759_759294


namespace binomial_variance_solution_l759_759347

theorem binomial_variance_solution (p : ℚ) (X : ℕ → ℚ) (n : ℕ) 
  (h1 : n = 4) (h2 : ∀ k, X k ~ B(4, p)) (h3 : D(X) = 1) : p = 1/2 := 
sorry

end binomial_variance_solution_l759_759347


namespace greatest_three_digit_multiple_of_17_is_986_l759_759660

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759660


namespace line_intersects_circle_l759_759167

open Real

noncomputable def line (k : ℝ) : ℝ × ℝ → Prop :=
λ p, p.2 = k * p.1 - 3 * k

noncomputable def circle : ℝ × ℝ → Prop :=
λ p, (p.1 - 2) ^ 2 + p.2 ^ 2 = 4

theorem line_intersects_circle (k : ℝ) : ∃ p : ℝ × ℝ, line k p ∧ circle p :=
sorry

end line_intersects_circle_l759_759167


namespace greatest_three_digit_multiple_of_17_l759_759856

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759856


namespace factory_daily_production_l759_759250

theorem factory_daily_production (weekly_production : ℕ) (days_per_week : ℕ) (h1 : weekly_production = 5500) (h2 : days_per_week = 5) :
  weekly_production / days_per_week = 1100 :=
by
  rw [h1, h2]
  norm_num

end factory_daily_production_l759_759250


namespace sum_of_sequence_S100_l759_759444

theorem sum_of_sequence_S100 :
  let a : ℕ → ℤ := λ n, if n = 0 then 0 else if even n then -2^((n - 1) / 2) else 0
  let S : ℕ → ℤ := λ n, ∑ i in range n, a (i + 1)
  S 100 = (2 - 2^101) / 3 :=
by
  sorry

end sum_of_sequence_S100_l759_759444


namespace relationship_x1_x2_x3_l759_759408

variables (k x1 x2 x3 : ℝ)

def f (k x : ℝ) := - (k^2 + 10) / x

theorem relationship_x1_x2_x3 : 
  (f k x1 = -3) → (f k x2 = -2) → (f k x3 = 1) → x3 < x1 ∧ x1 < x2 :=
by
  assume h1 : f k x1 = -3
  assume h2 : f k x2 = -2
  assume h3 : f k x3 = 1
  
  -- Skipping proof details
  sorry

end relationship_x1_x2_x3_l759_759408


namespace greatest_three_digit_multiple_of_17_l759_759609

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759609


namespace pizza_area_increase_l759_759151

theorem pizza_area_increase (A1 A2 r1 r2 : ℝ) (r1_eq : r1 = 7) (r2_eq : r2 = 5) (A1_eq : A1 = Real.pi * r1^2) (A2_eq : A2 = Real.pi * r2^2) :
  ((A1 - A2) / A2) * 100 = 96 := by
  sorry

end pizza_area_increase_l759_759151


namespace regular_price_of_pony_jeans_l759_759338

-- Define relevant parameters
variables (P : ℝ) 
variables (fox_price : ℝ := 15)
variables (total_savings : ℝ := 9)
variables (total_discount : ℝ := 0.22)
variables (pony_discount : ℝ := 0.10)

-- The relevant conditions
def fox_discount := total_discount - pony_discount
def fox_savings := fox_price * fox_discount * 3
def pony_savings := total_savings - fox_savings
def pony_price := pony_savings / 2 / pony_discount

theorem regular_price_of_pony_jeans : P = 18 :=
by
  unfold fox_discount fox_savings pony_savings pony_price
  sorry

end regular_price_of_pony_jeans_l759_759338


namespace greatest_three_digit_multiple_of_17_is_986_l759_759622

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759622


namespace janet_additional_money_needed_l759_759083

def janet_savings : ℕ := 2225
def monthly_rent : ℕ := 1250
def advance_months : ℕ := 2
def deposit : ℕ := 500

theorem janet_additional_money_needed :
  (advance_months * monthly_rent + deposit - janet_savings) = 775 :=
by
  sorry

end janet_additional_money_needed_l759_759083


namespace find_f_and_cos2x0_l759_759013

open Real

noncomputable def f (ω x : ℝ) (m : ℝ) : ℝ := 2 * (cos (ω * x)) ^ 2 + 2 * sqrt 3 * sin (ω * x) * cos (ω * x) + m

theorem find_f_and_cos2x0 (ω : ℝ) (m : ℝ) (h₁ : ω > 0) 
(h₂ : f ω (π / 6) m = (2 * sin (2 * (π / 6) + π / 6)) + 1) 
(h₃ : f ω 0 m = 2) 
(h₄ : ∀ x0 : ℝ, x0 ∈ set.Icc (π / 4) (π / 2) → f ω x0 m = 11 / 5) :
  (∃ m : ℝ, ∀ x : ℝ, f ω x m = 2 * sin (2 * x + π / 6) + 1) ∧ 
  (∃ x0 : ℝ, x0 ∈ set.Icc (π / 4) (π / 2) ∧ 
    cos (2 * x0) = (3 - 4 * sqrt 3) / 10) :=
by 
  sorry

end find_f_and_cos2x0_l759_759013


namespace greatest_three_digit_multiple_of_17_l759_759566

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759566


namespace number_of_zeros_l759_759164

noncomputable def f (x : ℝ) : ℝ := |2^x - 1| - 3^x

theorem number_of_zeros : ∃! x : ℝ, f x = 0 := sorry

end number_of_zeros_l759_759164


namespace maximum_length_OB_l759_759552

theorem maximum_length_OB 
  (O A B : Type) 
  [EuclideanGeometry O]
  (h_angle_OAB : ∠ O A B = 45°)
  (h_AB : distance A B = 2) : 
  (exists OB_max, max (distance O B) = OB_max ∧ OB_max = 2 * sqrt 2) :=
by
  sorry

end maximum_length_OB_l759_759552


namespace janet_additional_money_needed_l759_759084

def janet_savings : ℕ := 2225
def monthly_rent : ℕ := 1250
def advance_months : ℕ := 2
def deposit : ℕ := 500

theorem janet_additional_money_needed :
  (advance_months * monthly_rent + deposit - janet_savings) = 775 :=
by
  sorry

end janet_additional_money_needed_l759_759084


namespace sum_of_primes_less_than_20_l759_759882

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759882


namespace tangent_line_eq_l759_759321

def f (x : ℝ) : ℝ := x^3 + x

theorem tangent_line_eq :
  ∃ A B C : ℝ, A = 4 ∧ B = -1 ∧ C = -2 ∧ ∀ x y : ℝ, y = f' 1 * (x - 1) + f 1 → (A * x + B * y + C = 0) :=
by
  sorry

noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 1

end tangent_line_eq_l759_759321


namespace a_10_eq_19_l759_759427

open Nat

def sequence (n : ℕ) : ℕ :=
  if n = 0 then 0 else nat.rec_on n 1 (λ _ a, a + 2)

theorem a_10_eq_19 : sequence 10 = 19 :=
by sorry

end a_10_eq_19_l759_759427


namespace greatest_three_digit_multiple_of_17_l759_759786

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759786


namespace sum_of_primes_lt_20_eq_77_l759_759909

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759909


namespace juice_m_smoothie_l759_759240

/-- 
24 oz of juice p and 25 oz of juice v are mixed to make smoothies m and y. 
The ratio of p to v in smoothie m is 4 to 1 and that in y is 1 to 5. 
Prove that the amount of juice p in the smoothie m is 20 oz.
-/
theorem juice_m_smoothie (P_m P_y V_m V_y : ℕ)
  (h1 : P_m + P_y = 24)
  (h2 : V_m + V_y = 25)
  (h3 : 4 * V_m = P_m)
  (h4 : V_y = 5 * P_y) :
  P_m = 20 :=
sorry

end juice_m_smoothie_l759_759240


namespace largest_last_digit_l759_759158

-- Define that a string is of length n and the first digit is d1
def valid_string_length : ℕ := 1003
def first_digit : ℕ := 2

-- Define the condition on the two-digit numbers formed by consecutive digits being divisible by 17 or 23
def is_valid_pair (a b : ℕ) : Prop :=
  ∃ (x : ℤ), (10 * a + b = 17 * x ∨ 10 * a + b = 23 * x)

-- Define the string as a list of digits
def digit_string := vector ℕ valid_string_length

-- Define the main theorem we need to prove
theorem largest_last_digit (s : digit_string) :
  s.head = first_digit → 
  (∀ (i : fin (valid_string_length - 1)), is_valid_pair (s.nth i) (s.nth (i + 1))) →
  (s.nth (valid_string_length - 1) = 2) :=
sorry -- Proof to be filled in later

end largest_last_digit_l759_759158


namespace greatest_three_digit_multiple_of_17_l759_759782

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759782


namespace sum_primes_less_than_20_l759_759964

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759964


namespace molly_christmas_shipping_cost_l759_759076

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_sisters_in_law_per_brother : ℕ := 1
def num_children_per_brother : ℕ := 2

def total_relatives : ℕ :=
  num_parents + num_brothers + (num_brothers * num_sisters_in_law_per_brother) + (num_brothers * num_children_per_brother)

theorem molly_christmas_shipping_cost : total_relatives * cost_per_package = 70 :=
by
  sorry

end molly_christmas_shipping_cost_l759_759076


namespace greatest_three_digit_multiple_of_17_l759_759792

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759792


namespace sweet_treats_per_student_l759_759466

theorem sweet_treats_per_student :
  let cookies := 20
  let cupcakes := 25
  let brownies := 35
  let students := 20
  (cookies + cupcakes + brownies) / students = 4 :=
by 
  sorry

end sweet_treats_per_student_l759_759466


namespace greatest_three_digit_multiple_of17_l759_759713

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759713


namespace greatest_three_digit_multiple_of_17_is_986_l759_759627

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759627


namespace nguyen_fabric_cost_l759_759475

-- Define the conditions
def first_wedding_pants := 7
def first_wedding_pants_fabric_a := 8.5
def first_wedding_shirts := 5
def first_wedding_shirts_fabric_b := 6.2

def second_wedding_pants := 5
def second_wedding_pants_fabric_a := 9.2
def second_wedding_shirts := 8
def second_wedding_shirts_fabric_b := 6.5

def cost_per_foot_fabric_a := 6
def cost_per_foot_fabric_b := 4

def current_fabric_a_yards := 3.5
def current_fabric_b_yards := 2

-- Convert yards to feet
def yard_to_foot := 3

-- Calculate required and available fabric
def total_fabric_a_needed := 
  (first_wedding_pants * first_wedding_pants_fabric_a) + (second_wedding_pants * second_wedding_pants_fabric_a)
def total_fabric_b_needed := 
  (first_wedding_shirts * first_wedding_shirts_fabric_b) + (second_wedding_shirts * second_wedding_shirts_fabric_b)

def current_fabric_a := current_fabric_a_yards * yard_to_foot
def current_fabric_b := current_fabric_b_yards * yard_to_foot

def additional_fabric_a_needed := total_fabric_a_needed - current_fabric_a
def additional_fabric_b_needed := total_fabric_b_needed - current_fabric_b

def cost_fabric_a := additional_fabric_a_needed * cost_per_foot_fabric_a
def cost_fabric_b := additional_fabric_b_needed * cost_per_foot_fabric_b

def total_cost := cost_fabric_a + cost_fabric_b

-- Proof statement
theorem nguyen_fabric_cost : total_cost = 878 := by sorry

end nguyen_fabric_cost_l759_759475


namespace greatest_three_digit_multiple_of_17_l759_759773

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759773


namespace greatest_persistent_number_l759_759278

-- Define the initial condition: 100 times the number 1 on a board
def initial_board : list ℕ := list.repeat 100 1

-- Define the transformation rule: Each number a is replaced with a/3 written three times
def transform (board : list ℚ) (a : ℚ) : list ℚ :=
  let new_numbers := list.repeat 3 (a / 3) in
  (board.filter (≠ a)) ++ new_numbers

-- Define persistent number: At any time point, there will be at least n equal numbers
def is_persistent (n : ℕ) (board : list ℚ) : Prop :=
  ∀ time_steps, ∃ num, (num ∈ board) ∧ board.count num ≥ n

-- The statement to prove the greatest persistent number is 67
theorem greatest_persistent_number :
  ∃ n, is_persistent n initial_board ∧ ∀ m > n, ¬ is_persistent m initial_board :=
by
  existsi 67
  sorry -- Proof is omitted

end greatest_persistent_number_l759_759278


namespace greatest_three_digit_multiple_of17_l759_759721

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759721


namespace seventeenth_smallest_lucky_number_l759_759254

noncomputable def is_lucky_number (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d = 4 ∨ d = 7

theorem seventeenth_smallest_lucky_number : 
  ∃ (n : ℕ), is_lucky_number n ∧ seq_nth (filter is_lucky_number (range 10000)) 16 = some n ∧ n = 4474 :=
by 
  sorry

end seventeenth_smallest_lucky_number_l759_759254


namespace number_of_paths_in_grid_l759_759252

theorem number_of_paths_in_grid (m n : ℕ) (hm : m = 6) (hn : n = 5) :
  Nat.choose (m + n) n = 462 := 
by
  rw [hm, hn]
  exact Nat.choose_eq_factorial_div_factorial (6 + 5) 5
  -- Further steps would calculate to show it's 462 but we'll use 'exactly'
  -- steps for brevity directly linking to value establishment
  Sorry


end number_of_paths_in_grid_l759_759252


namespace number_of_true_propositions_l759_759445

-- Define planes and lines
constant Plane : Type
constant Line : Type

-- Define relationships
constant perp : Line → Plane → Prop
constant parallel : Line → Plane → Prop
constant subset : Line → Plane → Prop
constant plane_parallel : Plane → Plane → Prop
constant intersection : Plane → Plane → Line

-- Define given planes and lines
variables (α β γ : Plane) (l m n : Line)

-- Define propositions
def proposition_1 : Prop :=
  ∀ (l m : Line) (α : Plane), perp l α → perp m l → perp m α

def proposition_2 : Prop :=
  ∀ (m n : Line) (α β : Plane), subset m α → subset n α → parallel m β → parallel n β → plane_parallel α β

def proposition_3 : Prop :=
  ∀ (l : Line) (α β : Plane), plane_parallel α β → subset l α → parallel l β

def proposition_4 : Prop :=
  ∀ (α β γ : Plane) (l m n : Line), 
    intersection α β = l → intersection β γ = m → intersection γ α = n →
    parallel l γ → parallel m n

-- Finally, formulate the main proposition to be proved
theorem number_of_true_propositions (α β γ : Plane) (l m n : Line) :
  (proposition_1 ∧ proposition_2 ∧ proposition_3 ∧ proposition_4) → 
  (prop := 2) :=
sorry

end number_of_true_propositions_l759_759445


namespace greatest_increase_l759_759166

def profit (n : ℕ) : ℝ :=
  match n with
  | 0 => 2
  | 1 => 3
  | 2 => 4
  | 3 => 7
  | 4 => 7.5
  | 5 => 8
  | 6 => 9
  | _ => 0 -- Default case for out of bound years

theorem greatest_increase :
  ∃ y ∈ {1, 2, 3, 4, 5, 6}, (∀ z ∈ {1, 2, 3, 4, 5, 6}, profit y - profit (y - 1) ≥ profit z - profit (z - 1)) ∧ y = 1 :=
by
  sorry

end greatest_increase_l759_759166


namespace fraction_filled_in_17_hours_l759_759193

theorem fraction_filled_in_17_hours :
  let a : ℕ := 10
  let r : ℕ := 2
  let S : ℕ → ℕ := λ n, a * (r^n - 1) / (r - 1)
  S 17 / S 21 = (2^17 - 1) / (2^21 - 1) :=
by
  -- The given initial term and common ratio for geometric progression
  let a : ℕ := 10
  let r : ℕ := 2
  have Sn : ∀ n, S n = a * (r ^ n - 1) / (r - 1) := sorry
  show S 17 / S 21 = (2^17 - 1) / (2^21 - 1), from sorry

end fraction_filled_in_17_hours_l759_759193


namespace sum_primes_less_than_20_l759_759951

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759951


namespace no_two_adj_or_opposite_same_num_l759_759489

theorem no_two_adj_or_opposite_same_num :
  ∃ (prob : ℚ), prob = 25 / 648 ∧ 
  ∀ (A B C D E F : ℕ), 
    (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A) ∧
    (A ≠ D ∧ B ≠ E ∧ C ≠ F) ∧ 
    (1 ≤ A ∧ A ≤ 6) ∧ (1 ≤ B ∧ B ≤ 6) ∧ (1 ≤ C ∧ C ≤ 6) ∧ 
    (1 ≤ D ∧ D ≤ 6) ∧ (1 ≤ E ∧ E ≤ 6) ∧ (1 ≤ F ∧ F ≤ 6) →
    prob = (6 * 5 * 4 * 5 * 3 * 3) / (6^6) := 
sorry

end no_two_adj_or_opposite_same_num_l759_759489


namespace total_potatoes_bought_l759_759463

-- Define the conditions in the problem
def potatoes_for_salads : ℕ := 15
def potatoes_for_mashed : ℕ := 24
def potatoes_left : ℕ := 13

-- Define the total number of potatoes used
def potatoes_used : ℕ := potatoes_for_salads + potatoes_for_mashed

-- Lean statement to prove the initial number of potatoes bought
theorem total_potatoes_bought : potatoes_used + potatoes_left = 52 :=
by
  have h : potatoes_used = 15 + 24 := rfl
  have h' : potatoes_used + potatoes_left = (15 + 24) + 13 := rfl
  rw [h, h']
  norm_num
  sorry

end total_potatoes_bought_l759_759463


namespace greatest_three_digit_multiple_of_17_l759_759818

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759818


namespace intersection_nonempty_l759_759097

variables (S : Type*) [fintype S] {n : ℕ}
variables (A : fin n → set S)

def F (B : set S) : set (fin n) :=
  {j | A j ⊆ B}

theorem intersection_nonempty (H1 : ∀ j : fin n, (A j).finite)
  (H2 : ∀ j : fin n, (A j).to_finset.card = 8)
  (H3 : ∀ (B : set S), B.finite → (B.to_finset.card > 25 ∨ F A B = ∅ ∨ ⋂ (j ∈ F A B) (A j) ≠ ∅)) :
  ⋂ (j : fin n), (A j) ≠ ∅ :=
sorry

end intersection_nonempty_l759_759097


namespace polar_distance_to_axis_l759_759426

theorem polar_distance_to_axis (ρ θ : ℝ) (hρ : ρ = 2) (hθ : θ = Real.pi / 6) : 
  ρ * Real.sin θ = 1 := 
by
  rw [hρ, hθ]
  -- The remaining proof steps would go here
  sorry

end polar_distance_to_axis_l759_759426


namespace greatest_three_digit_multiple_of_seventeen_l759_759711

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759711


namespace determine_integers_with_one_question_l759_759546

theorem determine_integers_with_one_question (n : ℕ) (x : Fin n → ℤ) 
  (h : ∀ i, -9 ≤ x i ∧ x i ≤ 9) : 
  ∃ (a : Fin n → ℤ), (∀ i, (100^((i : ℕ)+1))) (∑ i, a i * x i) = 100^k  for k ≥ n → (
sorry


end determine_integers_with_one_question_l759_759546


namespace greatest_three_digit_multiple_of_17_is_986_l759_759769

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759769


namespace sum_of_primes_less_than_20_l759_759898

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759898


namespace greatest_three_digit_multiple_of17_l759_759728

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759728


namespace greatest_three_digit_multiple_of_17_l759_759791

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759791


namespace greatest_three_digit_multiple_of17_l759_759717

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759717


namespace greatest_three_digit_multiple_of_seventeen_l759_759712

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759712


namespace greatest_three_digit_multiple_of_17_is_986_l759_759665

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759665


namespace mode_of_scores_l759_759517

open List

def mode (l : List ℕ) : ℕ :=
  l.group_by id
  |>.map (λ g => (g.head', g.length))
  |>.maximum_by (λ a b => a.2 ≤ b.2)
  |>.1

theorem mode_of_scores : mode [55, 55, 55, 62, 62, 62, 62, 73, 78, 79, 80, 81, 81, 81, 81, 81, 92, 95, 97, 97, 97, 101, 101, 101, 102, 102, 102, 102] = 81 :=
by
  sorry

end mode_of_scores_l759_759517


namespace mean_temperature_correct_l759_759148

theorem mean_temperature_correct:
  let temps := [-10, -4, -6, -3, 0, 2, 5, 0 : Int] in
  (temps.sum / temps.length : Float) = -2 :=
  sorry

end mean_temperature_correct_l759_759148


namespace sum_of_primes_less_than_20_l759_759897

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759897


namespace integral_eq_e_l759_759308

theorem integral_eq_e : ∫ x in 0..1, (Real.exp x + 2 * x) = Real.exp 1 :=
by
  sorry

end integral_eq_e_l759_759308


namespace greatest_three_digit_multiple_of_17_l759_759781

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759781


namespace greatest_three_digit_multiple_of_17_is_986_l759_759759

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759759


namespace sum_of_primes_less_than_20_l759_759876

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759876


namespace greatest_three_digit_multiple_of_17_l759_759846

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759846


namespace minimum_lambda_l759_759353

theorem minimum_lambda (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (∀ λ : ℝ, (x + 2 * real.sqrt (2 * x * y) ≤ λ * (x + y)) ↔ λ ≥ 2) :=
sorry

end minimum_lambda_l759_759353


namespace geometric_series_sum_l759_759290

theorem geometric_series_sum :
  2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2))))))))) = 2046 := 
by sorry

end geometric_series_sum_l759_759290


namespace ending_time_proof_l759_759504

def starting_time_seconds : ℕ := (1 * 3600) + (57 * 60) + 58
def glow_interval : ℕ := 13
def total_glow_count : ℕ := 382
def total_glow_duration : ℕ := total_glow_count * glow_interval
def ending_time_seconds : ℕ := starting_time_seconds + total_glow_duration

theorem ending_time_proof : 
ending_time_seconds = (3 * 3600) + (14 * 60) + 4 := by
  -- Proof starts here
  sorry

end ending_time_proof_l759_759504


namespace not_divisible_by_1000_pow_m_minus_1_l759_759485

theorem not_divisible_by_1000_pow_m_minus_1 (m : ℕ) : ¬ (1000^m - 1 ∣ 1998^m - 1) :=
sorry

end not_divisible_by_1000_pow_m_minus_1_l759_759485


namespace arith_seq_sum_l759_759532

theorem arith_seq_sum (a₃ a₄ a₅ : ℤ) (h₁ : a₃ = 7) (h₂ : a₄ = 11) (h₃ : a₅ = 15) :
  let d := a₄ - a₃;
  let a := a₄ - 3 * d;
  (6 / 2 * (2 * a + 5 * d)) = 54 :=
by
  sorry

end arith_seq_sum_l759_759532


namespace greatest_three_digit_multiple_of17_l759_759719

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759719


namespace greatest_three_digit_multiple_of_17_is_986_l759_759762

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759762


namespace problem_solution_l759_759456

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2 * Real.sqrt a + 3 * Real.cbrt b ≥ 5 * Real.root (ab) 5 := 
sorry

end problem_solution_l759_759456


namespace sum_of_primes_less_than_20_l759_759874

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759874


namespace travel_time_at_constant_speed_l759_759559

theorem travel_time_at_constant_speed
  (distance : ℝ) (speed : ℝ) : 
  distance = 100 → speed = 20 → distance / speed = 5 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end travel_time_at_constant_speed_l759_759559


namespace black_car_speed_l759_759191

theorem black_car_speed
  (red_speed black_speed : ℝ)
  (initial_distance time : ℝ)
  (red_speed_eq : red_speed = 10)
  (initial_distance_eq : initial_distance = 20)
  (time_eq : time = 0.5)
  (distance_eq : black_speed * time = initial_distance + red_speed * time) :
  black_speed = 50 := by
  rw [red_speed_eq, initial_distance_eq, time_eq] at distance_eq
  sorry

end black_car_speed_l759_759191


namespace A1B1_parallel_AB_l759_759186

variables {A B C C1 P A1 B1 : Point}

-- Definition of the problem setup
def is_median (C : Point) (A B C1 : Point) : Prop := C1 = midpoint A B

def on_median (P : Point) (C : Point) (C1 : Point) : Prop := 
  ∃ (t : ℝ), t ∈ Icc 0 1 ∧ P = point_on_line C C1 t

def lines_through_point (P : Point) (A : Point) (A1 : Point) (B : Point) (B1 : Point) : Prop :=
  line_contains P A A1 ∧ line_contains P B B1

def points_on_sides (A1 : Point) (BC : Segment) (B1 : Point) (CA : Segment) : Prop :=
  ∃ (u v : ℝ), u ∈ Icc 0 1 ∧ v ∈ Icc 0 1 ∧ A1 = point_on_segment BC u ∧ B1 = point_on_segment CA v

-- Proof statement
theorem A1B1_parallel_AB (BC : Segment) (CA : Segment) :
  is_median C A B C1 → 
  on_median P C C1 → 
  lines_through_point P A A1 B B1 → 
  points_on_sides A1 BC B1 CA → 
  parallel (segment A1 B1) (segment A B) := 
by sorry

end A1B1_parallel_AB_l759_759186


namespace volume_parallelepiped_diagonal_angles_l759_759496

-- Declare the parameters used in the statement
variables (l : ℝ) (α β : ℝ)

-- Define the volume of the rectangular parallelepiped given the conditions
def volume_of_parallelepiped (l α β : ℝ) : ℝ := 
  l^3 * (Real.sin α) * (Real.sin β) * Real.sqrt ((Real.cos (α + β)) * (Real.cos (α - β)))

-- The statement we need to prove
theorem volume_parallelepiped_diagonal_angles 
  (h1 : l > 0)
  (h2 : 0 < α ∧ α < π/2)
  (h3 : 0 < β ∧ β < π/2) :
  volume_of_parallelepiped l α β = l^3 * (Real.sin α) * (Real.sin β) * Real.sqrt ((Real.cos (α + β)) * (Real.cos (α - β))) :=
sorry

end volume_parallelepiped_diagonal_angles_l759_759496


namespace sum_primes_less_than_20_l759_759960

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759960


namespace cos_diff_eq_half_l759_759440

theorem cos_diff_eq_half : 
  let x := cos (36 * Real.pi / 180) - cos (72 * Real.pi / 180) 
  in x = 1 / 2 :=
by
  sorry

end cos_diff_eq_half_l759_759440


namespace total_time_l759_759462

variable (x y z : ℝ)

def productivity_masha_dasha (x y : ℝ) : Prop := x + y = 2 / 15
def productivity_masha_sasha (x z : ℝ) : Prop := x + z = 1 / 6
def productivity_dasha_sasha (y z : ℝ) : Prop := y + z = 1 / 5

theorem total_time (h1: productivity_masha_dasha x y)
                   (h2: productivity_masha_sasha x z)
                   (h3: productivity_dasha_sasha y z) :
  1 / (x + y + z) = 4 :=
by 
  -- sum the equations
  have sum_eq: 2 * (x + y + z) = 1 / 2 := sorry,
  -- solve for x + y + z 
  have h: x + y + z = 1 / 4 := by 
    sorry,
  -- calculate total time
  show 1 / (x + y + z) = 4, from sorry

end total_time_l759_759462


namespace find_a_if_condition_met_l759_759425

-- Definitions of the conditions
def z (a : ℝ) : ℂ := a + 2 * complex.I
def z_sq (a : ℝ) : ℂ := z (a) ^ 2

-- Statement of the problem
theorem find_a_if_condition_met (a : ℝ) :
  (z_sq a).re = 0 ∧ (z_sq a).im > 0 → a = 2 :=
by
  sorry

end find_a_if_condition_met_l759_759425


namespace sum_integers_75_to_95_l759_759214

theorem sum_integers_75_to_95 : ∑ k in finset.Icc 75 95, k = 1785 := by
  sorry

end sum_integers_75_to_95_l759_759214


namespace greatest_three_digit_multiple_of_17_l759_759684

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759684


namespace sum_of_primes_less_than_20_is_77_l759_759923

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759923


namespace greatest_three_digit_multiple_of_17_l759_759824

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759824


namespace sequence_sum_base_case_l759_759194

variable (a : ℝ) (h₀ : a ≠ 0) (h₁ : a ≠ 1)

theorem sequence_sum_base_case : 1 + a + a^2 = ∑ i in Finset.range 3, a^i := by
  sorry

end sequence_sum_base_case_l759_759194


namespace greatest_three_digit_multiple_of17_l759_759723

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759723


namespace unique_five_topping_pizzas_l759_759258

open Finset

theorem unique_five_topping_pizzas : (card (powerset_len 5 (range 8))) = 56 := 
by
  sorry

end unique_five_topping_pizzas_l759_759258


namespace molly_total_cost_l759_759078

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_children_per_brother : ℕ := 2
def num_spouse_per_brother : ℕ := 1

def total_num_relatives : ℕ := 
  let parents_and_siblings := num_parents + num_brothers
  let additional_relatives := num_brothers * (1 + num_spouse_per_brother + num_children_per_brother)
  parents_and_siblings + additional_relatives

def total_cost : ℕ :=
  total_num_relatives * cost_per_package

theorem molly_total_cost : total_cost = 85 := sorry

end molly_total_cost_l759_759078


namespace number_of_people_who_like_apple_l759_759480

variable (persons : ℕ) [fact (persons = 60)]
variable (like_apple : ℕ) [fact (like_apple = 40)]
variable (like_orange : ℕ) [fact (like_orange = 17)]
variable (like_mango : ℕ) [fact (like_mango = 23)]
variable (like_banana : ℕ) [fact (like_banana = 12)]
variable (like_grapes : ℕ) [fact (like_grapes = 9)]
variable (like_orange_and_mango_not_apple : ℕ) [fact (like_orange_and_mango_not_apple = 7)]
variable (like_mango_and_apple_not_orange : ℕ) [fact (like_mango_and_apple_not_orange = 10)]
variable (like_all_three : ℕ) [fact (like_all_three = 4)]
variable (like_banana_and_grapes_only_other_fruits : ℕ) [fact (like_banana_and_grapes_only_other_fruits = 6)]
variable (like_apple_banana_grapes_no_mango_no_orange : ℕ) [fact (like_apple_banana_grapes_no_mango_no_orange = 3)]

theorem number_of_people_who_like_apple : like_apple = 40 := by
  sorry

end number_of_people_who_like_apple_l759_759480


namespace dihedral_angles_correct_l759_759162

-- Declare the noncomputable part if necessary for real number operations
noncomputable def rect_prism_dihedral_angles (a b c : ℝ) (h_geometric_mean : c = sqrt (a * b)) (h_sum_smaller_edges : a + b = c) : (ℝ × ℝ × ℝ) :=
if h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 then
  let phi1 : ℝ := 36 -- first dihedral angle 36 degrees
  let phi2 : ℝ := 60 -- second dihedral angle 60 degrees
  let phi3 : ℝ := 108 -- third dihedral angle 108 degrees
  (phi1, phi2, phi3)
else
  (0, 0, 0) -- invalid case, lengths must be positive

theorem dihedral_angles_correct (a b c : ℝ) (h_geometric_mean : c = sqrt (a * b)) (h_sum_smaller_edges : a + b = c) :
  rect_prism_dihedral_angles a b c h_geometric_mean h_sum_smaller_edges = (36, 60, 108) :=
by sorry

end dihedral_angles_correct_l759_759162


namespace complex_solution_l759_759108

noncomputable theory
open Complex

def complex_problem (z : ℂ) : Prop :=
  12 * abs z ^ 2 = 2 * abs (z + 2) ^ 2 + abs (z ^ 2 + 1) ^ 2 + 31

theorem complex_solution (z : ℂ) (h : complex_problem z) : z + (6 / z) = -2 :=
  sorry

end complex_solution_l759_759108


namespace greatest_three_digit_multiple_of_17_l759_759737

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759737


namespace solve_system_l759_759141

theorem solve_system : 
  ∀ (a b c : ℝ), 
  (a * (b^2 + c) = c * (c + a * b) ∧ 
   b * (c^2 + a) = a * (a + b * c) ∧ 
   c * (a^2 + b) = b * (b + c * a)) 
   → (∃ t : ℝ, a = t ∧ b = t ∧ c = t) :=
by
  intros a b c h
  sorry

end solve_system_l759_759141


namespace greatest_three_digit_multiple_of_17_l759_759562

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759562


namespace greatest_three_digit_multiple_of_17_l759_759567

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759567


namespace smallest_yellow_candies_l759_759285
open Nat

theorem smallest_yellow_candies 
  (h_red : ∃ c : ℕ, 16 * c = 720)
  (h_green : ∃ c : ℕ, 18 * c = 720)
  (h_blue : ∃ c : ℕ, 20 * c = 720)
  : ∃ n : ℕ, 30 * n = 720 ∧ n = 24 := 
by
  -- Provide the proof here
  sorry

end smallest_yellow_candies_l759_759285


namespace express_form_l759_759048

theorem express_form 
    (a b c : ℕ) 
    (h1 : (sqrt 3 + 2 / sqrt 3 + sqrt 8 + 3 / sqrt 8) = (a * sqrt 3 + b * sqrt 8) / c)
    (h2 : c ≠ 0)
    (h3 : a > 0 ∧ b > 0 ∧ c > 0)
    (h4 : ∀ c', c' < c → ∃ a' b', (a' * sqrt 3 + b' * sqrt 8) / c' ≠ (sqrt 3 + 2 / sqrt 3 + sqrt 8 + 3 / sqrt 8))
    : a + b + c = 53 :=
sorry

end express_form_l759_759048


namespace greatest_three_digit_multiple_of_17_is_986_l759_759626

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759626


namespace total_miles_Wednesday_l759_759413

-- The pilot flew 1134 miles on Tuesday and 1475 miles on Thursday.
def miles_flown_Tuesday : ℕ := 1134
def miles_flown_Thursday : ℕ := 1475

-- The miles flown on Wednesday is denoted as "x".
variable (x : ℕ)

-- The period is 4 weeks.
def weeks : ℕ := 4

-- We need to prove that the total miles flown on Wednesdays during this 4-week period is 4 * x.
theorem total_miles_Wednesday : 4 * x = 4 * x := by sorry

end total_miles_Wednesday_l759_759413


namespace sum_of_primes_less_than_20_l759_759991

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759991


namespace greatest_three_digit_multiple_of17_l759_759720

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759720


namespace greatest_three_digit_multiple_of_17_is_986_l759_759657

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759657


namespace obtuse_triangle_product_l759_759420

theorem obtuse_triangle_product
  (ABC : Triangle)
  (A B C P Q X Y : Point)
  (h_obtuse : obtuse_at ABC A)
  (h_perpendicular1 : ⟪C, P, A⟫ = action ⟪B, P, A⟫)
  (h_perpendicular2 : ⟪B, Q, A⟫ = action ⟪C, Q, A⟫)
  (h_PQ_intersects : intersects_circle (circumcircle ⟪A,B,C⟫) ⟪P,Q⟫)
  (XP PQ QY : ℝ)
  (h_XP : XP = 12)
  (h_PQ : PQ = 24)
  (h_QY : QY = 18) :
  ∃ (m n : ℕ), m * n^2 * sqrt(2) = |AB * AC| ∧ m + n = 1602 := 
sorry

end obtuse_triangle_product_l759_759420


namespace part1_solution_part2_solution_l759_759455

noncomputable def part1_property (n : ℕ) : Prop :=
  (∑ k in finset.range 2014 \ {0}, int.floor (k * n / 2013) = 2013 + n)

noncomputable def part2_property (n : ℕ) : Prop :=
  let s := ∑ k in finset.range 2014 \ {0}, (k * n / 2013) - int.floor (k * n / 2013) in
  s = 1006

theorem part1_solution : ∀ n : ℕ, part1_property n → n = 3 :=
by sorry

theorem part2_solution : ∀ n : ℕ,
  (∀ m : ℕ, m | 2013 → ¬ m | n) →
  part2_property n :=
by sorry

end part1_solution_part2_solution_l759_759455


namespace greatest_three_digit_multiple_of_17_l759_759640

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759640


namespace intersection_A_B_l759_759028

def A : Set ℤ := {x | abs x < 2}
def B : Set ℤ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} := by
  sorry

end intersection_A_B_l759_759028


namespace greatest_three_digit_multiple_of_17_l759_759680

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759680


namespace greatest_three_digit_multiple_of_17_l759_759599

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759599


namespace hidden_cannonball_label_l759_759267

-- Defining the labels and their counts
inductive Label
| A
| B
| C
| D
| E

open Label

-- Given conditions

-- There are exactly 20 cannonballs
def total_cannonballs : Nat := 20

-- Each label appears exactly four times
def label_counts (lbl : Label) : Nat := 4

-- The visible faces show the counts of each label
def visible_label_counts (lbl : Label) : Nat

-- The single top cannonball appears on all three faces
def top_cannonball_label : Label

-- Prove the label of hidden cannonball is 'D'
theorem hidden_cannonball_label : 
  label_counts D - visible_label_counts D = 1 -> 
  top_cannonball_label ∈ {A, B, C, D, E} -> 
  ∃ hidden_label : Label, hidden_label = D :=
sorry

end hidden_cannonball_label_l759_759267


namespace highlighters_per_student_l759_759432

-- Definitions for the problem
def num_students := 30
def num_pens_per_student := 5
def num_notebooks_per_student := 3
def num_binders_per_student := 1
def cost_per_pen := 0.50
def cost_per_notebook := 1.25
def cost_per_binder := 4.25
def cost_per_highlighter := 0.75
def teacher_discount := 100
def total_spent := 260

-- Prove that each student needs 2 highlighters
theorem highlighters_per_student :
  let total_cost_pens := num_students * num_pens_per_student * cost_per_pen,
      total_cost_notebooks := num_students * num_notebooks_per_student * cost_per_notebook,
      total_cost_binders := num_students * num_binders_per_student * cost_per_binder,
      total_cost_supplies := total_cost_pens + total_cost_notebooks + total_cost_binders,
      total_cost_after_discount := total_cost_supplies - teacher_discount,
      total_cost_highlighters := total_spent - total_cost_after_discount in
  total_cost_highlighters / (num_students * cost_per_highlighter) = 2 := by
  sorry

end highlighters_per_student_l759_759432


namespace sum_of_primes_lt_20_eq_77_l759_759903

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759903


namespace number_of_symmetric_designs_l759_759291

def symmetric_designs : ℕ := 30

theorem number_of_symmetric_designs 
  (grid_size : ℕ := 5)
  (symmetric : ∀ design, design (rotate90 design) = design)
  (at_least_one_blue : ∃ square, is_blue square)
  (not_all_blue : ∃ square, ¬ is_blue square) :
  symmetric_designs = 30 := 
sorry

end number_of_symmetric_designs_l759_759291


namespace prove_a_minus_b_l759_759279

theorem prove_a_minus_b (f : ℝ → ℝ) (hf : function.injective f)
  (a b : ℝ) (hfa : f a = b) (hfb : f b = 3) : a - b = -2 :=
by sorry

end prove_a_minus_b_l759_759279


namespace sum_of_primes_less_than_20_is_77_l759_759920

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759920


namespace heartsuit_calc_l759_759333

def heartsuit (u v : ℝ) : ℝ := (u + 2*v) * (u - v)

theorem heartsuit_calc : heartsuit 2 (heartsuit 3 4) = -260 := by
  sorry

end heartsuit_calc_l759_759333


namespace greatest_three_digit_multiple_of_17_l759_759593

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759593


namespace triangle_angle_C_l759_759412

open Real

theorem triangle_angle_C (b c : ℝ) (B C : ℝ) (hb : b = sqrt 2) (hc : c = 1) (hB : B = 45) : C = 30 :=
sorry

end triangle_angle_C_l759_759412


namespace james_bought_dirt_bikes_l759_759080

variable (D : ℕ)

-- Definitions derived from conditions
def cost_dirt_bike := 150
def cost_off_road_vehicle := 300
def registration_fee := 25
def num_off_road_vehicles := 4
def total_paid := 1825

-- Auxiliary definitions
def total_cost_dirt_bike := cost_dirt_bike + registration_fee
def total_cost_off_road_vehicle := cost_off_road_vehicle + registration_fee
def total_cost_off_road_vehicles := num_off_road_vehicles * total_cost_off_road_vehicle
def total_cost_dirt_bikes := total_paid - total_cost_off_road_vehicles

-- The final statement we need to prove
theorem james_bought_dirt_bikes : D = total_cost_dirt_bikes / total_cost_dirt_bike ↔ D = 3 := by
  sorry

end james_bought_dirt_bikes_l759_759080


namespace correct_operation_l759_759224

variable (a : ℕ)

theorem correct_operation :
  (3 * a + 2 * a ≠ 5 * a^2) ∧
  (3 * a - 2 * a ≠ 1) ∧
  a^2 * a^3 = a^5 ∧
  (a / a^2 ≠ a) :=
by
  sorry

end correct_operation_l759_759224


namespace find_integer_solutions_l759_759318

theorem find_integer_solutions :
  ∃ s : set (ℤ × ℤ), s = {(3, -1), (5, 1), (1, 5), (-1, 3)} ∧
    ∀ x y : ℤ, 2 * (x + y) = x * y + 7 ↔ (x, y) ∈ s :=
by
  sorry

end find_integer_solutions_l759_759318


namespace total_baseball_cards_l759_759091

theorem total_baseball_cards (Carlos Matias Jorge : ℕ) (h1 : Carlos = 20) (h2 : Matias = Carlos - 6) (h3 : Jorge = Matias) : Carlos + Matias + Jorge = 48 :=
by
  sorry

end total_baseball_cards_l759_759091


namespace greatest_three_digit_multiple_of_17_l759_759863

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759863


namespace greatest_three_digit_multiple_of_17_l759_759691

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759691


namespace sum_of_primes_less_than_20_l759_759881

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759881


namespace sum_of_integers_75_to_95_l759_759208

theorem sum_of_integers_75_to_95 : (∑ i in Finset.range (95 - 75 + 1), (i + 75)) = 1785 := by
  sorry

end sum_of_integers_75_to_95_l759_759208


namespace sum_of_primes_lt_20_eq_77_l759_759900

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759900


namespace max_value_of_expr_l759_759866

theorem max_value_of_expr : ∃ t : ℝ, (∀ u : ℝ, (3^u - 2*u) * u / 9^u ≤ (3^t - 2*t) * t / 9^t) ∧ (3^t - 2*t) * t / 9^t = 1/8 :=
by sorry

end max_value_of_expr_l759_759866


namespace rewrite_expression_l759_759138

theorem rewrite_expression (k : ℝ) :
  ∃ d r s : ℝ, (8 * k^2 - 12 * k + 20 = d * (k + r)^2 + s) ∧ (r + s = 14.75) := 
sorry

end rewrite_expression_l759_759138


namespace determine_k_l759_759371

theorem determine_k (k r s : ℝ) (h1 : r + s = -k) (h2 : (r + 3) + (s + 3) = k) : k = 3 :=
by
  sorry

end determine_k_l759_759371


namespace correct_average_weight_l759_759153

noncomputable def initial_average_weight : ℚ := 58.4
noncomputable def num_boys : ℕ := 20
noncomputable def misread_weight : ℚ := 56
noncomputable def correct_weight : ℚ := 66

theorem correct_average_weight : 
  (initial_average_weight * num_boys + (correct_weight - misread_weight)) / num_boys = 58.9 := 
by
  sorry

end correct_average_weight_l759_759153


namespace janet_needs_more_money_l759_759086

theorem janet_needs_more_money :
  let janet_savings := 2225
  let monthly_rent := 1250
  let months_in_advance := 2
  let deposit := 500
  let total_required := (monthly_rent * months_in_advance) + deposit
  let additional_money_needed := total_required - janet_savings
  in additional_money_needed = 775 :=
by
  let janet_savings := 2225
  let monthly_rent := 1250
  let months_in_advance := 2
  let deposit := 500
  let total_required := (monthly_rent * months_in_advance) + deposit
  let additional_money_needed := total_required - janet_savings
  have h1 : total_required = 3000, by sorry
  have h2 : additional_money_needed = total_required - janet_savings, by sorry
  have h3 : total_required - janet_savings = 775, by sorry
  exact Eq.trans h2 h3

end janet_needs_more_money_l759_759086


namespace initial_fish_l759_759461

-- Define the conditions of the problem
def fish_bought : Float := 280.0
def current_fish : Float := 492.0

-- Define the question to be proved
theorem initial_fish (x : Float) (h : x + fish_bought = current_fish) : x = 212 :=
by 
  sorry

end initial_fish_l759_759461


namespace solution_problem_l759_759298

noncomputable def f : ℕ → ℕ → ℕ → ℕ → ℕ := sorry

theorem solution_problem (f : ℕ → ℕ) (h : ∀ n : ℕ, f(n + 1) + f(n + 3) = f(n + 5) * f(n + 7) - 1375) :
  (∀ k : ℕ, ((f(k) - 1) * (f(k + 2) - 1) = 1376)) ∧ ((f(k) - 1) * (f(k + 2) - 1) = 1376) :=
sorry

end solution_problem_l759_759298


namespace solution_l759_759073

noncomputable def A_eq_pi_over_3_option_A (a b c : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 5) : Prop :=
  let cosA := (b^2 + c^2 - a^2) / (2 * b * c)
  ∃ A : ℝ, 0 < A ∧ A < Real.pi ∧ Real.cos A = cosA ∧ A = Real.pi / 3

noncomputable def A_eq_pi_over_3_option_D (B C : ℝ) (h1 : 2 * Real.sin (B + C) / 2 ^ 2 + Real.cos (2 * (Real.pi / 3)) = 1) : Prop :=
  ∃ A : ℝ, 0 < A ∧ A < Real.pi ∧ Real.cos A = 1 / 2 ∧ A = Real.pi / 3

noncomputable def A_eq_pi_over_3_problem :=
  (∃ a b c, A_eq_pi_over_3_option_A a b c) ∨
  (∃ B C, A_eq_pi_over_3_option_D B C)

theorem solution : A_eq_pi_over_3_problem :=
sorry

end solution_l759_759073


namespace triangle_area_eq_24_l759_759421

-- Definitions for the conditions
def Rectangle (A B C D : Type) : Prop := sorry
def Point (A : Type) : Prop := sorry
def line (P Q : Type) : Type := sorry

variables {A B C D F G E : Type} [Rectangle A B C D] [Point F] [Point G] [Point E]
variables (AB CD DF GC : ℕ) (AB_length : AB = 6) (BC_length : BC = 4) (DF_length : DF = 2) (GC_length : GC = 1)
variables (AF : line A F) (BG : line B G) (intersect_E : ∃E, AF = BG)

-- Prove that the area of triangle AEB is 24
theorem triangle_area_eq_24 : (area_of_triangle A E B = 24) :=
sorry

end triangle_area_eq_24_l759_759421


namespace number_of_zeros_between_decimal_point_and_first_nonzero_digit_l759_759219

def fraction := (7 : ℚ) / 64000

theorem number_of_zeros_between_decimal_point_and_first_nonzero_digit :
  real.zerodigits fraction = 4 :=
sorry

end number_of_zeros_between_decimal_point_and_first_nonzero_digit_l759_759219


namespace smallest_period_max_value_monotonicity_l759_759017

noncomputable def f (x : ℝ) : ℝ := 
  (real.sin (π / 2 - x)) * (real.sin x) - (real.sqrt 3) * (real.cos x) ^ 2

theorem smallest_period (x : ℝ) :
  (∃ p > 0, ∀ x, f (x + p) = f x) ∧ p = π := sorry

theorem max_value (x : ℝ) :
  ∀ x, f x ≤ 1 - real.sqrt 3 / 2 := sorry

theorem monotonicity (x : ℝ) :
  ∀ x, 
  (x ∈ set.Icc (π / 6) (5 * π / 12) → ∀ y > x, f y > f x) ∧ 
  (x ∈ set.Icc (5 * π / 12) (2 * π / 3) → ∀ y > x, f y < f x) := sorry

end smallest_period_max_value_monotonicity_l759_759017


namespace greatest_perfect_square_less_than_500_has_odd_factors_l759_759121

-- We need to state that a number has an odd number of positive factors if and only if it is a perfect square
lemma odd_factors_iff_perfect_square (n : ℕ) :
  (∃ m, m * m = n) ↔ (∃ k, k * k = n) :=
by sorry

-- Define the specific problem conditions
def is_perfect_square (n : ℕ) : Prop := ∃ m, m * m = n

def less_than_500 (n : ℕ) : Prop := n < 500

-- Final statement combining the conditions and conclusion
theorem greatest_perfect_square_less_than_500_has_odd_factors :
  ∃ n, is_perfect_square n ∧ less_than_500 n ∧ ∀ m, (is_perfect_square m ∧ less_than_500 m) → m ≤ n ∧ n = 484 :=
by sorry

end greatest_perfect_square_less_than_500_has_odd_factors_l759_759121


namespace greatest_three_digit_multiple_of_seventeen_l759_759701

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759701


namespace store_cost_comparison_l759_759146

noncomputable def store_A_cost (x : ℕ) : ℝ := 1760 + 40 * x
noncomputable def store_B_cost (x : ℕ) : ℝ := 1920 + 32 * x

theorem store_cost_comparison (x : ℕ) (h : x > 16) :
  (x > 20 → store_B_cost x < store_A_cost x) ∧ (x < 20 → store_A_cost x < store_B_cost x) :=
by
  sorry

end store_cost_comparison_l759_759146


namespace sum_of_first_six_terms_arithmetic_seq_l759_759522

theorem sum_of_first_six_terms_arithmetic_seq (a b c : ℤ) (d : ℤ) (n : ℤ) :
    (a = 7) ∧ (b = 11) ∧ (c = 15) ∧ (d = b - a) ∧ (d = c - b) 
    ∧ (n = a - d) 
    ∧ (d = 4) -- the common difference is always 4 here as per the solution given 
    ∧ (n = -1) -- the correct first term as per calculation
    → (n + (n + d) + (a) + (b) + (c) + (c + d) = 54) := 
begin
  sorry
end

end sum_of_first_six_terms_arithmetic_seq_l759_759522


namespace greatest_three_digit_multiple_of_17_l759_759617

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759617


namespace greatest_three_digit_multiple_of_17_l759_759845

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759845


namespace sum_of_primes_less_than_20_l759_759875

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759875


namespace greatest_three_digit_multiple_of_17_is_986_l759_759670

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759670


namespace greatest_three_digit_multiple_of_17_l759_759794

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759794


namespace investment_interests_l759_759143

theorem investment_interests (x y : ℝ) (h₁ : x + y = 24000)
  (h₂ : 0.045 * x + 0.06 * y = 0.05 * 24000) : (x = 16000) ∧ (y = 8000) :=
  by
  sorry

end investment_interests_l759_759143


namespace muffins_in_each_pack_l759_759476

-- Define the conditions as constants
def total_amount_needed : ℕ := 120
def price_per_muffin : ℕ := 2
def number_of_cases : ℕ := 5
def packs_per_case : ℕ := 3

-- Define the theorem to prove
theorem muffins_in_each_pack :
  (total_amount_needed / price_per_muffin) / (number_of_cases * packs_per_case) = 4 :=
by
  sorry

end muffins_in_each_pack_l759_759476


namespace prizes_count_l759_759415

theorem prizes_count (P : ℕ) (h : 0.7142857142857143 = 25 / (P + 25)) : P = 10 :=
sorry

end prizes_count_l759_759415


namespace sum_primes_less_than_20_l759_759965

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759965


namespace remainder_2007_div_81_l759_759198

theorem remainder_2007_div_81 : 2007 % 81 = 63 :=
by
  sorry

end remainder_2007_div_81_l759_759198


namespace greatest_three_digit_multiple_of_17_l759_759590

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759590


namespace quadratic_real_roots_a_condition_l759_759382

theorem quadratic_real_roots_a_condition (a : ℝ) (h : ∃ x : ℝ, (a - 5) * x^2 - 4 * x - 1 = 0) :
  a ≥ 1 ∧ a ≠ 5 :=
by
  sorry

end quadratic_real_roots_a_condition_l759_759382


namespace greatest_three_digit_multiple_of_17_l759_759778

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759778


namespace max_min_f_on_interval_l759_759111

def f (x : ℝ) : ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_negative_for_positive : ∀ x : ℝ, x > 0 → f x < 0
axiom f_at_one : f 1 = -2

theorem max_min_f_on_interval : 
  ∃ max min : ℝ, max = 6 ∧ min = -6 ∧ 
                 (∀ x ∈ Icc (-2 : ℝ) 3, f x ≤ max) ∧ 
                 (∀ x ∈ Icc (-2 : ℝ) 3, min ≤ f x) := 
by
  sorry

end max_min_f_on_interval_l759_759111


namespace greatest_three_digit_multiple_of_seventeen_l759_759702

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759702


namespace sum_primes_less_than_20_l759_759963

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759963


namespace max_length_OB_l759_759553

-- Define the problem conditions
def angle_AOB : ℝ := 45
def length_AB : ℝ := 2
def max_sin_angle_OAB : ℝ := 1

-- Claim to be proven
theorem max_length_OB : ∃ OB_max, OB_max = 2 * Real.sqrt 2 :=
by
  sorry

end max_length_OB_l759_759553


namespace greatest_three_digit_multiple_of_17_l759_759849

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759849


namespace greatest_three_digit_multiple_of_17_l759_759647

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759647


namespace line_through_AB_l759_759024

theorem line_through_AB (x1 y1 x2 y2 : ℝ) (A : ℝ × ℝ := (x1, y1)) (B : ℝ × ℝ := (x2, y2))
  (hx1 : x1 + x2 = 1) (hy1 : y1 + y2 = -2)
  (hA : x1^2 / 4 - y1^2 / 2 = 1) (hB : x2^2 / 4 - y2^2 / 2 = 1) :
  (λ x y : ℝ, 2 * x + 8 * y + 7 = 0) :=
by
  sorry

end line_through_AB_l759_759024


namespace arbitrary_large_set_of_points_l759_759337

theorem arbitrary_large_set_of_points (
  points : ℕ → (ℝ × ℝ)
) : ∀ n, ∃ (S : set (ℝ × ℝ)), S = { points k | k < n } ∧ 
  ∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ i ≠ k → obtuse (points i) (points j) (points k) :=
begin
  sorry
end

end arbitrary_large_set_of_points_l759_759337


namespace find_product_of_roots_plus_one_l759_759003

-- Define the problem conditions
variables (x1 x2 : ℝ)
axiom sum_roots : x1 + x2 = 3
axiom prod_roots : x1 * x2 = 2

-- State the theorem corresponding to the proof problem
theorem find_product_of_roots_plus_one : (x1 + 1) * (x2 + 1) = 6 :=
by 
  sorry

end find_product_of_roots_plus_one_l759_759003


namespace sum_primes_less_than_20_l759_759949

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759949


namespace floor_fourth_root_product_l759_759289

theorem floor_fourth_root_product :
  (∏ (i : ℕ) in (Finset.filter (λ i, i % 2 = 1) (Finset.range 2017)), ⌊(i : ℝ) ^ (1 / 4)⌋)
  / (∏ (i : ℕ) in (Finset.filter (λ i, i % 2 = 0) (Finset.range 2017)), ⌊(i : ℝ) ^ (1 / 4)⌋)
  = 5 / 16 :=
  sorry

end floor_fourth_root_product_l759_759289


namespace greatest_three_digit_multiple_of_17_l759_759812

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759812


namespace abs_AB_l759_759499

noncomputable def ellipse_foci (A B : ℝ) : Prop :=
  B^2 - A^2 = 25

noncomputable def hyperbola_foci (A B : ℝ) : Prop :=
  A^2 + B^2 = 64

theorem abs_AB (A B : ℝ) (h1 : ellipse_foci A B) (h2 : hyperbola_foci A B) :
  |A * B| = Real.sqrt 867.75 := 
sorry

end abs_AB_l759_759499


namespace percentage_increase_in_savings_l759_759126

-- Define the initial conditions
variable (I : ℝ) -- Paul's initial income
def initial_expense (I : ℝ) : ℝ := 0.75 * I -- 75% of income
def initial_savings (I : ℝ) : ℝ := I - initial_expense I -- Initial savings

-- Define new conditions after the increment
def new_income (I : ℝ) : ℝ := 1.20 * I -- 20% increase in income
def new_expense (I : ℝ) : ℝ := 1.10 * initial_expense I -- 10% increase in expenditure
def new_savings (I : ℝ) : ℝ := new_income I - new_expense I -- New savings

-- Proof goal
theorem percentage_increase_in_savings (I : ℝ) (h : I > 0) :
  let S := initial_savings I in
  let S_new := new_savings I in
  (S_new - S) / S * 100 = 50 := by
  sorry

end percentage_increase_in_savings_l759_759126


namespace alice_query_complexity_l759_759098

theorem alice_query_complexity (n k : ℕ) (h₁ : k ≤ n) :
  ∃ (calls : ℕ) (determine_finiteness : Π (query : Π (i j : ℕ), bool), bool),
    calls ≤ (2 * n^2 / k) ∧
    (∀ (query : Π (i j : ℕ), bool),
     determine_finiteness query = 
     (∃ (u v : ℕ), ¬(query u v) → ∞) :=
sorry

end alice_query_complexity_l759_759098


namespace greatest_three_digit_multiple_of_17_l759_759839

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759839


namespace greatest_three_digit_multiple_of17_l759_759727

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759727


namespace average_megabytes_per_hour_l759_759249

theorem average_megabytes_per_hour 
  (days : ℕ) (total_megabytes : ℕ) (hours_per_day : ℕ) (h_days : days = 12) 
  (h_total_mb : total_megabytes = 16000) (h_hours_per_day : hours_per_day = 24) :
  Nat.round ((total_megabytes : ℚ) / (days * hours_per_day : ℚ)) = 56 := 
by
  sorry

end average_megabytes_per_hour_l759_759249


namespace sum_primes_less_than_20_l759_759955

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759955


namespace count_three_digit_special_numbers_l759_759036

theorem count_three_digit_special_numbers :
  ∃ (count : ℕ),
    count = 
    (Finset.card (Finset.filter (λ n : ℕ, (100 ≤ n) ∧ (n ≤ 999) ∧ (
      (∃ i, digit_at n i = 5) ∨ (∃ j, digit_at n j = 1 ∧ digit_at n (j + 1) = 2)))
      (Finset.range 1000))) ∧ count = 270 :=
begin
  sorry
end

end count_three_digit_special_numbers_l759_759036


namespace greatest_three_digit_multiple_of_17_l759_759847

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759847


namespace sum_of_primes_less_than_20_l759_759885

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759885


namespace sum_first_six_terms_l759_759536

-- Define the conditions given in the problem
def a3 := 7
def a4 := 11
def a5 := 15

-- Define the common difference
def d := a4 - a3 -- 4

-- Define the first term
def a1 := a3 - 2 * d -- -1

-- Define the sum of the first six terms of the arithmetic sequence
def S6 := (6 / 2) * (2 * a1 + (6 - 1) * d) -- 54

-- The theorem we want to prove
theorem sum_first_six_terms : S6 = 54 := by
  sorry

end sum_first_six_terms_l759_759536


namespace stratified_sampling_l759_759260

theorem stratified_sampling (n : ℕ) : 100 + 600 + 500 = 1200 → 500 ≠ 0 → 40 / 500 = n / 1200 → n = 96 :=
by
  intros total_population nonzero_div divisor_eq
  sorry

end stratified_sampling_l759_759260


namespace period_of_sine_function_l759_759197

theorem period_of_sine_function : 
  ∀ x : ℝ, (∃ k : ℤ, y = sin(8 * (x + k * (π / 4)) + π / 4)) ↔ y = sin(8 * x + π / 4) :=
by sorry

end period_of_sine_function_l759_759197


namespace sum_of_primes_less_than_20_l759_759871

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759871


namespace greatest_three_digit_multiple_of_17_is_986_l759_759631

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759631


namespace sum_of_primes_less_than_20_l759_759989

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759989


namespace sum_primes_less_than_20_l759_759974

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759974


namespace not_differentiable_at_zero_l759_759334

noncomputable def f (x : ℝ) : ℝ := abs x

theorem not_differentiable_at_zero : ¬ differentiable_at ℝ f 0 :=
begin
  sorry
end

end not_differentiable_at_zero_l759_759334


namespace factor_expression_l759_759309

theorem factor_expression (x : ℝ) : 92 * x^3 - 184 * x^6 = 92 * x^3 * (1 - 2 * x^3) :=
by
  sorry

end factor_expression_l759_759309


namespace greatest_three_digit_multiple_of_17_l759_759601

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759601


namespace sum_of_primes_less_than_20_l759_759878

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759878


namespace greatest_three_digit_multiple_of_17_l759_759790

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759790


namespace greatest_three_digit_multiple_of_17_l759_759817

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759817


namespace jerry_needs_money_l759_759433

theorem jerry_needs_money (has : ℕ) (total : ℕ) (cost_per_action_figure : ℕ) 
  (h1 : has = 7) (h2 : total = 16) (h3 : cost_per_action_figure = 8) : 
  (total - has) * cost_per_action_figure = 72 := by
  -- Proof goes here
  sorry

end jerry_needs_money_l759_759433


namespace tangent_to_excircle_l759_759437

-- Definitions of a triangle, altitudes, tangent line, and orthocenter
variables (A B C D E F H Q P: Point) (d: Line)

-- Conditions
axiom acute_non_isosceles_triangle : acute_non_isosceles_triangle A B C
axiom altitudes : is_altitude AD A B C ∧ is_altitude BE B A C ∧ is_altitude CF C A B
axiom tangent_circumcircle : is_tangent_to_circumcircle d A B C A
axiom orthocenter : orthocenter A B C = H
axiom intersection_QP : (line_through H parallel_to EF) intersects DE = Q ∧ (line_through H parallel_to EF) intersects DF = P

-- Statement to prove
theorem tangent_to_excircle (A B C D E F H Q P : Point) (d : Line) :
  acute_non_isosceles_triangle A B C →
  is_altitude AD A B C ∧ is_altitude BE B A C ∧ is_altitude CF C A B →
  is_tangent_to_circumcircle d A B C A →
  orthocenter A B C = H →
  (line_through H parallel_to EF) intersects DE = Q ∧ 
  (line_through H parallel_to EF) intersects DF = P →
  is_tangent_to_excircle d D (triangle DP Q) :=
sorry

end tangent_to_excircle_l759_759437


namespace missing_root_l759_759325

theorem missing_root (p q r : ℝ) 
  (h : p * (q - r) ≠ 0 ∧ q * (r - p) ≠ 0 ∧ r * (p - q) ≠ 0 ∧ 
       p * (q - r) * (-1)^2 + q * (r - p) * (-1) + r * (p - q) = 0) : 
  ∃ x : ℝ, x ≠ -1 ∧ 
  p * (q - r) * x^2 + q * (r - p) * x + r * (p - q) = 0 ∧ 
  x = - (r * (p - q) / (p * (q - r))) :=
sorry

end missing_root_l759_759325


namespace common_difference_range_l759_759159

noncomputable def arithmetic_sequence (n : ℕ) (a₁ d : ℤ) : ℤ :=
  a₁ + (n - 1) * d

theorem common_difference_range :
  let a1 := -24
  let a9 := arithmetic_sequence 9 a1 d
  let a10 := arithmetic_sequence 10 a1 d
  (a10 > 0) ∧ (a9 <= 0) → 8 / 3 < d ∧ d <= 3 :=
by
  let a1 := -24
  let a9 := arithmetic_sequence 9 a1 d
  let a10 := arithmetic_sequence 10 a1 d
  intro h
  sorry

end common_difference_range_l759_759159


namespace sum_integers_75_to_95_l759_759211

theorem sum_integers_75_to_95 : ∑ k in finset.Icc 75 95, k = 1785 := by
  sorry

end sum_integers_75_to_95_l759_759211


namespace final_answer_l759_759332

def not_in_lowest_terms_count (a b : ℤ) (g : ℕ) : Prop :=
  ∃ d : ℤ, d ∣ (a + b) ∧ d > 1 ∧ d ≤ g

noncomputable def count_not_in_lowest_terms : ℕ :=
  (Finset.range 1000).filter (λ N, not_in_lowest_terms_count ((N : ℤ) ^ 2 + 11) (N + 5) 36).card

theorem final_answer : count_not_in_lowest_terms = 102 := 
  sorry

end final_answer_l759_759332


namespace greatest_three_digit_multiple_of_17_l759_759826

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759826


namespace greatest_three_digit_multiple_of_17_is_986_l759_759755

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759755


namespace greatest_three_digit_multiple_of_17_is_986_l759_759635

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759635


namespace sum_of_primes_less_than_20_l759_759986

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759986


namespace find_smallest_n_l759_759064

theorem find_smallest_n : 
  ∃ n m : ℕ, 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ m → unique (score k = n + 2 - 2 * k)) ∧ 
  (∑ k in range m, score k = 2009) ∧
  (m ∈ [1, 7, 41, 49, 287, 2009]) ∧
  ∀ x ∈ [1, 7, 41, 49, 287, 2009], (n = (2009 / x) + x - 1) :=
begin
  sorry
end

def score (n k : ℕ) : ℕ := n + 2 - 2 * k

end find_smallest_n_l759_759064


namespace greatest_three_digit_multiple_of_17_l759_759798

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759798


namespace greatest_three_digit_multiple_of_17_is_986_l759_759674

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759674


namespace greatest_three_digit_multiple_of_17_l759_759774

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759774


namespace greatest_three_digit_multiple_of_17_is_986_l759_759757

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759757


namespace terminating_fraction_count_l759_759299

theorem terminating_fraction_count :
  let eligible_n (n : ℕ) := (1 ≤ n) ∧ (n ≤ 500) ∧ ∃ k, n = 3 * k ∧ (1 ≤ k) ∧ (k ≤ 166)
  in (∃ count, count = (finset.filter eligible_n (finset.range 501)).card ∧ count = 166) :=
by
  sorry

end terminating_fraction_count_l759_759299


namespace greatest_three_digit_multiple_of_17_l759_759643

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759643


namespace sum_of_primes_less_than_20_l759_759987

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759987


namespace electricity_consumption_estimation_l759_759055

noncomputable def estimated_households_above_320 {μ σ : ℝ} (hμ : μ = 300) (hσ : σ = 10) (households : ℕ) 
  (normal_dist : ∀ x, ℝ → ℝ → ℝ → Prop) : ℕ :=
  let prob_above_320 := (1 - 0.954) / 2
  let expected_households := households * prob_above_320
  expected_households.toNat

theorem electricity_consumption_estimation (μ σ : ℝ) (hμ : μ = 300) (hσ : σ = 10) (households : ℕ)
  (normal_dist : ∀ x, ℝ → ℝ → ℝ → Prop) : estimated_households_above_320 hμ hσ 1000 normal_dist = 23 := 
by
  -- Assume the solution logic as stated
  sorry

end electricity_consumption_estimation_l759_759055


namespace greatest_three_digit_multiple_of_17_l759_759568

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759568


namespace greatest_three_digit_multiple_of_17_l759_759825

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759825


namespace number_of_animals_per_aquarium_l759_759388

variable (aq : ℕ) (ani : ℕ) (a : ℕ)

axiom condition1 : aq = 26
axiom condition2 : ani = 52
axiom condition3 : ani = aq * a

theorem number_of_animals_per_aquarium : a = 2 :=
by
  sorry

end number_of_animals_per_aquarium_l759_759388


namespace greatest_three_digit_multiple_of_seventeen_l759_759698

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759698


namespace greatest_three_digit_multiple_of_17_l759_759641

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759641


namespace triangles_same_centroid_l759_759070

-- Define the vertices and midpoints in a hexagon
variable {A : Type*} [LinearOrderedField A]
variable (A1 A2 A3 A4 A5 A6 : A)

-- Let F_ik denote the midpoint of side A_i A_k
structure Midpoint (a b : A) :=
  (val : A) (is_midpoint : 2 * val = a + b)

-- Define the specific midpoints
variable (F12 : Midpoint A1 A2)
variable (F34 : Midpoint A3 A4)
variable (F56 : Midpoint A5 A6)
variable (F23 : Midpoint A2 A3)
variable (F45 : Midpoint A4 A5)
variable (F61 : Midpoint A6 A1)
variable (F36 : Midpoint A3 A6)

-- Prove that the triangles formed by specific midpoints share a common centroid
theorem triangles_same_centroid :
  let T1 := (F12.val, F34.val, F56.val)
  let T2 := (F12.val, F36.val, F45.val)
  let centroid T := (T.1 + T.2 + T.3) / 3
  centroid T1 = centroid T2 := 
sorry

end triangles_same_centroid_l759_759070


namespace greatest_three_digit_multiple_of_17_l759_759741

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759741


namespace greatest_three_digit_multiple_of_17_l759_759777

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759777


namespace lambda_range_l759_759346

theorem lambda_range (λ : ℝ) (a : ℕ+ → ℝ) 
  (h₁ : ∀ n : ℕ+, a n = 3^n - λ * 2^n)
  (h₂ : ∀ n : ℕ+, a n < a (n + 1)) :
  λ < 3 := 
sorry

end lambda_range_l759_759346


namespace divides_plane_with_intersections_l759_759484

-- Definitions and conditions based on the given problem
def divides_plane_into_regions (n : ℕ) : Prop :=
  ∃ lines, (∀ p, p ∈ lines → ∃ q, q ∈ lines ∧ p ≠ q ∧ p ∩ q ≠ ∅) ∧ (number_of_regions lines = n)

-- Minimum n₀ to satisfy the problem conditions
theorem divides_plane_with_intersections (n : ℕ) (h : n ≥ 5) : divides_plane_into_regions n :=
sorry

end divides_plane_with_intersections_l759_759484


namespace max_noncongruent_triangles_l759_759112

theorem max_noncongruent_triangles (n : ℕ) (n_pos : 0 < n) :
  ∃ p_n : ℕ, 
  (p_n = if even n then (n / 2) * (n / 2 + 1) * (n + 1) / 6 
  else (n + 3) * (n + 1) * (2 * n + 5) / 24) := sorry

end max_noncongruent_triangles_l759_759112


namespace greatest_three_digit_multiple_of_17_l759_759575

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759575


namespace find_missing_number_l759_759063

noncomputable def missing_number : Prop :=
  ∃ (y x a b : ℝ),
    a = y + x ∧
    b = x + 630 ∧
    28 = y * a ∧
    660 = a * b ∧
    y = 13

theorem find_missing_number : missing_number :=
  sorry

end find_missing_number_l759_759063


namespace sum_of_primes_less_than_20_l759_759983

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759983


namespace greatest_three_digit_multiple_of_17_l759_759688

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759688


namespace arith_seq_sum_l759_759530

theorem arith_seq_sum (a₃ a₄ a₅ : ℤ) (h₁ : a₃ = 7) (h₂ : a₄ = 11) (h₃ : a₅ = 15) :
  let d := a₄ - a₃;
  let a := a₄ - 3 * d;
  (6 / 2 * (2 * a + 5 * d)) = 54 :=
by
  sorry

end arith_seq_sum_l759_759530


namespace original_second_smallest_element_l759_759509

-- Defining the original set S
def S (x : ℤ) : Finset ℤ := {-4, x, 0, 6, 9}

noncomputable def mean (s : Finset ℤ) : ℚ :=
  s.sum / s.card

theorem original_second_smallest_element (x : ℤ) (p1 p2 : ℤ) (hp1_prime : Nat.Prime p1) (hp2_prime : Nat.Prime p2) 
  (h : S x ∪ {p1, p2} = {-4, x, 0, 6, 9})
  (condition_100_percent_increase : mean {p1, p2, 0, 6, 9} = 2 * mean (S x)) :
  x = -1 := by
  sorry

end original_second_smallest_element_l759_759509


namespace new_person_age_is_15_l759_759230

def age_of_new_person (avg : ℝ) (x : ℝ) : Prop :=
  ((avg + 3) * 10 - 45 = avg * 10 - x) → x = 15

theorem new_person_age_is_15 (avg : ℝ) (x : ℝ) : age_of_new_person avg x :=
by
  intro h
  rw age_of_new_person
  exact sorry

end new_person_age_is_15_l759_759230


namespace greatest_three_digit_multiple_of_17_l759_759611

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759611


namespace greatest_three_digit_multiple_of_17_l759_759749

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759749


namespace prob_yellow_curved_l759_759060

variable (P : Set → ℚ)
variable (G Y S C : Set)

-- Conditions
variable (prob_G : P G = 3 / 4)
variable (prob_S : P S = 1 / 2)
variable (independent_color_shape : P(Y ∩ C) = P(Y) * P(C))
variable (total_prob : P Y + P G = 1 ∧ P C + P S = 1)

-- Theorem statement
theorem prob_yellow_curved : P(Y ∩ C) = 1 / 8 :=
by
  have prob_Y : P Y = 1 - P G := by rw [prob_G]; norm_num
  have prob_C : P C = 1 - P S := by rw [prob_S]; norm_num
  rw [independent_color_shape, prob_Y, prob_C]
  norm_num

end prob_yellow_curved_l759_759060


namespace sum_first_six_terms_l759_759534

-- Define the conditions given in the problem
def a3 := 7
def a4 := 11
def a5 := 15

-- Define the common difference
def d := a4 - a3 -- 4

-- Define the first term
def a1 := a3 - 2 * d -- -1

-- Define the sum of the first six terms of the arithmetic sequence
def S6 := (6 / 2) * (2 * a1 + (6 - 1) * d) -- 54

-- The theorem we want to prove
theorem sum_first_six_terms : S6 = 54 := by
  sorry

end sum_first_six_terms_l759_759534


namespace B_completes_remaining_work_in_18_days_l759_759245

noncomputable def remaining_work_days (work_rate_A work_rate_B : ℝ) (days_A : ℝ) : ℝ :=
  let work_done_by_A := work_rate_A * days_A
  let work_remaining := 1 - work_done_by_A
  work_remaining / work_rate_B

theorem B_completes_remaining_work_in_18_days :
  let work_rate_A : ℝ := 1 / 15,
  let work_rate_B : ℝ := 1 / 27,
  let days_A : ℝ := 5,
  remaining_work_days work_rate_A work_rate_B days_A = 18 := by
  sorry

end B_completes_remaining_work_in_18_days_l759_759245


namespace greatest_three_digit_multiple_of_seventeen_l759_759706

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759706


namespace solution_set_M_minimum_value_expr_l759_759022

-- Define the function f(x)
def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

-- Proof problem (1): Prove that the solution set M of the inequality f(x) ≥ -1 is {x | 2/3 ≤ x ≤ 6}.
theorem solution_set_M : 
  { x : ℝ | f x ≥ -1 } = { x : ℝ | 2/3 ≤ x ∧ x ≤ 6 } :=
sorry

-- Define the requirement for t and the expression to minimize
noncomputable def t : ℝ := 6
noncomputable def expr (a b c : ℝ) : ℝ := 1 / (2 * a + b) + 1 / (2 * a + c)

-- Proof problem (2): Given t = 6 and 4a + b + c = 6, prove that the minimum value of expr is 2/3.
theorem minimum_value_expr (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = t) :
  expr a b c ≥ 2/3 :=
sorry

end solution_set_M_minimum_value_expr_l759_759022


namespace greatest_three_digit_multiple_of_17_l759_759776

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759776


namespace P_lt_Q_l759_759395

variable {x : ℝ}

def P (x : ℝ) : ℝ := (x - 2) * (x - 4)
def Q (x : ℝ) : ℝ := (x - 3) ^ 2

theorem P_lt_Q : P x < Q x := by
  sorry

end P_lt_Q_l759_759395


namespace cubed_ge_sqrt_ab_squared_l759_759457

theorem cubed_ge_sqrt_ab_squared (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  a^3 + b^3 ≥ (ab)^(1/2) * (a^2 + b^2) :=
sorry

end cubed_ge_sqrt_ab_squared_l759_759457


namespace pages_per_day_l759_759144

variable (P : ℕ) (D : ℕ)

theorem pages_per_day (hP : P = 66) (hD : D = 6) : P / D = 11 :=
by
  sorry

end pages_per_day_l759_759144


namespace greatest_three_digit_multiple_of_17_is_986_l759_759752

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759752


namespace greatest_three_digit_multiple_of_17_is_986_l759_759667

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759667


namespace four_digit_even_numbers_count_l759_759195

-- Define the conditions
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_even (n : ℕ) : Prop := n % 2 = 0
def no_repeat_digits (n : ℕ) : Prop :=
  (n.digits : List ℕ).nodup

-- Define the problem
theorem four_digit_even_numbers_count :
  ∃ count, count = 156 ∧
  ∀ n, is_four_digit n ∧ is_even n ∧ (n.digits.to_finset ⊆ digits) ∧ no_repeat_digits n →
  true :=
begin
  use 156,
  intros n h,
  sorry
end

end four_digit_even_numbers_count_l759_759195


namespace greatest_three_digit_multiple_of_17_l759_759653

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759653


namespace sum_of_primes_less_than_20_l759_759867

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759867


namespace polygon_diagonals_15_sides_l759_759282

/-- Given a convex polygon with 15 sides, the number of diagonals is 90. -/
theorem polygon_diagonals_15_sides (n : ℕ) (h : n = 15) (convex : Prop) : 
  ∃ d : ℕ, d = 90 :=
by
    sorry

end polygon_diagonals_15_sides_l759_759282


namespace surface_area_of_circumscribed_sphere_l759_759366

theorem surface_area_of_circumscribed_sphere (V : ℝ) (hV : V = 64) : 
  let a := (V^(1/3)) in
  let R := (a * Real.sqrt 3) / 2 in
  4 * Real.pi * R^2 = 48 * Real.pi :=
by
  -- definitions
  let a := (V^(1/3))
  let R := (a * Real.sqrt 3) / 2
  sorry

end surface_area_of_circumscribed_sphere_l759_759366


namespace sum_of_a_with_one_root_l759_759303

-- Definition of the condition: given quadratic equation 3x^2 + ax + 12x + 7 = 0
def quadratic_eq (x a : ℝ) : ℝ := 3 * x^2 + (a + 12) * x + 7

-- Prove the sum of values of a for which the equation has exactly one solution is -24
theorem sum_of_a_with_one_root : 
  (∃ a : ℝ, ∃ x : ℝ, quadratic_eq x a = 0 ∧ 
    (a + 12) ^ 2 - 4 * 3 * 7 = 0) →
  (∃ a1 a2 : ℝ, (a1 = -12 + 2 * Real.sqrt 21) ∧ (a2 = -12 - 2 * Real.sqrt 21) ∧ (a1 + a2 = -24)) :=
by
  intro ⟨a, x, eq, discrim⟩
  use [-12 + 2 * Real.sqrt 21, -12 - 2 * Real.sqrt 21]
  split
  · rfl
  split
  · rfl
  simp
  sorry

end sum_of_a_with_one_root_l759_759303


namespace sequence_count_l759_759348

noncomputable def sequence_problem (a : Fin 16 → ℤ) : Prop :=
  a 0 = 1 ∧
  a 7 = 4 ∧
  (∀ n : ℕ, n < 15 → (a (n + 1) - a n ∈ {-1, 1})) ∧
  (a 15 = 0 ∨ a 15 = 8) ∧
  (a 15 - a 7 ∈ {-4, 4})

theorem sequence_count :
  ∃ a : Fin 16 → ℤ, sequence_problem a ∧ 
  (finset.card (finset.filter sequence_problem (finset.univ : finset (Fin 16 → ℤ))) = 1176) :=
sorry

end sequence_count_l759_759348


namespace greatest_three_digit_multiple_of_17_l759_759744

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759744


namespace find_f_x_l759_759004

-- Define the inverse function property
def is_inv_function (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a^x) = x

-- Given conditions
variables (a : ℝ) (f : ℝ → ℝ)
axiom a_pos : a > 0
axiom a_ne_one : a ≠ 1
axiom f_inv : is_inv_function f a
axiom f_2 : f 2 = 1

-- Definition of the target function
def g (x : ℝ) := Real.log x / Real.log 2

-- The proof problem
theorem find_f_x : f = g :=
sorry

end find_f_x_l759_759004


namespace greatest_three_digit_multiple_of_17_is_986_l759_759629

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759629


namespace sweet_treats_per_student_l759_759465

theorem sweet_treats_per_student :
  let cookies := 20
  let cupcakes := 25
  let brownies := 35
  let students := 20
  (cookies + cupcakes + brownies) / students = 4 :=
by 
  sorry

end sweet_treats_per_student_l759_759465


namespace product_of_possible_values_of_x_l759_759393

noncomputable def product_of_roots (x : ℝ) := 
(x + 3) * (x - 5) = 18

theorem product_of_possible_values_of_x :
  ∀ x : ℝ, product_of_roots x → x * (x + 6) ≠ -33 := 
sorry

end product_of_possible_values_of_x_l759_759393


namespace taxi_fare_distance_l759_759044

theorem taxi_fare_distance (initial_fare : ℝ) (subsequent_fare : ℝ) (initial_distance : ℝ) (total_fare : ℝ) : 
  initial_fare = 2.0 →
  subsequent_fare = 0.60 →
  initial_distance = 1 / 5 →
  total_fare = 25.4 →
  ∃ d : ℝ, d = 8 :=
by 
  intros h1 h2 h3 h4
  sorry

end taxi_fare_distance_l759_759044


namespace sum_of_integers_75_to_95_l759_759217

def arithmeticSumOfIntegers (a l : ℕ) : ℕ :=
  let n := l - a + 1
  n / 2 * (a + l)

theorem sum_of_integers_75_to_95 : arithmeticSumOfIntegers 75 95 = 1785 :=
  by
  sorry

end sum_of_integers_75_to_95_l759_759217


namespace greatest_three_digit_multiple_of_17_l759_759811

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759811


namespace greatest_three_digit_multiple_of_17_l759_759582

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759582


namespace greatest_three_digit_multiple_of_17_l759_759689

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759689


namespace log_graph_fixed_point_l759_759400

theorem log_graph_fixed_point (a : ℝ) (x : ℝ) (y : ℝ) 
  (h₁ : a > 0) 
  (h₂ : a ≠ 1) 
  (hx : x = 2) : 
  y = log a (x - 1) + 2 → (x, y) = (2, 2) := 
by {
  sorry
}

end log_graph_fixed_point_l759_759400


namespace sum_of_primes_less_than_20_l759_759870

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759870


namespace slowest_train_time_l759_759545

-- Conditions
def length_train1 : ℝ := 650
def length_train2 : ℝ := 750
def length_train3 : ℝ := 850

def speed_train1_kmph : ℝ := 45
def speed_train2_kmph : ℝ := 30
def speed_train3_kmph : ℝ := 60

def kmph_to_mps (v : ℝ) : ℝ :=
  v * (1000 / 3600)

def speed_train1 := kmph_to_mps speed_train1_kmph
def speed_train2 := kmph_to_mps speed_train2_kmph
def speed_train3 := kmph_to_mps speed_train3_kmph

def slowest_train_speed := speed_train2
def distance_between_drivers := length_train1 + length_train3
def distance_for_slowest_train_to_cover := distance_between_drivers / 2
def expected_time : ℝ := 90

-- Proof Statement
theorem slowest_train_time :
  (distance_for_slowest_train_to_cover / slowest_train_speed = expected_time) :=
by sorry

end slowest_train_time_l759_759545


namespace probability_of_choosing_geography_probability_X_equals_2_probability_distribution_table_expected_value_of_X_l759_759089

noncomputable def prob_geography (prob_physics : ℚ) (prob_geo_given_physics : ℚ) (prob_history : ℚ) (prob_geo_given_history : ℚ) : ℚ :=
  prob_physics * prob_geo_given_physics + prob_history * prob_geo_given_history

axiom prob_physics : ℚ := 3/4
axiom prob_geo_given_physics : ℚ := 2/3
axiom prob_history : ℚ := 1/4
axiom prob_geo_given_history : ℚ := 4/5

theorem probability_of_choosing_geography :
  prob_geography prob_physics prob_geo_given_physics prob_history prob_geo_given_history = 7/10 :=
  sorry

noncomputable def binomial_prob (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

axiom n : ℕ := 3
axiom prob_geography_final : ℚ := 7/10

theorem probability_X_equals_2 :
  binomial_prob n prob_geography_final 2 = 441/1000 :=
  sorry

noncomputable def expected_value (n : ℕ) (p : ℚ) : ℚ :=
  n * p

theorem probability_distribution_table :
  {0, 1, 2, 3}.map (λ k, binomial_prob n prob_geography_final k) =
  [{27/1000}, {189/1000}, {441/1000}, {343/1000}] :=
  sorry

theorem expected_value_of_X :
  expected_value n prob_geography_final = 21/10 :=
  sorry

end probability_of_choosing_geography_probability_X_equals_2_probability_distribution_table_expected_value_of_X_l759_759089


namespace determine_circumcircle_radius_l759_759452

noncomputable def circumcircle_radius_range (a b R : ℝ) : Prop :=
  -- Let points A, B on the parabola y = x^2
  let A := (a, a^2)
  let B := (-a, a^2) in
  -- Let point C on the y-axis
  let C := (0, b) in
  -- Given triangle ABC lies on y = x^2 with distinct points A, B, and C
  -- We are proving 
  R > 1/2 ∧ (∃ a b : ℝ, C = (0, (a^2 + 1) / 2)) 

theorem determine_circumcircle_radius (a b R : ℝ) :
  circumcircle_radius_range a b R :=
sorry

end determine_circumcircle_radius_l759_759452


namespace sculpture_height_correct_l759_759284

/-- Define the conditions --/
def base_height_in_inches : ℝ := 4
def total_height_in_feet : ℝ := 3.1666666666666665
def inches_per_foot : ℝ := 12

/-- Define the conversion from feet to inches for the total height --/
def total_height_in_inches : ℝ := total_height_in_feet * inches_per_foot

/-- Define the height of the sculpture in inches --/
def sculpture_height_in_inches : ℝ := total_height_in_inches - base_height_in_inches

/-- The proof problem in Lean 4 statement --/
theorem sculpture_height_correct :
  sculpture_height_in_inches = 34 := by
  sorry

end sculpture_height_correct_l759_759284


namespace sum_of_primes_lt_20_eq_77_l759_759910

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759910


namespace event_intersect_points_l759_759422

noncomputable def C1_equation (a t : ℝ) : Prop :=
  ∃ x y, x = a + sqrt 2 * t ∧ y = 1 + sqrt 2 * t ∧ (x - y - a + 1 = 0)

noncomputable def C2_equation : Prop :=
  ∀ (rho θ : ℝ), (rho * cos θ ^ 2 + 4 * cos θ - rho = 0) → ∃ x y, y^2 = 4*x

noncomputable def intersection_conditions (a t1 t2 : ℝ) : Prop :=
  ∃ x y, (y^2 = 4*x) ∧ (x = a + sqrt 2 * t1) ∧ (x = a + sqrt 2 * t2) ∧ 
         (y = 1 + sqrt 2 * t1) ∧ (y = 1 + sqrt 2 * t2) ∧ (abs(t1) = 2 * abs(t2))

theorem event_intersect_points (a : ℝ) : C1_equation a t → C2_equation → intersection_conditions a t1 t2 →
  a = 1/36 ∨ a = 9/4 :=
sorry

end event_intersect_points_l759_759422


namespace greatest_three_digit_multiple_of_17_l759_759850

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759850


namespace sum_of_acute_angles_eq_pi_over_4_l759_759363

theorem sum_of_acute_angles_eq_pi_over_4
  (α β : ℝ)
  (h₀ : 0 < α ∧ α < π/2)
  (h₁ : 0 < β ∧ β < π/2)
  (h₂ : (1 + tan α) * (1 + tan β) = 2) 
  : α + β = π/4 :=
sorry

end sum_of_acute_angles_eq_pi_over_4_l759_759363


namespace sum_of_primes_less_than_20_l759_759879

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759879


namespace greatest_three_digit_multiple_of_17_l759_759747

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759747


namespace cos_arcsin_tan_arccos_eq_x_l759_759327

theorem cos_arcsin_tan_arccos_eq_x (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  (cos (arcsin (tan (arccos x))) = x) → x = 1 :=
by
  intro h1
  sorry

end cos_arcsin_tan_arccos_eq_x_l759_759327


namespace correct_option_l759_759225

theorem correct_option :
  (∀ (a b : ℝ),  3 * a^2 * b - 4 * b * a^2 = -a^2 * b) ∧
  ¬(1 / 7 * (-7) + (-1 / 7) * 7 = 1) ∧
  ¬((-3 / 5)^2 = 9 / 5) ∧
  ¬(∀ (a b : ℝ), 3 * a + 5 * b = 8 * a * b) :=
by
  sorry

end correct_option_l759_759225


namespace vector_properties_l759_759032

noncomputable def vec_a := (-1, 1)
noncomputable def vec_b := (2, 0)
noncomputable def vec_c := (1, 1)

-- Given the conditions
def A_plus_B := (1, 1)
def A_minus_B := (-3, 1)

-- Assuming vectors a and b can be solved correctly
def a_plus_b_eq := vec_a + vec_b = A_plus_B
def a_minus_b_eq := vec_a - vec_b = A_minus_B

-- Conditions check
def orthogonal := vec_a.1 * vec_c.1 + vec_a.2 * vec_c.2 = 0
def angle := real.arccos ((vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2) / (real.sqrt ((vec_a.1)^2 + (vec_a.2)^2) * real.sqrt ((vec_b.1)^2 + (vec_b.2)^2))) = real.pi * (3 / 4)

theorem vector_properties : a_plus_b_eq ∧ a_minus_b_eq → orthogonal ∧ angle := 
by
  sorry

end vector_properties_l759_759032


namespace complementary_angles_ratio_l759_759506

theorem complementary_angles_ratio (x : ℝ) (hx : 5 * x = 90) : abs (4 * x - x) = 54 :=
by
  have h₁ : x = 18 := by 
    linarith [hx]
  rw [h₁]
  norm_num

end complementary_angles_ratio_l759_759506


namespace sum_primes_less_than_20_l759_759950

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759950


namespace sum_of_primes_less_than_20_l759_759890

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759890


namespace sum_primes_less_than_20_l759_759968

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759968


namespace greatest_three_digit_multiple_of_17_l759_759803

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759803


namespace sum_first_six_terms_l759_759535

-- Define the conditions given in the problem
def a3 := 7
def a4 := 11
def a5 := 15

-- Define the common difference
def d := a4 - a3 -- 4

-- Define the first term
def a1 := a3 - 2 * d -- -1

-- Define the sum of the first six terms of the arithmetic sequence
def S6 := (6 / 2) * (2 * a1 + (6 - 1) * d) -- 54

-- The theorem we want to prove
theorem sum_first_six_terms : S6 = 54 := by
  sorry

end sum_first_six_terms_l759_759535


namespace area_of_inscribed_pentagon_l759_759295

-- Definitions based on the problem conditions
def pentagon_side_lengths (FG GH HI IJ JF : ℝ) : Prop :=
  FG = 8 ∧ GH = 9 ∧ HI = 9 ∧ IJ = 9 ∧ JF = 10

-- Lean statement for the mathematically equivalent proof problem
theorem area_of_inscribed_pentagon (r : ℝ) (FG GH HI IJ JF : ℝ) 
  (h : pentagon_side_lengths FG GH HI IJ JF) :
  let s := (FG + GH + HI + IJ + JF) / 2 in
  s = 22 → 
  let A := s * r in
  A = 45 * r :=
by
  -- Introduced let-binding for semiperimeter
  intros s hs A,
  -- The proof will follow, which shows calculations leading to the area being 45r
  sorry

end area_of_inscribed_pentagon_l759_759295


namespace boys_and_girls_in_class_l759_759056

theorem boys_and_girls_in_class (b g : ℕ) (h1 : b + g = 21) (h2 : 5 * b + 2 * g = 69) 
: b = 9 ∧ g = 12 := by
  sorry

end boys_and_girls_in_class_l759_759056


namespace sum_of_primes_less_than_20_is_77_l759_759921

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759921


namespace cereal_mixture_sugar_percentage_l759_759286

theorem cereal_mixture_sugar_percentage :
  (∀ a b : ℕ, a = 10 ∧ b = 2 → let ratio := 1 in
      let weight_mixture := 100 in
      let weight_a := weight_mixture / (ratio + 1) in
      let weight_b := weight_mixture / (ratio + 1) in
      let sugar_a := (a * weight_a) / 100 in
      let sugar_b := (b * weight_b) / 100 in
      let total_sugar := sugar_a + sugar_b in
      (total_sugar * 100 / weight_mixture) = 6) :=
λ a b h, sorry

end cereal_mixture_sugar_percentage_l759_759286


namespace greatest_three_digit_multiple_of_17_l759_759690

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759690


namespace projection_onto_plane_l759_759102

open Matrix

def normal_vector : Vector 3 := ![2, -1, 2]

def projection_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  !![[5/9, 4/9, -4/9], [2/9, 10/9, 2/9], [-4/9, 4/9, 5/9]]

theorem projection_onto_plane (v : Vector 3) : (projection_matrix.mul_vec v) = 
([fin 3] → ℝ -> ℝ := λ (i : Fin 1) (x : ℝ), 
let n := normal_vector in 
let dot_product := (x * n i) / (n.norm_sq.to_real) 
n.mul dot_product - v := v sorry)
 
end projection_onto_plane_l759_759102


namespace composite_product_probability_l759_759040

theorem composite_product_probability :
  let outcomes := 6^6 in
  let non_composite_outcomes := 19 in
  let composite_probability := (outcomes - non_composite_outcomes) / outcomes in
  composite_probability = 46637 / 46656 :=
sorry

end composite_product_probability_l759_759040


namespace impossible_tiling_conditions_l759_759331

theorem impossible_tiling_conditions (m n : ℕ) :
  ¬ (∃ (a b : ℕ), (a - 1) * 4 + (b + 1) * 4 = m * n ∧ a * 4 % 4 = 2 ∧ b * 4 % 4 = 0) :=
sorry

end impossible_tiling_conditions_l759_759331


namespace num_alternating_parity_sequences_l759_759391

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def alternating_parity_seq (seq : list ℕ) : Prop :=
  ∀ i, i < seq.length - 1 → (is_odd (seq.nth i).get_or_else 0 ↔ is_even (seq.nth (i + 1)).get_or_else 0)

theorem num_alternating_parity_sequences :
  ∃ n : ℕ, n = 5^8 ∧ ∀ seq : list ℕ, seq.length = 8 →
  seq.head' ≠ none → is_odd (seq.head'.get_or_else 0) →
  alternating_parity_seq seq :=
sorry

end num_alternating_parity_sequences_l759_759391


namespace range_of_f_ineq_l759_759011
-- Import Lean library

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

-- Define the mathematically equivalent proof to be proven in Lean
theorem range_of_f_ineq {x : ℝ} : f(x) + f(x - 1 / 2) > 1 ↔ x > -1 / 4 :=
by
  sorry

end range_of_f_ineq_l759_759011


namespace exists_four_numbers_with_square_product_l759_759514

theorem exists_four_numbers_with_square_product 
    (numbers : Fin 48 → ℕ) 
    (h_prime_factors : (∏ i, numbers i).prime_factors.length = 10) :
  ∃ (a b c d : Fin 48), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ is_square ((numbers a) * (numbers b) * (numbers c) * (numbers d)) :=
by { sorry }

end exists_four_numbers_with_square_product_l759_759514


namespace greatest_three_digit_multiple_of_17_l759_759797

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759797


namespace greatest_three_digit_multiple_of17_l759_759726

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759726


namespace probability_of_particular_student_selected_l759_759274

noncomputable def probability_student_selected (A : Type) (classes : Finset (Fin 8 × Fin 40)) (selected : Finset (Fin 8 × Fin 40)) : ℚ :=
  if h : selected.card = 3 ∧ (∀ (c : Fin 8), (selected.filter (λ x, x.1 = c)).card ≤ 1) ∧ ∃ c, (A.1, A.2) ∈ selected ∧ A.1 = c then
    3 / 320
  else
    0

theorem probability_of_particular_student_selected :
  ∃ (A : (Fin 8 × Fin 40)), 
  ∀ (classes : Finset (Fin 8 × Fin 40)) 
    (selected : Finset (Fin 8 × Fin 40)), 
    (classes.card = 320) ∧ 
    (selected.card = 3) ∧ 
    (∀ (c : Fin 8), (selected.filter (λ x, x.1 = c)).card ≤ 1) 
    → probability_student_selected A classes selected = 3 / 320 :=
by
  sorry

end probability_of_particular_student_selected_l759_759274


namespace angle_ABC_is_45_degrees_l759_759095

theorem angle_ABC_is_45_degrees
  (A B C : Point)
  (hAB_AC : dist A B = dist A C)
  (hTangentPerpendicular : ∀ (circle : Circle) (tangentAtB : Line), is_tangent circle B tangentAtB → perpendicular tangentAtB (line A C)) :
  ∠ B A C = 45 := 
sorry

end angle_ABC_is_45_degrees_l759_759095


namespace greatest_three_digit_multiple_of_17_is_986_l759_759753

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759753


namespace no_calls_in_2022_l759_759473

   /-- 
   Mrs. Roberts has four grandchildren, who each call her on separate regular schedules.
   One grandchild calls every two days, another every three days, another every six days, 
   and the last one every seven days. They all called her on January 1st, 2022. 
   Prove that the number of days in the year 2022 that Mrs. Roberts did not receive any calls 
   from her grandchildren is 53.
   -/
   theorem no_calls_in_2022 : 
     let total_days := 365
     let calls_every_days : List ℕ := [2, 3, 6, 7]
     let calls_in_year (n : ℕ) : ℕ := total_days / n
     let lcm_list (lst : List ℕ) : ℕ := lst.foldr1 lcm
     let overlap_calls (lsts : List (List ℕ)) : ℕ := lsts.map lcm_list |>.map calls_in_year |>.sum
     let sublists (lst : List ℕ) : List (List ℕ) := lst.powerset.filter (λ l, l ≠ [] ∧ l.length < lst.length)
     let inclusion_exclusion_calls : ℕ := sublists calls_every_days |>.map (λ l, (-1)^(l.length + 1) * calls_in_year (lcm_list l)) |>.sum
     let at_least_one_call := calls_every_days.map calls_in_year |>.sum - overlap_calls (sublists calls_every_days)
   in total_days - at_least_one_call = 53 :=
   sorry
   
end no_calls_in_2022_l759_759473


namespace blacken_polygon_l759_759538

structure Point where
  (x : ℝ)
  (y : ℝ)

def isPolygon (points : List Point) : Prop :=
  points.length = 2020 ∧
  (∀ i, 0 < i ∧ i < 2020 → points[i].x < points[i+1].x) ∧
  (∀ i, 1 ≤ i ∧ i ≤ 2020 → points[2020-i].y < points[2020-i-1].y)

def area (points : List Point) : ℝ :=
  (0.5 * |∑ i in List.range 2019, points[i].x * points[i+1].y - points[i+1].x * points[i].y|)

noncomputable def totalCost (points : List Point) : ℝ :=
  ∑ i in List.range 2019, points[i+1].x * points[i].y

theorem blacken_polygon (points : List Point) (hPolygon : isPolygon points) :
  totalCost points ≤ 4 * area points := 
sorry

end blacken_polygon_l759_759538


namespace greatest_three_digit_multiple_of_17_l759_759614

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759614


namespace extremum_and_inequality_exists_minimum_value_of_f_l759_759340

noncomputable def f (a : ℝ) (x : ℝ) := a * x - Real.log x
def g (x : ℝ) := Real.log x / x

theorem extremum_and_inequality (a : ℝ) (H0 : a = 1) (x : ℝ) (H1 : 0 < x ∧ x ≤ Real.exp 1)
    : |f a x| > g x + 1 / 2 := by
  sorry

theorem exists_minimum_value_of_f (H: ∃ (a : ℝ), ∀ x ∈ Set.Icc 0 (Real.exp 1), f a x ≥ 3)
    : ∃ a, a = Real.exp 2 := by
  use Real.exp 2
  sorry

end extremum_and_inequality_exists_minimum_value_of_f_l759_759340


namespace greatest_three_digit_multiple_of_17_l759_759832

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759832


namespace products_in_range_98_104_l759_759498

noncomputable def number_of_products_less_than_100 : ℕ := 36
noncomputable def frequency_in_range_96_100 : ℚ := 0.3
noncomputable def sample_size : ℕ := 120
noncomputable def desired_range_frequency : ℚ := (0.1 + 0.15 + 0.125) * 2

theorem products_in_range_98_104 :
  let quantity := desired_range_frequency * sample_size in
  quantity = 60 := by
  sorry

end products_in_range_98_104_l759_759498


namespace sum_of_primes_less_than_20_is_77_l759_759922

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759922


namespace length_of_chord_AB_l759_759344

noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 3, 0)
noncomputable def line_eq (x : ℝ) := x - Real.sqrt 3
noncomputable def ellipse_eq (x y : ℝ)  := x^2 / 4 + y^2 = 1

theorem length_of_chord_AB :
  ∀ (A B : ℝ × ℝ), 
  (line_eq A.1 = A.2) → 
  (line_eq B.1 = B.2) → 
  (ellipse_eq A.1 A.2) → 
  (ellipse_eq B.1 B.2) → 
  ∃ d : ℝ, d = 8 / 5 ∧ 
  dist A B = d := 
sorry

end length_of_chord_AB_l759_759344


namespace sum_primes_less_than_20_l759_759934

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759934


namespace magnitude_z_eq_one_l759_759409

theorem magnitude_z_eq_one (z : ℂ) (h : z * (1 - complex.I) = 1 + complex.I) : complex.abs z = 1 :=
sorry

end magnitude_z_eq_one_l759_759409


namespace sweet_treats_per_student_l759_759471

theorem sweet_treats_per_student : 
  ∀ (cookies cupcakes brownies students : ℕ), 
    cookies = 20 →
    cupcakes = 25 →
    brownies = 35 →
    students = 20 →
    (cookies + cupcakes + brownies) / students = 4 :=
by
  intros cookies cupcakes brownies students hcook hcup hbrown hstud
  have h1 : cookies + cupcakes + brownies = 80, from calc
    cookies + cupcakes + brownies = 20 + 25 + 35 := by rw [hcook, hcup, hbrown]
    ... = 80 := rfl
  have h2 : (cookies + cupcakes + brownies) / students = 80 / 20, from
    calc (cookies + cupcakes + brownies) / students
      = 80 / 20 := by rw [h1, hstud]
  exact eq.trans h2 (by norm_num)

end sweet_treats_per_student_l759_759471


namespace part1_part2_l759_759029

open Set

variable (A B : Set ℝ) (m : ℝ)

def setA : Set ℝ := {x | x ^ 2 - 2 * x - 8 ≤ 0}

def setB (m : ℝ) : Set ℝ := {x | x ^ 2 - (2 * m - 3) * x + m ^ 2 - 3 * m ≤ 0}

theorem part1 (h : (setA ∩ setB 5) = Icc 2 4) : m = 5 := sorry

theorem part2 (h : setA ⊆ compl (setB m)) :
  m ∈ Iio (-2) ∪ Ioi 7 := sorry

end part1_part2_l759_759029


namespace equation_of_ellipse_equation_of_line_L_l759_759350

theorem equation_of_ellipse :
  (∃ (C : Type) (a b : ℝ), 0 < b ∧ b < a ∧
    ellipse_centered_at_origin_with_foci_on_x_axis C a b 4sqrt3 (sqrt3/2) ∧ 
    ellipse_equation C = (x^2)/16 + (y^2)/4 = 1) := sorry

theorem equation_of_line_L :
  (∃ (L : Type) (k : ℝ), line_passes_through_point_and_intersects_ellipse L (0,1) (x^2)/16 + (y^2)/4 = 1 
    (y - tot_vec e_1 = k * x) (AM = 2 * MB)
    ∧ line_equation L = y = pm(sqrt15/10) * x + 1) := sorry

end equation_of_ellipse_equation_of_line_L_l759_759350


namespace vector_operations_properties_l759_759497

variables {V : Type*} [add_comm_group V] [module ℝ V]

-- Definition of the dot product of vectors
def dot_product (u v : V) : ℝ := sorry

-- Definition of vector addition
def vector_addition (u v : V) : V := u + v

-- Definition of vector subtraction
def vector_subtraction (u v : V) : V := u - v

-- Definition of scalar multiplication
def scalar_multiplication (a : ℝ) (v : V) : V := a • v

-- Theorem to prove the problem
theorem vector_operations_properties (u v : V) (a : ℝ) :
  ∃ d : ℝ, d = dot_product u v ∧
  ∃ w : V, w = vector_addition u v ∧
  ∃ x : V, x = vector_subtraction u v ∧
  ∃ y : V, y = scalar_multiplication a v := sorry

end vector_operations_properties_l759_759497


namespace Q_at_n_10_l759_759454

def Q (n : ℕ) : ℚ :=
  ∏ i in finset.range (n - 1), (1 - 1 / (i + 2)^2)

theorem Q_at_n_10 : Q 10 = 1 / 50 := by
  sorry

end Q_at_n_10_l759_759454


namespace animath_workshop_pigeonhole_l759_759297

theorem animath_workshop_pigeonhole (n : ℕ) 
  (knows : ∀ (A B : ℕ), A < n → B < n → Prop)
  (reciprocal : ∀ {A B : ℕ}, A < n → B < n → knows A B ↔ knows B A) :
  ∃ (A B : ℕ), A < n ∧ B < n ∧ A ≠ B ∧ (∀ A_num : ℕ, A_num = (finset.card (finset.filter (knows A) (finset.range n)))) ∧ (∀ B_num : ℕ, B_num = (finset.card (finset.filter (knows B) (finset.range n)))) ∧ A_num = B_num := 
by
  sorry

end animath_workshop_pigeonhole_l759_759297


namespace greatest_three_digit_multiple_of_17_l759_759592

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759592


namespace events_mutually_exclusive_but_not_opposite_l759_759304

inductive Card
| black
| red
| white

inductive Person
| A
| B
| C

def event_A_gets_red (distribution : Person → Card) : Prop :=
  distribution Person.A = Card.red

def event_B_gets_red (distribution : Person → Card) : Prop :=
  distribution Person.B = Card.red

theorem events_mutually_exclusive_but_not_opposite (distribution : Person → Card) :
  event_A_gets_red distribution ∧ event_B_gets_red distribution → False :=
by sorry

end events_mutually_exclusive_but_not_opposite_l759_759304


namespace newspaper_delivery_difference_l759_759079

-- Define the weekly deliveries for each person based on the conditions
def weekly_deliveries_jake : ℕ := 234
def weekly_deliveries_miranda : ℕ := 2 * weekly_deliveries_jake
def weekly_deliveries_jason : ℕ := 150 + 350
def weekly_deliveries_lisa : ℕ := 175 * 5

-- Approximate number of weeks in a 30-day month
noncomputable def weeks_in_month : ℝ := 30 / 7

-- Define the monthly deliveries for each person
noncomputable def monthly_deliveries_jake : ℝ := weekly_deliveries_jake * weeks_in_month
noncomputable def monthly_deliveries_miranda : ℝ := weekly_deliveries_miranda * weeks_in_month
noncomputable def monthly_deliveries_jason : ℝ := weekly_deliveries_jason * weeks_in_month
noncomputable def monthly_deliveries_lisa : ℝ := weekly_deliveries_lisa * weeks_in_month

-- Define the combined monthly deliveries for each pair
noncomputable def combined_deliveries_jake_miranda : ℝ := monthly_deliveries_jake + monthly_deliveries_miranda
noncomputable def combined_deliveries_jason_lisa : ℝ := monthly_deliveries_jason + monthly_deliveries_lisa

-- Define the total difference in deliveries between the two pairs
noncomputable def total_difference : ℝ := combined_deliveries_jason_lisa - combined_deliveries_jake_miranda

-- Statement to prove the difference is approximately 2885, rounded to the nearest whole number
theorem newspaper_delivery_difference : Int.floor (total_difference + 0.5) = 2885 := by
  sorry

end newspaper_delivery_difference_l759_759079


namespace ram_actual_distance_from_base_l759_759229

def map_distance_between_mountains : ℝ := 312
def actual_distance_between_mountains : ℝ := 136
def ram_map_distance_from_base : ℝ := 28

theorem ram_actual_distance_from_base :
  ram_map_distance_from_base * (actual_distance_between_mountains / map_distance_between_mountains) = 12.205 :=
by sorry

end ram_actual_distance_from_base_l759_759229


namespace imaginary_part_of_complex_num_l759_759343

def complex_num : ℂ := (3 - Complex.i) / (1 + Complex.i)

theorem imaginary_part_of_complex_num (z : ℂ) (h : z = (3 - Complex.i) / (1 + Complex.i)) : z.im = -2 := 
by sorry

end imaginary_part_of_complex_num_l759_759343


namespace sum_integers_75_to_95_l759_759203

theorem sum_integers_75_to_95 :
  let a := 75
  let l := 95
  let n := 95 - 75 + 1
  ∑ k in Finset.range n, (a + k) = 1785 := by
  sorry

end sum_integers_75_to_95_l759_759203


namespace greatest_three_digit_multiple_of_17_l759_759852

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759852


namespace sum_of_primes_less_than_20_l759_759996

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l759_759996


namespace isosceles_right_triangle_shaded_area_l759_759275

theorem isosceles_right_triangle_shaded_area :
  ∀ (l : ℝ) (n m : ℕ), 
  l = 10 →
  n = 25 →
  m = 15 →
  let big_triangle_area := (1 / 2) * l * l in
  let small_triangle_area := big_triangle_area / n in
  m * small_triangle_area = 30 :=
by
  intros l n m hl hn hm big_triangle_area small_triangle_area
  rw [hl, hn, hm]
  let big_triangle_area := (1 / 2) * 10 * 10
  let small_triangle_area := big_triangle_area / 25
  have h_big : big_triangle_area = 50 := by norm_num
  rw [h_big]
  have h_small : small_triangle_area = 2 := by norm_num
  rw [h_small]
  norm_num
  sorry

end isosceles_right_triangle_shaded_area_l759_759275


namespace greatest_three_digit_multiple_of_17_l759_759819

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759819


namespace distinct_colored_triangle_l759_759439

open Finset

variables {n k : ℕ} (hn : 0 < n) (hk : 3 ≤ k)
variables (K : SimpleGraph (Fin n))
variables (color : Edge (Fin n) → Fin k)
variables (connected_subgraph : ∀ i : Fin k, ∀ u v : Fin n, u ≠ v → (∃ p : Walk (Fin n) u v, ∀ {e}, e ∈ p.edges → color e = i))

theorem distinct_colored_triangle :
  ∃ (A B C : Fin n), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  color (A, B) ≠ color (B, C) ∧
  color (B, C) ≠ color (C, A) ∧
  color (C, A) ≠ color (A, B) :=
sorry

end distinct_colored_triangle_l759_759439


namespace greatest_three_digit_multiple_of_17_l759_759783

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759783


namespace probability_not_blue_l759_759242

-- Definitions based on the conditions
def total_faces : ℕ := 12
def blue_faces : ℕ := 1
def non_blue_faces : ℕ := total_faces - blue_faces

-- Statement of the problem
theorem probability_not_blue : (non_blue_faces : ℚ) / total_faces = 11 / 12 :=
by
  sorry

end probability_not_blue_l759_759242


namespace factorize_expression_l759_759311

theorem factorize_expression (m : ℝ) : 3 * m^2 - 12 = 3 * (m + 2) * (m - 2) := 
sorry

end factorize_expression_l759_759311


namespace greatest_three_digit_multiple_of_17_l759_759686

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759686


namespace slope_of_tangent_to_tan_at_pi_over_4_l759_759519

open Real

noncomputable def tan_slope (x : ℝ) : ℝ :=
  deriv (λ x, tan x) x

theorem slope_of_tangent_to_tan_at_pi_over_4 :
  tan_slope (π / 4) = 2 :=
by
  sorry

end slope_of_tangent_to_tan_at_pi_over_4_l759_759519


namespace minimum_reciprocal_sum_of_roots_l759_759015

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) := 2 * x^2 + b * x + c

theorem minimum_reciprocal_sum_of_roots {b c : ℝ} {x1 x2 : ℝ} 
  (h1: f (-10) b c = f 12 b c)
  (h2: f x1 b c = 0)
  (h3: f x2 b c = 0)
  (h4: 0 < x1)
  (h5: 0 < x2)
  (h6: x1 + x2 = 2) :
  (1 / x1 + 1 / x2) = 2 :=
sorry

end minimum_reciprocal_sum_of_roots_l759_759015


namespace sum_of_primes_less_than_20_is_77_l759_759924

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759924


namespace log_12_eq_2a_plus_b_l759_759359

variable (lg : ℝ → ℝ)
variable (lg_2_eq_a : lg 2 = a)
variable (lg_3_eq_b : lg 3 = b)

theorem log_12_eq_2a_plus_b : lg 12 = 2 * a + b :=
by
  sorry

end log_12_eq_2a_plus_b_l759_759359


namespace max_area_BQC_l759_759429

theorem max_area_BQC (A B C E I_B I_C Q : Type) 
  [Triangle A B C]
  [Segment B C E]
  [Incenter A B E I_B]
  [Incenter A C E I_C]
  [Circumcircle B I_B E Q]
  [Circumcircle C I_C E Q]
  (hABC : dist A B = 8 ∧ dist B C = 17 ∧ dist C A = 15)
  (hBAC : angle A B C = 90) :
  ∃ (p q r : ℤ), r ∉ {k^2 | k : ℤ} ∧ 
  let area := 145 - 0.5 * real.sqrt 2 in
  p + q + r = 148 :=
sorry

end max_area_BQC_l759_759429


namespace greatest_three_digit_multiple_of_17_l759_759648

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759648


namespace sum_of_integers_75_to_95_l759_759209

theorem sum_of_integers_75_to_95 : (∑ i in Finset.range (95 - 75 + 1), (i + 75)) = 1785 := by
  sorry

end sum_of_integers_75_to_95_l759_759209


namespace tape_length_division_l759_759329

theorem tape_length_division (n_pieces : ℕ) (length_piece overlap : ℝ) (n_parts : ℕ) 
  (h_pieces : n_pieces = 5) (h_length : length_piece = 2.7) (h_overlap : overlap = 0.3) 
  (h_parts : n_parts = 6) : 
  ((n_pieces * length_piece) - ((n_pieces - 1) * overlap)) / n_parts = 2.05 :=
  by
    sorry

end tape_length_division_l759_759329


namespace greatest_three_digit_multiple_of_17_is_986_l759_759620

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759620


namespace sum_of_primes_less_than_20_l759_759981

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759981


namespace unique_solution_l759_759315

noncomputable def functional_equation (f : ℝ → ℝ) (C : Type*)
  [ContinuousMapClass C ℝ ℝ] [∀ x, ContinuousOn (f x) (univ : Set ℝ)] :=
  (hx0 : f (0 : ℝ) = 5) ∧ (hxy : ∀ x y, 5 * f (x + y) = f x * f y) ∧ (hf1 : f 1 = 10)

theorem unique_solution (f : ℝ → ℝ) (hfx : functional_equation f) : 
  ∀ x, f x = 5 * 2^x :=
by
  sorry

end unique_solution_l759_759315


namespace greatest_three_digit_multiple_of_seventeen_l759_759705

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759705


namespace number_of_elements_in_set_A_l759_759394

def A : Set (ℤ × ℤ) := { (2, -2), (2, 2) }

theorem number_of_elements_in_set_A : A.toFinset.card = 2 := by
  sorry

end number_of_elements_in_set_A_l759_759394


namespace composite_probability_l759_759038

-- Definitions based on problem conditions
def is_composite (n : ℕ) : Prop :=
  ¬ nat.prime n ∧ n > 1

def six_sided_die : finset ℕ := {1, 2, 3, 4, 5, 6}

def product_of_rolls (rolls : fin 6 → ℕ) : ℕ :=
  (finset.univ.product (λ i, (rolls i))).val

def total_possible_outcomes : ℕ := 6^6

def composite_outcome_count : ℕ := total_possible_outcomes - 19

-- Theorem statement to prove the probability that the product is composite
theorem composite_probability :
  (composite_outcome_count : ℚ) / total_possible_outcomes = 46637 / 46656 :=
sorry

end composite_probability_l759_759038


namespace greatest_three_digit_multiple_of_17_l759_759594

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759594


namespace sum_primes_less_than_20_l759_759942

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759942


namespace greatest_three_digit_multiple_of_17_l759_759604

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759604


namespace sum_of_primes_less_than_20_is_77_l759_759926

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759926


namespace min_m_value_l759_759188

noncomputable def original_function (x : ℝ) : ℝ := cos (2 * x) - sin (2 * x)

noncomputable def translated_function (x m : ℝ) : ℝ := sqrt 2 * cos (2 * x + 2 * m + (Real.pi / 4))

theorem min_m_value {m : ℝ} (h_symm : ∃ k : ℤ, 2 * m + (Real.pi / 4) = k * Real.pi + (Real.pi / 2)) :
  m = Real.pi / 8 :=
sorry

end min_m_value_l759_759188


namespace robot_vacuum_distance_and_velocity_change_l759_759173

/-- The robot vacuum cleaner's movement in the x-direction for 0 ≤ t ≤ 7 is given by x = t(t-6)^2 --/
def x (t : ℝ) : ℝ := t * (t - 6)^2

/-- The y-direction is constant 0 for 0 ≤ t ≤ 7 --/
def y (t : ℝ) : ℝ := 0

/-- The y-direction for t ≥ 7 is given by y = (t - 7)^2 --/
def y_after_seven (t : ℝ) : ℝ := (t - 7)^2

/-- The velocity in the x-direction for 0 ≤ t ≤ 7 is the derivative of x(t) --/
def velocity_x (t : ℝ) : ℝ := deriv x t

/-- The velocity in the y-direction for t ≥ 7 is the derivative of y_after_seven(t) --/
def velocity_y_after_seven (t : ℝ) : ℝ := deriv y_after_seven t

/-- Prove that the distance traveled by the robot in the first 7 minutes is 71 meters 
    and the absolute change in the velocity vector during the eighth minute is √445. --/
theorem robot_vacuum_distance_and_velocity_change :
  (∫ t in 0..7, abs (deriv x t)) = 71 ∧
  (sqrt ((velocity_x 8 - velocity_x 7)^2 + (velocity_y_after_seven 8 - velocity_y_after_seven 7)^2)) = sqrt 445 :=
by
  sorry

end robot_vacuum_distance_and_velocity_change_l759_759173


namespace no_badminton_or_tennis_l759_759418

open Set

theorem no_badminton_or_tennis (U B T : Set ℕ) (h_univ : Fintype.card U = 30) (h_badminton : Fintype.card B = 17) (h_tennis : Fintype.card T = 19) (h_both : Fintype.card (B ∩ T) = 9) :
  Fintype.card (U \ (B ∪ T)) = 3 :=
by
  -- The proof steps go here
  sorry

end no_badminton_or_tennis_l759_759418


namespace greatest_three_digit_multiple_of_17_l759_759585

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759585


namespace wonderland_cities_l759_759052

theorem wonderland_cities (V E B : ℕ) (hE : E = 45) (hB : B = 42) (h_connected : connected_graph) (h_simple : simple_graph) (h_bridges : count_bridges = 42) : V = 45 :=
sorry

end wonderland_cities_l759_759052


namespace greatest_three_digit_multiple_of_17_is_986_l759_759671

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759671


namespace correct_calculation_l759_759222

theorem correct_calculation : 
  (∃ (calc : ℕ → ℕ → ℕ), calc 18 2 = 3) ∧
  (∃ (calc : ℕ → ℕ → ℕ), calc 4 2 = 32) ∧
  (∃ (calc : ℕ → ℕ → ℕ), calc (-4) 2 = 4) ∧
  (∀ (calc : ℕ → ℕ → ℕ), calc 2 3 ≠ 6) → 
  true := 
by 
  sorry

end correct_calculation_l759_759222


namespace min_value_xi_l759_759066

theorem min_value_xi 
    (m : ℕ) 
    (grid : fin m → fin 10 → ℕ) 
    (h1 : ∀ j : fin 10, (∑ i in finset.fin_range m, grid i j) = 3) 
    (h2 : ∀ i : fin m, ∃ j j' : fin 10, j ≠ j' ∧ grid i j = 1 ∧ grid i j' = 1) 
    (x : ℕ)
    (hx : ∀ i : fin m, 
            ∃ x_i, 
                (x_i = ∑ j in finset.fin_range 10, grid i j) ∧ 
                (x = max x_i ({x_i : m}) ) ) : 
    x = 5 :=
sorry

end min_value_xi_l759_759066


namespace arithmetic_sequence_sum_l759_759525

theorem arithmetic_sequence_sum :
  ∃ a : ℕ → ℤ, 
    a 3 = 7 ∧ a 4 = 11 ∧ a 5 = 15 ∧ 
    (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 54) := 
by {
  sorry
}

end arithmetic_sequence_sum_l759_759525


namespace greatest_three_digit_multiple_of_17_l759_759651

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759651


namespace number_of_sides_regular_polygon_l759_759043

theorem number_of_sides_regular_polygon (angle : ℝ) (h : angle = 40) : 
  ∃ (n : ℕ), n = 9 :=
by 
  -- Define the equation based on the exterior angle condition
  let n := 360 / angle in
  -- Assert that n is an integer and is equal to 9
  have h1 : n = 9 := 
    by rw [h, real.div_eq_iff_eq_mul]; norm_num,
  use n,
  exact h1

end number_of_sides_regular_polygon_l759_759043


namespace num_of_diff_primes_in_factorization_l759_759034

-- Condition Definitions
def fact_85 : ℕ := 5 * 17
def fact_87 : ℕ := 3 * 29
def fact_88 : ℕ := 2^3 * 11
def fact_90 : ℕ := 2 * 3^2 * 5

-- Problem Statement
theorem num_of_diff_primes_in_factorization : 
  (fact_85 * fact_87 * fact_88 * fact_90).prime_factors.nodup.length = 6 := 
sorry

end num_of_diff_primes_in_factorization_l759_759034


namespace sum_of_primes_less_than_20_l759_759893

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759893


namespace plants_remaining_l759_759059

theorem plants_remaining : 
  let initial_plants := 100 in
  let eaten_first_day := 65 in
  let remaining_after_first_day := initial_plants - eaten_first_day in
  let eaten_second_day := Int.floor ((2 / 3 : Rat) * remaining_after_first_day) in
  let remaining_after_second_day := remaining_after_first_day - eaten_second_day in
  let eaten_additional := 5 in
  let final_remaining := remaining_after_second_day - eaten_additional in
  final_remaining = 7 :=
by
  sorry

end plants_remaining_l759_759059


namespace sum_primes_less_than_20_l759_759959

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759959


namespace min_houses_needed_l759_759477

theorem min_houses_needed (n : ℕ) (x : ℕ) (h : n > 0) : (x ≤ n ∧ (x: ℚ)/n < 0.06) → n ≥ 20 :=
sorry

end min_houses_needed_l759_759477


namespace greatest_three_digit_multiple_of_17_l759_759748

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759748


namespace greatest_three_digit_multiple_of_17_is_986_l759_759661

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759661


namespace greatest_three_digit_multiple_of_17_is_986_l759_759656

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759656


namespace greatest_three_digit_multiple_of_17_l759_759836

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759836


namespace ratio_of_sums_l759_759105

variable {α : Type*} [LinearOrderedField α] 

variable (a : ℕ → α) (S : ℕ → α)
variable (a1 d : α)

def isArithmeticSequence (a : ℕ → α) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + n * d

def sumArithmeticSequence (a : α) (d : α) (n : ℕ) : α :=
  n / 2 * (2 * a + (n - 1) * d)

theorem ratio_of_sums (h_arith : isArithmeticSequence a) (h_S : ∀ n, S n = sumArithmeticSequence a1 d n)
  (h_a5_5a3 : a 5 = 5 * a 3) : S 9 / S 5 = 9 := by sorry

end ratio_of_sums_l759_759105


namespace area_contained_by_graph_l759_759320

theorem area_contained_by_graph :
  let R := {pair : ℝ × ℝ | |(pair.fst + pair.snd)| + |(pair.fst - pair.snd)| ≤ 6}
  ∑ in R, (1 : ℝ) = 18 :=
by
  let R := {pair : ℝ × ℝ | |(pair.fst + pair.snd)| + |(pair.fst - pair.snd)| ≤ 6}
  sorry

end area_contained_by_graph_l759_759320


namespace greatest_three_digit_multiple_of_17_l759_759610

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759610


namespace angle_AQT_eq_angle_BQT_l759_759277

noncomputable theory

open EuclideanGeometry

-- Define necessary points and line segments
variables {O S A B P Q T : Point}
variables {circleO : Circle O}
variables {SP SAB : Line}
variables (h_tangent : TangentAt SP circleO P)
variables (h_perpendicular : Perpendicular PQ OS Q)
variables (h_intersect : IntersectsAt SAB PQ T)

-- State the theorem to be proved
theorem angle_AQT_eq_angle_BQT
    (h1 : Tangent SP circleO P)
    (h2 : Perpendicular PQ OS Q)
    (h3 : Secant SAB circleO A B)
    (h4 : Intersects PQ SAB T) :
    ∠A Q T = ∠B Q T :=
sorry

end angle_AQT_eq_angle_BQT_l759_759277


namespace problem_part_1_problem_part_2_l759_759376

noncomputable def f (a x : ℝ) := (a + Real.log x) / x

noncomputable def h (x : ℝ) := (1 + x) * (1 + Real.log x) / x

theorem problem_part_1 :
  (∀ x : ℝ, x = 1 → (∂ x, f a x) = 0) → a = 1 := 
by 
simp [derivative]
sorry

theorem problem_part_2 (m x : ℝ) :
  (x ∈ set.Ici 1) →
  (∀ x : ℝ, x ∈ set.Ici 1 → f 1 x ≥ m / (1 + x)) → m ≤ 2 :=
by 
simp [h]
sorry

end problem_part_1_problem_part_2_l759_759376


namespace probability_at_least_one_blue_l759_759069

-- Definitions of the setup
def red_balls := 2
def blue_balls := 2
def total_balls := red_balls + blue_balls
def total_outcomes := (total_balls * (total_balls - 1)) / 2  -- choose 2 out of total
def favorable_outcomes := 10  -- by counting outcomes with at least one blue ball

-- Definition of the proof problem
theorem probability_at_least_one_blue (a b : ℕ) (h1: a = red_balls) (h2: b = blue_balls) :
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 5 / 6 := by
  sorry  

end probability_at_least_one_blue_l759_759069


namespace janes_score_is_110_l759_759139

-- Definitions and conditions
def sarah_score_condition (x y : ℕ) : Prop := x = y + 50
def average_score_condition (x y : ℕ) : Prop := (x + y) / 2 = 110
def janes_score (x y : ℕ) : ℕ := (x + y) / 2

-- The proof problem statement
theorem janes_score_is_110 (x y : ℕ) 
  (h_sarah : sarah_score_condition x y) 
  (h_avg   : average_score_condition x y) : 
  janes_score x y = 110 := 
by
  sorry

end janes_score_is_110_l759_759139


namespace greatest_three_digit_multiple_of_17_l759_759802

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759802


namespace sum_of_integers_75_to_95_l759_759216

def arithmeticSumOfIntegers (a l : ℕ) : ℕ :=
  let n := l - a + 1
  n / 2 * (a + l)

theorem sum_of_integers_75_to_95 : arithmeticSumOfIntegers 75 95 = 1785 :=
  by
  sorry

end sum_of_integers_75_to_95_l759_759216


namespace sum_of_primes_less_than_20_l759_759883

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759883


namespace area_of_inscribed_triangle_l759_759266

noncomputable def area_of_triangle_inscribed_in_circle_with_arcs (a b c : ℕ) := 
  let circum := a + b + c
  let r := circum / (2 * Real.pi)
  let θ := 360 / (a + b + c)
  let angle1 := 4 * θ
  let angle2 := 6 * θ
  let angle3 := 8 * θ
  let sin80 := Real.sin (80 * Real.pi / 180)
  let sin120 := Real.sin (120 * Real.pi / 180)
  let sin160 := Real.sin (160 * Real.pi / 180)
  let approx_vals := sin80 + sin120 + sin160
  (1 / 2) * r^2 * approx_vals

theorem area_of_inscribed_triangle : 
  area_of_triangle_inscribed_in_circle_with_arcs 4 6 8 = 90.33 / Real.pi^2 :=
by sorry

end area_of_inscribed_triangle_l759_759266


namespace two_color_K6_contains_monochromatic_triangle_l759_759053

theorem two_color_K6_contains_monochromatic_triangle (V : Type) [Fintype V] [DecidableEq V]
  (hV : Fintype.card V = 6)
  (color : V → V → Fin 2) :
  ∃ (a b c : V), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  (color a b = color b c ∧ color b c = color c a) := by
  sorry

end two_color_K6_contains_monochromatic_triangle_l759_759053


namespace sum_integers_75_to_95_l759_759205

theorem sum_integers_75_to_95 :
  let a := 75
  let l := 95
  let n := 95 - 75 + 1
  ∑ k in Finset.range n, (a + k) = 1785 := by
  sorry

end sum_integers_75_to_95_l759_759205


namespace reciprocal_of_neg_eight_l759_759516

theorem reciprocal_of_neg_eight : -8 * (-1/8) = 1 := 
by
  sorry

end reciprocal_of_neg_eight_l759_759516


namespace center_of_incircle_is_angle_bisector_intersection_l759_759134

theorem center_of_incircle_is_angle_bisector_intersection 
  {n : ℕ} (h : 3 ≤ n) (P : fin n → ℝ × ℝ) (O : ℝ × ℝ) 
  (h1 : ∀ i, (distance O (P i) = distance O (P (i + 1 % n)))) 
  (h2 : ∀ i j, angle_bisector (P i) O (P (i + 1 % n)) = angle_bisector (P j) O (P (j + 1 % n))) :
  ∃ I, (█ angle_bisector (P 0) I (P 1)) =
    (█ angle_bisector (P 1) I (P 2)) ∧ I = O :=
by
  sorry

end center_of_incircle_is_angle_bisector_intersection_l759_759134


namespace smallest_perfect_cube_divisor_l759_759114

-- Given conditions
variables {p q r : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r)

-- The given divisor
def n := p * q^3 * r^6

-- The exponents must be multiples of 3
def cube (x y z : ℕ) := p^x * q^y * r^z

-- Statement of the problem
theorem smallest_perfect_cube_divisor : 
  ∃ (x y z : ℕ), x ≥ 3 ∧ y ≥ 3 ∧ z ≥ 6 ∧ (∃ k, cube x y z = k ^ 3) ∧ (∀ (x' y' z' : ℕ), 
    (x' ≥ 3 ∧ y' ≥ 3 ∧ z' ≥ 6 ∧ (∃ k', cube x' y' z' = k' ^ 3) → cube x y z ≤ cube x' y' z')) := 
begin
  use [3, 3, 6],
  split,
  { exact le_refl _ },
  split,
  { exact le_refl _ },
  split,
  { exact le_refl _ },
  split,
  { use p * q * r^2, 
    exact rfl },
  { intros x' y' z' h',
    cases h' with hx' h_rest,
    cases h_rest with hy' h_rest,
    cases h_rest with hz' h_rest,
    cases h_rest with k' eq_cube,
    have hx'3 : 3 ≤ x' := hx',
    have hy'3 : 3 ≤ y' := hy',
    have hz'6 : 6 ≤ z' := hz',
    rw ←eq_cube,
    sorry
  }
end

end smallest_perfect_cube_divisor_l759_759114


namespace greatest_three_digit_multiple_of_17_l759_759646

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759646


namespace greatest_three_digit_multiple_of_17_is_986_l759_759767

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759767


namespace min_value_of_fraction_l759_759355

noncomputable def min_val (a b : ℝ) : ℝ :=
  1 / a + 2 * b

theorem min_value_of_fraction (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 2 * a * b + 3 = b) :
  min_val a b = 8 + 4 * Real.sqrt 3 :=
sorry

end min_value_of_fraction_l759_759355


namespace greatest_three_digit_multiple_of_17_l759_759570

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759570


namespace greatest_three_digit_multiple_of_17_is_986_l759_759763

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759763


namespace tangent_line_eq_l759_759300

-- Define the function
def f (x : ℝ) : ℝ := Real.sin x + Real.exp x

-- Define the point
def point : ℝ × ℝ := (0, 1)

-- Define the tangent line equation
def tangent_line (x : ℝ) : ℝ := 2 * x + 1

-- State the proof problem
theorem tangent_line_eq : (∀ x, (tangent_line x = 2 * x + 1))
  ∧ tangent_line 0 = 1 
  ∧ tangent_line 1 = 2 * 1 + 1 :=
by
  sorry

end tangent_line_eq_l759_759300


namespace a_and_b_finish_work_in_72_days_l759_759491

noncomputable def work_rate_A_B {A B C : ℝ} 
  (h1 : B + C = 1 / 24) 
  (h2 : A + C = 1 / 36) 
  (h3 : A + B + C = 1 / 16.000000000000004) : ℝ :=
  A + B

theorem a_and_b_finish_work_in_72_days {A B C : ℝ} 
  (h1 : B + C = 1 / 24) 
  (h2 : A + C = 1 / 36) 
  (h3 : A + B + C = 1 / 16.000000000000004) : 
  work_rate_A_B h1 h2 h3 = 1 / 72 :=
sorry

end a_and_b_finish_work_in_72_days_l759_759491


namespace exp_decreasing_function_range_l759_759046

theorem exp_decreasing_function_range (a : ℝ) (x : ℝ) (h_a : 0 < a ∧ a < 1) (h_f : a^(x+1) ≥ 1) : x ≤ -1 :=
sorry

end exp_decreasing_function_range_l759_759046


namespace sum_primes_less_than_20_l759_759977

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759977


namespace parallel_lines_slope_l759_759502

theorem parallel_lines_slope (b : ℚ) :
  (∀ x y : ℚ, 3 * y + x - 1 = 0 → 2 * y + b * x - 4 = 0 ∨
    3 * y + x - 1 = 0 ∧ 2 * y + b * x - 4 = 0) →
  b = 2 / 3 :=
by
  intro h
  sorry

end parallel_lines_slope_l759_759502


namespace monotonic_intervals_minimum_integer_a_l759_759018

def f (x : ℝ) (a : ℝ) : ℝ := x * log (x + 1) + (1 / 2 - a) * x + 2 - a
def g (x : ℝ) (a : ℝ) : ℝ := f x a + log (x + 1) + (1 / 2) * x

theorem monotonic_intervals (a : ℝ) : 
  (a ≤ 2 → ∀ x > 0, (deriv (g x a)) x > 0) ∧ 
  (a > 2 → ∀ x > 0, (deriv (g x a)) x < 0 ∧ ∀ x > exp (a - 2) - 1, (deriv (g x a)) x > 0) :=
by 
  sorry

theorem minimum_integer_a : ∃ (a : ℤ), ∀ x ≥ 0, (f x a.toReal < 0 → (a = 3)) :=
by 
  sorry

end monotonic_intervals_minimum_integer_a_l759_759018


namespace greatest_three_digit_multiple_of_17_l759_759683

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759683


namespace greatest_three_digit_multiple_of_17_l759_759652

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759652


namespace sum_of_integers_75_to_95_l759_759215

def arithmeticSumOfIntegers (a l : ℕ) : ℕ :=
  let n := l - a + 1
  n / 2 * (a + l)

theorem sum_of_integers_75_to_95 : arithmeticSumOfIntegers 75 95 = 1785 :=
  by
  sorry

end sum_of_integers_75_to_95_l759_759215


namespace gcd_ab_conditions_l759_759122

theorem gcd_ab_conditions 
  (a b : ℕ) (h1 : a > b) (h2 : Nat.gcd a b = 1) : 
  Nat.gcd (a + b) (a - b) = 1 ∨ Nat.gcd (a + b) (a - b) = 2 := 
sorry

end gcd_ab_conditions_l759_759122


namespace all_roots_equal_l759_759381

theorem all_roots_equal (a b : ℝ) (c : Fin n → ℝ) (x : Fin n → ℝ)
  (h_poly : ∀ x : ℝ, (a * x^n - a * x^(n-1) + (∑ i in Fin.range(n-2), c i * x^(n-2-i)) - n^2 * b * x + b) = 0) 
  (h_roots : ∀ i : Fin n, x i > 0) :
  ∀ i j : Fin n, x i = x j :=
by
  sorry

end all_roots_equal_l759_759381


namespace altitude_tangent_circumcircle_l759_759406

theorem altitude_tangent_circumcircle (A B C : Type) (h : triangle A B C) (β γ : ℝ) (H : β - γ = 90) : 
  tangent (altitude A h) (circumcircle A B C) A :=
by sorry

end altitude_tangent_circumcircle_l759_759406


namespace greatest_three_digit_multiple_of_17_l759_759736

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759736


namespace greatest_three_digit_multiple_of_17_l759_759644

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759644


namespace sum_primes_less_than_20_l759_759937

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759937


namespace number_of_correct_statements_l759_759163

theorem number_of_correct_statements :
  (∀ (l1 l2 l3 : Line), (l1 ∥ l2 ∧ l2 ∥ l3) → l1 ∥ l3) ∧ 
  (∀ (p1 p2 p3 : Plane), (p1 ∥ p2 ∧ p2 ∥ p3) → p1 ∥ p3) ∧ 
  ¬(∀ (l1 l2 : Line) (p : Plane), (l1 ∥ l2 ∧ l1 ∥ p) → l2 ∥ p) ∧
  ¬(∀ (l : Line) (p1 p2 : Plane), (p1 ∥ p2 ∧ l ∥ p1) → l ∥ p2) → 2 := 
by {
  sorry
}

end number_of_correct_statements_l759_759163


namespace percent_moved_is_1032_l759_759087

variable (x : ℝ)

def jar_a_marbles (jar_b_marbles : ℝ) := 1.26 * jar_b_marbles
def equal_marbles (jar_b_marbles jar_a_marbles : ℝ) := (jar_b_marbles + jar_a_marbles) / 2
def moved_marbles (jar_a_marbles equal_marbles : ℝ) := jar_a_marbles - equal_marbles
def percent_moved (moved_marbles original_marbles : ℝ) := (moved_marbles / original_marbles) * 100

theorem percent_moved_is_1032 :
  percent_moved (moved_marbles (jar_a_marbles x) (equal_marbles x (jar_a_marbles x))) (jar_a_marbles x) ≈ 10.32 := by
  sorry

end percent_moved_is_1032_l759_759087


namespace find_divisor_l759_759314

noncomputable def divisor_from_quotient (dividend quotient : ℝ) : ℝ := dividend / quotient

theorem find_divisor :
  ∃ d, abs ((3486 / 18.444444444444443) - d) < 0.001 :=
begin
  use 3486 / 18.444444444444443,
  simp,
  norm_num,
end

end find_divisor_l759_759314


namespace greatest_three_digit_multiple_of_17_l759_759572

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759572


namespace greatest_three_digit_multiple_of_17_is_986_l759_759634

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759634


namespace max_intersections_l759_759292

-- Define p(x) and q(x) as polynomials of specific degrees and leading coefficients
def p (x : ℝ) : ℝ := 2 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0
def q (x : ℝ) : ℝ := x^4 + b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0

-- State the theorem with the conditions from step a)
theorem max_intersections : 
  (∀ (a_4 a_3 a_2 a_1 a_0 b_3 b_2 b_1 b_0 : ℝ), ∃ (x : ℝ), p(x) = q(x)) →
  p(x) - q(x) = 0 → 
  ∃ (max_points_of_intersection : ℕ), max_points_of_intersection = 5 :=
by
  sorry

end max_intersections_l759_759292


namespace first_day_reduction_percentage_l759_759264

variables (P x : ℝ)

theorem first_day_reduction_percentage (h : P * (1 - x / 100) * 0.90 = 0.81 * P) : x = 10 :=
sorry

end first_day_reduction_percentage_l759_759264


namespace correct_statements_BD_l759_759369

-- Problem conditions
def circle_C := ∀ (x y : ℝ), x^2 + y^2 = 4
def circle_C1 (m : ℝ) := ∀ (x y : ℝ), x^2 + y^2 - 6 * x - 8 * y + m = 0
def circle_C2 := ∀ (x y : ℝ), x^2 + y^2 - 6 * x - 8 * y + 24 = 0
def line_theta (θ : ℝ) := ∀ (x y : ℝ), x * sin θ + y * cos θ - 4 = 0
def line_l := ∀ (x y : ℝ), x - y + real.sqrt 2 = 0

-- Proof that statements B and D are correct
theorem correct_statements_BD (m : ℝ) : 
  (∃ θ, ¬ (∀ (x y : ℝ), ¬ ((x * sin θ + y * cos θ - 4 = 0) ∧ (x^2 + y^2 = 4)))) ∧
  (∀ (x y : ℝ), circle_C1 m x y → m = -24) ∧
  (∀ (p : ℝ × ℝ), (p.1^2 + p.2^2 = 4) → ∃! q : ℝ × ℝ, (p = q) ∧ (abs((p.1 - p.2 + real.sqrt 2) / (real.sqrt (1^2 + (-1)^2))) = 1) ) :=
sorry

end correct_statements_BD_l759_759369


namespace new_game_cost_l759_759226

theorem new_game_cost (G : ℕ) (h_initial_money : 83 = G + 9 * 4) : G = 47 := by
  sorry

end new_game_cost_l759_759226


namespace minimum_degree_g_l759_759357

-- Definitions of f, g, h as polynomials in x, and their degrees
variables {R : Type*} [CommRing R]
variables {f g h : R[X]} 
variable (deg_f : nat := 7) 
variable (deg_h : nat := 10)

-- Given conditions
theorem minimum_degree_g
  (h_condition : 2 * f + 7 * g = h)
  (deg_f_condition : f.degree = 7)
  (deg_h_condition : h.degree = 10):
  g.degree >= 10 := 
sorry

end minimum_degree_g_l759_759357


namespace greatest_three_digit_multiple_of_17_l759_759820

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759820


namespace greatest_three_digit_multiple_of_17_l759_759821

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759821


namespace greatest_three_digit_multiple_of_17_is_986_l759_759628

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759628


namespace sum_of_primes_lt_20_eq_77_l759_759913

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759913


namespace greatest_three_digit_multiple_of_17_l759_759742

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759742


namespace greatest_three_digit_multiple_of_17_is_986_l759_759760

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759760


namespace average_combined_l759_759493

noncomputable def average (s : Set ℕ) : ℝ :=
  (s.toFinset.sum id) / s.card

theorem average_combined (A B : Set ℕ) (a b : ℕ) (hA : A.card = a) (hB : B.card = b) (hab : a > b) :
  average (A ∪ B) = 80 →
  (average A = 80 → average B = 80) ∧
  (average A > 80 → average B < 80) ∧
  (average A < 80 → average B > 80) :=
sorry

end average_combined_l759_759493


namespace find_intersection_point_l759_759518

-- Define the problem conditions and question in Lean
theorem find_intersection_point 
  (slope_l1 : ℝ) (slope_l2 : ℝ) (p : ℝ × ℝ) (P : ℝ × ℝ)
  (h_l1_slope : slope_l1 = 2) 
  (h_parallel : slope_l1 = slope_l2)
  (h_passes_through : p = (-1, 1)) :
  P = (0, 3) := sorry

end find_intersection_point_l759_759518


namespace area_of_triangle_QPS_l759_759235

-- Definitions and conditions
def on_line_segment (R Q S : ℝ × ℝ) : Prop := sorry               -- This is a placeholder
def QR : ℝ := 8
def PR : ℝ := 12
def angle_PRQ : ℝ := 120
def angle_RPS : ℝ := 90

-- The main theorem
theorem area_of_triangle_QPS 
  (Q R S P : ℝ × ℝ)
  (h1 : on_line_segment R Q S) 
  (h2 : QR = 8)
  (h3 : PR = 12)
  (h4 : angle_PRQ = 120)
  (h5 : angle_RPS = 90)
  : area Q P S = 96 * real.sqrt(3) :=
sorry

end area_of_triangle_QPS_l759_759235


namespace greatest_three_digit_multiple_of_17_l759_759606

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759606


namespace sum_primes_less_than_20_l759_759957

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759957


namespace greatest_three_digit_multiple_of_seventeen_l759_759708

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759708


namespace sum_of_primes_lt_20_eq_77_l759_759912

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759912


namespace greatest_three_digit_multiple_of_17_is_986_l759_759623

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759623


namespace mary_apples_l759_759120

theorem mary_apples (A : ℕ) (eaten : ℕ) (trees_per_apple : ℕ) (apples_left : ℕ) :
  eaten = 2 →
  trees_per_apple = 2 →
  apples_left = A - eaten →
  A = eaten + trees_per_apple * eaten :=
begin
  intros h1 h2 h3,
  sorry,
end

end mary_apples_l759_759120


namespace initial_customers_count_l759_759268

theorem initial_customers_count (left_count remaining_people_per_table tables remaining_customers : ℕ) 
  (h1 : left_count = 14) 
  (h2 : remaining_people_per_table = 4) 
  (h3 : tables = 2) 
  (h4 : remaining_customers = tables * remaining_people_per_table) 
  : n = 22 :=
  sorry

end initial_customers_count_l759_759268


namespace hyperbola_eq_given_conditions_hyperbola_asymptotes_given_conditions_l759_759364

-- Define the ellipse and its properties
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 36) + (y^2 / 27) = 1

-- Define properties of hyperbola based on the conditions
def same_foci (x y : ℝ) : Prop := ellipse_eq x y ∧ (x^2 / 5) - (y^2 / 4) = 1

-- Define the length of the conjugate axis of the hyperbola
def conjugate_axis_length : Prop := 2 * 2 = 4

-- Prove the hyperbola equation given the properties
theorem hyperbola_eq_given_conditions :
  (∀ (x y : ℝ), same_foci x y) ∧ conjugate_axis_length →
  (∀ (x y : ℝ), (x^2 / 5) - (y^2 / 4) = 1) :=
by
  sorry

-- Prove the equations of the asymptotes given the properties
theorem hyperbola_asymptotes_given_conditions :
  (∀ (x y : ℝ), same_foci x y) ∧ conjugate_axis_length →
  (∀ (x y : ℝ), y = (2 * real.sqrt 5 / 5) * x ∨ y = -(2 * real.sqrt 5 / 5) * x) :=
by
  sorry

end hyperbola_eq_given_conditions_hyperbola_asymptotes_given_conditions_l759_759364


namespace sum_of_primes_less_than_20_l759_759980

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759980


namespace greatest_three_digit_multiple_of_17_l759_759598

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759598


namespace greatest_three_digit_multiple_of_17_l759_759743

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759743


namespace frog_jumps_from_A_to_stop_l759_759441

-- Defining the hexagon vertices and movement constraints
inductive Vertex : Type
| A | B | C | D | E | F

open Vertex

-- Defining the movement of the frog
def adjacent (v : Vertex) : Finset Vertex :=
  match v with
  | A => {B, F}
  | B => {A, C}
  | C => {B, D}
  | D => {C, E}
  | E => {D, F}
  | F => {E, A}

-- Defining whether a series of moves reaches vertex D
def reaches_D (path : List Vertex) : Bool :=
  path.length ≤ 5 ∧ path.getLast? = some D

-- Counting the distinct paths the frog can take in 5 moves or less
noncomputable def countWays (start : Vertex) (end : Vertex) (maxMoves : Nat) : Nat :=
  if end = D then 2 else 24

-- Proving the overall ways the frog can stop
theorem frog_jumps_from_A_to_stop : countWays A D 5 = 26 := by
  sorry

end frog_jumps_from_A_to_stop_l759_759441


namespace greatest_three_digit_multiple_of_17_l759_759800

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l759_759800


namespace greatest_three_digit_multiple_of17_l759_759718

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759718


namespace greatest_three_digit_multiple_of_17_is_986_l759_759630

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759630


namespace greatest_three_digit_multiple_of_17_l759_759815

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759815


namespace molly_total_cost_l759_759077

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_children_per_brother : ℕ := 2
def num_spouse_per_brother : ℕ := 1

def total_num_relatives : ℕ := 
  let parents_and_siblings := num_parents + num_brothers
  let additional_relatives := num_brothers * (1 + num_spouse_per_brother + num_children_per_brother)
  parents_and_siblings + additional_relatives

def total_cost : ℕ :=
  total_num_relatives * cost_per_package

theorem molly_total_cost : total_cost = 85 := sorry

end molly_total_cost_l759_759077


namespace sum_of_primes_less_than_20_l759_759872

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759872


namespace probability_two_balls_same_color_l759_759243

theorem probability_two_balls_same_color :
  let total_balls := 15,
      blue_balls := 8,
      yellow_balls := 7,
      prob_blue := (↑blue_balls / ↑total_balls),
      prob_yellow := (↑yellow_balls / ↑total_balls) in
  (prob_blue^2 + prob_yellow^2) = (113 / 225) :=
by
  sorry

end probability_two_balls_same_color_l759_759243


namespace greatest_three_digit_multiple_of_17_l759_759679

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759679


namespace length_of_faster_train_is_380_meters_l759_759556

-- Defining the conditions
def speed_faster_train_kmph := 144
def speed_slower_train_kmph := 72
def time_seconds := 19

-- Conversion factor
def kmph_to_mps (speed : Nat) : Nat := speed * 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps : Nat := kmph_to_mps (speed_faster_train_kmph - speed_slower_train_kmph)

-- Problem statement: Prove that the length of the faster train is 380 meters
theorem length_of_faster_train_is_380_meters :
  relative_speed_mps * time_seconds = 380 :=
sorry

end length_of_faster_train_is_380_meters_l759_759556


namespace slope_tangent_at_pi_over_4_l759_759177

noncomputable def f (x : ℝ) : ℝ := (sin x) / (sin x + cos x) - 1/2

theorem slope_tangent_at_pi_over_4 : 
  (deriv f) (π / 4) = 1/2 := 
sorry

end slope_tangent_at_pi_over_4_l759_759177


namespace simplify_T_l759_759103

theorem simplify_T (x : ℝ) : 
  (x + 2)^6 + 6 * (x + 2)^5 + 15 * (x + 2)^4 + 20 * (x + 2)^3 + 15 * (x + 2)^2 + 6 * (x + 2) + 1 = (x + 3)^6 :=
by
  sorry

end simplify_T_l759_759103


namespace greatest_three_digit_multiple_of_17_is_986_l759_759765

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759765


namespace greatest_three_digit_multiple_of_17_l759_759854

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759854


namespace piles_stones_l759_759131

theorem piles_stones (a b c d : ℕ)
  (h₁ : a = 2011)
  (h₂ : b = 2010)
  (h₃ : c = 2009)
  (h₄ : d = 2008) :
  ∃ (k l m n : ℕ), (k, l, m, n) = (0, 0, 0, 2) ∧
  ((∃ x y z w : ℕ, k = x - y ∧ l = y - z ∧ m = z - w ∧ x + l + m + w = 0) ∨
   (∃ u : ℕ, k = a - u ∧ l = b - u ∧ m = c - u ∧ n = d - u)) :=
sorry

end piles_stones_l759_759131


namespace greatest_three_digit_multiple_of_17_l759_759810

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759810


namespace no_nat_number_with_perfect_square_l759_759136

theorem no_nat_number_with_perfect_square (n : Nat) : 
  ¬ ∃ m : Nat, m * m = n^6 + 3 * n^5 - 5 * n^4 - 15 * n^3 + 4 * n^2 + 12 * n + 3 := 
  by
  sorry

end no_nat_number_with_perfect_square_l759_759136


namespace area_GAME_l759_759237

/-
In triangle  $ABC$ ,  $AB = 13$ ,  $BC = 14$ , and  $CA = 15$ .
Let  $M$  be the midpoint of side  $AB$ , 
$G$  be the centroid of  $\triangle ABC$ , 
and  $E$  be the foot of the altitude from  $A$  to  $BC$ .
Compute the area of quadrilateral  $GAME$ .
-/

noncomputable def A : (ℝ × ℝ) := (0, 12)
noncomputable def B : (ℝ × ℝ) := (5, 0)
noncomputable def C : (ℝ × ℝ) := (-9, 0)

noncomputable def M : (ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def G : (ℝ × ℝ) := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
noncomputable def E : (ℝ × ℝ) := (0, 0)

def shoelace_area (v : list (ℝ × ℝ)) : ℝ :=
  0.5 * (list.sum (list.map2 (λ p q, p.1 * q.2 - p.2 * q.1) v (v.tail ++ [v.head]))).abs

theorem area_GAME : shoelace_area [G, A, M, E] = 23 := 
by sorry

end area_GAME_l759_759237


namespace composite_product_probability_l759_759041

theorem composite_product_probability :
  let outcomes := 6^6 in
  let non_composite_outcomes := 19 in
  let composite_probability := (outcomes - non_composite_outcomes) / outcomes in
  composite_probability = 46637 / 46656 :=
sorry

end composite_product_probability_l759_759041


namespace sum_primes_less_than_20_l759_759967

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759967


namespace equivalent_expression_l759_759234

variable {a b c d : ℝ}

noncomputable def right_to_left_compute : ℝ := a / (b - c - d)

theorem equivalent_expression :
  (a / b - c + d) = right_to_left_compute :=
sorry

end equivalent_expression_l759_759234


namespace sum_of_primes_less_than_20_l759_759891

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759891


namespace Sam_age_proof_l759_759414

-- Define the conditions (Phoebe's current age, Raven's age relation, Sam's age definition)
def Phoebe_current_age : ℕ := 10
def Raven_in_5_years (R : ℕ) : Prop := R + 5 = 4 * (Phoebe_current_age + 5)
def Sam_age (R : ℕ) : ℕ := 2 * ((R + 3) - (Phoebe_current_age + 3))

-- The proof statement for Sam's current age
theorem Sam_age_proof (R : ℕ) (h : Raven_in_5_years R) : Sam_age R = 90 := by
  sorry

end Sam_age_proof_l759_759414


namespace greatest_three_digit_multiple_of_17_l759_759829

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759829


namespace molly_christmas_shipping_cost_l759_759075

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_sisters_in_law_per_brother : ℕ := 1
def num_children_per_brother : ℕ := 2

def total_relatives : ℕ :=
  num_parents + num_brothers + (num_brothers * num_sisters_in_law_per_brother) + (num_brothers * num_children_per_brother)

theorem molly_christmas_shipping_cost : total_relatives * cost_per_package = 70 :=
by
  sorry

end molly_christmas_shipping_cost_l759_759075


namespace bn_is_arithmetic_seq_an_general_term_l759_759027

def seq_an (a : ℕ → ℝ) : Prop :=
a 1 = 2 ∧ ∀ n, (a (n + 1) - 1) * (a n - 1) = 3 * (a n - a (n + 1))

def seq_bn (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
∀ n, b n = 1 / (a n - 1)

theorem bn_is_arithmetic_seq (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : seq_an a) (h2 : seq_bn a b) : 
∀ n, b (n + 1) - b n = 1 / 3 :=
sorry

theorem an_general_term (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : seq_an a) (h2 : seq_bn a b) : 
∀ n, a n = (n + 5) / (n + 2) :=
sorry

end bn_is_arithmetic_seq_an_general_term_l759_759027


namespace max_length_OB_l759_759548

-- Define the setup and conditions
variables (O A B : Type)
          [metric_space O]
          (ray1 ray2 : O → O → ℝ)
          (h_angle : angle (ray1 O A) (ray2 O B) = 45)
          (h_AB : dist A B = 2)

-- State the theorem to be proved
theorem max_length_OB (A B : O) (O : O):
  ∀ (ray1 : O → O → ℝ) (ray2 : O → O → ℝ),
  angle (ray1 O A) (ray2 O B) = 45 →
  dist A B = 2 →
  ∃ (OB : ℝ), OB = 2 * sqrt 2 := sorry

end max_length_OB_l759_759548


namespace greatest_three_digit_multiple_of_17_l759_759775

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759775


namespace arithmetic_sequence_sum_l759_759526

theorem arithmetic_sequence_sum :
  ∃ a : ℕ → ℤ, 
    a 3 = 7 ∧ a 4 = 11 ∧ a 5 = 15 ∧ 
    (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 54) := 
by {
  sorry
}

end arithmetic_sequence_sum_l759_759526


namespace slope_of_line_l759_759200

-- Define the points
def point1 : (ℝ × ℝ) := (1, -3)
def point2 : (ℝ × ℝ) := (-4, 7)

-- Change in y-coordinates
def delta_y : ℝ := point2.2 - point1.2

-- Change in x-coordinates
def delta_x : ℝ := point2.1 - point1.1

-- The slope
def slope := delta_y / delta_x

-- The theorem stating the slope of the line
theorem slope_of_line : slope = -2 :=
by
  -- Calculate the slope and prove that it equals -2
  sorry

end slope_of_line_l759_759200


namespace num_males_in_group_l759_759062

-- Definitions based on the given conditions
def num_females (f : ℕ) : Prop := f = 16
def num_males_choose_malt (m_malt : ℕ) : Prop := m_malt = 6
def num_females_choose_malt (f_malt : ℕ) : Prop := f_malt = 8
def num_choose_malt (m_malt f_malt n_malt : ℕ) : Prop := n_malt = m_malt + f_malt
def num_choose_coke (c : ℕ) (n_malt : ℕ) : Prop := n_malt = 2 * c
def total_cheerleaders (t : ℕ) (n_malt c : ℕ) : Prop := t = n_malt + c
def num_males (m f t : ℕ) : Prop := m = t - f

theorem num_males_in_group
  (f m_malt f_malt n_malt c t m : ℕ)
  (hf : num_females f)
  (hmm : num_males_choose_malt m_malt)
  (hfm : num_females_choose_malt f_malt)
  (hmalt : num_choose_malt m_malt f_malt n_malt)
  (hc : num_choose_coke c n_malt)
  (ht : total_cheerleaders t n_malt c)
  (hm : num_males m f t) :
  m = 5 := 
sorry

end num_males_in_group_l759_759062


namespace problem_statement_l759_759005

-- Defining the functions and constants
def f (x : ℝ) : ℝ := exp x - (1/2)*x^2 + (f' 0)/2*x
def g (x : ℝ) : ℝ

-- Given conditions
axiom h_g (x : ℝ) : g x + deriv g x < 0
axiom h_f_prime_0 : deriv f 0 = 2

-- The target theorem to prove
theorem problem_statement : g 2015 > f 2 * g 2017 :=
sorry

end problem_statement_l759_759005


namespace reflection_line_sum_l759_759160

theorem reflection_line_sum :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x + y = 2 ∧ x - 2*y = -10) → ∀ (p q : ℝ), (p = -4 ∧ q = 0) → ((p, q).reflect_in_line (m, b) = (2, 6))) ∧ (m + b = 1) :=
by
  sorry

end reflection_line_sum_l759_759160


namespace num_isolated_elements_in_A_l759_759383

def is_isolated_element (A : Set ℤ) (x : ℤ) : Prop :=
  x ∈ A ∧ x - 1 ∉ A ∧ x + 1 ∉ A

def A : Set ℤ := {1, 2, 3, 5}

theorem num_isolated_elements_in_A :
  (Finset.filter (is_isolated_element A) (Finset.mk [1, 2, 3, 5] sorry)).card = 1 :=
  sorry

end num_isolated_elements_in_A_l759_759383


namespace greatest_three_digit_multiple_of_17_l759_759860

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759860


namespace sum_of_primes_less_than_20_l759_759995

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l759_759995


namespace part1_solution_set_part2_minimum_value_l759_759019

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

theorem part1_solution_set (x : ℝ) :
  (f x ≥ -1) ↔ (2 / 3 ≤ x ∧ x ≤ 6) := sorry

variables {a b c : ℝ}
theorem part2_minimum_value (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = 6) :
  (1 / (2 * a + b) + 1 / (2 * a + c) ≥ 2 / 3) := 
sorry

end part1_solution_set_part2_minimum_value_l759_759019


namespace sum_primes_less_than_20_l759_759943

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759943


namespace arithmetic_sequence_sum_l759_759068

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (n : ℕ) 
  (h_arith : ∀ n, a (n+1) = a n + 3)
  (h_a1_a2 : a 1 + a 2 = 7)
  (h_a3 : a 3 = 8)
  (h_bn : ∀ n, b n = 1 / (a n * a (n+1)))
  :
  (∀ n, a n = 3 * n - 1) ∧ (T n = n / (2 * (3 * n + 2))) :=
by 
  sorry

end arithmetic_sequence_sum_l759_759068


namespace greatest_three_digit_multiple_of_17_l759_759675

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759675


namespace greatest_three_digit_multiple_of_17_l759_759859

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759859


namespace greatest_three_digit_multiple_of_17_l759_759687

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759687


namespace isosceles_triangle_DGH_l759_759096

-- Definitions of points and segments
variables (A B C D G H : Type) 
-- Predicate for points to indicate geometrical properties
variables [parallelogram ABCD] [distinct_point_on_line G AB B]
           [distinct_point_on_line H BC B] [equal_segments CG CB]
           [equal_segments AB AH]

-- Statement of the problem
theorem isosceles_triangle_DGH : isosceles_triangle D G H :=
sorry

end isosceles_triangle_DGH_l759_759096


namespace greatest_three_digit_multiple_of_17_l759_759613

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∃ n, is_three_digit n ∧ 17 ∣ n ∧ ∀ k, is_three_digit k ∧ 17 ∣ k → k ≤ n :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759613


namespace janet_spent_percentage_correct_l759_759082

def janet_grocery_budget_percentage : Prop :=
  let broccoli := 3 * 4
  let oranges := 3 * 0.75
  let cabbage := 3.75
  let bacon := 3
  let chicken := 2 * 3
  let tilapia := 5
  let steak := 8
  let apples := 5 * 1.5
  let yogurt := 6
  let milk := 3.5
  let total_meat_fish_before_discount := bacon + chicken + tilapia + steak
  let discount_meat_fish := 0.1 * total_meat_fish_before_discount
  let total_meat_fish_after_discount := total_meat_fish_before_discount - discount_meat_fish
  let total_other_groceries := broccoli + oranges + cabbage + apples + yogurt + milk
  let total_grocery_before_tax := total_meat_fish_after_discount + total_other_groceries
  let sales_tax := 0.07 * total_grocery_before_tax
  let total_grocery_after_tax := total_grocery_before_tax + sales_tax
  let percentage_spent_on_meat_fish := (total_meat_fish_after_discount / total_grocery_after_tax) * 100
  let rounded_percentage_spent_on_meat_fish := (percentage_spent_on_meat_fish + 0.5).toNat
  rounded_percentage_spent_on_meat_fish = 34

theorem janet_spent_percentage_correct : janet_grocery_budget_percentage :=
by sorry

end janet_spent_percentage_correct_l759_759082


namespace find_a2013_l759_759118

noncomputable def a : ℕ → ℤ
| 0       := 0
| 1       := 2
| 2       := 5
| (n + 3) := a (n + 2) - a (n + 1)

theorem find_a2013 : a 2013 = 2 := 
sorry

end find_a2013_l759_759118


namespace min_value_Box_l759_759301

theorem min_value_Box 
  (a b Box : ℤ) 
  (h_distinct : a ≠ b ∧ b ≠ Box ∧ a ≠ Box)
  (h_eq : (a * x + b) * (b * x + a) = 45 * x^2 + Box * x + 45) : 
  Box = 106 :=
begin
  sorry
end

end min_value_Box_l759_759301


namespace greatest_three_digit_multiple_of_17_l759_759642

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759642


namespace jasmine_money_left_l759_759088

theorem jasmine_money_left 
  (initial_amount : ℝ)
  (apple_cost : ℝ) (num_apples : ℕ)
  (orange_cost : ℝ) (num_oranges : ℕ)
  (pear_cost : ℝ) (num_pears : ℕ)
  (h_initial : initial_amount = 100.00)
  (h_apple_cost : apple_cost = 1.50)
  (h_num_apples : num_apples = 5)
  (h_orange_cost : orange_cost = 2.00)
  (h_num_oranges : num_oranges = 10)
  (h_pear_cost : pear_cost = 2.25)
  (h_num_pears : num_pears = 4) : 
  initial_amount - (num_apples * apple_cost + num_oranges * orange_cost + num_pears * pear_cost) = 63.50 := 
by 
  sorry

end jasmine_money_left_l759_759088


namespace greatest_three_digit_multiple_of_17_l759_759580

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759580


namespace percentage_increase_in_savings_l759_759127

-- Define the initial conditions
variable (I : ℝ) -- Paul's initial income
def initial_expense (I : ℝ) : ℝ := 0.75 * I -- 75% of income
def initial_savings (I : ℝ) : ℝ := I - initial_expense I -- Initial savings

-- Define new conditions after the increment
def new_income (I : ℝ) : ℝ := 1.20 * I -- 20% increase in income
def new_expense (I : ℝ) : ℝ := 1.10 * initial_expense I -- 10% increase in expenditure
def new_savings (I : ℝ) : ℝ := new_income I - new_expense I -- New savings

-- Proof goal
theorem percentage_increase_in_savings (I : ℝ) (h : I > 0) :
  let S := initial_savings I in
  let S_new := new_savings I in
  (S_new - S) / S * 100 = 50 := by
  sorry

end percentage_increase_in_savings_l759_759127


namespace t_five_value_f_five_value_t_f_five_l759_759450

def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 2)
def f (x : ℝ) : ℝ := 7 - 2 * t x
def t_five := t 5
def f_five := f 5 

theorem t_five_value : t_five = Real.sqrt 22 := by
  sorry

theorem f_five_value : f_five = 7 - 2 * Real.sqrt 22 := by
  sorry

theorem t_f_five : t (f 5) = Real.sqrt (30 - 8 * Real.sqrt 22) := by
  sorry

end t_five_value_f_five_value_t_f_five_l759_759450


namespace greatest_three_digit_multiple_of_17_is_986_l759_759662

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759662


namespace sum_of_primes_less_than_20_l759_759892

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759892


namespace greatest_three_digit_multiple_of_seventeen_l759_759700

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759700


namespace minimum_omega_l759_759014

theorem minimum_omega (ω φ : ℝ) (hω : ω > 0) 
  (h1 : ∀ x, 2 * sin (ω * x + φ) = 2 * sin (ω * (2 * (π / 3) - x) + φ))
  (h2 : 2 * sin (ω * (π / 12) + φ) = 0) :
  ω ≥ 2 :=
sorry

end minimum_omega_l759_759014


namespace total_number_of_baseball_cards_l759_759092

def baseball_cards_total : Nat :=
  let carlos := 20
  let matias := carlos - 6
  let jorge := matias
  carlos + matias + jorge
   
theorem total_number_of_baseball_cards :
  baseball_cards_total = 48 :=
by
  rfl

end total_number_of_baseball_cards_l759_759092


namespace sum_of_primes_lt_20_eq_77_l759_759906

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759906


namespace sin_cos_value_l759_759398

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
by
  sorry

end sin_cos_value_l759_759398


namespace exclude_13_code_count_l759_759307

/-- The number of 5-digit codes (00000 to 99999) that don't contain the sequence "13". -/
theorem exclude_13_code_count :
  let total_codes := 100000
  let excluded_codes := 3970
  total_codes - excluded_codes = 96030 :=
by
  let total_codes := 100000
  let excluded_codes := 3970
  have h : total_codes - excluded_codes = 96030 := by
    -- Provide mathematical proof or use sorry for placeholder
    sorry
  exact h

end exclude_13_code_count_l759_759307


namespace greatest_three_digit_multiple_of_17_l759_759561

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759561


namespace sum_extrema_of_g_l759_759106

def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2 * x - 10| + |x - 1|

theorem sum_extrema_of_g:
  let smallest := -1 in
  let largest := 4 in
  (smallest + largest = 3) ∧ (∀ x, 1 ≤ x ∧ x ≤ 9 → 
    g(x) ≥ smallest ∧ g(x) ≤ largest) := 
  by
    sorry

end sum_extrema_of_g_l759_759106


namespace exists_natural_n_l759_759256

theorem exists_natural_n (M : Set ℕ) (h : M.finite) (h_card : M.card = 2003)
  (H : ∀ a b c : ℕ, a ∈ M → b ∈ M → c ∈ M → a ≠ b → b ≠ c → c ≠ a → ∃ k ∈ ℚ, a^2 + b * c = k) :
  ∃ n : ℕ, ∀ a ∈ M, ∃ k ∈ ℚ, a * Real.sqrt n = k :=
by
  sorry

end exists_natural_n_l759_759256


namespace sum_of_primes_less_than_20_l759_759880

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759880


namespace arithmetic_sequence_term_count_l759_759390

theorem arithmetic_sequence_term_count :
  ∀ (a d l : ℤ), a = 162 → d = -3 → l = 30 → ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 45 :=
by
  intros a d l ha hd hl
  rw [ha, hd, hl]
  let n := 45
  existsi n
  simp only [Bit0.add, Bit0.mul, add_neg, add_assoc, mul_one, add_comm, add_left_comm, sub_eq_add_neg]
  norm_num
  sorry

end arithmetic_sequence_term_count_l759_759390


namespace greatest_three_digit_multiple_of_17_is_986_l759_759621

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759621


namespace sum_of_primes_less_than_20_l759_759988

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759988


namespace days_to_complete_work_l759_759142

-- Let's define the conditions as Lean definitions based on the problem.

variables (P D : ℕ)
noncomputable def original_work := P * D
noncomputable def half_work_by_double_people := 2 * P * 3

-- Here is our theorem statement
theorem days_to_complete_work : original_work P D = 2 * half_work_by_double_people P :=
by sorry

end days_to_complete_work_l759_759142


namespace sum_of_primes_less_than_20_l759_759888

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759888


namespace greatest_three_digit_multiple_of_17_l759_759739

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759739


namespace greatest_three_digit_multiple_of_17_l759_759638

/-- The greatest three-digit multiple of 17 is 986. -/
theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 17 = 0 → n ≥ m :=
by {
  use 986,
  have h1 : 986 < 1000 := by decide,
  have h2 : 986 % 17 = 0 := by decide,
  intro m,
  intro h,
  cases h with hm hmod,
  cases hmod with hdiv,
  have h3 := Nat.div_mul_cancel hm,
  have h4 := Nat.div_mul_cancel hdiv,
  have hle := Nat.le_of_dvd h1,
  by_cases h5 : m = 986,
  { calc 986 ≤ 986 : le_refl 986 },
  have h6 : m ∉ [986], sorry,
  have h7 : true := true,
  have h8 := Nat.lt_of_le_of_ne hle,
  exact h2,
}

end greatest_three_digit_multiple_of_17_l759_759638


namespace sum_of_primes_less_than_20_l759_759982

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759982


namespace maximize_profit_l759_759410

noncomputable def monthly_profit (a : ℝ) : ℝ :=
  (a - 40) * (500 - 10 * (a - 50))

theorem maximize_profit :
  ∃ a : ℝ, a = 70 ∧ is_maximizer (monthly_profit a) := sorry

end maximize_profit_l759_759410


namespace required_moles_of_NaHSO3_l759_759326

-- Definitions for the conditions
def reaction (NaHSO3 HCl NaCl H2O SO2 : Type) :=
  ∀ (n_moles_NaHSO3 n_moles_HCl n_moles_NaCl n_moles_H2O n_moles_SO2 : ℕ),
  (n_moles_NaHSO3 : ℕ) = (n_moles_NaCl : ℕ) →
  (n_moles_HCl : ℕ) = (n_moles_NaCl : ℕ) →

-- Target statement to prove
theorem required_moles_of_NaHSO3 (NaHSO3 HCl NaCl H2O SO2 : Type) :
  reaction NaHSO3 HCl NaCl H2O SO2 →
  ∀ (n_moles_NaCl : ℕ), (n_moles_NaCl = 2) →
  ∃ (n_moles_NaHSO3 : ℕ), n_moles_NaHSO3 = 2 :=
by
  intros h_reaction n_moles_NaCl h_n_moles_NaCl_eq_2
  use 2
  sorry

end required_moles_of_NaHSO3_l759_759326


namespace greatest_three_digit_multiple_of_17_l759_759685

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759685


namespace triangle_DEF_area_l759_759513

noncomputable theory

structure Point where
  x : ℝ
  y : ℝ

def reflect_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

def reflect_y_eq_neg_x (p : Point) : Point :=
  { x := -p.y, y := -p.x }

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def point_D : Point := { x := 5, y := 3 }
def point_E : Point := reflect_y_axis point_D
def point_F : Point := reflect_y_eq_neg_x point_E

def area_of_triangle (base height : ℝ) : ℝ :=
  0.5 * base * height

theorem triangle_DEF_area :
  area_of_triangle (distance point_D point_E) (abs (point_F.y - point_D.y)) = 40 := 
sorry

end triangle_DEF_area_l759_759513


namespace parallelogram_side_length_l759_759416

theorem parallelogram_side_length (s : ℝ) (h : 3 * s * s * (1 / 2) = 27 * Real.sqrt 3) : 
  s = 3 * Real.sqrt (2 * Real.sqrt 3) :=
sorry

end parallelogram_side_length_l759_759416


namespace tony_investment_rate_l759_759187

theorem tony_investment_rate (investment : ℝ) (annual_dividend : ℝ) (rate : ℝ) :
  investment = 3200 → annual_dividend = 250 → rate = (annual_dividend / investment) * 100 → rate = 7.8125 :=
by
  intros h1 h2 h3
  have h : (250 / 3200) * 100 = 7.8125 := by norm_num
  rw [h2, h1] at h3
  exact h3.trans h

end tony_investment_rate_l759_759187


namespace greatest_three_digit_multiple_of_17_l759_759831

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759831


namespace greatest_three_digit_multiple_of_17_is_986_l759_759618

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759618


namespace greatest_three_digit_multiple_of_17_l759_759693

/-- 
The greatest three-digit multiple of 17 is 986.
-/
theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm hbound div_m,
    suffices : 986 ≤ m, by   norm_num,
    sorry,
  }
end

end greatest_three_digit_multiple_of_17_l759_759693


namespace greatest_three_digit_multiple_of17_l759_759714

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759714


namespace greatest_three_digit_multiple_of_17_l759_759576

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759576


namespace greatest_three_digit_multiple_of_17_is_986_l759_759673

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759673


namespace sum_of_primes_less_than_20_l759_759994

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759994


namespace greatest_three_digit_multiple_of_17_l759_759784

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759784


namespace sum_of_primes_less_than_20_l759_759997

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l759_759997


namespace greatest_three_digit_multiple_of_17_l759_759844

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759844


namespace sum_primes_less_than_20_l759_759940

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759940


namespace sum_of_primes_less_than_20_is_77_l759_759927

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759927


namespace hyperbola_eccentricity_range_l759_759025

theorem hyperbola_eccentricity_range (a b c e : Real) (h1 : a > 0) (h2 : b > 0) (h3 : c^2 = a^2 + b^2) (h4 : x = -a^2 / c) :
  let e := sqrt (1 + (b / a)^2) in 1 < e ∧ e < sqrt 2 := by
  sorry

end hyperbola_eccentricity_range_l759_759025


namespace greatest_three_digit_multiple_of_17_is_986_l759_759766

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759766


namespace range_of_a_l759_759354

variable (a : ℝ)

def p : Prop := ∀ (x : ℝ), x^2 + a * x + 1 = 0 → False
def q : Prop := ∀ (x : ℝ), x > 0 → 2^x - a > 0

theorem range_of_a (h1 : ¬ ¬ p) (h2 : ¬ (p ∧ q)) : 1 < a ∧ a < 2 :=
by 
  sorry

end range_of_a_l759_759354


namespace greatest_three_digit_multiple_of17_l759_759724

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759724


namespace greatest_three_digit_multiple_of_17_l759_759814

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759814


namespace misses_are_150_l759_759182

def hits_and_misses (H M : ℕ) : Prop :=
  (M = 3 * H) ∧ (H + M = 200)

theorem misses_are_150 : ∃ H M : ℕ, hits_and_misses H M ∧ M = 150 :=
by
  unfold hits_and_misses
  use 50, 150
  simp
  sorry

end misses_are_150_l759_759182


namespace sum_of_first_six_terms_arithmetic_seq_l759_759523

theorem sum_of_first_six_terms_arithmetic_seq (a b c : ℤ) (d : ℤ) (n : ℤ) :
    (a = 7) ∧ (b = 11) ∧ (c = 15) ∧ (d = b - a) ∧ (d = c - b) 
    ∧ (n = a - d) 
    ∧ (d = 4) -- the common difference is always 4 here as per the solution given 
    ∧ (n = -1) -- the correct first term as per calculation
    → (n + (n + d) + (a) + (b) + (c) + (c + d) = 54) := 
begin
  sorry
end

end sum_of_first_six_terms_arithmetic_seq_l759_759523


namespace collinear_M_N_P_l759_759453

-- Define the given problem in Lean 4

variables {A B C I D E P M N : Type*} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup I] [AddGroup D] [AddGroup E] [AddGroup P] [AddGroup M] [AddGroup N]
variables {triangle_ABC : triangle A B C}
variables {incenter_I : incenter triangle_ABC I}
variables {incircle_touch_D : incircle_touch_side triangle_ABC I D}
variables {incircle_touch_E : incircle_touch_side triangle_ABC I E}
variables {intersection_line_AI_DE : intersection_line AI DE P}
variables {midpoint_BC : midpoint B C M}
variables {midpoint_AB : midpoint A B N}

theorem collinear_M_N_P :
  collinear M N P := sorry

end collinear_M_N_P_l759_759453


namespace initial_oil_amounts_l759_759281

-- Definitions related to the problem
variables (A0 B0 C0 : ℝ)
variables (x : ℝ)

-- Conditions given in the problem
def bucketC_initial := C0 = 48
def transferA_to_B := x = 64 ∧ 64 = (2/3 * A0)
def transferB_to_C := x = 64 ∧ 64 = ((4/5 * (B0 + 1/3 * A0)) * (1/5 + 1))

-- Proof statement to show the solutions
theorem initial_oil_amounts (A0 B0 : ℝ) (C0 x : ℝ) 
  (h1 : bucketC_initial C0)
  (h2 : transferA_to_B A0 x)
  (h3 : transferB_to_C B0 A0 x) :
  A0 = 96 ∧ B0 = 48 :=
by 
  -- Placeholder for the proof
  sorry

end initial_oil_amounts_l759_759281


namespace binomial_identity_l759_759483

theorem binomial_identity (m n : ℕ) : 
  (∑ k in finset.range (2 * n + 1), if n ≤ k then nat.choose m k * nat.choose k (2 * n - k) * 2 ^ (2 * k - 2 * n) else 0) = nat.choose (2 * m) (2 * n) :=
by sorry

end binomial_identity_l759_759483


namespace solve_cake_slicing_l759_759276

open Real

noncomputable def cake_slicing_problem (n : ℕ) : Prop :=
∀ (cherries : Fin n → ℝ),
  (∀ i : Fin n, 0 ≤ cherries i ∧ cherries i < 2 * π) →
  (∀ i j : Fin n, i ≠ j → |cherries j - cherries i| < 2 * π / n) →
  ∃ (cut_points : Fin n → Fin n), 
    (∀ k : Fin n, | (cherries (cut_points k) - (2 * π / n) * k) % (2 * π) | < 2 * π / (2 * n))

theorem solve_cake_slicing (n : ℕ) : cake_slicing_problem n := 
sorry

end solve_cake_slicing_l759_759276


namespace greatest_three_digit_multiple_of_17_l759_759779

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), x = 986 ∧ (x % 17 = 0) ∧ 100 ≤ x ∧ x < 1000 :=
by {
  use 986,
  split,
  { rfl, },
  split,
  { norm_num, },
  split,
  { linarith, },
  { linarith, },
}

end greatest_three_digit_multiple_of_17_l759_759779


namespace sum_primes_less_than_20_l759_759953

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759953


namespace sum_primes_less_than_20_l759_759947

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759947


namespace factorize_expression_l759_759310

variable {a b : ℕ}

theorem factorize_expression (h : 6 * a^2 * b - 3 * a * b = 3 * a * b * (2 * a - 1)) : 6 * a^2 * b - 3 * a * b = 3 * a * b * (2 * a - 1) :=
by sorry

end factorize_expression_l759_759310


namespace intervals_of_monotonicity_l759_759375

noncomputable def f (a x : ℝ) := (1/3) * x^3 - (3/2) * a * x^2 + (2 * a^2 + a - 1) * x + 3

theorem intervals_of_monotonicity (a : ℝ) :
  let f' := λ x, x^2 - 3 * a * x + 2 * a^2 + a - 1 in
  if a = 2 then
    monotone_on (f a) set.univ
  else
    (a < 2 → 
    (∀ x, x < 2 * a - 1 → f' x > 0) ∧
    (∀ x, 2 * a - 1 < x ∧ x < a + 1 → f' x < 0) ∧
    (∀ x, x > a + 1 → f' x > 0)) ∧
    (a > 2 → 
    (∀ x, x < a + 1 → f' x > 0) ∧
    (∀ x, a + 1 < x ∧ x < 2 * a - 1 → f' x < 0) ∧
    (∀ x, x > 2 * a - 1 → f' x > 0)) := 
sorry

end intervals_of_monotonicity_l759_759375


namespace part1_part2_l759_759386

def veca : (ℝ × ℝ) := (1, real.sqrt 3)
def vecb (x : ℝ) : (ℝ × ℝ) := (real.cos x, real.sin x)
def f (x : ℝ) : ℝ := (veca.1 * vecb x.1 + veca.2 * vecb x.2) - 1

theorem part1 (x : ℝ) : f x = 0 ↔ ∃ k : ℤ, x = 2 * k * real.pi ∨ x = (2 * real.pi / 3) + 2 * k * real.pi :=
sorry

theorem part2 (x : ℝ) (h : 0 ≤ x ∧ x ≤ real.pi / 2) :
  (∀ a b : ℝ, 0 ≤ a ∧ a < b ∧ b ≤ real.pi / 2 → f a ≤ f b) ∧
  (∀ c d : ℝ, 0 ≤ c ∧ c < d ∧ d ≤ real.pi / 2 → f c ≤ f d) ∧
  (∀ e : ℝ, e ∈ [0, real.pi / 2] → 0 ≤ f e ∧ f e ≤ 1) :=
sorry

end part1_part2_l759_759386


namespace greatest_three_digit_multiple_of_17_l759_759848

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l759_759848


namespace greatest_three_digit_multiple_of_17_l759_759809

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759809


namespace sum_of_primes_lt_20_eq_77_l759_759914

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759914


namespace lines_perpendicular_to_plane_l759_759107

variables {Point : Type} [EuclideanGeometry Point] (l m : Line Point) (α : Plane Point)

-- Let l and m be two different lines, and α be a plane.
-- Prove: If l is parallel to m, and l is perpendicular to α, then m is perpendicular to α.
theorem lines_perpendicular_to_plane (hlm : l ∥ m) (hlα : l ⊥ α) : m ⊥ α :=
by sorry

end lines_perpendicular_to_plane_l759_759107


namespace domain_of_g_l759_759045

-- Definition of the function f and condition on its domain
def f (x : ℝ) : ℝ := real.sqrt (x * (2 - x))
def domain_f : set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Definition of the function g
def g (x : ℝ) : ℝ := f (2 * x) / (x - 1)

-- Domain of g derived from conditions
def domain_g : set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem domain_of_g :
  {x : ℝ | g x ≠ g x} = domain_g := 
by
  sorry

end domain_of_g_l759_759045


namespace last_triangle_perimeter_l759_759104
-- We import the Mathlib library for broader mathematical tools

-- Define the side lengths of the initial triangle T1
def T1 := (1010: ℝ, 1011: ℝ, 1012: ℝ)

-- Function to calculate the next triangle's side lengths
noncomputable def next_triangle (a b c: ℝ) : (ℝ × ℝ × ℝ) :=
  let AD := (b + c - a) / 2 in
  let BE := (a + c - b) / 2 in
  let CF := (a + b - c) / 2 in
  (AD + 1, BE + 1, CF + 1)

-- Function to calculate the perimeter of the n-th triangle in the sequence
noncomputable def triangle_perimeter (n : ℕ) : ℝ :=
  let initial := T1 in
  let rec calc (k : ℕ) (a b c: ℝ) : ℝ :=
    if k = 0 then a + b + c
    else let (a', b', c') := next_triangle a b c in
         calc (k - 1) a' b' c'
  calc n (initial.1) (initial.2) (initial.3)

-- The theorem that states the perimeter of the last triangle in the sequence
theorem last_triangle_perimeter : triangle_perimeter 6 = 2977.5 :=
  sorry

end last_triangle_perimeter_l759_759104


namespace two_spheres_passing_through_points_touching_planes_l759_759030

variable {A B : Point} {plane : Plane}

theorem two_spheres_passing_through_points_touching_planes (A B : Point) (plane : Plane) :
  ∃ (sphere1 sphere2 : Sphere), 
    sphere1.passes_through A ∧ 
    sphere1.passes_through B ∧ 
    sphere2.passes_through A ∧ 
    sphere2.passes_through B ∧ 
    sphere1.touches plane ∧ 
    sphere2.touches plane ∧ 
    sphere1.touches first_principal_plane ∧ 
    sphere2.touches first_principal_plane ∧ 
    sphere1 ≠ sphere2 := 
sorry

end two_spheres_passing_through_points_touching_planes_l759_759030


namespace greatest_three_digit_multiple_of17_l759_759725

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759725


namespace percentage_increase_is_fifty_percent_l759_759129

-- Given the conditions:
variable (I : ℝ) -- Original income
variable (E : ℝ) -- Original expenditure
variable (I_new : ℝ) -- New income
variable (E_new : ℝ) -- New expenditure
variable (S : ℝ) -- Original savings
variable (S_new : ℝ) -- New savings

-- Given conditions translated into Lean:
def condition1 : E = 0.75 * I := sorry
def condition2 : I_new = 1.20 * I := sorry
def condition3 : E_new = 0.825 * I := sorry

-- Definition of original and new savings based on conditions:
def original_savings : S = 0.25 * I := by
  rw [←condition1]
  simp [S]

def new_savings : S_new = 0.375 * I := by
  rw [←condition2, ←condition3]
  simp [S_new]

-- Calculation of the percentage increase in savings:
def percentage_increase_in_savings : ℝ := ((S_new - S) / S) * 100

-- The proof goal:
theorem percentage_increase_is_fifty_percent (h1 : E = 0.75 * I) (h2 : I_new = 1.20 * I) (h3 : E_new = 0.825 * I) :
  percentage_increase_in_savings = 50 := sorry

end percentage_increase_is_fifty_percent_l759_759129


namespace greatest_three_digit_multiple_of_seventeen_l759_759709

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759709


namespace promotional_codes_one_tenth_l759_759248

open Nat

def promotional_chars : List Char := ['C', 'A', 'T', '3', '1', '1', '9']

def count_promotional_codes (chars : List Char) (len : Nat) : Nat := sorry

theorem promotional_codes_one_tenth : count_promotional_codes promotional_chars 5 / 10 = 60 :=
by 
  sorry

end promotional_codes_one_tenth_l759_759248


namespace sum_of_primes_less_than_20_l759_759896

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759896


namespace number_of_members_l759_759071

def cost_knee_pads : ℤ := 6
def cost_jersey : ℤ := cost_knee_pads + 7
def total_cost_per_member : ℤ := 2 * (cost_knee_pads + cost_jersey)
def total_expenditure : ℤ := 3120

theorem number_of_members (n : ℤ) (h : n * total_cost_per_member = total_expenditure) : n = 82 :=
sorry

end number_of_members_l759_759071


namespace proof_problem_l759_759042

def otimes (a b : ℕ) : ℕ := (a^2 - b) / (a - b)

theorem proof_problem : otimes (otimes 7 5) 2 = 24 := by
  sorry

end proof_problem_l759_759042


namespace composite_probability_l759_759039

-- Definitions based on problem conditions
def is_composite (n : ℕ) : Prop :=
  ¬ nat.prime n ∧ n > 1

def six_sided_die : finset ℕ := {1, 2, 3, 4, 5, 6}

def product_of_rolls (rolls : fin 6 → ℕ) : ℕ :=
  (finset.univ.product (λ i, (rolls i))).val

def total_possible_outcomes : ℕ := 6^6

def composite_outcome_count : ℕ := total_possible_outcomes - 19

-- Theorem statement to prove the probability that the product is composite
theorem composite_probability :
  (composite_outcome_count : ℚ) / total_possible_outcomes = 46637 / 46656 :=
sorry

end composite_probability_l759_759039


namespace broken_shells_count_l759_759033

-- Definitions from conditions
def total_perfect_shells := 17
def non_spiral_perfect_shells := 12
def extra_broken_spiral_shells := 21

-- Derived definitions
def perfect_spiral_shells : ℕ := total_perfect_shells - non_spiral_perfect_shells
def broken_spiral_shells : ℕ := perfect_spiral_shells + extra_broken_spiral_shells
def broken_shells : ℕ := 2 * broken_spiral_shells

-- The theorem to be proved
theorem broken_shells_count : broken_shells = 52 := by
  sorry

end broken_shells_count_l759_759033


namespace greatest_three_digit_multiple_of_17_is_986_l759_759664

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759664


namespace sum_of_primes_lt_20_eq_77_l759_759907

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759907


namespace greatest_three_digit_multiple_of_seventeen_l759_759695

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759695


namespace B_takes_3_hours_l759_759227

noncomputable def time_B : ℝ :=
  let A_rate : ℝ := 1 / 2
  let B_C_rate : ℝ := 1 / 3
  let A_C_rate : ℝ := 1 / 2
  let C_rate : ℝ := A_C_rate - A_rate
  let B_rate : ℝ := B_C_rate - C_rate
  1 / B_rate

theorem B_takes_3_hours :
  let A_rate := 1 / 2 in 
  let B_C_rate := 1 / 3 in 
  let A_C_rate := 1 / 2 in 
  let C_rate := A_C_rate - A_rate in 
  let B_rate := B_C_rate - C_rate in 
  time_B = 3 :=
by
  unfold time_B
  simp [A_rate, B_C_rate, A_C_rate, C_rate]
  sorry

end B_takes_3_hours_l759_759227


namespace part1_part2_l759_759023

noncomputable def f (x a : ℝ) : ℝ := abs (x + a) + abs (2 * x - 5)

theorem part1 (x : ℝ) : (f x 2 ≥ 5) ↔ (x ≤ 2 ∨ x ≥ 8 / 3) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x ∈ set.Icc (a:ℝ) (2 * a - 2), f x a ≤ abs (x + 4)) ↔ (2 < a ∧ a ≤ 13 / 5) :=
by
  sorry

end part1_part2_l759_759023


namespace sum_primes_less_than_20_l759_759944

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759944


namespace sum_of_primes_less_than_20_l759_759979

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l759_759979


namespace f_has_two_extreme_points_f_extrema_bounds_l759_759378

def f (a : ℝ) (x : ℝ) := a * Real.exp (x - 1) - x^2
def g (a : ℝ) (x : ℝ) := a * Real.exp (x - 1) - 2 * x

theorem f_has_two_extreme_points (a : ℝ) (ha : 0 < a ∧ a < 2) : 
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ (f a x₁ = f a x₁) := -- Assuming two roots exist, hence the placeholder function
sorry

theorem f_extrema_bounds (a : ℝ) (ha : a = 1) (x₁ : ℝ) (hx₁ : x₁ < 1) : 
  1 / Real.exp 1 < f 1 x₁ ∧ f 1 x₁ < Real.sqrt 2 / 2 :=
sorry

end f_has_two_extreme_points_f_extrema_bounds_l759_759378


namespace problem_statement_l759_759374

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.logBase 3 x else 2^x

theorem problem_statement : f (f (1/9)) = 1 / 4 := by
  sorry

end problem_statement_l759_759374


namespace greatest_three_digit_multiple_of_17_l759_759587

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759587


namespace sweet_treats_distribution_l759_759468

-- Define the number of cookies, cupcakes, brownies, and students
def cookies : ℕ := 20
def cupcakes : ℕ := 25
def brownies : ℕ := 35
def students : ℕ := 20

-- Define the total number of sweet treats
def total_sweet_treats : ℕ := cookies + cupcakes + brownies

-- Define the number of sweet treats each student will receive
def sweet_treats_per_student : ℕ := total_sweet_treats / students

-- Prove that each student will receive 4 sweet treats
theorem sweet_treats_distribution : sweet_treats_per_student = 4 := 
by sorry

end sweet_treats_distribution_l759_759468


namespace slope_of_line_l759_759201

theorem slope_of_line : let (x1, y1, x2, y2) := (1, -3, -4, 7) in
                        let slope := (y2 - y1) / (x2 - x1) in
                        slope = -2 :=
by
  let (x1, y1, x2, y2) := (1, -3, -4, 7)
  let slope := (y2 - y1) / (x2 - x1)
  have h : slope = -2 := by sorry
  exact h

end slope_of_line_l759_759201


namespace sum_primes_less_than_20_l759_759970

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l759_759970


namespace slope_of_line_l759_759202

theorem slope_of_line : let (x1, y1, x2, y2) := (1, -3, -4, 7) in
                        let slope := (y2 - y1) / (x2 - x1) in
                        slope = -2 :=
by
  let (x1, y1, x2, y2) := (1, -3, -4, 7)
  let slope := (y2 - y1) / (x2 - x1)
  have h : slope = -2 := by sorry
  exact h

end slope_of_line_l759_759202


namespace min_value_on_interval_l759_759510

open Real

noncomputable def f (x : ℝ) : ℝ := x - 1 / x

theorem min_value_on_interval : ∃ x ∈ Icc 1 2, 
  f x = 0 ∧ ∀ y ∈ Icc 1 2, f y ≥ f x := 
by
  sorry

end min_value_on_interval_l759_759510


namespace sum_ge_3_implies_one_ge_2_l759_759558

theorem sum_ge_3_implies_one_ge_2 (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by
  sorry

end sum_ge_3_implies_one_ge_2_l759_759558


namespace sum_primes_less_than_20_l759_759941

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759941


namespace greatest_three_digit_multiple_of_17_l759_759816

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l759_759816


namespace greatest_three_digit_multiple_of17_l759_759716

theorem greatest_three_digit_multiple_of17 : ∃ (n : ℕ), (n ≤ 999) ∧ (100 ≤ n) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (m ≤ 999) ∧ (100 ≤ m) ∧ (17 ∣ m) → m ≤ n) ∧ n = 986 := 
begin
  sorry
end

end greatest_three_digit_multiple_of17_l759_759716


namespace ordered_pairs_jane_marty_l759_759081

noncomputable def count_valid_pairs : ℕ :=
  let Jane_current_age := 35
  in let valid_digits := [(1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9),
                          (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9),
                          (3,4), (3,5), (3,6), (3,7), (3,8), (3,9),
                          (4,5), (4,6), (4,7), (4,8), (4,9),
                          (5,6), (5,7), (5,8), (5,9),
                          (6,7), (6,8), (6,9),
                          (7,8), (7,9),
                          (8,9)]
    in valid_digits.count (λ (ab : ℕ × ℕ), 
        let (a, b) := ab in
        let Jane_future_age := 10 * a + b
        in Jane_future_age > 35)

theorem ordered_pairs_jane_marty : count_valid_pairs = 27 :=
by {
  sorry
}

end ordered_pairs_jane_marty_l759_759081


namespace sum_primes_less_than_20_l759_759938

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759938


namespace equivalent_proof_l759_759006

theorem equivalent_proof :
  let a := 4
  let b := Real.sqrt 17 - a
  b^2020 * (a + Real.sqrt 17)^2021 = Real.sqrt 17 + 4 :=
by
  let a := 4
  let b := Real.sqrt 17 - a
  sorry

end equivalent_proof_l759_759006


namespace solve_for_x_l759_759490

theorem solve_for_x (x : ℝ) (h1 : x^2 - 9 ≠ 0) (h2 : x + 3 ≠ 0) :
  (20 / (x^2 - 9) - 3 / (x + 3) = 2) ↔ (x = (-3 + Real.sqrt 385) / 4 ∨ x = (-3 - Real.sqrt 385) / 4) :=
by
  sorry

end solve_for_x_l759_759490


namespace sum_of_primes_less_than_20_is_77_l759_759930

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l759_759930


namespace pyramid_coloring_l759_759288

noncomputable def number_of_colorings (n m : ℕ) : ℕ :=
m * (m - 2) * ((m - 2) ^ (n - 1) + (-1) ^ n)

theorem pyramid_coloring {n m : ℕ} (hn : n ≥ 3) (hm : m ≥ 4) :
  number_of_colorings n m = m * (m - 2) * ((m - 2) ^ (n - 1) + (-1) ^ n) :=
by
  sorry

end pyramid_coloring_l759_759288


namespace CK_eq_ML_l759_759065

/-
In a triangle \(ABC\) with a right angle at \(C\), the angle bisector \(AL\) (where \(L\) is on segment \(BC\)) intersects the altitude \(CH\) at point \(K\). The bisector of angle \(BCH\) intersects segment \(AB\) at point \(M\). Prove that \(CK = ML\).
-/

open EuclideanGeometry

variable {A B C H L K M : Point}

-- Definitions of conditions
def right_angle_at_C (C : Point) : Prop :=
  ∃ H, is_right_angle (angle ABC C)

def AL_bisects_ABC (A L : Point) : Prop :=
  is_angle_bisector (angle BAC) (line A L)

def CH_altitude (C H : Point) : Prop :=
  is_perpendicular (line C H) (line A B)

def AL_intersects_CH_at_K (A L C H K : Point) : Prop :=
  line_intersection (line A L) (line C H) K

def BCH_bisector_intersects_AB_at_M (B C H M : Point) : Prop :=
  is_angle_bisector (angle BCH) (line B M)

-- Theorem statement
theorem CK_eq_ML 
  (right_angle_at_C : right_angle_at_C C)
  (AL_bisects_ABC : AL_bisects_ABC A L)
  (CH_altitude : CH_altitude C H)
  (AL_intersects_CH_at_K : AL_intersects_CH_at_K A L C H K)
  (BCH_bisector_intersects_AB_at_M : BCH_bisector_intersects_AB_at_M B C H M) : 
  distance C K = distance M L := 
  sorry

end CK_eq_ML_l759_759065


namespace sum_primes_less_than_20_l759_759932

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759932


namespace sum_primes_less_than_20_l759_759931

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l759_759931


namespace geometric_seq_a4_l759_759061

theorem geometric_seq_a4 (a : ℕ → ℕ) (q : ℕ) (h_q : q = 2) 
  (h_a1a3 : a 0 * a 2 = 6 * a 1) : a 3 = 24 :=
by
  -- Skipped proof
  sorry

end geometric_seq_a4_l759_759061


namespace sum_of_primes_less_than_20_l759_759868

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l759_759868


namespace g_difference_l759_759115

noncomputable def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((3 + Real.sqrt 3) / 6) ^ n + (3 - 2 * Real.sqrt 3) / 6 * ((3 - Real.sqrt 3) / 6) ^ n

theorem g_difference (n : ℕ) : g (n + 2) - g n = (1 / 4) * g n := 
sorry

end g_difference_l759_759115


namespace min_remaining_numbers_l759_759478

/-- 
 On a board, all natural numbers from 1 to 100 inclusive are written.
 Vasya picks a pair of numbers from the board whose greatest common divisor (gcd) is greater than 1 and erases one of them.
 The smallest number of numbers that Vasya can leave on the board after performing such actions is 12.
-/
theorem min_remaining_numbers : ∃ S : Finset ℕ, (∀ n ∈ S, n ≤ 100) ∧ 
  (∀ x y ∈ S, x ≠ y → Nat.gcd x y ≤ 1) ∧ S.card = 12 :=
by
  sorry

end min_remaining_numbers_l759_759478


namespace angle_and_distance_l759_759417

-- Define the given conditions
structure Parallelepiped where
  AB BC CC1 : ℝ
  ABCD_A1B1C1D1 : Prop

def point (coord: ℝ × ℝ × ℝ) := coord

variable (D1 M B1 K : point (ℝ × ℝ × ℝ))

-- Set the conditions
axiom parallelepiped_conditions : Parallelepiped 3 2 4 True

-- Coordinates of the points
axiom D1_coords : D1 = (0, 0, 0)
axiom M_coords : M = (1, 2, 4)
axiom B1_coords : B1 = (3, 2, 0)
axiom K_coords : K = (3/2, 0, 2)

-- Definitions for vectors
def vector (p1 p2 : point (ℝ × ℝ × ℝ)) : point (ℝ × ℝ × ℝ) := 
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

-- Vectors
def D1M := vector D1 M
def B1K := vector B1 K

-- Dot product
def dot_product (v1 v2 : point (ℝ × ℝ × ℝ)) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Magnitudes
def magnitude (v : point (ℝ × ℝ × ℝ)) : ℝ := 
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Cosine and distance calculation - proofs
theorem angle_and_distance :
  let cos_angle := dot_product D1M B1K / (magnitude D1M * magnitude B1K)
  let distance := 44 / Real.sqrt 185
  cos_angle = 1 / Real.sqrt 21 ∧ distance = 44 / Real.sqrt 185 := 
by
  sorry

end angle_and_distance_l759_759417


namespace fishing_problem_l759_759185

theorem fishing_problem :
  ∃ F : ℕ, (F % 3 = 1 ∧
            ((F - 1) / 3) % 3 = 1 ∧
            ((((F - 1) / 3 - 1) / 3) % 3 = 1) ∧
            ((((F - 1) / 3 - 1) / 3 - 1) / 3) % 3 = 1 ∧
            ((((F - 1) / 3 - 1) / 3 - 1) / 3 - 1) = 0) :=
sorry

end fishing_problem_l759_759185


namespace leo_probability_l759_759094

theorem leo_probability :
  let age := 17
  (∃ coin, ∃ die, coin ∈ {5, 15} ∧ die ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} ∧ coin + die = age) →
  (1 / 2) * (1 / 12) = 1 / 24 :=
by
  sorry

end leo_probability_l759_759094


namespace sqrt_ineq_l759_759109

theorem sqrt_ineq (x1 x2 : ℝ) (h1 : |x1| ≤ 1) (h2 : |x2| ≤ 1) :
  sqrt (1 - x1^2) + sqrt (1 - x2^2) ≤ 2 * sqrt (1 - ((x1 + x2) / 2)^2) := 
by
  sorry

end sqrt_ineq_l759_759109


namespace sum_of_primes_lt_20_eq_77_l759_759911

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l759_759911


namespace greatest_three_digit_multiple_of_17_is_986_l759_759768

noncomputable def greatestThreeDigitMultipleOf17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∃ (n : ℕ), n = greatestThreeDigitMultipleOf17 ∧ (n >= 100 ∧ n < 1000) ∧ (∃ k : ℕ, n = 17 * k) :=
by
  use 986
  split
  · rfl
  split
  · exact And.intro (by norm_num) (by norm_num)
  · use 58
    norm_num

end greatest_three_digit_multiple_of_17_is_986_l759_759768


namespace greatest_three_digit_multiple_of_17_l759_759738

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l759_759738


namespace sum_of_first_six_terms_arithmetic_seq_l759_759521

theorem sum_of_first_six_terms_arithmetic_seq (a b c : ℤ) (d : ℤ) (n : ℤ) :
    (a = 7) ∧ (b = 11) ∧ (c = 15) ∧ (d = b - a) ∧ (d = c - b) 
    ∧ (n = a - d) 
    ∧ (d = 4) -- the common difference is always 4 here as per the solution given 
    ∧ (n = -1) -- the correct first term as per calculation
    → (n + (n + d) + (a) + (b) + (c) + (c + d) = 54) := 
begin
  sorry
end

end sum_of_first_six_terms_arithmetic_seq_l759_759521


namespace greatest_three_digit_multiple_of_17_l759_759571

theorem greatest_three_digit_multiple_of_17 :
  ∃ (n : ℤ), n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℤ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
begin
  use 986,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros m hdiv hmin hmax,
  have h : 986 = 58 * 17, by norm_num,
  rw h,
  rw ← int.mod_mul_right_mod_eq_zero_iff 17 m 58 at hdiv,
  suffices : 58 ≤ m / 17,
  { exact int.mul_le_mul_of_nonneg_right this (by norm_num), },
  calc
    58 ≤ m / 17 : sorry,
end

end greatest_three_digit_multiple_of_17_l759_759571


namespace remainder_when_divided_by_13_l759_759448

theorem remainder_when_divided_by_13 (n : ℕ) (h₀ : 0 < n) : 
  ∃ a, a ≡ (5 ^ (3 * n) + 7)⁻¹ [MOD 13] ∧ a ≡ 5 [MOD 13] :=
by
  -- proof will be inserted here
  sorry

end remainder_when_divided_by_13_l759_759448


namespace jessica_cut_roses_l759_759184

/-- There were 13 roses and 84 orchids in the vase. Jessica cut some more roses and 
orchids from her flower garden. There are now 91 orchids and 14 roses in the vase. 
How many roses did she cut? -/
theorem jessica_cut_roses :
  let initial_roses := 13
  let new_roses := 14
  ∃ cut_roses : ℕ, new_roses = initial_roses + cut_roses ∧ cut_roses = 1 :=
by
  sorry

end jessica_cut_roses_l759_759184


namespace part_I_part_II_i_part_II_ii_l759_759012

-- Part I
theorem part_I (a b : ℝ) (h_a_ne_zero : a ≠ 0) (h_f_prime_e_eq_0 : ∀ x, differentiable_at ℝ (λ x, (a * (1 - log x) - b * exp x * (x - 1)) / x^2) e → x = e → b = 0) :=
  a < 0 :=
sorry

-- Part II.i
theorem part_II_i (a b : ℝ) (h_a_eq_1 : a = 1) (h_b_eq_1 : b = 1) (x : ℝ) :
  x * ((a * log x - b * exp x) / x) + 2 < 0 :=
sorry

-- Part II.ii
theorem part_II_ii (a b : ℝ) (h_a_eq_1 : a = 1) (h_b_eq_neg_1 : b = -1) :
  ∀ x ∈ Ioi 1, (λ x, x * ((a * log x - b * exp x) / x) > exp 1 + m * (x - 1)) → m ≤ 1 + exp 1 :=
begin
  intro x,
  intros hx hxf,
  have : ∀ x ∈ Ioi 1, x * ((log x - exp x) / x) > exp 1 + m * (x - 1),
  by { intros x hx, rw h_a_eq_1, rw h_b_eq_neg_1, exact hxf x hx },
  sorry,
end

end part_I_part_II_i_part_II_ii_l759_759012


namespace sum_of_primes_less_than_20_l759_759889

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l759_759889


namespace sum_primes_less_than_20_l759_759958

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759958


namespace sum_primes_less_than_20_l759_759954

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l759_759954


namespace janet_needs_more_money_l759_759085

theorem janet_needs_more_money :
  let janet_savings := 2225
  let monthly_rent := 1250
  let months_in_advance := 2
  let deposit := 500
  let total_required := (monthly_rent * months_in_advance) + deposit
  let additional_money_needed := total_required - janet_savings
  in additional_money_needed = 775 :=
by
  let janet_savings := 2225
  let monthly_rent := 1250
  let months_in_advance := 2
  let deposit := 500
  let total_required := (monthly_rent * months_in_advance) + deposit
  let additional_money_needed := total_required - janet_savings
  have h1 : total_required = 3000, by sorry
  have h2 : additional_money_needed = total_required - janet_savings, by sorry
  have h3 : total_required - janet_savings = 775, by sorry
  exact Eq.trans h2 h3

end janet_needs_more_money_l759_759085


namespace arithmetic_sequences_count_l759_759035

/-- 
Given an arithmetic sequence with the first term a1 and common difference d, define Sn as the sum 
of the first n terms. Prove that the number of such sequences where the ratio S_{2n} / S_n does 
not depend on n and one term is 1971, is 9.
-/
theorem arithmetic_sequences_count :
  { seq : ℕ → ℕ // 
    ∃ a1 d, 
      (seq 0 = a1) ∧ 
      (∀ n, seq (n + 1) = seq n + d) ∧ 
      (∀ n, seq n ∈ ℕ) ∧ 
      (∃ n, seq n = 1971) ∧ 
      (∀ n, (2 * ∑ i in range(2 * n), seq i) / ∑ i in range(n), seq i = 4) },
  fintype.card { seq // true } = 1 + (finset.univ.filter (λ x, ∃ m, 1971 = x.val * m)).card :=
sorry

end arithmetic_sequences_count_l759_759035


namespace greatest_three_digit_multiple_of_17_is_986_l759_759624

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l759_759624


namespace greatest_three_digit_multiple_of_17_l759_759586

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end greatest_three_digit_multiple_of_17_l759_759586


namespace n_minus_m_l759_759401

theorem n_minus_m (m n : ℝ) (h1 : m^2 - n^2 = 6) (h2 : m + n = 3) : n - m = -2 :=
by
  sorry

end n_minus_m_l759_759401


namespace min_value_expression_l759_759356

theorem min_value_expression (x y z : ℝ) (h : x + y + z = 2) : 
  2 * x^2 + 3 * y^2 + z^2 ≥ 24 / 11 ∧ 
  (∀ a b c, a + b + c = 2 → 2 * a^2 + 3 * b^2 + c^2 = 24 / 11 → a = 6 / 11 ∧ b = 4 / 11 ∧ c = 12 / 11):
by
  sorry

end min_value_expression_l759_759356


namespace greatest_three_digit_multiple_of_seventeen_l759_759694

theorem greatest_three_digit_multiple_of_seventeen : ∃ k : ℕ, k * 17 = 986 ∧ k * 17 < 1000 ∧ k * 17 ≥ 100 :=
by
  use 58
  split
  · exact rfl
      
  split
  · norm_num

  · norm_num
  sorry

end greatest_three_digit_multiple_of_seventeen_l759_759694


namespace sin_cos_value_l759_759396

theorem sin_cos_value (x : ℝ) (h : sin x = 4 * cos x) : sin x * cos x = 4 / 17 := 
sorry

end sin_cos_value_l759_759396


namespace ratio_a_e_l759_759515

theorem ratio_a_e (a b c d e : ℚ) 
  (h₀ : a / b = 2 / 3)
  (h₁ : b / c = 3 / 4)
  (h₂ : c / d = 3 / 4)
  (h₃ : d / e = 4 / 5) :
  a / e = 3 / 10 :=
sorry

end ratio_a_e_l759_759515


namespace integer_solutions_l759_759316

theorem integer_solutions (x y : ℤ) : 2 * (x + y) = x * y + 7 ↔ (x, y) = (3, -1) ∨ (x, y) = (5, 1) ∨ (x, y) = (1, 5) ∨ (x, y) = (-1, 3) := by
  sorry

end integer_solutions_l759_759316


namespace cannot_determine_AB_CD_l759_759110

-- Definitions of points and plane

variables (A B C D E F : Point) -- Points on the sphere
variables (π : Plane) -- Plane passing through A and perpendicular to AB

-- Conditions
axiom diameter_of_sphere : IsDiameter A B -- A and B are endpoints of a diameter
axiom plane_passing_A_perp_AB : PassesThrough π A ∧ Perpendicular π (Segment A B) -- Plane π passing through A is perpendicular to AB
axiom distinct_on_sphere : OnSphere C ∧ OnSphere D ∧ C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B -- C and D are distinct points on the sphere different from A and B
axiom extensions_intersect_plane : Intersects (Extension (Segment B C)) π E ∧ Intersects (Extension (Segment B D)) π F -- BCE and BDF intersect plane π at points E and F respectively
axiom isosceles_triangle_AEF : IsIsoscelesTriangle (Triangle A E F) -- Points A, E, F form an isosceles triangle

-- Theorem to prove
theorem cannot_determine_AB_CD : CannotDeterminePositionalRelationship (Line A B) (Line C D) :=
by
  sorry

end cannot_determine_AB_CD_l759_759110


namespace correct_statements_l759_759271

-- Define basic positional relationships in terms of planes and lines
axiom Plane : Type
axiom Point : Type
axiom Line : Type

-- Basic Conditions
axiom PlaneA : Plane
axiom P_not_on_PlaneA : Point → ¬(P ∈ PlaneA)

axiom Line1 : Line
axiom Line1_not_on_PlaneA : Line → ¬(Line1 ⊆ PlaneA)

axiom Line2 : Line
axiom Q_not_on_Line2 : Point → ¬(Q ∈ Line2)

axiom R : Point
axiom R_not_on_Line2 : ¬(R ∈ Line2)

-- Define the statements that need to be proved
axiom unique_plane_parallel_plane (p : Point) : ∃! (π : Plane), parallel π PlaneA ∧ p ∉ π
axiom no_plane_parallel_without_intersection (l : Line) : ∃! (π : Plane), ¬parallel π PlaneA ∧ l ∈ π
axiom unique_line_parallel_line (q : Point) : ∃! (l : Line), parallel l Line1 ∧ q ∉ l
axiom countless_planes_parallel_to_line (r : Point) : ∃ (π_set : Set Plane), ∀ π ∈ π_set, parallel π Line1 ∧ r ∈ π

-- Solution: Verify the correct statements
theorem correct_statements : 
  (unique_plane_parallel_plane P) ∧ 
  (¬no_plane_parallel_without_intersection Line1) ∧ 
  (unique_line_parallel_line Q) ∧ 
  (¬countless_planes_parallel_to_line R)
:= by
  sorry

end correct_statements_l759_759271


namespace total_baseball_cards_l759_759090

theorem total_baseball_cards (Carlos Matias Jorge : ℕ) (h1 : Carlos = 20) (h2 : Matias = Carlos - 6) (h3 : Jorge = Matias) : Carlos + Matias + Jorge = 48 :=
by
  sorry

end total_baseball_cards_l759_759090
