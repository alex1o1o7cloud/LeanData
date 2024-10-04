import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Order.Nonneg.Real
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Basic
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.GCD
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Triangle
import Mathlib.Logic.Basic
import Mathlib.Mathlib
import Mathlib.Probability.Basic
import Mathlib.RingTheory.Determinant
import Mathlib.Tactic
import Mathlib.Topology.Basic
import Mathlib.Topology.Instances.Real
import Mathlib.Trigonometry.Basic

namespace sum_of_tangency_points_l723_723481

def q (x : ℝ) : ℝ := sorry -- Define the general quadratic polynomial
def f (x : ℝ) : ℝ := max (-6 * x - 23) (max (4 * x + 1) (7 * x + 4))

noncomputable def is_tangent_at (q f : ℝ → ℝ) (x₁ x₂ x₃ : ℝ) : Prop :=
  (q x₁ = f x₁ ∧ q' x₁ = f' x₁) ∧ (q x₂ = f x₂ ∧ q' x₂ = f' x₂) ∧ (q x₃ = f x₃ ∧ q' x₃ = f' x₃)
  where
    q' := deriv q
    f' := deriv f

theorem sum_of_tangency_points (x₁ x₂ x₃ : ℝ) (hq : is_tangent_at q f x₁ x₂ x₃) : 
  x₁ + x₂ + x₃ = -5.2 :=
by
  sorry -- Proof to be provided

end sum_of_tangency_points_l723_723481


namespace cross_product_correct_l723_723511

-- Define the vectors
def vec1 : ℝ × ℝ × ℝ := (3, 1, 4)
def vec2 : ℝ × ℝ × ℝ := (4, -2, 6)

-- Define the scaled vector
def scaled_vec2 : ℝ × ℝ × ℝ := (2 * 4, 2 * -2, 2 * 6)

-- Define the expected cross product
def expected_cross_product : ℝ × ℝ × ℝ := (28, -4, -20)

-- Define the cross product function
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.2 * v.3 - u.3 * v.2.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2.2 - u.2.2 * v.1)

-- The theorem to prove
theorem cross_product_correct : cross_product vec1 scaled_vec2 = expected_cross_product :=
by {
  -- This is where the proof would go
  sorry
}

end cross_product_correct_l723_723511


namespace monotonicity_a_0_min_value_a_le_1_l723_723958

noncomputable def f (x a : ℝ) : ℝ := x^3 + 3 * (abs (x - a)) + 2

theorem monotonicity_a_0 :
  (∀ x > 0, ∃ δ > 0, ∀ h : x ∈ Ioo (0 + δ) (x + δ), f(x, 0) < f(x+h, 0)) ∧
  (∀ x ≤ 0, ∃ δ > 0, (x ∈ Ioo (x - δ) (0 - δ) → f(x + h, 0) < f(x, 0)) ∧
                    (x ∈ Ioo (x - δ) (-1 - δ) → f(x + h, 0) > f(x, 0))) :=
by sorry

theorem min_value_a_le_1 (a : ℝ) (h_a : a ≤ 1) :
  (a ≤ 0 → ∀ x ∈ Icc 0 2, f(x, a) ≥ f(0, a) ∧ f(0, a) = 2 - 3 * a) ∧
  (0 < a → a ≤ 1 → ∀ x ∈ Icc 0 2, f(x, a) ≥ f(a, a) ∧ f(a, a) = 2 + a^3) :=
by sorry

end monotonicity_a_0_min_value_a_le_1_l723_723958


namespace common_element_in_subsets_l723_723284

open Finset

theorem common_element_in_subsets (A : Finset α) (n : ℕ) (h : A.card = n) (S : Finset (Finset α))
  (hS : S.card = 2^(n-1))
  (h_common : ∀ x y z ∈ S, ∃ a, a ∈ x ∧ a ∈ y ∧ a ∈ z) :
  ∃ e, ∀ s ∈ S, e ∈ s :=
sorry

end common_element_in_subsets_l723_723284


namespace sum_natural_numbers_6_to_21_l723_723921

def sum_arithmetic_series (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_natural_numbers_6_to_21 :
  sum_arithmetic_series 6 1 16 = 216 :=
by
  -- Definitions matching the conditions
  let first_term := 6
  let last_term := 21
  let common_difference := 1
  let num_terms := (last_term - first_term) / common_difference + 1
  have n_terms_eq : num_terms = 16 := by simp
  rw n_terms_eq
  -- The conclusion to be proven
  let sum_value := sum_arithmetic_series first_term common_difference num_terms
  have sum_eq : sum_value = 216 := by simp
  exact sum_eq

end sum_natural_numbers_6_to_21_l723_723921


namespace triangle_problem_l723_723274

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (ha : a = 1) (hA : A = Real.pi / 4) :
  (sqrt 2 * b) / (Real.sin C + Real.cos C) = sqrt 2 :=
by
  sorry

end triangle_problem_l723_723274


namespace num_prime_factors_of_30_l723_723227

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723227


namespace birds_after_changes_are_235_l723_723451

-- Define initial conditions for the problem
def initial_cages : Nat := 15
def parrots_per_cage : Nat := 3
def parakeets_per_cage : Nat := 8
def canaries_per_cage : Nat := 5
def parrots_sold : Nat := 5
def canaries_sold : Nat := 2
def parakeets_added : Nat := 2


-- Define the function to count total birds after the changes
def total_birds_after_changes (initial_cages parrots_per_cage parakeets_per_cage canaries_per_cage parrots_sold canaries_sold parakeets_added : Nat) : Nat :=
  let initial_parrots := initial_cages * parrots_per_cage
  let initial_parakeets := initial_cages * parakeets_per_cage
  let initial_canaries := initial_cages * canaries_per_cage
  
  let final_parrots := initial_parrots - parrots_sold
  let final_parakeets := initial_parakeets + parakeets_added
  let final_canaries := initial_canaries - canaries_sold
  
  final_parrots + final_parakeets + final_canaries

-- Prove that the total number of birds is 235
theorem birds_after_changes_are_235 : total_birds_after_changes 15 3 8 5 5 2 2 = 235 :=
  by 
    -- Proof is omitted as per the instructions
    sorry

end birds_after_changes_are_235_l723_723451


namespace different_prime_factors_of_factorial_eq_10_l723_723047

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723047


namespace prime_factors_of_30_factorial_l723_723097

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723097


namespace first_part_length_l723_723357

def total_length : ℝ := 74.5
def part_two : ℝ := 21.5
def part_three : ℝ := 21.5
def part_four : ℝ := 16

theorem first_part_length :
  total_length - (part_two + part_three + part_four) = 15.5 :=
by
  sorry

end first_part_length_l723_723357


namespace entire_meal_cost_correct_l723_723879

-- Define given conditions
def appetizer_cost : ℝ := 9.00
def entree_cost : ℝ := 20.00
def num_entrees : ℕ := 2
def dessert_cost : ℝ := 11.00
def tip_percentage : ℝ := 0.30

-- Calculate intermediate values
def total_cost_before_tip : ℝ := appetizer_cost + (entree_cost * num_entrees) + dessert_cost
def tip : ℝ := tip_percentage * total_cost_before_tip
def entire_meal_cost : ℝ := total_cost_before_tip + tip

-- Statement to be proved
theorem entire_meal_cost_correct : entire_meal_cost = 78.00 := by
  -- Proof will go here
  sorry

end entire_meal_cost_correct_l723_723879


namespace min_abs_ab_l723_723669

theorem min_abs_ab (a b : ℤ) (h : 1009 * a + 2 * b = 1) : ∃ k : ℤ, |a * b| = 504 :=
by
  sorry

end min_abs_ab_l723_723669


namespace solve_g_triple_application_l723_723325

def g : ℕ → ℕ :=
  λ n, if n < 3 then n^2 + 3 * n else if n < 6 then 2 * n + 1 else 5 * n - 4

theorem solve_g_triple_application : g (g (g 2)) = 226 := by
  sorry

end solve_g_triple_application_l723_723325


namespace prime_factors_of_30_l723_723204

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723204


namespace number_of_prime_factors_thirty_factorial_l723_723005

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723005


namespace smallest_integer_satisfying_inequality_l723_723749

theorem smallest_integer_satisfying_inequality : ∃ n: ℕ, (n ≥ 1) ∧ ∀ m: ℕ, (m ≥ 1) → (n ≤ m ↔ (√m - √(m-1) < 0.01)) := sorry

end smallest_integer_satisfying_inequality_l723_723749


namespace find_angle_C_find_area_triangle_l723_723617

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  (2 * sqrt 3 * a * sin C * sin B = a * sin A + b * sin B - c * sin C) ∧
  (a * cos(π/2 - B) = b * cos(2 * int.pi * k + A)) ∧ (a = 2)

theorem find_angle_C (a b c A B C: ℝ) (k : ℤ) (h1: triangle_problem a b c A B C):
  C = π / 6 :=
sorry

theorem find_area_triangle (a b c A B C: ℝ) (k : ℤ) (h1: triangle_problem a b c A B C):
  let A := π / 4 in
  let B := 5 * π / 12 in
  let c := sqrt 2 in
  1/2 * a * c * sin (π / 6 + π / 4) = (1 + sqrt 3) / 2 :=
sorry

end find_angle_C_find_area_triangle_l723_723617


namespace main_statement_l723_723922

-- Definitions for min and max functions
def M (x y : ℝ) := max x y
def m (x y : ℝ) := min x y

-- Main theorem statement
theorem main_statement (p q r s t : ℝ) (h : p < q) (h' : q < r) (h'' : r < s) (h''' : s < t) :
  M (M p (m q s)) (m r (m p t)) = q := by
  sorry

end main_statement_l723_723922


namespace house_numbers_count_l723_723500

noncomputable def two_digit_primes_less_than_60 : List Nat :=
  [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

def is_valid_house_number (WX YZ : Nat) : Bool :=
  WX < 100 ∧ WX > 9 ∧ YZ < 100 ∧ YZ > 9 ∧ WX ≠ YZ ∧ WX ∈ two_digit_primes_less_than_60 ∧ YZ ∈ two_digit_primes_less_than_60 ∧ 
  (WX / 10) ≠ 0 ∧ (WX % 10) ≠ 0 ∧ (YZ / 10) ≠ 0 ∧ (YZ % 10) ≠ 0

def count_valid_house_numbers : Nat :=
  List.length $ (List.product two_digit_primes_less_than_60 two_digit_primes_less_than_60).filter (λ p, is_valid_house_number p.fst p.snd)

theorem house_numbers_count : count_valid_house_numbers = 156 := by
  -- Proof details go here 
  sorry

end house_numbers_count_l723_723500


namespace employees_count_l723_723334

variable (revenue taxes marketing operational wagePerEmployee totalEmployees : ℝ)

def revenue := 400000
def taxes := 0.10 * revenue
def afterTaxes := revenue - taxes
def marketing := 0.05 * afterTaxes
def afterMarketing := afterTaxes - marketing
def operational := 0.20 * afterMarketing
def afterOperational := afterMarketing - operational
def totalWages := 0.15 * afterOperational
def wagePerEmployee := 4104

theorem employees_count : 
  totalEmployees = totalWages / wagePerEmployee := by
  sorry

end employees_count_l723_723334


namespace simplify_expr_l723_723694

variable (x : ℝ)

theorem simplify_expr : (2 * x^2 + 5 * x - 7) - (x^2 + 9 * x - 3) = x^2 - 4 * x - 4 :=
by
  sorry

end simplify_expr_l723_723694


namespace number_of_prime_factors_of_30_factorial_l723_723115

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723115


namespace num_prime_factors_30_factorial_l723_723246

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723246


namespace ratio_after_addition_l723_723406

theorem ratio_after_addition (a b : ℕ) (h1 : a * 3 = b * 2) (h2 : b - a = 8) : (a + 4) * 7 = (b + 4) * 5 :=
by
  sorry

end ratio_after_addition_l723_723406


namespace find_n_l723_723988

theorem find_n
  (n : ℕ)
  (h : (∑ k in finset.range (n), (1 : ℚ) / (k + 1) * (1 / (k + 2))) = 9 / 10) :
  n = 10 := 
sorry

end find_n_l723_723988


namespace smallest_rel_prime_1155_l723_723781

open Nat

theorem smallest_rel_prime_1155 : ∃ n : ℕ, n > 1 ∧ gcd n 1155 = 1 ∧ ∀ m : ℕ, m > 1 ∧ gcd m 1155 = 1 → n ≤ m := 
sorry

end smallest_rel_prime_1155_l723_723781


namespace exists_primitive_root_x_and_4x_l723_723812

theorem exists_primitive_root_x_and_4x (p : ℕ) [Fact (Nat.Prime p)] (h : p % 2 = 1) :
  ∃ x : ℕ, Nat.PrimitiveRoot x p ∧ Nat.PrimitiveRoot (4 * x) p :=
sorry

end exists_primitive_root_x_and_4x_l723_723812


namespace probability_is_1_6_l723_723271

-- Define the set of numbers and the condition of choosing two numbers without replacement
def num_set : set ℕ := {4, 6, 7, 9}

-- Define a function to calculate the probability 
def probability_multiple_12 (s : set ℕ) : ℚ :=
  let pairs := {xy | ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ xy = (x, y)} in
  let total_pairs := pairs.card in
  let mult_12_pairs := {xy ∈ pairs | (xy.1 * xy.2) % 12 = 0}.card in
  mult_12_pairs / total_pairs

-- Define the theorem to prove that the probability is 1/6
theorem probability_is_1_6 : probability_multiple_12 num_set = 1/6 :=
  sorry

end probability_is_1_6_l723_723271


namespace sum_1999_seq_l723_723963

noncomputable def seq (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else seq (n - 2) + seq (n - 1) - 1

theorem sum_1999_seq : 
  (∀ n : ℕ, seq n * seq (n + 1) * seq (n + 2) = seq n + seq (n + 1) + seq (n + 2)) →
  (∀ n : ℕ, seq (n + 1) * seq (n + 2) ≠ 1) →
  (∑ n in Finset.range 1999, seq (n + 1)) = 3997 :=
by
  sorry

end sum_1999_seq_l723_723963


namespace max_groups_eq_one_l723_723686

-- Defining the conditions 
def eggs : ℕ := 16
def marbles : ℕ := 3
def rubber_bands : ℕ := 5

-- The theorem statement
theorem max_groups_eq_one
  (h1 : eggs = 16)
  (h2 : marbles = 3)
  (h3 : rubber_bands = 5) :
  ∀ g : ℕ, (g ≤ eggs ∧ g ≤ marbles ∧ g ≤ rubber_bands) →
  (eggs % g = 0) ∧ (marbles % g = 0) ∧ (rubber_bands % g = 0) →
  g = 1 :=
by
  sorry

end max_groups_eq_one_l723_723686


namespace number_of_prime_factors_thirty_factorial_l723_723006

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723006


namespace angle_CED_correct_l723_723478

-- Define the coordinates of points C and D on the sphere
def coord_C : (ℝ × ℝ) := (0, 9)
def coord_D : (ℝ × ℝ) := (30, 91)

-- Coordinates of north pole, assuming Earth as a sphere for simplicity
def north_pole_at_30_lat (latitude: ℝ × ℝ) : (ℝ × ℝ) := (latitude.1 - 30, latitude.2)

-- Definition for finding longitude difference between two points
def longitude_difference (c1 c2: ℝ × ℝ): ℝ := abs (c2.2 - c1.2)

-- Proof that the calculated angle \(\angle CED = 278^\circ\) on a perfect sphere Earth
noncomputable def angle_CED : Prop :=
  let D' := north_pole_at_30_lat coord_D in
  let longitude_diff := longitude_difference coord_C D' in
  longitude_diff = 82 ∧ 360 - longitude_diff = 278

-- The proof statement for angle \(\angle CED = 278^\circ\)
theorem angle_CED_correct : angle_CED :=
sorry

end angle_CED_correct_l723_723478


namespace triangle_BXN_property_l723_723647

-- Definitions of points, lines, and triangle properties
section Geometry

variables {Point : Type} [AffineSpace Point]

-- Definitions for points A, B, C, M, N, and X
variable (A B C M N X : Point)

-- Definitions of line segments AC, BN, BX, and XN
variable (AC BN BX XN : Set Point)

-- Define the necessary geometric relationships and properties
def midpoint (M : Point) (A C : Point) : Prop :=
  dist A M = dist M C

def angle_bisector (BN : Set Point) (B : Point) (N : Point) : Prop :=
  BN ⊆ (line B N)

def isosceles_right_triangle (BX XN : Set Point) (B : Point) (X : Point) (N : Point) : Prop :=
  BX ⊆ (line B X) ∧ XN ⊆ (line X N) ∧ dist B X = dist X N ∧ ∠ B X N = 45

-- Main theorem statement
theorem triangle_BXN_property
  (h_midpoint : midpoint M A C)
  (h_angle_bisector : angle_bisector BN B N)
  (h_isosceles_right : isosceles_right_triangle BX XN B X N)
  (h_AC : dist A C = 4)
  : dist B X ^ 2 = (sqrt 129 - 1) / 16 :=
by
  sorry

end Geometry

end triangle_BXN_property_l723_723647


namespace initial_population_given_final_population_l723_723394

variable (P : ℝ) (P2 : ℝ)
variable (decrease_rate : ℝ := 0.2)

-- Define the population after 2 years
def population_after_2_years (initial_pop : ℝ) (rate : ℝ) : ℝ :=
  (1 - rate) * (1 - rate) * initial_pop

theorem initial_population_given_final_population:
  (population_after_2_years P decrease_rate = 12800) →
  P = 20000 :=
by
  intro h
  rw [population_after_2_years] at h
  field_simp at h
  sorry

end initial_population_given_final_population_l723_723394


namespace fraction_sum_l723_723883

theorem fraction_sum : (3 / 8) + (9 / 12) + (5 / 6) = 47 / 24 := by
  sorry

end fraction_sum_l723_723883


namespace number_of_distinct_prime_factors_30_fact_l723_723003

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l723_723003


namespace tissue_actual_diameter_l723_723415

theorem tissue_actual_diameter (magnification_factor : ℕ) (magnified_diameter : ℝ) :
  magnification_factor = 1000 ∧ magnified_diameter = 0.2 → (magnified_diameter / magnification_factor) = 0.0002 :=
by
  intro h
  cases h with magnification_eq diameter_eq
  rw [magnification_eq, diameter_eq]
  norm_num
  sorry

end tissue_actual_diameter_l723_723415


namespace num_prime_factors_of_30_l723_723237

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723237


namespace work_completion_rate_l723_723426

theorem work_completion_rate (A B D : ℝ) (W : ℝ) (hB : B = W / 9) (hA : A = W / 10) (hD : D = 90 / 19) : 
  (A + B) * D = W := 
by 
  sorry

end work_completion_rate_l723_723426


namespace equilibrium_possible_l723_723861

variables {a b θ : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : (b / 2) < a) (h4 : a ≤ b)

theorem equilibrium_possible :
  θ = 0 ∨ θ = Real.arccos ((b^2 + 2 * a^2) / (3 * a * b)) → 
  (b / 2) < a ∧ a ≤ b ∧ (0 ≤ θ ∧ θ ≤ π) :=
sorry

end equilibrium_possible_l723_723861


namespace cubic_polynomial_sum_equals_391_l723_723937

theorem cubic_polynomial_sum_equals_391
  (q : ℝ → ℝ)
  (hq : ∀ x, ∃ a b c d, q x = a*x^3 + b*x^2 + c*x + d)
  (hq3 : q 3 = 2)
  (hq8 : q 8 = 20)
  (hq18 : q 18 = 12)
  (hq25 : q 25 = 32) :
  (∑ x in finset.range (26 - 4 + 1), q (4 + x)) = 391 :=
sorry

end cubic_polynomial_sum_equals_391_l723_723937


namespace quadratic_condition_l723_723601

theorem quadratic_condition (m : ℝ) (h1 : m^2 - 2 = 2) (h2 : m + 2 ≠ 0) : m = 2 :=
by
  sorry

end quadratic_condition_l723_723601


namespace probability_of_dolphin_snout_within_2_m_from_edge_l723_723833

noncomputable def pool_length : ℝ := 30
noncomputable def pool_width : ℝ := 20
noncomputable def edge_distance : ℝ := 2

def pool_area : ℝ := pool_length * pool_width
def central_rect_length : ℝ := pool_length - 2 * edge_distance
def central_rect_width : ℝ := pool_width - 2 * edge_distance
def central_rect_area : ℝ := central_rect_length * central_rect_width
def edge_area : ℝ := pool_area - central_rect_area
def probability_edge : ℝ := edge_area / pool_area

theorem probability_of_dolphin_snout_within_2_m_from_edge :
  probability_edge ≈ 0.31 :=
sorry

end probability_of_dolphin_snout_within_2_m_from_edge_l723_723833


namespace num_prime_factors_30_factorial_l723_723250

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723250


namespace right_triangle_circle_area_l723_723385

/-- 
Given a right triangle ABC with legs AB = 6 cm and BC = 8 cm,
E is the midpoint of AB and D is the midpoint of AC.
A circle passes through points E and D and touches the hypotenuse AC.
Prove that the area of this circle is 100 * pi / 9 cm^2.
-/
theorem right_triangle_circle_area :
  ∃ (r : ℝ), 
  let AB := 6
  let BC := 8
  let AC := Real.sqrt (AB^2 + BC^2)
  let E := (AB / 2)
  let D := (AC / 2)
  let radius := (AC * (BC / 2) / AB)
  r = radius * radius * Real.pi ∧
  r = (100 * Real.pi / 9) := sorry

end right_triangle_circle_area_l723_723385


namespace complex_subtraction_calculation_l723_723602

variables (A M S : ℂ) (P Q : ℂ)

def A := 5 - 4 * complex.I
def M := -5 + 2 * complex.I
def S := 2 * complex.I
def P : ℂ := 3
def Q := 1 + complex.I

theorem complex_subtraction_calculation :
  A - M + S - P - Q = 6 - 5 * complex.I :=
by
  sorry

end complex_subtraction_calculation_l723_723602


namespace num_prime_factors_30_factorial_l723_723038

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723038


namespace train_speed_l723_723857

theorem train_speed (distance time : ℤ) (h_distance : distance = 500)
    (h_time : time = 3) :
    distance / time = 166 :=
by
  -- Proof steps will be filled in here
  sorry

end train_speed_l723_723857


namespace prove_polynomial_root_pow_eq_one_l723_723326

noncomputable def polynomial_root_pow_eq_one (n : ℕ) (a : fin n → ℝ) (λ : ℂ) : Prop :=
  (λ ^ (n + 1) = 1)

theorem prove_polynomial_root_pow_eq_one
  (n : ℕ)
  (a : fin n → ℝ)
  (λ : ℂ)
  (h0 : 0 < a 0)
  (h1 : ∀ (i : fin (n - 1)), a i ≤ a (i + 1))
  (h1_last : a (n - 1) ≤ 1)
  (h2 : is_root (λ x, x ^ n + ∑ i in range n, a i * x ^ i) λ)
  (h_abs : abs λ > 1) :
  polynomial_root_pow_eq_one n a λ := by
  sorry

end prove_polynomial_root_pow_eq_one_l723_723326


namespace exists_integers_u_v_l723_723362

theorem exists_integers_u_v (A : ℕ) (a b s : ℤ)
  (hA: A = 1 ∨ A = 2 ∨ A = 3)
  (hab_rel_prime: Int.gcd a b = 1)
  (h_eq: a^2 + A * b^2 = s^3) :
  ∃ u v : ℤ, s = u^2 + A * v^2 ∧ a = u^3 - 3 * A * u * v^2 ∧ b = 3 * u^2 * v - A * v^3 := 
sorry

end exists_integers_u_v_l723_723362


namespace train_length_l723_723464

-- Definitions based on conditions
def train_speed_kmh : ℝ := 60.994720422366214
def man_speed_kmh : ℝ := 5
def crossing_time_sec : ℕ := 6

-- Conversion factor from km/h to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

-- Calculate the relative speed in m/s
def relative_speed_ms : ℝ := kmh_to_ms (train_speed_kmh + man_speed_kmh)

-- Proof statement
theorem train_length : relative_speed_ms * crossing_time_sec = 109.99 :=
by
  sorry

end train_length_l723_723464


namespace transform_graph_sin_to_cos_l723_723765

theorem transform_graph_sin_to_cos (x : ℝ) :
  (∀ x, (∃ c1 c2, ∀ x, cos (2 * x) - 1 = sin (2 * (x + c1)) + c2) → c1 = -(π / 4) ∧ c2 = -1) :=
by
  sorry

end transform_graph_sin_to_cos_l723_723765


namespace probability_chords_intersect_l723_723762

theorem probability_chords_intersect (n : ℕ) (hn : n = 1011) :
  let points := finset.range (2 * n) \ finset.filter (λ x, x % 2 = 0) (finset.range (2 * n))
  let valid_orderings := {({A₁, B₁, C₁, D₁} : finset (fin 2022)) | A₁ < C₁ < B₁ < D₁ ∨ A₁ < D₁ < B₁ < C₁}
  let total_orderings := finset.power_set (finset.range (2 * n))
in (valid_orderings.card : ℚ) / (total_orderings.card : ℚ) = 1 / 12 := by
  sorry

end probability_chords_intersect_l723_723762


namespace find_angleQRS_l723_723532

noncomputable def angleQRS_proof
    (PQ RS PS : ℝ)
    (angleQPS angleRSP : ℝ) : Prop :=
  PQ = 40 ∧ RS = 20 ∧ PS = 60 ∧ angleQPS = 60 ∧ angleRSP = 60 → angleQRS = 150

theorem find_angleQRS {PQ RS PS angleQPS angleRSP : ℝ} :
  angleQRS_proof PQ RS PS angleQPS angleRSP :=
by
  intro h,
  cases h,
  sorry

end find_angleQRS_l723_723532


namespace one_minus_repeating_decimal_l723_723907

theorem one_minus_repeating_decimal : 
  let x := (0:ℚ) + 123 / 999 in 1 - x = 292 / 333 := 
by
  let x := (0:ℚ) + 123 / 999
  show 1 - x = 292 / 333
  sorry

end one_minus_repeating_decimal_l723_723907


namespace least_cost_proof_l723_723839

-- Define the types of bags and their respective costs and weights
structure Bag := 
  (weight : Nat)
  (cost : Float)

def five_pound : Bag := ⟨5, 13.85⟩
def ten_pound : Bag := ⟨10, 20.43⟩
def twenty_five_pound : Bag := ⟨25, 32.20⟩
def forty_pound : Bag := ⟨40, 54.30⟩

-- Define the problem conditions
structure ProblemConditions :=
  (bags : List Bag)
  (min_weight : Nat := 65)
  (max_weight : Nat := 80)
  (min_each_type : Nat := 1)
  (max_total_bags : Nat := 5)

-- Create an instance of the ProblemConditions with the given bags
def problem_conditions := ProblemConditions.mk [five_pound, ten_pound, twenty_five_pound, forty_pound]

-- Define a function to calculate the total weight
def total_weight (bags : List Bag) : Nat := bags.foldl (fun acc b => acc + b.weight) 0

-- Define a function to calculate the total cost
def total_cost (bags : List Bag) : Float := bags.foldl (fun acc b => acc + b.cost) 0.0

-- Define the target result to be proved as the least cost
def least_cost := 120.78

-- The theorem statement rewritten in Lean 4
theorem least_cost_proof (conds : ProblemConditions) : 
  (65 ≤ total_weight conds.bags ∧ total_weight conds.bags ≤ 80) ∧ 
  (conds.bags.length ≤ 5) ∧ 
  (conds.bags.count five_pound ≥ 1) ∧ 
  (conds.bags.count ten_pound ≥ 1) ∧ 
  (conds.bags.count twenty_five_pound ≥ 1) ∧ 
  (conds.bags.count forty_pound ≥ 1) → 
  total_cost conds.bags = least_cost :=
by sorry

end least_cost_proof_l723_723839


namespace tegwen_family_total_children_l723_723708

variable (Tegwen : Type)

-- Variables representing the number of girls and boys
variable (g b : ℕ)

-- Conditions from the problem
variable (h1 : b = g - 1)
variable (h2 : g = (3/2:ℚ) * (b - 1))

-- Proposition that the total number of children is 11
theorem tegwen_family_total_children : g + b = 11 := by
  sorry

end tegwen_family_total_children_l723_723708


namespace prime_factors_of_30_factorial_l723_723099

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723099


namespace prime_factors_30_fac_eq_10_l723_723148

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723148


namespace scaling_transformation_correct_l723_723766

theorem scaling_transformation_correct :
  ∀ (x y : ℝ),
    (let x' := (1/2) * x in
     let y' := (√3 / 3) * y in
     (x^2 / 4 + y^2 / 3 = 1) → (x'^2 + y'^2 = 1)) :=
by
  intro x y
  let x' := (1 / 2) * x
  let y' := (√3 / 3) * y
  intro h
  sorry

end scaling_transformation_correct_l723_723766


namespace find_k_l723_723991

noncomputable def line1_slope : ℝ := -1
noncomputable def line2_slope (k : ℝ) : ℝ := -k / 3

theorem find_k (k : ℝ) : 
  (line2_slope k) * line1_slope = -1 → k = -3 := 
by
  sorry

end find_k_l723_723991


namespace minimize_vector_magnitude_l723_723556

variables {α : Type*} [inner_product_space ℝ α]

theorem minimize_vector_magnitude 
  (a b : α) (h₁ : ∥a∥ = 1) (h₂ : ∥b∥ = 1) (h₃ : ⟪a, b⟫ = 1 / 2) :
  ∃ λ : ℝ, λ = -1/2 ∧ ∀ μ : ℝ, ∥a + μ • b∥ ≥ ∥a + (-1/2) • b∥ :=
by
  sorry

end minimize_vector_magnitude_l723_723556


namespace length_of_chord_l723_723608

theorem length_of_chord (x y : ℝ) 
  (h1 : (x - 1)^2 + y^2 = 4) 
  (h2 : x + y + 1 = 0) 
  : ∃ (l : ℝ), l = 2 * Real.sqrt 2 := by
  sorry

end length_of_chord_l723_723608


namespace approximation_irrational_quotient_l723_723349

theorem approximation_irrational_quotient 
  (r1 r2 : ℝ) (irrational : ¬ ∃ q : ℚ, r1 = q * r2) 
  (x : ℝ) (p : ℝ) (pos_p : p > 0) : 
  ∃ (k1 k2 : ℤ), |x - (k1 * r1 + k2 * r2)| < p :=
sorry

end approximation_irrational_quotient_l723_723349


namespace expected_rolls_equal_2021_l723_723473

/-- 
  Broady The Boar is playing a boring board game consisting 
  of a circle with 2021 points on it, labeled 0, 1, 2, ..., 2020 in that order clockwise. 
  Broady rolls a 2020-sided die producing a number between 1 and 2020.
  Broady starts at the point labelled 0. After each dice roll, 
  they move up the same number of points as the number rolled (point 2020 is followed by point 0). 
  Broady continues rolling until they return to the point labelled 0.
  The expected number of times they roll the dice is 2021.
-/
theorem expected_rolls_equal_2021 : 
  ∃ E : ℕ, 
    (E = 2021 ∧ 
      ∀ (die_roll : ℕ), die_roll ∈ {x | 1 ≤ x ∧ x ≤ 2020} → 
        let expected_moves (pos : ℕ) := 
          if pos = 0 then 0 else 1 + 2019 / 2020 * E in
        E = 1 + expected_moves 0) :=
begin
  sorry
end

end expected_rolls_equal_2021_l723_723473


namespace number_of_prime_factors_of_30_factorial_l723_723114

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723114


namespace Sharon_cups_of_coffee_per_day_l723_723693

-- Define the conditions
def vacation_days := 40
def price_per_box := 8 -- dollars
def pods_per_box := 30
def total_spent := 32 -- dollars

-- Define the problem: number of cups of coffee per day
def cups_of_coffee_per_day (days : ℕ) (cost_per_box : ℕ) (number_of_pods_per_box : ℕ) (total_spending : ℕ) : ℕ := do
  let number_of_boxes := total_spending / cost_per_box
  let total_pods := number_of_boxes * number_of_pods_per_box
  total_pods / days

-- The theorem statement
theorem Sharon_cups_of_coffee_per_day : 
  cups_of_coffee_per_day vacation_days price_per_box pods_per_box total_spent = 3 := by
sorry

end Sharon_cups_of_coffee_per_day_l723_723693


namespace lion_weight_l723_723400

theorem lion_weight :
  ∃ (L : ℝ), 
    (∃ (T P : ℝ), 
      L + T + P = 106.6 ∧ 
      P = T - 7.7 ∧ 
      T = L - 4.8) ∧ 
    L = 41.3 :=
by
  sorry

end lion_weight_l723_723400


namespace intersection_M_N_eq_l723_723966

def M : Set ℝ := {x | 2^(x - 1) < 1}
def N : Set ℝ := {x | log 2 x < 1}

theorem intersection_M_N_eq : (M ∩ N) = {x | 0 < x ∧ x < 1} := by
  sorry

end intersection_M_N_eq_l723_723966


namespace conjecture_f_l723_723537

def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

theorem conjecture_f (x : ℝ) : f (-x) + f (1 + x) = Real.sqrt 2 / 2 :=
by
  sorry

end conjecture_f_l723_723537


namespace length_of_base_of_vessel_l723_723829

noncomputable def volume_of_cube (edge : ℝ) := edge ^ 3

theorem length_of_base_of_vessel 
  (cube_edge : ℝ)
  (vessel_width : ℝ)
  (rise_in_water_level : ℝ)
  (volume_cube : ℝ)
  (h1 : cube_edge = 15)
  (h2 : vessel_width = 15)
  (h3 : rise_in_water_level = 11.25)
  (h4 : volume_cube = volume_of_cube cube_edge)
  : ∃ L : ℝ, L = volume_cube / (vessel_width * rise_in_water_level) ∧ L = 20 :=
by
  sorry

end length_of_base_of_vessel_l723_723829


namespace factorial_prime_factors_l723_723089

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723089


namespace midpoints_of_segments_form_regular_hexagon_l723_723340

theorem midpoints_of_segments_form_regular_hexagon
  (hexagon : Type)
  (vertices : List hexagon)
  (convex_symmetric_hexagon : ∀ {A B C D E F : hexagon}, A ≠ B → B ≠ C → C ≠ D → D ≠ E → E ≠ F → F ≠ A →
    equilateral_triangle A B C ∧ equilateral_triangle B C D ∧ equilateral_triangle C D E ∧
    equilateral_triangle D E F ∧ equilateral_triangle E F A ∧ equilateral_triangle F A B) :
  ∃ (midpoints : List hexagon), regular_hexagon midpoints := sorry

end midpoints_of_segments_form_regular_hexagon_l723_723340


namespace fly_distance_to_ceiling_l723_723444

theorem fly_distance_to_ceiling (d1 d2 d3 : ℝ) (h : d1 = 2 ∧ d2 = 6 ∧ d3 = 11) :
  ∃ z : ℝ, z = 9 ∧ d3 = Real.sqrt((d1 - 0)^2 + (d2 - 0)^2 + (z - 0)^2) :=
by
  sorry

end fly_distance_to_ceiling_l723_723444


namespace prime_factors_of_30_factorial_l723_723183

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723183


namespace cow_count_l723_723447

theorem cow_count
  (initial_cows : ℕ) (cows_died : ℕ) (cows_sold : ℕ)
  (increase_cows : ℕ) (gift_cows : ℕ) (final_cows : ℕ) (bought_cows : ℕ) :
  initial_cows = 39 ∧ cows_died = 25 ∧ cows_sold = 6 ∧
  increase_cows = 24 ∧ gift_cows = 8 ∧ final_cows = 83 →
  bought_cows = 43 :=
by
  sorry

end cow_count_l723_723447


namespace coefficient_binomial_expansion_l723_723372

theorem coefficient_binomial_expansion (a : ℝ) :
  (nat.choose 8 3 * a^5 = 56) → a = 1 :=
by
  sorry

end coefficient_binomial_expansion_l723_723372


namespace sum_of_possible_values_g1_non_const_poly_l723_723653

noncomputable def g (x : ℝ) : ℝ := 6051 * x -- this will be the assumption based on our proof 
-- but ideally, we derive it instead of defining directly for the automated proof.

theorem sum_of_possible_values_g1_non_const_poly (g : ℝ → ℝ) (h : ¬ ∀ x : ℝ, g x = 0) :
  (∀ x : ℝ, x ≠ 0 → g(x - 1) + g(x) + g(x + 1) = (g(x))^2 / (2021 * x)) →
  g 1 = 6051 :=
begin
  sorry
end

end sum_of_possible_values_g1_non_const_poly_l723_723653


namespace prime_factors_of_30_l723_723213

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723213


namespace ellipse_standard_equation_max_area_triangle_est_l723_723954

theorem ellipse_standard_equation (a b c : ℝ) (λ : ℝ) (h_λ_pos : λ > 0) (h_λ_ne_one : λ ≠ 1)
  (h_major_axis : 4 * sqrt 2 = 4 * sqrt 2) 
  (h_a : a = 2 * sqrt 2) (h_eq : a^2 = b^2 + c^2) (h_c_val : c^2 = 2) :
  (a = 2 * sqrt 2) → (a^2 = 8) → (b^2 = 6) → 
  (frac x^2 8 + frac y^2 6 = 1) :=
by
  sorry

theorem max_area_triangle_est (a : ℝ) (λ : ℝ) (h_a : a = 2 * sqrt 2) 
  (h_circle_eq : x^2 + y^2 = 4) 
  (h_line_eq : True) -- Assuming there exists a line through the point A
  (h_intersections : True)  -- Assuming that S and T are the points of intersection
  (h_y1_y2 : ∀ (m : ℝ), disjunction
  (max_area : 3) : 
  area_triangle_est ≤ 3 :=
by
  sorry

end ellipse_standard_equation_max_area_triangle_est_l723_723954


namespace factorial_30_prime_count_l723_723199

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723199


namespace sonya_falls_count_l723_723699

/-- The number of times Sonya fell down given the conditions. -/
theorem sonya_falls_count : 
  let steven_falls := 3 in
  let stephanie_falls := steven_falls + 13 in
  let sonya_falls := (stephanie_falls / 2) - 2 in
  sonya_falls = 6 := 
by
  sorry

end sonya_falls_count_l723_723699


namespace xiaoming_correct_answers_l723_723278

theorem xiaoming_correct_answers (x : ℕ) (h1 : x ≤ 10) (h2 : 5 * x - (10 - x) > 30) : x ≥ 7 := 
by
  sorry

end xiaoming_correct_answers_l723_723278


namespace pow_mod_79_l723_723776

theorem pow_mod_79 (a : ℕ) (h : a = 7) : a^79 % 11 = 6 := by
  sorry

end pow_mod_79_l723_723776


namespace num_prime_factors_30_factorial_l723_723247

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723247


namespace distance_A_beats_B_l723_723826

noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem distance_A_beats_B :
  let distance_A := 5 -- km
  let time_A := 10 / 60 -- hours (10 minutes)
  let time_B := 14 / 60 -- hours (14 minutes)
  let speed_A := speed distance_A time_A
  let speed_B := speed distance_A time_B
  let distance_A_in_time_B := speed_A * time_B
  distance_A_in_time_B - distance_A = 2 := -- km
by
  sorry

end distance_A_beats_B_l723_723826


namespace num_prime_factors_of_30_l723_723221

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723221


namespace arithmetic_sequences_count_geometric_sequences_count_l723_723971

-- Definition for arithmetic sequence count based on given conditions
theorem arithmetic_sequences_count :
  (∑ d in finset.range 22, (90 - 4 * (d + 1))) = 968 :=
begin
  sorry
end

-- Definition for geometric sequence count based on given conditions
theorem geometric_sequences_count :
  let count_q2 := 5,
      count_q3 := 1 in
  count_q2 + count_q3 + 1 = 7 :=
begin
  sorry
end

end arithmetic_sequences_count_geometric_sequences_count_l723_723971


namespace ratio_of_a_b_l723_723980

-- Define the problem
theorem ratio_of_a_b (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it.
  sorry

end ratio_of_a_b_l723_723980


namespace boxes_needed_to_pack_all_muffins_l723_723723

theorem boxes_needed_to_pack_all_muffins
  (total_muffins : ℕ := 95)
  (muffins_per_box : ℕ := 5)
  (available_boxes : ℕ := 10) :
  (total_muffins / muffins_per_box) - available_boxes = 9 :=
by
  sorry

end boxes_needed_to_pack_all_muffins_l723_723723


namespace factorial_prime_factors_l723_723091

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723091


namespace region_area_correct_l723_723396

-- Define the conditions for the region
def region (x y : ℝ) : Prop :=
  |x - 4| ≤ y ∧ y ≤ 5 - |x - 2|

-- Define the problem of computing the area of the region
theorem region_area_correct :
  (∃ area : ℝ, area = 12.875 ∧ 
  (∃ (points : set (ℝ × ℝ)), 
    (∀ p : ℝ × ℝ, p ∈ points ↔ region p.1 p.2) ∧ 
    (set.volume points = area))) :=
sorry

end region_area_correct_l723_723396


namespace factorial_30_prime_count_l723_723185

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723185


namespace max_filled_black_squares_l723_723823

def grid_size := 2^(2014) + 1
def max_black_squares := (2 / 3) * grid_size * grid_size - 1

/-- 
  Given a grid of size (2^(2014) + 1) × (2^(2014) + 1) with black squares forming non-cyclic snakes,
  the maximum number of such filled squares.
-/
theorem max_filled_black_squares :
  ∀ (n : ℕ), (n = grid_size) → 
  ∃ v : ℕ, v = max_black_squares :=
begin
  sorry
end

end max_filled_black_squares_l723_723823


namespace precision_of_approx_0_598_thousandth_l723_723517

theorem precision_of_approx_0_598_thousandth :
  precision_of_approx (0.598: ℝ) = "thousandth" :=
sorry

end precision_of_approx_0_598_thousandth_l723_723517


namespace find_vector_v_l723_723920

noncomputable def vector_v : Type := ℝ × ℝ
def proj (u v : vector_v) : vector_v := 
  let dot_uu := u.1 * u.1 + u.2 * u.2 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  (dot_uv / dot_uu * u.1, dot_uv / dot_uu * u.2)

def v (x y : ℝ) : vector_v := (x, y)
def u1 : vector_v := (3, 2)
def u2 : vector_v := (1, 4)
def p1 : vector_v := (45/13, 30/13)
def p2 : vector_v := (32/17, 128/17)

theorem find_vector_v (x y : ℝ) (h1 : proj u1 (v x y) = p1) (h2 : proj u2 (v x y) = p2) :
  ∃ (a b : ℝ), v x y = (a, b) :=
sorry

end find_vector_v_l723_723920


namespace factorial_30_prime_count_l723_723191

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723191


namespace combined_average_score_l723_723676

theorem combined_average_score (M E : ℕ) (m e : ℕ) (h1 : M = 82) (h2 : E = 68) (h3 : m = 5 * e / 7) :
  ((m * M) + (e * E)) / (m + e) = 72 :=
by
  -- Placeholder for the proof
  sorry

end combined_average_score_l723_723676


namespace value_of_g_at_8_l723_723604

def g (x : ℝ) : ℝ := (x - 2) / (2 * x + 3)

theorem value_of_g_at_8 : g 8 = 6 / 19 :=
by
  sorry

end value_of_g_at_8_l723_723604


namespace parallel_CX_BY_l723_723664

variables {A B C I X Y : Type} [triangle ABC] [point I] [point X] [point Y]

-- Introduce conditions
axiom angle_bisectors_meet_at_I : 
  (angle_bisector_ABC B I) ∧ (angle_bisector_ABC C I)

axiom circumcircle_BIC_intersects_AB_at_X : 
  (circumcircle B I C).intersects (segment A B) a second time at X

axiom circumcircle_BIC_intersects_AC_at_Y : 
  (circumcircle B I C).intersects (segment A C) a second time at Y

-- Statement to prove
theorem parallel_CX_BY : 
  ∀ (A B C I X Y : Type) [triangle ABC] [point I] [point X] [point Y], 
  (AB > AC) → 
  (angle_bisectors_meet_at_I I) → 
  (circumcircle_BIC_intersects_AB_at_X X) → 
  (circumcircle_BIC_intersects_AC_at_Y Y) → 
  (parallel (line C X) (line B Y)) :=
by
  intros,
  sorry

end parallel_CX_BY_l723_723664


namespace oranges_to_put_back_l723_723873

theorem oranges_to_put_back
  (price_apple price_orange : ℕ)
  (A_all O_all : ℕ)
  (mean_initial_fruit mean_final_fruit : ℕ)
  (A O x : ℕ)
  (h_price_apple : price_apple = 40)
  (h_price_orange : price_orange = 60)
  (h_total_fruit : A_all + O_all = 10)
  (h_mean_initial : mean_initial_fruit = 54)
  (h_mean_final : mean_final_fruit = 50)
  (h_total_cost_initial : price_apple * A_all + price_orange * O_all = mean_initial_fruit * (A_all + O_all))
  (h_total_cost_final : price_apple * A + price_orange * (O - x) = mean_final_fruit * (A + (O - x)))
  : x = 4 := 
  sorry

end oranges_to_put_back_l723_723873


namespace book_distribution_ways_l723_723847

/-- 
Problem Statement: We have 8 identical books. We want to find out the number of ways to distribute these books between the library and checked out such that at least one book is in the library and at least one book is checked out. The expected answer is 7.
-/
theorem book_distribution_ways : ∃ n : ℕ, n = 7 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 7 ↔ k books in library means exactly 8 - k books are checked out) :=
by
  sorry

end book_distribution_ways_l723_723847


namespace force_for_wrenches_l723_723731

open Real

theorem force_for_wrenches (F : ℝ) (k : ℝ) :
  (F * 12 = 3600) → 
  (k = 3600) →
  (3600 / 8 = 450) →
  (3600 / 18 = 200) →
  true :=
by
  intro hF hk h8 h18
  trivial

end force_for_wrenches_l723_723731


namespace percentage_owning_cats_percentage_owning_birds_l723_723623

def total_students : ℕ := 500
def students_owning_cats : ℕ := 80
def students_owning_birds : ℕ := 120

theorem percentage_owning_cats : students_owning_cats * 100 / total_students = 16 := 
by 
  sorry

theorem percentage_owning_birds : students_owning_birds * 100 / total_students = 24 := 
by 
  sorry

end percentage_owning_cats_percentage_owning_birds_l723_723623


namespace mitosis_correct_option_l723_723789

def option_A : Prop :=
  "The interphase prepares active substances for the division phase, completes DNA molecule replication, and synthesizes related proteins."

def option_B : Prop :=
  "Helicase, RNA polymerase, DNA polymerase, and DNAase are required throughout the entire cell division process."

def option_C : Prop :=
  "The number of DNA molecules and chromosomes in the late division phase is twice that of the previous phase."

def option_D : Prop :=
  "The cell plate in the late division phase expands from the center of the cell to the surrounding area, gradually forming the cell wall."

def correct_option : Prop :=
  option_A

theorem mitosis_correct_option : correct_option :=
  by 
  -- Proof needs to be filled in
  sorry

end mitosis_correct_option_l723_723789


namespace cistern_filling_time_l723_723806

theorem cistern_filling_time :
  let rate_P := (1 : ℚ) / 12
  let rate_Q := (1 : ℚ) / 15
  let combined_rate := rate_P + rate_Q
  let time_combined := 6
  let filled_after_combined := combined_rate * time_combined
  let remaining_after_combined := 1 - filled_after_combined
  let time_Q := remaining_after_combined / rate_Q
  time_Q = 1.5 := sorry

end cistern_filling_time_l723_723806


namespace jason_pears_l723_723657

theorem jason_pears (Pj Pm : ℤ) (Pk : ℚ) :
  Pm = 12 → 
  Pk = 47 → 
  Pj = Int.nearest (3 / 4 * (46 + Pm)) → 
  Pj + Pm = 56 :=
by
  intros hPm hPk hPj
  have h1 : 46 + Pm = 58, by linarith [hPm]
  have h2 : (3 / 4 : ℚ) * 58 = 43.5, by norm_num
  have h3 : Int.nearest 43.5 = 44, by norm_num
  have hPj' : Pj = 44, by rw [h2, h3] at hPj; exact hPj
  rw [hPj', hPm]
  norm_num

end jason_pears_l723_723657


namespace solve_equation_l723_723912

noncomputable def a := 3 + Real.sqrt 8
noncomputable def b := 3 - Real.sqrt 8

theorem solve_equation (x : ℝ) :
  (Real.sqrt (a^x) + Real.sqrt (b^x) = 6) ↔ (x = 2 ∨ x = -2) := 
  by
  sorry

end solve_equation_l723_723912


namespace quadratic_poly_unique_l723_723908

-- Define f as a quadratic polynomial
def is_quadratic_poly (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f(x) = a * x^2 + b * x + c

-- Define the expression for f(2x + 1)
def f (x : ℝ) : ℝ := x^2 + 5 * x + 1

-- State the theorem to prove f satisfies the given condition
theorem quadratic_poly_unique :
  is_quadratic_poly f ∧ (∀ x, f (2 * x + 1) = 4 * x^2 + 14 * x + 7) :=
sorry

end quadratic_poly_unique_l723_723908


namespace meeting_probability_l723_723370
noncomputable theory

-- We define our probability in Lean
def probability_successful_meeting : ℚ := 625 / 2700

theorem meeting_probability :
  probability_successful_meeting = 625 / 2700 :=
begin
  sorry  -- Proof will be provided elsewhere
end

end meeting_probability_l723_723370


namespace problem1_problem2_l723_723476

theorem problem1 : 27 ^ (2/3) + 16 ^ (-1/2) - (1/2) ^ (-2) - (8/27) ^ (-2/3) = 3 :=
sorry

theorem problem2 : real.logb 10 14 - 2 * real.logb 10 (7/3) + real.logb 10 7 - real.logb 10 18 = 0 :=
sorry

end problem1_problem2_l723_723476


namespace factorial_30_prime_count_l723_723189

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723189


namespace num_prime_factors_of_30_l723_723235

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723235


namespace calculation_l723_723477

theorem calculation :
  (- (1 / 2))^0 + (-2)^3 + (1 / 3)^(-1 : ℤ) + abs (-2) = -2 :=
by
  sorry

end calculation_l723_723477


namespace find_x_l723_723323

variable (a b x y : ℝ) (r : ℝ)

-- Conditions
def triple_base_exponent (a b : ℝ) : ℝ := (3 * a) ^ (3 * b)
axiom neq_zero_b : b ≠ 0
axiom r_equals_product : triple_base_exponent a b = a ^ b * ((x + y) ^ b)
axiom y_equals_3a : y = 3 * a

-- Question to prove
theorem find_x (a_ne_zero : a ≠ 0) : x = 27 * a ^ 2 - 3 * a :=
by
  sorry

end find_x_l723_723323


namespace num_prime_factors_of_30_l723_723236

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723236


namespace number_of_incorrect_statements_is_3_l723_723469
open Classical

-- Definitions of the propositions as conditions
def proposition_1 : Prop :=
  ∀ (a_n : ℕ → ℤ) (S_n : ℕ → ℤ), 
  (∀ n, S_n n = ∑ i in range (n+1), a_n i) →
  (∀ n, a_n n + a_n (n+1) = k) →
  (a_n 6 + a_n 7 > 0 ↔ S_n 9 ≥ S_n 3)

def proposition_2 : Prop :=
  ¬ (∃ x : ℝ, x > 1) ↔ ∀ x : ℝ, x < 1

def proposition_3 : Prop :=
  (x^2 - 4x + 3 = 0 → x = 1 ∨ x = 3) ↔ (x ≠ 1 ∨ x ≠ 3) → x^2 - 4x + 3 ≠ 0

def proposition_4 : Prop :=
  ∀ p q : Prop, ¬ (p ∨ q) → ¬ p ∧ ¬ q

-- Main theorem statement
theorem number_of_incorrect_statements_is_3 : (¬ proposition_1) ∧ (¬ proposition_2) ∧ (¬ proposition_3) ∧ proposition_4 → 3 :=
by
  -- Proof omitted
  sorry

end number_of_incorrect_statements_is_3_l723_723469


namespace number_of_prime_factors_thirty_factorial_l723_723018

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723018


namespace bases_inequality_l723_723659

-- Definitions
variable (a b : ℕ) (x0 x1 : ℕ) (n : ℕ)
variable (x : ℕ → ℕ)
variable (A_n A_n1 B_n B_n1 : ℕ)
variable (h0 : x0 ≠ 0) (h1 : x1 ≠ 0)

-- Statement (proof problem)
theorem bases_inequality (hAn : A_n = List.sum (List.map (λ i, x i * a^(n - i)) (List.range n)))
                         (hAn1 : A_n1 = x0 * a^n + A_n)
                         (hBn : B_n = List.sum (List.map (λ i, x i * b^(n - i)) (List.range n)))
                         (hBn1 : B_n1 = x0 * b^n + B_n) :
  a > b ↔ (A_n.to_rat / A_n1.to_rat) < (B_n.to_rat / B_n1.to_rat) := 
sorry

end bases_inequality_l723_723659


namespace triangle_abc_arithmetic_sequence_l723_723564

theorem triangle_abc_arithmetic_sequence (a b c : ℝ) (h1 : a + c = 2 * b) (h2 : a^2 + b^2 + c^2 = 21) :
  b ∈ Ioc (Real.sqrt 6) (Real.sqrt 7) :=
sorry

end triangle_abc_arithmetic_sequence_l723_723564


namespace find_a_purely_imaginary_z1_z2_l723_723568

noncomputable def z1 (a : ℝ) : ℂ := ⟨a^2 - 3, a + 5⟩
noncomputable def z2 (a : ℝ) : ℂ := ⟨a - 1, a^2 + 2 * a - 1⟩

theorem find_a_purely_imaginary_z1_z2 (a : ℝ)
    (h_imaginary : ∃ b : ℝ, z2 a - z1 a = ⟨0, b⟩) : 
    a = -1 :=
sorry

end find_a_purely_imaginary_z1_z2_l723_723568


namespace policeman_speed_l723_723850

theorem policeman_speed (D_i : ℕ) (v_c : ℕ) (t_min : ℕ) (D_f : ℚ) :
  let t := (t_min : ℚ) / 60,
      D_c := v_c * t,
      D_p := D_i - D_f
  in D_i = 180 ∧ v_c = 8 ∧ t_min = 5 ∧ D_f = 96.66666666666667 → (D_p / t = 1000) :=
by
  intros
  sorry

end policeman_speed_l723_723850


namespace sum_of_three_numbers_l723_723739

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a <= 10) (h2 : 10 <= c)
  (h3 : (a + 10 + c) / 3 = a + 8)
  (h4 : (a + 10 + c) / 3 = c - 20) :
  a + 10 + c = 66 :=
by
  sorry

end sum_of_three_numbers_l723_723739


namespace prime_factors_of_30_l723_723208

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723208


namespace proportion_students_le_64_students_between_80_and_96_l723_723828

theorem proportion_students_le_64 (mu sigma : ℝ) (n : ℕ) (hmu : mu = 72) (hsigma : sigma = 8) 
  (dist : X ∼ Normal mu sigma^2) (prob : ∀ k : ℤ, P(mu - k * sigma < X ∧ X < mu + k * sigma) = given_probs k)  :
  (P(X ≤ 64) = 0.15865) := 
sorry

theorem students_between_80_and_96 (mu sigma : ℝ) (n : ℕ) (hmu : mu = 72) (hsigma : sigma = 8) 
  (dist : X ∼ Normal mu sigma^2) (prob : ∀ k : ℤ, P(mu - k * sigma < X ∧ X < mu + k * sigma) = given_probs k) 
  (total_students : n = 2000) :
  ((P(80 < X ∧ X < 96) * n) = 314) :=
sorry

end proportion_students_le_64_students_between_80_and_96_l723_723828


namespace prime_factors_30_fac_eq_10_l723_723138

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723138


namespace point_on_transformed_plane_l723_723654

-- Define the conditions
def pointA : ℝ × ℝ × ℝ := (-1, 1, 1)
def planeA (x y z : ℝ) : Prop := 3 * x - y + 2 * z + 4 = 0
def scaleFactor : ℝ := 1 / 2

-- Define the transformed plane equation
def transformedPlane (x y z : ℝ) : Prop := 3 * x - y + 2 * z + (scaleFactor * 4) = 0

-- Prove that point A lies on the transformed plane
theorem point_on_transformed_plane : transformedPlane (pointA.1) (pointA.2) pointA.3 := by
  -- Substitute pointA into the transformedPlane equation and simplify
  calc
    3 * (-1) - 1 + 2 * 1 + 2 = 0 := by
      sorry -- Detailed proof step calculations here

end point_on_transformed_plane_l723_723654


namespace factorial_prime_factors_l723_723161

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723161


namespace factorial_prime_factors_l723_723158

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723158


namespace count_of_indivisible_numbers_gt_2_l723_723851

def is_indivisible (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 < k ∧ k < n ∧ Nat.gcd k n = 1 → Nat.Prime k

def indivisible_numbers : List ℕ :=
  [3, 4, 6, 8, 12, 18, 24, 30]

theorem count_of_indivisible_numbers_gt_2 :
  indivisible_numbers.length = 8 ∧
  ∀ n ∈ indivisible_numbers, is_indivisible n ∧ 2 < n :=
by
  sorry

end count_of_indivisible_numbers_gt_2_l723_723851


namespace clock_angles_correct_l723_723472

noncomputable def right_angle := 90
noncomputable def straight_angle := 180

def angle_at_three_oclock : ℤ := right_angle
def angle_at_six_oclock : ℤ := straight_angle

theorem clock_angles_correct :
  ((angle_at_three_oclock = right_angle) ∧ (angle_at_six_oclock = straight_angle)) :=
by
  split
  { 
    exact Eq.refl right_angle,
  }
  { 
    exact Eq.refl straight_angle,
  }

end clock_angles_correct_l723_723472


namespace factorial_prime_factors_l723_723078

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723078


namespace prime_factors_of_30_l723_723211

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723211


namespace inequality_solution_l723_723519

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := abs ((7 - 2 * x) / 4) < 3

-- Define the correct answer as an interval
def correct_interval (x : ℝ) : Prop := -2.5 < x ∧ x < 9.5

-- The theorem states that the inequality condition and correct interval are equivalent
theorem inequality_solution (x : ℝ) : inequality_condition x ↔ correct_interval x := 
sorry

end inequality_solution_l723_723519


namespace pi_irrational_l723_723786

theorem pi_irrational :
  ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (π = a / b) :=
by
  sorry

end pi_irrational_l723_723786


namespace graph_representation_l723_723460

-- Define the conditions of the problem
def semicircle_path (A B X : Point) (r : ℝ) : Prop :=
  distance A X = r ∧ distance B X = r ∧
  ∀ p, (on_path p A B) → distance p X = r

def curved_path (B C X Y Z : Point) (distance1 distance2 : ℝ) : Prop :=
  distance B X = distance1 ∧ distance Y X < distance1 ∧
  distance Z X > distance Y X ∧
  ∀ p, (on_path p Y Z) → distance p X = distance2

-- Define the journey from A to C
def journey_A_C (A B C X Y Z : Point) (r distance1 distance2 : ℝ) : Prop :=
  semicircle_path A B X r ∧ curved_path B C X Y Z distance1 distance2

-- Prove the graph characteristics
theorem graph_representation (A B C X Y Z : Point) (r distance1 distance2 : ℝ) :
  journey_A_C A B C X Y Z r distance1 distance2 → 
  graph_variation (horizontal_line A B r) ∧
  graph_variation (dips_rise A B C X Y Z) :=
sorry

end graph_representation_l723_723460


namespace taxi_fare_distance_l723_723754

theorem taxi_fare_distance :
  ∃ x : ℝ, x = 4.35 ∧
           (let fare := 3 + 0.25 * ((x - 0.75) / 0.1) in
            let total_fare := fare + 3 in
            total_fare = 15) :=
by
  sorry

end taxi_fare_distance_l723_723754


namespace num_prime_factors_30_factorial_l723_723252

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723252


namespace factorial_prime_factors_l723_723084

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723084


namespace quartic_polynomial_form_l723_723705

theorem quartic_polynomial_form (q : ℝ[X]) (h₁ : q.monic) 
(h₂ : q.eval (5 - 3 * complex.I) = 0) (h₃ : q.eval 0 = -108) :
  q = X^4 - 12*X^3 + 65*X^2 - 148*X - 108 :=
by
  sorry

end quartic_polynomial_form_l723_723705


namespace factorial_prime_factors_l723_723079

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723079


namespace find_a_range_l723_723990

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

theorem find_a_range (a : ℝ) :
  is_monotonically_increasing (λ x : ℝ, x^2 + 2*a*x + 1) 1 2 → (-1 ≤ a) :=
by
  intro h
  sorry

end find_a_range_l723_723990


namespace largest_n_unique_k_l723_723775

theorem largest_n_unique_k (n k : ℕ) :
  (frac9_17_lt_frac n (n + k) ∧ frac n (n + k) lt frac8_15) 
  ∧ (∀ (n1 k1 : ℕ), frac9_17_lt_frac n1 (n1 + k1) ∧ frac n1 (n1 + k1) lt frac8_15 
  → (n1 ≤ 136 
  ∧ ((n1 = 136) → (k1 = unique_k))))
  :=
sorry

def frac9_17_lt_frac (a b : ℕ) : Prop := 
  (9:ℚ) / 17 < (a : ℚ) / b

def frac (a b : ℕ) : ℚ :=
  (a : ℚ) / b

def frac8_15 := 
  (8:ℚ) / 15

def unique_k : ℕ :=
  119

end largest_n_unique_k_l723_723775


namespace circle_equation_l723_723513

theorem circle_equation {x y : ℝ} :
  let center := (1, 3)
  let line := λ x y : ℝ, 3 * x - 4 * y - 6 = 0
  let radius := 3
  (3 * 1 - 4 * 3 - 6).abs / real.sqrt (3^2 + (-4)^2) = radius →
  (x - 1)^2 + (y - 3)^2 = radius^2 :=
by
  intros h
  sorry

end circle_equation_l723_723513


namespace num_prime_factors_30_factorial_l723_723023

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723023


namespace rectangular_coordinate_equation_length_of_chord_l723_723279

theorem rectangular_coordinate_equation (rho theta : ℝ) :
  (rho * real.sin theta)^2 = 8 * (rho * real.cos theta) → 
  ∃ x y : ℝ, y^2 = 8 * x :=
by
  intro h
  sorry

theorem length_of_chord (t1 t2 : ℝ) :
  (2 + 1/2 * t1 + 2 + 1/2 * t2 = 8) →
  (sqrt(3)/2 * t1 + sqrt(3)/2 * t2)^2 = 8 * (2 + 1/2 * t1) * (2 + 1/2 * t2) →
  abs (t1 - t2) = sqrt ((t1 + t2)^2 - 4 * t1 * t2) = 32/3 :=
by
  intros h1 h2
  sorry

end rectangular_coordinate_equation_length_of_chord_l723_723279


namespace eval_floor_neg_sqrt_l723_723898

theorem eval_floor_neg_sqrt : (Int.floor (-Real.sqrt (64 / 9)) = -3) := sorry

end eval_floor_neg_sqrt_l723_723898


namespace gain_percent_calculation_l723_723374

theorem gain_percent_calculation (MP CP SP Discount Gain : ℝ) 
  (h1 : CP = 0.55 * MP)
  (h2 : Discount = 0.15 * MP)
  (h3 : SP = MP - Discount)
  (h4 : Gain = SP - CP)
  (h5 : Gain_percent = (Gain / CP) * 100) :
  Gain_percent ≈ 54.55 := by
  sorry

end gain_percent_calculation_l723_723374


namespace water_height_sum_l723_723862

def tank_radius := 12
def tank_height := 72
def water_percentage := 0.4

theorem water_height_sum :
  ∃ (a b : ℕ), 
    let h := a * (Real.cbrt b) in 
    a > 0 ∧ 
    b > 0 ∧ 
    b % 8 ≠ 0 ∧ 
    h = 36 * Real.cbrt (16 / 5) ∧ 
    a + b = 57 :=
sorry

end water_height_sum_l723_723862


namespace ratio_when_volume_maximized_l723_723793

-- Definitions based on conditions
def cylinder_perimeter := 24

-- Definition of properties derived from maximizing the volume
def max_volume_height := 4

def max_volume_circumference := 12 - max_volume_height

-- The ratio of the circumference of the cylinder's base to its height when the volume is maximized
def max_volume_ratio := max_volume_circumference / max_volume_height

-- The theorem to be proved
theorem ratio_when_volume_maximized :
  max_volume_ratio = 2 :=
by sorry

end ratio_when_volume_maximized_l723_723793


namespace correct_props_l723_723943

variables {α β γ : Plane} {m n l : Line}

-- Definitions of the propositions based on the conditions
def prop1 := ∀ m α β, (m ∥ α ∧ m ∥ β) → α ∥ β
def prop2 := ∀ m n l, (m ⟂ l ∧ n ⟂ l) → m ∥ n
def prop3 := ∀ l α β, (l ⟂ α ∧ l ∥ β) → α ⟂ β
def prop4 := ∀ α β l, (α ⟂ l ∧ β ⟂ l) → α ∥ β

-- Stating the problem of proof for propositions 3 and 4.
theorem correct_props : prop3 ∧ prop4 :=
by
  split;
  sorry

end correct_props_l723_723943


namespace baker_remaining_cakes_l723_723875

def initial_cakes : ℝ := 167.3
def sold_cakes : ℝ := 108.2
def remaining_cakes : ℝ := initial_cakes - sold_cakes

theorem baker_remaining_cakes : remaining_cakes = 59.1 := by
  sorry

end baker_remaining_cakes_l723_723875


namespace measure_angle_MFN_is_pi_over_2_l723_723734

open Real

-- Define hyperbola and its properties
def hyperbola (x y : ℝ) : Prop := x^2 - (y^2) / 3 = 1

-- Define points A and F related to the hyperbola
def A : ℝ × ℝ := (-1, 0)
def F : ℝ × ℝ := (2, 0)

-- Define the line passing through F and its intersection with the hyperbola
def line_through_F (k x : ℝ) : ℝ := k * (x - 2)

-- Define the vertical line l: x = 1/2
def line_l (x : ℝ) : Prop := x = 1/2

-- Define the points of intersection M and N on the line l
def point_M (x y : ℝ) : Prop := line_l x ∧ ∃ k : ℝ, hyperbola x y ∧ line_through_F k x = y
def point_N (x y : ℝ) : Prop := line_l x ∧ ∃ k : ℝ, hyperbola x y ∧ line_through_F k x = y

-- Define the measure of the angle MFN
def angle_MFN (M N F : ℝ × ℝ) : ℝ := 
ArcCos ((fst M - fst F) * (fst N - fst F) + (snd M - snd F) * (snd N - snd F) / 
(sqrt((fst M - fst F)^2 + (snd M - snd F)^2) * sqrt((fst N - fst F)^2 + (snd N - snd F)^2)))

-- The main theorem stating that the angle MFN is pi/2
theorem measure_angle_MFN_is_pi_over_2 (M N : ℝ × ℝ) (hM : point_M (fst M) (snd M))
  (hN : point_N (fst N) (snd N)) : 
  angle_MFN M N F = π / 2 := sorry

end measure_angle_MFN_is_pi_over_2_l723_723734


namespace mitzel_spent_amount_l723_723331

variable (AllowanceLeft : ℝ) (SpentPercentage : ℝ)
variable [fact (AllowanceLeft = 26)] [fact (SpentPercentage = 0.35)]

theorem mitzel_spent_amount (TotalAllowance : ℝ) (SpentAmount : ℝ) :
  (0.65 * TotalAllowance = 26) → (SpentAmount = 0.35 * TotalAllowance) → SpentAmount = 14 :=
by
  intro h1 h2
  sorry

end mitzel_spent_amount_l723_723331


namespace num_prime_factors_30_factorial_l723_723249

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723249


namespace bottom_face_is_red_l723_723691

/-- Problem conditions:
Seven squares are colored, front and back, (R = red, B = blue, O = orange, Y = yellow,
G = green, W = white, P = purple). They are hinged together as shown, then folded to form a cube
with one square remaining flat. If the purple face (P) is facing outward (i.e., not used in the
cube formation), and the white face (W) is at the top of the cube, --/

def bottom_face_color (squares : list char) (top_face : char) (outward_face : char) : char :=
sorry

theorem bottom_face_is_red : bottom_face_color ['R', 'B', 'O', 'Y', 'G', 'W', 'P'] 'W' 'P' = 'R' :=
by sorry

end bottom_face_is_red_l723_723691


namespace percentage_increase_of_x_l723_723361

theorem percentage_increase_of_x (x y k q : Real) (hq : 0 < q) (hy : y > 0) (hk : x * y = k) (y_decrease : y' = y - y * q / 100) :
  let x' := k / y' in
  (x' - x) / x * 100 = 100 * q / (100 - q) := 
by
  sorry

end percentage_increase_of_x_l723_723361


namespace min_words_to_score_l723_723976

theorem min_words_to_score (total_words : ℕ) (desired_percentage : ℕ) (required_score : ℕ) : 
  total_words = 800 ∧ desired_percentage = 90 ∧ required_score = (desired_percentage * total_words) / 100 → 
  required_score = 720 :=
by
  intro h
  cases h with h_total h1
  cases h1 with h_percentage h_score
  rw [h_total, h_percentage] at h_score
  exact h_score

end min_words_to_score_l723_723976


namespace prime_factors_30_fac_eq_10_l723_723143

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723143


namespace equilateral_cylinder_l723_723548

noncomputable def calculate_surface_and_volume
  (p : ℝ) -- semi-perimeter of the triangle
  (α : ℝ) -- angle α in degrees
  (β : ℝ) -- angle β in degrees
  : Prop :=
∃ (F V : ℝ),
  F = 310 ∧
  V = 421 ∧
  p = 21 ∧
  α = 53 + 7/60 + 50/3600 ∧
  β = 59 + 29/60 + 22/3600

-- Given conditions
theorem equilateral_cylinder
  (h_p : 2 * 21 = 42)
  (h_α : 53 + 7/60 + 50/3600 = 53.13056)
  (h_β : 59 + 29/60 + 22/3600 = 59.48944) :
  calculate_surface_and_volume 21 53.13056 59.48944 :=
by {
  use [310, 421],
  simp [calculate_surface_and_volume, h_p, h_α, h_β],
  sorry
}

end equilateral_cylinder_l723_723548


namespace num_prime_factors_of_30_l723_723233

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723233


namespace num_prime_factors_30_factorial_l723_723256

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723256


namespace optimal_strategy_l723_723681

-- Define the parameters and conditions
def harm_extra_feeding (α : ℝ) := α
def harm_missing_feeding (α : ℝ) := 2 * α

-- Probabilities
def prob_wrong_child : ℝ := 1 / 4

-- Expected harm calculations
def expected_harm_strategy1 (α : ℝ) := 1 / 2 * harm_extra_feeding α
def expected_harm_strategy2 (α : ℝ) := prob_wrong_child / 2 * harm_extra_feeding α + prob_wrong_child / 2 * harm_missing_feeding α

-- Statement of the proof problem: show that strategy 2 has less expected harm
theorem optimal_strategy (α : ℝ) : expected_harm_strategy2 α < expected_harm_strategy1 α := by {
  sorry
}

end optimal_strategy_l723_723681


namespace median_of_set_l723_723982

theorem median_of_set {a b : ℝ} (ha : a ≠ 0) (hb : b > 1) (h : a * b^2 = Real.log_b b 10) : 
  median_of ({0, 1, a, b, 1/b} : finset ℝ).sort (≤) = 1 := 
sorry

end median_of_set_l723_723982


namespace oranges_to_juice_l723_723434

theorem oranges_to_juice (oranges: ℕ) (juice: ℕ) (h: oranges = 18 ∧ juice = 27): 
  ∃ x, (juice / oranges) = (9 / x) ∧ x = 6 :=
by
  sorry

end oranges_to_juice_l723_723434


namespace find_x_l723_723399

def custom_op (a b : ℤ) : ℤ := 2 * a - b

theorem find_x :
  (∃ x : ℤ, custom_op x (custom_op 2 1) = 3) ↔ 3 :=
by
  sorry

end find_x_l723_723399


namespace range_a_l723_723957

noncomputable def f (x : ℝ) : ℝ := 2 ^ x

noncomputable def A : Set ℝ := { y | 1 < y ∧ y < 4 }

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log x (x - 2 * a) + real.sqrt (a + 1 - x)

noncomputable def B (a : ℝ) : Set ℝ := { x | 2 * a < x ∧ x < a + 1 }

theorem range_a (a : ℝ) (h : a < 1) : B a ⊆ A → 1 / 2 ≤ a ∧ a < 1 :=
begin
  sorry
end

end range_a_l723_723957


namespace length_of_chord_l723_723609

theorem length_of_chord
    (center : ℝ × ℝ) 
    (radius : ℝ) 
    (line : ℝ × ℝ × ℝ) 
    (circle_eq : (x : ℝ) → (y : ℝ) → ((x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2))
    (line_eq : (x : ℝ) → (y : ℝ) → (line.1 * x + line.2 * y + line.3 = 0)) :
    2 * radius * (if h : radius ≠ 0 then (1 - (1 / 2) * ((|line.1 * center.1 + line.2 * center.2 + line.3| / (real.sqrt (line.1 ^ 2 + line.2 ^ 2))) / radius) ^ 2) else 0).sqrt = 2 * real.sqrt 2 :=
by
    sorry

-- Definitions and conditions
def center : ℝ × ℝ := (1, 0)
def radius : ℝ := 2
def line : ℝ × ℝ × ℝ := (1, 1, 1)

noncomputable def circle_eq : (ℝ → ℝ → Prop) :=
    λ x y, (x - center.1) ^ 2 + y ^ 2 = 4

noncomputable def line_eq : (ℝ → ℝ → Prop) :=
    λ x y, x + y + 1 = 0

-- Applying the theorem
#eval (length_of_chord center radius line circle_eq line_eq)

end length_of_chord_l723_723609


namespace number_of_prime_factors_of_30_factorial_l723_723120

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723120


namespace determine_q_l723_723496

theorem determine_q (q : ℕ) (h : 81^10 = 3^q) : q = 40 :=
by
  sorry

end determine_q_l723_723496


namespace factorial_prime_factors_l723_723090

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723090


namespace expand_expression_l723_723902

theorem expand_expression (x y : ℤ) : (x + 7) * (3 * y + 8) = 3 * x * y + 8 * x + 21 * y + 56 :=
by
  sorry

end expand_expression_l723_723902


namespace f_eq_g_l723_723309

noncomputable def K : ℝ → ℝ → ℝ := sorry
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom K_pos_cont : ∀ x y ∈ Icc (0:ℝ) (1:ℝ), 0 < K x y
axiom f_pos_cont : ∀ x ∈ Icc (0:ℝ) (1:ℝ), 0 < f x
axiom g_pos_cont : ∀ x ∈ Icc (0:ℝ) (1:ℝ), 0 < g x

axiom f_integral_eq_g : ∀ x ∈ Icc (0:ℝ) (1:ℝ), ∫ y in 0..1, f y * K x y = g x
axiom g_integral_eq_f : ∀ x ∈ Icc (0:ℝ) (1:ℝ), ∫ y in 0..1, g y * K x y = f x

theorem f_eq_g : ∀ x ∈ Icc (0:ℝ) (1:ℝ), f x = g x := by
  sorry

end f_eq_g_l723_723309


namespace find_m_value_l723_723591

def is_valid_set (A : Set ℝ) : Prop :=
  ∃ (m : ℝ), A = {0, m, m^2 - 3 * m + 2} ∧ 2 ∈ A

theorem find_m_value :
  ∃ (m : ℝ), is_valid_set {0, m, m^2 - 3 * m + 2} ∧ m = 3 :=
begin
  sorry
end

end find_m_value_l723_723591


namespace problem_triangle_incircle_excircle_l723_723311

-- Define the main problem using the conditions
theorem problem_triangle_incircle_excircle
    (A B C D X Y E I M : Point)
    (h_triangle : Triangle A B C)
    (h_D_on_BC : Between D B C)
    (h_AB_plus_BD_eq_AC_plus_CD : dist A B + dist B D = dist A C + dist C D)
    (h_AD_intersects_incircle_at_XY : IncircleIntersectsAtXAndY A B C D X Y)
    (h_X_closer_to_A_than_Y : CloserTo A X Y)
    (h_BC_tangent_incircle_at_E : TangentToIncircleAt B C E)
    (h_I_incenter : Incenter I A B C)
    (h_M_midpoint_BC : Midpoint M B C) :
  (Perpendicular E Y (LineThrough A D)) ∧ (dist X D = 2 * dist I M) :=
by sorry

end problem_triangle_incircle_excircle_l723_723311


namespace count_incorrect_statements_l723_723381

open Real

def statement1 : Prop :=
  Residuals can be used to judge the effectiveness of model fitting

def statement2 : Prop :=
  ∀ (x : ℝ), let y := 3 - 5*x in x >= 0 → y increases by 5 units

def statement3 : Prop :=
  ∀ (x y : ℝ), linear_regression x y → passes_through_mean_point

def statement4 : Prop :=
  ∀ (chi_squared : ℝ), chi_squared = 13.079 → 
  P(chi_squared ≥ 10.828) = 0.001 → 99% confidence relationship

def incorrect_statements : ℕ :=
  (if ¬statement1 then 1 else 0) +
  (if ¬statement2 then 1 else 0) +
  (if ¬statement3 then 1 else 0) +
  (if ¬statement4 then 1 else 0)

theorem count_incorrect_statements :
  incorrect_statements = 1 :=
sorry

end count_incorrect_statements_l723_723381


namespace vacant_seats_l723_723802

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℝ) (vacant_percentage : ℝ) 
  (h_filled : filled_percentage = 75 / 100) 
  (h_total : total_seats = 700) :
  vacant_percentage * total_seats = 175 :=
by
  have h_vacant : vacant_percentage = 1 - filled_percentage,
  {
    rw h_filled,
    norm_num,
  },
  rw [h_vacant, h_total],
  norm_num,
  sorry

end vacant_seats_l723_723802


namespace num_prime_factors_30_fac_l723_723073

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723073


namespace stones_required_l723_723842

-- Definition of the problem conditions
def length_hall_m : ℕ := 45
def breadth_hall_m : ℕ := 25
def length_stone_dm : ℕ := 12
def breadth_stone_dm : ℕ := 7

-- Conversion factor
def meter_to_dm (m : ℕ) : ℕ := m * 10

-- Converted dimensions
def length_hall_dm := meter_to_dm length_hall_m
def breadth_hall_dm := meter_to_dm breadth_hall_m

-- Area calculations
def area_hall_dm2 := length_hall_dm * breadth_hall_dm
def area_stone_dm2 := length_stone_dm * breadth_stone_dm

-- The number of stones required, rounded up
noncomputable def number_of_stones : ℕ := Int.ceil (area_hall_dm2.toRat / area_stone_dm2.toRat)

theorem stones_required : number_of_stones = 1341 := by
  sorry

end stones_required_l723_723842


namespace prime_factors_of_30_l723_723209

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723209


namespace average_salary_of_technicians_l723_723627

theorem average_salary_of_technicians
  (total_workers : ℕ)
  (average_salary_all : ℕ)
  (average_salary_non_technicians : ℕ)
  (num_technicians : ℕ)
  (num_non_technicians : ℕ)
  (h1 : total_workers = 21)
  (h2 : average_salary_all = 8000)
  (h3 : average_salary_non_technicians = 6000)
  (h4 : num_technicians = 7)
  (h5 : num_non_technicians = 14) :
  (average_salary_all * total_workers - average_salary_non_technicians * num_non_technicians) / num_technicians = 12000 :=
by
  sorry

end average_salary_of_technicians_l723_723627


namespace pyramid_volume_l723_723282

-- Define the conditions as given
variables (AB BC CG : ℝ) 
variables (M : ℝ × ℝ × ℝ)

-- Define the midpoint formula for M
def midpoint (A G : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := ((A.1 + G.1) / 2, (A.2 + G.2) / 2, (A.3 + G.3) / 2)

-- Define points and the given conditions
noncomputable def A : ℝ × ℝ × ℝ := (0, 0, 0)
noncomputable def G : ℝ × ℝ × ℝ := (AB, BC, CG)

-- Using the midpoint definition
def M := midpoint A G

theorem pyramid_volume 
  (h1 : AB = 5) 
  (h2 : BC = 3) 
  (h3 : CG = 4)
  (hM : M = (2.5, 1.5, 2)) : 
  ∃ V, V = 2 * sqrt 34 := 
sorry

end pyramid_volume_l723_723282


namespace different_prime_factors_of_factorial_eq_10_l723_723042

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723042


namespace tan_theta_eq_two_implies_expression_l723_723946

theorem tan_theta_eq_two_implies_expression (θ : ℝ) (h : Real.tan θ = 2) :
    (1 - Real.sin (2 * θ)) / (2 * (Real.cos θ)^2) = 1 / 2 :=
by
  -- Define trig identities and given condition
  have h_sin_cos : Real.sin θ = 2 / Real.sqrt 5 ∧ Real.cos θ = 1 / Real.sqrt 5 :=
    sorry -- This will be derived from the given condition h
  
  -- Main proof
  sorry

end tan_theta_eq_two_implies_expression_l723_723946


namespace f_ge_one_seventh_l723_723259

theorem f_ge_one_seventh 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h_sum : a + b + c = 1) :
  let f := λ a b c : ℝ, (a^4 / (a^3 + b^2 + c^2) + b^4 / (b^3 + a^2 + c^2) + c^4 / (c^3 + b^2 + a^2)) in
  f a b c ≥ 1 / 7 :=
sorry

end f_ge_one_seventh_l723_723259


namespace number_of_prime_factors_of_30_factorial_l723_723116

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723116


namespace sampling_interval_correct_l723_723924

theorem sampling_interval_correct :
  ∀ (n k : ℕ), n = 2005 ∧ k = 20 ∧ n % k ≠ 0 →
  let m := n - (n % k) in
  let g := m / k in
  g = 100 :=
begin
  intros n k h,
  rcases h with ⟨hn, hk, hnk⟩,
  let m := n - (n % k),
  have hm : m = 2000,
  { calc m = 2005 - (2005 % 20) : by rw [hn, hk]
      ... = 2005 - 5 : by norm_num [mod_eq_of_lt] },
  let g := m / 20,
  have hg : g = 100,
  { calc g = 2000 / 20 : by rw hm
      ... = 100 : by norm_num },
  exact hg,
end

end sampling_interval_correct_l723_723924


namespace range_is_fixed_points_l723_723938

variable (f : ℕ → ℕ)

axiom functional_eq : ∀ m n, f (m + f n) = f (f m) + f n

theorem range_is_fixed_points :
  {n : ℕ | ∃ m : ℕ, f m = n} = {n : ℕ | f n = n} :=
sorry

end range_is_fixed_points_l723_723938


namespace roots_of_equations_l723_723572

theorem roots_of_equations (a : ℝ) :
  (∃ x : ℝ, x^2 + 4 * a * x - 4 * a + 3 = 0) ∨
  (∃ x : ℝ, x^2 + (a - 1) * x + a^2 = 0) ∨
  (∃ x : ℝ, x^2 + 2 * a * x - 2 * a = 0) ↔ 
  a ≤ -3 / 2 ∨ a ≥ -1 :=
sorry

end roots_of_equations_l723_723572


namespace problem_statement_l723_723489

theorem problem_statement : 
  ((∑ i in Finset.range 2020, (2021 - i) * (1 / (i+1))) /
  (∑ j in Finset.range 2020, 1 / (j + 2))) = 2021 := 
by
  sorry

end problem_statement_l723_723489


namespace ticket_cost_difference_l723_723678

theorem ticket_cost_difference
  (num_adults : ℕ) (num_children : ℕ)
  (cost_adult_ticket : ℕ) (cost_child_ticket : ℕ)
  (h1 : num_adults = 9)
  (h2 : num_children = 7)
  (h3 : cost_adult_ticket = 11)
  (h4 : cost_child_ticket = 7) :
  num_adults * cost_adult_ticket - num_children * cost_child_ticket = 50 := 
by
  sorry

end ticket_cost_difference_l723_723678


namespace linear_equation_must_be_neg2_l723_723605

theorem linear_equation_must_be_neg2 {m : ℝ} (h1 : |m| - 1 = 1) (h2 : m ≠ 2) : m = -2 :=
sorry

end linear_equation_must_be_neg2_l723_723605


namespace number_of_prime_factors_of_30_factorial_l723_723119

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723119


namespace base_7_to_base_10_conversion_l723_723441

theorem base_7_to_base_10_conversion :
  (6 * 7^2 + 5 * 7^1 + 3 * 7^0) = 332 :=
by sorry

end base_7_to_base_10_conversion_l723_723441


namespace equation_of_line_l723_723914

theorem equation_of_line :
  ∃ m : ℝ, ∀ x y : ℝ, (y = m * x - m ∧ (m = 2 ∧ x = 1 ∧ y = 0)) ∧ 
  ∀ x : ℝ, ¬(4 * x^2 - (m * x - m)^2 - 8 * x = 12) → m = 2 → y = 2 * x - 2 :=
by sorry

end equation_of_line_l723_723914


namespace num_prime_factors_of_30_l723_723231

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723231


namespace num_prime_factors_30_fac_l723_723074

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723074


namespace factorial_prime_factors_l723_723087

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723087


namespace num_prime_factors_30_fac_l723_723059

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723059


namespace number_of_pairs_sold_l723_723798

def total_revenue : ℝ := 539
def average_price : ℝ := 9.8
def num_pairs (revenue : ℝ) (price : ℝ) : ℝ := revenue / price

theorem number_of_pairs_sold :
  num_pairs total_revenue average_price = 55 :=
by
  sorry

end number_of_pairs_sold_l723_723798


namespace different_prime_factors_of_factorial_eq_10_l723_723054

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723054


namespace num_prime_factors_30_factorial_l723_723027

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723027


namespace ahmed_goats_is_13_l723_723866

def adam_goats : ℕ := 7

def andrew_goats : ℕ := 2 * adam_goats + 5

def ahmed_goats : ℕ := andrew_goats - 6

theorem ahmed_goats_is_13 : ahmed_goats = 13 :=
by
  sorry

end ahmed_goats_is_13_l723_723866


namespace tan_alpha_add_pi_over_3_l723_723536

theorem tan_alpha_add_pi_over_3 (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3 / 5) 
  (h2 : Real.tan (β - π / 3) = 1 / 4) : 
  Real.tan (α + π / 3) = 7 / 23 := 
by
  sorry

end tan_alpha_add_pi_over_3_l723_723536


namespace ravi_overall_profit_l723_723803

-- Definitions based on the conditions
def cost_price_refrigerator : ℕ := 15000
def cost_price_mobile : ℕ := 8000
def loss_percent_refrigerator : ℝ := 4 / 100
def profit_percent_mobile : ℝ := 10 / 100

-- Definitions of loss and profit calculations based on conditions
def loss_refrigerator : ℝ := cost_price_refrigerator * loss_percent_refrigerator
def selling_price_refrigerator : ℝ := cost_price_refrigerator - loss_refrigerator

def profit_mobile : ℝ := cost_price_mobile * profit_percent_mobile
def selling_price_mobile : ℝ := cost_price_mobile + profit_mobile

-- Overall calculations
def total_cost_price : ℕ := cost_price_refrigerator + cost_price_mobile
def total_selling_price : ℝ := selling_price_refrigerator + selling_price_mobile
def overall_profit : ℝ := total_selling_price - total_cost_price

-- Problem statement
theorem ravi_overall_profit : overall_profit = 200 := by
  sorry

end ravi_overall_profit_l723_723803


namespace chords_intersect_on_line_centers_l723_723890

theorem chords_intersect_on_line_centers
  {O1 O2 A1 A2 B1 B2 : Point}
  (circle1 : ∃ r1, metric.sphere O1 r1)
  (circle2 : ∃ r2, metric.sphere O2 r2)
  (A1_tangent : metric.tangent O1 A1)
  (A2_tangent : metric.tangent O2 A2)
  (B1_tangent : metric.tangent O1 B1)
  (B2_tangent : metric.tangent O2 B2)
  (external_tangent : is_common_tangent A1 A2 circle1 circle2)
  (internal_tangent : is_common_tangent B1 B2 circle1 circle2)
  (no_intersection : metric.disjoint sphere1 sphere2) :
  (∃ C : Point, lies_on C (line_through A1 B1) ∧ lies_on C (line_through A2 B2) ∧ lies_on C (line_through O1 O2)) :=
sorry

end chords_intersect_on_line_centers_l723_723890


namespace prime_factors_of_30_factorial_l723_723096

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723096


namespace number_of_prime_factors_thirty_factorial_l723_723009

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723009


namespace relation_among_abc_l723_723928

noncomputable def pi := Real.pi
noncomputable def a := pi^(1/3)
noncomputable def b := Real.log3 / Real.log pi
noncomputable def c := Real.log (Real.sqrt 3 - 1)

theorem relation_among_abc : c < b ∧ b < a := by
  sorry

end relation_among_abc_l723_723928


namespace tammy_average_speed_second_day_l723_723364

theorem tammy_average_speed_second_day :
  ∃ v t : ℝ, 
  t + (t - 2) + (t + 1) = 20 ∧
  v * t + (v + 0.5) * (t - 2) + (v - 0.5) * (t + 1) = 80 ∧
  (v + 0.5) = 4.575 :=
by 
  sorry

end tammy_average_speed_second_day_l723_723364


namespace different_prime_factors_of_factorial_eq_10_l723_723044

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723044


namespace smaller_solution_quadratic_equation_l723_723778

theorem smaller_solution_quadratic_equation :
  (∀ x : ℝ, x^2 + 7 * x - 30 = 0 → x = -10 ∨ x = 3) → -10 = min (-10) 3 :=
by
  sorry

end smaller_solution_quadratic_equation_l723_723778


namespace num_prime_factors_30_factorial_l723_723240

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723240


namespace prime_factors_of_30_factorial_l723_723171

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723171


namespace num_prime_factors_30_fac_l723_723071

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723071


namespace number_of_prime_factors_of_30_factorial_l723_723124

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723124


namespace cone_ratio_theorem_l723_723266

noncomputable def cone_ratio (r R : ℝ) (central_angle : ℝ) (circumference : ℝ) : Prop :=
  central_angle = (π / 2) ∧
  circumference = 2 * π * r ∧
  circumference = (π / 2) * R ∧
  R = 4 * r ∧
  4 * π * r^2 / (π * r^2) = 4

theorem cone_ratio_theorem (r R : ℝ) (h1 : (π / 2) = π / 2) (h2 : 2 * π * r = 2 * π * r) (h3 : 2 * π * r = (π / 2) * R) :
  4 * π * r^2 / (π * r^2) = 4 :=
by
  have h4 : R = 4 * r, from sorry,
  exact sorry

end cone_ratio_theorem_l723_723266


namespace alcohol_concentration_l723_723822

theorem alcohol_concentration (x : ℝ) (initial_volume : ℝ) (initial_concentration : ℝ) (target_concentration : ℝ) :
  initial_volume = 6 →
  initial_concentration = 0.35 →
  target_concentration = 0.50 →
  (2.1 + x) / (6 + x) = target_concentration →
  x = 1.8 :=
by
  intros h1 h2 h3 h4
  sorry

end alcohol_concentration_l723_723822


namespace value_of_f_x_minus_3_l723_723595

variable (f : ℝ → ℝ)
variable (h : ∀ x : ℝ, f(x) = 3)

theorem value_of_f_x_minus_3 (x : ℝ) : f(x - 3) = 3 :=
by
  sorry

end value_of_f_x_minus_3_l723_723595


namespace num_solution_pairs_l723_723916

theorem num_solution_pairs : 
  (∃ n : ℕ, n = { pair | ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 4 * x + 7 * y = 800 }.to_finset.card) ∧ n = 28 :=
sorry

end num_solution_pairs_l723_723916


namespace largest_n_unique_k_l723_723772

theorem largest_n_unique_k (n : ℕ) (h : ∃ k : ℕ, (9 / 17 : ℚ) < n / (n + k) ∧ n / (n + k) < (8 / 15 : ℚ) ∧ ∀ k' : ℕ, ((9 / 17 : ℚ) < n / (n + k') ∧ n / (n + k') < (8 / 15 : ℚ)) → k' = k) : n = 72 :=
sorry

end largest_n_unique_k_l723_723772


namespace pipe_total_length_l723_723452

def pipe_length (longer_piece : ℝ) (ratio : ℝ) : ℝ :=
  let shorter_piece := longer_piece / ratio
  shorter_piece + longer_piece

theorem pipe_total_length : pipe_length 118 2 = 177 :=
by
  unfold pipe_length
  sorry

end pipe_total_length_l723_723452


namespace volume_of_smaller_cube_l723_723445

noncomputable def volume_of_larger_cube : ℝ := 343
noncomputable def number_of_smaller_cubes : ℝ := 343
noncomputable def surface_area_difference : ℝ := 1764

theorem volume_of_smaller_cube (v_lc : ℝ) (n_sc : ℝ) (sa_diff : ℝ) :
  v_lc = volume_of_larger_cube →
  n_sc = number_of_smaller_cubes →
  sa_diff = surface_area_difference →
  ∃ (v_sc : ℝ), v_sc = 1 :=
by sorry

end volume_of_smaller_cube_l723_723445


namespace crayons_slightly_used_l723_723761

theorem crayons_slightly_used (total_crayons : ℕ) (new_fraction : ℚ) (broken_fraction : ℚ) 
  (htotal : total_crayons = 120) (hnew : new_fraction = 1 / 3) (hbroken : broken_fraction = 20 / 100) :
  let new_crayons := total_crayons * new_fraction
  let broken_crayons := total_crayons * broken_fraction
  let slightly_used_crayons := total_crayons - new_crayons - broken_crayons
  slightly_used_crayons = 56 := 
by
  -- This is where the proof would go
  sorry

end crayons_slightly_used_l723_723761


namespace optimal_juggling_time_l723_723877

def t : ℕ → ℕ
| 1 := 64
| 2 := 55
| 3 := 47
| 4 := 40
| 5 := 33
| 6 := 27
| 7 := 22
| 8 := 18
| 9 := 14
| 10 := 13
| 11 := 12
| 12 := 11
| 13 := 10
| 14 := 9
| 15 := 8
| 16 := 7
| 17 := 6
| 18 := 5
| 19 := 4
| 20 := 3
| 21 := 2
| 22 := 1
| n := 0  -- default to 0 for out of range values

def total_juggling_time (n : ℕ) : ℚ := n * t(n) / 60

theorem optimal_juggling_time 
  (maximizes_time : ∀ m : ℕ, m ≤ 22 → total_juggling_time 5 ≥ total_juggling_time m) : 
  total_juggling_time 5 = 2.75 :=
by
  sorry

end optimal_juggling_time_l723_723877


namespace parabola_directrix_l723_723962

theorem parabola_directrix (p : ℝ) (h : p > 0) : 
  (∃ (A B : ℝ × ℝ), -- A and B, the intersection points
   let x_mid := (A.1 + B.1) / 2 in -- x-coordinate of the midpoint
   y^2 = 2 * p * x ∧ -- parabola equation
   (B.2 - A.2) / (B.1 - A.1) = -1 ∧ -- slope condition
   x_mid = 3) → -- given condition
   ∃ q : ℝ, q = -1 → directrix = x = q := -- equation of the directrix
sorry

end parabola_directrix_l723_723962


namespace prime_factors_30_fac_eq_10_l723_723136

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723136


namespace remaining_mushroom_pieces_l723_723926

theorem remaining_mushroom_pieces 
  (mushrooms : ℕ) 
  (pieces_per_mushroom : ℕ) 
  (pieces_used_by_kenny : ℕ) 
  (pieces_used_by_karla : ℕ) 
  (mushrooms_cut : mushrooms = 22) 
  (pieces_per_mushroom_def : pieces_per_mushroom = 4) 
  (kenny_pieces_def : pieces_used_by_kenny = 38) 
  (karla_pieces_def : pieces_used_by_karla = 42) : 
  (mushrooms * pieces_per_mushroom - (pieces_used_by_kenny + pieces_used_by_karla)) = 8 := 
by 
  sorry

end remaining_mushroom_pieces_l723_723926


namespace factorial_30_prime_count_l723_723187

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723187


namespace number_of_prime_factors_thirty_factorial_l723_723021

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723021


namespace students_like_apple_and_chocolate_not_blueberry_l723_723276

variables (n A C B D : ℕ)

theorem students_like_apple_and_chocolate_not_blueberry
  (h1 : n = 50)
  (h2 : A = 25)
  (h3 : C = 20)
  (h4 : B = 5)
  (h5 : D = 15) :
  ∃ (x : ℕ), x = 10 ∧ x = n - D - (A + C - 2 * x) ∧ 0 ≤ 2 * x - A - C + B :=
sorry

end students_like_apple_and_chocolate_not_blueberry_l723_723276


namespace sqrt_ab_eq_18_l723_723658

noncomputable def a := Real.log 9 / Real.log 4
noncomputable def b := 108 * (Real.log 8 / Real.log 3)

theorem sqrt_ab_eq_18 : Real.sqrt (a * b) = 18 := by
  sorry

end sqrt_ab_eq_18_l723_723658


namespace compound_interest_years_l723_723397

-- Define the conditions as given in the problem
def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * R * T / 100

def compound_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * ((1 + R / 100)^T - 1)

def conditions (sum_simple : ℝ) (years_simple : ℝ) (rate_simple : ℝ) 
               (sum_compound : ℝ) (rate_compound : ℝ) (years_compound : ℝ): Prop :=
  let SI := simple_interest sum_simple rate_simple years_simple in
  SI * 2 = compound_interest sum_compound rate_compound years_compound

-- Question to prove
theorem compound_interest_years: 
  conditions 1400 3 10 4000 10 2 := 
by
  sorry

end compound_interest_years_l723_723397


namespace log_exp_simplification_l723_723818

/-- Using the properties and operation rules of logarithms and exponents, prove the given equation. -/
theorem log_exp_simplification :
  2 * log 10 5 * 2 * log 10 2 + Real.exp 1 * Real.log 3 = 5 :=
by
  sorry

end log_exp_simplification_l723_723818


namespace boxes_needed_l723_723715

theorem boxes_needed (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) (h1 : total_muffins = 95) (h2 : muffins_per_box = 5) (h3 : available_boxes = 10) : 
  total_muffins - (available_boxes * muffins_per_box) / muffins_per_box = 9 :=
by
  sorry

end boxes_needed_l723_723715


namespace closest_multiple_of_18_2021_l723_723783

def is_multiple_of (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def closest_multiple_of (n k : ℕ) : ℕ :=
if (n % k) * 2 < k then n - (n % k) else n + (k - n % k)

theorem closest_multiple_of_18_2021 :
  closest_multiple_of 2021 18 = 2016 := by
    sorry

end closest_multiple_of_18_2021_l723_723783


namespace price_of_chocolate_orange_l723_723677

/-- Prove that given candy bars are sold for $5 each, Nick sold out 20 chocolate oranges, 
    he needs to raise $1000, and he sells 160 candy bars, 
    the price of each chocolate orange is $10. -/
theorem price_of_chocolate_orange (candy_bar_price : ℕ) (choc_oranges_sold : ℕ) 
 (fundraising_goal : ℕ) (candy_bars_sold : ℕ) (total_candy_income : ℕ) (price_choc_orange : ℕ) :
   candy_bar_price = 5 →
   choc_oranges_sold = 20 →
   fundraising_goal = 1000 →
   candy_bars_sold = 160 →
   total_candy_income = candy_bars_sold * candy_bar_price →
   (choc_oranges_sold * price_choc_orange) + total_candy_income = fundraising_goal →
   price_choc_orange = 10 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  simp at h6
  -- remainder of proof
  sorry

end price_of_chocolate_orange_l723_723677


namespace number_of_valid_six_digit_integers_l723_723392

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def alternating_odd_even (n : ℕ) : Prop :=
  let digits := [n / 100000 % 10, n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  ∧ no_repeats digits
  ∧ ((is_odd (digits.head!) ∨ is_even (digits.head!)) ∧ (∀ i, i < 5 → (is_odd (digits.nth_le i sorry) ↔ is_even (digits.nth_le (i + 1) sorry))))

def no_repeats (l : List ℕ) : Prop :=
  ∀ i j, i < l.length → j < l.length → i ≠ j → l.nth_le i sorry ≠ l.nth_le j sorry

def six_digits_from_0_to_5 (n : ℕ) : Prop :=
  let digits := [n / 100000 % 10, n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  list.all (λ d, d ∈ [0, 1, 2, 3, 4, 5]) digits

theorem number_of_valid_six_digit_integers : 
  ∃ count : ℕ, count = 60 ∧
  count = (List.range 1000000).count (λ n, six_digits_from_0_to_5 n ∧ alternating_odd_even n) :=
by {
  sorry
}

end number_of_valid_six_digit_integers_l723_723392


namespace num_prime_factors_30_factorial_l723_723245

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723245


namespace Jake_has_8_peaches_l723_723304

variable (Steven Jill Jake : ℕ)

-- Conditions
axiom h1 : Steven = 15
axiom h2 : Steven = Jill + 14
axiom h3 : Jake = Steven - 7

-- Goal
theorem Jake_has_8_peaches : Jake = 8 := by
  sorry

end Jake_has_8_peaches_l723_723304


namespace apple_weight_l723_723867

variable (weight_small_box weight_large_box total_weight_kg : ℕ)
variable (apples_per_small_box small_boxes_per_large_box : ℕ)
variable (total_weight_g : ℕ := total_weight_kg * 1000)
variable (total_weight_packaging_g : ℕ := weight_large_box + (weight_small_box * small_boxes_per_large_box))
variable (weight_apples_g : ℕ := total_weight_g - total_weight_packaging_g)
variable (total_apples : ℕ := apples_per_small_box * small_boxes_per_large_box)
variable (weight_one_apple : ℕ := weight_apples_g / total_apples)

theorem apple_weight :
  (weight_small_box = 220) →
  (weight_large_box = 250) →
  (total_weight_kg = 13.3 * 1000) →
  (apples_per_small_box = 6) →
  (small_boxes_per_large_box = 9) →
  weight_one_apple = 205 :=
by
  intros,
  sorry

end apple_weight_l723_723867


namespace probability_of_sum_equals_age_l723_723355

noncomputable def probability_sum_equals_age
  (age : ℕ) (coin : ℕ → ℕ) (die : ℕ → ℕ) : ℚ :=
(coin 15 * die 1)

def fair_coin : ℕ → ℚ
| 5 := 1 / 2
| 15 := 1 / 2
| _ := 0

def standard_die : ℕ → ℚ
| n := if 1 ≤ n ∧ n ≤ 6 then 1 / 6 else 0

theorem probability_of_sum_equals_age :
  probability_sum_equals_age 16 fair_coin standard_die = 1 / 12 := by
  sorry

end probability_of_sum_equals_age_l723_723355


namespace num_prime_factors_of_30_l723_723226

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723226


namespace quadratic_m_value_l723_723989

theorem quadratic_m_value (m : ℕ) :
  (∃ x : ℝ, x^(m + 1) - (m + 1) * x - 2 = 0) →
  m + 1 = 2 →
  m = 1 :=
by {
  sorry
}

end quadratic_m_value_l723_723989


namespace exists_closed_self_intersecting_line_l723_723497

-- Definition of a triangular prism
structure TriangularPrism :=
  (A B C A' B' C' : Point)

-- Condition for the closed self-intersecting polygonal line
def closed_self_intersecting_line_exists (prism : TriangularPrism) : Prop :=
∃ path : List Segment, 
  isClosedPath path ∧ 
  selfIntersectsAtMidpoints path ∧ 
  allEdgesIntersectExactlyOnce path

-- Main theorem statement
theorem exists_closed_self_intersecting_line (prism : TriangularPrism) : closed_self_intersecting_line_exists prism :=
sorry

end exists_closed_self_intersecting_line_l723_723497


namespace range_of_a_l723_723436

def f (x a : ℝ) := |x - 2| + |x + a|

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 3) → a ≤ -5 ∨ a ≥ 1 :=
  sorry

end range_of_a_l723_723436


namespace inequality_solution_set_system_of_inequalities_solution_set_l723_723750

theorem inequality_solution_set (x : ℝ) (h : 3 * x - 5 > 5 * x + 3) : x < -4 :=
by sorry

theorem system_of_inequalities_solution_set (x : ℤ) 
  (h₁ : x - 1 ≥ 1 - x) 
  (h₂ : x + 8 > 4 * x - 1) : x = 1 ∨ x = 2 :=
by sorry

end inequality_solution_set_system_of_inequalities_solution_set_l723_723750


namespace find_a8_l723_723543

-- Define the geometric sequence condition
def is_geometric_sequence {α : Type*} [linear_ordered_field α] (a : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, a (n + 1) = a n * r

-- Given conditions: a_4 = 7, a_6 = 21 in a geometric sequence
variables {α : Type*} [linear_ordered_field α]
variables (a : ℕ → α)
variable (h_geom : is_geometric_sequence a)
variable (h_a4 : a 4 = 7)
variable (h_a6 : a 6 = 21)

theorem find_a8 : a 8 = 63 := by
  sorry

end find_a8_l723_723543


namespace anne_distance_l723_723262

-- Definitions based on conditions
def Time : ℕ := 5
def Speed : ℕ := 4
def Distance : ℕ := Speed * Time

-- Proof statement
theorem anne_distance : Distance = 20 := by
  sorry

end anne_distance_l723_723262


namespace num_prime_factors_30_factorial_l723_723254

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723254


namespace factorial_30_prime_count_l723_723201

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723201


namespace sum_of_decimals_as_fraction_l723_723905

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

end sum_of_decimals_as_fraction_l723_723905


namespace largest_n_unique_k_l723_723774

theorem largest_n_unique_k (n k : ℕ) :
  (frac9_17_lt_frac n (n + k) ∧ frac n (n + k) lt frac8_15) 
  ∧ (∀ (n1 k1 : ℕ), frac9_17_lt_frac n1 (n1 + k1) ∧ frac n1 (n1 + k1) lt frac8_15 
  → (n1 ≤ 136 
  ∧ ((n1 = 136) → (k1 = unique_k))))
  :=
sorry

def frac9_17_lt_frac (a b : ℕ) : Prop := 
  (9:ℚ) / 17 < (a : ℚ) / b

def frac (a b : ℕ) : ℚ :=
  (a : ℚ) / b

def frac8_15 := 
  (8:ℚ) / 15

def unique_k : ℕ :=
  119

end largest_n_unique_k_l723_723774


namespace area_shaded_region_l723_723637

noncomputable def radius_large_circle : ℝ := 6
noncomputable def radius_small_circle : ℝ := radius_large_circle / 2

noncomputable def area_circle (r : ℝ) : ℝ := π * r ^ 2

theorem area_shaded_region : 
  area_circle radius_large_circle - 2 * area_circle radius_small_circle = 18 * π := by
sorry

end area_shaded_region_l723_723637


namespace smallest_integer_greater_than_one_with_inverse_mod_1155_l723_723780

theorem smallest_integer_greater_than_one_with_inverse_mod_1155 :
  ∃ n : ℕ, n > 1 ∧ (∀ m : ℕ, m > 1 → (m % 1155 ≠ 0) → n ≤ m) ∧ (nat.gcd n 1155 = 1) ∧ n = 2 :=
sorry

end smallest_integer_greater_than_one_with_inverse_mod_1155_l723_723780


namespace prime_factors_30_fac_eq_10_l723_723134

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723134


namespace reflect_point_B_l723_723455

noncomputable def find_point_B : 
  (A C : ℝ × ℝ × ℝ) → (plane_normal : ℝ × ℝ × ℝ) → (d : ℝ) → 
  (B : ℝ × ℝ × ℝ) :=
λ A C (plane_normal : ℝ × ℝ × ℝ) d,
  let D := (2 * (A.1 + plane_normal.1) - A.1, 2 * (A.2 + plane_normal.2) - A.2, 2 * (A.3 + plane_normal.3) - A.3) in
  B

theorem reflect_point_B :
  ∀ (A : ℝ × ℝ × ℝ) (C : ℝ × ℝ × ℝ) (plane_normal : ℝ × ℝ × ℝ) (d : ℝ),
  A = (-2, 10, 12) →
  plane_normal = (1, 1, 1) →
  d = 14 →
  C = (4, 3, 7) →
  find_point_B A C plane_normal d = (4, 3, 7) :=
begin
  intros A C plane_normal d hA hnormal hd hC,
  rw [hA, hnormal, hC],
  sorry -- Proof to be completed
end

end reflect_point_B_l723_723455


namespace custom_mul_of_two_and_neg_three_l723_723993

-- Define the custom operation "*"
def custom.mul (a b : Int) : Int := a * b

-- The theorem to prove that 2 * (-3) using custom.mul equals -6
theorem custom_mul_of_two_and_neg_three : custom.mul 2 (-3) = -6 :=
by
  -- This is where the proof would go
  sorry

end custom_mul_of_two_and_neg_three_l723_723993


namespace prime_factors_of_30_factorial_l723_723106

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723106


namespace problem_equivalent_proof_l723_723910

noncomputable def sqrt (x : ℝ) : ℝ := x.sqrt -- Define the sqrt function for real numbers

theorem problem_equivalent_proof (x : ℝ) :
  sqrt ((3 + sqrt 8) ^ x) + sqrt ((3 - sqrt 8) ^ x) = 6 ↔ x = 2 ∨ x = -2 :=
begin
  sorry -- This is where the proof would be, omitted as instructed
end

end problem_equivalent_proof_l723_723910


namespace number_of_prime_factors_thirty_factorial_l723_723019

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723019


namespace max_triangles_with_area_one_l723_723661

theorem max_triangles_with_area_one (n : ℕ) (S : finset (euclidean_space ℝ (fin 2)))
  (h_pos : 0 < n)
  (h_card : S.card = n)
  (h_not_collinear : ∀ {a b c : euclidean_space ℝ (fin 2)},
    a ∈ S → b ∈ S → c ∈ S → (a ≠ b ∧ b ≠ c ∧ a ≠ c) → ¬collinear ℝ ({a, b, c} : set (euclidean_space ℝ (fin 2)))) :
  (∃ T : ℕ, T = finset.filter (λ t, let ⟨x, y, z⟩ := t in triangle_area x y z = 1) 
    (S.powerset.filter (λ t, t.card = 3)).card → T ≤ 2 * n * (n - 1) / 3) :=
sorry

end max_triangles_with_area_one_l723_723661


namespace expression_is_integer_iff_divisible_l723_723670

theorem expression_is_integer_iff_divisible (k n : ℤ) (h1 : 1 ≤ k) (h2 : k < n) :
  ∃ m : ℤ, n = m * (k + 2) ↔ (∃ C : ℤ, (3 * n - 4 * k + 2) / (k + 2) * C = (3 * n - 4 * k + 2) / (k + 2)) :=
sorry

end expression_is_integer_iff_divisible_l723_723670


namespace number_of_good_sets_is_three_l723_723590

def good_set (C : set (ℝ × ℝ)) : Prop :=
  ∀ (x1 y1 : ℝ), (x1, y1) ∈ C → ∃ (x2 y2 : ℝ), (x2, y2) ∈ C ∧ x1 * x2 + y1 * y2 = 0

def C1 : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}
def C2 : set (ℝ × ℝ) := {p | p.1^2 - p.2^2 = 9}
def C3 : set (ℝ × ℝ) := {p | 2 * p.1^2 + p.2^2 = 9}
def C4 : set (ℝ × ℝ) := {p | p.1^2 + p.2 = 9}

theorem number_of_good_sets_is_three :
  (if good_set C1 then 1 else 0) + 
  (if good_set C2 then 1 else 0) + 
  (if good_set C3 then 1 else 0) + 
  (if good_set C4 then 1 else 0) = 3 := 
sorry

end number_of_good_sets_is_three_l723_723590


namespace num_prime_factors_30_fac_l723_723066

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723066


namespace range_of_a_l723_723751

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ a < -4 ∨ a > 4 :=
by
  sorry

end range_of_a_l723_723751


namespace conjugate_in_third_quadrant_l723_723291

def complex_quadrant (z : ℂ) : ℕ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3 
  else if z.re > 0 ∧ z.im < 0 then 4
  else 0

theorem conjugate_in_third_quadrant (z : ℂ) (h : z = (i / (1 - i))) :
  complex_quadrant (z.conjugate) = 3 := by
  sorry

end conjugate_in_third_quadrant_l723_723291


namespace determinant_of_matrix_l723_723502

theorem determinant_of_matrix (a b : ℝ) : 
  det ![![1, a, b], ![1, a + b, b], ![1, a, a + b]] = ab + 2 * b^2 :=
by
  sorry

end determinant_of_matrix_l723_723502


namespace prime_factors_of_30_factorial_l723_723109

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723109


namespace floor_abs_sum_eq_510_l723_723707

theorem floor_abs_sum_eq_510 (x : Fin 100 → ℝ)
  (h : ∀ n : Fin 100, x n + 10 * n.val.to_nat + 10 = (∑ i in Finset.range 100, x ⟨i, sorry⟩) + 1010) :
  ⌊| ∑ i in Finset.range 100, x ⟨i, sorry⟩ |⌋ = 510 :=
by
  sorry

end floor_abs_sum_eq_510_l723_723707


namespace route_comparison_l723_723624

-- Definitions
def distance (P Z C : Type) : Type := ℝ

variables {P Z C : Type} -- P: Park, Z: Zoo, C: Circus
variables (x y C : ℝ)     -- x: direct distance from Park to Zoo, y: direct distance from Circus to Zoo, C: total circumference

-- Conditions
axiom h1 : x + 3 * x = C -- distance from Park to Zoo via Circus is three times longer than not via Circus
axiom h2 : y = (C - x) / 2 -- distance from Circus to Zoo directly is y
axiom h3 : 2 * y = C - x -- distance from Circus to Zoo via Park is twice as short as not via Park

-- Proof statement
theorem route_comparison (P Z C : Type) (x y C : ℝ) (h1 : x + 3 * x = C) (h2 : y = (C - x) / 2) (h3 : 2 * y = C - x) :
  let direct_route := x
  let via_zoo_route := 3 * x - x
  via_zoo_route = 11 * direct_route := 
sorry

end route_comparison_l723_723624


namespace solve_equation_l723_723911

noncomputable def a := 3 + Real.sqrt 8
noncomputable def b := 3 - Real.sqrt 8

theorem solve_equation (x : ℝ) :
  (Real.sqrt (a^x) + Real.sqrt (b^x) = 6) ↔ (x = 2 ∨ x = -2) := 
  by
  sorry

end solve_equation_l723_723911


namespace prime_factors_of_30_factorial_l723_723104

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723104


namespace grey_cats_in_house_l723_723997

/-- This proof shows that given 16 cats in a house,
    2 of them white, and 25% of them black,
    there are 10 grey cats in the house. -/ 

theorem grey_cats_in_house :
  ∀ (total_cats white_cats : ℕ) (percentage_black : ℝ),
    total_cats = 16 →
    white_cats = 2 →
    percentage_black = 0.25 →
    ∃ (grey_cats : ℕ), grey_cats = total_cats - (white_cats + nat.floor (percentage_black * total_cats)) ∧ grey_cats = 10 :=
by
  intros total_cats white_cats percentage_black ht hw hp
  use 10
  rw [ht, hw, hp]
  have hblack : nat.floor (0.25 * 16) = 4 := by rfl
  rw hblack
  simp
  sorry

end grey_cats_in_house_l723_723997


namespace midpoint_trajectory_tangent_conditions_l723_723562

-- (1) Prove the trajectory of the midpoint M is as described
theorem midpoint_trajectory :
  (∀ (A : ℝ × ℝ), (A.1 - 7)^2 + A.2^2 = 16 → 
  (let M := ((A.1 - 1) / 2, A.2 / 2) in (M.1 - 3)^2 + M.2^2 = 4)) :=
by sorry

-- (2) Prove the conditions for tangency and find the corresponding tangent lines
theorem tangent_conditions (a : ℝ) :
  (∀ (a : ℝ), (let C := (2, a) in 
  (∃ l : ℝ → ℝ, (∀ x, l x = a / 2 * x ∨ l x = 2 + a - x) ∧ 
    (∀ t : ℝ, distance ((t, l t)) (7, 0) = 4) → 
  a = 4 * Real.sqrt 5 / 5 ∨ a = -4 * Real.sqrt 5 / 5 ∨ 
  a = 1 + 2 * Real.sqrt 2 ∨ a = 1 - 2 * Real.sqrt 2 ∧ 
  (∃ l : ℝ → ℝ, l x = 2 + a -x ∨ l x = a / 2 * x)))) :=
by sorry

end midpoint_trajectory_tangent_conditions_l723_723562


namespace sonya_falls_6_l723_723701

def number_of_falls_steven : ℕ := 3
def number_of_falls_stephanie : ℕ := number_of_falls_steven + 13
def number_of_falls_sonya : ℕ := (number_of_falls_stephanie / 2) - 2

theorem sonya_falls_6 : number_of_falls_sonya = 6 := 
by
  -- The actual proof is to be filled in here
  sorry

end sonya_falls_6_l723_723701


namespace reasoning_is_inductive_l723_723788

-- Define the conditions as given in the problem
def cond_A := ∀ (A B P : ℝ) (a : ℝ), (|P - A| + |P - B| = 2 * a ∧ 2 * a > |A - B|) → 
  ∃ (e : set ℝ), P ∈ e

def cond_B := ∀ (n : ℕ), ∃ S₁ S₂ S₃ Sₙ : ℕ, 
  (S₁ = 1) ∧
  (∀ n, Sₙ = Sₙ + (3*n - 1))

def cond_C := ∀ (x y r a b : ℝ), (x^2 + y^2 = r^2 → S = π * r^2) → 
  (x^2 / a^2 + y^2 / b^2 = 1 → S = π * a * b)

def cond_D := ∃ P Q : ℝ, 
  (P = "fish buoyancy") → (Q = "create submarines")

-- Define the problem that option B is inductive reasoning
def is_inductive_reasoning (B : Prop) : Prop := B

-- The theorem to be proved
theorem reasoning_is_inductive : 
  is_inductive_reasoning cond_B :=
sorry

end reasoning_is_inductive_l723_723788


namespace num_prime_factors_30_factorial_l723_723253

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723253


namespace harmonic_sum_ratio_l723_723487

theorem harmonic_sum_ratio :
  (∑ k in Finset.range (2020 + 1), (2021 - k) / k) /
  (∑ k in Finset.range (2021 - 1), 1 / (k + 2)) = 2021 :=
by
  sorry

end harmonic_sum_ratio_l723_723487


namespace prime_factors_of_30_l723_723215

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723215


namespace factorial_30_prime_count_l723_723200

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723200


namespace max_value_f_l723_723525

def f (x : ℝ) : ℝ := min (2 - x) x

theorem max_value_f : ∃ x : ℝ, x ∈ set.Icc 1 1 ∧ f x = 1 :=
by sorry

end max_value_f_l723_723525


namespace tangent_circle_angles_l723_723403

/--
Through a point \(A\) outside a circle, two tangents \(A T_{1}\) and \(A T_{2}\) are drawn to the circle.
From an arbitrary point \(M\) on the circle, three rays \(M M_{1}, M T_{1}, M T_{2}\) are drawn.
The angles \(\varepsilon, \varphi, \psi\) are formed between these rays and the tangent to the circle at point \(M\).
Prove that:
\[ \operatorname{ctg} \varepsilon = \frac{1}{2} (\operatorname{ctg} \varphi + \operatorname{ctg} \psi) \].
-/
theorem tangent_circle_angles (A M T₁ T₂ : Point) (ε φ ψ : ℝ) :
  external_tangent A T₁ ∧ external_tangent A T₂ ∧
  on_circle M ∧
  rays_intersect_circle M M₁ ∧ rays_intersect_circle M T₁ ∧ rays_intersect_circle M T₂ ∧
  angles_formed ε φ ψ →
  cot ε = (1 / 2) * (cot φ + cot ψ) :=
by
  sorry

end tangent_circle_angles_l723_723403


namespace largest_angle_in_convex_pentagon_l723_723743

theorem largest_angle_in_convex_pentagon (x : ℕ) (h : (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 540) : 
  x + 2 = 110 :=
by
  sorry

end largest_angle_in_convex_pentagon_l723_723743


namespace range_of_area_of_acute_triangle_ABC_l723_723628

theorem range_of_area_of_acute_triangle_ABC
  (A B C : ℝ)
  (h_angle_A : A = π / 6)
  (h_BC : B = 1)
  (h_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) :
  let S_ABC := (1 / 2) * B * (cos C + sqrt 3 * sin C) * sin A in
  S_ABC ∈ set.Ioc (sqrt 3 / 4) (1 / 2 + sqrt 3 / 4) := 
sorry

end range_of_area_of_acute_triangle_ABC_l723_723628


namespace cubs_win_world_series_prob_l723_723710

-- Define the probabilities for Cubs and Red Sox
def prob_win_cubs : ℚ := 4/7
def prob_win_red_sox : ℚ := 3/7

-- Define the total number of wins required to win the series
def required_wins : ℕ := 5

-- Define the probability that the Cubs will win the World Series
noncomputable def prob_cubs_win_world_series : ℚ :=
  ∑ k in Finset.range 5, (Nat.choose (4 + k) k) * (prob_win_cubs ^ required_wins) * (prob_win_red_sox ^ k)

-- The statement we need to prove
theorem cubs_win_world_series_prob : (prob_cubs_win_world_series * 100).round = 72 := by
  sorry

end cubs_win_world_series_prob_l723_723710


namespace arithmetic_expression_eval_l723_723881

theorem arithmetic_expression_eval : 
  5 * 7.5 + 2 * 12 + 8.5 * 4 + 7 * 6 = 137.5 :=
by
  sorry

end arithmetic_expression_eval_l723_723881


namespace triangle_side_length_l723_723439

theorem triangle_side_length 
  (r : ℝ)                    -- radius of the inscribed circle
  (h_cos_ABC : ℝ)            -- cosine of angle ABC
  (h_midline : Bool)         -- the circle touches the midline parallel to AC
  (h_r : r = 1)              -- given radius is 1
  (h_cos : h_cos_ABC = 0.8)  -- given cos(ABC) = 0.8
  (h_touch : h_midline = true)  -- given that circle touches the midline
  : AC = 3 := 
sorry

end triangle_side_length_l723_723439


namespace count_special_quadruples_is_23_l723_723533

open Finset

noncomputable def count_special_quadruples : ℕ :=
  let s := range 11 \ {0} -- The set {1, 2, 3, ..., 10}
  (s.powerset.filter (λ t, t.card = 4 ∧ 
    (∃ a ∈ t, ∃ b ∈ t, ∃ c ∈ t, ∃ d ∈ t, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ 
    (a = 3 * (b + c + d) ∨ b = 3 * (a + c + d) ∨ c = 3 * (a + b + d) ∨ d = 3 * (a + b + c))))).card

theorem count_special_quadruples_is_23 : count_special_quadruples = 23 := 
by sorry

end count_special_quadruples_is_23_l723_723533


namespace inequality_has_nonempty_solution_set_l723_723981

theorem inequality_has_nonempty_solution_set (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a ∈ Ioi 1 :=
sorry

end inequality_has_nonempty_solution_set_l723_723981


namespace problem_angle_of_inclination_of_line_through_focus_l723_723848

theorem problem_angle_of_inclination_of_line_through_focus :
  ∀ (l : ℝ → ℝ) (A B : ℝ × ℝ), (∀ x, l(x) = l(1 + (x-1))) → A = (x₁, l(x₁)) ∧ B = (x₂, l(x₂)) →
  y^2 = 4 * x → (1 / |(A - (1, 0))|) - (1 / |(B - (1, 0))|) = 1/2 →
  0 < θ ∧ θ < π/2 →
  θ = π/3 :=
by
  sorry

end problem_angle_of_inclination_of_line_through_focus_l723_723848


namespace change_of_opinion_is_40_l723_723332

noncomputable def percentage_first_semester := { loved := 30%, neutral := 40%, not_loved := 30% }
noncomputable def percentage_second_semester := { loved := 40%, neutral := 20%, not_loved := 40% }
noncomputable def percentage_third_semester := { loved := 50%, neutral := 20%, not_loved := 30% }

theorem change_of_opinion_is_40 :
  let total_change := 
    percentage_first_semester.neutral - percentage_third_semester.neutral +
    percentage_third_semester.loved - percentage_first_semester.loved in
  total_change = 40% :=
by
  sorry

end change_of_opinion_is_40_l723_723332


namespace quadrilateral_area_l723_723995

theorem quadrilateral_area (a b x : ℝ)
  (h1: ∀ (y z : ℝ), y^2 + z^2 = a^2 ∧ (x + y)^2 + (x + z)^2 = b^2)
  (hx_perp: ∀ (p q : ℝ), x * q = 0 ∧ x * p = 0) :
  S = (1 / 4) * |b^2 - a^2| :=
by
  sorry

end quadrilateral_area_l723_723995


namespace painted_area_l723_723505

-- Definitions of given conditions
def sides (a b c : ℝ) : Prop := a = 30 ∧ b = 40 ∧ c = 50
def speed : ℝ := 1 / 6
def time : ℝ := 60  -- 1 minute in seconds

-- Distance covered in 1 minute
def distance (speed time : ℝ) : ℝ := speed * time

-- Area of a right triangle
def right_triangle_area (a b : ℝ) : ℝ := (1 / 2) * a * b

-- Radius of the inscribed circle
def inscribed_circle_radius (a b c : ℝ) : ℝ := (a + b - c) / 2

-- Perimeter of the triangle
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Total area calculation
def total_area (triangle_area rectangle_area sector_area : ℝ) : ℝ :=
  triangle_area + rectangle_area + sector_area

-- Main proof statement
theorem painted_area :
  ∀ (a b c : ℝ),
    sides a b c →
    let d := distance speed time in
    let triangle_area := right_triangle_area a b in
    let rectangle_area := d * perimeter a b c in
    let sector_area := 3 * (1/4) * (Real.pi * d^2) in
    round (total_area triangle_area rectangle_area sector_area) = 2114 :=
by
  intros a b c h
  have d_def : d = distance speed time := rfl
  have d_val : d = 10 := calc
    distance speed time = (1/6) * 60 := rfl
    ... = 10 := by norm_num
  have triangle_area_def : triangle_area = right_triangle_area a b := rfl
  have triangle_area_val : triangle_area = 600 := calc
    (1/2) * a * b = (1/2) * 30 * 40 := by { rw h.1, rw h.2.1 }
    ... = 600 := by norm_num
  have rectangle_area_def : rectangle_area = d * perimeter a b c := rfl
  have rectangle_area_val : rectangle_area = 1200 := calc
    10 * perimeter a b c = 10 * (a + b + c) := by { rw d_val }
    ... = 10 * (30 + 40 + 50) := by { rw h.1, rw h.2.1, rw h.2.2 }
    ... = 1200 := by norm_num
  have sector_area_def : sector_area = 3 * (1/4) * (Real.pi * d^2) := rfl
  have sector_area_val : sector_area = 100 * Real.pi := by
    calc
      3 * (1/4) * (Real.pi * d^2) = 3 * (1/4) * (Real.pi * 10^2) := by { rw d_val }
      ... = 3 * (1/4) * (Real.pi * 100) := rfl
      ... = 3 * 25 * Real.pi := by norm_num
      ... = 75 * Real.pi := by norm_num
  rw [total_area_def, triangle_area_def, rectangle_area_def, sector_area_def, triangle_area_val, rectangle_area_val, sector_area_val]
  norm_cast
  rw [add_assoc, add_assoc, add_comm (100 * Real.pi), add_assoc, add_assoc, add_comm (Real.pi * 100), add_assoc]
  have h_pi : Real.pi * 100 ≈ 314 := by norm_num1
  ring_exp
  sorry

end painted_area_l723_723505


namespace Dara_wait_years_l723_723741

/-- Given:
1. The minimum age required to be employed at the company is 25 years.
2. Dara will be half the age of Jane in six years.
3. Jane is currently 28 years old.
4. The company’s minimum age requirement increases by 1 year every 5 years.
5. Tom, who is 10 years older than Jane, joined the company when the minimum age requirement was 24 years.
6. Dara needs to complete a 3-year internship at age 22.
7. Dara needs to take a 2-year training at age 24.

Prove: Dara will have to wait 19 more years before she reaches the adjusted minimum age required to be employed at the company.
-/
theorem Dara_wait_years : 
  ∀ (min_age : ℕ) 
    (is_half_age : ∀ (jane_age dara_future_age : ℕ), (jane_age = 28 + 6) → (dara_future_age = jane_age / 2)) 
    (min_increase : ℕ → ℕ)
    (tom_age : ℕ = 38)
    (dara_internship : ℕ = 3)
    (dara_training : ℕ = 2),
  min_age = 25 →
  (min_increase 24 = 13 / 5) →
  (26 + dara_training + 1 - 27 + (15 / 5) - dara_internship)  = 19 :=
by 
  intros min_age is_half_age min_increase tom_age dara_internship dara_training
  apologize -- sorry, skip the proof

end Dara_wait_years_l723_723741


namespace find_value_of_f_neg_3_over_2_l723_723316

noncomputable def f : ℝ → ℝ := sorry

theorem find_value_of_f_neg_3_over_2 (h1 : ∀ x : ℝ, f (-x) = -f x) 
    (h2 : ∀ x : ℝ, f (x + 3/2) = -f x) : 
    f (- 3 / 2) = 0 := 
sorry

end find_value_of_f_neg_3_over_2_l723_723316


namespace math_problem_domain_monotonicity_and_max_value_and_range_of_a_l723_723574

-- Definitions of the functions
def f (x : ℝ) : ℝ := log 4 (2 * x + 3 - x ^ 2)
def g (a x : ℝ) : ℝ := log 4 ((a + 2) * x + 4)

-- Statement of the theorem
theorem math_problem_domain_monotonicity_and_max_value_and_range_of_a 
  : (∀ x, -1 < x ∧ x < 3 → (0 < 2 * x + 3 - x ^ 2)) ∧
    (∀ x, -1 < x ∧ x ≤ 1 → (2 * x + 3 - x ^ 2 is increasing)) ∧
    (∀ x, 1 ≤ x ∧ x < 3 → (2 * x + 3 - x ^ 2 is decreasing)) ∧
    (∀ x, 0 < x ∧ x < 3 → f(x) ≤ g(a, x)) ∧
    (∀ x, x = 1 → f(x) = 1) ∧
    (∀ a, ∀ x : ℝ, 0 < x ∧ x < 3 → (2 * x + 3 - x ^ 2 ≤ (a + 2) * x + 4) → a ≥ -2) :=
sorry

end math_problem_domain_monotonicity_and_max_value_and_range_of_a_l723_723574


namespace quadratic_condition_l723_723600

theorem quadratic_condition (m : ℝ) (h1 : m^2 - 2 = 2) (h2 : m + 2 ≠ 0) : m = 2 :=
by
  sorry

end quadratic_condition_l723_723600


namespace callie_caught_frogs_l723_723353

theorem callie_caught_frogs (A Q B C : ℝ) 
  (hA : A = 2)
  (hQ : Q = 2 * A)
  (hB : B = 3 * Q)
  (hC : C = (5 / 8) * B) : 
  C = 7.5 := by
  sorry

end callie_caught_frogs_l723_723353


namespace problem1_problem2_l723_723577

noncomputable def f (x m : ℝ) := x * log x - (1 / 2) * m * x^2 - x

-- Problem 1: Prove that if f is decreasing on (0, +∞), then m >= 1/e
theorem problem1 (m : ℝ) (h_decreasing : ∀ x > 0, (deriv (λ x, f x m)) x ≤ 0) : m ≥ 1 / real.exp 1 :=
sorry

-- Problem 2: Prove that if f has two extreme points on (0, +∞), then ln x_1 + ln x_2 > 2
theorem problem2 (m x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x1 < x2)
  (h_extreme : deriv (λ x, f x m) x1 = 0 ∧ deriv (λ x, f x m) x2 = 0) : log x1 + log x2 > 2 :=
sorry

end problem1_problem2_l723_723577


namespace number_of_prime_factors_thirty_factorial_l723_723011

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723011


namespace expression_for_a_expression_for_S_l723_723932

noncomputable def a : ℕ → ℕ 
| 1 := 3
| (n + 1) := S n + 2^(n + 1)

def S : ℕ → ℕ 
| 0 := 0
| (n + 1) := a (n + 1) - 2^(n + 1)

theorem expression_for_a (n : ℕ) : a (n + 1) = (n + 2) * 2^n := by
  sorry

theorem expression_for_S (n : ℕ) : S (n + 1) = (n + 1) * 2^(n + 1) := by
  sorry 

end expression_for_a_expression_for_S_l723_723932


namespace number_of_isosceles_triangles_l723_723640

noncomputable def total_isosceles_triangles (ABC : Triangle) (D E F : Point) : ℕ :=
  if (ABC.is_isosceles ∧
      ABC.angle_ABC = 90 ∧
      D.is_midpoint_of ABC.AC ∧
      BD.bisects ABC.angle_ABC ∧
      DE.parallel_to ABC.AB ∧
      F.is_midpoint_of ABC.BC) then
    5
  else
    0

theorem number_of_isosceles_triangles (ABC : Triangle) (D E F : Point)
  (h1 : ABC.is_isosceles)
  (h2 : ABC.angle_ABC = 90)
  (h3 : D.is_midpoint_of ABC.AC)
  (h4 : BD.bisects ABC.angle_ABC)
  (h5 : DE.parallel_to ABC.AB)
  (h6 : F.is_midpoint_of ABC.BC) :
  total_isosceles_triangles ABC D E F = 5 :=
  sorry

end number_of_isosceles_triangles_l723_723640


namespace neighbor_diff_at_least_n_plus_1_l723_723409

theorem neighbor_diff_at_least_n_plus_1 (n : ℕ) (h : n ≥ 2) :
  ∃ (grid : Matrix (Fin n) (Fin n) ℕ),
  (∀ (i j : Fin n), grid i j ∈ Finset.range (n^2 + 1)) ∧
  (∀ (i j : Fin n), ∀ (i' j' : Fin n), (i = i' ∧ abs (j - j') = 1 ∨ j = j' ∧ abs (i - i') = 1 ∨ abs (i - i') = 1 ∧ abs (j - j') = 1) →
    abs (grid i j - grid i' j') ≥ n+1) := sorry

end neighbor_diff_at_least_n_plus_1_l723_723409


namespace prime_factors_of_30_l723_723205

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723205


namespace fractional_factorial_design_test_points_l723_723416

theorem fractional_factorial_design_test_points (k : ℕ) (h : k = 6) :
  let F (n : ℕ) := 2^n - 1 in
  F (k + 1) = 20 :=
by
  intro k h
  sorry

end fractional_factorial_design_test_points_l723_723416


namespace find_large_number_l723_723800

theorem find_large_number (L S : ℤ)
  (h1 : L - S = 2415)
  (h2 : L = 21 * S + 15) : 
  L = 2535 := 
sorry

end find_large_number_l723_723800


namespace factorial_prime_factors_l723_723156

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723156


namespace Vasya_wins_l723_723338

-- Define the board as an 8x8 grid
def board : Type := fin 8 × fin 8

-- Define the initial positions
def initial_position : board × board := ((0,0), (2,2))

-- Define who goes first
def peter_starts : bool := tt

-- Define the target winning position
def winning_position : board := (7, 7)

-- Define the movement rules
def valid_move (p1 p2 : board) : Prop :=
    (p1.1 = p2.1 ∧ p1.2 < p2.2) ∨ (p1.2 = p2.2 ∧ p1.1 < p2.1)

-- Define the game state
structure game_state :=
    (piece1 piece2 : board)
    (turn : bool)

-- Ensure no overlap of pieces
def no_overlap (s : game_state) : Prop :=
    s.piece1 ≠ s.piece2

-- Define a move function
def move_piece (pos : board) (move_up move_right : fin 8 → fin 8) : board :=
{ pos with
  fst := move_up pos.fst,
  snd := move_right pos.snd }

-- Winning predicate
def winning_move (pos : board) : Prop :=
    pos = winning_position

-- Game state update on move
def update_state (s : game_state) (move_up1 move_right1 move_up2 move_right2 : fin 8 → fin 8) : game_state :=
{ piece1 := if s.turn then move_piece s.piece1 move_up1 move_right1 else s.piece1,
  piece2 := if ¬s.turn then move_piece s.piece2 move_up2 move_right2 else s.piece2,
  turn := ¬s.turn }

-- Define the game steps and winning strategy
def vasya_always_wins (initial_state : game_state) : Prop :=
    ∀ (s : game_state), 
    s = initial_state ∨ ∃ next_state : game_state, valid_move s.piece1 next_state.piece1 ∧ valid_move s.piece2 next_state.piece2 ∧ winning_move next_state.piece1 ∨ winning_move next_state.piece2

-- Theorem to prove
theorem Vasya_wins : ∃ initial_state : game_state, 
    vasya_always_wins initial_state :=
begin
  let initial_state := { piece1 := (0, 0), piece2 := (2, 2), turn := peter_starts },
  use initial_state,
  sorry,
end

end Vasya_wins_l723_723338


namespace work_scheduling_l723_723855

theorem work_scheduling (total_parts : ℕ) (days_ahead : ℕ) (extra_daily : ℕ) (intended_days : ℕ) (overachievement_percentage : ℕ) :
  total_parts = 8000 →
  days_ahead = 8 →
  extra_daily = 50 →
  intended_days = 40 →
  overachievement_percentage = 25 :=
by
  intros h_parts h_ahead h_extra h_days h_percentage
  sorry

end work_scheduling_l723_723855


namespace expected_deliveries_total_l723_723891

theorem expected_deliveries_total
  (A_yesterday : ℝ) (A_today_multiplier : ℝ) (A_success_rate : ℝ)
  (B_yesterday : ℝ) (B_today_multiplier : ℝ) (B_success_rate : ℝ) :
  A_yesterday = 80 →
  A_today_multiplier = 2 →
  A_success_rate = 0.9 →
  B_yesterday = 50 →
  B_today_multiplier = 3 →
  B_success_rate = 0.85 →
  let A_today := A_today_multiplier * A_yesterday in
  let A_deliveries := A_today * A_success_rate in
  let B_today := B_today_multiplier * B_yesterday in
  let B_deliveries := B_today * B_success_rate in
  let B_deliveries_floor := Real.floor B_deliveries in
  A_deliveries + B_deliveries_floor = 271 :=
begin
  sorry
end

end expected_deliveries_total_l723_723891


namespace isogonally_conjugate_sum_eq_vertices_l723_723757

noncomputable def unit_circle : set ℂ := {z : ℂ | complex.abs z = 1}

variables (a b c z w : ℂ)

def is_isogonal_conjugate (z w : ℂ) : Prop :=
∃ A B C : ℂ, (complex.im ((a - z) * (a - w) * (conj a - conj b) * (conj a - conj c)) = 0) ∧
  (complex.im ((b - z) * (b - w) * (conj b - conj a) * (conj b - conj c)) = 0) ∧
  (complex.im ((c - z) * (c - w) * (conj c - conj a) * (conj c - conj b)) = 0)

theorem isogonally_conjugate_sum_eq_vertices
  (ha : a ∈ unit_circle)
  (hb : b ∈ unit_circle)
  (hc : c ∈ unit_circle)
  (hz : is_isogonal_conjugate z w)
  : z + w + a * b * c * conj z * conj w = a + b + c :=
begin
  sorry
end

end isogonally_conjugate_sum_eq_vertices_l723_723757


namespace estimate_sqrt_difference_l723_723501

-- Define the statement that we want to prove
theorem estimate_sqrt_difference : 
  (abs ((sqrt 58 - sqrt 55) - 0.20) < 0.01) := 
sorry

end estimate_sqrt_difference_l723_723501


namespace max_value_of_expression_l723_723559

theorem max_value_of_expression 
  (x y : ℝ)
  (h : x^2 + y^2 = 20 * x + 9 * y + 9) :
  ∃ x y : ℝ, 4 * x + 3 * y = 83 := sorry

end max_value_of_expression_l723_723559


namespace num_prime_factors_30_fac_l723_723060

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723060


namespace triangle_parallel_lines_l723_723666

theorem triangle_parallel_lines
  (A B C I X Y : Type)
  [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty I] [Nonempty X] [Nonempty Y]
  {AB AC : ℝ} (h : AB > AC)
  (is_incenter : ∀ {Δ : Type}, Δ = {x // x = A ∧ x = B ∧ x = C} → (I = incenter_of Δ))
  (has_angle_bisectors : ∀ (Δ : Type), Δ = {x // x = A ∧ x = B ∧ x = C} → 
    (angle_bisectors_meet : ∀ {α β γ : Type}, α = angle_bisector B I -> β = angle_bisector C I -> γ = I -> meets(α, β, γ)))
  (X_Y_on_circumcircle : ∀ (Δ : Type), Δ = {x // x = B ∧ x = I ∧ x = C} → 
    (on_circumcircle : ∀ {ψ χ : Type}, ψ = circumcircle Δ -> intersects(χ, ψ, AB ∩ χ) ∧ intersects(χ, ψ, AC ∩ χ) → (χ = X ∨ χ = Y))) :
  parallel_lines (segment C X) (segment B Y) :=
sorry

end triangle_parallel_lines_l723_723666


namespace largest_r_condition_l723_723356

theorem largest_r_condition {r : ℝ} :
  (∀ (a : ℕ → ℕ), (∀ n : ℕ, a n ≤ a (n + 2) ∧ a (n + 2) ≤ nat.sqrt (a n * a n + r * a (n + 1))) →
  ∃ M : ℕ, ∀ n : ℕ, n ≥ M → a (n + 2) = a n) ↔ r = 2 := sorry

end largest_r_condition_l723_723356


namespace num_prime_factors_of_30_l723_723232

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723232


namespace tangent_lines_to_circle_l723_723545

theorem tangent_lines_to_circle (x y : ℝ) (c : ℝ) : 
  (x^2 + y^2 = 1) ∧ (d = Real.sqrt(2)) → 
  (x - y + Real.sqrt(2) = 0 ∨ x - y - Real.sqrt(2) = 0) :=
by
  sorry

end tangent_lines_to_circle_l723_723545


namespace num_prime_factors_30_fac_l723_723062

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723062


namespace different_prime_factors_of_factorial_eq_10_l723_723053

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723053


namespace correct_options_l723_723959

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x) ^ 2

theorem correct_options : 
  (∃ p : ℝ, Real.sin p > 0 ∧ Real.cos p > 0 ∧ p = π) ∧ 
  (f (-π/12) = -√3 / 2) ∧ 
  (¬ (∀ x : ℝ, |f x| = |f (-x)|)) ∧
  (∃ ϕ : ℝ, ϕ = 2 * π / 3 ∧ f(x + ϕ) = f(-x + ϕ)) :=
sorry

end correct_options_l723_723959


namespace factorial_30_prime_count_l723_723194

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723194


namespace grey_cats_in_house_l723_723996

/-- This proof shows that given 16 cats in a house,
    2 of them white, and 25% of them black,
    there are 10 grey cats in the house. -/ 

theorem grey_cats_in_house :
  ∀ (total_cats white_cats : ℕ) (percentage_black : ℝ),
    total_cats = 16 →
    white_cats = 2 →
    percentage_black = 0.25 →
    ∃ (grey_cats : ℕ), grey_cats = total_cats - (white_cats + nat.floor (percentage_black * total_cats)) ∧ grey_cats = 10 :=
by
  intros total_cats white_cats percentage_black ht hw hp
  use 10
  rw [ht, hw, hp]
  have hblack : nat.floor (0.25 * 16) = 4 := by rfl
  rw hblack
  simp
  sorry

end grey_cats_in_house_l723_723996


namespace comb_eq_proof_l723_723495

theorem comb_eq_proof (n : ℕ) : (3 * n + 6 ≤ 18) → nat.choose 18 (3 * n + 6) = nat.choose 18 (4 * n - 2) → n = 2 :=
by
  assume h1 : 3 * n + 6 ≤ 18,
  assume h2 : nat.choose 18 (3 * n + 6) = nat.choose 18 (4 * n - 2),
  sorry

end comb_eq_proof_l723_723495


namespace sine_function_properties_l723_723382

def phi_inequality (phi : Real) : Prop := 0 < phi ∧ -π < phi ∧ phi < 0

def period_condition (ω : Real) : Prop := (2 * π) / ω = π

def shifted_p_sine (ω : Real) (phi : Real) : Prop :=
  ∃ x : Real, (sin (2 * x + (2 * π / 3) + phi)) = 1 

def function_increasing (f : Real → Real) : Prop :=
  ∀ x₁ x₂ : Real, -π/6 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ π/3 → f(x₁) < f(x₂)

theorem sine_function_properties (ω : Real) (φ : Real) (f : Real → Real): 
  phi_inequality φ → period_condition ω → shifted_p_sine ω φ → (f = λ x, sin (2 * x - π / 6)) → function_increasing f :=
by 
  intros _ _ _ _
  sorry

end sine_function_properties_l723_723382


namespace evaluate_a9_l723_723923

noncomputable def a : ℕ → ℕ 
| 1 => 3
| 2 => 10
| n => 3 * a (n - 1) + 2 * a (n - 2)

theorem evaluate_a9 : a 9 = 73368 := by
  sorry

end evaluate_a9_l723_723923


namespace box_height_l723_723328

theorem box_height (x : ℝ) (hx : x + 5 = 10)
  (surface_area : 2*x^2 + 4*x*(x + 5) ≥ 150) : x + 5 = 10 :=
sorry

end box_height_l723_723328


namespace perfect_cubes_between_bounds_l723_723973

theorem perfect_cubes_between_bounds :
  let lower_bound := 3^5 - 1
  let upper_bound := 3^15 + 1
  ∃ n : ℕ, ∃ lower_n upper_n : ℕ,
    lower_n = 7 ∧ upper_n = 24 ∧
    n = upper_n - lower_n + 1 ∧ n = 18 :=
begin
  sorry
end

end perfect_cubes_between_bounds_l723_723973


namespace optimal_pole_l723_723499

def dotson_walking_time (k : ℕ) : ℝ :=
9 - (6 * k / 28)

def williams_walking_time (k : ℕ) : ℝ :=
3 + (2 * k / 7)

def optimal_k : ℕ := 12

theorem optimal_pole :
  dotson_walking_time optimal_k = williams_walking_time optimal_k :=
by
  -- This is the core proof where we show the equality of times for optimal_k = 12.
  sorry

end optimal_pole_l723_723499


namespace prime_factors_of_30_factorial_l723_723175

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723175


namespace complex_conjugate_solution_l723_723936

open Complex

-- Definition for the imaginary unit i
noncomputable def i := Complex.i

-- Definition for the complex number w
noncomputable def w : ℂ := i * (2 - i)

-- Given that z is the conjugate of w
axiom conj_eq (z : ℂ) : z.conj = w

-- The theorem we need to prove
theorem complex_conjugate_solution (z : ℂ) (h : z.conj = w) : z = 1 - 2 * i :=
sorry

end complex_conjugate_solution_l723_723936


namespace prove_inequality_l723_723787

theorem prove_inequality (x : ℝ) (h : x > 2) : x + 1 / (x - 2) ≥ 4 :=
  sorry

end prove_inequality_l723_723787


namespace trains_cross_time_l723_723430

theorem trains_cross_time (length : ℝ) (time1 time2 : ℝ) (speed1 speed2 relative_speed : ℝ) 
  (H1 : length = 120) 
  (H2 : time1 = 12) 
  (H3 : time2 = 20) 
  (H4 : speed1 = length / time1) 
  (H5 : speed2 = length / time2) 
  (H6 : relative_speed = speed1 + speed2) 
  (total_distance : ℝ) (H7 : total_distance = length + length) 
  (T : ℝ) (H8 : T = total_distance / relative_speed) :
  T = 15 := 
sorry

end trains_cross_time_l723_723430


namespace num_prime_factors_30_factorial_l723_723239

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723239


namespace abs_diff_51st_terms_l723_723405

theorem abs_diff_51st_terms :
  |((30 + 8 * (51 - 1)) - (30 - 12 * (51 - 1)))| = 1000 := 
by
  sorry

end abs_diff_51st_terms_l723_723405


namespace factorial_prime_factors_l723_723160

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723160


namespace candy_distribution_l723_723257

theorem candy_distribution : ∀ (pieces_of_candy friends each_friend : ℕ), 
  (pieces_of_candy = 45) → (each_friend = 5) → (friends = pieces_of_candy / each_friend) → (friends = 9) :=
by {
  intros,
  sorry
}

end candy_distribution_l723_723257


namespace exists_convex_figure_with_larger_area_l723_723346

theorem exists_convex_figure_with_larger_area (Φ : Type*) [convex_figure Φ] :
  (∀ AB : chord Φ, 
    let (Φ₁, Φ₂) := divide_by_chord Φ AB in
    perimeter Φ₁ = perimeter Φ₂ ∧ area Φ₁ ≠ area Φ₂) →
  ∃ Φ' : Type*, [convex_figure Φ'] ∧ perimeter Φ' = perimeter Φ ∧ area Φ' > area Φ :=
by
  sorry

end exists_convex_figure_with_larger_area_l723_723346


namespace minimum_value_of_quadratic_l723_723986

variable (b : ℝ)

theorem minimum_value_of_quadratic (b : ℝ) (h : b = 100) : 
  let c := b - 4 in
  c = 96 :=
by
  sorry

end minimum_value_of_quadratic_l723_723986


namespace option_B_option_C_option_D_l723_723263

section CharacteristicFunction

variable (f : ℝ → ℝ) (λ : ℝ)

-- Definition of a λ characteristic function
def λ_char_function : Prop :=
  ∀ x, f(x + λ) + λ * f(x) = 0

-- Prove that f(x) = 2x + 1 is not a λ characteristic function
theorem option_B : ¬ ∃ λ, λ_char_function (λ := λ) (f := fun x => 2 * x + 1) :=
sorry

-- Prove that a (1/3) characteristic function has at least one zero
theorem option_C : ∃ x, f(x) = 0 ∧ λ_char_function (λ := 1/3) f :=
sorry

-- Prove that f(x) = e^x is a λ characteristic function if λ satisfies a certain condition
theorem option_D (h : ∃ λ, λ_char_function (λ := λ) (f := fun x => Real.exp x)) : True :=
sorry

end CharacteristicFunction

end option_B_option_C_option_D_l723_723263


namespace number_of_prime_factors_thirty_factorial_l723_723013

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723013


namespace part1_part2_l723_723552

-- Part 1
theorem part1 (a : ℝ) (h1 : ∃ x ∈ Icc (0 : ℝ) 1, (exp (-x) - exp x) > -log a) : 
  a > 1 :=
begin
  sorry
end

-- Part 2
theorem part2 :
  ∃ x0 : ℝ, 
    (-log x0 + cos (π * x0 / (2 * exp 1)) = 0) ∧ 
    (1 - exp (2 : ℝ)) / exp 1 < exp (-cos (π * x0 / (2 * exp 1))) - exp (cos (π * x0 / (2 * exp 1))) ∧ 
    exp (-cos (π * x0 / (2 * exp 1))) - exp (cos (π * x0 / (2 * exp 1))) < (1 - exp 1) / exp (1 / 2) :=
begin
  sorry
end

end part1_part2_l723_723552


namespace rectangle_area_l723_723636

theorem rectangle_area {X Y Z : Point}
  (radius_X radius_Y radius_Z : ℝ)
  (hXY : distance X Y = 30)
  (hYZ : distance Y Z = 20)
  (hXZ : distance X Z = 40)
  (h_touch_X : touches_three_sides X)
  (h_touch_Z : touches_two_sides Z)
  : Area (rectangle_PQRS) = 3936.5 := 
sorry

end rectangle_area_l723_723636


namespace find_positive_number_l723_723799
-- Prove the positive number x that satisfies the condition is 8
theorem find_positive_number (x : ℝ) (hx : 0 < x) :
    x + 8 = 128 * (1 / x) → x = 8 :=
by
  intro h
  sorry

end find_positive_number_l723_723799


namespace find_fn_mod_2015_eq_neg_cos_l723_723260

noncomputable def f : ℕ → (ℝ → ℝ)
| 0       := λ x, Real.sin x
| (n + 1) := λ x, (f n)'.eval x

theorem find_fn_mod_2015_eq_neg_cos :
  (f 2015) = λ x, -Real.cos x :=
sorry

end find_fn_mod_2015_eq_neg_cos_l723_723260


namespace factorial_prime_factors_l723_723082

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723082


namespace standard_deviation_is_one_l723_723621

noncomputable def standard_deviation (μ : ℝ) (σ : ℝ) : Prop :=
  ∀ x : ℝ, (0.68 * μ ≤ x ∧ x ≤ 1.32 * μ) → σ = 1

theorem standard_deviation_is_one (a : ℝ) (σ : ℝ) :
  (0.68 * a ≤ a + σ ∧ a + σ ≤ 1.32 * a) → σ = 1 :=
by
  -- Proof omitted.
  sorry

end standard_deviation_is_one_l723_723621


namespace problem_equivalent_proof_l723_723909

noncomputable def sqrt (x : ℝ) : ℝ := x.sqrt -- Define the sqrt function for real numbers

theorem problem_equivalent_proof (x : ℝ) :
  sqrt ((3 + sqrt 8) ^ x) + sqrt ((3 - sqrt 8) ^ x) = 6 ↔ x = 2 ∨ x = -2 :=
begin
  sorry -- This is where the proof would be, omitted as instructed
end

end problem_equivalent_proof_l723_723909


namespace difference_of_results_l723_723784

theorem difference_of_results (x : ℕ) (h : x = 15) : 2 * x - (26 - x) = 19 :=
by
  rw [h]
  sorry

end difference_of_results_l723_723784


namespace num_prime_factors_30_factorial_l723_723033

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723033


namespace coefficient_x9_expansion_l723_723725

theorem coefficient_x9_expansion :
  ∀ (x : ℝ), coeff_of_x9 (∏ i in (1: Finset ℕ).range 11, (x - i)) = -55 := 
by sorry

-- The function that finds the coefficient of x^9
def coeff_of_x9 (polynomial : Polynomial ℝ) : ℝ := polynomial.coeff 9

-- Sorry stands as a placeholder for the proof, which is not required here.

end coefficient_x9_expansion_l723_723725


namespace profit_percentage_l723_723283

noncomputable def original_profit_percentage (C S : ℝ) := ((S - C) / C) * 100

theorem profit_percentage (C S : ℝ)
  (h1 : C = 0.4 * S)
  (h2 : ∀ S : ℝ, 0.552 * S = S - 1.12 * C) :
  original_profit_percentage C S = 150 :=
by
  unfold original_profit_percentage
  rw h1
  sorry

end profit_percentage_l723_723283


namespace students_answered_both_correct_l723_723709

theorem students_answered_both_correct (total_students : ℕ)
  (answered_sets_correctly : ℕ) (answered_functions_correctly : ℕ)
  (both_wrong : ℕ) (total : total_students = 50)
  (sets_correct : answered_sets_correctly = 40)
  (functions_correct : answered_functions_correctly = 31)
  (wrong_both : both_wrong = 4) :
  (40 + 31 - (total_students - 4) + both_wrong = 50) → total_students - (40 + 31 - (total_students - 4)) = 29 :=
by
  sorry

end students_answered_both_correct_l723_723709


namespace cube_sum_identity_l723_723594

theorem cube_sum_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^3 + 1/r^3 = 2 * Real.sqrt 5 ∨ r^3 + 1/r^3 = -2 * Real.sqrt 5 := by
  sorry

end cube_sum_identity_l723_723594


namespace number_of_valid_tuples_l723_723704

-- Definitions of the conditions
def valid_tuple (r n : ℕ) (x : Fin r → Fin n) : Prop :=
  ∀ i : Fin n, (∃ t : Fin r, x t ≤ i) → (∃ j : Fin n, t : Nat => i > x t, j < i.val)

-- Defining the function f(r, n) to count valid tuples
noncomputable def f (r n : ℕ) : ℕ :=
  if h : 1 ≤ r ∧ r ≤ n then
    -- Calculation of the number of valid tuples
    (n - r) * n ^ (r - 1)
  else
    0

-- The theorem statement to prove the problem solution
theorem number_of_valid_tuples (r n : ℕ) (h : 1 ≤ r ∧ r ≤ n) :
  ∃ count : ℕ, count = f r n :=
begin
  use (n - r) * n ^ (r - 1),
  split,
  { exact h },
  sorry
end

end number_of_valid_tuples_l723_723704


namespace compute_expression_l723_723314

theorem compute_expression : 
  let a := (5 : ℚ) / 7
  let b := (4 : ℚ) / 5
  a^(-3) * b^2 = 5488 / 3125 :=
by
  sorry

end compute_expression_l723_723314


namespace prime_factors_of_30_factorial_l723_723112

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723112


namespace three_points_in_circle_l723_723626

structure Square :=
(sideLength : ℝ)

structure Point :=
(x : ℝ)
(y : ℝ)

def unitSquare : Square := { sideLength := 1 }

def pointsInSquare : List Point := sorry -- Assume we have a list of 51 points within the unit square

noncomputable def circleRadius := 1 / 7

theorem three_points_in_circle (s : Square) (pts : List Point) (r : ℝ) 
  (h₁ : s.sideLength = 1)
  (h₂ : pts.length = 51)
  (h₃ : r = 1 / 7) :
  ∃ (a b c : Point), a ∈ pts ∧ b ∈ pts ∧ c ∈ pts ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  (dist a b ≤ 2 * r) ∧ (dist b c ≤ 2 * r) ∧ (dist c a ≤ 2 * r) :=
by
  sorry

end three_points_in_circle_l723_723626


namespace probability_Alex_paired_with_Jamie_l723_723277

theorem probability_Alex_paired_with_Jamie:
  (let total_students := 24 in
  let possible_partners := total_students - 1 in
  let favorable_outcomes := 1 in
  let probability := (favorable_outcomes : ℚ) / possible_partners in
  probability = 1 / 23) :=
by
  let total_students := 24
  let possible_partners := total_students - 1
  let favorable_outcomes := 1
  let probability := (favorable_outcomes : ℚ) / possible_partners
  show probability = 1 / 23
  sorry

end probability_Alex_paired_with_Jamie_l723_723277


namespace triangle_area_l723_723379

noncomputable def calculate_area (a b c : ℝ) (A C : ℝ) : ℝ := 
  let S := sqrt (1 / 4 * (a^2 * c^2 - ((a^2 + c^2 - b^2) / 2)^2)) in S

theorem triangle_area 
  (a b c A C : ℝ)
  (h1 : a^2 * sin C = 4 * sin A)
  (h2 : (a + c)^2 = 12 + b^2) :
  calculate_area a b c A C = sqrt 3 :=
by
  sorry

end triangle_area_l723_723379


namespace range_of_a_l723_723518

open Real 

noncomputable def trigonometric_inequality (θ a : ℝ) : Prop :=
  sin (2 * θ) - (2 * sqrt 2 + sqrt 2 * a) * sin (θ + π / 4) - 2 * sqrt 2 / cos (θ - π / 4) > -3 - 2 * a

theorem range_of_a (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → trigonometric_inequality θ a) ↔ (a > 3) :=
sorry

end range_of_a_l723_723518


namespace slightly_used_crayons_correct_l723_723758

def total_crayons : ℕ := 120
def new_crayons : ℕ := total_crayons / 3
def broken_crayons : ℕ := (total_crayons * 20) / 100
def slightly_used_crayons : ℕ := total_crayons - new_crayons - broken_crayons

theorem slightly_used_crayons_correct : slightly_used_crayons = 56 := sorry

end slightly_used_crayons_correct_l723_723758


namespace sin_squared_alpha_plus_pi_over_4_l723_723539

theorem sin_squared_alpha_plus_pi_over_4 (α : ℝ) (h₁ : 0 < α ∧ α < π / 4) (h₂ : cos (2 * α) = 4 / 5) : 
  sin (α + π / 4) ^ 2 = 4 / 5 :=
sorry

end sin_squared_alpha_plus_pi_over_4_l723_723539


namespace fourth_intersection_point_l723_723292

-- Conditions in Lean 4
def hyperbola (x y : ℝ) : Prop := x * y = 2
def circle_point_1 : (ℝ, ℝ) := (4, 1/2)
def circle_point_2 : (ℝ, ℝ) := (-2, -1)
def circle_point_3 : (ℝ, ℝ) := (2/3, 3)
def fourth_point := (-3/4, -8/3)

-- Problem statement in Lean
theorem fourth_intersection_point :
  (hyperbola (circle_point_1.1) (circle_point_1.2)) ∧
  (hyperbola (circle_point_2.1) (circle_point_2.2)) ∧
  (hyperbola (circle_point_3.1) (circle_point_3.2)) →
  hyperbola (fourth_point.1) (fourth_point.2) :=
sorry

end fourth_intersection_point_l723_723292


namespace arithmetic_seq_a8_l723_723729

theorem arithmetic_seq_a8
  (a : ℕ → ℤ)
  (h1 : a 5 = 10)
  (h2 : a 1 + a 2 + a 3 = 3) :
  a 8 = 19 := sorry

end arithmetic_seq_a8_l723_723729


namespace prime_factors_of_30_factorial_l723_723180

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723180


namespace min_value_transformation_l723_723612

theorem min_value_transformation:
  ∀ (f : ℝ → ℝ) (c : ℝ),
  (∀ x : ℝ, f x = x^2 + 4 * x + 5 - c) →
  (∀ x : ℝ, (x + 2)^2 + 2 = 2) →
  (∀ a : ℝ, f (a - 2015) = (a - 2013)^2 + 2) :=
begin
  sorry
end

end min_value_transformation_l723_723612


namespace arithmetic_sequence_n_is_17_l723_723633

theorem arithmetic_sequence_n_is_17
  (a : ℕ → ℤ)  -- An arithmetic sequence a_n
  (h1 : a 1 = 5)  -- First term is 5
  (h5 : a 5 = -3)  -- Fifth term is -3
  (hn : a n = -27) : n = 17 := sorry

end arithmetic_sequence_n_is_17_l723_723633


namespace factorial_prime_factors_l723_723081

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723081


namespace other_root_l723_723950

open Complex

-- Defining the conditions that are given in the problem
def quadratic_equation (x : ℂ) (m : ℝ) : Prop :=
  x^2 + (1 - 2 * I) * x + (3 * m - I) = 0

def has_real_root (x : ℂ) : Prop :=
  ∃ α : ℝ, x = α

-- The main theorem statement we need to prove
theorem other_root (m : ℝ) (α : ℝ) (α_real_root : quadratic_equation α m) :
  quadratic_equation (-1/2 + 2 * I) m :=
sorry

end other_root_l723_723950


namespace total_students_at_woojung_high_school_l723_723391

variable (num_non_first_year_students num_first_year_students : ℕ)
variable (total_students : ℕ)

def num_first_year_students_def : Prop := num_first_year_students = num_non_first_year_students - 468
def total_students_def : Prop := total_students = num_first_year_students + num_non_first_year_students

theorem total_students_at_woojung_high_school
    (h1 : num_non_first_year_students = 954)
    (h2 : num_first_year_students_def)
    (h3 : total_students_def) :
  total_students = 1440 :=
by
  sorry

end total_students_at_woojung_high_school_l723_723391


namespace f_simplified_f_value_l723_723948

noncomputable def f (α : Real) : Real :=
  (sin (α - π / 2) * cos (3 * π / 2 + α) * tan (π - α)) / 
  (tan (-α - π) * sin (-α - 3 * π))

theorem f_simplified (α : Real) (h1 : α > π ∧ α < 3 * π) :
  f(α) = -cos(α) :=
sorry

theorem f_value (α : Real) (h1 : α > π ∧ α < 3 * π) 
  (h2 : cos (α - 3 * π / 2) = 1 / 5) :
  f(α) = 2 * sqrt 6 / 5 :=
sorry

end f_simplified_f_value_l723_723948


namespace factorial_30_prime_count_l723_723192

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723192


namespace spiralAbove900_l723_723365

-- Description of the spiral pattern.
def isSpiralSequence (n : ℕ) (seq : ℕ → ℕ × ℕ) : Prop :=
  ∀ k : ℕ, k ∈ [1..n] → seq(k - 1) = -- the coordinate calculation for spiral sequence
  sorry

-- Given conditions about the spiral sequence and natural numbers.
def TanyaSpiral (seq : ℕ → ℕ × ℕ) : Prop :=
  isSpiralSequence 1000 seq

-- The theorem to be proved: 1023 is directly above 900 in Tanya's spiral sequence.
theorem spiralAbove900 (seq : ℕ → ℕ × ℕ) (h : TanyaSpiral seq) : seq(1023) = -- coordinates directly above seq(900)
  sorry

end spiralAbove900_l723_723365


namespace factorial_prime_factors_l723_723153

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723153


namespace area_under_4g_shifted_l723_723369

-- Definitions of the functions involved
def g (x : ℝ) : ℝ := sorry

-- The given condition about the area under y = g(x)
def area_under_g : ℝ := 8

-- Theorem statement about the area under y = 4g(x + 3)
theorem area_under_4g_shifted :
  let area_under_g_shifted := area_under_g in
  let vertical_scaling_factor := 4 in
  area_under_4g_shifted * vertical_scaling_factor = 32 :=
sorry

end area_under_4g_shifted_l723_723369


namespace number_of_prime_factors_of_30_factorial_l723_723126

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723126


namespace part_I_part_II_part_III_l723_723992

variable (a : ℕ → ℤ)(S : ℕ → ℤ)
-- Given conditions 
axiom a1 : a 1 = 2
axiom Sn : ∀ n, S (n + 1) = 4 * a n - 2

-- Definitions inferred from the given conditions
def a_2 := a 2
def a_3 := a 3

-- Proving Part (I)
theorem part_I : a_2 = 4 ∧ a_3 = 8 := by
  sorry

-- Proving Part (II)
theorem part_II : ∀ n ≥ 2, a n - 2 * a (n - 1) = 0 := by
  sorry

-- Proving Part (III)
theorem part_III : ∀ n ≥ 1, ∑ k in (Finset.range n), (a k + 1 - 1) / (a k + 1 - 1) < n / 2 := by
  sorry

end part_I_part_II_part_III_l723_723992


namespace maximum_value_of_parabola_eq_24_l723_723413

theorem maximum_value_of_parabola_eq_24 (x : ℝ) : 
  ∃ x, x = -2 ∧ (-2 * x^2 - 8 * x + 16) = 24 :=
by
  use -2
  sorry

end maximum_value_of_parabola_eq_24_l723_723413


namespace prime_factors_of_30_factorial_l723_723105

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723105


namespace rational_function_proof_l723_723360

noncomputable def f (x : ℚ) : ℚ := -- the function f, which is to be specified later
 sorry

theorem rational_function_proof (h : ∀ x ≠ 0, 3 * f (1/x) + 2 * f x / x = x ^ 2) :
  f (-2) = 67 / 20 :=
begin
  sorry
end

end rational_function_proof_l723_723360


namespace factorial_prime_factors_l723_723080

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723080


namespace intersection_M_N_l723_723953

def M : Set ℝ := {x | x < 1/2}
def N : Set ℝ := {y | y ≥ -4}

theorem intersection_M_N :
  (M ∩ N = {x | -4 ≤ x ∧ x < 1/2}) :=
sorry

end intersection_M_N_l723_723953


namespace area_quadrilateral_twice_incenter_triangle_l723_723646

variable (A B C D E I : Type) [Triangle A B C]
variable [Angle A B C = 90°]
variable [Is_incenter I A B C]
variable [Angle_bisects B D C E]
variable (S : A → ℝ)

theorem area_quadrilateral_twice_incenter_triangle :
  S (quadrilateral B C D E) = 2 * S (triangle I B C) :=
sorry

end area_quadrilateral_twice_incenter_triangle_l723_723646


namespace factorial_prime_factors_l723_723088

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723088


namespace coaching_fee_correct_l723_723688

noncomputable def total_coaching_fee : ℝ :=
  let daily_fee : ℝ := 39
  let discount_threshold : ℝ := 50
  let discount_rate : ℝ := 0.10
  let total_days : ℝ := 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 3 -- non-leap year days count up to Nov 3
  let discount_days : ℝ := total_days - discount_threshold
  let discounted_fee : ℝ := daily_fee * (1 - discount_rate)
  let fee_before_discount : ℝ := discount_threshold * daily_fee
  let fee_after_discount : ℝ := discount_days * discounted_fee
  fee_before_discount + fee_after_discount

theorem coaching_fee_correct :
  total_coaching_fee = 10967.7 := by
  sorry

end coaching_fee_correct_l723_723688


namespace point_in_fourth_quadrant_l723_723985

variable (a : ℝ)

theorem point_in_fourth_quadrant (h : a < -1) : 
    let x := a^2 - 2*a - 1
    let y := (a + 1) / abs (a + 1)
    (x > 0) ∧ (y < 0) := 
by
  let x := a^2 - 2*a - 1
  let y := (a + 1) / abs (a + 1)
  sorry

end point_in_fourth_quadrant_l723_723985


namespace no_solutions_exists_unique_l723_723510

def is_solution (a b c x y z : ℤ) : Prop :=
  2 * x - b * y + z = 2 * b ∧
  a * x + 5 * y - c * z = a

def no_solutions_for (a b c : ℤ) : Prop :=
  ∀ x y z : ℤ, ¬ is_solution a b c x y z

theorem no_solutions_exists_unique (a b c : ℤ) :
  (a = -2 ∧ b = 5 ∧ c = 1) ∨
  (a = 2 ∧ b = -5 ∧ c = -1) ∨
  (a = 10 ∧ b = -1 ∧ c = -5) ↔
  no_solutions_for a b c := 
sorry

end no_solutions_exists_unique_l723_723510


namespace part1_part2_part3_l723_723632

-- Part (1)
theorem part1 (x : ℝ) : (abs x + abs (1 - x) ≤ 1) → (0 ≤ x ∧ x ≤ 1) :=
sorry

-- Part (2)
theorem part2 (x1 x2 : ℝ) : 
  min (abs (x1 - x2) + abs (2 * x1 - 2 - x2^2)) = 1/2 :=
sorry

-- Part (3)
theorem part3 (a b : ℝ) : 
  let M := λ a b, max (abs (a - x) + abs (b - x^2)) -- A function to denote maximum distance 
  in (∃ x ∈ [-2, 2], M a b = 25/8) → (a = 0 ∧ b = 23/8) :=
sorry

end part1_part2_part3_l723_723632


namespace prime_factors_of_30_factorial_l723_723100

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723100


namespace parallel_CX_BY_l723_723663

variables {A B C I X Y : Type} [triangle ABC] [point I] [point X] [point Y]

-- Introduce conditions
axiom angle_bisectors_meet_at_I : 
  (angle_bisector_ABC B I) ∧ (angle_bisector_ABC C I)

axiom circumcircle_BIC_intersects_AB_at_X : 
  (circumcircle B I C).intersects (segment A B) a second time at X

axiom circumcircle_BIC_intersects_AC_at_Y : 
  (circumcircle B I C).intersects (segment A C) a second time at Y

-- Statement to prove
theorem parallel_CX_BY : 
  ∀ (A B C I X Y : Type) [triangle ABC] [point I] [point X] [point Y], 
  (AB > AC) → 
  (angle_bisectors_meet_at_I I) → 
  (circumcircle_BIC_intersects_AB_at_X X) → 
  (circumcircle_BIC_intersects_AC_at_Y Y) → 
  (parallel (line C X) (line B Y)) :=
by
  intros,
  sorry

end parallel_CX_BY_l723_723663


namespace num_prime_factors_30_fac_l723_723067

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723067


namespace different_prime_factors_of_factorial_eq_10_l723_723058

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723058


namespace number_of_prime_factors_of_30_factorial_l723_723122

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723122


namespace cocktail_cost_l723_723389

noncomputable def costPerLitreCocktail (cost_mixed_fruit_juice : ℝ) (cost_acai_juice : ℝ) (volume_mixed_fruit : ℝ) (volume_acai : ℝ) : ℝ :=
  let total_cost := cost_mixed_fruit_juice * volume_mixed_fruit + cost_acai_juice * volume_acai
  let total_volume := volume_mixed_fruit + volume_acai
  total_cost / total_volume

theorem cocktail_cost : costPerLitreCocktail 262.85 3104.35 32 21.333333333333332 = 1399.99 :=
  by
    sorry

end cocktail_cost_l723_723389


namespace weights_identical_l723_723401

theorem weights_identical (w : Fin 13 → ℤ) 
  (h : ∀ i, ∃ (A B : Finset (Fin 13)), A.card = 6 ∧ B.card = 6 ∧ A ∪ B = Finset.univ.erase i ∧ (A.sum w) = (B.sum w)) :
  ∀ i j, w i = w j :=
by
  sorry

end weights_identical_l723_723401


namespace num_prime_factors_30_factorial_l723_723029

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723029


namespace base_k_perfect_square_l723_723352

theorem base_k_perfect_square (k : ℤ) (h : k ≥ 6) : 
  (1 * k^8 + 2 * k^7 + 3 * k^6 + 4 * k^5 + 5 * k^4 + 4 * k^3 + 3 * k^2 + 2 * k + 1) = (k^4 + k^3 + k^2 + k + 1)^2 := 
by
  sorry

end base_k_perfect_square_l723_723352


namespace equal_angles_l723_723289

variables {A B C H I M K N : Type}
variables [EuclideanGeometry A] [Triangle A B C] [Circumcircle ABC A] [ArcMidpoint ABC A M]
variables [orthocenter ABC H] [incenter ABC I] [point_on_circle ABC K] [vertical_angle AK H ∘ A K H]
variables (N : Type) [intersection_point (line A H) (line M I) N]

theorem equal_angles (hA : AC > AB) 
  (hAH : orthocenter ABC H) 
  (hAI : incenter ABC I) 
  (hCircum : circumcircle ABC A) 
  (hM : midpoint_arc BAC M) 
  (hK : ∃ K, point_on_circle ABC K ∧ angle AKH = 90) 
  (hN : ∃ N, intersection_point (line AH) (line MI) N ∧ point_on_circle ABC N) :
  ∠ IKH = ∠ INH :=
sorry

end equal_angles_l723_723289


namespace factorial_prime_factors_l723_723163

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723163


namespace shortest_distance_is_line_segment_l723_723418

-- Definition of the shortest distance between two points in Euclidean geometry
theorem shortest_distance_is_line_segment 
  (P Q : EuclideanSpace ℝ 3) : 
  ∃ (L : segment P Q), ∀ (p ≠ Q), dist(P,Q) = segment.length :=
sorry

end shortest_distance_is_line_segment_l723_723418


namespace matrix_not_invertible_values_l723_723528

noncomputable theory

open Matrix

-- Define the matrix
def myMatrix (a b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![a + d, b, c], 
    ![b, c + d, a], 
    ![c, a, b + d]]

variables {a b c d : ℝ}

-- Prove the given statement:
theorem matrix_not_invertible_values :
  det (myMatrix a b c d) = 0 → 
  (d = - a - b - c ∨ a = b ∧ b = c ∨ a + b + c = 0 ∨ someOtherSpecialCase) → -- conditions for simplicity
  (∃ v, v = (a / (b + c) + b / (a + c) + c / (a + b)) ∧ (v = -3 ∨ v = 3 / 2)) := 
sorry

end matrix_not_invertible_values_l723_723528


namespace factorial_prime_factors_l723_723159

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723159


namespace factorial_prime_factors_l723_723077

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723077


namespace boxes_needed_to_pack_all_muffins_l723_723721

theorem boxes_needed_to_pack_all_muffins
  (total_muffins : ℕ := 95)
  (muffins_per_box : ℕ := 5)
  (available_boxes : ℕ := 10) :
  (total_muffins / muffins_per_box) - available_boxes = 9 :=
by
  sorry

end boxes_needed_to_pack_all_muffins_l723_723721


namespace folded_rectangle_side_length_l723_723456

theorem folded_rectangle_side_length (a b : ℝ) (h1 : a = 5) (h2 : b < 4) (h3 : sqrt (5^2 + b^2) = sqrt 6) : b = sqrt 5 :=
by
  sorry

end folded_rectangle_side_length_l723_723456


namespace different_prime_factors_of_factorial_eq_10_l723_723045

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723045


namespace Jim_remaining_miles_l723_723429

-- Define the total journey miles and miles already driven
def total_miles : ℕ := 1200
def miles_driven : ℕ := 215

-- Define the remaining miles Jim needs to drive
def remaining_miles (total driven : ℕ) : ℕ := total - driven

-- Statement to prove
theorem Jim_remaining_miles : remaining_miles total_miles miles_driven = 985 := by
  -- The proof is omitted
  sorry

end Jim_remaining_miles_l723_723429


namespace florida_texas_license_plates_difference_l723_723523

theorem florida_texas_license_plates_difference :
  let florida_plates := 26^6 * 10^2,
      texas_plates := 26^3 * 10^4
  in florida_plates - texas_plates = 54293545536 := by
  sorry

end florida_texas_license_plates_difference_l723_723523


namespace num_prime_factors_30_factorial_l723_723244

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723244


namespace find_distance_AB_l723_723288

-- Define C_1 in Cartesian coordinates
def curve_C1_parametric (θ : Real) : Real × Real :=
  (Real.cos θ, 1 + Real.sin θ)

-- Define C_2 in Cartesian coordinates
def curve_C2 (x y : Real) : Prop :=
  x^2 + y^2 / 2 = 1

-- Define the polar equation of C_1
def curve_C1_polar (θ : Real) : Real :=
  2 * Real.sin θ

-- Define the polar equation of C_2
def curve_C2_polar (θ : Real) (ρ : Real) : Prop :=
  ρ^2 * (1 + Real.cos θ^2) = 2

-- Define the distance between intersections
def distance_AB (θ : Real) (ρ1 ρ2 : Real) : Real :=
  Real.abs (ρ1 - ρ2)

-- Main theorem stating the result
theorem find_distance_AB :
  let θ := Real.pi / 3
  let ρ1 := curve_C1_polar θ
  let ρ2 := (2 * Real.sqrt 10) / 5
  ρ2^2 * (1 + (Real.cos θ)^2) = 2 → -- Condition for ρ2 from curve_C2_polar
  distance_AB θ ρ1 ρ2 = Real.sqrt 3 - (2 * Real.sqrt 10) / 5 :=
by
  intros θ ρ1 ρ2 h
  sorry

end find_distance_AB_l723_723288


namespace factorial_prime_factors_l723_723150

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723150


namespace women_who_do_not_speak_french_percentage_l723_723820

theorem women_who_do_not_speak_french_percentage :
  (∀ (total_employees men employees_speak_french : ℕ),
    total_employees = 100 →
    men = 35 →
    employees_speak_french = 40 →
    60% of 35 = 21 →
    40% of 100 = 40 →
    let women := total_employees - men in
    let women_speak_french := employees_speak_french - 21 in
    let women_not_speak_french := women - women_speak_french in
    (women_not_speak_french / women) * 100 = 70.77)
sorry

end women_who_do_not_speak_french_percentage_l723_723820


namespace f_three_element_bound_l723_723703

variable (S : Set ℕ := {n | n ∈ Set.range (Nat.succ 2017)}) 
variable (f : Set ℕ → ℝ)

-- Given conditions
axiom f_nonneg : ∀ A : Set ℕ, 0 ≤ f A
axiom submodular : ∀ A B : Set ℕ, f (A ∪ B) + f (A ∩ B) ≤ f A + f B
axiom monotonic : ∀ A B : Set ℕ, A ⊆ B → f A ≤ f B
axiom specific_inequality : ∀ k j : ℕ, k ∈ S → j ∈ S → f ({n | n ≤ k + 1} ∩ S) ≥ f (({n | n ≤ k} ∪ {j}) ∩ S)
axiom empty_set : f ∅ = 0

-- Proof goal
theorem f_three_element_bound (T : Set ℕ) (hT : T ⊆ S) (hT_card : T.card = 3) :
  f T ≤ (27 / 19) * f ({1, 2, 3} : Set ℕ) :=
sorry

end f_three_element_bound_l723_723703


namespace derivative_at_pi_over_4_l723_723583

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem derivative_at_pi_over_4 : (deriv f) (Real.pi / 4) = 0 := 
by
  sorry

end derivative_at_pi_over_4_l723_723583


namespace school_minimum_payment_l723_723755

noncomputable def individual_ticket_price : ℝ := 6
noncomputable def group_ticket_price : ℝ := 40
noncomputable def discount : ℝ := 0.9
noncomputable def students : ℕ := 1258

-- Define the minimum amount the school should pay
noncomputable def minimum_amount := 4536

theorem school_minimum_payment :
  (students / 10 : ℝ) * group_ticket_price * discount + 
  (students % 10) * individual_ticket_price * discount = minimum_amount := sorry

end school_minimum_payment_l723_723755


namespace perimeter_area_inequalities_l723_723431

open_locale real

noncomputable theory

variables {n : ℕ} (M M' : fin n → ℝ) (K : set ℝ) 

-- Conditions: M and M' are convex polygons inscribed in the same circle K
-- M' vertices are midpoints of arcs subtending sides of M
-- Define perimeter and area functions
def perimeter (M : fin n → ℝ) : ℝ := 2 * (finset.univ.sum (λ i, real.sin (M i)))
def area (M : fin n → ℝ) : ℝ := 0.5 * (finset.univ.sum (λ i, real.sin (2 * M i)))

-- Let P and S be the perimeter and area of M
-- Let P' and S' be the perimeter and area of M'
def P : ℝ := perimeter M
def S : ℝ := area M
def P' : ℝ := perimeter M'
def S' : ℝ := area M'

-- Theorem to prove P' >= P and S' >= S with equality conditions
theorem perimeter_area_inequalities : 
  P' ≥ P ∧ S' ≥ S ∧ (P' = P ↔ ∀ i, M i = M 0) ∧ (S' = S ↔ ∀ i, M i = M 0) :=
sorry

end perimeter_area_inequalities_l723_723431


namespace homogeneous_triangular_plate_l723_723917

noncomputable def staticMoments (a : ℝ) : Prop :=
  let σ := {p : ℝ × ℝ | 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ p.1 + p.2 ≤ a}
  let z (p : ℝ × ℝ) : ℝ := a - p.1 - p.2
  ∫ (p : ℝ × ℝ) in σ, z p * (√3) = (√3 / 6) * a^3

noncomputable def centerOfMass (a : ℝ) : Prop :=
  let xc := a / 3
  let yc := a / 3
  let zc := a / 3
  xc = (a / 3) ∧ yc = (a / 3) ∧ zc = (a / 3)

theorem homogeneous_triangular_plate (a : ℝ) (h : 0 < a) : 
  staticMoments a ∧ centerOfMass a :=
by 
  sorry

end homogeneous_triangular_plate_l723_723917


namespace race_distance_l723_723827

theorem race_distance (D : ℕ) 
  (A_time : D / 28) 
  (B_time : D / 32) 
  (B_distance_when_A_finishes : D - 20) : 
  D = 160 :=
by 
  sorry

end race_distance_l723_723827


namespace intersection_P_Q_l723_723987

def P : Set ℝ := { x | x^2 - x = 0 }
def Q : Set ℝ := { x | x^2 + x = 0 }

theorem intersection_P_Q : (P ∩ Q) = {0} := 
by
  sorry

end intersection_P_Q_l723_723987


namespace number_of_prime_factors_thirty_factorial_l723_723008

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723008


namespace prove_function_domain_l723_723727

def function_domain := {x : ℝ | (x + 4 ≥ 0 ∧ x ≠ 0)}

theorem prove_function_domain :
  function_domain = {x : ℝ | x ∈ (Set.Icc (-4:ℝ) 0).diff ({0}:Set ℝ) ∪ (Set.Ioi 0)} :=
by
  sorry

end prove_function_domain_l723_723727


namespace find_m_correct_l723_723547

noncomputable def find_m (data : List (ℝ × ℝ)) (regression_eq : ℝ → ℝ) : ℝ :=
  let mean_x := (2 + 3 + 4 + 5) / 4
  let mean_y := (15 + (data[1].2) + 30 + 35) / 4
  if regression_eq mean_x = mean_y then
    data[1].2
  else 
    -1 -- Just a placeholder to handle the case; more logic might be needed.

theorem find_m_correct :
  find_m [(2, 15), (3, 20), (4, 30), (5, 35)] (λ x, 7 * x + 0.5) = 20 :=
 sorry

end find_m_correct_l723_723547


namespace half_radius_of_circle_y_l723_723264

theorem half_radius_of_circle_y
  (area_eq : ∀ r_x r_y : ℝ, π * r_x^2 = π * r_y^2)
  (circ_x : ∃ r_x : ℝ, 2 * π * r_x = 12 * π)
  :
  ∃ r_y : ℝ, (r_y / 2 = 3) :=
by
  -- Conditions and definitions
  let r_x := 6
  have r_x_calc : 2 * π * r_x = 12 * π := by simp [r_x, mul_comm, mul_assoc]
  have area_eq_calc : π * r_x^2 = π * 6^2 := by rw [←mul_assoc, pow_two r_x]
  let r_y := 6
  have r_y_calc : r_y = 6 := by simp
  have half_r_y_calc : r_y / 2 = 3 := by simp [r_y_calc]
  -- Proof
  exact ⟨r_y, half_r_y_calc⟩

end half_radius_of_circle_y_l723_723264


namespace prime_factors_30_fac_eq_10_l723_723132

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723132


namespace range_of_a_l723_723671

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * x - 1) / (x - 1) < 0 ↔ x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0) → 
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  intro h
  sorry

end range_of_a_l723_723671


namespace forester_trees_planted_l723_723837

theorem forester_trees_planted (initial_trees : ℕ) (tripled_trees : ℕ) (trees_planted_monday : ℕ) (trees_planted_tuesday : ℕ) :
  initial_trees = 30 ∧ tripled_trees = 3 * initial_trees ∧ trees_planted_monday = tripled_trees - initial_trees ∧ trees_planted_tuesday = trees_planted_monday / 3 →
  trees_planted_monday + trees_planted_tuesday = 80 :=
by
  sorry

end forester_trees_planted_l723_723837


namespace polynomial_example_l723_723927

noncomputable theory
open_locale classical

-- Define the conditions and the polynomial function
constant f : ℕ → ℤ
constants p : ℕ → ℤ
constant m : ℕ

-- Assumptions: f is a polynomial with integer coefficients,
-- and p(1), p(2), ..., p(m) are m different positive integers.
axiom f_poly : ∃ g : polynomial ℤ, ∀ x : ℕ, f x = polynomial.eval x g
axiom p_pos : ∀ k : ℕ, 1 ≤ k → k ≤ m → p k > 0
axiom p_diff : ∀ i j : ℕ, 1 ≤ i → i ≤ m → 1 ≤ j → j ≤ m → i ≠ j → p i ≠ p j

-- Objective: Prove that f(p k) = p k for k = 1, 2, ..., m
theorem polynomial_example :
  ∃ (f : ℕ → ℤ), (∃ g : polynomial ℤ, ∀ x : ℕ, f x = polynomial.eval x g)
  ∧ (∀ k : ℕ, 1 ≤ k → k ≤ m → f (p k) = p k) :=
sorry

end polynomial_example_l723_723927


namespace factorial_30_prime_count_l723_723202

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723202


namespace num_prime_factors_30_factorial_l723_723039

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723039


namespace reflected_line_eq_l723_723446

noncomputable def point_symmetric_reflection :=
  ∃ (A : ℝ × ℝ) (B : ℝ × ℝ) (A' : ℝ × ℝ),
  A = (-1 / 2, 0) ∧ B = (0, 1) ∧ A' = (1 / 2, 0) ∧ 
  ∀ (x y : ℝ), 2 * x + y = 1 ↔
  (y - 1) / (0 - 1) = x / (1 / 2 - 0)

theorem reflected_line_eq :
  point_symmetric_reflection :=
sorry

end reflected_line_eq_l723_723446


namespace first_runner_meets_conditions_l723_723764

noncomputable def first_runner_time := 11

theorem first_runner_meets_conditions (T : ℕ) (second_runner_time third_runner_time : ℕ) (meet_time : ℕ)
  (h1 : second_runner_time = 4)
  (h2 : third_runner_time = 11 / 2)
  (h3 : meet_time = 44)
  (h4 : meet_time % T = 0)
  (h5 : meet_time % second_runner_time = 0)
  (h6 : meet_time % third_runner_time = 0) : 
  T = first_runner_time :=
by
  sorry

end first_runner_meets_conditions_l723_723764


namespace different_prime_factors_of_factorial_eq_10_l723_723049

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723049


namespace triangle_sides_inradius_is_one_l723_723860

-- Define the sides of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- We assume that the triangle is a right triangle
theorem triangle_sides (a b c : ℕ) : a^2 + b^2 = c^2 :=
  by simp [a, b, c]

-- Define the area of the triangle
def area (a b : ℕ) : ℝ := (1/2) * a * b

-- Define the area in terms of the inradius
def area_inradius (r : ℝ) : ℝ := 2*r + (3/2)*r + (5/2)*r

-- Prove that the radius of the inscribed circle is 1
theorem inradius_is_one (r : ℝ) : 
  r = 1 :=
by
  -- Given the triangle is a right triangle with sides 3, 4, and 5
  have h1 : 3^2 + 4^2 = 5^2 := by simp [triangle_sides 3 4 5]
  -- Given the area of the triangle is 6
  have h2 : area 3 4 = 6 := by norm_num [area 3 4]
  -- The area in terms of the inradius is also 6
  have h3 : area_inradius r = 6 := by norm_num [area_inradius r]
  sorry

end triangle_sides_inradius_is_one_l723_723860


namespace all_lines_can_be_paired_perpendicular_l723_723642

noncomputable def can_pair_perpendicular_lines : Prop := 
  ∀ (L1 L2 : ℝ), 
    L1 ≠ L2 → 
      ∃ (m : ℝ), 
        (m * L1 = -1/L2 ∨ L1 = 0 ∧ L2 ≠ 0 ∨ L2 = 0 ∧ L1 ≠ 0)

theorem all_lines_can_be_paired_perpendicular : can_pair_perpendicular_lines :=
sorry

end all_lines_can_be_paired_perpendicular_l723_723642


namespace smallest_rel_prime_1155_l723_723782

open Nat

theorem smallest_rel_prime_1155 : ∃ n : ℕ, n > 1 ∧ gcd n 1155 = 1 ∧ ∀ m : ℕ, m > 1 ∧ gcd m 1155 = 1 → n ≤ m := 
sorry

end smallest_rel_prime_1155_l723_723782


namespace stratified_sampling_expected_females_l723_723856

noncomputable def sample_size := 14
noncomputable def total_athletes := 44 + 33
noncomputable def female_athletes := 33
noncomputable def stratified_sample := (female_athletes * sample_size) / total_athletes

theorem stratified_sampling_expected_females :
  stratified_sample = 6 :=
by
  sorry

end stratified_sampling_expected_females_l723_723856


namespace second_flower_shop_groups_l723_723673

theorem second_flower_shop_groups (n : ℕ) (h1 : n ≠ 0) (h2 : n ≠ 9) (h3 : Nat.lcm 9 n = 171) : n = 19 := 
by
  sorry

end second_flower_shop_groups_l723_723673


namespace grid_possible_configuration_l723_723290

theorem grid_possible_configuration (m n : ℕ) (hm : m > 100) (hn : n > 100) : 
  ∃ grid : ℕ → ℕ → ℕ,
  (∀ i j, grid i j = (if i > 0 then grid (i - 1) j else 0) + 
                       (if i < m - 1 then grid (i + 1) j else 0) + 
                       (if j > 0 then grid i (j - 1) else 0) + 
                       (if j < n - 1 then grid i (j + 1) else 0)) 
  ∧ (∃ i j, grid i j ≠ 0) 
  ∧ m > 14 
  ∧ n > 14 := 
sorry

end grid_possible_configuration_l723_723290


namespace triangle_area_45_45_90_l723_723459

/--
A right triangle has one angle of 45 degrees, and its hypotenuse measures 10√2 inches.
Prove that the area of the triangle is 50 square inches.
-/
theorem triangle_area_45_45_90 {x : ℝ} (h1 : 0 < x) (h2 : x * Real.sqrt 2 = 10 * Real.sqrt 2) : 
  (1 / 2) * x * x = 50 :=
sorry

end triangle_area_45_45_90_l723_723459


namespace round_32_465_not_32_47_round_32_469_32_47_round_32_4701_32_47_round_32_474999_32_47_round_32_473_32_47_l723_723791

noncomputable def round_half_up (x : ℝ) (precision : ℕ) : ℝ :=
  let factor := 10 ^ precision
  in (Real.ceil ((x * factor) - 0.5)) / factor

theorem round_32_465_not_32_47 :
  round_half_up 32.465 2 ≠ 32.47 :=
by
  sorry

theorem round_32_469_32_47 :
  round_half_up 32.469 2 = 32.47 :=
by
  sorry

theorem round_32_4701_32_47 :
  round_half_up 32.4701 2 = 32.47 :=
by
  sorry

theorem round_32_474999_32_47 :
  round_half_up 32.474999 2 = 32.47 :=
by
  sorry

theorem round_32_473_32_47 :
  round_half_up 32.473 2 = 32.47 :=
by
  sorry

end round_32_465_not_32_47_round_32_469_32_47_round_32_4701_32_47_round_32_474999_32_47_round_32_473_32_47_l723_723791


namespace different_prime_factors_of_factorial_eq_10_l723_723043

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723043


namespace prime_factors_of_30_factorial_l723_723110

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723110


namespace prime_factors_30_fac_eq_10_l723_723137

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723137


namespace units_digit_of_expression_l723_723522

theorem units_digit_of_expression : 
  (7 * 13 * 1957 - 7^4) % 10 = 6 := by
  have h1 : 7 % 10 = 7 := rfl
  have h2 : 13 % 10 = 3 := rfl
  have h3 : 1957 % 10 = 7 := rfl
  have h4 : 7^4 % 10 = 1 := by
    -- 7^1 % 10 = 7
    -- 7^2 % 10 = 49 % 10 = 9
    -- 7^3 % 10 = 343 % 10 = 3
    -- 7^4 % 10 = 2401 % 10 = 1, using modulo patterns of 7 
    sorry
  have h5 : (7 * 13) % 10 = 21 % 10 := by rw [h1, h2]; exact rfl
  have h5 : 21 % 10 = 1 := rfl -- units digit of 7*13 is 1
  have h6 : (1 * 1957) % 10 = 7 := by rw [h3]; exact rfl
  show (7 - 1) % 10 = 6 := by rw [h4]; exact rfl

end units_digit_of_expression_l723_723522


namespace limit_proof_l723_723507

open Real

-- Define the conditions
axiom sin_6x_approx (x : ℝ) : ∀ ε > 0, x ≠ 0 → |sin (6 * x) / (6 * x) - 1| < ε
axiom arctg_2x_approx (x : ℝ) : ∀ ε > 0, x ≠ 0 → |arctan (2 * x) / (2 * x) - 1| < ε

-- State the limit proof problem
theorem limit_proof :
  (∃ ε > 0, ∀ x : ℝ, |x| < ε → x ≠ 0 →
  |(x * sin (6 * x)) / (arctan (2 * x)) ^ 2 - (3 / 2)| < ε) :=
sorry

end limit_proof_l723_723507


namespace new_triangle_acute_l723_723267

theorem new_triangle_acute (a b c x : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : c > 0) (h3 : x > 0) :
  let a' := a + x
      b' := b + x
      c' := c + x
  in a'^2 + b'^2 > c'^2 :=
by
  sorry

end new_triangle_acute_l723_723267


namespace num_prime_factors_of_30_l723_723222

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723222


namespace granary_circumference_correct_l723_723287

namespace Granary

noncomputable def granary_circumference (h χ : ℝ) (V : ℝ) (π : ℝ) (hu_to_chi³ : ℝ) : ℝ :=
2 * π * (sqrt (V * 3 / (π * h)))

theorem granary_circumference_correct :
  ∀ (h χ : ℝ) (V : ℝ) (π : ℝ) (hu_to_chi³ : ℝ),
  h = 40 / 3 →
  V = 2000 * hu_to_chi³ →
  π = 3 →
  hu_to_chi³ = 1.62 →
  granary_circumference h χ V π hu_to_chi³ = 5 + 4/10 :=
by
  intros h χ V π hu_to_chi³ h_def V_def π_def hu_def
  sorry

end Granary

end granary_circumference_correct_l723_723287


namespace rachel_remaining_pictures_l723_723813

theorem rachel_remaining_pictures 
  (p1 p2 p_colored : ℕ)
  (h1 : p1 = 23)
  (h2 : p2 = 32)
  (h3 : p_colored = 44) :
  (p1 + p2 - p_colored = 11) :=
by
  sorry

end rachel_remaining_pictures_l723_723813


namespace num_prime_factors_30_fac_l723_723061

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723061


namespace focal_length_correct_l723_723380

noncomputable def focal_length_of_ellipse : ℝ :=
  let a := 2
  let b := 1
  2 * real.sqrt (a^2 - b^2)

theorem focal_length_correct : focal_length_of_ellipse = 2 * real.sqrt 3 := by
  sorry

end focal_length_correct_l723_723380


namespace job_3_pay_per_hour_l723_723308

def total_earnings_first_two_jobs (h1 h2 : ℕ) (pay1 pay2 : ℕ) : ℕ :=
  h1 * pay1 + h2 * pay2

noncomputable def total_earnings_third_job (total_earnings : ℕ) (total_earnings_first_two_jobs : ℕ) : ℕ :=
  total_earnings - total_earnings_first_two_jobs

noncomputable def hourly_rate (total_earnings : ℕ) (total_hours : ℕ) : ℕ :=
  total_earnings / total_hours

theorem job_3_pay_per_hour :
  ∀ (h1 h2 h3 : ℕ) (pay1 pay2 total_days total_earnings : ℕ),
  h1 = 3 → pay1 = 7 → h2 = 2 → pay2 = 10 → h3 = 4 → total_days = 5 → total_earnings = 445 →
  let earnings_per_day_first_two_jobs := total_earnings_first_two_jobs h1 h2 pay1 pay2,
      work_hours_third_job_five_days := h3 * total_days,
      total_earnings_first_two_jobs_five_days := earnings_per_day_first_two_jobs * total_days,
      earnings_third_job_five_days := total_earnings_third_job total_earnings total_earnings_first_two_jobs_five_days,
      pay_per_hour_third_job := hourly_rate earnings_third_job_five_days work_hours_third_job_five_days
  in pay_per_hour_third_job = 12 :=
by 
  sorry

end job_3_pay_per_hour_l723_723308


namespace arithmetic_sequence_sum_first_15_terms_l723_723285

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def term_sum_eq (a : ℕ → ℝ) (n m : ℕ) (sum_val : ℝ) : Prop :=
  a n + a m = sum_val

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a 1 + a n)

-- Theorem
theorem arithmetic_sequence_sum_first_15_terms (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a)
  (h_term_sum : term_sum_eq a 3 13 12) : sum_of_first_n_terms a 15 = 90 :=
by
  sorry

end arithmetic_sequence_sum_first_15_terms_l723_723285


namespace orthocentric_tetrahedron_common_perpendiculars_l723_723685

theorem orthocentric_tetrahedron_common_perpendiculars 
  {T : Tetrahedron}
  (h1 : T.is_orthocentric) :
  ∃ (H : Point), (∀ (e1 e2 : Line), T.is_opposite_edge e1 e2 → T.common_perpendicular e1 e2 H) :=
sorry

end orthocentric_tetrahedron_common_perpendiculars_l723_723685


namespace surface_area_of_circumscribed_sphere_l723_723811

-- Define side length of the tetrahedron
def side_length : ℝ := 2

-- Define the circumradius calculation formula for a regular tetrahedron
def circumradius (a : ℝ) : ℝ := (a * Real.sqrt 3) / 3

-- Define the surface area formula of a sphere given the radius
def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- Problem statement: Prove that the surface area of the circumscribed sphere is 16π/3
theorem surface_area_of_circumscribed_sphere : surface_area (circumradius side_length) = (16 * Real.pi) / 3 :=
by 
  have r := circumradius side_length
  unfold surface_area
  unfold circumradius
  simp [Real.sqrt, Real.pi]
  sorry

end surface_area_of_circumscribed_sphere_l723_723811


namespace problem_2009_floor_l723_723888

theorem problem_2009_floor :
  let factorial (n : ℕ) := if n = 0 then 1 else n * factorial (n - 1)
  ⨁ (2010! + 2007!) = (2010 * 2009 * 2008! + (2007 / 2008) * 2008!)
  ∨ (2009! + 2008!) = (2009 * 2008! + 2008!)
  ∨ 2007! = (2007 * 2006!)
  ∨ 2010! = (2010 * 2009 * 2008!)
  ∨ (2007 / 2008 * 2008!)
  ∨ (2010! + 2007!)/ (2009! + 2008!)
  ∨ (Math.floor ((2010! + 2007!) / (2009! + 2008!)))
  ∧ 2009 = 9

end problem_2009_floor_l723_723888


namespace cubes_not_touching_tin_foil_volume_l723_723402

-- Definitions for the conditions given
variables (l w h : ℕ)
-- Condition 1: Width is twice the length
def width_twice_length := w = 2 * l
-- Condition 2: Width is twice the height
def width_twice_height := w = 2 * h
-- Condition 3: The adjusted width for the inner structure in inches
def adjusted_width := w = 8

-- The theorem statement to prove the final answer
theorem cubes_not_touching_tin_foil_volume : 
  width_twice_length l w → 
  width_twice_height w h →
  adjusted_width w →
  l * w * h = 128 :=
by
  intros h1 h2 h3
  sorry

end cubes_not_touching_tin_foil_volume_l723_723402


namespace geometric_sequence_sum_l723_723480

theorem geometric_sequence_sum (a r : ℝ) 
    (h1 : geom_sum (1 + r) 1499 a = 300) 
    (h2 : geom_sum (1 + r) 2999 a = 570) : 
    geom_sum (1 + r) 4499 a = 813 := 
sorry

end geometric_sequence_sum_l723_723480


namespace quadratic_two_distinct_real_roots_l723_723748

theorem quadratic_two_distinct_real_roots:
  ∃ (α β : ℝ), α ≠ β ∧ (∀ x : ℝ, x * (x - 2) = x - 2 ↔ x = α ∨ x = β) :=
by
  sorry

end quadratic_two_distinct_real_roots_l723_723748


namespace prime_factors_30_fac_eq_10_l723_723131

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723131


namespace ratio_of_students_to_professors_l723_723286

theorem ratio_of_students_to_professors (total : ℕ) (students : ℕ) (professors : ℕ)
  (h1 : total = 40000) (h2 : students = 37500) (h3 : total = students + professors) :
  students / professors = 15 :=
by
  sorry

end ratio_of_students_to_professors_l723_723286


namespace problem_1_problem_2_l723_723546

noncomputable def sequence_a (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) = a n - (a n)^2

def seq_positive (a : ℕ → ℝ) : Prop := ∀ n, a n > 0

def sum_of_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in (finset.range n), a i

theorem problem_1 {a : ℕ → ℝ}
  (h_seq : sequence_a a)
  (h_pos : seq_positive a) :
  ∀ n ≥ 2, a n ≤ 1 / (n + 2) :=
sorry

theorem problem_2 {a : ℕ → ℝ}
  (h_seq : sequence_a a)
  (h_pos : seq_positive a) 
  (S : ℕ → ℝ := sum_of_terms a) :
  ∀ n ≥ 2, S (2 * n) - S (n - 1) < real.log 2 :=
sorry

end problem_1_problem_2_l723_723546


namespace total_volume_combined_l723_723475

def radius (d : ℝ) : ℝ := d / 2

def volume_cylinder (r h : ℝ) : ℝ := real.pi * r^2 * h

def volume_cone (r h : ℝ) : ℝ := (1 / 3) * real.pi * r^2 * h

theorem total_volume_combined (d₁ d₂ : ℝ) (h₁ h₂ : ℝ) 
  (hd₁ : d₁ = 12) (hd₂ : d₂ = 12) (hh₁ : h₁ = 8) (hh₂ : h₂ = 5) 
  : volume_cylinder (radius d₁) h₁ + volume_cone (radius d₂) h₂ = 348 * real.pi :=
by
  rw [hd₁, hd₂, hh₁, hh₂]
  simp [radius, volume_cylinder, volume_cone, real.pi]
  norm_num
  sorry

end total_volume_combined_l723_723475


namespace num_prime_factors_30_factorial_l723_723248

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723248


namespace different_prime_factors_of_factorial_eq_10_l723_723048

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723048


namespace vectors_collinear_l723_723432

theorem vectors_collinear
  (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ)
  (h_a : a = (3, 4, -1))
  (h_b : b = (2, -1, 1))
  (c1 : ℝ × ℝ × ℝ) (c2 : ℝ × ℝ × ℝ)
  (h_c1 : c1 = (6 * 3 - 3 * 2, 6 * 4 - 3 * (-1), 6 * (-1) - 3 * 1))
  (h_c2 : c2 = (2 - 2 * 3, -1 - 2 * 4, 1 - 2 * (-1)))
  : ∃ γ : ℝ, c1 = (γ * c2.1, γ * c2.2, γ * c2.3) :=
begin
  sorry
end

end vectors_collinear_l723_723432


namespace positive_diff_solutions_l723_723411

theorem positive_diff_solutions (s : ℝ) (h : s ≠ -6) :
  let f := λ s : ℝ, (s^2 - 5 * s - 24) / (s + 6) = 3 * s + 10 in
  let sol1 := (-21/2 : ℝ) in
  let sol2 := (-4 : ℝ) in
  abs (sol1 - sol2) = 6.5 :=
by
  sorry

end positive_diff_solutions_l723_723411


namespace odometer_difference_l723_723675

theorem odometer_difference 
  (initial_reading : ℝ) (final_reading : ℝ)
  (h_initial : initial_reading = 212.3)
  (h_final : final_reading = 584.3) :
  final_reading - initial_reading = 372 :=
by 
  rw [h_initial, h_final]
  exact rfl

end odometer_difference_l723_723675


namespace factorial_prime_factors_l723_723155

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723155


namespace periodic_g_exists_sum_g_bounded_l723_723312

-- Conditions
variable (n : ℕ) (n_gt : n > 1) 
variable (a : Fin n → ℕ)

-- Definitions
def seq_b (i : Fin n) : ℕ := 
  if 0 < i ∧ i < n 
  then (List.sum (List.map a ((List.finRange n).erase i))) / (n - 1)
  else 0

def f (a : Fin n → ℕ) : Fin n → ℕ :=
  λ i, seq_b i

def g (m : ℕ) (a: Fin n → ℕ) : ℕ :=
  (f^[m] a).toList.eraseDup.length

-- Statement 1
theorem periodic_g_exists : ∃ k0 : ℕ, ∀ m ≥ k0, g m a = g (m + 1) a := sorry

-- Statement 2
theorem sum_g_bounded (k : ℕ) : 
  ∃ C : ℝ, ∑ m in Finset.range k, (g m a)/(m * (m + 1)) < C := sorry

end periodic_g_exists_sum_g_bounded_l723_723312


namespace num_prime_factors_30_factorial_l723_723034

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723034


namespace house_of_lords_possible_partition_l723_723367

-- Define the graph as a structure with vertices and edges, where vertices represent Lords and edges represent mutual enmity.
structure Graph (V : Type) :=
  (E : V → V → Prop)
  (symm : symmetric E)

-- Definition stating that each vertex has a degree of at most 3
def degree_at_most_three {V : Type} (G : Graph V) : Prop :=
  ∀ v : V, (finset.univ.filter (G.E v)).card ≤ 3

-- Definition for the bipartite property with an additional constraint
def almost_bipartite {V : Type} (G : Graph V) (A B : finset V) : Prop :=
  ∀ v ∈ A, (finset.univ.filter (λ u, u ∈ A ∧ G.E v u)).card ≤ 1 ∧
  ∀ u ∈ B, (finset.univ.filter (λ v, v ∈ B ∧ G.E u v)).card ≤ 1

-- Functional property to represent if such partitions A, B exist
def exists_partition_with_at_most_one_enemy {V : Type} (G : Graph V) : Prop :=
  ∃ (A B : finset V), (finset.univ = A ∪ B) ∧ (finset.disjoint A B) ∧ almost_bipartite G A B

-- Main theorem statement
theorem house_of_lords_possible_partition {V : Type} (G : Graph V) (hdeg : degree_at_most_three G) :
  exists_partition_with_at_most_one_enemy G :=
sorry

end house_of_lords_possible_partition_l723_723367


namespace number_of_prime_factors_thirty_factorial_l723_723010

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723010


namespace revenue_increase_l723_723427

theorem revenue_increase (n : ℕ) (C P : ℝ) 
  (h1 : n * P = 1.20 * C) : 
  (0.95 * n * P) = 1.14 * C :=
by
  sorry

end revenue_increase_l723_723427


namespace max_area_triangle_l723_723554

variable {a b x y : ℝ}
variable {F1 F2 P : ℝ × ℝ}
axiom hyp_foci : F1 = (-sqrt (a^2 + b^2), 0) ∧ F2 = (sqrt (a^2 + b^2), 0)
axiom hyp_hyperbola : 0 < a ∧ 0 < b ∧ (x^2 / a^2 - y^2 / b^2 = 1)
axiom hyp_foci_distance : dist F1 F2 = 2
axiom point_on_right_branch : P.1 > 0
axiom distance_relation : dist P F1 = 2 * dist P F2

theorem max_area_triangle : 
  ∃ (area : ℝ), area = 4 / 3 :=
by
  sorry

end max_area_triangle_l723_723554


namespace max_distance_midpoint_chord_directrix_l723_723589

theorem max_distance_midpoint_chord_directrix
  (A B : ℝ × ℝ)
  (F : ℝ × ℝ := (1, 0))
  (directrix : ℝ := -1)
  (λ : ℝ)
  (hλ : (1 / 3) ≤ λ ∧ λ ≤ 3)
  (hλ_ne_one : λ ≠ 1)
  (hA : A.2 ^ 2 = 4 * A.1)
  (hB : B.2 ^ 2 = 4 * B.1)
  (h1 : A.1 + 1 = λ * (B.1 + 1))
  (h2 : A.1 = λ ^ 2 * B.1) :
  dist (((A.1 + B.1) / 2, (A.2 + B.2) / 2)) (-1, ((A.2 + B.2) / 2)) = 8 / 3 :=
sorry

end max_distance_midpoint_chord_directrix_l723_723589


namespace blueberries_cartons_proof_l723_723870

def total_needed_cartons : ℕ := 26
def strawberries_cartons : ℕ := 10
def cartons_to_buy : ℕ := 7

theorem blueberries_cartons_proof :
  strawberries_cartons + cartons_to_buy + 9 = total_needed_cartons :=
by
  sorry

end blueberries_cartons_proof_l723_723870


namespace third_function_is_symmetric_l723_723967

noncomputable def is_symmetric_about_line (f g : ℝ → ℝ) (line : ℝ × ℝ → Prop) : Prop :=
  ∀ (x : ℝ), line (x, f x) → line (g x, -x)

theorem third_function_is_symmetric {φ : ℝ → ℝ} (h_inv : ∀ y, φ (φ⁻¹ y) = y)
    (h_inv2 : ∀ x, φ⁻¹ (φ x) = x) :
  is_symmetric_about_line φ⁻¹ (λ x, -φ⁻¹ (-x)) (λ p, p.1 + p.2 = 0) :=
by
  sorry

end third_function_is_symmetric_l723_723967


namespace curve_has_two_extreme_values_l723_723375

def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

theorem curve_has_two_extreme_values : 
  (∃ x : ℝ, (15 * x^4 - 15 * x^2 = 0) ∧ (∃ x, f' x = 0) ∧ local_extr (∂^2 / ∂x^2 f x = 60 * x^3 - 30 * x) (local_max x = -1 ∧ local_min x = 1)) → 2 := by
sorry

end curve_has_two_extreme_values_l723_723375


namespace prime_factors_of_30_factorial_l723_723181

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723181


namespace min_value_in_interval_l723_723575

theorem min_value_in_interval {a : ℝ} (h : 0 < a ∧ a < 1) : 
  ∃ (x ∈ Ioo 0 1), is_local_min_on (λ x, (1 / 2) * x^2 - a * real.log x + 1) (Ioo 0 1) x :=
sorry

end min_value_in_interval_l723_723575


namespace num_prime_factors_30_factorial_l723_723036

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723036


namespace num_prime_factors_30_factorial_l723_723031

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723031


namespace prime_factors_of_30_factorial_l723_723172

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723172


namespace number_of_prime_factors_of_30_factorial_l723_723125

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723125


namespace factor_theorem_example_l723_723656

theorem factor_theorem_example (m n : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x^2 + a * x + b) = x^3 - m * x^2 + n * x - 15 → (x = -3 → m = -14/3 - n / 3)):
begin
  sorry
end

end factor_theorem_example_l723_723656


namespace measure_angle_B1BM_l723_723299

-- Define the given conditions
variables {A B C A1 B1 C1 M : Type}
variables (angle_ABC : real) (angle_B_120 : angle_ABC = 120)
variables (is_angle_bisector_AA1 : angle_bisector A B C A1)
variables (is_angle_bisector_BB1 : angle_bisector B A C B1)
variables (is_angle_bisector_CC1 : angle_bisector C A B C1)
variables (A1B1_intersect_CC1_at_M : ∃ M, segment_intersection A1 B1 C1 = M)
-- Define the conclusion
theorem measure_angle_B1BM :
  measure_angle B1 B M = 30 :=
sorry

end measure_angle_B1BM_l723_723299


namespace problem_l723_723651

theorem problem (a b : ℝ) (h1 : 
  let f1 := λ x : ℝ, ((x + a) * (x + b) * (x + 10)) / ((x + 4)^2),
      has_three_distinct_roots : ∃ xa xb xc : ℝ, xa ≠ xb ∧ xb ≠ xc ∧ xa ≠ xc ∧ f1 xa = 0 ∧ f1 xb = 0 ∧ f1 xc = 0
  in has_three_distinct_roots) 
  (h2 : 
  let f2 := λ x : ℝ, ((x + 2 * a) * (x + 4) * (x + 5)) / ((x + b) * (x + 10)),
      has_one_distinct_root : ∃ xa : ℝ, ∀ xb : ℝ, f2 xb = 0 → xa = xb
  in has_one_distinct_root) : 
  100 * a + b = 205 := sorry

end problem_l723_723651


namespace constant_term_in_expansion_l723_723538

theorem constant_term_in_expansion (n : ℕ) (h_pos : n > 0) : 
  (∃ r : ℕ, n = 3 * r) ↔ (∃ r : ℕ, (x - 1 / x^2)^n = (C n r) * (-1)^r * x^0) :=
by {
  sorry
}

end constant_term_in_expansion_l723_723538


namespace solve_equation_in_nat_l723_723696

theorem solve_equation_in_nat {x y : ℕ} :
  (x - 1) / (1 + (x - 1) * y) + (y - 1) / (2 * y - 1) = x / (x + 1) →
  x = 2 ∧ y = 2 :=
by
  sorry

end solve_equation_in_nat_l723_723696


namespace num_prime_factors_30_factorial_l723_723037

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723037


namespace estimate_accuracy_with_larger_sample_l723_723296

theorem estimate_accuracy_with_larger_sample
    (fixed_population : Type)
    (sample : Type)
    [fintype fixed_population]
    (population_size : ℕ)
    (sample_size : ℕ)
    (estimate_accuracy : fixed_population → sample → Prop) :
    ∀ (n : ℕ), n > sample_size → estimate_accuracy fixed_population sample :=
sorry

end estimate_accuracy_with_larger_sample_l723_723296


namespace num_prime_factors_30_fac_l723_723072

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723072


namespace factorial_prime_factors_l723_723166

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723166


namespace different_prime_factors_of_factorial_eq_10_l723_723050

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723050


namespace total_rainfall_november_l723_723622

def rainfall_odd_days (n : ℕ) : ℕ :=
  2 * n + 2

def rainfall_even_days (n : ℕ) : ℕ :=
  22 - 2 * n

def sum_rainfall_first_15_days : ℕ :=
  (List.range 8).sumBy (λ n => rainfall_odd_days n) +
  (List.range 7).sumBy (λ n => rainfall_even_days n)

def rainfall_prime_days (n : ℕ) : ℕ :=
  5 + 3 * n

def rainfall_non_prime_days (n : ℕ) : ℕ :=
  (λ r, r * 3 / 2)^[n] 4

def sum_rainfall_next_12_days : ℕ :=
  (List.Iota.filter Nat.prime [|16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30|]).sumBy (λ n, rainfall_prime_days n) +
  (List.Iota.filter (λ n, ¬Nat.prime n) [|16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30|]).sumBy (λ n, rainfall_non_prime_days n)

theorem total_rainfall_november : sum_rainfall_first_15_days + sum_rainfall_next_12_days = 405.03125 := by
  sorry

end total_rainfall_november_l723_723622


namespace arithmetic_mean_increase_by_20_l723_723265

variable {a : Fin 10 → ℤ} -- represents a function from Finite (10) to Integers

theorem arithmetic_mean_increase_by_20 (a : Fin 10 → ℤ) :
    let S := ∑ i in Finset.univ, a i
    let mean := S / 10
    let new_mean := (S + 200) / 10
    new_mean = mean + 20 :=
by
  sorry

end arithmetic_mean_increase_by_20_l723_723265


namespace num_prime_factors_of_30_l723_723234

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723234


namespace find_p_l723_723294

theorem find_p (p : ℝ) : 
  let A := (3, 15)
  let B := (15, 0)
  let C := (0, p)
  let Q := (0, 15)
  let area_ABC := 36 in
  (1/2 * 15 * (3 + 15) - (1/2 * 3 * (15 - p) + 1/2 * 15 * p)) = area_ABC → p = 12.75 := 
by 
  intros
  sorry

end find_p_l723_723294


namespace polynomial_prime_divisors_l723_723483

theorem polynomial_prime_divisors (f : Polynomial ℤ) (h_nonconst : f.degree > 0) (k : ℕ) (h_k_pos : 0 < k)
  (h_property : ∀ (p : ℕ), Nat.Prime p → (f.eval p).factors.toFinset.card ≤ k) :
  ∃ (n : ℤ) (m : ℕ), f = Polynomial.C n * Polynomial.X ^ m :=
sorry

end polynomial_prime_divisors_l723_723483


namespace fraction_value_l723_723493

theorem fraction_value :
  (∑ i in Finset.range (2020+1), (2020+1 - i) / i) / 
  (∑ i in Finset.range (2021+1), 1 / (i+2)) = 2021 :=
by
  sorry

end fraction_value_l723_723493


namespace path_length_equals_perimeter_l723_723808

theorem path_length_equals_perimeter (A B C C0 A0 B0 A1 B1 C1: Point) 
  (h_midC0: C0 = midpoint A B)
  (h_midA0: A0 = midpoint B C)
  (h_midB0: B0 = midpoint C A)
  (h_footA1: A1 = foot A BC)
  (h_footB1: B1 = foot B CA)
  (h_footC1: C1 = foot C AB) :
  path_length A0 B1 C0 A1 B0 C1 A0 = perimeter A B C :=
sorry

end path_length_equals_perimeter_l723_723808


namespace factorial_prime_factors_l723_723094

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723094


namespace sin_neg_pi_div_two_l723_723508

theorem sin_neg_pi_div_two : Real.sin (-π / 2) = -1 := by
  -- Define the necessary conditions
  let π_in_deg : ℝ := 180 -- π radians equals 180 degrees
  have sin_neg_angle : ∀ θ : ℝ, Real.sin (-θ) = -Real.sin θ := sorry -- sin(-θ) = -sin(θ) for any θ
  have sin_90_deg : Real.sin (π_in_deg / 2) = 1 := sorry -- sin(90 degrees) = 1

  -- The main statement to prove
  sorry

end sin_neg_pi_div_two_l723_723508


namespace find_p_plus_q_p_plus_q_is_31_l723_723526

noncomputable def average_abs_diff_sum (l : List ℕ) : ℚ :=
  if h : l.length = 8 ∧ l.nodup ∧ l.toFinset = Finset.range 1 9 then
    let perms := (List.permutations l).toFinset in
    let sums := perms.map (λ perm,
      |perm.nthLe 0 (by simp [Finset.mem_univ]) - perm.nthLe 1 (by simp [Finset.mem_univ])| +
      |perm.nthLe 2 (by simp [Finset.mem_univ]) - perm.nthLe 3 (by simp [Finset.mem_univ])| +
      |perm.nthLe 4 (by simp [Finset.mem_univ]) - perm.nthLe 5 (by simp [Finset.mem_univ])| +
      |perm.nthLe 6 (by simp [Finset.mem_univ]) - perm.nthLe 7 (by simp [Finset.mem_univ])|) in
    (sums.sum / perms.card : ℚ)
  else
    0

theorem find_p_plus_q : average_abs_diff_sum [1, 2, 3, 4, 5, 6, 7, 8] = (28 / 3 : ℚ) :=
begin
  sorry
end

theorem p_plus_q_is_31 : 31 = 28 + 3 :=
by norm_num

end find_p_plus_q_p_plus_q_is_31_l723_723526


namespace prime_factors_30_fac_eq_10_l723_723145

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723145


namespace slightly_used_crayons_correct_l723_723759

def total_crayons : ℕ := 120
def new_crayons : ℕ := total_crayons / 3
def broken_crayons : ℕ := (total_crayons * 20) / 100
def slightly_used_crayons : ℕ := total_crayons - new_crayons - broken_crayons

theorem slightly_used_crayons_correct : slightly_used_crayons = 56 := sorry

end slightly_used_crayons_correct_l723_723759


namespace prob1_prob2_l723_723885

theorem prob1:
  (-(1:ℤ)^2 + (64:ℤ)^(1/3 : ℚ) - (-2:ℤ) * ((9:ℕ).sqrt : ℤ)) = 9 :=
by
  sorry

theorem prob2: 
  (-(1/2 : ℚ) * (-2)^2 - ((-1/8 : ℚ)^(1/3 : ℚ)) + ((-1/2 : ℚ)^2).sqrt) = -1 :=
by
  sorry

end prob1_prob2_l723_723885


namespace centroids_and_orthocenters_loci_are_rays_l723_723442

variables (S : Point) (T : TrihedralAngle) (P : FamilyOfParallelPlanes)

-- Definition of the trihedral angle with vertex S
def is_trihedral_angle_with_vertex (T : TrihedralAngle) (S : Point) : Prop :=
  T.vertex = S

-- Definition of centroids' locus as a ray from S.
def locus_of_centroids_is_ray_from (P : FamilyOfParallelPlanes) (T : TrihedralAngle) (S : Point) : Prop :=
  ∀ (t : Triangle), (t ∈ (P ∩ T.faces)) → (∃ (r : Ray), r.origin = S ∧ t.centroid ∈ r)

-- Definition of orthocenters' locus as a ray from S.
def locus_of_orthocenters_is_ray_from (P : FamilyOfParallelPlanes) (T : TrihedralAngle) (S : Point) : Prop :=
  ∀ (t : Triangle), (t ∈ (P ∩ T.faces)) → (∃ (r : Ray), r.origin = S ∧ t.orthocenter ∈ r)

-- Theorem statement combining both centroids and orthocenters loci properties.
theorem centroids_and_orthocenters_loci_are_rays (S : Point) (T : TrihedralAngle) (P : FamilyOfParallelPlanes)
  (hT : is_trihedral_angle_with_vertex T S) :
  locus_of_centroids_is_ray_from P T S ∧ locus_of_orthocenters_is_ray_from P T S :=
sorry

end centroids_and_orthocenters_loci_are_rays_l723_723442


namespace inscribed_circle_exists_l723_723531

theorem inscribed_circle_exists
  (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (r1 r2 r3 r4 d : ℝ) 
  (h1 : r1 + r3 = r2 + r4) 
  (h2 : r1 + r3 < d) : 
  ∃ k : unit_circle, inscribed_circle_quadrilateral := 
sorry

end inscribed_circle_exists_l723_723531


namespace boxes_needed_l723_723717

theorem boxes_needed (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) (h1 : total_muffins = 95) (h2 : muffins_per_box = 5) (h3 : available_boxes = 10) : 
  total_muffins - (available_boxes * muffins_per_box) / muffins_per_box = 9 :=
by
  sorry

end boxes_needed_l723_723717


namespace prime_factors_of_30_factorial_l723_723101

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723101


namespace probability_billy_bobbi_same_number_l723_723876

theorem probability_billy_bobbi_same_number :
  let count_multiples (k n : ℕ) := n / k
  let multiples_18 := 11
  let multiples_24 := 8
  let multiples_72 := 2
  let total_combinations := multiples_18 * multiples_24
  let probability := (multiples_72 : ℚ) / total_combinations
  probability = 1 / 44 :=
by
  intros count_multiples multiples_18 multiples_24 multiples_72 total_combinations probability
  have count_multiples : ∀ (k n : ℕ), ℕ := λ k n, n / k
  have multiples_18 : ℕ := count_multiples 18 199
  have multiples_24 : ℕ := count_multiples 24 199
  have multiples_72 : ℕ := count_multiples 72 199
  have total_combinations : ℕ := multiples_18 * multiples_24
  have probability : ℚ := (multiples_72 : ℚ) / total_combinations
  exact Eq.refl (1 / 44)

end probability_billy_bobbi_same_number_l723_723876


namespace problem_statement_l723_723859

theorem problem_statement
  (ABC : Triangle)
  (O : Point)
  (O' : Point)
  (M : Point)
  (circumscribed : ∀ (X : Point), is_on_circumcircle ABC O X)
  (inscribed : ∀ (X : Point), is_on_incircle ABC O' X)
  (midpoint_arc : is_midpoint_of_arc M (arc_not_containing C)):
  dist M A = dist M B ∧ dist M A = dist M O' :=
by
  sorry

end problem_statement_l723_723859


namespace area_quadrilateral_parabola_l723_723668

-- Definition of a parabola with focus F and directrix ℓ
-- Assuming necessary definitions and lemmas to support the geometrical properties and calculations

theorem area_quadrilateral_parabola 
  (P : Type) [Parabola P] (F : P) (ℓ : Line) 
  (A B : P) (D C : Point)
  (h1 : Line_through F A B)
  (h2 : Perpendicular D A ℓ)
  (h3 : Perpendicular C B ℓ)
  (h4 : distance A B = 20)
  (h5 : distance C D = 14) :
  area_quadrilateral A B C D = 140 := by
  sorry

end area_quadrilateral_parabola_l723_723668


namespace prime_factors_of_30_factorial_l723_723174

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723174


namespace forester_planted_total_trees_l723_723836

theorem forester_planted_total_trees :
  let initial_trees := 30 in
  let monday_trees := 3 * initial_trees in
  let new_trees_monday := monday_trees - initial_trees in
  let tuesday_trees := (1 / 3) * new_trees_monday in
  new_trees_monday + tuesday_trees = 80 :=
by
  repeat sorry

end forester_planted_total_trees_l723_723836


namespace folded_triangle_length_squared_l723_723450

-- Definitions used in the conditions
def equilateral_triangle_side_length : ℝ := 10
def distance_from_B_to_fold_point : ℝ := 3

-- Main theorem statement
theorem folded_triangle_length_squared :
  ∃ (L : ℝ), L = (37 / 4) ∧
    let s := equilateral_triangle_side_length,
    let d := distance_from_B_to_fold_point,
    eq_tri_ABC : equilateral_triangle ABC s, -- Assuming equilateral_triangle is defined
    ∀ A B C A' PA QA PQ : ℝ,
    A' = d →
    PA = (s - (s * d / 6)) / 2 →
    QA = (s * d / 6 - d) / 2 →
    PQ^2 = PA^2 + QA^2 - 2 * PA * QA * Math.cos (π / 3) := sorry

end folded_triangle_length_squared_l723_723450


namespace sine_ratio_comparison_l723_723417

theorem sine_ratio_comparison : (Real.sin (1 * Real.pi / 180) / Real.sin (2 * Real.pi / 180)) < (Real.sin (3 * Real.pi / 180) / Real.sin (4 * Real.pi / 180)) :=
sorry

end sine_ratio_comparison_l723_723417


namespace fraction_value_l723_723494

theorem fraction_value :
  (∑ i in Finset.range (2020+1), (2020+1 - i) / i) / 
  (∑ i in Finset.range (2021+1), 1 / (i+2)) = 2021 :=
by
  sorry

end fraction_value_l723_723494


namespace num_prime_factors_30_factorial_l723_723251

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723251


namespace eval_floor_neg_sqrt_l723_723899

theorem eval_floor_neg_sqrt : (Int.floor (-Real.sqrt (64 / 9)) = -3) := sorry

end eval_floor_neg_sqrt_l723_723899


namespace number_of_distinct_prime_factors_30_fact_l723_723001

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l723_723001


namespace find_trapezoid_bases_l723_723388

-- Define the conditions of the isosceles trapezoid
variables {AD BC : ℝ}
variables (h1 : ∀ (A B C D : ℝ), is_isosceles_trapezoid A B C D ∧ intersects_at_right_angle A B C D)
variables (h2 : ∀ {A B C D : ℝ}, trapezoid_area A B C D = 12)
variables (h3 : ∀ {A B C D : ℝ}, trapezoid_height A B C D = 2)

-- Prove the bases AD and BC are 8 and 4 respectively under the given conditions
theorem find_trapezoid_bases (AD BC : ℝ) : 
  AD = 8 ∧ BC = 4 :=
  sorry

end find_trapezoid_bases_l723_723388


namespace num_integer_exponent_terms_l723_723724

def binomial_expansion_integer_exponents (x : ℝ) (n : ℕ) (h_arith_seq : ∀ (k : ℕ), (C k) = (a + k * d)) : ℕ :=
  sorry

theorem num_integer_exponent_terms (x : ℝ) (h_arith_seq : ∀ (k : ℕ), 
  let C n := binom n k * (x^((n-k)/2) / (2 * x^(k/4))):
  (C 0 = a) ∧ (C 1 = a + d) ∧ (C 2 = a + 2*d)) : 
  binomial_expansion_integer_exponents x 8 h_arith_seq = 3 :=
sorry

end num_integer_exponent_terms_l723_723724


namespace min_lambda_mu_equilateral_triangle_l723_723940

theorem min_lambda_mu_equilateral_triangle
  (A B C P : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space P]
  (HABC : ∀ (x y : Type), dist x y = 1) 
  (HAP1 : dist A P = 1) 
  (HAP_decomp : ∃ λ μ : ℝ, vector_repr P = λ * vector_repr B + μ * vector_repr C)
  (Hangle : interior_angle A B C = π / 3) :
  ( ∃ λ μ : ℝ, HAP_decomp ∧ (λ + μ) ≥ -2 * sqrt(3) / 3) ∧
  ∀ λ μ : ℝ, HAP_decomp → (λ + μ) ≥ -2 * sqrt(3) / 3 :=
by
  sorry

end min_lambda_mu_equilateral_triangle_l723_723940


namespace days_with_equal_tues_and_thurs_l723_723849

-- Define that for a month with 30 days there are the same number of Tuesdays and Thursdays
def tues_and_thurs_equal_days (first_day : Nat) : Bool :=
  let days := Array.mkArray 7 4 -- Initially, 4 occurrences of each day in the first 28 days
  let days := days.set! (first_day % 7) (days.get! (first_day % 7) + 1) -- Increment the first extra day
  let days := days.set! ((first_day + 1) % 7) (days.get! ((first_day + 1) % 7) + 1) -- Increment the second extra day
  days.get! 2 == days.get! 4 -- Compare the occurrences of Tuesdays (index 2) and Thursdays (index 4)

theorem days_with_equal_tues_and_thurs : 
  (Finset.range 7).filter tues_and_thurs_equal_days).card = 5 :=
by
  sorry

end days_with_equal_tues_and_thurs_l723_723849


namespace find_angle4_l723_723929

theorem find_angle4 (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180) 
  (h2 : angle3 = angle4) 
  (h3 : angle3 + angle4 = 70) :
  angle4 = 35 := 
by 
  sorry

end find_angle4_l723_723929


namespace boxes_needed_l723_723719

theorem boxes_needed (total_muffins available_boxes muffins_per_box : ℕ) (h1 : total_muffins = 95) 
  (h2 : available_boxes = 10) (h3 : muffins_per_box = 5) : 
  ((total_muffins - (available_boxes * muffins_per_box)) / muffins_per_box) = 9 := 
by
  -- the proof will be constructed here
  sorry

end boxes_needed_l723_723719


namespace number_of_prime_factors_thirty_factorial_l723_723014

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723014


namespace sin_inverse_eq_l723_723422

theorem sin_inverse_eq (t : ℝ) (k l : ℤ) :
  (sin t ≠ 0) ∧ (sin (2 * t) ≠ 0) ∧ (sin (4 * t) ≠ 0) ∧ 
  (Real.arcsin t - Real.arcsin (2 * t) = Real.arcsin (4 * t)) →
  ∃ k, t = (Real.pi / 7) * (2 * k + 1) ∧ k ≠ 7 * l + 3 := 
by
  sorry

end sin_inverse_eq_l723_723422


namespace power_function_value_at_4_l723_723611

-- Define the power function and necessary conditions
def power_function (x : ℝ) (n : ℝ) : ℝ := x^n

theorem power_function_value_at_4 
  (n : ℝ)
  (h1 : (2 : ℝ)^n = (Real.sqrt 2) / 2)
  (h2 : ∀ x, power_function x n = x ^ n) :
  power_function 4 n = 1 / 2 :=
by
  sorry

end power_function_value_at_4_l723_723611


namespace trig_functions_symmetry_l723_723384

theorem trig_functions_symmetry :
  ∀ k₁ k₂ : ℤ,
  (∃ x, x = k₁ * π / 2 + π / 3 ∧ x = k₂ * π + π / 3) ∧
  (¬ ∃ x, (x, 0) = (k₁ * π / 2 + π / 12, 0) ∧ (x, 0) = (k₂ * π + 5 * π / 6, 0)) :=
by
  sorry

end trig_functions_symmetry_l723_723384


namespace num_prime_factors_30_factorial_l723_723255

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723255


namespace factorial_prime_factors_l723_723085

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723085


namespace A_div_B_l723_723482

noncomputable def A : ℝ := 
  ∑' (n : ℕ) in ({1, 7, 19, 23} \u {n | n % 5 = 0 ∧ n % 2 = 1}) ∪ (-({11, 17} \u {n | n % 5 != 0 ∧ n % 2 = 1})),
  (1 : ℝ) / (n ^ 2)

noncomputable def B : ℝ := 
  ∑' (n : ℕ) in ({n | n % 5 = 0 ∧ n % 2 = 1}),
  (1 : ℝ) / (n ^ 2) * (if (n / 5) % 2 = 0 then 1 else -1)

theorem A_div_B : A / B = 26 :=
by
  sorry

end A_div_B_l723_723482


namespace prime_factors_of_30_factorial_l723_723170

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723170


namespace prime_factors_of_30_factorial_l723_723179

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723179


namespace finitely_many_non_composite_uniform_coverings_l723_723821

def uniform_covering (n : ℕ) := multiset (finset (fin n))

def non_composite (S : uniform_covering n) : Prop :=
  ∀ (A B : uniform_covering n), S ≠ A ∪ B

theorem finitely_many_non_composite_uniform_coverings (n : ℕ) (hn : n ≥ 1) :
  {S : uniform_covering n | non_composite S}.finite :=
sorry

end finitely_many_non_composite_uniform_coverings_l723_723821


namespace prime_factors_of_30_factorial_l723_723108

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723108


namespace prime_factors_of_30_factorial_l723_723178

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723178


namespace max_pairs_l723_723925

theorem max_pairs  : 
  ∃ (k : ℕ), 
    (∀ (S : ℕ), 
      (∀ (i : ℕ), 1 ≤ i → i ≤ k → S ≥ k * (2 * k + 1)) ∧
      S ≤ (5019 - k) * k / 2) ∧
    k = 1003 :=
begin
  sorry
end

end max_pairs_l723_723925


namespace num_prime_factors_30_fac_l723_723076

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723076


namespace triangle_area_l723_723616

theorem triangle_area
  (a b c : ℝ)
  (h1 : b^2 - b * c - 2 * c^2 = 0)
  (h2 : a = Real.sqrt 6)
  (h3 : Real.cos (Real.acos (ofRat (7 / 8))) = 7 / 8) :
  let S := (1 / 2) * b * c * (Real.sin (Real.acos (7 / 8)))
  in S = Real.sqrt 15 / 2 :=
sorry

end triangle_area_l723_723616


namespace percentage_of_400_that_results_in_224_point_5_l723_723901

-- Let x be the unknown percentage of 400
variable (x : ℝ)

-- Condition: x% of 400 plus 45% of 250 equals 224.5
def condition (x : ℝ) : Prop := (400 * x / 100) + (250 * 45 / 100) = 224.5

theorem percentage_of_400_that_results_in_224_point_5 : condition 28 :=
by
  -- proof goes here
  sorry

end percentage_of_400_that_results_in_224_point_5_l723_723901


namespace find_angle_D_l723_723558

noncomputable def measure.angle_A := 80
noncomputable def measure.angle_B := 30
noncomputable def measure.angle_C := 20

def sum_angles_pentagon (A B C : ℕ) := 540 - (A + B + C)

theorem find_angle_D
  (A B C E F : ℕ)
  (hA : A = measure.angle_A)
  (hB : B = measure.angle_B)
  (hC : C = measure.angle_C)
  (h_sum_pentagon : A + B + C + D + E + F = 540)
  (h_triangle : D + E + F = 180) :
  D = 130 :=
by
  sorry

end find_angle_D_l723_723558


namespace g_at_8_l723_723732

-- Define the functional properties of g
axiom functional_property (g : ℝ → ℝ) : ∀ x y : ℝ, g(x) + g(3 * x + y) + 7 * x * y = g(4 * x - 2 * y) + 3 * x^2 + 2

-- Prove that g(8) = -46 given the functional property
theorem g_at_8 (g : ℝ → ℝ) (h : functional_property g) : g 8 = -46 := by
  sorry

end g_at_8_l723_723732


namespace average_fuel_efficiency_l723_723466

-- Definitions for the conditions
def distance_one_way : ℝ := 150
def sedan_mileage : ℝ := 25
def minivan_normal_mileage : ℝ := 20
def minivan_decreased_mileage : ℝ := minivan_normal_mileage - 5
def total_distance : ℝ := distance_one_way * 2

-- Theorem statement
theorem average_fuel_efficiency :
  let total_fuel_used := (distance_one_way / sedan_mileage) + (distance_one_way / minivan_decreased_mileage)
  in total_distance / total_fuel_used = 18.75 :=
sorry

end average_fuel_efficiency_l723_723466


namespace number_of_subsets_of_A_l723_723964

open Finset

def A : Finset ℕ := {n ∈ range 1 8 | 8 - n ∈ range 1 8}

theorem number_of_subsets_of_A : 2 ^ A.card = 128 := by sorry

end number_of_subsets_of_A_l723_723964


namespace problem_statement_l723_723490

theorem problem_statement : 
  ((∑ i in Finset.range 2020, (2021 - i) * (1 / (i+1))) /
  (∑ j in Finset.range 2020, 1 / (j + 2))) = 2021 := 
by
  sorry

end problem_statement_l723_723490


namespace geometric_locus_of_tangency_l723_723467

-- Given two intersecting planes P1 and P2
variables {P1 P2 : set (ℝ × ℝ × ℝ)}

def is_plane (P : set (ℝ × ℝ × ℝ)) : Prop := ∃ (a b c d : ℝ), ∀ (x y z : ℝ), 
  P (x, y, z) ↔ a * x + b * y + c * z = d

def intersecting_planes (P1 P2 : set (ℝ × ℝ × ℝ)) : Prop := 
  ∃ (L : set (ℝ × ℝ × ℝ)), L = {p | ∀ (ε ∈ P1) (ζ ∈ P2), p ∈ P1 ∧ p ∈ P2}

-- Indicates that there exists a line L that is the intersection of planes P1 and P2
axiom planes_intersect (hP1 : is_plane P1) (hP2 : is_plane P2) : ∃ (L : set (ℝ × ℝ × ℝ)), 
  (∀ p, p ∈ L ↔ p ∈ P1 ∧ p ∈ P2) 

-- Defining the sphere of radius r
def sphere (O : ℝ × ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ × ℝ) := 
  {P | ∥P - O∥ = r}

-- Define the geometric locus of points of tangency
def geometric_locus (L : set (ℝ × ℝ × ℝ)) (r : ℝ) (P1 P2 : set (ℝ × ℝ × ℝ)) : Prop :=
  ∃ (C1 C2 : set (ℝ × ℝ × ℝ)), parallel C1 L ∧ parallel C2 L ∧
  (∀ (S : set (ℝ × ℝ × ℝ)), S ∈ sphere O r → (S ∩ P1 ⊆ C1) ∧ (S ∩ P2 ⊆ C2))

-- The theorem
theorem geometric_locus_of_tangency 
  (P1 P2 : set (ℝ × ℝ × ℝ)) (r : ℝ)
  (hP1 : is_plane P1) (hP2 : is_plane P2) :
  ∃ L, planes_intersect hP1 hP2 → 
  ∃ (C1 C2 : set (ℝ × ℝ × ℝ)), geometric_locus L r P1 P2 :=
begin
  sorry
end

end geometric_locus_of_tangency_l723_723467


namespace sqrt_sum_eq_ten_l723_723900

theorem sqrt_sum_eq_ten :
  Real.sqrt ((5 - 4*Real.sqrt 2)^2) + Real.sqrt ((5 + 4*Real.sqrt 2)^2) = 10 := 
by 
  sorry

end sqrt_sum_eq_ten_l723_723900


namespace log_evaluation_l723_723603

theorem log_evaluation (x : ℝ) (h : Real.logBase 16 (x - 5) = 1 / 2) : 
  Real.logBase 64 x = 0.525 :=
by
  sorry

end log_evaluation_l723_723603


namespace find_value_of_nested_function_l723_723960

def f (x : ℝ) : ℝ :=
  if x ≥ 2 then real.sqrt x else 3 - x

theorem find_value_of_nested_function :
  f(f(-1)) = 2 :=
by
  sorry

end find_value_of_nested_function_l723_723960


namespace polygon_sides_l723_723270

theorem polygon_sides (sum_except_one : ℝ) (h : sum_except_one = 3330) : 
  ∃ (n : ℕ), n = 21 :=
by
  use 21
  sorry

end polygon_sides_l723_723270


namespace f_x_plus_1_minus_f_x_l723_723596

def f (x : ℝ) : ℝ := 8 ^ x

theorem f_x_plus_1_minus_f_x (x : ℝ) : f (x + 1) - f x = 7 * f x :=
by
  sorry

end f_x_plus_1_minus_f_x_l723_723596


namespace tom_800th_day_l723_723767

theorem tom_800th_day (tom_birth_day : ℕ) (day_of_week : ℕ) (birth_day : day_of_week = 2) (days_in_week : day_of_week = 7) :
  (tom_birth_day + 800) % days_in_week = 4 := 
sorry

end tom_800th_day_l723_723767


namespace number_of_prime_factors_thirty_factorial_l723_723016

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723016


namespace number_of_prime_factors_thirty_factorial_l723_723022

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723022


namespace log_21_not_computable_l723_723555

theorem log_21_not_computable (log2 log3 : ℝ) (h1 : log2 = 0.3010) (h2 : log3 = 0.4771) :
  ¬∃ (log21 : ℝ), (log21 = log3 + log 7) ∧ (log7 ∈ {log2, log3}) :=
by sorry

end log_21_not_computable_l723_723555


namespace badminton_members_l723_723625

def members_total : ℕ := 30
def members_tennis : ℕ := 19
def members_none : ℕ := 2
def members_both : ℕ := 7
def members_at_least_one : ℕ := members_total - members_none

theorem badminton_members : ∃ (B : ℕ), B = 16 :=
by
  let B := members_at_least_one + members_both - members_tennis
  use B
  have h1 : members_at_least_one = 28 := by rfl
  have h2 : B = 16 := by linarith
  exact h2

end badminton_members_l723_723625


namespace harmonic_sum_ratio_l723_723486

theorem harmonic_sum_ratio :
  (∑ k in Finset.range (2020 + 1), (2021 - k) / k) /
  (∑ k in Finset.range (2021 - 1), 1 / (k + 2)) = 2021 :=
by
  sorry

end harmonic_sum_ratio_l723_723486


namespace factorial_30_prime_count_l723_723197

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723197


namespace monkey_hop_distance_l723_723448

theorem monkey_hop_distance
    (total_height : ℕ)
    (slip_back : ℕ)
    (hours : ℕ)
    (reach_time : ℕ)
    (hop : ℕ)
    (H1 : total_height = 19)
    (H2 : slip_back = 2)
    (H3 : hours = 17)
    (H4 : reach_time = 16 * (hop - slip_back) + hop)
    (H5 : total_height = reach_time) :
    hop = 3 := by
  sorry

end monkey_hop_distance_l723_723448


namespace sum_of_decimals_is_one_l723_723687

-- Define digits for each decimal place
def digit_a : ℕ := 2
def digit_b : ℕ := 3
def digit_c : ℕ := 2
def digit_d : ℕ := 2

-- Define the decimal numbers with these digits
def decimal1 : Rat := (digit_b * 10 + digit_a) / 100
def decimal2 : Rat := (digit_d * 10 + digit_c) / 100
def decimal3 : Rat := (2 * 10 + 2) / 100
def decimal4 : Rat := (2 * 10 + 3) / 100

-- The main theorem that states the sum of these decimals is 1
theorem sum_of_decimals_is_one : decimal1 + decimal2 + decimal3 + decimal4 = 1 := by
  sorry

end sum_of_decimals_is_one_l723_723687


namespace f_strictly_increasing_l723_723573

noncomputable def f (x : ℝ) := log (4 * x - x^2) / log (1/2)

theorem f_strictly_increasing : ∀ x y : ℝ, 2 ≤ x ∧ x < 4 → 2 ≤ y ∧ y < 4 → x < y → f x < f y :=
by
  sorry

end f_strictly_increasing_l723_723573


namespace range_of_k_l723_723542

theorem range_of_k (f : ℝ → ℝ) (g : ℝ → ℝ) (k : ℝ) 
  (domain_R : ∀ x : ℝ, true) 
  (h_derivative : ∀ x : ℝ, deriv f x ≠ 0)
  (h_equation : ∀ x : ℝ, f (f x - 2017^x) = 2017)
  (h_g_def : ∀ x : ℝ, g x = sin x - cos x - k * x) :
  -∞ < k ∧ k ≤ -1 :=
begin
  sorry
end

end range_of_k_l723_723542


namespace domain_f_f_neg2_domain_f_shift_f_shift_eq_l723_723576

def f (x : ℝ) : ℝ := sqrt(x + 2) + 1 / (x + 1)

theorem domain_f :
  {x : ℝ | x + 2 ≥ 0 ∧ x + 1 ≠ 0} =
  {x : ℝ | x ≥ -2 ∧ x ≠ -1} :=
sorry

theorem f_neg2 : f (-2) = -1 :=
sorry

def f_shift (x : ℝ) : ℝ := sqrt(x + 1) + 1 / x

theorem domain_f_shift :
  {x : ℝ | x + 1 ≥ 0 ∧ x ≠ 0} =
  {x : ℝ | x ≥ -1 ∧ x ≠ 0} :=
sorry

theorem f_shift_eq : ∀ x, f (x - 1) = f_shift x :=
sorry

end domain_f_f_neg2_domain_f_shift_f_shift_eq_l723_723576


namespace cos_square_sum_l723_723557

variable (α β γ : ℝ)

theorem cos_square_sum 
  (h1 : sin α + sin β + sin γ = 0)
  (h2 : cos α + cos β + cos γ = 0) :
  cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 = 3 / 2 := by
  sorry

end cos_square_sum_l723_723557


namespace product_a2_a6_l723_723565

theorem product_a2_a6 : 
  (∃ S : ℕ → ℤ, 
    (∀ n, S n = 4 * n^2 - 10 * n) ∧ 
    (∀ n, ∃ a : ℕ → ℤ, a n = S n - S (n-1)) ∧ 
    (∃ a2 a6 : ℤ, a2 = (λ n : ℕ, (8 * n - 14)) 2 ∧ a6 = (λ n : ℕ, (8 * n - 14)) 6 ∧ a2 * a6 = 68)) :=
  sorry

end product_a2_a6_l723_723565


namespace quadratic_function_solution_l723_723599

theorem quadratic_function_solution (m : ℝ) :
  (m^2 - 2 = 2) ∧ (m + 2 ≠ 0) → m = 2 :=
by
  intro h
  cases h with h1 h2
  have h3 : m^2 = 4 := by linarith
  have h4 : m = 2 ∨ m = -2 := by nlinarith
  cases h4
  · exact h4
  · contradiction

end quadratic_function_solution_l723_723599


namespace integer_count_with_inverse_mod_13_l723_723972

theorem integer_count_with_inverse_mod_13 : 
  (Finset.card {a ∈ Finset.range 13 | Int.gcd a 13 = 1}) = 12 :=
by
  sorry

end integer_count_with_inverse_mod_13_l723_723972


namespace quadratic_double_root_eq1_quadratic_double_root_eq2_l723_723606

theorem quadratic_double_root_eq1 :
  (∃ r : ℝ , ∃ s : ℝ, (r ≠ s) ∧ (
  (1 : ℝ) * r^2 + (-3 : ℝ) * r + (2 : ℝ) = 0 ∧
  (1 : ℝ) * s^2 + (-3 : ℝ) * s + (2 : ℝ) = 0 ∧
  (r = 2 * s ∨ s = 2 * r) 
  )) := 
  sorry

theorem quadratic_double_root_eq2 :
  (∃ a b : ℝ, a ≠ 0 ∧
  ((∃ r : ℝ, (-b / a = 2 + r) ∧ (-6 / a = 2 * r)) ∨ 
  ((-b / a = 2 + 1) ∧ (-6 / a = 2 * 1))) ∧ 
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9))) :=
  sorry

end quadratic_double_root_eq1_quadratic_double_root_eq2_l723_723606


namespace range_of_k_l723_723570

theorem range_of_k (n : ℕ) (k : ℝ) (h : ∃ (x₁ x₂ : ℝ), (x₁ ≠ x₂) ∧ (2n - 1 < x₁) ∧ (x₁ ≤ 2n + 1) ∧ (2n - 1 < x₂) ∧ (x₂ ≤ 2n + 1) ∧ (|x₁ - 2n| = k * real.sqrt x₁) ∧ (|x₂ - 2n| = k * real.sqrt x₂)) :
  0 < k ∧ k ≤ 1 / real.sqrt (2n + 1) :=
  sorry

end range_of_k_l723_723570


namespace repeating_decimal_to_fraction_l723_723903

noncomputable def repeating_decimal := 36 / 99

theorem repeating_decimal_to_fraction : 0.363636... = 4 / 11 :=
by
  sorry

end repeating_decimal_to_fraction_l723_723903


namespace first_shaded_134_l723_723853

-- Define the rectangular board and its numbering
def board_number (rows cols : ℕ) (n : ℕ) : ℕ :=
  (n-1) % cols + 1 

-- Define the sequence of shaded squares
def shaded_sq (i : ℕ) : ℕ :=
  Nat.recOn i
    0 
    (λ n shaded_n, shaded_n + n + 2)

-- Specify the columns to be filled
noncomputable def filled_columns (max_shaded sqs : ℕ) : Finset ℕ :=
  Finset.image (λ n, board_number 12 12 (shaded_sq n)) (Finset.range max_shaded_sqs)

-- Proof statement: we need to prove that the first shaded square is 134 which fills all columns 1-12
theorem first_shaded_134 (H : ∀ n, (filled_columns (n+1).card = Finset.range 12)) : 
  shaded_sq 15 = 134 :=
by 
  sorry

end first_shaded_134_l723_723853


namespace proof_smallest_nonprime_greater_50_no_prime_factors_lt_20_in_range_smallest_nonprime_greater_50_no_prime_factors_lt_20_l723_723322

noncomputable def smallest_nonprime_greater_50_no_prime_factors_lt_20 : ℕ :=
  if h : ∃ m, (m > 50) ∧ (¬ Prime m) ∧ (∀ p, Prime p → p ∣ m → p ≥ 20) then
    Nat.find h
  else 0

theorem proof_smallest_nonprime_greater_50_no_prime_factors_lt_20 :
  smallest_nonprime_greater_50_no_prime_factors_lt_20 = 667 :=
by sorry

theorem in_range_smallest_nonprime_greater_50_no_prime_factors_lt_20 :
  650 < smallest_nonprime_greater_50_no_prime_factors_lt_20 ∧ 
  smallest_nonprime_greater_50_no_prime_factors_lt_20 ≤ 700 :=
by sorry

end proof_smallest_nonprime_greater_50_no_prime_factors_lt_20_in_range_smallest_nonprime_greater_50_no_prime_factors_lt_20_l723_723322


namespace max_episodes_l723_723368

theorem max_episodes (characters : ℕ) (h_characters : characters = 20) :
∃ episodes : ℕ, episodes = 780 :=
by
  use 780
  sorry

end max_episodes_l723_723368


namespace problem_sec_tan_csc_cot_l723_723363

theorem problem_sec_tan_csc_cot:
  ∀ (x : ℝ), (sec x + tan x = 15 / 4) → (∃ p q : ℕ, csc x + cot x = p / q ∧ Nat.gcd p q = 1 ∧ p + q = 21) :=
by
  intros x h_sec_tan
  sorry

end problem_sec_tan_csc_cot_l723_723363


namespace largest_n_unique_k_l723_723773

theorem largest_n_unique_k (n k : ℕ) :
  (frac9_17_lt_frac n (n + k) ∧ frac n (n + k) lt frac8_15) 
  ∧ (∀ (n1 k1 : ℕ), frac9_17_lt_frac n1 (n1 + k1) ∧ frac n1 (n1 + k1) lt frac8_15 
  → (n1 ≤ 136 
  ∧ ((n1 = 136) → (k1 = unique_k))))
  :=
sorry

def frac9_17_lt_frac (a b : ℕ) : Prop := 
  (9:ℚ) / 17 < (a : ℚ) / b

def frac (a b : ℕ) : ℚ :=
  (a : ℚ) / b

def frac8_15 := 
  (8:ℚ) / 15

def unique_k : ℕ :=
  119

end largest_n_unique_k_l723_723773


namespace tetrahedron_volume_from_octahedron_centers_l723_723939

def regular_octahedron_volume (s : ℝ) : ℝ :=
  (1 / 3) * real.sqrt 2 * s ^ 3

def tetrahedron_volume_from_octahedron (V_O : ℝ) : ℝ :=
  V_O / 4

theorem tetrahedron_volume_from_octahedron_centers
  (s : ℝ)
  (h : s = 2 * real.sqrt 2)
  (V_O : ℝ)
  (hV_O : V_O = regular_octahedron_volume s) :
  tetrahedron_volume_from_octahedron V_O = 4 * real.sqrt 2 / 3 :=
by
  rw [h, hV_O]
  sorry

end tetrahedron_volume_from_octahedron_centers_l723_723939


namespace boxes_needed_to_pack_all_muffins_l723_723722

theorem boxes_needed_to_pack_all_muffins
  (total_muffins : ℕ := 95)
  (muffins_per_box : ℕ := 5)
  (available_boxes : ℕ := 10) :
  (total_muffins / muffins_per_box) - available_boxes = 9 :=
by
  sorry

end boxes_needed_to_pack_all_muffins_l723_723722


namespace determine_c_plus_d_l723_723706

theorem determine_c_plus_d (x : ℝ) (c d : ℤ) (h1 : x^2 + 5*x + (5/x) + (1/(x^2)) = 35) (h2 : x = c + Real.sqrt d) : c + d = 5 :=
sorry

end determine_c_plus_d_l723_723706


namespace sum_of_pairs_l723_723777

theorem sum_of_pairs (n : ℕ) :
  (∑ k in finset.range n, (2 * (k + 1))^3) + (∑ k in finset.range n, (-2 * (k + 1))^3) = 0 :=
by
suffices : ∀ k : ℕ, k ∈ finset.range n → (2 * (k + 1))^3 + (-2 * (k + 1))^3 = 0,
  from finset.sum_congr rfl this
suffices : ∀ a : ℤ, a^3 + (-a)^3 = 0,
  from λ k _, this (2 * (k + 1))
λ a, by ring

end sum_of_pairs_l723_723777


namespace compute_expression_l723_723567

theorem compute_expression : 
  let z : ℂ := 1 + complex.i in
  (2 / z) + z ^ 2 = 1 + complex.i :=
by
  let z : ℂ := 1 + complex.i
  sorry

end compute_expression_l723_723567


namespace find_real_medal_in_two_weighings_l723_723795

-- Define what it means to have 9 medals, only one of which is real and the rest are replicas
def medal (n : ℕ) : Type := fin n → ℝ

-- Define the condition that only one medal is heavier than the others
def is_real_medal {n : ℕ} (w : medal n) : Prop :=
  ∃ i, w i > w j ∀ j, i ≠ j

-- Define the scale operation, which checks the weights of two groups of medals
def balance_scale {n : ℕ} (w : medal n) (s1 s2 : finset (fin n)) : Ordering := 
  if finset.sum (λ i, w i) s1 < finset.sum (λ i, w i) s2 then Ordering.lt
  else if finset.sum (λ i, w i) s1 > finset.sum (λ i, w i) s2 then Ordering.gt
  else Ordering.eq

-- Define the problem statement
theorem find_real_medal_in_two_weighings :
  ∀ (w : medal 9), is_real_medal w → ∃ (s1 s2: finset (fin 9)), s1.card = 4 ∧ s2.card = 4 ∧ (balance_scale w s1 s2 ≠ Ordering.eq) ∨ (balance_scale w s1 s2 = Ordering.eq ∧ ∃ i ≠ j, balance_scale w {i} {j}  ≠ Ordering.eq) : sorry

end find_real_medal_in_two_weighings_l723_723795


namespace total_pens_is_55_l723_723994

noncomputable def total_pens_excluding_purple : ℕ :=
  let red_pens := 8 in
  let black_pens := 2 * red_pens in
  let blue_pens := black_pens + 5 in
  let green_pens := (blue_pens / 2 : Float).toNat in
  red_pens + black_pens + blue_pens + green_pens

theorem total_pens_is_55 : total_pens_excluding_purple = 55 :=
by sorry

end total_pens_is_55_l723_723994


namespace number_of_customers_l723_723643

noncomputable def k : ℝ := 2000

def popularity (p c e : ℝ) : ℝ := (k * e) / c

theorem number_of_customers (c1 e1 c2 e2 : ℝ) :
  popularity 50 c1 e1 = 50 ∧ c1 = 400 ∧ e1 = 10 ∧ c2 = 800 ∧ e2 = 20 →
  popularity 50 c2 e2 = 50 :=
by
  intro h,
  cases h with h₁ h₂,
  cases h₂ with h₃ h₄,
  cases h₄ with h₅ h₆,
  have k_value : k = 2000 := rfl,
  simp [popularity, k_value],
  sorry

end number_of_customers_l723_723643


namespace OMN_area_correct_l723_723387

noncomputable def OMN_area_proof (x y : ℝ) (h_line : x - (sqrt 3) * y + (sqrt 3) = 0) (h_ellipse : (x^2 / 9) + (y^2 / 6) = 1) : ℝ :=
  let M N : ℝ × ℝ := sorry -- (x1, y1) and (x2, y2), the points of intersection
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (-sqrt 3, 0)
  let dist_OF := (sqrt 3 - 0) -- Distance OF
  let y_diff : ℝ := |(M.2 - N.2)| -- Absolute distance between y-coordinates of M and N
  let S : ℝ := (1 / 2) * dist_OF * y_diff
  S -- The area of triangle OMN is S

theorem OMN_area_correct (x y : ℝ) (h_line : x - (sqrt 3) * y + (sqrt 3) = 0) (h_ellipse : (x^2 / 9) + (y^2 / 6) = 1) :
  OMN_area_proof x y h_line h_ellipse = (4 * sqrt 3 / 3) :=
  sorry

end OMN_area_correct_l723_723387


namespace quartic_polynomial_exists_l723_723509

theorem quartic_polynomial_exists :
  ∃ (p : ℚ[X]), p.monic ∧ (p.coeff 3 = -2) ∧ (p.coeff 2 = -14) ∧ (p.coeff 1 = 24) ∧ (p.coeff 0 = -12) ∧ 
  (∃ (q : ℚ[X]), q = polynomial.C (3 + sqrt 5) * polynomial.C (3 - sqrt 5) * polynomial.C (-2 - sqrt 7) * polynomial.C (-2 + sqrt 7)) := 
by
  sorry

end quartic_polynomial_exists_l723_723509


namespace vector_addition_l723_723970

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, 3)

-- Stating the problem: proving the sum of vectors a and b
theorem vector_addition : a + b = (3, 4) := 
by 
  -- Proof is not required as per the instructions
  sorry

end vector_addition_l723_723970


namespace inclination_angle_of_pyramid_in_cone_l723_723376

theorem inclination_angle_of_pyramid_in_cone :
  (∃ (l : ℝ) (α β γ : ℝ), 
     l > 0 ∧
     α = 45 ∧ 
     β = 60 ∧ 
     γ = 75 ∧ 
     α + β + γ = 180 ∧ 
     cos (atan (l * sqrt(2 / 34)) / (l * sqrt(34) / 6)) = 1 / sqrt(17)) :=
sorry

end inclination_angle_of_pyramid_in_cone_l723_723376


namespace num_prime_factors_30_factorial_l723_723024

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723024


namespace proof_sin_alpha_plus_pi_over_3_l723_723534

open Real

noncomputable def sin_alpha_plus_pi_over_3 (α : ℝ) : Prop :=
  α ∈ (π / 2, π) ∧ (cos (α - π / 6) + sin α = (4 * sqrt 3) / 5) →
  sin (α + π / 3) = (4 * sqrt 3 - 3) / 10

theorem proof_sin_alpha_plus_pi_over_3 (α : ℝ) : sin_alpha_plus_pi_over_3 α :=
sorry

end proof_sin_alpha_plus_pi_over_3_l723_723534


namespace pigeon_percentage_l723_723281

-- Define the conditions
variables (total_birds : ℕ)
variables (geese swans herons ducks pigeons : ℕ)
variables (h1 : geese = total_birds * 20 / 100)
variables (h2 : swans = total_birds * 30 / 100)
variables (h3 : herons = total_birds * 15 / 100)
variables (h4 : ducks = total_birds * 25 / 100)
variables (h5 : pigeons = total_birds * 10 / 100)

-- Define the target problem
theorem pigeon_percentage (h_total : total_birds = 100) :
  (pigeons * 100 / (total_birds - swans)) = 14 :=
by sorry

end pigeon_percentage_l723_723281


namespace number_of_distinct_prime_factors_30_fact_l723_723000

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l723_723000


namespace different_prime_factors_of_factorial_eq_10_l723_723052

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723052


namespace prob_correct_l723_723896

-- Define the points on the square
def points_on_square : List (ℝ × ℝ) :=
[(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

-- Calculate the Euclidean distance between two points
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Find the count of pairs of points that are one unit apart
def one_unit_apart_count (points : List (ℝ × ℝ)) : Nat :=
(List.length (List.filter (λ (p : (ℝ × ℝ) × (ℝ × ℝ)), euclidean_distance p.1 p.2 = 1) ((List.product points points).filter (λ p, p.1 ≠ p.2))) / 2)

noncomputable def prob_two_points_one_unit_apart : ℝ :=
(one_unit_apart_count points_on_square : ℝ) / (Nat.choose 8 2)

theorem prob_correct : prob_two_points_one_unit_apart = 2 / 7 := by
  sorry

end prob_correct_l723_723896


namespace females_in_coach_class_l723_723335

noncomputable def total_passengers : ℕ := 120
noncomputable def female_percentage : ℚ := 0.40
noncomputable def first_class_percentage : ℚ := 0.10
noncomputable def male_first_class_ratio : ℚ := 1 / 3

theorem females_in_coach_class :
  let total_females := total_passengers * female_percentage,
      first_class_passengers := total_passengers * first_class_percentage,
      coach_class_passengers := total_passengers - first_class_passengers,
      males_first_class := first_class_passengers * male_first_class_ratio,
      females_first_class := first_class_passengers - males_first_class
  in total_females - females_first_class = 40 := by
  -- Proof is omitted
  sorry

end females_in_coach_class_l723_723335


namespace Jake_peach_count_l723_723303

theorem Jake_peach_count (Steven_peaches : ℕ) (Jake_peach_difference : ℕ) (h1 : Steven_peaches = 19) (h2 : Jake_peach_difference = 12) : 
  Steven_peaches - Jake_peach_difference = 7 :=
by
  sorry

end Jake_peach_count_l723_723303


namespace interval_monotonically_decreasing_sin_value_l723_723969

open Real

-- Given definitions
def vec_a (x : ℝ) : ℝ × ℝ := (2 * sin (x - π/4), sqrt 3 * sin x)
def vec_b (x : ℝ) : ℝ × ℝ := (sin (x + π/4), 2 * cos x)
def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

-- 1. Prove the interval where f(x) is monotonically decreasing
theorem interval_monotonically_decreasing (k : ℤ) : 
  ∀ x, (π/3 + k * π ≤ x ∧ x ≤ 5*π/6 + k * π) → 
  (derivative f) x < 0 := sorry

-- 2. Prove sin(2α + π/6) = 23/25 given f(α/2) = 2/5
theorem sin_value (α : ℝ) (h : f (α / 2) = 2 / 5) :
  sin (2 * α + π / 6) = 23 / 25 := sorry

end interval_monotonically_decreasing_sin_value_l723_723969


namespace inverse_matrix_of_M_l723_723588

noncomputable def M (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, a; 3, b]

def e : Matrix (Fin 2) Unit ℝ := !![1; -1]

def λ1 : ℝ := -1

theorem inverse_matrix_of_M (a b : ℝ)
  (h1 : M a b ⬝ e = λ1 • e)
  (ha : a = 2)
  (hb : b = 2) :
  inverse (M a b) = !![-1 / 2, 1 / 2; 3 / 4, -1 / 4] := by
  sorry

end inverse_matrix_of_M_l723_723588


namespace factorial_30_prime_count_l723_723195

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723195


namespace forester_planted_total_trees_l723_723835

theorem forester_planted_total_trees :
  let initial_trees := 30 in
  let monday_trees := 3 * initial_trees in
  let new_trees_monday := monday_trees - initial_trees in
  let tuesday_trees := (1 / 3) * new_trees_monday in
  new_trees_monday + tuesday_trees = 80 :=
by
  repeat sorry

end forester_planted_total_trees_l723_723835


namespace different_prime_factors_of_factorial_eq_10_l723_723041

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723041


namespace min_value_f_l723_723540

theorem min_value_f (x y : ℝ) (hx : 0 ≤ x) (hxi : x ≤ π) (hy : 0 ≤ y) (hyi : y ≤ 1)
  : ∃ (m : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ π → ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 1 → 
  (2*y - 1) * Real.sin x + (1 - y) * Real.sin ((1 - y) * x) ≥ m) ∧ m = 0 :=
by 
  use [0]
  intros x hx1 hx2 y hy1 hy2 
  sorry

end min_value_f_l723_723540


namespace rectangle_quadrilateral_inequality_l723_723889

theorem rectangle_quadrilateral_inequality 
  (a b c d : ℝ)
  (h_a : 0 < a) (h_a_bound : a < 3)
  (h_b : 0 < b) (h_b_bound : b < 4)
  (h_c : 0 < c) (h_c_bound : c < 3)
  (h_d : 0 < d) (h_d_bound : d < 4) :
  25 ≤ ((3 - a)^2 + a^2 + (4 - b)^2 + b^2 + (3 - c)^2 + c^2 + (4 - d)^2 + d^2) ∧
  ((3 - a)^2 + a^2 + (4 - b)^2 + b^2 + (3 - c)^2 + c^2 + (4 - d)^2 + d^2) < 50 :=
by 
  sorry

end rectangle_quadrilateral_inequality_l723_723889


namespace feathers_per_boa_l723_723330

-- Define variables and conditions.
variables {num_tail_feathers_per_flamingo : ℕ} (h1 : num_tail_feathers_per_flamingo = 20)
variables {percentage_pluckable : ℚ} (h2 : percentage_pluckable = 0.25)
variables {num_boys : ℕ} (h3 : num_boys = 12)
variables {num_flamingoes : ℕ} (h4 : num_flamingoes = 480)

-- Calculate and prove the number of feathers per boa
theorem feathers_per_boa :
  let feathers_per_flamingo := percentage_pluckable * num_tail_feathers_per_flamingo,
      total_feathers := num_flamingoes * feathers_per_flamingo,
      feathers_per_boa := total_feathers / num_boys
  in feathers_per_boa = 200 :=
by
  sorry

end feathers_per_boa_l723_723330


namespace num_prime_factors_30_factorial_l723_723026

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723026


namespace sum_of_intercepts_l723_723814

theorem sum_of_intercepts (x₀ y₀ : ℕ) (hx₀ : 4 * x₀ ≡ 2 [MOD 25]) (hy₀ : 5 * y₀ ≡ 23 [MOD 25]) 
  (hx_cond : x₀ < 25) (hy_cond : y₀ < 25) : x₀ + y₀ = 28 :=
  sorry

end sum_of_intercepts_l723_723814


namespace sine_angle_of_rhombus_l723_723520

-- Definitions and conditions
variables {α : ℝ} {A : ℝ}
variables {AB BC CD DA : ℝ} (K L : Point) (midpointK : K = midpoint AB) (midpointL : L = midpoint CD)
variables (angle_CKD : anglebetween (line CD) K = α)
variables (rhombus_properties : AB = BC ∧ BC = CD ∧ CD = DA ∧ DA = AB)

-- Theorem statement 
theorem sine_angle_of_rhombus (H : ∃ (angles : ℝ), angle (line AD) (line AB) = A) :
  sin A = (3 / 4) * tan α :=
by
  sorry

end sine_angle_of_rhombus_l723_723520


namespace length_of_chord_l723_723607

theorem length_of_chord (x y : ℝ) 
  (h1 : (x - 1)^2 + y^2 = 4) 
  (h2 : x + y + 1 = 0) 
  : ∃ (l : ℝ), l = 2 * Real.sqrt 2 := by
  sorry

end length_of_chord_l723_723607


namespace factorial_prime_factors_l723_723165

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723165


namespace excircle_inequality_l723_723348

variables {a b c : ℝ} -- The sides of the triangle

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2 -- Definition of semiperimeter

noncomputable def excircle_distance (p a : ℝ) : ℝ := p - a -- Distance from vertices to tangency points

theorem excircle_inequality (a b c : ℝ) (p : ℝ) 
    (h1 : p = semiperimeter a b c) : 
    (excircle_distance p a) + (excircle_distance p b) > p := 
by
    -- Placeholder for proof
    sorry

end excircle_inequality_l723_723348


namespace day_of_week_1801_11_20_l723_723882

noncomputable def is_leap_year (year : ℕ) : Bool :=
  if year % 4 = 0 then
    if year % 100 = 0 then
      year % 400 = 0
    else
      true
  else
    false

def day_of_week (year : ℕ) (month : ℕ) (day : ℕ) : String :=
  -- This function should be implemented to calculate the day of the week for the given date.
  -- For now, we'll leave it as a placeholder.
  sorry

theorem day_of_week_1801_11_20 :
  day_of_week 2021 11 20 = "Saturday" →
  (∀ year, is_leap_year year = (year % 4 = 0 ∧ (year % 100 ≠ 0 ∨ year % 400 = 0))) →
  day_of_week 1801 11 20 = "Wednesday" :=
begin
  -- proof to be completed
  sorry
end

end day_of_week_1801_11_20_l723_723882


namespace number_of_prime_factors_thirty_factorial_l723_723017

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723017


namespace two_digit_unit_greater_than_ten_count_l723_723629

theorem two_digit_unit_greater_than_ten_count : 
  (∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 10 > n / 10)) → set.count n = 36 := sorry

end two_digit_unit_greater_than_ten_count_l723_723629


namespace parts_B_total_l723_723407

noncomputable def parts_produced_by_B : ℕ :=
let N := 600 -- total parts calculated
let parts_together := (N * (12 / 5)) / 8 -- parts produced together initially
let parts_A := (N / 12) * (12 / 5) -- parts produced by A alone in the initial duration
let parts_B := parts_together - parts_A + 420 -- total parts by B in the initial phase + additional 420
in parts_B

theorem parts_B_total :
  let N := 600 in
  let parts_together := (N * (12 / 5)) / 8 in
  let parts_A := (N / 12) * (12 / 5) in
  let parts_B := parts_together - parts_A + 420 in
  parts_B = 480 := 
by
  unfold parts_produced_by_B
  intro
  sorry

end parts_B_total_l723_723407


namespace boxes_needed_l723_723720

theorem boxes_needed (total_muffins available_boxes muffins_per_box : ℕ) (h1 : total_muffins = 95) 
  (h2 : available_boxes = 10) (h3 : muffins_per_box = 5) : 
  ((total_muffins - (available_boxes * muffins_per_box)) / muffins_per_box) = 9 := 
by
  -- the proof will be constructed here
  sorry

end boxes_needed_l723_723720


namespace largest_n_unique_k_l723_723771

theorem largest_n_unique_k (n : ℕ) (h : ∃ k : ℕ, (9 / 17 : ℚ) < n / (n + k) ∧ n / (n + k) < (8 / 15 : ℚ) ∧ ∀ k' : ℕ, ((9 / 17 : ℚ) < n / (n + k') ∧ n / (n + k') < (8 / 15 : ℚ)) → k' = k) : n = 72 :=
sorry

end largest_n_unique_k_l723_723771


namespace least_n_ergetic_l723_723592

def is_n_ergetic (n k : ℕ) : Prop :=
  ∀ (coloring : Fin (k + 1) → Bool), ∃ (a : Fin (k + 1) → Fin (k + 1)) (h: ∀ i, a i ∈ coloring i = coloring (Fin (k + 1)), Σ i : Fin n, coloring (a i) = coloring i ∧ (∑ (i : Fin n), a i) ∈ coloring ((a i + 1) % (k + 1))

theorem least_n_ergetic (n : ℕ) (hn : 0 < n) : 
  ∃ k, is_n_ergetic n k ∧ ∀ k', (k' < k → ¬ is_n_ergetic n k') := by
  sorry

end least_n_ergetic_l723_723592


namespace cats_problem_l723_723998

theorem cats_problem :
  let total_cats := 16 in
  let white_cats := 2 in
  let black_cats := total_cats * 25 / 100 in
  let grey_cats := total_cats - black_cats - white_cats in
  grey_cats = 10 :=
by
  let total_cats := 16
  let white_cats := 2
  let black_cats := total_cats * 25 / 100
  let grey_cats := total_cats - black_cats - white_cats
  show grey_cats = 10
  sorry

end cats_problem_l723_723998


namespace class_size_incorrect_l723_723479

theorem class_size_incorrect (A G B : Finset ℕ) (hA : A.card = 20) (hG : G.card = 27) (hB : (A ∩ G).card = 8) (hU : (A ∪ G) = Finset.univ) : ∀ (n : ℕ), n ≠ 55 :=
by
  have uniq_union_card : (A ∪ G).card = 20 + 27 - 8
  rw [Finset.union_card, <- hA, <- hG, <- hB]
  sorry

lemma q_is_false : ∃ n, n = (20 + 27 - 8) :=
  have h : ∃ n, n = (20 + 27 - 8) := ⟨20 + 27 - 8, rfl⟩
  exact h

end class_size_incorrect_l723_723479


namespace find_a_l723_723951

theorem find_a (a n : ℕ) (h1 : (2 : ℕ) ^ n = 32) (h2 : (a + 1) ^ n = 243) : a = 2 := by
  sorry

end find_a_l723_723951


namespace variance_of_dataset_l723_723569

noncomputable def dataset : List ℝ := [3, 6, 9, 8, 4]

noncomputable def mean (x : List ℝ) : ℝ :=
  (x.foldr (λ y acc => y + acc) 0) / (x.length)

noncomputable def variance (x : List ℝ) : ℝ :=
  (x.foldr (λ y acc => (y - mean x)^2 + acc) 0) / (x.length)

theorem variance_of_dataset :
  variance dataset = 26 / 5 :=
by
  sorry

end variance_of_dataset_l723_723569


namespace rectangle_minimal_area_l723_723457

theorem rectangle_minimal_area (w l : ℕ) (h1 : l = 3 * w) (h2 : 2 * (l + w) = 120) : l * w = 675 :=
by
  -- Proof will go here
  sorry

end rectangle_minimal_area_l723_723457


namespace find_number_l723_723801

theorem find_number (n : ℝ) : (1 / 2) * n + 6 = 11 → n = 10 := by
  sorry

end find_number_l723_723801


namespace derivative_at_pi_over_4_l723_723579

-- Define the function f and its derivative
def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x

-- State the theorem we want to prove
theorem derivative_at_pi_over_4 : f' (π / 4) = 0 :=
by 
  -- This is the placeholder for the proof
  sorry

end derivative_at_pi_over_4_l723_723579


namespace determine_k_values_l723_723894

theorem determine_k_values (k : ℝ) :
  (∃ a b : ℝ, 3 * a ^ 2 + 6 * a + k = 0 ∧ 3 * b ^ 2 + 6 * b + k = 0 ∧ |a - b| = 1 / 2 * (a ^ 2 + b ^ 2)) → (k = 0 ∨ k = 12) :=
by
  sorry

end determine_k_values_l723_723894


namespace correct_arrangement_divisible_by_12_factorial_correct_arrangement_divisible_by_13_factorial_l723_723327

theorem correct_arrangement_divisible_by_12_factorial
    (deck : List (ℕ × ℕ))
    (h_correct : ∀ i : ℕ, i < 51 → (deck[i].fst = deck[i + 1].fst ∨ deck[i].snd = deck[i + 1].snd))
    (h_top_bottom : deck[0].fst = deck[51].fst ∨ deck[0].snd = deck[51].snd)
    (h_top : deck[0] = (1, 1)) :
  (number_of_correct_arrangements deck) % fact 12 = 0 := sorry

theorem correct_arrangement_divisible_by_13_factorial
    (deck : List (ℕ × ℕ))
    (h_correct : ∀ i : ℕ, i < 51 → (deck[i].fst = deck[i + 1].fst ∨ deck[i].snd = deck[i + 1].snd))
    (h_top_bottom : deck[0].fst = deck[51].fst ∨ deck[0].snd = deck[51].snd)
    (h_top : deck[0] = (1, 1)) :
  (number_of_correct_arrangements deck) % fact 13 = 0 := sorry

end correct_arrangement_divisible_by_12_factorial_correct_arrangement_divisible_by_13_factorial_l723_723327


namespace incorrect_statement_among_A_to_E_is_E_l723_723631

theorem incorrect_statement_among_A_to_E_is_E :
  let A := "Foundational assumptions in mathematics do not necessitate proof"
  let B := "Different mathematical proofs can be structured in various valid sequences"
  let C := "All variables and expressions in a mathematical proof must be explicitly defined beforehand"
  let D := "A mathematical proof cannot have a logically valid conclusion if its premises include falsehoods"
  let E := "A direct proof method is applicable every time there are conflicting premises"
  (¬ (E = "A direct proof method is applicable every time there are conflicting premises" ↔
       A ∨ B ∨ C ∨ D)) ∧
  ("A direct proof method is not applicable every time there are conflicting premises") :=
by
  sorry

end incorrect_statement_among_A_to_E_is_E_l723_723631


namespace larger_square_area_l723_723398

theorem larger_square_area (smaller_square_area grey_triangle_area : ℕ)
    (h1 : smaller_square_area = 16)
    (h2 : grey_triangle_area = 1) : ∃ larger_square_area : ℕ, larger_square_area = 18 :=
by
  exists 18
  sorry

end larger_square_area_l723_723398


namespace number_of_proper_subsets_of_A_range_of_m_l723_723965

def A : Set ℤ := {x : ℤ | -1 ≤ x + 1 ∧ x + 1 ≤ 6}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 1 ≤ x ∧ x < 2 * m + 1}

theorem number_of_proper_subsets_of_A 
  (hA : A = {-2, -1, 0, 1, 2, 3, 4, 5}) :
  ∃ n : ℤ, n = 253 :=
sorry

theorem range_of_m (m : ℝ) 
  (hA : A = {-2, -1, 0, 1, 2, 3, 4, 5})
  (hB : ∀ x ∈ B m, x ∈ A) :
  m < -2 ∨ -1 ≤ m ∧ m ≤ 2 :=
sorry

end number_of_proper_subsets_of_A_range_of_m_l723_723965


namespace sine_product_inequality_l723_723350

theorem sine_product_inequality :
  (1 / 8 : ℝ) < (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) * Real.sin (70 * Real.pi / 180)) ∧
                (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) * Real.sin (70 * Real.pi / 180)) < (1 / 4 : ℝ) :=
sorry

end sine_product_inequality_l723_723350


namespace prime_factors_30_fac_eq_10_l723_723133

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723133


namespace constant_term_in_expansion_l723_723485

theorem constant_term_in_expansion :
  let T : ℤ := ∑ r in finset.range (10), (nat.choose 9 r) * ((2 : ℤ) ^ (9 - r)) * ((-1 : ℤ) ^ r) :=
  (T * (x ^ ((9 - 3 * r) / 2))) = -5376
  ∃ r : ℤ, (9 - 3 * r) / 2 = 0 →
  T = -5376 :=
sorry

end constant_term_in_expansion_l723_723485


namespace factorial_prime_factors_l723_723151

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723151


namespace different_prime_factors_of_factorial_eq_10_l723_723056

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723056


namespace num_prime_factors_of_30_l723_723229

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723229


namespace harmonic_sum_ratio_l723_723488

theorem harmonic_sum_ratio :
  (∑ k in Finset.range (2020 + 1), (2021 - k) / k) /
  (∑ k in Finset.range (2021 - 1), 1 / (k + 2)) = 2021 :=
by
  sorry

end harmonic_sum_ratio_l723_723488


namespace fib_10_equals_55_l723_723728

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

-- Theorem statement: The 10th Fibonacci number is 55
theorem fib_10_equals_55 : fib 10 = 55 :=
by
  sorry

end fib_10_equals_55_l723_723728


namespace median_is_ten_l723_723587

-- Given initial set
def initial_set : List ℝ := [5, 6, 3, 8, 4]

-- Define the new numbers to insert
def new_numbers : List ℝ := [10.1, 10.2]

-- Define the new set after insertion
def new_set : List ℝ := List.append initial_set new_numbers

-- Property to sort the set
def sorted_set : List ℝ := List.sort new_set

-- Function to find the median of a list
noncomputable def median (l : List ℝ) : ℝ :=
  let sorted := List.sort l
  list_nth_le sorted (List.length sorted / 2) sorry

-- Prove that the median of the new set is 10
theorem median_is_ten : median new_set = 10 :=
sorry

end median_is_ten_l723_723587


namespace number_of_distinct_prime_factors_30_fact_l723_723004

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l723_723004


namespace ProofProblem_l723_723512

noncomputable def problem_statement : Prop :=
  let f (x : ℝ) := Real.tan ((1/2) * x - Real.pi / 6)
  ∀ x k : ℝ, k ∈ Int → 
  let domain := ¬ (x = 2 * k * Real.pi + 4 * Real.pi / 3) in
  let period := ∀ x, f (x + 2 * Real.pi) = f x in
  let intervals_of_monotonicity := (2 * k * Real.pi - 2 * Real.pi / 3 < x) ∧ (x < 2 * k * Real.pi + 4 * Real.pi / 3) in
  (domain ∧ period ∧ intervals_of_monotonicity)

-- This theorem asserts that our solution satisfies all the properties we proved.
theorem ProofProblem : problem_statement := sorry

end ProofProblem_l723_723512


namespace count_three_digit_odd_increasing_order_l723_723974

theorem count_three_digit_odd_increasing_order : 
  ∃ n : ℕ, n = 10 ∧
  ∀ a b c : ℕ, (100 * a + 10 * b + c) % 2 = 1 ∧ a < b ∧ b < c ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 → 
    (100 * a + 10 * b + c) % 2 = 1 := 
sorry

end count_three_digit_odd_increasing_order_l723_723974


namespace fraction_multiplication_l723_723474

-- Define the problem as a theorem in Lean
theorem fraction_multiplication
  (a b x : ℝ) (hx : x ≠ 0) (hb : b ≠ 0) (ha : a ≠ 0): 
  (3 * a * b / x) * (2 * x^2 / (9 * a * b^2)) = (2 * x) / (3 * b) := 
by
  sorry

end fraction_multiplication_l723_723474


namespace seashells_given_to_Jessica_l723_723892

-- Define the initial number of seashells Dan had
def initialSeashells : ℕ := 56

-- Define the number of seashells Dan has left
def seashellsLeft : ℕ := 22

-- Define the number of seashells Dan gave to Jessica
def seashellsGiven : ℕ := initialSeashells - seashellsLeft

-- State the theorem to prove
theorem seashells_given_to_Jessica :
  seashellsGiven = 34 :=
by
  -- Begin the proof here
  sorry

end seashells_given_to_Jessica_l723_723892


namespace smallest_integer_greater_than_one_with_inverse_mod_1155_l723_723779

theorem smallest_integer_greater_than_one_with_inverse_mod_1155 :
  ∃ n : ℕ, n > 1 ∧ (∀ m : ℕ, m > 1 → (m % 1155 ≠ 0) → n ≤ m) ∧ (nat.gcd n 1155 = 1) ∧ n = 2 :=
sorry

end smallest_integer_greater_than_one_with_inverse_mod_1155_l723_723779


namespace different_prime_factors_of_factorial_eq_10_l723_723055

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723055


namespace hyperbola_range_k_l723_723378

noncomputable def hyperbola_equation (x y k : ℝ) : Prop :=
    (x^2) / (|k|-2) + (y^2) / (5-k) = 1

theorem hyperbola_range_k (k : ℝ) :
    (∃ x y, hyperbola_equation x y k) → (k > 5 ∨ (-2 < k ∧ k < 2)) :=
by 
    sorry

end hyperbola_range_k_l723_723378


namespace greatest_integer_quotient_l723_723945

theorem greatest_integer_quotient (N : ℕ) (h : (∑ i in (finset.range 9).map (λ n, n + 2), 1 / (nat.factorial i * nat.factorial (21 - i))) = N / nat.factorial 21) :
  ⌊N / 100⌋ = 499 :=
by
  sorry

end greatest_integer_quotient_l723_723945


namespace num_prime_factors_30_fac_l723_723070

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723070


namespace cats_problem_l723_723999

theorem cats_problem :
  let total_cats := 16 in
  let white_cats := 2 in
  let black_cats := total_cats * 25 / 100 in
  let grey_cats := total_cats - black_cats - white_cats in
  grey_cats = 10 :=
by
  let total_cats := 16
  let white_cats := 2
  let black_cats := total_cats * 25 / 100
  let grey_cats := total_cats - black_cats - white_cats
  show grey_cats = 10
  sorry

end cats_problem_l723_723999


namespace number_of_markers_l723_723333

theorem number_of_markers (packages markers_per_package : ℕ) (h1 : packages = 7) (h2 : markers_per_package = 5) : packages * markers_per_package = 35 := by
  rw [h1, h2]
  norm_num

end number_of_markers_l723_723333


namespace olivier_winning_strategy_l723_723329

theorem olivier_winning_strategy (N : ℕ) : 
  ∃ (strategy : N → bool), 
    (∀ n < N, strategy n  ≠ strategy (n+1) .mod N) ∧
    (∃ (player : bool), 
      (player = true → player_cannot_move strategy n) ∧ 
      (player = false → player_cannot_move strategy n)) :=
sorry

-- Definitions and auxiliary functions might be needed to refine this further
-- e.g., player_cannot_move to define under what conditions a player cannot move

end olivier_winning_strategy_l723_723329


namespace find_solutions_l723_723521

theorem find_solutions (x : ℝ) :
  (16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 48 →
  (x = 1.2 ∨ x = -81.2) :=
by sorry

end find_solutions_l723_723521


namespace num_4digit_special_integers_l723_723593

noncomputable def count_valid_4digit_integers : ℕ :=
  let first_two_options := 3 * 3 -- options for the first two digits
  let valid_last_two_pairs := 4 -- (6,9), (7,8), (8,7), (9,6)
  first_two_options * valid_last_two_pairs

theorem num_4digit_special_integers : count_valid_4digit_integers = 36 :=
by
  sorry

end num_4digit_special_integers_l723_723593


namespace factorial_prime_factors_l723_723152

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723152


namespace factorial_prime_factors_l723_723092

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723092


namespace value_of_a2_l723_723641

theorem value_of_a2 
  (a1 a2 a3 : ℝ)
  (h_seq : ∃ d : ℝ, (-8) = -8 + d * 0 ∧ a1 = -8 + d * 1 ∧ 
                     a2 = -8 + d * 2 ∧ a3 = -8 + d * 3 ∧ 
                     10 = -8 + d * 4) :
  a2 = 1 :=
by {
  sorry
}

end value_of_a2_l723_723641


namespace different_prime_factors_of_factorial_eq_10_l723_723046

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723046


namespace emily_phone_numbers_count_l723_723868

theorem emily_phone_numbers_count : 
  let digits := {1, 2, 3, 4, 5, 6, 7, 8} in
  let num_choices := 7 in
  let distinct_ordered_combinations := Nat.choose (8 : ℕ) (7 : ℕ) in 
  distinct_ordered_combinations = 8 :=
by
  sorry

end emily_phone_numbers_count_l723_723868


namespace number_less_than_neg_two_l723_723393

theorem number_less_than_neg_two : ∃ x : Int, x = -2 - 1 := 
by
  use -3
  sorry

end number_less_than_neg_two_l723_723393


namespace neg_prop_not_even_function_exists_even_function_l723_723344

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = f(x)

theorem neg_prop_not_even_function_exists_even_function :
  (∀ φ : ℝ, ¬ is_even_function (λ x => Real.sin (2 * x + φ))) ↔
  (∃ φ : ℝ, is_even_function (λ x => Real.sin (2 * x + φ))) :=
by
  sorry

end neg_prop_not_even_function_exists_even_function_l723_723344


namespace number_of_prime_factors_of_30_factorial_l723_723121

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723121


namespace length_of_chord_l723_723610

theorem length_of_chord
    (center : ℝ × ℝ) 
    (radius : ℝ) 
    (line : ℝ × ℝ × ℝ) 
    (circle_eq : (x : ℝ) → (y : ℝ) → ((x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2))
    (line_eq : (x : ℝ) → (y : ℝ) → (line.1 * x + line.2 * y + line.3 = 0)) :
    2 * radius * (if h : radius ≠ 0 then (1 - (1 / 2) * ((|line.1 * center.1 + line.2 * center.2 + line.3| / (real.sqrt (line.1 ^ 2 + line.2 ^ 2))) / radius) ^ 2) else 0).sqrt = 2 * real.sqrt 2 :=
by
    sorry

-- Definitions and conditions
def center : ℝ × ℝ := (1, 0)
def radius : ℝ := 2
def line : ℝ × ℝ × ℝ := (1, 1, 1)

noncomputable def circle_eq : (ℝ → ℝ → Prop) :=
    λ x y, (x - center.1) ^ 2 + y ^ 2 = 4

noncomputable def line_eq : (ℝ → ℝ → Prop) :=
    λ x y, x + y + 1 = 0

-- Applying the theorem
#eval (length_of_chord center radius line circle_eq line_eq)

end length_of_chord_l723_723610


namespace flat_rate_first_night_l723_723843

theorem flat_rate_first_night
  (f n : ℚ)
  (h1 : f + 3 * n = 210)
  (h2 : f + 6 * n = 350)
  : f = 70 :=
by
  sorry

end flat_rate_first_night_l723_723843


namespace num_prime_factors_of_30_l723_723225

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723225


namespace average_score_of_class_l723_723797

-- Definitions based on the conditions
def class_size : ℕ := 20
def group1_size : ℕ := 10
def group2_size : ℕ := 10
def group1_avg_score : ℕ := 80
def group2_avg_score : ℕ := 60

-- Average score of the whole class
theorem average_score_of_class : 
  (group1_size * group1_avg_score + group2_size * group2_avg_score) / class_size = 70 := 
by sorry

end average_score_of_class_l723_723797


namespace distribution_plans_count_l723_723561

theorem distribution_plans_count :
  let researchers := 4
      counties := 3
  in ∃ plans: ℕ, plans = 36 :=
by
  let ways_to_form_groups = Nat.choose 4 2
  let ways_to_assign_groups = 3!
  let total_plans = ways_to_form_groups * ways_to_assign_groups
  exact ⟨total_plans, rfl⟩

end distribution_plans_count_l723_723561


namespace factorize_expression_simplify_fraction_expr_l723_723815

-- (1) Prove the factorization of m^3 - 4m^2 + 4m
theorem factorize_expression (m : ℝ) : 
  m^3 - 4 * m^2 + 4 * m = m * (m - 2)^2 :=
by
  sorry

-- (2) Simplify the fraction operation correctly
theorem simplify_fraction_expr (x : ℝ) (h : x ≠ 1) : 
  2 / (x^2 - 1) - 1 / (x - 1) = -1 / (x + 1) :=
by
  sorry

end factorize_expression_simplify_fraction_expr_l723_723815


namespace circle_convex_polygons_count_l723_723506

theorem circle_convex_polygons_count : 
  let total_subsets := (2^15 - 1) - (15 + 105 + 455 + 255)
  let final_count := total_subsets - 500
  final_count = 31437 :=
by
  sorry

end circle_convex_polygons_count_l723_723506


namespace continuous_fraction_proof_l723_723645

noncomputable section

def continued_fraction (ω : ℝ) : ℕ → ℕ := sorry

def X_n (ω : ℝ) (n : ℕ) : ℝ := sorry

def F_n (n : ℕ) (x : ℝ) : ℝ := sorry

axiom F1_def : ∀ x : ℝ, F_n 1 x = x

axiom fn_def (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, (f x = ∑ k in (finset.range 1).erase 0, 1 / ((↑ k + x) ^ 2) * f (1 / (↑ k + x)))

axiom limit_fn (n : ℕ) (x : ℝ) :
  (∀ f : ℝ → ℝ, fn_def f → ∃ a b : ℝ, 0 < a ∧ a < b ∧ (∀ k, a < b ∧ (a / (1 + x) < fn n x ∧ fn n x < b / (1 + x))) →
    tendsto (λ n, fn n x) at_top (𝓝 ((1 / log 2) * 1 / (1 + x))))

theorem continuous_fraction_proof :
  (∀ ω : ℝ, ω ∈ (Icc 0 1) ∧ (continued_fraction ω)) →
  (∀ n > 1, ∀ x : ℝ, x ∈ (Icc 0 1) →
    F_n n x = ∑ k in (finset.range 1).erase 0, 
      (F_n (n-1) (1 / ↑ k) - F_n (n-1) (1 / (↑ k + x)))) ∧
  (∀ f : ℝ → ℝ, fn_def f → ∃ a : ℝ, a > 0 ∧ f = λ x, a / (1+x)) ∧
  (limit_fn)
  :=
sorry

end continuous_fraction_proof_l723_723645


namespace find_ellipse_equation_find_min_pq_mn_l723_723550

-- Definitions from conditions
def ellipse_c (x y a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (a c : ℝ) := c / a = real.sqrt (2) / 2
def max_area_triangle (a c b : ℝ) := (a - c) * b / 2 = (real.sqrt 2 - 1) / 2
def line_l (m x y : ℝ) := x = m * y - 1
def line_x_2 (x : ℝ) := x = 2

-- Proof statements
theorem find_ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : a^2 = 2 * (a^2 - b^2)) (h4 : max_area_triangle a (real.sqrt (a^2 - b^2)) b) :
    ellipse_c = (λ x y, (x^2 / 2) + (y^2) = 1) :=
sorry

theorem find_min_pq_mn (m x y x_P y_P y_1 y_2 : ℝ) (h1 : x = x) 
    (h2 : line_l m x y) 
    (h3 : line_x_2 x) 
    (h4 : y_1 + y_2 = 2 * m / (m^2 + 2)) 
    (h5 : y_1 * y_2 = -1 / (m^2 + 2)) 
    (h6 : |x_P - 2| * real.sqrt (1 + m ^ 2) / (real.sqrt 2 * (m^2 + 1)) = 2):
    real.sqrt (1 + m^2) + (2 / real.sqrt (1 + m^2)) = 2 :=
sorry

end find_ellipse_equation_find_min_pq_mn_l723_723550


namespace intersect_xz_plane_at_point_l723_723515

-- Define points and vectors in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the points A and B
def A : Point3D := ⟨2, -1, 3⟩
def B : Point3D := ⟨6, 7, -2⟩

-- Define the direction vector as the difference between points A and B
def direction_vector (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

-- Function to parameterize the line given a point and direction vector
def parametric_line (P : Point3D) (v : Point3D) (t : ℝ) : Point3D :=
  ⟨P.x + t * v.x, P.y + t * v.y, P.z + t * v.z⟩

-- Define the xz-plane intersection condition (y coordinate should be 0)
def intersects_xz_plane (P : Point3D) (v : Point3D) (t : ℝ) : Prop :=
  (parametric_line P v t).y = 0

-- Define the intersection point as a Point3D
def intersection_point : Point3D := ⟨2.5, 0, 2.375⟩

-- Statement to prove the intersection
theorem intersect_xz_plane_at_point : 
  ∃ t : ℝ, intersects_xz_plane A (direction_vector A B) t ∧ parametric_line A (direction_vector A B) t = intersection_point :=
by
  sorry

end intersect_xz_plane_at_point_l723_723515


namespace segment_PM_is_sqrt_3125_l723_723698

-- Given a square PQRS with each side of length 5
def side_length : ℝ := 5
def area_square : ℝ := side_length^2

-- PM and PN divide the square into four equal parts
def segment_PM_length : ℝ := sqrt 31.25

-- The proof goal is to show that the length of PM is sqrt 31.25
theorem segment_PM_is_sqrt_3125 (h1 : side_length = 5)
  (h2 : area_square = side_length^2)
  (h3 : ∀ (PM PN : ℝ), 
    2 * (PM * 5 / 2) = area_square / 4): 
  segment_PM_length = sqrt (5^2 + (2.5)^2) :=
by
  sorry

end segment_PM_is_sqrt_3125_l723_723698


namespace measure_of_each_interior_angle_l723_723740

theorem measure_of_each_interior_angle (n : ℕ) (hn : 3 ≤ n) : 
  ∃ angle : ℝ, angle = (n - 2) * 180 / n :=
by
  sorry

end measure_of_each_interior_angle_l723_723740


namespace area_ADC_l723_723635

-- Given conditions
variables (BD DC : ℝ) (area_ABD : ℝ)
hypothesis h1 : BD / DC = 3 / 2
hypothesis h2 : area_ABD = 30

-- Proof statement
theorem area_ADC (BD DC area_ABD : ℝ) (h1 : BD / DC = 3 / 2) (h2 : area_ABD = 30) :
  \(\text{{Let}} area_ADC = 2 / 3 * area_ABD \) :
  area_ADC = 20 :=
sorry

end area_ADC_l723_723635


namespace problem_solution_l723_723952

theorem problem_solution (x y z w : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : w > 0) 
  (h5 : x^2 + y^2 + z^2 + w^2 = 1) : 
  x^2 * y * z * w + x * y^2 * z * w + x * y * z^2 * w + x * y * z * w^2 ≤ 1 / 8 := 
by
  sorry

end problem_solution_l723_723952


namespace number_of_prime_factors_of_30_factorial_l723_723128

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723128


namespace triangle_PQR_area_l723_723410

open Real -- open real number space for easier manipulation

-- Define points P, Q, and R
def P := (-2, 2) : ℝ × ℝ
def Q := (6, 2) : ℝ × ℝ
def R := (2, -4) : ℝ × ℝ

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Define a function that calculates the area of a triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * real.abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Lean statement to prove that the area of triangle PQR is 24 square units
theorem triangle_PQR_area : triangle_area P Q R = 24 := by
  sorry

end triangle_PQR_area_l723_723410


namespace factorial_prime_factors_l723_723149

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723149


namespace infinite_arithmetic_progression_digit_sum_eq_l723_723347

noncomputable theory

/--
In every infinite, non-constant arithmetic progression consisting of natural numbers,
there are two different terms whose sum of digits in the decimal system is equal.
-/
theorem infinite_arithmetic_progression_digit_sum_eq
  (x d: ℕ) 
  (h_inf: ∀ n: ℕ, x + n * d ∈ ℕ)
  (h_non_const: d ≠ 0) :
  ∃ (a b: ℕ), a ≠ b ∧ (sum_of_digits a = sum_of_digits b) :=
  sorry

def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).foldl (λ s d, s + d) 0


end infinite_arithmetic_progression_digit_sum_eq_l723_723347


namespace number_of_people_correct_l723_723841

noncomputable def totalBill : ℝ := 139.00
noncomputable def tipRate : ℝ := 0.10
noncomputable def perPersonCost : ℝ := 19.1125

noncomputable def tipAmount : ℝ := totalBill * tipRate
noncomputable def totalAmountPaid : ℝ := totalBill + tipAmount
noncomputable def numberOfPeople : ℝ := totalAmountPaid / perPersonCost

theorem number_of_people_correct : numberOfPeople ≈ 8 := by
  sorry

end number_of_people_correct_l723_723841


namespace num_prime_factors_30_fac_l723_723065

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723065


namespace ratio_of_a_b_l723_723979

-- Define the problem
theorem ratio_of_a_b (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it.
  sorry

end ratio_of_a_b_l723_723979


namespace calvin_has_gotten_haircuts_l723_723886

noncomputable def calvin_haircuts : ℕ :=
  let x := ∀ (n : ℕ), (80 * n = 100 * 8 - 20 * n) in 8

theorem calvin_has_gotten_haircuts (h1 : ∀ n, 80 * n = 100 * 8 - 20 * n) : calvin_haircuts = 8 :=
by
  unfold calvin_haircuts
  exact h1
  sorry

end calvin_has_gotten_haircuts_l723_723886


namespace expected_pairs_correct_l723_723371

-- Define the total number of cards in the deck.
def total_cards : ℕ := 52

-- Define the number of black cards in the deck.
def black_cards : ℕ := 26

-- Define the number of red cards in the deck.
def red_cards : ℕ := 26

-- Define the expected number of pairs of adjacent cards such that one is black and the other is red.
def expected_adjacent_pairs := 52 * (26 / 51)

-- Prove that the expected_adjacent_pairs is equal to 1352 / 51.
theorem expected_pairs_correct : expected_adjacent_pairs = 1352 / 51 := 
by
  have expected_adjacent_pairs_simplified : 52 * (26 / 51) = (1352 / 51) := 
    by sorry
  exact expected_adjacent_pairs_simplified

end expected_pairs_correct_l723_723371


namespace number_of_prime_factors_of_30_factorial_l723_723118

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723118


namespace milk_transfer_equal_l723_723869

theorem milk_transfer_equal (A B C x : ℕ) (hA : A = 1200) (hB : B = A - 750) (hC : C = A - B) (h_eq : B + x = C - x) :
  x = 150 :=
by
  sorry

end milk_transfer_equal_l723_723869


namespace find_angle_between_AB_CA_l723_723968

def point := (ℝ × ℝ × ℝ)

def A : point := (1, 1, 1)
def B : point := (-1, 0, 4)
def C : point := (2, -2, 3)

def vector_sub (p1 p2 : point) : (ℝ × ℝ × ℝ) := 
  (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ := 
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def cosine_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ := 
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

noncomputable def θ : ℝ := 
  Real.arccos (cosine_angle (vector_sub B A) (vector_sub A C))

theorem find_angle_between_AB_CA : θ = 120 :=
by 
  sorry

end find_angle_between_AB_CA_l723_723968


namespace correct_mean_of_values_l723_723738

variable (n : ℕ) (mu_incorrect : ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (mu_correct : ℝ)

theorem correct_mean_of_values
  (h1 : n = 30)
  (h2 : mu_incorrect = 150)
  (h3 : incorrect_value = 135)
  (h4 : correct_value = 165)
  : mu_correct = 151 :=
by
  let S_incorrect := mu_incorrect * n
  let S_correct := S_incorrect - incorrect_value + correct_value
  let mu_correct := S_correct / n
  sorry

end correct_mean_of_values_l723_723738


namespace line_equation_l723_723736

variable (l1 l2 : ℝ → ℝ → Prop) (A B Q P : ℝ × ℝ) (l : ℝ → ℝ → Prop)

def l1 := λ x y : ℝ, x + y - 2 = 0
def l2 := λ x y : ℝ, x - y - 4 = 0

def A := (-1, 3)
def B := (5, 1)
def P := (3, -1)
def Q := (2, 2)
def l := λ x y : ℝ, 3 * x + y - 8 = 0

theorem line_equation :
  (l1 P.1 P.2 ∧ l2 P.1 P.2) ∧ (Q = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2))) →
  (l P.1 P.2 ∧ l Q.1 Q.2) := by
  -- Proof goes here
  sorry

end line_equation_l723_723736


namespace find_base_b_l723_723768

theorem find_base_b : ∃ b : ℕ, (b > 1) ∧ (sum (range b) = 2*b + 8) := by
  sorry

end find_base_b_l723_723768


namespace prime_factors_of_30_factorial_l723_723176

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723176


namespace prime_factors_of_30_factorial_l723_723168

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723168


namespace factorial_prime_factors_l723_723086

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723086


namespace charge_per_trousers_l723_723310

-- Definitions
def pairs_of_trousers : ℕ := 10
def shirts : ℕ := 10
def bill : ℕ := 140
def charge_per_shirt : ℕ := 5

-- Theorem statement
theorem charge_per_trousers :
  ∃ (T : ℕ), (pairs_of_trousers * T + shirts * charge_per_shirt = bill) ∧ (T = 9) :=
by 
  sorry

end charge_per_trousers_l723_723310


namespace smallest_perfect_square_gt_100_has_odd_number_of_factors_l723_723977

theorem smallest_perfect_square_gt_100_has_odd_number_of_factors : 
  ∃ n : ℕ, (n > 100) ∧ (∃ k : ℕ, n = k * k) ∧ (∀ m > 100, ∃ t : ℕ, m = t * t → n ≤ m) := 
sorry

end smallest_perfect_square_gt_100_has_odd_number_of_factors_l723_723977


namespace combinations_of_painting_options_l723_723674

theorem combinations_of_painting_options : 
  let colors := 6
  let methods := 3
  let finishes := 2
  colors * methods * finishes = 36 := by
  sorry

end combinations_of_painting_options_l723_723674


namespace max_g_8_l723_723317

noncomputable def g (x : ℝ) : ℝ := sorry -- To be filled with the specific polynomial

theorem max_g_8 (g : ℝ → ℝ)
  (h_nonneg : ∀ x, 0 ≤ g x)
  (h4 : g 4 = 16)
  (h16 : g 16 = 1024) : g 8 ≤ 128 :=
sorry

end max_g_8_l723_723317


namespace one_cow_one_bag_in_39_days_l723_723280

-- Definitions
def cows : ℕ := 52
def husks : ℕ := 104
def days : ℕ := 78

-- Problem: Given that 52 cows eat 104 bags of husk in 78 days,
-- Prove that one cow will eat one bag of husk in 39 days.
theorem one_cow_one_bag_in_39_days (cows_cons : cows = 52) (husks_cons : husks = 104) (days_cons : days = 78) :
  ∃ d : ℕ, d = 39 :=
by
  -- Placeholder for the proof.
  sorry

end one_cow_one_bag_in_39_days_l723_723280


namespace different_prime_factors_of_factorial_eq_10_l723_723051

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723051


namespace factorial_prime_factors_l723_723083

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723083


namespace candy_bar_multiple_l723_723690

theorem candy_bar_multiple (s m x : ℕ) (h1 : s = m * x + 6) (h2 : x = 24) (h3 : s = 78) : m = 3 :=
by
  sorry

end candy_bar_multiple_l723_723690


namespace smallest_tree_height_correct_l723_723752

-- Defining the conditions
def TallestTreeHeight : ℕ := 108
def MiddleTreeHeight (tallest : ℕ) : ℕ := (tallest / 2) - 6
def SmallestTreeHeight (middle : ℕ) : ℕ := middle / 4

-- Proof statement
theorem smallest_tree_height_correct :
  SmallestTreeHeight (MiddleTreeHeight TallestTreeHeight) = 12 :=
by
  -- Here we would put the proof, but we are skipping it with sorry.
  sorry

end smallest_tree_height_correct_l723_723752


namespace num_prime_factors_30_factorial_l723_723040

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723040


namespace discount_threshold_l723_723373

-- Definitions based on given conditions
def photocopy_cost : ℝ := 0.02
def discount_percentage : ℝ := 0.25
def copies_needed_each : ℕ := 80
def total_savings : ℝ := 0.40 * 2 -- total savings for both Steve and Dennison

-- Minimum number of photocopies required to get the discount
def min_copies_for_discount : ℕ := 160

-- Lean statement to prove the minimum number of photocopies required for the discount
theorem discount_threshold :
  ∀ (x : ℕ),
  photocopy_cost * (x : ℝ) - (photocopy_cost * (1 - discount_percentage) * (x : ℝ)) * 2 = total_savings → 
  min_copies_for_discount = 160 :=
by sorry

end discount_threshold_l723_723373


namespace number_of_prime_factors_thirty_factorial_l723_723012

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723012


namespace triangle_ACD_perimeters_sum_l723_723343

theorem triangle_ACD_perimeters_sum (A B C D : Type) (AB BC AD CD : ℕ) 
  (h : (BD : ℕ) = AD) (h1 : AB = 12) (h2 : BC = 28) (h3 : AD = CD) 
  (h4 : AD > 0) (h5 : BD > 0) : ℕ :=
begin
  sorry
end

#eval triangle_ACD_perimeters_sum nat nat nat nat 12 28 14 14 -- Expected output: 68

end triangle_ACD_perimeters_sum_l723_723343


namespace prime_factors_30_fac_eq_10_l723_723140

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723140


namespace forester_trees_planted_l723_723838

theorem forester_trees_planted (initial_trees : ℕ) (tripled_trees : ℕ) (trees_planted_monday : ℕ) (trees_planted_tuesday : ℕ) :
  initial_trees = 30 ∧ tripled_trees = 3 * initial_trees ∧ trees_planted_monday = tripled_trees - initial_trees ∧ trees_planted_tuesday = trees_planted_monday / 3 →
  trees_planted_monday + trees_planted_tuesday = 80 :=
by
  sorry

end forester_trees_planted_l723_723838


namespace number_of_prime_factors_of_30_factorial_l723_723117

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723117


namespace number_of_prime_factors_thirty_factorial_l723_723007

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723007


namespace ratio_of_right_to_left_hand_l723_723437

variable (R L : ℕ)

-- Conditions
variable (H1 : 2 / 3 * (R + L) = R + L - 1 / 3 * (R + L)) -- Two-thirds of the players were absent from practice.
variable (H2 : 2 / 3 = (2 / 3 * (R + L)) * 2 / 3) -- Two-thirds of the players at practice were left-handed.
variable (H3 : 2 / 3 * R / (2 / 3 * L) = 1.4) -- The ratio of the number of right-handed players who were not at practice that day to the number of left-handed players who were not at practice is 1.4.

theorem ratio_of_right_to_left_hand (H_R : R * (3 / 2) = 1.4 * L * (3 / 2)) : (R / L = 1.4) := 
 by
  -- To be filled in with a detailed proof.
  sorry

end ratio_of_right_to_left_hand_l723_723437


namespace total_shoes_l723_723302

theorem total_shoes (Brian_shoes : ℕ) (Edward_shoes : ℕ) (Jacob_shoes : ℕ)
  (hBrian : Brian_shoes = 22)
  (hEdward : Edward_shoes = 3 * Brian_shoes)
  (hJacob : Jacob_shoes = Edward_shoes / 2) :
  Brian_shoes + Edward_shoes + Jacob_shoes = 121 :=
by 
  sorry

end total_shoes_l723_723302


namespace grandmaster_chess_games_l723_723840

theorem grandmaster_chess_games :
  ∀ (a : ℕ → ℕ), 
  (∀ i, 1 ≤ a i ∧ a i < a (i + 1) ∧ a 42 ≤ 60) → 
  ∃ (k n : ℕ), a (k + n) - a k = 21 ∧ k + n ≤ 42 :=
begin
  sorry
end

end grandmaster_chess_games_l723_723840


namespace factorial_30_prime_count_l723_723188

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723188


namespace find_S3_over_a2_l723_723544

-- Definitions based on conditions in the problem
variable {a : ℕ → ℝ} -- geometric sequence with positive terms
variable {S : ℕ → ℝ} -- sum of first n terms of the sequence
variable {q : ℝ}     -- common ratio

-- Assume the conditions provided in the problem
def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = a n * q
axiom positive_terms (a : ℕ → ℝ) : ∀ n : ℕ, 0 < a n
axiom relation (a : ℕ → ℝ) : 3 * a 2 + 2 * a 3 = a 4
axiom sum_definition (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) : ∀ n : ℕ, S n = a 0 * (1 - q ^ n) / (1 - q)

-- State the proof problem
theorem find_S3_over_a2 (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) [geom_seq a q] [positive_terms a] [relation a] [sum_definition S a q] :
  q = 3 → S 3 / a 2 = 13 / 3 := by sorry

end find_S3_over_a2_l723_723544


namespace f_value_2009_l723_723524

noncomputable def f : ℝ → ℝ := sorry

theorem f_value_2009
    (h1 : ∀ x y : ℝ, f (x * y) = f x * f y)
    (h2 : f 0 ≠ 0) :
    f 2009 = 1 :=
sorry

end f_value_2009_l723_723524


namespace negative_number_zero_exponent_l723_723756

theorem negative_number_zero_exponent (a : ℤ) (h : a ≠ 0) : a^0 = 1 :=
by sorry

end negative_number_zero_exponent_l723_723756


namespace sonya_falls_count_l723_723700

/-- The number of times Sonya fell down given the conditions. -/
theorem sonya_falls_count : 
  let steven_falls := 3 in
  let stephanie_falls := steven_falls + 13 in
  let sonya_falls := (stephanie_falls / 2) - 2 in
  sonya_falls = 6 := 
by
  sorry

end sonya_falls_count_l723_723700


namespace book_distribution_ways_l723_723845

theorem book_distribution_ways : 
  ∃ n : ℕ, n = 7 ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ 7 →
  ∃ l : ℕ, l + (8 - l) = 8 ∧ 1 ≤ l ∧ 1 ≤ 8 - l :=
by
  -- We will provide a proof here.
  sorry

end book_distribution_ways_l723_723845


namespace find_y_for_orthogonality_l723_723919

-- Define the vectors
def vec_u : ℝ × ℝ × ℝ := (2, -6, 3)
def vec_v (y : ℝ) : ℝ × ℝ × ℝ := (-4, y, 5)

-- Define the dot product of two vectors in ℝ³
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- The statement we want to prove
theorem find_y_for_orthogonality (y : ℝ) :
  dot_product vec_u (vec_v y) = 0 ↔ y = 7 / 6 :=
by
  sorry

end find_y_for_orthogonality_l723_723919


namespace probability_both_in_picture_zero_l723_723354

variable (lap_time_rachel lap_time_robert : ℕ)
variable (start_time end_time picture_time : ℕ)
variable (picture_coverage : ℕ)

def rachel_in_picture (time : ℕ) : Prop :=
  let time_in_lap := time % lap_time_rachel
  time_in_lap ≤ picture_coverage ∨ time_in_lap ≥ lap_time_rachel - picture_coverage

def robert_in_picture (time : ℕ) : Prop :=
  let time_in_lap := time % lap_time_robert
  time_in_lap ≤ picture_coverage ∨ time_in_lap ≥ lap_time_robert - picture_coverage

theorem probability_both_in_picture_zero : 
  lap_time_rachel = 120 → lap_time_robert = 75 →
  start_time = 15 * 60 → end_time = 16 * 60 → picture_time = 20 →
  (∃ time ∈ set.Ico start_time end_time, rachel_in_picture lap_time_rachel picture_time time ∧ robert_in_picture lap_time_robert picture_time time) = false :=
by
  sorry

end probability_both_in_picture_zero_l723_723354


namespace combined_work_rate_l723_723796

-- Defining the work rates for A, B and C
def work_rate_A : ℝ := 1 / 18
def work_rate_B : ℝ := 1 / 9
def work_rate_C : ℝ := 1 / 9

-- Theorem: Combined work rate of A, B, and C in one day
theorem combined_work_rate : work_rate_A + work_rate_B + work_rate_C = 5 / 18 :=
by
  -- The proof is skipped with 'sorry'
  sorry

end combined_work_rate_l723_723796


namespace worst_player_is_daughter_l723_723834

-- Definitions of players
inductive Gender | male | female
inductive Player | father | sister | daughter | son
open Gender Player

-- Gender of each player
def gender : Player → Gender
| father := male
| sister := female
| daughter := female
| son := male

-- Generational indicator: 0 for father/sister, 1 for daughter/son (same generation level)
def generation : Player → Nat
| father := 0
| sister := 0
| daughter := 1
| son := 1

-- Definition of the twin relation
def is_twin : Player → Player → Prop
| daughter son := true
| son daughter := true
| _ _ := false

-- Definition of worst and best player
noncomputable def worst_player : Player := sorry -- To be proven (daughter)
noncomputable def best_player : Player := sorry  -- Unknown for now (father in the proof)

-- Conditions
axiom twin_same_gender (p : Player) : is_twin worst_player p → gender p = gender best_player
axiom different_generation : generation worst_player ≠ generation best_player

-- Theorem statement
theorem worst_player_is_daughter : worst_player = daughter := by
  sorry

end worst_player_is_daughter_l723_723834


namespace permutation_sum_inequality_l723_723321

noncomputable def permutations (n : ℕ) : List (List ℚ) :=
  List.permutations ((List.range (n+1)).map (fun i => if i = 0 then (1 : ℚ) else (1 : ℚ) / i))

theorem permutation_sum_inequality (n : ℕ) (a b : Fin n → ℚ)
  (ha : ∃ p : List ℚ, p ∈ permutations n ∧ ∀ i, a i = p.get? i) 
  (hb : ∃ q : List ℚ, q ∈ permutations n ∧ ∀ i, b i = q.get? i)
  (h_sum : ∀ i j : Fin n, i ≤ j → a i + b i ≥ a j + b j) 
  (m : Fin n) :
  a m + b m ≤ 4 / (m + 1) :=
sorry

end permutation_sum_inequality_l723_723321


namespace complex_expression_value_l723_723463

theorem complex_expression_value :
  ((6^2 - 4^2) + 2)^3 / 2 = 5324 :=
by
  sorry

end complex_expression_value_l723_723463


namespace different_prime_factors_of_factorial_eq_10_l723_723057

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l723_723057


namespace function_properties_graph_transformation_l723_723961

noncomputable def f (x : ℝ) (m : ℝ) := m * sin x + cos x

theorem function_properties :
  (∃ m, f (π / 2) m = 1) →
  f x 1 = sqrt 2 * sin (x + π / 4) ∧
  Function.Periodic (f x 1) (2 * π) ∧
  ∃ x, f x 1 = sqrt 2 :=
by
  sorry

/-
Describe the transformation required to obtain the graph of f(2x) from the graph of f(x - π/4).
Proof not required.
-/
theorem graph_transformation :
  (∀ x, f (2 * x) 1 = sqrt 2 * sin (2 * x + π / 4)) →
  (∀ x, f (x - π / 4) 1 = sqrt 2 * sin (x - π / 4 + π / 4)) →
  True :=
by
  sorry

end function_properties_graph_transformation_l723_723961


namespace total_legs_arms_tentacles_correct_l723_723683

-- Define the counts of different animals
def num_horses : Nat := 2
def num_dogs : Nat := 5
def num_cats : Nat := 7
def num_turtles : Nat := 3
def num_goat : Nat := 1
def num_snakes : Nat := 4
def num_spiders : Nat := 2
def num_birds : Nat := 3
def num_starfish : Nat := 1
def num_octopus : Nat := 1
def num_three_legged_dogs : Nat := 1

-- Define the legs, arms, and tentacles for each type of animal
def legs_per_horse : Nat := 4
def legs_per_dog : Nat := 4
def legs_per_cat : Nat := 4
def legs_per_turtle : Nat := 4
def legs_per_goat : Nat := 4
def legs_per_snake : Nat := 0
def legs_per_spider : Nat := 8
def legs_per_bird : Nat := 2
def arms_per_starfish : Nat := 5
def tentacles_per_octopus : Nat := 6
def legs_per_three_legged_dog : Nat := 3

-- Define the total number of legs, arms, and tentacles
def total_legs_arms_tentacles : Nat := 
  (num_horses * legs_per_horse) + 
  (num_dogs * legs_per_dog) + 
  (num_cats * legs_per_cat) + 
  (num_turtles * legs_per_turtle) + 
  (num_goat * legs_per_goat) + 
  (num_snakes * legs_per_snake) + 
  (num_spiders * legs_per_spider) + 
  (num_birds * legs_per_bird) + 
  (num_starfish * arms_per_starfish) + 
  (num_octopus * tentacles_per_octopus) + 
  (num_three_legged_dogs * legs_per_three_legged_dog)

-- The theorem to prove
theorem total_legs_arms_tentacles_correct :
  total_legs_arms_tentacles = 108 := by
  -- Proof goes here
  sorry

end total_legs_arms_tentacles_correct_l723_723683


namespace problem_1_problem_2_l723_723816

-- Problem 1 statement
theorem problem_1 (m : ℝ) (x : ℝ) :
  (∀ x, 0 < x → x < 2 → - (1 / 2) * x^2 + 2 * x > m * x) → m = 1 :=
by sorry

-- Problem 2 statement
theorem problem_2 (A B C : ℝ) (sin_A : ℝ) (cos_B : ℝ) :
  sin_A = 5 / 13 ∧ cos_B = 3 / 5 → cos C = -16 / 65 :=
by sorry

end problem_1_problem_2_l723_723816


namespace part1_parallel_to_real_axis_part2_range_of_m_l723_723942

noncomputable def z1 (b : ℝ) : ℂ := 1 + b * complex.I
def z2 : ℂ := 2 - 3 * complex.I

theorem part1_parallel_to_real_axis (b : ℝ) (h1 : z2 - z1 b = (1, 0)) : b = -3 :=
sorry

noncomputable def z (m : ℝ) (b : ℝ) : ℂ := (m + z1 b)^2

theorem part2_range_of_m (m : ℝ) (b : ℝ) (h2 : b = -3) (h3 : z m b ∈ {z : ℂ | z.im < 0 ∧ z.re < 0}) : -1 < m ∧ m < 2 :=
sorry

end part1_parallel_to_real_axis_part2_range_of_m_l723_723942


namespace children_passed_the_test_l723_723763

theorem children_passed_the_test (total_children : ℕ) (children_retaking : ℕ) 
  (h1 : total_children = 698) (h2 : children_retaking = 593) : 
  total_children - children_retaking = 105 :=
by
  rw [h1, h2]
  norm_num
  sorry

end children_passed_the_test_l723_723763


namespace speed_first_train_eq_l723_723804

-- Define the speeds of the two trains
def speed_of_second_train : ℝ := 100 -- speed in km/h
def ratio_speed : ℝ := 7 / 8

-- Define the speed of the first train
def speed_of_first_train : ℝ := (ratio_speed * speed_of_second_train)

-- Prove that the speed of the first train is 87.5 km/h
theorem speed_first_train_eq :
  speed_of_first_train = 87.5 :=
by
  unfold speed_of_first_train
  unfold ratio_speed
  unfold speed_of_second_train
  sorry

end speed_first_train_eq_l723_723804


namespace minimum_provinces_l723_723293

-- Define types and constants
def is_large (total_population : ℕ) (province_population : ℕ) : Prop :=
  province_population > 7 * total_population / 100

def exists_smaller_pairs (total_population : ℕ) (populations : List ℕ) : Prop :=
  ∀ p ∈ populations, is_large total_population p →
    ∃ a b ∈ populations, a < p ∧ b < p ∧ a + b > p

-- Main theorem stating the minimum number of provinces
theorem minimum_provinces (total_population : ℕ) (populations : List ℕ) :
    (∀ p ∈ populations, is_large total_population p →
       ∃ a b ∈ populations, a < p ∧ b < p ∧ a + b > p) →
    List.length populations ≥ 6 :=
by
  sorry

end minimum_provinces_l723_723293


namespace algorithm_outputs_min_value_l723_723468

theorem algorithm_outputs_min_value (a b c d : ℕ) :
  let m := a;
  let m := if b < m then b else m;
  let m := if c < m then c else m;
  let m := if d < m then d else m;
  m = min (min (min a b) c) d :=
by
  sorry

end algorithm_outputs_min_value_l723_723468


namespace lcm_of_numbers_l723_723805

-- Define the conditions given in the problem
def ratio (a b : ℕ) : Prop := 7 * b = 13 * a
def hcf_23 (a b : ℕ) : Prop := Nat.gcd a b = 23

-- Main statement to prove
theorem lcm_of_numbers (a b : ℕ) (h_ratio : ratio a b) (h_hcf : hcf_23 a b) : Nat.lcm a b = 2093 := by
  sorry

end lcm_of_numbers_l723_723805


namespace prime_factors_of_30_l723_723216

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723216


namespace derivative_at_pi_over_4_l723_723582

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem derivative_at_pi_over_4 : (deriv f) (Real.pi / 4) = 0 := 
by
  sorry

end derivative_at_pi_over_4_l723_723582


namespace part1_part2_l723_723930

section MathProof

variables (θ : ℝ) (h1 : cos ((π / 4) - θ) = sqrt 2 / 10) (h2 : 0 < θ ∧ θ < π)

-- Prove that sin (π / 4 + θ) = sqrt 2 / 10
theorem part1 : sin (π / 4 + θ) = sqrt 2 / 10 :=
sorry

-- Prove that sin^4 θ - cos^4 θ = 7 / 25
theorem part2 : sin θ ^ 4 - cos θ ^ 4 = 7 / 25 :=
sorry

end MathProof

end part1_part2_l723_723930


namespace angle_bisector_XD_length_l723_723298

noncomputable def triangle_XYZ := {XY : ℝ := 4, XZ : ℝ := 8, cos_angle_X : ℝ := 1 / 9}

theorem angle_bisector_XD_length (XY XZ : ℝ) (cos_angle_X : ℝ) (h1 : XY = 4) (h2 : XZ = 8) (h3 : cos_angle_X = 1 / 9) :
  let YZ := Real.sqrt (XY^2 + XZ^2 - 2 * XY * XZ * cos_angle_X),
      YD := YZ / (1 + XZ/XY),
      DZ := YZ - YD,
      angle_Y := (XY^2 + YZ^2 - XZ^2) / (2 * XY * YZ),
      XD := Real.sqrt (XY^2 + YD^2 - 2 * XY * YD * angle_Y)
  in XD = 40 * Real.sqrt(6) / 9 := 
by {
  -- Definitions and assumptions
  sorry
}

end angle_bisector_XD_length_l723_723298


namespace integral_ln_x_l723_723819

theorem integral_ln_x (n : ℕ) (hn : 0 < n) : 
  ∫ x in 0..1, x ^ (n - 1) * log x = -(1 : ℝ) / (n : ℝ)^2 := by
  sorry

end integral_ln_x_l723_723819


namespace num_prime_factors_of_30_l723_723230

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723230


namespace solution_set_l723_723949

noncomputable def f : ℝ → ℝ := sorry
axiom f_deriv_pos (x : ℝ) : (deriv f x) - f x > 0
axiom f_value_at_2023 : f 2023 = real.exp 2023

theorem solution_set (x : ℝ) : f (real.log x) < x ↔ 0 < x ∧ x < real.exp 2023 :=
by
  sorry

end solution_set_l723_723949


namespace prime_factors_of_30_factorial_l723_723103

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723103


namespace number_of_prime_factors_of_30_factorial_l723_723127

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723127


namespace exists_n_gt_1958_l723_723471

noncomputable definition broken_line_points (n : ℕ) : ℝ × ℝ := sorry

noncomputable definition distance_from_origin (n : ℕ) : ℝ := sorry

noncomputable definition segment_sum_length (n : ℕ) : ℝ := sorry

theorem exists_n_gt_1958 :
  ∃ n : ℕ, segment_sum_length n / distance_from_origin n > 1958 :=
sorry

end exists_n_gt_1958_l723_723471


namespace prime_factors_30_fac_eq_10_l723_723135

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723135


namespace num_prime_factors_30_fac_l723_723064

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723064


namespace neg_and_eq_or_not_l723_723269

theorem neg_and_eq_or_not (p q : Prop) : ¬(p ∧ q) ↔ ¬p ∨ ¬q :=
by sorry

end neg_and_eq_or_not_l723_723269


namespace derivative_at_pi_over_4_l723_723580

-- Define the function f and its derivative
def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x

-- State the theorem we want to prove
theorem derivative_at_pi_over_4 : f' (π / 4) = 0 :=
by 
  -- This is the placeholder for the proof
  sorry

end derivative_at_pi_over_4_l723_723580


namespace prime_factors_of_30_factorial_l723_723169

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723169


namespace profit_percentage_no_discount_l723_723461

theorem profit_percentage_no_discount (CP SP_with_discount MP SP_no_discount: ℝ) : 
  (MP = CP / (1 - 0.10)) ∧
  (SP_with_discount = CP * (1 + 0.25)) ∧
  (SP_with_discount = MP * (1 - 0.10)) ∧
  (SP_no_discount = MP) →
  ((SP_no_discount - CP) / CP * 100 = 38.89) :=
by
  -- Given Conditions
  intros h
  cases h with hMP hSP
  cases hSP with hSPwd hSPmp
  
  -- Placeholders for solution steps
  let MP := CP / (1 - 0.10)
  let SP_with_discount := CP * (1 + 0.25)
  let SP_no_discount := MP
  
  -- Assert the conditions
  have h1 : MP = CP / (1 - 0.10) := hMP
  have h2 : SP_with_discount = CP * (1 + 0.25) := hSPmp.left
  have h3 : SP_with_discount = MP * (1 - 0.10) := hSPmp.right.left
  have h4 : SP_no_discount = MP := hSPmp.right.right
  
  -- Skipping the actual calculation
  sorry

end profit_percentage_no_discount_l723_723461


namespace quadratic_expression_sum_l723_723904

theorem quadratic_expression_sum :
  ∃ a h k : ℝ, (∀ x, 4 * x^2 - 8 * x + 1 = a * (x - h)^2 + k) ∧ (a + h + k = 2) :=
sorry

end quadratic_expression_sum_l723_723904


namespace ratio_of_areas_l723_723377

section TrapezoidArea

variables {K L M N P R : Type} [metric_space K]
variables {length : K → ℝ} (k l m n p kR : K)
variables (A_trap A_tri : ℝ)

-- Conditions Based Definitions
def diagonal_condition : Prop := length m = 3 * length kP
def base_condition : Prop := length kN = 3 * length lM

-- Theorem to Prove
theorem ratio_of_areas (d_cond : diagonal_condition kP) (b_cond : base_condition lM) :
  (A_trap = 32 / 3 * A_tri) :=
sorry

end TrapezoidArea

end ratio_of_areas_l723_723377


namespace floor_equation_solution_l723_723358

/-- An auxiliary definition of the floor function. -/
def my_floor (a : ℝ) : ℤ := Int.floor a

/-- The main theorem stating the solution set of the given equation. -/
theorem floor_equation_solution {x : ℝ} :
  my_floor (1 / (1 - x)) = my_floor (1 / (1.5 - x)) ↔ x ∈ Set.union Set.Iio 0 Set.Ici 2.5 :=
by
  sorry

end floor_equation_solution_l723_723358


namespace average_weight_of_Arun_l723_723428

def arun_opinion (w : ℝ) : Prop := 66 < w ∧ w < 72
def brother_opinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def mother_opinion (w : ℝ) : Prop := w ≤ 69

theorem average_weight_of_Arun :
  (∀ w, arun_opinion w → brother_opinion w → mother_opinion w → 
    (w = 67 ∨ w = 68 ∨ w = 69)) →
  avg_weight = 68 :=
sorry

end average_weight_of_Arun_l723_723428


namespace factorial_prime_factors_l723_723162

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723162


namespace sequence_periodic_l723_723672

noncomputable def a : ℕ → ℝ
| 0     := 3      -- any initial value, will not be used in this problem
| (n+1) := (1 + a n) / (1 - a n)

theorem sequence_periodic (initial_value : ℝ) (h : a 2017 = 3) : a 1 = 3 :=
by
  sorry

end sequence_periodic_l723_723672


namespace people_meet_probability_l723_723342

noncomputable def probability_people_meet : ℝ :=
  let Ω := {p : ℝ × ℝ | 7 < p.1 ∧ p.1 < 8 ∧ 7 + 1/3 < p.2 ∧ p.2 < 8 + 5/6}
  let A := {p : ℝ × ℝ | 7 < p.1 ∧ p.1 < 8 ∧ 7 + 1/3 < p.2 ∧ p.2 < 8 + 5/6 ∧ abs (p.1 - p.2) < 1/6}
  (∬ (λ p : ℝ × ℝ, 1 : ℝ) ∂ p ∈ A) / (∬ (λ p : ℝ × ℝ, 1 : ℝ) ∂ p ∈ Ω)

theorem people_meet_probability :
  probability_people_meet = 1 / 3 :=
sorry

end people_meet_probability_l723_723342


namespace sonya_falls_6_l723_723702

def number_of_falls_steven : ℕ := 3
def number_of_falls_stephanie : ℕ := number_of_falls_steven + 13
def number_of_falls_sonya : ℕ := (number_of_falls_stephanie / 2) - 2

theorem sonya_falls_6 : number_of_falls_sonya = 6 := 
by
  -- The actual proof is to be filled in here
  sorry

end sonya_falls_6_l723_723702


namespace prime_factors_30_fac_eq_10_l723_723139

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723139


namespace ferry_tourists_l723_723443

def tourists_on_trip (start_time : ℕ) (start_tourists : ℕ) (decrease_per_trip : ℕ) (trip_time : ℕ) : ℕ :=
start_tourists - decrease_per_trip * (trip_time - start_time)

def total_tourists_ferry (start_time : ℕ) (end_time : ℕ) (start_tourists : ℕ) (decrease_per_trip : ℕ) : ℕ :=
∑ t in finset.range (end_time - start_time + 1), tourists_on_trip start_time start_tourists decrease_per_trip t

theorem ferry_tourists : total_tourists_ferry 10 15 100 1 = 585 :=
by
  sorry

end ferry_tourists_l723_723443


namespace number_of_prime_factors_of_30_factorial_l723_723113

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723113


namespace factor_of_polynomial_l723_723893

def polynomial (x : ℝ) : ℝ := x^4 - 4*x^2 + 16
def q1 (x : ℝ) : ℝ := x^2 + 4
def q2 (x : ℝ) : ℝ := x - 2
def q3 (x : ℝ) : ℝ := x^2 - 4*x + 4
def q4 (x : ℝ) : ℝ := x^2 + 4*x + 4

theorem factor_of_polynomial : (∃ (f g : ℝ → ℝ), polynomial x = f x * g x) ∧ (q4 = f ∨ q4 = g) := by sorry

end factor_of_polynomial_l723_723893


namespace max_prime_area_of_rectangle_with_perimeter_40_is_19_l723_723852

-- Predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := sorry

-- Given conditions: perimeter of 40 units; perimeter condition and area as prime number.
def max_prime_area_of_rectangle_with_perimeter_40 : Prop :=
  ∃ (l w : ℕ), l + w = 20 ∧ is_prime (l * (20 - l)) ∧
  ∀ (l' w' : ℕ), l' + w' = 20 → is_prime (l' * (20 - l')) → (l * (20 - l)) ≥ (l' * (20 - l'))

theorem max_prime_area_of_rectangle_with_perimeter_40_is_19 :
  max_prime_area_of_rectangle_with_perimeter_40 :=
sorry

end max_prime_area_of_rectangle_with_perimeter_40_is_19_l723_723852


namespace count_ordered_triplets_l723_723956

theorem count_ordered_triplets (a b c : ℕ) (h : a + b + c = 50) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
    (∑ i in finset.range (49), 49 - i) = 1176 :=
sorry

end count_ordered_triplets_l723_723956


namespace prime_factors_of_30_l723_723206

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723206


namespace prime_factors_of_30_factorial_l723_723184

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723184


namespace initial_pens_eq_42_l723_723341

-- Definitions based on the conditions
def initial_books : ℕ := 143
def remaining_books : ℕ := 113
def remaining_pens : ℕ := 19
def sold_pens : ℕ := 23

-- Theorem to prove that the initial number of pens was 42
theorem initial_pens_eq_42 (b_init b_remain p_remain p_sold : ℕ) 
    (H_b_init : b_init = initial_books)
    (H_b_remain : b_remain = remaining_books)
    (H_p_remain : p_remain = remaining_pens)
    (H_p_sold : p_sold = sold_pens) : 
    (p_sold + p_remain = 42) := 
by {
    -- Provide proof later
    sorry
}

end initial_pens_eq_42_l723_723341


namespace find_some_number_l723_723613

def op (x w : ℕ) := (2^x) / (2^w)

theorem find_some_number (n : ℕ) (hn : 0 < n) : (op (op 4 n) n) = 4 → n = 2 :=
by
  sorry

end find_some_number_l723_723613


namespace number_of_oranges_l723_723305

theorem number_of_oranges (pieces_per_orange : ℤ) (pieces_per_person : ℤ) (people_count : ℤ) (calories_per_orange : ℤ) (calories_per_person : ℤ) :
    pieces_per_orange = 8 →
    pieces_per_person = 10 →
    people_count = 4 →
    calories_per_orange = 80 →
    calories_per_person = 100 →
    let total_pieces := people_count * pieces_per_person in
    let oranges_needed := total_pieces / pieces_per_orange in
    oranges_needed = 5 := 
by
    intros
    sorry

end number_of_oranges_l723_723305


namespace Ahmed_goat_count_l723_723864

theorem Ahmed_goat_count : 
  let A := 7 in
  let B := 2 * A + 5 in
  let C := B - 6 in
  C = 13 :=
by
  let A := 7
  let B := 2 * A + 5
  let C := B - 6
  show C = 13
  sorry

end Ahmed_goat_count_l723_723864


namespace all_remaining_are_even_l723_723692

-- Defining the conditions in the problem
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def last_digit (n : ℕ) : ℕ := n % 10
def second_last_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Rule 1: Remove numbers with an odd last digit and an even second last digit
def rule1 (n : ℕ) : Prop :=
  is_odd (last_digit n) ∧ is_even (second_last_digit n)

-- Rule 2: Remove numbers with an odd last digit not divisible by 3
def rule2 (n : ℕ) : Prop :=
  is_odd (last_digit n) ∧ ¬ (last_digit n % 3 = 0)

-- Rule 3: Remove numbers with an odd second last digit and divisible by 3
def rule3 (n : ℕ) : Prop :=
  is_odd (second_last_digit n) ∧ (second_last_digit n % 3 = 0)

-- Combining all rules
def remove_condition (n : ℕ) : Prop := rule1 n ∨ rule2 n ∨ rule3 n

theorem all_remaining_are_even (S : set ℕ) : 
  (∀ n ∈ S, remove_condition n → false) → (∀ n ∈ S, is_even (last_digit n)) :=
by
  intro h n hn
  sorry

end all_remaining_are_even_l723_723692


namespace find_f_l723_723810

theorem find_f (f : ℝ → ℝ) (x : ℝ)
  (h1: |f x + (cos x)^2| ≤ 3 / 4)
  (h2: |f x - (sin x)^2| ≤ 1 / 4) :
  f x = 3 / 4 - (cos x)^2 := 
sorry

end find_f_l723_723810


namespace prime_factors_of_30_l723_723219

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723219


namespace right_triangle_OA_length_l723_723465

variables {O A B M : ℝ}

axiom angle_A_right : ∠ A = 90

axiom altitude_OA_meets_M : true  -- This axiom signifies the geometry layout without specifying points
axiom distance_M_to_second_side : M = 2
axiom distance_B_to_second_side : B = 1

theorem right_triangle_OA_length :
  OA = sqrt(2) :=
sorry

end right_triangle_OA_length_l723_723465


namespace num_prime_factors_30_factorial_l723_723032

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723032


namespace range_of_b_l723_723597

open Real

variable (b : ℝ)

def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + b * log (2 * x + 4)

theorem range_of_b:
  (∀ x : ℝ, -2 < x → f' b x ≤ 0) → b ≤ -1 :=
by
  intro h
  sorry

#check range_of_b

end range_of_b_l723_723597


namespace book_distribution_ways_l723_723844

theorem book_distribution_ways : 
  ∃ n : ℕ, n = 7 ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ 7 →
  ∃ l : ℕ, l + (8 - l) = 8 ∧ 1 ≤ l ∧ 1 ≤ 8 - l :=
by
  -- We will provide a proof here.
  sorry

end book_distribution_ways_l723_723844


namespace jon_total_cost_l723_723880
-- Import the complete Mathlib library

-- Define the conditions
def MSRP : ℝ := 30
def insurance_rate : ℝ := 0.20
def tax_rate : ℝ := 0.50

-- Calculate intermediate values based on conditions
noncomputable def insurance_cost : ℝ := insurance_rate * MSRP
noncomputable def subtotal_before_tax : ℝ := MSRP + insurance_cost
noncomputable def state_tax : ℝ := tax_rate * subtotal_before_tax
noncomputable def total_cost : ℝ := subtotal_before_tax + state_tax

-- The theorem we need to prove
theorem jon_total_cost : total_cost = 54 := by
  -- Proof is omitted
  sorry

end jon_total_cost_l723_723880


namespace length_of_base_of_vessel_l723_723830

noncomputable def volume_of_cube (edge : ℝ) := edge ^ 3

theorem length_of_base_of_vessel 
  (cube_edge : ℝ)
  (vessel_width : ℝ)
  (rise_in_water_level : ℝ)
  (volume_cube : ℝ)
  (h1 : cube_edge = 15)
  (h2 : vessel_width = 15)
  (h3 : rise_in_water_level = 11.25)
  (h4 : volume_cube = volume_of_cube cube_edge)
  : ∃ L : ℝ, L = volume_cube / (vessel_width * rise_in_water_level) ∧ L = 20 :=
by
  sorry

end length_of_base_of_vessel_l723_723830


namespace paving_cost_l723_723386

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 300
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost : cost = 6187.50 := by
  -- length = 5.5
  -- width = 3.75
  -- rate = 300
  -- area = length * width = 20.625
  -- cost = area * rate = 6187.50
  sorry

end paving_cost_l723_723386


namespace prime_factors_30_fac_eq_10_l723_723142

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723142


namespace greatest_k_divides_n_l723_723454

-- Define the divisor function for clarity
def d (n : ℕ) : ℕ := (Finset.range (n+1)).count (λ i => i > 0 ∧ n % i = 0)

theorem greatest_k_divides_n (n : ℕ) (k : ℕ) (m : ℕ) (hk : n = 5^k * m) (hm : m % 5 ≠ 0)
  (h1: d n = 48) (h2: d (5 * n) = 72) : k = 1 := by
  sorry

end greatest_k_divides_n_l723_723454


namespace num_prime_factors_30_fac_l723_723063

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723063


namespace right_triangle_proportion_l723_723339

/-- Given a right triangle ABC with ∠C = 90°, AB = c, AC = b, and BC = a, 
    and a point P on the hypotenuse AB (or its extension) such that 
    AP = m, BP = n, and CP = k, prove that a²m² + b²n² = c²k². -/
theorem right_triangle_proportion
  {a b c m n k : ℝ}
  (h_right : ∀ A B C : ℝ, A^2 + B^2 = C^2)
  (h1 : ∀ P : ℝ, m^2 + n^2 = k^2)
  (h_geometry : a^2 + b^2 = c^2) :
  a^2 * m^2 + b^2 * n^2 = c^2 * k^2 := 
sorry

end right_triangle_proportion_l723_723339


namespace overall_discount_rate_l723_723440

variable (price_bag price_shirt price_shoes price_hat price_jacket : ℝ)
variable (paid_bag paid_shirt paid_shoes paid_hat paid_jacket : ℝ)

def total_marked_price :=
  price_bag + price_shirt + price_shoes + price_hat + price_jacket

def total_price_paid :=
  paid_bag + paid_shirt + paid_shoes + paid_hat + paid_jacket

def total_discount :=
  total_marked_price - total_price_paid

def discount_rate :=
  (total_discount / total_marked_price) * 100

theorem overall_discount_rate :
  price_bag = 200 ∧ paid_bag = 120 ∧
  price_shirt = 80 ∧ paid_shirt = 60 ∧
  price_shoes = 150 ∧ paid_shoes = 105 ∧
  price_hat = 50 ∧ paid_hat = 40 ∧
  price_jacket = 220 ∧ paid_jacket = 165 →
  discount_rate = 30 :=
by 
  intros h
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩ := h
  rw [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10]
  -- Calculate total marked price
  have total_marked_price_eq : total_marked_price = 700 := 
    by simp [total_marked_price]
  -- Calculate total price paid
  have total_price_paid_eq : total_price_paid = 490 := 
    by simp [total_price_paid]
  -- Calculate total discount
  have total_discount_eq : total_discount = 210 := 
    by dsimp [total_discount]; rw [total_marked_price_eq, total_price_paid_eq]; norm_num
  -- Calculate discount rate
  have discount_rate_eq : discount_rate = 30 := 
    by dsimp [discount_rate]; rw [total_discount_eq, total_marked_price_eq]; norm_num
  exact discount_rate_eq.symm

end overall_discount_rate_l723_723440


namespace prime_factors_of_30_factorial_l723_723095

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723095


namespace maximum_of_expression_l723_723313

theorem maximum_of_expression (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 3) : 
  a + b^2 + c^4 ≤ 3 ∧ (∃ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧ a + b^2 + c^4 = 3) :=
begin
  sorry
end

end maximum_of_expression_l723_723313


namespace ratio_is_seven_over_eight_l723_723359

variable (a b c x y z : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c > 0)
variable (h4 : x > 0)
variable (h5 : y > 0)
variable (h6 : z > 0)
variable (h7 : a^2 + b^2 + c^2 = 49)
variable (h8 : x^2 + y^2 + z^2 = 64)
variable (h9 : ax + by + cz = 56)

theorem ratio_is_seven_over_eight :
  (a + b + c) / (x + y + z) = 7 / 8 :=
by
  sorry

end ratio_is_seven_over_eight_l723_723359


namespace sum_of_leading_digits_of_roots_l723_723650

-- Definition of M: a 100-digit number where each digit is 8
def M : ℕ := 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

-- Function g(r) to get the leading digit of the r-th root of M
def leading_digit (r : ℕ) (n : ℕ) : ℕ :=
  let root := (n : ℝ)^(1/r : ℝ)
  let leading := Int.floor (root / (10 ^ (Int.floor (Real.log10 root))))
  leading.toNat

def g (r : ℕ) : ℕ := leading_digit r M

-- Task: Calculate the sum g(2) + g(3) + g(4) + g(5)
theorem sum_of_leading_digits_of_roots :
  g 2 + g 3 + g 4 + g 5 = 8 :=
by
  sorry

end sum_of_leading_digits_of_roots_l723_723650


namespace bahs_equal_to_yahs_l723_723984

theorem bahs_equal_to_yahs (bahs rahs yahs : ℝ) 
  (h1 : 18 * bahs = 30 * rahs) 
  (h2 : 6 * rahs = 10 * yahs) : 
  1200 * yahs = 432 * bahs := 
by
  sorry

end bahs_equal_to_yahs_l723_723984


namespace one_and_one_third_of_x_is_36_l723_723682

theorem one_and_one_third_of_x_is_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 := 
sorry

end one_and_one_third_of_x_is_36_l723_723682


namespace num_prime_factors_30_factorial_l723_723241

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723241


namespace find_dot_product_mag_l723_723667

noncomputable def magnitude {α} [inner_product_space ℝ α] (v : α) : ℝ := real.sqrt (inner v v)

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

theorem find_dot_product_mag (ha : magnitude a = 3) 
                            (hb : magnitude b = 7)
                            (h_cross : magnitude (a ×ᵥ b) = 15) :
  abs (inner a b) = 6 * real.sqrt 6 :=
by sorry

end find_dot_product_mag_l723_723667


namespace best_overall_value_l723_723336

structure Box where
  weight : ℝ
  cost : ℝ
  quality_rating : ℝ

def price_per_ounce (box : Box) : ℝ :=
  box.cost / box.weight

def overall_value (box : Box) : ℝ :=
  (box.quality_rating * 1000) / price_per_ounce(box)

constant BrandA_Box1 : Box := ⟨30, 4.80, 4.5⟩
constant BrandA_Box2 : Box := ⟨20, 3.40, 4.5⟩
constant BrandB_Box3 : Box := ⟨15, 2.00, 3.9⟩
constant BrandB_Box4 : Box := ⟨25, 3.25, 3.9⟩

theorem best_overall_value :
  overall_value(BrandA_Box1) > overall_value(BrandA_Box2) ∧
  overall_value(BrandA_Box1) > overall_value(BrandB_Box3) ∧
  overall_value(BrandA_Box1) > overall_value(BrandB_Box4) :=
by
  sorry

end best_overall_value_l723_723336


namespace solve_system_l723_723697

open Real

noncomputable def solution_set : Set (ℝ × ℝ) :=
  { p | let (x, y) := p in 
         (x = -sqrt 2/sqrt 5 ∧ y = 2*sqrt 2/sqrt 5) ∨ 
         (x = sqrt 2/sqrt 5 ∧ y = 2*sqrt 2/sqrt 5) ∨
         (x = sqrt 2/sqrt 5 ∧ y = -2*sqrt 2/sqrt 5) ∨ 
         (x = -sqrt 2/sqrt 5 ∧ y = -2*sqrt 2/sqrt 5) }

theorem solve_system (x y : ℝ) : 
  x^2 + y^2 ≤ 2 ∧ x^4 - 8*x^2*y^2 + 16*y^4 - 20*x^2 - 80*y^2 + 100 = 0 →
  (x, y) ∈ solution_set :=
by
  sorry

end solve_system_l723_723697


namespace base_length_of_vessel_l723_723831

def volume_of_cube (edge : ℝ) := edge^3

def volume_of_displaced_water (L width rise : ℝ) := L * width * rise

theorem base_length_of_vessel (edge width rise L : ℝ) 
  (h1 : edge = 15) (h2 : width = 15) (h3 : rise = 11.25) 
  (h4 : volume_of_displaced_water L width rise = volume_of_cube edge) : 
  L = 20 :=
by
  sorry

end base_length_of_vessel_l723_723831


namespace lakers_win_in_7_games_probability_l723_723366

theorem lakers_win_in_7_games_probability :
  let lakers_win_prob := (1/4 : ℝ)
  let knicks_win_prob := (3/4 : ℝ)
  P(Lakers win NBA finals in 7 games) = (540 / 16384 : ℝ) :=
begin
  -- Define the series conditions and probabilities
  let games_to_win := 4,
  let max_games := 7,
  have prob_knicks_win := knicks_win_prob,
  have prob_lakers_win := lakers_win_prob,
  
  -- Calculate the probability of the series being tied 3-3
  let comb := nat.choose 6 3,
  let prob_tied_series := comb * (prob_lakers_win ^ 3) * (prob_knicks_win ^ 3),
  
  -- Calculate the final probability of the Lakers winning in the 7th game
  let final_prob := prob_tied_series * prob_lakers_win,
  
  -- Showing the final probability as a fraction
  have result : final_prob = (540 / 16384 : ℝ),
  sorry
end

end lakers_win_in_7_games_probability_l723_723366


namespace length_of_goods_train_l723_723424

theorem length_of_goods_train 
  (speed_km_per_hr : ℕ) (platform_length_m : ℕ) (time_sec : ℕ) 
  (h1 : speed_km_per_hr = 72) (h2 : platform_length_m = 300) (h3 : time_sec = 26) : 
  ∃ length_of_train : ℕ, length_of_train = 220 :=
by
  sorry

end length_of_goods_train_l723_723424


namespace total_social_media_hours_in_a_week_l723_723504

variable (daily_social_media_hours : ℕ) (days_in_week : ℕ)

theorem total_social_media_hours_in_a_week
(h1 : daily_social_media_hours = 3)
(h2 : days_in_week = 7) :
daily_social_media_hours * days_in_week = 21 := by
  sorry

end total_social_media_hours_in_a_week_l723_723504


namespace total_miles_walked_l723_723684

-- Definitions based on the problem conditions
def pedometer_max_steps : ℕ := 99999
def pedometer_resets : ℕ := 50
def end_of_year_reading : ℕ := 25000
def steps_per_mile : ℕ := 2000

-- Prove that the total number of miles Pete walked during the year is 2512.5 miles
theorem total_miles_walked : 
  (pedometer_resets * (pedometer_max_steps + 1) + end_of_year_reading) / steps_per_mile = 2512.5 := 
by
  sorry

end total_miles_walked_l723_723684


namespace mean_of_primes_l723_723897

open Real

def is_three_digit_prime (p : ℕ) : Prop := 
  p >= 100 ∧ p < 1000 ∧ Nat.Prime p

def satisfy_condition (x p : ℕ) : Prop := 
  p = x^2 - 21

def mean (list : List ℕ) : Real :=
  (list.sum : ℕ) / list.length

theorem mean_of_primes :
  let list := [x | x in List.range 32, let p := x^2 - 21 in is_three_digit_prime p ∧ satisfy_condition x p]
  mean list = 421 := by
  sorry

end mean_of_primes_l723_723897


namespace num_prime_factors_30_fac_l723_723069

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723069


namespace count_five_digit_odd_numbers_without_repetition_l723_723408

-- Definitions taken directly from the problem conditions
def digits := {0, 1, 2, 3, 4, 5}
def valid_odd_digits := {1, 3, 5}
def n_digits := 5

-- Lean statement representing the property to be proved
theorem count_five_digit_odd_numbers_without_repetition : 
  (card {num : finset ℕ | num ⊆ digits ∧ num.card = n_digits ∧ 
         (num.to_list.last ∈ valid_odd_digits) 
         ∧ (num.to_list.head ≠ 0) 
         ∧ (nodup num.to_list)}) = 288 :=
sorry  -- Proof to be provided

end count_five_digit_odd_numbers_without_repetition_l723_723408


namespace minimum_perimeter_quadrilateral_l723_723712

noncomputable theory

-- Define the coordinates and constraints given in the problem
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)
def C : ℝ × ℝ := (2, 4)
def area_ABCD : ℝ := 20

-- Define the perimeter of quadrilateral ABCD
def perimeter (D : ℝ × ℝ) : ℝ := 
  dist A B + dist B C + dist C D + dist D A

-- Main theorem statement to prove minimum perimeter
theorem minimum_perimeter_quadrilateral : ∃ (D : ℝ × ℝ), perimeter D = 18.2 ∧ 
  (1 / 2 * abs ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))) + 
  (1 / 2 * abs ((A.1 * (C.2 - D.2)) + (C.1 * (D.2 - A.2)) + (D.1 * (A.2 - C.2)))))) = area_ABCD := 
by {
  sorry,
}

end minimum_perimeter_quadrilateral_l723_723712


namespace num_prime_factors_of_30_l723_723224

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723224


namespace number_of_zeros_l723_723978

-- Definitions based on the conditions
def five_thousand := 5 * 10 ^ 3
def one_hundred := 10 ^ 2

-- The main theorem that we want to prove
theorem number_of_zeros : (five_thousand ^ 50) * (one_hundred ^ 2) = 10 ^ 154 * 5 ^ 50 := 
by sorry

end number_of_zeros_l723_723978


namespace cos_power_trig_identity_l723_723351

theorem cos_power_trig_identity (n : ℕ) (hn : 0 < n) (θ : ℝ) : 
  cos (θ) ^ n = (1 / (2 ^ n)) * (Finset.sum (Finset.range (n + 1)) 
    (λ k, nat.choose n k * cos ((n - 2 * k) * θ))) := 
by
  sorry

end cos_power_trig_identity_l723_723351


namespace analytic_expression_l723_723560

noncomputable def f (a : ℝ) [fact (a > 0)] [fact (a ≠ 1)] (x : ℝ) : ℝ :=
if h : 1 ≤ x ∧ x ≤ 2 then log a x
else if h : (-1 ≤ x ∧ x ≤ 0) then log a (x + 2)
else if h : (0 < x ∧ x ≤ 1) then log a (2 - x)
else 0 -- placeholder for values outside the needed intervals

theorem analytic_expression (f : ℝ → ℝ) (a : ℝ) [fact (a > 0)] [fact (a ≠ 1)] :
  (∀ x, f (-x) = f x) →
  (∀ x, f (x + 1) = f (x - 1)) →
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f x = log a x) →
  (∀ x, -1 ≤ x ∧ x ≤ 0 → f x = log a (x + 2)) ∧ 
  (∀ x, 0 < x ∧ x ≤ 1 → f x = log a (2 - x)) :=
begin
  intros hf_even hf_shift hf_def,
  split,
  { intros x h,
    have h1 : 1 ≤ x + 2 ∧ x + 2 ≤ 2, from ⟨ by linarith, by linarith ⟩,
    exact hf_def (x + 2) h1 },
  { intros x h,
    have h2 : 1 ≤ 2 - x ∧ 2 - x < 2, from ⟨ by linarith, by linarith⟩,
    specialize hf_def (2 - x) h2,
    rw hf_even,
    exact hf_def }
end

end analytic_expression_l723_723560


namespace fraction_product_l723_723884

theorem fraction_product : 
  (4 / 2) * (3 / 6) * (10 / 5) * (15 / 30) * (20 / 10) * (45 / 90) * (50 / 25) * (60 / 120) = 1 := 
by
  sorry

end fraction_product_l723_723884


namespace ratio_of_areas_l723_723735

theorem ratio_of_areas (s : ℝ) : 
  let area_S := s^2,
      area_R := (1.1 * s) * (0.9 * s)
  in (area_R / area_S) = 99 / 100 := 
by
  sorry

end ratio_of_areas_l723_723735


namespace man_l723_723425

variable (v : ℝ) (speed_with_current : ℝ) (speed_of_current : ℝ)

theorem man's_speed_against_current :
  speed_with_current = 12 ∧ speed_of_current = 2 → v - speed_of_current = 8 :=
by
  sorry

end man_l723_723425


namespace velocity_at_3_velocity_at_4_l723_723453

-- Define the distance as a function of time
def s (t : ℝ) : ℝ := 3 * t^2 + 2 * t

-- Define the velocity as the derivative of the distance
noncomputable def v (t : ℝ) : ℝ := deriv s t

theorem velocity_at_3 : v 3 = 20 :=
by
  sorry

theorem velocity_at_4 : v 4 = 26 :=
by
  sorry

end velocity_at_3_velocity_at_4_l723_723453


namespace cyclic_quadrilateral_AXYI_l723_723648

noncomputable theory

variables {A B C D E F X Y I : Type} [HM : MetricSpace I] [I₀ : incenter A B C]

/-- A theorem stating that the quadrilateral AXYI is cyclic when A, B, and C are the vertices of a triangle
    with angle bisectors AD, BE, and CF and I as the incenter. The perpendicular bisector of AD intersects
    BE at X and CF at Y. -/
theorem cyclic_quadrilateral_AXYI
  (h₁ : is_triangle A B C)
  (h₂ : is_angle_bisector A D B C)
  (h₃ : is_angle_bisector B E A C)
  (h₄ : is_angle_bisector C F A B)
  (h₅ : is_incenter I A B C)
  (h₆ : is_perpendicular_bisector I AD)
  (h₇ : is_intersection I AD BE X)
  (h₈ : is_intersection I AD CF Y) :
  is_cyclic_quadrilateral A X Y I := 
sorry

end cyclic_quadrilateral_AXYI_l723_723648


namespace sin_cos_eq_cos_sin_cond_l723_723541

theorem sin_cos_eq_cos_sin_cond {α : ℝ} :
  ({Real.sin α, Real.cos (2 * α)} = {Real.cos α, Real.sin (2 * α)}) →
  (∃ k : ℤ, α = 2 * k * Real.pi) :=
by
  sorry

end sin_cos_eq_cos_sin_cond_l723_723541


namespace correct_statement_l723_723790

-- Conditions as definitions
def deductive_reasoning (p q r : Prop) : Prop :=
  (p → q) → (q → r) → (p → r)

def correctness_of_conclusion := true  -- Indicates statement is defined to be correct

def pattern_of_reasoning (p q r : Prop) : Prop :=
  deductive_reasoning p q r

-- Statement to prove
theorem correct_statement (p q r : Prop) :
  pattern_of_reasoning p q r = deductive_reasoning p q r :=
by sorry

end correct_statement_l723_723790


namespace oranges_thrown_away_l723_723854

theorem oranges_thrown_away (original_oranges: ℕ) (new_oranges: ℕ) (total_oranges: ℕ) (x: ℕ)
  (h1: original_oranges = 5) (h2: new_oranges = 28) (h3: total_oranges = 31) :
  original_oranges - x + new_oranges = total_oranges → x = 2 :=
by
  intros h_eq
  -- Proof omitted
  sorry

end oranges_thrown_away_l723_723854


namespace Ahmed_goat_count_l723_723863

theorem Ahmed_goat_count : 
  let A := 7 in
  let B := 2 * A + 5 in
  let C := B - 6 in
  C = 13 :=
by
  let A := 7
  let B := 2 * A + 5
  let C := B - 6
  show C = 13
  sorry

end Ahmed_goat_count_l723_723863


namespace exists_nat_number_reduce_by_57_l723_723498

theorem exists_nat_number_reduce_by_57 :
  ∃ (N : ℕ), ∃ (k : ℕ) (a x : ℕ),
    N = 10^k * a + x ∧
    10^k * a + x = 57 * x ∧
    N = 7125 :=
sorry

end exists_nat_number_reduce_by_57_l723_723498


namespace number_of_prime_factors_of_30_factorial_l723_723129

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723129


namespace factorial_prime_factors_l723_723157

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723157


namespace factorial_prime_factors_l723_723154

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723154


namespace harriet_speed_l723_723420

-- Define the conditions
def return_speed := 140 -- speed from B-town to A-ville in km/h
def total_trip_time := 5 -- total trip time in hours
def trip_time_to_B := 2.8 -- trip time from A-ville to B-town in hours

-- Define the theorem to prove
theorem harriet_speed {r_speed : ℝ} {t_time : ℝ} {t_time_B : ℝ} 
  (h1 : r_speed = 140) 
  (h2 : t_time = 5) 
  (h3 : t_time_B = 2.8) : 
  ((r_speed * (t_time - t_time_B)) / t_time_B) = 110 :=
by 
  -- Assume we have completed proof steps here.
  sorry

end harriet_speed_l723_723420


namespace problem1_problem2_l723_723817

-- Problem 1
theorem problem1 (α : ℝ) (h : Real.cos (π / 6 - α) = √3 / 2) :
  Real.cos (5 / 6 * π + α) - Real.sin (-α + 7 / 6 * π)^2 = - (1 + 2 * √3) / 4 :=
by  
  sorry

-- Problem 2
theorem problem2 (α : ℝ) (h1 : Real.cos α = 2 / 3) (h2 : α ∈ {x : ℝ | x ∈ (3 / 2 * π)..(2 * π)}) :
  (Real.sin(α - 2 * π) + Real.sin(-α - 3 * π) * Real.cos(α - 3 * π)) / 
  (Real.cos(π - α) - Real.cos(-π - α) * Real.cos(α - 4 * π)) = √5 / 2 :=
by 
  sorry

end problem1_problem2_l723_723817


namespace eccentricity_range_exists_lambda_l723_723551

noncomputable def ellipse_focal_distance (a b : ℝ) (h : a > b) : ℝ :=
  real.sqrt (a^2 - b^2)

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a > b) : ℝ :=
  ellipse_focal_distance a b h / a

theorem eccentricity_range (a b : ℝ) (h : a > b) :
  0 < ellipse_eccentricity a b h ∧ ellipse_eccentricity a b h < real.sqrt 3 :=
sorry

noncomputable def lambda_value (a b : ℝ) (h : a > b) : ℝ :=
  3

theorem exists_lambda (a b : ℝ) (h : a > b) (e : ℝ) (h1 : e = ellipse_eccentricity a b h) :
  λ > 0 ∧ ∀ (B A F1 : Point) (angle_BAF1 angle_BF1A : ℝ),
    angle B A F1 = lambda_value a b h * angle B F1 A :=
sorry

end eccentricity_range_exists_lambda_l723_723551


namespace solution_l723_723809

noncomputable def f (G : ℝ × ℝ → ℝ) : (ℝ × ℝ → ℝ) := sorry

noncomputable def f1 (x y : ℝ) : ℝ := 
  if x <= y then x else y

noncomputable def f2 (x y : ℝ) : ℝ := x * y

variables {I : Set ℝ} (k : ℝ) (G : ℝ × ℝ → ℝ) (x y z : ℝ)

def isMappingFromGtoI (f : ℝ × ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → (f(x, y) ∈ I)

axiom condition1 : ∀ x y z : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ z ∧ z ≤ 1 → 
                              f(f(x, y), z) = f(x, f(y, z))

axiom condition2x : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f(x, 1) = x
axiom condition2y : ∀ y : ℝ, 0 ≤ y ∧ y ≤ 1 → f(1, y) = y

axiom condition3 : ∀ x y z : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ z ∧ z ≤ 1 → 
                              f(z * x, z * y) = z^k * f(x, y)

theorem solution :
  (∀ f : ℝ × ℝ → ℝ, (isMappingFromGtoI f ∧ condition1 f ∧ condition2x f ∧ condition2y f ∧ condition3 f) →
    (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1,
      (f = f1 ∧ k = 1) ∨ (f = f2 ∧ k = 2))) :=
sorry

end solution_l723_723809


namespace find_a_l723_723944

theorem find_a (a : ℝ) : 
    ({0, 1, a^2} = {1, 0, 2a + 3}) → a = 3 :=
by
    intro h
    /- Proof will be filled in by user or another function using Lean's tactics -/
    sorry

end find_a_l723_723944


namespace prime_factors_of_30_factorial_l723_723098

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723098


namespace ordering_abc_l723_723652

noncomputable def a : ℝ := log 4 / log 5
noncomputable def b : ℝ := (log 3 / log 5) ^ 2
noncomputable def c : ℝ := log 5 / log 4

theorem ordering_abc : b < a ∧ a < c := by
  sorry

end ordering_abc_l723_723652


namespace Davids_daughter_age_l723_723404

-- Define David's age today
def David_age_today : ℕ := 40

-- Define daughter's age today as a variable
variable (D : ℕ)

-- Define the condition: in 16 years, David's age will be twice his daughter's age
def condition : Prop := (David_age_today + 16) = 2 * (D + 16)

-- The theorem statement we need to prove
theorem Davids_daughter_age :
  condition → D = 12 :=
by
  sorry  -- proof goes here

end Davids_daughter_age_l723_723404


namespace num_prime_factors_30_fac_l723_723068

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723068


namespace prime_factors_of_30_l723_723214

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723214


namespace sum_of_four_unique_digit_terms_l723_723887

theorem sum_of_four_unique_digit_terms (a b c d : ℕ) (unique_digits : (nat.digits 10 a).nodup ∧ (nat.digits 10 b).nodup ∧ (nat.digits 10 c).nodup ∧ (nat.digits 10 d).nodup)
  (different_lengths : (nat.length (nat.digits 10 a)) ≠ (nat.length (nat.digits 10 b)) ∧
                       (nat.length (nat.digits 10 a)) ≠ (nat.length (nat.digits 10 c)) ∧
                       (nat.length (nat.digits 10 a)) ≠ (nat.length (nat.digits 10 d)) ∧
                       (nat.length (nat.digits 10 b)) ≠ (nat.length (nat.digits 10 c)) ∧
                       (nat.length (nat.digits 10 b)) ≠ (nat.length (nat.digits 10 d)) ∧
                       (nat.length (nat.digits 10 c)) ≠ (nat.length (nat.digits 10 d))) 
  (sum : a + b + c + d = 2017) : 
  ∃ a b c d, a + b + c + d = 2017 :=
sorry

end sum_of_four_unique_digit_terms_l723_723887


namespace min_value_is_45_l723_723655

noncomputable def min_value (x y z : ℝ) (h : x + y + z = 5) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : ℝ :=
  9 / x + 25 / y + 49 / z

theorem min_value_is_45 : ∃ x y z : ℝ, x + y + z = 5 ∧ 0 < x ∧ 0 < y ∧ 0 < z ∧ min_value x y z (by assumption) (by assumption) (by assumption) = 45 :=
sorry

end min_value_is_45_l723_723655


namespace tangent_line_through_point_l723_723913

open Real

noncomputable def is_tangent_to_circle (line : ℝ → ℝ) (C : ℝ × ℝ) (r : ℝ) : Prop :=
    ∃ P : ℝ × ℝ, dist P C = r ∧ ∀ Q : ℝ × ℝ, dist Q C < r → line Q.1 ≠ Q.2

theorem tangent_line_through_point (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ) :
    P = (-2, 2) → C = (-1, 0) → r = 1 → 
    (∃ k : ℝ, ∀ x : ℝ, is_tangent_to_circle (λ x, k * (x + 2) + 2) C r) → 
    is_tangent_to_circle (λ x, -2) C r ∨ is_tangent_to_circle (λ x, -3/4 * x + 1/2) C r :=
by
  sorry

end tangent_line_through_point_l723_723913


namespace express_y_in_terms_of_x_l723_723529

theorem express_y_in_terms_of_x (x y : ℝ) (h : 5 * x + y = 4) : y = 4 - 5 * x :=
by
  /- Proof to be filled in here. -/
  sorry

end express_y_in_terms_of_x_l723_723529


namespace number_of_intersection_points_l723_723744

def curve1 (x y : ℝ) : Prop := x = y^2
def curve2 (x y : ℝ) : Prop := y = x^2

noncomputable def intersection_points : ℕ :=
  (by
    let points := {p : ℝ × ℝ | curve1 p.1 p.2 ∧ curve2 p.1 p.2}
    exact set.size points
  )

theorem number_of_intersection_points : intersection_points = 2 :=
sorry

end number_of_intersection_points_l723_723744


namespace compute_c_minus_d_cubed_l723_723315

def multiples_of_6_less_than (n : ℕ) : ℕ :=
  (List.range n).count (λ k => k > 0 ∧ k % 6 = 0)

def multiples_of_12_less_than (n : ℕ) : ℕ :=
  (List.range n).count (λ k => k > 0 ∧ k % 12 = 0)

theorem compute_c_minus_d_cubed : (multiples_of_6_less_than 50 - multiples_of_12_less_than 50) ^ 3 = 64 := by
  sorry

end compute_c_minus_d_cubed_l723_723315


namespace factorial_30_prime_count_l723_723190

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723190


namespace locus_of_points_M_l723_723549

noncomputable def angle (OA OB: Type) : Type :=
sorry -- define angle type

noncomputable def segment (P Q: Type) : Type :=
sorry -- define segment type

noncomputable def point_inside_angle (O A B: Type) : Type :=
sorry -- define point inside angle type

noncomputable def area_of_triangle (A B C: Type) : ℝ :=
sorry --define function to calculate area of triangle

theorem locus_of_points_M (OA OB : Type) (O P Q A B : Type)
  (angle_OA_OB : angle OA OB) (segment_AP : segment P A)
  (segment_BQ : segment Q B) (inside_O : point_inside_angle O A B)
  (h : area_of_triangle O A P + area_of_triangle O B Q = S) :
  ∀ M, M ∈ segment P Q → area_of_triangle M A P + area_of_triangle M B Q = S :=
sorry

end locus_of_points_M_l723_723549


namespace lines_concurrent_l723_723769

-- Definitions and setup for the problem
variables {A B C D K L P M N : Point}
variables (Gamma1 Gamma2 : Circle) (AC BD : Segment) (not_on_AD : \[point\] P \notin \[line\] AD)
variables (intersects_Gamma1 : \[line\] CP \intersect \[circle\] Gamma1 = {C, M})
variables (intersects_Gamma2 : \[line\] BP \intersect \[circle\] Gamma2 = {B, N})

-- The circles Gamma1 and Gamma2 intersect at points K and L
axiom Gamma1_intersects_Gamma2_at_KL : Gamma1 \intersect Gamma2 = {K, L}

-- The circles with given diameters
axiom Gamma1_diameter_AC : Gamma1.diameter = AC
axiom Gamma2_diameter_BD : Gamma2.diameter = BD

-- Proof goal: The lines (AM), (DN), and (KL) are concurrent
theorem lines_concurrent :
  concurrent \[line\] AM \[line\] DN \[line\] KL :=
sorry

end lines_concurrent_l723_723769


namespace projection_of_rectangular_frame_is_parallelogram_l723_723746

-- Defining the problem conditions
def is_rectangle (frame : Type) : Prop := 
  ∃ (a b : ℝ), a > 0 ∧ b > 0

def illuminated_by_sunlight (frame : Type) : Prop := 
  frame = "rectangular frame illuminated by sunlight"

def projection (frame : Type) : Type := sorry  -- placeholder, typically would be a function

-- Defining the theorem to prove
theorem projection_of_rectangular_frame_is_parallelogram (frame : Type) 
  (h1 : is_rectangle frame) 
  (h2 : illuminated_by_sunlight frame) :
  ∃ a b : ℝ, a ≠ b ∧ ∃ (shape : Type), shape = "parallelogram" :=
begin
  -- The proof would go here
  sorry
end

end projection_of_rectangular_frame_is_parallelogram_l723_723746


namespace prime_factors_30_fac_eq_10_l723_723147

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723147


namespace sqrt_solution_l723_723983

theorem sqrt_solution (x : ℝ) (h : x = Real.sqrt (1 + x)) : 1 < x ∧ x < 2 :=
by
  sorry

end sqrt_solution_l723_723983


namespace max_value_of_tan_B_l723_723566

noncomputable def max_tan_B (A B : ℝ) (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) (h : tan (A + B) = 2 * tan A) : ℝ :=
  max (tan B)

theorem max_value_of_tan_B (A B : ℝ) (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) (h : tan (A + B) = 2 * tan A) :
  max_tan_B A B hA hB h = (√2 / 4) :=
sorry

end max_value_of_tan_B_l723_723566


namespace prime_factors_30_fac_eq_10_l723_723146

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723146


namespace converse_false_inverse_false_contrapositive_true_l723_723794

-- Definitions
variable (l : Line)
def slope_is_minus_one (l : Line) : Prop := l.slope = -1
def intercepts_are_equal (l : Line) : Prop := l.x_intercept = l.y_intercept

-- Main Theorems
theorem converse_false (h : intercepts_are_equal l) : slope_is_minus_one l ↔ False := sorry

theorem inverse_false (h : ¬ slope_is_minus_one l) : ¬ intercepts_are_equal l ↔ False := sorry

theorem contrapositive_true (h : ¬ intercepts_are_equal l) : ¬ slope_is_minus_one l := sorry

end converse_false_inverse_false_contrapositive_true_l723_723794


namespace prime_factors_of_30_factorial_l723_723182

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723182


namespace prime_factors_of_30_factorial_l723_723177

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723177


namespace derivative_at_pi_over_4_l723_723581

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem derivative_at_pi_over_4 : (deriv f) (Real.pi / 4) = 0 := 
by
  sorry

end derivative_at_pi_over_4_l723_723581


namespace total_snow_volume_l723_723747

theorem total_snow_volume (length width initial_depth additional_depth: ℝ) 
  (h_length : length = 30) 
  (h_width : width = 3) 
  (h_initial_depth : initial_depth = 3 / 4) 
  (h_additional_depth : additional_depth = 1 / 4) : 
  (length * width * initial_depth) + (length * width * additional_depth) = 90 := 
by
  -- proof steps would go here
  sorry

end total_snow_volume_l723_723747


namespace number_of_prime_factors_thirty_factorial_l723_723020

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723020


namespace find_min_n_l723_723412

def Pigeonhole_smallest_n (s : Set ℕ) (p : ℕ → ℕ → Prop) : ℕ :=
  ∃ n : ℕ, (∀ subset : Finset ℕ, subset.card = n → ∃ x y ∈ subset, p x y) ∧
           ∀ m : ℕ, m < n → ∃ subset : Finset ℕ, subset.card = m ∧ ∀ x y ∈ subset, ¬ p x y

def diff_8 (x y : ℕ) : Prop := abs (x - y) = 8

theorem find_min_n : 
  Pigeonhole_smallest_n (Finset.range 21).toSet diff_8 = 9 :=
sorry

end find_min_n_l723_723412


namespace days_C_alone_l723_723435

theorem days_C_alone (r_A r_B r_C : ℝ) (h1 : r_A + r_B = 1 / 3) (h2 : r_B + r_C = 1 / 6) (h3 : r_A + r_C = 5 / 18) : 
  1 / r_C = 18 := 
  sorry

end days_C_alone_l723_723435


namespace prime_factors_of_30_factorial_l723_723107

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723107


namespace boxes_needed_l723_723718

theorem boxes_needed (total_muffins available_boxes muffins_per_box : ℕ) (h1 : total_muffins = 95) 
  (h2 : available_boxes = 10) (h3 : muffins_per_box = 5) : 
  ((total_muffins - (available_boxes * muffins_per_box)) / muffins_per_box) = 9 := 
by
  -- the proof will be constructed here
  sorry

end boxes_needed_l723_723718


namespace concurrency_of_gk_hl_ij_l723_723297

theorem concurrency_of_gk_hl_ij 
  (A B C D E F P G H I J K L : Type)
  [F : field A]
  [tA : T A]
  [tB : T B]
  [tC : T C]
  [tD : T D]
  [tE : T E]
  [tF : T F]
  [tG : midpoint B C G]
  [tH : midpoint C A H]
  [tI : midpoint A B I]
  [tJ : midpoint D E J]
  [tK : midpoint E F K]
  [tL : midpoint F D L]
  [tP : concurrent (line_through A D) (line_through B E) (line_through C F)]
  (eq : ∀ p q r : Type, field p ∧ field q ∧ field r → Prop): 
  concurrent (line_through G K) (line_through H L) (line_through I J) :=
sorry

end concurrency_of_gk_hl_ij_l723_723297


namespace probability_different_plants_l723_723745

theorem probability_different_plants :
  let plants := 4
  let total_combinations := plants * plants
  let favorable_combinations := total_combinations - plants
  (favorable_combinations : ℚ) / total_combinations = 3 / 4 :=
by
  sorry

end probability_different_plants_l723_723745


namespace sum_x_y_z_l723_723319

/- Define the polynomial and its roots -/
def poly : Polynomial ℚ := Polynomial.C (-8) + Polynomial.C 14 * Polynomial.X + Polynomial.C (-7) * Polynomial.X^2 + Polynomial.X^3

noncomputable def t (k : ℕ) : ℚ := 
  if k = 0 then 3 else if k = 1 then 7 else if k = 2 then 15 else sorry -- Define t_k as per the problem conditions

/- Define the recursive relationship for the sequence t_k -/
axiom t_recurrence (k : ℕ) : t (k + 1) = x * t k + y * t (k - 1) + z * t (k - 2)

/- Prove the main statement -/
theorem sum_x_y_z : x + y + z = 3 :=
sorry

end sum_x_y_z_l723_723319


namespace prime_factors_30_fac_eq_10_l723_723141

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723141


namespace eq_or_sum_zero_l723_723955

variables (a b c d : ℝ)

theorem eq_or_sum_zero (h : (3 * a + 2 * b) / (2 * b + 4 * c) = (4 * c + 3 * d) / (3 * d + 3 * a)) :
  3 * a = 4 * c ∨ 3 * a + 3 * d + 2 * b + 4 * c = 0 :=
by sorry

end eq_or_sum_zero_l723_723955


namespace crayons_slightly_used_l723_723760

theorem crayons_slightly_used (total_crayons : ℕ) (new_fraction : ℚ) (broken_fraction : ℚ) 
  (htotal : total_crayons = 120) (hnew : new_fraction = 1 / 3) (hbroken : broken_fraction = 20 / 100) :
  let new_crayons := total_crayons * new_fraction
  let broken_crayons := total_crayons * broken_fraction
  let slightly_used_crayons := total_crayons - new_crayons - broken_crayons
  slightly_used_crayons = 56 := 
by
  -- This is where the proof would go
  sorry

end crayons_slightly_used_l723_723760


namespace prime_factors_of_30_factorial_l723_723111

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723111


namespace factorial_30_prime_count_l723_723193

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723193


namespace num_prime_factors_30_factorial_l723_723028

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723028


namespace ferris_wheel_seats_l723_723383

theorem ferris_wheel_seats (people_waiting total_not_riding : ℕ) (h1: people_waiting = 92) (h2 : total_not_riding = 36) : 
  people_waiting - total_not_riding = 56 :=
by 
  rw [h1, h2]
  exact rfl

end ferris_wheel_seats_l723_723383


namespace prime_factors_of_30_l723_723217

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723217


namespace sum_zero_implies_inequality_l723_723660

variable {a b c d : ℝ}

theorem sum_zero_implies_inequality
  (h : a + b + c + d = 0) :
  5 * (a * b + b * c + c * d) + 8 * (a * c + a * d + b * d) ≤ 0 := 
sorry

end sum_zero_implies_inequality_l723_723660


namespace num_prime_factors_30_factorial_l723_723035

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723035


namespace sqrt_twelve_simplified_l723_723695

theorem sqrt_twelve_simplified :
  (∀ (a b : ℝ), sqrt (a * b) = sqrt a * sqrt b) →
  sqrt 4 = 2 →
  sqrt 12 = 2 * sqrt 3 :=
by
  sorry

end sqrt_twelve_simplified_l723_723695


namespace find_d_l723_723295

variables (d : ℝ)

-- Defining the line equation
def line_equation (x y d : ℝ) : Prop := 3 * x + 5 * y + d = 0

-- Defining the x-intercept condition when y = 0
def x_intercept (d : ℝ) : ℝ := -d / 3

-- Defining the y-intercept condition when x = 0
def y_intercept (d : ℝ) : ℝ := -d / 5

-- Defining the sum of intercepts being 16
def sum_of_intercepts (d : ℝ) : Prop := x_intercept d + y_intercept d = 16

-- The proof statement
theorem find_d (d : ℝ) (H : sum_of_intercepts d) : d = -30 :=
by {
  sorry
}

end find_d_l723_723295


namespace base_length_of_vessel_l723_723832

def volume_of_cube (edge : ℝ) := edge^3

def volume_of_displaced_water (L width rise : ℝ) := L * width * rise

theorem base_length_of_vessel (edge width rise L : ℝ) 
  (h1 : edge = 15) (h2 : width = 15) (h3 : rise = 11.25) 
  (h4 : volume_of_displaced_water L width rise = volume_of_cube edge) : 
  L = 20 :=
by
  sorry

end base_length_of_vessel_l723_723832


namespace inclination_angle_of_PQ_l723_723553

noncomputable section

def cos (x : ℝ) : ℝ := sorry
def sin (x : ℝ) : ℝ := sorry

def pointP : ℝ × ℝ := (2 * cos (10 * real.pi / 180), 2 * sin (10 * real.pi / 180))
def pointQ : ℝ × ℝ := (2 * cos (50 * real.pi / 180), -2 * sin (50 * real.pi / 180))

theorem inclination_angle_of_PQ : (angle_between_points (pointP) (pointQ)).angle = 70 * real.pi / 180 := sorry

end inclination_angle_of_PQ_l723_723553


namespace vertical_line_divides_triangle_l723_723634

-- Define the points A, B, C
def A : (ℝ × ℝ) := (0, 3)
def B : (ℝ × ℝ) := (0, 0)
def C : (ℝ × ℝ) := (4, 0)

-- Define the height of point A and the base BC
def height_A : ℝ := 3
def base_BC : ℝ := 4

-- Define the vertical line x = a
def vertical_line (a : ℝ) : Prop := ∃ (a : ℝ), ∀ y : ℝ, (a, y) ∈ ℝ × ℝ

-- Define the total area of triangle ABC
def area_ABC : ℝ := 1 / 2 * base_BC * height_A

-- Define the area of the right-side triangle after the vertical divide
def area_right (a : ℝ) : ℝ := 1 / 2 * (base_BC - a) * height_A

-- Define the condition for equal areas
def divides_in_equal_area (a : ℝ) : Prop :=
  area_ABC / 2 = area_right a

-- Prove that a = 2 satisfies the division into equal areas
theorem vertical_line_divides_triangle :
  divides_in_equal_area 2 :=
by
  sorry

end vertical_line_divides_triangle_l723_723634


namespace find_old_salary_l723_723644

-- Define the conditions
def new_salary : ℕ := 120
def percentage_increase : ℕ := 100

-- Main theorem statement
theorem find_old_salary (new_salary_percentage: new_salary = 120)
                        (percentage_increase_hundred: percentage_increase = 100) :
  ∃ old_salary: ℕ, new_salary = old_salary * 2 ∧ percentage_increase = 100 :=
by
  use 60
  sorry

end find_old_salary_l723_723644


namespace jerky_dinner_each_day_l723_723306

variables (d : ℕ) -- number of dinner pieces per day

def pieces_of_beef_jerky_each_day := d
def pieces_of_beef_jerky_initial := 40
def pieces_of_beef_jerky_remaining := 10
def camping_days := 5
def pieces_of_beef_jerky_breakfast := 1
def pieces_of_beef_jerky_lunch := 1

theorem jerky_dinner_each_day :
  (pieces_of_beef_jerky_each_day d) = 2 :=
begin
  let remaining_pieces_before_giving = 2 * pieces_of_beef_jerky_remaining,
  let consumed_pieces = pieces_of_beef_jerky_initial - remaining_pieces_before_giving,
  let consumed_breakfast_lunch = (pieces_of_beef_jerky_breakfast + pieces_of_beef_jerky_lunch) * camping_days,
  have h : remaining_pieces_before_giving = 20,
  have h1 : consumed_pieces = 20,
  have h2 : consumed_breakfast_lunch = 10,
  let consumed_dinner = consumed_pieces - consumed_breakfast_lunch,
  have h3 : consumed_dinner = 10,
  have h4 : consumed_dinner = pieces_of_beef_jerky_each_day d * camping_days,
  let pieces_dinner = consumed_dinner / camping_days,
  have h5 : pieces_dinner = 2,
  exact h5
end

end jerky_dinner_each_day_l723_723306


namespace convert_to_exponential_form_l723_723414

noncomputable def magnitude (z : ℂ) : ℝ :=
  real.sqrt (z.re ^ 2 + z.im ^ 2)

noncomputable def angle (z : ℂ) : ℝ :=
  real.arctan (z.im / z.re)

theorem convert_to_exponential_form (a b : ℝ) (z : ℂ) 
  (hz : z = a + b * complex.I) 
  (hlt : a = 2) 
  (hlb : b = -2 * real.sqrt 3) :
  angle z = 5 * real.pi / 3 := 
by sorry

end convert_to_exponential_form_l723_723414


namespace sum_of_angles_in_triangle_l723_723272

theorem sum_of_angles_in_triangle (A B : ℝ) (h : tan A + tan B + sqrt 3 = sqrt 3 * tan A * tan B) (h_pos : 0 < A + B) (h_less : A + B < π) : A + B = 2 * π / 3 :=
sorry

end sum_of_angles_in_triangle_l723_723272


namespace largest_n_unique_k_l723_723770

theorem largest_n_unique_k (n : ℕ) (h : ∃ k : ℕ, (9 / 17 : ℚ) < n / (n + k) ∧ n / (n + k) < (8 / 15 : ℚ) ∧ ∀ k' : ℕ, ((9 / 17 : ℚ) < n / (n + k') ∧ n / (n + k') < (8 / 15 : ℚ)) → k' = k) : n = 72 :=
sorry

end largest_n_unique_k_l723_723770


namespace train_crossing_time_l723_723858

-- Define the given conditions
def train_speed_km_per_hr : ℝ := 180
def train_length_meters : ℝ := 350

-- Convert speed from km/hr to m/s
def train_speed_m_per_s : ℝ := (train_speed_km_per_hr * 1000) / 3600

-- Define the expected time to cross the pole
def expected_time_seconds : ℝ := 7

-- Statement of the problem in Lean 4
theorem train_crossing_time :
  (train_length_meters / train_speed_m_per_s) = expected_time_seconds :=
by 
  sorry

end train_crossing_time_l723_723858


namespace quadratic_function_solution_l723_723598

theorem quadratic_function_solution (m : ℝ) :
  (m^2 - 2 = 2) ∧ (m + 2 ≠ 0) → m = 2 :=
by
  intro h
  cases h with h1 h2
  have h3 : m^2 = 4 := by linarith
  have h4 : m = 2 ∨ m = -2 := by nlinarith
  cases h4
  · exact h4
  · contradiction

end quadratic_function_solution_l723_723598


namespace breaking_load_l723_723449

-- Statement of the problem in Lean 4
theorem breaking_load (T H : ℝ) (h₁ : T = 3) (h₂ : H = 6) :
  let L := (30 * T^3) / H^2 in
  L = 22.5 :=
by
  -- Begin the lemma
  sorry -- Proof omitted

end breaking_load_l723_723449


namespace prime_factors_of_30_factorial_l723_723167

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723167


namespace sequence_properties_l723_723563

-- Declarations for the geometric sequence
variables {a : ℕ → ℕ}
axiom geometric_monotonic : ∀ n : ℕ, a (n+1) ≥ a n
axiom geometric_conditions : a 2 + a 3 + a 4 = 28 ∧ (a 3 + 2) * 2 = a 2 + a 4

-- Declarations for the sequence bn and the sum Sn
def b (n : ℕ) := a n * nat.log 2 (a n)
def S (n : ℕ) := ∑ i in finset.range n.succ, b i

-- Theorem stating the results
theorem sequence_properties :
  (∀ n : ℕ, a n = 2^n) ∧ (∀ n : ℕ, S n = (n-1) * 2^(n+1) + 2) :=
sorry

end sequence_properties_l723_723563


namespace fifth_equation_correct_l723_723679

def fifth_equation (x : ℕ) := x= 1^3 + 2^3 + 3^3 + 4^3 + 5^3 = 15^2

theorem fifth_equation_correct : fifth_equation (225) :=
by
  unfold fifth_equation
  sorry

end fifth_equation_correct_l723_723679


namespace derivative_at_pi_div_4_l723_723584

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem derivative_at_pi_div_4 :
  (deriv f) (Real.pi / 4) = 0 :=
sorry

end derivative_at_pi_div_4_l723_723584


namespace change_in_expression_l723_723571

theorem change_in_expression (x a : ℝ) (ha : 0 < a) :
  (x^3 - 3*x + 1) + (3*a*x^2 + 3*a^2*x + a^3 - 3*a) = (x + a)^3 - 3*(x + a) + 1 ∧
  (x^3 - 3*x + 1) + (-3*a*x^2 + 3*a^2*x - a^3 + 3*a) = (x - a)^3 - 3*(x - a) + 1 :=
by sorry

end change_in_expression_l723_723571


namespace number_of_prime_factors_of_30_factorial_l723_723123

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723123


namespace largest_convex_polygon_l723_723895

theorem largest_convex_polygon (n : ℕ) :
  ∀ (k : ℕ),
  k ≤ 561 →
  ∃ (vertices : fin 2004 × fin 2004 → Prop), 
  (∀ v ∈ vertices, ∃ i j, vertices (fin.mk i (by simp [nat.lt_succ_self])) (fin.mk j (by simp [nat.lt_succ_self]))) ∧
  (∀ v w u ∈ vertices, ∠v w u < 180) →
  n ≤ 561 :=
by intros
sorry

end largest_convex_polygon_l723_723895


namespace range_of_a_l723_723933

theorem range_of_a (a : ℝ) (p q : set ℝ)
  (hp : ∀ x, p x ↔ 2 * x^2 - 3 * x + 1 ≤ 0)
  (hq : ∀ x, q x ↔ x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0)
  (h_not_necessary_sufficient : ¬(∀ x, (p x ↔ q x) ↔ (¬p x ↔ ¬q x))) :
  0 ≤ a ∧ a ≤ (1/2) :=
by
  sorry

end range_of_a_l723_723933


namespace find_m_n_find_max_profit_day_count_days_with_min_profit_l723_723421

noncomputable def y (m n : ℝ) (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x < 20 then m * (x : ℝ) - 76 * m
  else if 20 ≤ x ∧ x ≤ 30 then n
  else 0

noncomputable def kilograms_sold (x : ℕ) : ℕ :=
  4 * x + 16

noncomputable def profit_per_day (m n : ℝ) (x : ℕ) : ℝ :=
  let yx := y m n x 
  let revenue := yx * (kilograms_sold x : ℝ)
  let cost := 18 * (kilograms_sold x : ℝ)
  revenue - cost

-- Problem 1: Find m and n
theorem find_m_n (m n : ℝ) :
  y m n 12 = 32 ∧ y m n 26 = 25 → 
  m = -0.5 ∧ n = 25 := sorry

-- Problem 2: Find the day of maximum profit and the corresponding profit
theorem find_max_profit_day (m n : ℝ) :
  m = -0.5 ∧ n = 25 →
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 30 ∧ 
  ∀ y : ℕ, 1 ≤ y ∧ y ≤ 30 → profit_per_day m n x ≥ profit_per_day m n y ∧
  profit_per_day m n x = 968 ∧ x = 18 := sorry

-- Problem 3: Find the number of days with profit at least 870 yuan
theorem count_days_with_min_profit (m n : ℝ) :
  m = -0.5 ∧ n = 25 →
  (∑ x in finset.range 31, if profit_per_day m n x ≥ 870 then 1 else 0) = 12 := sorry

end find_m_n_find_max_profit_day_count_days_with_min_profit_l723_723421


namespace circle_radius_squared_l723_723620

-- Let r be the radius of the circle.
-- Let AB and CD be chords of the circle with lengths 10 and 7 respectively.
-- Let the extensions of AB and CD intersect at a point P outside the circle.
-- Let ∠APD be 60 degrees.
-- Let BP be 8.

theorem circle_radius_squared
  (r : ℝ)       -- radius of the circle
  (AB : ℝ)     -- length of chord AB
  (CD : ℝ)     -- length of chord CD
  (APD : ℝ)    -- angle APD
  (BP : ℝ)     -- length of segment BP
  (hAB : AB = 10)
  (hCD : CD = 7)
  (hAPD : APD = 60)
  (hBP : BP = 8)
  : r^2 = 73 := 
  sorry

end circle_radius_squared_l723_723620


namespace min_period_of_function_l723_723742

noncomputable def f (x : ℝ) : ℝ := cos x ^ 4 - sin x ^ 4 + 2

theorem min_period_of_function : 
  ∀ x, f (x + π) = f x :=
by sorry

end min_period_of_function_l723_723742


namespace problem_solution_mLE1_simplify_expression_inequality_solution_l723_723535

theorem problem_solution_mLE1 (m : ℤ) (x y : ℤ) (h1 : -x - 2y = 1 - 3 * m)
    (h2 : 3 * x + 4 * y = 2 * m) (h3 : x + y ≥ 0) : m = 0 ∨ m = 1 := sorry

theorem simplify_expression (m : ℤ) (h : m ≤ 1) : |m - 3| + |5 - 2 * m| = 8 - 3 * m := sorry

theorem inequality_solution (m : ℤ) (x : ℤ) (h1 : m ≤ 1) (h2 : m > 0) (h3: x > -1) :
    m * (x + 1) > 0 := sorry

end problem_solution_mLE1_simplify_expression_inequality_solution_l723_723535


namespace marks_difference_l723_723713

theorem marks_difference (A B C D E : ℕ) 
  (h1 : (A + B + C) / 3 = 48) 
  (h2 : (A + B + C + D) / 4 = 47) 
  (h3 : E > D) 
  (h4 : (B + C + D + E) / 4 = 48) 
  (h5 : A = 43) : 
  E - D = 3 := 
sorry

end marks_difference_l723_723713


namespace triangle_problem_l723_723273

noncomputable def triangle_sum : Real := sorry

theorem triangle_problem
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (hA : A = π / 6) -- A = 30 degrees
  (h_a : a = Real.sqrt 3) -- a = √3
  (h_law_of_sines : ∀ (x : ℝ), x = 2 * triangle_sum * Real.sin x) -- Law of Sines
  (h_sin_30 : Real.sin (π / 6) = 1 / 2) -- sin 30 degrees = 1/2
  : (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) 
  = 2 * Real.sqrt 3 := sorry

end triangle_problem_l723_723273


namespace avg_decrease_l723_723714

-- Conditions
def avg_6 (s : Fin 6 → ℝ) : ℝ := 16
def seventh_observation : ℝ := 9

-- The theorem stating that the average decreases by 1
theorem avg_decrease :
  ∀ (s : Fin 6 → ℝ),
  (∑ i, s i) / 6 = 16 →
  ((∑ i, s i) + 9) / 7 = 15 :=
by
  intros s h
  sorry

end avg_decrease_l723_723714


namespace total_disks_in_bag_l723_723503

theorem total_disks_in_bag 
  (total_disks : ℕ) 
  (blue_disks yellow_disks green_disks : ℕ)
  (h1 : blue_disks + yellow_disks + green_disks = total_disks) 
  (h2 : blue_disks : yellow_disks : green_disks = 3 : 7 : 8)
  (h3 : green_disks = blue_disks + 35) :
  total_disks = 126 := 
sorry

end total_disks_in_bag_l723_723503


namespace cos_angle_subtract_l723_723935

theorem cos_angle_subtract (α : ℝ) (hα1 : α > 0) (hα2 : α < real.pi / 2) (h : real.tan α = 2) : 
  real.cos (α - real.pi / 4) = (3 * real.sqrt 10) / 10 :=
sorry

end cos_angle_subtract_l723_723935


namespace concyclic_AIMN_l723_723324

-- geometric setup based on the problem conditions
variables {A B C I D E F M N : Point}
variables {triangleABC : Triangle A B C}
variables {angle_bisectorAI : AngleBisector A I}
variables {angle_bisectorBI : AngleBisector B I}
variables {angle_bisectorCI : AngleBisector C I}
variables {incenterI : Incenter I A B C}
variables {perpendicular_bisectorAD : PerpendicularBisector A D}
variables {intersectionM : Intersection perpendicular_bisectorAD angle_bisectorBI M}
variables {intersectionN : Intersection perpendicular_bisectorAD angle_bisectorCI N}

-- main theorem to prove
theorem concyclic_AIMN :
  Concyclic A I M N :=
sorry

end concyclic_AIMN_l723_723324


namespace brendan_earnings_l723_723878

-- Definitions based on conditions
def recharge_half (E : ℝ) : ℝ := (1/2) * E

def remaining_money (E : ℝ) : ℝ := recharge_half E + 1000

-- The proof statement problem
theorem brendan_earnings (E : ℝ) (h1 : remaining_money E = 2500) : E = 3000 :=
by
  sorry

end brendan_earnings_l723_723878


namespace sequence_general_term_l723_723639

noncomputable def sequence (n : ℕ) : ℕ → ℝ
| 1 => 3
| (n + 1) => ((n * sequence n) / (n + 1))

theorem sequence_general_term (n : ℕ) (hn : n > 0) : sequence n = 3 / n := by
  sorry

end sequence_general_term_l723_723639


namespace length_AB_l723_723872

-- Definitions of the lengths and variables involved
variable (a b : ℝ) -- Assuming they're real numbers since they represent lengths

-- Definitions for angles and parallel lines
variable {B D : ℝ} -- Angles in radians or degrees, unspecified here but needed for relations

-- Conditions of the problem
axiom parallel_AB_CD : ∀ (AB CD : set ℝ₂), parallel AB CD
axiom angle_D_twice_B : ∀ {B D : ℝ}, D = 2 * B
axiom length_AD : ∀ {AD : ℝ}, AD = a
axiom length_CD : ∀ {CD : ℝ}, CD = b

-- Main theorem statement
theorem length_AB (a b : ℝ) 
  (AB CD : set ℝ₂)
  (B D : ℝ)
  (h_parallel : parallel AB CD)
  (h_angle : D = 2 * B)
  (h_AD : AD = a)
  (h_CD : CD = b) : 
  length AB = a + b :=
sorry

end length_AB_l723_723872


namespace valid_p_interval_l723_723484

theorem valid_p_interval :
  ∀ p, (∀ q, q > 0 → (4 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q) ↔ 0 ≤ p ∧ p < 4 :=
sorry

end valid_p_interval_l723_723484


namespace num_prime_factors_30_factorial_l723_723242

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723242


namespace tomatoes_ruined_percentage_l723_723390

-- The definitions from the problem conditions
def tomato_cost_per_pound : ℝ := 0.80
def tomato_selling_price_per_pound : ℝ := 0.977777777777778
def desired_profit_percent : ℝ := 0.10
def revenue_equal_cost_plus_profit_cost_fraction : ℝ := (tomato_cost_per_pound + (tomato_cost_per_pound * desired_profit_percent))

-- The theorem stating the problem and the expected result
theorem tomatoes_ruined_percentage :
  ∀ (W : ℝ) (P : ℝ),
  (0.977777777777778 * (1 - P / 100) * W = (0.80 * W + 0.08 * W)) →
  P = 10.00000000000001 :=
by
  intros W P h
  have eq1 : 0.977777777777778 * (1 - P / 100) = 0.88 := sorry
  have eq2 : 1 - P / 100 = 0.8999999999999999 := sorry
  have eq3 : P / 100 = 0.1000000000000001 := sorry
  exact sorry

end tomatoes_ruined_percentage_l723_723390


namespace sum_of_coefficients_eq_five_l723_723918

def polynomial := 3 * (2 * x^6 - x^5 + 4 * x^3 - 7) - 5 * (x^4 - 2 * x^3 + 3 * x^2 + 1) + 6 * (x^7 - 5)

theorem sum_of_coefficients_eq_five : 
  let p := polynomial 
  (eval 1 p) = 5 := by
  sorry

end sum_of_coefficients_eq_five_l723_723918


namespace num_prime_factors_of_30_l723_723228

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723228


namespace sequence_80th_number_is_8_l723_723395

theorem sequence_80th_number_is_8 :
  (∃ n : ℕ, ∑ i in finset.range (n + 1), (2 * (i + 1))^2 ≥ 80 ∧ ∑ i in finset.range n, (2 * (i + 1))^2 < 80) →
  2 * (4) = 8 :=
begin
  sorry
end

end sequence_80th_number_is_8_l723_723395


namespace sequence_to_geometric_l723_723433

variable (a : ℕ → ℝ)

def seq_geom (a : ℕ → ℝ) : Prop :=
∀ m n, a (m + n) = a m * a n

def condition (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 2) = a n * a (n + 1)

theorem sequence_to_geometric (a1 a2 : ℝ) (h1 : a 1 = a1) (h2 : a 2 = a2) (h : ∀ n, a (n + 2) = a n * a (n + 1)) :
  a1 = 1 → a2 = 1 → seq_geom a :=
by
  intros ha1 ha2
  have h_seq : ∀ n, a n = 1 := sorry
  intros m n
  sorry

end sequence_to_geometric_l723_723433


namespace book_distribution_ways_l723_723846

/-- 
Problem Statement: We have 8 identical books. We want to find out the number of ways to distribute these books between the library and checked out such that at least one book is in the library and at least one book is checked out. The expected answer is 7.
-/
theorem book_distribution_ways : ∃ n : ℕ, n = 7 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 7 ↔ k books in library means exactly 8 - k books are checked out) :=
by
  sorry

end book_distribution_ways_l723_723846


namespace hawks_win_at_least_4_l723_723711

noncomputable def hawks_win_probability : ℚ :=
  let p_win := 0.8 in
  let p_lose := 1 - p_win in
  let prob_4_wins := (5.choose 4) * (p_win ^ 4) * (p_lose) in
  let prob_5_wins := (p_win ^ 5) in
  prob_4_wins + prob_5_wins

theorem hawks_win_at_least_4 : hawks_win_probability = 73728 / 100000 :=
sorry

end hawks_win_at_least_4_l723_723711


namespace period_of_f_range_of_f_l723_723345

-- Define the function f
def f (x : ℝ) : ℝ := abs (sin x) + abs (cos x)

-- Prove that the period of the function f is π/2
theorem period_of_f : ∀ x : ℝ, f (x + π / 2) = f x :=
by 
  sorry

-- Prove that the range of the function f is [1, √2]
theorem range_of_f : Set.range f = set.Icc 1 (Real.sqrt 2) :=
by 
  sorry

end period_of_f_range_of_f_l723_723345


namespace prime_factors_of_30_l723_723203

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723203


namespace spending_on_games_l723_723975

-- Definitions converted from conditions
def totalAllowance := 48
def fractionClothes := 1 / 4
def fractionBooks := 1 / 3
def fractionSnacks := 1 / 6
def spentClothes := fractionClothes * totalAllowance
def spentBooks := fractionBooks * totalAllowance
def spentSnacks := fractionSnacks * totalAllowance
def spentGames := totalAllowance - (spentClothes + spentBooks + spentSnacks)

-- The theorem that needs to be proven
theorem spending_on_games : spentGames = 12 :=
by sorry

end spending_on_games_l723_723975


namespace sum_a_n_sum_b_n_l723_723680

-- Problem 1: Prove that a_1 + a_2 + ... + a_{199} = 199 / 200
theorem sum_a_n : (∑ n in Finset.range 199, (1 / (n + 1) - 1 / (n + 2))) = 199 / 200 := by
  sorry

-- Problem 2: Prove that b_1 + b_2 + ... + b_{29} = 667 / 930
theorem sum_b_n : (∑ n in Finset.range 29, (1 / 2 * (1 / (n + 1) - 1 / (n + 3)))) = 667 / 930 := by
  sorry

end sum_a_n_sum_b_n_l723_723680


namespace number_of_prime_factors_thirty_factorial_l723_723015

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l723_723015


namespace ratio_fenced_region_l723_723458

theorem ratio_fenced_region (L W : ℝ) (k : ℝ) 
  (area_eq : L * W = 200)
  (fence_eq : 2 * W + L = 40)
  (mult_eq : L = k * W) :
  k = 2 :=
by
  sorry

end ratio_fenced_region_l723_723458


namespace cyclic_quadrilateral_ABCD_l723_723649

-- Definitions for the problem
variables {A B C D P Q E : Type}

-- Assume the appropriate geometric properties
variables (isConvexQuadrilateral : convex_quadrilateral A B C D)
variables (P_in_ABC : point_in_quad P A B C)
variables (Q_in_ABD : point_in_quad Q A B D)
variables (cyclicPQDA : cyclic_quadrilateral P Q D A)
variables (cyclicQPBC : cyclic_quadrilateral Q P B C)
variables (E_on_PQ : E ∈ line P Q)
variables (angle_PAE_eq_QDE : ∠ P A E = ∠ Q D E)
variables (angle_PBE_eq_QCE : ∠ P B E = ∠ Q C E)

-- The theorem to prove: ABCD is a cyclic quadrilateral
theorem cyclic_quadrilateral_ABCD :
  cyclic_quadrilateral A B C D :=
sorry

end cyclic_quadrilateral_ABCD_l723_723649


namespace polynomial_equation_example_l723_723934

theorem polynomial_equation_example (a0 a1 a2 a3 a4 a5 a6 a7 a8 : ℤ)
  (h : x^5 * (x + 3)^3 = a8 * (x + 1)^8 + a7 * (x + 1)^7 + a6 * (x + 1)^6 + a5 * (x + 1)^5 + a4 * (x + 1)^4 + a3 * (x + 1)^3 + a2 * (x + 1)^2 + a1 * (x + 1) + a0) :
  7 * a7 + 5 * a5 + 3 * a3 + a1 = -8 :=
sorry

end polynomial_equation_example_l723_723934


namespace num_prime_factors_of_30_l723_723238

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723238


namespace shift_graph_l723_723733

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem shift_graph (ω : ℝ) (hω : ω > 0) :
  ∀ x : ℝ, ∃ d : ℝ, d = (-4 / 3 : ℝ) →
  (∃ a : ℝ, f ω (x + a) = Real.sin (2 * x + Real.pi / 6) :=
begin
  sorry,
end

end shift_graph_l723_723733


namespace prime_factors_30_fac_eq_10_l723_723144

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l723_723144


namespace problem_statement_l723_723491

theorem problem_statement : 
  ((∑ i in Finset.range 2020, (2021 - i) * (1 / (i+1))) /
  (∑ j in Finset.range 2020, 1 / (j + 2))) = 2021 := 
by
  sorry

end problem_statement_l723_723491


namespace factorial_30_prime_count_l723_723196

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723196


namespace C_work_completion_l723_723423

theorem C_work_completion (A_completion_days B_completion_days AB_completion_days : ℕ)
  (A_cond : A_completion_days = 8)
  (B_cond : B_completion_days = 12)
  (AB_cond : AB_completion_days = 4) :
  ∃ (C_completion_days : ℕ), C_completion_days = 24 := 
by
  sorry

end C_work_completion_l723_723423


namespace factorial_30_prime_count_l723_723186

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723186


namespace part_a_part_b_part_c_part_d_l723_723662

variables {p q x1 x2 : ℝ}

-- Conditions given by Vieta's formulas
axiom roots_of_quadratic (h : x1^2 + p*x1 + q = 0) (h' : x2^2 + p*x2 + q = 0)

-- We need to prove the following:

-- Part (a)
theorem part_a (h : x1^2 + p*x1 + q = 0) (h' : x2^2 + p*x2 + q = 0) :
  (1/x1 + 1/x2 = -p/q) := by
  sorry

-- Part (b)
theorem part_b (h : x1^2 + p*x1 + q = 0) (h' : x2^2 + p*x2 + q = 0) :
  (1/x1^2 + 1/x2^2 = (p^2 - 2*q) / q^2) := by
  sorry

-- Part (c)
theorem part_c (h : x1^2 + p*x1 + q = 0) (h' : x2^2 + p*x2 + q = 0) :
  (x1^3 + x2^3 = -p^3 + 3*p*q) := by
  sorry

-- Part (d)
theorem part_d (h : x1^2 + p*x1 + q = 0) (h' : x2^2 + p*x2 + q = 0) :
  (1 / (x1 + p)^2 + 1 / (x2 + p)^2 = (p^2 - 2*q) / q^2) := by
  sorry

end part_a_part_b_part_c_part_d_l723_723662


namespace graph_symmetric_about_one_l723_723419

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) = f(-x)

theorem graph_symmetric_about_one (f : ℝ → ℝ) (h : even_function (λ x, f(x + 1))) :
  ∀ x : ℝ, f(x) = f(2 - x) :=
sorry

end graph_symmetric_about_one_l723_723419


namespace stationary_points_l723_723516

noncomputable def z (x y : ℝ) : ℝ := Real.exp x * (x - y^3 + 3 * y)

def z_x (x y : ℝ) : ℝ := Real.exp x * (1 + x - y^3 + 3 * y)
def z_y (x y : ℝ) : ℝ := Real.exp x * (-3 * y^2 + 3)

theorem stationary_points :
  (z_x (-3) 1 = 0) ∧ (z_y (-3) 1 = 0) ∧
  (z_x 1 (-1) = 0) ∧ (z_y 1 (-1) = 0) :=
by
  sorry

end stationary_points_l723_723516


namespace prime_factors_of_30_l723_723220

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723220


namespace find_surcharge_l723_723618

-- The property tax in 1996 is increased by 6% over the 1995 tax.
def increased_tax (T_1995 : ℝ) : ℝ := T_1995 * 1.06

-- Petersons' property tax for the year 1995 is $1800.
def T_1995 : ℝ := 1800

-- The Petersons' 1996 tax totals $2108.
def T_1996 : ℝ := 2108

-- Additional surcharge for a special project.
def surcharge (T_1996 : ℝ) (increased_tax : ℝ) : ℝ := T_1996 - increased_tax

theorem find_surcharge : surcharge T_1996 (increased_tax T_1995) = 200 := by
  sorry

end find_surcharge_l723_723618


namespace polynomial_solution_l723_723615

theorem polynomial_solution (x : ℝ) (h : (2 * x - 1) ^ 2 = 9) : x = 2 ∨ x = -1 :=
by
  sorry

end polynomial_solution_l723_723615


namespace prime_factors_of_30_factorial_l723_723173

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l723_723173


namespace find_roots_range_l723_723638

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem find_roots_range 
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hx : -1 < -1/2 ∧ -1/2 < 0 ∧ 0 < 1/2 ∧ 1/2 < 1 ∧ 1 < 3/2 ∧ 3/2 < 2 ∧ 2 < 5/2 ∧ 5/2 < 3)
  (hy : ∀ {x : ℝ}, x = -1 → quadratic_function a b c x = -2 ∧
                   x = -1/2 → quadratic_function a b c x = -1/4 ∧
                   x = 0 → quadratic_function a b c x = 1 ∧
                   x = 1/2 → quadratic_function a b c x = 7/4 ∧
                   x = 1 → quadratic_function a b c x = 2 ∧
                   x = 3/2 → quadratic_function a b c x = 7/4 ∧
                   x = 2 → quadratic_function a b c x = 1 ∧
                   x = 5/2 → quadratic_function a b c x = -1/4 ∧
                   x = 3 → quadratic_function a b c x = -2) :
  ∃ x1 x2 : ℝ, -1/2 < x1 ∧ x1 < 0 ∧ 2 < x2 ∧ x2 < 5/2 ∧ quadratic_function a b c x1 = 0 ∧ quadratic_function a b c x2 = 0 :=
by sorry

end find_roots_range_l723_723638


namespace factorial_prime_factors_l723_723093

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l723_723093


namespace basketball_team_lineup_count_l723_723824

theorem basketball_team_lineup_count :
  ∃ (total_ways : ℕ), total_ways = 4320 ∧ 
  ∃ (players : ℕ) (centers : ℕ) (point_guards : ℕ),
    players = 12 ∧ centers = 3 ∧ point_guards = 2 ∧ 
    total_ways = centers * point_guards * (players - centers - point_guards + 2) * (players - centers - point_guards + 1) * (players - centers - point_guards) := 
begin
  sorry
end

end basketball_team_lineup_count_l723_723824


namespace num_prime_factors_30_factorial_l723_723025

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723025


namespace max_value_of_expression_l723_723915

noncomputable def max_value_expr : ℝ := 
  let expr (x y : ℝ) := 4 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10 in 
  -13

theorem max_value_of_expression : ∀ x y : ℝ, 
  -13 ≤ 4 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10 :=
begin
  sorry  -- Proof required
end

end max_value_of_expression_l723_723915


namespace tangent_line_to_ln_curve_l723_723737

theorem tangent_line_to_ln_curve (b : ℝ) :
  (∃ x > 0, (∀ y : ℝ, y = ln x → y = (1 / 2) * x + b)) →
  b = ln 2 - 1 := by
  sorry

end tangent_line_to_ln_curve_l723_723737


namespace fully_filled_boxes_l723_723307

theorem fully_filled_boxes (total_cards : ℕ) (cards_per_box : ℕ) (h1 : total_cards = 94) (h2 : cards_per_box = 8) : total_cards / cards_per_box = 11 :=
by {
  sorry
}

end fully_filled_boxes_l723_723307


namespace diametrically_opposite_point_exists_l723_723337

variable {Point Circle : Type}
variable [metric_space Point] [has_center Circle Point] [has_radius Circle ℝ] [has_distance Point ℝ]
variable (O₁ A : Point) (a : ℝ)
variable (C₁ : Circle)
variable [is_center O₁ C₁] [is_radius a C₁]
variable [is_on_circle A C₁]

theorem diametrically_opposite_point_exists :
  ∃ (B₂ : Point), is_diametrically_opposite C₁ A B₂ :=
sorry

end diametrically_opposite_point_exists_l723_723337


namespace smallest_tree_height_correct_l723_723753

-- Defining the conditions
def TallestTreeHeight : ℕ := 108
def MiddleTreeHeight (tallest : ℕ) : ℕ := (tallest / 2) - 6
def SmallestTreeHeight (middle : ℕ) : ℕ := middle / 4

-- Proof statement
theorem smallest_tree_height_correct :
  SmallestTreeHeight (MiddleTreeHeight TallestTreeHeight) = 12 :=
by
  -- Here we would put the proof, but we are skipping it with sorry.
  sorry

end smallest_tree_height_correct_l723_723753


namespace percent_water_evaporated_l723_723825

theorem percent_water_evaporated (W : ℝ) (E : ℝ) (T : ℝ) (hW : W = 10) (hE : E = 0.16) (hT : T = 75) : 
  ((min (E * T) W) / W) * 100 = 100 :=
by
  sorry

end percent_water_evaporated_l723_723825


namespace min_elements_in_union_l723_723689

theorem min_elements_in_union (A B C : Finset α) :
  A.card = 30 → B.card = 25 → C.card = 10 →
  (∀ x, x ∈ A ∪ B ∪ C → x ∈ A ∨ x ∈ B ∨ x ∈ C) →
  (|A ∪ B ∪ C| = 45) :=
begin
  sorry
end

end min_elements_in_union_l723_723689


namespace find_f_of_2_l723_723530

variable (f : ℝ → ℝ)

def functional_equation_condition :=
  ∀ x : ℝ, f (f (f x)) + 3 * f (f x) + 9 * f x + 27 * x = 0

theorem find_f_of_2
  (h : functional_equation_condition f) :
  f (f (f (f 2))) = 162 :=
sorry

end find_f_of_2_l723_723530


namespace boxes_produced_by_machine_A_in_10_minutes_l723_723792

-- Define the variables and constants involved
variables {A : ℕ} -- number of boxes machine A produces in 10 minutes

-- Define the condition that machine B produces 4*A boxes in 10 minutes
def boxes_produced_by_machine_B_in_10_minutes := 4 * A

-- Define the combined production working together for 20 minutes
def combined_production_in_20_minutes := 10 * A

-- Statement to prove that machine A produces A boxes in 10 minutes
theorem boxes_produced_by_machine_A_in_10_minutes :
  ∀ (boxes_produced_by_machine_B_in_10_minutes : ℕ) (combined_production_in_20_minutes : ℕ),
    boxes_produced_by_machine_B_in_10_minutes = 4 * A →
    combined_production_in_20_minutes = 10 * A →
    A = A :=
by
  intros _ _ hB hC
  sorry

end boxes_produced_by_machine_A_in_10_minutes_l723_723792


namespace triangle_parallel_lines_l723_723665

theorem triangle_parallel_lines
  (A B C I X Y : Type)
  [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty I] [Nonempty X] [Nonempty Y]
  {AB AC : ℝ} (h : AB > AC)
  (is_incenter : ∀ {Δ : Type}, Δ = {x // x = A ∧ x = B ∧ x = C} → (I = incenter_of Δ))
  (has_angle_bisectors : ∀ (Δ : Type), Δ = {x // x = A ∧ x = B ∧ x = C} → 
    (angle_bisectors_meet : ∀ {α β γ : Type}, α = angle_bisector B I -> β = angle_bisector C I -> γ = I -> meets(α, β, γ)))
  (X_Y_on_circumcircle : ∀ (Δ : Type), Δ = {x // x = B ∧ x = I ∧ x = C} → 
    (on_circumcircle : ∀ {ψ χ : Type}, ψ = circumcircle Δ -> intersects(χ, ψ, AB ∩ χ) ∧ intersects(χ, ψ, AC ∩ χ) → (χ = X ∨ χ = Y))) :
  parallel_lines (segment C X) (segment B Y) :=
sorry

end triangle_parallel_lines_l723_723665


namespace prime_factors_of_30_factorial_l723_723102

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l723_723102


namespace sin_identity_l723_723785

theorem sin_identity : (sin (160 * degree) = sin (20 * degree)) :=
by
  have h1 : cos (20 * degree) = sin (70 * degree) := by sorry
  have h2 : sin (-20 * degree) = -sin (20 * degree) := by sorry
  have h3 : sin (160 * degree) = sin ((180 - 20) * degree) := by sorry
  show sin (160 * degree) = sin (20 * degree) from h3.trans (by refl)

end sin_identity_l723_723785


namespace equilateral_triangle_dot_product_sum_l723_723941

theorem equilateral_triangle_dot_product_sum (n : ℕ) (h : n > 0) :
  let a := (1 : real )
  let b := (1 : real )
  let c := (1 : real )
  let p (k : ℕ) := (k : real )/n
  let S_n := ∑ k in finset.range n, (c * c + (2 * k + 1) * (c * p k) + k * (k - 1) * (p k) * (p k) )
  S_n = (5 * n ^ 2 - 2) / (6 * n) := sorry

end equilateral_triangle_dot_product_sum_l723_723941


namespace prime_factors_of_30_l723_723218

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723218


namespace num_prime_factors_30_factorial_l723_723243

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l723_723243


namespace gcd_228_1995_l723_723514

theorem gcd_228_1995 : Int.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l723_723514


namespace ab_product_equals_one_l723_723947

variable {a b θ : ℝ}
variable (sin cos : ℝ → ℝ) [trig : is_trig_function sin cos]

noncomputable def trig_sin_cos : Prop :=
  a * sin θ + cos θ = 1 ∧ b * sin θ - cos θ = 1 → a * b = 1

theorem ab_product_equals_one (h : trig_sin_cos sin cos) : a * b = 1 :=
by sorry

end ab_product_equals_one_l723_723947


namespace smallest_number_to_end_in_four_zeros_l723_723906

theorem smallest_number_to_end_in_four_zeros (x : ℕ) :
  let n1 := 225
  let n2 := 525
  let factor_needed := 16
  (∃ y : ℕ, y = n1 * n2 * x) ∧ (10^4 ∣ n1 * n2 * x) ↔ x = factor_needed :=
by
  sorry

end smallest_number_to_end_in_four_zeros_l723_723906


namespace true_proposition_l723_723470

-- Define each proposition as a logical statement
def propA (x : ℚ) : Prop := (|x| > 0)
def propB (l m n : Type) [linear_order l] [linear_order m] [linear_order n] : Prop := 
  ∀ (L₁ L₂ L₃ : set l), (L₁ ∥ L₂) ∧ (L₂ ∥ L₃) → (L₁ ∥ L₃)
def propC (α β : Type) [add_group α] [add_group β] : Prop := 
  ∀ (∠A ∠B : α), (∠A + ∠B = 180) → (∠A = ∠B)
def propD (l : Type) [metric_space l] : Prop := 
  ∀ (L : set l) (P : l), ∃! L', (P ∈ L' ∧ L ∥ L')

-- Proof problem
theorem true_proposition : propB :=
sorry

end true_proposition_l723_723470


namespace factorial_prime_factors_l723_723164

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l723_723164


namespace boxes_needed_l723_723716

theorem boxes_needed (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) (h1 : total_muffins = 95) (h2 : muffins_per_box = 5) (h3 : available_boxes = 10) : 
  total_muffins - (available_boxes * muffins_per_box) / muffins_per_box = 9 :=
by
  sorry

end boxes_needed_l723_723716


namespace second_group_books_magazines_l723_723726

theorem second_group_books_magazines: 
  ∀ (B M x y : ℕ), 
    (2 * B + 2 * M = 26) → 
    (M = 7) → 
    (6 * x + 7 * y = 27) → 
    x = 1 ∧ y = 3 :=
by
    intros B M x y h1 h2 h3
    have B_def := calc
      B = (26 - 2 * 7) / 2 : by sorry
      ... = 6 : by sorry
    show x = 1 ∧ y = 3, from sorry

end second_group_books_magazines_l723_723726


namespace derivative_at_pi_over_4_l723_723578

-- Define the function f and its derivative
def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x

-- State the theorem we want to prove
theorem derivative_at_pi_over_4 : f' (π / 4) = 0 :=
by 
  -- This is the placeholder for the proof
  sorry

end derivative_at_pi_over_4_l723_723578


namespace num_prime_factors_30_fac_l723_723075

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l723_723075


namespace green_fraction_after_tripling_l723_723619

theorem green_fraction_after_tripling
  (x : ℕ)
  (hx : x > 0)
  (blue_fraction : ℚ := 4/7)
  (green_fraction : ℚ := 1 - blue_fraction) :
  let initial_green_marbles := green_fraction * x,
      initial_blue_marbles := blue_fraction * x,
      new_green_marbles := 3 * initial_green_marbles,
      new_total_marbles := initial_blue_marbles + new_green_marbles in
  new_green_marbles / new_total_marbles = 9 / 13 :=
by
  -- Proof will not be provided here
  sorry

end green_fraction_after_tripling_l723_723619


namespace locus_of_intersection_of_tangents_l723_723300

theorem locus_of_intersection_of_tangents (O A : Point) (R : ℝ) (M P : Point)
  (h_circle : ∀ Q₁ Q₂ : Point, (dist O Q₁ = R ∧ dist O Q₂ = R) → 
              (P = midpoint Q₁ Q₂) → 
              (is_tangent O Q₁ ∧ is_tangent O Q₂) → 
              ∃ M, is_intersection_of_tangents Q₁ Q₂ M) :
  (dist A O < R) →
  ∃ line : Line, ∀ M, is_intersection_of_tangents ∧ M ∈ line ∧ is_perpendicular line (line_from_points O A) :=
sorry

end locus_of_intersection_of_tangents_l723_723300


namespace prime_factors_of_30_l723_723210

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723210


namespace boat_distance_against_stream_l723_723630

-- Definitions from Step a)
def speed_boat_still_water : ℝ := 15  -- speed of the boat in still water in km/hr
def distance_downstream : ℝ := 21  -- distance traveled downstream in one hour in km
def time_hours : ℝ := 1  -- time in hours

-- Translation of the described problem proof
theorem boat_distance_against_stream :
  ∃ (v_s : ℝ), (speed_boat_still_water + v_s = distance_downstream / time_hours) → 
               (15 - v_s = 9) :=
by
  sorry

end boat_distance_against_stream_l723_723630


namespace exact_diff_eq_general_solution_l723_723301

open Real

-- Define functions P and Q
def P (x y : ℝ) : ℝ := y^3 - 2 * x * y
def Q (x y : ℝ) : ℝ := 3 * x * y^2 - x^2

-- Prove the given differential equation is exact and find its general solution
theorem exact_diff_eq_general_solution (x y : ℝ) (C : ℝ) :
    ∃ u : ℝ → ℝ → ℝ, 
      (∀ x y, ∂ (u x y) / ∂ x = P x y) ∧ 
      (∀ x y, ∂ (u x y) / ∂ y = Q x y) ∧ 
      u x y = y^3 * x - y * x^2 + C :=
by
  sorry

end exact_diff_eq_general_solution_l723_723301


namespace find_M1M2_l723_723807

-- Definitions for the problem conditions
variables (O_1 O_2 : Point)
variables (M_1 M_2 : Point)
variables (r1 r2 : ℝ)
variables (O_1O_2 M_1M_2 : ℝ)
variables (l : Line)

-- Conditions
axiom radius_O1 : r1 = 12
axiom radius_O2 : r2 = 7
axiom circles_same_side : ∀ P, P ∈ l → ¬((O_1 ∈ circle O_1 r1) ∧ (O_2 ∈ circle O_2 r2) ∧ (P ∈ l))
axiom line_touch_M1 : M_1 ∈ l
axiom line_touch_M2 : M_2 ∈ l
axiom ratio_length : M_1M_2 / O_1O_2 = (2 * sqrt 5) / 5

-- The goal to prove
theorem find_M1M2 : M_1M_2 = 10 :=
by sorry

end find_M1M2_l723_723807


namespace junior_score_is_95_l723_723275

theorem junior_score_is_95:
  ∀ (n j s : ℕ) (x avg_total avg_seniors : ℕ),
    n = 20 →
    j = n * 15 / 100 →
    s = n * 85 / 100 →
    avg_total = 78 →
    avg_seniors = 75 →
    (j * x + s * avg_seniors) / n = avg_total →
    x = 95 :=
by
  sorry

end junior_score_is_95_l723_723275


namespace limit_sum_odd_terms_l723_723614

-- Given conditions
def geom_seq_sum (n : ℕ) : ℝ := (1 / 2) ^ n - 1

-- Definition of the sequence terms
noncomputable def a (n : ℕ) : ℝ := 
  if n = 0 then geom_seq_sum 1
  else geom_seq_sum (n + 1) - geom_seq_sum n

-- Summing terms at odd indices
noncomputable def sum_odd_terms (n : ℕ) : ℝ :=
  finset.sum (finset.range n) (λ i, a (2 * i + 1))

-- Statement of the proof to be shown
theorem limit_sum_odd_terms :
  tendsto (λ n, sum_odd_terms n) at_top (𝓝 (-2 / 3)) :=
sorry

end limit_sum_odd_terms_l723_723614


namespace num_prime_factors_of_30_l723_723223

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l723_723223


namespace kyle_vs_parker_l723_723874

-- Define the distances thrown by Parker, Grant, and Kyle.
def parker_distance : ℕ := 16
def grant_distance : ℕ := (125 * parker_distance) / 100
def kyle_distance : ℕ := 2 * grant_distance

-- Prove that Kyle threw the ball 24 yards farther than Parker.
theorem kyle_vs_parker : kyle_distance - parker_distance = 24 := 
by
  -- Sorry for proof
  sorry

end kyle_vs_parker_l723_723874


namespace derivative_at_pi_div_4_l723_723585

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem derivative_at_pi_div_4 :
  (deriv f) (Real.pi / 4) = 0 :=
sorry

end derivative_at_pi_div_4_l723_723585


namespace prime_factors_of_30_l723_723207

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723207


namespace base_number_of_exponentiation_l723_723261

theorem base_number_of_exponentiation (n : ℕ) (some_number : ℕ) (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = some_number^22) (h2 : n = 21) : some_number = 4 :=
  sorry

end base_number_of_exponentiation_l723_723261


namespace factorial_30_prime_count_l723_723198

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l723_723198


namespace real_solution_count_eq_zero_l723_723527

theorem real_solution_count_eq_zero : 
  ∀ c : ℝ, abs (1 + c - 3 * I) = 2 → false :=
begin
  intros c hc,
  -- Now we proceed from the conditions.
  sorry
end

end real_solution_count_eq_zero_l723_723527


namespace number_of_distinct_prime_factors_30_fact_l723_723002

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l723_723002


namespace beef_weight_before_processing_l723_723462

theorem beef_weight_before_processing (OW : ℝ) (after_processing_weight : ℝ) (processing_loss : ℝ) 
(h1 : processing_loss = 0.40) 
(h2 : after_processing_weight = 240) : 
0.60 * OW = after_processing_weight → OW = 400 :=
by
  intro h
  rw [h2, h1] at h
  sorry

end beef_weight_before_processing_l723_723462


namespace residue_system_mod_3n_l723_723318

theorem residue_system_mod_3n (n : ℕ) (h_odd : n % 2 = 1) :
  ∃ (a b : ℕ → ℕ) (k : ℕ), 
  (∀ i, a i = 3 * i - 2) ∧ 
  (∀ i, b i = 3 * i - 3) ∧
  (∀ i (k : ℕ), 0 < k ∧ k < n → 
    (a i + a (i + 1)) % (3 * n) ≠ (a i + b i) % (3 * n) ∧ 
    (a i + b i) % (3 * n) ≠ (b i + b (i + k)) % (3 * n) ∧ 
    (a i + a (i + 1)) % (3 * n) ≠ (b i + b (i + k)) % (3 * n)) :=
sorry

end residue_system_mod_3n_l723_723318


namespace derivative_at_pi_div_4_l723_723586

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem derivative_at_pi_div_4 :
  (deriv f) (Real.pi / 4) = 0 :=
sorry

end derivative_at_pi_div_4_l723_723586


namespace num_prime_factors_30_factorial_l723_723030

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l723_723030


namespace number_of_ways_to_assign_students_l723_723258

theorem number_of_ways_to_assign_students 
  (students : ℕ) (communities : ℕ) 
  (h_students : students = 5) (h_communities : communities = 3) 
  (condition : communities > 2) : 
  ∃ ways : ℕ, ways = 150 :=
by {
  have h_students := 5,
  have h_communities := 3,
  have condition := 3 > 2,
  sorry
}

end number_of_ways_to_assign_students_l723_723258


namespace pencils_per_box_l723_723871

-- Variables and Definitions based on the problem conditions
def num_boxes : ℕ := 10
def pencils_kept : ℕ := 10
def friends : ℕ := 5
def pencils_per_friend : ℕ := 8

-- Theorem to prove the solution
theorem pencils_per_box (pencils_total : ℕ)
  (h1 : pencils_total = pencils_kept + (friends * pencils_per_friend))
  (h2 : pencils_total = num_boxes * (pencils_total / num_boxes)) :
  (pencils_total / num_boxes) = 5 :=
sorry

end pencils_per_box_l723_723871


namespace shirts_not_all_on_sale_implications_l723_723268

variable (Shirts : Type) (store_contains : Shirts → Prop) (on_sale : Shirts → Prop)

theorem shirts_not_all_on_sale_implications :
  ¬ ∀ s, store_contains s → on_sale s → 
  (∃ s, store_contains s ∧ ¬ on_sale s) ∧ (∃ s, store_contains s ∧ ¬ on_sale s) :=
by
  sorry

end shirts_not_all_on_sale_implications_l723_723268


namespace triangle_covering_by_congruent_l723_723730

theorem triangle_covering_by_congruent (ABC : Triangle) :
  (∃ (T1 T2 : Triangle), congruent T1 T2 ∧ covers ABC T1 ∧ covers ABC T2 ∧ ¬(covers_only ABC T1 T2)) →
  (is_scalene ABC ∨ is_isosceles_distinct ABC) :=
by
  sorry

end triangle_covering_by_congruent_l723_723730


namespace ahmed_goats_is_13_l723_723865

def adam_goats : ℕ := 7

def andrew_goats : ℕ := 2 * adam_goats + 5

def ahmed_goats : ℕ := andrew_goats - 6

theorem ahmed_goats_is_13 : ahmed_goats = 13 :=
by
  sorry

end ahmed_goats_is_13_l723_723865


namespace final_score_is_89_l723_723438

def final_score (s_e s_l s_b : ℝ) (p_e p_l p_b : ℝ) : ℝ :=
  s_e * p_e + s_l * p_l + s_b * p_b

theorem final_score_is_89 :
  final_score 95 92 80 0.4 0.25 0.35 = 89 := 
by
  sorry

end final_score_is_89_l723_723438


namespace sum_binom_mod_1000_l723_723320

open Nat

theorem sum_binom_mod_1000 :
  let T := ∑ n in (Finset.range 804), (-1 : ℤ)^n * (Nat.choose 4015 (5 * n))
  T % 1000 = 6 :=
by
  let T := ∑ n in (Finset.range 804), (-1 : ℤ)^n * (Nat.choose 4015 (5 * n))
  show T % 1000 = 6
  sorry

end sum_binom_mod_1000_l723_723320


namespace prime_factors_of_30_l723_723212

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l723_723212


namespace number_of_prime_factors_of_30_factorial_l723_723130

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l723_723130


namespace problem_1_problem_2_l723_723931

theorem problem_1 (h : Real.tan (α / 2) = 2) : Real.tan (α + Real.arctan 1) = -1/7 :=
by
  sorry

theorem problem_2 (h : Real.tan (α / 2) = 2) : (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 :=
by
  sorry

end problem_1_problem_2_l723_723931


namespace fraction_value_l723_723492

theorem fraction_value :
  (∑ i in Finset.range (2020+1), (2020+1 - i) / i) / 
  (∑ i in Finset.range (2021+1), 1 / (i+2)) = 2021 :=
by
  sorry

end fraction_value_l723_723492
