import Complex
import Mathlib
import Mathlib.Algebra.Factorial.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Polynomial.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.ModEq
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Factorization.Basic
import Mathlib.Data.Nat.LCM
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.Int.Basic
import Mathlib.Integral
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace segments_length_bound_l391_391631

theorem segments_length_bound {n : ℕ} (h : n > 0) :
  ∀ (points : Fin n → ℝ × ℝ), 
  (∀ i, (points i).fst ^ 2 + (points i).snd ^ 2 = 1) → 
  (∑ i j in Finset.univ.product Finset.univ, 
   if (dist (points i) (points j) > sqrt 2) then 1 else 0) / 2 ≤ n * (n - 1) / 6 :=
by sorry

end segments_length_bound_l391_391631


namespace correct_transformation_l391_391914

-- Definitions of the equations and their transformations
def optionA := (forall (x : ℝ), ((x / 5) + 1 = x / 2) -> (2 * x + 10 = 5 * x))
def optionB := (forall (x : ℝ), (5 - 2 * (x - 1) = x + 3) -> (5 - 2 * x + 2 = x + 3))
def optionC := (forall (x : ℝ), (5 * x + 3 = 8) -> (5 * x = 8 - 3))
def optionD := (forall (x : ℝ), (3 * x = -7) -> (x = -7 / 3))

-- Theorem stating that option D is the correct transformation
theorem correct_transformation : optionD := 
by 
  sorry

end correct_transformation_l391_391914


namespace petya_friends_l391_391043

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l391_391043


namespace parity_uniformity_l391_391554

-- Define the set A_P
variable {A_P : Set ℤ}

-- Conditions:
-- 1. A_P is non-empty
noncomputable def non_empty (H : A_P ≠ ∅) := H

-- 2. c is the maximum element in A_P
variable {c : ℤ}
variable (H_max : ∀ a ∈ A_P, a ≤ c)

-- 3. Consideration of critical points around c
variable {f : ℤ → ℤ}
variable (H_critical : ∀ x ∈ A_P, f x = 0)

-- 4. Parity of the smallest and largest elements
def parity (n : ℤ) : Prop := n % 2 = 0 ∨ n % 2 = 1

-- Proof statement
theorem parity_uniformity (H_non_empty : non_empty A_P)
  (H_max_element : ∀ a ∈ A_P, a ≤ c)
  (H_critical_points : ∀ x ∈ A_P, f x = 0) :
  (∃ x ∈ A_P, ∀ y ∈ A_P, x ≤ y) → (parity (x : ℤ) = parity ((y : ℤ) : ℤ)) → (least x ∈ A_P, greatest y ∈ A_P, parity x = parity y) :=
by
  sorry

end parity_uniformity_l391_391554


namespace petya_friends_l391_391067

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l391_391067


namespace soccer_field_solution_l391_391540

noncomputable def soccer_field_problem : Prop :=
  ∃ (a b c d : ℝ), 
    (abs (a - b) = 1 ∨ abs (a - b) = 2 ∨ abs (a - b) = 3 ∨ abs (a - b) = 4 ∨ abs (a - b) = 5 ∨ abs (a - b) = 6) ∧
    (abs (a - c) = 1 ∨ abs (a - c) = 2 ∨ abs (a - c) = 3 ∨ abs (a - c) = 4 ∨ abs (a - c) = 5 ∨ abs (a - c) = 6) ∧
    (abs (a - d) = 1 ∨ abs (a - d) = 2 ∨ abs (a - d) = 3 ∨ abs (a - d) = 4 ∨ abs (a - d) = 5 ∨ abs (a - d) = 6) ∧
    (abs (b - c) = 1 ∨ abs (b - c) = 2 ∨ abs (b - c) = 3 ∨ abs (b - c) = 4 ∨ abs (b - c) = 5 ∨ abs (b - c) = 6) ∧
    (abs (b - d) = 1 ∨ abs (b - d) = 2 ∨ abs (b - d) = 3 ∨ abs (b - d) = 4 ∨ abs (b - d) = 5 ∨ abs (b - d) = 6) ∧
    (abs (c - d) = 1 ∨ abs (c - d) = 2 ∨ abs (c - d) = 3 ∨ abs (c - d) = 4 ∨ abs (c - d) = 5 ∨ abs (c - d) = 6)

theorem soccer_field_solution : soccer_field_problem :=
  sorry

end soccer_field_solution_l391_391540


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391319
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391319


namespace greatest_prime_factor_15f_plus_17f_l391_391193

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391193


namespace Vince_expenses_l391_391162

-- Define the conditions
def price_per_customer : ℕ := 18
def customers_per_month : ℕ := 80
def savings : ℕ := 872
def recreation_percentage : ℝ := 0.20

-- Define total earnings
def total_earnings : ℕ := price_per_customer * customers_per_month

-- Define allocation for recreation and relaxation
def recreation_allocation : ℕ := (recreation_percentage * total_earnings).toNat

-- Define the Lean statement for the proof problem
theorem Vince_expenses (monthly_expenses: ℕ) :
  (total_earnings = savings + recreation_allocation + monthly_expenses) → monthly_expenses = 280 :=
by
  sorry

end Vince_expenses_l391_391162


namespace not_harmonious_1_2_3_abscissas_harmonious_distance_OP_range_l391_391101

-- Problem 1: Are {1, 2, 3} a harmonious triplet?
def is_harmonious_triplet (x y z : ℝ) : Prop :=
  (1 / x = 1 / y + 1 / z) ∨ (1 / y = 1 / x + 1 / z) ∨ (1 / z = 1 / x + 1 / y)

theorem not_harmonious_1_2_3 : ¬ is_harmonious_triplet 1 2 3 :=
by sorry

-- Problem 2(i): Prove that x1, x2, x3 are harmonious triplets given the intersection conditions
variable {a b c : ℝ}

def f (x : ℝ) := a * x^2 + 3 * b * x + 3 * c

theorem abscissas_harmonious (h1 : ∀ x, 2 * b * x + 2 * c = 0 → x = -c / b)
  (h2 : ∀ x, f x = 2 * b * x + 2 * c → (∃ y, x = 1 ∨ x = -c / b))
  : is_harmonious_triplet (-c / b) 1 (-c / b) :=
by sorry

-- Problem 2(ii): Find the range of distance OP under given constraints
def P := (c / a, b / a)

theorem distance_OP_range (h1 : a > 2 * b)
  (h2 : 2 * b > 3 * c)
  (h3 : P.1 = c / a)
  (h4 : P.2 = b / a) :
  ∀ OP, (sqrt(2) / 2 ≤ OP ∧ OP < sqrt(10) / 2) ∧ OP ≠ 1 :=
by sorry

end not_harmonious_1_2_3_abscissas_harmonious_distance_OP_range_l391_391101


namespace megan_popsicles_l391_391026

theorem megan_popsicles (hours : ℕ) (duration_per_popsicle break_per_hour : ℕ) (total_hours time_per_hour : ℕ) (total_duration : ℕ) :
  hours = 5 →
  duration_per_popsicle = 20 →
  break_per_hour = 10 →
  time_per_hour = 60 →
  total_duration = total_hours * time_per_hour →
  total_hours = hours →
  let effective_minutes := (total_duration - (hours * break_per_hour)) in
  let popsicles_eaten := effective_minutes / duration_per_popsicle in
  popsicles_eaten = 12 :=
begin
  intros h_hours h_duration_per_popsicle h_break_per_hour h_time_per_hour h_total_duration h_total_hours,
  let effective_minutes := (total_duration - (hours * break_per_hour)),
  let popsicles_eaten := effective_minutes / duration_per_popsicle,
  calc
    popsicles_eaten = (hours * time_per_hour - (hours * break_per_hour)) / duration_per_popsicle : by {
      simp [effective_minutes, h_total_duration, h_total_hours],
      sorry
    }
                ... = 12 : by {
      simp [h_hours, h_duration_per_popsicle, h_break_per_hour, h_time_per_hour, h_total_duration, h_total_hours],
      sorry
    }
end

end megan_popsicles_l391_391026


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391294

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391294


namespace petya_friends_l391_391066

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l391_391066


namespace greatest_prime_factor_of_15_l391_391448

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391448


namespace smallest_sum_l391_391864

theorem smallest_sum (p q r s : ℕ) (h1 : p > 0) (h2 : q > 0) (h3 : r > 0) (h4 : s > 0) (h5 : p * q * r * s = 12!) :
  p + q + r + s = 777 :=
sorry

end smallest_sum_l391_391864


namespace number_of_quadratic_functions_l391_391467

def is_quadratic (f : ℝ → ℝ) : Prop := ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def f1 (x : ℝ) : ℝ := 3 * (x - 1)^2 + 1
def f2 (x : ℝ) : ℝ := x + 1 / x
def f3 (x : ℝ) : ℝ := 8 * x^2 + 1
def f4 (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2

theorem number_of_quadratic_functions : 
  (if is_quadratic f1 then 1 else 0) +
  (if is_quadratic f2 then 1 else 0) +
  (if is_quadratic f3 then 1 else 0) +
  (if is_quadratic f4 then 1 else 0) = 2 := 
by
  sorry

end number_of_quadratic_functions_l391_391467


namespace ironman_age_l391_391154

def age_Thor := 1456
def ratio_Thor_Cap := 13
def ratio_Cap_Peter := 7
def ratio_Peter_Strange := 1/4
def diff_Ironman_Peter := 32

theorem ironman_age :
  let age_Cap := age_Thor / ratio_Thor_Cap in
  let age_Peter := age_Cap / ratio_Cap_Peter in
  let age_Strange := age_Peter / ratio_Peter_Strange in
  let age_Ironman := age_Peter + diff_Ironman_Peter in
  age_Ironman = 48 := sorry

end ironman_age_l391_391154


namespace greatest_prime_factor_15_fact_17_fact_l391_391235

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391235


namespace total_age_of_group_l391_391891

theorem total_age_of_group (n_boys n_girls : ℕ) (youngest_boy_age : ℕ)
  (h_n_boys : n_boys = 6) (h_n_girls : n_girls = 3) (h_youngest_boy_age : youngest_boy_age = 5)
  (h_girls_age : ∀ i, i < n_girls → i < n_boys → (∃ age, age = youngest_boy_age + i + 2)) :
  let boys_ages := (List.range n_boys).map (λ i, youngest_boy_age + i),
      girls_ages := (List.range n_girls).map (λ i, youngest_boy_age + i + 2),
      total_age := (boys_ages.sum + girls_ages.sum)
  in total_age = 69 := 
by
  intros
  -- This is where the proof would go
  sorry

end total_age_of_group_l391_391891


namespace even_integers_count_l391_391683

def is_even (n : ℕ) : Prop := n % 2 = 0
def has_four_different_digits (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, n = 1000 * a + 100 * b + 10 * c + d ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem even_integers_count : 
  let nums := {n | 5000 ≤ n ∧ n < 8000 ∧ is_even n ∧ has_four_different_digits n} in
  nums.card = 784 :=
by 
  sorry

end even_integers_count_l391_391683


namespace abs_eq_iff_mul_nonpos_l391_391858

theorem abs_eq_iff_mul_nonpos (a b : ℝ) : |a - b| = |a| + |b| ↔ a * b ≤ 0 :=
sorry

end abs_eq_iff_mul_nonpos_l391_391858


namespace smallest_largest_same_parity_l391_391568

-- Define the set A_P
def A_P : Set ℕ := sorry

-- Define the maximum element c in A_P
def c := max A_P

-- Assuming shifting doesn't change fundamental counts
lemma shifting_preserves_counts (A : Set ℕ) (c : ℕ) : 
  (∀ x ∈ A, x ≠ c → x < c → exists y ∈ A, y < x) ∧
  (∃ p ∈ A, ∃ q ∈ A, p ≠ q ∧ p < q) :=
  sorry

-- Define parity
def parity (n : ℕ) := n % 2

-- Theorem statement
theorem smallest_largest_same_parity (A : Set ℕ) (a_max a_min : ℕ) 
  (h_max : a_max = max A) (h_min : a_min = min A)
  (h_shift : ∀ x ∈ A, shifting_preserves_counts A x) :
  parity a_max = parity a_min :=
sorry

end smallest_largest_same_parity_l391_391568


namespace greatest_prime_factor_15f_plus_17f_l391_391190

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391190


namespace integral_x_minus_reciprocal_l391_391547

theorem integral_x_minus_reciprocal :
  ∫ x in (1:ℝ)..(Real.exp 1), (x - (1 / x)) = (Real.exp 2 - 3) / 2 :=
by
  sorry

end integral_x_minus_reciprocal_l391_391547


namespace no_diff_parity_min_max_l391_391559

def A_P : Set ℤ := 
  {x | PositionProperty x}

variable (x y : ℤ) (hx : x ∈ A_P) (hy : y ∈ A_P)

theorem no_diff_parity_min_max :
  (∀ x ∈ A_P, PositionProperty x) →
  (∃ c, ∀ x ∈ A_P, x ≤ c) →
  (∀ c, (c = max_element A_P) → CriticalPointsProperty c) →
  ((min_element A_P) % 2 = (max_element A_P) % 2) :=
by
  sorry

end no_diff_parity_min_max_l391_391559


namespace non_acute_angles_among_l391_391835

-- Define the concept of a convex n-gon and point inside it
variables {n : ℕ} (A : Fin n → ℝ × ℝ) (O : ℝ × ℝ)

-- Define convexity and the point inside the polygon
def convex_polygon (A : Fin n → ℝ × ℝ) : Prop :=
  ∀ i j k : Fin n, 
    let (xi, yi) := A i in 
    let (xj, yj) := A j in 
    let (xk, yk) := A k in 
    (xi - xk) * (yj - yk) < (yi - yk) * (xj - xk)

def point_inside_polygon (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop :=
  ∃ i j k : Fin n,
    let (xo, yo) := O in
    let (xi, yi) := A i in
    let (xj, yj) := A j in
    let (xk, yk) := A k in
    (xi - xo) * (yj - yo) + (yj - yo) * (yj - yk) > 0 ∧
    (xj - xo) * (yk - yo) + (yk - yo) * (yk - xi) > 0 ∧
    (xk - xo) * (yi - yo) + (yi - yo) * (yi - xj) > 0

-- Define the angle ∠A_i O A_j to be non-acute
def non_acute_angle (O : ℝ × ℝ) (A_i A_j : ℝ × ℝ) : Prop :=
  let (ox, oy) := O in
  let (xi, yi) := A_i in
  let (xj, yj) := A_j in
  ((xi - ox) * (xj - ox) + (yi - oy) * (yj - oy)) ≤ 0

theorem non_acute_angles_among (A : Fin n → ℝ × ℝ) (O : ℝ × ℝ) (h_convex : convex_polygon A) (h_inside : point_inside_polygon O A) (h_n : 3 ≤ n) : 
  ∃ S : Finset (Fin n × Fin n), 
  (∀ i j, (i, j) ∈ S → non_acute_angle O (A i) (A j)) ∧
  S.card ≥ n - 1 :=
by sorry

end non_acute_angles_among_l391_391835


namespace more_action_figures_than_books_l391_391729

-- Definitions of initial conditions
def books : ℕ := 3
def initial_action_figures : ℕ := 4
def added_action_figures : ℕ := 2

-- Definition of final number of action figures
def final_action_figures : ℕ := initial_action_figures + added_action_figures

-- Proposition to be proved
theorem more_action_figures_than_books : final_action_figures - books = 3 := by
  -- We leave the proof empty
  sorry

end more_action_figures_than_books_l391_391729


namespace exponential_sequence_probability_l391_391018

open ProbabilityTheory MeasureTheory

/-- Given i.i.d. exponential random variables and specific functions involving \ln n and \alpha, -/
theorem exponential_sequence_probability
  (ξ : ℕ → ℝ)
  (h_indep : ∀ n m, n ≠ m → Independence (ξ n) (ξ m))
  (h_iid : ∀ n, Distribution (ξ(n + 1)) = Distribution (ξ(1)))
  (h_exp : ∀ x, x ≥ 0 → P(ξ 1 > x) = Real.exp (-x)) :
  (∀ α, (P(∀ᶠ n in at_top, ξ n > α * Real.log n) = 
  (if α ≤ 1 then 1 else 0))) ∧
  (∀ α, (P(∀ᶠ n in at_top, ξ n > Real.log n + α * Real.log (Real.log n)) = 
  (if α ≤ 1 then 1 else 0))) ∧ 
  (∀ α k, (P(∀ᶠ n in at_top,
    ξ n > Real.log n + 
                   (@List.foldl ℕ ℝ (λ acc i, Real.log acc) (Real.log n) (List.range k)) + 
                           α * Real.log ((List.foldl ℕ ℝ (λ acc i, Real.log acc) (Real.log n) (List.range (k + 1)))))
    = (if α ≤ 1 then 1 else 0))) :=
sorry

end exponential_sequence_probability_l391_391018


namespace solve_equation_l391_391852

theorem solve_equation :
  ∀ x : ℝ, (-x^2 = (2*x + 4) / (x + 2)) ↔ (x = -2 ∨ x = -1) :=
by
  intro x
  -- the proof steps would go here
  sorry

end solve_equation_l391_391852


namespace greatest_prime_factor_15_17_factorial_l391_391215

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391215


namespace conditions_neither_necessary_nor_sufficient_l391_391935

theorem conditions_neither_necessary_nor_sufficient :
  (¬(0 < x ∧ x < 2) ↔ (¬(-1 / 2 < x ∨ x < 1)) ∨ (¬(-1 / 2 < x ∧ x < 1))) :=
by sorry

end conditions_neither_necessary_nor_sufficient_l391_391935


namespace greatest_prime_factor_of_sum_l391_391390

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391390


namespace collinear_implication_l391_391640

variables {k : ℝ} (e1 e2 : ℝ → ℝ)

-- Assume e1 and e2 are non-collinear vectors
-- (non-collinearity not directly checked since it doesn't impact the specific linear combination solution)
def vector_a : ℝ → ℝ := λ x, 2 * e1 x - e2 x
def vector_b : ℝ → ℝ := λ x, k * e1 x + e2 x

theorem collinear_implication (h : ∃ t : ℝ, vector_b e1 e2 = t • (vector_a e1 e2)) : k = -2 :=
sorry

end collinear_implication_l391_391640


namespace chests_contents_l391_391790

-- Define the coin types
inductive CoinType
| Gold
| Silver
| Copper

open CoinType

-- Define the chests
def Chest : Type

-- Define the inscriptions on the chests (all of which are known to be incorrect)
def chest_1_inscription : Prop := ∀ c : Chest, c = Gold
def chest_2_inscription : Prop := ∀ c : Chest, c = Silver
def chest_3_inscription : Prop := ∀ c : Chest, c = Gold ∨ c = Silver

-- Define the truth about what the chests must contain
def contains_gold (c : Chest) : Prop
def contains_silver (c : Chest) : Prop
def contains_copper (c : Chest) : Prop

-- The conditions given in the problem, stating that the true contents are different from the inscriptions
axiom incorrect_inscriptions :
  (¬ chest_1_inscription) ∧
  (¬ chest_2_inscription) ∧
  (¬ chest_3_inscription)

-- The problem states we have exactly one chest of each type 
axiom one_gold : ∃ c : Chest, contains_gold c
axiom one_silver : ∃ c : Chest, contains_silver c
axiom one_copper : ∃ c : Chest, contains_copper c

-- The main theorem we need to prove
theorem chests_contents :
  (∀ c : Chest, contains_silver c ↔ c = Chest₁) ∧
  (∀ c : Chest, contains_gold c ↔ c = Chest₂) ∧
  (∀ c : Chest, contains_copper c ↔ c = Chest₃) :=
sorry

end chests_contents_l391_391790


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391408

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391408


namespace chest_contents_correct_l391_391762

-- Define the chests
inductive Chest
| chest1
| chest2
| chest3

open Chest

-- Define the contents
inductive Contents
| gold
| silver
| copper

open Contents

-- Define the labels, all of which are incorrect
def label (c : Chest) : Prop :=
  match c with
  | chest1 => gold
  | chest2 => silver
  | chest3 => gold ∨ silver

-- Assuming the labels are all incorrect
axiom label_incorrect : ∀ (c : Chest), ¬(label c = true)

-- One chest contains each type of coin
axiom one_gold : ∃ c, ∀ x, x ≠ c → Contents x ≠ gold
axiom one_silver : ∃ c, ∀ x, x ≠ c → Contents x ≠ silver
axiom one_copper : ∃ c, ∀ x, x ≠ c → Contents x ≠ copper

-- Determine the contents of each chest
def chest1_contents := silver
def chest2_contents := gold
def chest3_contents := copper

-- Prove the correspondence
theorem chest_contents_correct :
  (chest1_contents = silver) ∧ 
  (chest2_contents = gold) ∧ 
  (chest3_contents = copper) :=
by
  split
  · exact sorry
  split
  · exact sorry
  · exact sorry

end chest_contents_correct_l391_391762


namespace student_arrangement_l391_391892

theorem student_arrangement (students : Finset ℕ) (A B : ℕ) (classes : Finset ℕ) :
  (5 ∈ students) → A ∈ students → B ∈ students → (3 ∈ classes) → 
  (∀ c ∈ classes, ∃ s ∈ students, true) → (∀ c₁ c₂ ∈ classes, c₁ ≠ c₂ → ∃ s₁ s₂ ∈ students, s₁ ≠ s₂ ∧ (A = s₁ ∨ A = s₂ ∧ B = s₁ ∨ B = s₂)) → 
  (∃ num_arrangements : ℕ, num_arrangements = 36) :=
by
  sorry

end student_arrangement_l391_391892


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391312
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391312


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391407

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391407


namespace even_integers_between_5000_8000_with_four_distinct_digits_l391_391680

theorem even_integers_between_5000_8000_with_four_distinct_digits : 
  let count := (sum (λ d₁, if d₁ ∈ [5, 6, 7] then
     (sum (λ d₄, if d₄ ≠ d₁ ∧ d₄ % 2 = 0 then
     (sum (λ d₂, if d₂ ≠ d₁ ∧ d₂ ≠ d₄ then
     (sum (λ d₃, if d₃ ≠ d₁ ∧ d₃ ≠ d₂ ∧ d₃ ≠ d₄ then 1 else 0))
     else 0))
     else 0))
     else 0) 
  in count = 728 := 
by
  sorry

end even_integers_between_5000_8000_with_four_distinct_digits_l391_391680


namespace greatest_prime_factor_of_sum_factorials_l391_391274

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391274


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391383

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391383


namespace domain_of_f_zeros_of_f_l391_391655

-- Given condition
variable {a : ℝ} (ha : 0 < a ∧ a < 1)

-- Define the function
def f (x : ℝ) : ℝ := log a (1 - x) + log a (x + 3)

-- Domain condition
theorem domain_of_f : ∀ x, -3 < x ∧ x < 1 ↔ 1 - x > 0 ∧ x + 3 > 0 :=
by
  intro x
  split
  · intro h
    dsimp at h
    rw [←h.left, ←h.right]
    linarith
  · intro h
    linarith
  done

-- Zeros condition
theorem zeros_of_f : ∀ x, f x = 0 ↔ x = -1 + sqrt 3 ∨ x = -1 - sqrt 3 :=
by
  intro x
  have hx : f x = log a (-(x ^ 2) - 2 * x + 3) := sorry  -- Simplify f(x) expression
  rw [hx]
  -- Showing that f(x) = 0 implies log expression equal to 1
  rw [log_eq_zero ha]
  split
  · intro h
    apply quadratic.zero_0 h
    {
      linarith,
    linarith
      refined ⟨le_of_lt (sqrt_pos.mpr _), le_of_lt (sqrt_pos.mpr _)⟩ sorry sorry
    }
  use [ -1 + sqrt 3, -1 - sqrt 3 ],
  done

end domain_of_f_zeros_of_f_l391_391655


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391243

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391243


namespace greatest_prime_factor_of_sum_factorials_l391_391283

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391283


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391355

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391355


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391378

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391378


namespace max_soapboxes_in_carton_l391_391491

/-- Definitions of dimensions for carton and soap box --/
def carton_length : ℕ := 48
def carton_width : ℕ := 25
def carton_height : ℕ := 60

def soapbox_length : ℕ := 8
def soapbox_width : ℕ := 6
def soapbox_height : ℕ := 5

/-- Lean statement that captures the mathematical proof problem --/
theorem max_soapboxes_in_carton :
  let boxes_in_length := carton_length / soapbox_length,
      boxes_in_width := carton_width / soapbox_height,
      boxes_in_height := carton_height / soapbox_width
  in boxes_in_length * boxes_in_width * boxes_in_height = 300 := by
  sorry

end max_soapboxes_in_carton_l391_391491


namespace kite_angle_ADC_l391_391714

theorem kite_angle_ADC (ABCD : Type) [kite ABCD] {A B C D : ABCD}
  (h1: AB = AD) (h2: BC = CD) (h3: ∠ ABC = 2 * ∠ BCD) : ∠ ADC = 90 :=
sorry

end kite_angle_ADC_l391_391714


namespace petya_friends_l391_391048

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l391_391048


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391252

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391252


namespace trigonometric_identity_l391_391966

theorem trigonometric_identity (α : ℝ) : 
  sin (α * π / 180) ^ 2 + cos ((30 - α) * π / 180) ^ 2 - sin (α * π / 180) * cos ((30 - α) * π / 180) = 3 / 4 :=
by
  sorry

end trigonometric_identity_l391_391966


namespace Petya_friends_l391_391081

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l391_391081


namespace chests_content_l391_391749

-- Define the chests and their labels
inductive CoinContent where
  | gold : CoinContent
  | silver : CoinContent
  | copper : CoinContent

structure Chest where
  label : CoinContent
  contents : CoinContent

-- Given conditions and incorrect labels
def chest1 : Chest := { label := CoinContent.gold, contents := sorry }
def chest2 : Chest := { label := CoinContent.silver, contents := sorry }
def chest3 : Chest := { label := CoinContent.gold, contents := sorry }

-- The proof problem
theorem chests_content :
  chest1.contents ≠ CoinContent.gold ∧
  chest2.contents ≠ CoinContent.silver ∧
  chest3.contents ≠ CoinContent.gold ∨ chest3.contents ≠ CoinContent.silver →
  chest1.contents = CoinContent.silver ∧
  chest2.contents = CoinContent.gold ∧
  chest3.contents = CoinContent.copper := by
  sorry

end chests_content_l391_391749


namespace number_of_rows_seating_10_is_zero_l391_391708

theorem number_of_rows_seating_10_is_zero :
  ∀ (y : ℕ) (total_people : ℕ) (total_rows : ℕ),
    (∀ (r : ℕ), r * 9 + (total_rows - r) * 10 = total_people) →
    total_people = 54 →
    total_rows = 6 →
    y = 0 :=
by
  sorry

end number_of_rows_seating_10_is_zero_l391_391708


namespace greatest_prime_factor_15_17_factorial_l391_391206

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391206


namespace number_of_communal_subsets_l391_391023

open Set

def S : Set ℝ² := { (cos ((2 * n * π) / 12), sin ((2 * n * π) / 12)) | n : ℕ, n < 12 }

def is_communal_subset (Q : Set (ℝ × ℝ)) : Prop :=
  ∃ C : Set (ℝ × ℝ), ∀ P ∈ S, (P ∈ Q ↔ P ∈ interior C)

theorem number_of_communal_subsets : 
  {Q : Set (ℝ × ℝ) | is_communal_subset Q}.card = 134 :=
sorry

end number_of_communal_subsets_l391_391023


namespace original_faculty_members_l391_391962

theorem original_faculty_members (x : ℝ) (h : 0.87 * x = 195) : x ≈ 224 :=
by
  sorry

end original_faculty_members_l391_391962


namespace greatest_prime_factor_15_fact_17_fact_l391_391181

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391181


namespace total_rent_l391_391928

theorem total_rent (R40 R60 : ℕ) (h1 : 0.8 * (40 * R40 + 60 * R60) = 40 * (R40 + 10) + 60 * (R60 - 10))
  (h2 : 2 * R40 + 3 * R60 = 50) : (40 * R40 + 60 * R60) = 1000 :=
by
  sorry

end total_rent_l391_391928


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391314
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391314


namespace petya_friends_l391_391047

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l391_391047


namespace trains_crossing_time_l391_391161

theorem trains_crossing_time :
  let length_of_each_train := 120 -- in meters
  let speed_of_each_train := 12 -- in km/hr
  let total_distance := length_of_each_train * 2
  let relative_speed := (speed_of_each_train * 1000 / 3600 * 2) -- in m/s
  total_distance / relative_speed = 36 := 
by
  -- Since we only need to state the theorem, the proof is omitted.
  sorry

end trains_crossing_time_l391_391161


namespace solution_l391_391032

structure Point :=
(x : ℝ)
(y : ℝ)

structure Parallelogram :=
(P Q R S : Point)

def is_not_above_x_axis (p : Point) : Prop :=
  p.y ≤ 0

def probability_of_not_above_x_axis (pgram : Parallelogram) : ℝ :=
  if (area_parallelogram pgram) = 0 then 0 else
  let region_not_above_x_axis_area := (area_parallelogram pgram) / 2 in
    region_not_above_x_axis_area / (area_parallelogram pgram)

noncomputable def area_point (p : Point) (q : Point) : ℝ :=
  p.x * q.y - q.x * p.y

noncomputable def area_parallelogram (pgram : Parallelogram) : ℝ :=
  let Parallelogram.mk p q r s := pgram in
  (1/2) * abs ((area_point p q) + (area_point q r) + (area_point r s) + (area_point s p))

def probability_problem : Prop :=
  let pgram := Parallelogram.mk ⟨4, 4⟩ ⟨-2, -2⟩ ⟨-8, -2⟩ ⟨2, 4⟩ in
  probability_of_not_above_x_axis pgram = 1 / 2

theorem solution : probability_problem := sorry

end solution_l391_391032


namespace petya_friends_l391_391080

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l391_391080


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391260

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391260


namespace ones_digit_36_to_power_expr_l391_391601

theorem ones_digit_36_to_power_expr : ∀ (n : ℕ), n = 36 * 5^5 → (36 ^ n) % 10 = 6 :=
by
  intro n h
  rw h
  sorry

end ones_digit_36_to_power_expr_l391_391601


namespace petya_friends_l391_391036

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l391_391036


namespace prob_replace_exactly_4_expected_num_replacements_l391_391893

namespace StreetLights

def lights : ℕ := 9

def binomial (n k : ℕ) : ℕ := nat.choose n k

def valid_replacements : ℕ := (binomial 6 4) * (4.factorial)

def total_replacements : ℕ := binomial 9 4

def probability_replace_exactly_4 (valid total : ℕ) : ℚ :=
  valid / total

def expected_replacements (expected_value : ℚ) : ℚ := expected_value

def valid_probability (p : ℚ) : Prop := p = 20 / 21

def valid_expected (e : ℚ) : Prop := e = 3.32

theorem prob_replace_exactly_4 :
  valid_probability (probability_replace_exactly_4 valid_replacements total_replacements) :=
sorry

theorem expected_num_replacements :
  valid_expected (expected_replacements 3.32) :=
sorry

end StreetLights

end prob_replace_exactly_4_expected_num_replacements_l391_391893


namespace minimize_beta_delta_l391_391004

open Complex

noncomputable def f (z : ℂ) (β δ : ℂ) := (3 + 2*I) * z^2 + β * z + δ

theorem minimize_beta_delta (β δ : ℂ) (h1 : (f (1 + I) β δ).im = 0) (h2 : (f (-I) β δ).im = 0) :
  |β| + |δ| = Real.sqrt 5 + 3 := 
sorry

end minimize_beta_delta_l391_391004


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391371

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391371


namespace chests_content_l391_391767

theorem chests_content (A B C : Type) (chest1 chest2 chest3 : A)
  (label1 : chest1 = "gold coins") (label2 : chest2 = "silver coins") (label3 : chest3 = "gold or silver coins")
  (gold silver copper : A)
  (h1 : chest1 ≠ gold)
  (h2 : chest2 ≠ silver)
  (h3 : chest3 ≠ gold ∧ chest3 ≠ silver)
  (t1 : chest1 = silver)
  (t2 : chest2 = gold)
  (t3 : chest3 = copper) :
  True := sorry

end chests_content_l391_391767


namespace petya_has_19_friends_l391_391050

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l391_391050


namespace find_cos_A_find_c_l391_391706

-- Conditions of the problem definitions
variables {A B C : ℝ}
variables {a b c : ℝ}

-- Assuming a given value for a and the arithmetic sequence.
variables (h₁ : a = 2 * b)
variables (h₂ : sin A + sin B = 2 * sin C)
variables (h₃ : 1/2 * b * c * sin A = 8 * sqrt 15 / 3)

-- First theorem: Finding cos A
theorem find_cos_A : cos A = -1 / 4 :=
by {
    sorry
}

-- Second theorem: Finding c given the area of triangle and cos A
theorem find_c (h₄ : cos A = -1 / 4) : c = 4 * sqrt 2 :=
by {
    sorry
}

end find_cos_A_find_c_l391_391706


namespace evaluate_expression_correct_l391_391590

theorem evaluate_expression_correct (a : ℝ) (h : a = 3) : 
  (3 * a^(-2) + (a^(-1)) / 3) / a^2 = 4 / 81 := 
by 
  sorry

end evaluate_expression_correct_l391_391590


namespace ratio_m_div_x_l391_391929

variable (a b x m : ℝ) (k : ℝ) (h : a / b = 4 / 5) (ha : 0 < a) (hb : 0 < b)

-- Definitions from the conditions
def defined_x := x = a * 1.25
def defined_m := m = b * 0.2

-- Statement to prove
theorem ratio_m_div_x : defined_x → defined_m → m / x = 0.2 :=
by
  intros
  sorry

end ratio_m_div_x_l391_391929


namespace find_f_f_neg_two_l391_391653

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 1 else x

theorem find_f_f_neg_two : f (f (-2)) = 3 :=
by
  have h1 : f (-2) = 3 := by
    rw [f]
    simp
  have h2 : f (3) = 3 := by
    rw [f]
    simp
  rw [← h1]
  exact h2

end find_f_f_neg_two_l391_391653


namespace greatest_prime_factor_of_15_l391_391450

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391450


namespace incenter_inequality_l391_391099

theorem incenter_inequality (ABC : Triangle) (I : Incenter ABC) (l_A l_B l_C : AngleBisector ABC) :
  1 / 4 < (dist I A * dist I B * dist I C) / (l_A * l_B * l_C) ∧ 
  (dist I A * dist I B * dist I C) / (l_A * l_B * l_C) ≤ 8 / 27 :=
  sorry

end incenter_inequality_l391_391099


namespace chests_contents_l391_391792

-- Define the coin types
inductive CoinType
| Gold
| Silver
| Copper

open CoinType

-- Define the chests
def Chest : Type

-- Define the inscriptions on the chests (all of which are known to be incorrect)
def chest_1_inscription : Prop := ∀ c : Chest, c = Gold
def chest_2_inscription : Prop := ∀ c : Chest, c = Silver
def chest_3_inscription : Prop := ∀ c : Chest, c = Gold ∨ c = Silver

-- Define the truth about what the chests must contain
def contains_gold (c : Chest) : Prop
def contains_silver (c : Chest) : Prop
def contains_copper (c : Chest) : Prop

-- The conditions given in the problem, stating that the true contents are different from the inscriptions
axiom incorrect_inscriptions :
  (¬ chest_1_inscription) ∧
  (¬ chest_2_inscription) ∧
  (¬ chest_3_inscription)

-- The problem states we have exactly one chest of each type 
axiom one_gold : ∃ c : Chest, contains_gold c
axiom one_silver : ∃ c : Chest, contains_silver c
axiom one_copper : ∃ c : Chest, contains_copper c

-- The main theorem we need to prove
theorem chests_contents :
  (∀ c : Chest, contains_silver c ↔ c = Chest₁) ∧
  (∀ c : Chest, contains_gold c ↔ c = Chest₂) ∧
  (∀ c : Chest, contains_copper c ↔ c = Chest₃) :=
sorry

end chests_contents_l391_391792


namespace russian_players_pairing_probability_l391_391941

theorem russian_players_pairing_probability :
  let total_players := 10
  let russian_players := 4
  (russian_players * (russian_players - 1)) / (total_players * (total_players - 1)) * 
  ((russian_players - 2) * (russian_players - 3)) / ((total_players - 2) * (total_players - 3)) = 1 / 21 :=
by
  sorry

end russian_players_pairing_probability_l391_391941


namespace swimming_speed_proof_l391_391159

variables (S : ℝ)

def total_distance := 6
def running_speed := 10
def average_speed := 7.5
def run_distance := 3

def time_running := run_distance / running_speed
def time_swimming := run_distance / S
def total_time := total_distance / average_speed

theorem swimming_speed_proof :
  time_running + time_swimming = total_time → S = 6 :=
begin
  intro h,
  rw [←h],
  sorry, -- Proof not required as per instruction
end

end swimming_speed_proof_l391_391159


namespace greatest_prime_factor_15_fact_17_fact_l391_391229

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391229


namespace triangle_area_l391_391127

-- Define the polynomial equation
def polymial_eq (x : ℝ) : Prop := x^3 - 3 * x^2 + 4 * x - (8 / 5) = 0

-- Define the real roots
variables (a b c : ℝ)
axiom roots : polymial_eq a ∧ polymial_eq b ∧ polymial_eq c

-- Define the problem statement
theorem triangle_area : 
  (roots a b c) →
  a + b + c = 3 →
  let q := (a + b + c) / 2 in
  q * (q - a) * (q - b) * (q - c) = 12 / 5 :=
sorry

end triangle_area_l391_391127


namespace sin_alpha_l391_391698

theorem sin_alpha (α : ℝ) (h : (\sin (5 * Real.pi / 6), \cos (5 * Real.pi / 6)) = (\cos α, \sin α)) :
  \sin α = - \sqrt 3 / 2 :=
sorry

end sin_alpha_l391_391698


namespace greatest_prime_factor_of_sum_factorials_l391_391275

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391275


namespace required_fencing_l391_391957

-- Given definitions and conditions
def area (L W : ℕ) : ℕ := L * W

def fencing (W L : ℕ) : ℕ := 2 * W + L

theorem required_fencing
  (L W : ℕ)
  (hL : L = 10)
  (hA : area L W = 600) :
  fencing W L = 130 := by
  sorry

end required_fencing_l391_391957


namespace chests_contents_l391_391794

-- Define the coin types
inductive CoinType
| Gold
| Silver
| Copper

open CoinType

-- Define the chests
def Chest : Type

-- Define the inscriptions on the chests (all of which are known to be incorrect)
def chest_1_inscription : Prop := ∀ c : Chest, c = Gold
def chest_2_inscription : Prop := ∀ c : Chest, c = Silver
def chest_3_inscription : Prop := ∀ c : Chest, c = Gold ∨ c = Silver

-- Define the truth about what the chests must contain
def contains_gold (c : Chest) : Prop
def contains_silver (c : Chest) : Prop
def contains_copper (c : Chest) : Prop

-- The conditions given in the problem, stating that the true contents are different from the inscriptions
axiom incorrect_inscriptions :
  (¬ chest_1_inscription) ∧
  (¬ chest_2_inscription) ∧
  (¬ chest_3_inscription)

-- The problem states we have exactly one chest of each type 
axiom one_gold : ∃ c : Chest, contains_gold c
axiom one_silver : ∃ c : Chest, contains_silver c
axiom one_copper : ∃ c : Chest, contains_copper c

-- The main theorem we need to prove
theorem chests_contents :
  (∀ c : Chest, contains_silver c ↔ c = Chest₁) ∧
  (∀ c : Chest, contains_gold c ↔ c = Chest₂) ∧
  (∀ c : Chest, contains_copper c ↔ c = Chest₃) :=
sorry

end chests_contents_l391_391794


namespace rightmost_three_digits_of_7_pow_2023_l391_391166

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 637 :=
sorry

end rightmost_three_digits_of_7_pow_2023_l391_391166


namespace economical_purchase_method_l391_391921

theorem economical_purchase_method
  (p1 p2 : ℝ) (hp1 : p1 > 0) (hp2 : p2 > 0) (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2 * p1 * p2) / (p1 + p2) ≤ (p1 + p2) / 2 := 
begin
  apply am_gm,
end

end economical_purchase_method_l391_391921


namespace petya_friends_l391_391045

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l391_391045


namespace total_profit_l391_391925

theorem total_profit (a_investment b_investment c_investment c_share : ℕ)
  (h₁ : a_investment = 30000) (h₂ : b_investment = 45000) (h₃ : c_investment = 50000)
  (h₄ : c_share = 36000) : 
  let total_parts := 6 + 9 + 10 in
  let one_part := 3600 in
  let total_profit := one_part * total_parts in
  total_profit = 90000 :=
by
  -- Parameters for investments
  sorry

end total_profit_l391_391925


namespace find_numbers_l391_391503

theorem find_numbers (x y : ℕ) (h1 : 100 ≤ x ∧ x ≤ 999) (h2 : 100 ≤ y ∧ y ≤ 999) (h3 : 1000 * x + y = 7 * x * y) :
  x = 143 ∧ y = 143 :=
by
  sorry

end find_numbers_l391_391503


namespace f_analysis_angle_C_l391_391659

-- Definitions based on conditions
def f (x : ℝ) : ℝ := 1 / 2 * sin(π * x + π / 2)
def a : ℝ := 1
def b : ℝ := Real.sqrt 2
def A : ℝ := π / 6

-- Problem Statements
theorem f_analysis :
  f(x) = 1 / 2 * cos(π * x) :=
sorry

theorem angle_C :
  ∃ (C : ℝ), (C = 7 * π / 12 ∨ C = π / 12) :=
sorry

end f_analysis_angle_C_l391_391659


namespace soccer_players_arrangement_l391_391522

theorem soccer_players_arrangement : ∃ (x1 x2 x3 x4 : ℝ), 
    let dists := {(abs (x1 - x2)), (abs (x1 - x3)), (abs (x1 - x4)), (abs (x2 - x3)), (abs (x2 - x4)), (abs (x3 - x4))} in
    dists = {1, 2, 3, 4, 5, 6} :=
sorry

end soccer_players_arrangement_l391_391522


namespace chests_content_l391_391772

theorem chests_content (A B C : Type) (chest1 chest2 chest3 : A)
  (label1 : chest1 = "gold coins") (label2 : chest2 = "silver coins") (label3 : chest3 = "gold or silver coins")
  (gold silver copper : A)
  (h1 : chest1 ≠ gold)
  (h2 : chest2 ≠ silver)
  (h3 : chest3 ≠ gold ∧ chest3 ≠ silver)
  (t1 : chest1 = silver)
  (t2 : chest2 = gold)
  (t3 : chest3 = copper) :
  True := sorry

end chests_content_l391_391772


namespace petya_has_19_friends_l391_391056

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l391_391056


namespace sum_proper_divisors_600_l391_391980

theorem sum_proper_divisors_600 : ∑ d in (finset.filter (λ d, d < 600) (finset.divisors 600)), d = 1260 := 
sorry

end sum_proper_divisors_600_l391_391980


namespace greatest_prime_factor_15_17_factorial_l391_391217

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391217


namespace james_opened_three_windows_l391_391726

theorem james_opened_three_windows (W : ℕ) (h_same_windows : ∀ W1 W2, W1 = W2 → W1 = W ∧ W2 = W)
(h_total_tabs : ∀ W1 W2, 10 * W1 + 10 * W2 = 60 → W1 = W2) 
(h_eq_W: ∀ W1 W2, W1 = W2 → 20 * W = 60) :
  W = 3 := 
sorry

end james_opened_three_windows_l391_391726


namespace green_area_growth_time_l391_391145

theorem green_area_growth_time (a n : ℕ) (h₁ : 1.15^a = 4) (h₂ : 1.15^(a + n) = 12) : n = 8 :=
by
  sorry

end green_area_growth_time_l391_391145


namespace smallest_xym_sum_l391_391809

def is_two_digit_integer (n : ℤ) : Prop :=
  10 ≤ n ∧ n < 100

def reversed_digits (x y : ℤ) : Prop :=
  ∃ a b : ℤ, x = 10 * a + b ∧ y = 10 * b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9

def odd_multiple_of_9 (n : ℤ) : Prop :=
  ∃ k : ℤ, k % 2 = 1 ∧ n = 9 * k

theorem smallest_xym_sum :
  ∃ (x y m : ℤ), is_two_digit_integer x ∧ is_two_digit_integer y ∧ reversed_digits x y ∧ x^2 + y^2 = m^2 ∧ odd_multiple_of_9 (x + y) ∧ x + y + m = 169 :=
by
  sorry

end smallest_xym_sum_l391_391809


namespace range_of_a_l391_391719

open Real

variable (a x : ℝ)
def line_l (a x : ℝ) := a * x + -2 - a * x = 0
def point_A := (-3, 0)
def points_equal (M O A : ℝ × ℝ) := dist M A = 2 * dist M O

theorem range_of_a (a : ℝ):
  (∃ M : ℝ × ℝ, points_equal M (0, 0) point_A ∧ line_l a (M.1)) →
  a ≤ 0 ∨ a ≥ 4 / 3 :=
by 
  sorry

end range_of_a_l391_391719


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391246

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391246


namespace team_a_takes_fewer_hours_l391_391160

theorem team_a_takes_fewer_hours
  (distance : ℝ)
  (VR VA : ℝ)
  (VR_eq : VR = 20)
  (VA_eq : VA = VR + 5)
  (distance_eq : distance = 300) :
  let TA := distance / VA,
      TR := distance / VR in
  TR - TA = 3 :=
by
  sorry

end team_a_takes_fewer_hours_l391_391160


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391299

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391299


namespace max_cubes_in_box_l391_391457

-- Define the dimensions of the box and the volume of the cube
def length : ℕ := 13
def width : ℕ := 17
def height : ℕ := 22
def cube_volume : ℕ := 43

-- Define the volume calculation for the rectangular box
def box_volume : ℕ := length * width * height

-- Define the maximum number of cubes that can fit in the box
def max_cubes : ℕ := box_volume / cube_volume

-- Verify that the maximum number of 43 cubic centimetre cubes that can fit in the box is 114
theorem max_cubes_in_box : max_cubes = 114 := by
  sorry

end max_cubes_in_box_l391_391457


namespace car_speed_onschedule_l391_391944

variable (v : ℝ)

def on_time_speed (distance speed : ℝ) : ℝ := 
  distance / speed

def late_speed_time (distance : ℝ) : ℝ :=
  distance / 50

theorem car_speed_onschedule :
  on_time_speed 225 v + 3 / 4 = late_speed_time 225 → v = 60 :=
by 
  sorry

end car_speed_onschedule_l391_391944


namespace greatest_prime_factor_of_factorial_sum_l391_391334

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391334


namespace koschei_chests_l391_391750

theorem koschei_chests :
  ∃ (contents : Fin 3 → String), 
    -- All chests labels are incorrect
    (contents 0 ≠ "gold coins" ∧ contents 1 ≠ "silver coins" ∧ contents 2 ≠ "gold or silver coins") ∧ 
    -- Each chest contains exactly one type of coin 
    (∀ i j : Fin 3, i ≠ j → contents i ≠ contents j) ∧ 
    -- Providing the final conclusion about what is in each chest
    (contents 0 = "silver coins" ∧ contents 1 = "gold coins" ∧ contents 2 = "copper coins") :=
begin
  use (λ k, if k = 0 then "silver coins" else if k = 1 then "gold coins" else "copper coins"),
  split,
  {
    -- Proof of all labels being incorrect
    split; simp,
  },
  split,
  {
    -- Proof of each chest containing a unique type of coin
    intros i j h,
    cases i; cases j; simp [h],
  },
  {
    -- Proof of the final conclusion
    split; split; simp,
  },
end

end koschei_chests_l391_391750


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391356

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391356


namespace greatest_prime_factor_15f_plus_17f_l391_391186

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391186


namespace greatest_prime_factor_of_sum_l391_391398

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391398


namespace hexagon_side_lengths_l391_391990

theorem hexagon_side_lengths (n m : ℕ) (AB BC : ℕ) (P : ℕ) :
  n + m = 6 ∧ n * 4 + m * 7 = 38 ∧ AB = 4 ∧ BC = 7 → m = 4 :=
by
  sorry

end hexagon_side_lengths_l391_391990


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391357

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391357


namespace square_area_is_8_point_0_l391_391940

theorem square_area_is_8_point_0 (A B C D E F : ℝ) 
    (h_square : E + F = 4)
    (h_diag : 1 + 2 + 1 = 4) : 
    ∃ (s : ℝ), s^2 = 8 :=
by
  sorry

end square_area_is_8_point_0_l391_391940


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391434

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391434


namespace soccer_player_positions_exist_l391_391527

theorem soccer_player_positions_exist :
  ∃ x1 x2 x3 x4 : ℝ,
    ({| real.abs (x1 - x2),
       real.abs (x1 - x3),
       real.abs (x1 - x4),
       real.abs (x2 - x3),
       real.abs (x2 - x4),
       real.abs (x3 - x4) |} = {| 1, 2, 3, 4, 5, 6 |}) :=
begin
  use [0, 1, 4, 6],
  sorry
end

end soccer_player_positions_exist_l391_391527


namespace percentage_vets_recommend_puppy_kibble_l391_391486

theorem percentage_vets_recommend_puppy_kibble :
  ∀ (P : ℝ), (30 / 100 * 1000 = 300) → (1000 * P / 100 + 100 = 300) → P = 20 :=
by
  intros P h1 h2
  sorry

end percentage_vets_recommend_puppy_kibble_l391_391486


namespace petya_friends_l391_391038

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l391_391038


namespace greatest_prime_factor_15f_plus_17f_l391_391189

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391189


namespace library_students_time_l391_391901

theorem library_students_time (d e f : ℕ) (n : ℝ) 
  (h1 : ∃ d e f, n = d - e * real.sqrt f)
  (h2 : d > 0 ∧ e > 0 ∧ f > 0)
  (h3 : ∀ p : ℕ, prime p → p^2 ∣ f → false)
  (h4 : (60 - n)^2 / 3600 = 0.3) :
  d + e + f = 16 := 
sorry

end library_students_time_l391_391901


namespace find_k_for_parallel_vectors_l391_391664

variable (a b c : ℝ × ℝ)
variable (k : ℝ)

def vector_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem find_k_for_parallel_vectors 
  (h_a : a = (2, -1)) 
  (h_b : b = (1, 1)) 
  (h_c : c = (-5, 1)) 
  (h_parallel : vector_parallel (a.1 + k * b.1, a.2 + k * b.2) c) : 
  k = 1 / 2 :=
by
  unfold vector_parallel at h_parallel
  simp at h_parallel
  sorry

end find_k_for_parallel_vectors_l391_391664


namespace sequence_sum_l391_391647

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1
noncomputable def b_n (n : ℕ) : ℕ := 3^(n - 1)
noncomputable def c_n (n : ℕ) : ℕ := 2 * a_n n + b_n n

theorem sequence_sum (n : ℕ) : 
  (finset.range n).sum c_n = 2 * n^2 + 4 * n + (3^n * 3 / 2) - 3 / 2 :=
begin
  sorry
end

end sequence_sum_l391_391647


namespace line_AB_fixed_point_minimum_area_of_triangle_ABC_l391_391663

-- Definitions of given conditions and known fixed points
def parabola {p : ℝ} (x y : ℝ) : Prop := y^2 = 2 * p * x
def point_C (x y : ℝ) : Prop := (x = 1) ∧ (y  = 2)
def fixed_point_Q (x y : ℝ) : Prop := (x = 3) ∧ (y = 2)


-- Conditions for point A
def on_parabola (x y : ℝ) : Prop := ∃ (y0 : ℝ), y0 ≠ 2 ∧ (x = y0^2 / 4) ∧ (y = y0)
def line_AC (x y y0: ℝ) : Prop := y - 2 = (4 * (y0 - 2) / (y0^2 - 4)) * (x - 1) 

-- Intersection with y = x + 3 and point B conditions
def line_with_P (x y x_P y_P : ℝ) : Prop := (y = x + 3) ∧ (y - 2 = (4 * (y0 - 2) / (y0^2 - 4)) * (x - 1))
def parallel_x_axis (x y x_P y_P: ℝ) : Prop := (y = y_P) ∧ parabola x y

-- Definitions of the proof problem
theorem line_AB_fixed_point {p : ℝ} (x_A y_A : ℝ) (h_para : parabola 4 x_A y_A) (h_q : fixed_point_Q 3 2)  
 (h_C : point_C 1 2) :
  ∃ x_B y_B : ℝ, line_with_P x_A y_A x_B y_B ∧ parallel_x_axis x_B y_B x y ∧  (∀ x_A y_A: ℝ, x_A = y_A): y_B =
  fixed_point_Q 3 2 :=
sorry

theorem minimum_area_of_triangle_ABC {p : ℝ} (x_A y_A : ℝ) (h_para : parabola 4 x_A y_A) (h_q : fixed_point_Q 3 2)  
 (h_C : point_C 1 2) :
  ∃ x_B y_B : ℝ, 
  line_with_P x_A y_A x_B y_B ∧ parallel_x_axis x_B y_B x_A y_A  ∧ 
  (∀ x_A y_A: ℝ, x_A = _TriangleArea x):
 x = y :=
sorry

end line_AB_fixed_point_minimum_area_of_triangle_ABC_l391_391663


namespace log_sin_decrease_interval_l391_391596

open Real

noncomputable def interval_of_decrease (x : ℝ) : Prop :=
  ∃ (k : ℤ), (k * π + π / 8 < x ∧ x ≤ k * π + 3 * π / 8)

theorem log_sin_decrease_interval (x : ℝ) :
  interval_of_decrease x ↔ ∃ (k : ℤ), (k * π + π / 8 < x ∧ x ≤ k * π + 3 * π / 8) :=
by
  sorry

end log_sin_decrease_interval_l391_391596


namespace mean_weight_participants_l391_391884

def weights_120s := [123, 125]
def weights_130s := [130, 132, 133, 135, 137, 138]
def weights_140s := [141, 145, 145, 149, 149]
def weights_150s := [150, 152, 153, 155, 158]
def weights_160s := [164, 167, 167, 169]

def total_weights := weights_120s ++ weights_130s ++ weights_140s ++ weights_150s ++ weights_160s

def total_sum : ℕ := total_weights.sum
def total_count : ℕ := total_weights.length

theorem mean_weight_participants : (total_sum : ℚ) / total_count = 3217 / 22 := by
  sorry -- Proof goes here, but we're skipping it

end mean_weight_participants_l391_391884


namespace greatest_prime_factor_15_17_factorial_l391_391209

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391209


namespace rectangle_side_length_integer_l391_391012

theorem rectangle_side_length_integer 
    (R : Type) (R_i : ℕ → Type) (n : ℕ)
    (h_union : R = ⋃ (i : Fin n), (R_i i))
    (h_sides_parallel : ∀ i, ∀ (s1 s2 : ℕ), sides_parallel_of (R_i i) (s1) (s2))
    (h_non_overlap : ∀ i j, i ≠ j → non_overlap_of (R_i i) (R_i j))
    (h_integer_side : ∀ i, integer_side (R_i i)) : 
    ∃ s, integer_side R s := 
sorry

end rectangle_side_length_integer_l391_391012


namespace smallest_x_value_l391_391685

-- Definitions based on given problem conditions
def is_solution (x y : ℕ) : Prop :=
  0 < x ∧ 0 < y ∧ (3 : ℝ) / 4 = y / (252 + x)

theorem smallest_x_value : ∃ x : ℕ, ∀ y : ℕ, is_solution x y → x = 0 :=
by
  sorry

end smallest_x_value_l391_391685


namespace John_l391_391731

constant S : ℝ
constant savings_rate : ℝ := 0.10
constant general_savings : ℝ := 400

axiom savings_condition : savings_rate * S = general_savings

theorem John's_base_salary :
  S = 4000 := by
  have h1 : 0.10 * S = 400 := savings_condition
  have h2 : S = 4000 := by linarith
  exact h2

end John_l391_391731


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391409

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391409


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391264

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391264


namespace greatest_prime_factor_of_sum_l391_391402

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391402


namespace soccer_players_arrangement_l391_391521

theorem soccer_players_arrangement : ∃ (x1 x2 x3 x4 : ℝ), 
    let dists := {(abs (x1 - x2)), (abs (x1 - x3)), (abs (x1 - x4)), (abs (x2 - x3)), (abs (x2 - x4)), (abs (x3 - x4))} in
    dists = {1, 2, 3, 4, 5, 6} :=
sorry

end soccer_players_arrangement_l391_391521


namespace greatest_prime_factor_of_factorial_sum_l391_391322

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391322


namespace f_one_eq_zero_f_neg_one_eq_zero_f_is_even_l391_391993

noncomputable def f : ℝ → ℝ := sorry -- definition of f needs to be constructed

axiom functional_eq (x y : ℝ) : f(x * y) = f(x) + f(y)
axiom non_zero_f : ∃ x : ℝ, f(x) ≠ 0

theorem f_one_eq_zero : f(1) = 0 :=
by
  apply sorry

theorem f_neg_one_eq_zero : f(-1) = 0 :=
by
  apply sorry

theorem f_is_even : ∀ x : ℝ, f(-x) = f(x) :=
by
  apply sorry

end f_one_eq_zero_f_neg_one_eq_zero_f_is_even_l391_391993


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391360

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391360


namespace bounded_area_is_9_l391_391548

-- The definitions of the bounding functions
def f1 (x : ℝ) : ℝ := x^2 - 4*x + 3
def f2 (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- The proof statement: the area between f1 and f2 from x=0 to x=3 is 9
theorem bounded_area_is_9 :
  (∫ x in 0..3, f2 x - f1 x) = 9 := by
  sorry

end bounded_area_is_9_l391_391548


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391258

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391258


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391307
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391307


namespace distance_is_12_l391_391727

def distance_to_Mount_Overlook (D : ℝ) : Prop :=
  let T1 := D / 4
  let T2 := D / 6
  T1 + T2 = 5

theorem distance_is_12 : ∃ D : ℝ, distance_to_Mount_Overlook D ∧ D = 12 :=
by
  use 12
  rw [distance_to_Mount_Overlook]
  sorry

end distance_is_12_l391_391727


namespace greatest_prime_factor_of_sum_l391_391393

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391393


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391313
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391313


namespace walters_exceptional_days_l391_391163

variable (b w : ℕ)
variable (days_total dollars_total : ℕ)
variable (normal_earn exceptional_earn : ℕ)
variable (at_least_exceptional_days : ℕ)

-- Conditions
def conditions : Prop :=
  days_total = 15 ∧
  dollars_total = 70 ∧
  normal_earn = 4 ∧
  exceptional_earn = 6 ∧
  at_least_exceptional_days = 5 ∧
  b + w = days_total ∧
  normal_earn * b + exceptional_earn * w = dollars_total ∧
  w ≥ at_least_exceptional_days

-- Theorem to prove the number of exceptional days is 5
theorem walters_exceptional_days (h : conditions b w days_total dollars_total normal_earn exceptional_earn at_least_exceptional_days) : w = 5 :=
sorry

end walters_exceptional_days_l391_391163


namespace equation_solution_l391_391850

noncomputable def solveEquation (x : ℂ) : Prop :=
  -x^2 = (2*x + 4)/(x + 2)

theorem equation_solution (x : ℂ) (h : x ≠ -2) :
  solveEquation x ↔ x = -2 ∨ x = Complex.I * 2 ∨ x = - Complex.I * 2 :=
sorry

end equation_solution_l391_391850


namespace exists_positive_int_x_l391_391633

theorem exists_positive_int_x (a c : ℕ) (b : ℤ) (a_pos : 0 < a) (c_pos : 0 < c) :
  ∃ x : ℕ, 0 < x ∧ (a^x + x ≡ b [MOD c]) :=
by
  sorry

end exists_positive_int_x_l391_391633


namespace jill_net_monthly_salary_l391_391475

theorem jill_net_monthly_salary (S : ℝ) :
  (let discretionary_income := S / 5 in
   let vacation_fund := 0.30 * discretionary_income in
   let savings := 0.20 * discretionary_income in
   let socializing := 0.35 * discretionary_income in
   let remaining_amount := discretionary_income - vacation_fund - savings - socializing in
   remaining_amount = 105) →
  S = 3500 :=
begin
  intro h,
  sorry
end

end jill_net_monthly_salary_l391_391475


namespace greatest_prime_factor_15_fact_17_fact_l391_391220

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391220


namespace vertex_of_parabola_l391_391859

theorem vertex_of_parabola (y : ℝ → ℝ) (h : ∀ x : ℝ, y x = 3 * (x - 1) ^ 2 + 8) : ∃ v : ℝ × ℝ, v = (1, 8) :=
by
  use (1, 8)
  sorry

end vertex_of_parabola_l391_391859


namespace petya_friends_l391_391071

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l391_391071


namespace greatest_prime_factor_15_17_factorial_l391_391352

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391352


namespace jessica_birth_year_l391_391130

theorem jessica_birth_year (year_start : ℕ) (tenth_amc_8 : year_start + 9 = 1994) (age_jessica : 15) : 
  born_year = 1979 :=
by
  -- Add necessary conditions as Lean statements
  let year_start := 1985
  let tenth_amc_8 := year_start + 9
  let year_tenth_amc_8 := 1994
  let age_jessica := 15
  let born_year := year_tenth_amc_8 - age_jessica
  -- Write the final assertion for Lean theorem
  have h1 : born_year = 1994 - 15 := by rw [year_tenth_amc_8, age_jessica]
  exact h1

end jessica_birth_year_l391_391130


namespace probability_mask_with_ear_loops_l391_391952

-- Definitions from the conditions
def production_ratio_regular : ℝ := 0.8
def production_ratio_surgical : ℝ := 0.2
def proportion_ear_loops_regular : ℝ := 0.1
def proportion_ear_loops_surgical : ℝ := 0.2

-- Theorem statement based on the translated proof problem
theorem probability_mask_with_ear_loops :
  production_ratio_regular * proportion_ear_loops_regular +
  production_ratio_surgical * proportion_ear_loops_surgical = 0.12 :=
by
  -- Proof omitted
  sorry

end probability_mask_with_ear_loops_l391_391952


namespace complex_magnitude_relation_l391_391008

theorem complex_magnitude_relation (z1 z2 : ℂ) :
  (|z1 + z2|^2 + |z1 - z2|^2 = 2 * |z1|^2 + 2 * |z2|^2) := by
  sorry

end complex_magnitude_relation_l391_391008


namespace greatest_prime_factor_15_fact_17_fact_l391_391168

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391168


namespace rectangular_plot_breadth_l391_391479

theorem rectangular_plot_breadth (b l : ℝ) (A : ℝ)
  (h1 : l = 3 * b)
  (h2 : A = l * b)
  (h3 : A = 2700) : b = 30 :=
by sorry

end rectangular_plot_breadth_l391_391479


namespace soccer_players_positions_l391_391535

theorem soccer_players_positions :
  ∃ (a b c d : ℝ), a = 0 ∧ b = 1 ∧ c = 4 ∧ d = 6 ∧
  set_of (λ x, ∃ i j, i ≠ j ∧ x = abs (a - b) ∨ x = abs (a - c) ∨ x = abs (a - d) ∨ x = abs (b - c) ∨ x = abs (b - d) ∨ x = abs (c - d)) = {1, 2, 3, 4, 5, 6} :=
by
  use 0, 1, 4, 6
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  rw [set_of, abs]
  sorry

end soccer_players_positions_l391_391535


namespace probability_of_circumference_less_than_ten_times_area_l391_391500

theorem probability_of_circumference_less_than_ten_times_area :
  let outcomes := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  (∀ d ∈ outcomes, π * d < (10 * π * d^2) / 4) →
  set.univ.card = outcomes.card :=
by
  sorry

end probability_of_circumference_less_than_ten_times_area_l391_391500


namespace smallest_largest_same_parity_l391_391571

-- Here, we define the conditions, including the set A_P and the element c achieving the maximum.
def is_maximum (A_P : Set Int) (c : Int) : Prop := c ∈ A_P ∧ ∀ x ∈ A_P, x ≤ c
def has_uniform_parity (A_P : Set Int) : Prop := 
  ∀ a₁ a₂ ∈ A_P, (a₁ % 2 = 0 → a₂ % 2 = 0) ∧ (a₁ % 2 = 1 → a₂ % 2 = 1)

-- This statement confirms the parity uniformity of the smallest and largest elements of the set A_P.
theorem smallest_largest_same_parity (A_P : Set Int) (c : Int) 
  (hc_max: is_maximum A_P c) (h_uniform: has_uniform_parity A_P): 
  ∀ min max ∈ A_P, ((min = max ∨ min ≠ max) → (min % 2 = max % 2)) := 
by
  intros min max hmin hmax h_eq
  have h_parity := h_uniform min max hmin hmax
  cases nat.decidable_eq (min % 2) 0 with h_even h_odd
  { rw nat.mod_eq_zero_of_dvd h_even at h_parity,
    exact h_parity.1 h_even, },
  { rw nat.mod_eq_one_of_dvd h_odd at h_parity,
    exact h_parity.2 h_odd, }
  sorry

end smallest_largest_same_parity_l391_391571


namespace double_shot_espresso_cost_l391_391823

theorem double_shot_espresso_cost :
  let drip_coffee := 2 * 2.25
  let latte := 2 * 4.00
  let vanilla_syrup := 0.50
  let cold_brew := 2 * 2.50
  let cappuccino := 3.50
  let total_cost := 25.00 
  let known_costs := drip_coffee + latte + vanilla_syrup + cold_brew + cappuccino
    in total_cost - known_costs = 3.50 := by
  sorry

end double_shot_espresso_cost_l391_391823


namespace chest_contents_solution_l391_391782

-- Definitions corresponding to the conditions.
structure ChestContents :=
  (chest1 chest2 chest3 : String)

-- Given conditions
def labelsAreIncorrect (contents : ChestContents) : Prop :=
  contents.chest1 ≠ "Gold coins" ∧
  contents.chest2 ≠ "Silver coins" ∧
  contents.chest3 ≠ "Gold coins" ∧
  contents.chest3 ≠ "Silver coins"

def uniqueCoins (contents : ChestContents) : Prop :=
  (contents.chest1 = "Gold coins" ∨ contents.chest1 = "Silver coins" ∨ contents.chest1 = "Copper coins") ∧
  (contents.chest2 = "Gold coins" ∨ contents.chest2 = "Silver coins" ∨ contents.chest2 = "Copper coins") ∧
  (contents.chest3 = "Gold coins" ∨ contents.chest3 = "Silver coins" ∨ contents.chest3 = "Copper coins") ∧
  (contents.chest1 ≠ contents.chest2) ∧
  (contents.chest1 ≠ contents.chest3) ∧
  (contents.chest2 ≠ contents.chest3)

-- The proof statement
theorem chest_contents_solution : ∃ (contents : ChestContents),
  labelsAreIncorrect contents ∧ uniqueCoins contents ∧
  contents.chest1 = "Silver coins" ∧
  contents.chest2 = "Gold coins" ∧
  contents.chest3 = "Copper coins" :=
begin
  sorry
end

end chest_contents_solution_l391_391782


namespace required_barrels_of_pitch_l391_391497

def total_road_length : ℕ := 16
def bags_of_gravel_per_truckload : ℕ := 2
def barrels_of_pitch_per_truckload (bgt : ℕ) : ℚ := bgt / 5
def truckloads_per_mile : ℕ := 3

def miles_paved_day1 : ℕ := 4
def miles_paved_day2 : ℕ := (miles_paved_day1 * 2) - 1
def total_miles_paved_first_two_days : ℕ := miles_paved_day1 + miles_paved_day2
def remaining_miles_paved_day3 : ℕ := total_road_length - total_miles_paved_first_two_days

def truckloads_needed (miles : ℕ) : ℕ := miles * truckloads_per_mile
def barrels_of_pitch_needed (truckloads : ℕ) (bgt : ℕ) : ℚ := truckloads * barrels_of_pitch_per_truckload bgt

theorem required_barrels_of_pitch : 
  barrels_of_pitch_needed (truckloads_needed remaining_miles_paved_day3) bags_of_gravel_per_truckload = 6 := 
by
  sorry

end required_barrels_of_pitch_l391_391497


namespace angle_in_second_quadrant_l391_391699

theorem angle_in_second_quadrant 
  (θ : ℝ) 
  (h1 : sin θ * cos θ < 0) 
  (h2 : 2 * cos θ < 0) : 
  ∃ q, q = 2 := 
by {
  sorry
}

end angle_in_second_quadrant_l391_391699


namespace petya_friends_l391_391070

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l391_391070


namespace chests_contents_l391_391797

-- Define the coin types
inductive CoinType
| Gold
| Silver
| Copper

open CoinType

-- Define the chests
def Chest : Type

-- Define the inscriptions on the chests (all of which are known to be incorrect)
def chest_1_inscription : Prop := ∀ c : Chest, c = Gold
def chest_2_inscription : Prop := ∀ c : Chest, c = Silver
def chest_3_inscription : Prop := ∀ c : Chest, c = Gold ∨ c = Silver

-- Define the truth about what the chests must contain
def contains_gold (c : Chest) : Prop
def contains_silver (c : Chest) : Prop
def contains_copper (c : Chest) : Prop

-- The conditions given in the problem, stating that the true contents are different from the inscriptions
axiom incorrect_inscriptions :
  (¬ chest_1_inscription) ∧
  (¬ chest_2_inscription) ∧
  (¬ chest_3_inscription)

-- The problem states we have exactly one chest of each type 
axiom one_gold : ∃ c : Chest, contains_gold c
axiom one_silver : ∃ c : Chest, contains_silver c
axiom one_copper : ∃ c : Chest, contains_copper c

-- The main theorem we need to prove
theorem chests_contents :
  (∀ c : Chest, contains_silver c ↔ c = Chest₁) ∧
  (∀ c : Chest, contains_gold c ↔ c = Chest₂) ∧
  (∀ c : Chest, contains_copper c ↔ c = Chest₃) :=
sorry

end chests_contents_l391_391797


namespace simplify_abs_neg_pow_sub_l391_391843

theorem simplify_abs_neg_pow_sub (a b : ℤ) (h : a = 4) (h' : b = 6) : 
  (|-(a ^ 2) - b| = 22) := 
by
  sorry

end simplify_abs_neg_pow_sub_l391_391843


namespace petya_friends_l391_391076

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l391_391076


namespace intersection_product_l391_391715

-- Define the parametric equations of curve C1
def C1_parametric (t : ℝ) : ℝ × ℝ := (1 + t, 1 + sqrt 2 * t)

-- Define the Cartesian equation corresponding to the parametric form
def C1_cartesian (x y : ℝ) : Prop := sqrt 2 * x - y - sqrt 2 + 1 = 0

-- Define the polar coordinate equation of curve C2
def C2_polar (rho theta : ℝ) : Prop := rho * (1 - sin theta) = 1

-- Transform the polar coordinate to Cartesian coordinate for C2
def C2_cartesian (x y : ℝ) : Prop := x^2 = 2 * y + 1

-- Main theorem stating the value of |MA| * |MB| at the intersection points
theorem intersection_product : 
  let M := (1 : ℝ, 1 : ℝ),
      A := C1_parametric (-2 + sqrt 2),
      B := C1_parametric (-2 - sqrt 2) 
  in |(dist M A) * (dist M B)| = 6 :=
sorry

end intersection_product_l391_391715


namespace chests_contents_l391_391737

def ChestLabel (c : ℕ) : String := 
  if c = 1 then "Gold coins"
  else if c = 2 then "Silver coins"
  else if c = 3 then "Gold or silver coins"
  else "Invalid chest"

def CoinsInChest (c : ℕ) : String := 
  if c = 1 then "Silver coins"
  else if c = 2 then "Gold coins"
  else if c = 3 then "Copper coins"
  else "Invalid chest"

theorem chests_contents :
  ChestLabel 1 ≠ CoinsInChest 1 ∧
  ChestLabel 2 ≠ CoinsInChest 2 ∧
  ChestLabel 3 ≠ CoinsInChest 3 ∧
  (CoinsInChest 1 = "Gold coins" ∨ CoinsInChest 2 = "Gold coins" ∨ CoinsInChest 3 = "Gold coins") ∧
  (CoinsInChest 1 = "Silver coins" ∨ CoinsInChest 2 = "Silver coins" ∨ CoinsInChest 3 = "Silver coins") ∧
  (CoinsInChest 1 = "Copper coins" ∨ CoinsInChest 2 = "Copper coins" ∨ CoinsInChest 3 = "Copper coins") :=
begin
  sorry
end

end chests_contents_l391_391737


namespace apples_sold_l391_391885

theorem apples_sold (a1 a2 a3 : ℕ) (h1 : a3 = a2 / 4 + 8) (h2 : a2 = a1 / 4 + 8) (h3 : a3 = 18) : a1 = 128 :=
by
  sorry

end apples_sold_l391_391885


namespace integer_pairs_prime_P_l391_391803

theorem integer_pairs_prime_P (P : ℕ) (hP_prime : Prime P) 
  (h_condition : ∃ a b : ℤ, |a + b| + (a - b)^2 = P) : 
  P = 2 ∧ ((∃ a b : ℤ, |a + b| = 2 ∧ a - b = 0) ∨ 
           (∃ a b : ℤ, |a + b| = 1 ∧ (a - b = 1 ∨ a - b = -1))) :=
by
  sorry

end integer_pairs_prime_P_l391_391803


namespace shopkeeper_loss_l391_391961

theorem shopkeeper_loss
    (total_stock : ℝ)
    (stock_sold_profit_percent : ℝ)
    (stock_profit_percent : ℝ)
    (stock_sold_loss_percent : ℝ)
    (stock_loss_percent : ℝ) :
    total_stock = 12500 →
    stock_sold_profit_percent = 0.20 →
    stock_profit_percent = 0.10 →
    stock_sold_loss_percent = 0.80 →
    stock_loss_percent = 0.05 →
    ∃ loss_amount, loss_amount = 250 :=
by
  sorry

end shopkeeper_loss_l391_391961


namespace koschei_chests_l391_391756

theorem koschei_chests :
  ∃ (contents : Fin 3 → String), 
    -- All chests labels are incorrect
    (contents 0 ≠ "gold coins" ∧ contents 1 ≠ "silver coins" ∧ contents 2 ≠ "gold or silver coins") ∧ 
    -- Each chest contains exactly one type of coin 
    (∀ i j : Fin 3, i ≠ j → contents i ≠ contents j) ∧ 
    -- Providing the final conclusion about what is in each chest
    (contents 0 = "silver coins" ∧ contents 1 = "gold coins" ∧ contents 2 = "copper coins") :=
begin
  use (λ k, if k = 0 then "silver coins" else if k = 1 then "gold coins" else "copper coins"),
  split,
  {
    -- Proof of all labels being incorrect
    split; simp,
  },
  split,
  {
    -- Proof of each chest containing a unique type of coin
    intros i j h,
    cases i; cases j; simp [h],
  },
  {
    -- Proof of the final conclusion
    split; split; simp,
  },
end

end koschei_chests_l391_391756


namespace petya_friends_l391_391035

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l391_391035


namespace sequence_sum_l391_391627

noncomputable def a_seq (n : ℕ) : ℕ := 2 * 3^(n - 1)

def S_n (n : ℕ) : ℕ := 3^n - 1

def sum_a_seq_products (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, a_seq i * a_seq (i + 1)

theorem sequence_sum (n : ℕ) : sum_a_seq_products n = (3 / 2) * (9^n - 1) := by
  sorry

end sequence_sum_l391_391627


namespace find_C_l391_391025

-- Definitions for the points A, B, D in the complex plane
def A : ℂ := 0
def B : ℂ := 3 + 2i
def D : ℂ := 2 - 4i

-- Condition: ABCD forms a parallelogram.
def is_parallelogram (A B C D : ℂ) : Prop :=
  A + C = B + D

-- Goal: Prove the complex number corresponding to point C
theorem find_C : ∃ C : ℂ, is_parallelogram A B C D ∧ C = 5 - 2i :=
by
  -- Proof steps would go here, but are omitted
  sorry

end find_C_l391_391025


namespace susan_chairs_l391_391113

theorem susan_chairs : 
  ∀ (red yellow blue : ℕ), 
  red = 5 → 
  yellow = 4 * red → 
  blue = yellow - 2 → 
  red + yellow + blue = 43 :=
begin
  intros red yellow blue h1 h2 h3,
  rw h1,
  rw h2,
  rw h3,
  norm_num,
end

end susan_chairs_l391_391113


namespace interior_diagonal_length_l391_391149

theorem interior_diagonal_length (a b c : ℝ) (h1 : 2 * (a * b + b * c + c * a) = 26) (h2 : 4 * (a + b + c) = 28) :
  (a^2 + b^2 + c^2) = 23 :=
by
  -- We start by using the given conditions to verify the required proof
  have h_surface_area : a * b + b * c + c * a = 13,
  from (by linarith),

  have h_edge_length : a + b + c = 7,
  from (by linarith),

  -- Given a + b + c = 7 and ab + bc + ac = 13, we need to show a^2 + b^2 + c^2 = 23
  calc
    (a + b + c) ^ 2 = 49 : by norm_num [h_edge_length]
    ... = a^2 + b^2 + c^2 + 2 * (a * b + b * c + c * a) : by ring
    ... = a^2 + b^2 + c^2 + 2 * 13 : by rw [h_surface_area]
    ... = a^2 + b^2 + c^2 + 26 : by norm_num
    ... ↔ a^2 + b^2 + c^2 = 23 : by linarith


end interior_diagonal_length_l391_391149


namespace sum_geometric_series_example_l391_391460

theorem sum_geometric_series_example :
  let a := 1,
      r := 3,
      n := 8,
      series := [1, 3, 9, 27, 81, 243, 729, 2187],
      S := (a * (r^n - 1)) / (r - 1)
  in list.sum series = 3280 := 
by
  have h : S = (1 * (3^8 - 1)) / (3 - 1), by simp,
  have h2 : S = (1 * (6561 - 1)) / 2, by simp,
  have h3 : S = 6560 / 2, by simp,
  have h4 : S = 3280, by simp,
  show list.sum series = 3280, by sorry

end sum_geometric_series_example_l391_391460


namespace rightmost_three_digits_of_7_pow_2023_l391_391164

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 343 :=
sorry

end rightmost_three_digits_of_7_pow_2023_l391_391164


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391261

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391261


namespace relationship_between_abc_l391_391617

def a : ℝ := 2 / Real.log 2
def b : ℝ := 3 / Real.log 3
def c : ℝ := 7 / Real.log 7

theorem relationship_between_abc : b < a ∧ a < c := by
  sorry

end relationship_between_abc_l391_391617


namespace greatest_prime_factor_of_15_l391_391444

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391444


namespace greatest_prime_factor_15_fact_17_fact_l391_391224

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391224


namespace f_leq_2x_l391_391014

variable {α : Type*} [linear_ordered_comm_ring α] [archimedean α] [floor_ring α]

noncomputable def f : α → α := sorry

axiom f_domain (x : α) : 0 ≤ x ∧ x ≤ 1
axiom f_positive (x : α) : 0 < f x
axiom f_boundary : f 1 = 1
axiom f_superadditive (x y : α) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) :
  f (x + y) ≥ f x + f y

theorem f_leq_2x {x : α} (hx : 0 ≤ x ∧ x ≤ 1) : f x ≤ 2 * x := sorry

end f_leq_2x_l391_391014


namespace no_diff_parity_min_max_l391_391558

def A_P : Set ℤ := 
  {x | PositionProperty x}

variable (x y : ℤ) (hx : x ∈ A_P) (hy : y ∈ A_P)

theorem no_diff_parity_min_max :
  (∀ x ∈ A_P, PositionProperty x) →
  (∃ c, ∀ x ∈ A_P, x ≤ c) →
  (∀ c, (c = max_element A_P) → CriticalPointsProperty c) →
  ((min_element A_P) % 2 = (max_element A_P) % 2) :=
by
  sorry

end no_diff_parity_min_max_l391_391558


namespace other_divisor_l391_391597

theorem other_divisor (x : ℕ) (h1 : 266 % 33 = 2) (h2 : 266 % x = 2) : x = 132 :=
sorry

end other_divisor_l391_391597


namespace polynomial_addition_l391_391887

-- Define the polynomials
noncomputable def P_x : polynomial ℝ := polynomial.X ^ 2 + 3 * polynomial.X - 4
noncomputable def Q_x : polynomial ℝ := -3 * polynomial.X + 1

-- Proof statement
theorem polynomial_addition :
  P_x + Q_x = polynomial.X ^ 2 - 3 :=
by
  -- Proof steps skipped (replaced with sorry)
  sorry

end polynomial_addition_l391_391887


namespace chests_content_l391_391773

theorem chests_content (A B C : Type) (chest1 chest2 chest3 : A)
  (label1 : chest1 = "gold coins") (label2 : chest2 = "silver coins") (label3 : chest3 = "gold or silver coins")
  (gold silver copper : A)
  (h1 : chest1 ≠ gold)
  (h2 : chest2 ≠ silver)
  (h3 : chest3 ≠ gold ∧ chest3 ≠ silver)
  (t1 : chest1 = silver)
  (t2 : chest2 = gold)
  (t3 : chest3 = copper) :
  True := sorry

end chests_content_l391_391773


namespace limit_computation_l391_391982

noncomputable def limit_expression (n : ℕ) : ℝ :=
  (Real.sqrt ((n^2 + 5) * (n^4 + 2)) - Real.sqrt (n^6 - 3 * n^3 + 5)) / n

theorem limit_computation : 
  filter.tendsto (λ n : ℕ, limit_expression n) filter.at_top (nhds (5/2)) := 
sorry

end limit_computation_l391_391982


namespace find_f_l391_391131

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f {f : ℝ → ℝ} (h1 : ∀ x, x > 0 → f(x) > 0)
  (h2 : ∀ x t, x > 0 ∧ t > 0 →
    (x * f(x)) = t * (f(x) + f(t))) :
  ∃ c > 0, ∀ x, f(x) = c / x :=
begin
  sorry
end

end find_f_l391_391131


namespace leak_d_drain_time_l391_391155

def rate_A : ℝ := 1/4
def rate_B : ℝ := 1/6
def rate_C : ℝ := 1/12

def rate_pumps_without_leak : ℝ := rate_A + rate_B + rate_C -- rate of pumps without leak
def rate_pumps_with_leak : ℝ := 1 / 3 -- combined rate (with leak)

-- Find rate of leak
def rate_of_leak : ℝ := rate_pumps_without_leak - rate_pumps_with_leak

-- Find time to drain the tank by the leak
def time_to_drain (rate : ℝ) : ℝ := 1 / rate_of_leak

-- Theorem to prove the desired time for leak D to drain the tank
theorem leak_d_drain_time : time_to_drain rate_of_leak = 6 := by
  sorry

end leak_d_drain_time_l391_391155


namespace Petya_friends_l391_391082

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l391_391082


namespace proof_problem_l391_391632

noncomputable def z1 : ℂ := 1 - complex.I
noncomputable def z2 : ℂ := -2 + 3 * complex.I

theorem proof_problem :
  ¬((z1 + z2).re = -1 ∧ (z1 + z2).im = -2) ∧
  (∀ a b : ℝ, z1 * (a + complex.I) = z2 + b * complex.I → a * b = -3) ∧
  (∀ p q : ℝ, complex.root (polynomial.of_real p + polynomial.of_real q) z1 → p + q = 0) ∧
  ¬((z2 - z1).re = 3 ∧ (z2 - z1).im = -4) :=
sorry

end proof_problem_l391_391632


namespace area_of_inscribed_rectangle_l391_391974

variable (b h x : ℝ)

def is_isosceles_triangle (b h : ℝ) : Prop :=
  b > 0 ∧ h > 0

def is_inscribed_rectangle (b h x : ℝ) : Prop :=
  x > 0 ∧ x < h 

theorem area_of_inscribed_rectangle (h_pos : is_isosceles_triangle b h) 
                                    (rect_pos : is_inscribed_rectangle b h x) : 
                                    ∃ A : ℝ, A = (b / (2 * h)) * x ^ 2 :=
by
  sorry

end area_of_inscribed_rectangle_l391_391974


namespace chests_content_l391_391771

theorem chests_content (A B C : Type) (chest1 chest2 chest3 : A)
  (label1 : chest1 = "gold coins") (label2 : chest2 = "silver coins") (label3 : chest3 = "gold or silver coins")
  (gold silver copper : A)
  (h1 : chest1 ≠ gold)
  (h2 : chest2 ≠ silver)
  (h3 : chest3 ≠ gold ∧ chest3 ≠ silver)
  (t1 : chest1 = silver)
  (t2 : chest2 = gold)
  (t3 : chest3 = copper) :
  True := sorry

end chests_content_l391_391771


namespace chests_contents_l391_391795

-- Define the coin types
inductive CoinType
| Gold
| Silver
| Copper

open CoinType

-- Define the chests
def Chest : Type

-- Define the inscriptions on the chests (all of which are known to be incorrect)
def chest_1_inscription : Prop := ∀ c : Chest, c = Gold
def chest_2_inscription : Prop := ∀ c : Chest, c = Silver
def chest_3_inscription : Prop := ∀ c : Chest, c = Gold ∨ c = Silver

-- Define the truth about what the chests must contain
def contains_gold (c : Chest) : Prop
def contains_silver (c : Chest) : Prop
def contains_copper (c : Chest) : Prop

-- The conditions given in the problem, stating that the true contents are different from the inscriptions
axiom incorrect_inscriptions :
  (¬ chest_1_inscription) ∧
  (¬ chest_2_inscription) ∧
  (¬ chest_3_inscription)

-- The problem states we have exactly one chest of each type 
axiom one_gold : ∃ c : Chest, contains_gold c
axiom one_silver : ∃ c : Chest, contains_silver c
axiom one_copper : ∃ c : Chest, contains_copper c

-- The main theorem we need to prove
theorem chests_contents :
  (∀ c : Chest, contains_silver c ↔ c = Chest₁) ∧
  (∀ c : Chest, contains_gold c ↔ c = Chest₂) ∧
  (∀ c : Chest, contains_copper c ↔ c = Chest₃) :=
sorry

end chests_contents_l391_391795


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391250

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391250


namespace fare_relationship_fare_for_13_km_l391_391888

noncomputable def fare (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 3) then 8
else 1.4 * x + 3.8

theorem fare_relationship (x : ℝ) (h₁ : 0 < x) :
(fare x = if (0 < x ∧ x ≤ 3) then 8 else 1.4 * x + 3.8) :=
by {
  rw fare,
  split_ifs,
  refl,
}

theorem fare_for_13_km : fare 13 = 22 :=
by {
  rw fare,
  split_ifs,
  norm_num,
}

end fare_relationship_fare_for_13_km_l391_391888


namespace f_f_3_l391_391815

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2 / x

theorem f_f_3 : f (f 3) = 13 / 9 := by
  sorry

end f_f_3_l391_391815


namespace petya_friends_l391_391075

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l391_391075


namespace multiplication_pyramid_proof_l391_391712

variable {x : ℝ}
variables (h_x_ne_zero : x ≠ 0) 

theorem multiplication_pyramid_proof :
  (3 * x * (1 / x^2) = 3 * x^(-1)) ∧
  (3 * x^(-1) * ((x + 2) / (3 * x^(-1))) = x + 2) ∧
  (3 * x * (x + 2) = 3 * x^2 + 6 * x) ∧
  ((9 * x^4 - 36 * x^2) / (3 * x^2 + 6 * x) = 3 * x^2 - 6 * x) ∧
  ((3 * x * (x-2)) / (3 * x) = x - 2) ∧
  ((x - 2) * x^2 = x^3 - 2 * x^2) :=
by {
  intros,
  sorry
}

end multiplication_pyramid_proof_l391_391712


namespace number_of_bead_necklaces_sold_is_3_l391_391724

-- Definitions of the given conditions
def total_earnings : ℕ := 36
def gemstone_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 6

-- Define the earnings from gemstone necklaces as a separate definition
def earnings_gemstone_necklaces : ℕ := gemstone_necklaces * cost_per_necklace

-- Define the earnings from bead necklaces based on total earnings and earnings from gemstone necklaces
def earnings_bead_necklaces : ℕ := total_earnings - earnings_gemstone_necklaces

-- Define the number of bead necklaces sold
def bead_necklaces_sold : ℕ := earnings_bead_necklaces / cost_per_necklace

-- The theorem we want to prove
theorem number_of_bead_necklaces_sold_is_3 : bead_necklaces_sold = 3 :=
by
  sorry

end number_of_bead_necklaces_sold_is_3_l391_391724


namespace evaluate_expression_l391_391589

theorem evaluate_expression : ((5^2 + 3)^2 - (5^2 - 3)^2)^3 = 27000000 :=
by
  sorry

end evaluate_expression_l391_391589


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391244

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391244


namespace inequality_proof_l391_391616

variable {a b : ℝ}

theorem inequality_proof (h : a > b) : 2 - a < 2 - b :=
by
  sorry

end inequality_proof_l391_391616


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391382

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391382


namespace terminal_side_of_minus_685_in_first_quadrant_l391_391889

/-- The statement to be proven: the terminal side of angle -685° falls in the first quadrant. -/
theorem terminal_side_of_minus_685_in_first_quadrant :
  let angle := -685 in
  let reduced_angle := angle % 360 in 
  0 < reduced_angle ∧ reduced_angle < 90 :=
by
  sorry

end terminal_side_of_minus_685_in_first_quadrant_l391_391889


namespace find_E_l391_391499

variable (A H C S M N E : ℕ)
variable (x y z l : ℕ)

theorem find_E (h1 : A * x + H * y + C * z = l)
 (h2 : S * x + M * y + N * z = l)
 (h3 : E * x = l)
 (h4 : A ≠ S ∧ A ≠ H ∧ A ≠ C ∧ A ≠ M ∧ A ≠ N ∧ A ≠ E ∧ H ≠ C ∧ H ≠ M ∧ H ≠ N ∧ H ≠ E ∧ C ≠ M ∧ C ≠ N ∧ C ≠ E ∧ M ≠ N ∧ M ≠ E ∧ N ≠ E)
 : E = (A * M + C * N - S * H - N * H) / (M + N - H) := 
sorry

end find_E_l391_391499


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391366

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391366


namespace triangle_area_solution_l391_391124

noncomputable def triangle_area_equation : ℝ :=
  let f : Polynomial ℝ := Polynomial.C (-8 / 5) + Polynomial.X + Polynomial.C 4 * Polynomial.X - Polynomial.C 3 * Polynomial.X^2 + Polynomial.X^3
  let roots : Finset ℝ := f.roots
  let a := roots.toList[0]
  let b := roots.toList[1]
  let c := roots.toList[2]
  let q := (a + b + c) / 2
  let heron (a b c : ℝ) (q : ℝ) := q * (q - a) * (q - b) * (q - c)
  if heron a b c q < 0 then 0 else (heron a b c q).sqrt

theorem triangle_area_solution:
  triangle_area_equation x = Real.sqrt (6) / 4 := sorry

end triangle_area_solution_l391_391124


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391238

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391238


namespace max_height_of_A_l391_391509

theorem max_height_of_A 
(V1 V2 : ℝ) (β α g : ℝ) (nonneg_g : 0 < g) :
  let Vc0y := (V1 * Real.sin β + V2 * Real.sin α) / 2 in
  (Vc0y * Vc0y) / (2 * g) =
  (1 / (2 * g)) * (V1 * Real.sin β + V2 * Real.sin α)^2 / 4 :=
by
  let Vc0y := (V1 * Real.sin β + V2 * Real.sin α) / 2
  calc
    (Vc0y * Vc0y) / (2 * g)
      = (Vc0y^2) / (2 * g) : by simp
  ... = Vc0y^2 / 2 / g : by rw [div_div_eq_div_mul]
  ... = (1 / (2 * g)) * Vc0y^2 : by ring
  ... = (1 / (2 * g)) * ((V1 * Real.sin β + V2 * Real.sin α) / 2)^2 : by simp only [Vc0y]
  ... = (1 / (2 * g)) * (V1 * Real.sin β + V2 * Real.sin α)^2 / 4 : by field_simp
  ... = (1 / (2 * g)) * (V1 * Real.sin β + V2 * Real.sin α)^2 / 4 : by ring
  ... = (1 / (2 * g)) * ((V1 * Real.sin β + V2 * Real.sin α) / 2)^2 : by field_simp
  ... = (1 / (2 * g)) * (V1 * Real.sin β / 2 + V2 * Real.sin α / 2)^2 : by ring
  ... = (1 / (2 * g)) * ((V1 * Real.sin β + V2 * Real.sin α) / 2 * (V1 * Real.sin β + V2 * Real.sin α) / 2) : by ring

end max_height_of_A_l391_391509


namespace greatest_prime_factor_15_17_factorial_l391_391349

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391349


namespace greatest_prime_factor_15_17_factorial_l391_391211

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391211


namespace smallest_sum_l391_391865

theorem smallest_sum (p q r s : ℕ) (h1 : p > 0) (h2 : q > 0) (h3 : r > 0) (h4 : s > 0) (h5 : p * q * r * s = 12!) :
  p + q + r + s = 777 :=
sorry

end smallest_sum_l391_391865


namespace inscribe_rectangle_in_quadrilateral_l391_391495

-- Define the quadrilateral and the points E and F
noncomputable def quadrilateral (A B C D E F: Type) [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D] [decidable_eq E] [decidable_eq F] :
  Prop :=
  ∃ P Q R S, 
    (P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ S ≠ P) ∧ -- vertices of the rectangle
    (P ∈ line A B ∧ R ∈ line C D ∧ -- P and R are points on lines AB and CD
    Q ∈ line B C ∧ S ∈ line A C) ∧ -- Q and S are points on lines BC and AC
    (∃ BF CE, parallel BF CE) -- lines BF and CE parallel to given directions

-- Statement: Given the conditions, prove that there exists a rectangle inscribed in quadrilateral ABCD.
theorem inscribe_rectangle_in_quadrilateral (A B C D E F : Point) (BF CE : Line) (direction_condition : ∃ BF CE, parallel BF CE) :
  quadrilateral A B C D E F → 
  ∃ PQRS, inscribe_rectangle ABCD PQRS direction_condition :=
begin
  sorry
end

end inscribe_rectangle_in_quadrilateral_l391_391495


namespace petya_friends_l391_391089

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l391_391089


namespace scientific_notation_equivalence_l391_391872

-- Define constants and variables
def scientific_notation {a b : ℝ} (n : ℝ) (a b : ℝ) := n = a * (10^b)

-- State the conditions
def seven_nm_equals := (7 : ℝ) * (10 : ℝ) ^ (-9 : ℝ) = 0.000000007

-- Theorem to prove
theorem scientific_notation_equivalence : scientific_notation 0.000000007 7 (-9) :=
by
  apply (seven_nm_equals)

end scientific_notation_equivalence_l391_391872


namespace find_k_l391_391938

theorem find_k (k : ℕ) : 5 ^ k = 5 * 25^2 * 125^3 → k = 14 := by
  sorry

end find_k_l391_391938


namespace no_diff_parity_min_max_l391_391555

def A_P : Set ℤ := 
  {x | PositionProperty x}

variable (x y : ℤ) (hx : x ∈ A_P) (hy : y ∈ A_P)

theorem no_diff_parity_min_max :
  (∀ x ∈ A_P, PositionProperty x) →
  (∃ c, ∀ x ∈ A_P, x ≤ c) →
  (∀ c, (c = max_element A_P) → CriticalPointsProperty c) →
  ((min_element A_P) % 2 = (max_element A_P) % 2) :=
by
  sorry

end no_diff_parity_min_max_l391_391555


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391428

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391428


namespace matrix_symmetric_square_identity_l391_391108

theorem matrix_symmetric_square_identity
  (x y z : ℝ)
  (B : Matrix (Fin 2) (Fin 2) ℝ)
  (hB : B = !![x, y; y, z])
  (hB_symm : Bᵀ = B)
  (hB_sq : B ⬝ B = 1) :
  x^2 + 2 * y^2 + z^2 = 2 := by
    sorry

end matrix_symmetric_square_identity_l391_391108


namespace greatest_prime_factor_of_15_l391_391443

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391443


namespace max_notebooks_15_dollars_l391_391518

noncomputable def max_notebooks (money : ℕ) : ℕ :=
  let cost_individual   := 2
  let cost_pack_4       := 6
  let cost_pack_7       := 9
  let notebooks_budget  := 15
  if money >= 9 then 
    7 + max_notebooks (money - 9)
  else if money >= 6 then 
    4 + max_notebooks (money - 6)
  else 
    money / 2

theorem max_notebooks_15_dollars : max_notebooks 15 = 11 :=
by
  sorry

end max_notebooks_15_dollars_l391_391518


namespace general_term_formula_sum_sequence_na_n_l391_391805

variables {S : ℕ → ℕ} {a : ℕ → ℕ}

-- Define the condition
def condition (S a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → 2 * a n - 2 = S n

-- Define the nth term
def nth_term (a : ℕ → ℕ) : ℕ → ℕ := λ n : ℕ, 2^n

-- Define the sum of first n terms of sequence (na_n)
def sum_first_n_terms (a : ℕ → ℕ) : ℕ → ℕ :=
  λ n, (n - 1) * 2^(n + 1) + 2

/-- Problem (I): The general term of the sequence a_n is 2^n -/
theorem general_term_formula (h : condition S a) : ∀ n, a n = nth_term a n :=
begin
  sorry
end

/-- Problem (II): The sum of the first n terms of the sequence (na_n) is (n-1) * 2^(n+1) + 2 -/
theorem sum_sequence_na_n (h : condition S a) : ∀ n, sum_first_n_terms a n = (n-1) * 2^(n+1) + 2 :=
begin
  sorry
end

end general_term_formula_sum_sequence_na_n_l391_391805


namespace find_k_l391_391701

theorem find_k (x y k : ℝ) (h1 : x + y = 5 * k) (h2 : x - 2 * y = -k) (h3 : 2 * x - y = 8) : k = 2 :=
by
  sorry

end find_k_l391_391701


namespace circle_area_l391_391869

-- Define the condition in the problem
def polar_circle (θ : ℝ) : ℝ := 4 * Real.cos θ - 3 * Real.sin θ

-- The statement we aim to prove
theorem circle_area : (∀ θ : ℝ, ∃ r : ℝ, r = polar_circle θ) →
    ∃ A : ℝ, A = (25 * Real.pi) / 4 :=
by
  sorry

end circle_area_l391_391869


namespace symmetric_point_line_l391_391119

theorem symmetric_point_line (a b : ℝ) :
  (∀ (x y : ℝ), (y - 2) / (x - 1) = -2 → (x + 1)/2 + 2 * (y + 2)/2 - 10 = 0) →
  a = 3 ∧ b = 6 := by
  intro h
  sorry

end symmetric_point_line_l391_391119


namespace sufficient_but_not_necessary_l391_391652

def f (x ϕ : ℝ) : ℝ := 2 * Real.sin (x + π / 3 + ϕ)

theorem sufficient_but_not_necessary (φ : ℝ) :
  (∀ x, f x (2 * π / 3) = -f (-x) (2 * π / 3)) ∧
  ¬ (∀ x, f x φ = -f (-x) φ → φ = 2 * π / 3) :=
by
  sorry

end sufficient_but_not_necessary_l391_391652


namespace perpendicular_points_same_circle_l391_391630

-- Definitions based on conditions
structure IsoscelesTrapezoid (A B C D : Type) :=
  (parallel_AD_BC : AD ∥ BC)

variable {A B C D M K L N S : Type} [IsoscelesTrapezoid A B C D]

-- The proof problem statement
theorem perpendicular_points_same_circle (AD_parallel_BC : AD ∥ BC) (M_on_arc_AD : Point_on_arc M AD)
  (K_perp : perpendicular_at K A BM) (L_perp : perpendicular_at L A BM)
  (N_perp : perpendicular_at N D CM) (S_perp : perpendicular_at S D CM) :
  Points_on_same_circle K L N S :=
sorry

end perpendicular_points_same_circle_l391_391630


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391364

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391364


namespace soccer_field_solution_l391_391542

noncomputable def soccer_field_problem : Prop :=
  ∃ (a b c d : ℝ), 
    (abs (a - b) = 1 ∨ abs (a - b) = 2 ∨ abs (a - b) = 3 ∨ abs (a - b) = 4 ∨ abs (a - b) = 5 ∨ abs (a - b) = 6) ∧
    (abs (a - c) = 1 ∨ abs (a - c) = 2 ∨ abs (a - c) = 3 ∨ abs (a - c) = 4 ∨ abs (a - c) = 5 ∨ abs (a - c) = 6) ∧
    (abs (a - d) = 1 ∨ abs (a - d) = 2 ∨ abs (a - d) = 3 ∨ abs (a - d) = 4 ∨ abs (a - d) = 5 ∨ abs (a - d) = 6) ∧
    (abs (b - c) = 1 ∨ abs (b - c) = 2 ∨ abs (b - c) = 3 ∨ abs (b - c) = 4 ∨ abs (b - c) = 5 ∨ abs (b - c) = 6) ∧
    (abs (b - d) = 1 ∨ abs (b - d) = 2 ∨ abs (b - d) = 3 ∨ abs (b - d) = 4 ∨ abs (b - d) = 5 ∨ abs (b - d) = 6) ∧
    (abs (c - d) = 1 ∨ abs (c - d) = 2 ∨ abs (c - d) = 3 ∨ abs (c - d) = 4 ∨ abs (c - d) = 5 ∨ abs (c - d) = 6)

theorem soccer_field_solution : soccer_field_problem :=
  sorry

end soccer_field_solution_l391_391542


namespace chests_content_l391_391770

theorem chests_content (A B C : Type) (chest1 chest2 chest3 : A)
  (label1 : chest1 = "gold coins") (label2 : chest2 = "silver coins") (label3 : chest3 = "gold or silver coins")
  (gold silver copper : A)
  (h1 : chest1 ≠ gold)
  (h2 : chest2 ≠ silver)
  (h3 : chest3 ≠ gold ∧ chest3 ≠ silver)
  (t1 : chest1 = silver)
  (t2 : chest2 = gold)
  (t3 : chest3 = copper) :
  True := sorry

end chests_content_l391_391770


namespace correct_propositions_l391_391516

-- Definitions based on the conditions
def proposition_1 : Prop := ∀ r, (r > 0) → (r implies strong linear correlation)
def proposition_2 : Prop := ∀ residuals, (sum_of_squared_residuals residuals < ε) → (better_fitting_model residuals)
def proposition_3 : Prop := ∀ R2, (R2 < ε) → (better_model_fitting R2)
def proposition_4 : Prop := ∀ e, (random_error e) → (E(e) = 0)

-- Original Lean statement formulation based on the proof problem
theorem correct_propositions : proposition_2 ∧ proposition_4 :=
by
  sorry -- proof is omitted

end correct_propositions_l391_391516


namespace land_occupation_tax_rate_range_l391_391487

-- Definitions of conditions
def farmland_loss (t : ℝ) : ℝ := 20 - (5 / 2) * t
def value_per_acre : ℝ := 24000
def tax_revenue (t : ℝ) : ℝ := farmland_loss t * value_per_acre * (t / 100)

theorem land_occupation_tax_rate_range (t : ℝ) :
  (20 - (5 / 2) * t) * 24000 * (t / 100) ≥ 9000 ↔ 3 ≤ t ∧ t ≤ 5 :=
by
  apply sorry

end land_occupation_tax_rate_range_l391_391487


namespace greatest_prime_factor_15_17_factorial_l391_391205

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391205


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391253

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391253


namespace greatest_prime_factor_of_factorial_sum_l391_391329

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391329


namespace knicks_equiv_knocks_l391_391690

theorem knicks_equiv_knocks :
  ∀ (knicks knacks knocks : Type)
    (nine_knicks_eq_three_knacks : 9 * knicks = 3 * knacks)
    (two_knacks_eq_five_knocks : 2 * knacks = 5 * knocks),
    40 * knocks = 48 * knicks :=
by {
  intro knicks knacks knocks nine_knicks_eq_three_knacks two_knacks_eq_five_knocks,
  sorry
}

end knicks_equiv_knocks_l391_391690


namespace greatest_prime_factor_of_sum_factorials_l391_391279

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391279


namespace logan_buys_15_pounds_of_corn_l391_391991

theorem logan_buys_15_pounds_of_corn (c b : ℝ) 
    (h1 : 1.20 * c + 0.60 * b = 27) 
    (h2 : b + c = 30) : 
    c = 15.0 :=
by
  sorry

end logan_buys_15_pounds_of_corn_l391_391991


namespace greatest_prime_factor_15_17_factorial_l391_391208

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391208


namespace A_union_B_when_m_neg_half_B_subset_A_implies_m_geq_zero_l391_391668

def A : Set ℝ := { x | x^2 + x - 2 < 0 }
def B (m : ℝ) : Set ℝ := { x | 2 * m < x ∧ x < 1 - m }

theorem A_union_B_when_m_neg_half : A ∪ B (-1/2) = { x | -2 < x ∧ x < 3/2 } :=
by
  sorry

theorem B_subset_A_implies_m_geq_zero (m : ℝ) : B m ⊆ A → 0 ≤ m :=
by
  sorry

end A_union_B_when_m_neg_half_B_subset_A_implies_m_geq_zero_l391_391668


namespace chests_content_l391_391769

theorem chests_content (A B C : Type) (chest1 chest2 chest3 : A)
  (label1 : chest1 = "gold coins") (label2 : chest2 = "silver coins") (label3 : chest3 = "gold or silver coins")
  (gold silver copper : A)
  (h1 : chest1 ≠ gold)
  (h2 : chest2 ≠ silver)
  (h3 : chest3 ≠ gold ∧ chest3 ≠ silver)
  (t1 : chest1 = silver)
  (t2 : chest2 = gold)
  (t3 : chest3 = copper) :
  True := sorry

end chests_content_l391_391769


namespace greatest_prime_factor_15f_plus_17f_l391_391196

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391196


namespace greatest_prime_factor_of_15_l391_391442

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391442


namespace chests_content_l391_391744

-- Define the chests and their labels
inductive CoinContent where
  | gold : CoinContent
  | silver : CoinContent
  | copper : CoinContent

structure Chest where
  label : CoinContent
  contents : CoinContent

-- Given conditions and incorrect labels
def chest1 : Chest := { label := CoinContent.gold, contents := sorry }
def chest2 : Chest := { label := CoinContent.silver, contents := sorry }
def chest3 : Chest := { label := CoinContent.gold, contents := sorry }

-- The proof problem
theorem chests_content :
  chest1.contents ≠ CoinContent.gold ∧
  chest2.contents ≠ CoinContent.silver ∧
  chest3.contents ≠ CoinContent.gold ∨ chest3.contents ≠ CoinContent.silver →
  chest1.contents = CoinContent.silver ∧
  chest2.contents = CoinContent.gold ∧
  chest3.contents = CoinContent.copper := by
  sorry

end chests_content_l391_391744


namespace greatest_prime_factor_15_fact_17_fact_l391_391171

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391171


namespace mean_weight_BC_correct_l391_391939

variables (A B C : Type) (A_n B_n C_n : ℕ)
          (W_A W_B W_C : ℕ)

noncomputable def mean_weight (total_weight : ℕ) (total_number : ℕ) : ℕ := 
  total_weight / total_number

def mean_weight_A : Prop := mean_weight W_A A_n = 30
def mean_weight_B : Prop := mean_weight W_B B_n = 70
def mean_weight_AB : Prop := mean_weight (W_A + W_B) (A_n + B_n) = 50
def mean_weight_AC : Prop := mean_weight (W_A + W_C) (A_n + C_n) = 40

def mean_weight_BC (k n : ℕ) := mean_weight (70 * k + 10 * k + 40 * n) (k + n)
def correct_answer_BC := 80

theorem mean_weight_BC_correct
  (A_n B_n C_n : ℕ)
  (W_A W_B W_C : ℕ)
  (hnA : mean_weight W_A A_n = 30)
  (hnB : mean_weight W_B B_n = 70)
  (hnAB : mean_weight (W_A + W_B) (A_n + B_n) = 50)
  (hn_AC : mean_weight (W_A + W_C) (A_n + C_n) = 40) :
  mean_weight_BC A_n C_n = correct_answer_BC :=
sorry

end mean_weight_BC_correct_l391_391939


namespace greatest_prime_factor_15_fact_17_fact_l391_391183

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391183


namespace semicircle_inequality_l391_391100

open Real

theorem semicircle_inequality {A B C D E : ℝ} (h : A^2 + B^2 + C^2 + D^2 + E^2 = 1):
  (A - B)^2 + (B - C)^2 + (C - D)^2 + (D - E)^2 + (A - B) * (B - C) * (C - D) + (B - C) * (C - D) * (D - E) < 4 :=
by
  -- proof omitted
  sorry

end semicircle_inequality_l391_391100


namespace greatest_prime_factor_15_fact_17_fact_l391_391223

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391223


namespace chest_contents_l391_391780

def Chest : Type := ℕ
def Coins : Type := ℕ

variable (empty : Coins → Coins → Prop)

noncomputable def Chest1 : Chest := 1
noncomputable def Chest2 : Chest := 2
noncomputable def Chest3 : Chest := 3

variable (goldCoins : Coins)
variable (silverCoins : Coins)
variable (copperCoins : Coins)

-- Conditions
variable (labelsIncorrect : Chest → Coins → Prop)

axiom label1 : labelsIncorrect Chest1 goldCoins
axiom label2 : labelsIncorrect Chest2 silverCoins
axiom label3 : labelsIncorrect Chest3 (goldCoins ∨ silverCoins)

axiom uniqueContents : ∀ c : Chest, c = Chest1 ∨ c = Chest2 ∨ c = Chest3
axiom distinctContents : ∀ c1 c2 : Chest, c1 ≠ c2 → (c1 = Chest1 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest3) ∨ (c1 = Chest2 ∧ c2 ≠ Chest2 ∧ c2 ≠ Chest3) ∨ (c1 = Chest3 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest2)

theorem chest_contents : (exists! c1 : Coins, c1 = silverCoins ∧ labelsIncorrect Chest1 c1) ∧
                         (exists! c2 : Coins, c2 = goldCoins ∧ labelsIncorrect Chest2 c2) ∧
                         (exists! c3 : Coins, c3 = copperCoins ∧ labelsIncorrect Chest3 c3)
                         :=
by
  sorry

end chest_contents_l391_391780


namespace greatest_prime_factor_of_sum_l391_391397

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391397


namespace identical_lines_have_two_pairs_l391_391006

-- Definitions based on the conditions of the problem.
def first_line (a d : ℝ) : ℝ × ℝ → Prop := 
  λ p, 4 * p.1 + a * p.2 + d = 0

def second_line (d : ℝ) : ℝ × ℝ → Prop := 
  λ p, d * p.1 - 3 * p.2 + 18 = 0

-- Theorem stating the number of (a, d) pairs such that the lines are identical.
theorem identical_lines_have_two_pairs : 
  ∃ n : ℕ, (∀ a d : ℝ, (∀ p : ℝ × ℝ, first_line a d p ↔ second_line d p) → n = 2) :=
sorry

end identical_lines_have_two_pairs_l391_391006


namespace pyramid_new_volume_l391_391955

theorem pyramid_new_volume (V_original : ℕ) (s_scale_factor h_scale_factor : ℕ) :
  V_original = 60 →
  s_scale_factor = 3 →
  h_scale_factor = 2 →
  let V_new := V_original * s_scale_factor^2 * h_scale_factor in
  V_new = 1080 :=
by
  intros hV hs hh
  simp [V_original, s_scale_factor, h_scale_factor] at hV hs hh
  let V_new := 60 * 3^2 * 2
  have h_calc : V_new = 1080 := by norm_num
  exact h_calc

end pyramid_new_volume_l391_391955


namespace valid_pairs_l391_391116

def valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def valid_number (n : ℕ) : Prop :=
  let digits := [5, 3, 2, 9, n / 10 % 10, n % 10]
  (n % 2 = 0) ∧ (digits.sum % 3 = 0)

theorem valid_pairs (d₀ d₁ : ℕ) :
  valid_digit d₀ →
  valid_digit d₁ →
  (d₀ % 2 = 0) →
  valid_number (53290 * 10 + d₀ * 10 + d₁) →
  (d₀, d₁) = (0, 3) ∨ (d₀, d₁) = (2, 0) ∨ (d₀, d₁) = (2, 3) ∨ (d₀, d₁) = (2, 6) ∨
  (d₀, d₁) = (2, 9) ∨ (d₀, d₁) = (4, 1) ∨ (d₀, d₁) = (4, 4) ∨ (d₀, d₁) = (4, 7) ∨
  (d₀, d₁) = (6, 2) ∨ (d₀, d₁) = (6, 5) ∨ (d₀, d₁) = (6, 8) ∨ (d₀, d₁) = (8, 0) :=
by sorry

end valid_pairs_l391_391116


namespace heights_of_triangle_are_different_l391_391132

theorem heights_of_triangle_are_different (Δ : Triangle) :
  (¬ is_equilateral Δ) → (∀ h₁ h₂ h₃ : Height Δ, h₁ ≠ h₂ ∧ h₂ ≠ h₃ ∧ h₃ ≠ h₁) :=
sorry

end heights_of_triangle_are_different_l391_391132


namespace range_of_m_l391_391624

noncomputable def quadratic_function {a : ℝ} (x : ℝ) : ℝ := x^2 + a * x + 5

theorem range_of_m (a : ℝ) (m : ℝ) :
  (∀ t : ℝ, quadratic_function (a := a) t = quadratic_function (a := a) (-4 - t)) →
  (∀ x ∈ set.Icc m 0, 1 ≤ quadratic_function (a := a) x ∧ quadratic_function (a := a) x ≤ 5) →
  (-4 ≤ m ∧ m ≤ -2) :=
by
  sorry

end range_of_m_l391_391624


namespace possible_values_of_P_l391_391816

-- Definition of the conditions
variables (x y : ℕ) (h1 : x < y) (h2 : (x > 0)) (h3 : (y > 0))

-- Definition of P
def P : ℤ := (x^3 - y) / (1 + x * y)

-- Theorem statement
theorem possible_values_of_P : (P = 0) ∨ (P ≥ 2) :=
sorry

end possible_values_of_P_l391_391816


namespace trigonometric_simplification_l391_391922

noncomputable def tan : ℝ → ℝ := λ x => Real.sin x / Real.cos x
noncomputable def simp_expr : ℝ :=
  (tan (96 * Real.pi / 180) - tan (12 * Real.pi / 180) * (1 + 1 / Real.sin (6 * Real.pi / 180)))
  /
  (1 + tan (96 * Real.pi / 180) * tan (12 * Real.pi / 180) * (1 + 1 / Real.sin (6 * Real.pi / 180)))

theorem trigonometric_simplification : simp_expr = Real.sqrt 3 / 3 :=
by
  sorry

end trigonometric_simplification_l391_391922


namespace parity_uniformity_l391_391550

-- Define the set A_P
variable {A_P : Set ℤ}

-- Conditions:
-- 1. A_P is non-empty
noncomputable def non_empty (H : A_P ≠ ∅) := H

-- 2. c is the maximum element in A_P
variable {c : ℤ}
variable (H_max : ∀ a ∈ A_P, a ≤ c)

-- 3. Consideration of critical points around c
variable {f : ℤ → ℤ}
variable (H_critical : ∀ x ∈ A_P, f x = 0)

-- 4. Parity of the smallest and largest elements
def parity (n : ℤ) : Prop := n % 2 = 0 ∨ n % 2 = 1

-- Proof statement
theorem parity_uniformity (H_non_empty : non_empty A_P)
  (H_max_element : ∀ a ∈ A_P, a ≤ c)
  (H_critical_points : ∀ x ∈ A_P, f x = 0) :
  (∃ x ∈ A_P, ∀ y ∈ A_P, x ≤ y) → (parity (x : ℤ) = parity ((y : ℤ) : ℤ)) → (least x ∈ A_P, greatest y ∈ A_P, parity x = parity y) :=
by
  sorry

end parity_uniformity_l391_391550


namespace original_speed_proof_l391_391485

def bullet_train_original_speed (D : ℝ) (V_original : ℝ) : Prop :=
  V_original = 48

theorem original_speed_proof (D : ℝ) (h1 : D = 60 * (40 / 60)) 
  (h2 : V_original = D / (50 / 60)) : bullet_train_original_speed D V_original :=
by {
  rw [h1, h2],
  have d_value : D = 40,
  { calc
      D = 60 * (40 / 60) : by assumption
      ... = 40 : by norm_num },
  have v_value : V_original = 40 / (50 / 60),
  { calc
      V_original = D / (50 / 60) : by assumption
      ... = 40 / (50 / 60) : by rw d_value },
  calc 
    V_original = 40 / (50 / 60) : by rw v_value
    ... = 48 : by norm_num 
}

end original_speed_proof_l391_391485


namespace problem1_problem2_l391_391642

-- The conditions
variables (a b : ℝ)
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : √a + √b = 2

-- Goal 1: Prove that a√b + b√a ≤ 2
theorem problem1 (h1 : a > 0) (h2 : b > 0) (h3 : √a + √b = 2) : a * √b + b * √a ≤ 2 := by
  sorry

-- Goal 2: Prove that 2 ≤ a^2 + b^2 < 16
theorem problem2 (h1 : a > 0) (h2 : b > 0) (h3 : √a + √b = 2) : 2 ≤ a^2 + b^2 ∧ a^2 + b^2 < 16 := by
  sorry

end problem1_problem2_l391_391642


namespace option_B_more_cost_effective_l391_391703

def cost_option_A (x : ℕ) : ℕ := 60 + 18 * x
def cost_option_B (x : ℕ) : ℕ := 150 + 15 * x
def x : ℕ := 40

theorem option_B_more_cost_effective : cost_option_B x < cost_option_A x := by
  -- Placeholder for the proof steps
  sorry

end option_B_more_cost_effective_l391_391703


namespace fraction_bounds_l391_391998

theorem fraction_bounds (x y : ℝ) (h : x^2 * y^2 + x * y + 1 = 3 * y^2) : 
0 ≤ (y - x) / (x + 4 * y) ∧ (y - x) / (x + 4 * y) ≤ 4 := 
sorry

end fraction_bounds_l391_391998


namespace no_diff_parity_min_max_l391_391557

def A_P : Set ℤ := 
  {x | PositionProperty x}

variable (x y : ℤ) (hx : x ∈ A_P) (hy : y ∈ A_P)

theorem no_diff_parity_min_max :
  (∀ x ∈ A_P, PositionProperty x) →
  (∃ c, ∀ x ∈ A_P, x ≤ c) →
  (∀ c, (c = max_element A_P) → CriticalPointsProperty c) →
  ((min_element A_P) % 2 = (max_element A_P) % 2) :=
by
  sorry

end no_diff_parity_min_max_l391_391557


namespace petya_has_19_friends_l391_391055

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l391_391055


namespace greatest_prime_factor_of_sum_factorials_l391_391272

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391272


namespace area_tr_l391_391721

-- define the structure of the triangle and midpoints
variables {D E F G I H O J: Type}
variables [IsPoint D] [IsPoint E] [IsPoint F] [IsPoint G] [IsPoint I] [IsPoint H] [IsPoint O] [IsPoint J]

-- define medians and centroid
variable (triangle_DEF : Triangle D E F)
variable (median_DG : Median D G)
variable (median_EI : Median E I)
variable (centroid_O : Centroid O triangle_DEF median_DG median_EI)

-- define midpoints based on the conditions
variable (midpoint_G : Midpoint G E F)
variable (midpoint_I : Midpoint I D F)
variable (midpoint_H : Midpoint H D E)

-- define intersections
variable (intersection_J : Intersection (Line G H) (Median E I) J)

-- given area of triangle OJG
variable (area_OJG : ℝ) [non_neg area_OJG]

-- statement of the proof problem
theorem area_tr DEF_is_8m 
  (h1: triangle DEF)
  (h2: Medians_Intersect_At_Centroid DEF DG median_DG EI median_EI O centroid_O)
  (h3: Midpoint DEF G EF midpoint_G)
  (h4: Midpoint DEF I DF midpoint_I)
  (h5: Midpoint DEF H DE midpoint_H)
  (h6: IntersectionLineGH MedianEI J)
  (h7: Area triangle_OJG = m) 
      : Area triangle_DEF = 8 * m :=
sorry

end area_tr_l391_391721


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391372

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391372


namespace part1_part2_l391_391024

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Given conditions
axiom a_3a_5 : a 3 * a 5 = 63
axiom a_2a_6 : a 2 + a 6 = 16

-- Part (1) Proving the general formula
theorem part1 : 
  (∀ n : ℕ, a n = 12 - n) :=
sorry

-- Part (2) Proving the maximum value of S_n
theorem part2 :
  (∃ n : ℕ, (S n = (n * (12 - (n - 1) / 2)) → (n = 11 ∨ n = 12) ∧ (S n = 66))) :=
sorry

end part1_part2_l391_391024


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391248

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391248


namespace soccer_players_positions_l391_391538

theorem soccer_players_positions :
  ∃ (a b c d : ℝ), a = 0 ∧ b = 1 ∧ c = 4 ∧ d = 6 ∧
  set_of (λ x, ∃ i j, i ≠ j ∧ x = abs (a - b) ∨ x = abs (a - c) ∨ x = abs (a - d) ∨ x = abs (b - c) ∨ x = abs (b - d) ∨ x = abs (c - d)) = {1, 2, 3, 4, 5, 6} :=
by
  use 0, 1, 4, 6
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  rw [set_of, abs]
  sorry

end soccer_players_positions_l391_391538


namespace greatest_prime_factor_of_15_l391_391441

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391441


namespace fraction_N_div_M_l391_391000

def M : ℕ := Nat.lcm_list [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

def N : ℕ := Nat.lcm M (Nat.lcm_list [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45])

theorem fraction_N_div_M : N / M = 2^2 * 3 * 29 * 31 * 37 * 41 := 
 by sorry

end fraction_N_div_M_l391_391000


namespace s_of_1_l391_391007

def t (x : ℚ) : ℚ := 5 * x - 10
def s (y : ℚ) : ℚ := (y^2 / (5^2)) + (5 * y / 5) + 6  -- reformulated to fit conditions

theorem s_of_1 :
  s (1 : ℚ) = 546 / 25 := by
  sorry

end s_of_1_l391_391007


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391419

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391419


namespace chest_contents_l391_391781

def Chest : Type := ℕ
def Coins : Type := ℕ

variable (empty : Coins → Coins → Prop)

noncomputable def Chest1 : Chest := 1
noncomputable def Chest2 : Chest := 2
noncomputable def Chest3 : Chest := 3

variable (goldCoins : Coins)
variable (silverCoins : Coins)
variable (copperCoins : Coins)

-- Conditions
variable (labelsIncorrect : Chest → Coins → Prop)

axiom label1 : labelsIncorrect Chest1 goldCoins
axiom label2 : labelsIncorrect Chest2 silverCoins
axiom label3 : labelsIncorrect Chest3 (goldCoins ∨ silverCoins)

axiom uniqueContents : ∀ c : Chest, c = Chest1 ∨ c = Chest2 ∨ c = Chest3
axiom distinctContents : ∀ c1 c2 : Chest, c1 ≠ c2 → (c1 = Chest1 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest3) ∨ (c1 = Chest2 ∧ c2 ≠ Chest2 ∧ c2 ≠ Chest3) ∨ (c1 = Chest3 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest2)

theorem chest_contents : (exists! c1 : Coins, c1 = silverCoins ∧ labelsIncorrect Chest1 c1) ∧
                         (exists! c2 : Coins, c2 = goldCoins ∧ labelsIncorrect Chest2 c2) ∧
                         (exists! c3 : Coins, c3 = copperCoins ∧ labelsIncorrect Chest3 c3)
                         :=
by
  sorry

end chest_contents_l391_391781


namespace not_age_of_child_l391_391027

theorem not_age_of_child (ages : Set ℕ) (h_ages : ∀ x ∈ ages, 4 ≤ x ∧ x ≤ 10) : 
  5 ∉ ages := by
  let number := 1122
  have h_number : number % 5 ≠ 0 := by decide
  have h_divisible : ∀ x ∈ ages, number % x = 0 := sorry
  exact sorry

end not_age_of_child_l391_391027


namespace greatest_prime_factor_15_17_factorial_l391_391342

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391342


namespace equation_solution_l391_391849

noncomputable def solveEquation (x : ℂ) : Prop :=
  -x^2 = (2*x + 4)/(x + 2)

theorem equation_solution (x : ℂ) (h : x ≠ -2) :
  solveEquation x ↔ x = -2 ∨ x = Complex.I * 2 ∨ x = - Complex.I * 2 :=
sorry

end equation_solution_l391_391849


namespace greatest_prime_factor_15f_plus_17f_l391_391199

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391199


namespace digits_solution_l391_391139

theorem digits_solution (x y z : ℕ) 
  (h1 : K_def : K = 1 * 2^19 + 0 * 2^18 + 1 * 2^17 + 1 * 2^16 + 0 * 2^15 + 1 * 2^14 + 0 * 2^13 + 1 * 2^12 + 0 * 2^11 + 1 * 2^10 + 0 * 2^9 + 1 * 2^8 + x * 2^7 + y * 2^6 + z * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0)
  (h2 : K % 7 = 0) :
  x = 0 ∧ y = 1 ∧ z = 0 :=
by
  sorry

end digits_solution_l391_391139


namespace triangle_sin_B_l391_391704

noncomputable def sin_B (a b : ℝ) (A : ℝ) : ℝ :=
  (b * Real.sin A) / a

theorem triangle_sin_B :
  ∀ (a b : ℝ) (A B : ℝ), a = 3 * Real.sqrt 3 → b = 4 → A = Real.pi / 6 → sin_B a b A = 2 * Real.sqrt 3 / 9
  :=
begin
  intros a b A B h1 h2 h3,
  have h4 : sin_B a b A = (4 * Real.sin (Real.pi / 6)) / (3 * Real.sqrt 3), {
    rw [h1, h2, h3],
  },
  rw Real.sin_pi_div_six at h4,
  have h5 : (4 * (1 / 2)) / (3 * Real.sqrt 3) = 2 * Real.sqrt 3 / 9, {
    rw [←mul_div_right_comm, div_mul_eq_mul_div, mul_one_div, one_div_eq_inv],
    field_simp,
    ring,
  },
  rw h5 at h4,
  exact h4,
end

end triangle_sin_B_l391_391704


namespace greatest_prime_factor_of_factorial_sum_l391_391330

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391330


namespace chest_contents_l391_391779

def Chest : Type := ℕ
def Coins : Type := ℕ

variable (empty : Coins → Coins → Prop)

noncomputable def Chest1 : Chest := 1
noncomputable def Chest2 : Chest := 2
noncomputable def Chest3 : Chest := 3

variable (goldCoins : Coins)
variable (silverCoins : Coins)
variable (copperCoins : Coins)

-- Conditions
variable (labelsIncorrect : Chest → Coins → Prop)

axiom label1 : labelsIncorrect Chest1 goldCoins
axiom label2 : labelsIncorrect Chest2 silverCoins
axiom label3 : labelsIncorrect Chest3 (goldCoins ∨ silverCoins)

axiom uniqueContents : ∀ c : Chest, c = Chest1 ∨ c = Chest2 ∨ c = Chest3
axiom distinctContents : ∀ c1 c2 : Chest, c1 ≠ c2 → (c1 = Chest1 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest3) ∨ (c1 = Chest2 ∧ c2 ≠ Chest2 ∧ c2 ≠ Chest3) ∨ (c1 = Chest3 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest2)

theorem chest_contents : (exists! c1 : Coins, c1 = silverCoins ∧ labelsIncorrect Chest1 c1) ∧
                         (exists! c2 : Coins, c2 = goldCoins ∧ labelsIncorrect Chest2 c2) ∧
                         (exists! c3 : Coins, c3 = copperCoins ∧ labelsIncorrect Chest3 c3)
                         :=
by
  sorry

end chest_contents_l391_391779


namespace greatest_prime_factor_15f_plus_17f_l391_391200

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391200


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391369

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391369


namespace geometric_sequence_sum_l391_391710

-- Define the positive terms of the geometric sequence
variables {a_1 a_2 a_3 a_4 a_5 : ℝ}
-- Assume all terms are positive
variables (h1 : a_1 > 0) (h2 : a_2 > 0) (h3 : a_3 > 0) (h4 : a_4 > 0) (h5 : a_5 > 0)

-- Main condition given in the problem
variable (h_main : a_1 * a_3 + 2 * a_2 * a_4 + a_3 * a_5 = 16)

-- Goal: Prove that a_2 + a_4 = 4
theorem geometric_sequence_sum : a_2 + a_4 = 4 :=
by
  sorry

end geometric_sequence_sum_l391_391710


namespace greatest_prime_factor_15_17_factorial_l391_391210

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391210


namespace zhang_hua_monthly_repayment_l391_391906

-- Define constants 
def borrowed_amount : ℝ := 480000
def total_years : ℝ := 20
def monthly_interest_rate : ℝ := 0.004
def months_in_year : ℝ := 12
def repayment_periods : ℝ := total_years * months_in_year
def monthly_principal_repayment : ℝ := borrowed_amount / repayment_periods

-- Define the function for monthly repayment amount
def monthly_repayment (n : ℕ) : ℝ := 
  let principal_repaid : ℝ := (n-1) * monthly_principal_repayment
  let remaining_loan : ℝ := borrowed_amount - principal_repaid
  let interest_payment : ℝ := remaining_loan * monthly_interest_rate
  monthly_principal_repayment + interest_payment

theorem zhang_hua_monthly_repayment (n : ℕ) :
  1 ≤ n → n ≤ 240 → monthly_repayment n = 3928 - 8 * n :=
by
  sorry

end zhang_hua_monthly_repayment_l391_391906


namespace rain_probability_correct_l391_391151

noncomputable def prob_rain_friday := 0.30
noncomputable def prob_rain_saturday := 0.45
noncomputable def prob_rain_sunday := 0.55

def prob_no_rain : ℝ := 
  (1 - prob_rain_friday) * (1 - prob_rain_saturday) * (1 - prob_rain_sunday)

def prob_rain : ℝ := 1 - prob_no_rain

theorem rain_probability_correct : prob_rain = 0.82675 := sorry

end rain_probability_correct_l391_391151


namespace greatest_prime_factor_15_17_factorial_l391_391339

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391339


namespace even_integers_between_5000_8000_with_four_distinct_digits_l391_391678

theorem even_integers_between_5000_8000_with_four_distinct_digits : 
  let count := (sum (λ d₁, if d₁ ∈ [5, 6, 7] then
     (sum (λ d₄, if d₄ ≠ d₁ ∧ d₄ % 2 = 0 then
     (sum (λ d₂, if d₂ ≠ d₁ ∧ d₂ ≠ d₄ then
     (sum (λ d₃, if d₃ ≠ d₁ ∧ d₃ ≠ d₂ ∧ d₃ ≠ d₄ then 1 else 0))
     else 0))
     else 0))
     else 0) 
  in count = 728 := 
by
  sorry

end even_integers_between_5000_8000_with_four_distinct_digits_l391_391678


namespace koschei_chests_l391_391753

theorem koschei_chests :
  ∃ (contents : Fin 3 → String), 
    -- All chests labels are incorrect
    (contents 0 ≠ "gold coins" ∧ contents 1 ≠ "silver coins" ∧ contents 2 ≠ "gold or silver coins") ∧ 
    -- Each chest contains exactly one type of coin 
    (∀ i j : Fin 3, i ≠ j → contents i ≠ contents j) ∧ 
    -- Providing the final conclusion about what is in each chest
    (contents 0 = "silver coins" ∧ contents 1 = "gold coins" ∧ contents 2 = "copper coins") :=
begin
  use (λ k, if k = 0 then "silver coins" else if k = 1 then "gold coins" else "copper coins"),
  split,
  {
    -- Proof of all labels being incorrect
    split; simp,
  },
  split,
  {
    -- Proof of each chest containing a unique type of coin
    intros i j h,
    cases i; cases j; simp [h],
  },
  {
    -- Proof of the final conclusion
    split; split; simp,
  },
end

end koschei_chests_l391_391753


namespace pens_distributed_evenly_l391_391874

theorem pens_distributed_evenly (S : ℕ) (P : ℕ) (pencils : ℕ) 
  (hS : S = 10) (hpencils : pencils = 920) 
  (h_pencils_distributed : pencils % S = 0) 
  (h_pens_distributed : P % S = 0) : 
  ∃ k : ℕ, P = 10 * k :=
by 
  sorry

end pens_distributed_evenly_l391_391874


namespace petya_friends_l391_391037

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l391_391037


namespace greatest_prime_factor_15_17_factorial_l391_391214

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391214


namespace greatest_prime_factor_of_15_l391_391454

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391454


namespace even_product_probability_l391_391106

def SpinnerC_numbers := {1, 2, 3, 4, 5, 6}
def SpinnerD_numbers := {1, 2, 3}

noncomputable def probability_even_product : ℚ :=
  let total_outcomes := 6 * 3
  let even_C := {2, 4, 6}
  let odd_C := {1, 3, 5}
  let even_D := {2}
  let odd_D := {1, 3}
  let even_product_outcomes := (3 * 3) + (3 * 1)
  (even_product_outcomes : ℚ) / (total_outcomes : ℚ)

theorem even_product_probability : probability_even_product = 2 / 3 := by
  sorry

end even_product_probability_l391_391106


namespace quadratic_prime_at_another_point_l391_391956

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem quadratic_prime_at_another_point
  (f : ℤ → ℤ)
  (h_quad : ∃ a b c : ℤ, ∀ x : ℤ, f(x) = a*x^2 + b*x + c)
  (n : ℤ)
  (h_prime_n_minus_1 : is_prime (f(n - 1)))
  (h_prime_n : is_prime (f(n)))
  (h_prime_n_plus_1 : is_prime (f(n + 1))) :
  ∃ m : ℤ, m ≠ n - 1 ∧ m ≠ n ∧ m ≠ n + 1 ∧ is_prime (f(m)) :=
begin
  sorry
end

end quadratic_prime_at_another_point_l391_391956


namespace denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry_l391_391995

variable (DenyMotion : Prop) (AcknowledgeStillness : Prop) (LeadsToRelativism : Prop)
variable (LeadsToSophistry : Prop)

theorem denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry
  (h1 : DenyMotion)
  (h2 : AcknowledgeStillness)
  (h3 : DenyMotion ∧ AcknowledgeStillness → ¬LeadsToRelativism)
  (h4 : DenyMotion ∧ AcknowledgeStillness → ¬LeadsToSophistry):
  ¬ (DenyMotion ∧ AcknowledgeStillness → LeadsToRelativism ∧ LeadsToSophistry) :=
by sorry

end denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry_l391_391995


namespace greatest_prime_factor_15f_plus_17f_l391_391188

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391188


namespace player_A_advantage_l391_391472

theorem player_A_advantage (B A : ℤ) (rolls : ℕ) (h : rolls = 36) 
  (game_conditions : ∀ (x : ℕ), (x % 2 = 1 → A = A + x ∧ B = B - x) ∧ 
                      (x % 2 = 0 ∧ x ≠ 2 → A = A - x ∧ B = B + x) ∧ 
                      (x = 2 → A = A ∧ B = B)) : 
  (36 * (1 / 18 : ℚ) = 2) :=
by {
  -- Mathematical proof will be filled here
  sorry
}

end player_A_advantage_l391_391472


namespace greatest_prime_factor_15_17_factorial_l391_391345

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391345


namespace intersecting_lines_l391_391873

theorem intersecting_lines (a b : ℚ) :
  (3 = (1 / 3 : ℚ) * 4 + a) → 
  (4 = (1 / 2 : ℚ) * 3 + b) → 
  a + b = 25 / 6 :=
by
  intros h1 h2
  sorry

end intersecting_lines_l391_391873


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391263

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391263


namespace petya_friends_l391_391090

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l391_391090


namespace greatest_prime_factor_of_sum_factorials_l391_391273

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391273


namespace largest_num_of_hcf_and_lcm_factors_l391_391117

theorem largest_num_of_hcf_and_lcm_factors (hcf : ℕ) (f1 f2 : ℕ) (hcf_eq : hcf = 23) (f1_eq : f1 = 13) (f2_eq : f2 = 14) : 
    hcf * max f1 f2 = 322 :=
by
  -- use the conditions to find the largest number
  rw [hcf_eq, f1_eq, f2_eq]
  sorry

end largest_num_of_hcf_and_lcm_factors_l391_391117


namespace greatest_prime_factor_of_factorial_sum_l391_391324

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391324


namespace greatest_prime_factor_of_sum_l391_391400

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391400


namespace A_serves_on_50th_week_is_Friday_l391_391846

-- Define the people involved in the rotation
inductive Person
| A | B | C | D | E | F

open Person

-- Define the function that computes the day A serves on given the number of weeks
def day_A_serves (weeks : ℕ) : ℕ :=
  let days := weeks * 7
  (days % 6 + 0) % 7 -- 0 is the offset for the initial day when A serves (Sunday)

theorem A_serves_on_50th_week_is_Friday :
  day_A_serves 50 = 5 :=
by
  -- We provide the proof here
  sorry

end A_serves_on_50th_week_is_Friday_l391_391846


namespace find_f1_l391_391612

noncomputable theory

def f : ℝ → ℝ := sorry

axiom f_property : ∀ x : ℝ, f(f(x)) = x^2 * f(x) - x + 1

theorem find_f1 : f(1) = 1 :=
by
  -- The proof would go here
  sorry

end find_f1_l391_391612


namespace a_20_is_5_7_l391_391666

def sequence_rule (a : ℝ) : ℝ :=
  if 0 ≤ a ∧ a < 1 / 2 then 2 * a else 2 * a - 1

noncomputable def a_n (n : ℕ) : ℝ :=
  (nat.iterate sequence_rule n (6 / 7))

theorem a_20_is_5_7 : a_n 20 = 5 / 7 :=
sorry

end a_20_is_5_7_l391_391666


namespace greatest_prime_factor_15_fact_17_fact_l391_391182

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391182


namespace solution_set_of_inequality_l391_391882

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x ≥ 0} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 5} := by
  sorry

end solution_set_of_inequality_l391_391882


namespace greatest_prime_factor_15_fact_17_fact_l391_391170

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391170


namespace trigonometric_identity_l391_391615

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin α * Real.sin (3 * Real.pi / 2 - α) = -3 / 10 :=
by
  sorry

end trigonometric_identity_l391_391615


namespace power_function_increasing_on_pos_infty_l391_391468

open Real

theorem power_function_increasing_on_pos_infty (x : ℝ) (h : 0 < x) : 
  ∃ f : ℝ → ℝ, f x = x ^ (1 / 2) ∧ (∀ x : ℝ, 0 < x → 0 < f' x) :=
begin
  sorry
end

end power_function_increasing_on_pos_infty_l391_391468


namespace chest_contents_correct_l391_391763

-- Define the chests
inductive Chest
| chest1
| chest2
| chest3

open Chest

-- Define the contents
inductive Contents
| gold
| silver
| copper

open Contents

-- Define the labels, all of which are incorrect
def label (c : Chest) : Prop :=
  match c with
  | chest1 => gold
  | chest2 => silver
  | chest3 => gold ∨ silver

-- Assuming the labels are all incorrect
axiom label_incorrect : ∀ (c : Chest), ¬(label c = true)

-- One chest contains each type of coin
axiom one_gold : ∃ c, ∀ x, x ≠ c → Contents x ≠ gold
axiom one_silver : ∃ c, ∀ x, x ≠ c → Contents x ≠ silver
axiom one_copper : ∃ c, ∀ x, x ≠ c → Contents x ≠ copper

-- Determine the contents of each chest
def chest1_contents := silver
def chest2_contents := gold
def chest3_contents := copper

-- Prove the correspondence
theorem chest_contents_correct :
  (chest1_contents = silver) ∧ 
  (chest2_contents = gold) ∧ 
  (chest3_contents = copper) :=
by
  split
  · exact sorry
  split
  · exact sorry
  · exact sorry

end chest_contents_correct_l391_391763


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391375

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391375


namespace calculate_fraction_l391_391981

variable (a b : ℝ)

theorem calculate_fraction (h : a ≠ b) : (2 * a / (a - b)) + (2 * b / (b - a)) = 2 := by
  sorry

end calculate_fraction_l391_391981


namespace rate_of_interest_per_annum_l391_391927

theorem rate_of_interest_per_annum (P R : ℝ) (T : ℝ) 
  (h1 : T = 8)
  (h2 : (P / 5) = (P * R * T) / 100) : 
  R = 2.5 := 
by
  sorry

end rate_of_interest_per_annum_l391_391927


namespace smallest_largest_same_parity_l391_391570

-- Here, we define the conditions, including the set A_P and the element c achieving the maximum.
def is_maximum (A_P : Set Int) (c : Int) : Prop := c ∈ A_P ∧ ∀ x ∈ A_P, x ≤ c
def has_uniform_parity (A_P : Set Int) : Prop := 
  ∀ a₁ a₂ ∈ A_P, (a₁ % 2 = 0 → a₂ % 2 = 0) ∧ (a₁ % 2 = 1 → a₂ % 2 = 1)

-- This statement confirms the parity uniformity of the smallest and largest elements of the set A_P.
theorem smallest_largest_same_parity (A_P : Set Int) (c : Int) 
  (hc_max: is_maximum A_P c) (h_uniform: has_uniform_parity A_P): 
  ∀ min max ∈ A_P, ((min = max ∨ min ≠ max) → (min % 2 = max % 2)) := 
by
  intros min max hmin hmax h_eq
  have h_parity := h_uniform min max hmin hmax
  cases nat.decidable_eq (min % 2) 0 with h_even h_odd
  { rw nat.mod_eq_zero_of_dvd h_even at h_parity,
    exact h_parity.1 h_even, },
  { rw nat.mod_eq_one_of_dvd h_odd at h_parity,
    exact h_parity.2 h_odd, }
  sorry

end smallest_largest_same_parity_l391_391570


namespace petya_friends_l391_391034

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l391_391034


namespace greatest_prime_factor_of_sum_l391_391404

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391404


namespace number_of_teachers_l391_391498

theorem number_of_teachers (total_people : ℕ) (sampled_individuals : ℕ) (sampled_students : ℕ) 
    (school_total : total_people = 2400) 
    (sample_total : sampled_individuals = 160) 
    (sample_students : sampled_students = 150) : 
    ∃ teachers : ℕ, teachers = 150 := 
by
  -- Proof omitted
  sorry

end number_of_teachers_l391_391498


namespace cross_product_and_orthogonality_l391_391674

-- Define vectors a and b
def a : EuclideanSpace ℝ (Fin 3) := ![3, -2, 4]
def b : EuclideanSpace ℝ (Fin 3) := ![1, 5, -3]

-- Define the cross product
def c : EuclideanSpace ℝ (Fin 3) := ![-14, 13, 17]

-- Prove that c is the cross product of a and b, and that c is perpendicular to both a and b
theorem cross_product_and_orthogonality 
  (a b c : EuclideanSpace ℝ (Fin 3)) 
  (h1 : a = ![3, -2, 4])
  (h2 : b = ![1, 5, -3])
  (hc : c = cross_product a b)
  (orth_a : dot_product a c = 0)
  (orth_b : dot_product b c = 0) 
  : c = ![-14, 13, 17] :=
sorry

end cross_product_and_orthogonality_l391_391674


namespace greatest_prime_factor_of_15_l391_391446

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391446


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391420

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391420


namespace greatest_prime_factor_of_sum_l391_391396

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391396


namespace greatest_prime_factor_15_fact_17_fact_l391_391227

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391227


namespace correct_statement_l391_391470

noncomputable def problem_conditions : Prop :=
  (∀ (P : ℝ), (0 ≤ P ∧ P < 1) → P ≠ 0) ∧ -- Statement A: A probability between 0 and 1 is not impossible
  (∀ (P : ℝ), (0 < P ∧ P < 1) → P ≠ 1) ∧ -- Statement B: A probability not equal to 0 can be uncertain or certain
  (∀ (P : ℝ), (0 < P ∧ P < 1) → P ≠ 1) ∧ -- Statement D: A probability of 0.99999 is not certain

noncomputable def correct_answer (P : ℝ) : Prop :=
  (0 < P ∧ P < 1) → True -- Statement C: An uncertain probability relates to uncertainty

theorem correct_statement : problem_conditions → correct_answer :=
by
  intros cond
  sorry

end correct_statement_l391_391470


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391376

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391376


namespace correct_expression_for_f_l391_391813

-- Define a periodic and even function f with the given properties
noncomputable def f (x : ℝ) : ℝ := sorry

-- Define periodicity condition
def periodic (p : ℝ) (f : ℝ → ℝ) := ∀ x, f(x + p) = f(x)

-- Define even function condition
def even (f : ℝ → ℝ) := ∀ x, f(-x) = f(x)

-- The given function's property on a specific interval
def interval_property (f : ℝ → ℝ) := ∀ x ∈ set.Icc 2 3, f(x) = x

-- The proof statement
theorem correct_expression_for_f :
  periodic 2 f ∧ even f ∧ interval_property f → ∀ x ∈ set.Icc (-2) 0, f(x) = 3 - |x + 1| :=
begin
  sorry
end

end correct_expression_for_f_l391_391813


namespace greatest_prime_factor_15_fact_17_fact_l391_391228

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391228


namespace greatest_prime_factor_15_fact_17_fact_l391_391226

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391226


namespace greatest_prime_factor_of_15_l391_391451

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391451


namespace find_ellipse_equation_maximum_area_triangle_POQ_l391_391651

noncomputable def ellipse := {a b : ℝ // a > 0 ∧ b > 0 ∧ a > b}

def is_on_ellipse (a b x y : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem find_ellipse_equation (a b : ℝ) (h : a > b > 0) 
  (ecc : a > 0 ∧ b > 0 ∧ (sqrt 3) / 2 = sqrt (a^2 - b^2) / a)
  (point_on_ellipse : is_on_ellipse a b (-sqrt 3) (1 / 2)) :
  a = 2 ∧ b = 1 := 
by 
  sorry

theorem maximum_area_triangle_POQ (a b : ℝ) (ecc : a = 2 ∧ b = 1)
  (line_inter : ∃ l : ℝ → ℝ, ∀ P Q : (ℝ × ℝ),
    is_on_ellipse a b (fst P) (snd P) ∧ 
    is_on_ellipse a b (fst Q) (snd Q) ∧ 
    (fst P + fst Q) / 2 = 1 / sqrt 3 ) :
  ∃ area, area = 1 :=
by
  sorry

end find_ellipse_equation_maximum_area_triangle_POQ_l391_391651


namespace petya_has_19_friends_l391_391051

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l391_391051


namespace chest_contents_solution_l391_391786

-- Definitions corresponding to the conditions.
structure ChestContents :=
  (chest1 chest2 chest3 : String)

-- Given conditions
def labelsAreIncorrect (contents : ChestContents) : Prop :=
  contents.chest1 ≠ "Gold coins" ∧
  contents.chest2 ≠ "Silver coins" ∧
  contents.chest3 ≠ "Gold coins" ∧
  contents.chest3 ≠ "Silver coins"

def uniqueCoins (contents : ChestContents) : Prop :=
  (contents.chest1 = "Gold coins" ∨ contents.chest1 = "Silver coins" ∨ contents.chest1 = "Copper coins") ∧
  (contents.chest2 = "Gold coins" ∨ contents.chest2 = "Silver coins" ∨ contents.chest2 = "Copper coins") ∧
  (contents.chest3 = "Gold coins" ∨ contents.chest3 = "Silver coins" ∨ contents.chest3 = "Copper coins") ∧
  (contents.chest1 ≠ contents.chest2) ∧
  (contents.chest1 ≠ contents.chest3) ∧
  (contents.chest2 ≠ contents.chest3)

-- The proof statement
theorem chest_contents_solution : ∃ (contents : ChestContents),
  labelsAreIncorrect contents ∧ uniqueCoins contents ∧
  contents.chest1 = "Silver coins" ∧
  contents.chest2 = "Gold coins" ∧
  contents.chest3 = "Copper coins" :=
begin
  sorry
end

end chest_contents_solution_l391_391786


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391381

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391381


namespace jenny_mother_age_l391_391826

theorem jenny_mother_age:
  (∀ x : ℕ, (50 + x = 2 * (10 + x)) → (2010 + x = 2040)) :=
by
  sorry

end jenny_mother_age_l391_391826


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391267

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391267


namespace chests_contents_l391_391793

-- Define the coin types
inductive CoinType
| Gold
| Silver
| Copper

open CoinType

-- Define the chests
def Chest : Type

-- Define the inscriptions on the chests (all of which are known to be incorrect)
def chest_1_inscription : Prop := ∀ c : Chest, c = Gold
def chest_2_inscription : Prop := ∀ c : Chest, c = Silver
def chest_3_inscription : Prop := ∀ c : Chest, c = Gold ∨ c = Silver

-- Define the truth about what the chests must contain
def contains_gold (c : Chest) : Prop
def contains_silver (c : Chest) : Prop
def contains_copper (c : Chest) : Prop

-- The conditions given in the problem, stating that the true contents are different from the inscriptions
axiom incorrect_inscriptions :
  (¬ chest_1_inscription) ∧
  (¬ chest_2_inscription) ∧
  (¬ chest_3_inscription)

-- The problem states we have exactly one chest of each type 
axiom one_gold : ∃ c : Chest, contains_gold c
axiom one_silver : ∃ c : Chest, contains_silver c
axiom one_copper : ∃ c : Chest, contains_copper c

-- The main theorem we need to prove
theorem chests_contents :
  (∀ c : Chest, contains_silver c ↔ c = Chest₁) ∧
  (∀ c : Chest, contains_gold c ↔ c = Chest₂) ∧
  (∀ c : Chest, contains_copper c ↔ c = Chest₃) :=
sorry

end chests_contents_l391_391793


namespace find_CD_l391_391933

noncomputable def triangle_constants (a b c : ℝ) := 
  ∃ A B C D : Type,
    (BC = a ∧ AC = b ∧ AB = c) ∧ 
    ∃ (angle_A : ℝ), 
      angle_A > 0 ∧ angle_A < π ∧
      ∃ (angle_BCD : ℝ), 
        angle_BCD = angle_A 

theorem find_CD (a b c : ℝ) (h : triangle_constants a b c) : 
  ∃ CD : ℝ, CD = (a * b) / c := 
sorry

end find_CD_l391_391933


namespace pebbles_calculation_l391_391818

theorem pebbles_calculation (initial_pebbles : ℕ) (half_skipped : ℕ) (pebbles_given : ℕ) : 
  initial_pebbles = 18 → 
  half_skipped = initial_pebbles / 2 → 
  pebbles_given = 30 → 
  (half_skipped + pebbles_given) = 39 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end pebbles_calculation_l391_391818


namespace greatest_prime_factor_15_17_factorial_l391_391338

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391338


namespace find_m_range_l391_391638

noncomputable def proposition_p (x : ℝ) : Prop := (-2 : ℝ) ≤ x ∧ x ≤ 10
noncomputable def proposition_q (x : ℝ) (m : ℝ) : Prop := (1 - m ≤ x ∧ x ≤ 1 + m)

theorem find_m_range (m : ℝ) (h : m > 0) : (¬ ∃ x : ℝ, proposition_p x) → (¬ ∃ x : ℝ, proposition_q x m) → (¬ (¬ (¬ ∃ x : ℝ, proposition_q x m)) → ¬ (¬ ∃ x : ℝ, proposition_p x)) → m ≥ 9 := 
sorry

end find_m_range_l391_391638


namespace value_of_r_when_n_is_3_l391_391016

theorem value_of_r_when_n_is_3 : 
  let r := 3^s - 2*s
  let s := 2^(n^2) + n
  n = 3 → r = 3^515 - 1030 :=
by
  intros r s n h
  sorry

end value_of_r_when_n_is_3_l391_391016


namespace inradius_of_triangle_l391_391142

theorem inradius_of_triangle (P A : ℝ) (hP: P = 36) (hA: A = 45) :
  let s := P / 2 in
  let r := A / s in
  r = 2.5 :=
by
  rw [hP, hA]
  let s := 36 / 2
  let r := 45 / s
  have hs : s = 18 := by norm_num
  rw hs at r
  have hr : r = 2.5 := by norm_num
  exact hr

end inradius_of_triangle_l391_391142


namespace sum_b_eq_l391_391646
open Nat

def a (m n : ℕ) : ℕ := 2 * m * n + 2
def b (m : ℕ) (n : ℕ) [fact (0 < m)] : ℝ := (a m n) * log (a m 1) n

noncomputable def T (m n : ℕ) : ℝ := (finset.range n).sum (λ i, b m (i + 1))

theorem sum_b_eq (m n : ℕ) [fact (0 < m)] : T m n = (n * (2 * n + 1) : ℝ) / 6 := by
  sorry

end sum_b_eq_l391_391646


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391241

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391241


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391291

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391291


namespace counterexample_exists_l391_391988

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop := ¬ is_prime n ∧ n > 1

theorem counterexample_exists :
  ∃ n, is_composite n ∧ is_composite (n - 3) ∧ n = 18 := by
  sorry

end counterexample_exists_l391_391988


namespace greatest_prime_factor_15_fact_17_fact_l391_391173

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391173


namespace find_fencing_cost_l391_391594

theorem find_fencing_cost
  (d : ℝ) (cost_per_meter : ℝ) (π : ℝ)
  (h1 : d = 22)
  (h2 : cost_per_meter = 2.50)
  (hπ : π = Real.pi) :
  (cost_per_meter * (π * d) = 172.80) :=
sorry

end find_fencing_cost_l391_391594


namespace problem_part1_problem_part2_l391_391480

-- Definitions of an acute-angled triangle with given conditions.
variables {A B C D E F A' B' C' : Type} [HasTriangle A B C] [Incircle I D E F] [Circumcircle O A B C]

-- Definitions of circumcircles \Gamma_1, \Gamma_2, and \Gamma_3 and their intersections with circumcircle of \triangle ABC.
variables (Γ1 : Circumcircle A E F) (Γ2 : Circumcircle B D F) (Γ3 : Circumcircle C D E)
definition intersect_Γ_O (Γ : Circumcircle) : Set Point := {p | p ∈ O ∩ Γ}

variables (A' ∈ intersect_Γ_O(Γ1)) (B' ∈ intersect_Γ_O(Γ2)) (C' ∈ intersect_Γ_O(Γ3))

-- Proof goals.
theorem problem_part1 :
  Cyclic Quadrilateral D E A' B' := sorry

theorem problem_part2 :
  Concurrent (Line DA') (Line EB') (Line FC') := sorry

end problem_part1_problem_part2_l391_391480


namespace koschei_chests_l391_391752

theorem koschei_chests :
  ∃ (contents : Fin 3 → String), 
    -- All chests labels are incorrect
    (contents 0 ≠ "gold coins" ∧ contents 1 ≠ "silver coins" ∧ contents 2 ≠ "gold or silver coins") ∧ 
    -- Each chest contains exactly one type of coin 
    (∀ i j : Fin 3, i ≠ j → contents i ≠ contents j) ∧ 
    -- Providing the final conclusion about what is in each chest
    (contents 0 = "silver coins" ∧ contents 1 = "gold coins" ∧ contents 2 = "copper coins") :=
begin
  use (λ k, if k = 0 then "silver coins" else if k = 1 then "gold coins" else "copper coins"),
  split,
  {
    -- Proof of all labels being incorrect
    split; simp,
  },
  split,
  {
    -- Proof of each chest containing a unique type of coin
    intros i j h,
    cases i; cases j; simp [h],
  },
  {
    -- Proof of the final conclusion
    split; split; simp,
  },
end

end koschei_chests_l391_391752


namespace ratio_JK_JM_l391_391711

-- Given constants and definitions
variables (s : ℝ) (w h : ℝ)
-- Condition 1: Square QRST has side length s
def square_area : ℝ := s^2

-- Condition 2: JKLM has area w * h and shares 60% of its area with QRST
def overlap_JKLM_QRST1 : ℝ := 0.6 * w * h

-- Condition 3: QRST shares 30% of its area with JKLM
def overlap_QRST_JKLM : ℝ := 0.3 * square_area s

-- Equate the overlap conditions
def overlap_eq : Prop := overlap_JKLM_QRST1 s w h = overlap_QRST_JKLM s

-- When overlap_eq is true, prove ratio of JK (width) to JM (height) is 12.5
theorem ratio_JK_JM (h_eq : overlap_eq s w h) : w / h = 12.5 :=
by sorry

end ratio_JK_JM_l391_391711


namespace petya_friends_count_l391_391059

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l391_391059


namespace discount_rate_on_pony_jeans_l391_391477

variable (F P : ℝ)
variable (FoxPrice PonyPrice TotalSavings : ℝ)
variable (TotalDiscountPercent : ℝ)

def conditions (F P : ℝ) : Prop :=
  FoxPrice = 15 ∧ PonyPrice = 18 ∧ TotalSavings = 8.64 ∧ TotalDiscountPercent = 22 ∧
  F + P = TotalDiscountPercent ∧
  3 * FoxPrice * (F / 100) + 2 * PonyPrice * (P / 100) = TotalSavings

theorem discount_rate_on_pony_jeans : ∀ F P : ℝ,
  conditions F P → P = 14 :=
by
  intro F P h
  have h1 : FoxPrice = 15 := And.left h
  have h2 : PonyPrice = 18 := And.left (And.right h)
  have h3 : TotalSavings = 8.64 := And.left (And.right (And.right h))
  have h4 : TotalDiscountPercent = 22 := And.left (And.right (And.right (And.right h)))
  have sum_eq : F + P = TotalDiscountPercent := And.left (And.right (And.right (And.right (And.right h))))
  have savings_eq : 3 * FoxPrice * (F / 100) + 2 * PonyPrice * (P / 100) = TotalSavings := And.right (And.right (And.right (And.right (And.right h))))
  sorry

end discount_rate_on_pony_jeans_l391_391477


namespace isosceles_triangle_same_color_exists_l391_391877

theorem isosceles_triangle_same_color_exists (plane : Type) [finite_plane : Finite plane] 
  (white black : plane → Prop)
  (exists_white : ∃ p, white p)
  (exists_black : ∃ p, black p) : 
  ∃ (A B C : plane), (white A ∧ white B ∧ white C ∨ black A ∧ black B ∧ black C) ∧ 
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧ (dist A B = dist B C ∨ dist B C = dist C A ∨ dist C A = dist A B) :=
begin
  sorry
end

end isosceles_triangle_same_color_exists_l391_391877


namespace no_solution_eq1_l391_391484

   theorem no_solution_eq1 : ¬ ∃ x, (3 - x) / (x - 4) - 1 / (4 - x) = 1 :=
   by
     sorry
   
end no_solution_eq1_l391_391484


namespace necessary_but_not_sufficient_condition_l391_391654

open Real

noncomputable def f (x : ℝ) : ℝ := log x / log 2

theorem necessary_but_not_sufficient_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a > b) ↔ (f a > f b) :=
begin
  sorry
end

end necessary_but_not_sufficient_condition_l391_391654


namespace fixed_point_sum_l391_391696

noncomputable theory
open_locale classical

-- Define the function and the conditions
def f (a : ℝ) (x : ℝ) := a^(x - 1) + 2

-- State the theorem
theorem fixed_point_sum (a m n : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (h_passes : f a m = n):
  m + n = 4 :=
sorry

end fixed_point_sum_l391_391696


namespace integral_sqrt1_minus_x2_plus_sin_eq_pi_div_2_l391_391591

theorem integral_sqrt1_minus_x2_plus_sin_eq_pi_div_2 : 
  ∫ x in -1..1, (Real.sqrt (1 - x^2) + Real.sin x) = Real.pi / 2 := 
by
  sorry

end integral_sqrt1_minus_x2_plus_sin_eq_pi_div_2_l391_391591


namespace max_value_is_zero_l391_391808

noncomputable def max_value (x y : ℝ) (h : 2 * (x^3 + y^3) = x^2 + y^2) : ℝ :=
  x^2 - y^2

theorem max_value_is_zero (x y : ℝ) (h : 2 * (x^3 + y^3) = x^2 + y^2) : max_value x y h = 0 :=
sorry

end max_value_is_zero_l391_391808


namespace amount_saved_percentage_l391_391799

variable (S : ℝ) 

-- Condition: Last year, Sandy saved 7% of her annual salary
def amount_saved_last_year (S : ℝ) : ℝ := 0.07 * S

-- Condition: This year, she made 15% more money than last year
def salary_this_year (S : ℝ) : ℝ := 1.15 * S

-- Condition: This year, she saved 10% of her salary
def amount_saved_this_year (S : ℝ) : ℝ := 0.10 * salary_this_year S

-- The statement to prove
theorem amount_saved_percentage (S : ℝ) : 
  amount_saved_this_year S = 1.642857 * amount_saved_last_year S :=
by 
  sorry

end amount_saved_percentage_l391_391799


namespace find_tangent_line_l391_391861

noncomputable def tangent_line_equation : Prop :=
  ∀ (f : ℝ → ℝ) (x₀ y₀ : ℝ), 
  f = (λ x, 2 * x^2 - 3 * x) ∧ x₀ = 1 ∧ y₀ = -1 → 
  let k := deriv f x₀ in
  k = 1 ∧ y₀ = f x₀ → 
  ∀ (x y : ℝ), y = (f x₀) + k * (x - x₀) → x - y - 2 = 0

theorem find_tangent_line : tangent_line_equation :=
sorry

end find_tangent_line_l391_391861


namespace soccer_field_solution_l391_391541

noncomputable def soccer_field_problem : Prop :=
  ∃ (a b c d : ℝ), 
    (abs (a - b) = 1 ∨ abs (a - b) = 2 ∨ abs (a - b) = 3 ∨ abs (a - b) = 4 ∨ abs (a - b) = 5 ∨ abs (a - b) = 6) ∧
    (abs (a - c) = 1 ∨ abs (a - c) = 2 ∨ abs (a - c) = 3 ∨ abs (a - c) = 4 ∨ abs (a - c) = 5 ∨ abs (a - c) = 6) ∧
    (abs (a - d) = 1 ∨ abs (a - d) = 2 ∨ abs (a - d) = 3 ∨ abs (a - d) = 4 ∨ abs (a - d) = 5 ∨ abs (a - d) = 6) ∧
    (abs (b - c) = 1 ∨ abs (b - c) = 2 ∨ abs (b - c) = 3 ∨ abs (b - c) = 4 ∨ abs (b - c) = 5 ∨ abs (b - c) = 6) ∧
    (abs (b - d) = 1 ∨ abs (b - d) = 2 ∨ abs (b - d) = 3 ∨ abs (b - d) = 4 ∨ abs (b - d) = 5 ∨ abs (b - d) = 6) ∧
    (abs (c - d) = 1 ∨ abs (c - d) = 2 ∨ abs (c - d) = 3 ∨ abs (c - d) = 4 ∨ abs (c - d) = 5 ∨ abs (c - d) = 6)

theorem soccer_field_solution : soccer_field_problem :=
  sorry

end soccer_field_solution_l391_391541


namespace john_needs_20_nails_l391_391608

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

end john_needs_20_nails_l391_391608


namespace find_x_l391_391605

theorem find_x (x : ℝ) (h : 3 * x + 15 = (1 / 3) * (7 * x + 42)) : x = -3 / 2 :=
sorry

end find_x_l391_391605


namespace rightmost_three_digits_of_7_pow_2023_l391_391165

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 343 :=
sorry

end rightmost_three_digits_of_7_pow_2023_l391_391165


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391418

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391418


namespace greatest_prime_factor_15_fact_17_fact_l391_391221

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391221


namespace greatest_prime_factor_15_17_factorial_l391_391218

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391218


namespace translate_upwards_one_unit_l391_391157

theorem translate_upwards_one_unit (x y : ℝ) : (y = 2 * x) → (y + 1 = 2 * x + 1) := 
by sorry

end translate_upwards_one_unit_l391_391157


namespace sally_boxes_total_l391_391840

-- Define the conditions and the problem statement
theorem sally_boxes_total (sold_saturday : ℕ) (percent_more : ℕ) (sold_sunday : ℕ) (total_sold : ℕ) :
  sold_saturday = 60 →
  percent_more = 50 →
  sold_sunday = sold_saturday + (percent_more/100 * sold_saturday) →
  total_sold = sold_saturday + sold_sunday →
  total_sold = 150 :=
begin
  sorry
end

end sally_boxes_total_l391_391840


namespace radius_of_larger_circle_l391_391900

theorem radius_of_larger_circle
  (r : ℝ) -- radius of the smaller circle
  (R : ℝ) -- radius of the larger circle
  (ratio : R = 4 * r) -- radii ratio 1:4
  (AC : ℝ) -- diameter of the larger circle
  (BC : ℝ) -- chord of the larger circle
  (AB : ℝ := 16) -- given condition AB = 16
  (diameter_AC : AC = 2 * R) -- AC is diameter of the larger circle
  (tangent : BC^2 = AB^2 + (2 * R)^2) -- Pythagorean theorem for the right triangle ABC
  :
  R = 32 := 
sorry

end radius_of_larger_circle_l391_391900


namespace max_value_l391_391152

variable {a b c : ℝ} (x1 x2 x3 λ : ℝ)

def f (x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem max_value (h1 : f(x1) = 0) (h2 : f(x2) = 0) (h3 : f(x3) = 0)
  (h4 : x2 - x1 = λ) (h5 : x3 > (x1 + x2) / 2) : 
  ∃ M, M = 2 ∧ ∀ a b c λ, (f(x1) = 0 ∧ f(x2) = 0 ∧ f(x3) = 0 ∧ x2 - x1 = λ ∧ 
  x3 > (x1 + x2) / 2) → (2 * a^3 + 27 * c - 9 * a * b) / λ^3 ≤ M :=
sorry

end max_value_l391_391152


namespace greatest_prime_factor_of_sum_factorials_l391_391280

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391280


namespace calc_q1_add_rneg1_l391_391003

noncomputable def f (x : ℝ) : ℝ := 2 * x^4 + 8 * x^3 - 5 * x^2 + 2 * x + 5
noncomputable def d (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem calc_q1_add_rneg1 : 
  ∃ (q r : ℝ → ℝ), (∀ x, f x = q x * d x + r x) ∧ (degree r < degree d) ∧ (q 1 + r (-1) = -2) :=
by
  sorry

end calc_q1_add_rneg1_l391_391003


namespace Petya_friends_l391_391087

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l391_391087


namespace triangle_angle_and_series_sum_l391_391705

-- Define the preconditions and the goal

theorem triangle_angle_and_series_sum (a b c : ℝ) (a_n : ℕ → ℝ) (d : ℝ) (n : ℕ) 
  (h1 : ∠A = ∠B = ∠C) (h2 : C = (2 * π) / 3) (h3 : a^2 - (b - c)^2 = (2 - sqrt 3) * b * c)
  (h4 : ∀ n, a_{n + 1} - a_n = d) (h5 : a_1 * cos (2 * ∠B) = 1) (h6 : a 2 * a 4 * a 8 = 1) :
  ∃ B : ℝ, B = π / 6 ∧ (∑ k in range n, 4 / (a_n * a_(n + 1))) = n / (n + 1) := sorry

end triangle_angle_and_series_sum_l391_391705


namespace circle_a_center_radius_circle_b_center_radius_circle_c_center_radius_l391_391602

-- Part (a): Prove the center and radius for the given circle equation: (x-3)^2 + (y+2)^2 = 16
theorem circle_a_center_radius :
  (∃ (a b : ℤ) (R : ℕ), (∀ (x y : ℝ), (x - 3) ^ 2 + (y + 2) ^ 2 = 16 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 3 ∧ b = -2 ∧ R = 4) :=
by {
  sorry
}

-- Part (b): Prove the center and radius for the given circle equation: x^2 + y^2 - 2(x - 3y) - 15 = 0
theorem circle_b_center_radius :
  (∃ (a b : ℤ) (R : ℕ), (∀ (x y : ℝ), x^2 + y^2 - 2 * (x - 3 * y) - 15 = 0 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 1 ∧ b = -3 ∧ R = 5) :=
by {
  sorry
}

-- Part (c): Prove the center and radius for the given circle equation: x^2 + y^2 = x + y + 1/2
theorem circle_c_center_radius :
  (∃ (a b : ℚ) (R : ℚ), (∀ (x y : ℚ), x^2 + y^2 = x + y + 1/2 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 1/2 ∧ b = 1/2 ∧ R = 1) :=
by {
  sorry
}

end circle_a_center_radius_circle_b_center_radius_circle_c_center_radius_l391_391602


namespace simplify_evaluate_l391_391842

noncomputable def a := (1 / 2) + Real.sqrt (1 / 2)

theorem simplify_evaluate (a : ℝ) (h : a = (1 / 2) + Real.sqrt (1 / 2)) :
  (a + Real.sqrt 3) * (a - Real.sqrt 3) - a * (a - 6) = 3 * Real.sqrt 2 :=
by sorry

end simplify_evaluate_l391_391842


namespace greatest_prime_factor_15_fact_17_fact_l391_391177

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391177


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391315
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391315


namespace petya_friends_l391_391033

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l391_391033


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391306
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391306


namespace find_angle_between_vectors_l391_391649

noncomputable def angle_between_vectors (a b : EuclideanSpace.axis ℝ) : ℝ := angle a b

/-- Given unit vectors a and b such that (a + b)^2 = 1, prove that the angle between them is 120 degrees. -/
theorem find_angle_between_vectors
  (a b : EuclideanSpace.axis ℝ)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (h : ∥a + b∥^2 = 1) :
  angle_between_vectors a b = 120 :=
sorry

end find_angle_between_vectors_l391_391649


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391410

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391410


namespace chest_contents_solution_l391_391783

-- Definitions corresponding to the conditions.
structure ChestContents :=
  (chest1 chest2 chest3 : String)

-- Given conditions
def labelsAreIncorrect (contents : ChestContents) : Prop :=
  contents.chest1 ≠ "Gold coins" ∧
  contents.chest2 ≠ "Silver coins" ∧
  contents.chest3 ≠ "Gold coins" ∧
  contents.chest3 ≠ "Silver coins"

def uniqueCoins (contents : ChestContents) : Prop :=
  (contents.chest1 = "Gold coins" ∨ contents.chest1 = "Silver coins" ∨ contents.chest1 = "Copper coins") ∧
  (contents.chest2 = "Gold coins" ∨ contents.chest2 = "Silver coins" ∨ contents.chest2 = "Copper coins") ∧
  (contents.chest3 = "Gold coins" ∨ contents.chest3 = "Silver coins" ∨ contents.chest3 = "Copper coins") ∧
  (contents.chest1 ≠ contents.chest2) ∧
  (contents.chest1 ≠ contents.chest3) ∧
  (contents.chest2 ≠ contents.chest3)

-- The proof statement
theorem chest_contents_solution : ∃ (contents : ChestContents),
  labelsAreIncorrect contents ∧ uniqueCoins contents ∧
  contents.chest1 = "Silver coins" ∧
  contents.chest2 = "Gold coins" ∧
  contents.chest3 = "Copper coins" :=
begin
  sorry
end

end chest_contents_solution_l391_391783


namespace part1_part2_l391_391623

noncomputable def a_seq : ℕ → ℝ
| 0       := 2
| (n + 1) := real.sqrt (2 * (a_seq n) - 1)

def b_seq (n : ℕ) : ℝ :=
(a_seq n) / real.sqrt (a_seq (n + 1)) - (a_seq (n + 1)) / real.sqrt (a_seq n)

theorem part1 (n : ℕ) : 1 < a_seq (n + 1) ∧ a_seq (n + 1) < a_seq n :=
sorry

theorem part2 (n : ℕ) : ∑ i in finset.range n, b_seq i < 6 - 3 * real.sqrt 2 :=
sorry

end part1_part2_l391_391623


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391432

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391432


namespace greatest_prime_factor_of_sum_factorials_l391_391271

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391271


namespace tetrahedron_surface_area_l391_391702

theorem tetrahedron_surface_area
  (V : ℝ) (hV : V = (16 / 3) * (2:ℝ).sqrt)
  (a : ℝ) (ha : (2:ℝ).sqrt / 12 * a^3 = V) :
  ∃ S, S = 16 * (3:ℝ).sqrt :=
by
  use (√3) * a^2
  sorry

end tetrahedron_surface_area_l391_391702


namespace arrange_digits_l391_391975

theorem arrange_digits (a b c d e f: ℕ) 
  (h1: a + d + e = 15) 
  (h2: a + b + f = 15) 
  (h3: b + c + d = 15) 
  (h4: b + d + f = 15) 
  (h5: c + e + f = 15) 
  (h6: a + c + e = 15) 
  (h7: a + b + e = 15)
  (distinct: ∀ x y, (x ≠ y) → ({a, b, c, d, e, f} ⊆ {1, 2, 3, 4, 5, 6})) :
  a = 4 ∧ b = 1 ∧ c = 2 ∧ d = 5 ∧ e = 6 ∧ f = 3 :=
by
  sorry

end arrange_digits_l391_391975


namespace soccer_players_positions_l391_391532

noncomputable def positions : list ℝ := [0, 1, 4, 6]

def pairwise_distances (positions : list ℝ) : list ℝ :=
  let pairs := list.sigma positions positions
  let distances := pairs.map (λ p, abs (p.1 - p.2))
  distances.erase_dup

theorem soccer_players_positions :
  pairwise_distances positions = [1, 2, 3, 4, 5, 6] :=
by {
  sorry  -- Proof to be provided
}

end soccer_players_positions_l391_391532


namespace greatest_prime_factor_of_sum_l391_391391

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391391


namespace sequence_sum_l391_391994

noncomputable def b : ℕ → ℚ
| 0     := 3
| 1     := 5
| (n+2) := b (n+1) + 2 * b n

theorem sequence_sum :
  (∑' n, b n / 3^(n+1)) = 7 / 2 :=
by sorry

end sequence_sum_l391_391994


namespace merchant_marked_price_l391_391492

theorem merchant_marked_price (L : ℝ) (x : ℝ) : 
  (L = 100) →
  (L - 0.3 * L = 70) →
  (0.75 * x - 70 = 0.225 * x) →
  x = 133.33 :=
by
  intro h1 h2 h3
  sorry

end merchant_marked_price_l391_391492


namespace no_diff_parity_min_max_l391_391556

def A_P : Set ℤ := 
  {x | PositionProperty x}

variable (x y : ℤ) (hx : x ∈ A_P) (hy : y ∈ A_P)

theorem no_diff_parity_min_max :
  (∀ x ∈ A_P, PositionProperty x) →
  (∃ c, ∀ x ∈ A_P, x ≤ c) →
  (∀ c, (c = max_element A_P) → CriticalPointsProperty c) →
  ((min_element A_P) % 2 = (max_element A_P) % 2) :=
by
  sorry

end no_diff_parity_min_max_l391_391556


namespace find_integer_of_divisors_l391_391880

theorem find_integer_of_divisors:
  ∃ (N : ℕ), (∀ (l m n : ℕ), N = (2^l) * (3^m) * (5^n) → 
  (2^120) * (3^60) * (5^90) = (2^l * 3^m * 5^n)^( ((l+1)*(m+1)*(n+1)) / 2 ) ) → 
  N = 18000 :=
sorry

end find_integer_of_divisors_l391_391880


namespace trip_time_l391_391156

open Real

variables (d T : Real)

theorem trip_time :
  (T = d / 30 + (150 - d) / 6) ∧
  (T = 2 * (d / 30) + 1 + (150 - d) / 30) ∧
  (T - 1 = d / 6 + (150 - d) / 30) →
  T = 20 :=
by
  sorry

end trip_time_l391_391156


namespace area_of_intersection_polygon_l391_391984

structure Point := 
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

noncomputable def A : Point := ⟨0, 0, 0⟩
noncomputable def B : Point := ⟨20, 0, 0⟩
noncomputable def C : Point := ⟨20, 0, 20⟩
noncomputable def D : Point := ⟨20, 20, 20⟩
noncomputable def P : Point := ⟨3, 0, 0⟩
noncomputable def Q : Point := ⟨20, 0, 10⟩
noncomputable def R : Point := ⟨20, 5, 20⟩

def plane_eq (x y z : ℝ) : Prop := 
  (3/17) * x + (-20/17) * y + (-51/34) * z = (9/17)

def is_intersection (pt : Point) : Prop := 
  plane_eq pt.x pt.y pt.z

def cube_intersection_polygon_area : ℝ :=
  -- Calculate the area of the polygon formed by intersection points
  525

theorem area_of_intersection_polygon : 
  cube_intersection_polygon_area = 525 :=
by
  simp [cube_intersection_polygon_area]
  rfl

end area_of_intersection_polygon_l391_391984


namespace greatest_prime_factor_15_17_factorial_l391_391340

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391340


namespace minimal_sum_factorial_product_l391_391866

theorem minimal_sum_factorial_product :
  ∀ (p q r s : ℕ), (p * q * r * s = nat.factorial 12) → 
  (p + q + r + s ≥ 1613) := 
sorry

end minimal_sum_factorial_product_l391_391866


namespace number_of_complex_numbers_l391_391598

/-- 
  Prove that the number of different complex numbers \( z \) such that 
  \( |z| = 1 \) and \( z^{7!} - z^{3!} \) is a real number is 5019.
-/
theorem number_of_complex_numbers (z : ℂ) (h_abs : |z| = 1) (h_real : z^(7!) - z^(3!) ∈ ℝ) :
  {w : ℂ | |w| = 1 ∧ w^(7!) - w^(3!) ∈ ℝ}.to_finset.card = 5019 :=
by sorry

end number_of_complex_numbers_l391_391598


namespace existence_of_unique_root_l391_391141

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 5

theorem existence_of_unique_root :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  f 0 = -4 ∧
  f 2 = Real.exp 2 - 1 →
  ∃! c, f c = 0 :=
by
  sorry

end existence_of_unique_root_l391_391141


namespace incenter_distance_is_correct_l391_391150

noncomputable def incenter_distance (A B C I : Type) [triangle_props : IsIsoscelesRightTriangle A B C] (incenter_props : IsIncenter I A B C) (AB : Float) (angle_B : IsRightAngle B) : Float := sorry

theorem incenter_distance_is_correct (A B C I : Type) [triangle_props : IsIsoscelesRightTriangle A B C] (incenter_props : IsIncenter I A B C) (AB : Float) (angle_B : IsRightAngle B) (h1: AB = 4*Real.sqrt 2) : incenter_distance A B C I = 8 - 4*Real.sqrt 2 := by sorry

end incenter_distance_is_correct_l391_391150


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391415

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391415


namespace angle_of_inclination_of_line_l391_391695

-- Define the conditions
def hyperbola (m n : ℝ) : Prop := ∃ (x y : ℝ), (x^2 / (m^2) - y^2 / (n^2) = 1)
def eccentricity (m n e : ℝ) : Prop := e^2 = 1 + (n^2 / m^2)
def line_k (m n : ℝ) : ℝ := - (m / n)
def abs_fraction (m n sqrt_3 : ℝ) : Prop := abs (n / m) = sqrt_3

-- Prove the angle of inclination
theorem angle_of_inclination_of_line (m n : ℝ) (sqrt_3 : ℝ) (e : ℝ) 
  (H1 : hyperbola m n)
  (H2 : e = 2)
  (H3 : abs_fraction m n sqrt_3) :
  ∃ θ : ℝ, (θ = (Real.arctan (line_k m n)) ∨ θ = π - (Real.arctan (line_k m n))) ∧ 
           (θ = π / 6 ∨ θ = 5 * π / 6) :=
by
  sorry

end angle_of_inclination_of_line_l391_391695


namespace greatest_prime_factor_15_fact_17_fact_l391_391178

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391178


namespace find_a_l391_391881

def circle_equation (a x y : ℝ) : Prop := x^2 + y^2 + 2 * a * x + 4 * a * y = 0

theorem find_a (a : ℝ) (radius : ℝ) : radius = real.sqrt 5 → (∀ x y : ℝ, circle_equation a x y) → a = 1 ∨ a = -1 :=
by
  intros h_radius h_circle_eq
  sorry

end find_a_l391_391881


namespace distance_between_stripes_l391_391502

theorem distance_between_stripes (
  height_street : Real,
  base_curbs : Real,
  length_stripe1 : Real,
  length_stripe2 : Real
) :
  height_street = 60 ∧ base_curbs = 20 ∧ length_stripe1 = 75 ∧ length_stripe2 = 50 →
  ∃ distance : Real, distance = 19 :=
by
  sorry

end distance_between_stripes_l391_391502


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391416

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391416


namespace chest_contents_solution_l391_391787

-- Definitions corresponding to the conditions.
structure ChestContents :=
  (chest1 chest2 chest3 : String)

-- Given conditions
def labelsAreIncorrect (contents : ChestContents) : Prop :=
  contents.chest1 ≠ "Gold coins" ∧
  contents.chest2 ≠ "Silver coins" ∧
  contents.chest3 ≠ "Gold coins" ∧
  contents.chest3 ≠ "Silver coins"

def uniqueCoins (contents : ChestContents) : Prop :=
  (contents.chest1 = "Gold coins" ∨ contents.chest1 = "Silver coins" ∨ contents.chest1 = "Copper coins") ∧
  (contents.chest2 = "Gold coins" ∨ contents.chest2 = "Silver coins" ∨ contents.chest2 = "Copper coins") ∧
  (contents.chest3 = "Gold coins" ∨ contents.chest3 = "Silver coins" ∨ contents.chest3 = "Copper coins") ∧
  (contents.chest1 ≠ contents.chest2) ∧
  (contents.chest1 ≠ contents.chest3) ∧
  (contents.chest2 ≠ contents.chest3)

-- The proof statement
theorem chest_contents_solution : ∃ (contents : ChestContents),
  labelsAreIncorrect contents ∧ uniqueCoins contents ∧
  contents.chest1 = "Silver coins" ∧
  contents.chest2 = "Gold coins" ∧
  contents.chest3 = "Copper coins" :=
begin
  sorry
end

end chest_contents_solution_l391_391787


namespace subset_modulus_sum_geq_one_sixth_l391_391855

theorem subset_modulus_sum_geq_one_sixth 
  (n : ℕ) (z : Fin n → ℂ) (h : (∑ i, |z i|) = 1) :
  ∃ (s : Finset (Fin n)), |∑ i in s, z i| ≥ 1 / 6 :=
begin
  sorry,
end

end subset_modulus_sum_geq_one_sixth_l391_391855


namespace washer_cost_l391_391512

theorem washer_cost (D : ℝ) (H1 : D + (D + 220) = 1200) : D + 220 = 710 :=
by
  sorry

end washer_cost_l391_391512


namespace evaluate_stability_with_standard_deviation_l391_391896

-- Definition of the mathematical problem conditions
variables (n : ℕ) (x : fin n → ℝ)

-- Definition of the theorem that standard deviation can be used to evaluate stability
theorem evaluate_stability_with_standard_deviation:
  ( ∀ i, 0 ≤ x i ) →
  let mean_x := (finset.univ.sum (λ i, x i)) / n in
  let variance := (finset.univ.sum (λ i, (x i - mean_x) ^ 2)) / n in
  let std_dev := real.sqrt variance in
  true := sorry

end evaluate_stability_with_standard_deviation_l391_391896


namespace greatest_prime_factor_of_factorial_sum_l391_391323

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391323


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391422

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391422


namespace gender_badminton_confidence_l391_391876

variable (n : ℕ) (pB : ℝ) (pA_notB : ℝ) (pB_notA : ℝ)

theorem gender_badminton_confidence :
  let a := (pA_notB * n / 2).toInt in
  let b := (n / 2 - pA_notB * n / 2).toInt in
  let c := (n / 2 * (1 - pB_notA)).toInt in
  let d := (n / 2 - n / 2 * (1 - pB_notA)).toInt in
  let K2 := (n * (a * d - b * c)^2).toFloat / ((a + b) * (c + d) * (a + c) * (b + d)).toFloat in
  K2 > 10.828 ->
  True := 
by
  sorry

#check gender_badminton_confidence

end gender_badminton_confidence_l391_391876


namespace triangle_area_solution_l391_391125

noncomputable def triangle_area_equation : ℝ :=
  let f : Polynomial ℝ := Polynomial.C (-8 / 5) + Polynomial.X + Polynomial.C 4 * Polynomial.X - Polynomial.C 3 * Polynomial.X^2 + Polynomial.X^3
  let roots : Finset ℝ := f.roots
  let a := roots.toList[0]
  let b := roots.toList[1]
  let c := roots.toList[2]
  let q := (a + b + c) / 2
  let heron (a b c : ℝ) (q : ℝ) := q * (q - a) * (q - b) * (q - c)
  if heron a b c q < 0 then 0 else (heron a b c q).sqrt

theorem triangle_area_solution:
  triangle_area_equation x = Real.sqrt (6) / 4 := sorry

end triangle_area_solution_l391_391125


namespace length_of_train_is_200_meters_l391_391506

noncomputable def speed_kmh_to_ms (speed_kmh : ℕ) : ℚ :=
  speed_kmh * (5 / 18)

def length_of_train (speed_kmh : ℕ) (time_s : ℕ) : ℚ :=
  speed_kmh_to_ms speed_kmh * time_s

theorem length_of_train_is_200_meters (time_s : ℕ) (speed_kmh : ℕ) (h_time : time_s = 9) (h_speed : speed_kmh = 80) :
  length_of_train speed_kmh time_s = 200 :=
by
  rw [h_time, h_speed]
  have h_speed_ms: speed_kmh_to_ms speed_kmh = 200 / 9 := by
    rw [speed_kmh_to_ms, h_speed]
    norm_num
  rw [length_of_train, h_speed_ms, h_time]
  norm_num
  sorry

end length_of_train_is_200_meters_l391_391506


namespace minimum_reciprocal_sum_l391_391811

noncomputable def minimum_value_of_reciprocal_sum (x y z : ℝ) : ℝ :=
  if x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 then 
    max (1/x + 1/y + 1/z) (9/2)
  else
    0
  
theorem minimum_reciprocal_sum (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2): 
  1/x + 1/y + 1/z ≥ 9/2 :=
sorry

end minimum_reciprocal_sum_l391_391811


namespace find_equation_of_parabola_and_AF_find_lambda_val_l391_391620

variable (p : ℝ) (λ : ℝ)
variable (A F M N B : Point)
variable (ellipse_focus_eq : F = (1, 0))
variable (parabola_eq : ∀ x y, y^2 = 2 * p * x → p > 0)
variable (point_A : A = (x_0, 2))
variable (line_l_intersect_parabola : line_l l F -> intersects parabola_C at M N)
variable (directrix_intersect_x_axis : directrix parabola_C -> intersects x_axis at B)
variable (MF_FN_relation : vector MF = λ * vector FN)
variable (BM_BN_squares_sum : dist_sq B M + dist_sq B N = 40)

theorem find_equation_of_parabola_and_AF
  : (parabola_C_eqn = (λ x y, y^2 = 4 * x) ∧ |AF| = 2) :=
  sorry

theorem find_lambda_val
  : (λ = 2 + sqrt 3) ∨ (λ = 2 - sqrt 3) :=
  sorry

end find_equation_of_parabola_and_AF_find_lambda_val_l391_391620


namespace stationery_store_profit_l391_391501

variable (a : ℝ)

def store_cost : ℝ := 100 * a
def markup_price : ℝ := a * 1.2
def discount_price : ℝ := markup_price a * 0.8

def revenue_first_half : ℝ := 50 * markup_price a
def revenue_second_half : ℝ := 50 * discount_price a
def total_revenue : ℝ := revenue_first_half a + revenue_second_half a

def profit : ℝ := total_revenue a - store_cost a

theorem stationery_store_profit : profit a = 8 * a := 
by sorry

end stationery_store_profit_l391_391501


namespace greatest_prime_factor_15_17_factorial_l391_391346

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391346


namespace greatest_prime_factor_of_sum_factorials_l391_391284

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391284


namespace soccer_players_arrangement_l391_391523

theorem soccer_players_arrangement : ∃ (x1 x2 x3 x4 : ℝ), 
    let dists := {(abs (x1 - x2)), (abs (x1 - x3)), (abs (x1 - x4)), (abs (x2 - x3)), (abs (x2 - x4)), (abs (x3 - x4))} in
    dists = {1, 2, 3, 4, 5, 6} :=
sorry

end soccer_players_arrangement_l391_391523


namespace chests_contents_l391_391791

-- Define the coin types
inductive CoinType
| Gold
| Silver
| Copper

open CoinType

-- Define the chests
def Chest : Type

-- Define the inscriptions on the chests (all of which are known to be incorrect)
def chest_1_inscription : Prop := ∀ c : Chest, c = Gold
def chest_2_inscription : Prop := ∀ c : Chest, c = Silver
def chest_3_inscription : Prop := ∀ c : Chest, c = Gold ∨ c = Silver

-- Define the truth about what the chests must contain
def contains_gold (c : Chest) : Prop
def contains_silver (c : Chest) : Prop
def contains_copper (c : Chest) : Prop

-- The conditions given in the problem, stating that the true contents are different from the inscriptions
axiom incorrect_inscriptions :
  (¬ chest_1_inscription) ∧
  (¬ chest_2_inscription) ∧
  (¬ chest_3_inscription)

-- The problem states we have exactly one chest of each type 
axiom one_gold : ∃ c : Chest, contains_gold c
axiom one_silver : ∃ c : Chest, contains_silver c
axiom one_copper : ∃ c : Chest, contains_copper c

-- The main theorem we need to prove
theorem chests_contents :
  (∀ c : Chest, contains_silver c ↔ c = Chest₁) ∧
  (∀ c : Chest, contains_gold c ↔ c = Chest₂) ∧
  (∀ c : Chest, contains_copper c ↔ c = Chest₃) :=
sorry

end chests_contents_l391_391791


namespace greatest_prime_factor_15_fact_17_fact_l391_391233

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391233


namespace point_region_l391_391609

theorem point_region (m n : ℝ) (h : 2 ^ m + 2 ^ n < 4) : m + n < 2 := sorry

end point_region_l391_391609


namespace tens_digit_of_8_pow_2023_l391_391912

theorem tens_digit_of_8_pow_2023 :
    ∃ d, 0 ≤ d ∧ d < 10 ∧ (8^2023 % 100) / 10 = d ∧ d = 1 :=
by
  sorry

end tens_digit_of_8_pow_2023_l391_391912


namespace find_q_and_general_term_sum_first_n_seqB_l391_391626

-- Defining the sequence {a_n}
def seqA (n : ℕ) (q : ℝ) : ℕ → ℝ
| 1 => 1
| 2 => 2
| (n + 2) => q * seqA n

-- Conditions on the sequence a_n
def is_arithmetic (a2 a3 a4 a5 : ℝ) : Prop :=
  (a4 - a2) = (a5 - a3)

-- Defining the value of q and checking the general term formula
theorem find_q_and_general_term 
  (hä : seqA 1 q = 1)
  (hát : seqA 2 q = 2)
  (h_arith : is_arithmetic (seqA 2 q) (seqA 3 q) (seqA 4 q) (seqA 5 q))
  (hq : q ≠ 1) :
  q = 2 ∧ ∀ n, seqA n 2 = if n % 2 = 1 then 2^((n-1)/2) else 2^(n/2) :=
sorry

-- Defining sequence {b_n}
def seqB (n : ℕ) : ℝ :=
  (Real.log 4 (seqA (2 * n) 2)) / seqA (2 * n - 1) 2

-- Defining the sum of first n terms of sequence {b_n}
def sumSeqB (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, seqB (i + 1)

-- Theorem stating the closed form of sum {b_n}
theorem sum_first_n_seqB (n : ℕ) :
  sumSeqB n = 2 - (n + 2) / (2^n) :=
sorry

end find_q_and_general_term_sum_first_n_seqB_l391_391626


namespace koschei_chests_l391_391754

theorem koschei_chests :
  ∃ (contents : Fin 3 → String), 
    -- All chests labels are incorrect
    (contents 0 ≠ "gold coins" ∧ contents 1 ≠ "silver coins" ∧ contents 2 ≠ "gold or silver coins") ∧ 
    -- Each chest contains exactly one type of coin 
    (∀ i j : Fin 3, i ≠ j → contents i ≠ contents j) ∧ 
    -- Providing the final conclusion about what is in each chest
    (contents 0 = "silver coins" ∧ contents 1 = "gold coins" ∧ contents 2 = "copper coins") :=
begin
  use (λ k, if k = 0 then "silver coins" else if k = 1 then "gold coins" else "copper coins"),
  split,
  {
    -- Proof of all labels being incorrect
    split; simp,
  },
  split,
  {
    -- Proof of each chest containing a unique type of coin
    intros i j h,
    cases i; cases j; simp [h],
  },
  {
    -- Proof of the final conclusion
    split; split; simp,
  },
end

end koschei_chests_l391_391754


namespace infinite_double_perfect_squares_l391_391494

def is_double_number (n : ℕ) : Prop :=
  ∃ k m : ℕ, m > 0 ∧ n = m * 10^k + m

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem infinite_double_perfect_squares : ∀ n : ℕ, ∃ m, n < m ∧ is_double_number m ∧ is_perfect_square m :=
  sorry

end infinite_double_perfect_squares_l391_391494


namespace petya_has_19_friends_l391_391053

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l391_391053


namespace petya_friends_l391_391039

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l391_391039


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391374

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391374


namespace greatest_prime_factor_of_factorial_sum_l391_391333

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391333


namespace Carla_total_marbles_l391_391575

def initial_marbles : ℝ := 187.0
def bought_marbles : ℝ := 134.0

theorem Carla_total_marbles : initial_marbles + bought_marbles = 321.0 := 
by 
  sorry

end Carla_total_marbles_l391_391575


namespace range_of_a_fall_within_D_l391_391707

-- Define the conditions
variable (a : ℝ) (c : ℝ)
axiom A_through : c = 9
axiom D_through : a < 0 ∧ (6, 7) ∈ { (x, y) | y = a * x ^ 2 + c }

-- Prove the range of a given the conditions
theorem range_of_a : -1/4 < a ∧ a < -1/18 := sorry

-- Define the additional condition for point P
axiom P_through : (2, 8.1) ∈ { (x, y) | y = a * x ^ 2 + c }

-- Prove that the object can fall within interval D when passing through point P
theorem fall_within_D : a = -9/40 ∧ -1/4 < a ∧ a < -1/18 := sorry

end range_of_a_fall_within_D_l391_391707


namespace greatest_prime_factor_15_fact_17_fact_l391_391232

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391232


namespace ice_cream_depth_l391_391963

-- Define the initial conditions and variables
def initial_radius_sphere : ℝ := 3
def final_radius_cylinder : ℝ := 12

-- Volume formula for a sphere
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Volume formula for a cylinder
def volume_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Given the density is constant, equate the volumes
theorem ice_cream_depth :
  volume_sphere initial_radius_sphere = volume_cylinder final_radius_cylinder (1 / 4) :=
by sorry

end ice_cream_depth_l391_391963


namespace valid_x_values_l391_391970

noncomputable def valid_triangle_sides (x : ℕ) : Prop :=
  8 + 11 > x + 3 ∧ 8 + (x + 3) > 11 ∧ 11 + (x + 3) > 8

theorem valid_x_values :
  {x : ℕ | valid_triangle_sides x ∧ x > 0} = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} :=
by
  sorry

end valid_x_values_l391_391970


namespace petya_friends_l391_391069

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l391_391069


namespace complex_z_modulus_l391_391645

open Complex

theorem complex_z_modulus (z : ℂ) (h1 : (z + 2 * I).re = z + 2 * I) (h2 : (z / (2 - I)).re = z / (2 - I)) :
  (z = 4 - 2 * I) ∧ abs (z / (1 + I)) = Real.sqrt 10 := by
  sorry

end complex_z_modulus_l391_391645


namespace contradiction_assumption_l391_391466

theorem contradiction_assumption (a b c : ℕ) :
  (∃ k : ℕ, (k = a ∨ k = b ∨ k = c) ∧ ∃ n : ℕ, k = 2 * n + 1) →
  (∃ k1 k2 : ℕ, (k1 = a ∨ k1 = b ∨ k1 = c) ∧ (k2 = a ∨ k2 = b ∨ k2 = c) ∧ k1 ≠ k2 ∧ ∃ n1 n2 : ℕ, k1 = 2 * n1 ∧ k2 = 2 * n2) ∨
  (∀ k : ℕ, (k = a ∨ k = b ∨ k = c) → ∃ n : ℕ, k = 2 * n + 1) :=
sorry

end contradiction_assumption_l391_391466


namespace greatest_prime_factor_15_17_factorial_l391_391212

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391212


namespace fraction_pow_l391_391983

def a : ℕ := 88_888
def b : ℕ := 22_222

theorem fraction_pow : (a^5) / (b^5) = 1024 := by
  sorry

end fraction_pow_l391_391983


namespace chests_content_l391_391746

-- Define the chests and their labels
inductive CoinContent where
  | gold : CoinContent
  | silver : CoinContent
  | copper : CoinContent

structure Chest where
  label : CoinContent
  contents : CoinContent

-- Given conditions and incorrect labels
def chest1 : Chest := { label := CoinContent.gold, contents := sorry }
def chest2 : Chest := { label := CoinContent.silver, contents := sorry }
def chest3 : Chest := { label := CoinContent.gold, contents := sorry }

-- The proof problem
theorem chests_content :
  chest1.contents ≠ CoinContent.gold ∧
  chest2.contents ≠ CoinContent.silver ∧
  chest3.contents ≠ CoinContent.gold ∨ chest3.contents ≠ CoinContent.silver →
  chest1.contents = CoinContent.silver ∧
  chest2.contents = CoinContent.gold ∧
  chest3.contents = CoinContent.copper := by
  sorry

end chests_content_l391_391746


namespace functional_equation_solution_l391_391996

theorem functional_equation_solution
  (a : ℤ) (f g : ℚ → ℚ)
  (h : ∀ x y : ℚ, f (x + g y) = g x + f y + a * y) :
  ∃ n : ℤ, n ≠ 0 ∧ n ≠ 1 ∧ a = n^2 - n ∧ 
    (∀ v : ℚ, (f = λ x, n * x + v ∧ g = λ x, n * x) ∨ 
              (f = λ x, (1 - n) * x + v ∧ g = λ x, (1 - n) * x)) :=
sorry

end functional_equation_solution_l391_391996


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391254

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391254


namespace least_positive_integer_l391_391910

theorem least_positive_integer (n : ℕ) (h1 : n > 1) 
  (h2 : n % 2 = 1) (h3 : n % 3 = 1) (h4 : n % 5 = 1) 
  (h5 : n % 7 = 1) (h6 : n % 11 = 1): 
  n = 2311 := 
by
  sorry

end least_positive_integer_l391_391910


namespace find_natural_pairs_l391_391997

theorem find_natural_pairs (a b : ℕ) :
  (∃ A, A * A = a ^ 2 + 3 * b) ∧ (∃ B, B * B = b ^ 2 + 3 * a) ↔ 
  (a = 1 ∧ b = 1) ∨ (a = 11 ∧ b = 11) ∨ (a = 16 ∧ b = 11) :=
by
  sorry

end find_natural_pairs_l391_391997


namespace f_leq_2x_l391_391015

variable {α : Type*} [linear_ordered_comm_ring α] [archimedean α] [floor_ring α]

noncomputable def f : α → α := sorry

axiom f_domain (x : α) : 0 ≤ x ∧ x ≤ 1
axiom f_positive (x : α) : 0 < f x
axiom f_boundary : f 1 = 1
axiom f_superadditive (x y : α) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) :
  f (x + y) ≥ f x + f y

theorem f_leq_2x {x : α} (hx : 0 ≤ x ∧ x ≤ 1) : f x ≤ 2 * x := sorry

end f_leq_2x_l391_391015


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391435

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391435


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391266

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391266


namespace max_value_sqrt_sum_l391_391810

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x + y + z = 2)
  (h2 : x ≥ -1/2) (h3 : y ≥ -3/2) (h4 : z ≥ -1) :
  sqrt (3 * x + 1.5) + sqrt (3 * y + 4.5) + sqrt (3 * z + 3) ≤ 9 := 
  sorry

end max_value_sqrt_sum_l391_391810


namespace length_of_AB_l391_391948

noncomputable def circle1 : set (ℝ × ℝ) := { p | (p.1 - 1)^2 + p.2^2 = 1 }
def point_P := (3, 1)
def point_M := (1, 0)

def is_tangent (circle : set (ℝ × ℝ)) (p1 p2 : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, (tangent_line circle p1) = (tangent_line circle p2)

theorem length_of_AB :
  ∀ (circle2 : set (ℝ × ℝ)) (A B : ℝ × ℝ),
    (point_P ∈ circle2) ∧ is_tangent circle2 A B ∧ is_tangent circle1 A B →
    dist A B = 4 * real.sqrt 5 / 5 :=
sorry

end length_of_AB_l391_391948


namespace inequality_A_inequality_B_inequality_C_inequality_D_l391_391915

section

variable {a b x y : ℝ}

theorem inequality_A (ha : 0 < a) (hb : 0 < b) (h : a + b = 3) :
  (∃ m, (∀ a b : ℝ, 0 < a → 0 < b → a + b = 3 → (1 / (a + 1) + 1 / (b + 2)) ≥ m) ∧ m = 2 / 3) :=
sorry

theorem inequality_B :
  ¬ (∃ m, (∀ x : ℝ, let y := (sqrt (x^2 + 2) + 1 / sqrt (x^2 + 2)) in y ≥ m) ∧ m = 2) :=
sorry

theorem inequality_C (hx : 0 < x) (hy : 0 < y) (h : x + y = 2) :
  (∃ m, (∀ x y : ℝ, 0 < x → 0 < y → x + y = 2 → (x * y) ≤ m) ∧ m = 1) :=
sorry

theorem inequality_D (x : ℝ) (hx : 0 < x) (h : ∀ x > 0, x^3 + 5 * x^2 + 4 * x ≥ a * x^2) :
  a ≤ 9 :=
sorry

end

end inequality_A_inequality_B_inequality_C_inequality_D_l391_391915


namespace smallest_positive_integer_n_l391_391459

theorem smallest_positive_integer_n :
  ∃ n: ℕ, (n > 0) ∧ (∀ k: ℕ, 1 ≤ k ∧ k ≤ n → (∃ d: ℕ, d ∣ (n^2 - 2 * n) ∧ d ∣ k) ∧ (k ∣ (n^2 - 2 * n) → k = d)) ∧ n = 5 :=
by
  sorry

end smallest_positive_integer_n_l391_391459


namespace chest_contents_l391_391775

def Chest : Type := ℕ
def Coins : Type := ℕ

variable (empty : Coins → Coins → Prop)

noncomputable def Chest1 : Chest := 1
noncomputable def Chest2 : Chest := 2
noncomputable def Chest3 : Chest := 3

variable (goldCoins : Coins)
variable (silverCoins : Coins)
variable (copperCoins : Coins)

-- Conditions
variable (labelsIncorrect : Chest → Coins → Prop)

axiom label1 : labelsIncorrect Chest1 goldCoins
axiom label2 : labelsIncorrect Chest2 silverCoins
axiom label3 : labelsIncorrect Chest3 (goldCoins ∨ silverCoins)

axiom uniqueContents : ∀ c : Chest, c = Chest1 ∨ c = Chest2 ∨ c = Chest3
axiom distinctContents : ∀ c1 c2 : Chest, c1 ≠ c2 → (c1 = Chest1 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest3) ∨ (c1 = Chest2 ∧ c2 ≠ Chest2 ∧ c2 ≠ Chest3) ∨ (c1 = Chest3 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest2)

theorem chest_contents : (exists! c1 : Coins, c1 = silverCoins ∧ labelsIncorrect Chest1 c1) ∧
                         (exists! c2 : Coins, c2 = goldCoins ∧ labelsIncorrect Chest2 c2) ∧
                         (exists! c3 : Coins, c3 = copperCoins ∧ labelsIncorrect Chest3 c3)
                         :=
by
  sorry

end chest_contents_l391_391775


namespace petya_friends_l391_391092

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l391_391092


namespace chests_contents_l391_391796

-- Define the coin types
inductive CoinType
| Gold
| Silver
| Copper

open CoinType

-- Define the chests
def Chest : Type

-- Define the inscriptions on the chests (all of which are known to be incorrect)
def chest_1_inscription : Prop := ∀ c : Chest, c = Gold
def chest_2_inscription : Prop := ∀ c : Chest, c = Silver
def chest_3_inscription : Prop := ∀ c : Chest, c = Gold ∨ c = Silver

-- Define the truth about what the chests must contain
def contains_gold (c : Chest) : Prop
def contains_silver (c : Chest) : Prop
def contains_copper (c : Chest) : Prop

-- The conditions given in the problem, stating that the true contents are different from the inscriptions
axiom incorrect_inscriptions :
  (¬ chest_1_inscription) ∧
  (¬ chest_2_inscription) ∧
  (¬ chest_3_inscription)

-- The problem states we have exactly one chest of each type 
axiom one_gold : ∃ c : Chest, contains_gold c
axiom one_silver : ∃ c : Chest, contains_silver c
axiom one_copper : ∃ c : Chest, contains_copper c

-- The main theorem we need to prove
theorem chests_contents :
  (∀ c : Chest, contains_silver c ↔ c = Chest₁) ∧
  (∀ c : Chest, contains_gold c ↔ c = Chest₂) ∧
  (∀ c : Chest, contains_copper c ↔ c = Chest₃) :=
sorry

end chests_contents_l391_391796


namespace koschei_chests_l391_391751

theorem koschei_chests :
  ∃ (contents : Fin 3 → String), 
    -- All chests labels are incorrect
    (contents 0 ≠ "gold coins" ∧ contents 1 ≠ "silver coins" ∧ contents 2 ≠ "gold or silver coins") ∧ 
    -- Each chest contains exactly one type of coin 
    (∀ i j : Fin 3, i ≠ j → contents i ≠ contents j) ∧ 
    -- Providing the final conclusion about what is in each chest
    (contents 0 = "silver coins" ∧ contents 1 = "gold coins" ∧ contents 2 = "copper coins") :=
begin
  use (λ k, if k = 0 then "silver coins" else if k = 1 then "gold coins" else "copper coins"),
  split,
  {
    -- Proof of all labels being incorrect
    split; simp,
  },
  split,
  {
    -- Proof of each chest containing a unique type of coin
    intros i j h,
    cases i; cases j; simp [h],
  },
  {
    -- Proof of the final conclusion
    split; split; simp,
  },
end

end koschei_chests_l391_391751


namespace intersect_asymptotes_l391_391999

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 3) / (x^3 - 4*x^2 + 4*x)

theorem intersect_asymptotes : 
  (0, 0) ∈ set_of (λ p : ℝ × ℝ, p.2 = f p.1 ∧ (p.1 = 0 ∨ p.1 = 2)) ∧
  (2, 0) ∈ set_of (λ p : ℝ × ℝ, p.2 = f p.1 ∧ (p.1 = 0 ∨ p.1 = 2)) :=
sorry

end intersect_asymptotes_l391_391999


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391287

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391287


namespace min_value_of_geometric_sequence_l391_391002

theorem min_value_of_geometric_sequence :
  ∃ (s : ℝ), let b1 := 2 in
             let b2 := b1 * s in
             let b3 := b2 * s in
             3 * b2 + 6 * b3 = -3 / 4 :=
by {
  sorry -- Proof goes here
}

end min_value_of_geometric_sequence_l391_391002


namespace greatest_prime_factor_of_factorial_sum_l391_391326

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391326


namespace trajectory_midpoint_chord_l391_391129

theorem trajectory_midpoint_chord (x y : ℝ) 
  (h₀ : y^2 = 4 * x) : (y^2 = 2 * x - 2) :=
sorry

end trajectory_midpoint_chord_l391_391129


namespace even_integers_count_l391_391682

def is_even (n : ℕ) : Prop := n % 2 = 0
def has_four_different_digits (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, n = 1000 * a + 100 * b + 10 * c + d ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem even_integers_count : 
  let nums := {n | 5000 ≤ n ∧ n < 8000 ∧ is_even n ∧ has_four_different_digits n} in
  nums.card = 784 :=
by 
  sorry

end even_integers_count_l391_391682


namespace min_distance_from_point_to_tangent_l391_391505

theorem min_distance_from_point_to_tangent 
  : (∀ (x y : ℝ), (y = x + 1) → (∃ (p : ℝ × ℝ), (p.1 = 3 ∧ p.2 = 0) ∧ (p.dist (x, y))^2 = 2 * (x - 3)^2 + (y^2) - 1) → min_dist = sqrt (7)) :=
begin
  sorry
end

end min_distance_from_point_to_tangent_l391_391505


namespace petya_friends_l391_391068

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l391_391068


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391368

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391368


namespace Mahdi_swims_on_Wednesday_or_Sunday_l391_391817

-- Definitions for the days of the week
inductive Day
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr

-- Assign known sports to specific days
def sportDay : Day → Prop
| Day.Monday := "cycling"
| Day.Tuesday := "basketball"
| Day.Saturday := "golf"
| _ := "unknown"

-- Constraints
axiom runs_three_days_a_week : ∃ days : Fin 7 → Day, (∃ d1 d2 d3, d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ ∀ d, d ≠ Day.Monday → d ≠ Day.Tuesday → d ≠ Day.Saturday)
axiom runs_non_consecutive : ∀ d1 d2, sportDay d1 = "running" ∧ sportDay d2 = "running" → abs (Day.index d1 - Day.index d2) > 1
axiom not_play_tennis_after_running : ∀ d1 d2, sportDay d1 = "running" ∧ sportDay d2 = "tennis" → d2 ≠ Day.succ d1
axiom not_play_tennis_after_cycling : sportDay (Day.succ Day.Monday) ≠ "tennis"

-- Prove Mahdi swims on Wednesday or Sunday given configurations
theorem Mahdi_swims_on_Wednesday_or_Sunday :
  ∃ days : Fin 7 → Day, 
    (sportDay days !4 = "swimming" ∧ sportDay days !1 = "tennis") ∨
    (sportDay days !0 = "swimming" ∧ sportDay days !3 = "tennis") :=
sorry

end Mahdi_swims_on_Wednesday_or_Sunday_l391_391817


namespace chests_content_l391_391745

-- Define the chests and their labels
inductive CoinContent where
  | gold : CoinContent
  | silver : CoinContent
  | copper : CoinContent

structure Chest where
  label : CoinContent
  contents : CoinContent

-- Given conditions and incorrect labels
def chest1 : Chest := { label := CoinContent.gold, contents := sorry }
def chest2 : Chest := { label := CoinContent.silver, contents := sorry }
def chest3 : Chest := { label := CoinContent.gold, contents := sorry }

-- The proof problem
theorem chests_content :
  chest1.contents ≠ CoinContent.gold ∧
  chest2.contents ≠ CoinContent.silver ∧
  chest3.contents ≠ CoinContent.gold ∨ chest3.contents ≠ CoinContent.silver →
  chest1.contents = CoinContent.silver ∧
  chest2.contents = CoinContent.gold ∧
  chest3.contents = CoinContent.copper := by
  sorry

end chests_content_l391_391745


namespace chest_contents_solution_l391_391785

-- Definitions corresponding to the conditions.
structure ChestContents :=
  (chest1 chest2 chest3 : String)

-- Given conditions
def labelsAreIncorrect (contents : ChestContents) : Prop :=
  contents.chest1 ≠ "Gold coins" ∧
  contents.chest2 ≠ "Silver coins" ∧
  contents.chest3 ≠ "Gold coins" ∧
  contents.chest3 ≠ "Silver coins"

def uniqueCoins (contents : ChestContents) : Prop :=
  (contents.chest1 = "Gold coins" ∨ contents.chest1 = "Silver coins" ∨ contents.chest1 = "Copper coins") ∧
  (contents.chest2 = "Gold coins" ∨ contents.chest2 = "Silver coins" ∨ contents.chest2 = "Copper coins") ∧
  (contents.chest3 = "Gold coins" ∨ contents.chest3 = "Silver coins" ∨ contents.chest3 = "Copper coins") ∧
  (contents.chest1 ≠ contents.chest2) ∧
  (contents.chest1 ≠ contents.chest3) ∧
  (contents.chest2 ≠ contents.chest3)

-- The proof statement
theorem chest_contents_solution : ∃ (contents : ChestContents),
  labelsAreIncorrect contents ∧ uniqueCoins contents ∧
  contents.chest1 = "Silver coins" ∧
  contents.chest2 = "Gold coins" ∧
  contents.chest3 = "Copper coins" :=
begin
  sorry
end

end chest_contents_solution_l391_391785


namespace soccer_player_positions_exist_l391_391526

theorem soccer_player_positions_exist :
  ∃ x1 x2 x3 x4 : ℝ,
    ({| real.abs (x1 - x2),
       real.abs (x1 - x3),
       real.abs (x1 - x4),
       real.abs (x2 - x3),
       real.abs (x2 - x4),
       real.abs (x3 - x4) |} = {| 1, 2, 3, 4, 5, 6 |}) :=
begin
  use [0, 1, 4, 6],
  sorry
end

end soccer_player_positions_exist_l391_391526


namespace four_digit_number_difference_l391_391919

theorem four_digit_number_difference (a : ℤ) : 
  let original := 1000 * a + 100 * (a + 1) + 10 * (a + 2) + (a + 3)
  let reversed := 1000 * (a + 3) + 100 * (a + 2) + 10 * (a + 1) + a
  reversed - original = 3087 :=
by
  let original := 1000 * a + 100 * (a + 1) + 10 * (a + 2) + (a + 3)
  let reversed := 1000 * (a + 3) + 100 * (a + 2) + 10 * (a + 1) + a
  calc
    reversed - original = (1000 * (a + 3) + 100 * (a + 2) + 10 * (a + 1) + a) - 
                         (1000 * a + 100 * (a + 1) + 10 * (a + 2) + (a + 3)) : by rfl
    ... = (1000 * a + 3000 + 100 * a + 200 + 10 * a + 10 + a) - 
          (1000 * a + 100 * a + 100 + 10 * a + 20 + a + 3) : by ring
    ... = 3210 - 123 : by ring
    ... = 3087 : by norm_num

end four_digit_number_difference_l391_391919


namespace minimal_sum_factorial_product_l391_391867

theorem minimal_sum_factorial_product :
  ∀ (p q r s : ℕ), (p * q * r * s = nat.factorial 12) → 
  (p + q + r + s ≥ 1613) := 
sorry

end minimal_sum_factorial_product_l391_391867


namespace total_bill_l391_391854

theorem total_bill (n : ℝ) (h : 9 * (n / 10 + 3) = n) : n = 270 := 
sorry

end total_bill_l391_391854


namespace perp_DE_EF_l391_391097

-- Definitions corresponding to the conditions
variables {A B C D E F: Type} [inner_product_space ℝ Type]
variable (A B C D E F: Type)
noncomputable def is_midpoint (E : Type) (B C : Type): Prop := E = ((B + C) / 2)
noncomputable def is_divide (F AC : Type): Prop := F = ((2 * AC + 1) / 3)
noncomputable def angle_eq_30_deg (D A C : Type): Prop := ∠D A C = 30
noncomputable def angle_eq_60_deg (D B A : Type): Prop := ∠D B A = 60

-- Lean 4 statement
theorem perp_DE_EF
    (h1 : angle_eq_30_deg D A C)
    (h2 : angle_eq_30_deg D C A)
    (h3 : angle_eq_60_deg D B A)
    (h4 : is_midpoint E B C)
    (h5 : is_divide F A C)
    : inner_product_space.perp DE EF :=
    sorry

end perp_DE_EF_l391_391097


namespace shaded_area_ratio_l391_391899

theorem shaded_area_ratio (s : ℝ) (ABC : Triangle)
  (h_eq_sides : ABC.is_equilateral s)
  (D E F : Point)
  (h_midpoints_D : D.is_midpoint (ABC.ABC_side AB))
  (h_midpoints_E : E.is_midpoint (ABC.ABC_side BC))
  (h_midpoints_F : F.is_midpoint (ABC.ABC_side CA))
  (G H : Point)
  (h_midpoint_G : G.is_midpoint (segment DF))
  (h_midpoint_H : H.is_midpoint (segment FE)) :
  let A_ABC := (ABC.area s)
  let A_DEF := A_ABC / 4
  let A_DGF := A_DEF / 4
  (A_DEF - A_DGF) / (A_ABC - (A_DEF - A_DGF)) = 3 / 13 :=
sorry

end shaded_area_ratio_l391_391899


namespace soccer_players_positions_l391_391533

noncomputable def positions : list ℝ := [0, 1, 4, 6]

def pairwise_distances (positions : list ℝ) : list ℝ :=
  let pairs := list.sigma positions positions
  let distances := pairs.map (λ p, abs (p.1 - p.2))
  distances.erase_dup

theorem soccer_players_positions :
  pairwise_distances positions = [1, 2, 3, 4, 5, 6] :=
by {
  sorry  -- Proof to be provided
}

end soccer_players_positions_l391_391533


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391379

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391379


namespace greatest_prime_factor_15_fact_17_fact_l391_391169

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391169


namespace petya_friends_l391_391093

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l391_391093


namespace arrangement_count_l391_391713

theorem arrangement_count : 
  ∀ (d1 d2 d3 d4 d5 : ℕ), 
    multiset.mem d1 {5, 8, 0, 7, 0} ∧
    multiset.mem d2 {5, 8, 0, 7, 0} ∧
    multiset.mem d3 {5, 8, 0, 7, 0} ∧
    multiset.mem d4 {5, 8, 0, 7, 0} ∧
    multiset.mem d5 {5, 8, 0, 7, 0} ∧
    multiset.card {d1, d2, d3, d4, d5} = 5 ∧
    d1 ≠ 0 → 
    (∑ (perm : {d1, d2, d3, d4, d5} : multiset ℕ) in {multiset.perm {5, 8, 0, 7, 0}}, 1) = 96 :=
by sorry

end arrangement_count_l391_391713


namespace circumcenter_pass_through_triangles_l391_391031

theorem circumcenter_pass_through_triangles
  (A B C A1 B1 C1 O : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace A1] [MetricSpace B1] [MetricSpace C1] [MetricSpace O]
  (hA1 : SameSide A B C A1)
  (hB1 : SameSide A C B B1)
  (hC1 : SameSide B A C C1)
  (h_similar : Similar A B C A1 B1 C1)
  (h_opposite : OppositeOrientation A B C A1 B1 C1)
  (h_O : IsCircumcenter A B C O) :
  PassThroughCircumcenter (Circumcircle A B1 C1) O ∧
  PassThroughCircumcenter (Circumcircle A1 B C1) O ∧
  PassThroughCircumcenter (Circumcircle A1 B1 C) O :=
begin
  sorry
end

end circumcenter_pass_through_triangles_l391_391031


namespace volume_ratio_is_four_thirds_l391_391121

def diameter_sphere : ℝ := 6
def diameter_hemisphere : ℝ := 12

def radius_sphere := diameter_sphere / 2
def radius_hemisphere := diameter_hemisphere / 2

def volume_sphere := (4 / 3) * Real.pi * (radius_sphere ^ 3)
def volume_hemisphere := (1 / 2) * (4 / 3) * Real.pi * (radius_hemisphere ^ 3)
def volume_cylinder := Real.pi * (radius_hemisphere ^ 2) * diameter_hemisphere

def combined_volume := volume_sphere + volume_hemisphere
def ratio := volume_cylinder / combined_volume

theorem volume_ratio_is_four_thirds :
  ratio = (4 / 3) :=
  sorry

end volume_ratio_is_four_thirds_l391_391121


namespace even_integers_between_5000_8000_with_four_distinct_digits_l391_391679

theorem even_integers_between_5000_8000_with_four_distinct_digits : 
  let count := (sum (λ d₁, if d₁ ∈ [5, 6, 7] then
     (sum (λ d₄, if d₄ ≠ d₁ ∧ d₄ % 2 = 0 then
     (sum (λ d₂, if d₂ ≠ d₁ ∧ d₂ ≠ d₄ then
     (sum (λ d₃, if d₃ ≠ d₁ ∧ d₃ ≠ d₂ ∧ d₃ ≠ d₄ then 1 else 0))
     else 0))
     else 0))
     else 0) 
  in count = 728 := 
by
  sorry

end even_integers_between_5000_8000_with_four_distinct_digits_l391_391679


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391257

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391257


namespace arithmetic_sequence_30th_term_l391_391863

theorem arithmetic_sequence_30th_term 
  (a1 a2 a3 : ℤ) (h1 : a1 = 3) (h2 : a2 = 17) (h3 : a3 = 31) : 
  let d := a2 - a1 in 
  let aₙ (n : ℕ) := a1 + (n - 1 : ℤ) * d in 
  aₙ 30 = 409 :=
by
  sorry

end arithmetic_sequence_30th_term_l391_391863


namespace max_height_of_A_l391_391508

theorem max_height_of_A 
(V1 V2 : ℝ) (β α g : ℝ) (nonneg_g : 0 < g) :
  let Vc0y := (V1 * Real.sin β + V2 * Real.sin α) / 2 in
  (Vc0y * Vc0y) / (2 * g) =
  (1 / (2 * g)) * (V1 * Real.sin β + V2 * Real.sin α)^2 / 4 :=
by
  let Vc0y := (V1 * Real.sin β + V2 * Real.sin α) / 2
  calc
    (Vc0y * Vc0y) / (2 * g)
      = (Vc0y^2) / (2 * g) : by simp
  ... = Vc0y^2 / 2 / g : by rw [div_div_eq_div_mul]
  ... = (1 / (2 * g)) * Vc0y^2 : by ring
  ... = (1 / (2 * g)) * ((V1 * Real.sin β + V2 * Real.sin α) / 2)^2 : by simp only [Vc0y]
  ... = (1 / (2 * g)) * (V1 * Real.sin β + V2 * Real.sin α)^2 / 4 : by field_simp
  ... = (1 / (2 * g)) * (V1 * Real.sin β + V2 * Real.sin α)^2 / 4 : by ring
  ... = (1 / (2 * g)) * ((V1 * Real.sin β + V2 * Real.sin α) / 2)^2 : by field_simp
  ... = (1 / (2 * g)) * (V1 * Real.sin β / 2 + V2 * Real.sin α / 2)^2 : by ring
  ... = (1 / (2 * g)) * ((V1 * Real.sin β + V2 * Real.sin α) / 2 * (V1 * Real.sin β + V2 * Real.sin α) / 2) : by ring

end max_height_of_A_l391_391508


namespace greatest_prime_factor_of_15_l391_391440

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391440


namespace chest_contents_l391_391778

def Chest : Type := ℕ
def Coins : Type := ℕ

variable (empty : Coins → Coins → Prop)

noncomputable def Chest1 : Chest := 1
noncomputable def Chest2 : Chest := 2
noncomputable def Chest3 : Chest := 3

variable (goldCoins : Coins)
variable (silverCoins : Coins)
variable (copperCoins : Coins)

-- Conditions
variable (labelsIncorrect : Chest → Coins → Prop)

axiom label1 : labelsIncorrect Chest1 goldCoins
axiom label2 : labelsIncorrect Chest2 silverCoins
axiom label3 : labelsIncorrect Chest3 (goldCoins ∨ silverCoins)

axiom uniqueContents : ∀ c : Chest, c = Chest1 ∨ c = Chest2 ∨ c = Chest3
axiom distinctContents : ∀ c1 c2 : Chest, c1 ≠ c2 → (c1 = Chest1 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest3) ∨ (c1 = Chest2 ∧ c2 ≠ Chest2 ∧ c2 ≠ Chest3) ∨ (c1 = Chest3 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest2)

theorem chest_contents : (exists! c1 : Coins, c1 = silverCoins ∧ labelsIncorrect Chest1 c1) ∧
                         (exists! c2 : Coins, c2 = goldCoins ∧ labelsIncorrect Chest2 c2) ∧
                         (exists! c3 : Coins, c3 = copperCoins ∧ labelsIncorrect Chest3 c3)
                         :=
by
  sorry

end chest_contents_l391_391778


namespace molecular_weight_calc_l391_391911

namespace MolecularWeightProof

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00
def number_of_H : ℕ := 1
def number_of_Br : ℕ := 1
def number_of_O : ℕ := 3

theorem molecular_weight_calc :
  (number_of_H * atomic_weight_H + number_of_Br * atomic_weight_Br + number_of_O * atomic_weight_O) = 128.91 :=
by
  sorry

end MolecularWeightProof

end molecular_weight_calc_l391_391911


namespace ski_helmet_final_price_l391_391029

variables (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ)
def final_price_after_discounts (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  let after_first_discount := initial_price * (1 - discount1)
  let after_second_discount := after_first_discount * (1 - discount2)
  after_second_discount

theorem ski_helmet_final_price :
  final_price_after_discounts 120 0.40 0.20 = 57.60 := 
  sorry

end ski_helmet_final_price_l391_391029


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391247

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391247


namespace greatest_prime_factor_15_17_factorial_l391_391203

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391203


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391421

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391421


namespace petya_friends_l391_391046

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l391_391046


namespace greatest_prime_factor_15_17_factorial_l391_391202

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391202


namespace Jeff_running_time_l391_391728

theorem Jeff_running_time :
  ∃ x : ℕ, 3 * x + (x - 20) + (x + 10) = 290 ∧ x = 60 :=
begin
  sorry
end

end Jeff_running_time_l391_391728


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391365

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391365


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391316
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391316


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391433

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391433


namespace six_points_least_ratio_l391_391010

open_locale real

variables {A : Type*} [metric_space A]

theorem six_points_least_ratio {A_1 A_2 A_3 A_4 A_5 A_6 : A} (h_distinct : ∀i j, i ≠ j → dist i j > 0) :
  let D := max (dist A_1 A_2) (max (dist A_2 A_3) (max (dist A_3 A_4) (max (dist A_4 A_5) (max (dist A_5 A_6) (dist A_6 A_1)))))
  in let d := min (dist A_1 A_2) (min (dist A_2 A_3) (min (dist A_3 A_4) (min (dist A_4 A_5) (min (dist A_5 A_6) (dist A_6 A_1))))) 
  in D / d ≥ real.sqrt 3 :=
  sorry

end six_points_least_ratio_l391_391010


namespace greatest_prime_factor_of_sum_l391_391389

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391389


namespace paperboy_delivery_ways_l391_391493

noncomputable def E : ℕ → ℕ
| 0     := 1 -- Not used, for generality of the definition
| 1     := 2
| 2     := 4
| 3     := 8
| 4     := 15
| (n+5) := E n + E (n+1) + E (n+2) + E (n+3)

theorem paperboy_delivery_ways : E 12 = 2872 :=
by {
  have E_5 : E 5 = 29 := rfl,
  have E_6 : E 6 = 56 := rfl,
  have E_7 : E 7 = 108 := rfl,
  have E_8 : E 8 = 208 := rfl,
  have E_9 : E 9 = 401 := rfl,
  have E_10 : E 10 = 773 := rfl,
  have E_11 : E 11 = 1490 := rfl,
  exact rfl
}

end paperboy_delivery_ways_l391_391493


namespace functional_eq_f_l391_391017

noncomputable def f (q : ℚ) : ℚ := sorry

theorem functional_eq_f
  (f : ℚ → ℚ)
  (h_f : ∀ x y : ℚ, 0 < x → 0 < y → f(x * f(y)) = f(x) / y) :
  ∃ f : ℚ → ℚ, ∀ x y : ℚ, 0 < x → 0 < y → f(x * f(y)) = f(x) / y :=
sorry

end functional_eq_f_l391_391017


namespace greatest_prime_factor_of_sum_factorials_l391_391285

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391285


namespace incorrect_statement_l391_391917

theorem incorrect_statement (a b c : ℝ) (h : a > b) : ¬ (a > b → a * c > b * c) :=
by {
  intro h1,
  -- We assume the opposite for contradiction, that for all positive a, b and any c, if a > b then a * c > b * c
  have h2 : a * (-1) > b * (-1),
  { exact h1 (-1) h,
  },
  -- This implies that -a > -b or a < b, a contradiction
  exact lt_asymm h h2,
}

end incorrect_statement_l391_391917


namespace sufficient_but_not_necessary_condition_l391_391937

theorem sufficient_but_not_necessary_condition (a : ℝ) (m : ℝ) (h: a > 1) :
  (∀ a > 1, log a 2 + log 2 a ≥ m) → m = 2 :=
by
  sorry

end sufficient_but_not_necessary_condition_l391_391937


namespace greatest_prime_factor_15_fact_17_fact_l391_391172

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391172


namespace trigonometric_ceva_theorem_l391_391104

theorem trigonometric_ceva_theorem
  (A B C P Q R : Type*)
  [T1:(Triangle A B C)]
  [T2:(PointOnLineSegment BC P)]
  [T3:(PointOnLineSegment CA Q)]
  [T4:(PointOnLineSegment AB R)]
  (h : AreConcurrent (LineThrough A P) (LineThrough B Q) (LineThrough C R)) :
  (sin (Angle BAP) / sin (Angle PAC)) * (sin (Angle CBQ) / sin (Angle QBA)) * (sin (Angle ACR) / sin (Angle RCB)) = 1 := 
sorry

end trigonometric_ceva_theorem_l391_391104


namespace angle_BAH_in_equilateral_triangle_and_rectangle_l391_391517

theorem angle_BAH_in_equilateral_triangle_and_rectangle 
  (A B C F G H : Type)
  [triangle ABC]
  [rectangle BCFG]
  (BCFG_twice_as_long : 2 * (BCFG.width) = BCFG.length)
  (equilateral_ABC : ∀ x y z, is_equilateral_triangle ABC)
  (midpoint_H : ∀ x y, is_midpoint H CG)
  : ∠BAH = 60 :=
by
  sorry

end angle_BAH_in_equilateral_triangle_and_rectangle_l391_391517


namespace locus_intersection_hyperbola_l391_391153

theorem locus_intersection_hyperbola 
  (O O' : Type*)
  (S : O)
  (circle1 : ∀ x : O, dist x O = R)
  (circle2 : ∀ x : O', dist x O' = r)
  (secant : S → O × O)
  (A B : O)
  (A' B' : O') :
  let M := intersection (line_through O A) (line_through O' B') in
  ∃ (c : ℝ), dist M O - dist M O' = c := 
sorry

end locus_intersection_hyperbola_l391_391153


namespace petya_friends_count_l391_391058

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l391_391058


namespace find_lambda_l391_391979

variables {a b : EuclideanSpace ℝ (Fin 3)}
variables {lambda : ℝ}

theorem find_lambda 
  (h : ∀ (a b : EuclideanSpace ℝ (Fin 3)) (λ : ℝ), 
  inner a (a + λ • b) = 0) :
  λ = -2 := 
sorry

end find_lambda_l391_391979


namespace sum_of_squares_is_382_l391_391482

-- Definitions and conditions
variables (k1 k2 k3 k4 k5 k6 : ℤ)
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def odd_numbers := [k1, k2, k3, k4, k5, k6].map (λ k, 2 * k + 1)

-- The condition that their sum is 42
def sum_condition : Prop := (odd_numbers k1 k2 k3 k4 k5 k6).sum = 42

-- The requirement to prove the sum of squares
def sum_of_squares : ℤ := (odd_numbers k1 k2 k3 k4 k5 k6).map (λ n, n^2).sum

theorem sum_of_squares_is_382 (h1 : sum_condition k1 k2 k3 k4 k5 k6) : 
  sum_of_squares k1 k2 k3 k4 k5 k6 = 382 :=
by sorry

end sum_of_squares_is_382_l391_391482


namespace subset_eq_disjunction_l391_391700

variable {A : Prop} -- A is a proposition
variable {p q : Prop} -- p and q are propositions

theorem subset_eq_disjunction (h1 : (φ = A) = (p ∧ q)) (h2 : (φ ⊂ A) = (¬p ∧ q)) :
  (φ ⊆ A) = (p ∨ q) :=
by
  sorry

end subset_eq_disjunction_l391_391700


namespace algebraic_expression_value_l391_391021

theorem algebraic_expression_value 
  (p q r s : ℝ) 
  (hpq3 : p^2 / q^3 = 4 / 5) 
  (hrs2 : r^3 / s^2 = 7 / 9) : 
  11 / (7 - r^3 / s^2) + (2 * q^3 - p^2) / (2 * q^3 + p^2) = 123 / 56 := 
by 
  sorry

end algebraic_expression_value_l391_391021


namespace smallest_n_for_factorial_inequality_l391_391603

theorem smallest_n_for_factorial_inequality :
  ∃ (n : ℕ), 1 * 2 * ... * (n - 1) > (n! * n!) ∧ (∀ m : ℕ, 1 * 2 * ... * (m - 1) > (m! * m!) → n ≤ m) :=
sorry

end smallest_n_for_factorial_inequality_l391_391603


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391411

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391411


namespace parity_uniformity_l391_391553

-- Define the set A_P
variable {A_P : Set ℤ}

-- Conditions:
-- 1. A_P is non-empty
noncomputable def non_empty (H : A_P ≠ ∅) := H

-- 2. c is the maximum element in A_P
variable {c : ℤ}
variable (H_max : ∀ a ∈ A_P, a ≤ c)

-- 3. Consideration of critical points around c
variable {f : ℤ → ℤ}
variable (H_critical : ∀ x ∈ A_P, f x = 0)

-- 4. Parity of the smallest and largest elements
def parity (n : ℤ) : Prop := n % 2 = 0 ∨ n % 2 = 1

-- Proof statement
theorem parity_uniformity (H_non_empty : non_empty A_P)
  (H_max_element : ∀ a ∈ A_P, a ≤ c)
  (H_critical_points : ∀ x ∈ A_P, f x = 0) :
  (∃ x ∈ A_P, ∀ y ∈ A_P, x ≤ y) → (parity (x : ℤ) = parity ((y : ℤ) : ℤ)) → (least x ∈ A_P, greatest y ∈ A_P, parity x = parity y) :=
by
  sorry

end parity_uniformity_l391_391553


namespace petya_friends_l391_391072

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l391_391072


namespace triangle_angles_l391_391118

theorem triangle_angles (a b c R A B C : ℝ) (h : a * a + b * b = c * c - R * R) 
    (ha : a = 2 * R * real.sin A) (hb : b = 2 * R * real.sin B) (hc : c = 2 * R * real.sin C)
    (hA : A = real.pi / 6) (hB : B = real.pi / 6) (hC : C = 2 * real.pi / 3) :
    A = real.pi / 6 ∧ B = real.pi / 6 ∧ C = 2 * real.pi / 3 :=
by
  sorry

end triangle_angles_l391_391118


namespace soccer_field_solution_l391_391539

noncomputable def soccer_field_problem : Prop :=
  ∃ (a b c d : ℝ), 
    (abs (a - b) = 1 ∨ abs (a - b) = 2 ∨ abs (a - b) = 3 ∨ abs (a - b) = 4 ∨ abs (a - b) = 5 ∨ abs (a - b) = 6) ∧
    (abs (a - c) = 1 ∨ abs (a - c) = 2 ∨ abs (a - c) = 3 ∨ abs (a - c) = 4 ∨ abs (a - c) = 5 ∨ abs (a - c) = 6) ∧
    (abs (a - d) = 1 ∨ abs (a - d) = 2 ∨ abs (a - d) = 3 ∨ abs (a - d) = 4 ∨ abs (a - d) = 5 ∨ abs (a - d) = 6) ∧
    (abs (b - c) = 1 ∨ abs (b - c) = 2 ∨ abs (b - c) = 3 ∨ abs (b - c) = 4 ∨ abs (b - c) = 5 ∨ abs (b - c) = 6) ∧
    (abs (b - d) = 1 ∨ abs (b - d) = 2 ∨ abs (b - d) = 3 ∨ abs (b - d) = 4 ∨ abs (b - d) = 5 ∨ abs (b - d) = 6) ∧
    (abs (c - d) = 1 ∨ abs (c - d) = 2 ∨ abs (c - d) = 3 ∨ abs (c - d) = 4 ∨ abs (c - d) = 5 ∨ abs (c - d) = 6)

theorem soccer_field_solution : soccer_field_problem :=
  sorry

end soccer_field_solution_l391_391539


namespace even_integers_between_5000_and_8000_with_distinct_digits_l391_391675

theorem even_integers_between_5000_and_8000_with_distinct_digits :
  let four_digit_numbers := {n | 5000 ≤ n ∧ n < 8000 ∧ Nat.digits 10 n |>.length = 4 ∧ ∀ i j, i ≠ j → Nat.digits 10 n ![i] ≠ Nat.digits 10 n ![j]} in
  ∃ S ⊆ four_digit_numbers, ∀ n ∈ S, n % 2 = 0 ∧ ∃ k, Nat.digits 10 n ![k] = 0 ∨ Nat.digits 10 n ![k] = 2 ∨ Nat.digits 10 n ![k] = 4 ∨ Nat.digits 10 n ![k] = 8 ∧
    (k = 0 → (5 ≤ Nat.digits 10 n ![1] ∧ Nat.digits 10 n ![1] ≤ 7) ∨ (Nat.digits 10 n ![1] = 5 ∨ Nat.digits 10 n ![1] = 7)) ∧
    (k ≠ 0 → (Nat.digits 10 n ![k] = 6)) ∧ 
    (∀ m < 4, ∀ p < 4, m ≠ p → Nat.digits 10 n ![m] ≠ Nat.digits 10 n ![p]) ∧
  ∑ _ in S, 1 = 672 :=
sorry

end even_integers_between_5000_and_8000_with_distinct_digits_l391_391675


namespace petya_friends_l391_391073

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l391_391073


namespace greatest_prime_factor_15_17_factorial_l391_391343

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391343


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391431

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391431


namespace distance_to_other_focus_ellipse_l391_391829

theorem distance_to_other_focus_ellipse
  (P : ℝ × ℝ) (x y : ℝ)
  (h_ellipse : x^2 / 36 + y^2 / 9 = 1)
  (a b : ℝ) (h_a : a = 6) (h_b : b = 3)
  (c : ℝ) (h_c_squared : c^2 = a^2 - b^2)
  (h_c : c = real.sqrt (27))
  (d₁ : ℝ) (h_distance_P_to_focus1 : d₁ = 5) :
  ∃ d₂ : ℝ, d₂ = 7 := by
  sorry

end distance_to_other_focus_ellipse_l391_391829


namespace f_monotonically_increasing_l391_391868

noncomputable def f (x : ℝ) : ℝ := √3 * Real.sin (2 * x) + Real.cos (2 * x)

theorem f_monotonically_increasing : ∀ x1 x2, 0 < x1 ∧ x1 < x2 ∧ x2 < π / 6 → f x1 < f x2 :=
by
  sorry

end f_monotonically_increasing_l391_391868


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391373

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391373


namespace greatest_prime_factor_of_sum_factorials_l391_391281

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391281


namespace susans_total_chairs_l391_391111

def number_of_red_chairs := 5
def number_of_yellow_chairs := 4 * number_of_red_chairs
def number_of_blue_chairs := number_of_yellow_chairs - 2
def total_chairs := number_of_red_chairs + number_of_yellow_chairs + number_of_blue_chairs

theorem susans_total_chairs : total_chairs = 43 :=
by
  sorry

end susans_total_chairs_l391_391111


namespace time_at_simple_interest_l391_391504

theorem time_at_simple_interest 
  (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : P = 300) 
  (h2 : (P * (R + 5) / 100) * T = (P * (R / 100) * T) + 150) : 
  T = 10 := 
by 
  -- Proof is omitted.
  sorry

end time_at_simple_interest_l391_391504


namespace solve_for_y_compute_expr_l391_391847

theorem solve_for_y_compute_expr (y : ℝ) 
  (h1: 3^(2*y) - 3^(2*y - 1) = 81) : 
  (3*y)^y = (30^(5/2)) / 32 := 
sorry

end solve_for_y_compute_expr_l391_391847


namespace area_ratio_of_M_in_K_l391_391932

def int_part (a : ℝ) : ℤ := int.floor a

def square_K : set (ℝ × ℝ) :=
{x | 0 ≤ x.fst ∧ x.fst ≤ 10 ∧ 0 ≤ x.snd ∧ x.snd ≤ 10}

def set_M : set (ℝ × ℝ) :=
{x | int_part x.fst = int_part x.snd}

theorem area_ratio_of_M_in_K :
  (measure_of (set_M ∩ square_K)) / (measure_of square_K) = 0.1 :=
sorry


end area_ratio_of_M_in_K_l391_391932


namespace greatest_prime_factor_15f_plus_17f_l391_391187

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391187


namespace number_of_stones_needed_l391_391951

-- Definitions for the conditions
def courtyard_length : ℝ := 60
def courtyard_width : ℝ := 40
def stone_area : ℝ := 2.3
def tree_radius : ℝ := 3.14

-- Calculated variables
def courtyard_area : ℝ := courtyard_length * courtyard_width
def tree_area : ℝ := Real.pi * tree_radius^2
def pave_area : ℝ := courtyard_area - tree_area

-- The main theorem statement
theorem number_of_stones_needed : ⌈pave_area / stone_area⌉ = 1031 := 
by
  -- proof (to be filled in later)
  sorry

end number_of_stones_needed_l391_391951


namespace greatest_prime_factor_of_factorial_sum_l391_391328

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391328


namespace sum_of_decimals_as_fraction_l391_391593

theorem sum_of_decimals_as_fraction :
  (0.2 : ℝ) + (0.03 : ℝ) + (0.004 : ℝ) + (0.0006 : ℝ) + (0.00007 : ℝ) + (0.000008 : ℝ) + (0.0000009 : ℝ) = 
  (2340087 / 10000000 : ℝ) :=
sorry

end sum_of_decimals_as_fraction_l391_391593


namespace min_value_w_l391_391583

theorem min_value_w : 
  ∀ x y : ℝ, let w := 3 * x^2 + 5 * y^2 + 8 * x - 10 * y + 34 in
  w ≥ 71 / 3 ∧ (∃ x y, x = -4 / 3 ∧ y = 1 ∧ w = 71 / 3) :=
by {
  intros x y,
  let w := 3 * x^2 + 5 * y^2 + 8 * x - 10 * y + 34,
  have h1 : ∃ x, x = -4 / 3 := ⟨-4 / 3, rfl⟩,
  have h2 : ∃ y, y = 1 := ⟨1, rfl⟩,
  sorry
}

end min_value_w_l391_391583


namespace least_6_digit_number_sum_of_digits_l391_391478

-- Definitions based on conditions
def is_6_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def leaves_remainder2 (n : ℕ) (d : ℕ) : Prop := n % d = 2

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Problem statement
theorem least_6_digit_number_sum_of_digits :
  ∃ n : ℕ, is_6_digit n ∧ leaves_remainder2 n 4 ∧ leaves_remainder2 n 610 ∧ leaves_remainder2 n 15 ∧ sum_of_digits n = 17 :=
sorry

end least_6_digit_number_sum_of_digits_l391_391478


namespace even_integers_between_5000_and_8000_with_distinct_digits_l391_391676

theorem even_integers_between_5000_and_8000_with_distinct_digits :
  let four_digit_numbers := {n | 5000 ≤ n ∧ n < 8000 ∧ Nat.digits 10 n |>.length = 4 ∧ ∀ i j, i ≠ j → Nat.digits 10 n ![i] ≠ Nat.digits 10 n ![j]} in
  ∃ S ⊆ four_digit_numbers, ∀ n ∈ S, n % 2 = 0 ∧ ∃ k, Nat.digits 10 n ![k] = 0 ∨ Nat.digits 10 n ![k] = 2 ∨ Nat.digits 10 n ![k] = 4 ∨ Nat.digits 10 n ![k] = 8 ∧
    (k = 0 → (5 ≤ Nat.digits 10 n ![1] ∧ Nat.digits 10 n ![1] ≤ 7) ∨ (Nat.digits 10 n ![1] = 5 ∨ Nat.digits 10 n ![1] = 7)) ∧
    (k ≠ 0 → (Nat.digits 10 n ![k] = 6)) ∧ 
    (∀ m < 4, ∀ p < 4, m ≠ p → Nat.digits 10 n ![m] ≠ Nat.digits 10 n ![p]) ∧
  ∑ _ in S, 1 = 672 :=
sorry

end even_integers_between_5000_and_8000_with_distinct_digits_l391_391676


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391239

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391239


namespace perimeter_of_triangle_AEC_l391_391964

/-
  Given:
  - A square with side length 2 and vertices A, B, C, D at coordinates (0, 2), (0, 0), (2, 0), and (2, 2) respectively.
  - Point C' is on edge AD such that C'D = 1/2, therefore C' is at (2, 1/2).
  - Point E is the intersection of edges BC' and AB, with E at coordinates (8/7, 8/7).

  Show:
  The perimeter of triangle AEC' is 15/2.
-/

noncomputable def point := (ℝ × ℝ)

def A : point := (0, 2)
def B : point := (0, 0)
def C : point := (2, 0)
def D : point := (2, 2)
def C' : point := (2, 1/2)
def E : point := (8/7, 8/7)

def distance (p1 p2 : point) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

def perimeter (p1 p2 p3 : point) : ℝ :=
  (distance p1 p2) + (distance p2 p3) + (distance p3 p1)

theorem perimeter_of_triangle_AEC' : perimeter A E C' = 15 / 2 :=
by
  sorry

end perimeter_of_triangle_AEC_l391_391964


namespace greatest_prime_factor_15f_plus_17f_l391_391192

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391192


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391387

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391387


namespace solve_equation_l391_391851

theorem solve_equation :
  ∀ x : ℝ, (-x^2 = (2*x + 4) / (x + 2)) ↔ (x = -2 ∨ x = -1) :=
by
  intro x
  -- the proof steps would go here
  sorry

end solve_equation_l391_391851


namespace area_of_triangle_ABC_l391_391720

variable (A B C M : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
variable [MetricSpace.Point A] [MetricSpace.Point B] [MetricSpace.Point C] [MetricSpace.Point M]
variable {AB AC BC AM : ℝ}
variable (h1 : AB = BC) 
variable (h2 : BC = 2 * AC) 
variable (h3 : AM = 4)
variable (h4 : M ∈ LineSegment (Point B) (Point C)) 
variable (h5 : IsAngleBisector (∠(A B C)) (Segment AM))

noncomputable def area_triangle_ABC : ℝ :=
  (1 / 2) * AC * BC * (Real.sin (angleBAC (A, B, C)))

theorem area_of_triangle_ABC :
  area_triangle_ABC A B C M AB AC BC AM h1 h2 h3 h4 h5 = (36 * Real.sqrt 15) / 5 := by
  sorry

end area_of_triangle_ABC_l391_391720


namespace problem_solution_l391_391670

variable {Mem : Type}
variable {Ens : Type}
variable {Veens : Type}

axiom Hypothesis_I : ∃ mem : Mem, mem ∉ Ens
axiom Hypothesis_II : ∀ en : Ens, en ∉ Veens

theorem problem_solution : ¬((∃ mem : Mem, mem ∉ Veens) ∨ 
                           (∃ veen : Veens, veen ∉ Mems) ∨ 
                           (∀ mem : Mem, mem ∉ Veens) ∨ 
                           (∃ mem : Mem, mem ∈ Veens)) :=
by {
    sorry
}

end problem_solution_l391_391670


namespace chests_content_l391_391768

theorem chests_content (A B C : Type) (chest1 chest2 chest3 : A)
  (label1 : chest1 = "gold coins") (label2 : chest2 = "silver coins") (label3 : chest3 = "gold or silver coins")
  (gold silver copper : A)
  (h1 : chest1 ≠ gold)
  (h2 : chest2 ≠ silver)
  (h3 : chest3 ≠ gold ∧ chest3 ≠ silver)
  (t1 : chest1 = silver)
  (t2 : chest2 = gold)
  (t3 : chest3 = copper) :
  True := sorry

end chests_content_l391_391768


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391367

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391367


namespace lesser_fraction_l391_391147

theorem lesser_fraction (x y : ℚ) (h1 : x + y = 14 / 15) (h2 : x * y = 1 / 10) : min x y = 1 / 5 :=
sorry

end lesser_fraction_l391_391147


namespace value_of_phi_l391_391697

theorem value_of_phi { φ : ℝ } (hφ1 : 0 < φ) (hφ2 : φ < π)
  (symm_condition : ∃ k : ℤ, -π / 8 + φ = k * π + π / 2) : φ = 3 * π / 4 := 
by 
  sorry

end value_of_phi_l391_391697


namespace fraction_simplification_l391_391862

theorem fraction_simplification :
  (20 + 16 * 20) / (20 * 16) = 17 / 16 :=
by
  sorry

end fraction_simplification_l391_391862


namespace sum_of_angles_in_quadrilateral_l391_391718

theorem sum_of_angles_in_quadrilateral (A B C D F G : ℝ) 
  (h1 : A = ∠ACQ∠M + ∠EAC∠QPCM)
  (h2 : B = ∠ABD + ∠DBC)
  (h3 : C = ∠BCD + ∠ABC)
  (h4 : D = ∠CDA + ∠ADC)
  (h5: F = ∠CDF)
  (h6 : G = ∠CG) :
  A + B + C + D + F + G = 360 := by
  sorry

end sum_of_angles_in_quadrilateral_l391_391718


namespace Petya_friends_l391_391088

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l391_391088


namespace greatest_prime_factor_of_sum_factorials_l391_391286

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391286


namespace circumradius_of_triangle_l391_391507

theorem circumradius_of_triangle (a b c : ℝ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 14) : 
  R = (35 * Real.sqrt 2) / 3 :=
by
  sorry

end circumradius_of_triangle_l391_391507


namespace chest_contents_correct_l391_391765

-- Define the chests
inductive Chest
| chest1
| chest2
| chest3

open Chest

-- Define the contents
inductive Contents
| gold
| silver
| copper

open Contents

-- Define the labels, all of which are incorrect
def label (c : Chest) : Prop :=
  match c with
  | chest1 => gold
  | chest2 => silver
  | chest3 => gold ∨ silver

-- Assuming the labels are all incorrect
axiom label_incorrect : ∀ (c : Chest), ¬(label c = true)

-- One chest contains each type of coin
axiom one_gold : ∃ c, ∀ x, x ≠ c → Contents x ≠ gold
axiom one_silver : ∃ c, ∀ x, x ≠ c → Contents x ≠ silver
axiom one_copper : ∃ c, ∀ x, x ≠ c → Contents x ≠ copper

-- Determine the contents of each chest
def chest1_contents := silver
def chest2_contents := gold
def chest3_contents := copper

-- Prove the correspondence
theorem chest_contents_correct :
  (chest1_contents = silver) ∧ 
  (chest2_contents = gold) ∧ 
  (chest3_contents = copper) :=
by
  split
  · exact sorry
  split
  · exact sorry
  · exact sorry

end chest_contents_correct_l391_391765


namespace revenue_fall_percent_l391_391120

def old_value : ℝ := 85.0
def new_value : ℝ := 48.0
def percent_decrease (old_value new_value : ℝ) : ℝ := ((old_value - new_value) / old_value) * 100

theorem revenue_fall_percent :
  percent_decrease old_value new_value ≈ 43.53 :=
by
  sorry

end revenue_fall_percent_l391_391120


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391305
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391305


namespace product_positivity_l391_391136

theorem product_positivity (a b c d e f : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)
  (h : ∀ {x y z u v w : ℝ}, x * y * z > 0 → ¬(u * v * w > 0) ∧ ¬(x * v * u > 0) ∧ ¬(y * z * u > 0) ∧ ¬(y * v * x > 0)) :
  b * d * e > 0 ∧ a * c * d < 0 ∧ a * c * e < 0 ∧ b * d * f < 0 ∧ b * e * f < 0 := 
by
  sorry

end product_positivity_l391_391136


namespace petya_friends_count_l391_391057

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l391_391057


namespace petya_friends_count_l391_391060

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l391_391060


namespace soccer_players_positions_l391_391534

theorem soccer_players_positions :
  ∃ (a b c d : ℝ), a = 0 ∧ b = 1 ∧ c = 4 ∧ d = 6 ∧
  set_of (λ x, ∃ i j, i ≠ j ∧ x = abs (a - b) ∨ x = abs (a - c) ∨ x = abs (a - d) ∨ x = abs (b - c) ∨ x = abs (b - d) ∨ x = abs (c - d)) = {1, 2, 3, 4, 5, 6} :=
by
  use 0, 1, 4, 6
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  rw [set_of, abs]
  sorry

end soccer_players_positions_l391_391534


namespace distinct_numbers_in_list_l391_391599

def count_distinct_floors (l : List ℕ) : ℕ :=
  l.eraseDups.length

def generate_list : List ℕ :=
  List.map (λ n => Nat.floor ((n * n : ℚ) / 2000)) (List.range' 1 2000)

theorem distinct_numbers_in_list : count_distinct_floors generate_list = 1501 :=
by
  sorry

end distinct_numbers_in_list_l391_391599


namespace chest_contents_solution_l391_391789

-- Definitions corresponding to the conditions.
structure ChestContents :=
  (chest1 chest2 chest3 : String)

-- Given conditions
def labelsAreIncorrect (contents : ChestContents) : Prop :=
  contents.chest1 ≠ "Gold coins" ∧
  contents.chest2 ≠ "Silver coins" ∧
  contents.chest3 ≠ "Gold coins" ∧
  contents.chest3 ≠ "Silver coins"

def uniqueCoins (contents : ChestContents) : Prop :=
  (contents.chest1 = "Gold coins" ∨ contents.chest1 = "Silver coins" ∨ contents.chest1 = "Copper coins") ∧
  (contents.chest2 = "Gold coins" ∨ contents.chest2 = "Silver coins" ∨ contents.chest2 = "Copper coins") ∧
  (contents.chest3 = "Gold coins" ∨ contents.chest3 = "Silver coins" ∨ contents.chest3 = "Copper coins") ∧
  (contents.chest1 ≠ contents.chest2) ∧
  (contents.chest1 ≠ contents.chest3) ∧
  (contents.chest2 ≠ contents.chest3)

-- The proof statement
theorem chest_contents_solution : ∃ (contents : ChestContents),
  labelsAreIncorrect contents ∧ uniqueCoins contents ∧
  contents.chest1 = "Silver coins" ∧
  contents.chest2 = "Gold coins" ∧
  contents.chest3 = "Copper coins" :=
begin
  sorry
end

end chest_contents_solution_l391_391789


namespace chest_contents_correct_l391_391759

-- Define the chests
inductive Chest
| chest1
| chest2
| chest3

open Chest

-- Define the contents
inductive Contents
| gold
| silver
| copper

open Contents

-- Define the labels, all of which are incorrect
def label (c : Chest) : Prop :=
  match c with
  | chest1 => gold
  | chest2 => silver
  | chest3 => gold ∨ silver

-- Assuming the labels are all incorrect
axiom label_incorrect : ∀ (c : Chest), ¬(label c = true)

-- One chest contains each type of coin
axiom one_gold : ∃ c, ∀ x, x ≠ c → Contents x ≠ gold
axiom one_silver : ∃ c, ∀ x, x ≠ c → Contents x ≠ silver
axiom one_copper : ∃ c, ∀ x, x ≠ c → Contents x ≠ copper

-- Determine the contents of each chest
def chest1_contents := silver
def chest2_contents := gold
def chest3_contents := copper

-- Prove the correspondence
theorem chest_contents_correct :
  (chest1_contents = silver) ∧ 
  (chest2_contents = gold) ∧ 
  (chest3_contents = copper) :=
by
  split
  · exact sorry
  split
  · exact sorry
  · exact sorry

end chest_contents_correct_l391_391759


namespace candy_left_l391_391606

-- Definitions according to the conditions
def initialCandy : ℕ := 15
def candyGivenToHaley : ℕ := 6

-- Theorem statement formalizing the proof problem
theorem candy_left (c : ℕ) (h₁ : c = initialCandy - candyGivenToHaley) : c = 9 :=
by
  -- The proof is omitted as instructed.
  sorry

end candy_left_l391_391606


namespace problem_statement_l391_391662

noncomputable def g : ℝ → ℝ := sorry

axiom condition1 : ∀ c d : ℝ, d^2 * g(c) = c^2 * g(d)
axiom condition2 : g 3 ≠ 0

theorem problem_statement : (g 6 + g 2) / g 3 = 40 / 9 :=
by
  -- Insert proof steps here
  sorry

end problem_statement_l391_391662


namespace roots_abs_lt_one_l391_391098

theorem roots_abs_lt_one
  (a b : ℝ)
  (h1 : |a| + |b| < 1)
  (h2 : a^2 - 4 * b ≥ 0) :
  ∀ (x : ℝ), x^2 + a * x + b = 0 → |x| < 1 :=
sorry

end roots_abs_lt_one_l391_391098


namespace concurrent_lines_l391_391629

theorem concurrent_lines (A B C O D E F A' B' C' : Point)
  (h1 : Incircle O A B C)
  (h2 : OnCircle D O)
  (h3 : OnCircle E O)
  (h4 : OnCircle F O)
  (h5 : Touches D B C)
  (h6 : Touches E C A)
  (h7 : Touches F A B)
  (h8 : Intersects (Line D O) (Line E F) A')
  (h9 : Intersects (Line E O) (Line D F) B')
  (h10 : Intersects (Line F O) (Line D E) C') :
  Concurrent (Line A A') (Line B B') (Line C C') :=
sorry

end concurrent_lines_l391_391629


namespace max_value_of_f_l391_391656

def f (x : ℝ) : ℝ :=
  (2 - real.cos (π / 4 * (1 - x)) + real.sin (π / 4 * (1 - x))) / (x^2 + 4 * x + 5)

theorem max_value_of_f :
  ∃ x : ℝ, -4 ≤ x ∧ x ≤ 0 ∧ ∀ y : ℝ, -4 ≤ y ∧ y ≤ 0 → f y ≤ 2 + real.sqrt 2 :=
sorry

end max_value_of_f_l391_391656


namespace solve_for_w_l391_391582

-- Define the given function g
def g (t : ℝ) : ℝ := 2 * t / (1 - 2 * t)

-- Define the proof problem
theorem solve_for_w (w z : ℝ) (h : z = g w) (hw_ne_half : w ≠ 1 / 2) : w = z / (2 * (1 + z)) :=
sorry

end solve_for_w_l391_391582


namespace koschei_chests_l391_391757

theorem koschei_chests :
  ∃ (contents : Fin 3 → String), 
    -- All chests labels are incorrect
    (contents 0 ≠ "gold coins" ∧ contents 1 ≠ "silver coins" ∧ contents 2 ≠ "gold or silver coins") ∧ 
    -- Each chest contains exactly one type of coin 
    (∀ i j : Fin 3, i ≠ j → contents i ≠ contents j) ∧ 
    -- Providing the final conclusion about what is in each chest
    (contents 0 = "silver coins" ∧ contents 1 = "gold coins" ∧ contents 2 = "copper coins") :=
begin
  use (λ k, if k = 0 then "silver coins" else if k = 1 then "gold coins" else "copper coins"),
  split,
  {
    -- Proof of all labels being incorrect
    split; simp,
  },
  split,
  {
    -- Proof of each chest containing a unique type of coin
    intros i j h,
    cases i; cases j; simp [h],
  },
  {
    -- Proof of the final conclusion
    split; split; simp,
  },
end

end koschei_chests_l391_391757


namespace probability_sum_does_not_exceed_8_l391_391920

-- Definitions for the conditions
def uniform_die : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 6} -- A uniform die with faces from 1 to 6

-- Resulting probability that we need to prove
theorem probability_sum_does_not_exceed_8 :
  (∑ n1 in finset.univ.filter (λ n1 : uniform_die, ∑ n2 in finset.univ.filter (λ n2 : uniform_die, n1 ≠ n2), 
         ∑ n3 in finset.univ.filter (λ n3 : uniform_die, n1 ≠ n3 ∧ n2 ≠ n3),
            if (n1.1 + n2.1 + n3.1 ≤ 8) then 1 else 0).to_real) = (∑ _ in finset.univ, 1).to_real * (1/5 : ℝ) :=
sorry

end probability_sum_does_not_exceed_8_l391_391920


namespace smallest_largest_same_parity_l391_391561

-- Define the context where our elements and set are considered
variable {α : Type*} [LinearOrderedCommRing α] (A_P : Set α)

-- Assume a nonempty set A_P and define the smallest and largest elements in Lean
noncomputable def smallest_element (A_P : Set α) : α := Inf' A_P sorry
noncomputable def largest_element (A_P : Set α) : α := Sup' A_P sorry

-- State the proof goal
theorem smallest_largest_same_parity (h_nonempty : A_P.nonempty) : 
  (smallest_element A_P) % 2 = (largest_element A_P) % 2 := 
by 
  sorry

end smallest_largest_same_parity_l391_391561


namespace chests_contents_l391_391738

def ChestLabel (c : ℕ) : String := 
  if c = 1 then "Gold coins"
  else if c = 2 then "Silver coins"
  else if c = 3 then "Gold or silver coins"
  else "Invalid chest"

def CoinsInChest (c : ℕ) : String := 
  if c = 1 then "Silver coins"
  else if c = 2 then "Gold coins"
  else if c = 3 then "Copper coins"
  else "Invalid chest"

theorem chests_contents :
  ChestLabel 1 ≠ CoinsInChest 1 ∧
  ChestLabel 2 ≠ CoinsInChest 2 ∧
  ChestLabel 3 ≠ CoinsInChest 3 ∧
  (CoinsInChest 1 = "Gold coins" ∨ CoinsInChest 2 = "Gold coins" ∨ CoinsInChest 3 = "Gold coins") ∧
  (CoinsInChest 1 = "Silver coins" ∨ CoinsInChest 2 = "Silver coins" ∨ CoinsInChest 3 = "Silver coins") ∧
  (CoinsInChest 1 = "Copper coins" ∨ CoinsInChest 2 = "Copper coins" ∨ CoinsInChest 3 = "Copper coins") :=
begin
  sorry
end

end chests_contents_l391_391738


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391297

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391297


namespace four_digit_palindrome_divisible_by_11_probability_zero_l391_391953

theorem four_digit_palindrome_divisible_by_11_probability_zero :
  (∃ a b : ℕ, 2 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (1001 * a + 110 * b) % 11 = 0) = false :=
by sorry

end four_digit_palindrome_divisible_by_11_probability_zero_l391_391953


namespace greatest_prime_factor_15f_plus_17f_l391_391195

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391195


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391269

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391269


namespace petya_friends_l391_391079

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l391_391079


namespace binomial_sum_identity_l391_391838

theorem binomial_sum_identity (n : ℕ) :
  ∑ k in Finset.range (n + 1), (-1) ^ (n - k) * 2 ^ (2 * k) * Nat.choose (n + k + 1) (2 * k + 1) = n + 1 := 
sorry

end binomial_sum_identity_l391_391838


namespace ducks_at_Lake_Michigan_l391_391845

variable (D : ℕ)

def ducks_condition := 2 * D + 6 = 206

theorem ducks_at_Lake_Michigan (h : ducks_condition D) : D = 100 :=
by
  sorry

end ducks_at_Lake_Michigan_l391_391845


namespace angle_halving_quadrant_l391_391686

theorem angle_halving_quadrant (k : ℤ) (α : ℝ) 
  (h : k * 360 + 180 < α ∧ α < k * 360 + 270) : 
  k * 180 + 90 < α / 2 ∧ α / 2 < k * 180 + 135 :=
sorry

end angle_halving_quadrant_l391_391686


namespace distance_between_parallel_lines_eq_l391_391122

open Real

theorem distance_between_parallel_lines_eq
  (h₁ : ∀ (x y : ℝ), 3 * x + y - 3 = 0 → Prop)
  (h₂ : ∀ (x y : ℝ), 6 * x + 2 * y + 1 = 0 → Prop) :
  ∃ d : ℝ, d = (7 / 20) * sqrt 10 :=
sorry

end distance_between_parallel_lines_eq_l391_391122


namespace nate_ratio_is_four_to_one_l391_391825

def nate_exercise : Prop :=
  ∃ (D T L : ℕ), 
    T = D + 500 ∧ 
    T = 1172 ∧ 
    L = 168 ∧ 
    D / L = 4

theorem nate_ratio_is_four_to_one : nate_exercise := 
  sorry

end nate_ratio_is_four_to_one_l391_391825


namespace petya_friends_l391_391042

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l391_391042


namespace previous_painting_price_l391_391836

-- Define the amount received for the most recent painting
def recentPainting (p : ℕ) := 5 * p - 1000

-- Define the target amount
def target := 44000

-- State that the target amount is achieved by the prescribed function
theorem previous_painting_price : recentPainting 9000 = target :=
by
  sorry

end previous_painting_price_l391_391836


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391293

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391293


namespace greatest_prime_factor_15_fact_17_fact_l391_391219

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391219


namespace petya_friends_l391_391065

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l391_391065


namespace greatest_prime_factor_of_factorial_sum_l391_391325

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391325


namespace count_ordered_tuples_l391_391600

open Nat

theorem count_ordered_tuples : 
  (Finset.card {tup : Finset (ℕ × ℕ × ℕ × ℕ) | 
    ∃ (C A M B : ℕ), tup = (C, A, M, B) ∧ 
    C! + C! + A! + M! = B!}) = 7 := 
sorry

end count_ordered_tuples_l391_391600


namespace complex_point_quadrant_l391_391878

theorem complex_point_quadrant : 
  let z := 1 / (1 + complex.i) in 
  (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end complex_point_quadrant_l391_391878


namespace chests_content_l391_391742

-- Define the chests and their labels
inductive CoinContent where
  | gold : CoinContent
  | silver : CoinContent
  | copper : CoinContent

structure Chest where
  label : CoinContent
  contents : CoinContent

-- Given conditions and incorrect labels
def chest1 : Chest := { label := CoinContent.gold, contents := sorry }
def chest2 : Chest := { label := CoinContent.silver, contents := sorry }
def chest3 : Chest := { label := CoinContent.gold, contents := sorry }

-- The proof problem
theorem chests_content :
  chest1.contents ≠ CoinContent.gold ∧
  chest2.contents ≠ CoinContent.silver ∧
  chest3.contents ≠ CoinContent.gold ∨ chest3.contents ≠ CoinContent.silver →
  chest1.contents = CoinContent.silver ∧
  chest2.contents = CoinContent.gold ∧
  chest3.contents = CoinContent.copper := by
  sorry

end chests_content_l391_391742


namespace soccer_field_solution_l391_391543

noncomputable def soccer_field_problem : Prop :=
  ∃ (a b c d : ℝ), 
    (abs (a - b) = 1 ∨ abs (a - b) = 2 ∨ abs (a - b) = 3 ∨ abs (a - b) = 4 ∨ abs (a - b) = 5 ∨ abs (a - b) = 6) ∧
    (abs (a - c) = 1 ∨ abs (a - c) = 2 ∨ abs (a - c) = 3 ∨ abs (a - c) = 4 ∨ abs (a - c) = 5 ∨ abs (a - c) = 6) ∧
    (abs (a - d) = 1 ∨ abs (a - d) = 2 ∨ abs (a - d) = 3 ∨ abs (a - d) = 4 ∨ abs (a - d) = 5 ∨ abs (a - d) = 6) ∧
    (abs (b - c) = 1 ∨ abs (b - c) = 2 ∨ abs (b - c) = 3 ∨ abs (b - c) = 4 ∨ abs (b - c) = 5 ∨ abs (b - c) = 6) ∧
    (abs (b - d) = 1 ∨ abs (b - d) = 2 ∨ abs (b - d) = 3 ∨ abs (b - d) = 4 ∨ abs (b - d) = 5 ∨ abs (b - d) = 6) ∧
    (abs (c - d) = 1 ∨ abs (c - d) = 2 ∨ abs (c - d) = 3 ∨ abs (c - d) = 4 ∨ abs (c - d) = 5 ∨ abs (c - d) = 6)

theorem soccer_field_solution : soccer_field_problem :=
  sorry

end soccer_field_solution_l391_391543


namespace marbles_problem_l391_391950

theorem marbles_problem
  (total_marbles red_marbles blue_marbles yellow_marbles white_marbles black_marbles : ℕ)
  (h_total : total_marbles = 200)
  (h_red : red_marbles = 80)
  (h_blue : blue_marbles = 50)
  (h_yellow : yellow_marbles = 20)
  (h_white : white_marbles = 30)
  (h_black : black_marbles = 20)
  (h_red_fraction : red_marbles = 0.4 * total_marbles)
  (h_blue_fraction : blue_marbles = 0.25 * total_marbles)
  (h_yellow_fraction : yellow_marbles = 0.1 * total_marbles)
  (h_white_fraction : white_marbles = 0.15 * total_marbles)
  (h_black_count : black_marbles = total_marbles - (red_marbles + blue_marbles + yellow_marbles + white_marbles))
  : (blue_marbles + (red_marbles / 3).nat_ceil) = 77 := by
  sorry

end marbles_problem_l391_391950


namespace even_integers_between_5000_and_8000_with_distinct_digits_l391_391677

theorem even_integers_between_5000_and_8000_with_distinct_digits :
  let four_digit_numbers := {n | 5000 ≤ n ∧ n < 8000 ∧ Nat.digits 10 n |>.length = 4 ∧ ∀ i j, i ≠ j → Nat.digits 10 n ![i] ≠ Nat.digits 10 n ![j]} in
  ∃ S ⊆ four_digit_numbers, ∀ n ∈ S, n % 2 = 0 ∧ ∃ k, Nat.digits 10 n ![k] = 0 ∨ Nat.digits 10 n ![k] = 2 ∨ Nat.digits 10 n ![k] = 4 ∨ Nat.digits 10 n ![k] = 8 ∧
    (k = 0 → (5 ≤ Nat.digits 10 n ![1] ∧ Nat.digits 10 n ![1] ≤ 7) ∨ (Nat.digits 10 n ![1] = 5 ∨ Nat.digits 10 n ![1] = 7)) ∧
    (k ≠ 0 → (Nat.digits 10 n ![k] = 6)) ∧ 
    (∀ m < 4, ∀ p < 4, m ≠ p → Nat.digits 10 n ![m] ≠ Nat.digits 10 n ![p]) ∧
  ∑ _ in S, 1 = 672 :=
sorry

end even_integers_between_5000_and_8000_with_distinct_digits_l391_391677


namespace floor_e_eq_2_l391_391588

noncomputable def e : ℝ := Real.exp 1

theorem floor_e_eq_2 : ⌊e⌋ = 2 :=
by
  have e_approx : 2 < e ∧ e < 3 := 
    ⟨by norm_num [e, Real.exp], by norm_num [e, Real.exp]⟩
  sorry

end floor_e_eq_2_l391_391588


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391236

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391236


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391309
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391309


namespace tom_total_payment_l391_391897

variable (apples_kg : ℕ := 8)
variable (apples_rate : ℕ := 70)
variable (mangoes_kg : ℕ := 9)
variable (mangoes_rate : ℕ := 65)
variable (oranges_kg : ℕ := 5)
variable (oranges_rate : ℕ := 50)
variable (bananas_kg : ℕ := 3)
variable (bananas_rate : ℕ := 30)
variable (discount_apples : ℝ := 0.10)
variable (discount_oranges : ℝ := 0.15)

def total_cost_apple : ℝ := apples_kg * apples_rate
def total_cost_mango : ℝ := mangoes_kg * mangoes_rate
def total_cost_orange : ℝ := oranges_kg * oranges_rate
def total_cost_banana : ℝ := bananas_kg * bananas_rate
def discount_apples_amount : ℝ := discount_apples * total_cost_apple
def discount_oranges_amount : ℝ := discount_oranges * total_cost_orange
def apples_after_discount : ℝ := total_cost_apple - discount_apples_amount
def oranges_after_discount : ℝ := total_cost_orange - discount_oranges_amount

theorem tom_total_payment :
  apples_after_discount + total_cost_mango + oranges_after_discount + total_cost_banana = 1391.5 := by
  sorry

end tom_total_payment_l391_391897


namespace chest_contents_solution_l391_391788

-- Definitions corresponding to the conditions.
structure ChestContents :=
  (chest1 chest2 chest3 : String)

-- Given conditions
def labelsAreIncorrect (contents : ChestContents) : Prop :=
  contents.chest1 ≠ "Gold coins" ∧
  contents.chest2 ≠ "Silver coins" ∧
  contents.chest3 ≠ "Gold coins" ∧
  contents.chest3 ≠ "Silver coins"

def uniqueCoins (contents : ChestContents) : Prop :=
  (contents.chest1 = "Gold coins" ∨ contents.chest1 = "Silver coins" ∨ contents.chest1 = "Copper coins") ∧
  (contents.chest2 = "Gold coins" ∨ contents.chest2 = "Silver coins" ∨ contents.chest2 = "Copper coins") ∧
  (contents.chest3 = "Gold coins" ∨ contents.chest3 = "Silver coins" ∨ contents.chest3 = "Copper coins") ∧
  (contents.chest1 ≠ contents.chest2) ∧
  (contents.chest1 ≠ contents.chest3) ∧
  (contents.chest2 ≠ contents.chest3)

-- The proof statement
theorem chest_contents_solution : ∃ (contents : ChestContents),
  labelsAreIncorrect contents ∧ uniqueCoins contents ∧
  contents.chest1 = "Silver coins" ∧
  contents.chest2 = "Gold coins" ∧
  contents.chest3 = "Copper coins" :=
begin
  sorry
end

end chest_contents_solution_l391_391788


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391301

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391301


namespace Petya_friends_l391_391084

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l391_391084


namespace tanner_savings_in_october_l391_391114

theorem tanner_savings_in_october 
    (sept_savings : ℕ := 17) 
    (nov_savings : ℕ := 25)
    (spent : ℕ := 49) 
    (left : ℕ := 41) 
    (X : ℕ) 
    (h : sept_savings + X + nov_savings - spent = left) 
    : X = 48 :=
by
  sorry

end tanner_savings_in_october_l391_391114


namespace pepper_left_l391_391546

def initial_pepper : ℝ := 0.25
def used_pepper : ℝ := 0.16
def remaining_pepper : ℝ := 0.09

theorem pepper_left (h1 : initial_pepper = 0.25) (h2 : used_pepper = 0.16) :
  initial_pepper - used_pepper = remaining_pepper :=
by
  sorry

end pepper_left_l391_391546


namespace range_of_f_l391_391144

noncomputable def f (x : ℝ) : ℝ := log (2 : ℝ) (3 * x + 1)

theorem range_of_f : set.range f = {y : ℝ | 0 < y} :=
by
  sorry

end range_of_f_l391_391144


namespace determine_n_from_average_l391_391488

-- Definitions derived from conditions
def total_cards (n : ℕ) : ℕ := n * (n + 1) / 2
def sum_of_values (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6
def average_value (n : ℕ) : ℚ := sum_of_values n / total_cards n

-- Main statement for proving equivalence
theorem determine_n_from_average :
  (∃ n : ℕ, average_value n = 2023) ↔ (n = 3034) :=
by
  sorry

end determine_n_from_average_l391_391488


namespace find_f_minus1_plus_f_2_l391_391644

variable (f : ℝ → ℝ)

def even_function := ∀ x : ℝ, f (-x) = f x

def symmetric_about_origin := ∀ x : ℝ, f (x + 1) = -f (-(x + 1))

def f_value_at_zero := f 0 = 1

theorem find_f_minus1_plus_f_2 :
  even_function f →
  symmetric_about_origin f →
  f_value_at_zero f →
  f (-1) + f 2 = -1 :=
by
  intros
  sorry

end find_f_minus1_plus_f_2_l391_391644


namespace chest_contents_correct_l391_391761

-- Define the chests
inductive Chest
| chest1
| chest2
| chest3

open Chest

-- Define the contents
inductive Contents
| gold
| silver
| copper

open Contents

-- Define the labels, all of which are incorrect
def label (c : Chest) : Prop :=
  match c with
  | chest1 => gold
  | chest2 => silver
  | chest3 => gold ∨ silver

-- Assuming the labels are all incorrect
axiom label_incorrect : ∀ (c : Chest), ¬(label c = true)

-- One chest contains each type of coin
axiom one_gold : ∃ c, ∀ x, x ≠ c → Contents x ≠ gold
axiom one_silver : ∃ c, ∀ x, x ≠ c → Contents x ≠ silver
axiom one_copper : ∃ c, ∀ x, x ≠ c → Contents x ≠ copper

-- Determine the contents of each chest
def chest1_contents := silver
def chest2_contents := gold
def chest3_contents := copper

-- Prove the correspondence
theorem chest_contents_correct :
  (chest1_contents = silver) ∧ 
  (chest2_contents = gold) ∧ 
  (chest3_contents = copper) :=
by
  split
  · exact sorry
  split
  · exact sorry
  · exact sorry

end chest_contents_correct_l391_391761


namespace greatest_prime_factor_15_fact_17_fact_l391_391180

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391180


namespace total_amount_paid_l391_391978

-- Definitions
def original_aquarium_price : ℝ := 120
def aquarium_discount : ℝ := 0.5
def aquarium_coupon : ℝ := 0.1
def aquarium_sales_tax : ℝ := 0.05

def plants_decorations_price_before_discount : ℝ := 75
def plants_decorations_discount : ℝ := 0.15
def plants_decorations_sales_tax : ℝ := 0.08

def fish_food_price : ℝ := 25
def fish_food_sales_tax : ℝ := 0.06

-- Final result to be proved
theorem total_amount_paid : 
  let discounted_aquarium_price := original_aquarium_price * (1 - aquarium_discount)
  let coupon_aquarium_price := discounted_aquarium_price * (1 - aquarium_coupon)
  let total_aquarium_price := coupon_aquarium_price * (1 + aquarium_sales_tax)
  let discounted_plants_decorations_price := plants_decorations_price_before_discount * (1 - plants_decorations_discount)
  let total_plants_decorations_price := discounted_plants_decorations_price * (1 + plants_decorations_sales_tax)
  let total_fish_food_price := fish_food_price * (1 + fish_food_sales_tax)
  total_aquarium_price + total_plants_decorations_price + total_fish_food_price = 152.05 :=
by 
  sorry

end total_amount_paid_l391_391978


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391377

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391377


namespace average_mpg_l391_391977

theorem average_mpg (initial_odometer final_odometer_with_error actual_final_odometer initial_gas refill1_odometer refill1_gas refill2_odometer refill2_gas : ℝ)
(initial_miles initial_gasoline : ℝ := 10)
(refill1_miles : ℝ := 68_720)
(refill1_gasoline : ℝ := 15)
(refill2_miles_with_error refill2_corrected_miles : ℝ := 69_350)
(refill2_corrected_miles := 69_350 - 10)
(refill2_gasoline : ℝ := 25)
(start_odometer : ℝ := 68_300)
(final_miles := final_odometer_with_error - 10)
(total_distance : ℝ := actual_final_odometer - initial_odometer)
(total_gasoline : ℝ := initial_gas + refill1_gas + refill2_gas)
(average_mpg : ℝ := total_distance / total_gasoline)
(h1 : initial_odometer = start_odometer)
(h2 : actual_final_odometer = final_miles)
(h3 : actual_final_odometer - initial_odometer = 1_040)
(h4 : initial_gas + refill1_gas + refill2_gas = 50)
-- Add any missing intermediate conditions if necessary
: average_mpg = 20.8 := 
sorry

end average_mpg_l391_391977


namespace extremum_f_when_a_zero_monotonic_intervals_f_range_m_when_inequality_holds_l391_391658

-- Definitions based on given conditions
def f (a x : ℝ) : ℝ := (2 - a) * Real.log x + 1 / x + 2 * a * x

-- Question (1): Extremum of f(x) when a = 0
theorem extremum_f_when_a_zero : ∃ x : ℝ, x > 0 ∧ f 0 x = 2 - 2 * Real.log 2 := 
sorry

-- Question (2): Monotonic intervals of f(x) when a < 0
theorem monotonic_intervals_f (a : ℝ) (h : a < 0) : 
  (if a < -2 then 
     (∀ x, x ∈ (0, -1/a) ∪ (1/2, ∞) → f' a x < 0) ∧ 
     (∀ x, x ∈ (-1/a, 1/2) → f' a x > 0)
   else if a = -2 then 
     (∀ x > 0, f' a x ≤ 0)
   else 
     (∀ x, x ∈ (0, 1/2) ∪ (-1/a, ∞) → f' a x < 0) ∧ 
     (∀ x, x ∈ (1/2, -1/a) → f' a x > 0))
:=
sorry

-- Question (3): Range of m when -3 < a < -2 and given inequality holds
theorem range_m_when_inequality_holds (a : ℝ) (h₁ : -3 < a) (h₂ : a < -2) 
  (λ₁ λ₂ : ℝ) (hm : λ₁ ∈ [1, 3] ∧ λ₂ ∈ [1, 3] ∧ 
                 abs (f a λ₁ - f a λ₂) > (m + Real.log 3) * a - 2 * Real.log 3) : 
  m ≥ -38/9 :=
sorry

end extremum_f_when_a_zero_monotonic_intervals_f_range_m_when_inequality_holds_l391_391658


namespace effective_gain_approx_435_l391_391694

noncomputable def cost_price : ℝ := sorry -- Define the cost price of one article
noncomputable def selling_price : ℝ := sorry -- Define the selling price of one article

/-- The cost price of 80 articles is equal to the selling price of 60 articles. -/
axiom cost_price_80_eq_selling_price_60 : 80 * cost_price = 60 * selling_price

/-- Additional 15% tax on the cost price -/
def effective_cost_price : ℝ := cost_price * (1 + 0.15)

/-- Discount of 10% on the selling price -/
def effective_selling_price : ℝ := selling_price * (1 - 0.10)

/-- The effective gain percentage -/
def gain_percentage : ℝ := ((effective_selling_price - effective_cost_price) / effective_cost_price) * 100

/-- Prove that the gain percentage is approximately 4.35% -/
theorem effective_gain_approx_435 : gain_percentage ≈ 4.35 := sorry

end effective_gain_approx_435_l391_391694


namespace monotonic_increasing_interval_l391_391134

noncomputable def original_function (x : ℝ) := log (1 / 2 : ℝ) (6 + x - x^2)

noncomputable def is_increasing_interval_in_domain : set ℝ :=
{ x | -2 < x ∧ x < 3 }

theorem monotonic_increasing_interval :
  ∀ x ∈ is_increasing_interval_in_domain, 
  ∀ y ∈ is_increasing_interval_in_domain, 
  (x < y) → original_function x < original_function y ↔ (x ∈ set.Ico (1 / 2) 3) :=
by
  sorry

end monotonic_increasing_interval_l391_391134


namespace chests_content_l391_391747

-- Define the chests and their labels
inductive CoinContent where
  | gold : CoinContent
  | silver : CoinContent
  | copper : CoinContent

structure Chest where
  label : CoinContent
  contents : CoinContent

-- Given conditions and incorrect labels
def chest1 : Chest := { label := CoinContent.gold, contents := sorry }
def chest2 : Chest := { label := CoinContent.silver, contents := sorry }
def chest3 : Chest := { label := CoinContent.gold, contents := sorry }

-- The proof problem
theorem chests_content :
  chest1.contents ≠ CoinContent.gold ∧
  chest2.contents ≠ CoinContent.silver ∧
  chest3.contents ≠ CoinContent.gold ∨ chest3.contents ≠ CoinContent.silver →
  chest1.contents = CoinContent.silver ∧
  chest2.contents = CoinContent.gold ∧
  chest3.contents = CoinContent.copper := by
  sorry

end chests_content_l391_391747


namespace integral_right_angled_triangles_unique_l391_391103

theorem integral_right_angled_triangles_unique : 
  ∀ a b c : ℤ, (a < b) ∧ (b < c) ∧ (a^2 + b^2 = c^2) ∧ (a * b = 4 * (a + b + c))
  ↔ (a = 10 ∧ b = 24 ∧ c = 26)
  ∨ (a = 12 ∧ b = 16 ∧ c = 20)
  ∨ (a = 9 ∧ b = 40 ∧ c = 41) :=
by {
  sorry
}

end integral_right_angled_triangles_unique_l391_391103


namespace infinite_pairs_l391_391613

-- Define sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.toString.data.foldl (λ acc char_digit, acc + (char_digit.toNat - '0'.toNat)) 0

-- Define the main theorem
theorem infinite_pairs (k : ℤ) : 
  ∃ (m n : ℕ), (m ≠ n) ∧ 
               (n + sum_of_digits (2 * n) = m + sum_of_digits (2 * m)) ∧ 
               (k * n + sum_of_digits (n * n) = k * m + sum_of_digits (m * m)) := 
  sorry

end infinite_pairs_l391_391613


namespace find_positive_product_l391_391138

variable (a b c d e f : ℝ)

-- Define the condition that exactly one of the products is positive
def exactly_one_positive (p1 p2 p3 p4 p5 : ℝ) : Prop :=
  (p1 > 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 > 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 > 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 > 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 > 0)

theorem find_positive_product (h : a ≠ 0) (h' : b ≠ 0) (h'' : c ≠ 0) (h''' : d ≠ 0) (h'''' : e ≠ 0) (h''''' : f ≠ 0) 
  (exactly_one : exactly_one_positive (a * c * d) (a * c * e) (b * d * e) (b * d * f) (b * e * f)) :
  b * d * e > 0 :=
sorry

end find_positive_product_l391_391138


namespace minimum_value_l391_391464

theorem minimum_value (x : ℝ) (h : x > 1) : 2 * x + 7 / (x - 1) ≥ 2 * Real.sqrt 14 + 2 := by
  sorry

end minimum_value_l391_391464


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391388

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391388


namespace weng_babysitting_time_l391_391907

def hourly_rate : ℝ := 12
def earnings : ℝ := 10
def time_spent_hours : ℝ := earnings / hourly_rate
def time_spent_minutes : ℝ := time_spent_hours * 60

theorem weng_babysitting_time :
  time_spent_minutes = 50 := by
  sorry

end weng_babysitting_time_l391_391907


namespace value_of_4_Y_3_l391_391688

def Y (a b : ℕ) : ℕ := (2 * a ^ 2 - 3 * a * b + b ^ 2) ^ 2

theorem value_of_4_Y_3 : Y 4 3 = 25 := by
  sorry

end value_of_4_Y_3_l391_391688


namespace greatest_prime_factor_of_15_l391_391453

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391453


namespace greatest_prime_factor_of_15_l391_391445

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391445


namespace intersect_line_circle_slope_range_l391_391692

theorem intersect_line_circle_slope_range :
  ∀ (k : ℝ),
  (∃ x y : ℝ, y = k * x + 2 ∧ (x - 2)^2 + (y - 2)^2 = 1) ↔ k ∈ Icc (-real.sqrt 3 / 3) (real.sqrt 3 / 3) :=
by
  sorry

end intersect_line_circle_slope_range_l391_391692


namespace greatest_prime_factor_15f_plus_17f_l391_391194

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391194


namespace find_c_d_l391_391806

-- Define the set T
def T := { p : ℝ × ℝ × ℝ | ∃ x y z, p = (x, y, z) ∧ log 10 (3 * x + 2 * y) = z ∧ log 10 (x^3 + y^3) = 2 * z }

-- The proposition to prove
theorem find_c_d (x y z : ℝ) (h : (x, y, z) ∈ T) :
  ∃ c d : ℝ, (x^2 + y^2 = c * 10^(2*z) + d * 10^z) ∧ (c + d = 3 / 2) :=
sorry

end find_c_d_l391_391806


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391430

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391430


namespace inscribable_circle_l391_391947

-- Let O be the center of the circle
variables {O : point}
-- Let R be the radius of the circle
variables {R : ℝ}
-- Let a be half the length of the chord
variables {a : ℝ}
-- Let d be the distance from the center O to the sides of the quadrilateral
noncomputable def distance_to_side (O : point) (R a : ℝ) : ℝ :=
  real.sqrt (R^2 - a^2)

-- The theorem to prove
theorem inscribable_circle 
  (O : point) (R a : ℝ) (AB CD : line)
  (h1 : intercepted_by_equal_chords O R a AB CD) 
  : ∃ (d : ℝ), (d = distance_to_side O R a) ∧ (∀ side, equidistant_from_center O d side) := 
sorry

end inscribable_circle_l391_391947


namespace circumradius_of_triangle_l391_391949

theorem circumradius_of_triangle
  (a b c : ℝ) (h₁ : a = 8) (h₂ : b = 15) (h₃ : c = 17) (h_right_triangle : a^2 + b^2 = c^2) :
  let R := c / 2 in
  R = 17 / 2 :=
by
  -- Using the given conditions:
  sorry

end circumradius_of_triangle_l391_391949


namespace ellipse_is_correct_l391_391971

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = -1

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 16) = 1

-- Define the conditions
def ellipse_focus_vertex_of_hyperbola_vertex_and_focus (x y : ℝ) : Prop :=
  hyperbola_eq x y ∧ ellipse_eq x y

-- Theorem stating that the ellipse equation holds given the conditions
theorem ellipse_is_correct :
  ∀ (x y : ℝ), ellipse_focus_vertex_of_hyperbola_vertex_and_focus x y →
  ellipse_eq x y := by
  intros x y h
  sorry

end ellipse_is_correct_l391_391971


namespace greatest_prime_factor_15_fact_17_fact_l391_391176

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391176


namespace area_triangle_ABE_l391_391673

theorem area_triangle_ABE
  (A B C D E : Point)
  (h1 : collinear A B D)
  (h2 : collinear A C E)
  (h3 : collinear B D E)
  (h4 : distance A B = 10)
  (h5 : distance A C = 8)
  (h6 : distance B D = 5 * Real.sqrt 2)
  (h_circle : ∀ P : Point, distance P (midpoint A B) = distance A (midpoint A B)) : 
  area (triangle A B E) = 150 / 7 :=
by
  -- Proof needs to be filled in here
  sorry

end area_triangle_ABE_l391_391673


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391384

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391384


namespace smallest_largest_same_parity_l391_391572

-- Here, we define the conditions, including the set A_P and the element c achieving the maximum.
def is_maximum (A_P : Set Int) (c : Int) : Prop := c ∈ A_P ∧ ∀ x ∈ A_P, x ≤ c
def has_uniform_parity (A_P : Set Int) : Prop := 
  ∀ a₁ a₂ ∈ A_P, (a₁ % 2 = 0 → a₂ % 2 = 0) ∧ (a₁ % 2 = 1 → a₂ % 2 = 1)

-- This statement confirms the parity uniformity of the smallest and largest elements of the set A_P.
theorem smallest_largest_same_parity (A_P : Set Int) (c : Int) 
  (hc_max: is_maximum A_P c) (h_uniform: has_uniform_parity A_P): 
  ∀ min max ∈ A_P, ((min = max ∨ min ≠ max) → (min % 2 = max % 2)) := 
by
  intros min max hmin hmax h_eq
  have h_parity := h_uniform min max hmin hmax
  cases nat.decidable_eq (min % 2) 0 with h_even h_odd
  { rw nat.mod_eq_zero_of_dvd h_even at h_parity,
    exact h_parity.1 h_even, },
  { rw nat.mod_eq_one_of_dvd h_odd at h_parity,
    exact h_parity.2 h_odd, }
  sorry

end smallest_largest_same_parity_l391_391572


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391295

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391295


namespace complex_number_simplification_l391_391857

theorem complex_number_simplification (i : ℂ) (h : i^2 = -1) : i * (1 - i) - 1 = i := 
by
  sorry

end complex_number_simplification_l391_391857


namespace minimum_n_is_2_l391_391001

noncomputable def alpha (n : ℕ) : ℝ :=
  if n = 0 then 60.5 else 180 - 2 * beta (n - 1)

noncomputable def beta (n : ℕ) : ℝ :=
  if n = 0 then 60 else 180 - 2 * gamma (n - 1)

noncomputable def gamma (n : ℕ) : ℝ :=
  if n = 0 then 59.5 else 180 - 2 * alpha (n - 1)

def triangle_is_obtuse (n : ℕ) : Prop :=
  alpha n > 90 ∨ beta n > 90 ∨ gamma n > 90

theorem minimum_n_is_2 : ∃ n, triangle_is_obtuse n ∧ (∀ m, m < n → ¬ triangle_is_obtuse m) :=
begin
  use 2,
  split,
  { -- Show that at n = 2, the triangle becomes obtuse
    have : beta 2 = 62, by sorry,
    have : alpha 2 < 90, by sorry,
    have : gamma 2 < 90, by sorry,
    right,
    left,
    assumption,
  },
  { -- Show that for any m < 2, the triangle is not obtuse
    intros m hm,
    cases m,
    any_goals { simp [alpha, beta, gamma], linarith, },
    all_goals { simp [alpha, beta, gamma], linarith, },
  }
end

end minimum_n_is_2_l391_391001


namespace problem_proof_l391_391148

def geometric_sequence_positivity (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

def geometric_condition_1 (a : ℕ → ℝ) : Prop :=
  4 * a 1 - a 2 = 3

def geometric_condition_2 (a : ℕ → ℝ) : Prop :=
  a 5 ^ 2 = 9 * a 2 * a 6

noncomputable def general_formula (a : ℕ → ℝ) : ℕ → ℝ :=
  λ n, 3 ^ n

noncomputable def sequence_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (n + 1) - 3 + 3 ^ (n + 1)) / 2

theorem problem_proof
  (a : ℕ → ℝ)
  (h1 : geometric_sequence_positivity a)
  (h2 : geometric_condition_1 a)
  (h3 : geometric_condition_2 a) :
  (∀ n, a n = general_formula a n) ∧
  (∀ n, sequence_sum a (λ n, Real.log (a n) / Real.log 3) n = (n * (n + 1) - 3 + 3 ^ (n + 1)) / 2) :=
by
  sorry

end problem_proof_l391_391148


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391262

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391262


namespace petya_friends_l391_391041

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l391_391041


namespace quadratic_inequality_solution_l391_391648

variable {a : ℝ} (h_a_pos : a > 0)
variable {b c : ℝ} (h_b : b = -2 * a) (h_c : c = -8 * a)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution :
  ∀ x1 x2 x3 : ℝ,  (x1 = 2 ∧ x2 = -1 ∧ x3 = 5) →
  f x1 < f x2 ∧ f x2 < f x3 :=
by
  sorry

end quadratic_inequality_solution_l391_391648


namespace soccer_player_positions_exist_l391_391525

theorem soccer_player_positions_exist :
  ∃ x1 x2 x3 x4 : ℝ,
    ({| real.abs (x1 - x2),
       real.abs (x1 - x3),
       real.abs (x1 - x4),
       real.abs (x2 - x3),
       real.abs (x2 - x4),
       real.abs (x3 - x4) |} = {| 1, 2, 3, 4, 5, 6 |}) :=
begin
  use [0, 1, 4, 6],
  sorry
end

end soccer_player_positions_exist_l391_391525


namespace negation_of_rectangular_parallelepipeds_have_12_edges_l391_391875

-- Define a structure for Rectangular Parallelepiped and the property of having edges
structure RectangularParallelepiped where
  hasEdges : ℕ → Prop

-- Problem statement
theorem negation_of_rectangular_parallelepipeds_have_12_edges :
  (∀ rect_p : RectangularParallelepiped, rect_p.hasEdges 12) →
  ∃ rect_p : RectangularParallelepiped, ¬ rect_p.hasEdges 12 := 
by
  sorry

end negation_of_rectangular_parallelepipeds_have_12_edges_l391_391875


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391256

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391256


namespace farm_horses_cows_difference_l391_391030

-- Definitions based on provided conditions
def initial_ratio_horses_to_cows (horses cows : ℕ) : Prop := 5 * cows = horses
def transaction (horses cows sold bought : ℕ) : Prop :=
  horses - sold = 5 * cows - 15 ∧ cows + bought = cows + 15

-- Definitions to represent the ratios
def pre_transaction_ratio (horses cows : ℕ) : Prop := initial_ratio_horses_to_cows horses cows
def post_transaction_ratio (horses cows : ℕ) (sold bought : ℕ) : Prop :=
  transaction horses cows sold bought ∧ 7 * (horses - sold) = 17 * (cows + bought)

-- Statement of the theorem
theorem farm_horses_cows_difference :
  ∀ (horses cows : ℕ), 
    pre_transaction_ratio horses cows → 
    post_transaction_ratio horses cows 15 15 →
    (horses - 15) - (cows + 15) = 50 :=
by
  intros horses cows pre_ratio post_ratio
  sorry

end farm_horses_cows_difference_l391_391030


namespace prime_exponent_condition_l391_391931

theorem prime_exponent_condition (p a n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hn : 0 < n)
  (h : 2^p + 3^p = a^n) : n = 1 :=
sorry

end prime_exponent_condition_l391_391931


namespace geometric_sequence_ratio_l391_391804

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h0 : q ≠ 1) 
  (h1 : ∀ n, S n = a 0 * (1 - q^n) / (1 - q)) 
  (h2 : ∀ n, a n = a 0 * q^n) 
  (h3 : 2 * S 3 = 7 * a 2) :
  (S 5 / a 2 = 31 / 2) ∨ (S 5 / a 2 = 31 / 8) :=
by sorry

end geometric_sequence_ratio_l391_391804


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391318
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391318


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391426

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391426


namespace even_integers_count_l391_391681

def is_even (n : ℕ) : Prop := n % 2 = 0
def has_four_different_digits (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, n = 1000 * a + 100 * b + 10 * c + d ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem even_integers_count : 
  let nums := {n | 5000 ≤ n ∧ n < 8000 ∧ is_even n ∧ has_four_different_digits n} in
  nums.card = 784 :=
by 
  sorry

end even_integers_count_l391_391681


namespace greatest_prime_factor_15_17_factorial_l391_391347

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391347


namespace max_height_of_center_of_mass_l391_391510

theorem max_height_of_center_of_mass
  (m : ℝ) (V1 V2 : ℝ) (β α g : ℝ) :
  let Vc0_y := (1/2) * (V1 * Real.sin β + V2 * Real.sin α)
  in (1 / (2 * g)) * ((Vc0_y) ^ 2 / 4) = (1 / (2 * g)) * (1 / 4) * ((V1 * Real.sin β + V2 * Real.sin α) ^ 2) :=
by
  sorry

end max_height_of_center_of_mass_l391_391510


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391251

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391251


namespace population_percentage_l391_391946

theorem population_percentage (part whole : ℕ) (h1 : part = 23040) (h2 : whole = 28800) : 
  (part: ℝ) / whole * 100 = 80 :=
by
  rw [h1, h2]
  norm_num
  sorry

end population_percentage_l391_391946


namespace rightmost_three_digits_of_7_pow_2023_l391_391167

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 637 :=
sorry

end rightmost_three_digits_of_7_pow_2023_l391_391167


namespace greatest_prime_factor_of_sum_l391_391392

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391392


namespace only_prime_in_sequence_is_67_l391_391986

/-- Define the sequence of numbers composed by repetition of '67'. -/
def sequence (n : ℕ) : ℕ := 67 * (10^(2*n) - 1) / 99

/-- Prove that the only prime number in the sequence sequence(n) for n = 1 to 10 is 67. -/
theorem only_prime_in_sequence_is_67 : 
  (∀ n, 1 ≤ n → n ≤ 10 → ¬ prime (sequence n)) ∧ prime (sequence 1) := 
by
  sorry

end only_prime_in_sequence_is_67_l391_391986


namespace gcd_m_n_l391_391005

def m : ℕ := 333333333
def n : ℕ := 9999999999

theorem gcd_m_n : Nat.gcd m n = 9 := by
  sorry

end gcd_m_n_l391_391005


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391265

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391265


namespace soccer_player_positions_exist_l391_391524

theorem soccer_player_positions_exist :
  ∃ x1 x2 x3 x4 : ℝ,
    ({| real.abs (x1 - x2),
       real.abs (x1 - x3),
       real.abs (x1 - x4),
       real.abs (x2 - x3),
       real.abs (x2 - x4),
       real.abs (x3 - x4) |} = {| 1, 2, 3, 4, 5, 6 |}) :=
begin
  use [0, 1, 4, 6],
  sorry
end

end soccer_player_positions_exist_l391_391524


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391436

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391436


namespace greatest_prime_factor_of_sum_l391_391399

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391399


namespace function_max_min_values_l391_391661

theorem function_max_min_values :
  ∃ (a b : ℝ), 
    (∀ x : ℝ, x = 1 → f x = 3) ∧ (∃ x : ℝ, f'(x) = 0) ∧
    (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f(2) = -12 ∧ f(-1) = 15) :=
sorry

end function_max_min_values_l391_391661


namespace greatest_prime_factor_15_17_factorial_l391_391350

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391350


namespace train_speed_l391_391969

/-- Proof problem: Speed calculation of a train -/
theorem train_speed :
  ∀ (length : ℝ) (time_seconds : ℝ) (speed_kmph : ℝ),
    length = 40 →
    time_seconds = 0.9999200063994881 →
    speed_kmph = (length / 1000) / (time_seconds / 3600) →
    speed_kmph = 144 :=
by
  intros length time_seconds speed_kmph h_length h_time_seconds h_speed_kmph
  rw [h_length, h_time_seconds] at h_speed_kmph
  -- sorry is used to skip the proof steps
  sorry 

end train_speed_l391_391969


namespace greatest_prime_factor_15_17_factorial_l391_391341

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391341


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391380

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391380


namespace chests_content_l391_391743

-- Define the chests and their labels
inductive CoinContent where
  | gold : CoinContent
  | silver : CoinContent
  | copper : CoinContent

structure Chest where
  label : CoinContent
  contents : CoinContent

-- Given conditions and incorrect labels
def chest1 : Chest := { label := CoinContent.gold, contents := sorry }
def chest2 : Chest := { label := CoinContent.silver, contents := sorry }
def chest3 : Chest := { label := CoinContent.gold, contents := sorry }

-- The proof problem
theorem chests_content :
  chest1.contents ≠ CoinContent.gold ∧
  chest2.contents ≠ CoinContent.silver ∧
  chest3.contents ≠ CoinContent.gold ∨ chest3.contents ≠ CoinContent.silver →
  chest1.contents = CoinContent.silver ∧
  chest2.contents = CoinContent.gold ∧
  chest3.contents = CoinContent.copper := by
  sorry

end chests_content_l391_391743


namespace percentage_boys_not_attended_college_l391_391513

/-
Define the constants and given conditions.
-/
def number_of_boys : ℕ := 300
def number_of_girls : ℕ := 240
def total_students : ℕ := number_of_boys + number_of_girls
def percentage_class_attended_college : ℝ := 0.70
def percentage_girls_not_attended_college : ℝ := 0.30

/-
The proof problem statement: 
Prove the percentage of the boys class that did not attend college.
-/
theorem percentage_boys_not_attended_college :
  let students_attended_college := percentage_class_attended_college * total_students
  let not_attended_college_students := total_students - students_attended_college
  let not_attended_college_girls := percentage_girls_not_attended_college * number_of_girls
  let not_attended_college_boys := not_attended_college_students - not_attended_college_girls
  let percentage_boys_not_attended_college := (not_attended_college_boys / number_of_boys) * 100
  percentage_boys_not_attended_college = 30 := by
  sorry

end percentage_boys_not_attended_college_l391_391513


namespace passing_marks_l391_391924

theorem passing_marks
  (T P : ℝ)
  (h1 : 0.20 * T = P - 40)
  (h2 : 0.30 * T = P + 20) :
  P = 160 :=
by
  sorry

end passing_marks_l391_391924


namespace expected_balls_in_original_positions_six_l391_391844

noncomputable def expected_balls_in_original_positions :
  ℕ := 6

def probability_never_swapped :
  ℚ := (4 / 6) ^ 3

theorem expected_balls_in_original_positions_six :
  expected_balls_in_original_positions * probability_never_swapped = 48 / 27 :=
by 
  simp [expected_balls_in_original_positions, probability_never_swapped]
  norm_num
sorry

end expected_balls_in_original_positions_six_l391_391844


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391412

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391412


namespace total_tickets_sold_l391_391902

def cost_per_adult : ℕ := 21
def cost_per_senior : ℕ := 15
def total_receipts : ℕ := 8748
def senior_tickets_sold : ℕ := 327

theorem total_tickets_sold : 
  let total_from_seniors := senior_tickets_sold * cost_per_senior in
  let total_from_adults := total_receipts - total_from_seniors in
  let adult_tickets_sold := total_from_adults / cost_per_adult in
  let total_tickets := adult_tickets_sold + senior_tickets_sold in
  total_tickets = 510 := 
by
  sorry

end total_tickets_sold_l391_391902


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391296

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391296


namespace petya_friends_l391_391040

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l391_391040


namespace smallest_largest_same_parity_l391_391574

-- Here, we define the conditions, including the set A_P and the element c achieving the maximum.
def is_maximum (A_P : Set Int) (c : Int) : Prop := c ∈ A_P ∧ ∀ x ∈ A_P, x ≤ c
def has_uniform_parity (A_P : Set Int) : Prop := 
  ∀ a₁ a₂ ∈ A_P, (a₁ % 2 = 0 → a₂ % 2 = 0) ∧ (a₁ % 2 = 1 → a₂ % 2 = 1)

-- This statement confirms the parity uniformity of the smallest and largest elements of the set A_P.
theorem smallest_largest_same_parity (A_P : Set Int) (c : Int) 
  (hc_max: is_maximum A_P c) (h_uniform: has_uniform_parity A_P): 
  ∀ min max ∈ A_P, ((min = max ∨ min ≠ max) → (min % 2 = max % 2)) := 
by
  intros min max hmin hmax h_eq
  have h_parity := h_uniform min max hmin hmax
  cases nat.decidable_eq (min % 2) 0 with h_even h_odd
  { rw nat.mod_eq_zero_of_dvd h_even at h_parity,
    exact h_parity.1 h_even, },
  { rw nat.mod_eq_one_of_dvd h_odd at h_parity,
    exact h_parity.2 h_odd, }
  sorry

end smallest_largest_same_parity_l391_391574


namespace prime_factor_count_l391_391604

theorem prime_factor_count (n : ℕ) (H : 22 + n + 2 = 29) : n = 5 := 
  sorry

end prime_factor_count_l391_391604


namespace min_value_of_expression_l391_391019

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 27) 
  : x + 3 * y + 9 * z ≥ 27 :=
sorry

end min_value_of_expression_l391_391019


namespace operation_result_l391_391581

theorem operation_result :
  (-2) ※ (-1) = -1 :=
by 
  let operation := fun (a b : ℤ) => b^2 - a * b
  show operation (-2) (-1) = -1
  sorry

end operation_result_l391_391581


namespace find_common_ratio_l391_391886

noncomputable theory

def geometric_sequence_problem (a1 q : ℝ) (S3 : ℝ) : Prop :=
  S3 = a1 * (1 + q + q^2) ∧ 2 * (2 + a1 * q) = a1 + a1 * q^2

theorem find_common_ratio (a1 : ℝ) (q S3 : ℝ) (h : geometric_sequence_problem a1 q S3) : q = 3 ∨ q = 1/3 :=
by
  cases h with hS3 hGeo
  have eq1 : S3 = a1 * (1 + q + q^2) := hS3
  have eq2 : 2 * (2 + a1 * q) = a1 + a1 * q^2 := hGeo
  sorry

end find_common_ratio_l391_391886


namespace greatest_prime_factor_of_sum_l391_391394

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391394


namespace greatest_prime_factor_15_17_factorial_l391_391204

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391204


namespace smallest_largest_same_parity_l391_391567

-- Define the set A_P
def A_P : Set ℕ := sorry

-- Define the maximum element c in A_P
def c := max A_P

-- Assuming shifting doesn't change fundamental counts
lemma shifting_preserves_counts (A : Set ℕ) (c : ℕ) : 
  (∀ x ∈ A, x ≠ c → x < c → exists y ∈ A, y < x) ∧
  (∃ p ∈ A, ∃ q ∈ A, p ≠ q ∧ p < q) :=
  sorry

-- Define parity
def parity (n : ℕ) := n % 2

-- Theorem statement
theorem smallest_largest_same_parity (A : Set ℕ) (a_max a_min : ℕ) 
  (h_max : a_max = max A) (h_min : a_min = min A)
  (h_shift : ∀ x ∈ A, shifting_preserves_counts A x) :
  parity a_max = parity a_min :=
sorry

end smallest_largest_same_parity_l391_391567


namespace convert_1234_base_10_to_4_l391_391989

-- Define the conversion from base 10 to base 4
def convert_base_10_to_4 (n : ℕ) : ℕ :=
  let rec help (n : ℕ) (acc : ℕ) (base : ℕ) : ℕ :=
    if n = 0 then acc
    else help (n / base) (acc + (n % base) * base ^ ((Nat.log n).quot (Nat.log base))) base
  help n 0 4

-- Assert that converting 1234 from base 10 to base 4 yields 34102
theorem convert_1234_base_10_to_4 : convert_base_10_to_4 1234 = 34102 := by
  sorry

end convert_1234_base_10_to_4_l391_391989


namespace greatest_prime_factor_of_factorial_sum_l391_391331

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391331


namespace domain_of_f_l391_391123

def f (x : ℝ) := Real.log (x^2 - x)

theorem domain_of_f :
  {x : ℝ | x^2 - x > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end domain_of_f_l391_391123


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391292

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391292


namespace mohan_cookies_l391_391824

theorem mohan_cookies :
  ∃ a : ℕ, 
    a % 4 = 3 ∧
    a % 5 = 2 ∧
    a % 7 = 4 ∧
    a = 67 :=
by
  -- The proof will be written here.
  sorry

end mohan_cookies_l391_391824


namespace additional_money_needed_l391_391576

theorem additional_money_needed (cost_of_perfume : ℝ) (christian_initial_savings : ℝ) (sue_initial_savings : ℝ) 
  (christian_yards : ℕ) (christian_rate : ℝ) (sue_dogs : ℕ) (sue_rate : ℝ) : 
  (cost_of_perfume - (christian_initial_savings + christian_yards * christian_rate + 
                      sue_initial_savings + sue_dogs * sue_rate) = 3) :=
by {
  -- Given values
  let cost_of_perfume := 75,
  let christian_initial_savings := 5,
  let sue_initial_savings := 7,
  let christian_yards := 6,
  let christian_rate := 6,
  let sue_dogs := 8,
  let sue_rate := 3,

  -- Calculations
  let christian_earnings := christian_yards * christian_rate,
  let sue_earnings := sue_dogs * sue_rate,
  let total_christian := christian_initial_savings + christian_earnings,
  let total_sue := sue_initial_savings + sue_earnings,
  let total_savings := total_christian + total_sue,
  let additional_needed := cost_of_perfume - total_savings,

  -- Check for the correct answer
  have : additional_needed = 3 := rfl,
  
  exact this
}

end additional_money_needed_l391_391576


namespace smallest_largest_same_parity_l391_391563

-- Define the context where our elements and set are considered
variable {α : Type*} [LinearOrderedCommRing α] (A_P : Set α)

-- Assume a nonempty set A_P and define the smallest and largest elements in Lean
noncomputable def smallest_element (A_P : Set α) : α := Inf' A_P sorry
noncomputable def largest_element (A_P : Set α) : α := Sup' A_P sorry

-- State the proof goal
theorem smallest_largest_same_parity (h_nonempty : A_P.nonempty) : 
  (smallest_element A_P) % 2 = (largest_element A_P) % 2 := 
by 
  sorry

end smallest_largest_same_parity_l391_391563


namespace greatest_prime_factor_15f_plus_17f_l391_391185

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391185


namespace soccer_players_positions_l391_391531

noncomputable def positions : list ℝ := [0, 1, 4, 6]

def pairwise_distances (positions : list ℝ) : list ℝ :=
  let pairs := list.sigma positions positions
  let distances := pairs.map (λ p, abs (p.1 - p.2))
  distances.erase_dup

theorem soccer_players_positions :
  pairwise_distances positions = [1, 2, 3, 4, 5, 6] :=
by {
  sorry  -- Proof to be provided
}

end soccer_players_positions_l391_391531


namespace solve_x_l391_391926

theorem solve_x (x : ℝ) (h : 9 - 4 / x = 7 + 8 / x) : x = 6 := 
by 
  sorry

end solve_x_l391_391926


namespace Petya_friends_l391_391086

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l391_391086


namespace area_A0_and_sum_A0_to_A8_l391_391883

theorem area_A0_and_sum_A0_to_A8 (w4 : ℝ) (h4 : ℝ) (rat : w4/h4 = 1/√2) (A0_area : ℝ) (sum_areas : ℝ) :
  w4 = 2 → (∀ (i : ℕ), i ≤ 8 → A0_area = 64 * √2) → 
  sum_areas = 64 * √2 * (1 - (1 / 2) ^ 9) / (1 - 1 / 2) := 
sorry

#check area_A0_and_sum_A0_to_A8

end area_A0_and_sum_A0_to_A8_l391_391883


namespace parity_uniformity_l391_391551

-- Define the set A_P
variable {A_P : Set ℤ}

-- Conditions:
-- 1. A_P is non-empty
noncomputable def non_empty (H : A_P ≠ ∅) := H

-- 2. c is the maximum element in A_P
variable {c : ℤ}
variable (H_max : ∀ a ∈ A_P, a ≤ c)

-- 3. Consideration of critical points around c
variable {f : ℤ → ℤ}
variable (H_critical : ∀ x ∈ A_P, f x = 0)

-- 4. Parity of the smallest and largest elements
def parity (n : ℤ) : Prop := n % 2 = 0 ∨ n % 2 = 1

-- Proof statement
theorem parity_uniformity (H_non_empty : non_empty A_P)
  (H_max_element : ∀ a ∈ A_P, a ≤ c)
  (H_critical_points : ∀ x ∈ A_P, f x = 0) :
  (∃ x ∈ A_P, ∀ y ∈ A_P, x ≤ y) → (parity (x : ℤ) = parity ((y : ℤ) : ℤ)) → (least x ∈ A_P, greatest y ∈ A_P, parity x = parity y) :=
by
  sorry

end parity_uniformity_l391_391551


namespace circle_equation_center_on_line_tangent_at_point_typed_l391_391128

theorem circle_equation_center_on_line_tangent_at_point_typed
(center_on_line : ∀ a b : ℝ, b = -4 * a) 
(tangent_to_line : ∀ x y : ℝ, x + y = 1)
(tangent_point : (3, -2))
: ∃ a b : ℝ, ((b = -4 * a) ∧ (a = 1) ∧ (b = -4)) → 
((x - 1)^2 + (y + 4)^2 = 8) :=
begin
  sorry
end

end circle_equation_center_on_line_tangent_at_point_typed_l391_391128


namespace extra_men_needed_l391_391972

theorem extra_men_needed
  (total_length : ℕ) (total_days : ℕ) (initial_men : ℕ)
  (completed_days : ℕ) (completed_work : ℕ) (remaining_work : ℕ)
  (remaining_days : ℕ) (total_man_days_needed : ℕ)
  (number_of_men_needed : ℕ) (extra_men_needed : ℕ)
  (h1 : total_length = 10)
  (h2 : total_days = 60)
  (h3 : initial_men = 30)
  (h4 : completed_days = 20)
  (h5 : completed_work = 2)
  (h6 : remaining_work = total_length - completed_work)
  (h7 : remaining_days = total_days - completed_days)
  (h8 : total_man_days_needed = remaining_work * (completed_days * initial_men) / completed_work)
  (h9 : number_of_men_needed = total_man_days_needed / remaining_days)
  (h10 : extra_men_needed = number_of_men_needed - initial_men)
  : extra_men_needed = 30 :=
by sorry

end extra_men_needed_l391_391972


namespace select_three_numbers_condition_l391_391611

theorem select_three_numbers_condition:
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
  ∃ (a1 a2 a3 ∈ S), a1 < a2 ∧ a2 < a3 ∧ a2 - a1 ≥ 3 ∧ a3 - a2 ≥ 3 → 
  (∃ n, n = 120) :=
by
  sorry

end select_three_numbers_condition_l391_391611


namespace tutors_meet_again_l391_391895

theorem tutors_meet_again (tim uma victor xavier: ℕ) (h1: tim = 5) (h2: uma = 6) (h3: victor = 9) (h4: xavier = 8) :
  Nat.lcm (Nat.lcm tim uma) (Nat.lcm victor xavier) = 360 := 
by 
  rw [h1, h2, h3, h4]
  show Nat.lcm (Nat.lcm 5 6) (Nat.lcm 9 8) = 360
  sorry

end tutors_meet_again_l391_391895


namespace modulus_of_z_l391_391650

noncomputable def z : ℂ := (1 + complex.I) / complex.I

theorem modulus_of_z : complex.abs z = real.sqrt 2 := sorry

end modulus_of_z_l391_391650


namespace greatest_prime_factor_of_factorial_sum_l391_391327

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391327


namespace frac_add_eq_l391_391461

theorem frac_add_eq : (2 / 5) + (3 / 10) = 7 / 10 := 
by
  sorry

end frac_add_eq_l391_391461


namespace petya_friends_count_l391_391062

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l391_391062


namespace smallest_largest_same_parity_l391_391562

-- Define the context where our elements and set are considered
variable {α : Type*} [LinearOrderedCommRing α] (A_P : Set α)

-- Assume a nonempty set A_P and define the smallest and largest elements in Lean
noncomputable def smallest_element (A_P : Set α) : α := Inf' A_P sorry
noncomputable def largest_element (A_P : Set α) : α := Sup' A_P sorry

-- State the proof goal
theorem smallest_largest_same_parity (h_nonempty : A_P.nonempty) : 
  (smallest_element A_P) % 2 = (largest_element A_P) % 2 := 
by 
  sorry

end smallest_largest_same_parity_l391_391562


namespace greatest_prime_factor_15_17_factorial_l391_391351

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391351


namespace distance_to_other_focus_ellipse_l391_391830

theorem distance_to_other_focus_ellipse
  (P : ℝ × ℝ) (x y : ℝ)
  (h_ellipse : x^2 / 36 + y^2 / 9 = 1)
  (a b : ℝ) (h_a : a = 6) (h_b : b = 3)
  (c : ℝ) (h_c_squared : c^2 = a^2 - b^2)
  (h_c : c = real.sqrt (27))
  (d₁ : ℝ) (h_distance_P_to_focus1 : d₁ = 5) :
  ∃ d₂ : ℝ, d₂ = 7 := by
  sorry

end distance_to_other_focus_ellipse_l391_391830


namespace scientific_notation_equivalence_l391_391871

-- Define constants and variables
def scientific_notation {a b : ℝ} (n : ℝ) (a b : ℝ) := n = a * (10^b)

-- State the conditions
def seven_nm_equals := (7 : ℝ) * (10 : ℝ) ^ (-9 : ℝ) = 0.000000007

-- Theorem to prove
theorem scientific_notation_equivalence : scientific_notation 0.000000007 7 (-9) :=
by
  apply (seven_nm_equals)

end scientific_notation_equivalence_l391_391871


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391424

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391424


namespace greatest_prime_factor_of_sum_factorials_l391_391270

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391270


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391288

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391288


namespace ratio_IP_IQ_l391_391801

theorem ratio_IP_IQ (ABC : Type) [triangle : Triangle ABC] (I : incenter ABC) (P Q T : ABC) 
  (h1 : LineThrough I ⊥ LineThrough A I)
  (h2 : LiesOnCircumcircle P ABC)
  (h3 : LiesOnCircumcircle Q ABC)
  (h4 : P ≠ Q)
  (h5 : LiesOnSide T B C)
  (h6 : AB + BT = AC + CT)
  (h7 : AT^2 = AB * AC):
  IP / IQ = 1/2 ∨ IP / IQ = 2 :=
sorry

end ratio_IP_IQ_l391_391801


namespace greatest_prime_factor_of_sum_factorials_l391_391278

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391278


namespace less_than_reciprocal_l391_391918

theorem less_than_reciprocal (x : ℝ) (h : x ∈ { -1/2, -3, 1/3, 3, 3/2 }) :
  x < 1/x ↔ x = -3 ∨ x = 1/3 :=
by { sorry }

end less_than_reciprocal_l391_391918


namespace braid_time_l391_391730

theorem braid_time (dancers : ℕ) (braids_per_dancer : ℕ) (time_per_braid_sec : ℝ) (prep_time_min : ℝ) :
  dancers = 15 →
  braids_per_dancer = 10 →
  time_per_braid_sec = 45 →
  prep_time_min = 5 →
  let braid_time_min := time_per_braid_sec / 60 in
  let time_per_dancer := braid_time_min * braids_per_dancer + prep_time_min in
  time_per_dancer * dancers = 187.5 :=
by
  intros h1 h2 h3 h4
  let braid_time_min := time_per_braid_sec / 60
  let time_per_dancer := braid_time_min * braids_per_dancer + prep_time_min
  calc
    time_per_dancer * dancers
      = (braid_time_min * braids_per_dancer + prep_time_min) * dancers : by rw [h1, h2, h3, h4]
    ... = (0.75 * 10 + 5) * 15 : by norm_num
    ... = 187.5 : by norm_num

end braid_time_l391_391730


namespace susan_chairs_l391_391112

theorem susan_chairs : 
  ∀ (red yellow blue : ℕ), 
  red = 5 → 
  yellow = 4 * red → 
  blue = yellow - 2 → 
  red + yellow + blue = 43 :=
begin
  intros red yellow blue h1 h2 h3,
  rw h1,
  rw h2,
  rw h3,
  norm_num,
end

end susan_chairs_l391_391112


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391417

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391417


namespace inequality_proof_l391_391837

theorem inequality_proof (a b c : ℝ) (h : a ^ 2 + b ^ 2 + c ^ 2 = 3) :
  (a ^ 2) / (2 + b + c ^ 2) + (b ^ 2) / (2 + c + a ^ 2) + (c ^ 2) / (2 + a + b ^ 2) ≥ (a + b + c) ^ 2 / 12 :=
by sorry

end inequality_proof_l391_391837


namespace greatest_prime_factor_15_17_factorial_l391_391213

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391213


namespace triangle_area_l391_391828

variables {A B C D M N: Type}

-- Define the conditions and the proof 
theorem triangle_area
  (α β : ℝ)
  (CD : ℝ)
  (sin_Ratio : ℝ)
  (C_angle : ℝ)
  (MCN_Area : ℝ)
  (M_distance : ℝ)
  (N_distance : ℝ)
  (hCD : CD = Real.sqrt 13)
  (hSinRatio : (Real.sin α) / (Real.sin β) = 4 / 3)
  (hC_angle : C_angle = 120)
  (hMCN_Area : MCN_Area = 3 * Real.sqrt 3)
  (hDistance : M_distance = 2 * N_distance)
  : ∃ ABC_Area, ABC_Area = 27 * Real.sqrt 3 / 2 :=
sorry

end triangle_area_l391_391828


namespace chest_contents_correct_l391_391764

-- Define the chests
inductive Chest
| chest1
| chest2
| chest3

open Chest

-- Define the contents
inductive Contents
| gold
| silver
| copper

open Contents

-- Define the labels, all of which are incorrect
def label (c : Chest) : Prop :=
  match c with
  | chest1 => gold
  | chest2 => silver
  | chest3 => gold ∨ silver

-- Assuming the labels are all incorrect
axiom label_incorrect : ∀ (c : Chest), ¬(label c = true)

-- One chest contains each type of coin
axiom one_gold : ∃ c, ∀ x, x ≠ c → Contents x ≠ gold
axiom one_silver : ∃ c, ∀ x, x ≠ c → Contents x ≠ silver
axiom one_copper : ∃ c, ∀ x, x ≠ c → Contents x ≠ copper

-- Determine the contents of each chest
def chest1_contents := silver
def chest2_contents := gold
def chest3_contents := copper

-- Prove the correspondence
theorem chest_contents_correct :
  (chest1_contents = silver) ∧ 
  (chest2_contents = gold) ∧ 
  (chest3_contents = copper) :=
by
  split
  · exact sorry
  split
  · exact sorry
  · exact sorry

end chest_contents_correct_l391_391764


namespace intervals_of_decrease_l391_391870

open Real

noncomputable def func (x : ℝ) : ℝ :=
  cos (2 * x) + 2 * sin x

theorem intervals_of_decrease :
  {x | deriv func x < 0 ∧ 0 < x ∧ x < 2 * π} =
  {x | (π / 6 < x ∧ x < π / 2) ∨ (5 * π / 6 < x ∧ x < 3 * π / 2)} :=
by
  sorry

end intervals_of_decrease_l391_391870


namespace cost_per_liter_l391_391709

/-
Given:
- Service cost per vehicle: $2.10
- Number of mini-vans: 3
- Number of trucks: 2
- Total cost: $299.1
- Mini-van's tank size: 65 liters
- Truck's tank is 120% bigger than a mini-van's tank
- All tanks are empty

Prove that the cost per liter of fuel is $0.60
-/

theorem cost_per_liter (service_cost_per_vehicle : ℝ) 
(number_of_minivans number_of_trucks : ℕ)
(total_cost : ℝ)
(minivan_tank_size : ℝ)
(truck_tank_multiplier : ℝ)
(fuel_cost : ℝ)
(total_fuel : ℝ) :
  service_cost_per_vehicle = 2.10 ∧
  number_of_minivans = 3 ∧
  number_of_trucks = 2 ∧
  total_cost = 299.1 ∧
  minivan_tank_size = 65 ∧
  truck_tank_multiplier = 1.2 ∧
  fuel_cost = (total_cost - (number_of_minivans + number_of_trucks) * service_cost_per_vehicle) ∧
  total_fuel = (number_of_minivans * minivan_tank_size + number_of_trucks * (minivan_tank_size * (1 + truck_tank_multiplier))) →
  (fuel_cost / total_fuel) = 0.60 :=
sorry

end cost_per_liter_l391_391709


namespace greatest_prime_factor_of_sum_factorials_l391_391282

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391282


namespace average_monthly_balance_l391_391471

-- Definitions for the monthly balances
def January_balance : ℝ := 120
def February_balance : ℝ := 240
def March_balance : ℝ := 180
def April_balance : ℝ := 180
def May_balance : ℝ := 160
def June_balance : ℝ := 200

-- The average monthly balance theorem statement
theorem average_monthly_balance : 
    (January_balance + February_balance + March_balance + April_balance + May_balance + June_balance) / 6 = 180 := 
by 
  sorry

end average_monthly_balance_l391_391471


namespace greatest_prime_factor_15f_plus_17f_l391_391197

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391197


namespace soccer_players_arrangement_l391_391519

theorem soccer_players_arrangement : ∃ (x1 x2 x3 x4 : ℝ), 
    let dists := {(abs (x1 - x2)), (abs (x1 - x3)), (abs (x1 - x4)), (abs (x2 - x3)), (abs (x2 - x4)), (abs (x3 - x4))} in
    dists = {1, 2, 3, 4, 5, 6} :=
sorry

end soccer_players_arrangement_l391_391519


namespace algebra_identity_example_l391_391578

theorem algebra_identity_example (a b : ℕ) (h1 : a = 25) (h2 : b = 15) :
  (a + b)^2 - (a - b)^2 = 1500 :=
by
  -- Apply the algebraic identity
  have h : (a + b)^2 - (a - b)^2 = 4 * a * b := sorry
  -- Use the given values of a and b
  rw [h1, h2] at h
  -- Conclude the result
  rw h
  norm_num

end algebra_identity_example_l391_391578


namespace length_PZ_l391_391717

-- Define the given conditions
variables (CD WX : ℝ) -- segments CD and WX
variable (CW : ℝ) -- length of segment CW
variable (DP : ℝ) -- length of segment DP
variable (PX : ℝ) -- length of segment PX

-- Define the similarity condition
-- segment CD is parallel to segment WX implies that the triangles CDP and WXP are similar

-- Define what we want to prove
theorem length_PZ (hCD_WX_parallel : CD = WX)
                  (hCW : CW = 56)
                  (hDP : DP = 18)
                  (hPX : PX = 36) :
  ∃ PZ : ℝ, PZ = 4 / 3 :=
by
  -- proof steps here (omitted)
  sorry

end length_PZ_l391_391717


namespace greatest_prime_factor_15_17_factorial_l391_391353

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391353


namespace number_of_operations_l391_391011

-- Define the structure and properties of points and segments in the plane.
structure Point :=
  (x : ℝ) (y : ℝ)

structure Segment :=
  (a b : Point)
  -- segments are defined by two points

def collinear (p1 p2 p3 : Point) : Prop :=
  -- collinearity can be checked by the determinant method
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

def interior_point (s1 s2 : Segment) (p : Point) : Prop :=
  -- Define an interior point for segments
  sorry -- Placeholder for the definition

def replace_segments (s1 s2 s3 s4 : Segment) : Segment × Segment :=
  -- Replace s1 and s2 with s3 and s4
  (s3, s4)

-- Given n points, no three collinear, each connected by two segments initially:
def initial_segments (points : Finset Point) (segments : Finset Segment) (n : ℕ) : Prop :=
  points.card = n ∧
  ∀ (p : Point), p ∈ points → (∃! s1 s2 : Segment, s1.a = p ∨ s1.b = p ∧ s2.a = p ∨ s2.b = p)

theorem number_of_operations (n : ℕ) (points : Finset Point) (segments : Finset Segment)
    (h_points : points.card = n)
    (h_collinear : ∀ (p1 p2 p3 : Point), (p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points) → ¬ collinear p1 p2 p3)
    (h_initial : initial_segments points segments n)
    (h_segments : ∀ (s1 s2 : Segment), s1 ∈ segments → s2 ∈ segments →
      (∃ p : Point, interior_point s1 s2 p) →
      ∃ s3 s4 : Segment, (replace_segments s1 s2 s3 s4 ∧ s3 ∉ segments ∧ s4 ∉ segments))
  : true :=
begin
  sorry
end

end number_of_operations_l391_391011


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391320
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391320


namespace greatest_prime_factor_15_fact_17_fact_l391_391234

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391234


namespace tetrahedron_area_correct_l391_391628

-- Define the properties of the tetrahedron
def equilateral_triangle_area (a : ℝ) : ℝ := (real.sqrt 3) / 4 * a^2

noncomputable def tetrahedron_surface_area (edge_length : ℝ) : ℝ :=
  4 * equilateral_triangle_area edge_length

-- Statement of the proof problem
theorem tetrahedron_area_correct :
  tetrahedron_surface_area 2 = 4 * real.sqrt 3 :=
by
  sorry

end tetrahedron_area_correct_l391_391628


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391423

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391423


namespace complete_the_square_solution_l391_391465

theorem complete_the_square_solution (x : ℝ) :
  (∃ x, x^2 + 2 * x - 1 = 0) → (x + 1)^2 = 2 :=
sorry

end complete_the_square_solution_l391_391465


namespace Z_is_all_positive_integers_l391_391483

theorem Z_is_all_positive_integers (Z : Set ℕ) (h_nonempty : Z.Nonempty)
(h1 : ∀ x ∈ Z, 4 * x ∈ Z)
(h2 : ∀ x ∈ Z, (Nat.sqrt x) ∈ Z) : 
Z = { n : ℕ | n > 0 } :=
sorry

end Z_is_all_positive_integers_l391_391483


namespace chests_contents_l391_391736

def ChestLabel (c : ℕ) : String := 
  if c = 1 then "Gold coins"
  else if c = 2 then "Silver coins"
  else if c = 3 then "Gold or silver coins"
  else "Invalid chest"

def CoinsInChest (c : ℕ) : String := 
  if c = 1 then "Silver coins"
  else if c = 2 then "Gold coins"
  else if c = 3 then "Copper coins"
  else "Invalid chest"

theorem chests_contents :
  ChestLabel 1 ≠ CoinsInChest 1 ∧
  ChestLabel 2 ≠ CoinsInChest 2 ∧
  ChestLabel 3 ≠ CoinsInChest 3 ∧
  (CoinsInChest 1 = "Gold coins" ∨ CoinsInChest 2 = "Gold coins" ∨ CoinsInChest 3 = "Gold coins") ∧
  (CoinsInChest 1 = "Silver coins" ∨ CoinsInChest 2 = "Silver coins" ∨ CoinsInChest 3 = "Silver coins") ∧
  (CoinsInChest 1 = "Copper coins" ∨ CoinsInChest 2 = "Copper coins" ∨ CoinsInChest 3 = "Copper coins") :=
begin
  sorry
end

end chests_contents_l391_391736


namespace student_marks_l391_391967

theorem student_marks :
  let max_marks := 300
  let passing_percentage := 0.60
  let failed_by := 20
  let passing_marks := max_marks * passing_percentage
  let marks_obtained := passing_marks - failed_by
  marks_obtained = 160 := by
sorry

end student_marks_l391_391967


namespace members_in_both_A_and_B_l391_391890

-- Define the sets and the total number of members.
def U := fin 193
def A (u : U) : Prop  := sorry -- predicate for membership in A
def B (u : U) : Prop  := sorry -- predicate for membership in B

-- Define the conditions based on the given problem.
def members_in_set_B := 49
def members_not_in_set_A_or_B := 59
def members_in_set_A := 110

theorem members_in_both_A_and_B : ∃ x : ℕ, (x = 25) ∧
  (members_in_set_A - x) + (members_in_set_B - x) + x + members_not_in_set_A_or_B = 193 :=
by
  use 25
  split 
  · rfl
  · sorry

end members_in_both_A_and_B_l391_391890


namespace a_17_eq_13_l391_391667

-- Define the initial term a_1
def a1 : ℕ → ℚ := λ n, 1

-- Define the sequence a_n
def a (n : ℕ) : ℚ :=
if h : n = 0 then 1
else nat.rec_on n.succ (1 : ℚ) (λ n a_n, (4 * a_n + 3) / 4)

-- Prove a_{17} = 13
theorem a_17_eq_13 : a 17 = 13 :=
by
  sorry

end a_17_eq_13_l391_391667


namespace circles_externally_tangent_l391_391143

/-- Define a structure for the circle with center and radius -/
structure Circle where
  center : (ℝ × ℝ)
  radius : ℝ

/-- Define the two circles with the given centers and radii -/
def circle1 : Circle := {center := (0, 0), radius := 2}
def circle2 : Circle := {center := (3, 4), radius := 3}

/-- Define the function to calculate the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

/-- The positional relationship between the two circles is externally tangent -/
theorem circles_externally_tangent :
  distance circle1.center circle2.center = circle1.radius + circle2.radius :=
by
  sorry

end circles_externally_tangent_l391_391143


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391427

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391427


namespace incircles_tangent_l391_391607

variables {α : Type*} [ordered_comm_group α]

structure Quadrilateral (α : Type*) :=
  (A B C D : α)

structure Triangle (α : Type*) :=
  (A B C : α)

-- Conditions
variables (ABCD : Quadrilateral α)
variables (h_convex : convex ABCD.ABC ABCD.BCD ABCD.CDA ABCD.DAB)
variables (h_cond : ABCD.AB + ABCD.CD = ABCD.BC + ABCD.DA)

-- Define Incircle tangency in triangles 
def incircle_tangent {α : Type*} (t1 t2 : Triangle α) : Prop :=
  ∃ K : α, tangent_point t1.AC = K ∧ tangent_point t2.AC = K

noncomputable def tangent_to_each_other : Prop :=
  incircle_tangent (Triangle.mk ABCD.A ABCD.B ABCD.C) (Triangle.mk ABCD.A ABCD.C ABCD.D)

-- Lean statement for proof
theorem incircles_tangent :
  tangent_to_each_other (Triangle.mk ABCD.A ABCD.B ABCD.C) (Triangle.mk ABCD.A ABCD.C ABCD.D)
  := sorry

end incircles_tangent_l391_391607


namespace probability_at_least_three_heads_l391_391587

noncomputable def coin_toss_pmf : ProbabilityMassFunction ℕ :=
  ProbabilityMassFunction.ofFinite (λ i, if i ≤ 5 then (nat.succ_pmf 5).pmf i else 0)

def prob_three_or_more_heads : ℕ → ℝ
  | 5 => (∀ k, k ≥ 3 → coin_toss_pmf pmf k).sum

theorem probability_at_least_three_heads :
  prob_three_or_more_heads 5 = 1 / 2 :=
by
  sorry

end probability_at_least_three_heads_l391_391587


namespace smallest_n_coprime_groups_l391_391013

open Finset

def S : Finset ℕ := range 99 \ {0}

theorem smallest_n_coprime_groups (T : Finset ℕ) (hT : ∀ T ⊆ S, T.card = 50 → 
  ∃ X ⊆ T, X.card = 10 ∧ 
  (∀ G₁ G₂ : Finset ℕ, G₁ ∪ G₂ = X ∧ G₁.card = 5 ∧ G₂.card = 5 → 
    (∃ x ∈ G₁, ∀ y ∈ G₁ \ {x}, Nat.Coprime x y) ∨ 
    (∃ x ∈ G₂, ∀ y ∈ G₂ \ {x}, ¬Nat.Coprime x y))) : ∀ (n : ℕ), n = 50 := 
sorry

end smallest_n_coprime_groups_l391_391013


namespace num_schools_city_of_Euclid_l391_391592

/-- Define the high school contest scenario -/
def high_school_contest :=
  ∃ (num_students : ℕ) (num_schools : ℕ),
    (∀ s, num_students = 4 * num_schools) ∧
    (∀ scores : Fin num_students → ℝ, Injective scores) ∧
    ∃ (Andrea Beth Carla Dan : ℕ),
      (Beth = 47) ∧
      (Carla = 75) ∧
      (Dan = 98) ∧
      (Andrea < Beth) ∧
      (Beth < Carla) ∧
      (Carla < Dan) ∧
      (∃ median_score : ℝ, median_score = (scores (num_students / 2) + scores (num_students / 2 - 1)) / 2) ∧
      (∀ i < num_schools, HighScore i = Andrea)

/-- Prove that the number of schools in this city is 24, given the conditions -/
theorem num_schools_city_of_Euclid : ∃ num_schools, num_schools = 24 :=
by
  sorry

end num_schools_city_of_Euclid_l391_391592


namespace arithmetic_sequence_divisible_by_four_l391_391968

theorem arithmetic_sequence_divisible_by_four :
  let digits := set.range 11
  let is_arithmetic_sequence (a b c : ℕ) := b - a = c - b
  let sum_divisible_by_four (a b c : ℕ) := (a + b + c) % 4 = 0
  set.count
    (λ s : finset ℕ, ∃ a b c, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
      s = {a, b, c} ∧ is_arithmetic_sequence a b c ∧ sum_divisible_by_four a b c)
    (finset.powerset (finset.range 11)) = 6 :=
begin
  sorry
end

end arithmetic_sequence_divisible_by_four_l391_391968


namespace least_product_xy_l391_391637

theorem least_product_xy : ∀ (x y : ℕ), 0 < x → 0 < y →
  (1 : ℚ) / x + (1 : ℚ) / (3 * y) = 1 / 6 → x * y = 48 :=
by
  intros x y x_pos y_pos h
  sorry

end least_product_xy_l391_391637


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391363

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391363


namespace sufficient_condition_implies_range_of_p_l391_391936

open Set Real

theorem sufficient_condition_implies_range_of_p (p : ℝ) :
  (∀ x : ℝ, 4 * x + p < 0 → x^2 - x - 2 > 0) →
  (∃ x : ℝ, x^2 - x - 2 > 0 ∧ ¬ (4 * x + p < 0)) →
  p ∈ Set.Ici 4 :=
by
  sorry

end sufficient_condition_implies_range_of_p_l391_391936


namespace sequence_type_l391_391930

open Nat

theorem sequence_type (a : ℕ → ℝ) (p : ℝ) (S : ℕ → ℝ) 
  (h₁ : 4 ≤ length a) 
  (h₂ : ∀ n : ℕ, n > 0 → S n = n * p * a n)
  (h₃ : a 1 ≠ a 2) : 
  a 1 = 0 ∧ (∀ n : ℕ, n ≥ 2 → a n = (n - 1) * a 2) := 
sorry

end sequence_type_l391_391930


namespace greatest_prime_factor_of_factorial_sum_l391_391335

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391335


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391310
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391310


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391429

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391429


namespace car_speed_in_mph_l391_391473

theorem car_speed_in_mph :
  (∀ (ℓ : ℝ), ℓ ≠ 0 → 56 * ℓ * (3.9 * 3.8) / 1.6 / 5.7 = 91) :=
by
  intro ℓ hℓ
  have liters_used : ℝ := 3.9 * 3.8
  have distance_km : ℝ := 56 * liters_used
  have distance_mi : ℝ := distance_km / 1.6
  have time_h : ℝ := 5.7
  have speed_mph : ℝ := distance_mi / time_h
  rw [mul_assoc 56 liters_used (1 / 1.6), ← div_eq_mul_one_div distance_km 1.6, mul_comm liter_used 56, div_div, div_eq_mul_one_div distance_mi time_h)
  sorry

end car_speed_in_mph_l391_391473


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391255

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391255


namespace sum_cubed_powers_l391_391020

noncomputable def z : ℂ := (1 + complex.i) / real.sqrt 2

theorem sum_cubed_powers :
  (∑ k in finset.range 8, z ^ (k + 1) ^ 3) * (∑ k in finset.range 8, (z ^ (k + 1) ^ 3)⁻¹) = 64 :=
by sorry

end sum_cubed_powers_l391_391020


namespace chests_contents_l391_391739

def ChestLabel (c : ℕ) : String := 
  if c = 1 then "Gold coins"
  else if c = 2 then "Silver coins"
  else if c = 3 then "Gold or silver coins"
  else "Invalid chest"

def CoinsInChest (c : ℕ) : String := 
  if c = 1 then "Silver coins"
  else if c = 2 then "Gold coins"
  else if c = 3 then "Copper coins"
  else "Invalid chest"

theorem chests_contents :
  ChestLabel 1 ≠ CoinsInChest 1 ∧
  ChestLabel 2 ≠ CoinsInChest 2 ∧
  ChestLabel 3 ≠ CoinsInChest 3 ∧
  (CoinsInChest 1 = "Gold coins" ∨ CoinsInChest 2 = "Gold coins" ∨ CoinsInChest 3 = "Gold coins") ∧
  (CoinsInChest 1 = "Silver coins" ∨ CoinsInChest 2 = "Silver coins" ∨ CoinsInChest 3 = "Silver coins") ∧
  (CoinsInChest 1 = "Copper coins" ∨ CoinsInChest 2 = "Copper coins" ∨ CoinsInChest 3 = "Copper coins") :=
begin
  sorry
end

end chests_contents_l391_391739


namespace dice_probability_l391_391942

/-- 
Given five 15-sided dice, we want to prove the probability of exactly two dice showing a two-digit number (10-15) and three dice showing a one-digit number (1-9) is equal to 108/625.
-/
theorem dice_probability:
  let p_one_digit := 9 / 15 in
  let p_two_digit := 6 / 15 in
  (∃ f : (Fin 5) → Bool,
    (∃ s : Finset (Fin 5), s.card = 2 ∧ (∀ i ∈ s, f i = true) ∧ (∀ i ∉ s, f i = false))
      ∧ (bern_prob (prob {i | f i = false}) = p_one_digit)
      ∧ (bern_prob (prob {i | f i = true}) = p_two_digit)) →
  ((5.choose 2) * (p_two_digit ^ 2) * (p_one_digit ^ 3)) = 108 / 625 :=
by
  sorry

end dice_probability_l391_391942


namespace simplify_fraction_l391_391481

theorem simplify_fraction :
  (6 * x ^ 3 + 13 * x ^ 2 + 15 * x - 25) / (2 * x ^ 3 + 4 * x ^ 2 + 4 * x - 10) =
  (6 * x - 5) / (2 * x - 2) :=
by
  sorry

end simplify_fraction_l391_391481


namespace sum_of_powers_l391_391577

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem sum_of_powers (h: ω ^ 17 = 1) : (∑ k in Finset.range 16, ω^(k+1)) = -1 :=
by
  sorry

end sum_of_powers_l391_391577


namespace more_than_half_millet_on_day_three_l391_391827

-- Definition of the initial conditions
def seeds_in_feeder (n: ℕ) : ℝ :=
  1 + n

def millet_amount (n: ℕ) : ℝ :=
  0.6 * (1 - (0.5)^n)

-- The theorem we want to prove
theorem more_than_half_millet_on_day_three :
  ∀ n, n = 3 → (millet_amount n) / (seeds_in_feeder n) > 0.5 :=
by
  intros n hn
  rw [hn, seeds_in_feeder, millet_amount]
  sorry

end more_than_half_millet_on_day_three_l391_391827


namespace least_product_xy_l391_391636

theorem least_product_xy : ∀ (x y : ℕ), 0 < x → 0 < y →
  (1 : ℚ) / x + (1 : ℚ) / (3 * y) = 1 / 6 → x * y = 48 :=
by
  intros x y x_pos y_pos h
  sorry

end least_product_xy_l391_391636


namespace greatest_prime_factor_of_sum_factorials_l391_391277

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391277


namespace chests_content_l391_391748

-- Define the chests and their labels
inductive CoinContent where
  | gold : CoinContent
  | silver : CoinContent
  | copper : CoinContent

structure Chest where
  label : CoinContent
  contents : CoinContent

-- Given conditions and incorrect labels
def chest1 : Chest := { label := CoinContent.gold, contents := sorry }
def chest2 : Chest := { label := CoinContent.silver, contents := sorry }
def chest3 : Chest := { label := CoinContent.gold, contents := sorry }

-- The proof problem
theorem chests_content :
  chest1.contents ≠ CoinContent.gold ∧
  chest2.contents ≠ CoinContent.silver ∧
  chest3.contents ≠ CoinContent.gold ∨ chest3.contents ≠ CoinContent.silver →
  chest1.contents = CoinContent.silver ∧
  chest2.contents = CoinContent.gold ∧
  chest3.contents = CoinContent.copper := by
  sorry

end chests_content_l391_391748


namespace soccer_player_positions_exist_l391_391528

theorem soccer_player_positions_exist :
  ∃ x1 x2 x3 x4 : ℝ,
    ({| real.abs (x1 - x2),
       real.abs (x1 - x3),
       real.abs (x1 - x4),
       real.abs (x2 - x3),
       real.abs (x2 - x4),
       real.abs (x3 - x4) |} = {| 1, 2, 3, 4, 5, 6 |}) :=
begin
  use [0, 1, 4, 6],
  sorry
end

end soccer_player_positions_exist_l391_391528


namespace petya_friends_l391_391077

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l391_391077


namespace petya_friends_count_l391_391061

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l391_391061


namespace find_y_given_conditions_l391_391856

def is_value_y (x y : ℕ) : Prop :=
  (100 + 200 + 300 + x) / 4 = 250 ∧ (300 + 150 + 100 + x + y) / 5 = 200

theorem find_y_given_conditions : ∃ y : ℕ, ∀ x : ℕ, (100 + 200 + 300 + x) / 4 = 250 ∧ (300 + 150 + 100 + x + y) / 5 = 200 → y = 50 :=
by
  sorry

end find_y_given_conditions_l391_391856


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391303

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391303


namespace calculate_expression_l391_391689

theorem calculate_expression (x y : ℚ) (hx : x = 5 / 6) (hy : y = 6 / 5) : 
  (1 / 3) * (x ^ 8) * (y ^ 9) = 2 / 5 :=
by
  sorry

end calculate_expression_l391_391689


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391300

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391300


namespace train_stop_time_per_hour_l391_391476

theorem train_stop_time_per_hour :
  ∀ (speed_excluding_stoppages speed_including_stoppages : ℝ)
    (h1 : speed_excluding_stoppages = 30)
    (h2 : speed_including_stoppages = 21),
    let distance_lost := speed_excluding_stoppages - speed_including_stoppages,
        time_lost := distance_lost / speed_excluding_stoppages in
    (time_lost * 60) = 18 :=
by
  intros speed_excluding_stoppages speed_including_stoppages
  intros h1 h2
  dsimp only
  rw [h1, h2]
  norm_num
  sorry

end train_stop_time_per_hour_l391_391476


namespace petya_friends_l391_391091

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l391_391091


namespace no_consecutive_integers_with_square_difference_2000_l391_391860

theorem no_consecutive_integers_with_square_difference_2000 :
  ¬ ∃ (x : ℤ), (x + 1)^2 - x^2 = 2000 := by
begin
  intro h,
  cases h with x hx,
  have : (x + 1)^2 - x^2 = 2 * x + 1, by ring,
  rw this at hx,
  linarith,
end

end no_consecutive_integers_with_square_difference_2000_l391_391860


namespace range_a_condition_l391_391812

theorem range_a_condition (a : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ a → x^2 ≤ 2 * x + 3) ↔ (1 / 2 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_a_condition_l391_391812


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391240

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391240


namespace b_plus_c_is_square_l391_391879

-- Given the conditions:
variables (a b c : ℕ)
variable (h1 : a > 0 ∧ b > 0 ∧ c > 0)  -- Condition 1: Positive integers
variable (h2 : Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1)  -- Condition 2: Pairwise relatively prime
variable (h3 : a % 2 = 1 ∧ c % 2 = 1)  -- Condition 3: a and c are odd
variable (h4 : a^2 + b^2 = c^2)  -- Condition 4: Pythagorean triple equation

-- Prove that b + c is the square of an integer
theorem b_plus_c_is_square : ∃ k : ℕ, b + c = k^2 :=
by
  sorry

end b_plus_c_is_square_l391_391879


namespace volunteer_selection_probability_l391_391959

theorem volunteer_selection_probability :
  ∀ (students total_students remaining_students selected_volunteers : ℕ),
    total_students = 2018 →
    remaining_students = total_students - 18 →
    selected_volunteers = 50 →
    (selected_volunteers : ℚ) / total_students = (25 : ℚ) / 1009 :=
by
  intros students total_students remaining_students selected_volunteers
  intros h1 h2 h3
  sorry

end volunteer_selection_probability_l391_391959


namespace correct_statement_is_C_l391_391916

theorem correct_statement_is_C : 
  (∀ (a : ℝ), ¬(sqrt (5 * a) = (λ x, sqrt x) (a * (λ y, 1 + y^2) a / 1))) ∧
  (∀ (m : ℝ), sqrt (m^2 + 1) = (λ x, sqrt x) (m^2 + 1)) ∧
  (∀ (x y : ℝ) (h : x^2 = y^2), x = y ∨ x = -y) ∧
  (∀ (k : ℝ), ¬(is_quadratic_radical k → irrational k))
  → true := 
by 
  sorry

end correct_statement_is_C_l391_391916


namespace soccer_players_positions_l391_391530

noncomputable def positions : list ℝ := [0, 1, 4, 6]

def pairwise_distances (positions : list ℝ) : list ℝ :=
  let pairs := list.sigma positions positions
  let distances := pairs.map (λ p, abs (p.1 - p.2))
  distances.erase_dup

theorem soccer_players_positions :
  pairwise_distances positions = [1, 2, 3, 4, 5, 6] :=
by {
  sorry  -- Proof to be provided
}

end soccer_players_positions_l391_391530


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391317
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391317


namespace seq_eventually_reaches_one_l391_391960

theorem seq_eventually_reaches_one (a : ℕ → ℤ) (h₁ : a 1 > 0) :
  (∀ n, n % 4 = 0 → a (n + 1) = a n / 2) →
  (∀ n, n % 4 = 1 → a (n + 1) = 3 * a n + 1) →
  (∀ n, n % 4 = 2 → a (n + 1) = 2 * a n - 1) →
  (∀ n, n % 4 = 3 → a (n + 1) = (a n + 1) / 4) →
  ∃ m, a m = 1 :=
by
  sorry

end seq_eventually_reaches_one_l391_391960


namespace find_positive_product_l391_391137

variable (a b c d e f : ℝ)

-- Define the condition that exactly one of the products is positive
def exactly_one_positive (p1 p2 p3 p4 p5 : ℝ) : Prop :=
  (p1 > 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 > 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 > 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 > 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 > 0)

theorem find_positive_product (h : a ≠ 0) (h' : b ≠ 0) (h'' : c ≠ 0) (h''' : d ≠ 0) (h'''' : e ≠ 0) (h''''' : f ≠ 0) 
  (exactly_one : exactly_one_positive (a * c * d) (a * c * e) (b * d * e) (b * d * f) (b * e * f)) :
  b * d * e > 0 :=
sorry

end find_positive_product_l391_391137


namespace parity_uniformity_l391_391552

-- Define the set A_P
variable {A_P : Set ℤ}

-- Conditions:
-- 1. A_P is non-empty
noncomputable def non_empty (H : A_P ≠ ∅) := H

-- 2. c is the maximum element in A_P
variable {c : ℤ}
variable (H_max : ∀ a ∈ A_P, a ≤ c)

-- 3. Consideration of critical points around c
variable {f : ℤ → ℤ}
variable (H_critical : ∀ x ∈ A_P, f x = 0)

-- 4. Parity of the smallest and largest elements
def parity (n : ℤ) : Prop := n % 2 = 0 ∨ n % 2 = 1

-- Proof statement
theorem parity_uniformity (H_non_empty : non_empty A_P)
  (H_max_element : ∀ a ∈ A_P, a ≤ c)
  (H_critical_points : ∀ x ∈ A_P, f x = 0) :
  (∃ x ∈ A_P, ∀ y ∈ A_P, x ≤ y) → (parity (x : ℤ) = parity ((y : ℤ) : ℤ)) → (least x ∈ A_P, greatest y ∈ A_P, parity x = parity y) :=
by
  sorry

end parity_uniformity_l391_391552


namespace polygonal_superposition_possible_l391_391672

variable {P Q : Type} [metric_space P] [metric_space Q]
variable {polygonal_domain : P → Prop} {is_identical : P → Q → Prop}
variable {not_superpose_trans_rot : P → Q → Prop}
variable {finite_splitting : (P → Prop) → (Q → Prop)}

theorem polygonal_superposition_possible :
  ∀ (P Q : Type) [metric_space P] [metric_space Q], 
    polygonal_domain P →
    polygonal_domain Q →
    is_identical P Q →
    not_superpose_trans_rot P Q →
    ∃ (finite_splitting : (P → Prop) → (Q → Prop)),
      (∃ subdomains : list (P → Prop),
        (∀ sd ∈ subdomains, polygonal_domain sd) ∧
        (∀ sd ∈ subdomains, ∃ t : P → Q, rotation_or_translation t ∧ t sd = Q)) :=
by
  sorry

end polygonal_superposition_possible_l391_391672


namespace guayaquilean_sum_of_digits_l391_391954

def is_guayaquilean (n : ℕ) : Prop :=
  sum_of_digits n = sum_of_digits (n * n)

theorem guayaquilean_sum_of_digits (n : ℕ) (hn : is_guayaquilean n) :
  ∃ k : ℕ, sum_of_digits n = 9 * k ∨ sum_of_digits n = 9 * k + 1 :=
sorry

end guayaquilean_sum_of_digits_l391_391954


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391406

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391406


namespace magnitude_of_z_l391_391489

open Complex

theorem magnitude_of_z (z : ℂ) (h : z - 2 + Complex.i = 1) : Complex.abs z = Real.sqrt 10 :=
sorry

end magnitude_of_z_l391_391489


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391302

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391302


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391414

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391414


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391361

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391361


namespace find_distance_between_vertices_l391_391802

-- Define the vertex of a parabola given in the form y = x^2 + bx + c
def vertex (b c : ℕ) : ℕ × ℕ :=
  ((-b / 2), (4 * (c - (b * b) / 4)))

def distance (p1 p2 : ℕ × ℕ) : ℕ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem find_distance_between_vertices :
  let C := vertex 6 15,
      D := vertex -4 8
  in distance C D = 29 := by
  sorry

end find_distance_between_vertices_l391_391802


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391358

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391358


namespace greatest_prime_factor_15_17_factorial_l391_391216

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391216


namespace total_pages_in_book_l391_391610

-- Definitions from the conditions
def chapters : ℕ := 5
def pages_per_chapter : ℕ := 111

-- Statement of the problem as a theorem
theorem total_pages_in_book : (chapters * pages_per_chapter) = 555 :=
by
  calc
    chapters * pages_per_chapter = 5 * 111 : by rw [chapters, pages_per_chapter]
                         ...               = 555 : by norm_num

end total_pages_in_book_l391_391610


namespace find_constants_find_max_value_l391_391618

open Real

-- Defining the function and conditions
def f (x : ℝ) (a b c : ℝ) := a * x ^ 3 + b * x ^ 2 + c * x

-- Conditions
variables (a b c : ℝ) (h1 : f 1 a b c = -1) (h2 : a ≠ 0)
variables (h3 : 3 * a + 2 * b + c = 0) (h4 : 3 * a - 2 * b + c = 0)

-- Objective (Ⅰ)
theorem find_constants : 
  a = 1 / 2 ∧ b = 0 ∧ c = -3 / 2 := sorry
  
-- Substituting the values found
def f_specific (x : ℝ) := (1 / 2) * x ^ 3 - (3 / 2) * x

-- Objective (Ⅱ)
theorem find_max_value : 
  ∃ x ∈ Icc 0 2, ∀ y ∈ Icc 0 2, f_specific x ≥ f_specific y := 
  ⟨2, ⟨by norm_num, by norm_num⟩, 
  λ y hy, by cases hy with hy₁ hy₂; norm_num [f_specific, hy₁, hy₂]⟩

end find_constants_find_max_value_l391_391618


namespace product_positivity_l391_391135

theorem product_positivity (a b c d e f : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)
  (h : ∀ {x y z u v w : ℝ}, x * y * z > 0 → ¬(u * v * w > 0) ∧ ¬(x * v * u > 0) ∧ ¬(y * z * u > 0) ∧ ¬(y * v * x > 0)) :
  b * d * e > 0 ∧ a * c * d < 0 ∧ a * c * e < 0 ∧ b * d * f < 0 ∧ b * e * f < 0 := 
by
  sorry

end product_positivity_l391_391135


namespace smallest_largest_same_parity_l391_391565

-- Define the set A_P
def A_P : Set ℕ := sorry

-- Define the maximum element c in A_P
def c := max A_P

-- Assuming shifting doesn't change fundamental counts
lemma shifting_preserves_counts (A : Set ℕ) (c : ℕ) : 
  (∀ x ∈ A, x ≠ c → x < c → exists y ∈ A, y < x) ∧
  (∃ p ∈ A, ∃ q ∈ A, p ≠ q ∧ p < q) :=
  sorry

-- Define parity
def parity (n : ℕ) := n % 2

-- Theorem statement
theorem smallest_largest_same_parity (A : Set ℕ) (a_max a_min : ℕ) 
  (h_max : a_max = max A) (h_min : a_min = min A)
  (h_shift : ∀ x ∈ A, shifting_preserves_counts A x) :
  parity a_max = parity a_min :=
sorry

end smallest_largest_same_parity_l391_391565


namespace smallest_largest_same_parity_l391_391573

-- Here, we define the conditions, including the set A_P and the element c achieving the maximum.
def is_maximum (A_P : Set Int) (c : Int) : Prop := c ∈ A_P ∧ ∀ x ∈ A_P, x ≤ c
def has_uniform_parity (A_P : Set Int) : Prop := 
  ∀ a₁ a₂ ∈ A_P, (a₁ % 2 = 0 → a₂ % 2 = 0) ∧ (a₁ % 2 = 1 → a₂ % 2 = 1)

-- This statement confirms the parity uniformity of the smallest and largest elements of the set A_P.
theorem smallest_largest_same_parity (A_P : Set Int) (c : Int) 
  (hc_max: is_maximum A_P c) (h_uniform: has_uniform_parity A_P): 
  ∀ min max ∈ A_P, ((min = max ∨ min ≠ max) → (min % 2 = max % 2)) := 
by
  intros min max hmin hmax h_eq
  have h_parity := h_uniform min max hmin hmax
  cases nat.decidable_eq (min % 2) 0 with h_even h_odd
  { rw nat.mod_eq_zero_of_dvd h_even at h_parity,
    exact h_parity.1 h_even, },
  { rw nat.mod_eq_one_of_dvd h_odd at h_parity,
    exact h_parity.2 h_odd, }
  sorry

end smallest_largest_same_parity_l391_391573


namespace max_height_of_center_of_mass_l391_391511

theorem max_height_of_center_of_mass
  (m : ℝ) (V1 V2 : ℝ) (β α g : ℝ) :
  let Vc0_y := (1/2) * (V1 * Real.sin β + V2 * Real.sin α)
  in (1 / (2 * g)) * ((Vc0_y) ^ 2 / 4) = (1 / (2 * g)) * (1 / 4) * ((V1 * Real.sin β + V2 * Real.sin α) ^ 2) :=
by
  sorry

end max_height_of_center_of_mass_l391_391511


namespace probability_event_occurs_l391_391903

noncomputable def integral_x_squared (a : ℝ) : ℝ :=
  ∫ x in 0..a, x^2

def event_occurs (a : ℝ) : Prop :=
  integral_x_squared a > 1 / 81

def uniform_distribution (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 1

theorem probability_event_occurs :
  ∀ (a : ℝ), uniform_distribution a → (has_measure (set_of (λ a, event_occurs a)) (0..1) (2 / 3)) :=
by sorry

end probability_event_occurs_l391_391903


namespace petya_friends_l391_391044

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l391_391044


namespace find_a_l391_391657

def function_f (a x : ℝ) := (x / log x) + a * x

def derivative_f (a x : ℝ) := ((log x - 1) / (log x)^2) + a

theorem find_a (a : ℝ) : (∀ x > 1, derivative_f a x ≤ 0) → a ≤ -1/4 := sorry

end find_a_l391_391657


namespace petya_friends_l391_391078

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l391_391078


namespace total_right_handed_players_l391_391833

theorem total_right_handed_players (total_players throwers : ℕ) (frac_left_handers : ℚ) 
  (h1 : total_players = 120) (h2 : throwers = 67) (h3 : frac_left_handers = 2/5) 
  (h4 : ∀ t ∈ range throwers, t > 0) 
  (h5 : ∀ t ∈ range total_players, t ≠ throwers → t = total_players - throwers) :
  let non_throwers := total_players - throwers in
  let left_handers := nat.floor ((frac_left_handers : ℚ) * (non_throwers : ℚ)) in
  let right_handers := non_throwers - left_handers in
  let total_right_handers := throwers + right_handers in
  total_right_handers = 99 := 
by
  sorry

end total_right_handed_players_l391_391833


namespace number_of_diagonals_pentagon_difference_hexagon_pentagon_difference_successive_polygons_l391_391140

noncomputable def a (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem number_of_diagonals_pentagon : a 5 = 5 := sorry

theorem difference_hexagon_pentagon : a 6 - a 5 = 4 := sorry

theorem difference_successive_polygons (n : ℕ) (h : 4 ≤ n) : a (n + 1) - a n = n - 1 := sorry

end number_of_diagonals_pentagon_difference_hexagon_pentagon_difference_successive_polygons_l391_391140


namespace petya_friends_l391_391094

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l391_391094


namespace jacques_suitcase_weight_l391_391725

noncomputable def suitcase_weight_on_return : ℝ := 
  let initial_weight := 12
  let perfume_weight := (5 * 1.2) / 16
  let chocolate_weight := 4 + 1.5 + 3.25
  let soap_weight := (2 * 5) / 16
  let jam_weight := (8 + 6 + 10 + 12) / 16
  let sculpture_weight := 3.5 * 2.20462
  let shirts_weight := (3 * 300 * 0.03527396) / 16
  let cookies_weight := (450 * 0.03527396) / 16
  let wine_weight := (190 * 0.03527396) / 16
  initial_weight + perfume_weight + chocolate_weight + soap_weight + jam_weight + sculpture_weight + shirts_weight + cookies_weight + wine_weight

theorem jacques_suitcase_weight : suitcase_weight_on_return = 35.111288 := 
by 
  -- Calculation to verify that the total is 35.111288
  sorry

end jacques_suitcase_weight_l391_391725


namespace koschei_chests_l391_391755

theorem koschei_chests :
  ∃ (contents : Fin 3 → String), 
    -- All chests labels are incorrect
    (contents 0 ≠ "gold coins" ∧ contents 1 ≠ "silver coins" ∧ contents 2 ≠ "gold or silver coins") ∧ 
    -- Each chest contains exactly one type of coin 
    (∀ i j : Fin 3, i ≠ j → contents i ≠ contents j) ∧ 
    -- Providing the final conclusion about what is in each chest
    (contents 0 = "silver coins" ∧ contents 1 = "gold coins" ∧ contents 2 = "copper coins") :=
begin
  use (λ k, if k = 0 then "silver coins" else if k = 1 then "gold coins" else "copper coins"),
  split,
  {
    -- Proof of all labels being incorrect
    split; simp,
  },
  split,
  {
    -- Proof of each chest containing a unique type of coin
    intros i j h,
    cases i; cases j; simp [h],
  },
  {
    -- Proof of the final conclusion
    split; split; simp,
  },
end

end koschei_chests_l391_391755


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391385

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391385


namespace greatest_prime_factor_of_15_l391_391449

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391449


namespace time_to_cross_tree_is_120_seconds_l391_391943

-- Define the problem parameters based on the given conditions
def train_length : ℝ := 1200           -- The length of the train is 1200 meters
def platform_length : ℝ := 600         -- The length of the platform is 600 meters
def time_to_pass_platform : ℝ := 180   -- It takes 180 seconds to pass the platform

-- Define the formula for the speed of the train passing the platform
def speed : ℝ := (train_length + platform_length) / time_to_pass_platform

-- Calculate the time it takes to cross the tree using the train's speed
def time_to_cross_tree : ℝ := train_length / speed

-- State the theorem
theorem time_to_cross_tree_is_120_seconds : time_to_cross_tree = 120 := by
  sorry

end time_to_cross_tree_is_120_seconds_l391_391943


namespace solve_for_y_l391_391848

theorem solve_for_y (y : ℤ) (h : 4 * 5^y = 2500) : y = 4 :=
by
  sorry

end solve_for_y_l391_391848


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391439

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391439


namespace triangle_area_l391_391126

-- Define the polynomial equation
def polymial_eq (x : ℝ) : Prop := x^3 - 3 * x^2 + 4 * x - (8 / 5) = 0

-- Define the real roots
variables (a b c : ℝ)
axiom roots : polymial_eq a ∧ polymial_eq b ∧ polymial_eq c

-- Define the problem statement
theorem triangle_area : 
  (roots a b c) →
  a + b + c = 3 →
  let q := (a + b + c) / 2 in
  q * (q - a) * (q - b) * (q - c) = 12 / 5 :=
sorry

end triangle_area_l391_391126


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391370

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391370


namespace greatest_prime_factor_15_fact_17_fact_l391_391225

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391225


namespace relationship_x_y_l391_391643

theorem relationship_x_y (a b c : ℝ) (h₀ : a > b) (h₁ : b > c) (h₂ : x = Real.sqrt ((a - b) * (b - c))) (h₃ : y = (a - c) / 2) : 
  x ≤ y :=
by
  sorry

end relationship_x_y_l391_391643


namespace percentage_cut_second_week_l391_391965

noncomputable def calculate_final_weight (initial_weight : ℝ) (percentage1 : ℝ) (percentage2 : ℝ) (percentage3 : ℝ) : ℝ :=
  let weight_after_first_week := (1 - percentage1 / 100) * initial_weight
  let weight_after_second_week := (1 - percentage2 / 100) * weight_after_first_week
  let final_weight := (1 - percentage3 / 100) * weight_after_second_week
  final_weight

theorem percentage_cut_second_week : 
  ∀ (initial_weight : ℝ) (final_weight : ℝ), (initial_weight = 250) → (final_weight = 105) →
    (calculate_final_weight initial_weight 30 x 25 = final_weight) → 
    x = 20 := 
by 
  intros initial_weight final_weight h1 h2 h3
  sorry

end percentage_cut_second_week_l391_391965


namespace tenth_number_in_row_1_sum_of_2023rd_numbers_l391_391028

noncomputable def a (n : ℕ) := (-2)^n
noncomputable def b (n : ℕ) := a n + (n + 1)

theorem tenth_number_in_row_1 : a 10 = (-2)^10 := 
sorry

theorem sum_of_2023rd_numbers : a 2023 + b 2023 = -(2^2024) + 2024 := 
sorry

end tenth_number_in_row_1_sum_of_2023rd_numbers_l391_391028


namespace math_problem_proof_l391_391716

noncomputable def C_1 : (ℝ × ℝ) → Prop :=
  λ p, let (x, y) := p in x^2 + y^2 = 1

noncomputable def scaling_transformation : (ℝ × ℝ) → (ℝ × ℝ') :=
  λ p, let (x, y) := p in (3 * x, 2 * y)

noncomputable def C_2 : (ℝ × ℝ) → Prop :=
  λ p, let (x, y) := p in (x^2 / 9) + (y^2 / 4) = 1

noncomputable def polar_line (ρ θ : ℝ) : Prop :=
  cos θ + 2 * sin θ = 10 / ρ

noncomputable def cartesian_line : (ℝ × ℝ) → Prop :=
  λ p, let (x, y) := p in x + 2 * y = 10

noncomputable def point_on_C2 (M : ℝ × ℝ) : Prop :=
  let (x, y) := M in (x = 9 / 5) ∧ (y = 8 / 5)

noncomputable def distance_to_line (M : ℝ × ℝ) : ℝ :=
  let (x, y) := M in |x + 2 * y - 10| / sqrt 5

theorem math_problem_proof :
  ∃ M : ℝ × ℝ, C_2 M ∧ point_on_C2 M ∧ distance_to_line M = sqrt 5 :=
begin
  sorry
end

end math_problem_proof_l391_391716


namespace petya_friends_l391_391096

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l391_391096


namespace chests_contents_l391_391740

def ChestLabel (c : ℕ) : String := 
  if c = 1 then "Gold coins"
  else if c = 2 then "Silver coins"
  else if c = 3 then "Gold or silver coins"
  else "Invalid chest"

def CoinsInChest (c : ℕ) : String := 
  if c = 1 then "Silver coins"
  else if c = 2 then "Gold coins"
  else if c = 3 then "Copper coins"
  else "Invalid chest"

theorem chests_contents :
  ChestLabel 1 ≠ CoinsInChest 1 ∧
  ChestLabel 2 ≠ CoinsInChest 2 ∧
  ChestLabel 3 ≠ CoinsInChest 3 ∧
  (CoinsInChest 1 = "Gold coins" ∨ CoinsInChest 2 = "Gold coins" ∨ CoinsInChest 3 = "Gold coins") ∧
  (CoinsInChest 1 = "Silver coins" ∨ CoinsInChest 2 = "Silver coins" ∨ CoinsInChest 3 = "Silver coins") ∧
  (CoinsInChest 1 = "Copper coins" ∨ CoinsInChest 2 = "Copper coins" ∨ CoinsInChest 3 = "Copper coins") :=
begin
  sorry
end

end chests_contents_l391_391740


namespace decode_rebus_l391_391992

-- Definitions for digits and letters.
def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9
def unique_digits (digits : List ℕ) : Prop := (digits ≠ List.nil) ∧ digits.nodup

-- Given conditions
variables {A P M И Р : ℕ}

-- Statement of the problem in Lean 4
theorem decode_rebus (hA : is_digit A) 
                     (hP : is_digit P) 
                     (hM : is_digit M) 
                     (hИ : is_digit И) 
                     (hР : is_digit Р) 
                     (uniq : unique_digits [A, P, M, И, Р]) 
                     (hAP_sq : (10 * A + P)^2 = 100 * М + 10 * И + Р) :
  10 * A + P = 16 ∧ М = 2 ∧ 100 * М + 10 * И + Р = 256 :=
begin
  sorry,
end

end decode_rebus_l391_391992


namespace sequence_properties_l391_391625

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 2 * a n + n - 4

def a_n (n : ℕ) : ℝ :=
  if h : n > 0 then
    let a : ℕ → ℝ := λ k, if k = 1 then 3 else 2 * a (k - 1) - 1 in
    a n
  else
    0

def T (n : ℕ) : ℝ :=
  ∑ i in range n, (3 / a_n i)

theorem sequence_properties :
  (∀ n > 0, a_n n = 1 + 2^n) ∧ (∀ n > 0, 1 ≤ T n ∧ T n < 5 / 2) :=
by
  sorry

end sequence_properties_l391_391625


namespace greatest_prime_factor_15_fact_17_fact_l391_391222

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391222


namespace Petya_friends_l391_391083

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l391_391083


namespace greatest_prime_factor_of_sum_factorials_l391_391276

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l391_391276


namespace area_of_triangle_MOF_l391_391621

-- Define the given conditions and prove the required statement
theorem area_of_triangle_MOF
    (parabola : ∀ (x y : ℝ), y^2 = 4 * x)
    (vertex_origin : vertex = (0, 0))
    (focus : F = (1, 0))
    (point_M : M = (2, y_0))
    (dist_M_to_F : dist M F = 3) :
    area_of_triangle MOF = √2 := 
sorry

end area_of_triangle_MOF_l391_391621


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391359

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391359


namespace probability_not_win_l391_391923

theorem probability_not_win (A B : Fin 16) : 
  (256 - 16) / 256 = 15 / 16 := 
by
  sorry

end probability_not_win_l391_391923


namespace chest_contents_l391_391777

def Chest : Type := ℕ
def Coins : Type := ℕ

variable (empty : Coins → Coins → Prop)

noncomputable def Chest1 : Chest := 1
noncomputable def Chest2 : Chest := 2
noncomputable def Chest3 : Chest := 3

variable (goldCoins : Coins)
variable (silverCoins : Coins)
variable (copperCoins : Coins)

-- Conditions
variable (labelsIncorrect : Chest → Coins → Prop)

axiom label1 : labelsIncorrect Chest1 goldCoins
axiom label2 : labelsIncorrect Chest2 silverCoins
axiom label3 : labelsIncorrect Chest3 (goldCoins ∨ silverCoins)

axiom uniqueContents : ∀ c : Chest, c = Chest1 ∨ c = Chest2 ∨ c = Chest3
axiom distinctContents : ∀ c1 c2 : Chest, c1 ≠ c2 → (c1 = Chest1 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest3) ∨ (c1 = Chest2 ∧ c2 ≠ Chest2 ∧ c2 ≠ Chest3) ∨ (c1 = Chest3 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest2)

theorem chest_contents : (exists! c1 : Coins, c1 = silverCoins ∧ labelsIncorrect Chest1 c1) ∧
                         (exists! c2 : Coins, c2 = goldCoins ∧ labelsIncorrect Chest2 c2) ∧
                         (exists! c3 : Coins, c3 = copperCoins ∧ labelsIncorrect Chest3 c3)
                         :=
by
  sorry

end chest_contents_l391_391777


namespace simplify_expression_l391_391105

theorem simplify_expression : (Real.sin (15 * Real.pi / 180) + Real.sin (45 * Real.pi / 180)) / (Real.cos (15 * Real.pi / 180) + Real.cos (45 * Real.pi / 180)) = Real.tan (30 * Real.pi / 180) :=
by
  sorry

end simplify_expression_l391_391105


namespace fraction_conditions_l391_391913

theorem fraction_conditions {x : ℝ} : (x ≠ -3) ↔ ((x + 1 = 0) ∧ (2x - 3 ≠ 0)) :=
by
  sorry

end fraction_conditions_l391_391913


namespace petya_has_19_friends_l391_391052

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l391_391052


namespace right_triangle_area_perimeter_ratio_l391_391458

theorem right_triangle_area_perimeter_ratio :
  let a := 4
  let b := 8
  let area := (1/2) * a * b
  let c := Real.sqrt (a^2 + b^2)
  let perimeter := a + b + c
  let ratio := area / perimeter
  ratio = 3 - Real.sqrt 5 :=
by
  sorry

end right_triangle_area_perimeter_ratio_l391_391458


namespace greatest_prime_factor_of_15_l391_391447

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391447


namespace replaces_T_l391_391585

open Nat

noncomputable def numbers := {1, 2, 3, 4, 5, 6}

theorem replaces_T (P Q R S T U : ℕ) 
  (h1 : P ∈ numbers) 
  (h2 : Q ∈ numbers)
  (h3 : R ∈ numbers)
  (h4 : S ∈ numbers)
  (h5 : T ∈ numbers)
  (h6 : U ∈ numbers)
  (h_unique : ∀ x ∈ numbers, ∃! y, (P = x ∨ Q = x ∨ R = x ∨ S = x ∨ T = x ∨ U = x) ∧ y ∈ numbers)
  (h7 : P + Q = 5)
  (h8 : | R - S | = 5)
  (h9 : T > U) : 
  T = 5 := 
sorry

end replaces_T_l391_391585


namespace greatest_prime_factor_of_factorial_sum_l391_391337

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391337


namespace pyramid_volume_l391_391985

noncomputable def determinant (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  (1 / 2: ℚ) * | (x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1) |

theorem pyramid_volume :
  let A := (0 : ℚ, 0 : ℚ),
      B := (30 : ℚ, 0 : ℚ),
      C := (14 : ℚ, 22 : ℚ),
      D := ((B.1 + C.1) / 2, (B.2 + C.2) / 2),
      E := ((C.1 + A.1) / 2, (C.2 + A.2) / 2),
      F := ((A.1 + B.1) / 2, (A.2 + B.2) / 2),
      O := (14 : ℚ, 112 / 11 : ℚ),
      DEF_area := determinant D.1 D.2 E.1 E.2 F.1 F.2,
      height := O.2
  in (1 / 3) * DEF_area * height = 298.666666666667 := by
  sorry

end pyramid_volume_l391_391985


namespace greatest_prime_factor_15_fact_17_fact_l391_391174

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391174


namespace trapezoid_diagonal_length_l391_391898

/-- Trapezoid $EFGH$ has parallel sides $\overline{EF}$ of length 28 and $\overline{GH}$ of length 18.
The other two sides are of lengths 15 and 12. Prove that the length of the shorter diagonal
of $EFGH$ is $\sqrt{\frac{69639}{400}}$. --/
theorem trapezoid_diagonal_length : 
  ∀ (EF GH side1 side2 : ℝ), 
  EF = 28 ∧ GH = 18 ∧ side1 = 15 ∧ side2 = 12 → 
  ∃ (diagonal : ℝ), diagonal = Real.sqrt (69639 / 400) ∧ 
  (diagonal = (Real.sqrt (((181 / 20) - 10) ^ 2 + (69639 / 400))) ∨ 
   diagonal = (Real.sqrt (((181 / 20) + 18) ^ 2 + (69639 / 400))) :=
by
  intros EF GH side1 side2 h
  have h1 : EF = 28 := h.1 
  have h2 : GH = 18 := h.2.1 
  have h3 : side1 = 15 := h.2.2.1
  have h4 : side2 = 12 := h.2.2.2
  have diagonal_value : ∃ d, d = Real.sqrt ((69639) / (400)) :=
    exists.intro (Real.sqrt (69639 / 400)) rfl  
  exact diagonal_value

end trapezoid_diagonal_length_l391_391898


namespace order_of_magnitude_l391_391671

noncomputable def a : ℝ := (3 / 4) ^ (-1 / 3)
noncomputable def b : ℝ := (3 / 4) ^ (-1 / 4)
noncomputable def c : ℝ := (3 / 2) ^ (-1 / 4)

theorem order_of_magnitude : a > b ∧ b > c := 
by
  sorry

end order_of_magnitude_l391_391671


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391290

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391290


namespace exist_disjoint_translations_l391_391009

/-- Given a set A which is a subset of {1, 2, ..., 1000000} with exactly 101 elements,
prove that there exist numbers t1, t2, ..., t100 such that the sets
Aj = { x + tj | x ∈ A}, for j = 1, 2, ..., 100 are pairwise disjoint. -/
theorem exist_disjoint_translations :
  ∀ (A : Finset ℕ), 
  (A ⊆ Finset.range 1000001) → A.card = 101 → 
  ∃ (t : Fin 100 → ℕ), 
  ∀ (i j : Fin 100), i ≠ j → 
  (Finset.image (λ x, x + t i) A ∩ Finset.image (λ x, x + t j) A) = ∅ := 
sorry

end exist_disjoint_translations_l391_391009


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391438

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391438


namespace greatest_prime_factor_of_sum_l391_391401

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391401


namespace randy_walks_his_dog_l391_391839

theorem randy_walks_his_dog :
  ∀ (pack_count packs_per_day total_days : ℕ),
  packs_per_day * total_days = pack_count * 120 →
  pack_count = 6 →
  total_days = 360 →
  packs_per_day / total_days = 2 :=
by
  intros pack_count packs_per_day total_days
  assume h1 h2 h3
  rw [← h3] at h1
  rw [← h2] at h1
  calc packs_per_day / 360 = (6 * 120) / 360 : by rw [h1]
                     ... = 720 / 360        : by norm_num
                     ... = 2                : by norm_num
  sorry

end randy_walks_his_dog_l391_391839


namespace sum_positive_l391_391619

variable (a : Fin 1993 → ℝ)

def sum_of_all_numbers_positive (a : Fin 1993 → ℝ) :=
  (∀ i : Fin 1993, a i + a ((i + 1) % 1993) + a ((i + 2) % 1993) + a ((i + 3) % 1993) > 0) →
  (∑ i : Fin 1993, a i) > 0

theorem sum_positive : sum_of_all_numbers_positive a :=
sorry

end sum_positive_l391_391619


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391308
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391308


namespace chest_contents_correct_l391_391760

-- Define the chests
inductive Chest
| chest1
| chest2
| chest3

open Chest

-- Define the contents
inductive Contents
| gold
| silver
| copper

open Contents

-- Define the labels, all of which are incorrect
def label (c : Chest) : Prop :=
  match c with
  | chest1 => gold
  | chest2 => silver
  | chest3 => gold ∨ silver

-- Assuming the labels are all incorrect
axiom label_incorrect : ∀ (c : Chest), ¬(label c = true)

-- One chest contains each type of coin
axiom one_gold : ∃ c, ∀ x, x ≠ c → Contents x ≠ gold
axiom one_silver : ∃ c, ∀ x, x ≠ c → Contents x ≠ silver
axiom one_copper : ∃ c, ∀ x, x ≠ c → Contents x ≠ copper

-- Determine the contents of each chest
def chest1_contents := silver
def chest2_contents := gold
def chest3_contents := copper

-- Prove the correspondence
theorem chest_contents_correct :
  (chest1_contents = silver) ∧ 
  (chest2_contents = gold) ∧ 
  (chest3_contents = copper) :=
by
  split
  · exact sorry
  split
  · exact sorry
  · exact sorry

end chest_contents_correct_l391_391760


namespace greatest_prime_factor_15_fact_17_fact_l391_391184

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391184


namespace parabola_line_intersect_product_l391_391987

theorem parabola_line_intersect_product (k : ℝ) :
  let F := (0, 1 / 8) in
  let parabola := (λ x, 2 * x^2) in
  let line := (λ x, k * x + 1 / 8) in
  let intersection_pts := 
    {x | 2 * x^2 = k * x + 1 / 8} in
  ∃ x₁ x₂, x₁ ∈ intersection_pts ∧ x₂ ∈ intersection_pts ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -1 / 16 := sorry

end parabola_line_intersect_product_l391_391987


namespace petya_friends_l391_391074

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l391_391074


namespace positive_difference_1010_1000_l391_391584

-- Define the arithmetic sequence
def arithmetic_sequence (a d n : ℕ) : ℕ :=
  a + (n - 1) * d

-- Define the specific terms
def a_1000 := arithmetic_sequence 5 7 1000
def a_1010 := arithmetic_sequence 5 7 1010

-- Proof statement
theorem positive_difference_1010_1000 : a_1010 - a_1000 = 70 :=
by
  sorry

end positive_difference_1010_1000_l391_391584


namespace pebbles_count_l391_391820

theorem pebbles_count (initial_pebbles : ℕ) (pebbles_skipped_fraction : ℕ) (additional_pebbles : ℕ) :
  initial_pebbles = 18 →
  pebbles_skipped_fraction = 2 →
  additional_pebbles = 30 →
  (initial_pebbles - (initial_pebbles / pebbles_skipped_fraction) + additional_pebbles) = 39 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end pebbles_count_l391_391820


namespace rectangle_y_value_l391_391496

theorem rectangle_y_value (y : ℝ) (h₁ : (-2, y) ≠ (10, y))
  (h₂ : (-2, -1) ≠ (10, -1))
  (h₃ : 12 * (y + 1) = 108)
  (y_pos : 0 < y) :
  y = 8 :=
by
  sorry

end rectangle_y_value_l391_391496


namespace chests_contents_l391_391734

def ChestLabel (c : ℕ) : String := 
  if c = 1 then "Gold coins"
  else if c = 2 then "Silver coins"
  else if c = 3 then "Gold or silver coins"
  else "Invalid chest"

def CoinsInChest (c : ℕ) : String := 
  if c = 1 then "Silver coins"
  else if c = 2 then "Gold coins"
  else if c = 3 then "Copper coins"
  else "Invalid chest"

theorem chests_contents :
  ChestLabel 1 ≠ CoinsInChest 1 ∧
  ChestLabel 2 ≠ CoinsInChest 2 ∧
  ChestLabel 3 ≠ CoinsInChest 3 ∧
  (CoinsInChest 1 = "Gold coins" ∨ CoinsInChest 2 = "Gold coins" ∨ CoinsInChest 3 = "Gold coins") ∧
  (CoinsInChest 1 = "Silver coins" ∨ CoinsInChest 2 = "Silver coins" ∨ CoinsInChest 3 = "Silver coins") ∧
  (CoinsInChest 1 = "Copper coins" ∨ CoinsInChest 2 = "Copper coins" ∨ CoinsInChest 3 = "Copper coins") :=
begin
  sorry
end

end chests_contents_l391_391734


namespace hyperbola_focus_l391_391579

theorem hyperbola_focus :
  let a^2 := 29 / 3
  let b^2 := 29 / 4
  let h := -1
  let k := -3
  let c := (Real.sqrt (a^2 + b^2)) / (2 * Real.sqrt 3)
  ∃ f : ℝ × ℝ, f = (h, k + c) ∧ (3 * (f.1)^2 - 4 * (f.2)^2 + 6 * (f.1) - 24 * (f.2) - 8 = 0)
:= sorry

end hyperbola_focus_l391_391579


namespace incorrect_weight_estimation_l391_391976

variables (x y : ℝ)

/-- Conditions -/
def regression_equation (x : ℝ) : ℝ := 0.85 * x - 85.71

/-- Incorrect conclusion -/
theorem incorrect_weight_estimation : regression_equation 160 ≠ 50.29 :=
by 
  sorry

end incorrect_weight_estimation_l391_391976


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391259

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391259


namespace greatest_prime_factor_of_15_l391_391452

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391452


namespace pebbles_calculation_l391_391819

theorem pebbles_calculation (initial_pebbles : ℕ) (half_skipped : ℕ) (pebbles_given : ℕ) : 
  initial_pebbles = 18 → 
  half_skipped = initial_pebbles / 2 → 
  pebbles_given = 30 → 
  (half_skipped + pebbles_given) = 39 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end pebbles_calculation_l391_391819


namespace greatest_prime_factor_15f_plus_17f_l391_391201

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391201


namespace seating_arrangements_l391_391514

def alice_bob_condition (s : List String) : Prop :=
  let i_a := s.indexOf "Alice"
  let i_b := s.indexOf "Bob"
  (i_a ≠ i_b + 1) ∧ (i_a ≠ i_b - 1)

def derek_condition (s : List String) : Prop :=
  let i_a := s.indexOf "Alice"
  let i_d := s.indexOf "Derek"
  let i_e := s.indexOf "Eric"
  (i_d ≠ i_a + 1) ∧ (i_d ≠ i_a - 1) ∧ (i_d ≠ i_e + 1) ∧ (i_d ≠ i_e - 1)

def valid_seating (s : List String) : Prop :=
  alice_bob_condition s ∧ derek_condition s

def all_permutations := List.permutations ["Alice", "Bob", "Carla", "Derek", "Eric"]

def count_valid_permutations : Nat := (all_permutations.filter valid_seating).length

theorem seating_arrangements :
  count_valid_permutations = 20 := by
  sorry

end seating_arrangements_l391_391514


namespace geometry_inequality_l391_391723

theorem geometry_inequality
  (A B C D E : Point)
  (hABCEquilateral : equilateral_triangle A B C)
  (hDOnBC : collinear B C D)
  (hCEParallelAD : parallel (line_through C E) (line_through A D))
  (hEOnAB : collinear A B E) :
  CE / CD ≥ 2 * sqrt(3) := 
sorry

end geometry_inequality_l391_391723


namespace chest_contents_l391_391776

def Chest : Type := ℕ
def Coins : Type := ℕ

variable (empty : Coins → Coins → Prop)

noncomputable def Chest1 : Chest := 1
noncomputable def Chest2 : Chest := 2
noncomputable def Chest3 : Chest := 3

variable (goldCoins : Coins)
variable (silverCoins : Coins)
variable (copperCoins : Coins)

-- Conditions
variable (labelsIncorrect : Chest → Coins → Prop)

axiom label1 : labelsIncorrect Chest1 goldCoins
axiom label2 : labelsIncorrect Chest2 silverCoins
axiom label3 : labelsIncorrect Chest3 (goldCoins ∨ silverCoins)

axiom uniqueContents : ∀ c : Chest, c = Chest1 ∨ c = Chest2 ∨ c = Chest3
axiom distinctContents : ∀ c1 c2 : Chest, c1 ≠ c2 → (c1 = Chest1 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest3) ∨ (c1 = Chest2 ∧ c2 ≠ Chest2 ∧ c2 ≠ Chest3) ∨ (c1 = Chest3 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest2)

theorem chest_contents : (exists! c1 : Coins, c1 = silverCoins ∧ labelsIncorrect Chest1 c1) ∧
                         (exists! c2 : Coins, c2 = goldCoins ∧ labelsIncorrect Chest2 c2) ∧
                         (exists! c3 : Coins, c3 = copperCoins ∧ labelsIncorrect Chest3 c3)
                         :=
by
  sorry

end chest_contents_l391_391776


namespace distance_from_focus2_l391_391832

-- Define the properties and given conditions
def ellipse (x y : ℝ) : Prop := (x^2) / 36 + (y^2) / 9 = 1
def semiMajorAxis : ℝ := 6
def sum_of_distances_to_foci (P : ℝ × ℝ) : ℝ := 12

-- Define the condition that point P is at a distance of 5 from one focus
def distance_from_focus1 (P focus1 : ℝ × ℝ) : ℝ := 5

-- Hypothesize the position of focus1 and focus2
variables (focus1 focus2 : ℝ × ℝ)

-- Main statement to be proved
theorem distance_from_focus2 (P : ℝ × ℝ) (h_ellipse : ellipse P.1 P.2) 
  (h_dist1 : distance_from_focus1 P focus1) 
  (h_sum : sum_of_distances_to_foci P) : 
  ∃ d : ℝ, d = 7 :=
by
  sorry

end distance_from_focus2_l391_391832


namespace integer_part_of_product_l391_391133

theorem integer_part_of_product (x y : ℝ) (hx : 0 ≤ x) (hx1 : x < 1) (hy : 0 ≤ y) (hy1 : y < 1) :
  let a := 7 + x
  let b := 10 + y
  let smallest := int.floor (a * b)
  let largest := int.floor ((7 + 1 - 1e-9) * (10 + 1 - 1e-9))
  -- the follows equates to "number of possible integer values" is 18.
  (largest - smallest + 1) = 18 := sorry

end integer_part_of_product_l391_391133


namespace greatest_prime_factor_15_fact_17_fact_l391_391231

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391231


namespace greatest_prime_factor_of_factorial_sum_l391_391336

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391336


namespace pebbles_count_l391_391821

theorem pebbles_count (initial_pebbles : ℕ) (pebbles_skipped_fraction : ℕ) (additional_pebbles : ℕ) :
  initial_pebbles = 18 →
  pebbles_skipped_fraction = 2 →
  additional_pebbles = 30 →
  (initial_pebbles - (initial_pebbles / pebbles_skipped_fraction) + additional_pebbles) = 39 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end pebbles_count_l391_391821


namespace greatest_prime_factor_15_17_factorial_l391_391344

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391344


namespace grandview_high_lockers_l391_391146

noncomputable def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

def digit_cost (n : ℕ) : ℚ :=
  if n <= 1500 then 0.02 * num_digits n
  else 0.03 * num_digits n

-- We define the total cost function
noncomputable def total_cost (N : ℕ) : ℚ :=
  ∑ i in finset.range (N + 1), digit_cost i

theorem grandview_high_lockers :
  ∃ n, total_cost n = 278.94 ∧ n = 3009 := 
by
  sorry

end grandview_high_lockers_l391_391146


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391289

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391289


namespace smallest_largest_same_parity_l391_391566

-- Define the set A_P
def A_P : Set ℕ := sorry

-- Define the maximum element c in A_P
def c := max A_P

-- Assuming shifting doesn't change fundamental counts
lemma shifting_preserves_counts (A : Set ℕ) (c : ℕ) : 
  (∀ x ∈ A, x ≠ c → x < c → exists y ∈ A, y < x) ∧
  (∃ p ∈ A, ∃ q ∈ A, p ≠ q ∧ p < q) :=
  sorry

-- Define parity
def parity (n : ℕ) := n % 2

-- Theorem statement
theorem smallest_largest_same_parity (A : Set ℕ) (a_max a_min : ℕ) 
  (h_max : a_max = max A) (h_min : a_min = min A)
  (h_shift : ∀ x ∈ A, shifting_preserves_counts A x) :
  parity a_max = parity a_min :=
sorry

end smallest_largest_same_parity_l391_391566


namespace greatest_prime_factor_of_factorial_sum_l391_391332

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391332


namespace range_of_p_l391_391693

theorem range_of_p (x : ℝ) (hx : 0 < x ∧ x < π / 2) (p : ℝ) (hp : 0 < p) :
  (∀ x, (x ∈ Ioo 0 (π / 2)) → (1 / (Real.sin x)^2 + p / (Real.cos x)^2) ≥ 9) ↔ 4 ≤ p := 
sorry

end range_of_p_l391_391693


namespace line_plane_intersection_l391_391595

theorem line_plane_intersection :
  let x := 2
  let y := 3
  let z := -4 in
  (∃ (t : ℝ), (x = 1 + t ∧ y = 3 ∧ z = -2 - 2 * t)) ∧ (3 * x - 7 * y - 2 * z + 7 = 0) :=
by
  let x := 2
  let y := 3
  let z := -4
  sorry

end line_plane_intersection_l391_391595


namespace infinite_sum_areas_l391_391580

-- Define the ellipse with semi-major axis a and semi-minor axis b
variables (a b : ℝ) (h : a > b)

-- Sum of the areas of the infinitely many rectangles and ellipses created by the process
theorem infinite_sum_areas (a b : ℝ) (h : a > b) :
  let initial_ellipse_area := π * a * b in
  let initial_rectangle_area := 8 * a * b in
  let series_sum := 2 * initial_ellipse_area + 4 * a * b in
  series_sum = 2 * π * a * b + 4 * a * b :=
by
  -- Placeholder for proof
  sorry

end infinite_sum_areas_l391_391580


namespace constant_term_polynomial_l391_391687

theorem constant_term_polynomial (m : ℝ) (h1 : m > 1) (h2 : ∫ x in 1..m, (2 * x - 1) = 6) :
  constant_term_expansion (x^2 + (1/x^2) - 2) m = -20 := 
by
  sorry

end constant_term_polynomial_l391_391687


namespace chests_content_l391_391766

theorem chests_content (A B C : Type) (chest1 chest2 chest3 : A)
  (label1 : chest1 = "gold coins") (label2 : chest2 = "silver coins") (label3 : chest3 = "gold or silver coins")
  (gold silver copper : A)
  (h1 : chest1 ≠ gold)
  (h2 : chest2 ≠ silver)
  (h3 : chest3 ≠ gold ∧ chest3 ≠ silver)
  (t1 : chest1 = silver)
  (t2 : chest2 = gold)
  (t3 : chest3 = copper) :
  True := sorry

end chests_content_l391_391766


namespace measure_of_angle_Q_eq_72_l391_391841

noncomputable def degree_measure_of_angle_Q (A B C D E F G H I J Q : Type) 
  (h_regular_decagon : regular_polygon 10 {A, B, C, D, E, F, G, H, I, J}) 
  (h_extends : extends ⟨A, J⟩ ⟨E, F⟩ Q): 
  ℝ := 
  72

theorem measure_of_angle_Q_eq_72 
  (A B C D E F G H I J Q : Type)
  (h_regular_decagon : regular_polygon 10 {A, B, C, D, E, F, G, H, I, J})
  (h_extends : extends ⟨A, J⟩ ⟨E, F⟩ Q):
  degree_measure_of_angle_Q A B C D E F G H I J Q h_regular_decagon h_extends = 72 :=
sorry

end measure_of_angle_Q_eq_72_l391_391841


namespace greatest_prime_factor_of_sum_l391_391403

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391403


namespace petya_has_19_friends_l391_391049

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l391_391049


namespace perfect_fourth_powers_count_l391_391684

theorem perfect_fourth_powers_count : 
    let n := 2 in            -- smallest integer such that n^4 >= 10
    let m := 10 in           -- largest integer such that m^4 <= 10000
    (10 - 2 + 1) = 9        -- number of integers k such that 10 <= k^4 <= 10000
:= by {
    have h1 : n = 2, sorry,  -- small_step
    have h2 : m = 10, sorry  -- large_step
    exact 9                 -- 
}

end perfect_fourth_powers_count_l391_391684


namespace greatest_prime_factor_of_15_l391_391456

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391456


namespace greatest_prime_factor_15_17_factorial_l391_391348

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391348


namespace max_value_when_m_is_neg_one_extremal_points_range_m_l391_391022

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := log x + m * (x^2 - x)

-- Problem (I)
theorem max_value_when_m_is_neg_one :
  ∀ x ∈ set.Ioi (0 : ℝ), is_local_max_on (f x (-1)) (set.Ioi (0 : ℝ)) 1 ∧ ¬is_local_min_on (f x (-1)) (set.Ioi (0 : ℝ)) 1 :=
by sorry

-- Problem (II)
theorem extremal_points_range_m :
  ∀ m : ℝ, (∃ x ∈ set.Ioi (0 : ℝ), is_local_min_on (f x m) (set.Ioi (0 : ℝ)) x ∨ is_local_max_on (f x m) (set.Ioi (0 : ℝ)) x) ↔ (m < 0 ∨ m > 8) :=
by sorry

end max_value_when_m_is_neg_one_extremal_points_range_m_l391_391022


namespace find_b_for_parallel_lines_l391_391463

theorem find_b_for_parallel_lines :
  (∀ (b : ℝ), (∃ (f g : ℝ → ℝ),
  (∀ x, f x = 3 * x + b) ∧
  (∀ x, g x = (b + 9) * x - 2) ∧
  (∀ x, f x = g x → False)) →
  b = -6) :=
sorry

end find_b_for_parallel_lines_l391_391463


namespace isosceles_trapezoid_incircle_sum_l391_391973

theorem isosceles_trapezoid_incircle_sum 
    (ABCD : Trapezoid)
    (h_isosceles : ABCD.isIsoscelesTrapezoid)
    (N M K L : Point)
    (hN_midpoint : N.isMidpoint ABCD.A D)
    (hM_touch : ABCD.incircle.touches CD M)
    (hK_inter : ABCD.incircle.intersects AM K)
    (hL_inter : ABCD.incircle.intersects BM L)
    (a : ℝ)
    (h_AN_eq_ND : dist ABCD.A N = a)
    (h_AN_eq_DM : dist ABCD.A N = dist ABCD.D M) 
    (α : ℝ)
    (AM AK BM BL : ℝ) :
    AM = √(a^2 * (5 - 4 * cos α)) → 
    AK * AM = a^2 →
    BM = AM →
    BL = AK →
    (AK * BL) = a^2 →
    AM / AK + BM / BL = 10 :=
by
  sorry

end isosceles_trapezoid_incircle_sum_l391_391973


namespace problem_statement_l391_391665

noncomputable def a : ℕ → ℝ
| 1       := 1
| (n + 1) := (1/4)^n - a n

noncomputable def S : ℕ → ℝ
| 0       := 0
| (n + 1) := S n + 4^n * a (n + 1)

theorem problem_statement (n : ℕ) : 5 * S n - 4^n * a n = n :=
by
  sorry

end problem_statement_l391_391665


namespace greatest_prime_factor_of_sum_l391_391395

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391395


namespace chest_contents_correct_l391_391758

-- Define the chests
inductive Chest
| chest1
| chest2
| chest3

open Chest

-- Define the contents
inductive Contents
| gold
| silver
| copper

open Contents

-- Define the labels, all of which are incorrect
def label (c : Chest) : Prop :=
  match c with
  | chest1 => gold
  | chest2 => silver
  | chest3 => gold ∨ silver

-- Assuming the labels are all incorrect
axiom label_incorrect : ∀ (c : Chest), ¬(label c = true)

-- One chest contains each type of coin
axiom one_gold : ∃ c, ∀ x, x ≠ c → Contents x ≠ gold
axiom one_silver : ∃ c, ∀ x, x ≠ c → Contents x ≠ silver
axiom one_copper : ∃ c, ∀ x, x ≠ c → Contents x ≠ copper

-- Determine the contents of each chest
def chest1_contents := silver
def chest2_contents := gold
def chest3_contents := copper

-- Prove the correspondence
theorem chest_contents_correct :
  (chest1_contents = silver) ∧ 
  (chest2_contents = gold) ∧ 
  (chest3_contents = copper) :=
by
  split
  · exact sorry
  split
  · exact sorry
  · exact sorry

end chest_contents_correct_l391_391758


namespace max_area_angle_A_l391_391722

open Real

theorem max_area_angle_A (A B C : ℝ) (tan_A tan_B : ℝ) :
  tan A * tan B = 1 ∧ AB = sqrt 3 → 
  (∃ A, A = π / 4 ∧ area_maximized)
  :=
by sorry

end max_area_angle_A_l391_391722


namespace soccer_players_positions_l391_391536

theorem soccer_players_positions :
  ∃ (a b c d : ℝ), a = 0 ∧ b = 1 ∧ c = 4 ∧ d = 6 ∧
  set_of (λ x, ∃ i j, i ≠ j ∧ x = abs (a - b) ∨ x = abs (a - c) ∨ x = abs (a - d) ∨ x = abs (b - c) ∨ x = abs (b - d) ∨ x = abs (c - d)) = {1, 2, 3, 4, 5, 6} :=
by
  use 0, 1, 4, 6
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  rw [set_of, abs]
  sorry

end soccer_players_positions_l391_391536


namespace greatest_prime_factor_of_15_l391_391455

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l391_391455


namespace probability_no_wizard_picks_own_wand_l391_391800

theorem probability_no_wizard_picks_own_wand :
  let total_permutations := 3!
  let num_derangements := 2
  (num_derangements / total_permutations =  1 / 3) :=
by
  sorry

end probability_no_wizard_picks_own_wand_l391_391800


namespace smallest_largest_same_parity_l391_391564

-- Define the context where our elements and set are considered
variable {α : Type*} [LinearOrderedCommRing α] (A_P : Set α)

-- Assume a nonempty set A_P and define the smallest and largest elements in Lean
noncomputable def smallest_element (A_P : Set α) : α := Inf' A_P sorry
noncomputable def largest_element (A_P : Set α) : α := Sup' A_P sorry

-- State the proof goal
theorem smallest_largest_same_parity (h_nonempty : A_P.nonempty) : 
  (smallest_element A_P) % 2 = (largest_element A_P) % 2 := 
by 
  sorry

end smallest_largest_same_parity_l391_391564


namespace alpha_beta_sum_l391_391814

theorem alpha_beta_sum (α β : ℝ) 
  (hα : α^3 - 3 * α^2 + 5 * α - 4 = 0) 
  (hβ : β^3 - 3 * β^2 + 5 * β - 2 = 0) : 
  α + β = 2 := 
begin
  sorry,
end

end alpha_beta_sum_l391_391814


namespace right_triangle_hypotenuse_length_l391_391958

theorem right_triangle_hypotenuse_length (a b : ℝ) (c : ℝ) (h₁ : a = 10) (h₂ : b = 24) (h₃ : c^2 = a^2 + b^2) : c = 26 :=
by
  -- sorry is used to skip the actual proof
  sorry

end right_triangle_hypotenuse_length_l391_391958


namespace kim_fourth_exam_score_l391_391733

theorem kim_fourth_exam_score :
  ∃ x : ℝ, (87 + 83 + 88 + x) / 4 = 89 ∧ x = 98 :=
by {
  -- Introduce the conditions
  let s1 := 87,
  let s2 := 83,
  let s3 := 88,
  let target_avg := 89,

  -- Express the fourth score needed
  use 98,

  -- Prove the condition of average being equal to 89
  have : (s1 + s2 + s3 + 98) / 4 = target_avg,
  { calc
      (s1 + s2 + s3 + 98) / 4 
        = (87 + 83 + 88 + 98) / 4 : by rw [s1, s2, s3]
    ... = 356 / 4 : by norm_num
    ... = 89 : by norm_num },
  
  -- Conclude with a proof term
  exact ⟨this, rfl⟩,
}

end kim_fourth_exam_score_l391_391733


namespace optimal_play_winner_l391_391904

-- Definitions for the conditions
def chessboard_size (K N : ℕ) : Prop := True
def rook_initial_position (K N : ℕ) : (ℕ × ℕ) :=
  (K, N)
def move (r : ℕ × ℕ) (direction : ℕ) : (ℕ × ℕ) :=
  if direction = 0 then (r.1 - 1, r.2)
  else (r.1, r.2 - 1)
def rook_cannot_move (r : ℕ × ℕ) : Prop :=
  r.1 = 0 ∨ r.2 = 0

-- Theorem to prove the winner given the conditions
theorem optimal_play_winner (K N : ℕ) :
  (K = N → ∃ player : ℕ, player = 2) ∧ (K ≠ N → ∃ player : ℕ, player = 1) :=
by
  sorry

end optimal_play_winner_l391_391904


namespace BoatsRUs_canoes_l391_391545

theorem BoatsRUs_canoes :
  let a := 6
  let r := 3
  let n := 5
  let S := a * (r^n - 1) / (r - 1)
  S = 726 := by
  -- Proof
  sorry

end BoatsRUs_canoes_l391_391545


namespace maximum_value_of_complex_expr_l391_391622

open Complex

theorem maximum_value_of_complex_expr (n : ℕ) (z : ℕ → ℂ) (h : ∀ k, ‖z k‖ ≤ 1) :
  ∃ max_val, max_val = 2 * ⌊(n:ℝ) / 2⌋₊ ∧
  ∀ (sum_zk_squared : ℂ) (sum_zk : ℂ),
    sum_zk_squared = ∑ k in Finset.range n, (z k) ^ 2 →
    sum_zk = ∑ k in Finset.range n, z k →
    (abs sum_zk_squared - (abs sum_zk) ^ 2) ≤ max_val := 
begin
  sorry,
end

end maximum_value_of_complex_expr_l391_391622


namespace petya_friends_l391_391095

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l391_391095


namespace find_t_l391_391158

section TriangleArea
  open Set

  def point := ℝ × ℝ

  def A : point := (0, 10)
  def B : point := (4, 0)
  def C : point := (10, 0)

  def is_on_line_at_y (p : point) (t : ℝ) : Prop := p.2 = t

  def line_eq_AB (x : ℝ) : ℝ := -2.5 * x + 10
  def line_eq_AC (x : ℝ) : ℝ := -x + 10
  
  def T (t : ℝ) : point := 
    let x := (10 - t) / 2.5 in (x, t)
  def U (t : ℝ) : point := 
    let x := 10 - t in (x, t)

  def length_TU (t : ℝ) : ℝ := 
    abs (10 - t - ((10 - t) / 2.5))

  def area_ATU (t : ℝ) : ℝ := 
    0.5 * (length_TU t) * (10 - t)

  theorem find_t (t : ℝ) (h : area_ATU t = 15) : t = 5 * real.sqrt 2 :=
  sorry
end TriangleArea

end find_t_l391_391158


namespace Petya_friends_l391_391085

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l391_391085


namespace greatest_prime_factor_15f_plus_17f_l391_391198

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391198


namespace f_properties_l391_391109

variable (f : ℝ → ℝ)
variable (f_pos : ∀ x : ℝ, f x > 0)
variable (f_eq : ∀ a b : ℝ, f a * f b = f (a + b))

theorem f_properties :
  (f 0 = 1) ∧
  (∀ a : ℝ, f (-a) = 1 / f a) ∧
  (∀ a : ℝ, f a = (f (3 * a))^(1/3)) :=
by {
  sorry
}

end f_properties_l391_391109


namespace soccer_players_positions_l391_391537

theorem soccer_players_positions :
  ∃ (a b c d : ℝ), a = 0 ∧ b = 1 ∧ c = 4 ∧ d = 6 ∧
  set_of (λ x, ∃ i j, i ≠ j ∧ x = abs (a - b) ∨ x = abs (a - c) ∨ x = abs (a - d) ∨ x = abs (b - c) ∨ x = abs (b - d) ∨ x = abs (c - d)) = {1, 2, 3, 4, 5, 6} :=
by
  use 0, 1, 4, 6
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  rw [set_of, abs]
  sorry

end soccer_players_positions_l391_391537


namespace triangle_area_is_18_l391_391908

def point := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  (1 / 2) * abs ((fst A - fst C) * (snd B - snd A) - (fst A - fst B) * (snd C - snd A))

def A := (4, -1) : point
def B := (10, 3) : point
def C := (4, 5) : point

theorem triangle_area_is_18 : triangle_area A B C = 18 :=
  by
    sorry

end triangle_area_is_18_l391_391908


namespace greatest_prime_factor_15_17_factorial_l391_391207

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l391_391207


namespace victor_maximum_marks_l391_391905

theorem victor_maximum_marks :
  (∃ M : ℝ, 0.25 * M * 0.92 + 0.20 * M * 0.88 + 0.25 * M * 0.90 + 0.15 * M * 0.87 + 0.15 * M * 0.85 = 405) :=
begin
  use 456,
  sorry
end

end victor_maximum_marks_l391_391905


namespace sin_double_angle_eq_one_fourth_l391_391639

theorem sin_double_angle_eq_one_fourth (α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : Real.sin(π / 2 + 2 * α) = Real.cos(π / 4 - α)) : Real.sin(2 * α) = 1 / 4 :=
by
  sorry

end sin_double_angle_eq_one_fourth_l391_391639


namespace interquartile_range_theorem_l391_391107

variable (submissions : List ℝ)
variable (H_sorted : submissions.sorted (· ≤ ·))

def Q1 (l : List ℝ) : ℝ := (l[(l.length / 2 - 1)] + l[l.length / 2]) / 2
def Q3 (l : List ℝ) : ℝ := (l[(3 * l.length / 4 - 1)] + l[3 * l.length / 4]) / 2
def D (l : List ℝ) : ℝ := Q3 l - Q1 l

theorem interquartile_range_theorem :
  D [2, 5, 7, 10, 13, 14, 19, 19, 25, 25, 25, 30, 32, 40, 50, 50, 50, 50, 80, 85] = 36.5 :=
by
  sorry

end interquartile_range_theorem_l391_391107


namespace least_xy_value_l391_391635

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 96 :=
by sorry

end least_xy_value_l391_391635


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391242

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391242


namespace journey_distance_l391_391474

theorem journey_distance (D : ℝ) (h1 : (D / 40) + (D / 60) = 40) : D = 960 :=
by
  sorry

end journey_distance_l391_391474


namespace susans_total_chairs_l391_391110

def number_of_red_chairs := 5
def number_of_yellow_chairs := 4 * number_of_red_chairs
def number_of_blue_chairs := number_of_yellow_chairs - 2
def total_chairs := number_of_red_chairs + number_of_yellow_chairs + number_of_blue_chairs

theorem susans_total_chairs : total_chairs = 43 :=
by
  sorry

end susans_total_chairs_l391_391110


namespace greatest_prime_factor_of_15f_17f_is_17_l391_391362

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l391_391362


namespace petya_friends_count_l391_391064

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l391_391064


namespace greatest_prime_factor_15_fact_17_fact_l391_391230

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l391_391230


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391413

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391413


namespace find_first_week_customers_l391_391732

def commission_per_customer := 1
def first_week_customers (C : ℕ) := C
def second_week_customers (C : ℕ) := 2 * C
def third_week_customers (C : ℕ) := 3 * C
def salary := 500
def bonus := 50
def total_earnings := 760

theorem find_first_week_customers (C : ℕ) (H : salary + bonus + commission_per_customer * (first_week_customers C + second_week_customers C + third_week_customers C) = total_earnings) : 
  C = 35 :=
by
  sorry

end find_first_week_customers_l391_391732


namespace petya_friends_count_l391_391063

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l391_391063


namespace least_xy_value_l391_391634

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 96 :=
by sorry

end least_xy_value_l391_391634


namespace find_digit_for_multiple_of_6_l391_391115

-- Definitions based on conditions
def is_multiple_of_6 (n : ℕ) : Prop :=
  n % 6 = 0

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := [n / 100000 % 10, n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  digits.foldl (· + ·) 0

-- The main proof problem
theorem find_digit_for_multiple_of_6 :
  ∃ (d : ℕ), (d ∈ {0, 2, 4, 6, 8} ∧ is_multiple_of_6 (142850 + d)) ∧ d = 4 :=
sorry

end find_digit_for_multiple_of_6_l391_391115


namespace total_blossoms_l391_391822

theorem total_blossoms (first second third : ℕ) (h1 : first = 2) (h2 : second = 2 * first) (h3 : third = 4 * second) : first + second + third = 22 :=
by
  sorry

end total_blossoms_l391_391822


namespace min_value_two_x_plus_y_l391_391807

theorem min_value_two_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y + 2 * x * y = 5 / 4) : 2 * x + y ≥ 1 :=
by
  sorry

end min_value_two_x_plus_y_l391_391807


namespace greatest_prime_factor_of_15_add_17_factorial_l391_391268

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l391_391268


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391237

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391237


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391311
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391311


namespace find_M_plus_10m_l391_391102

def problem_statement
  (x y z : ℝ)
  (h : 3 * (x + y + z) = x^2 + y^2 + z^2) :
  xy + xz + yz :=
  sorry

theorem find_M_plus_10m :
  ∀ x y z : ℝ,
    (3 * (x + y + z) = x^2 + y^2 + z^2) →
    let M := maxv (xy + xz + yz)
    let m := minv (xy + xz + yz)
    M + 10 * m = -94.5 :=
sorry

end find_M_plus_10m_l391_391102


namespace hexagon_monochromatic_triangle_probability_l391_391586

-- Define the problem statement
theorem hexagon_monochromatic_triangle_probability :
  let hexagon_edges := 15 in
  let edge_colors := [red, blue, green] in
  let total_configurations := (edge_colors.length) ^ hexagon_edges in
  -- Correct answer is the probability of having at least one monochromatic triangle
  let probability_of_monochromatic_triangle := 223 / 256 in
  probability_of_monochromatic_triangle = (223 / 256) :=
sorry

end hexagon_monochromatic_triangle_probability_l391_391586


namespace distance_from_focus2_l391_391831

-- Define the properties and given conditions
def ellipse (x y : ℝ) : Prop := (x^2) / 36 + (y^2) / 9 = 1
def semiMajorAxis : ℝ := 6
def sum_of_distances_to_foci (P : ℝ × ℝ) : ℝ := 12

-- Define the condition that point P is at a distance of 5 from one focus
def distance_from_focus1 (P focus1 : ℝ × ℝ) : ℝ := 5

-- Hypothesize the position of focus1 and focus2
variables (focus1 focus2 : ℝ × ℝ)

-- Main statement to be proved
theorem distance_from_focus2 (P : ℝ × ℝ) (h_ellipse : ellipse P.1 P.2) 
  (h_dist1 : distance_from_focus1 P focus1) 
  (h_sum : sum_of_distances_to_foci P) : 
  ∃ d : ℝ, d = 7 :=
by
  sorry

end distance_from_focus2_l391_391831


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391249

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391249


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391386

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391386


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391425

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391425


namespace chests_contents_l391_391741

def ChestLabel (c : ℕ) : String := 
  if c = 1 then "Gold coins"
  else if c = 2 then "Silver coins"
  else if c = 3 then "Gold or silver coins"
  else "Invalid chest"

def CoinsInChest (c : ℕ) : String := 
  if c = 1 then "Silver coins"
  else if c = 2 then "Gold coins"
  else if c = 3 then "Copper coins"
  else "Invalid chest"

theorem chests_contents :
  ChestLabel 1 ≠ CoinsInChest 1 ∧
  ChestLabel 2 ≠ CoinsInChest 2 ∧
  ChestLabel 3 ≠ CoinsInChest 3 ∧
  (CoinsInChest 1 = "Gold coins" ∨ CoinsInChest 2 = "Gold coins" ∨ CoinsInChest 3 = "Gold coins") ∧
  (CoinsInChest 1 = "Silver coins" ∨ CoinsInChest 2 = "Silver coins" ∨ CoinsInChest 3 = "Silver coins") ∧
  (CoinsInChest 1 = "Copper coins" ∨ CoinsInChest 2 = "Copper coins" ∨ CoinsInChest 3 = "Copper coins") :=
begin
  sorry
end

end chests_contents_l391_391741


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391245

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391245


namespace area_of_triangle_ABC_with_O_l391_391834

-- Define the condition of the isosceles right triangle and distances from point O
variables (A B C O : ℝ) (a b c : ℝ) (OA OB OC : ℝ)
variable (area_ABC : ℝ)

-- Given conditions as hypotheses
axiom OA_dist : OA = 6
axiom OB_dist : OB = 4
axiom OC_dist : OC = 8

-- Define the isosceles right triangle and find the area
theorem area_of_triangle_ABC_with_O :
  is_isosceles_right_triangle A B C →
  dist O A = OA →
  dist O B = OB →
  dist O C = OC →
  area_ABC = 20 + 6 * Real.sqrt 7 :=
by
  sorry

end area_of_triangle_ABC_with_O_l391_391834


namespace fractions_order_l391_391469

theorem fractions_order :
  (25 / 21 < 23 / 19) ∧ (23 / 19 < 21 / 17) :=
by {
  sorry
}

end fractions_order_l391_391469


namespace max_min_f_at_a_neg_half_monotonicity_f_inequality_f_neg_one_lt_a_lt_zero_range_of_a_for_inequality_l391_391660

variable (a : ℝ) (x : ℝ)

noncomputable def f (x : ℝ) (a : ℝ) := a * Real.log x + (a + 1) / 2 * x ^ 2 + 1

-- Part I: maximum and minimum values
theorem max_min_f_at_a_neg_half : 
  a = -1/2 → 
  f x a = max (f (1/e) a) (f e a) :=
  sorry

-- Part II: monotonicity discussion
theorem monotonicity_f :
  if a <= -1 then ∀ x > 0, f' x <= 0 
  else if a >= 0 then ∀ x > 0, f' x >= 0
  else -1 < a < 0 → 
       ∀ x > sqrt (-a / (a + 1)), f' x >= 0 ∧ 
       ∀ x < sqrt (-a / (a + 1)), f' x <= 0 :=
  sorry

-- Part III: inequality for -1 < a < 0
theorem inequality_f_neg_one_lt_a_lt_zero :
  -1 < a ∧ a < 0 →
  ∀ x > 0, 
  f x a > 1 + a / 2 * Real.log (-a) :=
  sorry

-- Range of a for the given inequality
theorem range_of_a_for_inequality :
  -1 < a ∧ a < 0 →
  a > 1 / Real.exp 1 - 1 :=
  sorry

end max_min_f_at_a_neg_half_monotonicity_f_inequality_f_neg_one_lt_a_lt_zero_range_of_a_for_inequality_l391_391660


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l391_391298

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l391_391298


namespace inverse_function_l391_391909

noncomputable def f (x : ℝ) := 3 - 7 * x + x^2

noncomputable def g (x : ℝ) := (7 + Real.sqrt (37 + 4 * x)) / 2

theorem inverse_function :
  ∀ x : ℝ, f (g x) = x :=
by
  intros x
  sorry

end inverse_function_l391_391909


namespace sqrt_factorial_expression_l391_391462

theorem sqrt_factorial_expression :
    sqrt (4! * 4! + 4) = 2 * sqrt 145 :=
by
    sorry

end sqrt_factorial_expression_l391_391462


namespace evaluate_nested_function_l391_391614

def f (x : ℝ) : ℝ := if x >= 0 then x^2 else -x

theorem evaluate_nested_function : f (f (-2)) = 4 := by
  sorry

end evaluate_nested_function_l391_391614


namespace greatest_prime_factor_of_factorial_sum_l391_391321

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l391_391321


namespace trig_expr_eq_l391_391641

theorem trig_expr_eq (α : ℝ) (h : Real.sin α = -3/5) :
    (Real.cos (2 * Real.pi - α) * Real.cos (Real.pi + α) * Real.tan (2 * Real.pi - α) ^ 2) /
      (Real.cos (Real.pi/2 + α) * Real.sin (2 * Real.pi - α) * Real.cot (Real.pi - α) ^ 2) = -9/16 := sorry

end trig_expr_eq_l391_391641


namespace sum_of_squares_of_roots_l391_391549

noncomputable def given_polynomial : Polynomial ℝ :=
  Polynomial.X ^ 3 - 9 * Polynomial.X ^ 2 + 8 * Polynomial.X - 2

theorem sum_of_squares_of_roots :
  (∀ r s t : ℝ, (r, s, t).perm (Polynomial.roots given_polynomial) → 
  r > 0 ∧ s > 0 ∧ t > 0 ∨ 
  r + s + t = 9 ∧
  r * s + s * t + r * t = 8 →
  r ^ 2 + s ^ 2 + t ^ 2 = 65) :=
sorry

end sum_of_squares_of_roots_l391_391549


namespace greatest_prime_factor_15f_plus_17f_l391_391191

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l391_391191


namespace soccer_players_arrangement_l391_391520

theorem soccer_players_arrangement : ∃ (x1 x2 x3 x4 : ℝ), 
    let dists := {(abs (x1 - x2)), (abs (x1 - x3)), (abs (x1 - x4)), (abs (x2 - x3)), (abs (x2 - x4)), (abs (x3 - x4))} in
    dists = {1, 2, 3, 4, 5, 6} :=
sorry

end soccer_players_arrangement_l391_391520


namespace average_speed_round_trip_l391_391945

theorem average_speed_round_trip (D T : ℝ) (h1 : D = 51 * T) : (2 * D) / (3 * T) = 34 := 
by
  sorry

end average_speed_round_trip_l391_391945


namespace greatest_prime_factor_15_fact_plus_17_fact_l391_391437

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l391_391437


namespace petya_has_19_friends_l391_391054

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l391_391054


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l391_391304
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l391_391304


namespace probability_plane_contains_interior_points_of_tetrahedron_l391_391894

theorem probability_plane_contains_interior_points_of_tetrahedron :
  let vertices := ({A, B, C, D} : Finset ℝ^3)
  let planes (v: Finset (Finset ℝ^3))  := (v.image (λ s => s)).toFinset 
  let chosen_planes := (vertices.powerset.filter (λ s => s.card = 3)).toFinset 
  (∀ p ∈ chosen_planes, p ∈ planes {({A, B, C} : Finset ℝ^3), ({A, B, D} : Finset ℝ^3), ({A, C, D} : Finset ℝ^3), ({B, C, D} : Finset ℝ^3)}) →
  ∀ p ∈ chosen_planes, ∀ x ∈ vertices.image id, x ∈ p → The probability that the plane determined by three randomly chosen vertices of a tetrahedron
  contains points inside the tetrahedron is 1 := 1 :=
begin
  sorry
end

end probability_plane_contains_interior_points_of_tetrahedron_l391_391894


namespace greatest_prime_factor_15_fact_17_fact_l391_391175

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391175


namespace min_value_expr_A_not_min_value_expr_B_not_min_value_expr_C_min_value_expr_D_l391_391515

theorem min_value_expr_A (x : ℝ) : 
  ∃ x, (x^2 + 2 + (4 / (x^2 + 2)) - 2) ≥ 2 :=
sorry

theorem not_min_value_expr_B (x : ℝ) : 
  x + (1 / x) < 2 ∨ x + (1 / x) > 2 :=
sorry

theorem not_min_value_expr_C (x : ℝ) : 
  ∃ x, (sqrt(x^2 + 2) + (1 / sqrt(x^2 + 2))) ≠ 2 :=
sorry

theorem min_value_expr_D (a b : ℝ) : 
  ∃ a b, (a^2 + b^2) / |a * b| ≥ 2 :=
sorry

end min_value_expr_A_not_min_value_expr_B_not_min_value_expr_C_min_value_expr_D_l391_391515


namespace intersection_M_N_eq_l391_391669

open Set

theorem intersection_M_N_eq :
  let M := {x : ℝ | x - 2 > 0}
  let N := {y : ℝ | ∃ (x : ℝ), y = Real.sqrt (x^2 + 1)}
  M ∩ N = {x : ℝ | x > 2} :=
by
  sorry

end intersection_M_N_eq_l391_391669


namespace greatest_prime_factor_15_fact_17_fact_l391_391179

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l391_391179


namespace soccer_players_positions_l391_391529

noncomputable def positions : list ℝ := [0, 1, 4, 6]

def pairwise_distances (positions : list ℝ) : list ℝ :=
  let pairs := list.sigma positions positions
  let distances := pairs.map (λ p, abs (p.1 - p.2))
  distances.erase_dup

theorem soccer_players_positions :
  pairwise_distances positions = [1, 2, 3, 4, 5, 6] :=
by {
  sorry  -- Proof to be provided
}

end soccer_players_positions_l391_391529


namespace greatest_prime_factor_of_sum_l391_391405

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l391_391405


namespace dartboard_probability_l391_391490

noncomputable def central_hexagon_probability (s : ℝ) : ℝ :=
  (3 * real.sqrt 3 * (s / 2) ^ 2) / (3 * real.sqrt 3 * s ^ 2)
  
theorem dartboard_probability :
  ∀ (s : ℝ) (h : 0 < s), central_hexagon_probability s = 1 / 4 :=
by
  intro s h
  sorry

end dartboard_probability_l391_391490


namespace chest_contents_l391_391774

def Chest : Type := ℕ
def Coins : Type := ℕ

variable (empty : Coins → Coins → Prop)

noncomputable def Chest1 : Chest := 1
noncomputable def Chest2 : Chest := 2
noncomputable def Chest3 : Chest := 3

variable (goldCoins : Coins)
variable (silverCoins : Coins)
variable (copperCoins : Coins)

-- Conditions
variable (labelsIncorrect : Chest → Coins → Prop)

axiom label1 : labelsIncorrect Chest1 goldCoins
axiom label2 : labelsIncorrect Chest2 silverCoins
axiom label3 : labelsIncorrect Chest3 (goldCoins ∨ silverCoins)

axiom uniqueContents : ∀ c : Chest, c = Chest1 ∨ c = Chest2 ∨ c = Chest3
axiom distinctContents : ∀ c1 c2 : Chest, c1 ≠ c2 → (c1 = Chest1 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest3) ∨ (c1 = Chest2 ∧ c2 ≠ Chest2 ∧ c2 ≠ Chest3) ∨ (c1 = Chest3 ∧ c2 ≠ Chest1 ∧ c2 ≠ Chest2)

theorem chest_contents : (exists! c1 : Coins, c1 = silverCoins ∧ labelsIncorrect Chest1 c1) ∧
                         (exists! c2 : Coins, c2 = goldCoins ∧ labelsIncorrect Chest2 c2) ∧
                         (exists! c3 : Coins, c3 = copperCoins ∧ labelsIncorrect Chest3 c3)
                         :=
by
  sorry

end chest_contents_l391_391774


namespace chests_contents_l391_391735

def ChestLabel (c : ℕ) : String := 
  if c = 1 then "Gold coins"
  else if c = 2 then "Silver coins"
  else if c = 3 then "Gold or silver coins"
  else "Invalid chest"

def CoinsInChest (c : ℕ) : String := 
  if c = 1 then "Silver coins"
  else if c = 2 then "Gold coins"
  else if c = 3 then "Copper coins"
  else "Invalid chest"

theorem chests_contents :
  ChestLabel 1 ≠ CoinsInChest 1 ∧
  ChestLabel 2 ≠ CoinsInChest 2 ∧
  ChestLabel 3 ≠ CoinsInChest 3 ∧
  (CoinsInChest 1 = "Gold coins" ∨ CoinsInChest 2 = "Gold coins" ∨ CoinsInChest 3 = "Gold coins") ∧
  (CoinsInChest 1 = "Silver coins" ∨ CoinsInChest 2 = "Silver coins" ∨ CoinsInChest 3 = "Silver coins") ∧
  (CoinsInChest 1 = "Copper coins" ∨ CoinsInChest 2 = "Copper coins" ∨ CoinsInChest 3 = "Copper coins") :=
begin
  sorry
end

end chests_contents_l391_391735


namespace smallest_largest_same_parity_l391_391560

-- Define the context where our elements and set are considered
variable {α : Type*} [LinearOrderedCommRing α] (A_P : Set α)

-- Assume a nonempty set A_P and define the smallest and largest elements in Lean
noncomputable def smallest_element (A_P : Set α) : α := Inf' A_P sorry
noncomputable def largest_element (A_P : Set α) : α := Sup' A_P sorry

-- State the proof goal
theorem smallest_largest_same_parity (h_nonempty : A_P.nonempty) : 
  (smallest_element A_P) % 2 = (largest_element A_P) % 2 := 
by 
  sorry

end smallest_largest_same_parity_l391_391560


namespace chest_contents_solution_l391_391784

-- Definitions corresponding to the conditions.
structure ChestContents :=
  (chest1 chest2 chest3 : String)

-- Given conditions
def labelsAreIncorrect (contents : ChestContents) : Prop :=
  contents.chest1 ≠ "Gold coins" ∧
  contents.chest2 ≠ "Silver coins" ∧
  contents.chest3 ≠ "Gold coins" ∧
  contents.chest3 ≠ "Silver coins"

def uniqueCoins (contents : ChestContents) : Prop :=
  (contents.chest1 = "Gold coins" ∨ contents.chest1 = "Silver coins" ∨ contents.chest1 = "Copper coins") ∧
  (contents.chest2 = "Gold coins" ∨ contents.chest2 = "Silver coins" ∨ contents.chest2 = "Copper coins") ∧
  (contents.chest3 = "Gold coins" ∨ contents.chest3 = "Silver coins" ∨ contents.chest3 = "Copper coins") ∧
  (contents.chest1 ≠ contents.chest2) ∧
  (contents.chest1 ≠ contents.chest3) ∧
  (contents.chest2 ≠ contents.chest3)

-- The proof statement
theorem chest_contents_solution : ∃ (contents : ChestContents),
  labelsAreIncorrect contents ∧ uniqueCoins contents ∧
  contents.chest1 = "Silver coins" ∧
  contents.chest2 = "Gold coins" ∧
  contents.chest3 = "Copper coins" :=
begin
  sorry
end

end chest_contents_solution_l391_391784


namespace greatest_prime_factor_15_17_factorial_l391_391354

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l391_391354


namespace sequence_perfect_square_l391_391934

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 5 ∧ ∀ n > 1, a n = (∏ i in Finset.Ico 1 n, a i) + 4

theorem sequence_perfect_square (a : ℕ → ℕ) (h : sequence a) : ∀ n ≥ 2, ∃ k : ℕ, a n = k ^ 2 :=
by
  sorry

end sequence_perfect_square_l391_391934


namespace abs_fraction_inequality_l391_391853

theorem abs_fraction_inequality (x : ℝ) :
  (abs ((3 * x - 4) / (x - 2)) > 3) ↔
  (x ∈ Set.Iio (5 / 3) ∪ Set.Ioo (5 / 3) 2 ∪ Set.Ioi 2) :=
by 
  sorry

end abs_fraction_inequality_l391_391853


namespace gathering_handshakes_l391_391544

theorem gathering_handshakes:
  let num_twins := 12
  let num_triplets := 8
  let num_twins_total := num_twins * 2
  let num_triplets_total := num_triplets * 3
  let num_twins_participating := num_twins_total - 2
  let num_triplets_participating := num_triplets_total - 3
  let handshakes_among_twins := num_twins_participating * (num_twins_participating - 2)
  let handshakes_among_triplets := num_triplets_participating * (num_triplets_participating - 3)
  let handshakes_between_twins_and_triplets := num_twins_participating * (num_triplets_participating / 3).toNat + num_triplets_participating * (num_twins_participating / 4).toNat
  let total_handshakes := handshakes_among_twins + handshakes_among_triplets + handshakes_between_twins_and_triplets
  (total_handshakes / 2) = 539 := sorry

end gathering_handshakes_l391_391544


namespace find_z_l391_391691

-- Condition: there exists a constant k such that z = k * w
def direct_variation (z w : ℝ): Prop := ∃ k, z = k * w

-- We set up the conditions given in the problem.
theorem find_z (k : ℝ) (hw1 : 10 = k * 5) (hw2 : w = -15) : direct_variation z w → z = -30 :=
by
  sorry

end find_z_l391_391691


namespace BeethovenBoysCount_l391_391798

theorem BeethovenBoysCount (S B G M Bv G_M : ℕ) (h1 : S = 120) (h2 : B = 65) (h3 : G = 55) 
  (h4 : M = 50) (h5 : Bv = 70) (h6 : G_M = 17) : 
  let B_M = M - G_M in
  let Bv_B = B - B_M in
  Bv_B = 32 :=
by
  sorry

end BeethovenBoysCount_l391_391798


namespace smallest_largest_same_parity_l391_391569

-- Define the set A_P
def A_P : Set ℕ := sorry

-- Define the maximum element c in A_P
def c := max A_P

-- Assuming shifting doesn't change fundamental counts
lemma shifting_preserves_counts (A : Set ℕ) (c : ℕ) : 
  (∀ x ∈ A, x ≠ c → x < c → exists y ∈ A, y < x) ∧
  (∃ p ∈ A, ∃ q ∈ A, p ≠ q ∧ p < q) :=
  sorry

-- Define parity
def parity (n : ℕ) := n % 2

-- Theorem statement
theorem smallest_largest_same_parity (A : Set ℕ) (a_max a_min : ℕ) 
  (h_max : a_max = max A) (h_min : a_min = min A)
  (h_shift : ∀ x ∈ A, shifting_preserves_counts A x) :
  parity a_max = parity a_min :=
sorry

end smallest_largest_same_parity_l391_391569
