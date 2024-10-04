import Mathlib
import Mathlib.Algebra.Abs
import Mathlib.Algebra.BigOperators.Prod
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Prod
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Nonneg
import Mathlib.Algebra.Order.SquareRoot
import Mathlib.Algebra.Roots
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finite
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.Projection
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace find_angle4_l552_552857

def angle (x : Type*) := x

variables {x : Type*} 
variables (angle1 angle2 angle3 angle4 angleA angleB : angle x)

axiom angle_sum_180 : angle1 + angle2 = 180
axiom angle_eq_3 : angle3 = angle4
axiom angle_eq_3_half : angle3 = (1/2) * angle4
axiom angle_A : angleA = 80
axiom angle_B : angleB = 50

theorem find_angle4 : angle4 = 100 / 3 :=
by sorry

end find_angle4_l552_552857


namespace circles_tangent_internally_l552_552851

def circle1 (x y m : ℝ) : Prop :=
  (x - m)^2 + (y + 2)^2 = 9

def circle2 (x y m : ℝ) : Prop :=
  (x + 1)^2 + (y - m)^2 = 4

def centers_tangent_internally (x1 y1 x2 y2 r1 r2 d : ℝ) : Prop :=
  (x1 - x2)^2 + (y1 - y2)^2 = d²

theorem circles_tangent_internally (m : ℝ) :
  centers_tangent_internally m (-2) (-1) m 3 2 1 ↔ m = -2 ∨ m = -1 :=
  sorry

end circles_tangent_internally_l552_552851


namespace jane_green_sequins_rows_l552_552982

theorem jane_green_sequins_rows :
  ∀ (blue_rows purple_rows blue_per_row purple_per_row green_per_row total_sequins : ℕ),
  blue_rows = 6 →
  purple_rows = 5 →
  blue_per_row = 8 →
  purple_per_row = 12 →
  green_per_row = 6 →
  total_sequins = 162 →
  let blue_sequins := blue_rows * blue_per_row in
  let purple_sequins := purple_rows * purple_per_row in
  let green_sequins := total_sequins - (blue_sequins + purple_sequins) in
  ∃ (green_rows : ℕ), green_rows * green_per_row = green_sequins ∧ green_rows = 9 :=
by
  intros blue_rows purple_rows blue_per_row purple_per_row green_per_row total_sequins
  intros hbrows hprows hbprow hpprow hgprows htotal
  let blue_sequins := blue_rows * blue_per_row
  let purple_sequins := purple_rows * purple_per_row
  let green_sequins := total_sequins - (blue_sequins + purple_sequins)
  exists (green_sequins / green_per_row)
  split
  {
    exact nat.mul_div_cancel_left green_sequins (ne_of_gt hgprows)
  }
  {
    sorry
  }

end jane_green_sequins_rows_l552_552982


namespace city_routes_l552_552283

theorem city_routes (h v : ℕ) (H : h = 8) (V : v = 5) : (Nat.choose (h + v) v) = 1287 :=
by
  -- Proof goes here
  sorry

end city_routes_l552_552283


namespace red_numbers_1992_l552_552395

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

def is_red (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ a + b = n

def red_numbers : ℕ → ℕ
| 0     := 8 -- Start from the first red number
| (n+1) :=
  Nat.find (λ m, m > red_numbers n ∧ is_red m)

theorem red_numbers_1992 : red_numbers 1991 = 2001 := by
  sorry

end red_numbers_1992_l552_552395


namespace savings_percentage_l552_552200

variables (S : ℝ)
def saved_last_year := 0.105 * S
def saved_this_year := 0.15 * (1.155 * S - 0.17 * 1.155 * S - 0.08 * 1.155 * S)

theorem savings_percentage : (saved_this_year / saved_last_year) = 1.2375 := by
  sorry

end savings_percentage_l552_552200


namespace remainder_div_l552_552840

open Polynomial

noncomputable def dividend : ℤ[X] := X^4
noncomputable def divisor  : ℤ[X] := X^2 + 3 * X + 2

theorem remainder_div (f g : ℤ[X]) : (f % g) = -6 * X - 6 :=
by
  have f := dividend
  have g := divisor
  sorry

end remainder_div_l552_552840


namespace infinitely_many_elements_in_S_and_P_minus_S_l552_552816

def is_prime (p : ℕ) : Prop := ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p
def A (n : ℕ) := 2^(n^2 + 1) - 3^n
def S := { p : ℕ | ∃ n : ℕ, n > 0 ∧ p ∣ A n ∧ is_prime p }
def P := { p : ℕ | is_prime p }
def P_minus_S := P \ S

theorem infinitely_many_elements_in_S_and_P_minus_S:
  set.infinite S ∧ set.infinite P_minus_S := 
by 
  sorry

end infinitely_many_elements_in_S_and_P_minus_S_l552_552816


namespace partition_disjoint_sets_l552_552081

theorem partition_disjoint_sets (k : ℕ) (hk : 0 < k) :
  ∃ (x y : Finset ℕ),
  x.disjoint y ∧
  x ∪ y = Finset.range (2^(k+1)) ∧
  ∀ m ∈ Finset.range (k + 1), ∑ i in x, i ^ m = ∑ i in y, i ^ m :=
by admit

end partition_disjoint_sets_l552_552081


namespace ratio_of_volumes_l552_552742

-- Define the edge lengths
def edge_length_cube1 : ℝ := 9
def edge_length_cube2 : ℝ := 24

-- Theorem stating the ratio of the volumes
theorem ratio_of_volumes :
  (edge_length_cube1 / edge_length_cube2) ^ 3 = 27 / 512 :=
by
  sorry

end ratio_of_volumes_l552_552742


namespace restaurant_production_in_june_l552_552783

def cheese_pizzas_per_day (hot_dogs_per_day : ℕ) : ℕ :=
  hot_dogs_per_day + 40

def pepperoni_pizzas_per_day (cheese_pizzas_per_day : ℕ) : ℕ :=
  2 * cheese_pizzas_per_day

def hot_dogs_per_day := 60
def beef_hot_dogs_per_day := 30
def chicken_hot_dogs_per_day := 30
def days_in_june := 30

theorem restaurant_production_in_june :
  (cheese_pizzas_per_day hot_dogs_per_day * days_in_june = 3000) ∧
  (pepperoni_pizzas_per_day (cheese_pizzas_per_day hot_dogs_per_day) * days_in_june = 6000) ∧
  (beef_hot_dogs_per_day * days_in_june = 900) ∧
  (chicken_hot_dogs_per_day * days_in_june = 900) :=
by
  sorry

end restaurant_production_in_june_l552_552783


namespace area_enclosed_by_curves_l552_552024

-- Definitions for the curves y^2 = x and y = x^2
def curve_y2_eq_x (x y : ℝ) : Prop := y^2 = x
def curve_y_eq_x2 (x y : ℝ) : Prop := y = x^2

-- The statement to prove the area of the region enclosed by the curves y^2 = x and y = x^2 is 1/3
theorem area_enclosed_by_curves : 
  (∫ x in 0..1, (sqrt x - x^2)) = 1/3 := 
sorry

end area_enclosed_by_curves_l552_552024


namespace dormitory_arrangement_l552_552421

theorem dormitory_arrangement (students : Finset ℕ) (A B : Finset ℕ) (a b : ℕ) :
  students.card = 7 →
  A.card ≥ 2 →
  B.card ≥ 2 →
  a ∈ students →
  b ∈ students →
  a ≠ b →
  a ∈ A →
  b ∈ B →
  (∃ f : ℕ → ℕ, ∀ s ∈ students, ((s = a → f s = 1) ∧ (s = b → f s = 2) ∧ (s ≠ a ∧ s ≠ b → f s ∈ {1, 2})) ∧ (Finset.filter (λ x, f x = 1) students).card ≥ 2 ∧ (Finset.filter (λ x, f x = 2) students).card ≥ 2) →
  A.card + B.card = students.card →
  (A \ {a}).card + (B \ {b}).card = 5 →
  (finset.card (powerset_len 1 (students.erase a.bUnion (students \ {b \ {a}}))) + finset.card (powerset_len 2 (students.erase a.bUnion (students \ {b \ {a}}))) + finset.card (powerset_len 3 (students.erase a.bUnion (students \ {b \ {a}}))) + finset.card (powerset_len 4 (students.erase a.bUnion (students \ {b \ {a}}))) = 30) →
  30 + 30 = 60 :=
sorry

end dormitory_arrangement_l552_552421


namespace limit_fraction_seq_l552_552083

noncomputable def a_seq : ℕ → ℝ
| 0       := 1  -- Lean uses zero-based indexing, so a_1 is actually a_seq 0
| 1       := 3  -- a_2 -> a_seq 1
| (n + 2) := if n % 2 = 0 then a_seq n + 2^(n+1) else a_seq n - 2^(n+1)

-- Conditions
def is_increasing (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, f (2 * n + 1) < f (2 * n + 3)

def is_decreasing (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, f (2 * n + 2) > f (2 * n + 4)


-- Main theorem
theorem limit_fraction_seq : 
  (|a_seq (n+1) - a_seq n| = 2^n) → 
  is_increasing (λ n, a_seq (2*n)) → 
  is_decreasing (λ n, a_seq (2*n+1)) → 
  tendsto (λ n, a_seq (2 * n - 1) / a_seq (2 * n)) at_top (nhds (-1/2)) :=
begin
  sorry
end

end limit_fraction_seq_l552_552083


namespace find_k_such_that_product_minus_one_is_perfect_power_l552_552824

noncomputable def product_of_first_n_primes (n : ℕ) : ℕ :=
  (List.take n (List.filter (Nat.Prime) (List.range n.succ))).prod

theorem find_k_such_that_product_minus_one_is_perfect_power :
  ∀ k : ℕ, ∃ a n : ℕ, (product_of_first_n_primes k) - 1 = a^n ∧ n > 1 ∧ k = 1 :=
by
  sorry

end find_k_such_that_product_minus_one_is_perfect_power_l552_552824


namespace inequality_proof_l552_552262

variable {a b c d : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  64 * (abcd + 1) / (a + b + c + d)^2 ≤ a^2 + b^2 + c^2 + d^2 + 1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 :=
by 
  sorry

end inequality_proof_l552_552262


namespace no_seven_lines_intersection_points_l552_552044

theorem no_seven_lines_intersection_points :
  ¬ ∃ (l : Fin 7 → set (ℝ × ℝ)), 
    (∀ i j, i ≠ j → ∃! p, p ∈ l i ∧ p ∈ l j) ∧  -- 7 distinct lines with pairwise distinct intersection points
    (∃ S3 : set (ℝ × ℝ), (card S3 = 6) ∧ (∀ p ∈ S3, ∃! I : set (Fin 7), card I = 3 ∧ (∀ i ∈ I, p ∈ l i))) ∧  -- 6 points where exactly 3 lines intersect
    (∃ S2 : set (ℝ × ℝ), (card S2 ≥ 4) ∧ (∀ p ∈ S2, ∃! I : set (Fin 7), card I = 2 ∧ (∀ i ∈ I, p ∈ l i)))  -- at least 4 points where exactly 2 lines intersect.
:= sorry

end no_seven_lines_intersection_points_l552_552044


namespace complement_intersection_is_correct_l552_552627

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

def complement_intersection (U A B : Set ℕ) : Set ℕ :=
  U \ (A ∩ B)

theorem complement_intersection_is_correct :
  U = {1, 2, 3, 4, 5} → A = {1, 2, 3} → B = {2, 3, 4} →
  complement_intersection U A B = {1, 4, 5} :=
by
  intros U_set A_set B_set
  simp only [U_set, A_set, B_set, complement_intersection, Set.intersection_eq, Set.diff_eq]
  sorry

end complement_intersection_is_correct_l552_552627


namespace problem_solution_l552_552859

-- Define the problem
noncomputable def a_b_sum : ℝ := 
  let a := 5
  let b := 3
  a + b

-- Theorem statement
theorem problem_solution (a b i : ℝ) (h1 : a + b * i = (11 - 7 * i) / (1 - 2 * i)) (hi : i * i = -1) :
  a + b = 8 :=
by sorry

end problem_solution_l552_552859


namespace determine_blue_numbers_from_red_numbers_l552_552331

theorem determine_blue_numbers_from_red_numbers :
  ∀ (cards : Fin 101 → ℕ) (f : Fin 101 → ℕ), 
  (∀ c : Fin 101, cards c ∈ Finset.range 1 102) →
  (∀ c : Fin 101, f c = Finset.card {d : Fin 101 | cards d < cards c ∧ (d - c).natAbs ≤ 50}) →
  ∃ (determine : (Fin 101 → ℕ) → (Fin 101 → ℕ) → (Fin 101 → ℕ)), 
  (∀ c : Fin 101, determine cards f = cards) :=
begin
  sorry
end

end determine_blue_numbers_from_red_numbers_l552_552331


namespace parabola_directrix_intersection_l552_552284

theorem parabola_directrix_intersection : 
  let parabola := λ x y : ℝ, x^2 = 4 * y in
  let directrix := λ y : ℝ, y = -1 in
  parabola 0 (-1) :=
by
  sorry

end parabola_directrix_intersection_l552_552284


namespace number_of_valid_N_l552_552472

theorem number_of_valid_N : 
  { N : ℕ // 2017 ≡ 17 [MOD N] ∧ N > 17 }.card = 13 :=
by sorry

end number_of_valid_N_l552_552472


namespace min_value_of_a_range_of_m_l552_552119

open Real

noncomputable theory

def f (x : ℝ) := 2 * sin (x + π / 3) + sin x * cos x - sqrt 3 * sin x ^ 2

theorem min_value_of_a (a : ℝ) (h₀ : a > 0) :
  (∀ x, f(x) = f(2 * a - x)) → a = π / 12 :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ x₀ ∈ Icc (0 : ℝ) (5 / 12 * π), m * f x₀ - 2 = 0) → m ≥ 1 ∨ m ≤ -2 :=
sorry

end min_value_of_a_range_of_m_l552_552119


namespace more_sparrows_than_pigeons_l552_552424

-- Defining initial conditions
def initial_sparrows := 3
def initial_starlings := 5
def initial_pigeons := 2
def additional_sparrows := 4
def additional_starlings := 2
def additional_pigeons := 3

-- Final counts after additional birds join
def final_sparrows := initial_sparrows + additional_sparrows
def final_pigeons := initial_pigeons + additional_pigeons

-- The statement to be proved
theorem more_sparrows_than_pigeons:
  final_sparrows - final_pigeons = 2 :=
by
  -- proof skipped
  sorry

end more_sparrows_than_pigeons_l552_552424


namespace min_performances_l552_552764

theorem min_performances (total_singers : ℕ) (m : ℕ) (n_pairs : ℕ := 28) (pairs_performance : ℕ := 6)
  (condition : total_singers = 108) 
  (const_pairs : ∀ (r : ℕ), (n_pairs * r = pairs_performance * m)) : m ≥ 14 :=
by
  sorry

end min_performances_l552_552764


namespace numberOfCorrectPropositions_l552_552414

-- Definitions from conditions
def prop1 : Prop := ¬(0 > complex.I)
def prop2 : Prop := ∀ (z w : ℂ), (z + w).im = 0 → (z = conj w ∨ w = conj z)
def prop3 : Prop := ∀ (x y : ℝ), ¬(x + y * complex.I = 1 + complex.I ↔ x = 1 ∧ y = 1)
def prop4 : Prop := ¬∀ (a : ℝ), ∃ (b : ℂ), b = a * complex.I ∧ a ≠ 0 → b.im ≠ 0

-- The main theorem statement
theorem numberOfCorrectPropositions : (prop1 ∧ prop2 ∧ prop3 ∧ prop4) → 0 = 0 := 
by {
  intros,
  sorry
}

end numberOfCorrectPropositions_l552_552414


namespace nets_win_in_7_games_probability_l552_552968

theorem nets_win_in_7_games_probability :
  let warriors_win_prob := (1 : ℚ) / 4
  let nets_win_prob := (3 : ℚ) / 4
  let binom_coeff := Nat.choose 6 3
  let game_6_warriors_prob := (warriors_win_prob ^ 3) * (nets_win_prob ^ 3)
  let prob_before_game_7 := binom_coeff * game_6_warriors_prob
  let final_prob := prob_before_game_7 * nets_win_prob
  final_prob = 405 / 4096 :=
by
  let warriors_win_prob := (1 : ℚ) / 4
  let nets_win_prob := (3 : ℚ) / 4
  let binom_coeff := Nat.choose 6 3
  let game_6_warriors_prob := (warriors_win_prob ^ 3) * (nets_win_prob ^ 3)
  let prob_before_game_7 := binom_coeff * game_6_warriors_prob
  let final_prob := prob_before_game_7 * nets_win_prob
  have h1 : binom_coeff = 20 := by sorry
  have h2 : game_6_warriors_prob = 27 / 4096 := by sorry
  have h3 : prob_before_game_7 = 540 / 4096 := by sorry
  have h4 : final_prob = (540 / 4096) * (3 / 4) := by sorry
  have h5 : final_prob = 405 / 4096 := by sorry
  exact h5

end nets_win_in_7_games_probability_l552_552968


namespace general_term_formula_l552_552540

variable {a : ℕ → ℝ} -- Define the sequence as a function ℕ → ℝ

-- Conditions
axiom geom_seq (n : ℕ) (h : n ≥ 2): a (n + 1) = a 2 * (2 : ℝ) ^ (n - 1)
axiom a2_eq_2 : a 2 = 2
axiom a3_a4_cond : 2 * a 3 + a 4 = 16

theorem general_term_formula (n : ℕ) : a n = 2 ^ (n - 1) := by
  sorry -- Proof is not required

end general_term_formula_l552_552540


namespace monotonicity_extreme_points_inequality_range_for_a_l552_552113

-- Define the function F(x)
def F (x : ℝ) (a : ℝ) : ℝ := exp x - a * x^2 / 2 + a * x

-- Define the derivative of F
def f (x : ℝ) (a : ℝ) : ℝ := exp x - a * x + a

-- (I) Monotonicity Problem
theorem monotonicity (a : ℝ) :
  (∀ x : ℝ, f' a x > 0) ∨ (∀ x : ℝ, x < ln a → f' a x < 0) ∧ (∀ x : ℝ, x > ln a → f' a x > 0) :=
sorry

-- (II)(i) Inequality Problem
theorem extreme_points_inequality (x₁ x₂ a : ℝ) (h₁ : x₁ < x₂) :
  ln a - 2 < x₂ - x₁ ∧ x₂ - x₁ < 2 * ln a - 1 - a / (a - exp 1) :=
sorry

-- (II)(ii) Range for a
theorem range_for_a (x₁ x₂ a : ℝ) (h₁ : x₁ < x₂) (h₂ : 3 * x₁ - x₂ ≤ 2) :
  a ≥ 2 * sqrt 3 * exp 1 / ln 3 :=
sorry

end monotonicity_extreme_points_inequality_range_for_a_l552_552113


namespace sin_theta_plus_2pi_div_3_cos_theta_minus_5pi_div_6_l552_552509

variable (θ : ℝ)

theorem sin_theta_plus_2pi_div_3 (h : Real.sin (θ - Real.pi / 3) = 1 / 3) :
  Real.sin (θ + 2 * Real.pi / 3) = -1 / 3 :=
  sorry

theorem cos_theta_minus_5pi_div_6 (h : Real.sin (θ - Real.pi / 3) = 1 / 3) :
  Real.cos (θ - 5 * Real.pi / 6) = 1 / 3 :=
  sorry

end sin_theta_plus_2pi_div_3_cos_theta_minus_5pi_div_6_l552_552509


namespace find_k_l552_552848

theorem find_k (n m : ℕ) (hn : n > 0) (hm : m > 0) (h : (1 : ℚ) / n^2 + 1 / m^2 = k / (n^2 + m^2)) : k = 4 :=
sorry

end find_k_l552_552848


namespace expected_value_l552_552504

variable {X : Type} [MeasureSpace X]

def E (f : X → ℝ) : ℝ := ∫ x, f x ∂?m

theorem expected_value (h : E (λ x : X, id x) + E (λ x : X, 2 * id x + 1) = 8) : 
  E (λ x : X, id x) = 7 / 3 := 
by
  sorry -- The proof part is omitted as instructed.

end expected_value_l552_552504


namespace simplify_complex_expr_l552_552657

noncomputable def z1 : ℂ := (-1 + complex.I * real.sqrt 3) / 2
noncomputable def z2 : ℂ := (-1 - complex.I * real.sqrt 3) / 2

theorem simplify_complex_expr :
  z1^12 + z2^12 = 2 := 
  sorry

end simplify_complex_expr_l552_552657


namespace free_module_basis_exists_l552_552204

variables (p : ℕ) [fact (nat.prime p)]
def F_p : Type := zmod p
def tau (f : polynomial F_p) : polynomial F_p := polynomial.eval₂ polynomial.C (polynomial.x + 1) f
def R : subring (polynomial F_p) := subring.closure {f : polynomial F_p | tau f = f}

theorem free_module_basis_exists :
  ∃ (g : polynomial F_p), basis (fin p) (polynomial F_p) (polynomial F_p) :=
begin
  use polynomial.X ^ (p - 1),
  sorry
end

end free_module_basis_exists_l552_552204


namespace sin_value_l552_552096

theorem sin_value (α : ℝ) (h: cos (π / 6 - α) = (sqrt 3)/3) :
  sin (5 * π / 6 - 2 * α) = -1 / 3 :=
sorry

end sin_value_l552_552096


namespace sector_area_l552_552105

theorem sector_area (α : ℝ) (r : ℝ) (hα : α = 3 / 4 * Real.pi) (hr : r = 4) : 
  let S := 1 / 2 * α * r ^ 2 
  in S = 6 * Real.pi :=
by
  sorry

end sector_area_l552_552105


namespace cocos_August_bill_l552_552943

noncomputable def total_cost (a_monthly_cost: List (Float × Float)) :=
a_monthly_cost.foldr (fun x acc => (x.1 * x.2 * 0.09) + acc) 0

theorem cocos_August_bill :
  let oven        := (2.4, 25)
  let air_cond    := (1.6, 150)
  let refrigerator := (0.15, 720)
  let washing_mach := (0.5, 20) 
  total_cost [oven, air_cond, refrigerator, washing_mach] = 37.62 :=
by
  sorry

end cocos_August_bill_l552_552943


namespace min_distance_between_curves_l552_552486

noncomputable def y1 (x : ℝ) := Real.exp (3 * x + 11)
noncomputable def y2 (x : ℝ) := (Real.log x - 11) / 3

theorem min_distance_between_curves :
  let rho (x : ℝ) := Real.sqrt 2 * Real.abs (Real.exp (3 * x + 11) - x)
  rho ((-Real.log 3 - 11) / 3) = Real.sqrt 2 * ((Real.log 3 + 12) / 3) :=
by
  sorry

end min_distance_between_curves_l552_552486


namespace train_speed_l552_552791

def train_length : ℝ := 110
def bridge_length : ℝ := 265
def crossing_time : ℝ := 30
def conversion_factor : ℝ := 3.6

theorem train_speed (train_length bridge_length crossing_time conversion_factor : ℝ) :
  (train_length + bridge_length) / crossing_time * conversion_factor = 45 :=
by
  sorry

end train_speed_l552_552791


namespace num_nat_numbers_with_remainder_17_l552_552478

theorem num_nat_numbers_with_remainder_17 (N : ℕ) :
  (2017 % N = 17 ∧ N > 17) → 
  ({N | 2017 % N = 17 ∧ N > 17}.toFinset.card = 13) := 
by
  sorry

end num_nat_numbers_with_remainder_17_l552_552478


namespace Sarah_won_30_games_l552_552649

namespace TicTacToe

def total_games := 100
def tied_games := 40
def lost_money := 30

def won_games (W : ℕ) (L : ℕ) := 
  (W + 2 * L) = total_games ∧ (W - 2 * L) = (-lost_money)

theorem Sarah_won_30_games : ∃ W L : ℕ, won_games W L ∧ W = 30 :=
by
  sorry

end TicTacToe

end Sarah_won_30_games_l552_552649


namespace abs_diff_single_digit_base6_l552_552818

theorem abs_diff_single_digit_base6 (C D : ℕ) (hC : C < 6) (hD : D < 6)
  (h_eq : (D + D + C = 1 * 6^2 + 1 * 6^1 + 1 * 6^0) ∧
          (5 * 6^2 + 2 * 6^1 + D * 6^0 + C * 6^2 + 2 * 6^1 + 3 * 6^0 = C * 6^3 + 2 * 6^2 + 3 * 6^1 + 1 * 6^0)) :
  |C - D| = 0 := by
  sorry

end abs_diff_single_digit_base6_l552_552818


namespace count_prime_sums_is_two_l552_552311

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def prime_numbers : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def prime_sums : List ℕ := 
  let sums := foldl (λ acc p, acc ++ [acc.head! + p]) [3] prime_numbers
  sums.take 15

def count_primes (l : List ℕ) : ℕ :=
  l.countp is_prime

theorem count_prime_sums_is_two : count_primes prime_sums = 2 := 
  by
    sorry

end count_prime_sums_is_two_l552_552311


namespace nth_smallest_rel_prime_bound_l552_552203

open Nat

def sum_of_divisors (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (finset.filter (∣ n) (finset.range (n + 1))).sum id

def is_prime_power (n : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), prime p ∧ n = p ^ k

theorem nth_smallest_rel_prime_bound (n : ℕ) (hn : n ≥ 2) :
  let σ := sum_of_divisors n in
  ∃ k, (relatively_prime_seq n k) ≥ σ ∧ (relatively_prime_seq n k = σ ↔ is_prime_power n) :=
sorry

end nth_smallest_rel_prime_bound_l552_552203


namespace age_difference_l552_552425

theorem age_difference (b_age : ℕ) (bro_age : ℕ) (h1 : b_age = 5) (h2 : b_age + bro_age = 19) : 
  bro_age - b_age = 9 :=
by
  sorry

end age_difference_l552_552425


namespace median_of_list_is_2240012_5_l552_552342

/--
Given a list containing integers from 1 to 3000, their squares, and their cubes,
prove that the median of this list is $2240012.5$.
-/
theorem median_of_list_is_2240012_5 :
  let l : List ℕ := (List.range 3000).map (λ x, x + 1) ++ (List.range 3000).map (λ x, (x + 1)^2) ++ (List.range 3000).map (λ x, (x + 1)^3),
      sorted_l := l.toList.sorted,
      n := sorted_l.length / 2 -- Since length is 9000, median will be average of middle two elements.
  in
  (sorted_l.get n + sorted_l.get (n - 1)) / 2 = 2240012.5 :=
by sorry

end median_of_list_is_2240012_5_l552_552342


namespace isosceles_trapezoid_diagonal_l552_552695

-- Definitions based on the conditions:
variables (a b c d : ℝ)

-- The proof statement:
theorem isosceles_trapezoid_diagonal (h1 : ∀ ABCD, ABCD.bases = (a, b))
                                      (h2 : ∀ ABCD, ABCD.lateral_side = c)
                                      (h3 : ∀ ABCD, ABCD.diagonal = d) :
  d^2 = a * b + c^2 :=
by
  sorry

end isosceles_trapezoid_diagonal_l552_552695


namespace pattern_8_pattern_n_product_2019_l552_552847

-- Define the pattern as a theorem
theorem pattern_8 : 1 - (1 / (8 ^ 2)) = (7 / 8) * (9 / 8) := sorry

-- Define the general pattern as a theorem
theorem pattern_n (n : ℕ) (hn : n > 0) : 1 - (1 / (n ^ 2)) = ((n - 1) / n) * ((n + 1) / n) := sorry

-- Define the product theorem
theorem product_2019 : 
  ∏ k in (range 2018).map (λ n, n + 2), (1 - (1 / (k ^ 2))) = (1010 / 2019) := sorry

end pattern_8_pattern_n_product_2019_l552_552847


namespace rice_difference_l552_552391

-- A Chessboard has squares labeled from 1 to 64
-- Each square 'k' has '2^k' grains of rice
-- We need to prove that the rice difference between 12th square and the sum of rice on the first 10 squares equals 2050

theorem rice_difference :
  let grains_on_square : ℕ → ℕ := λ k, 2^k in
  let square_12 := grains_on_square 12 in
  let first_10_squares := ∑ i in (finset.range 10).image (λ n, n + 1), grains_on_square i in
  square_12 - first_10_squares = 2050 :=
by
  sorry

end rice_difference_l552_552391


namespace number_of_digits_in_product_l552_552357

/--
Given the conditions of multiplying the base numbers and adding exponents for powers of ten,
prove that the number of digits in the product of (8 × 10^10) and (10 × 10^5) is 17.
-/
theorem number_of_digits_in_product : 
  let a := 8 * 10
  let b := 10^10
  let c := 10 * 10^5
  let n := 15 in
  a * 10^n = 8 * 10^16 ->
  digits (8 * 10^16) = 17 :=
sorry

end number_of_digits_in_product_l552_552357


namespace first_15_prime_sums_count_zero_l552_552304

/-- A function to check if a number is prime -/
def is_prime (n : ℕ) : Prop := nat.prime n

/-- Defining the first 15 sums of consecutive primes starting from 3 -/
def prime_sums : ℕ → ℕ
| 1     := 3
| (n+1) := prime_sums n + nat.prime (n+1)

noncomputable def count_prime_sums_up_to_15 : ℕ :=
(finset.range 15).count (λ n, is_prime (prime_sums (n+1)))

theorem first_15_prime_sums_count_zero : count_prime_sums_up_to_15 = 0 :=
by sorry

end first_15_prime_sums_count_zero_l552_552304


namespace minimum_distance_point_l552_552878

-- Definitions of points A and B
def A : ℝ × ℝ := (-3, 8)
def B : ℝ × ℝ := (2, 2)

-- Definition of the minimum distance point M on the x-axis
def M : ℝ × ℝ := (1, 0)

-- Proof statement
theorem minimum_distance_point :
  ∃ M : ℝ × ℝ, M = (1, 0) ∧ (M.2 = 0) ∧ ∀ x ∈ (real.line 0 1), |dist A M + dist B M| ≥ |dist A M + dist B (1, 0)| :=
begin
  sorry
end

end minimum_distance_point_l552_552878


namespace petya_sum_expression_l552_552249

theorem petya_sum_expression : 
  (let expressions := finset.image (λ (s : list bool), 
    list.foldl (λ acc ⟨b, n⟩, if b then acc + n else acc - n) 1 (s.zip [2, 3, 4, 5, 6])) 
    (finset.univ : finset (vector bool 5))) in
    expressions.sum) = 32 := 
sorry

end petya_sum_expression_l552_552249


namespace pentagon_cannot_cover_and_can_cover_2015_points_l552_552871

noncomputable def regular_pentagon_inradius (r : ℝ) : Type :=
{ s : Type // is_regular_pentagon_with_inradius s r }

def exists_region_B (r : ℝ) : Prop :=
  ∃ (B : set ℝ), ∀ (x : ℝ)s ∈ regular_pentagon_inradius r,
  ¬(∀ t ∈ B, t ∈ s) ∧ 
  ∀ (P : fin 2015 → ℝ), (∀ i, P i ∈ B) → 
  ∃ (s : regular_pentagon_inradius r), ∀ i, P i ∈ s

theorem pentagon_cannot_cover_and_can_cover_2015_points (r : ℝ) (A : regular_pentagon_inradius r) :
  exists_region_B r :=
sorry

end pentagon_cannot_cover_and_can_cover_2015_points_l552_552871


namespace Brenda_bakes_cakes_l552_552018

theorem Brenda_bakes_cakes 
  (cakes_per_day : ℕ)
  (days : ℕ)
  (sell_fraction : ℚ)
  (total_cakes_baked : ℕ := cakes_per_day * days)
  (cakes_left : ℚ := total_cakes_baked * sell_fraction)
  (h1 : cakes_per_day = 20)
  (h2 : days = 9)
  (h3 : sell_fraction = 1 / 2) :
  cakes_left = 90 := 
by 
  -- Proof to be filled in later
  sorry

end Brenda_bakes_cakes_l552_552018


namespace dragon_heads_mean_gt_1988_l552_552166

open_locale big_operators

theorem dragon_heads_mean_gt_1988
  (n : ℕ)
  (a : fin n → ℝ)
  (h_exists_dragon : ∃ k l : fin n, k < k + l ∧ (∑ i in (finset.range l).map (finset.add k),
                                                 a i)/(l : ℝ) > 1988) :
  ∑ (i : fin n) in {i | ∃ l : ℕ, (i + l < n) ∧ ( (∑ j in (finset.range (l + 1)).map (finset.add i), a j) / (l + 1 : ℝ) > 1988)}, a i /
  (finset.card {i | ∃ l : ℕ, (i + l < n) ∧ ( (∑ j in (finset.range (l + 1)).map (finset.add i), a j) / (l + 1 : ℝ) > 1988)}: ℝ) > 1988 :=
sorry

end dragon_heads_mean_gt_1988_l552_552166


namespace vertex_of_parabola_point_symmetry_on_parabola_range_of_m_l552_552133

open Real

-- Problem 1: Prove the vertex of the parabola is at (1, -a)
theorem vertex_of_parabola (a : ℝ) (h : a ≠ 0) : 
  ∀ x : ℝ, y = a * x^2 - 2 * a * x → (1, -a) = ((1 : ℝ), - a) := 
sorry

-- Problem 2: Prove x_0 = 3 if m = n for given points on the parabola
theorem point_symmetry_on_parabola (a : ℝ) (h : a ≠ 0) (m n : ℝ) :
  m = n → ∀ (x0 : ℝ), y = a * x0 ^ 2 - 2 * a * x0 → x0 = 3 :=
sorry

-- Problem 3: Prove the conditions for y1 < y2 ≤ -a and the range of m
theorem range_of_m (a : ℝ) (h : a < 0) : 
  ∀ (m y1 y2 : ℝ), (y1 < y2) ∧ (y2 ≤ -a) → m < (1 / 2) := 
sorry

end vertex_of_parabola_point_symmetry_on_parabola_range_of_m_l552_552133


namespace area_of_rectangle_l552_552371

-- Define the given conditions
def length : Real := 5.9
def width : Real := 3
def expected_area : Real := 17.7

theorem area_of_rectangle : (length * width) = expected_area := 
by 
  sorry

end area_of_rectangle_l552_552371


namespace tan_2715_eq_half_l552_552442

noncomputable def compute_tan_2715 : ℝ := 
  let angle := 2715 % 360 in 
  if angle = 195 then
    (Real.sin 15) / (Real.cos 15)
  else
    0 -- placeholder, we know angle will be 195

theorem tan_2715_eq_half : compute_tan_2715 = 1 / 2 := by
  sorry

end tan_2715_eq_half_l552_552442


namespace determine_value_of_x_l552_552152

theorem determine_value_of_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 2 = 6 * y) : x = 48 :=
by
  sorry

end determine_value_of_x_l552_552152


namespace new_home_capacity_l552_552566

theorem new_home_capacity (M H : ℕ) (h1 : H = 1/2 * M) 
                          (h2 : new_home_capacity = 1/3 * H + 1/2 * M) :
                          new_home_capacity = 2/3 * M :=
sorry

end new_home_capacity_l552_552566


namespace points_lines_configuration_l552_552854

structure Point (α : Type*) := 
  (x : α)
  (y : α)

noncomputable def line_through_points {α : Type*} [LinearOrderedField α] (p q : Point α) := 
  { r : Point α // ∃λ : α, r.x = p.x + λ * (q.x - p.x) ∧ r.y = p.y + λ * (q.y - p.y)}

def points_on_line {α : Type*} [LinearOrderedField α] (l : set (Point α)) (points : set (Point α)) (k : ℕ) :=
  ∃ (subset : finset (Point α)), (↑subset ⊆ points) ∧ (subset.card = k) ∧ (∀ p ∈ subset, p ∈ l)

theorem points_lines_configuration : 
  ∃ (points : finset (Point ℝ)), 
    (points.card = 17) ∧ 
    (∀ k : ℕ, (1 ≤ k ∧ k ≤ 17) → (∃ l : set (Point ℝ), points_on_line l ↑points k)) :=
begin
  sorry
end

end points_lines_configuration_l552_552854


namespace remainder_of_2_pow_2018_plus_1_mod_2018_l552_552419

theorem remainder_of_2_pow_2018_plus_1_mod_2018 : (2 ^ 2018 + 1) % 2018 = 2 := by
  sorry

end remainder_of_2_pow_2018_plus_1_mod_2018_l552_552419


namespace probability_not_hear_favorite_song_l552_552047

theorem probability_not_hear_favorite_song (songs : Fin 12 → ℕ) 
  (h₀ : ∀ n : Fin 12, songs n = 40 * (n + 1)) 
  (favorite_song : ℕ) (h₁ : favorite_song = 240) 
  (H : ∃ n : Fin 12, songs n = favorite_song) 
  (random_play : Permutations (Fin 12)) : 
  (300 seconds listening - favorite_song / all combinations).probability  = 10 / 11 :=
sorry

end probability_not_hear_favorite_song_l552_552047


namespace point_P_coordinates_l552_552628

theorem point_P_coordinates : 
  (∃ P : ℝ × ℝ, (P.fst ^ 3 - P.fst = P.snd) ∧ (2 * P.fst - P.snd = 2)) → (1 : ℝ, 0 : ℝ) :=
by
  sorry

end point_P_coordinates_l552_552628


namespace lowest_years_of_service_l552_552301

theorem lowest_years_of_service (num_employees : ℕ) (range_of_service : ℕ) (second_lowest_service: ℕ) 
  (h1 : num_employees = 8) (h2 : range_of_service = 14) (h3 : second_lowest_service = 10) :
  ∃ (lowest_service : ℕ), lowest_service = 0 :=
by
  use 0
  sorry

end lowest_years_of_service_l552_552301


namespace Petya_sum_l552_552256

theorem Petya_sum : 
  let expr := [1, 2, 3, 4, 5, 6]
  let values := 2^(expr.length - 1)
  (sum_of_possible_values expr = values) := by 
  sorry

end Petya_sum_l552_552256


namespace arithmetic_square_root_l552_552932

theorem arithmetic_square_root (n : ℝ) (h : (-5)^2 = n) : Real.sqrt n = 5 :=
by
  sorry

end arithmetic_square_root_l552_552932


namespace f_log_two_three_l552_552288

noncomputable def f : ℝ → ℝ
| x => if x ≥ 4 then (1 / 2) ^ x else f (x + 1)

lemma log_two_gt_one_lt_two : 1 < Real.log 2 3 ∧ Real.log 2 3 < 2 :=
sorry

theorem f_log_two_three : f (Real.log 2 3) = 1 / 24 :=
by
  have h1 : 1 < Real.log 2 3 := (log_two_gt_one_lt_two.fst)
  have h2 : Real.log 2 3 < 2 := (log_two_gt_one_lt_two.snd)
  -- Further proof steps here
  sorry

end f_log_two_three_l552_552288


namespace prime_sum_of_digits_base_31_l552_552069

-- Define the sum of digits function in base k
def sum_of_digits_in_base (k n : ℕ) : ℕ :=
  let digits := (Nat.digits k n)
  digits.foldr (· + ·) 0

theorem prime_sum_of_digits_base_31 (p : ℕ) (hp : Nat.Prime p) (h_bound : p < 20000) : 
  sum_of_digits_in_base 31 p = 49 ∨ sum_of_digits_in_base 31 p = 77 :=
by
  sorry

end prime_sum_of_digits_base_31_l552_552069


namespace triangle_side_length_l552_552138

theorem triangle_side_length {AB BC CA DE EF : ℝ}
  (h_AB : AB = 9) (h_BC : BC = 21) (h_CA : CA = 15)
  (angle_ABC : real.angle = 60) (h_DE : DE = 6) (h_EF : EF = 12) (angle_DEF : real.angle = 60) :
  ∃ (DF : ℝ), DF = 14 :=
by
  use 14
  sorry

end triangle_side_length_l552_552138


namespace solve_asterisk_l552_552754

theorem solve_asterisk (x : ℝ) (h : (x / 21) * (x / 84) = 1) : x = 42 :=
sorry

end solve_asterisk_l552_552754


namespace cauchy_schwarz_inequality_l552_552889

theorem cauchy_schwarz_inequality (n : ℕ) (x : Fin n → ℝ) (y : Fin n → ℕ) 
  (hy_pos : ∀ i, 0 < y i) :
  (∑ i, (x i)^2 / (y i : ℝ)) ≥ ((∑ i, x i)^2 / (∑ i, (y i : ℝ))) :=
by
  sorry

end cauchy_schwarz_inequality_l552_552889


namespace rubles_for_eur_package_l552_552928

variable (EUR USD RUB : Type)
variable [HasScalar ℝ EUR] [HasScalar ℝ USD] [HasScalar ℝ RUB]

def exchange_rate_eur_to_usd : ℝ := 12 / 10
def exchange_rate_usd_to_rub : ℝ := 60
def package_cost_eur : ℝ := 600

theorem rubles_for_eur_package :
  (package_cost_eur * exchange_rate_eur_to_usd * exchange_rate_usd_to_rub) = 43200 :=
by
  sorry

end rubles_for_eur_package_l552_552928


namespace youngest_age_is_10_l552_552693

-- Define the total age of the family currently
def current_total_age (n : ℕ) (avg_current_age : ℕ) : ℕ := n * avg_current_age

-- Define the total age of the family at the time of the birth of the youngest member
def past_total_age (n_pred : ℕ) (avg_past_age : ℕ) : ℕ := n_pred * avg_past_age

-- Define the proof problem
theorem youngest_age_is_10 (n : ℕ) (avg_current_age : ℕ) (avg_past_age : ℕ) (Y : ℕ) 
  (h1 : n = 5) 
  (h2 : avg_current_age = 20) 
  (h3 : avg_past_age = 12.5) 
  (h4 : current_total_age n avg_current_age - past_total_age (n - 1) avg_past_age = 5 * Y) 
  : Y = 10 := 
by 
  -- We assert that the values hold as given in the conditions.
  rw [h1, h2, h3] at h4,
  -- We simplify the given equation to derive Y.
  have h5 : current_total_age 5 20 = 100 := by norm_num,
  have h6 : past_total_age 4 12.5 = 50 := by norm_num,
  rw [h5, h6] at h4,
  simp at h4,
  exact h4

end youngest_age_is_10_l552_552693


namespace final_price_including_tax_l552_552712

def original_suit_price: ℝ := 200
def original_tie_price: ℝ := 50
def suit_increase_percentage: ℝ := 0.30
def tie_increase_percentage: ℝ := 0.20
def suit_discount_percentage: ℝ := 0.30
def tie_discount_percentage: ℝ := 0.10
def sales_tax_percentage: ℝ := 0.07

theorem final_price_including_tax:
  let final_price := 
    let increased_suit_price := original_suit_price * (1 + suit_increase_percentage) in
    let increased_tie_price := original_tie_price * (1 + tie_increase_percentage) in
    let discounted_suit_price := increased_suit_price * (1 - suit_discount_percentage) in
    let discounted_tie_price := increased_tie_price * (1 - tie_discount_percentage) in
    let combined_discounted_price := discounted_suit_price + discounted_tie_price in
    combined_discounted_price * (1 + sales_tax_percentage) in
  final_price = 252.52 := by
  sorry

end final_price_including_tax_l552_552712


namespace area_bound_of_subsections_in_triangle_l552_552941

theorem area_bound_of_subsections_in_triangle
  (A B C P E F : Point)
  (hP_on_BC : OnSegment P B C)
  (hPE_parallel_BA : ∥(P, E) (B, A))
  (hPF_parallel_CA : ∥(P, F) (C, A))
  (area_ABC : area (Triangle A B C) = 1) :
  
  area (Triangle B P F) ≥ 4 / 9 ∨ area (Triangle P C E) ≥ 4 / 9 ∨ area (Quadrilateral P E A F) ≥ 4 / 9 :=
sorry

end area_bound_of_subsections_in_triangle_l552_552941


namespace nth_derivative_of_exponential_l552_552423

noncomputable def exponential_fn (a x : ℝ) : ℝ := a^x

def nth_derivative (f : ℝ → ℝ) (n : ℕ) : ℝ → ℝ :=
  if n = 0 then f
  else (fun x => (Real.log a) ^ n * f x)

theorem nth_derivative_of_exponential (a x : ℝ) (h : 0 < a) :
  nth_derivative (exponential_fn a) 2011 x = a^x * (Real.log a) ^ 2011 :=
by
  sorry

end nth_derivative_of_exponential_l552_552423


namespace hyperbola_eccentricity_is_correct_l552_552691

/-- Definition of a hyperbola centered at the origin with given conditions -/
def hyperbola_C (x y : ℝ) : Prop :=
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a^2 * y^2 - b^2 * x^2 = a^2 * b^2)

/-- Definition of the circle -/
def circle (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

/-- The asymptotes of the hyperbola C -/
def asymptotes_tangent_to_circle (k : ℝ) : Prop :=
  abs (2 * k) = sqrt (k^2 + 1)

/-- Eccentricity of hyperbola C -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_is_correct {a b : ℝ} (h_asymptotes : ∃ k, asymptotes_tangent_to_circle k)
  (h_C : hyperbola_C a b) : eccentricity a b = 2 * sqrt 3 / 3 ∨ eccentricity a b = 2 :=
sorry

end hyperbola_eccentricity_is_correct_l552_552691


namespace sum_sequence_2022_l552_552522

section
variables (n : ℕ) (x : ℕ → ℚ) 

def sequence_satisfies : Prop :=
  x 0 = 1 / n ∧ ∀ k : ℕ, k < n → x (k + 1) = (1 / (n - (k + 1))) * (∑ i in Finset.range (k + 1), x i)

theorem sum_sequence_2022 (h : sequence_satisfies 2022 x) : 
  ∑ i in Finset.range 2022, x i = 1 :=
begin
  -- proof goes here
  sorry
end

end

end sum_sequence_2022_l552_552522


namespace hyperbola_vertex_distance_l552_552487

noncomputable def distance_between_vertices : ℝ :=
  let equation : (ℝ × ℝ) → ℝ := λ (xy : ℝ × ℝ), 4 * xy.1^2 - 16 * xy.1 - 16 * xy.2^2 + 32 * xy.2 + 144
  in 12

theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, equation (x, y) = 0 ↔ distance_between_vertices = 12 :=
begin
  sorry
end

end hyperbola_vertex_distance_l552_552487


namespace scissor_count_l552_552327

theorem scissor_count :
  let initial_scissors := 54 
  let added_scissors := 22
  let removed_scissors := 15
  initial_scissors + added_scissors - removed_scissors = 61 := by
  sorry

end scissor_count_l552_552327


namespace mohit_discount_l552_552228

def discount_percentage (S C : ℝ) (P_percent : ℝ) : ℝ :=
  let profit := P_percent * C
  let selling_price_for_profit := C + profit
  let discount := S - selling_price_for_profit
  (discount / S) * 100

theorem mohit_discount (S C : ℝ) (P_percent : ℝ) (D : ℝ) : 
  S = 12000 ∧ C = 10000 ∧ P_percent = 0.08 → D = 10 :=
by
  intros h
  rw [h.1, h.2.1, h.2.2]
  sorry

end mohit_discount_l552_552228


namespace pizza_remained_l552_552954

noncomputable def number_of_people := 15
noncomputable def fraction_eating_pizza := 3 / 5
noncomputable def total_pizza_pieces := 50
noncomputable def pieces_per_person := 4
noncomputable def pizza_remaining := total_pizza_pieces - (pieces_per_person * (fraction_eating_pizza * number_of_people))

theorem pizza_remained :
  pizza_remaining = 14 :=
by {
  sorry
}

end pizza_remained_l552_552954


namespace TE_eq_TF_l552_552756

variables {A B C D E F K L T : Type} 
  (ABCD_convex : convex_quadrilateral A B C D)
  (E_on_AB : on_segment E A B)
  (F_on_CD : on_segment F C D)
  (AE_eq_BE : dist A E = dist B E)
  (CF_eq_DF : dist C F = dist D F)
  (EF_eq : dist E F = dist E F)
  (K_intersection : intersection_of_diagonals K B C F E)
  (L_intersection : intersection_of_diagonals L A D F E)
  (T_intersection : intersecting_perpendiculars T A D K B C L)

theorem TE_eq_TF : dist T E = dist T F :=
sorry

end TE_eq_TF_l552_552756


namespace general_term_sequence_l552_552068

noncomputable def a (n : ℕ) : ℝ := match n with
| 0 => 0 -- defined to avoid non-exhaustive match (sequence is supposed to start from 1)
| n+1 => (n + 1) / 4

theorem general_term_sequence (n : ℕ) (pos_terms: ∀ n, 0 < a (n + 1))
  (h1 : a 1 = 1 / 4)
  (h2 : ∀ n, (∑ i in Finset.range (n + 1), a (i + 1)) = 2 * a (n + 1) * a (n + 2)) :
  a (n + 1) = (n + 1) / 4 :=
by
  sorry

end general_term_sequence_l552_552068


namespace no_intersection_of_circles_l552_552624

-- Definitions for the conditions
def acute_triangle (ABC : Triangle) : Prop := 
    acute_angle ABC.A ∧ acute_angle ABC.B ∧ acute_angle ABC.C

def circumcircle (ABC : Triangle) : Circle := 
    Circle.circum ABC

def nine_point_circle (ABC : Triangle) : Circle := 
    Circle.nine_point_circle ABC

-- The main theorem statement
theorem no_intersection_of_circles (ABC : Triangle) 
    (h1 : acute_triangle ABC)
    (C := circumcircle ABC)
    (E := nine_point_circle ABC) :
    C ∩ E = ∅ :=
sorry

end no_intersection_of_circles_l552_552624


namespace equivalent_single_discount_l552_552653

-- Define the successive discounts
def discount1 := 0.10
def discount2 := 0.20
def discount3 := 0.05

-- Calculate the combined discount
def combined_discount (d1 d2 d3 : ℝ) : ℝ :=
  (1 - d3) * ((1 - d2) * (1 - d1))

-- State the problem
theorem equivalent_single_discount :
  combined_discount discount1 discount2 discount3 = 1 - 0.684 :=
by
  -- calculate the combined discount explicitly: 
  -- (1 - discount3) * (1 - discount2) * (1 - discount1) = 0.684
  sorry

end equivalent_single_discount_l552_552653


namespace find_ratio_l552_552987

-- Define the problem conditions
variables (E F G H A B C D E1 F1 G1 H1 : Point)
variable (λ : ℝ)
variable [convex_quadrilaterals : ∀ {x y z t : Point}, convex_quadrilaterals x y z t ↔ ∃ (w : Point), line_segment w x ∧ line_segment w y ∧ line_segment w z ∧ line_segment w t]

-- Assuming the conditions
axiom E_on_AB : E ∈ line_segment A B
axiom F_on_BC : F ∈ line_segment B C
axiom G_on_CD : G ∈ line_segment C D
axiom H_on_DA : H ∈ line_segment D A

axiom ratio_condition : (AE E B) * (BF F C) * (CG G D) * (DH H A) = 1

axiom A_on_H1E1 : A ∈ line_segment H1 E1
axiom B_on_E1F1 : B ∈ line_segment E1 F1
axiom C_on_F1G1 : C ∈ line_segment F1 G1
axiom D_on_G1H1 : D ∈ line_segment G1 H1

axiom parallel_conditions :
  parallel E1 F1 EF ∧
  parallel F1 G1 FG ∧
  parallel G1 H1 GH ∧
  parallel H1 E1 HE

axiom given_ratio : (E1A E1 A) / (AH1 A H1) = λ

-- The proof goal
theorem find_ratio : (F1C F1 C) / (CG1 C G1) = λ :=
sorry

end find_ratio_l552_552987


namespace sum_of_x_and_y_l552_552690

theorem sum_of_x_and_y 
  (x y : ℝ)
  (h : ((x + 1) + (y-1)) / 2 = 10) : x + y = 20 :=
sorry

end sum_of_x_and_y_l552_552690


namespace jake_watching_fraction_l552_552980

-- Define the conditions
def hours_mon : ℕ := 12
def hours_tue : ℕ := 4
def hours_wed (x : ℝ) : ℝ := 24 * x
def hours_thu (x : ℝ) : ℝ := (hours_mon + hours_tue + hours_wed x) / 2
def hours_fri : ℕ := 19
def total_hours : ℕ := 52
def mon_to_thu_hours (x : ℝ) : ℝ := hours_mon + hours_tue + hours_wed x + hours_thu x

-- Lean statement to prove
theorem jake_watching_fraction :
  ∃ x : ℝ, 0 ≤ x ∧ total_hours - hours_fri = mon_to_thu_hours x ∧ x = 1 / 4 :=
by
  sorry

end jake_watching_fraction_l552_552980


namespace sin_value_l552_552094

theorem sin_value (α : ℝ) (h : Real.cos (π / 6 - α) = (Real.sqrt 3) / 3) :
    Real.sin (5 * π / 6 - 2 * α) = -1 / 3 :=
by
  sorry

end sin_value_l552_552094


namespace lcm_45_75_180_is_900_l552_552831

def prime_factorization_45 : Multiset (ℕ × ℕ) := {(3, 2), (5, 1)}
def prime_factorization_75 : Multiset (ℕ × ℕ) := {(3, 1), (5, 2)}
def prime_factorization_180 : Multiset (ℕ × ℕ) := {(2, 2), (3, 2), (5, 1)}

theorem lcm_45_75_180_is_900 :
  (let lcm := 2^2 * 3^2 * 5^2 in lcm = 900) :=
  sorry

end lcm_45_75_180_is_900_l552_552831


namespace triangle_inscribed_lengths_l552_552162

theorem triangle_inscribed_lengths (XY YZ XZ ZO : ℕ) 
  (h1 : XY = 26) (h2 : YZ = 28) (h3 : XZ = 27) 
  (arc_cond1 : ∀ M O, ∃ k, XM = NO = k)
  (arc_cond2 : ∀ M O, ∃ j, MO = YN = j)
  (arc_cond3 : ∀ N M, ∃ l, NM = ZO = l):
  ZO = 14 :=
by 
  sorry

end triangle_inscribed_lengths_l552_552162


namespace bill_toilet_paper_usage_l552_552427

-- Conditions as definitions
def rolls : ℕ := 1000
def squares_per_roll : ℕ := 300
def days : ℕ := 20000
def times_per_day : ℕ := 3

-- Total number of squares of toilet paper
def total_squares : ℕ := rolls * squares_per_roll

-- Squares used per day
def squares_per_day : ℕ := total_squares / days

-- Squares used each time
def squares_per_time : ℕ := squares_per_day / times_per_day

-- Mathematical proof in Lean
theorem bill_toilet_paper_usage : 
  squares_per_time = 5 :=
by 
  -- Definitions provided from conditions
  have h_rolls : rolls = 1000 := rfl
  have h_squares_per_roll : squares_per_roll = 300 := rfl
  have h_days : days = 20000 := rfl
  have h_times_per_day : times_per_day = 3 := rfl

  -- Calculate total_squares, squares_per_day, squares_per_time
  unfold total_squares squares_per_day squares_per_time
  
  -- Fill in the proof or use sorry to denote that proof is not required here
  sorry

end bill_toilet_paper_usage_l552_552427


namespace sphere_surface_area_l552_552538

noncomputable def radiusCircumscribedCircleABC (AB BC CA : ℝ) : ℝ := 
  if h : AB = BC ∧ BC = CA ∧ AB = 2 then (2 * Real.sqrt 3) / 3 else 0

theorem sphere_surface_area (R : ℝ) (h : R^2 - (1 / 2 * R)^2 = 4 / 3) : 4 * π * R^2 = 64 * π / 9 := 
by
  sorry

def surface_area := 
  let R := radiusCircumscribedCircleABC 2 2 2
  if h : R^2 - (1 / 2 * R)^2 = 4 / 3 then 4 * π * R^2 else 0

example : surface_area = 64 * π / 9 :=
by
  sorry

end sphere_surface_area_l552_552538


namespace num_natural_numbers_divisors_count_l552_552480

theorem num_natural_numbers_divisors_count:
  ∃ N : ℕ, (2017 % N = 17 ∧ N ∣ 2000) ↔ 13 := 
sorry

end num_natural_numbers_divisors_count_l552_552480


namespace candle_height_relation_l552_552334

theorem candle_height_relation : 
  ∀ (h : ℝ) (t : ℝ), h = 1 → (∀ (h1_burn_rate : ℝ), h1_burn_rate = 1 / 5) → (∀ (h2_burn_rate : ℝ), h2_burn_rate = 1 / 6) →
  (1 - t * 1 / 5 = 3 * (1 - t * 1 / 6)) → t = 20 / 3 :=
by
  intros h t h_init h1_burn_rate h2_burn_rate height_eq
  sorry

end candle_height_relation_l552_552334


namespace intersection_M_N_l552_552916

def M (x : ℝ) : Prop := ∃ y : ℝ, y = log x / log 10
def N (x : ℝ) : Prop := ∃ y : ℝ, y = sqrt (1 - x)

theorem intersection_M_N : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 0 < x ∧ x ≤ 1} := 
sorry

end intersection_M_N_l552_552916


namespace max_vector_sum_l552_552103

theorem max_vector_sum
  (A B C : ℝ × ℝ)
  (P : ℝ × ℝ := (2, 0))
  (hA : A.1^2 + A.2^2 = 1)
  (hB : B.1^2 + B.2^2 = 1)
  (hC : C.1^2 + C.2^2 = 1)
  (h_perpendicular : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) :
  |(2,0) - A + (2,0) - B + (2,0) - C| = 7 := sorry

end max_vector_sum_l552_552103


namespace harmonic_series_expansion_l552_552608

open Classical
open BigOperators

noncomputable def harmonic_series (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), (1 : ℝ) / k

def f (n : ℕ) : ℝ := harmonic_series n

theorem harmonic_series_expansion :
  ∃ γ c d : ℝ, ∀ n : ℕ,
  f(n) = log n + γ + c / n + d / n^2 + 
    (O (λ (n : ℕ), 1 / n ^ 3)) :=
sorry

end harmonic_series_expansion_l552_552608


namespace man_work_days_l552_552777

theorem man_work_days :
  ∃ M : ℕ, (∀ M, ((1 : ℚ) / M + 1 / 6 = 1 / 3) -> M = 6) := by
  sorry

end man_work_days_l552_552777


namespace correct_time_for_J_l552_552369

variable (y : ℝ) 
variable (h1 : ∀ L, J takes 45/60 (3/4 hour) less time than L to travel a distance of 45 miles)
variable (h2 : J travels (1/2) mph faster than L)

/- Defining L's speed based on y -/
def speed_L := y - 1/2

/- Defining J's travel time -/
def time_J := 45 / y

/- Defining L's travel time -/
def time_L := 45 / speed_L

/- Restating the condition for the time difference -/
axiom time_diff : time_L - time_J = 3 / 4

/- Proving the correct time for J given the conditions -/
theorem correct_time_for_J : time_J = 45 / y := by
  sorry

end correct_time_for_J_l552_552369


namespace general_formula_a_general_formula_c_l552_552084

-- Definition of the sequence {a_n}
def S (n : ℕ) : ℕ := n^2 + 2 * n
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem general_formula_a (n : ℕ) (hn : n > 0) : a n = 2 * n + 1 := sorry

-- Definitions for the second problem
def f (x : ℝ) : ℝ := x^2 + 2 * x
def f' (x : ℝ) : ℝ := 2 * x + 2
def k (n : ℕ) : ℝ := 2 * n + 2

def Q (k : ℝ) : Prop := ∃ (n : ℕ), k = 2 * n + 2
def R (k : ℝ) : Prop := ∃ (n : ℕ), k = 4 * n + 2

def c (n : ℕ) : ℕ := 12 * n - 6

theorem general_formula_c (n : ℕ) (hn1 : 0 < c 10)
    (hn2 : c 10 < 115) : c n = 12 * n - 6 := sorry

end general_formula_a_general_formula_c_l552_552084


namespace total_students_l552_552820

theorem total_students (x : ℕ) (h1 : (x + 6) / (2*x + 6) = 2 / 3) : 2 * x + 6 = 18 :=
sorry

end total_students_l552_552820


namespace determinant_solution_l552_552865

theorem determinant_solution (b y : ℝ) (hb : b ≠ 0) : 
  (Matrix.det ![![y + 2 * b, y, y], ![y, y + 2 * b, y], ![y, y, y + 2 * b]] = 0) ↔ (y = -b ∨ y = 2b) :=
by
  sorry

end determinant_solution_l552_552865


namespace johns_tax_rate_is_30_percent_l552_552198

variable (IncomeJohn : ℕ) (IncomeIngrid : ℕ) (RateIngrid : ℝ) (RateCombined : ℝ)

def totalIncome : ℝ := (IncomeJohn + IncomeIngrid : ℝ)
def totalTax : ℝ := RateCombined * totalIncome
def IngridTax : ℝ := RateIngrid * (IncomeIngrid : ℝ)
def JohnTax : ℝ := totalTax - IngridTax
def JohnTaxRate : ℝ := JohnTax / (IncomeJohn : ℝ)

theorem johns_tax_rate_is_30_percent 
  (h1 : IncomeJohn = 56000) 
  (h2 : IncomeIngrid = 74000)
  (h3 : RateIngrid = 0.4) 
  (h4 : RateCombined = 0.3569) :
  JohnTaxRate IncomeJohn IncomeIngrid RateIngrid RateCombined = 0.3 := 
by 
  sorry

end johns_tax_rate_is_30_percent_l552_552198


namespace monotonic_intervals_and_maximum_find_m_l552_552550

noncomputable def e : ℝ := Real.exp 1

def f (x : ℝ) : ℝ := x / (e^(2 * x))

def g (x : ℝ) (m : ℝ) : ℝ := x / (e^(2 * x)) + m

theorem monotonic_intervals_and_maximum :
  (∀ x : ℝ, x < (1 / 2) → f' x > 0) ∧
  (∀ x : ℝ, x > (1 / 2) → f' x < 0) ∧
  f (1 / 2) = 1 / (2 * e) :=
sorry

theorem find_m (m : ℝ) :
  (g' (-1 / 2) = 2 * e) →
  (∀ (m : ℝ), LinePassingThroughTangent g (-1 / 2) m (1, 3 * e)) → 
  m = e / 2 :=
sorry

end monotonic_intervals_and_maximum_find_m_l552_552550


namespace bisect_BI_PQ_l552_552420

variables {A B C D E F G I K M N P Q : Type*}
variables [incircle_touches (triangle A B C) I D E F]
variables [line_intersect_line_at (BI) (EF) M]
variables [line_intersect_line_at (CI) (EF) N]
variables [line_intersect_line_at (DI) (EF) K]
variables [line_intersect_line_at (BN) (CM) P]
variables [line_intersect_line_at (AK) (BC) G]
variables [perpendicular_intersection (line_through I) (line_perpendicular_to PG) Q]
variables [perpendicular_intersection (line_through P) (line_perpendicular_to PB) Q]

theorem bisect_BI_PQ : bisects BI PQ :=
sorry

end bisect_BI_PQ_l552_552420


namespace ellipse_focus_intersection_l552_552901

theorem ellipse_focus_intersection :
  let F_1 := (-Real.sqrt 3, 0)
  let A := ∃ x y : ℝ, (x^2 / 4 + y^2 = 1) ∧ (y = x + Real.sqrt 3)
  let B := ∃ x y : ℝ, (x^2 / 4 + y^2 = 1) ∧ (y = x + Real.sqrt 3) ∧ (x, y) ≠ A
  let d_F1A := Real.sqrt ((A.1 - F_1.1)^2 + (A.2 - F_1.2)^2)
  let d_F1B := Real.sqrt ((B.1 - F_1.1)^2 + (B.2 - F_1.2)^2)
  let result := 1 / d_F1A + 1 / d_F1B
  result = 4 := sorry

end ellipse_focus_intersection_l552_552901


namespace inequality_solution_set_range_of_a_l552_552908

noncomputable def f (x a : ℝ) := x^2 - (a + 2) * x + 4

theorem inequality_solution_set (a x : ℝ) :
  (f x a ≤ -2 * a + 4) ↔ 
  ((a < 2 → a ≤ x ∧ x ≤ 2) ∧ 
   (a = 2 → x = 2) ∧ 
   (a > 2 → 2 ≤ x ∧ x ≤ a)) := sorry

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ set.Icc (1:ℝ) (4:ℝ), f x a + a + 1 ≥ 0) → a ≤ 4 := sorry

end inequality_solution_set_range_of_a_l552_552908


namespace strawberries_left_l552_552440

theorem strawberries_left (s b t : ℕ) (h_s : s = 300) (h_b : b = 5) (h_t : t = 20) :
  (s / b - t) = 40 :=
by
  rw [h_s, h_b, h_t]
  norm_num
  sorry

end strawberries_left_l552_552440


namespace number_of_students_in_sleeper_coach_l552_552460

-- Definitions from the problem conditions
def total_passengers : ℕ := 300
def percent_students : ℝ := 80 / 100
def percent_sleeper : ℝ := 15 / 100

-- The mathematical proof goal
theorem number_of_students_in_sleeper_coach :
  (total_passengers * percent_students * percent_sleeper).toInt = 36 :=
by
  sorry

end number_of_students_in_sleeper_coach_l552_552460


namespace increasing_f_x3_sub_sqrt3_div_2_x6_l552_552759

open Function

variables {f : ℝ → ℝ}

def increasing_on (s : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

theorem increasing_f_x3_sub_sqrt3_div_2_x6 (h1 : increasing_on (Set.Ioi 0) (λ x, f x - x))
  (h2 : increasing_on (Set.Ioi 0) (λ x, f (x ^ 2) - x ^ 6)) :
  increasing_on (Set.Ioi 0) (λ x, f (x ^ 3) - (Real.sqrt 3 / 2) * x ^ 6) :=
sorry

end increasing_f_x3_sub_sqrt3_div_2_x6_l552_552759


namespace num_natural_numbers_divisors_count_l552_552482

theorem num_natural_numbers_divisors_count:
  ∃ N : ℕ, (2017 % N = 17 ∧ N ∣ 2000) ↔ 13 := 
sorry

end num_natural_numbers_divisors_count_l552_552482


namespace incorrect_statement_l552_552207

open Set

theorem incorrect_statement 
  (M : Set ℝ := {x : ℝ | 0 < x ∧ x < 1})
  (N : Set ℝ := {y : ℝ | 0 < y})
  (R : Set ℝ := univ) : M ∪ N ≠ R :=
by
  sorry

end incorrect_statement_l552_552207


namespace simplify_expression_l552_552661

theorem simplify_expression :
  (Complex.mk (-1) (Real.sqrt 3) / 2) ^ 12 + (Complex.mk (-1) (-Real.sqrt 3) / 2) ^ 12 = 2 := by
  sorry

end simplify_expression_l552_552661


namespace fraction_of_menu_items_my_friend_can_eat_l552_552633

theorem fraction_of_menu_items_my_friend_can_eat {menu_size vegan_dishes nut_free_vegan_dishes : ℕ}
    (h1 : vegan_dishes = 6)
    (h2 : vegan_dishes = menu_size / 6)
    (h3 : nut_free_vegan_dishes = vegan_dishes - 5) :
    (nut_free_vegan_dishes : ℚ) / menu_size = 1 / 36 :=
by
  sorry

end fraction_of_menu_items_my_friend_can_eat_l552_552633


namespace michelle_drives_294_miles_l552_552333

theorem michelle_drives_294_miles
  (total_distance : ℕ)
  (michelle_drives : ℕ)
  (katie_drives : ℕ)
  (tracy_drives : ℕ)
  (h1 : total_distance = 1000)
  (h2 : michelle_drives = 3 * katie_drives)
  (h3 : tracy_drives = 2 * michelle_drives + 20)
  (h4 : katie_drives + michelle_drives + tracy_drives = total_distance) :
  michelle_drives = 294 := by
  sorry

end michelle_drives_294_miles_l552_552333


namespace Brenda_bakes_cakes_l552_552020

theorem Brenda_bakes_cakes 
  (cakes_per_day : ℕ)
  (days : ℕ)
  (sell_fraction : ℚ)
  (total_cakes_baked : ℕ := cakes_per_day * days)
  (cakes_left : ℚ := total_cakes_baked * sell_fraction)
  (h1 : cakes_per_day = 20)
  (h2 : days = 9)
  (h3 : sell_fraction = 1 / 2) :
  cakes_left = 90 := 
by 
  -- Proof to be filled in later
  sorry

end Brenda_bakes_cakes_l552_552020


namespace time_for_2km_l552_552291

def distance_over_time (t : ℕ) : ℝ := 
  sorry -- Function representing the distance walked over time

theorem time_for_2km : ∃ t : ℕ, distance_over_time t = 2 ∧ t = 105 :=
by
  sorry

end time_for_2km_l552_552291


namespace total_animals_l552_552727

theorem total_animals (ducks dogs : ℕ) (hd : ducks = 6) (hl : dogs * 4 + ducks * 2 = 32) : ducks + dogs = 11 :=
by
  rw hd at hl
  sorry

end total_animals_l552_552727


namespace simplify_cube_root_l552_552271

theorem simplify_cube_root (a b c d : ℕ) (h₁ : a = 20) (h₂ : b = 30) (h₃ : c = 40) (h₄ : d = 60) :
  (∛(a^3 + b^3 + c^3 + d^3)) = 10 * ∛315 := by
  sorry

end simplify_cube_root_l552_552271


namespace smallest_multiple_of_29_with_properties_l552_552744

theorem smallest_multiple_of_29_with_properties 
  (N : ℕ)
  (h1 : N % 100 = 29) 
  (h2 : N % 29 = 0) 
  (h3 : sum_digits N = 29) :
  N = 783 :=
sorry

end smallest_multiple_of_29_with_properties_l552_552744


namespace range_of_m_l552_552579

theorem range_of_m (m : ℝ) : 
  (∀ x, -∞ < x ∧ x ≤ 1 → (x^2 - 2 * m * x + 1) ≤ (m^2 - 2 * m * m + 1)) -> (∃ b : ℝ, b = 1 ∧ ∀ x : ℝ, m ∈ [b, +∞)) := 
begin
  sorry
end

end range_of_m_l552_552579


namespace reporters_not_covering_politics_l552_552641

theorem reporters_not_covering_politics (P_X P_Y P_Z intlPol otherPol econOthers : ℝ)
  (h1 : P_X = 0.15) (h2 : P_Y = 0.10) (h3 : P_Z = 0.08)
  (h4 : otherPol = 0.50) (h5 : intlPol = 0.05) (h6 : econOthers = 0.02) :
  (1 - (P_X + P_Y + P_Z + intlPol + otherPol + econOthers)) = 0.10 := by
  sorry

end reporters_not_covering_politics_l552_552641


namespace number_of_subsets_of_M_l552_552300

def M : Set ℝ := { x | x^2 - 2 * x + 1 = 0 }

theorem number_of_subsets_of_M : M = {1} → ∃ n, n = 2 := by
  sorry

end number_of_subsets_of_M_l552_552300


namespace binomial_expansion_coef_x4_l552_552042

theorem binomial_expansion_coef_x4 :
  let general_term (n k : ℕ) (a b x : ℕ) := binomial n k * (a^k) * (b^(n - k)) * (x^(n - k))
  let expr := (x^3 - (sqrt (2:ℝ) / (sqrt (x:ℝ))))^6 
  coeff_of_x4 (x : ℕ) := 
  (3:ℕ) := 18 - 4 
  240  := 4  :=
  240 = coeff_of_x4(expr, 4).sorry

end binomial_expansion_coef_x4_l552_552042


namespace victor_percentage_of_marks_l552_552338

theorem victor_percentage_of_marks (marks_obtained : ℝ) (maximum_marks : ℝ) (h1 : marks_obtained = 285) (h2 : maximum_marks = 300) : 
  (marks_obtained / maximum_marks) * 100 = 95 :=
by
  sorry

end victor_percentage_of_marks_l552_552338


namespace increased_time_between_maintenance_checks_l552_552377

theorem increased_time_between_maintenance_checks (original_time : ℕ) (percentage_increase : ℕ) : 
  original_time = 20 → percentage_increase = 25 →
  original_time + (original_time * percentage_increase / 100) = 25 :=
by
  intros
  sorry

end increased_time_between_maintenance_checks_l552_552377


namespace locus_of_points_C1_is_line_l552_552986

noncomputable def locus_of_points_C1 (a c : ℝ) (h_ne : a ≠ c) : set (ℝ × ℝ) :=
{ p : ℝ × ℝ | ∃ d : ℝ, p = (c, d) ∧ d = 1 + a^2 }

theorem locus_of_points_C1_is_line (a c : ℝ) (h_ne : a ≠ c) :
  locus_of_points_C1 a c h_ne = { p : ℝ × ℝ | ∃ x : ℝ, p = (x, 1 + a^2) } :=
by
  sorry

end locus_of_points_C1_is_line_l552_552986


namespace first_15_prime_sums_count_zero_l552_552306

/-- A function to check if a number is prime -/
def is_prime (n : ℕ) : Prop := nat.prime n

/-- Defining the first 15 sums of consecutive primes starting from 3 -/
def prime_sums : ℕ → ℕ
| 1     := 3
| (n+1) := prime_sums n + nat.prime (n+1)

noncomputable def count_prime_sums_up_to_15 : ℕ :=
(finset.range 15).count (λ n, is_prime (prime_sums (n+1)))

theorem first_15_prime_sums_count_zero : count_prime_sums_up_to_15 = 0 :=
by sorry

end first_15_prime_sums_count_zero_l552_552306


namespace count_prime_sums_is_two_l552_552310

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def prime_numbers : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def prime_sums : List ℕ := 
  let sums := foldl (λ acc p, acc ++ [acc.head! + p]) [3] prime_numbers
  sums.take 15

def count_primes (l : List ℕ) : ℕ :=
  l.countp is_prime

theorem count_prime_sums_is_two : count_primes prime_sums = 2 := 
  by
    sorry

end count_prime_sums_is_two_l552_552310


namespace seats_per_row_and_total_students_l552_552576

theorem seats_per_row_and_total_students (R S : ℕ) 
  (h1 : S = 5 * R + 6) 
  (h2 : S = 12 * (R - 3)) : 
  R = 6 ∧ S = 36 := 
by 
  sorry

end seats_per_row_and_total_students_l552_552576


namespace ratio_6_3_to_percent_l552_552362

theorem ratio_6_3_to_percent : (6 / 3) * 100 = 200 := by
  sorry

end ratio_6_3_to_percent_l552_552362


namespace find_f_2011_l552_552392

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_2011 :
  (∀ x : ℝ, f (x^2 + x) + 2 * f (x^2 - 3 * x + 2) = 9 * x^2 - 15 * x) →
  f 2011 = 6029 :=
by
  intros hf
  sorry

end find_f_2011_l552_552392


namespace coefficient_x_squared_term_l552_552828

-- Define the two polynomials as given in the conditions
def polynomial1 := 2 * (X^2) + 3 * X + 4
def polynomial2 := 5 * (X^2) + 6 * X + 7

-- Define a statement to prove that the coefficient of x^2 in their product is 52
theorem coefficient_x_squared_term :
  (polynomial1 * polynomial2).coeff 2 = 52 := sorry

end coefficient_x_squared_term_l552_552828


namespace rate_per_sq_meter_l552_552706

variables (L W C : Real)
def question := (R : Real) := C / (L * W)
hypothesis length_def : L = 10
hypothesis width_def : W = 4.75
hypothesis cost_def : C = 42750
theorem rate_per_sq_meter :
  R = 900 :=
by
  rw [length_def, width_def, cost_def]
  norm_num
  sorry

end rate_per_sq_meter_l552_552706


namespace scientific_notation_correct_l552_552004

def number := 56990000

theorem scientific_notation_correct : number = 5.699 * 10^7 :=
  by
    sorry

end scientific_notation_correct_l552_552004


namespace maximum_area_triangle_OAB_l552_552965

open Real

def parametric_line_C1 (t α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ × ℝ :=
  (t * cos α, t * sin α)

def intersection_A (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ × ℝ :=
  (8 * sin α, α)

def intersection_B (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ × ℝ :=
  (8 * cos α, α + π / 2)

def area_triangle_OAB (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ :=
  32 * sin α * cos α

theorem maximum_area_triangle_OAB (α : ℝ) (hα : 0 < α ∧ α < π / 2) : 
  ∃ α_max, 0 < α_max ∧ α_max < π / 2 ∧ area_triangle_OAB α_max (and.intro (by linarith) (by linarith)) = 16 :=
by
  use π / 4
  split
  { linarith [Real.pi_pos] }
  split
  { linarith [Real.pi_pos] }
  simp [area_triangle_OAB, sin_double, mul_comm, mul_assoc, mul_left_comm]
  rw sin_pi_div_two
  norm_num
  exact Real.sin_pos_of_pos_lt_two_pi (by linarith) (by linarith)

end maximum_area_triangle_OAB_l552_552965


namespace perimeter_is_140_l552_552705

-- Definitions for conditions
def width (w : ℝ) := w
def length (w : ℝ) := width w + 10
def perimeter (w : ℝ) := 2 * (length w + width w)

-- Cost condition
def cost_condition (w : ℝ) : Prop := (perimeter w) * 6.5 = 910

-- Proving that if cost_condition holds, the perimeter is 140
theorem perimeter_is_140 (w : ℝ) (h : cost_condition w) : perimeter w = 140 :=
by sorry

end perimeter_is_140_l552_552705


namespace exists_polynomial_satisfying_properties_l552_552757

theorem exists_polynomial_satisfying_properties :
  ∀ (n : ℕ), n ≥ 4 →
  ∃ (f : ℕ → ℤ), (∃ (a : ℕ → ℤ), ∀ i, 0 ≤ i ∧ i < n → a i > 0 ∧ f = λ x, (finset.range n.succ).sum (λ i, a i * x ^ i)) ∧
  ∀ (m : ℕ) (k : ℕ) (r : fin k.succ → ℕ),
    k ≥ 2 ∧ (∀ i j, i ≠ j → r i ≠ r j) →
    f m ≠ (finset.univ : finset (fin k.succ)).prod (λ i, f (r i)) :=
by
  sorry

end exists_polynomial_satisfying_properties_l552_552757


namespace stickers_count_l552_552231

theorem stickers_count (initial_stickers : ℝ) (given_away : ℝ) (final_stickers : ℝ) 
  (h1 : initial_stickers = 39.0) 
  (h2 : given_away = 22.0)
  (h3 : final_stickers = initial_stickers - given_away) : 
  final_stickers = 17.0 := 
by
  rw [h1, h2] at h3
  exact h3


end stickers_count_l552_552231


namespace sin_value_l552_552093

theorem sin_value (α : ℝ) (h : Real.cos (π / 6 - α) = (Real.sqrt 3) / 3) :
    Real.sin (5 * π / 6 - 2 * α) = -1 / 3 :=
by
  sorry

end sin_value_l552_552093


namespace parallel_vectors_x_value_l552_552098

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

-- Define the condition that vectors are parallel
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- State the problem: if a and b are parallel, then x = 1/2
theorem parallel_vectors_x_value (x : ℝ) (h : is_parallel a (b x)) : x = 1/2 :=
by
  sorry

end parallel_vectors_x_value_l552_552098


namespace number_of_possible_sets_l552_552880

theorem number_of_possible_sets (A : Set ℤ) :
  (A ∩ {-1, 0, 1} = {0, 1}) →
  (A ∪ {-2, 0, 2} = {-2, 0, 1, 2}) →
  (∃ n, n = 4 ∧ (n = ∥{ B : Set ℤ | B ∩ {-1, 0, 1} = {0, 1} ∧ B ∪ {-2, 0, 2} = {-2, 0, 1, 2} }∥))
  := by
  intro h1 h2
  sorry

end number_of_possible_sets_l552_552880


namespace cos_inequality_for_triangle_l552_552974

theorem cos_inequality_for_triangle (A B C : ℝ) (h : A + B + C = π) :
  (1 / 3) * (Real.cos A + Real.cos B + Real.cos C) ≤ (1 / 2) ∧
  (1 / 2) ≤ Real.sqrt ((1 / 3) * (Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2)) :=
by
  sorry

end cos_inequality_for_triangle_l552_552974


namespace binary_rep_of_21_l552_552035

theorem binary_rep_of_21 : 
  (Nat.digits 2 21) = [1, 0, 1, 0, 1] := 
by 
  sorry

end binary_rep_of_21_l552_552035


namespace correct_statements_l552_552549

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + 1

theorem correct_statements (a : ℝ) (h_pos : a > 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) ∧
  (let x := -a / 3 in let y := f a x in is_center_of_symmetry (f a) x y) ∧
  (∃ x0 : ℝ, deriv (f a) x0 = 1 ∧ f a x0 = x0 → (f a x0 - x0) + 1 = 0) :=
sorry

end correct_statements_l552_552549


namespace max_in_interval_l552_552129

noncomputable def f (x φ : ℝ) : ℝ := cos (2 * x - φ)

theorem max_in_interval
  (φ : ℝ)
  (hφ1 : -π < φ)
  (hφ2 : φ < 0)
  (hodd : ∀ x, cos (2 * (x + π / 6) + π / 3 - φ) = -cos (2 * (x + π / 6) + π / 3 - φ)) :
  ∃ x ∈ Ioo (-π/6) (π/3), f x φ = 1 :=
sorry

end max_in_interval_l552_552129


namespace minimum_n_exists_l552_552454

-- Define the essential components from the conditions
def polynomial_with_integer_coefficients (P : ℤ → ℤ) : Prop :=
  ∀ x, ∃ a b : ℤ, P(x) = a * x + b  -- It's simplified for the purpose of this example; actual coefficient constraints are more involved.

-- Define the main statement
theorem minimum_n_exists :
  ∃ (n : ℕ), n = 4 ∧
    (∀ (P : ℤ → ℤ),
      polynomial_with_integer_coefficients P →
      (∃ x1 x2 x3 x4 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
                          P x1 = 2 ∧ P x2 = 2 ∧ P x3 = 2 ∧ P x4 = 2) →
      (∀ x : ℤ, P x ≠ 4)) :=
sorry

end minimum_n_exists_l552_552454


namespace min_value_fx_range_a_seq_inequality_l552_552123

open Real

-- Definitions corresponding to the conditions
def f (x : ℝ) (a : ℝ) : ℝ := ln x + 1/x + a * x

-- Proof problem 1
theorem min_value_fx (a : ℝ) (h1 : ∃ (x : ℝ), x > 0 ∧ f x a = 1) :
  ∃ x, x > 0 ∧ f x a = 1 → f 1 a = 1 :=
by
  sorry

-- Proof problem 2
theorem range_a (h2 : ∃ x ∈ Ioo 2 3, deriv (f x a) = 0) :
  a ∈ Ioo (-1 / 4) (-2 / 9) :=
by
  sorry

-- Proof problem 3
theorem seq_inequality (x : ℕ → ℝ) (hx : ∀ n, ln (x n) + 1 / (x (n + 1)) < 1) :
  x 1 ≤ 1 :=
by
  sorry

end min_value_fx_range_a_seq_inequality_l552_552123


namespace age_difference_19_l552_552302

-- Declaring variables for ages X, Y, Z
variables {X Y Z : ℕ}

-- Defining the conditions
def cond1 : Prop := X + Y > Y + Z
def cond2 : Prop := Z = X - 19

-- Stating the theorem to prove that the difference is 19
theorem age_difference_19 (h1 : cond1) (h2 : cond2) : (X + Y) - (Y + Z) = 19 := 
sorry

end age_difference_19_l552_552302


namespace train_speed_is_correct_l552_552406

/-- Define the length of the train (in meters) -/
def train_length : ℕ := 120

/-- Define the length of the bridge (in meters) -/
def bridge_length : ℕ := 255

/-- Define the time to cross the bridge (in seconds) -/
def time_to_cross : ℕ := 30

/-- Define the total distance covered by the train while crossing the bridge -/
def total_distance : ℕ := train_length + bridge_length

/-- Define the speed of the train in meters per second -/
def speed_m_per_s : ℚ := total_distance / time_to_cross

/-- Conversion factor from m/s to km/hr -/
def m_per_s_to_km_per_hr : ℚ := 3.6

/-- The expected speed of the train in km/hr -/
def expected_speed_km_per_hr : ℕ := 45

/-- The theorem stating that the speed of the train is 45 km/hr -/
theorem train_speed_is_correct :
  (speed_m_per_s * m_per_s_to_km_per_hr) = expected_speed_km_per_hr := by
  sorry

end train_speed_is_correct_l552_552406


namespace dj_snake_engagement_treats_l552_552452

-- We will define the conditions as hypotheses and the target proof 

theorem dj_snake_engagement_treats :
  ∃ (C : ℤ), (let hotel_cost := 2 * 4000 in
               let house_value := 4 * C in
               let total_value := hotel_cost + C + house_value in
               total_value = 158000) ∧ C = 30000 :=
by
  use 30000
  -- hotel_cost = 2 * 4000
  -- house_value = 4 * C
  -- total_value = hotel_cost + C + house_value
  -- 8000 + C + 4 * C = 158000
  -- 8000 + 5 * C = 158000
  -- 5 * C = 150000
  -- C = 150000 / 5
  -- C = 30000
  sorry

end dj_snake_engagement_treats_l552_552452


namespace exists_a_l552_552154

noncomputable def a : ℕ → ℕ := sorry

theorem exists_a : a (a (a (a 1))) = 458329 :=
by
  -- proof skipped
  sorry

end exists_a_l552_552154


namespace pyramid_volume_inequality_l552_552315

theorem pyramid_volume_inequality
  (k : ℝ)
  (OA1 OB1 OC1 OA2 OB2 OC2 OA3 OB3 OC3 OB2 : ℝ)
  (V1 := k * |OA1| * |OB1| * |OC1|)
  (V2 := k * |OA2| * |OB2| * |OC2|)
  (V3 := k * |OA3| * |OB3| * |OC3|)
  (V := k * |OA1| * |OB2| * |OC3|) :
  V ≤ (V1 + V2 + V3) / 3 := 
  sorry

end pyramid_volume_inequality_l552_552315


namespace calculate_insurance_cost_l552_552686

def total_cost_apartment : ℝ := 7000000
def loan_amount : ℝ := 4000000
def loan_interest_rate : ℝ := 0.101
def property_insurance_tariff : ℝ := 0.0009
def life_health_insurance_female : ℝ := 0.0017
def life_health_insurance_male : ℝ := 0.0019
def title_insurance_tariff : ℝ := 0.0027
def svetlana_share : ℝ := 0.2
def dmitry_share : ℝ := 0.8

theorem calculate_insurance_cost :
  let total_loan_amount := loan_amount + loan_amount * loan_interest_rate in
  let property_insurance_cost := total_loan_amount * property_insurance_tariff in
  let title_insurance_cost := total_loan_amount * title_insurance_tariff in
  let svetlana_insurance_cost := total_loan_amount * svetlana_share * life_health_insurance_female in
  let dmitry_insurance_cost := total_loan_amount * dmitry_share * life_health_insurance_male in
  let total_insurance_cost := property_insurance_cost + title_insurance_cost + svetlana_insurance_cost + dmitry_insurance_cost in
  total_insurance_cost = 24045.84 :=
by
  sorry

end calculate_insurance_cost_l552_552686


namespace knights_on_red_chairs_l552_552722

theorem knights_on_red_chairs (K L K_r L_b : ℕ) (h1: K + L = 20)
  (h2: K - K_r + L_b = 10) (h3: K_r + L - L_b = 10) (h4: K_r = L_b) : K_r = 5 := by
  sorry

end knights_on_red_chairs_l552_552722


namespace avg_gas_mileage_40_mpg_eq_l552_552374

def avg_gas_mileage (first_leg_distance : ℕ) (second_leg_distance : ℕ) (third_leg_distance : ℕ) 
                    (electric_car_mileage : ℕ) (rented_car_mileage : ℕ) : ℕ :=
  let total_distance := first_leg_distance + second_leg_distance + third_leg_distance
  let first_leg_gal_eq := first_leg_distance / electric_car_mileage
  let second_leg_gal := second_leg_distance / rented_car_mileage
  let third_leg_gal_eq := third_leg_distance / electric_car_mileage
  let total_gal_eq := first_leg_gal_eq + second_leg_gal + third_leg_gal_eq
  total_distance / total_gal_eq

theorem avg_gas_mileage_40_mpg_eq (first_leg_distance second_leg_distance third_leg_distance : ℕ) 
                                   (electric_car_mileage rented_car_mileage : ℕ) :
  first_leg_distance = 150 ∧ second_leg_distance = 100 ∧ third_leg_distance = 150 ∧ 
  electric_car_mileage = 50 ∧ rented_car_mileage = 25 →
  avg_gas_mileage first_leg_distance second_leg_distance third_leg_distance electric_car_mileage rented_car_mileage = 40 :=
by
  intros h
  cases h with
  | intro h1 h2 h3 h4 h5 =>
    have h1 : first_leg_distance = 150 := h1
    have h2 : second_leg_distance = 100 := h2
    have h3 : third_leg_distance = 150 := h3
    have h4 : electric_car_mileage = 50 := h4
    have h5 : rented_car_mileage = 25 := h5
    -- Proof can be filled in later
    sorry

end avg_gas_mileage_40_mpg_eq_l552_552374


namespace petya_sum_of_expressions_l552_552239

theorem petya_sum_of_expressions :
  (∑ val in (Finset.univ : Finset (Fin 32)), (1 +
    (if val / 2^4 % 2 = 0 then 2 else -2) +
    (if val / 2^3 % 2 = 0 then 3 else -3) +
    (if val / 2^2 % 2 = 0 then 4 else -4) +
    (if val / 2 % 2 = 0 then 5 else -5) +
    (if val % 2 = 0 then 6 else -6))) = 32 := 
by
  sorry

end petya_sum_of_expressions_l552_552239


namespace value_of_f_pi_div_2_l552_552905

def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

theorem value_of_f_pi_div_2 : f (Real.pi / 2) = Real.pi / 2 := by
  sorry

end value_of_f_pi_div_2_l552_552905


namespace problem_f_six_neg_l552_552086

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then log (2 + x) / log 2 + (a - 1) * x + b else -f (-x)

variables (a b : ℝ)

theorem problem_f_six_neg (h1 : f 2 = -1)
  (h2 : ∀ x : ℝ, f (-x) = -f x)
  (h3 : ∀ x : ℝ, x ≥ 0 → f x = log (2 + x) / log 2 + (a - 1) * x + b)
  : f (-6) = 4 := sorry

end problem_f_six_neg_l552_552086


namespace positional_relationship_l552_552925

-- Definitions and conditions
variables {Point : Type*} [EuclideanSpace Point]  -- Assume Point is a type in Euclidean space

def is_parallel (l1 l2 : set Point) : Prop := ∃ v, ∀ p1 p2 ∈ l1, p1 - p2 = v ∧ ∃ w, ∀ p1 p2 ∈ l2, p1 - p2 = w ∧ v = w
def is_skew (l1 l2 : set Point) : Prop := l1 ∩ l2 = ∅ ∧ ¬ is_parallel l1 l2
def is_intersecting (l1 l2 : set Point) : Prop := ∃ p, p ∈ l1 ∧ p ∈ l2

variables (a b c : set Point)
hypothesis h1 : is_skew a b
hypothesis h2 : is_parallel c a

theorem positional_relationship : is_skew c b ∨ is_intersecting c b :=
sorry

end positional_relationship_l552_552925


namespace Theresa_video_games_l552_552330

variable (Tory Julia Theresa : ℕ)

def condition1 : Prop := Tory = 6
def condition2 : Prop := Julia = Tory / 3
def condition3 : Prop := Theresa = (Julia * 3) + 5

theorem Theresa_video_games : condition1 Tory → condition2 Tory Julia → condition3 Julia Theresa → Theresa = 11 := by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  sorry

end Theresa_video_games_l552_552330


namespace find_A_l552_552405

theorem find_A (A B C D E F G H I J : ℕ)
  (h1 : A > B ∧ B > C)
  (h2 : D > E ∧ E > F)
  (h3 : G > H ∧ H > I ∧ I > J)
  (h4 : (D = E + 2) ∧ (E = F + 2))
  (h5 : (G = H + 2) ∧ (H = I + 2) ∧ (I = J + 2))
  (h6 : A + B + C = 10) : A = 6 :=
sorry

end find_A_l552_552405


namespace brenda_cakes_l552_552016

-- Definitions based on the given conditions
def cakes_per_day : ℕ := 20
def days : ℕ := 9
def total_cakes_baked : ℕ := cakes_per_day * days
def cakes_sold : ℕ := total_cakes_baked / 2
def cakes_left : ℕ := total_cakes_baked - cakes_sold

-- Formulate the theorem
theorem brenda_cakes : cakes_left = 90 :=
by {
  -- To skip the proof steps
  sorry
}

end brenda_cakes_l552_552016


namespace angle_ADE_is_60_l552_552940

open Locale

-- Definitions and given conditions
variables {A B C D E : Type}
variables [metric_space A] [metric_space B] [metric_space C]
variables [has_dist A B] [has_dist B C] [has_dist C D] [has_dist D E]
variables {BD DC : ℝ}
variables (pointD : ℝ) (h1 : BD = DC) (h2 : angle B C D = 60) (BE EC : ℝ)
variables (pointE : ℝ)
variables (hE : BE = EC)

-- Proof goal
theorem angle_ADE_is_60 (h1 : BD = DC) (h2 : ∠ BCD = 60) (hE : BE = EC) : ∠ ADE = 60 := 
by
  sorry

end angle_ADE_is_60_l552_552940


namespace max_cables_l552_552768

theorem max_cables (num_employees : ℕ) 
  (num_brand_A : ℕ) (num_model_1 : ℕ) (num_model_2 : ℕ) 
  (num_brand_B : ℕ) 
  (model_1_cable_limit : ℕ) :
    num_employees = 40 →
    num_brand_A = 25 →
    num_model_1 = 10 →
    num_model_2 = 15 →
    num_brand_B = 15 →
    model_1_cable_limit = num_model_1 * num_brand_B →
    ∀ (connected_via_model_1 : ℕ), 
    connected_via_model_1 = num_model_2 →
    let max_cables := model_1_cable_limit + connected_via_model_1 in
    max_cables = 165 :=
begin
  sorry
end

end max_cables_l552_552768


namespace mul_sqrt_simplify_l552_552028

theorem mul_sqrt_simplify :
  -2 * real.sqrt 10 * (3 * real.sqrt 30) = -60 * real.sqrt 3 :=
by
  sorry

end mul_sqrt_simplify_l552_552028


namespace function_transformation_maximum_value_and_location_l552_552117

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 3) * Real.sin x * Real.cos x - (Real.cos x) ^ 2

def transformed_f (x : ℝ) : ℝ :=
  Real.sin(2 * x - Real.pi / 6) - 1 / 2

theorem function_transformation : ∀ x : ℝ, f x = transformed_f x :=
by sorry

theorem maximum_value_and_location :
  (∀ x : ℝ, f x ≤ 1 / 2) ∧ (∃ k : ℤ, ∃ x : ℝ, x = k * Real.pi + Real.pi / 3 ∧ f x = 1 / 2) :=
by sorry

end function_transformation_maximum_value_and_location_l552_552117


namespace KM_tangent_to_inscribed_circle_l552_552232

variable (A B C D K M : Point)
variable (s : ℝ) (circle_inscribed : Circle)
variable (square_ABCD : square A B C D)

variables (h1 : dist A B = s)
variables (h2 : 3 * dist A K = s)
variables (h3 : 4 * dist A M = s)
variables (h4 : circle_inscribed.center = (square_ABCD.center))

theorem KM_tangent_to_inscribed_circle :
  tangent (line_through K M) circle_inscribed :=
sorry

end KM_tangent_to_inscribed_circle_l552_552232


namespace seven_factorial_simplification_l552_552431

theorem seven_factorial_simplification : 7! - 6 * 6! - 6! = 0 := by
  sorry

end seven_factorial_simplification_l552_552431


namespace rescue_team_arrangements_l552_552802

-- Define the conditions and what needs to be proved
theorem rescue_team_arrangements :
  let teams : Finset ℕ := {1, 2, 3, 4, 5, 6} -- Representing 6 teams
  let site_A : Nat := 0
  let site_B : Nat := 1
  let site_C : Nat := 2
  let sites : Finset ℕ := {site_A, site_B, site_C}
  
  -- Each team going to one site
  -- Each disaster site must have at least one team
  -- Site A requires at least 2 teams
  -- Teams A and B cannot go to the same disaster site
  
  ∃ (team_assignment : Fin 6 → Fin 3),
  (Finset.map ⟨team_assignment, Subtype.val_inj⟩ teams).card = 6 ∧  -- Each team goes to one site
  ∀ s ∈ sites, (Finset.filter (λ t, team_assignment t = s) teams).card ≥ 1 ∧ -- Each site has at least one team
  (Finset.filter (λ t, team_assignment t = site_A) teams).card ≥ 2 ∧ -- Site A has at least 2 teams
  team_assignment ⟨0, by norm_num⟩ ≠ team_assignment ⟨1, by norm_num⟩  -- Teams A and B on different sites
  → (∃ arrangements : ℕ, arrangements = 266) := 
sorry

end rescue_team_arrangements_l552_552802


namespace S_values_general_formula_b_elements_in_P_2023_l552_552626

def a (n : ℕ) : ℕ → ℤ
| n := let k := nat.find (λ k, (k * (k + 1) / 2) ≥ n) in
             if (k * (k - 1) / 2 < n ∧ n ≤ k * (k + 1) / 2)
             then (-1)^(k-1) * ↑k
             else 0

def S (n : ℕ) : ℤ :=
  (finset.range n).sum a

def b (k : ℕ) : ℤ :=
  S (k * (k + 1) / 2)

def P (l : ℕ) : finset ℕ :=
  (finset.range l).filter (λ n, (S n) / (a n) ∈ ℤ)

-- Proof of S_1 = 1, S_2 = -1, S_3 = -3, S_4 = 0
theorem S_values : S 1 = 1 ∧ S 2 = -1 ∧ S 3 = -3 ∧ S 4 = 0 :=
by sorry

-- General formula for b_k
theorem general_formula_b (k : ℕ) : b k = (-1)^(k+1) * k * (k + 1) / 2 :=
by sorry

-- Number of elements in P_2023
theorem elements_in_P_2023 : finset.card (P 2023) = 1024 :=
by sorry

end S_values_general_formula_b_elements_in_P_2023_l552_552626


namespace parabola_focus_l552_552895

theorem parabola_focus (a : ℝ) (h_parabola : ∀ y x : ℝ, y^2 = a * x) (h_directrix : ∀ x : ℝ, x = -1) :
  (1, 0) ∈ {p : ℝ × ℝ | ∃ a, y^2 = a * x ∧ d(p, focus_vertex) = d(p, directrix)} :=
by
  sorry

end parabola_focus_l552_552895


namespace find_integers_with_conditions_l552_552108

theorem find_integers_with_conditions :
  ∃ a b c d : ℕ, (1 ≤ a) ∧ (1 ≤ b) ∧ (1 ≤ c) ∧ (1 ≤ d) ∧ a * b * c * d = 2002 ∧ a + b + c + d < 40 := sorry

end find_integers_with_conditions_l552_552108


namespace erin_walks_less_l552_552821

variable (total_distance : ℕ)
variable (susan_distance : ℕ)

theorem erin_walks_less (h1 : total_distance = 15) (h2 : susan_distance = 9) :
  susan_distance - (total_distance - susan_distance) = 3 := by
  sorry

end erin_walks_less_l552_552821


namespace solve_fractional_equation_l552_552675

-- Define the fractional equation as a function
def fractional_equation (x : ℝ) : Prop :=
  (3 / 2) - (2 * x) / (3 * x - 1) = 7 / (6 * x - 2)

-- State the theorem we need to prove
theorem solve_fractional_equation : fractional_equation 2 :=
by
  -- Placeholder for proof
  sorry

end solve_fractional_equation_l552_552675


namespace power_sum_is_integer_l552_552511

noncomputable theory

open Nat

theorem power_sum_is_integer 
  (x : ℝ) 
  (h0 : x > 0) 
  (h1 : x + 1 / x ∈ ℤ) : 
  ∀ n : ℕ, n > 0 → (x ^ n + 1 / x ^ n) ∈ ℤ := 
by
  sorry

end power_sum_is_integer_l552_552511


namespace trajectory_equation_max_value_on_trajectory_l552_552088

-- Given conditions
def distance_ratio (x y : ℝ) : Prop := 
  (Real.sqrt (x^2 + y^2)) = (1 / 2) * (Real.sqrt ((x - 3)^2 + y^2))

-- Prove the equation of the trajectory
theorem trajectory_equation (x y : ℝ) (h : distance_ratio x y) : 
  (x + 1)^2 + (4 / 3) * y^2 = 4 := 
sorry

-- Prove the maximum value of 2x^2 + y^2 on this trajectory
theorem max_value_on_trajectory (x y : ℝ) (h : (x + 1)^2 + (4 / 3) * y^2 = 4) : 
  (2 * x^2 + y^2) ≤ 18 := 
sorry

end trajectory_equation_max_value_on_trajectory_l552_552088


namespace smallest_m_last_four_digits_l552_552211

theorem smallest_m_last_four_digits : ∃ m : ℕ,
  (m % 5 = 0) ∧ (m % 8 = 0) ∧
  (∀ d ∈ Int.digits 10 m, d = 2 ∨ d = 7) ∧
  (2 ∈ Int.digits 10 m) ∧ (7 ∈ Int.digits 10 m) ∧
  (Int.digits 10 m).length ≥ 4 ∧
  (m % 10000 = 7272) := by
sorry

end smallest_m_last_four_digits_l552_552211


namespace negation_of_existential_l552_552710

theorem negation_of_existential :
  (∀ x : ℝ, x^2 + x - 1 ≤ 0) ↔ ¬ (∃ x : ℝ, x^2 + x - 1 > 0) :=
by sorry

end negation_of_existential_l552_552710


namespace rotate_parabola_180_l552_552341

theorem rotate_parabola_180 (x y : ℝ) : 
  (y = 2 * (x - 1)^2 + 2) → 
  (∃ x' y', x' = -x ∧ y' = -y ∧ y' = -2 * (x' + 1)^2 - 2) := 
sorry

end rotate_parabola_180_l552_552341


namespace unique_scores_proof_l552_552765

noncomputable def score := ℕ

-- Define a function result that returns true if team i wins against team j
def result (i j : ℕ) : Prop := sorry

-- Define a predicate to capture the conditions of the tournament
def unique_scores (teams : fin 93 → ℕ) : Prop := 
∀ t₁ t₂, t₁ ≠ t₂ → teams t₁ ≠ teams t₂

-- The condition that in any group of 19 teams, there's a team that defeated the other 18
def defeated_condition (teams : fin 93) : Prop := 
∀ (subset : finset (fin 93)), subset.card = 19 → 
  ∃ (i : fin 93), (ι ≠ i) →  (j ∈ subset \ {i}) → result i j

-- The condition that in any group of 19 teams, there's a team that lost to the other 18
def lost_condition (teams : fin 93) : Prop :=
∀ (subset : finset (fin 93)), subset.card = 19 →
  ∃ (i : fin 93), (ι ≠ i) →  (j ∈ subset \ {i}) → result j i 

theorem unique_scores_proof (teams : fin 93 → ℕ) 
  (h_defeated : defeated_condition teams) 
  (h_lost : lost_condition teams) :
  unique_scores teams :=
sorry

end unique_scores_proof_l552_552765


namespace find_p_value_l552_552911

open Set

/-- Given the parabola C: y^2 = 2px with p > 0, point A(0, sqrt(3)),
    and point B on the parabola such that AB is perpendicular to AF,
    and |BF| = 4. Determine the value of p. -/
theorem find_p_value (p : ℝ) (h : p > 0) :
  ∃ p, p = 2 ∨ p = 6 :=
sorry

end find_p_value_l552_552911


namespace solve_xy_l552_552929

theorem solve_xy (x y a b : ℝ) (h1 : x * y = 2 * b) (h2 : (1 / x^2) + (1 / y^2) = a) : 
  (x + y)^2 = 4 * a * b^2 + 4 * b := 
by 
  sorry

end solve_xy_l552_552929


namespace slope_of_line_between_midpoints_l552_552743

noncomputable def midpoint (A B : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def slope (P Q : (ℝ × ℝ)) : ℝ :=
  (Q.2 - P.2) / (Q.1 - P.1)

theorem slope_of_line_between_midpoints :
  let M1 := midpoint (3, 4) (7, 8)
  let M2 := midpoint (6, 2) (9, 5)
  slope M1 M2 = -1 := by
  sorry

end slope_of_line_between_midpoints_l552_552743


namespace exists_nat_concat_is_perfect_square_l552_552819

theorem exists_nat_concat_is_perfect_square :
  ∃ A : ℕ, ∃ n : ℕ, ∃ B : ℕ, (B * B = (10^n + 1) * A) :=
by sorry

end exists_nat_concat_is_perfect_square_l552_552819


namespace ellipse_incenter_line_exists_l552_552900

theorem ellipse_incenter_line_exists : 
  ∃ l : ℝ × ℝ × ℝ, 
    (∃ (x y : ℝ), (x^2 / 2 + y^2 = 1) ∧ (F = (1, 0)) ∧ (M = (0, 1)) ∧ (l.1 * x + l.2 * y + l.3 = 0)) ∧
    (l.1 = 1) ∧ (l.2 = 2 - Real.sqrt 6) ∧ (l.3 = 6 - 3 * Real.sqrt 6) :=
begin
  sorry
end

end ellipse_incenter_line_exists_l552_552900


namespace points_are_concyclic_l552_552601
noncomputable theory
open_locale classical

-- Define the basic geometric structures
variables {A B C D E F I M N : Type*}
           [incircle_property : is_incenter (triangle A B C) I]
           [angle_bisector_AD : is_angle_bisector (triangle A B C) A D]
           [angle_bisector_BE : is_angle_bisector (triangle A B C) B E]
           [angle_bisector_CF : is_angle_bisector (triangle A B C) C F]
           [perpendicular_bisector_AD : is_perpendicular_bisector (segment A D) (line A M)]
           [intersect_BE_M : line_intersects_segment (line M) (segment B E) M]
           [intersect_CF_N : line_intersects_segment (line N) (segment C F) N]

-- Statement of the problem
theorem points_are_concyclic : are_concyclic A I M N :=
sorry

end points_are_concyclic_l552_552601


namespace train_length_l552_552388

-- Definitions from the conditions
def jogger_speed_km_per_hr: ℝ := 9
def train_speed_km_per_hr: ℝ := 45
def jogger_lead_m: ℝ := 200
def train_passing_time_s: ℝ := 41

-- Convert speeds from km/hr to m/s
def jogger_speed_m_per_s: ℝ := jogger_speed_km_per_hr * (1000 / 3600) -- 9 * (1000m / 1km) / 3600s
def train_speed_m_per_s: ℝ := train_speed_km_per_hr * (1000 / 3600) -- 45 * (1000m / 1km) / 3600s

-- Relative speed of the train with respect to the jogger
def relative_speed_m_per_s: ℝ := train_speed_m_per_s - jogger_speed_m_per_s

-- Distance covered by the train relative to the jogger in given time
def distance_covered_m: ℝ := relative_speed_m_per_s * train_passing_time_s

-- Proof problem: Prove that the length of the train is 210 meters
theorem train_length : distance_covered_m - jogger_lead_m = 210 := by
  sorry

end train_length_l552_552388


namespace negation_of_proposition_l552_552297

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 1 → sqrt x > 1)) ↔ (∃ x : ℝ, x > 1 ∧ sqrt x ≤ 1) :=
by 
  sorry

end negation_of_proposition_l552_552297


namespace sequence_periodic_l552_552873

def sequence (a : ℕ → ℝ) :=
  a 1 + a 2 + a 3 = 6 ∧ ∀ n, a (n + 1) = -1 / (a n + 1)

theorem sequence_periodic (a : ℕ → ℝ) (h : sequence a) : 
  a 16 + a 17 + a 18 = 6 :=
sorry

end sequence_periodic_l552_552873


namespace sin_cos_equiv_l552_552882

theorem sin_cos_equiv (x : ℝ) (h : Real.cos x - 5 * Real.sin x = 2) :
  Real.sin x + 5 * Real.cos x = -1/2 ∨ Real.sin x + 5 * Real.cos x = 17/13 := 
by
  sorry

end sin_cos_equiv_l552_552882


namespace sum_sequence_2022_l552_552521

section
variables (n : ℕ) (x : ℕ → ℚ) 

def sequence_satisfies : Prop :=
  x 0 = 1 / n ∧ ∀ k : ℕ, k < n → x (k + 1) = (1 / (n - (k + 1))) * (∑ i in Finset.range (k + 1), x i)

theorem sum_sequence_2022 (h : sequence_satisfies 2022 x) : 
  ∑ i in Finset.range 2022, x i = 1 :=
begin
  -- proof goes here
  sorry
end

end

end sum_sequence_2022_l552_552521


namespace find_a_b_find_m_l552_552548

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x^3 + b * x^2

theorem find_a_b (a b : ℝ) (h₁ : f 1 a b = 4)
  (h₂ : 3 * a + 2 * b = 9) : a = 1 ∧ b = 3 :=
by
  sorry

theorem find_m (m : ℝ) (h : ∀ x, (m ≤ x ∧ x ≤ m + 1) → (3 * x^2 + 6 * x > 0)) :
  m ≥ 0 ∨ m ≤ -3 :=
by
  sorry

end find_a_b_find_m_l552_552548


namespace product_exceeds_100000_l552_552913

theorem product_exceeds_100000 (n : ℕ) (h : (∏ k in finset.range n, 10 ^ (k + 1) / 11) > 100000) : n = 11 :=
sorry

end product_exceeds_100000_l552_552913


namespace men_for_first_road_is_30_l552_552332

noncomputable def men_needed_for_first_road : ℝ :=
  let total_man_hours_1km := 12 * 8
  let total_man_hours_2km := 20 * 20.571428571428573 * 14 / 2
  total_man_hours_2km / total_man_hours_1km

theorem men_for_first_road_is_30 :
  men_needed_for_first_road ≈ 30 :=
by
  let x := men_needed_for_first_road
  have h1 : x = (20 * 20.571428571428573 * 14 / 2) / (12 * 8)
    := by
      rw [total_man_hours_1km, total_man_hours_2km]
  have h2 : x ≈ 30
    := by
      norm_num at h1
      linarith
  exact h2

end men_for_first_road_is_30_l552_552332


namespace collinear_vectors_iff_l552_552918

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Given two non-zero vectors e1 and e2 that are not collinear
variables (e1 e2 : V) (k : ℝ)
hypothesis (h_nonzero_e1 : e1 ≠ 0)
hypothesis (h_nonzero_e2 : e2 ≠ 0)
hypothesis (h_not_collinear : ¬(∃ (c : ℝ), e1 = c • e2))

-- If ke1 + 2e2 and 3e1 + ke2 are collinear, then k = ±√6
theorem collinear_vectors_iff :
  (∃ (λ : ℝ), ke1 + 2 • e2 = λ • (3 • e1 + k • e2)) → k = ±√6 :=
by
  sorry

end collinear_vectors_iff_l552_552918


namespace find_certain_number_l552_552939

noncomputable def certain_number (x : ℝ) : Prop :=
  3005 - 3000 + x = 2705

theorem find_certain_number : ∃ x : ℝ, certain_number x ∧ x = 2700 :=
by
  use 2700
  unfold certain_number
  sorry

end find_certain_number_l552_552939


namespace time_to_cross_pole_is_correct_l552_552975

-- Define the conversion factor to convert km/hr to m/s
def km_per_hr_to_m_per_s (speed_km_per_hr : ℕ) : ℕ := speed_km_per_hr * 1000 / 3600

-- Define the speed of the train in m/s
def train_speed_m_per_s : ℕ := km_per_hr_to_m_per_s 216

-- Define the length of the train
def train_length_m : ℕ := 480

-- Define the time to cross an electric pole
def time_to_cross_pole : ℕ := train_length_m / train_speed_m_per_s

-- Theorem stating that the computed time to cross the pole is 8 seconds
theorem time_to_cross_pole_is_correct :
  time_to_cross_pole = 8 := by
  sorry

end time_to_cross_pole_is_correct_l552_552975


namespace pyramid_volume_theorem_l552_552781

-- Define the variables, constants and relevant conditions
variables (length width : ℝ) (edge : ℝ)
variable h : 2 * sqrt 41

-- Define the condition consistent with the problem statement
def pyramid_volume (length width edge : ℝ) :=
  ∃ (h : ℝ), 
    volume = (1/3) * (length * width) * h ∧
    length = 10 ∧
    width = 12 ∧
    edge = 15 ∧
    h = 2 * sqrt 41

-- State the theorem
theorem pyramid_volume_theorem : 
  pyramid_volume 10 12 15 80 * sqrt 41 :=
    sorry

end pyramid_volume_theorem_l552_552781


namespace no_integer_solutions_l552_552830

theorem no_integer_solutions (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
  sorry

end no_integer_solutions_l552_552830


namespace domain_of_f_f_is_odd_f_positive_range_l552_552125

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) (a) - Real.log (1 - x) (a)

-- Define conditions
variables a x : ℝ
variable h_a : 0 < a ∧ a ≠ 1

-- Prove domain of f(x) is -1 < x < 1
theorem domain_of_f (h_a : 0 < a ∧ a ≠ 1) : -1 < x ∧ x < 1 ↔ (x + 1 > 0 ∧ 1 - x > 0) :=
sorry

-- Prove f(x) is odd
theorem f_is_odd (h_a : 0 < a ∧ a ≠ 1) : f a (-x) = -f a x :=
sorry

-- Prove for a > 1, f(x) > 0 ↔ 0 < x < 1
theorem f_positive_range (h_a : 1 < a) : 0 < x ∧ x < 1 ↔ f a x > 0 :=
sorry

end domain_of_f_f_is_odd_f_positive_range_l552_552125


namespace complement_union_l552_552629

-- Define the universal set I
def I : Set ℕ := {1, 2, 3, 4}

-- Define the set S
def S : Set ℕ := {1, 3}

-- Define the set T
def T : Set ℕ := {4}

-- Define the complement of S in I
def complement_I_S : Set ℕ := I \ S

-- State the theorem to be proved
theorem complement_union : (complement_I_S ∪ T) = {2, 4} := by
  sorry

end complement_union_l552_552629


namespace range_of_function_l552_552490

theorem range_of_function : 
  (set.range (λ x : ℝ, x / (x^2 + x + 1)) = set.Icc (1/3) 1) :=
sorry

end range_of_function_l552_552490


namespace truck_sand_at_arrival_l552_552793

-- Definitions based on conditions in part a)
def initial_sand : ℝ := 4.1
def lost_sand : ℝ := 2.4

-- Theorem statement corresponding to part c)
theorem truck_sand_at_arrival : initial_sand - lost_sand = 1.7 :=
by
  -- "sorry" placeholder to skip the proof
  sorry

end truck_sand_at_arrival_l552_552793


namespace sum_of_roots_l552_552358

theorem sum_of_roots {a b : Real} (h1 : a * (a - 4) = 5) (h2 : b * (b - 4) = 5) (h3 : a ≠ b) : a + b = 4 :=
by
  sorry

end sum_of_roots_l552_552358


namespace jane_mistake_l552_552403

theorem jane_mistake (x y z : ℤ) (h1 : x - y + z = 15) (h2 : x - y - z = 7) : x - y = 11 :=
by sorry

end jane_mistake_l552_552403


namespace ellipse_from_distances_l552_552179

open EuclideanGeometry

noncomputable def point := (ℝ × ℝ)

theorem ellipse_from_distances (A B : point) (P : point) :
  A = (0, 0) → B = (4, 0) → (dist P A + dist P B = 8) →
  (∃ c : point, c = (2, 0) ∧ is_ellipse_with_foci_and_major_axis A B 8 c) :=
by
  intros hA hB hP
  use (2, 0)
  sorry

end ellipse_from_distances_l552_552179


namespace pizza_remained_l552_552952

theorem pizza_remained (total_people : ℕ) (fraction_eating_pizza : ℚ)
  (total_pizza_pieces : ℕ) (pieces_per_person : ℕ)
  (h1 : total_people = 15)
  (h2 : fraction_eating_pizza = 3 / 5)
  (h3 : total_pizza_pieces = 50)
  (h4 : pieces_per_person = 4) :
  total_pizza_pieces - (((total_people : ℚ) * fraction_eating_pizza).natCast * pieces_per_person) = 14 := by
  sorry

end pizza_remained_l552_552952


namespace validate_sequences_and_minimization_l552_552872

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := a n + 2

-- Define the sum of the first n terms S_n
def S (n : ℕ) : ℕ := n * n

-- Define the sequence {b_n}
def b : ℕ → ℕ
| 0       := 17
| (n + 1) := b n + 2 * n

-- To find the value of n that minimizes b_n / sqrt(S_n)
def minimize_bn_over_sqrt_Sn : ℕ := 4

-- Main theorem statement
theorem validate_sequences_and_minimization :
  (∀ n, a n = 2 * n - 1) ∧
  (∀ n, S n = n * n) ∧
  (∀ n, b n = n * n - n + 17) ∧
  minimize_bn_over_sqrt_Sn = 4 :=
by
  -- Proof goes here
  sorry

end validate_sequences_and_minimization_l552_552872


namespace total_students_l552_552786

theorem total_students (rank_right rank_left : ℕ) (h_right : rank_right = 18) (h_left : rank_left = 12) : rank_right + rank_left - 1 = 29 := 
by
  sorry

end total_students_l552_552786


namespace equilateral_triangle_area_decrease_l552_552416

theorem equilateral_triangle_area_decrease (A : ℝ) (A' : ℝ) (s s' : ℝ) 
  (h1 : A = 121 * Real.sqrt 3) 
  (h2 : A = (s^2 * Real.sqrt 3) / 4) 
  (h3 : s' = s - 8) 
  (h4 : A' = (s'^2 * Real.sqrt 3) / 4) :
  A - A' = 72 * Real.sqrt 3 := 
by sorry

end equilateral_triangle_area_decrease_l552_552416


namespace find_BC_l552_552186

-- Define the conditions as Lean statements
def angle_B : ℝ := 45 * Real.pi / 180
def angle_C : ℝ := 60 * Real.pi / 180
def AB : ℝ := 6

-- Given the above conditions, prove that the length of BC is approximately 8.196
theorem find_BC (hc1 : angle_B = 45 * Real.pi / 180)
                (hc2 : angle_C = 60 * Real.pi / 180)
                (hc3 : AB = 6) : 
                ∃ BC : ℝ, BC ≈ 8.196 := by
  sorry

end find_BC_l552_552186


namespace eccentricity_of_hyperbola_l552_552702

-- Define the hyperbola and related conditions
variables (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0)
def hyperbola_C := ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the foci positions
def focus_F1 := (-c, 0)
def focus_F2 := (c, 0)

-- Define the points on the hyperbola and perpendicular condition
variables (M N : ℝ × ℝ)
def on_hyperbola (P : ℝ × ℝ) : Prop := (P.1^2 / a^2) - (P.2^2 / b^2) = 1
def perpendicular (P1 P2 : ℝ × ℝ) : Prop := P1.1 * P2.1 + P1.2 * P2.2 = 0

-- Define points meeting given conditions
def MN_perpendicular_F1F2 := perpendicular (M.1 - N.1, M.2 - N.2) (focus_F2.1 - focus_F1.1, focus_F2.2 - focus_F1.2)
def F1F2_length_relation := dist focus_F1 focus_F2 = 4 * dist M N

-- Define the intersection condition
variables (Q : ℝ × ℝ)
def intersection_condition := (dist focus_F1 Q = dist Q N) ∧ on_hyperbola N

-- Define the goal
theorem eccentricity_of_hyperbola : (∀ (x y : ℝ), hyperbola_C a b x y) → 
  MN_perpendicular_F1F2 → F1F2_length_relation → intersection_condition →
  (sqrt (1 + (b^2 / a^2))) = sqrt 6 :=
by
  sorry

end eccentricity_of_hyperbola_l552_552702


namespace median_perpendicular_to_projection_line_l552_552206

-- Definition of projections of the Lemoine point
variables {A B C K A1 B1 C1 : Type*} [MetricSpace K]
variables {ABC : Triangle A B C} 
variables {proj_A1 : Projection K BC}
variables {proj_B1 : Projection K CA}
variables {proj_C1 : Projection K AB}
variables {AM : Median A B C}

-- Proof that the median AM from A to BC is perpendicular to the line B1C1
theorem median_perpendicular_to_projection_line :
  (isLemoinePoint K ABC) →
  (projection A1 K BC) →
  (projection B1 K CA) →
  (projection C1 K AB) →
  perpendicular (median AM A BC) (line B1 C1) := sorry

end median_perpendicular_to_projection_line_l552_552206


namespace standard_circle_equation_passing_through_P_l552_552553

-- Define the condition that a point P is a solution to the system of equations derived from the line
def PointPCondition (x y : ℝ) : Prop :=
  (2 * x + 3 * y - 1 = 0) ∧ (3 * x - 2 * y + 5 = 0)

-- Define the center and radius of the given circle C
def CenterCircleC : ℝ × ℝ := (2, -3)
def RadiusCircleC : ℝ := 4  -- Since the radius squared is 16

-- Define the condition that a point is on a circle with a given center and radius
def OnCircle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.fst)^2 + (y + center.snd)^2 = radius^2

-- State the problem
theorem standard_circle_equation_passing_through_P :
  ∃ (x y : ℝ), PointPCondition x y ∧ OnCircle CenterCircleC 5 x y :=
sorry

end standard_circle_equation_passing_through_P_l552_552553


namespace arithmetic_sequence_formula_l552_552175

theorem arithmetic_sequence_formula (a : ℕ → ℤ) (d : ℤ) :
  (a 3 = 4) → (d = -2) → ∀ n : ℕ, a n = 10 - 2 * n :=
by
  intros h1 h2 n
  sorry

end arithmetic_sequence_formula_l552_552175


namespace quadrilateral_area_inequality_l552_552647

theorem quadrilateral_area_inequality (a b c d : ℝ) :
  ∃ (S_ABCD : ℝ), S_ABCD ≤ (1 / 4) * (a + c) ^ 2 + b * d :=
sorry

end quadrilateral_area_inequality_l552_552647


namespace tetrahedron_circumsphere_radius_l552_552970

/-- In the tetrahedron ABCD, it is known that ∠ADB = ∠BDC = ∠CDA = 60°, AD = BD = 3, and CD = 2. 
Find the radius of the circumscribed sphere of the tetrahedron ABCD. -/
theorem tetrahedron_circumsphere_radius {A B C D : Type} 
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
  (h_angle_ADB : ∠A D B = 60) 
  (h_angle_BDC : ∠B D C = 60) 
  (h_angle_CDA : ∠C D A = 60) 
  (h_AD : dist A D = 3) 
  (h_BD : dist B D = 3) 
  (h_CD : dist C D = 2) :
  ∃ (R : ℝ), R = √3 :=
by
  sorry

end tetrahedron_circumsphere_radius_l552_552970


namespace stable_scores_l552_552678

theorem stable_scores (S_A S_B S_C S_D : ℝ) (hA : S_A = 2.2) (hB : S_B = 6.6) (hC : S_C = 7.4) (hD : S_D = 10.8) : 
  S_A ≤ S_B ∧ S_A ≤ S_C ∧ S_A ≤ S_D :=
by
  sorry

end stable_scores_l552_552678


namespace probability_of_success_l552_552937

open Finset

noncomputable def numbers : Finset ℕ := {5, 14, 28, 35, 49, 54, 63}

def product_is_multiple_of_126 (a b : ℕ) : Prop :=
  ∃ (x y z : ℕ), 2^x * 3^(2*y) * 7^z ∣ a * b ∧ x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1

def successful_pairings : Finset (ℕ × ℕ) :=
  numbers.product numbers.filter (λ x, x ≠ x.1)

def count_successful : ℚ :=
  (successful_pairings.filter (λ p, product_is_multiple_of_126 p.1 p.2)).card

def total_pairings : ℚ := (numbers.card.choose 2 : ℚ)

theorem probability_of_success : (count_successful / total_pairings) = 4 / 21 :=
by
  sorry

end probability_of_success_l552_552937


namespace parallel_RS_CD_l552_552623

-- Conditions:
variables (A B C D P Q R S : Type)
variables [CyclicQuadrilateral A B C D]
variables (h1 : AB < CD)
variables (h2 : Intersect (AC) (BD) = P)
variables (h3 : Intersect (AD) (BC) = Q)
variables (h4 : AngleBisector (∠AQB) ∩ [AC] = R)
variables (h5 : AngleBisector (∠APD) ∩ (AD) = S)

-- Proof problem: Prove that (RS) is parallel to (CD) given the conditions.
theorem parallel_RS_CD
  (A B C D P Q R S : Type)
  [CyclicQuadrilateral A B C D]
  (h1 : AB < CD)
  (h2 : Intersect (AC) (BD) = P)
  (h3 : Intersect (AD) (BC) = Q)
  (h4 : AngleBisector (∠AQB) ∩ [AC] = R)
  (h5 : AngleBisector (∠APD) ∩ (AD) = S) :
  Parallel (RS) (CD) :=
sorry

end parallel_RS_CD_l552_552623


namespace find_exponent_l552_552146

theorem find_exponent (y : ℕ) (h : 5^12 = 25^y) : y = 6 :=
by {
  let base := 5,
  have hyp : 25 = base^2 := by norm_num,
  rw [← hyp, ← pow_mul] at h,
  simp at h,
  exact nat.eq_of_mul_eq_mul_right (nat.succ_pos 1) h
}

end find_exponent_l552_552146


namespace polar_to_cartesian_l552_552037

theorem polar_to_cartesian (ρ θ x y : ℝ) 
  (h1 : ρ^2 * real.cos θ - ρ = 0)
  (h2 : ρ^2 = x^2 + y^2)
  (h3 : ρ * real.cos θ = x)
  (h4 : ρ * real.sin θ = y) :
  x^2 + y^2 = 0 ∨ x = 1 := 
by 
  sorry

end polar_to_cartesian_l552_552037


namespace number_of_N_satisfying_condition_l552_552468

def is_solution (N : ℕ) : Prop :=
  2017 % N = 17

theorem number_of_N_satisfying_condition : 
  {N : ℕ | is_solution N}.to_finset.card = 13 :=
sorry

end number_of_N_satisfying_condition_l552_552468


namespace chess_team_photo_arrangements_l552_552796

theorem chess_team_photo_arrangements
  (boys : Finset ℕ) (girls : Finset ℕ) (teacher : ℕ)
  (h_boys : boys.card = 3) (h_girls : girls.card = 2) :
  ∃ arrangements : ℕ, arrangements = 144 :=
by
  -- Define boys, girls, and teacher such that the total number of arrangements can be determined
  have h_arrangements : arrangements = 144,
  {
    sorry
  }
  use 144

end chess_team_photo_arrangements_l552_552796


namespace only_even_n_satisfy_conditions_l552_552856

def arithmetic_mean_not_whole (l : list ℕ) : Prop :=
  ∀ (k : ℕ) (h : 2 ≤ k) (s : list ℕ), l.take k = s → (s.sum % k) ≠ 0

theorem only_even_n_satisfy_conditions (N : ℕ) : 
  ∃ (l : list ℕ), (∀ m ∈ l, 1 ≤ m ∧ m ≤ N) ∧ (l.length = N) ∧ arithmetic_mean_not_whole l ↔ ∃ k, N = 2 * k :=
by
  sorry

end only_even_n_satisfy_conditions_l552_552856


namespace product_ineq_l552_552535

theorem product_ineq (n : ℕ) (a : Fin n → ℝ) (h : ∏ i in Finset.finRange n, a i = 1)
  (hpos : ∀ i, 0 < a i) :
  ∏ i in Finset.finRange n, (2 + a i) ≥ 3 ^ n :=
by
  sorry

end product_ineq_l552_552535


namespace Petya_sum_l552_552255

theorem Petya_sum : 
  let expr := [1, 2, 3, 4, 5, 6]
  let values := 2^(expr.length - 1)
  (sum_of_possible_values expr = values) := by 
  sorry

end Petya_sum_l552_552255


namespace binomial_cubes_sum_l552_552065

theorem binomial_cubes_sum (x y : ℤ) :
  let B1 := x^4 + 9 * x * y^3
  let B2 := -(3 * x^3 * y) - 9 * y^4
  (B1 ^ 3 + B2 ^ 3 = x ^ 12 - 729 * y ^ 12) := by
  sorry

end binomial_cubes_sum_l552_552065


namespace angle_between_BC_and_AD_is_right_angle_l552_552949

-- Define the convex quadrilateral ABCD
structure ConvexQuadrilateral (A B C D : Type) :=
(sideBC : ℝ)
(sideAD : ℝ)
(midpoints_dist : ℝ)

-- Define the main problem statement
theorem angle_between_BC_and_AD_is_right_angle
  {A B C D : Type} 
  (quadr : ConvexQuadrilateral A B C D)
  (hBC : quadr.sideBC = 6)
  (hAD : quadr.sideAD = 8)
  (hMN : quadr.midpoints_dist = 5) : 
  (angle_between (vector_of_points B C) (vector_of_points A D)) = 90 :=
by
  sorry

end angle_between_BC_and_AD_is_right_angle_l552_552949


namespace petya_sum_of_expressions_l552_552238

theorem petya_sum_of_expressions :
  (∑ val in (Finset.univ : Finset (Fin 32)), (1 +
    (if val / 2^4 % 2 = 0 then 2 else -2) +
    (if val / 2^3 % 2 = 0 then 3 else -3) +
    (if val / 2^2 % 2 = 0 then 4 else -4) +
    (if val / 2 % 2 = 0 then 5 else -5) +
    (if val % 2 = 0 then 6 else -6))) = 32 := 
by
  sorry

end petya_sum_of_expressions_l552_552238


namespace sequence_difference_l552_552716

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a n + a (n + 1) = n^2 + (-1 : ℤ)^n

theorem sequence_difference (a : ℕ → ℤ) (h : sequence a) : a 101 - a 1 = 5150 := by
  sorry

end sequence_difference_l552_552716


namespace multiples_of_12_in_range_l552_552143

theorem multiples_of_12_in_range (a b : ℕ) (h1 : a = 35) (h2 : b = 247) : 
  (finset.filter (λ x, (x % 12 = 0) ∧ (35 < x) ∧ (x < 247)) (finset.range 247)).card = 18 :=
by
  sorry

end multiples_of_12_in_range_l552_552143


namespace ages_total_l552_552387

variable (A B C : ℕ)

theorem ages_total (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : B = 10) : A + B + C = 27 :=
by
  sorry

end ages_total_l552_552387


namespace negation_of_p_l552_552933

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, 2 * x^2 - 1 > 0

-- State the theorem that the negation of p is equivalent to the given existential statement
theorem negation_of_p :
  ¬p ↔ ∃ x : ℝ, 2 * x^2 - 1 ≤ 0 :=
by
  sorry

end negation_of_p_l552_552933


namespace petya_sum_expression_l552_552252

theorem petya_sum_expression : 
  (let expressions := finset.image (λ (s : list bool), 
    list.foldl (λ acc ⟨b, n⟩, if b then acc + n else acc - n) 1 (s.zip [2, 3, 4, 5, 6])) 
    (finset.univ : finset (vector bool 5))) in
    expressions.sum) = 32 := 
sorry

end petya_sum_expression_l552_552252


namespace triangle_area_max_eq_16_l552_552967

open Real

noncomputable def max_triangle_area (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ :=
  let OA := 8 * sin α
  let OB := 8 * cos α
  (1/2) * OA * OB

theorem triangle_area_max_eq_16 (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  max_triangle_area α hα ≤ 16 :=
by {
  have h1 : 0 < sin α ∧ sin α ≤ 1, from ⟨sin_pos_of_pos_of_lt_pi (hα.left) (lt_trans hα.right (by linarith)) (by linarith), sin_le_one α⟩,
  have h2 : 0 < cos α ∧ cos α ≤ 1, from ⟨cos_pos_of_neg_of_lt_pi_div_two (by linarith : α < π / 2) (by linarith : 0 < α), cos_le_one α⟩,
  
  calc
    max_triangle_area α hα = (1/2) * (8 * sin α) * (8 * cos α) : by rfl
    ... = 16 * (sin α * cos α) : by ring
    ... = 16 * (1/2 * sin (2 * α)) : by rw [← sin_double_angle]
    ... = 8 * sin (2 * α) : by ring
    ... ≤ 8 * 1 : mul_le_mul_of_nonneg_left (le_of_lt (sin_lt_one_of_lt_pi _)) (by norm_num)
    ... = 8 : by norm_num,
  linarith,
  rw two_mul,
  exact add_lt_add hα.left hα.right,
}

end triangle_area_max_eq_16_l552_552967


namespace problem_2009_sum_l552_552850

open Nat

noncomputable def greatest_integer (x : ℝ) : ℤ :=
  ⌊x⌋ -- Greatest integer function

def a (n : ℕ) : ℤ :=
  greatest_integer (n / 10)

def S (n : ℕ) : ℤ :=
  ∑ k in range (n + 1), a k

theorem problem_2009_sum : (S 2009) / 2010 = 100 :=
by
  sorry

end problem_2009_sum_l552_552850


namespace brenda_cakes_l552_552023

theorem brenda_cakes : 
  let cakes_per_day := 20
  let days := 9
  let total_cakes := cakes_per_day * days
  let sold_cakes := total_cakes / 2
  total_cakes - sold_cakes = 90 :=
by 
  sorry

end brenda_cakes_l552_552023


namespace max_sqrt_expr_l552_552209

theorem max_sqrt_expr (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 5) :
  ∃ y, y = sqrt (a + 1) + sqrt (b + 3) ∧ y ≤ 3 * sqrt 2 :=
begin
  sorry
end

end max_sqrt_expr_l552_552209


namespace extinction_criteria_l552_552390

variable (p0 p1 p2 p3 : ℝ)
variable (E_X : ℝ)
variable (p : ℝ)

noncomputable def expected_value (p1 p2 p3 : ℝ) : ℝ := p1 + 2 * p2 + 3 * p3

theorem extinction_criteria
  (h0 : p0 = 0.4)
  (h1 : p1 = 0.3)
  (h2 : p2 = 0.2)
  (h3 : p3 = 0.1)
  (hE : E_X = expected_value p1 p2 p3)
  (h_eq : ∀ x, p0 + p1 * x + p2 * x^2 + p3 * x^3 = x ↔ x ∈ {p}) :
  (E_X ≤ 1 → p = 1) ∧ (E_X > 1 → p < 1) := 
  sorry

end extinction_criteria_l552_552390


namespace digit_120_in_concatenated_numbers_l552_552214

def concatenated_digits := String.mk [toString n | n in List.range (51 + 1)]

def nth_digit (s : String) (n : Nat) : Char :=
s.get ⟨n - 1, by {
    simp;
    sorry /- needs proof that n-1 is within bounds -/
}⟩

theorem digit_120_in_concatenated_numbers 
  : nth_digit concatenated_digits 120 = '1' :=
sorry

end digit_120_in_concatenated_numbers_l552_552214


namespace find_angle_B_find_sin_C_l552_552185

-- Definitions
variables (A B C : ℝ) (a b c : ℝ)
variables (sin cos : ℝ → ℝ)
noncomputable def sin_A := sin A
noncomputable def sin_B := sin B
noncomputable def sin_C := sin C
noncomputable def cos_A := cos A

-- Part I: Prove B = π/4 given condition
theorem find_angle_B
  (h1 : a * sin_A + c * sin_C - (√2) * a * sin_C = b * sin_B) :
  B = π / 4 :=
sorry

-- Part II: Prove sin C = (4+√2)/6 given condition
theorem find_sin_C
  (h2 : cos A = 1 / 3) :
  sin C = (4 + √2) / 6 :=
sorry

end find_angle_B_find_sin_C_l552_552185


namespace circle_arrangements_correct_l552_552570

def is_rel_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def is_valid_circle (circle : List ℕ) : Prop :=
  ∀ i, is_rel_prime (circle.nth i) (circle.nth ((i + 1) % circle.length))

noncomputable def number_of_arrangements : ℕ :=
  36

theorem circle_arrangements_correct :
  let circle := { nums : List ℕ // nums ~ {1,2,3,4,5,6,7,8} ∧ is_valid_circle nums }
  Fintype.card circle = 36 := 
sorry

end circle_arrangements_correct_l552_552570


namespace determine_x_l552_552147

theorem determine_x (x y : ℝ) (h : x / (x - 1) = (y^3 + 2 * y^2 - 1) / (y^3 + 2 * y^2 - 2)) : 
  x = y^3 + 2 * y^2 - 1 :=
by
  sorry

end determine_x_l552_552147


namespace main_theorem_l552_552875

variables {A B C P O O_A O_B O_C : Type}
variables [acute_triangle ABC] [is_circumcircle ABC O] [point_on_circumcircle_but_not_vertex P ABC O]

def is_circumcenter (X Y Z : Type) (P : Type) := sorry
def perpendicular_to_side (X Y Z : Type) (P l : Type) := sorry

-- Conditions
hypothesis h1 : is_circumcenter A O P O_A
hypothesis h2 : is_circumcenter B O P O_B
hypothesis h3 : is_circumcenter C O P O_C

hypothesis h4 : perpendicular_to_side B C A O_A l_A
hypothesis h5 : perpendicular_to_side C A B O_B l_B
hypothesis h6 : perpendicular_to_side A B C O_C l_C

-- Theorem
noncomputable def circumcircle_tangent_to_OP (l_A l_B l_C : Type) : Prop :=
  ∃ (L_A L_B L_C : Type), 
    -- Points L_A, L_B, L_C are the intersections of corresponding lines
    (intersection_points L_A L_B L_C (l_A, l_B, l_C)) →
    (circumcircle {L_A, L_B, L_C}) → 
    tangent (circumcircle {intersection_points}) OP

theorem main_theorem : circumcircle_tangent_to_OP l_A l_B l_C := 
  sorry

end main_theorem_l552_552875


namespace perpendicular_line_and_parallel_planes_l552_552523

-- Definitions for the planes and line
variable (α β : Plane) (m : Line)

-- Conditions from the problem
variable (h1 : m ∥ α) (h2 : m ⟂ α) (h3 : m ⊆ α) (h4 : α ∥ β)

-- Theorem statement
theorem perpendicular_line_and_parallel_planes (h2 : m ⟂ α) (h4 : α ∥ β) : m ⟂ β :=
  sorry

end perpendicular_line_and_parallel_planes_l552_552523


namespace area_of_triangle_is_3_over_4_l552_552279

noncomputable def area_of_triangle_tangent_lines : ℝ :=
  let p₁ := (1, 1)
  let line₁ := λ x, -x + 2
  let line₂ := λ x, 2 * x - 1
  let x_intercept₁ := 2
  let x_intercept₂ := 1 / 2
  let vertices := (p₁.1, p₁.2, 2, 0, 1/2, 0)
  let area := (1 / 2) * (2 - 1 / 2) * 1
  area

theorem area_of_triangle_is_3_over_4 :
  area_of_triangle_tangent_lines = 3 / 4 :=
by
  sorry

end area_of_triangle_is_3_over_4_l552_552279


namespace f_constant_l552_552202
open Function

namespace ProofProblem

def X := { n : ℕ // n ≥ 8 }

def f (x : X) : X := sorry

theorem f_constant (f : X → X)
  (h1 : ∀ x y : ℕ, x ≥ 4 → y ≥ 4 → f ⟨x + y, _⟩ = f ⟨x * y, _⟩)
  (h2 : f ⟨8, sorry⟩ = ⟨9, sorry⟩) :
  f ⟨9, sorry⟩ = ⟨9, sorry⟩ :=
sorry

end ProofProblem

end f_constant_l552_552202


namespace sum_ratio_product_points_on_line_l552_552089

theorem sum_ratio_product_points_on_line (A : ℕ → ℂ) (B : ℕ → ℂ) (n : ℕ) (h : 2 ≤ n) :
  (∑ i in finset.range n, 
     ((∏ k in finset.range (n-1), (A i - B k)) / 
      (∏ j in finset.range n \ {i}, (A i - A j)))) = 1 :=
sorry

end sum_ratio_product_points_on_line_l552_552089


namespace sequence_sum_identity_l552_552316

theorem sequence_sum_identity :
  let a : ℕ → ℚ := λ n, n*(n+1)/2
  (finset.range 2018).sum (λ n, 1 / a (n+1)) = 4036/2019 := sorry

end sequence_sum_identity_l552_552316


namespace remainder_of_B_is_4_l552_552729

theorem remainder_of_B_is_4 (A B : ℕ) (h : B = A * 9 + 13) : B % 9 = 4 :=
by {
  sorry
}

end remainder_of_B_is_4_l552_552729


namespace angle_between_generatrix_and_height_l552_552379

-- Definitions for the conditions
variable (a : ℝ)  -- Edge length of the cube
variable (k : ℝ)  -- Ratio of the height of the cone to the edge of the cube

-- Necessary values and relations based on conditions
def cone_height := k * a
def distance_center_base_vertex := (a * Real.sqrt 2) / 2
def height_apex_to_cube_center := a * (k - 1)

-- Problem statement: prove the angle between the generatrix of the cone and its height
theorem angle_between_generatrix_and_height : 
  ∀ (θ : ℝ), θ = Real.arccot (Real.sqrt 2 * (k - 1)) :=
by
  sorry

end angle_between_generatrix_and_height_l552_552379


namespace coef_a5b3_in_expansion_l552_552697

theorem coef_a5b3_in_expansion (a b : ℝ) :
  (∃ c : ℝ, c = (finset.choose 8 3 : ℝ) * (-1 / 2)^3 ∧
  (a - b / 2) ^ 8 = c * a^5 * b^3 + _) :=
begin
  sorry
end

end coef_a5b3_in_expansion_l552_552697


namespace train_crosses_platform_in_26_seconds_l552_552382

def km_per_hr_to_m_per_s (km_per_hr : ℕ) : ℕ :=
  km_per_hr * 5 / 18

def train_crossing_time
  (train_speed_km_per_hr : ℕ)
  (train_length_m : ℕ)
  (platform_length_m : ℕ) : ℕ :=
  let total_distance_m := train_length_m + platform_length_m
  let train_speed_m_per_s := km_per_hr_to_m_per_s train_speed_km_per_hr
  total_distance_m / train_speed_m_per_s

theorem train_crosses_platform_in_26_seconds :
  train_crossing_time 72 300 220 = 26 :=
by
  sorry

end train_crosses_platform_in_26_seconds_l552_552382


namespace max_geq_four_ninths_sum_min_leq_quarter_sum_l552_552221

theorem max_geq_four_ninths_sum (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_discriminant : b^2 >= 4*a*c) :
  max a (max b c) >= 4 / 9 * (a + b + c) :=
by 
  sorry

theorem min_leq_quarter_sum (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_discriminant : b^2 >= 4*a*c) :
  min a (min b c) <= 1 / 4 * (a + b + c) :=
by 
  sorry

end max_geq_four_ninths_sum_min_leq_quarter_sum_l552_552221


namespace abs_diff_a_b_eq_94_l552_552498

-- Define the function τ(n) that counts the number of integer divisors of n
def tau (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n.divisors.count id)

-- Define the sum function S(n) that sums τ(k) for k from 1 to n
def S (n : ℕ) : ℕ :=
  (range(n + 1)).sum tau

-- Define a and b as described in the problem
def a := (range 1501).count (λ n => (S n) % 2 = 1)
def b := (range 1501).count (λ n => (S n) % 2 = 0)

-- Theorem to prove the difference |a - b| equals 94
theorem abs_diff_a_b_eq_94 : |a - b| = 94 :=
  by sorry

end abs_diff_a_b_eq_94_l552_552498


namespace minimum_distance_is_sqrt2_l552_552904

def f (x : ℝ) (f'0 : ℝ) : ℝ := -f'0 * Real.exp x + 2 * x

def minimum_distance (f'0 : ℝ) : ℝ :=
  let f := f 0 f'0
  let l := λ x : ℝ, x - 1
  let Q := (0, Real.exp 0) -- Coordinates of Q
  let distance (P₁ P₂ : ℝ × ℝ) : ℝ := Real.sqrt ((P₂.1 - P₁.1)^2 + (P₂.2 - P₁.2)^2)
  distance (0, f) Q

theorem minimum_distance_is_sqrt2 : minimum_distance 1 = Real.sqrt 2 := sorry

end minimum_distance_is_sqrt2_l552_552904


namespace angle_bisector_midpoint_arc_l552_552085

open EuclideanGeometry

noncomputable def midpoint_of_arc (A B C S : Point) : Prop :=
  let circumcircle := circumcircle A B C in
  S ∈ circumcircle ∧
  (bisects_angle A B C A S ∧
  (distance A S = distance B S))

theorem angle_bisector_midpoint_arc (A B C S : Point) 
  (h1 : IsTriangle A B C)
  (h2 : OnCircle (circumcircle A B C) S)
  (h3 : AngleBisector (∠BAC) A S) :
  midpoint_of_arc A B C S := 
begin
  sorry -- Proof omitted
end

end angle_bisector_midpoint_arc_l552_552085


namespace circle_center_l552_552484

theorem circle_center (x y : ℝ) (h : x^2 + 8*x + y^2 - 4*y = 16) : (x, y) = (-4, 2) :=
by 
  sorry

end circle_center_l552_552484


namespace sin_double_angle_l552_552508

theorem sin_double_angle (alpha : ℝ) (h : cos (alpha - π / 4) = sqrt 2 / 4) : sin (2 * alpha) = -3 / 4 :=
  sorry

end sin_double_angle_l552_552508


namespace watch_gains_expected_minutes_per_hour_l552_552634

def watch_gains_minutes_per_hour (degrees_per_minute : ℝ) : ℝ :=
  let degrees_extra := degrees_per_minute - 360
  let seconds_extra := degrees_extra / 6
  let minutes_extra := seconds_extra * 60 / 60
  minutes_extra

theorem watch_gains_expected_minutes_per_hour :
  watch_gains_minutes_per_hour 390 = 5 := 
by
  sorry

end watch_gains_expected_minutes_per_hour_l552_552634


namespace second_player_wins_l552_552596

theorem second_player_wins : 
  ∀ (a b c : ℝ), (a ≠ 0) → 
  (∃ (first_choice: ℝ), ∃ (second_choice: ℝ), 
    ∃ (third_choice: ℝ), 
    ((first_choice ≠ 0) → (b^2 + 4 * first_choice^2 > 0)) ∧ 
    ((first_choice = 0) → (b ≠ 0)) ∧ 
    first_choice * (first_choice * b + a) = 0 ↔ ∃ x : ℝ, a * x^2 + (first_choice + second_choice) * x + third_choice = 0) :=
by sorry

end second_player_wins_l552_552596


namespace elephant_weight_equivalence_l552_552393

-- Define the conditions as variables
def elephants := 1000000000
def buildings := 25000

-- Define the question and expected answer
def expected_answer := 40000

-- State the theorem
theorem elephant_weight_equivalence:
  (elephants / buildings = expected_answer) :=
by
  sorry

end elephant_weight_equivalence_l552_552393


namespace find_integers_for_perfect_square_l552_552055

theorem find_integers_for_perfect_square (x : ℤ) :
  (∃ k : ℤ, x * (x + 1) * (x + 7) * (x + 8) = k^2) ↔ 
  x = -9 ∨ x = -8 ∨ x = -7 ∨ x = -4 ∨ x = -1 ∨ x = 0 ∨ x = 1 :=
sorry

end find_integers_for_perfect_square_l552_552055


namespace find_s_2_l552_552615

def t (x : ℝ) : ℝ := 5 * x - 14
def s (y : ℝ) : ℝ := (y^2 + 5 * y - 4)

theorem find_s_2 : s 2 = 22.24 :=
by
  have h1 : 5 * (16 / 5 : ℝ) - 14 = 2, by
    calc
      5 * (16 / 5) - 14
      = 16 - 14 : by ring
      = 2 : by norm_num
  have h2 : s 2 = s (5 * (16 / 5 : ℝ) - 14), by rw h1
  have h3 : s 2 = (16 / 5)^2 + 5*(16 / 5) - 4, by rw [h2, s]
  have h4 : (16 / 5 : ℝ)^2 = (256 / 25), by norm_num
  have h5 : 5 * (16 / 5 : ℝ) = 16, by norm_num
  calc
    s 2 = (16 / 5 : ℝ)^2 + 5 * (16 / 5 : ℝ) - 4 : by rw h3
    ... = (256 / 25) + 16 - 4 : by rw [h4, h5]
    ... = (256 / 25) + 12 : by norm_num
    ... = (256 / 25) + (300 / 25) : by norm_num
    ... = (556 / 25) : by ring
    ... = 22.24 : by norm_num

end find_s_2_l552_552615


namespace stickers_earned_l552_552637

theorem stickers_earned (initial_stickers earned_stickers final_stickers : ℕ) (h_initial : initial_stickers = 39) (h_final : final_stickers = 61) (h_earned : final_stickers = initial_stickers + earned_stickers) : earned_stickers = 22 :=
by {
  rw [h_initial, h_final] at h_earned,
  linarith,
  sorry
}

end stickers_earned_l552_552637


namespace area_of_DEFG_l552_552155

theorem area_of_DEFG (Area_ABCD : ℝ) (trisect_AD_CD : ABCD.trisect_points(E, G)) : 
Area_DEFG = 48 := by
  have h1 : Area_ABCD = 108 := sorry
  have h2 : trisect_AD_CD := sorry
  have h3 : Area_DEFG = 48 := sorry
  show Area_DEFG = 48 from h3

end area_of_DEFG_l552_552155


namespace cost_of_first_ring_is_10000_l552_552192

theorem cost_of_first_ring_is_10000 (x : ℝ) (h₁ : x + 2*x - x/2 = 25000) : x = 10000 :=
sorry

end cost_of_first_ring_is_10000_l552_552192


namespace sequence_a6_is_63_l552_552178

def sequence (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ ∀ n, a (n + 1) = 2 * a n + 1

theorem sequence_a6_is_63 (a : ℕ → ℕ) (h : sequence a) : a 6 = 63 :=
sorry

end sequence_a6_is_63_l552_552178


namespace all_numbers_equal_l552_552679

theorem all_numbers_equal (n : ℕ) (a : Fin (2 * n + 1) → ℕ)
  (h : ∀ s : Finset (Fin (2 * n + 1)), s.card = 2 * n → 
    (∃ t : Finset (Fin (2 * n + 1)), t ⊆ s ∧ t.card = n ∧ 
      t.sum (λ i, a i) = (s \ t).sum (λ i, a i))) :
  ∃ k : ℕ, ∀ i, a i = k :=
sorry

end all_numbers_equal_l552_552679


namespace expected_value_unfair_die_l552_552503

noncomputable def probability_eight : ℚ := 1 / 3
noncomputable def probability_one_to_seven : ℚ := (2 / 3) / 7

theorem expected_value_unfair_die :
  let p8 := probability_eight,
      p1_to_7 := probability_one_to_seven,
      expected_value := (∑ i in finset.range 7, (i + 1) * p1_to_7) + 8 * p8 in
  expected_value = 16 / 3 :=
by
  sorry

end expected_value_unfair_die_l552_552503


namespace unique_solution_k_l552_552810

theorem unique_solution_k (k : ℝ) :
  (∀ x : ℝ, (x + 3) / (k * x + 2) = x) ↔ (k = -1 / 12) :=
  sorry

end unique_solution_k_l552_552810


namespace petya_sum_of_all_combinations_l552_552247

-- Define the expression with the possible placements of signs.
def petyaExpression : List (ℤ → ℤ → ℤ) :=
  [int.add, int.sub] -- Combination of possible operations at each position

-- Calculate the total number of ways to insert "+" and "-" in the expression
def number_of_combinations : ℕ := 2^5

-- Define the problem statement in Lean 4
theorem petya_sum_of_all_combinations : 
  (∑ idx in Finset.range number_of_combinations, 1) = 32 := by
  sorry

end petya_sum_of_all_combinations_l552_552247


namespace sam_new_crime_books_l552_552014

theorem sam_new_crime_books (used_adventure_books : ℝ) (used_mystery_books : ℝ) (total_books : ℝ) :
  used_adventure_books = 13.0 →
  used_mystery_books = 17.0 →
  total_books = 45.0 →
  total_books - (used_adventure_books + used_mystery_books) = 15.0 :=
by
  intros ha hm ht
  rw [ha, hm, ht]
  norm_num
  -- sorry

end sam_new_crime_books_l552_552014


namespace petya_sum_of_all_combinations_l552_552246

-- Define the expression with the possible placements of signs.
def petyaExpression : List (ℤ → ℤ → ℤ) :=
  [int.add, int.sub] -- Combination of possible operations at each position

-- Calculate the total number of ways to insert "+" and "-" in the expression
def number_of_combinations : ℕ := 2^5

-- Define the problem statement in Lean 4
theorem petya_sum_of_all_combinations : 
  (∑ idx in Finset.range number_of_combinations, 1) = 32 := by
  sorry

end petya_sum_of_all_combinations_l552_552246


namespace rotated_log2_eq_exp_neg_l552_552289

theorem rotated_log2_eq_exp_neg (x : ℝ) :
  let G := {p : ℝ × ℝ | p.2 = Real.log p.1 / Real.log 2} in
  let G_rotated := {p : ℝ × ℝ | p.1 = -Real.log p.2 / Real.log 2 ∧ p.2 > 0} in
  ∃ y, (x, y) ∈ G_rotated ↔ y = 2^(-x) :=
by 
  sorry

end rotated_log2_eq_exp_neg_l552_552289


namespace trapezoid_area_correct_l552_552339

-- Definition of the lines
def line1 (x : ℝ) : ℝ := x + 2
def line2 : ℝ := 12
def line3 : ℝ := 6
def y_axis : ℝ → ℝ := λ x, 0

-- Points of intersections
def vertex_a : ℝ × ℝ := (4, 6)
def vertex_b : ℝ × ℝ := (10, 12)
def vertex_c : ℝ × ℝ := (0, 12)
def vertex_d : ℝ × ℝ := (0, 6)

-- Lengths of bases and height
def upper_base : ℝ := 10
def lower_base : ℝ := 4
def height : ℝ := 6

-- Trapezoid area formula
def trapezoid_area (b1 b2 h : ℝ) : ℝ := (1 / 2) * (b1 + b2) * h

-- Theorem to be proven
theorem trapezoid_area_correct :
  trapezoid_area upper_base lower_base height = 42 :=
by 
  -- We'll leave the proof out
  sorry

end trapezoid_area_correct_l552_552339


namespace polynomial_expansion_equality_l552_552572

theorem polynomial_expansion_equality :
  let f := (2 * x + Real.sqrt 3) ^ 4
  let a := [f.subs x 0, f.coeff 1, f.coeff 2, f.coeff 3, f.coeff 4]
  (a[0] + a[2] + a[4]) ^ 2 - (a[1] + a[3]) ^ 2 = 1 := by
  sorry

end polynomial_expansion_equality_l552_552572


namespace monotonic_increasing_interval_l552_552466

noncomputable def is_monotonic_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x ≤ y → f x ≤ f y

theorem monotonic_increasing_interval :
  is_monotonic_increasing (λ x, 2 * sin (π / 4 - x)) (set.Icc (-5 * π / 4) (-π / 4)) :=
sorry

end monotonic_increasing_interval_l552_552466


namespace brenda_cakes_l552_552017

-- Definitions based on the given conditions
def cakes_per_day : ℕ := 20
def days : ℕ := 9
def total_cakes_baked : ℕ := cakes_per_day * days
def cakes_sold : ℕ := total_cakes_baked / 2
def cakes_left : ℕ := total_cakes_baked - cakes_sold

-- Formulate the theorem
theorem brenda_cakes : cakes_left = 90 :=
by {
  -- To skip the proof steps
  sorry
}

end brenda_cakes_l552_552017


namespace cos_pi_div_4_add_alpha_l552_552883

variable (α : ℝ)

theorem cos_pi_div_4_add_alpha (h : Real.sin (Real.pi / 4 - α) = Real.sqrt 2 / 2) :
  Real.cos (Real.pi / 4 + α) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_pi_div_4_add_alpha_l552_552883


namespace simplify_complex_expr_l552_552656

noncomputable def z1 : ℂ := (-1 + complex.I * real.sqrt 3) / 2
noncomputable def z2 : ℂ := (-1 - complex.I * real.sqrt 3) / 2

theorem simplify_complex_expr :
  z1^12 + z2^12 = 2 := 
  sorry

end simplify_complex_expr_l552_552656


namespace tangent_line_equation_l552_552861

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x * (f' 1)
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 2 * f' 1

theorem tangent_line_equation : 
    (f' 1 = -3) ∧ (f 1 = -5) ∧ (∀ x y : ℝ, y + 5 = -3 * (x - 1) → 3 * x + y + 2 = 0) :=
by
    sorry

end tangent_line_equation_l552_552861


namespace cone_lateral_surface_area_is_correct_l552_552104

noncomputable def cone_lateral_surface_area (r l : ℝ) : ℝ := π * r * l

theorem cone_lateral_surface_area_is_correct :
  ∀ (r l : ℝ), r = 3 → (π * l = 2 * π * 3) → cone_lateral_surface_area r l = 18 * π :=
by
  intros r l hr hl
  rw [cone_lateral_surface_area, hr, hl]
  norm_num
  sorry

end cone_lateral_surface_area_is_correct_l552_552104


namespace intersection_setA_setB_l552_552218

namespace SetProofs

variable {α : Type*} [LinearOrder α]

def setA : Set α := { x | -1 < x ∧ x ≤ 2 }
def setB : Set α := { x | 0 < x ∧ x < 3 }

theorem intersection_setA_setB :
  setA ∩ setB = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end SetProofs

end intersection_setA_setB_l552_552218


namespace find_k_l552_552563

variables (a b : ℝ × ℝ) (c : ℝ × ℝ)
variables (k : ℝ)
variables (h1 : a = (1,2)) (h2 : b = (0,-1)) (h3 : c = (k,-2))
variables (perp : (a.fst - 2 * b.fst, a.snd - 2 * b.snd) • c = 0)

theorem find_k : k = 8 :=
by {
  have h : (1 - 0, 2 - 2 * (-1)) = (1, 4), from sorry,
  have dot_product := (1 * k + 4 * (-2) = 0), from perpendiculiar_condition a b c k,
  sub_left 8 0 }, -- Placeholder, you can fill in with detailed steps

end find_k_l552_552563


namespace polynomial_remainder_l552_552842

noncomputable def remainder_division (f g : ℚ[X]) : ℚ[X] :=
  f % g

theorem polynomial_remainder : 
  let f := (X^4 : ℚ[X])
  let g := (X^2 + 3 * X + 2 : ℚ[X])
  remainder_division f g = -15 * X - 14 :=
by
  sorry

end polynomial_remainder_l552_552842


namespace max_colorable_points_l552_552512

theorem max_colorable_points (n : ℕ) (k : ℕ) (m : ℕ) (hn : n = 100) (hk : k = 50)
  (H : ∀ (pts : Fin n → Prop) (Hpts : ∀ i j, i ≠ j → pts i ≠ pts j)
          (colors : Fin n → ℕ) (hcolors : ∀ i, colors i = 0 ∨ colors i = 1)
          (hcolored_k_pts : ∃ Kis : Finset (Fin n), Kis.card = k
                                ∧ ∀ i, i ∈ Kis → colors i = 0 ∨ colors i = 1),
          ∃ remaining_colors : Fin (n - k) → ℕ,
          (∀ (i j : Fin (n - k)), i ≠ j → remaining_colors i = remaining_colors j) 
          ∧ Φ colors remaining_colors) :
  prove ∃ remaining_colors : Fin (n - k) → ℕ, 
        (∀ (i j : Fin (n - k)), i ≠ j → remaining_colors i = remaining_colors j) 
        ∧ Φ colors remaining_colors :=
sorry

end max_colorable_points_l552_552512


namespace probability_2_lt_X_le_4_l552_552555

variable (X : ℕ → ℝ)
variable (k : ℕ)

def probability_distribution (k : ℕ) : ℝ := 1 / 3^k

theorem probability_2_lt_X_le_4 :
  (P : ℕ → ℝ) → (∀ k, P k = probability_distribution k) → 
  P (2 < X k ≤ 4) = (P 3 + P 4) :=
by 
  -- Given: P(X=k) = 1 / 3^k for k ≥ 1
  -- Prove: P(2 < X ≤ 4) = 4 / 81
  sorry

end probability_2_lt_X_le_4_l552_552555


namespace sum_of_exponents_of_binary_representation_of_2023_l552_552051

theorem sum_of_exponents_of_binary_representation_of_2023 :
  ∃ exps : List ℕ, (∀ i j, i ≠ j → i ∈ exps → j ∈ exps → 2^i ≠ 2^j) ∧ (∑ i in exps, 2^i = 2023) ∧ (∑ i in exps, i = 48) :=
sorry

end sum_of_exponents_of_binary_representation_of_2023_l552_552051


namespace eggs_remainder_l552_552453

def daniel_eggs := 53
def eliza_eggs := 68
def fiona_eggs := 26
def george_eggs := 47
def total_eggs := daniel_eggs + eliza_eggs + fiona_eggs + george_eggs

theorem eggs_remainder :
  total_eggs % 15 = 14 :=
by
  sorry

end eggs_remainder_l552_552453


namespace petya_sum_l552_552233

theorem petya_sum : 
  let f (signs : fin 5 → bool) : ℤ :=
    1 + (if signs 0 then 2 else -2) + (if signs 1 then 3 else -3) + (if signs 2 then 4 else -4) + (if signs 3 then 5 else -5) + (if signs 4 then 6 else -6),
  sum (f '' (finset.univ : finset (fin 5 → bool))) = 32 :=
by
  sorry

end petya_sum_l552_552233


namespace midpoint_on_HA1_l552_552758

noncomputable def midpoint (P Q : Point) : Point := sorry
noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def diametrically_opposite (circ : Circle) (P : Point) : Point := sorry

theorem midpoint_on_HA1 {A B C : Point} 
  (H : Point := orthocenter A B C)
  (circ : Circle := circumcircle A B C)
  (A1 : Point := diametrically_opposite circ A)
  (M_BC : Point := midpoint B C) :
  (M_BC ∈ segment H A1) ∧ (2 * dist M_BC H = dist H A1) := 
sorry

end midpoint_on_HA1_l552_552758


namespace geometric_sequence_problem_l552_552517

noncomputable def geometric_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ m : ℕ, a (m+1) = a 1 * (1/2)^(m+1)

def sum_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop :=
  S n = (1 - (1/2)^(n-1))

def arithmetic_property (a : ℕ → ℝ) (S : ℕ → ℝ): Prop :=
  2 * (a 5 + S 5) = (a 4 + S 4) + (a 6 + S 6)

def inserted_terms_sum (b : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop :=
  b n = (3 / 4) * (3/2)^(n)
  
def sum_inserted_terms (T : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : Prop :=
  T n = (9 / 4) * (1 - (3/2)^n)

theorem geometric_sequence_problem :
  ∃ a S b T : ℕ → ℝ,
    geometric_sequence a 1 ∧
    sum_geometric_sequence S a ∧
    arithmetic_property a S ∧
    inserted_terms_sum b a ∧
    sum_inserted_terms T b :=
by sorry

end geometric_sequence_problem_l552_552517


namespace number_of_valid_combinations_l552_552142

theorem number_of_valid_combinations : 
  (finset.card (finset.filter (λ x : ℕ, 10 ≤ x ∧ x ≤ 100) 
    (finset.image (λ p : ℕ × ℕ, p.1 * 10 + p.2)
      (finset.filter (λ p : ℕ × ℕ, p.1 ≠ p.2) 
        (({1, 7, 9} : finset ℕ).product ({1, 7, 9} : finset ℕ)))))) = 6 :=
by
  sorry

end number_of_valid_combinations_l552_552142


namespace sequence_sum_2022_l552_552520

theorem sequence_sum_2022 (n : ℕ) (h_n : n = 2022) 
    (x : ℕ → ℝ) (h0 : x 0 = 1 / n)
    (h_rec : ∀ k, 1 ≤ k ∧ k < n → x k = (1 / (n - k)) * ∑ i in finset.range k, x i): 
    (∑ i in finset.range n, x i) = 1 :=
by {
  sorry
}

end sequence_sum_2022_l552_552520


namespace complex_magnitude_l552_552888

-- Given conditions
def i : ℂ := complex.I
def z : ℂ := -1 + complex.I * sqrt 2

-- Define the specific problem for Lean
theorem complex_magnitude :
  (i * z = sqrt 2 - i) →
  abs z = sqrt 3 :=
by
  sorry

end complex_magnitude_l552_552888


namespace volume_of_cylinders_correct_l552_552846

noncomputable def volume_of_cylinders (R : ℝ) (hR : R > 0) : ℝ :=
  let integrand (x : ℝ) : ℝ := R^2 - x^2
  8 * ∫ x in 0..R, integrand x

theorem volume_of_cylinders_correct (R : ℝ) (hR : R > 0) : 
  volume_of_cylinders R hR = (16 / 3) * R^3 := 
by
  sorry

end volume_of_cylinders_correct_l552_552846


namespace range_of_a_l552_552887

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

theorem range_of_a {f : ℝ → ℝ} (h_even : even_function f) 
  (h_mono : monotone_increasing f (Set.Iic 0)) 
  (h_ineq : ∀ a, f (2 ^ (Real.log 3 a)) > f (-Real.sqrt 2)) :
  ∀ a, 0 < a ∧ a < Real.sqrt 3 :=
sorry

end range_of_a_l552_552887


namespace hyperbola_equation_points_on_hyperbola_l552_552080

noncomputable def hyperbola_a : ℝ := 2 * Real.sqrt 3
noncomputable def hyperbola_b : ℝ := Real.sqrt 3
noncomputable def hyperbola_eq (x y : ℝ) : Prop := (x^2 / 12) - (y^2 / 3) = 1

def line_eq (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * x - 2

def intersect_x_values : Set ℝ :=
  {x : ℝ | ∃ y, hyperbola_eq x y ∧ y = line_eq x}

theorem hyperbola_equation :
  (∀ a b : ℝ, 
  ∀x y : ℝ, 
  (x^2 / a^2 - y^2 / b^2 = 1) ∧ a = hyperbola_a ∧ b = hyperbola_b -> hyperbola_eq x y)
:= sorry

theorem points_on_hyperbola (A B : ℝ × ℝ) (C : ℝ × ℝ) (m : ℝ) :
  (∃ x1 x2 : ℝ, 
  (x1 + x2 = 16 * Real.sqrt 3 
  ∧ ∃ y1 y2 : ℝ, y1 + y2 = 12 
  ∧ A = (x1, y1) ∧ B = (x2, y2) 
  ∧ intersect_x_values = {x1, x2} 
  ∧ y1 = line_eq x1 
  ∧ y2 = line_eq x2 
  ∧ C = (4 * Real.sqrt 3, 3)
  ∧ m = 4))
:= sorry

end hyperbola_equation_points_on_hyperbola_l552_552080


namespace largest_t_value_l552_552636

noncomputable def temperature : ℝ → ℝ :=
  λ t, -t^2 + 10 * t + 60

theorem largest_t_value (t : ℝ) (ht_eq : temperature t = 80) : t = 5 + 3 * Real.sqrt 5 :=
by sorry

end largest_t_value_l552_552636


namespace sum_of_squares_eq_three_l552_552891

theorem sum_of_squares_eq_three
  (a b s : ℝ)
  (h₀ : a ≠ b)
  (h₁ : a * s^2 + b * s + b = 0)
  (h₂ : a * (1 / s)^2 + a * (1 / s) + b = 0)
  (h₃ : s * (1 / s) = 1) :
  s^2 + (1 / s)^2 = 3 := 
sorry

end sum_of_squares_eq_three_l552_552891


namespace min_value_l552_552701

noncomputable def f (a : ℝ) (x : ℝ) := a * sin x + cos x

theorem min_value (a : ℝ) (φ : ℝ) 
  (h₀ : a < 0)
  (h₁ : ∃ k : ℤ, φ = 2 * (atan (1 / a)) - π / 2 - k * π ∨ φ = 2 * (atan (1 / a)) - π / 2)
  : 2 * sin (2 * φ) - a - 1 / a = 4 :=
sorry

end min_value_l552_552701


namespace square_GH_squared_l552_552677

theorem square_GH_squared :
  ∀ (A B C D G H : Point) (side length bg dh ag ch : ℝ),
  is_square A B C D side →
  bg = 7 → dh = 7 → ag = 17 → ch = 17 → 
  external_to_square A B C D G H →
  GH_squared A B C D G H = 98 :=
by
  sorry

end square_GH_squared_l552_552677


namespace simplify_complex_expr_l552_552654

noncomputable def z1 : ℂ := (-1 + complex.I * real.sqrt 3) / 2
noncomputable def z2 : ℂ := (-1 - complex.I * real.sqrt 3) / 2

theorem simplify_complex_expr :
  z1^12 + z2^12 = 2 := 
  sorry

end simplify_complex_expr_l552_552654


namespace sum_of_distinct_digits_l552_552066

theorem sum_of_distinct_digits:
  ∃ (a b c d e : ℕ),
  a + b = 13 ∧
  c + d + e = 10 ∧
  b = d ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  a ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : finset ℕ) ∧
  b ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : finset ℕ) ∧
  c ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : finset ℕ) ∧
  d ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : finset ℕ) ∧
  e ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : finset ℕ) ∧
  a + b + c + d + e = 18 :=
by
  sorry

end sum_of_distinct_digits_l552_552066


namespace smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7_l552_552345

theorem smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7 :
  ∃ n : ℕ, n % 45 = 0 ∧ (n - 100) % 7 = 0 ∧ n = 135 :=
sorry

end smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7_l552_552345


namespace convert_to_cylindrical_l552_552036

noncomputable def cylindrical_coordinates_conversion 
  (x y z : ℝ) (r θ : ℝ) (h_r_pos : r > 0) (h_θ_range : 0 ≤ θ ∧ θ < 2 * Real.pi) : Prop :=
  r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan2 y x ∧ z = z

theorem convert_to_cylindrical : 
  cylindrical_coordinates_conversion (-3) 0 4 3 Real.pi 
  (by norm_num) (by norm_num; split; norm_num) :=
by
  sorry

end convert_to_cylindrical_l552_552036


namespace value_of_a_linear_l552_552934

theorem value_of_a_linear (a : ℝ) : (∃ x y : ℝ, y = (a-2) * x^(Real.abs a - 1) + 4 ∧ linear) → a = -2 :=
by
  sorry

end value_of_a_linear_l552_552934


namespace incorrect_angle_statements_l552_552347

/--
Angle definitions and ranges:
- An obtuse angle is strictly between \(90^{\circ}\) and \(180^{\circ}\).
- An angle in the first quadrant is between \(0^{\circ}\) and \(90^{\circ}\) (mod \(360^{\circ}\)).
- An angle in the second quadrant is between \(90^{\circ}\) and \(180^{\circ}\) (mod \(360^{\circ}\)).
- An angle in the third quadrant is between \(180^{\circ}\) and \(270^{\circ}\) (mod \(360^{\circ}\)).
- An angle in the fourth quadrant is between \(270^{\circ}\) and \(360^{\circ}\) (mod \(360^{\circ}\)).
--/
theorem incorrect_angle_statements :
  ¬ (∀ θ : ℝ, (90 < θ ∧ θ < 180 ↔ θ ∈ second_quadrant)) ∧
  ¬ (∀ θ1 θ2 : ℝ, θ1 ∈ second_quadrant → θ2 ∈ first_quadrant → θ1 > θ2) ∧
  ¬ (∀ θ : ℝ, θ > 90 → θ ∈ obtuse_angles) ∧
  ¬ ((-165 : ℝ) ∈ second_quadrant) :=
by
  -- Definitions of quadrants and obtuse angles
  let first_quadrant θ := 0 ≤ θ ∧ θ < 90 ∨ ∃ k : ℤ, k ≠ 0 ∧ k * 360 ≤ θ ∧ θ < k * 360 + 90
  let second_quadrant θ := 90 ≤ θ ∧ θ < 180 ∨ ∃ k : ℤ, k ≠ 0 ∧ k * 360 + 90 ≤ θ ∧ θ < k * 360 + 180
  let third_quadrant θ := 180 ≤ θ ∧ θ < 270 ∨ ∃ k : ℤ, k ≠ 0 ∧ k * 360 + 180 ≤ θ ∧ θ < k * 360 + 270
  let fourth_quadrant θ := 270 ≤ θ ∧ θ < 360 ∨ ∃ k : ℤ, k ≠ 0 ∧ k * 360 + 270 ≤ θ ∧ θ < k * 360 + 360
  let obtuse_angles θ := 90 < θ ∧ θ < 180
  
  -- The incorrect statements to prove
  have h1 : ¬ (∀ θ : ℝ, (90 < θ ∧ θ < 180 ↔ second_quadrant θ)),
  { sorry },
  
  have h2 : ¬ (∀ θ1 θ2 : ℝ, second_quadrant θ1 → first_quadrant θ2 → θ1 > θ2),
  { sorry },
  
  have h3 : ¬ (∀ θ : ℝ, θ > 90 → obtuse_angles θ),
  { sorry },
  
  have h4 : ¬ ((-165 : ℝ) ∈ second_quadrant),
  { sorry },
  
  exact ⟨h1, h2, h3, h4⟩

end incorrect_angle_statements_l552_552347


namespace cos_555_deg_proof_l552_552492

theorem cos_555_deg_proof : 
  (555 = 360 + 195) ∧ 
  (cos 555 = cos 195) ∧ 
  (195 = 180 + 15) ∧ 
  (cos 195 = -cos 15) ∧ 
  (cos 15 = cos (45 - 30)) ∧ 
  (cos (45 - 30) = (√6 / 4 + √2 / 4)) → 
  cos 555 = -(√6 / 4 + √2 / 4) :=
by
  sorry

end cos_555_deg_proof_l552_552492


namespace monotonic_intervals_f_leq_zero_inequality_a_eq_1_l552_552126

variables {a m n : ℝ} (f : ℝ → ℝ) (g : ℝ → ℝ)

noncomputable def f := λ x, a * x - Real.exp x + 1
noncomputable def g := λ a, a * Real.log a - a + 1

-- (1) Determine monotonic intervals of f(x)
theorem monotonic_intervals (h : a ∈ ℝ) :
  (a ≤ 0 → ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2) ∧
  (a > 0 → (∀ x1 x2 : ℝ, x1 ∈ Ioo (-∞) (Real.log a) → x2 ∈ Ioo (-∞) (Real.log a) → x1 < x2 → f x1 < f x2) ∧
           (∀ x1 x2 : ℝ, x1 ∈ Ioo (Real.log a) (∞) → x2 ∈ Ioo (Real.log a) (∞) → x1 < x2 → f x1 > f x2)) := sorry

-- (2) Prove if f(x) ≤ 0 for all x ∈ ℝ, then a ∈ (0, ∞)
theorem f_leq_zero (h : ∀ x : ℝ, f x ≤ 0) : 0 < a ∧ a ∈ Ioo 0 +∞ := sorry

-- (3) Prove inequality for a = 1
theorem inequality_a_eq_1 (h0 : 0 < m) (h1 : m < n) :
  1 / n - 1 < (f (Real.log n) - f (Real.log m)) / (n - m) ∧
  (f (Real.log n) - f (Real.log m)) / (n - m) < 1 / m - 1 := sorry

end monotonic_intervals_f_leq_zero_inequality_a_eq_1_l552_552126


namespace john_toy_store_fraction_l552_552567

theorem john_toy_store_fraction :
  let allowance := 4.80
  let arcade_spent := 3 / 5 * allowance
  let remaining_after_arcade := allowance - arcade_spent
  let candy_store_spent := 1.28
  let toy_store_spent := remaining_after_arcade - candy_store_spent
  (toy_store_spent / remaining_after_arcade) = 1 / 3 := by
    sorry

end john_toy_store_fraction_l552_552567


namespace geometric_series_terms_l552_552109

theorem geometric_series_terms 
    (b1 q : ℝ)
    (h₁ : (b1^2 / (1 + q + q^2)) = 12)
    (h₂ : (b1^2 / (1 + q^2)) = (36 / 5)) :
    (b1 = 3 ∨ b1 = -3) ∧ q = -1/2 :=
by
  sorry

end geometric_series_terms_l552_552109


namespace polynomial_remainder_l552_552843

noncomputable def remainder_division (f g : ℚ[X]) : ℚ[X] :=
  f % g

theorem polynomial_remainder : 
  let f := (X^4 : ℚ[X])
  let g := (X^2 + 3 * X + 2 : ℚ[X])
  remainder_division f g = -15 * X - 14 :=
by
  sorry

end polynomial_remainder_l552_552843


namespace total_cans_in_display_l552_552164

theorem total_cans_in_display :
  ∃ (r : ℕ → ℕ), 
  (r 7 = 19) ∧ 
  (∀ i : ℕ, 1 ≤ i ∧ i < 9 → r (i + 1) = r i + 3) ∧ 
  (∑ i in finset.range 9, (r (i + 1)) = 117) :=
sorry

end total_cans_in_display_l552_552164


namespace find_compound_interest_rate_l552_552718

-- Definitions based on conditions
def SimpleInterest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := (P * R * T) / 100
def CompoundInterest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * ((1 + R / 100)^T - 1)

-- Given conditions
def P_simple := 1750.0000000000018
def T_simple := 3
def R_simple := 8

def P_compound := 4000
def T_compound := 2

-- Half relation between simple and compound interests
lemma half_relation (si : ℝ) (ci : ℝ) : ci = 2 * si := by sorry

-- Main goal
theorem find_compound_interest_rate (R : ℝ) :
  let si := SimpleInterest P_simple R_simple T_simple in
  let ci := CompoundInterest P_compound R T_compound in
  half_relation si ci → R = 10 :=
  by
    intros si ci h
    sorry

end find_compound_interest_rate_l552_552718


namespace power_multiplication_l552_552072

theorem power_multiplication (x y : ℝ) (hx : 3^x = 6) (hy : 3^y = 9) : 3^(x+y) = 54 := by
  sorry

end power_multiplication_l552_552072


namespace problem_solution_l552_552931

theorem problem_solution (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3) →
  (a₁ + a₂ + a₃ = 19) :=
by
  -- Given condition: for any real number x, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3
  -- We need to prove: a₁ + a₂ + a₃ = 19
  sorry

end problem_solution_l552_552931


namespace point_in_fourth_quadrant_l552_552893

theorem point_in_fourth_quadrant (a : ℝ) (h1 : a + 1 > 0) (h2 : 2a - 3 < 0) : -1 < a ∧ a < 3/2 :=
by
  sorry

end point_in_fourth_quadrant_l552_552893


namespace angle_C_is_pi_div_6_f_range_l552_552161

-- Defining the problem in Lean

/- Given conditions -/
variables {A B C : ℝ} {a b c : ℝ}
variable h1 : Real.tan A = -3 * Real.tan B
variable h2 : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * b

/- Problem (1): Proving the measure of angle C -/
theorem angle_C_is_pi_div_6 
  (h1 : Real.tan A = -3 * Real.tan B)
  (h2 : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * b) :
  C = Real.pi / 6 := 
sorry

/- Problem (2): Finding the range of function f(x) -/
def f (x : ℝ) : ℝ := Real.sin (x + 2 * Real.pi / 3) + Real.cos (x + Real.pi / 6) ^ 2

theorem f_range {x : ℝ} (hx : x ∈ set.Icc 0 (5 * Real.pi / 6)) :
  ∀ y ∈ set.Icc 0 (5 * Real.pi / 6), f y ∈ set.Icc (-(1 : ℝ)) ((3 * Real.sqrt 3 + 2) / 4) :=
sorry

end angle_C_is_pi_div_6_f_range_l552_552161


namespace slower_train_pass_time_l552_552735

theorem slower_train_pass_time :
  ∀ (l : ℝ) (v1 v2 : ℝ), l = 750 → v1 = 60 * (1000 / 3600) → v2 = 35 * (1000 / 3600) →
  l / (v1 + v2) = 56.84 :=
by
  intros l v1 v2 h1 h2 h3
  rw [h1, h2, h3]
  have rel_speed : v1 + v2 = 26.39 := by norm_num -- verify the relative speed calculation
  have dist : 2 * l = 1500 := by norm_num -- verify the total distance calculation
  rw [rel_speed, dist]
  norm_num -- final verification of time taken
  sorry -- temporarily skip the detailed step proof if needed

end slower_train_pass_time_l552_552735


namespace sum_of_17th_roots_of_unity_except_1_l552_552807

theorem sum_of_17th_roots_of_unity_except_1 :
  Complex.exp (2 * Real.pi * Complex.I / 17) +
  Complex.exp (4 * Real.pi * Complex.I / 17) +
  Complex.exp (6 * Real.pi * Complex.I / 17) +
  Complex.exp (8 * Real.pi * Complex.I / 17) +
  Complex.exp (10 * Real.pi * Complex.I / 17) +
  Complex.exp (12 * Real.pi * Complex.I / 17) +
  Complex.exp (14 * Real.pi * Complex.I / 17) +
  Complex.exp (16 * Real.pi * Complex.I / 17) +
  Complex.exp (18 * Real.pi * Complex.I / 17) +
  Complex.exp (20 * Real.pi * Complex.I / 17) +
  Complex.exp (22 * Real.pi * Complex.I / 17) +
  Complex.exp (24 * Real.pi * Complex.I / 17) +
  Complex.exp (26 * Real.pi * Complex.I / 17) +
  Complex.exp (28 * Real.pi * Complex.I / 17) +
  Complex.exp (30 * Real.pi * Complex.I / 17) +
  Complex.exp (32 * Real.pi * Complex.I / 17) = 0 := sorry

end sum_of_17th_roots_of_unity_except_1_l552_552807


namespace smallest_integer_odd_sequence_l552_552293

/-- Given the median of a set of consecutive odd integers is 157 and the greatest integer in the set is 171,
    prove that the smallest integer in the set is 149. -/
theorem smallest_integer_odd_sequence (median greatest : ℤ) (h_median : median = 157) (h_greatest : greatest = 171) :
  ∃ smallest : ℤ, smallest = 149 :=
by
  sorry

end smallest_integer_odd_sequence_l552_552293


namespace roots_order_l552_552927

theorem roots_order {a b m n : ℝ} (h1 : m < n) (h2 : a < b)
  (hm : 1 - (m - a) * (m - b) = 0) (hn : 1 - (n - a) * (n - b) = 0) :
  m < a ∧ a < b ∧ b < n :=
sorry

end roots_order_l552_552927


namespace sequence_sum_l552_552525

theorem sequence_sum :
  ∃ (a : ℕ → ℝ), (∀ n, a n * a (n + 1) * a (n + 2) * a (n + 3) = 24) ∧
  a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 3 ∧ 
  ∑ i in finset.range 2014, a i = 6037 :=
begin
  sorry
end

end sequence_sum_l552_552525


namespace range_of_a_l552_552909

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log (1 / 2)) + x

noncomputable def g (x a : ℝ) : ℝ :=
  if x < 0 then x^2 + 2 * a else Real.arccos x + 2

theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ set.Ici 2, ∃ x2 : ℝ, f x1 = g x2 a) ↔ a ∈ set.Iic (1 / 2) ∪ set.Icc 1 2 :=
sorry

end range_of_a_l552_552909


namespace eval_log_cuberoot_seven_l552_552048

theorem eval_log_cuberoot_seven : log 7 (7^(1/3 : ℝ)) = 1/3 :=
by
  sorry

end eval_log_cuberoot_seven_l552_552048


namespace fixed_point_exists_line_intersects_circle_shortest_chord_l552_552565

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25
noncomputable def line_l (x y m : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem fixed_point_exists : ∃ P : ℝ × ℝ, (∀ m : ℝ, line_l P.1 P.2 m) ∧ P = (3, 1) :=
by
  sorry

theorem line_intersects_circle : ∀ m : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ, line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
by
  sorry

theorem shortest_chord : ∃ m : ℝ, m = -3/4 ∧ (∀ x y, line_l x y m ↔ 2 * x - y - 5 = 0) :=
by
  sorry

end fixed_point_exists_line_intersects_circle_shortest_chord_l552_552565


namespace triangle_transform_same_l552_552408

def Point := ℝ × ℝ

def reflect_x (p : Point) : Point :=
(p.1, -p.2)

def rotate_180 (p : Point) : Point :=
(-p.1, -p.2)

def reflect_y (p : Point) : Point :=
(-p.1, p.2)

def transform (p : Point) : Point :=
reflect_y (rotate_180 (reflect_x p))

theorem triangle_transform_same (A B C : Point) :
A = (2, 1) → B = (4, 1) → C = (2, 3) →
(transform A = (2, 1) ∧ transform B = (4, 1) ∧ transform C = (2, 3)) :=
by
  intros
  sorry

end triangle_transform_same_l552_552408


namespace trains_crossing_time_l552_552753

noncomputable def length_first_train : ℝ := 120
noncomputable def length_second_train : ℝ := 160
noncomputable def speed_first_train_kmph : ℝ := 60
noncomputable def speed_second_train_kmph : ℝ := 40
noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

noncomputable def speed_first_train : ℝ := kmph_to_mps speed_first_train_kmph
noncomputable def speed_second_train : ℝ := kmph_to_mps speed_second_train_kmph
noncomputable def relative_speed : ℝ := speed_first_train + speed_second_train
noncomputable def total_distance : ℝ := length_first_train + length_second_train
noncomputable def crossing_time : ℝ := total_distance / relative_speed

theorem trains_crossing_time :
  crossing_time = 10.08 := by
  sorry

end trains_crossing_time_l552_552753


namespace expression_evaluates_to_one_l552_552276

theorem expression_evaluates_to_one
  (x y z : ℤ)
  (h1 : x > y)
  (h2 : y > 1)
  (h3 : x > 1)
  (hx_int : x > 0) -- Ensure x, y are natural numbers (strictly positive integers)
  (hy_int : y > 0)
  (hz_def : z = x - y) :
  (x ^ (z + y) * y ^ x) / (y ^ (z + y) * x ^ x) = 1 :=
by
  sorry

end expression_evaluates_to_one_l552_552276


namespace sum_x_max_sufficient_sum_for_n_l552_552530

-- The conditions for our problem.
variables (n : ℕ) (x : ℕ → ℝ)
hypothesis h_cond : (∀ i : ℕ, i < n → x i ≥ 0) ∧ 
  (∑ i in finset.range n, (x i)^2 + ∑ i in finset.range n, ∑ j in finset.range n, if i < j then (x i * x j)^2 else 0 = n * (n + 1) / 2)

-- Part 1: Find the maximum value of the sum of x_i
theorem sum_x_max (n : ℕ) (x : ℕ → ℝ) (h_cond) : ∑ i in finset.range n, x i ≤ n := sorry

-- Part 2: Determine all positive integers n such that the sum is sufficient
theorem sufficient_sum_for_n (x : ℕ → ℝ) (h_cond) : ∀ n, (∑ i in finset.range n, x i ≥ real.sqrt (n * (n + 1) / 2)) ↔ (n = 1 ∨ n = 2 ∨ n = 3) := sorry

end sum_x_max_sufficient_sum_for_n_l552_552530


namespace find_p_l552_552936

theorem find_p 
  (p q x y : ℤ)
  (h1 : p * x + q * y = 8)
  (h2 : 3 * x - q * y = 38)
  (hx : x = 2)
  (hy : y = -4) : 
  p = 20 := 
by 
  subst hx
  subst hy
  sorry

end find_p_l552_552936


namespace flour_needed_for_dozen_cookies_l552_552226

/--
Matt uses 4 bags of flour, each weighing 5 pounds, to make a total of 120 cookies.
Prove that 2 pounds of flour are needed to make a dozen cookies.
-/
theorem flour_needed_for_dozen_cookies :
  ∀ (bags_of_flour : ℕ) (weight_per_bag : ℕ) (total_cookies : ℕ),
  bags_of_flour = 4 →
  weight_per_bag = 5 →
  total_cookies = 120 →
  (12 * (bags_of_flour * weight_per_bag)) / total_cookies = 2 :=
by
  sorry

end flour_needed_for_dozen_cookies_l552_552226


namespace numbers_combination_to_24_l552_552337

theorem numbers_combination_to_24 :
  (40 / 4) + 12 + 2 = 24 :=
by
  sorry

end numbers_combination_to_24_l552_552337


namespace henry_distance_fixed_point_l552_552141

noncomputable def distance_gym_home : ℝ := 3
noncomputable def fractional_distance : ℝ := ⅔

def recursive_home (a : ℝ) : ℝ := (1 / 3) * a + 2
def recursive_gym (b : ℝ) : ℝ := (1 / 3) * b + 2

theorem henry_distance_fixed_point :
∃ A B : ℝ, 
  (
    A = distance_gym_home * (1 - fractional_distance ^ 2)⁻¹ 
    ∧ B = distance_gym_home * (1 - fractional_distance) * (1 - fractional_distance ^ 2)⁻¹ 
    ∧ |A - B| = ⅚
  ) := 
sorry

end henry_distance_fixed_point_l552_552141


namespace inequality_a_n_l552_552526

noncomputable def a_sequence : ℕ → ℝ
| 1 := 21 / 16
| (n + 1) := ((3 : ℝ) / 2^(n + 2) + 3 * a_sequence n) / 2

theorem inequality_a_n (n m : ℕ) (h1_m : 2 ≤ m) (h2_n : n ≤ m) :
  (a_sequence n + 3 / 2^(n + 3))^(1 / m) * (m - (2 / 3)^m) < (m^2 - 1) / (m - n + 1) :=
sorry

end inequality_a_n_l552_552526


namespace blue_socks_count_l552_552194

theorem blue_socks_count (total_socks : ℕ) (two_thirds_white : ℕ) (one_third_blue : ℕ) 
  (h1 : total_socks = 180) 
  (h2 : two_thirds_white = (2 / 3) * total_socks) 
  (h3 : one_third_blue = total_socks - two_thirds_white) : 
  one_third_blue = 60 :=
by
  sorry

end blue_socks_count_l552_552194


namespace triangle_perimeter_12_15_18_l552_552770

/-- The perimeter of a triangle with sides 12 cm, 15 cm, and 18 cm is 45 cm. -/
theorem triangle_perimeter_12_15_18 : 
  ∀ (a b c : ℕ), a = 12 → b = 15 → c = 18 → (a + b + c = 45) :=
by intros a b c ha hb hc
   rw [ha, hb, hc]
   exact rfl

end triangle_perimeter_12_15_18_l552_552770


namespace number_of_mappings_l552_552137

theorem number_of_mappings (A B : Finset ℝ) (h₁ : A.card = 10) (h₂ : B.card = 50)
  (f : (A → B)) (hf : ∀ b ∈ B, ∃ a ∈ A, f a = b)
  (hf_sorted : ∀ a1 a2 ∈ A, a1 ≤ a2 → f a1 ≤ f a2) :
  (Finset.card (Finset.image f A)).choose (H : nat.prime_card := 149) :
  nat.choose 149 49 :=
sorry

end number_of_mappings_l552_552137


namespace area_quad_eq_half_AC_sq_sin_angleA_l552_552720

noncomputable def area_of_quad (A B C D : Point) [InscribedQuadrilateral A B C D] (h : BC = CD) : ℝ :=
  1/2 * (distance A C)^2 * sin (angle A)

theorem area_quad_eq_half_AC_sq_sin_angleA
  (A B C D : Point) 
  [InscribedQuadrilateral A B C D]
  (BC_eq_CD : BC = CD) : 
  area_of_quad A B C D BC_eq_CD = 1/2 * (distance A C)^2 * sin (angle A) :=
sorry

end area_quad_eq_half_AC_sq_sin_angleA_l552_552720


namespace find_sum_of_angles_l552_552644

-- Given points A, B, Q, D, C, P lying on a circle
def on_circle (A B Q D C P : Point) (circ : Circle) : Prop := 
  circ.contains A ∧ circ.contains B ∧ circ.contains Q ∧ circ.contains D ∧ circ.contains C ∧ circ.contains P

-- Define measures of arcs
def arc_measure (A B : Point) (m : ℝ) := 
  Arc (A, B).measure = m

-- Given arc measures
def arc_BQ_QD_AP_PC (A B Q D C P : Point) (circ : Circle) : Prop :=
  arc_measure B Q 42 ∧
  arc_measure Q D 38 ∧
  arc_measure A P 20 ∧
  arc_measure P C 40

-- Given angle X subtended by arc BP at the center
def angle_X_BP (O A B Q D C P : Point) (circ : Circle) : Prop :=
  let BP := arc_measure B P (circ.contains_arc O B P) in
  (∠ O B P).measure = (BP / 2)

-- Given angle Q subtended by arc BQD at the center
def angle_Q_BQD (O A B Q D C P : Point) (circ : Circle) : Prop :=
  let BQD := arc_measure B Q 42 + arc_measure Q D 38 in
  (∠ O B Q).measure = (BQD / 2)

-- Proof problem in Lean
theorem find_sum_of_angles (A B Q D C P : Point) (circ : Circle):
  on_circle A B Q D C P circ →
  arc_BQ_QD_AP_PC A B Q D C P circ →
  angle_X_BP O A B Q D C P circ →
  angle_Q_BQD O A B Q D C P circ →
  (30 + 40) = 70 :=
by
  sorry

end find_sum_of_angles_l552_552644


namespace walt_age_l552_552295

variable (W M P : ℕ)

-- Conditions
def condition1 := M = 3 * W
def condition2 := M + 12 = 2 * (W + 12)
def condition3 := P = 4 * W
def condition4 := P + 15 = 3 * (W + 15)

theorem walt_age (W M P : ℕ) (h1 : condition1 W M) (h2 : condition2 W M) (h3 : condition3 W P) (h4 : condition4 W P) : 
  W = 30 :=
sorry

end walt_age_l552_552295


namespace more_boys_than_girls_l552_552183

noncomputable def class1_4th_girls : ℕ := 12
noncomputable def class1_4th_boys : ℕ := 13
noncomputable def class2_4th_girls : ℕ := 15
noncomputable def class2_4th_boys : ℕ := 11

noncomputable def class1_5th_girls : ℕ := 9
noncomputable def class1_5th_boys : ℕ := 13
noncomputable def class2_5th_girls : ℕ := 10
noncomputable def class2_5th_boys : ℕ := 11

noncomputable def total_4th_girls : ℕ := class1_4th_girls + class2_4th_girls
noncomputable def total_4th_boys : ℕ := class1_4th_boys + class2_4th_boys

noncomputable def total_5th_girls : ℕ := class1_5th_girls + class2_5th_girls
noncomputable def total_5th_boys : ℕ := class1_5th_boys + class2_5th_boys

noncomputable def total_girls : ℕ := total_4th_girls + total_5th_girls
noncomputable def total_boys : ℕ := total_4th_boys + total_5th_boys

theorem more_boys_than_girls :
  (total_boys - total_girls) = 2 :=
by
  -- placeholder for the proof
  sorry

end more_boys_than_girls_l552_552183


namespace imaginary_part_l552_552219

def z : ℂ := 3 + complex.i   -- Complex number z = 3 + i

theorem imaginary_part (z : ℂ) (hz : z = 3 + complex.i) : complex.im (z + 1 / z) = 9 / 10 :=
by
  rw hz
  have h := complex.inv_eq_one_div _
  rw [h, complex.add_im, complex.im, complex.mul_im, complex.add_im, complex.im_inv]
  simp
  sorry

end imaginary_part_l552_552219


namespace periodic_odd_function_property_l552_552612

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x * (x - 1)
  else if 1 < x ∧ x ≤ 2 then Real.sin (Real.pi * x)
  else if x < 0 then -f (-x)
  else f (x - 4)

theorem periodic_odd_function_property :
  f (29 / 4) + f (41 / 6) = 11 / 16 := sorry

end periodic_odd_function_property_l552_552612


namespace compute_geometric_sum_l552_552808

open Complex

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem compute_geometric_sum : 
  let ω := omega in
  (ω ^ 1 + ω ^ 2 + ω ^ 3 + ω ^ 4 + ω ^ 5 + ω ^ 6 + 
  ω ^ 7 + ω ^ 8 + ω ^ 9 + ω ^ 10 + ω ^ 11 + ω ^ 12 + 
  ω ^ 13 + ω ^ 14 + ω ^ 15 + ω ^ 16) = -1 :=
by 
  let ω := omega
  have h : ω ^ 17 = 1 := 
    by sorry
  have h1 : ω ^ 16 = 1 / ω := 
    by sorry
  sorry

end compute_geometric_sum_l552_552808


namespace number_of_N_satisfying_condition_l552_552467

def is_solution (N : ℕ) : Prop :=
  2017 % N = 17

theorem number_of_N_satisfying_condition : 
  {N : ℕ | is_solution N}.to_finset.card = 13 :=
sorry

end number_of_N_satisfying_condition_l552_552467


namespace algebraic_expression_value_l552_552541

theorem algebraic_expression_value (a x : ℝ) (h : 3 * a - x = x + 2) (hx : x = 2) : a^2 - 2 * a + 1 = 1 :=
by {
  sorry
}

end algebraic_expression_value_l552_552541


namespace simplify_expression_l552_552660

theorem simplify_expression :
  (Complex.mk (-1) (Real.sqrt 3) / 2) ^ 12 + (Complex.mk (-1) (-Real.sqrt 3) / 2) ^ 12 = 2 := by
  sorry

end simplify_expression_l552_552660


namespace spoiled_apples_count_l552_552373

-- Definitions of the problem's conditions
def total_apples : ℕ := 7
def at_least_one_spoiled_probability : ℝ := 0.2857142857142857

-- Define the probability function
def probability_two_good_apples (G : ℕ) : ℝ :=
  (G / total_apples) * ((G - 1) / (total_apples - 1))

-- Define the proof statement
theorem spoiled_apples_count : ∃ S G : ℕ, S + G = total_apples ∧ S = 1 ∧ (1 - probability_two_good_apples G) = at_least_one_spoiled_probability :=
by
  sorry

end spoiled_apples_count_l552_552373


namespace number_of_N_satisfying_condition_l552_552470

def is_solution (N : ℕ) : Prop :=
  2017 % N = 17

theorem number_of_N_satisfying_condition : 
  {N : ℕ | is_solution N}.to_finset.card = 13 :=
sorry

end number_of_N_satisfying_condition_l552_552470


namespace divisible_by_18_count_l552_552177

noncomputable def count_divisible_ways : Nat :=
  let fixed_digits_sum := 11
  let digit_range := {d : Nat // 1 ≤ d ∧ d ≤ 9}
  let number_of_choices := 4 -- for the last digit being even (2, 4, 6, 8)
  let possible_digits_combinations := (finset.range 9).card ^ 4
  let total_ways := possible_digits_combinations * number_of_choices
  total_ways

theorem divisible_by_18_count : count_divisible_ways = 26244 := sorry

end divisible_by_18_count_l552_552177


namespace scientific_notation_correct_l552_552002

def number := 56990000

theorem scientific_notation_correct : number = 5.699 * 10^7 :=
  by
    sorry

end scientific_notation_correct_l552_552002


namespace loan_proof_l552_552811

-- Definition of the conditions
def interest_rate_year_1 : ℝ := 0.10
def interest_rate_year_2 : ℝ := 0.12
def interest_rate_year_3 : ℝ := 0.14
def total_interest_paid : ℝ := 5400

-- Theorem proving the results
theorem loan_proof (P : ℝ) 
                   (annual_repayment : ℝ)
                   (remaining_principal : ℝ) :
  (interest_rate_year_1 * P) + 
  (interest_rate_year_2 * P) + 
  (interest_rate_year_3 * P) = total_interest_paid →
  3 * annual_repayment = total_interest_paid →
  remaining_principal = P →
  P = 15000 ∧ 
  annual_repayment = 1800 ∧ 
  remaining_principal = 15000 :=
by
  intros h1 h2 h3
  sorry

end loan_proof_l552_552811


namespace isabella_exchange_l552_552189

theorem isabella_exchange (d : ℚ) : 
  (8 * d / 5 - 72 = 4 * d) → d = -30 :=
by
  sorry

end isabella_exchange_l552_552189


namespace modulus_of_z_l552_552176

open Complex

def z : ℂ := ⟨Real.cos 3, Real.sin 3⟩

theorem modulus_of_z : Complex.abs z = 1 := by
  sorry

end modulus_of_z_l552_552176


namespace total_combinations_8_coefficients_l552_552921

theorem total_combinations_8_coefficients :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ), 
    (a_0 ∈ {0, 1, 2}) ∧ (a_1 ∈ {0, 1, 2}) ∧ (a_2 ∈ {0, 1, 2}) ∧ (a_3 ∈ {0, 1, 2}) ∧
    (a_4 ∈ {0, 1, 2}) ∧ (a_5 ∈ {0, 1, 2}) ∧ (a_6 ∈ {0, 1, 2}) ∧ (a_7 ∈ {0, 1, 2}) ∧
    (∃ (x : ℕ), x = a_7 * 4^7 + a_6 * 4^6 + a_5 * 4^5 + a_4 * 4^4 + a_3 * 4^3 +
                          a_2 * 4^2 + a_1 * 4^1 + a_0 * 4^0)) →
    6561 = 3^8 :=
begin
  -- Proof can be inserted here
  sorry
end

end total_combinations_8_coefficients_l552_552921


namespace expected_value_unfair_die_l552_552502

theorem expected_value_unfair_die :
  let p1 := (1 / 6 : ℚ) in
  let p2 := (1 / 8 : ℚ) in
  let p3 := (1 / 12 : ℚ) in
  let p4 := (1 / 12 : ℚ) in
  let p5 := (1 / 12 : ℚ) in
  let p6 := (1 - (p1 + p2 + p3 + p4 + p5) : ℚ) in
  let expected_value := p1 * 1 + p2 * 2 + p3 * 3 + p4 * 4 + p5 * 5 + p6 * 6 in
  expected_value = 1.125 :=
by
  sorry

end expected_value_unfair_die_l552_552502


namespace negation_of_exists_prop_l552_552709

theorem negation_of_exists_prop :
  (¬ ∃ x : ℝ, x > 0 ∧ sqrt x ≤ x - 1) ↔ (∀ x : ℝ, x > 0 → sqrt x > x - 1) :=
by sorry

end negation_of_exists_prop_l552_552709


namespace marks_deducted_per_wrong_answer_l552_552587

theorem marks_deducted_per_wrong_answer
  (correct_awarded : ℕ)
  (total_marks : ℕ)
  (total_questions : ℕ)
  (correct_answers : ℕ)
  (incorrect_answers : ℕ)
  (final_marks : ℕ) :
  correct_awarded = 3 →
  total_marks = 38 →
  total_questions = 70 →
  correct_answers = 27 →
  incorrect_answers = total_questions - correct_answers →
  final_marks = total_marks →
  final_marks = correct_answers * correct_awarded - incorrect_answers * 1 →
  1 = 1
  := by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end marks_deducted_per_wrong_answer_l552_552587


namespace smallest_n_for_T_n_integer_l552_552999

def L : ℚ := ∑ i in {1, 2, 3, 4}, 1 / i

theorem smallest_n_for_T_n_integer : ∃ n ∈ ℕ, n > 0 ∧ (n * 5^(n-1) * L).denom = 1 ∧ n = 12 :=
by
  have hL : L = 25 / 12 := by sorry
  existsi 12
  split
  exact Nat.succ_pos'
  split
  suffices (12 * 5^(12-1) * 25 / 12).denom = 1 by sorry
  sorry
  rfl

end smallest_n_for_T_n_integer_l552_552999


namespace F5_div_641_Fermat_rel_prime_l552_552465

def Fermat_number (n : ℕ) : ℕ := 2^(2^n) + 1

theorem F5_div_641 : Fermat_number 5 % 641 = 0 := 
  sorry

theorem Fermat_rel_prime (k n : ℕ) (hk: k ≠ n) : Nat.gcd (Fermat_number k) (Fermat_number n) = 1 :=
  sorry

end F5_div_641_Fermat_rel_prime_l552_552465


namespace pastries_left_to_take_home_l552_552853

def initial_cupcakes : ℕ := 7
def initial_cookies : ℕ := 5
def pastries_sold : ℕ := 4

theorem pastries_left_to_take_home :
  initial_cupcakes + initial_cookies - pastries_sold = 8 := by
  sorry

end pastries_left_to_take_home_l552_552853


namespace find_constants_l552_552058

noncomputable section

theorem find_constants (P Q R : ℝ)
  (h : ∀ x : ℝ, x ≠ 2 → x ≠ 4 →
    (5*x^2 + 7*x) / ((x - 2) * (x - 4)^2) =
    P / (x - 2) + Q / (x - 4) + R / (x - 4)^2) :
  P = 3.5 ∧ Q = 1.5 ∧ R = 18 :=
by
  sorry

end find_constants_l552_552058


namespace max_radius_squared_l552_552335

def base_radius : ℝ := 4
def height : ℝ := 10
def intersect_to_base : ℝ := 4
def r :=  (12 * Real.sqrt 29) / 29
def r_squared := (r * r)

theorem max_radius_squared (m n : ℕ) (h1 : Nat.gcd m n = 1) (h2 : r_squared = m / n) : 
  m + n = 5017 :=
sorry

end max_radius_squared_l552_552335


namespace seating_arrangement_l552_552794

def family_members : List String := ["A", "B", "C", "D", "E", "F", "G"]

def unique_pairs (l : List String) : List (String × String) :=
  let pairs := l.product l
  pairs.filter (λ x => x.1 < x.2)

theorem seating_arrangement :
  ∃ (meals : List (List String)), 
    meals.length > 1 ∧
    (∀ pair ∈ unique_pairs family_members, 
      ∃ (meal : List String) ∈ meals, 
        meal.contains pair.1 ∧ meal.contains pair.2 ∧ 
        (meal.indexOf pair.1 + 1 = meal.indexOf pair.2 ∨
         meal.indexOf pair.1 = meal.indexOf pair.2 + 1 ∨ 
         (meal.indexOf pair.1 = 0 ∧ meal.indexOf pair.2 = meal.length - 1) ∨ 
         (meal.indexOf pair.2 = 0 ∧ meal.indexOf pair.1 = meal.length - 1))) :=
by
  sorry

end seating_arrangement_l552_552794


namespace part1_part2_l552_552106

def f (a x : ℝ) := log (2 : ℝ) ((1 - a * x) / (1 + x))
def g (a m x : ℝ) := f a x - log (2 : ℝ) (m * x)

theorem part1 (h : ∀ x : ℝ, f 1 x = -f 1 (-x)) : 1 = 1 :=
by sorry

theorem part2 (h : ∀ x : ℝ, f 1 x = -f 1 (-x)) : ¬∃ (m : ℝ), m ≠ 0 ∧ ∀ x : ℝ, g 1 m x = 0 :=
by sorry

end part1_part2_l552_552106


namespace projection_orthogonal_vectors_l552_552616

def vector2 := ℝ × ℝ

variables (a b v : vector2)

def orthogonal (u v : vector2) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def proj (u v : vector2) : vector2 :=
  let k := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  in (k * v.1, k * v.2)

theorem projection_orthogonal_vectors (a b : vector2)
  (h_orthogonal : orthogonal a b)
  (h_proj_a : proj a (4, -2) = (-4 / 5, -8 / 5)) :
  proj b (4, -2) = (24 / 5, -2 / 5) := 
sorry

end projection_orthogonal_vectors_l552_552616


namespace smiths_laundry_loads_l552_552591

theorem smiths_laundry_loads
  (kylie_towels : ℕ)
  (daughters_towels : ℕ)
  (husband_towels : ℕ)
  (washing_machine_capacity : ℕ)
  (total_towels : ℕ)
  (num_loads : ℕ) 
  (h1 : kylie_towels = 3)
  (h2 : daughters_towels = 6)
  (h3 : husband_towels = 3)
  (h4 : washing_machine_capacity = 4)
  (h5 : total_towels = kylie_towels + daughters_towels + husband_towels)
  (h6 : num_loads = total_towels / washing_machine_capacity) :
  num_loads = 3 :=
by 
  simp [h1, h2, h3, h4, h5, h6]
  norm_num

end smiths_laundry_loads_l552_552591


namespace poly_div_remainder_l552_552837

noncomputable def x : ℝ[X] := polynomial.X

def poly1 : ℝ[X] := x^4
def poly2 : ℝ[X] := x^2 + 3 * x + 2
def remainder : ℝ[X] := -18 * x - 16

theorem poly_div_remainder :
  poly1 % poly2 = remainder :=
sorry

end poly_div_remainder_l552_552837


namespace cardinals_home_runs_second_l552_552163

-- Define the conditions
def cubs_home_runs_third : ℕ := 2
def cubs_home_runs_fifth : ℕ := 1
def cubs_home_runs_eighth : ℕ := 2
def cubs_total_home_runs := cubs_home_runs_third + cubs_home_runs_fifth + cubs_home_runs_eighth
def cubs_more_than_cardinals : ℕ := 3
def cardinals_home_runs_fifth : ℕ := 1

-- Define the proof problem
theorem cardinals_home_runs_second :
  (cubs_total_home_runs = cardinals_total_home_runs + cubs_more_than_cardinals) →
  (cardinals_total_home_runs - cardinals_home_runs_fifth = 1) :=
sorry

end cardinals_home_runs_second_l552_552163


namespace find_a_if_f_odd_l552_552150

def f (x : ℝ) (a : ℝ) : ℝ := 1 / (2^x - 1) + a

theorem find_a_if_f_odd (a : ℝ) : (∀ x : ℝ, f x a = - f (-x) a) ↔ a = 1 / 2 := by
  sorry

end find_a_if_f_odd_l552_552150


namespace molly_candle_count_l552_552229

theorem molly_candle_count :
  ∃ n : ℕ, n + 6 = 20 ∧ n = 14 :=
begin
  existsi 14,
  split,
  { norm_num, },
  { refl, }
end

end molly_candle_count_l552_552229


namespace simplify_expression_l552_552435

theorem simplify_expression :
  (sqrt 5 * 5^(1/2) + 18 / 3 * 4 - 8^(3/2) + 10 - 3^2) = 30 - 16 * sqrt 2 := 
by
  sorry

end simplify_expression_l552_552435


namespace blue_socks_count_l552_552195

-- Defining the total number of socks
def total_socks : ℕ := 180

-- Defining the number of white socks as two thirds of the total socks
def white_socks : ℕ := (2 * total_socks) / 3

-- Defining the number of blue socks as the difference between total socks and white socks
def blue_socks : ℕ := total_socks - white_socks

-- The theorem to prove
theorem blue_socks_count : blue_socks = 60 := by
  sorry

end blue_socks_count_l552_552195


namespace total_selling_price_of_cloth_l552_552790

variables (total_meters : ℕ) (profit_per_meter cost_per_meter : ℕ)

-- Defining the conditions
def total_meters_of_cloth_sold : ℕ := 66
def profit_per_meter_of_cloth : ℕ := 5
def cost_per_meter_of_cloth : ℕ := 5

-- Define the total selling price calculation problem
theorem total_selling_price_of_cloth :
  let selling_price_per_meter := cost_per_meter_of_cloth + profit_per_meter_of_cloth in
  let total_selling_price := selling_price_per_meter * total_meters_of_cloth_sold in
  total_selling_price = 660 :=
by
  unfold total_meters_of_cloth_sold profit_per_meter_of_cloth cost_per_meter_of_cloth
  sorry

end total_selling_price_of_cloth_l552_552790


namespace sum_trigonometric_sequence_l552_552798

noncomputable def sin_deg (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

noncomputable def trigonometric_sequence (k : ℕ) : ℝ :=
  if k % 4 = 0 then sin_deg (15 + 30 * k)
  else if k % 4 = 2 then cos_deg (45 * k)
  else 0 -- Other terms do not contribute

def S : ℝ := ∑ k in Finset.range 21, (Complex.I)^(2 * k) * (trigonometric_sequence k)

theorem sum_trigonometric_sequence : S = (Real.sqrt 2) / 2 := by
  sorry

end sum_trigonometric_sequence_l552_552798


namespace new_profit_is_31_25_percent_l552_552380

-- Define variables representing the original cost, profit percentage, and resulting selling price
variables {c : ℝ} {x : ℝ}

-- Original selling price definition
def original_selling_price (c x : ℝ) : ℝ := c * (1 + 0.01 * x)

-- Reduced cost (12% reduction)
def reduced_cost (c : ℝ) : ℝ := 0.88 * c

-- Increased selling price (5% increase)
def increased_selling_price (c x : ℝ) : ℝ := 1.05 * (original_selling_price c x)

-- New profit percentage calculation
def new_profit_percentage (c x : ℝ) : ℝ :=
  ((increased_selling_price c x - reduced_cost c) / reduced_cost c) * 100

-- Theorem stating the new profit percentage is 31.25% when x is appropriately chosen
theorem new_profit_is_31_25_percent (x : ℝ) (h : new_profit_percentage 1 x = 31.25) : x = 10 :=
  sorry

end new_profit_is_31_25_percent_l552_552380


namespace min_lambda_inequality_l552_552489

open Real

theorem min_lambda_inequality :
  ∃ λ : ℝ, (λ = exp 1) ∧ ∀ (n : ℕ) (x : Fin n → ℝ),
    (0 < n) →
    (∀ i, 0 < x i) →
    (∑ i, x i = 1) →
    λ * ∏ i, (1 - x i) ≥ 1 - ∑ i, (x i)^2 :=
by
  sorry

end min_lambda_inequality_l552_552489


namespace fraction_home_l552_552789

-- Defining the conditions
def fractionFun := 5 / 13
def fractionYouth := 4 / 13

-- Stating the theorem to be proven
theorem fraction_home : 1 - (fractionFun + fractionYouth) = 4 / 13 := by
  sorry

end fraction_home_l552_552789


namespace cos_alpha_correct_l552_552159

-- Define the point P
def P : ℝ × ℝ := (3, -4)

-- Define the hypotenuse using the Pythagorean theorem
noncomputable def r : ℝ :=
  Real.sqrt (P.1 * P.1 + P.2 * P.2)

-- Define x-coordinate of point P
def x : ℝ := P.1

-- Define the cosine of the angle
noncomputable def cos_alpha : ℝ :=
  x / r

-- Prove that cos_alpha equals 3/5 given the conditions
theorem cos_alpha_correct : cos_alpha = 3 / 5 :=
by
  sorry

end cos_alpha_correct_l552_552159


namespace volume_of_cylinder_l552_552402

theorem volume_of_cylinder : 
  (side_length : ℝ) (h_side_length : side_length = 2) :
  let radius := side_length in
  let height := side_length in
  ∃ V : ℝ, V = Real.pi * radius^2 * height ∧ V = 8 * Real.pi :=
by
  sorry

end volume_of_cylinder_l552_552402


namespace caroline_lassis_l552_552439

theorem caroline_lassis (c : ℕ → ℕ): c 3 = 13 → c 15 = 65 :=
by
  sorry

end caroline_lassis_l552_552439


namespace find_vector_c_l552_552564

variables {R : Type*} [linear_ordered_field R]
variables (a b c : euclidean_space (fin 2) R)
variable k : R

def vector_a : euclidean_space (fin 2) R := ![2, 1]
def vector_b : euclidean_space (fin 2) R := ![-3, 2]

def c_perp_ab (c : euclidean_space (fin 2) R) : Prop :=
  inner c (vector_a + vector_b) = 0

def b_parallel_ca (c : euclidean_space (fin 2) R) (k : R) : Prop :=
  c - vector_a = k • vector_b

theorem find_vector_c (c : euclidean_space (fin 2) R) (k : R) 
  (h1 : c_perp_ab c)
  (h2 : b_parallel_ca c k) :
  c = ![7/3, 7/9] :=
sorry

end find_vector_c_l552_552564


namespace carl_lawn_area_l552_552801

theorem carl_lawn_area :
  ∃ (width height : ℤ), 
    (width + 1) + (height + 1) - 4 = 24 ∧
    3 * width = height ∧
    3 * ((width + 1) * 3) * ((height + 1) * 3) = 243 :=
by
  sorry

end carl_lawn_area_l552_552801


namespace field_day_difference_l552_552180

theorem field_day_difference :
  let girls_class_4_1 := 12
  let boys_class_4_1 := 13
  let girls_class_4_2 := 15
  let boys_class_4_2 := 11
  let girls_class_5_1 := 9
  let boys_class_5_1 := 13
  let girls_class_5_2 := 10
  let boys_class_5_2 := 11
  let total_girls := girls_class_4_1 + girls_class_4_2 + girls_class_5_1 + girls_class_5_2
  let total_boys := boys_class_4_1 + boys_class_4_2 + boys_class_5_1 + boys_class_5_2
  total_boys - total_girls = 2 := by
  sorry

end field_day_difference_l552_552180


namespace log_graph_passes_through_point_l552_552924

theorem log_graph_passes_through_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
    ∀ y : ℝ, (y = log a (0 + 1) + 2012) → (0, 2012) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, log a (x + 1) + 2012)} :=
by
  sorry

end log_graph_passes_through_point_l552_552924


namespace geom_prog_p_l552_552826

theorem geom_prog_p (p : ℝ) :
  (p - 2, 3 * Real.sqrt p, -8 - p) forms_geometric_progression → p = 1 :=
begin
  sorry,
end

end geom_prog_p_l552_552826


namespace factorial_fraction_simplification_l552_552443

theorem factorial_fraction_simplification (N : ℕ) :
  (N + 1)! / ((N + 2)! + N!) = (N + 1) / (N ^ 2 + 3 * N + 3) :=
by sorry

end factorial_fraction_simplification_l552_552443


namespace angle_FDE_in_triangle_l552_552622

theorem angle_FDE_in_triangle 
  (A B C M D E F : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace M] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (h1 : ∠A = 18°) 
  (h2 : ∠B = 36°) 
  (hM : Midpoint M A B) 
  (hD : OnRay CM D)
  (hAD : dist A B = dist A D)
  (hE : OnRay BC E)
  (hBE : dist B E = dist A B)
  (hF : OnRay AC F)
  (hAF : dist A F = dist A B)
  : ∠FDE = 27° :=
sorry

end angle_FDE_in_triangle_l552_552622


namespace equal_black_white_cells_in_grid_l552_552944

theorem equal_black_white_cells_in_grid :
  ∀ (grid : fin 5 → fin 5 → bool),
    (count_equal_black_white_squares grid) = 16 :=
by sorry

end equal_black_white_cells_in_grid_l552_552944


namespace exists_arith_seq_l552_552976

-- Defining the arithmetic sequence and its generating function
def arith_seq (a_1 d : ℕ) (n : ℕ) : ℕ := a_1 + (n - 1) * d

-- Define binomial coefficients
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define sum formula
def sum_arith_seq_binom (a_1 d : ℕ) (m n : ℕ) : ℕ :=
∑ k in finset.range n, arith_seq a_1 d (k + 1) * binom m k

-- Define the condition
def satisfies_eq (a_1 d : ℕ) (m n : ℕ) : Prop :=
sum_arith_seq_binom a_1 d m n = n * 2^m

-- The theorem to be proven
theorem exists_arith_seq (m : ℕ) :
  ∃ a_1 d : ℕ, (∀ n : ℕ, n > 0 → satisfies_eq a_1 d m n) ∧ a_1 = 0 ∧ d = 2 :=
sorry

end exists_arith_seq_l552_552976


namespace find_initial_mice_l552_552797

theorem find_initial_mice : 
  ∃ x : ℕ, (∀ (h1 : ∀ (m : ℕ), m * 2 = m + m), (35 * x = 280) → x = 8) :=
by
  existsi 8
  intro h1 h2
  sorry

end find_initial_mice_l552_552797


namespace find_t_find_s_find_a_find_c_l552_552694

-- Proof Problem I4.1
theorem find_t (p q r t : ℝ) (h1 : (p + q + r) / 3 = 12) (h2 : (p + q + r + t + 2 * t) / 5 = 15) : t = 13 :=
sorry

-- Proof Problem I4.2
theorem find_s (k t s : ℝ) (hk : k ≠ 0) (h1 : k^4 + (1 / k^4) = t + 1) (h2 : t = 13) (h_s : s = k^2 + (1 / k^2)) : s = 4 :=
sorry

-- Proof Problem I4.3
theorem find_a (s a b : ℝ) (hxₘ : 1 ≠ 11) (hyₘ : 2 ≠ 7) (h1 : (a, b) = ((1 * 11 + s * 1) / (1 + s), (1 * 7 + s * 2) / (1 + s))) (h_s : s = 4) : a = 3 :=
sorry

-- Proof Problem I4.4
theorem find_c (a c : ℝ) (h1 : ∀ x, a * x^2 + 12 * x + c = 0 → (a*x^2 + 12 * x + c = 0)) (h2 : ∃ x, a * x^2 + 12 * x + c = 0) : c = 36 / a :=
sorry

end find_t_find_s_find_a_find_c_l552_552694


namespace arithmetic_mean_common_difference_l552_552886

theorem arithmetic_mean_common_difference (a : ℕ → ℝ) (d : ℝ) 
    (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 + a 4 = 2 * (a 2 + 1))
    : d = 2 := 
by 
  -- Proof is omitted as it is not required.
  sorry

end arithmetic_mean_common_difference_l552_552886


namespace simplify_expression_l552_552668

-- Define the complex numbers and conditions
def ω : ℂ := (-1 + complex.I * real.sqrt 3) / 2
def ω_conj : ℂ := (-1 - complex.I * real.sqrt 3) / 2

-- Conditions
axiom ω_is_root_of_unity : ω^3 = 1
axiom ω_conj_is_root_of_unity : ω_conj^3 = 1

-- Theorem statement
theorem simplify_expression : ω^12 + ω_conj^12 = 2 := by
  sorry

end simplify_expression_l552_552668


namespace triangle_split_equal_area_l552_552792

theorem triangle_split_equal_area (m : ℝ) :
  let A := (0, 0)
  let B := (2, 2)
  let C := (4 * m, 0)
  let line_eq := λ x : ℝ, 2 * m * x
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 4 * m ∧ line_eq x = 2 * m * (x + 2) / (2 * m + 1)) →
  (m = -1/2) :=
by
  sorry

end triangle_split_equal_area_l552_552792


namespace combinations_to_50_cents_l552_552817

def num_combinations (value total : ℕ) : ℕ :=
  if total = 0 then 1
  else
    if value = 1 then 1
    else if value = 5 then (total / 5) + 1
    else if value = 10 then num_combinations 5 total + (total >= 10).nat_cast * num_combinations 5 (total - 10) +
                         (total >= 20).nat_cast * num_combinations 5 (total - 20) +
                         (total >= 30).nat_cast * num_combinations 5 (total - 30) +
                         (total >= 40).nat_cast * num_combinations 5 (total - 40)
    else if value = 25 then 
      num_combinations 10 total + 
      (total >= 25).nat_cast * num_combinations 10 (total - 25)
    else 0

theorem combinations_to_50_cents : num_combinations 25 50 = 51 :=
by
  sorry

end combinations_to_50_cents_l552_552817


namespace find_x_l552_552493

def x_approx_solution (x : ℤ) : Prop :=
  abs (75625 - x * 54) <= 27

theorem find_x : ∃ x : ℤ, x = 1400 ∧ x_approx_solution x :=
by { use 1400, split, refl, sorry }

end find_x_l552_552493


namespace circular_permutations_count_l552_552833

-- Definitions of conditions
def A : Set ℕ := {0, 1, 2}  -- Representing {a, b, c} as {0, 1, 2} down with natural numbers for simplicity.

-- Number of elements in set A
def num_elements_A : ℕ := 3

-- Length of the circular permutations
def length : ℕ := 6

-- Function to calculate the Euler's totient function
def euler_totient (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ m => Nat.coprime m n).length

-- Calculation using Burnside's Lemma
def burnside_lemma (num_elements : ℕ) (length : ℕ) : ℕ :=
  let sum := (1 / length) * ((euler_totient 1) * num_elements ^ length +
                             (euler_totient 2) * num_elements ^ (length / 2) +
                             (euler_totient 3) * num_elements ^ (length / 3) +
                             (euler_totient length) * num_elements ^ (length / length))
  sum.to_nat  -- Convert to ℕ

-- Theorem statement
theorem circular_permutations_count : burnside_lemma num_elements_A length = 130 :=
by
  -- Proof would go here
  sorry

end circular_permutations_count_l552_552833


namespace manuscript_copy_cost_l552_552804

theorem manuscript_copy_cost (total_cost : ℝ) (binding_cost : ℝ) (num_manuscripts : ℕ) (pages_per_manuscript : ℕ) (x : ℝ) :
  total_cost = 250 ∧ binding_cost = 5 ∧ num_manuscripts = 10 ∧ pages_per_manuscript = 400 →
  x = (total_cost - binding_cost * num_manuscripts) / (num_manuscripts * pages_per_manuscript) →
  x = 0.05 :=
by
  sorry

end manuscript_copy_cost_l552_552804


namespace find_locations_for_R_l552_552645

noncomputable def num_locations_R (P Q : ℝ × ℝ) (PQ : ℝ) (A : ℝ) : ℕ :=
  let triangle_area (R : ℝ × ℝ) (a b : ℝ × ℝ) : ℝ :=
    (1 / 2) * |(a.1 - b.1) * (a.2 - R.2) - (a.2 - b.2) * (a.1 - R.1)| in
  let is_right_triangle (R : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
    (a.1 - R.1) * (b.1 - R.1) + (a.2 - R.2) * (b.2 - R.2) = 0 ∨
    (a.1 - b.1) * (b.1 - R.1) + (a.2 - b.2) * (b.2 - R.2) = 0 ∨
    (a.1 - R.1) * (a.1 - b.1) + (a.2 - R.2) * (a.2 - b.2) = 0 in
  let candidates : list (ℝ × ℝ) := [(5, 3), (5, -3), (-5, 3), (-5, -3), (6, 3), (-6, 3), (6, -3), (-6, -3)] in
  candidates.filter (λ R, triangle_area R P Q = A ∧ is_right_triangle R P Q).length

theorem find_locations_for_R :
  ∀ (P Q: ℝ × ℝ), PQ = 10 → A = 15 → num_locations_R P Q PQ A = 8 :=
by
  intros
  sorry

end find_locations_for_R_l552_552645


namespace initial_boys_provision_l552_552676

-- Define the variables according to the conditions
variables (B : ℕ) -- Initial number of boys
variables (P : ℕ) -- Total provisions

-- Define the conditions
def condition1 : Prop := P = B * 15
def condition2 : Prop := P = (B + 200) * 12.5

-- Statement of the problem
theorem initial_boys_provision (h1 : condition1) (h2 : condition2) : B = 1000 :=
by {
  sorry
}

end initial_boys_provision_l552_552676


namespace minimal_abs_diff_l552_552574

theorem minimal_abs_diff (a b : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b - 8 * a + 7 * b = 569) : abs (a - b) = 23 :=
sorry

end minimal_abs_diff_l552_552574


namespace incorrect_option_C_l552_552010

-- Definitions for the problem
variable {α β : Type} [OrderedRing α] [LinearOrderedField α]

def monotonically_increasing_on (f : α → β) (I : Set α) : Prop :=
  ∀ (x y : α), x ∈ I → y ∈ I → (x ≤ y → f x ≤ f y)

def symmetric_about_x_minus_1 (f : α → α) : Prop :=
  ∀ (x : α), f (1 - x) = f (1 + x)

-- Conditions for the problem
variables (f : α → β) (f_trans : α → β)
variables (domain_f : Set α) (range_f : Set β)
variables (I : Set α)
variable (x : α)

-- The Lean statement
theorem incorrect_option_C :
  (domain_f = Set.univ) →
  (range_f = Set.Ioi 0) →
  (monotonically_increasing_on f I) →
  (f_trans = λ x, f (2 * x - 1)) →
  (monotonically_increasing_on (λ x, f (2 * x - 1)) I)
  ↔ False := 
sorry

end incorrect_option_C_l552_552010


namespace min_value_of_expression_l552_552531

theorem min_value_of_expression {x y z : ℝ} 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 2 * x * (x + 1 / y + 1 / z) = y * z) : 
  (x + 1 / y) * (x + 1 / z) >= Real.sqrt 2 :=
by
  sorry

end min_value_of_expression_l552_552531


namespace fixed_point_exists_l552_552551

-- Given conditions
variables {a : ℝ} (h₀ : a > 0) (h₁ : a ≠ 1)

-- Function definition
def f (x : ℝ) := log a (x + 1) + 2

-- Theorem statement to prove the fixed point
theorem fixed_point_exists : f a h₀ h₁ 0 = 2 :=
sorry

end fixed_point_exists_l552_552551


namespace problem1_problem2_l552_552437

theorem problem1 : -24 - (-15) + (-1) + (-15) = -25 := 
by 
  sorry

theorem problem2 : -27 / (3 / 2) * (2 / 3) = -12 := 
by 
  sorry

end problem1_problem2_l552_552437


namespace circle_radius_formula_correct_l552_552527

noncomputable def touch_circles_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let Δ := (s - a) * (s - b) * (s - c)
  let numerator := c * Real.sqrt ((s - a) * (s - b) * (s - c))
  let denominator := c * Real.sqrt s + 2 * Real.sqrt ((s - a) * (s - b) * (s - c))
  numerator / denominator

theorem circle_radius_formula_correct (a b c : ℝ) : 
  let s := (a + b + c) / 2
  let Δ := (s - a) * (s - b) * (s - c)
  ∀ (r : ℝ), (r = touch_circles_radius a b c) :=
sorry

end circle_radius_formula_correct_l552_552527


namespace half_MN_correct_l552_552136

-- Definitions for the vectors OM and ON
def OM : ℝ × ℝ := (-2, 3)
def ON : ℝ × ℝ := (-1, -5)

-- Definition of the vector MN
def MN : ℝ × ℝ := (ON.1 - OM.1, ON.2 - OM.2)

-- Definition of 1/2 * MN
def half_MN : ℝ × ℝ := (1 / 2 * MN.1, 1 / 2 * MN.2)

-- The theorem we need to prove
theorem half_MN_correct : half_MN = (1 / 2, -4) :=
by
  sorry

end half_MN_correct_l552_552136


namespace points_on_circle_l552_552699

-- Definitions of points and lines
variables (A B C D O K L M P Q : Type) [Point A] [Point B] [Point C] [Point D] [Point O]
[Point K] [Point L] [Point M] [Point P] [Point Q]
(Line AB) (Line CD) (Line OL) (Line OM)

// Conditions described in the problem
axiom cyclic_quadrilateral (cycq : CyclicQuadrilateral A B C D)
axiom diagonals_meet (meets : DiagonalsMeet A B C D O)
axiom circumcircle_triangles (circ1 : Circumcircle_triangles A B O S1) 
                             (circ2 : Circumcircle_triangles C D O S2)
axiom circles_intersect (intersect : CirclesIntersect O S1 S2 K)
axiom parallel_lines (parallel_AB : Parallel OL AB) (parallel_CD : Parallel OM CD)
axiom points_on_segments (P_on_OL : OnSegment P OL) (Q_on_OM : OnSegment Q OM)
axiom ratio_condition (ratio_eq : Ratio OP PL = Ratio MQ QO)

-- The theorem we need to prove
theorem points_on_circle : OnCircle O K P Q :=
sorry

end points_on_circle_l552_552699


namespace problem_solution_l552_552130

namespace MathProof

-- Definitions based on the conditions
def f (x m : ℝ) := |2 * x - m|
def f_half_x (x m : ℝ) := |x / 2 + 3|

-- Problem statement: Prove the value of m and the range of x
theorem problem_solution (m : ℝ) (a b : ℝ) (h : a + b = 2) : 
  (∀ x : ℝ, f x m ≤ 6 ↔ -2 ≤ x ∧ x ≤ 4) → 
  f 2 m + f_half_x 2 m ≤ 9 → 
  m = 2 ∧ ∀ x : ℝ, (x ≥ -3 ∧ x ≤ 7 / 3) := by
  sorry

end MathProof

end problem_solution_l552_552130


namespace roots_farthest_apart_at_3_l552_552855

def quadratic_coefficient_a : ℝ := 1

def quadratic_coefficient_b (a : ℝ) : ℝ := -4 * a

def quadratic_coefficient_c (a : ℝ) : ℝ := 5 * a^2 - 6 * a

def discriminant (a : ℝ) : ℝ :=
  quadratic_coefficient_b(a)^2 - 4 * quadratic_coefficient_a * quadratic_coefficient_c(a)

def vertex (a : ℝ) : ℝ :=
  a * (6 - a)

theorem roots_farthest_apart_at_3 :
  argmax vertex = 3 :=
sorry

end roots_farthest_apart_at_3_l552_552855


namespace number_of_real_solutions_l552_552299

def f (x : ℝ) : ℝ := 2 ^ x * x ^ 2 - 1

theorem number_of_real_solutions : (∃! x : ℝ, f x = 0) → 3 :=
sorry

end number_of_real_solutions_l552_552299


namespace roots_of_unity_expression_l552_552662

-- Defining the complex cube roots of unity
def omega := Complex.exp (2 * Real.pi * Complex.I / 3)
def omega2 := Complex.exp (-2 * Real.pi * Complex.I / 3)

-- Main theorem statement to prove
theorem roots_of_unity_expression :
  ((-1 + Complex.i * Real.sqrt 3) / 2) ^ 12 + ((-1 - Complex.i * Real.sqrt 3) / 2) ^ 12 = 2 :=
by
  -- Definitions of the cube roots and their properties
  have h1 : omega ^ 3 = 1 := sorry
  have h2 : omega2 ^ 3 = 1 := sorry
  have h3 : (-1 + Complex.i * Real.sqrt 3) / 2 = omega := sorry
  have h4 : (-1 - Complex.i * Real.sqrt 3) / 2 = omega2 := sorry
  -- Using the properties of the roots and their definitions to prove the statement
  sorry

end roots_of_unity_expression_l552_552662


namespace mean_and_mode_of_data_set_l552_552396

noncomputable def data_set : List ℕ := [1, 1, 2, 3, 3, 3, 3, 4, 5, 5]

def mean (l : List ℕ) : ℚ := (l.map (λ x => (x : ℚ)).sum) / (l.length : ℚ)

def mode (l : List ℕ) : ℕ :=
l.foldl (λ acc x => if (l.count x > l.count acc) then x else acc) (l.headI)

theorem mean_and_mode_of_data_set :
  mean data_set = 3 ∧ mode data_set = 3 :=
by
  sorry

end mean_and_mode_of_data_set_l552_552396


namespace pizza_remained_l552_552953

noncomputable def number_of_people := 15
noncomputable def fraction_eating_pizza := 3 / 5
noncomputable def total_pizza_pieces := 50
noncomputable def pieces_per_person := 4
noncomputable def pizza_remaining := total_pizza_pieces - (pieces_per_person * (fraction_eating_pizza * number_of_people))

theorem pizza_remained :
  pizza_remaining = 14 :=
by {
  sorry
}

end pizza_remained_l552_552953


namespace complement_A_is_0_l552_552559

open Set

def A : Set ℤ := {x | abs x ≥ 1}

theorem complement_A_is_0 : (compl A : Set ℤ) = {0} :=
by
  sorry

end complement_A_is_0_l552_552559


namespace complex_div_conjugate_l552_552863

theorem complex_div_conjugate (z : ℂ) (h : z = 1 + complex.i) : 2 / z = 1 - complex.i :=
by
  rw h
  sorry

end complex_div_conjugate_l552_552863


namespace negation_of_at_least_three_is_at_most_two_l552_552296

theorem negation_of_at_least_three_is_at_most_two :
  (¬ (∀ n : ℕ, n ≥ 3)) ↔ (∃ n : ℕ, n ≤ 2) :=
sorry

end negation_of_at_least_three_is_at_most_two_l552_552296


namespace num_natural_numbers_divisors_count_l552_552479

theorem num_natural_numbers_divisors_count:
  ∃ N : ℕ, (2017 % N = 17 ∧ N ∣ 2000) ↔ 13 := 
sorry

end num_natural_numbers_divisors_count_l552_552479


namespace concurrence_of_AT_BU_PQ_l552_552586

-- Definitions based on conditions
variables {A B C D P Q R S U T : Type}
variables {parallelogram_ABCD : Parallelogram A B C D}
variables {circle_ABC : Incircle ABC}
variables {circle_ACD : Incircle ACD}
variables {P_on_BC : Point P (LineSegment B C)}
variables {Q_on_CA : Point Q (LineSegment C A)}
variables {R_on_CD : Point R (LineSegment C D)}
variables {S_on_AD : Point S (Intersection (LineSegment P Q) (LineSegment A D))}
variables {U_on_AR_CS : Point U (Intersection (LineSegment A R) (LineSegment C S))}
variables {T_on_BC : Point T (LineSegment B C)}
variables {T_eq_AB_BT : AB = BT}

-- Theorem statement
theorem concurrence_of_AT_BU_PQ :
  Concurrent (Line A T) (Line B U) (Line P Q) := sorry

end concurrence_of_AT_BU_PQ_l552_552586


namespace hyperbola_cond1_hyperbola_cond2_hyperbola_cond3_l552_552844

/-- Condition 1: Hyperbola with given a, b and foci on x-axis -/
theorem hyperbola_cond1 : (∀ x y : ℝ , 
  (∃ a b : ℝ, a = 3 ∧ b = 4 ∧ (∃ f1 f2: ℝ, f1 = (a,0) ∧ f2 = (-a, 0))
  → (x^2 / 9 - y^2 / 16 = 1)) := sorry

/-- Condition 2: Hyperbola with given foci coordinates and absolute difference -/
theorem hyperbola_cond2 : (∀ x y : ℝ, 
  (∃ a b c : ℝ, c = 10 ∧ abs (distance (x, y) (0, 10) - distance (x, y) (0, -10)) = 16 
    → (x^2 / 64 - y^2 / 36 = 1)) := sorry 

/-- Condition 3: Hyperbola with given foci coordinates and passing through a specific point -/
theorem hyperbola_cond3 : (∀ x y : ℝ, 
  (∃ a b c: ℝ, c = 5 ∧ (x, y) = (4 * sqrt 3 / 3, 2 * sqrt 3) 
    → (y^2 / 9 - x^2 / 16 = 1)) := sorry

end hyperbola_cond1_hyperbola_cond2_hyperbola_cond3_l552_552844


namespace triangle_inequality_l552_552977

variable {α : Type*} [MetricSpace α] [NormedAddCommGroup α] 

/-- Define the points and setup --/
variables (A B C U V W T : α)

/-- Define the angles --/
variables (angle_bac : Real)

/-- Hypotheses --/
hypothesis (h1 : angle_bac = min_angle (angle A B C))
hypothesis (h2 : arc_is_divided B C)
hypothesis (h3 : interior_point U (arc B C))
hypothesis (h4 : bisector_meets A B AU V)
hypothesis (h5 : bisector_meets A C AU W)
hypothesis (h6 : lines_meet B V C W T)

/-- Goal --/
theorem triangle_inequality :
  AU = BT + TC :=
sorry

end triangle_inequality_l552_552977


namespace molecular_weight_BaCl2_total_weight_of_8_moles_BaCl2_l552_552738

noncomputable theory

def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_Cl : ℝ := 35.45
def num_moles_BaCl2 : ℝ := 8

theorem molecular_weight_BaCl2 : ℝ :=
  atomic_weight_Ba + 2 * atomic_weight_Cl

theorem total_weight_of_8_moles_BaCl2 :
  (molecular_weight_BaCl2 * num_moles_BaCl2) = 1665.84 :=
by
  sorry

end molecular_weight_BaCl2_total_weight_of_8_moles_BaCl2_l552_552738


namespace roots_of_unity_expression_l552_552664

-- Defining the complex cube roots of unity
def omega := Complex.exp (2 * Real.pi * Complex.I / 3)
def omega2 := Complex.exp (-2 * Real.pi * Complex.I / 3)

-- Main theorem statement to prove
theorem roots_of_unity_expression :
  ((-1 + Complex.i * Real.sqrt 3) / 2) ^ 12 + ((-1 - Complex.i * Real.sqrt 3) / 2) ^ 12 = 2 :=
by
  -- Definitions of the cube roots and their properties
  have h1 : omega ^ 3 = 1 := sorry
  have h2 : omega2 ^ 3 = 1 := sorry
  have h3 : (-1 + Complex.i * Real.sqrt 3) / 2 = omega := sorry
  have h4 : (-1 - Complex.i * Real.sqrt 3) / 2 = omega2 := sorry
  -- Using the properties of the roots and their definitions to prove the statement
  sorry

end roots_of_unity_expression_l552_552664


namespace new_unit_for_graph_transformation_l552_552737

theorem new_unit_for_graph_transformation:
  (∀ (x : ℝ), (∃ e : ℝ, 0 < e ∧ (∀ (x : ℝ), x ≠ 0 → e = (real.sqrt 2) / 2) → 
  (∀ (a b : ℝ), (b = 1 / a) → 
                  (b / e = (2 / (a / e)) →
                    e = (real.sqrt 2) / 2)))) :=
begin
  sorry
end

end new_unit_for_graph_transformation_l552_552737


namespace percentage_by_which_x_is_more_than_y_l552_552938

variable {z : ℝ} 

-- Define x and y based on the given conditions
def x (z : ℝ) : ℝ := 0.78 * z
def y (z : ℝ) : ℝ := 0.60 * z

-- The main theorem we aim to prove
theorem percentage_by_which_x_is_more_than_y (z : ℝ) : x z = y z + 0.30 * y z := by
  sorry

end percentage_by_which_x_is_more_than_y_l552_552938


namespace find_phi_expression_l552_552903

noncomputable def proportional_f (x : ℝ) (m : ℝ) : ℝ := m * x
noncomputable def proportional_g (x : ℝ) (n : ℝ) : ℝ := n / x
noncomputable def phi (x : ℝ) (m n : ℝ) : ℝ := proportional_f x m + proportional_g x n

theorem find_phi_expression (m n : ℝ) (h1 : phi (1/3) m n = 16) (h2 : phi 1 m n = 8) :
  phi x 3 5 = 3 * x + 5 / x :=
by
  -- Skipping the proof
  sorry

#eval find_phi_expression 3 5 (by simp[phi, proportional_f, proportional_g]) (by simp[phi, proportional_f, proportional_g])

end find_phi_expression_l552_552903


namespace hyperbola_equation_l552_552386

theorem hyperbola_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ (x y : ℝ), (x, y) = (4 * real.sqrt 2, -3) → (x^2 / a^2 - y^2 / b^2 = 1)) ∧ 
  (∀ (c : ℝ), c^2 = a^2 + b^2 ∧ (5 / c) * (5 / -c) = -1) → (a = 4 ∧ b = 3) :=
by
  sorry

end hyperbola_equation_l552_552386


namespace largest_valid_five_digit_number_sum_l552_552996

-- Define the condition: five-digit number whose digits multiply to 180
def is_valid_five_digit_number (n : ℕ) : Prop :=
  n >= 10000 ∧ n < 100000 ∧ (∃ d1 d2 d3 d4 d5 : ℕ, d1 * d2 * d3 * d4 * d5 = 180 ∧
    n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5)

-- Define the largest five-digit number satisfying the condition
def largest_valid_five_digit_number : ℕ :=
  95221

-- Main statement to prove
theorem largest_valid_five_digit_number_sum :
  ∃ n : ℕ, is_valid_five_digit_number n ∧ n = largest_valid_five_digit_number ∧
  (∑ d in [9, 5, 2, 2, 1], d) = 19 :=
by
  existsi largest_valid_five_digit_number
  split
  -- Prove n is a valid five-digit number
  use [9, 5, 2, 2, 1]
  {
    -- Prove the digits multiply to 180
    sorry,
    -- Prove n is 95221
    sorry,
  }
  -- Prove the sum of the digits is 19
  sorry

end largest_valid_five_digit_number_sum_l552_552996


namespace velocity_at_t3_acceleration_at_t3_l552_552294

def motion_equation (t : ℝ) : ℝ :=
  -1/6 * t^3 + 3 * t^2 - 5

def velocity (t : ℝ) : ℝ :=
  -1/2 * t^2 + 6 * t

def acceleration (t : ℝ) : ℝ :=
  -t + 6

theorem velocity_at_t3 : velocity 3 = 27 / 2 :=
by
  sorry

theorem acceleration_at_t3 : acceleration 3 = 3 :=
by
  sorry

end velocity_at_t3_acceleration_at_t3_l552_552294


namespace petya_sum_l552_552234

theorem petya_sum : 
  let f (signs : fin 5 → bool) : ℤ :=
    1 + (if signs 0 then 2 else -2) + (if signs 1 then 3 else -3) + (if signs 2 then 4 else -4) + (if signs 3 then 5 else -5) + (if signs 4 then 6 else -6),
  sum (f '' (finset.univ : finset (fin 5 → bool))) = 32 :=
by
  sorry

end petya_sum_l552_552234


namespace base8_product_l552_552741

theorem base8_product (n : ℕ) (h : n = 7654) : 
  let base8_rep := [1, 6, 3, 0, 0] in
  let modified_digits := [0, 5, 2, 0, 0] in
  modified_digits.filter (· ≠ 0).prod = 10 := 
by 
  sorry

end base8_product_l552_552741


namespace socks_ratio_l552_552981

theorem socks_ratio (red_pairs : Nat) (total_socks : Nat) (ratio : Rat)
  (h1 : red_pairs = 20)
  (h2 : total_socks = 90)
  (h3 : ratio = 1 / 2) :
  let red_socks := red_pairs * 2
  let black_socks := red_socks / 2
  let combined_socks := red_socks + black_socks
  let white_socks := total_socks - combined_socks
  (white_socks : combined_socks) = ratio := sorry

end socks_ratio_l552_552981


namespace roots_of_unity_expression_l552_552665

-- Defining the complex cube roots of unity
def omega := Complex.exp (2 * Real.pi * Complex.I / 3)
def omega2 := Complex.exp (-2 * Real.pi * Complex.I / 3)

-- Main theorem statement to prove
theorem roots_of_unity_expression :
  ((-1 + Complex.i * Real.sqrt 3) / 2) ^ 12 + ((-1 - Complex.i * Real.sqrt 3) / 2) ^ 12 = 2 :=
by
  -- Definitions of the cube roots and their properties
  have h1 : omega ^ 3 = 1 := sorry
  have h2 : omega2 ^ 3 = 1 := sorry
  have h3 : (-1 + Complex.i * Real.sqrt 3) / 2 = omega := sorry
  have h4 : (-1 - Complex.i * Real.sqrt 3) / 2 = omega2 := sorry
  -- Using the properties of the roots and their definitions to prove the statement
  sorry

end roots_of_unity_expression_l552_552665


namespace even_divisors_10_factorial_l552_552568

theorem even_divisors_10_factorial : 
  let prime_factorization := 2^8 * 3^4 * 5^2 * 7 in
  ∃ (num_even_divisors : ℕ), num_even_divisors = 240 :=
begin
  let prime_factorization := 2^8 * 3^4 * 5^2 * 7,
  have : ∃ (num_even_divisors : ℕ), num_even_divisors = (8 * 5 * 3 * 2),
  { use 240,
    sorry },
  exact this,
end

end even_divisors_10_factorial_l552_552568


namespace m_plus_n_l552_552264

variable {O A B C D : Type}
variable [HasCos O A B C]
variable (angle_AOB : Angle O A B = 45)
variable (cos_theta : Cos (Angle O A B) = (m : ℤ) + Real.sqrt (n : ℤ))

theorem m_plus_n (m n : ℤ) : m + n = 5 :=
sorry

end m_plus_n_l552_552264


namespace number_of_terms_in_arithmetic_sequence_l552_552483

/-- Define the conditions. -/
def a : ℕ := 2
def d : ℕ := 5
def a_n : ℕ := 57

/-- Define the proof problem. -/
theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, a_n = a + (n - 1) * d ∧ n = 12 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l552_552483


namespace triangle_inequality_l552_552989

theorem triangle_inequality
  (α β γ a b c : ℝ)
  (h_angles_sum : α + β + γ = Real.pi)
  (h_pos_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_pos_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  a * (1 / β + 1 / γ) + b * (1 / γ + 1 / α) + c * (1 / α + 1 / β) ≥ 2 * (a / α + b / β + c / γ) := by
  sorry

end triangle_inequality_l552_552989


namespace delta_condition_l552_552501

def sequence (n : ℕ) : ℤ := n^3 + n

def delta (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ
| 0    := u
| 1    := λ n, u (n + 1) - u n
| (k+1) := λ n, delta 1 (delta k u) n

theorem delta_condition :
  ∀ n : ℕ, delta 4 (sequence) n = 0 :=
by
  sorry

end delta_condition_l552_552501


namespace Nancy_picked_l552_552412

def Alyssa_picked : ℕ := 42
def Total_picked : ℕ := 59

theorem Nancy_picked : Total_picked - Alyssa_picked = 17 := by
  sorry

end Nancy_picked_l552_552412


namespace simplify_expression_l552_552658

theorem simplify_expression :
  (Complex.mk (-1) (Real.sqrt 3) / 2) ^ 12 + (Complex.mk (-1) (-Real.sqrt 3) / 2) ^ 12 = 2 := by
  sorry

end simplify_expression_l552_552658


namespace exists_point_with_at_most_three_nearest_neighbors_l552_552079

variable {Point : Type} [fintype Point]

-- Distance metric on points
variable (dist : Point → Point → ℝ)

-- Neighborhood set N(p) defined for each point in the set based on the distance
def N (S : finset Point) (p : Point) : finset Point := 
  S.filter (λ q, dist p q = finset.min' (S \ {p}) (λ q, dist p q))

theorem exists_point_with_at_most_three_nearest_neighbors (S : finset Point) (hS : S.nonempty) :
  ∃ p ∈ S, (N dist S p).card ≤ 3 := 
by
  sorry

end exists_point_with_at_most_three_nearest_neighbors_l552_552079


namespace area_of_triangle_QMN_l552_552707

-- Given definitions
def PQ := 8 -- length of rectangle PQRS in inches
def PS := 6 -- width of rectangle PQRS in inches

-- Points M, N, O divide the diagonal PR into four equal segments
def num_segments := 4
def PR := Real.sqrt (PQ^2 + PS^2) -- length of diagonal PR
def segment_length := PR / num_segments

-- The height from Q to PR
def height := 24 / 5

-- Proving the area of triangle QMN is 6 square inches
theorem area_of_triangle_QMN : (1 / 2) * segment_length * height = 6 := by 
  sorry

end area_of_triangle_QMN_l552_552707


namespace divides_sequence_l552_552317

theorem divides_sequence (a : ℕ → ℕ) (n k: ℕ) (h0 : a 0 = 0) (h1 : a 1 = 1) 
  (hrec : ∀ m, a (m + 2) = 2 * a (m + 1) + a m) :
  (2^k ∣ a n) ↔ (2^k ∣ n) :=
sorry

end divides_sequence_l552_552317


namespace line_passing_through_P_with_equal_intercepts_line_passing_through_A_with_double_inclination_l552_552059

-- Definitions for the given conditions

-- Condition 1: Line through P(1,2) with equal intercepts
def line_eq_1 (x y : ℝ) : Prop :=
  (2 * x - y = 0) ∨ (x + y - 3 = 0)

-- Condition 2: Line through A(-1,-1) with angle of inclination twice that of y = (1/2)x
def line_eq_2 (x y : ℝ) : Prop :=
  4 * x - 3 * y + 1 = 0

-- Given points
def P := (1, 2) : ℝ × ℝ
def A := (-1, -1) : ℝ × ℝ

-- Theorem statements
theorem line_passing_through_P_with_equal_intercepts :
  ∃ f : ℝ → ℝ → Prop, f P.1 P.2 ∧ f = line_eq_1 :=
sorry

theorem line_passing_through_A_with_double_inclination :
  ∃ f : ℝ → ℝ → Prop, f A.1 A.2 ∧ f = line_eq_2 :=
sorry

end line_passing_through_P_with_equal_intercepts_line_passing_through_A_with_double_inclination_l552_552059


namespace proof_problem_l552_552110

noncomputable def z : ℂ := 2 / (1 - Complex.i)^2 + (3 + Complex.i) / (1 - Complex.i)
def m : ℝ := 3
def f (x : ℝ) := x + 4 / (x - 1)
def n : ℝ := 5
def area (a b : ℝ) := ∫ x in a..b, x

theorem proof_problem : (z.im = m) ∧ (∀ x ∈ set.Icc 2 3, f x ≥ n) ∧ (area 3 5 = 8) :=
by
  sorry

end proof_problem_l552_552110


namespace find_c_half_area_eq_l552_552672

theorem find_c_half_area_eq (c : ℝ) :
  (∃ h : c ∈ set.Icc 0 6,
    (∀ x ∈ set.Icc 0 6, x ≠ c → (∫ t in (0 : ℝ)..(x - c), (3 / (6 - c)) * t) = 3)) ↔ c = 4 :=
by
  sorry

end find_c_half_area_eq_l552_552672


namespace poly_constant_or_sum_constant_l552_552988

-- definitions of the polynomials as real-coefficient polynomials
variables (P Q R : Polynomial ℝ)

-- conditions
#check ∀ x, P.eval (Q.eval x) + P.eval (R.eval x) = (1 : ℝ) -- Considering 'constant' as 1 for simplicity

-- target
theorem poly_constant_or_sum_constant 
  (h : ∀ x, P.eval (Q.eval x) + P.eval (R.eval x) = (1 : ℝ)) :
  (∃ c : ℝ, ∀ x, P.eval x = c) ∨ (∃ c : ℝ, ∀ x, Q.eval x + R.eval x = c) :=
sorry

end poly_constant_or_sum_constant_l552_552988


namespace smallest_n_for_T_n_integer_l552_552998

def L : ℚ := ∑ i in {1, 2, 3, 4}, 1 / i

theorem smallest_n_for_T_n_integer : ∃ n ∈ ℕ, n > 0 ∧ (n * 5^(n-1) * L).denom = 1 ∧ n = 12 :=
by
  have hL : L = 25 / 12 := by sorry
  existsi 12
  split
  exact Nat.succ_pos'
  split
  suffices (12 * 5^(12-1) * 25 / 12).denom = 1 by sorry
  sorry
  rfl

end smallest_n_for_T_n_integer_l552_552998


namespace August_five_Tuesdays_l552_552681

-- Definitions based on conditions
def July_has_five_Mondays (N : ℕ) : Prop :=
  ∃(start_day : ℕ) (1 ≤ start_day ∧ start_day ≤ 7), 
    ∀(k : ℕ), k ∈ [0, 1, 2, 3, 4] → (start_day + k * 7) % 31 ∈ [0, 1, 2, 3, 4, 5, 6]

def August_has_30_days (N : ℕ) : Prop :=
  true  -- We don't actually need a complex definition here since it's given

-- The final theorem to prove
theorem August_five_Tuesdays (N : ℕ) (H_July : July_has_five_Mondays N) (H_August : August_has_30_days N) : 
  ∃ (k : ℕ), k = 2 ∧ ∀(m : ℕ), m ∈ [0, 1, 2, 3, 4] → (2 + m * 7) % 30 ∈ [0, 1, 2, 3, 4, 5, 6] :=
sorry

end August_five_Tuesdays_l552_552681


namespace intersect_four_points_l552_552447

noncomputable def graphs_intersection (B : ℝ) (hB : 0 < B) : Prop :=
  let y := λ (x : ℝ), B * x^2,
      eq2 := λ (x y : ℝ), y^2 + 3 = x^2 + 6 * y
  in ∃ (points : List (ℝ × ℝ)), points.length = 4 ∧
      ∀ (p : ℝ × ℝ), p ∈ points → (p.2 = y p.1 ∧ eq2 p.1 p.2)

theorem intersect_four_points (B : ℝ) (hB : 0 < B) :
  graphs_intersection B hB := 
sorry

end intersect_four_points_l552_552447


namespace primitive_root_set_equality_l552_552609

theorem primitive_root_set_equality 
  {p : ℕ} (hp : Nat.Prime p) (hodd: p % 2 = 1) (g : ℕ) (hg : g ^ (p - 1) % p = 1) :
  (∀ k, 1 ≤ k ∧ k ≤ (p - 1) / 2 → ∃ m, 1 ≤ m ∧ m ≤ (p - 1) / 2 ∧ (k^2 + 1) % p = g ^ m % p) ↔ p = 3 :=
by sorry

end primitive_root_set_equality_l552_552609


namespace simplify_expression_l552_552667

-- Define the complex numbers and conditions
def ω : ℂ := (-1 + complex.I * real.sqrt 3) / 2
def ω_conj : ℂ := (-1 - complex.I * real.sqrt 3) / 2

-- Conditions
axiom ω_is_root_of_unity : ω^3 = 1
axiom ω_conj_is_root_of_unity : ω_conj^3 = 1

-- Theorem statement
theorem simplify_expression : ω^12 + ω_conj^12 = 2 := by
  sorry

end simplify_expression_l552_552667


namespace find_expectation_l552_552506

variable (X : Type) [AddCommGroup X] [Module ℝ X] [Expectation X]

theorem find_expectation
  (E : X → ℝ)
  (h : E (λ x, x) + E (λ x, 2 * x + 1) = 8) :
  E (λ x, x) = 7 / 3 := 
by sorry

end find_expectation_l552_552506


namespace abc_solutions_count_l552_552834

theorem abc_solutions_count :
  ∃ n : ℕ, 
    (∀ (a b c : ℕ), (1 ≤ c ∧ c ≤ b ∧ b ≤ a) → (a * b * c = 2 * (a - 1) * (b - 1) * (c - 1)) ↔ (n = 5)) :=
begin
  sorry
end

end abc_solutions_count_l552_552834


namespace Brenda_bakes_cakes_l552_552019

theorem Brenda_bakes_cakes 
  (cakes_per_day : ℕ)
  (days : ℕ)
  (sell_fraction : ℚ)
  (total_cakes_baked : ℕ := cakes_per_day * days)
  (cakes_left : ℚ := total_cakes_baked * sell_fraction)
  (h1 : cakes_per_day = 20)
  (h2 : days = 9)
  (h3 : sell_fraction = 1 / 2) :
  cakes_left = 90 := 
by 
  -- Proof to be filled in later
  sorry

end Brenda_bakes_cakes_l552_552019


namespace triangle_is_isosceles_l552_552536

variable (A B C a b c : ℝ)
variable (sin : ℝ → ℝ)

theorem triangle_is_isosceles (h1 : a * sin A - b * sin B = 0) :
  a = b :=
by
  sorry

end triangle_is_isosceles_l552_552536


namespace probability_sum_odd_probability_B_wins_l552_552384

/-- Cards drawn by players -/
inductive Card where
  | A (n : ℕ) : Card -- A's card labeled with n
  | B (n : ℕ) : Card -- B's card labeled with n

/-- Possible outcomes of drawing two cards from B -/
def outcomesB : List (Card × Card) :=
  [(Card.B 1, Card.B 2), (Card.B 1, Card.B 3), (Card.B 1, Card.B 4),
   (Card.B 2, Card.B 3), (Card.B 2, Card.B 4), (Card.B 3, Card.B 4)]

/-- Possible outcomes of A and B drawing one card each -/
def outcomesAB : List (Card × Card) :=
  [(Card.A 2, Card.B 1), (Card.A 2, Card.B 2), (Card.A 2, Card.B 3), (Card.A 2, Card.B 4),
   (Card.A 3, Card.B 1), (Card.A 3, Card.B 2), (Card.A 3, Card.B 3), (Card.A 3, Card.B 4)]

/-- Prove that the probability that the sum of the numbers on two cards 
randomly drawn by B is odd is 2/3 -/
theorem probability_sum_odd :
  ((List.filter (fun (c : Card × Card) => (c.1.sum + c.2.sum) % 2 = 1) outcomesB).length) / 
  outcomesB.length = 2 / 3 :=
sorry

/-- Prove that the probability that B wins when A and B each draw a card is 3/8 -/
theorem probability_B_wins :
  ((List.filter (fun (c : Card × Card) => match (c.1, c.2) with
                                          | (Card.A a, Card.B b) => b > a
                                          | _ => false) outcomesAB).length) / 
  outcomesAB.length = 3 / 8 :=
sorry


end probability_sum_odd_probability_B_wins_l552_552384


namespace magic_square_y_value_l552_552585

-- Definitions for conditions
def magic_sum (square : ℕ → ℕ → ℕ) (n : ℕ) : ℕ :=
  square 0 0 + square 0 1 + square 0 2 

def is_magic_square (square : ℕ → ℕ → ℕ) : Prop :=
  magic_sum square 3 = square 0 0 + square 1 0 + square 2 0 ∧
  magic_sum square 3 = square 0 1 + square 1 1 + square 2 1 ∧
  magic_sum square 3 = square 0 2 + square 1 2 + square 2 2 ∧
  magic_sum square 3 = square 0 0 + square 1 1 + square 2 2 ∧
  magic_sum square 3 = square 0 2 + square 1 1 + square 2 0

-- Given magic square entries
def square : ℕ → ℕ → ℕ
| 0, 0 => y
| 0, 1 => 25
| 0, 2 => 70
| 1, 0 => 5
| 1, 1 => a
| 1, 2 => b
| 2, 0 => 90
| 2, 1 => d
| 2, 2 => e
| _, _ => 0  -- Default value for other entries

-- The target is to prove that y = 90
theorem magic_square_y_value : 
  is_magic_square square → y = 90 :=
by
  sorry

end magic_square_y_value_l552_552585


namespace triangle_inequality_FG_l552_552370

variable (E F G H : Type)
variable (EF EG HG HF : ℝ)
variable [Lean.SMG : StrictOrderedCommMonoid ℝ] -- Definition of real numbers with addition and multiplication providing strict order
variable [Lean.PRNG : Preorder (Type)]
variable [Lean.G : Group ℝ]

theorem triangle_inequality_FG (EF EG HG HF : ℝ) (hEF : EF = 7) (hEG : EG = 15) (hHG : HG = 10) (hHF : HF = 25) :
  ∃ FG : ℝ, FG = 15 ∧
  (FG > EG - EF) ∧
  (FG > HF - HG) :=
by
  use 15
  split
  . rfl
  {
    split
    . linarith [hEF, hEG]
    . linarith [hHG, hHF]
  }

end triangle_inequality_FG_l552_552370


namespace find_p_l552_552041

def f (p : ℕ) [prime p] : ℕ → ℕ
| 0       := 0
| (n + 1) := if n = 0 then p + 1 else f n * f n + p

theorem find_p (p : ℕ) [prime p] : (∃ k, ∃ m, f p k = m * m) ↔ p = 3 :=
by sorry

end find_p_l552_552041


namespace complement_not_greater_than_angle_l552_552714

theorem complement_not_greater_than_angle (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 90) : θ < 90 - θ :=
by
  sorry

example : ∃ θ, 0 ≤ θ ∧ θ < 90 ∧ θ < 90 - θ :=
by
  use 60
  norm_num
  split
  norm_num
  split
  norm_num
  sorry

end complement_not_greater_than_angle_l552_552714


namespace maximum_yellow_balls_l552_552583

theorem maximum_yellow_balls (y : ℕ) (h_cond1 : 63 + 5 * y ≥ 0.85 * (70 + 7 * y)) : 
  y ≤ 3 → 70 + 7 * y ≤ 91 :=
by
  sorry

end maximum_yellow_balls_l552_552583


namespace length_50_times_l552_552638

-- Define the increment sequence
def increment_factor (n : Nat) : ℚ :=
  (n + 3) / (n + 2)

-- Define the overall multiplication factor
def overall_factor (n : Nat) : ℚ :=
  ∏ i in Finset.range(n + 1), increment_factor i

-- Problem statement
theorem length_50_times (n : Nat) : overall_factor n = 50 ↔ n = 147 :=
by
  sorry

end length_50_times_l552_552638


namespace obtuse_triangle_range_x_l552_552884

theorem obtuse_triangle_range_x 
  (A B C : Type)
  (AB AC BC : ℝ)
  (h1 : AB = 2)
  (h2 : AC = 5)
  (h3 : ∃ x : ℝ, BC = x)
  (h4 : x > 3)
  (h5 : x < 7)
  (h6 : ∡BAC = obtuse)
  : 3 < x ∧ x < sqrt 21 ∨ sqrt 29 < x ∧ x < 7 := 
  sorry

end obtuse_triangle_range_x_l552_552884


namespace num_nat_numbers_with_remainder_17_l552_552476

theorem num_nat_numbers_with_remainder_17 (N : ℕ) :
  (2017 % N = 17 ∧ N > 17) → 
  ({N | 2017 % N = 17 ∧ N > 17}.toFinset.card = 13) := 
by
  sorry

end num_nat_numbers_with_remainder_17_l552_552476


namespace range_of_xy_l552_552100

theorem range_of_xy {x y : ℝ} (h₁ : 0 < x) (h₂ : 0 < y)
    (h₃ : x + 2/x + 3*y + 4/y = 10) : 
    1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
by
  sorry

end range_of_xy_l552_552100


namespace Oliver_9th_l552_552945

def person := ℕ → Prop

axiom Ruby : person
axiom Oliver : person
axiom Quinn : person
axiom Pedro : person
axiom Nina : person
axiom Samuel : person
axiom place : person → ℕ → Prop

-- Conditions given in the problem
axiom Ruby_Oliver : ∀ n, place Ruby n → place Oliver (n + 7)
axiom Quinn_Pedro : ∀ n, place Quinn n → place Pedro (n - 2)
axiom Nina_Oliver : ∀ n, place Nina n → place Oliver (n + 3)
axiom Pedro_Samuel : ∀ n, place Pedro n → place Samuel (n - 3)
axiom Samuel_Ruby : ∀ n, place Samuel n → place Ruby (n + 2)
axiom Quinn_5th : place Quinn 5

-- Question: Prove that Oliver finished in 9th place
theorem Oliver_9th : place Oliver 9 :=
sorry

end Oliver_9th_l552_552945


namespace number_of_particular_propositions_l552_552008

def some_triangles_are_isosceles : Prop := 
  ∃ t : Type, is_triangle t ∧ is_isosceles t

def exists_x_in_z : Prop := 
  ∃ x : ℤ, x^2 - 2 * x - 3 = 0

def exists_triangle_with_sum_170 : Prop := 
  ∃ t : Type, is_triangle t ∧ interior_angle_sum t = 170

def rectangles_are_parallelograms : Prop := 
  ∀ r : Type, is_rectangle r → is_parallelogram r

theorem number_of_particular_propositions : 
  (∃ t : Type, is_triangle t ∧ is_isosceles t) ∧ 
  (∃ x : ℤ, x^2 - 2 * x - 3 = 0) ∧ 
  (∃ t : Type, is_triangle t ∧ interior_angle_sum t = 170) → 
  3 := 
sorry

end number_of_particular_propositions_l552_552008


namespace pet_store_profit_l552_552429

theorem pet_store_profit : 
  let cost_price := 100 in
  let selling_price := 3 * cost_price + 5 in
  selling_price - cost_price = 205 := by
  let cost_price := 100
  let selling_price := 3 * cost_price + 5
  sorry

end pet_store_profit_l552_552429


namespace destroyed_cakes_l552_552030

theorem destroyed_cakes (initial_cakes : ℕ) (half_falls : ℕ) (half_saved : ℕ)
  (h1 : initial_cakes = 12)
  (h2 : half_falls = initial_cakes / 2)
  (h3 : half_saved = half_falls / 2) :
  initial_cakes - half_falls / 2 = 3 :=
by
  sorry

end destroyed_cakes_l552_552030


namespace poly_roots_equivalence_l552_552746

noncomputable def poly (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem poly_roots_equivalence (a b c d : ℝ) 
    (h1 : poly a b c d 4 = 102) 
    (h2 : poly a b c d 3 = 102) 
    (h3 : poly a b c d (-3) = 102) 
    (h4 : poly a b c d (-4) = 102) : 
    {x : ℝ | poly a b c d x = 246} = {0, 5, -5} := 
by 
    sorry

end poly_roots_equivalence_l552_552746


namespace number_of_true_propositions_l552_552902

-- Definitions of the propositions
def prop1 : Prop := ∀ (l1 l2: Line) (P1 P2: Plane), (l1 ∥ P1) → (l2 ∥ P1) → (P1 ∥ P2)
def prop2 : Prop := ∀ (l1 l2 l3: Line), (l1 ⊥ l3) → (l2 ⊥ l3) → (l1 ∥ l2)
def prop3 : Prop := ∀ (P: Plane) (A: Point), (A ∉ P) → ∃! l : Line, (l ∥ P)
def prop4 : Prop := ∀ (l1 l2: Line) (P: Plane), (l1 ⊥ P) → (l2 ⊥ P) → (l1 ∥ l2)

-- Theorem stating the correct number of true propositions
theorem number_of_true_propositions : (prop1 = false) ∧ (prop2 = false) ∧ (prop3 = false) ∧ (prop4 = true) := by
  sorry

end number_of_true_propositions_l552_552902


namespace subtraction_and_multiplication_problem_l552_552343

theorem subtraction_and_multiplication_problem :
  (5 / 6 - 1 / 3) * 3 / 4 = 3 / 8 :=
by sorry

end subtraction_and_multiplication_problem_l552_552343


namespace iggy_wednesday_run_6_l552_552160

open Nat

noncomputable def iggy_miles_wednesday : ℕ :=
  let total_time := 4 * 60    -- Iggy spends 4 hours running (240 minutes)
  let pace := 10              -- Iggy runs 1 mile in 10 minutes
  let monday := 3
  let tuesday := 4
  let thursday := 8
  let friday := 3
  let total_miles_other_days := monday + tuesday + thursday + friday
  let total_time_other_days := total_miles_other_days * pace
  let wednesday_time := total_time - total_time_other_days
  wednesday_time / pace

theorem iggy_wednesday_run_6 :
  iggy_miles_wednesday = 6 := by
  sorry

end iggy_wednesday_run_6_l552_552160


namespace petya_sum_of_all_combinations_l552_552243

-- Define the expression with the possible placements of signs.
def petyaExpression : List (ℤ → ℤ → ℤ) :=
  [int.add, int.sub] -- Combination of possible operations at each position

-- Calculate the total number of ways to insert "+" and "-" in the expression
def number_of_combinations : ℕ := 2^5

-- Define the problem statement in Lean 4
theorem petya_sum_of_all_combinations : 
  (∑ idx in Finset.range number_of_combinations, 1) = 32 := by
  sorry

end petya_sum_of_all_combinations_l552_552243


namespace area_enclosed_by_curves_l552_552827

noncomputable def enclosed_area : ℝ :=
  ∫ x in 0..1, (Real.sqrt x - x^2)

theorem area_enclosed_by_curves :
  enclosed_area = 1 / 3 :=
by
  -- Proof goes here
  sorry

end area_enclosed_by_curves_l552_552827


namespace winning_post_distance_l552_552749

theorem winning_post_distance (v_A v_B D : ℝ) (hvA : v_A = (5 / 3) * v_B) (head_start : 80 ≤ D) :
  (D / v_A = (D - 80) / v_B) → D = 200 :=
by
  sorry

end winning_post_distance_l552_552749


namespace alice_plates_l552_552006

theorem alice_plates (
  initial_plates : ℕ := 27
  total_plates : ℕ := 123
  a1 : ℕ := 12
  d : ℕ := 3) :
  let total_added := total_plates - initial_plates,
      n := 4,
      last_addition := a1 + (n - 1) * d in
  total_added = 96 ∧ last_addition = 21 := by
  -- Proof contents can be filled here
  sorry

end alice_plates_l552_552006


namespace number_of_labelings_l552_552265

theorem number_of_labelings (a b c d e f g : ℕ)
  (h1 : {a, b, c, d, e, f, g}.sum = 28)
  (h2 : {a, b, c, d, e, f, g}.pairwise (≠))
  (h3 : (a + g + c) = x)
  (h4 : (b + g + d) = x)
  (h5 : (c + g + e) = x)
  : {n : ℕ // n = 144} :=
by
  have h : ∀ {g ∈ {1, 4, 7}}, ∃! x, 3 * x = 28 + 2 * g := sorry
  exact ⟨144, sorry⟩

end number_of_labelings_l552_552265


namespace Cl_invalid_electrons_l552_552748

noncomputable def Cl_mass_number : ℕ := 35
noncomputable def Cl_protons : ℕ := 17
noncomputable def Cl_neutrons : ℕ := Cl_mass_number - Cl_protons
noncomputable def Cl_electrons : ℕ := Cl_protons

theorem Cl_invalid_electrons : Cl_electrons ≠ 18 :=
by
  sorry

end Cl_invalid_electrons_l552_552748


namespace find_y_values_l552_552422

theorem find_y_values
  (y₁ y₂ y₃ y₄ y₅ : ℝ)
  (h₁ : y₁ + 3 * y₂ + 6 * y₃ + 10 * y₄ + 15 * y₅ = 3)
  (h₂ : 3 * y₁ + 6 * y₂ + 10 * y₃ + 15 * y₄ + 21 * y₅ = 20)
  (h₃ : 6 * y₁ + 10 * y₂ + 15 * y₃ + 21 * y₄ + 28 * y₅ = 86)
  (h₄ : 10 * y₁ + 15 * y₂ + 21 * y₃ + 28 * y₄ + 36 * y₅ = 225) :
  15 * y₁ + 21 * y₂ + 28 * y₃ + 36 * y₄ + 45 * y₅ = 395 :=
by {
  sorry
}

end find_y_values_l552_552422


namespace num_integers_for_inequality_l552_552455

theorem num_integers_for_inequality :
  ∃ (n : ℕ), n = 9 ∧ ∀ (y : ℤ), (3 * y^2 + 17 * y + 14 ≤ 22 ↔ y ∈ (set.Icc (-8 : ℤ) 0)) :=
by sorry

end num_integers_for_inequality_l552_552455


namespace intervals_of_monotonicity_minimum_value_l552_552907

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 2

theorem intervals_of_monotonicity :
  (∀ x, f' x > 0 ↔ x < -1 ∨ x > 3) ∧ (∀ x, f' x < 0 ↔ -1 < x ∧ x < 3) :=
by {

  sorry
}

theorem minimum_value (m : ℝ) (hm : m > -1) :
  ∃ (minimum : ℝ), minimum = 
  if -1 < m ∧ m ≤ 3 then m^3 - 3*m^2 - 9*m + 2
  else if m > 3 then -25 else 0 :=
by {
  sorry
}

end intervals_of_monotonicity_minimum_value_l552_552907


namespace root_difference_l552_552829

theorem root_difference (a b c : ℝ) (ha : 81 * a^3 - 162 * a^2 + 90 * a - 10 = 0)
  (hb : 81 * b^3 - 162 * b^2 + 90 * b - 10 = 0)
  (hc : 81 * c^3 - 162 * c^2 + 90 * c - 10 = 0)
  (h : ∃ α β γ : ℝ, α + β + γ = 2 ∧ α * β * γ = 10 / 81 ∧ (α = 2 * β ∨ β = 2 * α)) :
  (∃ r1 r2 r3 : ℝ, r1 < r2 ∧ r2 < r3 ∧ r1 ∈ {a, b, c} ∧ r3 ∈ {a, b, c}) →
  (r3 - r1 = 1) :=
by
  sorry

end root_difference_l552_552829


namespace blue_socks_count_l552_552193

theorem blue_socks_count (total_socks : ℕ) (two_thirds_white : ℕ) (one_third_blue : ℕ) 
  (h1 : total_socks = 180) 
  (h2 : two_thirds_white = (2 / 3) * total_socks) 
  (h3 : one_third_blue = total_socks - two_thirds_white) : 
  one_third_blue = 60 :=
by
  sorry

end blue_socks_count_l552_552193


namespace blue_socks_count_l552_552196

-- Defining the total number of socks
def total_socks : ℕ := 180

-- Defining the number of white socks as two thirds of the total socks
def white_socks : ℕ := (2 * total_socks) / 3

-- Defining the number of blue socks as the difference between total socks and white socks
def blue_socks : ℕ := total_socks - white_socks

-- The theorem to prove
theorem blue_socks_count : blue_socks = 60 := by
  sorry

end blue_socks_count_l552_552196


namespace cube_adjacency_false_l552_552318

theorem cube_adjacency_false (space_partitioned_into_identical_cubes : Prop) : 
  ∃ (K : Cube), ∀ (K' : Cube), (shares_face_with(K, K') → false) := 
sorry

end cube_adjacency_false_l552_552318


namespace ratio_of_sum_of_divisors_l552_552619

theorem ratio_of_sum_of_divisors (N : ℕ) (h1 : N = 64 * 45 * 91 * 49) :
  let a := (1 + 3 + 3^2) * (1 + 5) * (1 + 7 + 7^2 + 7^3) * (1 + 13) in
  let sum_all_divisors := (1 + 2 + 2^2 + 2^3 + 2^4 + 2^5 + 2^6) * (1 + 3 + 3^2) * (1 + 5) * (1 + 7 + 7^2 + 7^3) * (1 + 13) in
  let sum_even_divisors := sum_all_divisors - a in
  (a : ℚ) / sum_even_divisors = 1 / 126 :=
by
  sorry

end ratio_of_sum_of_divisors_l552_552619


namespace quadrilateral_is_rectangle_l552_552170

variables (Q : Type) [quadrilateral Q]
variables [h1 : has_two_right_angles Q] [h2 : diagonals_equal Q]

theorem quadrilateral_is_rectangle (Q) :
  is_rectangle Q :=
by
 sorry

end quadrilateral_is_rectangle_l552_552170


namespace remainder_div_l552_552839

open Polynomial

noncomputable def dividend : ℤ[X] := X^4
noncomputable def divisor  : ℤ[X] := X^2 + 3 * X + 2

theorem remainder_div (f g : ℤ[X]) : (f % g) = -6 * X - 6 :=
by
  have f := dividend
  have g := divisor
  sorry

end remainder_div_l552_552839


namespace product_of_roots_eq_20_l552_552740

open Real

theorem product_of_roots_eq_20 :
  (∀ x : ℝ, (x^2 + 18 * x + 30 = 2 * sqrt (x^2 + 18 * x + 45)) → 
  (x^2 + 18 * x + 20 = 0)) → 
  ∀ α β : ℝ, (α ≠ β ∧ α * β = 20) :=
by
  intros h x hx
  sorry

end product_of_roots_eq_20_l552_552740


namespace magnitude_vector_proof_l552_552140

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (-1, 1)

theorem magnitude_vector_proof : ‖(2 • vector_a.1 - vector_b.1, 2 • vector_a.2 - vector_b.2)‖ = 3 * Real.sqrt 2 :=
by 
  have a : ℝ × ℝ := (1, 2)
  have b : ℝ × ℝ := (-1, 1)
  calc
    ‖(2 • (1, 2) - (-1, 1))‖
    = ‖ (2 * 1 - (-1), 2 * 2 - 1) ‖ : sorry
    ... = ‖ (2 + 1, 4 - 1) ‖ : sorry
    ... = ‖ (3, 3) ‖ : sorry
    ... = Real.sqrt (3^2 + 3^2) : sorry
    ... = Real.sqrt (9 + 9) : sorry
    ... = Real.sqrt 18 : sorry
    ... = 3 * Real.sqrt 2 : sorry

end magnitude_vector_proof_l552_552140


namespace distance_relationship_l552_552734

open Real

variables (O : Point) (r1 r2 : ℝ) (h : r1 > r2)
variables (AB : Line)
variables (A B D C E : Point)
variables (y x : ℝ)
variables (h1 : distance (center_of_circle O) A = r1)
variables (h2 : chord (circle O r1) A D tangent_to (circle O r2) D)
variables (h3 : tangent (circle O r1) B extension (line A D) (at C))
variables (h4 : distance A E = distance D C)
variables (h5 : y = perpendicular_distance E (line A B))
variables (h6 : x = distance E (tangent_line (circle O r1) A))

theorem distance_relationship :
  y^2 = x^3 / (2 * r1 - x) :=
sorry

end distance_relationship_l552_552734


namespace area_of_triangle_is_14_l552_552595

noncomputable theory
open Complex

def area_of_triangle (w: ℂ) : ℝ :=
  (1 / 2 : ℝ) * Complex.abs 
  (w * Complex.conj (w^2) - w^2 * Complex.conj (w^3) + w^3 * Complex.conj (w)
  - w * Complex.conj (w^3) + w^2 * Complex.conj (w) - w^3 * Complex.conj (w^2))

theorem area_of_triangle_is_14 (w: ℂ) (h₀: w ≠ 0) (h₁: w ≠ 1)
  (h₂: w^3 - w = i * (w^2 - w) ∨ w^3 - w = -i * (w^2 - w)) : 
  area_of_triangle w = 14 :=
sorry

end area_of_triangle_is_14_l552_552595


namespace junior_score_l552_552948

theorem junior_score (total_students juniors seniors avg_score avg_senior_score : ℕ) (h_juniors: juniors = total_students / 5)
                     (h_seniors: seniors = 4 * total_students / 5) 
                     (h_avg_score: avg_score = 85) (h_avg_senior_score: avg_senior_score = 82) 
                     (h_total_score: total_students * avg_score = juniors * junior_score + seniors * avg_senior_score) 
    : junior_score = 97 := by
  sorry

end junior_score_l552_552948


namespace maxBalancedSetSum_eq_12859_l552_552397

def isBalancedSet (S : Set ℕ) (n : ℕ) := 
  ∀ k : ℕ, (1 ≤ k ∧ k ≤ n) → 
  (∀ subset : Finset ℕ, subset.card = k → 
  ((∑ i in subset, i : ℕ) % k = 0))

def maxBalancedSum (S : Set ℕ) (n : ℕ) (M : ℕ) : ℕ :=
  if isBalancedSet S n ∧ (∀ s ∈ S, s ≤ M) then ∑ i in S, i else 0

theorem maxBalancedSetSum_eq_12859 :
  ∃ S : Set ℕ, isBalancedSet S 7 ∧ (∀ s ∈ S, s ≤ 2017) ∧ maxBalancedSum S 7 2017 = 12859 :=
sorry

end maxBalancedSetSum_eq_12859_l552_552397


namespace range_of_f_l552_552131

def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem range_of_f : 
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ Real.pi / 2) → 
  -3/2 ≤ f x ∧ f x ≤ 3 := 
by
  sorry

end range_of_f_l552_552131


namespace part_I_part_II_l552_552127

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

def is_monotonically_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ a b ∈ I, a < b → f a < f b

theorem part_I
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = Real.sin (2 * x - Real.pi / 6))
  (hω : 2 > 0)
  (hϕ : 0 < Real.pi / 6 ∧ Real.pi / 6 < Real.pi / 2) :
  (f = λ x, Real.sin (2 * x - Real.pi / 6)) ∧
  (is_monotonically_increasing_on f (Set.Ioc 0 (Real.pi / 3)) ∧
   is_monotonically_increasing_on f (Set.Ioc (5 * Real.pi / 6) Real.pi)) :=
sorry

theorem part_II
  (A : ℝ)
  (hA : f (A / 2) + Real.cos A = 1 / 2) :
  A = 2 * Real.pi / 3 :=
sorry

end part_I_part_II_l552_552127


namespace greatest_prime_factor_of_product_l552_552223

def even_product : ℕ := 
  2 * 4 * 6 * 8 * 10 * 12 * 14 * 16 * 18 * 20

theorem greatest_prime_factor_of_product : 
  Nat.greatest_prime_factor (even_product * even_product) = 7 :=
by
  sorry

end greatest_prime_factor_of_product_l552_552223


namespace translate_B_coordinates_l552_552963

namespace TranslatedSegment

def point : Type := ℝ × ℝ

def translate (p : point) (v : point) : point :=
(p.1 + v.1, p.2 + v.2)

theorem translate_B_coordinates :
  ∀ (A A' B B' : point),
  A = (-1, -1) → A' = (3, -1) →
  B = (1, 2) →
  let translation_vector := (A'.1 - A.1, A'.2 - A.2) in
  B' = translate B translation_vector →
  B' = (5, 2) := by
  intros
  sorry

end TranslatedSegment

end translate_B_coordinates_l552_552963


namespace general_term_sum_of_b_l552_552594

-- Definitions of sequences as given conditions
def a (n : ℕ) : ℤ := 2 * n - 1

-- Conditions: given 2a_1 + 3a_2 = 11 and 2a_3 = a_2 + a_6 - 4

lemma sequence_conditions :
  2 * a 1 + 3 * a 2 = 11 ∧
  2 * a 3 = a 2 + a 6 - 4 :=
by {
  -- Note: we skip the proof details as they are strictly solution steps
  sorry
}

-- Definition: Sum of first n terms of sequence a
def S (n : ℕ) : ℤ := n * n

-- Definition: Sequence b_n = 1 / (S_n + n)
def b (n : ℕ) : ℚ := 1 / (S n + n)

-- The problem's questions translated to proving in Lean

theorem general_term (n : ℕ) : a n = 2 * n - 1 :=
by {
  -- Problem statement directly derived from conditions
  sorry
}

theorem sum_of_b (n : ℕ) : (∑ k in Finset.range n, b (k + 1)) = n / (n + 1) :=
by {
  -- Translation of sum of series to problem statement
  sorry
}

end general_term_sum_of_b_l552_552594


namespace value_of_a_10_l552_552544

theorem value_of_a_10 : 
  (n : ℕ) → (h : n > 0) → (curve_eq : ∀ x : ℕ, x > 0 → y = x^2 + 1 → a x = y) → 
  a 10 = 101 :=
by
  sorry

end value_of_a_10_l552_552544


namespace range_of_m_l552_552581

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 3) (h4 : ∀ x y, x > 0 → y > 0 → x + y = 3 → (4 / (x + 1) + 16 / y > m^2 - 3 * m + 11)) : 1 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l552_552581


namespace focus_y_axis_range_alpha_l552_552899

theorem focus_y_axis_range_alpha (α : ℝ) : 
  (0 ≤ α ∧ α < 2 * Real.pi ∧ 
  (∃ x y : ℝ, x^2 * Real.sin α - y^2 * Real.cos α = 1)) →
  ((∃ x y : ℝ, x^2 * Real.sin α - y^2 * Real.cos α = 1) → 
  (∃ a b c : ℝ, a = 1 / Real.sin α ∧ b = 1 / -Real.cos α ∧ a < b ∧ 0 < b ∧ 0 < Real.sin α)):
  ((Real.pi / 2) < α ∧ α < (3 * Real.pi / 4)) :=
begin
  sorry
end

end focus_y_axis_range_alpha_l552_552899


namespace inconsistency_exists_l552_552011

-- Define the ages of Kristine, Ann and Brad
variables (K : ℕ) (Ann Brad : ℕ)

-- Conditions based on the problem description
def age_conditions : Prop :=
  Ann = K + 5 ∧
  Brad = K + 2 ∧
  Brad = 2 * K

-- Inconsistency in the second condition about ages in 10 years
def age_inconsistency (K Ann Brad : ℕ) : Prop :=
  (K + 10) = 24 + ((Ann + 10) - (K + 10))

theorem inconsistency_exists (K Ann Brad : ℕ) :
  age_conditions K Ann Brad → ¬ (age_inconsistency K Ann Brad) :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  sorry

end inconsistency_exists_l552_552011


namespace num_primes_between_70_and_80_correct_l552_552144

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

noncomputable def num_primes_between_70_and_80 : ℕ :=
  finset.card $ finset.filter is_prime (finset.range 81) \ (finset.range 71)

theorem num_primes_between_70_and_80_correct :
  num_primes_between_70_and_80 = 3 :=
by {
  sorry
}

end num_primes_between_70_and_80_correct_l552_552144


namespace joan_lost_balloons_l552_552983

theorem joan_lost_balloons :
  let initial_balloons := 9
  let current_balloons := 7
  let balloons_lost := initial_balloons - current_balloons
  balloons_lost = 2 :=
by
  sorry

end joan_lost_balloons_l552_552983


namespace total_area_of_five_equilateral_triangles_l552_552495

theorem total_area_of_five_equilateral_triangles
  (side_length : ℝ)
  (n : ℕ)
  (half_overlap : ℝ) 
  (area_one_triangle : ℝ)
  (area_overlap : ℝ)
  (total_area_no_overlap : ℝ)
  (total_overlap : ℝ) :
  n = 5 →
  side_length = 4 * real.sqrt 3 →
  area_one_triangle = (real.sqrt 3 / 4) * side_length^2 →
  total_area_no_overlap = n * area_one_triangle →
  half_overlap = 1 / 2 →
  area_overlap = half_overlap * area_one_triangle →
  total_overlap = (n - 1) * area_overlap →
  total_area_no_overlap - total_overlap = 36 * real.sqrt 3 :=
sorry

end total_area_of_five_equilateral_triangles_l552_552495


namespace three_digit_prob_div_3_with_ones_digit_3_l552_552320

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def has_ones_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_three_digit_positive (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def probability_divisible_by_3 (n : ℕ) : ℚ :=
  if is_divisible_by_3 n then 1 / 3 else 0

theorem three_digit_prob_div_3_with_ones_digit_3 :
  ∀ (N : ℕ), is_three_digit_positive N → has_ones_digit_3 N → probability_divisible_by_3 N = 1 / 3 :=
sorry

end three_digit_prob_div_3_with_ones_digit_3_l552_552320


namespace triangle_S_acute_and_triangle_T_obtuse_l552_552157

def angle (n : ℕ) : Type := {
  x // 0 < n ∧ n < 180
}

def Triangle := angle 60 :: angle 45 :: angle 75 :: []

def cos_eq_sin_of_triangle : Prop :=
  ∀ θ_S θ_T : angle, 
    θ_S ∈ Triangle → 
    θ_T ∈ [angle 30, angle 135, angle 15] → 
    cos θ_S = sin θ_T

theorem triangle_S_acute_and_triangle_T_obtuse : (cos_eq_sin_of_triangle) → 
  (is_acute Triangle) ∧ (is_obtuse [angle 30, angle 135, angle 15]) :=
sorry

end triangle_S_acute_and_triangle_T_obtuse_l552_552157


namespace valid_arrangements_count_l552_552496

-- There are 5 people standing in a row
variable (A B P1 P2 P3 : Type)

-- Definition of non-adjacency and ordering for persons A and B
-- In practice, these definitions would need to be formalized, but here we outline the overall approach
def non_adjacent (l : List Type) : Prop :=
  let pos_A := List.indexOf A l in
  let pos_B := List.indexOf B l in
  (List.length l > 1) ∧ (pos_A > -1) ∧ (pos_B > -1) ∧ abs (pos_A - pos_B) > 1

def A_left_of_B (l : List Type) : Prop :=
  let pos_A := List.indexOf A l in
  let pos_B := List.indexOf B l in
  pos_A < pos_B

-- Definition of valid arrangement: non-adjacent and A to the left of B
def valid_arrangement (l : List Type) : Prop :=
  non_adjacent A B l ∧ A_left_of_B A B l

-- Proving the number of valid arrangements is 36
theorem valid_arrangements_count : ∃ l : List Type, length l = 5 ∧ valid_arrangement A B l = 36 := sorry

end valid_arrangements_count_l552_552496


namespace maltese_cross_area_l552_552039

noncomputable def area_of_maltese_cross (area_A B: ℕ) : ℕ :=
  let area_A_B := area_A + area_B
  17 * area_A_B

theorem maltese_cross_area (area_A area_B : ℕ) (condition : area_A + area_B = 1) :
  area_of_maltese_cross area_A area_B = 17 :=
by
  -- Proof skipped
  sorry

end maltese_cross_area_l552_552039


namespace backyard_area_l552_552593

-- Definitions from conditions
def length : ℕ := 1000 / 25
def perimeter : ℕ := 1000 / 10
def width : ℕ := (perimeter - 2 * length) / 2

-- Theorem statement: Given the conditions, the area of the backyard is 400 square meters
theorem backyard_area : length * width = 400 :=
by 
  -- Sorry to skip the proof as instructed
  sorry

end backyard_area_l552_552593


namespace collinear_DIO_l552_552184

section TangentsAndCollinearity

variable {A B C D X Y I O : Type}

-- Definitions of points and collinearity
axiom triangle_ABC : Triangle A B C
axiom incircle_tangency_points : IsTangencyPoint incircle triangle_ABC (X, Y)
axiom circumcircle_tangent_at_A : IsTangent circumcircle triangle_ABC A D
axiom incircle_center : Center incircle I
axiom circumcircle_center : Center circumcircle O

-- Given condition: D, X, and Y are collinear
axiom collinear_DXY : Collinear D X Y

-- The theorem to prove: D, I, and O are collinear
theorem collinear_DIO : Collinear D I O := 
sorry

end TangentsAndCollinearity

end collinear_DIO_l552_552184


namespace C_is_14_years_younger_than_A_l552_552321

variable (A B C D : ℕ)

-- Conditions
axiom cond1 : A + B = (B + C) + 14
axiom cond2 : B + D = (C + A) + 10
axiom cond3 : D = C + 6

-- To prove
theorem C_is_14_years_younger_than_A : A - C = 14 :=
by
  sorry

end C_is_14_years_younger_than_A_l552_552321


namespace quadratic_expression_l552_552539

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_expression :
  ∃ (a b c : ℝ),
  (∀ x, quadratic_function a b c (2) = -6) ∧
  (∀ x, quadratic_function a b c (1) = -4) ∧
  (∃ x_max, ∀ x_neq : x ≠ x_max, quadratic_function a b c x ≤ quadratic_function a b c x_max) →
  quadratic_function a b c x = -2 * x^2 + 4 * x - 6 :=
sorry

end quadratic_expression_l552_552539


namespace log_sum_100_l552_552099

variable {a : ℕ → ℝ}

def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem log_sum_100 (h₁ : is_geometric a) (h₂ : a 1 * a 100 = 64) :
  ∑ n in Finset.range 100, Real.logBase 2 (a (n + 1)) = 300 := 
sorry

end log_sum_100_l552_552099


namespace total_trip_cost_l552_552957

def distance_AC : ℝ := 4000
def distance_AB : ℝ := 4250
def bus_rate : ℝ := 0.10
def plane_rate : ℝ := 0.15
def boarding_fee : ℝ := 150

theorem total_trip_cost :
  let distance_BC := Real.sqrt (distance_AB ^ 2 - distance_AC ^ 2)
  let flight_cost := distance_AB * plane_rate + boarding_fee
  let bus_cost := distance_BC * bus_rate
  flight_cost + bus_cost = 931.15 :=
by
  sorry

end total_trip_cost_l552_552957


namespace book_price_decrease_l552_552359

open Real

theorem book_price_decrease (P : ℝ) :
  let decreased_price := P - 0.25 * P in
  let increased_price := decreased_price + 0.20 * decreased_price in
  P - increased_price = 0.10 * P :=
by
  sorry

end book_price_decrease_l552_552359


namespace petya_sum_l552_552235

theorem petya_sum : 
  let f (signs : fin 5 → bool) : ℤ :=
    1 + (if signs 0 then 2 else -2) + (if signs 1 then 3 else -3) + (if signs 2 then 4 else -4) + (if signs 3 then 5 else -5) + (if signs 4 then 6 else -6),
  sum (f '' (finset.univ : finset (fin 5 → bool))) = 32 :=
by
  sorry

end petya_sum_l552_552235


namespace n_n_plus_1_divisible_by_2_l552_552418

theorem n_n_plus_1_divisible_by_2 (n : ℤ) (h1 : 1 ≤ n) (h2 : n ≤ 99) : (n * (n + 1)) % 2 = 0 := 
sorry

end n_n_plus_1_divisible_by_2_l552_552418


namespace player_A_wins_even_n_l552_552991

theorem player_A_wins_even_n (n : ℕ) (hn : n > 0) (even_n : Even n) :
  ∃ strategy_A : ℕ → Bool, 
    ∀ (P Q : ℕ), P % 2 = 0 → (Q + P) % 2 = 0 :=
by 
  sorry

end player_A_wins_even_n_l552_552991


namespace suraj_innings_l552_552277

theorem suraj_innings (n A : ℕ) (h1 : A + 6 = 16) (h2 : (n * A + 112) / (n + 1) = 16) : n = 16 :=
by
  sorry

end suraj_innings_l552_552277


namespace collinearity_equivalence_l552_552205

variables 
  (ABC A1 A2 B1 B2 C1 C2 : Type*)
  [inhabited ABC]
  [inhabited A1] [inhabited A2]
  [inhabited B1] [inhabited B2]
  [inhabited C1] [inhabited C2]
  (angle1 angle2 : Angle)
  (collinear : Set (Type*) → Prop)

namespace triangle_problem

axiom is_triangle (A B C : ABC)

axiom on_line (P Q R : Type*) : Prop 

axiom angle_eq {P Q R S T : Type*} (A B : angle1) (C D : angle2) : Prop

theorem collinearity_equivalence
  (h1 : on_line A1 A B)
  (h2 : on_line A2 A C)
  (h3 : on_line B1 B C)
  (h4 : on_line B2 B A)
  (h5 : on_line C1 C A)
  (h6 : on_line C2 C B)
  (h_angle1 : angle_eq (angle1.val ABC) (angle1.val A1 A B) (angle1.val A2 A C))
  (h_angle2 : angle_eq (angle1.val ABC) (angle1.val B1 B C) (angle1.val B2 B A))
  (h_angle3 : angle_eq (angle1.val ABC) (angle1.val C1 C A) (angle1.val C2 C B))
  : (collinear {A1, B1, C1} ↔ collinear {A2, B2, C2}) :=
sorry
end triangle_problem

end collinearity_equivalence_l552_552205


namespace sum_cos_sq_sin_sq_l552_552444

theorem sum_cos_sq_sin_sq :
  2 * (∑ i in (Finset.range 46), (Real.cos (i * Real.pi / 180))^2) + 
  2 * (∑ i in (Finset.range 46), (Real.sin (i * Real.pi / 180))^2) = 92 :=
by
  sorry

end sum_cos_sq_sin_sq_l552_552444


namespace smallest_n_rotation_identity_l552_552062

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![\![Real.cos θ, -Real.sin θ],
    [Real.sin θ, Real.cos θ]]!

theorem smallest_n_rotation_identity :
  let M := rotation_matrix (3 * Real.pi / 2) in
  (∃ (n : ℕ), M ^ n = Matrix.identity _ _ ∧ ∀ (m : ℕ), m > 0 → m < n → M ^ m ≠ Matrix.identity _ _) :=
begin
  let M := rotation_matrix (3 * Real.pi / 2),
  use 4,
  split,
  {
    sorry
  },
  {
    intros m hm1 hm2,
    sorry
  }
end

end smallest_n_rotation_identity_l552_552062


namespace petya_sum_of_expressions_l552_552241

theorem petya_sum_of_expressions :
  (∑ val in (Finset.univ : Finset (Fin 32)), (1 +
    (if val / 2^4 % 2 = 0 then 2 else -2) +
    (if val / 2^3 % 2 = 0 then 3 else -3) +
    (if val / 2^2 % 2 = 0 then 4 else -4) +
    (if val / 2 % 2 = 0 then 5 else -5) +
    (if val % 2 = 0 then 6 else -6))) = 32 := 
by
  sorry

end petya_sum_of_expressions_l552_552241


namespace hyperbola_properties_l552_552910

theorem hyperbola_properties (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) 
  (h₂ : 2 * a = 4) (h₃ : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1) 
  (h₄ : ∃ x1 x2, (4 * b^2) / |4 - b^2| = 20) :
  (a = 2 ∧ b^2 = 5) ∧ (∀ x y, (x^2 / 4) - (y^2 / 5) = 1) ∧ 
  (∀ x, y = x - 2 → x^2 / 4 - y^2 / 5 = 1 → ∃ k, y = k * x ∧ (k = (Real.sqrt 5 / 2) ∨ k = -(Real.sqrt 5 / 2))) :=
by
  sorry

end hyperbola_properties_l552_552910


namespace probability_A_wins_l552_552168

theorem probability_A_wins : 
  let n := 2020
  let total_numbers := n - 2 + 1 
  let sequence := list.range' 2 total_numbers 
  let gcd_one (x y : ℕ) := Nat.gcd x y = 1
  in (∃ (a b : ℕ), a ∈ sequence ∧ b ∈ sequence ∧ a ≠ b ∧ gcd_one a b) → 
     ((total_numbers - 1) / (n - 1)).to_rat = 1010 / 2019 :=
begin
  sorry
end

end probability_A_wins_l552_552168


namespace sin_difference_as_product_l552_552823

theorem sin_difference_as_product (a b : ℝ) :
  sin (2 * a + b) - sin b = 2 * cos (a + b) * sin a :=
sorry

end sin_difference_as_product_l552_552823


namespace min_number_of_students_l552_552773

theorem min_number_of_students 
  (n : ℕ)
  (h1 : 25 ≡ 99 [MOD n])
  (h2 : 8 ≡ 119 [MOD n]) : 
  n = 37 :=
by sorry

end min_number_of_students_l552_552773


namespace max_handshakes_l552_552684

theorem max_handshakes (n : ℕ) (h_cond1 : n = 20)
  (h_cond2 : ∀ (a b c : ℕ), (a < n ∧ b < n ∧ c < n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) →
    ¬(acquainted a b ∧ acquainted b c ∧ acquainted a c))
  (h_handshake : ∀ (a b : ℕ), a < n ∧ b < n ∧ a ≠ b → handshake a b) : Prop :=
  max_handshakes = 100

end max_handshakes_l552_552684


namespace roof_length_width_difference_l552_552363

noncomputable def length_and_width_difference : ℝ :=
  let w := Real.sqrt 147
  let l := 4 * w
  l - w

theorem roof_length_width_difference :
  ∃ (w l : ℝ), l = 4 * w ∧ l * w = 588 ∧ (l - w) = 36.372 :=
by
  use (Real.sqrt 147)
  use 4 * (Real.sqrt 147)
  split
  . simp
  split
  . nlinarith [Real.sqrt 147]
  . sorry

end roof_length_width_difference_l552_552363


namespace divisible_by_3_l552_552825

theorem divisible_by_3 (n : ℕ) : (n * 2^n + 1) % 3 = 0 ↔ n % 6 = 1 ∨ n % 6 = 2 := 
sorry

end divisible_by_3_l552_552825


namespace find_u_given_roots_of_quadratic_l552_552151

-- Define the initial conditions
variables (k l : ℝ)
variable (r1 r2 : ℝ)
hypothesis roots_initial : ∀ x, (x = r1 ∨ x = r2) ↔ x^2 + k*x + l = 0

-- Define the proof statement
theorem find_u_given_roots_of_quadratic :
  (r1 + r2 = -k) →
  (r1 * r2 = l) →
  ∃ u : ℝ, ∀ x, (x = r1^2 ∨ x = r2^2) ↔ x^2 + u*x + v = 0 ∧ u = -k^2 + 2*l :=
by
  intros h1 h2
  exists -k^2 + 2*l
  intros x
  split
  sorry

end find_u_given_roots_of_quadratic_l552_552151


namespace atomic_weight_of_iodine_is_correct_l552_552060

noncomputable def atomic_weight_iodine (atomic_weight_nitrogen : ℝ) (atomic_weight_hydrogen : ℝ) (molecular_weight_compound : ℝ) : ℝ :=
  molecular_weight_compound - (atomic_weight_nitrogen + 4 * atomic_weight_hydrogen)

theorem atomic_weight_of_iodine_is_correct :
  atomic_weight_iodine 14.01 1.008 145 = 126.958 :=
by
  unfold atomic_weight_iodine
  norm_num

end atomic_weight_of_iodine_is_correct_l552_552060


namespace tangent_lines_intersection_at_AB_l552_552524

noncomputable def line_tangency_equation (M : ℝ × ℝ) (C_center : ℝ × ℝ) (C_r : ℝ) : ℝ × ℝ → Prop :=
  λ (AB : ℝ × ℝ), AB.1 * sqrt 3 - AB.2 = 0
  
theorem tangent_lines_intersection_at_AB :
  ∀ (M : ℝ × ℝ) (C : ℝ × ℝ → Prop), 
  M = (sqrt 3, 0) →
  C = λ (p : ℝ × ℝ), (p.1)^2 + (p.2 - 1)^2 = 1 →
  ∃ (AB : ℝ × ℝ), line_tangency_equation M (0, 1) 1 AB :=
begin
  intros M C hM hC,
  use (1, sqrt 3), 
  rw line_tangency_equation,
  sorry,
end

end tangent_lines_intersection_at_AB_l552_552524


namespace min_cells_to_paint_l552_552736

structure Grid4x4 :=
(cells : Fin 4 × Fin 4 → Bool) -- Boolean indicates if a cell is red

structure LShape :=
(cells : Fin 3 → (Fin 4 × Fin 4))
(is_corner : ∃ (i j : Fin 4), (cells 0 = (i, j) ∧ cells 1 = (i, j.succ) ∧ cells 2 = (i.succ, j)) ∨
                                 (cells 0 = (i, j) ∧ cells 1 = (i.succ, j) ∧ cells 2 = (i, j.succ)) ∨
                                 (cells 0 = (i, j) ∧ cells 1 = (i.pred, j) ∧ cells 2 = (i.pred, j.succ)) ∨
                                 (cells 0 = (i, j) ∧ cells 1 = (i, j.pred) ∧ cells 2 = (i.succ, j.pred)))

noncomputable def prevent_L_shapes (g : Grid4x4) : Prop :=
∀ (s1 s2 s3 s4 : LShape), 
  ¬(∀ (n : Fin 3), ¬g.cells (s1.cells n) ∧
        ¬g.cells (s2.cells n) ∧
        ¬g.cells (s3.cells n) ∧
        ¬g.cells (s4.cells n))

theorem min_cells_to_paint : ∃ (m : Fin 17), m ≤ 3 ∧ (∃ (g : Grid4x4), prevent_L_shapes g)
   sorry

end min_cells_to_paint_l552_552736


namespace valid_students_count_l552_552652

-- Definitions of conditions
def is_valid_number_of_students (n : ℕ) : Prop :=
  n > 0 ∧ 108 % n = 0

-- The main proof statement
theorem valid_students_count : ∃ n : ℕ, n = 12 ∨ n = 36 ∨ n = 54 ∧ is_valid_number_of_students n :=
begin
  sorry
end

end valid_students_count_l552_552652


namespace find_solutions_l552_552057

theorem find_solutions (x y z : ℝ) :
  (x = 5 / 3 ∧ y = -4 / 3 ∧ z = -4 / 3) ∨
  (x = 4 / 3 ∧ y = 4 / 3 ∧ z = -5 / 3) →
  (x^2 - y * z = abs (y - z) + 1) ∧ 
  (y^2 - z * x = abs (z - x) + 1) ∧ 
  (z^2 - x * y = abs (x - y) + 1) :=
by
  sorry

end find_solutions_l552_552057


namespace intersection_complement_eq_l552_552557

-- Definitions of the sets M and N
def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 > 0}

-- Complements with respect to the reals
def complement_R (A : Set ℝ) : Set ℝ := {x | x ∉ A}

-- Target goal to prove
theorem intersection_complement_eq :
  M ∩ (complement_R N) = {1, 2} :=
by
  sorry

end intersection_complement_eq_l552_552557


namespace factorization_identity_l552_552464

noncomputable def factor_expression (a b c : ℝ) : ℝ :=
  ((a ^ 2 + 1 - (b ^ 2 + 1)) ^ 3 + ((b ^ 2 + 1) - (c ^ 2 + 1)) ^ 3 + ((c ^ 2 + 1) - (a ^ 2 + 1)) ^ 3) /
  ((a - b) ^ 3 + (b - c) ^ 3 + (c - a) ^ 3)

theorem factorization_identity (a b c : ℝ) : 
  factor_expression a b c = (a + b) * (b + c) * (c + a) := 
by 
  sorry

end factorization_identity_l552_552464


namespace is_odd_and_decreasing_l552_552415

-- Define the functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := if x = 0 then 0 else 1 / x
def h (x : ℝ) : ℝ := x
def k (x : ℝ) : ℝ := -x^3

-- State the theorem
theorem is_odd_and_decreasing : 
  (∀ x : ℝ, k (-x) = - k x) ∧ (∀ x y : ℝ, x < y → k x > k y) :=
by
  sorry

end is_odd_and_decreasing_l552_552415


namespace simplify_f_x_range_omega_find_a_l552_552120

noncomputable def f (x : ℝ) : ℝ := 
  4 * (sin (π / 4 + x / 2))^2 * sin x + (cos x + sin x) * (cos x - sin x) - 1

-- P1: Simplify f(x)
theorem simplify_f_x (x : ℝ) : f x = 2 * sin x := 
  sorry

-- P2: Range of ω for f(ω x) being increasing in interval
theorem range_omega (ω : ℝ) : (∀ x ∈ Icc (-(π / 2)) (2 * π / 3), (f (ω * x) = y) → y is increasing) →
  0 < ω ∧ ω ≤ (3 / 4) := 
  sorry

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 
  1 / 2 * (f (2 * x) + a * f x - a * f (π / 2 - x) - a) - 1

-- P3: Find value of a
theorem find_a (a : ℝ) (max_value : ℝ) (x : ℝ) : 
  (∀ x ∈ Icc (-(π / 4)) (π / 2), g x a = max_value) →
  max_value = 2 →
  (a = -2 ∨ a = 6) := 
  sorry

end simplify_f_x_range_omega_find_a_l552_552120


namespace smallest_n_for_T_n_integer_l552_552997

def L : ℚ := ∑ i in {1, 2, 3, 4}, 1 / i

theorem smallest_n_for_T_n_integer : ∃ n ∈ ℕ, n > 0 ∧ (n * 5^(n-1) * L).denom = 1 ∧ n = 12 :=
by
  have hL : L = 25 / 12 := by sorry
  existsi 12
  split
  exact Nat.succ_pos'
  split
  suffices (12 * 5^(12-1) * 25 / 12).denom = 1 by sorry
  sorry
  rfl

end smallest_n_for_T_n_integer_l552_552997


namespace min_value_a2_b2_l552_552534

theorem min_value_a2_b2 {a b x₀ : ℝ} (hx₀ : x₀ ∈ set.Icc (1/4 : ℝ) real.exp 1)
  (h_zero : 2 * a * real.sqrt x₀ + b = real.exp (x₀ / 2)) :
  a^2 + b^2 = real.exp (3 / 4) / 4 :=
sorry

end min_value_a2_b2_l552_552534


namespace segments_in_tetrahedron_form_triangle_l552_552201

variables {A B C D M N : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
           [MetricSpace M] [MetricSpace N]

-- Definitions based on the problem statement
def regular_tetrahedron (A B C D : Type) : Prop := 
  ∃ (e : ℝ), e > 0 ∧ dist A B = e ∧ dist B C = e ∧ dist C D = e ∧ dist A D = e ∧ dist A C = e ∧ dist B D = e

def point_in_plane (X : Type) (P1 P2 P3 : Type) : Prop :=
  ∃ (a b c : ℝ), a + b + c = 1 ∧ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ X = a • P1 + b • P2 + c • P3

def segments_form_triangle (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

theorem segments_in_tetrahedron_form_triangle 
  (A B C D M N : Type) [regular_tetrahedron A B C D]
  (hM : point_in_plane M A B C) (hN : point_in_plane N A D C) :
  segments_form_triangle (dist M N) (dist B N) (dist M D) :=
sorry

end segments_in_tetrahedron_form_triangle_l552_552201


namespace Dhoni_spending_difference_l552_552458

-- Definitions
def RentPercent := 20
def LeftOverPercent := 61
def TotalSpendPercent := 100 - LeftOverPercent
def DishwasherPercent := TotalSpendPercent - RentPercent

-- Theorem statement
theorem Dhoni_spending_difference :
  DishwasherPercent = RentPercent - 1 := 
by
  sorry

end Dhoni_spending_difference_l552_552458


namespace smallest_positive_y_l552_552063

theorem smallest_positive_y : ∃ y : ℕ, 58 * y + 14 ≡ 4 [MOD 36] ∧ y > 0 ∧ y = 26 :=
by
  use 26
  split
  · -- 58 * 26 + 14 ≡ 4 (mod 36) proof
    sorry
  split
  · -- 26 > 0 proof
    exact nat.zero_lt_succ 25
  · -- y = 26 proof
    rfl

end smallest_positive_y_l552_552063


namespace Jack_remaining_money_l552_552979

-- Definitions based on conditions
def initial_money : ℕ := 100
def initial_bottles : ℕ := 4
def bottle_cost : ℕ := 2
def extra_bottles : ℕ := 8
def cheese_cost_per_pound : ℕ := 10
def cheese_weight : ℚ := 1 / 2

-- The statement we want to prove
theorem Jack_remaining_money :
  let total_water_cost := (initial_bottles + extra_bottles) * bottle_cost,
      total_cheese_cost := cheese_cost_per_pound * cheese_weight,
      total_cost := total_water_cost + total_cheese_cost
  in initial_money - total_cost = 71 :=
by
  sorry

end Jack_remaining_money_l552_552979


namespace trig_identity_l552_552543

noncomputable def trig_vals (a : ℝ) (h : a < 0) :=
  let α : ℝ := atan (2)
  (cos α, tan α)

theorem trig_identity (a : ℝ) (h : a < 0) :
  let α : ℝ := atan (2) in 
  cos (π - α) * cos (2 * π - α) * sin (-α + 3 * π / 2) 
  / (tan (-α - π) * sin (-π - α)) = 1 / 10 :=
by
  sorry

end trig_identity_l552_552543


namespace total_number_of_coins_l552_552731

-- Definitions and conditions
def num_coins_25c := 17
def num_coins_10c := 17

-- Statement to prove
theorem total_number_of_coins : num_coins_25c + num_coins_10c = 34 := by
  sorry

end total_number_of_coins_l552_552731


namespace product_of_square_roots_of_nine_l552_552926

theorem product_of_square_roots_of_nine (a b : ℝ) (ha : a^2 = 9) (hb : b^2 = 9) : a * b = -9 :=
sorry

end product_of_square_roots_of_nine_l552_552926


namespace expected_value_l552_552505

variable {X : Type} [MeasureSpace X]

def E (f : X → ℝ) : ℝ := ∫ x, f x ∂?m

theorem expected_value (h : E (λ x : X, id x) + E (λ x : X, 2 * id x + 1) = 8) : 
  E (λ x : X, id x) = 7 / 3 := 
by
  sorry -- The proof part is omitted as instructed.

end expected_value_l552_552505


namespace distinguishable_arrangements_l552_552920

-- Define the conditions: number of tiles of each color
def num_brown_tiles := 2
def num_purple_tile := 1
def num_green_tiles := 3
def num_yellow_tiles := 4

-- Total number of tiles
def total_tiles := num_brown_tiles + num_purple_tile + num_green_tiles + num_yellow_tiles

-- Factorials (using Lean's built-in factorial function)
def brown_factorial := Nat.factorial num_brown_tiles
def purple_factorial := Nat.factorial num_purple_tile
def green_factorial := Nat.factorial num_green_tiles
def yellow_factorial := Nat.factorial num_yellow_tiles
def total_factorial := Nat.factorial total_tiles

-- The result of the permutation calculation
def number_of_arrangements := total_factorial / (brown_factorial * purple_factorial * green_factorial * yellow_factorial)

-- The theorem stating the expected correct answer
theorem distinguishable_arrangements : number_of_arrangements = 12600 := 
by
    simp [number_of_arrangements, total_tiles, brown_factorial, purple_factorial, green_factorial, yellow_factorial, total_factorial]
    sorry

end distinguishable_arrangements_l552_552920


namespace distance_between_foci_l552_552043

-- Given problem
def hyperbola_eq (x y : ℝ) : Prop := 9 * x^2 - 18 * x - 16 * y^2 + 32 * y = 144

theorem distance_between_foci :
  ∀ (x y : ℝ),
    hyperbola_eq x y →
    2 * Real.sqrt ((137 / 9) + (137 / 16)) / 72 = 38 * Real.sqrt 7 / 72 :=
by
  intros x y h
  sorry

end distance_between_foci_l552_552043


namespace solution_set_f_g_lt_x_plus_1_l552_552210

theorem solution_set_f_g_lt_x_plus_1 {f g : ℝ → ℝ} (h1 : ∀ x > 0, differentiable_at ℝ f x) (h2 : ∀ x > 0, differentiable_at ℝ g x)
  (h3 : ∀ x > 0, f' x * g x + f x * g' x < 1) (h4 : f 1 = 2) (h5 : g 1 = 1) :
  ∀ x, x ∈ { x | f x * g x < x + 1 } ↔ x ∈ set.Ioi 1 := sorry

end solution_set_f_g_lt_x_plus_1_l552_552210


namespace phone_numbers_count_l552_552404

theorem phone_numbers_count : (2^5 = 32) :=
by sorry

end phone_numbers_count_l552_552404


namespace games_per_team_l552_552721

noncomputable def num_teams : ℕ := 19
noncomputable def total_games_played : ℕ := 1710

theorem games_per_team (n : ℕ) (t : ℕ) (k : ℕ) (H1 : n = num_teams) (H2 : t = total_games_played) (H3: k = total_games_played / (num_teams * (num_teams - 1))) :
  k = 5 :=
by
  rw [H1, H2, H3]
  sorry

end games_per_team_l552_552721


namespace triangle_area_l552_552407

theorem triangle_area (a b c : ℕ) (h1 : a + b + c = 12) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) 
                      (h5 : a + b > c) (h6 : b + c > a) (h7 : c + a > b) :
  ∃ A : ℝ, A = 2 * Real.sqrt 6 ∧
    ∃ (h : 0 ≤ A), A = (Real.sqrt (A * 12 * (12 - a) * (12 - b) * (12 - c))) :=
sorry

end triangle_area_l552_552407


namespace simplify_cubic_root_l552_552269

theorem simplify_cubic_root :
  (∛(20^3 + 30^3 + 40^3 + 60^3) = 10 * ∛315) :=
by
  sorry

end simplify_cubic_root_l552_552269


namespace circular_fountain_area_l552_552394

theorem circular_fountain_area
  (AB : ℝ) (hAB : AB = 20)
  (DC : ℝ) (hDC : DC = 12)
  (D : ℝ) (hD : D = AB / 2) :
  let R := Real.sqrt (D ^ 2 + DC ^ 2) in
  let area := Real.pi * R ^ 2 in
  area = 244 * Real.pi :=
by
  sorry

end circular_fountain_area_l552_552394


namespace initial_birds_in_cage_l552_552728

-- Define a theorem to prove the initial number of birds in the cage
theorem initial_birds_in_cage (B : ℕ) 
  (H1 : 2 / 15 * B = 8) : B = 60 := 
by sorry

end initial_birds_in_cage_l552_552728


namespace total_tables_made_l552_552375

def carpenter_tables (T_this_month : ℕ) (n : ℕ) : ℕ :=
  T_this_month + (T_this_month - n)

theorem total_tables_made :
  ∀ (T_this_month : ℕ) (n : ℕ),
    T_this_month = 10 →
    n = 3 →
    carpenter_tables T_this_month n = 17 :=
by
  intros T_this_month n ht hn
  rw [ht, hn]
  simp [carpenter_tables]
  sorry

end total_tables_made_l552_552375


namespace lambda_magnitude_l552_552139

variables {R : Type*} [real_field R]

def vector (R : Type*) [has_neg R] := R × R

def magnitude (a : vector R) : R := real.sqrt (a.1 * a.1 + a.2 * a.2)

noncomputable def lambda_value (a b : vector R) (λ : R) : Prop :=
λ * a + b = (0, 0)

theorem lambda_magnitude
    (a b : vector R)
    (λ : R)
    (ha : magnitude a = 1)
    (hb : b = (2, 1))
    (h : lambda_value a b λ) :
    |λ| = real.sqrt 5 :=
sorry

end lambda_magnitude_l552_552139


namespace katie_remaining_juice_l552_552606

-- Define the initial condition: Katie initially has 5 gallons of juice
def initial_gallons : ℚ := 5

-- Define the amount of juice given to Mark
def juice_given : ℚ := 18 / 7

-- Define the expected remaining fraction of juice
def expected_remaining_gallons : ℚ := 17 / 7

-- The theorem statement that Katie should have 17/7 gallons of juice left
theorem katie_remaining_juice : initial_gallons - juice_given = expected_remaining_gallons := 
by
  -- proof would go here
  sorry

end katie_remaining_juice_l552_552606


namespace no_painted_faces_l552_552411

theorem no_painted_faces (n : ℕ) (h : n = 4) : 
  let total_small_cubes := n ^ 3,
      inner_n := n - 2,
      inner_cubes := inner_n ^ 3 
  in total_small_cubes = 64 → inner_cubes = 8 :=
sorry

end no_painted_faces_l552_552411


namespace probability_club_then_heart_eq_13_over_204_l552_552733

noncomputable def deck_prob : ℝ :=
  let prob_first_club := (13 : ℝ) / 52
  let prob_second_heart := (13 : ℝ) / 51
  prob_first_club * prob_second_heart

theorem probability_club_then_heart_eq_13_over_204 :
  deck_prob = 13 / 204 :=
by
  sorry

end probability_club_then_heart_eq_13_over_204_l552_552733


namespace monkeys_to_eat_48_bananas_in_48_minutes_l552_552680

-- Definitions from conditions
def monkeys := 8
def bananas := 8
def time_to_eat_bananas := sorry -- Time it takes for monkeys to eat given bananas

-- Assumption from condition
def eating_rate_per_monkey := bananas / time_to_eat_bananas / monkeys

-- Variables for question
def bananas_48 := 48
def time_48 := 48

-- Theorem statement
theorem monkeys_to_eat_48_bananas_in_48_minutes :
  (bananas_48 / time_48) / eating_rate_per_monkey = 48 :=
sorry

end monkeys_to_eat_48_bananas_in_48_minutes_l552_552680


namespace EF_distance_is_l552_552994

noncomputable def isosceles_trapezoid (A B C D E F : Type)
  (AD_parallel_BC : Prop) (angle_AD : Real) (diagonals_length : Real)
  (EA_distance : Real) (ED_distance : Real) (AF_distance : Real) : Prop :=
  angle_AD = π / 4 ∧
  diagonals_length = 15 * sqrt 10 ∧
  EA_distance = 15 * sqrt 5 ∧
  ED_distance = 45 * sqrt 5 ∧
  AD_parallel_BC ∧
  AF_distance = 15 * sqrt 5

theorem EF_distance_is (A B C D E F : Type)
  (AD_parallel_BC : Prop) (angle_AD : Real) (diagonals_length : Real)
  (EA_distance : Real) (ED_distance : Real) (AF_distance : Real)
  (EF_distance : Real)
  (h : isosceles_trapezoid A B C D E F AD_parallel_BC angle_AD diagonals_length EA_distance ED_distance AF_distance) :
  EF_distance = 60 * sqrt 5 :=
sorry

end EF_distance_is_l552_552994


namespace find_z_l552_552914

def M (z : ℂ) : Set ℂ := {1, 2, z * Complex.I}
def N : Set ℂ := {3, 4}

theorem find_z (z : ℂ) (h : M z ∩ N = {4}) : z = -4 * Complex.I := by
  sorry

end find_z_l552_552914


namespace shares_percentage_bounds_l552_552328

theorem shares_percentage_bounds 
  (k m n : ℕ) (x y z : ℝ)
  (h1 : k + m = n)
  (h2 : 4 * k * x = m * y)
  (h3 : k * x + m * y = n * z)
  (h4 : 1.6 ≤ y - x ∧ y - x ≤ 2)
  (h5 : 4.2 ≤ z ∧ z ≤ 6) :
  let f := (k:ℝ) / (k + m + n) * 100 in
  12.5 ≤ f ∧ f ≤ 15 :=
by
  sorry

end shares_percentage_bounds_l552_552328


namespace pigeon_count_unique_l552_552780

noncomputable def pigeon_count : ℕ :=
  sorry

theorem pigeon_count_unique :
  ∃! (n : ℕ), 300 < n ∧ n < 900 ∧ n % 2 = 1 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧
               n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 0 :=
begin
  use 539,
  split,
  { -- prove 539 satisfies all conditions
    split,
    { exact nat.lt_of_le_and_ne (le_of_lt 540) (by norm_num) },
    split,
    { exact nat.lt_of_le_and_ne (le_of_lt 900) (by norm_num) },
    split,
    {exact nat.mod_eq_of_lt (by norm_num)},
    split,
    {exact nat.mod_eq_of_lt (by norm_num)},
    split,
    {exact nat.mod_eq_of_lt (by norm_num)},
    split,
    {exact nat.mod_eq_of_lt (by norm_num)},
    split,
    {exact nat.mod_eq_of_lt (by norm_num)},
    {exact nat.mod_eq_of_lt (by norm_num)}},
  { -- prove uniqueness
    intros n h,
    rcases h with ⟨hn1, hn2, hn3, hn4, hn5, hn6, hn7⟩,
    calc n = 539 : sorry, -- detailed proof steps to show n can only be 539
  }
end

end pigeon_count_unique_l552_552780


namespace pentagon_with_at_most_four_yellow_points_l552_552326

/-- Definition of red and yellow points in the given conditions --/
variables (red_points : Finset ℝ) (yellow_points : Finset ℝ)
variables [Fintype red_points] [Fintype yellow_points]
variables (no_yellow_on_segment : ∀ (p1 p2 : ℝ), p1 ∈ red_points → p2 ∈ red_points → p1 ≠ p2 → ∀ y ∈ yellow_points, ¬ (y = (p1 + p2) / 2))

/-- The proof problem translated into Lean 4 statement --/
theorem pentagon_with_at_most_four_yellow_points
  (h_red : red_points.card = 11)
  (h_yellow : yellow_points.card = 11) :
  ∃ (R : Finset ℝ), R ⊆ red_points ∧ R.card = 5 ∧ (yellow_points.filter (λ y, y ∉ convex_hull (R : Set ℝ))).card ≤ 4 :=
begin
  sorry
end

end pentagon_with_at_most_four_yellow_points_l552_552326


namespace maximum_area_triangle_OAB_l552_552964

open Real

def parametric_line_C1 (t α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ × ℝ :=
  (t * cos α, t * sin α)

def intersection_A (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ × ℝ :=
  (8 * sin α, α)

def intersection_B (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ × ℝ :=
  (8 * cos α, α + π / 2)

def area_triangle_OAB (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ :=
  32 * sin α * cos α

theorem maximum_area_triangle_OAB (α : ℝ) (hα : 0 < α ∧ α < π / 2) : 
  ∃ α_max, 0 < α_max ∧ α_max < π / 2 ∧ area_triangle_OAB α_max (and.intro (by linarith) (by linarith)) = 16 :=
by
  use π / 4
  split
  { linarith [Real.pi_pos] }
  split
  { linarith [Real.pi_pos] }
  simp [area_triangle_OAB, sin_double, mul_comm, mul_assoc, mul_left_comm]
  rw sin_pi_div_two
  norm_num
  exact Real.sin_pos_of_pos_lt_two_pi (by linarith) (by linarith)

end maximum_area_triangle_OAB_l552_552964


namespace angle_acb_after_rotations_is_30_l552_552292

noncomputable def initial_angle : ℝ := 60
noncomputable def rotation_clockwise_540 : ℝ := -540
noncomputable def rotation_counterclockwise_90 : ℝ := 90
noncomputable def final_angle : ℝ := 30

theorem angle_acb_after_rotations_is_30 
  (initial_angle : ℝ)
  (rotation_clockwise_540 : ℝ)
  (rotation_counterclockwise_90 : ℝ) :
  final_angle = 30 :=
sorry

end angle_acb_after_rotations_is_30_l552_552292


namespace normal_distribution_probability_l552_552082

noncomputable def P_interval (ξ : ℝ → ℝ) (σ : ℝ) [h : MeasureTheory.MeasureSpace (ℝ → ℝ)] : Prop :=
let ξ := {x : ℝ | x > 3} in MeasureTheory.Measure μ ξ = 0.023

theorem normal_distribution_probability (σ : ℝ) :
  (∀ ξ, ξ ∼ Normal 0 σ^2) → (MeasureTheory.Measure μ {x : ℝ | x > 3} = 0.023) →
  MeasureTheory.Measure μ {x : ℝ | -3 ≤ x ∧ x ≤ 3} = 0.954 :=
begin
  sorry
end

end normal_distribution_probability_l552_552082


namespace math_proposition_verification_l552_552546

theorem math_proposition_verification :
  ((
    (∀ (f : ℝ → ℝ), (f 2 = 1) → (λ x, f (x - 1)) 3 = 1) ∧
    (∀ (x : ℝ), Real.log (abs x) = Real.log (abs (-x))) ∧ 
    (∀ (f : ℝ → ℝ),
      (∀ x, 1 < x ∧ x < 2 → f x ≤ f (x + 1)) →
      (∀ x, 1 < x ∧ x < 2 → -f x ≥ -f (x + 1))
    ) ∧
    (¬ (∃ x : ℝ, x^2 - 2*x + 3 = 0)) ∧
    (¬ (∃ x : ℝ, x^2 - x + 1 = 0))
  ) ↔ ¬(
    (
      (∀ (f : ℝ → ℝ), (f 2 = 1) → (λ x, f (x - 1)) 3 = 1) ∧
      (∀ (x : ℝ), Real.log (abs x) = Real.log (abs (-x))) ∧
      (∃ x : ℝ, x^2 - 2*x + 3 = 0)
    ) ∨ (
      ∀ (f : ℝ → ℝ),
      (∀ x, 1 < x ∧ x < 2 → f x ≤ f (x + 1)) →
      (∀ x, 1 < x ∧ x < 2 → -f x ≥ -f (x + 1))
    )
  )
  ) :=
begin
  -- sorry means skipping the proof for now
  sorry
end

end math_proposition_verification_l552_552546


namespace inequality_sqrt_sum_ge_2_l552_552607
open Real

theorem inequality_sqrt_sum_ge_2 {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  sqrt (a^3 / (1 + b * c)) + sqrt (b^3 / (1 + a * c)) + sqrt (c^3 / (1 + a * b)) ≥ 2 :=
by
  sorry

end inequality_sqrt_sum_ge_2_l552_552607


namespace range_of_m_l552_552499

theorem range_of_m (m : ℝ) (x : ℝ) (h_eq : m / (x - 2) = 3) (h_pos : x > 0) : m > -6 ∧ m ≠ 0 := 
sorry

end range_of_m_l552_552499


namespace find_a_m_l552_552868

theorem find_a_m :
  ∃ a m : ℤ,
    (a = -2) ∧ (m = -1 ∨ m = 3) ∧ 
    (∀ x : ℝ, (a - 1) * x^2 + a * x + 1 = 0 → 
               (m^2 + m) * x^2 + 3 * m * x - 3 = 0) := sorry

end find_a_m_l552_552868


namespace f_is_even_value_of_cos_2theta_pi_6_l552_552121

noncomputable def f (x : ℝ) : ℝ := Math.cos (x - Real.pi / 4) - Math.sin (x - Real.pi / 4)

/- Part I -/
theorem f_is_even : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

/- Part II -/
theorem value_of_cos_2theta_pi_6 (θ : ℝ) (h1 : 0 < θ) (h2 : θ < Real.pi / 2) (h3 : f (θ + Real.pi / 3) = (Real.sqrt 2) / 3) :
  Math.cos (2 * θ + Real.pi / 6) = (4 * Real.sqrt 2) / 9 :=
by
  sorry

end f_is_even_value_of_cos_2theta_pi_6_l552_552121


namespace sandy_paid_for_pants_l552_552267

-- Define the costs and change as constants
def cost_of_shirt : ℝ := 8.25
def amount_paid_with : ℝ := 20.00
def change_received : ℝ := 2.51

-- Define the amount paid for pants
def amount_paid_for_pants : ℝ := 9.24

-- The theorem stating the problem
theorem sandy_paid_for_pants : 
  amount_paid_with - (cost_of_shirt + change_received) = amount_paid_for_pants := 
by 
  -- proof is required here
  sorry

end sandy_paid_for_pants_l552_552267


namespace infinite_functions_l552_552128

-- Define the function and specify the range condition
def y (x : ℝ) : ℝ := x^2

-- Define the range condition
def range_condition : Set ℝ := {y | 1 ≤ y ∧ y ≤ 4}

-- State the theorem
theorem infinite_functions : ∃ (f : ℝ → ℝ), (∀ x, f x = y x) ∧ (∀ y, y ∈ range_condition) ∧ (Set.Infinite (SetOf f)) :=
by 
  sorry

end infinite_functions_l552_552128


namespace right_triangle_set_l552_552009

def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

theorem right_triangle_set :
  (is_right_triangle 3 4 2 = false) ∧
  (is_right_triangle 5 12 15 = false) ∧
  (is_right_triangle 8 15 17 = true) ∧
  (is_right_triangle (3^2) (4^2) (5^2) = false) :=
by
  sorry

end right_triangle_set_l552_552009


namespace perpendicular_BL_AC_l552_552216

-- Given conditions
variables {A B C M N K L O : Type} [EuclideanGeometry A B C] [Circumcircle ω A B C O] [Circumcircle ω₁ A O C K] [Reflection K MN L]

-- We need to prove (BL) is perpendicular to (AC)
theorem perpendicular_BL_AC : ∀ (A B C M N K L O : Type)
  [EuclideanGeometry A B C]
  [Circumcircle ω A B C O]
  [Circumcircle ω₁ A O C K]
  [Reflection K MN L],
  Perpendicular (Line B L) (Line A C) :=
begin
  sorry
end

end perpendicular_BL_AC_l552_552216


namespace arithmetic_sequence_properties_l552_552876

theorem arithmetic_sequence_properties :
  ∃ (a_n S_n b_n T_n : ℕ → ℝ) (a1 d : ℝ),
  (∀ n, a_n n = a1 + (n - 1) * d) ∧
  (∀ n, S_n n = (n / 2) * (2 * a1 + (n - 1) * d)) ∧
  a1 = d ∧
  S_n 4 = 20 ∧
  (a_n 1 = 2) ∧
  (d = 2) ∧
  (∀ n, a_n n = 2 * n) ∧
  (∀ n, S_n n = n^2 + n) ∧
  (∀ n, b_n n = 2^n + (1 / S_n n)) ∧
  (∀ n, T_n n = ∑ i in range n, b_n (i + 1)) ∧
  (∀ n, T_n n = 2^(n+1) - (1 / (n + 1)) - 1) := 
begin
  sorry
end

end arithmetic_sequence_properties_l552_552876


namespace gcd_g_x_l552_552533

noncomputable def g (x : ℕ) : ℕ :=
  (3 * x + 5) * (7 * x + 2) * (13 * x + 7) * (2 * x + 10)

theorem gcd_g_x (x : ℕ) (h : x % 19845 = 0) : Nat.gcd (g x) x = 700 :=
  sorry

end gcd_g_x_l552_552533


namespace initial_number_of_persons_l552_552281

-- Define the given conditions
def initial_weights (N : ℕ) : ℝ := 65 * N
def new_person_weight : ℝ := 80
def increased_average_weight : ℝ := 2.5
def weight_increase (N : ℕ) : ℝ := increased_average_weight * N

-- Mathematically equivalent proof problem
theorem initial_number_of_persons 
    (N : ℕ)
    (h : weight_increase N = new_person_weight - 65) : N = 6 :=
by
  -- Place proof here when necessary
  sorry

end initial_number_of_persons_l552_552281


namespace hypotenuse_exponentiation_l552_552704

-- Conditions
def leg1 : ℝ := Real.log 64 / Real.log 3
def leg2 : ℝ := Real.log 36 / Real.log 6

-- Using Pythagorean theorem to define the length of the hypotenuse
def h : ℝ := Real.sqrt ((leg1)^2 + (leg2)^2)

-- Proving the desired result
theorem hypotenuse_exponentiation : 3 ^ h = 3 ^ Real.sqrt 10 :=
by
  sorry

end hypotenuse_exponentiation_l552_552704


namespace quadratic_mono_increasing_l552_552787

theorem quadratic_mono_increasing {f : ℝ → ℝ} (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, x > 1 → deriv (λ x, x^2 + 2 * a * x - 1) x > 0) :=
by 
  sorry

end quadratic_mono_increasing_l552_552787


namespace _l552_552912

-- Here we define our conditions

def parabola (x y : ℝ) := y^2 = 8 * x

def focus : ℝ × ℝ := (2, 0)

def point_on_parabola (y : ℝ) : ℝ × ℝ := (4, y)

example (y_P : ℝ) (hP : parabola 4 y_P) :
  dist (point_on_parabola y_P) focus = 6 := by
  -- Since we only need the theorem statement, we finish with sorry
  sorry

end _l552_552912


namespace quadrilateral_CDA_correct_l552_552962

noncomputable def quadrilateral_proof (
  (AB BC CD : ℝ) 
  (angle_ABC angle_BCD: ℝ)
  (hAB: AB = Real.sqrt 2) 
  (hBC: BC = Real.sqrt 3) 
  (hCD: CD = 1)
  (hAngle_ABC: angle_ABC = 75)
  (hAngle_BCD: angle_BCD = 120)
) : Prop :=
  ∠CDA = 75

theorem quadrilateral_CDA_correct (AB BC CD angle_ABC angle_BCD : ℝ)
  (hAB: AB = Real.sqrt 2) 
  (hBC: BC = Real.sqrt 3) 
  (hCD: CD = 1)
  (hAngle_ABC: angle_ABC = 75)
  (hAngle_BCD: angle_BCD = 120)
: quadrilateral_proof AB BC CD angle_ABC angle_BCD hAB hBC hCD hAngle_ABC hAngle_BCD :=
sorry

end quadrilateral_CDA_correct_l552_552962


namespace valid_function_l552_552795

theorem valid_function (x : ℝ) (y : ℝ) : (y = sqrt (x - 1) ∧ (x = 1 ∨ x = 2)) ↔ (y = sqrt 0 ∨ y = sqrt 1) := by
  sorry

end valid_function_l552_552795


namespace problem_l552_552516

variable (f : ℝ → ℝ)
variable (x y : ℝ)

def condition1 : Prop := ∀ x y ∈ set.Icc (-1 : ℝ) 1, f(x + y) = f(x) + f(y)
def condition2 : Prop := ∀ x ∈ set.Icc (0 : ℝ) 1, f(x) > 0

theorem problem (h1 : condition1 f) (h2 : condition2 f) :
  f(0) = 0 ∧ (∀ x ∈ set.Icc (-1 : ℝ) 1, f(-x) = -f(x)) ∧ (∀ x1 x2 ∈ set.Icc (-1 : ℝ) 1, x1 < x2 → f(x1) < f(x2)) :=
by
  split
  case 1 sorry
  case 2 split
    case 1 sorry
    case 2 sorry

end problem_l552_552516


namespace gold_coins_count_l552_552381

theorem gold_coins_count (G : ℕ) 
  (h1 : 50 * G + 125 + 30 = 305) :
  G = 3 := 
by
  sorry

end gold_coins_count_l552_552381


namespace correct_option_l552_552862

variable {ℝ : Type*} [Nontrivial ℝ] [Zero ℝ] [Semiring ℝ]

def comp (f g : ℝ → ℝ) : ℝ → ℝ := λ x, f (g x)
def mult (f g : ℝ → ℝ) : ℝ → ℝ := λ x, f x * g x

theorem correct_option {f g h : ℝ → ℝ} : 
  (comp (mult f g) h) = (mult (comp f h) (comp g h)) :=
sorry

end correct_option_l552_552862


namespace find_expectation_l552_552507

variable (X : Type) [AddCommGroup X] [Module ℝ X] [Expectation X]

theorem find_expectation
  (E : X → ℝ)
  (h : E (λ x, x) + E (λ x, 2 * x + 1) = 8) :
  E (λ x, x) = 7 / 3 := 
by sorry

end find_expectation_l552_552507


namespace number_of_even_digits_in_625_base5_l552_552061

-- Define what constitutes an even digit in base-5
def is_even_digit_base5 (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4

-- Define a function to convert 625 from base-10 to base-5
def base10_to_base5 (n : ℕ) : list ℕ :=
  if h : n = 625 then [1, 0, 0, 0, 0] else []

-- Define a function to count the number of even digits in a base-5 representation
def count_even_digits_base5 (digits : list ℕ) : ℕ :=
  digits.countp is_even_digit_base5

-- The theorem that needs to be proven
theorem number_of_even_digits_in_625_base5 :
  count_even_digits_base5 (base10_to_base5 625) = 1 :=
by
  sorry

end number_of_even_digits_in_625_base5_l552_552061


namespace layla_swordtails_l552_552985

/-- Let Layla has 2 Goldfish and each Goldfish gets 1 teaspoon of food.
    Let she has 8 Guppies and each Guppy gets 0.5 teaspoons of food.
    Let she needs to give a total of 12 teaspoons of food to all her fish.
    Let each Swordtail gets 2 teaspoons of food.
    Prove that Layla has 3 Swordtails. -/
theorem layla_swordtails :
  let goldfish := 2 in
  let food_per_goldfish := 1 in
  let guppies := 8 in
  let food_per_guppy := 0.5 in
  let total_food := 12 in
  let food_per_swordtail := 2 in
  ∃ swordtails : ℕ, (goldfish * food_per_goldfish + guppies * food_per_guppy 
                    + swordtails * food_per_swordtail = total_food) ∧ swordtails = 3 :=
by
  sorry

end layla_swordtails_l552_552985


namespace age_difference_mandy_sarah_l552_552225

def mandy_age : ℕ := 3
def tom_age (mandy_age : ℕ) : ℕ := 5 * mandy_age
def julia_age (tom_age : ℕ) : ℕ := tom_age - 3
def max_age (julia_age : ℕ) : ℕ := 2 * julia_age + 2
def sarah_age (max_age : ℕ) : ℕ := max_age + 4

theorem age_difference_mandy_sarah : 
  let mandy := mandy_age,
      tom := tom_age mandy,
      julia := julia_age tom,
      max := max_age julia,
      sarah := sarah_age max
  in sarah - mandy = 27 :=
by {
  sorry
}

end age_difference_mandy_sarah_l552_552225


namespace prob_red_or_blue_l552_552726

theorem prob_red_or_blue (P : ℕ → ℝ) (R Y B : ℕ) (h1 : P R = 0.45) (h2 : P R + P Y = 0.65) : 
  P R + P B = 0.80 := 
begin
  have h3 : P B = 1 - 0.65,
  { sorry },
  rw h2 at h1,
  rw h3,
  sorry
end

end prob_red_or_blue_l552_552726


namespace minimum_sum_is_3_l552_552630

theorem minimum_sum_is_3 :
  ∃ (A B C D : ℕ), A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ 
  A ≠ C ∧ B ≠ D ∧ 
  A ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  C ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  D ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  (A + B) / (C + D) ∈ ℤ ∧ 
  (A + B) / (C + D) ≠ 0 ∧
  (A + B) = 3 :=
begin
  sorry -- Proof not required
end

end minimum_sum_is_3_l552_552630


namespace marked_price_l552_552761

theorem marked_price (x : ℝ) (purchase_price : ℝ) (selling_price : ℝ) (profit_margin : ℝ) 
  (h_purchase_price : purchase_price = 100)
  (h_profit_margin : profit_margin = 0.2)
  (h_selling_price : selling_price = purchase_price * (1 + profit_margin))
  (h_price_relation : 0.8 * x = selling_price) : 
  x = 150 :=
by sorry

end marked_price_l552_552761


namespace comparison_1_comparison_0_comparison_neg2_comparison_general_l552_552805

theorem comparison_1 (x : ℝ) (h : x = 1) : x^2 + 1 = 2 * x := 
by 
  rw h
  norm_num

theorem comparison_0 (x : ℝ) (h : x = 0) : x^2 + 1 > 2 * x := 
by 
  rw h
  norm_num

theorem comparison_neg2 (x : ℝ) (h : x = -2) : x^2 + 1 > 2 * x := 
by 
  rw h
  norm_num

theorem comparison_general (x : ℝ) : x^2 + 1 ≥ 2 * x := 
begin
  exact (le_of_eq (sub_nonneg.2 (pow_two_nonneg (x - 1)))) 
end

end comparison_1_comparison_0_comparison_neg2_comparison_general_l552_552805


namespace evaluate_expression_l552_552034

noncomputable def g (x : ℝ) : ℝ := x^3 + 3*x + 2*Real.sqrt x

theorem evaluate_expression : 
  3 * g 3 - 2 * g 9 = -1416 + 6 * Real.sqrt 3 :=
by
  sorry

end evaluate_expression_l552_552034


namespace three_coin_toss_l552_552346

theorem three_coin_toss (h : nat) (ht : nat) : 
  ∃ p, p = 7/8 :=
by
  let total_outcomes := 2 ^ 3
  let all_tails_outcomes := 1
  let prob_Tails := all_tails_outcomes / total_outcomes
  let prob_heads := 1 - prob_Tails

  have h : prob_heads = 7 / 8 := by
    sorry

  use prob_heads
  exact h

end three_coin_toss_l552_552346


namespace conjugate_of_z_l552_552078

theorem conjugate_of_z
  (z : ℂ)
  (h : (z + complex.I) / (z - 2 * complex.I) = 2 - complex.I) :
  complex.conj z = -7 / 2 - 3 / 2 * complex.I :=
sorry

end conjugate_of_z_l552_552078


namespace number_of_latte_days_l552_552632

variable (L : ℕ) -- Number of days a week Martha buys a latte

-- Conditions
def daily_latte_cost := 4
def daily_iced_coffee_cost := 2
def iced_coffee_days := 3
def weeks_per_year := 52
def savings := 338
def spending_reduction := 0.25

-- Weekly costs
def total_weekly_spending (L : ℕ) : ℝ :=
  (daily_latte_cost * L + daily_iced_coffee_cost * iced_coffee_days : ℕ)

-- Annual costs
def total_annual_spending (L : ℕ) : ℝ :=
  total_weekly_spending L * weeks_per_year

-- Reduced spending goal
def reduced_spending_goal (L : ℕ) : ℝ :=
  (1 - spending_reduction) * total_annual_spending L

-- Proposition
theorem number_of_latte_days : L = 5 ↔ 
  total_annual_spending L - reduced_spending_goal L = savings := by
    sorry

end number_of_latte_days_l552_552632


namespace probability_divisible_by_5_l552_552260

theorem probability_divisible_by_5 :
  (∑ x in Finset.range 2023, (∑ y in Finset.range 2023, (∑ z in Finset.range 2023,
    if (x + 1)*(y * (z + 1) + y + 1) % 5 = 0 then 1 else 0))) /
    (2023 * 2023 * 2023) = 26676 / 50575 :=
by
  sorry

end probability_divisible_by_5_l552_552260


namespace more_than_1000_triples_l552_552263

theorem more_than_1000_triples : ∃ (s : Finset (ℕ × ℕ × ℕ)), 1000 < s.card ∧ 
  (∀ (triple ∈ s), let (a, b, c) := triple in a^15 + b^15 = c^16) :=
sorry

end more_than_1000_triples_l552_552263


namespace sum_of_17th_roots_of_unity_except_1_l552_552806

theorem sum_of_17th_roots_of_unity_except_1 :
  Complex.exp (2 * Real.pi * Complex.I / 17) +
  Complex.exp (4 * Real.pi * Complex.I / 17) +
  Complex.exp (6 * Real.pi * Complex.I / 17) +
  Complex.exp (8 * Real.pi * Complex.I / 17) +
  Complex.exp (10 * Real.pi * Complex.I / 17) +
  Complex.exp (12 * Real.pi * Complex.I / 17) +
  Complex.exp (14 * Real.pi * Complex.I / 17) +
  Complex.exp (16 * Real.pi * Complex.I / 17) +
  Complex.exp (18 * Real.pi * Complex.I / 17) +
  Complex.exp (20 * Real.pi * Complex.I / 17) +
  Complex.exp (22 * Real.pi * Complex.I / 17) +
  Complex.exp (24 * Real.pi * Complex.I / 17) +
  Complex.exp (26 * Real.pi * Complex.I / 17) +
  Complex.exp (28 * Real.pi * Complex.I / 17) +
  Complex.exp (30 * Real.pi * Complex.I / 17) +
  Complex.exp (32 * Real.pi * Complex.I / 17) = 0 := sorry

end sum_of_17th_roots_of_unity_except_1_l552_552806


namespace scientific_notation_correct_l552_552000

def number := 56990000

theorem scientific_notation_correct : number = 5.699 * 10^7 :=
  by
    sorry

end scientific_notation_correct_l552_552000


namespace central_high_school_teachers_l552_552013

theorem central_high_school_teachers (num_students num_students_per_class num_classes_per_student num_classes_per_teacher : ℕ) 
    (h_students : num_students = 1500)
    (h_classes_per_student : num_classes_per_student = 6)
    (h_classes_per_teacher : num_classes_per_teacher = 3)
    (h_students_per_class : num_students_per_class = 25) :
    let total_classes := num_students * num_classes_per_student,
        num_unique_classes := total_classes / num_students_per_class,
        required_teachers := num_unique_classes / num_classes_per_teacher 
    in
    required_teachers = 120 := by
  -- Sorry is used to skip the proof
  sorry

end central_high_school_teachers_l552_552013


namespace time_for_B_alone_l552_552767

theorem time_for_B_alone (h1 : 4 * (1/15 + 1/x) = 7/15) : x = 20 :=
sorry

end time_for_B_alone_l552_552767


namespace terminal_side_in_second_quadrant_l552_552148

theorem terminal_side_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
    0 < α ∧ α < π :=
sorry

end terminal_side_in_second_quadrant_l552_552148


namespace doctors_assignment_l552_552494

theorem doctors_assignment :
  let doctors := ['A', 'B', 'C', 'D', 'E']
  let positions := ['A', 'B', 'C', 'D']
  let choose assignment := (positions : list Char) :=
    ∃ (f : doctors → positions), function.injective f ∧
    (∀ p ∈ positions, ∃ d ∈ doctors, f d = p) ∧
    f 'A' ≠ f 'B'
  (choose assignments).length = 72 :=
by
  sorry

end doctors_assignment_l552_552494


namespace Mike_saves_3_per_week_l552_552438

-- Define the initial conditions and weekly savings
def Carol_initial : ℕ := 60
def Carol_weekly_saving : ℕ := 9
def Mike_initial : ℕ := 90

-- Define the hypothesis that after 5 weeks they have the same amount of money
def equal_after_5_weeks (Mike_weekly_saving : ℕ) : Prop :=
  Carol_initial + 5 * Carol_weekly_saving = Mike_initial + 5 * Mike_weekly_saving

-- State that given the above conditions, Mike's weekly saving is $3
theorem Mike_saves_3_per_week : (Mike_weekly_saving : ℕ) = 3 :=
by
  have : equal_after_5_weeks 3,
  sorry

end Mike_saves_3_per_week_l552_552438


namespace total_dividend_l552_552389

-- Define the initial conditions in structured format
structure Investment :=
  (total_amount : ℝ)
  (invested_A : ℝ)
  (share_price_A : ℝ)
  (premium_A : ℝ)
  (dividend_rate_A : ℝ)
  (invested_B : ℝ)
  (share_price_B : ℝ)
  (premium_B : ℝ)
  (dividend_rate_B : ℝ)
  (share_price_C : ℝ)
  (premium_C : ℝ)
  (dividend_rate_C : ℝ)

-- Initialize with the given values
def myInvestment : Investment :=
  { total_amount := 50000,
    invested_A := 14400,
    share_price_A := 100,
    premium_A := 0.20,
    dividend_rate_A := 0.05,
    invested_B := 22000,
    share_price_B := 50,
    premium_B := 0.10,
    dividend_rate_B := 0.03,
    share_price_C := 200,
    premium_C := 0.05,
    dividend_rate_C := 0.07 }

-- The goal is to prove the total dividend amount received is Rs. 2096
theorem total_dividend (i : Investment) : 
  let 
    cost_per_share_A := i.share_price_A * (1 + i.premium_A)
    shares_A := i.invested_A / cost_per_share_A
    dividend_A := shares_A * i.share_price_A * i.dividend_rate_A

    cost_per_share_B := i.share_price_B * (1 + i.premium_B)
    shares_B := i.invested_B / cost_per_share_B
    dividend_B := shares_B * i.share_price_B * i.dividend_rate_B

    remaining_amount_C := i.total_amount - (i.invested_A + i.invested_B)
    cost_per_share_C := i.share_price_C * (1 + i.premium_C)
    shares_C := remaining_amount_C / cost_per_share_C
    dividend_C := shares_C * i.share_price_C * i.dividend_rate_C

    total_dividend := dividend_A + dividend_B + dividend_C
  in
    total_dividend = 2096 :=
by 
  sorry

end total_dividend_l552_552389


namespace projection_of_a_onto_b_cosine_of_angle_between_a_minus_b_and_a_plus_b_l552_552919

noncomputable def vec_a : ℝ × ℝ := (3/5, 4/5)
noncomputable def norm_b : ℝ := sqrt(2) / 2
noncomputable def theta : ℝ := π / 4

theorem projection_of_a_onto_b :
  let b : ℝ × ℝ := (cos theta * norm_b, sin theta * norm_b)
  in (vec_a.1 * b.1 + vec_a.2 * b.2) / norm_b = sqrt(2) / 2 := sorry

theorem cosine_of_angle_between_a_minus_b_and_a_plus_b :
  let b : ℝ × ℝ := (cos theta * norm_b, sin theta * norm_b)
      a_minus_b := (vec_a.1 - b.1, vec_a.2 - b.2)
      a_plus_b := (vec_a.1 + b.1, vec_a.2 + b.2)
      dot_product := (a_minus_b.1 * a_plus_b.1 + a_minus_b.2 * a_plus_b.2)
      norm_a_minus_b := sqrt ((a_minus_b.1)^2 + (a_minus_b.2)^2)
      norm_a_plus_b := sqrt ((a_plus_b.1)^2 + (a_plus_b.2)^2)
  in dot_product / (norm_a_minus_b * norm_a_plus_b) = sqrt(5) / 5 := sorry

end projection_of_a_onto_b_cosine_of_angle_between_a_minus_b_and_a_plus_b_l552_552919


namespace number_of_valid_subsets_l552_552739

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fib n + fib (n+1)

-- Define x_n as the number of valid subsets
def x (n : ℕ) : ℕ :=
  match n with
  | 0     => 2
  | 1     => 3
  | n+2   => x n + x (n + 1)

-- Theorem stating the required property
theorem number_of_valid_subsets (n : ℕ) : x n = fib (n + 2) :=
sorry

end number_of_valid_subsets_l552_552739


namespace min_value_of_a_plus_b_l552_552890

-- Define the absolute value of log base 3 function
def f (x : ℝ) : ℝ := abs (Real.log x / Real.log 3)

-- Assumptions
variables (a b : ℝ)
hypothesis cond1 : f (a - 1) = f (2 * b - 1)
hypothesis cond2 : a ≠ 2 * b
hypothesis cond3 : a = 1 + Real.sqrt 2 / 2
hypothesis cond4 : b = a / (2 * a - 2)

theorem min_value_of_a_plus_b : a + b = 3 / 2 + Real.sqrt 2 :=
by
  sorry

end min_value_of_a_plus_b_l552_552890


namespace range_of_g_l552_552456

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctanh x

theorem range_of_g : Set.univ = (Set.Ioc (-1 : ℝ) 1) → Set.range g = Set.univ := by
sorry

end range_of_g_l552_552456


namespace squares_centers_orthogonal_and_equal_distance_l552_552348

variable {α : Type*} [OrderedField α]

structure Point (α : Type*) :=
  (x : α)
  (y : α)

def vector_sub (P Q : Point α) : Point α :=
  ⟨P.x - Q.x, P.y - Q.y⟩

def vector_add (P Q : Point α) : Point α :=
  ⟨P.x + Q.x, P.y + Q.y⟩

def dot_product (P Q : Point α) : α :=
  P.x * Q.x + P.y * Q.y

def norm (P : Point α) : α :=
  (P.x ^ 2 + P.y ^ 2) ^ (1/2)

noncomputable def distance (P Q : Point α) : α :=
  norm (vector_sub P Q)

variables (A B C D O1 O2 O3 O4 : Point α)
variables (hABC : distance A B = distance B C)
variables (hBCD : distance B C = distance C D)
variables (hCDA : distance C D = distance D A)
variables (hDAB : distance D A = distance A B)
  
theorem squares_centers_orthogonal_and_equal_distance :
  let O1 := Point α
  let O2 := Point α
  let O3 := Point α
  let O4 := Point α
  ∀ A B C D O1 O2 O3 O4,
  distance O1 O3 = distance O2 O4 ∧ dot_product (vector_sub O1 O3) (vector_sub O2 O4) = 0 :=
by
  sorry

end squares_centers_orthogonal_and_equal_distance_l552_552348


namespace function_count_remainder_l552_552618

noncomputable def countFunctions := 
  let B := {i | i ∈ Finset.range 1 9} -- Set B as Finset {1, 2, 3, 4, 5, 6, 7, 8}
  (∑ k in Finset.range 1 8, Nat.choose 7 k * k^(7 - k)) * 8

def remainderMod1000 (n : ℕ) : ℕ := n % 1000

theorem function_count_remainder : remainderMod1000 countFunctions = 992 :=
by
  sorry

end function_count_remainder_l552_552618


namespace factory_scrap_rate_l552_552771

theorem factory_scrap_rate :
  let pA := 0.45 in
  let pB := 0.55 in
  let sA := 0.02 in
  let sB := 0.03 in
  pA * sA + pB * sB = 0.0255 :=
by
  sorry

end factory_scrap_rate_l552_552771


namespace dart_probability_l552_552769

noncomputable def probability_lands_within_center_square 
  (hex_area : ℝ) (square_area : ℝ) : ℝ :=
square_area / hex_area

theorem dart_probability {x : ℝ} (h : x > 0) :
  probability_lands_within_center_square 
    (3 * real.sqrt 3 * x^2 / 2) 
    (3 * x^2 / 4) = 1 / (2 * real.sqrt 3) :=
by {
  sorry,
}

end dart_probability_l552_552769


namespace solve_inequality_l552_552273

theorem solve_inequality (a : ℝ) (ha_pos : 0 < a) :
  (if 0 < a ∧ a < 1 then {x : ℝ | 1 < x ∧ x < 1 / a}
   else if a = 1 then ∅
   else {x : ℝ | 1 / a < x ∧ x < 1}) =
  {x : ℝ | ax^2 - (a + 1) * x + 1 < 0} :=
by sorry

end solve_inequality_l552_552273


namespace sum_series_eq_l552_552322

noncomputable def geometricSum (n : ℕ) : ℝ := (finset.range n).sum (λ k, (1 / 2 : ℝ)^(k + 1)) 

theorem sum_series_eq :
  (finset.range 10).sum (λ n, (finset.range (n+1)).sum (λ k, (1 / 2 : ℝ)^(k + 1))) = 9 + (1 / 2^10 : ℝ) :=
by 
  sorry

end sum_series_eq_l552_552322


namespace number_of_valid_N_l552_552473

theorem number_of_valid_N : 
  { N : ℕ // 2017 ≡ 17 [MOD N] ∧ N > 17 }.card = 13 :=
by sorry

end number_of_valid_N_l552_552473


namespace petya_sum_expression_l552_552250

theorem petya_sum_expression : 
  (let expressions := finset.image (λ (s : list bool), 
    list.foldl (λ acc ⟨b, n⟩, if b then acc + n else acc - n) 1 (s.zip [2, 3, 4, 5, 6])) 
    (finset.univ : finset (vector bool 5))) in
    expressions.sum) = 32 := 
sorry

end petya_sum_expression_l552_552250


namespace average_speed_with_stoppages_l552_552349

theorem average_speed_with_stoppages (distance : ℝ) (speed_without_stoppages : ℝ := 300) (stop_time_per_hour : ℝ := 20/60) :
  let speed_with_stoppages := (1 - stop_time_per_hour) * speed_without_stoppages
  in speed_with_stoppages = 200 :=
by {
  let speed_with_stoppages := (1 - stop_time_per_hour) * speed_without_stoppages,
  -- The above line is not necessary, but shows the computation of speed_with_stoppages
  sorry
}

end average_speed_with_stoppages_l552_552349


namespace chromatic_number_of_vertex_removal_l552_552153

theorem chromatic_number_of_vertex_removal (G : Type) [graph G] (C : ℕ) (v : V) (χG : chromatic_number G C) :
  ∃ (G' : Type) [graph G'], chromatic_number G' (C - 1) :=
by
  sorry

end chromatic_number_of_vertex_removal_l552_552153


namespace range_of_a_l552_552258

theorem range_of_a (a x y : ℝ) (h : x + a * y - 3 > 0) (hp : (x, y) = (a, a + 1)) : 
  a ∈ Set.Ioo (-∞) (-3) ∪ Set.Ioo (1) ∞ := 
by 
  sorry

end range_of_a_l552_552258


namespace minimum_edges_l552_552621

noncomputable theory

def V : Set (ℝ × ℝ × ℝ) := {x | ∃ (n : ℕ) (h : n < 2019), x = some_3d_point n h}
def E (V : Set (ℝ × ℝ × ℝ)) := {e : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ) | ∃ (x y : ℝ × ℝ × ℝ), x ∈ V ∧ y ∈ V ∧ e = (x, y)}

variables (V : Set (ℝ × ℝ × ℝ)) (E : Set ((ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)))

theorem minimum_edges (hV : ∀ (v1 v2 v3 v4 : ℝ × ℝ × ℝ), (v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4) → 
              ¬(v1 ∈ V ∧ v2 ∈ V ∧ v3 ∈ V ∧ v4 ∈ V ∧ coplanar {v1, v2, v3, v4}))
              (hE : E = {e ∈ (V.prod V) | e.1 ≠ e.2}) :
  ∃ n : ℕ, ∀ E' ⊆ E, E'.card ≥ 2795 → ∃ H : Set ((ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)), 
    H.card = 908 ∧ (∀ (e1 e2 : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)), e1 ∈ H → e2 ∈ H → e1 ≠ e2 → ¬(e1.1 = e2.1 ∨ e1.2 = e2.2)) :=
  sorry

end minimum_edges_l552_552621


namespace hyperbola_eccentricity_l552_552448

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : let A := (a, 0)
           B := (a^2 / (a + b), a * b / (a + b))
           C := (a^2 / (a - b), -a * b / (a - b))
           AB := (B.1 - A.1, B.2 - A.2)
           BC := (C.1 - B.1, C.2 - B.2)
       in AB = (1 / 2) • BC)
  : let c := Real.sqrt (5 * a^2)
    in c / a = Real.sqrt 5 := sorry

end hyperbola_eccentricity_l552_552448


namespace fourth_term_geometric_progression_l552_552145

theorem fourth_term_geometric_progression :
  ∀ (x : ℝ), 
  (2*x ≠ 0) ∧ (4*x + 4 ≠ 0) ∧ (6*x + 6 ≠ 0) →
  (4*x + 4)/(2*x) = (6*x + 6)/(4*x + 4) →
  (2*x = -8 ∨ 2*x = -12 ∨ 2*x = -18) →
  let r := (4*x+4)/(2*x) in
  r * (6*x + 6) = -27 :=
by
  sorry

end fourth_term_geometric_progression_l552_552145


namespace theorem1_theorem2_l552_552514

-- Define a structure for a Circle with specific properties
structure Circle (P : Type) :=
(diameter : P → P → Prop)
(tangent : P → (P × P) → Prop) -- tangent to the circle at a point

-- Define points and lines in the circle
variables {P : Type} (circle : Circle P) (A B M A' B' X : P)
(AM BM : P × P)
(tangentA tangentB tangentM : P × P)

-- Define conditions according to problem
def conditions :=
circle.diameter A B ∧
circle.tangent A tangentA ∧
circle.tangent B tangentB ∧
(M ∈ circle) ∧ -- M on the circle
(tangentA ⊂ circle) ∧
(tangentB ⊂ circle) ∧
(AM.coe ∧ BM.coe) ∧
(AM ∩ tangentB = A') ∧
(BM ∩ tangentA = B')

-- Theorem 1: Prove that \(AA' \times BB' = AB^2\)
theorem theorem1 {circle : Circle P} {A B M A' B' : P} (h : conditions circle A B M A' B') :
  (dist A A') * (dist B B') = (dist A B)^2 :=
begin
  sorry -- Proof is omitted
end

-- Theorem 2: Prove that the tangent at M to the circle and the line A'B' intersect on (AB)
theorem theorem2 {circle : Circle P} {A B M A' B' X : P} (h : conditions circle A B M A' B') :
  (tangent M ∩ line A' B' = X) → (X ∈ line A B) :=
begin
  sorry -- Proof is omitted
end

end theorem1_theorem2_l552_552514


namespace sector_area_l552_552537

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (radius : ℝ) (area : ℝ) 
  (h1 : arc_length = 6) 
  (h2 : central_angle = 2) 
  (h3 : radius = arc_length / central_angle): 
  area = (1 / 2) * arc_length * radius := 
  sorry

end sector_area_l552_552537


namespace length_of_FQ_l552_552683

theorem length_of_FQ
  (D E F : Type)
  [metric_space D] [metric_space E] [metric_space F]
  (DE DF EF FQ : ℝ)
  (sqrt_85 : ℝ)
  (h1 : DE = 7)
  (h2 : DF = sqrt_85)
  (h3 : sqrt_85 = Real.sqrt 85)
  (triangle_DE_F : D ≠ E ∧ E ≠ F ∧ D ≠ F) -- ensuring the points are distinct
  (right_triangle_DEF : ∃ θ : ℝ, θ = 90 ∧ is_right_triangle D E F θ)
  (circle_center_on_DE : ∃ C, C ∈ line D E ∧ dist C E = 1 ∧ 
        is_tangent C F DE DF EF)
  : FQ = 6 := by
  sorry

end length_of_FQ_l552_552683


namespace find_m_l552_552958

variables {A B C O : Point}
variables (α β γ m R : ℝ)
variables [AcuteTriangle A B C] -- Assume this definition exists in Mathlib for acute triangles
variables [Circumcenter O A B C] -- Assume this definition exists in Mathlib for circumcenters

-- Given conditions
axiom angle_A : ∠ A = 45
axiom vector_condition : 
  (cos β / sin γ) * (vector AB) + (cos γ / sin β) * (vector AC) = 2 * m * (vector AO)
-- Placeholder for vectors
axiom vector_AB : vector A B
axiom vector_AC : vector A C
axiom vector_AO : vector A O

theorem find_m :
  m = sqrt 2 / 2 :=
sorry

end find_m_l552_552958


namespace determine_k_l552_552673

theorem determine_k (k : ℝ) :
  let sq_area := 6
  let A := 2
  let area_below := A
  let area_above := 2 * A
  let total_area := area_below + area_above
  let line_eq := (x : ℝ) -> (k / (k - 2)) * (x - 2)
  let intersection_x := 2
  let triangle_area := (1 / 2) * (1 * (k - 2))
  let actual_area_below := 1 + triangle_area
  in total_area = 6 ∧ actual_area_below = 2 → k = 4 :=
by
  sorry

end determine_k_l552_552673


namespace b_is_geometric_T_sum_l552_552885

noncomputable def a (n : ℕ) : ℝ := 1/2 + (n-1) * (1/2)
noncomputable def S (n : ℕ) : ℝ := n * (1/2) + (n * (n-1) / 2) * (1/2)
noncomputable def b (n : ℕ) : ℝ := 4 ^ (a n)
noncomputable def c (n : ℕ) : ℝ := a n + b n
noncomputable def T (n : ℕ) : ℝ := (n * (n+1) / 4) + 2^(n+1) - 2

theorem b_is_geometric : ∀ n : ℕ, (n > 0) → b (n+1) / b n = 2 := by
  sorry

theorem T_sum : ∀ n : ℕ, T n = (n * (n + 1) / 4) + 2^(n + 1) - 2 := by
  sorry

end b_is_geometric_T_sum_l552_552885


namespace num_nat_numbers_with_remainder_17_l552_552477

theorem num_nat_numbers_with_remainder_17 (N : ℕ) :
  (2017 % N = 17 ∧ N > 17) → 
  ({N | 2017 % N = 17 ∧ N > 17}.toFinset.card = 13) := 
by
  sorry

end num_nat_numbers_with_remainder_17_l552_552477


namespace total_eBooks_readers_l552_552222

def eBooksReaderTotal (x y z : ℕ) : ℕ :=
  let john_initial := y - 15
  let mary_initial := 2 * john_initial
  let john_after_loss := john_initial - 3
  let mary_after_giveaway := mary_initial - 7
  let john_after_exchange := john_after_loss - 5 + 6
  let anna_after_exchange := y - 6 + 5
  let mary_after_sale := mary_after_giveaway - 10
  let anna_after_buy := anna_after_exchange + 10
  john_after_exchange + anna_after_buy + mary_after_sale

theorem total_eBooks_readers (y : ℕ) (h : y = 50) : eBooksReaderTotal (y - 15) y (2 * (y - 15)) = 145 := by
  simp [eBooksReaderTotal, h]
  sorry

end total_eBooks_readers_l552_552222


namespace value_of_expression_l552_552012

theorem value_of_expression 
  (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 + 49 * x7 = 1) 
  (h2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 + 64 * x7 = 12) 
  (h3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 + 81 * x7 = 123) 
  : 16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 + 100 * x7 = 334 :=
by
  sorry

end value_of_expression_l552_552012


namespace lattice_point_inequality_l552_552188

variable (S p : ℝ) (n : ℕ)
variable (convex_figure : Type)
variable (lattice_points_in_figure : ℕ)

def has_area (sh: convex_figure) : ℝ := S
def has_semiperimeter (sh: convex_figure) : ℝ := p
def lattice_points (sh: convex_figure) : ℕ := n
def lattice_points_count : ℕ := lattice_points_in_figure

theorem lattice_point_inequality
  (convex_shaped : convex_figure)
  (h1 : has_area convex_shaped = S)
  (h2 : has_semiperimeter convex_shaped = p)
  (h3 : lattice_points_count = n) :
  n > S - p :=
  sorry

end lattice_point_inequality_l552_552188


namespace find_distance_from_pole_l552_552961

noncomputable def common_point_polar_distance (θ : ℝ) (ρ : ℝ) : Prop :=
  (ρ = cos θ + 1) ∧ (ρ * cos θ = 1)

theorem find_distance_from_pole (θ ρ : ℝ) (h : common_point_polar_distance θ ρ) : ρ = (sqrt 5 + 1) / 2 :=
sorry

end find_distance_from_pole_l552_552961


namespace sqrt_sqrt_16_l552_552280

theorem sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := sorry

end sqrt_sqrt_16_l552_552280


namespace stratified_sample_l552_552165

theorem stratified_sample 
  (total_households : ℕ) 
  (high_income_households : ℕ) 
  (middle_income_households : ℕ) 
  (low_income_households : ℕ) 
  (sample_size : ℕ)
  (H1 : total_households = 600) 
  (H2 : high_income_households = 150)
  (H3 : middle_income_households = 360)
  (H4 : low_income_households = 90)
  (H5 : sample_size = 100) : 
  (middle_income_households * sample_size / total_households = 60) := 
by 
  sorry

end stratified_sample_l552_552165


namespace find_a_increasing_intervals_l552_552547

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := sin x + a * cos x

theorem find_a : ∃ a : ℝ, f (3 * π / 4) a = 0 ∧ a = 1 :=
by
  use 1
  split
  sorry -- proof for f(3π/4) a = 0
  rfl  -- a = 1 is by definition

noncomputable def g (x : ℝ) : ℝ := (f x 1) ^ 2 - 2 * (sin x) ^ 2

theorem increasing_intervals : ∀ k : ℤ, 
  monotone_on g (Icc (↑k * π - 3 * π / 8) (↑k * π + π / 8)) :=
by
  intro k
  sorry -- proof of monotonicity in the interval [kπ - 3π/8, kπ + π/8]

end find_a_increasing_intervals_l552_552547


namespace construct_triangle_given_midpoints_and_orthocenter_l552_552554

theorem construct_triangle_given_midpoints_and_orthocenter
  (A B C D D' M: Point)
  (h_midpoint_D : midpoint D B C)
  (h_midpoint_D' : midpoint D' (altitude B M) (altitude C M))
  (h_orthocenter : orthocenter M A B C) :
  ∃ A' B' C', MID D' (altitude B' M) (altitude C' M) ∧
  orthocenter M A' B' C' := sorry

end construct_triangle_given_midpoints_and_orthocenter_l552_552554


namespace percent_no_conditions_l552_552788

def survey_teachers : ℕ := 150
def high_blood_pressure : ℕ := 90
def heart_trouble : ℕ := 60
def diabetes : ℕ := 50
def high_blood_pressure_and_heart_trouble : ℕ := 30
def high_blood_pressure_and_diabetes : ℕ := 20
def heart_trouble_and_diabetes : ℕ := 10
def all_three_conditions : ℕ := 5

theorem percent_no_conditions :
  (survey_teachers - (high_blood_pressure + heart_trouble + diabetes
  - high_blood_pressure_and_heart_trouble
  - high_blood_pressure_and_diabetes
  - heart_trouble_and_diabetes
  + all_three_conditions)) / survey_teachers * 100 = 3.33 :=
sorry

end percent_no_conditions_l552_552788


namespace right_triangle_hypotenuse_length_l552_552597

theorem right_triangle_hypotenuse_length
  (A B C D E : Point)
  (x : ℝ)
  (h1 : right_triangle A B C)
  (h2 : trisection_points A B D E)
  (h3 : dist C D = 7)
  (h4 : dist C E = 6) :
  dist A B = 3 * Real.sqrt 17 :=
by
  sorry

end right_triangle_hypotenuse_length_l552_552597


namespace solution_a_solution_b_solution_c_l552_552071

theorem solution_a (x : ℝ) :
  sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = sqrt 2 ↔ 1 / 2 ≤ x ∧ x ≤ 1 := 
sorry

theorem solution_b (x : ℝ) :
  sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 1 ↔ false :=
sorry

theorem solution_c (x : ℝ) :
  sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 2 ↔ x = 3 / 2 :=
sorry

end solution_a_solution_b_solution_c_l552_552071


namespace tangent_line_k_minus_b_l552_552935

theorem tangent_line_k_minus_b {k b : ℝ} :
  (∀ x : ℝ, differentiable_at ℝ (λ x, real.log x + 2) x →
           deriv (λ x, real.log x + 2) x = k →
           f x = k * x + b) ∧
  (∀ x : ℝ, differentiable_at ℝ (λ x, real.log (x + 1)) x →
           deriv (λ x, real.log (x + 1)) x = k →
           g x = k * x + b) →
  k - b = 1 + real.log 2 :=
begin
  sorry
end

end tangent_line_k_minus_b_l552_552935


namespace smiths_laundry_loads_l552_552592

theorem smiths_laundry_loads
  (kylie_towels : ℕ)
  (daughters_towels : ℕ)
  (husband_towels : ℕ)
  (washing_machine_capacity : ℕ)
  (total_towels : ℕ)
  (num_loads : ℕ) 
  (h1 : kylie_towels = 3)
  (h2 : daughters_towels = 6)
  (h3 : husband_towels = 3)
  (h4 : washing_machine_capacity = 4)
  (h5 : total_towels = kylie_towels + daughters_towels + husband_towels)
  (h6 : num_loads = total_towels / washing_machine_capacity) :
  num_loads = 3 :=
by 
  simp [h1, h2, h3, h4, h5, h6]
  norm_num

end smiths_laundry_loads_l552_552592


namespace no_x_for_rational_sin_cos_l552_552045

-- Define rational predicate
def is_rational (r : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ r = a / b

-- Define the statement of the problem
theorem no_x_for_rational_sin_cos :
  ∀ x : ℝ, ¬ (is_rational (Real.sin x + Real.sqrt 2) ∧ is_rational (Real.cos x - Real.sqrt 2)) :=
by
  -- Placeholder for proof
  sorry

end no_x_for_rational_sin_cos_l552_552045


namespace calculator_sum_l552_552040

theorem calculator_sum :
  let A := 2
  let B := 0
  let C := -1
  let D := 3
  let n := 47
  let A' := if n % 2 = 1 then -A else A
  let B' := B -- B remains 0 after any number of sqrt operations
  let C' := if n % 2 = 1 then -C else C
  let D' := D ^ (3 ^ n)
  A' + B' + C' + D' = 3 ^ (3 ^ 47) - 3
:= by
  sorry

end calculator_sum_l552_552040


namespace sum_of_i_powers_l552_552031

theorem sum_of_i_powers : ∀ (i : ℂ), i^2 = -1 → (∑ k in Finset.range 1003, i^k) = i :=
by
  intro i h_i_sq_eq_neg1
  sorry

end sum_of_i_powers_l552_552031


namespace magnitude_b_cosine_theta_l552_552073

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, 3)
def b : ℝ × ℝ := (-1, 2)

-- Euclidean norm function
def euclidean_norm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- Dot product function
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Magnitude of vector b
theorem magnitude_b : euclidean_norm b = Real.sqrt 5 :=
sorry

-- Cosine of the angle between vectors a and b
theorem cosine_theta : 
  let cosθ := (dot_product a b) / (euclidean_norm a * euclidean_norm b)
  cosθ = 2 * Real.sqrt 5 / 25 :=
sorry

end magnitude_b_cosine_theta_l552_552073


namespace parabola_vertex_coordinates_l552_552323

noncomputable def parabola_vertex (x : ℝ) : ℝ := x^2 - 6 * x + 5

theorem parabola_vertex_coordinates : ∃ h k : ℝ, parabola_vertex = λ x, (x - h)^2 + k ∧ h = 3 ∧ k = -4 :=
by
  use 3, -4
  -- Rewrite and simplify
  calc
    parabola_vertex = λ x, x^2 - 6 * x + 5 : by funext; simp [parabola_vertex]
    ... = λ x, (x - 3)^2 - 4 : by funext; ring
  -- The solution has been simplified to the form where the vertex is determined
  sorry

end parabola_vertex_coordinates_l552_552323


namespace limit_a_n_n_l552_552990

noncomputable def a_sequence (d : ℝ) (m j : ℕ) : ℝ :=
  if j = 0 then d / 2^m
  else (a_sequence d m (j - 1))^2 + 2 * (a_sequence d m (j - 1))

theorem limit_a_n_n (d : ℝ) : 
  Filter.Tendsto (λ n, a_sequence d n n) Filter.atTop (𝓝 (Real.exp d - 1)) := 
  by
    sorry

end limit_a_n_n_l552_552990


namespace chess_game_probability_l552_552372

theorem chess_game_probability (p_A_wins p_draw : ℝ) (h1 : p_A_wins = 0.3) (h2 : p_draw = 0.2) :
  p_A_wins + p_draw = 0.5 :=
by
  rw [h1, h2]
  norm_num

end chess_game_probability_l552_552372


namespace find_digits_l552_552485

theorem find_digits (x y z : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9) (h3 : 0 ≤ z ∧ z ≤ 9) :
  (10 * x + 5) * (3 * 100 + y * 10 + z) = 7850 ↔ (x = 2 ∧ y = 1 ∧ z = 4) :=
by
  sorry

end find_digits_l552_552485


namespace maximum_overtakes_l552_552956

-- Definitions based on problem conditions
structure Team where
  members : List ℕ
  speed_const : ℕ → ℝ -- Speed of each member is constant but different
  run_segment : ℕ → ℕ -- Each member runs exactly one segment
  
def relay_race_condition (team1 team2 : Team) : Prop :=
  team1.members.length = 20 ∧
  team2.members.length = 20 ∧
  ∀ i, (team1.speed_const i ≠ team2.speed_const i)

def transitions (team : Team) : ℕ :=
  team.members.length - 1

-- The theorem to be proved
theorem maximum_overtakes (team1 team2 : Team) (hcond : relay_race_condition team1 team2) : 
  ∃ n, n = 38 :=
by
  sorry

end maximum_overtakes_l552_552956


namespace tangent_line_at_1_l552_552340

-- Assume the curve and the point of tangency
noncomputable def curve (x : ℝ) : ℝ := x^3 - 2*x^2 + 4*x + 5

-- Define the point of tangency
def point_of_tangency : ℝ := 1

-- Define the expected tangent line equation in standard form Ax + By + C = 0
def tangent_line (x y : ℝ) : Prop := 3 * x - y + 5 = 0

theorem tangent_line_at_1 :
  tangent_line point_of_tangency (curve point_of_tangency) := 
sorry

end tangent_line_at_1_l552_552340


namespace complex_modulus_l552_552575

def z (i : ℂ) : ℂ := 1 / (1 - i)

theorem complex_modulus (i z : ℂ) (h : (1 - i) * z = 1) : |4 * z - 3| = Real.sqrt 5 :=
by
  -- Given condition (h): (1 - i) * z = 1
  -- Statement to prove: |4 * z - 3| = Real.sqrt 5
  sorry

end complex_modulus_l552_552575


namespace gcd_91_49_l552_552336

theorem gcd_91_49 : Nat.gcd 91 49 = 7 :=
by
  -- Using the Euclidean algorithm
  -- 91 = 49 * 1 + 42
  -- 49 = 42 * 1 + 7
  -- 42 = 7 * 6 + 0
  sorry

end gcd_91_49_l552_552336


namespace product_fractions_l552_552434

open BigOperators

theorem product_fractions : ∏ n in Finset.range 28 \ Finset.singleton 0 ∪ Finset.singleton 1, (n + 2) / (n + 1) = 15 := by
  sorry

end product_fractions_l552_552434


namespace surface_area_of_inscribed_sphere_l552_552027

def surface_area_inscribed_sphere (a : ℝ) : ℝ :=
  let r := a / (2 * Real.sqrt 6) in
  4 * Real.pi * r^2

theorem surface_area_of_inscribed_sphere (a : ℝ) (h : a > 0) :
  surface_area_inscribed_sphere a = (Real.pi * a^2) / 6 :=
by sorry

end surface_area_of_inscribed_sphere_l552_552027


namespace correct_calculation_l552_552747

theorem correct_calculation : 
  ¬ (3 - 5 = 2) ∧
  ¬ (3 * a + 2 * b = 5 * a * b) ∧
  ¬ (3 * x^2 * y - 2 * x * y^2 = x * y) ∧
  (4 - | -3 | = 1) :=
by sorry

end correct_calculation_l552_552747


namespace find_abc_squares_l552_552984

variable (a b c x : ℕ)

theorem find_abc_squares (h1 : 1 ≤ a) (h2 : a + b + c = 9) (h3 : 99 * (c - a) = 65 * x) (h4 : 495 = 65 * x) : a^2 + b^2 + c^2 = 53 :=
  sorry

end find_abc_squares_l552_552984


namespace local_max_2_l552_552220

noncomputable def f (x m n : ℝ) := 2 * Real.log x - (1 / 2) * m * x^2 - n * x

theorem local_max_2 (m n : ℝ) (h : n = 1 - 2 * m) :
  ∃ m : ℝ, -1/2 < m ∧ (∀ x : ℝ, x > 0 → (∃ U : Set ℝ, IsOpen U ∧ (2 ∈ U) ∧ (∀ y ∈ U, f y m n ≤ f 2 m n))) :=
sorry

end local_max_2_l552_552220


namespace textbooks_probability_l552_552227

open Finset

theorem textbooks_probability (boxes : Fin 3 → ℕ) (math_books : ℕ) (total_books : ℕ) 
  (h_total_books : total_books = 15) (h_math_books : math_books = 4)
  (h_boxes : boxes 0 = 4 ∧ boxes 1 = 5 ∧ boxes 2 = 6) :
  let total_ways := choose 15 4 * choose 11 5 * choose 6 6 in
  let favorable_ways := choose 11 2 * choose 9 4 * choose 5 5 in
  let probability := favorable_ways / total_ways in
  let m := 1 in let n := 91 in
  m + n = 92 :=
by
  sorry

end textbooks_probability_l552_552227


namespace consecutive_even_product_l552_552313

-- Define that there exist three consecutive even numbers such that the product equals 87526608.
theorem consecutive_even_product (a : ℤ) : 
  (a - 2) * a * (a + 2) = 87526608 → ∃ b : ℤ, b = a - 2 ∧ b % 2 = 0 ∧ ∃ c : ℤ, c = a ∧ c % 2 = 0 ∧ ∃ d : ℤ, d = a + 2 ∧ d % 2 = 0 :=
sorry

end consecutive_even_product_l552_552313


namespace simplify_and_evaluate_expr_l552_552670

theorem simplify_and_evaluate_expr (a b : ℤ) (h₁ : a = -1) (h₂ : b = 2) :
  (2 * a + b - 2 * (3 * a - 2 * b)) = 14 := by
  rw [h₁, h₂]
  sorry

end simplify_and_evaluate_expr_l552_552670


namespace petya_sum_of_all_combinations_l552_552244

-- Define the expression with the possible placements of signs.
def petyaExpression : List (ℤ → ℤ → ℤ) :=
  [int.add, int.sub] -- Combination of possible operations at each position

-- Calculate the total number of ways to insert "+" and "-" in the expression
def number_of_combinations : ℕ := 2^5

-- Define the problem statement in Lean 4
theorem petya_sum_of_all_combinations : 
  (∑ idx in Finset.range number_of_combinations, 1) = 32 := by
  sorry

end petya_sum_of_all_combinations_l552_552244


namespace part_a_part_b_part_c_l552_552187

def is_frameable (n : ℕ) : Prop :=
  n = 3 ∨ n = 4 ∨ n = 6

theorem part_a : is_frameable 3 ∧ is_frameable 4 ∧ is_frameable 6 :=
  sorry

theorem part_b (n : ℕ) (h : n ≥ 7) : ¬ is_frameable n :=
  sorry

theorem part_c : ¬ is_frameable 5 :=
  sorry

end part_a_part_b_part_c_l552_552187


namespace max_value_of_M_l552_552832

theorem max_value_of_M (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (2 * x + y) + y / (2 * y + z) + z / (2 * z + x)) ≤ 1 :=
sorry -- Proof placeholder

end max_value_of_M_l552_552832


namespace tony_walking_speed_l552_552639

-- Define the conditions as hypotheses
def walking_speed_on_weekend (W : ℝ) : Prop := 
  let store_distance := 4 
  let run_speed := 10
  let day1_time := store_distance / W
  let day2_time := store_distance / run_speed
  let day3_time := store_distance / run_speed
  let avg_time := (day1_time + day2_time + day3_time) / 3
  avg_time = 56 / 60

-- State the theorem
theorem tony_walking_speed : ∃ W : ℝ, walking_speed_on_weekend W ∧ W = 2 := 
sorry

end tony_walking_speed_l552_552639


namespace least_positive_period_of_f_maximum_value_of_f_monotonically_increasing_intervals_of_f_l552_552116

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3) + 2

theorem least_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem maximum_value_of_f :
  ∃ x, f x = 3 :=
sorry

theorem monotonically_increasing_intervals_of_f :
  ∀ k : ℤ, ∃ a b : ℝ, a = -Real.pi / 12 + k * Real.pi ∧ b = 5 * Real.pi / 12 + k * Real.pi ∧ ∀ x, a < x ∧ x < b → ∀ x', a ≤ x' ∧ x' ≤ x → f x' < f x :=
sorry

end least_positive_period_of_f_maximum_value_of_f_monotonically_increasing_intervals_of_f_l552_552116


namespace petya_sum_l552_552237

theorem petya_sum : 
  let f (signs : fin 5 → bool) : ℤ :=
    1 + (if signs 0 then 2 else -2) + (if signs 1 then 3 else -3) + (if signs 2 then 4 else -4) + (if signs 3 then 5 else -5) + (if signs 4 then 6 else -6),
  sum (f '' (finset.univ : finset (fin 5 → bool))) = 32 :=
by
  sorry

end petya_sum_l552_552237


namespace pool_width_40_l552_552689

theorem pool_width_40
  (hose_rate : ℕ)
  (pool_length : ℕ)
  (pool_depth : ℕ)
  (pool_capacity_percent : ℚ)
  (drain_time : ℕ)
  (water_drained : ℕ)
  (total_capacity : ℚ)
  (pool_width : ℚ) :
  hose_rate = 60 ∧
  pool_length = 150 ∧
  pool_depth = 10 ∧
  pool_capacity_percent = 0.8 ∧
  drain_time = 800 ∧
  water_drained = hose_rate * drain_time ∧
  total_capacity = water_drained / pool_capacity_percent ∧
  total_capacity = pool_length * pool_width * pool_depth →
  pool_width = 40 :=
by
  sorry

end pool_width_40_l552_552689


namespace trajectory_of_center_is_parabola_find_equation_of_circle_n_const_t_and_dot_product_l552_552578

-- Assuming necessary data types for geometric entities and vector operations are present in Mathlib.

-- (I) Prove the equation of the curve E
theorem trajectory_of_center_is_parabola :
  ∀ C : Type, 
  (∃ M : Type, M = (0,1)) ∧
  (∃ l : Type, l = (y = -1)) → 
  (C passes_through_point M ∧ C is_tangent_to_line l) →
  (∃ E : Type, E = (x^2 = 4y)) := 
sorry

-- (II) Prove the equation of circle N given t = 6 and the slope of line AB is 1/2
theorem find_equation_of_circle_n :
  ∀ (A B : Type), 
  t = 6 ∧ slope_of_line_AB = 1/2 →
  (line_passes_through_ponts A B ∧ point A ∧ B on parabola x^2 = 4y) → 
  (circle_n passes_through A B ∧ shares_tangent_with_parabola_at A) →
  (∃ N : Type, N = (left(x+3/2)^2 + (left(y-23/2)^2 = 125/4)) :=
sorry

-- (III) Prove that both t and dot product of QA and QB are constants
theorem const_t_and_dot_product :
  ∀ (A B Q : Type), 
  (A B on parabola x^2 = 4y) ∧ 
  tangents_at_A_and_B_intersect_at Q ∧ 
  (Q line := (y = -1)) → 
  (exist t : Type, t = -1) ∧ 
  (vec(QA) ⋅ vec(QB) = 0) := 
sorry

end trajectory_of_center_is_parabola_find_equation_of_circle_n_const_t_and_dot_product_l552_552578


namespace volume_of_wedge_of_sphere_l552_552784

theorem volume_of_wedge_of_sphere (C : ℝ) (hC : C = 12 * real.pi) :
  let r := C / (2 * real.pi),
      V := (4/3) * real.pi * r^3,
      wedge_volume := V / 4
  in wedge_volume = 72 * real.pi :=
by
  let r := C / (2 * real.pi)
  have hr : r = 6, from by linarith [hC]
  let V := (4/3) * real.pi * r^3
  have hV : V = 288 * real.pi, from by simp [hr, real.pi, pow_three]
  let wedge_volume := V / 4
  have hwedge_volume : wedge_volume = 72 * real.pi, from by simp [hV, div_eq_mul_inv]
  exact hwedge_volume

end volume_of_wedge_of_sphere_l552_552784


namespace brenda_cakes_l552_552022

theorem brenda_cakes : 
  let cakes_per_day := 20
  let days := 9
  let total_cakes := cakes_per_day * days
  let sold_cakes := total_cakes / 2
  total_cakes - sold_cakes = 90 :=
by 
  sorry

end brenda_cakes_l552_552022


namespace least_changes_required_l552_552813

-- Define the initial matrix
def initial_matrix : Matrix (Fin 3) (Fin 3) ℕ := 
  ![
    ![5, 10, 0], 
    ![4, 6, 5], 
    ![6, 4, 5]
  ]

-- Define initial row sums
def initial_row_sums := [15, 15, 15]

-- Define initial column sums
def initial_column_sums := [15, 20, 10]

-- Problem statement: Prove that the least number of changes required to make 
-- all row and column sums distinct is 2.
theorem least_changes_required : ∃ (changes : List (Fin 3 × Fin 3 × ℕ)), 
  changes.length = 2 ∧ 
  let new_matrix := initial_matrix.map (λ r c m, 
    changes.find (λ ⟨x, y, v⟩, r = x ∧ c = y).map (λ ⟨_, _, v⟩, v).getOrElse m) in
  let new_row_sums := List.ofFn (λ r => Fin.fold (λ acc c => acc + new_matrix r c) 0) in
  let new_column_sums := List.ofFn (λ c => Fin.fold (λ acc r => acc + new_matrix r c) 0) in
  (new_row_sums ++ new_column_sums).nodup :=
by
  sorry -- proof is omitted as per instructions


end least_changes_required_l552_552813


namespace remaining_frustum_volume_fraction_l552_552400

-- Definitions and conditions from part a)
def original_base_edge : ℝ := 24
def original_altitude : ℝ := 18
def scaling_factor : ℝ := 1 / 3
def original_volume : ℝ := (original_base_edge / 2) ^ 2 * original_altitude * (1 / 3)
def smaller_volume : ℝ := (original_volume * (scaling_factor ^ 3))

-- The main theorem to prove the volume of the remaining frustum
theorem remaining_frustum_volume_fraction :
  let remaining_volume := original_volume - smaller_volume
  (remaining_volume / original_volume) = 26 / 27 :=
by
  sorry

end remaining_frustum_volume_fraction_l552_552400


namespace total_tables_made_l552_552376

def carpenter_tables (T_this_month : ℕ) (n : ℕ) : ℕ :=
  T_this_month + (T_this_month - n)

theorem total_tables_made :
  ∀ (T_this_month : ℕ) (n : ℕ),
    T_this_month = 10 →
    n = 3 →
    carpenter_tables T_this_month n = 17 :=
by
  intros T_this_month n ht hn
  rw [ht, hn]
  simp [carpenter_tables]
  sorry

end total_tables_made_l552_552376


namespace sufficient_condition_for_inequality_l552_552354

theorem sufficient_condition_for_inequality (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) → a ≥ 5 :=
by
  sorry

end sufficient_condition_for_inequality_l552_552354


namespace shaded_fraction_is_one_l552_552774

theorem shaded_fraction_is_one :
  let initial_fraction := 5 / 9 in
  let ratio := 4 / 9 in
  let infinite_sum := initial_fraction / (1 - ratio) in
  infinite_sum = 1 :=
begin
  have h1 : initial_fraction = 5 / 9 := rfl,
  have h2 : ratio = 4 / 9 := rfl,
  have h3 : infinite_sum = initial_fraction / (1 - ratio) := rfl,
  rcases h1, rcases h2, rcases h3,
  simp,
end

end shaded_fraction_is_one_l552_552774


namespace sum_altitudes_of_triangle_l552_552775
-- Import necessary libraries from Mathlib

-- Define the given line equation and declare the variables
variable (x y : ℝ)

-- Define the line equation for the given conditions
def line_eq (x y : ℝ) : Prop := 8 * x + 10 * y = 80

-- Define the sum of the lengths of the altitudes of the triangle formed by this line with the coordinate axes
def sum_of_altitudes (x y : ℝ) : ℝ := 18 + 40 / Real.sqrt 41

-- The proof statement we need to show
theorem sum_altitudes_of_triangle : (∃ (x y : ℝ), line_eq x y) → sum_of_altitudes x y = 18 + 40 / Real.sqrt 41 :=
by
  -- Instead of solving the equation, we use sorry to skip the actual proof
  sorry

end sum_altitudes_of_triangle_l552_552775


namespace equilateral_triangle_side_length_l552_552417

-- Definitions given in the conditions:
variable (r : ℝ)

-- Definitions derived from the problem setup:
noncomputable def inscribedRadius (a : ℝ) : ℝ := a * Real.sqrt 3 / 6
noncomputable def altitude (a : ℝ) : ℝ := a * Real.sqrt 3 / 3

-- The proof statement:
theorem equilateral_triangle_side_length :
  ∃ a : ℝ, a = 6 * r * Real.sqrt 3 :=
begin
  sorry
end

end equilateral_triangle_side_length_l552_552417


namespace determine_OP_l552_552671

noncomputable def point_on_line (a b c d e x : ℝ) : Prop :=
  a ≠ c + e - d ∧ x = (c * e - a * d) / (a - c + e - d)

theorem determine_OP (a b c d e : ℝ) (h : a ≠ c + e - d) :
  ∃ x : ℝ, point_on_line a b c d e x := 
begin
  use (c * e - a * d) / (a - c + e - d),
  split,
  exact h,
  refl,
end

end determine_OP_l552_552671


namespace fixed_points_temperature_count_l552_552497

def convert_to_celsius (F : ℤ) : ℚ :=
  5 / 9 * (F - 32)

def convert_to_fahrenheit (C : ℚ) : ℤ :=
  Int.floor (9 / 5 * C + 32)

def rounding_to_integer_conversion (F : ℤ) : ℤ :=
  convert_to_fahrenheit (convert_to_celsius F)

def is_fixed_point (F : ℤ) : Prop :=
  F = rounding_to_integer_conversion F

noncomputable def num_fixed_points_in_range (lo hi : ℤ) : ℕ :=
  (List.range' lo (hi - lo + 1)).filter (λ F, is_fixed_point F).length

theorem fixed_points_temperature_count :
  num_fixed_points_in_range (-100) 100 = 150 := 
sorry

end fixed_points_temperature_count_l552_552497


namespace sufficient_but_not_necessary_l552_552879

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a = 0) ↔ (a ≠ 0 ∧ b = 0) ↔ (a = 0 → ab = 0 ∧ ¬(ab = 0 → a = 0)) :=
by sorry

end sufficient_but_not_necessary_l552_552879


namespace sphere_hemisphere_radius_relationship_l552_552399

theorem sphere_hemisphere_radius_relationship (r : ℝ) (R : ℝ) (π : ℝ) (h : 0 < π):
  (4 / 3) * π * R^3 = (2 / 3) * π * r^3 →
  r = 3 * (2^(1/3 : ℝ)) →
  R = 3 :=
by
  sorry

end sphere_hemisphere_radius_relationship_l552_552399


namespace petya_sum_of_expressions_l552_552240

theorem petya_sum_of_expressions :
  (∑ val in (Finset.univ : Finset (Fin 32)), (1 +
    (if val / 2^4 % 2 = 0 then 2 else -2) +
    (if val / 2^3 % 2 = 0 then 3 else -3) +
    (if val / 2^2 % 2 = 0 then 4 else -4) +
    (if val / 2 % 2 = 0 then 5 else -5) +
    (if val % 2 = 0 then 6 else -6))) = 32 := 
by
  sorry

end petya_sum_of_expressions_l552_552240


namespace tangent_line_equation_l552_552286

noncomputable def tangentLineAtPoint (f : ℝ → ℝ) (x₀ y₀ : ℝ) := 
  let f' := deriv f x₀
  y₀ + f' * (x₀ - x₀)

theorem tangent_line_equation :
  ∀ (x : ℝ), (f : ℝ → ℝ) (y : ℝ → ℝ), 
  (∀ x, y x = 2 * x - Real.log x) →
  y 1 = 2 →
  (∀ x₀ : ℝ, deriv y x₀ = 2 - 1 / x₀) →
  tangentLineAtPoint y 1 2 = x + 1 := 
by
  simp
  sorry

end tangent_line_equation_l552_552286


namespace star_interior_angles_sum_l552_552446

theorem star_interior_angles_sum (n : ℕ) (h : n ≥ 7) : 
  let S := (fun n => 180 * (n - 2)) 
  in S n = 180 * (n - 2) :=
by
  sorry

end star_interior_angles_sum_l552_552446


namespace test_group_type_A_probability_atleast_one_type_A_group_probability_l552_552368

noncomputable def probability_type_A_group : ℝ :=
  let pA := 2 / 3
  let pB := 1 / 2
  let P_A1 := 2 * (1 - pA) * pA
  let P_A2 := pA * pA
  let P_B0 := (1 - pB) * (1 - pB)
  let P_B1 := 2 * (1 - pB) * pB
  P_B0 * P_A1 + P_B0 * P_A2 + P_B1 * P_A2

theorem test_group_type_A_probability :
  probability_type_A_group = 4 / 9 :=
by
  sorry

noncomputable def at_least_one_type_A_in_3_groups : ℝ :=
  let P_type_A_group := 4 / 9
  1 - (1 - P_type_A_group) ^ 3

theorem atleast_one_type_A_group_probability :
  at_least_one_type_A_in_3_groups = 604 / 729 :=
by
  sorry

end test_group_type_A_probability_atleast_one_type_A_group_probability_l552_552368


namespace scientific_notation_correct_l552_552003

def number := 56990000

theorem scientific_notation_correct : number = 5.699 * 10^7 :=
  by
    sorry

end scientific_notation_correct_l552_552003


namespace initial_mean_correctness_l552_552708

variable (M : ℝ)

theorem initial_mean_correctness (h1 : 50 * M + 20 = 50 * 36.5) : M = 36.1 :=
by 
  sorry

end initial_mean_correctness_l552_552708


namespace solve_problem_l552_552762
noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := abs (x - 1)

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0     := id
| (n+1) := λ x, g (f_n n (f x))

theorem solve_problem : set.count {x : ℝ | f_n 2015 x = 1}.to_finset = 2017 :=
sorry

end solve_problem_l552_552762


namespace roots_of_unity_expression_l552_552663

-- Defining the complex cube roots of unity
def omega := Complex.exp (2 * Real.pi * Complex.I / 3)
def omega2 := Complex.exp (-2 * Real.pi * Complex.I / 3)

-- Main theorem statement to prove
theorem roots_of_unity_expression :
  ((-1 + Complex.i * Real.sqrt 3) / 2) ^ 12 + ((-1 - Complex.i * Real.sqrt 3) / 2) ^ 12 = 2 :=
by
  -- Definitions of the cube roots and their properties
  have h1 : omega ^ 3 = 1 := sorry
  have h2 : omega2 ^ 3 = 1 := sorry
  have h3 : (-1 + Complex.i * Real.sqrt 3) / 2 = omega := sorry
  have h4 : (-1 - Complex.i * Real.sqrt 3) / 2 = omega2 := sorry
  -- Using the properties of the roots and their definitions to prove the statement
  sorry

end roots_of_unity_expression_l552_552663


namespace num_natural_numbers_divisors_count_l552_552481

theorem num_natural_numbers_divisors_count:
  ∃ N : ℕ, (2017 % N = 17 ∧ N ∣ 2000) ↔ 13 := 
sorry

end num_natural_numbers_divisors_count_l552_552481


namespace correct_mutually_exclusive_events_l552_552763

variables (balls : Finset (Fin 4)) (draws : Finset (Finset (Fin 4)))

-- Define events
def is_at_least_one_white_ball (drawn : Finset (Fin 4)) : Prop :=
  ∃ b ∈ drawn, b < 2

def is_both_white_balls (drawn : Finset (Fin 4)) : Prop :=
  drawn = {0, 1}

def is_at_least_one_red_ball (drawn : Finset (Fin 4)) : Prop :=
  ∃ b ∈ drawn, b >= 2

def is_both_red_balls (drawn : Finset (Fin 4)) : Prop :=
  drawn = {2, 3}

def is_exactly_one_white_ball (drawn : Finset (Fin 4)) : Prop :=
  ∃ b1 b2 ∈ drawn, b1 < 2 ∧ b2 >= 2

-- Define the event pairs
def event_pair_A (drawn : Finset (Fin 4)) : Prop :=
  is_at_least_one_white_ball drawn ∧ is_both_white_balls drawn

def event_pair_B (drawn : Finset (Fin 4)) : Prop :=
  is_at_least_one_white_ball drawn ∧ is_at_least_one_red_ball drawn

def event_pair_C (drawn : Finset (Fin 4)) : Prop :=
  is_exactly_one_white_ball drawn ∧ is_both_white_balls drawn

def event_pair_D (drawn : Finset (Fin 4)) : Prop :=
  is_at_least_one_white_ball drawn ∧ is_both_red_balls drawn

-- The goal is to prove that the correct answer is event pair D
theorem correct_mutually_exclusive_events : event_pair_D = true :=
by {
  sorry,  -- skipping the proof
}

end correct_mutually_exclusive_events_l552_552763


namespace mean_greater_median_by_six_l552_552849

theorem mean_greater_median_by_six (x : ℕ) (h : x > 0) :
  let nums := [x, x + 2, x + 4, x + 7, x + 37]
  let median := nums.nth (nums.length / 2)
  let mean := nums.sum / nums.length
  mean = median + 6 :=
by
  let median := x + 4
  let mean := x + 10
  have h1 : mean = median + 6, by
    simp [median, mean, add_comm]
  sorry

end mean_greater_median_by_six_l552_552849


namespace petya_sum_expression_l552_552251

theorem petya_sum_expression : 
  (let expressions := finset.image (λ (s : list bool), 
    list.foldl (λ acc ⟨b, n⟩, if b then acc + n else acc - n) 1 (s.zip [2, 3, 4, 5, 6])) 
    (finset.univ : finset (vector bool 5))) in
    expressions.sum) = 32 := 
sorry

end petya_sum_expression_l552_552251


namespace intersection_of_sets_example_l552_552560

open Set

theorem intersection_of_sets_example :
  let A := {1, 2, 3}
  let B := {2, 4, 5}
  A ∩ B = {2} := by
  sorry

end intersection_of_sets_example_l552_552560


namespace consequent_in_ratio_4_6_l552_552955

theorem consequent_in_ratio_4_6 (h : 4 = 6 * (20 / x)) : x = 30 := 
by
  have h' : 4 * x = 6 * 20 := sorry -- cross-multiplication
  have h'' : x = 120 / 4 := sorry -- solving for x
  have hx : x = 30 := sorry -- simplifying 120 / 4

  exact hx

end consequent_in_ratio_4_6_l552_552955


namespace log_ineq_l552_552866

open Real

theorem log_ineq (a : ℝ) (x y : ℝ) (h1 : 0 < a ∧ a < 1) (h2 : x^2 + y = 0) : log a (a^x + a^y) ≤ log a 2 + 1 / 8 :=
sorry

end log_ineq_l552_552866


namespace sqrt_D_irrational_l552_552995

theorem sqrt_D_irrational (x : ℤ) :
  let a := x
  let b := 2 * x
  let c := x + 2
  let D := a^2 + b^2 + c^2
  ∃ q : ℚ, q^2 = D → False :=
begin
  sorry -- Proof not required per instructions
end

end sqrt_D_irrational_l552_552995


namespace min_empty_cells_l552_552172

-- Given conditions
variables (nΔ n∇ : ℕ)
variables (grasshopper_in_Δ grasshopper_in_∇ : ℕ → Prop)
variables (jump_to_adjacent : ∀ n, grasshopper_in_Δ n → grasshopper_in_∇ (n - 1) ∨ grasshopper_in_∇ (n + 1))

-- Assumptions from the problem:
-- There are nΔ Δ cells, and n∇ ∇ cells. We know nΔ = n∇ + 3
axiom total_cells : nΔ = n∇ + 3

-- Theorem to prove:
theorem min_empty_cells
  (h_in_Δ : ∀ n, grasshopper_in_Δ n → Exists (grasshopper_in_∇))
  (h_in_∇ : ∀ n, grasshopper_in_∇ n → Exists (grasshopper_in_Δ)) :
  ∃ k, k = 3 ∧ ∀ m, empty_cells_after_jump nΔ n∇ jump_to_adjacent k  :=
begin
  sorry
end

end min_empty_cells_l552_552172


namespace permutation_of_candidates_l552_552799

theorem permutation_of_candidates : Nat.perm 15 5 = 360360 := by
  sorry

end permutation_of_candidates_l552_552799


namespace fraction_to_decimal_l552_552050

theorem fraction_to_decimal :
  (47 : ℚ) / (2 * 5^3) = 0.188 := 
by noncomputable def
  /-
  The problem is to show the equivalence of the fraction and its decimal form.
  Hence, we need to prove the terminating decimal form of the given fraction.
  Important note: "noncomputable def" is added to handle real number exact representations.
  -/
 sorry

end fraction_to_decimal_l552_552050


namespace remainder_of_product_l552_552611

theorem remainder_of_product (a b c : ℕ) (hc : c ≥ 3) (h1 : a % c = 1) (h2 : b % c = 2) : (a * b) % c = 2 :=
by
  sorry

end remainder_of_product_l552_552611


namespace allan_balloons_correct_l552_552007

def balloons_allan_brought (A J : ℕ) := A = J + 2 ∧ J = 3 → A = 5

theorem allan_balloons_correct : ∀ (A J : ℕ), balloons_allan_brought A J :=
by
  intros A J
  unfold balloons_allan_brought
  intro h
  cases h with h1 h2
  rw [h2, add_comm, add_assoc, add_zero] at h1
  exact h1

end allan_balloons_correct_l552_552007


namespace shortest_path_cylinder_shortest_path_cone_l552_552344

-- Defining the problem for the circular cylinder
def shortest_distance_on_cylinder (r h : ℝ) (A B : ℝ × ℝ) : ℝ :=
  sorry -- Helical distance calculation goes here

-- Theorem for Part (a): Circular Cylinder
theorem shortest_path_cylinder (r h : ℝ) (A B : ℝ × ℝ) : shortest_distance_on_cylinder r h A B = 
  begin
    -- Path length should be equal to the helical path length.
    sorry
  end

-- Defining the problem for the circular cone
def shortest_distance_on_cone (r h : ℝ) (A B : ℝ × ℝ) : ℝ :=
  sorry -- Straight-line segment mapped to cone goes here

-- Theorem for Part (b): Circular Cone
theorem shortest_path_cone (r h : ℝ) (A B : ℝ × ℝ) : shortest_distance_on_cone r h A B = 
  begin
    -- Path length should be equal to the line segment mapped back on the cone.
    sorry
  end

end shortest_path_cylinder_shortest_path_cone_l552_552344


namespace petya_sum_of_all_combinations_l552_552245

-- Define the expression with the possible placements of signs.
def petyaExpression : List (ℤ → ℤ → ℤ) :=
  [int.add, int.sub] -- Combination of possible operations at each position

-- Calculate the total number of ways to insert "+" and "-" in the expression
def number_of_combinations : ℕ := 2^5

-- Define the problem statement in Lean 4
theorem petya_sum_of_all_combinations : 
  (∑ idx in Finset.range number_of_combinations, 1) = 32 := by
  sorry

end petya_sum_of_all_combinations_l552_552245


namespace simplify_cubic_root_l552_552268

theorem simplify_cubic_root :
  (∛(20^3 + 30^3 + 40^3 + 60^3) = 10 * ∛315) :=
by
  sorry

end simplify_cubic_root_l552_552268


namespace sum_first_n_terms_l552_552135

noncomputable def a_n (n : ℕ) : ℚ :=
  ((n + 1)^4 + n^4 + 1) / ((n + 1)^2 + n^2 + 1)

theorem sum_first_n_terms (n : ℕ) : 
  (∑ k in Finset.range n, a_n (k + 1)) = (n^3 + 3 * n^2 + 5 * n) / 3 :=
sorry

end sum_first_n_terms_l552_552135


namespace probability_bypass_kth_intersection_l552_552385

variable (n k : ℕ)

def P (n k : ℕ) : ℚ := (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2

theorem probability_bypass_kth_intersection :
  P n k = (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2 :=
by
  sorry

end probability_bypass_kth_intersection_l552_552385


namespace find_b_plus_c_l552_552561

-- Define the parallel lines conditions
def line1 (x y : ℝ) : ℝ := 3*x + 4*y + 5
def line2 (x y : ℝ) (b c : ℝ) : ℝ := 6*x + b*y + c

-- Given the distance formula between parallel lines
def distance_between_lines {A B C1 C2 : ℝ} :
  abs (C1 - C2) / real.sqrt (A^2 + B^2) = 3 :=
sorry

-- Define the proof problem
theorem find_b_plus_c (b c : ℝ) (h1 : b = 8) (h2 : distance_between_lines 6 8 10 c) :
  b + c = -12 ∨ b + c = 48 :=
sorry

end find_b_plus_c_l552_552561


namespace trajectory_midpoint_P_maximum_area_and_line_l552_552112

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the condition of point M and line l passing through M
def point_M : (ℝ × ℝ) := (-1, 0)
def line_l (x y : ℝ) : Prop :=
  ∃ m : ℝ, x = m * y - 1

-- Define the midpoint of segment AB
def midpoint_P (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Proof for the trajectory of midpoint P
theorem trajectory_midpoint_P :
  ∀ (x y : ℝ), (∃ (x1 y1 x2 y2 : ℝ), ellipse x1 y1 ∧ ellipse x2 y2 ∧ (midpoint_P x1 y1 x2 y2 = (x, y)))
  ↔ x^2 + x + 4 * y^2 = 0 :=
by sorry

-- The area of triangle OAB
def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1 / 2) * real.abs (x1 * y2 - x2 * y1)

-- Proof for the maximum area and line equation at this time
theorem maximum_area_and_line :
  ∃ (m : ℝ) (x1 y1 x2 y2 : ℝ),
    ellipse x1 y1 ∧ ellipse x2 y2 ∧ line_l x1 y1 ∧ line_l x2 y2 ∧
    (triangle_area x1 y1 x2 y2 = real.sqrt 3 / 2 ∧ line_l -1 0) :=
by sorry

end trajectory_midpoint_P_maximum_area_and_line_l552_552112


namespace triangle_ineq_l552_552760

theorem triangle_ineq (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) : 2 * (a^2 + b^2) > c^2 := 
by 
  sorry

end triangle_ineq_l552_552760


namespace simplify_expression_l552_552669

-- Define the complex numbers and conditions
def ω : ℂ := (-1 + complex.I * real.sqrt 3) / 2
def ω_conj : ℂ := (-1 - complex.I * real.sqrt 3) / 2

-- Conditions
axiom ω_is_root_of_unity : ω^3 = 1
axiom ω_conj_is_root_of_unity : ω_conj^3 = 1

-- Theorem statement
theorem simplify_expression : ω^12 + ω_conj^12 = 2 := by
  sorry

end simplify_expression_l552_552669


namespace student_pickup_and_supervisor_average_l552_552703

theorem student_pickup_and_supervisor_average:
  ∃ (students_per_stop supervisors_per_bus : ℕ), 
  let buses_supervisors := [4, 5, 3, 6, 7],
      original_students := 200,
      stops := 3,
      additional_supervisors_per_stop := 2,
      supervisor_student_ratio := 1 / 10 in
  let total_supervisors := sum buses_supervisors in
  (total_supervisors * 10) ≥ original_students ∧
  let extra_supervisors := stops * additional_supervisors_per_stop in
  let total_supervisors_new := total_supervisors + extra_supervisors in
  let total_students_new := total_supervisors_new * 10 in
  let additional_students := total_students_new - original_students in
  let students_per_stop := additional_students / stops in
  students_per_stop = 36 ∧
  (total_supervisors_new / 5) = 6.2 :=
by
  -- We write the proof manually using the tactics
  rcases exists_eq_intro ((total_supervisors_new / 5)) _ with ⟨supervisors_per_bus, rfl⟩
  -- Skipping the proof
  sorry

end student_pickup_and_supervisor_average_l552_552703


namespace num_families_shared_vacation_rental_l552_552274

theorem num_families_shared_vacation_rental (f : ℕ)
  (h1 : ∀ (p d : ℕ), p = 4 → d = 7 → 28)
  (h2 : ∀ (l s : ℕ), l = 6 → s = 14 → 84)
  (h3 : 84 / 28 = f) :
  f = 3 := by
  -- definitions and theorem proof
  sorry

end num_families_shared_vacation_rental_l552_552274


namespace domain_of_lg_abs_x_minus_1_l552_552285

theorem domain_of_lg_abs_x_minus_1 (x : ℝ) : 
  (|x| - 1 > 0) ↔ (x < -1 ∨ x > 1) := 
by
  sorry

end domain_of_lg_abs_x_minus_1_l552_552285


namespace sum_of_powers_l552_552049

-- Given
variable (m n: ℤ)

def sum_m_to_n (m n : ℤ) : ℤ := m + m^2 + m^3 + ⋯ + m^n
def sum_m_to_nm (m n : ℤ) : ℤ := m + 2 * m^2 + 3 * m^3 + ... + n * m^n

theorem sum_of_powers (A : ℤ) (m n : ℤ) (h₁ : A = m + m^2 + ..., h₂ : A = (m^(n+1) - m) / (m - 1)) :
  sum_m_to_nm m n = ((n+1) * (m^(n+1) - m)) / (2 * (m - 1)) :=
sorry

end sum_of_powers_l552_552049


namespace range_of_a_l552_552107

noncomputable def f : ℝ → ℝ := sorry -- The function f is not explicitly defined

axiom A1 : ∀ x : ℝ, x > 0 → f x = sorry -- Define f on its domain (0, +∞)
axiom A2 : f 1 = Real.exp 1 -- f(1) = e
axiom A3 : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → (f(x1) - f(x2)) / (x1 * x2) > (Real.exp x2 / x1) - (Real.exp x1 / x2) 
axiom A4 : ∀ a : ℝ, 0 < a → f (Real.log a) > 2 * Real.exp 1 - a * Real.log a

theorem range_of_a (a : ℝ) (h : f (Real.log a) > 2 * Real.exp 1 - a * Real.log a) : 1 < a ∧ a < Real.exp 1 :=
sorry

end range_of_a_l552_552107


namespace population_net_change_l552_552325

theorem population_net_change :
  let initial_population := 1.0
  let factor1 := 1.3
  let factor2 := 0.7
  let factor3 := 1.3
  let factor4 := 0.85
  let factor5 := 0.8
  let final_population := initial_population * factor1 * factor2 * factor3 * factor4 * factor5
  let net_change := (final_population - initial_population) / initial_population * 100
  round net_change = -37 :=
by {
  sorry
}

end population_net_change_l552_552325


namespace seq_a_solution_l552_552864

noncomputable def seq_a : ℕ → ℤ
| 0       := -1
| 1       := 1
| (n+2) := 2 * seq_a (n+1) + 3 * seq_a n + 3^(n+2)

theorem seq_a_solution (n : ℕ) : 
  seq_a n = (1 : ℤ) / 16 * ((4 * (n : ℤ) - 3) * 3 ^ (n + 1) - 7 * (-1) ^ n) :=
sorry

end seq_a_solution_l552_552864


namespace miles_driven_each_day_l552_552462

theorem miles_driven_each_day
  (total_distance : ℕ)
  (days_in_semester : ℕ)
  (h_total : total_distance = 1600)
  (h_days : days_in_semester = 80):
  total_distance / days_in_semester = 20 := by
  sorry

end miles_driven_each_day_l552_552462


namespace inscribed_circle_radius_l552_552715

theorem inscribed_circle_radius (r a : ℝ) (h_r_pos : r > 0) (h_a_pos : a > 0) : 
  ∃ x : ℝ, x = r * a / (a + 2 * r) :=
by
  use r * a / (a + 2 * r)
  sorry

end inscribed_circle_radius_l552_552715


namespace distance_squared_center_l552_552776

theorem distance_squared_center
  (r : ℝ) (h_r : r = sqrt 72)
  (AB BC : ℝ)
  (h_AB : AB = 8) (h_BC : BC = 6)
  (h_right : ∠ABC = π / 2)
  (a b : ℝ)
  (h_A : (a, b + 8) ∈ { p | p.1^2 + p.2^2 = 72 })
  (h_C : (a + 6, b) ∈ { p | p.1^2 + p.2^2 = 72 }) :
  a^2 + b^2 = 17 :=
by
  sorry

end distance_squared_center_l552_552776


namespace calculate_expression_l552_552026

theorem calculate_expression : 1000 * 2.998 * 2.998 * 100 = (29980)^2 := 
by
  sorry

end calculate_expression_l552_552026


namespace smallest_positive_integer_satisfying_conditions_l552_552491

theorem smallest_positive_integer_satisfying_conditions :
  ∃ (N : ℕ), N = 242 ∧
    ( ∃ (i : Fin 4), (N + i) % 8 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 9 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 25 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 121 = 0 ) :=
sorry

end smallest_positive_integer_satisfying_conditions_l552_552491


namespace circle_BOC_bisects_AK_at_M_l552_552217

noncomputable def midpoint (A K : Point) : Point := sorry

variables (Ω ω : Circle) (A B C K : Point) (O : Point)

-- Given conditions
variable (h1 : Ω ∈ internalTangent ω A)
variable (h2 : BC_chord : BC ⊂ Ω ∧ touches ω K)
variable (h3 : O_center : O = center(ω))
variable (M : Point := midpoint A K)

-- Main statement
theorem circle_BOC_bisects_AK_at_M :
  bisects (circle_through B O C) (segment A K) M :=
sorry

end circle_BOC_bisects_AK_at_M_l552_552217


namespace investment_of_C_l552_552750

-- Define the conditions
variables (a b c : ℝ)
variables (c_share total_profit : ℝ)

def condition1 : a = 5000 := by sorry
def condition2 : b = 15000 := by sorry
def condition3 : c_share = 3000 := by sorry
def condition4 : total_profit = 5000 := by sorry

-- Define the equation to solve
def investment_c (x : ℝ) : Prop :=
  c_share / total_profit = x / (a + b + x)

-- Given all conditions, prove C's investment amount
theorem investment_of_C : ∃ x : ℝ, investment_c x ∧ x = 30000 :=
by
  sorry

end investment_of_C_l552_552750


namespace positive_difference_of_x_in_triangle_ABC_l552_552972

theorem positive_difference_of_x_in_triangle_ABC 
    (x : ℝ) (h₁ : 2 < x) (h₂ : x < 18) :
    (17 - 3 = 14) := by
  -- Ensure that the values being considered are integers
  let lower : ℤ := 3
  let upper : ℤ := 17
  have h3 : 0 < (upper - lower) := by
    norm_num
  exact h3

end positive_difference_of_x_in_triangle_ABC_l552_552972


namespace distance_to_right_focus_eq_4_l552_552881

noncomputable def hyperbola_focus_distance
  (P : ℝ × ℝ)
  (hP : P.1 ^ 2 / 16 - P.2 ^ 2 / 9 = 1)
  (distance_to_left_focus : ℝ)
  (h_distance_to_left_focus : distance_to_left_focus = 12) :
  ℝ :=
12 - 8

theorem distance_to_right_focus_eq_4
  (P : ℝ × ℝ)
  (hP : P.1 ^ 2 / 16 - P.2 ^ 2 / 9 = 1)
  (distance_to_left_focus : ℝ)
  (h_distance_to_left_focus : distance_to_left_focus = 12) :
  ∃ d, d = 4 :=
begin
  use 12 - 8,
  exact rfl,
end

end distance_to_right_focus_eq_4_l552_552881


namespace eesha_late_by_15_minutes_l552_552640

theorem eesha_late_by_15_minutes 
  (T usual_time : ℕ) (delay : ℕ) (slower_factor : ℚ) (T' : ℕ) 
  (usual_time_eq : usual_time = 60)
  (delay_eq : delay = 30)
  (slower_factor_eq : slower_factor = 0.75)
  (new_time_eq : T' = unusual_time * slower_factor) 
  (T'' : ℕ) (total_time_eq: T'' = T' + delay)
  (time_taken : ℕ) (time_diff_eq : time_taken = T'' - usual_time) :
  time_taken = 15 :=
by
  -- Proof construction
  sorry

end eesha_late_by_15_minutes_l552_552640


namespace Petya_sum_l552_552254

theorem Petya_sum : 
  let expr := [1, 2, 3, 4, 5, 6]
  let values := 2^(expr.length - 1)
  (sum_of_possible_values expr = values) := by 
  sorry

end Petya_sum_l552_552254


namespace average_annual_growth_rate_l552_552713

theorem average_annual_growth_rate (x : ℝ) (h : (1 + x)^2 = 1.20) : x < 0.1 :=
sorry

end average_annual_growth_rate_l552_552713


namespace log₅_twelve_in_terms_of_a_b_l552_552097

variables (a b : ℝ)

def log₅ (x : ℝ) : ℝ := log x / log 5

theorem log₅_twelve_in_terms_of_a_b
  (h₁ : log 2 = a)
  (h₂ : log 3 = b) :
  log₅ 12 = (2 * a + b) / (1 - a) :=
by sorry

end log₅_twelve_in_terms_of_a_b_l552_552097


namespace emily_subtract_l552_552730

theorem emily_subtract (n : ℕ) (h : n = 50) : (n - 1)^2 = n^2 - 99 :=
by
  rw h
  norm_num
  sorry

end emily_subtract_l552_552730


namespace remaining_clothes_correct_l552_552814

def fold_clothes (initial_shirts initial_pants initial_shorts : ℕ)
                 (rate_shirts rate_pants rate_shorts : ℕ)
                 (time_shirts time_pants total_time : ℕ) : (ℕ × ℕ × ℕ) :=
  let folded_shirts := rate_shirts * time_shirts / 60
  let remaining_shirts := initial_shirts - folded_shirts
  let folded_pants := rate_pants * time_pants / 60
  let remaining_pants := initial_pants - folded_pants
  let remaining_time := total_time - (time_shirts + time_pants + 15 + 10)
  let folded_shorts := rate_shorts * remaining_time / 60
  let remaining_shorts := initial_shorts - folded_shorts in
  (remaining_shirts, remaining_pants, remaining_shorts)

theorem remaining_clothes_correct :
  fold_clothes 30 15 20 12 8 10 45 30 120 = (21, 11, 17) :=
by
  sorry

end remaining_clothes_correct_l552_552814


namespace west_travel_recorded_as_neg_five_l552_552930

-- Define the distances traveled
def east_distance := 5
def west_distance := -5

-- Condition: Traveling east for 5 kilometers is recorded as 5 kilometers.
axiom travels_east (east_distance_recorded : ℤ) : east_distance_recorded = east_distance

-- Proposition: Traveling west for 5 kilometers should be recorded as -5 kilometers.
def travels_west (west_distance_recorded : ℤ) : Prop := west_distance_recorded = west_distance

-- Theorem: If traveling east is recorded as 5 kilometers, then traveling west is recorded as -5 kilometers.
theorem west_travel_recorded_as_neg_five (east_distance_recorded : ℤ) :
  travels_east east_distance_recorded → travels_west (-east_distance_recorded) :=
by
  intros h
  rw [h]
  exact sorry

end west_travel_recorded_as_neg_five_l552_552930


namespace find_angle_BMC_l552_552353

-- Define the conditions in Lean 4.
axiom AB : Type
axiom AC : Type
axiom B : AB → Prop
axiom C : AC → Prop
axiom M : Prop
axiom Circle : Type
axiom angle : AB → AC → Circle → ℝ -- Define the angle between two chords in a circle.
axiom tangent : Circle → Prop -- Define a tangent in a circle.
axiom intersection : tangent Circle → tangent Circle → Prop -- Tangents intersect.

-- Use the given conditions.
axiom angle_BAC_eq_70 : ∀ (circle : Circle) (ab : AB) (ac : AC), angle ab ac circle = 70
axiom tangents_at_BC : ∀ (circle : Circle) (b c : tangent circle), intersection b c → M

-- The proof statement.
theorem find_angle_BMC (circle : Circle) (ab : AB) (ac : AC) (b c : tangent circle) (m : M) :
  (intersection b c → angle ab ac circle = 140) →
  (angle ab ac circle = 140 → angle b c circle = 40) :=
by 
  sorry -- Proof is not required.

end find_angle_BMC_l552_552353


namespace num_prime_sums_first_15_l552_552307

noncomputable def seq_sums_of_primes : ℕ → ℕ
| 0 => 3
| n+1 => seq_sums_of_primes n + Nat.prime (n+1+1)

theorem num_prime_sums_first_15 : 
  (Finset.filter (Nat.Prime) (Finset.range 15)).card = 3 :=
by
  sorry

end num_prime_sums_first_15_l552_552307


namespace even_function_value_sum_l552_552614

noncomputable def g (x : ℝ) (d e f : ℝ) : ℝ :=
  d * x^8 - e * x^6 + f * x^2 + 5

theorem even_function_value_sum (d e f : ℝ) (h : g 15 d e f = 7) :
  g 15 d e f + g (-15) d e f = 14 := by
  sorry

end even_function_value_sum_l552_552614


namespace smiths_loads_of_laundry_l552_552589

/-
  Problem:
  Given:
  - Kylie uses 3 bath towels.
  - Her 2 daughters use a total of 6 bath towels.
  - Her husband uses a total of 3 bath towels.
  - The washing machine can fit 4 bath towels for one load of laundry.
  
  Prove:
  The Smiths need 3 loads of laundry to clean all of their used towels.
-/

def total_towels (k: ℕ) (d: ℕ) (h: ℕ) : ℕ := k + d + h
def loads_needed (towels: ℕ) (capacity: ℕ) : ℕ := towels / capacity

theorem smiths_loads_of_laundry : 
  total_towels 3 6 3 / 4 = 3 :=
by 
  simp [total_towels, loads_needed]
  exact rfl

end smiths_loads_of_laundry_l552_552589


namespace determine_a_l552_552092

-- Define the sets A and B
def A : Set ℕ := {1, 3}
def B (a : ℕ) : Set ℕ := {1, 2, a}

-- The proof statement
theorem determine_a (a : ℕ) (h : A ⊆ B a) : a = 3 :=
by 
  sorry

end determine_a_l552_552092


namespace sum_of_squares_invariant_l552_552724

theorem sum_of_squares_invariant (a b c : ℝ) :
  (a = 89) → (b = 12) → (c = 3) →
  (∀ x y z : ℝ, ((x = (a + b) / real.sqrt 2 ∧ y = (a - b) / real.sqrt 2 ∧ z = c) ∨
                 (x = (b + c) / real.sqrt 2 ∧ y = (b - c) / real.sqrt 2 ∧ z = a) ∨
                 (x = (a + c) / real.sqrt 2 ∧ y = (a - c) / real.sqrt 2 ∧ z = b) →
                 (x^2 + y^2 + z^2 = a^2 + b^2 + c^2))) →
  ¬ ((a', b', c') = (90, 10, 14))
:= 
begin
  intros h1 h2 h3 h4,
  sorry  -- Proof to be completed
end

end sum_of_squares_invariant_l552_552724


namespace sum_of_coefficients_l552_552463

noncomputable def polynomialExpr (c : ℝ) : ℝ :=
  2 * (c - 2) * (c^2 + c * (4 - c))

theorem sum_of_coefficients : 
  polynomialExpr = λ c, 8 * c^2 - 16 * c → 8 + (-16) = -8 :=
by sorry

end sum_of_coefficients_l552_552463


namespace product_of_lengths_l552_552620

-- Definitions and conditions
variables (C : Type*) [MetricSpace C] [NormedGroup C] [NormedSpace ℝ C]
variables (S N E W A B A' B' : C)
variables (l : AffineSubspace ℝ C)
variables (circle : C → ℝ → Set C) -- Circle definition with center and radius

-- Variables for circle and points
variables (radius : ℝ) (center : C) (on_circle : ∀ (c : C), c ∈ circle center radius → c = S ∨ c = N ∨ c = E ∨ c = W ∨ c = A ∨ c = B)

-- Conditions translation
def perpendicular (x y : C) : Prop := inner x y = 0
def tangent (l : AffineSubspace ℝ C) (S : C) (circle_center : C) (radius : ℝ) : Prop :=
  ∃ (v : C) (hv : v ∈ l.direction), v ≠ 0 ∧ ∀ (x ∈ l), dist x circle_center = radius ∧ inner v (x - S) = 0

axiom diameters_perpendicular : perpendicular (N - S) (E - W)
axiom tangent_at_S : tangent l S circle radius
axiom symmetric_AB : (A - E) = -(B - E) -- Symmetry condition

-- Intersection points with line l
axiom intersection_A' : ∃ (x ∈ l), x = A' ∧ ∃ (y ∈ l), y = N + tan (angle (N - S) (A - N))
axiom intersection_B' : ∃ (x ∈ l), x = B' ∧ ∃ (y ∈ l), y = N + tan (angle (N - S) (B - N))

-- Prove the product of distances equal to sq distance
theorem product_of_lengths :
  dist S A' * dist S B' = dist S N * dist S N :=
sorry

end product_of_lengths_l552_552620


namespace log_powers_comparison_l552_552075

open Real

theorem log_powers_comparison (m : ℝ) (hm : m > 1) :
  (m > 10 → (log m)^0.9 > (log m)^0.8) ∧
  (1 < m ∧ m < 10 → (log m)^0.9 < (log m)^0.8) ∧
  (m = 10 → (log m)^0.9 = (log m)^0.8) :=
by sorry

end log_powers_comparison_l552_552075


namespace number_of_valid_N_l552_552474

theorem number_of_valid_N : 
  { N : ℕ // 2017 ≡ 17 [MOD N] ∧ N > 17 }.card = 13 :=
by sorry

end number_of_valid_N_l552_552474


namespace simson_line_l552_552356

-- Define the conditions of the problem
variables {A B C P : Type}
variables [Point_type A] [Point_type B] [Point_type C] [Point_type P]
variable (Δ : Triangle A B C)
variable (circumcircle : Circle Δ P)
variable (A1 : Foot_of_perpendicular P B C)
variable (B1 : Foot_of_perpendicular P C A)
variable (C1 : Foot_of_perpendicular P A B)

-- Prove that A1, B1, C1 are collinear
theorem simson_line (hP : Point_on_circumcircle Δ P) :
  Collinear A1 B1 C1 := by 
  sorry

end simson_line_l552_552356


namespace find_angle4_l552_552858

theorem find_angle4 : 
  ∀ {angle1 angle2 angle3 angle4 : ℝ}, 
  (angle1 + angle2 = 180) → (angle3 = angle4) → (50 + 60 + angle1 = 180) → (angle4 = 35) :=
by 
  intro angle1 angle2 angle3 angle4
  assume h1 : angle1 + angle2 = 180
  assume h2 : angle3 = angle4
  assume h3 : 50 + 60 + angle1 = 180
  sorry

end find_angle4_l552_552858


namespace determine_a_l552_552091

-- Define the sets A and B
def A : Set ℕ := {1, 3}
def B (a : ℕ) : Set ℕ := {1, 2, a}

-- The proof statement
theorem determine_a (a : ℕ) (h : A ⊆ B a) : a = 3 :=
by 
  sorry

end determine_a_l552_552091


namespace f_2_value_l552_552510

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x - 4

theorem f_2_value :
  (f a b (-2)) = 2 → (f a b 2) = -10 :=
by
  intro h
  -- Provide the solution steps here, starting with simplifying the equation. Sorry for now
  sorry

end f_2_value_l552_552510


namespace part1_part2_l552_552860

-- Define the conditions and the problem statement
def f (α : Real) : Real :=
  (sin ((π / 2) + α) * sin (2 * π - α)) / (cos (-π - α) * sin ((3 / 2) * π + α))

theorem part1 (α : Real) (h1 : ∀ x : Real, π < x ∧ x < (3 * π / 2) → x = α)
(h2 : cos (α - (3 / 2) * π) = 1 / 5) : f α = sqrt 6 / 12 := sorry

theorem part2 (α : Real) (h1 : f α = -2) : 2 * sin α * cos α + cos α ^ 2 = 1 := sorry

end part1_part2_l552_552860


namespace wood_pieces_gathered_l552_552803

theorem wood_pieces_gathered (sacks : ℕ) (pieces_per_sack : ℕ) (total_pieces : ℕ)
  (h1 : sacks = 4)
  (h2 : pieces_per_sack = 20)
  (h3 : total_pieces = sacks * pieces_per_sack) :
  total_pieces = 80 :=
by
  sorry

end wood_pieces_gathered_l552_552803


namespace petya_sum_l552_552236

theorem petya_sum : 
  let f (signs : fin 5 → bool) : ℤ :=
    1 + (if signs 0 then 2 else -2) + (if signs 1 then 3 else -3) + (if signs 2 then 4 else -4) + (if signs 3 then 5 else -5) + (if signs 4 then 6 else -6),
  sum (f '' (finset.univ : finset (fin 5 → bool))) = 32 :=
by
  sorry

end petya_sum_l552_552236


namespace cases_in_1990_l552_552582

theorem cases_in_1990 (initial_cases : ℕ) (final_cases : ℕ) (start_year : ℕ) (end_year : ℕ) (query_year : ℕ)
  (h_initial : initial_cases = 600000)
  (h_final : final_cases = 2000)
  (h_start : start_year = 1970)
  (h_end : end_year = 2000)
  (h_query : query_year = 1990)
  (h_linear_decrease : ∀ t, t ∈ set.Icc start_year end_year → 
    initial_cases - (initial_cases - final_cases) * (t - start_year) / (end_year - start_year) = 
    (if t = start_year then initial_cases else if t = end_year then final_cases else initial_cases - (initial_cases - final_cases) * (t - start_year) / (end_year - start_year))) :
  let expected_cases := 201333 in
    initial_cases - (initial_cases - final_cases) * (query_year - start_year) / (end_year - start_year) = expected_cases :=
by 
  sorry

end cases_in_1990_l552_552582


namespace investment_of_q_is_correct_l552_552751

-- Define investments and the profit ratio
def p_investment : ℝ := 30000
def profit_ratio_p : ℝ := 2
def profit_ratio_q : ℝ := 3

-- Define q's investment as x
def q_investment : ℝ := 45000

-- The goal is to prove that q_investment is indeed 45000 given the above conditions
theorem investment_of_q_is_correct :
  (p_investment / q_investment) = (profit_ratio_p / profit_ratio_q) :=
sorry

end investment_of_q_is_correct_l552_552751


namespace pole_length_is_5_l552_552350

theorem pole_length_is_5 (x : ℝ) (gate_width gate_height : ℝ) 
  (h_gate_wide : gate_width = 3) 
  (h_pole_taller : gate_height = x - 1) 
  (h_diagonal : x^2 = gate_height^2 + gate_width^2) : 
  x = 5 :=
by
  sorry

end pole_length_is_5_l552_552350


namespace beth_total_crayons_l552_552426

-- Defining the conditions as variables
variable (packs : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ) (borrowed_crayons : ℕ)

-- Stating the main theorem
theorem beth_total_crayons :
  packs = 8 →
  crayons_per_pack = 12 →
  extra_crayons = 15 →
  borrowed_crayons = 7 →
  (packs * crayons_per_pack + extra_crayons + borrowed_crayons) = 118 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end beth_total_crayons_l552_552426


namespace statement_II_must_be_true_l552_552282

theorem statement_II_must_be_true 
  (statements : List Prop)
  (I: "The digit or symbol is not a number" ∈ statements)
  (II: "The digit or symbol is not '%' ∈ statements")
  (III: "The digit or symbol is 3" ∈ statements)
  (IV: "The digit or symbol is not 4" ∈ statements) 
  (three_true_one_false : ∃ exactly_three_true (statements)) :
  "The digit or symbol is not '%'" :=
sorry

end statement_II_must_be_true_l552_552282


namespace saturday_earnings_l552_552766

-- Lean 4 Statement

theorem saturday_earnings 
  (S Wednesday_earnings : ℝ)
  (h1 : S + Wednesday_earnings = 5182.50)
  (h2 : Wednesday_earnings = S - 142.50) 
  : S = 2662.50 := 
by
  sorry

end saturday_earnings_l552_552766


namespace vertex_of_parabola_minimum_value_for_x_ge_2_l552_552134

theorem vertex_of_parabola :
  ∀ x y : ℝ, y = x^2 + 2*x - 3 → ∃ (vx vy : ℝ), (vx = -1) ∧ (vy = -4) :=
by
  sorry

theorem minimum_value_for_x_ge_2 :
  ∀ x : ℝ, x ≥ 2 → y = x^2 + 2*x - 3 → ∃ (min_val : ℝ), min_val = 5 :=
by
  sorry

end vertex_of_parabola_minimum_value_for_x_ge_2_l552_552134


namespace angle_quadrant_l552_552278

theorem angle_quadrant (theta : ℤ) (h_theta : theta = -3290) : 
  ∃ q : ℕ, q = 4 := 
by 
  sorry

end angle_quadrant_l552_552278


namespace price_adjustment_l552_552785

theorem price_adjustment (P : ℝ) (x : ℝ) (hx : P * (1 - (x / 100)^2) = 0.75 * P) : 
  x = 50 :=
by
  -- skipping the proof with sorry
  sorry

end price_adjustment_l552_552785


namespace part_I_part_II_l552_552118

section part_I

def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 3) * Real.cos x

theorem part_I (h : x ∈ Icc 0 (Real.pi / 2)) : 
  0 ≤ f x ∧ f x ≤ 1 + Real.sqrt 3 / 2 :=
sorry

end part_I

section part_II

variables {A B : ℝ} {a b c : ℝ}

def angle_A_is_acute (A : ℝ) : Prop := 0 < A ∧ A < Real.pi / 2

def f_eq (A : ℝ) : Prop := f A = Real.sqrt 3 / 2

theorem part_II (ha : angle_A_is_acute A) (hf : f_eq A) (hb : b = 2) (hc : c = 3) :
  Real.cos (A - B) = 5 * Real.sqrt 7 / 14 :=
sorry

end part_II

end part_I_part_II_l552_552118


namespace fourth_powers_sum_is_8432_l552_552457

def sum_fourth_powers (x y : ℝ) : ℝ := x^4 + y^4

theorem fourth_powers_sum_is_8432 (x y : ℝ) (h₁ : x + y = 10) (h₂ : x * y = 4) : 
  sum_fourth_powers x y = 8432 :=
by
  sorry

end fourth_powers_sum_is_8432_l552_552457


namespace mean_of_solutions_l552_552488

/-- Define the cubic polynomial whose solutions we need -/
def cubic_polynomial (x : ℝ) : ℝ := x^3 + 3 * x^2 - 14 * x

/-- Define the quadratic polynomial from factorizing the cubic polynomial -/
def quadratic_polynomial (x : ℝ) : ℝ := x^2 + 3 * x - 14

/-- State the problem about the mean of the solutions of the cubic equation -/
theorem mean_of_solutions : 
  let x1 := (0 : ℝ),
      x2 := (-3 + Real.sqrt 65) / 2,
      x3 := (-3 - Real.sqrt 65) / 2
  in (x1 + x2 + x3) / 3 = -1 :=
begin
  sorry -- the proof will go here
end

end mean_of_solutions_l552_552488


namespace functional_eq_unique_solution_l552_552054

theorem functional_eq_unique_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end functional_eq_unique_solution_l552_552054


namespace max_gold_coins_l552_552351

theorem max_gold_coins
  (n : ℕ)
  (h1 : n < 150)
  (h2 : n % 15 = 3) :
  n ≤ 138 :=
begin
  sorry
end

end max_gold_coins_l552_552351


namespace simplify_expression_l552_552666

-- Define the complex numbers and conditions
def ω : ℂ := (-1 + complex.I * real.sqrt 3) / 2
def ω_conj : ℂ := (-1 - complex.I * real.sqrt 3) / 2

-- Conditions
axiom ω_is_root_of_unity : ω^3 = 1
axiom ω_conj_is_root_of_unity : ω_conj^3 = 1

-- Theorem statement
theorem simplify_expression : ω^12 + ω_conj^12 = 2 := by
  sorry

end simplify_expression_l552_552666


namespace jerry_had_six_pancakes_l552_552191

theorem jerry_had_six_pancakes :
  ∀ (P : ℕ),
    (120 * P) + 200 + 200 = 1120 → 
    P = 6 :=
by
  intro P
  intro h
  have h1 : 120 * P + 400 = 1120 := by
    simp [←h]
  have h2 : 120 * P = 720 := by
    linarith
  have h3 : P = 6 := by
    exact eq_of_mul_eq_mul_left (nat.zero_lt_succ 119) h2
  exact h3

end jerry_had_six_pancakes_l552_552191


namespace theta_in_second_quadrant_l552_552074

-- Define the problem conditions 
variable {θ : ℝ} (h1 : Real.sin θ > 0) (h2 : Real.tan θ < 0)

-- Define a function for quadrants
def inSecondQuadrant (θ : ℝ) : Prop :=
  θ > π / 2 ∧ θ < π

theorem theta_in_second_quadrant : inSecondQuadrant θ :=
by
  sorry

end theta_in_second_quadrant_l552_552074


namespace range_of_a_l552_552552

noncomputable def f (x : ℝ) : ℝ := 2 * x - 1
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -2 * x + m
def h (x : ℝ) : ℝ :=
  if x ≤ -2 then 3 * x
  else if x ≤ 1 then 4 - x
  else 3

theorem range_of_a (m : ℝ) (a : ℝ) :
  (∀ x, (f x) - (1 / 2) * (g x m) > 0) ∧ (∀ x, x ≥ 1 → x = 2) ↔ a ∈ set.Ici 3 :=
sorry

end range_of_a_l552_552552


namespace ellipse_equation_range_of_m_l552_552894

-- Definition and theorem to find the equation of the given ellipse.
theorem ellipse_equation (h₁ : ∃ foci, foci = (c, 0)) 
                        (h₂ : center = (0, 0)) 
                        (h₃ : eccentricity = √3 / 2) 
                        (h₄ : passes_through_origin = (4, 1)) :
  ∃ a b : ℝ, a^2 = 20 ∧ b^2 = 5 ∧ equation_of_ellipse = (x^2 / a^2 + y^2 / b^2 = 1) := 
sorry

-- Definition and theorem to find the range of values for m.
theorem range_of_m (ellipse_eq : equation_of_ellipse = (x^2 / 20 + y^2 / 5 = 1)) :
  ∀ (m : ℝ), (y = x + m) intersects the ellipse_at_two_distinct_points ↔ -5 < m ∧ m < 5 := 
sorry

end ellipse_equation_range_of_m_l552_552894


namespace polynomial_remainder_l552_552841

noncomputable def remainder_division (f g : ℚ[X]) : ℚ[X] :=
  f % g

theorem polynomial_remainder : 
  let f := (X^4 : ℚ[X])
  let g := (X^2 + 3 * X + 2 : ℚ[X])
  remainder_division f g = -15 * X - 14 :=
by
  sorry

end polynomial_remainder_l552_552841


namespace ab_greater_than_a_plus_b_l552_552261

theorem ab_greater_than_a_plus_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a - b = a / b) : ab > a + b :=
sorry

end ab_greater_than_a_plus_b_l552_552261


namespace find_value_of_N_l552_552064

theorem find_value_of_N :
  (2 * ((3.6 * 0.48 * 2.5) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002) :=
by {
  sorry
}

end find_value_of_N_l552_552064


namespace hypercube_line_segment_condition_l552_552450

theorem hypercube_line_segment_condition (d : ℕ) (A B : Fin d → Fin 2) : 
  (∃ a b : Fin d → Fin 2, a ≠ b ∧ ∀ P : Fin d → Fin 2, P ∈ (set.range (λ λ : ℝ, λ • a + (1 - λ) • b)) ↔ P = a ∨ P = b) :=
  sorry

end hypercube_line_segment_condition_l552_552450


namespace slope_of_chord_l552_552812

def ellipse_eq (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 9 = 1

def midpoint (x1 x2 y1 y2 : ℝ) (M : ℝ × ℝ) : Prop :=
  (x1 + x2) / 2 = M.1 ∧ (y1 + y2) / 2 = M.2

theorem slope_of_chord (x1 x2 y1 y2 : ℝ) (M : ℝ × ℝ) 
  (hM : M = (1, 2)) 
  (hEllipse1 : ellipse_eq x1 y1)
  (hEllipse2 : ellipse_eq x2 y2)
  (hMidpoint : midpoint x1 x2 y1 y2 M) : 
  (y1 - y2) / (x1 - x2) = -9 / 32 := by
  sorry

end slope_of_chord_l552_552812


namespace negation_correct_l552_552298

def original_statement (a : ℝ) : Prop :=
  a > 0 → a^2 > 0

def negated_statement (a : ℝ) : Prop :=
  a ≤ 0 → a^2 ≤ 0

theorem negation_correct (a : ℝ) : ¬ (original_statement a) ↔ negated_statement a :=
by
  sorry

end negation_correct_l552_552298


namespace sarah_wins_games_l552_552651

variable (total_games : ℕ)
variable (tied_games : ℕ)
variable (total_money_lost : ℤ)

theorem sarah_wins_games
  (h_total_games : total_games = 100)
  (h_tied_games : tied_games = 40)
  (h_total_money_lost : total_money_lost = -30) :
  let won_games := total_games - tied_games - (total_money_lost / -2) in
  won_games = 30 :=
by
  sorry

end sarah_wins_games_l552_552651


namespace least_possible_integer_l552_552383

theorem least_possible_integer :
  ∃ N : ℕ,
    (∀ k, 1 ≤ k ∧ k ≤ 30 → k ≠ 24 → k ≠ 25 → N % k = 0) ∧
    (N % 24 ≠ 0) ∧
    (N % 25 ≠ 0) ∧
    N = 659375723440 :=
by
  sorry

end least_possible_integer_l552_552383


namespace percentage_of_students_owning_only_cats_l552_552942

theorem percentage_of_students_owning_only_cats
  (total_students : ℕ) (students_owning_dogs : ℕ) (students_owning_cats : ℕ) (students_owning_both : ℕ)
  (h1 : total_students = 500) (h2 : students_owning_dogs = 200) (h3 : students_owning_cats = 100) (h4 : students_owning_both = 50) :
  ((students_owning_cats - students_owning_both) * 100 / total_students) = 10 :=
by
  -- Placeholder for proof
  sorry

end percentage_of_students_owning_only_cats_l552_552942


namespace die_face_opposite_seven_l552_552287

theorem die_face_opposite_seven (die_faces : Set ℕ) (face_op : ℕ → ℕ) :
  die_faces = {6, 7, 8, 9, 10, 11} →
  (∃ f1 f2 f3 f4, f1 ≠ 7 ∧ f2 ≠ 7 ∧ f3 ≠ 7 ∧ f4 ≠ 7 ∧ f1 + f2 + f3 + f4 = 33) →
  (∃ f5 f6 f7 f8, f5 ≠ 7 ∧ f6 ≠ 7 ∧ f7 ≠ 7 ∧ f8 ≠ 7 ∧ f5 + f6 + f7 + f8 = 35) →
  (face_op 7 = 9 ∨ face_op 7 = 11) :=
begin
  intros die_faces_eq roll1 roll2,
  have h_total : 6 + 7 + 8 + 9 + 10 + 11 = 51 := by norm_num,
  have roll1_faces_sum : ∃ f1 f2 f3 f4, f1 ≠ 7 ∧ f2 ≠ 7 ∧ f3 ≠ 7 ∧ f4 ≠ 7 ∧ f1 + f2 + f3 + f4 = 33 := roll1,
  have roll2_faces_sum : ∃ f5 f6 f7 f8, f5 ≠ 7 ∧ f6 ≠ 7 ∧ f7 ≠ 7 ∧ f8 ≠ 7 ∧ f5 + f6 + f7 + f8 = 35 := roll2,
  sorry
end

end die_face_opposite_seven_l552_552287


namespace num_nat_numbers_with_remainder_17_l552_552475

theorem num_nat_numbers_with_remainder_17 (N : ℕ) :
  (2017 % N = 17 ∧ N > 17) → 
  ({N | 2017 % N = 17 ∧ N > 17}.toFinset.card = 13) := 
by
  sorry

end num_nat_numbers_with_remainder_17_l552_552475


namespace range_of_x_l552_552124

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x - 1

theorem range_of_x (x : ℝ) (h : f (x^2 - 4) < 2) : 
  (-Real.sqrt 5 < x ∧ x < -2) ∨ (2 < x ∧ x < Real.sqrt 5) :=
sorry

end range_of_x_l552_552124


namespace area_of_triangle_VAB_l552_552778

-- Definitions for points V, F, A, B
variables (V F A B : Point)

-- Conditions
def VF := 10
def chord_length_AB := 100
def passes_through_F := on_line F A B   -- Each point is collinear and passes through F

-- Area of triangle VAB
def area_triangle_VAB : ℝ :=
  100 * real.sqrt 10

-- Theorem stating the condition implies the correct area
theorem area_of_triangle_VAB 
  (h1 : distance V F = VF)
  (h2 : chord_length V F A B = chord_length_AB)
  (h3 : passes_through_F) : 
  area V A B = area_triangle_VAB :=
sorry

end area_of_triangle_VAB_l552_552778


namespace magnitude_comparison_l552_552413

theorem magnitude_comparison :
  (2 < -(+5)) = false ∧
  (-1 > -0.01) = false ∧
  (| -3 | < | +3 |) = false ∧
  (-(-5) > +(-7)) = true :=
by
  sorry

end magnitude_comparison_l552_552413


namespace Ivanov_family_total_insurance_cost_l552_552688

def overall_insurance_cost
(totalApartmentCost : ℕ)
(loanAmount : ℕ)
(interestRate : ℚ)
(propertyInsuranceRate : ℚ)
(titleInsuranceRate : ℚ)
(lifeInsuranceRateFemale : ℚ)
(lifeInsuranceRateMale : ℚ)
(contributionFemale : ℚ)
(contributionMale : ℚ) : ℚ :=
let totalLoanAmount := loanAmount * (1 + interestRate),
    propertyInsuranceCost := totalLoanAmount * propertyInsuranceRate,
    titleInsuranceCost := totalLoanAmount * titleInsuranceRate,
    femaleInsuranceCost := totalLoanAmount * contributionFemale * lifeInsuranceRateFemale,
    maleInsuranceCost := totalLoanAmount * contributionMale * lifeInsuranceRateMale in
propertyInsuranceCost + titleInsuranceCost + femaleInsuranceCost + maleInsuranceCost

theorem Ivanov_family_total_insurance_cost : 
overall_insurance_cost 
7000000   -- totalApartmentCost
4000000   -- loanAmount
0.101    -- interestRate
0.0009   -- propertyInsuranceRate
0.0027  -- titleInsuranceRate
0.0017  -- lifeInsuranceRateFemale
0.0019  -- lifeInsuranceRateMale
0.2     -- contributionFemale
0.8     -- contributionMale = 24045.84 :=
by 
  -- Proof steps would go here
  sorry

end Ivanov_family_total_insurance_cost_l552_552688


namespace pq_or_l552_552646

-- Proposition p: For all x ∈ ℝ, 3^x > x.
def prop_p : Prop := ∀ x : ℝ, 3^x > x

-- Proposition q: The negation of (If y = f(x - 1) is an odd function, the graph of y = f(x) is symmetric about (1, 0)).
def prop_q : Prop := ¬ (∀ f : ℝ → ℝ, (∀ x : ℝ, f(x - 1) = -f(-x + 1)) → (∀ x : ℝ, f(x) = f(2 - x)))

-- Given conditions
axiom p_true : prop_p
axiom q_false : prop_q

-- Prove p ∨ q
theorem pq_or : prop_p ∨ prop_q :=
by 
  exact Or.inl p_true

end pq_or_l552_552646


namespace sin_alpha_through_point_l552_552542

theorem sin_alpha_through_point (m : ℝ) :
  (m ≠ 0) → (∃ α : ℝ, (m * tan α = 9 * 4) ∧ (sin α = 3 / 5)) → (sin α = 3 / 5) :=
by
  intros h1 h2
  cases h2 with α h3
  exact
    have h : tan α = 3 / 4 from by
      rw [← div_eq_mul_inv, ← mul_div_assoc, ← mul_div_mul_left (by linarith : (4:ℝ) ≠ 0)]
      exact eq_div_of_mul_eq _ _ h3.1
    (((Real.tan_eq_sin_div_cos α).trans h).symm ▸ h3.2)

sorry -- Proof will be filled out


end sin_alpha_through_point_l552_552542


namespace max_non_dominating_sets_l552_552033

theorem max_non_dominating_sets :
  let total_tuples := 2017^100 in
  let less_than_2017_tuples := 2016^100 in
  total_tuples - less_than_2017_tuples = 2017^100 - 2016^100 :=
begin
  sorry
end

end max_non_dominating_sets_l552_552033


namespace total_cost_of_items_l552_552361

noncomputable theory

def cost_of_mangos_kg : ℝ := sorry
def cost_of_rice_kg : ℝ := sorry
def cost_of_flour_kg : ℝ := 20.50

axiom condition1 : 10 * cost_of_mangos_kg = 24 * cost_of_rice_kg
axiom condition2 : 6 * cost_of_flour_kg = 2 * cost_of_rice_kg

theorem total_cost_of_items :
  let total_cost := 4 * cost_of_mangos_kg + 3 * cost_of_rice_kg + 5 * cost_of_flour_kg
  in total_cost = 877.40 :=
by
  sorry

end total_cost_of_items_l552_552361


namespace find_omega_and_max_value_of_f_l552_552513

-- Definition of the function
def f (x : ℝ) (ω : ℝ) := 2 * Real.sin (2 * ω * x)

-- Given conditions and what needs to be proved
theorem find_omega_and_max_value_of_f :
  (∀ x, (f x ω).periodic = π) → ω = 1 ∧ (∀ x ∈ Set.Icc (π / 6) (π / 3), f x ω ≤ 2) :=
begin
  sorry
end

end find_omega_and_max_value_of_f_l552_552513


namespace number_of_N_satisfying_condition_l552_552469

def is_solution (N : ℕ) : Prop :=
  2017 % N = 17

theorem number_of_N_satisfying_condition : 
  {N : ℕ | is_solution N}.to_finset.card = 13 :=
sorry

end number_of_N_satisfying_condition_l552_552469


namespace verify_condition_find_a_b_range_of_m_l552_552090

variable (a b m : ℝ)

-- (1) Verify that a = -2 and b = -8 satisfy the condition.
theorem verify_condition (x : ℝ) (h : |x^2 + (-2) * x + (-8)| ≤ |2 * x^2 - 4 * x - 16|) : a = -2 ∧ b = -8 := by
  sorry
  
-- (2) Find all real numbers a and b that satisfy the condition.
theorem find_a_b (h : ∀ x : ℝ, |x^2 + a * x + b| ≤ |2 * x^2 - 4 * x - 16|) : a = -2 ∧ b = -8 := by
  sorry
  
-- (3) Determine the range of m for all x > 2.
theorem range_of_m (h : ∀ x > 2, x^2 + a * x + b ≥ (m + 2) * x - m - 15) : m ∈ set.Iic (-11) := by
  sorry

end verify_condition_find_a_b_range_of_m_l552_552090


namespace simplify_complex_expr_l552_552655

noncomputable def z1 : ℂ := (-1 + complex.I * real.sqrt 3) / 2
noncomputable def z2 : ℂ := (-1 - complex.I * real.sqrt 3) / 2

theorem simplify_complex_expr :
  z1^12 + z2^12 = 2 := 
  sorry

end simplify_complex_expr_l552_552655


namespace solveAdultsMonday_l552_552635

def numAdultsMonday (A : ℕ) : Prop :=
  let childrenMondayCost := 7 * 3
  let childrenTuesdayCost := 4 * 3
  let adultsTuesdayCost := 2 * 4
  let totalChildrenCost := childrenMondayCost + childrenTuesdayCost
  let totalAdultsCost := A * 4 + adultsTuesdayCost
  let totalRevenue := totalChildrenCost + totalAdultsCost
  totalRevenue = 61

theorem solveAdultsMonday : numAdultsMonday 5 := 
  by 
    -- Proof goes here
    sorry

end solveAdultsMonday_l552_552635


namespace find_x_for_gx_eq_5_l552_552625

def g (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 2 else 3 * x - 4

theorem find_x_for_gx_eq_5 (x : ℝ) : g x = 5 ↔ x = 3 :=
by sorry

end find_x_for_gx_eq_5_l552_552625


namespace set_inter_complement_l552_552558

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}

theorem set_inter_complement :
  A ∩ (U \ B) = {1, 3} :=
by
  sorry

end set_inter_complement_l552_552558


namespace find_a3_l552_552950

-- Definition of terms based on geometric sequence properties and conditions
variables {a_1 q : ℝ} (a : ℕ → ℝ)

-- Conditions
def geo_seq (a : ℕ → ℝ) := ∀ n, a n = a_1 * q^n

def condition1 (a : ℕ → ℝ) := a 2 = 1
def condition2 (a : ℕ → ℝ) := a 8 = a 6 + 6 * a 4

-- Problem statement
theorem find_a3 (pos_q : q > 0) (geo_seq a) (cond1 : condition1 a) (cond2 : condition2 a) : a 3 = sqrt 3 := 
sorry

end find_a3_l552_552950


namespace max_points_colored_segments_correct_l552_552515

noncomputable def max_points_colored_segments (points : Finset (ℝ × ℝ)) : ℕ :=
  if h : ∃ p, ∀ q₁ q₂ q₃ ∈ points, p ≠ q₁ ∧ p ≠ q₂ ∧ p ≠ q₃ ∧ 
            (∃ c₁ c₂ c₃, c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃ ∧ 
            (segment_color p q₁ = c₁ ∧ segment_color p q₂ = c₂ ∧ segment_color p q₃ = c₃)) then
    4
  else 
    0

theorem max_points_colored_segments_correct :
  ∀ (points : Finset (ℝ × ℝ)),
  (∀ p₁ p₂ p₃ ∈ points, 
   (p₁.1 * (p₂.2 - p₃.2) + p₂.1 * (p₃.2 - p₁.2) + p₃.1 * (p₁.2 - p₂.2) ≠ 0)) →
  max_points_colored_segments points = 4 := 
sorry

end max_points_colored_segments_correct_l552_552515


namespace binomial_theorem_problem_l552_552923

theorem binomial_theorem_problem {n : ℕ} 
  (h : ∑ k in finset.range n, (3^k) * (nat.choose n (k+1)) = 85) : 
  n = 4 := 
sorry

end binomial_theorem_problem_l552_552923


namespace unique_k_for_equal_power_l552_552056

theorem unique_k_for_equal_power (k : ℕ) (hk : 0 < k) (h : ∃ m n : ℕ, n > 1 ∧ (3 ^ k + 5 ^ k = m ^ n)) : k = 1 :=
by
  sorry

end unique_k_for_equal_power_l552_552056


namespace count_prime_sums_is_two_l552_552312

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def prime_numbers : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def prime_sums : List ℕ := 
  let sums := foldl (λ acc p, acc ++ [acc.head! + p]) [3] prime_numbers
  sums.take 15

def count_primes (l : List ℕ) : ℕ :=
  l.countp is_prime

theorem count_prime_sums_is_two : count_primes prime_sums = 2 := 
  by
    sorry

end count_prime_sums_is_two_l552_552312


namespace simplify_expression_l552_552213

variables {x p q r : ℝ}

theorem simplify_expression (h1 : p ≠ q) (h2 : p ≠ r) (h3 : q ≠ r) :
   ( (x + p)^4 / ((p - q) * (p - r)) + (x + q)^4 / ((q - p) * (q - r)) + (x + r)^4 / ((r - p) * (r - q)) 
   ) = p + q + r + 4 * x :=
sorry

end simplify_expression_l552_552213


namespace select_eight_players_from_eighteen_l552_552518

theorem select_eight_players_from_eighteen :
  nat.choose 18 8 = 43758 := by
  sorry

end select_eight_players_from_eighteen_l552_552518


namespace total_questions_attempted_l552_552959

theorem total_questions_attempted (C W T : ℕ) 
    (hC : C = 36) 
    (hScore : 120 = (4 * C) - W) 
    (hT : T = C + W) : 
    T = 60 := 
by 
  sorry

end total_questions_attempted_l552_552959


namespace orange_pill_cost_l552_552410

theorem orange_pill_cost :
  ∃ y : ℝ, (21 * (y + (y - 2)) = 735 ∧ y = 18.5) :=
by
  use 18.5
  split
  calc
    21 * (18.5 + (18.5 - 2)) = 21 * 35 : by norm_num
    ... = 735 : by norm_num
  sorry

end orange_pill_cost_l552_552410


namespace asymptotes_y_eq_one_div_x_plus_one_piecewise_linear_y_eq_abs_x_plus_one_plus_2_abs_x_minus_three_continuity_piecewise_y_eq_piecewise_quadratic_piecewise_y_eq_two_minus_abs_x_minus_x_sq_l552_552674

-- 1. Prove vertical and horizontal asymptotes for y = 1 / (x + 1)
theorem asymptotes_y_eq_one_div_x_plus_one : 
  (∀ x : ℝ, (y = 1 / (x + 1)) → 
    (x ≠ -1 → has_lim_at y 0) ∧ 
    (x = -1 → ¬ is_finite y)) :=
sorry

-- 2. Prove piecewise linear nature of y = |x + 1| + 2|x - 3|
theorem piecewise_linear_y_eq_abs_x_plus_one_plus_2_abs_x_minus_three :
  (∀ x : ℝ, (y = |x + 1| + 2 * |x - 3|) → 
    (x < -1 → y = -3 * x + 5) ∧ 
    (-1 ≤ x ∧ x < 3 → y = -x + 7) ∧ 
    (x ≥ 3 → y = 3 * x - 5)) :=
sorry

-- 3. Prove continuity and piecewise nature of y = if x >= 0 then x^2 + x else x^2 - x
theorem continuity_piecewise_y_eq_piecewise_quadratic :
  (∀ x : ℝ, (y = if x >= 0 then x^2 + x else x^2 - x) → 
    (continuous_at y 0) ∧ 
    (x >= 0 → y = x^2 + x) ∧ 
    (x < 0 → y = x^2 - x)) :=
sorry

-- 4. Prove piecewise nature of y = 2 - |x - x^2|
theorem piecewise_y_eq_two_minus_abs_x_minus_x_sq :
  (∀ x : ℝ, (y = 2 - |x - x^2|) → 
    ((0 ≤ x ∧ x ≤ 1 → y = 2 - x + x^2) ∧ 
    (x < 0 ∨ x > 1 → y = 2 + x - x^2))) :=
sorry

end asymptotes_y_eq_one_div_x_plus_one_piecewise_linear_y_eq_abs_x_plus_one_plus_2_abs_x_minus_three_continuity_piecewise_y_eq_piecewise_quadratic_piecewise_y_eq_two_minus_abs_x_minus_x_sq_l552_552674


namespace triangle_ABC_equilateral_l552_552599

theorem triangle_ABC_equilateral
  (A B C E F P Q M N : Type)
  [triangle : triangle A B C]
  [alt_BE : is_altitude B E AC]
  [alt_CF : is_altitude C F AB]
  [perp_FP_BC : is_perpendicular FP BC P]
  [perp_FQ_AC : is_perpendicular FQ AC Q]
  [perp_EM_BC : is_perpendicular EM BC M]
  [perp_EN_AB : is_perpendicular EN AB N]
  (cond1 : FP + FQ = CF)
  (cond2 : EM + EN = BE) :
  is_equilateral_triangle A B C :=
begin
  sorry
end

end triangle_ABC_equilateral_l552_552599


namespace original_cost_of_statue_l552_552603

theorem original_cost_of_statue (sale_price : ℝ) (profit_percent : ℝ) (original_cost : ℝ) 
  (h1 : sale_price = 620) 
  (h2 : profit_percent = 0.25) 
  (h3 : sale_price = (1 + profit_percent) * original_cost) : 
  original_cost = 496 :=
by
  sorry

end original_cost_of_statue_l552_552603


namespace number_of_zeros_in_square_of_9999_l552_552433

theorem number_of_zeros_in_square_of_9999 : 
  let x := 9999
  in (num_zeros_in_decimal_expansion (x * x) = 3) := by
  sorry

noncomputable def num_zeros_in_decimal_expansion (n : ℕ) : ℕ :=
  let s := n.to_string
  s.foldr (λ d acc, if d = '0' then acc + 1 else acc) 0

end number_of_zeros_in_square_of_9999_l552_552433


namespace problem_statement_l552_552114

-- Define the function f
def f (x : ℝ) : ℝ := 1 - 3 * (x - 1) + 3 * (x - 1) ^ 2 - (x - 1) ^ 3

-- The main statement
theorem problem_statement : (∃ y, f(y) = 8 ∧ y + f(1) = 1) :=
sorry

end problem_statement_l552_552114


namespace arithmetic_sequence_a10_l552_552877

theorem arithmetic_sequence_a10 (d a_1 : ℕ) : 
  (∀ n, a_1 + (n - 1) * d) = 19 :=
by
  sorry

end arithmetic_sequence_a10_l552_552877


namespace ordering_of_numbers_l552_552711

noncomputable def a : ℝ := 4 ^ 0.2
noncomputable def b : ℝ := 3 ^ 0.4
noncomputable def c : ℝ := Real.log 0.5 / Real.log 0.4

theorem ordering_of_numbers : c < a ∧ a < b :=
by
  sorry

end ordering_of_numbers_l552_552711


namespace T_9_correct_l552_552700

def a_n (n : ℕ) : ℕ := 2 * n

def S_n (n : ℕ) : ℕ := n * (n + 1)

def term (n : ℕ) : ℚ := (a_n (n + 1) : ℚ) / (S_n n * S_n (n + 1))

def sum_of_first_nine_terms : ℚ := Σ' i in Finset.range 9, term i

theorem T_9_correct : sum_of_first_nine_terms = 27 / 55 := sorry

end T_9_correct_l552_552700


namespace five_by_five_circumference_l552_552642

def number_of_people_on_circumference (n : ℕ) := 
  if n ≥ 2 then 4 * n - 4 else n

theorem five_by_five_circumference: number_of_people_on_circumference 5 = 16 := 
by { unfold number_of_people_on_circumference, norm_num, }

#print five_by_five_circumference  -- only to ensure the statement is registered

end five_by_five_circumference_l552_552642


namespace proof_problem_l552_552896

noncomputable def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

noncomputable def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g x + g (-x) = 0

theorem proof_problem (f : ℝ → ℝ)
  (h1 : is_even_function f)
  (h2 : is_odd_function (λ x, f(2 * x + 1) - 1))
  : f 1 = 1 ∧ (∀ x, f (x + 4) = f x) ∧ 
    (∀ x, f x = f (-x + 12))
:= by
  sorry

end proof_problem_l552_552896


namespace number_of_students_l552_552692

theorem number_of_students (avg_age_students : ℕ) (teacher_age : ℕ) (new_avg_age : ℕ) (n : ℕ) (T : ℕ) 
    (h1 : avg_age_students = 10) (h2 : teacher_age = 26) (h3 : new_avg_age = 11)
    (h4 : T = n * avg_age_students) 
    (h5 : (T + teacher_age) / (n + 1) = new_avg_age) : n = 15 :=
by
  -- Proof should go here
  sorry

end number_of_students_l552_552692


namespace reflected_light_ray_equation_l552_552102

theorem reflected_light_ray_equation :
  (∀ P : ℝ × ℝ, ∃ M : ℝ × ℝ, (M.1 = P.2 ∧ M.2 = P.1) ∧ M.2 = 2 * M.1 + 1) →
  (∀ Q : ℝ × ℝ, Q.2 = Q.1) →
  (∀ R : ℝ × ℝ, (R.1 - 2 * R.2 - 1 = 0) := sorry

end reflected_light_ray_equation_l552_552102


namespace race_dead_heat_l552_552355

theorem race_dead_heat (va vb D : ℝ) (hva_vb : va = (15 / 16) * vb) (dist_a : D = D) (dist_b : D = (15 / 16) * D) (race_finish : D / va = (15 / 16) * D / vb) :
  va / vb = 15 / 16 :=
by sorry

end race_dead_heat_l552_552355


namespace unit_cost_first_purchase_correct_max_discounted_units_correct_l552_552398

noncomputable def unit_cost_first_purchase : ℝ :=
  let x := 2400
  in x

#check unit_cost_first_purchase

theorem unit_cost_first_purchase_correct :
  ∃ x : ℝ, x = 2400 ∧ (24000 / x * 2 = 52000 / (x + 200)) :=
begin
  use 2400,
  split,
  { refl },
  { simp,
    norm_num,
    linarith, }
end

noncomputable def max_discounted_units : ℝ :=
  let y := 8
  in y

#check max_discounted_units

theorem max_discounted_units_correct :
  ∃ y : ℝ, y ≤ 8 ∧ 
           (3000 * (24000 / 2400) + 
           (3000 + 200) * 0.95 * y + 
           (3000 + 200) * ((52000 / (2400 + 200)) - y) 
           ≥ (24000 + 52000) * (1 + 0.22)) :=
begin
  use 8,
  split,
  { linarith },
  { rw [mul_comm (3000 + 200), mul_comm 0.95 y],
    norm_num,
    linarith }
end

end unit_cost_first_purchase_correct_max_discounted_units_correct_l552_552398


namespace quadrilateral_proof_l552_552870

variable (A B C D : Type)
variable [metric_space A] [has_dist A] [metric_space B] [has_dist B] [metric_space C] [has_dist C] [metric_space D] [has_dist D]

def u (A D : A) : ℝ := dist A D ^ 2
def v (B D : B) : ℝ := dist B D ^ 2
def w (C D : C) : ℝ := dist C D ^ 2

def U (B C D : B) : ℝ := dist B D ^ 2 + dist C D ^ 2 - dist B C ^ 2
def V (A C D : C) : ℝ := dist A D ^ 2 + dist C D ^ 2 - dist A C ^ 2
def W (A B D : D) : ℝ := dist A D ^ 2 + dist B D ^ 2 - dist A B ^ 2

theorem quadrilateral_proof (A B C D : Type)
  [metric_space A] [has_dist A]
  [metric_space B] [has_dist B]
  [metric_space C] [has_dist C]
  [metric_space D] [has_dist D] :
  let u := dist A D ^ 2 in
  let v := dist B D ^ 2 in
  let w := dist C D ^ 2 in
  let U := dist B D ^ 2 + dist C D ^ 2 - dist B C ^ 2 in
  let V := dist A D ^ 2 + dist C D ^ 2 - dist A C ^ 2 in
  let W := dist A D ^ 2 + dist B D ^ 2 - dist A B ^ 2 in
  u * U^2 + v * V^2 + w * W^2 = U * V * W + 4 * u * v * w :=
sorry

end quadrilateral_proof_l552_552870


namespace mod_pow_sum_7_l552_552441

theorem mod_pow_sum_7 :
  (45 ^ 1234 + 27 ^ 1234) % 7 = 5 := by
  sorry

end mod_pow_sum_7_l552_552441


namespace Ivanov_family_total_insurance_cost_l552_552687

def overall_insurance_cost
(totalApartmentCost : ℕ)
(loanAmount : ℕ)
(interestRate : ℚ)
(propertyInsuranceRate : ℚ)
(titleInsuranceRate : ℚ)
(lifeInsuranceRateFemale : ℚ)
(lifeInsuranceRateMale : ℚ)
(contributionFemale : ℚ)
(contributionMale : ℚ) : ℚ :=
let totalLoanAmount := loanAmount * (1 + interestRate),
    propertyInsuranceCost := totalLoanAmount * propertyInsuranceRate,
    titleInsuranceCost := totalLoanAmount * titleInsuranceRate,
    femaleInsuranceCost := totalLoanAmount * contributionFemale * lifeInsuranceRateFemale,
    maleInsuranceCost := totalLoanAmount * contributionMale * lifeInsuranceRateMale in
propertyInsuranceCost + titleInsuranceCost + femaleInsuranceCost + maleInsuranceCost

theorem Ivanov_family_total_insurance_cost : 
overall_insurance_cost 
7000000   -- totalApartmentCost
4000000   -- loanAmount
0.101    -- interestRate
0.0009   -- propertyInsuranceRate
0.0027  -- titleInsuranceRate
0.0017  -- lifeInsuranceRateFemale
0.0019  -- lifeInsuranceRateMale
0.2     -- contributionFemale
0.8     -- contributionMale = 24045.84 :=
by 
  -- Proof steps would go here
  sorry

end Ivanov_family_total_insurance_cost_l552_552687


namespace ratio_of_price_l552_552732

-- Definitions from conditions
def original_price : ℝ := 3.00
def tom_pay_price : ℝ := 9.00

-- Theorem stating the ratio
theorem ratio_of_price : tom_pay_price / original_price = 3 := by
  sorry

end ratio_of_price_l552_552732


namespace sin_angle_F_l552_552174

theorem sin_angle_F (DE EF : ℝ) (h1 : DE = 8) (h2 : EF = 17) (D_right : ∠ D = 90) :
  sin (∠ F) = 8 / 17 :=
by
  sorry

end sin_angle_F_l552_552174


namespace calculate_A_l552_552755

noncomputable def A : ℝ := 
  ( ∏ n in Finset.range (2000 - 1000 + 1), (n + 1000)) / 
  ( ∏ n in Finset.range (2000 // 2 + 1), (2 * n - 1))

theorem calculate_A : A = 2^1000 :=
by
  sorry

end calculate_A_l552_552755


namespace number_of_valid_pairs_is_343_l552_552070

-- Define the given problem conditions
def given_number : Nat := 1003003001

-- Define the expression for LCM calculation
def LCM (x y : Nat) : Nat := (x * y) / (Nat.gcd x y)

-- Define the prime factorization of the given number
def is_prime_factorization_correct : Prop :=
  given_number = 7^3 * 11^3 * 13^3

-- Define x and y form as described
def is_valid_form (x y : Nat) : Prop :=
  ∃ (a b c d e f : ℕ), x = 7^a * 11^b * 13^c ∧ y = 7^d * 11^e * 13^f

-- Define the LCM condition for the ordered pairs
def meets_lcm_condition (x y : Nat) : Prop :=
  LCM x y = given_number

-- State the theorem to prove an equivalent problem
theorem number_of_valid_pairs_is_343 : is_prime_factorization_correct →
  (∃ (n : ℕ), n = 343 ∧ 
    (∀ (x y : ℕ), is_valid_form x y → meets_lcm_condition x y → x > 0 → y > 0 → True)
  ) :=
by
  intros h
  use 343
  sorry

end number_of_valid_pairs_is_343_l552_552070


namespace ratio_of_books_read_l552_552029

noncomputable def calc_ratio : ℕ :=
  let Amanda_books := 18 / 3 in
  let Kara_books := Amanda_books / 2 in
  let Patricia_books := 7 * Kara_books in
  let Taylor_books := (18 + Amanda_books + Kara_books + Patricia_books) / 4 in
  let total_books := 18 + Amanda_books + Kara_books + Patricia_books + Taylor_books in
  Taylor_books / total_books

theorem ratio_of_books_read :
  let Amanda_books := 18 / 3 in
  let Kara_books := Amanda_books / 2 in
  let Patricia_books := 7 * Kara_books in
  let Taylor_books := (18 + Amanda_books + Kara_books + Patricia_books) / 4 in
  let total_books := 18 + Amanda_books + Kara_books + Patricia_books + Taylor_books in
  (Taylor_books : ℚ) / total_books = 1 / 5 :=
by
  sorry

end ratio_of_books_read_l552_552029


namespace arithmetic_mean_18_27_45_l552_552025

theorem arithmetic_mean_18_27_45 : 
  (18 + 27 + 45) / 3 = 30 :=
by
  -- skipping proof
  sorry

end arithmetic_mean_18_27_45_l552_552025


namespace age_difference_l552_552364

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 18) : a - c = 18 :=
by
  sorry

end age_difference_l552_552364


namespace polar_curve_is_circle_l552_552698

theorem polar_curve_is_circle:
  ∀ θ ρ : ℝ, ρ = sin θ + 2 * cos θ → 
  ∃ k l r : ℝ, k ≠ 0 ∧ l ≠ 0 ∧ (ρ^2 = k * ρ * sin θ + l * ρ * cos θ) ∧ (k^2 + l^2 = r) :=
sorry

end polar_curve_is_circle_l552_552698


namespace triangle_area_max_eq_16_l552_552966

open Real

noncomputable def max_triangle_area (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ :=
  let OA := 8 * sin α
  let OB := 8 * cos α
  (1/2) * OA * OB

theorem triangle_area_max_eq_16 (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  max_triangle_area α hα ≤ 16 :=
by {
  have h1 : 0 < sin α ∧ sin α ≤ 1, from ⟨sin_pos_of_pos_of_lt_pi (hα.left) (lt_trans hα.right (by linarith)) (by linarith), sin_le_one α⟩,
  have h2 : 0 < cos α ∧ cos α ≤ 1, from ⟨cos_pos_of_neg_of_lt_pi_div_two (by linarith : α < π / 2) (by linarith : 0 < α), cos_le_one α⟩,
  
  calc
    max_triangle_area α hα = (1/2) * (8 * sin α) * (8 * cos α) : by rfl
    ... = 16 * (sin α * cos α) : by ring
    ... = 16 * (1/2 * sin (2 * α)) : by rw [← sin_double_angle]
    ... = 8 * sin (2 * α) : by ring
    ... ≤ 8 * 1 : mul_le_mul_of_nonneg_left (le_of_lt (sin_lt_one_of_lt_pi _)) (by norm_num)
    ... = 8 : by norm_num,
  linarith,
  rw two_mul,
  exact add_lt_add hα.left hα.right,
}

end triangle_area_max_eq_16_l552_552966


namespace number_of_persons_in_first_group_l552_552156

-- Define the amount of work done by one person in one day (unit work)
variable W : ℝ

-- Define the number of persons in the first group
variable P : ℝ

-- Define the first condition: P persons do 7 times the work in 7 days
axiom cond1 : P * 7 * W = 7

-- Define the second condition: 9 persons do 9 times the work in 7 days
axiom cond2 : 9 * 7 * W = 9

-- Theorem to prove that P = 9
theorem number_of_persons_in_first_group : P = 9 := by
  sorry

end number_of_persons_in_first_group_l552_552156


namespace tangent_lines_count_l552_552230

-- Define the conditions
def radius_c1 : ℝ := 4
def radius_c2 : ℝ := 5

-- Define the assertion
theorem tangent_lines_count :
  ∃ k : set ℕ, k = {0, 1, 2, 3, 4} :=
sorry

end tangent_lines_count_l552_552230


namespace total_valid_votes_l552_552360

theorem total_valid_votes (V : ℕ) (h1 : 0.70 * (V: ℝ) - 0.30 * (V: ℝ) = 184) : V = 460 :=
by sorry

end total_valid_votes_l552_552360


namespace problem_solution_l552_552528

noncomputable def sequence (n : ℕ) : ℕ := 2 * n + 1

def sum_first_n_terms (n : ℕ) : ℕ := n^2 + 2 * n

noncomputable def sum_first_n_exp_terms (n : ℕ) : ℤ := (8 * (4^n - 1)) / 3

theorem problem_solution (a : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℤ)
  (h₁ : a 3 = 7)
  (h₂ : S 3 = 15)
  (h₃ : ∀ n, a n = sequence n)
  (h₄ : ∀ n, S n = sum_first_n_terms n)
  (h₅ : ∀ n, T n = sum_first_n_exp_terms n) :
  (∃ a_n S_n T_n, 
    (∀ n, a_n n = 2 * n + 1) ∧ 
    (∀ n, S_n n = n^2 + 2 * n) ∧ 
    (∀ n, T_n n = (8 * (4^n - 1)) / 3)) :=
by {
  use [sequence, sum_first_n_terms, sum_first_n_exp_terms],
  split,
  intro n,
  exact h₃ n,
  split,
  intro n,
  exact h₄ n,
  intro n,
  exact h₅ n,
  sorry  -- Proof is omitted
}

end problem_solution_l552_552528


namespace alpha_is_beta_l552_552132

open Real

-- Define the parabola C: x^2 = 4y
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the line l: y = -1
def line_l (y : ℝ) : Prop := y = -1

-- Function to check if a point belongs to a given line
def point_on_line (P : ℝ × ℝ) (line : ℝ → Prop) : Prop := line P.2

-- Function to check perpendicularity of slopes k1 and k2
def perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

-- Given the points A and B on the parabola, the tangents PA and PB are given with slopes k_PA and k_PB respectively
def tangent_slope (x : ℝ) : ℝ := x / 2

-- Define the condition α (P is on l)
def alpha (P : ℝ × ℝ) : Prop := point_on_line P line_l

-- Define the condition β (PA is perpendicular to PB)
def beta (x1 x2 : ℝ) : Prop := perpendicular (tangent_slope x1) (tangent_slope x2)

-- Main statement asserting that condition α is necessary and sufficient for condition β
theorem alpha_is_beta (P : ℝ × ℝ) (x1 x2 : ℝ) :
  alpha P ↔ (parabola x1 P.2 ∧ parabola x2 P.2 ∧ beta x1 x2) :=
sorry

end alpha_is_beta_l552_552132


namespace number_of_valid_two_digit_numbers_l552_552922

-- Definitions of the problem
def digit_set := {0, 1, 2, 3}
def is_valid_two_digit_number (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100 ∧ (n / 10 ∈ digit_set) ∧ (n % 10 ∈ digit_set) ∧ (n / 10 ≠ 0)

-- The theorem we want to prove
theorem number_of_valid_two_digit_numbers : 
  (finset.filter is_valid_two_digit_number (finset.range 100)).card = 9 := 
by
  -- The actual proof is omitted
  sorry

end number_of_valid_two_digit_numbers_l552_552922


namespace find_m_given_a3_eq_40_l552_552076

theorem find_m_given_a3_eq_40 (m : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 - m * x) ^ 5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_3 = 40 →
  m = -1 := 
by 
  sorry

end find_m_given_a3_eq_40_l552_552076


namespace expected_value_correct_l552_552409

-- Define the probabilities
def prob_8 : ℚ := 3 / 8
def prob_other : ℚ := 5 / 56 -- Derived from the solution steps but using only given conditions explicitly.

-- Define the expected value calculation
def expected_value_die : ℚ :=
  (1 * prob_other) + (2 * prob_other) + (3 * prob_other) + (4 * prob_other) +
  (5 * prob_other) + (6 * prob_other) + (7 * prob_other) + (8 * prob_8)

-- The theorem to prove
theorem expected_value_correct : expected_value_die = 77 / 14 := by
  sorry

end expected_value_correct_l552_552409


namespace min_value_of_reciprocals_l552_552532

theorem min_value_of_reciprocals (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : real.sqrt 3 = real.geom_mean (3^a) (3^b)) : 
  ∃ (a b : ℝ), (a + b = 1) → (a > 0) ∧ (b > 0) ∧ ((1/a) + (1/b) = 4) :=
by 
  -- Placeholder for proof
  sorry

end min_value_of_reciprocals_l552_552532


namespace field_day_difference_l552_552181

theorem field_day_difference :
  let girls_class_4_1 := 12
  let boys_class_4_1 := 13
  let girls_class_4_2 := 15
  let boys_class_4_2 := 11
  let girls_class_5_1 := 9
  let boys_class_5_1 := 13
  let girls_class_5_2 := 10
  let boys_class_5_2 := 11
  let total_girls := girls_class_4_1 + girls_class_4_2 + girls_class_5_1 + girls_class_5_2
  let total_boys := boys_class_4_1 + boys_class_4_2 + boys_class_5_1 + boys_class_5_2
  total_boys - total_girls = 2 := by
  sorry

end field_day_difference_l552_552181


namespace lucas_sees_liam_duration_l552_552224

theorem lucas_sees_liam_duration :
  ∀ (lucas_speed liam_speed : ℝ) (initial_distance final_distance : ℝ),
  lucas_speed = 20 → liam_speed = 6 →
  initial_distance = 1 → final_distance = 1 →
  (let relative_speed := lucas_speed - liam_speed in
  let time_seeing := (initial_distance + final_distance) / relative_speed in
  time_seeing * 60 = 9) :=
by
  intros lucas_speed liam_speed initial_distance final_distance
  assume h₁ h₂ h₃ h₄
  let relative_speed := lucas_speed - liam_speed
  let time_seeing := (initial_distance + final_distance) / relative_speed
  have h₅ : relative_speed = 14 := sorry
  have h₆ : time_seeing = 2 / 14 := sorry
  have h₇ : time_seeing * 60 = 9 := sorry
  exact h₇

end lucas_sees_liam_duration_l552_552224


namespace possible_values_of_f1_l552_552613

theorem possible_values_of_f1 (f : ℝ → ℝ) (h : ∀ x y : ℝ, f(x + y) = 2 * f(x) * f(y)) : 
  f(1) = 0 ∨ f(1) = 1/2 :=
sorry

end possible_values_of_f1_l552_552613


namespace factor_and_sum_coeffs_l552_552052

noncomputable def sum_of_integer_coeffs_of_factorization (x y : ℤ) : ℤ :=
  let factors := ([(1 : ℤ), (-1 : ℤ), (5 : ℤ), (1 : ℤ), (6 : ℤ), (1 : ℤ), (1 : ℤ), (5 : ℤ), (-1 : ℤ), (6 : ℤ)])
  factors.sum

theorem factor_and_sum_coeffs (x y : ℤ) :
  (125 * (x^9:ℤ) - 216 * (y^9:ℤ) = (x - y) * (5 * x^2 + x * y + 6 * y^2) * (x + y) * (5 * x^2 - x * y + 6 * y^2))
  ∧ (sum_of_integer_coeffs_of_factorization x y = 24) :=
by
  sorry

end factor_and_sum_coeffs_l552_552052


namespace inequality_proof_l552_552215

theorem inequality_proof
  (a b c d : ℝ) (h0 : a ≥ 0) (h1 : b ≥ 0) (h2 : c ≥ 0) (h3 : d ≥ 0) (h4 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 1 / 3 :=
sorry

end inequality_proof_l552_552215


namespace problem_l552_552259

-- Define A, B coordinates and condition they lie on y = -x^2
def is_on_parabola (p : ℝ × ℝ) : Prop :=
  p.2 = -p.1^2

-- Define equilateral triangle condition
def is_equilateral (A B O : ℝ × ℝ) : Prop := 
  dist A B = dist A O ∧ dist A O = dist B O

-- Main theorem statement
theorem problem (A B : ℝ × ℝ) (O : ℝ × ℝ) :
  O = (0, 0) ∧ is_on_parabola A ∧ is_on_parabola B ∧ is_equilateral A B O ->
  (A = (sqrt 3, -3) ∧ B = (- sqrt 3, -3) ∧ dist A O = 2 * sqrt 3) :=
begin
  sorry
end

end problem_l552_552259


namespace minimum_cuboid_cubes_l552_552352

theorem minimum_cuboid_cubes (n : ℕ) (h : 0 < n) :
  let S := 128 * n in S > 2011 → n = 16 := 
by
  intro hS
  have hn : n > 2011 / 128 := by sorry -- calculation here 
  have hn_int : n = 16 := by sorry -- integer part and final step 
  exact hn_int

end minimum_cuboid_cubes_l552_552352


namespace poly_div_remainder_l552_552835

noncomputable def x : ℝ[X] := polynomial.X

def poly1 : ℝ[X] := x^4
def poly2 : ℝ[X] := x^2 + 3 * x + 2
def remainder : ℝ[X] := -18 * x - 16

theorem poly_div_remainder :
  poly1 % poly2 = remainder :=
sorry

end poly_div_remainder_l552_552835


namespace intersection_A_B_l552_552915

def A : Set ℝ := { x | Real.log x > 0 }

def B : Set ℝ := { x | Real.exp x < 3 }

theorem intersection_A_B :
  A ∩ B = { x | 1 < x ∧ x < Real.log 3 / Real.log 2 } :=
sorry

end intersection_A_B_l552_552915


namespace first_15_prime_sums_count_zero_l552_552305

/-- A function to check if a number is prime -/
def is_prime (n : ℕ) : Prop := nat.prime n

/-- Defining the first 15 sums of consecutive primes starting from 3 -/
def prime_sums : ℕ → ℕ
| 1     := 3
| (n+1) := prime_sums n + nat.prime (n+1)

noncomputable def count_prime_sums_up_to_15 : ℕ :=
(finset.range 15).count (λ n, is_prime (prime_sums (n+1)))

theorem first_15_prime_sums_count_zero : count_prime_sums_up_to_15 = 0 :=
by sorry

end first_15_prime_sums_count_zero_l552_552305


namespace sum_of_reciprocals_is_one_l552_552319

theorem sum_of_reciprocals_is_one (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (x : ℚ)) + (1 / (y : ℚ)) + (1 / (z : ℚ)) = 1 ↔ (x, y, z) = (2, 4, 4) ∨ 
                                                    (x, y, z) = (2, 3, 6) ∨ 
                                                    (x, y, z) = (3, 3, 3) :=
by 
  sorry

end sum_of_reciprocals_is_one_l552_552319


namespace limit_of_seq_l552_552365

open BigOperators

-- Define the sequence
def seq (n : ℕ) : ℝ := ((n + 4)! - (n + 2)!) / (n + 3)!

-- State the problem as a theorem to be proved
theorem limit_of_seq : filter.tendsto seq filter.at_top filter.at_top :=
sorry

end limit_of_seq_l552_552365


namespace log_expression_simplification_l552_552449

open Real

noncomputable def log_expr (a b c d x y z : ℝ) : ℝ :=
  log (a^2 / b) + log (b^2 / c) + log (c^2 / d) - log (a^2 * y * z / (d^2 * x))

theorem log_expression_simplification (a b c d x y z : ℝ) (h : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
(h4 : d ≠ 0) (h5 : x ≠ 0) (h6 : y ≠ 0) (h7 : z ≠ 0) :
  log_expr a b c d x y z = log (bdx / yz) :=
by
  -- Proof goes here
  sorry

end log_expression_simplification_l552_552449


namespace fraction_of_red_knights_are_magical_l552_552167

noncomputable def ratio_of_fraction (p q : ℚ) (k : ℚ) : ℚ := k * (p / (1 - p)) / q

theorem fraction_of_red_knights_are_magical :
  (frac_red : ℚ) 
  (frac_blue : ℚ) 
  (frac_magical : ℚ) 
  (ratio : ℚ)
  (total_knights : ℕ):
  frac_red = 3 / 8  ∧
  frac_blue = (1 - frac_red) ∧
  frac_magical = 1 / 4 ∧
  ratio = 1.5 →
  (magic_fraction_red : ℚ) (magic_fraction_blue : ℚ),
  magic_fraction_red = frac_magical * (total_knights * frac_blue) / (frac_red * total_knights * ratio + frac_blue * total_knights) →
  magic_fraction_red = 6 / 19 :=
begin
  intros frac_red frac_blue frac_magical ratio total_knights h_eqns magic_fraction_red magic_fraction_blue h_fraction_red,
  have eq1 : frac_red = 3 / 8 := by sorry, -- from h_eqns
  have eq2 : frac_blue = 5 / 8 := by sorry, -- from h_eqns
  have eq3 : frac_magical = 1 / 4 := by sorry, -- from h_eqns
  have eq4 : ratio = 1.5 := by sorry, -- from h_eqns
  have eq5 : magic_fraction_red = 6 / 19 := by sorry, -- from h_fraction_red and the provided conditions
  exact (eq5)
end

end fraction_of_red_knights_are_magical_l552_552167


namespace class_distances_l552_552038

theorem class_distances (x y z : ℕ) 
  (h1 : y = x + 8)
  (h2 : z = 3 * x)
  (h3 : x + y + z = 108) : 
  x = 20 ∧ y = 28 ∧ z = 60 := 
  by sorry

end class_distances_l552_552038


namespace find_x_l552_552053

theorem find_x :
  (∃ (x : ℝ), 2 * arctan (1/3) + arctan (1/5) + arctan (1/x) = π / 4 ∧ x = 10.5) :=
sorry

end find_x_l552_552053


namespace greatest_prime_factor_f36_l552_552852

def is_even (n : ℕ) : Prop := n % 2 = 0
def f (m : ℕ) : ℕ := ∏ i in Finset.filter is_even (Finset.range (m + 1)), i

lemma product_of_even_numbers (m : ℕ) (hm : is_even m) : 
  f(m) = ∏ i in Finset.filter is_even (Finset.range (m + 1)), i := rfl

theorem greatest_prime_factor_f36 : 
  Nat.greatest_prime_factor (f 36) = 17 := sorry

end greatest_prime_factor_f36_l552_552852


namespace max_val_of_m_if_p_true_range_of_m_if_one_prop_true_one_false_range_of_m_if_one_prop_false_one_true_l552_552212

variable {m x x0 : ℝ}

def proposition_p (m : ℝ) : Prop := ∀ x > -2, x + 49 / (x + 2) ≥ 6 * Real.sqrt 2 * m
def proposition_q (m : ℝ) : Prop := ∃ x0 : ℝ, x0 ^ 2 - m * x0 + 1 = 0

theorem max_val_of_m_if_p_true (h : proposition_p m) : m ≤ Real.sqrt 2 := by
  sorry

theorem range_of_m_if_one_prop_true_one_false (hp : proposition_p m) (hq : ¬ proposition_q m) : (-2 < m ∧ m ≤ Real.sqrt 2) ∨ (2 ≤ m) := by
  sorry

theorem range_of_m_if_one_prop_false_one_true (hp : ¬ proposition_p m) (hq : proposition_q m) : (m ≥ 2) := by
  sorry

end max_val_of_m_if_p_true_range_of_m_if_one_prop_true_one_false_range_of_m_if_one_prop_false_one_true_l552_552212


namespace sequence_sum_2022_l552_552519

theorem sequence_sum_2022 (n : ℕ) (h_n : n = 2022) 
    (x : ℕ → ℝ) (h0 : x 0 = 1 / n)
    (h_rec : ∀ k, 1 ≤ k ∧ k < n → x k = (1 / (n - k)) * ∑ i in finset.range k, x i): 
    (∑ i in finset.range n, x i) = 1 :=
by {
  sorry
}

end sequence_sum_2022_l552_552519


namespace compute_abs_diff_sum_l552_552745

theorem compute_abs_diff_sum : 
  let x := 12
  let y := 18
  |x - y| * (x + y) = 180 := 
by
  sorry

end compute_abs_diff_sum_l552_552745


namespace poly_div_remainder_l552_552836

noncomputable def x : ℝ[X] := polynomial.X

def poly1 : ℝ[X] := x^4
def poly2 : ℝ[X] := x^2 + 3 * x + 2
def remainder : ℝ[X] := -18 * x - 16

theorem poly_div_remainder :
  poly1 % poly2 = remainder :=
sorry

end poly_div_remainder_l552_552836


namespace sum_inscribed_angles_eq_720_l552_552779

open Real

-- Define the problem conditions
def isPentagonInscribed (P : Finset (Set (EuclideanSpace ℝ (Fin 2)))) : Prop :=
  ∃ A B C D E : EuclideanSpace ℝ (Fin 2),
    {A, B, C, D, E} ∈ P ∧ ∀ U V ∈ {A, B, C, D, E}, dist U V = dist (U + loci.circle_center) (V + loci.circle_center)

def inscribedAngles (P : Finset (Set (EuclideanSpace ℝ (Fin 2)))) (α β γ δ ε : ℝ) : Prop :=
  isPentagonInscribed P ∧
  α = ½ * (360 - arc P) ∧
  β = ½ * (360 - arc P) ∧
  γ = ½ * (360 - arc P) ∧
  δ = ½ * (360 - arc P) ∧
  ε = ½ * (360 - arc P)

-- State the theorem
theorem sum_inscribed_angles_eq_720 {P : Finset (Set (EuclideanSpace ℝ (Fin 2)))} {α β γ δ ε : ℝ}
    (h1 : isPentagonInscribed P) (h2 : inscribedAngles P α β γ δ ε) :
    α + β + γ + δ + ε = 720 :=
  by
  sorry

end sum_inscribed_angles_eq_720_l552_552779


namespace determine_by_median_l552_552725

-- Define the set of scores and conditions
variable (scores : List ℝ) (LittleRedScore : ℝ)

-- Condition: There are exactly 9 different scores
def nine_different_scores := scores.length = 9 ∧ scores.nodup

-- Condition: Little Red's score is in the list
def little_red_score := LittleRedScore ∈ scores

-- Theorem statement
theorem determine_by_median (h1 : nine_different_scores scores) (h2 : little_red_score scores LittleRedScore) :
  (∃ m, m = (List.nthLe (List.sort (· < ·) scores) 4 (by linarith [h1.left])) ∧
    (LittleRedScore < m ∨ LittleRedScore = m ∨ LittleRedScore > m)) :=
  sorry

end determine_by_median_l552_552725


namespace num_prime_sums_first_15_l552_552309

noncomputable def seq_sums_of_primes : ℕ → ℕ
| 0 => 3
| n+1 => seq_sums_of_primes n + Nat.prime (n+1+1)

theorem num_prime_sums_first_15 : 
  (Finset.filter (Nat.Prime) (Finset.range 15)).card = 3 :=
by
  sorry

end num_prime_sums_first_15_l552_552309


namespace apps_left_on_phone_l552_552815

-- Definitions for the given conditions
def initial_apps : ℕ := 15
def added_apps : ℕ := 71
def deleted_apps : ℕ := added_apps + 1

-- Proof statement
theorem apps_left_on_phone : initial_apps + added_apps - deleted_apps = 14 := by
  sorry

end apps_left_on_phone_l552_552815


namespace sum_of_squares_l552_552046

theorem sum_of_squares (b j s : ℕ) (h : b + j + s = 34) : b^2 + j^2 + s^2 = 406 :=
sorry

end sum_of_squares_l552_552046


namespace bobs_smallest_number_l552_552005

/-- Alice's number -/
def alice_number := 36

/-- Prime factors of Alice's number -/
def prime_factors_alice := {2, 3}

/-- Smallest prime factor governed by conditions for Bob -/
def smallest_prime := 2

/-- Bob's number is twice the prime factor -/
def bobs_number := 2 * smallest_prime

/-- Smallest possible number Bob could choose -/
theorem bobs_smallest_number : bobs_number = 4 :=
by
  -- Proof omitted
  sorry

end bobs_smallest_number_l552_552005


namespace age_ratio_l552_552314

theorem age_ratio (x : ℚ) (h1 : 6 * x - 4 = 3 * x + 4) : 
  let age_A := 6 * x;
  let age_B := 3 * x;
  let age_A_hence := age_A + 4;
  let age_B_ago := age_B - 4;
  ratio : ℚ := age_A_hence / age_B_ago
in ratio = 5 :=
by {
  let age_A := 6 * x;
  let age_B := 3 * x;
  let age_A_hence := age_A + 4;
  let age_B_ago := age_B - 4;
  have h : age_A_hence / age_B_ago = 5, sorry;
  exact h
}

end age_ratio_l552_552314


namespace student_A_clap_count_100_l552_552067

open Int

def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

def student_A_clap_count_up_to (n : ℕ) : ℕ :=
  (List.range n).countp (λ i => is_multiple_of_three (fib (5 * (i + 1) - 4)))

theorem student_A_clap_count_100 : student_A_clap_count_up_to 100 = 5 := 
by
  -- Proof omitted
  sorry

end student_A_clap_count_100_l552_552067


namespace unique_X_on_circle_l552_552077

-- Definitions
variable (C : Type) [circle C] (A B : point) (l : line) (X : point)

-- Specify that A and B are outside the circle C
axiom A_outside_C : ¬ (A ∈ C)
axiom B_outside_C : ¬ (B ∈ C)

-- We need to prove the existence of a unique X on the circle such that the chord is parallel to l
theorem unique_X_on_circle (C : Type) [circle C] (A B : point) (l : line) :
  ∃! (X : point), X ∈ C ∧ (chord_parallel_to_line (line_through A X) (line_through B X) l) := 
sorry

end unique_X_on_circle_l552_552077


namespace blue_pill_cost_l552_552190

theorem blue_pill_cost (y : ℕ) :
  -- Conditions
  (∀ t d : ℕ, t = 21 → 
     d = 14 → 
     (735 - d * 2 = t * ((2 * y) + (y + 2)) / t) →
     2 * y + (y + 2) = 35) →
  -- Conclusion
  y = 11 :=
by
  sorry

end blue_pill_cost_l552_552190


namespace ellipse_focus_x_axis_hyperbola_standard_form_l552_552845

theorem ellipse_focus_x_axis (x y : ℝ) :
  (∃ λ > 0, (2, -sqrt 3) ∈ λ⁻¹ • {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}) →
  ∃ x y : ℝ, x^2 / 8 + y^2 / 6 = 1 :=
by sorry

theorem hyperbola_standard_form (x y : ℝ) :
  (∃ a b : ℝ, 2 * a = 6 ∧ b / a = 1 / 3) →
  ∃ x y : ℝ, x^2 / 9 - y^2 = 1 :=
by sorry

end ellipse_focus_x_axis_hyperbola_standard_form_l552_552845


namespace triangle_altitude_length_l552_552971

variable (AB AC BC BA1 AA1 : ℝ)
variable (eq1 : AB = 8)
variable (eq2 : AC = 10)
variable (eq3 : BC = 12)

theorem triangle_altitude_length (h : ∃ AA1, AA1 * AA1 + BA1 * BA1 = 64 ∧ 
                                AA1 * AA1 + (BC - BA1) * (BC - BA1) = 100) :
    BA1 = 4.5 := by
  sorry 

end triangle_altitude_length_l552_552971


namespace curve_distance_property_l552_552275

-- Definitions
def curveC (x y : ℝ) : Prop := √(x^2 / 25) + √(y^2 / 9) = 1
def F1 := (-4, 0)
def F2 := (4, 0)
def distance (P1 P2 : ℝ × ℝ) : ℝ := real.sqrt ((P2.1 - P1.1)^2 + (P2.2 - P1.2)^2)

-- Problem Statement
theorem curve_distance_property (x y : ℝ) (P : ℝ × ℝ := (x, y)) :
  curveC x y → (distance P F1 + distance P F2) ≤ 10 := by
  sorry

end curve_distance_property_l552_552275


namespace smallest_positive_period_function_monotonically_increasing_intervals_l552_552772

noncomputable def A : ℝ := 3
noncomputable def ω : ℝ := π / 6
noncomputable def φ : ℝ := π / 6
noncomputable def c : ℝ := -1

def f (x : ℝ) := A * Real.sin (ω * x + φ) + c

theorem smallest_positive_period :
  ∃ T > 0, (∀⦃x⦄, f (x + T) = f x) ∧ T = 12 := sorry

theorem function_monotonically_increasing_intervals :
  ∀ (k : ℤ), (12 * k - 4 ≤ x ∧ x ≤ 12 * k + 2) → f' x ≥ 0 := sorry

end smallest_positive_period_function_monotonically_increasing_intervals_l552_552772


namespace angle_MBC_is_30_l552_552600

noncomputable def triangle_ABC (A B C : Type) := sorry
noncomputable def median_BM (A B C : Type) := sorry
noncomputable def altitude_AH (A B C : Type) := sorry
noncomputable def line_eq (x y : ℝ) := x = y

theorem angle_MBC_is_30 (A B C : Type) : 
  triangle_ABC A B C ∧
  median_BM A B C ∧
  altitude_AH A B C ∧
  line_eq (BM A B C) (AH A B C) → 
  angle M B C = 30 :=
by
  sorry

end angle_MBC_is_30_l552_552600


namespace birthday_problem_a_birthday_problem_b_l552_552366

theorem birthday_problem_a :
  (∀ (prob : ℕ → ℝ) (twelve : ℕ),
    (∀ month:ℕ, month > 0 ∧ month < 13 → prob month = 1 / 12) →
    twelve = 12 →
    prob(1) * prob(2) * prob(3) * prob(4) * prob(5) * prob(6) * prob(7) * prob(8) * prob(9) * prob(10) * prob(11) * prob(12)
    = 0.000053) :=
by
  sorry

theorem birthday_problem_b :
  (∀ (prob : ℕ → ℝ) (six : ℕ) (choose : ℕ → ℕ → ℕ),
    (∀ month:ℕ, month > 0 ∧ month < 13 → prob month = 1 / 12) →
    six = 6 →
    choose = fun n k => Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) →
    choose 12 2 * (1/2)^6
    = 0.015625) :=
by
  sorry

end birthday_problem_a_birthday_problem_b_l552_552366


namespace find_x_f_g_satisfies_l552_552087

def f : ℕ → ℕ
| 1 := 1
| 2 := 3
| 3 := 1
| _ := 0  -- Default case for completeness, although not needed

def g : ℕ → ℕ
| 1 := 3
| 2 := 2
| 3 := 1
| _ := 0  -- Default case for completeness, although not needed

theorem find_x_f_g_satisfies : ∀ x, f (g x) > g (f x) ↔ x = 2 := 
by
  sorry  -- proof goes here

end find_x_f_g_satisfies_l552_552087


namespace abs_five_minus_e_l552_552461

noncomputable def e : ℝ := Real.exp 1

theorem abs_five_minus_e : |5 - e| = 5 - e := by
  sorry

end abs_five_minus_e_l552_552461


namespace point_B_in_fourth_quadrant_l552_552892

theorem point_B_in_fourth_quadrant (x y : ℝ) (hx : x < 0) (hy : y < 0) : 
    -x > 0 ∧ y - 1 < 0 :=
by
  intro h
  exact and.intro (neg_pos.mpr hx) (sub_lt_zero_of_lt hy)

end point_B_in_fourth_quadrant_l552_552892


namespace brenda_cakes_l552_552021

theorem brenda_cakes : 
  let cakes_per_day := 20
  let days := 9
  let total_cakes := cakes_per_day * days
  let sold_cakes := total_cakes / 2
  total_cakes - sold_cakes = 90 :=
by 
  sorry

end brenda_cakes_l552_552021


namespace num_prime_sums_first_15_l552_552308

noncomputable def seq_sums_of_primes : ℕ → ℕ
| 0 => 3
| n+1 => seq_sums_of_primes n + Nat.prime (n+1+1)

theorem num_prime_sums_first_15 : 
  (Finset.filter (Nat.Prime) (Finset.range 15)).card = 3 :=
by
  sorry

end num_prime_sums_first_15_l552_552308


namespace solution_set_inequality_l552_552719

theorem solution_set_inequality (x : ℝ) : (x ≠ 1) → 
  ((x - 3) * (x + 2) / (x - 1) > 0 ↔ (-2 < x ∧ x < 1) ∨ x > 3) :=
by
  intros h
  sorry

end solution_set_inequality_l552_552719


namespace find_percentage_of_alcohol_in_second_solution_l552_552378

def alcohol_content_second_solution (V2: ℕ) (p1 p2 p_final: ℕ) (V1 V_final: ℕ) : ℕ :=
  ((V_final * p_final) - (V1 * p1)) * 100 / V2

def percentage_correct : Prop :=
  alcohol_content_second_solution 125 20 12 15 75 200 = 12

theorem find_percentage_of_alcohol_in_second_solution : percentage_correct :=
by
  sorry

end find_percentage_of_alcohol_in_second_solution_l552_552378


namespace Petya_sum_l552_552257

theorem Petya_sum : 
  let expr := [1, 2, 3, 4, 5, 6]
  let values := 2^(expr.length - 1)
  (sum_of_possible_values expr = values) := by 
  sorry

end Petya_sum_l552_552257


namespace pizza_remained_l552_552951

theorem pizza_remained (total_people : ℕ) (fraction_eating_pizza : ℚ)
  (total_pizza_pieces : ℕ) (pieces_per_person : ℕ)
  (h1 : total_people = 15)
  (h2 : fraction_eating_pizza = 3 / 5)
  (h3 : total_pizza_pieces = 50)
  (h4 : pieces_per_person = 4) :
  total_pizza_pieces - (((total_people : ℚ) * fraction_eating_pizza).natCast * pieces_per_person) = 14 := by
  sorry

end pizza_remained_l552_552951


namespace sequence_value_l552_552969

theorem sequence_value (x : ℕ) : 
  (5 - 2 = 1 * 3) ∧ 
  (11 - 5 = 2 * 3) ∧ 
  (20 - 11 = 3 * 3) ∧ 
  (x - 20 = 4 * 3) ∧ 
  (47 - x = 5 * 3) → 
  x = 32 :=
by 
  intros h 
  sorry

end sequence_value_l552_552969


namespace no_real_solution_log_eq_l552_552577

theorem no_real_solution_log_eq (x : ℝ) : 
  log (x + 5) + log (x - 3) = log (x^2 - 5 * x + 6) → False :=
by
  sorry

end no_real_solution_log_eq_l552_552577


namespace convert_5314_base8_to_base7_l552_552451

-- Definitions based on conditions
def base_convert (n : ℕ) (b : ℕ) : List ℕ :=
  if n = 0 then []
  else let (q, r) := quotRem n b
       in base_convert q b ++ [r]

def base8_to_dec (l : List ℕ) : ℕ :=
  l.reverse.foldl (λ s x, s * 8 + x) 0

def base7_representation := base_convert

-- The main theorem to prove
theorem convert_5314_base8_to_base7 : 
  base7_representation (base8_to_dec [5, 3, 1, 4]) 7 = [1, 1, 0, 2, 6] :=
by
  sorry

end convert_5314_base8_to_base7_l552_552451


namespace prob1_prob2_prob3_l552_552122

-- Problem 1
theorem prob1 (f : ℝ → ℝ) (m : ℝ) (f_def : ∀ x, f x = Real.log x + (1/2) * m * x^2)
  (tangent_line_slope : ℝ) (perpendicular_line_eq : ℝ) :
  (tangent_line_slope = 1 + m) →
  (perpendicular_line_eq = -1/2) →
  (tangent_line_slope * perpendicular_line_eq = -1) →
  m = 1 := sorry

-- Problem 2
theorem prob2 (f : ℝ → ℝ) (m : ℝ) (f_def : ∀ x, f x = Real.log x + (1/2) * m * x^2) :
  (∀ x, f x ≤ m * x^2 + (m - 1) * x - 1) →
  ∃ (m_ : ℤ), m_ ≥ 2 := sorry

-- Problem 3
theorem prob3 (f : ℝ → ℝ) (F : ℝ → ℝ) (x1 x2 : ℝ) (m : ℝ) 
  (f_def : ∀ x, f x = Real.log x + (1/2) * x^2)
  (F_def : ∀ x, F x = f x + x)
  (hx1 : 0 < x1) (hx2: 0 < x2) :
  m = 1 →
  F x1 = -F x2 →
  x1 + x2 ≥ Real.sqrt 3 - 1 := sorry

end prob1_prob2_prob3_l552_552122


namespace unique_line_through_P_forming_isosceles_with_line1_l552_552897

def Line1 (x y : ℝ) : Prop := x + 3 * y - 9 = 0
def PointP (x y : ℝ) : Prop := x = 3 ∧ y = 2

theorem unique_line_through_P_forming_isosceles_with_line1 :
  (∃ l : ℝ → ℝ → Prop,
    (∀ x y, l x y = (x - 3 * y + 3 = 0)) ∧
    (∃ x y, l x y ∧ PointP x y) ∧
    (∀ x y, Line1 x y → (x - 3 * y + 3 = 0))) :=
begin
  sorry
end

end unique_line_through_P_forming_isosceles_with_line1_l552_552897


namespace total_games_played_l552_552752

theorem total_games_played : nat.choose 11 2 = 55 := by
  sorry

end total_games_played_l552_552752


namespace seven_factorial_simplification_l552_552430

theorem seven_factorial_simplification : 7! - 6 * 6! - 6! = 0 := by
  sorry

end seven_factorial_simplification_l552_552430


namespace expression_simplifies_to_62_l552_552822

theorem expression_simplifies_to_62 (a b c : ℕ) (h1 : a = 14) (h2 : b = 19) (h3 : c = 29) :
  (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) / 
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = 62 := by {
  sorry -- Proof goes here
}

end expression_simplifies_to_62_l552_552822


namespace find_fx_find_m_l552_552101

def linear_increasing (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ f = (λ x, a * x + b)

def composition_equals (f : ℝ → ℝ) (h : linear_increasing f) : Prop :=
  ∀ x, f (f x) = 16 * x + 5

theorem find_fx (f : ℝ → ℝ) (h : linear_increasing f) (h_comp : composition_equals f h) :
  f = (λ x, 4 * x + 1) :=
sorry

def g (f : ℝ → ℝ) (m : ℝ) (x : ℝ) : ℝ :=
  f x * (x + m)

def g_increasing_on (g : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x1 x2, x1 ∈ I → x2 ∈ I → x1 < x2 → g x1 < g x2

theorem find_m (f : ℝ → ℝ) (h : linear_increasing f)
  (h_comp : composition_equals f h)
  (hf : f = (λ x, 4 * x + 1)) :
  ∀ m : ℝ, g_increasing_on (g f m) (set.Ioi 1) ↔ m ≥ -9 / 4 :=
sorry

end find_fx_find_m_l552_552101


namespace tangent_slope_at_A_midpoint_trajectory_l552_552867

-- Definitions based on the conditions
def circle (x y : ℝ) : Prop := x^2 + y^2 = 3
def A : ℝ × ℝ := (-2, 0)
def midpoint (B M : ℝ × ℝ) : Prop :=
  M = ((B.1 - 2) / 2, B.2 / 2)

-- Part 1: Proving the slope of the tangent line at point A
theorem tangent_slope_at_A (k : ℝ) : 
  (∀ (x y : ℝ), (circle x y → x = -2 ∧ y = k*(x + 2)) ∧ 
  (abs (2 * k) / Real.sqrt (k^2 + 1) = Real.sqrt 3)) ↔ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) := 
sorry

-- Part 2: Proving the trajectory equation of the midpoint M
theorem midpoint_trajectory (M : ℝ × ℝ) (B : ℝ × ℝ) (hB : circle B.1 B.2) : 
  midpoint B M → (M.1 + 1)^2 + M.2^2 = 3 / 4 := 
sorry

end tangent_slope_at_A_midpoint_trajectory_l552_552867


namespace perimeter_of_square_l552_552329

variable (s : ℝ) (side_length : ℝ)
def is_square_side_length_5 (s : ℝ) : Prop := s = 5
theorem perimeter_of_square (h: is_square_side_length_5 s) : 4 * s = 20 := sorry

end perimeter_of_square_l552_552329


namespace concurrency_of_perpendiculars_l552_552869

variables {A B C D P Q R P1 Q1 R1 : Type*}
variables {line : Type*} [line.line_l : line]
variables (AB CD AC BD BC AD l : line) [P : A ∈ AB] [Q : C ∈ CD] [R : A ∈ AC]
variables (P1 Q1 R1 : line) [P1_mid : midpoint P, Q ∈ P1] [Q1_mid : midpoint Q, R ∈ Q1] [R1_mid : midpoint R, P ∈ R1]

theorem concurrency_of_perpendiculars :
  ∃ X : Type*, concurrent PP1 QQ1 RR1 :=
sorry

end concurrency_of_perpendiculars_l552_552869


namespace compute_geometric_sum_l552_552809

open Complex

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem compute_geometric_sum : 
  let ω := omega in
  (ω ^ 1 + ω ^ 2 + ω ^ 3 + ω ^ 4 + ω ^ 5 + ω ^ 6 + 
  ω ^ 7 + ω ^ 8 + ω ^ 9 + ω ^ 10 + ω ^ 11 + ω ^ 12 + 
  ω ^ 13 + ω ^ 14 + ω ^ 15 + ω ^ 16) = -1 :=
by 
  let ω := omega
  have h : ω ^ 17 = 1 := 
    by sorry
  have h1 : ω ^ 16 = 1 / ω := 
    by sorry
  sorry

end compute_geometric_sum_l552_552809


namespace find_length_AC_l552_552588

noncomputable theory

open_locale real

variables {A B C D K : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space K]
variables (inscribed : ∃ (A B C D : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D], 
  angle_ratio : ∀ (A B C D : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D], 
  ∃ (angle_A angle_B angle_C : ℝ), 
  angle_A / angle_B = 2 / 3 ∧ angle_B / angle_C = 3 / 4)
variables (length_CD : ∀ (C D : Type*) [metric_space C] [metric_space D], 21)
variables (length_BC : ∀ (B C : Type*) [metric_space B] [metric_space C], 14 * real_sqrt(3) - 10.5)

theorem find_length_AC : 
  ∃ (length_AC : ℝ), 
  length_AC = 35 :=
by {
  sorry
}

end find_length_AC_l552_552588


namespace sum_of_digits_of_greatest_five_digit_number_with_product_210_l552_552610

theorem sum_of_digits_of_greatest_five_digit_number_with_product_210 :
  ∃ (M : ℕ), (digits M).length = 5 ∧ (digits M).prod = 210 ∧ (digits M).sum = 16 :=
by 
  sorry

end sum_of_digits_of_greatest_five_digit_number_with_product_210_l552_552610


namespace triangle_area_angle_A_condition_2_angle_A_condition_3_l552_552598

-- Definitions for the triangle
variables (a b c : ℝ) (A B C : ℝ)
variable (ΔABC : a = 1 ∧ b = 2 ∧ c = 2 * Real.sqrt 2)

-- Proof that the area of the triangle is given by sqrt(7)/4
theorem triangle_area (h : ΔABC) : 
  let S := 1 / 2 * 1 * 2 * ((Real.sqrt 7) / 4) in S = Real.sqrt 7 / 4 :=
by
  sorry

-- Definitions for the conditions in part 2
variable cond_1 : B = 2 * A
variable cond_2 : B = (Real.pi / 3) + A
variable cond_3 : C = 2 * A

-- Prove that angle A is π/6 given condition 2
theorem angle_A_condition_2 (κ : ΔABC ∧ cond_2) : A = Real.pi / 6 :=
by
  sorry

-- Prove that angle A is π/6 given condition 3
theorem angle_A_condition_3 (κ : ΔABC ∧ cond_3) : A = Real.pi / 6 :=
by
  sorry

end triangle_area_angle_A_condition_2_angle_A_condition_3_l552_552598


namespace triangle_csc_inequality_triangle_csc_inequality_eq_l552_552208

theorem triangle_csc_inequality (α β γ : ℝ) (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) (h_sum : α + β + γ = Real.pi) :
  Real.csc (α / 2) ^ 2 + Real.csc (β / 2) ^ 2 + Real.csc (γ / 2) ^ 2 ≥ 12 :=
sorry

-- Additionally, equality holds if and only if the triangle is equilateral.
theorem triangle_csc_inequality_eq (α β γ : ℝ) (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) (h_sum : α + β + γ = Real.pi) :
  (Real.csc (α / 2) ^ 2 + Real.csc (β / 2) ^ 2 + Real.csc (γ / 2) ^ 2 = 12) ↔ (α = β ∧ β = γ ∧ γ = α) :=
sorry

end triangle_csc_inequality_triangle_csc_inequality_eq_l552_552208


namespace count_digit_sequences_l552_552266

def is_valid_digit_sequence (l : List ℕ) : Prop :=
  l.count 1 = 3 ∧ l.count 2 = 3 ∧ l.count 3 = 2 ∧ l.count 4 = 1 ∧
  (∃ i j k, (i < j ∧ j < k ∧ l[i] = 1 ∧ l[j] = 2 ∧ l[k] = 3 ∧ l.counti_id 2 = 1)) ∧
  (∀ i, i < l.length - 1 → (l[i] = 2 → l[i + 1] ≠ 2))

noncomputable def count_valid_sequences : ℕ :=
  List.permutations [1, 1, 1, 2, 2, 2, 3, 3, 4] |>.count is_valid_digit_sequence

theorem count_digit_sequences : count_valid_sequences = 254 :=
by sorry

end count_digit_sequences_l552_552266


namespace ellipse_equation_fixed_point_l552_552529

/-- Given an ellipse C : (x^2 / a^2) + (y^2 / b^2) = 1 with a > b > 0,
    an equilateral triangle ∆MF₁F₂, and the point M(0, √3) passing through the ellipse C,
    (1) Prove that the equation of the ellipse C is (x^2 / 4) + (y^2 / 3) = 1,
    (2) Prove that for a line passing through the points A and E on the ellipse C,
    if the line passing through point P(4, 0), and a vertical line intersects ellipse C at A and B,
    the line AE intersects the x-axis at a fixed point (1, 0). -/
theorem ellipse_equation_fixed_point 
  {a b : ℝ} 
  (C : ℝ → ℝ → Prop)
  (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : a > b)
  (M : ℝ × ℝ)
  (eq1 : C (0 : ℝ) (Real.sqrt 3)) 
  (equilateral_triangle : ∀ F1 F2 : ℝ × ℝ, equilateral_triangle ∆ (0, √3) F1 F2) :
  (∀ x y, C x y ↔ (x^2 / 4) + (y^2 / 3) = 1) ∧ 
  (∀ A E : ℝ × ℝ, 
    intersects (line_through A E) (x_axis) ({1} : ℝ)) :=
sorry

end ellipse_equation_fixed_point_l552_552529


namespace num_multiples_of_7_ending_in_7_l552_552569

theorem num_multiples_of_7_ending_in_7 (n : ℕ) :
  ∃ m : ℕ, (m < 7) ∧ (finset.card (finset.filter (λ x, x < 500 ∧ x % 10 = 7) 
  (finset.image (λ k, 7 * k) (finset.range (10 * m + 3 + 1)))) = 7) := by
  sorry

end num_multiples_of_7_ending_in_7_l552_552569


namespace dot_product_conditioned_l552_552562

variables (a b : ℝ×ℝ)

def condition1 : Prop := 2 • a + b = (1, 6)
def condition2 : Prop := a + 2 • b = (-4, 9)
def dot_product (u v : ℝ×ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem dot_product_conditioned :
  condition1 a b ∧ condition2 a b → dot_product a b = -2 :=
by
  sorry

end dot_product_conditioned_l552_552562


namespace car_speed_ratio_l552_552800

theorem car_speed_ratio (v_A v_B : ℕ) (h1 : v_B = 50) (h2 : 6 * v_A + 2 * v_B = 1000) :
  v_A / v_B = 3 :=
sorry

end car_speed_ratio_l552_552800


namespace min_value_of_PA_PB_dot_product_l552_552874

noncomputable def triangle_min_dot_product : ℝ :=
  let BC := 3
  let AC := 4
  let AB := 5
  let γ := real.arccos(1/4) -- assuming angle γ calculation from side lengths
  
  have h : ∀ P : ℝ × ℝ, 
           (P.1, P.2) ∈ 
             ({0} ∪ 
             (Icc 0 BC ∪ (Icc 0 AC ∪ Icc 0 AB)))
           → ∀ (PA PB : ℝ × ℝ),
             minimum_value_of_dot_product (PA • PB) P := sorry,
             
             _ : ∀ P, 
              P x y ⬝ BC |
              P x y ∈ (segment ? _)|
  ∃ k, 0 ≤ k ∧ k ≤ 1 ∧ 
    minimum_value_of_dot_product((9 * k^2 - 5 * k * BC * cos(aba)) )
  
begin
  /-
  Here we would use the geometric interpretation, solver functions, and more advanced calculus-related methods 
  to formulate and solve for the parameter that minimizes PA • PB
  -/
  sorry

end

theorem min_value_of_PA_PB_dot_product 
  (P: Point (𝔹))
  (BC: ℝ)
  (AC:ℝ)
  (AB:ℝ) : 
  BC = 3 →
  AC = 4 →
  AB = 5 →
  min_val_of_dot_product == ⟪minimum_value_of_dot_product == (25/64)⟫  := sorry



end min_value_of_PA_PB_dot_product_l552_552874


namespace simplify_expression_l552_552659

theorem simplify_expression :
  (Complex.mk (-1) (Real.sqrt 3) / 2) ^ 12 + (Complex.mk (-1) (-Real.sqrt 3) / 2) ^ 12 = 2 := by
  sorry

end simplify_expression_l552_552659


namespace complex_point_in_fourth_quadrant_l552_552898

-- Definition of the complex number given in the problem
def complex_z : ℂ :=
  (3 - complex.i) * (2 - complex.i)

-- Statement to prove the problem
theorem complex_point_in_fourth_quadrant :
  (complex_z.re > 0) ∧ (complex_z.im < 0) :=
by
  sorry

end complex_point_in_fourth_quadrant_l552_552898


namespace bob_total_earnings_l552_552428

def hourly_rate_regular := 5
def hourly_rate_overtime := 6
def regular_hours_per_week := 40

def hours_worked_week1 := 44
def hours_worked_week2 := 48

def earnings_week1 : ℕ :=
  let regular_hours := regular_hours_per_week
  let overtime_hours := hours_worked_week1 - regular_hours_per_week
  (regular_hours * hourly_rate_regular) + (overtime_hours * hourly_rate_overtime)

def earnings_week2 : ℕ :=
  let regular_hours := regular_hours_per_week
  let overtime_hours := hours_worked_week2 - regular_hours_per_week
  (regular_hours * hourly_rate_regular) + (overtime_hours * hourly_rate_overtime)

def total_earnings : ℕ := earnings_week1 + earnings_week2

theorem bob_total_earnings : total_earnings = 472 := by
  sorry

end bob_total_earnings_l552_552428


namespace leading_coefficient_of_g_l552_552303

-- Given condition: g(x + 1) - g(x) = 6x^2 + 4x + 6
def poly_condition (g : ℕ → ℕ) : Prop :=
  ∀ x : ℕ, g(x + 1) - g(x) = 6 * x^2 + 4 * x + 6

-- Prove that the leading coefficient of g(x) is 2
theorem leading_coefficient_of_g (g : ℕ → ℕ) (h : poly_condition g) : True := sorry

end leading_coefficient_of_g_l552_552303


namespace increasing_log_pow_iff_l552_552158

open Real

theorem increasing_log_pow_iff (a : ℝ) : 
  (∀ x : ℝ, (log 0.5 a) ^ x > 0) → 0 < a ∧ a < 0.5 := by
  sorry

end increasing_log_pow_iff_l552_552158


namespace range_of_m_l552_552580

theorem range_of_m (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 4) : (x - 1)^2 + 4 < m → 5 < m :=
by {
  intro h,
  linarith,
}

end range_of_m_l552_552580


namespace chord_length_of_parabola_l552_552290

theorem chord_length_of_parabola {x1 x2 y1 y2 : ℝ} :
  let focus := (0, 1 / 8 : ℝ)
  let parabola := λ x : ℝ, 2 * x^2
  let slope := -Real.sqrt 3
  let chord_length := Real.sqrt (1 + slope ^ 2) * Real.sqrt ((x1 + x2) ^ 2 - 4 * x1 * x2)
  in (parabola x1 = y1) ∧ (parabola x2 = y2) ∧ (x1 + x2 = -slope / 2) ∧ (x1 * x2 = 1 / 16) ∧ y1 + y2 = 1 / 4 + 3 → chord_length = 2 := by
  sorry

end chord_length_of_parabola_l552_552290


namespace water_consumption_and_bill_34_7_l552_552324

noncomputable def calculate_bill (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 20.8 * x
  else if 1 < x ∧ x ≤ (5 / 3) then 27.8 * x - 7
  else 32 * x - 14

theorem water_consumption_and_bill_34_7 (x : ℝ) :
  calculate_bill 1.5 = 34.7 ∧ 5 * 1.5 = 7.5 ∧ 3 * 1.5 = 4.5 ∧ 
  5 * 2.6 + (5 * 1.5 - 5) * 4 = 23 ∧ 
  4.5 * 2.6 = 11.7 :=
  sorry

end water_consumption_and_bill_34_7_l552_552324


namespace time_spent_on_type_A_problems_l552_552946

theorem time_spent_on_type_A_problems : 
  (∃ (x : ℝ), 
     let total_minutes := 180 in
     let type_A_problems := 15 in
     let type_B_problems := 200 - type_A_problems in
     let time_A := 2 * x in
     let time_B := x in
     let total_time_A := type_A_problems * time_A in
     let total_time_B := type_B_problems * time_B in
     total_time_A + total_time_B = total_minutes) →
  (∃ T : ℝ, T = 25.116) :=
by
  sorry

end time_spent_on_type_A_problems_l552_552946


namespace pyramid_frustum_volume_l552_552401

theorem pyramid_frustum_volume
  (base_edge_original : ℕ)
  (altitude_original : ℕ)
  (altitude_ratio : ℚ)
  (h1 : base_edge_original = 40)
  (h2 : altitude_original = 20)
  (h3 : altitude_ratio = 1 / 3) :
  let volume_original := (1 / 3 : ℚ) * (base_edge_original^2) * altitude_original in
  let volume_smaller := (altitude_ratio^3) * volume_original in
  let volume_frustum := volume_original - volume_smaller in
  volume_frustum / volume_original = 26 / 27 :=
by
  -- Proof goes here
  sorry

end pyramid_frustum_volume_l552_552401


namespace median_list_1_to_300_l552_552445

def median_of_list {α : Type} [LinearOrder α] (l : List α) : α :=
  l.sorted l.length / 2

noncomputable def nums_list : List ℕ := 
  (List.range (300 + 1)).bind (λ n, List.repeat n n)

theorem median_list_1_to_300 : median_of_list nums_list = 212 := 
  sorry

end median_list_1_to_300_l552_552445


namespace graph_plot_inverse_relation_l552_552605

theorem graph_plot_inverse_relation:
  ∀ (w l : ℕ), 0 < w → 0 < l → w * l = 24 → ((w, l) = (1, 24) ∨ (w, l) = (2, 12) ∨ (w, l) = (3, 8) ∨
  (w, l) = (4, 6) ∨ (w, l) = (6, 4) ∨ (w, l) = (8, 3) ∨ (w, l) = (12, 2) ∨ (w, l) = (24, 1)) :=
begin
  sorry
end

end graph_plot_inverse_relation_l552_552605


namespace Jack_remaining_money_l552_552978

-- Definitions based on conditions
def initial_money : ℕ := 100
def initial_bottles : ℕ := 4
def bottle_cost : ℕ := 2
def extra_bottles : ℕ := 8
def cheese_cost_per_pound : ℕ := 10
def cheese_weight : ℚ := 1 / 2

-- The statement we want to prove
theorem Jack_remaining_money :
  let total_water_cost := (initial_bottles + extra_bottles) * bottle_cost,
      total_cheese_cost := cheese_cost_per_pound * cheese_weight,
      total_cost := total_water_cost + total_cheese_cost
  in initial_money - total_cost = 71 :=
by
  sorry

end Jack_remaining_money_l552_552978


namespace calculate_insurance_cost_l552_552685

def total_cost_apartment : ℝ := 7000000
def loan_amount : ℝ := 4000000
def loan_interest_rate : ℝ := 0.101
def property_insurance_tariff : ℝ := 0.0009
def life_health_insurance_female : ℝ := 0.0017
def life_health_insurance_male : ℝ := 0.0019
def title_insurance_tariff : ℝ := 0.0027
def svetlana_share : ℝ := 0.2
def dmitry_share : ℝ := 0.8

theorem calculate_insurance_cost :
  let total_loan_amount := loan_amount + loan_amount * loan_interest_rate in
  let property_insurance_cost := total_loan_amount * property_insurance_tariff in
  let title_insurance_cost := total_loan_amount * title_insurance_tariff in
  let svetlana_insurance_cost := total_loan_amount * svetlana_share * life_health_insurance_female in
  let dmitry_insurance_cost := total_loan_amount * dmitry_share * life_health_insurance_male in
  let total_insurance_cost := property_insurance_cost + title_insurance_cost + svetlana_insurance_cost + dmitry_insurance_cost in
  total_insurance_cost = 24045.84 :=
by
  sorry

end calculate_insurance_cost_l552_552685


namespace imaginary_part_l552_552111

theorem imaginary_part (z : ℂ) (hz : z = (1 - complex.i) / (1 + 3 * complex.i)) :
  complex.im z = -2 / 5 :=
by {
  sorry
}

end imaginary_part_l552_552111


namespace prism_volume_is_4_l552_552782

noncomputable def regular_triangular_prism_volume (ABC A1 B1 C1 : Type) (CD : Segment) (K L : Point)
  (DL DK : ℝ) (D_sqrt2 : DL = real.sqrt 2) (D_sqrt3 : DK = real.sqrt 3) : ℝ :=
  begin
    sorry
  end

theorem prism_volume_is_4 (ABC A1 B1 C1 CD K L : Type) (DL DK : ℝ)
  (D_sqrt2 : DL = real.sqrt 2) (D_sqrt3 : DK = real.sqrt 3) :
  regular_triangular_prism_volume ABC A1 B1 C1 CD K L DL DK D_sqrt2 D_sqrt3 = 4 :=
  by
    sorry

end prism_volume_is_4_l552_552782


namespace solve_nested_equation_l552_552272

theorem solve_nested_equation : ∀ (x : ℝ), 
  (∀ (n : ℕ), (n = 1985) → 
  (∃ (y : ℝ), y = 1 + sqrt (1 + x) ∧ 
  (x = (n : ℕ).recOn (2 + (fun _ r => (x / (2 + r))) y))) → 
  x ≠ 0 → x = 3) :=
by
  sorry

end solve_nested_equation_l552_552272


namespace find_x_for_g_eq_14_l552_552682

def f (x : ℝ) : ℝ := 18 / (x + 2)
def g (x : ℝ) : ℝ := 4 * (Function.inverse f x) - 2

theorem find_x_for_g_eq_14 : ∃ (x : ℝ), g(x) = 14 ∧ x = 3 :=
by {
    sorry
}

end find_x_for_g_eq_14_l552_552682


namespace smallest_nonappearing_integer_HMMT_2023_l552_552032

-- Problem statement:
-- Prove that the smallest positive integer that does not appear in any problem statement 
-- in any round of the HMMT November 2023 is 22, given the table of number appearances.

def appears_in_HMMT_2023 (n : ℕ) : Prop :=
  n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
       12, 13, 14, 15, 16, 17, 18, 19, 20, 21}

theorem smallest_nonappearing_integer_HMMT_2023 : 
  ∀ n : ℕ, n > 0 → (¬ appears_in_HMMT_2023 n ↔ n = 22) :=
by sorry

end smallest_nonappearing_integer_HMMT_2023_l552_552032


namespace exists_unit_vector_parallel_to_a_l552_552917

-- Define the vector a
def a : ℝ × ℝ := (3, 4)

-- Define the magnitude of vector a
def magnitude_a : ℝ := real.sqrt (a.1^2 + a.2^2)

-- Define the unit vector parallel to a
def unit_vector_a : ℝ × ℝ := (a.1 / magnitude_a, a.2 / magnitude_a)

-- Define the statement to be proved
theorem exists_unit_vector_parallel_to_a : 
  ∃ u : ℝ × ℝ, u = (3 / 5, 4 / 5) ∧ (u.1 = unit_vector_a.1 ∧ u.2 = unit_vector_a.2) :=
by
  use (3 / 5, 4 / 5)
  split
  case left => rfl
  case right => sorry

end exists_unit_vector_parallel_to_a_l552_552917


namespace scientific_notation_correct_l552_552001

def number := 56990000

theorem scientific_notation_correct : number = 5.699 * 10^7 :=
  by
    sorry

end scientific_notation_correct_l552_552001


namespace kiyana_gives_half_l552_552199

theorem kiyana_gives_half (total_grapes : ℕ) (h : total_grapes = 24) : 
  (total_grapes / 2) = 12 :=
by
  sorry

end kiyana_gives_half_l552_552199


namespace tan_theta_value_l552_552573

open Real

theorem tan_theta_value (θ : ℝ) (h : sin (θ / 2) - 2 * cos (θ / 2) = 0) : tan θ = -4 / 3 :=
sorry

end tan_theta_value_l552_552573


namespace length_of_AB_l552_552631

theorem length_of_AB 
  (AB BC CD AD : ℕ)
  (h1 : AB = 1 * BC / 2)
  (h2 : BC = 6 * CD / 5)
  (h3 : AB + BC + CD = 56)
  : AB = 12 := sorry

end length_of_AB_l552_552631


namespace quadratic_residues_count_l552_552171

theorem quadratic_residues_count (p : ℕ) (hp : Nat.Prime p) (hp2 : p % 2 = 1) :
  ∃ (q_residues : Finset (ZMod p)), q_residues.card = (p - 1) / 2 ∧
  ∃ (nq_residues : Finset (ZMod p)), nq_residues.card = (p - 1) / 2 ∧
  ∀ d ∈ q_residues, ∃ x y : ZMod p, x^2 = d ∧ y^2 = d ∧ x ≠ y :=
by
  sorry

end quadratic_residues_count_l552_552171


namespace subset_count_l552_552556

theorem subset_count (M : set ℕ) (hM : M = {0, 1, 2}) :
  ∃ (N : set (set ℕ)), N = {N | N ⊆ M} ∧ N.card = 8 := sorry

end subset_count_l552_552556


namespace Petya_sum_l552_552253

theorem Petya_sum : 
  let expr := [1, 2, 3, 4, 5, 6]
  let values := 2^(expr.length - 1)
  (sum_of_possible_values expr = values) := by 
  sorry

end Petya_sum_l552_552253


namespace state_b_selection_percentage_l552_552584

/-- 
In a competitive examination:
- State A has 8000 candidates appeared and 6% of them are selected.
- State B has the same number of candidates appeared as State A.
- The number of candidates selected in State B is 80 more than in State A.

Prove that the percentage of candidates selected in State B is 7%.
-/
theorem state_b_selection_percentage :
  let candidates_A := 8000
  let selected_A := 0.06 * candidates_A
  let candidates_B := candidates_A
  let selected_B := selected_A + 80
  let percentage_B := (selected_B / candidates_B) * 100
  percentage_B = 7 :=
by
  let candidates_A := 8000
  let selected_A := 0.06 * candidates_A
  let candidates_B := candidates_A
  let selected_B := selected_A + 80
  let percentage_B := (selected_B / candidates_B) * 100
  sorry

end state_b_selection_percentage_l552_552584


namespace remainder_div_l552_552838

open Polynomial

noncomputable def dividend : ℤ[X] := X^4
noncomputable def divisor  : ℤ[X] := X^2 + 3 * X + 2

theorem remainder_div (f g : ℤ[X]) : (f % g) = -6 * X - 6 :=
by
  have f := dividend
  have g := divisor
  sorry

end remainder_div_l552_552838


namespace points_on_ellipse_line_pass_through_fixed_point_l552_552545

noncomputable def ellipse : set (ℝ × ℝ) :=
  {p | ∃ (a b: ℝ), a^2 = 4 ∧ b^2 = 1 ∧ p.1^2 / a^2 + p.2^2 / b^2 = 1}

theorem points_on_ellipse :
  (-1, real.sqrt 3 / 2) ∈ ellipse ∧ (1, real.sqrt 3 / 2) ∈ ellipse ∧ (0, 1) ∈ ellipse :=
sorry

theorem line_pass_through_fixed_point (A B : ℝ × ℝ) \
  (h1 : A ∈ ellipse) (h2 : B ∈ ellipse) (P2 : ℝ × ℝ := (0, 1)) \
  (h3 : ¬ on_line P2) (h4 : sum_of_slopes (P2, A, B) = -1) :
  line_pass_through (2, -1) :=
sorry

end points_on_ellipse_line_pass_through_fixed_point_l552_552545


namespace triangle_similarity_l552_552643

variables {A B C D M N : Type} [PlanarGeometry]
variables (Parallelogram : ABCD.is_parallelogram)
variables (AM_perp_BC : is_perpendicular AM BC)
variables (AN_perp_CD : is_perpendicular AN CD)

theorem triangle_similarity 
  (Parallelogram : ABCD.is_parallelogram)
  (AM_perp_BC : is_perpendicular AM BC)
  (AN_perp_CD : is_perpendicular AN CD) :
  similar (triangle MAN) (triangle ABC) := 
sorry

end triangle_similarity_l552_552643


namespace brenda_cakes_l552_552015

-- Definitions based on the given conditions
def cakes_per_day : ℕ := 20
def days : ℕ := 9
def total_cakes_baked : ℕ := cakes_per_day * days
def cakes_sold : ℕ := total_cakes_baked / 2
def cakes_left : ℕ := total_cakes_baked - cakes_sold

-- Formulate the theorem
theorem brenda_cakes : cakes_left = 90 :=
by {
  -- To skip the proof steps
  sorry
}

end brenda_cakes_l552_552015


namespace minor_arc_circumference_l552_552993

theorem minor_arc_circumference (A B C : Type) 
  (hA : Point A) (hB : Point B) (hC : Point C)
  (r : ℝ) (h_r : r = 15)
  (angle_ACB : ℝ) (h_angle : angle_ACB = 45) :
  calc arc_length : ℝ = 7.5 * π :=
by 
  sorry

end minor_arc_circumference_l552_552993


namespace john_paid_more_than_jane_l552_552197

theorem john_paid_more_than_jane :
    let original_price : ℝ := 40.00
    let discount_percentage : ℝ := 0.10
    let tip_percentage : ℝ := 0.15
    let discounted_price : ℝ := original_price - (discount_percentage * original_price)
    let john_tip : ℝ := tip_percentage * original_price
    let john_total : ℝ := discounted_price + john_tip
    let jane_tip : ℝ := tip_percentage * discounted_price
    let jane_total : ℝ := discounted_price + jane_tip
    let difference : ℝ := john_total - jane_total
    difference = 0.60 :=
by
  sorry

end john_paid_more_than_jane_l552_552197


namespace angle_inequality_l552_552602

theorem angle_inequality
  {A B C M : Type*}
  [euclidean_geometry A B C M]
  (h_triangle : is_triangle A B C)
  (h_point : M ∈ interior A B C) :
  ∠ B M C > ∠ B A C :=
sorry

end angle_inequality_l552_552602


namespace probability_each_ball_differs_from_more_than_half_l552_552459

theorem probability_each_ball_differs_from_more_than_half (n : ℕ) (p : ℝ)
  (h1 : n = 8) 
  (h2 : p = 0.5) 
  : (∃ P : ℝ, P = 35 / 128) ∧ 
    P = (probability_each_ball_differs_from_more_than_half n p) := 
sorry

end probability_each_ball_differs_from_more_than_half_l552_552459


namespace petya_sum_expression_l552_552248

theorem petya_sum_expression : 
  (let expressions := finset.image (λ (s : list bool), 
    list.foldl (λ acc ⟨b, n⟩, if b then acc + n else acc - n) 1 (s.zip [2, 3, 4, 5, 6])) 
    (finset.univ : finset (vector bool 5))) in
    expressions.sum) = 32 := 
sorry

end petya_sum_expression_l552_552248


namespace monotonic_increase_interval_range_of_a_l552_552906

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x + 2 * Real.exp x - a * x^2
def h (x : ℝ) : ℝ := x

theorem monotonic_increase_interval :
  ∃ I : Set ℝ, I = Set.Ioi 1 ∧ ∀ x ∈ I, ∀ y ∈ I, x ≤ y → f x ≤ f y := 
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, (g x1 a - h x1) * (g x2 a - h x2) > 0) ↔ a ∈ Set.Iic 1 :=
  sorry

end monotonic_increase_interval_range_of_a_l552_552906


namespace number_of_valid_N_l552_552471

theorem number_of_valid_N : 
  { N : ℕ // 2017 ≡ 17 [MOD N] ∧ N > 17 }.card = 13 :=
by sorry

end number_of_valid_N_l552_552471


namespace balls_in_boxes_with_constraints_l552_552571

/- Define the conditions -/
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

/- Proof statement (no proof required) -/
theorem balls_in_boxes_with_constraints : 
  ∃! (ways : ℕ), ways = 21 ∧ ∀ (arr : list (ℕ × ℕ × ℕ)),
    (
      arr.length = num_boxes ∧
      (∑ b in arr, b.fst = num_balls) ∧
      -- Ensure at least one box contains at least one ball
      (∃ b ∈ arr, b.fst > 0)
    ) → ways = 21 :=
sorry

end balls_in_boxes_with_constraints_l552_552571


namespace smiths_loads_of_laundry_l552_552590

/-
  Problem:
  Given:
  - Kylie uses 3 bath towels.
  - Her 2 daughters use a total of 6 bath towels.
  - Her husband uses a total of 3 bath towels.
  - The washing machine can fit 4 bath towels for one load of laundry.
  
  Prove:
  The Smiths need 3 loads of laundry to clean all of their used towels.
-/

def total_towels (k: ℕ) (d: ℕ) (h: ℕ) : ℕ := k + d + h
def loads_needed (towels: ℕ) (capacity: ℕ) : ℕ := towels / capacity

theorem smiths_loads_of_laundry : 
  total_towels 3 6 3 / 4 = 3 :=
by 
  simp [total_towels, loads_needed]
  exact rfl

end smiths_loads_of_laundry_l552_552590


namespace solution_exists_l552_552992

theorem solution_exists (n k : ℕ) (hn : n ≥ 2 * k) (hk : 2 * k > 3) : 
  (nat.choose n k = (2 * n - k) * nat.choose n 2) → (n = 27 ∧ k = 4) :=
sorry

end solution_exists_l552_552992


namespace police_capture_bandit_l552_552696

theorem police_capture_bandit :
  ∃ (algorithm : ℕ → ℕ × ℕ), 
    (∀ (time : ℕ), 
      ∃ (police_position bandit_position : ℕ × ℕ), 
      (police_position.2 = 0 ∧ police_position.1 % 100 = 0) ∧
      (∀ (t : ℕ), police_position = algorithm t) ∧
      (police_position.1 % 200 = 0 → bandit_position.1 % 200 = 0 ∧ 
       bandit_position.1 % 100 ≠ 0 ∧ bandit_position.2 = 0 ∧ 
       (abs (bandit_position.1 - police_position.1) < 200 →
        ∃ (future_position : ℕ × ℕ), 
          (future_position = algorithm (t+1) ∧
           police_position.1 = future_position.1 ∧
           police_position.2 ≠ bandit_position.2 ∧ 
           bandit_position.2 = future_position.2 ∧
           future_position.1 = bandit_position.1)
       )
    ) → 
  ∀ (t : ℕ), ∃ (p b : ℕ × ℕ), 
    p.2 = 0 ∧ p.1 % 100 = 0 ∧ 
    p = algorithm t ∧ 
    (p.1 % 200 = 0 → b.1 % 200 = 0 ∧ 
    b.1 % 100 ≠ 0 ∧ b.2 = 0 ∧ 
    ∃ (future_p : ℕ × ℕ), 
      future_p = algorithm (t+1) ∧ 
      p.1 = future_p.1 ∧ p.2 ≠ b.2 ∧ 
      b.2 = future_p.2 ∧ future_p.1 = b.1
    ) → 
  ∃ (t : ℕ), algorithm (t+1) = algorithm t :=
sorry

end police_capture_bandit_l552_552696


namespace total_worth_of_stock_l552_552960

theorem total_worth_of_stock :
  let cost_expensive := 10
  let cost_cheaper := 3.5
  let total_modules := 11
  let cheaper_modules := 10
  let expensive_modules := total_modules - cheaper_modules
  let worth_cheaper_modules := cheaper_modules * cost_cheaper
  let worth_expensive_module := expensive_modules * cost_expensive 
  worth_cheaper_modules + worth_expensive_module = 45 := by
  sorry

end total_worth_of_stock_l552_552960


namespace pyramid_volume_l552_552436

theorem pyramid_volume
  (FB AC FA FC AB BC : ℝ)
  (hFB : FB = 12)
  (hAC : AC = 4)
  (hFA : FA = 7)
  (hFC : FC = 7)
  (hAB : AB = 7)
  (hBC : BC = 7) :
  (1/3 * AC * (1/2 * FB * 3)) = 24 := by sorry

end pyramid_volume_l552_552436


namespace geometry_problem_l552_552617

theorem geometry_problem
  (ABC : Triangle)
  (D : Point)
  (I : Point)
  (P : Point)
  (Q : Point)
  (h_acute : ABC.is_acute)
  (hD_on_BC : D ∈ segment BC)
  (hI_incenter : I = ABC.incenter)
  (hP_on_circum_ABD : P ∈ circumcircle (Triangle.mk A B D) ∧ P ∈ line BI)
  (hQ_on_circum_ACD : Q ∈ circumcircle (Triangle.mk A C D) ∧ Q ∈ line CI)
  (h_area_equal : area (Triangle.mk P I D) = area (Triangle.mk Q I D)) :
  dist P I * dist Q D = dist Q I * dist P D := sorry

end geometry_problem_l552_552617


namespace discounted_price_correct_l552_552717

noncomputable def discounted_price (original_price : ℝ) (discount : ℝ) : ℝ :=
  original_price - (discount / 100 * original_price)

theorem discounted_price_correct :
  discounted_price 800 30 = 560 :=
by
  -- Correctness of the discounted price calculation
  sorry

end discounted_price_correct_l552_552717


namespace tangent_at_x1_monotonic_intervals_range_of_b_l552_552115

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := log x - a * x + (1 - a) / x - 1

-- Define the function g(x) with a parameter b
def g (x : ℝ) (b : ℝ) : ℝ := x^2 - 2 * b * x - 5 / 12

-- Condition (1)
theorem tangent_at_x1 (h : ∀ x > 0, x ≠ 1) : 
  let fx := f 1 1 in fx = -2 := sorry

-- Condition (2)
theorem monotonic_intervals (a : ℝ) (h : a = 1/3) : 
  let f' (x : ℝ) := 1 / x - a + (a - 1) / x^2 in
  ∀ x > 0, (0 < x ∧ x < 1) → f' x < 0 ∧ (1 < x ∧ x < 2) → f' x > 0 ∧ (x > 2) → f' x < 0 := sorry

-- Condition (3)
theorem range_of_b (h : ∀ x1 ∈ Set.Icc (1 : ℝ) 2, ∃ x2 ∈ Set.Icc (0 : ℝ) 1, f x1 (1 / 3) ≥ g x2 b) : ∀ b : ℝ,
  (1 / 2 ≤ b ∧ b < ∞) := sorry

end tangent_at_x1_monotonic_intervals_range_of_b_l552_552115


namespace distinct_real_roots_iff_l552_552500

theorem distinct_real_roots_iff (k : ℝ) : x^2 - 4 * x + 1 = -2 * k → k < 3 / 2 :=
by
  assume h : x^2 - 4 * x + 1 = -2 * k
  sorry

end distinct_real_roots_iff_l552_552500


namespace angle_C_is_60_degree_l552_552973

variables (A B C D E : Type) [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ E]
variables {AD BC BE AC : ℝ}
variables (triangle_ABC : Triangle ℝ)
variables (angle_bisector_AD : AngleBisector ℝ)
variables (angle_bisector_BE : AngleBisector ℝ)
variables (H1 : AD * BC = BE * AC)
variables (H2 : AC ≠ BC)

theorem angle_C_is_60_degree (hAD : angle_bisector_AD (angleBAC A B C D))
                              (hBE : angle_bisector_BE (angleABC A B C E)) :
  ∠ C = 60 := 
sorry

end angle_C_is_60_degree_l552_552973


namespace min_students_for_property_P_l552_552947

def multiple_choice_questions : Type := Fin 4

structure answer :=
(q1 q2 q3 : multiple_choice_questions)

def property_P (group : List answer) : Prop :=
  ∀ (a b : answer), a ∈ group → b ∈ group → a ≠ b →
    (∀ {i j k}, (i = a.q1 ∧ i = b.q1) ∨ (j = a.q2 ∧ j = b.q2) ∨ (k = a.q3 ∧ k = b.q3) → false)

theorem min_students_for_property_P :
  ∃ (group : List answer), 
  (length group = 8) 
  ∧ property_P group 
  ∧ (∀ (s : answer), ¬property_P (s :: group)) :=
sorry

end min_students_for_property_P_l552_552947


namespace simplify_cube_root_l552_552270

theorem simplify_cube_root (a b c d : ℕ) (h₁ : a = 20) (h₂ : b = 30) (h₃ : c = 40) (h₄ : d = 60) :
  (∛(a^3 + b^3 + c^3 + d^3)) = 10 * ∛315 := by
  sorry

end simplify_cube_root_l552_552270


namespace sections_equidistant_from_center_of_sphere_have_equal_areas_l552_552169

theorem sections_equidistant_from_center_of_sphere_have_equal_areas
  (S : Type) [normed_space ℝ S] {c : S} (r : ℝ)
  (section1 section2: set S)
  (dist_eq_center : ∀ p ∈ section1,∀ q ∈ section2, dist p c = dist q c) :
  measure_theory.measure section1 = measure_theory.measure section2 :=
sorry

end sections_equidistant_from_center_of_sphere_have_equal_areas_l552_552169


namespace glass_pieces_same_color_l552_552723

theorem glass_pieces_same_color (r y b : ℕ) (h : r + y + b = 2002) :
  (∃ k : ℕ, ∀ n, n ≥ k → (r + y + b) = n ∧ (r = 0 ∨ y = 0 ∨ b = 0)) ∧
  (∀ (r1 y1 b1 r2 y2 b2 : ℕ),
    r1 + y1 + b1 = 2002 →
    r2 + y2 + b2 = 2002 →
    (∃ k : ℕ, ∀ n, n ≥ k → (r1 = 0 ∨ y1 = 0 ∨ b1 = 0)) →
    (∃ l : ℕ, ∀ m, m ≥ l → (r2 = 0 ∨ y2 = 0 ∨ b2 = 0)) →
    r1 = r2 ∧ y1 = y2 ∧ b1 = b2):=
by
  sorry

end glass_pieces_same_color_l552_552723


namespace Sarah_won_30_games_l552_552648

namespace TicTacToe

def total_games := 100
def tied_games := 40
def lost_money := 30

def won_games (W : ℕ) (L : ℕ) := 
  (W + 2 * L) = total_games ∧ (W - 2 * L) = (-lost_money)

theorem Sarah_won_30_games : ∃ W L : ℕ, won_games W L ∧ W = 30 :=
by
  sorry

end TicTacToe

end Sarah_won_30_games_l552_552648


namespace weaving_amount_l552_552367

noncomputable theory

-- Definition of the arithmetic sequence
def a (n : ℕ) (d : ℚ) := 5 + (n - 1) * d

-- Total sum S31 of the arithmetic sequence for 31 days
def S31 (d : ℚ) := ∑ i in (finset.range 31), a (i + 1) d

-- Total sum S30 of the arithmetic sequence for 30 days
def S30 (d : ℚ) := ∑ i in (finset.range 30), a (i + 1) d

-- Defining the problem
theorem weaving_amount (d : ℚ) (h : S31 d = 390) :
  (∑ i in (finset.filter (λ x, x % 2 = 0) (finset.range 31)), a (i + 1) d) /
  (∑ i in (finset.filter (λ x, x % 2 = 1) (finset.range 31)), a (i + 1) d) = 16 / 15 :=
sorry

end weaving_amount_l552_552367


namespace sin_value_l552_552095

theorem sin_value (α : ℝ) (h: cos (π / 6 - α) = (sqrt 3)/3) :
  sin (5 * π / 6 - 2 * α) = -1 / 3 :=
sorry

end sin_value_l552_552095


namespace books_sold_on_tuesday_l552_552604

theorem books_sold_on_tuesday (initial_stock : ℕ) (sold_mon : ℕ) (sold_wed : ℕ) (sold_thu : ℕ) (sold_fri : ℕ) (percentage_unsold : ℚ) :
  initial_stock = 700 →
  sold_mon = 50 →
  sold_wed = 60 →
  sold_thu = 48 →
  sold_fri = 40 →
  percentage_unsold = 0.60 →
  let unsold := percentage_unsold * initial_stock in
  let total_sold := initial_stock - unsold in
  let sold_other_days := sold_mon + sold_wed + sold_thu + sold_fri in
  let sold_tue := total_sold - sold_other_days in
  sold_tue = 82 :=
by
  intros
  sorry

end books_sold_on_tuesday_l552_552604


namespace seven_factorial_simplification_l552_552432

theorem seven_factorial_simplification : 7! - 6 * 6! - 6! = 0 := by
  sorry

end seven_factorial_simplification_l552_552432


namespace petya_sum_of_expressions_l552_552242

theorem petya_sum_of_expressions :
  (∑ val in (Finset.univ : Finset (Fin 32)), (1 +
    (if val / 2^4 % 2 = 0 then 2 else -2) +
    (if val / 2^3 % 2 = 0 then 3 else -3) +
    (if val / 2^2 % 2 = 0 then 4 else -4) +
    (if val / 2 % 2 = 0 then 5 else -5) +
    (if val % 2 = 0 then 6 else -6))) = 32 := 
by
  sorry

end petya_sum_of_expressions_l552_552242


namespace sarah_wins_games_l552_552650

variable (total_games : ℕ)
variable (tied_games : ℕ)
variable (total_money_lost : ℤ)

theorem sarah_wins_games
  (h_total_games : total_games = 100)
  (h_tied_games : tied_games = 40)
  (h_total_money_lost : total_money_lost = -30) :
  let won_games := total_games - tied_games - (total_money_lost / -2) in
  won_games = 30 :=
by
  sorry

end sarah_wins_games_l552_552650


namespace percentage_increase_is_20_l552_552173

def number_of_students_this_year : ℕ := 960
def number_of_students_last_year : ℕ := 800

theorem percentage_increase_is_20 :
  ((number_of_students_this_year - number_of_students_last_year : ℕ) / number_of_students_last_year * 100) = 20 := 
by
  sorry

end percentage_increase_is_20_l552_552173


namespace more_boys_than_girls_l552_552182

noncomputable def class1_4th_girls : ℕ := 12
noncomputable def class1_4th_boys : ℕ := 13
noncomputable def class2_4th_girls : ℕ := 15
noncomputable def class2_4th_boys : ℕ := 11

noncomputable def class1_5th_girls : ℕ := 9
noncomputable def class1_5th_boys : ℕ := 13
noncomputable def class2_5th_girls : ℕ := 10
noncomputable def class2_5th_boys : ℕ := 11

noncomputable def total_4th_girls : ℕ := class1_4th_girls + class2_4th_girls
noncomputable def total_4th_boys : ℕ := class1_4th_boys + class2_4th_boys

noncomputable def total_5th_girls : ℕ := class1_5th_girls + class2_5th_girls
noncomputable def total_5th_boys : ℕ := class1_5th_boys + class2_5th_boys

noncomputable def total_girls : ℕ := total_4th_girls + total_5th_girls
noncomputable def total_boys : ℕ := total_4th_boys + total_5th_boys

theorem more_boys_than_girls :
  (total_boys - total_girls) = 2 :=
by
  -- placeholder for the proof
  sorry

end more_boys_than_girls_l552_552182


namespace sqrt_expression_non_negative_l552_552149

theorem sqrt_expression_non_negative (x : ℝ) : 4 + 2 * x ≥ 0 ↔ x ≥ -2 :=
by sorry

end sqrt_expression_non_negative_l552_552149
