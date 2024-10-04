import Mathlib
import Mathlib.Algebra.Arithmetic.Progressions
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Mod
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Probability
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.SetTheory.Game.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace value_of_p_min_value_of_MA_plus_MF_l15_15293

-- Definition of the conditions
def parabola (p : ℝ) : set (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
def focus : (ℝ × ℝ) := (2, 0)
def point_A : (ℝ × ℝ) := (6, 3)
def on_parabola (p : ℝ) (M : ℝ × ℝ) : Prop := M ∈ parabola p

-- Theorem proving the value of p
theorem value_of_p : ∀ (p : ℝ), parabola p = { (x, y) | y^2 = 8 * x } → p = 4 :=
by
  intro p h,
  sorry

-- Theorem proving the minimum value of |MA| + |MF|
theorem min_value_of_MA_plus_MF : 
  ∀ (M : ℝ × ℝ), on_parabola 4 M →
  (8:ℝ) = @nat_abs ℝ _ (dist' M point_A) + @nat_abs ℝ _ (dist' M focus) :=
by
  intro M h,
  sorry

end value_of_p_min_value_of_MA_plus_MF_l15_15293


namespace sum_of_digits_proof_l15_15836

-- Definition of the problem
def is_five_digit_number_with_distinct_nonzero_digits (n : ℕ) : Prop :=
  let digits := (n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10)
  digits.1 ≠ 0 ∧ digits.2 ≠ 0 ∧ digits.3 ≠ 0 ∧ digits.4 ≠ 0 ∧ digits.5 ≠ 0 ∧
  digits.1 ≠ digits.2 ∧ digits.1 ≠ digits.3 ∧ digits.1 ≠ digits.4 ∧ digits.1 ≠ digits.5 ∧
  digits.2 ≠ digits.3 ∧ digits.2 ≠ digits.4 ∧ digits.2 ≠ digits.5 ∧
  digits.3 ≠ digits.4 ∧ digits.3 ≠ digits.5 ∧
  digits.4 ≠ digits.5

def reverse_number (n : ℕ) : ℕ :=
  let digits := (n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10)
  digits.5 * 10000 + digits.4 * 1000 + digits.3 * 100 + digits.2 * 10 + digits.1

-- The main theorem statement
theorem sum_of_digits_proof :
  ∀ (n : ℕ), is_five_digit_number_with_distinct_nonzero_digits n →
  reverse_number (4 * n) = n →
  let digits := (n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10) in
  digits.1 + digits.2 + digits.3 + digits.4 + digits.5 = 27 := by
  sorry

end sum_of_digits_proof_l15_15836


namespace log_base_four_exp_half_sum_l15_15613

theorem log_base_four_exp_half_sum :
  4^(1 / 2) + log 4 (1 / 2) = 3 / 2 :=
by
  sorry

end log_base_four_exp_half_sum_l15_15613


namespace mosel_fills_315_boxes_per_week_l15_15928

-- Definitions for the conditions given in the problem.
def hens : ℕ := 270
def eggs_per_hen_per_day : ℕ := 1
def boxes_capacity : ℕ := 6
def days_per_week : ℕ := 7

-- Objective: Prove that the number of boxes filled each week is 315
theorem mosel_fills_315_boxes_per_week :
  let eggs_per_day := hens * eggs_per_hen_per_day
  let boxes_per_day := eggs_per_day / boxes_capacity
  let boxes_per_week := boxes_per_day * days_per_week
  boxes_per_week = 315 := by
  sorry

end mosel_fills_315_boxes_per_week_l15_15928


namespace least_number_divisible_by_five_primes_l15_15060

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, (n > 0) ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → p ∣ n) ∧ n = 2310 :=
begin
  sorry
end

end least_number_divisible_by_five_primes_l15_15060


namespace stone_velocity_initial_l15_15827

def h : ℝ := 10
def τ : ℝ := 1
def g : ℝ := 10

theorem stone_velocity_initial (V : ℝ) :
  (∃ (t : ℝ), t = 0.5 ∧ (V * t - (1 / 2) * g * t^2 >= h)) ↔ 
  (V ∈ (set.Ioo (85 / 6) (33 / 2)) ∪ (set.Ioo (285 / 14) (45 / 2))) :=
sorry

end stone_velocity_initial_l15_15827


namespace exists_convex_polyhedron_with_n_diagonals_l15_15747

-- Define the concept of a convex polyhedron
structure ConvexPolyhedron where
  vertices : Set ℝ^3
  -- Additional properties such as convexity can be specified as needed

-- Define the concept of a diagonal
def is_diagonal (P : ConvexPolyhedron) (v1 v2 : ℝ^3) : Prop :=
  v1 ∈ P.vertices ∧ v2 ∈ P.vertices ∧ ¬(∃ face, {v1, v2} ⊆ face) ∧ v1 ≠ v2

-- Define the problem statement
theorem exists_convex_polyhedron_with_n_diagonals (n : ℕ) : 
  ∃ P : ConvexPolyhedron, (Finset.filter (λ (pair : (ℝ^3 × ℝ^3)), is_diagonal P pair.1 pair.2) 
    (Finset.cross_product P.vertices.to_finset P.vertices.to_finset)).card = n := by
  sorry

end exists_convex_polyhedron_with_n_diagonals_l15_15747


namespace volume_of_one_piece_l15_15137

def pizzaVolume (d : ℝ) (h : ℝ) (n : ℕ) : ℝ :=
  let r := d / 2
  let V := Real.pi * r^2 * h
  V / n

theorem volume_of_one_piece (d h : ℝ) (n : ℕ) (H1 : d = 18) (H2 : h = 1 / 3) (H3 : n = 18) :
  pizzaVolume d h n = 3 * Real.pi / 2 :=
by
  sorry

end volume_of_one_piece_l15_15137


namespace smallest_a_plus_b_l15_15660

theorem smallest_a_plus_b 
  (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : 2^10 * 3^5 = a^b) : a + b = 248833 :=
sorry

end smallest_a_plus_b_l15_15660


namespace tangent_line_eq_mon_incr_l15_15279

def f (a : ℝ) (x : ℝ) := (1 / x + a) * Real.log (1 + x)

theorem tangent_line_eq (a : ℝ) (h_a : a = -1) :
  ∃ m b : ℝ, (∀ x y, y = (1 / x - 1) * Real.log (1 + x) → y = m * x + b) ∧ m * 1 + b = 0 ∧ m = -Real.log 2 ∧ b = Real.log 2 :=
sorry

theorem mon_incr (a : ℝ) (h_mon : ∀ x : ℝ, 0 < x → (deriv (f a)) x ≥ 0) :
  1 / 2 ≤ a :=
sorry

end tangent_line_eq_mon_incr_l15_15279


namespace goods_train_length_l15_15130

noncomputable def length_of_goods_train (speed_first_train_kmph speed_goods_train_kmph time_seconds : ℕ) : ℝ :=
  let relative_speed_kmph := speed_first_train_kmph + speed_goods_train_kmph
  let relative_speed_mps := (relative_speed_kmph : ℝ) * (5.0 / 18.0)
  relative_speed_mps * (time_seconds : ℝ)

theorem goods_train_length
  (speed_first_train_kmph : ℕ) (speed_goods_train_kmph : ℕ) (time_seconds : ℕ) 
  (h1 : speed_first_train_kmph = 50)
  (h2 : speed_goods_train_kmph = 62)
  (h3 : time_seconds = 9) :
  length_of_goods_train speed_first_train_kmph speed_goods_train_kmph time_seconds = 280 :=
  sorry

end goods_train_length_l15_15130


namespace number_remaining_is_64_l15_15750

def last_remaining_number : ℕ :=
  let numbers : List ℕ := List.range 120 |>.tail!.map (· + 1) -- [1, 2, ..., 120]
  let mark_every_second (nums : List ℕ) : List ℕ :=
    nums.enum.filter (λ (i, _) => i % 2 = 1).map (λ (_, x) => x)
  let rec eliminate (nums : List ℕ) : ℕ :=
    match nums with
    | [x] => x
    | xs => eliminate (mark_every_second xs)
  eliminate numbers

theorem number_remaining_is_64 : last_remaining_number = 64 := 
  by
  sorry

end number_remaining_is_64_l15_15750


namespace inequality_proof_l15_15527

noncomputable theory
open real

theorem inequality_proof {a b c d e f : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_d : 0 < d) (h_pos_e : 0 < e) (h_pos_f : 0 < f) (h_ineq : |sqrt(a * d) - sqrt(b * c)| ≤ 1) :
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) :=
sorry

end inequality_proof_l15_15527


namespace alice_oranges_l15_15149

theorem alice_oranges (E A : ℕ) 
  (h1 : A = 2 * E) 
  (h2 : E + A = 180) : 
  A = 120 :=
by
  sorry

end alice_oranges_l15_15149


namespace exists_point_N_l15_15692

theorem exists_point_N (A B C N : Type) (hABC : ∠B = 90°)
    (hNAB : ∠NAB = 45°) (hNBC : ∠NBC = 45°) (hNCA : ∠NCA = 45°) :
    exists (N : Type), True :=
by
  sorry

end exists_point_N_l15_15692


namespace lim_sum_over_n_pow_l15_15544

theorem lim_sum_over_n_pow (k : ℝ) (h : k > -1) :
  (∃ l : ℝ, tendsto (λ n : ℕ, (∑ i in finset.range (n+1), (i : ℝ) ^ k) / (n : ℝ) ^ (k + 1)) at_top (𝓝 l) ∧ l = 1 / (k + 1)) :=
sorry

end lim_sum_over_n_pow_l15_15544


namespace net_amount_received_correct_l15_15443

def net_amount_received (cash_realized : ℝ) (brokerage_rate : ℝ) : ℝ :=
  let brokerage_paid := (brokerage_rate / 100) * cash_realized
  let brokerage_paid_rounded := Float.ofReal brokerage_paid
  cash_realized - brokerage_paid_rounded

theorem net_amount_received_correct :
  net_amount_received 120.50 0.25 = 120.20 :=
by
  sorry

end net_amount_received_correct_l15_15443


namespace part_a_part_b_l15_15367

-- Definition of n being greater than or equal to 2
def n_ge_2 (n : ℕ) : Prop := n ≥ 2

-- Definition of an n-tuple being expensive
def is_expensive (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, (k > 0) ∧ ((List.prod (List.map (λ i, (a i + a ((i + 1) % n))) (List.range n)) = 2^(2*k - 1)))

-- Statement 1: An expensive n-tuple exists if and only if n is odd
theorem part_a (n : ℕ) (h : n_ge_2 n) : 
  (∃ a : (ℕ → ℕ), is_expensive n a) ↔ (n % 2 = 1) := by
  sorry

-- Statement 2: For every odd positive integer m, there exists an integer n ≥ 2 such that m belongs to an expensive n-tuple.
theorem part_b (m : ℕ) (h : m % 2 = 1) :
  ∃ n : ℕ, n_ge_2 n ∧ ∃ a : (ℕ → ℕ), is_expensive n a ∧ (∃ i : ℕ, (i < n) ∧ (a i = m)) := by
  sorry

end part_a_part_b_l15_15367


namespace train_speed_l15_15878

noncomputable def trainLength : ℕ := 400
noncomputable def timeToCrossPole : ℕ := 20

theorem train_speed : (trainLength / timeToCrossPole) = 20 := by
  sorry

end train_speed_l15_15878


namespace lattice_points_equality_l15_15461

-- Definitions based on the problem statement conditions
def f (t q s : ℕ) : ℕ :=
  ∑ x in Finset.range (t + 1), (Int.floor ((s - 1) * x + t) / q) - (Int.ceil ((s + 1) * x - t) / q) + 1

-- Main theorem to prove
theorem lattice_points_equality (t q r s : ℕ) (hq_div : q ∣ (r * s - 1)) :
  f t q r = f t q s := sorry

end lattice_points_equality_l15_15461


namespace shirt_to_pants_ratio_l15_15979

noncomputable def cost_uniforms
  (pants_cost shirt_ratio socks_price total_spending : ℕ) : Prop :=
  ∃ (shirt_cost tie_cost : ℕ),
    shirt_cost = shirt_ratio * pants_cost ∧
    tie_cost = shirt_cost / 5 ∧
    5 * (pants_cost + shirt_cost + tie_cost + socks_price) = total_spending

theorem shirt_to_pants_ratio 
  (pants_cost socks_price total_spending : ℕ)
  (h1 : pants_cost = 20)
  (h2 : socks_price = 3)
  (h3 : total_spending = 355)
  (shirt_ratio : ℕ)
  (h4 : cost_uniforms pants_cost shirt_ratio socks_price total_spending) :
  shirt_ratio = 2 := by
  sorry

end shirt_to_pants_ratio_l15_15979


namespace part1_tangent_line_at_x_eq_1_part2_monotonic_increasing_l15_15273

noncomputable def f (x : ℝ) (a : ℝ) := (1/x + a) * real.log(1 + x)

theorem part1_tangent_line_at_x_eq_1 (x := 1) :
  let a := -1 
  let t : ℝ := real.log 2
  let f := (λ x, (1/x - 1) * real.log (1 + x))
  let tangent_line := λ x y, t * x + y - t 
  (tangent_line 1 (f 1) = 0) :=
by {
  sorry
}

theorem part2_monotonic_increasing (a : ℝ) : 
  (∀ x : ℝ, 0 < x → 0 < f x a) ↔ (a ≥ 1/2) := 
by {
  sorry
}

end part1_tangent_line_at_x_eq_1_part2_monotonic_increasing_l15_15273


namespace problem_statement_l15_15525

noncomputable def root_exists (x_list : Fin 2020 → ℝ) (a : ℝ) : Prop :=
  ∃ x ∈ Icc 0 4, (Finset.univ.sum (λ i, |x - x_list i|) = 2020 * a)

theorem problem_statement (x_list : Fin 2020 → ℝ) 
  (h : ∀ i, x_list i ∈ Icc 0 4) : 
  root_exists x_list 2 :=
sorry

end problem_statement_l15_15525


namespace alec_string_ways_l15_15922

theorem alec_string_ways :
  let letters := ['A', 'C', 'G', 'N']
  let num_ways := 24 * 2 * 2
  num_ways = 96 := 
by
  sorry

end alec_string_ways_l15_15922


namespace total_sales_15_days_l15_15617

def edgar_sales (n : ℕ) : ℕ := 3 * n - 1

def clara_sales (n : ℕ) : ℕ := 4 * n

def edgar_total_sales (d : ℕ) : ℕ := (d * (2 + (d * 3 - 1))) / 2

def clara_total_sales (d : ℕ) : ℕ := (d * (4 + (d * 4))) / 2

def total_sales (d : ℕ) : ℕ := edgar_total_sales d + clara_total_sales d

theorem total_sales_15_days : total_sales 15 = 810 :=
by
  sorry

end total_sales_15_days_l15_15617


namespace minimum_toothpicks_to_remove_l15_15635

def hexagonal_grid (toothpicks : ℕ) (hexagons : ℕ) : Prop :=
  toothpicks = 48 ∧ hexagons ≥ 1

theorem minimum_toothpicks_to_remove (toothpicks hexagons removal_needed : ℕ) (h : hexagonal_grid toothpicks hexagons) :
  removal_needed = 6 :=
begin
  sorry
end

end minimum_toothpicks_to_remove_l15_15635


namespace remainder_of_17_pow_65_mod_7_l15_15862

theorem remainder_of_17_pow_65_mod_7 :
  17 ^ 65 % 7 = 5 :=
by
  have h1 : 17 % 7 = 3 := rfl
  have h2 : 3 ^ 6 % 7 = 1 := rfl
  have h3 : 17 ^ 65 % 7 = (3 ^ 65) % 7
  have powered_value : 3 ^ 65 % 7 = 3 ^ (6 * 10 + 5) % 7
  have cycle_use : (3 ^ 6) ^ 10 * 3 ^ 5 % 7 = 1 ^ 10 * 3 ^ 5 % 7
  have final_value : 1 ^ 10 * 3 ^ 5 % 7 = 5
  exact final_value

end remainder_of_17_pow_65_mod_7_l15_15862


namespace line_through_fixed_point_l15_15686

-- Definitions and conditions from part (1)
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
def pointD : ℝ × ℝ := (4, 4)
def p_positive (p : ℝ) := p > 0
def parabola_through_D (p : ℝ) : Prop := parabola p pointD.1 pointD.2

-- Definitions and conditions from part (2)
def line_l (m b x y : ℝ) := x = m * y + b
def intersects_parabola (l : ℝ → ℝ → Prop) (p : ℝ) (P Q : ℝ × ℝ) :=
  l P.1 P.2 ∧ parabola p P.1 P.2 ∧ l Q.1 Q.2 ∧ parabola p Q.1 Q.2 ∧ P ≠ Q
def midpoint (A B E : ℝ × ℝ) := A.1 = 0.5 * (B.1 + E.1) ∧ A.2 = 0.5 * (B.2 + E.2)

-- Final theorem combining both parts
theorem line_through_fixed_point (p m b : ℝ) :
  p_positive p →
  parabola_through_D p →
  (∀ E F : ℝ × ℝ, intersects_parabola (line_l m b) p E F → 
    ∃ A B : ℝ × ℝ, A.1 = E.1 ∧ A.2 = E.1 ∧ midpoint A B E → 
    line_l m b A.1 A.2 ∧ (A.1 = 0 → A.2 = 2)) :=
sorry

end line_through_fixed_point_l15_15686


namespace least_number_divisible_by_five_primes_l15_15066

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, (n > 0) ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → p ∣ n) ∧ n = 2310 :=
begin
  sorry
end

end least_number_divisible_by_five_primes_l15_15066


namespace falling_time_of_earth_l15_15705

open Real

-- Definitions based on the conditions:
variables (γ : ℝ) -- gravitational constant
variables (M : ℝ) -- mass of the Sun
variables (m : ℝ) -- mass of the Earth
variables (R : ℝ) -- initial distance between the Earth and the Sun
variables (T : ℝ := 1) -- orbital period of the Earth, 1 year

-- Statement of Kepler's third law condition
def keplers_third_law (T R T_e R_e : ℝ) : Prop :=
  (T^2) / (T_e^2) = (R^3) / (R_e^3)

-- Statement of the equivalence of gravitational potential energy and kinetic energy during fall:
def energy_equivalence (γ M m R x : ℝ) (v : ℝ → ℝ): Prop :=
  γ * m * M * (1/(R - x) - 1/R) = (1/2) * m * (v x)^2

-- Final statement to be proved
theorem falling_time_of_earth (γ M m R : ℝ) (hγ : 0 < γ) (hM : 0 < M) (hm : 0 < m) (hR : 0 < R):
  let T := 1 in
  let R_e := R / 2 in
  let T_e := T * (sqrt (1 / 8)) in
  T_e / 2 = 64.5 / 365 :=
sorry

end falling_time_of_earth_l15_15705


namespace min_max_value_l15_15962

-- Definition of the function to be minimized and maximized
def f (x y : ℝ) : ℝ := |x^3 - x * y^2|

-- Conditions
def x_condition (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def y_condition (y : ℝ) : Prop := true

-- Goal: Prove the minimum of the maximum value
theorem min_max_value :
  ∃ y : ℝ, (∀ x : ℝ, x_condition x → f x y ≤ 8) ∧ (∀ y' : ℝ, (∀ x : ℝ, x_condition x → f x y' ≤ 8) → y' = y) :=
sorry

end min_max_value_l15_15962


namespace part_a_part_b_l15_15879

-- Definitions for part (a)
def regular_pentagon (A B C D E : Type) : Prop := sorry

def intersection_of_diagonals (A B C D E : Type) (K : Type) : Prop := sorry

def circumcircle_tangent (K C E : Type) (BC : Type) : Prop := sorry

-- Definition and theorem statement for part (a)
theorem part_a (A B C D E K : Type)
  [regular_pentagon A B C D E]
  [intersection_of_diagonals A B C D E K] :
  circumcircle_tangent K C E BC :=
sorry

-- Definitions for part (b)
def side_length (a : ℝ) (A B C D E : Type) : Prop := sorry

def diagonal_length (d : ℝ) (A B C D E : Type) : Prop := sorry

-- Theorem statement for part (b)
theorem part_b (a d : ℝ) (A B C D E : Type)
  [side_length a A B C D E]
  [diagonal_length d A B C D E] :
  d^2 = a^2 + a * d :=
sorry

end part_a_part_b_l15_15879


namespace range_m_inequality_l15_15289

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem range_m (m : ℝ) :
  (∀ x : ℝ, f x ≤ m * x ∧ m * x ≤ g x) ↔ m ∈ Icc (1 / (2 * Real.exp 1)) (Real.exp 1) :=
by sorry

theorem inequality (x1 x2 : ℝ) (h₁ : x1 > x2) (h₂ : x2 > 0) :
  (x1 * f x1 - x2 * f x2) * (x1^2 + x2^2) > 2 * x2 * (x1 - x2) :=
by sorry

end range_m_inequality_l15_15289


namespace exists_n_sum_three_digit_identical_digit_l15_15358

theorem exists_n_sum_three_digit_identical_digit:
  ∃ (n : ℕ), (∃ (k : ℕ), (k ≥ 1 ∧ k ≤ 9) ∧ (n*(n+1)/2 = 111*k)) ∧ n = 36 :=
by
  -- Placeholder for the proof
  sorry

end exists_n_sum_three_digit_identical_digit_l15_15358


namespace profit_percentage_B_is_25_l15_15139

-- Define the conditions as constants and parameters
def cp_A : ℝ := 144
def sp_C : ℝ := 225
def profit_percent_A : ℝ := 25

-- Calculate selling price of A to B (sp_B)
def sp_B := cp_A * (1 + profit_percent_A / 100)

-- Calculate profit B made when selling to C
def profit_B := sp_C - sp_B

-- Calculate profit percentage B made
def profit_percent_B := (profit_B / sp_B) * 100

-- Prove that profit percentage B is 25%
theorem profit_percentage_B_is_25 : profit_percent_B = 25 := by
  sorry

end profit_percentage_B_is_25_l15_15139


namespace volume_of_full_pyramid_l15_15437

variables (S1 S2 V : ℝ)
hypothesis (h1 : S1 < S2)

theorem volume_of_full_pyramid (h : S1 < S2) :
  let Vfull := (V * S2 * Real.sqrt S2) / (S2 * Real.sqrt S2 - S1 * Real.sqrt S1)
  in Vfull = (V * S2 * Real.sqrt S2) / (S2 * Real.sqrt S2 - S1 * Real.sqrt S1) := sorry

end volume_of_full_pyramid_l15_15437


namespace magnitude_of_a_l15_15302

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := 
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a (n : ℝ) (h_perpendicular : (1, n) • (-1, n) = 0) : 
  magnitude (1, n) = Real.sqrt 2 :=
by
  sorry

end magnitude_of_a_l15_15302


namespace find_a_plus_b_l15_15250

theorem find_a_plus_b (a b : ℝ) :
  (∀ x y : ℝ, (ax + y + 1 = 0) ∧ (2x - by - 1 = 0) → (x = 1 → y = 1)) → (a + b = -1) :=
by
  intros
  sorry

end find_a_plus_b_l15_15250


namespace profit_difference_l15_15826

-- Setting up the conditions
def construction_cost_others (C : ℝ) : ℝ := C

def construction_cost_certain (C : ℝ) : ℝ := C + 100000

def selling_price_others : ℝ := 320000

def selling_price_certain : ℝ := 1.5 * 320000

def profit_certain (C : ℝ) : ℝ := selling_price_certain - construction_cost_certain C

def profit_others (C : ℝ) : ℝ := selling_price_others - construction_cost_others C

-- Proving the difference in profit
theorem profit_difference (C : ℝ) : profit_certain C - profit_others C = 60000 :=
by
    simp [profit_certain, profit_others, selling_price_certain, selling_price_others, construction_cost_certain, construction_cost_others]
    ring
    sorry

end profit_difference_l15_15826


namespace intersection_on_incircle_l15_15224

-- Definitions based on the condition
structure Triangle := 
(A B C : Point)
(O : Point) -- The center of the incircle

structure Line := 
(p1 p2 : Point) -- A line passing through two points

def perpendicular_foot (p : Point) (l : Line) : Point := sorry
-- Function to get the foot of the perpendicular from point p onto line l

def diametrically_opposite (p : Point) (C : Circle) (e : ExistsOnCircle p C) : Point := sorry
-- Function to get the diametrically opposite point on the given circle

-- The main theorem to be proved
theorem intersection_on_incircle
    (ABC : Triangle)
    (l : Line)
    (O_incircle_center : l.p1 = ABC.O)
    (C_incircle : Circle)
    (A1 B1 C1 : Point)
    (hA1 : A1 = perpendicular_foot ABC.A l)
    (hB1 : B1 = perpendicular_foot ABC.B l)
    (hC1 : C1 = perpendicular_foot ABC.C l)
    (A2 B2 C2 : Point)
    (hA2 : A2 = diametrically_opposite A1 C_incircle sorry)
    (hB2 : B2 = diametrically_opposite B1 C_incircle sorry)
    (hC2 : C2 = diametrically_opposite C1 C_incircle sorry) :
    ∃ P : Point, (P ∈ C_incircle) ∧
    collinear P A1 A2 ∧ 
    collinear P B1 B2 ∧ 
    collinear P C1 C2 := sorry

end intersection_on_incircle_l15_15224


namespace least_N_for_A_success_l15_15781

noncomputable def game_proof_problem : Prop :=
  ∃ (N : ℕ), N = 4022 ∧ 
  ∀ (boxes : fin 2012 → ℕ), (∀ (i : fin 2012), boxes i ≥ 1) →
  (λ (moveB : fin 2012 → fin 2012), moveB = λ i, (i + 1) % 2012) ∧
  (λ (moveA : fin 2012 → fin 2012), moveA = λ i, (i % 2012)) →
  (∀ (moves : ℕ ), ∃ (A_play : fin (moves + 1) → Prop), (∀ i < moves, A_play i) → ∀ j < 2012, boxes j ≥ 1)

-- We state the main theorem we want to prove:
theorem least_N_for_A_success : game_proof_problem :=
sorry

end least_N_for_A_success_l15_15781


namespace sin_eq_cos_sufficient_for_cos2theta_zero_l15_15372

theorem sin_eq_cos_sufficient_for_cos2theta_zero (θ : ℝ) :
  (sin θ = cos θ) → (cos (2 * θ) = 0) ∧ ¬(cos (2 * θ) = 0 → sin θ = cos θ) :=
by
  sorry

end sin_eq_cos_sufficient_for_cos2theta_zero_l15_15372


namespace least_number_divisible_by_five_smallest_primes_l15_15000

theorem least_number_divisible_by_five_smallest_primes : 
  ∃ n ∈ ℕ+, n = 2 * 3 * 5 * 7 * 11 ∧ n = 2310 :=
by
  sorry

end least_number_divisible_by_five_smallest_primes_l15_15000


namespace trains_meet_in_8p78_seconds_l15_15097

noncomputable def time_for_trains_to_meet (distance_between_trains : ℝ) 
    (length_first_train : ℝ) (length_second_train : ℝ) 
    (speed_first_train_kmph : ℝ) (speed_second_train_kmph : ℝ) : ℝ :=
  let total_distance := distance_between_trains + length_first_train + length_second_train
  let relative_speed_mps := (speed_first_train_kmph + speed_second_train_kmph) * 1000 / 3600
  total_distance / relative_speed_mps

theorem trains_meet_in_8p78_seconds :
  time_for_trains_to_meet 200 90 100 71 89 ≈ 8.78 := 
begin
  sorry
end

end trains_meet_in_8p78_seconds_l15_15097


namespace males_do_not_listen_l15_15946

theorem males_do_not_listen (males_do_not_listen : ℕ) (females_listen : ℕ) (total_listen : ℕ) (total_do_not_listen : ℕ) :
  males_do_not_listen = 85 :=
by {
  -- Introduce the hypothesis
  have h1 : males_do_not_listen = 85 := by sorry,
  -- Direct proof
  exact h1,
}

end males_do_not_listen_l15_15946


namespace no_prime_satisfies_base_conversion_l15_15433

theorem no_prime_satisfies_base_conversion :
  ∀ p : ℕ, prime p →
    (2 * p^3 + 8 * p^2 + 5 * p + 20 = p^3 + 10 * p^2 + 7 * p + 8) →
    false :=
by
  sorry

end no_prime_satisfies_base_conversion_l15_15433


namespace find_center_radius_l15_15444

noncomputable def circle_center_radius (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y - 6 = 0 → 
  ∃ (h k r : ℝ), (x + 1) * (x + 1) + (y - 2) * (y - 2) = r ∧ h = -1 ∧ k = 2 ∧ r = 11

theorem find_center_radius :
  circle_center_radius x y :=
sorry

end find_center_radius_l15_15444


namespace sin_alpha_l15_15212

noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 2 - Real.pi / 4)

theorem sin_alpha (α : ℝ) (h : f α = 1 / 3) : Real.sin α = -7 / 9 :=
by 
  sorry

end sin_alpha_l15_15212


namespace parallelepiped_tangent_to_sphere_is_cube_l15_15788

-- Define the conditions under which our theorem holds.
structure Parallelepiped (P : Type) :=
  (edges_tangent_to_sphere : Prop)

-- Define the cube structure as a specific type of Parallelepiped.
structure Cube (C : Type) extends Parallelepiped C :=
  (is_cube : Prop)

-- Statement of the theorem in Lean
theorem parallelepiped_tangent_to_sphere_is_cube {P : Type} (p : Parallelepiped P) :
  p.edges_tangent_to_sphere → ∃ (c : Cube P), c.is_cube :=
begin
  assume h,
  sorry -- Proof is omitted as per instructions
end

end parallelepiped_tangent_to_sphere_is_cube_l15_15788


namespace car_speed_is_112_5_l15_15550

-- Define the constants
def distance : ℝ := 1 -- distance in kilometers
def speed_120 : ℝ := 120 -- speed in km/h for the reference car
def time_120 : ℝ := distance / speed_120 -- time to travel 1 km at 120 km/h in hours
def time_120_seconds : ℝ := time_120 * 3600 -- time in seconds

-- Define the additional time in seconds
def extra_time_seconds : ℝ := 2 -- extra time in seconds

-- Total time taken by the car to travel 1 km in seconds
def total_time_seconds : ℝ := time_120_seconds + extra_time_seconds

-- Convert the total time back to hours
def total_time_hours : ℝ := total_time_seconds / 3600

-- Define the speed of the car in km/h
def car_speed : ℝ := distance / total_time_hours

-- The theorem stating the speed of the car
theorem car_speed_is_112_5 : car_speed = 112.5 := by
  sorry

end car_speed_is_112_5_l15_15550


namespace triangle_perimeter_correct_l15_15746

noncomputable def triangle_perimeter (PQ QR PR mP mQ mR : ℝ) : ℝ :=
  if hPQ : PQ = 150 ∧ QR = 270 ∧ PR = 210 ∧ mP = 75 ∧ mQ = 60 ∧ mR = 30 
  then 239.314 
  else 0

theorem triangle_perimeter_correct : 
  ∀ (PQ QR PR mP mQ mR : ℝ),
  PQ = 150 → QR = 270 → PR = 210 → mP = 75 → mQ = 60 → mR = 30 →
  triangle_perimeter PQ QR PR mP mQ mR = 239.314 :=
by
  intros PQ QR PR mP mQ mR hPQ hQR hPR hmP hmQ hmR
  unfold triangle_perimeter
  rw [if_pos (and.intro hPQ (and.intro hQR (and.intro hPR (and.intro hmP (and.intro hmQ hmR)))))]
  rfl

end triangle_perimeter_correct_l15_15746


namespace possible_values_a_l15_15237

theorem possible_values_a (a : ℝ) :
  let A := {x : ℝ | x^2 = 1}
  let B := {x : ℝ | a * x = 1}
  B ⊆ A → a = -1 ∨ a = 1 :=
by
  intro h
  sorry

end possible_values_a_l15_15237


namespace change_received_l15_15463

theorem change_received (price_wooden_toy : ℕ) (price_hat : ℕ) (money_paid : ℕ) (num_wooden_toys : ℕ) (num_hats : ℕ) : 
  price_wooden_toy = 20 → price_hat = 10 → money_paid = 100 → num_wooden_toys = 2 → num_hats = 3 → 
  money_paid - (num_wooden_toys * price_wooden_toy + num_hats * price_hat) = 30 := 
by 
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5] 
  simp 
  norm_num 
  sorry

end change_received_l15_15463


namespace prob_exactly_M_laws_in_concept_l15_15713

theorem prob_exactly_M_laws_in_concept 
  (K N M : ℕ) (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let q := 1 - (1 - p)^N in
  (nat.choose K M) * q^M * (1 - q)^(K - M) = 
  (nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M) :=
by {
  let q := 1 - (1 - p)^N,
  have hq_def : q = 1 - (1 - p)^N := rfl,
  rw [hq_def],
  sorry
}

end prob_exactly_M_laws_in_concept_l15_15713


namespace quadratic_roots_real_coeff_l15_15659

theorem quadratic_roots_real_coeff (p q : ℝ) :
  (∀ (x : ℂ), (x = 2 + complex.i ∨ x = 2 - complex.i) → x^2 + p * x + q = 0) →
  p = -4 ∧ q = 5 :=
by
  sorry

end quadratic_roots_real_coeff_l15_15659


namespace smallest_lambda_inequality_l15_15631

theorem smallest_lambda_inequality 
  (a b c d : ℝ) (h_pos : ∀ x ∈ [a, b, c, d], 0 < x) (h_sum : a + b + c + d = 4) :
  5 * (a*b + a*c + a*d + b*c + b*d + c*d) ≤ 8 * (a*b*c*d) + 12 :=
sorry

end smallest_lambda_inequality_l15_15631


namespace area_enclosed_within_fence_l15_15568

theorem area_enclosed_within_fence :
  (let
      length := 20
      width := 18
      cutout1 := 4
      cutout2 := 2
      area_rectangle := length * width
      area_cutout1 := cutout1 * cutout1
      area_cutout2 := cutout2 * cutout2
      area_enclosed := area_rectangle - area_cutout1 - area_cutout2
    in
      area_enclosed = 340) :=
by
  sorry

end area_enclosed_within_fence_l15_15568


namespace count_permutation_multiples_of_13_l15_15313

/-- Define the range of integers being considered. -/
def range_100_to_999 := { n : ℕ | 100 ≤ n ∧ n ≤ 999 }

/-- Define multiples of 13 in the given range. -/
def multiples_of_13 := { n : ℕ | n ∈ range_100_to_999 ∧ 13 ∣ n }

/-- Check if some permutation of a number's digits is a multiple of 13. -/
def has_permutation_multiple_of_13 (n : ℕ) : Prop :=
  ∃ m : ℕ, (some_permutation_of_digits_eq n m) ∧ m ∈ multiples_of_13

/-- The final count of numbers in the range such that some permutation of its digits is a multiple of 13. -/
theorem count_permutation_multiples_of_13 : 
  { n : ℕ | n ∈ range_100_to_999 ∧ has_permutation_multiple_of_13 n }.to_finset.card = 195 := 
sorry

end count_permutation_multiples_of_13_l15_15313


namespace inequality_proof_l15_15541

theorem inequality_proof (a b c d e f : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
    (hcond : abs (sqrt (a * d) - sqrt (b * c)) ≤ 1) :
    (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := 
  sorry

end inequality_proof_l15_15541


namespace intersecting_lines_sum_l15_15248

theorem intersecting_lines_sum (a b : ℝ) 
  (h1 : a * 1 + 1 + 1 = 0)
  (h2 : 2 * 1 - b * 1 - 1 = 0) : 
  a + b = -1 := 
by 
  have ha : a = -2 := by linarith [h1]
  have hb : b = 1 := by linarith [h2]
  rw [ha, hb]
  exact by norm_num

end intersecting_lines_sum_l15_15248


namespace average_weight_decrease_proof_l15_15440

theorem average_weight_decrease_proof :
  ∀ (A1 A2 : ℝ),
  (A1 = 57) →
  (A2 = (57 * 20 + 48) / (20 + 1)) →
  A1 - A2 = 0.43 :=
by
  intros A1 A2 hA1 hA2
  rw [←hA1, hA2]
  sorry

end average_weight_decrease_proof_l15_15440


namespace prob_exactly_M_laws_in_concept_expected_laws_in_concept_l15_15725

section Anchuria
variables (K N M : ℕ) (p : ℝ)

-- Part (a): Define P_M as the binomial probability distribution result
def probability_exactly_M_laws : ℝ :=
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M)

-- Part (a): Prove the result for the probability that exactly M laws are included in the Concept
theorem prob_exactly_M_laws_in_concept :
  probability_exactly_M_laws K N M p =
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) := 
sorry

-- Part (b): Define the expected number of laws included in the Concept
def expected_number_of_laws : ℝ :=
  K * (1 - (1 - p) ^ N)

-- Part (b): Prove the result for the expected number of laws included in the Concept
theorem expected_laws_in_concept :
  expected_number_of_laws K N p = K * (1 - (1 - p) ^ N) :=
sorry
end Anchuria

end prob_exactly_M_laws_in_concept_expected_laws_in_concept_l15_15725


namespace least_positive_whole_number_divisible_by_five_primes_l15_15010

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l15_15010


namespace value_of_expression_l15_15406

variable (x y : ℝ)

theorem value_of_expression (h1 : x + y = 6) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 228498 := by
  sorry

end value_of_expression_l15_15406


namespace other_solution_of_quadratic_l15_15242

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop :=
  65 * x^2 - 104 * x + 31 = 0

-- Main theorem statement
theorem other_solution_of_quadratic :
  quadratic_eq (6 / 5) → quadratic_eq (5 / 13) :=
by
  intro h
  sorry

end other_solution_of_quadratic_l15_15242


namespace pet_food_cost_is_correct_l15_15753

-- Define the given conditions
def rabbit_toy_cost := 6.51
def cage_cost := 12.51
def total_cost := 24.81
def found_dollar := 1.00

-- Define the cost of pet food
def pet_food_cost := total_cost - (rabbit_toy_cost + cage_cost) + found_dollar

-- The statement to prove
theorem pet_food_cost_is_correct : pet_food_cost = 6.79 :=
by
  -- proof steps here
  sorry

end pet_food_cost_is_correct_l15_15753


namespace pets_bought_satisfy_conditions_l15_15132

theorem pets_bought_satisfy_conditions (puppies kittens hamsters rabbits : ℕ)
  (different_pet_per_person : 4 = 4) :
  puppies = 12 → kittens = 10 → hamsters = 8 → rabbits = 5 →
  (∑ a in Finset.univ (Fin 4), different_pet_per_person * (puppies * kittens * hamsters * rabbits)) = 115200 :=
by
  intros h_puppies h_kittens h_hamsters h_rabbits
  have h1 : puppies = 12 := h_puppies
  have h2 : kittens = 10 := h_kittens
  have h3 : hamsters = 8 := h_hamsters
  have h4 : rabbits = 5 := h_rabbits
  sorry

end pets_bought_satisfy_conditions_l15_15132


namespace minimum_function_value_l15_15971

theorem minimum_function_value :
  ∃ (x y : ℕ), 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 3 ∧
  (∀ x' y', 0 ≤ x' ∧ x' ≤ 2 → 0 ≤ y' ∧ y' ≤ 3 →
  (x^2 * y^2 : ℝ) / ((x^2 + y^2)^2 : ℝ) ≤ (x'^2 * y'^2 : ℝ) / ((x'^2 + y'^2)^2 : ℝ)) ∧
  (x = 0 ∨ y = 0) ∧ ((x^2 * y^2 : ℝ) / ((x^2 + y^2)^2 : ℝ) = 0) :=
by
  --; Implementation of the theorem would follow
  sorry

end minimum_function_value_l15_15971


namespace min_value_of_f_l15_15869

noncomputable def f (x : ℝ) : ℝ := (cos x)^2 / (2 * cos x * sin x - (sin x)^2)

theorem min_value_of_f : ∀ x : ℝ, 0 < x ∧ x < π / 3 → f(x) ≥ 1 := 
by 
  intro x hx,
  sorry

end min_value_of_f_l15_15869


namespace large_cube_surface_area_l15_15856

-- Definition of the small cube
def small_cube_edge_length : ℝ := 4
def number_of_small_cubes : ℕ := 27

-- Definition of the large cube
def large_cube_edge_length := (number_of_small_cubes^(1/3 : ℝ)) * small_cube_edge_length

-- Surface area of a cube formula
def surface_area_of_cube (edge_length : ℝ) : ℝ := 6 * edge_length^2

-- The theorem stating the surface area of the large cube
theorem large_cube_surface_area : surface_area_of_cube large_cube_edge_length = 864 := 
by
  sorry

end large_cube_surface_area_l15_15856


namespace problem_1_problem_2_problem_3_l15_15669

noncomputable def a (n : ℕ) : ℝ := if n = 0 then 0 else 2^(n-2)

def b (n : ℕ) : ℝ := 2*n - 1

def c (n : ℕ) : ℝ := a n * b n

noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, c (i + 1)

theorem problem_1 (n : ℕ) (h : n > 0) : a n = 2^(n-2) :=
sorry

theorem problem_2 (n : ℕ) : T n = ∑ i in Finset.range n, c (i + 1) :=
sorry

theorem problem_3 {λ k : ℝ} (hλ : λ > 0) (n : ℕ) : 
  a (2 * n) * (2 * λ^2 - k * λ + 2) > b n → k < 2 * λ + 1 / λ :=
sorry

end problem_1_problem_2_problem_3_l15_15669


namespace proof_problem_l15_15682

noncomputable def f (x : ℝ) : ℝ := x^2 * log x

-- Proof definitions
lemma statement_A (x : ℝ) (hx1 : x > 1) : f x > 0 :=
sorry

lemma statement_A' (x : ℝ) (hx2 : 0 < x ∧ x < 1) : f x < 0 :=
sorry

lemma statement_C : set.range f = { y | -1 / (2 * real.exp 1) <= y } :=
sorry

lemma statement_D (x : ℝ) : f x >= x - 1 :=
sorry

-- Combined statement to match the final problem request
theorem proof_problem :
  (∀ x > 1, f x > 0) ∧ (∀ x, 0 < x ∧ x < 1 → f x < 0) ∧ 
  (set.range f = { y | -1 / (2 * real.exp 1) <= y }) ∧ 
  (∀ x, f x >= x - 1) :=
by
  exact ⟨statement_A, statement_A', statement_C, statement_D⟩

end proof_problem_l15_15682


namespace money_distribution_l15_15921

theorem money_distribution (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 360) (h3 : C = 60) : A + B + C = 500 := by
  sorry

end money_distribution_l15_15921


namespace new_ratio_l15_15880

def milk_to_water_initial_ratio (M W : ℕ) : Prop := 4 * W = M

def total_volume (V M W : ℕ) : Prop := V = M + W

def new_water_volume (W_new W A : ℕ) : Prop := W_new = W + A

theorem new_ratio (V M W W_new A : ℕ) 
  (h1: milk_to_water_initial_ratio M W) 
  (h2: total_volume V M W) 
  (h3: A = 23) 
  (h4: new_water_volume W_new W A) 
  (h5: V = 45) 
  : 9 * W_new = 8 * M :=
by 
  sorry

end new_ratio_l15_15880


namespace sin_double_angle_sub_pi_four_l15_15639

theorem sin_double_angle_sub_pi_four (α : ℝ) (h : Real.tan (α - π / 4) = 2) : 
  Real.sin (2 * α - π / 4) = sqrt 2 / 10 := 
by 
  sorry

end sin_double_angle_sub_pi_four_l15_15639


namespace probability_even_sum_cards_l15_15330

theorem probability_even_sum_cards (cards : Finset ℕ) (h : cards = {1, 2, 3, 3, 4}) :
  let outcomes := (cards.card choose 2) in
  let evens := {2, 4} in
  let odds := {1, 3, 3} in
  let favorable := (evens.card choose 2) + (odds.card choose 2) in
  (favorable : ℚ) / outcomes = 2 / 5 :=
by
  sorry

end probability_even_sum_cards_l15_15330


namespace intersecting_lines_sum_l15_15247

theorem intersecting_lines_sum (a b : ℝ) 
  (h1 : a * 1 + 1 + 1 = 0)
  (h2 : 2 * 1 - b * 1 - 1 = 0) : 
  a + b = -1 := 
by 
  have ha : a = -2 := by linarith [h1]
  have hb : b = 1 := by linarith [h2]
  rw [ha, hb]
  exact by norm_num

end intersecting_lines_sum_l15_15247


namespace coefficient_x3_in_expansion_l15_15628

theorem coefficient_x3_in_expansion : 
  let coeff := (λ (n r : ℕ) => (2^r) * (Nat.choose n r))
  coeff 6 2 = 60 :=
by
  let coeff := (λ (n r : ℕ) => (2 ^ r) * (Nat.choose n r))
  have h : coeff 6 2 = 4 * (Nat.choose 6 2) := by rfl
  have h2 : (Nat.choose 6 2) = 15 := by sorry -- Here we will use properties of choose function
  show coeff 6 2 = 60 from sorry

end coefficient_x3_in_expansion_l15_15628


namespace vector_FB_l15_15351

   variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
   variables (a b : V)

   def E_midpoint_CD (C D : V) := midpoint ℝ C D
   def F_midpoint_AE (A E : V) := midpoint ℝ A E

   theorem vector_FB (A B C D E F : V) (hE : E = E_midpoint_CD C D) (hF : F = F_midpoint_AE A E) 
     (hAB : B - A = a) (hAD : D - A = b) : 
     F - B = - (3 / 4 : ℝ) • a + (1 / 2 : ℝ) • b := 
   by
     sorry
   
end vector_FB_l15_15351


namespace find_largest_n_l15_15199

theorem find_largest_n :
    ∃ n ∈ Finset.range 21, ∀ k ∈ finset.range (n+21), k^2 ∣ Finset.prod (Finset.range (n+21)) id → n = 20 :=
begin
    sorry
end

end find_largest_n_l15_15199


namespace pollutant_reduction_time_l15_15841

theorem pollutant_reduction_time (P0 : ℝ) (k t : ℝ) 
  (h_rel: ∀ t, P0 * exp (-k * t) = P0 * (1 - (1 / exp (k * 2)))) :
  t = 6 :=
by
  sorry

end pollutant_reduction_time_l15_15841


namespace exists_integers_gcd_eq_one_addition_l15_15755

theorem exists_integers_gcd_eq_one_addition 
  (n k : ℕ) 
  (hnk_pos : n > 0 ∧ k > 0) 
  (hn_even_or_nk_even : (¬ n % 2 = 0) ∨ (n % 2 = 0 ∧ k % 2 = 0)) :
  ∃ a b : ℤ, Int.gcd a ↑n = 1 ∧ Int.gcd b ↑n = 1 ∧ k = a + b :=
by
  sorry

end exists_integers_gcd_eq_one_addition_l15_15755


namespace coeff_x_squared_is_16_l15_15821

open Nat

noncomputable def coeff_x_squared := (∑ k in finset.range 5, (binom 4 k * (pow (-2) k) * (k = 1)) * (1 : ℚ))

theorem coeff_x_squared_is_16 :
  coeff_x_squared = 16 :=
by
  -- Coefficient of x^2 term calculation.
  sorry

end coeff_x_squared_is_16_l15_15821


namespace find_exponents_l15_15477

noncomputable def sum_of_divisors (n: ℕ): ℕ :=
  (List.range (n + 1)).filter (λ d, n % d = 0).sum

theorem find_exponents (i j : ℕ) (n : ℕ) (h1 : n = 2^i * 5^j)
  (h2 : sum_of_divisors n = 930) : i + j = 5 :=
sorry

end find_exponents_l15_15477


namespace periodic_sequence_l15_15990

def x (a : ℝ) : ℕ → ℝ
| 0     := a
| 1     := 1
| (n+2) := (x n) ^ 2 - 5 / (x (n - 1))

theorem periodic_sequence (a : ℝ) :
  (∀ n, x a (n + 4) = x a n) ↔ 
  (a = -2 ∨ a = 2 ∨ a = - (2 / 3) * Real.sqrt 6 ∨
     a = (2 / 3) * Real.sqrt 6) :=
by
  sorry

end periodic_sequence_l15_15990


namespace nest_building_twig_count_l15_15548

theorem nest_building_twig_count
    (total_twigs_to_weave : ℕ)
    (found_twigs : ℕ)
    (remaining_twigs : ℕ)
    (n : ℕ)
    (x : ℕ)
    (h1 : total_twigs_to_weave = 12 * x)
    (h2 : found_twigs = (total_twigs_to_weave) / 3)
    (h3 : remaining_twigs = 48)
    (h4 : found_twigs + remaining_twigs = total_twigs_to_weave) :
    x = 18 := 
by
  sorry

end nest_building_twig_count_l15_15548


namespace prob_exactly_M_laws_in_concept_l15_15714

theorem prob_exactly_M_laws_in_concept 
  (K N M : ℕ) (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let q := 1 - (1 - p)^N in
  (nat.choose K M) * q^M * (1 - q)^(K - M) = 
  (nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M) :=
by {
  let q := 1 - (1 - p)^N,
  have hq_def : q = 1 - (1 - p)^N := rfl,
  rw [hq_def],
  sorry
}

end prob_exactly_M_laws_in_concept_l15_15714


namespace find_s_l15_15629

theorem find_s : ∃ s : ℚ, (∀ x : ℚ, (3 * x^2 - 8 * x + 9) * (5 * x^2 + s * x + 15) = 15 * x^4 - 71 * x^3 + 174 * x^2 - 215 * x + 135) ∧ s = -95 / 9 := sorry

end find_s_l15_15629


namespace common_point_exists_l15_15382

variables {A B C P D E F Q : Point}
variables {ω_A ω_B ω_C ω : Circle}
variables {m_A m_B m_C : Line}

-- Conditions
axiom h1 : P ∈ circumscribed_circle(A, B, C)
axiom h2 : D = reflect P (midline_parallel_to BC)
axiom h3 : E = reflect P (midline_parallel_to CA)
axiom h4 : F = reflect P (midline_parallel_to AB)
axiom h5 : ω_A = circumscribed_circle(A, D, P)
axiom h6 : ω_B = circumscribed_circle(B, E, P)
axiom h7 : ω_C = circumscribed_circle(C, F, P)
axiom h8 : ω = circumscribed_circle(perpendicular_bisector AD, perpendicular_bisector BE, perpendicular_bisector CF)

-- Theorem
theorem common_point_exists : ∃ Q, Q ∈ ω_A ∧ Q ∈ ω_B ∧ Q ∈ ω_C ∧ Q ∈ ω := 
sorry

end common_point_exists_l15_15382


namespace sum_of_sin_squared_l15_15942

theorem sum_of_sin_squared (s : Fin 30 → Real) (hs : ∀ i, s i = (sin (3 * (i + 1) : Nat) * sin (3 * (i + 1) : Nat)).toReal) :
  (∑ i in Finset.range 30, s i) = 30 :=
by
  sorry

end sum_of_sin_squared_l15_15942


namespace tank_capacity_l15_15918

theorem tank_capacity (
  (fill_rate : ℝ) (drain_rate1 : ℝ) (drain_rate2 : ℝ) :
  fill_rate = 0.5 →
  drain_rate1 = 0.25 →
  drain_rate2 = 0.1667 →
  let net_flow_rate := fill_rate - (drain_rate1 + drain_rate2) in
  let time_taken := 48 in
  let amount_added := net_flow_rate * time_taken in
  (amount_added = 4) →
  let initial_capacity := amount_added in
  (2 * initial_capacity = 8))
:=
begin
  intros hf hd1 hd2,
  sorry,
end

end tank_capacity_l15_15918


namespace initial_fliers_l15_15512

variable (F : ℕ) -- Initial number of fliers

-- Conditions
axiom morning_send : F - (1 / 5) * F = (4 / 5) * F
axiom afternoon_send : (4 / 5) * F - (1 / 4) * ((4 / 5) * F) = (3 / 5) * F
axiom final_count : (3 / 5) * F = 600

theorem initial_fliers : F = 1000 := by
  sorry

end initial_fliers_l15_15512


namespace distinction_not_possible_l15_15357

-- Definitions of the types of problems
structure ProofProblem := 
  (description : String)
  (proof : String)

structure CalculationProblem := 
  (description : String)
  (calculation : String)

structure ConstructionProblem := 
  (description : String)
  (construction : String)

-- The main statement, expressing that it is not possible to clearly distinguish between the types
theorem distinction_not_possible 
  (pp : ProofProblem)
  (cp : CalculationProblem)
  (csp : ConstructionProblem) :
  ¬ (∀ pp, ∃ cp, ∃ csp, 
        (pp.description = cp.description) ∧ 
        (cp.description = csp.description) ∧ 
        (proof ∧ calculation ∧ construction are distinct)):
  sorry

end distinction_not_possible_l15_15357


namespace price_reduction_equation_l15_15120

variable (x : ℝ)

theorem price_reduction_equation 
    (original_price : ℝ)
    (final_price : ℝ)
    (two_reductions : original_price * (1 - x) ^ 2 = final_price) :
    100 * (1 - x) ^ 2 = 81 :=
by
  sorry

end price_reduction_equation_l15_15120


namespace sign_of_c_l15_15701

theorem sign_of_c (a b c : ℝ) (h1 : (a * b / c) < 0) (h2 : (a * b) < 0) : c > 0 :=
sorry

end sign_of_c_l15_15701


namespace circle_has_correct_center_and_radius_l15_15447

noncomputable def circleEquation := ∀ x y : ℝ, x^2 + y^2 - 2 * x = 0
noncomputable def circleCenter := (1 : ℝ, 0 : ℝ)
noncomputable def circleRadius := (1 : ℝ)

theorem circle_has_correct_center_and_radius :
  circleEquation = (x - 1)^2 + y^2 = 1 ∧ 
  circleCenter = (1 : ℝ, 0 : ℝ) ∧ 
  circleRadius = 1 :=
begin
  sorry
end

end circle_has_correct_center_and_radius_l15_15447


namespace probability_point_in_square_l15_15915

theorem probability_point_in_square
  (R : ℝ) : 
  let S := π * R^2,
      a := sqrt 2 * R,
      S1 := (sqrt 2 * R)^2 in
  (S1 / S) = (2 / π) := by
  sorry

end probability_point_in_square_l15_15915


namespace solve_problem_l15_15485

theorem solve_problem :
  (∃ f : ℕ → ℕ, ∀ k, (2 * f k + 5 * (1 + 2 * k) = 207)) ∧
  (∀ k m, (2 * f k + 5 * (1 + 2 * k) = 207) →
           (2 * f m + 5 * (1 + 2 * m) = 207) →
           k = m) ∧
  (∃ N : ℕ, ∀ k, k < 21 → (2 * f k + 5 * (1 + 2 * k) = 207)) :=
begin
  sorry
end

end solve_problem_l15_15485


namespace relationship_l15_15992

variable (x y a b c : ℝ)

-- Given conditions
def condition1 : a = x + y := sorry
def condition2 : b = x * y := sorry
def condition3 : c = x^2 + y^2 := sorry

-- To prove
theorem relationship : a^2 = c + 2 * b :=
by
  -- Assume the given conditions
  assume h1 : a = x + y,
  assume h2 : b = x * y,
  assume h3 : c = x^2 + y^2,
  sorry

end relationship_l15_15992


namespace price_of_ice_cream_bar_is_correct_l15_15895

noncomputable def price_ice_cream_bar (n_ice_cream_bars n_sundaes total_price price_of_sundae price_ice_cream_bar : ℝ) : Prop :=
  n_ice_cream_bars = 125 ∧
  n_sundaes = 125 ∧
  total_price = 225 ∧
  price_of_sundae = 1.2 →
  price_ice_cream_bar = 0.6

theorem price_of_ice_cream_bar_is_correct :
  price_ice_cream_bar 125 125 225 1.2 0.6 :=
by
  sorry

end price_of_ice_cream_bar_is_correct_l15_15895


namespace area_triangle_ABC_l15_15215

variables (A B C D O: Type)
variables [Triangle A B C] [Quadrilateral A B C D]
variables (AB BC CD AD AC: ℝ) (alpha: ℝ)

-- Given conditions
variable (h1: AB = BC)
variable (h2: AB = AD + CD)
variable (h3: ∠A B D = alpha)
variable (h4: AC = d)

-- To prove
theorem area_triangle_ABC:
  area (triangle A B C) = (1 / 2) * d^2 * sin(alpha) := 
sorry

end area_triangle_ABC_l15_15215


namespace at_least_two_inequalities_hold_l15_15768

variable {a b c : ℝ}

theorem at_least_two_inequalities_hold (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c ≥ a * b * c) :
  (2 / a + 3 / b + 6 / c ≥ 6 ∨ 2 / b + 3 / c + 6 / a ≥ 6) ∨
  (2 / b + 3 / c + 6 / a ≥ 6 ∨ 2 / c + 3 / a + 6 / b ≥ 6) ∨
  (2 / c + 3 / a + 6 / b ≥ 6 ∨ 2 / a + 3 / b + 6 / c ≥ 6) :=
  sorry

end at_least_two_inequalities_hold_l15_15768


namespace inequality_proof_l15_15531

variables (a b c d e f : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
variable (hcond : |sqrt (a * d) - sqrt (b * c)| ≤ 1)

theorem inequality_proof :
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end inequality_proof_l15_15531


namespace alice_oranges_proof_l15_15146

-- Definitions for conditions
def oranges_emily_sold (E : ℕ) := E
def oranges_alice_sold (E : ℕ) := 2 * E
def total_oranges_sold (E : ℕ) := E + 2 * E

-- Proof statement
theorem alice_oranges_proof : ∀ E : ℕ, total_oranges_sold E = 180 → oranges_alice_sold E = 120 :=
begin
  intros E h,
  sorry
end

end alice_oranges_proof_l15_15146


namespace find_c_for_given_radius_l15_15632

theorem find_c_for_given_radius (c : ℝ) : (∃ x y : ℝ, (x^2 - 2 * x + y^2 + 6 * y + c = 0) ∧ ((x - 1)^2 + (y + 3)^2 = 25)) → c = -15 :=
by
  sorry

end find_c_for_given_radius_l15_15632


namespace g_no_zeros_pos_l15_15665

variable {ℝ : Type*} [Real ℝ]

-- Define f as a continuous and differentiable function on ℝ
variable (f : ℝ → ℝ)
variable h_cont_diff : ∀ x : ℝ, ContinuousAt f x ∧ Differentiable ℝ f

-- Define the condition x f'(x) + f(x) > 0 for all x
variable h_ineq : ∀ x : ℝ, x * (deriv f x) + f x > 0

-- Define g(x) = x * f(x) + 1
def g (x : ℝ) : ℝ := x * f x + 1

-- Prove that g(x) has no zeros for x > 0
theorem g_no_zeros_pos : ∀ x > 0, g x ≠ 0 :=
by
  sorry

end g_no_zeros_pos_l15_15665


namespace quadratic_equation_root_zero_l15_15319

/-- Given that x = -3 is a root of the quadratic equation x^2 + 3x + k = 0,
    prove that the other root of the equation is 0 and k = 0. -/
theorem quadratic_equation_root_zero (k : ℝ) (h : -3^2 + 3 * -3 + k = 0) :
  (∀ t : ℝ, t^2 + 3 * t + k = 0 → t = 0) ∧ k = 0 :=
sorry

end quadratic_equation_root_zero_l15_15319


namespace pr_is_qr_squared_l15_15316

theorem pr_is_qr_squared (α β : ℝ) (r s p q : ℝ)
    (h1 : ∀ x, x^2 - rx + s = 0 → x = Real.tan α ∨ x = Real.tan β)
    (h2 : ∀ x, x^2 - px + q = 0 → x = Real.cot α ∨ x = Real.cot β) :
  pr = qr^2 :=
by
  sorry

end pr_is_qr_squared_l15_15316


namespace least_positive_divisible_by_five_primes_l15_15040

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l15_15040


namespace range_of_x_l15_15210

theorem range_of_x (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y ∧ y ≤ 7) :
  -1 ≤ x ∧ x < 5 / 4 :=
sorry

end range_of_x_l15_15210


namespace Q1_Q2_l15_15435

noncomputable def prob_A_scores_3_out_of_4 (p_A_serves : ℚ) (p_A_scores_A_serves: ℚ) (p_A_scores_B_serves: ℚ) : ℚ :=
  by
    -- Placeholder probability function
    sorry

theorem Q1 (p_A_serves : ℚ := 2/3) (p_A_scores_A_serves: ℚ := 2/3) (p_A_scores_B_serves: ℚ := 1/2) :
  prob_A_scores_3_out_of_4 p_A_serves p_A_scores_A_serves p_A_scores_B_serves = 1/3 :=
  by
    -- Proof of the theorem
    sorry

noncomputable def prob_X_lessthan_or_equal_4 (p_A_serves: ℚ) (p_A_scores_A_serves: ℚ) (p_A_scores_B_serves: ℚ) : ℚ :=
  by
    -- Placeholder probability function
    sorry

theorem Q2 (p_A_serves: ℚ := 2/3) (p_A_scores_A_serves: ℚ := 2/3) (p_A_scores_B_serves: ℚ := 1/2) :
  prob_X_lessthan_or_equal_4 p_A_serves p_A_scores_A_serves p_A_scores_B_serves = 3/4 :=
  by
    -- Proof of the theorem
    sorry

end Q1_Q2_l15_15435


namespace speed_of_current_is_correct_l15_15116

def upstream_rate := 1 / 20 -- km/min
def downstream_rate := 1 / 15 -- km/min
def min_to_hour := 60 -- 60 minutes per hour

noncomputable def speed_upstream := upstream_rate * min_to_hour -- km/h
noncomputable def speed_downstream := downstream_rate * min_to_hour -- km/h
noncomputable def speed_of_current := (speed_downstream - speed_upstream) / 2 -- result in km/h

theorem speed_of_current_is_correct : speed_of_current = 0.5 := by
  sorry

end speed_of_current_is_correct_l15_15116


namespace least_positive_number_divisible_by_primes_l15_15030

theorem least_positive_number_divisible_by_primes :
  ∃ n : ℕ, n > 0 ∧
    (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧
    (∀ m : ℕ, (m > 0 ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m)) → n ≤ m) ∧
    n = 2310 :=
by
  sorry

end least_positive_number_divisible_by_primes_l15_15030


namespace mn_value_l15_15322

theorem mn_value (m n : ℤ) (h1 : m + n = 1) (h2 : m - n + 2 = 1) : m * n = 0 := 
by 
  sorry

end mn_value_l15_15322


namespace pq_parallel_ab_l15_15812

variables {A B C D O P Q E : Point}
variables {circle : Circle}
variables [Quadrilateral ABCD : circumscribed circle]

-- Define the problem in Lean 4

theorem pq_parallel_ab (h1: O ∈ line AB)     -- center of circumcircle lies on AB
                    (h2: E = sym_point D AB) -- E is symmetric to D wrt AB
                    (h3: line_segment_inter AC DO P) -- AC and DO intersect at P
                    (h4: line_segment_inter BD CE Q) -- BD and CE intersect at Q
                    : parallel line PQ AB    -- PQ is parallel to AB
                    := 
sorry

end pq_parallel_ab_l15_15812


namespace alice_bob_meet_l15_15145

theorem alice_bob_meet (n : ℕ) (h_n : n = 18) (alice_move : ℕ) (bob_move : ℕ)
  (h_alice : alice_move = 7) (h_bob : bob_move = 13) :
  ∃ k : ℕ, alice_move * k % n = (n - bob_move) * k % n :=
by
  sorry

end alice_bob_meet_l15_15145


namespace find_dihedral_angle_tetrahedron_l15_15353

noncomputable def tetrahedron_angle_face (A B C D : Point) 
(angle_ACB : ∠ A C B = 90) 
(angle_CAD : ∠ C A D = 90) 
(eq_CA_CB : dist A C = dist B C) 
(eq_CA_half_AD : dist A C = dist (A D) / 2) 
(perp_CD_AB : line_perp (A B) (C D)) 
: real :=
  let angle_ACB_ACD := 60 in -- Correct answer for the angle between faces ACB and ACD
  angle_ACB_ACD

theorem find_dihedral_angle_tetrahedron (A B C D : Point)
  (angle_ACB : ∠ A C B = 90)
  (angle_CAD : ∠ C A D = 90)
  (eq_CA_CB : dist A C = dist B C)
  (eq_CA_half_AD : dist A C = dist (A D) / 2)
  (perp_CD_AB : line_perp (A B) (C D)) :
  tetrahedron_angle_face A B C D angle_ACB angle_CAD eq_CA_CB eq_CA_half_AD perp_CD_AB = 60 :=
by
  sorry

end find_dihedral_angle_tetrahedron_l15_15353


namespace total_profit_l15_15488

theorem total_profit (Tom_investment Jose_investment : ℝ) (Tom_months Jose_months : ℕ) (Jose_share : ℝ)
  (h1 : Tom_investment = 30000) 
  (h2 : Jose_investment = 45000) 
  (h3 : Tom_months = 12) 
  (h4 : Jose_months = 10) 
  (h5 : Jose_share = 15000) : 
  let Tom_total := Tom_investment * Tom_months
      Jose_total := Jose_investment * Jose_months
      ratio := Tom_total / Jose_total
      one_part := Jose_share / (Jose_total / Tom_total + 1)
      total_parts := (Jose_total / Tom_total) + 1
      total_profit := total_parts * one_part 
  in total_profit = 27000 :=
by 
  sorry

end total_profit_l15_15488


namespace range_of_a_l15_15684

def f (x a : ℝ) : ℝ := -x^3 + x^2 + x + a
def g (x a : ℝ) : ℝ := 2 * a - x^3

theorem range_of_a (a : ℝ) : (∀ x ∈ set.Icc (0 : ℝ) (1 : ℝ), g x a ≥ f x a) → a ≥ 2 :=
begin
  sorry
end

end range_of_a_l15_15684


namespace max_digit_d_divisible_by_33_l15_15188

theorem max_digit_d_divisible_by_33 (d e : ℕ) (h₀ : 0 ≤ d ∧ d ≤ 9) (h₁ : 0 ≤ e ∧ e ≤ 9) 
  (h₂ : d + e = 4) : d ≤ 4 :=
by {
  sorry
}

example : ∃ d e : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧ d + e = 4 ∧ 
(d = 4) :=
by {
  use [4, 0],
  repeat { split },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial }
}

end max_digit_d_divisible_by_33_l15_15188


namespace compute_g_f_1_l15_15377

def f (x : ℝ) : ℝ := x^3 - 2 * x + 3
def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem compute_g_f_1 : g (f 1) = 3 :=
by
  sorry

end compute_g_f_1_l15_15377


namespace wendy_initial_flowers_l15_15858

theorem wendy_initial_flowers (wilted: ℕ) (bouquets_made: ℕ) (flowers_per_bouquet: ℕ) (flowers_initially_picked: ℕ):
  wilted = 35 →
  bouquets_made = 2 →
  flowers_per_bouquet = 5 →
  flowers_initially_picked = wilted + bouquets_made * flowers_per_bouquet →
  flowers_initially_picked = 45 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end wendy_initial_flowers_l15_15858


namespace find_n_l15_15383

theorem find_n (x y : ℤ) (h_x : x = 3) (h_y : y = -1) : 
    let n := x - y^(x + y)
    in n = 2 :=
by
  sorry

end find_n_l15_15383


namespace evaluate_fraction_sum_l15_15183

theorem evaluate_fraction_sum : (5 / 50) + (4 / 40) + (6 / 60) = 0.3 :=
by
  sorry

end evaluate_fraction_sum_l15_15183


namespace least_positive_number_divisible_by_primes_l15_15034

theorem least_positive_number_divisible_by_primes :
  ∃ n : ℕ, n > 0 ∧
    (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧
    (∀ m : ℕ, (m > 0 ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m)) → n ≤ m) ∧
    n = 2310 :=
by
  sorry

end least_positive_number_divisible_by_primes_l15_15034


namespace inequality_proof_l15_15534

theorem inequality_proof
  {a b c d e f : ℝ}
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (h_abs : |sqrt (a * d) - sqrt (b * c)| ≤ 1) :
  (a * e + b / e) * (c * e + d / e) ≥ 
    (a^2 * f^2 - (b^2) / (f^2)) * ((d^2) / (f^2) - c^2 * f^2) :=
by
  sorry

end inequality_proof_l15_15534


namespace least_number_divisible_by_five_smallest_primes_l15_15004

theorem least_number_divisible_by_five_smallest_primes : 
  ∃ n ∈ ℕ+, n = 2 * 3 * 5 * 7 * 11 ∧ n = 2310 :=
by
  sorry

end least_number_divisible_by_five_smallest_primes_l15_15004


namespace relationship_x_y_z_l15_15983

open Real

noncomputable def ln_sq (x : ℝ) : ℝ := (ln x) ^ 2

theorem relationship_x_y_z
  (α : ℝ) (b : ℝ) (x y z : ℝ)
  (h1 : ln x = ln_sq (sin α) / ln b)
  (h2 : ln y = ln_sq (cos α) / ln b)
  (h3 : ln z = ln_sq (sin α * cos α) / ln b)
  (hα : α ∈ set.Ioo (π / 4) (π / 2))
  (hb : b ∈ set.Ioo 0 1) :
  x > y ∧ y > z := 
sorry

end relationship_x_y_z_l15_15983


namespace pq_parallel_ab_l15_15811

variables {A B C D O P Q E : Point}
variables {circle : Circle}
variables [Quadrilateral ABCD : circumscribed circle]

-- Define the problem in Lean 4

theorem pq_parallel_ab (h1: O ∈ line AB)     -- center of circumcircle lies on AB
                    (h2: E = sym_point D AB) -- E is symmetric to D wrt AB
                    (h3: line_segment_inter AC DO P) -- AC and DO intersect at P
                    (h4: line_segment_inter BD CE Q) -- BD and CE intersect at Q
                    : parallel line PQ AB    -- PQ is parallel to AB
                    := 
sorry

end pq_parallel_ab_l15_15811


namespace margie_drive_distance_l15_15390

theorem margie_drive_distance
  (miles_per_gallon : ℕ)
  (cost_per_gallon : ℕ)
  (dollar_amount : ℕ)
  (h₁ : miles_per_gallon = 32)
  (h₂ : cost_per_gallon = 4)
  (h₃ : dollar_amount = 20) :
  (dollar_amount / cost_per_gallon) * miles_per_gallon = 160 :=
by
  sorry

end margie_drive_distance_l15_15390


namespace least_positive_whole_number_divisible_by_five_primes_l15_15016

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l15_15016


namespace work_done_by_resistance_l15_15598

theorem work_done_by_resistance:
  ∀ (x t : ℝ) (F v : ℝ → ℝ),
    (∀ t, x = 4 * t^2) →
    (∀ v, F v = k * v) →
    (v = 10 → F v = 2) →
    ∫ (0 : ℝ) (2 : ℝ), - (4 / 5) * x^0.5 = - (16 / 15) := 
by
  intros x t F v h1 h2 h3
  sorry

end work_done_by_resistance_l15_15598


namespace ab_eq_bd_iff_de_eq_ac_l15_15365

variables (A B C E D : Type) [InnerProductSpace ℝ A]
variables (cyclic_AEBC : CyclicQuadrilateral A B E C)
variable (point_on_ray : D ∈ ray A E)
variable (outside_circumference : D ∉ circumcircle A B E C)
variable (angle_condition : ∠CAB = ∠BAE)

theorem ab_eq_bd_iff_de_eq_ac (AB BD DE AC CAB BAE : Real) :
  (AB = BD ↔ DE = AC) :=
by
  sorry

end ab_eq_bd_iff_de_eq_ac_l15_15365


namespace price_of_second_variety_per_kg_l15_15804

/-
Problem: Prove that the price of the second variety per kg is Rs. 135 given the following conditions:
- Price of the first variety per kg is Rs. 126.
- Price of the third variety per kg is Rs. 173.5.
- The three varieties are mixed in the ratio 1 : 1 : 2.
- The mixture is worth Rs. 152 per kg.

The total mixture weighs 4 kg.
-/

theorem price_of_second_variety_per_kg :
  ∀ (price_first price_third price_mixture total_weight: ℝ),
    price_first = 126 ∧ price_third = 173.5 ∧ price_mixture = 152 ∧ total_weight = 4 →
    ∃ x, (price_first * 1 + x * 1 + price_third * 2) / total_weight = price_mixture ∧ x = 135 :=
by
  intros price_first price_third price_mixture total_weight h
  have : price_first * 1 + x * 1 + price_third * 2 = 608 := sorry
  rw [←mul_eq_mul_right_iff] at this
  exists 135
  exact sorry
#align price_of_second_variety_per_kg price_of_second_variety_per_kg

end price_of_second_variety_per_kg_l15_15804


namespace expectation_exponential_distribution_l15_15198

-- Define the pdf of an exponentially distributed random variable
def exp_pdf (α : ℝ) (x : ℝ) : ℝ := 
  if x < 0 then 0 else α * real.exp (-α * x)

-- Define the expected value of an exponentially distributed random variable
noncomputable def expectation_exp (α : ℝ) : ℝ :=
  ∫ (x : ℝ), (x * exp_pdf α x)

-- Theorem statement
theorem expectation_exponential_distribution (α : ℝ) (hα : 0 < α) :
  expectation_exp α = 1 / α :=
sorry

end expectation_exponential_distribution_l15_15198


namespace A_inter_B_is_correct_l15_15105

def set_A : Set ℤ := { x : ℤ | x^2 - x - 2 ≤ 0 }
def set_B : Set ℤ := { x : ℤ | True }

theorem A_inter_B_is_correct : set_A ∩ set_B = { -1, 0, 1, 2 } := by
  sorry

end A_inter_B_is_correct_l15_15105


namespace range_of_mn_l15_15799
noncomputable def solve_problem : set ℝ :=
  let a := sqrt 3 / 2
  let b := 1 in
    {x | a ≤ x ∧ x ≤ b}

theorem range_of_mn:
  ∀ (M N : ℝ) (x: ℝ) (AM FN MN AC BF : ℝ),
  (square ABCD) ∧ (square ABEF) ∧ (lie_plane_angle ABCD ABEF 120) ∧
  (points_diagonal M AC) ∧ (points_diagonal N BF) ∧
  (AM = x) ∧ (FN = x) ∧ (AB = 1) →
  MN ∈ solve_problem :=
sorry

end range_of_mn_l15_15799


namespace dance_party_participants_l15_15848

theorem dance_party_participants :
  ∃ n m : ℕ, n + m = 42 ∧ n = 18 ∧ m = 24 ∧ (∀ k, 1 ≤ k ∧ k ≤ n → True) :=
begin
  sorry
end

end dance_party_participants_l15_15848


namespace trajectory_equation_cases_l15_15741

-- Variables and assumptions
variables {x y λ : ℝ}
variable (h_λ_pos : λ > 0)
variable (h_eq : (x^2 - 2) * λ^2 = x^2 - 2 + y^2)

-- The main theorem
theorem trajectory_equation_cases :
  (λ = 1 → y = 0) ∧ 
  (0 < λ ∧ λ < 1 → (x^2 / 2 + y^2 / (2 * (1 - λ^2)) = 1)) ∧ 
  (λ > 1 → (x^2 / 2 - y^2 / (2 * (λ^2 - 1)) = 1)) :=
sorry

end trajectory_equation_cases_l15_15741


namespace original_cost_lunch_l15_15159

theorem original_cost_lunch 
  {x : ℝ} 
  (regulatory_fee : x * 0.05)
  (sales_tax : x * 0.08)
  (tip : x * 0.18)
  (total_cost : x + x * 0.05 + x * 0.08 + x * 0.18 = 40) :
  x = 30.53 :=
by
  have h1 : 1.31 * x = 40 := by
    rw [← total_cost]
    ring
  have h2 : x = 40 / 1.31 := by
    rw [← h1]
    norm_num
  exact h2.symm

end original_cost_lunch_l15_15159


namespace minimum_colors_needed_l15_15854

def paint_fence_colors (B : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, B i ≠ B (i + 2)) ∧
  (∀ i : ℕ, B i ≠ B (i + 3)) ∧
  (∀ i : ℕ, B i ≠ B (i + 5))

theorem minimum_colors_needed : ∃ (c : ℕ), 
  (∀ B : ℕ → ℕ, paint_fence_colors B → c ≥ 3) ∧
  (∃ B : ℕ → ℕ, paint_fence_colors B ∧ c = 3) :=
sorry

end minimum_colors_needed_l15_15854


namespace common_tangent_line_number_of_common_tangents_l15_15297

noncomputable def eqn1 : ℝ → ℝ → ℝ :=
  λ x y, x^2 + y^2 + 2*x - 4*y + 4

noncomputable def eqn2 : ℝ → ℝ → ℝ :=
  λ x y, x^2 + y^2 - 4*x - 12*y + 4

theorem common_tangent_line (x y : ℝ) :
  eqn1 x y = 0 → eqn2 x y = 0 → (3 * x + 4 * y = 0) :=
  by {
    sorry
  }

theorem number_of_common_tangents :
  (∃ x y : ℝ, eqn1 x y = 0 ∧ eqn2 x y = 0) → 1 :=
  by {
    sorry
  }

end common_tangent_line_number_of_common_tangents_l15_15297


namespace num_square_tiles_l15_15894

theorem num_square_tiles (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 3 * a + 4 * b + 5 * c = 100) : b = 10 :=
  sorry

end num_square_tiles_l15_15894


namespace correct_propositions_l15_15925

variable (a b : Type) [LinearOrder a] (l : Type) [LinearOrder l] (α β: Type)
variable [Plane α β] (line_a line_b : l → Type) (proj : b → α)

-- Definitions for the propositions
def proposition_1 (a b : Type) [LinearOrder a] [LinearOrder b] : Prop :=
  ¬ ∃ x : l, x ∈ (a ∩ b)

def proposition_2 (l : Type) [LinearOrder l] (α : Type) [Plane α] : Prop :=
  (∀ (x : l → α) , perp l x) = perp l α

def proposition_3 (a b : Type) [LinearOrder a] [LinearOrder b] (α : Type) [Plane α] (proj : b → α) : Prop :=
  perp a (proj b α)

def proposition_4 (a : Type) [LinearOrder a] (β : Type) [Plane β] : Prop :=
  (∀ (x : a → β), para a x) = (∃ (y : l), para a β)

-- Proof problem stating which propositions are correct
theorem correct_propositions :
  proposition_2 l α ∧ proposition_4 l β :=
by sorry

end correct_propositions_l15_15925


namespace chips_reverse_impossible_l15_15889

-- Defining the allowed operations and the positions
def swapable (i : ℕ) (j : ℕ) : Prop :=
  (i + 2 = j ∨ j + 2 = i)

-- Define positions
def positions := {i : ℕ | 1 ≤ i ∧ i ≤ 100}

-- The main theorem statement
theorem chips_reverse_impossible : ¬ ∃ f : ℕ → ℕ,
  (∀ i ∈ positions, f i ∈ positions) ∧
  (∀ i j ∈ positions, swapable i j → swapable (f i) (f j)) ∧
  (∀ i ∈ positions, f i = 101 - i) := 
sorry

end chips_reverse_impossible_l15_15889


namespace circumcircle_perpendicular_to_three_circles_l15_15486

theorem circumcircle_perpendicular_to_three_circles
  (A B C A₁ B₁ C₁ : Point)
  (circle₁ : Circle A₁)
  (circle₂ : Circle B₁)
  (circle₃ : Circle C₁)
  (h1 : tangent (circle₁) (circle₂) A)
  (h2 : tangent (circle₂) (circle₃) B)
  (h3 : tangent (circle₃) (circle₁) C) :
  is_perpendicular (circumcircle (triangle A B C)) circle₁ ∧
  is_perpendicular (circumcircle (triangle A B C)) circle₂ ∧
  is_perpendicular (circumcircle (triangle A B C)) circle₃ := 
sorry

end circumcircle_perpendicular_to_three_circles_l15_15486


namespace store_discount_difference_l15_15141

theorem store_discount_difference 
  (p : ℝ) -- original price
  (p1 : ℝ := p * 0.60) -- price after initial discount
  (p2 : ℝ := p1 * 0.90) -- price after additional discount
  (claimed_discount : ℝ := 0.55) -- store's claimed discount
  (true_discount : ℝ := (p - p2) / p) -- calculated true discount
  (difference : ℝ := claimed_discount - true_discount)
  : difference = 0.09 :=
sorry

end store_discount_difference_l15_15141


namespace extra_profit_is_60000_l15_15823

theorem extra_profit_is_60000 (base_house_cost special_house_cost base_house_price special_house_price : ℝ) :
  (special_house_cost = base_house_cost + 100000) →
  (special_house_price = 1.5 * base_house_price) →
  (base_house_price = 320000) →
  (special_house_price - base_house_price - 100000 = 60000) :=
by
  -- Definitions and conditions
  intro h1 h2 h3
  -- Placeholder for the eventual proof
  sorry

end extra_profit_is_60000_l15_15823


namespace division_by_power_of_ten_l15_15104

theorem division_by_power_of_ten (a b : ℕ) (h_a : a = 10^7) (h_b : b = 5 * 10^4) : a / b = 200 := by
  sorry

end division_by_power_of_ten_l15_15104


namespace max_voters_after_t_l15_15589

theorem max_voters_after_t
  (n : ℕ) (x : ℕ) (h₁ : 1 ≤ n) (h₂ : x ≤ 10)
  (h_rating : ∃ y : ℕ, (nx + y) / (n + 1) = x - 1) :
  (∃ k : ℕ, k = 5) :=
begin
  sorry
end

end max_voters_after_t_l15_15589


namespace floor_ceiling_expression_l15_15181

theorem floor_ceiling_expression :
  (Int.floor (Real.ceil ((15 / 8)^2) + (19 / 5)^2)) = 18 :=
by
  have h1 : (15 / 8)^2 = 225 / 64 := by norm_num
  have h2 : Real.ceil (225 / 64) = 4 := by
    rw [div_eq_mul_inv, ← Real.to_rat_cast, ← of_rat_div (225 : ℚ) 64],
    have h_le : 225 / 64 < 4 := by norm_num,
    have h_gt : 225 / 64 > 3 := by norm_num,
    exact Real.ceil_eq_of_le (by num.cast_le_iff h_le) (by num.cast_le_iff h_gt)
  have h3 : (19 / 5)^2 = 361 / 25 := by norm_num
  calc
  Int.floor (Real.ceil ((15 / 8)^2) + (19 / 5)^2) 
      = Int.floor (4 + (361 / 25)) : by rw [h1, h2, h3]
  ... = Int.floor ((4 * 25 + 361) / 25) : by rw [← add_div]
  ... = 18 : by norm_num

end floor_ceiling_expression_l15_15181


namespace angle_BDC_l15_15326

theorem angle_BDC (a b : ℝ) (θ : ℝ) : 
  (a > 0 ∧ a < 180) → (b > 0 ∧ b < 180) →
  (a + b + 2*θ = 180) → 
  (θ = (180 - a - b)/2) → 
  ∃ c : ℝ, c = (a + b)/2 :=
by
  intros ha hb hsum hθ
  use (a + b) / 2
  sorry

end angle_BDC_l15_15326


namespace groupB_is_a_set_l15_15873

-- Define the group conditions explicitly
def groupA : Set ℕ := {x | abs (x - 8) < 1}
def groupB : Set ℕ := {x | x < 5 ∧ x % 2 = 0}
def groupC : Set (string × string) := {p | p.1 = "High school freshmen" ∧ p.2 = "good at basketball"}
def groupD : Set ℤ := {x | x < -1}

-- The statement asserting groupB forms a set
theorem groupB_is_a_set : IsSet (groupB) := by
  -- we assume groupB is a valid set because its elements are well-defined
  sorry

end groupB_is_a_set_l15_15873


namespace least_positive_whole_number_divisible_by_five_primes_l15_15017

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l15_15017


namespace hyperbola_eccentricity_l15_15290

variables (a b c e : ℝ) (a_pos : a > 0) (b_pos : b > 0)
          (c_eq : c = 4) (b_eq : b = 2 * Real.sqrt 3)
          (hyperbola_eq : c ^ 2 = a ^ 2 + b ^ 2)
          (projection_cond : 2 < (4 * (a * Real.sqrt (a ^ 2 + b ^ 2)) / (Real.sqrt (a ^ 2 + b ^ 2))) ∧ (4 * (a * Real.sqrt (a ^ 2 + b ^ 2)) / (Real.sqrt (a ^ 2 + b ^ 2))) ≤ 4)

theorem hyperbola_eccentricity : e = c / a := 
by
  sorry

end hyperbola_eccentricity_l15_15290


namespace range_of_a_l15_15689

variable (a : ℝ)
def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 12 → x^2 - a ≥ 0
def q : Prop := ∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0

theorem range_of_a (hpq : p a ∨ q a) (hpnq : ¬p a ∧ ¬q a) : 
  (-1 ≤ a ∧ a ≤ 1) ∨ (a > 3) :=
sorry

end range_of_a_l15_15689


namespace tetrahedron_ratio_l15_15710

variables {Point : Type} [AddGroup Point]
variables {O A B C D A1 B1 C1 D1 : Point}

def is_inside_tetrahedron (O A B C D : Point) : Prop :=
  sorry -- Define how a point is inside a tetrahedron

def opposite_face (P A B C D : Point) : Prop :=
  sorry -- Define what it means for a point to be on the opposite face

theorem tetrahedron_ratio 
  (h₁ : is_inside_tetrahedron O A B C D)
  (h₂ : opposite_face A1 B C D A)
  (h₃ : opposite_face B1 A C D B)
  (h₄ : opposite_face C1 A B D C)
  (h₅ : opposite_face D1 A B C D) :
  (dist A1 O / dist A1 A) + (dist B1 O / dist B1 B) + (dist C1 O / dist C1 C) + (dist D1 O / dist D1 D) = 1 :=
sorry

end tetrahedron_ratio_l15_15710


namespace correct_conclusions_l15_15355

variables {V : Type*} [inner_product_space ℝ V] 

def is_equilateral_triangle (A B C : V) (s : ℝ) : Prop :=
  dist A B = s ∧ dist B C = s ∧ dist C A = s

variables {A B C : V} (a b : V) (s : ℝ) (h_eq : is_equilateral_triangle A B C 3)

theorem correct_conclusions (h0 : A ≠ B)
  (h1 : B ≠ C)
  (h2 : C ≠ A)
  (h3 : dist A B = 3)
  (h4 : dist B C = 3)
  (h5 : dist C A = 3)
  (h6 : (B - A) = 3 • a)
  (h7 : (C - A) = 3 • a + b) :
  (is_unit_vector a ∧ (b = C - B) ∧ (inner_product a b ≠ 0) ∧ (parallel b (C - B)) ∧ orthogonal (6 • a + b) (C - B)) :=
sorry

end correct_conclusions_l15_15355


namespace least_positive_divisible_by_five_primes_l15_15044

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l15_15044


namespace sum_feminist_numbers_lt_40_equals_192_l15_15151

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

def amandine_wins (n : ℕ) : Prop :=
  -- Define the winning condition for Amandine
  sorry

def is_feminist (n : ℕ) : Prop :=
  gcd n 6 = 1 ∧ amandine_wins n

def sum_feminist_numbers_lt_40 : ℕ :=
  (List.range 40).filter is_feminist |>.sum

theorem sum_feminist_numbers_lt_40_equals_192 : sum_feminist_numbers_lt_40 = 192 := 
  by
    sorry

end sum_feminist_numbers_lt_40_equals_192_l15_15151


namespace candidate_X_votes_l15_15736

theorem candidate_X_votes (Z : ℕ) (Y : ℕ) (X : ℕ) (hZ : Z = 25000) 
                          (hY : Y = Z - (2 / 5) * Z) 
                          (hX : X = Y + (1 / 2) * Y) : 
                          X = 22500 :=
by
  sorry

end candidate_X_votes_l15_15736


namespace points_M_exists_l15_15459

-- Define the parabola y^2 = 2x and the line y = x - 1
def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 2 * x
def line (x : ℝ) (y : ℝ) : Prop := y = x - 1

-- Define points P and Q where the line intersects the parabola
def is_intersection_point (x y : ℝ) : Prop := parabola x y ∧ line x y

-- Define the area of the triangle MPQ
def triangle_area (M : ℝ × ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) : ℝ := 
  0.5 * abs ((P.1 * (Q.2 - M.2)) + (Q.1 * (M.2 - P.2)) + (M.1 * (P.2 - Q.2)))

-- Define the condition for given area
def has_given_area (M P Q : ℝ × ℝ) : Prop := 
  triangle_area M P Q = 3 * real.sqrt 3 / 2

-- Define the point M being on the parabola
def is_on_parabola (M : ℝ × ℝ) : Prop := parabola M.1 M.2

-- Prove that there exist three such points M
theorem points_M_exists :
  ∃ M1 M2 M3 : ℝ × ℝ,
  is_on_parabola M1 ∧
  is_on_parabola M2 ∧
  is_on_parabola M3 ∧
  distinct M1 M2 ∧
  distinct M2 M3 ∧
  distinct M1 M3 ∧
  ∃ P Q : ℝ × ℝ,
  is_intersection_point P.1 P.2 ∧
  is_intersection_point Q.1 Q.2 ∧
  has_given_area M1 P Q ∧
  has_given_area M2 P Q ∧
  has_given_area M3 P Q := sorry

/-- Define distinct to ensure points are different -/
def distinct (a b : ℝ × ℝ) : Prop := a ≠ b

end points_M_exists_l15_15459


namespace monster_perimeter_correct_l15_15340

noncomputable def monster_perimeter (radius : ℝ) (central_angle_missing : ℝ) : ℝ :=
  let full_circle_circumference := 2 * radius * Real.pi
  let arc_length := (1 - central_angle_missing / 360) * full_circle_circumference
  arc_length + 2 * radius

theorem monster_perimeter_correct :
  monster_perimeter 2 90 = 3 * Real.pi + 4 :=
by
  -- The proof would go here
  sorry

end monster_perimeter_correct_l15_15340


namespace circumcenter_on_line_XY_l15_15733

open EuclideanGeometry

variables {A B C D E X Y F G T : Point}

-- Define the conditions in the problem
variable (is_isosceles : Isosceles ABC AB AC)
variable (line_parallel_BC : Parallel (Line B C) (Line D E))
variable (angle_bisector_inter_A_B_C : AngleBisector A BAC (circumcircle ABC) = X)
variable (angle_bisector_inter_A_D_E : AngleBisector A BAC (circumcircle ADE) = Y)
variable (F_midpoint_BY : Midpoint F B (Line Y))
variable (G_midpoint_XY : Midpoint G X (Line Y))
variable (T_intersection_CY_DF : T = line_intersection (Line C Y) (Line D F))

-- Define the theorem to be proven
theorem circumcenter_on_line_XY :
  LiesOn (circumcenter (Triangle F G T)) (Line X Y) :=
by
  sorry

end circumcenter_on_line_XY_l15_15733


namespace rounds_played_l15_15510

-- Define the given conditions as Lean constants
def totalPoints : ℝ := 378.5
def pointsPerRound : ℝ := 83.25

-- Define the goal as a Lean theorem
theorem rounds_played :
  Int.ceil (totalPoints / pointsPerRound) = 5 := 
by 
  sorry

end rounds_played_l15_15510


namespace xiao_hong_correct_answers_l15_15699

theorem xiao_hong_correct_answers (q p_correct p_incorrect s y x : ℕ) 
  (hq : q = 12) 
  (hp_correct : p_correct = 10)
  (hp_incorrect : p_incorrect = -5)
  (hs : s = 75)
  (hy : y = (120 - 75) / 15)
  (hx : x = q - y) :
  x = 9 :=
by
  rw [hq, hp_correct, hp_incorrect, hs, hy, hx],
  simp,
  sorry

end xiao_hong_correct_answers_l15_15699


namespace alpha_minus_beta_l15_15158

-- Define the dimensions of Can A
def canADimensions : ℕ × ℕ × ℕ := (20, 16, 10)

-- Define the dimensions of Can B
def canBRadius : ℕ := 10
def canBHeight : ℕ := 10

-- Define the volumes of the cans
def volumeCanA : ℝ := (canADimensions.1 * canADimensions.2 * canADimensions.3 : ℝ)
def volumeCanB : ℝ := (π * (canBRadius^2 : ℝ) * (canBHeight : ℝ))

-- Define alpha and beta
def alpha : ℝ := max volumeCanA volumeCanB
def beta : ℝ := min volumeCanA volumeCanB

-- Prove the final result
theorem alpha_minus_beta : α - β = 3200 - 1000 * π :=
by sorry

end alpha_minus_beta_l15_15158


namespace least_positive_divisible_by_five_primes_l15_15045

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l15_15045


namespace grumblian_words_count_l15_15834

theorem grumblian_words_count (A : Finset ℕ) :
  A = {6} → ∑ i in A, (if i = 1 then 6 else if i = 2 then 6^2 else if i = 3 then 6^3 else if i = 4 then 6^4 else 0) = 1554 :=
by
  sorry

end grumblian_words_count_l15_15834


namespace complex_exchange_two_days_l15_15960

variable {F : Type} [Fintype F] [DecidableEq F]

/-- Define an involution representing a single day's exchange of apartments. -/
def involution (ϕ : F → F) : Prop :=
  ∀ f₁ f₂, ϕ f₁ = f₂ → ϕ f₂ = f₁

/-- Define a bijection representing a complex exchange of apartments. -/
def bijection (Φ : F → F) : Prop :=
  Function.bijective Φ

/-- Prove that any complex exchange of apartments can be carried out in two days. -/
theorem complex_exchange_two_days (Φ : F → F) (hΦ : bijection Φ) :
  ∃ u v : F → F, involution u ∧ involution v ∧ Φ = u ∘ v :=
by
  sorry

end complex_exchange_two_days_l15_15960


namespace scheduling_methods_42_l15_15553

theorem scheduling_methods_42 :
  ∃ f : Fin 6 → Fin 3, 
  (∀ i j, i ≠ j → f i ≠ f j) ∧ 
  (∃ a, f a ≠ 0) ∧ 
  (∃ b, f b ≠ 2) ∧ 
  (card {i | f i = 0} = 2 ∧ card {i | f i = 1} = 2 ∧ card {i | f i = 2} = 2) ∧ 
  fintype.card {g : Fin 3 → Finset (Fin 6) // ∀ i, card (g i) = 2 ∧ (∀ i j, i ≠ j → g i ∩ g j = ∅) ∧ 
    (∃ a, a ∈ g 0 → f a ≠ 0) ∧ 
    (∃ b, b ∈ g 2 → f b ≠ 2)
  } = 42 := 
sorry

end scheduling_methods_42_l15_15553


namespace correct_operation_l15_15078

theorem correct_operation : 
  (sqrt 3 * sqrt 5 = sqrt 15) ∧ 
  ¬(sqrt 2 + sqrt 3 = sqrt 5) ∧ 
  ¬(3 * sqrt 3 - sqrt 3 = 3) ∧ 
  ¬(sqrt 24 / sqrt 6 = 4) := by
  -- Introduce the hypotheses
  let A := ¬(sqrt 2 + sqrt 3 = sqrt 5)
  let B := ¬(3 * sqrt 3 - sqrt 3 = 3)
  let C := ¬(sqrt 24 / sqrt 6 = 4)
  let D := (sqrt 3 * sqrt 5 = sqrt 15)
  -- Combine them into a single proposition
  exact ⟨D, A, B, C⟩

-- Proof in this example is not required, so we leave it as sorry if needed.
-- sorry

end correct_operation_l15_15078


namespace probability_of_two_red_balls_l15_15089

open Nat

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : Nat) : Nat :=
  factorial n / (factorial k * factorial (n - k))

def total_possible_outcomes : Nat :=
  combination 10 2

def favorable_outcomes : Nat :=
  combination 4 2

theorem probability_of_two_red_balls :
  (favorable_outcomes : ℚ) / (total_possible_outcomes : ℚ) = 2 / 15 :=
by
  sorry

end probability_of_two_red_balls_l15_15089


namespace ratio_pentagon_rectangle_l15_15569

theorem ratio_pentagon_rectangle :
  ∀ (p q : ℕ) (h_p : p = 60) (h_q : q = 40), 
  let side_length_of_pentagon := p / 5,
      width_of_rectangle := q / 6 in 
  side_length_of_pentagon / width_of_rectangle = 9 / 5 :=
by {
  intros p q h_p h_q,
  let side_length_of_pentagon := p / 5,
  let width_of_rectangle := q / 6,
  have h_pentagon : side_length_of_pentagon = 12 := by
    simp [side_length_of_pentagon, h_p], sorry,
  have h_rectangle : width_of_rectangle = 6.67 := by
    simp [width_of_rectangle, h_q], sorry,
  have h_ratio : side_length_of_pentagon / width_of_rectangle = 9 / 5 := by 
    simp [h_pentagon, h_rectangle], sorry,
  exact sorry,
}

end ratio_pentagon_rectangle_l15_15569


namespace least_positive_number_divisible_by_five_primes_l15_15053

theorem least_positive_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ 
    (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) :=
by
  use 2310
  split
  { exact by norm_num }
  split
  { intros p hp, fin_cases hp; norm_num }
  { intros m hm, apply nat.le_of_dvd; norm_num }
  sorry

end least_positive_number_divisible_by_five_primes_l15_15053


namespace simple_interest_rate_l15_15572

theorem simple_interest_rate :
  ∀ (P R : ℝ), 
  (R * 25 / 100 = 1) → 
  R = 4 := 
by
  intros P R h
  sorry

end simple_interest_rate_l15_15572


namespace statement_A_statement_A_statement_C_statement_D_l15_15680

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

theorem statement_A (x : ℝ) (hx : x > 1) : f x > 0 := sorry

theorem statement_A' (x : ℝ) (hx : 0 < x ∧ x < 1) : f x < 0 := sorry

theorem statement_C : Set.range f = Set.Ici (-1 / (2 * Real.exp 1)) := sorry

theorem statement_D (x : ℝ) : f x ≥ x - 1 := sorry

end statement_A_statement_A_statement_C_statement_D_l15_15680


namespace tangent_line_equation_at_1_monotonically_increasing_range_of_a_l15_15265

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (1 / x + a) * real.log (1 + x)

-- Part 1: Tangent line equation at point (1, f(1)) when a = -1
theorem tangent_line_equation_at_1 (x y : ℝ) (h : y = f 1 (-1)) :
  (real.log 2) * x + y - (real.log 2) = 0 := 
sorry

-- Part 2: Range of a for which f(x) is monotonically increasing on (0, +∞)
theorem monotonically_increasing_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, 0 < x → 0 < f' x a) ↔ (1 / 2 ≤ a) :=  
sorry

end tangent_line_equation_at_1_monotonically_increasing_range_of_a_l15_15265


namespace derivative_of_function_l15_15828

theorem derivative_of_function (x : ℝ) : (derivative (λ x : ℝ, x + exp x) x) = 1 + exp x :=
by
  sorry

end derivative_of_function_l15_15828


namespace cookies_per_person_l15_15933

/-- Brenda's mother made cookies for 5 people. She prepared 35 cookies, 
    and each of them had the same number of cookies. 
    We aim to prove that each person had 7 cookies. --/
theorem cookies_per_person (total_cookies : ℕ) (number_of_people : ℕ) 
  (h1 : total_cookies = 35) (h2 : number_of_people = 5) : total_cookies / number_of_people = 7 := 
by
  sorry

end cookies_per_person_l15_15933


namespace tangent_line_eq_mon_incr_l15_15277

def f (a : ℝ) (x : ℝ) := (1 / x + a) * Real.log (1 + x)

theorem tangent_line_eq (a : ℝ) (h_a : a = -1) :
  ∃ m b : ℝ, (∀ x y, y = (1 / x - 1) * Real.log (1 + x) → y = m * x + b) ∧ m * 1 + b = 0 ∧ m = -Real.log 2 ∧ b = Real.log 2 :=
sorry

theorem mon_incr (a : ℝ) (h_mon : ∀ x : ℝ, 0 < x → (deriv (f a)) x ≥ 0) :
  1 / 2 ≤ a :=
sorry

end tangent_line_eq_mon_incr_l15_15277


namespace wire_length_from_poles_l15_15806

theorem wire_length_from_poles :
  let d := 20 -- horizontal distance between the poles
  let h1 := 8 -- height of the shorter pole
  let h2 := 18 -- height of the taller pole
  let base_diff := 2 -- base height difference due to slope
  let vertical_diff := (h2 - h1 + base_diff) -- actual vertical difference between the tops

  -- Final proof goal: the length of the wire
  sqrt (d^2 + vertical_diff^2) = sqrt 544 :=
by
  sorry

end wire_length_from_poles_l15_15806


namespace original_number_l15_15738

/-- Proof that the original three-digit number abc equals 118 under the given conditions. -/
theorem original_number (N : ℕ) (hN : N = 4332) (a b c : ℕ)
  (h : 100 * a + 10 * b + c = 118) :
  100 * a + 10 * b + c = 118 :=
by
  sorry

end original_number_l15_15738


namespace tangent_line_at_1_range_of_a_iff_monotonic_increasing_l15_15261

noncomputable def f (x a : ℝ) : ℝ := (1 / x + a) * real.log (1 + x)

theorem tangent_line_at_1 (a : ℝ) (h : a = -1) :
  let f_at := (1 / (1:ℝ) + a) * real.log (1 + 1)
  let f_prime := -real.log 2
  let tangent_line := λ x : ℝ, f_prime * (x - 1) + f_at in
  tangent_line = λ x, -real.log 2 * x + real.log 2 :=
by {
  -- Proof to be provided
  sorry
}

theorem range_of_a_iff_monotonic_increasing :
  ∀ x : ℝ, x > 0 → monotonic_increasing (λ a : ℝ, (1/x + a) * real.log (1 + x)) ↔ a ∈ set.Ici (1/2) :=
by {
  -- Proof to be provided
  sorry
}

end tangent_line_at_1_range_of_a_iff_monotonic_increasing_l15_15261


namespace possible_values_l15_15637

theorem possible_values (m n : ℕ) (h1 : 10 ≥ m) (h2 : m > n) (h3 : n ≥ 4) (h4 : (m - n) ^ 2 = m + n) :
    (m, n) = (10, 6) :=
sorry

end possible_values_l15_15637


namespace christmas_tree_seller_l15_15547

theorem christmas_tree_seller 
  (cost_spruce : ℕ := 220) 
  (cost_pine : ℕ := 250) 
  (cost_fir : ℕ := 330) 
  (total_revenue : ℕ := 36000) 
  (equal_trees: ℕ) 
  (h_costs : cost_spruce + cost_pine + cost_fir = 800) 
  (h_revenue : equal_trees * 800 = total_revenue):
  3 * equal_trees = 135 :=
sorry

end christmas_tree_seller_l15_15547


namespace fill_tanker_time_l15_15560

theorem fill_tanker_time :
  ∀ (A B C : ℝ), 
    (60 : ℝ) = A →
    (40 : ℝ) = B →
    (80 : ℝ) = C →
    ∃ T, 
      (T / 3) / B + ((T / 3) / A + (T / 3) / C) + ((T / 3) / A + (T / 3) / B + (T / 3) / C) = 1 ∧
      T ≈ 27.69 := 
by
  intros A B C hA hB hC
  use 27.69
  sorry

end fill_tanker_time_l15_15560


namespace new_boarder_ratio_l15_15469

structure School where
  initial_boarders : ℕ
  day_students : ℕ
  boarders_ratio : ℚ

theorem new_boarder_ratio (S : School) (additional_boarders : ℕ) :
  S.initial_boarders = 60 →
  S.boarders_ratio = 2 / 5 →
  additional_boarders = 15 →
  S.day_students = (60 * 5) / 2 →
  (S.initial_boarders + additional_boarders) / S.day_students = 1 / 2 :=
by
  sorry

end new_boarder_ratio_l15_15469


namespace arrange_three_numbers_three_digit_numbers_without_repetition_three_digit_odd_numbers_without_repetition_l15_15982

theorem arrange_three_numbers (s : Finset ℕ) (h : s = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) : 
  s.card.choose 3 * 3.factorial = 720 :=
by
  sorry

theorem three_digit_numbers_without_repetition (s : Finset ℕ) (h : s = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) : 
  (∑ k in s - {0}, card (s.erase k).choose 2 * 2.factorial) + card (s.erase 0).choose 2 * 2.factorial = 648 :=
by
  sorry

theorem three_digit_odd_numbers_without_repetition (s : Finset ℕ) (odd_nums : Finset ℕ) 
  (h : s = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) (hodd : odd_nums = {1, 3, 5, 7, 9}) :
  ∑ k in odd_nums, (∑ t in s - {0, k}, card (s.erase k).erase t = 320 :=
by
  sorry

end arrange_three_numbers_three_digit_numbers_without_repetition_three_digit_odd_numbers_without_repetition_l15_15982


namespace canoe_downstream_speed_l15_15549

theorem canoe_downstream_speed 
  (V_up : ℝ) (V_s : ℝ) 
  (h1 : V_up = 3)
  (h2 : V_s = 4.5) : 
  ∃ V_down, V_down = 12 :=
by
  let V_c := V_up + V_s
  have h3 : V_c = 7.5 := by
    rw [h1, h2]
    norm_num
    
  let V_down := V_c + V_s
  have h4 : V_down = 12 := by
    rw [h3, h2]
    norm_num
  use V_down
  exact h4

end canoe_downstream_speed_l15_15549


namespace PQ_parallel_AB_l15_15807

-- Definitions as per the conditions in the problem statement
variables {A B C D E O P Q : Point}
variable [plane_geometry] -- Assume a plane geometry context

-- Conditions:
-- 1. The center O of the circumcircle of quadrilateral ABCD lies on the side AB.
axiom center_O : Circumcenter ABCD O ∧ Collinear {A, B, O}

-- 2. Point E is symmetric to D with respect to the line AB.
axiom symmetric_E : SymmetricPoint D A B E

-- 3. Segments AC and DO intersect at point P.
axiom intersection_P : Intersect (Line A C) (Line D O) P

-- 4. Segments BD and CE intersect at point Q.
axiom intersection_Q : Intersect (Line B D) (Line C E) Q

-- Prove that PQ is parallel to AB.
theorem PQ_parallel_AB : Parallel (Line P Q) (Line A B) :=
by
  sorry

end PQ_parallel_AB_l15_15807


namespace least_number_divisible_by_five_primes_l15_15061

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, (n > 0) ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → p ∣ n) ∧ n = 2310 :=
begin
  sorry
end

end least_number_divisible_by_five_primes_l15_15061


namespace jones_elementary_school_students_l15_15891

theorem jones_elementary_school_students
  (X : ℕ)
  (boys_percent_total : ℚ)
  (num_students_represented : ℕ)
  (percent_of_boys : ℚ)
  (h1 : boys_percent_total = 0.60)
  (h2 : num_students_represented = 90)
  (h3 : percent_of_boys * (boys_percent_total * X) = 90)
  : X = 150 :=
by
  sorry

end jones_elementary_school_students_l15_15891


namespace volunteers_distribution_l15_15657

theorem volunteers_distribution:
  let num_volunteers := 5
  let group_distribution := (2, 2, 1)
  ∃ (ways : ℕ), ways = 15 :=
by
  sorry

end volunteers_distribution_l15_15657


namespace best_selling_model_is_mode_l15_15574

def sales_data : fin 6 → ℕ 
| ⟨0, _⟩ := 13
| ⟨1, _⟩ := 21
| ⟨2, _⟩ := 35
| ⟨3, _⟩ := 48
| ⟨4, _⟩ := 26
| ⟨5, _⟩ := 8

def mode_of_sales_data (data : fin 6 → ℕ) : ℕ := 41

theorem best_selling_model_is_mode : 
  mode_of_sales_data sales_data = 41 :=
sorry

end best_selling_model_is_mode_l15_15574


namespace continuous_stripe_probability_l15_15902

notation "ℙ" => ProbabilityTheory.ProbabilityMeasure

-- Define the problem setup.
def stripe_colors (cube : ℕ) : set (set ℕ) :=
  { colors | ∀ face ∈ 𝒰{1,2,3,4,5,6}, colors ∈ {0,1} }

-- Define the probability space.
noncomputable def tower_probability_space : ProbabilityTheory.ProbabilitySpace :=
  ProbabilityTheory.probability_space_of_finset ℙ (stripe_colors 3) sorry

-- Define the event of continuous stripe.
def continuous_stripe_event : Set (Set ℕ) :=
  {colors | ∃ (striped_faces : Finset ℕ), striped_faces.card = 1 ∧ -- One stripe connects top to bottom
  ∀ cube ∈ {1,2,3}, striped_faces ⊆ {face | face ∈ striped_faces }}

-- Statement of the theorem.
theorem continuous_stripe_probability : ℙ(tower_probability_space, continuous_stripe_event) = 1 / 4096 :=
sorry

end continuous_stripe_probability_l15_15902


namespace tangent_line_slope_at_origin_l15_15955

theorem tangent_line_slope_at_origin (a : ℝ) :
  (∀ x : ℝ, (x, a * x - log (x + 1)) = (0, 0) → a - 1 = 2) → a = 3 := by
  sorry

end tangent_line_slope_at_origin_l15_15955


namespace least_positive_number_divisible_by_five_smallest_primes_l15_15019

theorem least_positive_number_divisible_by_five_smallest_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  Nat.min (p1 * p2 * p3 * p4 * p5) := 
  2310 :=
by
  sorry

end least_positive_number_divisible_by_five_smallest_primes_l15_15019


namespace maximum_value_of_d_l15_15190

theorem maximum_value_of_d 
  (d e : ℕ) 
  (h1 : 0 ≤ d ∧ d < 10) 
  (h2: 0 ≤ e ∧ e < 10) 
  (h3 : (18 + d + e) % 3 = 0) 
  (h4 : (15 - (d + e)) % 11 = 0) 
  : d ≤ 0 := 
sorry

end maximum_value_of_d_l15_15190


namespace number_of_questions_correct_in_first_half_l15_15511

theorem number_of_questions_correct_in_first_half (correct_in_second_half questions_per_point final_score : ℕ) 
    (h1 : correct_in_second_half = 2) 
    (h2 : questions_per_point = 3) 
    (h3 : final_score = 15) : 
    ∃ (correct_in_first_half : ℕ), correct_in_first_half = 3 := 
by 
    use (final_score - (correct_in_second_half * questions_per_point)) / questions_per_point
    rw [h1, h2, h3]
    norm_num
    simp
    exact rfl

end number_of_questions_correct_in_first_half_l15_15511


namespace trig_identity_l15_15182

theorem trig_identity :
  sin (17 * pi / 180) * sin (223 * pi / 180) - sin (253 * pi / 180) * cos (43 * pi / 180) = 1 / 2 := 
by
  sorry

end trig_identity_l15_15182


namespace min_value_expr_l15_15972

theorem min_value_expr (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) :
  (3 * Real.sin θ + 2 / Real.cos θ + Real.sqrt 3 * Real.cot θ) = 7 := sorry

end min_value_expr_l15_15972


namespace profit_difference_l15_15825

-- Setting up the conditions
def construction_cost_others (C : ℝ) : ℝ := C

def construction_cost_certain (C : ℝ) : ℝ := C + 100000

def selling_price_others : ℝ := 320000

def selling_price_certain : ℝ := 1.5 * 320000

def profit_certain (C : ℝ) : ℝ := selling_price_certain - construction_cost_certain C

def profit_others (C : ℝ) : ℝ := selling_price_others - construction_cost_others C

-- Proving the difference in profit
theorem profit_difference (C : ℝ) : profit_certain C - profit_others C = 60000 :=
by
    simp [profit_certain, profit_others, selling_price_certain, selling_price_others, construction_cost_certain, construction_cost_others]
    ring
    sorry

end profit_difference_l15_15825


namespace masking_tape_needed_l15_15958

-- Define the room dimensions and conditions
def length_rect := 5
def width_rect := 3 
def side_square := 4
def height_room := 2.5
def window_height := 1.0
def window_width := 1.5
def door_height := 2.0
def door_width := 1.0
def num_windows := 3 
def tape_layers := 2

-- Total amount of masking tape Elijah needs
theorem masking_tape_needed : 
  let P_rect := 2 * (length_rect + width_rect)
  let P_square_3walls := 3 * side_square
  let P_total := P_rect + P_square_3walls
  let Width_windows := num_windows * window_width
  let Width_door := door_width
  let P_effective := P_total - Width_windows - Width_door in
  tape_layers * P_effective = 45 :=
by
  sorry

end masking_tape_needed_l15_15958


namespace ellipse_properties_l15_15671

theorem ellipse_properties :
  (∀ x y: ℝ, (x^2)/100 + (y^2)/36 = 1) →
  ∃ a b c e : ℝ, 
  a = 10 ∧ 
  b = 6 ∧ 
  c = 8 ∧ 
  2 * a = 20 ∧ 
  e = 4 / 5 :=
by
  intros
  sorry

end ellipse_properties_l15_15671


namespace average_of_three_numbers_l15_15852

theorem average_of_three_numbers
  (a b c : ℕ)
  (h1 : 2 * a + b + c = 130)
  (h2 : a + 2 * b + c = 138)
  (h3 : a + b + 2 * c = 152) :
  (a + b + c) / 3 = 35 :=
by
  sorry

end average_of_three_numbers_l15_15852


namespace least_positive_number_divisible_by_five_smallest_primes_l15_15021

theorem least_positive_number_divisible_by_five_smallest_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  Nat.min (p1 * p2 * p3 * p4 * p5) := 
  2310 :=
by
  sorry

end least_positive_number_divisible_by_five_smallest_primes_l15_15021


namespace product_of_real_roots_is_1_l15_15202

noncomputable def product_of_real_roots : ℝ :=
  let roots := {x : ℝ | x^(2 * Real.log x) = Real.exp 1} in
  roots.to_finset.prod id

theorem product_of_real_roots_is_1 : product_of_real_roots = 1 :=
by
  sorry

end product_of_real_roots_is_1_l15_15202


namespace cube_root_simplification_l15_15865

theorem cube_root_simplification : (∛(4^6 + 4^6 + 4^6 + 4^6) = 16 * ∛4) :=
by {
  -- Proof goes here
  sorry
}

end cube_root_simplification_l15_15865


namespace least_positive_divisible_by_five_primes_l15_15039

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l15_15039


namespace machine_minutes_worked_l15_15153

-- Definitions based on conditions
def shirts_made_yesterday : ℕ := 9
def shirts_per_minute : ℕ := 3

-- The proof problem statement
theorem machine_minutes_worked (shirts_made_yesterday shirts_per_minute : ℕ) : 
  shirts_made_yesterday / shirts_per_minute = 3 := 
by
  sorry

end machine_minutes_worked_l15_15153


namespace circle_tangent_eqn_l15_15994

theorem circle_tangent_eqn (x y : ℝ) (radius : ℝ) (P : ℝ × ℝ := (1, 1))
    (l_eqn : x + 2 * y - 3 = 0) (tangent_eqn : radius = 5) :
  ( ∀ (x_0 y_0 : ℝ), 
    (x_0 = 1 + real.sqrt 5 ∧ y_0 = 1 + 2 * real.sqrt 5) ∨ 
    (x_0 = 1 - real.sqrt 5 ∧ y_0 = 1 - 2 * real.sqrt 5) ∧
    ((x - (x_0))^2 + (y - (y_0))^2 = radius^2)) :=
begin
  sorry
end

end circle_tangent_eqn_l15_15994


namespace number_of_integers_satisfying_condition_l15_15973

noncomputable def g (n : ℤ) : ℤ := ⌈ 198 * n / 200 ⌉ - ⌊ 200 * n / 201 ⌋

theorem number_of_integers_satisfying_condition :
  {n : ℤ | 1 + ⌊ 200 * n / 201 ⌋ = ⌈ 198 * n / 200 ⌉ }.to_finset.card = 40200 :=
by
  sorry

end number_of_integers_satisfying_condition_l15_15973


namespace inequality_proof_l15_15530

variables (a b c d e f : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
variable (hcond : |sqrt (a * d) - sqrt (b * c)| ≤ 1)

theorem inequality_proof :
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end inequality_proof_l15_15530


namespace loss_equals_cost_price_of_balls_l15_15777

variable (selling_price : ℕ) (cost_price_ball : ℕ)
variable (number_of_balls : ℕ) (loss_incurred : ℕ) (x : ℕ)

-- Conditions
def condition1 : selling_price = 720 := sorry -- Selling price of 11 balls is Rs. 720
def condition2 : cost_price_ball = 120 := sorry -- Cost price of one ball is Rs. 120
def condition3 : number_of_balls = 11 := sorry -- Number of balls is 11

-- Cost price of 11 balls
def cost_price (n : ℕ) (cp_ball : ℕ): ℕ := n * cp_ball

-- Loss incurred on selling 11 balls
def loss (cp : ℕ) (sp : ℕ): ℕ := cp - sp

-- Equation for number of balls the loss equates to
def loss_equation (l : ℕ) (cp_ball : ℕ): ℕ := l / cp_ball

theorem loss_equals_cost_price_of_balls : 
  ∀ (n sp cp_ball cp l: ℕ), 
  sp = 720 ∧ cp_ball = 120 ∧ n = 11 ∧ 
  cp = cost_price n cp_ball ∧ 
  l = loss cp sp →
  loss_equation l cp_ball = 5 := sorry

end loss_equals_cost_price_of_balls_l15_15777


namespace prob_exactly_M_laws_expected_laws_included_l15_15720

noncomputable def prob_of_exactly_M_laws (K N M : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - (1 - p)^N
  (Nat.choose K M) * q^M * (1 - q)^(K - M)

noncomputable def expected_num_of_laws (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

-- Part (a): Prove that the probability of exactly M laws being included is as follows
theorem prob_exactly_M_laws (K N M : ℕ) (p : ℝ) :
  prob_of_exactly_M_laws K N M p =
    (Nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - (1 - p)^N)^(K - M)) :=
sorry

-- Part (b): Prove that the expected number of laws included is as follows
theorem expected_laws_included (K N : ℕ) (p : ℝ) :
  expected_num_of_laws K N p =
    K * (1 - (1 - p)^N) :=
sorry

end prob_exactly_M_laws_expected_laws_included_l15_15720


namespace part1_eq_tangent_line_at_one_l15_15283

def f (x a : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

theorem part1_eq_tangent_line_at_one (a : ℝ) (h : a = -1) :
  ∀ x y : ℝ, 
  f x a = (1/x - 1) * Real.log (1 + x) →
  x = 1 →
  y = 0 →
  (Real.log 2) * x + y - Real.log 2 = 0 := 
sorry

end part1_eq_tangent_line_at_one_l15_15283


namespace brownies_count_l15_15154

theorem brownies_count {B : ℕ} 
  (h1 : B/2 = (B - B / 2))
  (h2 : B/4 = (B - B / 2) / 2)
  (h3 : B/4 - 2 = B/4 - 2)
  (h4 : B/4 - 2 = 3) : 
  B = 20 := 
by 
  sorry

end brownies_count_l15_15154


namespace velocity_at_2_l15_15685

variable (t : ℝ) (s : ℝ)

noncomputable def displacement (t : ℝ) : ℝ := t^2 + 3 / t

noncomputable def velocity (t : ℝ) : ℝ := (deriv displacement) t

theorem velocity_at_2 : velocity t = 2 * 2 - (3 / 4) := by
  sorry

end velocity_at_2_l15_15685


namespace can_cut_one_more_square_l15_15886

theorem can_cut_one_more_square (G : Finset (Fin 29 × Fin 29)) (hG : G.card = 99) :
  (∃ S : Finset (Fin 29 × Fin 29), S.card = 4 ∧ (S ⊆ G) ∧ (∀ s1 s2 : Fin 29 × Fin 29, s1 ∈ S → s2 ∈ S → s1 ≠ s2 → (|s1.1 - s2.1| > 2 ∨ |s1.2 - s2.2| > 2))) :=
sorry

end can_cut_one_more_square_l15_15886


namespace maximal_subset_size_l15_15757

-- Define the set M = {1, 2, ..., 1995}
def M : Set ℕ := {n | 1 ≤ n ∧ n ≤ 1995}

-- Define the subset A of M with the condition: if x ∈ A, then 15x ∉ A
def valid_subset (A : Set ℕ) : Prop :=
  ∀ x ∈ A, 15 * x ∉ A

-- The theorem stating the maximum number of elements in a valid subset A of M
theorem maximal_subset_size : ∃ A ⊆ M, valid_subset A ∧ (A.card = 1870) :=
sorry

end maximal_subset_size_l15_15757


namespace square_area_l15_15398

theorem square_area
  (E_on_AD : ∃ E : ℝ × ℝ, ∃ s : ℝ, s > 0 ∧ E = (0, s))
  (F_on_extension_BC : ∃ F : ℝ × ℝ, ∃ s : ℝ, s > 0 ∧ F = (s, 0))
  (BE_20 : ∃ B E : ℝ × ℝ, ∃ s : ℝ, B = (s, 0) ∧ E = (0, s) ∧ dist B E = 20)
  (EF_25 : ∃ E F : ℝ × ℝ, ∃ s : ℝ, E = (0, s) ∧ F = (s, 0) ∧ dist E F = 25)
  (FD_20 : ∃ F D : ℝ × ℝ, ∃ s : ℝ, F = (s, 0) ∧ D = (s, s) ∧ dist F D = 20) :
  ∃ s : ℝ, s > 0 ∧ s^2 = 400 :=
by
  -- Hypotheses are laid out in conditions as defined above
  sorry

end square_area_l15_15398


namespace prob_exactly_M_laws_expected_laws_included_l15_15721

noncomputable def prob_of_exactly_M_laws (K N M : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - (1 - p)^N
  (Nat.choose K M) * q^M * (1 - q)^(K - M)

noncomputable def expected_num_of_laws (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

-- Part (a): Prove that the probability of exactly M laws being included is as follows
theorem prob_exactly_M_laws (K N M : ℕ) (p : ℝ) :
  prob_of_exactly_M_laws K N M p =
    (Nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - (1 - p)^N)^(K - M)) :=
sorry

-- Part (b): Prove that the expected number of laws included is as follows
theorem expected_laws_included (K N : ℕ) (p : ℝ) :
  expected_num_of_laws K N p =
    K * (1 - (1 - p)^N) :=
sorry

end prob_exactly_M_laws_expected_laws_included_l15_15721


namespace least_positive_number_divisible_by_five_smallest_primes_l15_15018

theorem least_positive_number_divisible_by_five_smallest_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  Nat.min (p1 * p2 * p3 * p4 * p5) := 
  2310 :=
by
  sorry

end least_positive_number_divisible_by_five_smallest_primes_l15_15018


namespace not_forall_abs_ge_zero_l15_15690

theorem not_forall_abs_ge_zero : (¬(∀ x : ℝ, |x + 1| ≥ 0)) ↔ (∃ x : ℝ, |x + 1| < 0) :=
by
  sorry

end not_forall_abs_ge_zero_l15_15690


namespace max_handshakes_l15_15109

theorem max_handshakes (n : ℕ) (h : n = 20) : ∃ k : ℕ, k = 190 :=
by
  have h1 : 20 * 19 = 380 := rfl
  have h2 : 380 / 2 = 190 := rfl
  use 190
  rw [h]
  rw [h1, h2]
  sorry

end max_handshakes_l15_15109


namespace can_weight_is_two_l15_15361

theorem can_weight_is_two (c : ℕ) (h1 : 100 = 20 * c + 6 * ((100 - 20 * c) / 6)) (h2 : 160 = 10 * ((100 - 20 * c) / 6) + 3 * 20) : c = 2 :=
by
  sorry

end can_weight_is_two_l15_15361


namespace find_OQ_find_area_and_QA_l15_15552

-- Define the initial geometry setup
noncomputable def O := (0,0)
noncomputable def r := 4
noncomputable def circle_ω := {center := O, radius := r}

noncomputable def T := (1,0) -- Assume some coordinate for T, actual coord doesn't matter for radius
noncomputable def r_Ω := (Real.sqrt 65) / 2
noncomputable def circle_Ω := {center := T, radius := r_Ω}

-- Define the points of tangency
noncomputable def M := (3, 4) -- Hypothetical coords where circle_ω touches FQ
noncomputable def P := (3, -4) -- Hypothetical coords where circle_ω touches EQ

-- Define the triangle vertices
noncomputable def F := (-1, -1)
noncomputable def E := (2, 3)
noncomputable def Q := (4, 5)
noncomputable def triangle_EFQ := {vertices := {F, E, Q}}

-- Given the radii and properties:
noncomputable def OQ := (Real.sqrt 65)

-- Properties for part b
noncomputable def area_ratio := 2/3
noncomputable def area_EFQ := 84
noncomputable def QA := 3 * r_Ω

-- Use these definitions to state the theorem
theorem find_OQ :
  -- Given circle_ω and circle_Ω as specified,
  OQ = Real.sqrt 65 := sorry

theorem find_area_and_QA :
  -- Given the ratio of areas and the properties derived,
  (area_EFQ = 84) ∧ (QA = 3 * r_Ω) := sorry

end find_OQ_find_area_and_QA_l15_15552


namespace opposite_of_sqrt_seven_l15_15462

theorem opposite_of_sqrt_seven : (∀ (x : ℝ), x = sqrt 7 → -x = -sqrt 7) :=
begin
  intro x,
  intro h,
  rw h,
  reflexivity
end

end opposite_of_sqrt_seven_l15_15462


namespace inverse_function_of_y_eq_3_pow_x_plus_1_l15_15833

noncomputable def inv_f : ℝ → ℝ := λ x, -1 + Real.logb 3 x

theorem inverse_function_of_y_eq_3_pow_x_plus_1 (x : ℝ) (h : 0 < x) : 
  (inv_f x = y) ↔ (x = 3^(y + 1)) :=
by
  sorry

end inverse_function_of_y_eq_3_pow_x_plus_1_l15_15833


namespace simplify_cos_cube_minus_sin_cube_l15_15422

theorem simplify_cos_cube_minus_sin_cube (θ : ℝ) :
  cos θ ^ 3 - sin θ ^ 3 = (cos θ - sin θ) * (1 + cos θ * sin θ) :=
by
  sorry

end simplify_cos_cube_minus_sin_cube_l15_15422


namespace average_percent_score_is_approx_71_33_l15_15393

-- Defining the input data
def scores : List ℕ := [95, 85, 75, 65, 55, 45, 0]
def num_students : List ℕ := [10, 20, 40, 30, 15, 3, 2]
def total_students : ℕ := 120

-- A helper definition to calculate weighted sum
def weighted_sum (scores : List ℕ) (num_students : List ℕ) : ℕ :=
  List.sum (List.map (λ (p : ℕ × ℕ), p.1 * p.2) (List.zip scores num_students))

-- The definition of average score
def average_score (scores : List ℕ) (num_students : List ℕ) (total_students : ℕ) : Float :=
  weighted_sum scores num_students / total_students.toFloat

-- The final theorem to prove
theorem average_percent_score_is_approx_71_33 :
  abs (average_score scores num_students total_students - 71.33) < 0.01 :=
by
  sorry

end average_percent_score_is_approx_71_33_l15_15393


namespace find_A_l15_15075

theorem find_A : ∃ A : ℕ, A = 6 * 5 + 4 := 
by
  use 34
  trivial

end find_A_l15_15075


namespace conference_total_duration_is_715_l15_15125

structure ConferenceSession where
  hours : ℕ
  minutes : ℕ

def totalDuration (s1 s2 : ConferenceSession): ℕ :=
  (s1.hours * 60 + s1.minutes) + (s2.hours * 60 + s2.minutes)

def session1 : ConferenceSession := { hours := 8, minutes := 15 }
def session2 : ConferenceSession := { hours := 3, minutes := 40 }

theorem conference_total_duration_is_715 :
  totalDuration session1 session2 = 715 := 
sorry

end conference_total_duration_is_715_l15_15125


namespace right_obtuse_triangle_impossible_l15_15079

def triangle_interior_angles_sum (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def is_right_angle (α : ℝ) : Prop :=
  α = 90

def is_obtuse_angle (α : ℝ) : Prop :=
  α > 90

theorem right_obtuse_triangle_impossible (α β γ : ℝ) (h1 : triangle_interior_angles_sum α β γ) (h2 : is_right_angle α) (h3 : is_obtuse_angle β) : false :=
  sorry

end right_obtuse_triangle_impossible_l15_15079


namespace alice_oranges_l15_15148

theorem alice_oranges (E A : ℕ) 
  (h1 : A = 2 * E) 
  (h2 : E + A = 180) : 
  A = 120 :=
by
  sorry

end alice_oranges_l15_15148


namespace vectors_are_perpendicular_l15_15638

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-2, 1)

theorem vectors_are_perpendicular :
  let dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2
  in dot_product vector_a vector_b = 0 → 
     ∀ v1 v2 : ℝ × ℝ, dot_product v1 v2 = 0 → (v1.1 * v2.1 + v1.2 * v2.2 = 0) :=
by
  sorry

end vectors_are_perpendicular_l15_15638


namespace third_derivative_correct_l15_15194

noncomputable def func (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative_correct :
  (deriv^[3] func) x = (4 / (1 + x^2)^2) :=
sorry

end third_derivative_correct_l15_15194


namespace inequality_proof_l15_15540

theorem inequality_proof (a b c d e f : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
    (hcond : abs (sqrt (a * d) - sqrt (b * c)) ≤ 1) :
    (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := 
  sorry

end inequality_proof_l15_15540


namespace max_viewers_after_T_l15_15583

theorem max_viewers_after_T 
  (rating : ℕ → ℕ) 
  (sum_ratings : ℕ → ℕ) 
  (n : ℕ) 
  (x : ℕ)
  (x_max : x = 10)
  (n_min : n = 1)
  (H : ∀ m, rating m = sum_ratings m / m)
  (HT : ∃ T, rating T = x ∧ ∀ k > T, rating k = rating (k - 1) - 1) :
  ∃ N, N = 5 := 
begin
  -- proof goes here
  sorry
end

end max_viewers_after_T_l15_15583


namespace product_of_next_choices_to_win_l15_15481

theorem product_of_next_choices_to_win (circles : Finset ℕ)
  (lines : Finset (Finset ℕ))
  (A_choices B_choices : Finset ℕ)
  (A_next_choices : Finset ℕ)
  (product : ℕ)
  (h_circles : circles = {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h_lines : lines = { {1, 2, 3}, {1, 4, 7}, {1, 5, 9}, {2, 5, 8}, {3, 6, 9}, {4, 5, 6}, {7, 8, 9}, {1, 6, 8}, {2, 4, 9}, {3, 5, 7} })
  (h_A_first_choice : A_choices = {1})
  (h_B_first_choice : B_choices = {5})
  (h_A_next_choices_equiv : A_next_choices = {3, 4, 6, 7})
  (h_product_correct : product = 504) :
  ∏ x in A_next_choices, x = product :=
sorry

end product_of_next_choices_to_win_l15_15481


namespace robert_elizabeth_age_difference_l15_15780

theorem robert_elizabeth_age_difference 
  (patrick_age_1_5_times_robert : ∀ (robert_age : ℝ), ∃ (patrick_age : ℝ), patrick_age = 1.5 * robert_age)
  (elizabeth_born_after_richard : ∀ (richard_age : ℝ), ∃ (elizabeth_age : ℝ), elizabeth_age = richard_age - 7 / 12)
  (elizabeth_younger_by_4_5_years : ∀ (patrick_age : ℝ), ∃ (elizabeth_age : ℝ), elizabeth_age = patrick_age - 4.5)
  (robert_will_be_30_3_after_2_5_years : ∃ (robert_age_current : ℝ), robert_age_current = 30.3 - 2.5) :
  ∃ (years : ℤ) (months : ℤ), years = 9 ∧ months = 4 := by
  sorry

end robert_elizabeth_age_difference_l15_15780


namespace max_additional_viewers_l15_15587

-- Given conditions:
-- viewers_rating_seq is a sequence of integer ratings given by viewers
-- At moment T, the rating (sum of all given scores divided by their number) is an integer
-- With each new voting viewer, the rating decreases by one unit

theorem max_additional_viewers (rating : ℕ → ℕ) (T : ℕ)
  (h_initial: ∃ x n, x ∈ (set.range (λ i, rating i)) ∧ n = (count (λ i, i ≤ T) rating)
    ∧ (∑ i in finset.range (n+1), rating i) / (n+1) = x)
  (h_decrease: ∀ m > T, (∑ i in finset.range (m+1), rating i) / (m+1) = (h_initial.some - (m - T))):
  ∃ n, n = 5 := 
by 
  sorry

end max_additional_viewers_l15_15587


namespace least_number_divisible_by_five_smallest_primes_l15_15006

theorem least_number_divisible_by_five_smallest_primes : 
  ∃ n ∈ ℕ+, n = 2 * 3 * 5 * 7 * 11 ∧ n = 2310 :=
by
  sorry

end least_number_divisible_by_five_smallest_primes_l15_15006


namespace smallest_sum_is_11_over_24_l15_15939

theorem smallest_sum_is_11_over_24 :
  min ((1/3 + 1/4), 
       (1/3 + 1/5), 
       (1/3 + 1/6), 
       (1/3 + 1/7), 
       (1/3 + 1/8)) = 11/24 := by
  sorry

end smallest_sum_is_11_over_24_l15_15939


namespace find_number_l15_15633

theorem find_number (x : ℕ) (h : 5 + 2 * (8 - x) = 15) : x = 3 :=
sorry

end find_number_l15_15633


namespace find_t_l15_15396

theorem find_t (t : ℤ) :
  ((t + 1) * (3 * t - 3)) = ((3 * t - 5) * (t + 2) + 2) → 
  t = 5 :=
by
  intros
  sorry

end find_t_l15_15396


namespace smallest_b_value_l15_15791

noncomputable def smallest_possible_value_of_b : ℝ :=
  (3 + Real.sqrt 5) / 2

theorem smallest_b_value
  (a b : ℝ)
  (h1 : 1 < a)
  (h2 : a < b)
  (h3 : b ≥ a + 1)
  (h4 : (1/b) + (1/a) ≤ 1) :
  b = smallest_possible_value_of_b :=
sorry

end smallest_b_value_l15_15791


namespace quadrilateral_division_l15_15579

variables (A B C D M N O P Q Q' E F: Point)

def convex_quadrilateral (A B C D : Point) : Prop :=
  convex_hull ({A, B, C, D} : set Point) = {A, B, C, D}

def midpoint (X Y : Point) : Point := 1/2 * (X + Y) -- simplistic definition

def area (Q : Quadrilateral) : ℝ := ∑ Q.sides.map (λ e, length (e.to_vector))

def subdivides (X Y : Point) : ConvexQuadrilateral -> Prop := sorry -- define appropriate conditions for subdivision

theorem quadrilateral_division :
  convex_quadrilateral A B C D → midpoint A C = E → midpoint B D = F → 
  midpoint A B = M → midpoint C D = P → midpoint B C = N → midpoint D A = O → 
  let Q := quadrilateral E F M P in
  let Q' := quadrilateral E F N O in
  area Q = area Q' →
  subdivides A C (quadrilateral A B C D) ∨ subdivides B D (quadrilateral A B C D) :=
begin
  sorry,
end

end quadrilateral_division_l15_15579


namespace least_number_divisible_by_five_smallest_primes_l15_15001

theorem least_number_divisible_by_five_smallest_primes : 
  ∃ n ∈ ℕ+, n = 2 * 3 * 5 * 7 * 11 ∧ n = 2310 :=
by
  sorry

end least_number_divisible_by_five_smallest_primes_l15_15001


namespace part1_part2_l15_15655

section math_proof_problem

-- Define the conditions based on the problem
variable {Point : Type} [LinearOrder Point] 

structure Line : Type :=
  (point1 point2 point3 : Point)

def intersects (l1 l2 : Line) : Prop :=
  l1.point1 ≠ l2.point1 ∧ l1.point2 ≠ l2.point2 ∧ l1.point3 ≠ l2.point3

def valid_intersections (lines : List Line) : Prop :=
  (∀ (i j : ℕ) (hi : i < lines.length) (hj : j < lines.length), i ≠ j → intersects (lines.get ⟨i, hi⟩) (lines.get ⟨j, hj⟩)) ∧
  lines.length = 4

def valid_segments (lines : List Line) : List ℕ :=
  (lines.map (λ l => [1, 1])).join 

-- Part (1): Prove the lengths 1 to 8 cannot be the lengths of the 8 segments
theorem part1 (lines : List Line) (h_inter : valid_intersections(lines)) : ¬ (valid_segments lines = [1, 2, 3, 4, 5, 6, 7, 8]) := sorry

-- Part (2): Prove the lengths of these 8 segments can be distinct natural numbers
theorem part2 (lines : List Line) (h_inter : valid_intersections(lines)) : ∃ (lengths : List ℕ), (lengths.length = 8) ∧ (∀ i j, i ≠ j → lengths.nth i ≠ lengths.nth j) := sorry

end math_proof_problem

end part1_part2_l15_15655


namespace solve_for_x_and_y_l15_15702

theorem solve_for_x_and_y : 
  (∃ x y : ℝ, 0.65 * 900 = 0.40 * x ∧ 0.35 * 1200 = 0.25 * y) → 
  ∃ x y : ℝ, x + y = 3142.5 :=
by
  sorry

end solve_for_x_and_y_l15_15702


namespace circle_and_tangent_line_l15_15742

-- Definitions and Lemmas to set up the context
def centered_at_origin (O : Point) : Prop := O = ⟨ 0, 0 ⟩

def tangent_to_line (C : Point) (r : ℝ) (line : ℝ × ℝ × ℝ) : Prop :=
  let (A, B, C) := line
  let d := |A * C.x + B * C.y + C| / Real.sqrt (A^2 + B^2) in
  d = r

def equation_of_circle (O : Point) (r : ℝ) (eq : ℝ × ℝ × ℝ) : Prop :=
  let (A, B, C) := eq in
  A * O.x^2 + B * O.y^2 = C

def equation_of_line_through_point (P : Point) (line : ℝ × ℝ × ℝ) : Prop :=
  let (A, B, C) := line in
  A * P.x + B * P.y + C = 0

-- Main statement with "sorry" indicating omitted proof
theorem circle_and_tangent_line :
  ∀ (O P : Point),
    centered_at_origin O →
    tangent_to_line O 2 (1, -√3, -4) →
    equation_of_circle O 2 (1, 1, 4) →
    P = ⟨ 2, 3 ⟩ →
    equation_of_line_through_point P (1, 0, -2) ∨ equation_of_line_through_point P (5, -12, 26) :=
by
  intros
  sorry

end circle_and_tangent_line_l15_15742


namespace at_least_six_parallelogram_faces_l15_15402

-- Definitions of the parameters involved
variables (c e l p : ℕ)
variable (C_sym : ℕ → Prop) -- C_sym(f) indicates f is centrally symmetric

-- Conditions
axiom faces_centrally_symmetric : ∀ f, C_sym(f)
axiom euler_formula : c - e + l = 2
axiom min_edges_per_vertex : 2 * e ≥ 3 * c
axiom double_edges_per_face : ∀ (x: ℕ), x > 0 → ∃ y, 2 * y = x -- Each edge belongs to exactly two faces
axiom even_sides_per_face : ∀ f, C_sym(f) → ∃ k : ℕ, k * 2 = f
axiom sides_of_faces : ∀ f, (C_sym(f) ∧ f = 4) ∨ (C_sym(f) ∧ f ≥ 6)

-- Proof statement
theorem at_least_six_parallelogram_faces : p ≥ 6 :=
by
  sorry

end at_least_six_parallelogram_faces_l15_15402


namespace sin_fourth_cos_fourth_l15_15366

theorem sin_fourth_cos_fourth (x : ℝ) (h : sin (2 * x) = 1 / 7) : sin(x)^4 + cos(x)^4 = 97 / 98 :=
by
  sorry

end sin_fourth_cos_fourth_l15_15366


namespace least_positive_number_divisible_by_primes_l15_15029

theorem least_positive_number_divisible_by_primes :
  ∃ n : ℕ, n > 0 ∧
    (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧
    (∀ m : ℕ, (m > 0 ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m)) → n ≤ m) ∧
    n = 2310 :=
by
  sorry

end least_positive_number_divisible_by_primes_l15_15029


namespace student_A_more_stable_l15_15497

-- Defining the variances of students A and B as constants
def S_A_sq : ℝ := 0.04
def S_B_sq : ℝ := 0.13

-- Statement of the theorem
theorem student_A_more_stable : S_A_sq < S_B_sq → true :=
by
  -- proof will go here
  sorry

end student_A_more_stable_l15_15497


namespace least_positive_number_divisible_by_five_primes_l15_15057

theorem least_positive_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ 
    (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) :=
by
  use 2310
  split
  { exact by norm_num }
  split
  { intros p hp, fin_cases hp; norm_num }
  { intros m hm, apply nat.le_of_dvd; norm_num }
  sorry

end least_positive_number_divisible_by_five_primes_l15_15057


namespace fraction_of_income_from_tips_l15_15515

variable (S T I : ℝ)

-- Conditions
def tips_as_fraction_of_salary : Prop := T = (3/4) * S
def total_income : Prop := I = S + T

-- Theorem stating the proof problem
theorem fraction_of_income_from_tips 
  (h1 : tips_as_fraction_of_salary S T)
  (h2 : total_income S T I) : (T / I) = 3 / 7 := by
  sorry

end fraction_of_income_from_tips_l15_15515


namespace max_value_fraction_l15_15634

theorem max_value_fraction (x : ℝ) : x ≠ 0 → 1 / (x^4 + 4*x^2 + 2 + 8/x^2 + 16/x^4) ≤ 1 / 31 :=
by sorry

end max_value_fraction_l15_15634


namespace max_additional_viewers_l15_15586

-- Given conditions:
-- viewers_rating_seq is a sequence of integer ratings given by viewers
-- At moment T, the rating (sum of all given scores divided by their number) is an integer
-- With each new voting viewer, the rating decreases by one unit

theorem max_additional_viewers (rating : ℕ → ℕ) (T : ℕ)
  (h_initial: ∃ x n, x ∈ (set.range (λ i, rating i)) ∧ n = (count (λ i, i ≤ T) rating)
    ∧ (∑ i in finset.range (n+1), rating i) / (n+1) = x)
  (h_decrease: ∀ m > T, (∑ i in finset.range (m+1), rating i) / (m+1) = (h_initial.some - (m - T))):
  ∃ n, n = 5 := 
by 
  sorry

end max_additional_viewers_l15_15586


namespace find_A_l15_15325

theorem find_A (A : ℕ) (h1 : A < 5) (h2 : (9 * 100 + A * 10 + 7) / 10 * 10 = 930) : A = 3 :=
sorry

end find_A_l15_15325


namespace lily_read_total_books_l15_15401

-- Definitions
def books_weekdays_last_month : ℕ := 4
def books_weekends_last_month : ℕ := 4

def books_weekdays_this_month : ℕ := 2 * books_weekdays_last_month
def books_weekends_this_month : ℕ := 3 * books_weekends_last_month

def total_books_last_month : ℕ := books_weekdays_last_month + books_weekends_last_month
def total_books_this_month : ℕ := books_weekdays_this_month + books_weekends_this_month
def total_books_two_months : ℕ := total_books_last_month + total_books_this_month

-- Proof problem statement
theorem lily_read_total_books : total_books_two_months = 28 :=
by
  sorry

end lily_read_total_books_l15_15401


namespace find_c_l15_15964

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 + 19 * x - 84
noncomputable def g (x : ℝ) : ℝ := 4 * x ^ 2 - 12 * x + 5

theorem find_c (c : ℝ) 
  (h1 : ∃ x : ℝ, (⌊c⌋ : ℝ) = x ∧ f x = 0)
  (h2 : ∃ x : ℝ, (c - ⌊c⌋) = x ∧ g x = 0) :
  c = -23 / 2 := by
  sorry

end find_c_l15_15964


namespace triangle_area_16_l15_15859

theorem triangle_area_16 : 
  let A := (0, 0)
  let B := (4, 0)
  let C := (3, 8)
  let base := (B.1 - A.1)
  let height := (C.2 - A.2)
  (base * height) / 2 = 16 := by
  sorry

end triangle_area_16_l15_15859


namespace third_smallest_is_four_probability_l15_15417

noncomputable def probability_third_smallest_is_four : ℚ :=
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 3 2) * (Nat.choose 8 4)
  favorable_ways / total_ways

theorem third_smallest_is_four_probability : 
  probability_third_smallest_is_four = 35 / 132 := 
sorry

end third_smallest_is_four_probability_l15_15417


namespace common_volume_of_cylinders_l15_15203

-- Define the cylinders
def cylinder1 (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = a^2
def cylinder2 (a : ℝ) (y z : ℝ) : Prop := y^2 + z^2 = a^2

-- Define the volume calculation problem
theorem common_volume_of_cylinders (a : ℝ) : 
  (∀ x y z : ℝ, cylinder1 a x y → cylinder2 a y z → (0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) → x^2 + y^2 ≤ a^2 ∧ y^2 + z^2 ≤ a^2) → 
  volume : ℝ := 
  ∃ x y z : ℝ, (cylinder1 a x y ∧ cylinder2 a y z) ∧ (0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) ∧ volume = (16 / 3) * a^3

end common_volume_of_cylinders_l15_15203


namespace tangent_line_at_1_range_of_a_iff_monotonic_increasing_l15_15260

noncomputable def f (x a : ℝ) : ℝ := (1 / x + a) * real.log (1 + x)

theorem tangent_line_at_1 (a : ℝ) (h : a = -1) :
  let f_at := (1 / (1:ℝ) + a) * real.log (1 + 1)
  let f_prime := -real.log 2
  let tangent_line := λ x : ℝ, f_prime * (x - 1) + f_at in
  tangent_line = λ x, -real.log 2 * x + real.log 2 :=
by {
  -- Proof to be provided
  sorry
}

theorem range_of_a_iff_monotonic_increasing :
  ∀ x : ℝ, x > 0 → monotonic_increasing (λ a : ℝ, (1/x + a) * real.log (1 + x)) ↔ a ∈ set.Ici (1/2) :=
by {
  -- Proof to be provided
  sorry
}

end tangent_line_at_1_range_of_a_iff_monotonic_increasing_l15_15260


namespace totalBottleCaps_l15_15845

-- Variables for the conditions
def bottleCapsPerBox : ℝ := 35.0
def numberOfBoxes : ℝ := 7.0

-- Theorem stating the equivalent proof problem
theorem totalBottleCaps : bottleCapsPerBox * numberOfBoxes = 245.0 := by
  sorry

end totalBottleCaps_l15_15845


namespace sin_double_angle_zero_l15_15386

theorem sin_double_angle_zero {a : ℝ} (f : ℝ → ℝ) (h1 : ∀ x, f x = sin x - cos x) (h2 : f a = 1) : sin (2 * a) = 0 := by
  sorry

end sin_double_angle_zero_l15_15386


namespace intervals_of_increase_triangle_properties_l15_15211

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6) - 1

-- Given conditions for intervals of increase
theorem intervals_of_increase (k : ℤ) :
  let I := set.Icc (↑k * Real.pi - Real.pi / 6) (↑k * Real.pi + Real.pi / 3)
  ∃ I, ∀ x y ∈ I, x < y → f x < f y :=
sorry

-- Triangle properties and area calculation
theorem triangle_properties :
  let B := Real.pi / 3
  let b := Real.sqrt 7
  let fB := 0 -- since f(B) = 0
  (∀ A C : ℝ, Real.sin A = 3 * Real.sin C) →
  ∃ a c : ℝ, a = 3 * c ∧ c = 1 ∧ a = 3 ∧
    let area := 1 / 2 * a * c * Real.sin B
    area = 3 * Real.sqrt 3 / 4 :=
sorry

end intervals_of_increase_triangle_properties_l15_15211


namespace num_distinct_real_solutions_l15_15376

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem num_distinct_real_solutions :
  {c : ℝ | f (f (f (f c))) = 3}.to_finset.card = 9 :=
by
  sorry

end num_distinct_real_solutions_l15_15376


namespace avg_problem_l15_15801

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of three numbers
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

-- Formulate the proof problem statement
theorem avg_problem : avg3 (avg3 1 1 0) (avg2 0 1) 0 = 7 / 18 := by
  sorry

end avg_problem_l15_15801


namespace maximum_value_of_d_l15_15191

theorem maximum_value_of_d 
  (d e : ℕ) 
  (h1 : 0 ≤ d ∧ d < 10) 
  (h2: 0 ≤ e ∧ e < 10) 
  (h3 : (18 + d + e) % 3 = 0) 
  (h4 : (15 - (d + e)) % 11 = 0) 
  : d ≤ 0 := 
sorry

end maximum_value_of_d_l15_15191


namespace decreasing_f_l15_15988

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≥ 1 then -4 * x + 2 * a else x^2 - a * x + 4

theorem decreasing_f (a : ℝ) : (∀ x, f a x = (if x ≥ 1 then -4 * x + 2 * a else x^2 - a * x + 4)) →
  (∀ x, differentiable_at ℝ (f a) x) →
  (∀ x, deriv (f a) x < 0) ↔ (2 < a ∧ a ≤ 3) :=
sorry

end decreasing_f_l15_15988


namespace OfficerHoppsTotalTickets_l15_15394

theorem OfficerHoppsTotalTickets : 
  (15 * 8 + (31 - 15) * 5 = 200) :=
  by
    sorry

end OfficerHoppsTotalTickets_l15_15394


namespace pentagon_coloring_number_l15_15204

/-- 
A convex pentagon with unequal sides and each side colored with one of three colors (red, yellow, or blue) such that no two adjacent sides are the same color, 
has exactly 30 different coloring methods.
-/
theorem pentagon_coloring_number : 
  let pentagon := ℕ -> ℕ in
  ∃ n : ℕ, (n = 30) ∧ 
  (∀ f : fin 5 → fin 3, 
    (∀ i : fin 5, f i ≠ f ((i + 1) % 5)) →
    (∃ f_shortest : (fin 5 → fin 3) → fin 3,
      (f_shortest f = 3))) :=
sorry

end pentagon_coloring_number_l15_15204


namespace whipped_cream_needed_l15_15185

/- Problem conditions -/
def pies_per_day : ℕ := 3
def days : ℕ := 11
def pies_total : ℕ := pies_per_day * days
def pies_eaten_by_tiffany : ℕ := 4
def pies_remaining : ℕ := pies_total - pies_eaten_by_tiffany
def whipped_cream_per_pie : ℕ := 2

/- Proof statement -/
theorem whipped_cream_needed : whipped_cream_per_pie * pies_remaining = 58 := by
  sorry

end whipped_cream_needed_l15_15185


namespace fraction_of_women_l15_15737

-- Define the total number of people
def total_people : ℕ := 80

-- Define the fraction of married people
def fraction_married : ℚ := 3 / 4

-- Define the number of unmarried women
def max_unmarried_women : ℕ := 20

-- Statement to prove that the fraction of people who are women is 1/4
theorem fraction_of_women : 
  let fraction_unmarried := 1 / 4,
      total_unmarried := fraction_unmarried * total_people,
      number_of_women := max_unmarried_women in
  number_of_women = (1 / 4) * total_people := 
    by
    sorry

end fraction_of_women_l15_15737


namespace degree_of_poly_l15_15611

-- Define the polynomials
def poly1 := (x^2 + x + 1)
def poly2 := (x^4 + 1)

-- State the theorem
theorem degree_of_poly :
  degree ((poly1^5) * (poly2^2)) = 18 :=
by
  sorry

end degree_of_poly_l15_15611


namespace triangle_side_sum_l15_15492

theorem triangle_side_sum (A B C : Type) [geometry : Triangle A B C] [Angle A = 50] [Angle C = 40]
  (side_opposite_C : length (side B A) = 8 * real.sqrt 3) :
  length (side A B) + length (side B C) = 59.5 :=
by sorry

end triangle_side_sum_l15_15492


namespace minimum_cable_length_l15_15846

def station_positions : List ℝ := [0, 3, 7, 11, 14]

def total_cable_length (x : ℝ) : ℝ :=
  abs x + abs (x - 3) + abs (x - 7) + abs (x - 11) + abs (x - 14)

theorem minimum_cable_length :
  (∀ x : ℝ, total_cable_length x ≥ 22) ∧ total_cable_length 7 = 22 :=
by
  sorry

end minimum_cable_length_l15_15846


namespace intersection_points_count_l15_15709

def line1 (x y : ℝ) : Prop := 4 * y - 3 * x = 2
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y = 9
def line3 (x y : ℝ) : Prop := x - y = 1

theorem intersection_points_count :
  ∃ p1 p2 p3 : ℝ × ℝ,
  (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
  (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
  (line1 p3.1 p3.2 ∧ line3 p3.1 p3.2) ∧
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3) :=
  sorry

end intersection_points_count_l15_15709


namespace addition_subtraction_questions_l15_15480

theorem addition_subtraction_questions (total_questions word_problems answered_questions add_sub_questions : ℕ)
  (h1 : total_questions = 45)
  (h2 : word_problems = 17)
  (h3 : answered_questions = total_questions - 7)
  (h4 : add_sub_questions = answered_questions - word_problems) : 
  add_sub_questions = 21 := 
by 
  -- the proof steps are skipped
  sorry

end addition_subtraction_questions_l15_15480


namespace polygon_sides_given_interior_angle_l15_15911

theorem polygon_sides_given_interior_angle
  (h : ∀ (n : ℕ), (n > 2) → ((n - 2) * 180 = n * 140)): n = 9 := by
  sorry

end polygon_sides_given_interior_angle_l15_15911


namespace polygon_area_is_odd_l15_15899

open Int

structure RectilinearPolygon :=
  (vertices : List (ℤ × ℤ))
  (sides : ℕ)
  (side_lengths : List ℤ)
  (parallel_to_axes : ∀ idx < sides, (vertices[idx].fst = vertices[(idx+1) % sides].fst) ∨ (vertices[idx].snd = vertices[(idx+1) % sides].snd))
  (integer_coordinates : ∀ v : vertices, (∃ x y : ℤ, v = (x, y)))
  (odd_side_lengths : ∀ l : side_lengths, (l % 2 = 1))

def is_rectilinear_polygon (p : RectilinearPolygon) : Prop :=
  p.sides = 100 ∧ p.parallel_to_axes ∧ p.integer_coordinates ∧ p.odd_side_lengths

theorem polygon_area_is_odd : ∀ p : RectilinearPolygon, is_rectilinear_polygon p → (∃ area : ℤ, area % 2 = 1) :=
by
  sorry

end polygon_area_is_odd_l15_15899


namespace find_first_number_l15_15896

theorem find_first_number : ∃ x : ℕ, x + 7314 = 3362 + 13500 ∧ x = 9548 :=
by
  -- This is where the proof would go
  sorry

end find_first_number_l15_15896


namespace least_positive_number_divisible_by_five_smallest_primes_l15_15027

theorem least_positive_number_divisible_by_five_smallest_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  Nat.min (p1 * p2 * p3 * p4 * p5) := 
  2310 :=
by
  sorry

end least_positive_number_divisible_by_five_smallest_primes_l15_15027


namespace log_sum_real_coeffs_expansion_l15_15172

theorem log_sum_real_coeffs_expansion :
  let S := 2 ^ 1010 in log2 S = 1010 :=
by
  let S : ℝ := 2 ^ 1010
  have h1 : S = 2 ^ 1010 := rfl
  have h2 : log2 S = log2 (2 ^ 1010) := congr_arg log2 h1
  rw [log2_pow] at h2
  exact h2

end log_sum_real_coeffs_expansion_l15_15172


namespace keegan_total_school_time_l15_15364

-- Definition of the conditions
def keegan_classes : Nat := 7
def history_and_chemistry_time : ℝ := 1.5
def other_class_time : ℝ := 1.2

-- The theorem stating that given these conditions, Keegan spends 7.5 hours a day in school.
theorem keegan_total_school_time : 
  (history_and_chemistry_time + 5 * other_class_time) = 7.5 := 
by
  sorry

end keegan_total_school_time_l15_15364


namespace problem_statement_l15_15759

noncomputable def T := { t : ℝ × ℝ × ℝ | 0 ≤ t.1 ∧ 0 ≤ t.2 ∧ 0 ≤ t.3 ∧ t.1 + t.2 + t.3 = 1 }
def supports (t a : ℝ × ℝ × ℝ) := (t.1 ≥ a.1 ∧ t.2 ≥ a.2 ∧ t.3 < a.3) ∨ (t.1 ≥ a.1 ∧ t.2 < a.2 ∧ t.3 ≥ a.3) ∨ (t.1 < a.1 ∧ t.2 ≥ a.2 ∧ t.3 ≥ a.3)

noncomputable def S := { t : ℝ × ℝ × ℝ | t ∈ T ∧ supports t (1/4, 1/4, 1/4) }

noncomputable def area (A : set (ℝ × ℝ × ℝ)) : ℝ :=
  -- Placeholder for the actual area computation function
  sorry

theorem problem_statement :
  (area S) / (area T) = 3 / 4 :=
sorry

end problem_statement_l15_15759


namespace Jasmine_initial_percentage_is_5_l15_15892

noncomputable def initial_percentage_of_jasmine 
  (V_initial : ℕ := 90) 
  (V_added_jasmine : ℕ := 8) 
  (V_added_water : ℕ := 2) 
  (V_final : ℕ := 100) 
  (P_final : ℚ := 12.5 / 100) : ℚ := 
  (P_final * V_final - V_added_jasmine) / V_initial * 100

theorem Jasmine_initial_percentage_is_5 :
  initial_percentage_of_jasmine = 5 := 
by 
  sorry

end Jasmine_initial_percentage_is_5_l15_15892


namespace min_sum_rationalize_denom_is_64_l15_15790

def rationalize_denom (x y : ℝ) : ℝ := x / y

theorem min_sum_rationalize_denom_is_64 :
  let A := 25
  let B := 2
  let C := 20
  let D := 17
  rationalize_denom (real.sqrt 50) (real.sqrt 25 - 2 * real.sqrt 2) = (A * real.sqrt B + C) / D ∧
  D > 0 ∧ ∀ p : ℕ, nat.prime p → p * p ∣ B → false →
  A + B + C + D = 64 := by
begin
  sorry
end

end min_sum_rationalize_denom_is_64_l15_15790


namespace sum_of_distinct_prime_divisors_of_1800_eq_10_l15_15506

theorem sum_of_distinct_prime_divisors_of_1800_eq_10 :
  ∑ p in {2, 3, 5}, p = 10 :=
by
  sorry

end sum_of_distinct_prime_divisors_of_1800_eq_10_l15_15506


namespace mario_haircut_price_l15_15602

theorem mario_haircut_price (P : ℝ) 
  (weekend_multiplier : ℝ := 1.50)
  (sunday_price : ℝ := 27) 
  (weekend_price_eq : sunday_price = P * weekend_multiplier) : 
  P = 18 := 
by
  sorry

end mario_haircut_price_l15_15602


namespace find_sum_of_coefficients_l15_15954

theorem find_sum_of_coefficients : 
  (∃ m n p : ℕ, 
    (n.gcd p = 1) ∧ 
    m + 36 = 72 ∧
    n + 33*3 = 103 ∧ 
    p = 3 ∧ 
    (72 + 33 * ℼ + (8 * (1/8 * (4 * π / 3))) + 36) = m + n * π / p) → 
  m + n + p = 430 :=
by {
  sorry
}

end find_sum_of_coefficients_l15_15954


namespace marks_lost_per_wrong_answer_l15_15739

theorem marks_lost_per_wrong_answer :
  ∃ x : ℕ, 
    let correct_answers := 34 in
    let total_questions := 60 in
    let total_marks := 110 in
    let marks_per_correct := 4 in
    let wrong_answers := total_questions - correct_answers in
    let marks_from_correct := correct_answers * marks_per_correct in
    marks_from_correct - wrong_answers * x = total_marks ∧ x = 1 :=
by
  sorry

end marks_lost_per_wrong_answer_l15_15739


namespace crates_with_oranges_l15_15573

theorem crates_with_oranges : ∀ (S : set ℕ), S.card = 150 →
  (∀ s ∈ S, 125 ≤ s ∧ s ≤ 149) →
  ∃ n, n = 6 ∧ ∀ T, T ⊆ S ∧ T.nonempty → T.card = 25 → (∃ t ∈ T, ∀ v ∈ T, t = v) :=
by
  intro S hS hBounds
  use 6
  sorry -- proof omitted

end crates_with_oranges_l15_15573


namespace quadrilateral_perimeter_l15_15161

open Real

/-- Define the points as a structure with x and y coordinates. -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Define the four points A, B, C, D as given in the problem. -/
def A : Point := { x := 0, y := 0 }
def B : Point := { x := 0, y := 10 }
def C : Point := { x := 8, y := 10 }
def D : Point := { x := 8, y := 0 }

/-- Helper function to calculate the distance between two points. -/
def dist (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Define the distances between consecutive vertices A, B, C, D, and back to A. -/
def AB := dist A B
def BC := dist B C
def CD := dist C D
def DA := dist D A

/-- The perimeter of the quadrilateral is the sum of the distances between consecutive vertices. -/
def perimeter : ℝ := AB + BC + CD + DA

/-- The theorem to prove that the perimeter of the quadrilateral is 36. -/
theorem quadrilateral_perimeter : perimeter = 36 := by
  sorry

end quadrilateral_perimeter_l15_15161


namespace sum_of_consecutive_odds_eq_169_l15_15507

theorem sum_of_consecutive_odds_eq_169 : 
  ∃ n : ℕ, (∑ i in Finset.range n, 2 * i + 1) = 169 ∧ (2 * n - 1) = 25 := 
by
  sorry

end sum_of_consecutive_odds_eq_169_l15_15507


namespace count_324_four_digit_numbers_l15_15208

def even_digit_sum (n: ℕ) : Prop :=
  let digits := List.map (λ x, n / 10^x % 10) [0, 1, 2, 3]
  (digits.headI + digits.getI 1 + digits.get 2) % 2 = 0

noncomputable def count_valid_numbers : ℕ :=
  (finset.range 10000).filter (λ n, ∀ d in n.digits, d ∈ {0, 1, 2, 3, 4, 5, 6} ∧ even_digit_sum n).card

theorem count_324_four_digit_numbers :
  count_valid_numbers = 324 :=
sorry

end count_324_four_digit_numbers_l15_15208


namespace cube_face_sum_l15_15176

theorem cube_face_sum
  (a d b e c f : ℕ)
  (pos_a : 0 < a) (pos_d : 0 < d) (pos_b : 0 < b) (pos_e : 0 < e) (pos_c : 0 < c) (pos_f : 0 < f)
  (hd : (a + d) * (b + e) * (c + f) = 2107) :
  a + d + b + e + c + f = 57 :=
sorry

end cube_face_sum_l15_15176


namespace boxes_filled_per_week_l15_15931

theorem boxes_filled_per_week : 
  (let hens := 270
       eggs_per_hen_per_day := 1
       days_in_week := 7
       eggs_per_week := hens * eggs_per_hen_per_day * days_in_week
       boxes_capacity := 6
   in eggs_per_week / boxes_capacity) = 315 :=
by
  let hens := 270
  let eggs_per_hen_per_day := 1
  let days_in_week := 7
  let eggs_per_week := hens * eggs_per_hen_per_day * days_in_week
  let boxes_capacity := 6
  have eggs_per_week_calc: eggs_per_week = 1890 := by 
  { calc 
      eggs_per_week = hens * eggs_per_hen_per_day * days_in_week : rfl
      ... = 270 * 1 * 7 : rfl
      ... = 1890 : by norm_num },
  
  have boxes_filled_calc: eggs_per_week / boxes_capacity = 315 := by 
  { calc 
      eggs_per_week / boxes_capacity = 1890 / boxes_capacity : by rw eggs_per_week_calc
      ... = 1890 / 6 : rfl
      ... = 315 : by norm_num },
  
  exact boxes_filled_calc

end boxes_filled_per_week_l15_15931


namespace find_g2_l15_15831

theorem find_g2
  (g : ℝ → ℝ)
  (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x ^ 2) :
  g 2 = 19 / 16 := 
sorry

end find_g2_l15_15831


namespace fencing_cost_l15_15970

noncomputable def diameter : ℝ := 14
noncomputable def cost_per_meter : ℝ := 2.50
noncomputable def pi := Real.pi

noncomputable def circumference (d : ℝ) : ℝ := pi * d

noncomputable def total_cost (c : ℝ) (r : ℝ) : ℝ := r * c

theorem fencing_cost : total_cost (circumference diameter) cost_per_meter = 109.95 := by
  sorry

end fencing_cost_l15_15970


namespace spacy_subsets_of_12_l15_15164

-- Definition of a spacy set
def isSpacy (s : Set ℕ) : Prop :=
  ∀ n ∈ s, ∀ m ∈ s, ∀ k ∈ s, n < m → m < k → k < n + 3 → False

-- Define the set {1, 2, ..., 12}
def S : Set ℕ := { x | 1 ≤ x ∧ x ≤ 12 }

-- The number of spacy subsets of S
def numSpacySubsets (S : Set ℕ) : ℕ := 
  { T | T ⊆ S ∧ isSpacy T }.toFinset.card

theorem spacy_subsets_of_12 :
  numSpacySubsets S = 129 :=
by
  sorry

end spacy_subsets_of_12_l15_15164


namespace least_positive_number_divisible_by_five_primes_l15_15049

theorem least_positive_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ 
    (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) :=
by
  use 2310
  split
  { exact by norm_num }
  split
  { intros p hp, fin_cases hp; norm_num }
  { intros m hm, apply nat.le_of_dvd; norm_num }
  sorry

end least_positive_number_divisible_by_five_primes_l15_15049


namespace stephanie_fewer_forks_l15_15432

noncomputable def fewer_forks := 
  (60 - 44) / 4

theorem stephanie_fewer_forks : fewer_forks = 4 := by
  sorry

end stephanie_fewer_forks_l15_15432


namespace max_pairs_in_grid_l15_15187

def is_multiple (a b : ℕ) : Prop := a % b = 0 ∨ b % a = 0

def is_adjacent (i j : ℕ) (a b : Fin 9) (grid : Fin 9 → Fin 9 → ℕ) : Prop :=
  (j - i = 1 ∨ i - j = 1 ∨ (i = 3 ∧ j = 6) ∨ (i = 6 ∧ j = 3)) ∧ 
  (is_multiple (grid i a) (grid j b) ∨ is_multiple (grid j b) (grid i a))

def count_pairs (grid : Fin 9 → Fin 9 → ℕ) : ℕ :=
  Finset.card (Finset.filter (λ (p : Fin 9 × Fin 9),
    let (i, j) := p in is_multiple (grid i (Fin i)) (grid j (Fin j)) ∨ is_multiple (grid j (Fin j)) (grid i (Fin i)))
  (Finset.univ.product Finset.univ))

theorem max_pairs_in_grid : ∃ grid : Fin 9 → Fin 9 → ℕ, count_pairs grid = 9 :=
sorry

end max_pairs_in_grid_l15_15187


namespace scooter_owners_without_bikes_l15_15734

theorem scooter_owners_without_bikes:
    let total_adults := 400
    let scooter_owners := 370
    let bike_owners := 80
    let both_owners := scooter_owners + bike_owners - total_adults
  in scooter_owners - both_owners = 320 :=
by
  let total_adults := 400
  let scooter_owners := 370
  let bike_owners := 80
  let both_owners := scooter_owners + bike_owners - total_adults
  have h : scooter_owners - both_owners = 320 := by 
    sorry
  exact h

end scooter_owners_without_bikes_l15_15734


namespace volume_ratio_l15_15223

-- Define the tetrahedron and centroid conditions
variable {A B C D D_0 A' B' C' : Type}
variable [Add A] [Add B] [Add C] [Add D] [Add D_0] [Add A'] [Add B'] [Add C']

-- Assume the volumes can be calculated
noncomputable def volume_tetrahedron (P Q R S : Type) : ℝ := sorry

-- Volume relationship theorem for the given tetrahedron configuration
theorem volume_ratio (hD_0 : ∀ (A B C D : Type), D_0 = (A + B + C) / 3)  
                      (hLines : ∀ (A B C D D_0: Type), 
                          (A', B', C') = (line_parallel A D_0 BCD, 
                                          line_parallel B D_0 CAD, 
                                          line_parallel C D_0 ABD)) :
  volume_tetrahedron A B C D = (1/3) * volume_tetrahedron A' B' C' D_0 :=
begin
  sorry,
end

end volume_ratio_l15_15223


namespace discount_problem_l15_15842

theorem discount_problem (x : ℝ) (h : 560 * (1 - x / 100) * 0.70 = 313.6) : x = 20 := 
by
  sorry

end discount_problem_l15_15842


namespace least_positive_number_divisible_by_five_primes_l15_15048

theorem least_positive_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ 
    (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) :=
by
  use 2310
  split
  { exact by norm_num }
  split
  { intros p hp, fin_cases hp; norm_num }
  { intros m hm, apply nat.le_of_dvd; norm_num }
  sorry

end least_positive_number_divisible_by_five_primes_l15_15048


namespace infinitely_many_n_divisible_by_d_l15_15205

noncomputable def exponent_of_prime_in_factorial (p : ℕ) (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), Nat.factorization i p

theorem infinitely_many_n_divisible_by_d
  (d : ℕ) (k : ℕ) (p : Fin k → ℕ) (h_prime : ∀ i, Nat.prime (p i)) :
  ∃ᶠ n in Filter.at_top, ∀ i, d ∣ exponent_of_prime_in_factorial (p i) n := by
  sorry

end infinitely_many_n_divisible_by_d_l15_15205


namespace least_positive_divisible_by_five_primes_l15_15047

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l15_15047


namespace part1_eq_tangent_line_at_one_l15_15285

def f (x a : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

theorem part1_eq_tangent_line_at_one (a : ℝ) (h : a = -1) :
  ∀ x y : ℝ, 
  f x a = (1/x - 1) * Real.log (1 + x) →
  x = 1 →
  y = 0 →
  (Real.log 2) * x + y - Real.log 2 = 0 := 
sorry

end part1_eq_tangent_line_at_one_l15_15285


namespace PQ_parallel_AB_l15_15808

-- Definitions as per the conditions in the problem statement
variables {A B C D E O P Q : Point}
variable [plane_geometry] -- Assume a plane geometry context

-- Conditions:
-- 1. The center O of the circumcircle of quadrilateral ABCD lies on the side AB.
axiom center_O : Circumcenter ABCD O ∧ Collinear {A, B, O}

-- 2. Point E is symmetric to D with respect to the line AB.
axiom symmetric_E : SymmetricPoint D A B E

-- 3. Segments AC and DO intersect at point P.
axiom intersection_P : Intersect (Line A C) (Line D O) P

-- 4. Segments BD and CE intersect at point Q.
axiom intersection_Q : Intersect (Line B D) (Line C E) Q

-- Prove that PQ is parallel to AB.
theorem PQ_parallel_AB : Parallel (Line P Q) (Line A B) :=
by
  sorry

end PQ_parallel_AB_l15_15808


namespace solve_for_n_l15_15423

theorem solve_for_n :
  ∃ n : ℝ, 0.07 * n + 0.12 * (30 + n) + 0.04 * n = 20.4 ∧ n = 73.0434782609 :=
by
  exists 73.0434782609
  split
  . sorry
  . rfl

end solve_for_n_l15_15423


namespace correct_operation_l15_15076

variable (a b : ℝ)

theorem correct_operation : 2 * (a - 1) = 2 * a - 2 :=
sorry

end correct_operation_l15_15076


namespace k_parallel_k_perpendicular_l15_15230

-- Define the parametric equation of line l_1
def l1 (t : ℝ) : ℝ × ℝ := (1 - 2 * t, 2 + k * t)

-- Define the parametric equation of line l_2
def l2 (s : ℝ) : ℝ × ℝ := (s, 1 - 2 * s)

-- Condition for parallel lines (l_1 parallel to l_2)
-- Slope of l_1 is -k/2 and slope of l_2 is -2
def parallel_condition (k : ℝ) : Prop :=
  -2 = -k / 2

-- Condition for perpendicular lines (l_1 perpendicular to l_2)
-- Product of slopes is -1
def perpendicular_condition (k : ℝ) : Prop :=
  (-2) * (k / -2) = -1

-- Prove that k = 4 if l_1 and l_2 are parallel
theorem k_parallel (k : ℝ) (h : parallel_condition k) : k = 4 := 
by
  sorry

-- Prove that k = -1 if l_1 and l_2 are perpendicular
theorem k_perpendicular (k : ℝ) (h : perpendicular_condition k) : k = -1 := 
by
  sorry

end k_parallel_k_perpendicular_l15_15230


namespace rectangle_area_exceeds_m_l15_15341

theorem rectangle_area_exceeds_m (m : ℤ) (h_m : m > 12) :
  ∃ x y : ℤ, x * y > m ∧ (x - 1) * y < m ∧ x * (y - 1) < m :=
by
  sorry

end rectangle_area_exceeds_m_l15_15341


namespace square_field_area_l15_15142

theorem square_field_area (s A : ℝ) (h1 : 10 * 4 * s = 9280) (h2 : A = s^2) : A = 53824 :=
by {
  sorry -- The proof goes here
}

end square_field_area_l15_15142


namespace maximum_true_statements_l15_15384

open Real

theorem maximum_true_statements (x : ℝ) :
  let stmt1 := (0 < x^2 ∧ x^2 < 1)
  let stmt2 := (x^2 > 1)
  let stmt3 := (-1 < x ∧ x < 0)
  let stmt4 := (0 < x ∧ x < 1)
  let stmt5 := (0 < 2*x - x^2 ∧ 2*x - x^2 < 2)
  (stmt1 ∨ stmt2 ∨ stmt3 ∨ stmt4 ∨ stmt5) ->
  (stmt1 ∧ stmt4 ∧ stmt5) -> 
  (stmt1 ∧ stmt2 = false) ∧ (stmt3 ∧ stmt4 = false) ∧
  (stmt1 ∧ stmt2 ∧ stmt4∧ stmt5 = false) ∧ 
  (stmt3 ∧ stmt2 ∧ stmt4∧ stmt5 = false) ∧ 
  3 :=
by
  sorry

end maximum_true_statements_l15_15384


namespace part1_tangent_line_at_x_eq_1_part2_monotonic_increasing_l15_15275

noncomputable def f (x : ℝ) (a : ℝ) := (1/x + a) * real.log(1 + x)

theorem part1_tangent_line_at_x_eq_1 (x := 1) :
  let a := -1 
  let t : ℝ := real.log 2
  let f := (λ x, (1/x - 1) * real.log (1 + x))
  let tangent_line := λ x y, t * x + y - t 
  (tangent_line 1 (f 1) = 0) :=
by {
  sorry
}

theorem part2_monotonic_increasing (a : ℝ) : 
  (∀ x : ℝ, 0 < x → 0 < f x a) ↔ (a ≥ 1/2) := 
by {
  sorry
}

end part1_tangent_line_at_x_eq_1_part2_monotonic_increasing_l15_15275


namespace probability_third_smallest_is_4_l15_15413

theorem probability_third_smallest_is_4 :
  (∃ (integers : Finset ℕ), integers.card = 7 ∧ integers ⊆ (Finset.range 13).erase 0 ∧ 
  ∃ (S : Finset ℕ), S = (Finset.filter (λ x, x < 4) integers) ∧ S.card = 2 ∧ 
  ∃ (T : Finset ℕ), T = (Finset.filter (λ x, 4 < x) integers) ∧ T.card = 5) → 
  let total_ways := Nat.choose 12 7 in
  let favorable_ways := (Nat.choose 3 2) * (Nat.choose 8 5) in
  (favorable_ways) / total_ways.toReal = 7 / 33 :=
by sorry

end probability_third_smallest_is_4_l15_15413


namespace tangent_line_equation_at_1_monotonically_increasing_range_of_a_l15_15267

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (1 / x + a) * real.log (1 + x)

-- Part 1: Tangent line equation at point (1, f(1)) when a = -1
theorem tangent_line_equation_at_1 (x y : ℝ) (h : y = f 1 (-1)) :
  (real.log 2) * x + y - (real.log 2) = 0 := 
sorry

-- Part 2: Range of a for which f(x) is monotonically increasing on (0, +∞)
theorem monotonically_increasing_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, 0 < x → 0 < f' x a) ↔ (1 / 2 ≤ a) :=  
sorry

end tangent_line_equation_at_1_monotonically_increasing_range_of_a_l15_15267


namespace total_investment_sum_l15_15500

-- Definitions of the problem
variable (Raghu Trishul Vishal : ℕ)
variable (h1 : Raghu = 2000)
variable (h2 : Trishul = Nat.div (Raghu * 9) 10)
variable (h3 : Vishal = Nat.div (Trishul * 11) 10)

-- The theorem to prove
theorem total_investment_sum :
  Vishal + Trishul + Raghu = 5780 :=
by
  sorry

end total_investment_sum_l15_15500


namespace smallest_number_meeting_conditions_l15_15070

theorem smallest_number_meeting_conditions : 
  ∃ a : ℕ, 
    (a % 4 = 1) ∧ 
    (a % 3 = 2) ∧ 
    (a % 5 = 2) ∧ 
    (∀ b : ℕ, 
      (b % 4 = 1) → 
      (b % 3 = 2) → 
      (b % 5 = 2) → 
      a ≤ b) :=
begin
  use 17,
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { sorry }
end

end smallest_number_meeting_conditions_l15_15070


namespace kendra_change_is_correct_l15_15465

-- Define the initial conditions
def price_wooden_toy : ℕ := 20
def price_hat : ℕ := 10
def kendra_initial_money : ℕ := 100
def num_wooden_toys : ℕ := 2
def num_hats : ℕ := 3

-- Calculate the total costs
def total_wooden_toys_cost : ℕ := price_wooden_toy * num_wooden_toys
def total_hats_cost : ℕ := price_hat * num_hats
def total_cost : ℕ := total_wooden_toys_cost + total_hats_cost

-- Calculate the change Kendra received
def kendra_change : ℕ := kendra_initial_money - total_cost

theorem kendra_change_is_correct : kendra_change = 30 := by
  sorry

end kendra_change_is_correct_l15_15465


namespace areas_equal_l15_15156

-- Define the basic geometrical setup and conditions
variables {A B C D E K O : Type} [Parallelogram A B C D] 
          [Line_extends_1 : Extension_line A B E] 
          [Line_extends_2 : Extension_line A D K] 
          [Intersection : Intersects (Line_through B K) (Line_through D E) O]

-- Define the areas of the quadrilaterals
def area_quad_ABOD : ℝ
def area_quad_ECKO : ℝ

-- Main theorem statement
theorem areas_equal (ABCD_is_parallelogram : Parallelogram A B C D)
    (E_on_extension_of_AB : Extension_line A B E)
    (K_on_extension_of_AD : Extension_line A D K)
    (BK_DE_intersect_at_O : Intersects (Line_through B K) (Line_through D E) O) :
  area_quad_ABOD = area_quad_ECKO :=
sorry

end areas_equal_l15_15156


namespace collinear_vectors_l15_15640

def collinear (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2, k * b.3)

theorem collinear_vectors (x y : ℝ) :
  collinear (2 * x, 1, 3) (1, -2 * y, 9) → x = 1/6 ∧ y = -3/2 :=
by sorry

end collinear_vectors_l15_15640


namespace ordered_sum_ge_disordered_sum_ge_reverse_sum_l15_15542

-- Definitions of ordered lists
variables {α : Type*} [linear_ordered_field α]
variables {a b : list α}
variables {p : list ℕ} -- permutation indices
variable {n : ℕ}

-- Assumptions
def ordered_a := ∀ i j, 1 ≤ i → i ≤ j → j ≤ n → a (i-1) ≤ a (j-1)
def ordered_b := ∀ i j, 1 ≤ i → i ≤ j → j ≤ n → b (i-1) ≤ b (j-1)
def complete_perm := p.permutations.contains p
-- Note: Let's assume lengths for simplicity.
def lengths := a.length = n ∧ b.length = n ∧ p.length = n

theorem ordered_sum_ge_disordered_sum_ge_reverse_sum (h1 : ordered_a) (h2 : ordered_b) (h3 : complete_perm) (h4 : lengths) :
(a.zip_with (*) b).sum ≥ (a.zip_with (*) (p.map (λ i, b (i-1)))).sum ∧
(a.zip_with (*) (p.map (λ i, b (i-1)))).sum ≥ (a.zip_with (*) b.reverse).sum := sorry

end ordered_sum_ge_disordered_sum_ge_reverse_sum_l15_15542


namespace proposition_1_proposition_2_proposition_3_proposition_4_l15_15231

variable {α β γ : Type} [Plane α] [Plane β] [Plane γ]
variable {a b c : Type} [Line a] [Line b] [Line c]
variable (parallel : ∀ {x y : Type} [Plane x] [Plane y], Prop)
variable (perpendicular : ∀ {x y : Type} [Plane x] [Line y], Prop)

/-- Proposition ① -/
theorem proposition_1 (h1 : parallel α β) (h2 : parallel β γ) : parallel α γ := by
  sorry

/-- Proposition ② -/
theorem proposition_2
  (h_parallel_lines : parallel a b) (h_parallel_lines_2 : parallel b c)
  (h_perp_a : perpendicular α a) (h_perp_b : perpendicular β b) (h_perp_c : perpendicular γ c) :
  parallel α β ∧ parallel β γ := by
  sorry

/-- Proposition ③ negation -/
theorem proposition_3 :
  ¬ (∀ (a b c α β γ : Type) [Line a] [Line b] [Line c] [Plane α] [Plane β] [Plane γ], 
    parallel a b → parallel b c → parallel a α → parallel b β → parallel c γ → parallel α β ∧ parallel β γ) := by
  sorry

/-- Proposition ④ -/
theorem proposition_4
  (h_perp_a : perpendicular α a) (h_perp_b : perpendicular β b) (h_perp_c : perpendicular γ c)
  (h_parallel_planes : parallel α β) (h_parallel_planes_2 : parallel β γ) :
  parallel a b ∧ parallel b c := by
  sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l15_15231


namespace alex_sandwich_count_l15_15923

theorem alex_sandwich_count :
  (Nat.choose 10 1) * (Nat.choose 12 2) * (Nat.choose 5 1) = 3300 :=
by
  sorry

end alex_sandwich_count_l15_15923


namespace AX_eq_AY_l15_15711

noncomputable def midpoint (A B : Point) : Point := 
  ∃ M, M = (A + B) / 2

noncomputable def circumcenter (A B C : Point) : Point := 
  ∃ O, O := -- appropriate conditions for O being the circumcenter of ΔABC

noncomputable def perpendicular (P Q R: Point) : Prop :=
  -- definition for PQR forming a right angle

noncomputable def intersection_with_circle (A B : Point) (C : Circle) : Point :=
  ∃ P, P ∈ C ∧ -- other intersection conditions

noncomputable def draw_perpendicular (P Q : Point) : Point :=
  ∃ X, X is the intersection of the perpendicular from P to Q

noncomputable def Point_on_line (P Q : Point) : Prop := 
  ∃ X, X ∈ line PQ

theorem AX_eq_AY 
  {A B C O M P X Y : Point}
  (O_circumcenter : circumcenter A B C = O)
  (M_midpoint : midpoint B C = M)
  (OP_perpend_AM : perpendicular O P A M)
  (P_on_circleBOC : intersection_with_circle O P (circleBOC B O C) = P)
  (X_on_BP : draw_perpendicular A O ∈ line BP)
  (Y_on_CP : draw_perpendicular A O ∈ line CP):
  AX = AY :=
by
  sorry

end AX_eq_AY_l15_15711


namespace calculate_speed_in_kmph_l15_15563

-- Definitions based on problem conditions
def distance_meters : ℕ := 1800
def time_minutes : ℕ := 12

-- Conversion factors
def meters_to_kilometers (m : ℕ) : ℝ := m / 1000.0
def minutes_to_hours (min : ℕ) : ℝ := min / 60.0

-- Target the speed calculation
theorem calculate_speed_in_kmph : 
  (meters_to_kilometers distance_meters) / 
  (minutes_to_hours time_minutes) = 9 := 
by 
  sorry

end calculate_speed_in_kmph_l15_15563


namespace identify_rhombus_l15_15338

theorem identify_rhombus (Q : Type) [quadrilateral Q] (parallel_equal_opposite_sides : Prop)
  (all_equal_interior_angles : Prop) (perpendicular_diagonals : Prop) :
  (∃ (all_sides_equal : Prop), (all_sides_equal ↔ is_rhombus Q)) :=
sorry

end identify_rhombus_l15_15338


namespace main_theorem_l15_15381

-- Definitions based on given conditions
def non_negative_seq (a : ℕ → ℝ) (N : ℕ) : Prop := ∀ (i : ℕ), 1 ≤ i ∧ i ≤ N → 0 ≤ a i

def sum_bound (a : ℕ → ℝ) (N : ℕ) : Prop := (finset.sum (finset.range (N + 1)) (λ i, a i)) ≤ 500

-- Statement of the problem
theorem main_theorem (N : ℕ) (hN : N > 1) (a : ℕ → ℝ) 
  (h_non_neg : non_negative_seq a N) (h_sum_bound : sum_bound a N) : 
  ∃ k : ℕ, ∃ (n : ℕ → ℕ), 1 = n 0 ∧ n k = N ∧ (∀ j : ℕ, j < k → n j < n (j + 1)) ∧ 
  (finset.sum (finset.range (k + 1)) (λ i, n i * (a (n (i - 1)))) < 2005 : ℝ) :=
sorry

end main_theorem_l15_15381


namespace clownfish_ratio_l15_15594

theorem clownfish_ratio (C B : ℕ) (h₁ : C = B) (h₂ : C + B = 100) (h₃ : C = B) : 
  (let B := 50; 
  let initially_clownfish := B - 26; -- Number of clownfish that initially joined display tank
  let swam_back := (B - 26) - 16; -- Number of clownfish that swam back
  initially_clownfish > 0 → 
  swam_back > 0 → 
  (swam_back : ℚ) / (initially_clownfish : ℚ) = 1 / 3) :=
by 
  sorry

end clownfish_ratio_l15_15594


namespace find_constant_a_l15_15674

noncomputable def a (f f' : ℝ → ℝ) (x : ℝ) : ℝ :=
  classical.some (
    have h : ∃ a, ∀ t, f t = a * t^3 + f' 2 * t^2 + 3 :=
      ⟨1, λ t, a * t^3 + f' 2 * t^2 + 3⟩
    h
  )

theorem find_constant_a (f f' : ℝ → ℝ) (h₁ : ∀ t, f t = classical.some (
                                      have h : ∃ a, ∀ t, f t = a * t^3 + f' 2 * t^2 + 3 :=
                                      ⟨1, λ t, a * t^3 + f' 2 * t^2 + 3⟩
                                    h
                                    ) * t^3 + f' 2 * t^2 + 3)
                                    (h₂ : f' 1 = -5) : classical.some (
                                      have h : ∃ a, ∀ t, f t = a * t^3 + f' 2 * t^2 + 3 :=
                                      ⟨1, λ t, a * t^3 + f' 2 * t^2 + 3⟩
                                    h) = 1 :=
sorry

end find_constant_a_l15_15674


namespace purely_imaginary_roots_of_positive_real_k_l15_15609

theorem purely_imaginary_roots_of_positive_real_k (k : ℝ) (h : k > 0) :
  let z := ℂ in
  let i_squared := complex.i ^ 2 = -1 in
  ∀ (r1 r2 : ℂ), (8 * r1^2 - 5 * complex.i * r1 - (k : ℂ) = 0) →
                 (8 * r2^2 - 5 * complex.i * r2 - (k : ℂ) = 0) →
                 r1.im ≠ 0 ∧ r2.im ≠ 0 ∧ r1.re = 0 ∧ r2.re = 0 :=
  by
  -- Assign intuitive names to disambiguate the real and imaginary parts.
  intro k h z i_squared r1 r2 h_eq1 h_eq2
  sorry

end purely_imaginary_roots_of_positive_real_k_l15_15609


namespace contradiction_proof_l15_15872

theorem contradiction_proof (a b c d : ℝ) :
  (¬ ((∀ x ∈ {a, b, c, d}, x > 0) → (∃ x₁ x₂, {xℓ : ℝ | xℓ ∈ {a, b, c, d} ∧ xℓ > 0 }.card ≤ 2))) → 
  (¬ (∀ P QR : Prop, (¬ Q → ¬ P) → P) → ∃ S: Prop, S = (a > 0 ∧ b > 0 ∧ (c ≤ 0 ∨ d ≤ 0))) := 
sorry

end contradiction_proof_l15_15872


namespace problem_a_problem_b_l15_15184

variable (α : ℝ)

theorem problem_a (hα : 0 < α ∧ α < π) :
  Real.sqrt ((2 * Real.sin α - Real.sin (2 * α)) / (2 * Real.sin α + Real.sin (2 * α))) = Real.tan (α / 2) :=
sorry

theorem problem_b (hα : π < α ∧ α < 2 * π) :
  Real.sqrt ((2 * Real.sin α - Real.sin (2 * α)) / (2 * Real.sin α + Real.sin (2 * α))) = -Real.tan (α / 2) :=
sorry

end problem_a_problem_b_l15_15184


namespace prob_exactly_M_laws_included_expected_laws_included_l15_15727

variables (K N M : ℕ) (p : ℝ)

-- Definition of the probabilities as given in the conditions and answers
def prob_no_minister_knows_law : ℝ := (1 - p) ^ N
def prob_law_included : ℝ := 1 - prob_no_minister_knows_law p N

-- Part (a)
theorem prob_exactly_M_laws_included :
  (nat.choose K M) * (prob_law_included p N) ^ M * (prob_no_minister_knows_law p N) ^ (K - M) = 
  (nat.choose K M) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) :=
by
  sorry

-- Part (b)
theorem expected_laws_included :
  K * (prob_law_included p N) = K * (1 - (1 - p) ^ N) :=
by
  sorry

end prob_exactly_M_laws_included_expected_laws_included_l15_15727


namespace white_to_red_ratio_l15_15562

-- Definitions based on conditions
def perfect_pink (red white : ℝ) := red + white
def red_paint_ratio : ℝ := 3/4

-- Stating the problem
theorem white_to_red_ratio : ∀ (red white perfect_pink_paint : ℝ),
  perfect_pink_paint = perfect_pink red white →
  15 ∕ perfect_pink_paint = red_paint_ratio →
  white = red → white / red = 1 :=
by sorry

end white_to_red_ratio_l15_15562


namespace cyclic_quadrilateral_diagonal_ratio_l15_15404

variables {R : Type*} [linear_ordered_field R]

-- Define the variables for the side lengths and diagonals of the quadrilateral
variables (a b c d e f : R)

-- Hypothesis: Quadrilateral ABCD is inscribed in a circle
def is_cyclic_quadrilateral (a b c d : R) : Prop := 
-- Placeholder for cyclic quadrilateral property
sorry 

-- The main statement to be proved
theorem cyclic_quadrilateral_diagonal_ratio 
  (h_cyclic : is_cyclic_quadrilateral a b c d) :
  e / f = (a * d + c * b) / (a * b + d * c) :=
sorry

end cyclic_quadrilateral_diagonal_ratio_l15_15404


namespace geometric_sequence_nine_l15_15349

theorem geometric_sequence_nine (a : ℕ → ℝ) (h_geo : ∀ n, a (n + 1) / a n = a 1 / a 0) 
  (h_a1 : a 1 = 2) (h_a5: a 5 = 4) : a 9 = 8 := 
by
  sorry

end geometric_sequence_nine_l15_15349


namespace proof_problem_l15_15244

noncomputable def ellipseEquation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, ⟪(x/a)^2 + (y/b)^2⟫ = 1

noncomputable def lineThroughP (k x0 y0 : ℝ) : ℝ → ℝ := 
  λ x, k * (x - x0) + y0

noncomputable def pointOnEllipse (a b x y : ℝ) : Prop :=
  (x/a)^2 + (y/b)^2 = 1

def existsLine (a b k : ℝ) (P M : ℝ → ℝ) (x1 y1 x2 y2 : ℝ) : Prop :=
  let A := (x1, y1)
  let B := (x2, y2)
  ∃ (l₁ : ℝ → ℝ), (∀ x, P x = l₁ x) ∧ pointOnEllipse a b x1 y1 ∧ pointOnEllipse a b x2 y2 ∧
  ((P x1 - 2) * (P x2 - 2) + (P y1 - 1) * (P y2 - 1) = (M 2 - 2)^2 + (M 1 - 1)^2)

/- The main proof statement combining everything -/
theorem proof_problem : 
  (∃ (a b : ℝ), a² = 4 ∧ b² = 3 ∧ ellipseEquation a b ∧ pointOnEllipse a b 1 (3/2)) ∧
  ∃ (k k1 : ℝ), k = 1/2 ∧ lineThroughP k 2 1 = lineThroughP k1 2 1 ∧ existsLine 2 1 k P M :=
sorry

end proof_problem_l15_15244


namespace geometric_sequence_common_ratio_l15_15128

theorem geometric_sequence_common_ratio :
  ∃ r, r = -1.5 ∧
    (10 * r = -15) ∧ 
    (-15 * r = 22.5) ∧ 
    (22.5 * r = -33.75) :=
begin
  use -1.5,
  split,
  { refl, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, },
end

end geometric_sequence_common_ratio_l15_15128


namespace two_a7_minus_a8_l15_15743

variable (a : ℕ → ℝ) -- Assuming the arithmetic sequence {a_n} is a sequence of real numbers

-- Definitions and conditions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

axiom a1_plus_3a6_plus_a11 : a 1 + 3 * (a 6) + a 11 = 120

-- The theorem to be proved
theorem two_a7_minus_a8 (h : is_arithmetic_sequence a) : 2 * a 7 - a 8 = 24 := 
sorry

end two_a7_minus_a8_l15_15743


namespace cos_angle_bac_l15_15819

theorem cos_angle_bac (A B C P : affine_point ℝ)
  (h1 : circumcenter A B C = P)
  (h2 : vector_to A P = (2 / 5) • (vector_to A B + vector_to A C)) :
  cos_angle A B C = 1 / 4 :=
by
  sorry

end cos_angle_bac_l15_15819


namespace area_of_square_l15_15744

open Classical

noncomputable def square_area (A : ℝ × ℝ) (O : ℝ × ℝ) : ℝ :=
  let d := Real.sqrt ((O.1 - A.1) ^ 2 + (O.2 - A.2) ^ 2)
  in d^2 / 2

theorem area_of_square : square_area (-6, -4) (3, 2) = 58.5 := by
  sorry

end area_of_square_l15_15744


namespace least_number_divisible_by_five_primes_l15_15059

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, (n > 0) ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → p ∣ n) ∧ n = 2310 :=
begin
  sorry
end

end least_number_divisible_by_five_primes_l15_15059


namespace arithmetic_sequence_product_l15_15762

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b n < b (n + 1))
  (h_condition : b 5 * b 6 = 14) :
  (b 4 * b 7 = -324) ∨ (b 4 * b 7 = -36) :=
sorry

end arithmetic_sequence_product_l15_15762


namespace f_2015_value_l15_15995

noncomputable def f : ℝ → ℝ := sorry

theorem f_2015_value : 
  (∀ x : ℝ, f(x + 4) = f(x) + 2 * f(2)) → 
  (f(2) = f(0)) → 
  (f(3) = 2) → 
  f(2015) = 2 :=
by
  intros h1 h2 h3
  sorry

end f_2015_value_l15_15995


namespace bricks_paving_courtyard_l15_15126

noncomputable def bricks_required (length_m : ℕ) (breadth_m : ℕ) (brick_length_cm : ℕ) (brick_breadth_cm : ℕ) : ℕ :=
  let length_cm := length_m * 100
  let breadth_cm := breadth_m * 100
  let area_courtyard := length_cm * breadth_cm
  let area_brick := brick_length_cm * brick_breadth_cm
  (area_courtyard / area_brick).ceil  -- This will be the ceiling value since we need whole bricks.

theorem bricks_paving_courtyard : bricks_required 18 12 15 13 = 11077 :=
  by
    have length_cm := 18 * 100
    have breadth_cm := 12 * 100
    have area_courtyard := length_cm * breadth_cm
    have area_brick := 15 * 13
    have n_bricks := (area_courtyard / area_brick).ceil
    exact n_bricks

end bricks_paving_courtyard_l15_15126


namespace randy_brother_ate_l15_15405

-- Definitions
def initial_biscuits : ℕ := 32
def biscuits_from_father : ℕ := 13
def biscuits_from_mother : ℕ := 15
def remaining_biscuits : ℕ := 40

-- Theorem to prove
theorem randy_brother_ate : 
  initial_biscuits + biscuits_from_father + biscuits_from_mother - remaining_biscuits = 20 :=
by
  sorry

end randy_brother_ate_l15_15405


namespace value_of_xy_l15_15664

noncomputable def distinct_nonzero_reals (x y : ℝ) : Prop :=
x ≠ 0 ∧ y ≠ 0 ∧ x ≠ y

theorem value_of_xy (x y : ℝ) (h : distinct_nonzero_reals x y) (h_eq : x + 4 / x = y + 4 / y) :
  x * y = 4 :=
sorry

end value_of_xy_l15_15664


namespace cube_edge_coloring_probability_l15_15615

/-- 
Each edge of a cube is painted either red or blue, each with a probability of 1/2.
The color of each edge is determined independently.
What is the probability that the cube can be placed on a horizontal surface so that 
every edge around each visible vertical face is of the same color?
-/
theorem cube_edge_coloring_probability : 
  ∀ (E : Type) [Fintype E] (color : E → bool) 
    (h_independent : ∀ e : E, (color e = tt) ∨ (color e = ff))
    (h_prob : ∀ e : E, ∃ p : ℝ, p = 1/2 ∧ p = if color e = tt then 1/2 else 1/2), 
  let cube := { e : E | e ∈ set.univ } in
  let successful_combinations := (2^4 : ℕ) in 
  successful_combinations / 2^12 = 1 / 256 :=
by {
  sorry
}

end cube_edge_coloring_probability_l15_15615


namespace inequality_proof_l15_15537

theorem inequality_proof
  {a b c d e f : ℝ}
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (h_abs : |sqrt (a * d) - sqrt (b * c)| ≤ 1) :
  (a * e + b / e) * (c * e + d / e) ≥ 
    (a^2 * f^2 - (b^2) / (f^2)) * ((d^2) / (f^2) - c^2 * f^2) :=
by
  sorry

end inequality_proof_l15_15537


namespace max_viewers_after_T_l15_15582

theorem max_viewers_after_T 
  (rating : ℕ → ℕ) 
  (sum_ratings : ℕ → ℕ) 
  (n : ℕ) 
  (x : ℕ)
  (x_max : x = 10)
  (n_min : n = 1)
  (H : ∀ m, rating m = sum_ratings m / m)
  (HT : ∃ T, rating T = x ∧ ∀ k > T, rating k = rating (k - 1) - 1) :
  ∃ N, N = 5 := 
begin
  -- proof goes here
  sorry
end

end max_viewers_after_T_l15_15582


namespace least_positive_number_divisible_by_primes_l15_15032

theorem least_positive_number_divisible_by_primes :
  ∃ n : ℕ, n > 0 ∧
    (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧
    (∀ m : ℕ, (m > 0 ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m)) → n ≤ m) ∧
    n = 2310 :=
by
  sorry

end least_positive_number_divisible_by_primes_l15_15032


namespace sum_of_roots_of_Q_sum_of_roots_of_Q_quadratic_l15_15449

open Polynomial

noncomputable def Q (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem sum_of_roots_of_Q (a b c d : ℝ)
  (h : ∀ x : ℝ, Q (x^2 + x - 1) ≥ Q (x^2 + 2)) :
  ∃ (b c : ℝ), Q(x) = b * x^2 + c * x + d :=
begin
  sorry -- Proof not required
end

theorem sum_of_roots_of_Q_quadratic (b c d : ℝ) :
  (∀ Q (x : ℝ), Q(x) = b * x^2 + c * x + d) →
  Polynomial.roots_sum (Q(x)) = -c / b :=
begin
  sorry -- Proof not required
end

end sum_of_roots_of_Q_sum_of_roots_of_Q_quadratic_l15_15449


namespace least_positive_divisible_by_five_primes_l15_15046

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l15_15046


namespace ellipse_equation_max_area_triangle_l15_15670

-- Define the ellipse parameters
variables (a c: ℝ) (h_a : a > sqrt 3)
variables (e: ℝ) 

-- Conditions 
def ellipse (x y : ℝ) : Prop := x^2 / (a^2) + y^2 / 3 = 1
def condition_on_e (OF OA AF : ℝ) : Prop := |OF| = c ∧ |OA| = a ∧ |AF| = a - c ∧ (1 / |OF| + 1 / |OA| = 3 * e / |AF|)
def point_lies_on_line (l: ℝ → ℝ → Prop) (x y : ℝ) : Prop := l x y

-- Goals
theorem ellipse_equation (a > sqrt 3) (h_conditions : condition_on_e a c) :
  ellipse 0 1 = ellipse (2*a) (2*b+a) :=
sorry

theorem max_area_triangle (a > sqrt 3) (a = 2) :
  ∀ (l : ℝ → ℝ → Prop), point_lies_on_line l 0 1 → 
  let area t1 t2 : ℝ := 2 * sqrt 6 / 3 in ∃ (OM : ℝ) (MN : ℝ), area OM MN = (2*sqrt(6)/3) :=
sorry

end ellipse_equation_max_area_triangle_l15_15670


namespace tangent_line_at_1_range_of_a_iff_monotonic_increasing_l15_15259

noncomputable def f (x a : ℝ) : ℝ := (1 / x + a) * real.log (1 + x)

theorem tangent_line_at_1 (a : ℝ) (h : a = -1) :
  let f_at := (1 / (1:ℝ) + a) * real.log (1 + 1)
  let f_prime := -real.log 2
  let tangent_line := λ x : ℝ, f_prime * (x - 1) + f_at in
  tangent_line = λ x, -real.log 2 * x + real.log 2 :=
by {
  -- Proof to be provided
  sorry
}

theorem range_of_a_iff_monotonic_increasing :
  ∀ x : ℝ, x > 0 → monotonic_increasing (λ a : ℝ, (1/x + a) * real.log (1 + x)) ↔ a ∈ set.Ici (1/2) :=
by {
  -- Proof to be provided
  sorry
}

end tangent_line_at_1_range_of_a_iff_monotonic_increasing_l15_15259


namespace overlapping_area_of_circles_within_square_is_correct_l15_15789

noncomputable def combined_overlapping_area (side : ℝ) : ℝ :=
  let π := Real.pi
  let area_of_square := side * side
  let area_of_one_sector := (1/4) * π * (side ^ 2)
  let total_area_of_sectors := 3 * area_of_one_sector
  let combined_area_of_overlaps := total_area_of_sectors - area_of_square
  Real.to_nearest_decimal (combined_area_of_overlaps, 1)

theorem overlapping_area_of_circles_within_square_is_correct (side : ℝ) :
  combined_overlapping_area side = 21.7 :=
by
  sorry

end overlapping_area_of_circles_within_square_is_correct_l15_15789


namespace sequence_formula_l15_15240

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 0 then 1 / 2
  else (3 * sequence (n - 1)) / (sequence (n - 1) + 3)

theorem sequence_formula : ∀ n : ℕ, sequence n = 3 / (n + 5) :=
begin
  intros n,
  induction n with k ih,
  { simp [sequence], norm_num, },
  { simp [sequence, ih], field_simp, rw [←add_assoc, ←add_assoc], congr, }
end

end sequence_formula_l15_15240


namespace least_positive_number_divisible_by_five_smallest_primes_l15_15024

theorem least_positive_number_divisible_by_five_smallest_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  Nat.min (p1 * p2 * p3 * p4 * p5) := 
  2310 :=
by
  sorry

end least_positive_number_divisible_by_five_smallest_primes_l15_15024


namespace total_donuts_l15_15175

def weekly_donut_consumption (week: ℕ) : ℝ :=
  let monday := (20: ℝ) * (1.1 ^ (week - 1)) in
  let tuesday := (1 / 3) * monday in
  let wednesday := 1.05 * (monday + tuesday) in
  monday + tuesday + wednesday

def total_donut_consumption (weeks : ℕ) : ℝ :=
  (finset.range weeks).sum weekly_donut_consumption

theorem total_donuts : total_donut_consumption 4 = 253 :=
by
  /- Proof steps go here -/
  sorry

end total_donuts_l15_15175


namespace log_expression_value_l15_15072

theorem log_expression_value:
  log 10 16 + 3 * log 10 4 + log 10 125 - 2 * log 10 5 + 4 * log 10 2 = 4.913 :=
by
  sorry

end log_expression_value_l15_15072


namespace simplify_f_cos_2alpha_plus_pi_over_4_l15_15986

open Real

-- Definitions of f(alpha) based on the given conditions
def f (alpha : ℝ) : ℝ :=
  (tan (π - alpha) * cos (2 * π - alpha) * sin (π / 2 + alpha)) / cos (-alpha - π)

-- Theorems to prove
theorem simplify_f (alpha : ℝ) : f α = sin α := 
  sorry

theorem cos_2alpha_plus_pi_over_4 (α : ℝ) (h1 : sin α = 4 / 5) (h2 : π / 2 < α ∧ α < π) : 
  cos (2 * α + π / 4) = 17 * sqrt 2 / 50 :=
  sorry

end simplify_f_cos_2alpha_plus_pi_over_4_l15_15986


namespace prob_exactly_M_laws_included_expected_laws_included_l15_15728

variables (K N M : ℕ) (p : ℝ)

-- Definition of the probabilities as given in the conditions and answers
def prob_no_minister_knows_law : ℝ := (1 - p) ^ N
def prob_law_included : ℝ := 1 - prob_no_minister_knows_law p N

-- Part (a)
theorem prob_exactly_M_laws_included :
  (nat.choose K M) * (prob_law_included p N) ^ M * (prob_no_minister_knows_law p N) ^ (K - M) = 
  (nat.choose K M) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) :=
by
  sorry

-- Part (b)
theorem expected_laws_included :
  K * (prob_law_included p N) = K * (1 - (1 - p) ^ N) :=
by
  sorry

end prob_exactly_M_laws_included_expected_laws_included_l15_15728


namespace carla_won_20_games_l15_15636

-- Variables to represent the number of games Carla and Frankie won
variables (C F : ℕ)

-- Condition given in the problem
-- 1. Frankie and Carla played 30 games
def total_games : Prop := F + C = 30

-- 2. Frankie won half as many games as Carla
def win_ratio : Prop := F = C / 2

-- Lean statement to show that Carla won 20 games
theorem carla_won_20_games (h1 : total_games) (h2 : win_ratio) : C = 20 :=
by
  sorry

end carla_won_20_games_l15_15636


namespace smallest_sum_of_elements_l15_15379

theorem smallest_sum_of_elements (A : Finset ℕ) (h1 : A.card = 7)
  (h2 : ∀ (S : Finset ℕ), S ≠ ∅ → S ⊆ A → ((∑ x in S, x) / S.card : ℚ).denom = 1) : 
  (∑ x in A, x) = 1267 := sorry

end smallest_sum_of_elements_l15_15379


namespace least_positive_number_divisible_by_five_primes_l15_15055

theorem least_positive_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ 
    (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) :=
by
  use 2310
  split
  { exact by norm_num }
  split
  { intros p hp, fin_cases hp; norm_num }
  { intros m hm, apply nat.le_of_dvd; norm_num }
  sorry

end least_positive_number_divisible_by_five_primes_l15_15055


namespace building_houses_200_people_l15_15114

-- Define number of floors, apartments per floor, and people per apartment as constants
def numFloors := 25
def apartmentsPerFloor := 4
def peoplePerApartment := 2

-- Define the total number of apartments
def totalApartments := numFloors * apartmentsPerFloor

-- Define the total number of people
def totalPeople := totalApartments * peoplePerApartment

theorem building_houses_200_people : totalPeople = 200 :=
by
  sorry

end building_houses_200_people_l15_15114


namespace least_positive_number_divisible_by_five_primes_l15_15054

theorem least_positive_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ 
    (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) :=
by
  use 2310
  split
  { exact by norm_num }
  split
  { intros p hp, fin_cases hp; norm_num }
  { intros m hm, apply nat.le_of_dvd; norm_num }
  sorry

end least_positive_number_divisible_by_five_primes_l15_15054


namespace sum_first_12_terms_arithmetic_seq_l15_15347

theorem sum_first_12_terms_arithmetic_seq (a d : ℤ) (a₆ a₇ : ℤ) 
  (h₁ : a₆ + a₇ = -2) 
  (h₂ : a = a₆ - 5 * d)
  (h₃ : a₇ = a₆ + d) 
  (h_eq : ∀ x : ℤ, x * x + 2 * x - 3 = 0 → x = a₆ ∨ x = a₇) : 
  (∑ i in finset.range 12, a + i * d) = -12 := by
  sorry

end sum_first_12_terms_arithmetic_seq_l15_15347


namespace tangent_line_eq_mon_incr_l15_15278

def f (a : ℝ) (x : ℝ) := (1 / x + a) * Real.log (1 + x)

theorem tangent_line_eq (a : ℝ) (h_a : a = -1) :
  ∃ m b : ℝ, (∀ x y, y = (1 / x - 1) * Real.log (1 + x) → y = m * x + b) ∧ m * 1 + b = 0 ∧ m = -Real.log 2 ∧ b = Real.log 2 :=
sorry

theorem mon_incr (a : ℝ) (h_mon : ∀ x : ℝ, 0 < x → (deriv (f a)) x ≥ 0) :
  1 / 2 ≤ a :=
sorry

end tangent_line_eq_mon_incr_l15_15278


namespace intersection_of_M_and_N_l15_15294

def set_M : Set ℝ := {x | -1 < x}
def set_N : Set ℝ := {x | x * (x + 2) ≤ 0}

theorem intersection_of_M_and_N : (set_M ∩ set_N) = {x | -1 < x ∧ x ≤ 0} :=
by
  sorry

end intersection_of_M_and_N_l15_15294


namespace recurring_decimal_as_fraction_l15_15086

theorem recurring_decimal_as_fraction : 
  (let a := 0.\overline{56} in 
    100 * a = 56.\overline{56}) → 
  0.\overline{56} = 56 / 99 := 
sorry

end recurring_decimal_as_fraction_l15_15086


namespace num_intersections_is_three_l15_15167

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

noncomputable def f₁ (x : ℝ) : ℝ := log_base 4 x
noncomputable def f₂ (x : ℝ) : ℝ := 1 / log_base 4 x
noncomputable def f₃ (x : ℝ) : ℝ := -log_base 4 x
noncomputable def f₄ (x : ℝ) : ℝ := -1 / log_base 4 x

def numIntersectionPoints : ℕ :=
  { (1, 0), (4, 1), (1/4, -1) }.size

theorem num_intersections_is_three :
  numIntersectionPoints = 3 :=
by
  sorry

end num_intersections_is_three_l15_15167


namespace susan_playground_area_l15_15802

theorem susan_playground_area :
  ∃ (l w : ℕ), 
  let posts := 30 in
  let corners := 4 in
  let post_distance := 6 in
  let longer_side_posts := 3 * (w + 1) in
  let total_posts := 2 * (l + w) - corners in
  total_posts = posts ∧
  l = 6 * (w + 1 - 1) ∧
  l * w = 1188 := sorry

end susan_playground_area_l15_15802


namespace sum_of_squares_pentagon_icosagon_l15_15133

theorem sum_of_squares_pentagon_icosagon {R : ℝ} (hR : R > 0) : 
  let a_5 := 2 * R * Real.sin (Real.pi / 5)
  let a_20 := 2 * R * Real.sin (Real.pi / 20)
  in 5 * a_5^2 > 20 * a_20^2 :=
by
  let a_5 := 2 * R * Real.sin (Real.pi / 5)
  let a_20 := 2 * R * Real.sin (Real.pi / 20)
  sorry

end sum_of_squares_pentagon_icosagon_l15_15133


namespace percentage_of_airlines_with_internet_l15_15122

theorem percentage_of_airlines_with_internet (S W_intersect : ℝ) (hS : S = 0.7) (hW_intersect : W_intersect = 0.2) : 
  ∃ W : ℝ, W = 0.2 :=
by
  use W_intersect
  exact hW_intersect

end percentage_of_airlines_with_internet_l15_15122


namespace twelve_plus_four_times_five_minus_five_cubed_equals_twelve_l15_15608

theorem twelve_plus_four_times_five_minus_five_cubed_equals_twelve :
  12 + 4 * (5 - 10 / 2) ^ 3 = 12 := by
  sorry

end twelve_plus_four_times_five_minus_five_cubed_equals_twelve_l15_15608


namespace cash_price_of_television_l15_15392

variable (DownPayment : ℕ := 120)
variable (MonthlyPayment : ℕ := 30)
variable (NumberOfMonths : ℕ := 12)
variable (Savings : ℕ := 80)

-- Define the total installment cost
def TotalInstallment := DownPayment + MonthlyPayment * NumberOfMonths

-- The main statement to prove
theorem cash_price_of_television : (TotalInstallment - Savings) = 400 := by
  sorry

end cash_price_of_television_l15_15392


namespace num_ints_satisfying_inequality_l15_15310

theorem num_ints_satisfying_inequality : ∃ n : ℕ, ∀ a : ℤ, (-4 ≤ a ∧ a ≤ 4) ∧ (-100 < a^3 ∧ a^3 < 100) → n = 9 :=
begin
  sorry
end

end num_ints_satisfying_inequality_l15_15310


namespace line_passes_fixed_point_max_area_triangle_l15_15253

noncomputable theory

/-- Given the ellipse \( C: \frac{x^2}{3} + y^2 = 1 \) and a line \( l \) passing through the upper vertex \( A(0, 1) \)
that intersects the ellipse \( C \) at points \( P \) and \( Q \), where \( \overrightarrow{\mathrm{AP}} \cdot \overrightarrow{\mathrm{AQ}} = 0 \),
prove that the line \( l \) always passes through the fixed point \((0, -\frac{1}{2})\). -/
theorem line_passes_fixed_point (x y k : ℝ) :
  (x ^ 2 / 3 + y ^ 2 = 1) ∧ (x = 0 ∧ y = 1) ∧ ((x - 0, y - 1) • (x, y - 1) = 0) →
  ∃ p q : ℝ, ∀ x : ℝ, l = p * x + q ∧ q = -1 / 2 :=
sorry

/-- Given the ellipse \( C: \frac{x^2}{3} + y^2 = 1 \) and a line \( l \) passing through the upper vertex \( A(0, 1) \)
that intersects the ellipse \( C \) at points \( P \) and \( Q \), where \( \overrightarrow{\mathrm{AP}} \cdot \overrightarrow{\mathrm{AQ}} = 0 \),
prove that the maximum area of triangle \( \triangle APQ \) is \( \frac{9}{4} \). -/
theorem max_area_triangle (x y k : ℝ) :
  (x ^ 2 / 3 + y ^ 2 = 1) ∧ (x = 0 ∧ y = 1) ∧ ((x - 0, y - 1) • (x, y - 1) = 0) →
  ∃ A P Q : ℝ, (triangle_area A P Q = 9 / 4) :=
sorry

end line_passes_fixed_point_max_area_triangle_l15_15253


namespace snack_eaters_left_l15_15556

theorem snack_eaters_left (initial_participants : ℕ)
    (snack_initial : ℕ)
    (new_outsiders1 : ℕ)
    (half_left1 : ℕ)
    (new_outsiders2 : ℕ)
    (left2 : ℕ)
    (half_left2 : ℕ)
    (h1 : initial_participants = 200)
    (h2 : snack_initial = 100)
    (h3 : new_outsiders1 = 20)
    (h4 : half_left1 = (snack_initial + new_outsiders1) / 2)
    (h5 : new_outsiders2 = 10)
    (h6 : left2 = 30)
    (h7 : half_left2 = (half_left1 + new_outsiders2 - left2) / 2) :
    half_left2 = 20 := 
  sorry

end snack_eaters_left_l15_15556


namespace is_minimum_value_l15_15645

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) - 2

theorem is_minimum_value (h : ∀ x > 0, f x ≥ 0) : ∃ (a : ℝ) (h : a > 0), f a = 0 :=
by {
  sorry
}

end is_minimum_value_l15_15645


namespace mosel_fills_315_boxes_per_week_l15_15929

-- Definitions for the conditions given in the problem.
def hens : ℕ := 270
def eggs_per_hen_per_day : ℕ := 1
def boxes_capacity : ℕ := 6
def days_per_week : ℕ := 7

-- Objective: Prove that the number of boxes filled each week is 315
theorem mosel_fills_315_boxes_per_week :
  let eggs_per_day := hens * eggs_per_hen_per_day
  let boxes_per_day := eggs_per_day / boxes_capacity
  let boxes_per_week := boxes_per_day * days_per_week
  boxes_per_week = 315 := by
  sorry

end mosel_fills_315_boxes_per_week_l15_15929


namespace increasing_intervals_l15_15683

theorem increasing_intervals (k : ℤ) : 
  ∃ (a b : ℝ), [a, b] = [2 * k * π / 3 - 5 * π / 18, 2 * k * π / 3 + π / 18] → 
  ∀ x : ℝ, a ≤ x ∧ x ≤ b → ∃ ω : ℝ, y = sin (ω * x + π / 3) ∧ ω = 3 ∧ 
  y is monotonically increasing :=
by
  sorry

end increasing_intervals_l15_15683


namespace building_houses_200_people_l15_15113

-- Define number of floors, apartments per floor, and people per apartment as constants
def numFloors := 25
def apartmentsPerFloor := 4
def peoplePerApartment := 2

-- Define the total number of apartments
def totalApartments := numFloors * apartmentsPerFloor

-- Define the total number of people
def totalPeople := totalApartments * peoplePerApartment

theorem building_houses_200_people : totalPeople = 200 :=
by
  sorry

end building_houses_200_people_l15_15113


namespace min_n_for_monochromatic_sum_l15_15200

def monochromatic_sum_exists (n : ℕ) : Prop :=
  ∀ (color : ℕ → bool),
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n ∧ 1 ≤ c ∧ c ≤ n ∧ 1 ≤ d ∧ d ≤ n ∧
  color a = color b ∧ color b = color c ∧ color c = color d ∧ a + b + c = d

theorem min_n_for_monochromatic_sum : monochromatic_sum_exists 11 :=
  sorry

end min_n_for_monochromatic_sum_l15_15200


namespace hyperbolas_same_asymptotes_l15_15832

theorem hyperbolas_same_asymptotes (M : ℚ) :
  (∀ x y : ℚ, (x^2 / 9 - y^2 / 16 = 1) → ∃ k : ℚ, y = k * x ∧ k = 4 / 3) →
  (∀ x y : ℚ, (y^2 / 25 - x^2 / M = 1) → ∃ k : ℚ, y = k * x ∧ k = 5 / (sqrt M)) →
  M = 225 / 16 :=
by
  sorry

end hyperbolas_same_asymptotes_l15_15832


namespace min_value_expression_l15_15761

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  8 * a^4 + 12 * b^4 + 50 * c^4 + (1 / (9 * a * b * c)) ≥ 2 * real.sqrt (20 / 3) :=
sorry

end min_value_expression_l15_15761


namespace min_f_ab_max_f_ab_min_max_f_ab_l15_15976

def is_skipping_set (S : Set ℕ) (a b : ℕ) : Prop :=
  ∀ s1 s2 ∈ S, s1 ≠ s2 → ∀ d, abs (s1 - s2) = d → d ≠ a ∧ d ≠ b

def f (a b : ℕ) : ℕ :=
  Sup { n | ∃ S ⊆ Finset.range 2003, S.card = n ∧ is_skipping_set S a b }

theorem min_f_ab (a b : ℕ) (h : 0 < a ∧ a < b ∧ b < 1000) : 668 ≤ f a b :=
sorry

theorem max_f_ab (a b : ℕ) (h : 0 < a ∧ a < b ∧ b < 1000) : f a b ≤ 1334 :=
sorry

theorem min_max_f_ab (a b : ℕ) (h : 0 < a ∧ a < b ∧ b < 1000) : 668 ≤ f a b ∧ f a b ≤ 1334 :=
  ⟨min_f_ab a b h, max_f_ab a b h⟩

end min_f_ab_max_f_ab_min_max_f_ab_l15_15976


namespace actual_distance_km_l15_15397

-- Define the conditions
def map_scale : ℕ := 1000000
def map_distance_cm : ℝ := 2.6

-- Define a theorem to prove the actual distance
theorem actual_distance_km (scale : ℕ) (distance_cm : ℝ) (h_scale : scale = 1000000) (h_distance : distance_cm = 2.6) : 
    let actual_distance_cm := distance_cm * scale in
    let actual_distance_m := actual_distance_cm / 100 in
    let actual_distance_km := actual_distance_m / 1000 in
    actual_distance_km = 26 :=
by
  sorry

end actual_distance_km_l15_15397


namespace ellipse_equation_no_real_m_l15_15226

noncomputable def ellipse_vertex : ℝ × ℝ := (0, -1)
noncomputable def foci_on_x_axis : Prop := ∀ E : ℝ × ℝ, E ∈ set_of_foci → E.2 = 0
noncomputable def distance_to_line (F : ℝ × ℝ) : Prop := 
  real.abs ((F.1 - F.2 + 2 * real.sqrt 2) / real.sqrt 2) = 3

theorem ellipse_equation 
  (vertex : ellipse_vertex) 
  (foci_condition : foci_on_x_axis) 
  (distance_condition : ∃ F : ℝ × ℝ, distance_to_line F) : 
  ∃ a b, a = 3 ∧ b = 1 ∧ ∀ x y, (x^2 / a + y^2 / b = 1) :=
sorry

theorem no_real_m
  (vertex : ellipse_vertex) 
  (foci_condition : foci_on_x_axis) 
  (distance_condition : ∃ F : ℝ × ℝ, distance_to_line F) :
  ∀ m : ℝ, ¬ (∃ P : ℝ × ℝ, |AM| = |AN|).
sorry

end ellipse_equation_no_real_m_l15_15226


namespace cross_ratio_equality_l15_15368

noncomputable theory

open_locale classical

variables {A B C D E P F : Type*}

/-- Assume A, B, C, and D are points on the same circle Γ,
chords (AB) and (CD) intersect at point E,
point P is on the line (BE),
the tangent at E to the circumcircle of ΔDEP intersects the line (AC) at point F.
Prove that (FE, FC) = (DB, DP). -/
theorem cross_ratio_equality {Γ : Type*} [circle Γ A B C D] 
  (h1 : intersection (chord A B) (chord C D) = E)
  (h2 : P ∈ (line B E))
  (h3 : tangent_at E (circumcircle (triangle D E P)) = F ∈ (line A C)) :
  cross_ratio (F E, F C) = cross_ratio (D B, D P) := sorry

end cross_ratio_equality_l15_15368


namespace smallest_period_f_range_f_increasing_intervals_f_l15_15678

-- Define the function f(x)
def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos (2 * x)

-- Statement 1: Smallest Positive Period
theorem smallest_period_f : ∀ x, f (x + π) = f x :=
sorry

-- Statement 2: Range of f(x)
theorem range_f : ∀ y, (∃ x, f x = y) ↔ -2 ≤ y ∧ y ≤ 2 :=
sorry

-- Statement 3: Intervals where f(x) is monotonically increasing
theorem increasing_intervals_f : ∀ k : ℤ, ∀ x, (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6) → ∀ y, (x ≤ y ∧ y ≤ k * π + π / 6) → f x ≤ f y :=
sorry

end smallest_period_f_range_f_increasing_intervals_f_l15_15678


namespace problem_solution_l15_15875

theorem problem_solution :
  ∃ a b c d : ℚ, 
  4 * a + 2 * b + 5 * c + 8 * d = 67 ∧ 
  4 * (d + c) = b ∧ 
  2 * b + 3 * c = a ∧ 
  c + 1 = d ∧ 
  a * b * c * d = (1201 * 572 * 19 * 124) / (105 ^ 4) :=
sorry

end problem_solution_l15_15875


namespace no_good_pair_3_3_l15_15908

def is_good_pair (r s : ℕ) : Prop :=
  ∃ (P : ℤ[x]), 
    ∃ (a : Fin r → ℤ) (b : Fin s → ℤ), 
      Function.Injective a ∧ Function.Injective b ∧
      (∀ i, P.eval (a i) = 2) ∧ 
      (∀ j, P.eval (b j) = 5)

theorem no_good_pair_3_3 : ¬ is_good_pair 3 3 :=
sorry

end no_good_pair_3_3_l15_15908


namespace weight_of_new_person_l15_15881

-- Definitions based on conditions
def average_weight_increase : ℝ := 2.5
def number_of_persons : ℕ := 8
def old_weight : ℝ := 65
def total_weight_increase : ℝ := number_of_persons * average_weight_increase

-- Proposition to prove
theorem weight_of_new_person : (old_weight + total_weight_increase) = 85 := by
  -- add the actual proof here
  sorry

end weight_of_new_person_l15_15881


namespace number_of_isometry_sequences_l15_15758

def isometry (f: Point → Point) : Prop := ∀ p q: Point, dist (f p) (f q) = dist p q

def rotation90 (p: Point) : Point := (-p.y, p.x)
def rotation180 (p: Point) : Point := (-p.x, -p.y)
def rotation270 (p: Point) : Point := (p.y, -p.x)
def reflection_x (p: Point) : Point := (p.x, -p.y)
def reflection_y (p: Point) : Point := (-p.x, p.y)

def square := {(0,0), (4,0), (4,4), (0,4)}

def is_same_square (transformation_seq: list (Point → Point)) : Prop :=
  transformation_seq.foldr (∘) id square = square

theorem number_of_isometry_sequences :
  { transformation_seq : list (Point → Point) // transformation_seq.length = 3 ∧ is_same_square transformation_seq }.card = 18 :=
sorry

end number_of_isometry_sequences_l15_15758


namespace count_ways_l15_15593

-- Definitions for the sets of songs liked by specific combinations of the girls
def sets_partition (Songs : Finset ℕ) : Prop :=
  ∃ (AB BC CA A B C N : Finset ℕ),
    AB.card = 1 ∧
    BC.card = 1 ∧
    CA.card = 1 ∧
    A ∪ B ∪ C ∪ AB ∪ BC ∪ CA ∪ N = Songs ∧
    AB ∩ BC = ∅ ∧ AB ∩ CA = ∅ ∧ BC ∩ CA = ∅ ∧
    A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
    Songs.card = 5

theorem count_ways : ∃ (count : ℕ), sets_partition finset.univ ∧ count = 1140 :=
by
  sorry

end count_ways_l15_15593


namespace peter_total_distance_is_six_l15_15519

def total_distance_covered (d : ℝ) :=
  let first_part_time := (2/3) * d / 4
  let second_part_time := (1/3) * d / 5
  (first_part_time + second_part_time) = 1.4

theorem peter_total_distance_is_six :
  ∃ d : ℝ, total_distance_covered d ∧ d = 6 := 
by
  -- Proof can be filled here
  sorry

end peter_total_distance_is_six_l15_15519


namespace percentage_change_in_area_l15_15094

theorem percentage_change_in_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let Area_original := L * B,
      Length_new := L / 2,
      Breadth_new := 3 * B,
      Area_new := Length_new * Breadth_new,
      Percentage_change := ((Area_new - Area_original) / Area_original) * 100
  in Percentage_change = 50 := by
  sorry

end percentage_change_in_area_l15_15094


namespace least_number_divisible_by_five_primes_l15_15067

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, (n > 0) ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → p ∣ n) ∧ n = 2310 :=
begin
  sorry
end

end least_number_divisible_by_five_primes_l15_15067


namespace wizard_elixir_combinations_l15_15578

def roots : ℕ := 4
def minerals : ℕ := 5
def incompatible_pairs : ℕ := 3
def total_combinations : ℕ := roots * minerals
def valid_combinations : ℕ := total_combinations - incompatible_pairs

theorem wizard_elixir_combinations : valid_combinations = 17 := by
  sorry

end wizard_elixir_combinations_l15_15578


namespace inverse_domain_l15_15829

theorem inverse_domain (f : ℝ → ℝ) (h : ∀ x, f x = 3^x) (dom : {x | 0 < x ∧ x ≤ 2}) :
  (∀ y, y ∈ (1, 9] ↔ ∃ x ∈ dom, f x = y) :=
sorry

end inverse_domain_l15_15829


namespace students_taking_science_or_history_not_both_l15_15621

/-- 
There are 15 students taking both science and history classes.
There are 30 students taking science.
There are 18 students taking only history.
We aim to prove that the number of students taking science or history but not both is 33.
-/
theorem students_taking_science_or_history_not_both :
  ∀ (a b c : ℕ), a = 15 → b = 30 → c = 18 → b - a + c = 33 :=
by
  intros a b c h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  sorry

end students_taking_science_or_history_not_both_l15_15621


namespace max_trig_expression_l15_15953

theorem max_trig_expression (θ₁ θ₂ θ₃ θ₄ : ℝ) :
  (∃ θ₁ θ₂ θ₃ θ₄ : ℝ, (cos θ₁ * sin θ₂ - sin θ₂ * cos θ₃ + cos θ₃ * sin θ₄ - sin θ₄ * cos θ₁) = 2) :=
sorry

end max_trig_expression_l15_15953


namespace range_of_a_l15_15989

theorem range_of_a (a x : ℝ) : (-4 < x - a ∧ x - a < 4) → ((x - 2) * (3 - x) > 0) → (-1 ≤ a ∧ a ≤ 6) :=
by {
  intros h₁ h₂,
  sorry
}

end range_of_a_l15_15989


namespace snack_eaters_remaining_l15_15558

theorem snack_eaters_remaining 
  (initial_population : ℕ)
  (initial_snackers : ℕ)
  (new_outsiders_1 : ℕ)
  (first_half_leave : ℕ)
  (new_outsiders_2 : ℕ)
  (second_leave : ℕ)
  (final_half_leave : ℕ) 
  (h_initial_population : initial_population = 200)
  (h_initial_snackers : initial_snackers = 100)
  (h_new_outsiders_1 : new_outsiders_1 = 20)
  (h_first_half_leave : first_half_leave = (initial_snackers + new_outsiders_1) / 2)
  (h_new_outsiders_2 : new_outsiders_2 = 10)
  (h_second_leave : second_leave = 30)
  (h_final_half_leave : final_half_leave = (first_half_leave + new_outsiders_2 - second_leave) / 2) : 
  final_half_leave = 20 := 
sorry

end snack_eaters_remaining_l15_15558


namespace min_cards_to_draw_l15_15752

theorem min_cards_to_draw (deck_size : ℕ) (range_cards : fin deck_size)
                           (lucky_set : set (finset ℕ)) :
  deck_size = 52 →
  (∀ (s1 s2 : fin 52), s1 ≠ s2 → ∃ (d1 d2 d3 : fin 52), 
    d1 ∈ range_cards ∧ d2 ∈ range_cards ∧ d3 ∈ range_cards ∧ 
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧ 
    lucky_set = {d1 + d2, d2 + d3, d1 + d3}) →
  26 ≤ deck_size :=
by
  sorry

end min_cards_to_draw_l15_15752


namespace twentieth_term_is_natural_l15_15216

-- Define the conditions as given in the problem
variable {a1 q : ℕ} -- First term of the geometric progression and the common ratio
variable (a10 a30 : ℕ) -- Tenth and thirtieth term of the geometric progression are natural numbers

-- The geometric progression terms can be written as follows:
-- a10 = a1 * q^9
-- a30 = a1 * q^29

theorem twentieth_term_is_natural
    (h1 : a1 ∈ Nat)
    (h10 : a10 = a1 * q^9)
    (h30 : a30 = a1 * q^29) :
    (a1 * q^19) ∈ Nat := 
by
  sorry

end twentieth_term_is_natural_l15_15216


namespace min_value_arithmetic_sequence_l15_15997

theorem min_value_arithmetic_sequence (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_arith_seq : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 1 + a 2014 = 2) :
  (∃ a2 a2013, a 2 = a2 ∧ a 2013 = a2013 ∧ a2 > 0 ∧ a2013 > 0 ∧ ∀ a2 a2013, a 2 = a2 ∧ a 2013 = a2013 → (1/a2 + 1/a2013) ≥ 2) :=
by
  sorry

end min_value_arithmetic_sequence_l15_15997


namespace problem1_problem2_l15_15236

open Set

noncomputable def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3 * a) < 0}

theorem problem1 (a : ℝ) (h1 : A ⊆ (A ∩ B a)) : (4 / 3 : ℝ) ≤ a ∧ a ≤ 2 :=
sorry

theorem problem2 (a : ℝ) (h2 : A ∩ B a = ∅) : a ≤ (2 / 3 : ℝ) ∨ a ≥ 4 :=
sorry

end problem1_problem2_l15_15236


namespace smallest_share_of_land_l15_15467

/-- Define the problem in Lean, where we need to prove the smallest share an individual can inherit. --/
theorem smallest_share_of_land (a : ℕ → ℕ) 
  (h_sum : ∑ i in range (n : ℕ), a i ≤ 199) 
  (h_nonzero : ∀ i, a i > 0) : 
  (∃ (x : ℝ), x = 1 / (4 * 3^65)) :=
begin
  sorry
end

end smallest_share_of_land_l15_15467


namespace inequality_proof_l15_15535

theorem inequality_proof
  {a b c d e f : ℝ}
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (h_abs : |sqrt (a * d) - sqrt (b * c)| ≤ 1) :
  (a * e + b / e) * (c * e + d / e) ≥ 
    (a^2 * f^2 - (b^2) / (f^2)) * ((d^2) / (f^2) - c^2 * f^2) :=
by
  sorry

end inequality_proof_l15_15535


namespace problem_f_f_neg2_eq_3_l15_15323

def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then x^2 + 1 else Real.log (x + 3) / Real.log 2

theorem problem_f_f_neg2_eq_3 : f (f (-2)) = 3 := 
by 
  unfold f,
  sorry

end problem_f_f_neg2_eq_3_l15_15323


namespace sin_Y_in_right_triangle_l15_15346

theorem sin_Y_in_right_triangle
  (XY XZ : ℝ) (YZ : ℝ) 
  (hXY : XY = 13) (hXZ : XZ = 5)
  (hYZ : YZ = Real.sqrt (XY^2 - XZ^2))
  (angleZ_right : ∠Z = π / 2) :
  Real.sin (angle Y) = 5 / 13 :=
by sorry

end sin_Y_in_right_triangle_l15_15346


namespace collinear_O_N_I_l15_15356

-- Given definitions
variables {A B C P K M N O I : Point}
variable {ω_a : Excircle}

-- Problem conditions
axiom circumcenter_def : ∃ O, O is_circumcenter ΔABC
axiom incenter_def : ∃ I, I is_incenter ΔABC
axiom excircle_touches_KM : ω_a touches (extension AB) at K ∧ ω_a touches (extension AC) at M ∧ ω_a touches BC at N
axiom midpoint_P_circumcircle : is_midpoint P K M ∧ P ∈ circumcircle ΔABC

-- Proof goal
theorem collinear_O_N_I : collinear {O, N, I} :=
sorry

end collinear_O_N_I_l15_15356


namespace base8_subtraction_l15_15963

theorem base8_subtraction : (7463 - 3154 = 4317) := by sorry

end base8_subtraction_l15_15963


namespace center_of_circle_l15_15448

theorem center_of_circle (x y : ℝ) (h : x^2 + y^2 + 4 * x - 6 * y + 9 = 0) : ((-2 : ℝ), 3 : ℝ) = (-2, 3) :=
sorry

end center_of_circle_l15_15448


namespace function_identity_l15_15786

theorem function_identity
    (f : ℝ → ℝ)
    (h1 : ∀ x : ℝ, f x ≤ x)
    (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) :
    ∀ x : ℝ, f x = x :=
by
    sorry

end function_identity_l15_15786


namespace building_houses_200_people_l15_15111

theorem building_houses_200_people 
    (num_floors : ℕ)
    (apartments_per_floor : ℕ)
    (people_per_apartment : ℕ) :
    num_floors = 25 →
    apartments_per_floor = 4 →
    people_per_apartment = 2 →
    num_floors * apartments_per_floor * people_per_apartment = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end building_houses_200_people_l15_15111


namespace integerGridsToPlanePartitions_l15_15967

-- Definitions for plane partitions as matrices with non-increasing rows and columns
def isPlanePartition (m : Matrix ℤ ℤ ℤ) : Prop :=
  ∀ i j m, i < m → j < n → m i j ≥ m i (j + 1) ∧ m i j ≥ m (i + 1) j

-- Define a size function for the plane partition
def partitionSize (m : Matrix ℤ ℤ ℤ) : ℤ :=
  ∑ i j, m i j

-- Define associating function between integer grids and plane partitions
def gridToPartition (g : Matrix ℤ ℤ ℤ) : Matrix ℤ ℤ ℤ :=
  -- The transformation logic would be defined here
  sorry

-- Generating function for plane partitions
noncomputable def generatingSeries (grid : Matrix ℤ ℤ ℤ) : ℕ :=
  -- The function calculating the generating series goes here
  sorry

-- Main theorem: properties of the bijection between grids and plane partitions
theorem integerGridsToPlanePartitions (g : Matrix ℤ ℤ ℤ) :
  isPlanePartition (gridToPartition g) ∧
  (partitionSize (gridToPartition g) = ∑ i j, g i j ∧
  generatingSeries g = -- the generating series formula goes here
  sorry

end integerGridsToPlanePartitions_l15_15967


namespace Misha_erased_33_numbers_l15_15408

/--
Sasha wrote down numbers from one to one hundred, and Misha erased some of them.
Among the remaining numbers, 20 contain the digit one, 19 contain the digit two, and 30 contain neither one nor two.
How many numbers did Misha erase?
-/
theorem Misha_erased_33_numbers 
  (S : Finset ℕ) (hS : S = Finset.range 101) 
  (R : Finset ℕ) (hR1 : (R.filter (λ n, '1' ∈ n.digits 10)).card = 20)
  (hR2 : (R.filter (λ n, '2' ∈ n.digits 10)).card = 19)
  (hR3 : (R.filter (λ n, ('1' ∈ n.digits 10) ∨ ('2' ∈ n.digits 10))).card = 67) :
  S.card - R.card = 33 :=
sorry

end Misha_erased_33_numbers_l15_15408


namespace intersection_M_N_complement_N_U_l15_15770

-- Definitions for the sets and the universal set
def U := Set ℝ
def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }
def N : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) } -- Simplified domain interpretation for N

-- Intersection and complement calculations
theorem intersection_M_N (x : ℝ) : x ∈ M ∧ x ∈ N ↔ x ∈ { x | -2 ≤ x ∧ x ≤ 1 } := by sorry

theorem complement_N_U (x : ℝ) : x ∉ N ↔ x ∈ { x | x > 1 } := by sorry

end intersection_M_N_complement_N_U_l15_15770


namespace set_of_points_sphere_l15_15695

variables {x1 y1 z1 x2 y2 z2 x y z : ℝ}

def x0 := (x1 + x2) / 2
def y0 := (y1 + y2) / 2
def z0 := (z1 + z2) / 2

def R_squared := ((x1 - x2) ^ 2 + (y1 - y2) ^ 2 + (z1 - z2) ^ 2) / 4

theorem set_of_points_sphere :
  (x - x0) ^ 2 + (y - y0) ^ 2 + (z - z0) ^ 2 = R_squared :=
sorry

end set_of_points_sphere_l15_15695


namespace pq_parallel_ab_l15_15813

variables {A B C D O P Q E : Point}
variables {circle : Circle}
variables [Quadrilateral ABCD : circumscribed circle]

-- Define the problem in Lean 4

theorem pq_parallel_ab (h1: O ∈ line AB)     -- center of circumcircle lies on AB
                    (h2: E = sym_point D AB) -- E is symmetric to D wrt AB
                    (h3: line_segment_inter AC DO P) -- AC and DO intersect at P
                    (h4: line_segment_inter BD CE Q) -- BD and CE intersect at Q
                    : parallel line PQ AB    -- PQ is parallel to AB
                    := 
sorry

end pq_parallel_ab_l15_15813


namespace proof_problem_l15_15800

-- Define the conditions
variables {θ₁ θ₂ θ₃ θ₄ : ℝ}
hypothesis hθ₁ : θ₁ ∈ Ioo (-π/2) (π/2)
hypothesis hθ₂ : θ₂ ∈ Ioo (-π/2) (π/2)
hypothesis hθ₃ : θ₃ ∈ Ioo (-π/2) (π/2)
hypothesis hθ₄ : θ₄ ∈ Ioo (-π/2) (π/2)

-- Main statement
theorem proof_problem :
  (∃ x : ℝ, 
    cos θ₁ ^ 2 * cos θ₂ ^ 2 - (sin θ₁ * sin θ₂ - x) ^ 2 ≥ 0 ∧
    cos θ₃ ^ 2 * cos θ₄ ^ 2 - (sin θ₃ * sin θ₄ - x) ^ 2 ≥ 0)
  ↔ 
  (sin θ₁ ^ 2 + sin θ₂ ^ 2 + sin θ₃ ^ 2 + sin θ₄ ^ 2 ≤ 2 * (1 + sin θ₁ * sin θ₂ * sin θ₃ * sin θ₄ + cos θ₁ * cos θ₂ * cos θ₃ * cos θ₄)) := 
sorry

end proof_problem_l15_15800


namespace average_of_three_numbers_l15_15851

theorem average_of_three_numbers
  (a b c : ℕ)
  (h1 : 2 * a + b + c = 130)
  (h2 : a + 2 * b + c = 138)
  (h3 : a + b + 2 * c = 152) :
  (a + b + c) / 3 = 35 :=
by
  sorry

end average_of_three_numbers_l15_15851


namespace trains_meet_time_l15_15522

theorem trains_meet_time :
  (∀ (D : ℝ) (s1 s2 t1 t2 : ℝ),
    D = 155 ∧ 
    s1 = 20 ∧ 
    s2 = 25 ∧ 
    t1 = 7 ∧ 
    t2 = 8 →
    (∃ t : ℝ, 20 * t + 25 * t = D - 20)) →
  8 + 3 = 11 :=
by {
  sorry
}

end trains_meet_time_l15_15522


namespace boat_avg_speed_ratio_l15_15893

theorem boat_avg_speed_ratio (boat_speed : ℕ) (current_speed : ℕ) (distance : ℕ) (break_time : ℕ) :
  boat_speed = 12 → current_speed = 3 → distance = 18 → break_time = 30 →
  (let downstream_speed := boat_speed + current_speed in
   let upstream_speed := boat_speed - current_speed in
   let downstream_time := distance / downstream_speed in
   let upstream_time := distance / upstream_speed + break_time / 60 in
   let total_time := downstream_time + upstream_time in
   let total_distance := 2 * distance in
   let avg_speed := total_distance / total_time in
   avg_speed / boat_speed = 0.81) :=
begin
  intros,
  -- placeholder for automated calculation and verification
  sorry
end

end boat_avg_speed_ratio_l15_15893


namespace correct_option_B_l15_15706

noncomputable def e : ℝ := real.exp 1 -- value of e is defined as exp(1)

def f (x : ℝ) : ℝ := x^2 + e

theorem correct_option_B : 
  ∀ a : ℝ, a > e → ∃ x : ℝ, x > 0 ∧ f(x) < a := 
by 
  intros a ha
  use 0
  simp [f, e, real.exp]
  exact ha

end correct_option_B_l15_15706


namespace solutions_to_deqs_l15_15420

noncomputable def x1 (t : ℝ) : ℝ := -1 / t^2
noncomputable def x2 (t : ℝ) : ℝ := -t * Real.log t

theorem solutions_to_deqs (t : ℝ) (ht : 0 < t) :
  (deriv x1 t = 2 * t * (x1 t)^2) ∧ (deriv x2 t = x2 t / t - 1) :=
by
  sorry

end solutions_to_deqs_l15_15420


namespace joseph_energy_cost_l15_15362

theorem joseph_energy_cost 
  (R W O : ℝ) 
  (h1 : R = 3 * W) 
  (h2 : O = 500) 
  (h3 : O = 2 * W) : 
  W + R + O = 1500 := 
by
  have W_val : W = 250 := 
    by
      rw [←h3] at h2
      have : 500 = 2 * W := h2
      exact (eq_div_of_mul_eq (by norm_num) this.symm)
  rw [W_val] at *
  have R_val : R = 3 * 250 := by rw [W_val, h1]
  rw [R_val, W_val, h2]
  norm_num

end joseph_energy_cost_l15_15362


namespace total_bill_amount_l15_15555

theorem total_bill_amount (n : ℕ) (cost_per_meal : ℕ) (gratuity_rate : ℚ) (total_bill_with_gratuity : ℚ)
  (h1 : n = 7) (h2 : cost_per_meal = 100) (h3 : gratuity_rate = 20 / 100) :
  total_bill_with_gratuity = (n * cost_per_meal : ℕ) * (1 + gratuity_rate) :=
sorry

end total_bill_amount_l15_15555


namespace intersection_of_M_and_N_l15_15385

def M : Set Type := {x | is_rectangular_prism x}
def N : Set Type := {x | is_cube x}

theorem intersection_of_M_and_N :
  M ∩ N = N :=
sorry

end intersection_of_M_and_N_l15_15385


namespace pq_parallel_ab_l15_15814

variables {A B C D O P Q E : Point}
variables {circle : Circle}
variables [Quadrilateral ABCD : circumscribed circle]

-- Define the problem in Lean 4

theorem pq_parallel_ab (h1: O ∈ line AB)     -- center of circumcircle lies on AB
                    (h2: E = sym_point D AB) -- E is symmetric to D wrt AB
                    (h3: line_segment_inter AC DO P) -- AC and DO intersect at P
                    (h4: line_segment_inter BD CE Q) -- BD and CE intersect at Q
                    : parallel line PQ AB    -- PQ is parallel to AB
                    := 
sorry

end pq_parallel_ab_l15_15814


namespace next_simultaneous_activation_l15_15904

theorem next_simultaneous_activation : 
  let fire_station_interval := 12
  let police_station_interval := 18
  let hospital_interval := 30
  let start_time := 8 * 60 -- 8:00 AM in minutes
  let lcm := Nat.lcm (Nat.lcm fire_station_interval police_station_interval) hospital_interval
  (start_time + lcm) = 11 * 60 := -- 11:00 AM in minutes
by
  let fire_station_interval := 12
  let police_station_interval := 18
  let hospital_interval := 30
  let start_time := 8 * 60
  let lcm := Nat.lcm (Nat.lcm fire_station_interval police_station_interval) hospital_interval
  have h1 : Nat.gcd fire_station_interval police_station_interval * lcm = fire_station_interval * police_station_interval := sorry
  have h2 : lcm : Nat := sorry
  have h3 : start_time + lcm = 8 * 60 + 180 := sorry
  have h4 : 8 * 60 + 180 = 11 * 60 := sorry
  exact h4
  -- sorry

end next_simultaneous_activation_l15_15904


namespace minimum_balls_to_draw_l15_15337

-- Defining the sizes for the different colors of balls
def red_balls : Nat := 40
def green_balls : Nat := 25
def yellow_balls : Nat := 20
def blue_balls : Nat := 15
def purple_balls : Nat := 10
def orange_balls : Nat := 5

-- Given conditions
def max_red_balls_before_18 : Nat := 17
def max_green_balls_before_18 : Nat := 17
def max_yellow_balls_before_18 : Nat := 17
def max_blue_balls_before_18 : Nat := 15
def max_purple_balls_before_18 : Nat := 10
def max_orange_balls_before_18 : Nat := 5

-- Sum of maximum balls of each color that can be drawn without ensuring 18 of any color
def max_balls_without_18 : Nat := 
  max_red_balls_before_18 + 
  max_green_balls_before_18 + 
  max_yellow_balls_before_18 + 
  max_blue_balls_before_18 + 
  max_purple_balls_before_18 + 
  max_orange_balls_before_18

theorem minimum_balls_to_draw {n : Nat} (h : n = max_balls_without_18 + 1) :
  n = 82 := by
  sorry

end minimum_balls_to_draw_l15_15337


namespace snack_eaters_remaining_l15_15559

theorem snack_eaters_remaining 
  (initial_population : ℕ)
  (initial_snackers : ℕ)
  (new_outsiders_1 : ℕ)
  (first_half_leave : ℕ)
  (new_outsiders_2 : ℕ)
  (second_leave : ℕ)
  (final_half_leave : ℕ) 
  (h_initial_population : initial_population = 200)
  (h_initial_snackers : initial_snackers = 100)
  (h_new_outsiders_1 : new_outsiders_1 = 20)
  (h_first_half_leave : first_half_leave = (initial_snackers + new_outsiders_1) / 2)
  (h_new_outsiders_2 : new_outsiders_2 = 10)
  (h_second_leave : second_leave = 30)
  (h_final_half_leave : final_half_leave = (first_half_leave + new_outsiders_2 - second_leave) / 2) : 
  final_half_leave = 20 := 
sorry

end snack_eaters_remaining_l15_15559


namespace least_number_divisible_by_five_smallest_primes_l15_15003

theorem least_number_divisible_by_five_smallest_primes : 
  ∃ n ∈ ℕ+, n = 2 * 3 * 5 * 7 * 11 ∧ n = 2310 :=
by
  sorry

end least_number_divisible_by_five_smallest_primes_l15_15003


namespace max_unique_sums_l15_15129

-- Define the coin values in cents
def penny := 1
def nickel := 5
def quarter := 25
def half_dollar := 50

-- Define the set of all coins and their counts
structure Coins :=
  (pennies : ℕ := 3)
  (nickels : ℕ := 3)
  (quarters : ℕ := 1)
  (half_dollars : ℕ := 2)

-- Define the list of all possible pairs and their sums
def possible_sums : Finset ℕ :=
  { 2, 6, 10, 26, 30, 51, 55, 75, 100 }

-- Prove that the count of unique sums is 9
theorem max_unique_sums (c : Coins) : c.pennies = 3 → c.nickels = 3 → c.quarters = 1 → c.half_dollars = 2 →
  possible_sums.card = 9 := 
by
  intros
  sorry

end max_unique_sums_l15_15129


namespace tangent_line_equation_at_1_monotonically_increasing_range_of_a_l15_15269

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (1 / x + a) * real.log (1 + x)

-- Part 1: Tangent line equation at point (1, f(1)) when a = -1
theorem tangent_line_equation_at_1 (x y : ℝ) (h : y = f 1 (-1)) :
  (real.log 2) * x + y - (real.log 2) = 0 := 
sorry

-- Part 2: Range of a for which f(x) is monotonically increasing on (0, +∞)
theorem monotonically_increasing_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, 0 < x → 0 < f' x a) ↔ (1 / 2 ≤ a) :=  
sorry

end tangent_line_equation_at_1_monotonically_increasing_range_of_a_l15_15269


namespace functional_equation_solution_l15_15455

theorem functional_equation_solution (f : ℝ → ℝ) 
  (hf : ∀ x : ℝ, f(x^2) * (deriv^[2] f x) = (deriv f x) * (deriv f (x^2)))
  (hf1 : f 1 = 1)
  (hf3 : (deriv^[3] f 1) = 8) : 
  (deriv f 1) + (deriv^[2] f 1) = 6 := 
by sorry

end functional_equation_solution_l15_15455


namespace new_average_after_deductions_l15_15438

theorem new_average_after_deductions (x : ℝ) (h_avg : (x + (x + 1) + (x + 2) + ⋯ + (x + 9)) / 10 = 25) : 
(x - 9 + (x + 1 - 8) + (x + 2 - 7) + ⋯ + (x + 9)) / 10 = 20.5 :=
by
  -- The proof will be provided here
  sorry

end new_average_after_deductions_l15_15438


namespace number_of_elements_in_A_inter_B_l15_15703

noncomputable def A : Set ℤ := {x | ∃ k : ℕ, x = 4 * k - 3}
noncomputable def B : Set ℤ := {x | -3 ≤ x ∧ x ≤ 9}

theorem number_of_elements_in_A_inter_B : (A ∩ B).toFinset.card = 4 := by
  sorry

end number_of_elements_in_A_inter_B_l15_15703


namespace max_primes_dasha_selected_6_l15_15209

/-
Given the set of natural numbers {1, 2, ... , 25} and the requirement 
that the difference between any two selected numbers is divisible by 4,
prove that the maximum number of selected prime numbers can be 5.
-/

theorem max_primes_dasha_selected_6 (S : Finset ℕ) (hS : ∀ x ∈ S, x ∈ Finset.range 26) (h_divisibility : ∀ x y ∈ S, (x - y) % 4 = 0) :
  S.filter Nat.Prime.card ≤ 5 :=
by
  sorry

end max_primes_dasha_selected_6_l15_15209


namespace least_positive_number_divisible_by_five_smallest_primes_l15_15023

theorem least_positive_number_divisible_by_five_smallest_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  Nat.min (p1 * p2 * p3 * p4 * p5) := 
  2310 :=
by
  sorry

end least_positive_number_divisible_by_five_smallest_primes_l15_15023


namespace math_proof_problem_l15_15649

/-- Mathematical problem statement in Lean 4 -/
noncomputable def problem_statement (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n ≥ 2) 
  (h_a : ∀ i, i ∈ Finset.range n → (1 < a i ∧ a i < Real.sqrt 7)) 
  (a_next : ∀ i, i % n = (i + 1) % n → a (i % n) = a ((i + 1) % n)) : Prop :=
  (∀ i ∈ Finset.range n, (1 / (a i ^ 2 - 1) + 1 / (7 - a i ^ 2)) ≥ 2 / 3) ∧ 
  (let S := (∑ i in Finset.range n, 1 / Real.sqrt ((a i ^ 2 - 1) * (7 - a ((i + 1) % n) ^ 2))) 
  in S ≥ n / 3)

-- Placeholder for a theorem to be proven
theorem math_proof_problem (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n ≥ 2) 
  (h_a : ∀ i, i ∈ Finset.range n → (1 < a i ∧ a i < Real.sqrt 7)) 
  (a_next : ∀ i, i % n = (i + 1) % n → a (i % n) = a ((i + 1) % n))
: problem_statement n a h_n h_a a_next := 
  sorry

end math_proof_problem_l15_15649


namespace find_a_l15_15375

noncomputable def f (x a : ℝ) := 2 * real.log x + a / x^2

theorem find_a (a : ℝ) (h1 : 0 < a) 
  (h2 : ∀ x : ℝ, 0 < x → 2 * real.log x + a / x^2 ≥ 2) :
  a ≥ real.exp 1 :=
sorry

end find_a_l15_15375


namespace chord_bisected_eqn_l15_15651

theorem chord_bisected_eqn : 
  (∃ (A B : (ℝ × ℝ)), 
    (∃ (P : ℝ × ℝ), P = (2, 1) ∧
     A ≠ B ∧ 
     (∃ (C : ℝ × ℝ), 
        C = (1, 0) ∧ 
        (∃ (r : ℝ), r = 5 ∧ 
         (((A.1 - 1)^2 + A.2^2 = 25) ∧ ((B.1 - 1)^2 + B.2^2 = 25)) ∧ 
         (P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))))) ∧
    ((P.2 - 1) = - (P.1 - 2))) → 
  (∃ m b : ℝ, (m * P.1 + P.2 + b = 0) ∧ m = -1 ∧ b = 3) :=
sorry

end chord_bisected_eqn_l15_15651


namespace marcus_dropped_8_pies_l15_15772

-- Step d): Rewrite as a Lean 4 statement
-- Define all conditions from the problem
def total_pies (pies_per_batch : ℕ) (batches : ℕ) : ℕ :=
  pies_per_batch * batches

def pies_dropped (total_pies : ℕ) (remaining_pies : ℕ) : ℕ :=
  total_pies - remaining_pies

-- Prove that Marcus dropped 8 pies
theorem marcus_dropped_8_pies : 
  total_pies 5 7 - 27 = 8 := by
  sorry

end marcus_dropped_8_pies_l15_15772


namespace find_a_minus_b_l15_15663

-- Define the given function
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 3 * a * x + 4

-- Define the condition for the function being even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the function f(x) with given parameters
theorem find_a_minus_b (a b : ℝ) (h_dom_range : ∀ x : ℝ, b - 3 ≤ x → x ≤ 2 * b) (h_even_f : is_even (f a)) :
  a - b = -1 :=
  sorry

end find_a_minus_b_l15_15663


namespace repeating_decimal_as_fraction_l15_15084

theorem repeating_decimal_as_fraction : (∃ x : ℚ, x = 0.565656... ) → (∃ y : ℚ, y = 56 / 99) :=
by
  intro h
  sorry

end repeating_decimal_as_fraction_l15_15084


namespace line_in_slope_intercept_form_l15_15561

theorem line_in_slope_intercept_form :
  ∀ (x y : ℝ), 
    (⟨2, -1⟩ : ℝ × ℝ) • (⟨x, y⟩ - ⟨4, -5⟩) = 0 →
    y = 2 * x - 13 :=
by {
  intros x y h,
  sorry
}

end line_in_slope_intercept_form_l15_15561


namespace ratio_of_teenagers_to_toddlers_l15_15956

theorem ratio_of_teenagers_to_toddlers
  (total_children : ℕ)
  (number_of_toddlers : ℕ)
  (number_of_newborns : ℕ)
  (h1 : total_children = 40)
  (h2 : number_of_toddlers = 6)
  (h3 : number_of_newborns = 4)
  : (total_children - number_of_toddlers - number_of_newborns) / number_of_toddlers = 5 :=
by
  sorry

end ratio_of_teenagers_to_toddlers_l15_15956


namespace tangent_line_eq_mon_incr_l15_15282

def f (a : ℝ) (x : ℝ) := (1 / x + a) * Real.log (1 + x)

theorem tangent_line_eq (a : ℝ) (h_a : a = -1) :
  ∃ m b : ℝ, (∀ x y, y = (1 / x - 1) * Real.log (1 + x) → y = m * x + b) ∧ m * 1 + b = 0 ∧ m = -Real.log 2 ∧ b = Real.log 2 :=
sorry

theorem mon_incr (a : ℝ) (h_mon : ∀ x : ℝ, 0 < x → (deriv (f a)) x ≥ 0) :
  1 / 2 ≤ a :=
sorry

end tangent_line_eq_mon_incr_l15_15282


namespace student_estimated_score_l15_15907

theorem student_estimated_score :
  (6 * 5 + 3 * 5 * (3 / 4) + 2 * 5 * (1 / 3) + 1 * 5 * (1 / 4)) = 41.25 :=
by
 sorry

end student_estimated_score_l15_15907


namespace find_a_minus_b_l15_15830

-- Given definitions for conditions
variables (a b : ℤ)

-- Given conditions as hypotheses
def condition1 := a + 2 * b = 5
def condition2 := a * b = -12

theorem find_a_minus_b (h1 : condition1 a b) (h2 : condition2 a b) : a - b = -7 :=
sorry

end find_a_minus_b_l15_15830


namespace solve_for_x_l15_15700

-- Given condition
def condition (x : ℝ) : Prop :=
  16^(x+2) = 112 + 16^x

-- The value we're trying to prove for x
def x_value (x : ℝ) : ℝ :=
  Real.log (112 / 255) / Real.log 16

theorem solve_for_x (x : ℝ) : condition x → x = x_value x := 
by
  sorry

end solve_for_x_l15_15700


namespace max_viewers_after_T_l15_15584

theorem max_viewers_after_T 
  (rating : ℕ → ℕ) 
  (sum_ratings : ℕ → ℕ) 
  (n : ℕ) 
  (x : ℕ)
  (x_max : x = 10)
  (n_min : n = 1)
  (H : ∀ m, rating m = sum_ratings m / m)
  (HT : ∃ T, rating T = x ∧ ∀ k > T, rating k = rating (k - 1) - 1) :
  ∃ N, N = 5 := 
begin
  -- proof goes here
  sorry
end

end max_viewers_after_T_l15_15584


namespace volume_pyramid_l15_15165

-- Define the basic geometry of the problem
def base_rectangle (a b : ℝ) : Prop := a = 2 ∧ b = 1

-- Define the center of the rectangle
def center (a b : ℝ) : ℝ := Math.sqrt ((a^2 + b^2) / 2)

-- Define the height QO using the given angle θ
def height_QO (θ : ℝ) : ℝ := Math.sqrt (Math.tan(θ)^2 + 0.25)

-- Final volume formula
def pyramid_volume (θ : ℝ) : ℝ := (2 / 3) * height_QO θ

-- Prove the volume for the given conditions
theorem volume_pyramid (a b : ℝ) (θ : ℝ) 
    (h_base : base_rectangle a b) :
    pyramid_volume θ = (2 / 3) * Math.sqrt (Math.tan(θ)^2 + 0.25) := by
  sorry

end volume_pyramid_l15_15165


namespace period_tan_div_three_l15_15502

theorem period_tan_div_three : ∀ x, ∃ T, (∀ x, y = tan (x / 3)) → (y = tan (x / 3) = y) ↔ (y = tan (x + T / 3))

end period_tan_div_three_l15_15502


namespace value_of_a_plus_b_minus_c_l15_15991

theorem value_of_a_plus_b_minus_c (a b c : ℝ) 
  (h1 : abs a = 1) 
  (h2 : abs b = 2) 
  (h3 : abs c = 3) 
  (h4 : a > b) 
  (h5 : b > c) : 
  a + b - c = 2 := 
sorry

end value_of_a_plus_b_minus_c_l15_15991


namespace boxes_filled_per_week_l15_15930

theorem boxes_filled_per_week : 
  (let hens := 270
       eggs_per_hen_per_day := 1
       days_in_week := 7
       eggs_per_week := hens * eggs_per_hen_per_day * days_in_week
       boxes_capacity := 6
   in eggs_per_week / boxes_capacity) = 315 :=
by
  let hens := 270
  let eggs_per_hen_per_day := 1
  let days_in_week := 7
  let eggs_per_week := hens * eggs_per_hen_per_day * days_in_week
  let boxes_capacity := 6
  have eggs_per_week_calc: eggs_per_week = 1890 := by 
  { calc 
      eggs_per_week = hens * eggs_per_hen_per_day * days_in_week : rfl
      ... = 270 * 1 * 7 : rfl
      ... = 1890 : by norm_num },
  
  have boxes_filled_calc: eggs_per_week / boxes_capacity = 315 := by 
  { calc 
      eggs_per_week / boxes_capacity = 1890 / boxes_capacity : by rw eggs_per_week_calc
      ... = 1890 / 6 : rfl
      ... = 315 : by norm_num },
  
  exact boxes_filled_calc

end boxes_filled_per_week_l15_15930


namespace repeating_decimal_as_fraction_l15_15085

theorem repeating_decimal_as_fraction : (∃ x : ℚ, x = 0.565656... ) → (∃ y : ℚ, y = 56 / 99) :=
by
  intro h
  sorry

end repeating_decimal_as_fraction_l15_15085


namespace inequality_solution_l15_15430

theorem inequality_solution (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  (3 - 2 / (3 * x + 4) < 5) ↔ x ∈ set.Ioo (-5/3 : ℝ) (-4/3) ∪ set.Ioi (-4/3) :=
by
  sorry

end inequality_solution_l15_15430


namespace exists_at_least_one_function_l15_15201

theorem exists_at_least_one_function (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x^2 + f(y^2)) = x^2 + y) → ∃ (f : ℝ → ℝ), true :=
sorry

end exists_at_least_one_function_l15_15201


namespace find_floor_l15_15407

-- Define the total number of floors
def totalFloors : ℕ := 9

-- Define the total number of entrances
def totalEntrances : ℕ := 10

-- Each floor has the same number of apartments
-- The claim we are to prove is that for entrance 10 and apartment 333, Petya needs to go to the 3rd floor.

theorem find_floor (apartment_number : ℕ) (entrance_number : ℕ) (floor : ℕ)
  (h1 : entrance_number = 10)
  (h2 : apartment_number = 333)
  (h3 : ∀ (f : ℕ), 0 < f ∧ f ≤ totalFloors)
  (h4 : ∃ (n : ℕ), totalEntrances * totalFloors * n >= apartment_number)
  : floor = 3 :=
  sorry

end find_floor_l15_15407


namespace inequality_solution_l15_15425

theorem inequality_solution (x : ℝ) (h : 3 * x + 4 ≠ 0) : 
  (3 - 2 / (3 * x + 4) < 5) ↔ (x < -(4 / 3) ∨ x > -(5 / 3)) := 
by
  sorry

end inequality_solution_l15_15425


namespace avg_transform_l15_15243

variable {n : ℕ} (x : Fin n → ℝ) (h : 0 < n)

theorem avg_transform (h_avg : (∑ i, x i) / n = 2) : (∑ i, (3 * x i + 2)) / n = 8 := 
by
  sorry

end avg_transform_l15_15243


namespace PQ_parallel_to_AB_l15_15816

variables (O A B C D E P Q : Type) [CircumscribedQuadrilateral ABCD] 

-- Given conditions as assumptions
variable (h1 : Center O of Circumcircle ABCD ∧ O lies_on AB)
variable (h2 : Reflection E of D across AB)
variable (h3 : Intersection P of AC and DO)
variable (h4 : Intersection Q of BD and CE)

theorem PQ_parallel_to_AB (h1 : Center O of Circumcircle ABCD ∧ O lies_on AB)
                         (h2 : Reflection E of D across AB)
                         (h3 : Intersection P of AC and DO)
                         (h4 : Intersection Q of BD and CE) : PQ ∥ AB :=
sorry


end PQ_parallel_to_AB_l15_15816


namespace count_multiples_of_lcm_in_range_l15_15314

theorem count_multiples_of_lcm_in_range : 
  let lcm_12_7 := Nat.lcm 12 7,
      start := 200,
      stop := 500 in
  (Finset.filter (λ n => n % lcm_12_7 = 0) (Finset.Ico start (stop + 1))).card = 3 := by sorry

end count_multiples_of_lcm_in_range_l15_15314


namespace third_derivative_correct_l15_15195

noncomputable def func (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative_correct :
  (deriv^[3] func) x = (4 / (1 + x^2)^2) :=
sorry

end third_derivative_correct_l15_15195


namespace card_statements_l15_15119

theorem card_statements {P1 P2 P3 P4 P5 : Prop}
  (h1 : P1 ↔ (exactly_one [P1, P2, P3, P4, P5]))
  (h2 : P2 ↔ (exactly_two [P1, P2, P3, P4, P5]))
  (h3 : P3 ↔ (exactly_three [P1, P2, P3, P4, P5]))
  (h4 : P4 ↔ (exactly_four [P1, P2, P3, P4, P5]))
  (h5 : P5 ↔ (none_false [P1, P2, P3, P4, P5])) :
  exactly_three_false [P1, P2, P3, P4, P5] :=
sorry

end card_statements_l15_15119


namespace timothy_mother_meet_distance_l15_15883

noncomputable def distance_mother_drives 
  (timothy_speed mother_speed : ℝ) 
  (time_delay : ℝ) 
  (initial_distance : ℝ) : ℝ :=
  let relative_speed := mother_speed - timothy_speed in
  let time_to_catch := initial_distance / relative_speed in
  mother_speed * time_to_catch

theorem timothy_mother_meet_distance :
  distance_mother_drives 6 36 (15 / 60) (6 * (15 / 60)) = 1.8 :=
by
  sorry

end timothy_mother_meet_distance_l15_15883


namespace tan_alpha_eq_l15_15760

def vector_a (α : ℝ) : ℝ × ℝ := (3 / 4, Real.sin α)
def vector_b (α : ℝ) : ℝ × ℝ := (Real.cos α, 1 / 3)

theorem tan_alpha_eq (α : ℝ) (h : (vector_a α).1 * (vector_b α).1 + (vector_a α).2 * (vector_b α).2 = 0) :
  Real.tan α = -9 / 4 :=
sorry

end tan_alpha_eq_l15_15760


namespace magnitude_of_a_l15_15301

def vector := ℝ × ℝ

noncomputable def magnitude (v : vector) : ℝ :=
real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (v w : vector) : ℝ :=
v.1 * w.1 + v.2 * w.2

theorem magnitude_of_a
  (x : ℝ)
  (a : vector := (1, x))
  (b : vector := (-1, x))
  (h : dot_product ((2 * 1, 2 * x) - b) b = 0) :
  magnitude a = 2 :=
by sorry

end magnitude_of_a_l15_15301


namespace least_number_divisible_by_five_primes_l15_15062

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, (n > 0) ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → p ∣ n) ∧ n = 2310 :=
begin
  sorry
end

end least_number_divisible_by_five_primes_l15_15062


namespace prob_exactly_M_laws_in_concept_l15_15716

theorem prob_exactly_M_laws_in_concept 
  (K N M : ℕ) (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let q := 1 - (1 - p)^N in
  (nat.choose K M) * q^M * (1 - q)^(K - M) = 
  (nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M) :=
by {
  let q := 1 - (1 - p)^N,
  have hq_def : q = 1 - (1 - p)^N := rfl,
  rw [hq_def],
  sorry
}

end prob_exactly_M_laws_in_concept_l15_15716


namespace range_of_b_l15_15317

noncomputable def f (x b : ℝ) : ℝ :=
  - (1/2) * x^2 + b * Real.log(x + 2)

def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x ≥ f y

theorem range_of_b (b : ℝ) :
  is_decreasing_on (λ x, - (1/2) * x^2 + b * Real.log(x + 2)) {x | -1 < x} →
  b ≤ -1 :=
sorry

end range_of_b_l15_15317


namespace parametric_to_rect_coords_intersection_points_l15_15745

theorem parametric_to_rect_coords (t α : ℝ) (x y : ℝ) (h₀ : 0 ≤ α ∧ α < π)
    (h_line : x = t * cos α ∧ y = 1 + t * sin α)
    (h_curve : (x^2 + y^2) * (cos θ)^2 = 4 * sin θ) :
    (sin α * x - cos α * y + cos α = 0) ∧ (x^2 = 4 * y) := 
  by
    sorry

theorem intersection_points (t1 t2 α : ℝ) (h_line : t1 ≠ t2) (h_dist : abs (t1 - t2) = 8) :
    α = π/4 ∨ α = 3*π/4 :=
  by
    sorry

end parametric_to_rect_coords_intersection_points_l15_15745


namespace repeating_decimal_eq_l15_15081

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : (0.565656...) = repeating_decimal :=
sorry

end repeating_decimal_eq_l15_15081


namespace calculate_expression_l15_15160

theorem calculate_expression : (5 + 7 + 3) / 3 - 2 / 3 = 13 / 3 := by
  sorry

end calculate_expression_l15_15160


namespace triangle_inertia_l15_15778

noncomputable def triangle_relation (A B C A1 A2 B1 B2 C1 C2 P O : Point) (R : ℝ) : Prop :=
  -- Definition of the conditions
  (are_colinear A B A1) ∧ (are_colinear A B A2) ∧ 
  (are_colinear B C B1) ∧ (are_colinear B C B2) ∧ 
  (are_colinear C A C1) ∧ (are_colinear C A C2) ∧
  parallel A1 A2 B C ∧ parallel B1 B2 C A ∧ parallel C1 C2 A B ∧ 
  intersect A1 A2 B1 B2 P ∧ intersect B1 B2 C1 C2 P ∧ intersect C1 C2 A1 A2 P ∧ 
  is_center_of_circumscribed_circle O A B C

-- The theorem to be proved
theorem triangle_inertia (A B C A1 A2 B1 B2 C1 C2 P O : Point) (R : ℝ) (h : triangle_relation A B C A1 A2 B1 B2 C1 C2 P O R) :
  PA1 P * PA2 P + PB1 P * PB2 P + PC1 P * PC2 P = R^2 - OP^2 :
  sorry

end triangle_inertia_l15_15778


namespace alice_oranges_proof_l15_15147

-- Definitions for conditions
def oranges_emily_sold (E : ℕ) := E
def oranges_alice_sold (E : ℕ) := 2 * E
def total_oranges_sold (E : ℕ) := E + 2 * E

-- Proof statement
theorem alice_oranges_proof : ∀ E : ℕ, total_oranges_sold E = 180 → oranges_alice_sold E = 120 :=
begin
  intros E h,
  sorry
end

end alice_oranges_proof_l15_15147


namespace certain_number_of_tenths_l15_15697

theorem certain_number_of_tenths (n : ℝ) (h : n = 375 * (1/10)) : n = 37.5 :=
by
  sorry

end certain_number_of_tenths_l15_15697


namespace graph_shift_property_l15_15987

theorem graph_shift_property :
  ∀ (f g : ℝ → ℝ),
    (∀ x, f x = sin (x + π / 2)) →
    (∀ x, g x = cos (x - π / 2)) →
    (∀ x, f x = g (x + π / 2)) :=
by
  sorry

end graph_shift_property_l15_15987


namespace work_together_days_l15_15516

theorem work_together_days (a_days : ℕ) (b_days : ℕ) :
  a_days = 10 → b_days = 9 → (1 / ((1 / (a_days : ℝ)) + (1 / (b_days : ℝ)))) = 90 / 19 :=
by
  intros ha hb
  sorry

end work_together_days_l15_15516


namespace tangent_line_eq_mon_incr_l15_15281

def f (a : ℝ) (x : ℝ) := (1 / x + a) * Real.log (1 + x)

theorem tangent_line_eq (a : ℝ) (h_a : a = -1) :
  ∃ m b : ℝ, (∀ x y, y = (1 / x - 1) * Real.log (1 + x) → y = m * x + b) ∧ m * 1 + b = 0 ∧ m = -Real.log 2 ∧ b = Real.log 2 :=
sorry

theorem mon_incr (a : ℝ) (h_mon : ∀ x : ℝ, 0 < x → (deriv (f a)) x ≥ 0) :
  1 / 2 ≤ a :=
sorry

end tangent_line_eq_mon_incr_l15_15281


namespace A_initial_investment_l15_15140

-- Define the conditions
variable (X : ℝ) -- A's initial investment
def A_share := 12 * X -- A's contribution in share
def B_share := 9000 * 7 -- B's contribution in share

-- Define the profit ratio
axiom profit_ratio : A_share / B_share = 2 / 3

-- Statement to be proved
theorem A_initial_investment : X = 3500 :=
by
  -- Using the given proportion and cross-multiplying
  have h1 : 3 * A_share = 2 * B_share := sorry
  -- Replacing A_share and B_share with the actual terms
  rw [A_share, B_share] at h1
  -- Simplifying the equation
  have h2 : 3 * 12 * X = 2 * (9000 * 7) := sorry
  -- Solving for X
  have h3 : 36 * X = 126000 := sorry
  -- Final answer
  have h4 : X = 126000 / 36 := by 
    norm_num 
  exact eq_comm.mp h4

end A_initial_investment_l15_15140


namespace camera_pictures_olivia_camera_pictures_l15_15395

theorem camera_pictures (phone_pics : Nat) (albums : Nat) (pics_per_album : Nat) (total_pics : Nat) : Prop :=
  phone_pics = 5 →
  albums = 8 →
  pics_per_album = 5 →
  total_pics = albums * pics_per_album →
  total_pics - phone_pics = 35

-- Here's the statement of the theorem followed by a sorry to indicate that the proof is not provided
theorem olivia_camera_pictures (phone_pics albums pics_per_album total_pics : Nat) (h1 : phone_pics = 5) (h2 : albums = 8) (h3 : pics_per_album = 5) (h4 : total_pics = albums * pics_per_album) : total_pics - phone_pics = 35 :=
by
  sorry

end camera_pictures_olivia_camera_pictures_l15_15395


namespace least_positive_number_divisible_by_five_smallest_primes_l15_15025

theorem least_positive_number_divisible_by_five_smallest_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  Nat.min (p1 * p2 * p3 * p4 * p5) := 
  2310 :=
by
  sorry

end least_positive_number_divisible_by_five_smallest_primes_l15_15025


namespace equal_or_sum_180_l15_15495

theorem equal_or_sum_180 (A B T C D : Point) (circle₁ circle₂ : Circle) :
  Intersect circle₁ circle₂ A ∧ Intersect circle₁ circle₂ B ∧
  TangentAt circle₁ T ∧ LineThrough T C D Intersect circle₂ ∧
  LineThrough T D C Intersect circle₂ →
  AngleSeen A T C = AngleSeen B T D ∨ AngleSeen A T C + AngleSeen B T D = 180 :=
sorry

end equal_or_sum_180_l15_15495


namespace least_number_divisible_by_five_smallest_primes_l15_15007

theorem least_number_divisible_by_five_smallest_primes : 
  ∃ n ∈ ℕ+, n = 2 * 3 * 5 * 7 * 11 ∧ n = 2310 :=
by
  sorry

end least_number_divisible_by_five_smallest_primes_l15_15007


namespace functions_with_inverses_functions_with_inverses_2_functions_with_inverses_3_functions_with_inverses_4_graphJ_has_inverse_l15_15696

-- Define the functions corresponding to the graphs
def graphF (x : ℝ) : ℝ := -0.5 * x^2 + 3
def graphG1 (x : ℝ) : ℝ := if -4 ≤ x ∧ x ≤ -1 then 2 else 0 / 0
def graphG2 (x : ℝ) : ℝ := if 1 ≤ x ∧ x ≤ 4 then -2 else 0 / 0
def graphH (x : ℝ) : ℝ := 1
def graphI (x : ℝ) : ℝ := Real.sin x
def graphJ (x : ℝ) : ℝ := x

theorem functions_with_inverses :
  (∃ (f : ℝ → ℝ), f = graphF ∧ ∃ (g : ℝ → ℝ), (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)) ↔ false :=
sorry

theorem functions_with_inverses_2 : 
  (∃ (f : ℝ → ℝ), f = graphG1 ∧ f = graphG2 ∧ ∃ (g : ℝ → ℝ), (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)) ↔ false :=
sorry

theorem functions_with_inverses_3 : 
  (∃ (f : ℝ → ℝ), f = graphH ∧ ∃ (g : ℝ → ℝ), (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)) ↔ false :=
sorry

theorem functions_with_inverses_4 : 
  (∃ (f : ℝ → ℝ), f = graphI ∧ ∃ (g : ℝ → ℝ), (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)) ↔ false :=
sorry

theorem graphJ_has_inverse : 
  ∃ (g : ℝ → ℝ), (∀ x, g (graphJ x) = x) ∧ (∀ y, graphJ (g y) = y) :=
sorry

end functions_with_inverses_functions_with_inverses_2_functions_with_inverses_3_functions_with_inverses_4_graphJ_has_inverse_l15_15696


namespace track_and_field_unit_incorrect_l15_15470

theorem track_and_field_unit_incorrect :
  ∀ (L : ℝ), L = 200 → "mm" ≠ "m" → false :=
by
  intros L hL hUnit
  sorry

end track_and_field_unit_incorrect_l15_15470


namespace triplet_sum_nonzero_l15_15509

def TripletA := (1/4, 1/2, -3/4)
def TripletB := (0.5, -1, 0.5)
def TripletC := (3, -5, 2)
def TripletD := (-0.1, 0.3, -0.3)
def TripletE := (1/3, 2/3, -1)

theorem triplet_sum_nonzero : 
  (TripletD.1 + TripletD.2 + TripletD.3 ≠ 0) :=
by 
  trivial

end triplet_sum_nonzero_l15_15509


namespace triangle_side_sum_l15_15493

theorem triangle_side_sum (A B C : Type) [geometry : Triangle A B C] [Angle A = 50] [Angle C = 40]
  (side_opposite_C : length (side B A) = 8 * real.sqrt 3) :
  length (side A B) + length (side B C) = 59.5 :=
by sorry

end triangle_side_sum_l15_15493


namespace sum_of_squares_pentagon_icosagon_l15_15134

theorem sum_of_squares_pentagon_icosagon {R : ℝ} (hR : R > 0) : 
  let a_5 := 2 * R * Real.sin (Real.pi / 5)
  let a_20 := 2 * R * Real.sin (Real.pi / 20)
  in 5 * a_5^2 > 20 * a_20^2 :=
by
  let a_5 := 2 * R * Real.sin (Real.pi / 5)
  let a_20 := 2 * R * Real.sin (Real.pi / 20)
  sorry

end sum_of_squares_pentagon_icosagon_l15_15134


namespace fewest_four_dollar_frisbees_l15_15091

-- Definitions based on the conditions
variables (x y : ℕ) -- The numbers of $3 and $4 frisbees, respectively.
def total_frisbees (x y : ℕ) : Prop := x + y = 60
def total_receipts (x y : ℕ) : Prop := 3 * x + 4 * y = 204

-- The statement to prove
theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : total_frisbees x y) (h2 : total_receipts x y) : y = 24 :=
sorry

end fewest_four_dollar_frisbees_l15_15091


namespace birds_on_fence_l15_15798

theorem birds_on_fence (B S : ℕ): 
  S = 3 →
  S + 6 = B + 5 →
  B = 4 :=
by
  intros h1 h2
  sorry

end birds_on_fence_l15_15798


namespace abc_divisibility_l15_15890

theorem abc_divisibility (a b c : Nat) (h1 : a^3 ∣ b) (h2 : b^3 ∣ c) (h3 : c^3 ∣ a) :
  ∃ k : Nat, (a + b + c)^13 = k * a * b * c :=
by
  sorry

end abc_divisibility_l15_15890


namespace inequality_proof_l15_15536

theorem inequality_proof
  {a b c d e f : ℝ}
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (h_abs : |sqrt (a * d) - sqrt (b * c)| ≤ 1) :
  (a * e + b / e) * (c * e + d / e) ≥ 
    (a^2 * f^2 - (b^2) / (f^2)) * ((d^2) / (f^2) - c^2 * f^2) :=
by
  sorry

end inequality_proof_l15_15536


namespace child_seating_problem_l15_15387

theorem child_seating_problem (x y : ℕ) (h1 : x + y = 7) (h2 : x ≤ 5) (h3 : 2 * x + y ≤ 8) (h4 : y ≤ 6) : x = 1 ∧ y = 6 :=
by {
  sorry,
}

end child_seating_problem_l15_15387


namespace graph_passes_through_fixed_point_l15_15456

theorem graph_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : ∃ (P : ℝ × ℝ), P = (-1, 2) ∧ ∀ x, y = a ^ (x + 1) + 1 → P = (-1, y) := 
by 
  use (-1, 2)
  intros x y h
  have hx : x = -1 := by
    sorry
  rw hx at h
  simp at h
  exact h

end graph_passes_through_fixed_point_l15_15456


namespace total_weight_of_rings_l15_15363

-- Conditions
def weight_orange : ℝ := 0.08333333333333333
def weight_purple : ℝ := 0.3333333333333333
def weight_white : ℝ := 0.4166666666666667

-- Goal
theorem total_weight_of_rings : weight_orange + weight_purple + weight_white = 0.8333333333333333 := by
  sorry

end total_weight_of_rings_l15_15363


namespace ball_hits_ground_at_l15_15157

-- Define the initial conditions and the equation of motion
def velocity : ℝ := -30
def height : ℝ := 200

-- Define the equation of motion for height h as a function of time t
def height_equation (t : ℝ) : ℝ := -16 * t^2 + velocity * t + height

-- Statement to prove that the ball hits the ground at the given time
theorem ball_hits_ground_at :
  ∃ t : ℝ, height_equation t = 0 ∧ t = ( -15 + Real.sqrt 3425 ) / 16 :=
by
  sorry

end ball_hits_ground_at_l15_15157


namespace max_additional_viewers_l15_15585

-- Given conditions:
-- viewers_rating_seq is a sequence of integer ratings given by viewers
-- At moment T, the rating (sum of all given scores divided by their number) is an integer
-- With each new voting viewer, the rating decreases by one unit

theorem max_additional_viewers (rating : ℕ → ℕ) (T : ℕ)
  (h_initial: ∃ x n, x ∈ (set.range (λ i, rating i)) ∧ n = (count (λ i, i ≤ T) rating)
    ∧ (∑ i in finset.range (n+1), rating i) / (n+1) = x)
  (h_decrease: ∀ m > T, (∑ i in finset.range (m+1), rating i) / (m+1) = (h_initial.some - (m - T))):
  ∃ n, n = 5 := 
by 
  sorry

end max_additional_viewers_l15_15585


namespace angle_YXZ_60_deg_l15_15354

noncomputable def triangle_geometry : Type := sorry

def X (t : triangle_geometry) : Type := sorry
def Y (t : triangle_geometry) : Type := sorry
def Z (t : triangle_geometry) : Type := sorry
def P (t : triangle_geometry) : Type := sorry
def Q (t : triangle_geometry) : Type := sorry
def R (t : triangle_geometry) : Type := sorry

axiom angle_XPZ_eq_angle_YPQ (t : triangle_geometry) : 
  -- Given that ∠XPZ = ∠YPQ
  sorry

axiom triangle_PRQ_equilateral (t : triangle_geometry) : 
  -- Given that ∆PRQ is equilateral
  sorry

axiom triangle_XYZ (t : triangle_geometry) (XZ_eq_three_YZ : Type) : 
  -- Given that XZ = 3 * YZ
  sorry

theorem angle_YXZ_60_deg (t : triangle_geometry) (XZ_eq_three_YZ : Type) :
  ∠ YXZ = 60° :=
begin
  sorry
end

end angle_YXZ_60_deg_l15_15354


namespace top_square_is_1_l15_15110

def initial_grid := 
  [[1, 2, 3, 4],
   [5, 6, 7, 8],
   [9, 10, 11, 12],
   [13, 14, 15, 16]]

-- Function to perform the folding operations
def fold_right (grid : List (List ℕ)) := 
  let left := grid.map (λ row => row.take 2)
  let right := grid.map (λ row => row.drop 2)
  let folded := (left.zip right).map (λ (l, r) => r.reverse ++ l)
  folded

def fold_top (grid : List (List ℕ)) := 
  let top := grid.take 2
  let bottom := grid.drop 2
  let folded := (bottom.reverse ++ top)
  folded

def fold_bottom (grid : List (List ℕ)) := 
  let top := grid.take 2
  let bottom := grid.drop 2
  let folded := (bottom ++ top.reverse)
  folded

def fold_left (grid : List (List ℕ)) :=
  let left := grid.map (λ row => row.take 2)
  let right := grid.map (λ row => row.drop 2)
  let folded := (right ++ left)
  folded

-- Folding sequence
def final_grid := fold_left (fold_bottom (fold_top (fold_right initial_grid)))

-- Prove the final answer
theorem top_square_is_1 : final_grid.nth 0 = some [1, 2, 3, 4] :=
by
  -- Placeholder for now
  sorry

end top_square_is_1_l15_15110


namespace cube_root_simplification_l15_15866

theorem cube_root_simplification : (∛(4^6 + 4^6 + 4^6 + 4^6) = 16 * ∛4) :=
by {
  -- Proof goes here
  sorry
}

end cube_root_simplification_l15_15866


namespace union_sets_l15_15235

-- Definitions of sets A and B
def set_A : Set ℝ := {x | x / (x - 1) < 0}
def set_B : Set ℝ := {x | abs (1 - x) > 1 / 2}

-- The problem: prove that the union of sets A and B is (-∞, 1) ∪ (3/2, ∞)
theorem union_sets :
  set_A ∪ set_B = {x | x < 1} ∪ {x | x > 3 / 2} :=
by
  sorry

end union_sets_l15_15235


namespace least_positive_number_divisible_by_five_primes_l15_15050

theorem least_positive_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ 
    (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) :=
by
  use 2310
  split
  { exact by norm_num }
  split
  { intros p hp, fin_cases hp; norm_num }
  { intros m hm, apply nat.le_of_dvd; norm_num }
  sorry

end least_positive_number_divisible_by_five_primes_l15_15050


namespace same_number_of_acquaintances_l15_15335

theorem same_number_of_acquaintances 
  (n : ℕ) 
  (know : fin n → fin n → Prop) 
  (mutual : ∀ a b, know a b ↔ know b a) 
  (h : ∀ a, ¬know a a) : 
  ∃ a b : fin n, a ≠ b ∧ (finset.card (finset.filter (know a) (finset.univ : finset (fin n))) = finset.card (finset.filter (know b) (finset.univ : finset (fin n)))) :=
sorry

end same_number_of_acquaintances_l15_15335


namespace inequality_proof_l15_15539

theorem inequality_proof (a b c d e f : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
    (hcond : abs (sqrt (a * d) - sqrt (b * c)) ≤ 1) :
    (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := 
  sorry

end inequality_proof_l15_15539


namespace intersection_A_B_l15_15295

def A : Set ℝ := {x | |x - 1| < 2}
def B : Set ℝ := {y | ∃ x ∈ (Icc 0 2), y = 2^x}

theorem intersection_A_B : A ∩ B = Ico 1 3 := by
  sorry

end intersection_A_B_l15_15295


namespace max_types_of_balls_same_type_in_box_l15_15494

structure Ball where
  color : String
  size : Nat

def Box := List Ball

def totalBalls (b1 b2 : Box) : Prop := (b1.length + b2.length = 65) 

def validBall (b : Ball) : Prop := 
  b.color ∈ ["white", "black", "red", "yellow"]

def conditionFiveBalls (b : List Ball) : Prop := 
  b.length = 5 → ∃ x y, x ≠ y ∧ x.size = y.size

theorem max_types_of_balls (b1 b2 : Box) (h_total: totalBalls b1 b2) (h_valid: ∀ b ∈ (b1 ++ b2), validBall b) (h_condition: ∀ c : String, conditionFiveBalls ((b1 ++ b2).filter (λ b => b.color = c))) : 
  (∃ T : Finset String × Finset Nat, T.1.card * T.2.card = 16) := 
by sorry

theorem same_type_in_box (b1 b2 : Box) (h_total: totalBalls b1 b2) (h_valid: ∀ b ∈ (b1 ++ b2), validBall b) (h_condition: ∀ c : String, conditionFiveBalls ((b1 ++ b2).filter (λ b => b.color = c))) :
  (∃ b : Box, b = b1 ∨ b = b2 ∧ ∃ x, (b.count x) ≥ 3) := 
by sorry

end max_types_of_balls_same_type_in_box_l15_15494


namespace average_of_three_numbers_l15_15850

theorem average_of_three_numbers (a b c : ℝ)
  (h1 : a + (b + c) / 2 = 65)
  (h2 : b + (a + c) / 2 = 69)
  (h3 : c + (a + b) / 2 = 76) :
  (a + b + c) / 3 = 35 := 
sorry

end average_of_three_numbers_l15_15850


namespace book_arrangement_l15_15919

theorem book_arrangement : 
  let total_books := 9 
  let geometry_books := 4 
  let number_theory_books := 3 
  let algebra_books := 2 
  ∀ (total_pos : ℕ = total_books) 
    (intro_geom : ℕ = geometry_books) 
    (intro_num_thy : ℕ = number_theory_books) 
    (intro_alg : ℕ = algebra_books), 
    (nat.choose 9 4) * (nat.choose 5 3) * (nat.choose 2 2) = 1260 := 
by 
  intros 
  sorry

end book_arrangement_l15_15919


namespace cube_root_of_sum_of_powers_l15_15867

theorem cube_root_of_sum_of_powers :
  ∃ (x : ℝ), x = 16 * (4 ^ (1 / 3)) ∧ x = (4^6 + 4^6 + 4^6 + 4^6) ^ (1 / 3) :=
by
  sorry

end cube_root_of_sum_of_powers_l15_15867


namespace joy_quadrilateral_rod_count_l15_15751

theorem joy_quadrilateral_rod_count :
  let rods := (Finset.range 30).map (λ x, x + 1)
  let selected_rods := {5, 11, 18}
  let remaining_rods := rods \ selected_rods
  remaining_rods.filter (λ d, 2 < d ∧ d < 30) = 25 :=
by
  sorry

end joy_quadrilateral_rod_count_l15_15751


namespace car_pollution_l15_15329

-- Definitions based on conditions
def population : ℕ := 80
def bus_pollution : ℕ := 100
def bus_capacity : ℕ := 40
def switch_to_bus_percentage : ℝ := 0.25
def carbon_reduction : ℕ := 100

-- The pollution per car per year that needs to be proved
def car_pollution_per_year : ℝ := 10

-- Statement to prove
theorem car_pollution: 
  (∃ C : ℝ, C = car_pollution_per_year ∧ 
    (0.75 * population * C + bus_pollution = 0.75 * population * C + bus_pollution + carbon_reduction)
  :=
begin
  sorry
end

end car_pollution_l15_15329


namespace part1_tangent_line_at_x_eq_1_part2_monotonic_increasing_l15_15274

noncomputable def f (x : ℝ) (a : ℝ) := (1/x + a) * real.log(1 + x)

theorem part1_tangent_line_at_x_eq_1 (x := 1) :
  let a := -1 
  let t : ℝ := real.log 2
  let f := (λ x, (1/x - 1) * real.log (1 + x))
  let tangent_line := λ x y, t * x + y - t 
  (tangent_line 1 (f 1) = 0) :=
by {
  sorry
}

theorem part2_monotonic_increasing (a : ℝ) : 
  (∀ x : ℝ, 0 < x → 0 < f x a) ↔ (a ≥ 1/2) := 
by {
  sorry
}

end part1_tangent_line_at_x_eq_1_part2_monotonic_increasing_l15_15274


namespace best_play_wins_probability_best_play_wins_with_multiple_plays_l15_15101

variables (n : ℕ) (m : ℕ) (h_m_gt_2 : m > 2)

-- Part (a) Statement
theorem best_play_wins_probability :
  let votes := 2 * n,
      honest_votes := n,
      biased_votes := n in
  (∃ honesty_bias_partition : (votes)! = (honest_votes)! * (biased_votes)! * ((2 * n - honest_votes - biased_votes)!),
  let tie_probability := (biased_votes!) * (honest_votes!) / (votes)!
  in 1 - tie_probability = (fact (2 * n) - (fact n)^2) / fact (2 * n)) :=
sorry

-- Part (b) Statement
theorem best_play_wins_with_multiple_plays (m_gt_2 : m > 2) :
  let votes := m * n,
      honest_votes := n,
      biased_votes := (m - 1) * n in
  (∃ distribution : ∀ k : ℕ, 1 ≤ k ∧ k < m → (biased_votes / (m - 1)) < honest_votes,
  ∀ k : ℕ, 1 ≤ k ∧ k < m → ¬(k = n)) → 
  (∃ probability := 1, probability = 1) :=
sorry

end best_play_wins_probability_best_play_wins_with_multiple_plays_l15_15101


namespace range_of_m_l15_15667

def f (x : ℝ) (m : ℝ) : ℝ := |2^x - m|
def g (x : ℝ) (m : ℝ) : ℝ := |(1/2) ^ x - m|

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, f (-x) m = f x m → g (-y) m = g y m) →
  ((∀ x : ℝ, f x m = g x m → f' x m ≥ 0) ∨ (∀ x : ℝ, f x m = g x m → f' x m ≤ 0)) →
  (1 / 2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l15_15667


namespace sum_of_digits_10_pow_95_minus_195_l15_15870

theorem sum_of_digits_10_pow_95_minus_195 :
  let n := 10 ^ 95 - 195
  \sum \list.digits n = 841 :=
sorry

end sum_of_digits_10_pow_95_minus_195_l15_15870


namespace tangent_line_at_1_range_of_a_iff_monotonic_increasing_l15_15264

noncomputable def f (x a : ℝ) : ℝ := (1 / x + a) * real.log (1 + x)

theorem tangent_line_at_1 (a : ℝ) (h : a = -1) :
  let f_at := (1 / (1:ℝ) + a) * real.log (1 + 1)
  let f_prime := -real.log 2
  let tangent_line := λ x : ℝ, f_prime * (x - 1) + f_at in
  tangent_line = λ x, -real.log 2 * x + real.log 2 :=
by {
  -- Proof to be provided
  sorry
}

theorem range_of_a_iff_monotonic_increasing :
  ∀ x : ℝ, x > 0 → monotonic_increasing (λ a : ℝ, (1/x + a) * real.log (1 + x)) ↔ a ∈ set.Ici (1/2) :=
by {
  -- Proof to be provided
  sorry
}

end tangent_line_at_1_range_of_a_iff_monotonic_increasing_l15_15264


namespace problem_I_problem_II_l15_15258

noncomputable def f (x : ℝ) : ℝ :=
  |x - 2| + |x - 4|

theorem problem_I : ∃ m, ∀ x, f(x) ≥ m ∧ ∃ y, f(y) = m :=
by {
  let m := 2,
  use m,
  sorry
}

theorem problem_II (a b c n p q : ℝ) (h1 : a^2 + b^2 + c^2 = 2) (h2 : n^2 + p^2 + q^2 = 2) :
  (n^4 / a^2 + p^4 / b^2 + q^4 / c^2) ≥ 2 :=
by {
  sorry
}

end problem_I_problem_II_l15_15258


namespace least_positive_number_divisible_by_five_primes_l15_15056

theorem least_positive_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ 
    (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) :=
by
  use 2310
  split
  { exact by norm_num }
  split
  { intros p hp, fin_cases hp; norm_num }
  { intros m hm, apply nat.le_of_dvd; norm_num }
  sorry

end least_positive_number_divisible_by_five_primes_l15_15056


namespace inequality_proof_l15_15529

noncomputable theory
open real

theorem inequality_proof {a b c d e f : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_d : 0 < d) (h_pos_e : 0 < e) (h_pos_f : 0 < f) (h_ineq : |sqrt(a * d) - sqrt(b * c)| ≤ 1) :
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) :=
sorry

end inequality_proof_l15_15529


namespace geometric_locus_of_point_l15_15228

-- Define the equilateral triangle ABC
variables {A B C : Point} (h_equilateral : equilateral_triangle A B C)

-- Define an internal point M
variable {M : Point} (h_internal : internal_point M (triangle A B C))

-- Define the perpendiculars from M to the sides of the triangle
variable {A' B' C' : Point}
(hA' : perpendicular M A' (line B C))
(hB' : perpendicular M B' (line A C))
(hC' : perpendicular M C' (line A B))

-- Define the condition that triangle A'B'C' is right-angled at A'
variable (h_right_triangle : right_angle (angle A' C' B'))

-- The main theorem to prove
theorem geometric_locus_of_point
  (h_equilateral : equilateral_triangle A B C)
  (h_internal : internal_point M (triangle A B C))
  (hA' : perpendicular M A' (line B C))
  (hB' : perpendicular M B' (line A C))
  (hC' : perpendicular M C' (line A B))
  (h_right_triangle : right_angle (angle A' B' C')) :
  locus_of_point M = union_of_arcs_with_150_deg_angles (triangle A B C) :=
sorry -- Proof is not required.

end geometric_locus_of_point_l15_15228


namespace PQ_parallel_AB_l15_15809

-- Definitions as per the conditions in the problem statement
variables {A B C D E O P Q : Point}
variable [plane_geometry] -- Assume a plane geometry context

-- Conditions:
-- 1. The center O of the circumcircle of quadrilateral ABCD lies on the side AB.
axiom center_O : Circumcenter ABCD O ∧ Collinear {A, B, O}

-- 2. Point E is symmetric to D with respect to the line AB.
axiom symmetric_E : SymmetricPoint D A B E

-- 3. Segments AC and DO intersect at point P.
axiom intersection_P : Intersect (Line A C) (Line D O) P

-- 4. Segments BD and CE intersect at point Q.
axiom intersection_Q : Intersect (Line B D) (Line C E) Q

-- Prove that PQ is parallel to AB.
theorem PQ_parallel_AB : Parallel (Line P Q) (Line A B) :=
by
  sorry

end PQ_parallel_AB_l15_15809


namespace common_tangent_line_number_of_common_tangents_l15_15296

noncomputable def eqn1 : ℝ → ℝ → ℝ :=
  λ x y, x^2 + y^2 + 2*x - 4*y + 4

noncomputable def eqn2 : ℝ → ℝ → ℝ :=
  λ x y, x^2 + y^2 - 4*x - 12*y + 4

theorem common_tangent_line (x y : ℝ) :
  eqn1 x y = 0 → eqn2 x y = 0 → (3 * x + 4 * y = 0) :=
  by {
    sorry
  }

theorem number_of_common_tangents :
  (∃ x y : ℝ, eqn1 x y = 0 ∧ eqn2 x y = 0) → 1 :=
  by {
    sorry
  }

end common_tangent_line_number_of_common_tangents_l15_15296


namespace fencing_cost_l15_15090

noncomputable def total_cost_of_fencing (area short_side cost_per_meter : ℝ) : ℝ :=
  let long_side := area / short_side
  let diagonal := real.sqrt (long_side^2 + short_side^2)
  let total_length := long_side + short_side + diagonal
  total_length * cost_per_meter

theorem fencing_cost (area short_side cost_per_meter total_cost : ℝ) 
  (h_area : area = 1200) 
  (h_short : short_side = 30) 
  (h_cost : cost_per_meter = 11) 
  (h_total_cost : total_cost = 1320) : 
  total_cost_of_fencing area short_side cost_per_meter = total_cost :=
by 
  -- This is where you would provide the proof
  sorry

end fencing_cost_l15_15090


namespace seller_loss_l15_15441

/--
Given:
1. The buyer took goods worth 10 rubles (v_goods : Real := 10).
2. The buyer gave 25 rubles (payment : Real := 25).
3. The seller exchanged 25 rubles of genuine currency with the neighbor (exchange : Real := 25).
4. The seller received 25 rubles in counterfeit currency from the neighbor (counterfeit : Real := 25).
5. The seller gave 15 rubles in genuine currency as change (change : Real := 15).
6. The neighbor discovered the counterfeit and the seller returned 25 rubles to the neighbor (returned : Real := 25).

Prove that the net loss incurred by the seller is 30 rubles.
-/
theorem seller_loss :
  let v_goods := 10
  let payment := 25
  let exchange := 25
  let counterfeit := 25
  let change := 15
  let returned := 25
  (exchange + change) - v_goods = 30 :=
by
  sorry

end seller_loss_l15_15441


namespace tangent_line_equation_at_1_monotonically_increasing_range_of_a_l15_15266

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (1 / x + a) * real.log (1 + x)

-- Part 1: Tangent line equation at point (1, f(1)) when a = -1
theorem tangent_line_equation_at_1 (x y : ℝ) (h : y = f 1 (-1)) :
  (real.log 2) * x + y - (real.log 2) = 0 := 
sorry

-- Part 2: Range of a for which f(x) is monotonically increasing on (0, +∞)
theorem monotonically_increasing_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, 0 < x → 0 < f' x a) ↔ (1 / 2 ≤ a) :=  
sorry

end tangent_line_equation_at_1_monotonically_increasing_range_of_a_l15_15266


namespace maximize_winning_probability_l15_15479

def ahmet_wins (n : ℕ) : Prop :=
  n = 13

theorem maximize_winning_probability :
  ∃ n ∈ {x : ℕ | x ≥ 1 ∧ x ≤ 25}, ahmet_wins n :=
by
  sorry

end maximize_winning_probability_l15_15479


namespace least_number_divisible_by_five_smallest_primes_l15_15002

theorem least_number_divisible_by_five_smallest_primes : 
  ∃ n ∈ ℕ+, n = 2 * 3 * 5 * 7 * 11 ∧ n = 2310 :=
by
  sorry

end least_number_divisible_by_five_smallest_primes_l15_15002


namespace simplify_fraction_l15_15421

theorem simplify_fraction : 
  (1 / (1 / (Real.sqrt 2 + 1) + 2 / (Real.sqrt 3 - 1))) = Real.sqrt 3 - Real.sqrt 2 :=
by
  sorry

end simplify_fraction_l15_15421


namespace calculation_correct_l15_15934

theorem calculation_correct : 1984 + 180 / 60 - 284 = 1703 := 
by 
  sorry

end calculation_correct_l15_15934


namespace solve_trig_eq_l15_15844

theorem solve_trig_eq (x : ℝ) (h : 0 ≤ x ∧ x ≤ π) :
  sin x + sqrt 3 * cos x = 0 ↔ x = 2 * π / 3 :=
by
  sorry

end solve_trig_eq_l15_15844


namespace part1_eq_tangent_line_at_one_l15_15288

def f (x a : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

theorem part1_eq_tangent_line_at_one (a : ℝ) (h : a = -1) :
  ∀ x y : ℝ, 
  f x a = (1/x - 1) * Real.log (1 + x) →
  x = 1 →
  y = 0 →
  (Real.log 2) * x + y - Real.log 2 = 0 := 
sorry

end part1_eq_tangent_line_at_one_l15_15288


namespace solution1_solution2_l15_15935

-- Definition for problem (1)
def problem1 : ℚ :=
  - (1 ^ 4 : ℚ) - (1 / 6) * (2 - (-3 : ℚ) ^ 2) / (-7 : ℚ)

theorem solution1 : problem1 = -7 / 6 :=
by
  sorry

-- Definition for problem (2)
def problem2 : ℚ :=
  ((3 / 2 : ℚ) - (5 / 8) + (7 / 12)) / (-1 / 24) - 8 * ((-1 / 2 : ℚ) ^ 3)

theorem solution2 : problem2 = -34 :=
by
  sorry

end solution1_solution2_l15_15935


namespace clowns_per_mobile_l15_15453

theorem clowns_per_mobile (n_mobiles : ℕ) (total_clowns : ℕ) (h_mobiles : n_mobiles = 5) (h_clowns : total_clowns = 140) :
  total_clowns / n_mobiles = 28 :=
by {
  subst h_mobiles,
  subst h_clowns,
  norm_num,
  sorry
}

end clowns_per_mobile_l15_15453


namespace lowest_possible_sale_price_l15_15571

theorem lowest_possible_sale_price (list_price : ℝ) (h1 : list_price = 80) :
  let discount70 := 0.70 * list_price,
      price_after_first_discount := list_price - discount70,
      additional_discount := 0.20 * list_price,
      lowest_sale_price := price_after_first_discount - additional_discount
  in lowest_sale_price / list_price * 100 = 10 :=
by
  sorry

end lowest_possible_sale_price_l15_15571


namespace triangle_side_calculation_l15_15225

-- Declare noncomputable theory
noncomputable theory

-- Open namespaces
open Real

-- Conditions:
-- Given sides a, b and a median k of a triangle

theorem triangle_side_calculation
  (a b k : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : k > 0) :
  ∃ c : ℝ, c = sqrt (2 * (a^2 + b^2 - 2*k^2) / 3) :=
begin
  sorry,
end

end triangle_side_calculation_l15_15225


namespace angle_BMC_obtuse_l15_15342

-- Defining necessary points and properties within the given conditions.
variables {A B C D M : Type} [Isom ℝ ℝ ℝ ℝ ℝ ℝ]

-- Given conditions:
-- 1. Triangle ABC is isosceles with base AB.
-- 2. Point D is on side CB such that CD = AC - AB.
-- 3. M is the midpoint of AD.
def given_conditions (h_isosceles : isosceles_triangle A B C)
  (h_CD : dist C D = dist A C - dist A B)
  (h_midpoint : midpoint M A D) : Prop := 
  True

-- Proving that angle BMC is obtuse (-- where obtuse means greater than 90 degrees).
theorem angle_BMC_obtuse (h_isosceles : isosceles_triangle A B C)
  (h_CD : dist C D = dist A C - dist A B)
  (h_midpoint : midpoint M A D) : 
  angle_obtuse (∠ B M C) :=
sorry

end angle_BMC_obtuse_l15_15342


namespace least_positive_number_divisible_by_primes_l15_15036

theorem least_positive_number_divisible_by_primes :
  ∃ n : ℕ, n > 0 ∧
    (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧
    (∀ m : ℕ, (m > 0 ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m)) → n ≤ m) ∧
    n = 2310 :=
by
  sorry

end least_positive_number_divisible_by_primes_l15_15036


namespace positive_difference_of_solutions_is_14_l15_15468

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 5 * x + 15 = x + 55

-- Define the positive difference between solutions of the quadratic equation
def positive_difference (a b : ℝ) : ℝ := |a - b|

-- State the theorem
theorem positive_difference_of_solutions_is_14 : 
  ∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ positive_difference a b = 14 :=
by
  sorry

end positive_difference_of_solutions_is_14_l15_15468


namespace roots_in_intervals_l15_15641

theorem roots_in_intervals {a b c : ℝ} (h₁ : a < b) (h₂ : b < c) :
  let f (x : ℝ) := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)
  -- statement that the roots are in the intervals (a, b) and (b, c)
  ∃ r₁ r₂, (a < r₁ ∧ r₁ < b) ∧ (b < r₂ ∧ r₂ < c) ∧ f r₁ = 0 ∧ f r₂ = 0 := 
sorry

end roots_in_intervals_l15_15641


namespace relationship_among_new_stationary_points_l15_15170

-- Define the new stationary point
def new_stationary_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = f' x

-- Define the functions g, h, and φ
def g (x : ℝ) : ℝ := x
def h (x : ℝ) : ℝ := Real.log (x + 1)
def φ (x : ℝ) : ℝ := Real.cos x

-- Define the new stationary points α, β, and γ
noncomputable def alpha : ℝ := (exists_intro 1 (new_stationary_point g 1))
noncomputable def beta : ℝ := classical.some (exists_lt_of_lt 1 (exists_intro (Real.log (1 - (1/exp 1) + 1)) (new_stationary_point h ((-1/exp 1) + 1))))
noncomputable def gamma : ℝ := (exists_intro (3*Real.pi / 4) (new_stationary_point φ (3*Real.pi / 4)))

-- Proof statement
theorem relationship_among_new_stationary_points : beta < alpha ∧ alpha < gamma :=
by {
  sorry
}

end relationship_among_new_stationary_points_l15_15170


namespace infinite_solutions_distinct_natural_numbers_l15_15785

theorem infinite_solutions_distinct_natural_numbers :
  ∃ (x y z : ℕ), (x ≠ y) ∧ (x ≠ z) ∧ (y ≠ z) ∧ (x ^ 2015 + y ^ 2015 = z ^ 2016) :=
by
  sorry

end infinite_solutions_distinct_natural_numbers_l15_15785


namespace probability_selecting_cooking_l15_15897

theorem probability_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let favorable_outcomes := 1
  let total_outcomes := courses.length
  let probability := favorable_outcomes / total_outcomes
  probability = 1 / 4 :=
by
  sorry

end probability_selecting_cooking_l15_15897


namespace nice_plus_sum_l15_15949

def is_nice_plus (n : ℕ) : Prop :=
  ∃ p q r : ℕ, p.prime ∧ q.prime ∧ r.prime ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    (n = p * q ∨ n = p * q * r)

def first_ten_nice_plus_sum (s : ℕ) : Prop :=
  let nice_plus_numbers := [6, 8, 10, 14, 15, 21, 22, 26, 27, 30]
  ∧ list.length nice_plus_numbers = 10
  ∧ s = list.sum nice_plus_numbers

theorem nice_plus_sum : first_ten_nice_plus_sum 179 := 
by {
  sorry
}

end nice_plus_sum_l15_15949


namespace max_voters_after_t_l15_15588

theorem max_voters_after_t
  (n : ℕ) (x : ℕ) (h₁ : 1 ≤ n) (h₂ : x ≤ 10)
  (h_rating : ∃ y : ℕ, (nx + y) / (n + 1) = x - 1) :
  (∃ k : ℕ, k = 5) :=
begin
  sorry
end

end max_voters_after_t_l15_15588


namespace find_value_of_f_f_l15_15254

def f (x : ℝ) (m : ℝ) : ℝ :=
  if x >= 0 then Real.log x / Real.log 3 + m else 1 / 2018

theorem find_value_of_f_f (m : ℝ) (h : f 3 m = 0) : f (f 6 m - 2) m = 1 / 2018 :=
by {
  have h_m : m = -2,
  { sorry }, -- here we will solve for the value of m using provided condition f 3 m = 0
  rw h_m,
  have hx : f 6 (-2) = Real.log 4 / Real.log 3 - 2,
  { sorry }, -- calculation showing f(6) when m=-2
  have h_neg : Real.log 4 / Real.log 3 - 2 < 0,
  { sorry }, -- showing that the result of f(6) - 2 is negative.
  simp only [f, if_neg h_neg],
  exact rfl
}

end find_value_of_f_f_l15_15254


namespace chord_length_of_circle_l15_15445

theorem chord_length_of_circle {
  (line_eq : ℝ → ℝ → Prop) (circle_eq : ℝ → ℝ → Prop)
  (hx : ∀ x y, line_eq x y ↔ x - y + √10 = 0)
  (hy : ∀ x y, circle_eq x y ↔ x^2 + y^2 - 4 * x - 4 * y - 1 = 0) 
} :
  ∃ l : ℝ, l = 4 := 
sorry

end chord_length_of_circle_l15_15445


namespace triangle_side_sum_l15_15491

def sum_of_remaining_sides_of_triangle (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) : Prop :=
  α = 40 ∧ β = 50 ∧ γ = 180 - α - β ∧ c = 8 * Real.sqrt 3 →
  (a + b) = 34.3

theorem triangle_side_sum (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) :
  sum_of_remaining_sides_of_triangle A B C a b c α β γ :=
sorry

end triangle_side_sum_l15_15491


namespace closest_point_on_line_l15_15974

structure Point (α : Type) :=
(x : α) (y : α) (z : α)

def line (s : ℚ) : Point ℚ :=
⟨3 + s, 2 - 3 * s, 4 * s⟩

def distance (p1 p2 : Point ℚ) : ℚ :=
(p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

def closestPoint : Point ℚ := ⟨37/17, 74/17, -56/17⟩

def givenPoint : Point ℚ := ⟨1, 4, -2⟩

theorem closest_point_on_line :
  ∃ s : ℚ, line s = closestPoint ∧ 
           ∀ t : ℚ, distance closestPoint givenPoint ≤ distance (line t) givenPoint :=
by
  sorry

end closest_point_on_line_l15_15974


namespace solution_of_bounded_l15_15514

theorem solution_of_bounded (f : ℝ → ℝ) : 
  (∀ x : ℝ, f'' x + exp x * f x = 0) → (∃ M : ℝ, ∀ x : ℝ, abs (f x) ≤ M) :=
by
  intros h
  sorry

end solution_of_bounded_l15_15514


namespace num_ints_satisfying_inequality_l15_15312

theorem num_ints_satisfying_inequality : ∃ n : ℕ, ∀ a : ℤ, (-4 ≤ a ∧ a ≤ 4) ∧ (-100 < a^3 ∧ a^3 < 100) → n = 9 :=
begin
  sorry
end

end num_ints_satisfying_inequality_l15_15312


namespace min_amount_spent_l15_15871

theorem min_amount_spent
  (box_length : ℝ)
  (box_width : ℝ)
  (box_height : ℝ)
  (box_cost : ℝ)
  (total_volume : ℝ)
  (volume_one_box : ℝ := box_length * box_width * box_height)
  (number_of_boxes : ℝ := (total_volume / volume_one_box).ceil)
  (total_cost : ℝ := number_of_boxes * box_cost) :
  box_length = 20 → box_width = 20 → box_height = 12 → box_cost = 0.50 → total_volume = 1920000 → total_cost = 200 :=
begin
  intros,
  sorry
end

end min_amount_spent_l15_15871


namespace sum_of_divisors_of_24_l15_15504

theorem sum_of_divisors_of_24 : (∑ k in (Finset.filter (λ k, 24 % k = 0) (Finset.range 25)), k) = 60 := 
by
  sorry

end sum_of_divisors_of_24_l15_15504


namespace percent_of_projected_revenue_is_57_69_l15_15518

noncomputable def actual_to_projected_percent (R : ℝ) : ℝ :=
  let projected_revenue := 1.30 * R
  let actual_revenue := 0.75 * R
  (actual_revenue / projected_revenue) * 100

theorem percent_of_projected_revenue_is_57_69 :
  ∀ (R : ℝ), actual_to_projected_percent R ≈ 57.69 :=
by {
  sorry
}

end percent_of_projected_revenue_is_57_69_l15_15518


namespace exist_intersection_point_l15_15100

variables {A B C D : Type} [Pyramid A B C D]
variables {K L M : Type} [OnEdge K A B] [OnEdge L B C] [OnEdge M B D]

def intersection_point (A B C D K L M : Type) [Pyramid A B C D] [OnEdge K A B] [OnEdge L B C] [OnEdge M B D] : Type := 
  -- Declare the existence of point F which is the intersection of planes ACM, CDK, ADL
  ∃ F : Type, IntersectionPlanes A C M && IntersectionPlanes C D K && IntersectionPlanes A D L && F ∈ IntersectionPlanes A C M && F ∈ IntersectionPlanes C D K && F ∈ IntersectionPlanes A D L

-- Now we define the theorem which asserts the existence of point F as required
theorem exist_intersection_point: intersection_point A B C D K L M := 
  sorry

end exist_intersection_point_l15_15100


namespace standard_equation_of_ellipse_l15_15656

noncomputable def ellipse (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :=
  {x y : ℝ // x^2 / a^2 + y^2 / b^2 = 1}

def point_on_ellipse (a b : ℝ) (hx : Real.sqrt 2 / 2) (hy : Real.sqrt 3 / 2) :=
  (hx^2 / a^2) + (hy^2 / b^2) = 1

def eccentricity (a b c : ℝ) (e : ℝ) := 
  e = Real.sqrt 2 / 2 ∧ c = a * Real.sqrt 2 / 2 ∧ a^2 = b^2 + c^2

def circle_intersects_ellipse (m : ℝ) :=
  1 < m ∧ m > 1 ∧ 1 + k^2 * (2 - m^2) > 0 ∧ m^4 - 3 * m^2 + 1 < 0

def max_area_triangle_OMN :=
  Real.sqrt 2 / 2

theorem standard_equation_of_ellipse : 
  ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ (hx : Real.sqrt 2 / 2) ∧ (hy : Real.sqrt 3 / 2) ∧ 
  point_on_ellipse a b hx hy ∧ eccentricity a b c (Real.sqrt 2 / 2) → 
  ∃ k : ℝ, (m : ℝ), circle_intersects_ellipse m ∧ max_area_triangle_OMN :=
sorry

end standard_equation_of_ellipse_l15_15656


namespace sum_powers_of_i_l15_15941

-- Define the conditions
def i : ℂ := Complex.I -- Complex.I is the imaginary unit in ℂ (ℂ is the set of complex numbers)

-- The theorem statement
theorem sum_powers_of_i : (i + i^2 + i^3 + i^4) * 150 + 1 + i + i^2 + i^3 = 0 := by
  sorry

end sum_powers_of_i_l15_15941


namespace speed_in_still_water_l15_15877

def upstream_speed : ℝ := 35
def downstream_speed : ℝ := 45

theorem speed_in_still_water:
  (upstream_speed + downstream_speed) / 2 = 40 := 
by
  sorry

end speed_in_still_water_l15_15877


namespace evaluate_expression_l15_15959

theorem evaluate_expression : 3 + (-3)^2 = 12 := by
  sorry

end evaluate_expression_l15_15959


namespace odd_multiples_of_5_perfect_cubes_l15_15315

theorem odd_multiples_of_5_perfect_cubes : 
  let count := (λ n, 1 ≤ n ∧ n^3 < 24) in
  nat.card { m : ℕ // count m ∧ m^3 < 24 } = 2 :=
by
  sorry

end odd_multiples_of_5_perfect_cubes_l15_15315


namespace ratio_of_areas_l15_15345

variables (A B C D F E : Type) [nonempty A] [nonempty B] [nonempty C] [nonempty D] [nonempty F] [nonempty E]
variables (AD BC AF FB CD DE : ℝ)
variables (area_ABCD area_BFD : ℝ)

-- Conditions
def condition1 := AD = 2 * BC
def condition2 := AF = 2 * FB
def condition3 := CD = 3 * DE

-- The area of rectangle ABCD
def area_rect := AD * BC

-- The area of triangle BFD
def area_triangle := (1/2) * FB * (CD - DE)

-- The Problem Statement in Lean
theorem ratio_of_areas (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  (area_triangle FB CD DE) / (area_rect AD BC) = 1/9 :=
sorry

end ratio_of_areas_l15_15345


namespace solution_set_of_inequality_l15_15246

noncomputable def f : ℝ → ℝ :=
  λ x, - (1/2)*x^2 - x + (3/2)

theorem solution_set_of_inequality (x : ℝ) :
  (f 10^x > 0) ↔ (x < 0) :=
by sorry

end solution_set_of_inequality_l15_15246


namespace least_positive_number_divisible_by_primes_l15_15037

theorem least_positive_number_divisible_by_primes :
  ∃ n : ℕ, n > 0 ∧
    (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧
    (∀ m : ℕ, (m > 0 ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m)) → n ≤ m) ∧
    n = 2310 :=
by
  sorry

end least_positive_number_divisible_by_primes_l15_15037


namespace fraction_each_friend_received_l15_15937

theorem fraction_each_friend_received :
  let original_puffs := 100
  let mom := 7
  let sister := 9
  let grandmother := 11
  let dog := 5
  let uncle := 4
  let cousin := 6
  let friends := 7
  let total_given := mom + sister + grandmother + dog + uncle + cousin
  let remaining_puffs := original_puffs - total_given
  let each_friend_fraction := remaining_puffs / friends
  let simplified_fraction := (8 : ℚ) / 58 in
  simplified_fraction = (4 : ℚ) / 29 :=
begin
  sorry
end

end fraction_each_friend_received_l15_15937


namespace change_received_l15_15464

theorem change_received (price_wooden_toy : ℕ) (price_hat : ℕ) (money_paid : ℕ) (num_wooden_toys : ℕ) (num_hats : ℕ) : 
  price_wooden_toy = 20 → price_hat = 10 → money_paid = 100 → num_wooden_toys = 2 → num_hats = 3 → 
  money_paid - (num_wooden_toys * price_wooden_toy + num_hats * price_hat) = 30 := 
by 
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5] 
  simp 
  norm_num 
  sorry

end change_received_l15_15464


namespace distance_between_X_and_Y_l15_15399

theorem distance_between_X_and_Y 
  (b_walked_distance : ℕ) 
  (time_difference : ℕ) 
  (yolanda_rate : ℕ) 
  (bob_rate : ℕ) 
  (time_bob_walked : ℕ) 
  (distance_when_met : ℕ) 
  (bob_walked_8_miles : b_walked_distance = 8) 
  (one_hour_time_difference : time_difference = 1) 
  (yolanda_3_mph : yolanda_rate = 3) 
  (bob_4_mph : bob_rate = 4) 
  (time_bob_2_hours : time_bob_walked = b_walked_distance / bob_rate)
  : 
  distance_when_met = yolanda_rate * (time_bob_walked + time_difference) + bob_rate * time_bob_walked :=
by
  sorry  -- proof steps

end distance_between_X_and_Y_l15_15399


namespace petya_cannot_have_equal_coins_l15_15460

def petya_initial_two_kopeck_coins : Nat := 1
def petya_initial_ten_kopeck_coins : Nat := 0
def petya_use_ten_kopeck (T G : Nat) : Nat := G - 1 + T + 5
def petya_use_two_kopeck (T G : Nat) : Nat := T - 1 + G + 5

theorem petya_cannot_have_equal_coins : ¬ (∃ n : Nat, 
  ∃ T G : Nat, 
    T = G ∧ 
    (n = petya_use_ten_kopeck T G ∨ n = petya_use_two_kopeck T G ∨ n = petya_initial_two_kopeck_coins + petya_initial_ten_kopeck_coins)) := 
by
  sorry

end petya_cannot_have_equal_coins_l15_15460


namespace max_sum_of_factors_l15_15857

theorem max_sum_of_factors (x y : ℕ) (h1 : x * y = 48) (h2 : x ≠ y) : x + y ≤ 49 :=
by
  sorry

end max_sum_of_factors_l15_15857


namespace max_students_dividing_pens_and_pencils_l15_15095

/-- Prove that the maximum number of students among whom 1020 pens and 860 pencils
can be distributed equally, with each student receiving the same number of pens and pencils, is 20. -/
theorem max_students_dividing_pens_and_pencils (pens : ℕ) (pencils : ℕ) (h_pens : pens = 1020) (h_pencils : pencils = 860) :
  Nat.gcd pens pencils = 20 :=
by {
  rw [h_pens, h_pencils],
  exact Nat.gcd_comm 1020 860,
  sorry 
}

end max_students_dividing_pens_and_pencils_l15_15095


namespace number_of_diagonals_l15_15567

-- Define a rectangular prism with its properties
structure RectangularPrism :=
  (vertices : Finset (Fin 12))
  (edges : Finset (Fin 18))

-- Define what it means for a segment to be diagonal
def is_diagonal (prism : RectangularPrism) (seg : (Fin 12) × (Fin 12)) : Prop :=
  ¬ prism.edges.contain seg ∧ 
  seg.1 ≠ seg.2

-- Define face and space diagonals separately
def face_diagonals (prism : RectangularPrism) : Nat :=
  6 * 2

def space_diagonals (prism : RectangularPrism) : Nat :=
  (12 * 2) // 2

-- Prove the total number of diagonals in a rectangular prism is 24
theorem number_of_diagonals (prism : RectangularPrism) : 
  face_diagonals prism + space_diagonals prism = 24 :=
by 
  sorry

end number_of_diagonals_l15_15567


namespace prob_hit_at_least_twice_l15_15910

-- Defining the probability of hitting the target for one shot
def prob_hit_one_shot : ℝ := 0.6

-- Defining the number of shots
def num_shots : ℕ := 3

-- Prove the probability of hitting at least twice
theorem prob_hit_at_least_twice : 
  (∑ k in {2, 3}, (nat.choose num_shots k : ℝ) * prob_hit_one_shot ^ k * (1 - prob_hit_one_shot) ^ (num_shots - k)) = 81 / 125 :=
by
  sorry

end prob_hit_at_least_twice_l15_15910


namespace cut_one_more_square_l15_15981

theorem cut_one_more_square (G : ℕ → ℕ → Prop) :
  (∃ m n, 11 = m ∧ 11 = n ∧
  (∀ x y, (x < 11 ∧ y < 11) → (G x y → (x % 2 = 0 ∧ y % 2 = 0) ∨ (x % 2 = 1 ∧ y % 2 = 1)))) →
  (∃ x y, x < 10 ∧ y < 10 ∧ (G x y = false) ∧ (G (x+1) y = false) ∧ (G x (y+1) = false) ∧ (G (x+1) (y+1) = false)) :=
begin
    sorry
end

end cut_one_more_square_l15_15981


namespace count_rational_numbers_in_set_l15_15837

def is_rational (x : ℝ) : Prop :=
  ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem count_rational_numbers_in_set :
  let S := ({0, -44 / 7, -3.14, 0.56, Real.pi / 2, -2.010010001} : set ℝ)
  (finite S) →
  (finset.filter is_rational (S.to_finset)).card = 4 :=
by
  intros S hS
  sorry

end count_rational_numbers_in_set_l15_15837


namespace min_n_for_arrangement_l15_15099

def are_coprime (x y : ℕ) : Prop :=
  Nat.gcd x y = 1

def has_common_divisor_greater_than_one (x y : ℕ) : Prop :=
  ∃ d > 1, d ∣ x ∧ d ∣ y

variable (n : ℕ)
variable (a b c d : ℕ)
variable (not_connected : a ≠ b) (connected : a ≠ c)

theorem min_n_for_arrangement :
  ∀ n, (∀ a b, (¬not_connected → are_coprime (a + b) n) ∧ (connected → has_common_divisor_greater_than_one (a + b) n)) → n ≥ 35 :=
sorry

end min_n_for_arrangement_l15_15099


namespace least_positive_whole_number_divisible_by_five_primes_l15_15009

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l15_15009


namespace carla_marbles_l15_15887

theorem carla_marbles (m : ℕ) : m + 134 = 187 ↔ m = 53 :=
by sorry

end carla_marbles_l15_15887


namespace guilt_proof_l15_15177

variables (E F G : Prop)

theorem guilt_proof
  (h1 : ¬G → F)
  (h2 : ¬E → G)
  (h3 : G → E)
  (h4 : E → ¬F)
  : E ∧ G :=
by
  sorry

end guilt_proof_l15_15177


namespace parallel_lines_iff_a_eq_neg3_l15_15102

theorem parallel_lines_iff_a_eq_neg3 (a : ℝ) :
  (∀ x y : ℝ, a * x + 3 * y + 1 = 0 → 2 * x + (a + 1) * y + 1 ≠ 0) ↔ a = -3 :=
sorry

end parallel_lines_iff_a_eq_neg3_l15_15102


namespace club_even_and_neutral_l15_15332

structure Member (G : Type) :=
(friend : G)
(enemy : G)

def ClubEvenAndNeutral (G : Type) [fintype G] [inhabited G]
(member : G → Member G) : Prop :=
  (∃ (n : ℕ), fintype.card G = 2 * n) ∧
  ∃ (neutral1 neutral2 : set G), is_partition {neutral1, neutral2} (λ g1 g2, g1 ≠ g2 ∧ member g1.friend ≠ g2 ∧ member g1.enemy ≠ g2)

theorem club_even_and_neutral {G : Type} [fintype G] [inhabited G]
(member : G → Member G) : ClubEvenAndNeutral G member :=
sorry

end club_even_and_neutral_l15_15332


namespace abba_language_prefix_condition_l15_15350

/-- The Abba language uses two letters: "A" and "B". There exists a condition 
    that no word in this language is a prefix of another word. We need to prove 
    that a dictionary cannot contain 3 four-letter words, 10 five-letter words, 
    30 six-letter words, and 5 seven-letter words under these conditions. -/
theorem abba_language_prefix_condition :
  ¬ ∃ (dict : list string),
    (∀ w ∈ dict, w.length ∈ {4, 5, 6, 7}) ∧
    (∃ four_letters, four_letters.length = 3 ∧
      ∀ w ∈ four_letters, w.length = 4 ∧ list.sublist four_letters dict) ∧
    (∃ five_letters, five_letters.length = 10 ∧
      ∀ w ∈ five_letters, w.length = 5 ∧ list.sublist five_letters dict) ∧
    (∃ six_letters, six_letters.length = 30 ∧
      ∀ w ∈ six_letters, w.length = 6 ∧ list.sublist six_letters dict) ∧
    (∃ seven_letters, seven_letters.length = 5 ∧
      ∀ w ∈ seven_letters, w.length = 7 ∧ list.sublist seven_letters dict) ∧
    (∀ w1 ∈ dict, ∀ w2 ∈ dict, w1 ≠ w2 → ¬ w2.starts_with w1) :=
sorry

end abba_language_prefix_condition_l15_15350


namespace ax2_x_plus_1_positive_l15_15446

theorem ax2_x_plus_1_positive (a : ℝ) :
  (∀ x : ℝ, ax^2 - x + 1 > 0) ↔ (a > 1/4) :=
by {
  sorry
}

end ax2_x_plus_1_positive_l15_15446


namespace least_positive_divisible_by_five_primes_l15_15041

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l15_15041


namespace odd_natural_sequence_sum_l15_15968

theorem odd_natural_sequence_sum (n : ℕ) :
  (∃ x : ℕ, ∑ i in finset.range n, (x + i + 1) = n^2) → 2 ∣ n - 1 :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  sorry

end odd_natural_sequence_sum_l15_15968


namespace axis_of_symmetry_l15_15255

noncomputable def f (x ϕ : ℝ) := Real.sin (x - ϕ)

theorem axis_of_symmetry (ϕ : ℝ) (hϕ : |ϕ| < Real.pi / 2) (h : ∫ x in 0..(2 * Real.pi / 3), f x ϕ = 0) :
  ∃ k : ℤ, x = k * Real.pi + 5 * Real.pi / 6 :=
begin
  sorry
end

end axis_of_symmetry_l15_15255


namespace integer_values_of_n_satisfy_inequality_l15_15305

theorem integer_values_of_n_satisfy_inequality :
  ∃ S : Finset ℤ, (∀ n ∈ S, -100 < n^3 ∧ n^3 < 100) ∧ S.card = 9 :=
by
  -- Sorry provides the placeholder for where the proof would go
  sorry

end integer_values_of_n_satisfy_inequality_l15_15305


namespace compute_y_when_x_is_0_l15_15672

def algorithm (x : ℕ) : ℝ :=
  if x > 4 then x + 2 else (Real.sqrt (4 - x))

theorem compute_y_when_x_is_0 : algorithm 0 = 2 := by
  sorry

end compute_y_when_x_is_0_l15_15672


namespace problem_proof_l15_15218

variables {x y : ℝ}

-- Given condition for the distance relationship
def dist_condition (x y : ℝ) : Prop :=
  real.sqrt ((x - 1)^2 + y^2) = abs (x + 2) - 1

-- Definition of the trajectory equation
def trajectory_equation (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Points and midpoints definitions
def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

def line_passing_through_fixed_point (A B M N P Q E : ℝ × ℝ) : Prop :=
  let PQ_slope := (Q.2 - P.2) / (Q.1 - P.1) in
  P.1 + (E.2 - P.2) / PQ_slope = E.1

theorem problem_proof :
  (∀ x y, dist_condition x y → trajectory_equation x y) ∧ 
  (∀ A B M N P Q,
    let P := midpoint A.1 A.2 B.1 B.2 in
    let Q := midpoint M.1 M.2 N.1 N.2 in
    ∀ E, E = (3, 0) → line_passing_through_fixed_point A B M N P Q E)
:=
by { sorry }

end problem_proof_l15_15218


namespace log_mul_l15_15213

variables (x y : ℝ)

theorem log_mul (hx : 0 < x) (hy : 0 < y) : 2^(log (x * y)) = 2^(log x) * 2^(log y) :=
sorry

end log_mul_l15_15213


namespace least_positive_whole_number_divisible_by_five_primes_l15_15011

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l15_15011


namespace find_n_l15_15647
-- Import the necessary library

-- Define the problem in Lean 4
theorem find_n (n : ℕ) (hn : n > 0) (h : ∑ i in finset.range n, (binomial n (i+1) * 2^i) = 40) : n = 4 := 
by
  sorry

end find_n_l15_15647


namespace least_positive_whole_number_divisible_by_five_primes_l15_15008

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l15_15008


namespace area_of_ellipse_l15_15600

noncomputable def ellipse_area (a b : ℝ) (pi : ℝ) : ℝ :=
  pi * a * b

theorem area_of_ellipse 
  (a b : ℝ) (pi : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : F = ⟨sqrt 6, 0⟩) 
  (h4 : ∃ A B : (ℝ × ℝ), line l passes through F and intersects the ellipse at A and B) 
  (h5 : ellipse_eq : ∀ (x y : ℝ), ((x^2) / (a^2)) + ((y^2) / (b^2)) = 1 )
  (h6 : midpoint_AB : ((sqrt 6) / 3, -1))
: ellipse_area a b pi = 12 * sqrt 3 * pi :=
sorry

end area_of_ellipse_l15_15600


namespace plane_equation_l15_15513

noncomputable def pointA : ℝ × ℝ × ℝ := (1, 0, -2)
noncomputable def pointB : ℝ × ℝ × ℝ := (2, -1, 3)
noncomputable def pointC : ℝ × ℝ × ℝ := (0, -3, 2)

noncomputable def vectorBC : ℝ × ℝ × ℝ :=
  (pointC.1 - pointB.1, pointC.2 - pointB.2, pointC.3 - pointB.3)

theorem plane_equation : ∀ x y z : ℝ, vectorBC.1 * (x - pointA.1) + vectorBC.2 * (y - pointA.2) + vectorBC.3 * (z - pointA.3) = 0 → 2 * x + 2 * y + z = 0 :=
by
  intros x y z h
  have eq1 : vectorBC = (-2, -2, -1) := by sorry
  sorry

end plane_equation_l15_15513


namespace sum_of_integer_solutions_eq_zero_l15_15975

theorem sum_of_integer_solutions_eq_zero :
  (∑ x in {x : ℤ | x^4 - 49 * x^2 + 576 = 0}, x) = 0 :=
by sorry

end sum_of_integer_solutions_eq_zero_l15_15975


namespace limit_position_Q_l15_15155

noncomputable def semicircle_eq (r : ℝ) : ℝ → ℝ := λ x, real.sqrt (r ^ 2 - x ^ 2)
def parabola_eq (x : ℝ) : ℝ := real.sqrt x

theorem limit_position_Q (r : ℝ) (h : r > 0) :
  let α := (-1 + real.sqrt (1 + 4 * r ^ 2)) / 2,
      β := real.sqrt α,
      P : ℝ × ℝ := (0, r),
      M : ℝ × ℝ := (α, β),
      slope := (β - r) / α,
      Q : ℝ := r * α / (r - β)
  in α ^ 2 + α = r ^ 2 → (y : ℝ) (y = 0) → limit (λ r, Q) (𝓝 0) = 2 := sorry

end limit_position_Q_l15_15155


namespace least_digits_repeating_block_seven_ninths_l15_15450

theorem least_digits_repeating_block_seven_ninths :
  ∃ n : ℕ, (n > 0 ∧ (to_digits 10 (7 / 9)) = (to_digits 10 (7 / 9)) ++ (to_digits 10 (7 / 9)).tail.repeat n) ∧ n = 1 :=
sorry

end least_digits_repeating_block_seven_ninths_l15_15450


namespace distance_S_to_plane_ABC_l15_15344

-- Given definitions translating the conditions
variable (a b c : ℝ)
-- Define that the edges are positive real numbers
variable (hpos : 0 < a ∧ 0 < b ∧ 0 < c)

-- Define the distance from point S to the plane ABC
def distance_from_point_to_plane (a b c : ℝ) : ℝ :=
  (a * b * c) / Real.sqrt (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)

-- The theorem stating that this formula gives the correct solution
theorem distance_S_to_plane_ABC :
  distance_from_point_to_plane a b c = (a * b * c) / Real.sqrt (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) :=
by
  sorry

end distance_S_to_plane_ABC_l15_15344


namespace inequality_proof_l15_15526

noncomputable theory
open real

theorem inequality_proof {a b c d e f : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_d : 0 < d) (h_pos_e : 0 < e) (h_pos_f : 0 < f) (h_ineq : |sqrt(a * d) - sqrt(b * c)| ≤ 1) :
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) :=
sorry

end inequality_proof_l15_15526


namespace maximize_rectangle_area_l15_15771

theorem maximize_rectangle_area (l w : ℝ) (h : l + w ≥ 40) : l * w ≤ 400 :=
by sorry

end maximize_rectangle_area_l15_15771


namespace tangent_line_eq_mon_incr_l15_15280

def f (a : ℝ) (x : ℝ) := (1 / x + a) * Real.log (1 + x)

theorem tangent_line_eq (a : ℝ) (h_a : a = -1) :
  ∃ m b : ℝ, (∀ x y, y = (1 / x - 1) * Real.log (1 + x) → y = m * x + b) ∧ m * 1 + b = 0 ∧ m = -Real.log 2 ∧ b = Real.log 2 :=
sorry

theorem mon_incr (a : ℝ) (h_mon : ∀ x : ℝ, 0 < x → (deriv (f a)) x ≥ 0) :
  1 / 2 ≤ a :=
sorry

end tangent_line_eq_mon_incr_l15_15280


namespace calculation_l15_15163

theorem calculation : ((sqrt 2 - 1) ^ 0) - (3 ^ (-1)) = (2 / 3) :=
by
  sorry

end calculation_l15_15163


namespace total_value_of_item_l15_15884

-- Let V denote the total value of the item
def V : ℝ

-- Condition 1: Import tax is 7% on the amount over $1000
def tax (V : ℝ) : ℝ := 0.07 * (V - 1000)

-- Condition 2: The import tax paid is $111.30
def tax_paid : ℝ := 111.30

-- The theorem we want to prove
theorem total_value_of_item : 
  (tax V = tax_paid) → (V = 2590) :=
by
sorry

end total_value_of_item_l15_15884


namespace second_smallest_is_3_probability_l15_15796

noncomputable def probability_of_second_smallest_is_3 : ℚ := 
  let total_ways := Nat.choose 10 6
  let favorable_ways := 2 * Nat.choose 7 4
  favorable_ways / total_ways

theorem second_smallest_is_3_probability : probability_of_second_smallest_is_3 = 1 / 3 := sorry

end second_smallest_is_3_probability_l15_15796


namespace num_integers_satisfying_inequalities_l15_15977

theorem num_integers_satisfying_inequalities :
  (∃ (s : Finset ℕ), s.card = 3 ∧ ∀ n ∈ s, (√(n + 2) ≤ √(3 * n + 1) ∧ √(3 * n + 1) < √(5 * n - 8))) :=
by
  sorry

end num_integers_satisfying_inequalities_l15_15977


namespace find_S_l15_15765

noncomputable def f : ℕ → ℕ
| 0     := 0
| (n+1) := f n + 3

theorem find_S (S : ℕ) (R : ℕ) (hR : R = 3996) (h : 2 * f S = R) : S = 666 :=
by sorry

end find_S_l15_15765


namespace simplify_expression_l15_15606

theorem simplify_expression (y : ℝ) :
  (18 * y^3) * (9 * y^2) * (1 / (6 * y)^2) = (9 / 2) * y^3 :=
by sorry

end simplify_expression_l15_15606


namespace least_positive_divisible_by_five_primes_l15_15038

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l15_15038


namespace part1_eq_tangent_line_at_one_l15_15284

def f (x a : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

theorem part1_eq_tangent_line_at_one (a : ℝ) (h : a = -1) :
  ∀ x y : ℝ, 
  f x a = (1/x - 1) * Real.log (1 + x) →
  x = 1 →
  y = 0 →
  (Real.log 2) * x + y - Real.log 2 = 0 := 
sorry

end part1_eq_tangent_line_at_one_l15_15284


namespace integer_values_of_n_satisfy_inequality_l15_15306

theorem integer_values_of_n_satisfy_inequality :
  ∃ S : Finset ℤ, (∀ n ∈ S, -100 < n^3 ∧ n^3 < 100) ∧ S.card = 9 :=
by
  -- Sorry provides the placeholder for where the proof would go
  sorry

end integer_values_of_n_satisfy_inequality_l15_15306


namespace extra_profit_is_60000_l15_15824

theorem extra_profit_is_60000 (base_house_cost special_house_cost base_house_price special_house_price : ℝ) :
  (special_house_cost = base_house_cost + 100000) →
  (special_house_price = 1.5 * base_house_price) →
  (base_house_price = 320000) →
  (special_house_price - base_house_price - 100000 = 60000) :=
by
  -- Definitions and conditions
  intro h1 h2 h3
  -- Placeholder for the eventual proof
  sorry

end extra_profit_is_60000_l15_15824


namespace least_positive_number_divisible_by_five_smallest_primes_l15_15020

theorem least_positive_number_divisible_by_five_smallest_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  Nat.min (p1 * p2 * p3 * p4 * p5) := 
  2310 :=
by
  sorry

end least_positive_number_divisible_by_five_smallest_primes_l15_15020


namespace triangle_side_sum_l15_15490

def sum_of_remaining_sides_of_triangle (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) : Prop :=
  α = 40 ∧ β = 50 ∧ γ = 180 - α - β ∧ c = 8 * Real.sqrt 3 →
  (a + b) = 34.3

theorem triangle_side_sum (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) :
  sum_of_remaining_sides_of_triangle A B C a b c α β γ :=
sorry

end triangle_side_sum_l15_15490


namespace must_be_odd_l15_15597

theorem must_be_odd (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) :=
sorry

end must_be_odd_l15_15597


namespace total_amount_paid_l15_15838

/-- The owner's markup percentage and the cost price are given. 
We need to find out the total amount paid by the customer, which is equivalent to proving the total cost. -/
theorem total_amount_paid (markup_percentage : ℝ) (cost_price : ℝ) (markup : ℝ) (total_paid : ℝ) 
    (h1 : markup_percentage = 0.24) 
    (h2 : cost_price = 6425) 
    (h3 : markup = markup_percentage * cost_price) 
    (h4 : total_paid = cost_price + markup) : 
    total_paid = 7967 := 
sorry

end total_amount_paid_l15_15838


namespace slant_asymptote_sum_l15_15965

theorem slant_asymptote_sum :
  let y := λx : ℝ, (3 * x^2 - x - 4) / (x - 2)
  let m := 3
  let b := 5
  m + b = 8 :=
by
  let y := λx : ℝ, (3 * x^2 - x - 4) / (x - 2)
  let m := 3
  let b := 5
  show m + b = 8
  exact Eq.refl 8

end slant_asymptote_sum_l15_15965


namespace trig_expression_l15_15643

noncomputable def f (x : ℝ) : ℝ := sin x + cos x
noncomputable def f' (x : ℝ) : ℝ := - sin x + cos x

theorem trig_expression (x : ℝ) (h : f' x = 3 * f x) : 
  (sin x ^ 2 - 3) / (cos x ^ 2 + 1) = -14 / 9 :=
by
  sorry

end trig_expression_l15_15643


namespace sector_area_l15_15998

theorem sector_area (r l : ℝ) (h1 : l + 2 * r = 10) (h2 : l = 2 * r) : 
  (1 / 2) * l * r = 25 / 4 :=
by 
  sorry

end sector_area_l15_15998


namespace right_triangle_hypotenuse_l15_15913

theorem right_triangle_hypotenuse (a b : ℝ) (h : ℝ) (ha : a = 15) (hb : b = 20) : h = 25 :=
by {
  have h_squared : h^2 = a^2 + b^2 := by sorry,
  rw [ha, hb] at h_squared,
  have hs : h^2 = 625 := by sorry,
  have h : h = real.sqrt 625 := by sorry,
  norm_num at h,
  exact h
}

end right_triangle_hypotenuse_l15_15913


namespace find_u_value_l15_15472

theorem find_u_value (h : ∃ n : ℕ, n = 2012) : ∃ u : ℕ, u = 2015 := 
by
  sorry

end find_u_value_l15_15472


namespace overall_percentage_gain_l15_15520

theorem overall_percentage_gain (P : ℝ) (hP1 : P > 0) : 
  let increased_price := P + (P * 36 / 100) in
  let first_discounted_price := increased_price - (increased_price * 10 / 100) in
  let final_price := first_discounted_price - (first_discounted_price * 15 / 100) in
  let gain := final_price - P in
  let percentage_gain := (gain / P) * 100 in
  percentage_gain = 4.04 :=
by
  sorry

end overall_percentage_gain_l15_15520


namespace part1_eq_tangent_line_at_one_l15_15286

def f (x a : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

theorem part1_eq_tangent_line_at_one (a : ℝ) (h : a = -1) :
  ∀ x y : ℝ, 
  f x a = (1/x - 1) * Real.log (1 + x) →
  x = 1 →
  y = 0 →
  (Real.log 2) * x + y - Real.log 2 = 0 := 
sorry

end part1_eq_tangent_line_at_one_l15_15286


namespace mixed_gender_selection_count_is_correct_l15_15980

/- Define the given constants -/
def num_male_students : ℕ := 5
def num_female_students : ℕ := 3
def total_students : ℕ := num_male_students + num_female_students
def selection_size : ℕ := 3

/- Define the function to compute binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

/- The Lean 4 statement -/
theorem mixed_gender_selection_count_is_correct
  (num_male_students num_female_students total_students selection_size : ℕ)
  (hc1 : num_male_students = 5)
  (hc2 : num_female_students = 3)
  (hc3 : total_students = num_male_students + num_female_students)
  (hc4 : selection_size = 3) :
  binom total_students selection_size 
  - binom num_male_students selection_size
  - binom num_female_students selection_size = 45 := 
  by 
    -- Only the statement is required
    sorry

end mixed_gender_selection_count_is_correct_l15_15980


namespace sum_of_all_solutions_l15_15864

variable (x : ℝ)

-- The main equation
def equation (x : ℝ) : Prop :=
  x = |3 * x - |90 - 3 * x||

-- Conditions derived from the absolute value cases
def condition1 (x : ℝ) : Prop := x ≤ 30
def condition2 (x : ℝ) : Prop := x > 30

-- Solutions based on each condition
def solution1 (x : ℝ) : Prop := x = 18 ∨ x = 90 / 7
def solution2 (x : ℝ) : Prop := x = 90

-- Sum of all solutions
def sum_of_solutions : ℝ := 18 + 90 / 7 + 90

-- Proof statement
theorem sum_of_all_solutions : sum_of_solutions = 120.86 :=
by {
  sorry
}

end sum_of_all_solutions_l15_15864


namespace seq_equals_solution_l15_15074

noncomputable def alternating_seq_solution : ℝ :=
  ∑' (n : ℕ), if even n then 3 else 2

theorem seq_equals_solution : 
  (3 + 9 / (2 + 9 / (3 + 9 / (2 + 9 / alternating_seq_solution)))) = (3 + 3 * Real.sqrt 7) / 2 :=
sorry

end seq_equals_solution_l15_15074


namespace isosceles_triangle_crease_length_l15_15926

theorem isosceles_triangle_crease_length
  (A B C D : ℝ × ℝ)
  (AC BC AB : ℝ)
  (hAC : AC = 6)
  (hBC : BC = 6)
  (hAB : AB = 8)
  (hD_midpoint : D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hD_on_AB : D.1 = (A.1 + B.1) / 2 ∧ D.2 = 0 ∧ D.1 = 4)
  (hC_fold_D : C = (D.1, C.2))
  (hC_coordinate : C = (4, 2 * Real.sqrt 5)) :
  (Real.dist C D = 2 * Real.sqrt 5) :=
by
  sorry

end isosceles_triangle_crease_length_l15_15926


namespace least_positive_divisible_by_five_primes_l15_15042

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l15_15042


namespace probability_neither_cake_nor_muffin_l15_15121

open ProbabilityTheory

def total_buyers : ℕ := 100
def cake_mix_buyers : ℕ := 50
def muffin_mix_buyers : ℕ := 40
def both_buyers : ℕ := 15

theorem probability_neither_cake_nor_muffin :
  let buyers_who_purchase_neither := total_buyers - (cake_mix_buyers + muffin_mix_buyers - both_buyers) in
  (buyers_who_purchase_neither : ℝ) / total_buyers = 0.25 :=
by
  sorry

end probability_neither_cake_nor_muffin_l15_15121


namespace total_payment_for_two_dogs_l15_15391

theorem total_payment_for_two_dogs :
  let cost_per_dog := 1000
  let profit_rate := 0.30
  let discount_rate := 0.10
  let tax_rate := 0.05
  let shipping_fee := 50
  let selling_price_per_dog := cost_per_dog * (1 + profit_rate)
  let total_price_before_discount := 2 * selling_price_per_dog
  let discount := discount_rate * total_price_before_discount
  let discounted_price := total_price_before_discount - discount
  let tax := tax_rate * discounted_price
  let price_with_tax := discounted_price + tax
  let total_price := price_with_tax + shipping_fee
  total_price = 2507 :=
by
  let cost_per_dog := 1000
  let profit_rate := 0.30
  let discount_rate := 0.10
  let tax_rate := 0.05
  let shipping_fee := 50
  let selling_price_per_dog := cost_per_dog * (1 + profit_rate)
  let total_price_before_discount := 2 * selling_price_per_dog
  let discount := discount_rate * total_price_before_discount
  let discounted_price := total_price_before_discount - discount
  let tax := tax_rate * discounted_price
  let price_with_tax := discounted_price + tax
  let total_price := price_with_tax + shipping_fee
  show total_price = 2507, from sorry

end total_payment_for_two_dogs_l15_15391


namespace repeating_decimal_eq_l15_15080

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : (0.565656...) = repeating_decimal :=
sorry

end repeating_decimal_eq_l15_15080


namespace probability_third_smallest_is_four_l15_15411

/--
Seven distinct integers are picked at random from the set {1, 2, 3, ..., 12}.
The probability that the third smallest number is 4 is 7/33.
-/
theorem probability_third_smallest_is_four : 
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.to_finset in
  ∀ s : Finset ℕ, s ⊆ S ∧ s.card = 7 →
  let event := { s | s.nth_le 2 (by simp [s.card_eq_coe] ; norm_num) = 4 }.to_finset in
  (event.card : ℚ) / (S.choose 7).card = 7 / 33 :=
by
  intros S S_prop event
  sorry

end probability_third_smallest_is_four_l15_15411


namespace car_moto_separation_time_l15_15885

-- Given conditions: Motorcyclist speed, car speed, and delay in departure time
def moto_speed : ℝ := 45  -- speed in km/h
def car_speed : ℝ := 60   -- speed in km/h
def delay_minutes : ℝ := 40
def distance_apart : ℝ := 36

-- The problem statement
theorem car_moto_separation_time :
  let delay_hours := delay_minutes / 60 in
  let motorcyclist_initial_distance := moto_speed * delay_hours in
  let time_hours := (motorcyclist_initial_distance + distance_apart) / (car_speed - moto_speed) in
  time_hours = 4 + 24 / 60 := 
sorry

end car_moto_separation_time_l15_15885


namespace complex_quadrant_l15_15320

noncomputable def z : ℂ := (complex.I / (2 + complex.I))

theorem complex_quadrant (z = (complex.I / (2 + complex.I))) :
  (0 < z.re) ∧ (0 < z.im) :=
by
  sorry

end complex_quadrant_l15_15320


namespace timber_logging_years_l15_15581

theorem timber_logging_years 
  (V0 : ℝ) (r : ℝ) (V : ℝ) (t : ℝ)
  (hV0 : V0 = 100000)
  (hr : r = 0.08)
  (hV : V = 400000)
  (hformula : V = V0 * (1 + r)^t)
  : t = (Real.log 4 / Real.log 1.08) :=
by
  sorry

end timber_logging_years_l15_15581


namespace repeating_decimal_as_fraction_l15_15083

theorem repeating_decimal_as_fraction : (∃ x : ℚ, x = 0.565656... ) → (∃ y : ℚ, y = 56 / 99) :=
by
  intro h
  sorry

end repeating_decimal_as_fraction_l15_15083


namespace sum_of_roots_l15_15505

-- Definitions based on the conditions
def cubic_equation (x : ℝ) : ℝ := x^3 - x^2 - 13 * x + 13

theorem sum_of_roots : 
  (roots (cubic_equation x)).sum = 1 := 
by sorry -- Proof using Vieta's formulas

end sum_of_roots_l15_15505


namespace total_money_l15_15605

def Billy_money (S : ℕ) := 3 * S - 150
def Lila_money (B S : ℕ) := B - S

theorem total_money (S B L : ℕ) (h1 : B = Billy_money S) (h2 : S = 200) (h3 : L = Lila_money B S) : 
  S + B + L = 900 :=
by
  -- The proof would go here.
  sorry

end total_money_l15_15605


namespace maximum_area_of_inscribed_rectangle_l15_15334

theorem maximum_area_of_inscribed_rectangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (A : ℝ), A = (a * b) / 4 :=
by
  sorry -- placeholder for the proof

end maximum_area_of_inscribed_rectangle_l15_15334


namespace task_completion_time_l15_15575

noncomputable def work_time (A B C : ℝ) : ℝ := 1 / (A + B + C)

theorem task_completion_time (x y z : ℝ) (h1 : 8 * (x + y) = 1) (h2 : 6 * (x + z) = 1) (h3 : 4.8 * (y + z) = 1) :
    work_time x y z = 4 :=
by
  sorry

end task_completion_time_l15_15575


namespace subset_cardinality_l15_15524

theorem subset_cardinality (n k : ℕ) (P Q : set (fin n → fin (k + 1))) :
  (∀ p ∈ P, ∀ q ∈ Q, ∃ m : fin n, p m = q m) →
  ∃ (P' Q' : finset (fin n → fin (k + 1))), 
    (P' ⊆ P ∧ Q' ⊆ Q ∧ (P' = ∅ ∨ Q' = ∅ ∨ P'.card ≤ k^(n - 1) ∨ Q'.card ≤ k^(n - 1))):=
  by
    sorry

end subset_cardinality_l15_15524


namespace cos_sum_inequality_l15_15622

theorem cos_sum_inequality (y : ℝ) (hy : 0 ≤ y ∧ y ≤ π) :
  (∀ x, 0 ≤ x ∧ x ≤ π → cos (x + y) ≥ cos x * cos y) ↔ (y = 0 ∨ y = π) := 
by sorry

end cos_sum_inequality_l15_15622


namespace adele_age_fraction_l15_15389

theorem adele_age_fraction 
  (jackson_age : ℕ) 
  (mandy_age : ℕ) 
  (adele_age_fraction : ℚ) 
  (total_age_10_years : ℕ)
  (H1 : jackson_age = 20)
  (H2 : mandy_age = jackson_age + 10)
  (H3 : total_age_10_years = (jackson_age + 10) + (mandy_age + 10) + (jackson_age * adele_age_fraction + 10))
  (H4 : total_age_10_years = 95) : 
  adele_age_fraction = 3 / 4 := 
sorry

end adele_age_fraction_l15_15389


namespace min_value_of_f_inequality_a_b_l15_15257

theorem min_value_of_f :
  ∃ m : ℝ, m = 4 ∧ (∀ x : ℝ, |x + 3| + |x - 1| ≥ m) :=
sorry

theorem inequality_a_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (1 / a + 4 / b ≥ 9 / 4) :=
sorry

end min_value_of_f_inequality_a_b_l15_15257


namespace find_possible_values_of_a_l15_15300

-- Define the sets P and Q
def P := { x : ℝ | x^2 + x - 6 = 0 }
def Q (a : ℝ) := { x : ℝ | a * x + 1 = 0 }

-- Q is a proper subset of P
def proper_subset (A B : set ℝ) : Prop := A ⊆ B ∧ A ≠ B

theorem find_possible_values_of_a :
  ∀ a : ℝ, proper_subset (Q a) P ↔ a ∈ {0, 1/3, -1/2} :=
by
  sorry -- Proof is skipped

end find_possible_values_of_a_l15_15300


namespace derivative_at_2_l15_15675

noncomputable def f (x : ℝ) : ℝ := x

theorem derivative_at_2 : (deriv f 2) = 1 :=
by
  -- sorry, proof not included
  sorry

end derivative_at_2_l15_15675


namespace triangle_isosceles_if_equal_bisectors_l15_15787

theorem triangle_isosceles_if_equal_bisectors
  (A B C : ℝ)
  (a b c l_a l_b : ℝ)
  (ha : l_a = l_b)
  (h1 : l_a = 2 * b * c * Real.cos (A / 2) / (b + c))
  (h2 : l_b = 2 * a * c * Real.cos (B / 2) / (a + c)) :
  a = b :=
by
  sorry

end triangle_isosceles_if_equal_bisectors_l15_15787


namespace ordered_pairs_eq_4_l15_15698

theorem ordered_pairs_eq_4 :
  {p : ℤ × ℤ | p.fst ^ 4 + p.fst + p.snd ^ 2 = 2 * p.snd + 3}.card = 4 :=
by sorry

end ordered_pairs_eq_4_l15_15698


namespace third_derivative_l15_15196

noncomputable def y (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative (x : ℝ) : (iterated_deriv 3 y) x = 4 / (1 + x^2)^2 :=
by
  sorry

end third_derivative_l15_15196


namespace dad_steps_l15_15168

theorem dad_steps (dad_steps_ratio: ℕ) (masha_steps_ratio: ℕ) (masha_steps: ℕ)
  (masha_and_yasha_steps: ℕ) (total_steps: ℕ)
  (h1: dad_steps_ratio * 3 = masha_steps_ratio * 5)
  (h2: masha_steps * 3 = masha_and_yasha_steps * 5)
  (h3: masha_and_yasha_steps = total_steps)
  (h4: total_steps = 400) :
  dad_steps_ratio * 30 = 90 :=
by
  sorry

end dad_steps_l15_15168


namespace part1_tangent_line_at_x_eq_1_part2_monotonic_increasing_l15_15271

noncomputable def f (x : ℝ) (a : ℝ) := (1/x + a) * real.log(1 + x)

theorem part1_tangent_line_at_x_eq_1 (x := 1) :
  let a := -1 
  let t : ℝ := real.log 2
  let f := (λ x, (1/x - 1) * real.log (1 + x))
  let tangent_line := λ x y, t * x + y - t 
  (tangent_line 1 (f 1) = 0) :=
by {
  sorry
}

theorem part2_monotonic_increasing (a : ℝ) : 
  (∀ x : ℝ, 0 < x → 0 < f x a) ↔ (a ≥ 1/2) := 
by {
  sorry
}

end part1_tangent_line_at_x_eq_1_part2_monotonic_increasing_l15_15271


namespace tangent_line_equation_at_1_monotonically_increasing_range_of_a_l15_15270

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (1 / x + a) * real.log (1 + x)

-- Part 1: Tangent line equation at point (1, f(1)) when a = -1
theorem tangent_line_equation_at_1 (x y : ℝ) (h : y = f 1 (-1)) :
  (real.log 2) * x + y - (real.log 2) = 0 := 
sorry

-- Part 2: Range of a for which f(x) is monotonically increasing on (0, +∞)
theorem monotonically_increasing_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, 0 < x → 0 < f' x a) ↔ (1 / 2 ≤ a) :=  
sorry

end tangent_line_equation_at_1_monotonically_increasing_range_of_a_l15_15270


namespace sum_of_distances_eq_third_line_l15_15853

-- Definition of the lines, point, and distances in the geometric plane
variable (Point : Type) [MetricSpace Point]
variable (Line : Type) [AffineSpace Line Point]

-- Line intersection and angle specifications
constant S : Point
constant ℓ1 ℓ2 ℓ3 : Line
constant angle_S : ∀ {L₁ L₂ : Line}, L₁ ≠ L₂ → ∠(L₁, S, L₂) = 60

-- Axiom or definition for the distance function from a point to a line
constant distance : Point → Line → ℝ

-- The theorem we want to prove
theorem sum_of_distances_eq_third_line (P : Point) : 
  distance P ℓ1 + distance P ℓ2 = distance P ℓ3 ∨
  distance P ℓ1 + distance P ℓ3 = distance P ℓ2 ∨
  distance P ℓ2 + distance P ℓ3 = distance P ℓ1 :=
sorry

end sum_of_distances_eq_third_line_l15_15853


namespace machine_a_produces_6_sprockets_per_hour_l15_15388

theorem machine_a_produces_6_sprockets_per_hour : 
  ∀ (A G T : ℝ), 
  (660 = A * (T + 10)) → 
  (660 = G * T) → 
  (G = 1.10 * A) → 
  A = 6 := 
by
  intros A G T h1 h2 h3
  sorry

end machine_a_produces_6_sprockets_per_hour_l15_15388


namespace trapezium_area_l15_15969

theorem trapezium_area (parallel_side1 parallel_side2 distance : ℝ) 
    (h1 : parallel_side1 = 10) 
    (h2 : parallel_side2 = 18) 
    (h3 : distance = 10.00001) :
    (1/2 * (parallel_side1 + parallel_side2) * distance = 140.00014) := 
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end trapezium_area_l15_15969


namespace statement_A_statement_A_statement_C_statement_D_l15_15679

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

theorem statement_A (x : ℝ) (hx : x > 1) : f x > 0 := sorry

theorem statement_A' (x : ℝ) (hx : 0 < x ∧ x < 1) : f x < 0 := sorry

theorem statement_C : Set.range f = Set.Ici (-1 / (2 * Real.exp 1)) := sorry

theorem statement_D (x : ℝ) : f x ≥ x - 1 := sorry

end statement_A_statement_A_statement_C_statement_D_l15_15679


namespace max_area_of_PQUT_l15_15792

-- Definitions of the conditions
variables (a h : ℝ) -- Side length of the square base and height of the prism
variable (X : ℝ) -- Point on the face TUQV

-- Given conditions
def conditions := (P_X = 12 ∧ Q_X = 10 ∧ R_X = 8 ∧ h^2 + n^2 + m^2 = 144 ∧ h^2 + n^2 + (a - m)^2 = 100 ∧ h^2 + (a - n)^2 + (a - m)^2 = 64)

-- Definition of the area of rectangle PQUT
def area_of_rectangle (a h : ℝ) := a * h

-- The theorem to be proved
theorem max_area_of_PQUT (a h : ℝ) :
  conditions a h →
  (area_of_rectangle a h) ≤ 67.82 :=
sorry

end max_area_of_PQUT_l15_15792


namespace katie_cupcakes_l15_15604

theorem katie_cupcakes (orig_cupcakes : ℕ) (sold_orig : ℕ) (new_batch_percent : ℝ) (yield_percent : ℝ) (choc_percent : ℝ) (van_percent : ℝ) (sold_choc_percent : ℝ) (sold_van_percent : ℝ) :
((orig_cupcakes - sold_orig) + (new_batch_percent * orig_cupcakes) * (1 - sold_choc_percent) * choc_percent + (new_batch_percent * orig_cupcakes) * (1 - sold_van_percent) * van_percent = 14) :=
begin
  have orig_cups : orig_cupcakes = 26 := sorry,
  have sold_cups : sold_orig = 20 := sorry,
  have new_pct : new_batch_percent = 0.80 := sorry,
  have yield_pct : yield_percent = 0.75 := sorry,
  have choc_pct : choc_percent = 0.40 := sorry,
  have van_pct : van_percent = 0.60 := sorry,
  have sold_choc_p : sold_choc_percent = 0.70 := sorry,
  have sold_van_p : sold_van_percent = 0.50 := sorry,
  sorry
end

end katie_cupcakes_l15_15604


namespace height_opposite_Y_l15_15596

-- Define a regular triangle XYZ with the given heights
structure RegularTriangle where
  X Y Z : ℝ × ℝ × ℝ -- Coordinates of the three vertices
  X_height : ℝ
  Y_height : ℝ
  Z_height : ℝ
  side_length : ℝ

-- Define that the side length is consistent
axiom regular_triangle_side_length (r : RegularTriangle) : 
  (dist r.X r.Y = r.side_length) ∧ (dist r.Y r.Z = r.side_length) ∧ (dist r.Z r.X = r.side_length)

-- Define the heights at X, Y, Z positions
def heights := 
(X_height : 8) ∧ (Y_height : 5) ∧ (Z_height : 7)

-- Define the point opposite Y
def point_opposite_Y (r : RegularTriangle) := 
r.X

theorem height_opposite_Y (r : RegularTriangle) : 
  heights r → height (point_opposite_Y r) = 8 :=
by
  sorry

end height_opposite_Y_l15_15596


namespace probability_prime_is_7_over_20_l15_15927

-- Define the condition of discs
def discA := {1, 2, 3, 5}  -- disc A numbers, considering possible sections
def discB := {1, 2, 3, 4, 5}  -- disc B numbers, considering possible sections

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

-- Define the two-digit number formed by a and b
def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Define the set of possible two-digit numbers ab
def possible_numbers : set ℕ := { two_digit_number a b | a ∈ discA, b ∈ discB }

-- Define the set of prime numbers in the possible two-digit numbers
def prime_numbers : set ℕ := {n ∈ possible_numbers | is_prime n}

-- Define the probability of a prime two-digit number formed by ab
def probability_prime : ℚ := (prime_numbers.size : ℚ) / (possible_numbers.size : ℚ)

-- State the theorem
theorem probability_prime_is_7_over_20 : probability_prime = 7 / 20 :=
by
  sorry

end probability_prime_is_7_over_20_l15_15927


namespace prob_exactly_M_laws_in_concept_l15_15712

theorem prob_exactly_M_laws_in_concept 
  (K N M : ℕ) (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let q := 1 - (1 - p)^N in
  (nat.choose K M) * q^M * (1 - q)^(K - M) = 
  (nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M) :=
by {
  let q := 1 - (1 - p)^N,
  have hq_def : q = 1 - (1 - p)^N := rfl,
  rw [hq_def],
  sorry
}

end prob_exactly_M_laws_in_concept_l15_15712


namespace linda_savings_l15_15093

theorem linda_savings :
  ∀ (S : ℝ), (5 / 6 * S + 500 = S) → S = 3000 :=
by
  intros S h
  sorry

end linda_savings_l15_15093


namespace problem_293_l15_15912

theorem problem_293 (s : ℝ) (R' : ℝ) (rectangle1 : ℝ) (circle1 : ℝ) 
  (condition1 : s = 4) 
  (condition2 : rectangle1 = 2 * 4) 
  (condition3 : circle1 = Real.pi * 1^2) 
  (condition4 : R' = s^2 - (rectangle1 + circle1)) 
  (fraction_form : ∃ m n : ℕ, gcd m n = 1 ∧ R' = m / n) : 
  (∃ m n : ℕ, gcd m n = 1 ∧ R' = m / n ∧ m + n = 293) := 
sorry

end problem_293_l15_15912


namespace increasing_interval_l15_15950

def function (x : ℝ) : ℝ := 2 * x^2 - Real.log (2 * x)

theorem increasing_interval :
  ∀ x : ℝ, (x > 0) → (∃ I : Set ℝ, I = Set.Ioi (1 / 2) ∧ (∀ z ∈ I, 0 < deriv function z)) :=
by
  sorry

end increasing_interval_l15_15950


namespace quadratic_tangent_to_x_axis_l15_15691

theorem quadratic_tangent_to_x_axis :
  ∀ (a b c : ℝ), g(x) = a * x^2 + b * x + c →
  a = 3 → b = -6 → c = (b^2) / (4 * a) →
  ∃ x : ℝ, g(x) = 0 ∧ ∀ y ≠ x, g(y) ≠ 0 :=
by
  intros a b c hg ha hb hc
  sorry

end quadratic_tangent_to_x_axis_l15_15691


namespace PQ_parallel_to_AB_l15_15815

variables (O A B C D E P Q : Type) [CircumscribedQuadrilateral ABCD] 

-- Given conditions as assumptions
variable (h1 : Center O of Circumcircle ABCD ∧ O lies_on AB)
variable (h2 : Reflection E of D across AB)
variable (h3 : Intersection P of AC and DO)
variable (h4 : Intersection Q of BD and CE)

theorem PQ_parallel_to_AB (h1 : Center O of Circumcircle ABCD ∧ O lies_on AB)
                         (h2 : Reflection E of D across AB)
                         (h3 : Intersection P of AC and DO)
                         (h4 : Intersection Q of BD and CE) : PQ ∥ AB :=
sorry


end PQ_parallel_to_AB_l15_15815


namespace intersection_is_correct_l15_15993

-- Define the sets A and B based on the given conditions
def setA : Set ℝ := { x | x > 1/3 }
def setB : Set ℝ := { y | -3 ≤ y ∧ y ≤ 3 }

-- Prove that the intersection of A and B is (1/3, 3]
theorem intersection_is_correct : setA ∩ setB = { x | 1/3 < x ∧ x ≤ 3 } := 
by
  sorry

end intersection_is_correct_l15_15993


namespace Nero_time_calculation_l15_15359

variable (Jerome_time : ℝ) (Jerome_speed : ℝ) (Nero_speed : ℝ)

-- Definitions based on conditions
def trail_length (Jerome_time Jerome_speed : ℝ) : ℝ := Jerome_time * Jerome_speed
def Nero_time (trail_length Nero_speed : ℝ) : ℝ := trail_length / Nero_speed

-- Statement of the problem
theorem Nero_time_calculation
  (Jerome_time : ℝ) (Jerome_speed : ℝ) (Nero_speed : ℝ)
  (h1 : Jerome_time = 6) (h2 : Jerome_speed = 4) (h3 : Nero_speed = 8) :
  Nero_time (trail_length Jerome_time Jerome_speed) Nero_speed = 3 :=
  sorry -- proof to be filled

end Nero_time_calculation_l15_15359


namespace sequence_a2012_l15_15222

noncomputable def sequence (n : ℕ) : ℚ :=
  Nat.rec_on n
    (1/2)
    (λ n a_n, 1 / (1 - a_n))

theorem sequence_a2012 : sequence 2012 = 2 := 
by
  -- skipping the proof as per the requirement
  sorry

end sequence_a2012_l15_15222


namespace cos_increasing_intervals_l15_15951

theorem cos_increasing_intervals :
  ∀ k : ℤ, ∀ x ∈ Set.Icc (-(2 * Real.pi / 9) + (2 * k * Real.pi / 3)) ((Real.pi / 9) + (2 * k * Real.pi / 3)),
    3 * Real.sin ((Real.pi / 3) - 3 * x) > 0 :=
begin
  sorry
end

end cos_increasing_intervals_l15_15951


namespace sum_first_10_terms_is_5_over_11_l15_15292

-- Define the sequence a_n based on the given area formula
def a_n (n : ℕ+) : ℚ := 1 / 2 * (1 / n - 1 / (n + 1))

-- Sum the first 10 terms of the sequence a_n
noncomputable def sum_first_10_terms : ℚ :=
  (Finset.range 10).sum (λ n, a_n (n + 1))

-- The proposition to be proved
theorem sum_first_10_terms_is_5_over_11 : sum_first_10_terms = 5 / 11 :=
by sorry

end sum_first_10_terms_is_5_over_11_l15_15292


namespace cosine_between_m_and_n_l15_15303

variables (λ : ℝ)
def m : ℝ × ℝ := (λ + 1, 1)
def n : ℝ × ℝ := (λ + 2, 2)

axiom perpendicular (m n : ℝ × ℝ) : (m + n).fst * (m - n).fst + (m + n).snd * (m - n).snd = 0

def cosine_of_angle_between_vectors (u v : ℝ × ℝ) : ℝ :=
  (u.fst * v.fst + u.snd * v.snd) / 
  (Real.sqrt (u.fst^2 + u.snd^2) * Real.sqrt (v.fst^2 + v.snd^2))

theorem cosine_between_m_and_n 
  (h : perpendicular m n) :
  cosine_of_angle_between_vectors m n = 4 / 5 := sorry

end cosine_between_m_and_n_l15_15303


namespace number_of_numbers_l15_15439

theorem number_of_numbers (n : ℕ) (S : ℕ) 
  (h1 : (S + 26) / n = 16) 
  (h2 : (S + 46) / n = 18) : 
  n = 10 := 
by 
  -- placeholder for the proof
  sorry

end number_of_numbers_l15_15439


namespace quadrilateral_cyclic_l15_15784

-- Define the quadrilateral and the given condition
variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] 

-- Define the angles at vertices
variables (angle_BAD angle_BCD : ℝ)

-- Given condition
axiom angle_sum_condition : angle_BAD + angle_BCD = 180

-- The proof we need to provide
theorem quadrilateral_cyclic (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
    (angle_BAD angle_BCD : ℝ)
    (h : angle_BAD + angle_BCD = 180) :
    ∃ (O : Type) [metric_space O], is_circumcircle O A B C D :=
by
  sorry

end quadrilateral_cyclic_l15_15784


namespace prob_exactly_M_laws_expected_laws_included_l15_15718

noncomputable def prob_of_exactly_M_laws (K N M : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - (1 - p)^N
  (Nat.choose K M) * q^M * (1 - q)^(K - M)

noncomputable def expected_num_of_laws (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

-- Part (a): Prove that the probability of exactly M laws being included is as follows
theorem prob_exactly_M_laws (K N M : ℕ) (p : ℝ) :
  prob_of_exactly_M_laws K N M p =
    (Nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - (1 - p)^N)^(K - M)) :=
sorry

-- Part (b): Prove that the expected number of laws included is as follows
theorem expected_laws_included (K N : ℕ) (p : ℝ) :
  expected_num_of_laws K N p =
    K * (1 - (1 - p)^N) :=
sorry

end prob_exactly_M_laws_expected_laws_included_l15_15718


namespace limit_seq_example_l15_15783

theorem limit_seq_example (a_n : ℕ → ℝ) (a : ℝ) :
  (∀ n, a_n n = (3 * n^2 + 2) / (4 * n^2 - 1)) →
  a = 3 / 4 →
  filter.tendsto a_n filter.at_top (nhds a) :=
by
  intro ha h.
  sorry

end limit_seq_example_l15_15783


namespace combined_perimeter_is_correct_l15_15914

-- Define the initial conditions
def side_length_square (area : ℝ) := sqrt(area)

def perimeter_square (s : ℝ) := 4 * s

def radius_circle (s : ℝ) := s

def circumference_circle (r : ℝ) := 2 * real.pi * r

-- Prove the combined perimeter of the square and the circle
theorem combined_perimeter_is_correct:
  (combined_perimeter : ℝ) =
  let s := side_length_square 1 in
  let p_square := perimeter_square s in
  let r := radius_circle s in
  let c_circle := circumference_circle r in
  p_square + c_circle := by 
      let s := sqrt 1
      have hs : s = 1 := by 
         rw sqrt_one      
      let p_square := 4 * s
      rw [hs, mul_one]
      let r := s
      rw hs at rwi
      let c_circle := 2 * π * 1
      rw mul_one at c_circle
      have h : p_square + c_circle = 4 + 2 * real.pi := by ring
      exact h

end combined_perimeter_is_correct_l15_15914


namespace solution_of_inequality_range_of_k_l15_15642

/-
Proof problem (I):
Given \( f(x) = |ax - 1| \) and the solution set of \( f(x) \leq 3 \) is \(\{ x \mid -1 \leq x \leq 2 \} \), prove \( a = 2 \).
-/
theorem solution_of_inequality (a : ℝ) (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x = |a * x - 1|)
  (h₂ : ∀ x, f x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 2) :
  a = 2 :=
sorry

/-
Proof problem (II):
Prove if \(\frac{f(x) + f(-x)}{3} < |k| \) has real solutions, then \( k \in (-\infty, -\frac{2}{3}) \cup (\frac{2}{3}, +\infty) \).
-/
theorem range_of_k (k : ℝ) (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x = |2 * x - 1|)
  (h₂ : ∀ x, (f x + f (-x)) / 3 < |k|) :
  k ∈ set.Ioo (-2/3 : ℝ) (2/3 : ℝ).compl :=
sorry

end solution_of_inequality_range_of_k_l15_15642


namespace least_positive_number_divisible_by_primes_l15_15031

theorem least_positive_number_divisible_by_primes :
  ∃ n : ℕ, n > 0 ∧
    (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧
    (∀ m : ℕ, (m > 0 ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m)) → n ≤ m) ∧
    n = 2310 :=
by
  sorry

end least_positive_number_divisible_by_primes_l15_15031


namespace valid_table_if_and_only_if_multiple_of_9_l15_15193

/-- Define the properties of the n x n table -/
def valid_table (n : ℕ) (table : List (List Char)) : Prop :=
  let letters := ['I', 'M', 'O']
  -- each row and column must have equal occurrences of 'I', 'M', and 'O'
  (∀ i < n, letters.all (λ c, (table[i].count (· == c)) = n / 3)) ∧
  (∀ j < n, letters.all (λ c, (table.map (λ row, row[j])).count (· == c) = n / 3)) ∧
  -- each diagonal whose length is a multiple of 3 must have equal occurrences of 'I', 'M', and 'O'
  (∀ k in [3, 6..n], letters.all (λ c,
    (List.diagonal table k).count (· == c) = k / 3 ∧
    (List.antidiagonal table k).count (· == c) = k / 3))

/-- Theorem: The table satisfies the given properties if and only if n is a multiple of 9 -/
theorem valid_table_if_and_only_if_multiple_of_9 (n : ℕ) :
  (∃ (table : List (List Char)), valid_table n table) ↔ ∃ (k : ℕ), n = 9 * k :=
by sorry

end valid_table_if_and_only_if_multiple_of_9_l15_15193


namespace ninety_times_ninety_l15_15618

theorem ninety_times_ninety : (90 * 90) = 8100 := by
  let a := 100
  let b := 10
  have h1 : (90 * 90) = (a - b) * (a - b) := by decide
  have h2 : (a - b) * (a - b) = a^2 - 2 * a * b + b^2 := by decide
  have h3 : a = 100 := rfl
  have h4 : b = 10 := rfl
  have h5 : 100^2 - 2 * 100 * 10 + 10^2 = 8100 := by decide
  sorry

end ninety_times_ninety_l15_15618


namespace james_total_calories_burned_l15_15360

def calories_per_hour_walking : ℕ := 300
def calories_per_hour_dancing : ℕ := 2 * calories_per_hour_walking
def calories_per_hour_swimming : ℕ := 1.5 * calories_per_hour_walking
def calories_per_hour_cycling : ℕ := calories_per_hour_walking

def dancing_days : list string := ["Monday", "Wednesday", "Friday"]
def swimming_days : list string := ["Tuesday", "Thursday"]
def cycling_days : list string := ["Sunday"]

def total_calories_dancing_week : ℕ := 
  (dancing_days.length * 600) + 600

def total_calories_swimming_week : ℕ :=
  swimming_days.length * 675

def total_calories_cycling_week : ℕ :=
  600

def total_calories_week : ℕ :=
  total_calories_dancing_week + total_calories_swimming_week + total_calories_cycling_week

theorem james_total_calories_burned :
  total_calories_week = 4350 := by
  sorry

end james_total_calories_burned_l15_15360


namespace problem1_problem2_problem3_problem4_l15_15936

noncomputable section

theorem problem1 : 
  -1 ^ 2023 + (2023 - Real.pi) ^ 0 + (-⅟2 : ℚ) ^ -2 = 4 := 
by sorry

theorem problem2 (x y : ℚ) : 
  (-⅟2 * x * y) ^ 3 * 2 * x * (2 * x ^ 3 * y ^ 2) / (x * y ^ 2) ^ 2 
= -⅟2 * x ^ 5 * y := 
by sorry

theorem problem3 (x y : ℚ) : 
  (x - y) * (x + y) * (x ^ 2 - y ^ 2) = x ^ 4 - 2 * x ^ 2 * y ^ 2 + y ^ 4 := 
by sorry

theorem problem4 (x y : ℚ) : 
  ((x + 2 * y) ^ 2 + (3 * x + 2 * y) * (3 * x - 2 * y)) / (⅟2 * x) = 20 * x + 8 * y := 
by sorry

end problem1_problem2_problem3_problem4_l15_15936


namespace hexagonal_tiles_in_box_l15_15118

theorem hexagonal_tiles_in_box :
  ∃ a b c : ℕ, a + b + c = 35 ∧ 3 * a + 4 * b + 6 * c = 128 ∧ c = 6 :=
by
  sorry

end hexagonal_tiles_in_box_l15_15118


namespace third_smallest_is_four_probability_l15_15416

noncomputable def probability_third_smallest_is_four : ℚ :=
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 3 2) * (Nat.choose 8 4)
  favorable_ways / total_ways

theorem third_smallest_is_four_probability : 
  probability_third_smallest_is_four = 35 / 132 := 
sorry

end third_smallest_is_four_probability_l15_15416


namespace probability_third_smallest_is_four_l15_15409

/--
Seven distinct integers are picked at random from the set {1, 2, 3, ..., 12}.
The probability that the third smallest number is 4 is 7/33.
-/
theorem probability_third_smallest_is_four : 
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.to_finset in
  ∀ s : Finset ℕ, s ⊆ S ∧ s.card = 7 →
  let event := { s | s.nth_le 2 (by simp [s.card_eq_coe] ; norm_num) = 4 }.to_finset in
  (event.card : ℚ) / (S.choose 7).card = 7 / 33 :=
by
  intros S S_prop event
  sorry

end probability_third_smallest_is_four_l15_15409


namespace prob_exactly_M_laws_expected_laws_included_l15_15719

noncomputable def prob_of_exactly_M_laws (K N M : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - (1 - p)^N
  (Nat.choose K M) * q^M * (1 - q)^(K - M)

noncomputable def expected_num_of_laws (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

-- Part (a): Prove that the probability of exactly M laws being included is as follows
theorem prob_exactly_M_laws (K N M : ℕ) (p : ℝ) :
  prob_of_exactly_M_laws K N M p =
    (Nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - (1 - p)^N)^(K - M)) :=
sorry

-- Part (b): Prove that the expected number of laws included is as follows
theorem expected_laws_included (K N : ℕ) (p : ℝ) :
  expected_num_of_laws K N p =
    K * (1 - (1 - p)^N) :=
sorry

end prob_exactly_M_laws_expected_laws_included_l15_15719


namespace coloring_pairs_of_integers_l15_15343

theorem coloring_pairs_of_integers :
  ∃ n : ℕ, n = 16 ∧
  ∀ (coloring : ℤ × ℤ → bool),
    (∀ a b : ℤ, coloring (a, b) = coloring (-1 - b, a + 1) ∧ coloring (a, b) = coloring (1 - b, a - 1)) →
    (∃ L1 L2 L3 L4 : set (ℤ × ℤ),
      (∀ p ∈ L1, ∀ q ∈ L1, coloring p = coloring q) ∧
      (∀ p ∈ L2, ∀ q ∈ L2, coloring p = coloring q) ∧
      (∀ p ∈ L3, ∀ q ∈ L3, coloring p = coloring q) ∧
      (∀ p ∈ L4, ∀ q ∈ L4, coloring p = coloring q) ∧
      (∀ p ∈ L1, ∀ q ∈ L2, coloring p ≠ coloring q) ∧
      (∀ p ∈ L1, ∀ q ∈ L3, coloring p ≠ coloring q) ∧
      (∀ p ∈ L1, ∀ q ∈ L4, coloring p ≠ coloring q) ∧
      (∀ p ∈ L2, ∀ q ∈ L3, coloring p ≠ coloring q) ∧
      (∀ p ∈ L2, ∀ q ∈ L4, coloring p ≠ coloring q) ∧
      (∀ p ∈ L3, ∀ q ∈ L4, coloring p ≠ coloring q)
  ) ∧
  (∀ (ccount : (ℤ × ℤ → bool) → ℕ), ccount = 16) :=
sorry

end coloring_pairs_of_integers_l15_15343


namespace esther_evening_speed_l15_15180

/-- Esther's average speed in the evening was 30 miles per hour -/
theorem esther_evening_speed : 
  let morning_speed := 45   -- miles per hour
  let total_commuting_time := 1 -- hour
  let morning_distance := 18  -- miles
  let evening_distance := 18  -- miles (same route)
  let time_morning := morning_distance / morning_speed
  let time_evening := total_commuting_time - time_morning
  let evening_speed := evening_distance / time_evening
  evening_speed = 30 := 
by sorry

end esther_evening_speed_l15_15180


namespace amount_A_received_l15_15174

-- Define the conditions
def total_amount : ℕ := 600
def ratio_a : ℕ := 1
def ratio_b : ℕ := 2

-- Define the total parts in the ratio
def total_parts : ℕ := ratio_a + ratio_b

-- Define the value of one part
def value_per_part : ℕ := total_amount / total_parts

-- Define the amount A gets
def amount_A_gets : ℕ := ratio_a * value_per_part

-- Lean statement to prove
theorem amount_A_received : amount_A_gets = 200 := by
  sorry

end amount_A_received_l15_15174


namespace f_100_at_1_l15_15948

-- Define polynomials f_n
noncomputable def f : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| (n + 1) := λ x, ∫ t in (0..x), (n + 1) * f n (t + 1)

-- Condition: f_n(0) = 0 for n ≥ 1
lemma f_n_zero (n : ℕ) (hn : n ≥ 1) : f n 0 = 0 := sorry

-- Condition: d/dx f_{n+1}(x) = (n+1) f_n(x+1) for n ≥ 0
lemma f_recurrence (n : ℕ) : deriv (f (n + 1)) = λ x, (n + 1) * f n (x + 1) := sorry

-- Prove that f_100(1) = 101^99
theorem f_100_at_1 : f 100 1 = 101^99 := sorry

end f_100_at_1_l15_15948


namespace largest_root_l15_15952

theorem largest_root (p q r : ℝ) (h1 : p + q + r = 3) (h2 : p * q + p * r + q * r = -6) (h3 : p * q * r = -8) :
  max (max p q) r = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end largest_root_l15_15952


namespace exists_point_M_l15_15996

noncomputable def find_point_on_line (A B : EuclideanSpace ℝ) (a : ℝ) (l : Set (EuclideanSpace ℝ)) : Prop :=
  ∃ (M : EuclideanSpace ℝ), M ∈ l ∧ dist M A + dist M B = a

-- Conditions
variables (l : Set (EuclideanSpace ℝ)) 
variables (A B : EuclideanSpace ℝ) 
variables (a : ℝ) 

-- Lean 4 statement
theorem exists_point_M (hA : A ∈ lᶜ) (hB : B ∈ lᶜ) (ha : a > 0) :
  find_point_on_line A B a l :=
sorry

end exists_point_M_l15_15996


namespace polygon_interior_angles_l15_15704

theorem polygon_interior_angles (n : ℕ) (h : n - 3 = 6) :
  ∑ (i : Fin n), interior_angle n i = 1260 :=
by
  -- Definition of the problem's conditions
  have h1 : n = 9 := by linarith
      
  -- The formula for the sum of interior angles
  let sum_interior_angles := 180 * (n - 2)
  
  -- Proving that the sum of the interior angles is 1260 degrees
  rw h1
  calc
    180 * (9 - 2) = 1260 : by norm_num
  sorry

end polygon_interior_angles_l15_15704


namespace part1_tangent_line_at_x_eq_1_part2_monotonic_increasing_l15_15272

noncomputable def f (x : ℝ) (a : ℝ) := (1/x + a) * real.log(1 + x)

theorem part1_tangent_line_at_x_eq_1 (x := 1) :
  let a := -1 
  let t : ℝ := real.log 2
  let f := (λ x, (1/x - 1) * real.log (1 + x))
  let tangent_line := λ x y, t * x + y - t 
  (tangent_line 1 (f 1) = 0) :=
by {
  sorry
}

theorem part2_monotonic_increasing (a : ℝ) : 
  (∀ x : ℝ, 0 < x → 0 < f x a) ↔ (a ≥ 1/2) := 
by {
  sorry
}

end part1_tangent_line_at_x_eq_1_part2_monotonic_increasing_l15_15272


namespace solve_inequality_l15_15431

theorem solve_inequality {x : ℝ} :
  (3 / (5 - 3 * x) > 1) ↔ (2/3 < x ∧ x < 5/3) :=
by
  sorry

end solve_inequality_l15_15431


namespace who_plays_piano_l15_15152

theorem who_plays_piano 
  (A : Prop)
  (B : Prop)
  (C : Prop)
  (hA : A = True)
  (hB : B = False)
  (hC : A = False)
  (only_one_true : (A ∧ ¬B ∧ ¬C) ∨ (¬A ∧ B ∧ ¬C) ∨ (¬A ∧ ¬B ∧ C)) : B = True := 
sorry

end who_plays_piano_l15_15152


namespace number_of_ones_in_black_squares_is_odd_l15_15616

-- Define the chessboard dimensions and properties
def chessboard_dim := (10, 14)

def is_filled_with_0_1 (a : ℕ × ℕ → ℕ) : Prop :=
∀ i j, a (i, j) = 0 ∨ a (i, j) = 1

def row_has_odd_ones (a : ℕ × ℕ → ℕ) (i : ℕ) : Prop :=
∑ j in finset.range 14, a (i, j) % 2 = 1

def col_has_odd_ones (a : ℕ × ℕ → ℕ) (j : ℕ) : Prop :=
∑ i in finset.range 10, a (i, j) % 2 = 1

def is_checkerboard_color (i j : ℕ) : bool :=
(i + j) % 2 = 0

-- Main theorem statement
theorem number_of_ones_in_black_squares_is_odd :
  ∃ (a : ℕ × ℕ → ℕ),
    is_filled_with_0_1 a ∧
    (∀ i, i < 10 → row_has_odd_ones a i) ∧
    (∀ j, j < 14 → col_has_odd_ones a j) →
    (∑ i in finset.range 10, 
       ∑ j in finset.range 14, 
         if is_checkerboard_color i j then a (i, j) else 0) % 2 = 1 := 
by
  sorry

end number_of_ones_in_black_squares_is_odd_l15_15616


namespace rectangle_to_square_l15_15546

theorem rectangle_to_square (a b : ℝ) (ha : a = 1) (hb : b = 2) :
  ∃ s : ℝ, s^2 = a * b :=
by
  use Real.sqrt (a * b)
  rw [ha, hb]
  exact Real.sqrt_sq (by norm_num)

end rectangle_to_square_l15_15546


namespace rajan_income_l15_15096

theorem rajan_income : 
  ∀ (x y : ℕ), 
  7 * x - 6 * y = 1000 → 
  6 * x - 5 * y = 1000 → 
  7 * x = 7000 := 
by 
  intros x y h1 h2
  sorry

end rajan_income_l15_15096


namespace ellipse_problem_l15_15233

noncomputable
def ellipse_equation : ℝ × ℝ → Prop :=
  λ (Q : ℝ × ℝ),
    let (x, y) := Q in 
    x = sqrt 5 ∧ y = 4 / 3 ∧
    ∃ (a b : ℝ), a > b ∧ b > 0 ∧
    (x^2 / a^2 + y^2 / b^2 = 1 ∧
    (∃ (F1 F2 : ℝ × ℝ), 
      let (f1x, f1y) := F1, (f2x, f2y) := F2 in
      (Q.1 - f2x) * (f2x - f1x) + (Q.2 - f2y) * (f2y - f1y) = 0) ∧
      (a^2 - b^2 = 5) ∧
      (b^2 / a = 4 / 3)) ∧
    (eq_sqrt 5 (λ f, (a, b))) -- auxiliary proof for non-negative values

theorem ellipse_problem (Q : ℝ × ℝ) :
  ellipse_equation Q →
  (∃ a b : ℝ, a > b ∧ b > 0 ∧
    (a^2 - b^2 = 5) ∧
    (b^2 / a = 4 / 3) ∧ 
    (a = 3 ∧ b = 2)) ∧
  (let C : set (ℝ × ℝ) := {P | (P.1^2 / 9 + P.2^2 / 4 = 1)}
  in
  ∃ (circle : set (ℝ × ℝ)) (M N : ℝ × ℝ), circle = {P | P.1^2 + P.2^2 = 4} ∧
    (M ∈ C ∧ M ∈ circle ∧ N ∈ C ∧ N ∈ circle) ∧
    (maxMN : ℝ, maxMN = 3)) :=
sorry

end ellipse_problem_l15_15233


namespace circles_tangent_and_tangent_line_l15_15298

theorem circles_tangent_and_tangent_line :
  let M : ℝ → ℝ → Prop := λ x y, x^2 + y^2 + 2*x - 4*y + 4 = 0
  let N : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 4*x - 12*y + 4 = 0
  -- Circle centers and radii
  let M_center : ℝ × ℝ := (-1, 2)
  let N_center : ℝ × ℝ := (2, 6)
  let rM : ℝ := 1
  let rN : ℝ := 6
  -- Distance between centers
  let d : ℝ := sqrt ((2 + 1)^2 + (6 - 2)^2)
  -- Equation of the common tangent line
  let tangent_line : ℝ → ℝ → Prop := λ x y, 3*x + 4*y = 0
  in (M_center, N_center, rM, rN, d) = ((-1, 2), (2, 6), 1, 6, 5)
  ∧ tangent_line = (λ x y, 3*x + 4*y = 0) :=
by sorry

end circles_tangent_and_tangent_line_l15_15298


namespace problem_inequality_solution_l15_15624

theorem problem_inequality_solution (x : ℝ) :
  5 ≤ (x - 1) / (3 * x - 7) ∧ (x - 1) / (3 * x - 7) < 10 ↔ (69 / 29) < x ∧ x ≤ (17 / 7) :=
by sorry

end problem_inequality_solution_l15_15624


namespace total_pieces_of_art_l15_15092

variable (A : ℕ) (displayed : ℕ) (sculptures_on_display : ℕ) (not_on_display : ℕ) (paintings_not_on_display : ℕ) (sculptures_not_on_display : ℕ)

-- Constants and conditions from the problem
axiom H1 : displayed = 1 / 3 * A
axiom H2 : sculptures_on_display = 1 / 6 * displayed
axiom H3 : not_on_display = 2 / 3 * A
axiom H4 : paintings_not_on_display = 1 / 3 * not_on_display
axiom H5 : sculptures_not_on_display = 800
axiom H6 : sculptures_not_on_display = 2 / 3 * not_on_display

-- Prove that the total number of pieces of art is 1800
theorem total_pieces_of_art : A = 1800 :=
by
  sorry

end total_pieces_of_art_l15_15092


namespace math_proof_equivalent_l15_15106

theorem math_proof_equivalent :
  (60 + 5 * 12) / (Real.sqrt 180 / 3) ^ 2 = 6 := by
  sorry

end math_proof_equivalent_l15_15106


namespace number_of_female_students_l15_15551

theorem number_of_female_students (total_students sample_size sampled_girls : ℕ) 
    (h1 : total_students = 1600) (h2 : sample_size = 200) (h3 : sampled_girls = 95) : 
    let x := (sampled_girls * total_students) / sample_size in x = 760 := 
by
  have h_proportion : sampled_girls / x = sample_size / total_students := sorry
  have h_intermediate : sampled_girls * total_students = sample_size * x := sorry
  sorry

end number_of_female_students_l15_15551


namespace relationship_among_a_b_c_l15_15169

noncomputable def f : ℝ → ℝ := sorry

def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_monotonically_increasing_on (f : ℝ → ℝ) (I : set ℝ) := ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y
def period_two (f : ℝ → ℝ) := ∀ x, f (x + 2) = f x

variables
  (h1 : is_odd_function f)
  (h2 : period_two f)
  (h3 : is_monotonically_increasing_on f (set.Icc 0 1))
  (a : ℝ) (b : ℝ) (c : ℝ)

theorem relationship_among_a_b_c :
  let a := f 3
  let b := f (real.sqrt 2)
  let c := f 2
  in a > c ∧ c > b :=
by {
  sorry
}

end relationship_among_a_b_c_l15_15169


namespace average_of_three_numbers_l15_15849

theorem average_of_three_numbers (a b c : ℝ)
  (h1 : a + (b + c) / 2 = 65)
  (h2 : b + (a + c) / 2 = 69)
  (h3 : c + (a + b) / 2 = 76) :
  (a + b + c) / 3 = 35 := 
sorry

end average_of_three_numbers_l15_15849


namespace finitely_many_n_with_subset_l15_15403

-- This is the statement for the proof problem provided
theorem finitely_many_n_with_subset :
  ∃ (N : ℕ), ∀ (n : ℕ), n > N → ¬ (∃ (S : Finset ℕ),
    (∀ x ∈ S, 1 ≤ x ∧ x ≤ n) ∧
    (S.card ≥ (Int.floor (Real.sqrt (n : ℝ))).toNat + 1) ∧
    (∀ x y ∈ S, ∃ a b : ℕ, xy = a^b ∧ b ≥ 2)) :=
begin
  sorry,
end

end finitely_many_n_with_subset_l15_15403


namespace PQ_parallel_to_AB_l15_15817

variables (O A B C D E P Q : Type) [CircumscribedQuadrilateral ABCD] 

-- Given conditions as assumptions
variable (h1 : Center O of Circumcircle ABCD ∧ O lies_on AB)
variable (h2 : Reflection E of D across AB)
variable (h3 : Intersection P of AC and DO)
variable (h4 : Intersection Q of BD and CE)

theorem PQ_parallel_to_AB (h1 : Center O of Circumcircle ABCD ∧ O lies_on AB)
                         (h2 : Reflection E of D across AB)
                         (h3 : Intersection P of AC and DO)
                         (h4 : Intersection Q of BD and CE) : PQ ∥ AB :=
sorry


end PQ_parallel_to_AB_l15_15817


namespace find_range_of_m_l15_15648

noncomputable def p (m : ℝ) : Prop := 1 - Real.sqrt 2 < m ∧ m < 1 + Real.sqrt 2
noncomputable def q (m : ℝ) : Prop := 0 < m ∧ m < 4

theorem find_range_of_m (m : ℝ) (hpq : p m ∨ q m) (hnp : ¬ p m) : 1 + Real.sqrt 2 ≤ m ∧ m < 4 :=
sorry

end find_range_of_m_l15_15648


namespace max_value_of_x_times_one_minus_2x_l15_15984

theorem max_value_of_x_times_one_minus_2x : 
  ∀ x : ℝ, 0 < x ∧ x < 1 / 2 → x * (1 - 2 * x) ≤ 1 / 8 :=
by
  intro x 
  intro hx
  sorry

end max_value_of_x_times_one_minus_2x_l15_15984


namespace geometric_seq_ineq_l15_15478

theorem geometric_seq_ineq (a : ℕ → ℝ) (n : ℕ) (h_pos : ∀ n, a n > 0)
  (h1 : 2 * a 1 + 3 * a 2 = 1) 
  (h2 : a 3 ^ 2 = 9 * a 2 * a 6) :
  let c (n : ℕ) := (Real.log (a n) / Real.log 3) ^ 2 in
  ∑ i in Finset.range n, 1 / c (i + 1) < 7 / 4 :=
by
  sorry

end geometric_seq_ineq_l15_15478


namespace prob_exactly_M_laws_included_expected_laws_included_l15_15730

variables (K N M : ℕ) (p : ℝ)

-- Definition of the probabilities as given in the conditions and answers
def prob_no_minister_knows_law : ℝ := (1 - p) ^ N
def prob_law_included : ℝ := 1 - prob_no_minister_knows_law p N

-- Part (a)
theorem prob_exactly_M_laws_included :
  (nat.choose K M) * (prob_law_included p N) ^ M * (prob_no_minister_knows_law p N) ^ (K - M) = 
  (nat.choose K M) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) :=
by
  sorry

-- Part (b)
theorem expected_laws_included :
  K * (prob_law_included p N) = K * (1 - (1 - p) ^ N) :=
by
  sorry

end prob_exactly_M_laws_included_expected_laws_included_l15_15730


namespace total_experimental_combinations_l15_15214

-- Define the number of elements of type A and type B
def num_elements_A := 6
def num_elements_B := 4

-- Define the conditions for selection
def condition_A (selected_A : ℕ) : Prop := selected_A ≤ num_elements_A ∧ selected_A % 2 = 0
def condition_B (selected_B : ℕ) : Prop := selected_B ≤ num_elements_B ∧ selected_B ≥ 2

-- Define the main theorem stating the total number of possible experimental setups
theorem total_experimental_combinations (combinations : ℕ) :
  combinations = 353 :=
begin
  -- Here we would prove the statement using the conditions and computation
  sorry
end

end total_experimental_combinations_l15_15214


namespace median_and_mode_of_data_set_l15_15835

/-
Given the data set defined, prove that:
- The median of the data set is 0
- The mode of the data set is 3
-/

def data_set := [-5, 3, -8, -2, 9, 0, 3]

theorem median_and_mode_of_data_set : 
  List.median (data_set.sort (<=)) = 0 ∧ List.mode data_set = 3 :=
  by
  sorry

end median_and_mode_of_data_set_l15_15835


namespace find_number_l15_15073

theorem find_number (x : ℕ) (h : x * 625 = 584638125) : x = 935420 :=
sorry

end find_number_l15_15073


namespace probability_third_smallest_is_four_l15_15410

/--
Seven distinct integers are picked at random from the set {1, 2, 3, ..., 12}.
The probability that the third smallest number is 4 is 7/33.
-/
theorem probability_third_smallest_is_four : 
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.to_finset in
  ∀ s : Finset ℕ, s ⊆ S ∧ s.card = 7 →
  let event := { s | s.nth_le 2 (by simp [s.card_eq_coe] ; norm_num) = 4 }.to_finset in
  (event.card : ℚ) / (S.choose 7).card = 7 / 33 :=
by
  intros S S_prop event
  sorry

end probability_third_smallest_is_four_l15_15410


namespace f_properties_a0_f_properties_an0_l15_15614

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if a = 0 then 2 - (1 / 2) * Real.log (Real.abs x)
else if x > -a then x^2 + a - (1 / 2) * Real.log x 
else -(x^2) - x - (1 / 2) * Real.log x

theorem f_properties_a0 (x : ℝ) : 
  0 < x → f 0 x = 2 - (1 / 2) * Real.log (Real.abs x) := 
by sorry

theorem f_properties_an0 (a x : ℝ) (Δ : ℝ) : 
  a ≠ 0 →
  Δ = 2 - 16 * a →
  (Δ > 0 → -2 ≤ a ∧ a < 0 → (-1/2 ≤ a ∧ a < 0 → ∀ x, f a x) ∧ (f a x < ∀ x)) ∧ 
  (Δ < 0 → a < -2 → ∀ x3 x4, x3 = (-a - Real.sqrt (a^2 - 4)) / 4 ∧ x4 = (-a + Real.sqrt (a^2 - 4)) / 4 ∧ x3 < x4 ∧ x4 < -a → ∀ x, f a x) →
  (Δ ≤ 0 → -2 ≤ a ∧ a ≤ (-Real.sqrt 2) / 2 → ∀ x, f a x) →
  (Δ ≥ 0 → -a ≤ (-Real.sqrt (a^2 + 4)) / 4 → a ≤ -Real.sqrt 2 / 2 → ∀ x, f a x) →
  (f' x ≠ 0 → ∀ x, f a x)
:=
by sorry

end f_properties_a0_f_properties_an0_l15_15614


namespace binary_to_decimal_eq_l15_15947

theorem binary_to_decimal_eq :
  (1 * 2^7 + 1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 205 :=
by
  sorry

end binary_to_decimal_eq_l15_15947


namespace ratio_a7_b7_l15_15661

variable (a b : ℕ → ℝ)
variable (S T : ℕ → ℝ)

-- Given conditions
axiom sum_S : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * a 2) -- Formula for sum of arithmetic series
axiom sum_T : ∀ n, T n = (n / 2) * (2 * b 1 + (n - 1) * b 2) -- Formula for sum of arithmetic series
axiom ratio_ST : ∀ n, S n / T n = (2 * n + 1) / (n + 3)

-- Prove the ratio of seventh terms
theorem ratio_a7_b7 : a 7 / b 7 = 27 / 16 :=
by
  sorry

end ratio_a7_b7_l15_15661


namespace part1_part2_l15_15676

open Real

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| - 1

-- Define the function g for the second part
def g (x : ℝ) : ℝ := |x - 2| + |x + 3|

theorem part1 (m : ℝ) : (∀ x, f x m ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5) → m = 2 :=
  by sorry

theorem part2 (t x: ℝ) (h: ∀ x: ℝ, f x 2 + f (x + 5) 2 ≥ t - 2) : t ≤ 5 :=
  by sorry

end part1_part2_l15_15676


namespace tree_pit_min_distance_l15_15107

theorem tree_pit_min_distance :
  let S (x : ℕ) := 10 * (abs (1 - x) + abs (2 - x) + abs (3 - x) + abs (4 - x) +
                          abs (5 - x) + abs (6 - x) + abs (7 - x) + abs (8 - x) +
                          abs (9 - x) + abs (10 - x) + abs (11 - x) + abs (12 - x) +
                          abs (13 - x) + abs (14 - x) + abs (15 - x) + abs (16 - x) +
                          abs (17 - x) + abs (18 - x) + abs (19 - x) + abs (20 - x))
  in S 10 = · ∧ S 11 = · ∧ (∀ x : ℕ, S x > S 10 ∨ S x > S 11) :=
sorry

end tree_pit_min_distance_l15_15107


namespace find_a_plus_b_l15_15249

theorem find_a_plus_b (a b : ℝ) :
  (∀ x y : ℝ, (ax + y + 1 = 0) ∧ (2x - by - 1 = 0) → (x = 1 → y = 1)) → (a + b = -1) :=
by
  intros
  sorry

end find_a_plus_b_l15_15249


namespace not_p_sufficient_not_necessary_for_not_q_l15_15321

theorem not_p_sufficient_not_necessary_for_not_q (p q : Prop) (h1 : q → p) (h2 : ¬ (p → q)) : 
  (¬p → ¬ q) ∧ ¬ (¬ q → ¬ p) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l15_15321


namespace solve_trigonometric_equation_l15_15473

open Set Real

theorem solve_trigonometric_equation :
  {x : ℝ | 0 < x ∧ x < π ∧ 2 * cos (x - π / 4) = sqrt 2} = {π / 2} :=
by
  sorry

end solve_trigonometric_equation_l15_15473


namespace collinear_I_H_B_H_C_J_l15_15767

-- Define the geometric entities and their properties
variables {A B C H_A H_B H_C I J : Type}
           [is_triangle ABC : Triangle A B C] 
           [is_foot_of_altitude H_A A B C]
           [is_foot_of_altitude H_B B A C]
           [is_foot_of_altitude H_C C A B]
           [is_reflection I H_A (line_through A B)]
           [is_reflection J H_A (line_through A C)]

theorem collinear_I_H_B_H_C_J 
  (hI : is_reflection I H_A (line_through A B))
  (hJ : is_reflection J H_A (line_through A C))
  (hAB : is_foot_of_altitude H_A A B C) 
  (hBC : is_foot_of_altitude H_B B A C)
  (hCA : is_foot_of_altitude H_C C A B) :
  collinear I H_B H_C J := 
sorry

end collinear_I_H_B_H_C_J_l15_15767


namespace least_number_divisible_by_five_primes_l15_15058

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, (n > 0) ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → p ∣ n) ∧ n = 2310 :=
begin
  sorry
end

end least_number_divisible_by_five_primes_l15_15058


namespace football_exercise_l15_15454

theorem football_exercise (runs : ℕ) (calories_per_stair : ℕ) (total_calories_burned : ℕ) (calories_burned : ℕ) : 
  (runs = 40) → 
  (calories_per_stair = 2) → 
  (total_calories_burned = 5120) → 
  (calories_burned = 4 * runs): 
  let S := total_calories_burned / calories_burned in
  S = 32 :=
by
  intros h_runs h_calories_per_stair h_total_calories_burned h_calories_burned
  let S := total_calories_burned / calories_burned
  have h1 : total_calories_burned = 160 * S,
  {
    rw [h_total_calories_burned, h_calories_burned, Nat.mul_comm],
  }
  have h2 : S = 32,
  {
    rw [←h1, Nat.div_self, Nat.div_eq_of_lt, one_mul],
    norm_num,
    norm_num,
  }
  exact h2

end football_exercise_l15_15454


namespace amount_spent_on_apples_l15_15773

theorem amount_spent_on_apples :
  ∀ (total berries peaches apples : ℝ),
    total = 34.72 →
    berries = 11.08 →
    peaches = 9.31 →
    apples = total - (berries + peaches) →
    apples = 14.33 :=
by
  intros total berries peaches apples h_total h_berries h_peaches h_apples
  rw [h_total, h_berries, h_peaches] at h_apples
  norm_num at h_apples
  exact h_apples

-- This proof statement would successfully compile and by using the appropriate Lean tactics,
-- we can ensure that the mathematically equivalent proof problem is correctly expressed. 

end amount_spent_on_apples_l15_15773


namespace gcf_360_270_lcm_360_270_l15_15501

def prime_factors_360 := [(2, 3), (3, 2), (5, 1)]
def prime_factors_270 := [(2, 1), (3, 3), (5, 1)]

def GCF (a b: ℕ) : ℕ := 2^1 * 3^2 * 5^1
def LCM (a b: ℕ) : ℕ := 2^3 * 3^3 * 5^1

-- Theorem: The GCF of 360 and 270 is 90
theorem gcf_360_270 : GCF 360 270 = 90 := by
  sorry

-- Theorem: The LCM of 360 and 270 is 1080
theorem lcm_360_270 : LCM 360 270 = 1080 := by
  sorry

end gcf_360_270_lcm_360_270_l15_15501


namespace least_positive_number_divisible_by_five_primes_l15_15052

theorem least_positive_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ 
    (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) :=
by
  use 2310
  split
  { exact by norm_num }
  split
  { intros p hp, fin_cases hp; norm_num }
  { intros m hm, apply nat.le_of_dvd; norm_num }
  sorry

end least_positive_number_divisible_by_five_primes_l15_15052


namespace inequality_l15_15232

noncomputable def inequality_proof (n : ℕ) (a b : Fin n → ℝ) (ha : ∀ i, 0 ≤ a i) (hb : ∀ i, 0 ≤ b i) : Prop :=
  (∏ i, a i)^(1/(n:ℝ)) + (∏ i, b i)^(1/(n:ℝ)) ≤ (∏ i, (a i + b i))^(1/(n:ℝ))

theorem inequality (n : ℕ) (a b : Fin n → ℝ) (ha : ∀ i, 0 ≤ a i) (hb : ∀ i, 0 ≤ b i) : 
  inequality_proof n a b ha hb := 
  sorry

end inequality_l15_15232


namespace oak_total_after_planting_l15_15847

-- Let oak_current represent the current number of oak trees in the park.
def oak_current : ℕ := 9

-- Let oak_new represent the number of new oak trees being planted.
def oak_new : ℕ := 2

-- The problem is to prove the total number of oak trees after planting equals 11
theorem oak_total_after_planting : oak_current + oak_new = 11 :=
by
  sorry

end oak_total_after_planting_l15_15847


namespace tangent_line_at_1_range_of_a_iff_monotonic_increasing_l15_15262

noncomputable def f (x a : ℝ) : ℝ := (1 / x + a) * real.log (1 + x)

theorem tangent_line_at_1 (a : ℝ) (h : a = -1) :
  let f_at := (1 / (1:ℝ) + a) * real.log (1 + 1)
  let f_prime := -real.log 2
  let tangent_line := λ x : ℝ, f_prime * (x - 1) + f_at in
  tangent_line = λ x, -real.log 2 * x + real.log 2 :=
by {
  -- Proof to be provided
  sorry
}

theorem range_of_a_iff_monotonic_increasing :
  ∀ x : ℝ, x > 0 → monotonic_increasing (λ a : ℝ, (1/x + a) * real.log (1 + x)) ↔ a ∈ set.Ici (1/2) :=
by {
  -- Proof to be provided
  sorry
}

end tangent_line_at_1_range_of_a_iff_monotonic_increasing_l15_15262


namespace math_proof_problem_l15_15874

-- Definitions
def PropA : Prop := ¬ (∀ n : ℤ, (3 ∣ n → ¬ (n % 2 = 1)))
def PropB : Prop := ¬ (¬ (∃ x : ℝ, x^2 + x + 1 ≥ 0))
def PropC : Prop := ∀ (α β : ℝ) (k : ℤ), α = k * Real.pi + β ↔ Real.tan α = Real.tan β
def PropD : Prop := ∀ (a b : ℝ), a ≠ 0 → a * b ≠ 0 → b ≠ 0

def correct_options : Prop := PropA ∧ PropC ∧ ¬PropB ∧ PropD

-- The theorem to be proven
theorem math_proof_problem : correct_options :=
by
  sorry

end math_proof_problem_l15_15874


namespace smallest_five_digit_palindrome_divisible_by_6_l15_15503

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

def is_divisible_by_2_and_3 (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n % 3 = 0

theorem smallest_five_digit_palindrome_divisible_by_6 : ∃ n, is_palindrome n ∧ is_divisible_by_2_and_3 n ∧ (∀ m, is_palindrome m ∧ is_divisible_by_2_and_3 m ∧ 10000 ≤ m ∧ m < 100000 → n ≤ m) :=
  ∃ n, is_palindrome n ∧ is_divisible_by_2_and_3 n ∧ 10000 ≤ n ∧ n < 100000 ∧ (∀ m, is_palindrome m ∧ is_divisible_by_2_and_3 m ∧ 10000 ≤ m ∧ m < 100000 → n ≤ m ∨ m = 21312) :=
sorry

end smallest_five_digit_palindrome_divisible_by_6_l15_15503


namespace largest_integer_a_can_be_less_than_l15_15876

theorem largest_integer_a_can_be_less_than (a b : ℕ) (h1 : 9 < a) (h2 : 19 < b) (h3 : b < 31) (h4 : a / b = 2 / 3) :
  a < 21 :=
sorry

end largest_integer_a_can_be_less_than_l15_15876


namespace solution_set_for_inequality_l15_15610

variable (f : ℝ → ℝ)
variable (h_deriv_pos : ∀ x, f x + deriv f x > 1)
variable (h_f_zero : f 0 = 4)

theorem solution_set_for_inequality :
  {x : ℝ | exp x * f x > exp x + 3} = {x : ℝ | 0 < x} := 
sorry

end solution_set_for_inequality_l15_15610


namespace least_positive_whole_number_divisible_by_five_primes_l15_15014

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l15_15014


namespace exists_x_satisfying_fx_eq_3_l15_15769

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 2 then 2 ^ x else Real.log 2 x

theorem exists_x_satisfying_fx_eq_3 :
  ∃ x : ℝ, (f x = 3) ∧ (x = Real.log 2 3 ∨ x = 8) :=
by
  sorry

end exists_x_satisfying_fx_eq_3_l15_15769


namespace prob_exactly_M_laws_expected_laws_included_l15_15717

noncomputable def prob_of_exactly_M_laws (K N M : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - (1 - p)^N
  (Nat.choose K M) * q^M * (1 - q)^(K - M)

noncomputable def expected_num_of_laws (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

-- Part (a): Prove that the probability of exactly M laws being included is as follows
theorem prob_exactly_M_laws (K N M : ℕ) (p : ℝ) :
  prob_of_exactly_M_laws K N M p =
    (Nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - (1 - p)^N)^(K - M)) :=
sorry

-- Part (b): Prove that the expected number of laws included is as follows
theorem expected_laws_included (K N : ℕ) (p : ℝ) :
  expected_num_of_laws K N p =
    K * (1 - (1 - p)^N) :=
sorry

end prob_exactly_M_laws_expected_laws_included_l15_15717


namespace sum_powers_of_i_l15_15940

-- Define the conditions
def i : ℂ := Complex.I -- Complex.I is the imaginary unit in ℂ (ℂ is the set of complex numbers)

-- The theorem statement
theorem sum_powers_of_i : (i + i^2 + i^3 + i^4) * 150 + 1 + i + i^2 + i^3 = 0 := by
  sorry

end sum_powers_of_i_l15_15940


namespace recurring_decimal_as_fraction_l15_15088

theorem recurring_decimal_as_fraction : 
  (let a := 0.\overline{56} in 
    100 * a = 56.\overline{56}) → 
  0.\overline{56} = 56 / 99 := 
sorry

end recurring_decimal_as_fraction_l15_15088


namespace tenth_day_of_month_is_monday_l15_15331

theorem tenth_day_of_month_is_monday (Sundays_on_even_dates : ℕ → Prop)
  (h1: Sundays_on_even_dates 2)
  (h2: Sundays_on_even_dates 16)
  (h3: Sundays_on_even_dates 30) :
  ∃ k : ℕ, 10 = k + 2 + 7 * 1 ∧ k.succ.succ.succ.succ.succ.succ.succ.succ.succ.succ = 1 :=
by sorry

end tenth_day_of_month_is_monday_l15_15331


namespace incenter_inequality_l15_15380

theorem incenter_inequality (A B C M A1 : Point) (r : ℝ) :
  M ∈ interior(A, B, C) →
  A1 ∈ circumcircle(A, B, C) →
  line(A, M) ∩ circumcircle(A, B, C) = {A1} →
  let BM := dist B M
  let CM := dist C M
  let A1M := dist A1 M
  let ris := inradius(A, B, C)
  ris = r →
  (BM * CM) / A1M ≥ 2 * r :=
by
  sorry

end incenter_inequality_l15_15380


namespace jerry_initial_action_figures_l15_15749

theorem jerry_initial_action_figures 
(A : ℕ) 
(h1 : ∀ A, A + 7 = 9 + 3)
: A = 5 :=
by
  sorry

end jerry_initial_action_figures_l15_15749


namespace inequality_solution_l15_15429

theorem inequality_solution (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  (3 - 2 / (3 * x + 4) < 5) ↔ x ∈ set.Ioo (-5/3 : ℝ) (-4/3) ∪ set.Ioi (-4/3) :=
by
  sorry

end inequality_solution_l15_15429


namespace odd_sided_polygon_not_divisible_l15_15508

def is_regular_polygon (n : ℕ) : Prop := 
  n ≥ 3 ∧ ∀ (i j : ℕ), i ≠ j → dist (vertex i) (vertex j) = dist (vertex 0) (vertex 1)

def diameter (n : ℕ) : ℝ := 
  max (dist (vertex 0) (vertex (n / 2))) (dist (vertex 0) (vertex (n / 2 + 1)))

-- Lean statement for the proof problem:
theorem odd_sided_polygon_not_divisible 
  (n : ℕ) (hne : n % 2 = 1) :
  ∀ (P : ℕ → Prop), is_regular_polygon n →
  ∀ (A B : set ℕ), A ∪ B = set.univ →
  diameter n ≤ max (diameter (P A)) (diameter (P B)) := sorry

end odd_sided_polygon_not_divisible_l15_15508


namespace arithmetic_progression_21st_term_l15_15626

theorem arithmetic_progression_21st_term :
  let a_1 := 3
  let d := 5
  let n := 21
  let a_n := a_1 + (n - 1) * d
  in a_n = 103 :=
by
  let a_1 := 3
  let d := 5
  let n := 21
  let a_n := a_1 + (n - 1) * d
  show a_n = 103
  sorry

end arithmetic_progression_21st_term_l15_15626


namespace Mike_age_l15_15774

-- We define the ages of Mike and Barbara
variables (M B : ℕ)

-- Conditions extracted from the problem
axiom h1 : B = M / 2
axiom h2 : M - B = 8

-- The theorem to prove
theorem Mike_age : M = 16 :=
by sorry

end Mike_age_l15_15774


namespace probability_of_red_black_or_white_l15_15117

def numberOfBalls := 12
def redBalls := 5
def blackBalls := 4
def whiteBalls := 2
def greenBalls := 1

def favorableOutcomes : Nat := redBalls + blackBalls + whiteBalls
def totalOutcomes : Nat := numberOfBalls

theorem probability_of_red_black_or_white :
  (favorableOutcomes : ℚ) / (totalOutcomes : ℚ) = 11 / 12 :=
by
  sorry

end probability_of_red_black_or_white_l15_15117


namespace silver_coin_worth_l15_15905

theorem silver_coin_worth :
  ∀ (g : ℕ) (S : ℕ) (n_gold n_silver cash : ℕ), 
  g = 50 →
  n_gold = 3 →
  n_silver = 5 →
  cash = 30 →
  n_gold * g + n_silver * S + cash = 305 →
  S = 25 :=
by
  intros g S n_gold n_silver cash
  intros hg hng hnsi hcash htotal
  sorry

end silver_coin_worth_l15_15905


namespace totalCost_l15_15775
-- Importing the necessary library

-- Defining the conditions
def numberOfHotDogs : Nat := 6
def costPerHotDog : Nat := 50

-- Proving the total cost
theorem totalCost : numberOfHotDogs * costPerHotDog = 300 := by
  sorry

end totalCost_l15_15775


namespace arithmetic_sequence_sum_l15_15843

/-- The second and twentieth terms of an arithmetic sequence are 3 and 39, respectively. 
    Prove that the sum of the first 50 terms of this sequence is 2500. -/
theorem arithmetic_sequence_sum (a2 a20 : ℤ) (h1 : a2 = 3) (h2 : a20 = 39) : 
  ∃ S₅₀, S₅₀ = 2500 :=
begin
  -- Definitions and conditions
  let d := 2,
  let a1 := 1,
  let S₅₀ := 25 * (2 * a1 + (50 - 1) * d),
  use S₅₀,

  -- Sum calculation
  have h3 : S₅₀ = 2500,
  { calc
    S₅₀ = 25 * (2 * 1 + 49 * 2) : by refl
    ... = 25 * (2 + 98) : by refl
    ... = 25 * 100 : by refl
    ... = 2500 : by refl },

  exact h3,
end

end arithmetic_sequence_sum_l15_15843


namespace sin_neg_iff_third_or_fourth_l15_15103

-- Define the angle and the sine condition
variable {α : ℝ}

-- The main statement stating that sin(α) < 0 is a necessary but not sufficient condition for α being in the third or fourth quadrant
theorem sin_neg_iff_third_or_fourth :
  (sin α < 0) ↔ ((∃ (k : ℤ), α = (2 * k + 1) * π + / 2) ∨ (∃ (k : ℤ), α = 2 * k * π + 3 * π / 2)) :=
sorry

end sin_neg_iff_third_or_fourth_l15_15103


namespace cosine_inequality_solution_l15_15192

theorem cosine_inequality_solution :
  ∀ (y : ℝ), (0 ≤ y ∧ y ≤ π) → (∀ (x : ℝ), (0 ≤ x ∧ x ≤ π) → (cos (x + y) ≥ cos x + cos y - 1)) → (y = 0 ∨ y = π) :=
by
  intros y hy hxy
  sorry

end cosine_inequality_solution_l15_15192


namespace range_of_m_l15_15650

noncomputable def problem (x m : ℝ) (p q : Prop) : Prop :=
  (¬ p → ¬ q) ∧ (¬ q → ¬ p → False) ∧ (p ↔ |1 - (x - 1) / 3| ≤ 2) ∧ 
  (q ↔ x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0)

theorem range_of_m (m : ℝ) (x : ℝ) (p q : Prop) 
  (h : problem x m p q) : m ≥ 9 :=
sorry

end range_of_m_l15_15650


namespace find_x_in_diagram_l15_15348

theorem find_x_in_diagram
  (AB CD : Line)
  (A B C D X : Point)
  (angle_AXB : ∠ A X B = 75)
  (angle_BXC : ∠ B X C = 40)
  (angle_CYX : ∠ C Y X = 110) :
  ∠ Y X Z = 45 :=
by
  sorry

end find_x_in_diagram_l15_15348


namespace find_temperature_on_friday_l15_15521

variable (M T W Th F : ℕ)

def problem_conditions : Prop :=
  (M + T + W + Th) / 4 = 48 ∧
  (T + W + Th + F) / 4 = 46 ∧
  M = 44

theorem find_temperature_on_friday (h : problem_conditions M T W Th F) : F = 36 := by
  sorry

end find_temperature_on_friday_l15_15521


namespace common_chord_length_l15_15496

-- Define the radii of the circles
def radius : ℝ := 12

-- Define the distance between centers of the circles
def center_dist : ℝ := 8

-- The theorem statement proving the length of the chord
theorem common_chord_length (radius : ℝ) (center_dist : ℝ) (h_radius : radius = 12) (h_center_dist : center_dist = 8) : 
  ∃ (length_of_chord : ℝ), length_of_chord = 16 * real.sqrt 2 :=
sorry

end common_chord_length_l15_15496


namespace max_chessboards_no_overlap_l15_15646

theorem max_chessboards_no_overlap : 
  ∃ (h : ℕ), (h ≤ 16) ∧ (∀ (B₁ B₂ : list (fin 64)), B₁ ≠ B₂ → 
    ∀ n : ℕ, (1 ≤ n) ∧ (n ≤ 64) → B₁.headI = n → B₂.headI ≠ n) := 
begin
  -- proof steps will be added here
  sorry  
end

end max_chessboards_no_overlap_l15_15646


namespace inequality_proof_l15_15373

theorem inequality_proof (a b c A α : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (h_sum : a + b + c = A) (hA : A ≤ 1) (hα : α > 0) :
  ( (1 / a - a) ^ α + (1 / b - b) ^ α + (1 / c - c) ^ α ) ≥ 3 * ( (3 / A) - (A / 3) ) ^ α :=
by
  sorry

end inequality_proof_l15_15373


namespace carl_additional_marbles_l15_15938

def initial_marbles := 12
def lost_marbles := initial_marbles / 2
def additional_marbles_from_mom := 25
def marbles_in_jar_after_game := 41

theorem carl_additional_marbles :
  (marbles_in_jar_after_game - additional_marbles_from_mom) + lost_marbles - initial_marbles = 10 :=
by
  sorry

end carl_additional_marbles_l15_15938


namespace prob_exactly_M_laws_in_concept_l15_15715

theorem prob_exactly_M_laws_in_concept 
  (K N M : ℕ) (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  let q := 1 - (1 - p)^N in
  (nat.choose K M) * q^M * (1 - q)^(K - M) = 
  (nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M) :=
by {
  let q := 1 - (1 - p)^N,
  have hq_def : q = 1 - (1 - p)^N := rfl,
  rw [hq_def],
  sorry
}

end prob_exactly_M_laws_in_concept_l15_15715


namespace function_properties_l15_15782

theorem function_properties (p q : ℝ) (f : ℝ → ℝ) 
  (h : ∀ x, f x = x^2 + p * x + q) :
  |f 1| < 1/2 ∧ |f 2| < 1/2 ∧ |f 3| < 1/2 → False :=
begin
  sorry
end

end function_properties_l15_15782


namespace annual_income_correct_l15_15517

-- Investment amount
def investment_amount : ℝ := 6800

-- Dividend rate
def dividend_rate : ℝ := 0.10

-- Market price per share
def price_per_share : ℝ := 136

-- Face value of the stock
def face_value : ℝ := 100

-- Number of shares purchasable
def number_of_shares : ℝ := investment_amount / price_per_share

-- Annual income derived from dividends
def annual_income : ℝ := dividend_rate * face_value * number_of_shares

-- Theorem: Prove the annual income is $500
theorem annual_income_correct : annual_income = 500 := by
  sorry

end annual_income_correct_l15_15517


namespace solution_proof_l15_15252

-- Define the circle C using its equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 6 * x - 4 * y + 8 = 0

-- Center of the circle C
def center_C : ℝ × ℝ := (-3, 2)

-- Radius of the circle C
def radius_C : ℝ := Real.sqrt 5

-- Equation of the line
def line_eq (x y : ℝ) : Prop := 3 * x - 4 * y + 7 = 0

-- Point to check
def point_check : ℝ × ℝ := (2, -3)

theorem solution_proof :
  let C := circle_eq;
  let line := line_eq;
  let center := center_C;
  let radius := radius_C;
  let area := radius ^ 2 * Real.pi;
  let point := point_check;
  (C center.1 center.2 ∧
   area = 5 * Real.pi ∧ 
   ∃ x y, C x y ∧ (line x y) ∧
   ¬ (Real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2) < radius)) :=
by
  sorry

end solution_proof_l15_15252


namespace least_positive_whole_number_divisible_by_five_primes_l15_15012

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l15_15012


namespace smallest_odd_four_digit_number_l15_15127

noncomputable def four_digit_number (a b c : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + b

theorem smallest_odd_four_digit_number : 
  ∃ (a b c : ℕ), c = a + 6 ∧ four_digit_number a b c % 9 = 8 ∧ 
  ∀ (a' b' c' : ℕ), c' = a' + 6 → 
                    four_digit_number a' b' c' % 9 = 8 → 
                    four_digit_number a' b' c' % 2 = 1 → 
                    four_digit_number a b c ≤ four_digit_number a' b' c' → 
                    four_digit_number a b c = 1979 :=
begin
  sorry
end

end smallest_odd_four_digit_number_l15_15127


namespace triangle_shortest_side_l15_15840

theorem triangle_shortest_side (x y z : ℝ) (h : x / y = 1 / 2) (h1 : x / z = 1 / 3) (hyp : x = 6) : z = 3 :=
sorry

end triangle_shortest_side_l15_15840


namespace min_num_cards_on_table_l15_15483

open Nat

theorem min_num_cards_on_table 
  (n a b c d e : ℕ) 
  (h_prime_a : Prime a)
  (h_prime_b : Prime b)
  (h_prime_c : Prime c)
  (h_prime_d : Prime d)
  (h_prime_e : Prime e)
  (h_mult_5 : (n - a) % 5 = 0)
  (h_mult_3 : (n - a - b) % 3 = 0)
  (h_mult_2 : (n - a - b - c) % 2 = 0)
  (h_eq_e : n - a - b - c - d = e)
  (h_order : a > b ∧ b > c ∧ c > d ∧ d > e) :
  n = 63 :=
sorry

end min_num_cards_on_table_l15_15483


namespace least_positive_number_divisible_by_five_smallest_primes_l15_15026

theorem least_positive_number_divisible_by_five_smallest_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  Nat.min (p1 * p2 * p3 * p4 * p5) := 
  2310 :=
by
  sorry

end least_positive_number_divisible_by_five_smallest_primes_l15_15026


namespace diagonal_BD_l15_15487

variables {A B C D : Point}
variables {AB BC BE : ℝ}
variables {parallelogram : ABCD A B C D}

-- Conditions
def side_AB : AB = 3 := sorry
def side_BC : BC = 5 := sorry
def intersection_BE : BE = 9 := sorry

-- Goal 
theorem diagonal_BD : ∀ (BD : ℝ), BD = 34 / 9 :=
by sorry

end diagonal_BD_l15_15487


namespace smallest_angle_of_triangle_l15_15471

theorem smallest_angle_of_triangle :
  ∀ a b c : ℝ, a = 2 * Real.sqrt 10 → b = 3 * Real.sqrt 5 → c = 5 → 
  ∃ α β γ : ℝ, α + β + γ = π ∧ α = 45 * (π / 180) ∧ (a = c → α < β ∧ α < γ) ∧ (b = c → β < α ∧ β < γ) ∧ (c = a → γ < α ∧ γ < β) → 
  α = 45 * (π / 180) := 
sorry

end smallest_angle_of_triangle_l15_15471


namespace simplify_fraction_l15_15795

theorem simplify_fraction :
  (2 / (3 + Real.sqrt 5)) * (2 / (3 - Real.sqrt 5)) = 1 := by
  sorry

end simplify_fraction_l15_15795


namespace max_value_3x_plus_4y_l15_15219

noncomputable def polar_curve_to_parametric (θ : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

theorem max_value_3x_plus_4y {θ : ℝ} :
  let (x, y) := polar_curve_to_parametric θ in 
  3 * x + 4 * y ≤ 17 :=
by 
  intro θ
  let (x, y) := polar_curve_to_parametric θ
  have h : 3 * x + 4 * y = 7 + 6 * Real.cos θ + 8 * Real.sin θ := sorry
  have trig_bound : 6 * Real.cos θ + 8 * Real.sin θ ≤ 10 := sorry
  calc 
    3 * x + 4 * y 
      = 7 + 6 * Real.cos θ + 8 * Real.sin θ : by exact h
  ... ≤ 7 + 10 : by linarith [trig_bound]
  ... = 17 : by norm_num

end max_value_3x_plus_4y_l15_15219


namespace trapezoid_areas_l15_15805

theorem trapezoid_areas (a b h : ℝ) (H_area : a + b = 80) (H_ratio : b = 3 * a) :
  let median := (a + b) / 2 in
  let height := h / 2 in
  let A1 := (1 / 2) * (a + median) * height in
  let A2 := (1 / 2) * (median + b) * height in
  A1 = 30 ∧ A2 = 50 :=
sorry

end trapezoid_areas_l15_15805


namespace leak_empties_cistern_in_84_hours_l15_15123

def cistern_fill_time_with_leak (R L : ℝ) : Prop :=
  R - L = 1 / 14

def leak_rate (R L : ℝ) : Prop :=
  L = 1 / 12 - 1 / 14

def time_to_empty (L : ℝ) : ℝ :=
  1 / L

theorem leak_empties_cistern_in_84_hours :
  ∀ (R L : ℝ), (R = 1 / 12) → cistern_fill_time_with_leak R L → leak_rate R L → time_to_empty L = 84 := by
  intros R L hR hCisternFill hLeakRate
  sorry

end leak_empties_cistern_in_84_hours_l15_15123


namespace clock_rings_in_a_day_l15_15577

theorem clock_rings_in_a_day (intervals : ℕ) (hours_in_a_day : ℕ) (time_between_rings : ℕ) : 
  intervals = hours_in_a_day / time_between_rings + 1 → intervals = 7 :=
sorry

end clock_rings_in_a_day_l15_15577


namespace solve_for_x_l15_15318

theorem solve_for_x (x y : ℚ) (h1 : x - y = 8) (h2 : x + 2 * y = 10) : x = 26 / 3 := by
  sorry

end solve_for_x_l15_15318


namespace complement_set_l15_15694

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- Define the complement of M in U
def complement_M_in_U : Set ℝ := {x | x < -2 ∨ x > 2}

-- The mathematical proof to be stated
theorem complement_set :
  U \ M = complement_M_in_U := sorry

end complement_set_l15_15694


namespace geometric_sequence_properties_l15_15217

theorem geometric_sequence_properties 
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 4) :
  (∀ n, a n = 2^(n - 1)) ∧ (S 6 = 63) := 
by 
  sorry

end geometric_sequence_properties_l15_15217


namespace map_length_l15_15576

theorem map_length 
  (width : ℝ) (area : ℝ) 
  (h_width : width = 10) (h_area : area = 20) : 
  ∃ length : ℝ, area = width * length ∧ length = 2 :=
by 
  sorry

end map_length_l15_15576


namespace least_number_divisible_by_five_primes_l15_15065

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, (n > 0) ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → p ∣ n) ∧ n = 2310 :=
begin
  sorry
end

end least_number_divisible_by_five_primes_l15_15065


namespace probability_third_smallest_is_4_l15_15412

theorem probability_third_smallest_is_4 :
  (∃ (integers : Finset ℕ), integers.card = 7 ∧ integers ⊆ (Finset.range 13).erase 0 ∧ 
  ∃ (S : Finset ℕ), S = (Finset.filter (λ x, x < 4) integers) ∧ S.card = 2 ∧ 
  ∃ (T : Finset ℕ), T = (Finset.filter (λ x, 4 < x) integers) ∧ T.card = 5) → 
  let total_ways := Nat.choose 12 7 in
  let favorable_ways := (Nat.choose 3 2) * (Nat.choose 8 5) in
  (favorable_ways) / total_ways.toReal = 7 / 33 :=
by sorry

end probability_third_smallest_is_4_l15_15412


namespace cars_per_salesperson_per_month_l15_15603

-- Declaring the conditions given in the problem
variables (total_cars : ℕ) (sales_people : ℕ) (months : ℕ)

-- Given conditions
def sterlingAutoStore := total_cars = 500
def numberOfSalesPeople := sales_people = 10
def monthsToSellAllCars := months = 5

-- The proof statement
theorem cars_per_salesperson_per_month (h1 : sterlingAutoStore) (h2 : numberOfSalesPeople) (h3 : monthsToSellAllCars): 
  500 / 5 / 10 = 10 := by 
    sorry

end cars_per_salesperson_per_month_l15_15603


namespace inequality_solution_l15_15427

theorem inequality_solution (x : ℝ) (h : 3 * x + 4 ≠ 0) : 
  (3 - 2 / (3 * x + 4) < 5) ↔ (x < -(4 / 3) ∨ x > -(5 / 3)) := 
by
  sorry

end inequality_solution_l15_15427


namespace prob_exactly_M_laws_in_concept_expected_laws_in_concept_l15_15726

section Anchuria
variables (K N M : ℕ) (p : ℝ)

-- Part (a): Define P_M as the binomial probability distribution result
def probability_exactly_M_laws : ℝ :=
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M)

-- Part (a): Prove the result for the probability that exactly M laws are included in the Concept
theorem prob_exactly_M_laws_in_concept :
  probability_exactly_M_laws K N M p =
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) := 
sorry

-- Part (b): Define the expected number of laws included in the Concept
def expected_number_of_laws : ℝ :=
  K * (1 - (1 - p) ^ N)

-- Part (b): Prove the result for the expected number of laws included in the Concept
theorem expected_laws_in_concept :
  expected_number_of_laws K N p = K * (1 - (1 - p) ^ N) :=
sorry
end Anchuria

end prob_exactly_M_laws_in_concept_expected_laws_in_concept_l15_15726


namespace min_cost_6000_l15_15400

noncomputable def min_cost_purchase : ℝ :=
  let x := 20
  let y := 16
  let total_jumpropes_shufflecocks := 400
  let discount_jumpropes := 0.8
  let discount_shufflecocks := 0.75
  let m := total_jumpropes_shufflecocks
  let w (m: ℝ) := (20 * discount_jumpropes * m) + (16 * discount_shufflecocks * (400 - m))
  in min (w 300) (w 310)

theorem min_cost_6000
  (cost1 : 5 * 20 + 6 * 16 = 196)
  (cost2 : 2 * 20 + 5 * 16 = 120)
  (total_eq : ∀ m, 300 ≤ m ∧ m ≤ 310)
: min_cost_purchase = 6000 := by
  sorry  -- detailed proof steps go here

end min_cost_6000_l15_15400


namespace janabel_sales_l15_15748

theorem janabel_sales : 
  let a₁ := 1
  let d := 4
  let n := 12
  let a (k : ℕ) := a₁ + k * d
  let S := ∑ i in Finset.range n, a i
  S = 276 :=
by
  let a₁ := 1
  let d := 4
  let n := 12
  let a (k : ℕ) := a₁ + k * d
  let S := ∑ i in Finset.range n, a i
  have h1 : a₁ + (n-1)*d = 45 := sorry
  have h2 : (n/2) * (a₁ + 45) = 276 := sorry
  exact h2

end janabel_sales_l15_15748


namespace find_values_l15_15245

-- Given conditions
variables {k b a : ℝ}
def linear_function (x : ℝ) := k * x + b
def proportional_function (x : ℝ) := (1 / 2) * x

-- Points (conditions)
def point1 := linear_function (-1) = -5
def point2 := linear_function 2 = a
def intersection := proportional_function 2 = a

-- Proving the requirements
theorem find_values (h1 : point1) (h2 : point2) (h3 : intersection) :
  a = 1 ∧ k = 2 ∧ b = -3 ∧ (1 / 2 * (3 / 2) * 1 = 3 / 4) :=
by
  sorry

end find_values_l15_15245


namespace find_a5_find_a31_div_a29_l15_15171

noncomputable def geo_diff_seq (a : ℕ → ℕ) (d : ℕ) :=
∀ n : ℕ, n > 0 → (a (n + 2) / a (n + 1)) - (a (n + 1) / a n) = d

theorem find_a5 (a : ℕ → ℕ) (d : ℕ) (h_geo_diff : geo_diff_seq a d)
  (h_init : a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3) : a 5 = 105 :=
sorry

theorem find_a31_div_a29 (a : ℕ → ℕ) (d : ℕ) (h_geo_diff : geo_diff_seq a d)
  (h_init : a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3) : a 31 / a 29 = 3363 :=
sorry

end find_a5_find_a31_div_a29_l15_15171


namespace part1_eq_tangent_line_at_one_l15_15287

def f (x a : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

theorem part1_eq_tangent_line_at_one (a : ℝ) (h : a = -1) :
  ∀ x y : ℝ, 
  f x a = (1/x - 1) * Real.log (1 + x) →
  x = 1 →
  y = 0 →
  (Real.log 2) * x + y - Real.log 2 = 0 := 
sorry

end part1_eq_tangent_line_at_one_l15_15287


namespace number_of_diagonals_l15_15566

-- Define a rectangular prism with its properties
structure RectangularPrism :=
  (vertices : Finset (Fin 12))
  (edges : Finset (Fin 18))

-- Define what it means for a segment to be diagonal
def is_diagonal (prism : RectangularPrism) (seg : (Fin 12) × (Fin 12)) : Prop :=
  ¬ prism.edges.contain seg ∧ 
  seg.1 ≠ seg.2

-- Define face and space diagonals separately
def face_diagonals (prism : RectangularPrism) : Nat :=
  6 * 2

def space_diagonals (prism : RectangularPrism) : Nat :=
  (12 * 2) // 2

-- Prove the total number of diagonals in a rectangular prism is 24
theorem number_of_diagonals (prism : RectangularPrism) : 
  face_diagonals prism + space_diagonals prism = 24 :=
by 
  sorry

end number_of_diagonals_l15_15566


namespace remaining_pieces_l15_15932

/-- Define the initial number of pieces on a standard chessboard. -/
def initial_pieces : Nat := 32

/-- Define the number of pieces lost by Audrey. -/
def audrey_lost : Nat := 6

/-- Define the number of pieces lost by Thomas. -/
def thomas_lost : Nat := 5

/-- Proof that the remaining number of pieces on the chessboard is 21. -/
theorem remaining_pieces : initial_pieces - (audrey_lost + thomas_lost) = 21 := by
  -- Mathematical equivalence to 32 - (6 + 5) = 21
  sorry

end remaining_pieces_l15_15932


namespace fibonacci_polynomial_l15_15251

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

-- Given conditions: Polynomial P of degree n
variables {P : ℕ → ℕ} {n : ℕ}

-- Hypothesis: P(k) = a_k for k = n+2 to 2n+2
def polynomial_condition (P : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ k, n+2 ≤ k ∧ k ≤ 2*n+2 → P k = fibonacci k

theorem fibonacci_polynomial (P : ℕ → ℕ) (n : ℕ)
  (h : polynomial_condition P n) : 
  P (2*n+3) = fibonacci (2*n+3) - 1 :=
sorry

-- We can use 'sorry' as a placeholder for the proof, indicating that the proof implementation is omitted.

end fibonacci_polynomial_l15_15251


namespace tangent_line_at_1_range_of_a_iff_monotonic_increasing_l15_15263

noncomputable def f (x a : ℝ) : ℝ := (1 / x + a) * real.log (1 + x)

theorem tangent_line_at_1 (a : ℝ) (h : a = -1) :
  let f_at := (1 / (1:ℝ) + a) * real.log (1 + 1)
  let f_prime := -real.log 2
  let tangent_line := λ x : ℝ, f_prime * (x - 1) + f_at in
  tangent_line = λ x, -real.log 2 * x + real.log 2 :=
by {
  -- Proof to be provided
  sorry
}

theorem range_of_a_iff_monotonic_increasing :
  ∀ x : ℝ, x > 0 → monotonic_increasing (λ a : ℝ, (1/x + a) * real.log (1 + x)) ↔ a ∈ set.Ici (1/2) :=
by {
  -- Proof to be provided
  sorry
}

end tangent_line_at_1_range_of_a_iff_monotonic_increasing_l15_15263


namespace num_digits_two_pow_seventy_l15_15108

theorem num_digits_two_pow_seventy (log10_two_approx : ℝ) (h : log10_two_approx ≈ 0.3010) : 
  (⌊70 * log10_two_approx⌋ + 1) = 22 :=
sorry

end num_digits_two_pow_seventy_l15_15108


namespace max_voters_after_t_l15_15590

theorem max_voters_after_t
  (n : ℕ) (x : ℕ) (h₁ : 1 ≤ n) (h₂ : x ≤ 10)
  (h_rating : ∃ y : ℕ, (nx + y) / (n + 1) = x - 1) :
  (∃ k : ℕ, k = 5) :=
begin
  sorry
end

end max_voters_after_t_l15_15590


namespace find_g2_l15_15763

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def even_function (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = g x

theorem find_g2 {f g : ℝ → ℝ}
  (h1 : odd_function f)
  (h2 : even_function g)
  (h3 : ∀ x : ℝ, f x + g x = 2^x) :
  g 2 = 17 / 8 :=
sorry

end find_g2_l15_15763


namespace num_ints_satisfying_inequality_l15_15311

theorem num_ints_satisfying_inequality : ∃ n : ℕ, ∀ a : ℤ, (-4 ≤ a ∧ a ≤ 4) ∧ (-100 < a^3 ∧ a^3 < 100) → n = 9 :=
begin
  sorry
end

end num_ints_satisfying_inequality_l15_15311


namespace similar_triangles_l15_15756

variables 
  (a b c : ℂ) -- Assumed complex vectors for points A, B, and C
  (s : ℝ) -- Scaling factor
  (θ : ℝ := 2 * Real.pi / 3) -- 120 degrees in radians

-- The centroid G of triangle ABC
def g : ℂ := (a + b + c) / 3

-- Representation of the rotation by 120 degrees
def rotate_120_deg (z : ℂ) : ℂ := (-1 / 2 + (Complex.I * Real.sqrt 3 / 2)) * z

-- Vertices A', B', C' after rotation and scaling
def a' : ℂ := g + s * rotate_120_deg (a - g)
def b' : ℂ := g + s * rotate_120_deg (b - g)
def c' : ℂ := g + s * rotate_120_deg (c - g)

theorem similar_triangles : s = 1 → 
  Triangle.similar (a, b, c) (a', b', c') :=
by
  sorry

end similar_triangles_l15_15756


namespace min_stick_value_l15_15482

theorem min_stick_value (n : ℕ) (hn : n ≥ 2)
  (c : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → c i ≠ c j)
  (h_pos : ∀ i, c i > 0)
  (h_100 : ∃ i, c i = 100)
  (h_stick : ∀ m, (∃ s : Finset ℕ, s.sum c = m) ↔ m ∈ Finset.range (Finset.univ.sum c + 1)) :
  Finset.univ.sum c = 199 :=
sorry

end min_stick_value_l15_15482


namespace segments_perpendicular_l15_15369

open_locale classical
noncomputable theory

-- Define the circle and points A, B, C, D on it
variables (Γ : Type) [metric_space Γ] [normed_group Γ] [normed_space ℝ Γ]
variables (A B C D I J K L : Γ)
variables [is_circle Γ]

-- Define the midpoints of the arcs
def is_midpoint_of_arc (X Y Z : Γ) : Prop :=
  ∃ θ : ℝ, 0 < θ ∧ θ < π ∧ ∃ R: ℝ, dist X Y = R ∧ dist Y Z = R

-- Conditions of the problem
axioms (h1 : ∃ (A B C D: Γ), A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) 
        (h2 : is_midpoint_of_arc A I B)
        (h3 : is_midpoint_of_arc B J C)
        (h4 : is_midpoint_of_arc C K D)
        (h5 : is_midpoint_of_arc D L A)

-- Theorem to prove that line segment (IK) is perpendicular to line segment (JL)
theorem segments_perpendicular : 
  is_perpendicular (line_through I K) (line_through J L) := 
sorry

end segments_perpendicular_l15_15369


namespace third_smallest_is_four_probability_l15_15415

noncomputable def probability_third_smallest_is_four : ℚ :=
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 3 2) * (Nat.choose 8 4)
  favorable_ways / total_ways

theorem third_smallest_is_four_probability : 
  probability_third_smallest_is_four = 35 / 132 := 
sorry

end third_smallest_is_four_probability_l15_15415


namespace find_f_of_f_neg_3_l15_15256

noncomputable def f : ℝ → ℝ
| x ≤ 0 := 2 ^ x
| x > 0 := x ^ (1 / 3)

theorem find_f_of_f_neg_3 :
  f (f (-3)) = 1 / 2 :=
  sorry

end find_f_of_f_neg_3_l15_15256


namespace rotate_90_clockwise_correct_l15_15115

-- Define the necessary conditions and operations
def complex_rotation_90_clockwise (z : ℂ) : ℂ :=
  z * (-complex.I)

-- Given problem setup
def given_complex_number : ℂ := 4 + 2 * complex.I

-- The resulting complex number after rotation
def resulting_complex_number : ℂ := 2 - 4 * complex.I

-- The theorem to prove
theorem rotate_90_clockwise_correct :
  complex_rotation_90_clockwise given_complex_number = resulting_complex_number :=
by sorry

end rotate_90_clockwise_correct_l15_15115


namespace problem_part_1_problem_part_2_problem_part_3_l15_15978

/-- There are 4 students and 4 seats, each occupied by exactly one student.
1. Number of ways in which exactly two students have matching seat and student numbers is 6.
2. Number of ways in which no student has matching seat and student numbers is 9.
3. Number of ways in which students numbered 1 and 2 must sit together is 12.
-/
theorem problem_part_1 (students seats : Fin 4 → Fin 4) : 
  ∃! count, count = 6 ∧ (exactly_two_matching_positions students seats count) := sorry

theorem problem_part_2 (students seats : Fin 4 → Fin 4) : 
  ∃! count, count = 9 ∧ (no_matching_positions students seats count) := sorry

theorem problem_part_3 (students seats : Fin 4 → Fin 4) : 
  ∃! count, count = 12 ∧ (students_1_and_2_together students seats count) := sorry

/-- Helper functions --/
def exactly_two_matching_positions (students seats : Fin 4 → Fin 4) (n : ℕ) : Prop := 
  -- the implementation of counting exactly two matching positions
  sorry

def no_matching_positions (students seats : Fin 4 → Fin 4) (n : ℕ) : Prop := 
  -- the implementation of counting no matching positions
  sorry

def students_1_and_2_together (students seats : Fin 4 → Fin 4) (n : ℕ) : Prop := 
  -- the implementation of counting students 1 and 2 sitting together
  sorry

end problem_part_1_problem_part_2_problem_part_3_l15_15978


namespace cricket_players_count_l15_15333

theorem cricket_players_count (Hockey Football Softball Total Cricket : ℕ) 
    (hHockey : Hockey = 12)
    (hFootball : Football = 18)
    (hSoftball : Softball = 13)
    (hTotal : Total = 59)
    (hTotalCalculation : Total = Hockey + Football + Softball + Cricket) : 
    Cricket = 16 := by
  sorry

end cricket_players_count_l15_15333


namespace cube_root_of_sum_of_powers_l15_15868

theorem cube_root_of_sum_of_powers :
  ∃ (x : ℝ), x = 16 * (4 ^ (1 / 3)) ∧ x = (4^6 + 4^6 + 4^6 + 4^6) ^ (1 / 3) :=
by
  sorry

end cube_root_of_sum_of_powers_l15_15868


namespace parabola_line_x_coords_l15_15839

noncomputable def parabola_line_intersection (a k b x1 x2 x3 : ℝ) : Prop :=
∃ (f g : ℝ → ℝ), 
  f = (λ x, a * x^2) ∧ 
  g = (λ x, k * x + b) ∧ 
  (f x1 = g x1 ∧ f x2 = g x2) ∧
  (g x3 = 0) ∧ 
  (a > 0)

theorem parabola_line_x_coords (a k b x1 x2 x3 : ℝ) 
  (h : parabola_line_intersection a k b x1 x2 x3) : 
  x1 * x2 = x2 * x3 + x1 * x3 :=
sorry

end parabola_line_x_coords_l15_15839


namespace symmetric_y_axis_iff_conj_neg_M_l15_15371

-- Given complex numbers z and z', M and M' are symmetric with respect to the y-axis 
-- if and only if z' = -conj(z).

open Complex

-- Definition of the complex numbers z and z'
variable (z z' : ℂ)

-- The main theorem
theorem symmetric_y_axis_iff_conj_neg :
  M.symm_y_axis_iff (z : ℂ) (z' : ℂ) ↔ z' = -conj(z) := sorry

-- Auxiliary theorem for symmetry around the y-axis
theorem M.symm_y_axis_iff (z z' : ℂ) : 
  z = -(z') ↔ z.re = -z'.re ∧ z.im = z'.im := sorry

end symmetric_y_axis_iff_conj_neg_M_l15_15371


namespace Tom_purchased_9_kg_of_mangoes_l15_15855

def mangoes_kg (A : ℕ) (R_a : ℕ) (P : ℕ) (R_m : ℕ) : ℕ :=
let cost_apples := A * R_a in
let cost_mangoes := P - cost_apples in
cost_mangoes / R_m

theorem Tom_purchased_9_kg_of_mangoes (A R_a P R_m : ℕ)
  (hA : A = 8) (hRa : R_a = 70) (hP : P = 1055) (hRm : R_m = 55) :
  mangoes_kg A R_a P R_m = 9 :=
by
  subst hA hRa hP hRm
  simp [mangoes_kg]
  sorry

end Tom_purchased_9_kg_of_mangoes_l15_15855


namespace watch_correction_is_28_minutes_l15_15920

theorem watch_correction_is_28_minutes :
  ∀ (loss_per_day : ℕ) (initial_time : Nat) (target_time : Nat), 
    (loss_per_day = 3) →
    (initial_time = 0) →
    (target_time = 224) →
    let loss_per_hour := loss_per_day / 24 in
    let total_loss := target_time * loss_per_hour in
    let correction := total_loss in
    correction = 28 :=
by 
  intros loss_per_day initial_time target_time h_loss_per_day h_initial_time h_target_time
  let loss_per_hour := 3 / 24
  let total_loss := 224 * loss_per_hour
  let correction := total_loss
  have : loss_per_hour = 1/8 := rfl
  have : total_loss = 224 * (1/8) := by simp
  have : total_loss = 28 := by norm_num
  exact this

end watch_correction_is_28_minutes_l15_15920


namespace trig_identity_l15_15607

theorem trig_identity : csc (π / 14) - 4 * cos (2 * π / 7) = 2 := by
  sorry

end trig_identity_l15_15607


namespace Emily_Collected_8484_Eggs_l15_15178

theorem Emily_Collected_8484_Eggs : 
  (let number_of_baskets := 303 in
   let number_of_eggs_per_basket := 28 in
   number_of_baskets * number_of_eggs_per_basket = 8484) :=
by
  let number_of_baskets := 303
  let number_of_eggs_per_basket := 28
  calc
    number_of_baskets * number_of_eggs_per_basket = 303 * 28 : by sorry
    ... = 8484 : by sorry

end Emily_Collected_8484_Eggs_l15_15178


namespace correct_statements_l15_15591

def Statement (n : Nat) : String :=
  match n with
  | 1 => "A census method is easiest for understanding population flow in and out of Yangzhou city on a particular day"
  | 2 => "Among 367 students born in the same year, there are at least two who share the same birthday"
  | 3 => "Turning on the TV and finding a children's program broadcasting is a random event"
  | 4 => "An event with one in a hundred thousand chance of happening can still occur"
  | _ => ""

theorem correct_statements :
  Statement 2 ∧ Statement 3 ∧ Statement 4
:= by
  sorry

end correct_statements_l15_15591


namespace ali_can_play_forever_l15_15144

noncomputable def ali_never_loses_game (P : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, P t ≥ 0

theorem ali_can_play_forever :
  ∀ P : ℝ → ℝ, (P = λ x, 1 + x ^ 1398 ∨ (∃ k : ℕ, k ∈ set.Icc 0 1398 ∧ P = λ x, P x + x ^ k)) → ali_never_loses_game P :=
sorry

end ali_can_play_forever_l15_15144


namespace prob_exactly_M_laws_included_expected_laws_included_l15_15731

variables (K N M : ℕ) (p : ℝ)

-- Definition of the probabilities as given in the conditions and answers
def prob_no_minister_knows_law : ℝ := (1 - p) ^ N
def prob_law_included : ℝ := 1 - prob_no_minister_knows_law p N

-- Part (a)
theorem prob_exactly_M_laws_included :
  (nat.choose K M) * (prob_law_included p N) ^ M * (prob_no_minister_knows_law p N) ^ (K - M) = 
  (nat.choose K M) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) :=
by
  sorry

-- Part (b)
theorem expected_laws_included :
  K * (prob_law_included p N) = K * (1 - (1 - p) ^ N) :=
by
  sorry

end prob_exactly_M_laws_included_expected_laws_included_l15_15731


namespace least_positive_number_divisible_by_primes_l15_15033

theorem least_positive_number_divisible_by_primes :
  ∃ n : ℕ, n > 0 ∧
    (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧
    (∀ m : ℕ, (m > 0 ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m)) → n ≤ m) ∧
    n = 2310 :=
by
  sorry

end least_positive_number_divisible_by_primes_l15_15033


namespace no_solution_inequality_l15_15708

noncomputable theory

theorem no_solution_inequality (a : ℝ) (h : a < 1) :
  ¬∃ (x : ℝ), |x - 1| + x ≤ a := by
  sorry

end no_solution_inequality_l15_15708


namespace rectangular_prism_diagonals_l15_15565

theorem rectangular_prism_diagonals
  (num_vertices : ℕ) (num_edges : ℕ)
  (h1 : num_vertices = 12) (h2 : num_edges = 18) :
  (total_diagonals : ℕ) → total_diagonals = 20 :=
by
  sorry

end rectangular_prism_diagonals_l15_15565


namespace volume_of_convex_polyhedron_l15_15901

def polyhedron_volume (faces_num_squares faces_num_hexagons : ℕ) (unit_length : ℝ) : ℝ :=
  let e := (faces_num_squares * 4 + faces_num_hexagons * 6) / 2 in
  let v := e - faces_num_squares - faces_num_hexagons + 2 in
  8 * real.sqrt 2

theorem volume_of_convex_polyhedron :
  ∀ (faces_num_squares faces_num_hexagons : ℕ) (unit_length : ℝ),
    faces_num_squares = 6 →
    faces_num_hexagons = 8 →
    (∀ (v1 v2 : ℕ), v1 ≠ v2 → ¬(v1, v2) ∈
      {(a, b) | ∀ a b : ℕ, (a ∈ faces_num_squares) ∧ (b ∈ faces_num_squares)}) →
    unit_length = 1 →
    polyhedron_volume faces_num_squares faces_num_hexagons unit_length = 8 * real.sqrt 2 :=
begin
  intros,
  sorry,
end

end volume_of_convex_polyhedron_l15_15901


namespace arithmetic_geometric_progression_l15_15668

theorem arithmetic_geometric_progression (a1 a2 b : ℝ) 
  (h1 : arithmetic_progression [-2, a1, a2, -8]) 
  (h2 : geometric_progression [-2, b, -8]) :
  (a2 - a1) / b = 1 / 2 ∨ (a2 - a1) / b = -1 / 2 := 
sorry 

end arithmetic_geometric_progression_l15_15668


namespace constant_term_is_neg240_l15_15822

noncomputable def constant_term_in_expansion : Prop :=
  let poly1 := (λ x : ℝ, x^2 + 3)
  let poly2 := (λ x : ℝ, (x - (2/x))^6)
  let expanded_poly := poly1 * poly2
  ∃ c : ℝ, (λ x : ℝ, expanded_poly) = (λ x : ℝ, c) ∧ c = -240

theorem constant_term_is_neg240 : constant_term_in_expansion :=
  sorry

end constant_term_is_neg240_l15_15822


namespace pentagon_icosagon_comparison_l15_15135

theorem pentagon_icosagon_comparison (R : ℝ) :
  5 * (2 * R * real.sin (real.pi / 5))^2 > 20 * (2 * R * real.sin (real.pi / 20))^2 :=
by sorry

end pentagon_icosagon_comparison_l15_15135


namespace part1_tangent_line_at_x_eq_1_part2_monotonic_increasing_l15_15276

noncomputable def f (x : ℝ) (a : ℝ) := (1/x + a) * real.log(1 + x)

theorem part1_tangent_line_at_x_eq_1 (x := 1) :
  let a := -1 
  let t : ℝ := real.log 2
  let f := (λ x, (1/x - 1) * real.log (1 + x))
  let tangent_line := λ x y, t * x + y - t 
  (tangent_line 1 (f 1) = 0) :=
by {
  sorry
}

theorem part2_monotonic_increasing (a : ℝ) : 
  (∀ x : ℝ, 0 < x → 0 < f x a) ↔ (a ≥ 1/2) := 
by {
  sorry
}

end part1_tangent_line_at_x_eq_1_part2_monotonic_increasing_l15_15276


namespace proof_l15_15666

noncomputable def question := ∀ x : ℝ, (0.12 * x = 36) → (0.5 * (0.4 * 0.3 * x) = 18) 

theorem proof : question :=
by
  intro x
  intro h
  sorry

end proof_l15_15666


namespace prob_exactly_M_laws_in_concept_expected_laws_in_concept_l15_15723

section Anchuria
variables (K N M : ℕ) (p : ℝ)

-- Part (a): Define P_M as the binomial probability distribution result
def probability_exactly_M_laws : ℝ :=
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M)

-- Part (a): Prove the result for the probability that exactly M laws are included in the Concept
theorem prob_exactly_M_laws_in_concept :
  probability_exactly_M_laws K N M p =
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) := 
sorry

-- Part (b): Define the expected number of laws included in the Concept
def expected_number_of_laws : ℝ :=
  K * (1 - (1 - p) ^ N)

-- Part (b): Prove the result for the expected number of laws included in the Concept
theorem expected_laws_in_concept :
  expected_number_of_laws K N p = K * (1 - (1 - p) ^ N) :=
sorry
end Anchuria

end prob_exactly_M_laws_in_concept_expected_laws_in_concept_l15_15723


namespace polar_coordinates_of_A_and_B_distance_MN_l15_15688

noncomputable def polar_eq_C1 (rho theta : ℝ) : Prop :=
  (rho^2) * Real.cos (2 * theta) = 18

def polar_eq_C2 (theta : ℝ) : Prop := 
  theta = Real.pi / 6

def line_eq (t : ℝ) : ℝ × ℝ :=
  (2 + (Real.sqrt 3) / 2 * t, 1 / 2 * t)

theorem polar_coordinates_of_A_and_B :
  polar_eq_C1 6 (Real.pi / 6) ∧ polar_eq_C2 (Real.pi / 6) ∧ polar_eq_C1 (-6) (Real.pi / 6) :=
by sorry

theorem distance_MN :
  ∀ (t : ℝ), (line_eq t).fst^2 - (line_eq t).snd^2 = 18 → ∀ t₁ t₂, 
  (fst (line_eq t₁) = 0 → snd (line_eq t₁) = 0 → fst (line_eq t₂) = 0 → snd (line_eq t₂) = 0 → sqrt (((line_eq t₁).fst - (line_eq t₂).fst)^2 + ((line_eq t₁).snd - (line_eq t₂).snd)^2) = 4 * Real.sqrt 10 :=
by sorry

end polar_coordinates_of_A_and_B_distance_MN_l15_15688


namespace smallest_number_increased_by_3_divisible_by_divisors_l15_15069

theorem smallest_number_increased_by_3_divisible_by_divisors
  (n : ℕ)
  (d1 d2 d3 d4 : ℕ)
  (h1 : d1 = 27)
  (h2 : d2 = 35)
  (h3 : d3 = 25)
  (h4 : d4 = 21) :
  (n + 3) % d1 = 0 →
  (n + 3) % d2 = 0 →
  (n + 3) % d3 = 0 →
  (n + 3) % d4 = 0 →
  n = 4722 :=
by
  sorry

end smallest_number_increased_by_3_divisible_by_divisors_l15_15069


namespace range_of_m_l15_15658

variable (m : ℝ)
def prop_p := abs (m + 1) ≤ 2
def prop_q := ∃ (x : ℝ), x^2 - 2*m*x + 1 = 0
def neg_p_false := ¬¬ prop_p
def p_and_q_false := ¬ (prop_p ∧ prop_q)

theorem range_of_m {m : ℝ} (hp : prop_p m) (hq : ¬ prop_q m) : -1 < m ∧ m < 1 := sorry

end range_of_m_l15_15658


namespace num_of_ints_l15_15308

theorem num_of_ints (n : ℤ) (h : -100 < n^3) (h2 : n^3 < 100) : 
    (finset.card (finset.filter (λ x : ℤ, -100 < x^3 ∧ x^3 < 100) (finset.Icc (-4) 4))) = 9 :=
sorry

end num_of_ints_l15_15308


namespace percent_of_plane_enclosed_by_triangles_l15_15339

theorem percent_of_plane_enclosed_by_triangles 
  (a : ℕ) 
  (as : set ℕ)
  (h1: ∀ x ∈ as, x = 2 ∨ x = 3)
  (h2: side_length_of_square : Nat := 3 * a)
  (h3: smaller_shapes_area : Nat := a^2 / 3) 
  (area_covered_by_triangles : ℕ := 2 * a^2) : 
  (area_covered_by_triangles / (9 * a^2) * 100 = 67) :=
by
  sorry

end percent_of_plane_enclosed_by_triangles_l15_15339


namespace part_a_part_b_l15_15776

-- Definitions for conditions
variable (figure : Type) [fintype figure] [decidable_eq figure]
variable (area_of : figure → ℝ)
variable (to_square : list figure → Prop)

-- First part a: No more than four parts to form a square
theorem part_a (h_area : ∃ A : ℝ, A = ∑ x in (finset.univ : finset figure), area_of x)
  (h_part : ∀ (p : list figure), p.length ≤ 4 → to_square p) :
  ∃ p : list figure, p.length ≤ 4 ∧ to_square p :=
sorry

-- Second part b: No more than five triangular parts to form a square
def is_triangle (fig : figure) : Prop := true -- This is a placeholder; the real definition of a triangular part must be provided.

theorem part_b (h_area : ∃ A : ℝ, A = ∑ x in (finset.univ : finset figure), area_of x)
  (h_part : ∀ (p : list figure), (∀ x ∈ p, is_triangle x) → p.length ≤ 5 → to_square p) :
  ∃ p : list figure, (∀ x ∈ p, is_triangle x) ∧ p.length ≤ 5 ∧ to_square p :=
sorry

end part_a_part_b_l15_15776


namespace least_number_divisible_by_five_primes_l15_15063

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, (n > 0) ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → p ∣ n) ∧ n = 2310 :=
begin
  sorry
end

end least_number_divisible_by_five_primes_l15_15063


namespace interest_rate_second_amount_l15_15328

-- Define the simple interest formula parameters and conditions.
def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

-- Given conditions
def P1 : ℝ := 100
def r1 : ℝ := 0.05
def t1 : ℝ := 8

def P2 : ℝ := 200
def t2 : ℝ := 2

-- Interest produced by P1 in t1 years at r1 rate
def I1 : ℝ := simple_interest P1 r1 t1

-- Interest produced by P2 in t2 years at r2 rate is the same as I1
def r2 (I2 : ℝ) : ℝ := I2 / (P2 * t2)

theorem interest_rate_second_amount : r2 I1 = 0.1 := by
  -- The proof will be done here
  sorry

end interest_rate_second_amount_l15_15328


namespace prob_exactly_M_laws_in_concept_expected_laws_in_concept_l15_15724

section Anchuria
variables (K N M : ℕ) (p : ℝ)

-- Part (a): Define P_M as the binomial probability distribution result
def probability_exactly_M_laws : ℝ :=
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M)

-- Part (a): Prove the result for the probability that exactly M laws are included in the Concept
theorem prob_exactly_M_laws_in_concept :
  probability_exactly_M_laws K N M p =
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) := 
sorry

-- Part (b): Define the expected number of laws included in the Concept
def expected_number_of_laws : ℝ :=
  K * (1 - (1 - p) ^ N)

-- Part (b): Prove the result for the expected number of laws included in the Concept
theorem expected_laws_in_concept :
  expected_number_of_laws K N p = K * (1 - (1 - p) ^ N) :=
sorry
end Anchuria

end prob_exactly_M_laws_in_concept_expected_laws_in_concept_l15_15724


namespace inequality_solution_l15_15428

theorem inequality_solution (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  (3 - 2 / (3 * x + 4) < 5) ↔ x ∈ set.Ioo (-5/3 : ℝ) (-4/3) ∪ set.Ioi (-4/3) :=
by
  sorry

end inequality_solution_l15_15428


namespace least_positive_number_divisible_by_five_primes_l15_15051

theorem least_positive_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ 
    (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) :=
by
  use 2310
  split
  { exact by norm_num }
  split
  { intros p hp, fin_cases hp; norm_num }
  { intros m hm, apply nat.le_of_dvd; norm_num }
  sorry

end least_positive_number_divisible_by_five_primes_l15_15051


namespace sum_of_a_b_l15_15985

noncomputable def a_b_sum (a b : ℝ) : ℝ :=
  a + b

theorem sum_of_a_b (a b : ℝ) (h1 : a ≠ b)
  (h2 : ∃ M, M = {a^2 - 4 * a, -1})
  (h3 : ∃ N, N = {b^2 - 4 * b + 1, -2})
  (h4 : ∀ x ∈ M, x ∈ N) :
  a + b = 4 :=
  sorry

end sum_of_a_b_l15_15985


namespace percentage_water_mixture_l15_15779

theorem percentage_water_mixture 
  (volume_A : ℝ) (volume_B : ℝ) (volume_C : ℝ)
  (ratio_A : ℝ := 5) (ratio_B : ℝ := 3) (ratio_C : ℝ := 2)
  (percentage_water_A : ℝ := 0.20) (percentage_water_B : ℝ := 0.35) (percentage_water_C : ℝ := 0.50) :
  (volume_A = ratio_A) → (volume_B = ratio_B) → (volume_C = ratio_C) → 
  ((percentage_water_A * volume_A + percentage_water_B * volume_B + percentage_water_C * volume_C) /
   (ratio_A + ratio_B + ratio_C)) * 100 = 30.5 := 
by 
  intros hA hB hC
  -- Proof steps would go here
  sorry

end percentage_water_mixture_l15_15779


namespace sequences_properties_l15_15693

-- Definition of sequences and their properties
variable {n : ℕ}

noncomputable def S (n : ℕ) : ℕ := n^2 - n
noncomputable def a (n : ℕ) : ℕ := if n = 1 then 0 else 2 * n - 2
noncomputable def b (n : ℕ) : ℕ := 3^(n-1)
noncomputable def c (n : ℕ) : ℕ := (2 * (n - 1)) / 3^(n - 1)
noncomputable def T (n : ℕ) : ℕ := 3 / 2 - (2 * n + 1) / (2 * 3^(n-1))

-- Main theorem
theorem sequences_properties (n : ℕ) (hn : n > 0) :
  S n = n^2 - n ∧
  (∀ n, a n = if n = 1 then 0 else 2 * n - 2) ∧
  (∀ n, b n = 3^(n-1)) ∧
  (∀ n, T n = 3 / 2 - (2 * n + 1) / (2 * 3^(n-1))) :=
by sorry

end sequences_properties_l15_15693


namespace inequality_solution_l15_15612

theorem inequality_solution (a : ℝ) (ha : 1 < a) :
  {x : ℝ | |x + real.log x / real.log a| < |x| + |real.log x / real.log a|} = {x : ℝ | 0 < x ∧ x < 1} :=
sorry

end inequality_solution_l15_15612


namespace relationship_among_a_b_c_l15_15652

-- Defining the properties and conditions of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Defining the function f based on the condition
noncomputable def f (x m : ℝ) : ℝ := 2 ^ |x - m| - 1

-- Defining the constants a, b, c
noncomputable def a : ℝ := f (Real.log 3 / Real.log 0.5) 0
noncomputable def b : ℝ := f (Real.log 5 / Real.log 2) 0
noncomputable def c : ℝ := f 0 0

-- The theorem stating the relationship among a, b, and c
theorem relationship_among_a_b_c : c < a ∧ a < b := by
  sorry

end relationship_among_a_b_c_l15_15652


namespace total_branches_in_pine_tree_l15_15916

-- Definitions based on the conditions
def middle_branch : ℕ := 0 -- arbitrary assignment to represent the middle branch

def jumps_up_5 (b : ℕ) : ℕ := b + 5
def jumps_down_7 (b : ℕ) : ℕ := b - 7
def jumps_up_4 (b : ℕ) : ℕ := b + 4
def jumps_up_9 (b : ℕ) : ℕ := b + 9

-- The statement to be proven
theorem total_branches_in_pine_tree : 
  (jumps_up_9 (jumps_up_4 (jumps_down_7 (jumps_up_5 middle_branch))) = 11) →
  ∃ n, n = 23 :=
by
  sorry

end total_branches_in_pine_tree_l15_15916


namespace sum_of_cubes_gt_40_l15_15476

theorem sum_of_cubes_gt_40 (n : ℕ) (a : Fin n → ℝ)
  (h_sum : ∑ i, a i = 10)
  (h_sum_sq : ∑ i, (a i)^2 > 20) :
  ∑ i, (a i)^3 > 40 :=
sorry

end sum_of_cubes_gt_40_l15_15476


namespace solution_set_l15_15677

def f : ℝ → ℝ := λ x, if x ≤ 0 then 3^(x + 1) else log (1/2) x

theorem solution_set (x : ℝ) : (f x > 1) ↔ (-1 < x ∧ x < 1/2) :=
by {
    sorry
}

end solution_set_l15_15677


namespace prove_perp_lines_l15_15601

variables {a b : Type*} {α β : set Type*} [linear_order a] [linear_order b] 
           [affine_plane α] [affine_plane β]

-- Assume a and b are different lines, α and β are different planes
def are_different_lines (a b : Type*) : Prop := a ≠ b
def are_different_planes (α β : set Type*) : Prop := α ≠ β

-- Assume a ⊂ α
def line_in_plane (a : Type*) (α : set Type*) : Prop := ∀ x ∈ α, x = a

-- Assume b ⊥ β and α ∥ β
def perp_line_plane (b : Type*) (β : set Type*) : Prop := ∀ y ⊥ β, y = b
def parallel_planes (α β : set Type*) : Prop := ∀ z ∥ β, z = α

theorem prove_perp_lines
  (h1: are_different_lines a b)
  (h2: are_different_planes α β)
  (h3: line_in_plane a α)
  (h4: perp_line_plane b β)
  (h5: parallel_planes α β) :
  a ⊥ b :=
sorry

end prove_perp_lines_l15_15601


namespace finite_region_exists_l15_15794

/-- Given a finite set of lines on a plane such that no two lines are parallel
   and no three lines are concurrent, prove that there exists at least one finite
   region among the regions into which these lines divide the plane. -/
theorem finite_region_exists (L : Finset (set (affine_plane ℝ))) (h1 : ∀ l1 l2 ∈ L, l1 ≠ l2 → ¬ parallel l1 l2)
  (h2 : ∀ l1 l2 l3 ∈ L, l1 ≠ l2 → l2 ≠ l3 → l1 ≠ l3 → ¬ concurrent l1 l2 l3) :
  ∃ R, finite_region R L :=
sorry

end finite_region_exists_l15_15794


namespace two_dollar_coin_is_toonie_l15_15599

/-- We define the $2 coin in Canada -/
def two_dollar_coin_name : String := "toonie"

/-- Antonella's wallet problem setup -/
def Antonella_has_ten_coins := 10
def loonies_value := 1
def toonies_value := 2
def coins_after_purchase := 11
def purchase_amount := 3
def initial_toonies := 4

/-- Proving that the $2 coin is called a "toonie" -/
theorem two_dollar_coin_is_toonie :
  two_dollar_coin_name = "toonie" :=
by
  -- Here, we place the logical steps to derive that two_dollar_coin_name = "toonie"
  sorry

end two_dollar_coin_is_toonie_l15_15599


namespace students_in_class_l15_15484

def total_eggs : Nat := 56
def eggs_per_student : Nat := 8
def num_students : Nat := 7

theorem students_in_class :
  total_eggs / eggs_per_student = num_students :=
by
  sorry

end students_in_class_l15_15484


namespace card_draw_multiple_of_10_l15_15150

theorem card_draw_multiple_of_10 :
  let points := λ c, if c ≤ 10 then c else 10 in
  let draws := [ (1 : ℕ) .. 10, 10, 10, 10 ] in
  let prob := (244 / 2197 : ℚ) in
  let m := 244 in
  let n := 2197 in
  gcd m n = 1 ∧ 100 * m + n = 26597
:= by
  -- Points of cards
  let points := λ c, if c ≤ 10 then c else 10 in
  -- Drawing cards probabilities
  let draws := [ (1 : ℕ) .. 10, 10, 10, 10 ] in
  let prob := (244 / 2197 : ℚ) in
  -- Values of m and n
  let m := 244 in
  let n := 2197 in
  -- Check gcd and calculation
  have h_gcd : gcd m n = 1 := by nat.gcd_def 244 2197 sorry,
  have h_calc : 100 * m + n = 26597 := by norm_num,
  exact ⟨h_gcd, h_calc⟩

end card_draw_multiple_of_10_l15_15150


namespace final_percentage_is_46_l15_15903

def initial_volume : ℚ := 50
def initial_concentration : ℚ := 0.60
def drained_volume : ℚ := 35
def replacement_concentration : ℚ := 0.40

def initial_chemical_amount : ℚ := initial_volume * initial_concentration
def drained_chemical_amount : ℚ := drained_volume * initial_concentration
def remaining_chemical_amount : ℚ := initial_chemical_amount - drained_chemical_amount
def added_chemical_amount : ℚ := drained_volume * replacement_concentration
def final_chemical_amount : ℚ := remaining_chemical_amount + added_chemical_amount
def final_volume : ℚ := initial_volume

def final_percentage : ℚ := (final_chemical_amount / final_volume) * 100

theorem final_percentage_is_46 :
  final_percentage = 46 := by
  sorry

end final_percentage_is_46_l15_15903


namespace sum_of_possible_values_of_g1_l15_15764

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, g(g(x - y)) = g(x) * g(y) - g(x) + g(y) - x^2 * y

theorem sum_of_possible_values_of_g1 : g(1) = 1 ∨ g(1) = -1 ∨ g(1) = 2 :=
sorry

end sum_of_possible_values_of_g1_l15_15764


namespace point_C_divides_AE_l15_15221

theorem point_C_divides_AE (ABC : Triangle) (D : Point) (E : Point) (ω : Circumcircle) 
    (h1 : is_right_triangle ABC)
    (h2 : is_on_extension D (BC))
    (h3 : is_tangent_line_to_circle AD ω)
    (h4 : intersects_circle AC (circumcircle (ABD)) E)
    (h5 : is_tangent_angle_bisector (angle_bisector ADE) ω) :
    divides_in_ratio C (segment AE) (1, 2) := 
sorry

end point_C_divides_AE_l15_15221


namespace num_zeros_of_g_l15_15241

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - 2 * x
else -(x^2 - 2 * -x)

noncomputable def g (x : ℝ) : ℝ := f x + 1

theorem num_zeros_of_g : ∃! x : ℝ, g x = 0 := sorry

end num_zeros_of_g_l15_15241


namespace find_y_l15_15966

open Real

theorem find_y (y: ℝ) (h: log 16 (4 * y - 5) = 5 / 4) : y = 37 / 4 :=
sorry

end find_y_l15_15966


namespace largest_n_with_100_trailing_zeros_l15_15630

def trailing_zeros_factorial (n : ℕ) : ℕ :=
  if n = 0 then 0 else n / 5 + trailing_zeros_factorial (n / 5)

theorem largest_n_with_100_trailing_zeros :
  ∃ (n : ℕ), trailing_zeros_factorial n = 100 ∧ ∀ (m : ℕ), (trailing_zeros_factorial m = 100 → m ≤ 409) :=
by
  sorry

end largest_n_with_100_trailing_zeros_l15_15630


namespace snack_eaters_left_l15_15557

theorem snack_eaters_left (initial_participants : ℕ)
    (snack_initial : ℕ)
    (new_outsiders1 : ℕ)
    (half_left1 : ℕ)
    (new_outsiders2 : ℕ)
    (left2 : ℕ)
    (half_left2 : ℕ)
    (h1 : initial_participants = 200)
    (h2 : snack_initial = 100)
    (h3 : new_outsiders1 = 20)
    (h4 : half_left1 = (snack_initial + new_outsiders1) / 2)
    (h5 : new_outsiders2 = 10)
    (h6 : left2 = 30)
    (h7 : half_left2 = (half_left1 + new_outsiders2 - left2) / 2) :
    half_left2 = 20 := 
  sorry

end snack_eaters_left_l15_15557


namespace gcd_2970_1722_lcm_2970_1722_l15_15860

-- Given numbers
def a : ℕ := 2970
def b : ℕ := 1722

-- GCD of a and b is 6
theorem gcd_2970_1722 : Nat.gcd a b = 6 :=
by sorry

-- LCM of a and b is 856170
theorem lcm_2970_1722 : Nat.lcm a b = 856170 :=
by 
  rw [Nat.lcm_eq (gcd_2970_1722)]  -- Use the result of gcd_2970_1722
  sorry

end gcd_2970_1722_lcm_2970_1722_l15_15860


namespace recurring_decimal_as_fraction_l15_15087

theorem recurring_decimal_as_fraction : 
  (let a := 0.\overline{56} in 
    100 * a = 56.\overline{56}) → 
  0.\overline{56} = 56 / 99 := 
sorry

end recurring_decimal_as_fraction_l15_15087


namespace parametric_eq_of_line_distance_AB_of_line_intersecting_curve_solution_set_M_inequality_for_M_l15_15888

noncomputable def theta := real.pi / 3

-- Proof Problem 1.1
theorem parametric_eq_of_line (t : ℝ) : 
  ∃ x y, x = (1/2) * t ∧ y = (real.sqrt 3 / 2) * t :=
sorry

-- Proof Problem 1.2
theorem distance_AB_of_line_intersecting_curve :
  ∀ (t₁ t₂ : ℝ), (t₁ + t₂ = real.sqrt 3 ∧ t₁ * t₂ = -3) → (|t₁ - t₂| = real.sqrt 15) :=
sorry

-- Proof Problem 2.1
theorem solution_set_M : 
  ∃ M : set ℝ, M = {x : ℝ | -1 / 3 < x ∧ x < 3} ∧ (∀ x ∈ M, |x + 2| - |2*x - 1| > 0) :=
sorry

-- Proof Problem 2.2
theorem inequality_for_M (x y : ℝ) (hx : -1 / 3 < x ∧ x < 3) (hy : -1 / 3 < y ∧ y < 3) : 
  |x + y + x * y| < 15 :=
sorry

end parametric_eq_of_line_distance_AB_of_line_intersecting_curve_solution_set_M_inequality_for_M_l15_15888


namespace product_of_numbers_l15_15451

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 185) : x * y = 26 :=
sorry

end product_of_numbers_l15_15451


namespace tangent_line_equation_at_1_monotonically_increasing_range_of_a_l15_15268

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (1 / x + a) * real.log (1 + x)

-- Part 1: Tangent line equation at point (1, f(1)) when a = -1
theorem tangent_line_equation_at_1 (x y : ℝ) (h : y = f 1 (-1)) :
  (real.log 2) * x + y - (real.log 2) = 0 := 
sorry

-- Part 2: Range of a for which f(x) is monotonically increasing on (0, +∞)
theorem monotonically_increasing_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, 0 < x → 0 < f' x a) ↔ (1 / 2 ≤ a) :=  
sorry

end tangent_line_equation_at_1_monotonically_increasing_range_of_a_l15_15268


namespace sequence_remainder_100_mod_7_l15_15457

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 3 else 3 ^ sequence (n - 1)

theorem sequence_remainder_100_mod_7 : (sequence 100) % 7 = 6 := 
  sorry

end sequence_remainder_100_mod_7_l15_15457


namespace magnitude_AB_is_sqrt_2_l15_15239

-- Define the vector AB and its components
def AB : ℝ × ℝ := (1, 1)

-- Define the magnitude function for a vector
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Prove that the magnitude of AB is sqrt(2)
theorem magnitude_AB_is_sqrt_2 : magnitude AB = real.sqrt 2 :=
by
  sorry

end magnitude_AB_is_sqrt_2_l15_15239


namespace integer_values_of_n_satisfy_inequality_l15_15304

theorem integer_values_of_n_satisfy_inequality :
  ∃ S : Finset ℤ, (∀ n ∈ S, -100 < n^3 ∧ n^3 < 100) ∧ S.card = 9 :=
by
  -- Sorry provides the placeholder for where the proof would go
  sorry

end integer_values_of_n_satisfy_inequality_l15_15304


namespace extreme_min_value_of_f_l15_15452

-- Define the function f(x) = x * ln(x)
def f (x : ℝ) : ℝ := x * Real.log x

-- Define the domain for the function
def domain := {x : ℝ | 0 < x}

-- Statement of the theorem
theorem extreme_min_value_of_f :
  ∃ x ∈ domain, ∀ y ∈ domain, f x ≤ f y ∧ f x = -1 / Real.exp 1 :=
sorry

end extreme_min_value_of_f_l15_15452


namespace price_per_eraser_l15_15474

-- Definitions of the given conditions
def boxes_donated : ℕ := 48
def erasers_per_box : ℕ := 24
def total_money_made : ℝ := 864

-- The Lean statement to prove the price per eraser is $0.75
theorem price_per_eraser : (total_money_made / (boxes_donated * erasers_per_box) = 0.75) := by
  sorry

end price_per_eraser_l15_15474


namespace PQ_parallel_AB_l15_15810

-- Definitions as per the conditions in the problem statement
variables {A B C D E O P Q : Point}
variable [plane_geometry] -- Assume a plane geometry context

-- Conditions:
-- 1. The center O of the circumcircle of quadrilateral ABCD lies on the side AB.
axiom center_O : Circumcenter ABCD O ∧ Collinear {A, B, O}

-- 2. Point E is symmetric to D with respect to the line AB.
axiom symmetric_E : SymmetricPoint D A B E

-- 3. Segments AC and DO intersect at point P.
axiom intersection_P : Intersect (Line A C) (Line D O) P

-- 4. Segments BD and CE intersect at point Q.
axiom intersection_Q : Intersect (Line B D) (Line C E) Q

-- Prove that PQ is parallel to AB.
theorem PQ_parallel_AB : Parallel (Line P Q) (Line A B) :=
by
  sorry

end PQ_parallel_AB_l15_15810


namespace abs_inequality_solution_l15_15797

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ -9 / 2 < x ∧ x < 7 / 2 :=
by
  sorry

end abs_inequality_solution_l15_15797


namespace pentagon_icosagon_comparison_l15_15136

theorem pentagon_icosagon_comparison (R : ℝ) :
  5 * (2 * R * real.sin (real.pi / 5))^2 > 20 * (2 * R * real.sin (real.pi / 20))^2 :=
by sorry

end pentagon_icosagon_comparison_l15_15136


namespace product_of_consecutive_not_perfect_power_l15_15419

theorem product_of_consecutive_not_perfect_power :
  ∀ x : ℤ, ¬ ∃ y n : ℤ, 1 < n ∧ x * (x + 1) = y ^ n :=
by sorry

end product_of_consecutive_not_perfect_power_l15_15419


namespace minimum_value_expr_l15_15068

theorem minimum_value_expr (x y : ℝ) : 
  (xy - 2)^2 + (x^2 + y^2)^2 ≥ 4 :=
sorry

end minimum_value_expr_l15_15068


namespace geometric_series_sum_l15_15442

theorem geometric_series_sum :
  2016 * (1 / (1 + (1 / 2) + (1 / 4) + (1 / 8) + (1 / 16) + (1 / 32))) = 1024 :=
by
  sorry

end geometric_series_sum_l15_15442


namespace inequality_proof_l15_15533

variables (a b c d e f : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
variable (hcond : |sqrt (a * d) - sqrt (b * c)| ≤ 1)

theorem inequality_proof :
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end inequality_proof_l15_15533


namespace student_marks_l15_15917

theorem student_marks (max_marks : ℕ) (pass_percentage : ℕ) (fail_by : ℕ)
    (h_max : max_marks = 300)
    (h_percentage: pass_percentage = 30)
    (h_fail_by: fail_by = 10) :
  let pass_marks := max_marks * pass_percentage / 100 in
  let student_marks := pass_marks - fail_by in
  student_marks = 80 :=
by
  sorry

end student_marks_l15_15917


namespace rhombus_longest_diagonal_l15_15570

theorem rhombus_longest_diagonal (area : ℝ) (ratio : ℝ) (h_area : area = 192) (h_ratio : ratio = 4 / 3) :
  ∃ d1 d2 : ℝ, d1 / d2 = 4 / 3 ∧ (d1 * d2) / 2 = 192 ∧ d1 = 16 * Real.sqrt 2 :=
by
  sorry

end rhombus_longest_diagonal_l15_15570


namespace standard_eq_of_ellipse_range_of_slopes_l15_15227

-- Define the given conditions
def ellipse_eq (x y : ℝ) (a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def a_condition (a : ℝ) : Prop := a = 2
def c_condition (c : ℝ) : Prop := c = 1
def b_condition (b : ℝ) : Prop := b = Real.sqrt(3)
def vertex_A (A : ℝ × ℝ) : Prop := A = (2, 0)
def focal_length (c : ℝ) : Prop := 2 * c = 2

-- State the proof for the standard equation of the ellipse
theorem standard_eq_of_ellipse (x y : ℝ) :
  ∃ a b : ℝ, a = 2 ∧ c = 1 ∧ b = Real.sqrt(3) ∧
  ellipse_eq x y a b :=
by
  sorry

-- Define additional needed parameters and conditions for the second part
def point_E_F (E F : ℝ × ℝ) (a b : ℝ) (A : ℝ × ℝ) : Prop := 
  let (xE, yE) := E in
  let (xF, yF) := F in
  xE ≠ 2 ∧ yE ≠ 0 ∧
  xF ≠ 2 ∧ yF ≠ 0 ∧
  (xE - 2) * (xF - 2) + yE * yF = 0 ∧ -- AE ⊥ AF
  xy_ellipse_eq E a b ∧
  xy_ellipse_eq F a b

def midpoint_condition (P E F : ℝ × ℝ) : Prop :=
  let (xP, yP) := P in
  let (xE, yE) := E in
  let (xF, yF) := F in
  (2 * xP, 2 * yP) = (xE + xF, yE + yF)

def slope_condition (A P : ℝ × ℝ) (t : ℝ) : Prop := 
  let (xA, yA) := A in
  let (xP, yP) := P in
  t = (yP - yA) / (xP - xA)

-- State the proof for the range of possible slopes for line AP
theorem range_of_slopes (A E F P : ℝ × ℝ) (a b : ℝ) : 
  point_E_F E F a b A →
  midpoint_condition P E F →
  slope_condition A P t →
  -Real.sqrt(14)/56 ≤ t ∧ t ≤ Real.sqrt(14)/56 :=
by
  sorry

end standard_eq_of_ellipse_range_of_slopes_l15_15227


namespace complete_the_square_l15_15498

theorem complete_the_square (x y : ℝ) :
  (y = -x^2 + 2 * x + 3) → (y = -(x - 1)^2 + 4) :=
by 
  intro h
  rw h
  sorry

end complete_the_square_l15_15498


namespace max_points_of_intersection_l15_15898

open Set

def Point := ℝ × ℝ

structure Circle :=
(center : Point)
(radius : ℝ)

structure Line :=
(coeffs : ℝ × ℝ × ℝ) -- Assume line equation in the form Ax + By + C = 0

def max_intersection_points (circle : Circle) (lines : List Line) : ℕ :=
  let circle_line_intersect_count := 2
  let line_line_intersect_count := 1
  
  let number_of_lines := lines.length
  let pairwise_line_intersections := number_of_lines.choose 2
  
  let circle_and_lines_intersections := circle_line_intersect_count * number_of_lines
  let total_intersections := circle_and_lines_intersections + pairwise_line_intersections

  total_intersections

theorem max_points_of_intersection (c : Circle) (l1 l2 l3 : Line) :
  max_intersection_points c [l1, l2, l3] = 9 :=
by
  sorry

end max_points_of_intersection_l15_15898


namespace total_cost_of_barbed_wire_l15_15436

noncomputable def cost_of_barbed_wire : ℝ :=
  let area : ℝ := 3136
  let side_length : ℝ := Real.sqrt area
  let perimeter_without_gates : ℝ := 4 * side_length - 2 * 1
  let rate_per_meter : ℝ := 1.10
  perimeter_without_gates * rate_per_meter

theorem total_cost_of_barbed_wire :
  cost_of_barbed_wire = 244.20 :=
sorry

end total_cost_of_barbed_wire_l15_15436


namespace find_geom_seq_n_l15_15740

noncomputable def geom_seq_n (a1 a3 an_2 sn : ℕ) : ℕ := 
  let a1 := 2
  let a3 := 32
  let a_n := 32
  let q := 4
  (nat.log a_n / nat.log q).to_nat + 1

theorem find_geom_seq_n (a1 a_n : ℕ) (h₁ : a1 + a_n = 34) (h₂ : a3 * a_n_2 = 64) (sn : ℕ) (h₃ : sn = 42) : geom_seq_n a1 a3 a_n := 
  by
    sorry

end find_geom_seq_n_l15_15740


namespace valid_N_count_l15_15207

theorem valid_N_count : 
  (∃ n : ℕ, 0 < n ∧ (49 % (n + 3) = 0) ∧ (49 / (n + 3)) % 2 = 1) → 
  (∃ count : ℕ, count = 2) :=
sorry

end valid_N_count_l15_15207


namespace arithmetic_sequence_difference_l15_15595

theorem arithmetic_sequence_difference :
  ∃ (L G : ℝ), ∀ d : ℝ, 
    (75 - 299 * |d| ≥ 20) ∧ (75 + 299 * |d| ≤ 150) → 
    L = 75 - 199 * d ∧ G = 75 + 199 * d ∧
    (G - L = 2 * (10945 / 299)) :=
begin
  sorry
end

end arithmetic_sequence_difference_l15_15595


namespace proof_problem_l15_15681

noncomputable def f (x : ℝ) : ℝ := x^2 * log x

-- Proof definitions
lemma statement_A (x : ℝ) (hx1 : x > 1) : f x > 0 :=
sorry

lemma statement_A' (x : ℝ) (hx2 : 0 < x ∧ x < 1) : f x < 0 :=
sorry

lemma statement_C : set.range f = { y | -1 / (2 * real.exp 1) <= y } :=
sorry

lemma statement_D (x : ℝ) : f x >= x - 1 :=
sorry

-- Combined statement to match the final problem request
theorem proof_problem :
  (∀ x > 1, f x > 0) ∧ (∀ x, 0 < x ∧ x < 1 → f x < 0) ∧ 
  (set.range f = { y | -1 / (2 * real.exp 1) <= y }) ∧ 
  (∀ x, f x >= x - 1) :=
by
  exact ⟨statement_A, statement_A', statement_C, statement_D⟩

end proof_problem_l15_15681


namespace max_temperature_range_l15_15378

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem max_temperature_range : 
  ∃ T1 T2 T3 T4 T5 : ℕ, 
    (T1 = 50) ∧ (T2 = 50) ∧ 
    (T3 ≤ T4) ∧ (T4 ≤ T5) ∧ 
    (T3 = T4 - 1) ∧ (T4 = T5 - 1) ∧ 
    (is_prime T3 ∨ is_prime T4 ∨ is_prime T5) ∧ 
    ((T1 + T2 + T3 + T4 + T5) = 300) ∧ 
    ((T5 - T1) = 18) :=
begin
  sorry
end

end max_temperature_range_l15_15378


namespace min_distance_between_parallel_lines_distance_when_line_parallel_to_x_axis_l15_15653

noncomputable def line_equation (A B C x y : ℝ) : Prop := A * x + B * y + C = 0

noncomputable def point_on_line (x y A B C : ℝ) : Prop := line_equation A B C x y

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  (|C2 - C1|) / (Real.sqrt (A^2 + B^2))

theorem min_distance_between_parallel_lines :
  ∀ (A B C1 C2 x y : ℝ),
  point_on_line x y A B C1 ∧ point_on_line x y A B C2 →
  distance_between_parallel_lines A B C1 C2 = 3 :=
by
  intros A B C1 C2 x y h
  sorry

theorem distance_when_line_parallel_to_x_axis :
  ∀ (x1 x2 y k A B C1 C2 : ℝ),
  k = 3 →
  point_on_line x1 k A B C1 →
  point_on_line x2 k A B C2 →
  |x2 - x1| = 5 :=
by
  intros x1 x2 y k A B C1 C2 hk h1 h2
  sorry

end min_distance_between_parallel_lines_distance_when_line_parallel_to_x_axis_l15_15653


namespace water_added_is_five_l15_15131

theorem water_added_is_five :
  ∃ W x : ℝ, (4 / 3 = 10 / W) ∧ (4 / 5 = 10 / (W + x)) ∧ x = 5 := by
  sorry

end water_added_is_five_l15_15131


namespace no_valid_integer_n_l15_15475

theorem no_valid_integer_n (n : ℕ) (a : ℤ) (d : ℤ) (S_n : ℕ) :
  d = 3 → S_n = 136 → n > 1 → 
  S_n = (n * (2 * a + (n - 1) * d)) / 2 → ¬∃ n > 1, a ∈ ℤ :=
by
  intros h_d h_S_n h_n_gt_1 h_Sn_formula,
  sorry

end no_valid_integer_n_l15_15475


namespace hyperbola_eccentricity_l15_15291

theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
    (h_asymptote : ∀ x, y = x * 3/4 → y^2 / b^2 - x^2 / a^2 = 1) :
    ∃ e, e = 5/4 :=
by
  sorry

end hyperbola_eccentricity_l15_15291


namespace min_c_plus_d_l15_15374

theorem min_c_plus_d (c d : ℤ) (h : c * d = 36) : c + d = -37 :=
sorry

end min_c_plus_d_l15_15374


namespace solve_for_a_l15_15662

def f (x : ℝ) : ℝ := x^2 + 10
def g (x : ℝ) : ℝ := x^2 - 6

theorem solve_for_a (a : ℝ) (h : a > 0) (h1 : f (g a) = 18) : a = Real.sqrt (2 * Real.sqrt 2 + 6) :=
by
  sorry

end solve_for_a_l15_15662


namespace common_point_of_four_convex_sets_l15_15499

open Set

variables {α : Type*} [TopologicalSpace α]

theorem common_point_of_four_convex_sets
  {Φ₁ Φ₂ Φ₃ Φ₄ : Set α} 
  (h₁ : Convex ℝ Φ₁) (h₂ : Convex ℝ Φ₂) (h₃ : Convex ℝ Φ₃) (h₄ : Convex ℝ Φ₄) 
  (hb₁ : Bounded Φ₁) (hb₂ : Bounded Φ₂) (hb₃ : Bounded Φ₃) (hb₄ : Bounded Φ₄)
  (hconn₁₂₃ : (Φ₁ ∩ Φ₂ ∩ Φ₃).Nonempty)
  (hconn₁₂₄ : (Φ₁ ∩ Φ₂ ∩ Φ₄).Nonempty)
  (hconn₁₃₄ : (Φ₁ ∩ Φ₃ ∩ Φ₄).Nonempty)
  (hconn₂₃₄ : (Φ₂ ∩ Φ₃ ∩ Φ₄).Nonempty) :
  (Φ₁ ∩ Φ₂ ∩ Φ₃ ∩ Φ₄).Nonempty :=
sorry

end common_point_of_four_convex_sets_l15_15499


namespace linear_regression_possible_l15_15732

noncomputable def correlation_coefficient (t : List ℝ) (y : List ℝ) : ℝ :=
  let n := t.length
  let t_mean := (t.sum) / n
  let y_mean := (y.sum) / n
  let numerator := List.range n |>.sum (λ i => (t[i] - t_mean) * (y[i] - y_mean))
  let t_variance := (List.range n).sum (λ i => (t[i] - t_mean) ^ 2)
  let y_variance := (List.range n).sum (λ i => (y[i] - y_mean) ^ 2)
  numerator / (Math.sqrt t_variance * Math.sqrt y_variance)

theorem linear_regression_possible
  (t : List ℝ) (y : List ℝ)
  (ht : t = [1, 2, 3, 4, 5])
  (hy : y = [2.4, 2.7, 4.1, 6.4, 7.9])
  (sum_t_y : ∑ i in [0, 1, 2, 3, 4], t[i] * y[i] = 85.2)
  (sqrt_t_var : Math.sqrt 10 ≈ 3.162)
  (sqrt_y_var : Math.sqrt 22.78 ≈ 4.773)
  : correlation_coefficient t y ≈ 0.97 :=
by
  sorry

end linear_regression_possible_l15_15732


namespace least_number_divisible_by_five_primes_l15_15064

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, (n > 0) ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → p ∣ n) ∧ n = 2310 :=
begin
  sorry
end

end least_number_divisible_by_five_primes_l15_15064


namespace building_houses_200_people_l15_15112

theorem building_houses_200_people 
    (num_floors : ℕ)
    (apartments_per_floor : ℕ)
    (people_per_apartment : ℕ) :
    num_floors = 25 →
    apartments_per_floor = 4 →
    people_per_apartment = 2 →
    num_floors * apartments_per_floor * people_per_apartment = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end building_houses_200_people_l15_15112


namespace ball_distance_proof_l15_15138

def initial_height := 120
def rebound_factor := 0.75
def distance_traveled_after_five_bounces := 612.1875

theorem ball_distance_proof :
  let descending_heights := List.scanl (· * rebound_factor) initial_height (List.replicate 4 rebound_factor)
  let ascending_heights := descending_heights.tail
  let total_descending := List.sum descending_heights
  let total_ascending := List.sum ascending_heights
  let total_distance := total_descending + total_ascending
  total_distance = distance_traveled_after_five_bounces := by
  sorry

end ball_distance_proof_l15_15138


namespace sum_of_digits_of_N_l15_15166

-- Define the sequence generating the number N
def sequence_sum (n : ℕ) : ℕ :=
  (List.range n).sum (λ k => 10^(k+1) - 2)

-- Define the number N based on the given series
def N : ℕ := sequence_sum 150

-- State the theorem for the sum of the digits of N
theorem sum_of_digits_of_N :
  (integer.sum_of_digits N) = 148 :=
by
  sorry

end sum_of_digits_of_N_l15_15166


namespace ratio_AO_OK_l15_15654

-- Definitions based on conditions
variables {A B C D O K : Type*}
variables [plane_geometry : plane_geometry ℝ]

-- Parallelogram ABCD and angles
def is_parallelogram [plane_geometry] (A B C D : point) : Prop :=
  A ≠ B ∧ A ≠ D ∧ A + B = C + D 

def angle_value [plane_geometry] (A B C : point) (θ : real) : Prop :=
  ∃ (α : real), α = θ

-- External angle bisector
def external_angle_bisector [plane_geometry] (A B C K : point) : Prop :=
  ∃ (d : direction), K = point_on_angle_bisector d A B C

-- Center of circumcircle (circumcenter) for a triangle
def is_circumcenter [plane_geometry] (O A B D : point) : Prop :=
  ∃ (circ : circle), is_circumcircle circ A B D ∧ O = circ.center

-- Main theorem statement
theorem ratio_AO_OK :
  ∀ (A B C D O K : point), 
    is_parallelogram A B C D →
    angle_value A B D (60 : ℝ) →
    is_circumcenter O A B D →
    external_angle_bisector C D A K →
    distance A O = 2 * distance O K :=
by sorry

end ratio_AO_OK_l15_15654


namespace expand_expression_l15_15186

variable {R : Type _} [CommRing R] (x : R)

theorem expand_expression :
  (3*x^2 + 7*x + 4) * (5*x - 2) = 15*x^3 + 29*x^2 + 6*x - 8 :=
by
  sorry

end expand_expression_l15_15186


namespace vector_subtraction_l15_15352

def a : ℝ × ℝ × ℝ := (1, -2, 1)
def b : ℝ × ℝ × ℝ := (1, 0, 2)

theorem vector_subtraction : a - b = (0, -2, -1) := 
by 
  unfold a b
  simp
  sorry

end vector_subtraction_l15_15352


namespace total_pieces_ten_row_triangle_l15_15943

-- Definitions based on the conditions
def rods (n : ℕ) : ℕ :=
  (n * (2 * 4 + (n - 1) * 5)) / 2

def connectors (n : ℕ) : ℕ :=
  ((n + 1) * (2 * 1 + n * 1)) / 2

def support_sticks (n : ℕ) : ℕ := 
  if n >= 3 then ((n - 2) * (2 * 2 + (n - 3) * 2)) / 2 else 0

-- The theorem stating the total number of pieces is 395 for a ten-row triangle
theorem total_pieces_ten_row_triangle : rods 10 + connectors 10 + support_sticks 10 = 395 :=
by
  sorry

end total_pieces_ten_row_triangle_l15_15943


namespace even_ngon_zero_vector_sum_odd_ngon_finite_zero_vector_sum_points_l15_15220

-- Define that the given n-gon is regular
structure RegularNGon (n : ℕ) :=
(vertices : Fin n → ℝ × ℝ) -- this only captures vertex positioning

-- Definition for the vector sum expression
def signed_vector_sum (M : ℝ × ℝ) {n : ℕ} (vertices : Fin n → ℝ × ℝ) (signs : Fin n → ℤ) : ℝ × ℝ :=
  ∑ i, signs i • (vertices i - M)

-- Part (1) statement
theorem even_ngon_zero_vector_sum (n : ℕ) (hn_even : Even n) (A : RegularNGon n) (M : ℝ × ℝ) :
  ∃ signs : Fin n → ℤ, signed_vector_sum M A.vertices signs = (0, 0) :=
by sorry

-- Part (2) statement
theorem odd_ngon_finite_zero_vector_sum_points (n : ℕ) (hn_odd : Odd n) (A : RegularNGon n) :
  { M : ℝ × ℝ // ∃ signs : Fin n → ℤ, signed_vector_sum M A.vertices signs = (0, 0) }.Finite :=
by sorry

end even_ngon_zero_vector_sum_odd_ngon_finite_zero_vector_sum_points_l15_15220


namespace inequality_solution_set_l15_15324

theorem inequality_solution_set (a b c : ℝ) (h_solution_set : ∀ x : ℝ, ax + b > c ↔ x < 4) : ∀ x : ℝ, a(x-3) + b > c ↔ x < 7 :=
sorry

end inequality_solution_set_l15_15324


namespace max_tan_value_l15_15707

theorem max_tan_value (x₀ : ℝ) (h : ∃ x, ∀ x, 3 * Real.sin x - 4 * Real.cos x ≤ 3 * Real.sin x₀ - 4 * Real.cos x₀):
  Real.tan x₀ = -3 / 4 :=
by 
  sorry

end max_tan_value_l15_15707


namespace regular_star_n_value_l15_15909

theorem regular_star_n_value
  (n : ℕ)
  (congruent_edges : Π i, i < 2 * n → ℝ)
  (A : Π i, i < n → ℝ)
  (B : Π i, i < n → ℝ)
  (angles_condition : ∀ i, i < n → B i - A i = 20)
  (sum_external_angles : 2 * n * (A 0 + B 0) = 360) :
  n = 36 :=
sorry

end regular_star_n_value_l15_15909


namespace find_digit_A_l15_15620

def sum_of_digits_divisible_by_3 (A : ℕ) : Prop :=
  (2 + A + 3) % 3 = 0

theorem find_digit_A (A : ℕ) (hA : sum_of_digits_divisible_by_3 A) : A = 1 ∨ A = 4 :=
  sorry

end find_digit_A_l15_15620


namespace circles_tangent_and_tangent_line_l15_15299

theorem circles_tangent_and_tangent_line :
  let M : ℝ → ℝ → Prop := λ x y, x^2 + y^2 + 2*x - 4*y + 4 = 0
  let N : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 4*x - 12*y + 4 = 0
  -- Circle centers and radii
  let M_center : ℝ × ℝ := (-1, 2)
  let N_center : ℝ × ℝ := (2, 6)
  let rM : ℝ := 1
  let rN : ℝ := 6
  -- Distance between centers
  let d : ℝ := sqrt ((2 + 1)^2 + (6 - 2)^2)
  -- Equation of the common tangent line
  let tangent_line : ℝ → ℝ → Prop := λ x y, 3*x + 4*y = 0
  in (M_center, N_center, rM, rN, d) = ((-1, 2), (2, 6), 1, 6, 5)
  ∧ tangent_line = (λ x y, 3*x + 4*y = 0) :=
by sorry

end circles_tangent_and_tangent_line_l15_15299


namespace least_positive_number_divisible_by_five_smallest_primes_l15_15022

theorem least_positive_number_divisible_by_five_smallest_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  Nat.min (p1 * p2 * p3 * p4 * p5) := 
  2310 :=
by
  sorry

end least_positive_number_divisible_by_five_smallest_primes_l15_15022


namespace num_of_ints_l15_15307

theorem num_of_ints (n : ℤ) (h : -100 < n^3) (h2 : n^3 < 100) : 
    (finset.card (finset.filter (λ x : ℤ, -100 < x^3 ∧ x^3 < 100) (finset.Icc (-4) 4))) = 9 :=
sorry

end num_of_ints_l15_15307


namespace A_investment_l15_15143

theorem A_investment (B_invest C_invest Total_profit A_share : ℝ) 
  (h1 : B_invest = 4200)
  (h2 : C_invest = 10500)
  (h3 : Total_profit = 12100)
  (h4 : A_share = 3630) 
  (h5 : ∀ {x : ℝ}, A_share / Total_profit = x / (x + B_invest + C_invest)) :
  ∃ A_invest : ℝ, A_invest = 6300 :=
by sorry

end A_investment_l15_15143


namespace part1_part2_part3_l15_15999

variable (t : ℝ) (n : ℕ)
-- Condition: Given a sequence {a_n} where a₁ = t and a₂ = t² for t > 0
def seq_a (a : ℕ → ℝ) : Prop :=
  a 1 = t ∧ a 2 = t^2 ∧ (∀ k ≥ 2, a (k + 1) = (t + 1) * a k - t * a (k - 1))

-- Problem (1): Prove that the sequence {a_{n+1} - a_n} is a geometric progression for t ≠ 1
theorem part1 (ht : t ≠ 1) (a : ℕ → ℝ) (h_seq_a : seq_a t a) : 
  ∃ (r : ℝ), ∀ k ≥ 2, a (k + 1) - a k = r * (a k - a (k - 1)) := 
sorry

-- Problem (2): Find the general formula for the sequence {a_n}
theorem part2 (a : ℕ → ℝ) (h_seq_a : seq_a t a) : 
  (t ≠ 1 → ∀ k, a k = t^k) ∧ (t = 1 → ∀ k, a k = 1) :=
sorry

-- Problem (3): Prove the inequality for the sum involving b_n
def seq_b (a b : ℕ → ℝ) : Prop :=
  (∀ k, b k = (2 * a k) / (1 + a k ^ 2)) ∧ (1/2 < t ∧ t < 2)

theorem part3 (a b : ℕ → ℝ) (h_seq_a : seq_a t a) (h_seq_b : seq_b a b) : 
  ∀ k, (∑ i in finset.range (k + 1), (1 / b (i + 1))) < 2^k - 2^(-(↑k / 2)) :=
sorry

end part1_part2_part3_l15_15999


namespace inequality_proof_l15_15532

variables (a b c d e f : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
variable (hcond : |sqrt (a * d) - sqrt (b * c)| ≤ 1)

theorem inequality_proof :
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end inequality_proof_l15_15532


namespace coin_toss_dice_roll_l15_15489

theorem coin_toss_dice_roll :
  let coin_toss := 2 -- two outcomes for same side coin toss
  let dice_roll := 2 -- two outcomes for multiple of 3 on dice roll
  coin_toss * dice_roll = 4 :=
by
  sorry

end coin_toss_dice_roll_l15_15489


namespace paper_rectangle_area_l15_15545

theorem paper_rectangle_area (ABCD: Type) (AB BC AD DC: ℝ)
  (E F : AB → Prop) (on_AB_E on_CD_F: Prop) 
  (BE_CF: Prop) (folds: Prop) (angle_ABpCp_BpEA: Prop)
  (AB_8 : AB = 8) (BE_10 : BE = 10) : 
  ∃ a b c, (a + b * real.sqrt c) ≈ 236.76 := sorry

end paper_rectangle_area_l15_15545


namespace sum_of_50th_terms_l15_15944

open Nat

-- Definition of arithmetic sequence
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Definition of geometric sequence
def geometric_sequence (g₁ r n : ℕ) : ℕ := g₁ * r^(n - 1)

-- Prove the sum of the 50th terms of the given sequences
theorem sum_of_50th_terms : 
  arithmetic_sequence 3 6 50 + geometric_sequence 2 3 50 = 297 + 2 * 3^49 :=
by
  sorry

end sum_of_50th_terms_l15_15944


namespace total_expenditure_eq_9500_l15_15336

def length : ℝ := 20
def width : ℝ := 15
def height : ℝ := 5
def cost_per_square_meter : ℝ := 10

theorem total_expenditure_eq_9500 : 
  let area_floor_ceiling := 2 * (length * width)
  let area_long_walls := 2 * (length * height)
  let area_short_walls := 2 * (width * height)
  let total_area := area_floor_ceiling + area_long_walls + area_short_walls
  let total_expenditure := total_area * cost_per_square_meter
  total_expenditure = 9500 := by 
  sorry

end total_expenditure_eq_9500_l15_15336


namespace inequality_proof_l15_15538

theorem inequality_proof (a b c d e f : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
    (hcond : abs (sqrt (a * d) - sqrt (b * c)) ≤ 1) :
    (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := 
  sorry

end inequality_proof_l15_15538


namespace factorize_square_diff_factorize_common_factor_l15_15961

-- Problem 1: Difference of squares
theorem factorize_square_diff (x : ℝ) : 4 * x^2 - 9 = (2 * x + 3) * (2 * x - 3) := 
by
  sorry

-- Problem 2: Factoring out common terms
theorem factorize_common_factor (a b x y : ℝ) (h : y - x = -(x - y)) : 
  2 * a * (x - y) - 3 * b * (y - x) = (x - y) * (2 * a + 3 * b) := 
by
  sorry

end factorize_square_diff_factorize_common_factor_l15_15961


namespace inequality_proof_l15_15528

noncomputable theory
open real

theorem inequality_proof {a b c d e f : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_d : 0 < d) (h_pos_e : 0 < e) (h_pos_f : 0 < f) (h_ineq : |sqrt(a * d) - sqrt(b * c)| ≤ 1) :
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) :=
sorry

end inequality_proof_l15_15528


namespace intersecting_circles_theorem_l15_15543

noncomputable def exists_line_intersecting_circles (side_length : ℝ) (circumference_sum : ℝ) : Prop :=
  ∀ (square : set (ℝ × ℝ)),
    (∃ circles : list (set (ℝ × ℝ)),
      all circles (λ c, is_circle c ∧ c ⊆ square) ∧
      side_length = 1 ∧
      sum (map circumference circles) = 10) →
    ∃ l : set (ℝ × ℝ), is_line l ∧ intersects_at_least l circles 4

theorem intersecting_circles_theorem : exists_line_intersecting_circles 1 10 :=
sorry

end intersecting_circles_theorem_l15_15543


namespace tank_filled_in_20_minutes_l15_15906

noncomputable def machine_A_rate : ℝ := 1 / 25
noncomputable def machine_B_rate : ℝ := -1 / 50

-- For machine C, calculate the effective rate per minute over a 10-minute cycle
noncomputable def machine_C_fill_rate : ℝ := 1 / 30
noncomputable def machine_C_empty_rate : ℝ := -1 / 45
noncomputable def machine_C_rate_per_10_minutes: ℝ := 5 * machine_C_fill_rate + 5 * machine_C_empty_rate
noncomputable def machine_C_rate_per_minute: ℝ := machine_C_rate_per_10_minutes / 10

noncomputable def combined_rate : ℝ := machine_A_rate + machine_B_rate + machine_C_rate_per_minute

noncomputable def half_tank : ℝ := 1 / 2

theorem tank_filled_in_20_minutes : 
  (half_tank / combined_rate).ceil = 20 :=
by sorry

end tank_filled_in_20_minutes_l15_15906


namespace inequality_solution_l15_15426

theorem inequality_solution (x : ℝ) (h : 3 * x + 4 ≠ 0) : 
  (3 - 2 / (3 * x + 4) < 5) ↔ (x < -(4 / 3) ∨ x > -(5 / 3)) := 
by
  sorry

end inequality_solution_l15_15426


namespace volume_is_zero_l15_15370

-- Define the vectors a and b
def a : ℝ^3 := ⟨3, 7, 1⟩
def b : ℝ^3 := ⟨4, 1, 2⟩

-- The cross product of vectors a and b
def cross_product (u v : ℝ^3) : ℝ^3 :=
  ⟨u.2*v.3 - u.3*v.2, u.3*v.1 - u.1*v.3, u.1*v.2 - u.2*v.1⟩

-- The dot product of vectors u and v
def dot_product (u v : ℝ^3) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- The volume of the parallelepiped defined by vectors a and b
def volume_parallelepiped := |dot_product a (cross_product a b)|

-- The theorem statement
theorem volume_is_zero : volume_parallelepiped = 0 := by
  sorry

end volume_is_zero_l15_15370


namespace differentiable_function_solution_l15_15623

noncomputable def func_solution (a : ℝ) (b : ℝ) (x : ℝ) : ℝ := a^(1 - a/b) * x^(a/b)

theorem differentiable_function_solution (f : ℝ → ℝ) (a : ℝ) (b : ℝ) 
  (h₁ : ∀ x > 0, f x > 0) 
  (h₂ : ∀ x > 0, differentiable_at ℝ f x) 
  (h₃ : ∀ x > 0, deriv f (a / x) = x / f x) : 
  ∀ x > 0, f x = func_solution a b x :=
sorry

end differentiable_function_solution_l15_15623


namespace prob_exactly_M_laws_included_expected_laws_included_l15_15729

variables (K N M : ℕ) (p : ℝ)

-- Definition of the probabilities as given in the conditions and answers
def prob_no_minister_knows_law : ℝ := (1 - p) ^ N
def prob_law_included : ℝ := 1 - prob_no_minister_knows_law p N

-- Part (a)
theorem prob_exactly_M_laws_included :
  (nat.choose K M) * (prob_law_included p N) ^ M * (prob_no_minister_knows_law p N) ^ (K - M) = 
  (nat.choose K M) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) :=
by
  sorry

-- Part (b)
theorem expected_laws_included :
  K * (prob_law_included p N) = K * (1 - (1 - p) ^ N) :=
by
  sorry

end prob_exactly_M_laws_included_expected_laws_included_l15_15729


namespace coeff_x3y3_in_expansion_l15_15627

theorem coeff_x3y3_in_expansion : 
  ∀ (x y : ℝ), coeff (x^3 * y^3) ((x + y) * (2 * x - y)^5) = 40 :=
by
  sorry

end coeff_x3y3_in_expansion_l15_15627


namespace intersection_product_l15_15861

-- Define the two circles using their equations
def circle1 (x y : ℝ) : Prop := x^2 - 4*x + y^2 - 6*y + 9 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 6*y + 21 = 0

-- Theorem stating the desired proof
theorem intersection_product :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y) →
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → x * y = 12) :=
by
  -- Proof omitted; use sorry for now
  intros h exists
  obtain ⟨x, y, h1, h2⟩ := exists
  sorry

end intersection_product_l15_15861


namespace least_positive_whole_number_divisible_by_five_primes_l15_15013

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l15_15013


namespace alyssa_hike_total_distance_l15_15924

theorem alyssa_hike_total_distance
  (e f g h i : ℝ)
  (h1 : e + f + g = 40)
  (h2 : f + g + h = 48)
  (h3 : g + h + i = 54)
  (h4 : e + h = 30) :
  e + f + g + h + i = 118 :=
by
  sorry

end alyssa_hike_total_distance_l15_15924


namespace num_of_ints_l15_15309

theorem num_of_ints (n : ℤ) (h : -100 < n^3) (h2 : n^3 < 100) : 
    (finset.card (finset.filter (λ x : ℤ, -100 < x^3 ∧ x^3 < 100) (finset.Icc (-4) 4))) = 9 :=
sorry

end num_of_ints_l15_15309


namespace max_digit_d_divisible_by_33_l15_15189

theorem max_digit_d_divisible_by_33 (d e : ℕ) (h₀ : 0 ≤ d ∧ d ≤ 9) (h₁ : 0 ≤ e ∧ e ≤ 9) 
  (h₂ : d + e = 4) : d ≤ 4 :=
by {
  sorry
}

example : ∃ d e : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧ d + e = 4 ∧ 
(d = 4) :=
by {
  use [4, 0],
  repeat { split },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial }
}

end max_digit_d_divisible_by_33_l15_15189


namespace shortest_distance_to_circle_l15_15863

theorem shortest_distance_to_circle :
  let x := ℝ, y := ℝ in
    ∃ c : ℝ, (∀ (x y : ℝ), x^2 - 16 * x + y^2 + 14 * y + 145 = 0 ↔ (x - 8)^2 + (y + 7)^2 = 32) → 
      c = Real.sqrt 113 - 4 * Real.sqrt 2 :=
  by
    sorry

end shortest_distance_to_circle_l15_15863


namespace correct_operation_l15_15077

theorem correct_operation : 
  (sqrt 3 * sqrt 5 = sqrt 15) ∧ 
  ¬(sqrt 2 + sqrt 3 = sqrt 5) ∧ 
  ¬(3 * sqrt 3 - sqrt 3 = 3) ∧ 
  ¬(sqrt 24 / sqrt 6 = 4) := by
  -- Introduce the hypotheses
  let A := ¬(sqrt 2 + sqrt 3 = sqrt 5)
  let B := ¬(3 * sqrt 3 - sqrt 3 = 3)
  let C := ¬(sqrt 24 / sqrt 6 = 4)
  let D := (sqrt 3 * sqrt 5 = sqrt 15)
  -- Combine them into a single proposition
  exact ⟨D, A, B, C⟩

-- Proof in this example is not required, so we leave it as sorry if needed.
-- sorry

end correct_operation_l15_15077


namespace z_join_months_after_xy_l15_15523

theorem z_join_months_after_xy
  (x_investment y_investment z_investment total_profit z_share : ℝ)
  (x_months y_months total_months : ℕ) 
  (h1 : x_investment = 36000)
  (h2 : y_investment = 42000)
  (h3 : z_investment = 48000)
  (h4 : total_profit = 13860)
  (h5 : z_share = 4032)
  (h6 : x_months = 12)
  (h7 : y_months = 12)
  (h8 : total_months = 12) :
  ∃ m : ℕ, (total_profit - z_share) / z_share = 
  ((x_investment * x_months + y_investment * y_months) / (z_investment * (total_months - m))) / 
  (m / total_months) ∧ m = 4 :=
begin
  sorry
end

end z_join_months_after_xy_l15_15523


namespace find_remainder_l15_15206
open Nat

def base_seven_digit_sum(n : ℕ) : ℕ := sorry -- Implementation of sum of base-7 digits.

def base_nine_digit_sum(m : ℕ) : ℕ := sorry -- Implementation of sum of base-9 digits.

def i (n : ℕ) : ℕ :=
  base_nine_digit_sum (base_seven_digit_sum n)

def least_n_value_condition : ℕ :=
  find (λ n, (i n).toString(20).toCharArray.any (λ c, c.toDigit > 9)) (range 10000) -- Find n such that i(n) base 20 representation has digit >9

noncomputable def M : ℕ := least_n_value_condition

theorem find_remainder : M % 1000 = 173 := by
  sorry

end find_remainder_l15_15206


namespace least_positive_number_divisible_by_primes_l15_15035

theorem least_positive_number_divisible_by_primes :
  ∃ n : ℕ, n > 0 ∧
    (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧
    (∀ m : ℕ, (m > 0 ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m)) → n ≤ m) ∧
    n = 2310 :=
by
  sorry

end least_positive_number_divisible_by_primes_l15_15035


namespace minimum_people_for_couple_minimum_people_for_same_gender_l15_15434

theorem minimum_people_for_couple {people : Set (ℕ × ℕ)} (h : people.card = 20) (couples : Set (ℕ × ℕ)) (hc : couples.card = 10)
  (each_person_in_couple : ∀ p ∈ people, ∃ c ∈ couples, p ∈ c) : 
  ∃ n, n = 13 ∧ (∀ group ∈ people.powerset, group.card ≥ 13 → ∃ c ∈ couples, c.1 ∈ group ∧ c.2 ∈ group) :=
sorry

theorem minimum_people_for_same_gender {people : Set (ℕ × bool)} (h : people.card = 20) (males females : Set ℕ) (hm : males.card = 10) (hf : females.card = 10)
  (each_person_gender : ∀ p ∈ people, (p.2 = tt → p.1 ∈ males) ∧ (p.2 = ff → p.1 ∈ females)) :
  ∃ n, n = 3 ∧ (∀ group ∈ people.powerset, group.card ≥ 3 → ∃ p₁ p₂ ∈ group, p₁.2 = p₂.2) :=
sorry

end minimum_people_for_couple_minimum_people_for_same_gender_l15_15434


namespace binomial_arithmetic_sequence_l15_15098

theorem binomial_arithmetic_sequence (n : ℕ) (h : n > 3)
  (C : ℕ → ℕ → ℕ)
  (hC1 : C n 1 = n)
  (hC2 : C n 2 = n * (n - 1) / 2)
  (hC3 : C n 3 = n * (n - 1) * (n - 2) / 6) :
  C n 2 - C n 1 = C n 3 - C n 2 → n = 7 := sorry

end binomial_arithmetic_sequence_l15_15098


namespace diameter_of_lid_is_2_inches_l15_15179

noncomputable def π : ℝ := 3.14
def C : ℝ := 6.28

theorem diameter_of_lid_is_2_inches (d : ℝ) : d = C / π → d = 2 :=
by
  intro h
  sorry

end diameter_of_lid_is_2_inches_l15_15179


namespace problem_I_problem_II_l15_15238

noncomputable def f (x a : ℝ) : ℝ := 2 * (Real.cos x)^2 + a * Real.sin (2 * x) + 1

-- Define the problem statement
theorem problem_I :
  (f (Real.pi / 3) a = 0) -> (a = -Real.sqrt 3) :=
sorry

theorem problem_II :
  ∀ k : ℤ, ∀ x : ℝ, (f x a = 2 * Real.sin(2 * x + 5 * Real.pi / 6) + 2) ->
  (a = -Real.sqrt 3) ->
  (k * Real.pi - 2 * Real.pi / 3 < x ∧ x < k * Real.pi - Real.pi / 6) :=
sorry

end problem_I_problem_II_l15_15238


namespace fraction_irreducible_l15_15418

theorem fraction_irreducible (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by 
  sorry

end fraction_irreducible_l15_15418


namespace seq_100th_term_is_981_l15_15229

def sequence_term (n : Nat) : Nat :=
  if n == 0 then 0 else
    let binary_repr := Nat.toDigits 2 (n + 1)
    binary_repr.foldl (λ acc bit, acc * 3 + bit) 0

theorem seq_100th_term_is_981 : sequence_term 100 = 981 := 
by
  sorry

end seq_100th_term_is_981_l15_15229


namespace smallest_positive_period_range_of_m_range_of_ratio_l15_15673

-- Definition of the function f
def f (x : ℝ) : ℝ := 2 * Math.sin (x + Real.pi / 3) + Math.sin x * Math.cos x - Real.sqrt 3 * Math.sin x ^ 2

-- (1) Prove that the smallest positive period of f(x) is π
theorem smallest_positive_period :
  ∃ p > 0, ∀ x, f (x + p) = f x ∧ ∀ q > 0, (∀ x, f (x + q) = f x) → p ≤ q :=
sorry

-- (2) Prove that for given conditions, m must be in the range (-∞, -2] ∪ [1, ∞)
theorem range_of_m (x_0 : ℝ) (m : ℝ) (hx0 : 0 ≤ x_0 ∧ x_0 ≤ 5 * Real.pi / 12) 
  (hmf : m * f x_0 - 2 = 0) :
  m ∈ set.Iic (-2) ∪ set.Ici 1 :=
sorry

-- (3) Prove that the range is (√2 / 2, 2√3 / 3)
theorem range_of_ratio (A B C : ℝ) (hB : B = 2 * A) (h_angle : 0 < A ∧ A < (Real.pi / 2)) :
  let num := f (C / 2 - Real.pi / 6)
  let denom := f (B / 2 - Real.pi / 6)
  2 * Math.cos A - 1 / (2 * Math.cos A) ∈ set.Ioo (Real.sqrt 2 / 2) (2 * Real.sqrt 3 / 3) :=
sorry

end smallest_positive_period_range_of_m_range_of_ratio_l15_15673


namespace least_positive_whole_number_divisible_by_five_primes_l15_15015

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l15_15015


namespace B_div_A_75_l15_15458

noncomputable def find_ratio (A B : ℝ) (x : ℝ) :=
  (A / (x + 3) + B / (x * (x - 9)) = (x^2 - 3*x + 15) / (x * (x + 3) * (x - 9)))

theorem B_div_A_75 :
  ∀ (A B : ℝ), (∀ (x : ℝ), x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 9 → find_ratio A B x) → 
  B/A = 7.5 :=
by
  sorry

end B_div_A_75_l15_15458


namespace range_of_m_l15_15234

theorem range_of_m (m : ℝ) 
  (p : m < 0) 
  (q : ∀ x : ℝ, x^2 + m * x + 1 > 0) : 
  -2 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l15_15234


namespace number_of_associates_l15_15124

theorem number_of_associates
  (num_managers : ℕ) 
  (avg_salary_managers : ℝ) 
  (avg_salary_associates : ℝ) 
  (avg_salary_company : ℝ)
  (total_employees : ℕ := num_managers + A) -- Adding a placeholder A for the associates
  (total_salary_company : ℝ := (num_managers * avg_salary_managers) + (A * avg_salary_associates)) 
  (average_calculation : avg_salary_company = total_salary_company / total_employees) :
  ∃ A : ℕ, A = 75 :=
by
  let A : ℕ := 75
  sorry

end number_of_associates_l15_15124


namespace line_equation_l15_15945

theorem line_equation (t : ℝ) : ∃ (x y : ℝ), x = 3 * t + 5 ∧ y = 6 * t - 8 ∧ y = 2 * x - 18 :=
by
  use 3 * t + 5, 6 * t - 8
  split
  . exact ⟨rfl, ⟨rfl, sorry⟩⟩

end line_equation_l15_15945


namespace repeating_decimal_eq_l15_15082

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : (0.565656...) = repeating_decimal :=
sorry

end repeating_decimal_eq_l15_15082


namespace least_number_divisible_by_five_smallest_primes_l15_15005

theorem least_number_divisible_by_five_smallest_primes : 
  ∃ n ∈ ℕ+, n = 2 * 3 * 5 * 7 * 11 ∧ n = 2310 :=
by
  sorry

end least_number_divisible_by_five_smallest_primes_l15_15005


namespace evaluate_expression_l15_15619

theorem evaluate_expression : 2 + (1 / (2 + (1 / (2 + 2)))) = 22 / 9 := by
  sorry

end evaluate_expression_l15_15619


namespace PQ_parallel_to_AB_l15_15818

variables (O A B C D E P Q : Type) [CircumscribedQuadrilateral ABCD] 

-- Given conditions as assumptions
variable (h1 : Center O of Circumcircle ABCD ∧ O lies_on AB)
variable (h2 : Reflection E of D across AB)
variable (h3 : Intersection P of AC and DO)
variable (h4 : Intersection Q of BD and CE)

theorem PQ_parallel_to_AB (h1 : Center O of Circumcircle ABCD ∧ O lies_on AB)
                         (h2 : Reflection E of D across AB)
                         (h3 : Intersection P of AC and DO)
                         (h4 : Intersection Q of BD and CE) : PQ ∥ AB :=
sorry


end PQ_parallel_to_AB_l15_15818


namespace kendra_change_is_correct_l15_15466

-- Define the initial conditions
def price_wooden_toy : ℕ := 20
def price_hat : ℕ := 10
def kendra_initial_money : ℕ := 100
def num_wooden_toys : ℕ := 2
def num_hats : ℕ := 3

-- Calculate the total costs
def total_wooden_toys_cost : ℕ := price_wooden_toy * num_wooden_toys
def total_hats_cost : ℕ := price_hat * num_hats
def total_cost : ℕ := total_wooden_toys_cost + total_hats_cost

-- Calculate the change Kendra received
def kendra_change : ℕ := kendra_initial_money - total_cost

theorem kendra_change_is_correct : kendra_change = 30 := by
  sorry

end kendra_change_is_correct_l15_15466


namespace maxProfitAchievable_l15_15554

namespace BarrelProduction

structure ProductionPlan where
  barrelsA : ℕ
  barrelsB : ℕ

def profit (plan : ProductionPlan) : ℕ :=
  300 * plan.barrelsA + 400 * plan.barrelsB

def materialAUsage (plan : ProductionPlan) : ℕ :=
  plan.barrelsA + 2 * plan.barrelsB

def materialBUsage (plan : ProductionPlan) : ℕ :=
  2 * plan.barrelsA + plan.barrelsB

def isValidPlan (plan : ProductionPlan) : Prop :=
  materialAUsage plan ≤ 12 ∧ materialBUsage plan ≤ 12

def maximumProfit : ℕ :=
  2800

theorem maxProfitAchievable : 
  ∃ (plan : ProductionPlan), isValidPlan plan ∧ profit plan = maximumProfit :=
sorry

end BarrelProduction

end maxProfitAchievable_l15_15554


namespace least_positive_number_divisible_by_primes_l15_15028

theorem least_positive_number_divisible_by_primes :
  ∃ n : ℕ, n > 0 ∧
    (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧
    (∀ m : ℕ, (m > 0 ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m)) → n ≤ m) ∧
    n = 2310 :=
by
  sorry

end least_positive_number_divisible_by_primes_l15_15028


namespace amy_pencil_count_l15_15592

theorem amy_pencil_count (initial_pencils total_pencils pencils_bought : ℕ) 
  (h1 : initial_pencils = 3) 
  (h2 : total_pencils = 10) 
  (h3 : total_pencils = initial_pencils + pencils_bought) : 
  pencils_bought = 7 :=
by
  rw [h1, h2] at h3
  linarith

#reduce amy_pencil_count 3 10 _ rfl rfl rfl -- This is to verify by reducing the theorem

end amy_pencil_count_l15_15592


namespace third_derivative_l15_15197

noncomputable def y (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative (x : ℝ) : (iterated_deriv 3 y) x = 4 / (1 + x^2)^2 :=
by
  sorry

end third_derivative_l15_15197


namespace number_of_students_like_photography_l15_15580

variable (n_dislike n_like n_neutral : ℕ)

theorem number_of_students_like_photography :
  (3 * n_dislike = n_dislike + 12) →
  (5 * n_dislike = n_like) →
  n_like = 30 :=
by
  sorry

end number_of_students_like_photography_l15_15580


namespace find_values_of_cubes_l15_15754

def N (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![a, c, b], ![c, b, a], ![b, a, c]]

theorem find_values_of_cubes (a b c : ℂ) (h1 : (N a b c) ^ 2 = 1) (h2 : a * b * c = 1) :
  a^3 + b^3 + c^3 = 2 ∨ a^3 + b^3 + c^3 = 4 :=
by
  sorry

end find_values_of_cubes_l15_15754


namespace range_of_set_X_l15_15793

theorem range_of_set_X 
  (X : Set ℕ) 
  (subsetX : X = {3, 11, 7, a, 17, 19})
  (prime_a : Nat.Prime a)
  (Y : ℕ) 
  (def_Y : Y = ∏ i in X, i)
  (even_11Y : 11 * Y % 2 = 0) 
  : 19 - 2 = 17 :=
by
  -- Proof is skipped which is indicated by sorry
  sorry

end range_of_set_X_l15_15793


namespace inequality_solution_l15_15625

open Real

theorem inequality_solution (x : ℝ) : 
  (1 / 3) + | x - (11 / 48) | < (1 / 2) ↔ (1 / 16) < x ∧ x < (19 / 48) :=
by
  sorry

end inequality_solution_l15_15625


namespace solve_for_y_l15_15071

theorem solve_for_y (y : ℚ) :
  (40 / 70 : ℚ) ^ 2 = (sqrt (y / 70 : ℚ)) → y = 17920 / 2401 :=
by
  intro h
  sorry

end solve_for_y_l15_15071


namespace find_a_for_f_f_a_eq_1_l15_15644

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 - x else Real.log x

theorem find_a_for_f_f_a_eq_1 (a : ℝ) :
  f (f a) = 1 → (a = 1 ∨ a = Real.exp 1) := 
sorry

end find_a_for_f_f_a_eq_1_l15_15644


namespace recommendation_methods_count_l15_15735

theorem recommendation_methods_count :
  let females := 3
  let males := 2
  let spots := 5
  let japanese_spots := 2
  let korean_spots := 2
  let russian_spots := 1
  (females * (females - 1) / 2) * (3!) - 2 * (3!) = 24 := 
by
  let females := 3
  let males := 2
  let spots := 5
  let japanese_spots := 2
  let korean_spots := 2
  let russian_spots := 1
  -- Correct mathematical computation:
  have h1 : (3 * 2 * 3!) - 2 * 3! = 24 := by sorry
  exact h1

end recommendation_methods_count_l15_15735


namespace roots_approximate_solution_l15_15162

-- Define the coefficients of the quadratic equation
def a : ℝ := Real.sqrt 3
def b : ℝ := Real.sqrt 17
def c : ℝ := - Real.sqrt 6

-- Define the equation
def equation (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the approximate roots
def root1_approx : ℝ := 0.492
def root2_approx : ℝ := -2.873

-- We need to prove that these are approximate roots
theorem roots_approximate_solution :
  ∃ x1 x2 : ℝ, equation x1 ∧ equation x2 ∧
  |x1 - root1_approx| < 0.001 ∧ |x2 - root2_approx| < 0.001 := 
sorry

end roots_approximate_solution_l15_15162


namespace find_b_find_vertex_compare_y1_y2_l15_15687

-- Condition that the parabola passes through the point (-1,0)
def parabola_passing_through (b : ℝ) : Prop :=
  let y : ℝ → ℝ := λ x, -x^2 + b * x - 3 in
  y (-1) = 0

-- Question 1: Find the value of b
theorem find_b (b : ℝ) (h : parabola_passing_through b) : b = -4 :=
sorry

-- Question 2: Find the coordinates of the vertex
def is_vertex (x y : ℝ) : Prop :=
  y = -(x + 2)^2 + 1

theorem find_vertex (b : ℝ) (h : b = -4) : is_vertex (-2) 1 :=
sorry

-- Question 3: Compare y_1 and y_2
def parabola (x : ℝ) : ℝ := -x^2 - 4 * x - 3

theorem compare_y1_y2 : parabola 1 < parabola (-1) :=
sorry

end find_b_find_vertex_compare_y1_y2_l15_15687


namespace cosine_of_A_l15_15327

theorem cosine_of_A (a b : ℝ) (A B : ℝ) (h1 : b = (5 / 8) * a) (h2 : A = 2 * B) :
  Real.cos A = 7 / 25 :=
by
  sorry

end cosine_of_A_l15_15327


namespace find_share_of_C_l15_15173

-- Definitions and assumptions
def share_in_ratio (x : ℕ) : Prop :=
  let a := 2 * x
  let b := 3 * x
  let c := 4 * x
  a + b + c = 945

-- Statement to prove
theorem find_share_of_C :
  ∃ x : ℕ, share_in_ratio x ∧ 4 * x = 420 :=
by
  -- We skip the proof here.
  sorry

end find_share_of_C_l15_15173


namespace bills_difference_l15_15957

noncomputable def Mike_tip : ℝ := 5
noncomputable def Joe_tip : ℝ := 10
noncomputable def Mike_percentage : ℝ := 20
noncomputable def Joe_percentage : ℝ := 25

theorem bills_difference
  (m j : ℝ)
  (Mike_condition : (Mike_percentage / 100) * m = Mike_tip)
  (Joe_condition : (Joe_percentage / 100) * j = Joe_tip) :
  |m - j| = 15 :=
by
  sorry

end bills_difference_l15_15957


namespace syllogism_problem_l15_15803

theorem syllogism_problem (P: Type) (Q: Type) (R: Type)
  (h₁: ∀ x: P, R x)   -- statement ③
  (h₂: ∃ x: Q, R x)   -- statement ①
  (h₃: ∃ x: Q, P x)   -- statement ②
  : (h₁ ↔ h₃) ∧ (h₂ ↔ h₃) := 
by
  -- Prove the major premise and minor premise equivalence.
  sorry

end syllogism_problem_l15_15803


namespace probability_third_smallest_is_4_l15_15414

theorem probability_third_smallest_is_4 :
  (∃ (integers : Finset ℕ), integers.card = 7 ∧ integers ⊆ (Finset.range 13).erase 0 ∧ 
  ∃ (S : Finset ℕ), S = (Finset.filter (λ x, x < 4) integers) ∧ S.card = 2 ∧ 
  ∃ (T : Finset ℕ), T = (Finset.filter (λ x, 4 < x) integers) ∧ T.card = 5) → 
  let total_ways := Nat.choose 12 7 in
  let favorable_ways := (Nat.choose 3 2) * (Nat.choose 8 5) in
  (favorable_ways) / total_ways.toReal = 7 / 33 :=
by sorry

end probability_third_smallest_is_4_l15_15414


namespace inequality_solution_l15_15424

theorem inequality_solution (x : ℝ) :
  (x < -2 ∨ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ (x > 8)) ↔
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 20) :=
sorry

end inequality_solution_l15_15424


namespace enclosed_area_of_curve_l15_15820

def circular_arc_length := (2 * Real.pi) / 3
def hexagon_side := 3
def radius_of_arcs := 1
def number_of_arcs := 9

theorem enclosed_area_of_curve :
  let hexagon_area := (3 * Real.sqrt 3 / 2) * (hexagon_side ^ 2)
  let sector_area := (circular_arc_length / (2 * Real.pi)) * Real.pi * (radius_of_arcs ^ 2)
  let total_sector_area := number_of_arcs * sector_area
  (hexagon_area + total_sector_area = 13.5 * Real.sqrt 3 + Real.pi) :=
begin
  sorry
end

end enclosed_area_of_curve_l15_15820


namespace least_positive_divisible_by_five_primes_l15_15043

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l15_15043


namespace prob_exactly_M_laws_in_concept_expected_laws_in_concept_l15_15722

section Anchuria
variables (K N M : ℕ) (p : ℝ)

-- Part (a): Define P_M as the binomial probability distribution result
def probability_exactly_M_laws : ℝ :=
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M)

-- Part (a): Prove the result for the probability that exactly M laws are included in the Concept
theorem prob_exactly_M_laws_in_concept :
  probability_exactly_M_laws K N M p =
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) := 
sorry

-- Part (b): Define the expected number of laws included in the Concept
def expected_number_of_laws : ℝ :=
  K * (1 - (1 - p) ^ N)

-- Part (b): Prove the result for the expected number of laws included in the Concept
theorem expected_laws_in_concept :
  expected_number_of_laws K N p = K * (1 - (1 - p) ^ N) :=
sorry
end Anchuria

end prob_exactly_M_laws_in_concept_expected_laws_in_concept_l15_15722


namespace no_7_flip_integer_l15_15766

theorem no_7_flip_integer : ¬ ∃ (n : ℕ), let m := n * 7 in (digits n).reverse = digits m := 
sorry

namespace Mathlib
  -- Auxiliary functions definitions below
  def digits (n : ℕ) : list ℕ :=
    if n = 0 then [] else (nat.mod n 10) :: digits (n / 10)

  def list.reverse {α : Type*} : list α → list α
  | []        := []
  | (h :: t) := (reverse t).append [h]
end Mathlib

end no_7_flip_integer_l15_15766


namespace rectangular_prism_diagonals_l15_15564

theorem rectangular_prism_diagonals
  (num_vertices : ℕ) (num_edges : ℕ)
  (h1 : num_vertices = 12) (h2 : num_edges = 18) :
  (total_diagonals : ℕ) → total_diagonals = 20 :=
by
  sorry

end rectangular_prism_diagonals_l15_15564


namespace most_appropriate_sampling_method_l15_15900

def lab1_mice : ℕ := 18
def lab2_mice : ℕ := 24
def lab3_mice : ℕ := 54
def lab4_mice : ℕ := 48
def total_mice_needed : ℕ := 24

def method_d (lab1_samples lab2_samples lab3_samples lab4_samples : ℕ) : Prop :=
  lab1_samples = 3 ∧ lab2_samples = 4 ∧ lab3_samples = 9 ∧ lab4_samples = 8 ∧ 
  lab1_samples + lab2_samples + lab3_samples + lab4_samples = total_mice_needed

theorem most_appropriate_sampling_method :
  method_d 3 4 9 8 :=
by
  repeat { constructor }
  all_goals { refl }
  sorry

end most_appropriate_sampling_method_l15_15900


namespace probability_neither_event_l15_15882

open ProbabilityTheory

variables (Ω : Type)
variables (P : ProbabilityMassFunction Ω)

variables (A B : Event Ω)
variables (hA : P A = 0.15)
variables (hB : P B = 0.40)
variables (hAB : P (A ∩ B) = 0.15)

theorem probability_neither_event : P (Aᶜ ∩ Bᶜ) = 0.60 := by
  have hAorB : P (A ∪ B) = P A + P B - P (A ∩ B) := 
    ProbabilityMassFunction.prob_union_add_inter A B
  rw [hA, hB, hAB] at hAorB
  have hAorB_value : P (A ∪ B) = 0.40 := by linarith
  have hComplement : P (Aᶜ ∩ Bᶜ) = 1 - P (A ∪ B) := 
    ProbabilityMassFunction.prob_compl_union_eq A B
  rw [hAorB_value] at hComplement
  linarith

end probability_neither_event_l15_15882
