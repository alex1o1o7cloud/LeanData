import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Combinatorics
import Mathlib.Algebra.Field
import Mathlib.Algebra.Fractional
import Mathlib.Algebra.GeomSeq
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry;
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Eulerian
import Mathlib.Combinatorics.Perm
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.MeasureTheory.ProbabilityMassFunction
import Mathlib.Order.Floor
import Mathlib.Tactic

namespace lemon_count_l255_255447

theorem lemon_count {total_fruits mangoes pears pawpaws : ℕ} (kiwi lemon : ℕ) :
  total_fruits = 58 ∧ 
  mangoes = 18 ∧ 
  pears = 10 ∧ 
  pawpaws = 12 ∧ 
  (kiwi = lemon) →
  lemon = 9 :=
by 
  sorry

end lemon_count_l255_255447


namespace semicircle_perimeter_approx_l255_255955

noncomputable theory
open_locale real

def semicircle_perimeter (r : ℝ) := (real.pi * r) + (2 * r)

theorem semicircle_perimeter_approx (r : ℝ) (h : r = 20) :
  semicircle_perimeter r ≈ 102.83 :=
by
  rw [← h, semicircle_perimeter]
  norm_num
  sorry

end semicircle_perimeter_approx_l255_255955


namespace reach_3_3_in_8_steps_l255_255441

def moves : List (Int × Int) := 
  [(1, 0), (-1, 0), (0, 1), (0, -1)]

noncomputable def probability_reach (start : Int × Int) (goal : Int × Int) (steps : Nat) : ℚ :=
  (175 : ℚ) / 8192

theorem reach_3_3_in_8_steps :
  let m := 175
  let n := 8192
  let p : ℚ := probability_reach (0, 0) (3, 3) 8
  p = (m : ℚ) / n ∧ Nat.gcd m n = 1 ∧ m + n = 8367 :=
by
  sorry

end reach_3_3_in_8_steps_l255_255441


namespace sum_of_extrema_on_interval_l255_255252

def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 2*x + 3

theorem sum_of_extrema_on_interval :
  let a := sup (set.image quadratic_function (set.Icc 0 4))
  let b := inf (set.image quadratic_function (set.Icc 0 4))
  a + b = 13 :=
by
  sorry

end sum_of_extrema_on_interval_l255_255252


namespace probability_of_different_topics_l255_255954

theorem probability_of_different_topics (n : ℕ) (m : ℕ) (prob : ℚ)
  (h1 : n = 36)
  (h2 : m = 30)
  (h3 : prob = 5/6) :
  (m : ℚ) / (n : ℚ) = prob :=
sorry

end probability_of_different_topics_l255_255954


namespace range_of_x_l255_255317

theorem range_of_x (x : ℝ) : 
  sqrt ((x - 1) / (x - 2)) ≥ 0 → (x > 2 ∨ x ≤ 1) :=
by
  sorry

end range_of_x_l255_255317


namespace expected_no_advice_l255_255600

theorem expected_no_advice {n : ℕ} (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  (Σ (j : ℕ) (hj : j < n), (1 - p)^j) = (1 - (1 - p)^n) / p :=
by
  sorry

end expected_no_advice_l255_255600


namespace parallelogram_area_l255_255068

theorem parallelogram_area (b h : ℕ) (hb : b = 20) (hh : h = 4) :
    b * h = 80 := by
  subst hb
  subst hh
  exact Nat.mul_comm 20 4 ▸ rfl

-- The following constants specify the base and height.
def base := 20
def height := 4

-- Now we want to state and prove the theorem that the area is 80.
example : base * height = 80 := 
by 
  exact parallelogram_area base height rfl rfl

end parallelogram_area_l255_255068


namespace range_of_function_l255_255313

theorem range_of_function (x : ℝ) : 
    (sqrt ((x - 1) / (x - 2))) = sqrt ((x - 1) / (x - 2)) → (x ≤ 1 ∨ x > 2) :=
by
  sorry

end range_of_function_l255_255313


namespace visual_range_percent_increase_l255_255081

-- Define the original and new visual ranges
def original_range : ℝ := 90
def new_range : ℝ := 150

-- Define the desired percent increase as a real number
def desired_percent_increase : ℝ := 66.67

-- The theorem to prove that the visual range is increased by the desired percentage
theorem visual_range_percent_increase :
  ((new_range - original_range) / original_range) * 100 = desired_percent_increase := 
sorry

end visual_range_percent_increase_l255_255081


namespace shopkeeper_profit_percent_l255_255956

theorem shopkeeper_profit_percent 
  (selling_price : ℝ) (cost_price : ℝ) (profit : ℝ) (profit_percent : ℝ)
  (h1 : selling_price = 2524.36) (h2 : cost_price = 2400)
  (h3 : profit = selling_price - cost_price) 
  (h4 : profit_percent = (profit / cost_price) * 100) : 
  profit_percent ≈ 5.18 :=
by {
  sorry
}

end shopkeeper_profit_percent_l255_255956


namespace prove_angle_PMN_l255_255308

noncomputable def geometry_problem : Prop :=
  ∀ (P Q R M N : Type) [metric_space P] [metric_space Q] [metric_space R] [metric_space M] [metric_space N]
  (angle_PQR angle_RPQ : ℝ) (equal_sides_PMR_PM angle_MPN univ_angle : ℝ),
  angle_PQR = 60 ∧
  equal_sides_PMR_PM = angle_PQR ∧
  angle_MPN = angle_RPQ ∧
  univ_angle = 180 ->
  angle_RPQ = angle_PQR ∧
  angle_MPN = univ_angle - angle_PQR - 120 ∧
  equal_sides_PMR_PM - (univ_angle - angle_MPN) = 60

theorem prove_angle_PMN : geometry_problem :=
sorry

end prove_angle_PMN_l255_255308


namespace find_f_neg4_l255_255232

variables {R : Type*} [LinearOrderedField R] (f : R → R)

def odd_function (g : R → R) : Prop :=
  ∀ (x : R), g (-x) = -g (x)

theorem find_f_neg4
  (h1 : odd_function (λ x, f (x - 1)))
  (h2 : f 2 = 1) :
  f (-4) = -1 :=
sorry

end find_f_neg4_l255_255232


namespace only_D_is_odd_and_increasing_l255_255975

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f (x) < f (y)

def A (x : ℝ) : ℝ := log (x^3)
def B (x : ℝ) : ℝ := -x^2
def C (x : ℝ) : ℝ := 1/x
def D (x : ℝ) : ℝ := x * abs x

theorem only_D_is_odd_and_increasing :
  is_odd D ∧ is_increasing D ∧
  ¬ ∃ f, f ∈ {A, B, C} ∧ (is_odd f ∧ is_increasing f) :=
by
  sorry

end only_D_is_odd_and_increasing_l255_255975


namespace max_third_side_triangle_l255_255019

theorem max_third_side_triangle (P Q R : ℝ) (a b c : ℝ) 
  (h_angles : P + Q + R = π)
  (h_sides : {a, b, c} = {7, 24, x}) :
  (cos (2 * P) + cos (2 * Q) + cos (2 * R) = 1) →
  max (max a b) c = 25 :=
sorry

end max_third_side_triangle_l255_255019


namespace min_odd_in_A_P_l255_255381

-- Define the polynomial P of degree 8
variable {R : Type*} [Ring R] (P : Polynomial R)
variable (hP : degree P = 8)

-- Define the set A_P for some value c in R
def A_P (c : R) : Set R := { x | P.eval x = c }

-- Given that the number 8 is included in the set A_P
variable (c : R) (h8 : (8 : R) ∈ A_P c)

-- Prove that the minimum number of odd numbers in the set A_P is 1
theorem min_odd_in_A_P : ∃ x ∈ A_P c, (x % 2 = 1) := sorry

end min_odd_in_A_P_l255_255381


namespace largest_n_for_98_99_100_factorial_l255_255672

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def divides (x n y : ℕ) : Prop :=
  (x^n ∣ y) ∧ ¬ (x^(n + 1) ∣ y)

theorem largest_n_for_98_99_100_factorial : 
  ∃ n : ℕ, divides 5 n (factorial 98 + factorial 99 + factorial 100) ∧ n = 26 := 
sorry

end largest_n_for_98_99_100_factorial_l255_255672


namespace solution_set_inequality_l255_255871

theorem solution_set_inequality (x : ℝ) : (2 * x : ℝ) / (x + 2) ≤ 3 ↔ x ∈ set.Iic (-6) ∪ set.Ioi (-2) := sorry

end solution_set_inequality_l255_255871


namespace seq_a_plus_one_geometric_sum_a_n_b_n_l255_255402

variables {ℕ : Type*}

def a_seq (n : ℕ) : ℕ :=
if n = 1 then 2 else 3 * a_seq (n - 1) + 2

def b_seq (n : ℕ) : ℕ :=
Real.log (a_seq n + 1) / Real.log 3

def S_n (n : ℕ) : ℕ :=
(n * (3 ^ n - 1))

theorem seq_a_plus_one_geometric (n : ℕ) (h : n ≥ 2) :
  ∃ r a, (∀ m, m ≥ 2 → a_seq m + 1 = r ^ (m - 1) * a) :=
sorry

theorem sum_a_n_b_n (n : ℕ) :
  ∑ i in finset.range n, a_seq i.succ * b_seq i.succ =
  (2 * n - 1) / 4 * 3 ^ (n + 1) + 3 / 4 - n * (n + 1) / 2 :=
sorry

end seq_a_plus_one_geometric_sum_a_n_b_n_l255_255402


namespace male_contestants_count_l255_255562

-- Define the total number of contestants.
def total_contestants : ℕ := 18

-- A third of the contestants are female.
def female_contestants : ℕ := total_contestants / 3

-- The rest are male.
def male_contestants : ℕ := total_contestants - female_contestants

-- The theorem to prove
theorem male_contestants_count : male_contestants = 12 := by
  -- Proof goes here.
  sorry

end male_contestants_count_l255_255562


namespace distribution_methods_l255_255482

theorem distribution_methods (books students : ℕ) (h_books : books = 5) (h_students : students = 3) :
  (∃ f : Fin books → Fin students, ∀ i : Fin students, ∃ b : Fin books, f b = i) → 
  (finset.univ.powerset.filter (λ s, s.card > 0).card = 150) :=
by
  sorry

end distribution_methods_l255_255482


namespace monotonicity_when_a_is_one_range_of_a_for_two_zeros_l255_255714

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * (x + 2)

theorem monotonicity_when_a_is_one :
  (∀ x < 0, deriv (f x 1) < 0) ∧ (∀ x > 0, deriv (f x 1) > 0) :=
by
  sorry

theorem range_of_a_for_two_zeros :
  (∀ a > 1 / Real.exp, ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) ∧
  (∀ a ≤ 1 / Real.exp, ∀ x, f x a ≠ 0) :=
by
  sorry

end monotonicity_when_a_is_one_range_of_a_for_two_zeros_l255_255714


namespace min_odd_in_A_P_l255_255385

-- Define the polynomial P of degree 8
variable {R : Type*} [Ring R] (P : Polynomial R)
variable (hP : degree P = 8)

-- Define the set A_P for some value c in R
def A_P (c : R) : Set R := { x | P.eval x = c }

-- Given that the number 8 is included in the set A_P
variable (c : R) (h8 : (8 : R) ∈ A_P c)

-- Prove that the minimum number of odd numbers in the set A_P is 1
theorem min_odd_in_A_P : ∃ x ∈ A_P c, (x % 2 = 1) := sorry

end min_odd_in_A_P_l255_255385


namespace rect_sections_with_5_lines_l255_255303

theorem rect_sections_with_5_lines : 
  ∀(n : ℕ), (∑ i in finset.range (n + 1), i + 1) = 16 :=
by
  sorry

end rect_sections_with_5_lines_l255_255303


namespace direction_vector_of_line_l255_255456

theorem direction_vector_of_line (x y : ℝ) : (x + 1 = 0) → (0, 1) = (0 : ℝ, 1 : ℝ) :=
by
  intro h.
  rw h.
  exact rfl.

end direction_vector_of_line_l255_255456


namespace minimum_odd_in_A_P_l255_255380

-- Define the polynomial and its properties
def polynomial_degree8 (P : ℝ[X]) : Prop :=
  P.degree = 8

-- Define the set A_P
def A_P (P : ℝ[X]) : set ℝ :=
  {x | P.eval x = 8}

-- Main theorem statement
theorem minimum_odd_in_A_P (P : ℝ[X]) (hP : polynomial_degree8 P) : ∃ n : ℕ, n = 1 ∧ ∃ x ∈ A_P P, odd x :=
sorry

end minimum_odd_in_A_P_l255_255380


namespace find_n_l255_255785

noncomputable def A : Set ℕ := {a : ℕ | a > 0}

def satisfies_condition (n : ℕ) : Prop :=
  ∀ a ∈ A, (a^n + a^(n-1) + ... + a + 1) ∣ (a^(n!) + a^((n-1)!) + ... + a^(1!) + 1)

theorem find_n (n : ℕ) :
  satisfies_condition n ↔ n = 1 ∨ n = 2 :=
sorry

end find_n_l255_255785


namespace distance_between_parallel_lines_l255_255195

open Real

theorem distance_between_parallel_lines 
    (L₁ L₂ L₃ L₄ : ℝ → ℝ → Prop) -- representing the four parallel lines
    (L₁_eq : ∃ k₁, ∀ x y, L₁ x y ↔ y = k₁ * x) -- Line equation of L₁
    (L₂_eq : ∃ k₂, ∀ x y, L₂ x y ↔ y = k₂ * x + d) -- Line equation of L₂
    (L₃_eq : ∃ k₃, ∀ x y, L₃ x y ↔ y = k₃ * x + 2 * d) -- Line equation of L₃
    (L₄_eq : ∃ k₄, ∀ x y, L₄ x y ↔ y = k₄ * x + 3 * d) -- Line equation of L₄
    (chords : list ℝ) -- representing lengths 42, 36, 36, 30
    (Hch : chords = [42, 36, 36, 30])
    (circle : ℝ → ℝ → Prop) -- representing the circle
    (Hcircle : ∃ r O, ∀ x y, circle x y ↔ (x - O.1) ^ 2 + (y - O.2) ^ 2 = r ^ 2) -- equation of the circle 
  : d = sqrt 2 := 
begin
  -- proof is not required
  sorry
end

end distance_between_parallel_lines_l255_255195


namespace confidence_level_related_l255_255903

theorem confidence_level_related (K_squared : ℝ) (h : K_squared > 6.635) : 
    "99% confidence level that event A is related to event B" :=
by
    sorry

end confidence_level_related_l255_255903


namespace S_30_value_l255_255862

def a (n : ℕ) : ℝ :=
  n^2 * (Real.cos (n * Real.pi / 3) ^ 2 - Real.sin (n * Real.pi / 3) ^ 2)

def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), a i

theorem S_30_value : S 30 = -3380 := by
  sorry

end S_30_value_l255_255862


namespace total_earnings_l255_255324

variable (work_hours_week_2 work_hours_week_3 : ℕ)
variable (wage_diff : ℝ) 
variable (earn_diff : ℝ)

theorem total_earnings (h1 : work_hours_week_2 = 24)
                       (h2 : work_hours_week_3 = 35)
                       (h3 : earn_diff = 84.50)
                       (h4 : work_hours_week_3 - work_hours_week_2 > 0)
                       (w : real) 
                       (hourly_wage : w = earn_diff / ((work_hours_week_3:ℝ) - (work_hours_week_2:ℝ))) :
  59 * hourly_wage = 453.12 :=
sorry

end total_earnings_l255_255324


namespace smallest_m_exists_l255_255794

def in_T (z : ℂ) : Prop :=
  (∃ (x y : ℝ), z = x + y * complex.I ∧ (real.sqrt 3) / 2 ≤ x ∧ x ≤ 2 / (real.sqrt 3))

theorem smallest_m_exists :
  ∃ m : ℕ, m = 12 ∧ (∀ n : ℕ, n ≥ m → (∃ z : ℂ, in_T z ∧ z^n = 1)) :=
sorry

end smallest_m_exists_l255_255794


namespace night_lamps_min_num_odd_l255_255023

theorem night_lamps_min_num_odd (n : ℕ) (h_odd : n % 2 = 1) :
  ∃ k : ℕ, k = (n + 1) ^ 2 / 2 ∧ 
            ∀ lamp_fails : ℕ, lamp_fails < k →
            (let positions := {positions : ℕ × ℕ // positions.1 < n + 1 ∧ positions.2 < n + 1} in
             ∀ pos in positions, (pos.1 - 1) % 2 = 0 ∨ (pos.2 - 1) % 2 = 0) :=
sorry

end night_lamps_min_num_odd_l255_255023


namespace circumcircle_radius_of_triangle_l255_255415

theorem circumcircle_radius_of_triangle (A B C M N : Point)
  (h1 : on_bisector B A C M)
  (h2 : on_extension A B N)
  (h3 : distance A C = 1)
  (h4 : distance A M = 1)
  (h5 : angle A N M = angle C N M) :
  radius (circumcircle C N M) = 1 := 
sorry

end circumcircle_radius_of_triangle_l255_255415


namespace distance_AP_BP_l255_255467

theorem distance_AP_BP (a b : ℝ) : 
  (y = x * sqrt 3 - 3) → 
  (2 * y^2 = 2 * x + 3) → 
  (P = (sqrt 3, 0)) →
  A = (a^2 - 3/2, a) → 
  B = (b^2 - 3/2, b) →
  a < 0 →
  b > 0 → 
  abs ((sqrt(a^2 + (a / sqrt 3)^2)) - (sqrt(b^2 + (b / sqrt 3)^2))) = 2 / 3 :=
begin
  sorry
end

end distance_AP_BP_l255_255467


namespace monotonic_increasing_interval_of_f_l255_255038

noncomputable def f : ℝ → ℝ := λ x, -(x - 5) * abs x

theorem monotonic_increasing_interval_of_f : {x : ℝ | x > 0 ∧ x < 5/2} = {x : ℝ | f' x > 0} :=
by
  -- It is recommended to provide the outline of the proof here
  -- Assume the calculations for the derivative and finding the interval by analysis
  sorry

end monotonic_increasing_interval_of_f_l255_255038


namespace triangle_angle_A_and_area_l255_255295

theorem triangle_angle_A_and_area 
  (a b c : ℝ)
  (A B C : ℝ)
  (r : ℝ := 1)
  (S : ℝ := sqrt 3 * sin B * sin C)
  (a_gt_b : a > b)
  (a_gt_c : a > c)
  (side_a : a = 2 * r * sin A)
  (side_b : b = 2 * r * sin B)
  (side_c : c = 2 * r * sin C)
  (A_in_range : 0 < A ∧ A < π)
  (D : ℝ)
  (BD DC : ℝ)
  (BD_2DC : BD = 2 * DC)
  (AD_perpendicular_AB : ⊥ AD)
  (angle_A : A = 2π / 3)
    : ∃ (area : ℝ), area = S ∧ S = sqrt 3 / 4 :=
by
  sorry

end triangle_angle_A_and_area_l255_255295


namespace sum_of_solutions_l255_255507

theorem sum_of_solutions :
  (∑ x in { x : ℚ | x = abs (2 * x - abs (50 - 2 * x))}.to_finset) = 230 / 3 := 
sorry

end sum_of_solutions_l255_255507


namespace number_of_members_of_set_A_l255_255050

variable (U A B C : Set α)
variable (nU nB nA_and_B nC nA_and_C nB_and_C nA_and_B_and_C : ℕ)

-- Given conditions
def conditions 
[card_U : #|U| = 350] 
[card_B : #|B| = 49] 
[card_A_and_B : #|A ∩ B| = 23] 
[card_C : #|C| = 36] 
[card_A_and_C : #|A ∩ C| = 14] 
[card_B_and_C : #|B ∩ C| = 19] 
[card_A_and_B_and_C : #|A ∩ B ∩ C| = 8]
[without_A_or_B : (#|U| -  #|A ∪ B|) = 59] : Prop :=
  sorry -- automatically true due to card_U and without_A_or_B

-- Target claim
theorem number_of_members_of_set_A 
[conditions U A B C nU nB nA_and_B nC nA_and_C nB_and_C nA_and_B_and_C] 
: (#|U| = 350) → (#|B| = 49) → (#|A ∩ B| = 23) → (#|C| = 36) → (#|A ∩ C| = 14) → (#|B ∩ C| = 19) → (#|A ∩ B ∩ C| = 8) → (#|U| - #|A ∪ B| = 59) → (#|A| = 265) :=
    sorry

end number_of_members_of_set_A_l255_255050


namespace probability_e_x_in_range_l255_255555

theorem probability_e_x_in_range :
  let e := Real.exp 1
  let x := ℝ
  let interval := Set.Icc 1 3
  let subinterval := Set.Icc e (e^2)
  ∀ (x ∈ interval),
    ∃ (p : ℝ), (p = 1 / 2) ∧
    (Set.Probability.subinterval_mapping (Real.exp) interval subinterval = p) := sorry

end probability_e_x_in_range_l255_255555


namespace expected_number_of_explorers_no_advice_l255_255591

-- Define the problem
theorem expected_number_of_explorers_no_advice
  (n : ℕ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∑ j in Finset.range n, (1 - p) ^ j) / p = (1 - (1 - p) ^ n) / p := by
  sorry

end expected_number_of_explorers_no_advice_l255_255591


namespace solve_otimes_l255_255330

variable (R : Type)[LinearOrderedField R]
variable (otimes : R → R → R)
variable (continuous_otimes : ∀ x y, Continuous (λ p : R × R, p.1 ⊗ p.2))
variable (commutative_otimes : ∀ (a b : R), a ⊗ b = b ⊗ a)
variable (distributive_otimes : ∀ (a b c : R), a ⊗ (b * c) = (a ⊗ b) * (a ⊗ c))
variable (otimes_two : (2 : R) ⊗ (2 : R) = 4)

theorem solve_otimes (x : R) (y : R) (hx : 1 < x) : (x ⊗ y = x) → (y = sqrt 2) := by
  sorry

end solve_otimes_l255_255330


namespace find_r_l255_255185

theorem find_r (r : ℝ) : log 8 (r + 8) = 7 / 3 → r = 120 :=
by
  sorry

end find_r_l255_255185


namespace triangle_area_l255_255067

theorem triangle_area (a b c : ℝ) (h_a : a = 15) (h_b : b = 36) (h_c : c = 39) (h_triangle : a * a + b * b = c * c) :
  1 / 2 * a * b = 270 :=
by
  rw [h_a, h_b, h_triangle]
  sorry

end triangle_area_l255_255067


namespace third_median_length_l255_255496

variable (a b : ℝ) (A : ℝ)

def two_medians (m₁ m₂ : ℝ) : Prop :=
  m₁ = 4.5 ∧ m₂ = 7.5

def triangle_area (area : ℝ) : Prop :=
  area = 6 * Real.sqrt 20

theorem third_median_length (m₁ m₂ m₃ : ℝ) (area : ℝ) (h₁ : two_medians m₁ m₂)
  (h₂ : triangle_area area) : m₃ = 3 * Real.sqrt 5 := by
  sorry

end third_median_length_l255_255496


namespace minimum_small_droppers_l255_255554

/-
Given:
1. A total volume to be filled: V = 265 milliliters.
2. Small droppers can hold: s = 19 milliliters each.
3. No large droppers are used.

Prove:
The minimum number of small droppers required to fill the container completely is 14.
-/

theorem minimum_small_droppers (V s: ℕ) (hV: V = 265) (hs: s = 19) : 
  ∃ n: ℕ, n = 14 ∧ n * s ≥ V ∧ (n - 1) * s < V :=
by
  sorry  -- proof to be provided

end minimum_small_droppers_l255_255554


namespace min_value_of_expression_l255_255681

theorem min_value_of_expression (a b : ℝ) (h1: a > b)
  (h2 : ∀ x : ℝ, a * x ^ 2 + 2 * x + b ≥ 0)
  (h3 : ∃ x₀ : ℝ, a * x₀ ^ 2 + 2 * x₀ + b = 0)
  (h4 : ∀ x : ℝ, a * x ^ 2 + 2 * x + b ≥ 0) :
  ∃ a b : ℝ, (a > 1 ∧ a * b = 1) ∧ (∀ x ∈ ℕ, a * (x: ℝ) ^ 2 + 2 * x + b  = 2 * sqrt(2)) := 
begin
 sorry
end

end min_value_of_expression_l255_255681


namespace train_length_l255_255963

noncomputable def speed_kmph : ℝ := 144
noncomputable def time_seconds : ℝ := 14.998800095992321

noncomputable def speed_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def distance (speed_mps : ℝ) (time_seconds : ℝ) : ℝ :=
  speed_mps * time_seconds

theorem train_length :
  distance (speed_mps speed_kmph) time_seconds ≈ 599.9520038396928 :=
sorry

end train_length_l255_255963


namespace problem_ineq_l255_255398

theorem problem_ineq (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
(h4 : x * y * z = 1) :
    (x^3 / ((1 + y)*(1 + z)) + y^3 / ((1 + z)*(1 + x)) + z^3 / ((1 + x)*(1 + y))) ≥ 3 / 4 := 
sorry

end problem_ineq_l255_255398


namespace sum_of_elements_of_T_l255_255799

def is_repeat_decimal (x : ℝ) (a b : ℕ) : Prop :=
  x = (10 * a + b) / 99

def elements_of_T : set ℝ :=
  {x | ∃ (a b : ℕ), a ≠ b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ is_repeat_decimal x a b}

theorem sum_of_elements_of_T : 
  (∑ x in elements_of_T, x) = 45 :=
sorry

end sum_of_elements_of_T_l255_255799


namespace smallest_area_triangle_l255_255998

/-- 
Mathematically equivalent proof problem: 
Prove that the line passing through the point A with coordinates (-2, 2) forms the smallest 
area triangle with the coordinate axes in the second quadrant when the line equation is x - y + 4 = 0.
--/
theorem smallest_area_triangle (A : ℝ × ℝ) (B : ℝ) (hA : A = (-2, 2)) 
    (hB : B = -(2 + 2 * m) := 4 + 2 * m) 
    (h_condition: ∀ (m: ℝ), m < 0):
  (A = (-2, 2) ∧ (line_eq : ∀ x y, y = mx + 2 + 2 * m) ):
  (line_eq = x - y + 4) :=
sorry

end smallest_area_triangle_l255_255998


namespace dividend_value_l255_255157

variable {y : ℝ}

theorem dividend_value (h : y > 3) : 
  let x := (3 * y + 5) * (2 * y - 1) + (5 * y - 13) in
  x = 6 * y^2 + 12 * y - 18 :=
by
  sorry

end dividend_value_l255_255157


namespace equation_of_line_and_length_of_segment_equation_of_line_l255_255225

theorem equation_of_line_and_length_of_segment (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop)
  (h1 : ∀ P, l P → ∃ P1 P2 : ℝ × ℝ, P = P1 ∨ P = P2)
  (h2 : ∀ P, l P ↔ (P.2 - 2 = P.1 - 3))
  (h3 : ∀ D, D = (A.1 + B.1) / 2 ∧ D = (A.2 + B.2) / 2 → D = (3, 2))
  (h4 : ∀ P, P ∈ set_of l → ∃ k, P.2 = k * (P.1 - 3) + 2)
  (h5 : parabolic_intersection: ∀ P, P ∈ set_of l ∩ set_of (λ P, P.2^2 = 4 * P.1)) :

  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 8^2 :=
by
  sorry

-- Additional theorem for the equation of the line:
theorem equation_of_line (l : ℝ → ℝ → Prop)
  (h1 : ∀ P, l P → (P.2 - 2 = P.1 - 3)) :
  ∀ x y, l (x, y) ↔ (x - y - 1 = 0) :=
by
  sorry

end equation_of_line_and_length_of_segment_equation_of_line_l255_255225


namespace sum_of_digits_of_x_l255_255911

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10 in s = s.reverse

theorem sum_of_digits_of_x 
  (x : ℕ) 
  (h1 : 100 ≤ x ∧ x ≤ 999)
  (h2 : is_palindrome x)
  (h3 : is_palindrome (x + 40))
  (h4 : 1000 ≤ x + 40 ∧ x + 40 ≤ 1039) :
  (x.digits 10).sum = 16 := 
sorry

end sum_of_digits_of_x_l255_255911


namespace detect_counterfeit_coins_l255_255576

def Coin := ℕ

structure Expert where
  balance_scale : Prop
  real_coins : Finset Coin
  counterfeit_coins : Finset Coin
  all_coins : Finset Coin

theorem detect_counterfeit_coins (e : Expert) 
  (h_balance: e.balance_scale) 
  (h_real: e.real_coins.card = 5)
  (h_counterfeit: e.counterfeit_coins.card = 5)
  (h_all: e.all_coins.card = 12) 
: ∃ detect_number_of_counterfeit (c: Finset Coin), e.all_coins = e.real_coins ∪ e.counterfeit_coins ∧ detect_number_of_counterfeit.card ≤ 4 :=
sorry

end detect_counterfeit_coins_l255_255576


namespace problem_1_2_a_problem_1_2_b_l255_255084

theorem problem_1_2_a (x : ℝ) : x * (1 - x) ≤ 1 / 4 := sorry

theorem problem_1_2_b (x a : ℝ) : x * (a - x) ≤ a^2 / 4 := sorry

end problem_1_2_a_problem_1_2_b_l255_255084


namespace find_a_l255_255255

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a {x0 a : ℝ} (h : f x0 a - g x0 a = 3) : a = -Real.log 2 - 1 :=
sorry

end find_a_l255_255255


namespace total_spider_legs_l255_255959

-- Define the number of legs per spider.
def legs_per_spider : ℕ := 8

-- Define half of the legs per spider.
def half_legs : ℕ := legs_per_spider / 2

-- Define the number of spiders in the group.
def num_spiders : ℕ := half_legs + 10

-- Prove the total number of spider legs in the group is 112.
theorem total_spider_legs : num_spiders * legs_per_spider = 112 := by
  -- Use 'sorry' to skip the detailed proof steps.
  sorry

end total_spider_legs_l255_255959


namespace number_of_foxes_in_forest_l255_255436

-- Define the total number of animals in the forest
def total_animals : ℕ := 160

-- Define the fraction of animals that are deer
def fraction_deer : ℚ := 7 / 8

-- Define the fraction of animals that are foxes
def fraction_foxes : ℚ := 1 - fraction_deer

-- Define what we want to prove: the number of foxes in the forest
def number_of_foxes (total : ℕ) (fraction_fox : ℚ) : ℕ := (fraction_fox * total).toNat

theorem number_of_foxes_in_forest : number_of_foxes total_animals fraction_foxes = 20 :=
by
  sorry

end number_of_foxes_in_forest_l255_255436


namespace probability_of_sum_5_l255_255075

theorem probability_of_sum_5 :
  let outcomes := {(i, j) | i in Finset.range 1 6, j in Finset.range 1 6},
      favorable_outcomes := {(i, j) ∈ outcomes | i + j = 5} 
  in (favorable_outcomes.card : ℚ) / outcomes.card = 1 / 9 := by
  sorry

end probability_of_sum_5_l255_255075


namespace set_inter_complement_eq_l255_255238

open Set

theorem set_inter_complement_eq {A B : Set ℝ} (hA : A = {x | -1 ≤ x ∧ x ≤ 4}) (hB : B = {x | 3 ≤ x ∧ x ≤ 5}) :
  A ∩ ((univ : Set ℝ) \ B) = {x | -1 ≤ x ∧ x < 3} :=
by
  rw [hA, hB]
  rw [compl_set_of]
  ext
  simp
  sorry

end set_inter_complement_eq_l255_255238


namespace farthest_point_l255_255907

def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1 ^ 2 + p.2 ^ 2)

theorem farthest_point 
  (p1 p2 p3 p4 p5 : ℝ × ℝ) 
  (h1 : p1 = (2, 3)) 
  (h2 : p2 = (4, 1)) 
  (h3 : p3 = (5, -3)) 
  (h4 : p4 = (7, 0)) 
  (h5 : p5 = (-3, -5)) : 
  distance_from_origin p4 = 7 ∧ 
  distance_from_origin p4 ≥ distance_from_origin p1 ∧ 
  distance_from_origin p4 ≥ distance_from_origin p2 ∧ 
  distance_from_origin p4 ≥ distance_from_origin p3 ∧ 
  distance_from_origin p4 ≥ distance_from_origin p5 :=
by
  sorry

end farthest_point_l255_255907


namespace difference_is_24750_l255_255045

noncomputable def difference_of_numbers (n1 n2 : ℕ) (sum : n1 + n2 = 25220)
  (div_by_12 : (n1 % 12 = 0) ∨ (n2 % 12 = 0))
  (erase_units_tens : n1 % 100 = n2 ∨ n2 % 100 = n1) : ℕ :=
  if h : (n1 % 12 = 0 ∧ n1 % 100 / 1 = n2)
         ∨ (n2 % 12 = 0 ∧ n2 % 100 / 1 = n1) then
    if (n1 % 12 = 0) then 99 * (n1 % 100) else 99 * (n2 % 100)
  else
    sorry

theorem difference_is_24750 :
  ∃ (n1 n2 : ℕ), n1 + n2 = 25220 ∧
  ((n1 % 12 = 0 ∧ (n1 % 100 / 1 = n2)) ∨ (n2 % 12 = 0 ∧ (n2 % 100 / 1 = n1))) ∧
  difference_of_numbers n1 n2 _ _ _ = 24_750 :=
by
  sorry

end difference_is_24750_l255_255045


namespace total_divisions_is_48_l255_255097

-- Definitions based on the conditions
def initial_cells := 1
def final_cells := 1993
def cells_added_division_42 := 41
def cells_added_division_44 := 43

-- The main statement we want to prove
theorem total_divisions_is_48 (a b : ℕ) 
  (h1 : cells_added_division_42 = 41)
  (h2 : cells_added_division_44 = 43)
  (h3 : cells_added_division_42 * a + cells_added_division_44 * b = final_cells - initial_cells) :
  a + b = 48 := 
sorry

end total_divisions_is_48_l255_255097


namespace squarish_count_eq_two_l255_255942

def is_nonzero_digit (n : ℕ) : Prop :=
  n > 0 ∧ n < 10

def is_four_digit (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_two_digit_perfect_square (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ is_perfect_square n

def is_squarish (n : ℕ) : Prop :=
  is_four_digit n ∧
  (∀ d : ℕ, d ∈ (to_digits n) → is_nonzero_digit d) ∧
  is_perfect_square n ∧
  let (a, b) := (n / 100, n % 100) in is_two_digit_perfect_square a ∧ is_two_digit_perfect_square b

theorem squarish_count_eq_two : (finset.filter is_squarish (finset.range 10000)).card = 2 :=
sorry

end squarish_count_eq_two_l255_255942


namespace tan_A_minus_B_l255_255770

theorem tan_A_minus_B (A B : ℝ) (h1: Real.cos A = -Real.sqrt 2 / 2) (h2 : Real.tan B = 1 / 3) : 
  Real.tan (A - B) = -2 := by
  sorry

end tan_A_minus_B_l255_255770


namespace parabola_vertex_l255_255044

theorem parabola_vertex (c d : ℝ) (h : ∀ (x : ℝ), (-x^2 + c * x + d ≤ 0) ↔ (x ≤ -5 ∨ x ≥ 3)) :
  (∃ a b : ℝ, a = 4 ∧ b = 1 ∧ (-x^2 + c * x + d = -x^2 + 8 * x - 15)) :=
by
  sorry

end parabola_vertex_l255_255044


namespace gain_represents_ten_meters_l255_255560

def cost_price (C : ℝ) := C
def selling_price (S : ℝ) (C : ℝ) := 1.25 * C
def total_cost (C : ℝ) := 50 * C
def total_sell (S : ℝ) := 50 * S
def gain (C : ℝ) (S : ℝ) := 50 * S - 50 * C
def meters_represented_by_gain (gain : ℝ) (S : ℝ) := gain / S

theorem gain_represents_ten_meters
  (C S : ℝ)
  (h : S = 1.25 * C) :
  meters_represented_by_gain (gain C S) S = 10 :=
by
  sorry

end gain_represents_ten_meters_l255_255560


namespace string_length_eq_l255_255937

-- Definitions corresponding to the conditions
def circumference : ℝ := 6
def loops : ℕ := 3
def height : ℝ := 15
def length_of_string : ℝ := 3 * Real.sqrt 61

-- Theorem to prove the length of the string equals the expected correct answer
theorem string_length_eq : 
  ∀ (circumference height : ℝ) (loops : ℕ), 
  circumference = 6 → height = 15 → loops = 3 → 
  length_of_string = 3 * Real.sqrt 61 :=
by
  intros circumference height loops h1 h2 h3
  sorry

end string_length_eq_l255_255937


namespace find_a_l255_255256

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a {x0 a : ℝ} (h : f x0 a - g x0 a = 3) : a = -Real.log 2 - 1 :=
sorry

end find_a_l255_255256


namespace largest_n_exists_l255_255671

def digitDivisorSum (n : ℕ) : ℕ :=
  List.sum (List.filter (λ d, d > 0) (List.map (λ d, List.sum (List.filter (λ i, i ∣ d) (List.range (d+1)))) (nat.digits 10 n)))

def fn (k : ℕ) (a : ℕ) : ℕ := (List.iterate digitDivisorSum k a).last

theorem largest_n_exists :
  ∃ n, (∀ k : ℕ, ∃ a : ℕ, fn k a = n) ∧ n = 15 := sorry

end largest_n_exists_l255_255671


namespace functional_equation_solution1_functional_equation_solution2_l255_255648

noncomputable def solution1 (x : ℝ) : ℝ :=
  x - 1

noncomputable def solution2 (x : ℝ) : ℝ :=
  -x - 1

theorem functional_equation_solution1 : ∀ (x y : ℝ), solution1 x * solution1 y + solution1 (x + y) = x * y :=
by
  intros x y
  calc
    solution1 x * solution1 y + solution1 (x + y) = (x - 1) * (y - 1) + (x + y - 1) : by rfl
    ... = x * y - x - y + 1 + x + y - 1 : by ring
    ... = x * y : by ring
    done

theorem functional_equation_solution2 : ∀ (x y : ℝ), solution2 x * solution2 y + solution2 (x + y) = x * y :=
by
  intros x y
  calc
    solution2 x * solution2 y + solution2 (x + y) = (-x - 1) * (-y - 1) + (-x - y - 1) : by rfl
    ... = x * y + x + y + 1 - x - y - 1 : by ring
    ... = x * y : by ring
    done

end functional_equation_solution1_functional_equation_solution2_l255_255648


namespace rahul_matches_played_l255_255836

theorem rahul_matches_played
  (current_avg : ℕ)
  (runs_today : ℕ)
  (new_avg : ℕ)
  (m: ℕ)
  (h1 : current_avg = 51)
  (h2 : runs_today = 78)
  (h3 : new_avg = 54)
  (h4 : (51 * m + runs_today) / (m + 1) = new_avg) :
  m = 8 :=
by
  sorry

end rahul_matches_played_l255_255836


namespace marble_count_l255_255113

theorem marble_count (x : ℝ) (h1 : 0.5 * x = x / 2) (h2 : 0.25 * x = x / 4) (h3 : (27 + 14) = 41) (h4 : 0.25 * x = 27 + 14)
  : x = 164 :=
by
  sorry

end marble_count_l255_255113


namespace question2_l255_255207

noncomputable def a (n : ℕ) : ℕ :=
  2^n

noncomputable def b (n : ℕ) : ℕ :=
  4^n - 2^n

noncomputable def S (n : ℕ) : ℕ :=
  2^n + 4 * (4^n - 2^n)

noncomputable def P (n : ℕ) : ℚ :=
  a n / S n

theorem question2 (n : ℕ) : ∑ i in range n, P (i+1) < 3/2 := by
  sorry

end question2_l255_255207


namespace union_of_M_N_l255_255094

def M : Set ℝ := { x | x^2 + 2*x = 0 }

def N : Set ℝ := { x | x^2 - 2*x = 0 }

theorem union_of_M_N : M ∪ N = {0, -2, 2} := sorry

end union_of_M_N_l255_255094


namespace observable_area_shrinkage_percentage_l255_255566

variables (L B : ℝ)

def original_area := L * B
def new_length := 0.80 * L
def new_breadth := 0.90 * B
def area_after_shrinkage := new_length * new_breadth
def cumulative_shrinkage := 0.95 * area_after_shrinkage
def observable_area_after_folding := 0.5 * cumulative_shrinkage

theorem observable_area_shrinkage_percentage :
  (observable_area_after_folding L B / original_area L B) = 0.342 :=
sorry

end observable_area_shrinkage_percentage_l255_255566


namespace matrix_problem_l255_255339

theorem matrix_problem (a b c d : ℤ) (h : (matrix.mul 
  (matrix.ofFunction (λ i j, if i = 0 ∧ j = 0 then a else if i = 0 ∧ j = 1 then b else if i = 1 ∧ j = 0 then c else d)) 
  (matrix.ofFunction (λ i j, if i = 0 ∧ j = 0 then a else if i = 0 ∧ j = 1 then b else if i = 1 ∧ j = 0 then c else d))) 
  = matrix.ofFunction (λ i j, if i = 0 ∧ j = 0 then 9 else if i = 0 ∧ j = 1 then 0 else if i = 1 ∧ j = 0 then 0 else 9)) :
  ∃ (a b c d : ℤ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∧ |a| + |b| + |c| + |d| = 8 :=
by
  sorry

end matrix_problem_l255_255339


namespace equal_tangent_lengths_locus_of_equal_tangents_l255_255892

noncomputable theory

-- Define the triangle and the medians
variables (A B C A1 B1 Oa Ob : Type*) [MetricSpace Oa] [MetricSpace Ob]
variables [Ops Oa Ob]

-- Define circles ka and kb
def ka : Set Oa := set.Icc 0 (dist A A1) 
def kb : Set Ob := set.Icc 0 (dist B B1) 

-- Define the tangents and the orthogonal line passing through C
def tangent_length_k_a (C : Oa) (tangent_point : Oa) := dist C tangent_point -- tangent length from C to ka
def tangent_length_k_b (C : Ob) (tangent_point : Ob) := dist C tangent_point -- tangent length from C to kb

theorem equal_tangent_lengths : angle A C B < pi / 2 → 
    tangent_length_k_a C = tangent_length_k_b C :=
sorry

theorem locus_of_equal_tangents : 
    forall (P : Type*) [MetricSpace P], (dist P Oa = dist P Ob) → 
    ∃ (P' : Type*), (is_orthogonal (line_through P' (line_through Oa Ob)) (line_through C P'))
    :=
sorry

end equal_tangent_lengths_locus_of_equal_tangents_l255_255892


namespace chicken_cost_l255_255145

def cost_per_pound_chicken (cost_steak_per_pound cost_total pounds_chicken pounds_steak total_spent : ℝ) : ℝ := 
    total_spent - (pounds_steak * cost_steak_per_pound) / pounds_chicken

theorem chicken_cost (cost_steak_per_pound : ℝ) (cost_total : ℝ) (pounds_chicken : ℝ) (pounds_steak : ℝ) (total_spent : ℝ) :
  cost_steak_per_pound = 15 →
  pounds_steak = 2 →
  total_spent = 42 →
  pounds_chicken = 1.5 →
  cost_per_pound_chicken 15 42 1.5 2 42 = 8 := 
by
  intros
  rw [cost_per_pound_chicken]
  simp
  sorry

end chicken_cost_l255_255145


namespace solve_equation_integers_l255_255012

theorem solve_equation_integers :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (1 + 1 / (x : ℚ)) * (1 + 1 / (y : ℚ)) * (1 + 1 / (z : ℚ)) = 2 ∧
  (x = 2 ∧ y = 4 ∧ z = 15 ∨
   x = 2 ∧ y = 5 ∧ z = 9 ∨
   x = 2 ∧ y = 6 ∧ z = 7 ∨
   x = 3 ∧ y = 4 ∧ z = 5 ∨
   x = 3 ∧ y = 3 ∧ z = 8 ∨
   x = 2 ∧ y = 15 ∧ z = 4 ∨
   x = 2 ∧ y = 9 ∧ z = 5 ∨
   x = 2 ∧ y = 7 ∧ z = 6 ∨
   x = 3 ∧ y = 5 ∧ z = 4 ∨
   x = 3 ∧ y = 8 ∧ z = 3) ∧
  (y = 2 ∧ x = 4 ∧ z = 15 ∨
   y = 2 ∧ x = 5 ∧ z = 9 ∨
   y = 2 ∧ x = 6 ∧ z = 7 ∨
   y = 3 ∧ x = 4 ∧ z = 5 ∨
   y = 3 ∧ x = 3 ∧ z = 8 ∨
   y = 15 ∧ x = 4 ∧ z = 2 ∨
   y = 9 ∧ x = 5 ∧ z = 2 ∨
   y = 7 ∧ x = 6 ∧ z = 2 ∨
   y = 5 ∧ x = 4 ∧ z = 3 ∨
   y = 8 ∧ x = 3 ∧ z = 3) ∧
  (z = 2 ∧ x = 4 ∧ y = 15 ∨
   z = 2 ∧ x = 5 ∧ y = 9 ∨
   z = 2 ∧ x = 6 ∧ y = 7 ∨
   z = 3 ∧ x = 4 ∧ y = 5 ∨
   z = 3 ∧ x = 3 ∧ y = 8 ∨
   z = 15 ∧ x = 4 ∧ y = 2 ∨
   z = 9 ∧ x = 5 ∧ y = 2 ∨
   z = 7 ∧ x = 6 ∧ y = 2 ∨
   z = 5 ∧ x = 4 ∧ y = 3 ∨
   z = 8 ∧ x = 3 ∧ y = 3)
:= sorry

end solve_equation_integers_l255_255012


namespace repeating_decimal_to_fraction_l255_255638

theorem repeating_decimal_to_fraction : (0.3666666 : ℚ) = 11 / 30 :=
by sorry

end repeating_decimal_to_fraction_l255_255638


namespace tan_product_inequality_l255_255039

theorem tan_product_inequality 
  (n : ℕ) 
  (x : Fin (n+1) → ℝ) 
  (hx1 : ∀ i, 0 < x i  ∧ x i < π / 2) 
  (hx2 : ∑ i, Real.tan (x i - π / 4) ≥ n - 1) :
  ∏ i, Real.tan (x i) ≥ n ^ (n+1) := 
by 
  sorry

end tan_product_inequality_l255_255039


namespace count_valid_configurations_l255_255846

def is_valid_placement (square : Char) : Bool :=
  square = 'A' ∨ square = 'B' ∨ square = 'D' ∨ square = 'E' ∨ square = 'F' ∨ square = 'H'

theorem count_valid_configurations : (Finset.filter is_valid_placement (Finset.ofList ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])).card = 6 :=
by
  -- Filter the valid placements
  let valid_squares := Finset.filter is_valid_placement (Finset.ofList ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
  -- Assert that the count of valid squares is 6
  have h : valid_squares.card = 6 := by
    -- Explicitly calculate the valid placements and their count
    sorry
  exact h

end count_valid_configurations_l255_255846


namespace bells_ring_together_l255_255101

theorem bells_ring_together (church school day_care library noon : ℕ) :
  church = 18 ∧ school = 24 ∧ day_care = 30 ∧ library = 35 ∧ noon = 0 →
  ∃ t : ℕ, t = 2520 ∧ ∀ n, (t - noon) % n = 0 := by
  sorry

end bells_ring_together_l255_255101


namespace max_chords_through_line_l255_255481

noncomputable def maxChords (n : ℕ) : ℕ :=
  let k := n / 2
  k * k + n

theorem max_chords_through_line (points : ℕ) (h : points = 2017) : maxChords 2016 = 1018080 :=
by
  have h1 : (2016 / 2) * (2016 / 2) + 2016 = 1018080 := by norm_num
  rw [← h1]; sorry

end max_chords_through_line_l255_255481


namespace probability_opposite_points_l255_255060

theorem probability_opposite_points (n : ℕ) (h₁ : n = 12) :
  (12.choose 2) / 6 = 1 / 11 :=
by sorry

end probability_opposite_points_l255_255060


namespace total_fruits_in_baskets_l255_255448

structure Baskets where
  mangoes : ℕ
  pears : ℕ
  pawpaws : ℕ
  kiwis : ℕ
  lemons : ℕ

def taniaBaskets : Baskets := {
  mangoes := 18,
  pears := 10,
  pawpaws := 12,
  kiwis := 9,
  lemons := 9
}

theorem total_fruits_in_baskets : taniaBaskets.mangoes + taniaBaskets.pears + taniaBaskets.pawpaws + taniaBaskets.kiwis + taniaBaskets.lemons = 58 :=
by
  sorry

end total_fruits_in_baskets_l255_255448


namespace problem_A_problem_B_problem_C_problem_D_l255_255702

def center := (3, 4)

def radius := sqrt 8

def M1 := (2, 0)

def M2 := (-2, 0)

def O := (0, 0)

def N1 := (1, 0)

def N2 := (-1, 0)

theorem problem_A : ∀ (A: ℝ × ℝ), (A ∈ circle center radius) → dist A M2 ≤ sqrt 41 + 2 * sqrt 2 := 
by sorry

theorem problem_B : ∀ (A: ℝ × ℝ), (A ∈ circle center radius) → 
  area (triangle A M1 M2) ≥ 8 :=
by sorry

theorem problem_C : ∀ (A: ℝ × ℝ), (A ∈ circle center radius) → 
  angle (A, N2, N1) ≥ 15 :=
by sorry

theorem problem_D : ∀ (A: ℝ × ℝ), (A ∈ circle center radius) → 
  dot_product (A, N1) (A, N2) ≥ 32 - 20 * sqrt 2 :=
by sorry

end problem_A_problem_B_problem_C_problem_D_l255_255702


namespace night_rides_total_l255_255516

-- Definitions corresponding to the conditions in the problem
def total_ferris_wheel_rides : Nat := 13
def total_roller_coaster_rides : Nat := 9
def ferris_wheel_day_rides : Nat := 7
def roller_coaster_day_rides : Nat := 4

-- The total night rides proof problem
theorem night_rides_total :
  let ferris_wheel_night_rides := total_ferris_wheel_rides - ferris_wheel_day_rides
  let roller_coaster_night_rides := total_roller_coaster_rides - roller_coaster_day_rides
  ferris_wheel_night_rides + roller_coaster_night_rides = 11 :=
by
  -- Proof skipped
  sorry

end night_rides_total_l255_255516


namespace minimum_odd_in_A_P_l255_255378

-- Define the polynomial and its properties
def polynomial_degree8 (P : ℝ[X]) : Prop :=
  P.degree = 8

-- Define the set A_P
def A_P (P : ℝ[X]) : set ℝ :=
  {x | P.eval x = 8}

-- Main theorem statement
theorem minimum_odd_in_A_P (P : ℝ[X]) (hP : polynomial_degree8 P) : ∃ n : ℕ, n = 1 ∧ ∃ x ∈ A_P P, odd x :=
sorry

end minimum_odd_in_A_P_l255_255378


namespace ellipse_properties_l255_255213

noncomputable def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a > 0)
  (h_eccentricity : a = 2 * (sqrt (a^2 - b^2)))
  (h_contains_P : ellipse_eq a b 1 (3/2))
  (tangent_lines_slope_const : ∀ r, (0 < r) → (r < 3/2) → 
    let k₁ := (-4 : ℝ) / sqrt (16 - r^2),
        k₂ := (4 : ℝ) / sqrt (16 - r^2) in
    ∃ (M N : ℝ × ℝ), MN slope MN = 1/2)
  (max_area_triangle_MON : ∃ (O M N : ℝ × ℝ), 
    let base_length := dist M N,
        height := (abs (2 * snd N)) / sqrt 5 in
    1/2 * base_length * height = sqrt 3) 
: ellipse_eq 2 (sqrt 3) := sorry

end ellipse_properties_l255_255213


namespace no_real_x_satisfies_quadratic_ineq_l255_255908

theorem no_real_x_satisfies_quadratic_ineq :
  ¬ ∃ x : ℝ, x^2 + 3 * x + 3 ≤ 0 :=
sorry

end no_real_x_satisfies_quadratic_ineq_l255_255908


namespace train_speed_l255_255129

theorem train_speed (lt lb t : ℕ) (h1 : lt = 360) (h2 : lb = 140) (h3 : t = 30) : 
  (lt + lb) / t * 3.6 = 60 :=
by
  sorry

end train_speed_l255_255129


namespace sum_of_ages_l255_255885

theorem sum_of_ages (X_c Y_c : ℕ) (h1 : X_c = 45) 
  (h2 : X_c - 3 = 2 * (Y_c - 3)) : 
  (X_c + 7) + (Y_c + 7) = 83 := 
by
  sorry

end sum_of_ages_l255_255885


namespace f_shift_l255_255247

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem f_shift (x : ℝ) : f (x - 1) = x^2 - 4 * x + 3 := by
  sorry

end f_shift_l255_255247


namespace min_odd_in_A_P_l255_255351

-- Define the polynomial P of degree 8
variables (P : Polynomial ℝ)
hypothesis degree_P : P.natDegree = 8

-- Define the set A_P
def A_P (P : Polynomial ℝ) : Set ℝ := { x : ℝ | True }

-- Hypothesis that the set A_P includes the number 8
hypothesis A_P_contains_8 : 8 ∈ A_P P

-- The minimum number of odd numbers in the set A_P
theorem min_odd_in_A_P : ∃ n : ℕ, odd n ∧ n = 1 := sorry

end min_odd_in_A_P_l255_255351


namespace bugs_convex_polygon_at_some_moment_l255_255437

-- Define conditions of the problem
variables (L : Type) [LinearOrder L]
-- Assume we have n lines in the plane that are pairwise non-parallel
variables (bugs : ℕ → L × L) 
-- Each bug moves at a speed of 1 cm/s along its line
-- Define a function that describes the position of each bug at time t
def bug_position (t : ℝ) (i : ℕ) : ℝ × ℝ := 
  let (x, y) := bugs i in (x + t, y)

-- The theorem we aim to prove
theorem bugs_convex_polygon_at_some_moment :
  ∃ t : ℝ, ∀ i : ℕ, ∃ j : ℕ, j ≠ i ∧
    ¬(is_on_line (bug_position L bugs t i) (bug_position L bugs t j)) ∧
    is_polygon_vertex (bug_position L bugs t i) := 
sorry

end bugs_convex_polygon_at_some_moment_l255_255437


namespace leading_coefficient_of_g_l255_255401

def f (α : ℝ) (x : ℝ) : ℝ := (x / 2) ^ α / (x - 1)

noncomputable def g (α : ℝ) : ℝ := derivn (f α) 4 2

#check g

theorem leading_coefficient_of_g (α : ℝ) : leading_coeff (g α) = 1 / 16 :=
sorry

end leading_coefficient_of_g_l255_255401


namespace BXYC_concyclic_l255_255307

-- Definitions based on conditions
variable {A B C H M X Y O : Point}
variable {Γ : Circle}

-- Conditions
axiom triangle_ABC_acute : acute_angle_triangle A B C
axiom H_is_orthocenter : orthocenter H A B C
axiom M_is_midpoint : midpoint M B C
axiom Gamma_diameter_HM : Gamma.diameter (dist H M)
axiom tangents_through_A_touch_at_X_Y : tangents_through A Γ X Y

-- Goal statement
theorem BXYC_concyclic : concyclic B X Y C :=
by
  sorry

end BXYC_concyclic_l255_255307


namespace Polynomial_cannot_have_odd_degree_l255_255868

variable (P : ℤ × ℤ → ℤ)

-- The condition that for any integer n ≥ 0, each of the polynomials P(n, y) and P(x, n) is either identically zero or has degree no higher than n
def condition (P : ℤ × ℤ → ℤ) : Prop :=
  ∀ n : ℤ, ∀ y x : ℤ, n ≥ 0 → (
    (∀ k : ℤ, k > n → P (k, y) = 0) ∨ (∀ k : ℤ, k > n → P (x, k) = 0) ∨ (
      (∀ k l : ℤ, k ≤ n → l ≤ n → degree (λ y, P (k, y)) ≤ n ∧ degree (λ x, P (x, l)) ≤ n
    )
  )

theorem Polynomial_cannot_have_odd_degree (P : ℤ × ℤ → ℤ) :
  condition P → ¬ (∃ d : ℕ, d % 2 = 1 ∧ ∃ f : ℤ → ℤ, ∀ x : ℤ, P (x, x) = poly f x d) := sorry

end Polynomial_cannot_have_odd_degree_l255_255868


namespace largest_a_l255_255738

theorem largest_a (a b : ℕ) (x : ℕ) (h_a_range : 2 < a ∧ a < x) (h_b_range : 4 < b ∧ b < 13) (h_fraction_range : 7 * a = 57) : a = 8 :=
sorry

end largest_a_l255_255738


namespace crayons_given_to_friends_l255_255830

def initial_crayons : ℕ := 440
def lost_crayons : ℕ := 106
def remaining_crayons : ℕ := 223

theorem crayons_given_to_friends :
  initial_crayons - remaining_crayons - lost_crayons = 111 := 
by
  sorry

end crayons_given_to_friends_l255_255830


namespace rate_per_kg_grapes_l255_255980

theorem rate_per_kg_grapes :
  ∃ (G : ℝ), 7 * G + 9 * 48 = 908 ∧ G = 68 :=
by
  existsi 68
  split
  { 
    calc
      7 * 68 + 9 * 48 = 476 + 432 : by ring
      ... = 908 : by norm_num
  }
  {
    exact rfl
  } 

end rate_per_kg_grapes_l255_255980


namespace right_pyramid_edge_sum_l255_255950

-- Define the given parameters for the pyramid
def side_length : ℝ := 20
def peak_height : ℝ := 15

-- Calculate the sum of the lengths of the pyramid's eight edges
def sum_of_edges (s h : ℝ) : ℝ :=
  let base_perimeter := 4 * s in
  let half_diagonal := s * Real.sqrt 2 / 2 in
  let slant_height := Real.sqrt (h^2 + half_diagonal^2) in
  base_perimeter + 4 * slant_height

-- Prove that the result is 162 when expressed to the nearest whole number
theorem right_pyramid_edge_sum :
  Real.floor (sum_of_edges side_length peak_height + 0.5) = 162 :=
by
  sorry

end right_pyramid_edge_sum_l255_255950


namespace position_USEAMO_l255_255498

theorem position_USEAMO :
  let letters := ['A', 'E', 'M', 'O', 'S', 'U']
  ∃ (position : ℕ), alphabetical_position letters "USEAMO" = position ∧ position = 697 :=
sorry

end position_USEAMO_l255_255498


namespace tangent_lines_of_circle_and_point_l255_255685

noncomputable def is_tangent_line (c : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  ∃ t : ℝ, l = λ x y, (y - snd p) = t * (x - fst p) ∧
  (abs (t * (fst c - 1) - snd c + 3 - 2 * t) / sqrt (t^2 + 1) = r)

theorem tangent_lines_of_circle_and_point :
  ∀ (x y : ℝ), 
  ∃ l : ℝ × ℝ → Prop, 
  (l (2, 3) ∧ 
  ((l = (λ x y, x = 2)) ∨ (l = (λ x y, 3 * x - 4 * y + 6 = 0))))
  := by
  sorry

end tangent_lines_of_circle_and_point_l255_255685


namespace bridget_profit_l255_255146

theorem bridget_profit : 
  let loaves_baked := 60
  let price_morning := 3.00 -- dollars per loaf in the morning
  let price_afternoon := 1.50 -- dollars per loaf in the afternoon
  let price_late_afternoon := 1.00 -- dollars per loaf in the late afternoon
  let cost_per_loaf := 1.00 -- dollar cost per loaf
  let fixed_cost := 10.00 -- fixed operating cost
  let morning_loaves_sold := loaves_baked / 3
  let morning_revenue := morning_loaves_sold * price_morning
  let loaves_remaining := loaves_baked - morning_loaves_sold
  let afternoon_loaves_sold := loaves_remaining / 2
  let afternoon_revenue := afternoon_loaves_sold * price_afternoon
  let late_afternoon_loaves_sold := loaves_remaining - afternoon_loaves_sold
  let late_afternoon_revenue := late_afternoon_loaves_sold * price_late_afternoon
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue
  let total_cost := (loaves_baked * cost_per_loaf) + fixed_cost
  let profit := total_revenue - total_cost
in profit = 40 := 
  sorry

end bridget_profit_l255_255146


namespace male_contestants_l255_255564

theorem male_contestants (total_contestants : ℕ) (female_proportion : ℕ) (total_females : ℕ) :
  female_proportion = 3 ∧ total_contestants = 18 ∧ total_females = total_contestants / female_proportion →
  (total_contestants - total_females) = 12 :=
by
  sorry

end male_contestants_l255_255564


namespace number_of_subsets_divisible_by_3_l255_255926

def S (A : Finset ℕ) : ℕ := A.sum id

theorem number_of_subsets_divisible_by_3 :
  (Finset.filter (λ A : Finset ℕ, S A % 3 = 0) ((Finset.powerset (Finset.range 6)).filter (λ x, x ≠ ∅))).card = 11 := 
sorry

end number_of_subsets_divisible_by_3_l255_255926


namespace example_equation_l255_255858

-- Define what it means to be an equation in terms of containing an unknown and being an equality
def is_equation (expr : Prop) (contains_unknown : Prop) : Prop :=
  (contains_unknown ∧ expr)

-- Prove that 4x + 2 = 10 is an equation
theorem example_equation : is_equation (4 * x + 2 = 10) (∃ x : ℝ, true) :=
  by sorry

end example_equation_l255_255858


namespace tim_watched_total_hours_tv_l255_255487

-- Define the conditions
def short_show_episodes : ℕ := 24
def short_show_duration_per_episode : ℝ := 0.5

def long_show_episodes : ℕ := 12
def long_show_duration_per_episode : ℝ := 1

-- Define the total duration for each show
def short_show_total_duration : ℝ :=
  short_show_episodes * short_show_duration_per_episode

def long_show_total_duration : ℝ :=
  long_show_episodes * long_show_duration_per_episode

-- Define the total TV hours watched
def total_tv_hours_watched : ℝ :=
  short_show_total_duration + long_show_total_duration

-- Write the theorem statement
theorem tim_watched_total_hours_tv : total_tv_hours_watched = 24 := 
by
  -- proof goes here
  sorry

end tim_watched_total_hours_tv_l255_255487


namespace least_element_of_T_l255_255337

theorem least_element_of_T (T : Set ℕ) (H : T ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}) 
  (H_card : T.card = 5) 
  (H_cond : ∀ a b ∈ T, a < b → b ≠ 2 * a) : 
  ∃ x ∈ T, x = 3 :=
by
  -- Proof is to be provided 
  sorry

end least_element_of_T_l255_255337


namespace phi_value_l255_255733

theorem phi_value
  (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f x = sin (2 * x + φ) + √3 * cos (2 * x + φ))
  (h2 : ∀ x : ℝ, f x = f (-x))
  (h3 : 0 < φ ∧ φ < π) :
  φ = π / 6 :=
by sorry

end phi_value_l255_255733


namespace A_is_equidistant_l255_255659

-- Definitions for the points A, B, and C
def Point : Type := ℝ × ℝ × ℝ
def A (y : ℝ) : Point := (0, y, 0)
def B : Point := (0, -2, 4)
def C : Point := (-4, 0, 4)

-- Distance function in 3D space
def dist (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

-- Proof statement
theorem A_is_equidistant (y : ℝ) : dist (A y) B = dist (A y) C ↔ y = 3 := 
by
  sorry

end A_is_equidistant_l255_255659


namespace geom_seq_result_l255_255235

variable (a : ℕ → ℚ) (S : ℕ → ℚ)
variable (n : ℕ)

-- Conditions
axiom h1 : a 1 + a 3 = 5 / 2
axiom h2 : a 2 + a 4 = 5 / 4

-- General properties
axiom geom_seq_common_ratio : ∃ q : ℚ, ∀ n, a (n + 1) = a n * q

-- Sum of the first n terms of the geometric sequence
axiom S_def : S n = (2 * (1 - (1 / 2)^n)) / (1 - 1 / 2)

-- General term of the geometric sequence
axiom a_n_def : a n = 2 * (1 / 2)^(n - 1)

-- Result to be proved
theorem geom_seq_result : S n / a n = 2^n - 1 := 
  by sorry

end geom_seq_result_l255_255235


namespace ladder_of_twos_l255_255783

theorem ladder_of_twos (n : ℕ) (h : n ≥ 3) : 
  ∃ N_n : ℕ, N_n = 2 ^ (n - 3) :=
by
  sorry

end ladder_of_twos_l255_255783


namespace calculation_l255_255150

-- Define the exponents and base values as conditions
def exponent : ℕ := 3 ^ 2
def neg_base : ℤ := -2
def pos_base : ℤ := 2

-- The calculation expressions as conditions
def term1 : ℤ := neg_base^exponent
def term2 : ℤ := pos_base^exponent

-- The proof statement: Show that the sum of the terms equals 0
theorem calculation : term1 + term2 = 0 := sorry

end calculation_l255_255150


namespace g_not_0_l255_255803

noncomputable def g (x : ℝ) : ℤ :=
  if x > -2 then ⌈ (1 / (2 * x + 4)) ⌉
  else ⌊ (1 / (2 * x + 4)) ⌋

theorem g_not_0 (x : ℝ) (H : x ≠ (-2)) : g(x) ≠ 0 :=
by
  sorry

end g_not_0_l255_255803


namespace solution_set_of_inequality_l255_255901

theorem solution_set_of_inequality (a x : ℝ) (h : a > 1) :
  (|x - log a x| < |x| + |log a x|) ↔ (x > 1) := 
sorry

end solution_set_of_inequality_l255_255901


namespace largest_multiple_of_8_less_than_120_l255_255069

theorem largest_multiple_of_8_less_than_120 : 
  ∃ k : ℕ, 8 * k < 120 ∧ ∀ n : ℕ, 8 * n < 120 → 8 * n ≤ 8 * k := 
begin
  use 14,
  split,
  { linarith, },
  { intros n hn,
    have hnk : n < 15,
    { calc n ≤ n : by linarith [hn, mul_le_mul_left (by norm_num : 8 > 0)], },
    linarith, }
end

end largest_multiple_of_8_less_than_120_l255_255069


namespace third_degree_polynomial_roots_l255_255172

theorem third_degree_polynomial_roots (α β γ : ℝ) (x1 x2 x3 : ℝ) :
  (α = x1 + x2) ∧ (β = x1 + x3) ∧ (γ = x2 + x3) ∧
  (Polynomial.roots (Polynomial.C (-46) + Polynomial.C 44 * Polynomial.X + Polynomial.C (-12) * Polynomial.X ^ 2 + Polynomial.C 1 * Polynomial.X ^ 3) = {α, β, γ}) →
  Polynomial.roots (Polynomial.C (-2) + Polynomial.C 8 * Polynomial.X + Polynomial.C (-6) * Polynomial.X ^ 2 + Polynomial.C 1 * Polynomial.X ^ 3) = {x1, x2, x3} :=
begin
  sorry
end

end third_degree_polynomial_roots_l255_255172


namespace standard_concession_l255_255126

theorem standard_concession (x : ℝ) : 
  (∀ (x : ℝ), (2000 - (x / 100) * 2000) - 0.2 * (2000 - (x / 100) * 2000) = 1120) → x = 30 := 
by 
  sorry

end standard_concession_l255_255126


namespace hexagon_chord_l255_255549

theorem hexagon_chord (m n : ℕ) (rel_prime : Nat.gcd m n = 1) 
  (h₁ : ∃ A B C D E F : ℕ, A = 4 ∧ B = 4 ∧ C = 4 ∧ D = 6 ∧ E = 6 ∧ F = 6)
  (h₂ : inscribed_in_circle hexagon)
  (h₃ : divides_chord hexagon = m / n) : m + n = 13 := 
sorry

end hexagon_chord_l255_255549


namespace coin_toss_problem_l255_255621

def coin_toss_sequences : Nat :=
  let T_combinations := binomial 12 5  -- 792
  let H_combinations := binomial 7 4   -- 35
  T_combinations * H_combinations      -- 27720

theorem coin_toss_problem :
  coin_toss_sequences = 27720 :=
by
  unfold coin_toss_sequences
  sorry

end coin_toss_problem_l255_255621


namespace inscribed_angles_sum_l255_255853

theorem inscribed_angles_sum (n : ℕ) (a b : ℕ) (total_degrees : ℝ) (arcs : list ℕ) :
  n = 16 →
  total_degrees = 360 →
  arcs = [3, 5] →
  (a = list.nth arcs 0).get_or_else 0 * (total_degrees / n) / 2 +
  (b = list.nth arcs 1).get_or_else 0 * (total_degrees / n) / 2 = 90 :=
by sorry

end inscribed_angles_sum_l255_255853


namespace value_of_n_l255_255477

theorem value_of_n 
  {a b n : ℕ} (ha : a > 0) (hb : b > 0) 
  (h : (1 + b)^n = 243) : 
  n = 5 := by 
  sorry

end value_of_n_l255_255477


namespace min_odd_in_A_P_l255_255387

-- Define the polynomial P of degree 8
variable {R : Type*} [Ring R] (P : Polynomial R)
variable (hP : degree P = 8)

-- Define the set A_P for some value c in R
def A_P (c : R) : Set R := { x | P.eval x = c }

-- Given that the number 8 is included in the set A_P
variable (c : R) (h8 : (8 : R) ∈ A_P c)

-- Prove that the minimum number of odd numbers in the set A_P is 1
theorem min_odd_in_A_P : ∃ x ∈ A_P c, (x % 2 = 1) := sorry

end min_odd_in_A_P_l255_255387


namespace equation_of_line_intersecting_circle_l255_255036

theorem equation_of_line_intersecting_circle (A B : ℝ×ℝ)
    (l : ℝ → ℝ)
    (midpoint_AB : ℝ×ℝ)
    (h_circle : ∀ (P : ℝ×ℝ), (fst P)^2 + (snd P)^2 + 2 * (fst P) - 4 * (snd P) + 1 = 0 → (P = A ∨ P = B))
    (h_midpoint : midpoint_AB = (-2, 3))
    (h_line_through_AB : ∀ (P : ℝ×ℝ), P = A ∨ P = B → snd P = l (fst P))
    : (∀ x, l x = x - 5 → ∀ (P : ℝ×ℝ), snd P = l (fst P) → (P = A ∨ P = B)) := 
sorry

end equation_of_line_intersecting_circle_l255_255036


namespace sum_first_80_terms_l255_255040

theorem sum_first_80_terms (a : ℕ → ℝ) (h : ∀ n, a (n + 1) + (-1)^n * a n = 2 * n - 1) :
  (∑ i in finset.range 80, a i) = 3240 :=
sorry

end sum_first_80_terms_l255_255040


namespace find_a_l255_255014

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom cond1 : a^2 / b = 5
axiom cond2 : b^2 / c = 3
axiom cond3 : c^2 / a = 7

theorem find_a : a = 15 := sorry

end find_a_l255_255014


namespace cat_mouse_same_cell_l255_255934

theorem cat_mouse_same_cell (m n : ℕ) (h_m : m > 1) (h_n : n > 1) :
  (m % 2 = 1 ∧ n % 2 = 1) ↔ ∃ k : ℕ, cat_position k = mouse_position k :=
sorry

-- Definitions for positions of cat and mouse which depend on the grid size and move counts,
-- assuming (m-1, n-1) is the start position for cat and (0, 0) is the start for mouse.
def cat_position (k : ℕ) : ℕ × ℕ := (m - 1 - k, n - 1 - k)
def mouse_position (k : ℕ) : ℕ × ℕ := (k, k)

end cat_mouse_same_cell_l255_255934


namespace rational_sum_is_negative_then_at_most_one_positive_l255_255904

theorem rational_sum_is_negative_then_at_most_one_positive (a b : ℚ) (h : a + b < 0) :
  (a > 0 ∧ b ≤ 0) ∨ (a ≤ 0 ∧ b > 0) ∨ (a ≤ 0 ∧ b ≤ 0) :=
by
  sorry

end rational_sum_is_negative_then_at_most_one_positive_l255_255904


namespace average_age_decrease_l255_255453

-- defining average_age_decrease as the main problem statement
theorem average_age_decrease :
  let O := 12 in  -- original number of students
  let N := 12 in  -- number of new students
  let original_average := 40 in  -- original average age in years
  let new_average := 32 in  -- average age of new students in years
  let total_age_original := O * original_average in  -- total age of original class
  let total_age_new := N * new_average in  -- total age of new students
  let total_strength := O + N in  -- total number of students
  -- calculation for the decrease in average age
  let decrease := original_average - (total_age_original + total_age_new) / total_strength in
  decrease = 4 := 
by 
  -- should be calculated based on the arithmetic given conditions
  sorry

end average_age_decrease_l255_255453


namespace repeating_decimal_to_fraction_l255_255631

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + 0.066666... ) : x = 11 / 30 :=
sorry

end repeating_decimal_to_fraction_l255_255631


namespace infinite_sequence_eventually_arithmetic_l255_255674

theorem infinite_sequence_eventually_arithmetic (a : ℕ → ℕ) (k : ℕ) (hk : 2 ≤ k) 
    (P : Polynomial ℤ) (Hform : P.degree = k)
    (Hcoef : ∀ i, i < k → P.coeff i ≥ 0)
    (Hrec : ∀ n, P.eval (a n) = (∏ i in Finset.range k, a (n + i + 1))) :
    ∃ d, ∀ N, ∀ n ≥ N, a (n + 1) = a n + d :=
    sorry

end infinite_sequence_eventually_arithmetic_l255_255674


namespace percent_of_x_is_y_l255_255522

variables (x y : ℝ)

theorem percent_of_x_is_y (h : 0.30 * (x - y) = 0.20 * (x + y)) : y = 0.20 * x :=
by sorry

end percent_of_x_is_y_l255_255522


namespace books_sold_l255_255831

theorem books_sold (initial_books sold_books remaining_books : ℕ) 
  (h_initial : initial_books = 242) 
  (h_remaining : remaining_books = 105)
  (h_relation : sold_books = initial_books - remaining_books) :
  sold_books = 137 := 
by
  sorry

end books_sold_l255_255831


namespace tens_digit_of_23_pow_2057_l255_255628

theorem tens_digit_of_23_pow_2057 : (23^2057 % 100) / 10 % 10 = 6 := 
by
  sorry

end tens_digit_of_23_pow_2057_l255_255628


namespace cos_sum_arcsin_arccos_l255_255093

theorem cos_sum_arcsin_arccos : 
  cos (arcsin (3/5) + arccos (-5/13)) = -56/65 := 
by
  sorry

end cos_sum_arcsin_arccos_l255_255093


namespace min_odd_in_A_P_l255_255350

-- Define the polynomial P of degree 8
variables (P : Polynomial ℝ)
hypothesis degree_P : P.natDegree = 8

-- Define the set A_P
def A_P (P : Polynomial ℝ) : Set ℝ := { x : ℝ | True }

-- Hypothesis that the set A_P includes the number 8
hypothesis A_P_contains_8 : 8 ∈ A_P P

-- The minimum number of odd numbers in the set A_P
theorem min_odd_in_A_P : ∃ n : ℕ, odd n ∧ n = 1 := sorry

end min_odd_in_A_P_l255_255350


namespace total_cases_after_third_day_l255_255972

-- Definitions for the conditions
def day1_cases : Nat := 2000
def day2_new_cases : Nat := 500
def day2_recoveries : Nat := 50
def day3_new_cases : Nat := 1500
def day3_recoveries : Nat := 200

-- Theorem stating the total number of cases after the third day
theorem total_cases_after_third_day : day1_cases + (day2_new_cases - day2_recoveries) + (day3_new_cases - day3_recoveries) = 3750 :=
by
  sorry

end total_cases_after_third_day_l255_255972


namespace sum_values_u_t_l255_255810

-- Definitions based on given conditions
def t : Int → Int
def u : Int → Int := λ x => x + 2
def t_domain : Set Int := {-2, -1, 0, 1}
def t_range : Set Int := {-1, 1, 3, 5}
def u_domain : Set Int := {0, 1, 2, 3}

-- Assumptions (these statements should be proved as separate lemmas if needed)
axiom t_injective : ∀ x, x ∈ t_domain → t(x) ∈ t_range
axiom u_defined : ∀ x, x ∈ u_domain → u(x) = x + 2

-- Statement of the problem to be proven
theorem sum_values_u_t : (∑ x in {1, 3}, u x) = 8 :=
by
  sorry

end sum_values_u_t_l255_255810


namespace probability_yellow_second_l255_255144

section MarbleProbabilities

def bag_A := (5, 6)     -- (white marbles, black marbles)
def bag_B := (3, 7)     -- (yellow marbles, blue marbles)
def bag_C := (5, 6)     -- (yellow marbles, blue marbles)

def P_white_A := 5 / 11
def P_black_A := 6 / 11
def P_yellow_given_B := 3 / 10
def P_yellow_given_C := 5 / 11

theorem probability_yellow_second :
  P_white_A * P_yellow_given_B + P_black_A * P_yellow_given_C = 33 / 121 :=
by
  -- Proof would be provided here
  sorry

end MarbleProbabilities

end probability_yellow_second_l255_255144


namespace regular_tiling_extend_l255_255064

-- Define a tiling as an arrangement of units into a rectangular grid.
-- Let's define what it means for a tiling to be "regular".
def regular_tiling (m n : ℕ) : Prop :=
  ∀ (i1 i2 j1 j2 : ℕ), 
    0 ≤ i1 ∧ i1 < i2 ∧ i2 ≤ m ∧ 
    0 ≤ j1 ∧ j1 < j2 ∧ j2 ≤ n → 
    ¬ (i2 - i1 = 2 ∧ j2 - j1 = 2)

-- The theorem that states the desired property
theorem regular_tiling_extend (m n : ℕ) :
  regular_tiling m n → regular_tiling (2 * m) (2 * n) :=
begin
  intro h,
  sorry -- Proof not provided
end

end regular_tiling_extend_l255_255064


namespace proof_problem_l255_255519

noncomputable def expr (a b : ℚ) : ℚ :=
  ((a / b + b / a + 2) * ((a + b) / (2 * a) - (b / (a + b)))) /
  ((a + 2 * b + b^2 / a) * (a / (a + b) + b / (a - b)))

theorem proof_problem : expr (3/4 : ℚ) (4/3 : ℚ) = -7/24 :=
by
  sorry

end proof_problem_l255_255519


namespace probability_of_two_boys_l255_255544

def club_has_15_members := 15
def number_of_boys := 8
def number_of_girls := 7

def total_ways_to_choose_2_members : ℕ :=
  (Nat.choose club_has_15_members 2)

def ways_to_choose_2_boys : ℕ :=
  (Nat.choose number_of_boys 2)

theorem probability_of_two_boys : (ways_to_choose_2_boys : ℚ) / (total_ways_to_choose_2_members : ℚ) = (4 / 15 : ℚ) :=
  by
  sorry

end probability_of_two_boys_l255_255544


namespace find_sums_l255_255092

-- Definitions for the conditions
def is_regular_pentagon (ABCDE : Set Point) : Prop := sorry
def perpendicular_from (P Q : Point) (L : Set Point) : Prop := sorry
def center_of (O : Point) (ABCDE : Set Point) : Prop := sorry

variable (A P Q R O : Point) (ABCDE : Set Point)
variable (OP : ℝ)
variable (AP AQ AR OA : ℝ)

-- The conditions
axiom pentagon_is_regular : is_regular_pentagon ABCDE
axiom AP_perpendicular_CD : perpendicular_from A P (segment C D)
axiom AQ_perpendicular_BC_extended : perpendicular_from A Q (extension B C)
axiom AR_perpendicular_DE_extended : perpendicular_from A R (extension D E)
axiom O_center_of_pentagon : center_of O ABCDE
axiom OP_is_two : OP = 2

-- The statement to prove
theorem find_sums : AO + AQ + AR = 8 := 
by
  sorry

end find_sums_l255_255092


namespace ratio_of_areas_l255_255128

theorem ratio_of_areas (s : ℝ) (h : s > 0) :
  let r := s / 2 in
  let s1 := s / Real.sqrt 2 in
  let area_circle := π * r^2 in
  let area_square := s1^2 in
  area_square / area_circle = 2 / π :=
by
  have hr : r = s / 2 := rfl
  have hs1 : s1 = s / Real.sqrt 2 := rfl
  have h_area_circle : area_circle = π * (s / 2)^2 := rfl
  have h_area_square : area_square = (s / Real.sqrt 2)^2 := rfl
  sorry

end ratio_of_areas_l255_255128


namespace prime_sum_product_eq_1978_l255_255805

open Nat

theorem prime_sum_product_eq_1978 (x₁ x₂ x₃ : ℕ) (h₁ : Prime x₁) (h₂ : Prime x₂) (h₃ : Prime x₃) 
(h₄ : x₁ + x₂ + x₃ = 68) (h₅ : x₁ * x₂ + x₁ * x₃ + x₂ * x₃ = 1121) : 
  x₁ * x₂ * x₃ = 1978 := by
  sorry

end prime_sum_product_eq_1978_l255_255805


namespace closure_of_M_is_closed_interval_l255_255798

noncomputable def U : Set ℝ := Set.univ

noncomputable def M : Set ℝ := {a | a^2 - 2 * a > 0}

theorem closure_of_M_is_closed_interval :
  closure M = {a | 0 ≤ a ∧ a ≤ 2} :=
by
  sorry

end closure_of_M_is_closed_interval_l255_255798


namespace div_seq_l255_255788

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 1, a (n + 1) = a n + a (nat.floor (real.sqrt n))

theorem div_seq (a : ℕ → ℕ) (h_sequence : sequence a) (k : ℕ) (hk : k > 0) : ∃ m ≥ 1, k ∣ a m := by
  sorry

end div_seq_l255_255788


namespace count_correct_conclusions_l255_255620

structure Point where
  x : ℝ
  y : ℝ

def isDoublingPoint (P Q : Point) : Prop :=
  2 * (P.x + Q.x) = P.y + Q.y

def P1 : Point := {x := 2, y := 0}

def Q1 : Point := {x := 2, y := 8}
def Q2 : Point := {x := -3, y := -2}

def onLine (P : Point) : Prop :=
  P.y = P.x + 2

def onParabola (P : Point) : Prop :=
  P.y = P.x ^ 2 - 2 * P.x - 3

def dist (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

theorem count_correct_conclusions :
  (isDoublingPoint P1 Q1) ∧
  (isDoublingPoint P1 Q2) ∧
  (∃ A : Point, onLine A ∧ isDoublingPoint P1 A ∧ A = {x := -2, y := 0}) ∧
  (∃ B₁ B₂ : Point, onParabola B₁ ∧ onParabola B₂ ∧ isDoublingPoint P1 B₁ ∧ isDoublingPoint P1 B₂) ∧
  (∃ B : Point, isDoublingPoint P1 B ∧
   ∀ P : Point, isDoublingPoint P1 P → dist P1 P ≥ dist P1 B ∧
   dist P1 B = 8 * (5:ℝ)^(1/2) / 5) :=
by sorry

end count_correct_conclusions_l255_255620


namespace solve_for_a_find_range_of_t_l255_255712

-- Define the function f(x)
def f (x a : ℝ) : ℝ := |2 * x - a| + a

-- First part: solve for a
theorem solve_for_a (a : ℝ) : (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f x a ≤ 4) ↔ a = 1 := sorry

-- Second part: find the range of t
theorem find_range_of_t (t : ℝ) 
  (h : ∃ n : ℝ, f n 1 ≤ t - f (-n) 1) :
  t ∈ set.Ici (4 : ℝ) := sorry

end solve_for_a_find_range_of_t_l255_255712


namespace matrix_inverse_correct_l255_255178

variable A : Matrix (Fin 2) (Fin 2) ℚ := ![![9, 18], ![6, 13]]
def B : Matrix (Fin 2) (Fin 2) ℚ := ![![13 / 9, -2], ![-2 / 3, 1]]

theorem matrix_inverse_correct : (A.det ≠ 0) → A⁻¹ = B :=
by
  sorry

end matrix_inverse_correct_l255_255178


namespace max_third_side_triangle_l255_255020

theorem max_third_side_triangle (P Q R : ℝ) (a b c : ℝ) 
  (h_angles : P + Q + R = π)
  (h_sides : {a, b, c} = {7, 24, x}) :
  (cos (2 * P) + cos (2 * Q) + cos (2 * R) = 1) →
  max (max a b) c = 25 :=
sorry

end max_third_side_triangle_l255_255020


namespace nicky_profit_l255_255819

theorem nicky_profit (value_traded_away value_received : ℤ)
  (h1 : value_traded_away = 2 * 8)
  (h2 : value_received = 21) :
  value_received - value_traded_away = 5 :=
by
  sorry

end nicky_profit_l255_255819


namespace true_discount_is_120_l255_255454

-- Definitions for given conditions
def BD : ℝ := 144
def PV : ℝ := 720

-- Definition of True Discount based on the given formula
def TD (BD PV : ℝ) : ℝ := BD / (1 + (BD / PV))

-- Theorem stating the required proof
theorem true_discount_is_120 : TD BD PV = 120 := 
by
  -- Proof omitted
  sorry

end true_discount_is_120_l255_255454


namespace coeff_x3_in_expansion_l255_255658

theorem coeff_x3_in_expansion : 
  let f := (1 + 2 * x^2) * (1 + x)^5,
      coeff := (n : ℕ) → (p : ℕ → ℤ) → ℤ := sorry
  in coeff 3 f = 20 :=
sorry

end coeff_x3_in_expansion_l255_255658


namespace complete_blue_or_red_subgraph_l255_255170

open Finset

-- Definitions of being a complete graph and colored edges
def is_complete_graph (G : SimpleGraph (Fin 9)) : Prop :=
  ∀ u v : Fin 9, u ≠ v → G.adj u v

def edge_colored (G : SimpleGraph (Fin 9)) : Type :=
  { f : G.edge_set → Prop // ∀ e, f e ∨ ¬f e }

noncomputable def blue_edges (G : SimpleGraph (Fin 9)) (f : edge_colored G) : G.edge_set → Prop :=
  f.1

noncomputable def red_edges (G : SimpleGraph (Fin 9)) (f : edge_colored G) : G.edge_set → Prop :=
  λ e, ¬(f.1 e)

-- The main theorem statement
theorem complete_blue_or_red_subgraph (G : SimpleGraph (Fin 9)) (hG : is_complete_graph G)
  (f : edge_colored G) :
  ∃ (S : Finset (Fin 9)), (S.card = 4 ∧ ∀ u v ∈ S, u ≠ v → blue_edges G f ⟨u, v, hG u v⟩) ∨ 
  ∃ (T : Finset (Fin 9)), (T.card = 3 ∧ ∀ u v ∈ T, u ≠ v → red_edges G f ⟨u, v, hG u v⟩) :=
sorry

end complete_blue_or_red_subgraph_l255_255170


namespace work_problem_l255_255520

theorem work_problem (W : ℝ) (A B C : ℝ)
  (h1 : B + C = W / 24)
  (h2 : C + A = W / 12)
  (h3 : C = W / 32) : A + B = W / 16 := 
by
  sorry

end work_problem_l255_255520


namespace trigonometric_problem_l255_255240

theorem trigonometric_problem (a : ℝ) (α β x t : ℝ)
  (cond_a : a > 1)
  (cond_roots : t^2 + 4 * a * t + 3 * a + 1 = 0)
  (cond_alpha_beta : α ∈ Ioo (-π / 2) (π / 2) ∧ β ∈ Ioo (-π / 2) (π / 2))
  (cond_x : x = α + β)
  (root_cond1 : tan α + tan β = -4 * a)
  (root_cond2 : tan α * tan β = 3 * a + 1) :
  tan x = 4 / 3 ∧ (cos (2 * x) / (sqrt 2 * cos ((π / 4) + x) * sin x) = 7 / 4) :=
sorry

end trigonometric_problem_l255_255240


namespace number_of_possible_scores_l255_255533

theorem number_of_possible_scores : ∃ n, n = 6 ∧ 
  ∀ t (b : ℕ → ℕ), 
  t ∈ (set.range (λ k, k * 3 + (5 - k) * 2)) ↔ t = b 10 ∨ t = b 11 ∨ t = b 12 ∨ t = b 13 ∨ t = b 14 ∨ t = b 15 :=
sorry

end number_of_possible_scores_l255_255533


namespace basketball_team_win_percentage_l255_255096

theorem basketball_team_win_percentage
  (wins_first_75 : ℕ)
  (games_first_75 : ℕ)
  (remaining_games : ℕ)
  (desired_percentage : ℚ) :
  wins_first_75 = 60 →
  games_first_75 = 75 →
  remaining_games = 45 →
  desired_percentage = 0.8 →
  ∃ x : ℕ, x = 36 ∧ (wins_first_75 + x) / (games_first_75 + remaining_games : ℚ) = desired_percentage := by
  intros h1 h2 h3 h4
  use 36
  split
  · rfl
  · rw [h1, h2, h3, h4]
    norm_num
    rw ←rat.div_num_eq_div_mul
    norm_num
    rw rat.div_num_eq_div_mul
    rw rat.mul_div_cancel_left
    norm_num
    norm_num
    norm_num
    sorry 

end basketball_team_win_percentage_l255_255096


namespace range_of_set_d_l255_255524

open Set

-- Definition of the set of prime numbers between 10 and 25
def prime_between (a b : ℕ) : Set ℕ := {n | nat.prime n ∧ a ≤ n ∧ n ≤ b}

def d : Set ℕ := prime_between 10 25

-- The statement of the problem: to prove the range of set d is 12
theorem range_of_set_d : range (d) = 12 := by
  sorry

end range_of_set_d_l255_255524


namespace number_of_elderly_employees_in_sample_l255_255542

variables (total_employees young_employees sample_young_employees elderly_employees : ℕ)
variables (sample_total : ℕ)

def conditions (total_employees young_employees sample_young_employees elderly_employees : ℕ) :=
  total_employees = 430 ∧
  young_employees = 160 ∧
  sample_young_employees = 32 ∧
  (∃ M, M = 2 * elderly_employees ∧ elderly_employees + M + young_employees = total_employees)

theorem number_of_elderly_employees_in_sample
  (total_employees young_employees sample_young_employees elderly_employees : ℕ)
  (sample_total : ℕ) :
  conditions total_employees young_employees sample_young_employees elderly_employees →
  sample_total = 430 * 32 / 160 →
  sample_total = 90 * 32 / 430 :=
by
  sorry

end number_of_elderly_employees_in_sample_l255_255542


namespace breadth_of_rectangular_plot_l255_255919

theorem breadth_of_rectangular_plot (b l : ℝ) (h1 : l = 3 * b) (h2 : l * b = 432) : b = 12 := 
sorry

end breadth_of_rectangular_plot_l255_255919


namespace n_is_power_of_2_l255_255835

theorem n_is_power_of_2 (n : ℤ) (h₀ : n ≥ 4) (h₁ : n ∈ ℤ)
  (h₂ : ∃ m : ℕ, nat.floor (2^n / n) = 2^m) : ∃ a : ℕ, n = 2^a := 
sorry

end n_is_power_of_2_l255_255835


namespace solve_for_m_l255_255149

theorem solve_for_m (a m : ℝ) : (a + m) * (a + 1 / 2) = a^2 + (m + 1 / 2) * a + (1 / 2) * m → m = -1 / 2 := 
by
  intro h
  have h1 : (m + 1 / 2) * a = 0 := sorry -- No term involving a to the first power
  have h2 : m + 1 / 2 = 0 := sorry -- Coefficient must be zero
  show m = -1 / 2 from sorry

end solve_for_m_l255_255149


namespace value_calculation_l255_255930

-- Definition of constants used in the problem
def a : ℝ := 1.3333
def b : ℝ := 3.615
def expected_value : ℝ := 4.81998845

-- The proposition to be proven
theorem value_calculation : a * b = expected_value :=
by sorry

end value_calculation_l255_255930


namespace roots_outside_unit_circle_l255_255395

theorem roots_outside_unit_circle
  (n : ℕ)
  (a : Fin (n + 1) → ℝ)
  (h : ∀ i j : Fin (n + 1), i < j → 0 < a i ∧ a i < a j) :
  ∀ r : ℂ, (∃ k : Fin (n + 1), a k ≠ 0) → (polynomial.eval r (polynomial.sum
    (λ i, polynomial.C (a (Fin.replicate i)) * polynomial.X ^ i))) = 0 → abs r > 1 :=
by
  sorry

end roots_outside_unit_circle_l255_255395


namespace initial_ratio_l255_255824

-- Define the initial number of horses and cows
def initial_horses (H : ℕ) : Prop := H = 120
def initial_cows (C : ℕ) : Prop := C = 20

-- Define the conditions of the problem
def condition1 (H C : ℕ) : Prop := H - 15 = 3 * (C + 15)
def condition2 (H C : ℕ) : Prop := H - 15 = C + 15 + 70

-- The statement that initial ratio is 6:1
theorem initial_ratio (H C : ℕ) (h1 : condition1 H C) (h2 : condition2 H C) : 
  H = 6 * C :=
by {
  sorry
}

end initial_ratio_l255_255824


namespace judy_shopping_trip_l255_255171

-- Define the quantities and prices of the items
def num_carrots : ℕ := 5
def price_carrot : ℕ := 1
def num_milk : ℕ := 4
def price_milk : ℕ := 3
def num_pineapples : ℕ := 2
def price_pineapple : ℕ := 4
def num_flour : ℕ := 2
def price_flour : ℕ := 5
def price_ice_cream : ℕ := 7

-- Define the promotion conditions
def pineapple_promotion : ℕ := num_pineapples / 2

-- Define the coupon condition
def coupon_threshold : ℕ := 40
def coupon_value : ℕ := 10

-- Define the total cost without coupon
def total_cost : ℕ := 
  (num_carrots * price_carrot) + 
  (num_milk * price_milk) +
  (pineapple_promotion * price_pineapple) +
  (num_flour * price_flour) +
  price_ice_cream

-- Define the final cost considering the coupon condition
def final_cost : ℕ :=
  if total_cost < coupon_threshold then total_cost else total_cost - coupon_value

-- The theorem to be proven
theorem judy_shopping_trip : final_cost = 38 := by
  sorry

end judy_shopping_trip_l255_255171


namespace age_difference_l255_255087

variable (a b c d : ℕ)
variable (h1 : a + b = b + c + 11)
variable (h2 : a + c = c + d + 15)
variable (h3 : b + d = 36)
variable (h4 : a * 2 = 3 * d)

theorem age_difference :
  a - b = 39 :=
by
  sorry

end age_difference_l255_255087


namespace min_odd_in_A_P_l255_255348

-- Define the polynomial P of degree 8
variables (P : Polynomial ℝ)
hypothesis degree_P : P.natDegree = 8

-- Define the set A_P
def A_P (P : Polynomial ℝ) : Set ℝ := { x : ℝ | True }

-- Hypothesis that the set A_P includes the number 8
hypothesis A_P_contains_8 : 8 ∈ A_P P

-- The minimum number of odd numbers in the set A_P
theorem min_odd_in_A_P : ∃ n : ℕ, odd n ∧ n = 1 := sorry

end min_odd_in_A_P_l255_255348


namespace incorrect_option_l255_255515

theorem incorrect_option :
  (¬ (∀ x : ℝ, x ≠ 0 → x^2 - 3 * x ≠ 0) = (∃ x : ℝ, x = 0 ∧ x^2 - 3 * x = 0)) ∧
  (¬ ∀ x : ℝ, real.log (x^2 - x + 1) ≥ 0) ∧
  (∃ x : ℝ, 3 * real.sin x = real.sqrt 3) ∧
  (¬ (∀ x : ℝ, if_vectors_collinear_then_converse_is_true x)) :=
begin
  sorry
end

noncomputable def if_vectors_collinear_then_converse_is_true (x : ℝ) : Prop :=
  x = 1 → let a := (-2 * x, 1), b := (-2, x) in (a.1 * b.2 = a.2 * b.1 ↔ x = 1)

end incorrect_option_l255_255515


namespace detect_counterfeit_coins_l255_255575

def Coin := ℕ

structure Expert where
  balance_scale : Prop
  real_coins : Finset Coin
  counterfeit_coins : Finset Coin
  all_coins : Finset Coin

theorem detect_counterfeit_coins (e : Expert) 
  (h_balance: e.balance_scale) 
  (h_real: e.real_coins.card = 5)
  (h_counterfeit: e.counterfeit_coins.card = 5)
  (h_all: e.all_coins.card = 12) 
: ∃ detect_number_of_counterfeit (c: Finset Coin), e.all_coins = e.real_coins ∪ e.counterfeit_coins ∧ detect_number_of_counterfeit.card ≤ 4 :=
sorry

end detect_counterfeit_coins_l255_255575


namespace min_value_x_3y_min_value_x_3y_iff_l255_255707

theorem min_value_x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + 4 * y = x * y) : x + 3 * y ≥ 25 :=
sorry

theorem min_value_x_3y_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + 4 * y = x * y) : x + 3 * y = 25 ↔ x = 10 ∧ y = 5 :=
sorry

end min_value_x_3y_min_value_x_3y_iff_l255_255707


namespace expected_no_advice_l255_255586

theorem expected_no_advice (n : ℕ) (p : ℝ) (h_p : 0 ≤ p ∧ p ≤ 1) : 
  (∑ j in finset.range n, (1 - p)^j) = (1 - (1 - p)^n) / p :=
by
  sorry

end expected_no_advice_l255_255586


namespace find_a_l255_255257

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x₀ a : ℝ) (h : f x₀ a - g x₀ a = 3) : a = -Real.log 2 - 1 :=
by
  sorry

end find_a_l255_255257


namespace expected_no_advice_l255_255601

theorem expected_no_advice {n : ℕ} (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  (Σ (j : ℕ) (hj : j < n), (1 - p)^j) = (1 - (1 - p)^n) / p :=
by
  sorry

end expected_no_advice_l255_255601


namespace least_k_value_l255_255523

theorem least_k_value (k : ℤ) (h : 0.00010101 * 10^k > 100) : k ≥ 6 := 
sorry

end least_k_value_l255_255523


namespace heartsuit_calc_l255_255160

-- Define the operation x ♡ y = 4x + 6y
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_calc : heartsuit 5 3 = 38 := by
  -- Proof omitted
  sorry

end heartsuit_calc_l255_255160


namespace shifted_parabola_eq_l255_255859

theorem shifted_parabola_eq :
  ∀ x : ℝ, (λ x, (x + 3) ^ 2 - 2) x = (λ x, (x + 3) ^ 2 + 2 - 4) x :=
by
  intro x
  conv_lhs { congr, skip, ring }
  congr
  sorry

end shifted_parabola_eq_l255_255859


namespace find_five_digit_numbers_l255_255173

theorem find_five_digit_numbers :
  {n : ℕ // 10000 ≤ n ∧ n < 100000 ∧
  let digits := (n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10) in
  digits.2 = 5 * digits.1 ∧
  digits.1 * digits.2 * digits.3 * digits.4 * digits.5 = 1000} =
  {15558, 15585, 15855} :=
by
  sorry

end find_five_digit_numbers_l255_255173


namespace magnitude_parallel_vectors_l255_255725

theorem magnitude_parallel_vectors (k : ℝ)
  (a : ℝ × ℝ := (-1, 2))
  (b : ℝ × ℝ := (2, k))
  (h_parallel : ∃ (λ: ℝ), a = (λ * (b.1), λ * b.2)) :
  (‖2 • a - b‖ = 4 * Real.sqrt 5) := 
by
  sorry

end magnitude_parallel_vectors_l255_255725


namespace determine_length_dc_l255_255766

theorem determine_length_dc (AB BD BC DC : ℝ) (h1: AB = 30) (h2: BD = 24) (h3: BC = 120) (h2a: sin (atan (BD/AB)) = 4/5) (h2b: sin (atan (BD/BC)) = 1/5): DC = 24 * sqrt(23) :=
by
  have h4: BD = (4 / 5) * AB := by sorry
  have h5: BC = 5 * BD := by sorry
  have h6: DC = sqrt (BC ^ 2 - BD ^ 2) := by sorry
  exact sorry

end determine_length_dc_l255_255766


namespace square_minus_sqrt_l255_255606

-- Variables and conditions
variable {y : ℝ}

-- The theorem to be proven
theorem square_minus_sqrt (y : ℝ) : (7 - real.sqrt (y^2 - 49))^2 = y^2 - 14 * real.sqrt (y^2 - 49) :=
sorry

end square_minus_sqrt_l255_255606


namespace range_of_function_l255_255312

theorem range_of_function (x : ℝ) : 
    (sqrt ((x - 1) / (x - 2))) = sqrt ((x - 1) / (x - 2)) → (x ≤ 1 ∨ x > 2) :=
by
  sorry

end range_of_function_l255_255312


namespace probability_sum_odd_less_than_10_l255_255062

theorem probability_sum_odd_less_than_10 :
  let possible_outcomes := 8 * 8
  let favorable_outcomes := 2 + 4 + 6 + 8
  possible_outcomes > 0 →
  (∃ r: ℚ, r = favorable_outcomes / possible_outcomes ∧ r = 5 / 16) :=
by
  intros
  let possible_outcomes := 8 * 8
  let favorable_outcomes := 2 + 4 + 6 + 8
  have h : possible_outcomes > 0 := by norm_num
  existsi (favorable_outcomes : ℚ) / (possible_outcomes : ℚ)
  split
  · norm_cast
    exact favorable_outcomes / possible_outcomes
  · norm_cast
    norm_num
  sorry

end probability_sum_odd_less_than_10_l255_255062


namespace samir_climbed_318_stairs_l255_255428

theorem samir_climbed_318_stairs 
  (S : ℕ)
  (h1 : ∀ {V : ℕ}, V = (S / 2) + 18 → S + V = 495) 
  (half_S : ∃ k : ℕ, S = k * 2) -- assumes S is even 
  : S = 318 := 
by
  sorry

end samir_climbed_318_stairs_l255_255428


namespace expected_number_of_explorers_no_advice_l255_255593

-- Define the problem
theorem expected_number_of_explorers_no_advice
  (n : ℕ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∑ j in Finset.range n, (1 - p) ^ j) / p = (1 - (1 - p) ^ n) / p := by
  sorry

end expected_number_of_explorers_no_advice_l255_255593


namespace tan_six_minus_tan_two_geo_seq_l255_255274

theorem tan_six_minus_tan_two_geo_seq (x : ℝ) (h : (cos x)^2 = (sin x) * (cos x) * (cot x)) :
  (tan x)^6 - (tan x)^2 = (sin x)^2 :=
sorry

end tan_six_minus_tan_two_geo_seq_l255_255274


namespace expected_no_advice_l255_255599

theorem expected_no_advice {n : ℕ} (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  (Σ (j : ℕ) (hj : j < n), (1 - p)^j) = (1 - (1 - p)^n) / p :=
by
  sorry

end expected_no_advice_l255_255599


namespace min_odd_numbers_in_A_P_l255_255357

theorem min_odd_numbers_in_A_P (P : ℝ → ℝ) 
  (hP : ∃ a8 a7 a6 a5 a4 a3 a2 a1 a0 : ℝ, P = λ x, a8 * x^8 + a7 * x^7 + a6 * x^6 + a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0)
  (h8_in_A_P : ∃ x : ℝ, P x = 8) : 
  (∀ x : ℝ, P x ∈ Set.A_P) → 
  (∃! y : ℝ, odd y ∧ y ∈ Set.A_P) := 
sorry

end min_odd_numbers_in_A_P_l255_255357


namespace locus_of_point_E_is_circular_arc_l255_255526

theorem locus_of_point_E_is_circular_arc {A B C D E : Type} 
  [Triangle A B C] (D_moving_on_AB : Moving D on segment AB) 
  (E_intersection : E is_intersection_of_common_external_tangent (incircle (triangle A C D)) (incircle (triangle B C D)) (line CD)) :
  ∃ (radius : ℝ), ∀ (D : Point AB), distance E C = radius :=
begin
  sorry
end

end locus_of_point_E_is_circular_arc_l255_255526


namespace num_natural_a_l255_255163

theorem num_natural_a (a b : ℕ) : 
  (a^2 + a + 100 = b^2) → ∃ n : ℕ, n = 4 := sorry

end num_natural_a_l255_255163


namespace caitlin_bracelets_l255_255984

/-- 
Caitlin makes bracelets to sell at the farmer’s market every weekend. 
Each bracelet takes twice as many small beads as it does large beads. 
If each bracelet uses 12 large beads, and Caitlin has 528 beads with equal amounts of large and small beads, 
prove that Caitlin can make 11 bracelets for this weekend.
-/
theorem caitlin_bracelets (total_beads large_beads_per_bracelet small_beads_per_bracelet total_large_beads total_small_beads bracelets : ℕ)
  (h1 : total_beads = 528)
  (h2 : total_beads = total_large_beads + total_small_beads)
  (h3 : total_large_beads = total_small_beads)
  (h4 : large_beads_per_bracelet = 12)
  (h5 : small_beads_per_bracelet = 2 * large_beads_per_bracelet)
  (h6 : bracelets = total_small_beads / small_beads_per_bracelet) : 
  bracelets = 11 := 
by {
  sorry
}

end caitlin_bracelets_l255_255984


namespace new_polyhedron_cannot_inscribe_sphere_l255_255974

/-- Define a truncated cube as a cube where each vertex is truncated by a plane, forming a tetrahedron. -/
structure TruncatedCube :=
  (original_cube : Cube)

/-- The condition that the new polyhedron cannot have an inscribed sphere. -/
theorem new_polyhedron_cannot_inscribe_sphere (tc : TruncatedCube) : 
  ¬ (∃ (p : Point3D), ∀ (f : Face tc), distance_to_face p f = distance_to_all_faces) :=
sorry

end new_polyhedron_cannot_inscribe_sphere_l255_255974


namespace marbles_count_l255_255111

theorem marbles_count (M : ℕ)
  (h_blue : (M / 2) = n_blue)
  (h_red : (M / 4) = n_red)
  (h_green : 27 = n_green)
  (h_yellow : 14 = n_yellow)
  (h_total : (n_blue + n_red + n_green + n_yellow) = M) :
  M = 164 :=
by
  sorry

end marbles_count_l255_255111


namespace min_odd_in_A_P_l255_255352

-- Define the polynomial P of degree 8
variables (P : Polynomial ℝ)
hypothesis degree_P : P.natDegree = 8

-- Define the set A_P
def A_P (P : Polynomial ℝ) : Set ℝ := { x : ℝ | True }

-- Hypothesis that the set A_P includes the number 8
hypothesis A_P_contains_8 : 8 ∈ A_P P

-- The minimum number of odd numbers in the set A_P
theorem min_odd_in_A_P : ∃ n : ℕ, odd n ∧ n = 1 := sorry

end min_odd_in_A_P_l255_255352


namespace largest_good_subset_element_l255_255400

def is_good_set (X : Set ℕ) : Prop :=
  ∃ (x y : ℕ), x ∈ X ∧ y ∈ X ∧ x < y ∧ x ∣ y

theorem largest_good_subset_element :
  let A := {x : ℕ | 1 ≤ x ∧ x ≤ 1997}
  ∃ (a : ℕ), a ≤ 665 ∧ (∀ (X : Finset ℕ), X.card = 999 → a ∈ X → is_good_set X) :=
begin
  sorry
end

end largest_good_subset_element_l255_255400


namespace rectangle_perimeter_l255_255948

variable (a b : ℝ)
variable (h1 : a * b = 24)
variable (h2 : a^2 + b^2 = 121)

theorem rectangle_perimeter : 2 * (a + b) = 26 := 
by
  sorry

end rectangle_perimeter_l255_255948


namespace transformed_set_mean_transformed_set_stddev_l255_255228

noncomputable def mean (s : Finset ℝ) : ℝ := (Finset.sum s id) / (s.card : ℝ)
noncomputable def stddev (s : Finset ℝ) : ℝ := Real.sqrt ((mean (s.map (fun x => x^2)) - (mean s)^2) : ℝ)

variable {n : ℕ} (s : Finset ℝ)

-- Given conditions
axiom mean_s : mean s = 4
axiom stddev_s : stddev s = 7

theorem transformed_set_mean : mean (s.map (fun x => 3 * x + 2)) = 14 := by
  sorry

theorem transformed_set_stddev : stddev (s.map (fun x => 3 * x + 2)) = 21 := by
  sorry

end transformed_set_mean_transformed_set_stddev_l255_255228


namespace arithmetic_sequence_problem_l255_255765

variable {a : ℕ → ℕ}

-- Define the arithmetic sequence condition
axiom arithmetic_seq (a : ℕ → ℕ) (d : ℕ) : 
  ∀ n : ℕ, a n = a 1 + (n - 1) * d

-- Define the specific condition from the problem
axiom condition (a : ℕ → ℕ) : 
  a 2 + 4 * a 7 + a 12 = 96

-- The main theorem to prove
theorem arithmetic_sequence_problem (a : ℕ → ℕ) (d : ℕ) 
  [arithmetic_seq a d] [condition a] : 
  2 * a 3 + a 15 = 48 := 
sorry

end arithmetic_sequence_problem_l255_255765


namespace number_subtracted_l255_255187

theorem number_subtracted (x : ℝ) : 3 + 2 * (8 - x) = 24.16 → x = -2.58 :=
by
  intro h
  sorry

end number_subtracted_l255_255187


namespace domain_of_f_l255_255855

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x / (10 - x)))

theorem domain_of_f (x : ℝ) : (0 ≤ x ∧ x < 10) -> ∃ y : ℝ, y = f(x) :=
begin
  intro h,
  have h1 : 10 - x > 0 := by
  {
    exact (sub_pos.mpr h.2),
  },
  have h2 : x / (10 - x) ≥ 0 := by
  {
    exact (div_nonneg h.1 (le_of_lt h1)),
  },
  have h3 : ∃ (y : ℝ), y = Real.sqrt (x / (10 - x)) := by
  {
    use Real.sqrt (x / (10 - x)),
    refl,
  },
  exact h3,
end

end domain_of_f_l255_255855


namespace find_point_B_l255_255790

noncomputable def parabola : ℝ → ℝ := λ x, x^2

def point_A : ℝ × ℝ := (1, 1)

def normal_slope (m : ℝ) : ℝ := -1 / m

def tangent_slope (x : ℝ) : ℝ := 2 * x

def normal_line (A : ℝ × ℝ) : ℝ → ℝ :=
  λ x, A.snd - normal_slope (tangent_slope A.fst) * (x - A.fst)

def intersection_B : ℝ × ℝ :=
  let x := -3 / 2 in (x, parabola x)

theorem find_point_B :
  ∃ B : ℝ × ℝ, B = intersection_B ∧ B ∈ setOf (λ P, parabola P.fst = P.snd) := 
begin
  use intersection_B,
  split,
  {refl},
  {
    simp [intersection_B, parabola],
  }
end

end find_point_B_l255_255790


namespace min_odd_in_A_P_l255_255365

-- Define the polynomial P of degree 8
variable (P : ℝ → ℝ)
-- Assume P is a polynomial of degree 8
axiom degree_P : degree P = 8

-- Define the set A_P as the set of all x for which P(x) gives a certain value
def A_P (c : ℝ) : Set ℝ := { x | P x = c }

-- Given condition
axiom include_8 : 8 ∈ A_P (P 8)

-- Define if a number is odd
def is_odd (n : ℝ) : Prop := ∃k : ℤ, n = 2 * k + 1

-- The minimum number of odd numbers that are in the set A_P,
-- given that the number 8 is included in A_P, is 1

theorem min_odd_in_A_P : ∀ P, (degree P = 8) → 8 ∈ A_P (P 8) → ∃ x ∈ A_P (P 8), is_odd x := 
by
  sorry -- proof goes here

end min_odd_in_A_P_l255_255365


namespace sum_consecutive_integers_l255_255758

theorem sum_consecutive_integers (S : ℕ) (hS : S = 221) :
  ∃ (k : ℕ) (hk : k ≥ 2) (n : ℕ), 
    (S = k * n + (k * (k - 1)) / 2) → k = 2 := sorry

end sum_consecutive_integers_l255_255758


namespace net_change_is_minus_0_19_l255_255048

-- Define the yearly change factors as provided in the conditions
def yearly_changes : List ℚ := [6/5, 11/10, 7/10, 4/5, 11/10]

-- Compute the net change over the five years
def net_change (changes : List ℚ) : ℚ :=
  changes.foldl (λ acc x => acc * x) 1 - 1

-- Define the target value for the net change
def target_net_change : ℚ := -19 / 100

-- The theorem to prove the net change calculated matches the target net change
theorem net_change_is_minus_0_19 : net_change yearly_changes = target_net_change :=
  by
    sorry

end net_change_is_minus_0_19_l255_255048


namespace add_and_round_l255_255971

theorem add_and_round :
  let x := 46.913
  let y := 58.27
  let sum := x + y
  let rounded := Real.round (sum * 100) / 100
  in rounded = 105.18 :=
by
  sorry

end add_and_round_l255_255971


namespace max_value_of_permutation_l255_255503

noncomputable def max_abs_sum (n : ℕ) : ℕ :=
  if n % 2 = 0 then 
    let m := n / 2 in 2 * m^2 - 1 
  else 
    let m := n / 2 in 2 * m^2 + 2 * m - 1

theorem max_value_of_permutation (n : ℕ) :
  ∃ s : fin n → fin n, (perm s) → (∑ i in fin (n-1), |s i - s (i+1)|) = max_abs_sum n := sorry

end max_value_of_permutation_l255_255503


namespace minimum_odd_numbers_in_A_P_l255_255367

-- Polynomials and assumptions
def degree (P : Polynomial ℝ) : ℕ := P.natDegree

-- The set A_P is defined as the set of all numbers x for which P(x) gives a certain value
def A_P (P : Polynomial ℝ) : Set ℝ := {x : ℝ | P.eval x = P.eval 8}

-- Define the main theorem statement
theorem minimum_odd_numbers_in_A_P (P : Polynomial ℝ) (hdeg : degree P = 8) (h8 : 8 ∈ A_P P) : 
  ∃ n, n = 1 ∧ ∃ (x : ℝ), x ∈ A_P P ∧ ¬ (x % 2 = 0) := sorry

end minimum_odd_numbers_in_A_P_l255_255367


namespace prove_a_equals_1_l255_255241

theorem prove_a_equals_1 
  {a b c d k m : ℤ}
  (h1 : d * a = b * c)
  (h2 : a + d = 2^k)
  (h3 : b + c = 2^m)
  (h4 : ∃ k m : ℤ, ∃ a b c d : ℤ, h1 ∧ h2 ∧ h3) 
  : a = 1 :=
sorry

end prove_a_equals_1_l255_255241


namespace first_guard_hours_l255_255548

-- Define conditions
def total_hours := 9
def last_guard_hours := 2
def each_middle_guard_hours := 2

-- Define the proof problem
theorem first_guard_hours : 
  (total_hours - last_guard_hours - 2 * each_middle_guard_hours) = 3 :=
by
  -- sorry is used to skip the proof
  sorry

end first_guard_hours_l255_255548


namespace sin_square_pi_over_4_l255_255200

theorem sin_square_pi_over_4 (β : ℝ) (h : Real.sin (2 * β) = 2 / 3) : 
  Real.sin (β + π/4) ^ 2 = 5 / 6 :=
by
  sorry

end sin_square_pi_over_4_l255_255200


namespace min_odd_numbers_in_A_P_l255_255355

theorem min_odd_numbers_in_A_P (P : ℝ → ℝ) 
  (hP : ∃ a8 a7 a6 a5 a4 a3 a2 a1 a0 : ℝ, P = λ x, a8 * x^8 + a7 * x^7 + a6 * x^6 + a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0)
  (h8_in_A_P : ∃ x : ℝ, P x = 8) : 
  (∀ x : ℝ, P x ∈ Set.A_P) → 
  (∃! y : ℝ, odd y ∧ y ∈ Set.A_P) := 
sorry

end min_odd_numbers_in_A_P_l255_255355


namespace number_of_digits_difference_l255_255512

/-- 
  Prove that the number of digits of 3200 in base-2 
  is 3 more than the number of digits of 400 in base-2. 
--/
theorem number_of_digits_difference : 
  let digits_in_base_2 (n : ℕ) : ℕ := (Nat.log2 n) + 1 in
  digits_in_base_2 3200 = digits_in_base_2 400 + 3 := 
by 
  sorry

end number_of_digits_difference_l255_255512


namespace focus_of_parabola_l255_255661

def parabola (x : ℝ) : ℝ := (x - 3) ^ 2

theorem focus_of_parabola :
  ∃ f : ℝ × ℝ, f = (3, 1 / 4) ∧
  ∀ x : ℝ, parabola x = (x - 3)^2 :=
sorry

end focus_of_parabola_l255_255661


namespace parallel_transitive_l255_255910

theorem parallel_transitive {l1 l2 l3 : Line} :
  (l1 ∥ l2) ∧ (l2 ∥ l3) → (l1 ∥ l3) :=
by
  sorry

end parallel_transitive_l255_255910


namespace index_difference_l255_255914

theorem index_difference (n f m : ℕ) (h_n : n = 25) (h_f : f = 8) (h_m : m = 25 - 8) :
  (n - f) / n - (n - m) / n = 9 / 25 :=
by
  -- The proof is to be completed here.
  sorry

end index_difference_l255_255914


namespace collinearity_X_Y_Z_l255_255925

open EuclideanGeometry

theorem collinearity_X_Y_Z 
  (A B C P H L M N X Y Z : Point)
  (hH_orthocenter : is_orthocenter H A B C)
  (hP_any_point : true)
  (hHL_perpendicular_PA : ∃ L, L ∈ Line PA ∧ perp H L P A)
  (hX_on_BC : ∃ X, X ∈ Line BC ∧ X ∈ Line HL)
  (hHM_perpendicular_PB : ∃ M, M ∈ Line PB ∧ perp H M P B)
  (hY_on_CA : ∃ Y, Y ∈ Line CA ∧ Y ∈ Line HM)
  (hHN_perpendicular_PC : ∃ N, N ∈ Line PC ∧ perp H N P C)
  (hZ_on_AB : ∃ Z, Z ∈ Line AB ∧ Z ∈ Line HN) :
  collinear X Y Z := 
sorry

end collinearity_X_Y_Z_l255_255925


namespace projections_equal_l255_255212

-- Given a tetrahedron ABCD
variables {A B C D : Point} -- Points representing vertices of the tetrahedron
-- Centers of two spheres tangent to the edges
variables {O1 O2 : Point} -- Centers of the spheres

-- Condition that spheres are tangent at specified edges
-- Define as Boolean conditions or geometric relationships
def sphere_tangency (P Q : Point) (center : Point) (r : Real) : Prop := dist center P = r ∧ dist center Q = r

-- Spheres are tangent to the specified edges at given points
axiom tangent_sphere1 : sphere_tangency A C O1 R1
axiom tangent_sphere2 : sphere_tangency B D O2 R2

-- Main statement to prove
theorem projections_equal : projection A C O1 O2 = projection B D O1 O2 := sorry

end projections_equal_l255_255212


namespace brown_eyed_brunettes_l255_255169

-- Definitions based on the conditions
variables (total number_girls : ℕ)
variables (green_eyed_redheads brunettes brown_eyed : ℕ)

-- The main proof statement
theorem brown_eyed_brunettes :
  (total = 60) → 
  (green_eyed_redheads = 20) →
  (brunettes = 35) →
  (brown_eyed = 25) →
  (let redheads := total - brunettes in
  let brown_eyed_redheads := redheads - green_eyed_redheads in
  brown_eyed - brown_eyed_redheads = 20) :=
by intros total number_girls green_eyed_redheads brunettes brown_eyed;
   sorry

end brown_eyed_brunettes_l255_255169


namespace mike_fell_short_by_22_l255_255813

def max_marks : ℕ := 780
def passing_percentage : ℝ := 0.30
def mike_score : ℕ := 212
def passing_marks : ℕ := (passing_percentage * max_marks).toNat -- Using toNat to convert from ℝ to ℕ

theorem mike_fell_short_by_22 : passing_marks - mike_score = 22 := by
  sorry

end mike_fell_short_by_22_l255_255813


namespace circle_tangent_to_fixed_circle_l255_255965

noncomputable theory

-- Define the problem setup
variables {A B : Point} (circleA : Circle A) (circleB : Circle B)

-- Definition of tangent
def is_tangent (P : Point) (C : Circle) : Prop :=
  ∃ line : Line, line.tangent_to P C

-- Define the main theorem statement
theorem circle_tangent_to_fixed_circle :
  ∀ (P : Point), is_tangent P circleA →
  ∃ (C D : Point), C ∈ circleB ∧ D ∈ circleB ∧
  Circle B C D remains_tangent_to_fixed (Circle A B) :=
sorry

end circle_tangent_to_fixed_circle_l255_255965


namespace symmetric_line_equation_l255_255461

theorem symmetric_line_equation (x y : ℝ) :
  (x - 2 * y + 1 = 0) → (symmetric_x := 2 - x) → (symmetric_y := y) → (x' := symmetric_x) → 
  (y' := symmetric_y) → (line := x' + 2 * y' - 3 = 0) → 
  line := x + 2 * y - 3 = 0 := 
sorry

end symmetric_line_equation_l255_255461


namespace nina_jewelry_ensemble_price_l255_255408

theorem nina_jewelry_ensemble_price :
  (necklace_price bracelet_price earring_price sales_necklaces sales_bracelets sales_earrings orders_ensemble total_income) :
  (necklace_price = 25) →
  (bracelet_price = 15) →
  (earring_price = 10) →
  (sales_necklaces = 5) →
  (sales_bracelets = 10) →
  (sales_earrings = 20) →
  (orders_ensemble = 2) →
  (total_income = 565) →
  let total_from_regular_sales : ℕ := sales_necklaces * necklace_price + sales_bracelets * bracelet_price + sales_earrings * earring_price,
      remaining_amount : ℕ := total_income - total_from_regular_sales,
      ensemble_price : ℕ := remaining_amount / orders_ensemble
  in
  ensemble_price = 45 :=
begin
  intros,
  unfold total_from_regular_sales remaining_amount ensemble_price,
  sorry
end

end nina_jewelry_ensemble_price_l255_255408


namespace range_of_g_when_a_is_two_a_value_for_minimum_f_is_zero_l255_255338

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 4^x - a * 2^(x + 1) + 1

theorem range_of_g_when_a_is_two :
  ∀ x ∈ (set.Icc 1 2), 1 / (4^x - 2 * 2^(x + 1) + 1) ∈ set.Icc (1 / 7) (1 / 3) :=
sorry

theorem a_value_for_minimum_f_is_zero :
  (∀ x ∈ (set.Icc 1 2), f x (5 / 4) ≥ 0) ∧ (∃ x ∈ (set.Icc 1 2), f x (5 / 4) = 0) :=
sorry

end range_of_g_when_a_is_two_a_value_for_minimum_f_is_zero_l255_255338


namespace lemon_count_l255_255446

theorem lemon_count {total_fruits mangoes pears pawpaws : ℕ} (kiwi lemon : ℕ) :
  total_fruits = 58 ∧ 
  mangoes = 18 ∧ 
  pears = 10 ∧ 
  pawpaws = 12 ∧ 
  (kiwi = lemon) →
  lemon = 9 :=
by 
  sorry

end lemon_count_l255_255446


namespace sum_of_15_consecutive_integers_perfect_square_l255_255872

open Nat

-- statement that defines the conditions and the objective of the problem
theorem sum_of_15_consecutive_integers_perfect_square :
  ∃ n k : ℕ, 15 * (n + 7) = k^2 ∧ 15 * (n + 7) ≥ 225 := 
sorry

end sum_of_15_consecutive_integers_perfect_square_l255_255872


namespace incorrect_proposition_statement_l255_255909

theorem incorrect_proposition_statement (p q : Prop) : (p ∨ q) → ¬ (p ∧ q) := 
sorry

end incorrect_proposition_statement_l255_255909


namespace julia_fourth_rod_count_l255_255779

theorem julia_fourth_rod_count:
  let rods := (1:ℕ, 35:ℕ) → ℕ in
  let a := 4 in
  let b := 8 in
  let c := 17 in
  let valid_d := λ d: ℕ, d > 5 ∧ d < 29 ∧ d ≠ b ∧ d ≠ c in
  (∃ valid_rods: finset ℕ, valid_rods.card = 21 ∧ ∀ d ∈ valid_rods, valid_d d) :=
  sorry

end julia_fourth_rod_count_l255_255779


namespace cannot_transport_stones_in_one_trip_l255_255756

theorem cannot_transport_stones_in_one_trip :
  let weight_first_stone := 370 -- Weight of the first stone in kg
  let weight_increase := 2 -- Weight increment for each subsequent stone in kg
  let num_stones := 50 -- Total number of stones
  let num_trucks := 7 -- Total number of trucks
  let truck_capacity := 3000 -- Capacity of each truck in kg
  let total_weight := (num_stones * weight_first_stone) + (weight_increase * (num_stones * (num_stones - 1) / 2))
  total_weight > (num_trucks * truck_capacity) :=
by
  let weight_first_stone := 370
  let weight_increase := 2
  let num_stones := 50
  let num_trucks := 7
  let truck_capacity := 3000
  let total_weight := (num_stones * weight_first_stone) + (weight_increase * (num_stones * (num_stones - 1) / 2))
  show total_weight > (num_trucks * truck_capacity), from sorry

end cannot_transport_stones_in_one_trip_l255_255756


namespace ones_digit_of_power_35_35_pow_17_17_is_five_l255_255181

theorem ones_digit_of_power_35_35_pow_17_17_is_five :
  (35 ^ (35 * (17 ^ 17))) % 10 = 5 := by
  sorry

end ones_digit_of_power_35_35_pow_17_17_is_five_l255_255181


namespace minimum_red_chips_l255_255536

variable (w b r : ℕ)

-- Define the conditions
def condition1 : Prop := b ≥ 3 * w / 4
def condition2 : Prop := b ≤ r / 4
def condition3 : Prop := 60 ≤ w + b ∧ w + b ≤ 80

-- Prove the minimum number of red chips r is 108
theorem minimum_red_chips (H1 : condition1 w b) (H2 : condition2 b r) (H3 : condition3 w b) : r ≥ 108 := 
sorry

end minimum_red_chips_l255_255536


namespace BC_length_is_100_l255_255748

noncomputable def prove_BC_length : Prop :=
  ∃ (x y : ℕ), 
    let AB := 90
    let AC := 100
    let BC := x + y in
    AB = 90 ∧ AC = 100 ∧
    (BC = AB + y ∨ BC = AC + x) ∧
    (AB^2 * x + AC^2 * y = BC * (x * y + AB^2)) ∧
    BC = 100

theorem BC_length_is_100 : prove_BC_length :=
by {
  sorry
}

end BC_length_is_100_l255_255748


namespace min_odd_numbers_in_A_P_l255_255354

theorem min_odd_numbers_in_A_P (P : ℝ → ℝ) 
  (hP : ∃ a8 a7 a6 a5 a4 a3 a2 a1 a0 : ℝ, P = λ x, a8 * x^8 + a7 * x^7 + a6 * x^6 + a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0)
  (h8_in_A_P : ∃ x : ℝ, P x = 8) : 
  (∀ x : ℝ, P x ∈ Set.A_P) → 
  (∃! y : ℝ, odd y ∧ y ∈ Set.A_P) := 
sorry

end min_odd_numbers_in_A_P_l255_255354


namespace min_odd_in_A_P_l255_255347

-- Define the polynomial P of degree 8
variables (P : Polynomial ℝ)
hypothesis degree_P : P.natDegree = 8

-- Define the set A_P
def A_P (P : Polynomial ℝ) : Set ℝ := { x : ℝ | True }

-- Hypothesis that the set A_P includes the number 8
hypothesis A_P_contains_8 : 8 ∈ A_P P

-- The minimum number of odd numbers in the set A_P
theorem min_odd_in_A_P : ∃ n : ℕ, odd n ∧ n = 1 := sorry

end min_odd_in_A_P_l255_255347


namespace pq_over_mn_squared_l255_255705

-- Define the ellipse and the points
def ellipse (x y : ℝ) := (x^2 / 2) + y^2 = 1

-- Define the left focus of the ellipse
def left_focus := (-1, 0)

-- Define the relation between MN line and the ellipse
def line_MN_through_focus (m n : ℝ × ℝ) (k b : ℝ) :=
  -- Line MN through left focus with slope k: y = k(x + 1)
  m.2 = k * (m.1 + 1) ∧ n.2 = k * (n.1 + 1) ∧
  ellipse m.1 m.2 ∧ ellipse n.1 n.2

-- Define the relation between PQ line and the ellipse
def line_PQ_through_origin (p q : ℝ × ℝ) (k : ℝ) :=
  -- Line PQ through origin with slope k: y = k * x
  p.2 = k * p.1 ∧ q.2 = k * q.1 ∧
  ellipse p.1 p.2 ∧ ellipse q.1 q.2

-- Define the length of a line segment
def length (a b : ℝ × ℝ) := 
  real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

-- Statement to prove
theorem pq_over_mn_squared (p q m n : ℝ × ℝ) (k : ℝ) (hMN : line_MN_through_focus m n k b) (hPQ : line_PQ_through_origin p q k) :
  (length p q)^2 / length m n = 2 * real.sqrt 2 := sorry

end pq_over_mn_squared_l255_255705


namespace smallest_positive_period_max_min_values_l255_255245

open Real

noncomputable def f (x : ℝ) : ℝ :=
  sin x ^ 2 - sin (x - π / 6) ^ 2

theorem smallest_positive_period : smallest_positive_period f = π := sorry

theorem max_min_values :
  (∀ x ∈ Icc (-π/3) (π/4), f x ≤ sqrt 3 / 4) ∧ 
  (∀ x ∈ Icc (-π/3) (π/4), f x ≥ -1 / 2)  := sorry

end smallest_positive_period_max_min_values_l255_255245


namespace range_of_x_l255_255318

theorem range_of_x (x : ℝ) : 
  sqrt ((x - 1) / (x - 2)) ≥ 0 → (x > 2 ∨ x ≤ 1) :=
by
  sorry

end range_of_x_l255_255318


namespace fiona_prob_reaches_12_l255_255484

/-- Lily pads are numbered from 0 to 15 -/
def is_valid_pad (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 15

/-- Predators are on lily pads 4 and 7 -/
def predator (n : ℕ) : Prop := n = 4 ∨ n = 7

/-- Fiona the frog's probability to hop to the next pad -/
def hop : ℚ := 1 / 2

/-- Fiona the frog's probability to jump 2 pads -/
def jump_two : ℚ := 1 / 2

/-- Probability that Fiona reaches pad 12 without landing on pads 4 or 7 is 1/32 -/
theorem fiona_prob_reaches_12 :
  ∀ p : ℕ, 
    (is_valid_pad p ∧ ¬ predator p ∧ (p = 12) ∧ 
    ((∀ k : ℕ, is_valid_pad k → ¬ predator k → k ≤ 3 → (hop ^ k) = 1 / 2) ∧
    hop * hop = 1 / 4 ∧ hop * jump_two = 1 / 8 ∧
    (jump_two * (hop * hop + jump_two)) = 1 / 4 → hop * 1 / 4 = 1 / 32)) := 
by intros; sorry

end fiona_prob_reaches_12_l255_255484


namespace axis_of_symmetry_of_translated_sin_function_l255_255033

theorem axis_of_symmetry_of_translated_sin_function:
  ∀ (x : ℝ), (g : ℝ → ℝ) (h1 : g = λ x, Real.sin (2 * x - π / 3)),
  ∃ (k : ℤ), x = k * (π / 2) + 5 * π / 12 → g x = Real.sin ((2 * x - π / 3)) :=
by
  sorry

end axis_of_symmetry_of_translated_sin_function_l255_255033


namespace conference_lecture_schedule_l255_255553

theorem conference_lecture_schedule :
  let total_lectures := 8
  let total_permutations := (Nat.fact total_lectures : ℕ)
  let valid_schedules := total_permutations / (2 * 2)
  valid_schedules = 10080 := 
by
  let total_lectures := 8
  let total_permutations := (Nat.fact total_lectures : ℕ)
  let valid_schedules := total_permutations / (2 * 2)
  show valid_schedules = 10080 from sorry

end conference_lecture_schedule_l255_255553


namespace min_odd_in_A_P_l255_255346

-- Define the polynomial P of degree 8
variables (P : Polynomial ℝ)
hypothesis degree_P : P.natDegree = 8

-- Define the set A_P
def A_P (P : Polynomial ℝ) : Set ℝ := { x : ℝ | True }

-- Hypothesis that the set A_P includes the number 8
hypothesis A_P_contains_8 : 8 ∈ A_P P

-- The minimum number of odd numbers in the set A_P
theorem min_odd_in_A_P : ∃ n : ℕ, odd n ∧ n = 1 := sorry

end min_odd_in_A_P_l255_255346


namespace repeating_decimal_to_fraction_l255_255633

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + 0.066666... ) : x = 11 / 30 :=
sorry

end repeating_decimal_to_fraction_l255_255633


namespace smallest_m_for_root_of_unity_l255_255797

def complex_in_T (z : ℂ) : Prop :=
  let x := z.re
  let y := z.im
  (real.sqrt 3 / 2) ≤ x ∧ x ≤ (2 / real.sqrt 3)

theorem smallest_m_for_root_of_unity :
  ∃ m : ℕ, ∀ n : ℕ, n ≥ m → (∃ z : ℂ, complex_in_T z ∧ z ^ n = 1) :=
begin
  use 18,
  sorry
end

end smallest_m_for_root_of_unity_l255_255797


namespace sqrt_diff_nat_l255_255215

open Nat

theorem sqrt_diff_nat (a b : ℕ) (h : 2015 * a^2 + a = 2016 * b^2 + b) : ∃ k : ℕ, a - b = k^2 := 
by
  sorry

end sqrt_diff_nat_l255_255215


namespace minimum_odd_numbers_in_A_P_l255_255368

-- Polynomials and assumptions
def degree (P : Polynomial ℝ) : ℕ := P.natDegree

-- The set A_P is defined as the set of all numbers x for which P(x) gives a certain value
def A_P (P : Polynomial ℝ) : Set ℝ := {x : ℝ | P.eval x = P.eval 8}

-- Define the main theorem statement
theorem minimum_odd_numbers_in_A_P (P : Polynomial ℝ) (hdeg : degree P = 8) (h8 : 8 ∈ A_P P) : 
  ∃ n, n = 1 ∧ ∃ (x : ℝ), x ∈ A_P P ∧ ¬ (x % 2 = 0) := sorry

end minimum_odd_numbers_in_A_P_l255_255368


namespace smallest_possible_sector_angle_l255_255405

theorem smallest_possible_sector_angle
  (n : ℕ) (angles : Fin n → ℕ)
  (a₁ d : ℕ)
  (h1 : n = 12)
  (h2 : ∀ i : Fin n, angles i = a₁ + i * d)
  (h3 : (angles 0 + angles 1 + angles 2 + angles 3 + angles 4 + angles 5 + angles 6 + angles 7 + angles 8 + angles 9 + angles 10 + angles 11) = 360) :
  ∃ a₁, a₁ = 8 := 
sorry

end smallest_possible_sector_angle_l255_255405


namespace find_x_l255_255851

noncomputable def sequence_avg_eq_50x (x : ℕ) : Prop :=
  let seq := List.range' 2 98 |> List.map (λ n, 2 * n)
  (seq.sum + x) / (seq.length + 1) = 50 * x

theorem find_x : ∃ x : ℕ, sequence_avg_eq_50x x ∧ x = 2 :=
by
  -- We need to specifically state the existence and equality
  existsi 2
  sorry

end find_x_l255_255851


namespace count_ranges_l255_255547

theorem count_ranges : 
  let count := ∑ b in finset.range 2 10,
                ∑ d in finset.range 1 10,
                  let a_valid := b - 1,
                  let e_valid := d - 1,
                  let c_valid := min b d - 1,
                  in a_valid * e_valid * c_valid
  count = 1260 :=
by {
  unfold count,
  sorry
}

end count_ranges_l255_255547


namespace P_x_x_cannot_have_odd_degree_l255_255866

noncomputable def P (x y : ℕ) : ℕ := sorry

-- Condition given in the problem
axiom condition (n : ℕ) : (∀ y, P n y = 0 ∨ ∃ k ≤ n, k = n) ∧ (∀ x, P x n = 0 ∨ ∃ k ≤ n, k = n)

-- The statement to be proved
theorem P_x_x_cannot_have_odd_degree :
  ∀ P : ℕ → ℕ → ℕ, 
  (∀ n : ℕ, (∀ y : ℕ, P n y = 0 ∨ ∃ k, k ≤ n ∧ P n y = k) ∧ (∀ x : ℕ, P x n = 0 ∨ ∃ k, k ≤ n ∧ P x n = k)) →
  ¬(∃ m, (deg (P x x) = 2 * m + 1)) :=
sorry

end P_x_x_cannot_have_odd_degree_l255_255866


namespace weight_of_larger_square_tile_l255_255960

theorem weight_of_larger_square_tile :
  ∀ (side_length1 side_length2 weight1 : ℝ),
  side_length1 = 4 →
  weight1 = 10 →
  side_length2 = 6 →
  (weight1 * side_length2^2) / side_length1^2 = 22.5 :=
by
  intros
  calc
    (10 * 6^2) / 4^2 = 22.5 : sorry

end weight_of_larger_square_tile_l255_255960


namespace AT_parallel_EF_l255_255771

open Triangle

variable {A B C D E F M T : Point}

-- Assume the following conditions
axiom incircle_touches_BC_at_D (h₁ : touches_incircle_triangle A B C D) : touches_incircle_triangle A B C D
axiom incircle_touches_AC_at_E (h₂ : touches_incircle_triangle A B C E) : touches_incircle_triangle A B C E
axiom incircle_touches_AB_at_F (h₃ : touches_incircle_triangle A B C F) : touches_incircle_triangle A B C F
axiom midpoint_of_EF_is_M (h₄ : midpoint E F M) : midpoint E F M
axiom intersection_of_DE_and_BM_is_T (h₅ : intersects DE BM T) : intersects DE BM T

theorem AT_parallel_EF :
  parallel (Line.mk A T) (Line.mk E F) :=
sorry

end AT_parallel_EF_l255_255771


namespace eating_time_l255_255814

-- Define the eating rates of Mr. Fat, Mr. Thin, and Mr. Medium
def mrFat_rate := 1 / 15
def mrThin_rate := 1 / 35
def mrMedium_rate := 1 / 25

-- Define the combined eating rate
def combined_rate := mrFat_rate + mrThin_rate + mrMedium_rate

-- Define the amount of cereal to be eaten
def amount_cereal := 5

-- Prove that the time taken to eat the cereal is 2625 / 71 minutes
theorem eating_time : amount_cereal / combined_rate = 2625 / 71 :=
by 
  -- Here should be the proof, but it is skipped
  sorry

end eating_time_l255_255814


namespace meadow_trees_count_l255_255434

theorem meadow_trees_count (n : ℕ) (f s m : ℕ → ℕ) :
  (f 20 = s 7) ∧ (f 7 = s 94) ∧ (s 7 > f 20) → 
  n = 100 :=
by
  sorry

end meadow_trees_count_l255_255434


namespace greatest_power_of_2_divides_expr_l255_255501

theorem greatest_power_of_2_divides_expr :
  maxPowerOf2 (9 ^ 456 - 3 ^ 684) = 459 := 
sorry

end greatest_power_of_2_divides_expr_l255_255501


namespace minimum_value_y_range_of_a_l255_255710

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*a*x - 1 + a

theorem minimum_value_y (x : ℝ) 
  (hx_pos : x > 0) : (f x 2 / x) = -2 :=
by sorry

theorem range_of_a : 
  ∀ a : ℝ, ∀ x ∈ (Set.Icc 0 2), (f x a) ≤ a ↔ a ≥ 3 / 4 :=
by sorry

end minimum_value_y_range_of_a_l255_255710


namespace solve_integral_equation_l255_255010

noncomputable def kernel (x t : ℝ) : ℝ :=
  (cos x) ^ 2 * cos (2 * t) + (cos t) ^ 3 * cos (3 * x)

-- Define the integral equation
def integral_equation (varphi : ℝ → ℝ) (lambda : ℝ) : Prop :=
  ∀ x : ℝ, varphi x = lambda * ∫ t in 0..π, kernel x t * varphi t

-- Specify the general solution
def general_solution (varphi : ℝ → ℝ) (lambda : ℝ) (C : ℝ) : Prop :=
  (lambda = 4 / π → varphi = (λ x, C * cos x ^ 2)) ∧
  (lambda = 8 / π → varphi = (λ x, C * cos (3 * x))) ∧
  (lambda ≠ 4 / π ∧ lambda ≠ 8 / π → varphi = (λ _, 0))

-- The main theorem combining the integral equation and its general solution
theorem solve_integral_equation (varphi : ℝ → ℝ) (lambda : ℝ) (C : ℝ) :
  integral_equation varphi lambda ↔ general_solution varphi lambda C :=
sorry

end solve_integral_equation_l255_255010


namespace correct_propositions_l255_255623

theorem correct_propositions : 
  (∀ x : ℝ, x^2 - x + (1 / 4) ≥ 0) ∧
  (∀ p q : Prop, (p ∧ q → p) ∧ (p ∧ q → q)) ∧ -- representing squares as rectangles
  (¬ ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) ∧
  (∃ x : ℝ, x^3 + 1 = 0) :=
by
  split
  -- proof for ①
  sorry
  split
  -- proof for ②
  sorry
  split
  -- proof for ③
  sorry
  -- proof for ④
  sorry

end correct_propositions_l255_255623


namespace fifty_presses_large_number_l255_255442

noncomputable def f (x : ℤ) : ℤ := x^2 - 2

def sequence (n : ℕ) : ℤ :=
  Nat.iterate f n 3

theorem fifty_presses_large_number :
  sequence 50 > 2207 := sorry

end fifty_presses_large_number_l255_255442


namespace odd_numbers_with_max_divisors_l255_255822

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def num_divisors (n : ℕ) : ℕ := (divisors n).length

def odd_numbers (lower upper : ℕ) : List ℕ := List.filter is_odd (List.range' lower (upper - lower + 1))

theorem odd_numbers_with_max_divisors :
  ∀ n ∈ odd_numbers 1 25, num_divisors n ≤ 4 
  ∧ ((num_divisors n = 4) → (n = 15 ∨ n = 21)) := 
by
  sorry

end odd_numbers_with_max_divisors_l255_255822


namespace domino_arrangement_possible_l255_255537

def set_contains {α : Type*} (s : set α) (a : α) : Prop := a ∈ s

noncomputable def domino_set : Type :=
  {s : set (ℕ × ℕ) // ∀ a b, a ≤ b → (a, b) ∈ s → a ≤ 9 ∧ b ≤ 9}

def is_single_line_arrangement (dominoes : set (ℕ × ℕ)) : Prop :=
  ∃ (path : list (ℕ × ℕ)), ∀ (p : (ℕ × ℕ)), p ∈ dominoes ↔ p ∈ path.to_finset

def remaining_dominoes : set (ℕ × ℕ) :=
  (domino_set \ {(7, 6), (5, 4), (3, 2), (1, 0)} : set (ℕ × ℕ))

theorem domino_arrangement_possible :
  is_single_line_arrangement remaining_dominoes :=
sorry

end domino_arrangement_possible_l255_255537


namespace solve_equation_l255_255011

theorem solve_equation (x : ℝ) : (⌊Real.sin x⌋:ℝ)^2 = Real.cos x ^ 2 - 1 ↔ ∃ n : ℤ, x = n * Real.pi := by
  sorry

end solve_equation_l255_255011


namespace max_tickets_sold_l255_255123

theorem max_tickets_sold (bus_capacity : ℕ) (num_stations : ℕ) (max_capacity : bus_capacity = 25) 
  (total_stations : num_stations = 14) : 
  ∃ (tickets : ℕ), tickets = 67 :=
by 
  sorry

end max_tickets_sold_l255_255123


namespace digit_certainty_proof_l255_255176

def a := 945.673
def delta_a := 0.03

-- Digits and their corresponding positional values
def tenths_place := 6
def hundredths_place := 7
def thousandths_place := 3

def positional_value_tenths := 0.1
def positional_value_hundredths := 0.01
def positional_value_thousandths := 0.001

theorem digit_certainty_proof :
  (positional_value_tenths > delta_a ∧
  tenths_place = 6 ∧
  positional_value_hundredths < delta_a ∧
  hundredths_place = 7 ∧
  positional_value_thousandths < delta_a ∧
  thousandths_place = 3) →
  (true) := 
by {
  intro h,
  sorry
}

end digit_certainty_proof_l255_255176


namespace min_odd_in_A_P_l255_255349

-- Define the polynomial P of degree 8
variables (P : Polynomial ℝ)
hypothesis degree_P : P.natDegree = 8

-- Define the set A_P
def A_P (P : Polynomial ℝ) : Set ℝ := { x : ℝ | True }

-- Hypothesis that the set A_P includes the number 8
hypothesis A_P_contains_8 : 8 ∈ A_P P

-- The minimum number of odd numbers in the set A_P
theorem min_odd_in_A_P : ∃ n : ℕ, odd n ∧ n = 1 := sorry

end min_odd_in_A_P_l255_255349


namespace stability_not_utilized_in_designA_l255_255906

def designA := "A retractable door made up of quadrilaterals" -- Description of Design A
def designB := "The triangular frame of a bicycle" -- Description of Design B
def designC := "A rectangular window frame secured with diagonal nails" -- Description of Design C
def designD := "The tripod of a camera" -- Description of Design D

theorem stability_not_utilized_in_designA:
  ∀ (D1 D2 D3 D4 : String), (D1 = designA) ∧ (D2 = designB) ∧ (D3 = designC) ∧ (D4 = designD) →
  (¬ utilizes_stability_of_triangle D1) ∧ utilizes_stability_of_triangle D2 ∧ 
  utilizes_stability_of_triangle D3 ∧ utilizes_stability_of_triangle D4 :=
sorry

end stability_not_utilized_in_designA_l255_255906


namespace measure_angle_BAG_l255_255422

theorem measure_angle_BAG
    (A B C D E F G H I : Type) [is_point A] [is_point B] [is_point C] [is_point D] [is_point E]
    [is_point F] [is_point G] [is_point H] [is_point I]
    (circle1 : Circle) (circle2 : Circle)
    (center_circle1 : circle1.center = O1) (center_circle2 : circle2.center = C)
    (all_points_on_circle1 : ∀ (P : Type), P ∈ {A, B, C, D, E} → is_point P)
    (all_points_on_circle2 : ∀ (P : Type), P ∈ {E, F, G, H, I, A} → is_point P)
    (equally_spaced_circle1 : EquallySpaced {A, B, C, D, E} circle1)
    (equally_spaced_circle2 : EquallySpaced {E, F, G, H, I, A} circle2)
    (angle_relation : ∀ a b d h g: Type, angle a b d - angle a h g = 12) :
    measure (angle A B G) = 58 :=
by
  sorry

end measure_angle_BAG_l255_255422


namespace repeating_decimal_to_fraction_l255_255640

theorem repeating_decimal_to_fraction : (0.36 : ℝ) = (11 / 30 : ℝ) :=
sorry

end repeating_decimal_to_fraction_l255_255640


namespace min_odd_in_A_P_l255_255383

-- Define the polynomial P of degree 8
variable {R : Type*} [Ring R] (P : Polynomial R)
variable (hP : degree P = 8)

-- Define the set A_P for some value c in R
def A_P (c : R) : Set R := { x | P.eval x = c }

-- Given that the number 8 is included in the set A_P
variable (c : R) (h8 : (8 : R) ∈ A_P c)

-- Prove that the minimum number of odd numbers in the set A_P is 1
theorem min_odd_in_A_P : ∃ x ∈ A_P c, (x % 2 = 1) := sorry

end min_odd_in_A_P_l255_255383


namespace circle_radii_ratio_l255_255085

theorem circle_radii_ratio (ℓ₁ ℓ₂ : Line) (ω₁ ω₂ : Circle) (O₁ O₂ : Point) (A B C D E : Point)
  (H_parallel : ℓ₁ ∥ ℓ₂)
  (H_tangent₁ : IsTangent ℓ₁ ω₁ A)
  (H_tangent₂ : IsTangent ℓ₂ ω₁ B)
  (H_tangent₃ : IsTangent ℓ₁ ω₂ D)
  (H_intersect₁ : IsIntersecting ω₂ ℓ₂ B)
  (H_intersect₂ : IsIntersecting ω₂ ℓ₂ E)
  (H_intersect₃ : IsIntersecting ω₂ ω₁ C)
  (H_between : Between O₂ ℓ₁ ℓ₂)
  (H_area_ratio : AreaRatio (BO₁CO₂) (O₂BE) = 2) :
  radius(ω₂) / radius(ω₁) = 1 / 2 :=
sorry

end circle_radii_ratio_l255_255085


namespace evaluate_polynomial_at_2_l255_255607

def polynomial (x : ℕ) : ℕ := 3 * x^4 + x^3 + 2 * x^2 + x + 4

def horner_method (x : ℕ) : ℕ :=
  let v_0 := x
  let v_1 := 3 * v_0 + 1
  let v_2 := v_1 * v_0 + 2
  v_2

theorem evaluate_polynomial_at_2 :
  horner_method 2 = 16 :=
by
  sorry

end evaluate_polynomial_at_2_l255_255607


namespace number_of_valid_cases_l255_255049

-- Definition of conditions
def hundreds_digit_is_one (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 200

def produces_four_digit_product (n m : ℕ) : Prop :=
  1000 ≤ n * m ∧ n * m < 10000

def second_row_three_digits_valid (n m : ℕ) : Prop :=
  100 ≤ (n * m / 10) ∧ (n * m / 10) < 1000

-- Problem statement
theorem number_of_valid_cases : ∃ n m, hundreds_digit_is_one n ∧ 
                                       produces_four_digit_product n m ∧ 
                                       second_row_three_digits_valid n m ∧
                                       m ≠ 0 ∧ ∀ m', (hundreds_digit_is_one n ∧
                                                     produces_four_digit_product n m' ∧ 
                                                     second_row_three_digits_valid n m') → m' ∈ {m} := 
begin
  sorry
end

end number_of_valid_cases_l255_255049


namespace problem_statement_l255_255800

variables (a b c : ℝ)
variables (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
variable (hsum : a + b + c = 1)

def s : ℝ := a * b + b * c + c * a

theorem problem_statement :
  (ab + bc + ca) / (a^2 + b^2 + c^2) = s / (1 - 2 * s) :=
by
  sorry

end problem_statement_l255_255800


namespace total_carpet_area_correct_l255_255327

-- Define dimensions of the rooms
def room1_width : ℝ := 12
def room1_length : ℝ := 15
def room2_width : ℝ := 7
def room2_length : ℝ := 9
def room3_width : ℝ := 10
def room3_length : ℝ := 11

-- Define the areas of the rooms
def room1_area : ℝ := room1_width * room1_length
def room2_area : ℝ := room2_width * room2_length
def room3_area : ℝ := room3_width * room3_length

-- Total carpet area
def total_carpet_area : ℝ := room1_area + room2_area + room3_area

-- The theorem to prove
theorem total_carpet_area_correct :
  total_carpet_area = 353 :=
sorry

end total_carpet_area_correct_l255_255327


namespace min_odd_in_A_P_l255_255360

-- Define the polynomial P of degree 8
variable (P : ℝ → ℝ)
-- Assume P is a polynomial of degree 8
axiom degree_P : degree P = 8

-- Define the set A_P as the set of all x for which P(x) gives a certain value
def A_P (c : ℝ) : Set ℝ := { x | P x = c }

-- Given condition
axiom include_8 : 8 ∈ A_P (P 8)

-- Define if a number is odd
def is_odd (n : ℝ) : Prop := ∃k : ℤ, n = 2 * k + 1

-- The minimum number of odd numbers that are in the set A_P,
-- given that the number 8 is included in A_P, is 1

theorem min_odd_in_A_P : ∀ P, (degree P = 8) → 8 ∈ A_P (P 8) → ∃ x ∈ A_P (P 8), is_odd x := 
by
  sorry -- proof goes here

end min_odd_in_A_P_l255_255360


namespace hexagon_area_ratio_l255_255333

theorem hexagon_area_ratio {ABCDEF : Hexagon}
  (h1 : ABCDEF.is_regular)
  (h2 : ∀ P Q R S T U, 
         P.is_midpoint ABCDEF.A ABCDEF.B ∧ 
         Q.is_midpoint ABCDEF.B ABCDEF.C ∧ 
         R.is_midpoint ABCDEF.C ABCDEF.D ∧ 
         S.is_midpoint ABCDEF.D ABCDEF.E ∧ 
         T.is_midpoint ABCDEF.E ABCDEF.F ∧ 
         U.is_midpoint ABCDEF.F ABCDEF.A)
  : ∃ VWXYZA' : Hexagon,
      VWXYZA'.is_intersection (line_through ABCDEF.A P) (opposite_side ABCDEF.A) ∧
      VWXYZA'.is_regular ∧
      (area VWXYZA').ratio (area ABCDEF) = (1 : ℚ) / 4 :=
sorry

end hexagon_area_ratio_l255_255333


namespace trick_deck_cost_l255_255888

theorem trick_deck_cost (x : ℝ) : (3 * x + 5 * x = 64) → x = 8 :=
by
  assume h : 3 * x + 5 * x = 64
  sorry

end trick_deck_cost_l255_255888


namespace ratio_games_lost_to_won_l255_255936

-- Define the necessary parameters
def total_games : ℕ := 44
def games_won : ℕ := 16
def games_lost : ℕ := total_games - games_won

-- Lean statement for the proof
theorem ratio_games_lost_to_won : 
  let r := Nat.gcd games_lost games_won,
  let simplified_lost := games_lost / r,
  let simplified_won := games_won / r in
  (simplified_lost, simplified_won) = (7, 4) :=
by
  -- Now, fill in the proof here
  sorry

end ratio_games_lost_to_won_l255_255936


namespace solve_trig_eq_l255_255043

-- Define the equation
def equation (x : ℝ) : Prop := 3 * Real.sin x = 1 + Real.cos (2 * x)

-- Define the solution set
def solution_set (x : ℝ) : Prop := ∃ k : ℤ, x = k * Real.pi + (-1)^k * (Real.pi / 6)

-- The proof problem statement
theorem solve_trig_eq {x : ℝ} : equation x ↔ solution_set x := sorry

end solve_trig_eq_l255_255043


namespace angle_in_third_quadrant_l255_255694

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α * Real.cos α > 0) (h2 : Real.sin α * Real.tan α < 0) : 
  (π < α ∧ α < 3 * π / 2) :=
by
  sorry

end angle_in_third_quadrant_l255_255694


namespace male_contestants_count_l255_255563

-- Define the total number of contestants.
def total_contestants : ℕ := 18

-- A third of the contestants are female.
def female_contestants : ℕ := total_contestants / 3

-- The rest are male.
def male_contestants : ℕ := total_contestants - female_contestants

-- The theorem to prove
theorem male_contestants_count : male_contestants = 12 := by
  -- Proof goes here.
  sorry

end male_contestants_count_l255_255563


namespace distinct_fractions_exists_l255_255825

theorem distinct_fractions_exists : 
  ∃ (fractions : Finset (ℚ → ℚ)), 
  (∀ (f : ℚ → ℚ), f ∈ fractions ↔ 
    ∃ (a b c d : ℚ), 
      (Set.mem (perm (2, 1, 0, 0)) (a, b, c, d) ∧ 
      (c ≠ 0 ∨ d ≠ 0) ∧ 
      f = λ x, (a * x + b) / (c * x + d))) ∧
  fractions.card = 7 :=
begin
  sorry
end

end distinct_fractions_exists_l255_255825


namespace max_and_min_sum_l255_255837

theorem max_and_min_sum (x y z w : ℝ) (h : 3 * (x + y + z + w) = x^2 + y^2 + z^2 + w^2) :
  let C := x * y + x * z + y * z + x * w + y * w + z * w in
  let N := 54 in -- Maximum value
  let n := -4.5 in -- Minimum value
  N + 10 * n = 9 :=
sorry

end max_and_min_sum_l255_255837


namespace find_vasya_floor_l255_255508

variable steps_to_3rd_floor : ℕ
variable vasya_steps : ℕ

-- Given conditions
def steps_per_floor(steps_to_3rd_floor : ℕ) : ℕ := steps_to_3rd_floor / 2
def vasya_floors(steps_per_floor : ℕ) (vasya_steps : ℕ) : ℕ := vasya_steps / steps_per_floor

-- The proof problem
theorem find_vasya_floor (h1 : steps_to_3rd_floor = 36) (h2 : vasya_steps = 72) :
  vasya_floors (steps_per_floor steps_to_3rd_floor) vasya_steps + 1 = 5 :=
by
  sorry

end find_vasya_floor_l255_255508


namespace curves_intersection_four_points_l255_255514

theorem curves_intersection_four_points (a : ℝ) (h : a < 0) :
  ∃ x y : ℝ, (x^2 + y^2 = a^2 ∧ y = x^2 + a) ∧
  ((x = 0 ∧ y = a) ∨ (∃ k : ℝ, k^2 = -1 - 2a ∧ x = k ∧ y = k^2 + a) ∨ (∃ k : ℝ, k^2 = -1 - 2a ∧ x = -k ∧ y = k^2 + a)) :=
  sorry

end curves_intersection_four_points_l255_255514


namespace tom_age_ratio_l255_255889

noncomputable def Tom_age (T N : ℕ) : Prop := 
  (T - N = 3 * (T - 4 * N)) ∧ (T / N = 11 / 2)

theorem tom_age_ratio (T N : ℕ) (h : Tom_age T N) : T / N = 11 / 2 :=
begin
  sorry
end

end tom_age_ratio_l255_255889


namespace caitlin_bracelets_l255_255986

-- Define the conditions
def twice_as_many_small_beads (x y : Nat) : Prop :=
  y = 2 * x

def total_large_small_beads (total large small : Nat) : Prop :=
  total = large + small ∧ large = small

def bracelet_beads (large_beads_per_bracelet small_beads_per_bracelet large_per_bracelet : Nat) : Prop :=
  small_beads_per_bracelet = 2 * large_per_bracelet

def total_bracelets (total_large_beads large_per_bracelet bracelets : Nat) : Prop :=
  bracelets = total_large_beads / large_per_bracelet

-- The theorem to be proved
theorem caitlin_bracelets (total_beads large_per_bracelet small_per_bracelet : Nat) (bracelets : Nat) :
    total_beads = 528 ∧
    large_per_bracelet = 12 ∧
    twice_as_many_small_beads large_per_bracelet small_per_bracelet ∧
    total_large_small_beads total_beads 264 264 ∧
    bracelet_beads large_per_bracelet small_per_bracelet 12 ∧
    total_bracelets 264 12 bracelets
  → bracelets = 22 := by
  sorry

end caitlin_bracelets_l255_255986


namespace gcd_of_repeated_three_digit_integers_l255_255122

theorem gcd_of_repeated_three_digit_integers : 
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) → (∃ n, n = 1001001 * (m + 1) ∧ gcd (1001001 * m) n = 1001001) :=
by
  intros m hm
  use 1001001 * (m + 1)
  split
  . refl
  sorry

end gcd_of_repeated_three_digit_integers_l255_255122


namespace total_lawns_to_mow_l255_255569

/-- Adam earned 9 dollars for each lawn he mowed. He had a certain number of lawns to mow, 
    but forgot to mow 8 of them. He actually earned 36 dollars.
    How many lawns did he have to mow in total? -/

theorem total_lawns_to_mow (dollars_per_lawn : ℕ) (forgotten_lawns : ℕ) (total_earned : ℕ) 
    (H1 : dollars_per_lawn = 9) (H2 : forgotten_lawns = 8) (H3 : total_earned = 36) :
    let lawns_mowed := total_earned / dollars_per_lawn in
    let total_lawns := lawns_mowed + forgotten_lawns in
    total_lawns = 12 :=
by
  sorry

end total_lawns_to_mow_l255_255569


namespace fridge_cost_more_than_computer_l255_255816

theorem fridge_cost_more_than_computer :
  ∀ (F C T : ℝ), 
  (1600 = T + F + C) ∧ 
  (T = 600) ∧ 
  (C = 250) → 
  (F = 750) → 
  (F - C = 500) :=
by
  intros F C T h₁ h₂ h₃ h₄
  sorry

end fridge_cost_more_than_computer_l255_255816


namespace how_much_y_invested_l255_255088

theorem how_much_y_invested (x_investment : ℝ) (y_investment : ℝ) (total_profit : ℝ) (x_profit_share : ℝ) :
  x_investment = 5000 ∧ total_profit = 1600 ∧ x_profit_share = 400 →
    y_investment = 15000 :=
by
  -- Given the conditions
  intros h,
  cases h with hx_investment h_rest,
  cases h_rest with htotal_profit hx_profit_share,
  sorry

end how_much_y_invested_l255_255088


namespace last_digits_nn_periodic_l255_255001

theorem last_digits_nn_periodic (n : ℕ) : 
  ∃ p > 0, ∀ k, (n + k * p)^(n + k * p) % 10 = n^n % 10 := 
sorry

end last_digits_nn_periodic_l255_255001


namespace Vasya_lives_on_the_5th_floor_l255_255511

theorem Vasya_lives_on_the_5th_floor :
  ∀ (steps_Petya: ℕ) (floors_Petya: ℕ) (steps_Vasya: ℕ),
    steps_Petya = 36 ∧ floors_Petya = 2 ∧ steps_Vasya = 72 →
    ((steps_Vasya / (steps_Petya / floors_Petya)) + 1 = 5) :=
by
  intros steps_Petya floors_Petya steps_Vasya h
  cases h with h1 h2
  cases h2 with h3 h4
  have steps_per_floor : ℕ := steps_Petya / floors_Petya
  have floors_Vasya : ℕ := steps_Vasya / steps_per_floor
  have Vasya_floor : ℕ := floors_Vasya + 1
  sorry

end Vasya_lives_on_the_5th_floor_l255_255511


namespace Monica_max_correct_answers_l255_255753

theorem Monica_max_correct_answers :
  ∃ a b c : ℤ, 
    (a + b + c = 60) ∧
    (5 * a - 2 * c = 150) ∧
    (0 ≤ b) ∧ 
    a ≤ 38 :=
begin
  sorry
end

end Monica_max_correct_answers_l255_255753


namespace variance_of_remaining_scores_l255_255757

def scores : List ℕ := [91, 89, 91, 96, 94, 95, 94]

def remaining_scores : List ℕ := [91, 91, 94, 95, 94]

def mean (l : List ℕ) : ℚ := l.sum / l.length

def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

theorem variance_of_remaining_scores :
  variance remaining_scores = 2.8 := by
  sorry

end variance_of_remaining_scores_l255_255757


namespace scott_earnings_l255_255432

theorem scott_earnings
  (price_smoothie : ℝ)
  (price_cake : ℝ)
  (cups_sold : ℝ)
  (cakes_sold : ℝ)
  (earnings_smoothies : ℝ := cups_sold * price_smoothie)
  (earnings_cakes : ℝ := cakes_sold * price_cake) :
  price_smoothie = 3 → price_cake = 2 → cups_sold = 40 → cakes_sold = 18 → 
  (earnings_smoothies + earnings_cakes) = 156 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end scott_earnings_l255_255432


namespace connect_50_cities_with_49_routes_l255_255267

noncomputable def minimum_connections (n : ℕ) : ℕ :=
if h : n > 1 then n - 1 else 0

theorem connect_50_cities_with_49_routes :
  ∃ C : Finset ℕ, ∀ A B : Finset ℕ, (A ≠ C ∧ B ≠ C) → 
  (C.card = 1 ∧ A.card = 1 ∧ B.card = 1 ∧
  minimum_connections 50 = 49 ∧
  ∀ A B : ℕ, (A ≠ B ∧ A ≠ C ∧ B ≠ C) → (∀ route : Finset (Finset ℕ), route ∈ (A :: B :: []).subsets ∧ route.card = 1)

end connect_50_cities_with_49_routes_l255_255267


namespace number_of_valid_subsets_l255_255929

theorem number_of_valid_subsets (n : ℕ) :
  let total      := 16^n
  let invalid1   := 3 * 12^n
  let invalid2   := 2 * 10^n
  let invalidAll := 8^n
  let valid      := total - invalid1 + invalid2 + 9^n - invalidAll
  valid = 16^n - 3 * 12^n + 2 * 10^n + 9^n - 8^n :=
by {
  -- Proof steps would go here
  sorry
}

end number_of_valid_subsets_l255_255929


namespace f_at_1_eq_e_add_1_l255_255251

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp x + 2 * (derivative f 0) * x + 3

theorem f_at_1_eq_e_add_1 (x : ℝ) (h : f 0 = Real.exp 0 - 2 * x + 3): f 1 = Real.exp 1 + 1 :=
by
  sorry

end f_at_1_eq_e_add_1_l255_255251


namespace geometric_sequence_sum_l255_255208

variable {a : ℕ → ℝ} -- Sequence terms
variable {S : ℕ → ℝ} -- Sum of the first n terms

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n, S n = a 0 * (1 - (a n)) / (1 - a 1)
def is_arithmetic_sequence (x y z : ℝ) := 2 * y = x + z
def term_1_equals_1 (a : ℕ → ℝ) := a 0 = 1

-- Question: Prove that given the conditions, S_5 = 31
theorem geometric_sequence_sum (q : ℝ) (h_geom : is_geometric_sequence a q) 
  (h_sum : sum_of_first_n_terms a S) (h_arith : is_arithmetic_sequence (4 * a 0) (2 * a 1) (a 2)) 
  (h_a1 : term_1_equals_1 a) : S 5 = 31 :=
sorry

end geometric_sequence_sum_l255_255208


namespace smallest_sum_of_15_consecutive_positive_integers_is_375_l255_255874

noncomputable def sum_of_15_consecutive_integers_is_perfect_square
  (m : ℕ) : Prop :=
  let sum := 15 * (m + 7) in
  (∃ k : ℕ, sum = k * k)

theorem smallest_sum_of_15_consecutive_positive_integers_is_375 :
  ∃ m : ℕ, m > 0 ∧ sum_of_15_consecutive_integers_is_perfect_square m ∧ 15 * (m + 7) = 375 :=
by
  sorry

end smallest_sum_of_15_consecutive_positive_integers_is_375_l255_255874


namespace max_value_of_quadratic_l255_255897

theorem max_value_of_quadratic :
  ∀ z : ℝ, -6*z^2 + 24*z - 12 ≤ 12 :=
by
  sorry

end max_value_of_quadratic_l255_255897


namespace find_side_c_l255_255696

variable {A B C : Type} [triangle A B C]
variable (a b c : ℝ) (cosC : ℝ)
variable (sinB sinC : ℝ)

axiom given_conditions :
  (a = 2) ∧
  (cosC = -1/8) ∧
  (sinB = 2/3 * sinC)

theorem find_side_c (h : given_conditions) : c = 3 := sorry

end find_side_c_l255_255696


namespace number_of_x_squared_congruent_1_mod_n_l255_255669

-- Define the main theorem
theorem number_of_x_squared_congruent_1_mod_n (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℕ, k = 2 * (Finset.filter (λ (ab : ℕ × ℕ), ab.1 * ab.2 = n ∧ Nat.gcd ab.1 ab.2 = 1) 
    (Finset.product (Finset.Icc 1 n) (Finset.Icc 1 n))).card :=
by
  sorry

end number_of_x_squared_congruent_1_mod_n_l255_255669


namespace train_speed_l255_255130

theorem train_speed
    (train_length : ℕ := 800)
    (tunnel_length : ℕ := 500)
    (time_minutes : ℕ := 1)
    : (train_length + tunnel_length) * (60 / time_minutes) / 1000 = 78 := by
  sorry

end train_speed_l255_255130


namespace capacity_of_tank_is_823_l255_255944

def leak_rate (C : ℝ) : ℝ := C / 6
def inlet_rate := 240 -- 4 liters/min converted to liters/hour
def net_emptying_rate (C : ℝ) : ℝ := C / 8

theorem capacity_of_tank_is_823 :
  ∃ C : ℝ, C ≈ 823 ∧ inlet_rate - leak_rate C = net_emptying_rate C :=
by
  sorry

end capacity_of_tank_is_823_l255_255944


namespace tan_identity_l255_255680

variable {α : ℝ} (h : Real.tan α = 3)

theorem tan_identity : 
  (1 - Real.cos α) / Real.sin α + Real.sin α / (1 + Real.cos α) = sqrt 10 / (3 + sqrt 10) := 
sorry

end tan_identity_l255_255680


namespace sin_alpha_expression_value_l255_255237

variable {α : ℝ}
variable {P : ℝ × ℝ}
variable hP : P = (4/5, -3/5)
variable hα : ∃ α, (cos α, sin α) = (4/5, -3/5)

theorem sin_alpha :
  sin α = -3/5 :=
by sorry

theorem expression_value :
  (sin (π / 2 - α) / sin (α + π)) * (tan (α - π) / cos (3 * π - α)) = 5 / 4 :=
by sorry

end sin_alpha_expression_value_l255_255237


namespace min_odd_in_A_P_l255_255384

-- Define the polynomial P of degree 8
variable {R : Type*} [Ring R] (P : Polynomial R)
variable (hP : degree P = 8)

-- Define the set A_P for some value c in R
def A_P (c : R) : Set R := { x | P.eval x = c }

-- Given that the number 8 is included in the set A_P
variable (c : R) (h8 : (8 : R) ∈ A_P c)

-- Prove that the minimum number of odd numbers in the set A_P is 1
theorem min_odd_in_A_P : ∃ x ∈ A_P c, (x % 2 = 1) := sorry

end min_odd_in_A_P_l255_255384


namespace total_hours_charged_l255_255421

variables (K P M : ℕ)

theorem total_hours_charged (h1 : P = 2 * K) 
                            (h2 : P = (1/3 : ℚ) * (M : ℚ)) 
                            (h3 : M = K + 85) : 
  K + P + M = 153 := 
by 
  sorry

end total_hours_charged_l255_255421


namespace determine_counterfeit_l255_255573

namespace CounterfeitCoins

-- Definitions based on given conditions
def real_coins : Finset ℕ := {1, 2, 3, 4, 5}
def counterfeit_coins : Finset ℕ := {6, 7, 8, 9, 10}
def unknown_coins : Finset ℕ := {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}

-- The main problem statement:
theorem determine_counterfeit :
  ∃ (group_A group_B : Finset ℕ),
  group_A.card = 6 ∧
  group_B.card = 6 ∧
  group_A ∪ group_B = unknown_coins ∧
  (∃ num_of_weighings ≤ 4, can_determine_counterfeits group_A group_B real_coins counterfeit_coins num_of_weighings) :=
sorry

end CounterfeitCoins

end determine_counterfeit_l255_255573


namespace constant_term_in_expansion_l255_255476

theorem constant_term_in_expansion : 
  (∃ n : ℕ, (3 - 1/(2*3)) ^ n = 1/64) → 
  ∃ (c : ℝ), c = (-1/2) ^ 3 * (nat.choose 6 3) ∧ c = -5/2 :=
by
  sorry

end constant_term_in_expansion_l255_255476


namespace prove_statements_l255_255704

noncomputable def circle_center : ℝ × ℝ := ((5 + 1) / 2, (6 + 2) / 2)
noncomputable def circle_radius : ℝ := real.sqrt ((5 - circle_center.1)^2 + (2 - circle_center.2)^2)
noncomputable def circle (p : ℝ × ℝ) : Prop := (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2

def M1 : ℝ × ℝ := (2, 0)
def M2 : ℝ × ℝ := (-2, 0)
def O : ℝ × ℝ := (0, 0)

def N1 : ℝ × ℝ := ((O.1 + M1.1) / 2, (O.2 + M1.2) / 2)
def N2 : ℝ × ℝ := ((O.1 + M2.1) / 2, (O.2 + M2.2) / 2)

theorem prove_statements (A : ℝ × ℝ) (hA : circle A) :
  dist A M2 ≤ real.sqrt 41 + 2 * real.sqrt 2  ∧
  (∃ θ : ℝ, θ ≥ 15 ∧ angle A N2 N1 = θ) ∧
  ∃ min_val : ℝ, min_val = 32 - 20 * real.sqrt 2 ∧ (A.1 - N1.1) * (A.2 - N2.2) = min_val :=
by
  sorry

end prove_statements_l255_255704


namespace angle_inclination_range_l255_255627

-- Define the angle of inclination for the line
def angle_of_inclination (θ : ℝ) : ℝ :=
  if θ = 0 then π / 2 else Real.atan (1 / θ.sin)

-- The Lean statement to prove the range of the angle of inclination
theorem angle_inclination_range (θ : ℝ) : ∃ α : ℝ, angle_of_inclination θ = α ∧ (π / 4 ≤ α ∧ α ≤ 3 * π / 4) :=
sorry

end angle_inclination_range_l255_255627


namespace largest_possible_n_l255_255143

theorem largest_possible_n (b g : ℕ) (n : ℕ) (h1 : g = 3 * b)
  (h2 : ∀ (boy : ℕ), boy < b → ∀ (girlfriend : ℕ), girlfriend < g → girlfriend ≤ 2013)
  (h3 : ∀ (girl : ℕ), girl < g → ∀ (boyfriend : ℕ), boyfriend < b → boyfriend ≥ n) :
  n ≤ 671 := by
    sorry

end largest_possible_n_l255_255143


namespace shaded_region_area_l255_255309

theorem shaded_region_area (AH GF HF : ℝ) (AH_eq : AH = 12) (GF_eq : GF = 4) (HF_eq : HF = 16):
  let DG := 12 * GF / HF,
      area_triangle_DGF := 1/2 * DG * GF,
      area_square := 4 * 4,
      shaded_area := area_square - area_triangle_DGF
  in shaded_area = 10 :=
by
  let DG := 12 * GF / HF
  let area_triangle_DGF := 1/2 * DG * GF
  let area_square := 4 * 4
  let shaded_area := area_square - area_triangle_DGF
  sorry  

end shaded_region_area_l255_255309


namespace quadratic_condition_solutions_specific_a_for_x_values_l255_255262

theorem quadratic_condition_solutions:
  ∀ a : ℝ,
  (∀ x : ℝ, (-6 < x ∧ x ≤ -2) → x ≠ -5 ∧ x ≠ -4 ∧ x ≠ -3 →
    (x^2 - (a - 12) * x + 36 - 5 * a = 0)) →
    (a ∈ Ioo 4 (4.5) ∨ a ∈ Ioc 4.5 ((16 : ℝ) / 3)) := by
  sorry

theorem specific_a_for_x_values:
  (∀ x : ℝ, x = -4 → ∃ a : ℝ, a = 4 ∧ (x^2 - (a - 12) * x + 36 - 5 * a = 0)) ∧
  (∀ x : ℝ, x = -3 → ∃ a : ℝ, a = 4.5 ∧ (x^2 - (a - 12) * x + 36 - 5 * a = 0)) := by
  sorry

end quadratic_condition_solutions_specific_a_for_x_values_l255_255262


namespace sphere_divided_by_great_circles_l255_255773

/-- The maximum number of parts into which the surface of a sphere can be divided by n great circles is given by n^2 - n + 2. -/
theorem sphere_divided_by_great_circles (n : ℕ) (h : n > 0) : 
  ∃ (P : ℕ), P = n^2 - n + 2 :=
by
  use n^2 - n + 2
  sorry

end sphere_divided_by_great_circles_l255_255773


namespace polynomial_evaluation_l255_255073

theorem polynomial_evaluation :
  101^4 - 4 * 101^3 + 6 * 101^2 - 4 * 101 + 1 = 100000000 := sorry

end polynomial_evaluation_l255_255073


namespace ellipse_eccentricity_range_l255_255214

theorem ellipse_eccentricity_range (a b e : ℝ) (F1 F2 P : ℝ × ℝ) 
  (h_ellipse : ∀ P : ℝ × ℝ, P ∈ { P : ℝ × ℝ | (P.1^2 / a^2 + P.2^2 / b^2) = 1 }) 
  (h_foci : e = (F1.1 - 0) / (F2.1 - 0)) 
  (h_condition : a > b ∧ b > 0 ∧ (∀ P : ℝ × ℝ, P ∈ {P : ℝ × ℝ | (dist P F1) / (dist P F2) = e})) :
  sqrt 2 - 1 ≤ e ∧ e < 1 :=
sorry

end ellipse_eccentricity_range_l255_255214


namespace solution_set_inequality_l255_255030

noncomputable def f : ℝ → ℝ := sorry

axiom increasing_f (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x < y → f(x) < f(y)
axiom functional_eq_f (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f(x * y) = f(x) + f(y)
axiom f_three : f(3) = 1

theorem solution_set_inequality (x : ℝ) (hx : 0 < x) (hx8 : 0 < x - 8) :
  f(x) + f(x - 8) < 2 ↔ 8 < x ∧ x < 9 :=
by
  sorry

end solution_set_inequality_l255_255030


namespace number_of_sets_A_l255_255731

def setB : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def setC : Set ℕ := {0, 2, 4, 6, 8, 10}
def setIntersection := setB ∩ setC

theorem number_of_sets_A : (setIntersection.to_finset.powerset.card = 16) :=
by sorry

end number_of_sets_A_l255_255731


namespace intersection_points_l255_255616

noncomputable def h (x : ℝ) : ℝ := -x^2 - 4 * x + 1
noncomputable def j (x : ℝ) : ℝ := -h x
noncomputable def k (x : ℝ) : ℝ := h (-x)

def c : ℕ := 2 -- Number of intersections of y = h(x) and y = j(x)
def d : ℕ := 1 -- Number of intersections of y = h(x) and y = k(x)

theorem intersection_points :
  10 * c + d = 21 := by
  sorry

end intersection_points_l255_255616


namespace max_tourists_and_changes_l255_255527

theorem max_tourists_and_changes (m : ℕ) (k : ℕ) (x : ℕ) (a : ℕ) :
  (∃ m x, m = 1 ∧ k = 10 * m - x ∧ k = a^2 ∧
  (x = 1 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 9) ∧ 
  (∀ t1 t2 : ℕ, t1 ≠ t2 → 10 - t1 ≠ 10 - t2)) →
  ∃ n : ℕ, n = 5 ∧ ∀ x : ℕ, x ∈ {1, 4, 5, 6, 9}. sorry

end max_tourists_and_changes_l255_255527


namespace min_odd_numbers_in_A_P_l255_255356

theorem min_odd_numbers_in_A_P (P : ℝ → ℝ) 
  (hP : ∃ a8 a7 a6 a5 a4 a3 a2 a1 a0 : ℝ, P = λ x, a8 * x^8 + a7 * x^7 + a6 * x^6 + a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0)
  (h8_in_A_P : ∃ x : ℝ, P x = 8) : 
  (∀ x : ℝ, P x ∈ Set.A_P) → 
  (∃! y : ℝ, odd y ∧ y ∈ Set.A_P) := 
sorry

end min_odd_numbers_in_A_P_l255_255356


namespace cost_equality_store_comparison_for_10_l255_255063

-- price definitions
def teapot_price := 30
def teacup_price := 5
def teapot_count := 5

-- store A and B promotional conditions
def storeA_cost (x : Nat) : Real := 5 * x + 125
def storeB_cost (x : Nat) : Real := 4.5 * x + 135

theorem cost_equality (x : Nat) (h : x > 5) :
  storeA_cost x = storeB_cost x → x = 20 := by
  sorry

theorem store_comparison_for_10 (x : Nat) (h : x = 10) :
  storeA_cost x < storeB_cost x := by
  sorry

end cost_equality_store_comparison_for_10_l255_255063


namespace toll_for_18_wheel_truck_l255_255047

-- Define the conditions
def wheels_per_axle : Nat := 2
def total_wheels : Nat := 18
def toll_formula (x : Nat) : ℝ := 1.5 + 0.5 * (x - 2)

-- Calculate number of axles from the number of wheels
def number_of_axles := total_wheels / wheels_per_axle

-- Target statement: The toll for the given truck
theorem toll_for_18_wheel_truck : toll_formula number_of_axles = 5.0 := by
  sorry

end toll_for_18_wheel_truck_l255_255047


namespace circle_cos_intersections_l255_255989

-- Define the conditions
def circle_eq (a b c x y : ℝ) : Prop := (x - a) ^ 2 + (y - b) ^ 2 = c ^ 2
def cos_eq (x : ℝ) : ℝ := Real.cos x

-- Lean 4 statement for the problem
theorem circle_cos_intersections (a b c : ℝ) :
  ∃ (points : List (ℝ × ℝ)) (n : ℕ), n ≤ 8 ∧
  ∀ p ∈ points, circle_eq a b c p.1 (cos_eq p.1) :=
sorry

end circle_cos_intersections_l255_255989


namespace probability_chord_intersects_inner_circle_l255_255493

theorem probability_chord_intersects_inner_circle :
  let R_out := 4,
      R_in := 2,
      intersection_angle := 120
  in (intersection_angle / 360 : ℝ) = 1 / 3 := by {
  let R_out := 4
  let R_in := 2
  let intersection_angle := 120
  have h : (intersection_angle / 360 : ℝ) = 1 / 3 := sorry
  exact h
}

end probability_chord_intersects_inner_circle_l255_255493


namespace solve_for_square_l255_255270

theorem solve_for_square (x : ℝ) 
  (h : 10 + 9 + 8 * 7 / x + 6 - 5 * 4 - 3 * 2 = 1) : 
  x = 28 := 
by 
  sorry

end solve_for_square_l255_255270


namespace circumcircle_radius_of_triangle_l255_255414

theorem circumcircle_radius_of_triangle (A B C M N : Point)
  (h1 : on_bisector B A C M)
  (h2 : on_extension A B N)
  (h3 : distance A C = 1)
  (h4 : distance A M = 1)
  (h5 : angle A N M = angle C N M) :
  radius (circumcircle C N M) = 1 := 
sorry

end circumcircle_radius_of_triangle_l255_255414


namespace sqrt_square_identity_sqrt_529441_squared_l255_255612

theorem sqrt_square_identity (n : ℕ) : (Nat.sqrt n) ^ 2 = n := sorry

theorem sqrt_529441_squared : (Nat.sqrt 529441) ^ 2 = 529441 := 
begin
  exact sqrt_square_identity 529441,
end

end sqrt_square_identity_sqrt_529441_squared_l255_255612


namespace max_books_borrowed_theorem_l255_255300

def max_books_borrowed (n_students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ) (avg_books : ℕ) : ℕ :=
  let remaining_students := n_students - (no_books + one_book + two_books)
  let total_borrowed_books := n_students * avg_books
  let total_counted_books := one_book * 1 + two_books * 2
  let remaining_books := total_borrowed_books - total_counted_books
  let atleast_three_books := remaining_books - remaining_students * 3 in
  atleast_three_books + 3

theorem max_books_borrowed_theorem :
  max_books_borrowed 40 2 12 14 2 = 10 :=
by 
  -- This proof is omitted
  sorry

end max_books_borrowed_theorem_l255_255300


namespace emt_selection_ways_l255_255726

theorem emt_selection_ways :
  let groupA_nurses := 4
  let groupA_doctors := 1
  let groupB_nurses := 6
  let groupB_doctors := 2
  let ways_scenario1 := (nat.choose groupA_doctors 1) * (nat.choose groupA_nurses 1) * 
                        (nat.choose groupB_nurses 2) * (nat.choose groupB_doctors 0)
  let ways_scenario2 := (nat.choose groupA_doctors 0) * (nat.choose groupA_nurses 2) * 
                        (nat.choose groupB_doctors 1) * (nat.choose groupB_nurses 1)
  ways_scenario1 + ways_scenario2 = 132 := by 
let groupA_nurses := 4
let groupA_doctors := 1
let groupB_nurses := 6
let groupB_doctors := 2
let ways_scenario1 := (nat.choose groupA_doctors 1) * (nat.choose groupA_nurses 1) * 
                      (nat.choose groupB_nurses 2) * (nat.choose groupB_doctors 0)
let ways_scenario2 := (nat.choose groupA_doctors 0) * (nat.choose groupA_nurses 2) * 
                      (nat.choose groupB_doctors 1) * (nat.choose groupB_nurses 1)
have : ways_scenario1 = 60 :=
  by norm_num [nat.choose] 
have : ways_scenario2 = 72 := 
  by norm_num [nat.choose] 
show (60 + 72 = 132) from rfl

end emt_selection_ways_l255_255726


namespace same_number_written_every_vertex_l255_255125

theorem same_number_written_every_vertex (a : ℕ → ℝ) (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → a i > 0) 
(h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → (a i) ^ 2 = a (i - 1) + a (i + 1) ) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → a i = 2 :=
by
  sorry

end same_number_written_every_vertex_l255_255125


namespace units_digit_35_pow_35_mul_17_pow_17_l255_255183

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_35_pow_35_mul_17_pow_17:
  units_digit (35 ^ (35 * 17 ^ 17)) = 5 := 
by {
  -- Here we're skipping the proof.
  sorry
}

end units_digit_35_pow_35_mul_17_pow_17_l255_255183


namespace theta_in_second_quadrant_l255_255451

theorem theta_in_second_quadrant
  (θ : ℝ)
  (h1 : sin (2 * θ) < 0)
  (h2 : cos θ - sin θ < 0) :
  π / 2 < θ ∧ θ < π :=
sorry

end theta_in_second_quadrant_l255_255451


namespace tan_six_minus_tan_two_geo_seq_l255_255271

theorem tan_six_minus_tan_two_geo_seq (x : ℝ) (h : (cos x)^2 = (sin x) * (cos x) * (cot x)) :
  (tan x)^6 - (tan x)^2 = (sin x)^2 :=
sorry

end tan_six_minus_tan_two_geo_seq_l255_255271


namespace unique_triple_l255_255649

theorem unique_triple (x y z : ℤ) (h₁ : x + y = z) (h₂ : y + z = x) (h₃ : z + x = y) :
  (x = 0) ∧ (y = 0) ∧ (z = 0) :=
sorry

end unique_triple_l255_255649


namespace no_such_cut_possible_l255_255774

theorem no_such_cut_possible (a b c d : ℤ) :
  let A := 1 + 0 * real.sqrt 3
  let B := a + b * real.sqrt 3
  let C := c + d * real.sqrt 3
  ¬ ∃ (A_hex : ℝ) (n : ℕ), A_hex = 3 * real.sqrt 3 / 2 ∧ n * (real.sqrt 3 / 2) = 3 * real.sqrt 3 / 2 ∧
  ((a * c + 3 * b * d) + (a * d + b * c) * real.sqrt 3 = A_hex) :=
sorry

end no_such_cut_possible_l255_255774


namespace midpoint_prod_l255_255792

-- Conditions
def midpoint (P Q D : ℝ × ℝ) : Prop :=
  D = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def P : ℝ × ℝ := (2, 9)
def D : ℝ × ℝ := (-4, 2)

-- To prove
theorem midpoint_prod (a b : ℝ) (Q : ℝ × ℝ := (a, b)) : midpoint P Q D → a * b = 50 := by
  sorry

end midpoint_prod_l255_255792


namespace correct_answer_is_f2_l255_255976

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : {a // a > 0}, x1 < x2 → f x1 > f x2

def f1 (x : ℝ) : ℝ := x^2 - 1
def f2 (x : ℝ) : ℝ := -2^(|x|)
def f3 (x : ℝ) : ℝ := 1 / x
def f4 (x : ℝ) : ℝ := cos x

theorem correct_answer_is_f2 :
  (is_even f2 ∧ is_monotonically_decreasing f2) ∧
  ¬(is_even f1 ∧ is_monotonically_decreasing f1) ∧
  ¬(is_even f3 ∧ is_monotonically_decreasing f3) ∧
  ¬(is_even f4 ∧ is_monotonically_decreasing f4) :=
by
  sorry

end correct_answer_is_f2_l255_255976


namespace constant_term_of_expansion_l255_255802

noncomputable theory

def a_value : ℝ := ∫ x in 0..(Real.pi / 2), Real.sin x

theorem constant_term_of_expansion :
  a_value = 1 → 
  (∃ c : ℝ, ∀ x : ℝ, c = constant_term_of_expansion((2*x + a_value/x)^6)) :=
by
  intro h
  have a_eq_one : a_value = 1 := h
  sorry

end constant_term_of_expansion_l255_255802


namespace problem_equivalent_l255_255744

variable (p : ℤ) 

theorem problem_equivalent (h : p = (-2023) * 100) : (-2023) * 99 = p + 2023 :=
by sorry

end problem_equivalent_l255_255744


namespace LM_eq_AK_l255_255834

variable {α β γ : ℝ} -- angles of triangle
variable {t : ℝ} -- length of KL and LC
variable {A B C K L M : Point} -- points of the triangle and on the sides

-- Definitions related to the problem
def is_parallel (p q : Line) : Prop := sorry
def length (p q : Point) : ℝ := sorry
def angle (p q r : Point) : ℝ := sorry

-- Conditions given in the problem
axiom triangle_conditions :
  ∃ (A B C K L M : Point), 
    (length K L = t) ∧ 
    (length L C = t) ∧ 
    (is_parallel (line K L) (line B C)) ∧ 
    (angle L M B = α) ∧ 
    (length M L = length A K)

-- Problem statement
theorem LM_eq_AK :
  ∀ (A B C K L M : Point), 
    (length K L = t) → 
    (length L C = t) →
    (is_parallel (line K L) (line B C)) → 
    (angle L M B = α) → 
    length M L = length A K :=
by
  assume A B C K L M hKL hLC h_parallel h_angle
  exact triangle_conditions_elim A B C K L M hKL hLC h_parallel h_angle

end LM_eq_AK_l255_255834


namespace license_plates_count_l255_255116

noncomputable def number_of_distinct_license_plates : ℕ :=
  let digits_choices := 10^5
  let letter_block_choices := 26^3
  let positions_choices := 6
  positions_choices * digits_choices * letter_block_choices

theorem license_plates_count : number_of_distinct_license_plates = 105456000 := by
  unfold number_of_distinct_license_plates
  calc
    6 * 10^5 * 26^3 = 6 * 100000 * 17576 : by norm_num
                  ... = 105456000 : by norm_num
  sorry

end license_plates_count_l255_255116


namespace ff1_eq_3_l255_255249

def f (x : ℝ) : ℝ :=
if x >= 3 then x^2 - 2 * x else 2 * x + 1

theorem ff1_eq_3 : f (f 1) = 3 :=
by
  sorry

end ff1_eq_3_l255_255249


namespace eq_sin_intersect_16_solutions_l255_255662

theorem eq_sin_intersect_16_solutions :
  ∃ S : Finset ℝ, (∀ x ∈ S, 0 ≤ x ∧ x ≤ 50 ∧ (x / 50 = Real.sin x)) ∧ (S.card = 16) :=
  sorry

end eq_sin_intersect_16_solutions_l255_255662


namespace find_m_l255_255610

open Real

/-- Define Circle C1 and C2 as having the given equations
and verify their internal tangency to find the possible m values -/
theorem find_m (m : ℝ) :
  (∃ (x y : ℝ), (x - m)^2 + (y + 2)^2 = 9) ∧ 
  (∃ (x y : ℝ), (x + 1)^2 + (y - m)^2 = 4) ∧ 
  (by exact (sqrt ((m + 1)^2 + (-2 - m)^2)) = 3 - 2) → 
  m = -2 ∨ m = -1 := 
sorry -- Proof is omitted

end find_m_l255_255610


namespace triangle_area_ratio_l255_255079

open set

variable {α : Type*} [fintype α] [linear_ordered_field α]

/--Given ΔABC where points D, E, F divide sides BC, CA, AB respectively into segments such that 
  ∥CD∥ = 1/4 ∥BC∥, ∥AE∥ = 1/4 ∥CA∥, ∥BF∥ = 1/4 ∥AB∥, 
  and internal points N1, N2, N3 divide respective segments proportionally,
  the area of triangle N1N2N3 is 1/16 of the area of triangle ABC.-/
theorem triangle_area_ratio 
  {a b c d e f n1 n2 n3 : α}
  (h1 : d = b + (c - b) / 4) 
  (h2 : e = a + (c - a) / 4)
  (h3 : f = b + (a - b) / 4)
  (h4 : (a - n2) / (n2 - n1) = 4 / 3 ∧ (n2 - n1) / (n1 - d) = 3 / 1) : 
  (area_of_triangle n1 n2 n3) = (1 / 16) * (area_of_triangle a b c) :=
  
sorry 
-- Proof omitted

end triangle_area_ratio_l255_255079


namespace problem_statement_l255_255742

theorem problem_statement (a b : ℝ) (h : (b + 5)^2 = 0) : a + b = a - 5 :=
by
  have hb : b + 5 = 0 := by
    exact eq_zero_of_square_eq_zero h
  have hbneg : b = -5 := by
    linarith
  rw [hbneg]
  linarith

end problem_statement_l255_255742


namespace sum_of_first_9_y_terms_l255_255234

noncomputable def a (n : ℕ) := (n - 1) * d + a_1 -- Definition of the arithmetic sequence

def f (x : ℝ) : ℝ := sin (2 * x) + cos x + 1

def y (n : ℕ) : ℝ := f (a n)

theorem sum_of_first_9_y_terms (a_5 : ℝ) (d : ℝ) (h : a 5 = π / 2) :
  (y 1 + y 2 + y 3 + y 4 + y 5 + y 6 + y 7 + y 8 + y 9) = 9 := by
  sorry

end sum_of_first_9_y_terms_l255_255234


namespace land_suitable_acres_l255_255778

variable (previousProperty newProperty pond landSuitable : ℕ)

-- Define the conditions based on the given problem
def previous_property : Prop := previousProperty = 2
def new_property : Prop := newProperty = 8 * previousProperty
def pond_size : Prop := pond = 3
def land_suitable : Prop := landSuitable = newProperty - pond

-- State the theorem
theorem land_suitable_acres (h1 : previous_property) (h2 : new_property) (h3 : pond_size) : land_suitable :=
by
  sorry

end land_suitable_acres_l255_255778


namespace tracy_has_2_dogs_l255_255491

-- Definitions according to the conditions in a)

def cups_per_meal := 1.5
def meals_per_day := 3
def pounds_of_food := 4
def cups_per_pound := 2.25

-- The Lean statement to prove
theorem tracy_has_2_dogs : 
  (pounds_of_food * cups_per_pound) / (cups_per_meal * meals_per_day) = 2 := 
by sorry

end tracy_has_2_dogs_l255_255491


namespace minimum_odd_numbers_in_set_l255_255394

-- Definitions
variable (P : ℝ → ℝ)
variable (degree_P : ℕ)
variable (A_P : Set ℝ)

-- The conditions: P is a polynomial of degree 8, and 8 is included in A_P
def is_polynomial_of_degree_eight (P : ℝ → ℝ) (degree_P : ℕ) : Prop :=
  degree_P = 8

def set_includes_eight (A_P : Set ℝ) : Prop := 
  8 ∈ A_P

-- The goal: prove the minimum number of odd numbers in A_P is 1
theorem minimum_odd_numbers_in_set {P : ℝ → ℝ} {degree_P : ℕ} {A_P : Set ℝ} :
  is_polynomial_of_degree_eight P degree_P → 
  set_includes_eight A_P → 
  ∃ odd_numbers : ℕ, odd_numbers = 1 :=
sorry

end minimum_odd_numbers_in_set_l255_255394


namespace smallest_positive_multiple_of_37_l255_255506

theorem smallest_positive_multiple_of_37 :
  ∃ n, n > 0 ∧ (∃ a, n = 37 * a) ∧ (∃ k, n = 76 * k + 7) ∧ n = 2405 := 
by
  sorry

end smallest_positive_multiple_of_37_l255_255506


namespace vector_sum_to_zero_l255_255722

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

def collinear (u v : V) : Prop :=
∃ (λ : ℝ), u = λ • v

theorem vector_sum_to_zero
  (h_collinear_ab_c : collinear (a + b) c)
  (h_collinear_bc_a : collinear (b + c) a)
  (h_not_collinear : ¬ collinear a c) :
  a + b + c = 0 :=
sorry

end vector_sum_to_zero_l255_255722


namespace average_age_of_choir_l255_255852

theorem average_age_of_choir 
  (num_females : ℕ) (avg_age_females : ℝ)
  (num_males : ℕ) (avg_age_males : ℝ)
  (total_people : ℕ) (total_people_eq : total_people = num_females + num_males) :
  num_females = 12 → avg_age_females = 28 → num_males = 18 → avg_age_males = 38 → total_people = 30 →
  (num_females * avg_age_females + num_males * avg_age_males) / total_people = 34 := by
  intros
  sorry

end average_age_of_choir_l255_255852


namespace inscribed_triangle_angle_l255_255304

-- Define the conditions as hypotheses.
theorem inscribed_triangle_angle : 
  ∀ (x : ℝ), 
  (x + 85 + 2 * x + 15 + 3 * x - 32 = 360) → 
  (1/2 * (2 * x + 15) = 57 ∨ 1/2 * (3 * x - 32) = 57 ∨ 1/2 * (x + 85) = 57) := 
by {
  intro x,
  intro h_eq,
  sorry -- Proof goes here.
}

end inscribed_triangle_angle_l255_255304


namespace determine_counterfeit_l255_255574

namespace CounterfeitCoins

-- Definitions based on given conditions
def real_coins : Finset ℕ := {1, 2, 3, 4, 5}
def counterfeit_coins : Finset ℕ := {6, 7, 8, 9, 10}
def unknown_coins : Finset ℕ := {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}

-- The main problem statement:
theorem determine_counterfeit :
  ∃ (group_A group_B : Finset ℕ),
  group_A.card = 6 ∧
  group_B.card = 6 ∧
  group_A ∪ group_B = unknown_coins ∧
  (∃ num_of_weighings ≤ 4, can_determine_counterfeits group_A group_B real_coins counterfeit_coins num_of_weighings) :=
sorry

end CounterfeitCoins

end determine_counterfeit_l255_255574


namespace max_length_common_chord_l255_255037

theorem max_length_common_chord (a b : ℝ) :
  let circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * a * p.1 + 2 * a * p.2 + 2 * a^2 - 1 = 0},
      circle2 := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * b * p.1 + 2 * b * p.2 + 2 * b^2 - 2 = 0} in
  ∃ chord_len, chord_len = 2 :=
by
  sorry

end max_length_common_chord_l255_255037


namespace geometric_sequence_a5_l255_255209

theorem geometric_sequence_a5 (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r) (h2 : a 2 * a 8 = 4) : a 5 = 2 :=
sorry

end geometric_sequence_a5_l255_255209


namespace largest_non_representable_integer_l255_255070

theorem largest_non_representable_integer (n a b : ℕ) (h₁ : n = 42 * a + b)
  (h₂ : 0 ≤ b) (h₃ : b < 42) (h₄ : ¬ (b % 6 = 0)) :
  n ≤ 252 :=
sorry

end largest_non_representable_integer_l255_255070


namespace maximize_profit_l255_255540

-- Define the parameters given in the conditions.
def cost_per_unit := 40
def factory_price_per_unit := 60
def daily_sales_volume := 1000

-- Define the percentage increases as functions of x.
def percentage_increase_cost (x : ℝ) := 1 + x
def percentage_increase_price (x : ℝ) := 1 + 0.5 * x
def percentage_increase_sales (x : ℝ) := 1 + 0.8 * x

-- Define the profit function y in terms of x.
noncomputable def profit (x : ℝ) : ℝ :=
  (factory_price_per_unit * percentage_increase_price x - cost_per_unit * percentage_increase_cost x) * daily_sales_volume * percentage_increase_sales x
 
-- Statement of the math proof problem.
theorem maximize_profit :
  (∀ x, 0 < x ∧ x < 1 → profit x = 2000 * (-4 * x^2 + 3 * x + 10)) ∧
  (∃ x, 0 < x ∧ x < 1 ∧ is_max (profit x) = (x = 0.375)) :=
by sorry

end maximize_profit_l255_255540


namespace selection_schemes_l255_255939

theorem selection_schemes (boys girls : ℕ) (hb : boys = 4) (hg : girls = 2) :
  (boys * girls = 8) :=
by
  -- Proof goes here
  intros
  sorry

end selection_schemes_l255_255939


namespace marbles_in_jar_l255_255106

theorem marbles_in_jar (x : ℕ)
  (h1 : \frac{1}{2} * x + \frac{1}{4} * x + 27 + 14 = x) : x = 164 := sorry

end marbles_in_jar_l255_255106


namespace range_of_H_l255_255899

def H (x : ℝ) : ℝ := |x + 3| - |x - 2|

theorem range_of_H : set.range H = set.Iic 5 := 
by
  sorry

end range_of_H_l255_255899


namespace smallest_number_of_students_l255_255301

theorem smallest_number_of_students :
  ∃ n : ℕ, n > 0 ∧
    (∀ (students : Finset ℕ) (friend_count : ℕ → ℕ),
      (∀ student ∈ students, friend_count student = 5 ∨ friend_count student = 7) ∧
      (∀ s1 s2 ∈ students, s1 ≠ s2 → friend_count s1 ≠ friend_count s2 → 
        (friend_count s1 = 5 ∧ friend_count s2 = 7 ∨ friend_count s1 = 7 ∧ friend_count s2 = 5)) ∧
      (∀ s1 s2, (s1 ∈ students ∧ s2 ∈ students ∧ friend_count s1 ≠ friend_count s2) →
        s1 ≠ s2 ∧ ∃ fs, fs ⊆ students ∧ count fs s1 = 5 ∧ count fs s2 = 7)) ∧
    n = 12 :=
sorry

end smallest_number_of_students_l255_255301


namespace speed_with_stream_l255_255947

noncomputable def man_speed_still_water : ℝ := 5
noncomputable def speed_against_stream : ℝ := 4

theorem speed_with_stream :
  ∃ V_s, man_speed_still_water + V_s = 6 :=
by
  use man_speed_still_water - speed_against_stream
  sorry

end speed_with_stream_l255_255947


namespace exists_vertex_set_l255_255688
open Set

variable (P : Type) [Polygon P]
variable [NonIntersecting P]
variable [NonConvex P]
variable (n : ℕ)

theorem exists_vertex_set (hP : Vertices P = n) :
  ∃ A : Set (Vertices P), 
    A.card = (n / 3) ∧ 
    ∀ X ∈ P, ∃ C ∈ A, Segment C X ⊆ P :=
by 
  sorry

end exists_vertex_set_l255_255688


namespace grape_star_probability_l255_255535

theorem grape_star_probability 
  (num_squares : ℕ) (num_triangles : ℕ) (num_stars : ℕ)
  (num_flavours : ℕ)
  (h1 : num_squares = 60)
  (h2 : num_triangles = 60)
  (h3 : num_stars = 60)
  (h4 : num_flavours = 3) :
  let total_tablets := num_squares + num_triangles + num_stars in
  let grape_stars := num_stars / num_flavours in
  (grape_stars : ℚ) / total_tablets = 1 / 9 :=
by 
  sorry

end grape_star_probability_l255_255535


namespace cos_graph_transformation_l255_255886

theorem cos_graph_transformation :
  ∀ x : ℝ, cos (2 * x + π) = cos (x + π) := by
sorry

end cos_graph_transformation_l255_255886


namespace water_tank_equilibrium_l255_255132

theorem water_tank_equilibrium :
  (1 / 15 : ℝ) + (1 / 10 : ℝ) - (1 / 6 : ℝ) = 0 :=
by
  sorry

end water_tank_equilibrium_l255_255132


namespace B_contribution_l255_255913

-- Define the conditions
def capitalA : ℝ := 3500
def monthsA : ℕ := 12
def monthsB : ℕ := 7
def profit_ratio_A : ℕ := 2
def profit_ratio_B : ℕ := 3

-- Statement: B's contribution to the capital
theorem B_contribution :
  (capitalA * monthsA * profit_ratio_B) / (monthsB * profit_ratio_A) = 4500 := by
  sorry

end B_contribution_l255_255913


namespace max_rectangles_l255_255860

theorem max_rectangles (figure : Type) (is_grid : figure -> Prop)
  (coloring_rule : ∀ (x y : ℕ), is_black (x, y) <-> (x + y) % 2 = 0)
  (count_black_squares : ∀ f, count f is_black = 5) :
  ∃ n, n = 5 ∧ ∀ (rect : (ℕ × ℕ) × (ℕ × ℕ)), is_rectangle rect -> covers_black_and_white rect ∧ total_rectangles figure = n := 
sorry

end max_rectangles_l255_255860


namespace James_vegetable_intake_in_third_week_l255_255775

noncomputable def third_week_vegetable_intake : ℝ :=
  let asparagus_per_day_first_week : ℝ := 0.25
  let broccoli_per_day_first_week : ℝ := 0.25
  let cauliflower_per_day_first_week : ℝ := 0.5

  let asparagus_per_day_second_week := 2 * asparagus_per_day_first_week
  let broccoli_per_day_second_week := 3 * broccoli_per_day_first_week
  let cauliflower_per_day_second_week := cauliflower_per_day_first_week * 1.75
  let spinach_per_day_second_week : ℝ := 0.5
  
  let daily_intake_second_week := asparagus_per_day_second_week +
                                  broccoli_per_day_second_week +
                                  cauliflower_per_day_second_week +
                                  spinach_per_day_second_week
  
  let kale_per_day_third_week : ℝ := 0.5
  let zucchini_per_day_third_week : ℝ := 0.15
  
  let daily_intake_third_week := asparagus_per_day_second_week +
                                 broccoli_per_day_second_week +
                                 cauliflower_per_day_second_week +
                                 spinach_per_day_second_week +
                                 kale_per_day_third_week +
                                 zucchini_per_day_third_week
  
  daily_intake_third_week * 7

theorem James_vegetable_intake_in_third_week : 
  third_week_vegetable_intake = 22.925 :=
  by
    sorry

end James_vegetable_intake_in_third_week_l255_255775


namespace expression_eq_one_l255_255343

theorem expression_eq_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a + b + c = 1) :
  (a^2 * b^2 / ((a^2 - b * c) * (b^2 - a * c)) +
   a^2 * c^2 / ((a^2 - b * c) * (c^2 - a * b)) +
   b^2 * c^2 / ((b^2 - a * c) * (c^2 - a * b))) = 1 := 
by
  sorry

end expression_eq_one_l255_255343


namespace solve_monkey_bananas_l255_255676

noncomputable def monkey_bananas (b1 b2 b3 b4 : ℤ) : Prop :=
  (5 * (b1 * 5 / 6 + b2 * 4 / 15 + b3 * 8 / 27 + b4 * 8 / 36) =
   3 * (b1 * 1 / 15 + b2 * 2 / 3 + b3 * 8 / 27 + b4 * 8 / 36)) ∧
  (5 * (b1 * 5 / 6 + b2 * 4 / 15 + b3 * 8 / 27 + b4 * 8 / 36) =
   2 * (b1 * 1 / 15 + b2 * 4 / 15 + b3 * 1 / 3 + b4 * 8 / 36)) ∧
  (5 * (b1 * 5 / 6 + b2 * 4 / 15 + b3 * 8 / 27 + b4 * 8 / 36) =
   1 * (b1 * 1 / 15 + b2 * 4 / 15 + b3 * 8 / 27 + b4 * 1 / 9)) ∧
  (b1 % 5 = 0) ∧ (b2 % 5 = 0) ∧ (b3 % 5 = 0) ∧ (b4 % 5 = 0)

/-- The minimal possible total number of bananas under given conditions -/
def minimal_bananas : ℤ :=
  Inf {n : ℤ | ∃ b1 b2 b3 b4, monkey_bananas b1 b2 b3 b4 ∧ b1 + b2 + b3 + b4 = n}

-- The statement of the problem
theorem solve_monkey_bananas : minimal_bananas > 0 :=
sorry

end solve_monkey_bananas_l255_255676


namespace monotonicity_when_a_is_one_range_of_a_for_two_zeros_l255_255713

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * (x + 2)

theorem monotonicity_when_a_is_one :
  (∀ x < 0, deriv (f x 1) < 0) ∧ (∀ x > 0, deriv (f x 1) > 0) :=
by
  sorry

theorem range_of_a_for_two_zeros :
  (∀ a > 1 / Real.exp, ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) ∧
  (∀ a ≤ 1 / Real.exp, ∀ x, f x a ≠ 0) :=
by
  sorry

end monotonicity_when_a_is_one_range_of_a_for_two_zeros_l255_255713


namespace find_semi_perimeter_l255_255427

noncomputable def semi_perimeter_of_rectangle (a b : ℝ) (h₁ : a * b = 4024) (h₂ : a = 2 * b) : ℝ :=
  (a + b) / 2

theorem find_semi_perimeter (a b : ℝ) (h₁ : a * b = 4024) (h₂ : a = 2 * b) : semi_perimeter_of_rectangle a b h₁ h₂ = (3 / 2) * Real.sqrt 2012 :=
  sorry

end find_semi_perimeter_l255_255427


namespace total_hours_proof_l255_255488

-- Conditions
def half_hour_show_episodes : ℕ := 24
def one_hour_show_episodes : ℕ := 12
def half_hour_per_episode : ℝ := 0.5
def one_hour_per_episode : ℝ := 1.0

-- Define the total hours Tim watched
def total_hours_watched : ℝ :=
  half_hour_show_episodes * half_hour_per_episode + one_hour_show_episodes * one_hour_per_episode

-- Prove that the total hours watched is 24
theorem total_hours_proof : total_hours_watched = 24 := by
  sorry

end total_hours_proof_l255_255488


namespace probability_ab_even_l255_255494

theorem probability_ab_even (a b : ℕ) (h : a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} ∧ b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} ∧ a ≠ b) : 
  ∃ p q : ℚ, p = 5 ∧ q = 22 ∧ (∃ n : ℕ, (n = 66) ∧ 
  ∃ k : ℕ, (k = 15) ∧ p / q = (k : ℚ) / (n : ℚ)) := 
sorry

end probability_ab_even_l255_255494


namespace find_x_l255_255189

theorem find_x (b x : ℝ) (hb : 1 < b) (hx : 0 < x)
  (h : (5 * x) ^ (Real.log b 5) - (7 * x) ^ (Real.log b 7) = 0) :
  x = (7 / 5) ^ (Real.log (7 / 5) b) :=
by
  sorry

end find_x_l255_255189


namespace minimum_odd_numbers_in_A_P_l255_255371

-- Polynomials and assumptions
def degree (P : Polynomial ℝ) : ℕ := P.natDegree

-- The set A_P is defined as the set of all numbers x for which P(x) gives a certain value
def A_P (P : Polynomial ℝ) : Set ℝ := {x : ℝ | P.eval x = P.eval 8}

-- Define the main theorem statement
theorem minimum_odd_numbers_in_A_P (P : Polynomial ℝ) (hdeg : degree P = 8) (h8 : 8 ∈ A_P P) : 
  ∃ n, n = 1 ∧ ∃ (x : ℝ), x ∈ A_P P ∧ ¬ (x % 2 = 0) := sorry

end minimum_odd_numbers_in_A_P_l255_255371


namespace arithmetic_sequence_ratio_l255_255691

theorem arithmetic_sequence_ratio
  (d : ℕ) (h₀ : d ≠ 0)
  (a : ℕ → ℕ)
  (h₁ : ∀ n, a (n + 1) = a n + d)
  (h₂ : (a 3)^2 = (a 1) * (a 9)) :
  (a 1 + a 3 + a 6) / (a 2 + a 4 + a 10) = 5 / 8 :=
  sorry

end arithmetic_sequence_ratio_l255_255691


namespace part1_part2_part3_l255_255821

-- Define the statement for part (1)
theorem part1 : ∀ x : ℝ, x + 20 / x = -9 ↔ (x = -4 ∨ x = -5) :=
by
  intro x
  split
  sorry

-- Define the statement for part (2)
theorem part2 (n : ℕ) : ∀ x : ℝ, x + (n^2 + n) / x = -(2 * n + 1) ↔ (x = -n ∨ x = - (n + 1)) :=
by
  intro x
  split
  sorry

-- Define the statement for part (3)
theorem part3 (n : ℕ) (h : n > 0) : ∀ x : ℝ, x + (n^2 + n) / (x + 3) = -2 * (n + 2) ↔ (x = -n - 3 ∨ x = - (n + 4)) :=
by
  intro x
  split
  sorry

end part1_part2_part3_l255_255821


namespace expected_no_advice_formula_l255_255595

noncomputable def expected_no_advice (n : ℕ) (p : ℝ) : ℝ :=
  ∑ j in Finset.range n, (1 - p) ^ j

theorem expected_no_advice_formula (n : ℕ) (p : ℝ) (hp : 0 ≤ p ∧ p < 1) : 
  expected_no_advice n p = (1 - (1 - p) ^ n) / p :=
by
  sorry

end expected_no_advice_formula_l255_255595


namespace percentage_disliked_by_both_l255_255829

theorem percentage_disliked_by_both 
  (total_comic_books : ℕ) 
  (percentage_females_like : ℕ) 
  (comic_books_males_like : ℕ) :
  total_comic_books = 300 →
  percentage_females_like = 30 →
  comic_books_males_like = 120 →
  ((total_comic_books - (total_comic_books * percentage_females_like / 100) - comic_books_males_like) * 100 / total_comic_books) = 30 :=
by
  intros h1 h2 h3
  sorry

end percentage_disliked_by_both_l255_255829


namespace trigonometric_identity_l255_255683

variable x y : ℝ

theorem trigonometric_identity (h : x^2 + y^2 = 12 * x - 4 * y - 40) : 
  x * Real.cos (-23 / 3 * Real.pi) + y * Real.tan (-15 / 4 * Real.pi) = 1 := 
by
  sorry

end trigonometric_identity_l255_255683


namespace age_double_in_future_l255_255121

-- Definitions based on conditions
def man_age (son_age : ℕ) : ℕ := son_age + 28
def present_son_age : ℕ := 26

-- Lean statement to prove that the number of years Y from now when the man's age will be twice his son's age is 2
theorem age_double_in_future (Y : ℕ) : Y = 2 :=
  let S := present_son_age
  let M := man_age S
  (M + Y) = 2 * (S + Y) → Y = 2
  sorry

end age_double_in_future_l255_255121


namespace part1_part2_l255_255246

-- Part 1: Prove that if the minimum value of f(x) is 0, then a = 2
theorem part1 (a : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = Real.exp (2 * x) - a * x - 1)
  (h₂ : ∃ c, ∀ x, f(x) ≥ f(c)) (h₃ : f 0 = 0) : a = 2 := by
  sorry

-- Part 2: Prove that if g(x) is increasing, the range of a is (-∞, 4]
theorem part2 (a : ℝ) (g : ℝ → ℝ) (h₄ : ∀ x, g x = (let f := Real.exp (2 * x) - a * x - 1 in 
                                          f - (Real.log x) ^ 2 - 2 * (Real.log x)) )
  (h₅ : ∀ x, g' x ≥ 0 ) : a ≤ 4 := by
  sorry

end part1_part2_l255_255246


namespace range_of_b_l255_255203

def M := {p : ℝ × ℝ | p.1 ^ 2 + 2 * p.2 ^ 2 = 3}
def N (m b : ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 + b}

theorem range_of_b (b : ℝ) : (∀ (m : ℝ), (∃ (p : ℝ × ℝ), p ∈ M ∧ p ∈ N m b)) ↔ 
  -Real.sqrt (6) / 2 ≤ b ∧ b ≤ Real.sqrt (6) / 2 :=
by
  sorry

end range_of_b_l255_255203


namespace range_of_m_l255_255244

theorem range_of_m (m : ℝ) :
  (∀ x_1 ∈ Icc (-1 : ℝ) 2, ∃ x_2 ∈ Icc (0 : ℝ) 3, (3 - m * 3^x_1) / 3^x_1 ≥ log 2 (x_2^2 + x_2 + 2))
  ↔ m ≤ -2 / 3 :=
  sorry

end range_of_m_l255_255244


namespace workshop_production_profit_l255_255970

theorem workshop_production_profit :
  ∀ (x : ℕ), 0 ≤ x ∧ x ≤ 20 →
  (y = 150 * 6 * x + 260 * 5 * (20 - x)) →
  y ≥ 24000 →
  20 - x ≥ 15 :=
by
  intros x hx y_eq y_ge_24000
  have h1 : y = -400 * x + 26000 := by sorry
  rw h1 at y_ge_24000
  exact sorry

end workshop_production_profit_l255_255970


namespace GCF_of_60_and_72_correct_l255_255893

-- Factoring the numbers
def factors_60 : Multiset ℕ := {2, 2, 3, 5}
def factors_72 : Multiset ℕ := {2, 2, 2, 3, 3}

-- Calculate the GCF based on their factorizations
def GCF (a b : ℕ) : ℕ :=
  let fa := factors_60
  let fb := factors_72
  let common_factors := fa.filter (λ x, fb.count x > 0)
  common_factors.prod

theorem GCF_of_60_and_72_correct : GCF 60 72 = 12 :=
sorry

end GCF_of_60_and_72_correct_l255_255893


namespace simplify_and_evaluate_l255_255439

noncomputable def simplifyExpression (a : ℚ) : ℚ :=
  (a - 3 + (1 / (a - 1))) / ((a^2 - 4) / (a^2 + 2*a)) * (1 / (a - 2))

theorem simplify_and_evaluate
  (h : ∀ a, a ∈ [-2, -1, 0, 1, 2]) :
  ∀ a, (a - 1) ≠ 0 → a ≠ 0 → a ≠ 2  →
  simplifyExpression a = a / (a - 1) ∧ simplifyExpression (-1) = 1 / 2 :=
by
  intro a ha_ne_zero ha_ne_two
  sorry

end simplify_and_evaluate_l255_255439


namespace largest_possible_integer_in_list_l255_255946

noncomputable def list_of_five_integers := { l : List ℕ // l.length = 5 ∧ Multiset.card (Multiset.filter (λ x => x = 7) (l.to_multiset)) > 1 }

noncomputable def meets_conditions (l : list ℕ) : Prop :=
  l.Nth 2 = 10 ∧ (l.sum = 60)

theorem largest_possible_integer_in_list (l : List ℕ) (h : list_of_five_integers l ∧ meets_conditions l) :
  ∃ x ∈ l, x = 25 :=
sorry

end largest_possible_integer_in_list_l255_255946


namespace parabola_x_coordinate_l255_255026

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p, 0)

theorem parabola_x_coordinate
  (M : ℝ × ℝ)
  (h_parabola : (M.2)^2 = 4 * M.1)
  (h_distance : dist M (parabola_focus 2) = 3) :
  M.1 = 1 :=
by
  sorry

end parabola_x_coordinate_l255_255026


namespace ab_over_a_minus_b_l255_255198

theorem ab_over_a_minus_b (a b : ℝ) (h : (1 / a) - (1 / b) = 1 / 3) : (a * b) / (a - b) = -3 := by
  sorry

end ab_over_a_minus_b_l255_255198


namespace seq_fifth_powers_l255_255806

theorem seq_fifth_powers (n : ℕ) (x : ℕ → ℤ)
  (h1 : ∑ i in Finset.range n, x i = -5)
  (h2 : ∑ i in Finset.range n, (x i)^2 = 19)
  (h3 : ∀ i : ℕ, i < n → x i ∈ {0, 1, -2}) :
  ∑ i in Finset.range n, (x i)^5 = -125 :=
sorry

end seq_fifth_powers_l255_255806


namespace intersection_area_l255_255882

-- Define the square vertices
def vertex1 : (ℝ × ℝ) := (2, 8)
def vertex2 : (ℝ × ℝ) := (13, 8)
def vertex3 : (ℝ × ℝ) := (13, -3)
def vertex4 : (ℝ × ℝ) := (2, -3)  -- Derived from the conditions

-- Define the circle with center and radius
def circle_center : (ℝ × ℝ) := (2, -3)
def circle_radius : ℝ := 4

-- Define the square side length
def square_side_length : ℝ := 11  -- From vertex (2, 8) to vertex (2, -3)

-- Prove the intersection area
theorem intersection_area :
  let area := (1 / 4) * Real.pi * (circle_radius^2)
  area = 4 * Real.pi :=
by
  sorry

end intersection_area_l255_255882


namespace terminating_decimal_expansion_of_7_over_72_l255_255665

theorem terminating_decimal_expansion_of_7_over_72 : (7 / 72) = 0.175 := 
sorry

end terminating_decimal_expansion_of_7_over_72_l255_255665


namespace maximize_angle_distance_l255_255531

noncomputable def f (x : ℝ) : ℝ :=
  40 * x / (x * x + 500)

theorem maximize_angle_distance :
  ∃ x : ℝ, x = 10 * Real.sqrt 5 ∧ ∀ y : ℝ, y ≠ x → f y < f x :=
sorry

end maximize_angle_distance_l255_255531


namespace find_f_2015_l255_255206

noncomputable def f : ℤ → ℤ := sorry

axiom periodicity : ∀ x : ℤ, f(x + 6) + f(x) = 0
axiom odd_property : ∀ x : ℤ, f(-x) = -f(x)
axiom initial_value : f 1 = -2

theorem find_f_2015 : f 2015 = 2 := by
  sorry

end find_f_2015_l255_255206


namespace find_y_in_set_l255_255227

noncomputable def arithmetic_mean (s : List ℝ) : ℝ :=
  s.sum / s.length

theorem find_y_in_set :
  ∀ (y : ℝ), arithmetic_mean [8, 15, 20, 5, y] = 12 ↔ y = 12 :=
by
  intro y
  unfold arithmetic_mean
  simp [List.sum_cons, List.length_cons]
  sorry

end find_y_in_set_l255_255227


namespace faye_has_62_pieces_of_candy_l255_255670

-- Define initial conditions
def initialCandy : Nat := 47
def eatenCandy : Nat := 25
def receivedCandy : Nat := 40

-- Define the resulting number of candies after eating and receiving more candies
def resultingCandy : Nat := initialCandy - eatenCandy + receivedCandy

-- State the theorem and provide the proof
theorem faye_has_62_pieces_of_candy :
  resultingCandy = 62 :=
by
  -- proof goes here
  sorry

end faye_has_62_pieces_of_candy_l255_255670


namespace cosine_equation_solution_count_l255_255626

open Real

noncomputable def number_of_solutions : ℕ := sorry

theorem cosine_equation_solution_count :
  number_of_solutions = 2 :=
by
  -- Let x be an angle in [0, 2π].
  sorry

end cosine_equation_solution_count_l255_255626


namespace yard_length_l255_255877

theorem yard_length (n_trees : ℕ) (d : ℕ) (h_trees : n_trees = 32) (h_distance : d = 14) : 
  let n_gaps := n_trees - 1 in
  let length := n_gaps * d in
  length = 434 :=
by
  sorry

end yard_length_l255_255877


namespace average_of_first_12_l255_255850

theorem average_of_first_12 (avg25 : ℝ) (avg12 : ℝ) (avg_last12 : ℝ) (result_13th : ℝ) : 
  (avg25 = 18) → (avg_last12 = 17) → (result_13th = 78) → 
  25 * avg25 = (12 * avg12) + result_13th + (12 * avg_last12) → avg12 = 14 :=
by 
  sorry

end average_of_first_12_l255_255850


namespace perpendicular_AH_BP_l255_255141

-- Define the points A, B, C as vertices of the isosceles triangle
variables {A B C M H P : Point}

-- Conditions:
variables (isosceles_triangle : IsIsoscelesTriangle A B C)
variables (midpoint_AC : Midpoint M A C)
variables (perpendicular_MH_BC : Perpendicular MH BC)
variables (intersection_MH : Intersection H M BC)
variables (midpoint_MH : Midpoint P M H)

-- Statement to prove:
theorem perpendicular_AH_BP :
  Perpendicular AH BP :=
sorry

end perpendicular_AH_BP_l255_255141


namespace min_odd_in_A_P_l255_255363

-- Define the polynomial P of degree 8
variable (P : ℝ → ℝ)
-- Assume P is a polynomial of degree 8
axiom degree_P : degree P = 8

-- Define the set A_P as the set of all x for which P(x) gives a certain value
def A_P (c : ℝ) : Set ℝ := { x | P x = c }

-- Given condition
axiom include_8 : 8 ∈ A_P (P 8)

-- Define if a number is odd
def is_odd (n : ℝ) : Prop := ∃k : ℤ, n = 2 * k + 1

-- The minimum number of odd numbers that are in the set A_P,
-- given that the number 8 is included in A_P, is 1

theorem min_odd_in_A_P : ∀ P, (degree P = 8) → 8 ∈ A_P (P 8) → ∃ x ∈ A_P (P 8), is_odd x := 
by
  sorry -- proof goes here

end min_odd_in_A_P_l255_255363


namespace area_of_backyard_l255_255056

theorem area_of_backyard (length_of_backyard perimeter_of_backyard : ℕ) (L_times_50_eq_2000 : length_of_backyard * 50 = 2000) (P_times_20_eq_2000 : perimeter_of_backyard * 20 = 2000) :
  (L_times_50_eq_2000 ∧ P_times_20_eq_2000 → (let L := length_of_backyard, P := perimeter_of_backyard, W := (P - 2 * L) / 2 in L * W = 400)) :=
begin
  sorry
end

end area_of_backyard_l255_255056


namespace min_m_value_l255_255282

noncomputable def f (x a : ℝ) : ℝ := 2 ^ (abs (x - a))

theorem min_m_value :
  ∀ a, (∀ x, f (1 + x) a = f (1 - x) a) →
  ∃ m : ℝ, (∀ x : ℝ, x ≥ m → ∀ y : ℝ, y ≥ x → f y a ≥ f x a) ∧ m = 1 :=
by
  intros a h
  sorry

end min_m_value_l255_255282


namespace inequality_solution_l255_255675

theorem inequality_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 6) : x^3 - 12 * x^2 + 36 * x > 0 :=
sorry

end inequality_solution_l255_255675


namespace find_two_numbers_l255_255061

noncomputable def quadratic_roots (a b : ℝ) : Prop :=
  a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2

theorem find_two_numbers (a b : ℝ) (h1 : a * b = 5) (h2 : 2 * (a * b) / (a + b) = 5 / 2) :
  quadratic_roots a b :=
by
  sorry

end find_two_numbers_l255_255061


namespace permissible_range_n_l255_255218

theorem permissible_range_n (n x y m : ℝ) (hn : n ≤ x) (hxy : x < y) (hy : y ≤ n+1)
  (hm_in: x < m ∧ m < y) (habs_eq : |y| = |m| + |x|): 
  -1 < n ∧ n < 1 := sorry

end permissible_range_n_l255_255218


namespace sec_150_eq_neg_2_sqrt_3_div_3_l255_255647

theorem sec_150_eq_neg_2_sqrt_3_div_3 : Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l255_255647


namespace hyperbola_eccentricity_proof_l255_255423

open Real

def hyperbola_eccentricity (a b : ℝ) (h : a > 0) (h : b > 0) : ℝ :=
  sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_proof (a b : ℝ) (h : a > 0) (h : b > 0)
  (P : ℝ × ℝ)
  (hyp : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (F1 F2 : ℝ × ℝ)
  (focus_def : (F1.1 = -c) ∧ (F2.1 = c) ∧ (F1.2 = 0) ∧ (F2.2 = 0))
  (midpt_bisector : dist ((P.1 + F1.1) / 2, (P.2 + F1.2) / 2) (F2.1, F2.2) = 0)
  (distance_from_origin : dist (0, 0) (F1.1, F1.2) = a) :
  hyperbola_eccentricity a b = 5 / 3 :=
sorry

end hyperbola_eccentricity_proof_l255_255423


namespace theresa_more_than_thrice_julia_l255_255052

-- Define the problem parameters
variable (tory julia theresa : ℕ)

def tory_videogames : ℕ := 6
def theresa_videogames : ℕ := 11

-- Define the relationships between the numbers of video games
def julia_relationship := julia = tory / 3
def theresa_compared_to_julia := theresa = theresa_videogames
def tory_value := tory = tory_videogames

theorem theresa_more_than_thrice_julia (h1 : julia_relationship tory julia) 
                                       (h2 : tory_value tory)
                                       (h3 : theresa_compared_to_julia theresa) :
  theresa - 3 * julia = 5 :=
by 
  -- Here comes the proof (not required for the task)
  sorry

end theresa_more_than_thrice_julia_l255_255052


namespace central_cell_value_l255_255168

theorem central_cell_value (table : Fin 29 → Fin 29 → Fin 29) 
  (h1 : ∀ n : Fin 29, ∃! (i j : Fin 29), table i j = n)
  (h2 : ∃S, (∑ i j, if i ≤ j then 0 else table i j) = S ∧ (∑ i j, if i ≥ j then 0 else table i j) = 3 * S)
  : table ⟨14, _⟩ ⟨14, _⟩ = 15 :=
begin
  sorry
end

end central_cell_value_l255_255168


namespace quadratic_completing_square_solution_l255_255840

theorem quadratic_completing_square_solution (a : ℝ) (h : a < 0) :
  (∃ n : ℝ, (x : ℝ) → x^2 + a * x + 1/4 = (x + n)^2 + 1/16) →
  a = - (sqrt 3 / 2) :=
by
  intro h_ex
  sorry

end quadratic_completing_square_solution_l255_255840


namespace partition_set_iff_multiple_of_3_l255_255328

open Nat

theorem partition_set_iff_multiple_of_3 (n : ℕ) (S : Set ℕ) (hS : S = {i | i ∈ range (n+1)} \ {0}) (hn : n ≥ 6) :
  (∃ T U V : Set ℕ, (T ∩ U = ∅) ∧ (U ∩ V = ∅) ∧ (V ∩ T = ∅) ∧ (T ∪ U ∪ V = S) ∧
  (∃ k : ℕ, T.card = k ∧ U.card = k ∧ V.card = k) ∧
  (∃ sum_val : ℕ, T.sum = sum_val ∧ U.sum = sum_val ∧ V.sum = sum_val))
  ↔ (∃ m : ℕ, n = 3 * m) :=
by
  sorry

end partition_set_iff_multiple_of_3_l255_255328


namespace sequence_sum_l255_255260

theorem sequence_sum (n : ℕ) : 
  let a : ℕ → ℝ := λ k, (2 * k - 1 : ℕ) + (1 / (2:ℝ) ^ k)
  let S : ℕ → ℝ := λ n, ∑ k in Finset.range (n + 1), a k
  in S n = n^2 + 1 - (1 / (2:ℝ) ^ n) :=
by
  sorry

end sequence_sum_l255_255260


namespace part1_part2_l255_255924

-- Part (1)
theorem part1 (a b c d e : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) (h5 : e < 0) : 
  (e / (a - c)) > (e / (b - d)) := 
sorry

-- Part (2)
theorem part2 (a b : ℝ) (h1 : a > 1) (h2 : b > 1) : 
  let M := (a^2 / (a - 1)) + (b^2 / (b - 1))
  let N := (b^2 / (a - 1)) + (a^2 / (b - 1))
  in M ≤ N :=
sorry

end part1_part2_l255_255924


namespace beta_max_two_day_ratio_l255_255135

noncomputable def alpha_first_day_score : ℚ := 160 / 300
noncomputable def alpha_second_day_score : ℚ := 140 / 200
noncomputable def alpha_two_day_ratio : ℚ := 300 / 500

theorem beta_max_two_day_ratio :
  ∃ (p q r : ℕ), 
  p < 300 ∧
  q < (8 * p / 15) ∧
  r < ((3500 - 7 * p) / 10) ∧
  q + r = 299 ∧
  gcd 299 500 = 1 ∧
  (299 + 500) = 799 := 
sorry

end beta_max_two_day_ratio_l255_255135


namespace abs_neg_three_l255_255988

theorem abs_neg_three : abs (-3) = 3 := 
by
  sorry

end abs_neg_three_l255_255988


namespace undefined_expression_l255_255629

theorem undefined_expression (y : ℝ) : (y^2 - 16 * y + 64 = 0) ↔ (y = 8) := by
  sorry

end undefined_expression_l255_255629


namespace sasha_can_determine_X_l255_255159

theorem sasha_can_determine_X :
  ∀ (X : ℕ), X ≤ 100 → ∃ (M N : ℕ → ℕ),
  (∀ i, 1 ≤ i ∧ i ≤ 7 → M i < 100 ∧ N i < 100) ∧ (
  ∀ G : ℕ → ℕ, (∀ i, 1 ≤ i ∧ i ≤ 7 → G i = Nat.gcd (X + M i) (N i)) →
  ∃ x, x = X) :=
begin
  sorry
end

end sasha_can_determine_X_l255_255159


namespace impossibility_of_8_points_l255_255878

theorem impossibility_of_8_points
  (n : ℕ)
  (plays_match : ℕ)
  (win_pts lose_pts draw_pts max_pts : ℕ)
  (number_of_teams : ℕ)
  (points_per_win : ℕ)
  (points_per_loss : ℕ)
  (points_per_draw : ℕ)
  (maximum_points : ℕ)
  : number_of_teams = 4 →
    points_per_win = 3 →
    points_per_loss = 0 →
    points_per_draw = 1 →
    plays_match = number_of_teams - 1 →
    max_pts = plays_match * points_per_win →
    ¬(exists p : ℕ, p = 8 ∧ (
      (exists (w d l : ℕ), w + d + l = plays_match ∧
      w * points_per_win + d * points_per_draw + l * points_per_loss = p)))
  :=
  begin
    sorry
  end

end impossibility_of_8_points_l255_255878


namespace grace_reading_time_l255_255265

theorem grace_reading_time 
  (reads_200_pages_in_20_hours : ∀ (P T : ℕ), P = 200 → T = 20 → true)
  (constant_reading_speed : ∀ (P1 T1 P2 T2 : ℕ), (P1 / T1) = (P2 / T2)) :
  Grace (P T : ℕ) → P = 250 → T = 25 := sorry

end grace_reading_time_l255_255265


namespace find_integer_x_l255_255082

theorem find_integer_x (x : ℕ) (pos_x : 0 < x) (ineq : x + 1000 > 1000 * x) : x = 1 :=
sorry

end find_integer_x_l255_255082


namespace minimum_odd_numbers_in_set_l255_255391

-- Definitions
variable (P : ℝ → ℝ)
variable (degree_P : ℕ)
variable (A_P : Set ℝ)

-- The conditions: P is a polynomial of degree 8, and 8 is included in A_P
def is_polynomial_of_degree_eight (P : ℝ → ℝ) (degree_P : ℕ) : Prop :=
  degree_P = 8

def set_includes_eight (A_P : Set ℝ) : Prop := 
  8 ∈ A_P

-- The goal: prove the minimum number of odd numbers in A_P is 1
theorem minimum_odd_numbers_in_set {P : ℝ → ℝ} {degree_P : ℕ} {A_P : Set ℝ} :
  is_polynomial_of_degree_eight P degree_P → 
  set_includes_eight A_P → 
  ∃ odd_numbers : ℕ, odd_numbers = 1 :=
sorry

end minimum_odd_numbers_in_set_l255_255391


namespace total_birds_count_l255_255969

def blackbirds_per_tree : ℕ := 3
def tree_count : ℕ := 7
def magpies : ℕ := 13

theorem total_birds_count : (blackbirds_per_tree * tree_count) + magpies = 34 := by
  sorry

end total_birds_count_l255_255969


namespace path_problem_l255_255345

noncomputable def path_bounds (N : ℕ) (h : 0 < N) : Prop :=
  ∃ p : ℕ, 4 * N ≤ p ∧ p ≤ 2 * N^2 + 2 * N

theorem path_problem (N : ℕ) (h : 0 < N) : path_bounds N h :=
  sorry

end path_problem_l255_255345


namespace sqrt_squared_eq_original_l255_255615

theorem sqrt_squared_eq_original (m : ℕ) (h : m = 529441) : (Real.sqrt (m : ℝ))^2 = m :=
by
  rw [h]
  exact Real.sqrt_sq (by norm_num : m = 529441)

end sqrt_squared_eq_original_l255_255615


namespace ratio_problem_l255_255287

theorem ratio_problem (a b c d : ℚ) (h1 : a / b = 5 / 4) (h2 : c / d = 4 / 1) (h3 : d / b = 1 / 8) :
  a / c = 5 / 2 := by
  sorry

end ratio_problem_l255_255287


namespace collinear_points_k_value_l255_255865

theorem collinear_points_k_value (k : ℝ) :
  ∃ (k : ℝ), (k = 39) :=
begin
  have m := (4 - -2) / (3 - 1),
  have line_eq := λ x, m * x + ( -2 - m * 1),
  have y := line_eq 6,
  have h₁ : y = 13, from
    calc y = 3 * 6 + -5  : by rw [line_eq]
       ...  = 13         : by linarith,
  have h₂ : k/3 = 13, from
    calc k/3 = y : by rw [line_eq]
       ...  = 13 : by exact h₁,
  have h₃ : k = 39, from
    calc k = 13 * 3 : by exact h₂
       ...  = 39    : by linarith,
  use h₃,
end

end collinear_points_k_value_l255_255865


namespace molecular_weight_calculation_l255_255071

-- Define the condition given in the problem
def molecular_weight_of_4_moles := 488 -- molecular weight of 4 moles in g/mol

-- Define the number of moles
def number_of_moles := 4

-- Define the expected molecular weight of 1 mole
def expected_molecular_weight_of_1_mole := 122 -- molecular weight of 1 mole in g/mol

-- Theorem statement
theorem molecular_weight_calculation : 
  molecular_weight_of_4_moles / number_of_moles = expected_molecular_weight_of_1_mole := 
by
  sorry

end molecular_weight_calculation_l255_255071


namespace cross_section_area_l255_255021

theorem cross_section_area (V : ℝ) (α : ℝ) (α_pos : 0 < α ∧ α < Real.pi / 2) :
  ∃ S : ℝ, S = Real.cbrt (3 * Real.sqrt 3 * V^2 / (Real.sin α)^2 / Real.cos α) :=
by
  use Real.cbrt (3 * Real.sqrt 3 * V^2 / (Real.sin α)^2 / Real.cos α)
  exact sorry

end cross_section_area_l255_255021


namespace isosceles_triangle_angles_l255_255022

theorem isosceles_triangle_angles {α : ℝ} (hα : α = 18) :
  ∀ (A B C M D : Point) {t : Triangle},
  is_isosceles t A B C →
  is_bisector A M (∠BAC) →
  is_bisector B D (∠ABC) →
  length A M = 1/2 * length B D →
  angles_of_triangle t = (36, 36, 108) :=
begin
  intros,
  sorry
end

end isosceles_triangle_angles_l255_255022


namespace angle_IKH_eq_angle_INH_l255_255764

open EuclideanGeometry

theorem angle_IKH_eq_angle_INH 
  (A B C : Point)
  (H I : Point)
  (O : Circle)
  (M K N : Point)
  (h1 : IsAcuteTriangle A B C)
  (h2 : IsOrthocenter H A B C)
  (h3 : IsIncenter I A B C)
  (h4 : Circumcircle O A B C)
  (h5 : IsMidpointOfArc M O A B C)
  (h6 : OnCircle K O)
  (h7 : ∠AKH = 90)
  (h8 : LineIntersectAt AH MI N)
  (h9 : OnCircle N O) :
  ∠IKH = ∠INH :=
by
  sorry

end angle_IKH_eq_angle_INH_l255_255764


namespace trajectory_hyperbola_l255_255220

def F1 : (ℝ × ℝ) := (-5, 0)
def F2 : (ℝ × ℝ) := (5, 0)
def P : (ℝ × ℝ) := sorry  -- Point P is not fixed, it's a variable.

noncomputable def distance (x y : ℝ × ℝ) : ℝ := ((x.1 - y.1)^2 + (x.2 - y.2)^2).sqrt

noncomputable def hyperbola_cond (a : ℝ) (P : ℝ × ℝ) :=
  abs (distance P F1 - distance P F2) = 2 * a

theorem trajectory_hyperbola : hyperbola_cond 3 P ∧ hyperbola_cond 5 P →
  (exists b : ℝ, (b > 0 ∧ hyperbola_cond b P)) :=
begin
  sorry
end

end trajectory_hyperbola_l255_255220


namespace theta_arithmetic_sequence_condition_l255_255993

noncomputable def is_arithmetic_sequence (a : ℕ → ℂ) : Prop :=
∃ (c : ℂ), ∀ n : ℕ, a n - a (n - 1) = c

theorem theta_arithmetic_sequence_condition (θ : ℝ) :
  (∀ n : ℕ, cos (n * θ) + complex.i * sin (n * θ) - (cos ((n - 1) * θ) + complex.i * sin ((n - 1) * θ)) = 
     cos θ + complex.i * sin θ) ↔ ∃ k : ℤ, θ = 2 * k * real.pi := 
sorry

end theta_arithmetic_sequence_condition_l255_255993


namespace cost_decrease_l255_255140

variable (a : ℝ) (h_pos : a > 0)

theorem cost_decrease (h : a * 0.9216 = a * (0.96)^2) : a * 0.9216 < a :=
by
  have h₀ : 0.9216 = (0.96)^2 := by norm_num
  rw [h₀] at h
  have h₁ : (0.96)^2 < 1 := by norm_num
  calc a * (0.96)^2 = a * 0.9216 : by rw h₀
                 ... < a * 1     : by nlinarith [h1, h_pos]
                 ... = a         : by rw mul_one

end cost_decrease_l255_255140


namespace range_of_x_l255_255314

theorem range_of_x (x : ℝ) : (sqrt ((x - 1) / (x - 2)) : ℝ) ≥ 0 → (x > 2 ∨ x ≤ 1) :=
by
  sorry

end range_of_x_l255_255314


namespace trajectory_is_ellipse_max_area_triangle_l255_255689

noncomputable theory
open_locale classical

structure Point :=
(x : ℝ)
(y : ℝ)

def distance (p q : Point) : ℝ :=
(real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2))

def trajectory_eq (M : Point) : Prop :=
(distance M ⟨1, 0⟩ / real.abs (4 - M.x) = 1/2)

def ellipse_eq (M : Point) : Prop :=
(M.x^2 / 4 + M.y^2 / 3 = 1)

theorem trajectory_is_ellipse (M : Point) : 
  trajectory_eq M → ellipse_eq M :=
by sorry

def line_eq (m : ℝ) (y : ℝ) : ℝ := 
  m * y + 1

def intersects_ellipse (m : ℝ) (M : Point) : Prop :=
let y := M.y in M.x = line_eq m y ∧ ellipse_eq M

def area_triangle (F1 A B : Point) : ℝ :=
0.5 * distance F1 A * real.abs (A.y - B.y)

theorem max_area_triangle (A B F1 F2 : Point) (m : ℝ) :
  intersects_ellipse m A → intersects_ellipse m B → 
  F1.x = -1 ∧ F1.y = 0 ∧ F2.x = 1 ∧ F2.y = 0 →
  area_triangle F1 A B ≤ 3 :=
by sorry

end trajectory_is_ellipse_max_area_triangle_l255_255689


namespace length_of_AB_l255_255741

-- Definition of the parabola and its properties
def parabola (m : ℤ) : ℝ → ℝ := λ x => x^2 - (m : ℝ) * x - 3

-- Definition of the quadratic equation resulting from setting y = 0 in the parabola
def quadratic_eq (m : ℤ) (x : ℝ) : Prop := x^2 - (m : ℝ) * x - 3 = 0

-- The theorem stating the length of AB is 4
theorem length_of_AB (m : ℤ) (A B : ℝ) (hA : quadratic_eq m A) (hB : quadratic_eq m B) (hA_ne_B : A ≠ B) : 
  (B - A).abs = 4 := 
  sorry 

end length_of_AB_l255_255741


namespace minimum_odd_in_A_P_l255_255374

-- Define the polynomial and its properties
def polynomial_degree8 (P : ℝ[X]) : Prop :=
  P.degree = 8

-- Define the set A_P
def A_P (P : ℝ[X]) : set ℝ :=
  {x | P.eval x = 8}

-- Main theorem statement
theorem minimum_odd_in_A_P (P : ℝ[X]) (hP : polynomial_degree8 P) : ∃ n : ℕ, n = 1 ∧ ∃ x ∈ A_P P, odd x :=
sorry

end minimum_odd_in_A_P_l255_255374


namespace probability_same_color_l255_255483

theorem probability_same_color (g r b : ℕ) (total_balls : ℕ) (h_g : g = 8) (h_r : r = 6) (h_b : b = 4)
  (h_total_balls : g + r + b = 18) : 
  ((g / (g + r + b)) ^ 2 + (r / (g + r + b)) ^ 2 + (b / (g + r + b)) ^ 2 = 29 / 81) :=
by
  rw [h_g, h_r, h_b, h_total_balls]
  norm_num
  sorry

end probability_same_color_l255_255483


namespace minimum_odd_numbers_in_A_P_l255_255370

-- Polynomials and assumptions
def degree (P : Polynomial ℝ) : ℕ := P.natDegree

-- The set A_P is defined as the set of all numbers x for which P(x) gives a certain value
def A_P (P : Polynomial ℝ) : Set ℝ := {x : ℝ | P.eval x = P.eval 8}

-- Define the main theorem statement
theorem minimum_odd_numbers_in_A_P (P : Polynomial ℝ) (hdeg : degree P = 8) (h8 : 8 ∈ A_P P) : 
  ∃ n, n = 1 ∧ ∃ (x : ℝ), x ∈ A_P P ∧ ¬ (x % 2 = 0) := sorry

end minimum_odd_numbers_in_A_P_l255_255370


namespace Megs_earnings_correct_l255_255266

-- Define the hours worked on different days
def Monday_hours : ℝ := 1 + (3 / 4) -- 1.75 hours
def Wednesday_hours : ℝ := 1 + (15 / 60) -- 1.25 hours
def Thursday_hours : ℝ := 2 + (15 / 60) -- 2.25 hours (9:15 to 11:30)
def Saturday_hours : ℝ := 45 / 60 -- 0.75 hours

-- Define hourly rate
def hourly_rate : ℝ := 4 -- $4 per hour

-- Define the total hours worked during the week
def total_hours_worked : ℝ := Monday_hours + Wednesday_hours + Thursday_hours + Saturday_hours

-- Define the total earnings
def total_earnings : ℝ := total_hours_worked * hourly_rate

-- The theorem to prove
theorem Megs_earnings_correct : total_earnings = 24 :=
by
  -- Sorry - The proof is omitted
  sorry

end Megs_earnings_correct_l255_255266


namespace part_a_part_b_l255_255723

variable {p q n : ℕ}

-- Conditions
def coprime (a b : ℕ) : Prop := gcd a b = 1
def differ_by_more_than_one (p q : ℕ) : Prop := (q > p + 1) ∨ (p > q + 1)

-- Part (a): Prove there exists a natural number n such that p + n and q + n are not coprime
theorem part_a (coprime_pq : coprime p q) (diff : differ_by_more_than_one p q) : 
  ∃ n : ℕ, ¬ coprime (p + n) (q + n) :=
sorry

-- Part (b): Prove the smallest such n is 41 for p = 2 and q = 2023
theorem part_b (h : p = 2) (h1 : q = 2023) : 
  ∃ n : ℕ, (n = 41) ∧ (¬ coprime (2 + n) (2023 + n)) :=
sorry

end part_a_part_b_l255_255723


namespace volume_maximized_at_r_5_h_8_l255_255131

noncomputable def V (r : ℝ) : ℝ := (Real.pi / 5) * (300 * r - 4 * r^3)

/-- (1) Given that the total construction cost is 12000π yuan, 
express the volume V as a function of the radius r, and determine its domain. -/
def volume_function (r : ℝ) (h : ℝ) (cost : ℝ) : Prop :=
  cost = 12000 * Real.pi ∧
  h = 1 / (5 * r) * (300 - 4 * r^2) ∧
  V r = Real.pi * r^2 * h ∧
  0 < r ∧ r < 5 * Real.sqrt 3

/-- (2) Prove V(r) is maximized when r = 5 and h = 8 -/
theorem volume_maximized_at_r_5_h_8 :
  ∀ (r : ℝ) (h : ℝ) (cost : ℝ), volume_function r h cost → 
  ∃ (r_max : ℝ) (h_max : ℝ), r_max = 5 ∧ h_max = 8 ∧ ∀ x, 0 < x → x < 5 * Real.sqrt 3 → V x ≤ V r_max :=
by
  intros r h cost hvolfunc
  sorry

end volume_maximized_at_r_5_h_8_l255_255131


namespace problem_statement_l255_255687

section math_proof

-- Define the circle equation and the fixed point
def circle_eq (x y : ℝ) : Prop := x^2 + 4 * x + y^2 - 32 = 0
def Q := (6 : ℝ, 0 : ℝ)

-- Define the midpoint M
def is_midpoint (P Q M : ℝ × ℝ) : Prop :=
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the locus equation for M
def locus_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the line equation l passing through a point
def line_eq (k : ℝ) (x y : ℝ) : Prop := y = k * x - 3
def point_eq (x y : ℝ) : Prop := (x = 0) ∧ (y = -3)

-- Main theorem proving the conditions and answers
theorem problem_statement :
  (∀ (P : ℝ × ℝ), circle_eq P.1 P.2 → ∃ M : ℝ × ℝ, is_midpoint P Q M ∧ locus_eq M.1 M.2) ∧
  (∀ (k : ℝ) (A B : ℝ × ℝ), (point_eq 0 (-3)) ∧ 
   line_eq k A.1 A.2 ∧ line_eq k B.1 B.2 ∧ locus_eq A.1 A.2 ∧ locus_eq B.1 B.2 ∧ 
   ((A.1 / B.1) + (B.1 / A.1) = 21 / 2) → 
  (line_eq 1 0 (-3) ∨ line_eq (17 / 7) 0 (-3))) :=
begin
  sorry
end

end math_proof

end problem_statement_l255_255687


namespace arrangement_count_l255_255435

theorem arrangement_count (h : Finset ℕ) (ht: h.card = 7) :
  let tallest := h.max
  \exists f : Finset ℕ, h.erase tallest = f \and 
    (f.card = 6) \and ∃ l r : finset ℕ, (l.card = 3 \and f = l ∪ r ∪ tallest) 
    ∧ (∃ ! m ∈ l, m < tallest) ∧ (∃ ! n ∈ r, n < tallest) ∧
    (∀ a b ∈ l, a < b → (position l b < position l a)) ∧ 
    (∀ c d ∈ r, c \neq d → (position r d < position r c)) :=
  20 :=
sorry

end arrangement_count_l255_255435


namespace expected_no_advice_l255_255587

theorem expected_no_advice (n : ℕ) (p : ℝ) (h_p : 0 ≤ p ∧ p ≤ 1) : 
  (∑ j in finset.range n, (1 - p)^j) = (1 - (1 - p)^n) / p :=
by
  sorry

end expected_no_advice_l255_255587


namespace activity_popularity_order_l255_255584

theorem activity_popularity_order :
  let dodgeball := 13 / 40
  let movie_screening := 9 / 25
  let quiz_bowl := 7 / 20
  let gardening := 6 / 15
  min gardening (min movie_screening (min quiz_bowl dodgeball)) == gardening ∧
  min dodgeball (min movie_screening (min quiz_bowl gardening)) == dodgeball ∧
  min quiz_bowl (min gardening (min movie_screening dodgeball)) == quiz_bowl ∧
  min movie_screening (min gardening (min quiz_bowl dodgeball)) == movie_screening :=
by
  sorry

end activity_popularity_order_l255_255584


namespace smallest_sum_of_15_consecutive_positive_integers_is_375_l255_255875

noncomputable def sum_of_15_consecutive_integers_is_perfect_square
  (m : ℕ) : Prop :=
  let sum := 15 * (m + 7) in
  (∃ k : ℕ, sum = k * k)

theorem smallest_sum_of_15_consecutive_positive_integers_is_375 :
  ∃ m : ℕ, m > 0 ∧ sum_of_15_consecutive_integers_is_perfect_square m ∧ 15 * (m + 7) = 375 :=
by
  sorry

end smallest_sum_of_15_consecutive_positive_integers_is_375_l255_255875


namespace total_profit_4650_l255_255736

-- Conditions
variable {P Q R : ℝ}
variable (h1 : 4 * P = 6 * Q)
variable (h2 : 6 * Q = 10 * R)
variable (h3 : R_share = 900)

theorem total_profit_4650 (h1 : 4 * P = 6 * Q) (h2 : 6 * Q = 10 * R) (h3 : R = 900 / 6) : 
  let total_profit := 31 * (R / 6) * 6 in total_profit = 4650 :=
by
  sorry

end total_profit_4650_l255_255736


namespace prudence_total_sleep_l255_255297

def total_sleep_week (sleep_night: ℕ) (fri_sleep: ℕ) (sat_sleep: ℕ) (sat_nap: ℕ) (sun_nap: ℕ) (tue_nap: ℕ) 
  (thu_nap: ℕ) (wed_interruption: ℕ) (mon_interruption: ℕ := 0) (yoga_interruption: ℕ := 0) 
  (project_interruption: ℕ := 0) : ℕ :=
  6 * 5 + fri_sleep + sat_sleep + sat_nap + sun_nap + tue_nap + thu_nap - wed_interruption - 
  mon_interruption - yoga_interruption - project_interruption

def total_sleep_4_weeks : ℕ :=
  total_sleep_week 6 9 9 1 1 (1 / 2) (1 / 2) 1 +
  total_sleep_week 6 7 8 1 1 (1 / 2) (1 / 2) 0.5 +
  total_sleep_week 6 9 9 1 1 (1 / 2) (1 / 2) 0 0.5 +
  total_sleep_week 6 7 10 1 1 (1 / 2) (1 / 2) 0 0 1

theorem prudence_total_sleep : total_sleep_4_weeks = 197 := by
  sorry

end prudence_total_sleep_l255_255297


namespace total_birds_correct_l255_255966

-- Define the conditions
def number_of_trees : ℕ := 7
def blackbirds_per_tree : ℕ := 3
def magpies : ℕ := 13

-- Define the total number of blackbirds using the conditions
def total_blackbirds : ℕ := number_of_trees * blackbirds_per_tree

-- Define the total number of birds using the total number of blackbirds and the number of magpies
def total_birds : ℕ := total_blackbirds + magpies

-- The theorem statement that should be proven
theorem total_birds_correct : total_birds = 34 := 
sorry

end total_birds_correct_l255_255966


namespace bulgarian_inequality_l255_255090

theorem bulgarian_inequality (a b c d : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d) :
    (a^4 / (a^3 + a^2 * b + a * b^2 + b^3) + 
     b^4 / (b^3 + b^2 * c + b * c^2 + c^3) + 
     c^4 / (c^3 + c^2 * d + c * d^2 + d^3) + 
     d^4 / (d^3 + d^2 * a + d * a^2 + a^3)) 
    ≥ (a + b + c + d) / 4 :=
sorry

end bulgarian_inequality_l255_255090


namespace work_efficiency_ratio_l255_255080

theorem work_efficiency_ratio (A B : ℝ) 
  (h1 : B = 1 / 18) 
  (h2 : A + B = 1 / 6) : 
  A / B = 2 :=
by
  have hA : A = 1 / 6 - 1 / 18 := by sorry
  rw [h1] at h2
  have hA_eq : A = 1 / 9 := by sorry
  rw [hA_eq, h1]
  have hratio : A / B = (1 / 9) / (1 / 18) := by sorry
  rw [hratio]
  apply sorry

end work_efficiency_ratio_l255_255080


namespace smallest_m_exists_l255_255795

def in_T (z : ℂ) : Prop :=
  (∃ (x y : ℝ), z = x + y * complex.I ∧ (real.sqrt 3) / 2 ≤ x ∧ x ≤ 2 / (real.sqrt 3))

theorem smallest_m_exists :
  ∃ m : ℕ, m = 12 ∧ (∀ n : ℕ, n ≥ m → (∃ z : ℂ, in_T z ∧ z^n = 1)) :=
sorry

end smallest_m_exists_l255_255795


namespace problem_5a_5b_2_6cd_7m_eq_l255_255699

variable (a b c d m : ℝ)

theorem problem_5a_5b_2_6cd_7m_eq :
  a + b = 0 → 
  c * d = 1 → 
  |m| = 4 → 
  (5 * a + 5 * b - 2 + 6 * c * d - 7 * m = -24 ∨ 5 * a + 5 * b - 2 + 6 * c * d - 7 * m = 32) :=
by 
  intros h1 h2 h3
  have h4 : 0 := by sorry
  have h5 : 1 := by sorry
  have h6 : 4 := by sorry
  sorry

end problem_5a_5b_2_6cd_7m_eq_l255_255699


namespace positive_integer_pairs_l255_255650

theorem positive_integer_pairs (m n : ℕ) (p : ℕ) (hp_prime : Prime p) (h_diff : m - n = p) (h_square : ∃ k : ℕ, m * n = k^2) :
  ∃ p' : ℕ, (Prime p') ∧ m = (p' + 1) / 2 ^ 2 ∧ n = (p' - 1) / 2 ^ 2 :=
sorry

end positive_integer_pairs_l255_255650


namespace rope_length_l255_255777

theorem rope_length (initial_length : ℝ) (h1: initial_length = 100) :
  let first_cut := initial_length / 2 in
  let second_cut := first_cut / 2 in
  let final_piece := second_cut / 5 in
  final_piece = 5 :=
by
  sorry

end rope_length_l255_255777


namespace yo_yos_collected_l255_255781

-- Define the given conditions
def stuffed_animals : ℕ := 14
def frisbees : ℕ := 18
def total_prizes : ℕ := 50

-- Define the problem to prove that the number of yo-yos is 18
theorem yo_yos_collected : (total_prizes - (stuffed_animals + frisbees) = 18) :=
by
  sorry

end yo_yos_collected_l255_255781


namespace compare_a_b_c_l255_255682

noncomputable def a : ℝ := Real.log 0.2
noncomputable def b : ℝ := 2^0.3
noncomputable def c : ℝ := 0.3^0.2

theorem compare_a_b_c : a < c ∧ c < b := by
  sorry

end compare_a_b_c_l255_255682


namespace number_of_possible_meals_l255_255603

def entrees : Set String := {"Pizza", "Chicken Teriyaki", "Corn Dog", "Fish and Chips"}
def drinks : Set String := {"Lemonade", "Root Beer", "Cola"}
def desserts : Set String := {"Frozen Yogurt", "Ice Cream"}

def constraint (entree : String) (drink : String) : Prop := 
  entree = "Fish and Chips" → drink ≠ "Lemonade"

def total_possible_meals : Nat :=
  (entrees.erase "Fish and Chips").card * drinks.card * desserts.card +
  2 * desserts.card -- 2 valid drinks for "Fish and Chips" and 2 desserts

theorem number_of_possible_meals : total_possible_meals = 22 := by
  sorry

end number_of_possible_meals_l255_255603


namespace num_of_subsets_is_16_l255_255148

open Set
open Finset

noncomputable def numValidSubsets : ℕ :=
  let S := {1, 2, 3, 4, 5, 6} in
  (powerset S.filter (λ x, x ≠ 1 ∧ x ≠ 2)).card

theorem num_of_subsets_is_16 : numValidSubsets = 16 := by
  sorry

end num_of_subsets_is_16_l255_255148


namespace min_colors_required_l255_255504

-- Defining the color type
def Color := ℕ

-- Defining a 6x6 grid
def Grid := Fin 6 → Fin 6 → Color

-- Defining the conditions of the problem for a valid coloring
def is_valid_coloring (c : Grid) : Prop :=
  (∀ i j k, i ≠ j → c i k ≠ c j k) ∧ -- each row has all cells with different colors
  (∀ i j k, i ≠ j → c k i ≠ c k j) ∧ -- each column has all cells with different colors
  (∀ i j, i ≠ j → c i (i+j) ≠ c j (i+j)) ∧ -- each 45° diagonal has all different colors
  (∀ i j, i ≠ j → (i-j ≥ 0 → c (i-j) i ≠ c (i-j) j) ∧ (j-i ≥ 0 → c i (j-i) ≠ c j (j-i))) -- each 135° diagonal has all different colors

-- The formal statement of the math problem
theorem min_colors_required : ∃ (n : ℕ), (∀ c : Grid, is_valid_coloring c → n ≥ 7) :=
sorry

end min_colors_required_l255_255504


namespace interval_for_systematic_sampling_l255_255055

-- Define the total number of students
def total_students : ℕ := 1200

-- Define the sample size
def sample_size : ℕ := 30

-- Define the interval for systematic sampling
def interval_k : ℕ := total_students / sample_size

-- The theorem to prove that the interval k should be 40
theorem interval_for_systematic_sampling :
  interval_k = 40 := sorry

end interval_for_systematic_sampling_l255_255055


namespace lamp_prices_purchasing_plan_count_most_cost_effective_plan_l255_255430

-- Problem conditions
variable (x y m w : ℝ)
variable (m_plan : ℕ)
variable (cost : ℕ → ℝ)

axiom price_conditions :
  x + 3 * y = 26 ∧ 3 * x + 2 * y = 29

axiom purchase_conditions :
  2 * (50 - m) ≤ m ∧ m ≤ 3 * (50 - m)

axiom total_quantity :
  m + (50 - m) = 50

noncomputable def purchasing_plans : List ℕ := [34, 35, 36, 37]

axiom cost_per_plan (m : ℕ) : ℝ :=
  5 * m + 7 * (50 - m)

theorem lamp_prices : (x, y) = (5, 7) :=
sorry

theorem purchasing_plan_count : purchasing_plans.length = 4 :=
sorry

theorem most_cost_effective_plan :
  ∃ m, m ∈ purchasing_plans ∧ cost_per_plan m = 276 ∧ (m = 37 ∧ (50 - m) = 13) :=
sorry

end lamp_prices_purchasing_plan_count_most_cost_effective_plan_l255_255430


namespace repeating_decimal_to_fraction_l255_255641

theorem repeating_decimal_to_fraction : (0.36 : ℝ) = (11 / 30 : ℝ) :=
sorry

end repeating_decimal_to_fraction_l255_255641


namespace add_in_base8_l255_255133

def base8_add (a b : ℕ) (n : ℕ): ℕ :=
  a * (8 ^ n) + b

theorem add_in_base8 : base8_add 123 56 0 = 202 := by
  sorry

end add_in_base8_l255_255133


namespace domain_of_my_function_l255_255857

-- Define the function
def my_function (x : ℝ) : ℝ := 
  Real.sqrt (3 - x) + Real.log (x - 1)

-- Define the domain condition
def in_domain (x : ℝ) : Prop :=
  1 < x ∧ x ≤ 3

-- The domain of the function is (1, 3]
theorem domain_of_my_function :
  ∀ x, (∃ y, y = my_function x) ↔ in_domain x :=
sorry

end domain_of_my_function_l255_255857


namespace series_sum_inequality_l255_255425

theorem series_sum_inequality (n : ℕ) (h : n ≥ 2) : 
  (1 + ∑ i in finset.range n, 1 / ((i + 1) * (i + 1) : ℝ)) < 2 - 1 / (n : ℝ) :=
by {
  sorry
}

end series_sum_inequality_l255_255425


namespace simplify_fraction_l255_255006

theorem simplify_fraction :
  (1 / (1 + Real.sqrt 3) * 1 / (1 - Real.sqrt 5)) = 
  (1 / (1 - Real.sqrt 5 + Real.sqrt 3 - Real.sqrt 15)) :=
by
  sorry

end simplify_fraction_l255_255006


namespace arcsin_cos_eq_neg_pi_div_4_l255_255611

theorem arcsin_cos_eq_neg_pi_div_4 :
  ∀ (x : ℝ), x = 3 * Real.pi / 4 → Real.arcsin (Real.cos x) = - Real.pi / 4 :=
by
  intro x hx
  rw hx
  sorry

end arcsin_cos_eq_neg_pi_div_4_l255_255611


namespace repeating_decimal_to_fraction_l255_255646

theorem repeating_decimal_to_fraction (x : ℝ) (h : x = 0.3 + 0.0666...) : x = 11 / 30 := by
  sorry

end repeating_decimal_to_fraction_l255_255646


namespace max_difference_from_set_l255_255502

open Set

def largest_difference : ℕ := 45

theorem max_difference_from_set (s : Set ℤ) (hs : s = {-20, -10, 0, 5, 15, 25}) :
  ∃ a b ∈ s, (a - b = largest_difference) :=
  sorry

end max_difference_from_set_l255_255502


namespace ellipse_equation_find_m_and_area_l255_255693

-- Definitions based on given conditions
def eccentricity (c a : ℝ) : ℝ := c / a
def ellipse (x y a b : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1
def distance_from_origin_to_line (a b : ℝ) : ℝ := (sqrt 3) / 2

-- Given constants
constant a b c : ℝ
axiom a_pos : a > 0
axiom b_pos : b > 0

-- Proof problem
theorem ellipse_equation 
  (h1 : eccentricity (sqrt 6) 3 = c / a)
  (h2 : distance_from_origin_to_line (a) (b) = (sqrt 3) / 2)
  (h3 : 2 * a^2 = 3 * c^2)
  (h4 :  a^2 = b^2 + c^2)
  :
  ellipse x y (sqrt 3) (1) := sorry

theorem find_m_and_area
  (k : ℝ) (h5 : k = (sqrt 6) / 3)
  (h6 : ∀ x y, ellipse x y (sqrt 3) 1 → ∃ C D, C ≠ D ∧ 
       (C.1 = - (sqrt 6) * (3 / 2) / 3) ∧ (C.2 = (3 / 2) / 3) ∧ 
       distance_from_origin_to_line (A.1) (A.2) = sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)) : 
  ∃ m : ℝ, m = 3 / 2 ∧
  triangle_area (A.1) (A.2) (C.1) (C.2) (D.1) (D.2) = 5 / 4 := sorry

#hide definitions to avoid unncessary assumptions in theorems
#hide eccentricity
#hide ellipse
#hide distance_from_origin_to_line

end ellipse_equation_find_m_and_area_l255_255693


namespace single_reduction_equivalent_l255_255086

theorem single_reduction_equivalent (P : ℝ) : 
  let first_reduction := 0.75 * P in
  let final_price := 0.5 * first_reduction in
  let single_reduction_percentage := 1 - final_price / P in
  single_reduction_percentage = 0.375 := 
by
  sorry

end single_reduction_equivalent_l255_255086


namespace tetrahedron_volume_l255_255769

/-- The volume of a tetrahedron SABC, given that triangle ABC is equilateral,
    the projection H of point A onto the plane SBC is the orthocenter of ΔSBC,
    the dihedral angle between H-AB-C is 30 degrees, and SA = 2√3,
    is equal to 9√3/4.
-/
theorem tetrahedron_volume (S A B C H: Point) (SA : ℝ) (Angle30: ℝ):
  is_equilateral_triangle A B C ∧
  projection_orthocenter A S B C H ∧
  dihedral_angle H A B C = 30 ∧
  SA = 2 * sqrt 3
  → tetrahedron_volume S A B C = 9 * sqrt 3 / 4 := 
by
  sorry

end tetrahedron_volume_l255_255769


namespace tan_alpha_is_minus_one_l255_255679

noncomputable def tan_alpha (α : ℝ) : Prop :=
  sin (π / 6 - α) = cos (π / 6 + α) → tan α = -1

theorem tan_alpha_is_minus_one (α : ℝ) : tan_alpha α :=
  by
  sorry

end tan_alpha_is_minus_one_l255_255679


namespace triangle_angle_60_l255_255293

open Real Trigonometry

def LawOfSines (A B C : ℝ) (a b c R : ℝ) : Prop :=
  a = 2 * R * sin A ∧
  b = 2 * R * sin B ∧
  c = 2 * R * sin C

theorem triangle_angle_60 
  (A B C a b c : ℝ)
  (h1 : (a + b + c) * (sin A + sin B - sin C) = 3 * a * sin B)
  (h2 : LawOfSines A B C a b c (2 * Real.pi)): 
  C = (Real.pi / 3) :=
by
  sorry

end triangle_angle_60_l255_255293


namespace tangent_at_point_range_of_a_l255_255254
-- Import the broader Mathlib library

-- Define the functions given in the problem
def f (x: ℝ) : ℝ := x * Real.log x
def g (x: ℝ) (a: ℝ) : ℝ := -x^2 + a * x - 3

-- Equation of the tangent line at the point (1, 0)
def tangent_line (x: ℝ) : ℝ := x - 1

-- Prove that the tangent line to f at (1, 0) is y = x - 1
theorem tangent_at_point : (∀ x : ℝ, (f 1 = 0) ∧ (f' 1 = 1) → (∀ x : ℝ, tangent_line x = f' 1 * (x - 1) + f 1)) :=
  sorry

-- Prove that if 2f(x) = g(x) for all x in (0, +∞) then a ∈ (-∞, 4]
theorem range_of_a (a : ℝ) : (∀ x : ℝ, (0 < x) → 2 * f x ≥ g x a) → a ≤ 4 :=
  sorry

end tangent_at_point_range_of_a_l255_255254


namespace minimum_odd_numbers_in_set_l255_255393

-- Definitions
variable (P : ℝ → ℝ)
variable (degree_P : ℕ)
variable (A_P : Set ℝ)

-- The conditions: P is a polynomial of degree 8, and 8 is included in A_P
def is_polynomial_of_degree_eight (P : ℝ → ℝ) (degree_P : ℕ) : Prop :=
  degree_P = 8

def set_includes_eight (A_P : Set ℝ) : Prop := 
  8 ∈ A_P

-- The goal: prove the minimum number of odd numbers in A_P is 1
theorem minimum_odd_numbers_in_set {P : ℝ → ℝ} {degree_P : ℕ} {A_P : Set ℝ} :
  is_polynomial_of_degree_eight P degree_P → 
  set_includes_eight A_P → 
  ∃ odd_numbers : ℕ, odd_numbers = 1 :=
sorry

end minimum_odd_numbers_in_set_l255_255393


namespace q_polynomial_sum_roots_l255_255793

-- Given: The polynomial Q(x) = x^3 + ax^2 + bx + c has real coefficients a, b, and c.
-- There exists a complex number u such that the roots of Q(z) are u + 2i, u - 2i, and 2u + 3.

theorem q_polynomial_sum_roots {a b c m n : ℝ} :
  let u := m + n * complex.I in
  let Q := λ x: ℝ, x^3 + a * x^2 + b * x + c in
  (Q (u + 2 * complex.I)).im = 0 ∧ (Q (u - 2 * complex.I)).im = 0 ∧ 
  (Q (2 * u + 3)).im = 0 →
  a + b + c = -2 * m^3 - 2 * m^2 + 7 * m - 4 * n^2 + 1 :=
sorry

end q_polynomial_sum_roots_l255_255793


namespace expected_no_advice_l255_255588

theorem expected_no_advice (n : ℕ) (p : ℝ) (h_p : 0 ≤ p ∧ p ≤ 1) : 
  (∑ j in finset.range n, (1 - p)^j) = (1 - (1 - p)^n) / p :=
by
  sorry

end expected_no_advice_l255_255588


namespace distinct_sum_values_401_l255_255089

def is_subset_of_40 (A : Finset ℕ) : Prop :=
  A.card = 40 ∧ A ⊆ (Finset.range 50) ∧ 0 ∉ A

def sum_elements (A : Finset ℕ) : ℕ :=
  A.sum id

theorem distinct_sum_values_401 : 
  ∀ (A : Finset ℕ), is_subset_of_40 A →
  ∃ (n : ℕ), 820 ≤ n ∧ n ≤ 1220 ∧
  (sum_elements A = n) ∧
  (Finset.range (1220 - 820 + 1)).card = 401 :=
begin
  sorry
end

end distinct_sum_values_401_l255_255089


namespace aria_analysis_time_l255_255580

-- Definitions for the number of bones in each section
def skull_bones : ℕ := 29
def spine_bones : ℕ := 33
def thorax_bones : ℕ := 37
def upper_limb_bones : ℕ := 64
def lower_limb_bones : ℕ := 62

-- Definitions for the time spent per bone in each section (in minutes)
def time_per_skull_bone : ℕ := 15
def time_per_spine_bone : ℕ := 10
def time_per_thorax_bone : ℕ := 12
def time_per_upper_limb_bone : ℕ := 8
def time_per_lower_limb_bone : ℕ := 10

-- Definition for the total time needed in minutes
def total_time_in_minutes : ℕ :=
  (skull_bones * time_per_skull_bone) +
  (spine_bones * time_per_spine_bone) +
  (thorax_bones * time_per_thorax_bone) +
  (upper_limb_bones * time_per_upper_limb_bone) +
  (lower_limb_bones * time_per_lower_limb_bone)

-- Definition for the total time needed in hours
def total_time_in_hours : ℚ := total_time_in_minutes / 60

-- Theorem to prove the total time needed in hours is approximately 39.02
theorem aria_analysis_time : abs (total_time_in_hours - 39.02) < 0.01 :=
by
  sorry

end aria_analysis_time_l255_255580


namespace marbles_in_jar_l255_255107

theorem marbles_in_jar (x : ℕ)
  (h1 : \frac{1}{2} * x + \frac{1}{4} * x + 27 + 14 = x) : x = 164 := sorry

end marbles_in_jar_l255_255107


namespace seashells_left_sam_seashells_now_l255_255839

-- Problem conditions
def initial_seashells : ℕ := 35
def seashells_given : ℕ := 18

-- Proof problem statement
theorem seashells_left (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

-- The required statement
theorem sam_seashells_now : seashells_left initial_seashells seashells_given = 17 := by 
  sorry

end seashells_left_sam_seashells_now_l255_255839


namespace ramola_rank_from_first_l255_255005

-- Conditions definitions
def total_students : ℕ := 26
def ramola_rank_from_last : ℕ := 13

-- Theorem statement
theorem ramola_rank_from_first : total_students - (ramola_rank_from_last - 1) = 14 := 
by 
-- We use 'by' to begin the proof block
sorry 
-- We use 'sorry' to indicate the proof is omitted

end ramola_rank_from_first_l255_255005


namespace tan_pi_over_12_plus_tan_7pi_over_12_l255_255007

theorem tan_pi_over_12_plus_tan_7pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (7 * Real.pi / 12)) = -4 * (3 - Real.sqrt 3) / 5 :=
by
  sorry

end tan_pi_over_12_plus_tan_7pi_over_12_l255_255007


namespace problem1_problem2_l255_255684

theorem problem1 (x : ℝ) : (4 * x ^ 2 + 12 * x - 7 ≤ 0) ∧ (a = 0) ∧ (x < -3 ∨ x > 3) → (-7/2 ≤ x ∧ x < -3) := by
  sorry

theorem problem2 (a : ℝ) : (∀ x : ℝ, 4 * x ^ 2 + 12 * x - 7 ≤ 0 → a - 3 ≤ x ∧ x ≤ a + 3) → (-5/2 ≤ a ∧ a ≤ -1/2) := by
  sorry

end problem1_problem2_l255_255684


namespace line_JK_through_midpoint_BC_l255_255224

noncomputable def midpoint_arc (Γ : Circle) (A B C : Point) : Point := 
classical.some (exists_midpoint_minor_arc Γ A B C)

noncomputable def intersect_circles (ω Γ : Circle) : Point := 
classical.some (Circle.intersection ω Γ)

noncomputable def intersect_segments (P S : Segment) : Point :=
classical.some (Segment.intersection P S)

noncomputable def is_midpoint (M B C : Point) : Prop :=
dist M B = dist M C

theorem line_JK_through_midpoint_BC 
(Γ : Circle) ( A B C D E F G H J K P : Point)
(A1: circle Γ)
(A2: acute_triangl ABC)
(A3: AC > AB)
(D1 : midpoint_arc Γ A B C = D)
(EF1: AE = AF)
(ω : Circle)
(C1: circumcircle_of ω (A, E, F))
(P1: intersect_circles ω Γ = P)
(G1: intersect_segments P E = G)
(H1: intersect_segments P F = H)
(J1: intersect_lines D G AB = J)
(K1: intersect_lines D H AC = K)
: ∃ M : Point, is_midpoint M B C ∧ collinear J K M := 
begin
  sorry
end

end line_JK_through_midpoint_BC_l255_255224


namespace find_largest_modulus_l255_255801

noncomputable def largest_possible_modulus (a b c d z : ℂ) 
  (ha : |a| = |b|)
  (hb : |b| = |c|)
  (hc : |c| = |d|)
  (hpos : |a| > 0)
  (h_eqn : a * z^3 + b * z^2 + c * z + d = 0) : ℝ :=
1.84

theorem find_largest_modulus (a b c d z : ℂ) 
  (ha : |a| = |b|)
  (hb : |b| = |c|)
  (hc : |c| = |d|)
  (hpos : |a| > 0)
  (h_eqn : a * z^3 + b * z^2 + c * z + d = 0) :
  |z| ≤ largest_possible_modulus a b c d z ha hb hc hpos h_eqn :=
by
  sorry

end find_largest_modulus_l255_255801


namespace minimum_odd_numbers_in_set_l255_255389

-- Definitions
variable (P : ℝ → ℝ)
variable (degree_P : ℕ)
variable (A_P : Set ℝ)

-- The conditions: P is a polynomial of degree 8, and 8 is included in A_P
def is_polynomial_of_degree_eight (P : ℝ → ℝ) (degree_P : ℕ) : Prop :=
  degree_P = 8

def set_includes_eight (A_P : Set ℝ) : Prop := 
  8 ∈ A_P

-- The goal: prove the minimum number of odd numbers in A_P is 1
theorem minimum_odd_numbers_in_set {P : ℝ → ℝ} {degree_P : ℕ} {A_P : Set ℝ} :
  is_polynomial_of_degree_eight P degree_P → 
  set_includes_eight A_P → 
  ∃ odd_numbers : ℕ, odd_numbers = 1 :=
sorry

end minimum_odd_numbers_in_set_l255_255389


namespace Louie_monthly_payment_correct_l255_255917

-- Given conditions
def P : ℕ := 2000
def r : ℝ := 0.10
def t : ℝ := 3 / 12

-- Calculate the total amount owed after 3 months with monthly compounding
noncomputable def A : ℝ := P * (1 + r)^(t)

-- Total amount that Louie has to pay each month, rounded to the nearest dollar
noncomputable def monthly_payment : ℝ := A / 3

theorem Louie_monthly_payment_correct : round (monthly_payment) = 716 :=
by sorry

end Louie_monthly_payment_correct_l255_255917


namespace perfect_square_unique_n_l255_255651

theorem perfect_square_unique_n (n : ℕ) (hn : n > 0) : 
  (∃ m : ℕ, 2^n + 12^n + 2011^n = m^2) ↔ n = 1 := by
  sorry

end perfect_square_unique_n_l255_255651


namespace bus_driver_total_hours_l255_255539

variables (R OT : ℕ)

-- Conditions
def regular_rate := 16
def overtime_rate := 28
def max_regular_hours := 40
def total_compensation := 864

-- Proof goal: total hours worked is 48
theorem bus_driver_total_hours :
  (regular_rate * R + overtime_rate * OT = total_compensation) →
  (R ≤ max_regular_hours) →
  (R + OT = 48) :=
by
  sorry

end bus_driver_total_hours_l255_255539


namespace lines_in_parallel_planes_l255_255746

-- Define the necessary conditions
variables {α : Type*} [affine_space α] 
variables (P Q : affine_subspace α) -- Represent two planes P and Q
variable (l₁ : line_in_subspace P)
variable (l₂ : line_in_subspace Q)
variable (hP : ∀ p₁ p₂ : α, p₁ ∈ P → p₂ ∈ Q → p₁ ≠ p₂)

-- Define the theorem statement
theorem lines_in_parallel_planes (hPQ : P.parallel Q) : 
  parallel l₁ l₂ ∨ skew l₁ l₂ :=
sorry

end lines_in_parallel_planes_l255_255746


namespace vector_inequality_l255_255264

open Real

variables {n : ℕ} 
variables (a b c : EuclideanSpace ℝ n)

/-- Given vectors a, b, c ∈ Rⁿ, the following inequality holds:
(‖a‖ ⟨b, c⟩)² + (‖b‖ ⟨a, c⟩)² ≤ ‖a‖ ‖b‖ (‖a‖ ‖b‖ + |⟨a, b⟩|) ‖c‖²
where ⟨x, y⟩ denotes the inner product of vectors x and y 
and ‖x‖² = ⟨x, x⟩. -/
theorem vector_inequality :
  (‖a‖ * ⟨b, c⟩) ^ 2 + (‖b‖ * ⟨a, c⟩) ^ 2 
  ≤ ‖a‖ * ‖b‖ * (‖a‖ * ‖b‖ + |⟨a, b⟩|) * ‖c‖^2 :=
sorry

end vector_inequality_l255_255264


namespace number_of_primes_with_ones_digit_3_l255_255269

noncomputable def count_primes_with_ones_digit_3 : Nat :=
  let primes := [3, 13, 23, 33, 43]
  primes.count (fun n => (Nat.Prime n) ∧ n < 50)

theorem number_of_primes_with_ones_digit_3 : count_primes_with_ones_digit_3 = 4 := by
  sorry

end number_of_primes_with_ones_digit_3_l255_255269


namespace ones_digit_of_power_35_35_pow_17_17_is_five_l255_255182

theorem ones_digit_of_power_35_35_pow_17_17_is_five :
  (35 ^ (35 * (17 ^ 17))) % 10 = 5 := by
  sorry

end ones_digit_of_power_35_35_pow_17_17_is_five_l255_255182


namespace geometric_sequence_a4_l255_255473

theorem geometric_sequence_a4 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 4)
  (h3 : a 6 = 16) : 
  a 4 = 8 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_a4_l255_255473


namespace segment_area_eq_l255_255870

def equilateral_triangle_side_length := a : ℝ 

def radius_of_circumcircle_triangle := a / Real.sqrt 3

def area_of_sector := (Real.pi * (radius_of_circumcircle_triangle ^ 2)) / 3

def area_of_triangle := a * radius_of_circumcircle_triangle / 4

theorem segment_area_eq :
  (a^2 * (4 * Real.pi - 3 * Real.sqrt 3)) / 36 = 
    area_of_sector - area_of_triangle := by
  sorry

end segment_area_eq_l255_255870


namespace region_R_area_correct_l255_255013

noncomputable def side_length : ℝ := 3
noncomputable def vertex_angle_W : ℝ := 90
noncomputable def region_R_area : ℝ := (9 * Real.pi) / 8

theorem region_R_area_correct (s : ℝ) (a : ℝ) :
  s = side_length → a = vertex_angle_W → region_R_area = (9 * Real.pi) / 8 := by
  intros hs ha
  rw [hs, ha]
  sorry

end region_R_area_correct_l255_255013


namespace plane_through_point_and_line_l255_255177

def point := (ℝ × ℝ × ℝ)
def line_eq (a b c d e f : ℝ) (p : point) : Prop :=
  let (x, y, z) := p
  (x - a) / b = (y - c) / d ∧ (y - c) / d = (z - e) / f

def plane_eq (A B C D : ℝ) (p : point) : Prop :=
  let (x, y, z) := p
  A * x + B * y + C * z + D = 0

theorem plane_through_point_and_line:
  ∃ A B C D : ℤ, 
    A > 0 ∧ Int.gcd_pure A B C D = 1 ∧
    plane_eq A B C D (1, 4, -5) ∧
    (∃ t : ℝ, line_eq 2 4 1 (-1) 3 5 (1 + t * 4, 4 - t, -5 + t * 5)) ∧
    plane_eq A B C D = plane_eq 2 7 6 (-66) :=
begin
  -- Proof goes here
  sorry
end

end plane_through_point_and_line_l255_255177


namespace teacher_arrangement_l255_255541

theorem teacher_arrangement:
  let num_mt = 3 in    -- number of teachers who can only teach "Matrix and Transformation"
  let num_is = 3 in    -- number of teachers who can only teach "Information Security and Cryptography"
  let num_sc = 2 in    -- number of teachers who can only teach "Switching Circuits and Boolean Algebra"
  let num_all = 2 in   -- number of teachers who can teach all three courses
  let total_teachers = 10 in
  let teachers_needed = 9 in
  let course_teachers = 3 in
  ∃ arrangements : ℕ 
    (h1 : arrangements = 16), arrangements = 16 :=
begin
  sorry
end

end teacher_arrangement_l255_255541


namespace repeating_decimal_to_fraction_l255_255635

theorem repeating_decimal_to_fraction : (0.3666666 : ℚ) = 11 / 30 :=
by sorry

end repeating_decimal_to_fraction_l255_255635


namespace angle_minus_a_c_l255_255263

variables (a b : ℝ^3) (θ_ab : ℝ) (c : ℝ^3)

-- Conditions
def angle_between (u v : ℝ^3) : ℝ := sorry
def vector_addition (u v : ℝ^3) : ℝ^3 := u + v

axiom angle_ab : angle_between a b = 75
axiom vector_c_definition : c = vector_addition a b

-- The proof problem
theorem angle_minus_a_c : angle_between (-a) c = 142.5 := sorry

end angle_minus_a_c_l255_255263


namespace prob_pass_first_round_expected_rounds_l255_255538

noncomputable def probability_pass_first_round : ℝ :=
  1 - (1 - 0.6) * (1 - 0.6)

theorem prob_pass_first_round : probability_pass_first_round = 0.84 :=
by
  unfold probability_pass_first_round
  simp
  norm_num
  sorry -- details of norm_num result

noncomputable def expected_rounds_passed (n : ℕ) (p : ℝ) : ℝ :=
  n * p

theorem expected_rounds : expected_rounds_passed 5 probability_pass_first_round = 4.2 :=
by
  unfold expected_rounds_passed probability_pass_first_round
  simp
  norm_num
  sorry -- details of norm_num result

end prob_pass_first_round_expected_rounds_l255_255538


namespace sum_of_integers_l255_255884

theorem sum_of_integers (a b c : ℕ) (h1 : 2 < a) (h2 : 2 < b) (h3 : 2 < c)
  (h_prod : a * b * c = 19683) (h_coprime_ab : Nat.coprime a b)
  (h_coprime_bc : Nat.coprime b c) (h_coprime_ca : Nat.coprime c a) : a + b + c = 117 := 
sorry

end sum_of_integers_l255_255884


namespace vacuum_pump_operations_l255_255568

theorem vacuum_pump_operations (n : ℕ) (h : n ≥ 10) : 
  ∀ a : ℝ, 
  a > 0 → 
  (0.5 ^ n) * a < 0.001 * a :=
by
  intros a h_a
  sorry

end vacuum_pump_operations_l255_255568


namespace repeating_decimal_to_fraction_l255_255644

theorem repeating_decimal_to_fraction (x : ℝ) (h : x = 0.3 + 0.0666...) : x = 11 / 30 := by
  sorry

end repeating_decimal_to_fraction_l255_255644


namespace no_real_solutions_l255_255031

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else (2 - x^2) / x

theorem no_real_solutions :
  (∀ x : ℝ, x ≠ 0 → (f x + 2 * f (1 / x) = 3 * x)) →
  (∀ x : ℝ, f x = f (-x) → false) :=
by
  intro h1 h2
  sorry

end no_real_solutions_l255_255031


namespace largest_perfect_square_factor_of_3465_l255_255895

theorem largest_perfect_square_factor_of_3465 : ∃ x, is_square x ∧ x ∣ 3465 ∧ ∀ y, is_square y ∧ y ∣ 3465 → y ≤ x :=
begin
  sorry
end

end largest_perfect_square_factor_of_3465_l255_255895


namespace domain_of_function_l255_255624

def y (x : ℝ) : ℝ := real.sqrt (-real.log (1 + x))

theorem domain_of_function :
  {x : ℝ | 1 + x > 0 ∧ -real.log (1 + x) ≥ 0} = set.Ioo (-1) 0 ∪ {0} :=
by
  sorry

end domain_of_function_l255_255624


namespace find_point_D_l255_255057

-- Define the given conditions

variables {α : Type} [metric_space α] [euclidean_geometry α]
variables {A B C D : α}

def scalene_triangle (A B C : α) : Prop :=
(A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

def obtuse_angle (A B C : α) : Prop :=
∃ θ : nnreal, (θ > π / 2) ∧ ((angle A B C) = θ)

def on_extended_line (D B C : α) : Prop :=
∃ t : ℝ, t ≠ 0 ∧ ((D = B + t • (C - B)) ∨ (D = C + t • (B - C))) -- assuming Euclidean geometry with vector operations

noncomputable def satisfies_condition (A B C D : α) : Prop :=
(dist A D) = real.sqrt ((dist B D) * (dist C D))

theorem find_point_D (A B C : α) (h_scalene : scalene_triangle A B C) (h_obtuse : obtuse_angle A B C) :
  ∃ D, on_extended_line D B C ∧ satisfies_condition A B C D :=
begin
  sorry
end

end find_point_D_l255_255057


namespace sequence_formula_l255_255323

theorem sequence_formula (a : ℕ → ℕ) (h₀ : a 0 = 2) (h₁ : ∀ n, a (n + 1) = (n + 3) * a n) :
  ∀ n, a n = (n + 2)! :=
by sorry

end sequence_formula_l255_255323


namespace probability_of_different_topics_l255_255953

theorem probability_of_different_topics (n : ℕ) (m : ℕ) (prob : ℚ)
  (h1 : n = 36)
  (h2 : m = 30)
  (h3 : prob = 5/6) :
  (m : ℚ) / (n : ℚ) = prob :=
sorry

end probability_of_different_topics_l255_255953


namespace problem_statement_l255_255718

def sequence (n : ℕ) : ℝ :=
  if n = 1 then 1 else sequence (n - 1) / (sequence (n - 1) + 1)

noncomputable def sum_of_squares (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), (sequence i) ^ 2

theorem problem_statement : (⌊ sum_of_squares 2017 ⌋ = 1) :=
  sorry

end problem_statement_l255_255718


namespace chlorine_needed_l255_255268

-- Define the balanced reaction
def reaction : Prop := 
  ∀ (n : ℕ), (n * (1 : ℕ)) = (n * (1 : ℕ))

-- Define that 3 moles of Cl2 are needed for 3 moles of ethane
def chlorine_needed_for_ethane : ℕ := 3

-- The proof statement
theorem chlorine_needed : reaction ∧ (chlorine_needed_for_ethane = 3) := 
  by { split, exact sorry, exact rfl }

end chlorine_needed_l255_255268


namespace min_value_of_f_max_value_of_f_l255_255248

-- Part 1: If a = 2, prove the minimum value of f(x) in [0,3] is -1.
theorem min_value_of_f (a : ℝ) (h : a = 2) : 
  ∃ x ∈ set.Icc (0 : ℝ) 3, (λ x, -x^2 + 2*a*x + 1 - a) x = -1 :=
sorry

-- Part 2: If f(x) has a maximum value of 3 in [0,1], prove a = -2 or a = 3.
theorem max_value_of_f (h : ∃ x ∈ set.Icc (0 : ℝ) 1, (λ x, -x^2 + 2*a*x + 1 - a) x = 3) : 
  a = -2 ∨ a = 3 :=
sorry

end min_value_of_f_max_value_of_f_l255_255248


namespace dot_pathway_length_l255_255941

noncomputable def length_of_pathway_traveled_by_dot : ℝ :=
  let edge_length := 2 in
  let radius := edge_length in
  let circumference := 2 * real.pi * radius in
  4 * circumference

theorem dot_pathway_length (edge_length : ℝ) (initial_dot_position : bool) :
  edge_length = 2 → initial_dot_position = tt →
  length_of_pathway_traveled_by_dot = 8 * real.pi :=
by
  intros h_edge_length h_dot_position
  rw [length_of_pathway_traveled_by_dot, h_edge_length]
  sorry

end dot_pathway_length_l255_255941


namespace minimum_length_PA_equation_common_tangent_l255_255763

theorem minimum_length_PA
    (m : ℝ) (m_pos : m > 0)
    (P : ℝ × ℝ) (x1 y1 : ℝ) (H_P : P.1 - P.2 + 2 = 0)
    (O1 : ℝ × ℝ) (H_O1 : O1 = (3, 1))
    (r : ℝ) (H_r : r = 1) :
    let PA := Real.sqrt ((P.1 - O1.1)^2 + (P.2 - O1.2)^2 - r^2) in
    PA = Real.sqrt 7 := sorry

theorem equation_common_tangent
    (m : ℝ) (m_pos : m > 0)
    (r1 r2 : ℝ) 
    (H_radii_product : r1 * r2 = 2)
    (Q : ℝ × ℝ) (H_Q : Q = (2, 2))
    (H_O1 : (2 - r1 / m) ^ 2 + (2 - r1) ^ 2 = r1 ^ 2)
    (H_O2 : (2 - r2 / m) ^ 2 + (2 - r2) ^ 2 = r2 ^ 2)
    (H_m: m = 1/2) :
    let tangent_line_slope := 2 * m / (1 - m ^ 2) in
    tangent_line_slope = 4/3 ∧
    let tangent_line : ℝ × ℝ := (4/3, 0) in
    ∀ x : ℝ, tangent_line.2 = 4 / 3 * x := sorry

end minimum_length_PA_equation_common_tangent_l255_255763


namespace range_of_m_l255_255721

-- Definition of sets A and B
def A : set ℝ := {x | x^2 ≥ 16}
def B (m : ℝ) : set ℝ := {m}

-- Statement that B ⊆ A when A ∪ B = A
theorem range_of_m (m : ℝ) : A ∪ B m = A → B m ⊆ A := by
  intro h
  unfold A at *
  unfold B at *
  sorry

end range_of_m_l255_255721


namespace curve_crossing_l255_255581

structure Point where
  x : ℝ
  y : ℝ

def curve (t : ℝ) : Point :=
  { x := 2 * t^2 - 3, y := 2 * t^4 - 9 * t^2 + 6 }

theorem curve_crossing : ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve 1 = { x := -1, y := -1 } := by
  sorry

end curve_crossing_l255_255581


namespace problem_statement_l255_255409

theorem problem_statement :
  ∃ (a b c : ℕ), gcd a (gcd b c) = 1 ∧
  (∃ x y : ℝ, 2 * y = 8 * x - 7) ∧
  a ^ 2 + b ^ 2 + (c:ℤ) ^ 2 = 117 :=
sorry

end problem_statement_l255_255409


namespace find_circumcircle_radius_l255_255419

noncomputable def circumcircle_radius (A B C M N : Point) (r : ℝ) : Prop :=
let dist := Euclidean.dist in
M.on_bisector_of_angle ∠ BAC ∧
N.extends_line A B ∧
dist A C = 1 ∧
dist A M = 1 ∧
∠ANM = ∠CNM ∧
circumradius_triangle C N M = r

theorem find_circumcircle_radius (A B C M N : Point) :
  circumcircle_radius A B C M N 1 :=
sorry

end find_circumcircle_radius_l255_255419


namespace radius_of_circle_of_complex_roots_proven_l255_255571

noncomputable def radius_of_circle_of_complex_roots (z : ℂ) (h : (z - 2)^4 = 64 * z^4) : ℝ :=
  √((4 : ℝ) / 15)

theorem radius_of_circle_of_complex_roots_proven :
  ∀ (z : ℂ) (h : (z - 2)^4 = 64 * z^4), radius_of_circle_of_complex_roots z h = 4 / 15 :=
by
  sorry

end radius_of_circle_of_complex_roots_proven_l255_255571


namespace monotonicity_of_f_range_of_a_for_two_zeros_l255_255716

noncomputable def f (x a : ℝ) := Real.exp x - a * (x + 2)

theorem monotonicity_of_f (a : ℝ) (h : a = 1) :
  (∀ x, x < 0 → deriv (λ x, f x 1) x < 0) ∧
  (∀ x, x > 0 → deriv (λ x, f x 1) x > 0) :=
sorry

theorem range_of_a_for_two_zeros (a : ℝ) (h_has_two_zeros : ∃ x1 x2, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) :
  a > (1 / Real.exp 1) :=
sorry

end monotonicity_of_f_range_of_a_for_two_zeros_l255_255716


namespace sqrt_squared_eq_original_l255_255614

theorem sqrt_squared_eq_original (m : ℕ) (h : m = 529441) : (Real.sqrt (m : ℝ))^2 = m :=
by
  rw [h]
  exact Real.sqrt_sq (by norm_num : m = 529441)

end sqrt_squared_eq_original_l255_255614


namespace mean_of_set_l255_255898

theorem mean_of_set {m : ℝ} 
  (median_condition : (m + 8 + m + 11) / 2 = 19) : 
  (m + (m + 6) + (m + 8) + (m + 11) + (m + 18) + (m + 20)) / 6 = 20 := 
by 
  sorry

end mean_of_set_l255_255898


namespace number_of_valid_pairs_l255_255153

theorem number_of_valid_pairs :
  ∃ n : ℕ, (∀ x y : ℤ, 1 ≤ x ∧ x < y ∧ y ≤ 150 ∧ (complex.I^x + complex.I^y).im = 0 → n = 4218) :=
begin
  use 4218,
  intros x y hx hy h_ix_iy_reality,
  sorry
end

end number_of_valid_pairs_l255_255153


namespace water_usage_in_May_l255_255099

theorem water_usage_in_May (x : ℝ) (h_cost : 45 = if x ≤ 12 then 2 * x 
                                                else if x ≤ 18 then 24 + 2.5 * (x - 12) 
                                                else 39 + 3 * (x - 18)) : x = 20 :=
sorry

end water_usage_in_May_l255_255099


namespace round_table_chairs_l255_255134

theorem round_table_chairs :
  ∃ x : ℕ, (2 * x + 2 * 7 = 26) ∧ x = 6 :=
by
  sorry

end round_table_chairs_l255_255134


namespace product_last_digit_divisible_by_3_l255_255002

theorem product_last_digit_divisible_by_3 (n : ℕ) : 
  let a := (2^n % 10) in
  3 ∣ a * (2^n - a) :=
by
  let a := (2^n % 10)
  sorry

end product_last_digit_divisible_by_3_l255_255002


namespace total_spider_legs_l255_255518

variable (numSpiders : ℕ)
variable (legsPerSpider : ℕ)
axiom h1 : numSpiders = 5
axiom h2 : legsPerSpider = 8

theorem total_spider_legs : numSpiders * legsPerSpider = 40 :=
by
  -- necessary for build without proof.
  sorry

end total_spider_legs_l255_255518


namespace find_max_third_side_l255_255017

-- Definitions from problem
variables (P Q R : ℝ) (a b : ℝ)

-- Condition
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Lean statement
theorem find_max_third_side (h1 : cos (2*P) + cos (2*Q) + cos (2*R) = 1)
  (h2 : a = 7) (h3 : b = 24) :
  ∃ c, is_right_triangle a b c ∧ c = 25 :=
by
  use 25
  split
  · -- Proof for is_right_triangle
    sorry
  · rfl
  sorry

end find_max_third_side_l255_255017


namespace compare_expression_l255_255804

variable (m x : ℝ)

theorem compare_expression : x^2 - x + 1 > -2 * m^2 - 2 * m * x := 
sorry

end compare_expression_l255_255804


namespace arithmetic_mean_of_set_l255_255617

theorem arithmetic_mean_of_set (n : ℕ) (h : n > 2) :
  let a_i := (1 - (2 : ℚ) / n) ^ 2
  let a_set := finset.cons a_i (finset.replicate (n - 1) 1) sorry
  ∑ x in a_set, x / n = 1 - 4 / n ^ 2 := 
sorry

end arithmetic_mean_of_set_l255_255617


namespace find_pairs_composite_sum_l255_255530

-- Part (a)
theorem find_pairs (m n : ℕ) :
  (5 * m + 8 * n = 120) ↔ ((m, n) = (24, 0) ∨ (m, n) = (16, 5) ∨ (m, n) = (8, 10) ∨ (m, n) = (0, 15)) :=
sorry

-- Part (b)
theorem composite_sum (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 1)
  (h_eq : (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / c) :
  Nat.isComposite (a + c) ∨ Nat.isComposite (b + c) :=
sorry

end find_pairs_composite_sum_l255_255530


namespace repeating_decimal_to_fraction_l255_255643

theorem repeating_decimal_to_fraction (x : ℝ) (h : x = 0.3 + 0.0666...) : x = 11 / 30 := by
  sorry

end repeating_decimal_to_fraction_l255_255643


namespace find_number_l255_255024

def digits := {n : ℕ // n < 10}

theorem find_number (明 白 清 楚 : digits) :
  (明 ≠ 白) ∧ (明 ≠ 清) ∧ (明 ≠ 楚) ∧ (白 ≠ 清) ∧ (白 ≠ 楚) ∧ (清 ≠ 楚) →
  (11 * 明) * (11 * 白) = 1111 * 清 + 101 * 楚 →
  let number := 1000 * 明 + 100 * 白 + 10 * 清 + 楚 in
  number = 4738 ∨ number = 7438 ∨ number = 8874 :=
by { intros h1 h2, sorry }

end find_number_l255_255024


namespace dogwood_trees_proof_l255_255296

def dogwood_trees_left (a b c : Float) : Float :=
  a + b - c

theorem dogwood_trees_proof : dogwood_trees_left 5.0 4.0 7.0 = 2.0 :=
by
  -- The proof itself is left out intentionally as per the instructions
  sorry

end dogwood_trees_proof_l255_255296


namespace number_of_valid_n_is_seven_l255_255194

theorem number_of_valid_n_is_seven :
  {n : ℕ | 0 < n ∧ n < 24 ∧ ∃ k : ℕ, n = k * (24 - n)}.card = 7 :=
by
  sorry

end number_of_valid_n_is_seven_l255_255194


namespace triangle_area_on_ellipse_l255_255479

def onEllipse (p : ℝ × ℝ) : Prop := (p.1)^2 + 4 * (p.2)^2 = 4

def isCentroid (C : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  C = ((A.1 + B.1) / 3, (A.2 + B.2) / 3)

theorem triangle_area_on_ellipse
  (A B C : ℝ × ℝ)
  (h₁ : A ≠ B)
  (h₂ : B ≠ C)
  (h₃ : C ≠ A)
  (h₄ : onEllipse A)
  (h₅ : onEllipse B)
  (h₆ : onEllipse C)
  (h₇ : isCentroid C A B)
  (h₈ : C = (0, 0))  : 
  1 / 2 * (A.1 - B.1) * (B.2 - A.2) = 1 :=
by
  sorry

end triangle_area_on_ellipse_l255_255479


namespace C1_rectangular_eqn_C2_rectangular_eqn_max_distance_PQ_l255_255322

-- Define the curves
def C1 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

-- Polar to rectangular conversion for C2
def C2 (ρ θ : ℝ) : Prop := ρ^2 = 4 * ρ * Real.sin θ - 3

theorem C1_rectangular_eqn (x y θ : ℝ) : (x = 2 * Real.cos θ) → (y = Real.sin θ) → (x^2 / 4 + y^2 = 1) :=
by
  intros hx hy
  rw [hx, hy]
  have h1 : (2 * Real.cos θ)^2 / 4 = Real.cos θ^2 * 4 / 4 := sorry,
  rw h1
  have h2 : (2 * Real.cos θ)^2 / 4 = Real.cos θ^2 := sorry,
  rw [Real.sin]
  sorry

theorem C2_rectangular_eqn (ρ θ x y : ℝ) (h : C2 ρ θ) : (x = ρ * Real.cos θ) → (y = ρ * Real.sin θ) → (x^2 + (y - 2)^2 = 1) :=
by
  intros hx hy
  have hx_squared := sorry,
  have hy_squared := sorry,
  simp only [hx, hy, sq] at hx_squared hy_squared,
  rw [hx_squared, hy_squared]
  sorry

theorem max_distance_PQ (x1 y1 x2 y2 : ℝ) (h1 : x1^2 / 4 + y1^2 = 1) (h2 : x2^2 + (y2-2)^2 = 1) :
  ∃ θ, ∃ ρ, C1 θ = (x1, y1) ∧ (C2 ρ θ) ∧ (x2² + (y2 - 2)² = 1) →
  ∀ θ ∈ set.Icc (-π) π, 
  (x2 = ρ_arc cos (θ) ∧ (distance P Q := by √ (-4*sin θ - 2)² + 4 cos θ²)) max (abs (θ - (asin (2/√17)))) (ρ ARC sin θ - 3)
  sorry

end C1_rectangular_eqn_C2_rectangular_eqn_max_distance_PQ_l255_255322


namespace gcd_digits_le_two_l255_255284

open Nat

theorem gcd_digits_le_two (a b : ℕ) (ha : a < 10^5) (hb : b < 10^5)
  (hlcm_digits : 10^8 ≤ lcm a b ∧ lcm a b < 10^9) : gcd a b < 100 :=
by
  -- Proof here
  sorry

end gcd_digits_le_two_l255_255284


namespace find_max_third_side_l255_255018

-- Definitions from problem
variables (P Q R : ℝ) (a b : ℝ)

-- Condition
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Lean statement
theorem find_max_third_side (h1 : cos (2*P) + cos (2*Q) + cos (2*R) = 1)
  (h2 : a = 7) (h3 : b = 24) :
  ∃ c, is_right_triangle a b c ∧ c = 25 :=
by
  use 25
  split
  · -- Proof for is_right_triangle
    sorry
  · rfl
  sorry

end find_max_third_side_l255_255018


namespace find_cost_price_find_max_profit_l255_255543

noncomputable def cost_price (selling_price discount_rate profit_rate : ℝ) : ℝ :=
  let discounted_price := selling_price * discount_rate
  in discounted_price / (1 + profit_rate)

noncomputable def max_profit (discounted_price original_cost : ℝ) : ℝ :=
  let profit_per_piece (x : ℝ) := (discounted_price - x - original_cost)
  let total_profit (x : ℝ) := (20 + 4 * x) * profit_per_piece x
  in total_profit (15 / 2)

-- Definitions based on problem conditions
def selling_price : ℝ := 75
def discount_rate : ℝ := 0.8
def profit_rate : ℝ := 0.5

-- Given cost price (conditioned by profit and discounted selling price)
theorem find_cost_price : 
  cost_price selling_price discount_rate profit_rate = 40 := 
sorry

-- Given maximum profit (conditioned by profit formula and sales relationship)
theorem find_max_profit : 
  max_profit (selling_price * discount_rate) 40 = 625 := 
sorry

end find_cost_price_find_max_profit_l255_255543


namespace distance_from_focus_to_point_l255_255027

theorem distance_from_focus_to_point :
  let x := 2
  let y := 5
  let f_x := 0
  let f_y := 4
  (x^2 = 16*y) → 
  (f_x = 0 ∧ f_y = 4) →
  (∃ d, d = Real.sqrt ((x - f_x)^2 + (y - f_y)^2) ∧ d = Real.sqrt 5) :=
by
  -- Definition of parabola condition: x^2 = 16y
  intros x y f_x f_y h1 h2
  -- Focus coordinates
  cases h2 with fx_eq0 fy_eq4
  rw [fx_eq0, fy_eq4]
  use Real.sqrt ((x - 0)^2 + (y - 4)^2)
  rw [← fx_eq0, ← fy_eq4]
  -- Proving specific distance
  sorry

end distance_from_focus_to_point_l255_255027


namespace width_of_room_l255_255466

-- Definitions from conditions
def length : ℝ := 8
def total_cost : ℝ := 34200
def cost_per_sqm : ℝ := 900

-- Theorem stating the width of the room
theorem width_of_room : (total_cost / cost_per_sqm) / length = 4.75 := by 
  sorry

end width_of_room_l255_255466


namespace number_exceeds_fraction_80_l255_255812

theorem number_exceeds_fraction_80 (x : ℝ) (h : x = (3 / 7) * x + 0.8 * (3 / 7) * x) : x = 0 := 
by
  sorry

end number_exceeds_fraction_80_l255_255812


namespace number_of_lemons_l255_255445

theorem number_of_lemons
  (total_fruits : ℕ)
  (mangoes : ℕ)
  (pears : ℕ)
  (pawpaws : ℕ)
  (kiwis : ℕ)
  (lemons : ℕ)
  (h_total : total_fruits = 58)
  (h_mangoes : mangoes = 18)
  (h_pears : pears = 10)
  (h_pawpaws : pawpaws = 12)
  (h_kiwis_lemons_equal : kiwis = lemons) :
  lemons = 9 :=
by
  sorry

end number_of_lemons_l255_255445


namespace polynomial_bound_l255_255424

theorem polynomial_bound (n : ℕ) (a : fin n → ℝ) (P : ℝ → ℝ) 
  (hP : ∀ x, P x = (x^n + ∑ i in finset.fin_range n, a i * x^i)) :
  ∃ k ∈ finset.Icc 1 (n + 1), |P k| ≥ (nat.factorial n / 2^n) :=
by { sorry }

end polynomial_bound_l255_255424


namespace engineers_percentage_calculation_l255_255298

noncomputable def percentageEngineers (num_marketers num_engineers num_managers total_salary: ℝ) : ℝ := 
  let num_employees := num_marketers + num_engineers + num_managers 
  if num_employees = 0 then 0 else num_engineers / num_employees * 100

theorem engineers_percentage_calculation : 
  let marketers_percentage := 0.7 
  let engineers_salary := 80000
  let average_salary := 80000
  let marketers_salary_total := 50000 * marketers_percentage 
  let managers_total_percent := 1 - marketers_percentage - x / 100
  let managers_salary := 370000 * managers_total_percent 
  marketers_salary_total + engineers_salary * x / 100 + managers_salary = average_salary -> 
  x = 22.76 
:= 
sorry

end engineers_percentage_calculation_l255_255298


namespace fifth_term_is_six_l255_255462

noncomputable def sequence (n : ℕ) (x y : ℝ) : ℝ :=
  match n with
  | 1 => x + 2*y
  | 2 => x - 2*y
  | 3 => x^2 * y
  | 4 => x / (2*y)
  | _ => 0 -- generic case

theorem fifth_term_is_six (x y : ℝ) (H1 : x^2 * y = x - 6*y) (H2 : x / (2*y) = x - 10*y) (H3 : x = 20 * (y^2) / (2*y - 1)) (H4 : y = 1) :
  sequence 5 x y = 6 :=
by
  sorry

end fifth_term_is_six_l255_255462


namespace highest_power_of_3_l255_255848

noncomputable def M : ℕ :=
  (List.foldl (λ acc x, acc * 100 + x) 0
     (List.map (λ n, n / 10 * 100 + n % 10)
     (List.range' 15 64)))

theorem highest_power_of_3 (j : ℕ) (h : 3^j ∣ M) : j = 0 :=
by sorry

end highest_power_of_3_l255_255848


namespace probability_twelfth_roll_last_l255_255844

/--
Calculate the probability that, when rolling a six-sided die until the same number appears on consecutive rolls, the 12th roll is the last roll.
-/
theorem probability_twelfth_roll_last : 
  (let p := (5/6 : ℝ) ^ 10 * (1/6 : ℝ) in p = (5^10 / 6^11 : ℝ)) :=
begin
  sorry
end

end probability_twelfth_roll_last_l255_255844


namespace lcm_factor_is_one_l255_255034

theorem lcm_factor_is_one
  (A B : ℕ)
  (hcf : A.gcd B = 42)
  (larger_A : A = 588)
  (other_factor : ∃ X, A.lcm B = 42 * X * 14) :
  ∃ X, X = 1 :=
  sorry

end lcm_factor_is_one_l255_255034


namespace odd_function_proof_l255_255700

-- Define the function f(x) and properties
def f (x : ℝ) : ℝ := if x > 0 then Real.log (x + 1) else -Real.log (1 - x)

-- Define the condition that the function is odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem odd_function_proof :
  (∀ x > 0, f x = Real.log (x + 1)) ∧ is_odd f →
  ∀ x < 0, f x = -Real.log(1 - x) := by
  intros h x hlt
  sorry

end odd_function_proof_l255_255700


namespace percent_of_employed_females_l255_255916

theorem percent_of_employed_females (p e m f : ℝ) (h1 : e = 0.60 * p) (h2 : m = 0.15 * p) (h3 : f = e - m):
  (f / e) * 100 = 75 :=
by
  -- We place the proof here
  sorry

end percent_of_employed_females_l255_255916


namespace first_year_with_digit_sum_seven_l255_255751

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_with_digit_sum_seven : ∃ y, y > 2023 ∧ sum_of_digits y = 7 ∧ ∀ z, z > 2023 ∧ z < y → sum_of_digits z ≠ 7 :=
by
  use 2032
  sorry

end first_year_with_digit_sum_seven_l255_255751


namespace find_angle_l255_255325

noncomputable theory

-- Definitions of triangle and their angles
def is_triangle (A B C : Type) (angle_B : Type) (A1 B1 C1 : Type) : Prop :=
angle_B = 120 ∧
-- Angle bisectors definitions
angle_bisectors A A1 ∧
angle_bisectors B B1 ∧
angle_bisectors C C1 ∧
-- Intersection point M
∃ M, (A1B1 ∩ CC1 = M)

-- Prove that \angle B_1 C_1 M = 60°
theorem find_angle (A B C A1 B1 C1 M : Type) (angle_B : Type) (h_triangle : is_triangle A B C angle_B A1 B1 C1) : 
angle B1 C1 M = 60 :=
sorry

end find_angle_l255_255325


namespace cone_surface_area_l255_255739

-- Given the central angle of the unfolded lateral surface of a cone is 90° and the radius is r,
-- the total surface area of the cone is 5πr²/16.

theorem cone_surface_area (r : ℝ) (h_angle : angle = 90) :
  let lateral_area := (1 / 4) * π * r^2,
      arc_length := (1 / 4) * (2 * π * r),
      r' := r / 4,
      base_area := π * (r / 4)^2
  in lateral_area + base_area = (5 * π * r^2) / 16 :=
sorry

end cone_surface_area_l255_255739


namespace inradius_circumradius_inequality_l255_255329

variable {R r a b c : ℝ}

def inradius (ABC : Triangle) := r
def circumradius (ABC : Triangle) := R
def side_a (ABC : Triangle) := a
def side_b (ABC : Triangle) := b
def side_c (ABC : Triangle) := c

theorem inradius_circumradius_inequality (ABC : Triangle) :
  R / (2 * r) ≥ (64 * a^2 * b^2 * c^2 / ((4 * a^2 - (b - c)^2) * (4 * b^2 - (c - a)^2) * (4 * c^2 - (a - b)^2)))^2 :=
sorry

end inradius_circumradius_inequality_l255_255329


namespace original_problem_condition1_variant_problem_condition2_l255_255784

variable {a b c : ℝ}

-- Define the constraints for abc = 9/4
def condition1 (a b c : ℝ) := 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 9 / 4

-- Define the constraints for abc = 2 (Jury's variant)
def condition2 (a b c : ℝ) := 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 2

-- Define the inequality statement
def inequality (a b c : ℝ) := a^3 + b^3 + c^3 > a * sqrt (b + c) + b * sqrt (c + a) + c * sqrt (a + b)

-- Original problem, constrained by condition1
theorem original_problem_condition1 (a b c : ℝ) (h : condition1 a b c) : inequality a b c := 
sorry

-- Jury's variant, constrained by condition2
theorem variant_problem_condition2 (a b c : ℝ) (h : condition2 a b c) : inequality a b c := 
sorry

end original_problem_condition1_variant_problem_condition2_l255_255784


namespace seq_pos_l255_255042

-- Definition of the sequence a
def seq (a : ℕ → ℝ) : Prop :=
  a 0 = -1 ∧
  ∀ n ≥ 1, (∑ k in finset.range (n + 1), a (n - k) / (k + 1)) = 0

-- Property to prove
theorem seq_pos (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, n ≥ 1 → a n > 0 :=
begin
  sorry
end

end seq_pos_l255_255042


namespace nicky_profit_l255_255820

theorem nicky_profit (value_traded_away value_received : ℤ)
  (h1 : value_traded_away = 2 * 8)
  (h2 : value_received = 21) :
  value_received - value_traded_away = 5 :=
by
  sorry

end nicky_profit_l255_255820


namespace terminating_decimal_n_div_450_l255_255193

-- Define the given problem constraints and prove the final solution
theorem terminating_decimal_n_div_450 :
  (∃ count, count = 166 ∧
   ∀ n, 1 ≤ n ∧ n ≤ 499 →
     (∃ k, k ∈ { x | 1 ≤ x ∧ x ≤ 166 }) → (450 / gcd n 450) = (2^a * 5^b) ∃ a b : ℕ) :=
begin
  sorry
end

end terminating_decimal_n_div_450_l255_255193


namespace number_of_possible_IDs_l255_255190

theorem number_of_possible_IDs : 
  ∃ (n : ℕ), 
  (∀ (a b : Fin 26) (x y : Fin 10),
    a = b ∨ x = y ∨ (a = b ∧ x = y) → 
    n = 9100) :=
sorry

end number_of_possible_IDs_l255_255190


namespace find_z_eq_neg_2i_l255_255222

variable {z : ℂ} (hi : Complex.I)
variable (pure_imaginary : ∃ b : ℝ, z = b * Complex.I)
variable (real_axis_condition : Complex.im ((z + 2) / (1 - hi)) = 0)

theorem find_z_eq_neg_2i : z = -2 * Complex.I :=
by
  sorry

end find_z_eq_neg_2i_l255_255222


namespace max_current_correct_l255_255139

noncomputable def electric_field_func (t : ℝ) : ℝ :=
  0.57 * Real.sin (1720 * t)

def area : ℝ := 1.0 -- m^2

def permittivity_free_space : ℝ := 8.85 * 10^(-12) -- F/m

def max_displacement_current : ℝ :=
  permittivity_free_space * 0.57 * 1720

theorem max_current_correct :
  max_displacement_current = 8.7 * 10^(-9) :=
by
  sorry

end max_current_correct_l255_255139


namespace determine_quadrant_find_m_and_sin_alpha_l255_255196

-- Definition of the problem conditions and the required proof
theorem determine_quadrant (α : ℝ) (h1 : 1 / |Real.sin α| = -1 / Real.sin α) (h2 : Real.log (Real.cos α) ≠ 0) :
  α ∈ set.Icc (3 * Real.pi / 2) (2 * Real.pi) ∨ α = 0 := sorry

theorem find_m_and_sin_alpha (m : ℝ) 
  (h1 : (3 / 5) ^ 2 + m ^ 2 = 1) 
  (h2 : m < 0) :
  m = -4 / 5 ∧ Real.sin (Real.atan2 m (3 / 5)) = -4 / 5 := sorry

end determine_quadrant_find_m_and_sin_alpha_l255_255196


namespace probability_of_odd_sum_given_even_product_l255_255668

-- Define a function to represent the probability of an event given the conditions
noncomputable def conditional_probability_odd_sum_even_product (dice : Fin 5 → Fin 8) : ℚ :=
  if h : (∃ i, (dice i).val % 2 = 0)  -- At least one die is even (product is even)
  then (1/2) / (31/32)  -- Probability of odd sum given even product
  else 0  -- If product is not even (not possible under conditions)

theorem probability_of_odd_sum_given_even_product :
  ∀ (dice : Fin 5 → Fin 8),
  conditional_probability_odd_sum_even_product dice = 16/31 :=
sorry  -- Proof omitted

end probability_of_odd_sum_given_even_product_l255_255668


namespace rectangle_bisectors_proof_l255_255760

theorem rectangle_bisectors_proof
(ABCD : Type)
[rectangle : rectangle ABCD]
(E D C : point ABCD)
(is_segment : is_segment C D)
(is_extension : is_extension_point D E CD)
(A B : point ABCD)
(K M : point ABCD)
(MK : segment M K)
(h_MK : length MK = 8)
(h_AB : length (segment A B) = 3)
(h_bisector_ABC : is_angle_bisector (angle B A C) K)
(h_bisector_ADE : is_angle_bisector (angle A D E) M)
: length (segment B C) = real.sqrt 55 := 
sorry

end rectangle_bisectors_proof_l255_255760


namespace average_age_of_omi_kimiko_arlette_l255_255823

theorem average_age_of_omi_kimiko_arlette (Kimiko Omi Arlette : ℕ) (hK : Kimiko = 28) (hO : Omi = 2 * Kimiko) (hA : Arlette = (3 * Kimiko) / 4) : 
  (Omi + Kimiko + Arlette) / 3 = 35 := 
by
  sorry

end average_age_of_omi_kimiko_arlette_l255_255823


namespace fruit_stand_shelves_l255_255104

theorem fruit_stand_shelves (n : ℕ) : 
  (∑ k in Finset.range n, 3 + k * 5) = 325 → n = 11 :=
by
  sorry

end fruit_stand_shelves_l255_255104


namespace card_draws_with_conditions_l255_255051

-- Definitions based on the conditions in the problem
structure Card :=
(color : Fin 3) -- Three colors: red (0), yellow (1), green (2)
(letter : Fin 5) -- Letters A, B, C, D, E

-- Helper function to count the number of ways to draw cards under given conditions
def valid_draws : Finset (Finset Card) :=
  { s : Finset Card | s.cardinality = 4 ∧ (∀ c : Fin 3, ∃ card ∈ s, card.color = c)
    ∧ function.injective (λ card, card.letter) }.to_finset

-- Main theorem statement
theorem card_draws_with_conditions :
  Fintype.card valid_draws = 360 :=
sorry

end card_draws_with_conditions_l255_255051


namespace Nicky_profit_l255_255817

-- Definitions for conditions
def card1_value : ℕ := 8
def card2_value : ℕ := 8
def received_card_value : ℕ := 21

-- The statement to be proven
theorem Nicky_profit : (received_card_value - (card1_value + card2_value)) = 5 :=
by
  sorry

end Nicky_profit_l255_255817


namespace expected_number_of_explorers_no_advice_l255_255592

-- Define the problem
theorem expected_number_of_explorers_no_advice
  (n : ℕ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∑ j in Finset.range n, (1 - p) ^ j) / p = (1 - (1 - p) ^ n) / p := by
  sorry

end expected_number_of_explorers_no_advice_l255_255592


namespace expected_no_advice_formula_l255_255596

noncomputable def expected_no_advice (n : ℕ) (p : ℝ) : ℝ :=
  ∑ j in Finset.range n, (1 - p) ^ j

theorem expected_no_advice_formula (n : ℕ) (p : ℝ) (hp : 0 ≤ p ∧ p < 1) : 
  expected_no_advice n p = (1 - (1 - p) ^ n) / p :=
by
  sorry

end expected_no_advice_formula_l255_255596


namespace unique_y_for_star_l255_255161

def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y - 3

theorem unique_y_for_star : (∀ y : ℝ, star 4 y = 17 → y = 0) ∧ (∃! y : ℝ, star 4 y = 17) := by
  sorry

end unique_y_for_star_l255_255161


namespace P_started_following_J_l255_255142

theorem P_started_following_J :
  ∀ (t : ℝ),
    (6 * 7.3 + 3 = 8 * (7.3 - t)) → t = 1.45 → t + 12 = 13.45 :=
by
  sorry

end P_started_following_J_l255_255142


namespace bookshop_customers_l255_255429

theorem bookshop_customers
  (p : ℕ) (c : ℕ) (k : ℚ)
  (h_k : p * c = k)
  (h_initial : p = 40)
  (h_initial_cost : c = 30)
  (h_new_cost : 45)
  (h_promo : new_p_base = k / 45)
  (new_p_base : ℚ)
  (new_p : ℚ := 1.1 * new_p_base) :
  p = 29 := by
  sorry

end bookshop_customers_l255_255429


namespace probability_no_shaded_rectangle_l255_255532

theorem probability_no_shaded_rectangle :
  let n := (1002 * 1001) / 2
  let m := 501 * 501
  (1 - (m / n) = 500 / 1001) := sorry

end probability_no_shaded_rectangle_l255_255532


namespace min_value_of_f_l255_255711

noncomputable def φ_bound := (real.pi / 2) 

def shifted_odd (φ : ℝ) (k : ℤ) : Prop :=
  (abs φ < φ_bound) ∧ (φ + real.pi / 3 = k * real.pi)

def f (x φ : ℝ) : ℝ := real.sin (2 * x + φ)

theorem min_value_of_f (φ : ℝ) (h_shifted_odd : ∃ (k : ℤ), shifted_odd φ k) :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ real.pi / 2 ∧ f 0 (-real.pi / 3) = -real.sqrt 3 / 2 :=
sorry

end min_value_of_f_l255_255711


namespace simplify_expression_l255_255529

theorem simplify_expression : (Real.sqrt 12 - |1 - Real.sqrt 3| + (7 + Real.pi)^0) = (Real.sqrt 3 + 2) :=
by
  sorry

end simplify_expression_l255_255529


namespace max_profit_at_300_l255_255545

noncomputable def revenue (x : ℕ) : ℝ :=
  if x ≤ 400 then 400 * x - (1 / 2) * x ^ 2 else 80000

noncomputable def total_cost (x : ℕ) : ℝ :=
  20000 + 100 * x

noncomputable def profit (x : ℕ) : ℝ :=
  revenue x - total_cost x

theorem max_profit_at_300 : ∃ x, 0 ≤ x ∧ profit x = 25000 ∧ ∀ y, profit y ≤ 25000 :=
begin
  use 300,
  have h1 : profit 300 = 25000, sorry,
  exact ⟨0.le, h1, sorry⟩,
end

end max_profit_at_300_l255_255545


namespace amoeba_population_doubles_l255_255138

theorem amoeba_population_doubles (initial_amoebas : ℕ) (days : ℕ) :
  initial_amoebas = 1 →
  days = 7 →
  (∀ n : ℕ, n > 0 → amoebas_on_day (initial_amoebas : ℕ) (n : ℕ) = initial_amoebas * 2 ^ (n - 1)) →
  amoebas_on_day initial_amoebas days = 64 :=
by
  intros h_initial h_days h_population
  simp [h_initial, h_days]
  -- Proof skipped with sorry
  sorry

noncomputable def amoebas_on_day (initial : ℕ) (n : ℕ) : ℕ :=
  initial * 2 ^ (n - 1)

end amoeba_population_doubles_l255_255138


namespace platform_length_l255_255083

theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (cross_time_sec : ℝ) :
  train_length = 225 → train_speed_kmph = 90 → cross_time_sec = 25 → 
  let train_speed_mps := train_speed_kmph * 1000 / 3600 in
  let total_distance := train_speed_mps * cross_time_sec in
  total_distance - train_length = 400 :=
by
  sorry

end platform_length_l255_255083


namespace least_distinct_values_l255_255120

theorem least_distinct_values (n : ℕ) (m : ℕ) (k : ℕ) (h1 : n = 3000) (h2 : m = 15) (h3 : k = (n - m)) :
  ∃ x, (x ≥ (k / 14 + 1)) ∧ (x = 215) :=
by
  -- Assume n = 3000, m = 15, k = n - m
  intros,
  let x := 215, -- Calculated x
  use x,
  sorry -- Proof omitted

end least_distinct_values_l255_255120


namespace option_b_correct_option_c_correct_option_d_correct_l255_255290

def half_increasing_difference (a : ℕ → ℝ) :=
∀ n : ℕ, n ≥ 2 → a n - (1 / 2) * a (n - 1) < a (n + 1) - (1 / 2) * a n

theorem option_b_correct (q : ℝ) (h : q > 1) :
  half_increasing_difference (λ n, q ^ n) :=
sorry

theorem option_c_correct (a : ℕ → ℝ) (d : ℝ) (h : d > 0) (h_arith : ∀ n : ℕ, a (n + 1) = a n + d) :
  half_increasing_difference a :=
sorry

theorem option_d_correct (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ)
  (h_half_increasing : half_increasing_difference a)
  (h_sum : ∀ n : ℕ, S n = 2 * a n - 2^(n + 1) - t) :
  t > -32 / 3 :=
sorry

end option_b_correct_option_c_correct_option_d_correct_l255_255290


namespace min_of_sin_cub_cos_cub_l255_255625

open Real Trig

theorem min_of_sin_cub_cos_cub :
  ∃ x : ℝ, sin x ^ 3 + 2 * cos x ^ 3 = -4 * real.sqrt 2 / 3 :=
sorry

end min_of_sin_cub_cos_cub_l255_255625


namespace divide_8x8_into_7_equal_perimeter_figures_l255_255326

theorem divide_8x8_into_7_equal_perimeter_figures :
  ∃ (figures : Finset (Finset (ℕ × ℕ))), 
  (∀ f ∈ figures, ∃ p : ℕ, (∑ p in f, 1) = 64 / 7 ∧ -- Equal area constraint
   (∑ p in f, borde_length p) = 32 / 7) -- Equal perimeter constraint
    ∧ (Union figures = Finset.univ) := -- The figures cover the entire grid
sorry

end divide_8x8_into_7_equal_perimeter_figures_l255_255326


namespace sector_area_proof_l255_255230

-- Define the central angle θ and the arc length L
def θ : ℝ := 120 * Real.pi / 180  -- convert degrees to radians
def L : ℝ := 6 * Real.pi

-- Define the radius r, note that 6 * Real.pi simplifies as 6π term
def r : ℝ := L * 3 / (2 * θ)  -- rearranging arc length formula to solve for r

-- Define the area of the sector
def S_sector : ℝ := θ / (2 * Real.pi) * Real.pi * r^2

-- Prove that the area of the sector is 27π
theorem sector_area_proof : S_sector = 27 * Real.pi := by
  sorry

end sector_area_proof_l255_255230


namespace tan_six_minus_tan_two_geo_seq_l255_255273

theorem tan_six_minus_tan_two_geo_seq (x : ℝ) (h : (cos x)^2 = (sin x) * (cos x) * (cot x)) :
  (tan x)^6 - (tan x)^2 = (sin x)^2 :=
sorry

end tan_six_minus_tan_two_geo_seq_l255_255273


namespace chess_tournament_green_teams_l255_255299

theorem chess_tournament_green_teams :
  ∀ (R G total_teams : ℕ)
  (red_team_count : ℕ → ℕ)
  (green_team_count : ℕ → ℕ)
  (mixed_team_count : ℕ → ℕ),
  R = 64 → G = 68 → total_teams = 66 →
  red_team_count R = 20 →
  (R + G = 132) →
  -- Details derived from mixed_team_count and green_team_count
  -- are inferred from the conditions provided
  mixed_team_count R + red_team_count R = 32 → 
  -- Total teams by definition including mixed teams 
  mixed_team_count G = G - (2 * red_team_count R) - green_team_count G →
  green_team_count (G - (mixed_team_count R)) = 2 → 
  2 * (green_team_count G) = 22 :=
by sorry

end chess_tournament_green_teams_l255_255299


namespace expected_no_advice_l255_255598

theorem expected_no_advice {n : ℕ} (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  (Σ (j : ℕ) (hj : j < n), (1 - p)^j) = (1 - (1 - p)^n) / p :=
by
  sorry

end expected_no_advice_l255_255598


namespace winner_last_year_ounces_l255_255490

/-- Definition of the problem conditions -/
def ouncesPerHamburger : ℕ := 4
def hamburgersTonyaAte : ℕ := 22

/-- Theorem stating the desired result -/
theorem winner_last_year_ounces :
  hamburgersTonyaAte * ouncesPerHamburger = 88 :=
by
  sorry

end winner_last_year_ounces_l255_255490


namespace minimum_odd_in_A_P_l255_255376

-- Define the polynomial and its properties
def polynomial_degree8 (P : ℝ[X]) : Prop :=
  P.degree = 8

-- Define the set A_P
def A_P (P : ℝ[X]) : set ℝ :=
  {x | P.eval x = 8}

-- Main theorem statement
theorem minimum_odd_in_A_P (P : ℝ[X]) (hP : polynomial_degree8 P) : ∃ n : ℕ, n = 1 ∧ ∃ x ∈ A_P P, odd x :=
sorry

end minimum_odd_in_A_P_l255_255376


namespace cost_of_toast_l255_255585

theorem cost_of_toast (egg_cost : ℕ) (toast_cost : ℕ)
  (dale_toasts : ℕ) (dale_eggs : ℕ)
  (andrew_toasts : ℕ) (andrew_eggs : ℕ)
  (total_cost : ℕ)
  (h1 : egg_cost = 3)
  (h2 : dale_toasts = 2)
  (h3 : dale_eggs = 2)
  (h4 : andrew_toasts = 1)
  (h5 : andrew_eggs = 2)
  (h6 : 2 * toast_cost + dale_eggs * egg_cost 
        + andrew_toasts * toast_cost + andrew_eggs * egg_cost = total_cost) :
  total_cost = 15 → toast_cost = 1 :=
by
  -- Proof not needed
  sorry

end cost_of_toast_l255_255585


namespace determine_numbers_l255_255478

theorem determine_numbers (a b c : ℕ) (h₁ : a + b + c = 15) 
  (h₂ : (1 / (a : ℝ)) + (1 / (b : ℝ)) + (1 / (c : ℝ)) = 71 / 105) : 
  (a = 3 ∧ b = 5 ∧ c = 7) ∨ (a = 3 ∧ b = 7 ∧ c = 5) ∨ (a = 5 ∧ b = 3 ∧ c = 7) ∨ 
  (a = 5 ∧ b = 7 ∧ c = 3) ∨ (a = 7 ∧ b = 3 ∧ c = 5) ∨ (a = 7 ∧ b = 5 ∧ c = 3) :=
sorry

end determine_numbers_l255_255478


namespace sphere_radius_l255_255552

theorem sphere_radius (tree_height sphere_shadow tree_shadow : ℝ) 
  (h_tree_shadow_pos : tree_shadow > 0) 
  (h_sphere_shadow_pos : sphere_shadow > 0) 
  (h_tree_height_pos : tree_height > 0)
  (h_tangent : (tree_height / tree_shadow) = (sphere_shadow / 15)) : 
  sphere_shadow = 11.25 :=
by
  sorry

end sphere_radius_l255_255552


namespace tim_watched_total_hours_tv_l255_255486

-- Define the conditions
def short_show_episodes : ℕ := 24
def short_show_duration_per_episode : ℝ := 0.5

def long_show_episodes : ℕ := 12
def long_show_duration_per_episode : ℝ := 1

-- Define the total duration for each show
def short_show_total_duration : ℝ :=
  short_show_episodes * short_show_duration_per_episode

def long_show_total_duration : ℝ :=
  long_show_episodes * long_show_duration_per_episode

-- Define the total TV hours watched
def total_tv_hours_watched : ℝ :=
  short_show_total_duration + long_show_total_duration

-- Write the theorem statement
theorem tim_watched_total_hours_tv : total_tv_hours_watched = 24 := 
by
  -- proof goes here
  sorry

end tim_watched_total_hours_tv_l255_255486


namespace points_on_same_circle_l255_255211

section ProofCircle
variables {n : ℕ} {S : fin n → ℤ × ℤ} {O : ℤ × ℤ}
open_locale big_operators

-- Define that any movement of any point to the position of any other point results in the same set
def movement_invariance (S : fin n → ℤ × ℤ) :=
∀ (i j : fin n), ∃ f : fin n → fin n, (∀ k, S (f k) = S k)

-- Define centroid O of the set S
def centroid (S : fin n → ℤ × ℤ) (O : ℤ × ℤ) := 
∑ i, S i / n = O

-- Statement to prove that all points in the set S lie on a circle centered at O
theorem points_on_same_circle (h : movement_invariance S) : ∃ r : ℕ, ∀ i, dist (S i) O = r  :=
sorry

end ProofCircle

end points_on_same_circle_l255_255211


namespace min_odd_numbers_in_A_P_l255_255358

theorem min_odd_numbers_in_A_P (P : ℝ → ℝ) 
  (hP : ∃ a8 a7 a6 a5 a4 a3 a2 a1 a0 : ℝ, P = λ x, a8 * x^8 + a7 * x^7 + a6 * x^6 + a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0)
  (h8_in_A_P : ∃ x : ℝ, P x = 8) : 
  (∀ x : ℝ, P x ∈ Set.A_P) → 
  (∃! y : ℝ, odd y ∧ y ∈ Set.A_P) := 
sorry

end min_odd_numbers_in_A_P_l255_255358


namespace sum_of_digits_of_palindromes_l255_255156

theorem sum_of_digits_of_palindromes :
  (∑ a in Finset.range 10 \ {0}, ∑ b in Finset.range 10 \ {0}, 
    ((1001 * a + 110 * b) : ℕ)) % 9 = 36 :=
by
  sorry

end sum_of_digits_of_palindromes_l255_255156


namespace sqrt_x_minus_1_same_type_as_sqrt_18_l255_255743

theorem sqrt_x_minus_1_same_type_as_sqrt_18 (x : ℝ) : 
  sqrt (x - 1) = sqrt 18 → x = 3 :=
by 
  intro h
  have h1 : sqrt 18 = 3 * sqrt 2 := 
    by sorry -- Simplify sqrt(18) correctly here.
  rw h1 at h
  have h2 : sqrt (x - 1) = 3 * sqrt 2 := 
    by rw [h]
  sorry -- Show that x - 1 = 2, therefore x = 3.

end sqrt_x_minus_1_same_type_as_sqrt_18_l255_255743


namespace ratio_areas_APQ_PQC_l255_255306

section areas
variables {A B C D P Q : Type} [has_area : has_area (parallelogram A B C D)]

-- Condition 1: ABCD is a parallelogram
def parallelogram (A B C D : Type) := (A B C D)

-- Condition 2: Point P on BC such that 3PB = 2PC
def pointP_on_BC (P : Type) (B C : Type) (ratio_pb_pc : 3 = 2) := P

-- Condition 3: Point Q on CD such that 4CQ = 5QD
def pointQ_on_CD (Q : Type) (C D : Type) (ratio_cq_qd : 4 = 5) := Q

-- The ratio of the area of triangle APQ to the area of triangle PQC
theorem ratio_areas_APQ_PQC (h1 : parallelogram A B C D) 
  (h2 : pointP_on_BC P B C 3 2) 
  (h3 : pointQ_on_CD Q C D 4 5) : 
  area_triangle A P Q / area_triangle P Q C = 37 / 15 := 
sorry
end areas

end ratio_areas_APQ_PQC_l255_255306


namespace arithmetic_sequence_eq_x_l255_255708

theorem arithmetic_sequence_eq_x
  (x : ℝ)
  (h1 : (1 + (real.sqrt 3)))
  (h2 : x)
  (h3 : (1 - (real.sqrt 3)))
  (h_arith_seq : 2 * x = (1 + (real.sqrt 3)) + (1 - (real.sqrt 3))) :
  x = 1 :=
by
  sorry

end arithmetic_sequence_eq_x_l255_255708


namespace license_plates_count_l255_255115

noncomputable def number_of_distinct_license_plates : ℕ :=
  let digits_choices := 10^5
  let letter_block_choices := 26^3
  let positions_choices := 6
  positions_choices * digits_choices * letter_block_choices

theorem license_plates_count : number_of_distinct_license_plates = 105456000 := by
  unfold number_of_distinct_license_plates
  calc
    6 * 10^5 * 26^3 = 6 * 100000 * 17576 : by norm_num
                  ... = 105456000 : by norm_num
  sorry

end license_plates_count_l255_255115


namespace Jaden_estimate_larger_l255_255305

variable (p q δ γ : ℝ)
variable (hpq : p > q)
variable (hq0 : q > 0)
variable (hδγ : δ > γ)
variable (hγ0 : γ > 0)

theorem Jaden_estimate_larger : (p + δ) - (q - γ) > p - q := 
by
  have : δ > 0 := lt_trans hγ0 hδγ
  calc
    (p + δ) - (q - γ) = p - q + δ + γ : by ring
    ... > p - q : add_pos_of_pos_of_nonneg this (le_of_lt hγ0)

end Jaden_estimate_larger_l255_255305


namespace matrix_product_l255_255147

open Matrix

def sequence_of_matrices : List (Matrix (Fin 2) (Fin 2) ℤ) :=
  List.map (λ n => ![![1, 4*n + 2], ![0, 1]]) (List.range 50)

theorem matrix_product :
  List.foldl (λ (acc : Matrix (Fin 2) (Fin 2) ℤ) (m : Matrix (Fin 2) (Fin 2) ℤ) => acc ⬝ m) 1 sequence_of_matrices
  = ![![1, 5000], ![0, 1]] :=
by
  sorry

end matrix_product_l255_255147


namespace marble_count_l255_255114

theorem marble_count (x : ℝ) (h1 : 0.5 * x = x / 2) (h2 : 0.25 * x = x / 4) (h3 : (27 + 14) = 41) (h4 : 0.25 * x = 27 + 14)
  : x = 164 :=
by
  sorry

end marble_count_l255_255114


namespace profit_percentage_equals_neg_1_52_l255_255561

noncomputable def calculate_loss_percentage
  (marked_price : ℝ)
  (number_of_pens : ℕ)
  (purchase_price : ℝ)
  (tax_rate : ℝ)
  (shipping_fee : ℝ)
  (discount_rate : ℝ)
  : ℝ :=
let cost_price := purchase_price + (tax_rate * purchase_price) + shipping_fee in
let selling_price_per_pen := marked_price - (discount_rate * marked_price) in
let total_selling_price := (number_of_pens : ℝ) * selling_price_per_pen in
let profit := total_selling_price - cost_price in
(profit / cost_price) * 100

theorem profit_percentage_equals_neg_1_52
  : calculate_loss_percentage 1 500 450 0.05 20 0.03 ≈ -1.52 := sorry

end profit_percentage_equals_neg_1_52_l255_255561


namespace larger_root_exceeds_smaller_root_by_5_5_l255_255982

theorem larger_root_exceeds_smaller_root_by_5_5 : 
  ∀ (q : ℝ), 2*q^2 + 5*q = 12 → 
  let q1 := ((-5 + Real.sqrt(25 + 96)) / 4) in
  let q2 := ((-5 - Real.sqrt(25 + 96)) / 4) in
  q1 - q2 = 5.5 :=
by
  sorry

end larger_root_exceeds_smaller_root_by_5_5_l255_255982


namespace metal_mixture_ratio_l255_255772

theorem metal_mixture_ratio :
  ∃ (x y z : ℕ), 
  68 * x + 96 * y + 110 * z = 72 * (x + y + z) ∧ 
  x : y : z = 6 : 1 : 0 :=
sorry

end metal_mixture_ratio_l255_255772


namespace solve_sin_exp_ineq_l255_255655

noncomputable def sin_exp_ineq (x : ℝ) : Prop :=
  (sin x) ^ 2018 + (cos x) ^ -2019 ≤ (cos x) ^ 2018 + (sin x) ^ -2019

theorem solve_sin_exp_ineq :
  ∀ x ∈ Set.Icc (- (5 * Real.pi) / 4) ((3 * Real.pi) / 4),
  sin_exp_ineq x ↔
    x ∈ (Set.Ico (- (5 * Real.pi) / 4) (- Real.pi))
      ∪ (Set.Ico (- (3 * Real.pi) / 4) (- (Real.pi / 2)))
      ∪ (Set.Ioc 0 (Real.pi / 4))
      ∪ (Set.Ioc (Real.pi / 2) (3 * Real.pi / 4)) :=
  by
  sorry

end solve_sin_exp_ineq_l255_255655


namespace find_sector_perimeter_l255_255706

-- Definition of the input values as per conditions given
def theta : ℝ := 54 * (Real.pi / 180)  -- converting degrees to radians
def r : ℝ := 20  -- radius in cm

-- Define the formula for the arc length
def arc_length (theta r : ℝ) : ℝ := (theta / (2 * Real.pi)) * (2 * Real.pi * r)

-- Define the formula for perimeter
def perimeter_sector (arc_length : ℝ) (radius : ℝ) : ℝ := arc_length + 2 * radius

-- The theorem statement
theorem find_sector_perimeter : perimeter_sector (arc_length theta r) r = 6 * Real.pi + 40 := by
  sorry

end find_sector_perimeter_l255_255706


namespace y_coord_vertex_C_l255_255426

/-- The coordinates of vertices A, B, and D are given as A(0,0), B(0,1), and D(3,1).
 Vertex C is directly above vertex B. The quadrilateral ABCD has a vertical line of symmetry 
 and the area of quadrilateral ABCD is 18 square units.
 Prove that the y-coordinate of vertex C is 11. -/
theorem y_coord_vertex_C (h : ℝ) 
  (A : ℝ × ℝ := (0, 0)) 
  (B : ℝ × ℝ := (0, 1)) 
  (D : ℝ × ℝ := (3, 1)) 
  (C : ℝ × ℝ := (0, h)) 
  (symmetry : C.fst = B.fst) 
  (area : 18 = 3 * 1 + (1 / 2) * 3 * (h - 1)) :
  h = 11 := 
by
  sorry

end y_coord_vertex_C_l255_255426


namespace inequality_proof_l255_255808

variable (a b c λ μ : ℝ)
variable (m n : ℕ)

theorem inequality_proof
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hλ : 0 < λ) (hμ : 0 < μ)
  (sum_eq_one : a + b + c = 1)
  (hm : 2 ≤ m) (hn : 2 ≤ n) :
  (a ^ m + b ^ n) / (λ * b + μ * c) +
  (b ^ m + c ^ n) / (λ * c + μ * a) +
  (c ^ m + a ^ n) / (λ * a + μ * b) ≥
  (3^ (2 - m) + 3^ (2 - n)) / (λ + μ) :=
by
  sorry

end inequality_proof_l255_255808


namespace units_digit_35_pow_35_mul_17_pow_17_l255_255184

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_35_pow_35_mul_17_pow_17:
  units_digit (35 ^ (35 * 17 ^ 17)) = 5 := 
by {
  -- Here we're skipping the proof.
  sorry
}

end units_digit_35_pow_35_mul_17_pow_17_l255_255184


namespace monotonicity_of_f_range_of_a_for_two_zeros_l255_255715

noncomputable def f (x a : ℝ) := Real.exp x - a * (x + 2)

theorem monotonicity_of_f (a : ℝ) (h : a = 1) :
  (∀ x, x < 0 → deriv (λ x, f x 1) x < 0) ∧
  (∀ x, x > 0 → deriv (λ x, f x 1) x > 0) :=
sorry

theorem range_of_a_for_two_zeros (a : ℝ) (h_has_two_zeros : ∃ x1 x2, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) :
  a > (1 / Real.exp 1) :=
sorry

end monotonicity_of_f_range_of_a_for_two_zeros_l255_255715


namespace sin_cos_correct_statements_l255_255202

theorem sin_cos_correct_statements :
  let y := λ x : ℝ, Real.sin x + Real.cos x
  let statement1 := ∀ x ∈ Set.Icc 0 Real.pi, y x ∈ Set.Icc 1 (Real.sqrt 2)
  let statement2 := ∀ x, y x = y (Real.pi / 2 - x)
  let statement3 := ∀ x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4), MonotoneOn y (Set.Icc (Real.pi / 4) (5 * Real.pi / 4))
  let statement4 := ∀ x, y x = Real.sqrt 2 * Real.cos (x - Real.pi / 4)
  (¬ statement1) ∧ statement2 ∧ (¬ statement3) ∧ statement4 :=
by sorry

end sin_cos_correct_statements_l255_255202


namespace sum_of_odd_indexed_terms_of_geometric_series_l255_255321

theorem sum_of_odd_indexed_terms_of_geometric_series :
  ∃ a_n : ℕ → ℝ, 
    a_n 1 = sqrt 3 ∧
    a_n 2 = 1 ∧
    (∀ (ε : ℝ) (ε > 0), ∃ N : ℕ, ∀ n ≥ N, abs (a_n - (3 * sqrt 3 / 2)) < ε :=
begin
  sorry
end

end sum_of_odd_indexed_terms_of_geometric_series_l255_255321


namespace largest_n_integer_expr_l255_255179

theorem largest_n_integer_expr (n : ℕ) :
  (∃ k, 2n - 1 = 3^k ∨
   (k ∣ n - 2 ∧ k ∣ n + 1 ∧ k = 3)) →
  (3 ∣ 2 * n - 1) →
  (n = 14) :=
sorry

end largest_n_integer_expr_l255_255179


namespace remainder_N_mod_1000_l255_255335

def base3_more_ones (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), digits.length ≤ 7 ∧ (∀ d ∈ digits, d < 3) ∧ (
    let ones := digits.count (λ d => d = 1)
    in let others := digits.count (λ d => d ≠ 1)
    in ones > others
  )

noncomputable def N : ℕ :=
  (List.range 5001).count (λ n => base3_more_ones n)

theorem remainder_N_mod_1000 : N % 1000 = 379 := 
by sorry

end remainder_N_mod_1000_l255_255335


namespace simplify_radicals_l255_255008

theorem simplify_radicals :
  (sqrt (sqrt[3] (sqrt (sqrt (1 / 512))))) = (1 / 2^(1/4)) :=
by
  sorry

end simplify_radicals_l255_255008


namespace find_vasya_floor_l255_255509

variable steps_to_3rd_floor : ℕ
variable vasya_steps : ℕ

-- Given conditions
def steps_per_floor(steps_to_3rd_floor : ℕ) : ℕ := steps_to_3rd_floor / 2
def vasya_floors(steps_per_floor : ℕ) (vasya_steps : ℕ) : ℕ := vasya_steps / steps_per_floor

-- The proof problem
theorem find_vasya_floor (h1 : steps_to_3rd_floor = 36) (h2 : vasya_steps = 72) :
  vasya_floors (steps_per_floor steps_to_3rd_floor) vasya_steps + 1 = 5 :=
by
  sorry

end find_vasya_floor_l255_255509


namespace max_m_sq_plus_n_sq_l255_255697

theorem max_m_sq_plus_n_sq (m n : ℤ) (h1 : 1 ≤ m ∧ m ≤ 1981) (h2 : 1 ≤ n ∧ n ≤ 1981) (h3 : (n^2 - m*n - m^2)^2 = 1) : m^2 + n^2 ≤ 3524578 :=
sorry

end max_m_sq_plus_n_sq_l255_255697


namespace matrix_problem_l255_255340

theorem matrix_problem (a b c d : ℤ) (h : (matrix.mul 
  (matrix.ofFunction (λ i j, if i = 0 ∧ j = 0 then a else if i = 0 ∧ j = 1 then b else if i = 1 ∧ j = 0 then c else d)) 
  (matrix.ofFunction (λ i j, if i = 0 ∧ j = 0 then a else if i = 0 ∧ j = 1 then b else if i = 1 ∧ j = 0 then c else d))) 
  = matrix.ofFunction (λ i j, if i = 0 ∧ j = 0 then 9 else if i = 0 ∧ j = 1 then 0 else if i = 1 ∧ j = 0 then 0 else 9)) :
  ∃ (a b c d : ℤ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∧ |a| + |b| + |c| + |d| = 8 :=
by
  sorry

end matrix_problem_l255_255340


namespace round_to_nearest_whole_l255_255838

theorem round_to_nearest_whole (x : ℝ) (h : x = 7523.4987) : Int.round x = 7523 :=
by
  rw [h]
  norm_num

end round_to_nearest_whole_l255_255838


namespace minimum_odd_numbers_in_set_l255_255388

-- Definitions
variable (P : ℝ → ℝ)
variable (degree_P : ℕ)
variable (A_P : Set ℝ)

-- The conditions: P is a polynomial of degree 8, and 8 is included in A_P
def is_polynomial_of_degree_eight (P : ℝ → ℝ) (degree_P : ℕ) : Prop :=
  degree_P = 8

def set_includes_eight (A_P : Set ℝ) : Prop := 
  8 ∈ A_P

-- The goal: prove the minimum number of odd numbers in A_P is 1
theorem minimum_odd_numbers_in_set {P : ℝ → ℝ} {degree_P : ℕ} {A_P : Set ℝ} :
  is_polynomial_of_degree_eight P degree_P → 
  set_includes_eight A_P → 
  ∃ odd_numbers : ℕ, odd_numbers = 1 :=
sorry

end minimum_odd_numbers_in_set_l255_255388


namespace melted_ice_cream_depth_l255_255127

theorem melted_ice_cream_depth :
  ∀ (r_sphere r_cylinder : ℝ) (h : ℝ),
    r_sphere = 3 →
    r_cylinder = 10 →
    (4 / 3) * π * r_sphere^3 = 100 * π * h →
    h = 9 / 25 :=
  by
    intros r_sphere r_cylinder h
    intros hr_sphere hr_cylinder
    intros h_volume_eq
    sorry

end melted_ice_cream_depth_l255_255127


namespace find_parallelogram_base_length_l255_255657

variable (A h b : ℕ)
variable (parallelogram_area : A = 240)
variable (parallelogram_height : h = 10)
variable (area_formula : A = b * h)

theorem find_parallelogram_base_length : b = 24 :=
by
  have h₁ : A = 240 := parallelogram_area
  have h₂ : h = 10 := parallelogram_height
  have h₃ : A = b * h := area_formula
  sorry

end find_parallelogram_base_length_l255_255657


namespace smallest_m_for_root_of_unity_l255_255796

def complex_in_T (z : ℂ) : Prop :=
  let x := z.re
  let y := z.im
  (real.sqrt 3 / 2) ≤ x ∧ x ≤ (2 / real.sqrt 3)

theorem smallest_m_for_root_of_unity :
  ∃ m : ℕ, ∀ n : ℕ, n ≥ m → (∃ z : ℂ, complex_in_T z ∧ z ^ n = 1) :=
begin
  use 18,
  sorry
end

end smallest_m_for_root_of_unity_l255_255796


namespace range_of_x_l255_255316

theorem range_of_x (x : ℝ) : (sqrt ((x - 1) / (x - 2)) : ℝ) ≥ 0 → (x > 2 ∨ x ≤ 1) :=
by
  sorry

end range_of_x_l255_255316


namespace gasoline_amount_added_l255_255098

noncomputable def initial_fill (capacity : ℝ) : ℝ := (3 / 4) * capacity
noncomputable def final_fill (capacity : ℝ) : ℝ := (9 / 10) * capacity
noncomputable def gasoline_added (capacity : ℝ) : ℝ := final_fill capacity - initial_fill capacity

theorem gasoline_amount_added :
  ∀ (capacity : ℝ), capacity = 24 → gasoline_added capacity = 3.6 :=
  by
    intros capacity h
    rw [h]
    have initial_fill_24 : initial_fill 24 = 18 := by norm_num [initial_fill]
    have final_fill_24 : final_fill 24 = 21.6 := by norm_num [final_fill]
    have gasoline_added_24 : gasoline_added 24 = 3.6 :=
      by rw [gasoline_added, initial_fill_24, final_fill_24]; norm_num
    exact gasoline_added_24

end gasoline_amount_added_l255_255098


namespace root_range_of_quadratic_eq_l255_255459

theorem root_range_of_quadratic_eq (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 < x2 ∧ x1^2 + k * x1 - k = 0 ∧ x2^2 + k * x2 - k = 0 ∧ 1 < x1 ∧ x1 < 2 ∧ 2 < x2 ∧ x2 < 3) ↔  (-9 / 2) < k ∧ k < -4 :=
by
  sorry

end root_range_of_quadratic_eq_l255_255459


namespace min_odd_numbers_in_A_P_l255_255353

theorem min_odd_numbers_in_A_P (P : ℝ → ℝ) 
  (hP : ∃ a8 a7 a6 a5 a4 a3 a2 a1 a0 : ℝ, P = λ x, a8 * x^8 + a7 * x^7 + a6 * x^6 + a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0)
  (h8_in_A_P : ∃ x : ℝ, P x = 8) : 
  (∀ x : ℝ, P x ∈ Set.A_P) → 
  (∃! y : ℝ, odd y ∧ y ∈ Set.A_P) := 
sorry

end min_odd_numbers_in_A_P_l255_255353


namespace range_of_m_l255_255015

noncomputable def p (m : ℝ) : Prop :=
  1^2 + 1^2 - 2*m*1 + 2*m*1 + 2*m^2 - 4 < 0

noncomputable def q (m : ℝ) : Prop :=
  ∀ k : ℝ, (mx - y + 1 + 2m = 0) → mx ≤ 0 ∨ y ≥ 0

theorem range_of_m (m : ℝ) (h_p_or_q : p m ∨ q m) (h_p_and_q : ¬(p m ∧ q m)) : 
  (-1 < m ∧ m < 0) ∨ (m ≥ 1) :=
sorry

end range_of_m_l255_255015


namespace crackers_initial_l255_255406

variables (cakes friends items : ℕ)

theorem crackers_initial (h1 : cakes = 21) (h2 : friends = 7) (h3 : items = 5 * friends) : 
  let cakes_per_friend := cakes / friends in
  let total_items := friends * items in
  let crackers_per_friend := items - cakes_per_friend in
  let total_cakes := friends * cakes_per_friend in
  let total_crackers := total_items - total_cakes in
  let initial_crackers := total_crackers + (friends * crackers_per_friend) in
  initial_crackers = 28 :=
by
  sorry

end crackers_initial_l255_255406


namespace period_of_sum_l255_255505

noncomputable def period_cos_3x : ℝ := 2 * π / 3
noncomputable def period_sin_6x : ℝ := π / 3

theorem period_of_sum (x : ℝ) :
  (cos (3 * x) + sin (6 * x)) = (cos (3 * (x + period_cos_3x)) + sin (6 * (x + period_cos_3x))) :=
by
  sorry

end period_of_sum_l255_255505


namespace license_plates_count_l255_255118

theorem license_plates_count : (6 * 10^5 * 26^3) = 10584576000 := by
  sorry

end license_plates_count_l255_255118


namespace value_calculation_l255_255931

-- Definition of constants used in the problem
def a : ℝ := 1.3333
def b : ℝ := 3.615
def expected_value : ℝ := 4.81998845

-- The proposition to be proven
theorem value_calculation : a * b = expected_value :=
by sorry

end value_calculation_l255_255931


namespace pete_total_blocks_traveled_l255_255832

theorem pete_total_blocks_traveled : 
    ∀ (walk_to_garage : ℕ) (bus_to_post_office : ℕ), 
    walk_to_garage = 5 → bus_to_post_office = 20 → 
    ((walk_to_garage + bus_to_post_office) * 2) = 50 :=
by
  intros walk_to_garage bus_to_post_office h_walk h_bus
  sorry

end pete_total_blocks_traveled_l255_255832


namespace distribution_of_letters_l255_255164

theorem distribution_of_letters (l m : ℕ) (condition1 : l = 5) (condition2 : m = 3) :
  ∃ n, n = 150 ∧ (
    ∀ a1 a2 a3 : ℕ, 
    a1 + a2 + a3 = l →
    a1 > 0 ∧ a2 > 0 ∧ a3 > 0 →
    (∏(x : ℕ) in {a1, a2, a3}, nat.choose l x) * nat.factorial m = n
  ) :=
sorry

end distribution_of_letters_l255_255164


namespace find_a_l255_255258

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x₀ a : ℝ) (h : f x₀ a - g x₀ a = 3) : a = -Real.log 2 - 1 :=
by
  sorry

end find_a_l255_255258


namespace determine_phi_l255_255032

def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)
def g (x : ℝ) : ℝ := Real.cos (2 * x)

theorem determine_phi (ω : ℝ) (φ : ℝ)
    (h1 : ω < 0)
    (h2 : ∀ x, f ω φ (x - π / 12) = g x) :
    ∃ k : ℤ, φ = 2 * k * π + π / 3 :=
by
  sorry

end determine_phi_l255_255032


namespace triangle_ABC_area_l255_255175

noncomputable def triangle_area (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem triangle_ABC_area :
  let A := (3, -1)
  let B := (1, -3)
  let C := (-6, 6)
  triangle_area A B C = 16 :=
by
  let A := (3, -1)
  let B := (1, -3)
  let C := (-6, 6)
  have : triangle_area A B C = 16 := sorry
  exact this

end triangle_ABC_area_l255_255175


namespace sqrt_multiplication_l255_255815

theorem sqrt_multiplication :
  (Real.sqrt 8 - Real.sqrt 2) * (Real.sqrt 7 - Real.sqrt 3) = Real.sqrt 14 - Real.sqrt 6 :=
by
  -- statement follows
  sorry

end sqrt_multiplication_l255_255815


namespace triangle_exists_l255_255158

theorem triangle_exists 
  (c m_c d : ℝ) 
  (h1 : d > c) 
  (h2 : m_c < sqrt ((d^2 - c^2) / 4)) :
  ∃ a b : ℝ, a + b = d ∧ 
  let s := (d + c) / 2 in 
  let ρ := (2 * m_c * c) / (d + c) in
  (h := (a + b + c) / 2, 
  s = h ∧
  ρ = (2 * m_c * c) / (d + c) ∧
  m_c^2 < (d^2 - c^2) / 4) :=
sorry

end triangle_exists_l255_255158


namespace probability_sum_7_9_11_l255_255028

def Die1 := {1, 2, 3, 3, 4, 4}
def Die2 := {2, 3, 4, 7, 7, 10}

def is_valid_sum (n : ℕ) : Prop := n = 7 ∨ n = 9 ∨ n = 11

def event_probability : ℚ := 
   let outcomes := ((1 : ℚ)/6) * ((1 : ℚ)/6) + 
                   ((1 : ℚ)/6) * ((1 : ℚ)/3) +
                   ((1 : ℚ)/3) * ((1 : ℚ)/6) +
                   ((1 : ℚ)/3) * ((1 : ℚ)/2)
   outcomes

theorem probability_sum_7_9_11 : event_probability = 11 / 36 :=
by {
  sorry
}

end probability_sum_7_9_11_l255_255028


namespace remainder_seven_div_by_nine_l255_255734

-- Define the main problem statement in Lean
theorem remainder_seven_div_by_nine (n : ℕ) (hn : Odd n) : 
  let expr := (∑ k in Finset.range n, Nat.choose n k * 7^(n - k)) 
  expr % 9 = 7 := 
by
  sorry

end remainder_seven_div_by_nine_l255_255734


namespace min_sum_six_l255_255217

theorem min_sum_six (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h : a * b = a + b + 3) :
  a + b ≥ 6 :=
sorry

end min_sum_six_l255_255217


namespace marbles_count_l255_255109

theorem marbles_count (M : ℕ)
  (h_blue : (M / 2) = n_blue)
  (h_red : (M / 4) = n_red)
  (h_green : 27 = n_green)
  (h_yellow : 14 = n_yellow)
  (h_total : (n_blue + n_red + n_green + n_yellow) = M) :
  M = 164 :=
by
  sorry

end marbles_count_l255_255109


namespace min_odd_in_A_P_l255_255361

-- Define the polynomial P of degree 8
variable (P : ℝ → ℝ)
-- Assume P is a polynomial of degree 8
axiom degree_P : degree P = 8

-- Define the set A_P as the set of all x for which P(x) gives a certain value
def A_P (c : ℝ) : Set ℝ := { x | P x = c }

-- Given condition
axiom include_8 : 8 ∈ A_P (P 8)

-- Define if a number is odd
def is_odd (n : ℝ) : Prop := ∃k : ℤ, n = 2 * k + 1

-- The minimum number of odd numbers that are in the set A_P,
-- given that the number 8 is included in A_P, is 1

theorem min_odd_in_A_P : ∀ P, (degree P = 8) → 8 ∈ A_P (P 8) → ∃ x ∈ A_P (P 8), is_odd x := 
by
  sorry -- proof goes here

end min_odd_in_A_P_l255_255361


namespace billy_can_play_l255_255981

-- Define the conditions
def total_songs : ℕ := 52
def songs_to_learn : ℕ := 28

-- Define the statement to be proved
theorem billy_can_play : total_songs - songs_to_learn = 24 := by
  -- Proof goes here
  sorry

end billy_can_play_l255_255981


namespace remainder_of_polynomial_division_l255_255664

theorem remainder_of_polynomial_division :
  let f := λ y : ℝ, y^4 - 3y^3 + 2y^2 - y + 1
  let g := λ y : ℝ, (y^2 - 1) * (y + 2)
  let remainder := λ y : ℝ, y^2 - 4y + 3
  ∀ y : ℝ, f y = g y * polynomial.div_by g. y f + remainder :=
by
  sorry

end remainder_of_polynomial_division_l255_255664


namespace hyperbola_eccentricity_eq_sqrt2_l255_255219

variables {a b c : ℝ}
variables {M N F : ℝ × ℝ}
variables (hxpos : 0 < a)
variables (hypos : 0 < b)
variables (h_hyp : ∀ (x y : ℝ), x = M.1 ∧ y = M.2 → x = N.1 ∧ y = N.2 → x^2 / a^2 - y^2 / b^2 = 1)
variables (h_MN : ∃ (M N : ℝ × ℝ), M ≠ N ∧ M.1 * N.1 + M.2 * N.2 = 0)
variables (h_area : ∃ (M N F : ℝ × ℝ), 0.5 * (| M.1 * N.2 - M.2 * N.1 | = ab))

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  √(a^2 + b^2) / a

theorem hyperbola_eccentricity_eq_sqrt2 :
  ∀ (a b : ℝ), 0 < a → 0 < b → (∃ (M N F : ℝ × ℝ), M ≠ N ∧ 0.5 * (| M.1 * N.2 - M.2 * N.1 | = ab))
  → (∃ (M N : ℝ × ℝ), M ≠ N ∧ M.1 * N.1 + M.2 * N.2 = 0)
  → (∀ (x y : ℝ), x = M.1 ∧ y = M.2 → x = N.1 ∧ y = N.2 → x^2 / a^2 - y^2 / b^2 = 1)
  → eccentricity a b = √2 := 
by
  intros
  sorry

end hyperbola_eccentricity_eq_sqrt2_l255_255219


namespace exists_subseq_sum_squares_div_by_n_l255_255205

theorem exists_subseq_sum_squares_div_by_n (a : ℕ → ℤ) (n : ℕ) (h : 2 ≤ n) :
  ∃ k : Finset ℕ, (∀ x ∈ k, x ≤ n) ∧ ∑ x in k, (a x)^2 % n = 0 :=
sorry

end exists_subseq_sum_squares_div_by_n_l255_255205


namespace ratio_blue_to_total_l255_255943

theorem ratio_blue_to_total (total_marbles red_marbles green_marbles yellow_marbles blue_marbles : ℕ)
    (h_total : total_marbles = 164)
    (h_red : red_marbles = total_marbles / 4)
    (h_green : green_marbles = 27)
    (h_yellow : yellow_marbles = 14)
    (h_blue : blue_marbles = total_marbles - (red_marbles + green_marbles + yellow_marbles)) :
  blue_marbles / total_marbles = 1 / 2 :=
by
  sorry

end ratio_blue_to_total_l255_255943


namespace pascal_triangle_12_l255_255729

-- Definitions from conditions
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Main proof statement
theorem pascal_triangle_12 (n : ℕ) (k : ℕ) : n = 12 → binomial 12 1 = 12 → binomial 12 11 = 12 →
  (∀ k', k' ≠ 1 ∧ k' ≠ 11 → binomial 12 k' ≠ 12) → 
  (∀ n', n' > 12 → binomial n' 1 ≠ 12) →
  (∀ n' k', n' ≠ 12 ∧ k' ≠ 1 ∧ k' ≠ n' - 1 → binomial n' k' ≠ 12) →
  n = 12 ∧ k = 1 ∨ n = 12 ∧ k = 11 :=
begin
  intros h1 h2 h3 h4 h5 h6,
  have h : n = 12 := h1,
  by_cases hk : k = 1; { cases hk },
  { left, exact ⟨h, hk⟩ },
  by_cases hkn : k = 11; { cases hkn },
  { right, exact ⟨h, hkn⟩ },
  have h7 : k ≠ 1 ∧ k ≠ 11 := ⟨neq_of_not_mem (mt or.inl hk), neq_of_not_mem (mt or.inr hkn)⟩,
  exact (h5 ⟨h, h7.1, h7.2⟩),
end

end pascal_triangle_12_l255_255729


namespace swimming_pool_problem_l255_255495

noncomputable def solution : ℕ × ℕ × ℕ :=
  let p := 60
  let q := 30
  let r := 2
  (p, q, r)

theorem swimming_pool_problem :
  let (p, q, r) := solution in
  let n := p - q * Real.sqrt r in
  p + q + r = 92 ∧ n = 60 - 30 * Real.sqrt 2 :=
by
  have hp : p = 60 := rfl
  have hq : q = 30 := rfl
  have hr : r = 2 := rfl
  have hn : n = 60 - 30 * Real.sqrt 2 :=
    by simp [hp, hq, hr, solution]
  simp [hn, hp, hq, hr]
  sorry

end swimming_pool_problem_l255_255495


namespace find_cubic_polynomial_l255_255259

noncomputable def h (x : ℝ) : ℝ := x^3 - 2 * x^2 + 3 * x - 4

noncomputable def j (x : ℝ) : ℝ := x^3 - 8 * x^2 + 108 * x - 64

theorem find_cubic_polynomial
  (roots_h : set ℝ) 
  (distinct_roots : roots_h = {s | h s = 0} ∧ roots_h.finite ∧ roots_h.card = 3) :
  (∃ a b c : ℝ, j = λ x, x^3 + a * x^2 + b * x + c ∧ 
    ∀ s ∈ roots_h, j (s^3) = 0) :=
begin
  use [-8, 108, -64],
  split,
  { funext,
    exact j (x) },
  { intros s hs,
    sorry }
end

end find_cubic_polynomial_l255_255259


namespace sqrt_square_identity_sqrt_529441_squared_l255_255613

theorem sqrt_square_identity (n : ℕ) : (Nat.sqrt n) ^ 2 = n := sorry

theorem sqrt_529441_squared : (Nat.sqrt 529441) ^ 2 = 529441 := 
begin
  exact sqrt_square_identity 529441,
end

end sqrt_square_identity_sqrt_529441_squared_l255_255613


namespace min_abs_sum_l255_255341

theorem min_abs_sum (a b c d : ℤ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h : ∀m n, (Matrix.mul (Matrix.of 2 2 ![(a, b), (c, d)]) (Matrix.of 2 2 ![(a, b), (c, d)])
    m n) = (Matrix.of 2 2 ![(9, 0), (0, 9)]) m n) :
    |a| + |b| + |c| + |d| = 8 :=
begin
  sorry
end

end min_abs_sum_l255_255341


namespace gcd_digits_le_two_l255_255283

open Nat

theorem gcd_digits_le_two (a b : ℕ) (ha : a < 10^5) (hb : b < 10^5)
  (hlcm_digits : 10^8 ≤ lcm a b ∧ lcm a b < 10^9) : gcd a b < 100 :=
by
  -- Proof here
  sorry

end gcd_digits_le_two_l255_255283


namespace trigonometric_identity_l255_255842

theorem trigonometric_identity (x y : ℝ) :
  sin (x + y) * sin x + cos (x + y) * cos x = cos y := 
sorry

end trigonometric_identity_l255_255842


namespace minimum_odd_numbers_in_A_P_l255_255373

-- Polynomials and assumptions
def degree (P : Polynomial ℝ) : ℕ := P.natDegree

-- The set A_P is defined as the set of all numbers x for which P(x) gives a certain value
def A_P (P : Polynomial ℝ) : Set ℝ := {x : ℝ | P.eval x = P.eval 8}

-- Define the main theorem statement
theorem minimum_odd_numbers_in_A_P (P : Polynomial ℝ) (hdeg : degree P = 8) (h8 : 8 ∈ A_P P) : 
  ∃ n, n = 1 ∧ ∃ (x : ℝ), x ∈ A_P P ∧ ¬ (x % 2 = 0) := sorry

end minimum_odd_numbers_in_A_P_l255_255373


namespace Vasya_lives_on_the_5th_floor_l255_255510

theorem Vasya_lives_on_the_5th_floor :
  ∀ (steps_Petya: ℕ) (floors_Petya: ℕ) (steps_Vasya: ℕ),
    steps_Petya = 36 ∧ floors_Petya = 2 ∧ steps_Vasya = 72 →
    ((steps_Vasya / (steps_Petya / floors_Petya)) + 1 = 5) :=
by
  intros steps_Petya floors_Petya steps_Vasya h
  cases h with h1 h2
  cases h2 with h3 h4
  have steps_per_floor : ℕ := steps_Petya / floors_Petya
  have floors_Vasya : ℕ := steps_Vasya / steps_per_floor
  have Vasya_floor : ℕ := floors_Vasya + 1
  sorry

end Vasya_lives_on_the_5th_floor_l255_255510


namespace male_contestants_l255_255565

theorem male_contestants (total_contestants : ℕ) (female_proportion : ℕ) (total_females : ℕ) :
  female_proportion = 3 ∧ total_contestants = 18 ∧ total_females = total_contestants / female_proportion →
  (total_contestants - total_females) = 12 :=
by
  sorry

end male_contestants_l255_255565


namespace no_real_roots_of_quadratic_l255_255887

theorem no_real_roots_of_quadratic (a b : ℝ) : (∀ x : ℝ, x^2 + a * x + b ≠ 0) ↔ ¬ (∃ x : ℝ, x^2 + a * x + b = 0) := sorry

end no_real_roots_of_quadratic_l255_255887


namespace integral_f_l255_255861

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 - x else real.sqrt (4 - x^2)

theorem integral_f : (∫ x in -2..2, f x) = 6 + real.pi :=
by
  sorry

end integral_f_l255_255861


namespace L_of_specific_set_L_of_arithmetic_sequence_l255_255719

-- Definitions for set and arithmetic sequence properties
variable {α : Type*} [Add α] [LT α] [DecidableEq α] 

def sum_of_two_elements (A : Finset α) : Finset α := 
  (A.product A).filter (λ ⟨x, y⟩, x < y).image (λ ⟨x, y⟩, x + y)

def L (A : Finset α) : ℕ := (sum_of_two_elements A).card

-- Problem 1: Specific set A = {2, 4, 6, 8}
def specific_set_A := ({2, 4, 6, 8} : Finset ℕ)

-- Problem 2: Arithmetic sequence
def is_arithmetic_sequence (A : Finset ℕ) : Prop := 
  ∃ (a d : ℕ), A = Finset.image (λ n, a + n * d) (Finset.range A.card)

theorem L_of_specific_set : L specific_set_A = 5 := 
  sorry

theorem L_of_arithmetic_sequence (m : ℕ) (h : 2 < m) (A : Finset ℕ) (ha : A.card = m) (has : is_arithmetic_sequence A) : 
  L A = 2 * m - 3 :=
  sorry

end L_of_specific_set_L_of_arithmetic_sequence_l255_255719


namespace part1_part2_l255_255041

def sequence : ℕ → ℤ
| 0     := 1
| 1     := 3
| n + 2 := if n % 2 = 0 then sequence (n + 1) + 9 * sequence n else 9 * sequence (n + 1) + 5 * sequence n

theorem part1 : (sequence 1995) ^ 2 + (sequence 1996) ^ 2 + (sequence 1997) ^ 2 + (sequence 1998) ^ 2 +
                (sequence 1999) ^ 2 + (sequence 2000) ^ 2 ≡ 12 [MOD 20] :=
by
  sorry

theorem part2 (n : ℕ) : ∃ k : ℤ, a_(2n+1) = k^2 → False :=
by
  sorry

end part1_part2_l255_255041


namespace circle_equation_center_1_1_passes_origin_l255_255460

theorem circle_equation_center_1_1_passes_origin :
  ∀ (x y : ℝ), ((x - 1)^2 + (y - 1)^2 = 2) ↔ ((x, y) = (0, 0) ∨ ∃ r : ℝ, r = sqrt 2) := 
by
  sorry

end circle_equation_center_1_1_passes_origin_l255_255460


namespace min_odd_in_A_P_l255_255386

-- Define the polynomial P of degree 8
variable {R : Type*} [Ring R] (P : Polynomial R)
variable (hP : degree P = 8)

-- Define the set A_P for some value c in R
def A_P (c : R) : Set R := { x | P.eval x = c }

-- Given that the number 8 is included in the set A_P
variable (c : R) (h8 : (8 : R) ∈ A_P c)

-- Prove that the minimum number of odd numbers in the set A_P is 1
theorem min_odd_in_A_P : ∃ x ∈ A_P c, (x % 2 = 1) := sorry

end min_odd_in_A_P_l255_255386


namespace minimum_odd_in_A_P_l255_255377

-- Define the polynomial and its properties
def polynomial_degree8 (P : ℝ[X]) : Prop :=
  P.degree = 8

-- Define the set A_P
def A_P (P : ℝ[X]) : set ℝ :=
  {x | P.eval x = 8}

-- Main theorem statement
theorem minimum_odd_in_A_P (P : ℝ[X]) (hP : polynomial_degree8 P) : ∃ n : ℕ, n = 1 ∧ ∃ x ∈ A_P P, odd x :=
sorry

end minimum_odd_in_A_P_l255_255377


namespace max_value_of_diff_squares_l255_255279

variables {R : Type*} [field R] [normed_space ℝ R]
variables (a b c d : R)

/-- condition : a, b, c, d are unit vectors in R^4 -/
def is_unit_vector (v : R) : Prop := ∥v∥ = 1

/-- main theorem : maximum value of the given expression is 24 -/
theorem max_value_of_diff_squares (ha : is_unit_vector a) (hb : is_unit_vector b)
  (hc : is_unit_vector c) (hd : is_unit_vector d) :
  ∥a - b∥^2 + ∥a - c∥^2 + ∥a - d∥^2 + ∥b - c∥^2 + ∥b - d∥^2 + ∥c - d∥^2 ≤ 24 :=
begin
  sorry
end

end max_value_of_diff_squares_l255_255279


namespace equation_count_correct_l255_255990

def is_even (n : ℕ) : Prop := n % 2 = 0

def valid_b_values := {1, 2, 3, 7, 8, 9}

def valid_c_values := {2, 8}

def has_real_roots (b c : ℕ) : Prop :=
  b^2 - 4 * c ≥ 0

def valid_equation_count: ℕ :=
  valid_b_values.to_finset.sum (λ b, valid_c_values.to_finset.count (λ c, has_real_roots b c))

theorem equation_count_correct :
  valid_equation_count = 7 :=
by
  sorry

end equation_count_correct_l255_255990


namespace part1_infinitely_many_pairs_part1_finitely_many_pairs_part2_finitely_many_pairs_part2_infinitely_many_pairs_l255_255525

def sequence_def (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n, a (n + 2) = 2 * a (n + 1) + a n

def pell_eqn_solns (p q : ℕ) : Prop := p ^ 2 - 2 * q ^ 2 = 1

theorem part1_infinitely_many_pairs (β : ℝ) : 
  (β >= Real.sqrt 2 / 4) → ∃^∞ (p q : ℕ), abs (p / q - Real.sqrt 2) < β / (q ^ 2) :=
sorry

theorem part1_finitely_many_pairs (β : ℝ) :
  (β < Real.sqrt 2 / 4) → ∃* (p q : ℕ), abs (p / q - Real.sqrt 2) < β / (q ^ 2) :=
sorry

theorem part2_finitely_many_pairs (a : ℕ → ℕ) (h_seq : sequence_def a) (β : ℝ) :
  (β < Real.sqrt 2 / 2) → ∃* (p q : ℕ), q ∉ set.range a ∧ abs (p / q - Real.sqrt 2) < β / q :=
sorry

theorem part2_infinitely_many_pairs (a : ℕ → ℕ) (h_seq : sequence_def a) (β : ℝ) :
  (β >= Real.sqrt 2 / 2) → ∃^∞ (p q : ℕ), q ∉ set.range a ∧ abs (p / q - Real.sqrt 2) < β / q :=
sorry

end part1_infinitely_many_pairs_part1_finitely_many_pairs_part2_finitely_many_pairs_part2_infinitely_many_pairs_l255_255525


namespace expected_no_advice_formula_l255_255597

noncomputable def expected_no_advice (n : ℕ) (p : ℝ) : ℝ :=
  ∑ j in Finset.range n, (1 - p) ^ j

theorem expected_no_advice_formula (n : ℕ) (p : ℝ) (hp : 0 ≤ p ∧ p < 1) : 
  expected_no_advice n p = (1 - (1 - p) ^ n) / p :=
by
  sorry

end expected_no_advice_formula_l255_255597


namespace gcd_two_5_digit_integers_l255_255286

theorem gcd_two_5_digit_integers (a b : ℕ) 
  (h1 : 10^4 ≤ a ∧ a < 10^5)
  (h2 : 10^4 ≤ b ∧ b < 10^5)
  (h3 : 10^8 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^9) :
  Nat.gcd a b < 10^2 :=
by
  sorry  -- Skip the proof

end gcd_two_5_digit_integers_l255_255286


namespace max_planes_from_points_l255_255896

def no_three_collinear (pts : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ p1 p2 p3 : ℝ × ℝ × ℝ, p1 ∈ pts → p2 ∈ pts → p3 ∈ pts → 
    ¬ collinear {p1, p2, p3}

theorem max_planes_from_points (P : Set (ℝ × ℝ × ℝ)) (hP : P.card = 10) (hnc : no_three_collinear P) :
  ∃! (n : ℕ), n = 120 :=
begin
  -- We would add the proof here
  sorry
end

end max_planes_from_points_l255_255896


namespace parabola_equation_l255_255469

theorem parabola_equation (p : ℝ) :
  (∃ (a : ℝ), (a = 4 * real.sqrt 3 ∧ ∀ (x y : ℝ), y^2 = a * x → x = real.sqrt 3 → y = -2 * real.sqrt 3) ∨ 
  (a = -real.sqrt 3 / 2 ∧ ∀ (x y : ℝ), x^2 = a * y → x = real.sqrt 3 → y = -2 * real.sqrt 3)) :=
by
  sorry

end parabola_equation_l255_255469


namespace number_of_lemons_l255_255444

theorem number_of_lemons
  (total_fruits : ℕ)
  (mangoes : ℕ)
  (pears : ℕ)
  (pawpaws : ℕ)
  (kiwis : ℕ)
  (lemons : ℕ)
  (h_total : total_fruits = 58)
  (h_mangoes : mangoes = 18)
  (h_pears : pears = 10)
  (h_pawpaws : pawpaws = 12)
  (h_kiwis_lemons_equal : kiwis = lemons) :
  lemons = 9 :=
by
  sorry

end number_of_lemons_l255_255444


namespace find_k_l255_255666

theorem find_k 
  (a : ℝ)
  (h_a_pos : 0 < a)
  (h_a_neq_1 : a ≠ 1) 
  (h_odd : ∀ x : ℝ, f x = -f (-x))
  (h_f_eq_3 : f x = 3)
  (h_min_g : ∀ x ∈ Icc (2 : ℝ) (∞ : ℝ), g x = x^2 + f x)
  (h_min_value : ∃ x ∈ Icc (2 : ℝ) (∞ : ℝ), g x = -2)
  : ∃ k : ℝ, k = 1 := 
sorry

end find_k_l255_255666


namespace quadrilateral_AC_length_l255_255759

theorem quadrilateral_AC_length
    (A B C D : Point)
    (AB AD : ℝ)
    (angle_A : angle A B D = 120)
    (angle_B : angle B A C = 90)
    (angle_D : angle D A B = 90)
    (AB_length : dist A B = 13)
    (AD_length : dist A D = 46) :
  dist A C = 62 :=
sorry

end quadrilateral_AC_length_l255_255759


namespace points_on_opposite_sides_l255_255216

theorem points_on_opposite_sides (a : ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) 
    (hA : A = (a, 1)) 
    (hB : B = (2, a)) 
    (opposite_sides : A.1 < 0 ∧ B.1 > 0 ∨ A.1 > 0 ∧ B.1 < 0) 
    : a < 0 := 
  sorry

end points_on_opposite_sides_l255_255216


namespace bee_flight_distance_l255_255933

-- Define the problem and assert the result
theorem bee_flight_distance :
  ∃ (a b c d : ℕ), a + b + c + d = 2024 ∧ 
  b %% 2 ≠ 0 ∧ d %% 2 ≠ 0 ∧ 
  ∀ (P : ℕ → ℂ), P 0 = 1 ∧ 
  (∀ j : ℕ, j ≥ 1 → P (j + 1) = P j + (exp (complex.I * (π / 6) * j) * (j + 1))) ∧ 
  abs (P 2015 - 1) = a * real.sqrt b + c * real.sqrt d :=
begin
  sorry
end

end bee_flight_distance_l255_255933


namespace caitlin_bracelets_l255_255983

/-- 
Caitlin makes bracelets to sell at the farmer’s market every weekend. 
Each bracelet takes twice as many small beads as it does large beads. 
If each bracelet uses 12 large beads, and Caitlin has 528 beads with equal amounts of large and small beads, 
prove that Caitlin can make 11 bracelets for this weekend.
-/
theorem caitlin_bracelets (total_beads large_beads_per_bracelet small_beads_per_bracelet total_large_beads total_small_beads bracelets : ℕ)
  (h1 : total_beads = 528)
  (h2 : total_beads = total_large_beads + total_small_beads)
  (h3 : total_large_beads = total_small_beads)
  (h4 : large_beads_per_bracelet = 12)
  (h5 : small_beads_per_bracelet = 2 * large_beads_per_bracelet)
  (h6 : bracelets = total_small_beads / small_beads_per_bracelet) : 
  bracelets = 11 := 
by {
  sorry
}

end caitlin_bracelets_l255_255983


namespace find_square_l255_255281

-- Definitions based on conditions for original values
variables (x y : ℝ)

-- The value of the fraction given original x^2 + square
def fraction_original (square : ℝ) := 2 * x * y / (x^2 + square)

-- Definitions based on increased values
def x_increased := 3 * x
def y_increased := 3 * y
def fraction_increased (square : ℝ) := 2 * x_increased * y_increased / (x_increased^2 + square)

-- The theorem statement that the unchanged fraction condition implies square = y^2
theorem find_square: 
  (∀ (square : ℝ), fraction_original x y square = fraction_increased x y square) → 
  square = y^2 := 
sorry

end find_square_l255_255281


namespace caitlin_bracelets_l255_255985

-- Define the conditions
def twice_as_many_small_beads (x y : Nat) : Prop :=
  y = 2 * x

def total_large_small_beads (total large small : Nat) : Prop :=
  total = large + small ∧ large = small

def bracelet_beads (large_beads_per_bracelet small_beads_per_bracelet large_per_bracelet : Nat) : Prop :=
  small_beads_per_bracelet = 2 * large_per_bracelet

def total_bracelets (total_large_beads large_per_bracelet bracelets : Nat) : Prop :=
  bracelets = total_large_beads / large_per_bracelet

-- The theorem to be proved
theorem caitlin_bracelets (total_beads large_per_bracelet small_per_bracelet : Nat) (bracelets : Nat) :
    total_beads = 528 ∧
    large_per_bracelet = 12 ∧
    twice_as_many_small_beads large_per_bracelet small_per_bracelet ∧
    total_large_small_beads total_beads 264 264 ∧
    bracelet_beads large_per_bracelet small_per_bracelet 12 ∧
    total_bracelets 264 12 bracelets
  → bracelets = 22 := by
  sorry

end caitlin_bracelets_l255_255985


namespace inequality_solutions_l255_255843

theorem inequality_solutions (n : ℕ) (h : n > 0) : n^3 - n < n! ↔ (n = 1 ∨ n ≥ 6) := 
by
  sorry

end inequality_solutions_l255_255843


namespace find_increasing_function_l255_255572

-- Define the functions
def f1 (x : ℝ) : ℝ := Real.exp (-x)
def f2 (x : ℝ) : ℝ := x^3
def f3 (x : ℝ) : ℝ := Real.log x
def f4 (x : ℝ) : ℝ := Real.abs x

-- Define increasing function property
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Statement to prove
theorem find_increasing_function :
  (∀ x : ℝ, ∃ y : ℝ, f1 x = y → false) ∧               -- f1 is not increasing on ℝ
  (∀ x : ℝ, ∃ y : ℝ, f3 x = y → false) ∧               -- f3 is not increasing on ℝ
  (∀ x : ℝ, ∃ y : ℝ, f4 x = y → false) ∧               -- f4 is not increasing on ℝ
  (is_increasing f2) :=                                -- f2 is increasing on ℝ
sorry

end find_increasing_function_l255_255572


namespace julia_kids_count_l255_255780

theorem julia_kids_count 
  (tuesday_kids : ℕ) (wednesday_kids : ℕ) (total_kids : ℕ)
  (h1 : tuesday_kids = 14)
  (h2 : wednesday_kids = 22)
  (h3 : total_kids = 75) :
  (total_kids - tuesday_kids - wednesday_kids) = 39 :=
by
  rw [h1, h2, h3]
  decide
  sorry

end julia_kids_count_l255_255780


namespace shortest_path_octahedron_l255_255577

theorem shortest_path_octahedron 
  (edge_length : ℝ) (h : edge_length = 2) 
  (d : ℝ) : d = 2 :=
by
  sorry

end shortest_path_octahedron_l255_255577


namespace smallest_height_l255_255567

variables {A B C D P : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P]

-- Given points A, B, C lying on a plane and point D in space
-- Assume P is the foot of the perpendicular from D to the plane ABC
-- Given that the height DP is the smallest of the four heights in tetrahedron ABCD
-- Prove that P must lie within the smaller interior triangle A₁B₁C₁ formed by trisecting sides of triangle ABC

theorem smallest_height (A B C D P: Type*) 
  (h1 : is_tetrahedron ABCD)
  (h2 : is_perpendicular D P ABC)
  (h3 : smallest_height DP ABCD):
  lies_within_interior_mytriangle P ABC := 
sorry

end smallest_height_l255_255567


namespace amy_baskets_l255_255137

theorem amy_baskets : 
  let chocolate_bars := 5 in
  let m_and_ms := 7 * chocolate_bars in
  let marshmallows := 6 * m_and_ms in
  let total_candies := chocolate_bars + m_and_ms + marshmallows in
  let candies_per_basket := 10 in
  total_candies / candies_per_basket = 25 := 
by
  sorry

end amy_baskets_l255_255137


namespace inequality_proof_l255_255396

variables {n : ℕ} (a : ℕ → ℝ) (s : ℝ)

noncomputable def A (i : ℕ) : ℝ := (a i) ^ 2 + (a (i + 1)) ^ 2 - (a (i + 2)) ^ 2) / ((a i) + (a (i + 1)) - (a (i + 2)))

theorem inequality_proof (h1 : 3 ≤ n) 
  (h2 : ∀ i : ℕ, i < n → 2 ≤ a i ∧ a i ≤ 3) 
  (hs : s = ∑ i in finset.range n, a i) :
  (∑ i in finset.range n, A a i) ≤ 2 * s - 2 * n :=
sorry

end inequality_proof_l255_255396


namespace find_y_l255_255197

noncomputable def vector_a : (ℝ × ℝ × ℝ) := (2, -1, 3)
noncomputable def vector_b (y : ℝ) : (ℝ × ℝ × ℝ) := (-4, y, 2)
noncomputable def vector_sum (y : ℝ) : (ℝ × ℝ × ℝ) := (2 + -4, -1 + y, 3 + 2)

theorem find_y (y : ℝ) :
  (vector_a.1 * vector_sum y.1) +
  (vector_a.2 * vector_sum y.2) +
  (vector_a.3 * vector_sum y.3) = 0 → y = 12 :=
by {
  calc (2) * (-2) + (-1) * (y - 1) + (3) * (5) = 0 
  → -4 - y + 1 + 15 = 0 
  → 12 - y = 0 
  → y = 12
}

end find_y_l255_255197


namespace final_scores_possible_l255_255302

theorem final_scores_possible : 
  let base_scores := {1, 2, 3, 4, 5, 6, 7, 8, 9} ∪ {0, 10},
      time_bonuses := {1, 2, 3, 4},
      bonus_if_all_correct := 20,
      all_correct_scores := {10 * tb + bonus_if_all_correct | tb in time_bonuses},
      possible_scores := {bs * tb | bs in base_scores, tb in time_bonuses} ∪ all_correct_scores
  in
  cardinality possible_scores = 25 :=
by
  sorry

end final_scores_possible_l255_255302


namespace max_value_of_xy_l255_255709

noncomputable def max_xy : ℝ :=
  max (x * y)

theorem max_value_of_xy
  (x y : ℝ)
  (h1 : 1 + 2 * x = 2 - 2 * y)
  (hx : x > 0) (hy : y > 0):
  max_xy x y = 1 / 4 :=
by
  sorry

end max_value_of_xy_l255_255709


namespace simple_interest_correct_l255_255962

def principal : ℝ := 10040.625
def rate : ℝ := 8
def time : ℕ := 5

theorem simple_interest_correct :
  (principal * rate * time / 100) = 40162.5 :=
by 
  sorry

end simple_interest_correct_l255_255962


namespace hypotenuse_not_less_than_two_l255_255559

noncomputable def vertex_a (a : ℝ) : ℝ × ℝ := (a, a^2)
noncomputable def vertex_b (b : ℝ) : ℝ × ℝ := (b, b^2)
noncomputable def vertex_c (c : ℝ) : ℝ × ℝ := (c, c^2)

noncomputable def length_sq (A B : ℝ × ℝ) : ℝ :=
(A.1 - B.1)^2 + (A.2 - B.2)^2

theorem hypotenuse_not_less_than_two {a b c : ℝ}
  (h₁ : (a * b) + (b * c) + (c * a) = -1)
  (h₂ : a^2 + (b + c) * a + b * c + 1 = 0) :
  length_sq (vertex_b b) (vertex_c c) ≥ 4 :=
begin
  sorry
end

end hypotenuse_not_less_than_two_l255_255559


namespace similar_figures_count_l255_255242

-- Define the geometric figures in conditions
def circles : Type := { x : unit // true } 
def squares : Type := { x : unit // true }
def rectangles : Type := { x : unit // true }
def regular_hexagons : Type := { x : unit // true }
def isosceles_triangles : Type := { x : unit // true }
def right_angled_triangles : Type := { x : unit // true }
def isosceles_trapezoids : Type := { x : unit // true }
def rhombus_40 : Type := { x : unit // true }

-- The proposition we want to prove
theorem similar_figures_count : 
  ∃ (pairs_similar : ℕ), pairs_similar = 4 :=
by
  sorry

end similar_figures_count_l255_255242


namespace sum_last_two_digits_l255_255072

theorem sum_last_two_digits (a b : ℕ) (ha : a = 7) (hb : b = 13) :
  (a ^ 30 + b ^ 30) % 100 = 0 := 
by
  sorry

end sum_last_two_digits_l255_255072


namespace hyperbola_with_given_asymptotes_l255_255997

/-
Problem: Prove that the equation of the hyperbola with foci on the x-axis 
and asymptotes given by y = ± 2x is x^2 - (y^2 / 4) = 1.
-/

def hyperbola_equation (a b : ℝ) : Prop := (x² / (a^2)) - (y² / (b^2)) = 1

def asymptote_slope (a b : ℝ) : Prop := b / a = 2

theorem hyperbola_with_given_asymptotes 
  (a b : ℝ)
  (h₁ : hyperbola_equation a b)
  (h₂ : asymptote_slope a b) : 
  (x^2 - (y^2 / 4) = 1) :=
sorry

end hyperbola_with_given_asymptotes_l255_255997


namespace repeating_decimal_to_fraction_l255_255632

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + 0.066666... ) : x = 11 / 30 :=
sorry

end repeating_decimal_to_fraction_l255_255632


namespace find_B_find_area_l255_255294

noncomputable theory
open_locale big_operators complex_conjugate

variables {a b c A B C : ℝ} (h1 : a * tan B = 2 * b * sin A)
variables (h2 : b = sqrt 3) (h3 : A = 5 * real.pi / 12)

-- Proof statement for B
theorem find_B (h1 : a * tan B = 2 * b * sin A) :
  ∃ B, B = real.pi / 3 :=
sorry

-- Proof statement for the area
theorem find_area (h1 : a * tan B = 2 * b * sin A) (h2 : b = sqrt 3) (h3 : A = 5 * real.pi / 12) :
  ∃ area, area = (3 + sqrt 3) / 4 :=
sorry

end find_B_find_area_l255_255294


namespace complex_product_l255_255695

theorem complex_product (z1 z2 : ℂ) (h1 : z1 = 4 + i) (h2 : z2 = 1 - 2i) : 
  z1 * z2 = 6 - 7i := 
by
  sorry

end complex_product_l255_255695


namespace ellipse_problem_proof_l255_255229

noncomputable def ellipse_eqn (a b: ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

noncomputable def area_Δ (F1 F2 Ak : ℝ × ℝ) : ℝ :=
  let (F1x, F1y) := F1
  let (F2x, F2y) := F2
  let (Akx, Aky) := Ak
  0.5 * real.sqrt ((F2x - F1x)^2 + (F2y - F1y)^2) * 2

noncomputable def encloses (Ck eq_ellipse : ℝ × ℝ → Prop) : Prop :=
  ∀ p : ℝ × ℝ, ¬Ck p → eq_ellipse p

theorem ellipse_problem_proof :
  ∃ (a b : ℝ) (F1 F2 Ak : ℝ × ℝ),
    (ellipse_eqn 6 3 → true) ∧
    (area_Δ F1 F2 (-k, 2) = 6 * real.sqrt 3) ∧
    (¬∃ (k : ℝ), encloses (fun (p : ℝ × ℝ) => p.fst ^ 2 + p.snd ^ 2 + 2 * k * p.fst - 4 * p.snd - 21 = 0) (ellipse_eqn 6 3) ) :=
begin
  sorry
end

end ellipse_problem_proof_l255_255229


namespace regular_tetrahedron_l255_255791

variables {A1 A2 A3 A4 Q : Type}
variables {r R : ℝ}
variables {S1 S2 S3 S4 : Sphere}

/-- The tetrahedron is regular if all spheres centered at its vertices are pairwise tangent,
and there is a point where spheres centered at this point and with radii r and R meet certain tangency conditions --/
theorem regular_tetrahedron
  (centers: set (Point ℝ)) (h_centers: centers = {A1, A2, A3, A4})
  (spheres_tangent: ∀ (S1 S2 S3 S4 : Sphere), tangent S1 S2 ∧ tangent S1 S3 ∧ tangent S1 S4 ∧ 
    tangent S2 S3 ∧ tangent S2 S4 ∧ tangent S3 S4)
  (exists_point: ∃ Q : Point ℝ, ∃ r R > 0,
    tangent (Sphere.mk Q r) S1 ∧ tangent (Sphere.mk Q r) S2 ∧ tangent (Sphere.mk Q r) S3 ∧ tangent (Sphere.mk Q r) S4 ∧
    tangent (Sphere.mk Q R) (Edge.mk A1 A2) ∧ tangent (Sphere.mk Q R) (Edge.mk A1 A3) ∧ tangent (Sphere.mk Q R) (Edge.mk A1 A4) ∧
    tangent (Sphere.mk Q R) (Edge.mk A2 A3) ∧ tangent (Sphere.mk Q R) (Edge.mk A2 A4) ∧
    tangent (Sphere.mk Q R) (Edge.mk A3 A4)) :
  (∃ (l : ℝ), l > 0 ∧ ∀ (A B : centers), dist A B = l) := sorry

end regular_tetrahedron_l255_255791


namespace pete_solved_4_percent_l255_255833

theorem pete_solved_4_percent
  (N F M : ℕ)
  (h1 : N = 0.05 * F)
  (h2 : N = 0.20 * M) :
  (N / (F + M)) * 100 = 4 := by
  sorry

end pete_solved_4_percent_l255_255833


namespace max_f_on_interval_l255_255180

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Define the domain
def domain := set.Icc (-1 : ℝ) 3

-- Prove that the maximum value of f on the domain is 18
theorem max_f_on_interval : ∃ x ∈ domain, ∀ y ∈ domain, f x ≥ f y ∧ f x = 18 :=
by
  sorry

end max_f_on_interval_l255_255180


namespace number_of_good_subsets_l255_255399

-- Define the conditions in Lean
def is_odd (n : ℕ) : Prop := ∃ (k : ℕ), n = 2 * k + 1

def is_good_subset (M T : Finset ℕ) : Prop :=
  let sumM := M.sum id in
  (sumM ∣ ∏ x in T, x) ∧ ¬ (sumM * sumM ∣ ∏ x in T, x)

-- Formalize the statement
theorem number_of_good_subsets (n : ℕ) (M : Finset ℕ) (h1 : is_odd n) (h2 : M.card = n) (h3 : (∀ x ∈ M, ∀ y ∈ M, x ≠ y → x ≠ y)) :
  ∃ k : ℕ, k = 2^(n-1) ∧ (∀ T, T ⊆ M → is_good_subset M T → T.card = k) :=
  sorry

end number_of_good_subsets_l255_255399


namespace second_largest_of_five_l255_255905

theorem second_largest_of_five :
  ∀ (a b c d e : ℕ), (a = 5 ∧ b = 8 ∧ c = 4 ∧ d = 3 ∧ e = 2) →
  ∃ x, (x = 5 ∧ x ≠ max b (max a (max c (max d e))) ∧ 
  is_second_largest x [a, b, c, d, e]) :=
by
  -- Introduce the parameters and hypothesis
  intros a b c d e h
  -- Specify the second largest number is 5
  existsi (5 : ℕ)
  -- Provide the required proof (skipped here with sorry)
  sorry

-- Define what it means to be the second largest element in a list
def is_second_largest (x : ℕ) (l : List ℕ) : Prop :=
  ∃ y : ℕ, y ≠ x ∧ y = max_list l ∧ ∀ z : ℕ, z ∈ l → z ≤ y → z ≠ x

end second_largest_of_five_l255_255905


namespace repeating_decimal_to_fraction_l255_255642

theorem repeating_decimal_to_fraction : (0.36 : ℝ) = (11 / 30 : ℝ) :=
sorry

end repeating_decimal_to_fraction_l255_255642


namespace range_of_x_l255_255319

theorem range_of_x (x : ℝ) : 
  sqrt ((x - 1) / (x - 2)) ≥ 0 → (x > 2 ∨ x ≤ 1) :=
by
  sorry

end range_of_x_l255_255319


namespace max_height_l255_255095

def height (t : ℝ) : ℝ := -16 * t^2 + 96 * t + 15 

theorem max_height : ∃ t : ℝ, height t = 159 :=
sorry

end max_height_l255_255095


namespace increasing_f_interval_l255_255201

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log a (abs (x + 1))

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (abs (x + 1))

theorem increasing_f_interval (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (h₃ : ∀ x ∈ Ioo (-1 : ℝ) (0 : ℝ), g a x > 0) :
  ∀ x y : ℝ, x < y → x < -1 → y < -1 → f a x < f a y := sorry

end increasing_f_interval_l255_255201


namespace option_B_option_C_option_D_l255_255289

-- Given a sequence that is half-increasing difference
def half_increasing_difference (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, (a n - (1/2) * a (n-1)) < (a (n+1) - (1/2) * a n)

-- Option B
theorem option_B (q : ℝ) (h : q > 1) : half_increasing_difference (λ n, q^n) :=
sorry

-- Option C
theorem option_C (a d : ℝ) (h : d > 0) : half_increasing_difference (λ n, a + (n-1) * d) :=
sorry

-- Option D
theorem option_D {a : ℕ → ℝ} (h : half_increasing_difference a) (S : ℕ → ℝ) (t : ℝ) (hS : ∀ n, S n = 2 * a n - 2^(n + 1) - t) :
  t ∈ Ioi (-32/3) :=
sorry

end option_B_option_C_option_D_l255_255289


namespace total_paving_cost_l255_255864

def main_floor_length := 5.5
def main_floor_width := 3.75
def mezzanine_floor_length := 3
def mezzanine_floor_width := 2
def price_per_sq_meter_type_a := 800
def price_per_sq_meter_type_b := 1200

def main_floor_area := main_floor_length * main_floor_width
def mezzanine_floor_area := mezzanine_floor_length * mezzanine_floor_width
def main_floor_cost := main_floor_area * price_per_sq_meter_type_a
def mezzanine_floor_cost := mezzanine_floor_area * price_per_sq_meter_type_b
def total_cost := main_floor_cost + mezzanine_floor_cost

theorem total_paving_cost : total_cost = 23700 := by
  sorry

end total_paving_cost_l255_255864


namespace children_got_on_bus_l255_255932

-- Definitions based on conditions
def initial_children : ℕ := 22
def children_got_off : ℕ := 60
def children_after_stop : ℕ := 2

-- Define the problem
theorem children_got_on_bus : ∃ x : ℕ, initial_children - children_got_off + x = children_after_stop ∧ x = 40 :=
by
  sorry

end children_got_on_bus_l255_255932


namespace min_odd_in_A_P_l255_255364

-- Define the polynomial P of degree 8
variable (P : ℝ → ℝ)
-- Assume P is a polynomial of degree 8
axiom degree_P : degree P = 8

-- Define the set A_P as the set of all x for which P(x) gives a certain value
def A_P (c : ℝ) : Set ℝ := { x | P x = c }

-- Given condition
axiom include_8 : 8 ∈ A_P (P 8)

-- Define if a number is odd
def is_odd (n : ℝ) : Prop := ∃k : ℤ, n = 2 * k + 1

-- The minimum number of odd numbers that are in the set A_P,
-- given that the number 8 is included in A_P, is 1

theorem min_odd_in_A_P : ∀ P, (degree P = 8) → 8 ∈ A_P (P 8) → ∃ x ∈ A_P (P 8), is_odd x := 
by
  sorry -- proof goes here

end min_odd_in_A_P_l255_255364


namespace perpendiculars_from_projections_intersect_single_point_l255_255091

theorem perpendiculars_from_projections_intersect_single_point
  (A B C A1 B1 C1 A2 B2 C2 : Point)
  (h_alt_A : IsAltitude A A1)
  (h_alt_B : IsAltitude B B1)
  (h_alt_C : IsAltitude C C1)
  (h_proj_A : IsProjection A A2 B1 C1)
  (h_proj_B : IsProjection B B2 C1 A1)
  (h_proj_C : IsProjection C C2 A1 B1) :
  IntersectAtSinglePoint (PerpendicularFrom A2 to BC) (PerpendicularFrom B2 to CA) (PerpendicularFrom C2 to AB) :=
sorry

end perpendiculars_from_projections_intersect_single_point_l255_255091


namespace marbles_count_l255_255110

theorem marbles_count (M : ℕ)
  (h_blue : (M / 2) = n_blue)
  (h_red : (M / 4) = n_red)
  (h_green : 27 = n_green)
  (h_yellow : 14 = n_yellow)
  (h_total : (n_blue + n_red + n_green + n_yellow) = M) :
  M = 164 :=
by
  sorry

end marbles_count_l255_255110


namespace minimum_odd_in_A_P_l255_255375

-- Define the polynomial and its properties
def polynomial_degree8 (P : ℝ[X]) : Prop :=
  P.degree = 8

-- Define the set A_P
def A_P (P : ℝ[X]) : set ℝ :=
  {x | P.eval x = 8}

-- Main theorem statement
theorem minimum_odd_in_A_P (P : ℝ[X]) (hP : polynomial_degree8 P) : ∃ n : ℕ, n = 1 ∧ ∃ x ∈ A_P P, odd x :=
sorry

end minimum_odd_in_A_P_l255_255375


namespace area_of_regular_dodecagon_inscribed_in_circle_l255_255499

-- Definitions
def is_regular_dodecagon (dodecagon : Type) (r : ℝ) : Prop := sorry
def is_inscribed_in_circle (dodecagon : Type) (r : ℝ) : Prop := sorry
def radius (dodecagon : Type) : ℝ := sorry

-- Theorem statement
theorem area_of_regular_dodecagon_inscribed_in_circle (dodecagon : Type) (r : ℝ)
  (h1 : is_regular_dodecagon dodecagon r)
  (h2 : is_inscribed_in_circle dodecagon r) :
  area dodecagon = 3 * r^2 :=
sorry

end area_of_regular_dodecagon_inscribed_in_circle_l255_255499


namespace area_increase_by_16_percent_l255_255035

theorem area_increase_by_16_percent (L B : ℝ) :
  ((1.45 * L) * (0.80 * B)) / (L * B) = 1.16 :=
by
  sorry

end area_increase_by_16_percent_l255_255035


namespace circumcircle_radius_of_triangle_l255_255416

theorem circumcircle_radius_of_triangle (A B C M N : Point)
  (h1 : on_bisector B A C M)
  (h2 : on_extension A B N)
  (h3 : distance A C = 1)
  (h4 : distance A M = 1)
  (h5 : angle A N M = angle C N M) :
  radius (circumcircle C N M) = 1 := 
sorry

end circumcircle_radius_of_triangle_l255_255416


namespace total_hours_proof_l255_255489

-- Conditions
def half_hour_show_episodes : ℕ := 24
def one_hour_show_episodes : ℕ := 12
def half_hour_per_episode : ℝ := 0.5
def one_hour_per_episode : ℝ := 1.0

-- Define the total hours Tim watched
def total_hours_watched : ℝ :=
  half_hour_show_episodes * half_hour_per_episode + one_hour_show_episodes * one_hour_per_episode

-- Prove that the total hours watched is 24
theorem total_hours_proof : total_hours_watched = 24 := by
  sorry

end total_hours_proof_l255_255489


namespace inequality_condition_l255_255961

theorem inequality_condition (x : ℝ) : (x - real.pi) * (x - real.exp 1) ≤ 0 ↔ x ∈ set.Ioo (real.exp 1) real.pi :=
sorry

end inequality_condition_l255_255961


namespace sum_of_squares_l255_255404

theorem sum_of_squares (k₁ k₂ k₃ : ℝ)
  (h_sum : k₁ + k₂ + k₃ = 1) : k₁^2 + k₂^2 + k₃^2 ≥ 1/3 :=
by sorry

end sum_of_squares_l255_255404


namespace percentage_disliked_by_both_l255_255828

theorem percentage_disliked_by_both 
  (total_comic_books : ℕ) 
  (percentage_females_like : ℕ) 
  (comic_books_males_like : ℕ) :
  total_comic_books = 300 →
  percentage_females_like = 30 →
  comic_books_males_like = 120 →
  ((total_comic_books - (total_comic_books * percentage_females_like / 100) - comic_books_males_like) * 100 / total_comic_books) = 30 :=
by
  intros h1 h2 h3
  sorry

end percentage_disliked_by_both_l255_255828


namespace correct_proposition_l255_255076

-- Define the conditions
def f (x : ℝ) : ℝ := sorry -- Linear function
def g (x : ℝ) : ℝ := 1 / (x + 1) -- Function for proposition B
def h (x : ℝ) : ℝ := sqrt (5 + 4 * x - x^2) -- Function for proposition C

-- Proposition A
def proposition_A : Prop :=
  f (f x) = 16 * x + 5 → f x = 4 * x + 1 ∨ f x = -4 * x - 5 / 3

-- Proposition B
def proposition_B : Prop :=
  ∀ x : ℝ, (x < -1 ∨ x > -1) → deriv g x < 0 

-- Proposition C
def proposition_C : Prop :=
  ∀ x : ℝ, 2 ≤ x → deriv h x < 0

-- Proposition D
def proposition_D : Prop :=
  ∀ f : ℝ → ℝ, ∃! y : ℝ, ∃ x : ℝ, y = f x ∧ x = 0

-- Prove that proposition D is the correct one
theorem correct_proposition : proposition_D :=
sorry

end correct_proposition_l255_255076


namespace exactly_two_succeed_probability_l255_255485

/-- Define the probabilities of three independent events -/
def P1 : ℚ := 1 / 2
def P2 : ℚ := 1 / 3
def P3 : ℚ := 3 / 4

/-- Define the probability that exactly two out of the three people successfully decrypt the password -/
def prob_exactly_two_succeed : ℚ := P1 * P2 * (1 - P3) + P1 * (1 - P2) * P3 + (1 - P1) * P2 * P3

theorem exactly_two_succeed_probability :
  prob_exactly_two_succeed = 5 / 12 :=
sorry

end exactly_two_succeed_probability_l255_255485


namespace exists_real_number_l255_255863

def sequence_nth_term (n : ℕ) : ℕ := 
  if n = 0 then 2 else if n % 5 = 0 then 2 else 3

def sequence_property (s : ℕ → ℕ) : Prop :=
  ∀ (n m : ℕ), (s n = 2 ∧ s m = 2 ∧ n < m) → 
  (∃ k : ℕ, m = n + k + 1 ∧ (∀ (i : ℕ), n < i < m → s i = 3) ∧ k = m - n - 1)

theorem exists_real_number (r : ℝ) : 
  (∀ (s : ℕ → ℕ), sequence_property s) → 
  (∀ (n : ℕ), sequence_nth_term n = 2 ↔ ∃ (m : ℕ), n = 1 + ⌊r * m⌋) → 
  r = 2 + Real.sqrt 3 :=
by
  sorry

end exists_real_number_l255_255863


namespace Nicky_profit_l255_255818

-- Definitions for conditions
def card1_value : ℕ := 8
def card2_value : ℕ := 8
def received_card_value : ℕ := 21

-- The statement to be proven
theorem Nicky_profit : (received_card_value - (card1_value + card2_value)) = 5 :=
by
  sorry

end Nicky_profit_l255_255818


namespace basketball_team_starters_l255_255550

theorem basketball_team_starters :
  let total_players := 18
  let quadruplets := 4
  let remaining_players := total_players - quadruplets
  let starters := 7
  let binom := Nat.choose
in
binom total_players starters - binom remaining_players (starters - quadruplets) = 31460 := 
sorry

end basketball_team_starters_l255_255550


namespace general_pattern_specific_computation_l255_255410

theorem general_pattern (n : ℕ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 :=
by
  sorry

theorem specific_computation : 2000 * 2001 * 2002 * 2003 + 1 = 4006001^2 :=
by
  have h := general_pattern 2000
  exact h

end general_pattern_specific_computation_l255_255410


namespace no_integers_for_sum_of_squares_l255_255579

theorem no_integers_for_sum_of_squares :
  ¬ ∃ a b : ℤ, a^2 + b^2 = 10^100 + 3 :=
by
  sorry

end no_integers_for_sum_of_squares_l255_255579


namespace can_be_cut_with_n_equals_five_can_be_cut_with_n_equals_four_l255_255578

-- Definition of the problem
def unconvex_ngon_can_be_cut (n : ℕ) : Prop :=
  ∃ (parts : fin 3 → set (ℝ × ℝ)), 
    (∃ (polygon1 polygon2 : set (ℝ × ℝ)),
      parts 0 = polygon1 ∧ parts 1 = polygon2 ∧ 
      is_polygon_with_n_sides polygon1 n ∧
      is_polygon_with_n_sides polygon2 n ∧
      congruent_polygon (polygon1 ∪ polygon2) parts 2) ∧ 
    (∀ i, ¬ convex (parts i)) ∧ 
    (∃ l : set (ℝ × ℝ), is_line l ∧ straight_cut parts l) 

-- Theorem for n = 5
theorem can_be_cut_with_n_equals_five : unconvex_ngon_can_be_cut 5 :=
  sorry

-- Theorem for n = 4
theorem can_be_cut_with_n_equals_four : unconvex_ngon_can_be_cut 4 :=
  sorry

end can_be_cut_with_n_equals_five_can_be_cut_with_n_equals_four_l255_255578


namespace tan_power_difference_l255_255278

noncomputable def cot (x : ℝ) : ℝ := 1 / tan x

theorem tan_power_difference (x : ℝ) (h : cos x * (1 / tan x) = sin x^2) :
  tan x^6 - tan x^2 = 1 :=
by
  sorry

end tan_power_difference_l255_255278


namespace tan_six_minus_tan_two_geo_seq_l255_255272

theorem tan_six_minus_tan_two_geo_seq (x : ℝ) (h : (cos x)^2 = (sin x) * (cos x) * (cot x)) :
  (tan x)^6 - (tan x)^2 = (sin x)^2 :=
sorry

end tan_six_minus_tan_two_geo_seq_l255_255272


namespace log2_sufficient_not_necessary_l255_255854

theorem log2_sufficient_not_necessary (x : ℝ) : (log 2 x < 1) → (x < 2) ∧ (∃ x, x < 2 ∧ log 2 x ≥ 1) :=
by 
  sorry

end log2_sufficient_not_necessary_l255_255854


namespace expected_no_advice_l255_255589

theorem expected_no_advice (n : ℕ) (p : ℝ) (h_p : 0 ≤ p ∧ p ≤ 1) : 
  (∑ j in finset.range n, (1 - p)^j) = (1 - (1 - p)^n) / p :=
by
  sorry

end expected_no_advice_l255_255589


namespace correct_system_of_equations_l255_255054

theorem correct_system_of_equations : 
  ∃ (x y : ℕ), x + y = 12 ∧ 4 * x + 3 * y = 40 := by
  -- we are stating the existence of x and y that satisfy both equations given as conditions.
  sorry

end correct_system_of_equations_l255_255054


namespace probability_diff_topic_l255_255951

theorem probability_diff_topic (n : ℕ) (m : ℕ) : 
  n = 6 → m = 5 → 
  (n * m : ℚ) / (6 * 6 : ℚ) = 5 / 6 := 
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end probability_diff_topic_l255_255951


namespace minimum_odd_numbers_in_set_l255_255390

-- Definitions
variable (P : ℝ → ℝ)
variable (degree_P : ℕ)
variable (A_P : Set ℝ)

-- The conditions: P is a polynomial of degree 8, and 8 is included in A_P
def is_polynomial_of_degree_eight (P : ℝ → ℝ) (degree_P : ℕ) : Prop :=
  degree_P = 8

def set_includes_eight (A_P : Set ℝ) : Prop := 
  8 ∈ A_P

-- The goal: prove the minimum number of odd numbers in A_P is 1
theorem minimum_odd_numbers_in_set {P : ℝ → ℝ} {degree_P : ℕ} {A_P : Set ℝ} :
  is_polynomial_of_degree_eight P degree_P → 
  set_includes_eight A_P → 
  ∃ odd_numbers : ℕ, odd_numbers = 1 :=
sorry

end minimum_odd_numbers_in_set_l255_255390


namespace mode_and_median_of_set_l255_255690

theorem mode_and_median_of_set :
  (∃ (A : Set ℕ) (mode median : ℕ), A = {1, 1, 4, 5, 5, 5} ∧ 
    (∀ x ∈ A, x = 5 → mode = 5) ∧ median = (4 + 5) / 2) :=
begin
  use ({1, 1, 4, 5, 5, 5} : Set ℕ),
  sorry
end

end mode_and_median_of_set_l255_255690


namespace average_of_distinct_p_for_polynomial_with_integer_roots_l255_255233

theorem average_of_distinct_p_for_polynomial_with_integer_roots :
  (∑ p in {p | ∃ a b : ℕ, a * b = 18 ∧ a + b = p}, p) / (finset.card {p | ∃ a b : ℕ, a * b = 18 ∧ a + b = p}) = 13 :=
by
  sorry

end average_of_distinct_p_for_polynomial_with_integer_roots_l255_255233


namespace monotonically_increasing_function_satisfying_functional_equation_l255_255978

theorem monotonically_increasing_function_satisfying_functional_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x + y) = f(x) * f(y)) ∧ (∀ a b : ℝ, a < b → f(a) < f(b)) ∧ 
  (f = (λ x, x^3) ∨ f = (λ x, 3^x) ∨ f = (λ x, x^(1/2)) ∨ f = (λ x, (1/2)^x)) → f = (λ x, 3^x) :=
by
  sorry

end monotonically_increasing_function_satisfying_functional_equation_l255_255978


namespace probability_A_B_C_adjacent_l255_255188

theorem probability_A_B_C_adjacent (students : Fin 5 → Prop) (A B C : Fin 5) :
  (students A ∧ students B ∧ students C) →
  (∃ n m : ℕ, n = 48 ∧ m = 12 ∧ m / n = (1 : ℚ) / 4) :=
by
  sorry

end probability_A_B_C_adjacent_l255_255188


namespace printing_press_brochures_l255_255556

theorem printing_press_brochures :
  ∀ (single_page_spreads double_page_spreads pages_per_ad_block page_fraction ads_per_block pages_per_brochure : ℕ),
  single_page_spreads = 20 →
  double_page_spreads = 2 * single_page_spreads →
  pages_per_ad_block = 4 →
  page_fraction = 4 →
  ads_per_block = 4 →
  pages_per_brochure = 5 →
  let total_spread_pages := single_page_spreads + double_page_spreads * 2 in
  let num_ad_blocks := total_spread_pages / pages_per_ad_block in
  let total_ad_pages := num_ad_blocks * ads_per_block / page_fraction in
  let total_pages := total_spread_pages + total_ad_pages in
  total_pages / pages_per_brochure = 25 :=
begin
  intros single_page_spreads double_page_spreads pages_per_ad_block page_fraction ads_per_block pages_per_brochure,
  assume h1 h2 h3 h4 h5 h6,
  let total_spread_pages := single_page_spreads + double_page_spreads * 2,
  let num_ad_blocks := total_spread_pages / pages_per_ad_block,
  let total_ad_pages := num_ad_blocks * ads_per_block / page_fraction,
  let total_pages := total_spread_pages + total_ad_pages,
  have h7 : total_spread_pages = 100, from calc
    total_spread_pages = single_page_spreads + double_page_spreads * 2 : by rw [h2]
    ... = 20 + (2 * 20) * 2 : by rw [h1]
    ... = 20 + 80 : rfl
    ... = 100 : rfl,
  have h8 : num_ad_blocks = 25, from calc
    num_ad_blocks = total_spread_pages / pages_per_ad_block : rfl
    ... = 100 / 4 : by rw [h7, h3]
    ... = 25 : rfl,
  have h9 : total_ad_pages = 25, from calc
    total_ad_pages = num_ad_blocks * ads_per_block / page_fraction : rfl
    ... = 25 * 4 / 4 : by rw [h8, h5, h4]
    ... = 25 : rfl,
  have h10 : total_pages = 125, from calc
    total_pages = total_spread_pages + total_ad_pages : rfl
    ... = 100 + 25 : by rw [h7, h9]
    ... = 125 : rfl,
  show total_pages / pages_per_brochure = 25, from calc
    total_pages / pages_per_brochure = 125 / 5 : by rw [h10, h6]
    ... = 25 : rfl,
end

end printing_press_brochures_l255_255556


namespace leading_digit_of_power_eq_three_l255_255673

open Nat

theorem leading_digit_of_power_eq_three (n : ℕ) (hn : n > 3) : 
  ∃ s t : ℤ, 
  (10^s ≤ 2^n ∧ 2^n < 10^(s+1)) ∧ 
  (10^t ≤ 5^n ∧ 5^n < 10^(t+1)) → 
  ∃ d, first_digit (2^n) = d ∧ first_digit (5^n) = d ∧ d = 3 := 
sorry

end leading_digit_of_power_eq_three_l255_255673


namespace height_of_house_l255_255894

-- Define the given values as constants.
def shadow_house : ℕ := 84
def height_pole : ℕ := 14
def shadow_pole : ℕ := 28

-- State the problem requiring a proof.
theorem height_of_house (h : ℕ) (H1 : shadow_house = 84) (H2 : height_pole = 14) 
  (H3 : shadow_pole = 28) (H4 : shadow_house / shadow_pole = height_of_house / height_pole) :
  h = 42 := 
  sorry

end height_of_house_l255_255894


namespace statement_A_is_incorrect_statement_B_is_correct_statement_C_is_correct_statement_D_is_correct_final_answer_l255_255077

theorem statement_A_is_incorrect {A B : Prop} : 
  (¬(∃ (h : A ∧ B), true) → ¬(A ∨ B) → (A ∨ B) ∧ ¬(A ∧ B) → A ∨ B) := 
sorry

theorem statement_B_is_correct {X : Type} (B : X → Prop) : 
  (∃ (X : ℕ), ∀ (n : ℕ), D (3 * X + 2) = 20) := 
sorry

theorem statement_C_is_correct {X : Type} [measure_space X] :
  (X ∼ N(1, σ^2) → P(X < 4) = 0.79 →  P(X < -2) = 0.21) := 
sorry

theorem statement_D_is_correct {x y : ℝ} : 
  (∃ c k : ℝ, y = ce^{kx} ∧ z = ln y ∧  hat{z} = -4x + 4 ∧ ∃ (ceq : c = e^4) (keq : k = -4)) := sorry

theorem final_answer : true :=
  statement_A_is_incorrect ∧ statement_B_is_correct ∧ statement_C_is_correct ∧ statement_D_is_correct :=
sorry

end statement_A_is_incorrect_statement_B_is_correct_statement_C_is_correct_statement_D_is_correct_final_answer_l255_255077


namespace lasso_probability_l255_255412

theorem lasso_probability :
  let p_event := 1 / 2 in
  let p_not_event := 1 - p_event in
  let p_not_4_attempts := p_not_event ^ 4 in
  let p_at_least_once := 1 - p_not_4_attempts in
  p_at_least_once = 15 / 16 := 
by 
  let p_event := 1 / 2 in
  let p_not_event := 1 - p_event in
  let p_not_4_attempts := p_not_event ^ 4 in
  let p_at_least_once := 1 - p_not_4_attempts in
  sorry

end lasso_probability_l255_255412


namespace domain_of_f_l255_255856

theorem domain_of_f (x : ℝ) : 
  (ln (x + 1) ≠ 0) ∧ (x + 1 > 0) ∧ (4 - x^2 ≥ 0) ↔ (-1 < x ∧ x ≤ 2 ∧ x ≠ 0) :=
begin
  sorry -- proof to be completed
end

end domain_of_f_l255_255856


namespace initial_big_bottles_l255_255958

theorem initial_big_bottles (B : ℝ)
  (initial_small : ℝ := 6000)
  (sold_small : ℝ := 0.11)
  (sold_big : ℝ := 0.12)
  (remaining_total : ℝ := 18540) :
  (initial_small * (1 - sold_small) + B * (1 - sold_big) = remaining_total) → B = 15000 :=
by
  intro h
  sorry

end initial_big_bottles_l255_255958


namespace volume_and_height_of_tetrahedron_l255_255608

def point := ℝ × ℝ × ℝ

variables (A1 A2 A3 A4 : point)

-- Coordinates of the points
def A1 := (-2 : ℝ, -1 : ℝ, -1 : ℝ)
def A2 := (0 : ℝ, 3 : ℝ, 2 : ℝ)
def A3 := (3 : ℝ, 1 : ℝ, -4 : ℝ)
def A4 := (-4 : ℝ, 7 : ℝ, 3 : ℝ)

-- Vector between two points
def vector (p1 p2 : point) : point :=
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

-- Scalar triple product (volume computation)
def scalarTripleProduct (u v w : point) : ℝ :=
  u.1 * (v.2 * w.3 - v.3 * w.2) - u.2 * (v.1 * w.3 - v.3 * w.1) + u.3 * (v.1 * w.2 - v.2 * w.1)

-- Cross product of two vectors
def crossProduct (u v : point) : point :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

-- Magnitude of a vector
def magnitude (u : point) : ℝ :=
  Real.sqrt (u.1^2 + u.2^2 + u.3^2)

-- Volume of the tetrahedron
def tetrahedronVolume (A1 A2 A3 A4 : point) : ℝ :=
  Real.abs (scalarTripleProduct (vector A1 A2) (vector A1 A3) (vector A1 A4)) / 6

-- Area of the triangle A1A2A3
def triangleArea (A1 A2 A3 : point) : ℝ :=
  magnitude (crossProduct (vector A1 A2) (vector A1 A3)) / 2

-- Height from A4 to the base face A1A2A3
def height (A1 A2 A3 A4 : point) : ℝ :=
  (3 * tetrahedronVolume A1 A2 A3 A4) / (triangleArea A1 A2 A3)

theorem volume_and_height_of_tetrahedron :
  tetrahedronVolume A1 A2 A3 A4 = 70 / 3 ∧ height A1 A2 A3 A4 = 140 / Real.sqrt 1021 :=
  by
    sorry

end volume_and_height_of_tetrahedron_l255_255608


namespace inequality_solution_set_l255_255475

theorem inequality_solution_set : 
  { x : ℝ | (1 - x) * (x + 1) ≤ 0 ∧ x ≠ -1 } = { x : ℝ | x < -1 ∨ x ≥ 1 } :=
sorry

end inequality_solution_set_l255_255475


namespace andrew_apples_l255_255979

theorem andrew_apples : ∃ (A n : ℕ), (6 * n = A) ∧ (5 * (n + 2) = A) ∧ (A = 60) :=
by 
  sorry

end andrew_apples_l255_255979


namespace christmas_tree_decorations_l255_255407

def is_valid_design (triangle : Fin 3 → Bool) : Prop :=
(triangle 0 = triangle 1 ∧ triangle 2 ≠ triangle 0) ∨
(triangle 0 = triangle 2 ∧ triangle 1 ≠ triangle 0) ∨
(triangle 1 = triangle 2 ∧ triangle 0 ≠ triangle 1)

def count_distinct_designs : Nat :=
2

theorem christmas_tree_decorations : ∃ n : Nat, n = 2 ∧ (
    ∀ triangle : Fin 3 → Bool, ∃ valid_designs : Fin 3 → Fin 2 → Bool,
    is_valid_design valid_designs
) :=
begin
  use count_distinct_designs,
  split,
  { reflexivity },
  { intro triangle,
    existsi triangle,
    sorry
  }
end

end christmas_tree_decorations_l255_255407


namespace deformable_to_triangle_l255_255124

-- Definition of the planar polygon with n sides
structure Polygon (n : ℕ) := 
  (vertices : Fin n → ℝ × ℝ) -- This is a simplified representation of a planar polygon using vertex coordinates

noncomputable def canDeformToTriangle (poly : Polygon n) : Prop := sorry

theorem deformable_to_triangle (n : ℕ) (h : n > 4) (poly : Polygon n) : canDeformToTriangle poly := 
  sorry

end deformable_to_triangle_l255_255124


namespace field_area_change_l255_255557

noncomputable def length_increase_perc : ℝ := 0.35
noncomputable def width_decrease_perc : ℝ := 0.14

theorem field_area_change 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b): 
  let new_length := a * (1 + length_increase_perc),
      new_width := b * (1 - width_decrease_perc),
      initial_area := a * b,
      new_area := new_length * new_width,
      area_change := new_area - initial_area 
  in (area_change / initial_area) * 100 = 16.1 :=
by
  sorry

end field_area_change_l255_255557


namespace greatest_integer_value_x_l255_255500

theorem greatest_integer_value_x :
  ∀ x : ℤ, (∃ k : ℤ, x^2 + 2 * x + 9 = k * (x - 5)) ↔ x ≤ 49 :=
by
  sorry

end greatest_integer_value_x_l255_255500


namespace green_dots_first_row_l255_255411

theorem green_dots_first_row :
  ∀ (n : ℕ), (row_dots : ℕ → ℕ) (row_dots 2 = 6) (row_dots 3 = 9) (row_dots 4 = 12) (row_dots 5 = 15),
  row_dots 1 = 3 :=
by
  sorry -- proof steps are omitted as per the instructions

end green_dots_first_row_l255_255411


namespace mid_length_AT_range_l255_255692

noncomputable def ellipse (a b : ℝ) (h : a > 0) : set (ℝ × ℝ) :=
{ p | (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1 }

def eccentricity (a b : ℝ) (h : a > 0) : ℝ :=
(1 - (b ^ 2) / (a ^ 2)).sqrt

def range_mid_length_AT (a b : ℝ) (h : a > 0) (e : ℝ) (him : e = (sqrt 2) / 2)
  (λ : ℝ) (hλ : 1 ≤ λ ∧ λ ≤ 2) : set ℝ :=
{ l | ∃ x y z w : ℝ, ((x / a) ^ 2 + (y / b) ^ 2 = 1) ∧ ((z / a) ^ 2 + (w / b) ^ 2 = 1) ∧
                      (λ * (x^2 + y^2).sqrt = (z^2 + w^2).sqrt) ∧
                      ∃ t : ℝ, (1 / √(t ^ 2 + 2) ≤ t ∧ t ≤ (7 / 16)) ∧
                               (√(2 / (t ^ 2 + 2) ^ 2 - 7 / (t ^ 2 + 2) + 4) ∈ l) ∧
                               (1 ≤ √(2 / (t ^ 2 + 2) ^ 2 - 7 / (t ^ 2 + 2) + 4) ∧
                               √(2 / (t ^ 2 + 2) ^ 2 - 7 / (t ^ 2 + 2) + 4) ≤ (13 * sqrt 2) / 16)}

theorem mid_length_AT_range (a b : ℝ) (h : a > 0) (e : ℝ) (him : e = (sqrt 2) / 2)
  (λ : ℝ) (hλ : 1 ≤ λ ∧ λ ≤ 2) :
  range_mid_length_AT a b h e him λ hλ = { l | 1 ≤ l ∧ l ≤ (13 * sqrt 2) / 16 } :=
sorry

end mid_length_AT_range_l255_255692


namespace percent_students_75_84_l255_255103

-- Defining the frequencies based on the given problem statement.
def freq_95_100 := 4
def freq_85_94 := 5
def freq_75_84 := 8
def freq_65_74 := 6
def freq_55_64 := 4
def freq_below_55 := 3

-- Compute the total number of students.
def total_students := freq_95_100 + freq_85_94 + freq_75_84 + freq_65_74 + freq_55_64 + freq_below_55

-- Compute the percentage of students in the 75%-84% range.
def percentage_75_84 := (freq_75_84 : ℝ) / total_students * 100

-- State the theorem to be proved.
theorem percent_students_75_84 : percentage_75_84 ≈ 26.67 := by
  unfold percentage_75_84 total_students
  norm_num
  sorry

end percent_students_75_84_l255_255103


namespace polar_curve_two_intersecting_lines_l255_255025

theorem polar_curve_two_intersecting_lines (ρ θ : ℝ) :
  ρ * (cos θ ^ 2 - sin θ ^ 2) = 0 → (∀ (x y : ℝ), y = x ∨ y = -x) :=
by
  sorry

end polar_curve_two_intersecting_lines_l255_255025


namespace probability_diff_topic_l255_255952

theorem probability_diff_topic (n : ℕ) (m : ℕ) : 
  n = 6 → m = 5 → 
  (n * m : ℚ) / (6 * 6 : ℚ) = 5 / 6 := 
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end probability_diff_topic_l255_255952


namespace expectation_zero_l255_255457

noncomputable def X_distribution : List (ℝ × ℝ) :=
  [(1, 0.1), (2, 0.3), (3, 0.2), (4, 0.3), (5, 0.1)]

theorem expectation_zero :
  let E_X := ∑ x in X_distribution, x.1 * x.2 in
  E_X = 3 → -- Given that E[X] = 3
  ∑ x in X_distribution, (x.1 - E_X) * x.2 = 0 :=
by
  intros E_X hE
  sorry

end expectation_zero_l255_255457


namespace minimum_odd_numbers_in_set_l255_255392

-- Definitions
variable (P : ℝ → ℝ)
variable (degree_P : ℕ)
variable (A_P : Set ℝ)

-- The conditions: P is a polynomial of degree 8, and 8 is included in A_P
def is_polynomial_of_degree_eight (P : ℝ → ℝ) (degree_P : ℕ) : Prop :=
  degree_P = 8

def set_includes_eight (A_P : Set ℝ) : Prop := 
  8 ∈ A_P

-- The goal: prove the minimum number of odd numbers in A_P is 1
theorem minimum_odd_numbers_in_set {P : ℝ → ℝ} {degree_P : ℕ} {A_P : Set ℝ} :
  is_polynomial_of_degree_eight P degree_P → 
  set_includes_eight A_P → 
  ∃ odd_numbers : ℕ, odd_numbers = 1 :=
sorry

end minimum_odd_numbers_in_set_l255_255392


namespace valid_pairs_count_eq_two_l255_255949

theorem valid_pairs_count_eq_two (a b : ℕ) (h : a < b) 
  (h_pos: 0 < a) (h_pos_b: 0 < b) (border_width : ℕ) 
  (h_border: border_width = 2) :
  let A_outer := a * b,
      A_inner := (a - 2 * border_width) * (b - 2 * border_width),
      A_border := A_outer - A_inner
  in 3 * A_inner = A_outer →
     (∃ ab_pairs : list (ℕ × ℕ), 
        (∀ (p : ℕ × ℕ), p ∈ ab_pairs → (p.1 < p.2 ∧ p.1 > 0 ∧ p.2 > 0 ∧ 
          let A_outer' := p.1 * p.2,
              A_inner' := (p.1 - 2 * border_width) * (p.2 - 2 * border_width)
          in 3 * A_inner' = A_outer') 
         ∧ ab_pairs.length = 2) := 
  sorry

end valid_pairs_count_eq_two_l255_255949


namespace find_circumcircle_radius_l255_255418

noncomputable def circumcircle_radius (A B C M N : Point) (r : ℝ) : Prop :=
let dist := Euclidean.dist in
M.on_bisector_of_angle ∠ BAC ∧
N.extends_line A B ∧
dist A C = 1 ∧
dist A M = 1 ∧
∠ANM = ∠CNM ∧
circumradius_triangle C N M = r

theorem find_circumcircle_radius (A B C M N : Point) :
  circumcircle_radius A B C M N 1 :=
sorry

end find_circumcircle_radius_l255_255418


namespace P_x_x_cannot_have_odd_degree_l255_255867

noncomputable def P (x y : ℕ) : ℕ := sorry

-- Condition given in the problem
axiom condition (n : ℕ) : (∀ y, P n y = 0 ∨ ∃ k ≤ n, k = n) ∧ (∀ x, P x n = 0 ∨ ∃ k ≤ n, k = n)

-- The statement to be proved
theorem P_x_x_cannot_have_odd_degree :
  ∀ P : ℕ → ℕ → ℕ, 
  (∀ n : ℕ, (∀ y : ℕ, P n y = 0 ∨ ∃ k, k ≤ n ∧ P n y = k) ∧ (∀ x : ℕ, P x n = 0 ∨ ∃ k, k ≤ n ∧ P x n = k)) →
  ¬(∃ m, (deg (P x x) = 2 * m + 1)) :=
sorry

end P_x_x_cannot_have_odd_degree_l255_255867


namespace order_of_t_t2_neg_t_l255_255210

theorem order_of_t_t2_neg_t (t : ℝ) (h : t^2 + t < 0) : t < t^2 ∧ t^2 < -t :=
by
  sorry

end order_of_t_t2_neg_t_l255_255210


namespace min_students_in_class_l255_255752

theorem min_students_in_class (b g : ℕ) (hb : 3 * b = 4 * g) : b + g = 7 :=
sorry

end min_students_in_class_l255_255752


namespace surface_area_rectangular_solid_l255_255920

def length := 5
def width := 4
def depth := 1

def surface_area (l w d : ℕ) := 2 * (l * w) + 2 * (l * d) + 2 * (w * d)

theorem surface_area_rectangular_solid : surface_area length width depth = 58 := 
by 
sorry

end surface_area_rectangular_solid_l255_255920


namespace hyperbola_eccentricity_value_l255_255686

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  let c := Real.sqrt (a^2 + b^2) in
  c / a

theorem hyperbola_eccentricity_value {a b : ℝ} (h_a : 0 < a) (h_b : 0 < b) :
  let e := hyperbola_eccentricity a b h_a h_b in
  e = (Real.sqrt 5 + 1) / 2 :=
by
  sorry

end hyperbola_eccentricity_value_l255_255686


namespace geometric_sequence_product_l255_255320

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_product {a : ℕ → ℝ}
    (h_geom : geometric_sequence a)
    (h_log : log 2 (a 2 * a 98) = 4) :
    a 40 * a 60 = 16 :=
by 
  sorry

end geometric_sequence_product_l255_255320


namespace find_real_solutions_eqns_l255_255921

theorem find_real_solutions_eqns :
  { x y z : ℝ //  
    3 * (x^2 + y^2 + z^2) = 1 ∧ 
    x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = x * y * z * (x + y + z)^2  } = 
  { (0, 0, (sqrt 3) / 3), 
    (0, 0, - (sqrt 3) / 3),
    (0, (sqrt 3) / 3, 0), 
    (0, - (sqrt 3) / 3, 0),
    ((sqrt 3) / 3, 0, 0),  
    (- (sqrt 3) / 3, 0, 0),  
    (1 / 3, 1 / 3, 1 / 3),    
    (- 1 / 3, - 1 / 3, - 1 / 3)} :=
by {
  sorry
}

end find_real_solutions_eqns_l255_255921


namespace express_in_scientific_notation_l255_255166

theorem express_in_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 388800 = a * 10 ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.888 ∧ n = 5 :=
by
  sorry

end express_in_scientific_notation_l255_255166


namespace find_k_l255_255762

noncomputable def point (x y : ℝ) := (x, y)
noncomputable def inverse_point (P : ℝ × ℝ) : ℝ × ℝ := (1 / P.1, 1 / P.2)

theorem find_k :
  ∃ (k : ℝ), 
    let A := (a, -a + 1),
        B := (b, -b + 1),
        A' := inverse_point A,
        B' := inverse_point B in
      (b = a + 2) ∧
      (dist A B = 2 * Real.sqrt 2) ∧ 
      (A'.2 = k / A'.1) ∧
      (B'.2 = k / B'.1) → 
    k = -4 / 3 := 
by
  sorry

end find_k_l255_255762


namespace triangle_eqns_l255_255046

variable (A B C : ℝ × ℝ)
variable (midpoint : ℝ × ℝ)
variable (median_eqn altitude_eqn : ℝ × ℝ → Prop)

-- Define the vertices of the triangle
def A := (4, 0)
def B := (6, 7)
def C := (0, 3)

-- Define the midpoint of BC
def midpoint := ((6 + 0) / 2, (7 + 3) / 2)

-- Equation of the median on side BC
def median_eqn (p : ℝ × ℝ) : Prop := 5 * p.1 + p.2 = 20

-- Equation of the altitude on side BC
def altitude_eqn (p : ℝ × ℝ) : Prop := 3 * p.1 + 2 * p.2 = 12

-- The main theorem to prove
theorem triangle_eqns :
  (midpoint = (3, 5)) ∧ (median_eqn midpoint) ∧ (altitude_eqn A) :=
by
  have midpoint_correct : midpoint = (3, 5) := by
    sorry -- midpoint computation
  
  have median_correct : median_eqn midpoint := by
    sorry -- equation of median

  have altitude_correct : altitude_eqn A := by
    sorry -- equation of altitude

  exact ⟨midpoint_correct, median_correct, altitude_correct⟩

end triangle_eqns_l255_255046


namespace fraction_meaningful_l255_255074

theorem fraction_meaningful (x : ℝ) (h : x = -1) : ((x^2 + 1) ≠ 0) :=
by {
  rw h,
  calc
    (-1)^2 + 1 = 1 + 1 : by norm_num,
    1 + 1 ≠ 0 : by norm_num
}

end fraction_meaningful_l255_255074


namespace repeating_decimal_to_fraction_l255_255639

theorem repeating_decimal_to_fraction : (0.36 : ℝ) = (11 / 30 : ℝ) :=
sorry

end repeating_decimal_to_fraction_l255_255639


namespace sum_divisible_by_4003_l255_255003

open Nat

theorem sum_divisible_by_4003 :
  let n := (∏ k in Finset.range (2001 + 1), k)
  let m := (∏ k in Finset.Ico 2002 (4002 + 1), k)
  4003 ∣ (n + m) := by
  let n := (∏ k in Finset.range (2001 + 1), k)
  let m := (∏ k in Finset.Ico 2002 (4002 + 1), k)
  sorry

end sum_divisible_by_4003_l255_255003


namespace marbles_in_jar_l255_255108

theorem marbles_in_jar (x : ℕ)
  (h1 : \frac{1}{2} * x + \frac{1}{4} * x + 27 + 14 = x) : x = 164 := sorry

end marbles_in_jar_l255_255108


namespace min_odd_in_A_P_l255_255366

-- Define the polynomial P of degree 8
variable (P : ℝ → ℝ)
-- Assume P is a polynomial of degree 8
axiom degree_P : degree P = 8

-- Define the set A_P as the set of all x for which P(x) gives a certain value
def A_P (c : ℝ) : Set ℝ := { x | P x = c }

-- Given condition
axiom include_8 : 8 ∈ A_P (P 8)

-- Define if a number is odd
def is_odd (n : ℝ) : Prop := ∃k : ℤ, n = 2 * k + 1

-- The minimum number of odd numbers that are in the set A_P,
-- given that the number 8 is included in A_P, is 1

theorem min_odd_in_A_P : ∀ P, (degree P = 8) → 8 ∈ A_P (P 8) → ∃ x ∈ A_P (P 8), is_odd x := 
by
  sorry -- proof goes here

end min_odd_in_A_P_l255_255366


namespace problem_A_problem_B_problem_C_problem_D_l255_255701

def center := (3, 4)

def radius := sqrt 8

def M1 := (2, 0)

def M2 := (-2, 0)

def O := (0, 0)

def N1 := (1, 0)

def N2 := (-1, 0)

theorem problem_A : ∀ (A: ℝ × ℝ), (A ∈ circle center radius) → dist A M2 ≤ sqrt 41 + 2 * sqrt 2 := 
by sorry

theorem problem_B : ∀ (A: ℝ × ℝ), (A ∈ circle center radius) → 
  area (triangle A M1 M2) ≥ 8 :=
by sorry

theorem problem_C : ∀ (A: ℝ × ℝ), (A ∈ circle center radius) → 
  angle (A, N2, N1) ≥ 15 :=
by sorry

theorem problem_D : ∀ (A: ℝ × ℝ), (A ∈ circle center radius) → 
  dot_product (A, N1) (A, N2) ≥ 32 - 20 * sqrt 2 :=
by sorry

end problem_A_problem_B_problem_C_problem_D_l255_255701


namespace angle_between_vectors_l255_255226

variables {V : Type*} [inner_product_space ℝ V]
variables {a b : V} (ha : a ≠ 0) (hb : b ≠ 0)
variables (h1 : inner a (a + b) = 0) (h2 : 2 * ‖a‖ = ‖b‖)

theorem angle_between_vectors (ha : a ≠ 0) (hb : b ≠ 0) (h1 : inner a (a + b) = 0) (h2 : 2 * ‖a‖ = ‖b‖) :
  real.angle.cos (real.angle a b) = -1/2 :=
sorry

end angle_between_vectors_l255_255226


namespace calc_expression_l255_255152

theorem calc_expression :
  (2 + Real.sqrt 3)^0 + 3 * Real.tan (π / 6) - Real.abs (Real.sqrt 3 - 2) + (1 / 2)^(-1) = 1 + 2 * Real.sqrt 3 := by
  sorry

end calc_expression_l255_255152


namespace area_quadrilateral_ADEC_l255_255310

/--
Given the following conditions:
1. Angle C = 90 degrees.
2. AD = DB.
3. DE is perpendicular to AB.
4. AB = 30 units.
5. AC = 18 units.
6. DE = 8 units.

Prove that the area of quadrilateral ADEC is 96 square units.
-/
theorem area_quadrilateral_ADEC
  (A B C D E : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
  (angle_C : ∠C = π / 2)
  (AD_eq_DB : dist A D = dist D B)
  (DE_perp_AB : ∠DE = π / 2)
  (AB_30 : dist A B = 30)
  (AC_18 : dist A C = 18)
  (DE_8 : dist D E = 8) : 
  area_quadrilateral A D E C = 96 :=
sorry

end area_quadrilateral_ADEC_l255_255310


namespace milan_monthly_fee_l255_255192

-- Define the constants and conditions
def monthly_fee (m : ℝ) (cents_per_minute : ℝ) (total_bill : ℝ) (minutes : ℕ) : Prop :=
  total_bill = m + (cents_per_minute * minutes)

-- Stating the conditions
def conditions : Prop :=
  monthly_fee m 0.12 23.36 178

-- Stating the main problem: Prove the monthly fee is $2.00
theorem milan_monthly_fee : conditions → m = 2.00 :=
by
  sorry

end milan_monthly_fee_l255_255192


namespace exists_n_l255_255787

theorem exists_n (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k > 0 → ¬(2^n ∣ a^k + b^k + c^k) :=
by
  sorry

end exists_n_l255_255787


namespace product_of_roots_l255_255336

noncomputable def Q : Polynomial ℚ := Polynomial.Cubic(1, 0, -6, -24)

theorem product_of_roots : Q.roots.prod = -24 := by sorry

end product_of_roots_l255_255336


namespace perimeter_of_region_l255_255452

-- Given definitions
def area_of_region (area : ℝ) := area = 294
def number_of_squares (n : ℕ) := n = 6

-- Question to be proved
theorem perimeter_of_region : 
  ∀ (area n : ℝ), area_of_region area → number_of_squares n → 
  (2 * (n * (area / n) ^ (1 / 2)) = 98) := by
  sorry

end perimeter_of_region_l255_255452


namespace minimum_odd_in_A_P_l255_255379

-- Define the polynomial and its properties
def polynomial_degree8 (P : ℝ[X]) : Prop :=
  P.degree = 8

-- Define the set A_P
def A_P (P : ℝ[X]) : set ℝ :=
  {x | P.eval x = 8}

-- Main theorem statement
theorem minimum_odd_in_A_P (P : ℝ[X]) (hP : polynomial_degree8 P) : ∃ n : ℕ, n = 1 ∧ ∃ x ∈ A_P P, odd x :=
sorry

end minimum_odd_in_A_P_l255_255379


namespace sample_size_is_100_l255_255570

variable (Population : Set Student)
variable (Sample : Set Student)
variable (n m : Nat)
variable (selected : Sample ⊆ Population)
variable (Population_size : Population.card = 1000)
variable (Sample_size : Sample.card = 100)

theorem sample_size_is_100 : Sample.card = 100 :=
by
  rw [Sample_size]
  sorry

end sample_size_is_100_l255_255570


namespace find_n_l255_255221

variable (x n : ℝ)
variable h1 : log 10 (sin x) + log 10 (cos x) = -2
variable h2 : log 10 (sin x + cos x) = (1 / 2) * (log 10 n + 1)

theorem find_n : n = 51 / 100 :=
by
  sorry

end find_n_l255_255221


namespace four_people_complete_task_in_18_days_l255_255883

theorem four_people_complete_task_in_18_days :
  (forall r : ℝ, (3 * 24 * r = 1) → (4 * 18 * r = 1)) :=
by
  intro r
  intro h
  sorry

end four_people_complete_task_in_18_days_l255_255883


namespace complex_solution_l255_255162

theorem complex_solution (z : ℂ) (h : 3 * z - 4 * (complex.I * conj z) = -8 + 5 * complex.I) :
  z = (4 / 7) + (17 / 7) * complex.I :=
by sorry

end complex_solution_l255_255162


namespace tan_power_difference_l255_255277

noncomputable def cot (x : ℝ) : ℝ := 1 / tan x

theorem tan_power_difference (x : ℝ) (h : cos x * (1 / tan x) = sin x^2) :
  tan x^6 - tan x^2 = 1 :=
by
  sorry

end tan_power_difference_l255_255277


namespace find_c_and_general_formula_l255_255768

noncomputable def sequence (c : ℝ) : ℕ → ℝ
| 1     := 2
| (n+2) := sequence (n+1) + c * (n+1)

theorem find_c_and_general_formula (c : ℝ) (a : ℕ → ℝ) :
  (a 1 = 2) ∧ (∀ n, a (n + 1) = a n + c * n) ∧ (∃ r ≠ 1, a 2 / a 1 = r ∧ a 3 / a 2 = r)
  → c = 2 ∧ (∀ n, a n = n^2 - n + 2) :=
by 
  sorry

end find_c_and_general_formula_l255_255768


namespace arithmetic_sequence_common_difference_l255_255236

theorem arithmetic_sequence_common_difference (a₁ d : ℝ) :
  let S_n (n : ℝ) := n / 2 * (2 * a₁ + (n - 1) * d)
  in 2 * S_n 3 - 3 * S_n 2 = 12 → d = 4 :=
by
  -- Definitions and conditions
  let S_n := fun n => n / 2 * (2 * a₁ + (n - 1) * d)
  have eq1 : 2 * S_n 3 - 3 * S_n 2 = 12 := sorry
  show d = 4, from sorry

end arithmetic_sequence_common_difference_l255_255236


namespace calculate_expression_l255_255609

theorem calculate_expression : (1 / 2) ^ (-1) + (Real.pi - 3.14) ^ 0 - abs (-3) + Real.sqrt 12 = 2 * Real.sqrt 3 :=
by
  sorry

end calculate_expression_l255_255609


namespace min_odd_in_A_P_l255_255362

-- Define the polynomial P of degree 8
variable (P : ℝ → ℝ)
-- Assume P is a polynomial of degree 8
axiom degree_P : degree P = 8

-- Define the set A_P as the set of all x for which P(x) gives a certain value
def A_P (c : ℝ) : Set ℝ := { x | P x = c }

-- Given condition
axiom include_8 : 8 ∈ A_P (P 8)

-- Define if a number is odd
def is_odd (n : ℝ) : Prop := ∃k : ℤ, n = 2 * k + 1

-- The minimum number of odd numbers that are in the set A_P,
-- given that the number 8 is included in A_P, is 1

theorem min_odd_in_A_P : ∀ P, (degree P = 8) → 8 ∈ A_P (P 8) → ∃ x ∈ A_P (P 8), is_odd x := 
by
  sorry -- proof goes here

end min_odd_in_A_P_l255_255362


namespace thales_circles_area_l255_255165

open set real

noncomputable def thales_area_equivalence (R : ℝ) : Prop :=
  let r := R / 2 in
  let C := metric.ball (0:ℝ×ℝ) R in
  let S1 := metric.ball (0:ℝ×ℝ) r in
  let S2 := metric.ball (r, 0) r in
  let S3 := metric.ball (0, r) r in
  let S4 := metric.ball (-r, 0) r in
  let outside_circles := C \ (S1 ∪ S2 ∪ S3 ∪ S4) in
  let exactly_two_overlap :=
    (S1 ∩ (S2 ∪ S3 ∪ S4) ∪ S2 ∩ (S1 ∪ S3 ∪ S4) ∪ S3 ∩ (S1 ∪ S2 ∪ S4) ∪ S4 ∩ (S1 ∪ S2 ∪ S3)) in
  meas outside_circles = meas exactly_two_overlap

theorem thales_circles_area (R : ℝ) : thales_area_equivalence R :=
sorry

end thales_circles_area_l255_255165


namespace f_monotonically_decreasing_l255_255999

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - Real.log (2 * x)

theorem f_monotonically_decreasing :
  ∀ x : ℝ, (0 < x ∧ x < Real.sqrt 2 / 2) → f '(x) < 0 :=
by
  sorry

end f_monotonically_decreasing_l255_255999


namespace tank_salt_solution_l255_255521

theorem tank_salt_solution (x : ℝ) (h1 : (0.20 * x + 14) / ((3 / 4) * x + 21) = 1 / 3) : x = 140 :=
sorry

end tank_salt_solution_l255_255521


namespace sum_of_15_consecutive_integers_perfect_square_l255_255873

open Nat

-- statement that defines the conditions and the objective of the problem
theorem sum_of_15_consecutive_integers_perfect_square :
  ∃ n k : ℕ, 15 * (n + 7) = k^2 ∧ 15 * (n + 7) ≥ 225 := 
sorry

end sum_of_15_consecutive_integers_perfect_square_l255_255873


namespace quadratic_inequality_solution_l255_255735

open Real

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 15 < 0) : 3 < x ∧ x < 5 :=
sorry

end quadratic_inequality_solution_l255_255735


namespace ratio_Bipin_Alok_l255_255605

-- Definitions based on conditions
def Alok_age : Nat := 5
def Chandan_age : Nat := 10
def Bipin_age : Nat := 30
def Bipin_age_condition (B C : Nat) : Prop := B + 10 = 2 * (C + 10)

-- Statement to prove
theorem ratio_Bipin_Alok : 
  Bipin_age_condition Bipin_age Chandan_age -> 
  Alok_age = 5 -> 
  Chandan_age = 10 -> 
  Bipin_age / Alok_age = 6 :=
by
  sorry

end ratio_Bipin_Alok_l255_255605


namespace imons_no_entanglements_l255_255102

-- Define the fundamental structure for imons and their entanglements.
universe u
variable {α : Type u}

-- Define a graph structure to represent imons and their entanglement.
structure Graph (α : Type u) where
  vertices : Finset α
  edges : Finset (α × α)
  edge_sym : ∀ {x y}, (x, y) ∈ edges → (y, x) ∈ edges

-- Define the operations that can be performed on imons.
structure ImonOps (G : Graph α) where
  destroy : {v : α} → G.vertices.card % 2 = 1
  double : Graph α

-- Prove the main theorem
theorem imons_no_entanglements (G : Graph α) (op : ImonOps G) : 
  ∃ seq : List (ImonOps G), ∀ g : Graph α, g ∈ (seq.map (λ h => h.double)) → g.edges = ∅ :=
by
  sorry -- The proof would be constructed here.

end imons_no_entanglements_l255_255102


namespace tangency_of_circumcircle_PCQ_to_l_l255_255119

open EuclideanGeometry

-- Definitions of the geometric entities involved
variable {A B C D X Y P Q: Point}
variable {l: Line}

-- Conditions of the problem
axiom rhombus_ABCD : IsRhombus A B C D
axiom line_through_C : OnLine l C
axiom X_on_AB_extension : OnLine l X ∧ Collinear A B X
axiom Y_on_AD_extension : OnLine l Y ∧ Collinear A D Y
axiom P_on_circumcircle_AXY : OnCircumcircle P A X Y
axiom Q_on_circumcircle_AXY : OnCircumcircle Q A X Y
axiom P_on_DX : OnLine (LineThrough D X) P
axiom Q_on_BY : OnLine (LineThrough B Y) Q

-- Question: Prove that the circumcircle of triangle PCQ is tangent to line l
theorem tangency_of_circumcircle_PCQ_to_l:
  TangentToCircumcircle l P Q C :=
sorry

end tangency_of_circumcircle_PCQ_to_l_l255_255119


namespace problem_solution_l255_255995

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def multiplicative_inverse (a m : ℕ) (inv : ℕ) : Prop := 
  (a * inv) % m = 1

theorem problem_solution :
  is_right_triangle 60 144 156 ∧ multiplicative_inverse 300 3751 3618 :=
by
  sorry

end problem_solution_l255_255995


namespace tan_cot_identity_simplify_expression_l255_255841

theorem tan_cot_identity (θ : ℝ) : cot θ - 2 * cot (2 * θ) = tan θ :=
sorry

theorem simplify_expression (x : ℝ) :
  (tan x + 4 * tan (2 * x) + 8 * tan (4 * x) + 16 * cot (8 * x) + 32 * cot (16 * x)) = cot x :=
by
  have cot_id1 := tan_cot_identity x
  have cot_id2 := tan_cot_identity (2 * x)
  have cot_id3 := tan_cot_identity (4 * x)
  have cot_id4 := tan_cot_identity (8 * x)
  sorry

end tan_cot_identity_simplify_expression_l255_255841


namespace solve_for_a_l255_255732

theorem solve_for_a (a : ℝ) (h : (2 + complex.of_real a * complex.I) * (complex.of_real a - 2 * complex.I) = -4 * complex.I) : a = 0 :=
sorry

end solve_for_a_l255_255732


namespace A_empty_iff_A_single_element_iff_and_value_A_at_most_one_element_iff_l255_255261

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 - 3 * x + 2 = 0}

-- Statement for (1)
theorem A_empty_iff (a : ℝ) : A a = ∅ ↔ a ∈ Set.Ioi 0 :=
sorry

-- Statement for (2)
theorem A_single_element_iff_and_value (a : ℝ) : 
  (∃ x, A a = {x}) ↔ (a = 0 ∨ a = 9 / 8) ∧ A a = {2 / 3} :=
sorry

-- Statement for (3)
theorem A_at_most_one_element_iff (a : ℝ) : 
  (∃ x, A a = {x} ∨ A a = ∅) ↔ (a = 0 ∨ a ∈ Set.Ici (9 / 8)) :=
sorry

end A_empty_iff_A_single_element_iff_and_value_A_at_most_one_element_iff_l255_255261


namespace repeating_decimal_to_fraction_l255_255636

theorem repeating_decimal_to_fraction : (0.3666666 : ℚ) = 11 / 30 :=
by sorry

end repeating_decimal_to_fraction_l255_255636


namespace second_order_derivative_parametric_l255_255922

noncomputable def x (t : ℝ) : ℝ := t ^ (1/2)
noncomputable def y (t : ℝ) : ℝ := (t - 1) ^ (1/3)

-- Define the first derivative of x with respect to t
noncomputable def dx_dt (t : ℝ) : ℝ :=
  deriv x t

-- Define the first derivative of y with respect to t
noncomputable def dy_dt (t : ℝ) : ℝ :=
  deriv y t

-- Define dy/dx
noncomputable def dy_dx (t : ℝ) : ℝ :=
  dy_dt t / dx_dt t

-- Define the second derivative y''_xx
noncomputable def d2y_xx (t : ℝ) : ℝ :=
  deriv (λ t, dy_dx t) t / dx_dt t

-- The theorem that matches the question to the answer
theorem second_order_derivative_parametric :
  ∀ t : ℝ, d2y_xx t = - (4 * (t + 3)) / (9 * ((t - 1) ^ (5/3))) :=
by
  sorry

end second_order_derivative_parametric_l255_255922


namespace exists_pair_divisible_by_n_exists_multiple_digits_0_or_1_l255_255927

-- Problem 1: 
theorem exists_pair_divisible_by_n (n : ℕ) (a : Fin n.succ → ℤ) :
  ∃ (i j : Fin n.succ), i ≠ j ∧ (a i - a j) % n = 0 :=
sorry

-- Problem 2:
theorem exists_multiple_digits_0_or_1 (n : ℕ) :
  ∃ (k : ℕ), k ≤ 10^n ∧ (k % n = 0) ∧ (∀ d, d ∈ Int.digits 10 k → d = 0 ∨ d = 1) :=
sorry

end exists_pair_divisible_by_n_exists_multiple_digits_0_or_1_l255_255927


namespace problem_trapezoid_l255_255767

noncomputable def ratio_of_areas (AB CD : ℝ) (h : ℝ) (ratio : ℝ) :=
  let area_trapezoid := (AB + CD) * h / 2
  let area_triangle_AZW := (4 * h) / 15
  ratio = area_triangle_AZW / area_trapezoid

theorem problem_trapezoid :
  ratio_of_areas 2 5 h (8 / 105) :=
by
  sorry

end problem_trapezoid_l255_255767


namespace unique_triangle_areas_l255_255630

-- Define the conditions of the problem
variables {W X Y Z P Q : Type} [dist : W → X → ℝ] [dist : X → Y → ℝ] 
          [dist : Y → Z → ℝ] [dist : Z → W → ℝ]
          [dist : P → Q → ℝ]

-- Creating aliases for the given distances
def WX := dist W X
def XY := dist X Y
def YZ := dist Y Z
def PQ := dist P Q

-- Conditions given in the problem
axiom h_WX : WX = 1
axiom h_XY : XY = 2
axiom h_YZ : YZ = 3
axiom h_PQ : PQ = 4

-- Define a statement for the theorem to prove
theorem unique_triangle_areas (W X Y Z P Q : Type)
  (dist : W → X → ℝ) [dist : X → Y → ℝ] [dist : Y → Z → ℝ] [dist : P → Q → ℝ]
  (h_WX : dist W X = 1) (h_XY : dist X Y = 2) (h_YZ : dist Y Z = 3) (h_PQ : dist P Q = 4) :
  ∃ (area_vals : set ℝ), area_vals = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0} ∧
  ∀ (A B C : Type), A ∈ {W, X, Y, Z, P, Q} ∧ B ∈ {W, X, Y, Z, P, Q} ∧ C ∈ {W, X, Y, Z, P, Q} 
  ∧ (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧ collinear A B ∧ collinear B C → (∃ (area : ℝ), area ∈ area_vals) :=
by sorry

end unique_triangle_areas_l255_255630


namespace grunters_win_at_least_3_of_6_l255_255450

open Probability.ProbabilityTheory
open Nnreal BigOperators

noncomputable def grunters_win_probability : ℚ :=
1 - (∑ k in finset.range 3, (nat.choose 6 k : ℚ) * (2/3)^k * (1/3)^(6 - k))

theorem grunters_win_at_least_3_of_6 :
  grunters_win_probability = 656 / 729 := sorry

end grunters_win_at_least_3_of_6_l255_255450


namespace wendy_baked_29_cookies_l255_255065

variables (cupcakes : ℕ) (pastries_taken_home : ℕ) (pastries_sold : ℕ)

def total_initial_pastries (cupcakes pastries_taken_home pastries_sold : ℕ) : ℕ :=
  pastries_taken_home + pastries_sold

def cookies_baked (total_initial_pastries cupcakes : ℕ) : ℕ :=
  total_initial_pastries - cupcakes

theorem wendy_baked_29_cookies :
  cupcakes = 4 →
  pastries_taken_home = 24 →
  pastries_sold = 9 →
  cookies_baked (total_initial_pastries cupcakes pastries_taken_home pastries_sold) cupcakes = 29 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end wendy_baked_29_cookies_l255_255065


namespace min_odd_in_A_P_l255_255382

-- Define the polynomial P of degree 8
variable {R : Type*} [Ring R] (P : Polynomial R)
variable (hP : degree P = 8)

-- Define the set A_P for some value c in R
def A_P (c : R) : Set R := { x | P.eval x = c }

-- Given that the number 8 is included in the set A_P
variable (c : R) (h8 : (8 : R) ∈ A_P c)

-- Prove that the minimum number of odd numbers in the set A_P is 1
theorem min_odd_in_A_P : ∃ x ∈ A_P c, (x % 2 = 1) := sorry

end min_odd_in_A_P_l255_255382


namespace connie_marbles_l255_255154

theorem connie_marbles (juan_marbles : ℕ) (h1 : juan_marbles = 64) (h2 : ∃ (connie_marbles : ℕ), juan_marbles = connie_marbles + 25) : ∃ (connie_marbles : ℕ), connie_marbles = 39 :=
by
  obtain ⟨connie_marbles, h⟩ := h2
  rw [← h1, h]
  exact ⟨39, by norm_num⟩

end connie_marbles_l255_255154


namespace congruent_triangles_of_equal_area_and_perimeter_l255_255443

-- Define the structure of a triangle
structure Triangle (α : Type) [LinearOrderedField α] :=
  (a b c : α) -- sides of the triangle
  (area : α)
  (perimeter : α)
  (a_nonneg : 0 ≤ a)
  (b_nonneg : 0 ≤ b)
  (c_nonneg : 0 ≤ c)
  
-- Assuming Heron's formula for the area of a triangle
def heron_formula {α : Type} [LinearOrderedField α] (a b c : α) (s : α) : α :=
  (s * (s - a) * (s - b) * (s - c)).sqrt

-- The statement of the problem
theorem congruent_triangles_of_equal_area_and_perimeter
  {α : Type} [LinearOrderedField α]
  (T1 T2 : Triangle α)
  (h_area : T1.area = T2.area)
  (h_perimeter : T1.perimeter = T2.perimeter)
  (h_side : T1.a = T2.a) :
  T1 = T2 :=
sorry

end congruent_triangles_of_equal_area_and_perimeter_l255_255443


namespace center_of_curvature_limit_l255_255000

variable {ε : ℝ} {f : ℝ → ℝ}

theorem center_of_curvature_limit (h_deriv_0 : deriv f 0 = 0) :
  (λ ε, 0, f ε + ε / (deriv f ε)) → (0, f 0 + 1 / (second_deriv f 0)) as ε → 0 :=
sorry

end center_of_curvature_limit_l255_255000


namespace alcohol_to_water_ratio_l255_255053

theorem alcohol_to_water_ratio (p q r : ℝ) :
  let alcohol := (p / (p + 1) + q / (q + 1) + r / (r + 1))
  let water := (1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1))
  (alcohol / water) = (p * q * r + p * q + p * r + q * r + p + q + r) / (p * q + p * r + q * r + p + q + r + 1) :=
sorry

end alcohol_to_water_ratio_l255_255053


namespace fraction_identity_l255_255199

theorem fraction_identity (a b : ℝ) (h1 : 1/a + 2/b = 1) (h2 : a ≠ -b) : 
  (ab - a)/(a + b) = 1 := 
by 
  sorry

end fraction_identity_l255_255199


namespace basis_is_setB_l255_255977

noncomputable def setA : list (ℝ × ℝ) := [(0, 0), (1, -2)]
noncomputable def setB : list (ℝ × ℝ) := [(-1, 2), (5, 7)]
noncomputable def setC : list (ℝ × ℝ) := [(3, 5), (6, 10)]
noncomputable def setD : list (ℝ × ℝ) := [(2, -3), (1/2, -3/4)]

def are_collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * u.1, k * u.2)

def is_basis (vecs : list (ℝ × ℝ)) : Prop :=
  vecs.length = 2 ∧ ¬are_collinear vecs.head vecs.tail.head

theorem basis_is_setB :
  is_basis setB :=
  by
  sorry

end basis_is_setB_l255_255977


namespace tan_power_difference_l255_255276

noncomputable def cot (x : ℝ) : ℝ := 1 / tan x

theorem tan_power_difference (x : ℝ) (h : cos x * (1 / tan x) = sin x^2) :
  tan x^6 - tan x^2 = 1 :=
by
  sorry

end tan_power_difference_l255_255276


namespace sin_cos_solutions_l255_255653

theorem sin_cos_solutions :
  {x | x ∈ Icc (-5 * π / 4) (3 * π / 4) ∧ (sin x)^2018 + (cos x)^(-2019) ≤ (cos x)^2018 + (sin x)^(-2019)} =
  {x | x ∈ Ico (-5 * π / 4) (-π)} ∪
  {x | x ∈ Ico (-3 * π / 4) (-π / 2)} ∪
  {x | x ∈ Ioc 0 (π / 4)} ∪
  {x | x ∈ Ioc (π / 2) (3 * π / 4)} :=
sorry

end sin_cos_solutions_l255_255653


namespace evaluate_expression_l255_255009

theorem evaluate_expression (x : ℚ) (hx : x = 7/6) :
  (x - 1) / x / (x - (2 * x - 1) / x) = 6 := 
by {
  rw hx,
  -- simplifying will be handled here
  sorry
}

end evaluate_expression_l255_255009


namespace cases_in_1990_l255_255750

theorem cases_in_1990 (cases_1980 cases_2000 : ℕ) (decrease_linear : ∀ (t : ℕ), 1980 ≤ t ∧ t ≤ 2000 →
  cases_1980 + (t - 1980) * (cases_2000 - cases_1980) / (2000 - 1980)
) :
  let cases_1990 := cases_1980 + (1990 - 1980) * (cases_2000 - cases_1980) / (2000 - 1980)
  in cases_1990 = 300500 :=
begin
  -- specify the values
  let cases_1980 := 600000,
  let cases_2000 := 1000,
  have : cases_1980 + (1990 - 1980) * (cases_2000 - cases_1980) / (2000 - 1980) = 300500 := sorry,
  exact this
end

end cases_in_1990_l255_255750


namespace solve_sin_exp_ineq_l255_255654

noncomputable def sin_exp_ineq (x : ℝ) : Prop :=
  (sin x) ^ 2018 + (cos x) ^ -2019 ≤ (cos x) ^ 2018 + (sin x) ^ -2019

theorem solve_sin_exp_ineq :
  ∀ x ∈ Set.Icc (- (5 * Real.pi) / 4) ((3 * Real.pi) / 4),
  sin_exp_ineq x ↔
    x ∈ (Set.Ico (- (5 * Real.pi) / 4) (- Real.pi))
      ∪ (Set.Ico (- (3 * Real.pi) / 4) (- (Real.pi / 2)))
      ∪ (Set.Ioc 0 (Real.pi / 4))
      ∪ (Set.Ioc (Real.pi / 2) (3 * Real.pi / 4)) :=
  by
  sorry

end solve_sin_exp_ineq_l255_255654


namespace percentage_increase_in_length_answer_l255_255987

noncomputable def percentage_increase_in_length (L B : ℝ) (A A' : ℝ) (B' : ℝ) :=
  B' = B * 0.75 ∧ A' = A * 1.05 ∧ A = L * B → (A' = L * B' * (1 + 0.4))

theorem percentage_increase_in_length_answer (L B : ℝ) (A A' : ℝ) (B' : ℝ) :
  percentage_increase_in_length L B A A' B' :=
begin
  intros h,
  cases h with h1 h2,
  cases h2 with h2 h3,
  rw h1 at h2,
  rw h2 at h3,
  sorry
end

end percentage_increase_in_length_answer_l255_255987


namespace percentage_disliked_by_both_l255_255827

theorem percentage_disliked_by_both (total_comics liked_by_females liked_by_males disliked_by_both : ℕ) 
  (total_comics_eq : total_comics = 300)
  (liked_by_females_eq : liked_by_females = 30 * total_comics / 100)
  (liked_by_males_eq : liked_by_males = 120)
  (disliked_by_both_eq : disliked_by_both = total_comics - (liked_by_females + liked_by_males)) :
  (disliked_by_both * 100 / total_comics) = 30 := by
  sorry

end percentage_disliked_by_both_l255_255827


namespace find_principal_l255_255918

theorem find_principal (r t1 t2 ΔI : ℝ) (h_r : r = 0.15) (h_t1 : t1 = 3.5) (h_t2 : t2 = 5) (h_ΔI : ΔI = 144) :
  ∃ P : ℝ, P = 640 :=
by
  sorry

end find_principal_l255_255918


namespace sample_size_l255_255100

theorem sample_size 
  (n_A n_B n_C : ℕ)
  (h1 : n_A = 15)
  (h2 : 3 * n_B = 4 * n_A)
  (h3 : 3 * n_C = 7 * n_A) :
  n_A + n_B + n_C = 70 :=
by
sorry

end sample_size_l255_255100


namespace ruler_is_perpendicular_l255_255880

theorem ruler_is_perpendicular
  (ruler_placed : ℝ → Prop)
  (parallel_line_exists : ∀ θ : ℝ, ∃ l : ℝ, parallel l θ)
  (ruler_orientation : ℝ):
  perpendicular ruler_orientation ground_lines := sorry

end ruler_is_perpendicular_l255_255880


namespace find_circumcircle_radius_l255_255417

noncomputable def circumcircle_radius (A B C M N : Point) (r : ℝ) : Prop :=
let dist := Euclidean.dist in
M.on_bisector_of_angle ∠ BAC ∧
N.extends_line A B ∧
dist A C = 1 ∧
dist A M = 1 ∧
∠ANM = ∠CNM ∧
circumradius_triangle C N M = r

theorem find_circumcircle_radius (A B C M N : Point) :
  circumcircle_radius A B C M N 1 :=
sorry

end find_circumcircle_radius_l255_255417


namespace sum_of_reciprocals_l255_255583

def parabola (p : ℝ) : Set (ℝ × ℝ) := { point | ∃ x y : ℝ, point = (x, y) ∧ y^2 = 2 * p * x }

def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

def directrix (p : ℝ) : Set (ℝ × ℝ) := { line | line = (x, -p / 2) for all x : ℝ }

def distance_to_directrix (p : ℝ) (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P in 
  abs (y + (p / 2))

def angle_between_points (i j : ℕ) (n : ℕ) : ℝ :=
  2 * real.pi * (i - 1) / (2^n) - 2 * real.pi * (j - 1) / (2^n)

theorem sum_of_reciprocals (p : ℝ) (n : ℕ)
  (P : Fin (2^n) → ℝ × ℝ)
  (hP : ∀ i, P i ∈ parabola p)
  (hF : ∀ i j, angle_between_points i j n = 2 * real.pi / (2^n)) :
  (∑ i in Finset.range (2^n), 1 / distance_to_directrix p (P i)) = 2^n / p := by
  sorry

end sum_of_reciprocals_l255_255583


namespace vector_subtraction_correct_l255_255678

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (-4, 2)

theorem vector_subtraction_correct :
  vector_a - 2 • vector_b = (10, -5) :=
sorry

end vector_subtraction_correct_l255_255678


namespace min_odd_numbers_in_A_P_l255_255359

theorem min_odd_numbers_in_A_P (P : ℝ → ℝ) 
  (hP : ∃ a8 a7 a6 a5 a4 a3 a2 a1 a0 : ℝ, P = λ x, a8 * x^8 + a7 * x^7 + a6 * x^6 + a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0)
  (h8_in_A_P : ∃ x : ℝ, P x = 8) : 
  (∀ x : ℝ, P x ∈ Set.A_P) → 
  (∃! y : ℝ, odd y ∧ y ∈ Set.A_P) := 
sorry

end min_odd_numbers_in_A_P_l255_255359


namespace option_B_option_C_option_D_l255_255288

-- Given a sequence that is half-increasing difference
def half_increasing_difference (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, (a n - (1/2) * a (n-1)) < (a (n+1) - (1/2) * a n)

-- Option B
theorem option_B (q : ℝ) (h : q > 1) : half_increasing_difference (λ n, q^n) :=
sorry

-- Option C
theorem option_C (a d : ℝ) (h : d > 0) : half_increasing_difference (λ n, a + (n-1) * d) :=
sorry

-- Option D
theorem option_D {a : ℕ → ℝ} (h : half_increasing_difference a) (S : ℕ → ℝ) (t : ℝ) (hS : ∀ n, S n = 2 * a n - 2^(n + 1) - t) :
  t ∈ Ioi (-32/3) :=
sorry

end option_B_option_C_option_D_l255_255288


namespace repeating_decimal_to_fraction_l255_255637

theorem repeating_decimal_to_fraction : (0.3666666 : ℚ) = 11 / 30 :=
by sorry

end repeating_decimal_to_fraction_l255_255637


namespace correct_proposition_l255_255698

variables (a x : ℝ) (m n p q : ℕ)

-- Define proposition p
def prop_p : Prop := a > 1 → a ^ x > Real.log a x

-- Define proposition q in the context of an arithmetic sequence
def arithmetic_seq (a_n : ℕ → ℝ) := ∀ {m n p q : ℕ}, m + n = p + q → a_n m + a_n n = a_n p + a_n q

-- Define proposition q
def prop_q : Prop := ∀ a_n : ℕ → ℝ, arithmetic_seq a_n

-- State to prove that \( \neg p \lor \neg q \) is the true proposition
theorem correct_proposition : ¬ prop_p ∨ ¬ prop_q :=
sorry

end correct_proposition_l255_255698


namespace sum_of_edges_rectangular_solid_l255_255480

theorem sum_of_edges_rectangular_solid
  (a r : ℝ)
  (hr : r ≠ 0)
  (volume_eq : (a / r) * a * (a * r) = 512)
  (surface_area_eq : 2 * ((a ^ 2) / r + a ^ 2 + (a ^ 2) * r) = 384)
  (geo_progression : true) : -- This is implicitly understood in the construction
  4 * ((a / r) + a + (a * r)) = 112 :=
by
  -- The proof will be placed here
  sorry

end sum_of_edges_rectangular_solid_l255_255480


namespace hyperbola_max_eccentricity_l255_255717

theorem hyperbola_max_eccentricity (a b e c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3: ∀ P : ℝ × ℝ, let x := P.1, let y := P.2 in (x^2 / a^2 - y^2 / b^2 = 1) → 
    (|PF₁ P| = 4 * |PF₂ P|)) 
  (h4: c = a * e) :
  e ≤ 5 / 3 := 
sorry

def PF₁ (P : ℝ × ℝ) : ℝ := -- Definition of distance PF₁
sorry

def PF₂ (P : ℝ × ℝ) : ℝ := -- Definition of distance PF₂
sorry

end hyperbola_max_eccentricity_l255_255717


namespace sum_a_n_eq_847_l255_255191

noncomputable def a_n (n : ℕ) : ℕ :=
if n % 30 = 0 then 15
else if n % 60 = 0 then 10
else if n % 60 = 0 then 12
else 0

theorem sum_a_n_eq_847 :
  ∑ n in finset.range 1000, a_n (n + 1) = 847 :=
by
  sorry

end sum_a_n_eq_847_l255_255191


namespace locus_of_N_l255_255334

theorem locus_of_N (M : ℝ × ℝ) (N : ℝ × ℝ) (A : ℝ × ℝ) : 
  (A = (3, 0)) 
  ∧
  (M.1 ^ 2 + M.2 ^ 2 = 1)
  ∧
  (is_equilateral_triangle (M, N, A))
  →
  ((N.1 - 3 / 2) ^ 2 + (N.2 + (3 * Real.sqrt 3) / 2) ^ 2 = 1) :=
sorry


end locus_of_N_l255_255334


namespace perimeter_of_region_l255_255849

theorem perimeter_of_region (A : ℝ) (n : ℕ) (h1 : A = 588) (h2 : n = 14) :
  let s := Real.sqrt (A / n) in
  let perimeter := (n + 1) * s in
  perimeter = 15 * Real.sqrt 42 := 
by
  sorry

end perimeter_of_region_l255_255849


namespace scott_earnings_l255_255433

theorem scott_earnings
  (price_smoothie : ℝ)
  (price_cake : ℝ)
  (cups_sold : ℝ)
  (cakes_sold : ℝ)
  (earnings_smoothies : ℝ := cups_sold * price_smoothie)
  (earnings_cakes : ℝ := cakes_sold * price_cake) :
  price_smoothie = 3 → price_cake = 2 → cups_sold = 40 → cakes_sold = 18 → 
  (earnings_smoothies + earnings_cakes) = 156 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end scott_earnings_l255_255433


namespace find_x_value_divisible_by_18_l255_255186

open Nat

theorem find_x_value_divisible_by_18 :
  ∃ (x : ℕ), x < 10 ∧ (2 * x + 7) % 9 = 0 ∧ ((x % 2 = 0) ∧ (x = 6)) :=
begin
  sorry
end

end find_x_value_divisible_by_18_l255_255186


namespace geometric_mean_l255_255745

theorem geometric_mean (a b c : ℝ) (h1 : a = 5 + 2 * Real.sqrt 6) (h2 : c = 5 - 2 * Real.sqrt 6) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) (h6 : b^2 = a * c) : b = 1 :=
sorry

end geometric_mean_l255_255745


namespace assign_teachers_to_classes_l255_255879

theorem assign_teachers_to_classes :
  let num_classes := 6
  let num_teachers := 3
  let classes_per_teacher := 2
  (num_classes.choose classes_per_teacher) * ((num_classes - classes_per_teacher).choose classes_per_teacher) * ((num_classes - 2 * classes_per_teacher).choose classes_per_teacher) / num_teachers.factorial = 15 :=
by
  let num_classes := 6
  let num_teachers := 3
  let classes_per_teacher := 2
  have h1 : (num_classes.choose classes_per_teacher) = 15 := by sorry
  have h2 : ((num_classes - classes_per_teacher).choose classes_per_teacher) = 6 := by sorry
  have h3 : ((num_classes - 2 * classes_per_teacher).choose classes_per_teacher) = 1 := by sorry
  calc
    (num_classes.choose classes_per_teacher) * ((num_classes - classes_per_teacher).choose classes_per_teacher) * ((num_classes - 2 * classes_per_teacher).choose classes_per_teacher) / num_teachers.factorial
        = 15 * 6 * 1 / 6 : by rw [h1, h2, h3]
    ... = 15 : by norm_num

end assign_teachers_to_classes_l255_255879


namespace triangle_ratio_l255_255058

theorem triangle_ratio (XYZ : Triangle)
  (right_angle_Z : is_right_angle XYZ.point_Z)
  (angle_XYZ_lt_45 : XYZ.angle_XYZ < 45)
  (XY_len : XYZ.XY = 5)
  (point_Q_on_XY : Q.on_line_segment XYZ.XY)
  (angle_relation : XYZ.angle_YQZ = 2 * XYZ.angle_QZY)
  (QZ_len : QZ = 2) :
  let XQ := XYZ.XQ
  let QY := XYZ.QY
  let ratio := XQ / QY
  let expr := 19 + 5 * sqrt 13 / 6
  let s := 19
  let t := 5
  let u := 13
  in ratio = expr ∧ (s + t + u) = 37 := 
begin
  sorry
end

end triangle_ratio_l255_255058


namespace quadratic_discriminant_relation_l255_255737

theorem quadratic_discriminant_relation
  (a b c x₀ : ℝ)
  (h : a ≠ 0)
  (hx₀ : a * x₀^2 + b * x₀ + c = 0) :
  let Δ := b^2 - 4 * a * c,
      M := (2 * a * x₀ + b)^2
  in Δ = M :=
by
  sorry

end quadratic_discriminant_relation_l255_255737


namespace find_b_find_area_of_ABC_l255_255749

variable {a b c : ℝ}
variable {B : ℝ}

-- Given Conditions
def given_conditions (a b c B : ℝ) := a = 4 ∧ c = 3 ∧ B = Real.arccos (1 / 8)

-- Proving b = sqrt(22)
theorem find_b (h : given_conditions a b c B) : b = Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) :=
by
  sorry

-- Proving the area of triangle ABC
theorem find_area_of_ABC (h : given_conditions a b c B) 
  (sinB : Real.sin B = 3 * Real.sqrt 7 / 8) : 
  (1 / 2) * a * c * Real.sin B = 9 * Real.sqrt 7 / 4 :=
by
  sorry

end find_b_find_area_of_ABC_l255_255749


namespace triangle_third_side_l255_255747

noncomputable def length_of_third_side
  (a b : ℝ) (θ : ℝ) (cosθ : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 - 2 * a * b * cosθ)

theorem triangle_third_side : 
  length_of_third_side 8 15 (Real.pi / 6) (Real.cos (Real.pi / 6)) = Real.sqrt (289 - 120 * Real.sqrt 3) :=
by
  sorry

end triangle_third_side_l255_255747


namespace double_prime_dates_in_2007_l255_255994

noncomputable def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m ∈ {2, ..., n // 1}, n % m ≠ 0

noncomputable def is_double_prime_date (day month : ℕ) : Prop :=
  is_prime day ∧ is_prime month ∧ is_prime (day + month)

noncomputable def count_double_prime_dates (year : ℕ) : ℕ :=
  if year == 2007 then
    let prime_days := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    let prime_months := [2, 3, 5, 7, 11]
    prime_months.sum (λ m, (prime_days.filter (λ d, is_prime (d + m))).length)
  else 0

theorem double_prime_dates_in_2007 : count_double_prime_dates 2007 = 7 :=
by
  sorry

end double_prime_dates_in_2007_l255_255994


namespace circle_with_diameter_MN_intersects_x_axis_at_fixed_points_l255_255582

noncomputable
def circle_intersects_x_axis (t p : ℝ) (ht : t > 0) (hp : p > 0) : Prop :=
  let A := (A(t, 0))
  let B := (B(x1, y1))
  let C := (C(x2, y2))
  let M := M(-t, -t * y1 / x1)
  let N := N(-t, -t * y2 / x2)
  let k := slope_line_through_A
  have x1x2_eq_t2 : x1 * x2 = t^2, from sorry,
  have M_N_diameter_circle_eq : diameter_circle_eq_MN, from sorry,
  x = -t - sqrt(2 * p * t) ∨ x = -t + sqrt(2 * p * t)

theorem circle_with_diameter_MN_intersects_x_axis_at_fixed_points (t p : ℝ) (ht : t > 0) (hp : p > 0) : circle_intersects_x_axis t p ht hp :=
  by
    sorry

end circle_with_diameter_MN_intersects_x_axis_at_fixed_points_l255_255582


namespace platelet_diameter_scientific_notation_l255_255474

theorem platelet_diameter_scientific_notation :
  (5,000,000 : ℝ) * diameter = 1 → diameter = 2 * 10^(-7) :=
by
  intro h
  have h' : diameter = 1 / (5,000,000 : ℝ) := by sorry
  rw h'
  norm_num
  reflexivity

end platelet_diameter_scientific_notation_l255_255474


namespace convert_to_rectangular_form_l255_255618

theorem convert_to_rectangular_form :
  2 * real.sqrt 3 * complex.exp (complex.I * 17 * real.pi / 6) = 
  -3 + complex.I * real.sqrt 3 := by
sorry

end convert_to_rectangular_form_l255_255618


namespace determine_polynomial_l255_255174

theorem determine_polynomial (p : ℝ → ℝ → ℝ) :
  (∀ x y u v : ℝ, p(x, y) * p(u, v) = p(x * u + y * v, x * v + y * u)) →
  (∃ m n : ℕ, ∀ x y : ℝ, p(x, y) = (x + y)^m * (x - y)^n) ∨ (∀ x y : ℝ, p(x, y) = 0) :=
by
  sorry

end determine_polynomial_l255_255174


namespace find_difference_l255_255891

-- Assume S_1 and S_2 are unit squares with specific constraints
def S1 : set (ℝ × ℝ) := {(x, y) | 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1}
def S2 (a b : ℝ) : set (ℝ × ℝ) := {(x, y) | a ≤ x ∧ x ≤ a + 1 ∧ b ≤ y ∧ y ≤ b + 1}

-- Define the minimum distance function
def minDist (p1 p2 : (ℝ × ℝ)) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the maximum distance function
def maxDist (p1 p2 : (ℝ × ℝ)) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Condition x = 5
def x : ℝ := 5

-- Prove the mathematically equivalent problem
theorem find_difference (a b : ℝ) (h : S2 a b ⊆ S2 6 1) :
  let y_min := minDist (0, 0) (6, 1)
  let y_max := maxDist (0, 0) (6, 1)
  ∃ a b c : ℤ, 100 * a + 10 * b + c = 472 := 
by 
  sorry

end find_difference_l255_255891


namespace minValue_l255_255809

noncomputable def minValueOfExpression (a b c : ℝ) : ℝ :=
  (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a))

theorem minValue (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 2 * a + 2 * b + 2 * c = 3) : 
  minValueOfExpression a b c = 2 :=
  sorry

end minValue_l255_255809


namespace range_of_k_for_monotonic_decreasing_l255_255464

theorem range_of_k_for_monotonic_decreasing (k : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → f(x1) > f(x2)) → k < (1:ℝ) / (2:ℝ) :=
by
  let f := λ x : ℝ, (2 * k - 1) * x + 1
  sorry

end range_of_k_for_monotonic_decreasing_l255_255464


namespace point_T_on_line_AC_l255_255807

variables {A B C O T : Type}
variables [Triangle ABC] [CenterOfCircumcircle O ABC] [IntersectionOfCircumcircleAndAngleBisector T OBC (AngleBisector AOB)]
variables (Angle_ABC_gt_90 : IsObtuse (∆ABC))

theorem point_T_on_line_AC (h : IsCollinear T A C) : T ∈ Line(A, C) :=
sorry

end point_T_on_line_AC_l255_255807


namespace william_total_tickets_l255_255517

def initial_tickets : ℕ := 15
def additional_tickets : ℕ := 3
def total_tickets : ℕ := initial_tickets + additional_tickets

theorem william_total_tickets :
  total_tickets = 18 := by
  -- proof goes here
  sorry

end william_total_tickets_l255_255517


namespace discount_percentage_l255_255004

noncomputable def labelledPrice : ℝ := 16000
noncomputable def purchasePrice : ℝ := 12500
noncomputable def requiredProfitPrice : ℝ := 19200

theorem discount_percentage (D : ℝ) (P : ℝ) 
  (h1 : P * (1 - D/100) = purchasePrice)
  (h2 : P * 1.20 = requiredProfitPrice):
  D = 21.875 :=
by
  have eqP : P = labelledPrice := by
    calc
      P = requiredProfitPrice / 1.20 := by
        field_simp
        sorry -- calculation steps here if needed
      
  rw [← eqP] at h1
  have hP : (1 - D / 100) = purchasePrice / labelledPrice := by
    field_simp [eqP.symm, labelledPrice, purchasePrice]
    sorry -- calculation steps here if needed
  
  have calcD : D = 21.875 := by
    sorry -- calculation steps here if needed
  
  exact calcD

end discount_percentage_l255_255004


namespace expected_no_advice_formula_l255_255594

noncomputable def expected_no_advice (n : ℕ) (p : ℝ) : ℝ :=
  ∑ j in Finset.range n, (1 - p) ^ j

theorem expected_no_advice_formula (n : ℕ) (p : ℝ) (hp : 0 ≤ p ∧ p < 1) : 
  expected_no_advice n p = (1 - (1 - p) ^ n) / p :=
by
  sorry

end expected_no_advice_formula_l255_255594


namespace minimum_odd_numbers_in_A_P_l255_255372

-- Polynomials and assumptions
def degree (P : Polynomial ℝ) : ℕ := P.natDegree

-- The set A_P is defined as the set of all numbers x for which P(x) gives a certain value
def A_P (P : Polynomial ℝ) : Set ℝ := {x : ℝ | P.eval x = P.eval 8}

-- Define the main theorem statement
theorem minimum_odd_numbers_in_A_P (P : Polynomial ℝ) (hdeg : degree P = 8) (h8 : 8 ∈ A_P P) : 
  ∃ n, n = 1 ∧ ∃ (x : ℝ), x ∈ A_P P ∧ ¬ (x % 2 = 0) := sorry

end minimum_odd_numbers_in_A_P_l255_255372


namespace toy_cross_section_area_l255_255551

noncomputable def height_of_tetrahedron (a : ℝ) : ℝ :=
  (a * real.sqrt 6) / 3

noncomputable def radius_of_sphere (a : ℝ) : ℝ :=
  a / real.sqrt 2

noncomputable def distance_to_cross_section (r : ℝ) : ℝ :=
  r / 3

noncomputable def cross_section_radius (r d : ℝ) : ℝ :=
  real.sqrt (r^2 - d^2)

noncomputable def cross_section_area (r : ℝ) : ℝ :=
  real.pi * r^2

theorem toy_cross_section_area : 
  let edge_length := 4
      height := height_of_tetrahedron edge_length
      radius_sphere := radius_of_sphere edge_length
      distance_midpoints := distance_to_cross_section radius_sphere
      radius_cross_section := cross_section_radius radius_sphere distance_midpoints
  in cross_section_area radius_cross_section = (16 * real.pi) / 3 :=
sorry

end toy_cross_section_area_l255_255551


namespace trig_identity_simplification_l255_255440

theorem trig_identity_simplification (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (2 * (Real.cos (α / 2))^2) = 2 * Real.sin α :=
by sorry

end trig_identity_simplification_l255_255440


namespace range_of_b_l255_255720

theorem range_of_b (a : ℝ) (b : ℝ) (A : Set ℝ) (B : Set ℝ) 
  (h1 : A = {x : ℝ | log (1/2 : ℝ) (x + 2) < 0})
  (h2 : B = {x : ℝ | (x + 3) * (x - b) < 0})
  (h3 : a = -3)
  (h4 : A ∩ B ≠ ∅) : b > -1 :=
sorry

end range_of_b_l255_255720


namespace total_dogs_at_center_l255_255602

structure PawsitiveTrainingCenter :=
  (sit : Nat)
  (stay : Nat)
  (fetch : Nat)
  (roll_over : Nat)
  (sit_stay : Nat)
  (sit_fetch : Nat)
  (sit_roll_over : Nat)
  (stay_fetch : Nat)
  (stay_roll_over : Nat)
  (fetch_roll_over : Nat)
  (sit_stay_fetch : Nat)
  (sit_stay_roll_over : Nat)
  (sit_fetch_roll_over : Nat)
  (stay_fetch_roll_over : Nat)
  (all_four : Nat)
  (none : Nat)

def PawsitiveTrainingCenter.total_dogs (p : PawsitiveTrainingCenter) : Nat :=
  p.sit + p.stay + p.fetch + p.roll_over
  - p.sit_stay - p.sit_fetch - p.sit_roll_over - p.stay_fetch - p.stay_roll_over - p.fetch_roll_over
  + p.sit_stay_fetch + p.sit_stay_roll_over + p.sit_fetch_roll_over + p.stay_fetch_roll_over
  - p.all_four + p.none

theorem total_dogs_at_center (p : PawsitiveTrainingCenter) (h : 
  p.sit = 60 ∧
  p.stay = 35 ∧
  p.fetch = 45 ∧
  p.roll_over = 40 ∧
  p.sit_stay = 20 ∧
  p.sit_fetch = 15 ∧
  p.sit_roll_over = 10 ∧
  p.stay_fetch = 5 ∧
  p.stay_roll_over = 8 ∧
  p.fetch_roll_over = 6 ∧
  p.sit_stay_fetch = 4 ∧
  p.sit_stay_roll_over = 3 ∧
  p.sit_fetch_roll_over = 2 ∧
  p.stay_fetch_roll_over = 1 ∧
  p.all_four = 2 ∧
  p.none = 12
) : PawsitiveTrainingCenter.total_dogs p = 135 := by
  sorry

end total_dogs_at_center_l255_255602


namespace total_birds_correct_l255_255967

-- Define the conditions
def number_of_trees : ℕ := 7
def blackbirds_per_tree : ℕ := 3
def magpies : ℕ := 13

-- Define the total number of blackbirds using the conditions
def total_blackbirds : ℕ := number_of_trees * blackbirds_per_tree

-- Define the total number of birds using the total number of blackbirds and the number of magpies
def total_birds : ℕ := total_blackbirds + magpies

-- The theorem statement that should be proven
theorem total_birds_correct : total_birds = 34 := 
sorry

end total_birds_correct_l255_255967


namespace max_points_guaranteed_l255_255497

/-- Two players are playing a game. Player1 has 1000 even-numbered cards (2,4,...,2000) and Player2 has 1001 odd-numbered cards (1,3,...,2001).
They take turns starting with Player1. In each turn, the player whose turn it is plays one of their cards, and the other player, after looking
at it, plays one of their cards; the player with the higher number on their card scores a point, and both cards are discarded. There are a total of 
1000 turns (one card of the second player is not used). Prove the maximum number of points each player can guarantee themselves. 
-/
theorem max_points_guaranteed :
  let player1_points := 499,
      player2_points := 501 in
  player1_points = 499 ∧ player2_points = 501 :=
begin
  sorry
end

end max_points_guaranteed_l255_255497


namespace repeating_decimal_to_fraction_l255_255634

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + 0.066666... ) : x = 11 / 30 :=
sorry

end repeating_decimal_to_fraction_l255_255634


namespace systemic_sampling_seventh_group_number_l255_255938

open Locale classical

theorem systemic_sampling_seventh_group_number :
  ∀ (total_students : ℕ) (groups : ℕ) (group_size : ℕ) (group3_start : ℕ) (group3_number : ℕ) (group7_start : ℕ),
    total_students = 50 →
    groups = 10 →
    group_size = 5 →
    group3_start = 11 →
    group3_number = 13 →
    group7_start = 31 →
    (group3_number - group3_start) mod group_size = (group3_number - group3_start) % group_size →
    let offset := (group3_number - group3_start) % group_size
    in group7_start + offset = 33 :=
begin
  intros,
  sorry,
end

end systemic_sampling_seventh_group_number_l255_255938


namespace scott_earnings_l255_255431

theorem scott_earnings
  (price_smoothie : ℝ)
  (price_cake : ℝ)
  (cups_sold : ℝ)
  (cakes_sold : ℝ)
  (earnings_smoothies : ℝ := cups_sold * price_smoothie)
  (earnings_cakes : ℝ := cakes_sold * price_cake) :
  price_smoothie = 3 → price_cake = 2 → cups_sold = 40 → cakes_sold = 18 → 
  (earnings_smoothies + earnings_cakes) = 156 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end scott_earnings_l255_255431


namespace sin_cos_solutions_l255_255652

theorem sin_cos_solutions :
  {x | x ∈ Icc (-5 * π / 4) (3 * π / 4) ∧ (sin x)^2018 + (cos x)^(-2019) ≤ (cos x)^2018 + (sin x)^(-2019)} =
  {x | x ∈ Ico (-5 * π / 4) (-π)} ∪
  {x | x ∈ Ico (-3 * π / 4) (-π / 2)} ∪
  {x | x ∈ Ioc 0 (π / 4)} ∪
  {x | x ∈ Ioc (π / 2) (3 * π / 4)} :=
sorry

end sin_cos_solutions_l255_255652


namespace sqrt2_plus_sqrt3_irrational_by_contradiction_l255_255513

theorem sqrt2_plus_sqrt3_irrational_by_contradiction :
  (∀ x y : ℝ, x = Real.sqrt 2 ∧ y = Real.sqrt 3 → irrational (x + y))
:= 
by
  intro x y
  assume h : x = Real.sqrt 2 ∧ y = Real.sqrt 3
  by_contradiction h_rational
  have h1 : rational (x + y) := h_rational
  -- proof steps would go here
  sorry

end sqrt2_plus_sqrt3_irrational_by_contradiction_l255_255513


namespace BC_at_least_17_l255_255059

-- Given conditions
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
-- Distances given
variables (AB AC EC BD BC : ℝ)
variables (AB_pos : AB = 7)
variables (AC_pos : AC = 15)
variables (EC_pos : EC = 9)
variables (BD_pos : BD = 26)
-- Triangle Inequalities
variables (triangle_ABC : ∀ {x y z : Type} [MetricSpace x] [MetricSpace y] [MetricSpace z], AC - AB < BC)
variables (triangle_DEC : ∀ {x y z : Type} [MetricSpace x] [MetricSpace y] [MetricSpace z], BD - EC < BC)

-- Proof statement
theorem BC_at_least_17 : BC ≥ 17 := by
  sorry

end BC_at_least_17_l255_255059


namespace friends_count_l255_255438

-- Define the conditions
def num_kids : ℕ := 2
def shonda_present : Prop := True  -- Shonda is present, we may just incorporate it as part of count for clarity
def num_adults : ℕ := 7
def num_baskets : ℕ := 15
def eggs_per_basket : ℕ := 12
def eggs_per_person : ℕ := 9

-- Define the total number of eggs
def total_eggs : ℕ := num_baskets * eggs_per_basket

-- Define the total number of people
def total_people : ℕ := total_eggs / eggs_per_person

-- Define the number of known people (Shonda, her kids, and the other adults)
def known_people : ℕ := num_kids + 1 + num_adults  -- 1 represents Shonda

-- Define the number of friends
def num_friends : ℕ := total_people - known_people

-- The theorem we need to prove
theorem friends_count : num_friends = 10 :=
by
  sorry

end friends_count_l255_255438


namespace average_speed_v2_l255_255492

theorem average_speed_v2 (v1 : ℝ) (t : ℝ) (S1 : ℝ) (S2 : ℝ) : 
  (v1 = 30) → (t = 30) → (S1 = 800) → (S2 = 200) → 
  (v2 = (v1 - (S1 - S2) / t) ∨ v2 = (v1 + (S1 - S2) / t)) :=
by
  intros h1 h2 h3 h4
  sorry

end average_speed_v2_l255_255492


namespace smallest_positive_period_and_monotonically_increasing_interval_l255_255253

noncomputable def f (x : ℝ) : ℝ :=
  let a := (2 * Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
  let b := (Real.cos x, 1)
  a.1 * b.1 + a.2 * b.2
  
theorem smallest_positive_period_and_monotonically_increasing_interval :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∀ k : ℤ, ∀ x, -π / 3 + k * π ≤ x ∧ x ≤ π / 6 + k * π → (f' x ≥ 0 ∧ f' x ≤ 0)) := sorry

variables {A : ℝ} (a b c : ℝ)

lemma area_of_triangle_ABC
  (h1 : f A = 2)
  (h2 : a = Real.sqrt 7)
  (h3 : Real.sin B = 2 * Real.sin C) :
  0 < A ∧ A < π ∧ A = π / 3 ∧ ∃ S, S = 7 * Real.sqrt 3 / 6 :=
sorry

end smallest_positive_period_and_monotonically_increasing_interval_l255_255253


namespace percentage_disliked_by_both_l255_255826

theorem percentage_disliked_by_both (total_comics liked_by_females liked_by_males disliked_by_both : ℕ) 
  (total_comics_eq : total_comics = 300)
  (liked_by_females_eq : liked_by_females = 30 * total_comics / 100)
  (liked_by_males_eq : liked_by_males = 120)
  (disliked_by_both_eq : disliked_by_both = total_comics - (liked_by_females + liked_by_males)) :
  (disliked_by_both * 100 / total_comics) = 30 := by
  sorry

end percentage_disliked_by_both_l255_255826


namespace cosine_quartic_representation_l255_255622

theorem cosine_quartic_representation :
  ∃ (a b c : ℝ), (∀ θ : ℝ, cos θ = cos (θ % (2 * π))) → 
    (∀ θ : ℝ, cos θ = cos ((-θ) % (2 * π))) → 
    a = 1/8 ∧ b = 1/2 ∧ c = 0 ∧
    (∀ θ : ℝ, cos (4*θ) = 8 * (cos θ)^4 - 8 * (cos θ)^2 + 1) ∧ 
    (∀ θ : ℝ, cos (2*θ) = 2 * (cos θ)^2 - 1) ∧
    (∀ θ : ℝ, (cos θ)^4 = a * cos (4*θ) + b * cos (2*θ) + c * cos θ) :=
by
  use 1/8, 1/2, 0
  split
  · refl  -- a = 1/8
  split
  · refl  -- b = 1/2
  split
  · refl  -- c = 0
  split
  · assume θ
    sorry  -- proof for cos(4θ) formula
  split
  · assume θ
    sorry  -- proof for cos(2θ) formula
  assume θ
  sorry  -- proof for the main goal

end cosine_quartic_representation_l255_255622


namespace combined_total_footprints_l255_255902

-- Definitions for footprints by Pogo
def footprints_per_meter_pogo := 4
def distance_mars := 6000

-- Definitions for footprints by Grimzi
def footprints_per_six_meters_grimzi := 3
def distance_pluto := 6000

-- Theorem for combined total number of footprints
theorem combined_total_footprints :
  (footprints_per_meter_pogo * distance_mars) +
  ((footprints_per_six_meters_grimzi / 6) * distance_pluto) = 27000 :=
by
  have footprints_pogo : ℕ := footprints_per_meter_pogo * distance_mars
  have footprints_grimzi : ℕ := (footprints_per_six_meters_grimzi / 6) * distance_pluto
  have total_footprints : ℕ := footprints_pogo + footprints_grimzi
  calc
    total_footprints = 24000 + 3000 : by sorry
    ... = 27000 : by sorry

end combined_total_footprints_l255_255902


namespace cos_alpha_minus_pi_over_4_l255_255223

theorem cos_alpha_minus_pi_over_4 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.tan α = 2) :
  Real.cos (α - π / 4) = 3 * Real.sqrt 10 / 10 :=
sorry

end cos_alpha_minus_pi_over_4_l255_255223


namespace rhombus_area_example_l255_255066

-- Definition of the problem
def rhombus_area (s : ℝ) (d_diff : ℝ) : ℝ :=
  ∃ a b : ℝ, a^2 + (b + d_diff)^2 = s^2 ∧
             a * (b + d_diff) = 236 + 6 * Real.sqrt 254

-- Statement to prove
theorem rhombus_area_example : rhombus_area (Real.sqrt 145) 12 :=
by {
  symmetry,
  sorry
}

end rhombus_area_example_l255_255066


namespace cost_price_computation_l255_255472

-- Define the conditions given in the problem
variables (C : ℝ) (S : ℝ) (Tax : ℝ)

-- Condition: Sale price including sales tax is Rs. 616
def sale_price_condition := S = 616

-- Condition: Rate of sales tax is 10%
def rate_of_sales_tax_condition := Tax = 0.1 * S

-- Condition: Shopkeeper made a profit of 13%
def profit_condition := S = 1.13 * C

-- Combined condition: The cost price of the article computes correctly
theorem cost_price_computation (h1 : sale_price_condition) (h2 : rate_of_sales_tax_condition) (h3 : profit_condition) :
  C = 616 / 1.243 :=
by
  -- Proof not required
  sorry

end cost_price_computation_l255_255472


namespace domain_of_composite_function_l255_255458

open Real

def domain_f : Set ℝ := { x : ℝ | x ≤ 1 }

def domain_log : Set ℝ := { y : ℝ | y > 0 }

def domain_inner : Set ℝ :=
  { x : ℝ | x < -1 } ∪ { x : ℝ | x > 1 } ∩ { x : ℝ | -sqrt 3 ≤ x ∧ x ≤ sqrt 3 }

theorem domain_of_composite_function :
  ( f ).domain → ( f (log 2 (x^2 - 1)) ) = [ - (sqrt 3), -1) ∪ (1, sqrt 3 ] :=
by
  sorry

end domain_of_composite_function_l255_255458


namespace min_value_of_M_l255_255845

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

noncomputable def M : ℝ :=
  (Real.rpow (a / (b + c)) (1 / 4)) + (Real.rpow (b / (c + a)) (1 / 4)) + (Real.rpow (c / (b + a)) (1 / 4)) +
  Real.sqrt ((b + c) / a) + Real.sqrt ((a + c) / b) + Real.sqrt ((a + b) / c)

theorem min_value_of_M : M a b c = 3 * Real.sqrt 2 + (3 * Real.rpow 8 (1 / 4)) / 2 := sorry

end min_value_of_M_l255_255845


namespace sum_lent_l255_255912

theorem sum_lent (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ) 
  (h1 : r = 0.06) (h2 : t = 8) (h3 : I = P - 520) : P * r * t = I → P = 1000 :=
by
  -- Given conditions
  intros
  -- Sorry placeholder
  sorry

end sum_lent_l255_255912


namespace option_b_correct_option_c_correct_option_d_correct_l255_255291

def half_increasing_difference (a : ℕ → ℝ) :=
∀ n : ℕ, n ≥ 2 → a n - (1 / 2) * a (n - 1) < a (n + 1) - (1 / 2) * a n

theorem option_b_correct (q : ℝ) (h : q > 1) :
  half_increasing_difference (λ n, q ^ n) :=
sorry

theorem option_c_correct (a : ℕ → ℝ) (d : ℝ) (h : d > 0) (h_arith : ∀ n : ℕ, a (n + 1) = a n + d) :
  half_increasing_difference a :=
sorry

theorem option_d_correct (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ)
  (h_half_increasing : half_increasing_difference a)
  (h_sum : ∀ n : ℕ, S n = 2 * a n - 2^(n + 1) - t) :
  t > -32 / 3 :=
sorry

end option_b_correct_option_c_correct_option_d_correct_l255_255291


namespace no_lighter_sentence_for_liar_l255_255881

theorem no_lighter_sentence_for_liar
  (total_eggs : ℕ)
  (stolen_eggs1 stolen_eggs2 stolen_eggs3 : ℕ)
  (different_stolen_eggs : stolen_eggs1 ≠ stolen_eggs2 ∧ stolen_eggs2 ≠ stolen_eggs3 ∧ stolen_eggs1 ≠ stolen_eggs3)
  (stolen_eggs1_max : stolen_eggs1 > stolen_eggs2 ∧ stolen_eggs1 > stolen_eggs3)
  (stole_7 : stolen_eggs1 = 7)
  (total_eq_20 : stolen_eggs1 + stolen_eggs2 + stolen_eggs3 = 20) :
  false :=
by
  sorry

end no_lighter_sentence_for_liar_l255_255881


namespace distance_from_point_to_line_l255_255455

def direction_vector : ℝ × ℝ × ℝ := (1, -1, 0)
def A : ℝ × ℝ × ℝ := (1, 1, 2)
def P : ℝ × ℝ × ℝ := (2, -2, 1)

theorem distance_from_point_to_line :
  let AP := (P.1 - A.1, P.2 - A.2, P.3 - A.3) in
  let m := direction_vector in
  let projection_len := ((AP.1 * m.1 + AP.2 * m.2 + AP.3 * m.3) / (m.1 ^ 2 + m.2 ^ 2 + m.3 ^ 2).sqrt) in
  let distance := ((AP.1 ^ 2 + AP.2 ^ 2 + AP.3 ^ 2 - projection_len ^ 2)).sqrt in
  distance = Real.sqrt 3 :=
sorry

end distance_from_point_to_line_l255_255455


namespace seat_to_right_proof_l255_255078

def Xiaofang_seat : ℕ × ℕ := (3, 5)

def seat_to_right (seat : ℕ × ℕ) : ℕ × ℕ :=
  (seat.1 + 1, seat.2)

theorem seat_to_right_proof : seat_to_right Xiaofang_seat = (4, 5) := by
  unfold Xiaofang_seat
  unfold seat_to_right
  sorry

end seat_to_right_proof_l255_255078


namespace extremum_condition_l255_255029

def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

-- f has an extremum of 10 at x = 1 which means f'(1) = 0 and f(1) = 10
theorem extremum_condition (a b : ℝ) : 
  (f' : ∀ x, deriv (f x a b) x) → 
  f' 1 = 0 → f 1 a b = 10 → a = -4 ∧ b = 11 :=
by
  -- Direct derivation of the necessary conditions given in the problem
  -- omitting intermediate steps and including the final answer as required by the problem statement
  -- logic to be filled in
  sorry

end extremum_condition_l255_255029


namespace main_theorem_l255_255786

noncomputable def prove_trig_inequality (α β γ : ℝ) (A B C : Triangle) 
  (hA : A.ang = α) (hB : B.ang = β) (hC : C.ang = γ) : Prop := 
  ∑ (cyc : list (Triangle)),
  (tan (cyc.head.ang / 2)) * (tan (cyc.last.ang / 2)) * (cot (cyc.tail.head.ang / 2)) ≥ sqrt 3

theorem main_theorem (A B C : Triangle) 
  (α β γ : ℝ) (hA : A.angle = α) (hB : B.angle = β) (hC : C.angle = γ) 
  (h_sum_angles : α + β + γ = π) :
  prove_trig_inequality α β γ A B C hA hB hC := 
by
  -- proof goes here
  sorry

end main_theorem_l255_255786


namespace function_condition_unique_function_l255_255992

noncomputable def f : ℕ+ → ℕ+ := sorry

theorem function_condition (a b : ℕ+) :
  ∃ x y z : ℕ+, x = a ∧ y = f b ∧ z = f (b + f a - 1) ∧ 
  (x + y > z ∧ y + z > x ∧ z + x > y) :=
sorry

theorem unique_function : (∀ f : ℕ+ → ℕ+, (∀ a b : ℕ+, 
    ∃ x y z : ℕ+, (x = a ∧ y = f b ∧ z = f (b + f a - 1) ∧ 
    (x + y > z ∧ y + z > x ∧ z + x > y)) →
    f = λ n, n)) :=
sorry

end function_condition_unique_function_l255_255992


namespace sum_coeffs_mod_2_min_sum_coeffs_l255_255789

noncomputable def u : ℝ := (Real.sqrt 17 - 1) / 2

-- Definition of the polynomial
structure Polynomial (α : Type _) :=
  (a : ℕ → α)
  (degree : ℕ)
  (coefficients_nonneg : ∀ (n : ℕ), 0 ≤ a n)
  (degree_positive : 0 < degree)

noncomputable def P : Polynomial ℝ :=
{ a := λ n, if n = 12 then 1 else if n = 11 then 2 else if n = 10 then 3 else if n = 9 then 5 else if n = 8 then 8 else if n = 7 then 13 else if n = 6 then 21 else if n = 5 then 31 else if n = 4 then 56 else if n = 3 then 70 else if n = 2 then 157 else if n = 1 then 126 else if n = 0 then 504 else 0,
  degree := 12,
  coefficients_nonneg := by intros; simp; apply Nat.zero_le,
  degree_positive := by decide }

-- Sum of the coefficients of the polynomial
noncomputable def sum_coeffs (P : Polynomial ℝ) : ℝ :=
  Finset.sum (Finset.range (P.degree + 1)) (λ i, P.a i)

-- 1. Proof that the sum of the coefficients is equivalent to 1 modulo 2
theorem sum_coeffs_mod_2 (P : Polynomial ℝ) (hP : ∑ i in Finset.range (P.degree + 1), P.a i = 1) :
  sum_coeffs P % 2 = 1 :=
sorry

-- 2. Proof that the minimum possible sum of the coefficients is 23
theorem min_sum_coeffs (P : Polynomial ℝ) (hP : ∑ i in Finset.range (P.degree + 1), P.a i = 2017) :
  ∃ Q : Polynomial ℝ, sum_coeffs Q = 23 :=
sorry

end sum_coeffs_mod_2_min_sum_coeffs_l255_255789


namespace regular_dodecagon_product_eq_531440_l255_255558

noncomputable def regularDodecagonProduct : ℂ := 
  let Q1 := (4 : ℝ, 0 : ℝ)
  let Q7 := (2 : ℝ, 0 : ℝ)
  let dodecagonRoots := finset.univ.image (λ k, (3:ℂ) * exp (2 * real.pi * complex.I * (k : ℂ) / 12))
  dodecagonRoots.prod id

theorem regular_dodecagon_product_eq_531440 :
  regularDodecagonProduct = 531440 := 
by
  sorry

end regular_dodecagon_product_eq_531440_l255_255558


namespace angle_DEA_half_difference_l255_255167

theorem angle_DEA_half_difference (A B C D E : Point) {O : Circle}
(h_midpoint: E.is_midpoint (arc O B C) ∧ E.opposite_side B C A)
(h_diameter: O.diameter D E):
  ∠ D E A = (∠ B - ∠ C) / 2 :=
sorry

end angle_DEA_half_difference_l255_255167


namespace bob_winning_strategy_2018_bob_winning_strategy_2019_l255_255973

-- Define the pudding movement rules for Alice and Bob.
def puddingMoveAlice (x y : ℤ) : set (ℤ × ℤ) :=
  {(x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2)}

def puddingMoveBob (x y : ℤ) : set (ℤ × ℤ) :=
  {(x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1)}

-- Define the modulo condition for occupied positions.
def isOccupied (a b c d n : ℤ) : Prop :=
  (c % n = a % n) ∧ (d % n = b % n)

-- Define the main proof obligations.
theorem bob_winning_strategy_2018 (x₀ y₀ : ℤ) : 
  ∃ strategy : ℕ → ℤ × ℤ, 
  (∀ k, puddingMoveAlice (strategy (2*k)).1 (strategy (2*k)).2 = {strategy (2*k+1)} ∧ 
       puddingMoveBob (strategy (2*k+1)).1 (strategy (2*k+1)).2 = {strategy (2*k+2)}) 
  ∧ (∀ i j, i ≠ j → ¬ isOccupied (strategy i).1 (strategy i).2 (strategy j).1 (strategy j).2 2018) :=
sorry

theorem bob_winning_strategy_2019 (x₀ y₀ : ℤ) : 
  ∃ strategy : ℕ → ℤ × ℤ, 
  (∀ k, puddingMoveAlice (strategy (2*k)).1 (strategy (2*k)).2 = {strategy (2*k+1)} ∧ 
       puddingMoveBob (strategy (2*k+1)).1 (strategy (2*k+1)).2 = {strategy (2*k+2)}) 
  ∧ (∀ i j, i ≠ j → ¬ isOccupied (strategy i).1 (strategy i).2 (strategy j).1 (strategy j).2 2019) :=
sorry

end bob_winning_strategy_2018_bob_winning_strategy_2019_l255_255973


namespace marble_count_l255_255112

theorem marble_count (x : ℝ) (h1 : 0.5 * x = x / 2) (h2 : 0.25 * x = x / 4) (h3 : (27 + 14) = 41) (h4 : 0.25 * x = 27 + 14)
  : x = 164 :=
by
  sorry

end marble_count_l255_255112


namespace total_marbles_l255_255935

theorem total_marbles (x : ℕ) (h1 : 5 * x - 2 = 18) : 4 * x + 5 * x = 36 :=
by
  sorry

end total_marbles_l255_255935


namespace expected_balls_in_original_position_proof_l255_255847

-- Define the problem conditions as Lean definitions
def n_balls : ℕ := 10

def probability_not_moved_by_one_rotation : ℚ := 7 / 10

def probability_not_moved_by_two_rotations : ℚ := (7 / 10) * (7 / 10)

def expected_balls_in_original_position : ℚ := n_balls * probability_not_moved_by_two_rotations

-- The statement representing the proof problem
theorem expected_balls_in_original_position_proof :
  expected_balls_in_original_position = 4.9 :=
  sorry

end expected_balls_in_original_position_proof_l255_255847


namespace seats_capacity_l255_255754

-- Given conditions
variables (x : ℕ) -- the number of people each seat can hold
constant left_seats : ℕ := 15 -- number of seats on the left side
constant right_seats : ℕ := 12 -- number of seats on the right side (15 - 3)
constant back_seat : ℕ := 10 -- number of people that can sit in the back seat
constant total_capacity : ℕ := 91 -- total number of people the bus can hold

-- The mathematical proof problem
theorem seats_capacity (h : 15 * x + 12 * x + 10 = 91) : x = 3 :=
by
  sorry

end seats_capacity_l255_255754


namespace smallest_m_l255_255155

-- Define the sequence recursively
def x : ℕ → ℚ
| 0     := 3
| (n+1) := (x n)^2 + 4 * (x n) + 6) / ((x n) + 4)

-- Define the problem statement
theorem smallest_m (m : ℕ) : 
  (∀ n, x n = 3) → (∃ m, m = 0) :=
by {
  intro h,
  use 0,
  apply h,
  sorry
}

end smallest_m_l255_255155


namespace minimum_odd_numbers_in_A_P_l255_255369

-- Polynomials and assumptions
def degree (P : Polynomial ℝ) : ℕ := P.natDegree

-- The set A_P is defined as the set of all numbers x for which P(x) gives a certain value
def A_P (P : Polynomial ℝ) : Set ℝ := {x : ℝ | P.eval x = P.eval 8}

-- Define the main theorem statement
theorem minimum_odd_numbers_in_A_P (P : Polynomial ℝ) (hdeg : degree P = 8) (h8 : 8 ∈ A_P P) : 
  ∃ n, n = 1 ∧ ∃ (x : ℝ), x ∈ A_P P ∧ ¬ (x % 2 = 0) := sorry

end minimum_odd_numbers_in_A_P_l255_255369


namespace color_of_241st_marble_l255_255945

def sequence_color (n : ℕ) : String :=
  if n % 14 < 6 then "blue"
  else if n % 14 < 11 then "red"
  else "green"

theorem color_of_241st_marble : sequence_color 240 = "blue" :=
  by
  sorry

end color_of_241st_marble_l255_255945


namespace solid_volume_l255_255667

noncomputable def volume_of_solid {v : ℝ^3} (h : RE'inner v v = RE'inner v (⟨6, -18, 12⟩ : ℝ^3)) : ℝ :=
  126 * Real.pi * Real.sqrt 14

theorem solid_volume (v : ℝ^3) (h : RE'inner v v = RE'inner v (⟨6, -18, 12⟩ : ℝ^3)) :
  volume_of_solid h = 126 * Real.pi * Real.sqrt 14 :=
sorry

end solid_volume_l255_255667


namespace spotted_females_less_than_combined_horned_and_unique_l255_255755

def num_cows := 450
def ratio_m : ℕ := 3
def ratio_f : ℕ := 2
def ratio_t : ℕ := 1
def percent_m_horned : ℝ := 0.6
def percent_f_spotted : ℝ := 0.5
def percent_t_unique : ℝ := 0.7

theorem spotted_females_less_than_combined_horned_and_unique :
  let num_m := (num_cows * ratio_m) / (ratio_m + ratio_f + ratio_t),
      num_f := (num_cows * ratio_f) / (ratio_m + ratio_f + ratio_t),
      num_t := (num_cows * ratio_t) / (ratio_m + ratio_f + ratio_t),
      m_horned := num_m * percent_m_horned,
      f_spotted := num_f * percent_f_spotted,
      t_unique := num_t * percent_t_unique
  in f_spotted - (m_horned + t_unique) = -112 :=
  sorry

end spotted_females_less_than_combined_horned_and_unique_l255_255755


namespace ratio_of_fractions_l255_255915

theorem ratio_of_fractions (x y : ℝ) (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) : 
  (1 / 5 * x) / (1 / 6 * y) = 0.72 :=
sorry

end ratio_of_fractions_l255_255915


namespace cylinder_section_volume_l255_255204

theorem cylinder_section_volume (a : ℝ) :
  let volume := (π * a^3 / 4)
  let section1_volume := volume * (1 / 3)
  let section2_volume := volume * (1 / 4)
  let enclosed_volume := (section1_volume - section2_volume) / 2
  enclosed_volume = π * a^3 / 24 := by
  sorry

end cylinder_section_volume_l255_255204


namespace count_terms_divisible_by_19_l255_255728

theorem count_terms_divisible_by_19 
: ∃ (N : ℕ), N = (finset.range 500).filter (λ n, (9^n.val + 1) % 19 = 0).card :=
by 
  sorry

end count_terms_divisible_by_19_l255_255728


namespace sine_shift_left_l255_255465

theorem sine_shift_left (x : ℝ) :
  (λ x, Real.sin (2 * x)) (x + π / 6) = Real.sin (2 * x + π / 3) :=
by
  sorry

end sine_shift_left_l255_255465


namespace john_shower_usage_l255_255776

/-- John showers every other day for 4 weeks, with each shower 10 minutes long, and uses 280 gallons of water in those 4 weeks. Prove that he uses 2 gallons of water per minute. -/
theorem john_shower_usage :
  let total_days := 4 * 7 in
  let days_between_showers := 2 in
  let number_of_showers := total_days / days_between_showers in
  let minutes_per_shower := 10 in
  let total_minutes := number_of_showers * minutes_per_shower in
  let total_gallons := 280 in
  total_gallons / total_minutes = 2 :=
sorry

end john_shower_usage_l255_255776


namespace range_of_function_l255_255311

theorem range_of_function (x : ℝ) : 
    (sqrt ((x - 1) / (x - 2))) = sqrt ((x - 1) / (x - 2)) → (x ≤ 1 ∨ x > 2) :=
by
  sorry

end range_of_function_l255_255311


namespace license_plates_count_l255_255117

theorem license_plates_count : (6 * 10^5 * 26^3) = 10584576000 := by
  sorry

end license_plates_count_l255_255117


namespace snail_returns_to_starting_point_l255_255957

theorem snail_returns_to_starting_point (v : ℝ) :
  (∀ t : ℝ, (t ≥ 0) →
      (∃ n : ℤ, t = n * 3600) ↔  -- 3600 seconds in an hour
      (∃ m : ℕ, t = m * 900) →  -- 15 minutes is 900 seconds
        let x := m % 4 = 0 in x → (0,0)) :=
sorry

end snail_returns_to_starting_point_l255_255957


namespace transformed_roots_eq_original_roots_l255_255280

theorem transformed_roots_eq_original_roots:
  (∃ a b c d : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∧ (a^4 + 2*a^3 - 5 = 0 ∧ 
  b^4 + 2*b^3 - 5 = 0 ∧ c^4 + 2*c^3 - 5 = 0 ∧ d^4 + 2*d^3 - 5 = 0)) →
  (∃ a b c d : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∧ 
  (\(frac{a * b * c}{d})^4 + 2*\(frac{a * b * c}{d})^3 - 5 = 0 ∧ 
  (\frac{a * b * d}{c})^4 + 2*\(\frac{a * b * d}{c})^3 - 5 = 0 ∧ 
  (\frac{a * c * d}{b})^4 + 2*\(\frac{a * c * d}{b})^3 - 5 = 0 ∧ 
  (\frac{b * c * d}{a})^4 + 2*\(\frac{b * c * d}{a})^3 - 5 = 0))
  :=
sorry

end transformed_roots_eq_original_roots_l255_255280


namespace smallest_arithmetic_geometric_seq_sum_l255_255470

variable (A B C D : ℕ)

noncomputable def arithmetic_seq (A B C : ℕ) (d : ℕ) : Prop :=
  B - A = d ∧ C - B = d

noncomputable def geometric_seq (B C D : ℕ) : Prop :=
  C = (5 / 3) * B ∧ D = (25 / 9) * B

theorem smallest_arithmetic_geometric_seq_sum :
  ∃ A B C D : ℕ, 
    arithmetic_seq A B C 12 ∧ 
    geometric_seq B C D ∧ 
    (A + B + C + D = 104) :=
sorry

end smallest_arithmetic_geometric_seq_sum_l255_255470


namespace min_abs_sum_l255_255342

theorem min_abs_sum (a b c d : ℤ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h : ∀m n, (Matrix.mul (Matrix.of 2 2 ![(a, b), (c, d)]) (Matrix.of 2 2 ![(a, b), (c, d)])
    m n) = (Matrix.of 2 2 ![(9, 0), (0, 9)]) m n) :
    |a| + |b| + |c| + |d| = 8 :=
begin
  sorry
end

end min_abs_sum_l255_255342


namespace average_distance_per_day_l255_255449

def distance_Monday : ℝ := 4.2
def distance_Tuesday : ℝ := 3.8
def distance_Wednesday : ℝ := 3.6
def distance_Thursday : ℝ := 4.4

def total_distance : ℝ := distance_Monday + distance_Tuesday + distance_Wednesday + distance_Thursday

def number_of_days : ℕ := 4

theorem average_distance_per_day : total_distance / number_of_days = 4 := by
  sorry

end average_distance_per_day_l255_255449


namespace angle_is_pi_over_4_l255_255724

noncomputable def angle_between_vectors
  (a b : ℝ × ℝ)
  (norm_b : ℝ)
  (norm_a_plus_2b : ℝ) : ℝ :=
if h : (a = (1, real.sqrt 3)) ∧ (norm_b = real.sqrt 2) ∧ (norm_a_plus_2b = 2 * real.sqrt 5) then
  real.arccos ((a.1 * b.1 + a.2 * b.2) / (real.sqrt (a.1^2 + a.2^2) * norm_b))
else
  0

theorem angle_is_pi_over_4 : ∀ (a b : ℝ × ℝ), 
  a = (1, real.sqrt 3) →
  ∃ θ, θ = angle_between_vectors a b (real.sqrt 2) (2 * real.sqrt 5) ∧ θ = real.pi / 4 :=
begin
  intros a b ha, 
  use real.pi / 4,
  split,
  { sorry },
  { sorry }
end

end angle_is_pi_over_4_l255_255724


namespace distance_between_lamps_l255_255420

/-- 
A rectangular classroom measures 10 meters in length. Two lamps emitting conical light beams with a 90° opening angle 
are installed on the ceiling. The first lamp is located at the center of the ceiling and illuminates a circle on the 
floor with a diameter of 6 meters. The second lamp is adjusted such that the illuminated area along the length 
of the classroom spans a 10-meter section without reaching the opposite walls. Prove that the distance between the 
two lamps is 4 meters.
-/
theorem distance_between_lamps : 
  ∀ (length width height : ℝ) (center_illum_radius illum_length : ℝ) (d_center_to_lamp1 d_center_to_lamp2 dist_lamps : ℝ),
  length = 10 ∧ d_center_to_lamp1 = 3 ∧ d_center_to_lamp2 = 1 ∧ dist_lamps = 4 → d_center_to_lamp1 - d_center_to_lamp2 = dist_lamps :=
by
  intros length width height center_illum_radius illum_length d_center_to_lamp1 d_center_to_lamp2 dist_lamps conditions
  sorry

end distance_between_lamps_l255_255420


namespace batsman_average_increase_l255_255534

def batsman_problem : Prop :=
  ∀ (runs_in_17th_inning : ℕ) (average_after_17th : ℕ),
    runs_in_17th_inning = 56 →
    average_after_17th = 8 →
    let total_runs_after_17th := 17 * average_after_17th in 
    let total_runs_before_17th := total_runs_after_17th - runs_in_17th_inning in
    let average_before_17th := total_runs_before_17th / 16 in
    average_after_17th - average_before_17th = 3

theorem batsman_average_increase : batsman_problem :=
by
  intro runs_in_17th_inning average_after_17th
  intros h1 h2
  rw [h1, h2]
  let total_runs_after_17th := 17 * 8
  let total_runs_before_17th := total_runs_after_17th - 56
  let average_before_17th := total_runs_before_17th / 16
  sorry

end batsman_average_increase_l255_255534


namespace combinations_eq_765_l255_255730

theorem combinations_eq_765 : 
  ∃ C,
  C = (∑ a in Finset.range 7, (Nat.choose 6 a) * (Nat.choose 9 (16 - 2 * a))) ∧ 
  C = 765 :=
by
  sorry

end combinations_eq_765_l255_255730


namespace existence_of_polynomials_l255_255923

-- Definitions of the given conditions
def Γ (f : ℚ[X]) : ℚ :=
  f.coeffs.map (λ a, a^2).sum

/-- Main statement of the problem -/
theorem existence_of_polynomials :
  Σ' (Q : ℚ[X] → Prop), (∀ k, (1 ≤ k ∧ k ≤ 2^2019) → Q k)
  ∧ (∀ k, Q k → degree Q k = 2020)
  ∧ (∀ (k : ℕ) (n : ℕ), (1 ≤ k ∧ k ≤ 2^2019 ∧ 1 ≤ n) → Γ(Q k ^ n) = Γ(P ^ n))
  ∧ ∃ S, 2^2019 ⊆ S ∧ ∀ (k : ℕ), S k → (Q k)
  where Σ' (Q : ℚ[X] → Prop) := 
    ∃ Q : ℕ → ℚ[X], 
      (∀ k, 1 ≤ k ∧ k ≤ 2^2019 → Q k)
      ∧ (∀ k, ∀ n, 1 ≤ k ∧ k ≤ 2^2019 ∧ 1 ≤ n → Γ (Q k ^ n) = Γ (P ^ n)) := 
    sorry

end existence_of_polynomials_l255_255923


namespace binary_to_base4_conversion_l255_255991

theorem binary_to_base4_conversion :
  let b := 110110100
  let b_2 := Nat.ofDigits 2 [1, 1, 0, 1, 1, 0, 1, 0, 0]
  let b_4 := Nat.ofDigits 4 [3, 1, 2, 2, 0]
  b_2 = b → b_4 = 31220 :=
by
  intros b b_2 b_4 h
  sorry

end binary_to_base4_conversion_l255_255991


namespace range_of_x_l255_255315

theorem range_of_x (x : ℝ) : (sqrt ((x - 1) / (x - 2)) : ℝ) ≥ 0 → (x > 2 ∨ x ≤ 1) :=
by
  sorry

end range_of_x_l255_255315


namespace tan_power_difference_l255_255275

noncomputable def cot (x : ℝ) : ℝ := 1 / tan x

theorem tan_power_difference (x : ℝ) (h : cos x * (1 / tan x) = sin x^2) :
  tan x^6 - tan x^2 = 1 :=
by
  sorry

end tan_power_difference_l255_255275


namespace eventually_periodic_l255_255016

open Nat

noncomputable def sequence_a (a0 : ℕ) (b0 : ℕ) : ℕ → ℕ 
| 0     := a0
| (n+1) := gcd (sequence_a n) (sequence_b a0 b0 n) + 1

noncomputable def sequence_b (a0 : ℕ) (b0 : ℕ) : ℕ → ℕ 
| 0     := b0
| (n+1) := lcm (sequence_a a0 b0 n) (sequence_b a0 b0 n) - 1

theorem eventually_periodic (a0 b0 : ℕ) (h0 : a0 ≥ 2) (h1 : b0 ≥ 2) :
  ∃ N t, t > 0 ∧ ∀ n, N ≤ n → sequence_a a0 b0 (n + t) = sequence_a a0 b0 n :=
sorry

end eventually_periodic_l255_255016


namespace f_is_even_and_periodic_value_of_f_neg2010_minus_f_2009_l255_255231

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ Ico 0 2 then logBase 2 (x + 1)
else if x.mod 4 ∈ Ico 0 2 then logBase 2 ((x.mod 4) + 1)
else -logBase 2 ((x.mod 4) + 1) -- this helps to deal with periodicity and even function

theorem f_is_even_and_periodic :
  (∀ x, f (-x) = f x) ∧ (∀ x, 0 ≤ x → f (x + 2) = - f x) := 
by sorry

theorem value_of_f_neg2010_minus_f_2009 : 
  f (-2010) - f 2009 = -1 :=
by sorry

end f_is_even_and_periodic_value_of_f_neg2010_minus_f_2009_l255_255231


namespace tangents_from_point_to_circle_l255_255292

theorem tangents_from_point_to_circle
  (k : ℝ) :
  let circle_eq := λ x y : ℝ, x^2 + y^2 - 2 * k * x - 2 * y + k^2 - k = 0
  let P := (2 : ℝ, 2 : ℝ)
  (∃ a b : ℝ, ∀ x y : ℝ, circle_eq x y = 0 → (a * x + b * y + 1 = 0 ∧ (a - 2)^2 + (b-2)^2 > a^2 + b^2 + k + 1)) ↔
  k ∈ Ioo (-1 : ℝ) (1 : ℝ) ∪ Ioi (4 : ℝ) := sorry

end tangents_from_point_to_circle_l255_255292


namespace expression_value_l255_255151

theorem expression_value : 
  29^2 - 27^2 + 25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 389 :=
by
  sorry

end expression_value_l255_255151


namespace find_m_l255_255250

noncomputable def f (x : ℝ) (a : ℝ) (m : ℝ) := real.log (1 - mx) / (x - 1)

def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f(x)

theorem find_m (a m : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  odd_function (λ x, real.log (1 - mx) / (x - 1)) → m = -1 :=
by {
  sorry
}

end find_m_l255_255250


namespace smallest_prime_factor_2005_2007_l255_255900

def is_odd (n : ℕ) : Prop := ¬ (n % 2 = 0)

theorem smallest_prime_factor_2005_2007 : 
  is_odd 2005 → is_odd 2007 → Nat.min_fac (2005 ^ 2007 + 2007 ^ 20015) = 2 :=
by
  sorry

end smallest_prime_factor_2005_2007_l255_255900


namespace repeating_decimal_to_fraction_l255_255645

theorem repeating_decimal_to_fraction (x : ℝ) (h : x = 0.3 + 0.0666...) : x = 11 / 30 := by
  sorry

end repeating_decimal_to_fraction_l255_255645


namespace monotonic_intervals_range_of_k_l255_255243

noncomputable def f (x a : ℝ) : ℝ :=
  (x - a - 1) * Real.exp x - (1/2) * x^2 + a * x

-- Conditions: a > 0
variables (a : ℝ) (h_a : 0 < a)

-- Part (1): Monotonic Intervals
theorem monotonic_intervals :
  (∀ x, f x a < f (x + 1) a ↔ x < 0 ∨ a < x) ∧
  (∀ x, f (x + 1) a < f x a ↔ 0 < x ∧ x < a) :=
  sorry

-- Part (2): Range of k
theorem range_of_k (x1 x2 : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) :
  (f x1 a - f x2 a < k * a^3) ↔ k ≥ -1/6 :=
  sorry

end monotonic_intervals_range_of_k_l255_255243


namespace probability_meeting_a_l255_255782

theorem probability_meeting_a (t_K t_Z : ℝ) (h_K : 0 ≤ t_K ∧ t_K ≤ 60) (h_Z : 0 ≤ t_Z ∧ t_Z ≤ 60) :
  (| t_K - t_Z | ≤ 10) → (sorry : set.univ) :=
begin
  sorry
end

end probability_meeting_a_l255_255782


namespace convex_pentagon_medians_not_collinear_l255_255928

theorem convex_pentagon_medians_not_collinear :
  ∀ (A B C D E : ℝ×ℝ), 
  convex_pentagon A B C D E →
  (∃ M N P : ℝ×ℝ, 
    is_centroid A B C M ∧
    is_centroid A C D N ∧
    is_centroid A D E P ∧
    ¬ collinear M N P) :=
sorry

end convex_pentagon_medians_not_collinear_l255_255928


namespace distinct_x_intercepts_count_l255_255727

theorem distinct_x_intercepts_count :
    let y := (λ x : ℝ, (x - 5) * (x^2 + 5 * x + 6) * (x - 1))
    {x : ℝ | y x = 0}.toFinset.card = 4 := by
  sorry

end distinct_x_intercepts_count_l255_255727


namespace liam_art_kits_l255_255403

theorem liam_art_kits (students total_artworks : ℕ) (art_kits_per_2_students : students % 2 = 0) 
    (half_students1 half_students2 : students / 2)
    (artworks1 artworks2 : ℕ) (total_half_artworks1 total_half_artworks2 : ℕ) : 
    students = 10 →
    (half_students1 = 5 ∧ half_students2 = 5) →
    (artworks1 = 3 ∧ artworks2 = 4) →
    (total_half_artworks1 = half_students1 * artworks1 ∧ total_half_artworks2 = half_students2 * artworks2) →
    total_artworks = total_half_artworks1 + total_half_artworks2 →
    total_artworks = 35 →
    students / 2 = 5 :=
by
  sorry

end liam_art_kits_l255_255403


namespace expected_number_of_explorers_no_advice_l255_255590

-- Define the problem
theorem expected_number_of_explorers_no_advice
  (n : ℕ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∑ j in Finset.range n, (1 - p) ^ j) / p = (1 - (1 - p) ^ n) / p := by
  sorry

end expected_number_of_explorers_no_advice_l255_255590


namespace sum_of_ages_l255_255876

variables (Henry Jill : ℕ)
variable (h_age : Henry = 25)
variable (j_age : Jill = 16)

theorem sum_of_ages : Henry + Jill = 41 :=
by
  rw [h_age, j_age]
  exact rfl

end sum_of_ages_l255_255876


namespace arithmetic_geometric_mean_inequality_l255_255332

theorem arithmetic_geometric_mean_inequality (n : ℕ) (a : Fin n → ℝ)
  (h : ∀ i, 0 < a i) :
  let m_g := (∏ i, a i)^(1 / n)
  let m_a := (∑ i, a i) / n
  in (1 + m_g)^n ≤ (∏ i, (1 + a i)) ∧ (∏ i, (1 + a i)) ≤ (1 + m_a)^n :=
by
  sorry

end arithmetic_geometric_mean_inequality_l255_255332


namespace correct_propositions_l255_255136

open Classical

variable {a b q : ℝ} {p : Prop} {x : ℝ}

-- Definitions for proposition conditions
def prop1 := ∀ a b : ℝ, ¬ (a = 0 → ab = 0) = (a = 0 → ab ≠ 0)
def prop2 := ∀ q : ℝ, ∀ x, (q ≤ 1 → ∃ x : ℝ, x^2 + 2*x + q = 0)
def prop3 := ¬ p ∧ (p ∨ q) → q
def prop4 := ∀ a : ℝ, (0 < a ∧ a < 1) → (log a (a + 1) < log a (1 + 1/a))

-- Problem statement: identify correct propositions
theorem correct_propositions : (prop2 ∧ prop3) :=
by
  sorry

end correct_propositions_l255_255136


namespace percent_decrease_of_larger_angle_l255_255471

theorem percent_decrease_of_larger_angle (a b : ℝ) (h1 : a + b = 90)
  (h2 : a / b = 3 / 4) : 
  let new_a := a * 1.20,
      new_b := 90 - new_a,
      percent_decrease := ((b - new_b) / b) * 100
  in percent_decrease ≈ 14.97 := 
sorry

end percent_decrease_of_larger_angle_l255_255471


namespace gage_needs_to_skate_l255_255677

noncomputable def gage_average_skating_time (d1 d2: ℕ) (t1 t2 t8: ℕ) : ℕ :=
  let total_time := (d1 * t1) + (d2 * t2) + t8
  (total_time / (d1 + d2 + 1))

theorem gage_needs_to_skate (t1 t2: ℕ) (d1 d2: ℕ) (avg: ℕ) 
  (t1_minutes: t1 = 80) (t2_minutes: t2 = 105) 
  (days1: d1 = 4) (days2: d2 = 3) (avg_goal: avg = 95) :
  gage_average_skating_time d1 d2 t1 t2 125 = avg :=
by
  sorry

end gage_needs_to_skate_l255_255677


namespace real_part_of_complex_expr_l255_255663

-- Conditions
def i := Complex.I

-- Question
def complex_expr := (i) / (1 + i) * i

-- Answer
def real_part := Complex.re complex_expr

-- Proof Statement
theorem real_part_of_complex_expr : real_part = 1 / 2 := 
  sorry

end real_part_of_complex_expr_l255_255663


namespace circumscribed_sphere_volume_of_regular_tetrahedron_l255_255239

noncomputable def volume_of_circumscribed_sphere (r: ℝ) : ℝ :=
  let R := 3 * r in
  (4 / 3) * Real.pi * (R^3)

theorem circumscribed_sphere_volume_of_regular_tetrahedron (h: (4 / 3) * Real.pi * (1 / 4) = 1) :
  volume_of_circumscribed_sphere (real.sqrt (3 / 4) / 4) = 27 :=
by
  -- Definitions and conditions
  have h1 : (4 / 3) * Real.pi * (real.sqrt (3 / 4) / 4)^3 = 1 := h
  sorry -- proof of 27

end circumscribed_sphere_volume_of_regular_tetrahedron_l255_255239


namespace maximum_triangle_area_l255_255964

-- Define the maximum area of a triangle given two sides.
theorem maximum_triangle_area (a b : ℝ) (h_a : a = 1984) (h_b : b = 2016) :
  ∃ (max_area : ℝ), max_area = 1998912 :=
by
  sorry

end maximum_triangle_area_l255_255964


namespace club_planning_committee_l255_255940

theorem club_planning_committee : Nat.choose 20 3 = 1140 := 
by sorry

end club_planning_committee_l255_255940


namespace check_numbers_has_property_P_l255_255397

def has_property_P (n : ℤ) : Prop :=
  ∃ x y z : ℤ, n = x^3 + y^3 + z^3 - 3 * x * y * z

theorem check_numbers_has_property_P :
  has_property_P 1 ∧ has_property_P 5 ∧ has_property_P 2014 ∧ ¬has_property_P 2013 :=
by
  sorry

end check_numbers_has_property_P_l255_255397


namespace log_domain_l255_255660

theorem log_domain (x : ℝ) : 3 - 2 * x > 0 ↔ x < 3 / 2 :=
by
  sorry

end log_domain_l255_255660


namespace calculate_F_2_f_3_l255_255344

def f (a : ℕ) : ℕ := a ^ 2 - 3 * a + 2

def F (a b : ℕ) : ℕ := b ^ 2 + a + 1

theorem calculate_F_2_f_3 : F 2 (f 3) = 7 :=
by
  show F 2 (f 3) = 7
  sorry

end calculate_F_2_f_3_l255_255344


namespace count_pos_ints_satisfying_condition_l255_255996

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem count_pos_ints_satisfying_condition :
  ∃! (S : Finset ℕ), S.card = 180 ∧ ∀ n ∈ S, n % 6 = 0 ∧ lcm (factorial 6) n = 6 * Nat.gcd (factorial 12) n :=
by {
  sorry
}

end count_pos_ints_satisfying_condition_l255_255996


namespace solve_exp_log_eq_correct_l255_255331

noncomputable def solve_exp_log_eq (a x : ℝ) : Prop :=
  x > 0 ∧ a ≠ 1 ∧ a > 0 → a ^ x = x ^ x + log a (log a x) → x = a

theorem solve_exp_log_eq_correct (a : ℝ) (h₁ : a ≠ 1) (h₂ : a > 0) :
  ∀ x : ℝ, solve_exp_log_eq a x :=
begin
  intros x h,
  sorry
end

end solve_exp_log_eq_correct_l255_255331


namespace prove_statements_l255_255703

noncomputable def circle_center : ℝ × ℝ := ((5 + 1) / 2, (6 + 2) / 2)
noncomputable def circle_radius : ℝ := real.sqrt ((5 - circle_center.1)^2 + (2 - circle_center.2)^2)
noncomputable def circle (p : ℝ × ℝ) : Prop := (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2

def M1 : ℝ × ℝ := (2, 0)
def M2 : ℝ × ℝ := (-2, 0)
def O : ℝ × ℝ := (0, 0)

def N1 : ℝ × ℝ := ((O.1 + M1.1) / 2, (O.2 + M1.2) / 2)
def N2 : ℝ × ℝ := ((O.1 + M2.1) / 2, (O.2 + M2.2) / 2)

theorem prove_statements (A : ℝ × ℝ) (hA : circle A) :
  dist A M2 ≤ real.sqrt 41 + 2 * real.sqrt 2  ∧
  (∃ θ : ℝ, θ ≥ 15 ∧ angle A N2 N1 = θ) ∧
  ∃ min_val : ℝ, min_val = 32 - 20 * real.sqrt 2 ∧ (A.1 - N1.1) * (A.2 - N2.2) = min_val :=
by
  sorry

end prove_statements_l255_255703


namespace shape_of_theta_eq_c_is_plane_l255_255761

-- Define the spherical coordinates and the condition
def spherical_coordinates (ρ θ φ : ℝ) := (ρ >= 0) ∧ (0 ≤ φ) ∧ (φ ≤ π)

-- State the theorem describing the shape in spherical coordinates when θ = c
theorem shape_of_theta_eq_c_is_plane (ρ φ c : ℝ) (h: spherical_coordinates ρ c φ) : 
    ∃ x y z : ℝ, z = ρ * cos φ ∧ sqrt (x^2 + y^2) = ρ * sin φ ∧ θ = c ∧ ∃ α : ℝ, y = alpha then
    sorry

end shape_of_theta_eq_c_is_plane_l255_255761


namespace inequality_has_solutions_l255_255811

noncomputable def f (n : ℕ) (a : ℝ) (x : ℝ) : ℝ :=
  if n ≥ 2 then
    Real.log ((Finset.range n).sum (λ i, (i+1)^x) + n^x * a) / n
  else
    0

theorem inequality_has_solutions (n : ℕ) (a : ℝ) :
  (∃ x ∈ set.Ici 1, f n a x > (x-1) * Real.log n) ↔ a > 1/2 :=
begin
  sorry
end

end inequality_has_solutions_l255_255811


namespace calculate_three_Z_five_l255_255619

def Z (a b : ℤ) : ℤ := b + 15 * a - a^3

theorem calculate_three_Z_five : Z 3 5 = 23 :=
by
  -- The proof goes here
  sorry

end calculate_three_Z_five_l255_255619


namespace minimum_value_theorem_l255_255740

noncomputable def minimum_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  let circle_center := (-1, 2)
  let radius := 2
  let line := 2 * a * circle_center.1 - b * circle_center.2 + 2
  let d := abs line / sqrt (4 * a^2 + b^2)
  ∃ (a b : ℝ), 
    (ha a) → (hb b) → 
    d = radius ∧ 
    2 * sqrt (4 * a^2 + b^2) = abs (2 - 2 * a - 2 * b) ∧ 
    (1 - a - b = -2) ∧ 
    let value := (1/a + 4/b) 
    min value = 9

theorem minimum_value_theorem : ∀ (a b : ℝ), minimum_value_problem a b := 
begin
  intros a b,
  unfold minimum_value_problem,
  sorry
end

end minimum_value_theorem_l255_255740


namespace Polynomial_cannot_have_odd_degree_l255_255869

variable (P : ℤ × ℤ → ℤ)

-- The condition that for any integer n ≥ 0, each of the polynomials P(n, y) and P(x, n) is either identically zero or has degree no higher than n
def condition (P : ℤ × ℤ → ℤ) : Prop :=
  ∀ n : ℤ, ∀ y x : ℤ, n ≥ 0 → (
    (∀ k : ℤ, k > n → P (k, y) = 0) ∨ (∀ k : ℤ, k > n → P (x, k) = 0) ∨ (
      (∀ k l : ℤ, k ≤ n → l ≤ n → degree (λ y, P (k, y)) ≤ n ∧ degree (λ x, P (x, l)) ≤ n
    )
  )

theorem Polynomial_cannot_have_odd_degree (P : ℤ × ℤ → ℤ) :
  condition P → ¬ (∃ d : ℕ, d % 2 = 1 ∧ ∃ f : ℤ → ℤ, ∀ x : ℤ, P (x, x) = poly f x d) := sorry

end Polynomial_cannot_have_odd_degree_l255_255869


namespace find_r_plus_s_l255_255890

noncomputable def midpoint (D E : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((D.fst + E.fst) / 2, (D.snd + E.snd) / 2)

def triangle_area (D E F : (ℝ × ℝ)) : ℝ :=
  ((D.fst * (E.snd - F.snd) + E.fst * (F.snd - D.snd) + F.fst * (D.snd - E.snd)) / 2).abs

def median_slope (D E M : (ℝ × ℝ)) (slope : ℝ) : Prop :=
  (M.snd - E.snd) / (M.fst - E.fst) = slope

theorem find_r_plus_s :
  ∃(r s : ℝ), 
    let D := (13, 22)
    let E := (24, 23)
    let F := (r, s)
    let M := midpoint D E
    let area := triangle_area D E F
    let slope := median_slope D E M (-7)
    in 
    area = 84 ∧ slope = True ∧ (r + s = 38) :=
begin
  sorry
end

end find_r_plus_s_l255_255890


namespace beads_initial_state_repeats_l255_255604

-- Define the setup of beads on a circular wire
structure BeadConfig (n : ℕ) :=
(beads : Fin n → ℝ)  -- Each bead's position indexed by a finite set, ℝ denotes angular position

-- Define the instantaneous collision swapping function
def swap (n : ℕ) (i j : Fin n) (config : BeadConfig n) : BeadConfig n :=
⟨fun k => if k = i then config.beads j else if k = j then config.beads i else config.beads k⟩

-- Define what it means for a configuration to return to its initial state
def returns_to_initial (n : ℕ) (initial : BeadConfig n) (t : ℝ) : Prop :=
  ∃ (config : BeadConfig n), (∀ k, config.beads k = initial.beads k) ∧ (config = initial)

-- Specification of the problem
theorem beads_initial_state_repeats (n : ℕ) (initial : BeadConfig n) (ω : Fin n → ℝ) :
  (∀ k, ω k > 0) →  -- condition that all beads have positive angular speed, either clockwise or counterclockwise
  ∃ t : ℝ, t > 0 ∧ returns_to_initial n initial t := 
by
  sorry

end beads_initial_state_repeats_l255_255604


namespace cyclists_meeting_time_l255_255463

-- Define the parameters
def perimeter : ℝ := 1 / 3
def speed_a : ℝ := 6
def speed_b : ℝ := 9
def speed_c : ℝ := 12
def speed_d : ℝ := 15

-- Define the times for one complete lap for each cyclist
def time_for_one_lap (perimeter speed : ℝ) : ℝ := perimeter / speed

-- Cyclist times
def time_a := time_for_one_lap perimeter speed_a
def time_b := time_for_one_lap perimeter speed_b
def time_c := time_for_one_lap perimeter speed_c
def time_d := time_for_one_lap perimeter speed_d

-- Define the LCM of the times for one complete lap
def lcm_of_times : ℝ := 1 / Int.gcd 18 (Int.gcd 27 (Int.gcd 36 45)).toNat

-- Define the first common meeting time
def first_meeting_time : ℝ := lcm_of_times

-- Define the fourth meeting time
def fourth_meeting_time : ℝ := 4 * first_meeting_time

-- Convert the result to minutes and seconds
def meeting_time_minutes : ℕ := (fourth_meeting_time * 60).toInt
def meeting_time_seconds : ℝ := fourth_meeting_time * 3600 - meeting_time_minutes * 60

-- State the final theorem
theorem cyclists_meeting_time :
  fourth_meeting_time = 26 + 40 / 60 :=
sorry

end cyclists_meeting_time_l255_255463


namespace cos_15_eq_l255_255528

theorem cos_15_eq :
  real.cos (15 * real.pi / 180) = (real.sqrt 6 + real.sqrt 2) / 4 :=
by
  sorry

end cos_15_eq_l255_255528


namespace collinear_points_b_values_l255_255656

theorem collinear_points_b_values (b : ℝ) :
  let determinant := λ b : ℝ, 2 * 2 - b * b
  in determinant b = 0 ↔ (b = 2 ∨ b = -2) :=
by
  sorry

end collinear_points_b_values_l255_255656


namespace total_birds_count_l255_255968

def blackbirds_per_tree : ℕ := 3
def tree_count : ℕ := 7
def magpies : ℕ := 13

theorem total_birds_count : (blackbirds_per_tree * tree_count) + magpies = 34 := by
  sorry

end total_birds_count_l255_255968


namespace five_digit_numbers_with_conditions_l255_255468

noncomputable def num_five_digit_numbers_with_conditions : ℕ :=
  let N := {n : ℕ | 10000 ≤ n ∧ n < 100000}
  let A := {n ∈ N | (n / 10000) = 2} -- The number starts with 2
  let B := {n ∈ A | (n % 10000).digits.count (n / 10000) = 2} -- Exactly two identical digits
  B.card

theorem five_digit_numbers_with_conditions :
  num_five_digit_numbers_with_conditions = 3776 :=
sorry

end five_digit_numbers_with_conditions_l255_255468


namespace board_game_max_n_l255_255105

theorem board_game_max_n (n : ℕ) (P : Finset (Fin n)) (Rounds : Finset (Finset (Fin n))) 
  (h1 : ∀ r ∈ Rounds, r.card = 3) 
  (h2 : Rounds.card = n)
  (h3 : ∀ (x y : Fin n), x ≠ y → ∃ r ∈ Rounds, x ∈ r ∧ y ∈ r) :
  n ≤ 7 :=
begin
  sorry
end

end board_game_max_n_l255_255105


namespace gcd_two_5_digit_integers_l255_255285

theorem gcd_two_5_digit_integers (a b : ℕ) 
  (h1 : 10^4 ≤ a ∧ a < 10^5)
  (h2 : 10^4 ≤ b ∧ b < 10^5)
  (h3 : 10^8 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^9) :
  Nat.gcd a b < 10^2 :=
by
  sorry  -- Skip the proof

end gcd_two_5_digit_integers_l255_255285


namespace cricket_boundaries_l255_255546

theorem cricket_boundaries (total_score runs_percent : ℝ) (sixes boundaries : ℕ) 
  (H1 : total_score = 132)
  (H2 : runs_percent = 54.54545454545454)
  (H3 : sixes = 2)
  (H4 : total_score * (runs_percent / 100) ≈ 72)
  (H5 : 6 * sixes = 12)
  (H6 : total_score - (total_score * (runs_percent / 100)) - (6 * sixes) = 48)
  (H7 : boundaries = 48 / 4) : boundaries = 12 :=
sorry

end cricket_boundaries_l255_255546


namespace mark_sold_345_kites_in_15_days_l255_255413

noncomputable def kites_sold_on_nth_day (n : ℕ) : ℕ :=
  if h : n ≥ 1 then 2 + (n - 1) * 3 else 0

noncomputable def total_kites_sold (days : ℕ) : ℕ :=
  finset.sum (finset.range days) (λ n, kites_sold_on_nth_day (n + 1))

theorem mark_sold_345_kites_in_15_days :
  total_kites_sold 15 = 345 :=
sorry

end mark_sold_345_kites_in_15_days_l255_255413
