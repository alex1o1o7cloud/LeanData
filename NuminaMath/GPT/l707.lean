import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.LinearAlgebra.Basic
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.PrimePower
import Mathlib.Analysis.Gon
import Mathlib.Analysis.Integral.IntervalIntegral
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Centroid
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.NumberTheory.Divisors
import Mathlib.NumberTheory.Primes
import Mathlib.Order.Basic
import Mathlib.Probability.Binomial
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Topology.EuclideanSpace
import Mathlib.Trigonometry.Trigonometric

namespace triangles_not_similar_equal_area_perimeter_l707_707006

-- Definitions for the problem conditions
structure Point :=
  (x : ℝ) (y : ℝ)

structure Triangle :=
  (A B C : Point)

def arbitrary_point_in_triangle (T : Triangle) (P : Point) : Prop :=
  P ≠ ((T.A.x + T.B.x + T.C.x) / 3, (T.A.y + T.B.y + T.C.y) / 3) -- P is not the centroid

-- Lean statement for the math proof problem
theorem triangles_not_similar_equal_area_perimeter (T : Triangle) (P : Point)
  (h : arbitrary_point_in_triangle T P) :
  ¬ (∀ (t1 t2 : Triangle), (t1 ≈ t2) ∨ (area t1 = area t2) ∨ (perimeter t1 = perimeter t2)) := 
sorry

end triangles_not_similar_equal_area_perimeter_l707_707006


namespace price_reduction_percentage_l707_707151

def original_price : ℝ := 74.95
def sale_price : ℝ := 59.95
def expected_percentage_decrease : ℝ := 20.0

theorem price_reduction_percentage : 
  approximately_equal (((original_price - sale_price) / original_price) * 100) expected_percentage_decrease := 
by
  sorry

def approximately_equal (a b : ℝ) : Prop := abs (a - b) < 0.1

end price_reduction_percentage_l707_707151


namespace find_intersection_complement_B_find_A_minus_B_find_A_minus_A_minus_B_l707_707298

def U := Set ℝ
def A : Set ℝ := {x | x > 4}
def B : Set ℝ := {x | -6 < x ∧ x < 6}

theorem find_intersection (x : ℝ) : x ∈ A ∧ x ∈ B ↔ 4 < x ∧ x < 6 :=
by
  sorry

theorem complement_B (x : ℝ) : x ∉ B ↔ x ≥ 6 ∨ x ≤ -6 :=
by
  sorry

def A_minus_B : Set ℝ := {x | x ∈ A ∧ x ∉ B}

theorem find_A_minus_B (x : ℝ) : x ∈ A_minus_B ↔ x ≥ 6 :=
by
  sorry

theorem find_A_minus_A_minus_B (x : ℝ) : x ∈ (A \ A_minus_B) ↔ 4 < x ∧ x < 6 :=
by
  sorry

end find_intersection_complement_B_find_A_minus_B_find_A_minus_A_minus_B_l707_707298


namespace initial_ratio_l707_707350

-- Definitions of the initial state and conditions
variables (M W : ℕ)
def initial_men : ℕ := M
def initial_women : ℕ := W
def men_after_entry : ℕ := M + 2
def women_after_exit_and_doubling : ℕ := (W - 3) * 2
def current_men : ℕ := 14
def current_women : ℕ := 24

-- Theorem to prove the initial ratio
theorem initial_ratio (M W : ℕ) 
    (hm : men_after_entry M = current_men)
    (hw : women_after_exit_and_doubling W = current_women) :
  M / Nat.gcd M W = 4 ∧ W / Nat.gcd M W = 5 :=
by
  sorry

end initial_ratio_l707_707350


namespace sin_minus_cos_eq_neg_sqrt2_div3_l707_707993

theorem sin_minus_cos_eq_neg_sqrt2_div3
  (θ : ℝ)
  (h1 : sin θ + cos θ = 4 / 3)
  (h2 : 0 < θ ∧ θ < π / 4) :
  sin θ - cos θ = -real.sqrt 2 / 3 := 
sorry

end sin_minus_cos_eq_neg_sqrt2_div3_l707_707993


namespace sum_of_squares_of_primes_divisible_by_6_implies_n_divisible_by_6_l707_707805

/-
Problem description:
Given the sum of the squares of n prime numbers, each greater than 5, is divisible by 6,
prove that n itself is divisible by 6.
-/
def divisible_by_6 (n : Nat) : Prop :=
  ∃ k : Nat, n = 6 * k

theorem sum_of_squares_of_primes_divisible_by_6_implies_n_divisible_by_6
  (n : Nat) (primes : Fin n → ℕ)
  (h_primes : ∀ i, primes i > 5)
  (h_sum_divisible : (∑ i : Fin n, (primes i)^2) % 6 = 0) : divisible_by_6 n :=
by
  sorry

end sum_of_squares_of_primes_divisible_by_6_implies_n_divisible_by_6_l707_707805


namespace model_N_completion_time_l707_707879

variable (T : ℕ)

def model_M_time : ℕ := 36
def number_of_M_computers : ℕ := 12
def number_of_N_computers := number_of_M_computers -- given that they are the same.

-- Statement of the problem: Given the conditions, prove T = 18
theorem model_N_completion_time :
  (number_of_M_computers : ℝ) * (1 / model_M_time) + (number_of_N_computers : ℝ) * (1 / T) = 1 →
  T = 18 :=
by
  sorry

end model_N_completion_time_l707_707879


namespace roger_earned_correct_amount_l707_707069

def small_lawn_rate : ℕ := 9
def medium_lawn_rate : ℕ := 12
def large_lawn_rate : ℕ := 15

def initial_small_lawns : ℕ := 5
def initial_medium_lawns : ℕ := 4
def initial_large_lawns : ℕ := 5

def forgot_small_lawns : ℕ := 2
def forgot_medium_lawns : ℕ := 3
def forgot_large_lawns : ℕ := 3

def actual_small_lawns := initial_small_lawns - forgot_small_lawns
def actual_medium_lawns := initial_medium_lawns - forgot_medium_lawns
def actual_large_lawns := initial_large_lawns - forgot_large_lawns

def money_earned_small := actual_small_lawns * small_lawn_rate
def money_earned_medium := actual_medium_lawns * medium_lawn_rate
def money_earned_large := actual_large_lawns * large_lawn_rate

def total_money_earned := money_earned_small + money_earned_medium + money_earned_large

theorem roger_earned_correct_amount : total_money_earned = 69 := by
  sorry

end roger_earned_correct_amount_l707_707069


namespace train_pass_bridge_time_l707_707139

def train_length : ℕ := 360
def bridge_length : ℕ := 140
def train_speed_kmh : ℕ := 60
def total_length : ℕ := train_length + bridge_length
def conversion_factor : ℚ := 1000 / 3600
def train_speed_ms : ℚ := train_speed_kmh * conversion_factor
def expected_time : ℚ := total_length / train_speed_ms

theorem train_pass_bridge_time : expected_time ≈ 30 :=
by
  have h1 : total_length = 500 := rfl
  have h2 : train_speed_ms ≈ 16.67 := 
  by
    dsimp [train_speed_ms, conversion_factor, train_speed_kmh]
    norm_num
  dsimp [expected_time, total_length, train_speed_ms, h1]
  norm_num
  sorry

end train_pass_bridge_time_l707_707139


namespace coefficients_ratio_of_polynomial_expansion_l707_707206

theorem coefficients_ratio_of_polynomial_expansion :
  let a : Fin 6 → ℤ := λ i, (expand (n := 5) (2 : ℤ) (-1 : ℤ) ![a_0, a_1, a_2, a_3, a_4, a_5]) i in
  (a 0 + a 2 + a 4) / (a 1 + a 3 + a 5) = -122 / 121 :=
by
  sorry

end coefficients_ratio_of_polynomial_expansion_l707_707206


namespace unique_digit_10D4_count_unique_digit_10D4_l707_707603

theorem unique_digit_10D4 (D : ℕ) (hD : D < 10) : 
  (5 + D) % 3 = 0 ∧ (10 * D + 4) % 4 = 0 ↔ D = 4 :=
by
  sorry

theorem count_unique_digit_10D4 :
  ∃! D, (D < 10 ∧ (5 + D) % 3 = 0 ∧ (10 * D + 4) % 4 = 0) :=
by
  use 4
  simp [unique_digit_10D4]
  sorry

end unique_digit_10D4_count_unique_digit_10D4_l707_707603


namespace tom_total_distance_l707_707468

def tomTimeSwimming : ℝ := 2
def tomSpeedSwimming : ℝ := 2
def tomTimeRunning : ℝ := (1 / 2) * tomTimeSwimming
def tomSpeedRunning : ℝ := 4 * tomSpeedSwimming

theorem tom_total_distance : 
  let distanceSwimming := tomSpeedSwimming * tomTimeSwimming
  let distanceRunning := tomSpeedRunning * tomTimeRunning
  distanceSwimming + distanceRunning = 12 := by
  sorry

end tom_total_distance_l707_707468


namespace fruit_salad_cost_3_l707_707808

def cost_per_fruit_salad (num_people sodas_per_person soda_cost sandwich_cost num_snacks snack_cost total_cost : ℕ) : ℕ :=
  let total_soda_cost := num_people * sodas_per_person * soda_cost
  let total_sandwich_cost := num_people * sandwich_cost
  let total_snack_cost := num_snacks * snack_cost
  let total_known_cost := total_soda_cost + total_sandwich_cost + total_snack_cost
  let total_fruit_salad_cost := total_cost - total_known_cost
  total_fruit_salad_cost / num_people

theorem fruit_salad_cost_3 :
  cost_per_fruit_salad 4 2 2 5 3 4 60 = 3 :=
by
  sorry

end fruit_salad_cost_3_l707_707808


namespace speed_with_current_l707_707885

-- Define the constants
def speed_of_current : ℝ := 2.5
def speed_against_current : ℝ := 20

-- Define the man's speed in still water
axiom speed_in_still_water : ℝ
axiom speed_against_current_eq : speed_in_still_water - speed_of_current = speed_against_current

-- The statement we need to prove
theorem speed_with_current : speed_in_still_water + speed_of_current = 25 := sorry

end speed_with_current_l707_707885


namespace sum_of_200th_row_l707_707203

-- Definitions related to the conditions of the problem
def triangular_array_interior (prev_row : List ℕ) : List ℕ :=
  List.zipWith (· + ·) (prev_row ++ [0]) ([0] ++ prev_row)

def sum_of_elements (lst : List ℕ) : ℕ :=
  lst.foldr (· + ·) 0

def sum_of_row (n : ℕ) : ℕ :=
  Nat.recOn n 0 (λ n sum_prev, 
    let row := List.range (n + 1)
    let next_row := triangular_array_interior row
    sum_of_elements next_row)

-- The theorem to prove that the sum of the numbers in the 200th row is 2^200 - 2
theorem sum_of_200th_row :
  sum_of_row 200 = 2^200 - 2 :=
sorry

end sum_of_200th_row_l707_707203


namespace nadine_white_pebbles_l707_707391

variable (W R : ℝ)

theorem nadine_white_pebbles :
  (R = 1/2 * W) →
  (W + R = 30) →
  W = 20 :=
by
  sorry

end nadine_white_pebbles_l707_707391


namespace four_digit_numbers_starting_with_1_l707_707796

theorem four_digit_numbers_starting_with_1 
: ∃ n : ℕ, (n = 234) ∧ 
  (∀ (x y z : ℕ), 
    (x ≠ y → x ≠ z → y ≠ z → -- ensuring these constraints
    x ≠ 1 → y ≠ 1 → z = 1 → -- exactly two identical digits which include 1
    (x * 1000 + y * 100 + z * 10 + 1) / 1000 = 1 ∨ (x * 1000 + z * 100 + y * 10 + 1) / 1000 = 1) ∨ 
    (∃ (x y : ℕ),  
    (x ≠ y → x ≠ 1 → y = 1 → 
    (x * 110 + y * 10 + 1) + (x * 11 + y * 10 + 1) + (x * 100 + y * 10 + 1) + (x * 110 + 1) = n))) := sorry

end four_digit_numbers_starting_with_1_l707_707796


namespace find_a_l707_707289

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1
def g : ℝ → ℝ := λ x, -1/4 * x

theorem find_a (a : ℝ) (h_tangent_perpendicular : (deriv (f a) 1 + 4 * deriv g 1 = 0)) : a = 1 := 
sorry

end find_a_l707_707289


namespace problem_a2024_l707_707096

def sequence_a : ℕ → ℚ
| 0     := 1
| (n+1) := 1 / (2 * ⌊sequence_a n⌋ - sequence_a n + 1)

theorem problem_a2024 :
  sequence_a 2024 = 1 / 2 :=
sorry

end problem_a2024_l707_707096


namespace axis_of_symmetry_l707_707674

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = f (4 - x)) : ∀ y, f 2 = y ↔ f 2 = y := 
by
  sorry

end axis_of_symmetry_l707_707674


namespace correct_conclusions_l707_707621

variable {R : Type} [LinearOrder R] {f : R → R}

def is_odd_function (f : R → R) := ∀ x, f (-x) = -f x

def satisfies_equation (f : R → R) := ∀ x, f (x + 2) + 2 * f (-x) = 0

theorem correct_conclusions (h_odd : is_odd_function f) (h_eqn : satisfies_equation f) :
  (f 2 = 0) ∧ (∀ x, f (x + 2) = 2 * f x) ∧ (∀ x, f (x + 4) = 4 * f x) ∧ ¬ (∀ x, f (x + 6) = 6 * f x) :=
by
  sorry

end correct_conclusions_l707_707621


namespace largest_number_with_9_factors_l707_707049

def count_factors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ x => n % x = 0).length

theorem largest_number_with_9_factors : ∀ n < 150, count_factors n = 9 → n = 100 := by
  assume n
  assume h2 : n < 150
  assume h3 : count_factors n = 9
  sorry

end largest_number_with_9_factors_l707_707049


namespace macy_miles_left_l707_707740

theorem macy_miles_left (goal : ℕ) (daily_miles : ℕ) (days_run : ℕ) 
  (H1 : goal = 24) 
  (H2 : daily_miles = 3) 
  (H3 : days_run = 6) 
  : goal - daily_miles * days_run = 6 := 
by 
  sorry

end macy_miles_left_l707_707740


namespace A_is_correct_B_is_correct_m_range_correct_l707_707650

def f (x a : ℝ) := x ^ 2 - 2 * a * x + 1

theorem A_is_correct (a : ℝ) : 
  (-1 < a ∧ a < 1) ↔ (∀ x : ℝ, f x a ≠ 0) := 
sorry

theorem B_is_correct (a m : ℝ) : 
  (m < a ∧ a < m + 3) ↔ (∃ x : ℝ, ∃ y : ℝ, x ∈ set.Ioo m (m+3) ∧ f x a = f y a) := 
sorry

theorem m_range_correct (m : ℝ) : 
  (∀ a : ℝ, (-1 < a ∧ a < 1) → (m < a ∧ a < m + 3)) → (-2 ≤ m ∧ m ≤ -1) :=
sorry

end A_is_correct_B_is_correct_m_range_correct_l707_707650


namespace relationship_between_a_b_c_l707_707312

noncomputable def a : ℝ := 5 ^ 0.3
noncomputable def b : ℝ := 0.3 ^ 5
noncomputable def c : ℝ := Real.logBase 5 0.3

-- Prove the relationship c < b < a
theorem relationship_between_a_b_c : c < b ∧ b < a :=
by
  -- Sorry as placeholder for the actual proof
  sorry

end relationship_between_a_b_c_l707_707312


namespace distance_between_points_l707_707227

-- Define the coordinates of the points
def pointA : ℝ × ℝ := (0, 15)
def pointB : ℝ × ℝ := (8, 0)

-- Define the Euclidean distance function
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- The theorem to prove the distance is 17
theorem distance_between_points : euclidean_distance pointA pointB = 17 := by
  sorry

end distance_between_points_l707_707227


namespace domain_of_function_l707_707963

theorem domain_of_function : 
  {x : ℝ | -10 * x^2 - 11 * x + 6 ≥ 0} = set.Icc (-3/2 : ℝ) (2/5 : ℝ) := 
by
  sorry

end domain_of_function_l707_707963


namespace non_square_300th_term_l707_707497

theorem non_square_300th_term (N : ℕ) (hN : N = 300) : 
  ∃ x : ℕ, (∀ (t : ℕ), 0 < t ∧ t ≤ x → ¬ (∃ k : ℕ, t = k^2)) ∧ 
           (∑ t in finset.range (N + 17), if ∃ k : ℕ, t = k^2 then 0 else 1 = N) ∧ 
           x = 317 := by
  sorry

end non_square_300th_term_l707_707497


namespace probability_zack_and_andrew_same_team_is_correct_l707_707114

noncomputable def probability_zack_and_andrew_same_team
    (players : Finset ℕ)
    (team1 team2 team3 : Finset ℕ)
    (Z M A : ℕ)
    (h1 : Z ≠ M)
    (h2 : M ≠ A)
    (h3 : team1.card = 9)
    (h4 : team2.card = 9)
    (h5 : team3.card = 9)
    (h6 : (team1 ∪ team2 ∪ team3) = players)
    (h7 : team1 ∩ team2 = ∅)
    (h8 : team1 ∩ team3 = ∅)
    (h9 : team2 ∩ team3 = ∅)
    (h10 : Z ∈ team1)
    (h11 : M ∈ team2) :
    ℚ :=
  let count_favorable := (team1.card - 1) in
  let total_possible := (team1.card + team3.card - 1) in
  (count_favorable : ℚ) / total_possible

theorem probability_zack_and_andrew_same_team_is_correct
    (players : Finset ℕ)
    (team1 team2 team3 : Finset ℕ)
    (Z M A : ℕ)
    (h1 : Z ≠ M)
    (h2 : M ≠ A)
    (h3 : team1.card = 9)
    (h4 : team2.card = 9)
    (h5 : team3.card = 9)
    (h6 : (team1 ∪ team2 ∪ team3) = players)
    (h7 : team1 ∩ team2 = ∅)
    (h8 : team1 ∩ team3 = ∅)
    (h9 : team2 ∩ team3 = ∅)
    (h10 : Z ∈ team1)
    (h11 : M ∈ team2) :
    probability_zack_and_andrew_same_team players team1 team2 team3 Z M A h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 = 8 / 17 :=
by
  sorry

end probability_zack_and_andrew_same_team_is_correct_l707_707114


namespace all_points_lie_on_line_l707_707604

theorem all_points_lie_on_line:
  ∀ (s : ℝ), s ≠ 0 → ∀ (x y : ℝ),
  x = (2 * s + 3) / s → y = (2 * s - 3) / s → x + y = 4 :=
by
  intros s hs x y hx hy
  sorry

end all_points_lie_on_line_l707_707604


namespace sum_of_positive_integer_values_l707_707125

theorem sum_of_positive_integer_values :
  (∀ (n : ℕ), n > 0 ∧ (30 % n = 0) → n ∣ 30) →
  (∑ n in (Finset.filter (λ n, n > 0 ∧ 30 % n = 0) (Finset.range 31)), n) = 72 :=
by
  -- sorry serves as a placeholder for the actual proof
  sorry

end sum_of_positive_integer_values_l707_707125


namespace box_dimensions_l707_707856

theorem box_dimensions (a b c : ℕ) 
  (h1 : a + c = 17) 
  (h2 : a + b = 13) 
  (h3 : b + c = 20) : 
  {a, b, c} = {5, 8, 12} := 
by
  sorry

end box_dimensions_l707_707856


namespace tangent_line_y_intercept_l707_707787

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

theorem tangent_line_y_intercept (a : ℝ) (h : 3 * (1:ℝ)^2 - a = 1) :
  (∃ (m b : ℝ), ∀ (x : ℝ), m = 1 ∧ y = x - 2 → y = m * x + b) := 
 by
  sorry

end tangent_line_y_intercept_l707_707787


namespace syllogism_correct_order_l707_707839

-- Definitions from the problem conditions
def is_trigonometric (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = real.sin x

def is_periodic (f : ℝ → ℝ) : Prop :=
  ∃ p > 0, ∀ x, f (x + p) = f x

-- Statement definitions according to the problem
def statement_1 : Prop := is_trigonometric (real.sin)
def statement_2 : Prop := ∀ f, is_trigonometric f → is_periodic f
def statement_3 : Prop := is_periodic (real.sin)

-- The syllogism proof setup
theorem syllogism_correct_order : statement_2 ∧ statement_1 → statement_3 :=
by
  sorry

end syllogism_correct_order_l707_707839


namespace area_of_triangle_ABF_eq_l707_707341

variables (A B C D E F : Type) [AffineSpace A] [MetricSpace A] [InnerProductSpace ℝ A]
variables (a α : ℝ) (S : Set (Set A))
variable (P : A → A → A → Set A)
variable [IsParallelogram A B C D]
variable [IsMidpoint E A D]
variable [IsFootOfPerpendicular F B (LineThrough C E)]

noncomputable def area_triangle_ABF : ℝ :=
  1 / 2 * a^2 * sin(α)

theorem area_of_triangle_ABF_eq : 
  ∀ (h_parallel : IsParallelogram A B C D) 
    (h_midpoint : IsMidpoint E A D)
    (h_perpendicular : IsFootOfPerpendicular F B (LineThrough C E)),
  area_triangle_ABF A B F a α = 1 / 2 * a^2 * sin(α) :=
sorry

end area_of_triangle_ABF_eq_l707_707341


namespace base7_number_divisible_by_29_l707_707581

theorem base7_number_divisible_by_29 (y : ℕ) (h : y < 7) :
    let n := 2 * 7^3 + y * 7^2 + 6 * 7 + 3 in
    n % 29 = 0 ↔ y = 6 :=
by
  sorry

end base7_number_divisible_by_29_l707_707581


namespace correct_conclusions_l707_707601

def rounding (x : ℝ) : ℤ := 
  if x - 0.5 ≤ int.floor x + 0.5 then int.floor x else int.floor x + 1

lemma rounding_1 : rounding 1.499 = 1 := 
by sorry

lemma rounding_2 (x : ℝ) (h : rounding (0.5 * x - 3) = 3) : 11 ≤ x ∧ x < 13 := 
by sorry

lemma rounding_3 (x y : ℝ) : rounding (x + y) ≠ rounding x + rounding y := 
by sorry

lemma rounding_4 (x : ℝ) (m : ℕ) (h : 0 ≤ x) : rounding (m + 2017 * x) = m + rounding (2017 * x) := 
by sorry

theorem correct_conclusions : rounding_1 ∧ rounding_2 ∧ rounding_4 :=
by sorry

end correct_conclusions_l707_707601


namespace batsman_average_increase_l707_707155

theorem batsman_average_increase
  (A : ℕ)
  (h_average_after_17th : (16 * A + 90) / 17 = 42) :
  42 - A = 3 :=
by
  sorry

end batsman_average_increase_l707_707155


namespace fish_weight_l707_707790

variable (Γ T : ℝ)
variable (X : ℝ := 1)  -- The tail's weight is given to be 1 kg

theorem fish_weight : 
  (Γ = X + T / 2) → 
  (T = Γ + X) →
  (Γ + T + X = 8) :=
by
  intros h1 h2
  sorry

end fish_weight_l707_707790


namespace sum_of_m_values_l707_707903

-- Coordinates of the vertices of the triangle
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 2)
def C (m : ℝ) : ℝ × ℝ := (8 * m, 0)

-- Define a function to check if the line divides the triangle into two equal areas
def divides_triangle_equal_areas (m : ℝ) : Prop :=
  let D := (4 * m + 1, m * (4 * m + 1))
  4 * m ^ 2 + m - 1 = 0

-- The sum of all possible values of m
theorem sum_of_m_values : 
  (∑ m in (finset.filter divides_triangle_equal_areas ({x : ℝ | divides_triangle_equal_areas x}.to_finset)), id m) = - 1 / 4 :=
sorry -- Proof can be filled in separately

end sum_of_m_values_l707_707903


namespace distance_from_center_to_plane_tangent_of_acute_angle_l707_707752

-- Definitions based on the given conditions
def sphere_radius : ℝ := 13
def point_distances := (AB : ℝ, BC : ℝ, CA : ℝ)
def distances : point_distances := (6, 8, 10)

-- Lean 4 statement for the proof problem
theorem distance_from_center_to_plane (sphere_radius : ℝ) (distances : point_distances) 
    (h1 : distances.AB = 6) (h2 : distances.BC = 8) (h3 : distances.CA = 10) : 
    distance_center_to_plane ABC sphere_radius distances = 12 := sorry

theorem tangent_of_acute_angle (sphere_radius : ℝ) (distances : point_distances) 
    (h1 : distances.AB = 6) (h2 : distances.BC = 8) (h3 : distances.CA = 10)
    (d_center_to_plane : distance_center_to_plane ABC sphere_radius distances = 12) : 
    tangent_acute_angle_great_circle_plane ABC AB sphere_radius distances = 3 := sorry

end distance_from_center_to_plane_tangent_of_acute_angle_l707_707752


namespace interference_facts_proof_l707_707906

-- Definitions corresponding to the given facts
def light_fact1 : Prop := ¬(fact_related_to_interference "Transmitting signals using optical fibers")
def light_fact2 : Prop := fact_related_to_interference "Using a transparent standard plate and monochromatic light to check the flatness of a surface"
def light_fact3 : Prop := ¬(fact_related_to_interference "A beam of white light passing through a prism forms a colored light band")
def light_fact4 : Prop := fact_related_to_interference "A film of oil on water showing colors"

-- Proving that facts 2 and 4 are related to light interference
theorem interference_facts_proof : light_fact2 ∧ light_fact4 :=
by {
  -- This placeholder represents the logical proof process for the theorem.
  sorry
}

end interference_facts_proof_l707_707906


namespace balls_into_boxes_l707_707304

theorem balls_into_boxes (n k : ℕ) (h : 4 ≤ n ∧ n ≤ k) : 
  let binom (a b : ℕ) := Nat.choose a b in
  (∑ i in Finset.range 4, binom n i * binom (k - 1) (n - i - 1)) = 
  -- Original problem's result: the number of ways 
  -- to place k indistinguishable balls into n distinguishable boxes 
  -- with at most 3 empty boxes.
  sorry

end balls_into_boxes_l707_707304


namespace monic_quartic_polynomial_l707_707953

theorem monic_quartic_polynomial :
  ∃ (p : Polynomial ℚ), p.leadingCoeff = 1 ∧
  (p.eval (3 + Real.sqrt 5) = 0) ∧
  (p.eval (2 - Real.sqrt 6) = 0) :=
by {
  let p := Polynomial.X^4 - 10*Polynomial.X^3 + 20*Polynomial.X^2 + 4*Polynomial.X - 8,
  use p,
  split,
  { simp [p], }, -- Check that the leading coefficient is 1
  split,
  { -- Check that (3 + sqrt(5)) is a root
    sorry
  },
  { -- Check that (2 - sqrt(6)) is a root
    sorry
  }
}

end monic_quartic_polynomial_l707_707953


namespace total_students_in_first_year_l707_707179

theorem total_students_in_first_year (N : ℕ)
  (survey1 : ℕ) (survey2 : ℕ) (overlap : ℕ) 
  (condition1 : survey1 = 80)
  (condition2 : survey2 = 100)
  (condition3 : overlap = 20) :
  N = 400 :=
by
  -- Definitions and conditions for the Lean proof
  have prob1 := survey1 / N
  have prob2 := survey2 / N
  have prob_both := overlap / N
  have product_probs := prob1 * prob2
  -- Establish the relationship between the probabilities and solve for N
  have h : product_probs = prob_both / N := sorry
  have N_squared := (survey1 * survey2) / overlap := by sorry
  have N := sqrt N_squared := by sorry
  exact N

end total_students_in_first_year_l707_707179


namespace sum_of_abs_roots_squared_l707_707950

noncomputable def f (x : ℝ) : ℝ := sqrt 19 + 91 / x

theorem sum_of_abs_roots_squared : 
  let A := |(sqrt 19 + sqrt 383) / 2| + |(sqrt 383 - sqrt 19) / 2| 
  in A^2 = 383 := 

by {
  -- Define the specific equation to solve.
  have h : ∀ x, x = f(x)
    := λ x, by sorry,

  -- Define the roots of the quadratic equation.
  let x1 := (sqrt 19 + sqrt 383) / 2,
      x2 := (sqrt 383 - sqrt 19) / 2,

  -- Define the sum of the absolute values of the roots.
  let A := abs x1 + abs x2,

  -- Show that the sum squared is 383.
  show A^2 = 383,
  by sorry
}

end sum_of_abs_roots_squared_l707_707950


namespace minimum_value_of_f_l707_707979

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 + 8 * x + 13) / (6 * (1 + Real.exp (-x)))

theorem minimum_value_of_f : ∀ x : ℝ, 0 ≤ x → f x ≥ f 0 :=
by
  intro x hx
  unfold f
  admit

end minimum_value_of_f_l707_707979


namespace smallest_possible_value_of_c_l707_707067

theorem smallest_possible_value_of_c (b c : ℝ) (h1 : 1 < b) (h2 : b < c)
    (h3 : ¬∃ (u v w : ℝ), u = 1 ∧ v = b ∧ w = c ∧ u + v > w ∧ u + w > v ∧ v + w > u)
    (h4 : ¬∃ (x y z : ℝ), x = 1 ∧ y = 1/b ∧ z = 1/c ∧ x + y > z ∧ x + z > y ∧ y + z > x) :
    c = (5 + Real.sqrt 5) / 2 :=
by
  sorry

end smallest_possible_value_of_c_l707_707067


namespace discs_contain_equal_minutes_l707_707543

theorem discs_contain_equal_minutes (total_time discs_capacity : ℕ) 
  (h1 : total_time = 520) (h2 : discs_capacity = 65) :
  ∃ discs_needed : ℕ, discs_needed = total_time / discs_capacity ∧ 
  ∀ (k : ℕ), k = total_time / discs_needed → k = 65 :=
by
  sorry

end discs_contain_equal_minutes_l707_707543


namespace expansion_term_no_x_210_l707_707608

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem expansion_term_no_x_210 : 
  let n := 10
  let general_term := λ (r : ℕ), binomial n r * (x ^ (3 * (n - r))) * ((1 / x ^ 2) ^ r)
  let r := 6
  let term_does_not_contain_x := (3 * (n - r) - 2 * r = 0)
  term_does_not_contain_x → binomial n r = 210 :=
  sorry

end expansion_term_no_x_210_l707_707608


namespace calculation_of_product_l707_707527

theorem calculation_of_product : (0.09)^3 * 0.0007 = 0.0000005103 := 
by
  sorry

end calculation_of_product_l707_707527


namespace triangle_DEF_area_l707_707344

theorem triangle_DEF_area (DE height : ℝ) (hDE : DE = 12) (hHeight : height = 15) : 
  (1/2) * DE * height = 90 :=
by
  rw [hDE, hHeight]
  norm_num

end triangle_DEF_area_l707_707344


namespace shirt_cost_l707_707607

def george_initial_money : ℕ := 100
def total_spent_on_clothes (initial_money remaining_money : ℕ) : ℕ := initial_money - remaining_money
def socks_cost : ℕ := 11
def remaining_money_after_purchase : ℕ := 65

theorem shirt_cost
  (initial_money : ℕ)
  (remaining_money : ℕ)
  (total_spent : ℕ)
  (socks_cost : ℕ)
  (remaining_money_after_purchase : ℕ) :
  initial_money = 100 →
  remaining_money = 65 →
  total_spent = initial_money - remaining_money →
  total_spent = 35 →
  socks_cost = 11 →
  remaining_money_after_purchase = remaining_money →
  (total_spent - socks_cost = 24) :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h4] at *
  exact sorry

end shirt_cost_l707_707607


namespace count_three_digit_numbers_with_digit_sum_24_l707_707303

-- Define the conditions:
def isThreeDigitNumber (a b c : ℕ) : Prop :=
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (100 * a + 10 * b + c ≥ 100)

def digitSumEquals24 (a b c : ℕ) : Prop :=
  a + b + c = 24

-- State the theorem:
theorem count_three_digit_numbers_with_digit_sum_24 :
  (∃ (count : ℕ), count = 10 ∧ 
   ∀ (a b c : ℕ), isThreeDigitNumber a b c ∧ digitSumEquals24 a b c → (count = 10)) :=
sorry

end count_three_digit_numbers_with_digit_sum_24_l707_707303


namespace union_of_M_and_Q_is_correct_l707_707263

-- Given sets M and Q
def M : Set ℕ := {0, 2, 4, 6}
def Q : Set ℕ := {0, 1, 3, 5}

-- Statement to prove
theorem union_of_M_and_Q_is_correct : M ∪ Q = {0, 1, 2, 3, 4, 5, 6} :=
by
  sorry

end union_of_M_and_Q_is_correct_l707_707263


namespace domain_of_g_domain_of_g_l707_707962

theorem domain_of_g :
  (∀ x, g(x) = sqrt(1 - sqrt(3 - sqrt(4 - x))) → -5 ≤ x ∧ x ≤ 0) :=
sorry

noncomputable def g (x : ℝ) : ℝ :=
  sqrt (1 - sqrt (3 - sqrt (4 - x)))

# Reduce the problem to inequalities for usability in Lean
def condition1 (x : ℝ) : Prop := 1 - sqrt (3 - sqrt (4 - x)) ≥ 0
def condition2 (x : ℝ) : Prop := sqrt (3 - sqrt (4 - x)) ≤ 1
def condition3 (x : ℝ) : Prop := 3 - sqrt (4 - x) ≥ 0
def condition4 (x : ℝ) : Prop := sqrt (4 - x) ≤ 3

theorem domain_of_g :
  (∀ x, condition1 x ∧ condition2 x ∧ condition3 x ∧ condition4 x ↔ (-5 ≤ x ∧ x ≤ 0)) :=
sorry

end domain_of_g_domain_of_g_l707_707962


namespace tradesman_gain_on_outlay_l707_707138

-- Define the percentage defrauded and the percentage gain in both buying and selling
def defraud_percent := 20
def original_value := 100
def buying_price := original_value * (1 - (defraud_percent / 100))
def selling_price := original_value * (1 + (defraud_percent / 100))
def gain := selling_price - buying_price
def gain_percent := (gain / buying_price) * 100

theorem tradesman_gain_on_outlay :
  gain_percent = 50 := 
sorry

end tradesman_gain_on_outlay_l707_707138


namespace no_intersection_radius_l707_707255

theorem no_intersection_radius 
  (M : set (ℝ × ℝ))
  (hM : ∀ (x y : ℝ), (x, y) ∈ M ↔ sin (3 * x + 4 * y) = sin (3 * x) + sin (4 * y))
  (R : ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ M → (x^2 + y^2) > R^2) ↔ R < π / 6 :=
sorry

end no_intersection_radius_l707_707255


namespace f_symmetric_about_1_0_l707_707789

def f (x : ℝ) : ℝ := abs (floor x) - abs (floor (2 - x))

theorem f_symmetric_about_1_0 : ∀ x : ℝ, f (1 + x) = f (1 - x) := by
  sorry

end f_symmetric_about_1_0_l707_707789


namespace determinant_of_A_l707_707921

def matrix_A : Matrix (Fin 3) (Fin 3) ℤ := 
![![3, 1, -2], 
  ![8, 5, -4], 
  ![3, 3, 6]]

theorem determinant_of_A : matrix.det matrix_A = 48 := by
  sorry

end determinant_of_A_l707_707921


namespace cartons_less_received_theorem_l707_707383

noncomputable def numberOfCartonsLessReceived 
  (total_cartons : ℕ)
  (jars_per_carton : ℕ)
  (cartons_received : ℕ → ℕ)
  (damaged_jars_in5cartons : ℕ)
  (totally_damaged_cartons : ℕ)
  (good_jars_for_sale : ℕ) : ℕ :=
let total_jars := total_cartons * jars_per_carton in
let jars_after_total_damage := total_jars - (totally_damaged_cartons * jars_per_carton) in
let jars_after_all_damages := jars_after_total_damage - (damaged_jars_in5cartons * 5) in
total_jars - good_jars_for_sale

theorem cartons_less_received_theorem 
  (habitual_cartons : ℕ)
  (cartons_with_20_jars_each : ℕ)
  (default_cartons : ℕ)
  (damaged_jars_in_5_cartons : ℕ)
  (a_totally_damaged_carton : ℕ)
  (saleable_jars : ℕ)
  (correct_answer : ℕ) :
  correct_answer = numberOfCartonsLessReceived 50 20 default_cartons damaged_jars_in_5_cartons a_totally_damaged_carton saleable_jars :=
by
  have h1 : habitual_cartons = 50,
  have h2 : cartons_with_20_jars_each = 20,
  have h3 : default_cartons = 20,
  have h4 : damaged_jars_in_5_cartons = 3,
  have h5 : a_totally_damaged_carton = 1,
  have h6 : saleable_jars = 565,
  have h7 : correct_answer = 20,
  sorry

end cartons_less_received_theorem_l707_707383


namespace perimeter_new_square_is_approx_l707_707797

noncomputable def perimeter_of_new_square : ℝ :=
  let side1 := 24 / 4 in
  let side2 := 32 / 4 in
  let side3 := 40 / 4 in
  let area1 := side1 ^ 2 in
  let area2 := side2 ^ 2 in
  let area3 := side3 ^ 2 in
  let total_area := area1 + area2 + area3 in
  let new_side := Real.sqrt total_area in
  4 * new_side

theorem perimeter_new_square_is_approx : 
|perimeter_of_new_square - 56.56| < 0.01 :=
by sorry

end perimeter_new_square_is_approx_l707_707797


namespace cos_sum_series_l707_707283

theorem cos_sum_series : 
  (∑' n : ℤ, if (n % 2 = 1 ∨ n % 2 = -1) then (1 : ℝ) / (n : ℝ)^2 else 0) = (π^2) / 8 := by
  sorry

end cos_sum_series_l707_707283


namespace geometric_sequence_general_formula_sum_of_arithmetic_sequence_l707_707997

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧ a 4 = 24

noncomputable def arithmetic_sequence_condition (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 2 = a 2 ∧ b 9 = a 5

theorem geometric_sequence_general_formula (a : ℕ → ℝ) (n : ℕ) (H : geometric_sequence a) :
  a n = 3 * 2 ^ (n - 1) :=
sorry

theorem sum_of_arithmetic_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ)
  (Hgeom : geometric_sequence a)
  (Harith : arithmetic_sequence_condition b a) :
  (∑ k in finset.range n, b (k + 1)) = 3 * n^2 - 3 * n :=
sorry

end geometric_sequence_general_formula_sum_of_arithmetic_sequence_l707_707997


namespace range_of_fraction_l707_707522

open Real

noncomputable def quadratic_eq (a b : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, 
    (0 < x1 ∧ x1 < 1) ∧ 
    (1 < x2 ∧ x2 < 2) ∧ 
    x1 + x2 = -a ∧ 
    x1 * x2 = 2 * b - 2 ∧ 
    ((∀ x : ℝ, x*x + a*x + 2*b - 2 = 0) → (x = x1 ∨ x = x2))

theorem range_of_fraction (a b : ℝ) (h : quadratic_eq a b) :
  1 / 2 < (b - 4) / (a - 1) ∧ (b - 4) / (a - 1) < 3 / 2 :=
sorry

end range_of_fraction_l707_707522


namespace simplify_expression_l707_707193

theorem simplify_expression (a : ℝ) (h : a ≠ -1) : a - 1 + 1 / (a + 1) = a^2 / (a + 1) :=
  sorry

end simplify_expression_l707_707193


namespace log_comparison_l707_707573

theorem log_comparison (a : ℝ) (h : a > 1) : real.log a / real.log (a - 1) > real.log (a + 1) / real.log a :=
by sorry

end log_comparison_l707_707573


namespace term_omit_perfect_squares_300_l707_707478

theorem term_omit_perfect_squares_300 (n : ℕ) (hn : n = 300) : 
  ∃ k : ℕ, k = 317 ∧ (∀ m : ℕ, (m < k → m * m ≠ k)) := 
sorry

end term_omit_perfect_squares_300_l707_707478


namespace express_x_in_terms_of_y_l707_707284

def f (t : ℝ) : ℝ := t / (1 - t^2)

theorem express_x_in_terms_of_y (x y : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : y = f x) :
  x = y / (1 + y^2) :=
sorry

end express_x_in_terms_of_y_l707_707284


namespace evaluate_ceil_expression_l707_707946

theorem evaluate_ceil_expression :
  (⌈Real.sqrt (16 / 9)⌉ + ⌈(16 / 9)⌉ + ⌈(16 / 9)^2⌉) = 8 :=
by
  -- note: proof steps are not needed, so we skip the proof using sorry
  sorry

end evaluate_ceil_expression_l707_707946


namespace number_of_players_l707_707012

/-- Jane bought 600 minnows, each prize has 3 minnows, 15% of the players win a prize, 
and 240 minnows are left over. To find the total number of players -/
theorem number_of_players (total_minnows left_over_minnows minnows_per_prize prizes_win_percent : ℕ) 
(h1 : total_minnows = 600) 
(h2 : minnows_per_prize = 3)
(h3 : prizes_win_percent * 100 = 15)
(h4 : left_over_minnows = 240) : 
total_minnows - left_over_minnows = 360 → 
  360 / minnows_per_prize = 120 → 
  (prizes_win_percent * 100 / 100) * P = 120 → 
  P = 800 := 
by 
  sorry

end number_of_players_l707_707012


namespace total_amount_paid_after_discount_l707_707544

-- Define the given conditions
def marked_price_per_article : ℝ := 10
def discount_percentage : ℝ := 0.60
def number_of_articles : ℕ := 2

-- Proving the total amount paid
theorem total_amount_paid_after_discount : 
  (marked_price_per_article * number_of_articles) * (1 - discount_percentage) = 8 := by
  sorry

end total_amount_paid_after_discount_l707_707544


namespace triangle_has_equal_angles_of_enlarged_sides_l707_707761

theorem triangle_has_equal_angles_of_enlarged_sides 
  {A B C A' B' C' : Type} 
  (triangleABC : triangle A B C) 
  (triangleA'B'C' : triangle A' B' C') 
  (n : ℝ) 
  (side_cond : ∀ {a b c : ℝ}, 
              ((side_length triangleABC A B = n * side_length triangleA'B'C' A' B') 
              ∧ (side_length triangleABC B C = n * side_length triangleA'B'C' B' C') 
              ∧ (side_length triangleABC C A = n * side_length triangleA'B'C' C' A')))
  : 
  (angle triangleABC A B C = angle triangleA'B'C' A' B' C') 
  ∧ (angle triangleABC B C A = angle triangleA'B'C' B' C' A') 
  ∧ (angle triangleABC C A B = angle triangleA'B'C' C' A' B') :=
sorry

end triangle_has_equal_angles_of_enlarged_sides_l707_707761


namespace gift_cost_l707_707595

theorem gift_cost (C F : ℕ) (hF : F = 15) (h_eq : C / (F - 4) = C / F + 12) : C = 495 :=
by
  -- Using the conditions given, we need to show that C computes to 495.
  -- Details are skipped using sorry.
  sorry

end gift_cost_l707_707595


namespace curve_is_semicircle_l707_707292

-- Define the parametric equations and the conditions
def parametric_x (θ : ℝ) : ℝ := 2 * Real.cos θ
def parametric_y (θ : ℝ) : ℝ := 1 + 2 * Real.sin θ

-- The range of θ is [-π/2, π/2]
def θ_range (θ : ℝ) : Prop :=
  -Real.pi / 2 ≤ θ ∧ θ ≤ Real.pi / 2

-- Define the final statement that needs to be proved: the curve represents a semicircle
theorem curve_is_semicircle (θ : ℝ) (hθ : θ_range θ) :
  ∃ x y, (x = parametric_x θ) ∧ (y = parametric_y θ) ∧
  (x^2 + (y - 1)^2 = 4) ∧ (0 ≤ x) :=
sorry

end curve_is_semicircle_l707_707292


namespace number_of_elements_in_S_l707_707720

def S : Set ℕ := { n : ℕ | ∃ k : ℕ, n > 1 ∧ (10^10 - 1) % n = 0 }

theorem number_of_elements_in_S (h1 : Nat.Prime 9091) :
  ∃ T : Finset ℕ, T.card = 127 ∧ ∀ n, n ∈ T ↔ n ∈ S :=
sorry

end number_of_elements_in_S_l707_707720


namespace forty_second_card_is_three_l707_707945

theorem forty_second_card_is_three : 
  (∀ n : ℕ, let seq := ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'] in 
  n < seq.length → seq[(42 - 1) % seq.length] = '3') :=
by
  sorry

end forty_second_card_is_three_l707_707945


namespace math_problem_l707_707625

-- Define all necessary components and conditions.
variables {G : Type*} [graph G] {V : set G} {T : set (edge G)} {W : set G}
  [∀ t ∈ T, v_t : set G]

theorem math_problem (hW : W ⊆ V)
  (hT_sub_edges : T ⊆ edges G)
  (hV_t_sub_V : ∀ t ∈ T, v_t t ⊆ V) :
  (∃ t ∈ T, W ⊆ v_t t) ∨
  (∃ (w1 w2 ∈ W) (t1 t2 ∈ T), 
    w1 ∉ (v_t t1 ∩ v_t t2) ∧ 
    w2 ∉ (v_t t1 ∩ v_t t2) ∧ 
    separated_by (v_t t1 ∩ v_t t2) w1 w2) :=
by {
  sorry
}

end math_problem_l707_707625


namespace pyramid_volume_l707_707170

-- Define the given conditions
def base_area : ℝ := 6 * 8
def slant_height : ℝ := 13
def diagonal_length : ℝ := (6^2 + 8^2).sqrt
def half_diagonal : ℝ := diagonal_length / 2
def height : ℝ := (slant_height^2 - half_diagonal^2).sqrt

-- The statement to be proven
theorem pyramid_volume :
  (1 / 3) * base_area * height = 192 := sorry

end pyramid_volume_l707_707170


namespace sum_max_min_on_interval_l707_707102

-- Defining the function f
def f (x : ℝ) : ℝ := x + 2

-- The proof statement
theorem sum_max_min_on_interval : 
  let M := max (f 0) (f 4)
  let N := min (f 0) (f 4)
  M + N = 8 := by
  -- Placeholder for proof
  sorry

end sum_max_min_on_interval_l707_707102


namespace median_AQI_Chengdu_l707_707694

theorem median_AQI_Chengdu :
  let data := [33, 27, 34, 40, 26]
  let sorted_data := data.sort
  sorted_data[2] = 33 :=
by
  let data := [33, 27, 34, 40, 26]
  let sorted_data := list.sort data
  sorry

end median_AQI_Chengdu_l707_707694


namespace area_union_is_correct_l707_707174

-- Define the side length of the square and the radius of the circle
def side_length : ℝ := 12
def radius : ℝ := 12

-- Define the area of the square
def area_square : ℝ := side_length ^ 2

-- Define the area of the circle
def area_circle : ℝ := Real.pi * (radius ^ 2)

-- Define the area of the quarter circle inside the square
def area_quarter_circle : ℝ := (1 / 4) * area_circle

-- The total area of the union of the regions enclosed by the square and the circle
def area_union : ℝ := area_square + area_circle - area_quarter_circle

-- The theorem stating the desired result
theorem area_union_is_correct : area_union = 144 + 108 * Real.pi :=
by
  sorry

end area_union_is_correct_l707_707174


namespace sum_alternating_eight_l707_707978

-- Define the alternating sum function f for non-empty subsets
def alternatingSum (S : Finset ℕ) : ℤ :=
  let sorted := S.sort (· > ·)  -- Sort S in decreasing order
  sorted.foldr (λ x acc, if (sorted.indexOf x) % 2 = 0 then acc + x else acc - x) 0

-- Define the problem statement for n = 8
def problem_sum_alternating (n : ℕ) : ℤ :=
  if n = 8 then
    let subsets := (Finset.powerset (Finset.range 8).erase 0).filter (λ S, ¬S.isEmpty)  -- Non-empty subsets
    let initial_sum := subsets.sum (λ S, alternatingSum S + if S.card = 3 then 3 else 0)
    initial_sum
  else
    0

-- Prove that for n = 8, the sum of all alternating sums is 1192
theorem sum_alternating_eight : problem_sum_alternating 8 = 1192 := 
sorry

end sum_alternating_eight_l707_707978


namespace sin_pi_div_4_sub_alpha_is_correct_l707_707271

noncomputable def sin_pi_div_4_sub_alpha (α : ℝ) : ℝ :=
  Real.sin (π / 4 - α)

-- Problem statement in Lean 4
theorem sin_pi_div_4_sub_alpha_is_correct (α : ℝ)
  (h1: Real.sin α = -4 / 5)
  (h2: 3 / 2 * π < α ∧ α < 2 * π) :
  sin_pi_div_4_sub_alpha α = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end sin_pi_div_4_sub_alpha_is_correct_l707_707271


namespace sqrt_10_plus_2_range_l707_707590

theorem sqrt_10_plus_2_range : 5 < Real.sqrt 10 + 2 ∧ Real.sqrt 10 + 2 < 6 := by
  have h1 : 9 < 10 := by norm_num
  have h2 : 10 < 16 := by norm_num
  have h3 : Real.sqrt 9 = 3 := by norm_num
  have h4 : Real.sqrt 16 = 4 := by norm_num
  have h5 : Real.sqrt 9 < Real.sqrt 10 := Real.sqrt_lt.mpr (by norm_num)
  have h6 : Real.sqrt 10 < Real.sqrt 16 := Real.sqrt_lt.mpr (by norm_num)
  have h7 : 3 < Real.sqrt 10 := by linarith
  have h8 : Real.sqrt 10 < 4 := by linarith
  have h9 : 3 + 2 < Real.sqrt 10 + 2 := by linarith
  have h10 : Real.sqrt 10 + 2 < 4 + 2 := by linarith
  exact ⟨h9, h10⟩

end sqrt_10_plus_2_range_l707_707590


namespace mod_equiv_n_l707_707823

theorem mod_equiv_n (n : ℤ) : 0 ≤ n ∧ n < 9 ∧ -1234 % 9 = n := 
by
  sorry

end mod_equiv_n_l707_707823


namespace vertices_of_parabolas_is_parabola_l707_707723

theorem vertices_of_parabolas_is_parabola 
  (a c k : ℝ) (ha : 0 < a) (hc : 0 < c) (hk : 0 < k) :
  ∃ (f : ℝ → ℝ), (∀ t : ℝ, f t = (-k^2 / (4 * a)) * t^2 + c) ∧ 
  ∀ (pt : ℝ × ℝ), (∃ t : ℝ, pt = (-(k * t) / (2 * a), f t)) → 
  ∃ a' b' c', (∀ t : ℝ, pt.2 = a' * pt.1^2 + b' * pt.1 + c') ∧ (a < 0) :=
by sorry

end vertices_of_parabolas_is_parabola_l707_707723


namespace mutually_exclusive_not_complementary_l707_707407

-- Define the events and conditions
universe u
variable {Ω : Type u} [Fintype Ω] [DecidableEq Ω]
def events (A B C : Ω) : Event Ω := {e | e = A ∨ e = B ∨ e = C}
def receives_white (e : Event Ω) (p : Ω) : Prop := e = {p}

-- Given the conditions
variables {A B C : Ω}
axiom cards_unique : ∀ {p1 p2 : Ω}, (receives_white p1 A ∧ receives_white p2 B) → p1 ≠ p2

-- Define the problem statement
theorem mutually_exclusive_not_complementary : 
  (receives_white A A ∩ receives_white B B = ∅) ∧ (receives_white A A ∪ receives_white B B ≠ events A B C) :=
by sorry

end mutually_exclusive_not_complementary_l707_707407


namespace part_a_ctgs_eq_frac_squares_div_4S_part_b_sum_squares_ctgs_eq_4S_l707_707850

variable (a b c α β γ S : ℝ)

-- Part (a)
theorem part_a_ctgs_eq_frac_squares_div_4S
  (h : cot α + cot β + cot γ = (a * a + b * b + c * c) / (4 * S)) :
  cot α + cot β + cot γ = (a * a + b * b + c * c) / (4 * S) :=
by
  sorry

-- Part (b)
theorem part_b_sum_squares_ctgs_eq_4S
  (h : a * a * cot α + b * b * cot β + c * c * cot γ = 4 * S) :
  a * a * cot α + b * b * cot β + c * c * cot γ = 4 * S :=
by
  sorry

end part_a_ctgs_eq_frac_squares_div_4S_part_b_sum_squares_ctgs_eq_4S_l707_707850


namespace final_result_is_four_l707_707890

theorem final_result_is_four (x : ℕ) (h1 : x = 208) (y : ℕ) (h2 : y = x / 2) (z : ℕ) (h3 : z = y - 100) : z = 4 :=
by {
  sorry
}

end final_result_is_four_l707_707890


namespace divide_composite_structure_into_three_equal_parts_l707_707813

-- Define a data structure for a cube
structure Cube :=
  (vertices : Fin 8 → ℝ × ℝ × ℝ)

-- Define a function to glue additional cubes to a given face of the initial cube
def glue_three_cubes (initial_cube : Cube) (common_vertex : ℝ × ℝ × ℝ) : Cube :=
  sorry -- placeholder for the concrete construction of the composite cube

-- The main statement requesting proof
theorem divide_composite_structure_into_three_equal_parts (initial_cube : Cube) (D1 : ℝ × ℝ × ℝ)
  (composite_cube : Cube := glue_three_cubes initial_cube D1) :
  ∃ (parts : Fin 3 → Set (ℝ × ℝ × ℝ)), 
  (∀ i, parts i ≠ ∅) ∧ 
  (∀ i j, i ≠ j → parts i ∩ parts j = ∅) ∧ 
  (∪ i, parts i = {v | v ∈ composite_cube.vertices ∘ Fin.to_nat}) ∧ 
  (∀ i j, ∀ x ∈ parts i, ∀ y ∈ parts j, congruent x y) :=
sorry -- this statement means we need a proof here

end divide_composite_structure_into_three_equal_parts_l707_707813


namespace part1_part2_part3_l707_707447

noncomputable 
def test_scores_seventh : List ℤ := [86, 94, 79, 84, 71, 90, 76, 83, 90, 87]

noncomputable
def test_scores_eighth : List ℤ := [88, 76, 90, 78, 87, 93, 75, 87, 87, 79]

noncomputable
def mean_seventh := 84

noncomputable
def variance_seventh := 44.4

noncomputable
def median_seventh := 85

noncomputable
def mode_seventh := 90

noncomputable
def mean_eighth := 84

noncomputable
def variance_eighth := 36.6

noncomputable
def median_eighth := 87

noncomputable
def mode_eighth := 87

noncomputable
def student_A_score := 86

theorem part1 : 
  median_seventh = 85 ∧ 
  mode_eighth = 87 ∧ 
  (student_A_score > mean_seventh ∧ student_A_score > median_seventh → "seventh" = "seventh") := 
  sorry

theorem part2 :
  let excellent_threshold := 85 in
  let seventh_excellent_count := 5 in
  let eighth_excellent_count := 6 in
  seventh_excellent_count / 10 * 200 + eighth_excellent_count / 10 * 200 = 220 :=
  sorry

theorem part3 :
  variance_eighth < variance_seventh → 
  "Eighth grade students have a better overall level of understanding of national security knowledge." := 
  sorry

end part1_part2_part3_l707_707447


namespace zero_in_interval_l707_707677

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

theorem zero_in_interval :
  ∃ x₀ : ℝ, f x₀ = 0 ∧ x₀ ∈ Ioo (-1 : ℝ) 0 :=
begin
  sorry
end

end zero_in_interval_l707_707677


namespace solve_polynomial_l707_707226

theorem solve_polynomial (z : ℂ) : z^6 - 9 * z^3 + 8 = 0 ↔ z = 1 ∨ z = 2 := 
by
  sorry

end solve_polynomial_l707_707226


namespace count_success_permutations_l707_707941

theorem count_success_permutations : 
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  (Nat.factorial total_letters) / ((Nat.factorial s_count) * (Nat.factorial c_count)) = 420 := 
by
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  sorry

end count_success_permutations_l707_707941


namespace three_hundredth_term_without_squares_l707_707492

noncomputable def sequence_without_squares (n : ℕ) : ℕ :=
(n + (n / Int.natAbs (Int.sqrt (n.succ - 1))))

theorem three_hundredth_term_without_squares : 
  sequence_without_squares 300 = 307 :=
sorry

end three_hundredth_term_without_squares_l707_707492


namespace arrangement_count_SUCCESS_l707_707937

-- Define the conditions for the problem
def letters : Finset String := {"S", "U", "C", "C", "E", "S", "S"}
def occurrences_S : Nat := 3
def occurrences_C : Nat := 2
def occurrences_other : Nat := 1 -- For 'U' and 'E'

-- State the theorem using these conditions
theorem arrangement_count_SUCCESS : 
  let N := letters.card
  N = 7 →
  occurrences_S = 3 →
  occurrences_C = 2 →
  occurrences_other = 1 →
  Nat.factorial N / (Nat.factorial occurrences_S * Nat.factorial occurrences_C * Nat.factorial occurrences_other * Nat.factorial occurrences_other) = 420 :=
by
  sorry

end arrangement_count_SUCCESS_l707_707937


namespace trapezoid_perimeter_log_l707_707340

theorem trapezoid_perimeter_log (p q : ℕ) (AB CD altitude perimeter : ℝ)
  (hAB : AB = log 3) 
  (hCD : CD = log 192) 
  (haltitude : altitude = log 16)
  (hperimeter : perimeter = AB + 2 * log 32 + CD) 
  (hlog_form : perimeter = log (2^p * 3^q)) : p + q = 18 :=
by
  sorry

end trapezoid_perimeter_log_l707_707340


namespace magnitude_of_z_l707_707680

theorem magnitude_of_z : ∀ (z : ℂ), z = (4 : ℂ) + 3 * complex.I → complex.abs z = 5 :=
by
  intro z
  intro hz
  rw [hz, complex.abs]
  -- proof steps go here
  sorry

end magnitude_of_z_l707_707680


namespace coles_average_speed_l707_707919

theorem coles_average_speed (work_travel_time_hours : ℝ)
  (home_to_work_speed km_per_hour : ℝ)
  (round_trip_time_hours : ℝ)
  (home_to_work_time_minutes : ℝ)
  (home_to_work_time_in_hours : work_travel_time_hours = home_to_work_time_minutes / 60)
  (distance_to_work : distance_of_work = home_to_work_speed * work_travel_time_hours.sum) (home_to_work_speed : home_to_work_speed = 75)
  (round_trip_time : round_trip_time_hours = 2)
  (home_to_work_time : home_to_work_time_minutes = 70)
  (distance : distance_of_work = 62.5) :
  returns_home_speed = 75 :=
by
  sorry

end coles_average_speed_l707_707919


namespace points_4_units_away_largest_neg_int_l707_707750

theorem points_4_units_away_largest_neg_int :
  (largest_negative_int : ℤ) (h : largest_negative_int = -1) :
  ∃ x : ℤ, (|x - largest_negative_int| = 4 ∧ (x = 3 ∨ x = -5)) :=
by 
  existsi [(3 : ℤ), (-5 : ℤ)]
  rw h
  split
  rfl
  rfl
  sorry

end points_4_units_away_largest_neg_int_l707_707750


namespace greatest_possible_selling_price_l707_707851

variable (products : ℕ)
variable (average_price : ℝ)
variable (min_price : ℝ)
variable (less_than_1000_products : ℕ)

theorem greatest_possible_selling_price
  (h1 : products = 20)
  (h2 : average_price = 1200)
  (h3 : min_price = 400)
  (h4 : less_than_1000_products = 10) :
  ∃ max_price, max_price = 11000 := 
by
  sorry

end greatest_possible_selling_price_l707_707851


namespace min_value_expression_l707_707311

theorem min_value_expression (a b : ℤ) (h : a > b) : 
  ∃ x > 0, x = (a + 2 * b) / (a - b) + (a - b) / (a + 2 * b) ∧ x = 2 :=
begin
  sorry
end

end min_value_expression_l707_707311


namespace non_square_300th_term_l707_707500

theorem non_square_300th_term (N : ℕ) (hN : N = 300) : 
  ∃ x : ℕ, (∀ (t : ℕ), 0 < t ∧ t ≤ x → ¬ (∃ k : ℕ, t = k^2)) ∧ 
           (∑ t in finset.range (N + 17), if ∃ k : ℕ, t = k^2 then 0 else 1 = N) ∧ 
           x = 317 := by
  sorry

end non_square_300th_term_l707_707500


namespace sin_double_angle_l707_707611

theorem sin_double_angle (α : ℝ) (h_tan : Real.tan α < 0) (h_sin : Real.sin α = - (Real.sqrt 3) / 3) :
  Real.sin (2 * α) = - (2 * Real.sqrt 2) / 3 := 
by
  sorry

end sin_double_angle_l707_707611


namespace sum_of_divisors_divisible_by_24_l707_707731

theorem sum_of_divisors_divisible_by_24 
  (n : ℕ) (h : 24 ∣ (n + 1)) : 24 ∣ ∑ d in (finset.divisors n), d :=
sorry

end sum_of_divisors_divisible_by_24_l707_707731


namespace triangle_area_l707_707000

-- Define the side lengths of the triangle
def PQ : ℝ := 13
def PR : ℝ := 14
def QR : ℝ := 15

-- Define the semi-perimeter
def s : ℝ := (PQ + PR + QR) / 2

-- State the theorem to find the area using Heron's formula
theorem triangle_area :
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  area = 84 := 
by
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  have : area = 84 := sorry
  exact this

end triangle_area_l707_707000


namespace three_hundredth_term_without_squares_l707_707493

noncomputable def sequence_without_squares (n : ℕ) : ℕ :=
(n + (n / Int.natAbs (Int.sqrt (n.succ - 1))))

theorem three_hundredth_term_without_squares : 
  sequence_without_squares 300 = 307 :=
sorry

end three_hundredth_term_without_squares_l707_707493


namespace equal_areas_of_quadrilateral_l707_707541

variables {A B C D M N : Type} [affine_space ℝ A] [affine_space ℝ B] [affine_space ℝ C] [affine_space ℝ D]
variables [affine_space ℝ M] [affine_space ℝ N]

def midpoint (x y : Type) [affine_space ℝ x] [affine_space ℝ y] : Type := sorry -- define midpoint function

def area (x y z : Type) [affine_space ℝ x] [affine_space ℝ y] [affine_space ℝ z] : ℝ := sorry -- define area function

theorem equal_areas_of_quadrilateral
  (K : Type) (L : Type) [affine_space ℝ K] [affine_space ℝ L]
  (hK : K = midpoint D B) (hL : L = midpoint A C) :
  area D C M = area A B N :=
by sorry

end equal_areas_of_quadrilateral_l707_707541


namespace BEIH_area_is_60_over_91_l707_707845

open Set Real EuclideanGeometry Affine

noncomputable section
def midpoint (p q : Point ℝ) : Point ℝ :=
  (1/2 : ℝ) • p + (1/2 : ℝ) • q

def lineFromPoints (p q : Point ℝ) : Line ℝ :=
  linearMap ℝ ℝ ℝ (q - p)

def quad_area (p1 p2 p3 p4 : Point ℝ) : ℝ :=
  abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p4.2 + p4.1 * p1.2) - 
       (p1.2 * p2.1 + p2.2 * p3.1 + p3.2 * p4.1 + p4.2 * p1.1)) / 2

def area_BEIH : ℝ :=
  @quad_area ℝ (0, 0) (0, 3 / 2) (8 / 13, 30 / 13) (5 / 7, 15 / 7)

theorem BEIH_area_is_60_over_91 : area_BEIH = 60 / 91 := by
  sorry

end BEIH_area_is_60_over_91_l707_707845


namespace count_integers_in_abs_inequality_l707_707667

theorem count_integers_in_abs_inequality : 
  (set.count (set_of (λ x : ℤ, abs (x - 3) ≤ 6))) = 13 :=
by sorry

end count_integers_in_abs_inequality_l707_707667


namespace find_k_for_eccentricity_l707_707273

def is_ellipse_formula (a b : ℝ) : Prop := ∃ x y, (x^2 / a) + (y^2 / b) = 1

def eccentricity_ellipse (a b : ℝ) : ℝ := sqrt (1 - (b^2 / a^2))

theorem find_k_for_eccentricity (k : ℝ) :
  is_ellipse_formula (k + 4) 9 ∧
  eccentricity_ellipse (k + 4) 9 = 1 / 2 ∧
  k + 4 > 9 → k = 8 := 
sorry

end find_k_for_eccentricity_l707_707273


namespace parallel_planes_necessary_not_sufficient_l707_707363

-- Definitions for the conditions
variables {α β : Type*} [Plane α] [Plane β]
variable (m : Line α)

-- Given conditions
axiom m_subset_alpha : m ∈ α
axiom m_parallel_beta : m ∥ β

-- Statement representing the problem condition
theorem parallel_planes_necessary_not_sufficient :
  (m ∥ β) → (m ∈ α) → (α || β = False) :=
by
  assume h_m_parallel_beta h_m_subset_alpha,
  sorry

end parallel_planes_necessary_not_sufficient_l707_707363


namespace relationship_among_functions_l707_707245

theorem relationship_among_functions 
  (f g h : ℝ → ℝ)
  (a : ℝ)
  (h₀ : 0 < a)
  (h₁ : a < 1)
  (h_f : ∀ x, f x = a^x)
  (h_g : ∀ x, g x = log a x)
  (h_h : ∀ x, h x = x^a) :
  h 2 > f 2 ∧ f 2 > g 2 := 
sorry

end relationship_among_functions_l707_707245


namespace triangle_inequality_l707_707913

variables (T : Type) [IsTriangle T]
variables (a b c : ℝ) -- lengths of the sides of the triangle
variables (β_a β_b β_c : ℝ) -- internal angle bisectors
variables (m_a m_b m_c : ℝ) -- medians
variables (p r R : ℝ) -- semiperimeter, inradius, circumradius

-- Conditions on bisectors and medians
def is_internal_angle_bisector (T : Type) (β : ℝ) : Prop := sorry
def is_median (T : Type) (m : ℝ) : Prop := sorry

-- Semiperimeter
def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Inradius
def incircle_radius (T : Type) : ℝ := sorry

-- Circumradius
def circumradius (T : Type) : ℝ := sorry

theorem triangle_inequality (T : Type) [IsTriangle T] 
  (a b c : ℝ) (β_a β_b β_c m_a m_b m_c p r R : ℝ) 
  (hβa : is_internal_angle_bisector T β_a) 
  (hβb : is_internal_angle_bisector T β_b) 
  (hβc : is_internal_angle_bisector T β_c)
  (hma : is_median T m_a)
  (hmb : is_median T m_b)
  (hmc : is_median T m_c)
  (h_p : p = semiperimeter a b c)
  (h_r : r = incircle_radius T)
  (h_R : R = circumradius T) :
  (β_a ^ 6 + β_b ^ 6 + β_c ^ 6 ≤ p ^ 4 * (p ^ 2 - 12 * r * R) ∧ 
   p ^ 4 * (p ^ 2 - 12 * r * R) ≤ m_a ^ 6 + m_b ^ 6 + m_c ^ 6) ↔ 
  (is_equilateral_triangle T) := 
sorry

end triangle_inequality_l707_707913


namespace diameter_of_circle_l707_707121

noncomputable def circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : Prop :=
  A = 400 * π ∧ A = π * r^2 ∧ d = 2 * r

theorem diameter_of_circle : circle_diameter 400π 20 40 :=
  by {
    -- conditions
    split,
    { exact rfl }, -- A = 400 * π
    split,
    { norm_num, ring }, -- A = π * r^2
    { norm_num }, -- d = 2 * r
  }

end diameter_of_circle_l707_707121


namespace ratio_m_n_l707_707310

theorem ratio_m_n (m n : ℕ) (h : (n : ℚ) / m = 3 / 7) : (m + n : ℚ) / m = 10 / 7 := by 
  sorry

end ratio_m_n_l707_707310


namespace root_polynomial_l707_707438

noncomputable theory

def polynomial (a b c d e : ℝ) : polynomial ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem root_polynomial (a b c d e : ℝ) (h : a ≠ 0) (h_roots : (polynomial a b c d e).roots = [1, 2, 3, 4]) :
  c / e = 35 / 24 := 
by 
  have h_Vieta_sum := by sorry -- By Vieta's formula for roots sum
  have h_Vieta_sum_prod_two := by sorry -- By Vieta's formula for sum of product of roots taken two at a time
  have h_Vieta_prod_three := by sorry -- By Vieta's formula for product of roots taken three at a time
  have h_Vieta_all := by sorry -- By Vieta's formula for product of all roots
  exact sorry

end root_polynomial_l707_707438


namespace street_length_l707_707168

theorem street_length
  (time_minutes : ℕ)
  (speed_kmph : ℕ)
  (length_meters : ℕ)
  (h1 : time_minutes = 12)
  (h2 : speed_kmph = 9)
  (h3 : length_meters = 1800) :
  length_meters = (speed_kmph * 1000 / 60) * time_minutes :=
by sorry

end street_length_l707_707168


namespace Vasilisa_minimum_points_l707_707397

open Classical

-- Define the basic properties and the deck
inductive Suit : Type
| Hearts
| Diamonds
| Clubs
| Spades

inductive Rank : Type
| One | Two | Three | Four | Five | Six | Seven | Eight | Nine

structure Card :=
  (suit : Suit)
  (rank : Rank)

def deck := { c : Card // true }

def splitDeck (cards : set Card) : Prop :=
  ∃ p q : set Card, p ∪ q = cards ∧ p ∩ q = ∅ ∧ p.card = 18 ∧ q.card = 18

def Vasilisa_guarantee_points (p q : set Card) (p_turn : bool) (v_points : ℕ) : Prop :=
  ∃ n, v_points ≥ 15

theorem Vasilisa_minimum_points :
  ∀ (cards : set Card) (p q : set Card), splitDeck cards → Vasilisa_guarantee_points p q true 0 ↔ 15 :=
by sorry

end Vasilisa_minimum_points_l707_707397


namespace satisfies_equation_l707_707146

noncomputable def y (x : ℝ) : ℝ := -Real.sqrt (x^4 - x^2)
noncomputable def dy (x : ℝ) : ℝ := x * (1 - 2 * x^2) / Real.sqrt (x^4 - x^2)

theorem satisfies_equation (x : ℝ) (hx : x ≠ 0) : x * y x * dy x - (y x)^2 = x^4 := 
sorry

end satisfies_equation_l707_707146


namespace mrs_wong_initial_valentines_l707_707390

theorem mrs_wong_initial_valentines (x : ℕ) (given left : ℕ) (h_given : given = 8) (h_left : left = 22) (h_initial : x = left + given) : x = 30 :=
by
  rw [h_left, h_given] at h_initial
  exact h_initial

end mrs_wong_initial_valentines_l707_707390


namespace student_loses_one_mark_per_wrong_answer_l707_707339

noncomputable def marks_lost_per_wrong_answer (x : ℝ) : Prop :=
  let total_questions := 60
  let correct_answers := 42
  let wrong_answers := total_questions - correct_answers
  let marks_per_correct := 4
  let total_marks := 150
  correct_answers * marks_per_correct - wrong_answers * x = total_marks

theorem student_loses_one_mark_per_wrong_answer : marks_lost_per_wrong_answer 1 :=
by
  sorry

end student_loses_one_mark_per_wrong_answer_l707_707339


namespace average_speed_round_trip_l707_707178

theorem average_speed_round_trip
  (n : ℕ)
  (distance_km : ℝ := n / 1000)
  (pace_west_min_per_km : ℝ := 2)
  (speed_east_kmh : ℝ := 3)
  (wait_time_hr : ℝ := 30 / 60) :
  (2 * distance_km) / 
  ((pace_west_min_per_km * distance_km / 60) + wait_time_hr + (distance_km / speed_east_kmh)) = 
  60 * n / (11 * n + 150000) := by
  sorry

end average_speed_round_trip_l707_707178


namespace problem_abc_l707_707376

noncomputable def valid_subsets_count : ℕ :=
  let U := Finset.range (11) -- The universal set {1, 2, ..., 10}
  let is_valid (A B : Finset ℕ) : Prop := 
    A ∪ B = U ∧ A ∩ B = ∅ ∧ 
    (A.card ∉ A) ∧ (B.card ∉ B)
  Finset.filter (λ A : Finset ℕ, 
    is_valid A (U \ A)).card

theorem problem_abc :
  valid_subsets_count = 186 := 
sorry

end problem_abc_l707_707376


namespace solve_theta_l707_707180

theorem solve_theta 
  (θ : ℝ) 
  (h1 : 0 ≤ θ ∧ θ ≤ 180) 
  (h2 : sqrt 2 * cos (2 * θ) = cos θ + sin θ) : 
  θ = 15 ∨ θ = 135 :=
sorry

end solve_theta_l707_707180


namespace tangent_parallel_to_CD_l707_707086

-- Definitions of the entities involved
variables {A B C D K : Type} 

-- Conditions: inscribed quadrilateral and diagonal intersection
axiom inscribed_quadrilateral (A B C D : Type) : Prop
axiom diagonals_intersect (A B C D K : Type) : Prop

-- Assumptions
variable [h1 : inscribed_quadrilateral A B C D]
variable [h2 : diagonals_intersect A B C D K]

-- Statement to prove
theorem tangent_parallel_to_CD 
  (circumcircle_ABK : Prop) 
  (tangent_at_K : Prop) :
  (parallel : tangent_at_K = CD) := 
sorry

end tangent_parallel_to_CD_l707_707086


namespace infinite_distinct_solutions_l707_707400

theorem infinite_distinct_solutions (k : ℕ) : ∃ (infinitely_many: set ℤ × set ℤ × set ℤ),
(∀ (x y z : ℤ), (x, y, z) ∈ infinitely_many → x^2 = y^2 + k * z^2)
∧ ∀ (a b c a₁ b₁ c₁ : ℤ), ((a, b, c) ∈ infinitely_many ∧ (a₁, b₁, c₁) ∈ infinitely_many) → 
(¬ ∃ r : ℚ, a₁ = r * a ∧ b₁ = r * b ∧ c₁ = r * c) := 
sorry

end infinite_distinct_solutions_l707_707400


namespace housewife_oil_cost_l707_707895

theorem housewife_oil_cost (P R M : ℝ) (hR : R = 45) (hReduction : (P - R) = (15 / 100) * P)
  (hMoreOil : M / P = M / R + 4) : M = 150.61 := 
by
  sorry

end housewife_oil_cost_l707_707895


namespace jane_mean_score_l707_707355

-- Define the six quiz scores Jane took
def score1 : ℕ := 86
def score2 : ℕ := 91
def score3 : ℕ := 89
def score4 : ℕ := 95
def score5 : ℕ := 88
def score6 : ℕ := 94

-- The number of quizzes
def num_quizzes : ℕ := 6

-- The sum of all quiz scores
def total_score : ℕ := score1 + score2 + score3 + score4 + score5 + score6 

-- The expected mean score
def mean_score : ℚ := 90.5

-- The proof statement
theorem jane_mean_score (h : total_score = 543) : total_score / num_quizzes = mean_score := 
by sorry

end jane_mean_score_l707_707355


namespace set_equality_l707_707295

theorem set_equality : 
  { x : ℕ | ∃ k : ℕ, 6 - x = k ∧ 8 % k = 0 } = { 2, 4, 5 } :=
by
  sorry

end set_equality_l707_707295


namespace robin_packages_gum_l707_707763

/-
Conditions:
1. Robin has 14 packages of candy.
2. There are 6 pieces in each candy package.
3. Robin has 7 additional pieces.
4. Each package of gum contains 6 pieces.

Proof Problem:
Prove that the number of packages of gum Robin has is 15.
-/
theorem robin_packages_gum (candies_packages : ℕ) (pieces_per_candy_package : ℕ)
                          (additional_pieces : ℕ) (pieces_per_gum_package : ℕ) :
  candies_packages = 14 →
  pieces_per_candy_package = 6 →
  additional_pieces = 7 →
  pieces_per_gum_package = 6 →
  (candies_packages * pieces_per_candy_package + additional_pieces) / pieces_per_gum_package = 15 :=
by intros h1 h2 h3 h4; sorry

end robin_packages_gum_l707_707763


namespace largest_three_digit_n_l707_707831

theorem largest_three_digit_n (n : ℕ) : 
  (70 * n ≡ 210 [MOD 350]) ∧ (n < 1000) → 
  n = 998 := by
  sorry

end largest_three_digit_n_l707_707831


namespace success_permutations_correct_l707_707935

theorem success_permutations_correct :
  let word := "SUCCESS"
  let n := 7
  let s_count := 3
  let c_count := 2
  let u_count := 1
  let e_count := 1
  let total_permutations := (Nat.factorial n) / ((Nat.factorial s_count) * (Nat.factorial c_count) * (Nat.factorial u_count) * (Nat.factorial e_count))
  total_permutations = 420 :=
by
  sorry

end success_permutations_correct_l707_707935


namespace triangle_ABC_perimeter_l707_707701

theorem triangle_ABC_perimeter :
  let P Q R S : Type := ℝ
  let radius := 2
  let circle_distance := radius * 2
  let PQ PR PS RQ RS := circle_distance
  let PQR_angle := 120 * (π / 180)
  let PRS_angle := 60 * (π / 180)
  let triangle_ABC_perimeter := 16 + 4 * Real.sqrt 3
  
  (circle_distance = 4) →
  (PQ = circle_distance) →
  (PR = circle_distance) →
  (PS = circle_distance) →
  (RQ = circle_distance) →
  (RS = circle_distance) →
  ∃ (ABC : Type),
  (triangle_ABC_perimeter = 16 + 4 * Real.sqrt 3) :=
by
  intros
  sorry

end triangle_ABC_perimeter_l707_707701


namespace simon_paid_amount_l707_707413

theorem simon_paid_amount:
  let pansy_price := 2.50
  let hydrangea_price := 12.50
  let petunia_price := 1.00
  let pansies_count := 5
  let hydrangeas_count := 1
  let petunias_count := 5
  let discount_rate := 0.10
  let change_received := 23.00

  let total_cost_before_discount := (pansies_count * pansy_price) + (hydrangeas_count * hydrangea_price) + (petunias_count * petunia_price)
  let discount := discount_rate * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount
  let amount_paid_with := total_cost_after_discount + change_received

  amount_paid_with = 50.00 :=
by
  sorry

end simon_paid_amount_l707_707413


namespace min_product_angle_bisector_l707_707261

noncomputable def min_product {A B P C : Type*}
  [has_inner_product A B P C]
  (P : Point) (A : Angle) : Prop :=
  ∃ (B C : Point), 
  P ∈ interior_angle A ∧
  B ∈ side1_of_angle A ∧
  C ∈ side2_of_angle A ∧
  perpendicular (angle_bisector A) (line_through P (B::C::[])) ∧
  ∀ (B' C' : Point), B ≠ B' ∧ C ≠ C' → (BP * PC) ≤ (B'P * P'C')

theorem min_product_angle_bisector 
  (P : Point) 
  (A : Angle) 
  (H: P ∈ interior_angle A) : min_product P A :=
begin
  sorry
end

end min_product_angle_bisector_l707_707261


namespace smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum_l707_707093

theorem smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum :
  ∃ (a : ℤ), (∃ (l : List ℤ), l.length = 50 ∧ List.prod l = 0 ∧ 0 < List.sum l ∧ List.sum l = 25) :=
by
  sorry

end smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum_l707_707093


namespace percentage_of_left_handed_women_l707_707143

variable (x y : Nat) (h_ratio_rh_lh : 3 * x = 1 * x)
variable (h_ratio_men_women : 3 * y = 2 * y)
variable (h_rh_men_max : True)

theorem percentage_of_left_handed_women :
  (x / (4 * x)) * 100 = 25 :=
by sorry

end percentage_of_left_handed_women_l707_707143


namespace king_travel_moves_l707_707024

theorem king_travel_moves (n : ℕ) (h_odd : n % 2 = 1) (h_pos : 0 < n) 
  (board : ℕ × ℕ → bool) (green : ∀ x y, (x ≤ n) → (y ≤ n) → bool) 
  (h_connected : ∀ x y z w, green x y → green z w → connected x y z w green) :
  ∀ x1 y1 x2 y2, (green x1 y1) → (green x2 y2) → 
    ∃ p : list (ℕ × ℕ), (∀ (i : ℕ), i < list.length p - 1 → adjacent (list.nth_le p i sorry) (list.nth_le p (i + 1) sorry)) ∧ (list.nth_le p 0 sorry = (x1, y1)) ∧ (list.nth_le p (list.length p - 1) sorry = (x2, y2)) ∧ (list.length p - 1 ≤ (n^2 - 1) / 2) :=
by sorry

end king_travel_moves_l707_707024


namespace coordinates_of_T_l707_707220

theorem coordinates_of_T {O Q P R T : ℝ × ℝ}
  (hO : O = (0,0))
  (hQ : Q = (3,3))
  (hSquare : ∃ d, Q = (d, d) ∧ P = (d, 0) ∧ R = (0, d))
  (hArea_Triangle : ∃ T, 2 * (abs ((Q.1 - P.1) * (T.2 - P.2) - (Q.2 - P.2) * (T.1 - P.1)) / 2) = ((Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2)) :
  T = (3, 6) :=
sorry

end coordinates_of_T_l707_707220


namespace cubic_polynomial_sum_l707_707201

noncomputable def cubic_polynomial (q : ℝ → ℝ) :=
  ∃ a b c d : ℝ, ∀ x, q x = a * x^3 + b * x^2 + c * x + d

theorem cubic_polynomial_sum (q : ℝ → ℝ)
  (h_cubic : cubic_polynomial q)
  (h1 : q 3 = 4)
  (h2 : q 10 = 26)
  (h3 : q 12 = 8)
  (h4 : q 22 = 36) :
  (∑ n in finset.range 20, q (n + 3)) = 400 :=
by
  sorry

end cubic_polynomial_sum_l707_707201


namespace tom_total_distance_l707_707469

-- Definitions corresponding to the problem conditions
def Time_swim := 2 -- hours
def Speed_swim := 2 -- miles per hour
def Time_run := Time_swim / 2 -- Tom runs for half the time he spent swimming
def Speed_run := Speed_swim * 4 -- Tom's running speed is 4 times his swimming speed

-- The Lean theorem to prove the total distance covered by Tom is 12 miles
theorem tom_total_distance : 
  let Distance_swim := Time_swim * Speed_swim in
  let Distance_run := Time_run * Speed_run in
  Distance_swim + Distance_run = 12 :=
by
  sorry

end tom_total_distance_l707_707469


namespace macy_running_goal_l707_707744

/-- Macy's weekly running goal is 24 miles. She runs 3 miles per day. Calculate the miles 
    she has left to run after 6 days to meet her goal. --/
theorem macy_running_goal (miles_per_week goal_per_week : ℕ) (miles_per_day: ℕ) (days_run: ℕ) 
  (h1 : miles_per_week = 24) (h2 : miles_per_day = 3) (h3 : days_run = 6) : 
  miles_per_week - miles_per_day * days_run = 6 := 
  by 
    rw [h1, h2, h3]
    exact Nat.sub_eq_of_eq_add (by norm_num)

end macy_running_goal_l707_707744


namespace water_purification_problem_l707_707858

variable (x : ℝ) (h : x > 0)

theorem water_purification_problem
  (h1 : ∀ (p : ℝ), p = 2400)
  (h2 : ∀ (eff : ℝ), eff = 1.2)
  (h3 : ∀ (time_saved : ℝ), time_saved = 40) :
  (2400 * 1.2 / x) - (2400 / x) = 40 := by
  sorry

end water_purification_problem_l707_707858


namespace minimum_resultant_vector_length_l707_707749

open Real

-- Definitions according to the conditions
def edge_length : ℝ := 1
def main_diagonal_length : ℝ := sqrt 3

-- Sum of vectors according to the given conditions
def sum_vectors (k m n : ℕ) : ℝ := 
  sqrt (k^2 + m^2 + n^2)

-- The minimum length of the resultant vector formed by the vectors
theorem minimum_resultant_vector_length :
  (∃ k m n : ℕ, (k % 2 = 1) ∧ (m % 2 = 1) ∧ (n % 2 = 1) ∧ (sum_vectors k m n = sqrt 3)) :=
sorry

end minimum_resultant_vector_length_l707_707749


namespace probability_real_complex_expression_l707_707066

/-- Rational numbers are chosen at random among all rational numbers in the interval [0, 3)
that can be written as fractions n/d where n and d are integers with 1 ≤ d ≤ 6. 
What is the probability that (cos(a * pi) + I * sin(b * pi))^6 is a real number? -/
theorem probability_real_complex_expression : 
  let S := {r : ℚ | ∃ n d : ℤ, 1 ≤ d ∧ d ≤ 6 ∧ 0 ≤ (n : ℚ) / d ∧ (n : ℚ) / d < 3} in
  let total_pairs := S.prod S in
  let real_pairs := {p : ℚ × ℚ | p.1 ∈ S ∧ p.2 ∈ S ∧ ∃ k : ℤ, p.2 = k / 6} in
  real_pairs.card / total_pairs.card = 57 / 361 := 
sorry

end probability_real_complex_expression_l707_707066


namespace max_median_value_l707_707336

noncomputable def max_median_ratio (a b : ℝ) : ℝ :=
  let s_a := sqrt((a^2 / 4) + b^2)
  let s_b := sqrt((b^2 / 4) + a^2)
  let s_c := sqrt((a^2 + b^2) / 4)
  (s_a + s_b) / s_c

theorem max_median_value (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  a = b → max_median_ratio a b = sqrt(10) := sorry

end max_median_value_l707_707336


namespace exponent_sum_equality_l707_707670

theorem exponent_sum_equality {a : ℕ} (h1 : 2^12 + 1 = 17 * a) (h2: a = 2^8 + 2^7 + 2^6 + 2^5 + 2^0) : 
  ∃ a1 a2 a3 a4 a5 : ℕ, 
    a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ 
    2^a1 + 2^a2 + 2^a3 + 2^a4 + 2^a5 = a ∧ 
    a1 = 0 ∧ a2 = 5 ∧ a3 = 6 ∧ a4 = 7 ∧ a5 = 8 ∧ 
    5 = 5 :=
by {
  sorry
}

end exponent_sum_equality_l707_707670


namespace avg_speed_between_B_and_C_l707_707887

noncomputable def avg_speed_from_B_to_C : ℕ := 20

theorem avg_speed_between_B_and_C
    (A_to_B_dist : ℕ := 120)
    (A_to_B_time : ℕ := 4)
    (B_to_C_dist : ℕ := 120) -- three-thirds of A_to_B_dist
    (C_to_D_dist : ℕ := 60) -- half of B_to_C_dist
    (C_to_D_time : ℕ := 2)
    (total_avg_speed : ℕ := 25)
    : avg_speed_from_B_to_C = 20 := 
  sorry

end avg_speed_between_B_and_C_l707_707887


namespace max_tan_BAD_l707_707471

-- Given conditions in Lean
variable (A B C D : Type*)
variable [EuclideanGeometry A B C D]

-- Define the triangle and its properties
variable (angleC : ∠ C = 45)
variable (BC_length : BC = 6)
variable (midpointD : midpoint D B C)

-- Expected result
theorem max_tan_BAD (A B C D : Type*) [EuclideanGeometry A B C D]
  (angleC : ∠ C = 45)
  (BC_length : BC = 6)
  (midpointD : midpoint D B C) :
  ∃ tan_BAD : ℝ, tan_BAD = 1 / (4 * sqrt 2 - 3) := 
sorry

end max_tan_BAD_l707_707471


namespace periodic_function_period_l707_707236

theorem periodic_function_period (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x + 2) + f(x - 2) = f(x)) :
  ∃ p : ℝ, p = 12 ∧ ∀ x : ℝ, f(x + p) = f(x) :=
begin
  use 12,
  split,
  { refl, },
  sorry,
end

end periodic_function_period_l707_707236


namespace find_integer_l707_707821

theorem find_integer (n : ℤ) (h1 : 0 ≤ n) (h2 : n < 9) (h3 : -1234 ≡ n [MOD 9]) : n = 8 := 
sorry

end find_integer_l707_707821


namespace correct_propositions_l707_707026

variables {x1 y1 x2 y2 a b c : ℝ}

def M := (x1, y1)
def N := (x2, y2)
def l (x y : ℝ) := a * x + b * y + c = 0

def delta := (a * x1 + b * y1 + c) / (a * x2 + b * y2 + c)

-- Proposition ①: Regardless of the value of δ, point N is not on line l.
def prop1 := ¬ l x2 y2

-- Proposition ②: If δ = 1, then line l perpendicularly bisects segment MN.
def prop2 := delta = 1 → ¬ (l x2 y2) ∧ ((b ≠ 0 → (y2 - y1) / (x2 - x1) = -a / b ∧ y2 ≠ y1) ∨ (b = 0 → x1 = x2 ∧ y1 ≠ y2))

-- Proposition ③: If δ = -1, then line l passes through the midpoint of segment MN.
def prop3 := delta = -1 → l ((x1 + x2) / 2) ((y1 + y2) / 2)

-- Proposition ④: If δ > 1, then points M and N are on the same side of line l, and l intersects the extension of segment MN.
def prop4 := delta > 1 → (a * x1 + b * y1 + c) * (a * x2 + b * y2 + c) > 0 ∧ |a * x1 + b * y1 + c| > |a * x2 + b * y2 + c|

-- The final proof statement combining all propositions
theorem correct_propositions : prop1 ∧ prop3 ∧ prop4 :=
by sorry

end correct_propositions_l707_707026


namespace largest_three_digit_n_l707_707832

theorem largest_three_digit_n (n : ℕ) : 
  (70 * n ≡ 210 [MOD 350]) ∧ (n < 1000) → 
  n = 998 := by
  sorry

end largest_three_digit_n_l707_707832


namespace problem_l707_707931

noncomputable def A : ℕ → ℝ
| 0     := 1
| (n+1) := (A n + 2) / (A n + 1)

noncomputable def B : ℕ → ℝ
| 0     := 1
| (n+1) := (B n ^ 2 + 2) / (2 * B n)

theorem problem (n : ℕ) : B (n + 1) = A (2^n) := 
sorry

end problem_l707_707931


namespace pizza_slices_l707_707872

theorem pizza_slices (total_slices slices_with_pepperoni slices_with_mushrooms : ℕ) (h1 : total_slices = 15)
  (h2 : slices_with_pepperoni = 8) (h3 : slices_with_mushrooms = 12)
  (h4 : ∀ slice, slice < total_slices → (slice ∈ {x | x < slices_with_pepperoni} ∨ slice ∈ {x | x < slices_with_mushrooms})) :
  ∃ n : ℕ, (slices_with_pepperoni - n) + (slices_with_mushrooms - n) + n = total_slices ∧ n = 5 :=
by simp [h1, h2, h3]; use 5; linarith; sorry

end pizza_slices_l707_707872


namespace four_digit_palindrome_div_by_11_probability_four_digit_palindrome_div_by_11_l707_707882

-- Definitions and conditions
def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 1001 * a + 110 * b

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Main theorem statement
theorem four_digit_palindrome_div_by_11 :
  ∀ n, is_palindrome n → is_divisible_by_11 n :=
by
  intro n
  intro h
  obtain ⟨a, b, ha1, ha2, ha3, hb1, hb2, hn⟩ := h
  rw [hn]
  sorry

-- Final probability calculation theorem
theorem probability_four_digit_palindrome_div_by_11 :
  ∑ n in finset.range (9000 + 1), if is_palindrome n ∧ is_divisible_by_11 n then 1 else 0 = ∑ n in finset.range (9000 + 1), if is_palindrome n then 1 else 0 :=
by
  sorry

end four_digit_palindrome_div_by_11_probability_four_digit_palindrome_div_by_11_l707_707882


namespace horse_catches_up_l707_707881

-- Definitions based on given conditions
def dog_speed := 20 -- derived from 5 steps * 4 meters
def horse_speed := 21 -- derived from 3 steps * 7 meters
def initial_distance := 30 -- dog has already run 30 meters

-- Statement to be proved
theorem horse_catches_up (d h : ℕ) (time : ℕ) :
  d = dog_speed → h = horse_speed →
  initial_distance = 30 →
  h * time = initial_distance + dog_speed * time →
  time = 600 / (h - d) ∧ h * time - initial_distance = 600 :=
by
  intros
  -- Proof placeholders
  sorry  -- Omit the actual proof steps

end horse_catches_up_l707_707881


namespace neither_sufficient_nor_necessary_l707_707078

theorem neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a + b > 0 → ab > 0) ∧ (ab > 0 → a + b > 0)) :=
by {
  sorry
}

end neither_sufficient_nor_necessary_l707_707078


namespace M_subset_N_l707_707296

variable (M : Set ℝ) (N : Set ℝ)
variable (k : ℤ)

def M_def : M = {x | ∃ k : ℤ, x = ↑k/4 + 1/4 } := by sorry
def N_def : N = {x | ∃ k : ℤ, x = ↑k/8 - 1/4 } := by sorry

theorem M_subset_N (hM : M = {x | ∃ k : ℤ, x = ↑k/4 + 1/4 })
  (hN : N = {x | ∃ k : ℤ, x = ↑k/8 - 1/4 }) :
  M ⊆ N := by sorry

end M_subset_N_l707_707296


namespace systematic_sampling_interval_l707_707095

-- Definitions for the given conditions
def total_students : ℕ := 1203
def sample_size : ℕ := 40

-- Theorem statement to be proven
theorem systematic_sampling_interval (N n : ℕ) (hN : N = total_students) (hn : n = sample_size) : 
  N % n ≠ 0 → ∃ k : ℕ, k = 30 :=
by
  sorry

end systematic_sampling_interval_l707_707095


namespace isosceles_triangle_perimeter_l707_707267

def triangle := {a b c : ℝ} 

def isosceles_triangle (a b c : ℝ) :=
  (a = b ∧ a + b > c) ∨ (a = c ∧ a + c > b) ∨ (b = c ∧ b + c > a)

theorem isosceles_triangle_perimeter
  (a b c : ℝ) 
  (cond1 : isosceles_triangle a b c) 
  (cond2 : (a = 7 ∧ b = 7 ∧ c = 3) ∨ (a = 3 ∧ b = 3 ∧ c = 7)) :
  a + b + c = 17 :=
by
  cases cond2
  . cases cond2
    . contradiction 
    . sorry 
  . sorry 

end isosceles_triangle_perimeter_l707_707267


namespace complex_inequality_equality_condition_l707_707038

open Complex

theorem complex_inequality (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) :
  |z - w| ≥ (1 / 2) * (|z| + |w|) * |z / |z| - w / |w|| :=
by
  sorry

theorem equality_condition (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) :
  |z - w| = (1 / 2) * (|z| + |w|) * |z / |z| - w / |w|| ↔ (|z| = |w| ∨ z / w < 0) :=
by
  sorry

end complex_inequality_equality_condition_l707_707038


namespace number_of_pupils_l707_707169

theorem number_of_pupils (n : ℕ) 
  (incorrect_mark : ℕ = 85) 
  (correct_mark : ℕ = 45) 
  (increase_due_to_wrong_entry : ℕ = incorrect_mark - correct_mark) 
  (average_increase : ℝ = 0.5) 
  (incorrect_entry_effect : n / 2 = increase_due_to_wrong_entry) :
  n = 80 :=
sorry

end number_of_pupils_l707_707169


namespace quartic_poly_has_roots_l707_707951

noncomputable def quartic_polynomial := 
  (λ x : ℝ, x^4 - 10*x^3 + 32*x^2 - 28*x - 8)

theorem quartic_poly_has_roots :
  ∃ (p : ℝ → ℝ), 
    (p = quartic_polynomial) ∧
    (∀ r: ℝ, r ∈ {3 + Real.sqrt 5, 3 - Real.sqrt 5, 2 + Real.sqrt 6, 2 - Real.sqrt 6} → p r = 0) :=
begin
  use quartic_polynomial,
  split,
  { refl },
  { intro r,
    intro hr,
    fin_cases hr,
    sorry, -- Remaining proof steps 
  }
end

end quartic_poly_has_roots_l707_707951


namespace range_of_a_if_q_sufficient_but_not_necessary_for_p_l707_707615

variable {x a : ℝ}

def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

theorem range_of_a_if_q_sufficient_but_not_necessary_for_p :
  (∀ x, q x a → p x) ∧ ¬(∀ x, p x → q x a) → a ∈ Set.Ici 1 := 
sorry

end range_of_a_if_q_sufficient_but_not_necessary_for_p_l707_707615


namespace five_digit_num_prob_div_9_l707_707320

theorem five_digit_num_prob_div_9 
    (N : ℕ) 
    (h1 : ∃ (a b c d e: ℕ), N = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧ a ≠ 0)
    (h2 : ∀ (a b c d e: ℕ), a + b + c + d + e = 30)
    (h3 : N ≥ 10000 ∧ N < 100000) :
    (∑ (x : fin 10), x) % 9 = 0 :=
by
  sorry

end five_digit_num_prob_div_9_l707_707320


namespace verify_conclusions_l707_707806

def squares_table : List (ℝ × ℝ) := [
  (16, 256), (16.1, 259.21), (16.2, 262.44), (16.3, 265.69), 
  (16.4, 268.96), (16.5, 272.25), (16.6, 275.56), (16.7, 278.89), 
  (16.8, 282.24), (16.9, 285.61), (17, 289)
]

theorem verify_conclusions :
  let c1 := sqrt 285.61 = 16.9
  let c2 := sqrt 26896 = 164 ∨ sqrt 26896 = -164
  let c3 := (20 - sqrt 260).floor = 4
  -- Note: floor function in Lean gives the greatest integer less than or equal to the number.
  let c4 := ∃ a b c : ℤ, 16.1 < a ∧ a ≤ 16.2 ∧ 16.1 < b ∧ b ≤ 16.2 ∧ 16.1 < c ∧ c ≤ 16.2
  c1 ∧ c2 ∧ c4 ∧ ¬c3 := by
  sorry

end verify_conclusions_l707_707806


namespace volume_of_sphere_l707_707276

theorem volume_of_sphere (h d : ℝ) (r : ℝ) (v_cylinder v_sphere : ℝ) (ratio : h = 2 * d)
  (cylinder_in_sphere : r = (d / 2) * sqrt (5 : ℝ))
  (cylinder_volume : v_cylinder = 500 * π) :
  v_sphere = (2500 * sqrt (5 : ℝ) / 3) * π :=
by
  sorry

end volume_of_sphere_l707_707276


namespace find_x_y_max_x_y_l707_707327

variable {A : ℝ} {x y : ℝ}
variable {AB AC AO : ℝ → E → ℝ}
variable [InnerProductSpace ℝ E]

-- Conditions
def triangle_conditions (A AB AC AO : ℝ) : Prop :=
  A = 2 * π / 3 ∧
  AB = 1 ∧
  AC = 2

-- Question 1
theorem find_x_y (h : triangle_conditions A AB AC AO) :
  (x = 4 / 3 ∧ y = 5 / 6) :=
sorry

-- Conditions for question 2
def cos_cond (cosA : ℝ) : Prop :=
  cosA = 1 / 3

-- Question 2
theorem max_x_y (hcos : cos_cond (Real.cos A)) :
  (x + y ≤ 3 / 4) :=
sorry

end find_x_y_max_x_y_l707_707327


namespace ellen_painted_roses_l707_707215

theorem ellen_painted_roses :
  ∀ (r : ℕ),
    (5 * 17 + 7 * r + 3 * 6 + 2 * 20 = 213) → (r = 10) :=
by
  intros r h
  sorry

end ellen_painted_roses_l707_707215


namespace probability_of_three_green_marbles_l707_707563

noncomputable def choose_exactly_three_green_in_7_trials : ℚ :=
  (nat.choose 7 3) * ((8/15)^3 * (7/15)^4)

theorem probability_of_three_green_marbles :
  choose_exactly_three_green_in_7_trials = 17210408 / 68343750 :=
by
  -- Add proof steps here
  sorry

end probability_of_three_green_marbles_l707_707563


namespace part1_part2_l707_707653

variable (x m : ℝ)

def A : set ℝ := { x | 1 < x ∧ x < 3 }
def B (m : ℝ) : set ℝ := { x | 2 * m < x ∧ x < 1 - m }

theorem part1 (h : A ∩ B m = A) : m ≤ -2 := by
  sorry

theorem part2 (h : A ∩ B m = ∅) : 0 ≤ m := by
  sorry

end part1_part2_l707_707653


namespace paige_mp3_player_songs_l707_707059

/--
Paige had 11 songs on her mp3 player.
She deleted 9 old songs.
She added 8 new songs.

We are to prove:
- The final number of songs on her mp3 player is 10.
-/
theorem paige_mp3_player_songs (initial_songs deleted_songs added_songs final_songs : ℕ)
  (h₁ : initial_songs = 11)
  (h₂ : deleted_songs = 9)
  (h₃ : added_songs = 8) :
  final_songs = initial_songs - deleted_songs + added_songs :=
by
  sorry

end paige_mp3_player_songs_l707_707059


namespace triangle_PQR_area_l707_707542

-- Definitions for points P, Q, and R
structure Point where
  x : ℝ
  y : ℝ

-- Definitions of lines with given slopes and their intersections with the x-axis
def line1 (P : Point) (slope : ℝ) : Point := {
  x := P.x - P.y / slope,
  y := 0
}

def line2 (P : Point) (slope : ℝ) : Point := {
  x := P.x - P.y / slope,
  y := 0
}

-- Point P where lines intersect
def P : Point := ⟨2, 5⟩

-- Point Q and R are the intersections of the respective lines with the x-axis
def Q : Point := line1 P 3
def R : Point := line2 P 1

-- Function to calculate the area of triangle formed by points P, Q, and R
def triangle_area (P Q R : Point) : ℝ :=
  let base := abs (Q.x - R.x)
  let height := P.y
  (base * height) / 2

theorem triangle_PQR_area : triangle_area P Q R = 25 / 3 := by
  sorry

end triangle_PQR_area_l707_707542


namespace find_a_l707_707783

theorem find_a 
  (f : ℝ → ℝ)
  (a : ℝ) 
  (h : ∀ x, f x = x^3 - a * x^2 + x)
  (h_tangent_parallel : f' 1 = 2) :
  a = 1 := by
  sorry

end find_a_l707_707783


namespace derivative_is_three_l707_707672

-- Define the condition
def f'' (x : ℝ) : ℝ := 3

-- State the proof problem
theorem derivative_is_three (f : ℝ → ℝ) (h : ∀ x, f'' x = 3) :
  ∀ x, ∀ Δx, tendsto (λ Δx, (f(x + Δx) - f(x)) / Δx) (nhds 0) (nhds 3) := by
  sorry

end derivative_is_three_l707_707672


namespace product_of_chords_l707_707718

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 18)

noncomputable def A : ℂ := 4
noncomputable def B : ℂ := -4
noncomputable def C (k : ℕ) : ℂ := 4 * omega^(k)

theorem product_of_chords :
  (∏ k in Finset.range 8, Complex.abs (A - C (k+1))) * 
  (∏ k in Finset.range 8, Complex.abs (B - C (k+1))) = 38654705664 :=
by
  -- problem statement to be proved
  sorry  -- proof goes here

end product_of_chords_l707_707718


namespace player_B_wins_l707_707473

/-- The proof problem equivalent: Player B has a winning strategy in the polynomial game. -/
theorem player_B_wins {n : ℕ} (h : n ≥ 2) :
  ∀ (coefficients : Fin 2n → ℝ),
  ∃ (P : ℝ → ℝ),
  (P = λ x, x ^ (2 * n) + (coefficients 0) * x ^ (2 * n - 1) + (coefficients 1) * x ^ (2 * n - 2) + ... + (coefficients (2n - 2)) * x + 1) ∧
  ∃ r ∈ ℝ, P r = 0 :=
sorry

end player_B_wins_l707_707473


namespace find_number_l707_707129

theorem find_number (x : ℝ) (h : 7 * x = 50.68) : x = 7.24 :=
sorry

end find_number_l707_707129


namespace inscribed_sphere_radius_l707_707536

theorem inscribed_sphere_radius (r_c h_c : ℝ) (r_s : ℝ) (b d : ℝ) (h_c_eq : h_c = 30) (r_c_eq : r_c = 15) (r_s_eq : r_s = b * Real.sqrt d - b) (b_eq : b = 7.5) (d_eq : d = 5) : r_s = 7.5 * Real.sqrt 5 - 7.5 :=
by
  rw [b_eq, d_eq]
  exact r_s_eq.symm

end inscribed_sphere_radius_l707_707536


namespace angle_BAC_eq_108_degrees_l707_707706

-- Definitions for the problem.
variables {A B C D E P : Type*}
variables [triangle_eq A B C] [bisector_eq A D] [bisector_eq B E]
variables (AB AC : A = C) (BE_2AD : B = E) -- AB = AC and BE = 2AD represented

-- Declare the main result to prove
theorem angle_BAC_eq_108_degrees (isosceles_AB_AC : AB = AC)
  (angle_bisector_A : bisector_eq A D)
  (angle_bisector_B : bisector_eq B E)
  (BE_eq_2AD : BE_2AD = 2 * angle_bisector_A) : 
  measure_angle (angle A B C) = 108 :=
begin
  sorry 
end

end angle_BAC_eq_108_degrees_l707_707706


namespace eq_holds_for_n_l707_707580

theorem eq_holds_for_n (n : ℕ) (a b c d : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a + b + c + d = n * Real.sqrt (a * b * c * d) ↔ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by
  sorry

end eq_holds_for_n_l707_707580


namespace intersection_y_coordinate_perpendicular_tangents_l707_707719

theorem intersection_y_coordinate_perpendicular_tangents (a b : ℝ) (ha: (∃ (A : ℝ × ℝ), A = (a, a^3))) 
  (hb: (∃ (B : ℝ × ℝ), B = (b, b^3))) (h_perpendicular: (3 * a^2) * (3 * b^2) = -1) :
  let P : ℝ × ℝ := (a + b, 3 * a^2 * b + a^3) in P.2 = -1 / 3 :=
by
  sorry

end intersection_y_coordinate_perpendicular_tangents_l707_707719


namespace f_at_63_l707_707788

-- Define the function f: ℤ → ℤ with given properties
def f : ℤ → ℤ :=
  sorry -- Placeholder, as we are only stating the problem, not the solution

-- Conditions
axiom f_at_1 : f 1 = 6
axiom f_eq : ∀ x : ℤ, f (2 * x + 1) = 3 * f x

-- The goal is to prove f(63) = 1458
theorem f_at_63 : f 63 = 1458 :=
  sorry

end f_at_63_l707_707788


namespace find_300th_term_excl_squares_l707_707489

def is_perfect_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

def nth_term_excl_squares (n : ℕ) : ℕ :=
  let excluded := (List.range (n + n / 10)).filter (λ x, ¬ is_perfect_square x)
  excluded.nth n

theorem find_300th_term_excl_squares :
  nth_term_excl_squares 299 = 317 :=
by
  sorry

end find_300th_term_excl_squares_l707_707489


namespace seashells_count_l707_707712

theorem seashells_count (j_seashells : ℕ) (jess_seashells : ℕ) (h1 : j_seashells = 6) (h2 : jess_seashells = 8) :
  j_seashells + jess_seashells = 14 :=
by {
  -- Introduce variables and hypotheses
  rw [h1, h2], -- Rewrite using the given values
  exact rfl -- The computation 6 + 8 = 14 is true
}

end seashells_count_l707_707712


namespace measure_angle_BAF_l707_707589

theorem measure_angle_BAF
  (A B C D E F : Type) [equilateral_triangle A B C] [regular_pentagon B C D E F] [coplanar A B C D E F] :
  angle A B F = 6 := 
sorry

end measure_angle_BAF_l707_707589


namespace vertices_labeling_l707_707037

theorem vertices_labeling (n : ℕ) (h : n ≥ 2) :
  ∃ (C : List (Fin (2 ^ n) → Fin (2 : ℕ + 1))) 
  (H1 : ∀ i, (C i).val ∈ {1, 2})
  (H2: ∀ i, (C i).val < 10^n)
  (H3 : ∀ (i j : Fin (2 ^ n)), i ≠ j → C i ≠ C j)
  (H4 : ∀ (i : Fin (2 ^ n)), ∃ j, (i.val + 1) % (2 ^ n) = j.val ∧ (C i).val ≠ (C j).val)
  := 
sorry

end vertices_labeling_l707_707037


namespace number_of_starting_positions_l707_707035

def hyperbola_C (x y : ℝ) : Prop := 2 * y^2 - x^2 = 2

def sequence_of_points (x₀ : ℝ) : ℕ → ℝ
| 0     := x₀
| (n+1) := let x_n := sequence_of_points n in (4 * x_n^2 - 2) / (2 * x_n)

def theta_n (θ₀ : ℝ) (n : ℕ) : ℝ := 2^n * θ₀

theorem number_of_starting_positions : ∀ (x₀ : ℝ), 
  (∃ θ₀ ∈ (0 : ℝ, Real.pi), x₀ = Real.cot θ₀ ∧ 
    sequence_of_points x₀ 1000 = x₀) → 
  ∃ n ∈ ℕ, n = 2^1000 - 2 :=
sorry

end number_of_starting_positions_l707_707035


namespace tangent_line_y_intercept_l707_707784

-- First, we define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

-- Define the derivative of the function f
def f_prime (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - a

-- Define the slope of the line x - y - 1 = 0
def line_slope : ℝ := 1

-- Given condition: the tangent line at x = 1 is parallel to the line, thus it has a slope of 1.
-- Solve for a such that f_prime 1 a = 1
def solve_a : ℝ := 3 - 1

-- Recalculate specific version of f with a = 2
def f_specific (x : ℝ) : ℝ := f x 2

-- Define the tangent line at x = 1 and determine its y-intercept
noncomputable def tangent_line_intercept (x₀ : ℝ) (m : ℝ) (y₀ : ℝ) : ℝ := y₀ - m * x₀

-- Prove the intercept is -2
theorem tangent_line_y_intercept : tangent_line_intercept 1 1 (f_specific 1) = -2 := 
by 
  have a_eq_2 : solve_a = 2 := (by linarith)
  rw [solve_a, a_eq_2]
  sorry -- Placeholder for the complete proof.

end tangent_line_y_intercept_l707_707784


namespace lawn_width_l707_707583

-- Define the conditions as variables/constants
variables (bags : ℕ) (coverage_per_bag : ℕ) (extra_area : ℕ) (length : ℕ)
variables (total_area : ℕ) (actual_area : ℕ) (width : ℕ)

-- Assign specific values based on the problem
def bags := 4
def coverage_per_bag := 250
def extra_area := 208
def length := 22

-- Derived values based on the conditions
def total_area := bags * coverage_per_bag
def actual_area := total_area - extra_area
def width := actual_area / length

-- Theorem statement
theorem lawn_width :
  bags = 4 →
  coverage_per_bag = 250 →
  extra_area = 208 →
  length = 22 →
  total_area = 4 * 250 →
  actual_area = total_area - 208 →
  width = actual_area / 22 →
  width = 36 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end lawn_width_l707_707583


namespace marbles_leftover_l707_707409

theorem marbles_leftover (r p g : ℕ) (hr : r % 7 = 5) (hp : p % 7 = 4) (hg : g % 7 = 2) : 
  (r + p + g) % 7 = 4 :=
by
  sorry

end marbles_leftover_l707_707409


namespace volume_and_surface_area_l707_707699

theorem volume_and_surface_area 
  (AB BC BF : ℝ)
  (hAB : AB = 8) 
  (hBC : BC = 6) 
  (hBF : BF = 4) : 
  (volume_of_polyhedron AB BC BF = 96) ∧ 
  (surface_area_of_polyhedron AB BC BF = 139.4) :=
by
  -- Definitions of the formulae would go here
  sorry

end volume_and_surface_area_l707_707699


namespace integral_of_2x_plus_exp_x_l707_707574

open Real

theorem integral_of_2x_plus_exp_x :
  ∫ x in -1..1, (2 * x + exp x) = exp 1 - exp (-1) := 
sorry

end integral_of_2x_plus_exp_x_l707_707574


namespace mod_equiv_n_l707_707824

theorem mod_equiv_n (n : ℤ) : 0 ≤ n ∧ n < 9 ∧ -1234 % 9 = n := 
by
  sorry

end mod_equiv_n_l707_707824


namespace balanced_ternary_nonnegative_count_l707_707192

theorem balanced_ternary_nonnegative_count :
  let n := 8
  let coefficients := {-1, 0, 1}
  let max_val := ∑ i in finRange(f n), (coefficients i * 3^i).abs
  let nonneg_val := {x | ∃ (a_i : ℕ → ℤ), ∀ i, i ≤ n → a_i i ∈ coefficients ∧ x = ∑ i in finRange(f n), a_i i * 3^i}
  nonneg_val.count = 9842 :=
  sorry

end balanced_ternary_nonnegative_count_l707_707192


namespace inequality_theta_range_l707_707616

theorem inequality_theta_range (k : ℤ) (x θ : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) :
  (2 * k * ℝ.pi + ℝ.pi / 12 < θ ∧ θ < 2 * k * ℝ.pi + 5 * ℝ.pi / 12) →
  x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0 :=
by {
  sorry
}

end inequality_theta_range_l707_707616


namespace exists_points_same_color_one_meter_apart_l707_707189

-- Predicate to describe points in the 2x2 square
structure Point where
  x : ℝ
  y : ℝ
  h_x : 0 ≤ x ∧ x ≤ 2
  h_y : 0 ≤ y ∧ y ≤ 2

-- Function to describe the color assignment
def color (p : Point) : Prop := sorry -- True = Black, False = White

-- The main theorem to be proven
theorem exists_points_same_color_one_meter_apart :
  ∃ p1 p2 : Point, color p1 = color p2 ∧ dist (p1.1, p1.2) (p2.1, p2.2) = 1 :=
by
  sorry

end exists_points_same_color_one_meter_apart_l707_707189


namespace find_n_l707_707234

theorem find_n :
  (∑ k in finset.range (n + 1), 1 / (real.sqrt k + real.sqrt (k + 1))) = 2013 →
  n = 4056195 :=
by
  -- The proof steps would go here.
  sorry

end find_n_l707_707234


namespace range_of_a_l707_707244

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
if x > 0 then 3 / x + a 
else x^2 + 1

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (f x1 a) / (x1 - 1) = -2 ∧ (f x2 a) / (x2 - 1) = -2 ∧ (f x3 a) / (x3 - 1) = -2)
  ↔ a ∈ set.Ioo (-∞) (-3) ∪ set.Ioo (-3) (2 - 2 * real.sqrt 6) :=
by sorry

end range_of_a_l707_707244


namespace probability_three_digit_l707_707071

open Set

noncomputable def S : Set ℕ := {n | 60 ≤ n ∧ n ≤ 1000}
noncomputable def T : Set ℕ := {n | n ∈ S ∧ 100 ≤ n ∧ n ≤ 999}

theorem probability_three_digit :
  |T| / |S| = 901 / 941 :=
sorry

end probability_three_digit_l707_707071


namespace ellipse_eccentricity_l707_707375

theorem ellipse_eccentricity (a b : ℝ) (P F₁ F₂ : ℝ × ℝ)
  (ha : a > b) (hb : b > 0)
  (hP₁ : P ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1})
  (hP₂ : P ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 3 * b^2})
  (hF₁F₂ : distance P F₁ = 3 * distance P F₂) : 
  sqrt 14 / 4 = (sqrt 7 / 4) := -- Please correct this if necessary.
begin
  sorry
end

end ellipse_eccentricity_l707_707375


namespace fraction_filled_l707_707892

-- Definitions for the given conditions
variables (x C : ℝ) (h₁ : 20 * x / 3 = 25 * C / 5) 

-- The goal is to show that x / C = 3 / 4
theorem fraction_filled (h₁ : 20 * x / 3 = 25 * C / 5) : x / C = 3 / 4 :=
by sorry

end fraction_filled_l707_707892


namespace exists_sum_and_sum_of_squares_lt_l707_707804

theorem exists_sum_and_sum_of_squares_lt :
  ∃ (n : ℕ) (x : Fin n → ℝ), (∑ i, x i = 10) ∧ (∑ i, (x i)^2 < 0.2) :=
  sorry

end exists_sum_and_sum_of_squares_lt_l707_707804


namespace number_of_lines_with_8_points_in_grid_l707_707869

def point := (ℤ × ℤ × ℤ)

def in_grid (p : point) : Prop :=
  let (x, y, z) := p
  1 ≤ x ∧ x ≤ 10 ∧ 1 ≤ y ∧ y ≤ 10 ∧ 1 ≤ z ∧ z ≤ 10

def line_contains_exactly_n_points (start : point) (direction : point) (n : ℕ) : Prop :=
  ∃ t : ℤ, ∀i, 0 ≤ i ∧ i < n → let p := (start.1 + i * direction.1, start.2 + i * direction.2, start.3 + i * direction.3)
  in in_grid p ∧ ¬ in_grid (start.1 + n * direction.1, start.2 + n * direction.2, start.3 + n * direction.3)

theorem number_of_lines_with_8_points_in_grid :
  ∃ count, count = 168 ∧
    ∃ lines :
      list (point × point),
      (∀ line ∈ lines, line_contains_exactly_n_points line.1 line.2 8) ∧
      count = lines.length := sorry

end number_of_lines_with_8_points_in_grid_l707_707869


namespace sum_of_fractions_l707_707835

theorem sum_of_fractions :
  (∑ n in Finset.range 2010, (2 / ((n + 1) * ((n + 1) + 3)))) = 1.499 :=
by
  sorry

end sum_of_fractions_l707_707835


namespace units_digit_7_pow_62_l707_707454

theorem units_digit_7_pow_62 : (7 ^ 62) % 10 = 9 := 
by 
  have cycle : [7 % 10, 7^2 % 10, 7^3 % 10, 7^4 % 10] = [7, 9, 3, 1] := by native_decide
  have pattern : (7 ^ (4 * n + r)) % 10 = ([7, 9, 3, 1]).nth (r % 4) for n r := 
    by sorry
  calc (7 ^ 62) % 10 = (7 ^ (4 * 15 + 2)) % 10 := by congr; ring 
               ...     = ([7, 9, 3, 1]).nth (2 % 4) := by apply pattern
               ...     = 9 := rfl

end units_digit_7_pow_62_l707_707454


namespace relative_error_comparison_l707_707183

variable (e1 l1 e2 l2 : ℝ)

def relative_error (e l : ℝ) : ℝ := (e / l) * 100

theorem relative_error_comparison
  (e1_eq : e1 = 0.05)
  (l1_eq : l1 = 20)
  (e2_eq : e2 = 0.25)
  (l2_eq : l2 = 80) :
  relative_error e2 l2 > relative_error e1 l1 :=
by {
  rw [relative_error, relative_error, e1_eq, l1_eq, e2_eq, l2_eq],
  norm_num,
  sorry,
}

end relative_error_comparison_l707_707183


namespace nth_term_arithmetic_seq_l707_707628

variable (a_n : Nat → Int)
variable (S : Nat → Int)
variable (a_1 : Int)

-- Conditions
def is_arithmetic_sequence (a_n : Nat → Int) : Prop :=
  ∃ d : Int, ∀ n : Nat, a_n (n + 1) = a_n n + d

def first_term (a_1 : Int) : Prop :=
  a_1 = 1

def sum_first_three_terms (S : Nat → Int) : Prop :=
  S 3 = 9

theorem nth_term_arithmetic_seq :
  (is_arithmetic_sequence a_n) →
  (first_term 1) →
  (sum_first_three_terms S) →
  ∀ n : Nat, a_n n = 2 * n - 1 :=
  sorry

end nth_term_arithmetic_seq_l707_707628


namespace TruckY_average_speed_is_63_l707_707110

noncomputable def average_speed_TruckY (initial_gap : ℕ) (extra_distance : ℕ) (hours : ℕ) (distance_X_per_hour : ℕ) : ℕ :=
  let distance_X := distance_X_per_hour * hours
  let total_distance_Y := distance_X + initial_gap + extra_distance
  total_distance_Y / hours

theorem TruckY_average_speed_is_63 
  (initial_gap : ℕ := 14) 
  (extra_distance : ℕ := 4) 
  (hours : ℕ := 3)
  (distance_X_per_hour : ℕ := 57) : 
  average_speed_TruckY initial_gap extra_distance hours distance_X_per_hour = 63 :=
by
  -- Proof goes here
  sorry

end TruckY_average_speed_is_63_l707_707110


namespace permutation_equality_l707_707679

variables {n : ℕ}
variables (a : Fin n.succ → ℕ)

-- The given conditions as hypotheses
def is_permutation (a : Fin n.succ → ℕ) : Prop :=
  ∀ i : Fin n.succ, ∃ j : Fin n.succ, a j = i.val.succ

def satisfies_condition (a : Fin n.succ → ℕ) : Prop :=
  ∀ k : Fin n, (a k)^2 / a (k.succ) ≤ k.val.succ + 2

-- Statement of the proof problem
theorem permutation_equality 
  (h_perm : is_permutation a)
  (h_cond : satisfies_condition a) 
  : ∀ k : Fin n.succ, a k = k.val.succ :=
begin
  sorry
end

end permutation_equality_l707_707679


namespace percentage_of_boys_and_additional_boys_l707_707687

theorem percentage_of_boys_and_additional_boys (total_students : ℕ) (boys_ratio : ℕ) (girls_ratio : ℕ)
  (total_students_eq : total_students = 42) (ratio_condition : boys_ratio = 3 ∧ girls_ratio = 4) :
  let total_groups := total_students / (boys_ratio + girls_ratio)
  let total_boys := boys_ratio * total_groups
  (total_boys * 100 / total_students = 300 / 7) ∧ (21 - total_boys = 3) :=
by {
  sorry
}

end percentage_of_boys_and_additional_boys_l707_707687


namespace area_quadrilateral_FDBG_l707_707703

variable {A B C D E F G : Type} [RealAngle A] [RealSegment B] [RealSegment C] [RealPoint D]
variables (AB AC : ℝ) (area_ABC : ℝ) (one_third_AD : RealSegment → RealPoint) 
variables (midpoint_AC : RealSegment → RealPoint)
variables (angle_bisector_BAC : RealSegment → RealPoint → RealSegment) 
variables (intersection : RealPoint → RealSegment → RealPoint)

def triangle_area := (s: RealTriangle) → ℝ

-- Conditions
variables (AB_length : AB = 40)
variables (AC_length : AC = 20)
variables (area_ABC_value : area_ABC = 200)
variables (D_def : D = one_third_AD AB)
variables (E_def : E = midpoint_AC AC)
variables (F_def : F = intersection (angle_bisector_BAC A C) (RealSegment DE))
variables (G_def : G = intersection (angle_bisector_BAC A C) (RealSegment BC))

-- Problem Statement
theorem area_quadrilateral_FDBG :
  triangle_area (RealQuadrilateral F D B G) = 140 :=
by sorry

end area_quadrilateral_FDBG_l707_707703


namespace inexperienced_sailors_count_l707_707551

theorem inexperienced_sailors_count
  (I E : ℕ)
  (h1 : I + E = 17)
  (h2 : ∀ (rate_inexperienced hourly_rate experienced_rate : ℕ), hourly_rate = 10 → experienced_rate = 12 → rate_inexperienced = 2400)
  (h3 : ∀ (total_income experienced_salary : ℕ), total_income = 34560 → experienced_salary = 2880)
  (h4 : ∀ (monthly_income : ℕ), monthly_income = 34560)
  : I = 5 := sorry

end inexperienced_sailors_count_l707_707551


namespace domain_ell_eq_l707_707122

def ell (y : ℝ) : ℝ := 1 / ((y - 2) + (y - 8))

theorem domain_ell_eq : 
  {x : ℝ | ∃ y : ℝ, ell y = x} = {y : ℝ | y ≠ 5} := 
by
  sorry

end domain_ell_eq_l707_707122


namespace find_a_value_l707_707316

namespace Proof

-- Define the context and variables
variables (a b c : ℝ)
variables (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
variables (h2 : a * 15 * 2 = 4)

-- State the theorem we want to prove
theorem find_a_value: a = 6 :=
by
  sorry

end Proof

end find_a_value_l707_707316


namespace ball_returns_to_Ami_after_14_throws_l707_707812

def next_position (current : ℕ) (step_size : ℕ) (total : ℕ) : ℕ :=
  if current + step_size > total then current + step_size - total else current + step_size

theorem ball_returns_to_Ami_after_14_throws :
  ∀ (n step_size : ℕ), n = 13 → step_size = 5 → 
    let positions := (list.range n).map (λ i, nat.iterate (next_position · step_size n) (i+1) 14) in
    positions.head = positions.get_last :=
sorry

end ball_returns_to_Ami_after_14_throws_l707_707812


namespace cupboard_cost_price_l707_707520

theorem cupboard_cost_price (C : ℕ)
  (h1 : ∃ SP, SP = 0.84 * C)
  (h2 : ∃ NSP, NSP = 1.16 * C)
  (h3 : ∀ SP NSP, NSP - SP = 1800) : C = 5625 := by
  sorry

end cupboard_cost_price_l707_707520


namespace commuting_time_difference_l707_707925

noncomputable def distance_to_work : ℝ := 1.5
noncomputable def walking_speed : ℝ := 3
noncomputable def train_speed : ℝ := 20
noncomputable def additional_train_time : ℝ := 23.5

theorem commuting_time_difference :
  let time_walking := (distance_to_work / walking_speed) * 60,
      time_train_travel := (distance_to_work / train_speed) * 60,
      total_time_train := time_train_travel + additional_train_time
  in time_walking - total_time_train = 2 :=
by {
  sorry
}

end commuting_time_difference_l707_707925


namespace collinear_A_M_O_l707_707756

variables {A B C K L M O : Point}

-- Conditions
variables (hK : K ∈ line_segment A B)
          (hL : L ∈ line_segment A C)
          (hParallelKLBC : parallel KL BC)
          (hPerpKM : perp_to_line KM AB)
          (hPerpLM : perp_to_line LM AC)
          (hCircumcenterO : circumcenter ABC O)

-- Theorem statement
theorem collinear_A_M_O :
  collinear {A, M, O} :=
sorry

end collinear_A_M_O_l707_707756


namespace residues_exponent_residues_divides_p_minus_one_primitive_roots_phi_l707_707849

noncomputable def phi (n : ℕ) : ℕ := Nat.totient n

theorem residues_exponent (p : ℕ) (d : ℕ) [hp : Fact (Nat.Prime p)] (hd : d ∣ p - 1) : 
  ∃ (S : Finset ℕ), S.card = phi d ∧ ∀ x ∈ S, x^d % p = 1 :=
by sorry

theorem residues_divides_p_minus_one (p : ℕ) (d : ℕ) [hp : Fact (Nat.Prime p)] (hd : d ∣ p - 1) : 
  ∃ (S : Finset ℕ), S.card = phi d :=
by sorry
  
theorem primitive_roots_phi (p : ℕ) [hp : Fact (Nat.Prime p)] : 
  ∃ (S : Finset ℕ), S.card = phi (p-1) ∧ ∀ g ∈ S, IsPrimitiveRoot g p :=
by sorry

end residues_exponent_residues_divides_p_minus_one_primitive_roots_phi_l707_707849


namespace garage_sale_records_l707_707410

/--
Roberta started off with 8 vinyl records. Her friends gave her 12
records for her birthday and she bought some more at a garage
sale. It takes her 2 days to listen to 1 record. It will take her
100 days to listen to her record collection. Prove that she bought
30 records at the garage sale.
-/
theorem garage_sale_records :
  let initial_records := 8
  let gift_records := 12
  let days_per_record := 2
  let total_listening_days := 100
  let total_records := total_listening_days / days_per_record
  let records_before_sale := initial_records + gift_records
  let records_bought := total_records - records_before_sale
  records_bought = 30 := 
by
  -- Variable assumptions
  let initial_records := 8
  let gift_records := 12
  let days_per_record := 2
  let total_listening_days := 100

  -- Definitions
  let total_records := total_listening_days / days_per_record
  let records_before_sale := initial_records + gift_records
  let records_bought := total_records - records_before_sale

  -- Conclusion to prove
  show records_bought = 30
  sorry

end garage_sale_records_l707_707410


namespace percentage_increase_100_80_l707_707148

theorem percentage_increase_100_80 : (100 - 80) / 80 * 100 = 25 := by
  calc
    (100 - 80) / 80 * 100 = 20 / 80 * 100 : by rw [sub_eq_add_neg, add_neg_cancel_right]
    ... = (20 / 80 : ℝ) * 100 : by norm_cast
    ... = (1 / 4) * 100 : by norm_num
    ... = 25 : by norm_num

end percentage_increase_100_80_l707_707148


namespace cosine_of_angle_l707_707300

variables (x : ℝ)

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (1 - x, x)
def vector_c : ℝ × ℝ := (-3 * x, 3 * x)

def parallel_condition : Prop := vector_a.1 * vector_b.2 = vector_a.2 * vector_b.1

def cosine_between (v w : ℝ × ℝ) : ℝ :=
(v.1 * w.1 + v.2 * w.2) / (real.sqrt (v.1^2 + v.2^2) * real.sqrt (w.1^2 + w.2^2))

theorem cosine_of_angle (h : parallel_condition x) : cosine_between (vector_b x) (vector_c x) = - real.sqrt 10 / 10 := sorry

end cosine_of_angle_l707_707300


namespace correct_condition_l707_707130

section proof_problem

variable (a : ℝ)

def cond1 : Prop := (a ^ 6 / a ^ 3 = a ^ 2)
def cond2 : Prop := (2 * a ^ 2 + 3 * a ^ 3 = 5 * a ^ 5)
def cond3 : Prop := (a ^ 4 * a ^ 2 = a ^ 8)
def cond4 : Prop := ((-a ^ 3) ^ 2 = a ^ 6)

theorem correct_condition : cond4 a :=
by
  sorry

end proof_problem

end correct_condition_l707_707130


namespace not_f3_eq_3_l707_707648

theorem not_f3_eq_3 
  (a : ℝ) (b c : ℤ)
  (h : (a : ℤ)∗ -8 ∗ a - b + c =2 ) : 8 ∗ a + 3∗ b + c != 3 :=
   by {
sorry
}

end not_f3_eq_3_l707_707648


namespace find_n_values_l707_707596

-- Define a function that calculates the polynomial expression
def prime_expression (n : ℕ) : ℕ :=
  n^4 - 27 * n^2 + 121

-- State the problem as a theorem
theorem find_n_values (n : ℕ) (h : Nat.Prime (prime_expression n)) : n = 2 ∨ n = 5 :=
  sorry

end find_n_values_l707_707596


namespace sum_of_solutions_l707_707834

theorem sum_of_solutions : 
  ∑ x in Finset.filter (λ x => 0 < x ∧ x ≤ 30 ∧ (7 * (5 * x - 3) ≡ 35 [MOD 12])) (Finset.range 31), x = 48 :=
by
  sorry

end sum_of_solutions_l707_707834


namespace does_not_satisfy_differential_equation_l707_707767

noncomputable def y (x : ℝ) : ℝ :=
  (x + 1) * Real.exp (x ^ 2)

def y_prime (x : ℝ) : ℝ :=
  Real.exp (x ^ 2) * (1 + 2 * x * (x + 1))

theorem does_not_satisfy_differential_equation (x : ℝ) :
  y_prime x - 2 * x * y x ≠ 2 * x * Real.exp (x ^ 2) := by
  sorry

end does_not_satisfy_differential_equation_l707_707767


namespace lucas_150_mod_9_l707_707083

-- Define the Lucas sequence recursively
def lucas (n : ℕ) : ℕ :=
  match n with
  | 0 => 1 -- Since L_1 in the sequence provided is actually the first Lucas number (index starts from 1)
  | 1 => 3
  | (n + 2) => lucas n + lucas (n + 1)

-- Define the theorem for the remainder when the 150th term is divided by 9
theorem lucas_150_mod_9 : lucas 149 % 9 = 3 := by
  sorry

end lucas_150_mod_9_l707_707083


namespace probability_valid_pairings_l707_707429

theorem probability_valid_pairings (m n : ℕ) (h_rel_prime : Nat.coprime m n)
  (h_prob : (no_bad_pairing_probability 10 5) = (m / n)) :
  m + n = 28 :=
sorry

end probability_valid_pairings_l707_707429


namespace shaded_area_is_20_l707_707221

-- Represents the square PQRS with the necessary labeled side lengths
noncomputable def square_side_length : ℝ := 8

-- Represents the four labeled smaller squares' positions and their side lengths
noncomputable def smaller_square_side_lengths : List ℝ := [2, 2, 2, 6]

-- The coordinates or relations to describe their overlaying positions are not needed for the proof.

-- Define the calculated areas from the solution steps
noncomputable def vertical_rectangle_area : ℝ := 6 * 2
noncomputable def horizontal_rectangle_area : ℝ := 6 * 2
noncomputable def overlap_area : ℝ := 2 * 2

-- The total shaded T-shaped region area calculation
noncomputable def total_shaded_area : ℝ := vertical_rectangle_area + horizontal_rectangle_area - overlap_area

-- Theorem statement to prove the area of the T-shaped region is 20
theorem shaded_area_is_20 : total_shaded_area = 20 :=
by
  -- Proof steps are not required as per the instruction.
  sorry

end shaded_area_is_20_l707_707221


namespace unique_quantities_not_determinable_l707_707126

noncomputable def impossible_to_determine_unique_quantities 
(x y : ℝ) : Prop :=
  let acid1 := 54 * 0.35
  let acid2 := 48 * 0.25
  ∀ (final_acid : ℝ), ¬(0.35 * x + 0.25 * y = final_acid ∧ final_acid = 0.75 * (x + y))

theorem unique_quantities_not_determinable :
  impossible_to_determine_unique_quantities 54 48 :=
by
  sorry

end unique_quantities_not_determinable_l707_707126


namespace quadrilateral_lines_concurrent_l707_707065

noncomputable theory
open_locale classical

variables {A B C D E F G H : Type} [incircle : tangent_points A B C D E F G H]

theorem quadrilateral_lines_concurrent
  {A B C D : Point} {E F G H : Point}
  (hA : is_tangent E A B) (hB : is_tangent F B C)
  (hC : is_tangent G C D) (hD : is_tangent H D A) 
  (h_inc : incircle E F G H) :
  concurrent (Line.through A C) (Line.through B D) (Line.through H F) (Line.through G E) :=
sorry

end quadrilateral_lines_concurrent_l707_707065


namespace perimeter_of_square_D_l707_707425

-- Definition of the perimeter of square C
def perimeter_C := 40
-- Definition of the area of square D in terms of the area of square C
def area_C := ((perimeter_C / 4) ^ 2)
def area_D := area_C / 3
-- Define the side of square D in terms of its area
def side_D := Real.sqrt area_D
-- Prove the perimeter of square D
def perimeter_D := 4 * side_D

-- Statement to prove the perimeter of square D equals the given value
theorem perimeter_of_square_D :
  perimeter_D = 40 * Real.sqrt 3 / 3 :=
by
  sorry

end perimeter_of_square_D_l707_707425


namespace stratified_sampling_teachers_l707_707897

theorem stratified_sampling_teachers :
  ∀ (total_teachers senior_teachers intermediate_teachers junior_teachers sample_size : ℕ),
  total_teachers = 300 →
  senior_teachers = 90 →
  intermediate_teachers = 150 →
  junior_teachers = 60 →
  sample_size = 40 →
  let ratio := sample_size / total_teachers in
  let senior_sample := ratio * senior_teachers in
  let intermediate_sample := ratio * intermediate_teachers in
  let junior_sample := ratio * junior_teachers in
  senior_sample = 12 ∧ intermediate_sample = 20 ∧ junior_sample = 8 :=
by
  intros total_teachers senior_teachers intermediate_teachers junior_teachers sample_size h_total h_senior h_intermediate h_junior h_sample
  let ratio := sample_size / total_teachers
  let senior_sample := ratio * senior_teachers
  let intermediate_sample := ratio * intermediate_teachers
  let junior_sample := ratio * junior_teachers
  sorry

end stratified_sampling_teachers_l707_707897


namespace ines_money_left_l707_707008

theorem ines_money_left (P C B D T_disc T_f : ℝ) :
  P = 3 * 2 →
  C = 2 * 3.5 →
  B = 4 * 1.25 →
  let T := P + C + B in
  T > 10 →
  D = 0.1 * T →
  T_disc = T - D →
  let tax := 0.05 * T_disc in
  T_f = T_disc + tax →
  20 - T_f = 2.99 :=
by
  intros hP hC hB hT_gt_10 hD hT_disc hT_f
  sorry

end ines_money_left_l707_707008


namespace number_of_valid_arrangements_l707_707990

def possible_numbers : Finset ℕ := {0, 1, 2, 3, 4, 5}
def required_differences : Finset ℕ := {1, 2, 3, 4, 5}

noncomputable def chosen_numbers (s: Finset ℕ) : Prop :=
  s ⊆ possible_numbers ∧ s.card = 4 ∧ 0 ∈ s ∧ 5 ∈ s

noncomputable def valid_differences (s: Finset ℕ) : Finset ℕ :=
  s.bUnion (λ a, s.erase a.image (λ b, abs (a - b)))

noncomputable def count_valid_arrangements : ℕ :=
  (possible_numbers.powerset.filter (λ s, chosen_numbers s ∧ required_differences ⊆ valid_differences s)).card

theorem number_of_valid_arrangements : count_valid_arrangements = 32 :=
  sorry

end number_of_valid_arrangements_l707_707990


namespace squareD_perimeter_l707_707420

-- Let perimeterC be the perimeter of square C
def perimeterC : ℝ := 40

-- Let sideC be the side length of square C
def sideC := perimeterC / 4

-- Let areaC be the area of square C
def areaC := sideC * sideC

-- Let areaD be the area of square D, which is one-third the area of square C
def areaD := (1 / 3) * areaC

-- Let sideD be the side length of square D
def sideD := Real.sqrt areaD

-- Let perimeterD be the perimeter of square D
def perimeterD := 4 * sideD

-- The theorem to prove
theorem squareD_perimeter (h : perimeterC = 40) (h' : areaD = (1 / 3) * areaC) : perimeterD = (40 * Real.sqrt 3) / 3 := by
  sorry

end squareD_perimeter_l707_707420


namespace four_digit_even_numbers_l707_707119

theorem four_digit_even_numbers :
  let digits := {0, 1, 2, 3, 4, 5}
  let num_even_numbers := 156
  ∃ f : Fin 156 → Finset ℕ, (∀ n ∈ digits, ∃ m ∈ digits, f(n) = {m}) ∧ (∀ x ∈ (f' n), f x = real.even) → num_even_numbers = 156 := by
  sorry

end four_digit_even_numbers_l707_707119


namespace slices_with_both_toppings_l707_707873

theorem slices_with_both_toppings
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (all_with_topping : total_slices = 15 ∧ pepperoni_slices = 8 ∧ mushroom_slices = 12 ∧ ∀ i, i < 15 → (i < 8 ∨ i < 12)) :
  ∃ n, (pepperoni_slices - n) + (mushroom_slices - n) + n = total_slices ∧ n = 5 :=
by
  sorry

end slices_with_both_toppings_l707_707873


namespace angle_EMF_90_deg_l707_707735

theorem angle_EMF_90_deg (S1 S2 : Circle) (A B C D M N K E F : Point)
    (h1 : S1 ∩ S2 = {A, B})
    (h2 : A ∈ line (C, D) ∧ C ∈ S1 ∧ D ∈ S2)
    (h3 : M ∈ segment (C, D))
    (h4 : N ∈ segment (B, C))
    (h5 : K ∈ segment (B, D))
    (h6 : parallel (line (M, N)) (line (B, D)))
    (h7 : parallel (line (M, K)) (line (B, C)))
    (h8 : perpendicular (line (N, E)) (line (B, C)) ∧ E ∈ S1)
    (h9 : perpendicular (line (K, F)) (line (B, D)) ∧ F ∈ S2)
    (h10 : opposite_sides A E (line (B, C)))
    (h11 : opposite_sides A F (line (B, D))) :
    ∠EMF = 90° := by
  sorry

end angle_EMF_90_deg_l707_707735


namespace find_CD_l707_707222

theorem find_CD (C D : ℚ) :
  (∀ x : ℚ, x ≠ 12 ∧ x ≠ -3 → (7 * x - 4) / (x ^ 2 - 9 * x - 36) = C / (x - 12) + D / (x + 3))
  → C = 16 / 3 ∧ D = 5 / 3 :=
by
  sorry

end find_CD_l707_707222


namespace simplify_expression_l707_707414

theorem simplify_expression (x : ℝ) (h1 : ∀ (θ : ℝ), cot θ - 2 * cot (2 * θ) = tan θ) :
  tan x + 2 * tan (2 * x) + 4 * tan (4 * x) + 8 * tan (8 * x) = cot x :=
by sorry

end simplify_expression_l707_707414


namespace find_n_mod_l707_707228

theorem find_n_mod (n : ℤ) (h_cond : 0 ≤ n ∧ n ≤ 12) (h_mod : n ≡ -2050 [MOD 13]) : n = 4 :=
by
  sorry

end find_n_mod_l707_707228


namespace min_variance_new_sample_data_l707_707626

theorem min_variance_new_sample_data 
  (original_set : Fin 8 → ℝ)
  (h_avg_original : (∑ i, original_set i) / 8 = 8)
  (h_var_original : (∑ i, (original_set i - 8)^2) / 8 = 12)
  (x y : ℝ)
  (h_sum : (∑ i, original_set i) + x + y = 90)
  (h_avg_new : ((∑ i, original_set i) + x + y) / 10 = 9) :
  (∑ i, (original_set i - 9)^2 + (x - 9)^2 + (y - 9)^2) / 10 ≥ 13.6 := 
sorry

end min_variance_new_sample_data_l707_707626


namespace youseff_distance_l707_707135

theorem youseff_distance (x : ℕ) 
  (walk_time_per_block : ℕ := 1)
  (bike_time_per_block_secs : ℕ := 20)
  (time_difference : ℕ := 12) :
  (x : ℕ) = 18 :=
by
  -- walking time
  let walk_time := x * walk_time_per_block
  
  -- convert bike time per block to minutes
  let bike_time_per_block := (bike_time_per_block_secs : ℚ) / 60

  -- biking time
  let bike_time := x * bike_time_per_block

  -- set up the equation for time difference
  have time_eq := walk_time - bike_time = time_difference
  
  -- from here, the actual proof steps would follow, 
  -- but we include "sorry" as a placeholder since the focus is on the statement.
  sorry

end youseff_distance_l707_707135


namespace general_term_a_n_sum_of_b_n_l707_707249

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n * a (n + 1) = 2^(2*n + 1)

noncomputable def a_n (n : ℕ) : ℝ := 2^n
noncomputable def b_n (n : ℕ) : ℝ := real.log (a_n n) / real.log 2

theorem general_term_a_n (a : ℕ → ℝ) :
  geometric_sequence a → ∀ n, a n = 2^n :=
by
  intros h n
  sorry -- Proof of a_n being 2^n

theorem sum_of_b_n (n : ℕ) :
  ∑ i in finset.range (2*n + 1), b_n (i + 1) = 2*n^2 + 3*n + 1 :=
by
  sorry -- Proof of the sum of b_n terms

end general_term_a_n_sum_of_b_n_l707_707249


namespace points_opposite_sides_of_circle_l707_707043

noncomputable def general_position (points : set (ℝ × ℝ)) := 
  ∀ A B C D E : (ℝ × ℝ), 
  A ∈ points → B ∈ points → C ∈ points → D ∈ points → E ∈ points → 
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ 
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ 
  (C ≠ D) ∧ (C ≠ E) ∧ (D ≠ E) ∧ 
  ¬ collinear ℝ [A, B, C] ∧ 
  ¬ cocircular ℝ [A, B, C, D] ∧ 
  ¬ cocircular ℝ [A, B, C, E] ∧ 
  ¬ cocircular ℝ [A, B, D, E] ∧ 
  ¬ cocircular ℝ [A, C, D, E] ∧ 
  ¬ cocircular ℝ [B, C, D, E]

theorem points_opposite_sides_of_circle 
(A B C D E : (ℝ × ℝ)) (h : general_position {A, B, C, D, E}) : 
  ∃ (P Q : (ℝ × ℝ)), 
    P ∈ {C, D, E} → Q ∈ {C, D, E} → 
    P ≠ Q → 
    ∃ (R S T : (ℝ × ℝ)), 
      R ∈ {A, B, C, D, E} → 
      S ∈ {A, B, C, D, E} → 
      T ∈ {A, B, C, D, E} → 
      circle ℝ R S T :=
      sorry

end points_opposite_sides_of_circle_l707_707043


namespace find_a_l707_707315

theorem find_a (a r : ℝ) (h1 : a * r = 24) (h2 : a * r^4 = 3) : a = 48 :=
sorry

end find_a_l707_707315


namespace count_integers_in_abs_inequality_l707_707666

theorem count_integers_in_abs_inequality : 
  (set.count (set_of (λ x : ℤ, abs (x - 3) ≤ 6))) = 13 :=
by sorry

end count_integers_in_abs_inequality_l707_707666


namespace range_of_a_l707_707722

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, ((1 / 2) ^ |x - 1|) < a

def q : Prop := ∀ x : ℝ, ax^2 + (a-2)x + 9/8 > 0

theorem range_of_a (hp : p ∨ q) (hnp : ¬ (p ∧ q)) : (1/2 < a ∧ a ≤ 1) ∨ (8 ≤ a) :=
  sorry

end range_of_a_l707_707722


namespace wendy_sold_9_pastries_l707_707986

def pastries_made (cupcakes cookies : Nat) : Nat := 
  cupcakes + cookies

def pastries_sold (total_past leftover_past : Nat) : Nat := 
  total_past - leftover_past

theorem wendy_sold_9_pastries : 
  ∀ (cupcakes cookies leftover total_sold : Nat), 
  cupcakes = 4 → 
  cookies = 29 → 
  leftover = 24 → 
  pastries_made cupcakes cookies = 33 → 
  pastries_sold 33 leftover = 9 := 
by
  intros cupcakes cookies leftover total_sold H1 H2 H3 H4 H5
  sorry

end wendy_sold_9_pastries_l707_707986


namespace find_A_minus_C_l707_707795

theorem find_A_minus_C (A B C : ℤ) 
  (h1 : A = B - 397)
  (h2 : A = 742)
  (h3 : B = C + 693) : 
  A - C = 296 :=
by
  sorry

end find_A_minus_C_l707_707795


namespace solve_for_y_l707_707075

theorem solve_for_y (y : ℕ) (h : 2^y + 8 = 4 * 2^y - 40) : y = 4 :=
by
  sorry

end solve_for_y_l707_707075


namespace child_l707_707888

noncomputable def child's_ticket_cost : ℕ :=
  let adult_ticket_price := 7
  let total_tickets := 900
  let total_revenue := 5100
  let childs_tickets_sold := 400
  let adult_tickets_sold := total_tickets - childs_tickets_sold
  let total_adult_revenue := adult_tickets_sold * adult_ticket_price
  let total_child_revenue := total_revenue - total_adult_revenue
  let child's_ticket_price := total_child_revenue / childs_tickets_sold
  child's_ticket_price

theorem child's_ticket_cost_is_4 : child's_ticket_cost = 4 :=
by
  have adult_ticket_price := 7
  have total_tickets := 900
  have total_revenue := 5100
  have childs_tickets_sold := 400
  have adult_tickets_sold := total_tickets - childs_tickets_sold
  have total_adult_revenue := adult_tickets_sold * adult_ticket_price
  have total_child_revenue := total_revenue - total_adult_revenue
  have child's_ticket_price := total_child_revenue / childs_tickets_sold
  show child's_ticket_cost = 4
  sorry

end child_l707_707888


namespace upstream_speed_l707_707884

theorem upstream_speed (Vm Vdownstream Vupstream Vs : ℝ) 
  (h1 : Vm = 50) 
  (h2 : Vdownstream = 55) 
  (h3 : Vdownstream = Vm + Vs) 
  (h4 : Vupstream = Vm - Vs) : 
  Vupstream = 45 :=
by
  sorry

end upstream_speed_l707_707884


namespace ellipse_properties_l707_707280

-- Define the conditions for a, b, and the given point
variables (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hA : 4 / a^2 + 9 / b^2 = 1) (he : (a^2 - b^2) / a^2 = 1 / 4)

-- Define the condition for the eccentricity
def eccentricity := 1 / 2

-- Define the point A
def A := (2, 3)

-- The main theorem that describes the problem statement and the solution requirements
theorem ellipse_properties :
  (a = 4 ∧ b = 2 * √3 ∧
   ∃ C : ℝ × ℝ,
     C = (-16 * √19 / 19, 6 * √19 / 19) ∧
     ∃ l : ℝ → ℝ,
       ∀ x y : ℝ, (l x = y) ∧
         (2 * x - y - 1 = 0) ∧
         (2 * x - y + 2 * √19 = 0)) :=
begin
  -- Proof is omitted
  sorry
end

end ellipse_properties_l707_707280


namespace myrtle_eggs_count_l707_707048

-- Definition for daily egg production
def daily_eggs : ℕ := 3 * 3

-- Definition for the number of days Myrtle is gone
def days_gone : ℕ := 7

-- Definition for total eggs laid
def total_eggs : ℕ := daily_eggs * days_gone

-- Definition for eggs taken by neighbor
def eggs_taken_by_neighbor : ℕ := 12

-- Definition for eggs remaining after neighbor takes some
def eggs_after_neighbor : ℕ := total_eggs - eggs_taken_by_neighbor

-- Definition for eggs dropped by Myrtle
def eggs_dropped_by_myrtle : ℕ := 5

-- Definition for total remaining eggs Myrtle has
def eggs_remaining : ℕ := eggs_after_neighbor - eggs_dropped_by_myrtle

-- Theorem statement
theorem myrtle_eggs_count : eggs_remaining = 46 := by
  sorry

end myrtle_eggs_count_l707_707048


namespace general_term_and_minimum_value_l707_707277

noncomputable def a_n (n : ℕ) := n - 8

def S_n (n : ℕ) : ℤ := n * (a_n 1 + a_n n) / 2

def f (n : ℕ) : ℤ := (2 * S_n n - 2 * a_n n) / n

theorem general_term_and_minimum_value :
  (∀ n : ℕ, a_n n = n - 8) ∧ minimum_value_of_f : ∃ n : ℕ, f n = -9 :=
by
  sorry

end general_term_and_minimum_value_l707_707277


namespace final_total_price_correct_l707_707585

-- Define the original prices
def original_price_running_shoes : ℝ := 100
def original_price_formal_shoes : ℝ := 150
def original_price_casual_shoes : ℝ := 75

-- Define the store-wide discount
def store_wide_discount (price : ℝ) : ℝ := price * 0.2

-- Define the "buy 2, get 1 at 50% off" discount
def discount_buy2_get1_50off (price : ℝ) : ℝ := price * 0.5

-- Define the final price after all discounts
def final_price_running_shoes : ℝ := original_price_running_shoes - store_wide_discount(original_price_running_shoes)
def final_price_formal_shoes : ℝ := original_price_formal_shoes - store_wide_discount(original_price_formal_shoes)
def final_price_casual_shoes_pre50 : ℝ := original_price_casual_shoes - store_wide_discount(original_price_casual_shoes)
def final_price_casual_shoes : ℝ := final_price_casual_shoes_pre50 - discount_buy2_get1_50off(final_price_casual_shoes_pre50)

-- Prove the final prices and total amount
theorem final_total_price_correct :
  final_price_running_shoes = 80 ∧
  final_price_formal_shoes = 120 ∧
  final_price_casual_shoes = 30 ∧
  final_price_running_shoes + final_price_formal_shoes + final_price_casual_shoes = 230 :=
by
  sorry

end final_total_price_correct_l707_707585


namespace smallest_area_square_l707_707981

theorem smallest_area_square (
  (parabola_eq : ∀ {x : ℝ}, y = x^2 - 2) 
  (line_eq : ∀ {x : ℝ}, y = -2x + 17))
  (exists_pair_parabola : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ y = x1^2 - 2 ∧ y = x2^2 - 2)
  (exists_pair_line : ∃ x3 x4 : ℝ, x3 ≠ x4 ∧ y = -2x3 + 17 ∧ y = -2x4 + 17):
  (5 * ((x1 - x2)^2 - 4 * (-k - 2))) = 160 := 
sorry

end smallest_area_square_l707_707981


namespace total_items_l707_707578

theorem total_items (dozen_eggs : Int) (pounds_flour : Int) (cases_butter : Int) (sticks_per_case : Int) (bottles_vanilla : Int) : 
  (dozen_eggs * 12 + pounds_flour + cases_butter * sticks_per_case + bottles_vanilla) = 198 := 
by 
  have eggs := dozen_eggs * 12
  have flour := pounds_flour
  have butter := cases_butter * sticks_per_case
  have vanilla := bottles_vanilla
  have total := eggs + flour + butter + vanilla
  trivial

end total_items_l707_707578


namespace minimum_students_l707_707329

variables (b g : ℕ) -- Define variables for boys and girls

-- Define the conditions
def boys_passed : ℕ := (3 * b) / 4
def girls_passed : ℕ := (2 * g) / 3
def equal_passed := boys_passed b = girls_passed g

def total_students := b + g + 4

-- Statement to prove minimum students in the class
theorem minimum_students (h1 : equal_passed b g)
  (h2 : ∃ multiple_of_nine : ℕ, g = 9 * multiple_of_nine ∧ 3 * b = 4 * multiple_of_nine * 2) :
  total_students b g = 21 :=
sorry

end minimum_students_l707_707329


namespace count_solutions_l707_707693

def isSolution (x y z : ℕ) : Prop :=
  x < y ∧ y < z ∧ x + y + z = 100

noncomputable def numberOfWays : ℕ :=
  ((finset.range 100).filter (λ x, x > 0)).sum (λ x,
    ((finset.range 100).filter (λ a, a > 0 ∧ 3*x + 2*a < 100)).sum (λ a,
      if 1 <= (100 - 3*x - 2*a) then 1 else 0))

theorem count_solutions : numberOfWays = 784 := 
by
  sorry

end count_solutions_l707_707693


namespace sector_area_150_deg_radius_6_is_15_pi_l707_707250

def area_of_sector (theta r : ℝ) : ℝ :=
  (theta / 360) * π * r^2

theorem sector_area_150_deg_radius_6_is_15_pi :
  area_of_sector 150 6 = 15 * π := by
  sorry

end sector_area_150_deg_radius_6_is_15_pi_l707_707250


namespace total_weight_peppers_l707_707662

def weight_green_peppers : ℝ := 0.3333333333333333
def weight_red_peppers : ℝ := 0.3333333333333333

theorem total_weight_peppers : weight_green_peppers + weight_red_peppers = 0.6666666666666666 := 
by sorry

end total_weight_peppers_l707_707662


namespace range_of_a_l707_707325

open Real

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, ¬ (a * x^2 - 2 * a * x - 3 > 0)) : a ∈ Icc (-3) 0 :=
sorry

end range_of_a_l707_707325


namespace limit_a_n_n_l707_707639

noncomputable def a : ℕ → ℕ → ℝ → ℝ
| i, 0, x := x / (2 ^ i)
| i, j + 1, x := (a i j x) ^ 2 + 2 * (a i j x)

theorem limit_a_n_n (x : ℝ) :
  (tendsto (λ n, a n n x) at_top (𝓝 (Real.exp x - 1))) :=
sorry

end limit_a_n_n_l707_707639


namespace hexagonal_estate_area_l707_707886

theorem hexagonal_estate_area 
    (s_map : ℝ) (scale : ℝ) (s_real : ℝ) (area : ℝ) 
    (h_scale : scale = 300) 
    (h_s_map : s_map = 6)
    (h_s_real : s_real = s_map * scale)
    (h_area : area = (3 * real.sqrt 3 / 2) * s_real ^ 2) :
  area = 4860000 * real.sqrt 3 :=
by
  sorry

end hexagonal_estate_area_l707_707886


namespace ratio_of_a_to_b_l707_707445

variable (a b x m : ℝ)
variable (h_a_pos : a > 0) (h_b_pos : b > 0)
variable (h_x : x = 1.25 * a) (h_m : m = 0.6 * b)
variable (h_ratio : m / x = 0.6)

theorem ratio_of_a_to_b (h_x : x = 1.25 * a) (h_m : m = 0.6 * b) (h_ratio : m / x = 0.6) : a / b = 0.8 :=
by
  sorry

end ratio_of_a_to_b_l707_707445


namespace ship_sailing_rate_l707_707898

/-- Conditions of the problem -/
noncomputable def ship_distance_to_shore : ℝ := 77
noncomputable def water_ingress_rate : ℝ := (9/4) / (11/120)
noncomputable def sinking_threshold : ℝ := 92
noncomputable def pump_rate : ℝ := 12

/-- Question: Find the average rate of sailing so that the ship may just reach the shore as it begins to sink. --/
theorem ship_sailing_rate :
  let net_ingress_rate := water_ingress_rate - pump_rate in
  let time_to_sinking := sinking_threshold / net_ingress_rate in
  let sailing_rate := ship_distance_to_shore / time_to_sinking in
  sailing_rate = 10.5 :=
begin
  sorry
end

end ship_sailing_rate_l707_707898


namespace sweets_distribution_l707_707025

theorem sweets_distribution
  (n k : ℕ) (h1 : 0 < n) (h2 : n ≥ k) 
  (x : Fin k → ℕ) (h3 : ∀ i, x i > 0) 
  (h4 : ∑ i, x i = n) :
  ∃ (a : Fin k → ℕ), 
    (∀ i, a i > 0) ∧
    (∀ i j, i < j → a i > a j) ∧
    (∑ i, x i * a i = n^2) :=
sorry

end sweets_distribution_l707_707025


namespace average_playtime_l707_707387

-- Definitions based on conditions
def h_w := 2 -- Hours played on Wednesday
def h_t := 2 -- Hours played on Thursday
def h_f := h_w + 3 -- Hours played on Friday (3 hours more than Wednesday)

-- Statement to prove
theorem average_playtime :
  (h_w + h_t + h_f) / 3 = 3 := by
  sorry

end average_playtime_l707_707387


namespace simplify_binomial_sum_l707_707769

noncomputable def binomial_sum (n : ℕ) : ℤ :=
∑ k in finset.range n, nat.choose n k * (3 ^ (2 * (n - k)))

theorem simplify_binomial_sum (n : ℕ) : 
  binomial_sum n = 10^n - 1 :=
by
  sorry

end simplify_binomial_sum_l707_707769


namespace prove_range_of_p_l707_707042

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x - 1

def A (x : ℝ) : Prop := x > 2
def no_pre_image_in_A (p : ℝ) : Prop := ∀ x, A x → f x ≠ p

theorem prove_range_of_p (p : ℝ) : no_pre_image_in_A p ↔ p > -1 := by
  sorry

end prove_range_of_p_l707_707042


namespace total_books_now_l707_707013

def jerry_has_4_shelves : Prop := (∀ n : ℕ, n < 5 → (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4))

def books_on_first_shelf_initial := 9

def books_added_first_shelf := 10

def books_on_first_shelf := books_on_first_shelf_initial + books_added_first_shelf

def books_on_second_shelf := 0

def books_on_third_shelf := books_on_first_shelf_initial * 1.3

def books_on_fourth_shelf := books_on_third_shelf / 2 + 5

def total_books := books_on_first_shelf + books_on_second_shelf + books_on_third_shelf + books_on_fourth_shelf

theorem total_books_now : total_books = 42 :=
by
  -- Proof is required here
  exact sorry

end total_books_now_l707_707013


namespace cell_count_at_end_of_days_l707_707875

-- Defining the conditions
def initial_cells : ℕ := 2
def split_ratio : ℕ := 3
def days : ℕ := 9
def cycle_days : ℕ := 3

-- The main statement to be proved
theorem cell_count_at_end_of_days :
  (initial_cells * split_ratio^((days / cycle_days) - 1)) = 18 :=
by
  sorry

end cell_count_at_end_of_days_l707_707875


namespace combinatorial_identity_l707_707072

theorem combinatorial_identity (n : ℕ) (h : 0 < n) :
  (∑ k in finset.range(n+1), if k % 2 = n % 2 then 2^k * nat.choose n k * nat.choose (n-k) ((n-k) / 2) else 0) 
  = nat.choose (2 * n) n :=
by sorry

end combinatorial_identity_l707_707072


namespace boxes_to_eliminate_l707_707692

noncomputable def total_boxes : ℕ := 26
noncomputable def high_value_boxes : ℕ := 6
noncomputable def threshold_probability : ℚ := 1 / 2

-- Define the condition for having the minimum number of boxes
def min_boxes_needed_for_probability (total high_value : ℕ) (prob : ℚ) : ℕ :=
  total - high_value - ((total - high_value) / 2)

theorem boxes_to_eliminate :
  min_boxes_needed_for_probability total_boxes high_value_boxes threshold_probability = 15 :=
by
  sorry

end boxes_to_eliminate_l707_707692


namespace safe_combinations_l707_707356

theorem safe_combinations : 
  let digits := {1, 2, 3, 4, 5}
  in
  (∀ d1 d2 d3 d4 d5 : ℕ, 
    d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d4 ∈ digits ∧ d5 ∈ digits ∧
    (d1 % 2 = 0 → d2 % 2 = 1) ∧ (d1 % 2 = 1 → d2 % 2 = 0) ∧
    (d2 % 2 = 0 → d3 % 2 = 1) ∧ (d2 % 2 = 1 → d3 % 2 = 0) ∧
    (d3 % 2 = 0 → d4 % 2 = 1) ∧ (d3 % 2 = 1 → d4 % 2 = 0) ∧
    (d4 % 2 = 0 → d5 % 2 = 1) ∧ (d4 % 2 = 1 → d5 % 2 = 0)) → 
    (number_of_combinations = 180) :=
by
  sorry

end safe_combinations_l707_707356


namespace is_angle_acb_45_degrees_l707_707084

noncomputable def orthocenter (A B C : Point) : Point := sorry
def is_altitude (A B C : Point) (H : Point) : Prop := sorry

theorem is_angle_acb_45_degrees (A B C H : Point) 
  (H_is_orthocenter : H = orthocenter A B C) 
  (AB_eq_CH : (AB.dist (A, B) = AC.dist (C, H))) :
  ∠ ACB = 45 :=
by
sorry

end is_angle_acb_45_degrees_l707_707084


namespace button_purchase_total_l707_707176

theorem button_purchase_total (g y b r : ℕ)
  (h_g : g = 90)
  (h_y : y = g + 10)
  (h_b : b = g - 5)
  (h_r : r = 2 * (y + b)) :
  g + y + b + r = 645 := by
  rw [h_g] at h_y h_b h_r ⊢
  rw [h_y, h_b, h_r]
  sorry

end button_purchase_total_l707_707176


namespace boat_placement_possible_l707_707754

def is_valid_configuration (grid : array (array nat 10) 10): Prop :=
  ∀ i j, 
  (i < 10 ∧ j < 10) →
  (grid[i][j] = 0 ∨ grid[i][j] = 1) ∧ -- Each cell is either water (0) or part of a boat (1)
  ∃ boats, ∀ b ∈ boats, 
    b.length ≤ 4 ∧ -- Boats have limited size
    (∃ r, ∀ c, c ∈ b → grid[r][c] = 1) ∨ (∃ c, ∀ r, r ∈ b → grid[r][c] = 1) -- Boats are placed horizontally or vertically

def rows_and_columns_valid (grid : array (array nat 10) 10) (row_counts col_counts : array nat 10) : Prop :=
  ∀ i < 10, 
  ∑ j in range 10, grid[i][j] = row_counts[i] ∧ -- Check boat part counts in rows
  ∑ i in range 10, grid[i][j] = col_counts[j] -- Check boat part counts in columns

def no_adjacent_boats (grid : array (array nat 10) 10): Prop :=
  ∀ i j, 
  (i < 10 ∧ j < 10) →
  (∀ di dj, (di = -1 ∨ di = 0 ∨ di = 1) → 
    (dj = -1 ∨ dj = 0 ∨ dj = 1) →
    (di ≠ 0 ∨ dj ≠ 0) →
    i + di ≥ 0 ∧ i + di < 10 ∧ j + dj ≥ 0 ∧ j + dj < 10 →
    grid[i][j] = 0 ∨ grid[i + di][j + dj] = 0) -- Boats are not adjacent to each other

def valid_boat_placement (grid : array (array nat 10) 10) 
   (row_counts col_counts : array nat 10) : Prop :=
  is_valid_configuration grid ∧ 
  rows_and_columns_valid grid row_counts col_counts ∧ 
  no_adjacent_boats grid 

theorem boat_placement_possible
  (row_counts : array nat 10)
  (col_counts : array nat 10)
  (wave_cells : list (nat × nat)) : 
  ∃ grid : array (array nat 10) 10, valid_boat_placement grid row_counts col_counts :=
begin
  sorry
end

end boat_placement_possible_l707_707754


namespace digit_difference_l707_707853

theorem digit_difference (X Y : ℕ) (hX : 1 ≤ X ∧ X ≤ 9) (hY : 0 ≤ Y ∧ Y ≤ 9) (h : 10 * X + Y - (10 * Y + X) = 81) : X - Y = 9 :=
sorry

end digit_difference_l707_707853


namespace smallest_absolute_difference_l707_707794

-- Define the number and conditions
def num : ℕ := 2541

def is_factorial_expr (a b : list ℕ) : Prop :=
  num = (a.foldl (λ acc x, acc * factorial x) 1) / (b.foldl (λ acc x, acc * factorial x) 1)

noncomputable def a1 := 77
noncomputable def b1 := 10

def valid_factors (a b : list ℕ) : Prop :=
  ∀ i, i < a.length → ∀ j, j < b.length → a[i] ≥ a[i+1] ∧ b[j] ≥ b[j+1]

theorem smallest_absolute_difference :
  ∃ (a b : list ℕ), is_factorial_expr a b ∧ valid_factors a b ∧ (a1 + b1) = 87 ∧ abs (a1 - b1) = 67 :=
by
  sorry

end smallest_absolute_difference_l707_707794


namespace find_a_l707_707290

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1
def g : ℝ → ℝ := λ x, -1/4 * x

theorem find_a (a : ℝ) (h_tangent_perpendicular : (deriv (f a) 1 + 4 * deriv g 1 = 0)) : a = 1 := 
sorry

end find_a_l707_707290


namespace digit_after_decimal_is_4_l707_707700

noncomputable def sum_fractions : ℚ := (2 / 9) + (3 / 11)

theorem digit_after_decimal_is_4 :
  (sum_fractions - sum_fractions.floor) * 10 = 4 :=
by
  sorry

end digit_after_decimal_is_4_l707_707700


namespace cannot_eat_166_candies_l707_707188

-- Define parameters for sandwiches and candies equations
def sandwiches_eq (x y z : ℕ) := x + 2 * y + 3 * z = 100
def candies_eq (x y z : ℕ) := 3 * x + 4 * y + 5 * z = 166

theorem cannot_eat_166_candies (x y z : ℕ) : ¬ (sandwiches_eq x y z ∧ candies_eq x y z) :=
by {
  -- Proof will show impossibility of (x, y, z) as nonnegative integers solution
  sorry
}

end cannot_eat_166_candies_l707_707188


namespace sufficiency_condition_for_continuity_l707_707673

noncomputable theory

def continuous_at (f : ℝ → ℝ) (x₀ : ℝ) :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε

def is_defined (f : ℝ → ℝ) (x₀ : ℝ) :=
  ∃ y, f x₀ = y

theorem sufficiency_condition_for_continuity (f : ℝ → ℝ) (x₀ : ℝ) (h1 : continuous_at f x₀) : 
  is_defined f x₀ ∧ (¬ is_defined f x₀ → ¬ continuous_at f x₀) :=
sorry

end sufficiency_condition_for_continuity_l707_707673


namespace sum_of_ten_consecutive_natural_numbers_is_odd_l707_707403

theorem sum_of_ten_consecutive_natural_numbers_is_odd 
  (n : ℕ) :
  let sum := n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8) + (n + 9)
  in sum % 2 = 1 :=
by
  sorry

end sum_of_ten_consecutive_natural_numbers_is_odd_l707_707403


namespace angle_measure_is_zero_l707_707960

-- Definitions corresponding to conditions
variable (x : ℝ)

def complement (x : ℝ) := 90 - x
def supplement (x : ℝ) := 180 - x

-- Final proof statement
theorem angle_measure_is_zero (h : complement x = (1 / 2) * supplement x) : x = 0 :=
  sorry

end angle_measure_is_zero_l707_707960


namespace count_values_n_in_integer_range_l707_707999

theorem count_values_n_in_integer_range :
  (finset.filter (λ n : ℕ, (n ≥ 1) ∧ (n ≤ 50) ∧ ((factorial (n^2 - 1)) % (factorial n ^ n) = 0))
                 (finset.range 51)).card = 34 := 
sorry

end count_values_n_in_integer_range_l707_707999


namespace jessys_jewelry_total_l707_707711

def jessy_initial := {
    necklaces := 10,
    earrings := 15,
    bracelets := 5,
    rings := 8
}

def store_a_transaction (jessy: {necklaces: Nat, earrings: Nat, bracelets: Nat, rings: Nat}) := {
    necklaces := jessy.necklaces + 10 - 2,
    earrings := jessy.earrings + (2 * jessy.earrings / 3),
    bracelets := jessy.bracelets + 3,
    rings := jessy.rings
}

def store_b_transaction (jessy: {necklaces: Nat, earrings: Nat, bracelets: Nat, rings: Nat}) := {
    necklaces := jessy.necklaces + 4,
    earrings := jessy.earrings,
    bracelets := jessy.bracelets + 3 - 3 * 0.35,
    rings := jessy.rings + 2 * jessy.rings
}

def home_transaction (jessy: {necklaces: Nat, earrings: Nat, bracelets: Nat, rings: Nat}, store_a_earrings: Nat) := {
    necklaces := jessy.necklaces,
    earrings := jessy.earrings + store_a_earrings / 5,
    bracelets := jessy.bracelets,
    rings := jessy.rings
}

def sale_transaction (jessy: {necklaces: Nat, earrings: Nat, bracelets: Nat, rings: Nat}) := {
    necklaces := jessy.necklaces + 2,
    earrings := jessy.earrings + 2,
    bracelets := jessy.bracelets,
    rings := jessy.rings + 1
}

def total_jewelry_pieces (jessy: {necklaces: Nat, earrings: Nat, bracelets: Nat, rings: Nat}) :=
  jessy.necklaces + jessy.earrings + jessy.bracelets + jessy.rings

theorem jessys_jewelry_total : 
  let initial := jessy_initial in
  let after_store_a := store_a_transaction initial in
  let after_store_b := store_b_transaction after_store_a in
  let after_home := home_transaction after_store_b after_store_a.earrings in
  let final := sale_transaction after_home in
  total_jewelry_pieces final = 85 :=
  by
  sorry

end jessys_jewelry_total_l707_707711


namespace log_evaluation_l707_707217

theorem log_evaluation : ∀ (a : ℕ), a = 8 → log 2 (a ^ 3) = 9 := by
  intro a h
  have h1 : a = 2 ^ 3 := by
    rw [h]
    rfl
  have h2 : (a ^ 3) = 2 ^ 9 := by
    rw [h1]
    exact pow_mul 2 3 3
  rw [h2]
  exact log_pow 2 9
  sorry

end log_evaluation_l707_707217


namespace smallest_M_l707_707253

-- Define the sequence a_n
def a : ℕ+ → ℕ
| ⟨1, _⟩ := 1
| ⟨n+1, h⟩ := (finset.range (n+1)).prod (λ k, a ⟨k+1, nat.succ_pos k⟩) + 1

-- Define the sum of reciprocals of the sequence a_n
def sumRecip (m : ℕ+) := (finset.range m).sum (λ n, (1 : ℚ) / a ⟨n+1, nat.succ_pos n⟩)

-- The theorem to prove
theorem smallest_M : ∀ m : ℕ+, sumRecip m < 2 := by
  sorry

end smallest_M_l707_707253


namespace largest_integer_less_than_sum_of_logs_l707_707827

theorem largest_integer_less_than_sum_of_logs :
  let expr := List.sum (List.map (λ k, Real.logb 2 ((k + 1) ^ 2 / k ^ 2)) (List.range 2009)) in
  ∃ n : ℕ, n = 21 ∧ ↑n < expr ∧ expr < ↑(n + 1) :=
sorry

end largest_integer_less_than_sum_of_logs_l707_707827


namespace find_m_l707_707652

variable (A B C : ×ℝ × ℝ)
variable (D : ℝ × ℝ)
variable (m : ℝ)

/- Conditions -/
def point_A := (1, -1)
def point_B := (4, -2)
def point_C := (-1, 2)
def point_D := (3, m)
def slope (P Q : ℝ × ℝ) := (Q.2 - P.2) / (Q.1 - P.1)

/- Slope of AB and CD should be equal -/
theorem find_m (m : ℝ) :
  slope point_A point_B = slope point_C point_D → 
  m = 2/3 := 
by
  sorry

end find_m_l707_707652


namespace smallest_number_of_students_l707_707915

theorem smallest_number_of_students (n9 n7 n8 : ℕ) (h7 : 9 * n7 = 7 * n9) (h8 : 5 * n8 = 9 * n9) :
  n9 + n7 + n8 = 134 :=
by
  -- Skipping proof with sorry
  sorry

end smallest_number_of_students_l707_707915


namespace pepa_wins_in_3k_rounds_l707_707360

noncomputable def game_finish_in_at_most_3k_rounds (k : ℕ) (n : ℕ) (x : Fin n → ℤ) : Prop :=
  ∀ (moves : List (List (Fin n → Fin n))), -- List of moves for each round, where each move is a list of partitions of indices
  -- Define conditions of the game given the moves
  (∀ move ∈ moves, 
    (1 < List.length move ∧ 
    ∀ (subseq : List (Fin n)), subseq ∈ move → k ∣ (List.sum (List.map x subseq)))) →
  -- Prove the game finishes in at most 3k rounds
  moves.length ≤ 3 * k

theorem pepa_wins_in_3k_rounds (k n : ℕ) (x : Fin n → ℤ) (h_k : 0 < k) :
  game_finish_in_at_most_3k_rounds k n x :=
sorry -- Proof goes here

end pepa_wins_in_3k_rounds_l707_707360


namespace probability_adjacent_same_color_l707_707609

-- Definitions based on Conditions
def balls : list string := ["R", "R", "W", "W"]

def possible_arrangements (l : list string) : list (list string) :=
  l.permutations

def adjacent_same_color (arrangement : list string) : bool :=
  arrangement.head? = arrangement.tail?.head?

-- Lean 4 formal statement
theorem probability_adjacent_same_color : 
  let favorable_arrangements := (possible_arrangements balls).filter adjacent_same_color
  let total_arrangements := possible_arrangements balls
  (favorable_arrangements.length : ℚ) / (total_arrangements.length : ℚ) = 1 / 3 :=
by
  sorry

end probability_adjacent_same_color_l707_707609


namespace probability_not_siblings_l707_707517

-- Define the number of people and the sibling condition
def number_of_people : ℕ := 6
def siblings_count (x : Fin number_of_people) : ℕ := 2

-- Define the probability that two individuals randomly selected are not siblings
theorem probability_not_siblings (P Q : Fin number_of_people) (h : P ≠ Q) :
  let K := number_of_people - 1
  let non_siblings := K - siblings_count P
  (non_siblings / K : ℚ) = 3 / 5 :=
by
  intros
  sorry

end probability_not_siblings_l707_707517


namespace arithmetic_sequence_sum_first_three_terms_l707_707440

theorem arithmetic_sequence_sum_first_three_terms (a : ℕ → ℤ) 
  (h4 : a 4 = 4) (h5 : a 5 = 7) (h6 : a 6 = 10) : a 1 + a 2 + a 3 = -6 :=
sorry

end arithmetic_sequence_sum_first_three_terms_l707_707440


namespace thm_300th_term_non_square_seq_l707_707485

theorem thm_300th_term_non_square_seq : 
  let non_square_seq (n : ℕ) := { k : ℕ // k > 0 ∧ ∀ m : ℕ, m * m ≠ k } in
  (non_square_seq 300).val = 318 :=
by
  sorry

end thm_300th_term_non_square_seq_l707_707485


namespace ludwig_total_earnings_l707_707737

def regular_salary (weekdays weekenddays : ℕ) (weekday_rate weekend_rate : ℕ) : ℕ :=
  (weekdays * weekday_rate) + (weekenddays * weekend_rate)

def overtime_pay (hours_worked overtime_rate overtime_hours : ℕ) : ℕ :=
  overtime_hours * overtime_rate

def total_earnings (regular_salary overtime_salary : ℕ) : ℕ :=
  regular_salary + overtime_salary

theorem ludwig_total_earnings :
  let weekdays := 4
  let weekenddays := 3
  let weekday_rate := 12
  let weekend_rate := 15
  let half_day_hours := 4
  let total_hours_worked := 52
  let standard_hours := 48
  let overtime_rate := (15 * 3) / 2 -- $15 for 4 hours, 1.5 times
  let overtime_hours := total_hours_worked - standard_hours
  regular_salary weekdays weekenddays weekday_rate (weekend_rate * half_day_hours / 8) +
  overtime_pay total_hours_worked overtime_rate overtime_hours 
  = 115.50 := 
by sorry

end ludwig_total_earnings_l707_707737


namespace correct_statements_l707_707840

variable (P Q : Prop)

-- Define statements
def is_neg_false_if_orig_true := (P → ¬P) = False
def is_converse_not_nec_true_if_orig_true := (P → Q) → ¬(Q → P)
def is_neg_true_if_converse_true := (Q → P) → (¬P → ¬Q)
def is_neg_true_if_contrapositive_true := (¬Q → ¬P) → (¬P → False)

-- Main proposition
theorem correct_statements : 
  is_converse_not_nec_true_if_orig_true P Q ∧ 
  is_neg_true_if_converse_true P Q :=
by
  sorry

end correct_statements_l707_707840


namespace solve_coeffs_l707_707028

noncomputable def polynomial : ℝ → ℝ → ℝ[X] :=
  λ a b, X^4 + a * X^3 - X^2 + b * X - 6

theorem solve_coeffs (a b : ℝ) (h : (2 : ℂ) - (1 : ℂ) * complex.I ∈ (polynomial a b).roots) :
    (a, b) = (-4, 0) :=
begin
  -- We do not need to provide the proof here
  sorry
end

end solve_coeffs_l707_707028


namespace find_cos_C_find_area_l707_707705

open Real

variables {a b c : ℝ}
variables {A B C : ℝ} -- Angles of the triangle

-- Initial conditions of the triangle
axiom triangle_sides_opposite_angles : a = opposite A ∧ b = opposite B ∧ c = opposite C

-- Specified conditions for the proof
axiom condition_1 : 3 * a + b = 2 * c
axiom condition_2 : b = 2
axiom condition_3 : (1 / sin A) + (1 / sin C) = (4 * sqrt 3) / 3
axiom condition_4 : (2 * c - a) * cos B = b * cos A
axiom condition_5 : a ^ 2 + c ^ 2 - b ^ 2 = (4 * sqrt 3 / 3) * S
axiom condition_6 : 2 * b * sin (A + π / 6) = a + c

-- Proving questions
theorem find_cos_C (h1 : 3 * a + b = 2 * c) : cos C = -1 / 7 := sorry

theorem find_area (h2 : b = 2) (h3 : (1 / sin A) + (1 / sin C) = (4 * sqrt 3) / 3) : S = sqrt 3 := sorry

end find_cos_C_find_area_l707_707705


namespace complex_div_problem_l707_707269

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the complex number 3 + 4i
def num : ℂ := 3 + 4 * i

-- Define the complex number i (which is the denominator in the original problem)
def denom : ℂ := i

-- The target is to prove that (3 + 4i) / i = 4 - 3i
theorem complex_div_problem : num / denom = 4 - 3 * i := by
  have h1 : i * i = -1 := by simp [i]
  sorry

end complex_div_problem_l707_707269


namespace angle_APB_obtuse_prob_l707_707755

-- Define vertices of the hexagon
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (9, 3)
def D : ℝ × ℝ := (6, 6)
def E : ℝ × ℝ := (0, 6)
def F : ℝ × ℝ := (-3, 3)

-- Define center and radius of the circle
def center : ℝ × ℝ := (3, 3)
def radius : ℝ := 3

-- Define areas based on given conditions
def area_circle : ℝ := 9 * Real.pi
def area_hexagon : ℝ := 81

-- Define probability as the ratio of areas
def probability : ℝ := area_circle / area_hexagon

-- Problem statement: the probability that ∠APB is obtuse
theorem angle_APB_obtuse_prob :
  probability = Real.pi / 9 := by
  sorry

end angle_APB_obtuse_prob_l707_707755


namespace sum_sin_squared_angles_l707_707199

theorem sum_sin_squared_angles :
  (∑ n in Finset.range 60, (Real.sin ((n + 1) * 6 * Real.pi / 180))^2) = 30.5 := 
by
  sorry

end sum_sin_squared_angles_l707_707199


namespace vertex_of_parabola_l707_707435

theorem vertex_of_parabola :
  ∀ (x y : ℝ), y = -(x + 1)^2 → ∃ v : ℝ × ℝ, v = (-1, 0) :=
begin
  intros x y h,
  use (-1, 0),
  sorry
end

end vertex_of_parabola_l707_707435


namespace find_m_n_sum_l707_707820

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

def has_exactly_three_divisors (n : ℕ) : Prop :=
  ∃ p, is_prime p ∧ n = p * p

theorem find_m_n_sum : 
  let m := 2 in
  let n := 169 in
  (is_prime m ∧ has_exactly_three_divisors n ∧ m + n = 171) :=
by
  have m_prime : is_prime 2 := sorry
  have n_three_divisors : has_exactly_three_divisors 169 := sorry
  exact (m_prime, n_three_divisors, rfl)

end find_m_n_sum_l707_707820


namespace common_sum_matrix_l707_707792

theorem common_sum_matrix 
  (matrix : List (List Int))
  (h_size : matrix.length = 6 ∧ ∀ r, (matrix.getD r []).length = 6)
  (h_elements : ∀ r c, ∃ n, -12 ≤ n ∧ n ≤ 23 ∧ n = (matrix.getD r []).getD c 0)
  (h_sum : ∀ (index : Fin 6), 
     List.sum (matrix.getD index 0) = 
     List.sum (List.map (λ m, List.getD m index 0) matrix) ∧ 
     List.sum (List.map (λ i, List.getD (matrix.getD i 0) i 0) (List.range 6)) =
     List.sum (List.map (λ i, List.getD (matrix.getD i 0) (5 - i) 0) (List.range 6))) :
  ∀ (index : Fin 6), List.sum (matrix.getD index 0) = 33 :=
by
  sorry

end common_sum_matrix_l707_707792


namespace modular_inverse_example_l707_707833

open Int

theorem modular_inverse_example :
  ∃ b : ℤ, 0 ≤ b ∧ b < 120 ∧ (7 * b) % 120 = 1 ∧ b = 103 :=
by
  sorry

end modular_inverse_example_l707_707833


namespace polynomials_divide_x15_minus_1_l707_707402

open Polynomial

-- Define the polynomial f(x) = x^15 - 1
def f : Polynomial ℤ := (X ^ 15) - 1

-- Now, we need to prove the main theorem
theorem polynomials_divide_x15_minus_1 :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 14 → ∃ g : Polynomial ℤ, degree g = k ∧ g ∣ f :=
by
  sorry

end polynomials_divide_x15_minus_1_l707_707402


namespace g_difference_l707_707675

theorem g_difference (x h : ℝ) : 
  let g (x : ℝ) := 3 * x^2 + 4 * x + 5 in
  g (x + h) - g x = h * (6 * x + 3 * h + 4) :=
by 
  let g := λ x : ℝ, 3 * x^2 + 4 * x + 5
  sorry

end g_difference_l707_707675


namespace minimum_value_of_expression_l707_707636

theorem minimum_value_of_expression (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 1 / a + 1 / b = 1) :
  ∃ (x : ℝ), x = (1 / (a - 1) + 9 / (b - 1)) ∧ x = 6 :=
by
  sorry

end minimum_value_of_expression_l707_707636


namespace quartic_poly_has_roots_l707_707952

noncomputable def quartic_polynomial := 
  (λ x : ℝ, x^4 - 10*x^3 + 32*x^2 - 28*x - 8)

theorem quartic_poly_has_roots :
  ∃ (p : ℝ → ℝ), 
    (p = quartic_polynomial) ∧
    (∀ r: ℝ, r ∈ {3 + Real.sqrt 5, 3 - Real.sqrt 5, 2 + Real.sqrt 6, 2 - Real.sqrt 6} → p r = 0) :=
begin
  use quartic_polynomial,
  split,
  { refl },
  { intro r,
    intro hr,
    fin_cases hr,
    sorry, -- Remaining proof steps 
  }
end

end quartic_poly_has_roots_l707_707952


namespace fourth_root_of_2560000_l707_707195

theorem fourth_root_of_2560000 : real.sqrt (real.sqrt (real.sqrt (real.sqrt 2560000))) = 40 := 
by sorry

end fourth_root_of_2560000_l707_707195


namespace tetrahedron_circumscribed_sphere_center_l707_707190

theorem tetrahedron_circumscribed_sphere_center
  (T : Tetrahedron)
  (c1 c2 c3 c4 : Point) -- circumcenters of faces
  (h1 : perpendicular (face T 1) (line_through c1))
  (h2 : perpendicular (face T 2) (line_through c2))
  (h3 : perpendicular (face T 3) (line_through c3))
  (h4 : perpendicular (face T 4) (line_through c4))
  : intersecting_point (line_through c1) (line_through c2) (line_through c3) (line_through c4) = circumscribed_sphere_center T :=
sorry

end tetrahedron_circumscribed_sphere_center_l707_707190


namespace sum_of_first_2n_terms_C_n_l707_707252

noncomputable def a_n (n : ℕ) : ℕ := n + 1

noncomputable def b_n (n : ℕ) : ℕ := 2^n

noncomputable def C_n (n : ℕ) : ℤ :=
  (-1)^n * (n + 1) * 2^n

noncomputable def T_2n (n : ℕ) : ℤ :=
  -(3*n + 2) / 9 * (-2)^(n+1) - 2 / 9

theorem sum_of_first_2n_terms_C_n (n : ℕ) :
  ∑ k in Finset.range (2*n), C_n k = T_2n n :=
sorry

end sum_of_first_2n_terms_C_n_l707_707252


namespace sunil_investment_l707_707077

noncomputable def total_amount (P : ℝ) : ℝ :=
  let r1 := 0.025  -- 5% per annum compounded semi-annually
  let r2 := 0.03   -- 6% per annum compounded semi-annually
  let A2 := P * (1 + r1) * (1 + r1)
  let A3 := (A2 + 0.5 * P) * (1 + r2)
  let A4 := A3 * (1 + r2)
  A4

theorem sunil_investment (P : ℝ) : total_amount P = 1.645187625 * P :=
by
  sorry

end sunil_investment_l707_707077


namespace cesaro_sum_51_term_sequence_l707_707980

noncomputable def cesaro_sum (B : List ℝ) : ℝ :=
  let T := List.scanl (· + ·) 0 B
  T.drop 1 |>.sum / B.length

theorem cesaro_sum_51_term_sequence (B : List ℝ) (h_length : B.length = 49)
  (h_cesaro_sum_49 : cesaro_sum B = 500) :
  cesaro_sum (B ++ [0, 0]) = 1441.18 :=
by
  sorry

end cesaro_sum_51_term_sequence_l707_707980


namespace incenter_of_triangle_l707_707001

noncomputable def internal_excircle (A B C D E F I: Point) (O: Circle) :=
  let incircle := Circle.mk A
  ∃ (I: Point),
    (I = midpoint E F) ∧
    (isTangent O A) ∧
    touches E B D O ∧
    touches F C D O ∧
    (∃ incircle,
     (incircle = Point.mk O) ∧
     (is_incenter I))

theorem incenter_of_triangle (A B C D E F I: Point) (O: Circle):
  (internal_excircle A B C D E F I O) → incenter_of_triangle A B C I :=
by sorry

end incenter_of_triangle_l707_707001


namespace final_laptop_price_l707_707883

theorem final_laptop_price :
  let original_price := 1000.00
  let first_discounted_price := original_price * (1 - 0.10)
  let second_discounted_price := first_discounted_price * (1 - 0.25)
  let recycling_fee := second_discounted_price * 0.05
  let final_price := second_discounted_price + recycling_fee
  final_price = 708.75 :=
by
  sorry

end final_laptop_price_l707_707883


namespace num_subsets_P_l707_707734

open Finset Nat

theorem num_subsets_P {P : Finset ℕ} (h₁ : P = Finset.range 16 \ Finset.singleton 0) :
    {A : Finset ℕ // A.card = 3 ∧ ∀ a ∈ A, a ∈ P ∧ ∃ (a₁ a₂ a₃ : ℕ),
    A = {a₁, a₂, a₃} ∧ a₁ < a₂ ∧ a₂ < a₃ ∧ a₁ + 6 ≤ a₂ + 3 ∧ a₂ + 3 ≤ a₃}.card = 165 := by
  -- To be proved
  sorry

end num_subsets_P_l707_707734


namespace problem_II_l707_707002

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 3)^n

noncomputable def S_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 2) * (1 - (1 / 3)^n)

lemma problem_I_1 (n : ℕ) (hn : n > 0) : a_n n = (1 / 3)^n := by
  sorry

lemma problem_I_2 (n : ℕ) (hn : n > 0) : S_n n = (1 / 2) * (1 - (1 / 3)^n) := by
  sorry

theorem problem_II (t : ℝ) : S_n 1 = 1 / 3 ∧ S_n 2 = 4 / 9 ∧ S_n 3 = 13 / 27 ∧
  (S_n 1 + 3 * (S_n 2 + S_n 3) = 2 * (S_n 1 + S_n 2) * t) ↔ t = 2 := by
  sorry

end problem_II_l707_707002


namespace least_number_leaving_remainder_4_l707_707506

theorem least_number_leaving_remainder_4 (x : ℤ) : 
  (x % 6 = 4) ∧ (x % 9 = 4) ∧ (x % 12 = 4) ∧ (x % 18 = 4) → x = 40 :=
by
  sorry

end least_number_leaving_remainder_4_l707_707506


namespace find_salary_l707_707847

variable (S : ℝ)

def salary := 1 / 5 * S + 1 / 10 * S + 3 / 5 * S + 19000 = S

theorem find_salary (h : 19000 = (1 - ((1 / 5) + (1 / 10) + (3 / 5))) * S) : S = 190000 :=
by
  have h1 : (1 / 5) + (1 / 10) + (3 / 5) = 9 / 10 := sorry
  rw [h1] at h
  have h2 : 19000 = (1 - 9 / 10) * S := h
  have h3 : 1 - 9 / 10 = 1 / 10 := sorry
  rw [h3] at h2
  have h4 : 19000 = (1 / 10) * S := h2
  have h5 : S = 19000 * 10 := sorry
  exact h5

end find_salary_l707_707847


namespace initial_quantity_of_milk_l707_707555

-- Define initial condition for the quantity of milk in container A
noncomputable def container_A : ℝ := 1184

-- Define the quantities of milk in containers B and C
def container_B (A : ℝ) : ℝ := 0.375 * A
def container_C (A : ℝ) : ℝ := 0.625 * A

-- Define the final equal quantities of milk after transfer
def equal_quantity (A : ℝ) : ℝ := container_B A + 148

-- The proof statement that must be true
theorem initial_quantity_of_milk :
  ∀ (A : ℝ), container_B A + 148 = equal_quantity A → A = container_A :=
by
  intros A h
  rw [equal_quantity] at h
  sorry

end initial_quantity_of_milk_l707_707555


namespace find_300th_term_excl_squares_l707_707488

def is_perfect_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

def nth_term_excl_squares (n : ℕ) : ℕ :=
  let excluded := (List.range (n + n / 10)).filter (λ x, ¬ is_perfect_square x)
  excluded.nth n

theorem find_300th_term_excl_squares :
  nth_term_excl_squares 299 = 317 :=
by
  sorry

end find_300th_term_excl_squares_l707_707488


namespace sum_perpendiculars_is_s_l707_707576

-- Define the isosceles right triangle
def isosceles_right_triangle (A B C: Type) [metric_space A] [metric_space B] [metric_space C]
  (AB AC BC: ℝ) : Prop :=
  AB = AC ∧ ∃ s: ℝ, AB = s ∧ BC = s * sqrt 2

-- Define the perpendicular function (dummy definition for now)
def perpendicular_length {A B C P A' B' C'} : ℝ := sorry

-- Main theorem statement
theorem sum_perpendiculars_is_s (A B C P A' B' C': Type) [metric_space A] [metric_space B] [metric_space C] [metric_space P] [metric_space A'] [metric_space B'] [metric_space C']:
  ∃ s: ℝ, 
  (isosceles_right_triangle A B C s s (s * sqrt 2)) →
  (perpendicular_length P A' B C + perpendicular_length P B' A C + perpendicular_length P C' A B) = s :=
sorry

end sum_perpendiculars_is_s_l707_707576


namespace power_inequality_l707_707073

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a ^ (3 / 4) + b ^ (3 / 4) + c ^ (3 / 4) > (a + b + c) ^ (3 / 4) :=
sorry

end power_inequality_l707_707073


namespace f_value_l707_707647

noncomputable def f : ℝ → ℝ
| x => if x > 1 then 2^(x-1) else Real.tan (Real.pi * x / 3)

theorem f_value : f (1 / f 2) = Real.sqrt 3 / 3 := by
  sorry

end f_value_l707_707647


namespace find_range_m_l707_707033

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_range_m :
  (∀ x ∈ set.Icc (-3 : ℝ) 3, f x = f (-x)) ∧
  (∀ x y ∈ set.Icc (0 : ℝ) 3, x < y → f x > f y) ∧
  (∀ m : ℝ, -3 ≤ m ∧ m ≤ 3 → f (1-2*m) < f m) ↔
  (∀ m : ℝ, (-1 ≤ m ∧ m < 1/3) ∨ (1 < m ∧ m ≤ 2)) :=
begin
  sorry
end

end find_range_m_l707_707033


namespace no_eight_in_decimal_expansion_l707_707056

noncomputable def decimal_expansion := ∑' (n : ℕ), (10 ^ (-n) * (n : ℝ))

theorem no_eight_in_decimal_expansion:
  ¬(8 ∈ (to_digits 10 decimal_expansion)) :=
sorry

end no_eight_in_decimal_expansion_l707_707056


namespace integer_part_sum_l707_707448

noncomputable def sequence_x : ℕ → ℚ
| 0 => 1 / 2
| (n+1) => (sequence_x n)^2 + (sequence_x n)

def sum_S : ℚ := (Finset.range 100).sum (λ n => 1 / (sequence_x n + 1))

theorem integer_part_sum : ⌊sum_S⌋ = 1 := sorry

end integer_part_sum_l707_707448


namespace gumballs_given_to_Todd_l707_707301

variable (T : ℕ)

theorem gumballs_given_to_Todd :
  (∀ T, ∃ (A B R : ℕ),
    A = 2 * T ∧
    B = 4 * A - 5 ∧
    R = 6 ∧
    T + A + B + R = 45
    ) →
  T = 4 :=
by
  intro h
  obtain ⟨A, A_eq, hA⟩ := h T
  obtain ⟨B, B_eq, hB⟩ := hA
  obtain ⟨R, R_eq, hR⟩ := hB
  have eq1 : T + A + B + R = 45 := hR
  have eq2 : A = 2 * T := A_eq
  have eq3 : B = 4 * A - 5 := B_eq
  have eq4 : R = 6 := R_eq
  sorry

end gumballs_given_to_Todd_l707_707301


namespace quirkyville_reading_fraction_l707_707187

theorem quirkyville_reading_fraction : 
  (∀ (students : Nat), 
    let enjoy_reading := (7 * students) / 10
    let do_not_enjoy_reading := students - enjoy_reading
    let enjoy_but_say_not := (1 * enjoy_reading) / 4
    let not_enjoy_and_say_not := (4 * do_not_enjoy_reading) / 5
    let total_say_not := enjoy_but_say_not + not_enjoy_and_say_not
    total_say_not ≠ 0 → 
    enjoy_but_say_not / total_say_not = 35 / 83) :=
begin
  assume students,
  let enjoy_reading := (7 * students) / 10,
  let do_not_enjoy_reading := students - enjoy_reading,
  let enjoy_but_say_not := (1 * enjoy_reading) / 4,
  let not_enjoy_and_say_not := (4 * do_not_enjoy_reading) / 5,
  let total_say_not := enjoy_but_say_not + not_enjoy_and_say_not,
  assume h : total_say_not ≠ 0,
  sorry
end

end quirkyville_reading_fraction_l707_707187


namespace perimeter_of_square_D_l707_707427

-- Definition of the perimeter of square C
def perimeter_C := 40
-- Definition of the area of square D in terms of the area of square C
def area_C := ((perimeter_C / 4) ^ 2)
def area_D := area_C / 3
-- Define the side of square D in terms of its area
def side_D := Real.sqrt area_D
-- Prove the perimeter of square D
def perimeter_D := 4 * side_D

-- Statement to prove the perimeter of square D equals the given value
theorem perimeter_of_square_D :
  perimeter_D = 40 * Real.sqrt 3 / 3 :=
by
  sorry

end perimeter_of_square_D_l707_707427


namespace find_f_prime_one_l707_707642

noncomputable theory

open Real

-- Define the function f and its conditions
def f (x : ℝ) := x^3 - (deriv f 1) * x^2 + 1

-- Statement to prove
theorem find_f_prime_one : (deriv f 1) = 1 :=
by {
  sorry
}

end find_f_prime_one_l707_707642


namespace initial_plant_length_l707_707876

variables (L : ℝ)
def length_on_day (n : ℝ) := L + 0.6875 * n

theorem initial_plant_length :
  (length_on_day L 10) = 1.30 * (length_on_day L 4) → L = 11 :=
by
  intro h,
  sorry

end initial_plant_length_l707_707876


namespace find_p_plus_q_l707_707092

-- Define the given conditions
variables (XY XZ YZ OY : ℝ)
variable (p q : ℕ+)
variable (h1 : ∠ XYZ = 90)
variable (h2 : XY + XZ + YZ = 180)
variable (h3 : OY = p / q)
variables (rel_prime : nat.gcd p q = 1)
variable (radius_O : O.radius = 25)
variable (tangent_XZ : IsTangent O XZ)
variable (tangent_YZ : IsTangent O YZ)

-- The statement of the theorem
theorem find_p_plus_q : p + q = 145 := 
sorry

end find_p_plus_q_l707_707092


namespace compute_roots_sum_l707_707721

def roots_quadratic_eq_a_b (a b : ℂ) : Prop :=
  a^2 - 6 * a + 8 = 0 ∧ b^2 - 6 * b + 8 = 0

theorem compute_roots_sum (a b : ℂ) (ha : roots_quadratic_eq_a_b a b) :
  a^5 + a^3 * b^3 + b^5 = -568 := by
  sorry

end compute_roots_sum_l707_707721


namespace find_a_l707_707294

def sequence_a (a : ℕ → ℝ) := ∀ n : ℕ, n > 0 → a n - a (n + 1) = a (n + 1) * a n
def sequence_b (a : ℕ → ℝ) (b : ℕ → ℝ) := ∀ n : ℕ, n > 0 → b n = 1 / a n
def sum_b_condition (b : ℕ → ℝ) := ∑ i in finset.range 10, b (i + 1) = 65

theorem find_a (a b : ℕ → ℝ) (h_a : sequence_a a) (h_b : sequence_b a b) (h_sum_b : sum_b_condition b) : 
  ∀ n : ℕ, n > 0 → a n = 1 / (n + 1) := by
  sorry

end find_a_l707_707294


namespace dutch_americans_with_window_seats_l707_707053

theorem dutch_americans_with_window_seats :
  let total_people := 90
  let dutch_fraction := 3 / 5
  let dutch_american_fraction := 1 / 2
  let window_seat_fraction := 1 / 3
  let dutch_people := total_people * dutch_fraction
  let dutch_americans := dutch_people * dutch_american_fraction
  let dutch_americans_window_seats := dutch_americans * window_seat_fraction
  dutch_americans_window_seats = 9 := by
sorry

end dutch_americans_with_window_seats_l707_707053


namespace stacy_pages_per_day_l707_707518

theorem stacy_pages_per_day :
  ∀ (total_pages days : ℕ), total_pages = 33 ∧ days = 3 → total_pages / days = 11 :=
by
  intros total_pages days h
  cases h
  rw [h_left, h_right]
  exact Nat.div_eq_of_lt dec_trivial dec_trivial sorry

end stacy_pages_per_day_l707_707518


namespace derivative_of_y_l707_707145

theorem derivative_of_y :
  (deriv (λ x : ℝ, 3 * (sin x / cos x^2) + 2 * (sin x / cos x^4))) =
  (λ x : ℝ, (3 + 3 * (sin x)^2) / (cos x)^3 + (2 - 6 * (sin x)^2) / (cos x)^5) :=
by
  sorry

end derivative_of_y_l707_707145


namespace mod_z_eq_sqrt_5_l707_707645

theorem mod_z_eq_sqrt_5
  (z : ℂ)
  (h : (1 + complex.I) * z = 3 + complex.I) :
  complex.abs z = real.sqrt 5 :=
sorry

end mod_z_eq_sqrt_5_l707_707645


namespace proportional_segments_l707_707638

-- Given conditions
variables (l1 l2 l3 : Line)
variables (A B C D E F : Point)
variable AC : Line
variable DF : Line

-- Conditions for parallel lines and intersections
axiom parallel_l1_l2 : l1 ∥ l2
axiom parallel_l2_l3 : l2 ∥ l3
axiom parallel_l1_l3 : l1 ∥ l3
axiom intersection_AC_l1 : A ∈ l1
axiom intersection_AC_l2 : B ∈ l2
axiom intersection_AC_l3 : C ∈ l3
axiom intersection_DF_l1 : D ∈ l1
axiom intersection_DF_l2 : E ∈ l2
axiom intersection_DF_l3 : F ∈ l3
axiom intersection_A_AC : A ∈ AC
axiom intersection_C_AC : C ∈ AC
axiom intersection_D_DF : D ∈ DF
axiom intersection_F_DF : F ∈ DF

-- Proof statement
theorem proportional_segments :
  (AB : ℝ) / (BC : ℝ) = (DE : ℝ) / (EF : ℝ) :=
sorry

end proportional_segments_l707_707638


namespace infinite_primes_of_form_m2_mn_n2_l707_707404

theorem infinite_primes_of_form_m2_mn_n2 : ∀ m n : ℤ, ∃ p : ℕ, ∃ k : ℕ, (p = k^2 + k * m + n^2) ∧ Prime k :=
sorry

end infinite_primes_of_form_m2_mn_n2_l707_707404


namespace relationship_between_m1_m2_l707_707800

noncomputable def quadratic_roots 
  (m : ℝ) : Prop := 
  ∀ x: ℝ, m * x^2 + (1 / 3) * x + 1 = 0

-- Defining a predicate for the root relation
def root_relation 
  (m1 m2 x1 x2 x3 x4 : ℝ) : Prop :=
  quadratic_roots m1 x1 ∧ quadratic_roots m1 x2 ∧ 
  quadratic_roots m2 x3 ∧ quadratic_roots m2 x4 ∧ 
  x1 < x3 ∧ x3 < x4 ∧ x4 < x2 ∧ x2 < 0

-- The main theorem to prove the relationship between m1 and m2
theorem relationship_between_m1_m2
  (m1 m2 x1 x2 x3 x4 : ℝ)
  (h : root_relation m1 m2 x1 x2 x3 x4):
  (m2 > m1 ∧ m1 > 0) := 
begin
  sorry,
end

end relationship_between_m1_m2_l707_707800


namespace number_of_elements_l707_707247

def A : Set ℤ := { x | -2 < x ∧ x < 4 }
def B : Set ℝ := { x | 2 / (x - 1) ≥ 0 }
def complement_B : Set ℝ := { x | x ≤ 1 }

-- Lean proposition to prove that the number of elements in A intersected with the complement of B equals 3
theorem number_of_elements : 
  (Set.card (A ∩ (complement_B : Set ℤ)) = 3) := 
by
  -- proof left as an exercise
  sorry

end number_of_elements_l707_707247


namespace find_m_l707_707537

noncomputable def g (n : ℤ) : ℤ :=
if n % 2 ≠ 0 then 2 * n + 3
else if n % 3 = 0 then n / 3
else n - 1

theorem find_m :
  ∃ m : ℤ, m % 2 ≠ 0 ∧ g (g (g m)) = 36 ∧ m = 54 :=
by
  sorry

end find_m_l707_707537


namespace count_isosceles_triangle_numbers_l707_707730

def isosceles_triangle_numbers (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ c = a) ∧ (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9)

theorem count_isosceles_triangle_numbers : {n : ℕ | ∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ isosceles_triangle_numbers a b c}.finset.card = 165 := sorry

end count_isosceles_triangle_numbers_l707_707730


namespace thm_300th_term_non_square_seq_l707_707484

theorem thm_300th_term_non_square_seq : 
  let non_square_seq (n : ℕ) := { k : ℕ // k > 0 ∧ ∀ m : ℕ, m * m ≠ k } in
  (non_square_seq 300).val = 318 :=
by
  sorry

end thm_300th_term_non_square_seq_l707_707484


namespace no_dapper_number_exists_l707_707929

def is_dapper (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 10 * a + b = a + b^3

theorem no_dapper_number_exists : ∃ n : ℕ, ¬is_dapper n :=
by {
  use 0, -- Select 0 as our candidate number
  intro h, -- Assume there is a dapper number
  cases h with a h,
  cases h with b h,
  rcases h with ⟨_⟩ }; sorry -- Placeholder for handling actual proof details.

end no_dapper_number_exists_l707_707929


namespace cylinder_surface_area_l707_707070

theorem cylinder_surface_area (a b : ℝ) (h1 : a = 4 * Real.pi) (h2 : b = 8 * Real.pi) :
  (∃ S, S = 32 * Real.pi^2 + 8 * Real.pi ∨ S = 32 * Real.pi^2 + 32 * Real.pi) :=
by
  sorry

end cylinder_surface_area_l707_707070


namespace analytical_expression_f_range_of_a_not_monotonic_l707_707248

-- Define the function f(x) with coefficients a, b, c
variables (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Conditions given in the problem
axiom h1 : a ≠ 0
axiom h2 : f 0 = 1
axiom h3 : f 1 = 6
axiom h4 : ∀ x : ℝ,  f (-2 + x) = f (-2 - x)

-- Conclusion 1: Prove the analytical expression for f(x)
theorem analytical_expression_f (h1 : a ≠ 0) (h2 : f 0 = 1) (h3 : f 1 = 6) (h4 : ∀ x : ℝ,  f (-2 + x) = f (-2 - x)) :
  ∃ (a b c : ℝ), f = λ x : ℝ, - (1 / 5) * x ^ 2 - (4 / 5) * x + 1 :=
sorry

-- Conditions for f(x) not being monotonic on [a-1, 2a+1]
axiom h5 : a - 1 < -2
axiom h6 : -2 < 2 * a + 1

-- Conclusion 2: Prove the range of a
theorem range_of_a_not_monotonic (h5 : a - 1 < -2) (h6 : -2 < 2 * a + 1) :
  - (3 / 2) < a ∧ a < -1 :=
sorry

end analytical_expression_f_range_of_a_not_monotonic_l707_707248


namespace polyhedron_body_diagonals_l707_707157

theorem polyhedron_body_diagonals :
  ∀ (n_squares n_hexagons n_octagons : ℕ) (vertices : ℕ),
  n_squares = 12 →
  n_hexagons = 8 →
  n_octagons = 6 →
  vertices = 48 →
  let edges := (3 * vertices) / 2 in
  let face_diagonals := (12 * (4 * (4 - 3) / 2)) + (8 * (6 * (6 - 3) / 2)) + (6 * (8 * (8 - 3) / 2)) in
  let total_diagonals := (vertices * (vertices - 1)) / 2 in
  (total_diagonals - edges - face_diagonals) = 840 :=
begin
  intros,
  sorry
end

end polyhedron_body_diagonals_l707_707157


namespace sequence_sum_l707_707451

theorem sequence_sum (S : ℕ → ℕ) (a : ℕ → ℕ) : 
  (∀ n, S n = 2^n) →
  (a 1 = S 1) ∧ (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  a 3 + a 4 = 12 :=
by
  sorry

end sequence_sum_l707_707451


namespace area_triangle_CEF_l707_707004

variables {A B C E F : Type} [AddCommGroup A] [Module ℝ A]
variables {area : A → ℝ}

-- Given conditions
axiom triangle_ABC : A
axiom area_triangle_ABC : area triangle_ABC = 36
axiom E_divides_AC (A C : A) (E : A) : dist A E / dist E C = 1 / 2
axiom F_midpoint_AB (A B : A) (F : A) : 2 * dist A F = dist A B

-- Question
theorem area_triangle_CEF {CEF : A} (area_AFC : area (triangle_ABC / 2) = 18)
  (part_of_AFC : ∀ (CEF : A), dist E C / dist E F = 2 / 3) : area CEF = 12 :=
sorry

end area_triangle_CEF_l707_707004


namespace horner_method_correct_l707_707118

noncomputable def poly (x : ℝ) : ℝ :=
  5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

-- By Horner's method
noncomputable def horner_eval (x : ℝ) : ℝ :=
  let v0 := 5
  let v1 := v0 * x + 2
  let v2 := v1 * x + 3.5
  let v3 := v2 * x - 2.6
  let v4 := v3 * x + 1.7
  let v5 := v4 * x - 0.8
  v5

theorem horner_method_correct :
  horner_eval 1 = 8.8 ∧ (let x := 1 in let v0 := 5 in let v1 := v0 * x + 2 in let v2 := v1 * x + 3.5 in let v3 := v2 * x - 2.6 in v3 = 7.9) :=
by {
  -- Prove horner_eval 1 = 8.8
  sorry,
  -- Prove v3 = 7.9 using intermediate steps
  sorry
}

end horner_method_correct_l707_707118


namespace problem1_problem2_imaginary_problem3_pure_imaginary_l707_707646

noncomputable def z (a : ℝ) : Complex :=
  (a^2 - 7*a + 6) / (a + 1) + Complex.i * (a^2 - 5*a - 6)

theorem problem1 (a : ℝ) (h : z a.im = 0) : a = 6 :=
  sorry

theorem problem2_imaginary (a : ℝ) (h1 : z a.re = 0) (h2 : ∃ b, z = b * Complex.i) :
  a ≠ -1 ∧ a ≠ 6 :=
  sorry

theorem problem3_pure_imaginary (a : ℝ) (h : z a.re = 0 ∧ z a.im ≠ 0) : a = 1 :=
  sorry

end problem1_problem2_imaginary_problem3_pure_imaginary_l707_707646


namespace cube_coloring_count_l707_707306

-- Define the vertices and edges of the cube
def vertices : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7}
def edges : Finset (Nat × Nat) :=
  { (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
    (4, 5), (4, 6), (5, 7), (6, 7) }

-- Define what it means for a coloring to be valid
def is_valid_coloring (coloring : Nat -> Bool) : Prop :=
  ∀ (v1 v2 : Nat), (v1, v2) ∈ edges → ¬(coloring v1 = true ∧ coloring v2 = true)

-- Main theorem statement
theorem cube_coloring_count : 
  (Finset.filter (is_valid_coloring) (Finset.univ.image (λ f : Nat -> Bool, f)).toFinset).card = 35 := 
sorry

end cube_coloring_count_l707_707306


namespace minimal_length_AX_XB_l707_707370

theorem minimal_length_AX_XB 
  (AA' BB' : ℕ) (A'B' : ℕ) 
  (h1 : AA' = 680) (h2 : BB' = 2000) (h3 : A'B' = 2010) 
  : ∃ X : ℕ, AX + XB = 3350 := 
sorry

end minimal_length_AX_XB_l707_707370


namespace max_sum_l707_707446

open Real

theorem max_sum (a b c : ℝ) (h : a^2 + (b^2) / 4 + (c^2) / 9 = 1) : a + b + c ≤ sqrt 14 :=
sorry

end max_sum_l707_707446


namespace max_consecutive_numbers_with_four_divisors_max_three_consecutive_numbers_with_four_divisors_l707_707504

noncomputable def has_exactly_four_divisors (n : ℕ) : Prop :=
  ∃ (p q : ℕ), (nat.prime p ∨ nat.prime q) ∧
                (n = p ^ 3 ∨ (n = p * q ∧ p ≠ q))

theorem max_consecutive_numbers_with_four_divisors : ∀ (a b c d : ℕ),
  (has_exactly_four_divisors a ∧ has_exactly_four_divisors b ∧ has_exactly_four_divisors c ∧ has_exactly_four_divisors d) → false :=
begin
  -- Proof would go here, but this is just the statement
  sorry
end

theorem max_three_consecutive_numbers_with_four_divisors : ∃ (a b c : ℕ),
  (has_exactly_four_divisors a ∧ has_exactly_four_divisors b ∧ has_exactly_four_divisors c) :=
begin
  -- Proof would go here, but this is just the statement
  sorry
end

end max_consecutive_numbers_with_four_divisors_max_three_consecutive_numbers_with_four_divisors_l707_707504


namespace find_ABC_sum_l707_707088

-- The original problem's conditions
variable (A B C : ℤ)

-- Defining the vertical asymptotes at x = -3, 0, 2.
def has_asymptotes_at (A B C : ℤ) : Prop :=
  ∃ d : ℤ[X], d = X^3 + A * X^2 + B * X + C ∧ 
               d = (X + 3) * X * (X - 2)

-- The theorem we need to prove
theorem find_ABC_sum (A B C : ℤ) (h : has_asymptotes_at A B C) : A + B + C = -5 :=
by 
  -- Including the Lean statement required for the theorem.
  sorry

end find_ABC_sum_l707_707088


namespace problem1_problem2_problem3_problem4_l707_707917

-- Problem (1)
theorem problem1 : (-8 - 6 + 24) = 10 :=
by sorry

-- Problem (2)
theorem problem2 : (-48 / 6 + -21 * (-1 / 3)) = -1 :=
by sorry

-- Problem (3)
theorem problem3 : ((1 / 8 - 1 / 3 + 1 / 4) * -24) = -1 :=
by sorry

-- Problem (4)
theorem problem4 : (-1^4 - (1 + 0.5) * (1 / 3) * (1 - (-2)^2)) = 0.5 :=
by sorry

end problem1_problem2_problem3_problem4_l707_707917


namespace score_of_29_impossible_l707_707552

theorem score_of_29_impossible :
  ¬ ∃ (c u w : ℕ), c + u + w = 10 ∧ 3 * c + u = 29 :=
by {
  sorry
}

end score_of_29_impossible_l707_707552


namespace min_cost_trip_l707_707747

theorem min_cost_trip (d_XZ d_XY : ℝ) (cost_bus_per_km : ℝ) (cost_airplane_per_km booking_fee : ℝ) 
(hd_XZ : d_XZ = 4000) (hd_XY : d_XY = 4500) 
(h_cost_bus : cost_bus_per_km = 0.20) 
(h_cost_airplane : cost_airplane_per_km = 0.12) 
(h_booking_fee : booking_fee = 120) :
  let d_YZ := Real.sqrt (d_XY ^ 2 - d_XZ ^ 2),
      cost_XY_bus := d_XY * cost_bus_per_km,
      cost_XY_airplane := d_XY * cost_airplane_per_km + booking_fee,
      cost_YZ_bus := d_YZ * cost_bus_per_km,
      cost_YZ_airplane := d_YZ * cost_airplane_per_km + booking_fee,
      cost_XZ_bus := d_XZ * cost_bus_per_km,
      cost_XZ_airplane := d_XZ * cost_airplane_per_km + booking_fee
  in min cost_XY_bus cost_XY_airplane 
   + min cost_YZ_bus cost_YZ_airplane 
   + min cost_XZ_bus cost_XZ_airplane 
   = 1627.386 := 
by
  sorry

end min_cost_trip_l707_707747


namespace lakshmi_share_annual_gain_l707_707406

theorem lakshmi_share_annual_gain (x : ℝ) (annual_gain : ℝ) (Raman_inv_months : ℝ) (Lakshmi_inv_months : ℝ) (Muthu_inv_months : ℝ) (Gowtham_inv_months : ℝ) (Pradeep_inv_months : ℝ)
  (total_inv_months : ℝ) (lakshmi_share : ℝ) :
  Raman_inv_months = x * 12 →
  Lakshmi_inv_months = 2 * x * 6 →
  Muthu_inv_months = 3 * x * 4 →
  Gowtham_inv_months = 4 * x * 9 →
  Pradeep_inv_months = 5 * x * 1 →
  total_inv_months = Raman_inv_months + Lakshmi_inv_months + Muthu_inv_months + Gowtham_inv_months + Pradeep_inv_months →
  annual_gain = 58000 →
  lakshmi_share = (Lakshmi_inv_months / total_inv_months) * annual_gain →
  lakshmi_share = 9350.65 :=
by
  sorry

end lakshmi_share_annual_gain_l707_707406


namespace mean_of_normal_distribution_l707_707380

variables (μ σ : ℝ) (X : ℝ → ℝ)

-- Definition of the normal distribution
noncomputable def normal (μ σ : ℝ) : probability_mass_function ℝ :=
sorry -- Placeholder for the actual normal distribution definition

-- Assumptions 
axiom normal_distribution (h : X ~ normal μ σ^2)
axiom prob_equality (h : P(X > 4) = P(X < 0))

-- Theorem statement
theorem mean_of_normal_distribution : μ = 2 :=
begin
  -- Proof would go here.
  sorry
end

end mean_of_normal_distribution_l707_707380


namespace reciprocal_of_point_three_l707_707094

theorem reciprocal_of_point_three : (0.3 : ℚ) = 3 / 10 → (3 / 10)⁻¹ = 10 / 3 :=
by
  intro h
  rw [h]
  have reciprocal_of_fraction : (3 / 10 : ℚ)⁻¹ = (10 / 3 : ℚ), by
    simp [inv_div]
  exact reciprocal_of_fraction

end reciprocal_of_point_three_l707_707094


namespace find_300th_term_excl_squares_l707_707491

def is_perfect_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

def nth_term_excl_squares (n : ℕ) : ℕ :=
  let excluded := (List.range (n + n / 10)).filter (λ x, ¬ is_perfect_square x)
  excluded.nth n

theorem find_300th_term_excl_squares :
  nth_term_excl_squares 299 = 317 :=
by
  sorry

end find_300th_term_excl_squares_l707_707491


namespace total_handshakes_in_tournament_l707_707338

theorem total_handshakes_in_tournament :
  let players := 3 * 4
  let player_to_player_handshakes := 3 * (4 * 2 * 4)
  let player_to_official_handshakes := players * (3 + 1)
  in player_to_player_handshakes + player_to_official_handshakes = 144 :=
by
  let players := 3 * 4
  let player_to_player_handshakes := 3 * (4 * 2 * 4)
  let player_to_official_handshakes := players * (3 + 1)
  show player_to_player_handshakes + player_to_official_handshakes = 144
  -- Here we would need to carry out the proof steps in detail.
  sorry

end total_handshakes_in_tournament_l707_707338


namespace perimeter_of_rectangle_l707_707860

variables {Point : Type} [Add Group Point] [AffineSpace Point Point]

structure Rectangle (p1 p2 p3 p4 : Point) :=
(is_rectangle : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p4 ≠ p1)

variables {A B C D E F B' : Point}
variables (ABCD : Rectangle A B C D)
variables (AE BE CF : ℕ)
variables (AE_12 : AE = 12)
variables (BE_25 : BE = 25)
variables (CF_5 : CF = 5)
variables (E_on_AB : E ∈ line A B)
variables (F_on_CD : F ∈ line C D)
variables (B_on_AD : B' ∈ line A D)

theorem perimeter_of_rectangle (ABCD : Rectangle A B C D) (AE BE CF : ℕ) 
  (AE_12 : AE = 12) (BE_25 : BE = 25) (CF_5 : CF = 5) :
  ∃ P : ℕ, P = 148 
:= 
sorry

end perimeter_of_rectangle_l707_707860


namespace segment_bisected_at_A_l707_707163

variable {P Q A : EuclideanSpace ℝ ℝ}

theorem segment_bisected_at_A (A_inside_angle : ∃ (l1 l2 : set (EuclideanSpace ℝ ℝ)), 
    A ∈ l1 ∧ A ∈ l2 ∧ angle A l1 l2 = 2 ∧ ∀ (l : set (EuclideanSpace ℝ ℝ)), 
    (A ∈ l) → (area (triangle A l1 l2 l) = smallest_area_triangle)) : 
    bisection A P Q :=
sorry

end segment_bisected_at_A_l707_707163


namespace three_hundredth_term_without_squares_l707_707496

noncomputable def sequence_without_squares (n : ℕ) : ℕ :=
(n + (n / Int.natAbs (Int.sqrt (n.succ - 1))))

theorem three_hundredth_term_without_squares : 
  sequence_without_squares 300 = 307 :=
sorry

end three_hundredth_term_without_squares_l707_707496


namespace macy_miles_left_to_run_l707_707741

-- Define the given conditions
def goal : ℕ := 24
def miles_per_day : ℕ := 3
def days : ℕ := 6

-- Define the statement to be proven
theorem macy_miles_left_to_run :
  goal - (miles_per_day * days) = 6 :=
by
  sorry

end macy_miles_left_to_run_l707_707741


namespace eccentricity_of_ellipse_l707_707264

variable {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b)
variable {PF1 PF2 : ℝ} (h₃ : |PF1| - |PF2| = 3 * b) (h₄ : |PF1| * |PF2| = (9 / 4) * a * b)

theorem eccentricity_of_ellipse
  (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) 
  (h₃ : |PF1| - |PF2| = 3 * b) (h₄ : |PF1| * |PF2| = (9 / 4) * a * b) :
  (sqrt (a^2 - b^2)) / a = 2 * sqrt 2 / 3 :=
by
  sorry

end eccentricity_of_ellipse_l707_707264


namespace quadrilateral_cyclic_lines_intersect_l707_707729

-- Define a line
variable (g : Line)

-- Define points A and B where circles k1 and k2 touch the line g
variable (A B : Point)

-- Define circles k1 and k2, which touch the line g at points A and B respectively
variable (k1 k2 : Circle)
variable (k1_touches_g : k1.TouchesLine g A)
variable (k2_touches_g : k2.TouchesLine g B)

-- Define circle k3, which touches k1 at point D and k2 at point C
variable (k3 : Circle)
variable (D C : Point)
variable (k3_touches_k1_at_D : k3.TouchesCircle k1 D)
variable (k3_touches_k2_at_C : k3.TouchesCircle k2 C)

-- Defining points D and C properly implies C and D are on circle k3
-- Now, we need to prove the two statements

-- Proof that quadrilateral ABCD is cyclic
theorem quadrilateral_cyclic :
  cyclicQuadrilateral A B C D :=
sorry

-- Proof that lines BC and AD intersect on circle k3
theorem lines_intersect : 
  intersectsOnCircle k3 B C A D :=
sorry

end quadrilateral_cyclic_lines_intersect_l707_707729


namespace volume_to_surface_area_ratio_l707_707924

def volume (cubes : Nat) : Nat :=
  8

def surface_area (cubes : Nat) : Nat :=
  33

theorem volume_to_surface_area_ratio : 
  volume 8 = 8 → surface_area 8 = 33 → (volume 8) / (surface_area 8) = 8 / 33 :=
by {
  intros h_volume h_surface_area,
  rw [h_volume, h_surface_area],
  norm_num
}

end volume_to_surface_area_ratio_l707_707924


namespace math_clubs_probability_l707_707105

theorem math_clubs_probability :
  let num_students := [6, 8, 9, 10]
  let co_presidents := 3
  let probability (n : ℕ) := (choose co_presidents 2 * choose (n - co_presidents) 2) / (choose n 4 : ℚ)
  (1 / 4 : ℚ) * (probability 6 + probability 8 + probability 9 + probability 10) = 59 / 140 := by
  -- Proof will go here
  sorry

end math_clubs_probability_l707_707105


namespace domain_sqrt_one_minus_x_squared_domain_fraction_cubrt_domain_arccos_domain_fraction_sinx_domain_log_l707_707961

namespace DomainProblems

-- Problem 1
theorem domain_sqrt_one_minus_x_squared (x : ℝ) : 1 - x^2 ≥ 0 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

-- Problem 2
theorem domain_fraction_cubrt (x : ℝ) : (x^2 - 5 * x + 6) ≠ 0 ↔ x ∉ {2, 3} :=
by sorry

-- Problem 3 
theorem domain_arccos (x : ℝ) : -1 ≤ (1 - 2 * x) / 3 ∧ (1 - 2 * x) / 3 ≤ 1 ↔ -1 ≤ x ∧ x ≤ 2 :=
by sorry

-- Problem 4
theorem domain_fraction_sinx (x : ℝ) : sin x ≠ 0 ↔ x ∉ {k * Real.pi | k : ℤ} :=
by sorry

-- Problem 5
theorem domain_log (x : ℝ) : x^2 - 9 > 0 ↔ x < -3 ∨ x > 3 :=
by sorry

end DomainProblems

end domain_sqrt_one_minus_x_squared_domain_fraction_cubrt_domain_arccos_domain_fraction_sinx_domain_log_l707_707961


namespace not_square_sum_l707_707036

theorem not_square_sum (n : ℕ) (d : ℕ) (x : ℤ) (h : d ∣ 2 * n^2) : n^2 + d ≠ x^2 := by
  sorry

end not_square_sum_l707_707036


namespace magnitude_difference_of_vectors_l707_707640

variables {V : Type*} [inner_product_space ℝ V]

theorem magnitude_difference_of_vectors 
  (a b : V) 
  (h1 : ∥a∥ = 1) 
  (h2 : ∥b∥ = 1) 
  (h3 : ⟪a, b⟫ = 1 / 2) : 
  ∥a - b∥ = 1 :=
sorry

end magnitude_difference_of_vectors_l707_707640


namespace infinite_nosol_eq_l707_707760

theorem infinite_nosol_eq : ∃^∞ n : ℕ, ∀ x y z : ℤ, x^2 + y^11 - z^(nat.factorial 2022) ≠ n :=
by
  sorry

end infinite_nosol_eq_l707_707760


namespace tangent_line_eqn_c_range_l707_707285

noncomputable def f (x : ℝ) := 3 * x * Real.log x + 2

theorem tangent_line_eqn :
  let k := 3 
  let x₀ := 1 
  let y₀ := f x₀ 
  y = k * (x - x₀) + y₀ ↔ 3*x - y - 1 = 0 :=
by sorry

theorem c_range (x : ℝ) (hx : 1 < x) (c : ℝ) :
  f x ≤ x^2 - c * x → c ≤ 1 - 3 * Real.log 2 :=
by sorry

end tangent_line_eqn_c_range_l707_707285


namespace graph_not_in_first_quadrant_l707_707441

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2)^x - 2

-- Prove that the graph of f(x) does not pass through the first quadrant
theorem graph_not_in_first_quadrant : ∀ (x : ℝ), x > 0 → f x ≤ 0 := by
  intro x hx
  sorry

end graph_not_in_first_quadrant_l707_707441


namespace no_terms_of_form_3_pow_alpha_mul_5_pow_beta_l707_707246

def sequence_v : ℕ → ℤ
| 0 := 0
| 1 := 1
| (n + 1) := 8 * sequence_v n - sequence_v (n - 1)

theorem no_terms_of_form_3_pow_alpha_mul_5_pow_beta:
  ∀ (n α β : ℕ), α > 0 → β > 0 → sequence_v n ≠ 3 ^ α * 5 ^ β := 
by
  intros n α β hα hβ
  sorry

end no_terms_of_form_3_pow_alpha_mul_5_pow_beta_l707_707246


namespace possible_values_n_of_arithmetic_chords_l707_707843

theorem possible_values_n_of_arithmetic_chords 
  (x y : ℝ)
  (h_circle : x^2 + y^2 = 5 * x)
  (px py : ℝ)
  (h_point : px = 5 / 2 ∧ py = 3 / 2)
  (n : ℕ)
  (a1 an : ℝ)
  (d : ℝ)
  (h_shortest_chord : a1 = 4)
  (h_longest_chord : an = 5)
  (h_arithmetic_seq : ∀ (i : ℕ), 1 ≤ i → i ≤ n → (a1 + (i - 1) * d))
  (h_d_bound : 1 / 6 ≤ d ∧ d ≤ 1 / 3) :
  n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 :=
by
  sorry  -- Proof omitted

end possible_values_n_of_arithmetic_chords_l707_707843


namespace solve_quadratic_eq_l707_707416

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 4 * x - 1 = 0) : x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
sorry

end solve_quadratic_eq_l707_707416


namespace ellipse_equation_max_area_l707_707258

-- Given conditions
def isEllipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ b = 1 ∧ (a^2 = b^2 + (a^2 - b^2))

def hasEccentricity (a b c : ℝ) : Prop :=
  c = (a^2 - b^2)^0.5 ∧ c / a = sqrt(6) / 3

def pointOnEllipse (a : ℝ) : Prop :=
  1 / a^2 + 1 = 1 -- plugged in (0, 1)

def slopesAndSides (M N P Q : (ℝ × ℝ)) : Prop :=
  let diagSlope := (P.2 - M.2) / (P.1 - M.1) = -1 in
  let MN := sqrt((N.1 - M.1)^2 + (N.2 - M.2)^2) in
  let MQ := sqrt((Q.1 - M.1)^2 + (Q.2 - M.2)^2) in
  let PN := sqrt((N.1 - P.1)^2 + (N.2 - P.2)^2) in
  let PQ := sqrt((Q.1 - P.1)^2 + (Q.2 - P.2)^2) in
  diagSlope ∧ MN = MQ ∧ PN = PQ

-- 1. Prove the equation of the ellipse
theorem ellipse_equation (a b : ℝ) (h₁ : isEllipse a b) (h₂: hasEccentricity a b ((a^2 - b^2)^0.5)) (h₃ : pointOnEllipse a) : 
  ∃ (a : ℝ), (a = sqrt 3) → (1 / (a^2) * x^2 + y^2 = 1) :=
begin
  sorry -- proof here
end

-- 2. Prove the maximum area of the quadrilateral
theorem max_area (M N P Q : (ℝ × ℝ)) (a b : ℝ) (h₁ : isEllipse a b) (h₂: hasEccentricity a b ((a^2 - b^2)^0.5))
  (h₃ : pointOnEllipse a) (h₄: slopesAndSides M N P Q) : 
  ∃ (S : ℝ), S = 3 :=
begin
  sorry -- proof here
end

end ellipse_equation_max_area_l707_707258


namespace probability_other_side_heads_l707_707985

noncomputable def probability_heads_other_side : ℚ :=
  let total_heads := 1 + 2 in -- head count from each coin type
  let probability_hh_coin := 2 / total_heads in -- probability that the selected coin is H-H given it showed heads
  probability_hh_coin

theorem probability_other_side_heads : 
  probability_heads_other_side = 2 / 3 :=
sorry

end probability_other_side_heads_l707_707985


namespace collinear_incenter_intersection_circumcenter_l707_707914

variables {A B C : Type} [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C]
variables {a b c : Triangle A}

theorem collinear_incenter_intersection_circumcenter
  (O : Point A) (I : incenter a) (E : circumcenter a)
  (hO : Intersects (circumcircle a b c) O)
  (h_incenter_angle_bisectors : Incenter := I)
  (h_incenter_tangent_condition : Tangent I (Side a) ∧ Tangent I (Side b) ∧ Tangent I (Side c))
  (h_circles_similar : Similar (Triangle A'B'C') (Triangle ABC))
  : Collinear A (O : incenter) (E : circumcenter) :=
begin
  sorry
end

end collinear_incenter_intersection_circumcenter_l707_707914


namespace rational_solutions_k_l707_707988

theorem rational_solutions_k (k : ℕ) (h : k > 0) : (∃ x : ℚ, 2 * (k : ℚ) * x^2 + 36 * x + 3 * (k : ℚ) = 0) → k = 6 :=
by
  -- proof to be written
  sorry

end rational_solutions_k_l707_707988


namespace no_solution_15x_29y_43z_t2_l707_707857

theorem no_solution_15x_29y_43z_t2 (x y z t : ℕ) : ¬ (15 ^ x + 29 ^ y + 43 ^ z = t ^ 2) :=
by {
  -- We'll insert the necessary conditions for the proof here
  sorry -- proof goes here
}

end no_solution_15x_29y_43z_t2_l707_707857


namespace macy_miles_left_to_run_l707_707742

-- Define the given conditions
def goal : ℕ := 24
def miles_per_day : ℕ := 3
def days : ℕ := 6

-- Define the statement to be proven
theorem macy_miles_left_to_run :
  goal - (miles_per_day * days) = 6 :=
by
  sorry

end macy_miles_left_to_run_l707_707742


namespace fuel_savings_l707_707944

theorem fuel_savings
  (x : ℝ)  -- fuel efficiency of old car in km/L
  (c : ℝ)  -- cost of gasoline per liter
  (hx_pos : 0 < x)
  (hc_pos : 0 < c) :
  let xf := (17/10 : ℝ) * x,                         -- fuel efficiency of new car
      cd := (5/4 : ℝ) * c,                          -- cost of diesel per liter
      cost_old := c/x,                              -- cost per km with old car
      cost_new := cd/xf,                            -- cost per km with new car
      savings := cost_old - cost_new,               -- savings per km
      percent_savings := (savings / cost_old) * 100 -- percentage savings
  in percent_savings = 26.5 :=
by
  sorry

end fuel_savings_l707_707944


namespace find_side_lengths_and_angles_of_triangle_by_tangency_points_l707_707707

noncomputable def side_lengths_and_angles (a1 b1 c1 : ℝ) (α β γ a b c : ℝ) : Prop :=
  ∃ r s,
    s = (a1 + b1 + c1) / 2 ∧ 
    r = (a1 * b1 * c1) / (4 * real.sqrt (s * (s - a1) * (s - b1) * (s - c1))) ∧
    α ≈ 2 * real.arccos (a1 / (2 * r)) ∧
    β ≈ 2 * real.arccos (b1 / (2 * r)) ∧
    γ ≈ 2 * real.arccos (c1 / (2 * r)) ∧
    a ≈ r * (real.cot (real.arccos (b1 / (2 * r))) + real.cot (real.arccos (c1 / (2 * r)))) ∧
    b ≈ r * (real.cot (real.arccos (a1 / (2 * r))) + real.cot (real.arccos (c1 / (2 * r)))) ∧
    c ≈ r * (real.cot (real.arccos (a1 / (2 * r))) + real.cot (real.arccos (b1 / (2 * r))))

theorem find_side_lengths_and_angles_of_triangle_by_tangency_points :
  side_lengths_and_angles 25 29 36 
    92.7944 (deg) 73.7392 (deg) 13.4658 (deg)
    177.7 170.8 41.4 :=
  sorry

end find_side_lengths_and_angles_of_triangle_by_tangency_points_l707_707707


namespace ab_plus_cd_eq_12_l707_707575

theorem ab_plus_cd_eq_12 (a b c d : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a + b + d = -1) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 5) :
  a * b + c * d = 12 := by
  sorry

end ab_plus_cd_eq_12_l707_707575


namespace exponent_for_half_eq_025_l707_707826

theorem exponent_for_half_eq_025 :
  (0.25 : ℝ) = (1 / 4 : ℝ) →
  ((1 / 2 : ℝ) ^ 2 = (1 / 4 : ℝ)) →
  ∃ x : ℝ, (1 / 2 : ℝ) ^ x = 0.25 ∧ x = 2 :=
by
  intros h₁ h₂
  use 2
  have h : (1 / 2 : ℝ) ^ 2 = 0.25 := by rw [h₂, h₁]
  exact ⟨h, rfl⟩

end exponent_for_half_eq_025_l707_707826


namespace percentage_of_money_saved_is_correct_l707_707975

-- Definitions of the conditions
def family_size : ℕ := 4
def cost_per_orange : ℝ := 1.5
def planned_expenditure : ℝ := 15

-- Correct answer to be proven
def percentage_saved : ℝ :=
  ((family_size * cost_per_orange) / planned_expenditure) * 100

-- Proof goal
theorem percentage_of_money_saved_is_correct :
  percentage_saved = 40 := by
  sorry

end percentage_of_money_saved_is_correct_l707_707975


namespace sin_ratios_in_triangle_l707_707005

theorem sin_ratios_in_triangle 
  {A B C D : Type*} 
  (angleB : ∠ B = 45)
  (angleC : ∠ C = 30)
  (D_divides_BC : D divides BC in ratio 1 : 2) 
  : (sin (∠ BAD) / sin (∠ CAD)) = sqrt(2) / 8 := 
sorry

end sin_ratios_in_triangle_l707_707005


namespace min_n_probability_l707_707775

-- Define the number of members in teams
def num_members (n : ℕ) : ℕ := n

-- Define the total number of handshakes
def total_handshakes (n : ℕ) : ℕ := n * n

-- Define the number of ways to choose 2 handshakes from total handshakes
def choose_two_handshakes (n : ℕ) : ℕ := (total_handshakes n).choose 2

-- Define the number of ways to choose event A (involves exactly 3 different members)
def event_a_count (n : ℕ) : ℕ := 2 * n.choose 1 * (n - 1).choose 1

-- Define the probability of event A
def probability_event_a (n : ℕ) : ℚ := (event_a_count n : ℚ) / (choose_two_handshakes n : ℚ)

-- The minimum value of n such that the probability of event A is less than 1/10
theorem min_n_probability :
  ∃ n : ℕ, (probability_event_a n < (1 : ℚ) / 10) ∧ n ≥ 20 :=
by {
  sorry
}

end min_n_probability_l707_707775


namespace closed_figure_area_l707_707274

-- Definitions of the conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2
def line (x : ℝ) : ℝ := x

-- Our problem statement: Given the focus of the parabola at (0, 1)
-- Prove that the area of the closed figure formed by the parabola and the line is 8/3.
theorem closed_figure_area :
  (∀ a : ℝ, parabola a 0 = 0 → (parabola a 1 = 1 → a = 1/4)) →
  (∫ x in 0..4, line x - parabola (1/4) x) = 8/3 :=
by
  sorry

end closed_figure_area_l707_707274


namespace area_of_shaded_region_l707_707345

-- Definitions of points and semicircles
structure Point (α : Type) :=
(x : α) (y : α)

def radius := 1

noncomputable def Arc (A B : Point ℝ) :=
  -- We define an arc from point A to point B
  semicircle -- Details abstracted

noncomputable def midpoint (A B : Point ℝ) : Point ℝ :=
  ⟨ (A.x + B.x) / 2, (A.y + B.y) / 2 ⟩

def D := midpoint A B
def E := midpoint B C
def F := midpoint D E

-- Defining the areas involved
def shaded_area (R : ℝ) := R * R

-- The proof about the area of the shaded region
theorem area_of_shaded_region : shaded_area 1 = 2 := by
  -- Here we would normally provide the proof steps
  sorry

end area_of_shaded_region_l707_707345


namespace pen_more_expensive_than_two_notebooks_l707_707889

variable (T R C : ℝ)

-- Conditions
axiom cond1 : T + R + C = 120
axiom cond2 : 5 * T + 2 * R + 3 * C = 350

-- Theorem statement
theorem pen_more_expensive_than_two_notebooks :
  R > 2 * T :=
by
  -- omit the actual proof, but check statement correctness
  sorry

end pen_more_expensive_than_two_notebooks_l707_707889


namespace sum_of_coefficients_l707_707669

noncomputable def polynomial_sum (a : ℕ → ℤ) := ∀ x : ℤ, 
  (1 - 2 * x)^9 = a(9) * x^9 + a(8) * x^8 + a(7) * x^7 + a(6) * x^6 +
  a(5) * x^5 + a(4) * x^4 + a(3) * x^3 + a(2) * x^2 + a(1) * x + a(0)

theorem sum_of_coefficients : 
  ∃ (a : ℕ → ℤ), polynomial_sum a ∧ (a(1) + a(2) + a(3) + a(4) + a(5) + a(6) + a(7) + a(8) + a(9) = -2) :=
sorry

end sum_of_coefficients_l707_707669


namespace april_roses_l707_707564

theorem april_roses (price_per_rose earnings roses_left : ℤ) 
  (h1 : price_per_rose = 4)
  (h2 : earnings = 36)
  (h3 : roses_left = 4) :
  4 + (earnings / price_per_rose) = 13 :=
by
  sorry

end april_roses_l707_707564


namespace acute_angle_between_diagonals_l707_707430

-- Define given conditions
variables (R H m n : ℝ)
-- Define the areas
def S_base : ℝ := π * R^2
def S_section : ℝ := 2 * R * H
-- Define the initial ratio condition
def ratio_condition := (π * R^2) / (2 * R * H) = m / n
-- Define the tangent of half of the acute angle
def tan_half_alpha := (2 * R) / H

-- Lean 4 statement for the proof problem
theorem acute_angle_between_diagonals (h_ratio : ratio_condition) :
  let k := 4 * m / (π * n) in
  (if (m / n < π / 4) then
    2 * Real.arctan (4 * m / (π * n))
  else 
    2 * Real.arctan (π * n / (4 * m))) = 
    (if (m / n < π / 4) then
      2 * Real.arctan k
    else 
      2 * Real.arctan (π * n / (4 * m))) :=
by sorry

end acute_angle_between_diagonals_l707_707430


namespace john_initial_bench_weight_l707_707713

variable (B : ℕ)

theorem john_initial_bench_weight (B : ℕ) (HNewTotal : 1490 = 490 + B + 600) : B = 400 :=
by
  sorry

end john_initial_bench_weight_l707_707713


namespace transformation_possible_iff_l707_707237

theorem transformation_possible_iff (n : ℕ) (h : n > 1) :
  (∃ f : list ℕ → list ℕ,  f (list.range (n + 1) \\ [0]) = n :: (list.range (n - 1) \\ [0])) ↔ (n % 3 = 0 ∨ n % 3 = 1) :=
sorry

end transformation_possible_iff_l707_707237


namespace scientific_notation_l707_707524

def zhangjiajie_visitors : ℕ := 864000

theorem scientific_notation : zhangjiajie_visitors = 8.64 * 10^5 := by
  sorry

end scientific_notation_l707_707524


namespace min_distance_l707_707525

noncomputable def line1 : ℝ → ℝ → Prop := λ x y, 3 * x + 4 * y - 10 = 0
noncomputable def line2 : ℝ → ℝ → Prop := λ x y, 6 * x + 8 * y + 5 = 0

theorem min_distance : ∀ (P Q : ℝ × ℝ), line1 P.1 P.2 → line2 Q.1 Q.2 → dist P Q = 5 / 2 :=
sorry

end min_distance_l707_707525


namespace intersection_M_N_l707_707655

def M := {y : ℝ | y <= 4}
def N := {x : ℝ | x > 0}

theorem intersection_M_N : {x : ℝ | x > 0} ∩ {y : ℝ | y <= 4} = {z : ℝ | 0 < z ∧ z <= 4} :=
by
  sorry

end intersection_M_N_l707_707655


namespace percentage_profits_to_revenues_l707_707690

open Real

theorem percentage_profits_to_revenues
  (R P : ℝ)
  (h1 : 0.14 * R = 1.3999999999999997 * P) :
  (P / R) * 100 = 10 :=
by
  have h2 : P = 0.14 * R / 1.3999999999999997 := by simp [div_eq_mul_inv, ← mul_assoc]; sorry
  simp [h2, mul_div_cancel_left 0.14 (ne_of_gt $ by norm_num : 1.3999999999999997 ≠ 0)]
  norm_num
  done

end percentage_profits_to_revenues_l707_707690


namespace minimum_queries_2022_gon_l707_707023

theorem minimum_queries_2022_gon : ∃ (Q : ℕ), Q = 22 ∧ ∀ (choose_point_color : ℕ → Prop), 
  let A : Fin 2022 → (ℕ × ℕ) := fun i => has_property (colors : Fin 2022 → Bool),
    ∃Q, Q = 22 ∧ ∀(choose_point_color : Fin 2022 → Bool), 
      ∃Q, Q = 22 ∧ ∀(choose_point_color : Fin 2022 → Bool),
        Q = 22 :=

begin
  sorry
end

end minimum_queries_2022_gon_l707_707023


namespace slices_with_both_toppings_l707_707874

theorem slices_with_both_toppings
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (all_with_topping : total_slices = 15 ∧ pepperoni_slices = 8 ∧ mushroom_slices = 12 ∧ ∀ i, i < 15 → (i < 8 ∨ i < 12)) :
  ∃ n, (pepperoni_slices - n) + (mushroom_slices - n) + n = total_slices ∧ n = 5 :=
by
  sorry

end slices_with_both_toppings_l707_707874


namespace find_correct_volume_y_l707_707415

-- Define constants representing the conditions
def alcohol_concentration_x := 0.10
def alcohol_concentration_y := 0.30
def volume_x := 300
def target_concentration := 0.25

-- Define the function to calculate the final alcohol concentration
def final_concentration (volume_y : ℝ) : ℝ :=
  (alcohol_concentration_x * volume_x + alcohol_concentration_y * volume_y) / (volume_x + volume_y)

-- State the theorem we need to prove
theorem find_correct_volume_y : 
  ∃ (volume_y : ℝ), final_concentration volume_y = target_concentration ∧ volume_y = 900 :=
  by
  sorry

end find_correct_volume_y_l707_707415


namespace integer_part_sum_l707_707801

noncomputable def x_seq (n : ℕ) : ℚ :=
  if n = 1 then 1 / 2
  else x_seq (n - 1) ^ 2 + x_seq (n - 1)

theorem integer_part_sum (n : ℕ) (h : n = 100) :
  ⌊∑ i in Finset.range n, 1 / (x_seq i.succ + 1)⌋ = 1 :=
by
  sorry

end integer_part_sum_l707_707801


namespace impossible_draw_1006_2012gons_l707_707353

theorem impossible_draw_1006_2012gons :
  ¬ ∃ (polygons : set (set ℕ)), 
    (∀ p ∈ polygons, set.card p = 2012) ∧
    (set.card polygons = 1006) ∧ 
    (∀ p1 p2 ∈ polygons, p1 ≠ p2 → (p1 ∩ p2).card ≤ (2012 - 2)) :=
sorry

end impossible_draw_1006_2012gons_l707_707353


namespace average_playtime_l707_707388

-- Definitions based on conditions
def h_w := 2 -- Hours played on Wednesday
def h_t := 2 -- Hours played on Thursday
def h_f := h_w + 3 -- Hours played on Friday (3 hours more than Wednesday)

-- Statement to prove
theorem average_playtime :
  (h_w + h_t + h_f) / 3 = 3 := by
  sorry

end average_playtime_l707_707388


namespace jimmy_drinks_8_ounces_each_time_l707_707015

theorem jimmy_drinks_8_ounces_each_time : 
  let daily_drinks := 8
  let total_days := 5
  let total_gallons := 2.5
  let ounce_to_gallon := 0.0078125
  let total_ounces := total_gallons / ounce_to_gallon
  let total_drink_times := daily_drinks * total_days
  (total_ounces / total_drink_times) = 8 :=
by
  let daily_drinks := 8
  let total_days := 5
  let total_gallons := 2.5
  let ounce_to_gallon := 0.0078125
  let total_ounces := total_gallons / ounce_to_gallon
  let total_drink_times := daily_drinks * total_days
  have h : total_ounces = 320 := by sorry
  have h' : total_drink_times = 40 := by sorry
  exact calc
    total_ounces / total_drink_times = 320 / 40 : by rw [h, h']
    ... = 8 : by norm_num

end jimmy_drinks_8_ounces_each_time_l707_707015


namespace HM_perpendicular_to_common_chord_l707_707697

variables {A B C H D E : Type} [metric_space A] [metric_space B] [metric_space C]
[metric_space H] [metric_space D] [metric_space E]

noncomputable def triangle := {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
structure data := 
  (acute ∶ ∀ x y z ∈ triangle, 
  ∠ x y z < π / 2 )

structure points := 
  (A B C : data)
  (H : A → B → C)
  (M: Type)

variables [triangle ABC_] : data
variables {AB AC : data} 
variables (H : is_orthocenter H ABC_)
variables (M : is_midpoint M B C)
variables (D : point_on_line D A B)
variables (AD_perpendicular : Perpendicular AD D)
variables (E : point_on_line E A C)
variables (AD_eq_AE : equal_length AD AE)
variables (DHE_collinear : Collinear D H E)
variables (common_chord : Type) 

theorem HM_perpendicular_to_common_chord :
  perpendicular HM common_chord := sorry

end HM_perpendicular_to_common_chord_l707_707697


namespace relationship_between_x_y_z_l707_707633

/-- Given positive integers x, y, z and real numbers a, b, c, d 
which satisfy x <= y <= z, x^a = y^b = z^c = 70^d, 
and 1/a + 1/b + 1/c = 1/d, prove that x + y = z. -/
theorem relationship_between_x_y_z (x y z : ℕ) (a b c d : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxy : x ≤ y) (hyz : y ≤ z)
  (h1 : x^a = 70^d) (h2 : y^b = 70^d) (h3 : z^c = 70^d)
  (h4 : 1/a + 1/b + 1/c = 1/d) : x + y = z :=
sorry

end relationship_between_x_y_z_l707_707633


namespace constant_term_expansion_l707_707526

theorem constant_term_expansion (c : ℤ) : 
  (\exists c : ℤ, (∀ x : ℤ, c = \sum r in (finset.range 7), (binom 6 r) * (pow (x^2) (6-r)) * pow ((-1 / x) r)) 
  ∧ (c = 15)) := sorry

end constant_term_expansion_l707_707526


namespace winnie_the_pooh_largest_piece_l707_707238

theorem winnie_the_pooh_largest_piece :
  let cake_weight : ℝ := 10
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 100 ∧
  (∀ m : ℕ, 1 ≤ m ∧ m ≤ 100 →
    n = 10 ∧ 
    (m = 1 ∨ m = 10 ∨ (m ≠ n) → 
      let remaining_weight : ℝ := cake_weight - (n / 100) * cake_weight in 
      remaining_weight - ((n-1) / 100) * remaining_weight ≤
      remaining_weight - (m / 100) * remaining_weight )) :=
sorry

end winnie_the_pooh_largest_piece_l707_707238


namespace term_omit_perfect_squares_300_l707_707480

theorem term_omit_perfect_squares_300 (n : ℕ) (hn : n = 300) : 
  ∃ k : ℕ, k = 317 ∧ (∀ m : ℕ, (m < k → m * m ≠ k)) := 
sorry

end term_omit_perfect_squares_300_l707_707480


namespace sum_of_squares_eq_192_l707_707365

theorem sum_of_squares_eq_192 (a b c: ℕ) (h1 : a + b + c = 24)
  (h2 : nat.gcd a b + nat.gcd b c + nat.gcd c a = 10)
  (h3 : (∃ x y, (x, y) ∈ {(a, b), (b, c), (c, a)} ∧ x % 2 = 0 ∧ y % 2 = 0 )) :
  a^2 + b^2 + c^2 = 192 :=
sorry

end sum_of_squares_eq_192_l707_707365


namespace value_exponentiation_l707_707509

theorem value_exponentiation:
  (125 = 5^3) →
  (∀ a m n, (a^m)^n = a^(m * n)) →
  (∀ b a n, log b (a^n) = n * log b a) →
  (∀ a x, a^(log a x) = x) →
  (∀ a n m, (a^n)^(1/m) = a^(n/m)) →
  (125^(log 5 2023))^(1/3) = 2023 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end value_exponentiation_l707_707509


namespace number_of_unequal_sided_triangles_with_perimeter_lt_13_l707_707561

theorem number_of_unequal_sided_triangles_with_perimeter_lt_13 :
  (∃ (x y z : ℕ), x < y ∧ y < z ∧ x + y + z < 13 ∧ x + y > z ∧ y + z > x ∧ z + x > y) ∧
  (∀ x y z : ℕ, (x < y ∧ y < z ∧ x + y + z < 13 ∧ x + y > z ∧ y + z > x ∧ z + x > y) → (x, y, z) = (2, 3, 4) ∨ (2, 4, 5) ∨ (3, 4, 5)) ∧
  (∀ x y z : ℕ, x < y ∧ y < z ∧ x + y + z < 13 ∧ x + y > z ∧ y + z > x ∧ z + x > y → True) :=
by
  sorry

end number_of_unequal_sided_triangles_with_perimeter_lt_13_l707_707561


namespace part_I_part_II_l707_707613

noncomputable def f (x : ℝ) : ℝ := |2 * x - 3/4| + |2 * x + 5/4|

theorem part_I (a : ℝ) : -1 ≤ a ∧ a ≤ 2 ↔ ∀ x : ℝ, f x ≥ a^2 - a :=
by
  sorry

theorem part_II (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m + n = 1) :
  sqrt (2 * m + 1) + sqrt (2 * n + 1) ≤ 2 * sqrt 2 :=
by
  sorry

end part_I_part_II_l707_707613


namespace verify_total_bill_l707_707474

def fixed_charge : ℝ := 20
def daytime_rate : ℝ := 0.10
def evening_rate : ℝ := 0.05
def free_evening_minutes : ℕ := 200

def daytime_minutes : ℕ := 200
def evening_minutes : ℕ := 300

noncomputable def total_bill : ℝ :=
  fixed_charge + (daytime_minutes * daytime_rate) +
  ((evening_minutes - free_evening_minutes) * evening_rate)

theorem verify_total_bill : total_bill = 45 := by
  sorry

end verify_total_bill_l707_707474


namespace intersect_range_l707_707696

noncomputable def general_eq_line (a : ℝ) (t : ℝ) : ℝ × ℝ :=
(a - 2 * t, -4 * t)

noncomputable def general_eq_circle : ℝ × ℝ → Prop :=
λ (p : ℝ × ℝ), p.1^2 + p.2^2 = 16

theorem intersect_range (a : ℝ) :
  (∃ t : ℝ, general_eq_circle (general_eq_line a t)) ↔ (-2 * Real.sqrt 5 ≤ a ∧ a ≤ 2 * Real.sqrt 5) := by
sorry

end intersect_range_l707_707696


namespace conjugate_of_z_l707_707323

theorem conjugate_of_z :
  ∀ (z : ℂ), (sqrt 2 + complex.i) * z = 3 * complex.i → complex.conj z = 1 - sqrt 2 * complex.i :=
by
  intro z h
  sorry

end conjugate_of_z_l707_707323


namespace real_roots_of_quadratic_l707_707207

theorem real_roots_of_quadratic (k : ℝ) : (k ≤ 0 ∨ 1 ≤ k) →
  ∃ x : ℝ, x^2 + 2 * k * x + k = 0 :=
by
  intro h
  sorry

end real_roots_of_quadratic_l707_707207


namespace integral_quadratic_l707_707593

variable {f : ℝ → ℝ}
variable (h : ∀ x : ℝ, ∃ a b c: ℝ, f(x) = a * x^2 + b * x + c)

theorem integral_quadratic (f : ℝ → ℝ)
  (h : ∀ x : ℝ, ∃ a b c: ℝ, f(x) = a * x^2 + b * x + c) :
  ∫ x in 0..2, f(x) = (f 0 + 4 * f 1 + f 2) / 3 :=
sorry

end integral_quadratic_l707_707593


namespace find_a_plus_b_l707_707309

theorem find_a_plus_b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 4 * a + 3 * b = 39) :
  a + b = 82 / 7 :=
sorry

end find_a_plus_b_l707_707309


namespace isosceles_right_triangle_properties_l707_707793

theorem isosceles_right_triangle_properties
  (length_median_hypotenuse : ℝ)
  (h_median : length_median_hypotenuse = 12) :
  ∃ (leg_length area : ℝ), leg_length = 12 * Real.sqrt 2 ∧ area = 144 :=
by 
  use 12 * Real.sqrt 2
  use 144
  split
  sorry
  sorry

end isosceles_right_triangle_properties_l707_707793


namespace residue_system_sum_divisible_l707_707021

open Nat

theorem residue_system_sum_divisible {n : ℕ} 
  (h_gcd : gcd n (2 * (2 ^ 1386 - 1)) = 1)
  (a : Finset ℕ) (h_a : a.card = φ n) 
  (h_rres : ∀ x ∈ a, gcd x n = 1) : 
  n ∣ (a.sum (λ x, x ^ 1386)) := 
sorry

end residue_system_sum_divisible_l707_707021


namespace initial_ratio_l707_707349

-- Definitions of the initial state and conditions
variables (M W : ℕ)
def initial_men : ℕ := M
def initial_women : ℕ := W
def men_after_entry : ℕ := M + 2
def women_after_exit_and_doubling : ℕ := (W - 3) * 2
def current_men : ℕ := 14
def current_women : ℕ := 24

-- Theorem to prove the initial ratio
theorem initial_ratio (M W : ℕ) 
    (hm : men_after_entry M = current_men)
    (hw : women_after_exit_and_doubling W = current_women) :
  M / Nat.gcd M W = 4 ∧ W / Nat.gcd M W = 5 :=
by
  sorry

end initial_ratio_l707_707349


namespace dimensions_of_triangle_cut_diagonally_l707_707900

-- Define a constant for the side length of the square
constant side_length : ℝ := 10

-- Define the dimension theorem to prove the dimensions of the resulting triangle
theorem dimensions_of_triangle_cut_diagonally (a b hypotenuse : ℝ) :
  a = side_length ∧ b = side_length ∧ hypotenuse = side_length * Real.sqrt 2 :=
by
  -- Assume a and b are the legs of the right-angled triangle
  assume a_eq : a = side_length,
  assume b_eq : b = side_length,

  -- Use Pythagorean theorem for right-angled triangle
  have hyp_eq : hypotenuse = Real.sqrt (a * a + b * b),
  calc
    hypotenuse = Real.sqrt (side_length * side_length + side_length * side_length) : by rw [a_eq, b_eq]
    ... = Real.sqrt (2 * (side_length * side_length)) : by rw [mul_add, one_add_one_eq_two]
    ... = Real.sqrt 2 * side_length : by rw [Real.sqrt_mul, Real.sqrt_sq (0.le_trans zero_le_two), mul_comm],
  
  -- Conclude the dimensions satisfy the required values
  exact ⟨a_eq, b_eq, hyp_eq⟩

end dimensions_of_triangle_cut_diagonally_l707_707900


namespace isosceles_trapezoid_area_correct_l707_707502

def isosceles_trapezoid_area (a b c: ℝ) (h: ℝ) : ℝ :=
  (1/2) * (a + b) * h

theorem isosceles_trapezoid_area_correct :
  ∀ (a b c d: ℝ) (h: ℝ), 
  isosceles_trapezoid(a, b, c, d, h) → 
  a = 10 → b = 16 → c = 5 → d = 5 → h = 4 → 
  isosceles_trapezoid_area a b h = 52 :=
by
  intros a b c d h ht a_eq b_eq c_eq d_eq h_eq
  sorry

end isosceles_trapezoid_area_correct_l707_707502


namespace quadratic_inequality_l707_707748

theorem quadratic_inequality (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c > c * x) : ∀ x : ℝ, c * x^2 - b * x + a > c * x - b := 
by
  sorry

end quadratic_inequality_l707_707748


namespace fraction_broke_off_l707_707539

variable (p p_1 p_2 : ℝ)
variable (k : ℝ)

-- Conditions
def initial_mass : Prop := p_1 + p_2 = p
def value_relation : Prop := p_1^2 + p_2^2 = 0.68 * p^2

-- Goal
theorem fraction_broke_off (h1 : initial_mass p p_1 p_2)
                           (h2 : value_relation p p_1 p_2) :
  (p_2 / p) = 1 / 5 :=
sorry

end fraction_broke_off_l707_707539


namespace container_capacity_l707_707846

theorem container_capacity (C : ℝ) 
  (h1 : (0.30 * C : ℝ) + 27 = 0.75 * C) : C = 60 :=
sorry

end container_capacity_l707_707846


namespace shaded_area_eq_l707_707534

theorem shaded_area_eq (O A B P Q R : Point)
  (h_circle : Circle O 2)
  (h_angle_AOB : angle A O B = π / 2)
  (h_point_P : P ∈ Circle O 2)
  (h_triangle_OAP : is_right_triangle O A P)
  (h_triangle_OBP : is_right_triangle O B P)
  (h_extend_AQ : extend_to_circle A P = Q)
  (h_extend_BR : extend_to_circle B P = R)
  (h_perpendicular_OQ_OP : perpendicular (O, Q) (O, P))
  (h_perpendicular_OR_OP : perpendicular (O, R) (O, P)) :
  area_shaded_region O A B Q R P - area_triangle O A B = 3 * π / 2 - 2 :=
sorry

end shaded_area_eq_l707_707534


namespace reducible_fraction_least_n_l707_707598

theorem reducible_fraction_least_n : ∃ n : ℕ, (0 < n) ∧ (n-15 > 0) ∧ (gcd (n-15) (3*n+4) > 1) ∧
  (∀ m : ℕ, (0 < m) ∧ (m-15 > 0) ∧ (gcd (m-15) (3*m+4) > 1) → n ≤ m) :=
by
  sorry

end reducible_fraction_least_n_l707_707598


namespace harry_worked_41_hours_l707_707943

def james_earnings (x : ℝ) : ℝ :=
  (40 * x) + (7 * 2 * x)

def harry_earnings (x : ℝ) (h : ℝ) : ℝ :=
  (24 * x) + (11 * 1.5 * x) + (2 * h * x)

def harry_hours_worked (h : ℝ) : ℝ :=
  24 + 11 + h

theorem harry_worked_41_hours (x : ℝ) (h : ℝ) 
  (james_worked : james_earnings x = 54 * x)
  (harry_paid_same : harry_earnings x h = james_earnings x) :
  harry_hours_worked h = 41 :=
by
  -- sorry is used to skip the proof steps
  sorry

end harry_worked_41_hours_l707_707943


namespace sum_converges_l707_707374

open_locale big_operators

noncomputable def a (n : ℕ) : ℕ := -- Definition Placeholder
noncomputable def b (n : ℕ) : ℕ := -- Definition Placeholder

theorem sum_converges (han : ∀ n : ℕ, a (n+1) > a n) (hbn : ∀ n : ℕ, b n = Nat.lcm_list (List.ofFn (λ i, a i) n)) :
  (∑' n, 1 / (b n : ℝ)) < ∞ :=
by
  sorry

end sum_converges_l707_707374


namespace logarithm_argument_positive_l707_707324

open Real

theorem logarithm_argument_positive (a : ℝ) : 
  (∀ x : ℝ, sin x ^ 6 + cos x ^ 6 + a * sin x * cos x > 0) ↔ -1 / 2 < a ∧ a < 1 / 2 :=
by
  -- placeholder for the proof
  sorry

end logarithm_argument_positive_l707_707324


namespace continuous_circle_within_points_l707_707810

-- Definitions
def ring := ℕ -- A ring is represented by a natural number
def segment := list ring -- A segment is a list of rings

-- Initial conditions
def segments : list segment := [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
def open_cost : ℕ := 2  -- Cost to open a ring
def connect_cost : ℕ := 3  -- Cost to connect an opened ring

-- Total points limit
def total_points_limit : ℕ := 15

-- Prove that it is possible to form a continuous circle within the given point limit
theorem continuous_circle_within_points : ∃ (opened : list ring) (connected : list (ring × ring)),
    list.length opened = 3 ∧
    list.length connected = 3 ∧
    open_cost * list.length opened + connect_cost * list.length connected ≤ total_points_limit :=
begin
    sorry
end

end continuous_circle_within_points_l707_707810


namespace largest_initial_value_l707_707040

noncomputable def sequence (a_0 : ℝ) := 
  λ n : ℕ, if h : n = 0 then a_0 else 
    let a : ℕ → ℝ := sequence a_0 
    in a (n-1)^2 - 1 / 2^(2020 * 2^(n-1) - 1)

theorem largest_initial_value (a_0 : ℝ) :
  (∀ n, sequence a_0 n > -1 ∧ sequence a_0 n < 1) → 
  a_0 ≤ 1 + 1 / 2^2020 := sorry

end largest_initial_value_l707_707040


namespace not_divisible_by_n_plus_4_l707_707401

theorem not_divisible_by_n_plus_4 (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 8*n + 15 = k * (n + 4) :=
sorry

end not_divisible_by_n_plus_4_l707_707401


namespace arithmetic_sequence_solution_l707_707322

def arithmetic_sequence_problem (a_1 d: ℤ) :=
  let a (n : ℕ) := a_1 + (n - 1) * d in
  let S (n : ℕ) := n * (2 * a_1 + (n - 1) * d) / 2 in
  a 2 + S 3 = 4 ∧ a 3 + S 5 = 12 → a 4 + S 7 = 24

theorem arithmetic_sequence_solution :
  ∀ a_1 d : ℤ, arithmetic_sequence_problem a_1 d :=
by
  intros a_1 d
  let a (n : ℕ) := a_1 + (n - 1) * d
  let S (n : ℕ) := n * (2 * a_1 + (n - 1) * d) / 2
  assume h : a 2 + S 3 = 4 ∧ a 3 + S 5 = 12
  sorry

end arithmetic_sequence_solution_l707_707322


namespace find_solutions_l707_707958

def equation (x : ℝ) : Prop :=
  (1 / (x^2 + 12*x - 8)) + (1 / (x^2 + 3*x - 8)) + (1 / (x^2 - 14*x - 8)) = 0

theorem find_solutions : {x : ℝ | equation x} = {2, -4, 1, -8} :=
  by
  sorry

end find_solutions_l707_707958


namespace unique_root_multiset_l707_707923

noncomputable def T : Multiset ℤ := {1, 1, 1, 1, -1, -1, -1, -1}

theorem unique_root_multiset (b_0 b_1 b_2 b_3 b_4 b_5 b_6 b_7 b_8 : ℤ)
  (h1 : ∃ (s : Multiset ℤ), s.card = 8 ∧ (∀ x ∈ s, x ∈ ℤ) ∧ s.sum = 0)
  (h2 : ∀ (s : ℤ), s ∈ T → b_8 * s^8 + b_7 * s^7 + b_6 * s^6 + b_5 * s^5 + b_4 * s^4 + b_3 * s^3 + b_2 * s^2 + b_1 * s + b_0 = 0)
  (h3 : ∀ (s : ℤ), s ∈ T → b_0 * s^8 + b_1 * s^7 + b_2 * s^6 + b_3 * s^5 + b_4 * s^4 + b_5 * s^3 + b_6 * s^2 + b_7 * s + b_8 = 0) :
  ∃! t : Multiset ℤ, t = T :=
by sorry

end unique_root_multiset_l707_707923


namespace binom_coeff_div_prime_l707_707982

open Nat

theorem binom_coeff_div_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  p ∣ Nat.choose n p :=
by
  sorry

end binom_coeff_div_prime_l707_707982


namespace matrix_diagonal_neg5_l707_707212

variable (M : Matrix (Fin 3) (Fin 3) ℝ)

theorem matrix_diagonal_neg5 
    (h : ∀ v : Fin 3 → ℝ, (M.mulVec v) = -5 • v) : 
    M = !![-5, 0, 0; 0, -5, 0; 0, 0, -5] :=
by
  sorry

end matrix_diagonal_neg5_l707_707212


namespace no_two_right_angles_in_triangle_l707_707762

theorem no_two_right_angles_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 90) (h3 : B = 90): false :=
by
  -- we assume A = 90 and B = 90,
  -- then A + B + C > 180, which contradicts h1,
  sorry
  
example : (3 = 3) := by sorry  -- Given the context of the multiple-choice problem.

end no_two_right_angles_in_triangle_l707_707762


namespace nabla_2_3_2_eq_4099_l707_707210

def nabla (a b : ℕ) : ℕ := 3 + b ^ a

theorem nabla_2_3_2_eq_4099 : nabla (nabla 2 3) 2 = 4099 :=
by
  sorry

end nabla_2_3_2_eq_4099_l707_707210


namespace three_hundredth_term_without_squares_l707_707495

noncomputable def sequence_without_squares (n : ℕ) : ℕ :=
(n + (n / Int.natAbs (Int.sqrt (n.succ - 1))))

theorem three_hundredth_term_without_squares : 
  sequence_without_squares 300 = 307 :=
sorry

end three_hundredth_term_without_squares_l707_707495


namespace locus_of_points_tangent_arc_l707_707968

noncomputable def is_on_tangent {C : Circle} (P : Point) : Prop := 
  ∃ Q ∈ C, tangent P Q 

noncomputable def auxiliary_circle {A B : Point} : Circle := sorry -- Placeholder for auxiliary circle construction

noncomputable def locus_points {C : Circle} {A B : Point} : Set Point := 
  { X | is_on_tangent X ∧ 
  (outside_A : Point := belongs to auxiliary_circle A B) ∧ 
  (within_boundaries : Point := X ∈ exterior region defined).

theorem locus_of_points_tangent_arc {C : Circle} {A B : Point} :
  let locus := locus_points C A B 
  in locus = { X | tangent to arc A B ∧ within combined auxiliary circles bounds } :=
  sorry  -- Proof to be provided

end locus_of_points_tangent_arc_l707_707968


namespace triangle_DMN_is_right_l707_707299

variables {A B C D E M N O : Type*}
variables [inner_product_space ℝ O] [circumcenter_of_triangle ℝ A B C O]
variables [point E : midpoint A C] [line OE : line_segment O E]
variables [intersection D : OE.intersection_segment AB = D]
variables [circumcenter M : circumcenter_of_triangle ℝ B C D M]
variables [incenter N : incenter_of_triangle ℝ B C D N]
variables [length : length_of_segment ℝ A B = 2 * length_of_segment ℝ B C]

theorem triangle_DMN_is_right :
  is_right_triangle ℝ D M N :=
by
  sorry

end triangle_DMN_is_right_l707_707299


namespace problem_to_prove_l707_707266

theorem problem_to_prove
  (α : ℝ)
  (h : Real.sin (3 * Real.pi / 2 + α) = 1 / 3) :
  Real.cos (Real.pi - 2 * α) = -7 / 9 :=
by
  sorry -- proof required

end problem_to_prove_l707_707266


namespace unit_vector_theorem_l707_707453

def unit_vector_makes_same_angle (c : ℝ × ℝ) :=
  ∃ (λ : ℝ),
    c = (λ * 4, λ * 2) ∧ 
    (|λ| * 2 * Real.sqrt 5 = 1)

theorem unit_vector_theorem :
  unit_vector_makes_same_angle (2 * Real.sqrt 5 / 5, Real.sqrt 5 / 5) ∨
  unit_vector_makes_same_angle (-2 * Real.sqrt 5 / 5, -Real.sqrt 5 / 5) :=
sorry

end unit_vector_theorem_l707_707453


namespace perimeter_of_square_D_l707_707428

-- Definition of the perimeter of square C
def perimeter_C := 40
-- Definition of the area of square D in terms of the area of square C
def area_C := ((perimeter_C / 4) ^ 2)
def area_D := area_C / 3
-- Define the side of square D in terms of its area
def side_D := Real.sqrt area_D
-- Prove the perimeter of square D
def perimeter_D := 4 * side_D

-- Statement to prove the perimeter of square D equals the given value
theorem perimeter_of_square_D :
  perimeter_D = 40 * Real.sqrt 3 / 3 :=
by
  sorry

end perimeter_of_square_D_l707_707428


namespace problem_f_sum_zero_l707_707725

variable (f : ℝ → ℝ)

def odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def symmetrical (f : ℝ → ℝ) : Prop := ∀ x, f (1 - x) = f x

-- Prove the required sum is zero given the conditions.
theorem problem_f_sum_zero (hf_odd : odd f) (hf_symm : symmetrical f) : 
  f 1 + f 2 + f 3 + f 4 + f 5 = 0 := by
  sorry

end problem_f_sum_zero_l707_707725


namespace circumcircle_tangent_l707_707708

-- The conditions in Lean 4 setting
variables (A B C M D Y Z : Type) [Geometry A B C D M Y Z]

-- Midpoint definition
def is_midpoint (M : Point) (B C : Point) [Geometry] : Prop := 
  dist B M = dist C M

-- Given conditions
def conditions (T : Triangle) (D Y Z : Point) : Prop :=
(∃ (M : Point), is_midpoint M (T.b, T.c) ∧ D ∈ line_segment A M ∧ Y ∈ ray A T.c ∧ Z ∈ ray A T.b ∧
∠ D Y T.c = ∠ D C B ∧ ∠ D B C = ∠ D Z B)

-- Proof problem in Lean 4
theorem circumcircle_tangent {A B C D M Y Z : Point} [Geometry]
  (h1 : is_midpoint M B C)
  (h2 : D ∈ line_segment A M)
  (h3 : Y ∈ ray A C)
  (h4 : Z ∈ ray A B)
  (h5 : ∠ D Y C = ∠ D C B)
  (h6 : ∠ D B C = ∠ D Z B) :
  tangent (circumcircle Δ D Y Z) (circumcircle Δ D B C) :=
sorry

end circumcircle_tangent_l707_707708


namespace stone_99_is_3_l707_707392

theorem stone_99_is_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 9 ∧ n ≡ 99 [MOD 16] :=
by
  use 3
  split
  · norm_num
  split
  · norm_num
  · norm_num
  sorry

end stone_99_is_3_l707_707392


namespace infinitely_many_solutions_l707_707064

theorem infinitely_many_solutions :
  ∃∞ (x y z : ℕ+), (x + y + z)^2 + 2 * (x + y + z) = 5 * (x * y + y * z + z * x) :=
sorry

end infinitely_many_solutions_l707_707064


namespace find_other_number_in_game_l707_707333

theorem find_other_number_in_game :
  let numbers := (1 : ℕ) :: List.range 2003 in
  let final_two_numbers := List.filter (fun x => x = 4 ∨ x = 1000) numbers in
  List.length final_two_numbers = 2 ∧ (1000 ∈ final_two_numbers) ∧ (4 ∈ final_two_numbers) :=
by
  sorry

end find_other_number_in_game_l707_707333


namespace evaluate_expression_l707_707592

theorem evaluate_expression : 
  (1 - (2 / 5)) / (1 - (1 / 4)) = (4 / 5) := 
by 
  sorry

end evaluate_expression_l707_707592


namespace complex_quadrant_l707_707682

open Complex

theorem complex_quadrant (z1 z2 : ℂ) (hz1 : z1 = 2 - I) (h_sym : z2 = Complex.conj (-z1)) :
  let z := z1 / z2
  (z.re < 0 ∧ z.im > 0) :=
by
  let z : ℂ := z1 / z2
  have hz2 : z2 = -2 - I, from h_sym ▸ by rw [hz1]; simp
  have h_re : z.re = -3 / 5, from by simp [z, hz1, hz2]; linarith
  have h_im : z.im = 4 / 5, from by simp [z, hz1, hz2]; linarith
  exact ⟨h_re, h_im⟩

#check complex_quadrant

end complex_quadrant_l707_707682


namespace fraction_comparison_l707_707837

theorem fraction_comparison :
  (1998:ℝ) ^ 2000 / (2000:ℝ) ^ 1998 > (1997:ℝ) ^ 1999 / (1999:ℝ) ^ 1997 :=
by sorry

end fraction_comparison_l707_707837


namespace magnitude_subtract_vector_l707_707661

variables (a b : ℝ^3) -- Assume a and b are vectors in ℝ^3.
constants (a_mag : ℝ) (b_mag : ℝ) (theta : ℝ)
constants (a_len : |a| = 2) (b_len : |b| = 3) (angle : theta = (60:ℝ) * real.pi / 180)

theorem magnitude_subtract_vector : 
  |a - b| = real.sqrt 7 :=
by
  -- Statement and conditions
  have h_a : |a| = 2 := a_len,
  have h_b : |b| = 3 := b_len,
  have h_theta : theta = 60 * real.pi / 180 := angle,
  sorry

end magnitude_subtract_vector_l707_707661


namespace distinct_real_roots_l707_707715

-- Define an odd degree polynomial P with real coefficients
variable {P : Polynomial ℝ}

def isOddDegree (p : Polynomial ℝ) : Prop :=
  ∃ n, n % 2 = 1 ∧ p.degree = some n

theorem distinct_real_roots (h1 : isOddDegree P) :
    ∀ x : ℝ, P (P x) = 0 → ∃ y : ℝ, P y = 0 :=
  sorry

end distinct_real_roots_l707_707715


namespace balls_in_boxes_dist_count_l707_707305

theorem balls_in_boxes_dist_count : (∃ (f : Fin 4 → ℕ), (∑ i, f i = 5) ∧ fintype (Fin 4) = 4) → 56 :=
sorry

end balls_in_boxes_dist_count_l707_707305


namespace black_cards_remaining_proof_l707_707867

def initial_black_cards := 26
def black_cards_taken_out := 4
def black_cards_remaining := initial_black_cards - black_cards_taken_out

theorem black_cards_remaining_proof : black_cards_remaining = 22 := 
by sorry

end black_cards_remaining_proof_l707_707867


namespace algebraic_identity_l707_707678

theorem algebraic_identity (x Q : ℂ) (h : 2 * (5 * x + 3 * real.sqrt 2) = Q) :
  4 * (10 * x + 6 * real.sqrt 2) = 4 * Q := by
  sorry

end algebraic_identity_l707_707678


namespace cottage_cheese_quantity_l707_707991

theorem cottage_cheese_quantity (x : ℝ) 
    (milk_fat : ℝ := 0.05) 
    (curd_fat : ℝ := 0.155) 
    (whey_fat : ℝ := 0.005) 
    (milk_mass : ℝ := 1) 
    (h : (curd_fat * x + whey_fat * (milk_mass - x)) = milk_fat * milk_mass) : 
    x = 0.3 :=
    sorry

end cottage_cheese_quantity_l707_707991


namespace flowers_per_pot_l707_707106

theorem flowers_per_pot (total_pots total_flowers : ℕ) (h_pots : total_pots = 544) (h_flowers : total_flowers = 17408) :
    total_flowers / total_pots = 32 :=
by
  rw [h_pots, h_flowers]
  norm_num

end flowers_per_pot_l707_707106


namespace laptop_sticker_price_l707_707572

theorem laptop_sticker_price (y : ℝ) : 
  let final_price_C := 0.80 * y - 120
  let final_price_D := 0.70 * y
  (final_price_C = final_price_D + 10) → y = 1100 :=
begin
  intro h,
  sorry
end

end laptop_sticker_price_l707_707572


namespace find_c_l707_707972

noncomputable def f (c x : ℝ) : ℝ :=
  c * x^3 + 17 * x^2 - 4 * c * x + 45

theorem find_c (h : f c (-5) = 0) : c = 94 / 21 :=
by sorry

end find_c_l707_707972


namespace am_gm_inequality_l707_707398

theorem am_gm_inequality {n : ℕ} (a : Fin n → ℝ) 
  (h1 : n > 0) 
  (h2 : ∀ i, 0 < a i) 
  : 
  n * ∏ i, (a i) ≤ ∑ i, (a i) ^ n := 
begin 
  sorry -- proof goes here 
end

end am_gm_inequality_l707_707398


namespace number_of_perpendicular_lines_l707_707819

/-
Problem: Prove that the number of lines that intersect both given lines at a right angle in 3-dimensional space is:
        - Exactly 1 if the lines are skew.
        - Infinitely many if the lines are parallel.
        - Exactly 1 if the lines intersect.
-/
theorem number_of_perpendicular_lines 
  (L1 L2 : ℝ^3 → Prop) :
  (∃ p1 p2 : ℝ^3, (¬∃ x, L1 x ∧ L2 x) ∧ ¬parallel L1 L2 → (∃! L, perpendicular L L1 ∧ perpendicular L L2)) 
  ∨ 
  (parallel L1 L2 → (∃ L, perpendicular L L1 ∧ perpendicular L L2 ∧ ∃ infinitelyMany L)) 
  ∨ 
  (∃ x, L1 x ∧ L2 x → (∃! L, perpendicular L L1 ∧ perpendicular L L2)) 
:= sorry

end number_of_perpendicular_lines_l707_707819


namespace macy_miles_left_l707_707738

theorem macy_miles_left (goal : ℕ) (daily_miles : ℕ) (days_run : ℕ) 
  (H1 : goal = 24) 
  (H2 : daily_miles = 3) 
  (H3 : days_run = 6) 
  : goal - daily_miles * days_run = 6 := 
by 
  sorry

end macy_miles_left_l707_707738


namespace irrational_sum_sqrt2_pi_l707_707131

theorem irrational_sum_sqrt2_pi : ¬ ∃ (q : ℚ), (q : ℝ) = (real.sqrt 2) + real.pi := by 
  sorry

end irrational_sum_sqrt2_pi_l707_707131


namespace john_spent_correct_amount_l707_707016

-- Definitions of the conditions
def pin_prices : List ℝ := [23, 18, 20, 15, 25, 22, 19, 16, 24, 17]
def discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08
def exchange_rate : ℝ := 0.85

-- The Lean 4 statement to prove the final amount spent in Euros is approximately €155.28
theorem john_spent_correct_amount :
  let total_price := pin_prices.sum
  let discounted_price := total_price * (1 - discount_rate)
  let total_with_tax := discounted_price * (1 + sales_tax_rate)
  let final_price_euros := total_with_tax * exchange_rate
  final_price_euros ≈ 155.28 :=
by
  sorry

end john_spent_correct_amount_l707_707016


namespace initial_ratio_men_to_women_l707_707351

theorem initial_ratio_men_to_women (M W : ℕ) (h1 : (W - 3) * 2 = 24) (h2 : 14 - 2 = M) : M / gcd M W = 4 ∧ W / gcd M W = 5 := by 
  sorry

end initial_ratio_men_to_women_l707_707351


namespace max_value_condition_min_value_condition_l707_707861

theorem max_value_condition (x : ℝ) (h : x < 0) : (x^2 + x + 1) / x ≤ -1 :=
sorry

theorem min_value_condition (x : ℝ) (h : x > -1) : ((x + 5) * (x + 2)) / (x + 1) ≥ 9 :=
sorry

end max_value_condition_min_value_condition_l707_707861


namespace prob_zack_andrew_same_team_l707_707112

-- Define the conditions as Lean definitions/constants
def total_players := 27
def team_size := 9
def zack_idx : Fin total_players := 0  -- Assume Zack is at index 0
def mihir_idx : Fin total_players  -- Assume Mihir's index to be defined later
def andrew_idx : Fin total_players  -- Assume Andrew's index to be defined later

-- Define the probability proof statement
theorem prob_zack_andrew_same_team 
  (h1 : total_players = 27) 
  (h2 : team_size * 3 = total_players) 
  (hz_m : (zack_idx.to_nat / team_size) ≠ (mihir_idx.to_nat / team_size)) 
  (hm_a : (mihir_idx.to_nat / team_size) ≠ (andrew_idx.to_nat / team_size)) 
  (h_team_sizes : ∀ i, (team_size * i < total_players) → (team_size * (i + 1) ≤ total_players)) :
  ∃ i, (andrew_idx.to_nat / team_size) = i ∧ (zack_idx.to_nat / team_size) = i →
    (8 / 17 : ℚ) = 8 / 17 := by
  -- The proof is omitted
  sorry

end prob_zack_andrew_same_team_l707_707112


namespace problem1_problem2_l707_707658

-- Define the basic vectors
def a (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 * Real.sin α)
def b (α : ℝ) : ℝ × ℝ := (-Real.sin α, Real.cos α)

-- Derived vectors
def x (α : ℝ) (t : ℝ) : ℝ × ℝ := let (ax, ay) := a α; let (bx, by) := b α; (ax + (t^2 - 3) * bx, ay + (t^2 - 3) * by)
def y (α : ℝ) (k : ℝ) : ℝ × ℝ := let (ax, ay) := a α; let (bx, by) := b α; (-k * ax + bx, -k * ay + by)

-- Inner product condition
def inner_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
def condition (α t k : ℝ) : Prop := inner_product (x α t) (y α k) = 0

-- Mathematically equivalent proof problem
theorem problem1 (α t : ℝ) (h : condition α t (1/4 * (t^2 - 3))) : k = 1/4 * (t^2 - 3) :=
  sorry

theorem problem2 (t : ℝ) (λ : ℝ) (h1 : 0 ≤ t ∧ t ≤ 4)
  (h2 : ∀ t, t ∈ Icc 0 4 → (t^2 - λ * t + 3 + λ > 0)) : -3 < λ ∧ λ < 6 :=
  sorry

end problem1_problem2_l707_707658


namespace last_digit_fibonacci_mod_9_is_0_l707_707205

def Fibonacci (n : ℕ) : ℕ :=
  if n = 1 ∨ n = 2 then 1 else
  let F : ℕ → ℕ
    | 1 => 1
    | 2 => 1
    | n+2 => (F n + F (n + 1)) % 9
  in F n

theorem last_digit_fibonacci_mod_9_is_0 : ∃ N : ℕ, ∀ n : ℕ, n ≥ N → Fibonacci n % 9 ≠ 0 → Fibonacci n % 9 = 0 := sorry

end last_digit_fibonacci_mod_9_is_0_l707_707205


namespace min_value_of_expression_l707_707031

theorem min_value_of_expression (a b c : ℝ) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) (h3 : a * b * c = 27) :
  a^2 + 2*a*b + b^2 + 3*c^2 ≥ 324 :=
sorry

end min_value_of_expression_l707_707031


namespace train_speed_correct_l707_707553

variables {length_of_train length_of_bridge time_seconds : ℝ}
def train_speed_in_km_per_hr
  (length_of_train : ℝ) (length_of_bridge : ℝ) (time_seconds : ℝ) : ℝ := 
  ((length_of_train + length_of_bridge) / time_seconds) * 3.6

theorem train_speed_correct :
  train_speed_in_km_per_hr 150 225 30 = 45 := 
by
  unfold train_speed_in_km_per_hr
  norm_num
  calc ((150 + 225) / 30) * 3.6 = (375 / 30) * 3.6 : by norm_num
                           ... = 12.5 * 3.6     : by norm_num
                           ... = 45            : by norm_num

end train_speed_correct_l707_707553


namespace exists_inscribed_circle_with_radius_l707_707272

variable (ΔABC : Triangle) (R : ℝ) (P : Point)

axiom circumradius_eq_R : circumradius ΔABC = R
axiom point_inside : point_in_triangle P ΔABC

theorem exists_inscribed_circle_with_radius 
  (P : Point) (hP : point_in_triangle P ΔABC) :
  ∃ Δ, Δ ∈ {triangle P A B, triangle P B C, triangle P C A} ∧ circumradius Δ = R :=
sorry

end exists_inscribed_circle_with_radius_l707_707272


namespace centroid_of_triangle_ABC_l707_707372

noncomputable def distance_from_origin_to_plane (α β γ : ℝ) : ℝ :=
  1 / (real.sqrt ((1 / α^2) + (1 / β^2) + (1 / γ^2)))

theorem centroid_of_triangle_ABC (α β γ : ℝ) (h1 : α ≠ 0) (h2 : β ≠ 0) (h3 : γ ≠ 0) 
  (h_dist : distance_from_origin_to_plane α β γ = 2) :
  (1 / ((α / 3) ^ 2)) + (1 / ((β / 3) ^ 2)) + (1 / ((γ / 3) ^ 2)) = 2.25 :=
by 
  sorry

end centroid_of_triangle_ABC_l707_707372


namespace books_arrangement_l707_707307

theorem books_arrangement :
  ∃ (n : ℕ), n = 6.factorial / (2.factorial * 2.factorial) ∧ n = 180 :=
by
  use 180
  split
  . simp [Nat.factorial]
  . refl

end books_arrangement_l707_707307


namespace cat_catches_mouse_l707_707859

-- Define the distances
def AB := 200
def BC := 140
def CD := 20

-- Define the speeds (in meters per minute)
def mouse_speed := 60
def cat_speed := 80

-- Define the total distances the mouse and cat travel
def mouse_total_distance := 320 -- The mouse path is along a zigzag route initially specified in the problem
def cat_total_distance := AB + BC + CD -- 360 meters as calculated

-- Define the times they take to reach point D
def mouse_time := mouse_total_distance / mouse_speed -- 5.33 minutes
def cat_time := cat_total_distance / cat_speed -- 4.5 minutes

-- Proof problem statement
theorem cat_catches_mouse : cat_time < mouse_time := 
by
  sorry

end cat_catches_mouse_l707_707859


namespace object_speed_approximation_l707_707515

noncomputable def object_speed_in_mph (distance_in_feet : ℕ) (time_in_seconds : ℕ) : ℝ :=
  let distance_in_miles := (distance_in_feet : ℝ) / 5280
  let time_in_hours := (time_in_seconds : ℝ) / 3600
  distance_in_miles / time_in_hours

theorem object_speed_approximation :
  object_speed_in_mph 70 2 ≈ 23.856 := by
  -- sorry to skip the proof
  sorry

end object_speed_approximation_l707_707515


namespace area_of_square_l707_707346

theorem area_of_square (s : ℝ) (E F : ℝ × ℝ)
  (h1 : E.1 = s ∨ E.2 = 0)
  (h2 : F.1 = 0 ∨ F.2 = s)
  (h3 : E ≠ F)
  (h4 : ∃B C D : ℝ × ℝ, B = (0, s) ∧ C = (s, 0) ∧ D = (s, s) ∧ 
       (¬collinear (0,0) B E ) ∧ (¬collinear (0,0) D F))
  (h5 : (E.1 - 0) ^ 2 + (E.2 - 0) ^ 2 = 4 ^ 2)
  (h6 : (E.1 - F.1) ^ 2 + (E.2 - F.2) ^ 2 = 3 ^ 2) : 
  s ^ 2 = 256 / 17 :=
sorry

end area_of_square_l707_707346


namespace minimum_abs_sum_l707_707204

def abs_sum (x : ℝ) : ℝ := abs (x + 3) + abs (x + 5) + abs (x + 6) + abs (x + 7)

theorem minimum_abs_sum : ∃ x : ℝ, ∀ y : ℝ, abs_sum x ≤ abs_sum y :=
begin
  use -5,
  intros y,
  by_cases hy1 : y ≤ -7,
  { -- For y ≤ -7:
    have : abs_sum y = -4 * y - 21, by {
      rw [abs_of_nonpos (by linarith), abs_of_nonpos (by linarith), abs_of_nonpos (by linarith), abs_of_nonpos (by linarith)],
      ring,
    },
    have : abs_sum (-5) = 5, by norm_num,
    linarith,
  },
  by_cases hy2 : y ≤ -6,
  { -- For -7 < y ≤ -6:
    have : abs_sum y = -3 * y - 7, by {
      rw [abs_of_nonpos (by linarith), abs_of_nonpos (by linarith), abs_of_nonpos (by linarith), abs_of_nonneg (by linarith)],
      ring,
    },
    have : abs_sum (-5) = 5, by norm_num,
    linarith,
  },
  by_cases hy3 : y ≤ -5,
  { -- For -6 < y ≤ -5:
    have : abs_sum y = -y + 5, by {
      rw [abs_of_nonpos (by linarith), abs_of_nonpos (by linarith), abs_of_nonneg (by linarith), abs_of_nonneg (by linarith)],
      ring,
    },
    have : abs_sum (-5) = 5, by norm_num,
    linarith,
  },
  by_cases hy4 : y ≤ -3,
  { -- For -5 < y ≤ -3:
    have : abs_sum y = y + 15, by {
      rw [abs_of_nonneg (by linarith), abs_of_nonneg (by linarith), abs_of_nonneg (by linarith), abs_of_nonneg (by linarith)],
      ring,
    },
    have : abs_sum (-5) = 5, by norm_num,
    linarith,
  },
  { -- For y > -3:
    have : abs_sum y = 4 * y + 21, by {
      rw [abs_of_nonneg (by linarith), abs_of_nonneg (by linarith), abs_of_nonneg (by linarith), abs_of_nonneg (by linarith)],
      ring,
    },
    have : abs_sum (-5) = 5, by norm_num,
    linarith,
  },
  sorry
end

end minimum_abs_sum_l707_707204


namespace rank_of_matrix_A_l707_707971

def matrix_A : Matrix (Fin 4) (Fin 5) ℤ :=
  ![
    ![2, 1, 2, 1, 2],
    ![1, 1, 5, -2, 3],
    ![-1, 0, -4, 4, 1],
    ![3, 3, 8, 1, 9]
  ]

theorem rank_of_matrix_A : matrix.rank matrix_A = 3 :=
sorry

end rank_of_matrix_A_l707_707971


namespace compute_expression_l707_707733

def Q : ℂ := 7 + 3 * complex.I
def E : ℂ := 1 + complex.I
def D : ℂ := 7 - 3 * complex.I

theorem compute_expression : (Q * E * D)^2 = 8400 + 8000 * complex.I := by
  sorry

end compute_expression_l707_707733


namespace cards_red_side_up_l707_707158

theorem cards_red_side_up :
  let n := 100 in
  let divisible_by (x y : ℕ) := x % y = 0 in
  let initial_red := {k | 1 ≤ k ∧ k ≤ n} in
  let first_pass := {k ∈ initial_red | ¬ divisible_by k 2} in
  let odd_div3 := {k ∈ first_pass | divisible_by k 3} in
  let even_div3 := {k | k ∈ initial_red ∧ divisible_by k 2 ∧ divisible_by k 3} in
  (|first_pass| - |odd_div3| + |even_div3| = 49) :=
sorry

end cards_red_side_up_l707_707158


namespace success_permutations_correct_l707_707934

theorem success_permutations_correct :
  let word := "SUCCESS"
  let n := 7
  let s_count := 3
  let c_count := 2
  let u_count := 1
  let e_count := 1
  let total_permutations := (Nat.factorial n) / ((Nat.factorial s_count) * (Nat.factorial c_count) * (Nat.factorial u_count) * (Nat.factorial e_count))
  total_permutations = 420 :=
by
  sorry

end success_permutations_correct_l707_707934


namespace sqrt_of_36_is_6_l707_707510

theorem sqrt_of_36_is_6 : ∃ (x : ℕ), x * x = 36 ∧ x = 6 :=
by
  use 6
  sorry

end sqrt_of_36_is_6_l707_707510


namespace range_of_2a_minus_b_l707_707308

theorem range_of_2a_minus_b (a b : ℝ) (h1 : 2 < a) (h2 : a < 3) (h3 : 1 < b) (h4 : b < 2) :
  2 < 2 * a - b ∧ 2 * a - b < 5 := 
sorry

end range_of_2a_minus_b_l707_707308


namespace triangle_bc_length_l707_707007

theorem triangle_bc_length (A B C X : Type) [MetricSpace X] 
  (AB AC BC : ℝ) 
  (hAB : AB = 86)
  (hAC : AC = 97)
  (hCircle : ∀ X, dist A B = AB ∧ dist A X = AB → X ∈ SetOfXWithBCIntersection)
  (hBX CX : ℕ)
  (hSum : BX + CX = BC)
  : BC = 61 := 
sorry

end triangle_bc_length_l707_707007


namespace find_divisor_l707_707318

theorem find_divisor (x d : ℕ) (h1 : x ≡ 7 [MOD d]) (h2 : (x + 11) ≡ 18 [MOD 31]) : d = 31 := 
sorry

end find_divisor_l707_707318


namespace squareD_perimeter_l707_707418

-- Let perimeterC be the perimeter of square C
def perimeterC : ℝ := 40

-- Let sideC be the side length of square C
def sideC := perimeterC / 4

-- Let areaC be the area of square C
def areaC := sideC * sideC

-- Let areaD be the area of square D, which is one-third the area of square C
def areaD := (1 / 3) * areaC

-- Let sideD be the side length of square D
def sideD := Real.sqrt areaD

-- Let perimeterD be the perimeter of square D
def perimeterD := 4 * sideD

-- The theorem to prove
theorem squareD_perimeter (h : perimeterC = 40) (h' : areaD = (1 / 3) * areaC) : perimeterD = (40 * Real.sqrt 3) / 3 := by
  sorry

end squareD_perimeter_l707_707418


namespace complex_magnitude_l707_707681

variable (z : ℂ)

theorem complex_magnitude (h : z * (1 - 2 * complex.I) = 3 + complex.I) :
  complex.abs z = Real.sqrt 2 := 
sorry

end complex_magnitude_l707_707681


namespace max_value_l707_707727

noncomputable def find_max_value (x : ℝ) (hx : 0 < x) : ℝ :=
  (x^3 + 1 - real.sqrt (x^6 + 8)) / x

theorem max_value (x : ℝ) (hx : 0 < x) : 
  (x^3 + 1 - real.sqrt (x^6 + 8)) / x ≤ -1.25 + 1.25 * real.sqrt 2 := by
  sorry

end max_value_l707_707727


namespace arrangement_count_SUCCESS_l707_707936

-- Define the conditions for the problem
def letters : Finset String := {"S", "U", "C", "C", "E", "S", "S"}
def occurrences_S : Nat := 3
def occurrences_C : Nat := 2
def occurrences_other : Nat := 1 -- For 'U' and 'E'

-- State the theorem using these conditions
theorem arrangement_count_SUCCESS : 
  let N := letters.card
  N = 7 →
  occurrences_S = 3 →
  occurrences_C = 2 →
  occurrences_other = 1 →
  Nat.factorial N / (Nat.factorial occurrences_S * Nat.factorial occurrences_C * Nat.factorial occurrences_other * Nat.factorial occurrences_other) = 420 :=
by
  sorry

end arrangement_count_SUCCESS_l707_707936


namespace distance_scientific_notation_l707_707437

theorem distance_scientific_notation :
  55000000 = 5.5 * 10^7 :=
sorry

end distance_scientific_notation_l707_707437


namespace prob_zack_andrew_same_team_l707_707111

-- Define the conditions as Lean definitions/constants
def total_players := 27
def team_size := 9
def zack_idx : Fin total_players := 0  -- Assume Zack is at index 0
def mihir_idx : Fin total_players  -- Assume Mihir's index to be defined later
def andrew_idx : Fin total_players  -- Assume Andrew's index to be defined later

-- Define the probability proof statement
theorem prob_zack_andrew_same_team 
  (h1 : total_players = 27) 
  (h2 : team_size * 3 = total_players) 
  (hz_m : (zack_idx.to_nat / team_size) ≠ (mihir_idx.to_nat / team_size)) 
  (hm_a : (mihir_idx.to_nat / team_size) ≠ (andrew_idx.to_nat / team_size)) 
  (h_team_sizes : ∀ i, (team_size * i < total_players) → (team_size * (i + 1) ≤ total_players)) :
  ∃ i, (andrew_idx.to_nat / team_size) = i ∧ (zack_idx.to_nat / team_size) = i →
    (8 / 17 : ℚ) = 8 / 17 := by
  -- The proof is omitted
  sorry

end prob_zack_andrew_same_team_l707_707111


namespace perimeter_of_square_D_l707_707424

-- Definitions based on the conditions in the problem
def square (s : ℝ) := s * s

def perimeter (s : ℝ) := 4 * s

-- Given conditions
def perimeter_C : ℝ := 40
def side_length_C : ℝ := perimeter_C / 4
def area_C : ℝ := square side_length_C
def area_D : ℝ := area_C / 3
def side_length_D : ℝ := real.sqrt area_D

-- Proof statement to be proved
theorem perimeter_of_square_D : perimeter side_length_D = (40 * real.sqrt 3) / 3 := by
  sorry

end perimeter_of_square_D_l707_707424


namespace radius_increase_rate_l707_707777

theorem radius_increase_rate (r : ℝ) (u : ℝ)
  (h : r = 20) (dS_dt : ℝ) (h_dS_dt : dS_dt = 10 * Real.pi) :
  u = 1 / 4 :=
by
  have S := Real.pi * r^2
  have dS_dt_eq : dS_dt = 2 * Real.pi * r * u := sorry
  rw [h_dS_dt, h] at dS_dt_eq
  exact sorry

end radius_increase_rate_l707_707777


namespace final_apples_count_l707_707354

-- Definitions from the problem conditions
def initialApples : ℕ := 150
def soldToJill (initial : ℕ) : ℕ := initial * 30 / 100
def remainingAfterJill (initial : ℕ) := initial - soldToJill initial
def soldToJune (remaining : ℕ) : ℕ := remaining * 20 / 100
def remainingAfterJune (remaining : ℕ) := remaining - soldToJune remaining
def givenToFriend (current : ℕ) : ℕ := current - 2
def soldAfterFriend (current : ℕ) : ℕ := current * 10 / 100
def remainingAfterAll (current : ℕ) := current - soldAfterFriend current

theorem final_apples_count : remainingAfterAll (givenToFriend (remainingAfterJune (remainingAfterJill initialApples))) = 74 :=
by
  sorry

end final_apples_count_l707_707354


namespace sequence_formula_l707_707251

noncomputable def seq (a : ℕ+ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, (a n - 3) * a (n + 1) - a n + 4 = 0

theorem sequence_formula (a : ℕ+ → ℚ) (h : seq a) :
  ∀ n : ℕ+, a n = (2 * n - 1) / n :=
by
  sorry

end sequence_formula_l707_707251


namespace compute_h_x_l707_707920

variable (x : ℝ)

def given_expr : ℝ[x] := 4*x^4 + 2*x^2 - 5*x + 1
def target_expr : ℝ[x] := x^3 - 3*x^2 + 2*x - 4
def h_expr : ℝ[x] := -4*x^4 + x^3 - 5*x^2 + 7*x - 5

theorem compute_h_x : given_expr + h_expr = target_expr := 
by 
  unfold given_expr target_expr h_expr
  -- proof steps here
  sorry

end compute_h_x_l707_707920


namespace production_company_profit_l707_707167

noncomputable def profit (dom_earnings_opening_weekend : ℝ)
                         (dom_total_multiplier : ℝ)
                         (intl_multiplier : ℝ)
                         (keep_dom_percentage : ℝ)
                         (keep_intl_percentage : ℝ)
                         (production_cost : ℝ)
                         (marketing_cost : ℝ) : ℝ :=
  let dom_total := dom_earnings_opening_weekend * dom_total_multiplier in
  let intl_total := dom_total * intl_multiplier in
  let kept_dom := dom_total * keep_dom_percentage in
  let kept_intl := intl_total * keep_intl_percentage in
  let total_costs := production_cost + marketing_cost in
  (kept_dom + kept_intl) - total_costs

theorem production_company_profit :
  (profit 120 3.5 1.8 0.60 0.45 60 40) = 492.2 :=
by
  -- Proof goes here
  sorry

end production_company_profit_l707_707167


namespace find_n_mod_6_l707_707230

theorem find_n_mod_6 : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -7458 [MOD 6] :=
by
  use 0
  sorry

end find_n_mod_6_l707_707230


namespace find_circle_eq_find_line_eq_l707_707293

noncomputable def region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + 2 * y ≤ 4

noncomputable def circle_C (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + (y - 1) ^ 2 = 5

noncomputable def line_l (x y : ℝ) : ℝ → Prop
| b => y = x + b

theorem find_circle_eq : 
  (∀ x y, region x y → ∃ (C : ℝ × ℝ) (r : ℝ), circle_C (C.1) (C.2) ∧ r = 5 ∧ C.1 = 2 ∧ C.2 = 1) :=
sorry

theorem find_line_eq :
  ∀ A B, (∃ x1 y1 x2 y2, A = (x1, y1) ∧ B = (x2, y2) ∧ line_l y1 x1 = y1 - 1 + sqrt 5 ∧ circle_C x1 y1 ∧ circle_C x2 y2 ∧ ⟪(x1 - 2, y1-1) + (x2-2, y2-1)⟫ = 0) → 
  (∃ b, line_l (y - 1 + sqrt 5)) :=
sorry

end find_circle_eq_find_line_eq_l707_707293


namespace votes_diff_eq_70_l707_707054

noncomputable def T : ℝ := 350
def votes_against (T : ℝ) : ℝ := 0.40 * T
def votes_favor (T : ℝ) (X : ℝ) : ℝ := votes_against T + X

theorem votes_diff_eq_70 :
  ∃ X : ℝ, 350 = votes_against T + votes_favor T X → X = 70 :=
by
  sorry

end votes_diff_eq_70_l707_707054


namespace find_value_l707_707732

noncomputable def roots_polynomial := {p q r : ℝ // 
  Polynomial.aeval p (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 11 * Polynomial.X - Polynomial.C 3) = 0 ∧ 
  Polynomial.aeval q (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 11 * Polynomial.X - Polynomial.C 3) = 0 ∧ 
  Polynomial.aeval r (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 11 * Polynomial.X - Polynomial.C 3) = 0}

theorem find_value (p q r : ℝ) (h : p ∈ roots_polynomial ∧ q ∈ roots_polynomial ∧ r ∈ roots_polynomial) :
  (p / (q * r - 1) + q / (p * r - 1) + r / (p * q - 1)) = 17 / 29 :=
sorry

end find_value_l707_707732


namespace limit_one_plus_inv_x_to_e_l707_707758

-- Lean 4 statement for the proof problem
theorem limit_one_plus_inv_x_to_e (hx_pos : ∀ x, 0 < x) :
  (∀ x : ℝ, x ≠ 0 → tendsto (λ x, (1 + 1 / x) ^ x) at_top (𝓝 real.exp 1)) :=
begin
  sorry
end

end limit_one_plus_inv_x_to_e_l707_707758


namespace find_number_l707_707866

theorem find_number {x : ℝ} (h : (1/3) * x = 130.00000000000003) : x = 390 := 
sorry

end find_number_l707_707866


namespace girls_combined_avg_across_both_schools_l707_707949

-- Given conditions as definitions
def CedarBoysAvg := 68
def CedarGirlsAvg := 74
def CedarCombinedAvg := 70
def DixonBoysAvg := 75
def DixonGirlsAvg := 86
def DixonCombinedAvg := 80
def CombinedBoysAvg := 70

-- Hypotheses based on conditions
axiom CedarCombined : ∀ C c: ℝ, (68 * C + 74 * c) / (C + c) = 70
axiom DixonCombined : ∀ D d: ℝ, (75 * D + 86 * d) / (D + d) = 80
axiom BoysCombined : ∀ C D : ℝ, (68 * C + 75 * D) / (C + D) = 70

-- The theorem to be proved
theorem girls_combined_avg_across_both_schools : ∀ C c D d: ℝ, CedarCombined C c → DixonCombined D d → BoysCombined C D →
   (∀ C c D d : ℝ, C = 2 * c ∧ D = (6/5) * d →
    (C + D) * CedarGirlsAvg + (c + d) * DixonGirlsAvg = 81) := 
sorry

end girls_combined_avg_across_both_schools_l707_707949


namespace polynomial_roots_arithmetic_progression_l707_707439

theorem polynomial_roots_arithmetic_progression (m n : ℝ)
  (h : ∃ a : ℝ, ∃ d : ℝ, ∃ b : ℝ,
   (a = b ∧ (b + d) + (b + 2*d) + (b + 3*d) + b = 0) ∧
   (b * (b + d) * (b + 2*d) * (b + 3*d) = 144) ∧
   b ≠ (b + d) ∧ (b + d) ≠ (b + 2*d) ∧ (b + 2*d) ≠ (b + 3*d)) :
  m = -40 := sorry

end polynomial_roots_arithmetic_progression_l707_707439


namespace roger_trays_l707_707764

theorem roger_trays (trays_per_trip trips trays_first_table : ℕ) 
  (h1 : trays_per_trip = 4) 
  (h2 : trips = 3) 
  (h3 : trays_first_table = 10) : 
  trays_per_trip * trips - trays_first_table = 2 :=
by
  -- Step proofs are omitted
  sorry

end roger_trays_l707_707764


namespace anna_prob_l707_707912

noncomputable def probability_PLUM : ℚ :=
  let PLANET := { 'P', 'L', 'A', 'N', 'E', 'T' }.to_finset
  let CIRCUS := { 'C', 'I', 'R', 'C', 'U', 'S' }.to_finset
  let GAMES := { 'G', 'A', 'M', 'E', 'S' }.to_finset
  let prob_PLANET : ℚ := (Nat.choose 4 1).to_nat / (Nat.choose 6 3).to_nat -- P & L
  let prob_CIRCUS : ℚ := (Nat.choose 5 2).to_nat / (Nat.choose 6 3).to_nat -- U
  let prob_GAMES : ℚ := (Nat.choose 4 1).to_nat / (Nat.choose 5 2).to_nat -- M
  prob_PLANET * prob_CIRCUS * prob_GAMES
  
theorem anna_prob : probability_PLUM = 1 / 25 := by
  -- Convert the probabilities
  have h1 : (Nat.choose 4 1).to_nat = 4 := by sorry
  have h2 : (Nat.choose 6 3).to_nat = 20 := by sorry
  have h3 : (Nat.choose 5 2).to_nat = 10 := by sorry
  
  -- Calculate the individual probabilities
  have prob_PLANET : ℚ := 4 / 20 := by sorry
  have prob_CIRCUS : ℚ := 10 / 20 := by sorry
  have prob_GAMES : ℚ := 4 / 10 := by sorry

  -- Calculate the overall probability
  have : probability_PLUM = prob_PLANET * prob_CIRCUS * prob_GAMES := by sorry
  -- Explicitly show that the result is 1/25
  calc probability_PLUM
    = (4 / 20) * (10 / 20) * (4 / 10) : by sorry
    ... = 1 / 25 : by sorry

end anna_prob_l707_707912


namespace solve_beta_possible_values_l707_707364

noncomputable def possible_values_of_beta (β : ℂ) : Prop :=
  β ≠ -1 ∧ (|β^3 - 1| = 3 * |β - 1|) ∧ (|β^6 - 1| = 6 * |β - 1|) →
    (|β^3 + 1| = 3 ∧ |β^6 + 1| = 3)

theorem solve_beta_possible_values (β : ℂ) :
  possible_values_of_beta β :=
sorry

end solve_beta_possible_values_l707_707364


namespace saved_percentage_l707_707976

-- Define the given constants and values
def number_of_passengers : ℕ := 4
def cost_per_orange : ℝ := 1.5
def total_planned_spending : ℝ := 15

-- Define the computed cost of oranges if bought at the stop
def total_cost_of_oranges := number_of_passengers * cost_per_orange

-- Define the proportion of the original spending plan that the cost of the oranges represents
def proportion_saved := total_cost_of_oranges / total_planned_spending

-- Define the percentage of the money saved
def percentage_saved := proportion_saved * 100

-- Lean theorem statement
theorem saved_percentage :
  percentage_saved = 40 :=
sorry

end saved_percentage_l707_707976


namespace trees_in_garden_l707_707334

theorem trees_in_garden (yard_length distance_between_trees : ℕ) (h1 : yard_length = 800) (h2 : distance_between_trees = 32) :
  ∃ n : ℕ, n = (yard_length / distance_between_trees) + 1 ∧ n = 26 :=
by
  sorry

end trees_in_garden_l707_707334


namespace perpendicular_lines_imply_parallel_lines_l707_707631

variables {a b l : Line} {α : Plane}

theorem perpendicular_lines_imply_parallel_lines
  (h1 : a ⊥ α)
  (h2 : b ⊥ α) 
: a ∥ b := sorry

end perpendicular_lines_imply_parallel_lines_l707_707631


namespace abs_inequality_for_all_x_l707_707791

theorem abs_inequality_for_all_x (a : ℝ) : (∀ x : ℝ, |x - 2| + |x + 3| > a) ↔ a ∈ set.Iio 5 :=
by {
  sorry
}

end abs_inequality_for_all_x_l707_707791


namespace fourth_root_of_2560000_l707_707197

theorem fourth_root_of_2560000 :
  (2560000 : ℝ) = (256 * 10000) →
  (10000 : ℝ) = (10^4) →
  (256 : ℝ) = (16^2) →
  (Real.root 4 2560000) = 40 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end fourth_root_of_2560000_l707_707197


namespace success_permutations_correct_l707_707933

theorem success_permutations_correct :
  let word := "SUCCESS"
  let n := 7
  let s_count := 3
  let c_count := 2
  let u_count := 1
  let e_count := 1
  let total_permutations := (Nat.factorial n) / ((Nat.factorial s_count) * (Nat.factorial c_count) * (Nat.factorial u_count) * (Nat.factorial e_count))
  total_permutations = 420 :=
by
  sorry

end success_permutations_correct_l707_707933


namespace rectangles_in_grid_l707_707393

-- Define the grid size
def n : ℕ := 4

-- Prove the number of rectangles that can be formed on the n x n grid
theorem rectangles_in_grid : (∑ i in range (n+1), i * ((n + 1 - i) - 1)) * (∑ j in range (n+1), j * ((n + 1 - j) - 1)) = 36 := by
  sorry

end rectangles_in_grid_l707_707393


namespace measure_of_DEF_angle_l707_707546

-- Definitions according to conditions specified in part (a)

def is_regular_polygon (n : ℕ) (P : List (ℤ × ℤ)) : Prop :=
  (P.length = n) ∧ ∀ i, let (x1, y1) := P.nth_le i (sorry) in
  let (x2, y2) := P.nth_le ((i + 1) % n) (sorry) in
  (x1 - x2)^2 + (y1 - y2)^2 = (x2 - x1)^2 + (y2 - y1)^2

def coplanar (P Q : List (ℤ × ℤ)) : Prop :=
  ∀ p1 p2 ∈ P, ∀ q1 q2 ∈ Q, (let (x1, y1) := p1 in let (x2, y2) := p2 in y1 = y2) ∧
  (let (x1, y1) := q1 in let (x2, y2) := q2 in y1 = y2)

def opposite_sides (P : List (ℤ × ℤ)) (Q : List (ℤ × ℤ)) (A E : (ℤ × ℤ)) : Prop :=
  P.head = A ∧ P.nth_le 4 sorry = E ∧ Q.head = A ∧ Q.nth_le 1 sorry = E

-- The actual theorem statement based on proof problem

theorem measure_of_DEF_angle :
  ∃ (A E : (ℤ × ℤ)) (square_points octagon_points : List (ℤ × ℤ)),
  is_regular_polygon 5 square_points ∧ 
  is_regular_polygon 9 octagon_points ∧ 
  coplanar square_points octagon_points ∧ 
  opposite_sides square_points octagon_points A E →
  exterior_angle_measure DEF = 225 :=
begin
  sorry -- The proof is omitted as per the instructions
end

end measure_of_DEF_angle_l707_707546


namespace pentagon_total_area_l707_707194

-- Conditions definition
variables {a b c d e : ℕ}
variables {side1 side2 side3 side4 side5 : ℕ} 
variables {h : ℕ}
variables {triangle_area : ℕ}
variables {trapezoid_area : ℕ}
variables {total_area : ℕ}

-- Specific conditions given in the problem
def pentagon_sides (a b c d e : ℕ) : Prop :=
  a = 18 ∧ b = 25 ∧ c = 30 ∧ d = 28 ∧ e = 25

def can_be_divided (triangle_area trapezoid_area total_area : ℕ) : Prop :=
  triangle_area = 225 ∧ trapezoid_area = 770 ∧ total_area = 995

-- Total area of the pentagon under given conditions
theorem pentagon_total_area 
  (h_div: can_be_divided triangle_area trapezoid_area total_area) 
  (h_sides: pentagon_sides a b c d e)
  (h: triangle_area + trapezoid_area = total_area) :
  total_area = 995 := 
by
  sorry

end pentagon_total_area_l707_707194


namespace num_four_digit_integers_with_thousands_digit_3_l707_707663

theorem num_four_digit_integers_with_thousands_digit_3 : 
  ∃ n : ℕ, (n = 1000) ∧ 
    (∀ x : ℕ, (3000 ≤ x ∧ x < 4000) → 
      (exists_thousands_digit_3 : x / 1000 = 3)) := sorry

end num_four_digit_integers_with_thousands_digit_3_l707_707663


namespace miles_walked_on_Tuesday_l707_707014

theorem miles_walked_on_Tuesday (monday_miles total_miles : ℕ) (hmonday : monday_miles = 9) (htotal : total_miles = 18) :
  total_miles - monday_miles = 9 :=
by
  sorry

end miles_walked_on_Tuesday_l707_707014


namespace lawrence_walked_each_day_l707_707020

theorem lawrence_walked_each_day (d t : ℝ) (h1 : d = 3.0) (h2 : t = 12.0) : (t / d = 4.0) :=
by {
    rw [h1, h2],
    norm_num,
    sorry
}

end lawrence_walked_each_day_l707_707020


namespace jake_cat_food_l707_707836

theorem jake_cat_food (total_food : ℝ) (extra_food_second_cat : ℝ) (food_one_cat : ℝ) :
  total_food = 0.9 → extra_food_second_cat = 0.4 → food_one_cat = total_food - extra_food_second_cat → food_one_cat = 0.5 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jake_cat_food_l707_707836


namespace total_number_of_sequences_is_24_l707_707239

theorem total_number_of_sequences_is_24
  (guestA guestB guestC guestD security1 security2 : Type) :
  let entities := [⟨guestA, guestB⟩, guestC, guestD, security1, security2] in
  let possible_arrangements := [guestA, guestB, guestC, guestD, security1, security2].all_permutations in
  ∃ perm ∈ possible_arrangements, (perm.head = security1 ∧ perm.last = security2 ∧ are_together (guestA, guestB) perm) →
  perm.count = 24 :=
by sorry

end total_number_of_sequences_is_24_l707_707239


namespace days_playing_video_games_l707_707046

-- Define the conditions
def watchesTVDailyHours : ℕ := 4
def videoGameHoursPerPlay : ℕ := 2
def totalWeeklyHours : ℕ := 34
def weeklyTVDailyHours : ℕ := 7 * watchesTVDailyHours

-- Define the number of days playing video games
def playsVideoGamesDays (d : ℕ) : ℕ := d * videoGameHoursPerPlay

-- Define the number of days Mike plays video games
theorem days_playing_video_games (d : ℕ) :
  weeklyTVDailyHours + playsVideoGamesDays d = totalWeeklyHours → d = 3 :=
by
  -- The proof is omitted
  sorry

end days_playing_video_games_l707_707046


namespace prime_divisors_difference_l707_707359

def prime_factors (n : ℕ) : ℕ := sorry -- definition placeholder

theorem prime_divisors_difference (n : ℕ) (hn : 0 < n) : 
  ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ k - m = n ∧ prime_factors k - prime_factors m = 1 := 
sorry

end prime_divisors_difference_l707_707359


namespace marble_prob_red_or_white_l707_707154

theorem marble_prob_red_or_white :
  ∀ (total_marbles blue_marbles red_marbles white_marbles : ℕ),
    total_marbles = 20 →
    blue_marbles = 5 →
    red_marbles = 9 →
    white_marbles = total_marbles - (blue_marbles + red_marbles) →
    (red_marbles + white_marbles) / total_marbles = 3 / 4 :=
by
  intros total_marbles blue_marbles red_marbles white_marbles
  assume h1 : total_marbles = 20
  assume h2 : blue_marbles = 5
  assume h3 : red_marbles = 9
  assume h4 : white_marbles = total_marbles - (blue_marbles + red_marbles)
  sorry

end marble_prob_red_or_white_l707_707154


namespace g_1200_value_l707_707034

noncomputable def g : ℝ → ℝ := sorry

-- Assume the given condition as a definition
axiom functional_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y

-- Assume the given value of g(1000)
axiom g_1000_value : g 1000 = 4

-- Prove that g(1200) = 10/3
theorem g_1200_value : g 1200 = 10 / 3 := by
  sorry

end g_1200_value_l707_707034


namespace determine_b_l707_707579

noncomputable def has_exactly_one_real_solution (b : ℝ) : Prop :=
  ∃ x : ℝ, x^4 - b*x^3 - 3*b*x + b^2 - 2 = 0 ∧ ∀ y : ℝ, y ≠ x → y^4 - b*y^3 - 3*b*y + b^2 - 2 ≠ 0

theorem determine_b (b : ℝ) :
  has_exactly_one_real_solution b → b < 7 / 4 :=
by
  sorry

end determine_b_l707_707579


namespace small_cone_altitude_l707_707162

theorem small_cone_altitude (h_f : ℝ) (A_lower A_upper : ℝ) (h_f_alt : h_f = 18) 
  (A_lower_eq : A_lower = 144 * Real.pi) (A_upper_eq : A_upper = 36 * Real.pi) :
  let r_lower := Real.sqrt (A_lower / Real.pi)
      r_upper := Real.sqrt (A_upper / Real.pi)
      ratio := r_upper / r_lower
      H := h_f / (1 - ratio)
      h_s := H - h_f
  in h_s = 18 := 
by sorry

end small_cone_altitude_l707_707162


namespace log_add_log_five_two_l707_707864

theorem log_add_log_five_two : log 5 + log 2 = 1 := by
  sorry

end log_add_log_five_two_l707_707864


namespace solve_equation_l707_707076

theorem solve_equation (x : ℂ) (h : (x^2 + 3*x + 4) / (x + 3) = x + 6) : x = -7 / 3 := sorry

end solve_equation_l707_707076


namespace weather_conclusion_l707_707916

variables (T C : ℝ) (visitors : ℕ)

def condition1 : Prop :=
  (T ≥ 75.0 ∧ C < 10) → visitors > 100

def condition2 : Prop :=
  visitors ≤ 100

theorem weather_conclusion (h1 : condition1 T C visitors) (h2 : condition2 visitors) : 
  T < 75.0 ∨ C ≥ 10 :=
by 
  sorry

end weather_conclusion_l707_707916


namespace median_of_81_consecutive_integers_sum_3_pow_8_l707_707098

theorem median_of_81_consecutive_integers_sum_3_pow_8 :
  ∃ n : ℤ, (sum (list.range 81).map (λ x, x + n) = 3^8) ∧ (list.range 81).nth 40 = some n := sorry

end median_of_81_consecutive_integers_sum_3_pow_8_l707_707098


namespace cylinder_volume_l707_707455

-- Define the volume of the cone
def V_cone : ℝ := 18.84

-- Define the volume of the cylinder
def V_cylinder : ℝ := 3 * V_cone

-- Prove that the volume of the cylinder is 56.52 cubic meters
theorem cylinder_volume :
  V_cylinder = 56.52 := 
by 
  -- the proof will go here
  sorry

end cylinder_volume_l707_707455


namespace equidistant_implies_perpendicular_plane_l707_707326

variables (A B : Point) (α : Plane)

-- Definition of points being at equal distance from a plane
def equidistant (A B : Point) (α : Plane) : Prop :=
  distance A α = distance B α

theorem equidistant_implies_perpendicular_plane
  (h_diff : A ≠ B)
  (h_eqdist : equidistant A B α) :
  exists (β : Plane), (perpendicular β α ∧ A ∈ β ∧ B ∈ β) :=
sorry

end equidistant_implies_perpendicular_plane_l707_707326


namespace conjugate_in_second_quadrant_l707_707434

theorem conjugate_in_second_quadrant :
  (let z : ℂ := (2 : ℂ) / (⟨-1, 1⟩ : ℂ) in
  let conj_z := conj z in
  conj_z.re < 0 ∧ conj_z.im > 0) :=
by
  sorry

end conjugate_in_second_quadrant_l707_707434


namespace find_AB_l707_707695

variables (A B C D P Q : Type) [rect : rectangle A B C D]
variables [proj_P_on_AD : project P AD Q]
variables (x : ℝ) (h_AB_PQ : x = AB) 
variables (h_BP : BP = 12) (h_CP : CP = 6) (h_tan_APD : real.tan (angle A P D) = 2)

theorem find_AB : AB = 12 :=
by
  sorry

end find_AB_l707_707695


namespace max_average_hours_l707_707386

theorem max_average_hours :
  let hours_Wednesday := 2
  let hours_Thursday := 2
  let hours_Friday := hours_Wednesday + 3
  let total_hours := hours_Wednesday + hours_Thursday + hours_Friday
  let average_hours := total_hours / 3
  average_hours = 3 :=
by
  sorry

end max_average_hours_l707_707386


namespace salary_spending_l707_707152

theorem salary_spending (S_A S_B : ℝ) (P_A P_B : ℝ) 
  (h1 : S_A = 4500) 
  (h2 : S_A + S_B = 6000)
  (h3 : P_B = 0.85) 
  (h4 : S_A * (1 - P_A) = S_B * (1 - P_B)) : 
  P_A = 0.95 :=
by
  -- Start proofs here
  sorry

end salary_spending_l707_707152


namespace twelve_times_reciprocal_sum_l707_707475

theorem twelve_times_reciprocal_sum (a b c : ℚ) (h₁ : a = 1/3) (h₂ : b = 1/4) (h₃ : c = 1/6) :
  12 * (a + b + c)⁻¹ = 16 := 
by
  sorry

end twelve_times_reciprocal_sum_l707_707475


namespace not_all_mages_are_wizards_l707_707689

variable (M S W : Type → Prop)

theorem not_all_mages_are_wizards
  (h1 : ∃ x, M x ∧ ¬ S x)
  (h2 : ∀ x, M x ∧ W x → S x) :
  ∃ x, M x ∧ ¬ W x :=
sorry

end not_all_mages_are_wizards_l707_707689


namespace volume_of_prism_l707_707108

variables (a b c : ℝ)

def face_areas (a b c : ℝ) :=
(a * b = 100) ∧ (b * c = 200) ∧ (c * a = 300)

theorem volume_of_prism (a b c : ℝ) (h : face_areas a b c) : 
  a * b * c ≈ 2449 :=
sorry

end volume_of_prism_l707_707108


namespace sum_palindromic_primes_lt_75_l707_707395

/-- 
A function to reverse the digits of a two-digit number
-/
def reverseDigits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

/--
Predicate to determine if a number is a palindromic prime
-/
def isPalindromicPrime (n : ℕ) : Prop :=
  Prime n ∧ Prime (reverseDigits n)

/--
Function to compute the sum of all palindromic primes less than a given number
-/
def sumPalindromicPrimes (n : ℕ) : ℕ :=
  (Finset.filter isPalindromicPrime (Finset.range n)).sum id

theorem sum_palindromic_primes_lt_75 :
  sumPalindromicPrimes 75 = 253 :=
by
  sorry

end sum_palindromic_primes_lt_75_l707_707395


namespace geometric_sequence_second_term_l707_707807

theorem geometric_sequence_second_term (b : ℝ) (hb : b > 0) 
  (h1 : ∃ r : ℝ, 210 * r = b) 
  (h2 : ∃ r : ℝ, b * r = 135 / 56) : 
  b = 22.5 := 
sorry

end geometric_sequence_second_term_l707_707807


namespace min_value_expression_l707_707366

variable (a b c d : ℝ)

theorem min_value_expression (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  min (
    (a + b + c) / d + (a + b + d) / c +
    (a + c + d) / b + (b + c + d) / a
  ) 12 :=
by sorry

end min_value_expression_l707_707366


namespace translated_quadratic_range_of_translated_quadratic_min_value_on_interval_l707_707863

def f_translation : ℝ → ℝ :=
  fun x => (x - 1)^2 - 2

def f_quad (a x : ℝ) : ℝ :=
  x^2 - 2*a*x - 1

theorem translated_quadratic :
  ∀ x, f_translation x = (x - 1)^2 - 2 :=
by
  intro x
  simp [f_translation]

theorem range_of_translated_quadratic :
  ∀ x, 0 ≤ x ∧ x ≤ 4 → -2 ≤ f_translation x ∧ f_translation x ≤ 7 :=
by
  sorry

theorem min_value_on_interval :
  ∀ a, 
    (a ≤ 0 → ∀ x, 0 ≤ x ∧ x ≤ 2 → (f_quad a x ≥ -1)) ∧
    (0 < a ∧ a < 2 → f_quad a a = -a^2 - 1) ∧
    (a ≥ 2 → ∀ x, 0 ≤ x ∧ x ≤ 2 → (f_quad a 2 = -4*a + 3)) :=
by
  sorry

end translated_quadratic_range_of_translated_quadratic_min_value_on_interval_l707_707863


namespace count_integers_in_abs_leq_six_l707_707664

theorem count_integers_in_abs_leq_six : 
  {x : ℤ | abs (x - 3) ≤ 6}.card = 13 := 
by
  sorry

end count_integers_in_abs_leq_six_l707_707664


namespace range_of_a_if_f_decreasing_l707_707313

noncomputable def f (a x : ℝ) : ℝ := Real.sqrt (x^2 - a * x + 4)

theorem range_of_a_if_f_decreasing:
  ∀ (a : ℝ),
    (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y → f a y < f a x) →
    2 ≤ a ∧ a ≤ 5 :=
by
  intros a h
  sorry

end range_of_a_if_f_decreasing_l707_707313


namespace ratio_of_ages_l707_707452

-- Given conditions
def present_age_sum (H J : ℕ) : Prop :=
  H + J = 43

def present_ages (H J : ℕ) : Prop := 
  H = 27 ∧ J = 16

def multiple_of_age (H J k : ℕ) : Prop :=
  H - 5 = k * (J - 5)

-- Prove that the ratio of Henry's age to Jill's age 5 years ago was 2:1
theorem ratio_of_ages (H J k : ℕ) 
  (h_sum : present_age_sum H J)
  (h_present : present_ages H J)
  (h_multiple : multiple_of_age H J k) :
  (H - 5) / (J - 5) = 2 :=
by
  sorry

end ratio_of_ages_l707_707452


namespace problem_inequality_l707_707637

theorem problem_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 ≥ 6 * real.sqrt 3 ∧ 
  (a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 = 6 * real.sqrt 3 ↔ a = b ∧ b = c ∧ a = real.sqrt (real.sqrt 3)) :=
sorry

end problem_inequality_l707_707637


namespace head_start_l707_707514

namespace RaceProblem

variables (v_A v_B L H : ℝ)
hypotheses (h1 : v_A = 16 / 15 * v_B)
(h2 : L / v_A = (L - H) / v_B)

theorem head_start : H = L / 16 :=
by
  sorry

end RaceProblem

end head_start_l707_707514


namespace twice_age_in_2001_l707_707685

def age_in_year_2005 := 2005
def brother_age : ℕ := 16
def sister_age : ℕ := 10

theorem twice_age_in_2001 : ∃ n : ℤ, (n : ℕ) = 4 ∧ (age_in_year_2005 - n) = 2001 :=
by
  -- using ℤ for n as it results in a negative number and conversion to ℕ for comparison
  use -4
  split
  { -- proof that n = 4 (from -4 since we are going back in time from 2005 by 4 years)
    norm_num
  },
  { -- proof that the year is 2001
    norm_num,
    exact eq.refl 2001
  }

end twice_age_in_2001_l707_707685


namespace average_first_8_matches_l707_707431

/--
Assume we have the following conditions:
1. The average score for 12 matches is 48 runs.
2. The average score for the last 4 matches is 64 runs.
Prove that the average score for the first 8 matches is 40 runs.
-/
theorem average_first_8_matches (A1 A2 : ℕ) :
  (A1 / 12 = 48) → 
  (A2 / 4 = 64) →
  ((A1 - A2) / 8 = 40) :=
by
  sorry

end average_first_8_matches_l707_707431


namespace integral_f_value_l707_707367

def f (x : ℝ) : ℝ :=
  if x >= -1 ∧ x < 1 then real.sqrt (1 - x^2)
  else if x >= 1 ∧ x <= 2 then x^2 - 1
  else 0

theorem integral_f_value :
  ∫ x in -1..2, f x = (real.pi / 2) + (4 / 3) :=
by
  sorry

end integral_f_value_l707_707367


namespace find_incorrect_option_l707_707081

-- The given conditions from the problem
def incomes : List ℝ := [2, 2.5, 2.5, 2.5, 3, 3, 3, 3, 3, 4, 4, 5, 5, 9, 13]
def mean_incorrect : Prop := (incomes.sum / incomes.length) = 4
def option_incorrect : Prop := ¬ mean_incorrect

-- The goal is to prove that the statement about the mean being 4 is incorrect
theorem find_incorrect_option : option_incorrect := by
  sorry

end find_incorrect_option_l707_707081


namespace travel_time_reduction_impossible_proof_l707_707562

noncomputable def travel_time_reduction_impossible : Prop :=
  ∀ (x : ℝ), x > 60 → ¬ (1 / x * 60 = 1 - 1)

theorem travel_time_reduction_impossible_proof : travel_time_reduction_impossible :=
sorry

end travel_time_reduction_impossible_proof_l707_707562


namespace ai_is_permutation_of_0_to_n_minus_1_l707_707773

variable {n : ℕ}
variable (j : ℕ → ℕ) (hj : ∀ i, 0 ≤ j(i) < n)
variable (hj_valid : Function.Bijective (fun i => (j i + i) % n))

def a (i : ℕ) : ℕ := (j i + i) % n

theorem ai_is_permutation_of_0_to_n_minus_1 :
    ∀ i, i < n → 
    ∀ j, j < n → 
    (∃! i, a(j) = i) := sorry

end ai_is_permutation_of_0_to_n_minus_1_l707_707773


namespace lemonade_calories_l707_707241

theorem lemonade_calories 
    (lime_juice_weight : ℕ)
    (lime_juice_calories_per_grams : ℕ)
    (sugar_weight : ℕ)
    (sugar_calories_per_grams : ℕ)
    (water_weight : ℕ)
    (water_calories_per_grams : ℕ)
    (mint_weight : ℕ)
    (mint_calories_per_grams : ℕ)
    :
    lime_juice_weight = 150 →
    lime_juice_calories_per_grams = 30 →
    sugar_weight = 200 →
    sugar_calories_per_grams = 390 →
    water_weight = 500 →
    water_calories_per_grams = 0 →
    mint_weight = 50 →
    mint_calories_per_grams = 7 →
    (300 * ((150 * 30 + 200 * 390 + 500 * 0 + 50 * 7) / 900) = 276) :=
by
  sorry

end lemonade_calories_l707_707241


namespace count_success_permutations_l707_707939

theorem count_success_permutations : 
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  (Nat.factorial total_letters) / ((Nat.factorial s_count) * (Nat.factorial c_count)) = 420 := 
by
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  sorry

end count_success_permutations_l707_707939


namespace inequality_has_real_solutions_l707_707932

noncomputable
def satisfies_inequality (d : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 8 * x + d < 0

def has_real_solutions_f (d : ℝ) : Prop :=
  d ∈ set.Ioo 0 16

theorem inequality_has_real_solutions (d : ℝ) (h : d > 0) :
  satisfies_inequality d ↔ has_real_solutions_f d :=
sorry

end inequality_has_real_solutions_l707_707932


namespace correct_proposition_2_correct_proposition_4_final_correct_propositions_l707_707511

-- Define the initial condition for the contrapositive statement.
def contrapositive_statement :=
  ∀ (a b : ℕ), (¬ (a ≠ 0 ∧ b ≠ 0)) → (a = 0 ∨ b = 0)

-- Proof problem for proposition ②.
theorem correct_proposition_2 : contrapositive_statement :=
by
  sorry

-- Define the function and its derivative for the extreme values problem.
def func (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2
def func_deriv (x : ℝ) : ℝ := 6 * x ^ 2 - 6 * x

-- Proof problem for proposition ④.
theorem correct_proposition_4 :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ func_deriv x1 = 0 ∧ func_deriv x2 = 0 ∧ 
  ((∀ y : ℝ, y < x1 → func_deriv y > 0) ∧ (∀ y : ℝ, y > x2 → func_deriv y > 0)) ∧
  (∀ y : ℝ, x1 < y ∧ y < x2 → func_deriv y < 0)) :=
by
  sorry

-- Final theorem combining the correct propositions
theorem final_correct_propositions :
  correct_proposition_2 ∧ correct_proposition_4 :=
by
  exact ⟨correct_proposition_2, correct_proposition_4⟩

end correct_proposition_2_correct_proposition_4_final_correct_propositions_l707_707511


namespace find_integer_l707_707822

theorem find_integer (n : ℤ) (h1 : 0 ≤ n) (h2 : n < 9) (h3 : -1234 ≡ n [MOD 9]) : n = 8 := 
sorry

end find_integer_l707_707822


namespace exists_line_equidistant_from_AB_CD_l707_707260

noncomputable def Line : Type := sorry  -- This would be replaced with an appropriate definition of a line in space

def Point : Type := sorry  -- Similarly, a point in space type definition

variables (A B C D : Point)

def perpendicularBisector (P Q : Point) : Type := sorry  -- Definition for perpendicular bisector plane of two points

def is_perpendicularBisector_of (e : Line) (P Q : Point) : Prop := sorry  -- e is perpendicular bisector plane of P and Q

theorem exists_line_equidistant_from_AB_CD (A B C D : Point) :
  ∃ e : Line, is_perpendicularBisector_of e A C ∧ is_perpendicularBisector_of e B D :=
by
  sorry

end exists_line_equidistant_from_AB_CD_l707_707260


namespace volume_ratio_l707_707120

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

-- Definitions
variables (r x : ℝ)

-- Conditions
def golden_ratio_condition : Prop := 
  x / r = r / (r + x)

def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r ^ 3

def volume_cone (r₁ h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r₁ ^ 2 * h

-- Theorem: Prove the ratio of the volumes
theorem volume_ratio (r x r₁ : ℝ) (hr₁ : r₁ * r₁ = r * x)
  (hr_ratio : golden_ratio_condition r x) : 
  volume_sphere r / (volume_cone r₁ (r + x)) = 4 :=
sorry

end volume_ratio_l707_707120


namespace snakes_not_hiding_l707_707567

theorem snakes_not_hiding (total_snakes : ℕ) (hiding_snakes : ℕ) :
  total_snakes = 95 →
  hiding_snakes = 64 →
  total_snakes - hiding_snakes = 31 :=
by
  intros h1 h2
  rw [h1, h2]
  simp
  sorry

end snakes_not_hiding_l707_707567


namespace area_PMN_8th_area_ABC_l707_707328

-- Definitions for points and regions
variables {Point : Type} [MetricSpace Point] {A B C D E M N P : Point}
variables (triangle : Triangle Point)
variables (midpoint_AE mid_AE : Point)
variables (midpoint_BC mid_BC : Point)
variables (centroid_M centroid_M : Point)

-- Assumptions based on the problem
def is_median (triangle : Triangle Point) (M : Point) (D E : Point) : Prop :=
  triangle.has_median M D && triangle.has_median M E

def is_centroid (triangle : Triangle Point) (M : Point) : Prop :=
  is_median triangle M centroid_M

def midpoint (P Q M : Point) : Prop :=
  dist P M = dist Q M

-- Conditions as assumptions
axiom AD_median : is_median triangle A D
axiom CE_median : is_median triangle C E
axiom M_centroid : is_centroid triangle M
axiom N_midpoint : midpoint A E N
axiom P_midpoint : midpoint B C P

-- Goal: Area[PMN] = (1/8) * Area[ABC]
theorem area_PMN_8th_area_ABC (Area : Triangle Point → ℝ) :
  Area (mk_triangle P M N) = (1 / 8) * Area (mk_triangle A B C) :=
sorry

end area_PMN_8th_area_ABC_l707_707328


namespace speed_of_train_is_35_l707_707513

-- Definitions based on the conditions
def length_of_train : ℝ := 560  -- in meters
def time_to_cross_post : ℝ := 16  -- in seconds

-- The speed of the train is defined as distance (length of train) divided by time.
def speed_of_train (length : ℝ) (time : ℝ) : ℝ := length / time

-- Theorem statement
theorem speed_of_train_is_35 : speed_of_train length_of_train time_to_cross_post = 35 := by
  sorry

end speed_of_train_is_35_l707_707513


namespace variance_of_series_is_2_l707_707254

theorem variance_of_series_is_2 :
  let data := [1, 2, 3, 4, 5 : ℝ]
  let mean := (1 + 2 + 3 + 4 + 5) / 5
  let variance := (1 / 5) * List.sum (List.map (λ x, (x - mean)^2) data)
  variance = 2 :=
by
  -- provided that all computations are correct
  sorry

end variance_of_series_is_2_l707_707254


namespace irreducible_polynomial_sequence_l707_707770

theorem irreducible_polynomial_sequence (S : Set ℕ) : 
  (∃ (a_i : ℕ → ℕ), (∀ i, a_i i ∈ S) ∧ ∀ n, Irreducible (∑ i in Finset.range (n + 1), a_i i * (x ^ i))) ↔ S.card ≥ 2 := 
sorry

end irreducible_polynomial_sequence_l707_707770


namespace evaluate_expression_l707_707948

theorem evaluate_expression :
  2 * (Real.log2 (1 / 4)) + Real.log10 (1 / 100) + (Real.sqrt 2 - 1) ^ (Real.log10 1) = -5 :=
by sorry

end evaluate_expression_l707_707948


namespace modulus_of_z_minus_1_l707_707377

theorem modulus_of_z_minus_1 :
  let z := (⟨-1, -2⟩ : ℂ) / (⟨0, 1⟩ : ℂ)
  in complex.abs (z - 1) = real.sqrt 2 :=
by
  let z := (⟨-1, -2⟩ : ℂ) / (⟨0, 1⟩ : ℂ)
  sorry

end modulus_of_z_minus_1_l707_707377


namespace shaded_area_fraction_l707_707781

-- Define the problem conditions
def total_squares : ℕ := 18
def half_squares : ℕ := 10
def whole_squares : ℕ := 3

-- Define the total shaded area given the conditions
def shaded_area := (half_squares * (1/2) + whole_squares)

-- Define the total area of the rectangle
def total_area := total_squares

-- Lean 4 theorem statement
theorem shaded_area_fraction :
  shaded_area / total_area = (4 : ℚ) / 9 :=
by sorry

end shaded_area_fraction_l707_707781


namespace variance_transformed_l707_707278

-- Given definitions
variables (n : ℕ) (a : ℕ → ℝ) (σ : ℝ)
-- Condition: Variance of the original data
def variance_a := (1 / n) * (∑ i in finset.range n, (a i - (1 / n) * (∑ j in finset.range n, a j)) ^ 2)
-- Assume the given condition about the variance
axiom variance_a_sigma : variance_a n a = σ^2

-- Prove the variance of the transformed data
def variance_2a := (1 / n) * (∑ i in finset.range n, (2 * a i - (1 / n) * (∑ j in finset.range n, 2 * a j)) ^ 2)

theorem variance_transformed : variance_2a n a = 4 * σ^2 :=
by
  sorry

end variance_transformed_l707_707278


namespace solve_for_V_l707_707265

open Real

theorem solve_for_V :
  ∃ k V, 
    (U = k * (V / W) ∧ (U = 16 ∧ W = 1 / 4 ∧ V = 2) ∧ (U = 25 ∧ W = 1 / 5 ∧ V = 2.5)) :=
by {
  sorry
}

end solve_for_V_l707_707265


namespace largest_value_is_D_l707_707134

theorem largest_value_is_D :
  let A := 15432 + 1/3241
  let B := 15432 - 1/3241
  let C := 15432 * (1/3241)
  let D := 15432 / (1/3241)
  let E := 15432.3241
  max (max (max A B) (max C D)) E = D := by
{
  sorry -- proof not required
}

end largest_value_is_D_l707_707134


namespace magnitude_cos_sin_eq_one_l707_707257

theorem magnitude_cos_sin_eq_one (x : ℝ) : 
  let a := (Real.cos x, Real.sin x) in |a| = 1 :=
by
  sorry

end magnitude_cos_sin_eq_one_l707_707257


namespace base_ten_representation_HMT_l707_707432

theorem base_ten_representation_HMT (H M T : ℕ) 
  (H_digit : H < 10) (M_digit : M < 10) (T_digit : T < 10)
  (H_zero : H = 0)
  (sum_mod_nine : T + M ≡ 4 [MOD 9])
  (alt_sum_mod_eleven : T - M ≡ -3 [MOD 11]) :
  H + M + T = 17 := by
  sorry

end base_ten_representation_HMT_l707_707432


namespace a_gt_b_iff_a_cubed_gt_b_cubed_l707_707030

theorem a_gt_b_iff_a_cubed_gt_b_cubed (a b : ℝ) : a > b ↔ a ^ 3 > b ^ 3 :=
begin
  sorry
end

end a_gt_b_iff_a_cubed_gt_b_cubed_l707_707030


namespace circle_angle_ACB_eq_23_l707_707686

theorem circle_angle_ACB_eq_23
  (O A B C D : Point)  -- Points O, A, B, C, and D
  (hO_center : Circle O)
  (hA_on_circle : OnCircle A O)
  (hB_on_circle : OnCircle B O)
  (hC_on_circle : OnCircle C O)
  (hOA_parallel_BC : Parallel (LineSegment O A) (LineSegment B C))
  (hD_intersection : Intersection (LineSegment O B) (LineSegment A C) = D)
  (hBDC_eq : ∠ B D C = (2 * β - 1 : ℝ))
  (hACB_eq : ∠ A C B = γ) :
  γ = 23 :=
sorry

end circle_angle_ACB_eq_23_l707_707686


namespace dot_product_condition_satisfaction_l707_707381

variable (a b : ℝ^3)

def vector_length (v : ℝ^3) : ℝ := real.sqrt (v.dot_product v)

theorem dot_product_condition_satisfaction (a b : ℝ^3) (h₁ : vector_length (a + b) = real.sqrt 10) (h₂ : vector_length (a - b) = real.sqrt 6) : a.dot_product b = 1 :=
sorry

end dot_product_condition_satisfaction_l707_707381


namespace angle_bisector_segments_l707_707347

noncomputable def divide_leg_BC (A B C S : Point ℝ) (BF FC : ℝ) :=
  -- Define the properties of the triangle ABC
  right_triangle ABC ∧ 
  distance A B = 1 ∧
  angle A B C = 30 ∧ 
  centroid S A B C ∧
  angle_bisector S A B C B F C ∧
  
  -- Define the lengths of the segments BF and FC
  BF + FC = distance B C ∧
  BF/FC = distance B S / distance S C

-- Define the theorem we want to state and prove
theorem angle_bisector_segments : 
  ∀ (A B C S : Point ℝ) (BF FC : ℝ), 
  divide_leg_BC A B C S BF FC → 
  BF / FC = sorry ∧ BF + FC = BC := 
-- The proof is left as an exercise
by 
  sorry
  -- The proof should involve setting up the coordinates of points, using trigonometric identities, and applying geometric theorems.

end angle_bisector_segments_l707_707347


namespace min_total_spent_l707_707803

theorem min_total_spent : 
  ∀(prices : list ℕ), 
  (∀ p, p ∈ prices → p ∈ (list.range 21).tail) ∧ prices.length = 20 → 
  (∃ min_cost, min_cost = 136 ∧ min_cost = 
  list.sum prices - list.sum (list.take 4 (list.reverse (list.sort prices)))) :=
begin
  intro prices,
  intro h,
  have mem_range := h.1,
  have length_20 := h.2,
  use 136,
  split,
  { refl },
  { have sorted_prices := list.sort prices,
    have reversed_sorted_prices := list.reverse sorted_prices,
    have top_4_free := list.take 4 reversed_sorted_prices,
    have all_items_sum := list.sum prices,
    have free_items_sum := list.sum top_4_free,
    have min_cost := all_items_sum - free_items_sum,
    exact min_cost }
end

end min_total_spent_l707_707803


namespace non_square_300th_term_l707_707498

theorem non_square_300th_term (N : ℕ) (hN : N = 300) : 
  ∃ x : ℕ, (∀ (t : ℕ), 0 < t ∧ t ≤ x → ¬ (∃ k : ℕ, t = k^2)) ∧ 
           (∑ t in finset.range (N + 17), if ∃ k : ℕ, t = k^2 then 0 else 1 = N) ∧ 
           x = 317 := by
  sorry

end non_square_300th_term_l707_707498


namespace stamens_in_bouquet_l707_707144

-- Define the number of pistils, leaves, stamens for black roses and crimson flowers
def pistils_black_rose : ℕ := 4
def stamens_black_rose : ℕ := 4
def leaves_black_rose : ℕ := 2

def pistils_crimson_flower : ℕ := 8
def stamens_crimson_flower : ℕ := 10
def leaves_crimson_flower : ℕ := 3

-- Define the number of black roses and crimson flowers (as variables x and y)
variables (x y : ℕ)

-- Define the total number of pistils and leaves in the bouquet
def total_pistils : ℕ := pistils_black_rose * x + pistils_crimson_flower * y
def total_leaves : ℕ := leaves_black_rose * x + leaves_crimson_flower * y

-- Condition: There are 108 fewer leaves than pistils
axiom leaves_pistils_relation : total_leaves = total_pistils - 108

-- Calculate the total number of stamens in the bouquet
def total_stamens : ℕ := stamens_black_rose * x + stamens_crimson_flower * y

-- The theorem to be proved
theorem stamens_in_bouquet : total_stamens = 216 :=
by
  sorry

end stamens_in_bouquet_l707_707144


namespace find_x_for_parallel_line_plane_l707_707779

variables (s : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ)

def is_orthogonal (a b : ℝ × ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0

theorem find_x_for_parallel_line_plane (x : ℝ) :
  let s := (-1, 1, 1) in
  let n := (2, x^2 + x, -x^2) in
  is_orthogonal s n → x = 2 :=
by
  sorry

end find_x_for_parallel_line_plane_l707_707779


namespace units_digit_6_power_l707_707508

theorem units_digit_6_power (n : ℕ) : (6^n % 10) = 6 :=
sorry

end units_digit_6_power_l707_707508


namespace number_of_valid_pairs_l707_707302

open Finset

theorem number_of_valid_pairs:
  let pairs := {p : ℕ × ℕ | nat.gcd p.1 p.2 = 5! ∧ nat.lcm p.1 p.2 = 50!} in
  pairs.card = 2^14 := 
sorry

end number_of_valid_pairs_l707_707302


namespace shop_owner_profit_percentage_l707_707137

-- Definitions for the conditions
def actual_weight_when_buying (initial_weight : ℝ) : ℝ := initial_weight * 1.12
def actual_weight_when_selling (claimed_weight : ℝ) : ℝ := claimed_weight * 0.7
def selling_price (claimed_weight : ℝ) (price_per_100g : ℝ) : ℝ := (claimed_weight / 100) * price_per_100g

-- Theorem statement for the proof problem
theorem shop_owner_profit_percentage : 
  ∀ (initial_weight claimed_weight : ℝ) (cost_price profit_price : ℝ), 
  initial_weight = 100 → 
  claimed_weight = 100 → 
  cost_price = 100 → 
  profit_price = 160 → 
  (actual_weight_when_buying initial_weight = 112) ∧ 
  (actual_weight_when_selling claimed_weight = 70) ∧ 
  (profit_price = selling_price (actual_weight_when_buying initial_weight / actual_weight_when_selling claimed_weight * 100) cost_price) →
  (profit_price - cost_price) / cost_price * 100 = 60 := 
by
  intros
  sorry  -- proof omitted

end shop_owner_profit_percentage_l707_707137


namespace perpendicular_intersect_iff_eq_l707_707240

theorem perpendicular_intersect_iff_eq (ABC : Triangle) 
  (K E H : Point) 
  (hK : K ∈ LineSegment ABC.A ABC.B) 
  (hE : E ∈ LineSegment ABC.B ABC.C) 
  (hH : H ∈ LineSegment ABC.C ABC.A) 
  (hK_perp : IsPerpendicular (line_through K) (ABC.side AB)) 
  (hE_perp : IsPerpendicular (line_through E) (ABC.side BC)) 
  (hH_perp : IsPerpendicular (line_through H) (ABC.side CA)) :
  (∃ O : Point, 
    is_intersection (perpendicular_from K) (perpendicular_from E) (perpendicular_from H) O) ↔
  (AB.length_sq K + BC.length_sq E + CA.length_sq H = 
   K.length_sq AB + E.length_sq BC + H.length_sq CA) :=
sorry

end perpendicular_intersect_iff_eq_l707_707240


namespace graphs_intersect_inverse_function_l707_707089

theorem graphs_intersect_inverse_function (c d : ℤ) (h₁ : ∀ x : ℝ, f x = 4 * x + c)
  (h₂ : ∀ y : ℝ, f (f⁻¹ y) = y)
  (h₃ : f 2 = d)
  (h₄ : ∃ y : ℝ, f (y) = 2) :
  d = 2 :=
by
  sorry

end graphs_intersect_inverse_function_l707_707089


namespace find_a_and_union_A_B_l707_707654

variable (a : ℝ)
def A := {2, 3, a^2 + 4 * a + 2}
def B := {0, 7, 2 - a, a^2 + 4 * a - 2}

theorem find_a_and_union_A_B (h : A ∩ B = {3, 7}) :
  a = 1 ∧ (A ∪ B = {0, 1, 2, 3, 7}) :=
sorry

end find_a_and_union_A_B_l707_707654


namespace unique_solution_sum_l707_707774

theorem unique_solution_sum {x y b: ℕ} 
    (h1 : x^2 * y - x^2 - 3 * y - 14 = 0)
    (h2 : ∀ (x' y' : ℕ), x^2 * y - x^2 - 3 * y - 14 = 0 → (x = x' ∧ y = y')) :
    x + y = b :=
begin
  sorry
end

end unique_solution_sum_l707_707774


namespace isosceles_triangle_l707_707061

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C O : V)

theorem isosceles_triangle (h : (B - C) • (B + C - 2 • O) = 0) : dist A B = dist A C :=
sorry

end isosceles_triangle_l707_707061


namespace total_profit_is_5000_l707_707140

-- Definitions of initial investments
def investment_a : ℝ := 5000
def investment_b : ℝ := 15000
def investment_c : ℝ := 30000

-- Definition of C's share of profit
def profit_c : ℝ := 3000

-- Definition for total profit
def total_profit : ℝ :=
  let ratio_c := 6
  let one_part_value := profit_c / ratio_c
  let ratio_total := 1 + 3 + 6
  ratio_total * one_part_value

-- The statement we want to prove
theorem total_profit_is_5000 : total_profit = 5000 := 
by
  sorry

end total_profit_is_5000_l707_707140


namespace mode_and_median_data_set_l707_707549

theorem mode_and_median_data_set : 
  let data := [2, 2, 4, 3, 6, 5, 2] in
  let mode := 2 in
  let median := 3 in
  (mode_of data = mode) ∧ (median_of data = median) :=
by
  sorry

end mode_and_median_data_set_l707_707549


namespace intersection_of_A_and_B_l707_707641

-- Definitions of sets A and B based on the conditions
def A : Set ℝ := {x | 0 < x}
def B : Set ℝ := {0, 1, 2}

-- Theorem statement to prove A ∩ B = {1, 2}
theorem intersection_of_A_and_B : A ∩ B = {1, 2} := 
  sorry

end intersection_of_A_and_B_l707_707641


namespace macy_miles_left_l707_707739

theorem macy_miles_left (goal : ℕ) (daily_miles : ℕ) (days_run : ℕ) 
  (H1 : goal = 24) 
  (H2 : daily_miles = 3) 
  (H3 : days_run = 6) 
  : goal - daily_miles * days_run = 6 := 
by 
  sorry

end macy_miles_left_l707_707739


namespace sasha_photo_color_l707_707412

theorem sasha_photo_color (cube_colors: ℕ → ℕ → ℕ) (face_photo_color: ℕ → ℕ)
  (photo_visible_faces: ℕ → ℕ) (bw_visible_color: ℕ)
  (h_cubes : ∀ i, 1 ≤ i ∧ i ≤ 4)
  (h_cube_colors : ∃ i, ∀ j, 1 ≤ j ∧ j ≤ 6 → cube_colors i j ∈ {1, 2, 3})
  (h_photo_colors : ∀ k, 1 ≤ k ∧ k ≤ 3 → face_photo_color k ∈ {1, 2, 3})
  (h_faces_per_photo : ∀ k, 1 ≤ k ∧ k ≤ 3 → ∑ i in (range 1 5), face_photo_color (photo_visible_faces k) = 8)
  (h_total_faces : ∑ k in (range 1 4), ∑ i in (range 1 7), cube_colors k i = 24)
  (h_bw_photo : ∑ i in (range 1 5), bw_visible_color = 8)
  (h_bw_same_color : ∀ i j, i ≠ j → bw_visible_color i = bw_visible_color j) :
  bw_visible_color = 2 :=
by sorry

end sasha_photo_color_l707_707412


namespace ab_minus_l707_707568

-- Define the function g is an invertible function
axiom exists_inv {g : ℝ → ℝ} : (∀ x y, g x = g y → x = y) ∧ (∀ y, ∃ x, g x = y)

-- Define the hypothesis for the given problem
variables (g : ℝ → ℝ) (a b : ℝ)
hypothesis h_inv: ∀ x y, g x = g y → x = y
hypothesis h_surj: ∀ y, ∃ x, g x = y
hypothesis h1: g a = b
hypothesis h2: g b = 3

-- Prove that a - b = -3
theorem ab_minus : a - b = -3 :=
by
  sorry

end ab_minus_l707_707568


namespace factorial_division_l707_707150

-- We can add the definition for factorial to be used in the problem if necessary
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem factorial_division :
  (factorial 8) / (factorial (8 - 2)) = 56 :=
by
  sorry

end factorial_division_l707_707150


namespace g_of_4_l707_707772

-- Define the function f
def f (x : ℝ) : ℝ := 4 / (3 - x)

-- Define the inverse function of f
def f_inv (y : ℝ) : ℝ := (3 * y - 4) / y

-- Define the function g
def g (x : ℝ) : ℝ := 1 / (f_inv x) + 5

-- State the theorem
theorem g_of_4 : g 4 = 5.5 := by
  -- This skips the proof
  sorry

end g_of_4_l707_707772


namespace squareD_perimeter_l707_707417

-- Let perimeterC be the perimeter of square C
def perimeterC : ℝ := 40

-- Let sideC be the side length of square C
def sideC := perimeterC / 4

-- Let areaC be the area of square C
def areaC := sideC * sideC

-- Let areaD be the area of square D, which is one-third the area of square C
def areaD := (1 / 3) * areaC

-- Let sideD be the side length of square D
def sideD := Real.sqrt areaD

-- Let perimeterD be the perimeter of square D
def perimeterD := 4 * sideD

-- The theorem to prove
theorem squareD_perimeter (h : perimeterC = 40) (h' : areaD = (1 / 3) * areaC) : perimeterD = (40 * Real.sqrt 3) / 3 := by
  sorry

end squareD_perimeter_l707_707417


namespace perimeter_of_square_D_l707_707426

-- Definition of the perimeter of square C
def perimeter_C := 40
-- Definition of the area of square D in terms of the area of square C
def area_C := ((perimeter_C / 4) ^ 2)
def area_D := area_C / 3
-- Define the side of square D in terms of its area
def side_D := Real.sqrt area_D
-- Prove the perimeter of square D
def perimeter_D := 4 * side_D

-- Statement to prove the perimeter of square D equals the given value
theorem perimeter_of_square_D :
  perimeter_D = 40 * Real.sqrt 3 / 3 :=
by
  sorry

end perimeter_of_square_D_l707_707426


namespace P_shape_points_length_10_l707_707844

def P_shape_points (side_length : ℕ) : ℕ :=
  let points_per_side := side_length + 1
  let total_points := points_per_side * 3
  total_points - 2

theorem P_shape_points_length_10 :
  P_shape_points 10 = 31 := 
by 
  sorry

end P_shape_points_length_10_l707_707844


namespace hairstylist_charge_l707_707538

theorem hairstylist_charge 
  (charge_normal : ℝ)
  (charge_special : ℝ = 6)
  (charge_trendy : ℝ = 8)
  (num_normal_per_day : ℝ = 5)
  (num_special_per_day : ℝ = 3)
  (num_trendy_per_day : ℝ = 2)
  (earnings_per_week : ℝ = 413) :
  charge_normal = 5 :=
by 
  sorry

end hairstylist_charge_l707_707538


namespace prob_simultaneous_sequences_l707_707816

-- Definitions for coin probabilities
def prob_heads_A : ℝ := 0.3
def prob_tails_A : ℝ := 0.7
def prob_heads_B : ℝ := 0.4
def prob_tails_B : ℝ := 0.6

-- Definitions for required sequences
def seq_TTH_A : ℝ := prob_tails_A * prob_tails_A * prob_heads_A
def seq_HTT_B : ℝ := prob_heads_B * prob_tails_B * prob_tails_B

-- Main assertion
theorem prob_simultaneous_sequences :
  seq_TTH_A * seq_HTT_B = 0.021168 :=
by
  sorry

end prob_simultaneous_sequences_l707_707816


namespace parabola_equation_l707_707097

theorem parabola_equation (h1 : Vertex (0,0)) 
                          (h2 : AxisOfSymmetry y_axis) 
                          (h3 : Distance vertex focus = 6) :
  equation parabola = "x^2 = ± 24y" :=
sorry

end parabola_equation_l707_707097


namespace area_is_25_l707_707181

noncomputable def area_of_square (x : ℝ) : ℝ :=
  let side1 := 5 * x - 20
  let side2 := 25 - 4 * x
  if h : side1 = side2 then 
    side1 * side1
  else 
    0

theorem area_is_25 (x : ℝ) (h_eq : 5 * x - 20 = 25 - 4 * x) : area_of_square x = 25 :=
by
  sorry

end area_is_25_l707_707181


namespace total_area_is_correct_l707_707331

-- Define the coordinates of points
def A := (0 : ℝ, 0 : ℝ)
def B := (1 : ℝ, 2 : ℝ)
def C := (2 : ℝ, 1 : ℝ)
def D := (2 : ℝ, 2 : ℝ)

-- Define the formula for the area of a triangle based on coordinates
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

-- Define the areas of triangles ABD and BCD
def area_ABD : ℝ := triangle_area A B D
def area_BCD : ℝ := triangle_area B C D

-- Define the sum of the areas
def total_area : ℝ := area_ABD + area_BCD

-- The theorem to be proved
theorem total_area_is_correct : total_area = 3 / 2 :=
sorry

end total_area_is_correct_l707_707331


namespace students_with_both_pets_l707_707335

theorem students_with_both_pets (total_students dog_owners cat_owners both_pets : ℕ)
  (h1 : total_students = 45)
  (h2 : dog_owners = 27)
  (h3 : cat_owners = 30)
  (h4 : total_students = dog_owners + cat_owners - both_pets) :
  both_pets = 12 :=
by
  rw [h1, h2, h3] at h4
  linarith
  sorry

end students_with_both_pets_l707_707335


namespace monic_quartic_polynomial_l707_707954

theorem monic_quartic_polynomial :
  ∃ (p : Polynomial ℚ), p.leadingCoeff = 1 ∧
  (p.eval (3 + Real.sqrt 5) = 0) ∧
  (p.eval (2 - Real.sqrt 6) = 0) :=
by {
  let p := Polynomial.X^4 - 10*Polynomial.X^3 + 20*Polynomial.X^2 + 4*Polynomial.X - 8,
  use p,
  split,
  { simp [p], }, -- Check that the leading coefficient is 1
  split,
  { -- Check that (3 + sqrt(5)) is a root
    sorry
  },
  { -- Check that (2 - sqrt(6)) is a root
    sorry
  }
}

end monic_quartic_polynomial_l707_707954


namespace number_of_children_went_to_show_l707_707149

theorem number_of_children_went_to_show :
  ∃ C : ℕ, let P := 16 in 
  let A := 32 in 
  let total := 16000 in 
  (400 * A + C * P = total) → 
  C = 200 :=
by
  sorry

end number_of_children_went_to_show_l707_707149


namespace largest_n_for_factored_polynomial_l707_707964

theorem largest_n_for_factored_polynomial :
  ∃ (n : ℕ), (∃ (A B : ℕ), (5 * A * B = 80) ∧ (nat.prime A ∨ nat.prime B) ∧ n = 5 * B + A) ∧ n = 401 :=
by {
  use 401,
  use (1, 80),
  split,
  { split,
    { exact sorry }, -- We need to show 5 * 80 = 80
    { left, exact sorry } -- We need to show 1 is a prime
  },
  { exact sorry } -- We need to show 401 = 5 * 80 + 1
}

end largest_n_for_factored_polynomial_l707_707964


namespace sin_C_eq_cos_B_l707_707348

theorem sin_C_eq_cos_B (A B C : Type*) [triangle A B C] (angle_B : angle A B C = 90) (cos_B : cos angle_B = 3 / 5) :
  sin (90 - angle C) = 3 / 5 := 
sorry

end sin_C_eq_cos_B_l707_707348


namespace inequality_holds_infinitely_many_times_l707_707358

variable {a : ℕ → ℝ}

theorem inequality_holds_infinitely_many_times
    (h_pos : ∀ n, 0 < a n) :
    ∃ᶠ n in at_top, 1 + a n > a (n - 1) * 2^(1 / n) :=
sorry

end inequality_holds_infinitely_many_times_l707_707358


namespace locus_of_points_tangent_arc_l707_707969

noncomputable def is_on_tangent {C : Circle} (P : Point) : Prop := 
  ∃ Q ∈ C, tangent P Q 

noncomputable def auxiliary_circle {A B : Point} : Circle := sorry -- Placeholder for auxiliary circle construction

noncomputable def locus_points {C : Circle} {A B : Point} : Set Point := 
  { X | is_on_tangent X ∧ 
  (outside_A : Point := belongs to auxiliary_circle A B) ∧ 
  (within_boundaries : Point := X ∈ exterior region defined).

theorem locus_of_points_tangent_arc {C : Circle} {A B : Point} :
  let locus := locus_points C A B 
  in locus = { X | tangent to arc A B ∧ within combined auxiliary circles bounds } :=
  sorry  -- Proof to be provided

end locus_of_points_tangent_arc_l707_707969


namespace find_k_l707_707955

theorem find_k (k : ℤ) (h : k ≥ 1) : 
  (∃ n m : ℤ, 9 * n^6 = 2^k + 5 * m^2 + 2) ↔ k = 1 :=
by sorry

end find_k_l707_707955


namespace time_taken_to_cross_platform_l707_707918

noncomputable def length_of_train : ℝ := 100 -- in meters
noncomputable def speed_of_train_km_hr : ℝ := 60 -- in km/hr
noncomputable def length_of_platform : ℝ := 150 -- in meters

noncomputable def speed_of_train_m_s := speed_of_train_km_hr * (1000 / 3600) -- converting km/hr to m/s
noncomputable def total_distance := length_of_train + length_of_platform
noncomputable def time_taken := total_distance / speed_of_train_m_s

theorem time_taken_to_cross_platform : abs (time_taken - 15) < 0.1 :=
by
  sorry

end time_taken_to_cross_platform_l707_707918


namespace max_removable_rooks_l707_707709

-- Definitions based on the conditions provided
def initial_rook_positions : Set (ℕ × ℕ) := {(i, j) | 1 ≤ i ∧ i ≤ 8 ∧ 1 ≤ j ∧ j ≤ 8}

def attacks : (ℕ × ℕ) → (ℕ × ℕ) → Prop
| (i1, j1), (i2, j2) := (i1 = i2 ∨ j1 = j2) ∧ ((i1 ≠ i2) ∨ (j1 ≠ j2))

def attacks_odd (p : (ℕ × ℕ)) (rooks : Set (ℕ × ℕ)) : Prop :=
  (rooks.filter (λ r, attacks p r)).card % 2 = 1

-- Theorem based on the question and correct answer
theorem max_removable_rooks :
  ∃ R ⊆ initial_rook_positions, R.card = 59 ∧ 
  ∀ r ∈ R, attacks_odd r (initial_rook_positions \ R) :=
sorry

end max_removable_rooks_l707_707709


namespace volleyball_team_selection_l707_707058

theorem volleyball_team_selection
  (team : Finset ℕ)
  (christine : ℕ)
  (ethan evan : ℕ)
  (h_team_size : team.card = 16)
  (h_christine_in_team : christine ∈ team)
  (h_twins_in_team : ethan ∈ team ∧ evan ∈ team)
  (starters : Finset ℕ)
  (h_christine_starter : christine ∈ starters)
  (h_starters_size : starters.card = 6)
  (h_at_most_one_twin : ∀ (e : ℕ), e ∈ starters → e ≠ ethan ∨ e ≠ evan) :
  ∃ (n : ℕ), n = 2717 :=
sorry

end volleyball_team_selection_l707_707058


namespace radians_to_degrees_l707_707577

theorem radians_to_degrees (π_is_180 : ℝ) (h : π_is_180 = 180) : (2 / 3 * π_is_180 = 120) :=
by {
  -- Since π_is_180 = 180, we substitute and simplify
  simp [h], 
  norm_num
}

end radians_to_degrees_l707_707577


namespace coin_toss_correct_answer_l707_707922

/-- Define event A as a coin toss resulting in heads up. -/
def eventA : String := "heads up"

/-- Define the problem conditions and the correct answer. -/
theorem coin_toss_correct_answer :
  let number_of_tosses := Nat.succ (Nat.succ 6) in
  (forall n : ℕ, (number_of_tosses = 8) → 
   (∀ outcomes : list String, outcomes.length = (Nat.succ 1) ^ 2 →
    (outcomes.filter (λ x, x = "HT" ∨ x = "TH")).length = 2 →
    (2 / 4 = 1 / 2))) ∧
  (statementB := "Tossing the coin 8 times, the number of times event A occurs must be 4") →
  (∀ n : ℕ, n = 8 →
    (∀ heads_count : ℕ, heads_count ≤ 8 ∧ heads_count ≥ 0)) ∧
  (statementC := "Repeatedly tossing the coin, the frequency of event A occurring is equal to the probability of event A occurring") →
  (forall n : ℕ, (statementC = false)) ∧
  (statementD := "When the number of tosses is large enough, the frequency of event A occurring approaches 0.5") →
  (forall n : ℕ, n → 1000 → (0.5 / 1 = 0.5)) →
  statementD = true :=
sorry

end coin_toss_correct_answer_l707_707922


namespace polynomial_degree_bound_l707_707728

-- Define P as a polynomial with integer coefficients
variable {P : Polynomial ℤ}

-- Assume P is not a constant polynomial
instance : Nontrivial (Polynomial ℤ) := {
  exists_pair_ne := ⟨P, 0, sorry⟩,  -- The proof of non-triviality (not constant) is omitted here
}

-- Definition of n(P) as the number of distinct integer values k such that [P(k)]² = 1
def n_P (P : Polynomial ℤ) : ℕ :=
count (fun k : ℤ => P.eval k ∈ {1, -1}) finset.Icc (min_bound P) (max_bound P)

-- The main theorem to be proved: n(P) - deg(P) ≤ 2
theorem polynomial_degree_bound :
  n_P P - P.degree.to_nat ≤ 2 :=
sorry

end polynomial_degree_bound_l707_707728


namespace simplify_expression_l707_707074

noncomputable def complex_num1 : ℂ := (-1 + complex.I * Real.sqrt 3) / 2
noncomputable def complex_num2 : ℂ := (-1 - complex.I * Real.sqrt 3) / 2

theorem simplify_expression :
  complex_num1 ^ 6 + complex_num2 ^ 6 = 2 := by
  sorry

end simplify_expression_l707_707074


namespace spider_eyes_solution_l707_707051

def spider_eyes_problem: Prop :=
  ∃ (x : ℕ), (3 * x) + (50 * 2) = 124 ∧ x = 8

theorem spider_eyes_solution : spider_eyes_problem :=
  sorry

end spider_eyes_solution_l707_707051


namespace macy_running_goal_l707_707745

/-- Macy's weekly running goal is 24 miles. She runs 3 miles per day. Calculate the miles 
    she has left to run after 6 days to meet her goal. --/
theorem macy_running_goal (miles_per_week goal_per_week : ℕ) (miles_per_day: ℕ) (days_run: ℕ) 
  (h1 : miles_per_week = 24) (h2 : miles_per_day = 3) (h3 : days_run = 6) : 
  miles_per_week - miles_per_day * days_run = 6 := 
  by 
    rw [h1, h2, h3]
    exact Nat.sub_eq_of_eq_add (by norm_num)

end macy_running_goal_l707_707745


namespace sum_possible_a_l707_707127

noncomputable def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem sum_possible_a :
  ∑ a in {a | ∃ b, a^2 = (b + 2)^2 ∧ is_right_triangle a b (b + 2) ∧ a < 20 ∧ b < 20}, a = 187 :=
by
  sorry

end sum_possible_a_l707_707127


namespace students_in_high_school_l707_707878

-- Definitions from conditions
def H (L: ℝ) : ℝ := 4 * L
def middleSchoolStudents : ℝ := 300
def combinedStudents (H: ℝ) (L: ℝ) : ℝ := H + L
def combinedIsSevenTimesMiddle (H: ℝ) (L: ℝ) : Prop := combinedStudents H L = 7 * middleSchoolStudents

-- The main goal to prove
theorem students_in_high_school (L H: ℝ) (h1: H = 4 * L) (h2: combinedIsSevenTimesMiddle H L) : H = 1680 := by
  sorry

end students_in_high_school_l707_707878


namespace profit_percentage_l707_707902

/-- Assume a store owner purchases 200 pens at the marked price of 180 pens. Each pen has a marked price of $1.
If the store owner sells the pens offering a discount of 2%, the profit percentage is approximately 8.89%. -/
theorem profit_percentage (marked_price per_pen : ℝ) (cost_price discount_percentage selling_price total_revenue profit profit_percentage : ℝ) :
  marked_price = 1 →
  per_pen = 200 →
  cost_price = 180 →
  discount_percentage = 2 →
  selling_price = (marked_price - marked_price * discount_percentage / 100) →
  total_revenue = per_pen * selling_price →
  profit = total_revenue - cost_price →
  profit_percentage = (profit / cost_price) * 100 →
  profit_percentage ≈ 8.89 :=
by
  intros
  sorry

end profit_percentage_l707_707902


namespace smallest_positive_sum_eq_112_l707_707214

theorem smallest_positive_sum_eq_112 (a : Fin 100 → ℤ) (h : ∀ i, a i = 1 ∨ a i = -1) :
  ∃ S > 0, S = ∑ i in Finset.filter (λ i, ∃ j, i < j) (Finset.univ : Finset (Fin 100 × Fin 100)), a i.1 * a i.2 ∧ S = 112 :=
begin
  -- proof
  sorry
end

end smallest_positive_sum_eq_112_l707_707214


namespace least_positive_integer_solution_l707_707965

theorem least_positive_integer_solution (x : ℕ) :
  (∃ x, ((x + 5123) % 12 = 2900 % 12) ∧ x > 0) → (x % 12 = 9) :=
by-sorry

end least_positive_integer_solution_l707_707965


namespace distance_between_numbered_streets_l707_707185

theorem distance_between_numbered_streets :
  let total_length := 3.2 * 1000  -- total length of Apple Street in meters
  let highest_numbered_street := 15
  (15 - 1) > 0 ∧ total_length > 0 → 
  total_length / (15 - 1) ≈ 228.57 := 
by sorry

end distance_between_numbered_streets_l707_707185


namespace swim_time_l707_707165

-- Definitions based on conditions:
def speed_in_still_water : ℝ := 6.5 -- speed of the man in still water (km/h)
def distance_downstream : ℝ := 16 -- distance swam downstream (km)
def distance_upstream : ℝ := 10 -- distance swam upstream (km)
def time_downstream := 2 -- time taken to swim downstream (hours)
def time_upstream := 2 -- time taken to swim upstream (hours)

-- Defining the speeds taking the current into account:
def speed_downstream (c : ℝ) : ℝ := speed_in_still_water + c
def speed_upstream (c : ℝ) : ℝ := speed_in_still_water - c

-- Assumption that the time took for both downstream and upstream are equal
def time_eq (c : ℝ) : Prop :=
  distance_downstream / (speed_downstream c) = distance_upstream / (speed_upstream c)

-- The proof we need to establish:
theorem swim_time (c : ℝ) (h : time_eq c) : time_downstream = time_upstream := by
  sorry

end swim_time_l707_707165


namespace archimedes_schools_l707_707698

-- Define the conditions
def students_per_school := 4
def participants_archimedes := 169 -- 69 from Euclid's contest + 100 more participants
def andrea_higher_than_two_teammates_and_not_highest := 
  ∃ (a b c d : Nat) (team_members : List Nat) (i : Fin (4 * num_schools)), 
  team_members.nth i.val = some a ∧ 
  a > b ∧ a > c ∧ a < d ∧ 
  team_members = [a, b, c, d]
def beth_place := 45
def carla_place := 80

-- Define the Lean theorem to prove the correct number of schools
theorem archimedes_schools : 
  ∃ num_schools : Nat, 
  4 * num_schools = participants_archimedes ∧ num_schools = 43 :=
by 
  use 43
  exact ⟨by simp [participants_archimedes], by simp⟩

end archimedes_schools_l707_707698


namespace value_at_minus_two_l707_707286

def f (x : ℝ) (a b c : ℝ) := a * x^5 + b * x^3 + c * x + 1

theorem value_at_minus_two (a b c : ℝ) (h : f 2 a b c = -1) : f (-2) a b c = 3 := by
  sorry

end value_at_minus_two_l707_707286


namespace total_distance_travelled_l707_707060

def dist_spain_russia : ℕ := 7019
def dist_spain_germany : ℕ := 1615

def dist_germany_russia : ℕ := dist_spain_russia - dist_spain_germany -- Remaining distance from Germany to Russia

theorem total_distance_travelled : dist_germany_russia + (dist_germany_russia + dist_spain_germany) = 12423 :=
by
  have dist_return_trip : ℕ := dist_germany_russia + dist_spain_germany
  have dist_total := dist_germany_russia + dist_return_trip
  show dist_total = 12423
  sorry

end total_distance_travelled_l707_707060


namespace male_students_not_like_math_l707_707459

theorem male_students_not_like_math
  (total_students : ℕ) 
  (proportion_females : ℚ) 
  (proportion_males_like_math : ℚ)
  (h1 : total_students = 1500) 
  (h2 : proportion_females = 0.4) 
  (h3 : proportion_males_like_math = 0.65) : 
  ∃ (male_students_not_like_math : ℕ), male_students_not_like_math = 315 :=
by
  let total_female_students := total_students * proportion_females
  let total_male_students := total_students - total_female_students
  let male_students_like_math := total_male_students * proportion_males_like_math
  let male_students_not_like_math := total_male_students - male_students_like_math
  have h4 : total_female_students = 600 := by sorry
  have h5 : total_male_students = 900 := by sorry
  have h6 : male_students_like_math = 585 := by sorry
  have h7 : male_students_not_like_math = 315 := by sorry
  use 315
  exact h7

end male_students_not_like_math_l707_707459


namespace fourth_root_of_2560000_l707_707196

theorem fourth_root_of_2560000 : real.sqrt (real.sqrt (real.sqrt (real.sqrt 2560000))) = 40 := 
by sorry

end fourth_root_of_2560000_l707_707196


namespace base4_addition_correct_l707_707442

noncomputable def decimal_to_base4 (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let rec convert_aux (m quotient acc : ℕ) : ℕ :=
    if quotient = 0 then acc
    else convert_aux (m * 10) (quotient / 4) ((quotient % 4) * m + acc)
  in convert_aux 1 n 0

theorem base4_addition_correct (a b : ℕ) : decimal_to_base4 a = 2200 ∧ decimal_to_base4 b = 231 →
  decimal_to_base4 (a + b) = 2431 :=
begin
  sorry
end

# Check it using: 
# base4_addition_correct 160 45 (by split; apply rfl)

end base4_addition_correct_l707_707442


namespace problem_solution_l707_707622

theorem problem_solution 
    (a : ℕ → ℝ) (S : ℕ → ℝ) (P Q : ℕ → ℝ)
    (h1 : ∀ n, a n = 2 * n)
    (h2 : ∀ n, S n = (n * (n + 1)) / 2)
    (h3 : ∀ n, P n = (Finset.range n).sum (λ i, 1 / a (2 ^ i)))
    (h4 : ∀ n, Q n = (Finset.range n).sum (λ k, 1 / S (k + 1)))
    : ∀ n, 1 - 1 / (2 ^ n) ≥ 1 - 1 / (n + 1) := by 
    sorry

end problem_solution_l707_707622


namespace prism_vertices_on_sphere_surface_area_eq_12pi_l707_707606

noncomputable def surface_area_of_circumscribed_sphere : ℝ :=
  let edge_length := 2
  let radius := (3 * edge_length ^ 2).sqrt / 2
  4 * Real.pi * radius ^ 2

theorem prism_vertices_on_sphere_surface_area_eq_12pi :
  surface_area_of_circumscribed_sphere = 12 * Real.pi := by
  sorry

end prism_vertices_on_sphere_surface_area_eq_12pi_l707_707606


namespace books_left_correct_l707_707045

variable (initial_books : ℝ) (sold_books : ℝ)

def number_of_books_left (initial_books sold_books : ℝ) : ℝ :=
  initial_books - sold_books

theorem books_left_correct :
  number_of_books_left 51.5 45.75 = 5.75 :=
by
  sorry

end books_left_correct_l707_707045


namespace mass_percentage_O_in_ascorbic_acid_l707_707599

theorem mass_percentage_O_in_ascorbic_acid :
  let C_molar_mass := 12.01
  let H_molar_mass := 1.008
  let O_molar_mass := 16.00
  let n_C := 6
  let n_H := 8
  let n_O := 6
  let mass_C := n_C * C_molar_mass
  let mass_H := n_H * H_molar_mass
  let mass_O := n_O * O_molar_mass
  let total_mass := mass_C + mass_H + mass_O
  mass_O / total_mass * 100 ≈ 54.5 :=
by
  sorry

end mass_percentage_O_in_ascorbic_acid_l707_707599


namespace ellipse_eq_l707_707123

-- Given conditions as definitions
def hyperbola_eq (x y : ℝ) : Prop := 3 * x^2 - y^2 = 3
def reciprocal_eccentricity (e : ℝ) : Prop := e = 1 / 2

-- Proof problem statement
theorem ellipse_eq (x y : ℝ) (e : ℝ) :
  hyperbola_eq x y → reciprocal_eccentricity e →
  (∃ a b : ℝ, a = 4 ∧ b^2 = 12 ∧ e = 1 / 2 ∧ (x = 2 ∨ y = 0) ∧
              (x^2) / (a^2) + (y^2) / (b^2) = 1) :=
begin
  sorry
end

end ellipse_eq_l707_707123


namespace area_union_correct_l707_707172

noncomputable def side_length : ℝ := 12
noncomputable def radius : ℝ := 12

def area_square : ℝ := side_length ^ 2
def area_circle : ℝ := π * radius ^ 2
def overlapping_area : ℝ := (1 / 4) * area_circle

def area_union : ℝ := area_square + area_circle - overlapping_area

theorem area_union_correct : area_union = 144 + 108 * π := by
  unfold area_square area_circle overlapping_area area_union
  norm_num
  simp
  sorry

end area_union_correct_l707_707172


namespace total_candy_count_l707_707463

def numberOfRedCandies : ℕ := 145
def numberOfBlueCandies : ℕ := 3264
def totalNumberOfCandies : ℕ := numberOfRedCandies + numberOfBlueCandies

theorem total_candy_count :
  totalNumberOfCandies = 3409 :=
by
  unfold totalNumberOfCandies
  unfold numberOfRedCandies
  unfold numberOfBlueCandies
  sorry

end total_candy_count_l707_707463


namespace remaining_surface_area_unaltered_l707_707586

theorem remaining_surface_area_unaltered : 
  let original_cube_dim := (4 : ℝ)
  let corner_cube_dim := (2 : ℝ)
  let num_corners := (8 : ℕ)
  let original_surface_area := 6 * (original_cube_dim ^ 2)
  let corner_faces_area_effect := 3 * (corner_cube_dim ^ 2)
  let net_surface_area_change := 0
  
  6 * (4^2) + num_corners * net_surface_area_change = 96 :=
by simp [original_surface_area, net_surface_area_change, original_cube_dim, corner_cube_dim, num_corners]; sorry

end remaining_surface_area_unaltered_l707_707586


namespace line_integral_along_path_L_l707_707521

noncomputable def vector_field (ρ φ z : ℝ) : ℝ³ := (4 * ρ * Real.sin φ, z * Real.exp ρ, ρ + φ)

def path_L (ρ : ℝ) : ℝ³ := (ρ, Real.pi / 4, 0)

theorem line_integral_along_path_L :
  ∫ (ρ : ℝ) in 0..1, (4 * ρ * Real.sin (Real.pi / 4)) = Real.sqrt 2 :=
by
  -- Proof goes here
  sorry

end line_integral_along_path_L_l707_707521


namespace guards_convex_hull_protection_l707_707010

theorem guards_convex_hull_protection :
  ∃ (guards : Finset Point) (n : ℕ), n ≥ 6 ∧
  ∀ g ∈ guards, 
  (∃ (sight_range : ℝ) (h : sight_range = 100), 
    let vision_area := convex_hull (visibility_segment g sight_range) in
    ∀ (point : Point), point ∉ vision_area → 
    ¬(can_approach_unnoticed point g h)) :=
sorry

end guards_convex_hull_protection_l707_707010


namespace correct_statement_A_l707_707908

namespace ProgramFlowchart

-- Conditions on what a program flowchart includes
structure ProgramFlowchart where
  operations : Prop
  flow_lines : Prop
  textual_explanations : Prop

-- Conditions represent statements A, B, C, D
def statements : List (ProgramFlowchart → Prop) := [
  λ pf => pf.operations ∧ pf.flow_lines ∧ pf.textual_explanations,            -- statement A
  λ pf => ∃ unique (i o : Prop), i = pf.operations ∧ o = pf.textual_explanations, -- statement B
  λ pf => ¬ (pf.operations ∧ pf.flow_lines ∧ pf.textual_explanations),       -- statement C
  λ pf => true                                                                -- statement D (must include decision box)
]

-- The formal statement
theorem correct_statement_A (pf : ProgramFlowchart) : statements.head pf :=
by { sorry }

end ProgramFlowchart

end correct_statement_A_l707_707908


namespace remainder_of_polynomial_division_l707_707600

theorem remainder_of_polynomial_division (x : ℝ) :
  ∃ Q : ℝ → ℝ, ∃ a b : ℝ, 
    (x^6 + x^4 - 5*x^2 + 9) = ((x - 1) * (x - 2)) * (Q x) + a * x + b 
    ∧ a = 19.5 ∧ b = 5 :=
begin
  sorry
end

end remainder_of_polynomial_division_l707_707600


namespace value_of_a_l707_707612

theorem value_of_a (a : ℝ) (h : (1 + a * complex.i) * complex.i = 3 + complex.i) : a = -3 :=
by {
  sorry
}

end value_of_a_l707_707612


namespace quad_midpoints_rectangle_is_rhombus_l707_707841

/-- A quadrilateral obtained by connecting the midpoints of the sides of a rectangle is a rhombus. -/
theorem quad_midpoints_rectangle_is_rhombus
  (A B C D : Type)
  [rectangle A B C D]
  (M N O P : Type)
  [midpoint A B M] [midpoint B C N] [midpoint C D O] [midpoint D A P] :
  rhombus M N O P :=
sorry

end quad_midpoints_rectangle_is_rhombus_l707_707841


namespace volume_of_rotated_square_l707_707233

-- Define the problem conditions
def side_length (s : ℝ) := s = 20

-- Define the volume formula for a cylinder
def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- Define the radius derived from the diameter which is the side length
def cylinder_radius (d: ℝ) : ℝ := d / 2

-- The volume of the cylinder formed by rotating the square
theorem volume_of_rotated_square (s : ℝ) (V : ℝ) (h : ℝ) 
    (h_side_length : side_length s)
    (h_height : h = s) 
    (h_radius : cylinder_radius s = 10) :
    V = 2000 * π :=
by
    sorry

end volume_of_rotated_square_l707_707233


namespace sample_capacity_l707_707171

theorem sample_capacity (frequency : ℕ) (frequency_rate : ℚ) (n : ℕ)
  (h1 : frequency = 30)
  (h2 : frequency_rate = 25 / 100) :
  n = 120 :=
by
  sorry

end sample_capacity_l707_707171


namespace lower_limit_of_f_l707_707231

theorem lower_limit_of_f (x : ℝ) :
  (x - 5 ≤ 8) → (x ≤ 13) :=
by
suffices h : x - 5 ≤ 8 → x ≤ 13 from λ h, (h : x - 5 ≤ 8)
sorry

end lower_limit_of_f_l707_707231


namespace iso_right_triangle_area_l707_707560

theorem iso_right_triangle_area (x y : ℝ) (s : ℝ) :
  (x^2 + 4*y^2 = 4) ∧ (x = 2) ∧ (y = 0) ∧ 
  (x = 2 - s) ∧ (y = s/Real.sqrt 2) ∧ (2 - s) ∧ (-s/Real.sqrt 2) ∧ 
  (3 * s^2 - 4 * s = 0) ∧ (s ≠ 0) → 
  ∃ s, s = 4/3 ∧ 
  (1/2 * s * s = 8/9) :=
begin 
  sorry 
end

end iso_right_triangle_area_l707_707560


namespace find_c_l707_707605

def conditions (c d : ℝ) : Prop :=
  -- The polynomial 6x^3 + 7cx^2 + 3dx + 2c = 0 has three distinct positive roots
  ∃ u v w : ℝ, 0 < u ∧ 0 < v ∧ 0 < w ∧ u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
  (6 * u^3 + 7 * c * u^2 + 3 * d * u + 2 * c = 0) ∧
  (6 * v^3 + 7 * c * v^2 + 3 * d * v + 2 * c = 0) ∧
  (6 * w^3 + 7 * c * w^2 + 3 * d * w + 2 * c = 0) ∧
  -- Sum of the base-2 logarithms of the roots is 6
  Real.log (u * v * w) / Real.log 2 = 6

theorem find_c (c d : ℝ) (h : conditions c d) : c = -192 :=
sorry

end find_c_l707_707605


namespace smallest_prime_factor_in_set_C_l707_707766

-- Define the set C
def C : Finset ℕ := {66, 68, 71, 73, 75}

-- State the theorem to prove
theorem smallest_prime_factor_in_set_C : 
  ∃ x ∈ C, ∀ y ∈ C, x ≤ y ∧ Nat.prime x ∧ Nat.min_fac x = 2 := by
  sorry

end smallest_prime_factor_in_set_C_l707_707766


namespace chess_tournament_games_l707_707519

theorem chess_tournament_games (n : ℕ) (h : n = 17) (k : n - 1 = 16) :
  (n * (n - 1)) / 2 = 136 := by
  sorry

end chess_tournament_games_l707_707519


namespace distance_between_lines_l707_707780

def line1 (x y : ℝ) : Prop := x - y - 1 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

theorem distance_between_lines : 
  ∀ (x y : ℝ), line1 x y → line2 x y → (|1 - (-1)| / Real.sqrt (1^2 + (-1)^2)) = Real.sqrt 2 := 
by 
  sorry

end distance_between_lines_l707_707780


namespace correct_derivative_B_incorrect_derivatives_l707_707838

noncomputable def y1 (x : ℝ) := sin (x^2)
noncomputable def y1' (x : ℝ) := x * cos (x^2)

noncomputable def y2 (x : ℝ) := sqrt (1 + x^2)
noncomputable def y2' (x : ℝ) := x / sqrt (1 + x^2)

noncomputable def y3 (x : ℝ) := (x^2) / (x^2 + x)
noncomputable def y3' (x : ℝ) := 2 * x / (2 * x + 1)

noncomputable def y4 (x : ℝ) := cos (3 * x - (π / 6))
noncomputable def y4' (x : ℝ) := sin (3 * x - (π / 6))

theorem correct_derivative_B : 
  (∀ x, deriv (λ x : ℝ, sqrt (1 + x^2)) x = (λ x, x / sqrt (1 + x^2)) x) :=
begin
  sorry
end

theorem incorrect_derivatives :
  (∀ x, deriv (λ x : ℝ, sin (x^2)) x ≠ (λ x, x * cos (x^2)) x) ∧ 
  (∀ x, deriv (λ x : ℝ, (x^2) / (x^2 + x)) x ≠ (λ x, 2 * x / (2 * x + 1)) x) ∧
  (∀ x, deriv (λ x : ℝ, cos (3 * x - (π / 6))) x ≠ (λ x, sin (3 * x - (π / 6))) x) :=
begin
  sorry,
end

end correct_derivative_B_incorrect_derivatives_l707_707838


namespace max_problems_to_miss_l707_707565

theorem max_problems_to_miss (total_problems : ℕ) (required_percent : ℕ) (problems_can_miss : ℕ) :
  total_problems = 50 →
  required_percent = 85 →
  problems_can_miss = 7 →
  problems_can_miss ≤ total_problems * (100 - required_percent) / 100 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end max_problems_to_miss_l707_707565


namespace prob_point_closer_to_center_than_boundary_l707_707893

-- Define a function that calculates the area of a circle
def area_of_circle (r : ℝ) : ℝ := real.pi * r^2

-- Define the probability function
def prob_closer_to_center (r_outer r_inner : ℝ) : ℝ :=
  area_of_circle r_inner / area_of_circle r_outer

-- Define the main theorem statement with a proof placeholder
theorem prob_point_closer_to_center_than_boundary :
  prob_closer_to_center 4 1 = 1 / 16 :=
by
  unfold prob_closer_to_center
  unfold area_of_circle
  sorry  -- Add the proof here

end prob_point_closer_to_center_than_boundary_l707_707893


namespace num_squarish_numbers_l707_707550

-- Define a perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define a seven-digit squarish number
def squarish (M : ℕ) : Prop :=
  let fst := M / 10000
  let snd := (M % 10000) / 100
  let lst := M % 100
  (M > 999999 && M < 10000000) &&
  (perfect_square M) &&
  (perfect_square fst) &&
  (perfect_square snd) &&
  (perfect_square lst) &&
  (snd / 10 != 0 || snd % 10 != 0)

-- Define the main theorem
theorem num_squarish_numbers : 
  (∃! M : ℕ, squarish M) → (∃! y : ℕ, squarish (y * y) && (y >= 3163 && y <= 4472)) :=
sorry

end num_squarish_numbers_l707_707550


namespace proof_problem_l707_707062

-- Let P, Q, R be points on a circle of radius s
-- Given: PQ = PR, PQ > s, and minor arc QR is 2s
-- Prove: PQ / QR = sin(1)

noncomputable def point_on_circle (s : ℝ) : ℝ → ℝ × ℝ := sorry
def radius {s : ℝ} (P Q : ℝ × ℝ ) : Prop := dist P Q = s

theorem proof_problem (s : ℝ) (P Q R : ℝ × ℝ)
  (hPQ : dist P Q = dist P R)
  (hPQ_gt_s : dist P Q > s)
  (hQR_arc_len : 1 = s) :
  dist P Q / (2 * s) = Real.sin 1 := 
sorry

end proof_problem_l707_707062


namespace n_cubed_plus_20n_div_48_l707_707399

theorem n_cubed_plus_20n_div_48 (n : ℕ) (h_even : n % 2 = 0) : (n^3 + 20 * n) % 48 = 0 :=
sorry

end n_cubed_plus_20n_div_48_l707_707399


namespace correct_option_is_C_l707_707512

theorem correct_option_is_C 
  (A : Prop)
  (B : Prop)
  (C : Prop)
  (D : Prop)
  (hA : ¬ A)
  (hB : ¬ B)
  (hD : ¬ D)
  (hC : C) :
  C := by
  exact hC

end correct_option_is_C_l707_707512


namespace angle_ADC_eq_90_l707_707736

-- Define the necessary entities and conditions
variables {A B C D K M N : Type*} 
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace K] [MetricSpace M] [MetricSpace N]

-- Assume necessary points and lines, and geometric properties
noncomputable def inscribed_quadrilateral (A B C D : Type*) : Prop :=
  ∃ (O : Type*) [Circumference O A B C D], True

noncomputable def intersection_at_K (A B D C K : Type*) : Prop :=
  ∃ (L AB DC : Type*) [LineThrough AB A B] [LineThrough DC D C] [Intersection K AB DC], True

noncomputable def concyclic_points (B D M N : Type*) : Prop :=
  ∃ (O : Type*) [Circumference O B D M N], True

-- Given all conditions, we need to prove the angle condition
theorem angle_ADC_eq_90 (A B C D K M N : Type*)
  [inscribed_quadrilateral A B C D]
  [intersection_at_K A B D C K]
  [concyclic_points B D M N] :
  ∃ (angle_ADC : ℝ), angle_ADC = 90 :=
sorry

end angle_ADC_eq_90_l707_707736


namespace Sam_total_distance_l707_707384

theorem Sam_total_distance :
  let v_m := 150 / 3 in   -- Marguerite's average speed
  let t_first := 2 in     -- Time Sam drives at Marguerite's speed
  let t_next := 2 in      -- Time Sam drives at increased speed
  let v_s := v_m * 1.2 in -- Sam's increased speed
  let d_first := v_m * t_first in -- Distance Sam travels at Marguerite's speed
  let d_next := v_s * t_next in   -- Distance Sam travels at increased speed
  let total_distance := d_first + d_next in
  total_distance = 220 :=
by
  let v_m := 50
  let t_first := 2
  let t_next := 2
  let v_s := v_m * 1.2
  let d_first := v_m * t_first
  let d_next := v_s * t_next
  let total_distance := d_first + d_next
  show total_distance = 220 from sorry

end Sam_total_distance_l707_707384


namespace sec_neg_450_undefined_l707_707223

theorem sec_neg_450_undefined : ¬ ∃ x, x = 1 / Real.cos (-450 * Real.pi / 180) :=
by
  -- Proof skipped using 'sorry'
  sorry

end sec_neg_450_undefined_l707_707223


namespace largest_integer_log_sum_l707_707828

theorem largest_integer_log_sum : 
  ∃ (n : ℤ), n = 10 ∧ n < (∑ i in Finset.range 2009, Real.log2 ((i + 2) / (i + 1))) ∧
    (∑ i in Finset.range 2009, Real.log2 ((i + 2) / (i + 1))) < n + 1 :=
by
  sorry

end largest_integer_log_sum_l707_707828


namespace projection_of_vector_difference_l707_707660

variables {V : Type*} [inner_product_space ℝ V] (a b : V)
variable [normed_space ℝ V]
variable (h1 : ∥b∥ = 1)
variable (h2 : inner_product_space.inner a b = 0)

theorem projection_of_vector_difference :
  (projection b (a - 2 • b)) = -2 • b :=
by sorry

end projection_of_vector_difference_l707_707660


namespace minimum_red_chips_l707_707531

theorem minimum_red_chips (w b r : ℕ) (h1 : b ≥ (1 / 3) * w) (h2 : b ≤ (1 / 4) * r) (h3 : w + b ≥ 70) : r ≥ 72 := by
  sorry

end minimum_red_chips_l707_707531


namespace volume_ratio_l707_707456

-- Define the variables representing volumes of substances A, B, and C
variables (V_A V_B V_C : ℝ)

-- Define the conditions as hypotheses
def condition1 := 2 * V_A = V_B + V_C
def condition2 := 5 * V_B = V_A + V_C

-- Proving the ratio of V_C to (V_A + V_B) is 1
theorem volume_ratio (h1 : condition1) (h2 : condition2) : 
  V_C / (V_A + V_B) = 1 :=
sorry

end volume_ratio_l707_707456


namespace total_unique_working_games_l707_707018

-- Define the given conditions
def initial_games_from_friend := 25
def non_working_games_from_friend := 12

def games_from_garage_sale := 15
def non_working_games_from_garage_sale := 8
def duplicate_games := 3

-- Calculate the number of working games from each source
def working_games_from_friend := initial_games_from_friend - non_working_games_from_friend
def total_garage_sale_games := games_from_garage_sale - non_working_games_from_garage_sale
def unique_working_games_from_garage_sale := total_garage_sale_games - duplicate_games

-- Theorem statement
theorem total_unique_working_games : 
  working_games_from_friend + unique_working_games_from_garage_sale = 17 := by
  sorry

end total_unique_working_games_l707_707018


namespace find_300th_term_excl_squares_l707_707490

def is_perfect_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

def nth_term_excl_squares (n : ℕ) : ℕ :=
  let excluded := (List.range (n + n / 10)).filter (λ x, ¬ is_perfect_square x)
  excluded.nth n

theorem find_300th_term_excl_squares :
  nth_term_excl_squares 299 = 317 :=
by
  sorry

end find_300th_term_excl_squares_l707_707490


namespace sum_S_2016_l707_707643

def sequence (n : ℕ) : ℕ → ℝ
| 0     => 1
| (n+1) => 3 + sequence n

def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
∃ (d : ℝ), ∀ n : ℕ, a_n n = a_n 0 + n * d

def S (n : ℕ) : ℝ :=
(range n).sum (λ k, (-1)^k * sequence k)

theorem sum_S_2016 :
  let a_n := sequence in
  arithmetic_sequence a_n ->
  a_n 0 = 1 ->
  a_n 4 = 13 ->
  S 2016 = 3024 :=
begin
  sorry
end

end sum_S_2016_l707_707643


namespace probability_rel_prime_to_6_probability_at_least_one_of_two_rel_prime_to_6_l707_707507

noncomputable def prob_rel_prime_to_6 : ℚ :=
  let possible_residues := [0, 1, 2, 3, 4, 5]
  let rel_prime_residues := possible_residues.filter (λ n, Int.gcd n 6 = 1)
  (rel_prime_residues.length : ℚ) / (possible_residues.length : ℚ)

theorem probability_rel_prime_to_6 :
  prob_rel_prime_to_6 = 1 / 3 :=
by
  sorry

noncomputable def prob_at_least_one_rel_prime_to_6 : ℚ :=
  1 - (2 / 3) ^ 2

theorem probability_at_least_one_of_two_rel_prime_to_6 :
  prob_at_least_one_rel_prime_to_6 = 5 / 9 :=
by
  sorry

end probability_rel_prime_to_6_probability_at_least_one_of_two_rel_prime_to_6_l707_707507


namespace find_q_sum_of_bn_l707_707259

-- Defining the sequences and conditions
def a (n : ℕ) (q : ℝ) : ℝ := q^(n-1)

def b (n : ℕ) (q : ℝ) : ℝ := a n q + n

-- Given that 2a_1, (1/2)a_3, a_2 form an arithmetic sequence
def condition_arithmetic_sequence (q : ℝ) : Prop :=
  2 * a 1 q + a 2 q = (1 / 2) * a 3 q + (1 / 2) * a 3 q

-- To be proved: Given conditions, prove q = 2
theorem find_q : ∃ q > 0, a 1 q = 1 ∧ a 2 q = q ∧ a 3 q = q^2 ∧ condition_arithmetic_sequence q ∧ q = 2 :=
by {
  sorry
}

-- Given b_n = a_n + n, prove T_n = (n(n+1))/2 + 2^n - 1
theorem sum_of_bn (n : ℕ) : 
  ∃ T_n : ℕ → ℝ, T_n n = (n * (n + 1)) / 2 + (2^n) - 1 :=
by {
  sorry
}

end find_q_sum_of_bn_l707_707259


namespace determine_k_l707_707540

-- Definitions of the vectors a and b.
variables (a b : ℝ)

-- Noncomputable definition of the scalar k.
noncomputable def k_value : ℝ :=
  (2 : ℚ) / 7

-- Definition of line through vectors a and b as a parametric equation.
def line_through (a b : ℝ) (t : ℝ) : ℝ :=
  a + t * (b - a)

-- Hypothesis: The vector k * a + (5/7) * b is on the line passing through a and b.
def vector_on_line (a b : ℝ) (k : ℝ) : Prop :=
  ∃ t : ℝ, k * a + (5/7) * b = line_through a b t

-- Proof that k must be 2/7 for the vector to be on the line.
theorem determine_k (a b : ℝ) : vector_on_line a b k_value :=
by sorry

end determine_k_l707_707540


namespace commission_percentage_l707_707983

def commission_rate (amount: ℕ) : ℚ :=
  if amount <= 500 then
    0.20 * amount
  else
    0.20 * 500 + 0.50 * (amount - 500)

theorem commission_percentage (total_sale : ℕ) (h : total_sale = 800) :
  (commission_rate total_sale) / total_sale * 100 = 31.25 :=
by
  sorry

end commission_percentage_l707_707983


namespace hyperbola_eccentricity_range_l707_707202

-- Define the eccentricity of the hyperbola with given conditions
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (cond1 : 2 < b / a)
  (cond2 : b / a < 3) : 
  sqrt(5) < sqrt(1 + (b / a)^2) ∧ sqrt(1 + (b / a)^2) < sqrt(10) :=
  by sorry

end hyperbola_eccentricity_range_l707_707202


namespace problem_a_problem_b_l707_707039

noncomputable def triangle_geom (A B C H D E F P Q : Point) (circumcircle : Set Point) : Prop :=
  ∃ X : Point, X ∈ circumcircle ∧ line [P, E].contains X ∧ line [Q, D].contains X

theorem problem_a (A B C H D E F P Q : Point)
  (triangle : is_triangle A B C)
  (orthocenter : is_orthocenter H A B C)
  (midpoints : is_midpoint D A B ∧ is_midpoint E A C ∧ is_midpoint F A H)
  (reflections : is_reflection P B F ∧ is_reflection Q C F)
  (circumcircle : Set Point)
  (circumcircle_property : ∀ (X : Point), X ∈ circumcircle ↔ concyclic A B C X) :
  ∃ X : Point, X ∈ circumcircle ∧ line [P, E].contains X ∧ line [Q, D].contains X := 
sorry

theorem problem_b (A B C H D E F P Q : Point)
  (triangle : is_triangle A B C)
  (orthocenter : is_orthocenter H A B C)
  (midpoints : is_midpoint D A B ∧ is_midpoint E A C ∧ is_midpoint F A H)
  (reflections : is_reflection P B F ∧ is_reflection Q C F) :
  ∃ Y : Point, line [P, D].contains Y ∧ line [Q, E].contains Y ∧ segment [A, H].contains Y := 
sorry

end problem_a_problem_b_l707_707039


namespace omega_value_monotonicity_sin_alpha_l707_707288

noncomputable def f (ω x : ℝ) : ℝ := 2*(cos (ω * x))^2 - 1 + 2 * sqrt 3 * sin (ω * x) * cos (ω * x)
noncomputable def g (x : ℝ) : ℝ := 2 * cos (x / 2)

theorem omega_value (ω : ℝ) (hω : 0 < ω ∧ ω < 1) :
  (∃ k : ℤ, 2 * ω * (π / 3) + (π / 6) = 2 * k * π + (π / 2)) -> ω = 1 / 2 :=
by
  sorry

theorem monotonicity (x : ℝ) (h1 : x ∈ Icc (-π / 2) (π / 3)) (h2 : x ∈ Icc (π / 3) (π / 2)) :
  (∀ x ∈ Icc (-π / 2) (π / 3), StrictMono (f (1/2))) ∧
  (∀ x ∈ Icc (π / 3) (π / 2), StrictAnti (f (1/2))) :=
by
  sorry

theorem sin_alpha (α : ℝ) (hα : α ∈ Icc 0 (π / 2))\
  (hg: g (2 * α + π / 3) = 8 / 5) : sin α = (4 * sqrt 3 - 3) / 10 :=
by
  sorry

end omega_value_monotonicity_sin_alpha_l707_707288


namespace length_AB_l707_707321

-- Define the two circles O and O1
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 5

def circle_O1 (m x y : ℝ) : Prop := (x - m)^2 + y^2 = 20

-- Define the condition that the tangents at point A are perpendicular
def tangents_perpendicular (A : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_O x y ∧ circle_O1 m x y ∧ (x, y) = A ∧ ((circle_O y (-x) ∧ circle_O1 y m) → False)

-- State the mathematical proof problem
theorem length_AB 
  (m : ℝ) 
  (A B : ℝ × ℝ) 
  (h1 : ∃ A B, circle_O A.1 A.2 ∧ circle_O1 m A.1 A.2 ∧ circle_O B.1 B.2 ∧ circle_O1 m B.1 B.2)
  (h2 : tangents_perpendicular A m)
  : dist A B = 4 := 
sorry

end length_AB_l707_707321


namespace remainder_500th_in_T_l707_707027

def sequence_T (n : ℕ) : ℕ := sorry -- Assume a definition for the sequence T where n represents the position and the sequence contains numbers having exactly 9 ones in their binary representation.

theorem remainder_500th_in_T :
  (sequence_T 500) % 500 = 191 := 
sorry

end remainder_500th_in_T_l707_707027


namespace area_union_is_correct_l707_707175

-- Define the side length of the square and the radius of the circle
def side_length : ℝ := 12
def radius : ℝ := 12

-- Define the area of the square
def area_square : ℝ := side_length ^ 2

-- Define the area of the circle
def area_circle : ℝ := Real.pi * (radius ^ 2)

-- Define the area of the quarter circle inside the square
def area_quarter_circle : ℝ := (1 / 4) * area_circle

-- The total area of the union of the regions enclosed by the square and the circle
def area_union : ℝ := area_square + area_circle - area_quarter_circle

-- The theorem stating the desired result
theorem area_union_is_correct : area_union = 144 + 108 * Real.pi :=
by
  sorry

end area_union_is_correct_l707_707175


namespace proof_problem_l707_707649

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem proof_problem (x1 x2 : ℝ) (h₁ : x1 ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) 
                                (h₂ : x2 ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) 
                                (h₃ : f x1 + f x2 > 0) : 
  x1 + x2 > 0 :=
sorry

end proof_problem_l707_707649


namespace probability_zack_and_andrew_same_team_is_correct_l707_707113

noncomputable def probability_zack_and_andrew_same_team
    (players : Finset ℕ)
    (team1 team2 team3 : Finset ℕ)
    (Z M A : ℕ)
    (h1 : Z ≠ M)
    (h2 : M ≠ A)
    (h3 : team1.card = 9)
    (h4 : team2.card = 9)
    (h5 : team3.card = 9)
    (h6 : (team1 ∪ team2 ∪ team3) = players)
    (h7 : team1 ∩ team2 = ∅)
    (h8 : team1 ∩ team3 = ∅)
    (h9 : team2 ∩ team3 = ∅)
    (h10 : Z ∈ team1)
    (h11 : M ∈ team2) :
    ℚ :=
  let count_favorable := (team1.card - 1) in
  let total_possible := (team1.card + team3.card - 1) in
  (count_favorable : ℚ) / total_possible

theorem probability_zack_and_andrew_same_team_is_correct
    (players : Finset ℕ)
    (team1 team2 team3 : Finset ℕ)
    (Z M A : ℕ)
    (h1 : Z ≠ M)
    (h2 : M ≠ A)
    (h3 : team1.card = 9)
    (h4 : team2.card = 9)
    (h5 : team3.card = 9)
    (h6 : (team1 ∪ team2 ∪ team3) = players)
    (h7 : team1 ∩ team2 = ∅)
    (h8 : team1 ∩ team3 = ∅)
    (h9 : team2 ∩ team3 = ∅)
    (h10 : Z ∈ team1)
    (h11 : M ∈ team2) :
    probability_zack_and_andrew_same_team players team1 team2 team3 Z M A h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 = 8 / 17 :=
by
  sorry

end probability_zack_and_andrew_same_team_is_correct_l707_707113


namespace fewest_occupied_seats_l707_707457

theorem fewest_occupied_seats (n m : ℕ) (h₁ : n = 150) (h₂ : (m * 4 + 3 < 150)) : m = 37 :=
by
  sorry

end fewest_occupied_seats_l707_707457


namespace compare_f_n_l707_707635

def f (n : ℕ) : ℝ :=
  1 + ∑ i in Finset.range n, 1 / Real.sqrt (i + 2)

theorem compare_f_n (n : ℕ) (h : n > 0) : 
  (n = 1 ∨ n = 2 → f n < Real.sqrt (n + 1)) ∧ 
  (n ≥ 3 → f n > Real.sqrt (n + 1)) :=
by
  sorry

end compare_f_n_l707_707635


namespace num_two_digit_integers_l707_707282

theorem num_two_digit_integers (digits : Finset ℕ) (h : digits = {3, 5, 7, 8, 9}) :
  (digits.card * (digits.card - 1) = 20) :=
by
  -- conditions from the problem
  have h_card : digits.card = 5 := by rw [h]; simp
  -- goal is to prove the equation
  rw [h_card]
  norm_num
  exact sorry

end num_two_digit_integers_l707_707282


namespace power_of_point_locus_of_points_l707_707967

/-- Define the elements involved in the problem -/
variables {X O A B P Q : Point}
variable {r : ℝ}
variable {circle : Circle O r}
variable {arcAB : Arc A B}
variable {X_tangent_points : (Point × Point)}

/-- The power of a point theorem -/
theorem power_of_point (X P Q : Point) (r : ℝ) :
  dist(X, P) * dist(X, Q) = (dist(X, O) ^ 2) - r ^ 2 := sorry

/-- Distance condition for tangents from X to arc AB -/
def distance_condition (X A B : Point) (r : ℝ) : Prop :=
  dist(X, A) = r ∨ dist(X, B) = r

/-- The locus of points X from which tangents can be drawn to arc AB -/
theorem locus_of_points (X : Point) (arcAB : Arc A B) (X_tangent_points : (Point × Point))
    (distance_condition : distance_condition X A B r)
    (power_theorem : power_of_point X (fst X_tangent_points) (snd X_tangent_points) r) :
  is_locus_of_tangency X arcAB := sorry

end power_of_point_locus_of_points_l707_707967


namespace check_point_on_curve_l707_707213

def point_on_curve (x y : ℝ) : Prop :=
  x^2 - x * y + 2 * y + 1 = 0

theorem check_point_on_curve :
  point_on_curve 0 (-1/2) :=
by
  sorry

end check_point_on_curve_l707_707213


namespace measure_angle_ENG_l707_707408

-- Define the problem parameters
variables (E F G H N : Type)

-- Define the properties and conditions of the rectangle and points
variables [rect : is_rectangle E F G H]
variables [EF_eq_8 : distance E F = 8]
variables [FG_eq_4 : distance F G = 4]
variables [N_on_EF : point_on_line N E F]
variables [angle_condition : angle E N G = angle H N G]

-- Define the goal: Measure of the angle ENG is 90 degrees
theorem measure_angle_ENG : measure_angle E N G = 90 :=
sorry

end measure_angle_ENG_l707_707408


namespace max_digit_sum_in_time_range_l707_707159

theorem max_digit_sum_in_time_range : 
  ∃ hh mm ss : ℕ, (13 ≤ hh ∧ hh ≤ 23) ∧ (0 ≤ mm ∧ mm ≤ 59) ∧ (0 ≤ ss ∧ ss ≤ 59) ∧ 
  (sum_digits hh + sum_digits mm + sum_digits ss = 33) :=
sorry

end max_digit_sum_in_time_range_l707_707159


namespace four_digit_square_number_divisible_by_11_with_unit_1_l707_707811

theorem four_digit_square_number_divisible_by_11_with_unit_1 
  : ∃ y : ℕ, y >= 1000 ∧ y <= 9999 ∧ (∃ n : ℤ, y = n^2) ∧ y % 11 = 0 ∧ y % 10 = 1 ∧ y = 9801 := 
by {
  -- sorry statement to skip the proof.
  sorry 
}

end four_digit_square_number_divisible_by_11_with_unit_1_l707_707811


namespace real_part_eq_zero_modulus_z_l707_707644

-- Define the problem conditions
def cond (a : ℝ) : Prop :=
  a^2 - 1 = 0 ∧ a + 1 ≠ 0

-- Prove that a = 1 under the given conditions
theorem real_part_eq_zero (a : ℝ) (h : cond a) : a = 1 :=
  by sorry

-- Define the complex number z and compute its modulus
noncomputable def z (a : ℝ) : ℂ := 
  (a + complex.I.sqrt 3) / (a * complex.I)

-- Compute the modulus of z
theorem modulus_z (a : ℝ) (h : cond a) (a_is_one : a = 1) : complex.abs (z a) = 2 :=
  by rw [a_is_one, z, complex.abs, complex.of_real_mul, complex.mul_re, complex.mul_im];
     sorry

end real_part_eq_zero_modulus_z_l707_707644


namespace wax_needed_l707_707050

theorem wax_needed (total_wax : ℕ) (wax_has : ℕ) : total_wax = 288 → wax_has = 28 → total_wax - wax_has = 260 := by
  intros h1 h2
  rw [h1, h2]
  norm_num

end wax_needed_l707_707050


namespace probability_reach_l707_707771

theorem probability_reach (q : ℚ) (x y : ℕ) (h : Nat.coprime x y) :
  (q = 165 / 8192) ∧ (x + y = 8357) :=
by 
  sorry

end probability_reach_l707_707771


namespace surface_area_of_sliced_solid_l707_707899

noncomputable def surface_area_QVWX (height : ℝ) (side_length : ℝ) (midpoints : Fin 3 → ℝ × ℝ) : ℝ :=
  let area_QVX := (1/2) * (side_length / 2) * height
  let area_QWX := (1/2) * (side_length / 2) * height
  let area_QVW := (1/2) * side_length * (side_length * Mathlib.sqrt 3 / 2)
  let VW := side_length / 2
  let VX := Real.sqrt ((side_length / 2) * (side_length / 2) + height * height)
  let area_VWX := (1/2) * VW * (Real.sqrt (VX * VX - (VW / 2) * (VW / 2)))
  area_QVX + area_QWX + area_QVW + area_VWX

theorem surface_area_of_sliced_solid {height side_length : ℝ} (h_height : height = 20) (h_side_length : side_length = 10) 
    (midpoints : Fin 3 → ℝ × ℝ) :
  surface_area_QVWX height side_length midpoints = 50 + (25 * Mathlib.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 :=
by
  sorry

end surface_area_of_sliced_solid_l707_707899


namespace P_seq_eq_iff_derived_eq_l707_707319

-- Definitions
def P_seq (n : ℕ) (A : Fin n → ℕ) : Prop :=
  (∀ i, 1 ≤ A i ∧ A i ≤ n) ∧ Function.Injective A

def derived_seq (n : ℕ) (A : Fin n → ℕ) : Fin n → ℤ :=
  fun i =>
    match i with
    | ⟨0, _⟩ => 0
    | ⟨i+1, hi⟩ =>
      ∑ j in Finset.range (i+1), (A ⟨i+1, lt_trans (Finset.mem_range.mp (Finset.mem_range.mpr hi)) (Nat.succ_pos (i+1))⟩ - A ⟨j, Finset.mem_range.mp (Finset.mem_range.mpr (Nat.lt.base j))⟩) / 
      |(A ⟨i+1, _⟩ - A ⟨j, _⟩)|

-- Theorem statement
theorem P_seq_eq_iff_derived_eq {n : ℕ} (A A' : Fin n → ℕ) 
  (hA : P_seq n A) (hA' : P_seq n A') :
  (∀ i, A i = A' i) ↔ (∀ i, derived_seq n A i = derived_seq n A' i) :=
by
  sorry

end P_seq_eq_iff_derived_eq_l707_707319


namespace angle_bisector_slope_l707_707558

-- Definitions of the conditions
def line1_slope := 2
def line2_slope := 4

-- The proof statement: Prove that the slope of the angle bisector is -12/7
theorem angle_bisector_slope : (line1_slope + line2_slope + Real.sqrt (line1_slope^2 + line2_slope^2 + 2 * line1_slope * line2_slope)) / 
                               (1 - line1_slope * line2_slope) = -12/7 :=
by
  sorry

end angle_bisector_slope_l707_707558


namespace cosine_angle_BAD_l707_707704

theorem cosine_angle_BAD (AB AC BC : ℝ) (hAB : AB = 4) (hAC : AC = 7) (hBC : BC = 9)
  (D : Point) (h_bisects : bisects_angle D A B C) :
  cos_angle BAD = (Real.sqrt (5 / 14)) := by
  sorry

end cosine_angle_BAD_l707_707704


namespace man_speed_km_per_hour_l707_707164

theorem man_speed_km_per_hour:
  let distance_m := 600 in
  let distance_km := distance_m / 1000 in
  let time_min := 5 in
  let time_hr := time_min / 60 in
  (distance_km / time_hr) = 7.2 :=
by
  sorry

end man_speed_km_per_hour_l707_707164


namespace dihedral_angle_range_regular_prism_l707_707523

theorem dihedral_angle_range_regular_prism (n : ℕ) (h_n : n ≥ 3) :
  ∃ θ : Set ℝ, θ = Ioo (↑(n-2) / ↑n * Real.pi) Real.pi :=
sorry

end dihedral_angle_range_regular_prism_l707_707523


namespace not_possible_transform_l707_707877

-- Define bead colors
inductive Color
| black
| blue
| green

open Color

-- Define the initial configuration of the chain
def initial_chain : ℕ → Color
| 0 => black
| 1 => black
| _ => blue

-- Define the transformation rule
def transform (left right : Color) : Color :=
  if left = right then left else
    match (left, right) with
    | (black, blue) => green
    | (blue, black) => green
    | (black, green) => blue
    | (green, black) => blue
    | (blue, green) => black
    | (green, blue) => black
    | _ => left

-- Transformation step function
def step (chain : ℕ → Color) : ℕ → Color
| n => transform (chain n) (chain ((n + 1) % 2016))

-- Define the main theorem
theorem not_possible_transform :
  ¬(∃ final_chain : ℕ → Color, 
     final_chain ((2016 + 1) % 2016) = green ∧
     (∀ n ≠ ((2016 + 1) % 2016), final_chain n = blue) ∧
     (initial_chain → step) → final_chain) :=
sorry

end not_possible_transform_l707_707877


namespace arrangement_count_SUCCESS_l707_707938

-- Define the conditions for the problem
def letters : Finset String := {"S", "U", "C", "C", "E", "S", "S"}
def occurrences_S : Nat := 3
def occurrences_C : Nat := 2
def occurrences_other : Nat := 1 -- For 'U' and 'E'

-- State the theorem using these conditions
theorem arrangement_count_SUCCESS : 
  let N := letters.card
  N = 7 →
  occurrences_S = 3 →
  occurrences_C = 2 →
  occurrences_other = 1 →
  Nat.factorial N / (Nat.factorial occurrences_S * Nat.factorial occurrences_C * Nat.factorial occurrences_other * Nat.factorial occurrences_other) = 420 :=
by
  sorry

end arrangement_count_SUCCESS_l707_707938


namespace tangent_lines_count_l707_707668

noncomputable def circle1 : Real × Real × Real := (-2, 2, 1) -- center and radius of circle1
noncomputable def circle2 : Real × Real × Real := (2, 5, 4)   -- center and radius of circle2

theorem tangent_lines_count:
  let c1 := (-2:Real, 2:Real, 1:Real) in
  let c2 := (2:Real, 5:Real, 4:Real) in
  (c1, c2) = ((-2,4,1),(2,5,4)) → 3 := by
  sorry

end tangent_lines_count_l707_707668


namespace count_success_permutations_l707_707940

theorem count_success_permutations : 
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  (Nat.factorial total_letters) / ((Nat.factorial s_count) * (Nat.factorial c_count)) = 420 := 
by
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  sorry

end count_success_permutations_l707_707940


namespace compare_abc_l707_707651

open Real

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Assuming the conditions
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom derivative : ∀ x : ℝ, f' x = deriv f x
axiom monotonicity_condition : ∀ x > 0, x * f' x < f x

-- Definitions of a, b, and c
noncomputable def a := 2 * f (1 / 2)
noncomputable def b := - (1 / 2) * f (-2)
noncomputable def c := - (1 / log 2) * f (log (1 / 2))

theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l707_707651


namespace eq_dist_DF_KF_l707_707684

-- Definitions of points and line segments in ΔABC with given conditions
variables (A B C D E G H K F : Type)
variables [Point A] [Point B] [Point C] [Point D] [Point E] [Point G] [Point H] [Point K] [Point F]
variables [Triangle ABC]

-- Conditions
variable (h1 : AB = AC)
variable (h2 : Perpendicular AD BC)
variable (h3 : Perpendicular DE AC)
variable (h4 : Midpoint G BE)
variable (h5 : Perpendicular EH AG)
variable (h6 : IntersectsAt EH AD K)
variable (h7 : IntersectsAt BE AD F)

-- Theorem to prove DF = KF
theorem eq_dist_DF_KF :
  DF = KF := 
sorry

end eq_dist_DF_KF_l707_707684


namespace factorize_expression_l707_707594

theorem factorize_expression (x : ℝ) :
  9 * x^2 - 6 * x + 1 = (3 * x - 1)^2 := 
by sorry

end factorize_expression_l707_707594


namespace proof_problem_l707_707984

noncomputable def greatestIntLe (x : ℝ) : ℤ := 
  int.floor x

theorem proof_problem :
  greatestIntLe 6.5 * greatestIntLe (2 / 3) + greatestIntLe 2 * (7.2 : ℝ) + greatestIntLe 8.4 - (9.8 : ℝ) = 12.599999999999998 := 
by 
  sorry

end proof_problem_l707_707984


namespace range_of_m_for_circle_l707_707676

theorem range_of_m_for_circle (m : ℝ) :
  (∃ x y, x^2 + y^2 - 4 * x - 2 * y + m = 0) → m < 5 :=
by
  sorry

end range_of_m_for_circle_l707_707676


namespace evaluate_expression_l707_707591

theorem evaluate_expression : 
  ( (7 : ℝ) ^ (1 / 4) / (7 : ℝ) ^ (1 / 7) ) = 7 ^ (3 / 28) := 
by {
  sorry
}

end evaluate_expression_l707_707591


namespace collinearity_condition_l707_707632

noncomputable def collinear (O A B P : Type) [AddGroup O] [AddGroup A] [AddGroup B] [AddGroup P] 
  [Module ℝ O] [Module ℝ A] [Module ℝ B] [Module ℝ P] (m n : ℝ) 
  (h : ¬(Vector3d.mkOAB 0 = 0)) : Prop :=
∃ (OP OA OB : A → A), OP = m • OA + n • OB → m + n = 1 ↔ ∃ (λ : ℝ), P = (1 - λ) • A + λ • B

theorem collinearity_condition {O A B P : Type} [AddGroup O] [AddGroup A] [AddGroup B] [AddGroup P] 
  [Module ℝ O] [Module ℝ A] [Module ℝ B] [Module ℝ P] (m n : ℝ) 
  (h : ¬(Vector3d.mkOAB 0 = 0)) : collinear O A B P m n h :=
sorry

end collinearity_condition_l707_707632


namespace macy_miles_left_to_run_l707_707743

-- Define the given conditions
def goal : ℕ := 24
def miles_per_day : ℕ := 3
def days : ℕ := 6

-- Define the statement to be proven
theorem macy_miles_left_to_run :
  goal - (miles_per_day * days) = 6 :=
by
  sorry

end macy_miles_left_to_run_l707_707743


namespace C_plus_D_l707_707361

def four_digit_numbers := {n : ℕ // 1000 ≤ n ∧ n ≤ 9999}

def odd_and_divisible_by_5 (n : ℕ) : Prop := n % 2 = 1 ∧ n % 5 = 0

def number_of_odd_and_divisible_by_5 : ℕ :=
  {n : ℕ // n % 2 = 1 ∧ n % 5 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999}.card

lemma C_eq :
  number_of_odd_and_divisible_by_5 = 900 :=
sorry

def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def number_of_multiples_of_3 : ℕ :=
  {n : ℕ // n % 3 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999}.card

lemma D_eq :
  number_of_multiples_of_3 = 3000 :=
sorry

theorem C_plus_D :
  number_of_odd_and_divisible_by_5 + number_of_multiples_of_3 = 3900 :=
by
  rw [C_eq, D_eq]
  exact rfl

end C_plus_D_l707_707361


namespace max_average_hours_l707_707385

theorem max_average_hours :
  let hours_Wednesday := 2
  let hours_Thursday := 2
  let hours_Friday := hours_Wednesday + 3
  let total_hours := hours_Wednesday + hours_Thursday + hours_Friday
  let average_hours := total_hours / 3
  average_hours = 3 :=
by
  sorry

end max_average_hours_l707_707385


namespace library_leftover_space_l707_707182

theorem library_leftover_space :
  let desk_length := 2
  let bookcase_length := 1.5
  let chair_length := 0.5
  let gap := 0.5
  let total_wall_length := 60
  let set_length := desk_length + gap + bookcase_length + gap + chair_length + gap
  let max_sets := (total_wall_length / set_length).toNat
  let occupied_length := max_sets * set_length
  total_wall_length - occupied_length = 5 :=
by
  let desk_length := 2
  let bookcase_length := 1.5
  let chair_length := 0.5
  let gap := 0.5
  let total_wall_length := 60
  let set_length := desk_length + gap + bookcase_length + gap + chair_length + gap
  let max_sets := (total_wall_length / set_length).toNat
  let occupied_length := max_sets * set_length
  have h1 : set_length = 5.5 := by norm_num
  have h2 : (total_wall_length / set_length).toNat = 10 := by norm_num
  have h3 : occupied_length = 55 := by norm_num
  have h4 : total_wall_length - occupied_length = 5 := by norm_num
  exact h4

end library_leftover_space_l707_707182


namespace angle_AKC_obtuse_l707_707751

open Real EuclideanGeometry

noncomputable def is_cyclic_quad (A B C D : Point) : Prop :=
∃ O, Circle O ≠ ∅ ∧ Quadrilateral A B C D ∧ A, B, C, D ∈ Circle O

theorem angle_AKC_obtuse
  (A B C C1 A1 K I : Point)
  (hC1 : lies_on C1 (line_through A B))
  (hA1 : lies_on A1 (line_through B C))
  (hC1_ne_A : C1 ≠ A) 
  (hA1_ne_C : A1 ≠ C)
  (hK : midpoint A1 C1 K)
  (hI : incenter I A B C)
  (hCyclic : is_cyclic_quad A1 B C1 I) :
  ∠ A K C > π / 2 :=
sorry

end angle_AKC_obtuse_l707_707751


namespace squareD_perimeter_l707_707419

-- Let perimeterC be the perimeter of square C
def perimeterC : ℝ := 40

-- Let sideC be the side length of square C
def sideC := perimeterC / 4

-- Let areaC be the area of square C
def areaC := sideC * sideC

-- Let areaD be the area of square D, which is one-third the area of square C
def areaD := (1 / 3) * areaC

-- Let sideD be the side length of square D
def sideD := Real.sqrt areaD

-- Let perimeterD be the perimeter of square D
def perimeterD := 4 * sideD

-- The theorem to prove
theorem squareD_perimeter (h : perimeterC = 40) (h' : areaD = (1 / 3) * areaC) : perimeterD = (40 * Real.sqrt 3) / 3 := by
  sorry

end squareD_perimeter_l707_707419


namespace log8_32a_eq_4p_plus_4_over_3_l707_707671

variable (a p q : ℝ)

-- Definitions based on conditions
def log16_2a_eq_p := log 16 (2 * a) = p
def log2_a_plus_5_eq_q := log 2 (a + 5) = q

-- The final theorem statement
theorem log8_32a_eq_4p_plus_4_over_3 (hyp1 : log16_2a_eq_p a p) (hyp2 : log2_a_plus_5_eq_q a q) :
  log 8 (32 * a) = (4 * p + 4) / 3 :=
by
  sorry

end log8_32a_eq_4p_plus_4_over_3_l707_707671


namespace length_of_hook_is_67_l707_707450

-- Definitions of the conditions:
def area_of_square (s : ℝ) : ℝ := s * s
def diagonal_of_square (s : ℝ) : ℝ := s * Real.sqrt 2
def radius_of_circle (d : ℝ) : ℝ := d / 2
def circumference_of_circle (r : ℝ) : ℝ := 2 * Real.pi * r
def major_arc_length (circumference : ℝ) : ℝ := 3 / 4 * circumference
def segment_AB_length (r : ℝ) : ℝ := 2 * r

-- Given conditions:
constant area : ℝ := 200
constant s : ℝ := Real.sqrt 200
constant d : ℝ := s * Real.sqrt 2
constant r : ℝ := d / 2
constant major_arc_BE : ℝ := major_arc_length (circumference_of_circle r)
constant segment_AB : ℝ := segment_AB_length r
constant total_hook_length : ℝ := segment_AB + major_arc_BE

-- Proof statement:
theorem length_of_hook_is_67 : Real.round total_hook_length = 67 := by
  sorry

end length_of_hook_is_67_l707_707450


namespace paint_needs_and_savings_and_cost_l707_707855

noncomputable def total_paint_volume (large_volume small_volume : ℕ) (shortage leftover : ℕ) : ℕ :=
  large_volume * 4 + shortage

noncomputable def savings_from_promotion (small_volume small_price : ℕ) (free_buckets discount : ℕ) : ℕ :=
  let total_buckets := 15
  let paid_buckets := total_buckets * free_buckets / (free_buckets + 1)
  let original_cost := total_buckets * small_price
  let discounted_cost := paid_buckets * small_price - discount
  original_cost - discounted_cost

noncomputable def calculate_cost_per_bucket (small_price total_buckets discount profit_margin : ℕ) : ℕ :=
  let profit_equation_rhs := small_price * (total_buckets + total_buckets * profit_margin / 100)
  (small_price - discount) * total_buckets / profit_margin

theorem paint_needs_and_savings_and_cost {large_volume small_volume : ℕ} {shortage leftover small_price free_buckets discount profit_margin: ℕ}
  (h1 : large_volume = 18) (h2 : small_volume = 5) (h3 : shortage = 2) (h4 : leftover = 1) (h5 : small_price = 90) (h6 : free_buckets = 4) (h7 : discount = 120) (h8 : profit_margin = 25):
  total_paint_volume large_volume small_volume shortage leftover = 74 ∧
  savings_from_promotion small_volume small_price free_buckets discount = 390 ∧
  calculate_cost_per_bucket small_price 15 discount profit_margin = 51.2 :=
by sorry

end paint_needs_and_savings_and_cost_l707_707855


namespace pyramid_surface_area_and_volume_l707_707436

noncomputable def total_surface_area (a : ℝ) : ℝ :=
  a^2 * (Real.sqrt 7 + 1) / 2

noncomputable def volume (a : ℝ) : ℝ :=
  Real.sqrt 3 * a^3 / 12

theorem pyramid_surface_area_and_volume (a : ℝ) (h1 : a > 0) :
  ∃ (S V : ℝ), S = total_surface_area a ∧ V = volume a :=
by
  use total_surface_area a
  use volume a
  rw [total_surface_area, volume]
  sorry   -- proof steps would go here

end pyramid_surface_area_and_volume_l707_707436


namespace revenue_increase_by_36percent_l707_707394

def revenue_effect (P Q : ℝ) : ℝ :=
  let R := P * Q
  let P_new := 1.70 * P
  let Q_new := 0.80 * Q
  let R_new := P_new * Q_new
  (R_new - R) / R

theorem revenue_increase_by_36percent (P Q : ℝ) (hP : P > 0) (hQ : Q > 0)
  (hP_increase : P_new = 1.70 * P)
  (hQ_decrease : Q_new = 0.80 * Q) :
  revenue_effect P Q = 0.36 :=
by
  sorry

end revenue_increase_by_36percent_l707_707394


namespace ab_value_l707_707268

theorem ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 30) (h4 : 2 * a * b + 12 * a = 3 * b + 240) :
  a * b = 255 :=
sorry

end ab_value_l707_707268


namespace thm_300th_term_non_square_seq_l707_707486

theorem thm_300th_term_non_square_seq : 
  let non_square_seq (n : ℕ) := { k : ℕ // k > 0 ∧ ∀ m : ℕ, m * m ≠ k } in
  (non_square_seq 300).val = 318 :=
by
  sorry

end thm_300th_term_non_square_seq_l707_707486


namespace probability_triangle_PBC_l707_707396

-- Definitions based on the given conditions
def is_on_side (P A B : Point) : Prop :=
  collinear P A B ∧ distance A P + distance P B = distance A B

def probability_area_triangle (ABC PBC : Triangle) : ℝ :=
  let S_ABC := area ABC
  let S_PBC := area PBC
  if S_ABC > 0 then S_PBC / S_ABC else 0

-- Lean statement based on the equivalent proof problem
theorem probability_triangle_PBC (A B C P : Point) (h d : ℝ) (ABC : Triangle) (P_on_AB : is_on_side P A B)
  (h_le_d : h ≤ (1/3) * d) :
  probability_area_triangle ABC (Triangle.mk P B C) = 1 / 3 := by
  sorry

end probability_triangle_PBC_l707_707396


namespace share_of_c_l707_707136

noncomputable def investment_ratio (a b c : ℝ) : Prop :=
  a = 3 * b ∧ a = (2 / 3) * c

noncomputable def total_profit : ℝ := 55000

noncomputable def c_share_of_profit (c_ratio total_parts : ℝ) : ℝ :=
  (c_ratio / total_parts) * total_profit

theorem share_of_c (a b c total_parts : ℝ) (h1 : a = 3 * b) (h2 : a = (2 / 3) * c) :
  c_share_of_profit (9 * b / 2) (17 * b) = 29117.65 :=
by
  have h3 : a * 3 / 2 = c := by sorry
  have h4 : total_parts = 17 * b := by sorry
  have h5 : (9 * b / 17 * b) * total_profit = 29117.65 := by sorry
  exact h5

end share_of_c_l707_707136


namespace max_min_comparisons_needed_max_min_comparisons_insufficiency_l707_707337

noncomputable def find_max_min_comparisons (n : ℕ) : ℕ :=
  3 * n - 2

theorem max_min_comparisons_needed (n : ℕ) : 
  (find_max_min_comparisons n) = 3 * n - 2 := by
  sorry

theorem max_min_comparisons_insufficiency (n : ℕ) (k : ℕ) :
  k < 3 * n - 2 → ¬(∃ max min : ℕ, ∀ seq : list ℕ, seq.length = 2 * n → 
    ∃ c : ℕ, c < 3 * n - 2 ∧ c = (pairwise_comparisons_needed seq)) := by
  sorry

end max_min_comparisons_needed_max_min_comparisons_insufficiency_l707_707337


namespace trapezoid_area_division_ratio_l707_707085

theorem trapezoid_area_division_ratio (ABCD : Trapezoid) 
  (h1 : ratio_area_divided_by_diagonal ABCD 3 7)
  (h2 : equal_triangle_heights ABCD) : 
  ratio_area_divided_by_median_line ABCD 2 3 :=
sorry

end trapezoid_area_division_ratio_l707_707085


namespace tangent_line_y_intercept_l707_707786

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

theorem tangent_line_y_intercept (a : ℝ) (h : 3 * (1:ℝ)^2 - a = 1) :
  (∃ (m b : ℝ), ∀ (x : ℝ), m = 1 ∧ y = x - 2 → y = m * x + b) := 
 by
  sorry

end tangent_line_y_intercept_l707_707786


namespace power_of_point_locus_of_points_l707_707966

/-- Define the elements involved in the problem -/
variables {X O A B P Q : Point}
variable {r : ℝ}
variable {circle : Circle O r}
variable {arcAB : Arc A B}
variable {X_tangent_points : (Point × Point)}

/-- The power of a point theorem -/
theorem power_of_point (X P Q : Point) (r : ℝ) :
  dist(X, P) * dist(X, Q) = (dist(X, O) ^ 2) - r ^ 2 := sorry

/-- Distance condition for tangents from X to arc AB -/
def distance_condition (X A B : Point) (r : ℝ) : Prop :=
  dist(X, A) = r ∨ dist(X, B) = r

/-- The locus of points X from which tangents can be drawn to arc AB -/
theorem locus_of_points (X : Point) (arcAB : Arc A B) (X_tangent_points : (Point × Point))
    (distance_condition : distance_condition X A B r)
    (power_theorem : power_of_point X (fst X_tangent_points) (snd X_tangent_points) r) :
  is_locus_of_tangency X arcAB := sorry

end power_of_point_locus_of_points_l707_707966


namespace polynomial_not_factorable_l707_707759

theorem polynomial_not_factorable :
  ¬ ∃ (A B : Polynomial ℤ), A.degree < 5 ∧ B.degree < 5 ∧ A * B = (Polynomial.C 1 * Polynomial.X ^ 5 - Polynomial.C 3 * Polynomial.X ^ 4 + Polynomial.C 6 * Polynomial.X ^ 3 - Polynomial.C 3 * Polynomial.X ^ 2 + Polynomial.C 9 * Polynomial.X - Polynomial.C 6) :=
by
  sorry

end polynomial_not_factorable_l707_707759


namespace product_of_simplified_fraction_is_fifteen_l707_707128

theorem product_of_simplified_fraction_is_fifteen :
  let num := 45
  let denom := 75
  let simplified_fraction := (num / Nat.gcd num denom) / (denom / Nat.gcd num denom)
  let simp_num := Nat.gcd num denom
  let simp_denom := Nat.gcd denom denom
  (simp_num * simp_denom = 15) := by
  let num := 45
  let denom := 75
  let gcd_num_denom := Nat.gcd num denom
  let simplified_num := num / gcd_num_denom
  let simplified_denom := denom / gcd_num_denom
  let product := simplified_num * simplified_denom 
  show product = 15, from sorry

end product_of_simplified_fraction_is_fifteen_l707_707128


namespace largest_solution_of_equation_l707_707505

theorem largest_solution_of_equation :
  let eq := λ x : ℝ => x^4 - 50 * x^2 + 625
  ∃ x : ℝ, eq x = 0 ∧ ∀ y : ℝ, eq y = 0 → y ≤ x :=
sorry

end largest_solution_of_equation_l707_707505


namespace books_in_bin_after_actions_l707_707382

theorem books_in_bin_after_actions (x y : ℕ) (z : ℕ) (hx : x = 4) (hy : y = 3) (hz : z = 250) : x - y + (z / 100) * x = 11 :=
by
  rw [hx, hy, hz]
  -- x - y + (z / 100) * x = 4 - 3 + (250 / 100) * 4
  norm_num
  sorry

end books_in_bin_after_actions_l707_707382


namespace median_of_consecutive_integers_l707_707101

theorem median_of_consecutive_integers (sum_n : ℕ) (n : ℕ) 
  (h_sum : sum_n = 3 ^ 8) (h_n : n = 81) : 
  let median := (sum_n / n) in
  median = 81 :=
by
  sorry

end median_of_consecutive_integers_l707_707101


namespace proportion_solution_l707_707317

theorem proportion_solution (a b c x : ℝ) (h : a / x = 4 * a * b / (17.5 * c)) : 
  x = 17.5 * c / (4 * b) := 
sorry

end proportion_solution_l707_707317


namespace tan_sum_identity_l707_707992

theorem tan_sum_identity (α : ℝ) (h₁ : Real.cos α = -4 / 5) (h₂ : α ∈ Ioc (Real.pi / 2) Real.pi) :
  Real.tan (α + Real.pi / 4) = 1 / 7 :=
sorry

end tan_sum_identity_l707_707992


namespace triangle_area_l707_707332

theorem triangle_area (p : ℝ) (h : 0 ≤ p ∧ p ≤ 15) :
  let C := (0, p)
  let O := (0, 0)
  let B := (15, 0)
  area_triangle : ℝ :=
  (1 / 2) * 15 * p = (15 * p) / 2 :=
by
  sorry

end triangle_area_l707_707332


namespace middle_distance_lines_l707_707256

-- Defining the structure of a triangle
structure Triangle (α : Type _) :=
(A B C : α)

-- Defining the concept of centroid
def centroid {α : Type _} [Add α] [Div α] 
    (t : Triangle α) : α := (t.A + t.B + t.C) / 3

-- Define the absolute distance from a point to a line
def line_distance {α : Type _} [LinearOrder α] (p : α) (line : α → Prop) : α := 
    sorry -- Implementation of distance from point to line

-- Define middle distance based on conditions in the problem
def middle_distance {α : Type _} [LinearOrder α] (t : Triangle α) 
    (line : α → Prop) : α :=
    sorry -- Implementation of middle distance calculation

-- Define the main theorem
theorem middle_distance_lines {α : Type _} [LinearOrder α] 
    (t : Triangle α) (d : α) :
    {line : α → Prop | middle_distance t line = d} = 
    {line : α → Prop | line_distance (centroid t) line = d / 3} :=
by sorry

end middle_distance_lines_l707_707256


namespace find_x_l707_707724

-- Define the operation c bowtie x
def bowtie (c x : ℝ) : ℝ := c + real.sqrt (x + real.sqrt (x + real.sqrt (x + ...)))

-- Given condition
axiom condition : bowtie 5 x = 11

-- Theorem statement
theorem find_x (x : ℝ) (h : bowtie 5 x = 11) : x = 30 :=
sorry

end find_x_l707_707724


namespace functional_eqn_solution_l707_707225

def f : ℝ → ℝ := sorry

theorem functional_eqn_solution (f : ℝ → ℝ) :
  (∀ a b c : ℝ, a + b + c ≥ 0 → f(a^3) + f(b^3) + f(c^3) ≥ 3 * f(a * b * c)) ∧
  (∀ a b c : ℝ, a + b + c ≤ 0 → f(a^3) + f(b^3) + f(c^3) ≤ 3 * f(a * b * c)) →
  ∃ k : ℝ, k ≥ 0 ∧ ∀ x : ℝ, f(x) = k * x :=
sorry

end functional_eqn_solution_l707_707225


namespace min_value_of_f_l707_707443

noncomputable def f (x : ℝ) : ℝ := (1 / real.sqrt (x^2 + 2)) + real.sqrt (x^2 + 2)

theorem min_value_of_f : f 0 = (3 * real.sqrt 2) / 2 := by
  sorry

end min_value_of_f_l707_707443


namespace probability_odd_divisor_15_factorial_l707_707091

theorem probability_odd_divisor_15_factorial :
  let number_of_divisors_15_fact : ℕ := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  let number_of_odd_divisors_15_fact : ℕ := (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  (number_of_odd_divisors_15_fact : ℝ) / (number_of_divisors_15_fact : ℝ) = 1 / 12 :=
by
  sorry

end probability_odd_divisor_15_factorial_l707_707091


namespace ab_value_l707_707996

theorem ab_value (a b : ℝ) (h : (1 : ℂ) + 3 * complex.I * (a + b * complex.I) = 10 * complex.I) : a * b = 3 :=
by
  sorry

end ab_value_l707_707996


namespace non_square_300th_term_l707_707501

theorem non_square_300th_term (N : ℕ) (hN : N = 300) : 
  ∃ x : ℕ, (∀ (t : ℕ), 0 < t ∧ t ≤ x → ¬ (∃ k : ℕ, t = k^2)) ∧ 
           (∑ t in finset.range (N + 17), if ∃ k : ℕ, t = k^2 then 0 else 1 = N) ∧ 
           x = 317 := by
  sorry

end non_square_300th_term_l707_707501


namespace project_presentation_period_length_l707_707535

theorem project_presentation_period_length
  (students : ℕ)
  (presentation_time_per_student : ℕ)
  (number_of_periods : ℕ)
  (total_students : students = 32)
  (time_per_student : presentation_time_per_student = 5)
  (periods_needed : number_of_periods = 4) :
  (32 * 5) / 4 = 40 := 
by {
  sorry
}

end project_presentation_period_length_l707_707535


namespace three_hundredth_term_without_squares_l707_707494

noncomputable def sequence_without_squares (n : ℕ) : ℕ :=
(n + (n / Int.natAbs (Int.sqrt (n.succ - 1))))

theorem three_hundredth_term_without_squares : 
  sequence_without_squares 300 = 307 :=
sorry

end three_hundredth_term_without_squares_l707_707494


namespace new_area_of_rectangle_l707_707778

theorem new_area_of_rectangle (L W : ℝ) (h : L * W = 600) :
  let new_length := 0.8 * L
  let new_width := 1.05 * W
  new_length * new_width = 504 :=
by 
  sorry

end new_area_of_rectangle_l707_707778


namespace number_of_petri_dishes_l707_707342

noncomputable def total_germs : ℝ := 0.036 * 10^5
noncomputable def germs_per_dish : ℝ := 99.99999999999999

theorem number_of_petri_dishes : 36 = total_germs / germs_per_dish :=
by sorry

end number_of_petri_dishes_l707_707342


namespace train_length_problem_l707_707177

/-- The statement of the problem in Lean 4 -/
theorem train_length_problem :
  ∀ (v t : ℝ) (p : ℝ), v = 90 * (1000 / 3600) → t = 25 → p = 400.05 → 
  let train_length := v * t - p in
  train_length = 224.95 :=
begin
  intros v t p hv ht hp,
  have train_speed : v = 25 := by rw [hv, mul_div_cancel_left], -- speed conversion: 90 kmph = 25 m/s
  have total_distance : v * t = 625 := by rw [train_speed, ht, mul_comm, ←mul_assoc], -- distance calculation: 25 m/s * 25 s = 625 m
  have train_eqn : 625 = 224.95 + 400.05 := by norm_num, -- Given equation: Distance covered equals the sum of train length and platform length
  rw [←train_eqn] at total_distance,
  have train_length_eq : 625 - 400.05 = 224.95 := by norm_num, -- Solving for train length
  exact train_length_eq,
end

end train_length_problem_l707_707177


namespace median_of_consecutive_integers_l707_707100

theorem median_of_consecutive_integers (sum_n : ℕ) (n : ℕ) 
  (h_sum : sum_n = 3 ^ 8) (h_n : n = 81) : 
  let median := (sum_n / n) in
  median = 81 :=
by
  sorry

end median_of_consecutive_integers_l707_707100


namespace eight_sharp_two_equals_six_thousand_l707_707928

def new_operation (a b : ℕ) : ℕ :=
  (a + b) ^ 3 * (a - b)

theorem eight_sharp_two_equals_six_thousand : new_operation 8 2 = 6000 := 
  by
    sorry

end eight_sharp_two_equals_six_thousand_l707_707928


namespace distance_origin_point_l707_707691

noncomputable def distance_from_origin (x y : ℝ) : ℝ :=
  real.sqrt (x^2 + y^2)

theorem distance_origin_point (x y : ℝ) (h1 : x = 3.5) (h2 : y = -18) :
  distance_from_origin x y = 18.3 :=
by
  rw [h1, h2]
  sorry -- Calculation steps to be filled in as the proof

end distance_origin_point_l707_707691


namespace maximize_expression_l707_707617

theorem maximize_expression 
  (x y z: ℝ)
  (hx: 0 < x)
  (hy: 0 < y)
  (hz: 0 < z)
  (hxyz: x + y + z = 1) : 
  x + Real.sqrt(2 * x * y) + 3 * Real.cbrt(x * y * z) ≤ 2 ∧  -- Upper Bound
  (∃ u v w, 0 < u ∧ 0 < v ∧ 0 < w ∧ u + v + w = 1 ∧ 
      u + Real.sqrt(2 * u * v) + 3 * Real.cbrt(u * v * w) = 2) := sorry

end maximize_expression_l707_707617


namespace anita_cartons_of_strawberries_l707_707184

theorem anita_cartons_of_strawberries (total_cartons_needed : ℕ)
  (blueberry_cartons : ℕ) (cartons_to_buy : ℕ) :
  total_cartons_needed = 26 →
  blueberry_cartons = 9 →
  cartons_to_buy = 7 →
  (total_cartons_needed - (blueberry_cartons + cartons_to_buy)) = 10 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end anita_cartons_of_strawberries_l707_707184


namespace range_of_a_l707_707029

open Real

noncomputable def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, (1/2) ^ abs x < a

noncomputable def prop_q (a : ℝ) : Prop :=
  ∀ x : ℝ, ax^2 + (a-2)*x + 9/8 > 0

theorem range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬ (prop_p a ∧ prop_q a) ↔ (a ≥ 8 ∨ 1/2 < a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l707_707029


namespace large_cube_surface_area_l707_707141

-- Define given conditions
def small_cube_volume := 512 -- volume in cm^3
def num_small_cubes := 8

-- Define side length of small cube
def small_cube_side_length := (small_cube_volume : ℝ)^(1/3)

-- Define side length of large cube
def large_cube_side_length := 2 * small_cube_side_length

-- Surface area formula for a cube
def surface_area (side_length : ℝ) := 6 * side_length^2

-- Theorem: The surface area of the large cube is 1536 cm^2
theorem large_cube_surface_area :
  surface_area large_cube_side_length = 1536 :=
sorry

end large_cube_surface_area_l707_707141


namespace tom_total_distance_l707_707467

def tomTimeSwimming : ℝ := 2
def tomSpeedSwimming : ℝ := 2
def tomTimeRunning : ℝ := (1 / 2) * tomTimeSwimming
def tomSpeedRunning : ℝ := 4 * tomSpeedSwimming

theorem tom_total_distance : 
  let distanceSwimming := tomSpeedSwimming * tomTimeSwimming
  let distanceRunning := tomSpeedRunning * tomTimeRunning
  distanceSwimming + distanceRunning = 12 := by
  sorry

end tom_total_distance_l707_707467


namespace area_perimeter_inequality_l707_707842

structure EquilateralTriangle :=
(a b c : ℝ)
(ab bc ca : ℝ) -- lengths of sides, implicitly equilateral

structure Point2D :=
(x y : ℝ)

structure Triangle :=
(a b c : Point2D)

def area (T : Triangle) : ℝ := sorry
def perimeter (T : Triangle) : ℝ := sorry

theorem area_perimeter_inequality
  (ABC : EquilateralTriangle)
  (A₁ A₂ : Point2D)
  (hA₁ : ∃ A : Point2D, Triangle.mk A ABC.b ABC.c = ABC ∧ A = A₁) -- A₁ is inside ABC
  (hA₂ : ∃ A : Point2D, Triangle.mk A ABC.b ABC.c = Triangle.mk A₁ ABC.b ABC.c ∧ A = A₂) -- A₂ is inside A₁BC
  (S₁ S₂ P₁ P₂ : ℝ)
  (hS₁ : S₁ = area (Triangle.mk A₁ ABC.b ABC.c))
  (hS₂ : S₂ = area (Triangle.mk A₂ ABC.b ABC.c))
  (hP₁ : P₁ = perimeter (Triangle.mk A₁ ABC.b ABC.c))
  (hP₂ : P₂ = perimeter (Triangle.mk A₂ ABC.b ABC.c)) :
  (S₁ / P₁^2 > S₂ / P₂^2) :=
begin
  sorry
end

end area_perimeter_inequality_l707_707842


namespace sequence_divisibility_exists_l707_707957

theorem sequence_divisibility_exists (n : ℕ) (h1 : ∃ (x : Fin n → ℕ), (∀ (i : Fin n), ∃ (k : ℕ), x i = k ∧ k ∈ (Finset.range n).image (+1)) 
  ∧ (∀ (k : ℕ), k ∈ Finset.range n.succ → k ∣ Finset.sum (Finset.range n) (λ i => x ⟨i, h1 i⟩))) :
  n = 1 ∨ n = 3 :=
sorry

end sequence_divisibility_exists_l707_707957


namespace isosceles_triangle_circle_ratio_l707_707055

theorem isosceles_triangle_circle_ratio 
  (A B C N : Point)
  (isosceles : AC = C)
  (circle : Circle)
  (diameter : circle.diameter = AC)
  (intersect : circle ∩ BC = {N})
  (ratio_bn_nc : BN / NC = 7 / 2) : AN / BC = 4 * Real.sqrt 2 / 9 := by
sorry

end isosceles_triangle_circle_ratio_l707_707055


namespace true_propositions_count_l707_707242

theorem true_propositions_count (a b c : ℝ) : 
  (¬(a > b → ac ^ 2 > bc ^ 2) ∧ (ac ^ 2 > bc ^ 2 → a > b) ∧ (a ≤ b → ac ^ 2 ≤ bc ^ 2) ∧ ¬(ac ^ 2 ≤ bc ^ 2 → a ≤ b)) → 
  (count_true_propositions 2) :=
sorry

-- Helper function to count true propositions
def count_true_propositions : ℕ := 
  if (¬(a > b → ac ^ 2 > bc ^ 2)) + 
     (ac ^ 2 > bc ^ 2 → a > b) + 
     (a ≤ b → ac ^ 2 ≤ bc ^ 2) + 
     ¬(ac ^ 2 ≤ bc ^ 2 → a ≤ b) = 2 then
  2
else 
  0

end true_propositions_count_l707_707242


namespace proof_by_contradiction_example_l707_707757

theorem proof_by_contradiction_example (a b c : ℝ) (h : a < 3 ∧ b < 3 ∧ c < 3) : a < 1 ∨ b < 1 ∨ c < 1 := 
by
  have h1 : a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1 := sorry
  sorry

end proof_by_contradiction_example_l707_707757


namespace no_such_positive_integer_exists_l707_707942

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.to_list.sum

theorem no_such_positive_integer_exists :
  ¬ ∃ (n : ℕ), 0 < n ∧ sum_of_digits (n * sum_of_digits n) = 3 :=
by {
  sorry
}

end no_such_positive_integer_exists_l707_707942


namespace math_team_combinations_l707_707090

def numGirls : ℕ := 4
def numBoys : ℕ := 7
def girlsToChoose : ℕ := 2
def boysToChoose : ℕ := 3

def comb (n k : ℕ) : ℕ := n.choose k

theorem math_team_combinations : 
  comb numGirls girlsToChoose * comb numBoys boysToChoose = 210 := 
by
  sorry

end math_team_combinations_l707_707090


namespace proof_problem_l707_707009

variables {A B C O : Type*} [euclidean_geometry A B C O]

-- Definition of the triangle sides and semi-perimeter
variable (a b c p : ℝ)
variable [order p a b c]

-- Orthogonal projections of point inside triangle
variable (AO BO CO : ℝ)
variable (angle_BOC angle_AOC angle_AOB : ℝ)
variable {sin : ℝ → ℝ} -- sine function

-- Assuming sine function properties and bounds
axiom sin_nonneg {x : ℝ} (h : 0 ≤ x) : 0 ≤ sin x

-- Given conditions
axiom point_o_inside_triangle (h1 : 0 ≤ AO)
                             (h2 : 0 ≤ BO)
                             (h3 : 0 ≤ CO)
                             (h4 : 0 ≤ sin angle_BOC)
                             (h5 : 0 ≤ sin angle_AOC)
                             (h6 : 0 ≤ sin angle_AOB)
                             (h7 : AO ≤ p)
                             (h8 : BO ≤ p)
                             (h9 : CO ≤ p)
                             (h10 : p = (a + b + c) / 2)

-- Proof of the final inequality
theorem proof_problem : AO * sin angle_BOC + BO * sin angle_AOC + CO * sin angle_AOB ≤ p := 
by
  sorry

end proof_problem_l707_707009


namespace f_zero_f_odd_range_of_k_l707_707927

variable {f : ℝ → ℝ}
variable (k : ℝ) 

-- Conditions
axiom f_mono : Monotone f
axiom f_at_3 : f 3 = Real.log 3 / Real.log 2
axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y

-- Questions translated to proof goals

-- 1. Prove that f(0) = 0
theorem f_zero : f 0 = 0 :=
sorry

-- 2. Prove that f(x) is an odd function
theorem f_odd (x : ℝ) : f (-x) = -f x :=
sorry

-- 3. Prove that if ∀ x, f(k * 3^x) + f(3^x - 9^x) < 0, then k ∈ (-∞, -1]
theorem range_of_k (h : ∀ x : ℝ, f(k * 3^x) + f(3^x - 9^x) < 0) : k ≤ -1 :=
sorry

end f_zero_f_odd_range_of_k_l707_707927


namespace range_of_f_l707_707994

noncomputable def f (x : ℝ) : ℝ :=
  3 * Real.cos x - 4 * Real.sin x

theorem range_of_f :
  (∀ x : ℝ, x ∈ Set.Icc 0 Real.pi → f x ∈ Set.Icc (-5) 3) ∧
  (∀ y : ℝ, y ∈ Set.Icc (-5) 3 → ∃ x : ℝ, x ∈ Set.Icc 0 Real.pi ∧ f x = y) :=
by
  sorry

end range_of_f_l707_707994


namespace find_x_l707_707587

variables (t x : ℕ)

theorem find_x (h1 : 0 < t) (h2 : t = 4) (h3 : ((9 / 10 : ℚ) * (t * x : ℚ)) - 6 = 48) : x = 15 :=
by
  sorry

end find_x_l707_707587


namespace angle_between_combinations_l707_707279

-- Assume a and b are vectors in some inner product space (like ℝ^n)
variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Given conditions
def unit_vector (v : V) : Prop := ∥v∥ = 1

def between_angle (a b : V) (θ : ℝ) : Prop := real_inner a b = ∥a∥ * ∥b∥ * real.cos θ

-- Proof problem
theorem angle_between_combinations (h1 : unit_vector a) (h2 : unit_vector b) (h3 : between_angle a b (real.pi / 3)) :
  real.angle (2 • a + b) (-3 • a + 2 • b) = 2 * real.pi / 3 :=
sorry

end angle_between_combinations_l707_707279


namespace problem_statement_l707_707330

noncomputable def C : ℝ := 49
noncomputable def D : ℝ := 3.75

theorem problem_statement : C + D = 52.75 := by
  sorry

end problem_statement_l707_707330


namespace median_of_81_consecutive_integers_sum_3_pow_8_l707_707099

theorem median_of_81_consecutive_integers_sum_3_pow_8 :
  ∃ n : ℤ, (sum (list.range 81).map (λ x, x + n) = 3^8) ∧ (list.range 81).nth 40 = some n := sorry

end median_of_81_consecutive_integers_sum_3_pow_8_l707_707099


namespace no_calls_days_2020_l707_707047

theorem no_calls_days_2020 :
  let total_days := 366
  let calls := [4, 5, 6, 8]
  let lcm_of_list (l : List ℕ) : ℕ := l.foldl Nat.lcm 1
  let floor_div (n m : ℕ) := n / m
  let inclusion_exclusion (total : ℕ) (calls : List ℕ) : ℕ :=
    let single := calls.map (floor_div total)
    let pairs := calls.combinations 2 |>.map (λ l, floor_div total (lcm_of_list l))
    let triples := calls.combinations 3 |>.map (λ l, floor_div total (lcm_of_list l))
    let quads := calls.combinations 4 |>.map (λ l, floor_div total (lcm_of_list l))
    single.sum - pairs.sum + triples.sum - quads.sum
  inclusion_exclusion total_days calls = 216 →
  (total_days - 216) = 150 :=
begin
  intros h_inc_exc,
  rw ← h_inc_exc,
  exact (366 - 216),
end

end no_calls_days_2020_l707_707047


namespace perimeter_of_square_D_l707_707421

-- Definitions based on the conditions in the problem
def square (s : ℝ) := s * s

def perimeter (s : ℝ) := 4 * s

-- Given conditions
def perimeter_C : ℝ := 40
def side_length_C : ℝ := perimeter_C / 4
def area_C : ℝ := square side_length_C
def area_D : ℝ := area_C / 3
def side_length_D : ℝ := real.sqrt area_D

-- Proof statement to be proved
theorem perimeter_of_square_D : perimeter side_length_D = (40 * real.sqrt 3) / 3 := by
  sorry

end perimeter_of_square_D_l707_707421


namespace perimeter_of_square_D_l707_707423

-- Definitions based on the conditions in the problem
def square (s : ℝ) := s * s

def perimeter (s : ℝ) := 4 * s

-- Given conditions
def perimeter_C : ℝ := 40
def side_length_C : ℝ := perimeter_C / 4
def area_C : ℝ := square side_length_C
def area_D : ℝ := area_C / 3
def side_length_D : ℝ := real.sqrt area_D

-- Proof statement to be proved
theorem perimeter_of_square_D : perimeter side_length_D = (40 * real.sqrt 3) / 3 := by
  sorry

end perimeter_of_square_D_l707_707423


namespace find_DP_l707_707115

theorem find_DP (AP BP CP DP : ℚ) (h1 : AP = 4) (h2 : BP = 6) (h3 : CP = 9) (h4 : AP * BP = CP * DP) :
  DP = 8 / 3 :=
by
  rw [h1, h2, h3] at h4
  sorry

end find_DP_l707_707115


namespace frac_subtraction_l707_707476

-- Define the sums of numerators and denominators as conditions
def sum_num1 : ℕ := 2 + 4 + 6 + 8
def sum_den1 : ℕ := 1 + 3 + 5 + 7
def sum_num2 : ℕ := 1 + 3 + 5 + 7
def sum_den2 : ℕ := 2 + 4 + 6 + 8

-- Prove that the given mathematical expression equals to 9/20
theorem frac_subtraction : 
  sum_num1 / sum_den1 - sum_num2 / sum_den2 = 9 / 20 := 
  by 
    have h1 : sum_num1 = 20 := by norm_num
    have h2 : sum_den1 = 16 := by norm_num
    have h3 : sum_num2 = 16 := by norm_num
    have h4 : sum_den2 = 20 := by norm_num
    rw [h1, h2, h3, h4]
    norm_num

end frac_subtraction_l707_707476


namespace angle_of_incline_l707_707776

-- Define the given line equation and the condition based on it
def line_equation := λ x : ℝ, sqrt 3 * x

-- Proof statement: Angle of incline of the given line is 60 degrees
theorem angle_of_incline : ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 180 ∧ tan θ = sqrt 3 ∧ θ = 60 :=
by
  sorry

end angle_of_incline_l707_707776


namespace equilibrium_condition_l707_707656

-- Define the forces as constants
def f1 := (-2, -1 : ℝ × ℝ)
def f2 := (-3, 2 : ℝ × ℝ)
def f3 := (4, -3 : ℝ × ℝ)

-- Definition for the required force f4 that keeps the system in equilibrium
def required_f4 := (1, 2 : ℝ × ℝ)

-- Define the equilibrium condition theorem
theorem equilibrium_condition : f1 + f2 + f3 + required_f4 = (0, 0 : ℝ × ℝ) :=
by
  -- The proof would go here, but it is not needed for this task.
  sorry

end equilibrium_condition_l707_707656


namespace find_fourth_vertex_l707_707466

def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def area_of_quadrilateral (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : ℝ :=
  area_of_triangle x1 y1 x2 y2 x3 y3 + area_of_triangle x2 y2 x3 y3 x4 y4

def is_positive_branch_of_y_axis (x y : ℝ) : Prop :=
  x = 0 ∧ y > 0

theorem find_fourth_vertex :
  ∃ y : ℝ, is_positive_branch_of_y_axis 0 y ∧ area_of_quadrilateral 6 4 0 0 (-15) 0 0 y = 60 ∧ y = 4 :=
by
  sorry

end find_fourth_vertex_l707_707466


namespace trig_identity_l707_707998

theorem trig_identity (α : ℝ) (h : sin α = 2 * cos α) : (cos (2 * α)) / (sin α - cos α) ^ 2 = -3 := 
by
  sorry

end trig_identity_l707_707998


namespace identify_incorrect_algorithm_statement_l707_707557

-- Define the conditions of the problem
def steps_of_algorithm_must_be_finite : Prop := 
  True  -- Placeholder for the condition "The steps of an algorithm must be finite"

def algorithm_not_unique : Prop := 
  True  -- Placeholder for the condition "The algorithm for solving a certain problem is not necessarily unique"

def steps_clear_instructions : Prop := 
  True  -- Placeholder for the condition "Each step of the algorithm must have clear instructions"

def algorithm_must_produce_definite_result : Prop := 
  True  -- Placeholder for the condition "The execution of an algorithm must produce a definite result"

-- The proof problem statement
theorem identify_incorrect_algorithm_statement :
  steps_of_algorithm_must_be_finite ->
  algorithm_not_unique ->
  steps_clear_instructions ->
  algorithm_must_produce_definite_result ->
  (B : Prop) :=
by
  intros h1 h2 h3 h4
  exact algorithm_not_unique

end identify_incorrect_algorithm_statement_l707_707557


namespace term_omit_perfect_squares_300_l707_707481

theorem term_omit_perfect_squares_300 (n : ℕ) (hn : n = 300) : 
  ∃ k : ℕ, k = 317 ∧ (∀ m : ℕ, (m < k → m * m ≠ k)) := 
sorry

end term_omit_perfect_squares_300_l707_707481


namespace part1_part2_l707_707634

noncomputable def A (a : ℝ) := { x : ℝ | x^2 - a * x + a^2 - 19 = 0 }
def B := { x : ℝ | x^2 - 5 * x + 6 = 0 }
def C := { x : ℝ | x^2 + 2 * x - 8 = 0 }

-- Proof Problem 1: Prove that if A ∩ B ≠ ∅ and A ∩ C = ∅, then a = -2
theorem part1 (a : ℝ) (h1 : (A a ∩ B) ≠ ∅) (h2 : (A a ∩ C) = ∅) : a = -2 :=
sorry

-- Proof Problem 2: Prove that if A ∩ B = A ∩ C ≠ ∅, then a = -3
theorem part2 (a : ℝ) (h1 : (A a ∩ B = A a ∩ C) ∧ (A a ∩ B) ≠ ∅) : a = -3 :=
sorry

end part1_part2_l707_707634


namespace raja_second_half_speed_l707_707405

noncomputable def speed_of_second_half (total_dist : ℝ) (total_time : ℝ) (first_half_speed : ℝ) : ℝ :=
let half_dist := total_dist / 2
let first_half_time := half_dist / first_half_speed
let second_half_time := total_time - first_half_time
half_dist / second_half_time

theorem raja_second_half_speed :
  speed_of_second_half 225 10 21 ≈ 24.23 :=
by
  have total_dist := 225
  have total_time := 10
  have first_half_speed := 21
  let half_dist := total_dist / 2
  let first_half_time := half_dist / first_half_speed
  let second_half_time := total_time - first_half_time
  let second_half_speed := half_dist / second_half_time
  show second_half_speed ≈ 24.23, from
    sorry

end raja_second_half_speed_l707_707405


namespace largest_three_digit_n_l707_707829

-- Define the conditions and the proof statement
theorem largest_three_digit_n (n : ℕ) : 
  (70 * n ≡ 210 [MOD 350]) ∧ (n ≤ 999) ∧ (n ≥ 100) → n = 998 :=
begin
  -- Sorry as a placeholder for the proof
  sorry,
end

end largest_three_digit_n_l707_707829


namespace simplify_expression_l707_707768

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (x⁻¹ - x + 2) = (1 - (x - 1)^2) / x := 
sorry

end simplify_expression_l707_707768


namespace find_k_l707_707003

theorem find_k
  (angle_C : ℝ)
  (AB : ℝ × ℝ)
  (AC : ℝ × ℝ)
  (h1 : angle_C = 90)
  (h2 : AB = (k, 1))
  (h3 : AC = (2, 3)) :
  k = 5 := by
  sorry

end find_k_l707_707003


namespace find_w_l707_707926

open Matrix

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ :=
  !![![1, 0], ![0, -1]]

def w : Fin 2 → ℝ :=
  !![3/2, -3/2]

theorem find_w :
  let I := (1 : Matrix (Fin 2) (Fin 2) ℝ)
  (B^6 + B^4 + B^2 + I) * col w = col !![6, -6] :=
by
  sorry

end find_w_l707_707926


namespace samuel_discontinued_coaching_on_november_3rd_l707_707765

-- Definitions used directly appearing in conditions
def total_amount_to_pay : ℕ := 7038
def daily_coaching_charges : ℕ := 23
def year_days : ℕ := 365
def months_with_days : list (string × ℕ) :=
  [("January", 31), ("February", 28), ("March", 31), ("April", 30), ("May", 31), ("June", 30), 
   ("July", 31), ("August", 31), ("September", 30), ("October", 31), ("November", 30), ("December", 31)]

-- Theorem stating the equivalence
theorem samuel_discontinued_coaching_on_november_3rd : 
  ∃ (day : ℕ) (month : string), (total_amount_to_pay / daily_coaching_charges = 306) ∧
                                (day = 3) ∧ (month = "November") := 
by
  -- The proof is omitted as per instructions
  sorry

end samuel_discontinued_coaching_on_november_3rd_l707_707765


namespace unbounded_k_n_l707_707717

noncomputable def f : ℕ → ℕ := sorry
noncomputable def f_pow (f : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, n   => n
| (m+1), n => f (f_pow m n)

lemma exists_k (n : ℕ) : ∃ k : ℕ, f_pow f (2 * k) n = n + k := sorry

def k_n (n : ℕ) : ℕ := Nat.find (exists_k n)

theorem unbounded_k_n : ∀ N : ℕ, ∃ n : ℕ, k_n n > N :=
by
  sorry

end unbounded_k_n_l707_707717


namespace no_prism_with_diagonals_5_12_13_14_l707_707132

-- Define the problem statement
theorem no_prism_with_diagonals_5_12_13_14 :
  ¬ ∃ (a b c : ℝ),
    let d₁ := real.sqrt (a^2 + b^2),
        d₂ := real.sqrt (b^2 + c^2),
        d₃ := real.sqrt (a^2 + c^2),
        d₄ := real.sqrt (a^2 + b^2 + c^2) in
    {d₁, d₂, d₃, d₄} = {5, 12, 13, 14} := 
  sorry

end no_prism_with_diagonals_5_12_13_14_l707_707132


namespace max_product_of_three_distinct_numbers_l707_707019

-- Define the set of numbers
def num_set : Set ℤ := {-6, -4, -2, 0, 1, 3, 5, 7}

-- Define the condition that the numbers are distinct and come from the set
structure distinct_numbers (a b c : ℤ) :=
  (ha : a ∈ num_set)
  (hb : b ∈ num_set)
  (hc : c ∈ num_set)
  (hab : a ≠ b)
  (hac : a ≠ c)
  (hbc : b ≠ c)

-- Defining the mathematical proof problem
theorem max_product_of_three_distinct_numbers :
  ∃ a b c : ℤ, distinct_numbers a b c ∧ a * b * c = 168 :=
sorry

end max_product_of_three_distinct_numbers_l707_707019


namespace find_C_l707_707905

variable (A B C : ℕ)

theorem find_C (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 310) : 
  C = 10 := 
by
  sorry

end find_C_l707_707905


namespace dot_path_length_l707_707880

-- Definition of the cube and initial conditions
def side_length : ℝ := 2
def dot_position_from_edge : ℝ := side_length / 3
def distance_MD : ℝ := real.sqrt ((dot_position_from_edge)^2 + (side_length)^2)

-- State the theorem to prove the path length of the dot
theorem dot_path_length :
  let arc_length := (1/4) * 2 * real.pi * distance_MD in
  let total_path := 4 * arc_length in
  total_path = (4 * real.pi * real.sqrt 10) / 3 :=
sorry

end dot_path_length_l707_707880


namespace coefficient_of_x6_in_expansion_l707_707503

theorem coefficient_of_x6_in_expansion :
  let a := 3 * x
  let b := 2
  let n := 8
  ∀ x : ℝ, 
  (coeff_of_x6 : ℕ) := 
  ∃ k : ℕ, 
  k = n - 6 ∧ 
  ((Nat.choose n k) * (a ^ (n - k)) * (b ^ k) = coeff_of_x6 * x^6) ∧ 
  coeff_of_x6 = 82272
:= 
by
  sorry

end coefficient_of_x6_in_expansion_l707_707503


namespace problem_statement_l707_707614

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β)

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
    (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h4 : f 2015 a α b β = -1) : 
    f 2016 a α b β = 1 :=
by
  -- Proof omitted (sorry)
  sorry

end problem_statement_l707_707614


namespace exists_arc_with_given_counts_l707_707161

-- Define the basic structure and conditions
def arc (L : Nat) := List (Fin 2) -- List of digits (either 0 or 1)

def Z (w : arc) : Nat := w.count (fun d => d = 0)
def N (w : arc) : Nat := w.count (fun d => d = 1)

-- Main theorem to be proved
theorem exists_arc_with_given_counts
  (w_collection : List (arc)) -- finite collection of arcs
  (h1 : ∀ (w w' : arc), w.length = w'.length → |Z(w) - Z(w')| ≤ 1)
  (w_parts : List (arc))
  (Z_avg : Nat)
  (N_avg : Nat)
  (h2 : Z_avg = (1 / w_parts.length.to_nat) * (w_parts.map Z).sum)
  (h3 : N_avg = (1 / w_parts.length.to_nat) * (w_parts.map N).sum)
  (h4 : Z_avg ∈ ℤ)
  (h5 : N_avg ∈ ℤ)
  : ∃ (w : arc), Z(w) = Z_avg ∧ N(w) = N_avg :=
by
  sorry

end exists_arc_with_given_counts_l707_707161


namespace non_mundane_primes_l707_707894

-- Definition of a prime number
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Definition of a mundane prime number
def is_mundane (p : ℕ) : Prop :=
  ∃ (a b : ℕ), 0 < a ∧ a < p / 2 ∧ 0 < b ∧ b < p / 2 ∧ (ab - 1) % p = 0

-- Main theorem statement
theorem non_mundane_primes :
  {p : ℕ | is_prime p ∧ ¬ is_mundane p} = {2, 3, 5, 7, 13} :=
sorry

end non_mundane_primes_l707_707894


namespace missing_age_not_6_l707_707389

theorem missing_age_not_6 (ages : Finset ℕ) :
  (∀ age ∈ ages, age ∈ ({2, 3, 4, 5, 7, 8, 9, 10, 11} : Finset ℕ)) ∧
  5 ∈ ages ∧
  (3339 % 2 = 0) ∧
  (3339 % 3 = 0) ∧
  (3339 % 4 = 0) ∧
  (3339 % 5 = 0) ∧
  (3339 % 7 = 0) ∧
  (3339 % 8 = 0) ∧
  (3339 % 9 = 0) ∧
  (3339 % 10 = 0) ∧
  (3339 % 11 = 0) →
  ¬(6 ∈ ages) :=
by {
  intros h,
  rcases h with ⟨h1, h5, h2, h3, h4, h5, h7, h8, h9, h10, h11⟩,
  sorry
}

end missing_age_not_6_l707_707389


namespace base_of_parallelogram_l707_707959

theorem base_of_parallelogram (A h b : ℝ) (hₐ : A = 96) (hₕ : h = 8) : b = A / h := by
  have hb : b = 96 / 8 := by
    rw [hₐ, hₕ]
  exact hb

end base_of_parallelogram_l707_707959


namespace g_1000_eq_9_div_2_l707_707369

-- Define the function g with the given properties
variable {g : ℝ → ℝ}

-- The function g is such that for all positive real numbers x and y, g(xy) = g(x) / y
def g_property (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : Prop := g(x * y) = g(x) / y

-- Function value of g at 900 is given
def g_900_eq_5 : Prop := g 900 = 5

-- Prove that g(1000) = 9 / 2
theorem g_1000_eq_9_div_2 (h : ∀ x y : ℝ, 0 < x → 0 < y → g_property x y) (h900 : g_900_eq_5) : g 1000 = 9 / 2 :=
by
  sorry

end g_1000_eq_9_div_2_l707_707369


namespace unit_prices_max_toys_l707_707533

-- For question 1
theorem unit_prices (x y : ℕ)
  (h₁ : y = x + 25)
  (h₂ : 2*y + x = 200) : x = 50 ∧ y = 75 :=
by {
  sorry
}

-- For question 2
theorem max_toys (cost_a cost_b q_a q_b : ℕ)
  (h₁ : cost_a = 50)
  (h₂ : cost_b = 75)
  (h₃ : q_b = 2 * q_a)
  (h₄ : 50 * q_a + 75 * q_b ≤ 20000) : q_a ≤ 100 :=
by {
  sorry
}

end unit_prices_max_toys_l707_707533


namespace count_odd_tens_digit_squares_l707_707909

def is_tens_digit_odd (n : ℕ) : Prop :=
  (n / 10) % 2 = 1

def numbers_with_odd_tens_digit : ℕ :=
  ((list.range 96).filter (λ n, is_tens_digit_odd (n ^ 2))).length

theorem count_odd_tens_digit_squares : 
  numbers_with_odd_tens_digit = 19 := 
sorry

end count_odd_tens_digit_squares_l707_707909


namespace find_q_l707_707281

noncomputable def q_value (m q : ℕ) : Prop := 
  ((1 ^ m) / (5 ^ m)) * ((1 ^ 16) / (4 ^ 16)) = 1 / (q * 10 ^ 31)

theorem find_q (m : ℕ) (q : ℕ) (h1 : m = 31) (h2 : q_value m q) : q = 2 :=
by
  sorry

end find_q_l707_707281


namespace perimeter_of_square_D_l707_707422

-- Definitions based on the conditions in the problem
def square (s : ℝ) := s * s

def perimeter (s : ℝ) := 4 * s

-- Given conditions
def perimeter_C : ℝ := 40
def side_length_C : ℝ := perimeter_C / 4
def area_C : ℝ := square side_length_C
def area_D : ℝ := area_C / 3
def side_length_D : ℝ := real.sqrt area_D

-- Proof statement to be proved
theorem perimeter_of_square_D : perimeter side_length_D = (40 * real.sqrt 3) / 3 := by
  sorry

end perimeter_of_square_D_l707_707422


namespace initial_pigs_is_64_l707_707461

-- Define the initial number of pigs, joined pigs, and total pigs
def initial_pigs : ℕ := x
def joined_pigs : ℕ := 22
def total_pigs : ℕ := 86

-- Lean statement to prove that given the conditions, the initial number of pigs is 64
theorem initial_pigs_is_64 : initial_pigs + joined_pigs = total_pigs → initial_pigs = 64 :=
by {
  intro h,
  sorry
}

end initial_pigs_is_64_l707_707461


namespace probability_two_dice_sum_seven_l707_707798

theorem probability_two_dice_sum_seven (z : ℕ) (w : ℚ) (h : z = 2) : w = 1 / 6 :=
by sorry

end probability_two_dice_sum_seven_l707_707798


namespace hyperbola_equation_l707_707753

theorem hyperbola_equation 
  (vertex : ℝ × ℝ) 
  (asymptote_slope : ℝ) 
  (h_vertex : vertex = (2, 0))
  (h_asymptote : asymptote_slope = Real.sqrt 2) : 
  (∀ x y : ℝ, x^2 / 4 - y^2 / 8 = 1) := 
by
    sorry

end hyperbola_equation_l707_707753


namespace gain_percentage_proof_l707_707904

-- Define the given conditions as variables and assumptions
variable (CP : ℝ) (SP : ℝ) (NewSP : ℝ)
variable (LossPercentage : ℝ) (SaleIncrement : ℝ)

-- Initial conditions
def CP := 1400
def LossPercentage := 10
def SP := CP - (LossPercentage / 100 * CP)
def SaleIncrement := 196
def NewSP := SP + SaleIncrement

-- Final condition
theorem gain_percentage_proof :
  ((NewSP - CP) / CP * 100) = 4 :=
by
  sorry

end gain_percentage_proof_l707_707904


namespace ratio_of_segments_circumcircle_orthogonality_l707_707022

variables {α : Type*} [euclidean_geometry α]

-- Given four points A, B, C, D in the plane
variables {A B C D : α}

-- Given conditions on the points
variables (h1 : same_side A B C D)
variables (h2 : distance A C * distance B D = distance A D * distance B C)
variables (h3 : angle A D B = 90 + angle A C B)

-- 1. Prove the ratio of segments
theorem ratio_of_segments (h1 : same_side A B C D) 
  (h2 : distance A C * distance B D = distance A D * distance B C)
  (h3 : angle A D B = 90 + angle A C B) :
  distance A B * distance C D / (distance A C * distance B D) = sqrt 2 :=
  sorry

-- 2. Prove the orthogonality of circumcircles
theorem circumcircle_orthogonality (h1 : same_side A B C D) 
  (h2 : distance A C * distance B D = distance A D * distance B C) 
  (h3 : angle A D B = 90 + angle A C B) :
  orthogonal_circumcircles A C D B C D :=
  sorry

end ratio_of_segments_circumcircle_orthogonality_l707_707022


namespace trajectory_equation_l707_707630

-- Define the fixed points F1 and F2
structure Point where
  x : ℝ
  y : ℝ

def F1 : Point := ⟨-2, 0⟩
def F2 : Point := ⟨2, 0⟩

-- Define the moving point M and the condition it must satisfy
def satisfies_condition (M : Point) : Prop :=
  (Real.sqrt ((M.x + 2)^2 + M.y^2) - Real.sqrt ((M.x - 2)^2 + M.y^2)) = 4

-- The trajectory of the point M must satisfy y = 0 and x >= 2
def on_trajectory (M : Point) : Prop :=
  M.y = 0 ∧ M.x ≥ 2

-- The final theorem to be proved
theorem trajectory_equation (M : Point) (h : satisfies_condition M) : on_trajectory M := by
  sorry

end trajectory_equation_l707_707630


namespace solve_for_a_l707_707930

def F (a b c : ℝ) : ℝ := a * b ^ 2 + c

theorem solve_for_a : F a 3 8 = F a 5 10 → a = -(1/8) :=
by 
  intro h
  have h₁ : F a 3 8 = 9 * a + 8 := by rw [F]; simp
  have h₂ : F a 5 10 = 25 * a + 10 := by rw [F]; simp
  rw [h₁, h₂] at h
  linarith

end solve_for_a_l707_707930


namespace matrix_inverse_self_l707_707973

-- Define a theorem that proves the values of c and d
theorem matrix_inverse_self (c d : ℚ) : 
  (∀ (M : Matrix (Fin 2) (Fin 2) ℚ), M = ![![4, -2], ![c, d]] → M * M = 1) →
  (c = 15 / 2 ∧ d = -4) :=
by
  intros h M hM
  sorry

end matrix_inverse_self_l707_707973


namespace divide_three_squares_into_four_parts_l707_707582

theorem divide_three_squares_into_four_parts (A : ℝ) (hA : 0 < A) :
  ∃ (parts : list ℝ), list.length parts = 4 ∧ (∀ p ∈ parts, p = (3 * A) / 4) ∧ (list.sum parts = 3 * A) :=
by
  sorry

end divide_three_squares_into_four_parts_l707_707582


namespace evaluate_polynomial_at_2_l707_707218

def polynomial (x : ℝ) := x^2 + 5*x - 14

theorem evaluate_polynomial_at_2 : polynomial 2 = 0 := by
  sorry

end evaluate_polynomial_at_2_l707_707218


namespace mean_of_solutions_l707_707232

theorem mean_of_solutions (x : ℝ) :
  (x^3 - 4*x^2 - 7*x = 0) → (x + 2 + sqrt(11) + x + 2 - sqrt(11)) / 3 = 4/3 :=
sorry

end mean_of_solutions_l707_707232


namespace initial_volume_of_solution_l707_707532

variable (V : ℝ)

theorem initial_volume_of_solution (alcohol_initial : V * 0.05) (total_alcohol_added : 3.5) (total_volume_added : 3.5 + 6.5)
      (alcohol_new_percent : (alcohol_initial + total_alcohol_added) = 0.11 * (V + total_volume_added)) :
  V = 40 := 
by 
  sorry

end initial_volume_of_solution_l707_707532


namespace thm_300th_term_non_square_seq_l707_707483

theorem thm_300th_term_non_square_seq : 
  let non_square_seq (n : ℕ) := { k : ℕ // k > 0 ∧ ∀ m : ℕ, m * m ≠ k } in
  (non_square_seq 300).val = 318 :=
by
  sorry

end thm_300th_term_non_square_seq_l707_707483


namespace non_square_300th_term_l707_707499

theorem non_square_300th_term (N : ℕ) (hN : N = 300) : 
  ∃ x : ℕ, (∀ (t : ℕ), 0 < t ∧ t ≤ x → ¬ (∃ k : ℕ, t = k^2)) ∧ 
           (∑ t in finset.range (N + 17), if ∃ k : ℕ, t = k^2 then 0 else 1 = N) ∧ 
           x = 317 := by
  sorry

end non_square_300th_term_l707_707499


namespace problem1_l707_707528

theorem problem1 (α β : ℝ) 
  (tan_sum : Real.tan (α + β) = 2 / 5) 
  (tan_diff : Real.tan (β - Real.pi / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 := 
sorry

end problem1_l707_707528


namespace count_integers_in_abs_leq_six_l707_707665

theorem count_integers_in_abs_leq_six : 
  {x : ℤ | abs (x - 3) ≤ 6}.card = 13 := 
by
  sorry

end count_integers_in_abs_leq_six_l707_707665


namespace mass_percentage_of_Na_in_NaClO_l707_707970

theorem mass_percentage_of_Na_in_NaClO:
  let M_Na: ℝ := 22.99
  let M_Cl: ℝ := 35.45
  let M_O: ℝ := 16.00
  let M_NaClO: ℝ := M_Na + M_Cl + M_O
  let mass_percentage_Na: ℝ := (M_Na / M_NaClO) * 100
  mass_percentage_Na = 30.89 :=
by
  -- Define the molar masses
  let M_Na := 22.99
  let M_Cl := 35.45
  let M_O := 16.00
  -- Calculate the molar mass of NaClO
  let M_NaClO := M_Na + M_Cl + M_O
  -- Calculate the mass percentage of Na in NaClO
  let mass_percentage_Na := (M_Na / M_NaClO) * 100
  show mass_percentage_Na = 30.89, from
  sorry

end mass_percentage_of_Na_in_NaClO_l707_707970


namespace point_outside_circle_l707_707444

noncomputable def P (a : ℝ) : ℝ × ℝ := (a, 10) -- Definition of point P

def circle_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 1) ^ 2 = 2 -- Definition of circle equation

theorem point_outside_circle (a : ℝ) :
  let x := a in
  let y := 10 in
  (x - 1)^2 + (y - 1)^2 > 2 := by
  sorry

end point_outside_circle_l707_707444


namespace carrie_bought_tshirts_l707_707571

variable (cost_per_tshirt : ℝ) (total_spent : ℝ)

theorem carrie_bought_tshirts (h1 : cost_per_tshirt = 9.95) (h2 : total_spent = 248) :
  ⌊total_spent / cost_per_tshirt⌋ = 24 :=
by
  sorry

end carrie_bought_tshirts_l707_707571


namespace yellow_yellow_pairs_count_l707_707566

-- Definitions for the conditions
def students_total := 144
def pairs_total := 72
def blue_students := 60
def yellow_students := 84
def blue_blue_pairs := 28

-- The main conjecture to prove
theorem yellow_yellow_pairs_count :
  let blue_blue_students := 2 * blue_blue_pairs in 
  let blue_mixed_students := blue_students - blue_blue_students in 
  let yellow_mixed_students := blue_mixed_students in 
  let yellow_yellow_students := yellow_students - yellow_mixed_students in 
  let yellow_yellow_pairs := yellow_yellow_students / 2 in 
  yellow_yellow_pairs = 40 := by
  sorry

end yellow_yellow_pairs_count_l707_707566


namespace Suzanne_runs_5_kilometers_l707_707080

theorem Suzanne_runs_5_kilometers 
  (a : ℕ) 
  (r : ℕ) 
  (total_donation : ℕ) 
  (n : ℕ)
  (h1 : a = 10) 
  (h2 : r = 2) 
  (h3 : total_donation = 310) 
  (h4 : total_donation = a * (1 - r^n) / (1 - r)) 
  : n = 5 :=
by
  sorry

end Suzanne_runs_5_kilometers_l707_707080


namespace unique_f_fixed_point_condition_fixed_point_expression_l707_707079

section china_nat_team_exam

variables (a b c : ℕ) (h1 : a < b) (h2 : b < c)

def f : ℕ → ℕ
| n := if n > c then n - a else f (f (n + b))

-- Statement 1: Prove the uniqueness of the function f
theorem unique_f : ∀ n : ℕ, ∃! y : ℕ, y = f n := 
sorry

-- Statement 2: Prove the necessary and sufficient condition for f to have a fixed point
theorem fixed_point_condition : (b - a) ∣ a ↔ ∃ n : ℕ, f n = n :=
sorry

-- Statement 3: Express such a fixed point in terms of a, b, and c
theorem fixed_point_expression : (b - a) ∣ a → ∃ n : ℕ, n = c - ((c - a) / (b - a)) * (b - a) :=
sorry

end china_nat_team_exam

end unique_f_fixed_point_condition_fixed_point_expression_l707_707079


namespace increasing_power_function_l707_707782

theorem increasing_power_function (m : ℝ) (h_power : m^2 - 1 = 1)
    (h_increasing : ∀ x : ℝ, x > 0 → (m^2 - 1) * m * x^(m-1) > 0) : m = Real.sqrt 2 :=
by
  sorry

end increasing_power_function_l707_707782


namespace number_of_people_on_boats_l707_707107

theorem number_of_people_on_boats : 
    (∀ (first_boats : ℕ) (first_people_per_boat : ℕ) (remaining_boats : ℕ) (remaining_people_per_boat : ℕ),
        first_boats = 4 ∧ first_people_per_boat = 4 ∧ remaining_boats = 3 ∧ remaining_people_per_boat = 5 →
        first_boats * first_people_per_boat + remaining_boats * remaining_people_per_boat = 31) :=
by
    intros first_boats first_people_per_boat remaining_boats remaining_people_per_boat h
    cases h with h1 h2
    -- Destructuring the conditions
    cases h2 with h3 h4
    -- Verifying the conditions
    cases h4 with h5 h6
    -- Now use the given values to prove the conclusion
    have h_first : first_boats * first_people_per_boat = 4 * 4 := by rw [h1, h3]
    have h_remaining : remaining_boats * remaining_people_per_boat = 3 * 5 := by rw [h5, h6]
    rw [h_first, h_remaining]
    trivial
    sorry -- filler to avoid actual proof until the final step
    rfl -- reflexivity of equality makes this actually trivial


end number_of_people_on_boats_l707_707107


namespace bird_count_l707_707891

theorem bird_count (total_animals : ℕ) (kittens : ℕ) (hamsters : ℕ) 
    (h_total : total_animals = 77)
    (h_kittens : kittens = 32)
    (h_hamsters : hamsters = 15) : 
    total_animals - kittens - hamsters = 30 := by
  rw [h_total, h_kittens, h_hamsters]
  norm_num

end bird_count_l707_707891


namespace joe_lists_count_l707_707529

theorem joe_lists_count : ∃ (n : ℕ), n = 15 * 14 := sorry

end joe_lists_count_l707_707529


namespace smallest_z_minus_x_l707_707109

theorem smallest_z_minus_x (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = nat.factorial 10) (hxy : x < y) (hyz : y < z) :
  z - x = 2139 :=
sorry

end smallest_z_minus_x_l707_707109


namespace beetle_speed_correct_l707_707559

-- Define the conditions
def ant_distance_meters : ℝ := 1000
def ant_time_minutes : ℝ := 30
def beetle_distance_percentage : ℝ := 0.9

-- Conversion factors
def minutes_to_hours : ℝ := 60 / ant_time_minutes
def meters_to_kilometers : ℝ := 0.001

-- Beetle's speed in km/h
def beetle_speed_kmh : ℝ := 
  (ant_distance_meters * beetle_distance_percentage * meters_to_kilometers) * minutes_to_hours

theorem beetle_speed_correct : beetle_speed_kmh = 1.8 :=
by
  unfold beetle_speed_kmh
  unfold ant_distance_meters ant_time_minutes beetle_distance_percentage minutes_to_hours meters_to_kilometers
  norm_num
  sorry

end beetle_speed_correct_l707_707559


namespace systematic_sampling_method_l707_707554

-- Define conditions of the problem
noncomputable def test_numbers_end_in_5 (n : Nat) : Prop :=
  n % 10 = 5

noncomputable def uniform_interval_of_10 (nums : List Nat) : Prop :=
  ∀ (i j : Nat), i < j → j < nums.length → nums[i] % 10 = 5 → nums[j] % 10 = 5 → nums[j] - nums[i] = 10 * (j - i)

-- Define a theorem to state the problem
theorem systematic_sampling_method (nums : List Nat) :
  (∀ (n : Nat), n ∈ nums → test_numbers_end_in_5 n) →
  uniform_interval_of_10 nums →
  sampling_method nums = "Systematic Sampling" :=
by
  sorry

end systematic_sampling_method_l707_707554


namespace distinct_or_eventually_identical_sequences_l707_707989

theorem distinct_or_eventually_identical_sequences 
  (a b c d : ℝ) :
  ∀ n : ℕ, 
  let a_seq := λ n, nat.rec_on n a (λ n prev, prev * (nat.cases_on (n+1) a b)),
      b_seq := λ n, nat.rec_on n b (λ n prev, prev * (nat.cases_on (n+1) b c)),
      c_seq := λ n, nat.rec_on n c (λ n prev, prev * (nat.cases_on (n+1) c d)),
      d_seq := λ n, nat.rec_on n d (λ n prev, prev * (nat.cases_on (n+1) d a)) in
  (∀ i j, i ≠ j → (a_seq i, b_seq i, c_seq i, d_seq i) ≠ (a_seq j, b_seq j, c_seq j, d_seq j)) ∨
  (∃ k, ∀ m ≥ k, (a_seq m, b_seq m, c_seq m, d_seq m) = (a_seq k, b_seq k, c_seq k, d_seq k)) :=
by sorry

end distinct_or_eventually_identical_sequences_l707_707989


namespace find_n_in_range_l707_707229

theorem find_n_in_range : ∃ n, 5 ≤ n ∧ n ≤ 10 ∧ n ≡ 10543 [MOD 7] ∧ n = 8 := 
by
  sorry

end find_n_in_range_l707_707229


namespace problem1_arithmetic_sequence_problem2_geometric_sequence_l707_707862

-- Problem (1)
variable (S : Nat → Int)
variable (a : Nat → Int)

axiom S10_eq_50 : S 10 = 50
axiom S20_eq_300 : S 20 = 300
axiom S_def : (∀ n : Nat, n > 0 → S n = n * a 1 + (n * (n-1) / 2) * (a 2 - a 1))

theorem problem1_arithmetic_sequence (n : Nat) : a n = 2 * n - 6 := sorry

-- Problem (2)
variable (a : Nat → Int)

axiom S3_eq_a2_plus_10a1 : S 3 = a 2 + 10 * a 1
axiom a5_eq_81 : a 5 = 81
axiom positive_terms : ∀ n, a n > 0

theorem problem2_geometric_sequence (n : Nat) : S n = (3 ^ n - 1) / 2 := sorry

end problem1_arithmetic_sequence_problem2_geometric_sequence_l707_707862


namespace principal_amount_borrowed_l707_707848

theorem principal_amount_borrowed
  (R : ℝ) (T : ℝ) (SI : ℝ) (P : ℝ) 
  (hR : R = 12) 
  (hT : T = 20) 
  (hSI : SI = 2100) 
  (hFormula : SI = (P * R * T) / 100) : 
  P = 875 := 
by 
  -- Assuming the initial steps 
  sorry

end principal_amount_borrowed_l707_707848


namespace area_union_correct_l707_707173

noncomputable def side_length : ℝ := 12
noncomputable def radius : ℝ := 12

def area_square : ℝ := side_length ^ 2
def area_circle : ℝ := π * radius ^ 2
def overlapping_area : ℝ := (1 / 4) * area_circle

def area_union : ℝ := area_square + area_circle - overlapping_area

theorem area_union_correct : area_union = 144 + 108 * π := by
  unfold area_square area_circle overlapping_area area_union
  norm_num
  simp
  sorry

end area_union_correct_l707_707173


namespace probability_of_ace_ten_king_l707_707865

noncomputable def probability_first_ace_second_ten_third_king : ℚ :=
  (4/52) * (4/51) * (4/50)

theorem probability_of_ace_ten_king :
  probability_first_ace_second_ten_third_king = 2/16575 :=
by
  sorry

end probability_of_ace_ten_king_l707_707865


namespace largest_k_for_right_triangle_l707_707597

noncomputable def k : ℝ := (3 * Real.sqrt 2 - 4) / 2

theorem largest_k_for_right_triangle (a b c : ℝ) (h : c^2 = a^2 + b^2) :
    a^3 + b^3 + c^3 ≥ k * (a + b + c)^3 :=
sorry

end largest_k_for_right_triangle_l707_707597


namespace pipe_pumping_rate_l707_707472

theorem pipe_pumping_rate (R : ℕ) (h : 5 * R + 5 * 192 = 1200) : R = 48 := by
  sorry

end pipe_pumping_rate_l707_707472


namespace dutch_americans_with_window_seats_l707_707052

theorem dutch_americans_with_window_seats :
  let total_people := 90
  let dutch_fraction := 3 / 5
  let dutch_american_fraction := 1 / 2
  let window_seat_fraction := 1 / 3
  let dutch_people := total_people * dutch_fraction
  let dutch_americans := dutch_people * dutch_american_fraction
  let dutch_americans_window_seats := dutch_americans * window_seat_fraction
  dutch_americans_window_seats = 9 := by
sorry

end dutch_americans_with_window_seats_l707_707052


namespace real_roots_exist_for_all_real_K_l707_707987

theorem real_roots_exist_for_all_real_K (K : ℝ) : ∃ x : ℝ, x = K^3 * (x-1) * (x-2) * (x-3) :=
by
  sorry

end real_roots_exist_for_all_real_K_l707_707987


namespace point_in_second_quadrant_l707_707262

structure Point where
  x : Int
  y : Int

-- Define point P
def P : Point := { x := -1, y := 2 }

-- Define the second quadrant condition
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

-- The mathematical statement to prove
theorem point_in_second_quadrant : second_quadrant P := by
  sorry

end point_in_second_quadrant_l707_707262


namespace min_total_spent_l707_707802

theorem min_total_spent : 
  ∀(prices : list ℕ), 
  (∀ p, p ∈ prices → p ∈ (list.range 21).tail) ∧ prices.length = 20 → 
  (∃ min_cost, min_cost = 136 ∧ min_cost = 
  list.sum prices - list.sum (list.take 4 (list.reverse (list.sort prices)))) :=
begin
  intro prices,
  intro h,
  have mem_range := h.1,
  have length_20 := h.2,
  use 136,
  split,
  { refl },
  { have sorted_prices := list.sort prices,
    have reversed_sorted_prices := list.reverse sorted_prices,
    have top_4_free := list.take 4 reversed_sorted_prices,
    have all_items_sum := list.sum prices,
    have free_items_sum := list.sum top_4_free,
    have min_cost := all_items_sum - free_items_sum,
    exact min_cost }
end

end min_total_spent_l707_707802


namespace smallest_difference_l707_707063

theorem smallest_difference {a b : ℕ} (h1: a * b = 2010) (h2: a > b) : a - b = 37 :=
sorry

end smallest_difference_l707_707063


namespace max_value_expression_l707_707373

variable {a b c d e : ℝ}
variable {a_pos : 0 < a} {b_pos : 0 < b} {c_pos : 0 < c} {d_pos : 0 < d} {e_pos : 0 < e}
variable {sum_sq : a^2 + b^2 + c^2 + d^2 + e^2 = 504}

theorem max_value_expression :
  ∃ (M a_M b_M c_M d_M e_M : ℝ),
    (M = 252 * Real.sqrt 62) ∧
    (a_M = 2) ∧
    (b_M = 6) ∧
    (c_M = 6 * Real.sqrt 7) ∧
    (d_M = 8) ∧
    (e_M = 12) ∧
    a^2 + b^2 + c^2 + d^2 + e^2 = 504 ∧
    a_M^2 + b_M^2 + c_M^2 + d_M^2 + e_M^2 = 504 ∧
    ac + 3bc + 4cd + 6ce ≤ M ∧
    (M + a_M + b_M + c_M + d_M + e_M = 28 + 252 * Real.sqrt 62 + 6 * Real.sqrt 7) :=
sorry

end max_value_expression_l707_707373


namespace largest_three_digit_n_l707_707830

-- Define the conditions and the proof statement
theorem largest_three_digit_n (n : ℕ) : 
  (70 * n ≡ 210 [MOD 350]) ∧ (n ≤ 999) ∧ (n ≥ 100) → n = 998 :=
begin
  -- Sorry as a placeholder for the proof
  sorry,
end

end largest_three_digit_n_l707_707830


namespace alison_birth_weekday_l707_707017

-- Definitions for the problem conditions
def days_in_week : ℕ := 7

-- John's birth day
def john_birth_weekday : ℕ := 3  -- Assuming Monday=0, Tuesday=1, ..., Wednesday=3, ...

-- Number of days Alison was born later
def days_later : ℕ := 72

-- Proof that the resultant day is Friday
theorem alison_birth_weekday : (john_birth_weekday + days_later) % days_in_week = 5 :=
by
  sorry

end alison_birth_weekday_l707_707017


namespace find_Y_payment_l707_707465

theorem find_Y_payment 
  (P X Z : ℝ)
  (total_payment : ℝ)
  (h1 : P + X + Z = total_payment)
  (h2 : X = 1.2 * P)
  (h3 : Z = 0.96 * P) :
  P = 332.28 := by
  sorry

end find_Y_payment_l707_707465


namespace pizza_slices_l707_707871

theorem pizza_slices (total_slices slices_with_pepperoni slices_with_mushrooms : ℕ) (h1 : total_slices = 15)
  (h2 : slices_with_pepperoni = 8) (h3 : slices_with_mushrooms = 12)
  (h4 : ∀ slice, slice < total_slices → (slice ∈ {x | x < slices_with_pepperoni} ∨ slice ∈ {x | x < slices_with_mushrooms})) :
  ∃ n : ℕ, (slices_with_pepperoni - n) + (slices_with_mushrooms - n) + n = total_slices ∧ n = 5 :=
by simp [h1, h2, h3]; use 5; linarith; sorry

end pizza_slices_l707_707871


namespace shane_chewed_pieces_l707_707588

theorem shane_chewed_pieces :
  ∀ (Elyse Rick Shane: ℕ),
  Elyse = 100 →
  Rick = Elyse / 2 →
  Shane = Rick / 2 →
  Shane_left = 14 →
  (Shane - Shane_left) = 11 :=
by
  intros Elyse Rick Shane Elyse_def Rick_def Shane_def Shane_left_def
  sorry

end shane_chewed_pieces_l707_707588


namespace parametric_eq_of_curve_C_max_x_plus_y_on_curve_C_l707_707702

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := 
(2 + Real.sqrt 2 * Real.cos θ, 
 2 + Real.sqrt 2 * Real.sin θ)

theorem parametric_eq_of_curve_C (θ : ℝ) : 
    ∃ x y, 
    (x, y) = curve_C θ ∧ 
    (x - 2)^2 + (y - 2)^2 = 2 := by sorry

theorem max_x_plus_y_on_curve_C :
    ∃ x y θ, 
    (x, y) = curve_C θ ∧ 
    (∀ p : ℝ × ℝ, (p.1, p.2) = curve_C θ → 
    p.1 + p.2 ≤ 6) ∧
    x + y = 6 ∧
    x = 3 ∧ 
    y = 3 := by sorry

end parametric_eq_of_curve_C_max_x_plus_y_on_curve_C_l707_707702


namespace find_angle_x_l707_707530

theorem find_angle_x (PT_per_Rs : ∀ (PT : ℝ) (RS : ℝ), PT ⊥ RS)
    (SPA_eq : ∠SPA = 180 - 90 - 26)
    (angles_equal : ∀ (MPA : ℝ), MPA = x)
    (inc_reflect_equal : ∠SPA = 2 * x) : 
    x = 32 :=
by
  sorry

end find_angle_x_l707_707530


namespace find_t_l707_707659

variable (a b c : ℝ × ℝ)
variable (t : ℝ)

-- Definitions based on given conditions
def vec_a : ℝ × ℝ := (3, 1)
def vec_b : ℝ × ℝ := (1, 3)
def vec_c (t : ℝ) : ℝ × ℝ := (t, 2)

-- Dot product definition
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Condition that (vec_a - vec_c) is perpendicular to vec_b
def perpendicular_condition (t : ℝ) : Prop :=
  dot_product (vec_a - vec_c t) vec_b = 0

-- Proof statement
theorem find_t : ∃ t : ℝ, perpendicular_condition t ∧ t = 0 := 
by
  sorry

end find_t_l707_707659


namespace min_people_with_all_luxuries_l707_707142

theorem min_people_with_all_luxuries {total_population : ℕ} 
  (refrigerator_percentage : ℕ)
  (television_percentage : ℕ)
  (computer_percentage : ℕ)
  (air_conditioner_percentage : ℕ) :
  (refrigerator_percentage = 70) →
  (television_percentage = 75) →
  (computer_percentage = 90) →
  (air_conditioner_percentage = 85) →
  ∃ (n : ℕ), n = 70 :=
begin
  intros,
  use 70,
  sorry,
end

end min_people_with_all_luxuries_l707_707142


namespace correct_propositions_l707_707068

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3) + Real.cos (2 * x + Real.pi / 6)

theorem correct_propositions :
  (∀ x, f x = Real.sqrt 2 * Real.cos (2 * x - Real.pi / 12)) ∧
  (Real.sqrt 2 = f (Real.pi / 24)) ∧
  (f (-1) ≠ f 1) ∧
  (∀ x, Real.pi / 24 ≤ x ∧ x ≤ 13 * Real.pi / 24 -> (f (x + 1e-6) < f x)) ∧
  (∀ x, (Real.sqrt 2 * Real.cos (2 * (x - Real.pi / 24))) = f x)
  := by
    sorry

end correct_propositions_l707_707068


namespace area_of_triangle_correct_l707_707275

def area_of_triangle_ABC_passing_through_A : Prop :=
  ∃ (m n : ℝ), 
    (∀ x y : ℝ, y = (3/2) * x + m → y = 0 ∧ x = -4) ∧
    (∀ x y : ℝ, y = (-1/2) * x + n → y = 0 ∧ x = -4) ∧
    let B := (0, 6) in
    let C := (0, -2) in
    let A := (-4, 0) in
    let base := dist B C in
    let height := |fst A| in
    0.5 * base * height = 16

theorem area_of_triangle_correct : area_of_triangle_ABC_passing_through_A :=
  sorry

end area_of_triangle_correct_l707_707275


namespace seq_arithmetic_seq_formula_l707_707624

-- Define the sequence a_n with the given conditions
def seq (n : ℕ) : ℕ → ℕ
| 0     := 0   -- start with 0 for convenience
| 1     := 1
| (n+1) := (n+1) * (seq n) + n * (n+1)

-- Prove that the sequence {a_n / n} is an arithmetic sequence
theorem seq_arithmetic:
  ∃ d : ℕ, ∀ n : ℕ, n ≠ 0 → (seq n) / n + d = (seq (n+1)) / (n + 1) := sorry

-- Prove that the general formula for {a_n} is a_n = n^2
theorem seq_formula:
  ∀ n : ℕ, n ≠ 0 → seq n = n * n := sorry

end seq_arithmetic_seq_formula_l707_707624


namespace total_amount_to_be_divided_l707_707910

theorem total_amount_to_be_divided
  (k m x : ℕ)
  (h1 : 18 * k = x)
  (h2 : 20 * m = x)
  (h3 : 13 * m = 11 * k + 1400) :
  x = 36000 := 
sorry

end total_amount_to_be_divided_l707_707910


namespace common_root_sum_and_product_max_l707_707378

theorem common_root_sum_and_product_max (a b : ℝ) (x0 x1 x2 : ℝ) 
  (h1 : a < 0) (h2 : b < 0) (h3 : a ≠ b) 
  (h4 : x0^2 + a*x0 + b = 0) (h5 : x0^2 + b*x0 + a = 0) 
  (h6 : roots_of_x1_x2 : x1 + x2 = -a - b - 2) 
  (h7 : x1 * x2 = (-1 - a) * (-1 - b)) :
  (x1 + x2 = -1) ∧ (∃ a : ℝ, a = -1/2 ∧ x1 * x2 = 1/4) := 
by
  apply And.intro
  -- proof for x1 + x2 = -1
  sorry
  -- proof for maximum of x1 * x2
  sorry

end common_root_sum_and_product_max_l707_707378


namespace problem_l707_707620

namespace ComplexNumbers

noncomputable def i7 : ℝ := Complex.exp (Complex.I * π / 2)^7

noncomputable def z : ℂ := (-3 + Complex.I) / i7

noncomputable def OA : ℂ := -1 - 3 * Complex.I

noncomputable def OB : ℂ := 2 - Complex.I

noncomputable def AB : ℂ := OB - OA

theorem problem
  (hz : z = -1 - 3 * Complex.I)
  (hAB : AB = 3 + 2 * Complex.I) :
  true := by
  rw [hz, hAB]
  exact trivial

end ComplexNumbers

end problem_l707_707620


namespace school_can_accommodate_l707_707896

-- Define the conditions
def total_classrooms : ℕ := 30
def desks_per_classroom_1 : ℕ := 40
def desks_per_classroom_2 : ℕ := 35
def desks_per_classroom_3 : ℕ := 28

def fraction_classroom_1 : ℝ := 1 / 5
def fraction_classroom_2 : ℝ := 1 / 3

noncomputable def classrooms_with_40_desks := (fraction_classroom_1 * total_classrooms).to_nat
noncomputable def classrooms_with_35_desks := (fraction_classroom_2 * total_classrooms).to_nat
def classrooms_with_28_desks := total_classrooms - classrooms_with_40_desks - classrooms_with_35_desks

-- Calculate the total number of desks
def total_desks : ℕ := 
  (classrooms_with_40_desks * desks_per_classroom_1) +
  (classrooms_with_35_desks * desks_per_classroom_2) +
  (classrooms_with_28_desks * desks_per_classroom_3)

-- The theorem to prove
theorem school_can_accommodate : total_desks = 982 :=
by sorry

end school_can_accommodate_l707_707896


namespace proof_of_first_plot_seeds_l707_707602

noncomputable def seeds_planted_in_first_plot : ℕ :=
  let x := 300
  let seeds_in_second_plot := 200
  let percent_germinated_first_plot := 0.20
  let percent_germinated_second_plot := 0.35
  let percent_total_germinated := 0.26
  have germinated_in_first_plot : ℝ := percent_germinated_first_plot * x
  have germinated_in_second_plot : ℝ := percent_germinated_second_plot * seeds_in_second_plot
  have total_seeds := x + seeds_in_second_plot
  have total_germinated := germinated_in_first_plot + germinated_in_second_plot
  have germination_equation := (total_germinated / total_seeds = percent_total_germinated)
  x

theorem proof_of_first_plot_seeds : seeds_planted_in_first_plot = 300 :=
by
  sorry

end proof_of_first_plot_seeds_l707_707602


namespace evaluate_expression_l707_707216

theorem evaluate_expression (x y z : ℝ) (hx : x = 3) (hy : y = 1/2) (hz : z = -18) :
  (x ^ (-2)) / (y ^ 3 * z) = -16 :=
by
  -- proof starts here
  sorry

end evaluate_expression_l707_707216


namespace systematic_sampling_count_l707_707547

def graduates : ℕ := 640
def sample_size : ℕ := 32
def interval_lower : ℕ := 161
def interval_upper : ℕ := 380

theorem systematic_sampling_count :
  let k := graduates / sample_size in
  let first_valid := (interval_lower + k - 1) / k * k in
  let last_valid := interval_upper / k * k in
  (last_valid - first_valid) / k + 1 = 11 :=
sorry

end systematic_sampling_count_l707_707547


namespace circle_radius_of_diameter_l707_707852

theorem circle_radius_of_diameter (d : ℝ) (h : d = 22) : d / 2 = 11 :=
by
  sorry

end circle_radius_of_diameter_l707_707852


namespace inv_g_sum_l707_707714

def g (x : ℝ) : ℝ :=
if x < 10 then 2 * x + 4 else 3 * x - 3

theorem inv_g_sum :
  let g_inv (y : ℝ) : ℝ :=
    if y = 8 then 2 else if y = 30 then 11 else 0 -- defining the inverse only for the given y values
  in g_inv 8 + g_inv 30 = 13 :=
by
  sorry

end inv_g_sum_l707_707714


namespace correct_arithmetic_square_root_l707_707133

noncomputable def arithmetic_square_root (x : ℝ) : ℝ :=
  if x < 0 then -real.sqrt (-x) else real.sqrt x

theorem correct_arithmetic_square_root : arithmetic_square_root ((π - 4) ^ 2) = 4 - π :=
by
  -- proof goes here
  sorry

end correct_arithmetic_square_root_l707_707133


namespace find_other_number_l707_707082

theorem find_other_number 
  (A B : ℕ) 
  (h1 : A = 385) 
  (h2 : Nat.lcm A B = 2310) 
  (h3 : Nat.gcd A B = 30) : 
  B = 180 := 
by
  sorry

end find_other_number_l707_707082


namespace empty_pencil_cases_l707_707458

theorem empty_pencil_cases (total_cases pencil_cases pen_cases both_cases : ℕ) 
  (h1 : total_cases = 10)
  (h2 : pencil_cases = 5)
  (h3 : pen_cases = 4)
  (h4 : both_cases = 2) : total_cases - (pencil_cases + pen_cases - both_cases) = 3 := by
  sorry

end empty_pencil_cases_l707_707458


namespace probability_no_defective_pencils_l707_707688

-- Definitions based on conditions
def total_pencils : ℕ := 11
def defective_pencils : ℕ := 2
def selected_pencils : ℕ := 3

-- Helper function to compute combinations
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The proof statement
theorem probability_no_defective_pencils :
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination (total_pencils - defective_pencils) selected_pencils
  total_ways ≠ 0 → 
  (non_defective_ways / total_ways : ℚ) = 28 / 55 := 
by
  sorry

end probability_no_defective_pencils_l707_707688


namespace second_number_in_pair_l707_707809

theorem second_number_in_pair (n m : ℕ) (h1 : (n, m) = (57, 58)) (h2 : ∃ (n m : ℕ), n < 1500 ∧ m < 1500 ∧ (n + m) % 5 = 0) : m = 58 :=
by {
  sorry
}

end second_number_in_pair_l707_707809


namespace sum_of_entries_possible_values_l707_707716

def is_5x5_matrix (X : Matrix (Fin 5) (Fin 5) ℕ) : Prop := 
  ∀ i j, X i j = 0 ∨ X i j = 1

def are_all_sequences_unique (X : Matrix (Fin 5) (Fin 5) ℕ) : Prop := 
  let rows : Fin 5 → List ℕ := λ i => List.ofFn λ j => X i j
  let cols : Fin 5 → List ℕ := λ j => List.ofFn λ i => X i j
  let main_diag1 : List ℕ := List.ofFn λ k : Fin 5 => X k k
  let main_diag2 : List ℕ := List.ofFn λ k : Fin 5 => X k.reverseIdx k
  let anti_diag1 : List ℕ := List.ofFn λ k : Fin 5 => X k (Fin.rev k)
  let anti_diag2 : List ℕ := List.ofFn λ k : Fin 5 => X (Fin.rev k) k
  (List.nodup (rows.toList ++ cols.toList ++ [main_diag1, main_diag2, anti_diag1, anti_diag2]))

def sum_entries (X : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  ∑ i j, X i j

theorem sum_of_entries_possible_values (X : Matrix (Fin 5) (Fin 5) ℕ) :
  is_5x5_matrix X →
  are_all_sequences_unique X →
  sum_entries X = 12 ∨ sum_entries X = 13 :=
by
  sorry

end sum_of_entries_possible_values_l707_707716


namespace sequence_of_moves_cannot_be_infinite_l707_707104

-- Definitions of conditions
def non_degenerate_non_equilateral_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧ (a ≠ b ∨ b ≠ c ∨ a ≠ c)

def perform_move (a b c : ℕ) : ℕ × ℕ × ℕ :=
  (a + b - c, b + c - a, c + a - b)

noncomputable def product (lst : List ℕ) : ℕ :=
  List.prod lst

theorem sequence_of_moves_cannot_be_infinite (n : ℕ) (n_ge_3 : n ≥ 3) (numbers : List ℕ)
    (initial_condition : ∀ a b c ∈ numbers, non_degenerate_non_equilateral_triangle a b c) :
  ¬(∀ m : ℕ, ∃ numbers_m : List ℕ, numbers_m ≠ [] ∧ List.length numbers_m = n ∧
    ∀ a b c ∈ numbers_m, non_degenerate_non_equilateral_triangle a b c ∧
    product numbers_m < product numbers) :=
sorry

end sequence_of_moves_cannot_be_infinite_l707_707104


namespace unique_polynomial_identity_l707_707224

open Polynomial

theorem unique_polynomial_identity :
  ∀ (P : Polynomial ℝ), P.eval 0 = 0 ∧ (∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) → P = Polynomial.C 0 + Polynomial.X :=
begin
  sorry
end

end unique_polynomial_identity_l707_707224


namespace find_x_for_fn_l707_707368

noncomputable def f_1 (x : ℝ) : ℝ := (1 / 2) - (4 / (4 * x + 2))

noncomputable def f_n : ℕ → ℝ → ℝ
| 0, x := x
| 1, x := f_1 x
| (n+2), x := f_1 (f_n (n+1) x)

theorem find_x_for_fn (x : ℝ) : f_n 1001 x = x - 2 ↔ x = 3 / 5 ∨ x = 1 := sorry

end find_x_for_fn_l707_707368


namespace original_triangle_area_l707_707087

theorem original_triangle_area (A_orig A_new : ℝ) (h1 : A_new = 256) (h2 : A_new = 16 * A_orig) : A_orig = 16 :=
by
  sorry

end original_triangle_area_l707_707087


namespace sara_total_peaches_l707_707411

-- conditions
variable (initialPeaches orchardPeaches : ℝ)
variable (noncomputable totalPeaches : ℝ)

-- hypothesis based on conditions
axiom initialPeaches_eq : initialPeaches = 61.0
axiom orchardPeaches_eq : orchardPeaches = 24.0
axiom totalPeaches_eq : totalPeaches = 85.0

-- statement we need to prove
theorem sara_total_peaches :
  initialPeaches + orchardPeaches = totalPeaches := 
by 
  rw [initialPeaches_eq, orchardPeaches_eq, totalPeaches_eq]
  sorry

end sara_total_peaches_l707_707411


namespace g_at_9_l707_707371

noncomputable def f (x : ℝ) : ℝ := x^3 + x + 1

noncomputable def g (x : ℝ) : ℝ :=
  let a := root1 (f)
  let b := root2 (f)
  let c := root3 (f)
  -(x - a^2) * (x - b^2) * (x - c^2)

theorem g_at_9 : g 9 = 899 :=
by
  sorry

end g_at_9_l707_707371


namespace tangent_line_y_intercept_l707_707785

-- First, we define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

-- Define the derivative of the function f
def f_prime (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - a

-- Define the slope of the line x - y - 1 = 0
def line_slope : ℝ := 1

-- Given condition: the tangent line at x = 1 is parallel to the line, thus it has a slope of 1.
-- Solve for a such that f_prime 1 a = 1
def solve_a : ℝ := 3 - 1

-- Recalculate specific version of f with a = 2
def f_specific (x : ℝ) : ℝ := f x 2

-- Define the tangent line at x = 1 and determine its y-intercept
noncomputable def tangent_line_intercept (x₀ : ℝ) (m : ℝ) (y₀ : ℝ) : ℝ := y₀ - m * x₀

-- Prove the intercept is -2
theorem tangent_line_y_intercept : tangent_line_intercept 1 1 (f_specific 1) = -2 := 
by 
  have a_eq_2 : solve_a = 2 := (by linarith)
  rw [solve_a, a_eq_2]
  sorry -- Placeholder for the complete proof.

end tangent_line_y_intercept_l707_707785


namespace expand_polynomial_l707_707219

variable {R : Type*} [CommRing R]

theorem expand_polynomial (x : R) : (2 * x + 3) * (x + 6) = 2 * x^2 + 15 * x + 18 := 
sorry

end expand_polynomial_l707_707219


namespace right_triangle_exists_l707_707209

variables {A H P B C : Type}
variables [metric_space A] [metric_space H] [metric_space P] [metric_space B] [metric_space C]
variables (AH : segment A H) (AP : segment A P)

noncomputable def construct_right_triangle (A H P : Type) [metric_space A] [metric_space H] [metric_space P] (AH : segment A H) (AP : segment A P) : Type :=
  ∃ (B C : Type) [metric_space B] [metric_space C], 
    (segment B H).length + (segment C H).length = (segment B P).length + (segment C P).length

theorem right_triangle_exists (A H P : Type) [metric_space A] [metric_space H] [metric_space P] 
  (AH : segment A H) (AP : segment A P) : construct_right_triangle A H P AH AP :=
sorry

end right_triangle_exists_l707_707209


namespace carrot_lettuce_ratio_l707_707710

theorem carrot_lettuce_ratio :
  let lettuce_cal := 50
  let dressing_cal := 210
  let crust_cal := 600
  let pepperoni_cal := crust_cal / 3
  let cheese_cal := 400
  let total_pizza_cal := crust_cal + pepperoni_cal + cheese_cal
  let carrot_cal := C
  let total_salad_cal := lettuce_cal + carrot_cal + dressing_cal
  let jackson_salad_cal := (1 / 4) * total_salad_cal
  let jackson_pizza_cal := (1 / 5) * total_pizza_cal
  jackson_salad_cal + jackson_pizza_cal = 330 →
  carrot_cal / lettuce_cal = 2 :=
by
  intro lettuce_cal dressing_cal crust_cal pepperoni_cal cheese_cal total_pizza_cal carrot_cal total_salad_cal jackson_salad_cal jackson_pizza_cal h
  sorry

end carrot_lettuce_ratio_l707_707710


namespace find_300th_term_excl_squares_l707_707487

def is_perfect_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

def nth_term_excl_squares (n : ℕ) : ℕ :=
  let excluded := (List.range (n + n / 10)).filter (λ x, ¬ is_perfect_square x)
  excluded.nth n

theorem find_300th_term_excl_squares :
  nth_term_excl_squares 299 = 317 :=
by
  sorry

end find_300th_term_excl_squares_l707_707487


namespace min_expression_value_l707_707610

theorem min_expression_value (x : ℝ) (h1 : 0 < x) (h2 : log x + 1 ≤ x) : 
  (∃ y, ∀ x > 0, y ≤ (x^2 - log x + x) / x) ∧
  (∀ x > 0, ((x^2 - log x + x) / x = 2 ↔ x = 1)) :=
by 
  sorry

end min_expression_value_l707_707610


namespace integral_evaluation_l707_707947

noncomputable def integral_value : Real :=
  ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt (1 - (x - 1)^2) - x)

theorem integral_evaluation :
  integral_value = (Real.pi / 4) - 1 / 2 :=
by
  sorry

end integral_evaluation_l707_707947


namespace range_of_x_l707_707186

-- Let us define some basics about our points and vectors in a geometric space.
variables {ABC : Type*} [innerProductSpace ℝ ABC]
variables (A B C D E F P : ABC)
variables (BC AC AB : ℝ)
variables (CD EC AF : ℝ)
variables {x : ℝ}

-- Define vector relations based on conditions in the problem.
def in_ΔABC (A B C P : ABC) : Prop := -- Define that four points are in a quadrilateral manner.
P ≠ A ∧ P ≠ B ∧ P ≠ C

def lies_on (A B P : ABC) : Prop := -- Define what it means for a point to lie on a line segment.
∃ (r : ℝ), P = (1-r) • A + r • B

-- Given geometric conditions described.
axiom cond_CD : CD = (3/5) * BC
axiom cond_EC : EC = (1/2) * AC
axiom cond_AF : AF = (1/3) * AB
axiom P_in_AEDF : in_ΔABC A E D F P -- Point P inside quadrilateral AEDF.

-- Main statement to prove the equivalence about x range.
theorem range_of_x : 
  (∀A B C D E F P CD EC AF BC AC AB,
   lies_on D B C ∧ lies_on E C A ∧ lies_on F A B ∧
   (P_in_AEDF A E D F P) ∧
   (cond_CD) ∧ (cond_EC) ∧ (cond_AF) ∧ cond_vector_relation (x : ℝ) 
   (∀ x, (1/2) < x ∧ x < (4/3))) :=
sorry

end range_of_x_l707_707186


namespace find_number_ge_40_l707_707868

theorem find_number_ge_40 (x : ℝ) : 0.90 * x > 0.80 * 30 + 12 → x > 40 :=
by sorry

end find_number_ge_40_l707_707868


namespace sarah_total_distance_walked_l707_707584

noncomputable def total_distance : ℝ :=
  let rest_time : ℝ := 1 / 3
  let total_time : ℝ := 3.5
  let time_spent_walking : ℝ := total_time - rest_time -- time spent walking
  let uphill_speed : ℝ := 3 -- in mph
  let downhill_speed : ℝ := 4 -- in mph
  let d := time_spent_walking * (uphill_speed * downhill_speed) / (uphill_speed + downhill_speed) -- half distance D
  2 * d

theorem sarah_total_distance_walked :
  total_distance = 10.858 := sorry

end sarah_total_distance_walked_l707_707584


namespace least_weight_of_oranges_l707_707569

theorem least_weight_of_oranges :
  ∀ (a o : ℝ), (a ≥ 8 + 3 * o) → (a ≤ 4 * o) → (o ≥ 8) :=
by
  intros a o h1 h2
  sorry

end least_weight_of_oranges_l707_707569


namespace remain_when_divided_l707_707995

open Nat

def sum_of_squares_primes_mod_3 (ps : List ℕ) : ℕ :=
  (ps.map (λ p, p^2)).sum % 3

theorem remain_when_divided (ps : List ℕ) (h₀ : ps.length = 99) 
  (h₁ : ∀ p ∈ ps, Prime p) 
  (h₂ : ∀ p₁ p₂ ∈ ps, p₁ ≠ p₂ → p₁ ≠ p₂) :
  sum_of_squares_primes_mod_3 ps = 0 ∨ sum_of_squares_primes_mod_3 ps = 2 := by {
  sorry
}

end remain_when_divided_l707_707995


namespace intersection_M_N_l707_707297

def M (x : ℝ) : Prop := (x-1) * (x-3) * (x-5) < 0
def N (x : ℝ) : Prop := (x-2) * (x-4) * (x-6) > 0

theorem intersection_M_N : set_of M ∩ set_of N = set.Ioo 3 4 := 
sorry

end intersection_M_N_l707_707297


namespace factorial_div_45_over_42_eq_l707_707200

theorem factorial_div_45_over_42_eq :
  45! / 42! = 85140 := 
sorry

end factorial_div_45_over_42_eq_l707_707200


namespace parallel_tangents_l707_707657

theorem parallel_tangents (x_0 : ℝ) :
  (deriv (λ x : ℝ, x^2 - 1) x_0 = deriv (λ x : ℝ, 1 - x^3) x_0) →
  (x_0 = 0 ∨ x_0 = -2 / 3) :=
by
  sorry

end parallel_tangents_l707_707657


namespace saved_percentage_l707_707977

-- Define the given constants and values
def number_of_passengers : ℕ := 4
def cost_per_orange : ℝ := 1.5
def total_planned_spending : ℝ := 15

-- Define the computed cost of oranges if bought at the stop
def total_cost_of_oranges := number_of_passengers * cost_per_orange

-- Define the proportion of the original spending plan that the cost of the oranges represents
def proportion_saved := total_cost_of_oranges / total_planned_spending

-- Define the percentage of the money saved
def percentage_saved := proportion_saved * 100

-- Lean theorem statement
theorem saved_percentage :
  percentage_saved = 40 :=
sorry

end saved_percentage_l707_707977


namespace arithmetic_sequence_sum_l707_707627

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h1 : a 1 + a 2 = -1)
  (h2 : a 3 = 4)
  (h3 : ∀ n, a (n + 1) - a n = a 2 - a 1) :
  a 4 + a 5 = 17 :=
  sorry

end arithmetic_sequence_sum_l707_707627


namespace only_one_zero_point_l707_707291

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 2

theorem only_one_zero_point (a : ℝ) :
  (∃! x0 : ℝ, f a x0 = 0) ↔ a ∈ set.Iio (-real.sqrt 2) ∪ set.Ioi (real.sqrt 2) :=
by
  sorry

end only_one_zero_point_l707_707291


namespace solution_set_of_inequality_l707_707449

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 4*x - 3 ≥ 0} = set.Icc 1 3 :=
sorry

end solution_set_of_inequality_l707_707449


namespace harmonic_sets_preservation_l707_707619

noncomputable theory

open_locale classical

variables {P : Type} [projective_space P]

-- Define the harmonic property
def is_harmonic (A B C D : P) : Prop :=
  cross_ratio (A, B; C, D) = -1

-- Main theorem
theorem harmonic_sets_preservation
  (A B C D A1 B1 C1 D1 : P)
  (P : P)
  (h_inter: concurrent {A, A1} {B, B1} {C, C1} {D, D1})
  (h_harmonic: is_harmonic A B C D) :
  is_harmonic A1 B1 C1 D1 :=
sorry

end harmonic_sets_preservation_l707_707619


namespace length_train1_correct_l707_707116

-- Define the given conditions
def speed_train1_kmph : ℝ := 42
def speed_train2_kmph : ℝ := 30
def length_train2_meters : ℝ := 280
def time_seconds : ℝ := 23.998

-- Define conversion from kmph to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 5 / 18

-- Calculate the relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (speed_train1_kmph + speed_train2_kmph)

-- Calculate the total distance travelled in the given time
def total_distance_meters : ℝ := relative_speed_mps * time_seconds

-- Define the statement to prove
theorem length_train1_correct : 
  let L := total_distance_meters - length_train2_meters in
  L = 199.96 :=
by
  sorry

end length_train1_correct_l707_707116


namespace correct_propositions_count_l707_707907

-- Definitions for each proposition
def proposition_1 (L1 L2 : Line) (P : Plane) : Prop := 
  L1 ∥ P ∧ L2 ∥ P → L1 ∥ L2

def proposition_2 (P1 P2 : Plane) (L : Line) : Prop := 
  P1 ∥ L ∧ P2 ∥ L → P1 ∥ P2

def proposition_3 (L1 L2 : Line) (P : Plane) : Prop := 
  L1 ⟂ P ∧ L2 ⟂ P → L1 ∥ L2

def proposition_4 (P1 P2 : Plane) (L : Line) : Prop := 
  P1 ⟂ L ∧ P2 ⟂ L → P1 ∥ P2

-- Main theorem stating exactly two of these propositions are correct
theorem correct_propositions_count :
  ∃ (p1 p2 p3 p4 : Prop), 
    (p1 = proposition_1 ∧ p2 = proposition_2 ∧ p3 = proposition_3 ∧ p4 = proposition_4) ∧
    (¬ p1 ∧ ¬ p2 ∧ p3 ∧ p4) :=
by
  sorry

end correct_propositions_count_l707_707907


namespace middle_card_is_four_l707_707464

theorem middle_card_is_four (a b c : ℕ) (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
                            (h2 : a + b + c = 15)
                            (h3 : a < b ∧ b < c)
                            (h_casey : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            (h_tracy : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            (h_stacy : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            : b = 4 := 
sorry

end middle_card_is_four_l707_707464


namespace add_coefficients_l707_707570

theorem add_coefficients (a : ℕ) : 2 * a + a = 3 * a :=
by 
  sorry

end add_coefficients_l707_707570


namespace relationship_among_abc_l707_707243

noncomputable def a : ℝ := Real.log (1 / 2)
def b : ℝ := Real.sin (1 / 2)
noncomputable def c : ℝ := 2 ^ (-1 / 2 : ℝ)

theorem relationship_among_abc : a < b ∧ b < c :=
by
  have h1 : a = Real.log (1 / 2), by rfl
  have h2 : b = Real.sin (1 / 2), by rfl
  have h3 : c = 2 ^ (-1 / 2 : ℝ), by rfl
  sorry

end relationship_among_abc_l707_707243


namespace general_term_formula_l707_707362

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}

axiom a_pos (n : ℕ) (hn : 0 < n) : 0 < a_n n
axiom sum_eq (n : ℕ) (hn : 0 < n) : 4 * S_n n = (a_n n + 3) * (a_n n - 1)
axiom sum_def (n : ℕ) : S_n n = ∑ i in finset.range n, a_n (i + 1)

theorem general_term_formula (n : ℕ) (hn : 0 < n) : a_n n = 2 * n + 1 := by
  sorry

end general_term_formula_l707_707362


namespace find_m_l707_707433

-- Definitions
variable {A B C O H : Type}
variable {O_is_circumcenter : is_circumcenter O A B C}
variable {H_is_altitude_intersection : is_altitude_intersection H A B C}
variable (AH BH CH OA OB OC : ℝ)

-- Problem Statement
theorem find_m (h : AH * BH * CH = m * (OA * OB * OC)) : m = 1 :=
sorry

end find_m_l707_707433


namespace triangle_area_l707_707825

theorem triangle_area (a b c : ℝ) (ha : a = 6) (hb : b = 5) (hc : c = 5) (isosceles : a = 2 * b) :
  let s := (a + b + c) / 2
  let area := (s * (s - a) * (s - b) * (s - c)).sqrt
  area = 12 :=
by
  sorry

end triangle_area_l707_707825


namespace sum_of_angles_l707_707156

variable [Real]

def central_angle (n : ℝ) : ℝ := 360 / 16 * n
def inscribed_angle (central : ℝ) : ℝ := central / 2

theorem sum_of_angles
  (n m : ℝ)
  (hn : n = 3) 
  (hm : m = 5) : inscribed_angle (central_angle n) + inscribed_angle (central_angle m) = 90 :=
by
  have h_n : central_angle n = 67.5, from sorry
  have h_m : central_angle m = 112.5, from sorry
  have h_x : inscribed_angle (central_angle n) = 33.75, from sorry
  have h_y : inscribed_angle (central_angle m) = 56.25, from sorry
  rw [h_x, h_y]
  exact sorry

end sum_of_angles_l707_707156


namespace percentage_of_money_saved_is_correct_l707_707974

-- Definitions of the conditions
def family_size : ℕ := 4
def cost_per_orange : ℝ := 1.5
def planned_expenditure : ℝ := 15

-- Correct answer to be proven
def percentage_saved : ℝ :=
  ((family_size * cost_per_orange) / planned_expenditure) * 100

-- Proof goal
theorem percentage_of_money_saved_is_correct :
  percentage_saved = 40 := by
  sorry

end percentage_of_money_saved_is_correct_l707_707974


namespace unique_selection_l707_707629

open Nat

/-- Given a finite set S of natural numbers, the first player selects a number s from S.
The second player says a number x (not necessarily in S), and then the first player says σ₀(xs).
Both players know all elements of S. Prove that the second player can say only one number x and 
understand which number the first player has selected. -/
theorem unique_selection (S : Finset ℕ) (s : ℕ) (hS : s ∈ S) :
  ∃ (x : ℕ), ∀ s' ∈ S, s' ≠ s → 
  ∃! d, d = (x * s) ∧ (∀ t ∈ S, t ≠ s → (σ₀ (x * s) = σ₀ (x * t) → s = t)) :=
by
  sorry

end unique_selection_l707_707629


namespace shaded_region_area_is_one_third_l707_707901

noncomputable def area_of_shaded_region (s : ℝ) (beta : ℝ) : ℝ :=
  if 0 < beta ∧ beta < real.pi / 2 ∧ real.cos beta = 3 / 5 then
    1 / 3
  else
    0

theorem shaded_region_area_is_one_third :
  ∀ (s : ℝ) (beta : ℝ),
    s = 1 →
    0 < beta ∧ beta < real.pi / 2 →
    real.cos beta = 3 / 5 →
    area_of_shaded_region s beta = 1 / 3 :=
by
  intros s beta h_s h_beta h_cos
  rw [h_s] -- Use the fact that side length s = 1
  simp only [area_of_shaded_region, h_beta, h_cos]
  split_ifs
  · exact rfl
  all_goals { simp [*] }
  done

end shaded_region_area_is_one_third_l707_707901


namespace two_planes_parallel_to_third_l707_707683

-- Plane type
variable (plane : Type)

-- Definitions for parallelism
def parallel (π₁ π₂ : plane) : Prop := 
  sorry  -- Definition of what it means for two planes to be parallel

-- The statement of the problem
theorem two_planes_parallel_to_third (π₁ π₂ π₃ : plane) 
  (h₁ : parallel π₁ π₃) 
  (h₂ : parallel π₂ π₃) : 
  parallel π₁ π₂ :=
sorry

end two_planes_parallel_to_third_l707_707683


namespace all_terms_form_l707_707211

def sequence (n : ℕ) : ℕ
| 0 := 1
| 1 := 4
| (nat.succ (nat.succ n)) := 5 * sequence (nat.succ n) - sequence n

theorem all_terms_form {n : ℕ} : ∃ c d : ℤ, sequence n = c^2 + 3 * d^2 :=
by
  apply sorry

end all_terms_form_l707_707211


namespace binomial_variance_l707_707618

variable {n : ℕ}
variable {p : ℚ}

theorem binomial_variance (h₁ : n = 4) (h₂ : p = 1/2) : 
  variance (binomial n p) = 1 := by
  sorry

end binomial_variance_l707_707618


namespace macy_running_goal_l707_707746

/-- Macy's weekly running goal is 24 miles. She runs 3 miles per day. Calculate the miles 
    she has left to run after 6 days to meet her goal. --/
theorem macy_running_goal (miles_per_week goal_per_week : ℕ) (miles_per_day: ℕ) (days_run: ℕ) 
  (h1 : miles_per_week = 24) (h2 : miles_per_day = 3) (h3 : days_run = 6) : 
  miles_per_week - miles_per_day * days_run = 6 := 
  by 
    rw [h1, h2, h3]
    exact Nat.sub_eq_of_eq_add (by norm_num)

end macy_running_goal_l707_707746


namespace min_steps_equivalence_l707_707147

-- Define the structure for the tree and paths
variables {G : Type} [Graph G]

-- Define the conditions for the problem
noncomputable def is_tree (G : Graph) : Prop :=
  (∃ (n : ℕ), G.vertex_count = n) ∧ 
  (∃ (m : ℕ), G.edge_count = m) ∧ 
  ∀ x y, ∃! (p : Path G x y), true

noncomputable def removing_leaves_results_in_path (G : Graph) : Prop :=
  let leaves := {v ∈ G.vertices | G.degree v = 1} in
  let G' := G.remove_vertices leaves in
  G'.is_path

noncomputable def min_steps_to_form_path (G : Graph) : ℕ :=
  sorry -- dummy function to be implemented

noncomputable def min_steps_to_form_path_with_neighbors (G : Graph) : ℕ :=
  sorry -- dummy function to be implemented

theorem min_steps_equivalence (G : Graph) :
  is_tree G → 
  (min_steps_to_form_path G = min_steps_to_form_path_with_neighbors G) ↔ 
  removing_leaves_results_in_path G :=
begin
  intros h_tree,
  split,
  { intro h_eq,
    sorry, -- proof to be implemented
  },
  { intro h_path,
    sorry, -- proof to be implemented
  }
end

end min_steps_equivalence_l707_707147


namespace cos_double_angle_of_acute_angle_l707_707270

theorem cos_double_angle_of_acute_angle (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (h : sin (α - π / 4) = 1 / 4) : cos (2 * α) = -sqrt (15) / 8 :=
by sorry

end cos_double_angle_of_acute_angle_l707_707270


namespace red_car_distance_ahead_l707_707817

theorem red_car_distance_ahead (v_red v_black time : ℝ) (h_v_red : v_red = 40) (h_v_black : v_black = 50) (h_time : time = 1) :
  ∃ d : ℝ, d = 10 :=
by
  have relative_speed := v_black - v_red
  have h_relative_speed : relative_speed = 10 := by rw [h_v_red, h_v_black]; norm_num
  use relative_speed * time
  rw [h_relative_speed, h_time]
  norm_num
  done

end red_car_distance_ahead_l707_707817


namespace fourth_root_of_2560000_l707_707198

theorem fourth_root_of_2560000 :
  (2560000 : ℝ) = (256 * 10000) →
  (10000 : ℝ) = (10^4) →
  (256 : ℝ) = (16^2) →
  (Real.root 4 2560000) = 40 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end fourth_root_of_2560000_l707_707198


namespace problem_prob_more_than_two_thirds_circumference_l707_707815

theorem problem_prob_more_than_two_thirds_circumference :
  let dice_sides := 8
  let total_outcomes := dice_sides * dice_sides
  let valid_sums := {d | d ∈ finset.range (2 * dice_sides) ∧ d > 8 / 3}.card
  valid_sums.to_rat / total_outcomes = 7 / 32 :=
by sorry

end problem_prob_more_than_two_thirds_circumference_l707_707815


namespace initial_ratio_men_to_women_l707_707352

theorem initial_ratio_men_to_women (M W : ℕ) (h1 : (W - 3) * 2 = 24) (h2 : 14 - 2 = M) : M / gcd M W = 4 ∧ W / gcd M W = 5 := by 
  sorry

end initial_ratio_men_to_women_l707_707352


namespace number_of_correct_propositions_l707_707462

noncomputable def P1 : Prop :=
  ∀ (α β : ℝ), (∃ (k : ℤ), α = β + 2 * k * Real.pi) → ∃ (f : ℝ → ℝ), f α = f β

noncomputable def P2 : Prop :=
  ∀ (α : ℝ), (∃ (k : ℤ), α = 2 * k * Real.pi) ↔ (∃ (k : ℤ), α = k * Real.pi)

noncomputable def P3 : Prop :=
  ∀ (α : ℝ), (Real.sin α > 0) → (0 < α ∧ α < Real.pi)

noncomputable def P4 : Prop :=
  ∀ (α β : ℝ), (Real.sin α = Real.sin β) → ∃ (k : ℤ), α = β + 2 * k * Real.pi

theorem number_of_correct_propositions : (if P1 then 1 else 0) + 
                                         (if P2 then 1 else 0) + 
                                         (if P3 then 1 else 0) + 
                                         (if P4 then 1 else 0) = 1 := 
by 
  sorry

end number_of_correct_propositions_l707_707462


namespace cost_of_each_steak_l707_707044

def beef_cost := 15 * 5
def total_ounces := 15 * 16
def num_steaks := total_ounces / 12
def cost_per_steak := beef_cost / num_steaks

theorem cost_of_each_steak :
  cost_per_steak = 3.75 :=
by
  sorry

end cost_of_each_steak_l707_707044


namespace dividend_calculation_l707_707854

theorem dividend_calculation
  (divisor : Int)
  (quotient : Int)
  (remainder : Int)
  (dividend : Int)
  (h_divisor : divisor = 800)
  (h_quotient : quotient = 594)
  (h_remainder : remainder = -968)
  (h_dividend : dividend = (divisor * quotient) + remainder) :
  dividend = 474232 := by
  sorry

end dividend_calculation_l707_707854


namespace problem1_problem2_l707_707153

-- Definitions
def total_questions := 5
def multiple_choice := 3
def true_false := 2
def total_outcomes := total_questions * (total_questions - 1)

-- (1) Probability of A drawing a true/false question and B drawing a multiple-choice question
def favorable_outcomes_1 := true_false * multiple_choice

-- (2) Probability of at least one of A or B drawing a multiple-choice question
def unfavorable_outcomes_2 := true_false * (true_false - 1)

-- Statements to be proved
theorem problem1 : favorable_outcomes_1 / total_outcomes = 3 / 10 := by sorry

theorem problem2 : 1 - (unfavorable_outcomes_2 / total_outcomes) = 9 / 10 := by sorry

end problem1_problem2_l707_707153


namespace determine_JK_length_l707_707208

variables (F G H I J K : Type) [metric_space F] [metric_space G] [metric_space H] [metric_space I] [metric_space J] [metric_space K]
variables (tri_FGH : triangle F G H) (tri_IJK : triangle I J K)

def GH_length : ℝ := 30
def IJ_length : ℝ := 18
def HK_length : ℝ := 15
def similar_triangles : Prop := tri_FGH ∼ tri_IJK

theorem determine_JK_length (h_similar : similar_triangles)
  (h_GH : GH_length = 30)
  (h_IJ : IJ_length = 18)
  (h_HK : HK_length = 15) : 
  JK_length = 9 :=
begin
  sorry
end

end determine_JK_length_l707_707208


namespace tile_8x8_theorem_cannot_tile_10x10_theorem_l707_707011

-- Definitions
def given_shape_tiles_4x4 : Prop := 
  ∀ (b : ℕ × ℕ), (b = (4, 4)) → (∃ t : ℕ → ℕ → Prop, ∀ x y, t x y ↔ (x, y) ∈ (finset.range 4).product (finset.range 4))

def tile_8x8 (t : ℕ → ℕ → Prop) : Prop :=
  ∃ s : ℕ → ℕ → Prop, ∀ x y, s x y ↔ (x, y) ∈ (finset.range 8).product (finset.range 8)

def can_tile_8x8 : Prop :=
  given_shape_tiles_4x4 → tile_8x8 (λ x y, (x % 4, y % 4))

def tile_10x10 (t : ℕ → ℕ → Prop) : Prop :=
  ∃ s : ℕ → ℕ → Prop, ∀ x y, s x y ↔ (x, y) ∈ (finset.range 10).product (finset.range 10)

def cannot_tile_10x10 : Prop :=
  given_shape_tiles_4x4 → ¬(tile_10x10 (λ x y, (x % 4, y % 4)))

-- Theorems to be proved
theorem tile_8x8_theorem : can_tile_8x8 :=
by sorry

theorem cannot_tile_10x10_theorem : cannot_tile_10x10 :=
by sorry

end tile_8x8_theorem_cannot_tile_10x10_theorem_l707_707011


namespace length_of_XY_l707_707343

-- Definitions corresponding to the conditions
def YZ : ℝ := 15
def tan_Z : ℝ := 2
def tan_X : ℝ := 2.5

-- Theorem to be proved
theorem length_of_XY (YZ : ℝ) (tan_Z : ℝ) (tan_X : ℝ) (h_YZ : YZ = 15) (h_tan_Z : tan_Z = 2) (h_tan_X : tan_X = 2.5)
  (h_parallel : ∀ (WX ZY : set ℝ), WX ∥ ZY) (h_perpendicular : ∀ (WY ZY : set ℝ), WY ⊥ ZY) :
  XY = 6 * real.sqrt 29 := 
  sorry

end length_of_XY_l707_707343


namespace uncovered_square_positions_l707_707870

def triminó (x : ℕ) := x % 3 == 0

theorem uncovered_square_positions : ∃ (board : Finset (ℕ × ℕ)),
  board = {(1,3), (1,6), (2,3), (2,6), (3,3), (3,6), (4,3), (4,6), 
           (5,3), (5,6), (6,3), (6,6), (7,3), (7,6), (8,3), (8,6)} ∧
  ∀ (pos : ℕ × ℕ), 
    pos ∈ board → 
    (triminó pos.fst ∧ triminó pos.snd) :=
by {
  apply Exists.intro (Finset.mk [(1,3), (1,6), (2,3), (2,6), (3,3), (3,6), 
                                 (4,3), (4,6), (5,3), (5,6), (6,3), (6,6),
                                 (7,3), (7,6), (8,3), (8,6)]) sorry
}

end uncovered_square_positions_l707_707870


namespace probability_point_closer_to_origin_l707_707545

-- Define the vertices of the rectangle
def vertex1 := (0 : ℝ, 0 : ℝ)
def vertex2 := (4 : ℝ, 0 : ℝ)
def vertex3 := (4 : ℝ, 2 : ℝ)
def vertex4 := (0 : ℝ, 2 : ℝ)

-- Define the points (0,0) and (5,2)
def origin := (0 : ℝ, 0 : ℝ)
def point := (5 : ℝ, 2 : ℝ)

-- Define the probability proof statement
theorem probability_point_closer_to_origin :
  let area_rectangle := 4 * 2 in
  let area_closer_region := 5 in
  (area_closer_region / area_rectangle) = (5 / 8) :=
by
  let area_rectangle := 4 * 2
  let area_closer_region := 5
  show area_closer_region / area_rectangle = (5 / 8)
  sorry

end probability_point_closer_to_origin_l707_707545


namespace term_omit_perfect_squares_300_l707_707477

theorem term_omit_perfect_squares_300 (n : ℕ) (hn : n = 300) : 
  ∃ k : ℕ, k = 317 ∧ (∀ m : ℕ, (m < k → m * m ≠ k)) := 
sorry

end term_omit_perfect_squares_300_l707_707477


namespace tom_total_distance_l707_707470

-- Definitions corresponding to the problem conditions
def Time_swim := 2 -- hours
def Speed_swim := 2 -- miles per hour
def Time_run := Time_swim / 2 -- Tom runs for half the time he spent swimming
def Speed_run := Speed_swim * 4 -- Tom's running speed is 4 times his swimming speed

-- The Lean theorem to prove the total distance covered by Tom is 12 miles
theorem tom_total_distance : 
  let Distance_swim := Time_swim * Speed_swim in
  let Distance_run := Time_run * Speed_run in
  Distance_swim + Distance_run = 12 :=
by
  sorry

end tom_total_distance_l707_707470


namespace initial_pencils_on_desk_l707_707460

variable (initial_pencils_desk : ℕ) -- The number of pencils initially on the desk
variable (pencils_drawer : ℕ) -- Pencils in the drawer
variable (pencils_added : ℕ) -- Pencils added by Dan
variable (total_pencils : ℕ) -- Total pencils after adding

theorem initial_pencils_on_desk :
  pencils_drawer = 43 →
  pencils_added = 16 →
  total_pencils = 78 →
  initial_pencils_desk = total_pencils - pencils_drawer - pencils_added :=
begin
  sorry
end

end initial_pencils_on_desk_l707_707460


namespace find_n_l707_707235

theorem find_n
  (n : ℕ) (hn : n > 0)
  (a : ℕ → ℝ) (ha_pos : ∀ k, 1 ≤ k ∧ k ≤ n → a k > 0)
  (ha_sum : ∑ k in finset.range (n + 1), a k = 17)
  (S_n : ℝ) (hS_n : S_n = ∑ k in finset.range (n + 1), real.sqrt ((2 * k + 1) ^ 2 + (a k) ^ 2))
  (hS_n_int : S_n ∈ ℤ) :
  n = 12 :=
by sorry

end find_n_l707_707235


namespace monotonically_increasing_min_value_max_value_l707_707287

def f (x : ℝ) : ℝ := 1 - 3 / (x + 2)

theorem monotonically_increasing : ∀ (x1 x2 : ℝ), 3 ≤ x1 → x1 < x2 → x2 ≤ 5 → f x1 < f x2 :=
by
  sorry

theorem min_value : f 3 = 2 / 5 :=
by
  sorry

theorem max_value : f 5 = 4 / 7 :=
by
  sorry

end monotonically_increasing_min_value_max_value_l707_707287


namespace heidi_and_liam_paint_in_15_minutes_l707_707314

-- Definitions
def Heidi_rate : ℚ := 1 / 60
def Liam_rate : ℚ := 1 / 90
def combined_rate : ℚ := Heidi_rate + Liam_rate
def painting_time : ℚ := 15

-- Theorem to Prove
theorem heidi_and_liam_paint_in_15_minutes : painting_time * combined_rate = 5 / 12 := by
  sorry

end heidi_and_liam_paint_in_15_minutes_l707_707314


namespace carpet_breadth_l707_707911

theorem carpet_breadth (b : ℝ)
  (H1 : b > 0)
  (H2 : ∀ L, L = 1.44 * b)
  (H3 : ∀ L' B', L' = 1.40 * 1.44 * b ∧ B' = 1.25 * b)
  (H4 : ∀ C' R, C' = 4082.4 ∧ R = 45)
  (H5 : ∀ L' B' C' R, 2.52 * b^2 = C' / R) : b = 6 :=
by
  sorry

end carpet_breadth_l707_707911


namespace amanda_initial_candy_bars_l707_707556

def initial_candy_bars (initial_given: ℕ) (added_next_day: ℕ) (first_give: ℕ) (mult: ℕ)
(given_second_day: ℕ) (total_kept: ℕ): ℕ :=
initial_given - first_give + added_next_day - given_second_day

theorem amanda_initial_candy_bars (x: ℕ):
  let given_first_day: ℕ := 3,
      added_next: ℕ := 30,
      times_given: ℕ := 4,
      kept_total: ℕ := 22,
      given_second_day: ℕ := times_given * given_first_day
  in
  initial_candy_bars x added_next given_first_day times_given given_second_day kept_total = kept_total → 
  x = 7 :=
by
  sorry

end amanda_initial_candy_bars_l707_707556


namespace lily_pads_cover_half_l707_707516

theorem lily_pads_cover_half (P D : ℕ) (cover_entire : P * (2 ^ 25) = D) : P * (2 ^ 24) = D / 2 :=
by sorry

end lily_pads_cover_half_l707_707516


namespace thm_300th_term_non_square_seq_l707_707482

theorem thm_300th_term_non_square_seq : 
  let non_square_seq (n : ℕ) := { k : ℕ // k > 0 ∧ ∀ m : ℕ, m * m ≠ k } in
  (non_square_seq 300).val = 318 :=
by
  sorry

end thm_300th_term_non_square_seq_l707_707482


namespace jills_daily_earnings_first_month_l707_707357

-- Definitions based on conditions
variable (x : ℕ) -- daily earnings in the first month
def total_earnings_first_month := 30 * x
def total_earnings_second_month := 30 * (2 * x)
def total_earnings_third_month := 15 * (2 * x)
def total_earnings_three_months := total_earnings_first_month x + total_earnings_second_month x + total_earnings_third_month x

-- The theorem we need to prove
theorem jills_daily_earnings_first_month
  (h : total_earnings_three_months x = 1200) : x = 10 :=
sorry

end jills_daily_earnings_first_month_l707_707357


namespace tangent_circles_radius_l707_707818

theorem tangent_circles_radius :
  ∃ (r : ℝ), 
    ∀ (A B : EuclideanSpace ℝ (Fin 2)),
      ⇑Metric.dist A B = 6 ∧
      ∃ (r_A r_B : ℝ),
        r_A = 2 ∧ r_B = 3 ∧
        (A.2 = 0 ∧ B.2 = 0 ∧
        ∃ (C : EuclideanSpace ℝ (Fin 2)),
          ∃ r_C, 
            Metric.dist A C = r_A + r ∧ 
            Metric.dist B C = r_B + r ∧
            C.2 = r ∧ 
            r = 3) :=
sorry

end tangent_circles_radius_l707_707818


namespace sum_of_first_fifty_terms_l707_707032

-- Define the arithmetic sequences and their initial terms
def a1 : ℕ → ℝ := λ n => 10 + (n-1) * da
def b1 : ℕ → ℝ := λ n => 90 + (n-1) * db

-- Define the conditions
def condition1 : a1 50 + b1 50 = 200 := sorry
def condition2 : a1 1 = 10 := sorry
def condition3 : b1 1 = 90 := sorry

-- Define the sum of the first fifty terms
def s : ℕ → ℝ := λ n => a1 n + b1 n

theorem sum_of_first_fifty_terms : ∑ k in (finset.range 50), s (k + 1) = 7500 :=
  by sorry

end sum_of_first_fifty_terms_l707_707032


namespace midpoint_expression_l707_707041

def A : ℝ × ℝ := (30, 10)
def B : ℝ × ℝ := (6, 6)
def C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem midpoint_expression :
  let x := C.1
  let y := C.2
  3 * x - 5 * y = 14 :=
by
  sorry

end midpoint_expression_l707_707041


namespace g_18_equals_5832_l707_707726

noncomputable def g (n : ℕ) : ℕ := sorry

axiom cond1 : ∀ (n : ℕ), (0 < n) → g (n + 1) > g n
axiom cond2 : ∀ (m n : ℕ), (0 < m ∧ 0 < n) → g (m * n) = g m * g n
axiom cond3 : ∀ (m n : ℕ), (0 < m ∧ 0 < n ∧ m ≠ n ∧ m^2 = n^3) → (g m = n ∨ g n = m)

theorem g_18_equals_5832 : g 18 = 5832 :=
by sorry

end g_18_equals_5832_l707_707726


namespace find_valid_pairs_l707_707956

def satisfies_condition (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ (a ^ 2017 + b) % (a * b) = 0

theorem find_valid_pairs : 
  ∀ (a b : ℕ), satisfies_condition a b → (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 2 ^ 2017) := 
by
  sorry

end find_valid_pairs_l707_707956


namespace square_area_proof_square_area_square_area_final_square_area_correct_l707_707057

theorem square_area_proof (x : ℝ) (s1 : ℝ) (s2 : ℝ) (A : ℝ)
  (h1 : s1 = 5 * x - 20)
  (h2 : s2 = 25 - 2 * x)
  (h3 : s1 = s2) :
  A = (s1 * s1) := by
  -- We need to prove A = s1 * s1
  sorry

theorem square_area (x : ℝ) (s : ℝ) (h : s = 85 / 7) :
  s ^ 2 = 7225 / 49 := by
  -- We need to prove s^2 = 7225 / 49
  sorry

theorem square_area_final (x : ℝ)
  (h1 : 5 * x - 20 = 25 - 2 * x)
  (A : ℝ) :
  A = (85 / 7) ^ 2 := by
  -- We need to prove A = (85 / 7) ^ 2
  sorry

theorem square_area_correct (x : ℝ)
  (A : ℝ)
  (h1 : 5 * x - 20 = 25 - 2 * x)
  (h2 : A = (85 / 7) ^ 2) :
  A = 7225 / 49 := by
  -- We need to prove A = 7225 / 49
  sorry

end square_area_proof_square_area_square_area_final_square_area_correct_l707_707057


namespace domino_chain_max_length_l707_707160

theorem domino_chain_max_length :
  ∃ max_length count_of_chains, 
  max_length = 16 ∧ count_of_chains = 3456 ∧ 
  ∀ dominos: list (ℕ × ℕ), 
    (∀ (d : (ℕ × ℕ)), d ∈ dominos → (0 ≤ d.fst ∧ d.fst ≤ 6) ∧ (0 ≤ d.snd ∧ d.snd ≤ 6)) ∧ 
    (∀ pairs : list ((ℕ × ℕ) × (ℕ × ℕ)) , 
      (pairs.head.fst ≠ pairs.head.snd) → 
      (pairs.last.fst ≠ pairs.last.snd) →
      (∀ p : (ℕ × ℕ) × (ℕ × ℕ), p ∈ pairs → p.fst.snd = p.snd.fst) ∧ 
      (∀ d : ℕ × ℕ, d ∈ dominos → (d.fst + d.snd) % 2 = 1) → 
    (dominos.length ≤ max_length) ∧ (number_of_chains dominos = count_of_chains)) :=
  sorry

end domino_chain_max_length_l707_707160


namespace factorial_expression_evaluation_l707_707191

theorem factorial_expression_evaluation : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 5 * Nat.factorial 5 = 5760 := 
by 
  sorry

end factorial_expression_evaluation_l707_707191


namespace rectangular_plot_area_l707_707799

/-- The ratio between the length and the breadth of a rectangular plot is 7 : 5.
    If the perimeter of the plot is 288 meters, then the area of the plot is 5040 square meters.
-/
theorem rectangular_plot_area
    (L B : ℝ)
    (h1 : L / B = 7 / 5)
    (h2 : 2 * (L + B) = 288) :
    L * B = 5040 :=
by
  sorry

end rectangular_plot_area_l707_707799


namespace musical_roles_assignment_l707_707166

theorem musical_roles_assignment :
  let male_roles := 3
  let female_roles := 3
  let either_roles := 2
  let men := 7
  let women := 8
  let total_people := men + women
  let remaining_people := total_people - male_roles - female_roles
  let male_assignment := men * (men - 1) * (men - 2)
  let female_assignment := women * (women - 1) * (women - 2)
  let either_assignment := remaining_people * (remaining_people - 1)
  let total_assignments := male_assignment * female_assignment * either_assignment
  in total_assignments = 5080320 := 
begin
  sorry
end

end musical_roles_assignment_l707_707166


namespace smallest_root_of_quadratic_l707_707124

theorem smallest_root_of_quadratic :
  ∃ x, (10 * x ^ 2 - 48 * x + 44 = 0) ∧ (∀ y, (10 * y ^ 2 - 48 * y + 44 = 0) → x ≤ y) ∧ x = 1.234 :=
begin
  sorry
end

end smallest_root_of_quadratic_l707_707124


namespace pile_stabilization_l707_707103

theorem pile_stabilization (n : ℕ) (piles : list ℕ) (h1 : ∑ piles = n * (n + 1) / 2) (h2 : ∀ x ∈ piles, x > 0) :
  ∃ final_piles, (∀ x ∈ final_piles, x > 0) ∧ (∑ final_piles = n * (n + 1) / 2) ∧ (∀ (piles' : list ℕ), (piles' ≠ final_piles ∧ ∑ piles' = n * (n + 1) / 2) → false) :=
by
  sorry

end pile_stabilization_l707_707103


namespace radius_of_sphere_through_points_l707_707623

def Tetrahedron (A B C D : ℝ^3) : Prop :=
  ∃ a : ℝ, a > 0 ∧ 
  dist A B = a ∧ dist A C = a ∧ dist A D = a ∧ 
  dist B C = a ∧ dist B D = a ∧ dist C D = a

noncomputable def midpoint (P Q : ℝ^3) : ℝ^3 :=
  (P + Q) / 2

noncomputable def distance (P Q : ℝ^3) : ℝ :=
  real.sqrt (∥P - Q∥^2)

theorem radius_of_sphere_through_points 
  (a : ℝ) (A B C D : ℝ^3) (hTetra : Tetrahedron A B C D) :
  let M := midpoint A B,
      N := midpoint A C in
  ∃ (R : ℝ), 
  R = distance (midpoint (C + D) (M + N)) (M + N) ∧ 
  R = a * real.sqrt 22 / 8 := 
sorry

end radius_of_sphere_through_points_l707_707623


namespace find_x0_l707_707379

noncomputable def f (x : ℝ) (a c : ℝ) : ℝ := a * x^2 + c
noncomputable def int_f (a c : ℝ) : ℝ := ∫ x in (0 : ℝ)..1, f x a c

theorem find_x0 (a c x0 : ℝ) (h : a ≠ 0) (hx0 : 0 ≤ x0 ∧ x0 ≤ 1)
  (h_eq : int_f a c = f x0 a c) : x0 = Real.sqrt 3 / 3 := sorry

end find_x0_l707_707379


namespace term_omit_perfect_squares_300_l707_707479

theorem term_omit_perfect_squares_300 (n : ℕ) (hn : n = 300) : 
  ∃ k : ℕ, k = 317 ∧ (∀ m : ℕ, (m < k → m * m ≠ k)) := 
sorry

end term_omit_perfect_squares_300_l707_707479


namespace max_value_of_seq_b_is_1_over_8_l707_707548

-- Definitions for the sequences a_n and b_n
def seq_a (n : ℕ) : ℚ := if n = 1 then 1 else  2 - (1 / 2) ^ (n - 1)
def seq_b (n : ℕ) : ℚ := (2 - n) / 2 * (seq_a n - 2)

-- Conditions
axiom seq_a_sum_cond (n : ℕ) : n > 0 → (∑ i in Finset.range n, seq_a (i + 1)) = 2 * n - seq_a n

-- Question restated as Lean theorem
theorem max_value_of_seq_b_is_1_over_8 :
  ∃ n : ℕ, seq_b n = 1 / 8 ∧ ∀ m : ℕ, seq_b m ≤ 1 / 8 :=
sorry

end max_value_of_seq_b_is_1_over_8_l707_707548


namespace triangle_RSC_positive_diff_of_coordinates_l707_707814

-- Declaring the points A, B, and C as ordered pairs
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 0 6
def B := Point.mk 3 0
def C := Point.mk 9 0

-- Declaring the function y(x) for the line AC
noncomputable def y (x : ℝ) : ℝ := (-2/3) * x + 6

-- Line BC is horizontal (y = 0)
-- Points R and S on vertical line, xR = xS = x

-- Given that the area of triangle RSC is 15
def area_RSC (x : ℝ) : ℝ := (1/2) * |9 - x| * |y x|

-- Correct answer for the positive difference between x and y coordinates of R
theorem triangle_RSC_positive_diff_of_coordinates :
  ∃ x : ℝ, x ≠ 9 ∧ area_RSC x = 15 ∧ |x - y x| = 17 / 3 :=
sorry

end triangle_RSC_positive_diff_of_coordinates_l707_707814


namespace transformation_result_l707_707117

noncomputable def resulting_complex_number (z : ℂ) : ℂ :=
  let cis60 : ℂ := (1/2) + (complex.I * real.sqrt 3 / 2)
  let dilation : ℂ := 2
  in z * cis60 * dilation

theorem transformation_result :
  resulting_complex_number (-4 - 6 * complex.I) = (-4 + 6 * real.sqrt 3) - (4 * real.sqrt 3 + 6) * complex.I :=
by
  sorry

end transformation_result_l707_707117
