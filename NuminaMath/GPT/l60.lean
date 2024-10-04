import Data.Set.Fin
import Mathbin.Probability.ProbabilityMassFunction
import Mathlib
import Mathlib.Algebra.AddGroup
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group.Definitions
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Quadratic.Discriminant
import Mathlib.Algebra.Quadratics
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.OptForm
import Mathlib.Analysis.SpecialFunctions.Gaussian
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Perm
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.ModEq
import Mathlib.Data.Matrix.Det
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityNotation
import Mathlib.Probability.ProbabilityTheory
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic
import Mathlib.analysis.special_functions.trigonometric

namespace matrix_determinant_value_eq_l60_60639

open Real

theorem matrix_determinant_value_eq (x : ℝ) :
  let M := matrix.of ![![3 * x, 2], ![3, 2 * x]] in
  matrix.det M = 10 ↔ x = sqrt (8 / 3) ∨ x = -sqrt (8 / 3) := sorry

end matrix_determinant_value_eq_l60_60639


namespace nine_sided_convex_polygon_diagonals_l60_60578

theorem nine_sided_convex_polygon_diagonals : 
  ∀ (P : Type) [polygon P] (H1 : convex P) (H2 : sides P = 9) (H3 : right_angles P = 2), 
  diagonals P = 27 := 
by 
  sorry

end nine_sided_convex_polygon_diagonals_l60_60578


namespace a_m_power_m_divides_a_n_power_n_a1_does_not_divide_any_an_power_n_l60_60635

theorem a_m_power_m_divides_a_n_power_n:
  ∀ (a : ℕ → ℕ) (m : ℕ), (a 1).gcd (a 2) = 1 ∧ (∀ n, a (n + 2) = a (n + 1) * a n + 1) ∧ m > 1 → ∃ n > m, (a m) ^ m ∣ (a n) ^ n := by 
  sorry

theorem a1_does_not_divide_any_an_power_n:
  ∀ (a : ℕ → ℕ), (a 1).gcd (a 2) = 1 ∧ (∀ n, a (n + 2) = a (n + 1) * a n + 1) → ¬ ∃ n > 1, (a 1) ∣ (a n) ^ n := by
  sorry

end a_m_power_m_divides_a_n_power_n_a1_does_not_divide_any_an_power_n_l60_60635


namespace fraction_ordering_l60_60154

theorem fraction_ordering :
  (8 : ℚ) / 24 < (6 : ℚ) / 17 ∧ (6 : ℚ) / 17 < (10 : ℚ) / 27 :=
by
  sorry

end fraction_ordering_l60_60154


namespace distribution_possibilities_l60_60572

-- Definitions corresponding to the conditions
def numberOfSchools : Nat := 3
def numberOfStaff : Nat := 5
def minStaffPerSchool : Nat := 1
def maxStaffPerSchool : Nat := 2

-- Mathematical statement
theorem distribution_possibilities :
  ∃ (d : Fin numberOfSchools → Fin numberOfStaff), 
    (∀ s, minStaffPerSchool ≤ (List.filter (· = s) (List.ofFn d)).length) ∧
    (∀ s, (List.filter (· = s) (List.ofFn d)).length ≤ maxStaffPerSchool) →
    (List.filter (λ d, (List.filter (· = d) (List.ofFn d)).length ≥ minStaffPerSchool ∧ (List.filter (· = d) (List.ofFn d)).length ≤ maxStaffPerSchool) (List.finRange numberOfSchools)).length = 90 :=
sorry

end distribution_possibilities_l60_60572


namespace asia_population_status_l60_60501

theorem asia_population_status (population_1950 : ℕ) (population_2000 : ℕ) 
  (h1 : population_1950 = 1402000000) 
  (h2 : population_2000 = 3683000000) 
  (highest_1950 : ∀ c, c != population_1950 → c < population_1950)
  (highest_2000 : ∀ c, c != population_2000 → c < population_2000) : 
  (population_1950 = 1402000000 ∧ population_2000 = 3683000000) → 
  (highest_1950 ∧ highest_2000) :=
by
  intros 
  split
  all_goals { sorry }

end asia_population_status_l60_60501


namespace nth_term_arithmetic_seq_l60_60598

theorem nth_term_arithmetic_seq (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : ∀ n : ℕ, ∃ m : ℝ, a (n + 1) = a n + m)
  (h_d_neg : d < 0)
  (h_condition1 : a 2 * a 4 = 12)
  (h_condition2 : a 2 + a 4 = 8):
  ∀ n : ℕ, a n = -2 * n + 10 :=
by
  sorry

end nth_term_arithmetic_seq_l60_60598


namespace angle_B_is_pi_over_3_l60_60433

variables {A B C : ℝ} -- Angles of the triangle
variables {GA GB GC : Vector ℝ} -- Position vectors from G to the vertices
variable {G : Vector ℝ} -- Centroid of the triangle

-- Assume G is the centroid of triangle ABC
axiom centroid_condition : GA + GB + GC = 0

-- Given condition in the problem
axiom sine_condition : sin A • GA + sin B • GB + sin C • GC = 0

-- Goal is to prove that angle B equals π/3
theorem angle_B_is_pi_over_3 (h₁ : GA + GB + GC = 0)
                            (h₂ : sin A • GA + sin B • GB + sin C • GC = 0) :
                             B = π / 3 :=
sorry

end angle_B_is_pi_over_3_l60_60433


namespace equilateral_triangles_count_l60_60198

theorem equilateral_triangles_count :
  let k_sequence := (List.range (2*10 + 1)).map (λ n, n - 10) in
  let lines_y_eq_k := k_sequence.map (λ k, λ x, k : ℝ) in
  let lines_y_eq_sqrt3x_plus_2k := k_sequence.map (λ k, λ x, (√3) * x + 2 * k : ℝ) in
  let lines_y_eq_neg_sqrt3x_plus_2k := k_sequence.map (λ k, λ x, -(√3) * x + 2 * k : ℝ) in
  let all_lines := lines_y_eq_k ++ lines_y_eq_sqrt3x_plus_2k ++ lines_y_eq_neg_sqrt3x_plus_2k in
  let triangles_count := 660 in
  countEquilateralTriangles all_lines (2/√3) = triangles_count :=
sorry

end equilateral_triangles_count_l60_60198


namespace sum_inverse_binom_le_one_l60_60046

open Finset

theorem sum_inverse_binom_le_one (A : Finset (Finset (Fin n))) (M s : ℕ)
  (a : Fin s → ℕ) (h1 : ∀ i j : Fin s, ¬ (A i ⊆ A j) ∨ i = j)
  (h2 : ∀ i : Fin s, a i = (A i).card) :
  ∑ i in Finset.univ, (1 : ℝ) / (nat.choose M (a i)) ≤ 1 := 
sorry

end sum_inverse_binom_le_one_l60_60046


namespace expectation_X_is_1_l60_60254

-- Define the sample space and probabilities
def p0 : ℚ := 19 / 78
def p1 : ℚ := 20 / 39
def p2 : ℚ := 19 / 78

-- Define X as the random variable
def X : ℕ → ℚ
| 0 := p0
| 1 := p1
| 2 := p2
| _ := 0

-- Define the expectation of X
def expectation_X : ℚ := 0 * p0 + 1 * p1 + 2 * p2

-- Statement to be proven
theorem expectation_X_is_1 : expectation_X = 1 := by
  sorry

end expectation_X_is_1_l60_60254


namespace min_blue_beads_78_l60_60986

noncomputable def min_blue_beads_in_necklace : ℕ :=
sorry

theorem min_blue_beads_78 (R B : ℕ) (h_red : R = 100)
(h_condition : ∀ s : finset ℕ, s.card = 16 → 
(s.filter (λ n, n < 10)).card = 10 → 
(s.filter (λ n, n >= 10)).card ≤ 9 →
(s.card - (s.filter (λ n, n < 10)).card) ≥ 7) :
  min_blue_beads_in_necklace = 78 :=
by sorry

end min_blue_beads_78_l60_60986


namespace log_inequality_solution_set_l60_60128

theorem log_inequality_solution_set (x : ℝ) : 
  lg (x - 1) < 2 ↔ 1 < x ∧ x < 101 := 
by 
  sorry

end log_inequality_solution_set_l60_60128


namespace reciprocal_of_neg_two_l60_60879

-- Define the condition of reciprocal
def is_reciprocal (y x : ℝ) : Prop := y * x = 1

-- Define the number in question
def y : ℝ := -2

-- Define the expected reciprocal
def x : ℝ := -1 / 2

-- State the theorem
theorem reciprocal_of_neg_two : is_reciprocal y x :=
by
  -- Proof can be provided here
  sorry

end reciprocal_of_neg_two_l60_60879


namespace value_of_f_l60_60052

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then (1/2)^x else f (x - 1)

theorem value_of_f (x : ℝ) (h : x = 1 + real.log 3 / real.log 2) : f x = 4 / 3 :=
by
  intro x h
  rw h
  -- skipped proof
  sorry

end value_of_f_l60_60052


namespace job_completion_time_l60_60947

theorem job_completion_time (A_rate D_rate Combined_rate : ℝ) (hA : A_rate = 1 / 3) (hD : D_rate = 1 / 6) (hCombined : Combined_rate = A_rate + D_rate) :
  (1 / Combined_rate) = 2 :=
by sorry

end job_completion_time_l60_60947


namespace observable_sea_creatures_l60_60250

theorem observable_sea_creatures (P_shark : ℝ) (P_truth : ℝ) (n : ℕ)
  (h1 : P_shark = 0.027777777777777773)
  (h2 : P_truth = 1/6)
  (h3 : P_shark = P_truth * (1/n : ℝ)) : 
  n = 6 := 
  sorry

end observable_sea_creatures_l60_60250


namespace find_k_for_line_l60_60675

theorem find_k_for_line (k : ℝ) : (2 * k * (-1/2) + 1 = -7 * 3) → k = 22 :=
by
  intro h
  sorry

end find_k_for_line_l60_60675


namespace remainder_of_17_power_1801_mod_28_l60_60940

theorem remainder_of_17_power_1801_mod_28 : (17 ^ 1801) % 28 = 17 := 
by
  sorry

end remainder_of_17_power_1801_mod_28_l60_60940


namespace john_spent_170_l60_60425

def discount_amount (orig_price : ℝ) (discount_percent : ℝ) : ℝ :=
  orig_price * (discount_percent / 100)

def sale_price (orig_price : ℝ) (discount_amount : ℝ) : ℝ :=
  orig_price - discount_amount

def total_amount_spent (sale_price : ℝ) (quantity : ℝ) : ℝ :=
  sale_price * quantity

theorem john_spent_170 :
  let orig_price := 20
  let discount_percent := 15
  let quantity := 10
  discount_amount orig_price discount_percent = 3 →
  sale_price orig_price (discount_amount orig_price discount_percent) = 17 →
  total_amount_spent (sale_price orig_price (discount_amount orig_price discount_percent)) quantity = 170 :=
by
  intros h1 h2
  rw h1
  rw h2
  sorry

end john_spent_170_l60_60425


namespace least_possible_value_minimum_at_zero_zero_l60_60527

theorem least_possible_value (x y : ℝ) : (xy - 1)^2 + (x + y)^2 ≥ 1 :=
begin
  sorry
end

theorem minimum_at_zero_zero : (xy - 1)^2 + (x + y)^2 = 1 ↔ x = 0 ∧ y = 0 :=
begin
  sorry
end

end least_possible_value_minimum_at_zero_zero_l60_60527


namespace investor_amount_after_two_years_l60_60937

noncomputable def compound_interest
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem investor_amount_after_two_years :
  compound_interest 3000 0.10 1 2 = 3630 :=
by
  -- Calculation goes here
  sorry

end investor_amount_after_two_years_l60_60937


namespace sum_of_g_of_49_l60_60042

-- Define the functions f and g
def f (x : ℝ) := 5 * x^2 - 4
def g (y : ℝ) := 2 * y + 3 * (Real.sqrt (53 / 5)) + 2

-- State the theorem that needs to be proven
theorem sum_of_g_of_49 : 
  let x := Real.sqrt (53 / 5) in
  let y := 49 in
  g (f x) + g (f (-x)) = 46.4 :=
sorry

end sum_of_g_of_49_l60_60042


namespace at_least_one_root_irrational_l60_60443

-- Definitions and conditions as stated in the problem
def polynomial_in_ZX := ∃ (a b c d : ℤ), ∀ x, 
  (x : ℝ) -> (a * x^3 + b * x^2 + c * x + d = 0) → ℝ

def condition_ad_odd (a d : ℤ) : Prop := 
  odd (a * d)

def condition_bc_even (b c : ℤ) : Prop := 
  even (b * c)

def all_real_roots (P : polynomial_in_ZX) : Prop :=
  ∀ r, P r → ∃ x : ℝ, P x

-- Main theorem statement as per the problem translation
theorem at_least_one_root_irrational 
  (a b c d : ℤ) 
  (P : polynomial_in_ZX) 
  (h1 : condition_ad_odd a d) 
  (h2 : condition_bc_even b c) 
  (h3 : all_real_roots P)
  : ∃ x : ℝ, P x ∧ ¬ (∃ y : ℚ, x = ↑y) := 
begin
  sorry
end

end at_least_one_root_irrational_l60_60443


namespace points_in_quadrants_l60_60721

theorem points_in_quadrants (x y : ℝ) (h_line : 4 * x + 7 * y = 28)
  (h_equidistant : |x| = |y|) : 
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end points_in_quadrants_l60_60721


namespace total_students_on_field_trip_l60_60126

theorem total_students_on_field_trip (seats_per_bus : ℕ) (num_buses : ℕ) (h1 : seats_per_bus = 60) (h2 : num_buses = 3) : 
  seats_per_bus * num_buses = 180 :=
by
  rw [h1, h2]
  exact rfl

end total_students_on_field_trip_l60_60126


namespace factorial_div_9_4_l60_60335

theorem factorial_div_9_4 :
  (9! / 4!) = 15120 :=
by
  have h₁ : 9! = 362880 := by sorry
  have h₂ : 4! = 24 := by sorry
  rw [h₁, h₂]
  norm_num

end factorial_div_9_4_l60_60335


namespace reciprocal_of_neg2_l60_60914

-- Define the number
def num : ℤ := -2

-- Define the reciprocal function
def reciprocal (x : ℤ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_neg2 : reciprocal num = -1 / 2 :=
by
  -- Proof is skipped
  sorry

end reciprocal_of_neg2_l60_60914


namespace find_p_value_l60_60706

variables {a b : Vector ℝ} [NonCollinear a b]

-- Defining the vectors
def AB := 2 • a + p • b
def BC := a + b
def CD := a - 2 • b
def BD := BC + CD

-- Expressing collinearity condition
axiom collinear (h : AB = λ • BD)

-- The theorem to be proved
theorem find_p_value (h : AB = λ • BD) : p = -1 :=
sorry

end find_p_value_l60_60706


namespace Petya_wins_odd_l60_60962

-- Define the game conditions
def game_conditions (n : ℕ) (h : n ≥ 3): Prop :=
  (∀ m: ℕ, m ≠ n → m % 2 = 1) -> Petya_wins n

-- Define Petya's win condition
def Petya_wins : ℕ → Prop
| n := n % 2 = 1

-- The theorem stating that Petya wins for odd n
theorem Petya_wins_odd (n : ℕ) (h₁ : n ≥ 3) (h₂ : n % 2 = 1) : Petya_wins n :=
by
  sorry


end Petya_wins_odd_l60_60962


namespace determine_k_completed_square_l60_60003

theorem determine_k_completed_square (x : ℝ) :
  ∃ (a h k : ℝ), a * (x - h)^2 + k = x^2 - 7 * x ∧ k = -49/4 := sorry

end determine_k_completed_square_l60_60003


namespace bandage_overlap_l60_60948

theorem bandage_overlap
  (n : ℕ) (l : ℝ) (total_length : ℝ) (required_length : ℝ)
  (h_n : n = 20) (h_l : l = 15.25) (h_required_length : required_length = 248) :
  (required_length = l * n - (n - 1) * 3) :=
by
  sorry

end bandage_overlap_l60_60948


namespace sum_of_first_30_terms_l60_60364

noncomputable def sequence (n : ℕ) : ℕ :=
if n < 1 then 0
else if n = 1 then 1
else if n = 2 then 2
else if (n - 1) % 2 = 1 then sequence (n - 2)
else sequence (n - 2) + 2

lemma sequence_relation (n : ℕ) (h : 1 ≤ n) : 
  sequence (n + 2) - sequence n = 1 + (-1)^n :=
sorry

theorem sum_of_first_30_terms : 
  (∑ n in Finset.range 30, sequence (n+1)) = 255 :=
sorry

end sum_of_first_30_terms_l60_60364


namespace midpoint_distance_eq_four_l60_60720

def hyperbola (x y : ℝ) : Prop := (x^2 / 25) - (y^2 / 9) = 1

def point_on_hyperbola (M : ℝ × ℝ) : Prop :=
  let (x, y) := M in hyperbola x y

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem midpoint_distance_eq_four
  (M : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (O N : ℝ × ℝ)
  (h1 : point_on_hyperbola M)
  (h2 : distance M F₂ = 18)
  (h3 : N = ((fst M + fst F₂) / 2, (snd M + snd F₂) / 2))
  (h4 : O = (0, 0))
  (h5 : F₁ = (-5 * real.sqrt 2, 0))
  (h6 : F₂ = (5 * real.sqrt 2, 0)) :
  distance O N = 4 :=
sorry

end midpoint_distance_eq_four_l60_60720


namespace f_mn_sum_lt_zero_l60_60353

def f : ℝ → ℝ

axiom symmetry_condition (x : ℝ) : f(x) = -f(4 - x)
axiom monotonic_increasing (x : ℝ) (h : x ≤ 2) : ∀ y, y ≤ x → f(y) ≤ f(x)
axiom m_n_conditions (m n : ℝ) : m + n < 4 ∧ m < 2 ∧ n > 2

theorem f_mn_sum_lt_zero (m n : ℝ) (h_mn : m + n < 4) (h_m : m < 2) (h_n : n > 2) :
  f(m) + f(n) < 0 :=
by
  -- Proof goes here
  sorry

end f_mn_sum_lt_zero_l60_60353


namespace find_k_l60_60682

-- Defining the vectors a, b, and c
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (2, -1)

-- Defining the parallel condition
def is_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, v1 = (λ * v2.1, λ * v2.2)

-- The Lean statement we aim to prove
theorem find_k (k : ℝ) :
  let a_plus_kc := (a.1 + k * c.1, a.2 + k * c.2)
      two_b_minus_a := (2 * b.1 - a.1, 2 * b.2 - a.2)
  in a_plus_kc = (3 + 2 * k, 2 - k) ∧ 
     two_b_minus_a = (-5, 2) ∧ 
     is_parallel a_plus_kc two_b_minus_a → k = 16 := 
by
  sorry

end find_k_l60_60682


namespace tan_A_equals_sqrt2_div_4_l60_60680

theorem tan_A_equals_sqrt2_div_4
  (A B : ℝ) 
  (h1 : A + B = Real.pi)
  (h2 : B > Real.pi / 2 ∧ B < Real.pi)
  (h3 : Real.sin B = 1 / 3) : 
  Real.tan A = Real.sqrt 2 / 4 :=
begin
  sorry,
end

end tan_A_equals_sqrt2_div_4_l60_60680


namespace smallest_result_from_process_l60_60534

theorem smallest_result_from_process : 
  ∃ (a b c : ℕ) (S : Finset ℕ), 
  S = {4, 5, 7, 11, 13, 17} ∧ 
  {a, b, c} ⊆ S ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (min ((a + b) * c) ((min ((a + c) * b) ((b + c) * a))) = 48 :=
by {
  use [5, 7, 4, {4, 5, 7, 11, 13, 17}],
  split,
  { refl },
  split,
  { simp },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  sorry
}

end smallest_result_from_process_l60_60534


namespace find_real_medal_min_weighings_l60_60178

axiom has_9_medals : Prop
axiom one_real_medal : Prop
axiom real_medal_heavier : Prop
axiom has_balance_scale : Prop

theorem find_real_medal_min_weighings
  (h1 : has_9_medals)
  (h2 : one_real_medal)
  (h3 : real_medal_heavier)
  (h4 : has_balance_scale) :
  ∃ (minimum_weighings : ℕ), minimum_weighings = 2 := 
  sorry

end find_real_medal_min_weighings_l60_60178


namespace find_k_b_solve_inequality_l60_60352

/-- Definition of the exponential function with parameters k, a, and b. -/
def f (k a b x : ℝ) := (k + 3) * a^x + 3 - b

/-- Problem statement for part 1: values of k and b. -/
theorem find_k_b (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ k b : ℝ, (∀ x : ℝ, f k a b x = a^x) ∧ k = -2 ∧ b = 3) :=
sorry

/-- Problem statement for part 2: solving the inequality based on the value of a. -/
theorem solve_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, (if a > 1 then f (-2) a 3 (2 * x - 7) > f (-2) a 3 (4 * x - 3) ↔ x < -2
                         else 0 < a ∧ a < 1 → (f (-2) a 3 (2 * x - 7) > f (-2) a 3 (4 * x - 3) ↔ x > -2))) :=
sorry

end find_k_b_solve_inequality_l60_60352


namespace find_square_number_divisible_by_six_l60_60287

theorem find_square_number_divisible_by_six :
  ∃ x : ℕ, (∃ k : ℕ, x = k^2) ∧ x % 6 = 0 ∧ 24 < x ∧ x < 150 ∧ (x = 36 ∨ x = 144) :=
by {
  sorry
}

end find_square_number_divisible_by_six_l60_60287


namespace compute_fraction_l60_60262

theorem compute_fraction :
  let a := 1630
  let b := 1623
  let c := 1640
  let d := 1613
  (a^2 - b^2) / (c^2 - d^2) = 7 / 27 :=
by
  let a := 1630
  let b := 1623
  let c := 1640
  let d := 1613
  have num_eq : a^2 - b^2 = 7 * 3253 := sorry
  have denom_eq : c^2 - d^2 = 27 * 3253 := sorry
  calc
    (a^2 - b^2) / (c^2 - d^2) = (7 * 3253) / (27 * 3253) : by rw [num_eq, denom_eq]
                          ... = 7 / 27 : by simp

end compute_fraction_l60_60262


namespace reciprocal_of_neg2_l60_60913

-- Define the number
def num : ℤ := -2

-- Define the reciprocal function
def reciprocal (x : ℤ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_neg2 : reciprocal num = -1 / 2 :=
by
  -- Proof is skipped
  sorry

end reciprocal_of_neg2_l60_60913


namespace geometric_series_condition_l60_60400

theorem geometric_series_condition (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = q * a n) →
  (∀ n, S n = (finset.range n).sum (λ k, a (k + 1))) →
  (q ≠ -1 ↔ ∀ n, ∃ r : ℝ, S (2 * n) - S n = r * S n ∧ S (3 * n) - S (2 * n) = r * (S (2 * n) - S n)) :=
sorry

end geometric_series_condition_l60_60400


namespace part_one_part_two_l60_60731

variable {x : ℝ}

def setA (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 5}
def setB : Set ℝ := {x | -1 / 2 < x ∧ x ≤ 2}

theorem part_one (a : ℝ) (h : a = 1) : setB ⊆ setA a :=
by
  sorry

theorem part_two (a : ℝ) : (setA a ⊆ setB) ↔ (a < -8 ∨ a ≥ 2) :=
by
  sorry

end part_one_part_two_l60_60731


namespace find_apex_angle_of_first_two_cones_l60_60512

theorem find_apex_angle_of_first_two_cones :
  let y := 2 * Real.arctan (Real.sqrt 3 - 1)
  in ∀ (A : Point) 
          (cone1 cone2 : Cone) 
          (cone3 : Cone) 
          (cone4 : Cone),
       cone1.apex = A ∧ cone2.apex = A ∧ cone3.apex = A ∧ cone4.apex = A ∧
       cone1.apex_angle = cone2.apex_angle ∧
       cone3.apex_angle = Real.pi / 3 ∧
       cone4.apex_angle = 5 * Real.pi / 6 ∧
       cone1.touches_cone cone4 ∧
       cone2.touches_cone cone4 ∧
       cone3.touches_cone cone4 ∧
       cone1.touches_cone cone2 ∧
       cone1.touches_cone cone3 ∧
       cone2.touches_cone cone3  
    → cone1.apex_angle = y := 
by sorry

end find_apex_angle_of_first_two_cones_l60_60512


namespace solution_x_2021_l60_60292

theorem solution_x_2021 (x : ℝ) :
  (∑ k in finset.range 21, (x - (2020 - k)) / (k + 1)) = (∑ k in finset.range 21, (x - (k + 1)) / (2020 - k)) ↔
  x = 2021 :=
by
  sorry

end solution_x_2021_l60_60292


namespace count_valid_3x3_grids_l60_60285

-- Define the conditions for the 3x3 grid problem
def is_valid_grid (grid : List (List ℕ)) : Prop :=
  grid.length = 3 ∧ (∀ row, row ∈ grid → row.length = 3) ∧
  (∀ (i : Fin 3), List.chain' (· < ·) (List.map (λ r, r.nthLe i.val sorry) grid)) ∧
  (∀ row, row ∈ grid → List.chain (· < ·) row ∧ List.sorted (· < ·) row) ∧
  List.sort (· < ·) (List.join grid) = [1,2,3,4,5,6,7,8,9]

-- Theorem stating the number of ways to fill the grid under given conditions
theorem count_valid_3x3_grids : 
  ∃! (solutions : Finset (List (List ℕ))), 
    ∀ (g : List (List ℕ)), g ∈ solutions ↔ is_valid_grid g ∧ solutions.card = 42 :=
sorry

end count_valid_3x3_grids_l60_60285


namespace probability_three_red_jellybeans_l60_60971

theorem probability_three_red_jellybeans :
  let total_jellybeans := 12
  let red_jellybeans := 5
  let blue_jellybeans := 2
  let yellow_jellybeans := 5
  let picked_jellybeans := 4
  let combinations (n k : ℕ) := Nat.choose n k
  let total_combinations := combinations total_jellybeans picked_jellybeans
  let successful_red_combinations := combinations red_jellybeans 3
  let non_red_jellybeans := total_jellybeans - red_jellybeans
  let successful_non_red_combinations := combinations non_red_jellybeans 1
  let successful_combinations := successful_red_combinations * successful_non_red_combinations
  let probability := successful_combinations.toReal / total_combinations.toReal
  in probability = (14 : ℚ) / 99 :=
by
  sorry

end probability_three_red_jellybeans_l60_60971


namespace tetrahedron_proj_orthocenter_l60_60392

theorem tetrahedron_proj_orthocenter 
  (T : Type) [EuclideanSpace ℝ T]
  (A B C D : T)
  (h1 : A ≠ B)
  (h2 : B ≠ C)
  (h3 : C ≠ A)
  (h4 : ∀ t : ℝ, A + t • (B - A) ≠ C)
  (perp1 : ∠ (B - A) (C - A) = π / 2)
  (perp2 : ∠ (C - A) (D - A) = π / 2)
  (perp3 : ∠ (D - A) (B - A) = π / 2) :
  let P := proj_to_plane_plane A B C D in
  IsOrthocenter P A B C := 
sorry

end tetrahedron_proj_orthocenter_l60_60392


namespace reciprocal_of_minus_one_half_l60_60121

theorem reciprocal_of_minus_one_half : (1 / (-1 / 2)) = -2 := 
by sorry

end reciprocal_of_minus_one_half_l60_60121


namespace complex_square_l60_60318

variables {a b : ℝ} {i : ℂ}

theorem complex_square (h : a + complex.I = 2 - b * complex.I) :
  (complex.mk a b)^2 = complex.mk 3 (-4) := 
sorry

end complex_square_l60_60318


namespace shopkeeper_oranges_count_l60_60995

theorem shopkeeper_oranges_count
  (O : ℕ)
  (banana_count : ℕ := 400)
  (oranges_in_good_condition : ℝ := 0.85)
  (bananas_in_good_condition : ℝ := 0.95)
  (total_good_percentage : ℝ := 0.89) :
  (0.85 * O + 0.95 * 400) / (O + 400) = 0.89 → O = 600 :=
by
  intro h
  sorry

end shopkeeper_oranges_count_l60_60995


namespace members_not_playing_either_l60_60763

variable (total_members badminton_players tennis_players both_players : ℕ)

theorem members_not_playing_either (h1 : total_members = 40)
                                   (h2 : badminton_players = 20)
                                   (h3 : tennis_players = 18)
                                   (h4 : both_players = 3) :
  total_members - (badminton_players + tennis_players - both_players) = 5 := by
  sorry

end members_not_playing_either_l60_60763


namespace change_5_dollars_l60_60273

-- Definition for the problem
def ways_to_change (amount : ℕ) : ℕ := 
  -- Number of valid ways to change given amount into nickels and dimes with at least one of each coin.
  let total_ways := (finset.range ((amount / 10) - 1)).card -- We use range because Lean's range is exclusive of the upper bound, and (amount / 10) - 1 captures the range up to d < 50 and d ≥ 1
  total_ways

-- Prove that the number of ways to change 500 cents (5 dollars) with at least one nickel and one dime is 49
theorem change_5_dollars : ways_to_change 500 = 49 := 
by
  -- Here we assume the correctness of the problem translation, but leave the proof to be filled in
  sorry

end change_5_dollars_l60_60273


namespace area_circle_AC1D1_eq_3pi_angle_B1C1A_eq_90_volume_prism_eq_3sqrt3_l60_60094

-- Definitions based on conditions
variable (A B C D A1 B1 C1 D1 : point)
variable (r : ℝ := 2) -- radius of the sphere
variable [geometry_class : GeometryField] -- necessary field for geometric operations

-- Given Conditions
-- 1. Rhombus ABCD with BD = 3 and angle ADC = 60 degrees
axiom rhombus_ABCD : rhombus A B C D
axiom BD_eq_3 : distance B D = 3
axiom angle_ADC_60 : angle A D C = 60 * (π / 180)

-- 2. Sphere passing through points D, C, B, B1, A1, D1
axiom sphere_through_points : sphere D C B B1 A1 D1

-- Part (a) - Prove that the area of the circle is 3π
theorem area_circle_AC1D1_eq_3pi : area_circle (plane_through_points A1 C1 D1) = 3 * π := sorry

-- Part (b) - Prove that the angle B1C1A is 90 degrees
theorem angle_B1C1A_eq_90 : angle B1 C1 A = 90 * (π / 180) := sorry

-- Part (c) - Given radius of the sphere, prove that the volume of prism is 3√3
theorem volume_prism_eq_3sqrt3 : volume_prism A B C D A1 B1 C1 D1 = 3 * sqrt 3 := sorry

end area_circle_AC1D1_eq_3pi_angle_B1C1A_eq_90_volume_prism_eq_3sqrt3_l60_60094


namespace count_sets_without_perfect_square_l60_60435

theorem count_sets_without_perfect_square :
  let S := λ i : ℕ, {n : ℤ | 150 * (i : ℤ) ≤ n ∧ n < 150 * (i + 1)} in
  let sets := finset.range 667 in
  let perfect_squares := finset.range 317 |>.image (λ x, x * x : ℤ) in
  let count := sets.filter (λ i, ∀ n ∈ S i, n ∉ perfect_squares) in
  count.card = 2 :=
by
  sorry

end count_sets_without_perfect_square_l60_60435


namespace value_of_80th_number_l60_60864

-- Define a function to determine the value for any given row
def row_value (n : ℕ) : ℕ :=
  3 * n

-- Define a function to calculate the total number of elements up to and including row n
def total_elements_up_to_row (n : ℕ) : ℕ :=
  (finset.range (n + 1)).sum (λ i, (i : ℕ) ^ 3)

theorem value_of_80th_number : ∃ n, total_elements_up_to_row n ≥ 80 ∧ row_value (n + 1) = 12 :=
by
  sorry

end value_of_80th_number_l60_60864


namespace height_after_max_l60_60602

def height (k v m : ℝ) (t : ℝ) : ℝ := -15 * (t - 2) ^ 2 + 150 - k * (v ^ 2) * t / m

theorem height_after_max (k v m : ℝ) : height k v m 4 = 90 - k * (v ^ 2) * 4 / m := by
  sorry

end height_after_max_l60_60602


namespace mother_money_l60_60848

noncomputable theory

def sandra_savings : ℝ := 10
def mother_contribution : ℝ := M
def father_contribution : ℝ := 2 * M
def candy_cost : ℝ := 0.5
def jellybean_cost : ℝ := 0.2
def number_of_candies : ℕ := 14
def number_of_jellybeans : ℕ := 20
def money_left : ℝ := 11

theorem mother_money (M : ℝ) :
  sandra_savings + mother_contribution + father_contribution - 
  (number_of_candies * candy_cost + number_of_jellybeans * jellybean_cost) = money_left  →
  mother_contribution = 4 :=
by
  sorry

end mother_money_l60_60848


namespace infinite_primes_quadratic_non_residue_l60_60809

noncomputable def is_odd (a : ℤ) : Prop :=
  ∃ k : ℤ, a = 2 * k + 1

noncomputable def is_not_perfect_square (a : ℤ) : Prop :=
  ∀ k : ℤ, a ≠ k * k

theorem infinite_primes_quadratic_non_residue (a : ℤ) (h1 : is_odd a) (h2 : is_not_perfect_square a) :
  ∃^∞ p, p.Prime ∧ ¬ is_quadratic_residue a p :=
sorry

end infinite_primes_quadratic_non_residue_l60_60809


namespace parabola_axis_of_symmetry_l60_60745

theorem parabola_axis_of_symmetry (a b : ℝ) (h : a ≠ 0) (hx : (a * -2 + b) = 0) : 
  (y = ax^2 + bx).axis_symmetry = -1 :=
by
  sorry

end parabola_axis_of_symmetry_l60_60745


namespace remainder_div_eq_4_l60_60539

theorem remainder_div_eq_4 {x y : ℕ} (h1 : y = 25) (h2 : (x / y : ℝ) = 96.16) : x % y = 4 := 
sorry

end remainder_div_eq_4_l60_60539


namespace winston_initial_quarters_l60_60545

-- Defining the conditions
def spent_candy := 50 -- 50 cents spent on candy
def remaining_cents := 300 -- 300 cents left

-- Defining the value of a quarter in cents
def value_of_quarter := 25

-- Calculating the number of quarters Winston initially had
def initial_quarters := (spent_candy + remaining_cents) / value_of_quarter

-- Proof statement
theorem winston_initial_quarters : initial_quarters = 14 := 
by sorry

end winston_initial_quarters_l60_60545


namespace product_divisible_by_5_l60_60838

theorem product_divisible_by_5 (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h : ∃ k, a * b = 5 * k) : a % 5 = 0 ∨ b % 5 = 0 :=
by
  sorry

end product_divisible_by_5_l60_60838


namespace sin_cos_beta_and_sin_beta_minus_alpha_l60_60685

noncomputable def sin_alpha : ℝ := -2 / 3
noncomputable def cos_half_beta : ℝ := - (Real.sqrt 14) / 4

axiom alpha_in_interval : α ∈ Set.Ioo π (3 * π / 2)
axiom beta_in_interval : β ∈ Set.Ioo (3 * π / 2) 2π

theorem sin_cos_beta_and_sin_beta_minus_alpha :
  (sin β = - (Real.sqrt 7) / 4) ∧
  (cos β = 3 / 4) ∧
  (sin (β - α) = (Real.sqrt 35 + 6) / 12) :=
sorry

end sin_cos_beta_and_sin_beta_minus_alpha_l60_60685


namespace girl_squirrel_walnuts_added_l60_60208

-- Definitions for the given conditions
def boy_squirrel_walnuts_added : ℕ := 6 - 1
def initial_walnuts_in_burrow : ℕ := 12
def total_after_boy_squirrel : ℕ := initial_walnuts_in_burrow + boy_squirrel_walnuts_added
def final_walnuts (x : ℕ) : ℕ := total_after_boy_squirrel + x - 2

-- The theorem to prove
theorem girl_squirrel_walnuts_added : ∃ (x : ℕ), final_walnuts x = 20 ∧ x = 5 := 
by
  existsi 5
  unfold final_walnuts
  unfold total_after_boy_squirrel
  unfold boy_squirrel_walnuts_added
  unfold initial_walnuts_in_burrow 
  simp
  sorry

end girl_squirrel_walnuts_added_l60_60208


namespace initial_green_mms_l60_60619

variable (G : ℕ) -- initially there are G green M&Ms

-- conditions
variable (H_initial_G : G > 0)
variable (H_red: 20 > 0)
variable (H_eaten_green : 12 > 0)
variable (H_added_yellow : 14 > 0)
variable (H_pick_green : (G - 12).toRat / ((G - 12) + 10 + 14).toRat = 0.25)

theorem initial_green_mms : G = 20 := 
by
  sorry

end initial_green_mms_l60_60619


namespace find_speed_of_man_in_still_water_l60_60185

noncomputable def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
  (v_m + v_s) * 3 = 42 ∧ (v_m - v_s) * 3 = 18

theorem find_speed_of_man_in_still_water (v_s : ℝ) : ∃ v_m : ℝ, speed_of_man_in_still_water v_m v_s ∧ v_m = 10 :=
by
  sorry

end find_speed_of_man_in_still_water_l60_60185


namespace sum_of_2012_terms_l60_60106

def sequence (n : ℕ) : ℝ := 1 / 4 + Real.cos (n * Real.pi / 2)

def partialSum (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k => sequence k)

theorem sum_of_2012_terms : partialSum 2012 = 503 := sorry

end sum_of_2012_terms_l60_60106


namespace difference_max_min_y_l60_60260

theorem difference_max_min_y {total_students : ℕ} (initial_yes_pct initial_no_pct final_yes_pct final_no_pct : ℝ)
  (initial_conditions : initial_yes_pct = 0.4 ∧ initial_no_pct = 0.6)
  (final_conditions : final_yes_pct = 0.8 ∧ final_no_pct = 0.2) :
  ∃ (min_change max_change : ℝ), max_change - min_change = 0.2 := by
  sorry

end difference_max_min_y_l60_60260


namespace reciprocal_of_neg_two_l60_60876

-- Define the condition of reciprocal
def is_reciprocal (y x : ℝ) : Prop := y * x = 1

-- Define the number in question
def y : ℝ := -2

-- Define the expected reciprocal
def x : ℝ := -1 / 2

-- State the theorem
theorem reciprocal_of_neg_two : is_reciprocal y x :=
by
  -- Proof can be provided here
  sorry

end reciprocal_of_neg_two_l60_60876


namespace area_of_region_l60_60830

def satisfies_abs_eqn (x y : ℝ) : Prop :=
  |4 * x| + |3 * y| + |24 - 4 * x - 3 * y| = 24

def in_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ 4 * x + 3 * y ≤ 24

theorem area_of_region :
  (∀ x y : ℝ, satisfies_abs_eqn x y ↔ in_region x y) →
  (∃ (area : ℝ), area = 24) :=
begin
  intro h,
  -- Proof of area calculation, which we don't need to provide
  sorry
end

end area_of_region_l60_60830


namespace class_mean_calculation_correct_l60_60755

variable (s1 s2 : ℕ) (mean1 mean2 : ℕ)
variable (n : ℕ) (mean_total : ℕ)

def overall_class_mean (s1 s2 mean1 mean2 : ℕ) : ℕ :=
  let total_score := (s1 * mean1) + (s2 * mean2)
  total_score / (s1 + s2)

theorem class_mean_calculation_correct
  (h1 : s1 = 40)
  (h2 : s2 = 10)
  (h3 : mean1 = 80)
  (h4 : mean2 = 90)
  (h5 : n = 50)
  (h6 : mean_total = 82) :
  overall_class_mean s1 s2 mean1 mean2 = mean_total :=
  sorry

end class_mean_calculation_correct_l60_60755


namespace therapy_hours_l60_60977

theorem therapy_hours (x n : ℕ) : 
  (x + 30) + 2 * x = 252 → 
  104 + (n - 1) * x = 400 → 
  x = 74 → 
  n = 5 := 
by
  sorry

end therapy_hours_l60_60977


namespace negation_proposition_l60_60113

theorem negation_proposition :
  (∀ x : ℝ, |x - 2| + |x - 4| > 3) = ¬(∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) :=
  by sorry

end negation_proposition_l60_60113


namespace sqrt_four_squared_l60_60171

theorem sqrt_four_squared : (Real.sqrt 4) ^ 2 = 4 :=
  by
    sorry

end sqrt_four_squared_l60_60171


namespace strawberry_quality_meets_standard_l60_60499

def acceptable_weight_range (w : ℝ) : Prop :=
  4.97 ≤ w ∧ w ≤ 5.03

theorem strawberry_quality_meets_standard :
  acceptable_weight_range 4.98 :=
by
  sorry

end strawberry_quality_meets_standard_l60_60499


namespace pam_bags_count_l60_60828

noncomputable def geralds_bag_apples : ℕ := 40

noncomputable def pams_bag_apples := 3 * geralds_bag_apples

noncomputable def pams_total_apples : ℕ := 1200

theorem pam_bags_count : pams_total_apples / pams_bag_apples = 10 := by 
  sorry

end pam_bags_count_l60_60828


namespace total_heads_of_cabbage_l60_60248

-- Problem definition for the first patch
def first_patch : ℕ := 12 * 15

-- Problem definition for the second patch
def second_patch : ℕ := 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24

-- Problem statement
theorem total_heads_of_cabbage : first_patch + second_patch = 316 := by
  sorry

end total_heads_of_cabbage_l60_60248


namespace minimum_cannons_needed_l60_60240

theorem minimum_cannons_needed
  (p : ℝ := 0.8) : (∃ n : ℕ, 1 - (1 - p)^n ≥ 0.99 ∧ ∀ m : ℕ, m < n → 1 - (1 - p)^m < 0.99) :=
by
  have h : ∃ n : ℕ, 1 - (1 - p)^n ≥ 0.99 ∧ ∀ m : ℕ, m < n → 1 - (1 - p)^m < 0.99,
  {
    use 3,
    sorry -- Proof confirming that n=3 is the minimum integer satisfying the inequality
  }
  exact h

end minimum_cannons_needed_l60_60240


namespace probability_inequality_l60_60081

open ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {P : MeasureTheory.Measure Ω}

def symm_diff (B C : Set Ω) : Set Ω := (B \ C) ∪ (C \ B)

theorem probability_inequality (A B C : Set Ω) :
  |P (A ∩ B) - P (A ∩ C)| ≤ P (symm_diff B C) := 
sorry

end probability_inequality_l60_60081


namespace reciprocal_of_neg_two_l60_60882

theorem reciprocal_of_neg_two : ∃ x : ℝ, (-2) * x = 1 ∧ x = -1/2 :=
by
  use -1/2
  split
  · exact (by norm_num : (-2) * (-1/2) = 1)
  · rfl

end reciprocal_of_neg_two_l60_60882


namespace evaluate_expression_l60_60276

theorem evaluate_expression : 
  ∃ q : ℤ, ∀ (a : ℤ), a = 2022 → (2023 : ℚ) / 2022 - (2022 : ℚ) / 2023 = 4045 / q :=
by
  sorry

end evaluate_expression_l60_60276


namespace parallel_planes_of_lines_parallel_l60_60697

open Plane Line

variables (m n l1 l2 : Line) (α β : Plane) (M : Point)
variables (hm : m ∈ α) (hn : n ∈ α)
variables (hl1 : l1 ∈ β) (hl2 : l2 ∈ β)
variable (hl1_intersect_l2 : ∃ M, M ∈ l1 ∧ M ∈ l2)
variable (h_parallel_m_l1 : m ∥ l1)
variable (h_parallel_n_l2 : n ∥ l2)

theorem parallel_planes_of_lines_parallel :
  α ∥ β :=
sorry

end parallel_planes_of_lines_parallel_l60_60697


namespace angle_BOC_eq_angle_AOD_l60_60048

open Point Line Segment Angle ConvexQuadrilateral

variables {A B C D E F P O : Point}
variable {ABCD : ConvexQuadrilateral A B C D}

namespace Geometry

axiom given_conditions :
  ∃ (E F : Point), OppositeSidesIntersect ABCD E F ∧
  (AC ∩ BD = P) ∧
  (Projection P (Line E F) = O)

theorem angle_BOC_eq_angle_AOD :
  ∃ E F : Point, OppositeSidesIntersect ABCD E F ∧
  (AC ∩ BD = P) ∧
  (Projection P (Line E F) = O) →
  angle B O C = angle A O D :=
by {
  intro h,
  cases h with E hEF,
  cases hEF with F hOPP,
  cases hOPP with hEOS hBDP,
  cases hBDP with hIntersection hProj,
  -- Proof steps would go here
  sorry
}

end Geometry

end angle_BOC_eq_angle_AOD_l60_60048


namespace range_of_a_l60_60486

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ)
  (h_cond : ∀ (n : ℕ), n > 0 → (a_seq n = if n ≤ 4 then 2^n - 1 else -n^2 + (a - 1) * n))
  (h_max_a5 : ∀ (n : ℕ), n > 0 → a_seq n ≤ a_seq 5) :
  9 ≤ a ∧ a ≤ 12 := 
by
  sorry

end range_of_a_l60_60486


namespace boriss_minimum_bailing_rate_l60_60177

-- Define initial conditions
def boat_distance_from_shore : ℝ := 2
def leak_rate_gpm : ℝ := 15
def max_gallons_before_sinking : ℝ := 50
def initial_speed_mph : ℝ := 2
def speed_increment_mph_half_hour : ℝ := 1
def speed_increment_time_hours : ℝ := 0.5

-- Define time calculation based on Amy's rowing pattern
def time_to_reach_shore_in_hours : ℝ :=
  let first_leg_time := speed_increment_time_hours
  let first_leg_distance := initial_speed_mph * speed_increment_time_hours
  let remaining_distance := boat_distance_from_shore - first_leg_distance
  let second_leg_speed := initial_speed_mph + speed_increment_mph_half_hour
  let second_leg_time := remaining_distance / second_leg_speed
  first_leg_time + second_leg_time

def time_to_reach_shore_in_minutes : ℝ := time_to_reach_shore_in_hours * 60

-- Define total water intake calculation
def total_water_intake_gallons_boriss_rate (boriss_rate : ℝ) : ℝ :=
  (leak_rate_gpm - boriss_rate) * time_to_reach_shore_in_minutes

-- State the proof goal
theorem boriss_minimum_bailing_rate : 
  ∃ (r : ℝ), r >= 14 ∧ total_water_intake_gallons_boriss_rate(r) <= max_gallons_before_sinking :=
begin
  let r := 14,
  use r,
  split,
  { linarith }, -- r >= 14
  { sorry } -- total_water_intake_gallons_boriss_rate(r) <= 50
end

end boriss_minimum_bailing_rate_l60_60177


namespace right_scale_at_least_left_l60_60419

noncomputable def are_balanced {α : Type*} [AddCommMonoid α] (left right : multiset α) : Prop := 
  left.sum = right.sum

theorem right_scale_at_least_left {α : Type*} [AddCommMonoid α] (left right : multiset α) 
  (h_distinct: multiset.nodup left)
  (h_balanced : are_balanced left right) : 
  right.card ≥ left.card := 
sorry

end right_scale_at_least_left_l60_60419


namespace find_least_positive_integer_l60_60524

def least_positive_integer_leaving_remainder_two (m : ℕ) : Prop :=
  m > 1 ∧ ∀ d ∈ {2, 3, 4, 5, 6}, (m - 2) % d = 0

theorem find_least_positive_integer : ∃ m : ℕ, least_positive_integer_leaving_remainder_two m ∧ m = 62 :=
by
  use 62
  unfold least_positive_integer_leaving_remainder_two
  split
  {
    exact dec_trivial,
  }
  {
    intro d
    intro h
    fin_cases h;
    all_goals
    {
      norm_num
    }
  }
  sorry

end find_least_positive_integer_l60_60524


namespace determine_k_completed_square_l60_60002

theorem determine_k_completed_square (x : ℝ) :
  ∃ (a h k : ℝ), a * (x - h)^2 + k = x^2 - 7 * x ∧ k = -49/4 := sorry

end determine_k_completed_square_l60_60002


namespace find_symbols_l60_60537

theorem find_symbols (x y otimes oplus : ℝ) 
  (h1 : x + otimes * y = 3) 
  (h2 : 3 * x - otimes * y = 1) 
  (h3 : x = oplus) 
  (h4 : y = 1) : 
  otimes = 2 ∧ oplus = 1 := 
by
  sorry

end find_symbols_l60_60537


namespace cultural_shirts_proof_l60_60200

variables (x y m : ℝ)

-- Define the conditions from the problem
def condition1 := 3 * x + 2 * y = 190
def condition2 := 5 * x + y = 235
def condition3 := 1000 - m -- the number of shirts purchased
def condition4 := m >= 3 * (1000 - m)

-- Define the assertions to prove
def unit_price_A := x = 40
def unit_price_B := y = 35
def min_cost_purchasing_plan := m = 750 ∧ 1000 - m = 250

-- The main theorem combining conditions and what needs to be proved
theorem cultural_shirts_proof :
  (condition1 ∧ condition2 ∧ condition3 ∧ condition4) →
  (unit_price_A ∧ unit_price_B ∧ min_cost_purchasing_plan) :=
by {
  -- We include sorry here to denote that we are skipping the actual proof steps
  sorry
}

end cultural_shirts_proof_l60_60200


namespace sum_m_values_for_minimal_area_l60_60500

-- Define the coordinates of the points
def point1 : (ℝ × ℝ) := (2, 8)
def point2 : (ℝ × ℝ) := (14, 17)
def pointX : ℝ := 6

-- Define the proof statement
theorem sum_m_values_for_minimal_area : 
  ∃ (m1 m2 : ℤ), m1 ≠ m2 ∧
    let area1 := abs (pointX * (point1.2 - m1) + 2 * (m1 - 17) + 14 * (8 - point1.2)) / 2 in
    let area2 := abs (pointX * (point1.2 - m2) + 2 * (m2 - 17) + 14 * (8 - point1.2)) / 2 in
    area1 ≠ 0 ∧ area2 ≠ 0 ∧ (area1 = area2) ∧ (m1 + m2 = 16) := 
  sorry

end sum_m_values_for_minimal_area_l60_60500


namespace break_even_machines_l60_60629

theorem break_even_machines (cost_parts cost_patent machine_price total_costs : ℝ):
    cost_parts = 3600 ∧ cost_patent = 4500 ∧ machine_price = 180 ∧ total_costs = cost_parts + cost_patent → 
    total_costs / machine_price = 45 :=
by
  intros h
  cases h with h1 h'
  cases h' with h2 h''
  cases h'' with h3 h4
  sorry

end break_even_machines_l60_60629


namespace LCM_of_numbers_l60_60158

theorem LCM_of_numbers : Nat.lcm (8, Nat.lcm (24, Nat.lcm (36, Nat.lcm (54, Nat.lcm (42, Nat.lcm (51, Nat.lcm (64, 87))))))) = 5963328 := by
  sorry

end LCM_of_numbers_l60_60158


namespace part_I_part_II_l60_60324

theorem part_I (a b : ℝ) (z : ℂ) (h1 : z = a + b * complex.I) (h2 : (z^2).im = 0) (h3 : ∥z + 1∥ = real.sqrt 2) : b = 1 :=
sorry

theorem part_II (a b : ℤ) (h1 : a ∈ {-1, -2, 0, 1}) (h2 : b ∈ {1, 2, 3}) :
  let event_A := (a < 0 ∧ b > 0)
    in (finset.filter (λ ab : ℤ × ℤ, event_A) (finset.product {-1, -2, 0, 1} {1, 2, 3})).card = 6
    ∧ (finset.product {-1, -2, 0, 1} {1, 2, 3}).card = 12 
    ∧  6 / 12 = 1 / 2 :=
sorry

end part_I_part_II_l60_60324


namespace normal_distribution_probability_number_of_students_within_60_and_120_l60_60343

-- Define the normal distribution characteristic function
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
(1 / (σ * sqrt (2 * Math.pi))) * exp (-(x - μ) ^ 2 / (2 * σ ^ 2))

-- Define the CDF for the normal distribution
noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

-- Define the probability for the specified intervals
theorem normal_distribution_probability
  (μ σ : ℝ) (P_of_interval : ℕ → ℝ)
  (h₁ : P_of_interval 1 = 0.683)
  (h₂ : P_of_interval 2 = 0.954)
  (h₃ : P_of_interval 3 = 0.997) : 
  ∀ k, P_of_interval k = ∫ (x : ℝ) in (μ - k * σ)..(μ + k * σ), normal_pdf μ σ x := sorry

-- Define the problem-specific theorem
theorem number_of_students_within_60_and_120
  (N : ℕ) (score_mean score_variance : ℝ) (prob_within_2_std_dev : ℝ)
  (h₁ : score_mean = 90)
  (h₂ : score_variance = 15^2)
  (h₃ : prob_within_2_std_dev = 0.954)
  (hN : N = 1000) :
  ∃ (count : ℕ), count = 954 :=
begin
  use (N * prob_within_2_std_dev).to_nat,
  have h_prob_calculation : (N * prob_within_2_std_dev).to_nat = 954,
  { 
    calc (N * prob_within_2_std_dev).to_nat 
      = (1000 * 0.954).to_nat : by rw [hN, h₃]
    ... = 954 : by norm_num },
  exact h_prob_calculation,
end

end normal_distribution_probability_number_of_students_within_60_and_120_l60_60343


namespace find_a2015_l60_60563

def seq (a : ℕ → ℕ) :=
  (a 1 = 1) ∧
  (a 2 = 4) ∧
  (a 3 = 9) ∧
  (∀ n, 4 ≤ n → a n = a (n-1) + a (n-2) - a (n-3))

theorem find_a2015 (a : ℕ → ℕ) (h_seq : seq a) : a 2015 = 8057 :=
sorry

end find_a2015_l60_60563


namespace percentage_difference_l60_60480

variables (G P R : ℝ)

-- Conditions
def condition1 : Prop := P = 0.9 * G
def condition2 : Prop := R = 3.0000000000000006 * G

-- Theorem to prove
theorem percentage_difference (h1 : condition1 P G) (h2 : condition2 R G) : 
  (R - P) / R * 100 = 70 :=
sorry

end percentage_difference_l60_60480


namespace greatest_integer_le_100x_l60_60814

def sum_cos (n : ℕ) : ℝ :=
Σ k in Finset.range n, Real.cos (k + 1 : ℝ * Real.pi / 180)

def sum_sin (n : ℕ) : ℝ :=
Σ k in Finset.range n, Real.sin (k + 1 : ℝ * Real.pi / 180)

def x : ℝ := (sum_cos 60) / (sum_sin 60)

theorem greatest_integer_le_100x : ⌊100 * x⌋ = 173 :=
by
  sorry

end greatest_integer_le_100x_l60_60814


namespace additional_area_l60_60233

-- Define the original and increased rope lengths
def r1 : ℝ := 9
def r2 : ℝ := 23

-- Define the area function for a circle
def area (r : ℝ) : ℝ := Real.pi * r ^ 2

-- State the proof problem
theorem additional_area : area r2 - area r1 = 448 * Real.pi := 
sorry

end additional_area_l60_60233


namespace reciprocal_of_neg_two_l60_60906

variable a : ℤ

theorem reciprocal_of_neg_two (h : a = -2) : ∃ b : ℚ, a * b = 1 ∧ b = -1 / 2 := 
by
  exists (-1 / 2)
  rw [h]
  split
  · norm_num
  · refl

end reciprocal_of_neg_two_l60_60906


namespace AD_plus_CE_eq_AE_l60_60748

open Real

-- Definitions from the conditions
variables {A B C D E : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables (triangle_ABC : triangle ℝ A B C)
variables (angle_acb : ∠ACB = π / 2)
variables (D_is_foot : foot D C A B)
variables (E_on_BC : ∃ E : segment B C, ∃ x : ℝ, 0 < x ∧ x < 1 ∧ CE = BD / 2)

-- Goal: To prove
theorem AD_plus_CE_eq_AE : AD + CE = AE :=
sorry

end AD_plus_CE_eq_AE_l60_60748


namespace ratio_largest_to_sum_l60_60267

theorem ratio_largest_to_sum : 
  let s := {1, 5, 5^2, 5^3, 5^4, 5^5, 5^6, 5^7, 5^8, 5^9, 5^{10}} in
  let largest := 5^10 in
  let sum_others := (1 + 5 + 5^2 + 5^3 + 5^4 + 5^5 + 5^6 + 5^7 + 5^8 + 5^9) in
  (largest : ℝ) / (sum_others : ℝ) = 4 :=
by
  have h_largest : largest = 5^10 := rfl
  have h_sum_others : sum_others = (1 + 5 + 5^2 + 5^3 + 5^4 + 5^5 + 5^6 + 5^7 + 5^8 + 5^9) := rfl
  sorry

end ratio_largest_to_sum_l60_60267


namespace correct_m_n_sum_hexagon_area_correct_mn_sum_l60_60581

noncomputable def hexagon_area_proof : Prop :=
  let s := 3
  let area_triangle : ℝ := (sqrt 3 / 4) * s^2
  let total_area := 6 * area_triangle
  total_area = sqrt 729 + sqrt 27

theorem correct_m_n_sum : (729 + 27 = 756) :=
  sorry

theorem hexagon_area_correct_mn_sum (h : hexagon_area_proof) : 729 + 27 = 756 :=
  correct_m_n_sum

end correct_m_n_sum_hexagon_area_correct_mn_sum_l60_60581


namespace jason_initial_cards_l60_60028

theorem jason_initial_cards (cards_sold : Nat) (cards_after_selling : Nat) (initial_cards : Nat) 
  (h1 : cards_sold = 224) 
  (h2 : cards_after_selling = 452) 
  (h3 : initial_cards = cards_after_selling + cards_sold) : 
  initial_cards = 676 := 
sorry

end jason_initial_cards_l60_60028


namespace tangent_condition_l60_60199

theorem tangent_condition (a : ℝ) :
  (4 - Real.sqrt 30 < a ∧ a < 4 + Real.sqrt 30) ↔
  (∃ x y : ℝ, (x, y) ∈ { p : ℝ × ℝ | 2 * p.1 - p.2 = 1 } ∧ (x, y) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * a * p.1 - 2 * p.2 + 3 = 0 }) :=
begin
  -- The hard work goes here
  sorry
end

end tangent_condition_l60_60199


namespace greatest_common_multiple_of_3_and_5_in_three_digit_range_is_990_l60_60544

theorem greatest_common_multiple_of_3_and_5_in_three_digit_range_is_990 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∃ m : ℕ, n = 15 * m) ∧ (∀ k : ℕ, 100 ≤ k ∧ k < 1000 ∧ (∃ l : ℕ, k = 15 * l) → k ≤ n) :=
begin
  use 990,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { use 66,
    norm_num,
  },
  { intros k hk,
    obtain ⟨l, rfl⟩ := hk.2,
    have hkl : l ≤ 66,
    { linarith },
    refine mul_le_mul_right' hkl 15,
  },
end

end greatest_common_multiple_of_3_and_5_in_three_digit_range_is_990_l60_60544


namespace remainder_div_197_l60_60536

theorem remainder_div_197 (x q : ℕ) (h_pos : 0 < x) (h_div : 100 = q * x + 3) : 197 % x = 3 :=
sorry

end remainder_div_197_l60_60536


namespace max_value_3xy_sqrt3_plus_9yz_l60_60441

theorem max_value_3xy_sqrt3_plus_9yz (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * real.sqrt 3 + 9 * y * z ≤ real.sqrt 255 := 
sorry

end max_value_3xy_sqrt3_plus_9yz_l60_60441


namespace sum_alpha_beta_pi_div_4_l60_60337

noncomputable def cosAlpha := 2 * sqrt 5 / 5
noncomputable def sinBeta := sqrt 10 / 10
variable (alpha β : ℝ)

axiom h1 : cos alpha = cosAlpha
axiom h2 : sin β = sinBeta
axiom h3 : 0 < alpha ∧ alpha < π / 2
axiom h4 : 0 < β ∧ β < π / 2

theorem sum_alpha_beta_pi_div_4 : alpha + β = π / 4 :=
by sorry

end sum_alpha_beta_pi_div_4_l60_60337


namespace minimum_value_inequality_l60_60338

theorem minimum_value_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → (2 / x + 1 / y) ≥ 9)) :=
by
  -- skipping the proof
  sorry

end minimum_value_inequality_l60_60338


namespace cost_per_serving_l60_60846

def pasta_cost : ℝ := 1.0
def sauce_cost : ℝ := 2.0
def meatballs_cost : ℝ := 5.0
def total_servings : ℝ := 8.0

theorem cost_per_serving : (pasta_cost + sauce_cost + meatballs_cost) / total_servings = 1.0 :=
by sorry

end cost_per_serving_l60_60846


namespace mathe_matics_equals_2014_l60_60841

/-- 
Given the following mappings for characters in the word "MATHEMATICS":
M = 1, A = 8, T = 3, E = '+', I = 9, K = '-',
verify that the resulting numerical expression 183 + 1839 - 8 equals 2014.
-/
theorem mathe_matics_equals_2014 :
  183 + 1839 - 8 = 2014 :=
by
  sorry

end mathe_matics_equals_2014_l60_60841


namespace car_city_mileage_l60_60974

theorem car_city_mileage:
  ∀ (miles_highway_per_tankful tank_size miles_per_gallon_city miles_per_gallon_highway: ℝ),
  miles_highway_per_tankful = 448 →
  miles_per_gallon_city = 18 →
  miles_per_gallon_highway = miles_per_gallon_city + 6 →
  tank_size = miles_highway_per_tankful / miles_per_gallon_highway →
  miles_per_gallon_city * tank_size = 336.06 :=
by
  assume miles_highway_per_tankful tank_size miles_per_gallon_city miles_per_gallon_highway,
  assume h1 h2 h3 h4,
  sorry

end car_city_mileage_l60_60974


namespace total_eggs_found_l60_60788

def eggs_club_house := 12
def eggs_park := 5
def eggs_town_hall_garden := 3

theorem total_eggs_found : eggs_club_house + eggs_park + eggs_town_hall_garden = 20 :=
by
  sorry

end total_eggs_found_l60_60788


namespace function_properties_l60_60357

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a ((x + 1) / (x - 1))

theorem function_properties (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  (∀ x, x ∈ ((-∞, -1) ∪ (1, ∞)) ↔ f a x ∈ ℝ) ∧
  (∀ x, f a (-x) = -f a x) ∧
  (a > 1 → (∀ x, x ∈ (-∞, -1) ∨ x ∈ (1, ∞) → f a x < 0)) ∧
  (0 < a ∧ a < 1 → (∀ x, x ∈ (-∞, -1) ∨ x ∈ (1, ∞) → f a x > 0)) :=
by
  sorry

end function_properties_l60_60357


namespace sum_of_prime_factors_510_l60_60156

-- Define the sum of prime factors function.
def sum_of_prime_factors (n : ℕ) : ℕ :=
  if h : n > 0 then
    let factors := Multiset.toFinset ((uniqueFactorizationMonoid.factors n).val.map (λ p, if is_prime p then p else 0))
    factors.sum
  else 0

-- The statement to prove the sum of prime factors of 510 is 27.
theorem sum_of_prime_factors_510 : sum_of_prime_factors 510 = 27 :=
sorry

end sum_of_prime_factors_510_l60_60156


namespace combined_tax_rate_l60_60191

theorem combined_tax_rate
  (john_income : ℝ) (john_tax_rate : ℝ)
  (ingrid_income : ℝ) (ingrid_tax_rate : ℝ)
  (h_john_income : john_income = 58000)
  (h_john_tax_rate : john_tax_rate = 0.30)
  (h_ingrid_income : ingrid_income = 72000)
  (h_ingrid_tax_rate : ingrid_tax_rate = 0.40) :
  ((john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income)) = 0.3553846154 :=
by
  sorry

end combined_tax_rate_l60_60191


namespace man_speed_with_stream_is_4_l60_60222

noncomputable def man's_speed_with_stream (Vm Vs : ℝ) : ℝ := Vm + Vs

theorem man_speed_with_stream_is_4 (Vm : ℝ) (Vs : ℝ) 
  (h1 : Vm - Vs = 4) 
  (h2 : Vm = 4) : man's_speed_with_stream Vm Vs = 4 :=
by 
  -- The proof is omitted as per instructions
  sorry

end man_speed_with_stream_is_4_l60_60222


namespace faye_total_crayons_l60_60283

-- Define the number of rows and the number of crayons per row as given conditions.
def num_rows : ℕ := 7
def crayons_per_row : ℕ := 30

-- State the theorem we need to prove.
theorem faye_total_crayons : (num_rows * crayons_per_row) = 210 :=
by
  sorry

end faye_total_crayons_l60_60283


namespace angle_A_find_c_and_area_l60_60749

theorem angle_A (a b c : ℝ) (A B C : ℝ) 
  (ha : a = 6)
  (h_parallel : (cos^2 (A/2), 1) = (cos^2 (B+C), 1)) :
  A = 120 :=
by sorry

theorem find_c_and_area (a b c : ℝ) (S : ℝ)
  (ha : a = 6)
  (h_area : sqrt 3 = (a^2 + b^2 - c^2) / (4 * S))
  (hA : degrees A = 120)
  (hC : degrees C = 30) :
  c = 2 * sqrt 3 ∧ S = 3 * sqrt 3 :=
by sorry

end angle_A_find_c_and_area_l60_60749


namespace martian_half_circle_clerts_l60_60059

theorem martian_half_circle_clerts (full_circle_clerts : ℕ) (h : full_circle_clerts = 600) : full_circle_clerts / 2 = 300 :=
by {
  rw h,
  norm_num,
  sorry,
}

end martian_half_circle_clerts_l60_60059


namespace calculate_sum_l60_60670

noncomputable def g (n : ℕ) : ℝ := real.logb 1001 (n^2)

theorem calculate_sum : g 7 + g 11 + g 13 = 2 := by
  sorry

end calculate_sum_l60_60670


namespace prod_sum_inequality_l60_60460

theorem prod_sum_inequality {n : ℕ} (a : Fin n → ℝ) (h₀ : ∀ i, a i > -1)
  (h₁ : ∀ i j, (0 < a i ∧ 0 < a j) ∨ (0 ≥ a i ∧ 0 ≥ a j)) :
  (∏ i, (1 + a i)) ≥ (1 + ∑ i, a i) :=
sorry

end prod_sum_inequality_l60_60460


namespace correct_function_is_C_l60_60162

def f_A (x : ℝ) : ℝ := (x^2 - 2 * x) / (x - 2)
def f_B (x : ℝ) : ℝ := x - (1 / x)
def f_C (x : ℝ) : ℝ := 2^x - 2^(-x)
def f_D (x : ℝ) : ℝ := x * abs (sin x)

theorem correct_function_is_C : 
  (∀ x, f_C (-x) = -f_C x) ∧ (∀ x y, x < y → f_C x < f_C y) :=
  sorry

end correct_function_is_C_l60_60162


namespace upper_set_join_inter_lower_set_meet_inter_l60_60203

-- Definitions for upper set and lower set (as they might be relevant):
def is_upper_set (B : Set (Set α)) : Prop :=
  ∀ ⦃x y : Set α⦄, x ∈ B → x ⊆ y → y ∈ B

def is_lower_set (A : Set (Set α)) : Prop :=
  ∀ ⦃x y : Set α⦄, x ∈ A → y ⊆ x → y ∈ A

-- Problem (i): Given that B is an upper set, prove A ⋁ B = A ∩ B
theorem upper_set_join_inter (A B : Set (Set α)) (hB : is_upper_set B) :
  A ∨ B = A ∩ B :=
by sorry

-- Problem (ii): Given that A (or B) is a lower set, prove A ⋀ B = A ∩ B
theorem lower_set_meet_inter (A B : Set (Set α)) (hA : is_lower_set A) :
  A ∧ B = A ∩ B :=
by sorry

end upper_set_join_inter_lower_set_meet_inter_l60_60203


namespace profit_function_profitable_range_maximize_profit_l60_60224

-- Definitions from conditions
def total_cost (x : ℝ) : ℝ := 2.8 + x

def sales_revenue (x : ℝ) : ℝ :=
  if x ≤ 5 then -0.4 * x^2 + 4.2 * x else 11

def profit (x : ℝ) : ℝ := sales_revenue x - total_cost x

-- Problem (1): Proving the expression for the profit function
theorem profit_function (x : ℝ) :
  profit x =
  (if x ≤ 5 then -0.4 * x^2 + 3.2 * x - 2.8 else 8.2 - x) := by
  sorry

-- Problem (2): Proving the range of x for profitability
theorem profitable_range (x : ℝ) :
  (1 < x ∧ x < 8.2) ↔ profit x > 0 := by
  sorry

-- Problem (3): Proving the production quantity to maximize profit
theorem maximize_profit : ∀ x, (0 ≤ x ∧ x ≤ 5) → 
  profit 4 ≥ profit x := by
  sorry

end profit_function_profitable_range_maximize_profit_l60_60224


namespace reciprocal_neg_half_l60_60118

theorem reciprocal_neg_half : 1 / (-1 / 2 : ℝ) = (-2 : ℝ) :=
by
  sorry

end reciprocal_neg_half_l60_60118


namespace reciprocal_of_neg_two_l60_60874

-- Define the condition of reciprocal
def is_reciprocal (y x : ℝ) : Prop := y * x = 1

-- Define the number in question
def y : ℝ := -2

-- Define the expected reciprocal
def x : ℝ := -1 / 2

-- State the theorem
theorem reciprocal_of_neg_two : is_reciprocal y x :=
by
  -- Proof can be provided here
  sorry

end reciprocal_of_neg_two_l60_60874


namespace negation_equiv_l60_60495

theorem negation_equiv (a b : ℝ) : ¬ (a > b → 2^a > 2^b - 1) ↔ (a ≤ b → 2^a ≤ 2^b - 1) :=
by 
  sorry

end negation_equiv_l60_60495


namespace shade_half_grid_additional_squares_l60_60412

/-- A 4x5 grid consists of 20 squares, of which 3 are already shaded. 
Prove that the number of additional 1x1 squares needed to shade half the grid is 7. -/
theorem shade_half_grid_additional_squares (total_squares shaded_squares remaining_squares: ℕ) 
  (h1 : total_squares = 4 * 5)
  (h2 : shaded_squares = 3)
  (h3 : remaining_squares = total_squares / 2 - shaded_squares) :
  remaining_squares = 7 :=
by
  -- Proof not required.
  sorry

end shade_half_grid_additional_squares_l60_60412


namespace correct_operation_among_options_l60_60166

theorem correct_operation_among_options (A B C D : Prop) (cond_A : A = (sqrt 4 = ±2))
  (cond_B : B = (sqrt 4)^2 = 4) (cond_C : C = (sqrt (-4)^2) = -4) (cond_D : D = (-sqrt 4)^2 = -4) :
  B ∧ ¬A ∧ ¬C ∧ ¬D :=
by
  sorry

end correct_operation_among_options_l60_60166


namespace integral_f_eq_pi_l60_60053

noncomputable def f (x : ℝ) : ℝ := sin x ^ 5 + 1

theorem integral_f_eq_pi :
  ∫ x in -((π : ℝ) / 2)..((π : ℝ) / 2), f x = π := by
  sorry

end integral_f_eq_pi_l60_60053


namespace total_toys_l60_60029

-- Definitions based on the conditions
def Jaxon_toys : ℕ := 15
def Gabriel_toys : ℕ := 2 * Jaxon_toys
def Jerry_toys : ℕ := Gabriel_toys + 8

-- Problem statement to be proved
theorem total_toys : Jerry_toys + Gabriel_toys + Jaxon_toys = 83 :=
by
  have gabriel_toys_correct : Gabriel_toys = 30 := by
    unfold Gabriel_toys
    rw [Jaxon_toys]
    norm_num

  have jerry_toys_correct : Jerry_toys = 38 := by
    unfold Jerry_toys
    rw [gabriel_toys_correct]
    norm_num

  rw [jerry_toys_correct, gabriel_toys_correct]
  norm_num
  sorry

end total_toys_l60_60029


namespace number_of_valid_lines_l60_60014

noncomputable def countValidLinesThroughPoint (P: ℝ × ℝ) : ℕ :=
  Nat.card { (a : ℕ) × (b : ℕ) | prime b ∧ (P.1 : ℝ) / (a : ℝ) + (P.2 : ℝ) / (b : ℝ) = 1 }

theorem number_of_valid_lines :
  countValidLinesThroughPoint (5, 4) = 1 :=
sorry

end number_of_valid_lines_l60_60014


namespace fraction_raised_to_zero_l60_60523

theorem fraction_raised_to_zero:
  (↑(-4305835) / ↑1092370457 : ℚ)^0 = 1 := 
by
  sorry

end fraction_raised_to_zero_l60_60523


namespace f_n_roots_l60_60798

noncomputable def f1 (x : ℝ) : ℝ := sorry

def f : ℕ → (ℝ → ℝ)
| 0 => f1
| (n+1) => λ x => f1 (f n x)

theorem f_n_roots (f1 : ℝ → ℝ) (h_f1_poly_deg2 : ∀ x, ∃ a b c : ℝ, a > 0 ∧ f1 x = a * x^2 + b * x + c)
  (f : ∀ n, ℝ → ℝ) (h_fn_rec : ∀ n x, f (n+1) x = f1 (f n x))
  (h_f2_roots : ∃ x1 x2 x3 x4 : ℝ, x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ f 1 x1 = 0 ∧ f 1 x2 = 0 ∧ f 1 x3 = 0 ∧ f 1 x4 = 0 ∧ x1 ≤ 0 ∧ x4 ≤ 0) :
  ∀ n, ∃ S : finset ℝ, S.card = 2^n ∧ ∀ x ∈ S, f n x = 0 :=
sorry

end f_n_roots_l60_60798


namespace highest_probability_white_ball_l60_60396

theorem highest_probability_white_ball :
  let red_balls := 2
  let black_balls := 3
  let white_balls := 4
  let total_balls := red_balls + black_balls + white_balls
  let prob_red := red_balls / total_balls
  let prob_black := black_balls / total_balls
  let prob_white := white_balls / total_balls
  prob_white > prob_black ∧ prob_black > prob_red :=
by
  sorry

end highest_probability_white_ball_l60_60396


namespace find_p_and_q_l60_60422

-- Definitions needed for the conditions to hold
def parabola (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q
def line (x : ℝ) : ℝ := 2 * x - 3
def point_of_tangency (x y: ℝ) : Prop := (x = 2) ∧ (y = 1)

-- Hypotheses needed to be satisfied
def tangent_point (p q : ℝ) : Prop :=
  (parabola 2 p q = 1) ∧ ((deriv (λ x, parabola x p q) 2) = 2)

-- The problem statement we aim to prove
theorem find_p_and_q : ∃ (p q : ℝ), (tangent_point p q) → (p = -2 ∧ q = 1) :=
by {
  sorry
}

end find_p_and_q_l60_60422


namespace curve_standard_equation_range_of_2x_plus_y_l60_60201

noncomputable def curve_parametric (φ : ℝ) : ℝ × ℝ :=
  (4 * Real.cos φ, 3 * Real.sin φ)

noncomputable def curve_standard (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 9 = 1

noncomputable def range_2x_plus_y : Set ℝ :=
  {z : ℝ | -Real.sqrt 73 ≤ z ∧ z ≤ Real.sqrt 73}

theorem curve_standard_equation (φ : ℝ) :
  let (x, y) := curve_parametric φ in
  curve_standard x y :=
by
  sorry

theorem range_of_2x_plus_y (φ : ℝ) :
  let (x, y) := curve_parametric φ in
  2 * x + y ∈ range_2x_plus_y :=
by
  sorry

end curve_standard_equation_range_of_2x_plus_y_l60_60201


namespace reciprocal_of_neg_two_l60_60892

theorem reciprocal_of_neg_two : ∃ y : ℝ, (-2) * y = 1 ∧ y = -1/2 :=
by
  use -1/2
  split
  · -- Prove that (-2) * (-1/2) = 1
    sorry
  · -- Prove that y = -1/2
    rfl

end reciprocal_of_neg_two_l60_60892


namespace outer_perimeter_of_fence_l60_60147

-- Define the given conditions
def total_posts : ℕ := 28
def posts_per_longer_side : ℕ := 6
def gap_between_posts : ℕ := 4 -- in feet

-- Objective: Define the outer perimeter and prove it equals 112
theorem outer_perimeter_of_fence
  (total_posts = 28)
  (posts_per_longer_side = 6)
  (gap_between_posts = 4) :
  2 * (5 * gap_between_posts + 9 * gap_between_posts) = 112 :=
by
  have length_of_longer_side := 5 * gap_between_posts
  have length_of_shorter_side := 9 * gap_between_posts
  have outer_perimeter := 2 * (length_of_longer_side + length_of_shorter_side)
  show outer_perimeter = 112
  sorry

end outer_perimeter_of_fence_l60_60147


namespace ratio_radii_l60_60220

-- Given conditions
def V_large : ℝ := 216 * real.pi
def V_small : ℝ := 0.2 * V_large

-- The ratio of the radii of the spheres
theorem ratio_radii (V_large V_small : ℝ) (h1 : V_large = 216 * real.pi) (h2 : V_small = 0.2 * V_large) :
  (let R := real.cbrt (3 * V_large / (4 * real.pi))) in
  (let r := real.cbrt (3 * V_small / (4 * real.pi))) in
  r / R = 1 / real.cbrt 5 := 
by {
  sorry
}

end ratio_radii_l60_60220


namespace fred_speed_l60_60677

variable {F : ℝ} -- Fred's speed
variable {T : ℝ} -- Time in hours

-- Conditions
def initial_distance : ℝ := 35
def sam_speed : ℝ := 5
def sam_distance : ℝ := 25
def fred_distance := initial_distance - sam_distance

-- Theorem to prove
theorem fred_speed (h1 : T = sam_distance / sam_speed) (h2 : fred_distance = F * T) :
  F = 2 :=
by
  sorry

end fred_speed_l60_60677


namespace find_a_from_square_and_logs_l60_60470

theorem find_a_from_square_and_logs (a x y : ℝ) (h1 : ∀ (a : ℝ) (ha : 0 < a) (hA : (x, y) = (a^y, log a x)),
  square_area : x ^ 2 = 64) 
  (h2 : y = log a x)
  (h3 : y = 3 * log a (x + 8))
  (h4 : y + 8 = 5 * log a (x + 8)) :
  a = nat_root 6 64 :=
by
  sorry

end find_a_from_square_and_logs_l60_60470


namespace largest_funny_number_l60_60589

def is_funny (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 0 ∧ d ∣ n → Nat.prime (d + 2)

theorem largest_funny_number : ∃ n : ℕ, is_funny n ∧ ∀ m : ℕ, is_funny m → Nat.card (Nat.divisors m) ≤ Nat.card (Nat.divisors n) := 
  exists.intro 135 sorry

end largest_funny_number_l60_60589


namespace find_P_in_parabola_l60_60988

open Real

-- Vertex of the parabola
def V : Real × Real := (0, 0)

-- Focus of the parabola
def F : Real × Real := (0, 3)

-- Assume P is in the first quadrant
variable {x y : Real}
def P : Real × Real := (x, y)
def InFirstQuadrant (p : Real × Real) : Prop := p.1 > 0 ∧ p.2 > 0

-- Distance PF is 50
def distance (p1 p2 : Real × Real) : Real :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def distance_PF_eq_50 (p : Real × Real) : Prop :=
  distance p F = 50

-- Equation of the parabola derived from focus and directrix
def parabola_eq (p : Real × Real) : Prop :=
  sqrt (p.1^2 + (p.2 - 3)^2) = p.2 + 3

-- Combining all the conditions to find a solution P
theorem find_P_in_parabola :
  parabola_eq P → distance_PF_eq_50 P → InFirstQuadrant P → P = (23.75, 47) :=
by
  sorry

end find_P_in_parabola_l60_60988


namespace valid_triangle_option_l60_60541

/-- Define the types for the sides of a triangle and the angle -/
def sideLength := ℝ
def angle := ℝ

structure Triangle :=
  (AB : sideLength)
  (BC : sideLength)
  (AC : sideLength)
  (angleABC : angle)

/-- Define the conditions for option A, B, C, and D. -/
def option_A := Triangle.mk 2 3 5 0
def option_B := {AB := 2, BC := 3, AC := 0, angleABC := 0}
def option_C := {AB := 2, BC := 3, AC := 0, angleABC := 50}
def option_D := {AB := 0, BC := 0, AC := 0, angleABC := 0}

/-- State the theorem to determine the valid option for a unique triangle. -/
theorem valid_triangle_option : ∃ (t : Triangle), t = option_C ∧
  (t.AB = 2) ∧
  (t.BC = 3) ∧
  (t.angleABC = 50) := by
  sorry

end valid_triangle_option_l60_60541


namespace complete_square_k_value_l60_60004

noncomputable def complete_square_form (x : ℝ) : ℝ := x^2 - 7 * x

theorem complete_square_k_value : ∃ a h k : ℝ, complete_square_form x = a * (x - h)^2 + k ∧ k = -49 / 4 :=
by
  use [1, 7/2, -49/4]
  -- This proof step will establish the relationships and the equality
  sorry

end complete_square_k_value_l60_60004


namespace correct_square_root_operation_l60_60168

theorem correct_square_root_operation : 
  (sqrt 4)^2 = 4 ∧ sqrt 4 ≠ 2 ∨ -2 ∧ sqrt ((-4)^2) ≠ -4 ∧ (-sqrt 4)^2 ≠ -4 :=
by
  have a : (sqrt 4)^2 = 4, from sorry,
  have b : sqrt 4 ≠ 2 ∨ -2, from sorry,
  have c : sqrt ((-4)^2) ≠ -4, from sorry,
  have d : (-sqrt 4)^2 ≠ -4, from sorry,
  exact ⟨a, b, c, d⟩

end correct_square_root_operation_l60_60168


namespace soccer_team_substitutions_mod_1000_l60_60996

theorem soccer_team_substitutions_mod_1000 :
  let a₀ := 1 in
  let a₁ := 11 * 12 * a₀ in
  let a₂ := 11 * 11 * a₁ in
  let a₃ := 11 * 10 * a₂ in
  let a₄ := 11 * 9 * a₃ in
  (a₀ + a₁ + a₂ + a₃ + a₄) % 1000 = 25 :=
by
  sorry

end soccer_team_substitutions_mod_1000_l60_60996


namespace set_intersections_and_unions_l60_60966

open Set

noncomputable def A := {x ∈ ℕ | 1 ≤ x ∧ x < 9}
def B := {1, 2, 3}
def C := {3, 4, 5, 6}

theorem set_intersections_and_unions : 
  (A ∩ B = {1, 2, 3}) ∧ 
  (A ∩ C = {3, 4, 5, 6}) ∧
  (A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6}) ∧
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8}) :=
by {
  -- proof should go here
  sorry
}

end set_intersections_and_unions_l60_60966


namespace mark_all_integer_points_possible_l60_60959

theorem mark_all_integer_points_possible 
  (n : ℕ) 
  (h_pos : 1 < n)
  (h_coprime_segments: ∀ (a b : ℕ), a ≠ b → nat.coprime (b - a) n)
  (h_endpoints: ∀ (i : ℕ), 0 ≤ i ∧ i ≤ 2002)
  : ∃ (mark_points : finset ℕ), (∀ x ∈ mark_points, x ≤ 2002)
    ∧ (∀ s ∈ mark_points, ∀ t ∈ mark_points, s ≠ t → nat.coprime (t - s) n)
    ∧ finset.range (2003) ⊆ mark_points := 
sorry

end mark_all_integer_points_possible_l60_60959


namespace meet_first_time_after_15_seconds_l60_60517

-- Define the speeds in km/h
def speed_first_person_kmph : ℝ := 20
def speed_second_person_kmph : ℝ := 40

-- Convert speeds to m/s
def kmph_to_ms (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)
def speed_first_person_ms : ℝ := kmph_to_ms speed_first_person_kmph
def speed_second_person_ms : ℝ := kmph_to_ms speed_second_person_kmph

-- Define the length of track in meters
def length_of_track : ℝ := 250

-- Define the time they will meet for the first time
def time_to_meet : ℝ := length_of_track / (speed_first_person_ms + speed_second_person_ms)

-- Statement to prove
theorem meet_first_time_after_15_seconds
  (h1 : speed_first_person_kmph = 20)
  (h2 : speed_second_person_kmph = 40)
  (h3 : length_of_track = 250) :
  time_to_meet ≈ 15 :=
by sorry

end meet_first_time_after_15_seconds_l60_60517


namespace relationship_among_a_b_c_l60_60041

-- Definitions from the conditions
def a : ℝ := 5^(0.3)
def b : ℝ := 0.3^5
def c : ℝ := logb 5 0.3 + logb 5 2

-- Theorem statement proving the relationship among a, b, and c
theorem relationship_among_a_b_c : c < b ∧ b < a := by
  sorry

end relationship_among_a_b_c_l60_60041


namespace total_students_in_high_school_l60_60975

-- Definitions based on the problem conditions
def freshman_students : ℕ := 400
def sample_students : ℕ := 45
def sophomore_sample_students : ℕ := 15
def senior_sample_students : ℕ := 10

-- The theorem to be proved
theorem total_students_in_high_school : (sample_students = 45) → (freshman_students = 400) → (sophomore_sample_students = 15) → (senior_sample_students = 10) → ∃ total_students : ℕ, total_students = 900 :=
by
  sorry

end total_students_in_high_school_l60_60975


namespace count_sequences_l60_60269

theorem count_sequences : 
    let S := {s : Fin₅ × Fin₅ × Fin₅ × Fin₅ × Fin₅ // s.1 ≤ s.2 ∧ s.2 ≤ s.3 ∧ s.3 ≤ s.4 ∧ s.4 ≤ s.5 ∧ 
                                               s.1 ≤ 1 ∧ s.2 ≤ 2 ∧ s.3 ≤ 3 ∧ s.4 ≤ 4 ∧ s.5 ≤ 5}
    in S.card = 42 :=
by {
    sorry
}

end count_sequences_l60_60269


namespace tim_total_tv_watching_time_l60_60928

def show1_episodes := 24
def show1_duration_per_episode := 0.5
def show2_episodes := 12
def show2_duration_per_episode := 1

theorem tim_total_tv_watching_time :
  (show1_episodes * show1_duration_per_episode) + (show2_episodes * show2_duration_per_episode) = 24 := 
by
  simp [show1_episodes, show1_duration_per_episode, show2_episodes, show2_duration_per_episode] 
  sorry

end tim_total_tv_watching_time_l60_60928


namespace number_of_differences_l60_60372

theorem number_of_differences (S : Set ℕ) (hS : S = {1, 2, 3, 4, 5, 6, 7}) :
  ∃! n, n = 6 ∧ 
    ∀ k ∈ {1, 2, 3, 4, 5, 6}, ∃ a b ∈ S, a ≠ b ∧ k = |a - b| :=
by
  exist n,
    ( n = 6 ∧ ∀ k ∈ {1, 2, 3, 4, 5, 6},
      ∃ a b ∈ {1, 2, 3, 4, 5, 6, 7}, a ≠ b ∧ k = |a - b| )
  sorry

end number_of_differences_l60_60372


namespace triangle_even_number_in_each_row_from_third_l60_60416

/-- Each number in the (n+1)-th row of the triangle is the sum of three numbers 
  from the n-th row directly above this number and its immediate left and right neighbors.
  If such neighbors do not exist, they are considered as zeros.
  Prove that in each row of the triangle, starting from the third row,
  there is at least one even number. -/

theorem triangle_even_number_in_each_row_from_third (triangle : ℕ → ℕ → ℕ) :
  (∀ n i : ℕ, i > n → triangle n i = 0) →
  (∀ n i : ℕ, triangle (n+1) i = triangle n (i-1) + triangle n i + triangle n (i+1)) →
  ∀ n : ℕ, n ≥ 2 → ∃ i : ℕ, i ≤ n ∧ 2 ∣ triangle n i :=
by
  intros
  sorry

end triangle_even_number_in_each_row_from_third_l60_60416


namespace at_least_one_nonnegative_l60_60562

theorem at_least_one_nonnegative (x : ℝ) :
  let m := x^2 - 1,
      n := 2*x + 2
  in m >= 0 ∨ n >= 0 := 
sorry

end at_least_one_nonnegative_l60_60562


namespace cupcake_difference_l60_60606

def betty_rate : ℕ := 10
def dora_rate : ℕ := 8
def total_hours : ℕ := 5
def betty_break_hours : ℕ := 2

theorem cupcake_difference :
  (dora_rate * total_hours) - (betty_rate * (total_hours - betty_break_hours)) = 10 :=
by
  sorry

end cupcake_difference_l60_60606


namespace person_with_avg_5_picked_zero_l60_60648

def people := list ℤ

def avg_neighbors (a : people) (i : ℕ) :=
  (a.get! ((i + 14) % 15) + a.get! ((i + 16) % 15)) / 2

def person_eighth_announcement := avg_neighbors [some list of 15 integers] 7 = 10

theorem person_with_avg_5_picked_zero : ∀ (a : people), person_eighth_announcement → avg_neighbors a 12 = 5 → a.get! 12 = 0 :=
by
  intros
  sorry

end person_with_avg_5_picked_zero_l60_60648


namespace arithmetic_sequence_a4_a5_sum_l60_60344

theorem arithmetic_sequence_a4_a5_sum
  (a_n : ℕ → ℝ)
  (a1_a2_sum : a_n 1 + a_n 2 = -1)
  (a3_val : a_n 3 = 4)
  (h_arith : ∃ d : ℝ, ∀ (n : ℕ), a_n (n + 1) = a_n n + d) :
  a_n 4 + a_n 5 = 17 := 
by
  sorry

end arithmetic_sequence_a4_a5_sum_l60_60344


namespace planter_pots_cost_l60_60936

/-- Wes wants to place a large planter pot at each corner of his rectangle-shaped pool. 
Each planter will have a large palm fern that is $15.00 per plant, 4 creeping jennies 
that costs $4.00 per plant and 4 geraniums that cost $3.50 per plant. 
Prove that it will cost $180.00 to fill all the planter pots. -/
theorem planter_pots_cost :
  let palm_fern_cost : ℝ := 15.00
  let creeping_jenny_cost : ℝ := 4.00
  let geranium_cost : ℝ := 3.50
  let number_of_corners : ℕ := 4
  let total_cost : ℝ := (palm_fern_cost + 4 * creeping_jenny_cost + 4 * geranium_cost) * number_of_corners
  total_cost = 180.00 :=
by
  -- Definitions of the costs
  let palm_fern_cost : ℝ := 15.00
  let creeping_jenny_cost : ℝ := 4.00
  let geranium_cost : ℝ := 3.50
  let number_of_corners : ℕ := 4

  -- Calculate the total cost per planter
  let cost_per_planter := palm_fern_cost + 4 * creeping_jenny_cost + 4 * geranium_cost

  -- Calculate the total cost for all planters
  let total_cost := cost_per_planter * number_of_corners

  -- Prove that total_cost equals 180.00
  have h1 : total_cost = (15.00 + 4 * 4.00 + 4 * 3.50) * 4 := by rfl
  have h2 : (15.00 + 4 * 4.00 + 4 * 3.50) = 45.00 := by norm_num
  have h3 : 45.00 * 4 = 180.00 := by norm_num

  show total_cost = 180.00 from by rw [h1, h2, h3]

-- Sorry for the proof, mathlib is necessary for proof techniques
sorry

end planter_pots_cost_l60_60936


namespace trajectory_eq_lambda_sum_constant_l60_60692

section trajectory_equation

variable (x y : ℝ)

-- Define points M and N
def M := (4 : ℝ, 0 : ℝ)
def N := (1 : ℝ, 0 : ℝ)

-- Define vector MN
def vector_MN := (fst N - fst M, snd N - snd M) -- (1-4, 0-0) = (-3, 0)

-- Define point P (x, y)
def P := (x, y)

-- Define vectors MP and PN
def vector_MP := (fst P - fst M, snd P - snd M)
def vector_PN := (fst N - fst P, snd N - snd P)

-- Define dot product and magnitude
def dot_product (v1 v2 : ℝ × ℝ) := v1.fst * v2.fst + v1.snd * v2.snd
def magnitude (v : ℝ × ℝ) := Real.sqrt (v.fst * v.fst + v.snd * v.snd)

-- Condition given in the problem
def cond := dot_product vector_MN vector_MP = 6 * magnitude vector_PN

-- Prove the trajectory equation
theorem trajectory_eq : cond → (x^2 / 4 + y^2 / 3 = 1) :=
by
  intro h,
  sorry

end trajectory_equation

section lambda_constant

-- Assume the values λ_1 and λ_2 derived from the problem
variable (λ_1 λ_2 : ℝ)

-- Prove that the sum of λ_1 and λ_2 is a constant
theorem lambda_sum_constant (h1 : λ_1 = -2 / 3) (h2 : λ_2 = -2) : λ_1 + λ_2 = -8 / 3 :=
by
  rw [h1, h2],
  norm_num
  sorry

end lambda_constant

end trajectory_eq_lambda_sum_constant_l60_60692


namespace not_invited_students_l60_60398

-- Definition of the problem conditions
def students := 15
def direct_friends_of_mia := 4
def unique_friends_of_each_friend := 2

-- Problem statement
theorem not_invited_students : (students - (1 + direct_friends_of_mia + direct_friends_of_mia * unique_friends_of_each_friend) = 2) :=
by
  sorry

end not_invited_students_l60_60398


namespace polynomial_transformation_l60_60811

theorem polynomial_transformation (g : Polynomial ℝ) (x : ℝ)
  (h : g.eval (x^2 + 2) = x^4 + 6 * x^2 + 8 * x) : 
  g.eval (x^2 - 1) = x^4 - 1 := by
  sorry

end polynomial_transformation_l60_60811


namespace min_value_of_f_l60_60485

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem min_value_of_f : 
  ∃ x ∈ Icc (-Real.pi / 2) 0, 
    ∀ y ∈ Icc (-Real.pi / 2) 0, f y ≥ f x ∧ x = -Real.pi / 2 := 
by
  sorry

end min_value_of_f_l60_60485


namespace geometric_sequence_tenth_term_l60_60628

theorem geometric_sequence_tenth_term :
  let a := 4
  let r := (12 / 3) / 4
  let nth_term (n : ℕ) := a * r^(n-1)
  nth_term 10 = 4 :=
  by sorry

end geometric_sequence_tenth_term_l60_60628


namespace sum_of_triangle_areas_half_total_area_l60_60795

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem sum_of_triangle_areas_half_total_area (A B C P D E F : ℝ × ℝ)
  (h_PA : P ∈ open_segment ℝ A A)
  (h_PB : P ∈ open_segment ℝ B B)
  (h_PC : P ∈ open_segment ℝ C C)
  (h_AD : D ∈ open_segment ℝ A A)
  (h_BE : E ∈ open_segment ℝ B B)
  (h_CF : F ∈ open_segment ℝ C C)
  (h_medians : P ∈ segment ℝ (midpoint ℝ A B) (midpoint ℝ B C) ∧ P ∈ segment ℝ (midpoint ℝ B C) (midpoint ℝ C A) ∧ P ∈ segment ℝ (midpoint ℝ C A) (midpoint ℝ A B)) :
  triangle_area P A F + triangle_area P B D + triangle_area P C E = 0.5 * triangle_area A B C :=
sorry

end sum_of_triangle_areas_half_total_area_l60_60795


namespace dispatch_plans_l60_60235

-- Define the problem conditions
def num_teachers : ℕ := 8
def num_selected : ℕ := 4

-- Teacher constraints
def constraint1 (A B : Prop) : Prop := ¬(A ∧ B)  -- A and B cannot go together
def constraint2 (A C : Prop) : Prop := (A ∧ C) ∨ (¬A ∧ ¬C)  -- A and C can only go together or not at all

theorem dispatch_plans (A B C: Prop) 
                      (h1: constraint1 A B)
                      (h2: constraint2 A C)
                      (h3: num_teachers = 8)  
                      (h4: num_selected = 4) 
                      : num_ways_to_dispatch = 600 :=
by 
  sorry

end dispatch_plans_l60_60235


namespace equality_of_lengths_l60_60840

-- Definitions for cyclic quadrilateral, points, and lengths
variables {A B C D M N : Type} [InCircle A B C D]

-- Conditions given in the problem
variable (intersect_AM_DC : Intersect A M D C)
variable (intersect_BC_AD : Intersect B C A D)
variable (BM_eq_DN : Length (B, M) = Length (D, N))

-- Goal to prove
theorem equality_of_lengths (h : InCircle A B C D) (h1 : Intersect A M D C) (h2 : Intersect B C A D) (h3 : Length (B, M) = Length (D, N)) 
: Length (C, M) = Length (C, N) :=
by
  sorry

end equality_of_lengths_l60_60840


namespace stock_increase_l60_60249

theorem stock_increase (x : ℝ) (h₁ : x > 0) :
  (1.25 * (0.85 * x) - x) / x * 100 = 6.25 :=
by 
  -- {proof steps would go here}
  sorry

end stock_increase_l60_60249


namespace median_of_set_l60_60040

theorem median_of_set (a : ℤ) (c : ℝ) (h1 : a ≠ 0) (h2 : 0 < c) (h3 : a * c^3 = Real.log (c) / Real.log 10) :
  median ({0, 1, a, c, 1/c} : set ℝ) = 1 :=
by
  sorry

end median_of_set_l60_60040


namespace sum_of_a_for_one_solution_l60_60274

theorem sum_of_a_for_one_solution (a : ℝ) :
  (∀ x : ℝ, 3 * x^2 + (a + 15) * x + 18 = 0 ↔ (a + 15) ^ 2 - 4 * 3 * 18 = 0) →
  a = -15 + 6 * Real.sqrt 6 ∨ a = -15 - 6 * Real.sqrt 6 → a + (-15 + 6 * Real.sqrt 6) + (-15 - 6 * Real.sqrt 6) = -30 :=
by
  intros h1 h2
  have hsum : (-15 + 6 * Real.sqrt 6) + (-15 - 6 * Real.sqrt 6) = -30 := by linarith [Real.sqrt 6]
  sorry

end sum_of_a_for_one_solution_l60_60274


namespace ab_not_divisible_by_5_then_neither_divisible_l60_60521

theorem ab_not_divisible_by_5_then_neither_divisible (a b : ℕ) : ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) → ¬(5 ∣ (a * b)) :=
by
  -- Mathematical statement for proof by contradiction:
  have H1: ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) := sorry
  -- Rest of the proof would go here  
  sorry

end ab_not_divisible_by_5_then_neither_divisible_l60_60521


namespace two_same_color_probability_l60_60756

-- Definitions based on the given conditions
def total_balls := 5
def black_balls := 3
def red_balls := 2

-- Definition for drawing two balls at random
def draw_two_same_color_probability : ℚ :=
  let total_ways := Nat.choose total_balls 2
  let black_pairs := Nat.choose black_balls 2
  let red_pairs := Nat.choose red_balls 2
  (black_pairs + red_pairs) / total_ways

-- Statement of the theorem
theorem two_same_color_probability :
  draw_two_same_color_probability = 2 / 5 :=
  sorry

end two_same_color_probability_l60_60756


namespace tan_A_mul_tan_B_lt_one_l60_60754

theorem tan_A_mul_tan_B_lt_one (A B C : ℝ) (hC: C > 90) (hABC : A + B + C = 180) :
    Real.tan A * Real.tan B < 1 :=
sorry

end tan_A_mul_tan_B_lt_one_l60_60754


namespace part1_part2_inequality_l60_60709

noncomputable def f (x : ℝ) := Real.ln x
noncomputable def g (a b x : ℝ) := (1 / 2) * a * x + b
noncomputable def φ (m x : ℝ) := (m * (x - 1)) / (x + 1) - Real.ln x

theorem part1 :
  (f' 1 = (1 / 2) * a) ∧ (g 1 = 0) → g x = x - 1 := sorry
  
theorem part2 :
  (∀ x ≥ 1, φ m x ≤ 0) → m ≤ 2 := sorry
  
theorem inequality (n : ℕ) (hn : n ≥ 2) :
  (2 * n / (n + 1) < ∑ i in (range n).map succ, 1 / (Real.ln (i + 1))) ∧ 
  (∑ i in (range n).map succ, 1 / (Real.ln (i + 1)) < (n / 2) + 1 + ∑ i in (range n), 1 / (i + 1)) := sorry

end part1_part2_inequality_l60_60709


namespace goose_eggs_calculation_l60_60064

noncomputable def goose_eggs_total (E : ℕ) : Prop :=
  let hatched := (2/3) * E
  let survived_first_month := (3/4) * hatched
  let survived_first_year := (2/5) * survived_first_month
  survived_first_year = 110

theorem goose_eggs_calculation :
  goose_eggs_total 3300 :=
by
  have h1 : (2 : ℝ) / (3 : ℝ) ≠ 0 := by norm_num
  have h2 : (3 : ℝ) / (4 : ℝ) ≠ 0 := by norm_num
  have h3 : (2 : ℝ) / (5 : ℝ) ≠ 0 := by norm_num
  sorry

end goose_eggs_calculation_l60_60064


namespace no_solutions_exist_l60_60640

theorem no_solutions_exist (m n : ℤ) : ¬(m^2 = n^2 + 1954) :=
by sorry

end no_solutions_exist_l60_60640


namespace integer_pairs_poly_l60_60291

theorem integer_pairs_poly (a b : ℤ) :
  ∃ (P : Polynomial ℤ), 
  (∀ n (c : Fin (n + 1) → ℤ), 
     (x^2 + a * Polynomial.X + Polynomial.C b) * P = 
     Polynomial.monomial n 1 + ∑ i in Finset.range n, Polynomial.monomial i (c i))
     ∧ (∀ i ∈ Finset.range n, ∃ ci, (c i) = 1 ∨ (c i) = -1) ↔ ((a, b) ∈ {(-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1)})) :=
sorry

end integer_pairs_poly_l60_60291


namespace least_time_and_digit_sum_l60_60508

-- Define the list of prime numbers (minutes per lap for each horse):
def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Check if at least 5 out of 10 prime numbers divide T
def at_least_five_primes_divide (T : ℕ) : Prop :=
  (primes.filter (fun p => T % p = 0)).length ≥ 5

-- Sum of the digits of a number
def digit_sum (n : ℕ) : ℕ :=
  (n.to_string.to_list.map (λ c => c.to_nat - '0'.to_nat)).sum

-- Main theorem statement
theorem least_time_and_digit_sum :
  ∃ T > 0, at_least_five_primes_divide T ∧ digit_sum T = 6 :=
sorry

end least_time_and_digit_sum_l60_60508


namespace soccer_team_substitutions_mod_1000_l60_60997

theorem soccer_team_substitutions_mod_1000 :
  let a₀ := 1 in
  let a₁ := 11 * 12 * a₀ in
  let a₂ := 11 * 11 * a₁ in
  let a₃ := 11 * 10 * a₂ in
  let a₄ := 11 * 9 * a₃ in
  (a₀ + a₁ + a₂ + a₃ + a₄) % 1000 = 25 :=
by
  sorry

end soccer_team_substitutions_mod_1000_l60_60997


namespace find_angle_A_l60_60753

variables {A B C a b c : ℝ}
variables {triangle_ABC : (2 * b - c) * (Real.cos A) = a * (Real.cos C)}

theorem find_angle_A (h : (2 * b - c) * (Real.cos A) = a * (Real.cos C)) : A = Real.pi / 3 :=
by
  sorry

end find_angle_A_l60_60753


namespace concurrency_of_circumcenter_lines_l60_60228

/-- The quadrilateral ABCD is inscribed in a circle O with diagonals AC and BD intersecting at point P. 
    The centers of the circumcircles of triangles ABP, BCP, CDP, and DAP are O_1, O_2, O_3, and O_4, respectively.
    Prove that the lines OP, O_1O_3, and O_2O_4 are concurrent. -/
theorem concurrency_of_circumcenter_lines 
  {A B C D O P O_1 O_2 O_3 O_4 : Type}
  [h_inscribed : circle O [A, B, C, D]]
  (h_intersections : intersect (line AC) (line BD) = P)
  (h_centers_abp : circumcenter O1 (triangle A B P))
  (h_centers_bcp : circumcenter O2 (triangle B C P))
  (h_centers_cdp : circumcenter O3 (triangle C D P))
  (h_centers_dap : circumcenter O4 (triangle D A P)) :
  concurrent (line OP) (line O1 O3) (line O2 O4) :=
sorry

end concurrency_of_circumcenter_lines_l60_60228


namespace cylinder_surface_area_l60_60232

-- Define the radius and height
def r : ℕ := 3
def h : ℕ := 2 * r

-- Define the formula for the total surface area of a cylinder
def total_surface_area (r h : ℕ) : ℕ := 2 * r * h + 2 * r * r

-- Statement to be proved
theorem cylinder_surface_area : total_surface_area r h = 54 * Int.pi := 
by sorry

end cylinder_surface_area_l60_60232


namespace reduced_price_is_60_l60_60952

variable (P R: ℝ) -- Declare the variables P and R as real numbers.

-- Define the conditions as hypotheses.
axiom h1 : R = 0.7 * P
axiom h2 : 1800 / R = 1800 / P + 9

-- The theorem stating the problem to prove.
theorem reduced_price_is_60 (P R : ℝ) (h1 : R = 0.7 * P) (h2 : 1800 / R = 1800 / P + 9) : R = 60 :=
by sorry

end reduced_price_is_60_l60_60952


namespace problem_1_problem_2_l60_60016

-- Define point M in polar coordinates
def point_M : ℝ × ℝ := (2 * Real.sqrt 2, Real.pi / 4)

-- Define the parametric equations of the curve C
def parametric_curve_C (α : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos α, 2 * Real.sin α)

-- Line l passing through M and tangent to curve C
def tangent_polar_eq1 : ℝ → ℝ → Prop :=
  λ ρ θ, ρ * Real.sin θ = 2

def tangent_polar_eq2 : ℝ → ℝ → Prop :=
  λ ρ θ, 4 * ρ * Real.cos θ + 3 * ρ * Real.sin θ - 8 = 0

-- Point N symmetric to M with respect to the y-axis
def point_N : ℝ × ℝ := (-2, 2)
def sqrt13 := Real.sqrt 13

-- Distance range from points on curve C to point N
def distance_range : Set ℝ := Set.Icc (sqrt13 - 2) (sqrt13 + 2)

theorem problem_1 :
  ∃ ρ θ, tangent_polar_eq1 ρ θ ∨ tangent_polar_eq2 ρ θ :=
sorry

theorem problem_2 :
  ∀ α : ℝ, parametric_curve_C α ≠ ∅ →
  ∃ d : ℝ, d ∈ distance_range :=
sorry

end problem_1_problem_2_l60_60016


namespace quadrilateral_midpoints_area_l60_60432

-- We set up the geometric context and define the problem in Lean 4.

noncomputable def area_of_midpoint_quadrilateral
  (AB CD : ℝ) (AD BC : ℝ)
  (h_AB_CD : AB = 15) (h_CD_AB : CD = 15)
  (h_AD_BC : AD = 10) (h_BC_AD : BC = 10)
  (mid_AB : Prop) (mid_BC : Prop) (mid_CD : Prop) (mid_DA : Prop) : ℝ :=
  37.5

-- The theorem statement validating the area of the quadrilateral.
theorem quadrilateral_midpoints_area (AB CD AD BC : ℝ) 
  (h_AB_CD : AB = 15) (h_CD_AB : CD = 15)
  (h_AD_BC : AD = 10) (h_BC_AD : BC = 10)
  (mid_AB : Prop) (mid_BC : Prop) (mid_CD : Prop) (mid_DA : Prop) :
  area_of_midpoint_quadrilateral AB CD AD BC h_AB_CD h_CD_AB h_AD_BC h_BC_AD mid_AB mid_BC mid_CD mid_DA = 37.5 :=
by 
  sorry  -- Proof is omitted.

end quadrilateral_midpoints_area_l60_60432


namespace ruth_started_with_89_apples_l60_60463

theorem ruth_started_with_89_apples 
  (initial_apples : ℕ)
  (shared_apples : ℕ)
  (remaining_apples : ℕ)
  (h1 : shared_apples = 5)
  (h2 : remaining_apples = 84)
  (h3 : remaining_apples = initial_apples - shared_apples) : 
  initial_apples = 89 :=
by
  sorry

end ruth_started_with_89_apples_l60_60463


namespace no_point_with_integer_distances_l60_60835

theorem no_point_with_integer_distances (x y : ℕ) (hx : x % 2 = 1) (hy : y % 2 = 1) :
  ¬ ∃ (P : ℝ × ℝ), (dist P (0, 0) ∈ ℤ) ∧ (dist P (x, 0) ∈ ℤ) ∧ (dist P (x, y) ∈ ℤ) ∧ (dist P (0, y) ∈ ℤ) :=
by
  sorry

end no_point_with_integer_distances_l60_60835


namespace real_solutions_l60_60650

theorem real_solutions {
  (x : ℝ) :
  ( \frac{2}{(x - 1)*(x - 2)} + \frac{2}{(x - 2)*(x - 3)} + \frac{2}{(x - 3)*(x - 4)} = \frac{1}{3} ) ↔ (x = 8 ∨ x = -2.5) := 
sorry

end real_solutions_l60_60650


namespace cost_per_serving_l60_60847

def pasta_cost : ℝ := 1.0
def sauce_cost : ℝ := 2.0
def meatballs_cost : ℝ := 5.0
def total_servings : ℝ := 8.0

theorem cost_per_serving : (pasta_cost + sauce_cost + meatballs_cost) / total_servings = 1.0 :=
by sorry

end cost_per_serving_l60_60847


namespace isosceles_trapezoid_with_inscribed_circle_area_is_20_l60_60295

def isosceles_trapezoid_area (a b c1 c2 h : ℕ) : ℕ :=
  (a + b) * h / 2

theorem isosceles_trapezoid_with_inscribed_circle_area_is_20
  (a b c h : ℕ)
  (ha : a = 2)
  (hb : b = 8)
  (hc : a + b = 2 * c)
  (hh : h ^ 2 = c ^ 2 - ((b - a) / 2) ^ 2) :
  isosceles_trapezoid_area a b c c h = 20 := 
by {
  sorry
}

end isosceles_trapezoid_with_inscribed_circle_area_is_20_l60_60295


namespace highest_probability_l60_60604

open ProbabilityTheory

variables (Ω : Type) [ProbabilitySpace Ω]

def event_A : Event Ω := {ω | True} -- Anya waits for at least one minute
def event_B : Event Ω := {ω | True} -- Anya waits for at least two minutes
def event_C : Event Ω := {ω | True} -- Anya waits for at least five minutes

axiom A_superset_B : event_B ⊆ event_A
axiom B_superset_C : event_C ⊆ event_B

theorem highest_probability :
  P[event_A] ≥ P[event_B] ∧ P[event_B] ≥ P[event_C] :=
by
  sorry

end highest_probability_l60_60604


namespace youngest_age_is_20_l60_60196

-- Definitions of the ages
def siblings_ages (y : ℕ) : List ℕ := [y, y+2, y+7, y+11]

-- Condition of the problem: average age is 25
def average_age_25 (y : ℕ) : Prop := (siblings_ages y).sum = 100

-- The statement to be proven
theorem youngest_age_is_20 (y : ℕ) (h : average_age_25 y) : y = 20 :=
  sorry

end youngest_age_is_20_l60_60196


namespace value_of_a_for_parallel_lines_l60_60132

theorem value_of_a_for_parallel_lines (a : ℝ) :
  (a ≠ 0) → (a ≠ -4) →
  (2 * (a + 4) = a^2) ↔ (a = 4 ∨ a = -2) :=
by
  intro ha hn4
  split
  · intro h
    have : 2 * (a + 4) = a * a, by assumption
    rw [mul_comm, mul_assoc] at this
    sorry
  · intro h
    cases h with h h
    · rw h
      linarith
    · rw h
      linarith

end value_of_a_for_parallel_lines_l60_60132


namespace reciprocal_of_minus_one_half_l60_60120

theorem reciprocal_of_minus_one_half : (1 / (-1 / 2)) = -2 := 
by sorry

end reciprocal_of_minus_one_half_l60_60120


namespace seq_is_arithmetic_sum_series_lt_three_fourths_l60_60806

theorem seq_is_arithmetic (n : ℕ) (h : 0 < n) :
  let a_n := n^2 + 2*n,
      b_n := a_n + 1 - a_n  -- updated as the difference between a terms changed
  in b_{n+1} - b_n = 2 := by sorry

theorem sum_series_lt_three_fourths (n : ℕ) (h : 0 < n) :
  let a_n := n^2 + 2*n,
      sum := (Finset.range n).sum (λ k, 1/(a_(k+1)))
  in sum < 3/4 := by sorry

end seq_is_arithmetic_sum_series_lt_three_fourths_l60_60806


namespace find_months_contributed_l60_60970

theorem find_months_contributed (x : ℕ) (profit_A profit_total : ℝ)
  (contrib_A : ℝ) (contrib_B : ℝ) (months_B : ℕ) :
  profit_A / profit_total = (contrib_A * x) / (contrib_A * x + contrib_B * months_B) →
  profit_A = 4800 →
  profit_total = 8400 →
  contrib_A = 5000 →
  contrib_B = 6000 →
  months_B = 5 →
  x = 8 :=
by
  intros h₁ h₂ h₃ h₄ h₅ h₆
  sorry

end find_months_contributed_l60_60970


namespace sum_of_c_n_l60_60346

noncomputable def a (n : ℕ) : ℕ := 3^(n-1)
noncomputable def b (n : ℕ) : ℕ := 3 * n - 2
noncomputable def c (n : ℕ) : ℕ := (-1) ^ n * b n + a n * b (2 * n)

noncomputable def T (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    (3 * n + 5) / 2 + (6 * n - 5) / 2 * 3 ^ n
  else
    (6 - 3 * n) / 2 + (6 * n - 5) / 2 * 3 ^ n

theorem sum_of_c_n (n : ℕ) :
  let a : ℕ → ℕ := λ n, 3^(n-1),
      b : ℕ → ℕ := λ n, 3 * n - 2,
      c : ℕ → ℕ := λ n, (-1) ^ n * b n + a n * b (2 * n)
  in T n = ∑ i in range n, c i := sorry

end sum_of_c_n_l60_60346


namespace leak_time_period_l60_60538

-- Define the rates of leakage for the holes
def rate_largest := 3 -- ounces per minute
def rate_medium := (1 / 2) * rate_largest
def rate_smallest := (1 / 3) * rate_medium
def total_leakage := 600 -- total amount of water leaked in ounces

-- Define the combined rate of leakage for the holes
def combined_rate := rate_largest + rate_medium + rate_smallest

-- Define the time period function
def time_period (total_leakage : ℝ) (combined_rate : ℝ) : ℝ :=
  total_leakage / combined_rate

-- State the theorem to be proved
theorem leak_time_period :
  time_period total_leakage combined_rate = 120 :=
by
  sorry

end leak_time_period_l60_60538


namespace reciprocal_of_neg_two_l60_60886

theorem reciprocal_of_neg_two : ∃ x : ℝ, (-2) * x = 1 ∧ x = -1/2 :=
by
  use -1/2
  split
  · exact (by norm_num : (-2) * (-1/2) = 1)
  · rfl

end reciprocal_of_neg_two_l60_60886


namespace probability_of_scoring_at_most_once_l60_60851

theorem probability_of_scoring_at_most_once (p : ℚ) (n : ℕ) (k : ℕ)
  (h_prob : p = 1/3) (h_n : n = 3) (h_k : k = 1):
  (1 - p)^n + n * p * (1 - p)^(n - 1) = 20/27 :=
by
  rw [h_prob, h_n, h_k]
  sorry

end probability_of_scoring_at_most_once_l60_60851


namespace solve_trig_equation_l60_60468

theorem solve_trig_equation
  (x : ℝ)
  (k: ℤ)
  (h: (7/4 - 2 * cos (2 * x)) * |2 * cos (2 * x) + 1| = cos x * (cos x - cos (5 * x))) :
  x = (k * π) / 2 + π / 6 ∨ 
  x = (k * π) / 2 - π / 6 := sorry

end solve_trig_equation_l60_60468


namespace exists_parallel_plane_l60_60627

-- Definitions for skew lines, intersection of planes, and parallel planes
variables {α β γ : Type*}
variables {m n l : Set α} -- Lines are sets of points
variables {P Q : α → β → Prop} -- P: points on plane α, Q: points on plane β

-- Condition: m and n are skew lines on planes α and β
def skew_lines (m n : Set α) (P Q : α → β → Prop) : Prop :=
  ∀ (x ∈ m) (y ∈ n), ¬P x y

-- Condition: l is the intersection of planes α and β
def intersection_line (l : Set α) (P Q : α → β → Prop) : Prop :=
  ∀ x ∈ l, P x x ∧ Q x x

-- Condition: Gamma is a plane parallel to both skew lines m and n
def parallel_plane (γ : Set α) (m n : Set α) : Prop :=
  ∃ (k : Set α), ∀ (x ∈ m) (y ∈ n), γ x y

-- Statement: Prove that such a plane γ exists that is parallel to both m and n
theorem exists_parallel_plane (m n : Set α) (P Q : α → β → Prop) :
  skew_lines m n P Q → intersection_line l P Q → ∃ γ, parallel_plane γ m n :=
sorry

end exists_parallel_plane_l60_60627


namespace intervals_of_monotonicity_exists_x0_range_of_a_l60_60356

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x^2

theorem intervals_of_monotonicity (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, f' x > 0) ∧ 
  (a > 0 → ∀ x > 0, (x < Real.sqrt (2 * a) / (2 * a) → f' x > 0) ∧ 
              (x > Real.sqrt (2 * a) / (2 * a) → f' x < 0)) := sorry

theorem exists_x0 (a : ℝ) (h_div : a = 1 / 8) :
  ∃ x_0 ∈ Set.Ici 2, f x_0 a = f (3 / 2) a := sorry

theorem range_of_a (α β : ℝ) (h_diff : β - α = 1) (h_alpha_beta_dom : α ∈ Set.Icc 1 3) (h_f_eq : f α a = f β a):
  ∃ a ∈ Set.Icc (Real.log (3 / 2) / 5) (Real.log 2 / 3) :=
  (∃ a ∈ Set.Icc (1 / 18) (Real.log 2 / 3),  (1 < (Real.sqrt (2 * a) / (2 * a)) ≤ 2) ∨ (2 < (Real.sqrt (2 * a) / (2 * a)) < 3)) := sorry

end intervals_of_monotonicity_exists_x0_range_of_a_l60_60356


namespace system1_solution_system2_solution_l60_60088

theorem system1_solution (x y : ℚ) :
  x + y = 4 ∧ 5 * (x - y) - 2 * (x + y) = -1 →
  x = 27 / 10 ∧ y = 13 / 10 := by
  sorry

theorem system2_solution (x y : ℚ) :
  (2 * (x - y) / 3) - ((x + y) / 4) = -1 / 12 ∧ 3 * (x + y) - 2 * (2 * x - y) = 3 →
  x = 2 ∧ y = 1 := by
  sorry

end system1_solution_system2_solution_l60_60088


namespace geometric_sequence_common_ratio_l60_60703

-- Definitions from the problem conditions
def a (n : ℕ) : ℝ := 3 * 2^(n-1)

-- Main statement to prove
theorem geometric_sequence_common_ratio : ∃ q : ℝ, (∀ n : ℕ, n ≥ 2 → (a n) / (a (n-1)) = q) ∧ q = 2 :=
by
  use 2
  intros n hn
  sorry

end geometric_sequence_common_ratio_l60_60703


namespace sum_sequence_2023_l60_60824

def reciprocal_difference_point (a : ℚ) : ℚ := 1 / (1 - a)

noncomputable def sequence_a : ℕ → ℚ
| 0     := 1/2
| (n+1) := reciprocal_difference_point (sequence_a n)

theorem sum_sequence_2023 :
  (Finset.range 2023).sum (λ n, sequence_a n) = 1013 :=
sorry

end sum_sequence_2023_l60_60824


namespace increasing_function_m_ge_4_l60_60719

def g (x m : ℝ) : ℝ :=
  if x >= m then (1 / 4) * x^2 else x

theorem increasing_function_m_ge_4 (m : ℝ) (h : 0 < m) :
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → g x m ≤ g y m) → 4 ≤ m :=
  by
  sorry

end increasing_function_m_ge_4_l60_60719


namespace range_of_a_l60_60736

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (|x + a| < 3 ↔ 2 < x ∧ x < 3)) ↔ -5 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l60_60736


namespace number_of_pairs_l60_60050

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x / (1 + |x|)

def M (a b : ℝ) : set ℝ := set.Icc a b
def N (m a b : ℝ) : set ℝ := { y | ∃ x ∈ set.Icc a b, y = f m x }

theorem number_of_pairs (m a b : ℝ) (h : |m| > 1) (hM : M a b = N m a b) :
  (if m < -1 then 1 else if m > 1 then 3 else 0) = if m < -1 then 1 else if m > 1 then 3 else 0 :=
by sorry

end number_of_pairs_l60_60050


namespace max_value_of_f_l60_60774

def max_value_oplus (f : ℝ → ℝ) (oplus : ℝ → ℝ → ℝ) : Prop :=
  (∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x ≤ 6) ∧ (∃ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x = 6)

def oplus (a b : ℝ) : ℝ :=
  if a >= b then a else b^2

noncomputable def f (x : ℝ) : ℝ :=
  (oplus 1 x) * x - (oplus 2 x)

theorem max_value_of_f : max_value_oplus f oplus :=
  sorry

end max_value_of_f_l60_60774


namespace probability_not_red_l60_60389

theorem probability_not_red (h : odds_red = 1 / 3) : probability_not_red_card = 3 / 4 :=
by
  sorry

end probability_not_red_l60_60389


namespace concurrency_of_BD_EM_FX_l60_60399

-- Define the given conditions
variables (A B C D E F M X : Type) 
variables [HasMidpoint C F M] [Parallelogram A M X E]
variables (θ : ℝ) (angle_condition : 
  ∠FAB = θ ∧ ∠FBA = θ ∧ ∠DAC = θ ∧ ∠DCA = θ ∧ ∠EAD = θ ∧ ∠EDA = θ)
variables (right_angle_FBC : ∠FBC = 90)
variables (triangles_similar_isosceles : (triangle_similar_isosceles ABF ∧ triangle_similar_isosceles ACD ∧ triangle_similar_isosceles ADE))

-- The statement to be proved
theorem concurrency_of_BD_EM_FX : concurrent BD EM FX :=
sorry

end concurrency_of_BD_EM_FX_l60_60399


namespace jessie_weight_before_jogging_l60_60030

theorem jessie_weight_before_jogging (current_weight lost_weight : ℕ) 
(hc : current_weight = 67)
(hl : lost_weight = 7) : 
current_weight + lost_weight = 74 := 
by
  -- Here we skip the proof part
  sorry

end jessie_weight_before_jogging_l60_60030


namespace integer_solution_exists_l60_60082

theorem integer_solution_exists : ∃ (x y : ℤ), x^2 - 2 = 7 * y := by
  use 3, 1
  simp
  sorry

end integer_solution_exists_l60_60082


namespace population_doubles_in_35_years_l60_60554

noncomputable def birth_rate : ℝ := 39.4 / 1000
noncomputable def death_rate : ℝ := 19.4 / 1000
noncomputable def natural_increase_rate : ℝ := birth_rate - death_rate
noncomputable def doubling_time (r: ℝ) : ℝ := 70 / (r * 100)

theorem population_doubles_in_35_years :
  doubling_time natural_increase_rate = 35 := by sorry

end population_doubles_in_35_years_l60_60554


namespace range_of_a_l60_60737

theorem range_of_a (a : ℝ) (x : ℝ) : (|x + a| < 3) ↔ (2 < x ∧ x < 3) →  a ∈ Icc (-5 : ℝ) (0 : ℝ) := 
sorry

end range_of_a_l60_60737


namespace reciprocal_neg_half_l60_60119

theorem reciprocal_neg_half : 1 / (-1 / 2 : ℝ) = (-2 : ℝ) :=
by
  sorry

end reciprocal_neg_half_l60_60119


namespace singular_iff_exists_anticommuting_matrix_l60_60035

open Matrix

variable {R : Type*} [CommRing R] [Fintype R] [DecidableEq R]

def conditions (A : Matrix (Fin 4) (Fin 4) R) (adjugate : Matrix (Fin 4) (Fin 4) R) : Prop :=
  A.det ≠ 0 ∧ trace A = trace adjugate ∧ trace A ≠ 0

theorem singular_iff_exists_anticommuting_matrix (A : Matrix (Fin 4) (Fin 4) R) (adjugate : Matrix (Fin 4) (Fin 4) R) :
  conditions A adjugate →
  (∃ B : Matrix (Fin 4) (Fin 4) R, B ≠ 0 ∧ A.mul B = -B.mul A) ↔
  ¬ is_unit (A.mul A + 1) :=
sorry

end singular_iff_exists_anticommuting_matrix_l60_60035


namespace pounds_per_mile_is_correct_l60_60223

-- Define the conditions given in the problem
def hike_rate : ℝ := 2.5 -- mph
def hours_per_day : ℝ := 8 -- hours a day
def days : ℝ := 5 -- total days
def first_pack_weight : ℝ := 40 -- pounds
def resupply_fraction : ℝ := 0.25 -- 25%

-- Define the corresponding calculations based on the conditions
def total_distance : ℝ := hike_rate * hours_per_day * days
def resupply_weight : ℝ := resupply_fraction * first_pack_weight
def total_supplies_weight : ℝ := first_pack_weight + resupply_weight
def pounds_per_mile : ℝ := total_supplies_weight / total_distance

-- State the theorem to prove the required pounds of supplies per mile
theorem pounds_per_mile_is_correct :
  pounds_per_mile = 0.5 :=
  by
  -- Sorry is used to skip the actual proof
  sorry

end pounds_per_mile_is_correct_l60_60223


namespace rectangle_length_l60_60591

theorem rectangle_length (P w : ℝ) (hP : P = 70) (hw : w = 16) : 
  ∃ l : ℝ, P = 2 * (l + w) ∧ l = 19 :=
by
  use 19
  split
  · rw [hP, hw]
    norm_num
  · sorry

end rectangle_length_l60_60591


namespace quadrilateral_BD_length_l60_60409

theorem quadrilateral_BD_length :
  ∃ (BD : ℕ), 
    (ABCD.exists
      ∧ AB = 5
      ∧ BC = 17
      ∧ CD = 5
      ∧ DA = 9
      ∧ BD = 13) :=
sorry

end quadrilateral_BD_length_l60_60409


namespace units_digit_probability_3a_add_7b_is_8_l60_60668

theorem units_digit_probability_3a_add_7b_is_8 :
  let a_set := {n : ℕ | 1 ≤ n ∧ n ≤ 100},
      b_set := {n : ℕ | 1 ≤ n ∧ n ≤ 100}
  in (∃ (a ∈ a_set) (b ∈ b_set), (3 ^ a + 7 ^ b) % 10 = 8) →
     (finset.card a_set * finset.card b_set = 10000) →
     (finset.card {p : ℕ × ℕ | p.1 ∈ a_set ∧ p.2 ∈ b_set ∧ (3 ^ p.1 + 7 ^ p.2) % 10 = 8} = 1875) →
     (1875 / 10000 = 3 / 16) :=
by sorry

end units_digit_probability_3a_add_7b_is_8_l60_60668


namespace max_abs_m_minus_n_l60_60227

theorem max_abs_m_minus_n 
(m n : ℤ) (hm : m > 0) (h : m^2 = 4 * n^2 - 5 * n + 16) : 
  ∃ d, d = abs(m - n) ∧ d ≤ 33 :=
sorry

end max_abs_m_minus_n_l60_60227


namespace minimum_value_of_f_is_2_point_5_l60_60665

def f (x : ℝ) : ℝ := x + (1 / x) + (1 / (x^2 + 1 / x^2))

theorem minimum_value_of_f_is_2_point_5 (x : ℝ) (h : x > 0) : 
  ∃ c, (c = 2.5) ∧ (∀ y, y > 0 → f y ≥ c) := 
begin
  use 2.5,
  split,
  { exact rfl },
  { sorry }
end

end minimum_value_of_f_is_2_point_5_l60_60665


namespace combined_capacity_percentage_l60_60477

def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def circumference_to_radius (C : ℝ) : ℝ := C / (2 * π)

theorem combined_capacity_percentage (hA hB hC CA CB CC : ℝ) (hA_eq : hA = 6) (CA_eq : CA = 8)
  (hB_eq : hB = 8) (CB_eq : CB = 10) (hC_eq : hC = 10) (CC_eq : CC = 12) :
  let rA := circumference_to_radius CA,
      rB := circumference_to_radius CB,
      rC := circumference_to_radius CC,
      VA := volume_cylinder rA hA,
      VB := volume_cylinder rB hB,
      VC := volume_cone rC hC,
      combined_volume := VA + VB,
      percentage := (combined_volume / VC) * 100
  in percentage ≈ 246.67 :=
by
  sorry

end combined_capacity_percentage_l60_60477


namespace ideal_number_of_new_sequence_l60_60705

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

noncomputable def ideal_number (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, sum_first_n_terms a (i + 1)) / n

theorem ideal_number_of_new_sequence {a : ℕ → ℝ} (hn : ideal_number a 502 = 2012) :
  ideal_number (λ i, if i = 0 then 2 else a (i - 1)) 503 = 2010 :=
sorry

end ideal_number_of_new_sequence_l60_60705


namespace find_pq_l60_60445

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 8 * y - 23 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6 * x + 8 * y + 89 = 0
def tangent_circle (x y r : ℝ) : Prop :=
  ∃ (x₃ y₃) (r₃ : ℝ),
    x₃^2 + y₃^2 - 6 * x₃ + 8 * y₃ - 23 = 0 ∧
    x₃^2 + y₃^2 + 6 * x₃ + 8 * y₃ + 89 = 0 ∧
    r₃ = r + sqrt 50 ∧
    x^2 + y^2 = r^2

theorem find_pq : ∃ (p q : ℕ), p + q = 10 ∧ gcd p q = 1 ∧ (
  ∃ m (a : ℝ), m = 3 ∧ m^2 = (p : ℝ) / (q : ℝ) ∧
  (call it as another theorem?)
  ∀ x y r, tangent_circle x y r → (y = a * x + 3) ∧ a = -3
) :=
sorry

end find_pq_l60_60445


namespace ferris_wheel_capacity_l60_60855

-- Define the conditions
def number_of_seats : ℕ := 14
def people_per_seat : ℕ := 6

-- Theorem to prove the total capacity is 84
theorem ferris_wheel_capacity : number_of_seats * people_per_seat = 84 := sorry

end ferris_wheel_capacity_l60_60855


namespace minimal_words_to_learn_l60_60303

theorem minimal_words_to_learn (total_words guess_success_rate required_percentage : ℕ) 
  (H1 : total_words = 800) (H2 : guess_success_rate = 10) (H3 : required_percentage = 90) : 
  ∃ (x : ℕ), x >= 712 ∧ x = (λ x, ⌈640 / 0.90⌉) :=
by sorry

end minimal_words_to_learn_l60_60303


namespace ratio_largest_sum_approx_14_l60_60268

def geometric_sum (a r n : ℕ) : ℕ :=
  (a * (r ^ n - 1)) / (r - 1)

theorem ratio_largest_sum_approx_14 :
  let S := geometric_sum 1 15 10,
      largest := 15 ^ 10,
      R := (largest * 14) / S
  in R = 14 :=
by
  sorry

end ratio_largest_sum_approx_14_l60_60268


namespace susan_took_longer_l60_60206
variables (M S J T x : ℕ)
theorem susan_took_longer (h1 : M = 2 * S)
                         (h2 : S = J + x)
                         (h3 : J = 30)
                         (h4 : T = M - 7)
                         (h5 : M + S + J + T = 223) : x = 10 :=
sorry

end susan_took_longer_l60_60206


namespace exists_pairs_with_equal_sums_and_product_difference_l60_60071

theorem exists_pairs_with_equal_sums_and_product_difference (N : ℕ) :
  ∃ a1 b1 a2 b2 : ℕ, a1 + b1 = a2 + b2 ∧ (a2 * b2 - a1 * b1 = N) :=
begin
  -- Skipping the proof body, as it’s not required.
  sorry,
end

end exists_pairs_with_equal_sums_and_product_difference_l60_60071


namespace hyperbola_eccentricity_range_l60_60323

theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∃ x y : ℝ, x^2 + y^2 = b^2 ∧ x^2 / a^2 - y^2 / b^2 = 1) :
  ∃ e : ℝ, e = a * sqrt(2) ∧ e ∈ [sqrt(2), +∞) :=
sorry

end hyperbola_eccentricity_range_l60_60323


namespace correct_total_distance_l60_60451

theorem correct_total_distance (km_to_m : 3.5 * 1000 = 3500) (add_m : 3500 + 200 = 3700) : 
  3.5 * 1000 + 200 = 3700 :=
by
  -- The proof would be filled here.
  sorry

end correct_total_distance_l60_60451


namespace reciprocal_of_neg_two_l60_60897

noncomputable theory

def is_reciprocal (a x : ℝ) := a * x = 1

theorem reciprocal_of_neg_two : 
  ∃ x : ℝ, is_reciprocal (-2) x ∧ x = -1 / 2 := 
sorry

end reciprocal_of_neg_two_l60_60897


namespace tom_jerry_coffee_total_same_amount_total_coffee_l60_60145

noncomputable def total_coffee_drunk (x : ℚ) : ℚ := 
  let jerry_coffee := 1.25 * x
  let tom_drinks := (2/3) * x
  let jerry_drinks := (2/3) * jerry_coffee
  let jerry_remainder := (5/12) * x
  let jerry_gives_tom := (5/48) * x + 3
  tom_drinks + jerry_gives_tom

theorem tom_jerry_coffee_total (x : ℚ) : total_coffee_drunk x = jerry_drinks + (1.25 * x - jerry_gives_tom) := sorry

theorem same_amount_total_coffee (x : ℚ) 
  (h : total_coffee_drunk x = (5/4) * x - ((5/48) * x + 3)) : 
  (1.25 * x + x = 36) :=
by sorry

end tom_jerry_coffee_total_same_amount_total_coffee_l60_60145


namespace roots_quadratic_expression_l60_60700

theorem roots_quadratic_expression (α β : ℝ) (hα : α^2 - 3 * α - 2 = 0) (hβ : β^2 - 3 * β - 2 = 0) :
    7 * α^4 + 10 * β^3 = 544 := 
sorry

end roots_quadratic_expression_l60_60700


namespace probability_B_inter_A_l60_60580

-- Define the sample space and events as sets
def sample_space := { (c1: Bool, c2: Bool) | true }
def A := { (c1: Bool, c2: Bool) | c1 = true ∨ c2 = true } -- At least one boy (true represents a boy)
def B := { (c1: Bool, c2: Bool) | c1 = true ∧ c2 = true } -- Both are boys

-- Formalize the given problem statement in Lean 4
theorem probability_B_inter_A : (|B ∩ A| : ℝ) / (|sample_space| : ℝ) = 1/4 := by
  sorry

end probability_B_inter_A_l60_60580


namespace part1_part2_l60_60336

theorem part1 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (hA : A + B + C = π) 
  (ha : a = 2) 
  (hcosC : Real.cos C = -1 / 4) 
  (hsinA_sinB : Real.sin A = 2 * Real.sin B) : b = 1 ∧ c = Real.sqrt 6 := 
  sorry

theorem part2
  (A B C : ℝ) 
  (a b c : ℝ) 
  (hA : A + B + C = π) 
  (ha : a = 2) 
  (hcosC : Real.cos C = -1 / 4)
  (hcosA_minus_pi_div_4 : Real.cos (A - π / 4) = 4 / 5) : c = 5 * Real.sqrt 30 / 2 := 
  sorry

end part1_part2_l60_60336


namespace percent_increase_correct_l60_60152

def cost_after_increase (initial_cost : ℝ) (first_increase : ℝ) (second_increase : ℝ) : ℝ :=
  let cost_after_first_increase := initial_cost * (1 + first_increase)
  in cost_after_first_increase * (1 + second_increase)

def main_costs : ℝ × ℝ × ℝ := (200, 50, 10)
def first_year_increases : ℝ × ℝ × ℝ := (0.08, 0.12, 0.20)
def second_year_increases : ℝ × ℝ × ℝ := (0.06, 0.08, 0.10)

def total_initial_cost (costs : ℝ × ℝ × ℝ) : ℝ :=
  costs.1 + costs.2 + costs.3

def total_new_cost (initial_costs : ℝ × ℝ × ℝ) (first_increases : ℝ × ℝ × ℝ) (second_increases : ℝ × ℝ × ℝ) : ℝ :=
  (cost_after_increase initial_costs.1 first_increases.1 second_increases.1) +
  (cost_after_increase initial_costs.2 first_increases.2 second_increases.2) +
  (cost_after_increase initial_costs.3 first_increases.3 second_increases.3)

def percent_increase (initial_cost : ℝ) (new_cost : ℝ) : ℝ :=
  ((new_cost - initial_cost) / initial_cost) * 100

theorem percent_increase_correct :
  percent_increase (total_initial_cost main_costs) (total_new_cost main_costs first_year_increases second_year_increases) = 16.43 := 
  by
    sorry

end percent_increase_correct_l60_60152


namespace number_of_intersection_points_l60_60497

open Real

def parametric_line (t : ℝ) : ℝ × ℝ := (2 * t, t)
def parametric_curve (θ : ℝ) : ℝ × ℝ := (2 + cos θ, sin θ)

theorem number_of_intersection_points : 
  ∃ (x1 y1 x2 y2 : ℝ), 
  (x1, y1) = parametric_line y1 ∧ (x1, y1) = parametric_curve (arcsin y1) ∧
  (x2, y2) = parametric_line y2 ∧ (x2, y2) = parametric_curve (arccos (x2 - 2 + sin 0)) ∧
  (x1, y1) ≠ (x2, y2) :=
sorry

end number_of_intersection_points_l60_60497


namespace range_of_m_l60_60923

theorem range_of_m (m : ℝ) (h : ∀ x : ℤ, x > m ∧ x < 4 → (x ∈ {0, 1, 2, 3})) : -1 ≤ m ∧ m < 0 :=
sorry

end range_of_m_l60_60923


namespace breadth_of_rectangular_plot_l60_60556

theorem breadth_of_rectangular_plot (b : ℝ) (h1 : 3 * b * b = 972) : b = 18 :=
sorry

end breadth_of_rectangular_plot_l60_60556


namespace Petya_wins_odd_l60_60961

-- Define the game conditions
def game_conditions (n : ℕ) (h : n ≥ 3): Prop :=
  (∀ m: ℕ, m ≠ n → m % 2 = 1) -> Petya_wins n

-- Define Petya's win condition
def Petya_wins : ℕ → Prop
| n := n % 2 = 1

-- The theorem stating that Petya wins for odd n
theorem Petya_wins_odd (n : ℕ) (h₁ : n ≥ 3) (h₂ : n % 2 = 1) : Petya_wins n :=
by
  sorry


end Petya_wins_odd_l60_60961


namespace ABCD_is_trapezoid_or_parallelogram_l60_60833

-- Definitions and conditions from step a
variables {A B C D M N : Type*} [AffineSpace A] [LinearMapClass B] [LinearMapClass C] [LinearMapClass D]

-- Quadrilateral ABCD and segment MN assumptions
variables (ABCD : ConvexQuad A B C D) (MN : LineSegment M N)
          (M_on_AB : PointOnLineSegment M A B) (N_on_CD : PointOnLineSegment N C D)

-- Similar quadrilaterals condition
variables (similar_AMND_NMBC : SimilarQuadrilateral (A M N D) (N M B C))

-- Goal
theorem ABCD_is_trapezoid_or_parallelogram : is_trapezoid ABCD ∨ is_parallelogram ABCD :=
by
  sorry

end ABCD_is_trapezoid_or_parallelogram_l60_60833


namespace factorial_div_9_4_l60_60334

theorem factorial_div_9_4 :
  (9! / 4!) = 15120 :=
by
  have h₁ : 9! = 362880 := by sorry
  have h₂ : 4! = 24 := by sorry
  rw [h₁, h₂]
  norm_num

end factorial_div_9_4_l60_60334


namespace reciprocal_of_neg_two_l60_60907

variable a : ℤ

theorem reciprocal_of_neg_two (h : a = -2) : ∃ b : ℚ, a * b = 1 ∧ b = -1 / 2 := 
by
  exists (-1 / 2)
  rw [h]
  split
  · norm_num
  · refl

end reciprocal_of_neg_two_l60_60907


namespace average_speed_l60_60958

theorem average_speed
  (distance1 : ℝ)
  (time1 : ℝ)
  (distance2 : ℝ)
  (time2 : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (average_speed : ℝ)
  (h1 : distance1 = 90)
  (h2 : time1 = 1)
  (h3 : distance2 = 50)
  (h4 : time2 = 1)
  (h5 : total_distance = distance1 + distance2)
  (h6 : total_time = time1 + time2)
  (h7 : average_speed = total_distance / total_time) :
  average_speed = 70 := 
sorry

end average_speed_l60_60958


namespace extreme_value_f_h_has_one_zero_point_l60_60331

noncomputable def f (a x : ℝ) : ℝ := x - a * real.log x
noncomputable def g (x : ℝ) : ℝ := (x ^ 2) / real.exp x
noncomputable def h (x : ℝ) : ℝ := g x - f (-1) x

theorem extreme_value_f {a : ℝ} (ha : 0 < a) :
  ∃ x_min : ℝ, x_min = a ∧ f a a = a - a * real.log a := sorry

theorem h_has_one_zero_point :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < 1 ∧ h x₀ = 0 := sorry

end extreme_value_f_h_has_one_zero_point_l60_60331


namespace probability_of_event_l60_60074

noncomputable def probability_event (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 ∧ -1 ≤ log (1 / 2) (x + 0.5) ∧ log (1 / 2) (x + 0.5) ≤ 1 then 1 else 0

theorem probability_of_event :
  ∫ (x : ℝ) in 0..2, probability_event x = 3 / 4 :=
by
  sorry

end probability_of_event_l60_60074


namespace L_shaped_region_area_l60_60626

noncomputable def area_L_shaped_region (length full_width : ℕ) (sub_length sub_width : ℕ) : ℕ :=
  let area_full_rect := length * full_width
  let small_width := length - sub_length
  let small_height := full_width - sub_width
  let area_small_rect := small_width * small_height
  area_full_rect - area_small_rect

theorem L_shaped_region_area :
  area_L_shaped_region 10 7 3 4 = 49 :=
by sorry

end L_shaped_region_area_l60_60626


namespace reciprocal_of_neg_two_l60_60896

noncomputable theory

def is_reciprocal (a x : ℝ) := a * x = 1

theorem reciprocal_of_neg_two : 
  ∃ x : ℝ, is_reciprocal (-2) x ∧ x = -1 / 2 := 
sorry

end reciprocal_of_neg_two_l60_60896


namespace game_probability_l60_60769

noncomputable def probability_Alex := 2 / 5
noncomputable def probability_Chelsea := 3 / 20
noncomputable def probability_Mel := 9 / 20

noncomputable def probability_sequence :=
  (probability_Alex^4) * (probability_Mel^3) * (probability_Chelsea)

noncomputable def num_orders := Nat.fact 8 / (Nat.fact 4 * Nat.fact 3 * Nat.fact 1)

noncomputable def total_probability :=
  probability_sequence * num_orders

theorem game_probability :
  total_probability = 49 / 50 :=
by
  sorry

end game_probability_l60_60769


namespace steve_bought_3_boxes_of_cookies_l60_60091

variable (total_cost : ℝ)
variable (milk_cost : ℝ)
variable (cereal_cost : ℝ)
variable (banana_cost : ℝ)
variable (apple_cost : ℝ)
variable (chicken_cost : ℝ)
variable (peanut_butter_cost : ℝ)
variable (bread_cost : ℝ)
variable (cookie_box_cost : ℝ)
variable (cookie_box_count : ℝ)

noncomputable def proves_steve_cookie_boxes : Prop :=
  total_cost = 50 ∧
  milk_cost = 4 ∧
  cereal_cost = 3 ∧
  banana_cost = 0.2 ∧
  apple_cost = 0.75 ∧
  chicken_cost = 10 ∧
  peanut_butter_cost = 5 ∧
  bread_cost = (2 * cereal_cost) / 2 ∧
  cookie_box_cost = (milk_cost + peanut_butter_cost) / 3 ∧
  cookie_box_count = (total_cost - (milk_cost + 3 * cereal_cost + 6 * banana_cost + 8 * apple_cost + chicken_cost + peanut_butter_cost + bread_cost)) / cookie_box_cost

theorem steve_bought_3_boxes_of_cookies :
  proves_steve_cookie_boxes 50 4 3 0.2 0.75 10 5 3 ((4 + 5) / 3) 3 :=
by
  sorry

end steve_bought_3_boxes_of_cookies_l60_60091


namespace smallest_possible_theta_l60_60803

theorem smallest_possible_theta
  (a b c : ℝ → ℝ → ℝ)
  (unit_a : ∥a∥ = 1)
  (unit_b : ∥b∥ = 1)
  (unit_c : ∥c∥ = 1)
  (theta : ℝ)
  (beta : ℝ)
  (h_beta : beta = π / 3)
  (h_dot : b • (c × a) = 1/6) :
  theta = Real.arcsin (1/3) := 
sorry

end smallest_possible_theta_l60_60803


namespace reciprocal_of_neg2_l60_60911

-- Define the number
def num : ℤ := -2

-- Define the reciprocal function
def reciprocal (x : ℤ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_neg2 : reciprocal num = -1 / 2 :=
by
  -- Proof is skipped
  sorry

end reciprocal_of_neg2_l60_60911


namespace three_digit_number_proof_l60_60013

noncomputable def is_prime (n : ℕ) : Prop := (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem three_digit_number_proof (H T U : ℕ) (h1 : H = 2 * T)
  (h2 : U = 2 * T^3)
  (h3 : is_prime (H + T + U))
  (h_digits : H < 10 ∧ T < 10 ∧ U < 10)
  (h_nonzero : T > 0) : H * 100 + T * 10 + U = 212 := 
by
  sorry

end three_digit_number_proof_l60_60013


namespace total_kids_on_soccer_field_l60_60925

theorem total_kids_on_soccer_field (initial_kids : ℕ) (joining_kids : ℕ) (total_kids : ℕ)
  (h₁ : initial_kids = 14)
  (h₂ : joining_kids = 22)
  (h₃ : total_kids = initial_kids + joining_kids) :
  total_kids = 36 :=
by
  sorry

end total_kids_on_soccer_field_l60_60925


namespace find_inverse_of_f_at_4_l60_60439

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^2

-- Statement of the problem
theorem find_inverse_of_f_at_4 : ∃ t : ℝ, f t = 4 ∧ t ≤ 1 ∧ t = -1 := by
  sorry

end find_inverse_of_f_at_4_l60_60439


namespace simplify_fraction_l60_60849

theorem simplify_fraction (a : ℝ) (h : a ≠ 0) : 
  (a^2 - 2 + a^(-2)) / (a^2 - a^(-2)) = (a^2 - 1) / (a^2 + 1) :=
by
  sorry

end simplify_fraction_l60_60849


namespace distance_between_centers_l60_60148

-- Define the rectangle dimensions
def width_rectangle : ℝ := 15
def height_rectangle : ℝ := 17

-- Define the diameter of the circles
def diameter_circle : ℝ := 7

-- Define the required distance
def greatest_possible_distance : ℝ := Real.sqrt (8^2 + 10^2)

-- Create a proposition to state the problem
theorem distance_between_centers :
  ∀ (w h d : ℝ), w = 15 → h = 17 → d = 7 →
  (15 - 2 * (d / 2)) ^ 2 + (17 - 2 * (d / 2)) ^ 2 = 164 :=
by
  intros w h d hw hh hd
  rw [hw, hh, hd]
  norm_num
  apply rfl

end distance_between_centers_l60_60148


namespace log_sqrt_defined_range_l60_60312

theorem log_sqrt_defined_range (x: ℝ) : 
  (∃ (y: ℝ), y = (log (5-x) / sqrt (x+2))) ↔ (-2 ≤ x ∧ x < 5) :=
by
  sorry

end log_sqrt_defined_range_l60_60312


namespace integral_proof_l60_60286

noncomputable def integral_solution : ℝ → ℝ := 
  λ x, - (1 / (Real.sqrt 7)) * Real.log ((Real.sqrt 7 + Real.sqrt (7 + x^2)) / x)

theorem integral_proof (x : ℝ) (hx0 : x ≠ 0) (hx : x^2 + 7 ≥ 0) : 
  ∫ t in 0..x, (1 / (t * Real.sqrt (t^2 + 7))) = integral_solution x + C :=
sorry

end integral_proof_l60_60286


namespace reese_spending_l60_60076

-- Definitions used in Lean 4 statement
variable (S : ℝ := 11000)
variable (M : ℝ := 0.4 * S)
variable (A : ℝ := 1500)
variable (L : ℝ := 2900)

-- Lean 4 verification statement
theorem reese_spending :
  ∃ (P : ℝ), S - (P * S + M + A) = L ∧ P * 100 = 20 :=
by
  sorry

end reese_spending_l60_60076


namespace TankA_volume_percent_TankB_l60_60957

-- Define the conditions for the tanks
def TankA_height := 10.0 -- height of tank A in meters
def TankA_circumference := 9.0 -- circumference of tank A in meters
def TankB_height := 9.0 -- height of tank B in meters
def TankB_circumference := 10.0 -- circumference of tank B in meters

-- The volume formula for a right circular cylinder
def volume (r h : ℝ) := Real.pi * r^2 * h

-- Calculate the radii
def radiusA := TankA_circumference / (2 * Real.pi)
def radiusB := TankB_circumference / (2 * Real.pi)

-- Volumes of the tanks
def volumeA := volume radiusA TankA_height
def volumeB := volume radiusB TankB_height

-- The proof problem statement
theorem TankA_volume_percent_TankB :
  ((volumeA / volumeB) * 100) = 90 :=
  sorry -- The proof is not required

end TankA_volume_percent_TankB_l60_60957


namespace find_radius_of_semi_circle_l60_60873

noncomputable def semi_circle_radius (P : ℝ) (π : ℝ) : ℝ :=
  P / (π + 2)

theorem find_radius_of_semi_circle : semi_circle_radius 34.44867077905162 real.pi ≈ 6.7 := sorry

end find_radius_of_semi_circle_l60_60873


namespace john_daily_earnings_l60_60428

theorem john_daily_earnings :
  ∃ x : ℝ, let days_in_april := 30,
               sundays := 4,
               days_walked := days_in_april - sundays,
               total_spent := 50 + 50,
               remaining_money := 160,
               total_money_earned := 160 + total_spent in
           days_walked * x = total_money_earned ∧ x = 10 :=
by
  let days_in_april := 30
  let sundays := 4
  let days_walked := days_in_april - sundays
  let total_spent := 50 + 50
  let remaining_money := 160
  let total_money_earned := remaining_money + total_spent
  use 10
  split
  { sorry }
  { sorry }

end john_daily_earnings_l60_60428


namespace six_digit_palindromic_count_l60_60405

theorem six_digit_palindromic_count : 
  let palindromic_num (n : ℕ) : Prop := 
        n / 100000 = n % 10 ∧
        (n / 10000) % 10 = (n / 10) % 10 ∧
        (n / 1000) % 10 = (n / 100) % 10 
  in
  ∃ n : ℕ, (99999 < n ∧ n < 1000000 ∧ palindromic_num n) → 
  (finset.filter palindromic_num (finset.Icc 100000 999999)).card = 900 := 
by
  sorry

end six_digit_palindromic_count_l60_60405


namespace domain_of_function_l60_60862

theorem domain_of_function :
  {x : ℝ | x ≥ -1 ∧ x ≠ 1 / 2} =
  {x : ℝ | 2 * x - 1 ≠ 0 ∧ x + 1 ≥ 0} :=
by {
  sorry
}

end domain_of_function_l60_60862


namespace complex_point_second_quadrant_l60_60115

open Complex

theorem complex_point_second_quadrant :
  let z := (2 + 3 * Complex.i) / (1 - Complex.i)
  -1 / 2 < 0 ∧ 5 / 2 > 0 :=
by
  sorry

end complex_point_second_quadrant_l60_60115


namespace solve_equation_l60_60636

-- Define the equation
def equation (x : ℝ) : Prop :=
  ((x - 1)^3 * (x - 2)^3 * (x - 3)^3 * (x - 4)^3) / ((x - 2) * (x - 4) * (x - 2)^2) = 64

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  x = 2 + real.sqrt 5 ∨ x = 2 - real.sqrt 5

-- Main theorem statement
theorem solve_equation : ∀ x : ℝ, equation x ↔ solution_set x := by
  intros x
  sorry

end solve_equation_l60_60636


namespace find_PQ_length_l60_60773

-- Defining the problem parameters
variables {X Y Z P Q R : Type}
variables (dXY dXZ dPQ dPR : ℝ)
variable (angle_common : ℝ)

-- Conditions:
def angle_XYZ_PQR_common : Prop :=
  angle_common = 150 ∧ 
  dXY = 10 ∧
  dXZ = 20 ∧
  dPQ = 5 ∧
  dPR = 12

-- Question: Prove PQ = 2.5 given the conditions
theorem find_PQ_length
  (h : angle_XYZ_PQR_common dXY dXZ dPQ dPR angle_common) :
  dPQ = 2.5 :=
sorry

end find_PQ_length_l60_60773


namespace total_value_proof_l60_60793

structure Person :=
  (pennies : ℕ)
  (nickels : ℕ)
  (dimes : ℕ)
  (quarters : ℕ)
  (halfdollars : ℕ)
  (dollars : ℕ)

def value_in_cents (person : Person) : ℕ :=
  person.pennies * 1 +
  person.nickels * 5 +
  person.dimes * 10 +
  person.quarters * 25 +
  person.halfdollars * 50 +
  person.dollars * 100

def total_value (kate john marie george : Person) : ℕ :=
  value_in_cents kate +
  value_in_cents john +
  value_in_cents marie +
  value_in_cents george

theorem total_value_proof :
  let kate := Person.mk 223 156 87 25 7 4;
  let john := Person.mk 388 94 105 45 15 6;
  let marie := Person.mk 517 64 78 63 12 9;
  let george := Person.mk 289 72 132 50 4 3
  in total_value kate john marie george = 16042 :=
by
  sorry

end total_value_proof_l60_60793


namespace part1_part2_l60_60727

noncomputable def A (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 1 = 0}

theorem part1 (b : ℝ) (h : b = 2)
  : {a | ∃ x, x ∈ A a b} = {a | a = 0 ∨ a ≥ 1} := sorry

theorem part2
  : {a b | ∀ x ∈ A a b, False} = {⟨a, b⟩ | a = 0 ∧ b = 0} ∪ {⟨a, b⟩ | a ≠ 0 ∧ b^2 - 4 * a < 0} := sorry

end part1_part2_l60_60727


namespace tangent_and_normal_at_t2_l60_60658

def parametric_curve (t : ℝ) : ℝ × ℝ :=
  ( (1 + t^3) / (t^2 - 1), t / (t^2 - 1) )

def tangent_line (t₀ : ℝ) : ℝ := (parametric_curve t₀).fst

def normal_line (t₀ : ℝ) : ℝ := (parametric_curve t₀).snd

theorem tangent_and_normal_at_t2 :
  (parametric_curve 2).fst = 3 ∧ (parametric_curve 2).snd = 2/3 := by
  sorry

end tangent_and_normal_at_t2_l60_60658


namespace train_speed_kmph_l60_60244

noncomputable def train_length : ℝ := 200
noncomputable def crossing_time : ℝ := 3.3330666879982935

theorem train_speed_kmph : (train_length / crossing_time) * 3.6 = 216.00072 := by
  sorry

end train_speed_kmph_l60_60244


namespace unique_solution_iff_a_eq_10_l60_60744

theorem unique_solution_iff_a_eq_10 (a : ℝ) :
  (∃! x : ℝ, (x * log a * log a - 1) / (x + log a) = x) ↔ a = 10 :=
sorry

end unique_solution_iff_a_eq_10_l60_60744


namespace ratio_area_rect_sq_l60_60492

/-- 
  Given:
  1. The longer side of rectangle R is 1.2 times the length of a side of square S.
  2. The shorter side of rectangle R is 0.85 times the length of a side of square S.
  Prove that the ratio of the area of rectangle R to the area of square S is 51/50.
-/
theorem ratio_area_rect_sq (s : ℝ) 
  (h1 : ∃ r1, r1 = 1.2 * s) 
  (h2 : ∃ r2, r2 = 0.85 * s) : 
  (1.2 * s * 0.85 * s) / (s * s) = 51 / 50 := 
by
  sorry

end ratio_area_rect_sq_l60_60492


namespace B_Bons_wins_probability_l60_60105

theorem B_Bons_wins_probability :
  let roll_six := (1 : ℚ) / 6
  let not_roll_six := (5 : ℚ) / 6
  let p := (5 : ℚ) / 11
  p = (5 / 36) + (25 / 36) * p :=
by
  sorry

end B_Bons_wins_probability_l60_60105


namespace reciprocal_of_neg_two_l60_60885

theorem reciprocal_of_neg_two : ∃ x : ℝ, (-2) * x = 1 ∧ x = -1/2 :=
by
  use -1/2
  split
  · exact (by norm_num : (-2) * (-1/2) = 1)
  · rfl

end reciprocal_of_neg_two_l60_60885


namespace count_divisible_by_sum_of_first_n_l60_60671

-- Define the calculation for the sum of the first n numbers
def sum_of_first_n (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Define the condition for n! divisibility
def is_factorial_divisible (n : ℕ) : Prop :=
  let s := sum_of_first_n n in
    s ∣ n!

-- Define the main theorem
theorem count_divisible_by_sum_of_first_n :
  ∃ (count : ℕ), count = 20 ∧ 
  (∀ n : ℕ, n ≤ 30 → is_factorial_divisible n) :=
by
  sorry

end count_divisible_by_sum_of_first_n_l60_60671


namespace H_H_H_one_eq_three_l60_60588

noncomputable def H : ℝ → ℝ := sorry

theorem H_H_H_one_eq_three :
  H 1 = -3 ∧ H (-3) = 3 ∧ H 3 = 3 → H (H (H 1)) = 3 :=
by
  sorry

end H_H_H_one_eq_three_l60_60588


namespace xiao_ming_selects_cooking_probability_l60_60213

theorem xiao_ming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let probability (event: String) := if event ∈ courses then 1 / (courses.length : ℝ) else 0
  probability "cooking" = 1 / 4 :=
by
  sorry

end xiao_ming_selects_cooking_probability_l60_60213


namespace range_a_l60_60684

-- Define the piecewise function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else log a x

-- Define the increasing condition for the piecewise function f on ℝ
def is_increasing (f : ℝ → ℝ) : Prop := 
  ∀ x y : ℝ, x < y → f x ≤ f y

-- Define the conditions for the parameter a
def conditions (a : ℝ) : Prop :=
  (3 - a > 0) ∧ (a > 1) ∧ (log a 1 ≥ 3 - 2 * a)

theorem range_a (a : ℝ) : conditions a ↔ a ∈ Icc (3 / 2) 3 :=
by { sorry }

end range_a_l60_60684


namespace binomial_x3_term_coefficient_l60_60320

theorem binomial_x3_term_coefficient :
  let a := -∫ (x : ℝ) in - (π / 2) .. (π / 2), cos x in
  (a = 2) → (62- r=3) 
  (∀ (x : ℝ), ((x^2 + a / x)^6 = 160)) :=
begin
  intros : h_cond

),
end

end binomial_x3_term_coefficient_l60_60320


namespace relative_value_ex1_max_value_of_m_plus_n_l60_60674

-- Definition of relative relationship value
def relative_relationship_value (a b n : ℚ) : ℚ := abs (a - n) + abs (b - n)

-- First problem statement
theorem relative_value_ex1 : relative_relationship_value 2 (-5) 2 = 7 := by
  sorry

-- Second problem statement: maximum value of m + n given the relative relationship value is 2
theorem max_value_of_m_plus_n (m n : ℚ) (h : relative_relationship_value m n 2 = 2) : m + n ≤ 6 := by
  sorry

end relative_value_ex1_max_value_of_m_plus_n_l60_60674


namespace geometric_sequence_a9_l60_60415

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

variable (a : ℕ → ℝ)
variable (q : ℝ)

theorem geometric_sequence_a9
  (h_seq : geometric_sequence a q)
  (h2 : a 1 * a 4 = -32)
  (h3 : a 2 + a 3 = 4)
  (hq : ∃ n : ℤ, q = ↑n) :
  a 8 = -256 := 
sorry

end geometric_sequence_a9_l60_60415


namespace required_mass_NaHCO3_is_correct_l60_60535

noncomputable def neutralize_mass_NaHCO3 (volume_H2SO4 : ℝ) (molarity_H2SO4 : ℝ) (molar_mass_NaHCO3 : ℝ) : ℝ :=
  let moles_H2SO4 := volume_H2SO4 * molarity_H2SO4
  let moles_NaHCO3 := 2 * moles_H2SO4
  moles_NaHCO3 * molar_mass_NaHCO3

theorem required_mass_NaHCO3_is_correct :
  neutralize_mass_NaHCO3 0.025 0.125 84 = 0.525 :=
by
  -- You may leave the proof to be constructed by the user
  sorry

end required_mass_NaHCO3_is_correct_l60_60535


namespace eq_x4_inv_x4_l60_60393

theorem eq_x4_inv_x4 (x : ℝ) (h : x^2 + (1 / x^2) = 2) : 
  x^4 + (1 / x^4) = 2 := 
by 
  sorry

end eq_x4_inv_x4_l60_60393


namespace find_x_l60_60018

-- Defining the conditions
def angle_PQR : ℝ := 180
def angle_PQS : ℝ := 125
def angle_QSR (x : ℝ) : ℝ := x
def SQ_eq_SR : Prop := true -- Assuming an isosceles triangle where SQ = SR.

-- The theorem to be proved
theorem find_x (x : ℝ) :
  angle_PQR = 180 → angle_PQS = 125 → SQ_eq_SR → angle_QSR x = 70 :=
by
  intros _ _ _
  sorry

end find_x_l60_60018


namespace find_b_of_square_binomial_l60_60733

theorem find_b_of_square_binomial (b : ℚ) 
  (h : ∃ c : ℚ, ∀ x : ℚ, (3 * x + c) ^ 2 = 9 * x ^ 2 + 21 * x + b) : 
  b = 49 / 4 := 
sorry

end find_b_of_square_binomial_l60_60733


namespace tom_average_speed_l60_60513

-- Definitions based on given conditions
def distance1 : ℝ := 10
def speed1 : ℝ := 12

def distance2 : ℝ := 10
def speed2 : ℝ := 10

def total_distance : ℝ := 20

-- Total time for each segment of the trip
def time1 := distance1 / speed1
def time2 := distance2 / speed2

-- Total time for the entire trip
def total_time := time1 + time2

-- The average speed
def average_speed := total_distance / total_time

-- Statement to prove that the average speed is 120 / 11 miles per hour
theorem tom_average_speed : average_speed = 120 / 11 :=
by 
  sorry

end tom_average_speed_l60_60513


namespace quadrilateral_diagonals_equal_area_l60_60857

theorem quadrilateral_diagonals_equal_area 
  (quad : Type)
  (area : quad → ℝ)
  (diagonals_halve_area : ∀ q : quad, (area q) / 2 = (area (q)) / 2) :
  ∀ q : quad, 
  divided_into_four_equal_parts (q) :=
sorry

end quadrilateral_diagonals_equal_area_l60_60857


namespace range_of_a_l60_60270

def tensor (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 2 → tensor (x - a) x ≤ a + 2) → a ≤ 7 :=
by
  sorry

end range_of_a_l60_60270


namespace exists_positive_integer_n_l60_60836

theorem exists_positive_integer_n :
  ∃ n : ℕ+, ∀ k : ℕ+, 2 ≤ k ∧ k ≤ 10 → 
  ∃ p : ℕ+, 
    let lower := (p + 2015 / 10000) ^ (k : ℝ)
    let upper := (p + 2016 / 10000) ^ (k : ℝ) in
    lower < (n : ℝ) ∧ (n : ℝ) < upper :=
sorry

end exists_positive_integer_n_l60_60836


namespace hare_can_be_killed_in_at_most_2_to_the_N_moves_l60_60219

-- Definitions:

-- A graph with N vertices
variable (N : ℕ)

-- An algorithm that guarantees killing the hare in at most N! moves
def existing_algorithm_kills_hare_in_at_most_N_factorial_moves : Prop :=
  ∃ algorithm : (Σ V : fin N, fin N → list (fin N)) → list (Σ V : fin N, fin N) → ℕ,
  ∀ algorithm_run : list (Σ V : fin N, fin N),
  ∃ move_count : ℕ, move_count ≤ N.factorial ∧
  (∀ position : fin N, ∃ move_step : Σ V : fin N, fin N, move_step ∈ algorithm_run ∧ position = move_step.2)

-- The goal to prove:
theorem hare_can_be_killed_in_at_most_2_to_the_N_moves
  (hunters_have_alg : existing_algorithm_kills_hare_in_at_most_N_factorial_moves N) :
  ∃ algorithm : (Σ V : fin N, fin N → list (fin N)) → list (Σ V : fin N, fin N) → ℕ,
  ∀ algorithm_run : list (Σ V : fin N, fin N),
  ∃ move_count : ℕ, move_count ≤ 2^N ∧
  (∀ position : fin N, ∃ move_step : Σ V : fin N, fin N, move_step ∈ algorithm_run ∧ position = move_step.2) := 
sorry

end hare_can_be_killed_in_at_most_2_to_the_N_moves_l60_60219


namespace distance_between_some_two_points_lt_half_l60_60787

-- Defining the main problem as a theorem statement
theorem distance_between_some_two_points_lt_half (triangle : Triangle) (points : Fin 5 → Point)
  (h_triangle : triangle.is_equilateral)
  (h_side_length : triangle.side_length = 1)
  (h_points_inside : ∀ i, triangle.contains_point (points i)) :
  ∃ (i j : Fin 5), i ≠ j ∧ dist (points i) (points j) < 0.5 :=
sorry

end distance_between_some_two_points_lt_half_l60_60787


namespace problem_statement_l60_60941

theorem problem_statement (x : ℕ) (h : x = 3) :
  (∏ i in (finset.range 16).filter (λ n, n % 2 = 0) - {0}, x^i) /
  (∏ i in (finset.range 11).filter (λ n, n % 3 = 0) - {0}, x^i) = 3^75 :=
by
  have h1 : ∏ i in (finset.range 16).filter (λ n, n % 2 = 0) - {0}, x^i = x^240 := sorry
  have h2 : ∏ i in (finset.range 11).filter (λ n, n % 3 = 0) - {0}, x^i = x^165 := sorry
  rw [h1, h2, ←h]
  sorry

end problem_statement_l60_60941


namespace value_proof_l60_60319

noncomputable def find_value (a b c : ℕ) (h : a + b + c = 240) (h_rat : ∃ (x : ℕ), a = 4 * x ∧ b = 5 * x ∧ c = 7 * x) : Prop :=
  2 * b - a + c = 195

theorem value_proof : ∃ (a b c : ℕ) (h : a + b + c = 240) (h_rat : ∃ (x : ℕ), a = 4 * x ∧ b = 5 * x ∧ c = 7 * x), find_value a b c h h_rat :=
  sorry

end value_proof_l60_60319


namespace find_matrix_triples_elements_l60_60661

theorem find_matrix_triples_elements (M A : Matrix (Fin 2) (Fin 2) ℝ)
  (h : ∀ (a b c d : ℝ), A = ![![a, b], ![c, d]] -> M * A = ![![3 * a, 3 * b], ![3 * c, 3 * d]]) :
  M = ![![3, 0], ![0, 3]] :=
by
  sorry

end find_matrix_triples_elements_l60_60661


namespace arithmetic_sequence_general_formula_arithmetic_sequence_min_sum_l60_60017

theorem arithmetic_sequence_general_formula (a : ℕ → ℝ) (d : ℝ) (h1 : a 4 = -15) (h2 : d = 3) :
  (∀ n, a n = 3 * n - 27) := sorry

theorem arithmetic_sequence_min_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) (h1 : a 4 = -15) (h2 : d = 3)
  (h3 : ∀ n, a n = 3 * n - 27) :
  (∀ n, S n = 1/2 * n * (a 1 + a n)) →
  (∀ n ∈ {8, 9}, S n = -108) := sorry

end arithmetic_sequence_general_formula_arithmetic_sequence_min_sum_l60_60017


namespace find_x_l60_60377

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n : ℕ, (2 * n + 1) * x^n

theorem find_x (x : ℝ) (H : series_sum x = 16) : 
  x = (33 - Real.sqrt 129) / 32 :=
by
  sorry

end find_x_l60_60377


namespace remainder_of_product_eq_one_l60_60813

theorem remainder_of_product_eq_one {n : ℕ} {a b : ℤ} (hn : 0 < n) (h : a * b ≡ 1 [MOD n]) : (a * b) % n = 1 := by
  sorry

end remainder_of_product_eq_one_l60_60813


namespace ratio_of_ripe_mangoes_l60_60429

theorem ratio_of_ripe_mangoes (total_mangoes : ℕ) (unripe_two_thirds : ℚ)
  (kept_unripe_mangoes : ℕ) (mangoes_per_jar : ℕ) (jars_made : ℕ)
  (h1 : total_mangoes = 54)
  (h2 : unripe_two_thirds = 2 / 3)
  (h3 : kept_unripe_mangoes = 16)
  (h4 : mangoes_per_jar = 4)
  (h5 : jars_made = 5) :
  1 / 3 = 18 / 54 :=
sorry

end ratio_of_ripe_mangoes_l60_60429


namespace sum_of_divisors_of_2i3j_eq_540_l60_60130

def sum_of_divisors (x : ℕ) : ℕ := ∑ d in (finset.range (x+1)).filter (λ n, x % n = 0), n

theorem sum_of_divisors_of_2i3j_eq_540 (i j : ℕ) (h : sum_of_divisors (2^i * 3^j) = 540) : i + j = 5 :=
sorry

end sum_of_divisors_of_2i3j_eq_540_l60_60130


namespace arrange_students_teachers_not_adjacent_l60_60729

noncomputable def num_ways_not_adjacent (num_students : ℕ) (num_teachers : ℕ) : ℕ :=
  if num_students = 5 ∧ num_teachers = 2 then
    let student_arrangements := Nat.factorial num_students
    let positions := (num_students + 1).choose num_teachers
    student_arrangements * positions
  else 0

theorem arrange_students_teachers_not_adjacent :
  num_ways_not_adjacent 5 2 = 3600 :=
by
  unfold num_ways_not_adjacent
  simp [Nat.factorial, Nat.choose]
  -- Steps to compute the factorial and combinatorial choices
  have h1 : Nat.factorial 5 = 120 := by norm_num
  have h2 : 6.choose 2 = 15 := by norm_num
  rw [h1, h2]
  norm_num
  -- Final computation
  sorry

end arrange_students_teachers_not_adjacent_l60_60729


namespace AR_perpendicular_PQ_l60_60797

variables (A B C P Q R : Point)
variables (triangle_ABC : Triangle A B C)
variables (h1 : P ∈ interior triangle_ABC)
variables (h2 : Q ∈ interior triangle_ABC)
variables (h3 : ∠ A P C = 90°)
variables (h4 : ∠ A Q B = 90°)
variables (h5 : ∠ A C P = ∠ P B C)
variables (h6 : ∠ A B Q = ∠ Q C B)
variables (h7 : intersection (line B P) (line C Q) = Some R)

theorem AR_perpendicular_PQ : perpendicular (line A R) (line P Q) :=
sorry

end AR_perpendicular_PQ_l60_60797


namespace volume_of_cone_l60_60688

theorem volume_of_cone (p q : ℕ) (hq : q ≠ 0) : 
  let m := (p : ℚ) / q in
  let r := Real.sqrt m in
  let h := Real.sqrt m in
  (V := (1/3) * Real.pi * r^2 * h) in
  V = (1/3) * Real.pi * (p : ℚ)^(3/2) / q^(3/2) :=
by
  let m := (p : ℚ) / q
  let r := Real.sqrt m
  let h := Real.sqrt m
  sorry

end volume_of_cone_l60_60688


namespace log_sum_geometric_sequence_l60_60487

variable {a : ℕ → ℝ} 
variable {r : ℝ} -- Common ratio for the geometric sequence
variable h_geometric : ∀ n : ℕ, a (n + 1) = a n * r
variable h_positive : ∀ n : ℕ, a n > 0
variable h_condition : a 2 * a 8 = 4

theorem log_sum_geometric_sequence : (Real.log 2 (a 1) + Real.log 2 (a 2) + Real.log 2 (a 3) + Real.log 2 (a 4) + Real.log 2 (a 5) + Real.log 2 (a 6) + Real.log 2 (a 7) + Real.log 2 (a 8) + Real.log 2 (a 9)) = 9 := 
by
  sorry

end log_sum_geometric_sequence_l60_60487


namespace complex_ratio_proof_l60_60044

noncomputable def complex_ratio (x y : ℂ) : ℂ :=
  ((x^6 + y^6) / (x^6 - y^6)) - ((x^6 - y^6) / (x^6 + y^6))

theorem complex_ratio_proof (x y : ℂ) (h : ((x - y) / (x + y)) - ((x + y) / (x - y)) = 2) :
  complex_ratio x y = L :=
  sorry

end complex_ratio_proof_l60_60044


namespace fraction_of_girls_at_soccer_match_l60_60057

theorem fraction_of_girls_at_soccer_match (students_Magnolia students_Jasmine : ℕ)
    (boys_ratio_Magnolia girls_ratio_Magnolia : ℕ)
    (boys_ratio_Jasmine girls_ratio_Jasmine : ℕ)
    (h_Magnolia : students_Magnolia = 160) (h_Jasmine : students_Jasmine = 225)
    (h_ratio_Magnolia : boys_ratio_Magnolia = 3 ∧ girls_ratio_Magnolia = 5)
    (h_ratio_Jasmine : boys_ratio_Jasmine = 2 ∧ girls_ratio_Jasmine = 7) :
    (let total_girls := (5 * (160 / (3 + 5))) + (7 * (225 / (2 + 7))) in
     let total_students := 160 + 225 in
     total_girls / total_students = 5 / 7) :=
by
  sorry

end fraction_of_girls_at_soccer_match_l60_60057


namespace point_distance_and_midpoint_l60_60938
open Real

def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  sqrt ((x₂ - x₁) ^ 2 + (y₂ - y₁) ^ 2)

def midpoint (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ :=
  ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

theorem point_distance_and_midpoint :
  let p1 := (-3 : ℝ, 7 : ℝ)
  let p2 := (4 : ℝ, -9 : ℝ)
  distance p1.1 p1.2 p2.1 p2.2 = sqrt 305 ∧ midpoint p1.1 p1.2 p2.1 p2.2 = (1 / 2, -1) := by
  sorry

end point_distance_and_midpoint_l60_60938


namespace initial_books_from_library_l60_60069

-- Definitions of the problem conditions
def booksGivenAway : ℝ := 23.0
def booksLeft : ℝ := 31.0

-- Statement of the problem, proving that the initial number of books
def initialBooks (x : ℝ) : Prop :=
  x = booksGivenAway + booksLeft

-- Main theorem
theorem initial_books_from_library : initialBooks 54.0 :=
by
  -- Proof pending
  sorry

end initial_books_from_library_l60_60069


namespace factorial_mod_sum_l60_60391

theorem factorial_mod_sum :
  (1! + 2! + 3! + 4! + 5! + 6! + 7!) % 12 = 9 :=
by
  -- Definitions of factorial values for n <= 3
  have h1 : 1! = 1 := by simp
  have h2 : 2! = 2 := by simp
  have h3 : 3! = 6 := by simp
  -- Factorials for n >= 4 should be divisible by 12
  have h4 : 4! % 12 = 0 := by norm_num
  have h5 : 5! % 12 = 0 := by norm_num
  have h6 : 6! % 12 = 0 := by norm_num
  have h7 : 7! % 12 = 0 := by norm_num
  -- Summing and applying modulo operation
  have sum_fact : (1! + 2! + 3!) % 12 = 9 := by norm_num
  exact sum_fact

end factorial_mod_sum_l60_60391


namespace circumcircle_radius_of_triangle_l60_60786

theorem circumcircle_radius_of_triangle 
  (A B C : ℝ)
  (angle_A : A = π / 3)
  (AB : B = 2)
  (AC : C = 3) :
  ∃ R, R = sqrt 21 / 3 :=
sorry

end circumcircle_radius_of_triangle_l60_60786


namespace strictly_increasing_interval_l60_60272

noncomputable def f (x : ℝ) := log 1/4 (-x^2 + 2 * x + 3)

theorem strictly_increasing_interval :
  ∀ x, 1 ≤ x ∧ x < 3 → strict_mono_incr_on f [1, 3) :=
by
  sorry

end strictly_increasing_interval_l60_60272


namespace initial_points_count_l60_60080

theorem initial_points_count (k : ℕ) (h : (4 * k - 3) = 101): k = 26 :=
by 
  sorry

end initial_points_count_l60_60080


namespace jasmine_solution_water_added_l60_60567

theorem jasmine_solution_water_added 
    (initial_volume : ℝ) 
    (initial_jasmine_fraction : ℝ)
    (added_jasmine : ℝ)
    (new_jasmine_fraction : ℝ)
    (W : ℝ) 
    (h_initial_volume : initial_volume = 100)
    (h_initial_jasmine_fraction : initial_jasmine_fraction = 0.10)
    (h_added_jasmine : added_jasmine = 5)
    (h_new_jasmine_fraction : new_jasmine_fraction = 0.08695652173913043)
    (h_total_new_volume : initial_volume + added_jasmine + W = 100 + 5 + W) :
    initial_jasmine_fraction * initial_volume + added_jasmine = new_jasmine_fraction * (initial_volume + added_jasmine + W) :=
by
  -- Given the conditions in the problem
  have h1 : initial_jasmine_fraction * initial_volume = 10 := by
    rw [h_initial_volume, h_initial_jasmine_fraction]
    norm_num

  have h2 : 10 + added_jasmine = 15 := by
    rw [h_added_jasmine]
    norm_num

  have h3 : 15 = new_jasmine_fraction * (100 + 5 + W) := by
    rw [←h1, ←h2, h_new_jasmine_fraction, h_total_new_volume]
    sorry -- Remaining parts skipped

  sorry -- Final conclusion skipped

end jasmine_solution_water_added_l60_60567


namespace exists_indices_l60_60043

-- Define the problem conditions
variable {n : ℕ} {k : ℕ}
variable {S : Fin n → Set ℕ}
variable {x : Fin n → ℕ} 

-- Assume non-negative integers
axiom S_nonneg : ∀ i, ∀ a ∈ S i, 0 ≤ a

-- x_i is the sum of elements in S_i
axiom x_def : ∀ i, x i = S i.sum id

-- Inequality condition
axiom inequality_condition : 
  1 < k ∧ k < n ∧
  (∑ i, x i) ≤ (1 / (k + 1)) * (k * ((n * (n + 1) * (2 * n + 1)) / 6) - ((k + 1) ^ 2 * (n * (n + 1)) / 2))

-- Prove that there exist indices i, j, t, l such that x_i + x_j = x_i + x_l with i ≠ j, i ≠ l, and j ≠ l
theorem exists_indices : 
  ∃ (i j t l : Fin n), i ≠ j ∧ i ≠ l ∧ j ≠ l ∧ x i + x j = x i + x l := 
sorry

end exists_indices_l60_60043


namespace reciprocal_of_neg_two_l60_60901

noncomputable theory

def is_reciprocal (a x : ℝ) := a * x = 1

theorem reciprocal_of_neg_two : 
  ∃ x : ℝ, is_reciprocal (-2) x ∧ x = -1 / 2 := 
sorry

end reciprocal_of_neg_two_l60_60901


namespace packs_of_string_cheese_l60_60430

theorem packs_of_string_cheese (cost_per_piece: ℕ) (pieces_per_pack: ℕ) (total_cost_dollars: ℕ) 
                                (h1: cost_per_piece = 10) 
                                (h2: pieces_per_pack = 20) 
                                (h3: total_cost_dollars = 6) : 
  (total_cost_dollars * 100) / (cost_per_piece * pieces_per_pack) = 3 := 
by
  -- Insert proof here
  sorry

end packs_of_string_cheese_l60_60430


namespace define_interval_l60_60306

theorem define_interval (x : ℝ) : 
  (0 < x + 2) → (0 < 5 - x) → (-2 < x ∧ x < 5) :=
by
  intros h1 h2
  sorry

end define_interval_l60_60306


namespace solve_inequality_l60_60089

theorem solve_inequality (x : ℝ) : (3 * x - 5) / 2 > 2 * x → x < -5 :=
by
  sorry

end solve_inequality_l60_60089


namespace reciprocal_of_neg_two_l60_60889

theorem reciprocal_of_neg_two : ∃ y : ℝ, (-2) * y = 1 ∧ y = -1/2 :=
by
  use -1/2
  split
  · -- Prove that (-2) * (-1/2) = 1
    sorry
  · -- Prove that y = -1/2
    rfl

end reciprocal_of_neg_two_l60_60889


namespace profit_percentage_after_cost_increase_is_approx_56_34_l60_60762

def initial_cost := 100.0
def initial_profit_percentage := 1.50
def selling_price := initial_cost * (1 + initial_profit_percentage)

def increased_cost (initial_cost : Float) (increase_percentage : Float) : Float :=
  initial_cost * (1 + increase_percentage)

def new_cost : Float :=
  let meat := increased_cost 30.0 0.12
  let vegetables := increased_cost 25.0 0.10
  let dairy := increased_cost 20.0 0.08
  let grains := increased_cost 20.0 0.06
  let labor := increased_cost 5.0 0.05
  meat + vegetables + dairy + grains + labor

def new_profit : Float :=
  selling_price - new_cost

def new_profit_percentage : Float :=
  (new_profit / selling_price) * 100

theorem profit_percentage_after_cost_increase_is_approx_56_34
  (initial_cost : Float)
  (initial_profit_percentage : Float)
  (selling_price := initial_cost * (1 + initial_profit_percentage))
  (new_cost := increased_cost 30.0 0.12 + increased_cost 25.0 0.10 +
                increased_cost 20.0 0.08 + increased_cost 20.0 0.06 +
                increased_cost 5.0 0.05)
  (new_profit := selling_price - new_cost)
  (new_profit_percentage := (new_profit / selling_price) * 100)
  : Abs (new_profit_percentage - 56.34) < 0.01 := sorry

end profit_percentage_after_cost_increase_is_approx_56_34_l60_60762


namespace cross_section_polygon_even_sides_l60_60759

def convex_polyhedron (P : Type) [finite_dimensional ℝ P] [convex ℝ P] : Prop :=
  ∀ v ∈ vertices_of P, even (degree_of_vertex v)

def cross_section_even_sided (P : Type) [finite_dimensional ℝ P] [convex ℝ P] (π : affine_subspace ℝ P) : Prop :=
  ¬ ∃ v ∈ vertices_of P, v ∈ π ∧ ∃ polygon ⊆ (intersection_of P π), even (number_of_sides polygon)

theorem cross_section_polygon_even_sides
  (P : Type) [finite_dimensional ℝ P] [convex ℝ P]
  (h1 : convex_polyhedron P)
  (π : affine_subspace ℝ P) (h2 : ¬ ∃ v ∈ vertices_of P, v ∈ π) :
  cross_section_even_sided P π :=
sorry

end cross_section_polygon_even_sides_l60_60759


namespace tv_width_40_inch_l60_60058

theorem tv_width_40_inch (d : ℝ) (a b : ℝ) : 
  a / (real.sqrt (a^2 + b^2)) * 40 = 34.84 :=
by
  have h_aspect_ratio : a / b = 16 / 9 := sorry
  have h_diagonal : d = 40 := sorry
  have h_calc_aspect_ratio : (16 / 9)^2 = ((a / b)^2) := sorry
  have h_diag_via_pythagorean : (a / (real.sqrt (a^2 + b^2))) * 40 = 34.84 := sorry
  exact h_diag_via_pythagorean

end tv_width_40_inch_l60_60058


namespace sum_of_vertices_l60_60924

theorem sum_of_vertices (num_triangle num_hexagon : ℕ) (vertices_triangle vertices_hexagon : ℕ) :
  num_triangle = 1 → vertices_triangle = 3 →
  num_hexagon = 3 → vertices_hexagon = 6 →
  num_triangle * vertices_triangle + num_hexagon * vertices_hexagon = 21 :=
by
  intros h1 h2 h3 h4
  sorry

end sum_of_vertices_l60_60924


namespace complement_of_set_a_range_of_real_number_l60_60726

-- Problem 1
theorem complement_of_set_a (a : ℝ) (x : ℝ) (h_a : a = 1) :
  let A := { x : ℝ | 0 < a * x + 1 ∧ a * x + 1 ≤ 3 }
  let B := { x : ℝ | -1 / 2 < x ∧ x < 2 }
  let complement_A_B := { x : ℝ | -1 < x ∧ x ≤ 1 / 2 } ∪ {2}
  (x ∈ A) → (x ∈ B) → (x ∈ complement_A_B) := 
sorry

-- Problem 2
theorem range_of_real_number (a : ℝ) :
  let A := { x : ℝ | 0 < a * x + 1 ∧ a * x + 1 ≤ 3 }
  let B := { x : ℝ | -1 / 2 < x ∧ x < 2 }
  (A ∩ B = A) → (a ∈ Set.Ioo (-∞) (-4) ∪ Set.Ici (2)) :=
sorry

end complement_of_set_a_range_of_real_number_l60_60726


namespace bricks_in_chimney_900_l60_60611

theorem bricks_in_chimney_900 (h : ℕ) :
  let Brenda_rate := h / 9
  let Brandon_rate := h / 10
  let combined_rate := (Brenda_rate + Brandon_rate) - 10
  5 * combined_rate = h → h = 900 :=
by
  intros Brenda_rate Brandon_rate combined_rate
  sorry

end bricks_in_chimney_900_l60_60611


namespace impossible_cover_8x8_with_tiles_l60_60459

theorem impossible_cover_8x8_with_tiles :
  ¬ ∃ (tiles : List (Σ n : ℕ, fin n × fin n → bool)),
      (∀ tile ∈ tiles, tile.1 = 4 ∨ tile.1 = 2) ∧
      (tiles.filter (λ t, t.1 = 4)).length = 15 ∧
      (tiles.filter (λ t, t.1 = 2)).length = 1 ∧
      covers_grid 8 8 tiles :=
sorry

end impossible_cover_8x8_with_tiles_l60_60459


namespace find_m_l60_60326

theorem find_m 
  (m : ℝ) 
  (h₁ : ∀ x y : ℝ, (x + y = m) → (x^2 + y^2 = 1) → (∃ P Q: (ℝ × ℝ), 
                  (P.1 + P.2 = m) ∧ (P.1^2 + P.2^2 = 1) ∧ 
                  (Q.1 + Q.2 = m) ∧ (Q.1^2 + Q.2^2 = 1) ∧ 
                  (angle (O P Q) = 120)))
  (h₂ : m > 0) :
  m = (Real.sqrt 2) / 2 :=
by
  sorry

end find_m_l60_60326


namespace find_t_l60_60641

-- Define the context of the problem
variables (c o u n t s : ℕ) -- Representing the non-zero digits as natural numbers

-- Conditions of the problem
axioms (h1 : c + o = u)
       (h2 : u + n = t)
       (h3 : t + c = s)
       (h4 : o + n + s = 12)

-- Theorem to prove t = 6
theorem find_t : t = 6 :=
by sorry

end find_t_l60_60641


namespace value_of_x_plus_inv_x_l60_60010

theorem value_of_x_plus_inv_x (x : ℝ) (h : x + (1 / x) = v) (hr : x^2 + (1 / x)^2 = 23) : v = 5 :=
sorry

end value_of_x_plus_inv_x_l60_60010


namespace arithmetic_sequence_y_value_l60_60265

theorem arithmetic_sequence_y_value (y : ℚ) :
  ∃ y : ℚ, 
    (y - 2) - (2/3) = (4 * y - 1) - (y - 2) → 
    y = 11/6 := by
  sorry

end arithmetic_sequence_y_value_l60_60265


namespace a_7_eq_3_l60_60768

-- Define the arithmetic sequence and its properties
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a (i + 1)

-- Given condition
def S_13_eq_39 (a : ℕ → ℝ) : Prop :=
  sum_arithmetic_sequence a 13 = 39

-- Goal: Prove that a_7 = 3
theorem a_7_eq_3 (a : ℕ → ℝ) (h : arithmetic_sequence a) (hS : S_13_eq_39 a) : a 7 = 3 := 
  sorry

end a_7_eq_3_l60_60768


namespace eq_triangle_intersects_circle_l60_60411

theorem eq_triangle_intersects_circle (a : ℝ) :
  (∃ (A B : ℝ × ℝ),
    (∃ x y : ℝ, (a * x + y - 2 = 0) ∧ ((x - 1)^2 + (y - a)^2 = 16 / 3) ∧
    (∃ C : ℝ × ℝ, C = (1, a) ∧ equilateral_triangle A B C)) → a = 0 :=
sorry

end eq_triangle_intersects_circle_l60_60411


namespace fish_ratio_l60_60062

noncomputable def fish_ratio_problem : Prop :=
  let mike : ℕ := 30
  let bob (jim : ℕ) : ℕ := (3 / 2 * jim).toNat
  let total_fish (jim mike bob: ℕ) : ℕ := 
    (2 / 3 * mike).toNat + (2 / 3 * jim).toNat + (2 / 3 * bob).toNat + (1 / 3 * jim).toNat
  ∃ (jim : ℕ), total_fish jim mike (bob jim) = 140 ∧ (jim / mike) = 4

theorem fish_ratio (jim mike bob: ℕ) (h1: mike = 30) (h2: bob = (3 / 2 * jim).toNat)
(h3: total_fish jim mike bob = 140) : (jim / mike = 4) :=
by {
  sorry
}

end fish_ratio_l60_60062


namespace monotonic_increasing_interval_l60_60868

theorem monotonic_increasing_interval (t : ℝ) (h : 2 * t - t ^ 2 ≥ 0) :
  (1 < t) → (t < 2) → y = (1/3) ^ ℝ.sqrt (2 * t - t ^ 2) :=
sorry

end monotonic_increasing_interval_l60_60868


namespace log_div_sqrt_defined_l60_60311

theorem log_div_sqrt_defined (x : ℝ) : -2 < x ∧ x < 5 ↔ ∃ y : ℝ, y = x ∧ ∃ z : ℝ, z = 5-x ∧ log(z) / sqrt(x+2) ∈ ℝ :=
by
  sorry

end log_div_sqrt_defined_l60_60311


namespace log_div_sqrt_defined_l60_60310

theorem log_div_sqrt_defined (x : ℝ) : -2 < x ∧ x < 5 ↔ ∃ y : ℝ, y = x ∧ ∃ z : ℝ, z = 5-x ∧ log(z) / sqrt(x+2) ∈ ℝ :=
by
  sorry

end log_div_sqrt_defined_l60_60310


namespace exist_positive_m_l60_60799

theorem exist_positive_m {n p q : ℕ} (hn_pos : 0 < n) (hp_prime : Prime p) (hq_prime : Prime q) 
  (h1 : pq ∣ n ^ p + 2) (h2 : n + 2 ∣ n ^ p + q ^ p) : ∃ m : ℕ, q ∣ 4 ^ m * n + 2 := 
sorry

end exist_positive_m_l60_60799


namespace exists_clique_of_7_l60_60510

variables {Employee : Type} [fintype Employee]

-- Definitions and assumptions drawn from the conditions
def employees : finset Employee := finset.univ
def knows (a b : Employee) : Prop := sorry

axiom total_employees : finset.card employees = 2023
axiom knows_exactly_1686 (e : Employee) : (finset.filter (knows e) employees).card = 1686
axiom symmetric_knowing : ∀ {a b : Employee}, knows a b → knows b a

-- Proof statement
theorem exists_clique_of_7 :
  ∃ S : finset Employee, S.card = 7 ∧ ∀ (a b : Employee), a ∈ S → b ∈ S → a ≠ b → knows a b :=
sorry

end exists_clique_of_7_l60_60510


namespace reciprocal_of_neg_two_l60_60894

theorem reciprocal_of_neg_two : ∃ y : ℝ, (-2) * y = 1 ∧ y = -1/2 :=
by
  use -1/2
  split
  · -- Prove that (-2) * (-1/2) = 1
    sorry
  · -- Prove that y = -1/2
    rfl

end reciprocal_of_neg_two_l60_60894


namespace range_of_a_l60_60355

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 + 2*x else real.log (x + 1)

theorem range_of_a (a : ℝ) (x : ℝ) (h : |f x| ≥ a * x) :
  -2 ≤ a ∧ a ≤ 0 :=
begin
  sorry
end

end range_of_a_l60_60355


namespace reciprocal_of_neg_two_l60_60880

-- Define the condition of reciprocal
def is_reciprocal (y x : ℝ) : Prop := y * x = 1

-- Define the number in question
def y : ℝ := -2

-- Define the expected reciprocal
def x : ℝ := -1 / 2

-- State the theorem
theorem reciprocal_of_neg_two : is_reciprocal y x :=
by
  -- Proof can be provided here
  sorry

end reciprocal_of_neg_two_l60_60880


namespace angle_of_inclination_l60_60363

theorem angle_of_inclination (t : ℝ) (x y : ℝ) :
  (x = 1 + (1/2)*t) →
  (y = 1 + (sqrt 3/2)*t) →
  ∃ θ, θ = π / 3 :=
by
  intros hx hy
  use π / 3
  sorry

end angle_of_inclination_l60_60363


namespace payment_difference_correct_l60_60449

-- Define the given parameters
def principal : ℝ := 8000
def annual_rate : ℝ := 0.08
def compoundings_monthly : ℕ := 12
def first_period_years : ℕ := 5
def second_period_years : ℕ := 10
def compoundings_annual : ℕ := 1

-- Define the formula for compound interest
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- Calculate Plan 1 payments
def plan1_payment : ℝ :=
  let A1 := compound_interest principal annual_rate compoundings_monthly first_period_years
  let payment_5_years := (3 / 4) * A1
  let remaining_balance := (1 / 4) * A1
  let A2 := compound_interest remaining_balance annual_rate compoundings_monthly first_period_years
  payment_5_years + A2

-- Calculate Plan 2 payments
def plan2_payment : ℝ :=
  compound_interest principal annual_rate compoundings_annual second_period_years

-- Calculate the difference
def payment_difference : ℝ :=
  abs (plan2_payment - plan1_payment)

-- Theorem stating that the calculated difference is approximately 3767
theorem payment_difference_correct : abs (payment_difference - 3767) < 1 := sorry

end payment_difference_correct_l60_60449


namespace range_of_a_l60_60738

theorem range_of_a (a : ℝ) (x : ℝ) : (|x + a| < 3) ↔ (2 < x ∧ x < 3) →  a ∈ Icc (-5 : ℝ) (0 : ℝ) := 
sorry

end range_of_a_l60_60738


namespace vector_magnitude_proof_l60_60384

def vector_subtraction (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2, a.3 - b.3)

def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem vector_magnitude_proof : 
  let a := (2, 3, -1)
  let b := (-2, 1, 3)
  in vector_magnitude (vector_subtraction a b) = 6 :=
by 
  let a := (2, 3, -1)
  let b := (-2, 1, 3)
  have sub : vector_subtraction a b = (4, 2, -4) := rfl
  have mag : vector_magnitude (4, 2, -4) = 6 := rfl
  exact mag

end vector_magnitude_proof_l60_60384


namespace find_n_value_l60_60472

theorem find_n_value :
  ∃ (n : ℕ), (a_1 : ℕ) (a_2 : ℕ) ... (a_n : ℕ),
  (n > 0) ∧
  (a_1 + a_2 / 2! + a_3 / 3! + ... + a_n / n! = 1 / 2013^1000) ∧
  (∀ k : ℕ, 2 ≤ k ≤ n → a_k < k) ∧
  (a_n > 0) ∧
  (n = 61000) :=
sorry

end find_n_value_l60_60472


namespace number_of_diagonals_of_polygon_with_120_degree_interior_angle_l60_60328

def interior_angle (n : ℕ) : ℝ := (n.succ : ℝ - 2) * 180 / n.succ

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem number_of_diagonals_of_polygon_with_120_degree_interior_angle :
  ∀ n : ℕ, interior_angle n = 120 → number_of_diagonals n = 9 :=
by 
  intro n
  intro h
  sorry

end number_of_diagonals_of_polygon_with_120_degree_interior_angle_l60_60328


namespace smallest_perimeter_l60_60011

variable (D E F : ℝ)
variable (cos_D : ℝ) (cos_E : ℝ) (cos_F : ℝ) (d e f : ℤ)

def sin_x (cos_x : ℝ) : ℝ := real.sqrt (1 - cos_x^2)

def sides_ratio (sin_D sin_E sin_F : ℝ) : ℝ × ℝ × ℝ :=
let d := sin_D;
let e := sin_E;
let f := sin_F;
(d, e, f)

theorem smallest_perimeter (h1 : cos D = 3 / 5)
                           (h2 : cos E = 1 / 2)
                           (h3 : cos F = -1 / 3)
                           (h_d : d ∈ set.Ici 1) 
                           (h_e : e ∈ set.Ici 1) 
                           (h_f : f ∈ set.Ici 1) :
  d + e + f = 60 := sorry

end smallest_perimeter_l60_60011


namespace product_real_parts_solution_l60_60086

theorem product_real_parts_solution (x : ℂ) (h : x^2 + 2 * x = -1 + 2 * Complex.i) : 
  (re (-1 + (sqrt 5) ^ (1/4) * cos (atan 2 / 2)) * re (-1 - (sqrt 5) ^ (1/4) * cos (atan 2 / 2))) = 
  1 - sqrt 5 * (cos (atan 2 / 2)) ^ 2 :=
sorry

end product_real_parts_solution_l60_60086


namespace area_of_region_l60_60829

def satisfies_abs_eqn (x y : ℝ) : Prop :=
  |4 * x| + |3 * y| + |24 - 4 * x - 3 * y| = 24

def in_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ 4 * x + 3 * y ≤ 24

theorem area_of_region :
  (∀ x y : ℝ, satisfies_abs_eqn x y ↔ in_region x y) →
  (∃ (area : ℝ), area = 24) :=
begin
  intro h,
  -- Proof of area calculation, which we don't need to provide
  sorry
end

end area_of_region_l60_60829


namespace shortest_distance_correct_l60_60502

def shortest_distance_curve_line : ℝ :=
  let curve := λ x : ℝ, 2 + Real.log x
  let line := λ x : ℝ, x + 3
  Real.sqrt 2

theorem shortest_distance_correct :
  ∃ (x y : ℝ), y = 2 + Real.log x ∧ (∀ (p : ℝ × ℝ), p.2 = 2 + Real.log p.1 →
    let d := Real.abs (p.1 - (p.2 - 3)) / Real.sqrt 2 
    in d ≥ shortest_distance_curve_line) ∧ 
  (shortest_distance_curve_line = Real.sqrt 2) :=
by
  sorry

end shortest_distance_correct_l60_60502


namespace component_prob_gt_9_years_l60_60980

noncomputable def componentServiceLifeNormalDist (ξ : ℝ → ℝ) : Prop :=
  ∀ t t', (t < 3 → ξ t = 0.2) ∧ (t' > 9 → ξ t' = 0.2)

def componentParallel (P : ℝ) : ℝ := 1 - P 

theorem component_prob_gt_9_years (ξ : ℝ → ℝ) (P : ℝ) :
  componentServiceLifeNormalDist ξ →
  P = 0.8 → -- Since the probability that each electronic component can not work normally within 9 years is 0.8.
  (componentParallel P)^3 = 0.512 →
  1 - 0.512 = 0.488 :=
by {
  intros hξ hP hProb,
  exact rfl
}

#check component_prob_gt_9_years -- should be valid

end component_prob_gt_9_years_l60_60980


namespace octagon_inscribed_in_circle_area_l60_60230

open Real

-- Definitions based on conditions:
def radius (r : ℝ) := r
def octagon_area (r : ℝ) := 2 * r^2 * sqrt 2

-- The proof statement:
theorem octagon_inscribed_in_circle_area (r : ℝ) : octagon_area r = 2 * r^2 * sqrt 2 :=
sorry

end octagon_inscribed_in_circle_area_l60_60230


namespace reciprocal_of_neg2_l60_60912

-- Define the number
def num : ℤ := -2

-- Define the reciprocal function
def reciprocal (x : ℤ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_neg2 : reciprocal num = -1 / 2 :=
by
  -- Proof is skipped
  sorry

end reciprocal_of_neg2_l60_60912


namespace simplify_expression_l60_60466

variable (m n : ℝ)

theorem simplify_expression : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end simplify_expression_l60_60466


namespace theater_rows_l60_60740

theorem theater_rows (R : ℕ) (h1 : R < 30 → ∃ r : ℕ, r < R ∧ r * 2 ≥ 30) (h2 : R ≥ 29 → 26 + 3 ≤ R) : R = 29 :=
by
  sorry

end theater_rows_l60_60740


namespace isabel_bouquets_l60_60027

theorem isabel_bouquets : 
  ∀ (total_flowers wilted flowers_per_bouquet : ℕ), 
  total_flowers = 66 → wilted = 10 → flowers_per_bouquet = 8 → 
  (total_flowers - wilted) / flowers_per_bouquet = 7 :=
by
  intros total_flowers wilted flowers_per_bouquet ht hw hb
  rw [ht, hw, hb]
  exact Eq.refl 7

end isabel_bouquets_l60_60027


namespace circle_tangent_symmetric_line_dot_product_range_l60_60015

-- Definitions and conditions
def is_tangent (M : ℝ × ℝ) (l : ℝ → ℝ) (r : ℝ) : Prop := 
  distance M l = r

def is_symmetric (M : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  y = -mx + 1 -- line passes through M

def point P : ℝ × ℝ := (x, y)
def Points (A B : ℝ × ℝ) := A = (-2, 0) ∧ B = (2, 0)

-- Problem's Lean statement
theorem circle_tangent (M : ℝ × ℝ) (l : ℝ → ℝ) (r : ℝ) :
  is_tangent (-1, 0) (λ x, x - √3 * y - 3) 2 :=
sorry

theorem symmetric_line (M : ℝ × ℝ) :
  is_symmetric (-1, 0) (λ m, mx + y + 1 = 0) → m = 1 :=
sorry

theorem dot_product_range (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  Points A B → |PA| * |PB| = |PO|^2 → ∀ (x y : ℝ), (x^2 - y^2 = 2) ∧ (x + 1)^2 + y^2 < 4 →
  -2 ≤ x^2 - 4 + y^2 ∧ x^2 - 4 + y^2 < 6 :=
sorry

end circle_tangent_symmetric_line_dot_product_range_l60_60015


namespace sqrt_13_plus_1_parts_l60_60490

theorem sqrt_13_plus_1_parts : 
  let a := Nat.floor (Real.sqrt 13 + 1)
  let b := Real.sqrt 13 + 1 - a in
  a = 4 ∧ b = Real.sqrt 13 - 3 :=
by
  sorry

end sqrt_13_plus_1_parts_l60_60490


namespace correct_statements_l60_60275

-- Definitions based on conditions.
def frequency_is_random (freq : ℕ → ℝ) : Prop := ∀ (n : ℕ), freq n ∈ set.Icc 0 1
def probability_is_constant (prob : ℝ) : Prop := prob ∈ set.Icc 0 1

-- Statements to be evaluated.
def statement1 (freq : ℕ → ℝ) (prob : ℝ) : Prop :=
  ∀ (n : ℕ), freq n = prob

def statement2 (freq : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), freq n = freq (succ n)

def statement3 (freq : ℕ → ℝ) : Prop :=
  ∃ (n : ℕ), freq n ≠ freq 0

def statement4 (freq : ℕ → ℝ) (prob : ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (freq n - prob) < ε

-- The main theorem combining all the statements with correct answers.
theorem correct_statements (freq : ℕ → ℝ) (prob : ℝ) :
  frequency_is_random freq →
  probability_is_constant prob →
  ¬ statement1 freq prob ∧
  ¬ statement2 freq ∧
  statement3 freq ∧
  statement4 freq prob :=
by
  intros h_freq_is_random h_prob_is_constant,
  sorry

end correct_statements_l60_60275


namespace max_cubes_in_box_l60_60155

noncomputable def cube_volume : ℕ := 27
noncomputable def box_length : ℕ := 9
noncomputable def box_width : ℕ := 8
noncomputable def box_height : ℕ := 12
noncomputable def maximum_cubes : ℕ := 24

theorem max_cubes_in_box : 
  let V_box := box_length * box_width * box_height
  let V_cube := cube_volume
  V_box / V_cube = maximum_cubes := by
{
  let V_box := box_length * box_width * box_height,
  let V_cube := cube_volume,
  have h1 : V_box = 9 * 8 * 12 := rfl,
  have h2 : V_cube = 27 := rfl,
  have h3 : V_box / V_cube = 864 / 27 := by rw [h1, h2],
  have h4 : 864 / 27 = 32 := by norm_num,
  have h5 : 32 ≠ 24 := by norm_num,
  have h6 : 24 = 24 := rfl,
  sorry
}

end max_cubes_in_box_l60_60155


namespace min_value_xy_expression_l60_60526

theorem min_value_xy_expression (x y : ℝ) : ∃ c : ℝ, (∀ x y : ℝ, (xy - 1)^2 + (x + y)^2 ≥ c) ∧ c = 1 :=
by {
  -- Placeholder for proof
  sorry
}

end min_value_xy_expression_l60_60526


namespace solve_abs_equation_l60_60504

theorem solve_abs_equation (x : ℝ) (h : |2001 * x - 2001| = 2001) : x = 0 ∨ x = 2 := by
  sorry

end solve_abs_equation_l60_60504


namespace find_a_in_triangle_l60_60781

noncomputable def a (A B C : ℝ) (b c : ℝ) : ℝ := 
  let cos_B_minus_C := (17 : ℝ) / 18
  let cos_A := 8 / 9
  real.sqrt (58 - 42 * cos_A)

theorem find_a_in_triangle :
  ∀ (A B C : ℝ) (b c : ℝ),
    b = 7 →
    c = 3 →
    (real.cos (B - C) = (17 : ℝ) / 18) →
    a A B C b c = 40 / 3 :=
by
  intros A B C b c hb hc hcos
  rw [a, hb, hc, hcos]
  -- Here would be the proof steps
  sorry

end find_a_in_triangle_l60_60781


namespace sum_greater_than_l60_60800

theorem sum_greater_than (n : ℕ) (h_n : 1 ≤ n) (x : Fin (n+2) → ℝ) 
(h_nonneg : ∀ i, 0 ≤ x i)
(h_condition : ∀ i : Fin n, x i * x (Fin.succ i) - x (Fin.pred i) ^ 2 ≥ 1) :
 ∑ i in Finset.range (n + 2), x i > (2 * n / 3) ^ (3 / 2) := sorry

end sum_greater_than_l60_60800


namespace triangle_area_ratio_l60_60418

-- Defining the given right triangle ABC with ∠B = 90°
variables {A B C P M D E N : Type*}

-- Conditions
def is_right_triangle (A B C : Type*) := ∠B = 90 -- Right triangle condition

def point_on_angle_bisector (P : Type*) (A : Type*) (triangle : A ∈ triangle) :=
  P ∈ angle_bisector A ∧ P ∈ interior_triangle

def point_on_side (M : Type*) (side : Type*) :=
  M ∈ side ∧ M ≠ A ∧ M ≠ B

def intersection_points (D E N : Type*) (lines : Type*) := 
  D ∈ (line_through A P ∩ side BC) ∧ E ∈ (line_through C P ∩ side AB) ∧ N ∈ (line_through M P ∩ side AC)

def angle_equalities (MPB PCN NPC MBP : Type*) :=
  ∠MPB = ∠PCN ∧ ∠NPC = ∠MBP

-- Proof problem: Given the conditions above, we need to prove that the area ratio holds true.
theorem triangle_area_ratio (triangle : is_right_triangle A B C) 
  (angle_bisector : point_on_angle_bisector P A triangle) 
  (point_on_AB : point_on_side M AB) 
  (intersections : intersection_points D E N (lines AP CP MP))
  (angles : angle_equalities (∠MPB ∠PCN ∠NPC ∠MBP)) : 
  S_Δ APC / S_ACDE = 1 / 2 :=
sorry

end triangle_area_ratio_l60_60418


namespace problem_1_problem_2_l60_60918

def A (x : ℝ) : Prop := x^2 - 3*x - 10 ≤ 0
def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2*m - 1

theorem problem_1 (m : ℝ) : (∀ x, B m x → A x)  →  m ≤ 3 := 
sorry

theorem problem_2 (m : ℝ) : (¬ ∃ x, A x ∧ B m x) ↔ (m < 2 ∨ 4 < m) := 
sorry

end problem_1_problem_2_l60_60918


namespace equilateral_triangle_is_most_stable_l60_60174

-- Definitions of shapes
structure EquilateralTriangle :=
  (side_length : ℝ)
  (angle : ℝ := 60)

structure Square :=
  (side_length : ℝ)
  (angle : ℝ := 90)

structure Parallelogram :=
  (side_length_1 : ℝ)
  (side_length_2 : ℝ)
  (angle_1 : ℝ)
  (angle_2 : ℝ)

structure Trapezoid :=
  (side_length_1 : ℝ)
  (side_length_2 : ℝ)
  (side_length_3 : ℝ)
  (side_length_4 : ℝ)
  (parallel_sides : Prop)

-- Definition of stability
def isStable (shape : Type) [Inhabited shape] : Prop :=
  -- In reality, you would define a precise mathematical definition of stability here
  sorry

-- Theorem: Equilateral Triangle is the most stable shape
theorem equilateral_triangle_is_most_stable : isStable EquilateralTriangle :=
  sorry

end equilateral_triangle_is_most_stable_l60_60174


namespace interest_rate_l60_60188

/-- 
Given a principal amount that doubles itself in 10 years at simple interest,
prove that the rate of interest per annum is 10%.
-/
theorem interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (h1 : SI = P) (h2 : T = 10) (h3 : SI = P * R * T / 100) : 
  R = 10 := by
  sorry

end interest_rate_l60_60188


namespace correct_product_of_a_and_b_l60_60406

theorem correct_product_of_a_and_b :
  ∃ (a b : ℕ), 
    (10 ≤ a ∧ a < 100) ∧
    (let rev_a := (a % 10) * 10 + a / 10 in rev_a * b = 189) ∧
    (a * b = 108) :=
by
  sorry

end correct_product_of_a_and_b_l60_60406


namespace paint_needed_for_snake_l60_60423

theorem paint_needed_for_snake (cube_count : ℕ) 
  (paint_per_cube : ℕ) 
  (cubes_per_fragment : ℕ) 
  (additional_paint : ℕ) 
  : cube_count = 2016 
  -> paint_per_cube = 60 
  -> cubes_per_fragment = 6 
  -> additional_paint = 20 
  -> let fragments := cube_count / cubes_per_fragment in
     let paint_per_fragment := cubes_per_fragment * paint_per_cube in
     let total_paint := fragments * paint_per_fragment + additional_paint in
     total_paint = 120980 := 
by
  intros h1 h2 h3 h4
  let fragments := cube_count / cubes_per_fragment
  let paint_per_fragment := cubes_per_fragment * paint_per_cube
  let total_paint := fragments * paint_per_fragment + additional_paint
  have hfragments : fragments = 336 := by sorry
  have hpaint_per_fragment : paint_per_fragment = 360 := by sorry
  have htotal_paint : total_paint = 120980 := by sorry
  exact htotal_paint


end paint_needed_for_snake_l60_60423


namespace find_number_l60_60967

theorem find_number (x : ℝ) (h : 120 = 1.5 * x) : x = 80 :=
by
  sorry

end find_number_l60_60967


namespace max_lambda_inequality_l60_60341

theorem max_lambda_inequality 
  (a b x y : ℝ) 
  (h1 : a ≥ 0) 
  (h2 : b ≥ 0)
  (h3 : x ≥ 0)
  (h4 : y ≥ 0)
  (h5 : a + b = 27) : 
  (a * x^2 + b * y^2 + 4 * x * y)^3 ≥ 4 * (a * x^2 * y + b * x * y^2)^2 :=
sorry

end max_lambda_inequality_l60_60341


namespace julie_can_print_100_newspapers_l60_60032

def num_boxes : ℕ := 2
def packages_per_box : ℕ := 5
def sheets_per_package : ℕ := 250
def sheets_per_newspaper : ℕ := 25

theorem julie_can_print_100_newspapers :
  (num_boxes * packages_per_box * sheets_per_package) / sheets_per_newspaper = 100 := by
  sorry

end julie_can_print_100_newspapers_l60_60032


namespace max_three_topping_pizzas_l60_60225

-- Define the combinations function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Assert the condition and the question with the expected answer
theorem max_three_topping_pizzas : combination 8 3 = 56 :=
by
  sorry

end max_three_topping_pizzas_l60_60225


namespace correct_reasoning_type_l60_60163

-- Definitions based on given conditions
def is_analogical_reasoning_specific_to_general : Prop :=
  AnalogicalReasoning → SpecificToGeneral

def is_deductive_reasoning_specific_to_general : Prop :=
  DeductiveReasoning → SpecificToGeneral

def is_inductive_reasoning_specific_to_general : Prop :=
  InductiveReasoning → SpecificToGeneral

def is_emotional_reasoning_valid_in_proof : Prop :=
  EmotionalReasoning → ValidInProof

-- Proof statement
theorem correct_reasoning_type :
  is_inductive_reasoning_specific_to_general ∧
  ¬ is_analogical_reasoning_specific_to_general ∧
  ¬ is_deductive_reasoning_specific_to_general ∧
  ¬ is_emotional_reasoning_valid_in_proof :=
sorry

end correct_reasoning_type_l60_60163


namespace units_digit_base7_product_l60_60095

def remainder (a b : ℕ) : ℕ := a % b

theorem units_digit_base7_product (n m : ℕ) (h1 : n = 301) (h2 : m = 52) :
  remainder (n * m) 7 = 0 :=
by
  rw [h1, h2]
  have h3 : remainder 301 7 = 0 := by norm_num
  have h4 : remainder 52 7 = 3 := by norm_num
  have h5 : remainder (301 * 52) 7 = (0 * 3) % 7 := by rw [Nat.mul_mod, h3, h4]
  simp only [zero_mul, Nat.zero_mod] at h5
  exact h5

end units_digit_base7_product_l60_60095


namespace probability_even_sum_includes_ball_15_l60_60973

-- Definition of the conditions in Lean
def balls : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 15}

def odd_balls : Set ℕ := {n ∈ balls | n % 2 = 1}
def even_balls : Set ℕ := {n ∈ balls | n % 2 = 0}
def ball_15 : ℕ := 15

-- The number of ways to choose k elements from a set of n elements
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Number of ways to draw 7 balls ensuring the sum is even and ball 15 is included
def favorable_outcomes : ℕ :=
  choose 6 5 * choose 8 1 +   -- 5 other odd and 1 even
  choose 6 3 * choose 8 3 +   -- 3 other odd and 3 even
  choose 6 1 * choose 8 5     -- 1 other odd and 5 even

-- Total number of ways to choose 7 balls including ball 15:
def total_outcomes : ℕ := choose 14 6

-- Probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- The proof we require
theorem probability_even_sum_includes_ball_15 :
  probability = 1504 / 3003 :=
by
  -- proof omitted for brevity
  sorry

end probability_even_sum_includes_ball_15_l60_60973


namespace max_gold_coins_l60_60181

theorem max_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 100) : n = 94 := by
  sorry

end max_gold_coins_l60_60181


namespace number_of_classes_is_correct_l60_60143

-- Define the list of heights
def heights : List ℕ := [149, 159, 142, 160, 156, 163, 145, 150, 148, 151,
                         156, 144, 148, 149, 153, 143, 168, 168, 152, 155]

-- Define the class interval
def class_interval : ℕ := 4

-- Define the expected number of classes
def expected_number_of_classes : ℕ := 7

-- Prove that the number of classes is equal to the expected number of classes
theorem number_of_classes_is_correct :
  (let range := heights.maximum?.get! - heights.minimum?.get! + 1 in 
  (range + class_interval - 1) / class_interval = expected_number_of_classes) := 
by 
  sorry

end number_of_classes_is_correct_l60_60143


namespace reciprocal_of_neg_two_l60_60890

theorem reciprocal_of_neg_two : ∃ y : ℝ, (-2) * y = 1 ∧ y = -1/2 :=
by
  use -1/2
  split
  · -- Prove that (-2) * (-1/2) = 1
    sorry
  · -- Prove that y = -1/2
    rfl

end reciprocal_of_neg_two_l60_60890


namespace max_value_condition_l60_60351
    
def f (x a : ℝ) : ℝ :=
  if x < a + 1 then (1 / 2) ^ |x - a|
  else -|x + 1| - a

theorem max_value_condition (a : ℝ) :
  (∀ x : ℝ, f x a ≤ 1) ↔ (- (3 : ℝ) / 2 ≤ a) :=
by
  sorry

end max_value_condition_l60_60351


namespace tangent_line_at_1_1_l60_60656

noncomputable def f (x : ℝ) : ℝ := x / (2 * x - 1)

theorem tangent_line_at_1_1 :
  let m := -((2 * 1 - 1 - 2 * 1) / (2 * 1 - 1)^2) -- Derivative evaluated at x = 1
  let tangent_line (x y : ℝ) := x + y - 2
  ∀ x y : ℝ, tangent_line x y = 0 → (f x = y ∧ x = 1 → y = 1 → m = -1) :=
by
  sorry

end tangent_line_at_1_1_l60_60656


namespace compute_z_pow_6_l60_60034

noncomputable def z : ℂ := (-1 + complex.I * real.sqrt 3) / 2

theorem compute_z_pow_6 : z^6 = 1/4 := 
by
  sorry

end compute_z_pow_6_l60_60034


namespace pencils_to_sell_for_profit_l60_60242

theorem pencils_to_sell_for_profit 
    (total_pencils : ℕ) 
    (buy_price sell_price : ℝ) 
    (desired_profit : ℝ) 
    (h_total_pencils : total_pencils = 2000) 
    (h_buy_price : buy_price = 0.15) 
    (h_sell_price : sell_price = 0.30) 
    (h_desired_profit : desired_profit = 150) :
    total_pencils * buy_price + desired_profit = total_pencils * sell_price → total_pencils = 1500 :=
by
    sorry

end pencils_to_sell_for_profit_l60_60242


namespace evaluate_g_at_neg2_l60_60810

-- Definition of the polynomial g
def g (x : ℝ) : ℝ := 3 * x^5 - 20 * x^4 + 40 * x^3 - 25 * x^2 - 75 * x + 90

-- Statement to prove using the condition
theorem evaluate_g_at_neg2 : g (-2) = -596 := 
by 
   sorry

end evaluate_g_at_neg2_l60_60810


namespace number_of_members_after_four_years_l60_60576

theorem number_of_members_after_four_years (b : ℕ → ℕ) (initial_condition : b 0 = 21) 
    (yearly_update : ∀ k, b (k + 1) = 4 * b k - 9) : 
    b 4 = 4611 :=
    sorry

end number_of_members_after_four_years_l60_60576


namespace circles_properties_l60_60770

-- Define the circles in Cartesian coordinates
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the output polar equations and intersection points
def polar_eq_C1 : Prop := ∀ (ρ θ : ℝ), ρ = 2
def polar_eq_C2 : Prop := ∀ (ρ θ : ℝ), ρ = 4 * Real.cos θ
def intersection_points : Set (ℝ × ℝ) := { (2, 2 * Int.ofNat k * Real.pi + Real.pi / 3) | k : ℤ } ∪ { (2, 2 * Int.ofNat k * Real.pi - Real.pi / 3) | k : ℤ }

-- Define the parametric equation of the common chord
def common_chord : Set (ℝ × ℝ) := { (1, t) | t : ℝ, -Real.sqrt 3 ≤ t ∧ t ≤ Real.sqrt 3 }

-- The theorem that combines the above definitions
theorem circles_properties :
  (polar_eq_C1) ∧
  (polar_eq_C2) ∧
  (∀ p ∈ intersection_points, C1 p.1 p.2 ∧ C2 p.1 p.2) ∧
  (∀ p ∈ common_chord, C1 p.1 p.2 ∧ C2 p.1 p.2) :=
by
  sorry

end circles_properties_l60_60770


namespace BD_value_l60_60408

def quadrilateral_ABCD_sides (AB BC CD DA : ℕ) (BD : ℕ) : Prop :=
  AB = 5 ∧ BC = 17 ∧ CD = 5 ∧ DA = 9 ∧ 12 < BD ∧ BD < 14 ∧ BD = 13

theorem BD_value (AB BC CD DA : ℕ) (BD : ℕ) : 
  quadrilateral_ABCD_sides AB BC CD DA BD → BD = 13 :=
by
  sorry

end BD_value_l60_60408


namespace Pam_bags_count_l60_60826

theorem Pam_bags_count :
  ∀ (apples_in_geralds_bag : ℕ) (multiple_factor : ℕ) (total_apples_pam_has : ℕ) (expected_pam_bags : ℕ),
  (apples_in_geralds_bag = 40) →
  (multiple_factor = 3) →
  (total_apples_pam_has = 1200) →
  (expected_pam_bags = 10) →
  let apples_in_pams_bag := apples_in_geralds_bag * multiple_factor in
  let pam_bags := total_apples_pam_has / apples_in_pams_bag in
  pam_bags = expected_pam_bags :=
by
  intros apples_in_geralds_bag multiple_factor total_apples_pam_has expected_pam_bags
  intros h1 h2 h3 h4
  let apples_in_pams_bag := apples_in_geralds_bag * multiple_factor
  let pam_bags := total_apples_pam_has / apples_in_pams_bag
  rw [h1, h2, h3, h4]
  sorry

end Pam_bags_count_l60_60826


namespace number_of_non_empty_subsets_meeting_condition_l60_60681

open Set

noncomputable def M : Set ℕ := { a, b }

-- Proof statement
theorem number_of_non_empty_subsets_meeting_condition :
  ∀ (N : Set ℕ), (N ⊆ { a, b, c }) →
                (N ≠ ∅) →
                (M ∪ N ⊆ { a, b, c }) →
                (card { P : Set ℕ | P ⊆ { a, b, c } ∧ P ≠ ∅ ∧ M ∪ P ⊆ { a, b, c } } = 7) :=
by
  sorry

end number_of_non_empty_subsets_meeting_condition_l60_60681


namespace cooking_and_weaving_l60_60951

-- Define the given conditions
variables (Y C W C_only C_and_Y All : ℕ)
variables (H1 : Y = 25) (H2 : C = 15) (H3 : W = 8)
variables (H4 : C_only = 2) (H5 : C_and_Y = 7) (H6 : All = 3)

-- Define the statement to prove
theorem cooking_and_weaving : ∃ (C_and_W : ℕ), C_and_W = 6 :=
begin
  use C - (C_and_Y + C_only),
  simp [H2, H4, H5],
  norm_num,
end

end cooking_and_weaving_l60_60951


namespace range_of_a_mono_increase_l60_60714

def f (x : ℝ) := (1 / 2) * x^2 + 2 * x - 2 * log x

theorem range_of_a_mono_increase :
  (∀ x > 0, derivative ℝ f x ≥ 0) ↔ (a ≤ 0) := sorry

end range_of_a_mono_increase_l60_60714


namespace reciprocal_of_neg_two_l60_60908

variable a : ℤ

theorem reciprocal_of_neg_two (h : a = -2) : ∃ b : ℚ, a * b = 1 ∧ b = -1 / 2 := 
by
  exists (-1 / 2)
  rw [h]
  split
  · norm_num
  · refl

end reciprocal_of_neg_two_l60_60908


namespace zero_elements_bound_l60_60187

theorem zero_elements_bound 
    (n : ℕ) (hn : n ≥ 2) 
    (A : Matrix (Fin n) (Fin n) ℝ) 
    (symm_A : Aᵀ = A) 
    (inv_A : Invertible A)
    (pos_A : ∀ i j : Fin n, A i j > 0) :
    let z_n := (A⁻¹).toMatrix.toList.count (λ x => x = 0)
    in z_n ≤ n^2 - 2 * n :=
by
  sorry

end zero_elements_bound_l60_60187


namespace midpoints_and_projections_concyclic_l60_60182

noncomputable theory

variables {A B C D P K L M N P_AB P_BC P_CD P_DA : Type}
variables (A B C D : Point) [InCircle A B C D]
variables [PerpendicularDiagonals A C B D]
variables (P : Intersection A C B D)
variables (K : Midpoint A B) (L : Midpoint B C) (M : Midpoint C D) (N : Midpoint D A)
variables (P_AB : FootPerpendicular P A B) 
variables (P_BC : FootPerpendicular P B C)
variables (P_CD : FootPerpendicular P C D)
variables (P_DA : FootPerpendicular P D A)

theorem midpoints_and_projections_concyclic
  : AreConcyclic K L M N P_AB P_BC P_CD P_DA :=
sorry

end midpoints_and_projections_concyclic_l60_60182


namespace percentage_decrease_l60_60100

theorem percentage_decrease (t : ℝ) (x : ℝ) :
  t = 80 →
  let increased_value := t + 0.125 * t in
  let decreased_value := t - (x / 100) * t in
  increased_value - decreased_value = 30 →
  x = 25 :=
by
  sorry

end percentage_decrease_l60_60100


namespace reciprocal_of_neg_two_l60_60905

variable a : ℤ

theorem reciprocal_of_neg_two (h : a = -2) : ∃ b : ℚ, a * b = 1 ∧ b = -1 / 2 := 
by
  exists (-1 / 2)
  rw [h]
  split
  · norm_num
  · refl

end reciprocal_of_neg_two_l60_60905


namespace cupcakes_difference_l60_60609

theorem cupcakes_difference 
    (B_hrly_rate : ℕ) 
    (D_hrly_rate : ℕ) 
    (B_break : ℕ) 
    (total_hours : ℕ) 
    (B_hrly_rate = 10) 
    (D_hrly_rate = 8) 
    (B_break = 2) 
    (total_hours = 5) :
    (D_hrly_rate * total_hours) - (B_hrly_rate * (total_hours - B_break)) = 10 := 
by sorry

end cupcakes_difference_l60_60609


namespace minimize_r1_r2_l60_60431

noncomputable def circle_touching_x_axis_at_one (C : Type*) [MetricSpace C] [NormedSpace ℝ C] :=
  ∃ c : C, dist (c, (1 : C), 0) = dist (c, 0)

noncomputable def circle_touching_y_axis (C : Type*) [MetricSpace C] [NormedSpace ℝ C] :=
  ∃ c : C, dist (c, 0, (c, 0)) = 0

theorem minimize_r1_r2 {α : Type*} [LinearOrderedField α] (l : α)
  (h_line_origin_pos_slope : ∃ m : α, m > 0 ∧ l = fun x => m * x)
  (h_C1_domain : C_1.x >= 0 ∧ C_1.y >= 0)
  (h_C2_domain : C_2.x >= 0 ∧ C_2.y >= 0)
  (h_C1_C2_touch_l : ∃ p : α × α, p ∈ l ∧ C_1 p = C_2 p)
  (h_C1_touch_x_axis : circle_touching_x_axis_at_one C_1)
  (h_C2_touch_y_axis : circle_touching_y_axis C_2) :
  ∃ m : α, l = fun x => m * x ∧ 8 * C_1.radius + 9 * C_2.radius = 7 :=
sorry

end minimize_r1_r2_l60_60431


namespace degree_sequence_a_invalid_degree_sequence_b_invalid_degree_sequence_c_invalid_all_sequences_invalid_l60_60617

-- Definition of the "isValidGraph" function based on degree sequences
-- Placeholder for the actual definition
def isValidGraph (degrees : List ℕ) : Prop :=
  sorry

-- Degree sequences given in the problem
def d_a := [8, 6, 5, 4, 4, 3, 2, 2]
def d_b := [7, 7, 6, 5, 4, 2, 2, 1]
def d_c := [6, 6, 6, 5, 5, 3, 2, 2]

-- Statement that proves none of these sequences can form a valid graph
theorem degree_sequence_a_invalid : ¬ isValidGraph d_a :=
  sorry

theorem degree_sequence_b_invalid : ¬ isValidGraph d_b :=
  sorry

theorem degree_sequence_c_invalid : ¬ isValidGraph d_c :=
  sorry

-- Final statement combining all individual proofs
theorem all_sequences_invalid :
  ¬ isValidGraph d_a ∧ ¬ isValidGraph d_b ∧ ¬ isValidGraph d_c :=
  ⟨degree_sequence_a_invalid, degree_sequence_b_invalid, degree_sequence_c_invalid⟩

end degree_sequence_a_invalid_degree_sequence_b_invalid_degree_sequence_c_invalid_all_sequences_invalid_l60_60617


namespace geometric_sequence_ratio_l60_60339

variables {a b c q : ℝ}

theorem geometric_sequence_ratio (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sequence : ∃ q : ℝ, (a + b + c) * q = b + c - a ∧
                         (a + b + c) * q^2 = c + a - b ∧
                         (a + b + c) * q^3 = a + b - c) :
  q^3 + q^2 + q = 1 := 
sorry

end geometric_sequence_ratio_l60_60339


namespace parallel_lines_angle_sum_l60_60056

namespace Geometry

/-- Given conditions: Lines l and k are parallel, m∠A = 130°, m∠C = 70° -/
theorem parallel_lines_angle_sum
  (l k : Type) [Parallel l k] (mA mC : ℝ)
  (hA : mA = 130) (hC : mC = 70) :
  ∃ mB : ℝ, mB = 160 :=
by
  use 160
  sorry

end Geometry

end parallel_lines_angle_sum_l60_60056


namespace problem_statement_l60_60778

def S : Finset ℤ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def T : Finset ℤ := {-4, -3, -2, -1, 0, 1, 2, 3, 4}

def X (A : Finset ℤ) : ℚ := (A.sum id) / A.card

def is_average_subset (A : Finset ℤ) (S : Finset ℤ) : Prop :=
  X A = X S

def f (S : Finset ℤ) (k : ℕ) : ℕ :=
  (S.powerset.filter (λ A, A.card = k ∧ is_average_subset A S)).card

theorem problem_statement : ¬ (f S 2 + f S 3 = f T 4) :=
sorry

end problem_statement_l60_60778


namespace sum_of_squares_of_odds_l60_60256

theorem sum_of_squares_of_odds:
  (∑ n in finset.range 50, (2 * n + 1) ^ 2) = 1^2 + 3^2 + 5^2 + ... + 99^2 :=
sorry

end sum_of_squares_of_odds_l60_60256


namespace area_of_square_divided_into_congruent_figures_l60_60241

theorem area_of_square_divided_into_congruent_figures (area_congruent_figure : ℝ) (number_of_figures : ℕ) (h1 : area_congruent_figure = 1) (h2 : number_of_figures = 4) : 
  ∃ (area_square : ℝ), area_square = 4 := 
by
  use area_congruent_figure * number_of_figures
  have : area_congruent_figure * number_of_figures = 4
  { 
    rw [h1, h2],
    ring,
  }
  exact this

end area_of_square_divided_into_congruent_figures_l60_60241


namespace min_value_xy_expression_l60_60525

theorem min_value_xy_expression (x y : ℝ) : ∃ c : ℝ, (∀ x y : ℝ, (xy - 1)^2 + (x + y)^2 ≥ c) ∧ c = 1 :=
by {
  -- Placeholder for proof
  sorry
}

end min_value_xy_expression_l60_60525


namespace beam_reflection_equation_l60_60568

theorem beam_reflection_equation:
  ∃ (line : ℝ → ℝ → Prop), 
  (∀ (x y : ℝ), line x y ↔ (5 * x - 2 * y - 10 = 0)) ∧
  (line 4 5) ∧ 
  (line 2 0) :=
by
  sorry

end beam_reflection_equation_l60_60568


namespace maxwell_walking_speed_l60_60061

variable (distance : ℕ) (brad_speed : ℕ) (maxwell_time : ℕ) (brad_time : ℕ) (maxwell_speed : ℕ)

-- Given conditions
def conditions := distance = 54 ∧ brad_speed = 6 ∧ maxwell_time = 6 ∧ brad_time = 5

-- Problem statement
theorem maxwell_walking_speed (h : conditions distance brad_speed maxwell_time brad_time) : maxwell_speed = 4 := sorry

end maxwell_walking_speed_l60_60061


namespace number_of_small_circles_l60_60584

-- Definitions based on the problem conditions:
def large_circle_radius : ℝ := 2
def small_circle_radius : ℝ := 1
def distance_between_centers (R r : ℝ) : ℝ := R + r
def tangent_distance_between_small_circles (r : ℝ) : ℝ := 2 * r

-- The key geometric relationship in this specific problem:
theorem number_of_small_circles :
  ∀ (R r : ℝ), tangent_distance_between_small_circles r = 2 →
  R = large_circle_radius →
  r = small_circle_radius →
  let n := (2 : ℕ) * (R + r) / tangent_distance_between_small_circles r in
  n = 4 :=
by
  intros R r htangent hlarge hsmall;
  have h_eq : distance_between_centers R r = 3 :=
    by simp [distance_between_centers, hlarge, hsmall];
  sorry

end number_of_small_circles_l60_60584


namespace remainder_when_divided_by_8_l60_60197

theorem remainder_when_divided_by_8 (x : ℤ) (h : ∃ k : ℤ, x = 72 * k + 19) : x % 8 = 3 :=
by
  sorry

end remainder_when_divided_by_8_l60_60197


namespace evaluate_expression_l60_60645

theorem evaluate_expression :
  (827 * 827) - ((827 - 1) * (827 + 1)) = 1 :=
sorry

end evaluate_expression_l60_60645


namespace circumcircles_common_point_l60_60330

theorem circumcircles_common_point
  (A B C P Q R : Point)
  (hP : P ∈ segment B C)
  (hQ : Q ∈ segment A C)
  (hR : R ∈ segment A B) :
  ∃ T : Point, T ∈ circumcircle (triangle A R Q) ∧ 
                T ∈ circumcircle (triangle B P R) ∧ 
                T ∈ circumcircle (triangle C Q P) := 
by
  sorry

end circumcircles_common_point_l60_60330


namespace sequence_sum_proof_l60_60725

theorem sequence_sum_proof (n : ℕ) :
  let a := λ n : ℕ, if n = 1 then 1 else (λ i, have h : i ≤ n, from sorry, a (i-1) + a i = 1 / 2^(i-1)),
      S := λ k, ∑ i in range (k+1), a i
  in
  S (2 * n + 1) = (4 / 3) * (1 - (1 / 4)^(n+1)) :=
by sorry

end sequence_sum_proof_l60_60725


namespace reciprocal_twice_l60_60381

theorem reciprocal_twice {x : ℝ} (h : x = 48) : (1 / (1 / x)) = 48 :=
by {
  rw [h],
  norm_num
}

end reciprocal_twice_l60_60381


namespace average_number_div_by_3_between_40_80_l60_60953

def numbersDivBy3 (lo hi : ℕ) : List ℕ :=
  List.filter (λ x => x % 3 = 0) (List.range' lo (hi - lo + 1))

def average (l : List ℕ) : ℚ :=
  (l.foldl (· + ·) 0 : ℚ) / l.length

theorem average_number_div_by_3_between_40_80 :
  average (numbersDivBy3 41 79) = 63 := by
  sorry

end average_number_div_by_3_between_40_80_l60_60953


namespace tangent_line_eq_l60_60655

noncomputable def f (x : ℝ) := x / (2 * x - 1)

def tangentLineAtPoint (x : ℝ) : ℝ := -x + 2

theorem tangent_line_eq {x y : ℝ} (hxy : y = f 1) (f_deriv : deriv f 1 = -1) :
  y = 1 → tangentLineAtPoint x = -x + 2 :=
by
  intros
  sorry

end tangent_line_eq_l60_60655


namespace num_solution_sets_l60_60417

/-- 
  Prove that the number of sets of positive integer solutions 
  (x, y, z) to the system of equations 
  (x, y) = 60, (y, z) = 90, [z, x] = 360 
  where y ≤ 1000, is 3. 
-/
theorem num_solution_sets : 
  (∃! (s : ℕ × ℕ × ℕ), let x := s.1, y := s.2.1, z := s.2.2 
    in Nat.gcd x y = 60 ∧ Nat.gcd y z = 90 ∧ Nat.lcm z x = 360 ∧ y ≤ 1000) :=
sorry

end num_solution_sets_l60_60417


namespace no_int_solutions_except_zero_l60_60073

theorem no_int_solutions_except_zero 
  (a b c n : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
by
  sorry

end no_int_solutions_except_zero_l60_60073


namespace no_such_sequence_exists_l60_60421

theorem no_such_sequence_exists :
  ¬ ∃ (a : ℕ → ℤ), (∀ n, a n ≠ 0) ∧ 
    (∀ n ≥ 2020, ∃ r : ℝ, 
      (∃ m, m ≥ n ∧ Polynomial.eval r (∑ i in Finset.range (m + 1), a i * X ^ i) = 0) ∧ 
      |r| > 2.001) :=
by {
    sorry
}

end no_such_sequence_exists_l60_60421


namespace geometric_seq_value_l60_60690

theorem geometric_seq_value (a : ℕ → ℝ) (h : a 4 + a 8 = -2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 :=
sorry

end geometric_seq_value_l60_60690


namespace rectangle_area_l60_60555

theorem rectangle_area (w : ℝ) (h : ℝ) (area : ℝ) 
  (h1 : w = 5)
  (h2 : h = 2 * w) :
  area = h * w := by
  sorry

end rectangle_area_l60_60555


namespace convex_hull_has_at_least_n_vertices_l60_60453

-- Define a regular n-gon
structure RegularPolygon (n : ℕ) :=
(vertices : Fin n → ℝ × ℝ)

-- Define the convex hull function (simplified for purposes of this problem)
noncomputable def convexHull (vs : Finset (ℝ × ℝ)) : Finset (ℝ × ℝ) := sorry

-- State the theorem
theorem convex_hull_has_at_least_n_vertices {n : ℕ} (h : 3 ≤ n) (pgs : Finset (RegularPolygon n)) :
  n ≤ (convexHull (Finset.bUnion pgs (λ pg, Finset.univ.map ⟨pg.vertices, sorry⟩))).card := sorry

end convex_hull_has_at_least_n_vertices_l60_60453


namespace volume_of_regular_quadrilateral_pyramid_is_750_l60_60549

noncomputable def volume_of_pyramid (P V O A B C D : Point) : ℝ :=
  let h := dist P (Plane.abc V A B C D) in
  let d_to_faces := dist P (Face.abc V A B, Face.bcd V B C, Face.cda V C D, Face.dab V D A) in
  let d_to_base := dist P (Plane.abc A B C D) in
  if h = (1 / 2) * d_to_base ∧ d_to_faces = 3 ∧ d_to_base = 5 then
    750
  else
    0

theorem volume_of_regular_quadrilateral_pyramid_is_750 
  (P V O A B C D : Point) 
  (h : dist P (Plane.abc A B C D) = (1 / 2) * dist V (Plane.abc A B C D))
  (d_to_faces : dist P (Face.abc V A B, Face.bcd V B C, Face.cda V C D, Face.dab V D A) = 3)
  (d_to_base : dist P (Plane.abc A B C D) = 5)
  : volume_of_pyramid P V O A B C D = 750 :=
begin
  sorry
end

end volume_of_regular_quadrilateral_pyramid_is_750_l60_60549


namespace ellipse_focus_proof_l60_60103

def ellipse_focus (k : ℝ) : Prop :=
  let a := 1
  let c := abs (sqrt ((k * a^2 + a^2 - 5)/k))
  c = 2

theorem ellipse_focus_proof : ellipse_focus 1 :=
by
  unfold ellipse_focus
  sorry

end ellipse_focus_proof_l60_60103


namespace max_area_enclosed_l60_60078

theorem max_area_enclosed (p : ℕ) (hp : p = 156) (hside : ∀ x, x ∈ ([0, p / 2])) : 
  ∃ A, ∀ x y : ℕ, 2 * (x + y) = p → A ≤ x * y := 
begin
  sorry
end

end max_area_enclosed_l60_60078


namespace bus_speed_l60_60387

theorem bus_speed (distance time : ℝ) (h_distance : distance = 201) (h_time : time = 3) : 
  distance / time = 67 :=
by
  sorry

end bus_speed_l60_60387


namespace moles_of_NaHCO3_combined_l60_60651

theorem moles_of_NaHCO3_combined (n_HNO3 n_NaHCO3 : ℕ) (mass_H2O : ℝ) : 
  n_HNO3 = 2 ∧ mass_H2O = 36 ∧ n_HNO3 = n_NaHCO3 → n_NaHCO3 = 2 := by
  sorry

end moles_of_NaHCO3_combined_l60_60651


namespace original_number_is_80_l60_60193

theorem original_number_is_80 (x : ℝ) (h1 : 1.125 * x - 0.75 * x = 30) : x = 80 :=
by
  sorry

end original_number_is_80_l60_60193


namespace problem_a_lt_b_lt_c_l60_60322

noncomputable def a := Real.logBase 0.3 2
noncomputable def b := Real.sin (Real.pi / 18)
noncomputable def c := 0.5 ^ (-2: Int)

theorem problem_a_lt_b_lt_c (ha : a = Real.logBase 0.3 2) (hb : b = Real.sin (Real.pi / 18)) (hc : c = 0.5 ^ (-2: Int)) : a < b ∧ b < c := by
  rw [ha, hb, hc]
  sorry

end problem_a_lt_b_lt_c_l60_60322


namespace valid_parametrizations_l60_60494

noncomputable def valid_parametrization (p d : ℝ × ℝ) (t : ℝ) :=
  ∃ t : ℝ, p.2 = 2 * p.1 - 4 ∧ d = (1, 2)

theorem valid_parametrizations :
  valid_parametrization (3, -2) (1, 2) ∧
  valid_parametrization (4, 0) (2, 4) ∧
  valid_parametrization (0, -4) (1, 2) ∧
  valid_parametrization (1, -1) (0.5, 1) ∧
  valid_parametrization (-1, -6) (-2, -4) :=
by {
  sorry
}

end valid_parametrizations_l60_60494


namespace find_t_l60_60516

-- Define the vertices of the triangle PQR
structure Point where
  x : ℝ
  y : ℝ

def P : Point := {x := 1, y := 10}
def Q : Point := {x := 4, y := 0}
def R : Point := {x := 10, y := 0}

-- Define the line y = t
variable (t : ℝ)

-- Define the line equations for PQ and PR
def line_PQ (x : ℝ) : ℝ := (-10/3) * x + (70/3)
def line_PR (x : ℝ) : ℝ := (-10/9) * x + (100/9)

-- Define the intersection points V and W
def V : Point := {x := (70 - 3*t) / 10, y := t}
def W : Point := {x := (100 - 9*t) / 10, y := t}

-- Define the function for the area of triangle PVW
def area_PVW (t : ℝ) : ℝ :=
  let VW_x_dist := (100 - 9*t)/10 - (70 - 3*t)/10
  (1/2) * (VW_x_dist) * (10 - t)

-- Statement to prove: Area of triangle PVW is 18 if and only if t = 10
theorem find_t : area_PVW t = 18 → t = 10 := by
  sorry

end find_t_l60_60516


namespace correct_statement_is_D_l60_60543

-- Definitions according to the problem context
def height : Type := ℝ  -- Treat height as a real number, a scalar
def temperature : Type := ℝ  -- Treat temperature as a real number, a scalar

-- Definition of directed line segment with start point and end point
structure DirectedLineSegment :=
(start : ℝ × ℝ)  -- Coordinates for the start point
(end : ℝ × ℝ)  -- Coordinates for the end point

-- Length of a directed line segment
def length (dls : DirectedLineSegment) : ℝ :=
  real.sqrt ((dls.end.1 - dls.start.1) ^ 2 + (dls.end.2 - dls.start.2) ^ 2)

-- Theorem to state the lengths of directed line segments are equal and thus, statement D is correct
theorem correct_statement_is_D : 
  (height → ℝ) ∧ 
  (temperature → ℝ) ∧ 
  (∀ (dls : DirectedLineSegment), dls.length = dls.length) → 
  (length ⟨(0,0), (1,1)⟩ = length ⟨(1,1), (0,0)⟩) :=
by
  intros height temperature dls_equal_length
  sorry

end correct_statement_is_D_l60_60543


namespace polynomial_remainder_l60_60613

theorem polynomial_remainder (y : ℂ) (h1 : y^5 + y^4 + y^3 + y^2 + y + 1 = 0) (h2 : y^6 = 1) :
  (y^55 + y^40 + y^25 + y^10 + 1) % (y^5 + y^4 + y^3 + y^2 + y + 1) = 2 * y + 3 :=
sorry

end polynomial_remainder_l60_60613


namespace geometric_series_sum_l60_60157

theorem geometric_series_sum :
  (∑ k in (Finset.range 7), (1:ℚ) / (5 * (-3) ^ k)) = (1641 / 10935 : ℚ) :=
by
  sorry

end geometric_series_sum_l60_60157


namespace KM_not_parallel_IH_l60_60796

-- Definitions of the geometric entities involved and their properties
axiom scalene_triangle (ABC : Triangle) : ABC.is_scalene
axiom incenter (ABC : Triangle) : Point
axiom is_incenter (I : Point) (ABC : Triangle) : Prop
axiom orthocenter (ABC : Triangle) : Point
axiom is_orthocenter (H : Point) (ABC : Triangle) : Prop
axiom incircle_touch_points (ABC : Triangle) : (Point × Point × Point)
axiom is_incircle_touch_points (D E F I : Point) (ABC : Triangle) : Prop
axiom intersection_DF_AC (DF AC : Line) : Point
axiom intersection_EF_BC (EF BC : Line) : Point
axiom are_parallel (KM IH : Line) : Prop
axiom line_of_points (X Y : Point) : Line

-- Translate the problem to Lean statement
theorem KM_not_parallel_IH
  (ABC : Triangle)
  (h_scalene : scalene_triangle ABC)
  (I : Point) (h_incenter : is_incenter I ABC)
  (H : Point) (h_orthocenter : is_orthocenter H ABC)
  (D E F : Point) (h_touch : is_incircle_touch_points D E F I ABC)
  (K : Point) (h_K : K = intersection_DF_AC (line_of_points D F) (line_of_points ABC.C ABC.A))
  (M : Point) (h_M : M = intersection_EF_BC (line_of_points E F) (line_of_points ABC.B ABC.C))
  : ¬ are_parallel (line_of_points K M) (line_of_points I H) :=
sorry

end KM_not_parallel_IH_l60_60796


namespace hyperbola_eccentricity_sum_l60_60362

noncomputable def hyperbola (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : Prop :=
  ∃ e : ℝ, e = 5 / 4 ∧ (a ^ 2 + b ^ 2 = (e * a) ^ 2) ∧ 
  (∀ P : ℝ × ℝ, (P.1, P.2) ∈ { (x, y) | y^2 / a^2 - x^2 / b^2 = 1 } → 
   let c := e * a in
   (c, 0) ∈ { (x, y) | y^2 / a^2 - x^2 / b^2 = 1 } ∧ 
   (-c, 0) ∈ { (x, y) | y^2 / a^2 - x^2 / b^2 = 1 } ∧
   let PF1 := (P.1 - c, P.2) in
   let PF2 := (P.1 + c, P.2) in
   (PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0) ∧
   1/2 * 2 * c * (9 / c) = 9)

theorem hyperbola_eccentricity_sum (a b : ℝ) (h : hyperbola a b (by simp) (by simp)) : a + b = 7 :=
sorry

end hyperbola_eccentricity_sum_l60_60362


namespace find_z_l60_60872

noncomputable def vec1 : ℝ × ℝ × ℝ := (1, 4, z)
noncomputable def vec2 : ℝ × ℝ × ℝ := (-2, 6, -2)
noncomputable def projection_factor := 16 / 28

theorem find_z (z : ℝ) 
  (h : (vec1.fst * vec2.fst + vec1.snd * vec2.snd + vec1.3rd * vec2.3) /
       (vec2.fst * vec2.fst + vec2.snd * vec2.snd + vec2.3rd * vec2.3rd) = projection_factor) : 
  z = 7 :=
sorry

end find_z_l60_60872


namespace sequence_100th_term_l60_60600

def isInSequence (n : ℕ) : Prop :=
  ∃ k : ℕ, n < 3^(k+1) ∧ 
    (bit0 n + 1 == bitShift 1 k
    ∨ bit1 n == bitShift 0 k + 1
    ∨ ∃ l : ℕ, l < k ∧ bit0 (bit1 n) == bitShift l (k-l))

theorem sequence_100th_term : 
  (∀ n, isInSequence n → isInSequence (n+1)) → 
  isInSequence 981 := 
by 
  sorry

end sequence_100th_term_l60_60600


namespace helen_chocolate_chip_cookies_l60_60369

def number_of_raisin_cookies := 231
def difference := 25

theorem helen_chocolate_chip_cookies :
  ∃ C, C = number_of_raisin_cookies + difference ∧ C = 256 :=
by
  sorry -- Skipping the proof

end helen_chocolate_chip_cookies_l60_60369


namespace max_binomial_term_l60_60280

theorem max_binomial_term :
  ∃ k : ℕ, (k = 163) ∧
    (∀ n : ℕ, 0 ≤ n ∧ n ≤ 212 → 
      C 212 n * (11^(n/2)) ≤ C 212 k * (11^(k/2))) :=
by
  sorry

end max_binomial_term_l60_60280


namespace totalWatermelons_l60_60845

def initialWatermelons : ℕ := 4
def additionalWatermelons : ℕ := 3

theorem totalWatermelons : initialWatermelons + additionalWatermelons = 7 := by
  sorry

end totalWatermelons_l60_60845


namespace lyndee_ate_one_piece_l60_60818

def pieces_made_by_mrs_crocker : ℕ := 11
def pieces_per_friend (friends : ℕ) : ℕ := 2 * friends
def total_friends : ℕ := 5
def fried_chicken_by_friends (total_friends : ℕ) : ℕ := pieces_per_friend(total_friends)
def fried_chicken_by_lyndee (total_pieces : ℕ) (pieces_by_friends : ℕ) : ℕ := total_pieces - pieces_by_friends

theorem lyndee_ate_one_piece :
  fried_chicken_by_lyndee pieces_made_by_mrs_crocker (fried_chicken_by_friends total_friends) = 1 :=
by 
  sorry

end lyndee_ate_one_piece_l60_60818


namespace probability_of_selecting_cooking_l60_60215

-- Define a type representing the courses.
inductive Course
| planting : Course
| cooking : Course
| pottery : Course
| carpentry : Course

-- Define the set of all courses
def all_courses : Finset Course := {Course.planting, Course.cooking, Course.pottery, Course.carpentry}

-- The condition that Xiao Ming randomly selects one of the four courses
def uniform_probability (s : Finset Course) (a : Course) : ℚ := 1 / s.card

-- Prove that the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking : uniform_probability all_courses Course.cooking = 1 / 4 :=
sorry

end probability_of_selecting_cooking_l60_60215


namespace log_div_sqrt_defined_l60_60309

theorem log_div_sqrt_defined (x : ℝ) : -2 < x ∧ x < 5 ↔ ∃ y : ℝ, y = x ∧ ∃ z : ℝ, z = 5-x ∧ log(z) / sqrt(x+2) ∈ ℝ :=
by
  sorry

end log_div_sqrt_defined_l60_60309


namespace T_positive_l60_60436

theorem T_positive (α : ℝ) 
  (hα : ∀ k : ℤ, α ≠ k * π / 2) : 
  (sin α + tan α) / (cos α + cot α) > 0 := 
sorry

end T_positive_l60_60436


namespace find_b_l60_60361

-- Define a hyperbola and its asymptotes
def asymptote_equation (b : ℝ) (x y : ℝ) : Prop :=
  (sqrt 3 * x + y = 0) ∨ (sqrt 3 * x - y = 0)

theorem find_b (b : ℝ) (h : b > 0):
  (asymptote_equation b x y) →
  (sqrt 3 = b/2) →
  b = 2 * sqrt 3 :=
by
  sorry

end find_b_l60_60361


namespace sum_of_integers_l60_60507

-- Define the conditions as given in the problem:
def sums_are_consecutive (l : List ℤ) : Prop :=
  ∃ t : ℤ, ∀ i j : ℕ, (i < j ∧ j < l.length) → l[i] + l[j] = t + (i+j)

-- Translate the math problem into a Lean theorem statement:
theorem sum_of_integers {n : ℕ} (hn : n ≥ 3) :
  (∃ l : List ℤ, l.length = n ∧ sums_are_consecutive l) ↔ (n = 3 ∨ n = 4) :=
sorry

end sum_of_integers_l60_60507


namespace sum_value_correct_l60_60263

noncomputable def compute_sum : ℕ :=
  (List.range 25).sum (λ n => (n + 1)^(25 - n))

theorem sum_value_correct : compute_sum = 66071772829247409 := 
by sorry

end sum_value_correct_l60_60263


namespace fixed_intersection_of_circumcircle_of_PIJ_l60_60801

theorem fixed_intersection_of_circumcircle_of_PIJ 
  (A B C : Point) 
  (circumcircle_ABC : Circle)
  (P : Point)
  (hP_on_arc_BC : ¬(A ∈ arc_BC P circumcircle_ABC)) 
  (I : Point) 
  (J : Point) 
  (incircle_PAB : Circle)
  (incircle_PAC : Circle)
  (hI_center : center incircle_PAB = I)
  (hJ_center : center incircle_PAC = J)
  (hP_on_circumcircle_ABC : P ∈ circumcircle_ABC) (hP_not_A : P ≠ A) :
  ∃ Q : Point, Q ∈ circumcircle (triangle P I J) ∧ 
    Q ∈ circumcircle_ABC ∧ 
    Q ∈ circumcircle_ABC.fixed  :=
  sorry

end fixed_intersection_of_circumcircle_of_PIJ_l60_60801


namespace reciprocal_of_neg_two_l60_60900

noncomputable theory

def is_reciprocal (a x : ℝ) := a * x = 1

theorem reciprocal_of_neg_two : 
  ∃ x : ℝ, is_reciprocal (-2) x ∧ x = -1 / 2 := 
sorry

end reciprocal_of_neg_two_l60_60900


namespace value_of_k_l60_60919

theorem value_of_k (x y k : ℝ) (h1 : x - y = k + 2) (h2 : x + 3y = k) (h3 : x + y = 2) : k = 1 :=
by
  sorry

end value_of_k_l60_60919


namespace total_matches_in_chess_tournament_l60_60509

open Nat

theorem total_matches_in_chess_tournament:
  ∃ (n : ℕ), n = 150 ∧ (3 * (n.choose 2)) = 33750 :=
by
  use 150
  simp
  sorry

end total_matches_in_chess_tournament_l60_60509


namespace two_AF_eq_AB_sub_AC_l60_60750

theorem two_AF_eq_AB_sub_AC 
  (A B C E F : Type*)
  [euc_geometry : EuclideanGeometry A B C E F]
  (h1 : AB > AC)
  (h2 : external_angle_bisector E A B C)
  (h3 : perpendicular_foot F E AB):
  2 * AF = AB - AC := 
sorry

end two_AF_eq_AB_sub_AC_l60_60750


namespace parabola_translation_correct_l60_60515

-- Define the original equation of the parabola
def original_parabola (x : ℝ) : ℝ := 8 * x^2

-- Define the transformation of translating 3 units to the left and 5 units down
def translate_parabola (f : ℝ → ℝ) (h k : ℝ) :=
  λ x, f (x + h) - k

-- Define the expected result after translation
def expected_result (x : ℝ) : ℝ := 8 * (x + 3)^2 - 5

-- The theorem statement proving the expected transformation
theorem parabola_translation_correct :
  ∀ x : ℝ, translate_parabola original_parabola 3 5 x = expected_result x :=
by
  intro x
  sorry

end parabola_translation_correct_l60_60515


namespace fundraising_goal_shortfall_l60_60079

open Real

theorem fundraising_goal_shortfall : 
  let ken := 800
  let mary := 5 * ken
  let scott := (mary / 3)
  let amy := 2 * ken
  let total := ken + mary + scott + amy
  total = 9600 →
  10000 - total = 400 :=
by
  intro h
  rw h
  norm_num
  exact rfl

end fundraising_goal_shortfall_l60_60079


namespace sum_a_b_eq_5_l60_60321

theorem sum_a_b_eq_5 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * b = a - 2) (h4 : (-2)^2 = b * (2 * b + 2)) : a + b = 5 :=
sorry

end sum_a_b_eq_5_l60_60321


namespace quadratic_eq_solution_1_quadratic_eq_solution_2_l60_60469

theorem quadratic_eq_solution_1 (x : ℝ) : x^2 - 4 * x + 2 = 0 → x = 2 + real.sqrt 2 ∨ x = 2 - real.sqrt 2 :=
  by { sorry }

theorem quadratic_eq_solution_2 (x : ℝ) : (x - 3)^2 = 2 * x - 6 → x = 3 ∨ x = 5 :=
  by { sorry }

end quadratic_eq_solution_1_quadratic_eq_solution_2_l60_60469


namespace player_A_winning_probability_l60_60403

noncomputable def probability_A_wins_match (p_A_wins_set : ℚ) : ℚ :=
  let p1 := p_A_wins_set ^ 2
  let p2 := 2 * p_A_wins_set * (1 - p_A_wins_set) * p_A_wins_set
  p1 + p2

theorem player_A_winning_probability :
  probability_A_wins_match 0.6 = 0.648 := 
by 
  sorry

end player_A_winning_probability_l60_60403


namespace determinant_rotation_75_degrees_l60_60437

noncomputable def rotation_matrix_75_degrees : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos (Float.pi * 75 / 180), - Real.sin (Float.pi * 75 / 180)], 
    ![Real.sin (Float.pi * 75 / 180), Real.cos (Float.pi * 75 / 180)]]

theorem determinant_rotation_75_degrees :
  Matrix.det rotation_matrix_75_degrees = 1 := by sorry

end determinant_rotation_75_degrees_l60_60437


namespace first_term_correct_l60_60506

noncomputable def first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^3 / (1 - (r^3)) = 80) : ℝ :=
a

theorem first_term_correct (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^3 / (1 - (r^3)) = 80) :
  first_term a r h1 h2 = 3.42 :=
sorry

end first_term_correct_l60_60506


namespace identity_transformation_if_and_only_if_AC_eq_BD_and_perpendicular_l60_60990

variables {A B C D : Point} (M : Point)
noncomputable def rotation_90 (O : Point) : Transformation := sorry

-- Conditions
axiom rotation_around_A : transformation := rotation_90 A
axiom rotation_around_B : transformation := rotation_90 B
axiom rotation_around_C : transformation := rotation_90 C
axiom rotation_around_D : transformation := rotation_90 D

-- Theorem to prove
theorem identity_transformation_if_and_only_if_AC_eq_BD_and_perpendicular 
  (h_composite : rotation_around_A ∘ rotation_around_B ∘ rotation_around_C ∘ rotation_around_D = id_transformation) :
  segment_length AC = segment_length BD ∧ is_perpendicular AC BD :=
by sorry

end identity_transformation_if_and_only_if_AC_eq_BD_and_perpendicular_l60_60990


namespace range_of_a_l60_60746

theorem range_of_a (a : ℝ) : (∀ x : ℝ, abs (x + 2) - abs (x - 1) ≥ a^3 - 4 * a^2 - 3) → a ≤ 4 :=
sorry

end range_of_a_l60_60746


namespace convexity_of_function_l60_60070

variables {n : ℕ} {r : ℝ} (x : Fin n → ℝ)

noncomputable def convex_function := (∑ i : Fin n, (x i) ^ r / n) ^ (1 / r)

theorem convexity_of_function (hr : r > 1) : convex_on (set.univ : set (Fin n → ℝ)) convex_function := 
sorry

end convexity_of_function_l60_60070


namespace judy_shopping_total_l60_60279

noncomputable def carrot_price := 1
noncomputable def milk_price := 3
noncomputable def pineapple_price := 4 / 2 -- half price
noncomputable def flour_price := 5
noncomputable def ice_cream_price := 7

noncomputable def carrot_quantity := 5
noncomputable def milk_quantity := 3
noncomputable def pineapple_quantity := 2
noncomputable def flour_quantity := 2
noncomputable def ice_cream_quantity := 1

noncomputable def initial_cost : ℝ := 
  carrot_quantity * carrot_price 
  + milk_quantity * milk_price 
  + pineapple_quantity * pineapple_price 
  + flour_quantity * flour_price 
  + ice_cream_quantity * ice_cream_price

noncomputable def final_cost (initial_cost: ℝ) := if initial_cost ≥ 25 then initial_cost - 5 else initial_cost

theorem judy_shopping_total : final_cost initial_cost = 30 := by
  sorry

end judy_shopping_total_l60_60279


namespace bobbit_worm_fish_count_l60_60926

theorem bobbit_worm_fish_count 
  (initial_fish : ℕ)
  (fish_eaten_per_day : ℕ)
  (days_before_adding_fish : ℕ)
  (additional_fish : ℕ)
  (days_after_adding_fish : ℕ) :
  days_before_adding_fish = 14 →
  days_after_adding_fish = 7 →
  fish_eaten_per_day = 2 →
  initial_fish = 60 →
  additional_fish = 8 →
  (initial_fish - days_before_adding_fish * fish_eaten_per_day + additional_fish - days_after_adding_fish * fish_eaten_per_day) = 26 :=
by
  intros 
  -- sorry proof goes here
  sorry

end bobbit_worm_fish_count_l60_60926


namespace exists_centrally_symmetric_inscribed_hexagon_l60_60301

noncomputable def convex_polygon_unit_area (W : Type) := 
  -- Definition of convex polygon with unit area
  -- This definition assumes the existence of appropriate definitions in Mathlib
  sorry 

def is_centrally_symmetric (V : Type) : Prop :=
  -- Definition for centrally symmetric property of hexagon
  sorry

def is_inscribed (V W : Type) : Prop :=
  -- Definition for an inscribed hexagon in a polygon
  sorry

def area (V : Type) : ℝ :=
  -- Definition for area of hexagon
  sorry

theorem exists_centrally_symmetric_inscribed_hexagon (W : Type)
  [convex_polygon_unit_area W] :
  ∃ (V : Type), 
    (is_centrally_symmetric V) ∧ (is_inscribed V W) ∧ (area V ≥ 2 / 3) :=
sorry

end exists_centrally_symmetric_inscribed_hexagon_l60_60301


namespace range_of_a_l60_60735

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (|x + a| < 3 ↔ 2 < x ∧ x < 3)) ↔ -5 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l60_60735


namespace find_k_l60_60780

theorem find_k (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] 
               (dist_AB: ℝ) (dist_BC: ℝ) (dist_AC: ℝ)
               (h_AB : dist_AB = 6) (h_BC : dist_BC = 8) (h_AC : dist_AC = 10)
               (BD : ℝ) (h_BD_angle_bisector : ∃ B D, B ≠ D) (h_BD_length : BD = k * Real.sqrt 6)
               (k : ℝ) :
               k = 26 / 21 :=
by
  sorry

end find_k_l60_60780


namespace percentage_of_married_employees_correct_l60_60258

theorem percentage_of_married_employees_correct :
  ∀ (E : ℝ)
    (total_employees_pos : E > 0)
    (women_percentage : ℝ) (single_men_fraction : ℝ) (married_women_percentage : ℝ),
  women_percentage = 0.76 →
  single_men_fraction = 2/3 →
  married_women_percentage = 0.6842 →
  let men_percentage := 1 - women_percentage in
  let married_men_percentage := (1 - single_men_fraction) * men_percentage in
  let married_women := married_women_percentage * women_percentage in
  let married_employees_percentage := married_women + married_men_percentage in
  married_employees_percentage * 100 ≈ 60.04 :=
by
  sorry

end percentage_of_married_employees_correct_l60_60258


namespace perpendicular_line_through_point_l60_60653

theorem perpendicular_line_through_point 
 {x y : ℝ}
 (p : (ℝ × ℝ)) 
 (point : p = (-2, 1)) 
 (perpendicular : ∀ x y, 2 * x - y + 4 = 0) : 
 (∀ x y, x + 2 * y = 0) ∧ (p.fst = -2 ∧ p.snd = 1) :=
by
  sorry

end perpendicular_line_through_point_l60_60653


namespace sum_powers_mod_5_l60_60839

theorem sum_powers_mod_5 (n : ℕ) (h : ¬ (n % 4 = 0)) : 
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 :=
by
  sorry

end sum_powers_mod_5_l60_60839


namespace sum_of_numbers_l60_60194

-- Let a and b be the numbers described in the problem
variables (a b : ℕ)

-- Conditions based on the problem statement
def condition_lcm := nat.lcm a b = 42
def condition_ratio := a * 3 = b * 2

-- Lean 4 statement to prove that a + b = 70 given the conditions
theorem sum_of_numbers (a b : ℕ) (h1 : condition_lcm a b) (h2 : condition_ratio a b) : a + b = 70 := 
by { sorry }

end sum_of_numbers_l60_60194


namespace correct_calculation_l60_60949

theorem correct_calculation (x : ℤ) (h : 7 * (x + 24) / 5 = 70) :
  (5 * x + 24) / 7 = 22 :=
sorry

end correct_calculation_l60_60949


namespace scaling_transformation_l60_60022

-- Definitions of the original and transformed lines
def line1 (x y : ℝ) : Prop := x + y + 2 = 0
def line2 (x y : ℝ) : Prop := 8 * x + y + 8 = 0

-- Scaling transformation formulas
def transform_x (x : ℝ) : ℝ := (1/2) * x
def transform_y (y : ℝ) : ℝ := 4 * y

-- Statement to prove
theorem scaling_transformation :
  (∀ x' y', line1 ((1/2) * x') (4 * y') ↔ line2 x' y') :=
by
  sorry

end scaling_transformation_l60_60022


namespace complete_the_square_k_l60_60007

theorem complete_the_square_k (x : ℝ) : ∃ k : ℝ, (∃ a h: ℝ, (a = 1) ∧ (h = 7 / 2) ∧ (x^2 - 7*x = a * (x - h)^2 + k)) → k = -49 / 4 :=
by
  sorry

end complete_the_square_k_l60_60007


namespace reaction_yields_one_mole_of_CaCl2_l60_60299

theorem reaction_yields_one_mole_of_CaCl2
  (moles_HCl : ℕ) (moles_CaCO3 : ℕ)
  (balanced_reaction : CaCO3 + 2 * HCl = CaCl2 + CO2 + H2O) 
  (hHCl : moles_HCl = 2) (hCaCO3 : moles_CaCO3 = 1) :
  (moles_CaCl2 : ℕ) :=
begin
  have stoichiometric_ratio : moles_CaCO3 = 1 → moles_HCl = 2 → moles_CaCl2 = 1,
  { intro h1, intro h2,
    rw [h1, h2],
    exact rfl,
  },
  exact stoichiometric_ratio hCaCO3 hHCl,
end

end reaction_yields_one_mole_of_CaCl2_l60_60299


namespace equal_segments_am_bm_cm_l60_60564

open EuclideanGeometry 

variables {A B C D M : Point}
variables {triangle_ABC : Triangle}
variables {triangle_ADC : Triangle}

-- Conditions
axiom right_angle_B : right_angle (B)
axiom congruent_triangles : congruent triangle_ABC triangle_ADC
axiom rectangle_ABCD : rectangle (A) (B) (C) (D)
axiom diagonals_intersect_M : intersect (diagonals (A) (C) (B) (D) at (M))

-- Proof Statement
theorem equal_segments_am_bm_cm : AM = BM ∧ BM = CM := 
sorry

end equal_segments_am_bm_cm_l60_60564


namespace value_of_x_l60_60350

theorem value_of_x (x : ℝ) (m : ℕ) (h1 : m = 31) :
  ((x ^ m) / (5 ^ m)) * ((x ^ 16) / (4 ^ 16)) = 1 / (2 * 10 ^ 31) → x = 1 := by
  sorry

end value_of_x_l60_60350


namespace reciprocal_of_neg_two_l60_60877

-- Define the condition of reciprocal
def is_reciprocal (y x : ℝ) : Prop := y * x = 1

-- Define the number in question
def y : ℝ := -2

-- Define the expected reciprocal
def x : ℝ := -1 / 2

-- State the theorem
theorem reciprocal_of_neg_two : is_reciprocal y x :=
by
  -- Proof can be provided here
  sorry

end reciprocal_of_neg_two_l60_60877


namespace min_value_of_f_l60_60662

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) + (1 / (x^2 + (1 / x^2)))

theorem min_value_of_f : ∃ (c > 0), (∀ x > 0, f x ≥ f c) ∧ f c = 2.5 := by
  sorry

end min_value_of_f_l60_60662


namespace triangles_similar_l60_60559

-- Definitions for the problem
structure Parallelogram (A B C D : Type) :=
(angle_ABC : ℝ)
(angle_ADC : ℝ)
(angle_BAD : ℝ)
(angle_BCD : ℝ)
-- Conditions of being parallelogram (opposite angles are equal, adjacent angles supplementary)
(h1 : angle_ABC = angle_ADC)
(h2 : angle_BAD = 180 - angle_ABC)
(h3 : angle_BCD = 180 - angle_BAD)

-- Perpendiculars
def Perpendicular (P Q R : Type) : Prop := -- Perpendicular definition using implicit angle 
  sorry

-- Lean statement for the problem
theorem triangles_similar (A B C D M N : Type) [Parallelogram A B C D]
  (AM_perp_BC : Perpendicular A M B C)
  (AN_perp_CD : Perpendicular A N C D) :
  Similar (triangle M A N) (triangle A B C) :=
sorry

end triangles_similar_l60_60559


namespace proof_P_B_given_A_l60_60852

noncomputable def number_of_activities : ℕ := 5

def event_A (a b : ℕ) (total_activities : ℕ) : Prop :=
  a ≠ b ∧ a < total_activities ∧ b < total_activities

def event_B (a b : ℕ) : Prop :=
  a = 1 ∧ b = 1

def P_B_given_A (n_A : ℕ) (n_AB : ℕ) : ℚ := n_AB / n_A

theorem proof_P_B_given_A :
  let total_activities := number_of_activities in
  let n_A := (total_activities * (total_activities - 1)) in
  let n_AB := 0 in
  P_B_given_A n_A n_AB = 2 / 5 :=
by
  sorry

end proof_P_B_given_A_l60_60852


namespace find_length_AB_l60_60420

-- Definitions of the problem conditions
def triangle (A B C : Type) := A ≠ B ∧ B ≠ C ∧ C ≠ A
def median (X Y Z : Type) (M : Type) := 
  ∃ (P : Type), (M = P ∧ X = P ∧ Y = P ∧ Z = P) -- Placeholder definition to denote medians

-- The problem statement
theorem find_length_AB (ABC : Type) (A B C M N : Type) 
  (h_triangle : triangle A B C)
  (h_median_AM : median A B C M)
  (h_median_BN : median B C A N)
  (h_perpendicular : ∃ G : Type, M ≠ G ∧ N ≠ G) -- Placeholder for perpendicular medians
  (h_AM_length : M = 21) (h_BN_length : N = 28) :
  A = 70/3 := 
sorry

end find_length_AB_l60_60420


namespace tangent_line_at_P_is_y_equals_1_l60_60104

theorem tangent_line_at_P_is_y_equals_1 :
  ∀ (x : ℝ), (y = x^3 - 3 * x^2 + 1) at P(0, 1), ∃ (y : ℝ), (y = 1)
  sorry

end tangent_line_at_P_is_y_equals_1_l60_60104


namespace reciprocal_of_neg2_l60_60910

-- Define the number
def num : ℤ := -2

-- Define the reciprocal function
def reciprocal (x : ℤ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_neg2 : reciprocal num = -1 / 2 :=
by
  -- Proof is skipped
  sorry

end reciprocal_of_neg2_l60_60910


namespace find_side_length_of_square_l60_60068

theorem find_side_length_of_square (n k : ℕ) (hk : k ≥ 1) (h : (n + k) * (n + k) - n * n = 47) : n = 23 :=
  sorry

end find_side_length_of_square_l60_60068


namespace number_of_common_points_max_value_and_coordinates_l60_60772

noncomputable section

def parametric_line (t : ℝ) : ℝ × ℝ :=
  (√2 + t, t)

def polar_circle_C (ρ θ : ℝ) : ℝ × ℝ :=
  (cos θ, sin θ)

def scaling_transformation (x y : ℝ) : ℝ × ℝ :=
  (x, 2*y)

theorem number_of_common_points : 
  ∃! (t : ℝ), ((√2 + t)^2 + t^2 = 1) :=
sorry

theorem max_value_and_coordinates :
  ∃ θ ∈ set.Ico 0 (2 * Real.pi), 
    let x := cos θ
    y := 2 * sin θ
    4 * x^2 + x * y + y^2 = 5 
    ∧ (x, y) = (Real.sqrt 2 / 2, Real.sqrt 2) 
    ∨ (x, y) = (-Real.sqrt 2 / 2, -Real.sqrt 2) :=
sorry

end number_of_common_points_max_value_and_coordinates_l60_60772


namespace phone_purchase_initial_max_profit_additional_purchase_l60_60856

-- Definitions for phone purchase prices and selling prices
def purchase_price_A : ℕ := 3000
def selling_price_A : ℕ := 3400
def purchase_price_B : ℕ := 3500
def selling_price_B : ℕ := 4000

-- Definitions for total expenditure and profit
def total_spent : ℕ := 32000
def total_profit : ℕ := 4400

-- Definitions for initial number of units purchased
def initial_units_A : ℕ := 6
def initial_units_B : ℕ := 4

-- Definitions for the additional purchase constraints and profit calculation
def max_additional_units : ℕ := 30
def additional_units_A : ℕ := 10
def additional_units_B : ℕ := max_additional_units - additional_units_A 
def max_profit : ℕ := 14000

theorem phone_purchase_initial:
  3000 * initial_units_A + 3500 * initial_units_B = total_spent ∧
  (selling_price_A - purchase_price_A) * initial_units_A + (selling_price_B - purchase_price_B) * initial_units_B = total_profit := by
  sorry 

theorem max_profit_additional_purchase:
  additional_units_A + additional_units_B = max_additional_units ∧
  additional_units_B ≤ 2 * additional_units_A ∧
  (selling_price_A - purchase_price_A) * additional_units_A + (selling_price_B - purchase_price_B) * additional_units_B = max_profit := by
  sorry

end phone_purchase_initial_max_profit_additional_purchase_l60_60856


namespace find_x_l60_60807

variable {a b x r : ℝ}
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)
variable (h₂ : r = (4 * a)^(2 * b))
variable (h₃ : r = (a^b * x^b)^2)
variable (h₄ : 0 < x)

theorem find_x : x = 4 := by
  sorry

end find_x_l60_60807


namespace monotonic_increasing_implies_range_l60_60716

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / 2 * x ^ 2 + 2 * x - 2 * log x

theorem monotonic_increasing_implies_range (a : ℝ) :
  (∀ x > (0 : ℝ), deriv f x ≥ 0) → a ≤ 1 :=
  by 
  sorry

end monotonic_increasing_implies_range_l60_60716


namespace substitution_ways_mod_1000_l60_60998

def num_ways (k : ℕ) : ℕ :=
if k = 0 then 1 
else 11 ^ (2 * k) * (10 * 9 * 8) ^ (k - 1)

def total_ways : ℕ := (0 : ℕ → ℕ).sum (λ i, num_ways i)

theorem substitution_ways_mod_1000 (n : ℕ) (k : ℕ) (n = total_ways k) : n % 1000 = 722 :=
by {
  sorry
}

end substitution_ways_mod_1000_l60_60998


namespace candles_lit_at_correct_time_l60_60929

theorem candles_lit_at_correct_time :
  let ℓ := 1 -- Assume the initial length of each candle is 1 unit
  let f := λ t : ℝ, ℓ - (ℓ / 300) * t -- First candle's stub length after t minutes
  let g := λ t : ℝ, ℓ - (ℓ / 420) * t -- Second candle's stub length after t minutes
  let t := 262.5 -- 262.5 minutes from the lighting time to 6 PM
  (g(t) = 3 * f(t)) → (t = 262.5) :=
by 
  sorry

end candles_lit_at_correct_time_l60_60929


namespace parametric_to_ordinary_equiv_l60_60596

theorem parametric_to_ordinary_equiv (t φ θ : ℝ) :
  (∃ t : ℝ, (sin t)^2 + cos t^2 - 1 = 0 ∧ (cos t)^2 ∈ [0,1] ∧ sin t ∈ [-1,1])
  ∨ (∃ φ : ℝ, (tan φ)^2 + (1 - (tan φ)^2) - 1 = 0)
  ∨ (∃ t : ℝ, (sqrt (1 - t))^2 + t - 1 = 0)
  ∨ (∃ θ : ℝ, (cos θ)^2 + (sin θ)^2 - 1 = 0 ∧ (sin θ)^2 ∈ [0,1] ∧ cos θ ∈ [-1,1]) :=
by {
  sorry
}

end parametric_to_ordinary_equiv_l60_60596


namespace probability_correct_l60_60000

noncomputable def probability_product_multiple_of_72 : ℚ := by
  let s : Finset ℕ := {4, 6, 12, 18, 24, 36, 48}
  let subsets := s.powerset.filter (λ p, 2 ≤ p.card)
  let total_pairs := subsets.card.to_rat / 2
  let valid_pairs := subsets.filter (λ p, let ⟨a, b⟩ := p.to_finset.pair 2 in (a * b) % 72 = 0)
  let valid_pairs_count := valid_pairs.card.to_rat / 2
  exact valid_pairs_count / total_pairs

theorem probability_correct : probability_product_multiple_of_72 = 2 / 21 := sorry

end probability_correct_l60_60000


namespace distance_between_lines_l60_60108

/-- The graph of the function y = x^2 + ax + b is drawn on a board.
Let the parabola intersect the horizontal lines y = s and y = t at points A, B and C, D respectively,
with A B = 5 and C D = 11. Then the distance between the lines y = s and y = t is 24. -/
theorem distance_between_lines 
  (a b s t : ℝ)
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + a * x1 + b = s) ∧ (x2^2 + a * x2 + b = s) ∧ |x1 - x2| = 5)
  (h2 : ∃ x3 x4 : ℝ, x3 ≠ x4 ∧ (x3^2 + a * x3 + b = t) ∧ (x4^2 + a * x4 + b = t) ∧ |x3 - x4| = 11) :
  |t - s| = 24 := 
by
  sorry

end distance_between_lines_l60_60108


namespace product_of_possible_values_of_N_l60_60259

theorem product_of_possible_values_of_N (N B D : ℤ) 
  (h1 : B = D - N) 
  (h2 : B + 10 - (D - 4) = 1 ∨ B + 10 - (D - 4) = -1) :
  N = 13 ∨ N = 15 → (13 * 15) = 195 :=
by sorry

end product_of_possible_values_of_N_l60_60259


namespace evaluate_expression_l60_60277

theorem evaluate_expression : (36 + 12) / (6 - (2 + 1)) = 16 := by
  sorry

end evaluate_expression_l60_60277


namespace xiao_ming_selects_cooking_probability_l60_60212

theorem xiao_ming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let probability (event: String) := if event ∈ courses then 1 / (courses.length : ℝ) else 0
  probability "cooking" = 1 / 4 :=
by
  sorry

end xiao_ming_selects_cooking_probability_l60_60212


namespace smallest_even_divisible_by_20_and_60_l60_60533

theorem smallest_even_divisible_by_20_and_60 : ∃ x, (Even x) ∧ (x % 20 = 0) ∧ (x % 60 = 0) ∧ (∀ y, (Even y) ∧ (y % 20 = 0) ∧ (y % 60 = 0) → x ≤ y) → x = 60 :=
by
  sorry

end smallest_even_divisible_by_20_and_60_l60_60533


namespace num_pairs_satisfying_equation_and_sum_leq_50_l60_60373

theorem num_pairs_satisfying_equation_and_sum_leq_50 :
  {p : ℕ × ℕ | let a := p.1; let b := p.2 in a + b ≤ 50 ∧ (a ≠ 0 ∧ b ≠ 0) ∧
                                   (a : ℚ) + (b⁻¹ : ℚ) = 9 * ((a⁻¹ : ℚ) + (b : ℚ))}.card = 5 :=
by
  sorry

end num_pairs_satisfying_equation_and_sum_leq_50_l60_60373


namespace remainder_when_divided_by_5_l60_60942

theorem remainder_when_divided_by_5 
  (n : ℕ) 
  (h : n % 10 = 7) : 
  n % 5 = 2 := 
by 
  sorry

end remainder_when_divided_by_5_l60_60942


namespace grain_loss_theorem_l60_60236

noncomputable def grain_loss_problem : Prop :=
  let remaining_grain := 2750
  let loss_rate_initial := 1250
  let rate_increase := 1.15
  let hours := 5
  let grain_lost_1 := loss_rate_initial
  let grain_lost_2 := loss_rate_initial * rate_increase
  let grain_lost_3 := loss_rate_initial * rate_increase^2
  let grain_lost_4 := loss_rate_initial * rate_increase^3
  let grain_lost_5 := loss_rate_initial * rate_increase^4
  let total_grain_lost := grain_lost_1 + grain_lost_2 + grain_lost_3 + grain_lost_4 + grain_lost_5
  let G := total_grain_lost + remaining_grain
  G = 11178.03

theorem grain_loss_theorem : grain_loss_problem := 
begin
  -- the detailed proof would go here
  sorry
end

end grain_loss_theorem_l60_60236


namespace find_k_l60_60921

theorem find_k (x y k : ℝ) :
  (x - y = k + 2) →
  (x + 3y = k) →
  (x + y = 2) →
  k = 1 :=
by
  intros h1 h2 h3
  -- The proof steps would go here
  sorry

end find_k_l60_60921


namespace sin_cos_B_range_l60_60348

theorem sin_cos_B_range (a b c : ℝ) (A B C : ℝ) 
  (h_geometric : b / a = c / b) (h_triangle : ∠A = A ∧ ∠B = B ∧ ∠C = C) 
  (h_sides_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  1 < sin B + cos B ∧ sin B + cos B ≤ sqrt 2 := 
sorry

end sin_cos_B_range_l60_60348


namespace define_interval_l60_60308

theorem define_interval (x : ℝ) : 
  (0 < x + 2) → (0 < 5 - x) → (-2 < x ∧ x < 5) :=
by
  intros h1 h2
  sorry

end define_interval_l60_60308


namespace johns_total_spent_l60_60426

def original_price : Float := 20
def discount_rate : Float := 0.15
def number_of_pins : Nat := 10

theorem johns_total_spent : 
  let discount_on_each := original_price * discount_rate
  let discounted_price := original_price - discount_on_each
  let total_cost := discounted_price * (number_of_pins : Float)
  total_cost = 170 := 
by 
  sorry

end johns_total_spent_l60_60426


namespace probability_of_1_10_100_1000_l60_60571

noncomputable def prob_of_display (N : ℕ) (target : ℕ) : ℝ :=
  if target < N then 1 / (target + 1) else 0

theorem probability_of_1_10_100_1000 (N : ℕ) :
  N = 2003 →
  ((prob_of_display N 1000) * (prob_of_display 1000 100) *
  (prob_of_display 100 10) * (prob_of_display 10 1)) = 1 / 2224222 :=
by
  intro hN
  rw [hN]
  dsimp [prob_of_display]
  norm_num
  exact sorry

end probability_of_1_10_100_1000_l60_60571


namespace find_f_neg3_l60_60634

noncomputable def f : ℝ → ℝ :=
sorry

axiom f_xy (x y : ℝ) : f(x + y) = f(x) + f(y) + 2 * x * y
axiom f_1 : f 1 = 2

theorem find_f_neg3 : f (-3) = 6 :=
by
  sorry

end find_f_neg3_l60_60634


namespace loss_percentage_grinder_l60_60791

-- Conditions
def CP_grinder : ℝ := 15000
def CP_mobile : ℝ := 8000
def profit_mobile : ℝ := 0.10
def total_profit : ℝ := 200

-- Theorem to prove the loss percentage on the grinder
theorem loss_percentage_grinder : 
  ( (CP_grinder - (23200 - (CP_mobile * (1 + profit_mobile)))) / CP_grinder ) * 100 = 4 :=
by
  sorry

end loss_percentage_grinder_l60_60791


namespace geometric_sequence_11th_term_l60_60484

theorem geometric_sequence_11th_term (a r : ℝ) (h₁ : a * r ^ 4 = 8) (h₂ : a * r ^ 7 = 64) : 
  a * r ^ 10 = 512 :=
by sorry

end geometric_sequence_11th_term_l60_60484


namespace value_of_k_l60_60920

theorem value_of_k (x y k : ℝ) (h1 : x - y = k + 2) (h2 : x + 3y = k) (h3 : x + y = 2) : k = 1 :=
by
  sorry

end value_of_k_l60_60920


namespace distance_equality_l60_60047

variables {V : Type*} [inner_product_space ℝ V]
variables {A B C P G : V}

def is_centroid (G A B C : V) : Prop :=
  G = (A + B + C) / 3

theorem distance_equality (hG : is_centroid G A B C) :
  dist P A ^ 2 + dist P B ^ 2 + dist P C ^ 2 =
  3 * dist P G ^ 2 + (dist G A ^ 2 + dist G B ^ 2 + dist G C ^ 2) / 2 :=
sorry

end distance_equality_l60_60047


namespace subset_cardinality_l60_60360

def function_subset_intersection (f : ℝ → ℝ) (a b : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd = f p.fst ∧ a ≤ p.fst ∧ p.fst ≤ b} ∩ {p : ℝ × ℝ | p.fst = 2}

theorem subset_cardinality (f : ℝ → ℝ) (a b : ℝ) :
  #((function_subset_intersection f a b).to_finset) = 0 ∨
  #(function_subset_intersection f a b).to_finset = 1 ∨
  #(function_subset_intersection f a b).to_finset = 2 :=
sorry

end subset_cardinality_l60_60360


namespace quadrant_of_z_l60_60687

open Complex

theorem quadrant_of_z (m : ℝ) (h1 : ∀ z : ℂ, z * I = I + m) (h2 : ∀ z : ℂ, z.im = 1) :
  (1 : ℝ) ∈ set.Ioi (0 : ℝ) :=
sorry

end quadrant_of_z_l60_60687


namespace most_followers_is_sarah_l60_60853

def initial_followers_susy : ℕ := 100
def initial_followers_sarah : ℕ := 50

def susy_week1_new : ℕ := 40
def susy_week2_new := susy_week1_new / 2
def susy_week3_new := susy_week2_new / 2
def susy_total_new := susy_week1_new + susy_week2_new + susy_week3_new
def susy_final_followers := initial_followers_susy + susy_total_new

def sarah_week1_new : ℕ := 90
def sarah_week2_new := sarah_week1_new / 3
def sarah_week3_new := sarah_week2_new / 3
def sarah_total_new := sarah_week1_new + sarah_week2_new + sarah_week3_new
def sarah_final_followers := initial_followers_sarah + sarah_total_new

theorem most_followers_is_sarah : 
    sarah_final_followers ≥ susy_final_followers := by
  sorry

end most_followers_is_sarah_l60_60853


namespace valid_lineup_count_l60_60454

noncomputable def num_valid_lineups : ℕ :=
  let total_lineups := Nat.choose 18 8
  let unwanted_lineups := Nat.choose 14 4
  total_lineups - unwanted_lineups

theorem valid_lineup_count : num_valid_lineups = 42757 := by
  sorry

end valid_lineup_count_l60_60454


namespace binomial_coefficient_odd_probability_l60_60531

theorem binomial_coefficient_odd_probability :
  let coefficients := list.map (λ r : ℕ, nat.choose 10 r) (list.range 11),
      odd_coefficients := list.filter (λ c, c % 2 = 1) coefficients in
  (odd_coefficients.length : ℚ) / (coefficients.length : ℚ) = 4 / 11 :=
by sorry

end binomial_coefficient_odd_probability_l60_60531


namespace gcf_of_40_and_14_l60_60932

theorem gcf_of_40_and_14 : ∀ (n : ℕ), n = 40 → Int.gcd n 14 = 10 :=
by
  intro n
  intros h_n_eq_40
  rw [h_n_eq_40, Int.coe_nat_gcd]
  have h_lcm : Int.lcm 40 14 = 56 := by sorry
  have prod_eq : (40:ℤ) * 14 = 560 := by norm_num
  have gcf_eq : 56 * (Int.gcd 40 14) = 560 := by sorry
  have h_gcf : Int.gcd 40 14 = 10 := by {
    rw [← prod_eq, ← mul_assoc, ← eq_div_iff_mul_eq] at gcf_eq,
    norm_num at *,
    assumption,
  }
  exact h_gcf

end gcf_of_40_and_14_l60_60932


namespace label_of_4th_selected_individual_l60_60993

def population : List String := ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
                                 "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                                 "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
                                 "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"]

def random_number_table : List String := ["0618", "0765", "4544", "1816", "5809", "7983", "8619",
                                          "7606", "8350", "0310", "5923", "4605", "0526", "6238"]

-- conditions as definitions
noncomputable def select_4th_individual : String :=
  let selection := [random_number_table[0].substr 2 2,
                    random_number_table[0].substr 4 2,
                    random_number_table[1].substr 0 2,
                    random_number_table[1].substr 2 2]
  selection[3]

-- statement to prove
theorem label_of_4th_selected_individual : select_4th_individual = "09" := by
  sorry

end label_of_4th_selected_individual_l60_60993


namespace area_of_isosceles_trapezoid_with_inscribed_circle_l60_60294

-- Definitions
def is_isosceles_trapezoid_with_inscribed_circle (a b c : ℕ) : Prop :=
  a + b = 2 * c

def height_of_trapezoid (c : ℕ) (half_diff_bases : ℕ) : ℕ :=
  (c^2 - half_diff_bases^2).sqrt

noncomputable def area_of_trapezoid (a b h : ℕ) : ℕ :=
  (a + b) * h / 2

-- Given values
def base1 := 2
def base2 := 8
def leg := 5
def height := 4

-- Proof Statement
theorem area_of_isosceles_trapezoid_with_inscribed_circle :
  is_isosceles_trapezoid_with_inscribed_circle base1 base2 leg →
  height_of_trapezoid leg ((base2 - base1) / 2) = height →
  area_of_trapezoid base1 base2 height = 20 :=
by
  intro h1 h2
  sorry

end area_of_isosceles_trapezoid_with_inscribed_circle_l60_60294


namespace substitution_ways_mod_1000_l60_60999

def num_ways (k : ℕ) : ℕ :=
if k = 0 then 1 
else 11 ^ (2 * k) * (10 * 9 * 8) ^ (k - 1)

def total_ways : ℕ := (0 : ℕ → ℕ).sum (λ i, num_ways i)

theorem substitution_ways_mod_1000 (n : ℕ) (k : ℕ) (n = total_ways k) : n % 1000 = 722 :=
by {
  sorry
}

end substitution_ways_mod_1000_l60_60999


namespace remaining_number_is_odd_l60_60823

theorem remaining_number_is_odd : 
  ∃ n ∈ ({1, 3, 5, ..., 49} : set ℕ), 
  (∃ f : fin 50 → ℕ, 
    (∀ i, 1 ≤ f i ∧ f i ≤ 50) ∧ 
    -- condition that f represents the sequence of numbers on the board
    (∀ n < 49, 
      ∃ i j, i ≠ j ∧ f (n + 1) = |f n i - f n j| ∧ 
      -- repeated operation that results in a single number
      (∀ k, k ≠ i ∧ k ≠ j → f (n + 1) k = f n k))) ∧
    (∀ n1 n2 ∈ ({1, 3, 5, ..., 49} : set ℕ), parity n1 = parity n2) → 
    (∃ k, f 49 k ∈ ({1, 3, 5, ..., 49} : set ℕ)) :=
begin
  -- Insert proof here
  sorry
end

end remaining_number_is_odd_l60_60823


namespace union_A_B_l60_60448

def A : Set ℝ := {x | ∃ y : ℝ, y = Real.log x}
def B : Set ℝ := {x | x < 1}

theorem union_A_B : (A ∪ B) = Set.univ :=
by
  sorry

end union_A_B_l60_60448


namespace fill_20_cans_l60_60587

theorem fill_20_cans (total_gallons : ℕ) (rate_full_capacity : ℕ → ℕ → ℕ) (hours : ℕ) 
  (num_cans_full_capacity : ℕ) (gallons_per_can : ℕ) (capacity_fraction : ℚ) : 
  (rate_full_capacity 25 5) = 5 → 
  (total_gallons = rate_full_capacity 25 5 * gallons_per_can * hours) → 
  (capacity_fraction = 3 / 4) → 
  ∃ num_cans_quarter_capacity, (gallons_per_can * capacity_fraction) * num_cans_quarter_capacity = total_gallons → 
  num_cans_quarter_capacity = 20 :=
by
  intros rate_full_capacity_eq total_gallons_eq capacity_fraction_eq
  have : total_gallons = 120 := by rw [total_gallons_eq, rate_full_capacity_eq]; norm_num
  have : gallons_per_can * capacity_fraction = 6 := by rw [capacity_fraction_eq]; norm_num
  use 20
  rw [this, mul_comm]
  sorry

end fill_20_cans_l60_60587


namespace find_radius_square_l60_60575

-- Define the given conditions; AB, CD are chords, intersect P, angle APD = 90 degrees, BP = 7
def chord_ab := 12
def chord_cd := 9
def bp := 7
def angle_apd := 90

-- The main theorem to prove
theorem find_radius_square (chord_ab : ℝ) (chord_cd : ℝ) (bp : ℝ) (angle_apd : ℝ): (r^2 : ℝ) := 
sorry

-- Lean theorem statement based on given math problem and solution.

end find_radius_square_l60_60575


namespace least_possible_value_minimum_at_zero_zero_l60_60528

theorem least_possible_value (x y : ℝ) : (xy - 1)^2 + (x + y)^2 ≥ 1 :=
begin
  sorry
end

theorem minimum_at_zero_zero : (xy - 1)^2 + (x + y)^2 = 1 ↔ x = 0 ∧ y = 0 :=
begin
  sorry
end

end least_possible_value_minimum_at_zero_zero_l60_60528


namespace cartesian_equation_of_curve_perpendicular_value_l60_60704

-- Define the polar equation
def polar_equation (rho theta : ℝ) : Prop :=
  rho ^ 2 = 4 / (cos theta ^ 2 + 4 * sin theta ^ 2)

-- Proof for Cartesian equation conversion
theorem cartesian_equation_of_curve (rho theta : ℝ) (x y : ℝ) :
  polar_equation rho theta →
  x = rho * cos theta →
  y = rho * sin theta →
  x^2 / 4 + y^2 = 1 :=
by
  intro h_polar h_x h_y
  sorry

-- Define the perpendicularity condition
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  x₁ * x₂ + y₁ * y₂ = 0

-- Proof for the value when OP ⊥ OQ
theorem perpendicular_value (P Q : ℝ × ℝ) (rho_P rho_Q : ℝ) (theta_P theta_Q : ℝ) :
  perpendicular P Q →
  polar_equation rho_P theta_P →
  polar_equation rho_Q theta_Q →
  P = (rho_P * cos theta_P, rho_P * sin theta_P) →
  Q = (rho_Q * cos theta_Q, rho_Q * sin theta_Q) →
  theta_Q = theta_P + π / 2 ∨ theta_Q = theta_P - π / 2 →
  1 / (rho_P ^ 2) + 1 / (rho_Q ^ 2) = 5 / 4 :=
by
  intro h_perpendicular h_polar_P h_polar_Q h_P h_Q h_theta
  sorry

end cartesian_equation_of_curve_perpendicular_value_l60_60704


namespace minimum_blue_beads_l60_60985

theorem minimum_blue_beads (n : ℕ) (hr : n = 100) (h : ∀ s : finset ℕ, s.card = 10 → ∃ t : finset ℕ, t.card ≥ 7 ∧ t ⊆ {x | x ∉ s}) : ∃ k : ℕ, k = 78 :=
sorry

end minimum_blue_beads_l60_60985


namespace even_of_central_symmetry_l60_60209

theorem even_of_central_symmetry (n k : ℕ) : 
  (∃ (figure : Type), centrally_symmetric figure ∧ corners figure = n ∧ rectangles_1x4 figure = k) 
  → Even n :=
by
  sorry

end even_of_central_symmetry_l60_60209


namespace john_spent_170_l60_60424

def discount_amount (orig_price : ℝ) (discount_percent : ℝ) : ℝ :=
  orig_price * (discount_percent / 100)

def sale_price (orig_price : ℝ) (discount_amount : ℝ) : ℝ :=
  orig_price - discount_amount

def total_amount_spent (sale_price : ℝ) (quantity : ℝ) : ℝ :=
  sale_price * quantity

theorem john_spent_170 :
  let orig_price := 20
  let discount_percent := 15
  let quantity := 10
  discount_amount orig_price discount_percent = 3 →
  sale_price orig_price (discount_amount orig_price discount_percent) = 17 →
  total_amount_spent (sale_price orig_price (discount_amount orig_price discount_percent)) quantity = 170 :=
by
  intros h1 h2
  rw h1
  rw h2
  sorry

end john_spent_170_l60_60424


namespace samantha_more_posters_l60_60450

theorem samantha_more_posters :
  ∃ S : ℕ, S > 18 ∧ 18 + S = 51 ∧ S - 18 = 15 :=
by
  sorry

end samantha_more_posters_l60_60450


namespace no_valid_m_n_l60_60667

theorem no_valid_m_n (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) : ¬ (m * n ∣ 3^m + 1 ∧ m * n ∣ 3^n + 1) :=
by
  sorry

end no_valid_m_n_l60_60667


namespace seq_arithmetic_general_formula_sum_formula_sum_inequality_l60_60366

open Nat

def a (n : ℕ) : ℕ → ℕ
| 1        := 1
| (n+1) := 2 * a n + 2^(n+1)

def b (n : ℕ) : ℕ → ℚ
| 1     := 1 / 2
| (n+1) := b n + 1

theorem seq_arithmetic (n : ℕ) (hn : 1 < n) : 
    ∃ d : ℚ, d = 1 ∧ b n = b 1 + (n - 1) * d := 
sorry

theorem general_formula (n : ℕ) : 
    ∃ f : ℕ → ℚ, a n = (n - 1 / 2) * 2^n := 
sorry

def S (n : ℕ) : ℚ := 
  ∑ i in range n, a i

theorem sum_formula (n : ℕ) : 
    S n = (2 * n - 3) * 2^n + 3 := 
sorry

theorem sum_inequality (n : ℕ) : 
    (S n) / (2^n) > 2 * n - 3 := 
sorry

end seq_arithmetic_general_formula_sum_formula_sum_inequality_l60_60366


namespace ratio_fraction_eq_l60_60297

theorem ratio_fraction_eq (x y : ℚ) :
  (3 / 7) / (x / y) = (2 / 5) / (1 / 7) → (x / y) = 15 / 98 :=
by
  assume h : (3 / 7) / (x / y) = (2 / 5) / (1 / 7)
  -- continue from here; skipping proof
  sorry

end ratio_fraction_eq_l60_60297


namespace parallel_lines_implies_value_of_m_l60_60483

theorem parallel_lines_implies_value_of_m :
  ∀ (m : ℝ), (∀ (x y : ℝ), 3 * x + 2 * y - 2 = 0) ∧ (∀ (x y : ℝ), (2 * m - 1) * x + m * y + 1 = 0) → 
  m = 2 := 
by
  sorry

end parallel_lines_implies_value_of_m_l60_60483


namespace reciprocal_neg_half_l60_60117

theorem reciprocal_neg_half : 1 / (-1 / 2 : ℝ) = (-2 : ℝ) :=
by
  sorry

end reciprocal_neg_half_l60_60117


namespace coastal_city_spending_l60_60107

def beginning_of_may_spending : ℝ := 1.2
def end_of_september_spending : ℝ := 4.5

theorem coastal_city_spending :
  (end_of_september_spending - beginning_of_may_spending) = 3.3 :=
by
  -- Proof can be filled in here
  sorry

end coastal_city_spending_l60_60107


namespace smallest_triangle_perimeter_l60_60347

noncomputable def triangle_perimeter (a b c : ℕ) : ℕ :=
  a + b + c

theorem smallest_triangle_perimeter :
  ∃ (a b c : ℕ),
    (a + b + c = 9) ∧
    (cos (A : ℝ) = 11 / 16) ∧
    (cos (B : ℝ) = 7 / 8) ∧
    (cos (C : ℝ) = -1 / 4) ∧
    (a + b > c) ∧ (a + c > b) ∧ (b + c > a) := sorry

end smallest_triangle_perimeter_l60_60347


namespace complete_the_square_k_l60_60009

theorem complete_the_square_k (x : ℝ) : ∃ k : ℝ, (∃ a h: ℝ, (a = 1) ∧ (h = 7 / 2) ∧ (x^2 - 7*x = a * (x - h)^2 + k)) → k = -49 / 4 :=
by
  sorry

end complete_the_square_k_l60_60009


namespace dice_opposite_face_l60_60863

theorem dice_opposite_face (a b : ℕ) (ab_condition_1 : a ≠ 7) (ab_condition_2 : b ≠ 7) 
    (numbers : Finset ℕ := {6, 7, 8, 9, 10, 11}) 
    (sum_equal_33 : (33 : ℕ) = (numbers.sum - a - b))
    (sum_equal_35 : (35 : ℕ) = (numbers.sum - a - 7)) 
    (sum_of_faces : numbers.sum = 51) :
    (a = 9 ∧ b = 11) ∨ (a = 11 ∧ b = 9) := 
by
  sorry

end dice_opposite_face_l60_60863


namespace sufficient_but_not_necessary_l60_60386

theorem sufficient_but_not_necessary (a : ℝ) : (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  (∀ x : ℝ, (x - 1) * (x - 2) = 0 → x ≠ 2 → x = 1) ∧
  (a = 2 → (1 ≠ 2)) :=
by {
  sorry
}

end sufficient_but_not_necessary_l60_60386


namespace Pam_bags_count_l60_60825

theorem Pam_bags_count :
  ∀ (apples_in_geralds_bag : ℕ) (multiple_factor : ℕ) (total_apples_pam_has : ℕ) (expected_pam_bags : ℕ),
  (apples_in_geralds_bag = 40) →
  (multiple_factor = 3) →
  (total_apples_pam_has = 1200) →
  (expected_pam_bags = 10) →
  let apples_in_pams_bag := apples_in_geralds_bag * multiple_factor in
  let pam_bags := total_apples_pam_has / apples_in_pams_bag in
  pam_bags = expected_pam_bags :=
by
  intros apples_in_geralds_bag multiple_factor total_apples_pam_has expected_pam_bags
  intros h1 h2 h3 h4
  let apples_in_pams_bag := apples_in_geralds_bag * multiple_factor
  let pam_bags := total_apples_pam_has / apples_in_pams_bag
  rw [h1, h2, h3, h4]
  sorry

end Pam_bags_count_l60_60825


namespace cricket_player_average_increase_l60_60093

theorem cricket_player_average_increase
  (average : ℕ) (n : ℕ) (next_innings_runs : ℕ) 
  (x : ℕ) 
  (h1 : average = 32)
  (h2 : n = 20)
  (h3 : next_innings_runs = 200)
  (total_runs := average * n)
  (new_total_runs := total_runs + next_innings_runs)
  (new_average := (average + x))
  (new_total := new_average * (n + 1)):
  new_total_runs = 840 →
  new_total = 840 →
  x = 8 :=
by
  sorry

end cricket_player_average_increase_l60_60093


namespace reciprocal_of_neg_two_l60_60887

theorem reciprocal_of_neg_two : ∃ x : ℝ, (-2) * x = 1 ∧ x = -1/2 :=
by
  use -1/2
  split
  · exact (by norm_num : (-2) * (-1/2) = 1)
  · rfl

end reciprocal_of_neg_two_l60_60887


namespace original_square_side_length_l60_60065

theorem original_square_side_length :
  ∃ n k : ℕ, (n + k) * (n + k) - n * n = 47 ∧ k ≤ 5 ∧ k % 2 = 1 ∧ n = 23 :=
by
  sorry

end original_square_side_length_l60_60065


namespace f_diff_f_eqn_f_prime_at_1_l60_60051
noncomputable theory

-- Given function f is differentiable in (0, +∞) and f(e^x) = x + e^x
def f (x : ℝ) : ℝ := sorry

-- Prove that f is differentiable in (0, +∞) and satisfies the given condition
theorem f_diff (x : ℝ) (h : x > 0) : differentiable_at ℝ f x := sorry
theorem f_eqn (x : ℝ) : f (exp x) = x + exp x := sorry

-- Prove that f'(1) = 2
theorem f_prime_at_1 : deriv f 1 = 2 := sorry

end f_diff_f_eqn_f_prime_at_1_l60_60051


namespace seating_arrangements_l60_60404

def count_arrangements (n k : ℕ) : ℕ :=
  (n.factorial) / (n - k).factorial

theorem seating_arrangements : count_arrangements 6 5 * 3 = 360 :=
  sorry

end seating_arrangements_l60_60404


namespace function_range_l60_60540

theorem function_range (x : ℝ) (h : 0 < x) : 
  ¬set.range (λ x, real.sqrt (x^2 - 2 * x)) = set.Ioi 0 ∧
  ¬set.range (λ x, (x + 2) / (x + 1)) = set.Ioi 0 ∧
  ¬set.range (λ (x : ℕ), 1 / (x^2 + 2 * x + 1)) = set.Ioi 0 ∧
  set.range (λ x, 1 / |x - 1|) = set.Ioi 0 := 
by sorry

end function_range_l60_60540


namespace num_valid_x_l60_60672

def is_valid_x (x : ℕ) : Prop := 30 < x^2 + 6x + 16 ∧ x^2 + 6x + 16 < 50

theorem num_valid_x : {x : ℕ | is_valid_x x}.to_finset.card = 2 := by
  sorry

end num_valid_x_l60_60672


namespace angle_measure_ade_l60_60747

/-- In ΔABC, D is the midpoint of side AC, and E is a point on side BC such that BE = EC. 
Given that ∠ECB = 50°, prove that the degree measure of ∠ADE is 40°. -/
theorem angle_measure_ade (A B C D E : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
  (h1: D = (A + C) / 2) 
  (h2: BE = EC) 
  (h3: ∠ECB = 50) :
  ∠ADE = 40 :=
by {
  sorry
}

end angle_measure_ade_l60_60747


namespace monotonic_increasing_implies_range_l60_60715

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / 2 * x ^ 2 + 2 * x - 2 * log x

theorem monotonic_increasing_implies_range (a : ℝ) :
  (∀ x > (0 : ℝ), deriv f x ≥ 0) → a ≤ 1 :=
  by 
  sorry

end monotonic_increasing_implies_range_l60_60715


namespace maximum_candies_karlson_l60_60511

theorem maximum_candies_karlson (n : ℕ) (h_n : n = 40) :
  ∃ k, k = 780 :=
by
  sorry

end maximum_candies_karlson_l60_60511


namespace tractor_can_scoop_75_pounds_per_minute_l60_60647

def tractor_rate (T : ℝ) : Prop :=
  let Darrel_Rate := 10  -- Darrel's rate in pounds per minute
  let combined_rate := T + Darrel_Rate
  let compost_loaded := combined_rate * 30  -- Compost loaded in 30 minutes
  compost_loaded = 2550  -- Total compost loaded

theorem tractor_can_scoop_75_pounds_per_minute :
  ∃ T : ℝ, tractor_rate T ∧ T = 75 :=
by
  use 75
  unfold tractor_rate
  simp only [add_mul, one_mul, mul_add, algebra.id.smul_eq_mul]
  sorry

end tractor_can_scoop_75_pounds_per_minute_l60_60647


namespace find_expression_for_fx_l60_60146

theorem find_expression_for_fx (f : ℝ → ℝ) :
  (∀ x, f (x + 2) = -(2 : ℝ) ^ x) ↔ (f = λ x, -(2 : ℝ) ^ (x + 2)) :=
by
  sorry

end find_expression_for_fx_l60_60146


namespace part_a_part_b_part_c_l60_60553

noncomputable def f (x y : ℝ) : ℝ := x^2 + x * y + y^2

theorem part_a (x y : ℝ) : ∃ (m n : ℤ), f (x - m) (y - n) ≤ 1/2 := 
sorry


theorem part_b (x y : ℝ) : 
  let bar_f := infi (λ mn : ℤ × ℤ, f (x - mn.1) (y - mn.2)) in
  bar_f ≤ 1/3 :=
sorry

noncomputable def f_a (a x y : ℝ) : ℝ := x^2 + a * x * y + y^2

theorem part_c (a : ℝ) (h : 0 ≤ a ∧ a ≤ 2) :
  ∃ (c : ℝ), ∀ (x y : ℝ), abs (f_a a x y) ≤ c := 
sorry

end part_a_part_b_part_c_l60_60553


namespace area_of_closed_figure_when_k_eq_zero_monotonic_intervals_when_k_gt_zero_l60_60718

noncomputable def f (k x : ℝ) := k * x^3 - 3 * x^2 + 3

theorem area_of_closed_figure_when_k_eq_zero :
  ∫ x in (-4/3 : ℝ)..1, (-3 * x^2 + 3 - (x - 1)) = 343 / 54 := 
begin
  sorry
end

theorem monotonic_intervals_when_k_gt_zero (k : ℝ) (h : 0 < k) :
  ∀ x : ℝ, (f' k x > 0 ↔ x < 0 ∨ x > 2 / k) ∧ (f' k x < 0 ↔ 0 < x ∧ x < 2 / k) := 
begin
  sorry
end

noncomputable def f' (k x : ℝ) := 3 * k * x^2 - 6 * x

end area_of_closed_figure_when_k_eq_zero_monotonic_intervals_when_k_gt_zero_l60_60718


namespace reciprocal_of_neg_two_l60_60891

theorem reciprocal_of_neg_two : ∃ y : ℝ, (-2) * y = 1 ∧ y = -1/2 :=
by
  use -1/2
  split
  · -- Prove that (-2) * (-1/2) = 1
    sorry
  · -- Prove that y = -1/2
    rfl

end reciprocal_of_neg_two_l60_60891


namespace logarithm_simplification_l60_60083

theorem logarithm_simplification :
  (1 / (Real.log 3 / Real.log 12 + 1) + 1 / (Real.log 2 / Real.log 8 + 1) + 1 / (Real.log 7 / Real.log 9 + 1)) =
  1 - (Real.log 7 / Real.log 1008) :=
sorry

end logarithm_simplification_l60_60083


namespace intercept_sum_modulo_l60_60960

theorem intercept_sum_modulo (x_0 y_0 : ℤ) (h1 : 0 ≤ x_0) (h2 : x_0 < 17) (h3 : 0 ≤ y_0) (h4 : y_0 < 17)
                       (hx : 5 * x_0 ≡ 2 [ZMOD 17])
                       (hy : 3 * y_0 ≡ 15 [ZMOD 17]) :
    x_0 + y_0 = 19 := 
by
  sorry

end intercept_sum_modulo_l60_60960


namespace real_solution_eq_l60_60603

theorem real_solution_eq (x : ℝ) : 
  (x^2 - 2 * x) ^ (x^2 + x - 6) = 1 ↔ 
  x = -3 ∨ x = 1 + real.sqrt 2 ∨ x = 1 - real.sqrt 2 ∨ x = 1 :=
sorry

end real_solution_eq_l60_60603


namespace proof_1_proof_2_proof_3_l60_60566

-- Condition 1: 1,000 cm³ = 1 L
def cm³_to_L: ℝ := 1000

-- Proof 1: 3500 cm³ = 3.5 L
theorem proof_1 : 3500 / cm³_to_L = 3.5 :=
by
  sorry

-- Condition 2: 1 L = 1,000 mL
def L_to_mL: ℝ := 1000

-- Proof 2: 7.2 L = 7200 mL
theorem proof_2 : 7.2 * L_to_mL = 7200 :=
by
  sorry

-- Condition 3: 1 m³ = 1,000 dm³
def m³_to_dm³: ℝ := 1000

-- Proof 3: 5 m³ = 5000 dm³
theorem proof_3 : 5 * m³_to_dm³ = 5000 :=
by
  sorry

end proof_1_proof_2_proof_3_l60_60566


namespace prove_BH_eq_CX_l60_60766

-- Define the main problem, starting with the acute and not isosceles triangle
variables (A B C M Q P X H: Type) 
variables [triangle ABC]

-- Define specific conditions given in the problem statement
variable (acute_angled : is_acutriangle A B C)
variable (not_isosceles : ¬ is_isosceles A B C)
variable (median : is_median AM)
variable (height : is_height_from_vertex AH)
variable (perpendicular1 : perpendicular QM AC)
variable (perpendicular2 : perpendicular PM AB)
variable (circumcircle_intersection : is_circumcircle_intersection PMQ BC X)
variable (BH : Type)
variable (CX : Type)

-- State the main theorem to be proved
theorem prove_BH_eq_CX : BH = CX :=
sorry

end prove_BH_eq_CX_l60_60766


namespace sum_of_interior_angles_octagon_l60_60505

theorem sum_of_interior_angles_octagon : (8 - 2) * 180 = 1080 :=
by
  sorry

end sum_of_interior_angles_octagon_l60_60505


namespace probability_Laurent_greater_Chloe_l60_60620

-- Definitions for conditions
def Chloe (x : ℝ) := x ∈ set.Icc 0 100
def Laurent (y : ℝ) := y ∈ set.Icc 0 200

-- Statement of the problem
theorem probability_Laurent_greater_Chloe :
  Prob (λ (x y : ℝ), Laurent y ∧ Chloe x ∧ y > x) = 3 / 4 := sorry

end probability_Laurent_greater_Chloe_l60_60620


namespace total_dots_not_visible_eq_54_l60_60676

theorem total_dots_not_visible_eq_54 :
  let die_sum := 21
  let num_dice := 4
  let total_sum := num_dice * die_sum
  let visible_sum := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6
  total_sum - visible_sum = 54 :=
by
  let die_sum := 21
  let num_dice := 4
  let total_sum := num_dice * die_sum
  let visible_sum := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6
  show total_sum - visible_sum = 54
  sorry

end total_dots_not_visible_eq_54_l60_60676


namespace find_side_length_of_square_l60_60067

theorem find_side_length_of_square (n k : ℕ) (hk : k ≥ 1) (h : (n + k) * (n + k) - n * n = 47) : n = 23 :=
  sorry

end find_side_length_of_square_l60_60067


namespace number_of_correct_propositions_l60_60461

noncomputable def is_correct_proposition_1 (k : ℝ) := ∀ x : ℝ, x > 0 → x^k ≥ 0
noncomputable def is_correct_proposition_2 (k : ℝ) := k < 0 → ∀ x : ℝ, x > 0 → x^k = (x⁻¹)^-k
noncomputable def is_correct_proposition_3 (k : ℝ) := k > 0 → ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁^k < x₂^k
noncomputable def has_at_least_two_intersections (k : ℝ) := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^k = x₁^(-k) ∧ x₂^k = x₂^(-k)

theorem number_of_correct_propositions (k : ℝ) : 
  (is_correct_proposition_1 k → true) ∧
  (¬is_correct_proposition_2 k → true) ∧
  (¬is_correct_proposition_3 k → true) ∧
  (¬has_at_least_two_intersections k → true) → 
  1 = 1 := 
by
  sorry

end number_of_correct_propositions_l60_60461


namespace find_s_l60_60440

noncomputable def g (x : ℝ) (p q r s : ℝ) := x^4 + p*x^3 + q*x^2 + r*x + s

theorem find_s (p q r s : ℝ) (neg_roots : ∀ r_ : ℝ, r_ ∈ {-1, -2, -12, -25}) :
  p + q + r + s = 2023 → 
  (∃ s_1 s_2 s_3 s_4 : ℝ, s_1∈{1,2,12,25} ∧ s_2∈{1,2,12,25} ∧ s_3∈{1,2,12,25} ∧ s_4∈{1,2,12,25} ∧
    g x p q r s = (x + s_1)*(x + s_2)*(x + s_3)*(x + s_4) ∧
    s = 600) :=
begin
  sorry
end

end find_s_l60_60440


namespace congruent_if_completely_overlap_l60_60946

theorem congruent_if_completely_overlap (T1 T2 : Triangle) 
  (A : ¬(same_shape T1 T2 → congruent T1 T2))
  (B : ¬(equal_area T1 T2 → congruent T1 T2))
  (C : (completely_overlap T1 T2 → congruent T1 T2))
  (D : congruent T1 T2 → (equal_perimeter T1 T2 ∧ equal_area T1 T2)) :
  completely_overlap T1 T2 → congruent T1 T2 :=
C

end congruent_if_completely_overlap_l60_60946


namespace tangent_line_at_1_1_l60_60657

noncomputable def f (x : ℝ) : ℝ := x / (2 * x - 1)

theorem tangent_line_at_1_1 :
  let m := -((2 * 1 - 1 - 2 * 1) / (2 * 1 - 1)^2) -- Derivative evaluated at x = 1
  let tangent_line (x y : ℝ) := x + y - 2
  ∀ x y : ℝ, tangent_line x y = 0 → (f x = y ∧ x = 1 → y = 1 → m = -1) :=
by
  sorry

end tangent_line_at_1_1_l60_60657


namespace students_present_l60_60136

theorem students_present (total_students : ℕ) (absent_percent : ℝ) (total_absent : ℝ) (total_present : ℝ) :
  total_students = 50 → absent_percent = 0.12 → total_absent = total_students * absent_percent →
  total_present = total_students - total_absent →
  total_present = 44 :=
by
  intros _ _ _ _; sorry

end students_present_l60_60136


namespace reciprocal_of_neg_two_l60_60878

-- Define the condition of reciprocal
def is_reciprocal (y x : ℝ) : Prop := y * x = 1

-- Define the number in question
def y : ℝ := -2

-- Define the expected reciprocal
def x : ℝ := -1 / 2

-- State the theorem
theorem reciprocal_of_neg_two : is_reciprocal y x :=
by
  -- Proof can be provided here
  sorry

end reciprocal_of_neg_two_l60_60878


namespace train_B_time_to_destination_l60_60519

theorem train_B_time_to_destination (speed_A : ℕ) (time_A : ℕ) (speed_B : ℕ) (dA : ℕ) :
  speed_A = 100 ∧ time_A = 9 ∧ speed_B = 150 ∧ dA = speed_A * time_A →
  dA / speed_B = 6 := 
by
  sorry

end train_B_time_to_destination_l60_60519


namespace problem_sum_of_k_l60_60438

theorem problem_sum_of_k {a b c k : ℂ} (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_ratio : a / (1 - b) = k ∧ b / (1 - c) = k ∧ c / (1 - a) = k) :
  (if (k^2 - k + 1 = 0) then -(-1)/1 else 0) = 1 :=
sorry

end problem_sum_of_k_l60_60438


namespace direction_vector_of_line_l60_60101

theorem direction_vector_of_line : ∃ Δx Δy : ℚ, y = - (1/2) * x + 1 → Δx = 2 ∧ Δy = -1 :=
sorry

end direction_vector_of_line_l60_60101


namespace find_subtracted_value_l60_60211

theorem find_subtracted_value (n x : ℕ) (h₁ : n = 36) (h₂ : ((n + 10) * 2 / 2 - x) = 44) : x = 2 :=
by
  sorry

end find_subtracted_value_l60_60211


namespace intersection_eq_zero_l60_60038

def M := { x : ℤ | abs (x - 3) < 4 }
def N := { x : ℤ | x^2 + x - 2 < 0 }

theorem intersection_eq_zero : M ∩ N = {0} := 
  by
    sorry

end intersection_eq_zero_l60_60038


namespace wine_with_cork_cost_is_2_10_l60_60972

noncomputable def cork_cost : ℝ := 0.05
noncomputable def wine_without_cork_cost : ℝ := cork_cost + 2.00
noncomputable def wine_with_cork_cost : ℝ := wine_without_cork_cost + cork_cost

theorem wine_with_cork_cost_is_2_10 : wine_with_cork_cost = 2.10 :=
by
  -- skipped proof
  sorry

end wine_with_cork_cost_is_2_10_l60_60972


namespace inequality_floors_factorial_ratio_pos_int_l60_60202

-- Part (a): Prove the inequality [5x] + [5y] ≥ [3x + y] + [3y + x]
theorem inequality_floors (x y : ℝ) (hx1 : 1 > x) (hx2 : x ≥ 0.1) (hy1 : 0.1 > y) (hy2 : y ≥ 0) :
  nat.floor (5 * x) + nat.floor (5 * y) ≥ nat.floor (3 * x + y) + nat.floor (3 * y + x) :=
  sorry

-- Part (b): Prove that the given ratio of factorial products is an integer
theorem factorial_ratio_pos_int {m n : ℕ} (hm : m > 0) (hn : n > 0) :
  ∃ k : ℕ, (5 * m).factorial * (5 * n).factorial = k * (m.factorial * n.factorial * (3 * m + n).factorial * (3 * n + m).factorial) :=
  sorry

end inequality_floors_factorial_ratio_pos_int_l60_60202


namespace Petya_wins_for_odd_n_l60_60964

theorem Petya_wins_for_odd_n (n : ℕ) (h : n ≥ 3 ∧ n % 2 = 1) : 
  Petya_wins_for_n n :=
by
  sorry

end Petya_wins_for_odd_n_l60_60964


namespace roots_quadratic_l60_60446

-- Define the roots of the quadratic equation
def is_root (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop :=
  is_root 3 4 (-7) x

-- State that d and e are roots of the given quadratic equation
axiom d : ℝ
axiom e : ℝ
axiom d_root : quadratic_eq d
axiom e_root : quadratic_eq e

-- Use Vieta's formulas to derive d + e and de
lemma vieta_sum : d + e = -4 / 3 :=
by sorry -- Vieta's formula for sum of roots

lemma vieta_product : d * e = -7 / 3 :=
by sorry -- Vieta's formula for product of roots

-- The goal: prove that (d-2)(e-2) = 13/3
theorem roots_quadratic : (d - 2) * (e - 2) = 13 / 3 :=
by calc
  (d - 2) * (e - 2)
      = d * e - 2 * (d + e) + 4   : by ring
  ... = -7 / 3 - 2 * (-4 / 3) + 4 : by rw [vieta_product, vieta_sum]
  ... = (-7 / 3) + (8 / 3) + 4    : by ring
  ... = 1 / 3 + 4                 : by norm_num
  ... = 1 / 3 + 12 / 3            : by norm_num
  ... = 13 / 3                    : by norm_num

end roots_quadratic_l60_60446


namespace total_amount_shared_l60_60452

-- Define the amounts for Ken and Tony based on the conditions
def ken_amt : ℤ := 1750
def tony_amt : ℤ := 2 * ken_amt

-- The proof statement that the total amount shared is $5250
theorem total_amount_shared : ken_amt + tony_amt = 5250 :=
by 
  sorry

end total_amount_shared_l60_60452


namespace sum_fractions_correct_l60_60614

def sum_of_fractions : Prop :=
  (3 / 15 + 5 / 150 + 7 / 1500 + 9 / 15000 = 0.2386)

theorem sum_fractions_correct : sum_of_fractions :=
by
  sorry

end sum_fractions_correct_l60_60614


namespace log_sqrt_defined_range_l60_60313

theorem log_sqrt_defined_range (x: ℝ) : 
  (∃ (y: ℝ), y = (log (5-x) / sqrt (x+2))) ↔ (-2 ≤ x ∧ x < 5) :=
by
  sorry

end log_sqrt_defined_range_l60_60313


namespace log_sum_eq_one_l60_60333

theorem log_sum_eq_one (x y : ℝ) (hx : 2 ^ x = 10) (hy : 5 ^ y = 10) : (1 / x) + (1 / y) = 1 :=
by sorry

end log_sum_eq_one_l60_60333


namespace mul_in_P_l60_60434
noncomputable theory

-- Definition of the set P where P = {m^2 | m ∈ ℕ*}
def P : Set ℕ := {n | ∃ m : ℕ, m > 0 ∧ n = m^2}

-- The theorem we want to prove
theorem mul_in_P {a b : ℕ} (ha : a ∈ P) (hb : b ∈ P) : a * b ∈ P :=
sorry

end mul_in_P_l60_60434


namespace find_x_l60_60378

noncomputable def S (x : ℝ) : ℝ := 1 + 3 * x + 5 * x^2 + 7 * x^3 + ∑' n, (2 * n - 1) * x^n

theorem find_x (x : ℝ) (h : S x = 16) : x = 3/4 :=
sorry

end find_x_l60_60378


namespace curve_equation_satisfied_l60_60579

def curve_parametric (t : ℝ) : ℝ × ℝ :=
  (3 * Real.cos t - 2 * Real.sin t, 3 * Real.sin t)

noncomputable def a : ℝ := 1/9
noncomputable def b : ℝ := -4/27
noncomputable def c : ℝ := 5/81
noncomputable def d : ℝ := 0
noncomputable def e : ℝ := 1/3

theorem curve_equation_satisfied :
  ∀ t : ℝ, let (x, y) := curve_parametric t in
  a * x^2 + b * x * y + c * y^2 + d * x + e * y = 1 := 
by
  sorry

end curve_equation_satisfied_l60_60579


namespace calculate_expression_l60_60261

theorem calculate_expression : (Real.sqrt 8 + Real.sqrt (1 / 2)) * Real.sqrt 32 = 20 := by
  sorry

end calculate_expression_l60_60261


namespace minimum_value_of_f_is_2_point_5_l60_60664

def f (x : ℝ) : ℝ := x + (1 / x) + (1 / (x^2 + 1 / x^2))

theorem minimum_value_of_f_is_2_point_5 (x : ℝ) (h : x > 0) : 
  ∃ c, (c = 2.5) ∧ (∀ y, y > 0 → f y ≥ c) := 
begin
  use 2.5,
  split,
  { exact rfl },
  { sorry }
end

end minimum_value_of_f_is_2_point_5_l60_60664


namespace range_of_f_l60_60049

noncomputable def f (x : ℝ) : ℝ := if x < 1 then 3 * x - 1 else 2 * x ^ 2

theorem range_of_f (a : ℝ) : (f (f a) = 2 * (f a) ^ 2) ↔ (a ≥ 2 / 3 ∨ a = 1 / 2) := 
  sorry

end range_of_f_l60_60049


namespace bricks_required_for_courtyard_l60_60601

noncomputable def courtyard_bricks_required : Nat := by
  -- Given conditions
  let length_min := 20 -- meters
  let length_max := 30 -- meters
  let width_min := 12 -- meters
  let width_max := 20 -- meters
  
  let smallest_area := length_min * width_min -- 240 m²
  let largest_area := length_max * width_max -- 600 m²
  let average_area := (smallest_area + largest_area) / 2 -- 420 m²
  
  let brick_length := 0.20 -- meters
  let brick_width := 0.10 -- meters
  let brick_area := brick_length * brick_width -- 0.02 m²
  
  let number_of_bricks := average_area / brick_area
  exact 21000
  
theorem bricks_required_for_courtyard :
  courtyard_bricks_required = 21000 := sorry

end bricks_required_for_courtyard_l60_60601


namespace juniors_score_l60_60757

theorem juniors_score (n_students : ℕ) (pct_juniors pct_seniors avg_score avg_senior_score : ℝ) (same_score_junior : Prop) 
  (h_students : n_students = 20)
  (h_pct_juniors : pct_juniors = 0.15)
  (h_pct_seniors : pct_seniors = 0.85)
  (h_avg_score : avg_score = 78)
  (h_avg_senior_score : avg_senior_score = 75)
  (h_same_score_junior : same_score_junior = true)
  : ∃ (junior_score : ℝ), junior_score = 95 :=
by
  have h_n_juniors : n_students * pct_juniors = 3 := by linarith [h_students, h_pct_juniors]
  have h_n_seniors : n_students * pct_seniors = 17 := by linarith [h_students, h_pct_seniors]
  have h_total_score : n_students * avg_score = 1560 := by linarith [h_students, h_avg_score]
  have h_total_senior_score : 17 * avg_senior_score = 1275 := by linarith [h_students, h_avg_senior_score, h_pct_seniors]
  have h_total_junior_score : 1560 - 1275 = 285 := by linarith [h_total_score, h_total_senior_score]
  have h_junior_score : 285 / 3 = 95 := by field_simp
  exact ⟨95, h_junior_score⟩
end


end juniors_score_l60_60757


namespace alice_bob_number_sum_l60_60284

noncomputable def is_prime (n : ℕ) : Prop := nat.prime n

theorem alice_bob_number_sum :
  (∃ (A B : ℕ), 1 ≤ A ∧ A ≤ 50 ∧ 1 ≤ B ∧ B ≤ 50 ∧ A ≠ B ∧
    (∀ x, 1 ≤ x ∧ x ≤ 50 → x ≠ A → x ≠ B → x < B → x ≠ B → true) ∧
    is_prime B ∧
    (∃ k, 130 * B + A = k * k)) → 
  (A + B = 13) :=
by 
  sorry

end alice_bob_number_sum_l60_60284


namespace division_equals_fraction_l60_60612

theorem division_equals_fraction:
  180 / (8 + 9 * 3 - 4) = 180 / 31 := 
by
  sorry

end division_equals_fraction_l60_60612


namespace problems_left_to_grade_l60_60246

-- Definitions based on provided conditions
def problems_per_worksheet : ℕ := 4
def total_worksheets : ℕ := 16
def graded_worksheets : ℕ := 8

-- The statement for the required proof with the correct answer included
theorem problems_left_to_grade : 4 * (16 - 8) = 32 := by
  sorry

end problems_left_to_grade_l60_60246


namespace min_period_f_max_min_f_l60_60354

def f (x : ℝ) := 2 * sin (π - x) * cos x + cos (2 * x)

theorem min_period_f : ∀ x, f (x + π) = f x := by
  sorry

theorem max_min_f : ∃ x_max x_min, (x_max ∈ Icc (π/4) (π/2)) ∧ (x_min ∈ Icc (π/4) (π/2)) ∧ 
                        (f x_max = 1) ∧ (f x_min = -1) := by
  sorry

end min_period_f_max_min_f_l60_60354


namespace equal_adjacent_lateral_face_angles_l60_60072

theorem equal_adjacent_lateral_face_angles (n : ℕ) (P : Point) (O : Point) (V : Fin n → Point)
  (hpyramid : RegularPyramid P O V) :
  ∀ i j : Fin n, ∠ (lateral_face P V i) (lateral_face P V j) = ∠ (lateral_face P V (i + 1)) (lateral_face P V (j + 1)) :=
by
  sorry

end equal_adjacent_lateral_face_angles_l60_60072


namespace south_speed_40_l60_60149

variable (north_cyclist_speed : ℝ) (relative_distance : ℝ) (time_hours : ℝ)

-- Given the conditions
axiom north_speed : north_cyclist_speed = 10
axiom distance : relative_distance = 50
axiom time : time_hours = 1

-- Define the south cyclist's speed based on conditions
noncomputable def south_cyclist_speed : ℝ :=
  relative_distance / time_hours - north_cyclist_speed

-- Prove the speed of the cyclist going towards the south is 40 kmph
theorem south_speed_40 : south_cyclist_speed north_cyclist_speed relative_distance time_hours = 40 :=
by
  rw [north_speed, distance, time]
  unfold south_cyclist_speed
  norm_num
  sorry

end south_speed_40_l60_60149


namespace constant_term_binomial_expansion_l60_60860

theorem constant_term_binomial_expansion :
  let binomial_expansion := (λ x : ℝ, (√5 / 5) * x^2 + 1 / x) in
  let general_term := (λ r : ℕ, (nat.choose 6 r) * ((√5 / 5) ^ (6 - r)) * (x ^ (12 - 3 * r))) in
  (∀ r : ℕ, 12 - 3 * r = 0 → general_term r) = 3 :=
by
  let binomial_expansion := (λ x : ℝ, (√5 / 5) * x^2 + 1 / x)
  let general_term := (λ r : ℕ, (nat.choose 6 r) * ((√5 / 5) ^ (6 - r)) * (x ^ (12 - 3 * r)))
  sorry

end constant_term_binomial_expansion_l60_60860


namespace circumcircle_radius_l60_60784

theorem circumcircle_radius (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] :
  ∠A = π / 3 ∧ AB = 2 ∧ AC = 3 → radius (circumcircle_of_triangle ABC) = sqrt 21 / 3 :=
by
  sorry

end circumcircle_radius_l60_60784


namespace similar_oppositely_oriented_triangles_transform_l60_60837

theorem similar_oppositely_oriented_triangles_transform
  (A B C A' B' C' : Type*)
  [AffineSpace A B C] [AffineSpace A' B' C']
  (h_similar : similar_shape A B C A' B' C')
  (h_opposite : opposite_orientation A B C A' B' C') :
  ∃ (l_1 l_2 : Line)
  (h_perpendicular : mutually_perpendicular l_1 l_2)
  (S : Point),
  is_center_of_homothety S l_1 l_2 ∧
  (∃ f, is_composition_of_symmetry_and_homothety f l_1 l_2 S ∧
  f A = A' ∧ f B = B' ∧ f C = C') :=
sorry

end similar_oppositely_oriented_triangles_transform_l60_60837


namespace dice_probability_l60_60943

def is_valid_combination (a b c : ℕ) : Prop := a * b * c = 6

theorem dice_probability : 
  (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ is_valid_combination a b c) → 
  (probability (is_valid_combination a b c) = 1 / 24) :=
by
  sorry

end dice_probability_l60_60943


namespace final_discount_is_30_percent_l60_60792

-- Definitions corresponding to the problem conditions
def original_price : ℝ := 100
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.15
def clearance_discount : ℝ := 0.30

-- Proof statement: proving the final discount is 30%
theorem final_discount_is_30_percent :
  ∃ final_discount : ℝ, 
    final_discount = clearance_discount :=
by
  -- Start of the proof
  use clearance_discount
  exact rfl -- This is a placeholder for "reflexivity" proof
  -- End of proof
sorry

end final_discount_is_30_percent_l60_60792


namespace factor_theorem_l60_60289

theorem factor_theorem (t : ℝ) : (5 * t^2 + 15 * t - 20 = 0) ↔ (t = 1 ∨ t = -4) :=
by
  sorry

end factor_theorem_l60_60289


namespace break_even_machines_l60_60631

def cost_parts : ℤ := 3600
def cost_patent : ℤ := 4500
def selling_price : ℤ := 180

def total_costs : ℤ := cost_parts + cost_patent

def machines_to_break_even : ℤ := total_costs / selling_price

theorem break_even_machines :
  machines_to_break_even = 45 := by
  sorry

end break_even_machines_l60_60631


namespace triangle_side_relation_l60_60834

variable {α β γ : ℝ} -- angles in the triangle
variable {a b c : ℝ} -- sides opposite to the angles

theorem triangle_side_relation
  (h1 : α = 3 * β)
  (h2 : α = 6 * γ)
  (h_sum : α + β + γ = 180)
  : b * c^2 = (a + b) * (a - b)^2 := 
by
  sorry

end triangle_side_relation_l60_60834


namespace domain_of_f_l60_60652

noncomputable def domain_f : Set ℝ := { x : ℝ | ∃ k : ℤ, (π / 6) + 2 * ↑k * π < x ∧ x ≤ (3 * π / 4) + 2 * ↑k * π }

theorem domain_of_f :
  ∀ x, x ∈ domain_f ↔ ∃ k : ℤ, (π / 6) + 2 * ↑k * π < x ∧ x ≤ (3 * π / 4) + 2 * ↑k * π :=
by
  sorry

end domain_of_f_l60_60652


namespace f_strictly_increasing_on_3_inf_l60_60638

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (x^2 - 5*x + 6)

theorem f_strictly_increasing_on_3_inf :
  ∀ x y : ℝ, 3 < x → x < y → y < ∞ → f x < f y := sorry

end f_strictly_increasing_on_3_inf_l60_60638


namespace chocolates_needed_l60_60180

theorem chocolates_needed (chocolates_per_box : ℕ) (total_chocolates : ℕ) (boxes_filled : ℕ) (chocolates_left : ℕ) (additional_needed : ℕ) :
  chocolates_per_box = 30 →
  total_chocolates = 254 →
  boxes_filled = total_chocolates / chocolates_per_box →
  chocolates_left = total_chocolates % chocolates_per_box →
  additional_needed = chocolates_per_box - chocolates_left →
  additional_needed = 16 :=
by
  intros hc ht hf hl ha
  rw [hc, ht, Nat.div_eq_of_lt hc ht] at hf
  rw [hc, ht, Nat.mod_eq_of_lt hc ht] at hl
  rw [hc, hl] at ha
  sorry

end chocolates_needed_l60_60180


namespace custom_op_two_neg_four_l60_60340

-- Define the binary operation *
def custom_op (x y : ℚ) : ℚ := (x * y) / (x + y)

-- Proposition stating 2 * (-4) = 4 using the custom operation
theorem custom_op_two_neg_four : custom_op 2 (-4) = 4 :=
by
  sorry

end custom_op_two_neg_four_l60_60340


namespace minimum_possible_value_of_n_l60_60476

    theorem minimum_possible_value_of_n (n : ℕ) (S : Finset ℕ) 
      (hS_card : S.card = n)
      (hS_mean1 : (∑ x in S, (x : ℚ)) / n = (2/5 : ℚ) * S.max' (Finset.card_pos.2 ⟨n, hS_card.symm.le.trans zero_le_one⟩))
      (hS_mean2 : (∑ x in S, (x : ℚ)) / n = (7/4 : ℚ) * S.min' (Finset.card_pos.2 ⟨n, hS_card.symm.le.trans zero_le_one⟩)) : 
      n ≥ 5 :=
    by
    sorry
    
end minimum_possible_value_of_n_l60_60476


namespace original_square_side_length_l60_60066

theorem original_square_side_length :
  ∃ n k : ℕ, (n + k) * (n + k) - n * n = 47 ∧ k ≤ 5 ∧ k % 2 = 1 ∧ n = 23 :=
by
  sorry

end original_square_side_length_l60_60066


namespace coeff_x4_expansion_l60_60775

theorem coeff_x4_expansion
  (n : ℕ)
  (h1 : (2 : ℕ)^n = 256) :
  ∑ k in Finset.range (n + 1), (Binomial.binom n k * (2 / x : ℚ))^(n - k) * (-x : ℚ)^k = 112 :=
begin
  sorry
end

end coeff_x4_expansion_l60_60775


namespace break_even_machines_l60_60630

theorem break_even_machines (cost_parts cost_patent machine_price total_costs : ℝ):
    cost_parts = 3600 ∧ cost_patent = 4500 ∧ machine_price = 180 ∧ total_costs = cost_parts + cost_patent → 
    total_costs / machine_price = 45 :=
by
  intros h
  cases h with h1 h'
  cases h' with h2 h''
  cases h'' with h3 h4
  sorry

end break_even_machines_l60_60630


namespace power_function_pass_through_point_l60_60722

theorem power_function_pass_through_point (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ a) (h_point : f 2 = 16) : a = 4 :=
sorry

end power_function_pass_through_point_l60_60722


namespace probability_distinct_real_roots_l60_60804

def quadratic_has_two_distinct_real_roots (a b : ℕ) : Prop :=
  a^2 > 8 * b

def count_valid_pairs : ℕ :=
  List.length [ (a, b) | a ← [1, 2, 3, 4, 5, 6], b ← [1, 2, 3, 4, 5, 6], quadratic_has_two_distinct_real_roots a b ]

theorem probability_distinct_real_roots : 
  (count_valid_pairs : ℚ) / 36 = 1 / 4 :=
by
  sorry

end probability_distinct_real_roots_l60_60804


namespace find_largest_n_with_nonempty_domain_l60_60812

def g1 (x : ℝ) := sqrt (4 - x)

noncomputable def gn (n : ℕ) : ℝ → ℝ
| 0     := g1
| (n+1) := (λ x, gn n (sqrt ((n+2)^2 - x)))

theorem find_largest_n_with_nonempty_domain :
  ∃ M d, (∀ n : ℕ, n < M → nonempty (set_of (λ x, ∃ y, gn n y = x))) ∧
  M = 5 ∧ (set_of (λ x, ∃ y, gn 5 y = x)).nonempty ∧ (set_of (λ x, ∃ y, gn 5 y = x)) = {-589} :=
sorry

end find_largest_n_with_nonempty_domain_l60_60812


namespace minimize_PR_PQ_l60_60693

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem minimize_PR_PQ (m : ℝ) :
  let P := (-2, -2)
  let Q := (0, -1)
  ∃ R y, (R = (2, y)) → 
  y = -2 → 
  distance P R + distance P Q = distance P (2, -2) + distance P Q := sorry

end minimize_PR_PQ_l60_60693


namespace cupcakes_difference_l60_60608

theorem cupcakes_difference 
    (B_hrly_rate : ℕ) 
    (D_hrly_rate : ℕ) 
    (B_break : ℕ) 
    (total_hours : ℕ) 
    (B_hrly_rate = 10) 
    (D_hrly_rate = 8) 
    (B_break = 2) 
    (total_hours = 5) :
    (D_hrly_rate * total_hours) - (B_hrly_rate * (total_hours - B_break)) = 10 := 
by sorry

end cupcakes_difference_l60_60608


namespace dinner_seating_l60_60644

theorem dinner_seating (eight_people : Finset ℕ) (h_card : eight_people.card = 8) :
  ∃ S : Finset (Finset ℕ), S.card = 3360 ∧ ∀ s ∈ S, s.card = 6 := by
sorry

end dinner_seating_l60_60644


namespace no_cubic_term_l60_60743

noncomputable def p1 (a b k : ℝ) : ℝ := -2 * a * b + (1 / 3) * k * a^2 * b + 5 * b^2
noncomputable def p2 (a b : ℝ) : ℝ := b^2 + 3 * a^2 * b - 5 * a * b + 1
noncomputable def diff (a b k : ℝ) : ℝ := p1 a b k - p2 a b
noncomputable def cubic_term_coeff (a b k : ℝ) : ℝ := (1 / 3) * k - 3

theorem no_cubic_term (a b : ℝ) : ∀ k, (cubic_term_coeff a b k = 0) → k = 9 :=
by
  intro k h
  sorry

end no_cubic_term_l60_60743


namespace car_travel_time_difference_l60_60618

theorem car_travel_time_difference
  (speed_A : ℕ) (speed_B : ℕ) (distance : ℕ)
  (h_speed_A : speed_A = 60) (h_speed_B : speed_B = 45) (h_distance : distance = 360) :
  ((distance / speed_B) - (distance / speed_A)) * 60 = 120 := 
by
  rw [h_speed_A, h_speed_B, h_distance]
  sorry

end car_travel_time_difference_l60_60618


namespace count_roots_of_unity_and_quadratic_l60_60994

theorem count_roots_of_unity_and_quadratic (a b : ℤ) :
  ∃ (z : ℂ), (z^2 + (a:ℂ) * z + (b:ℂ) = 0) ∧ (z^nat_root_of_unity 8) := sorry

end count_roots_of_unity_and_quadratic_l60_60994


namespace number_of_smartphones_l60_60237

noncomputable def defective_smartphones : ℕ := 84
noncomputable def probability_both_defective : ℝ := 0.14470734744707348

theorem number_of_smartphones (N : ℕ) :
  (84 * 83) / (N * (N - 1) : ℝ) = 0.14470734744707348 → N = 221 :=
by
  intro h
  have h_eq : (84 : ℝ) * 83 = 7012 := by norm_num
  rw [h_eq] at h
  rw [mul_comm, ←mul_div_assoc] at h
  sorry

end number_of_smartphones_l60_60237


namespace work_completion_time_l60_60192

theorem work_completion_time (p q W : ℝ) (hp : W / 80) (hq : W / 48) (t_p : ℝ := 16) (t_q : ℝ) :
  p * t_p + (p + q) * t_q = W ∧ t_q = 24 → t_p + t_q = 40 :=
by
  sorry

end work_completion_time_l60_60192


namespace symmetric_point_is_one_pi_l60_60020

-- Define the concept of a point in polar coordinates
structure PolarPoint :=
  (r : ℝ)
  (θ : ℝ)

-- Define the condition of symmetry with respect to the pole
def symmetric_to_pole (p : PolarPoint) : PolarPoint :=
  ⟨p.r, p.θ + π⟩

-- Prove that the point symmetric to (1, 0) is (1, π)
theorem symmetric_point_is_one_pi :
  symmetric_to_pole ⟨1, 0⟩ = ⟨1, π⟩ :=
sorry

end symmetric_point_is_one_pi_l60_60020


namespace ratio_of_prices_l60_60252

-- Definitions for conditions
def CP : ℝ := 100
def profit_percent : ℝ := 42.5 / 100
def loss_percent : ℝ := 5 / 100

def SP1 : ℝ := CP + (profit_percent * CP)
def SP2 : ℝ := CP - (loss_percent * CP)

def Ratio : ℝ := SP2 / SP1

-- Theorem to prove the ratio is 38/57
theorem ratio_of_prices : Ratio = (38 / 57) := 
  by sorry

end ratio_of_prices_l60_60252


namespace compare_ex_ln_l60_60380

variable {x y : ℝ}

theorem compare_ex_ln (h : 3^(-x) - 3^(-y) < log 3 x - log 3 y) :
  (exp (x - y) > 1) ∧ (log (x - y + 1) > 0) :=
sorry

end compare_ex_ln_l60_60380


namespace amgm_inequality_abcd_l60_60683

-- Define the variables and their conditions
variables {a b c d : ℝ}
variable (ha : 0 < a)
variable (hb : 0 < b)
variable (hc : 0 < c)
variable (hd : 0 < d)

-- State the theorem
theorem amgm_inequality_abcd :
  a^4 * b + b^4 * c + c^4 * d + d^4 * a ≥ a * b * c * d * (a + b + c + d) :=
by
  sorry

end amgm_inequality_abcd_l60_60683


namespace line_BC_eq_perp_bisector_BC_eq_l60_60349

-- Define the points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := {x := -3, y := 0}
def B : Point := {x := 2, y := 1}
def C : Point := {x := -2, y := 3}

-- Define functions for the equations
def line_eq (p1 p2 : Point) (x y : ℝ) : Prop :=
  (x - p1.x) * (p2.y - p1.y) = (y - p1.y) * (p2.x - p1.x)

def perp_bisector (p1 p2 : Point) (x y : ℝ) : Prop :=
  (x - (p1.x + p2.x) / 2) * (p1.x - p2.x) + (y - (p1.y + p2.y) / 2) * (p1.y - p2.y) = 0

theorem line_BC_eq : ∀ x y : ℝ,
  line_eq B C x y ↔ x + 2 * y - 4 = 0 :=
by
  intros,
  sorry

theorem perp_bisector_BC_eq : ∀ x y : ℝ,
  perp_bisector B C x y ↔ 2 * x - y + 2 = 0 :=
by
  intros,
  sorry

end line_BC_eq_perp_bisector_BC_eq_l60_60349


namespace first_999_digits_zero_l60_60458

def a : ℝ := 6 + Real.sqrt 37
def b : ℝ := 6 - Real.sqrt 37
def N : ℤ := ⌊a^999 + b^999⌋

theorem first_999_digits_zero (h_bn : b < 1) (h_ni : (a^999 + b^999) = (N : ℝ)) (h_b_small : b^999 < 1 / (10 ^ 990)) :
  (a^999 - N).digits 10 = 0 := sorry

end first_999_digits_zero_l60_60458


namespace graph_of_equation_l60_60175

theorem graph_of_equation :
  ∀ (x y : ℝ), (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (x + y + 2 = 0 ∨ x+y = 0 ∨ x-y = 0) ∧ 
  ¬(∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
    (x₁ + y₁ + 2 = 0 ∧ x₁ + y₁ = 0) ∧ 
    (x₂ + y₂ + 2 = 0 ∧ x₂ = -x₂) ∧ 
    (x₃ + y₃ + 2 = 0 ∧ x₃ - y₃ = 0)) := 
sorry

end graph_of_equation_l60_60175


namespace percentage_increase_B_more_than_C_l60_60112

noncomputable def percentage_increase :=
  let C_m := 14000
  let A_annual := 470400
  let A_m := A_annual / 12
  let B_m := (2 / 5) * A_m
  ((B_m - C_m) / C_m) * 100

theorem percentage_increase_B_more_than_C : percentage_increase = 12 :=
  sorry

end percentage_increase_B_more_than_C_l60_60112


namespace find_a9_in_sequence_l60_60012

theorem find_a9_in_sequence :
    ∃ a : ℕ → ℕ, (monotonic_increasing a) ∧
        (a 1 = 1) ∧
        (a 2 = 2) ∧
        (∀ k : ℕ, (1 < k) → 
                  (1 + a k / a (k + 3)) * (1 + a (k + 1) / a (k + 2)) = 2) ∧
        (a 9 = 55) := 
sorry

end find_a9_in_sequence_l60_60012


namespace reciprocal_of_neg_two_l60_60895

noncomputable theory

def is_reciprocal (a x : ℝ) := a * x = 1

theorem reciprocal_of_neg_two : 
  ∃ x : ℝ, is_reciprocal (-2) x ∧ x = -1 / 2 := 
sorry

end reciprocal_of_neg_two_l60_60895


namespace odd_difference_even_odd_l60_60739

theorem odd_difference_even_odd (a b : ℤ) (ha : a % 2 = 0) (hb : b % 2 = 1) : (a - b) % 2 = 1 :=
sorry

end odd_difference_even_odd_l60_60739


namespace chord_length_from_line_and_circle_l60_60097

theorem chord_length_from_line_and_circle :
  let C := (0, 2) in
  let r := 2 in
  let line (k : ℝ) := λ x : ℝ, k * x + 2 in
  let circle (x y : ℝ) := x^2 + y^2 - 4 * y = 0 in
  ∀ k : ℝ, 
  (let l := line k in
   (∀ x y : ℝ, l x = y → x^2 + (l x)^2 - 4 * (l x) = 0) →
   ∃ x₁ x₂ : ℝ, 
   x₁ ≠ x₂ ∧ (∀ x : ℝ, l x^2 + y^2 - 4 * y = 0) ∧ 
   (dist (x₁, l x₁) (x₂, l x₂) = 4)) :=
begin
  sorry
end

end chord_length_from_line_and_circle_l60_60097


namespace cartesian_coordinates_problem_l60_60771

theorem cartesian_coordinates_problem
  (x1 y1 x2 y2 : ℕ)
  (h1 : x1 < y1)
  (h2 : x2 > y2)
  (h3 : x2 * y2 = x1 * y1 + 67)
  (h4 : 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2)
  : Nat.digits 10 (x1 * 1000 + y1 * 100 + x2 * 10 + y2) = [1, 9, 8, 5] :=
by
  sorry

end cartesian_coordinates_problem_l60_60771


namespace smallest_positive_period_and_max_value_monotonically_decreasing_intervals_l60_60712

noncomputable def f (x : ℝ) : ℝ :=
  cos (π / 3 + x) * cos (π / 3 - x) - sin x * cos x + 1 / 4

theorem smallest_positive_period_and_max_value :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∃ M, ∀ x, f x ≤ M ∧ M = sqrt 2 / 2) :=
sorry

theorem monotonically_decreasing_intervals :
  ∀ x ∈ Icc 0 π, (monotone_decreasing_in_interval x) :=
sorry

end smallest_positive_period_and_max_value_monotonically_decreasing_intervals_l60_60712


namespace bob_and_bill_same_class_probability_l60_60610

-- Definitions based on the conditions mentioned in the original problem
def total_people : ℕ := 32
def allowed_per_class : ℕ := 30
def number_chosen : ℕ := 2
def number_of_classes : ℕ := 2
def bob_and_bill_pair : ℕ := 1

-- Binomial coefficient calculation (32 choose 2)
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k
def total_ways := binomial_coefficient total_people number_chosen

-- Probability that Bob and Bill are chosen
def probability_chosen : ℚ := bob_and_bill_pair / total_ways

-- Probability that Bob and Bill are placed in the same class
def probability_same_class : ℚ := 1 / number_of_classes

-- Total combined probability
def combined_probability : ℚ := probability_chosen * probability_same_class

-- Statement of the theorem
theorem bob_and_bill_same_class_probability :
  combined_probability = 1 / 992 := 
sorry

end bob_and_bill_same_class_probability_l60_60610


namespace perfect_square_candidates_l60_60288

def is_six_digit_number (n : ℕ) : Prop := 
  100000 ≤ n ∧ n < 1000000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def reverse_digits (n : ℕ) : ℕ :=
  n.digits.reverse.foldl (λ a b, a * 10 + b) 0

def is_valid_solution (N : ℕ) : Prop :=
  is_six_digit_number N ∧ is_perfect_square N ∧ 
  is_perfect_square (reverse_digits N)

theorem perfect_square_candidates :
  ∀ N : ℕ, is_valid_solution N ↔ (N = 108900 ∨ N = 698896 ∨ N = 980100) :=
by
  sorry

end perfect_square_candidates_l60_60288


namespace smallest_int_m_ge_x1_l60_60442

noncomputable def f (t : ℝ) : ℝ := 2 * Real.log t - 1 - 2 / t

theorem smallest_int_m_ge_x1 (x1 : ℝ) (h1 : x1 < 0)
  (hx1_intersection : ∃ x, x < 0 ∧ -1 / x = Real.log x ∧ x1 = x) :
  ∃ m : ℤ, m ≥ x1 ∧ ∀ n : ℤ, n ≥ x1 → n ≥ m :=
begin
  sorry
end

end smallest_int_m_ge_x1_l60_60442


namespace intersecting_line_exists_l60_60397

theorem intersecting_line_exists (r : ℝ) 
  (inner_circles : ℕ → ℝ)
  (h_radius : r = 3)
  (h_sum_radii : (∑ i, inner_circles i) = 25) :
  ∃ (m : ℕ), ∃ (l : ℝ), (m ≥ 9 ∧ (∃ (c_list : list ℕ), (∀ i ∈ c_list, ∃ C_i, inner_circles i = C_i ∧ line l intersects C_i) ∧ list.length c_list = m) ):
  sorry

end intersecting_line_exists_l60_60397


namespace collinear_D_E_F_l60_60327

variables {A B C P D E F : Type}
variables [EuclideanGeometry A B C]
variables [PointOnCircumcircle P (triangle A B C)]
variables [PerpendicularIntersection P D (side B C)]
variables [PerpendicularIntersection P E (side C A)]
variables [PerpendicularIntersection P F (side A B)]
variables [EqualAngles (angle P D B) (angle P E C)]
variables [EqualAngles (angle P E C) (angle P F B)]

theorem collinear_D_E_F :
  Collinear D E F :=
by sorry

end collinear_D_E_F_l60_60327


namespace max_product_value_l60_60474

variable (h k : ℝ → ℝ)
variable (Hh : range h = Set.Icc (-3 : ℝ) 4)
variable (Hk : range k = Set.Icc (0 : ℝ) 2)

theorem max_product_value : ∃ c : ℝ, c = 8 ∧ ∀ x : ℝ, h x * k x ≤ c := 
sorry

end max_product_value_l60_60474


namespace monotonicity_intervals_max_min_values_l60_60711

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 + Real.sqrt 3 * Real.sin (2 * x) + 1

theorem monotonicity_intervals :
  ∀ k : ℤ, ∀ x : ℝ,
  (-Real.pi / 6 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 3 + k * Real.pi) →
  ∀ y : ℝ, (x ≤ y ∧ y ≤ Real.pi / 3 + k * Real.pi) → f x ≤ f y :=
sorry

theorem max_min_values :
  min_on (f '' Icc 0 (Real.pi / 2)) = 1 ∧ max_on (f '' Icc 0 (Real.pi / 2)) = 3 + Real.sqrt 3 :=
sorry

end monotonicity_intervals_max_min_values_l60_60711


namespace binomial_coefficient_example_l60_60623

theorem binomial_coefficient_example :
  2 * (Nat.choose 7 4) = 70 := 
sorry

end binomial_coefficient_example_l60_60623


namespace find_m_l60_60723

theorem find_m (m : ℕ) (h1 : (3 * m - 7) % 2 = 0) (h2 : 3 * m - 7 < 0) : m = 1 := 
by
  sorry

end find_m_l60_60723


namespace weight_of_rod_is_expected_l60_60385

-- Define the length and weight of a given segment of the rod
def length_segment : ℝ := 5
def weight_segment : ℝ := 19

-- Define the length of the rod we are interested in
def length_rod : ℝ := 11.25

-- Define the weight per meter
def weight_per_meter : ℝ := weight_segment / length_segment

-- Define the expected weight of the rod
def expected_weight : ℝ := 42.75

-- The theorem to prove the weight of the 11.25m rod is as expected
theorem weight_of_rod_is_expected : weight_per_meter * length_rod = expected_weight := by
  sorry

end weight_of_rod_is_expected_l60_60385


namespace distinct_patterns_4x4_three_squares_l60_60371

noncomputable def count_distinct_patterns : ℕ :=
  sorry

theorem distinct_patterns_4x4_three_squares :
  count_distinct_patterns = 12 :=
by sorry

end distinct_patterns_4x4_three_squares_l60_60371


namespace reciprocal_of_neg_two_l60_60884

theorem reciprocal_of_neg_two : ∃ x : ℝ, (-2) * x = 1 ∧ x = -1/2 :=
by
  use -1/2
  split
  · exact (by norm_num : (-2) * (-1/2) = 1)
  · rfl

end reciprocal_of_neg_two_l60_60884


namespace math_expression_equals_2014_l60_60843

-- Define the mapping for each letter
def M : Nat := 1
def A : Nat := 8
def T : Nat := 3
def I : Nat := 9
def K : Nat := 0 -- K corresponds to 'minus', to be used in expression

-- Verification that the expression evaluates to 2014
theorem math_expression_equals_2014 : (100 * M + 10 * A + T) + (1000 * (M + 10 * A + 100 * T) + 100 * M + 10 * A + T + I) - A = 2014 := by
  calc
    (100 * M + 10 * A + T) + (1000 * (M + 10 * A + 100 * T) + 100 * M + 10 * A + T + I) - A
        = (100 * 1 + 10 * 8 + 3) + (1000 * (1 + 10 * 8 + 100 * 3) + 100 * 1 + 10 * 8 + 3 + 9) - 8 : by rfl
    ... = 183 + 1839 - 8 : by rfl
    ... = 2014 : by rfl

end math_expression_equals_2014_l60_60843


namespace part_I_part_II_l60_60329

-- Define the sequence {a_n}
def a : ℕ → ℚ
| 0     := 2/3
| (n+1) := 2 * a n / (a n + 1)

-- Define the sequence {1/a_n - 1}
def b (n : ℕ) : ℚ := 1 / a n - 1

-- Part I: Prove that the sequence {1/a_n - 1} is a geometric progression
theorem part_I : ∃ q : ℚ, ∀ n : ℕ, b (n+1) = q * b n := by
  sorry

-- Define the sum of the first n terms of the sequence {n/a_n}
def S (n : ℕ) : ℚ := ∑ i in finset.range n, i / a i

-- Part II: Prove the sum formula
theorem part_II (n : ℕ) : 
  S n = (n^2 + n + 4) / 2 - (n + 2) / 2^n := by
  sorry

end part_I_part_II_l60_60329


namespace reciprocal_of_neg_two_l60_60903

variable a : ℤ

theorem reciprocal_of_neg_two (h : a = -2) : ∃ b : ℚ, a * b = 1 ∧ b = -1 / 2 := 
by
  exists (-1 / 2)
  rw [h]
  split
  · norm_num
  · refl

end reciprocal_of_neg_two_l60_60903


namespace min_value_of_y_l60_60867

noncomputable def y (x : ℝ) : ℝ :=
  2 * sin x * cos x - 2 * (sin x)^2

theorem min_value_of_y : ∃ (m : ℝ), m = -sqrt 2 - 1 ∧ ∀ x : ℝ, y x ≥ m :=
sorry

end min_value_of_y_l60_60867


namespace geometric_sequence_sum_l60_60696

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (n : ℕ) 
  (h_geom : ∀ k, a (k + 1) = a k * 0.5) 
  (h_a2 : a 2 = 2) 
  (h_a5 : a 5 = 1/4) : 
  a 1 * a 2 + ∑ i in finset.range n, a (i + 1) * a (i + 2) = (32/3) * (1 - (4 : ℝ) ^ (-n)) :=
sorry

end geometric_sequence_sum_l60_60696


namespace distance_to_city_hall_l60_60031

variable (d : ℝ) (t : ℝ)

-- Conditions
def condition1 : Prop := d = 45 * (t + 1.5)
def condition2 : Prop := d - 45 = 65 * (t - 1.25)
def condition3 : Prop := t > 0

theorem distance_to_city_hall
  (h1 : condition1 d t)
  (h2 : condition2 d t)
  (h3 : condition3 t)
  : d = 300 := by
  sorry

end distance_to_city_hall_l60_60031


namespace triangle_side_length_b_l60_60752

theorem triangle_side_length_b (a b c : ℝ) (A B C : ℝ)
  (hB : B = 30) 
  (h_area : 1/2 * a * c * Real.sin (B * Real.pi/180) = 3/2) 
  (h_sine : Real.sin (A * Real.pi/180) + Real.sin (C * Real.pi/180) = 2 * Real.sin (B * Real.pi/180)) :
  b = Real.sqrt 3 + 1 :=
by
  sorry

end triangle_side_length_b_l60_60752


namespace reciprocal_neg_half_l60_60125

theorem reciprocal_neg_half : (1 / (- (1 / 2) : ℚ) = -2) :=
by
  sorry

end reciprocal_neg_half_l60_60125


namespace basic_astrophysics_degrees_l60_60183

-- Define the percentages for various sectors
def microphotonics := 14
def home_electronics := 24
def food_additives := 15
def genetically_modified_microorganisms := 19
def industrial_lubricants := 8

-- The sum of the given percentages
def total_other_percentages := 
    microphotonics + home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants

-- The remaining percentage for basic astrophysics
def basic_astrophysics_percentage := 100 - total_other_percentages

-- Number of degrees in a full circle
def full_circle_degrees := 360

-- Calculate the degrees representing basic astrophysics
def degrees_for_basic_astrophysics := (basic_astrophysics_percentage * full_circle_degrees) / 100

-- Theorem statement
theorem basic_astrophysics_degrees : degrees_for_basic_astrophysics = 72 := 
by
  sorry

end basic_astrophysics_degrees_l60_60183


namespace reciprocal_of_neg_two_l60_60875

-- Define the condition of reciprocal
def is_reciprocal (y x : ℝ) : Prop := y * x = 1

-- Define the number in question
def y : ℝ := -2

-- Define the expected reciprocal
def x : ℝ := -1 / 2

-- State the theorem
theorem reciprocal_of_neg_two : is_reciprocal y x :=
by
  -- Proof can be provided here
  sorry

end reciprocal_of_neg_two_l60_60875


namespace Petya_wins_for_odd_n_l60_60963

theorem Petya_wins_for_odd_n (n : ℕ) (h : n ≥ 3 ∧ n % 2 = 1) : 
  Petya_wins_for_n n :=
by
  sorry

end Petya_wins_for_odd_n_l60_60963


namespace reciprocal_of_neg_two_l60_60898

noncomputable theory

def is_reciprocal (a x : ℝ) := a * x = 1

theorem reciprocal_of_neg_two : 
  ∃ x : ℝ, is_reciprocal (-2) x ∧ x = -1 / 2 := 
sorry

end reciprocal_of_neg_two_l60_60898


namespace largest_palindrome_sum_l60_60205

theorem largest_palindrome_sum : 
  ∃ n : ℕ, 
    (n ≥ 100 ∧ n < 1000) ∧ 
    (∀ i j k, n = 100 * i + 10 * j + i → i ≠ 0 ∧ 0 ≤ i ∧ i < 10 ∧ 0 ≤ j ∧ j < 10) ∧ 
    (n % 6 = 0) ∧ 
    (∀ m, (m ≥ 100 ∧ m < 1000) ∧ 
          (∀ i' j' k', m = 100 * i' + 10 * j' + i' → i' ≠ 0 ∧ 0 ≤ i' ∧ i' < 10 ∧ 0 ≤ j' ∧ j' < 10) ∧ 
          (m % 6 = 0) → n ≥ m) → 
  (n / 100 + (n / 10 % 10) + (n % 10) = 24) :=
by
  sorry

end largest_palindrome_sum_l60_60205


namespace percentage_of_oil_in_original_mixture_A_l60_60063

-- Define the percentage of oil in the original mixture A
variables (x : ℝ)

-- Define conditions given in the problem
def condition1 : Prop := x ≤ 100 ∧ x ≥ 0

def condition2 : Prop :=
  let weight_oil_in_mixture_A := x / 100 * 8 in
  let weight_materialB_in_mixture_A := (100 - x) / 100 * 8 in
  let new_weight_oil := weight_oil_in_mixture_A + 2 in
  let new_mixture_total_weight := 8 + 2 in
  let final_oil_weight := new_weight_oil + (x / 100 * 6) in
  let final_materialB_weight := weight_materialB_in_mixture_A + (100 - x) / 100 * 6 in
  let final_total_weight := new_mixture_total_weight + 6 in
  let materialB_percentage_in_final_mixture := 11.2 in
  ((100 - x) / 100) * 14 = 11.2

-- Prove that the percentage of oil in the original mixture A is 20%.
theorem percentage_of_oil_in_original_mixture_A : condition1 → condition2 → x = 20 :=
by
  intros
  sorry

end percentage_of_oil_in_original_mixture_A_l60_60063


namespace true_props_l60_60370

open Complex

-- Define the complex number z
def z : ℂ := 2 / (⟨-1,1⟩ : ℂ)

-- Define the propositions
def p1 : Prop := abs z = 2
def p2 : Prop := z^2 = 2 * I
def p3 : Prop := conj z = 1 + I
def p4 : Prop := z.im = -1

-- Prove that the true propositions are p2 and p4
theorem true_props : (p2 ∧ p4) ∧ ¬p1 ∧ ¬p3 := by {
    sorry
}

end true_props_l60_60370


namespace girls_not_consecutive_l60_60401

variable (boys girls selectBoys selectGirls notConsecutiveWays : ℕ)

axiom boys_count : boys = 4
axiom girls_count : girls = 3
axiom select_boys : selectBoys = 3
axiom select_girls : selectGirls = 2
axiom select_ways : (nat.choose boys selectBoys) * (nat.choose girls selectGirls) = 12
axiom not_consecutive_ways : (select_ways * 72) = 864

theorem girls_not_consecutive {boys girls selectBoys selectGirls select_ways : ℕ} :
  boys = 4 → girls = 3 → selectBoys = 3 → selectGirls = 2 →
  (nat.choose boys selectBoys) * (nat.choose girls selectGirls) = 12 →
  (select_ways * 72) = 864 :=
by
  intro h1 h2 h3 h4 h5
  exact not_consecutive_ways

end girls_not_consecutive_l60_60401


namespace tangent_line_eq_l60_60654

noncomputable def f (x : ℝ) := x / (2 * x - 1)

def tangentLineAtPoint (x : ℝ) : ℝ := -x + 2

theorem tangent_line_eq {x y : ℝ} (hxy : y = f 1) (f_deriv : deriv f 1 = -1) :
  y = 1 → tangentLineAtPoint x = -x + 2 :=
by
  intros
  sorry

end tangent_line_eq_l60_60654


namespace angle_between_slant_height_and_base_height_l60_60110

-- Define the regular triangular pyramid with base side length 2
structure RegularTriangularPyramid where
  base_side_length : ℝ
  lateral_edge_length : ℝ
  height_SO : ℝ
  apothem_SF : ℝ
  height_base_CD : ℝ

-- Example pyramid satisfying the conditions
def pyramid : RegularTriangularPyramid :=
  { base_side_length := 2,
    lateral_edge_length := 4,  -- twice the side of the base
    height_SO := sorry,  -- this should be computed from geometric constraints
    apothem_SF := sqrt 15,
    height_base_CD := sqrt 3
  }

-- Trustive definitions for the context
variable (p : RegularTriangularPyramid)

def OF := (p.base_side_length * sqrt 3) / 6

theorem angle_between_slant_height_and_base_height :
  let angle_SFK := real.arccos (sqrt 5 / 30)
  in angle_SFK = real.arccos (sqrt 5 / 30) :=
by
  sorry  -- Proof goes here

end angle_between_slant_height_and_base_height_l60_60110


namespace probability_six_largest_selected_l60_60570

theorem probability_six_largest_selected :
  let cards := {1, 2, 3, 4, 5, 6, 7}
  let events := { subset | subset ⊆ cards ∧ subset.card = 3 }
  let event_six_largest := { subset | subset ∈ events ∧ 6 ∈ subset ∧ ∀ x ∈ subset, x ≤ 6 }
  (event_six_largest.card : ℝ) / (events.card : ℝ) = 2 / 7 := 
sorry

end probability_six_largest_selected_l60_60570


namespace cone_lateral_area_l60_60479

noncomputable def lateral_area_of_cone (θ : ℝ) (r_base : ℝ) : ℝ :=
  if θ = 120 ∧ r_base = 2 then 
    12 * Real.pi 
  else 
    0 -- default case for the sake of definition, not used in our proof

theorem cone_lateral_area :
  lateral_area_of_cone 120 2 = 12 * Real.pi :=
by
  -- This is where the proof would go
  sorry

end cone_lateral_area_l60_60479


namespace circumcircle_radius_l60_60783

theorem circumcircle_radius (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] :
  ∠A = π / 3 ∧ AB = 2 ∧ AC = 3 → radius (circumcircle_of_triangle ABC) = sqrt 21 / 3 :=
by
  sorry

end circumcircle_radius_l60_60783


namespace number_of_men_in_first_group_l60_60976

-- Definitions based on the conditions provided
def work_done (men : ℕ) (days : ℕ) (work_rate : ℝ) : ℝ :=
  men * days * work_rate

-- Given conditions
def condition1 (M : ℕ) : Prop :=
  ∃ work_rate : ℝ, work_done M 12 work_rate = 66

def condition2 : Prop :=
  ∃ work_rate : ℝ, work_done 86 8 work_rate = 189.2

-- Proof goal
theorem number_of_men_in_first_group : 
  ∀ M : ℕ, condition1 M → condition2 → M = 57 := by
  sorry

end number_of_men_in_first_group_l60_60976


namespace train_journey_times_l60_60935

theorem train_journey_times (
  VA : ℝ
  VB : ℝ
  TA : ℝ
  TB : ℝ) 
  (h1 : (4/5) * TA + 1/2 = TA)
  (h2 : (3/4) * TB + 2/3 = TB) :
  TA = 2 ∧ TB = 2 :=
by
  sorry

end train_journey_times_l60_60935


namespace absolute_value_neg_2022_l60_60084

theorem absolute_value_neg_2022 : abs (-2022) = 2022 :=
by sorry

end absolute_value_neg_2022_l60_60084


namespace reciprocal_neg_half_l60_60124

theorem reciprocal_neg_half : (1 / (- (1 / 2) : ℚ) = -2) :=
by
  sorry

end reciprocal_neg_half_l60_60124


namespace quadrant_of_angle_l60_60316

theorem quadrant_of_angle (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
    (∃ (n : ℤ), α = n * π + π + real.pi) ∨ (-- sorry: proof required statement for third quadrant.

end quadrant_of_angle_l60_60316


namespace reciprocal_of_neg2_l60_60909

-- Define the number
def num : ℤ := -2

-- Define the reciprocal function
def reciprocal (x : ℤ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_neg2 : reciprocal num = -1 / 2 :=
by
  -- Proof is skipped
  sorry

end reciprocal_of_neg2_l60_60909


namespace mean_value_of_square_angles_l60_60529

-- Given definitions
def is_polygon (n : ℕ) : Prop := n ≥ 3
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def is_square (n : ℕ) : Prop := n = 4

-- The statement to prove
theorem mean_value_of_square_angles (n : ℕ) (h1 : is_square n) (h2 : is_polygon n) : sum_of_interior_angles n / n = 90 :=
by
  have h3 : sum_of_interior_angles 4 = 360 := by rfl  -- step identifying the interior angles sum for a square
  have h4 : 360 / 4 = 90 := by rfl                    -- step identifying the mean value of the angles
  exact h4

end mean_value_of_square_angles_l60_60529


namespace cupcake_difference_l60_60607

def betty_rate : ℕ := 10
def dora_rate : ℕ := 8
def total_hours : ℕ := 5
def betty_break_hours : ℕ := 2

theorem cupcake_difference :
  (dora_rate * total_hours) - (betty_rate * (total_hours - betty_break_hours)) = 10 :=
by
  sorry

end cupcake_difference_l60_60607


namespace determine_k_completed_square_l60_60001

theorem determine_k_completed_square (x : ℝ) :
  ∃ (a h k : ℝ), a * (x - h)^2 + k = x^2 - 7 * x ∧ k = -49/4 := sorry

end determine_k_completed_square_l60_60001


namespace fourth_power_of_x_l60_60659

-- Definition of the problem conditions
def x := sqrt (2 + sqrt (3 + sqrt 4))

-- Statement of the theorem to be proved
theorem fourth_power_of_x :
  (x^4 = 9 + 4 * sqrt 5) :=
sorry

end fourth_power_of_x_l60_60659


namespace seq_b_arithmetic_sum_seq_c_first_n_terms_l60_60724

-- Defining the sequence {a_n}
def seq_a : ℕ → ℕ
| 0     := 1  -- Note: Lean uses natural number indexing starting from 0
| (n+1) := 2 * (seq_a n) + 1

-- Defining the sequence {b_n} where b_n = log2(a_n + 1)
def seq_b (n : ℕ) : ℕ :=
  (Nat.log2 (seq_a n + 1))

-- Defining the sequence {c_n} where c_n = b_n / (a_n + 1)
def seq_c (n : ℕ) : ℚ :=
  (seq_b n : ℚ) / (seq_a n + 1 : ℚ)

-- The proof problem statements
theorem seq_b_arithmetic : ∀ n, seq_b n = n := sorry

theorem sum_seq_c_first_n_terms (n : ℕ) : ∑ i in Finset.range (n + 1), seq_c i = 2 - (2 + n) / 2^n := sorry

end seq_b_arithmetic_sum_seq_c_first_n_terms_l60_60724


namespace mean_temperature_l60_60496

theorem mean_temperature (temps : List ℝ)
  (h1 : temps = [82, 84, 83, 85, 86]) :
  (temps.sum / temps.length) = 84 :=
by
  sorry

end mean_temperature_l60_60496


namespace reciprocal_of_neg_two_l60_60902

variable a : ℤ

theorem reciprocal_of_neg_two (h : a = -2) : ∃ b : ℚ, a * b = 1 ∧ b = -1 / 2 := 
by
  exists (-1 / 2)
  rw [h]
  split
  · norm_num
  · refl

end reciprocal_of_neg_two_l60_60902


namespace additional_shaded_squares_needed_l60_60090

-- Define the primary structure and conditions
def Square (WXYZ : Type) := sorry

variables (WXYZ : Square Type) (total_small_squares : ℕ)
variable (unshaded_squares : ℕ)

-- Condition: The square WXYZ is divided into 100 small identical squares
axiom total_squares_eq_100 : total_small_squares = 100

-- Condition: The number of currently unshaded small squares
axiom unshaded_squares_eq_28 : unshaded_squares = 28

-- Lean theorem proving the given assertion
theorem additional_shaded_squares_needed :
  ∃ required_shaded_squares : ℕ, required_shaded_squares = 3 :=
by
  -- Initiate variables relevant to the problem conditions
  let shaded_squares := total_small_squares - unshaded_squares
  let required_shaded_squares := 0.75 * total_small_squares
  exact sorry

end additional_shaded_squares_needed_l60_60090


namespace number_of_beavers_l60_60271

-- Definitions of the problem conditions
def total_workers : Nat := 862
def number_of_spiders : Nat := 544

-- The statement we need to prove
theorem number_of_beavers : (total_workers - number_of_spiders) = 318 := 
by 
  sorry

end number_of_beavers_l60_60271


namespace calculations_collections_correct_l60_60822

open Nat

def count_possible_collections : Nat :=
  let vowels := [("A", 3), ("U", 1), ("I", 1), ("O", 1)]
  let consonants := [("C", 3), ("L", 3), ("T", 1), ("S", 1), ("N", 1)]

  -- Calculate the number of ways to choose the vowels
  let vowel_ways := (binom 4 1) + (binom 4 2) + (binom 4 3)

  -- Calculate the number of ways to choose the consonants
  let consonant_ways :=
    (binom 4 2) * 3 + (binom 4 2) * 3 + (binom 4 1) * 3 +
    (binom 3 1) * 3 + (binom 3 1) * 3 + (binom 2 0) * 1 +
    (binom 2 1) * 1 + (binom 1 0) * 1

  -- Total number of ways
  vowel_ways * consonant_ways

theorem calculations_collections_correct :
  count_possible_collections = 126 :=
by
  -- Placeholder for the actual proof
  sorry

end calculations_collections_correct_l60_60822


namespace first_two_decimal_places_of_pow_l60_60160

theorem first_two_decimal_places_of_pow (h : (2 / 3 : ℝ)^10 = x) : (floor (x * 100) / 100 = 0.01) :=
by
  sorry

end first_two_decimal_places_of_pow_l60_60160


namespace find_x_l60_60376

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n : ℕ, (2 * n + 1) * x^n

theorem find_x (x : ℝ) (H : series_sum x = 16) : 
  x = (33 - Real.sqrt 129) / 32 :=
by
  sorry

end find_x_l60_60376


namespace problem_statement_l60_60444

noncomputable def r_s_solution : ℝ × ℝ :=
  let f : ℝ → ℝ := λ x, (x - 7) * (3 * x + 11) - (x^2 - 16 * x + 55)
  let roots := (polynomial.roots (polynomial.of_fn [55, -16, 1])).val.to_list
  (roots.nth 0, roots.nth 1)

theorem problem_statement : let (r, s) := r_s_solution
                           in (r + 4) * (s + 4) = 25 := sorry

end problem_statement_l60_60444


namespace min_vector_sum_magnitude_l60_60383

variables (a b : ℝ × ℝ) -- assuming 2D vectors for simplicity
noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := 
  real.sqrt(v.1^2 + v.2^2)

theorem min_vector_sum_magnitude (ha : vector_magnitude a = 8) 
                                (hb : vector_magnitude b = 12) : 
  ∃ θ : ℝ, θ ∈ set.Icc 0 real.pi ∧ 
  vector_magnitude (a.1 + b.1, a.2 + b.2) = 4 :=
sorry

end min_vector_sum_magnitude_l60_60383


namespace problem1_problem2_problem3_problem4_l60_60751

variables (a b c R r s : ℝ) (A B C : ℝ)
noncomputable def semi_perimeter := (a + b + c) / 2

lemma half_angle_cos_C : cos^2 (C / 2) = (1 + cos C) / 2 := sorry
lemma half_angle_cos_B : cos^2 (B / 2) = (1 + cos B) / 2 := sorry
lemma half_angle_sin_C : sin^2 (C / 2) = 1 - cos^2 (C / 2) := sorry
lemma half_angle_sin_B : sin^2 (B / 2) = 1 - cos^2 (B / 2) := sorry
lemma circumradius_relation : a = 2 * R * sin A := sorry
lemma cot_A_half : cot (A / 2) = (s - a) / r := sorry
lemma cot_B_half : cot (B / 2) = (s - b) / r := sorry
lemma cot_C_half : cot (C / 2) = (s - c) / r := sorry

theorem problem1 : s = b * cos^2 (C / 2) + c * cos^2 (B / 2) := 
begin
  sorry
end

theorem problem2 : s = a + b * sin^2 (C / 2) + c * sin^2 (B / 2) := 
begin
  sorry
end

theorem problem3 : s = 4 * R * cos (A / 2) * cos (B / 2) * cos (C / 2) := 
begin
  sorry
end

theorem problem4 : s = r * (cot (A / 2) + cot (B / 2) + cot (C / 2)) := 
begin
  sorry
end

end problem1_problem2_problem3_problem4_l60_60751


namespace math_exam_problem_l60_60402

open ProbabilityTheory

noncomputable def number_of_students_between_100_and_110 (a : ℝ) (h : a > 0) : ℝ :=
  let n := 1000
  let μ := 100
  let σ := a
  let Z := NormalDist.mk μ σ
  let p := cdf Z 90
  if hp : p = 0.1 then
    n * (cdf Z 110 - cdf Z 100)
  else 0 -- this should never be triggered given the condition

theorem math_exam_problem : 
  ∀ a : ℝ, a > 0 →
  let n := 1000
  let μ := 100
  let Z := NormalDist.mk μ a
  (cdf Z 90 = 0.1) →
  number_of_students_between_100_and_110 a (by assumption) = 400 :=
by
  intros a ha n μ Z h_cdf
  rw [number_of_students_between_100_and_110]
  rw [h_cdf]
  sorry -- proof to be provided

end math_exam_problem_l60_60402


namespace correct_oblique_projection_conclusions_l60_60153

def oblique_projection (shape : Type) : Type := shape

theorem correct_oblique_projection_conclusions :
  (oblique_projection Triangle = Triangle) ∧
  (oblique_projection Parallelogram = Parallelogram) ↔
  (oblique_projection Square ≠ Square) ∧
  (oblique_projection Rhombus ≠ Rhombus) :=
by
  sorry

end correct_oblique_projection_conclusions_l60_60153


namespace cone_volume_correct_l60_60760

noncomputable def volume_of_cone_tangent_to_spheres_in_cylinder
  (radius height : ℝ)
  (h_radius : radius = 1)
  (h_height : height = 12 / (3 + 2 * sqrt 3))
  : ℝ :=
  (1 / 9) * π

theorem cone_volume_correct :
  ∀ (radius height : ℝ),
    radius = 1 →
    height = 12 / (3 + 2 * sqrt 3) →
    volume_of_cone_tangent_to_spheres_in_cylinder radius height = (1 / 9) * π :=
by
  intros radius height h_radius h_height
  simp [volume_of_cone_tangent_to_spheres_in_cylinder, h_radius, h_height]
  sorry

end cone_volume_correct_l60_60760


namespace m_square_if_divisible_l60_60092

theorem m_square_if_divisible (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : mn | m^2 + n^2 + m) : ∃ k, m = k^2 := by
  sorry

end m_square_if_divisible_l60_60092


namespace prob_all_fail_prob_at_least_one_pass_l60_60140

def prob_pass := 1 / 2
def prob_fail := 1 - prob_pass

def indep (A B C : Prop) : Prop := true -- Usually we prove independence in a detailed manner, but let's assume it's given as true.

theorem prob_all_fail (A B C : Prop) (hA : prob_pass = 1 / 2) (hB : prob_pass = 1 / 2) (hC : prob_pass = 1 / 2) 
  (indepABC : indep A B C) : (prob_fail * prob_fail * prob_fail) = 1 / 8 :=
by
  sorry

theorem prob_at_least_one_pass (A B C : Prop) (hA : prob_pass = 1 / 2) (hB : prob_pass = 1 / 2) (hC : prob_pass = 1 / 2) 
  (indepABC : indep A B C) : 1 - (prob_fail * prob_fail * prob_fail) = 7 / 8 :=
by
  sorry

end prob_all_fail_prob_at_least_one_pass_l60_60140


namespace ten_row_geometric_figure_has_286_pieces_l60_60594

noncomputable def rods (rows : ℕ) : ℕ := 3 * rows * (rows + 1) / 2
noncomputable def connectors (rows : ℕ) : ℕ := (rows +1) * (rows + 2) / 2
noncomputable def squares (rows : ℕ) : ℕ := rows * (rows + 1) / 2

theorem ten_row_geometric_figure_has_286_pieces :
    rods 10 + connectors 10 + squares 10 = 286 := by
  sorry

end ten_row_geometric_figure_has_286_pieces_l60_60594


namespace max_visible_sum_of_stacked_cubes_l60_60315

theorem max_visible_sum_of_stacked_cubes :
  let cube1 := {1, 3, 9, 27, 81, 243}
  let cube2 := {2, 6, 18, 54, 162, 486}
  let cube3 := {4, 12, 36, 108, 324, 972}
  let cube4 := {5, 15, 45, 135, 405, 1215}
  in ∃ visible_faces : finset ℕ,
    visible_faces.card = 19 ∧
    visible_faces.sum = 4310 :=
by 
  let cube1 := {1, 3, 9, 27, 81, 243}
  let cube2 := {2, 6, 18, 54, 162, 486}
  let cube3 := {4, 12, 36, 108, 324, 972}
  let cube4 := {5, 15, 45, 135, 405, 1215}
  let visible_faces := {243, 81, 27, 9, 3, 486, 162, 54, 6, 972, 324, 108, 12, 1215, 405, 135, 45, 15}
  use visible_faces
  sorry

end max_visible_sum_of_stacked_cubes_l60_60315


namespace movie_marathon_first_movie_length_l60_60927

theorem movie_marathon_first_movie_length 
  (x : ℝ)
  (h2 : 1.5 * x = second_movie)
  (h3 : second_movie + x - 1 = last_movie)
  (h4 : (x + second_movie + last_movie) = 9)
  (h5 : last_movie = 2.5 * x - 1) :
  x = 2 :=
by
  sorry

end movie_marathon_first_movie_length_l60_60927


namespace sum_of_first_six_cubes_l60_60820

theorem sum_of_first_six_cubes :
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2) :=
begin
  -- Given conditions
  have eq1 : 1^3 + 2^3 = 3^2 := by sorry,
  have eq2 : 1^3 + 2^3 + 3^3 = 6^2 := by sorry,
  have eq3 : 1^3 + 2^3 + 3^3 + 4^3 = 10^2 := by sorry,
  
  -- Proof of the main statement
  -- (Detailed steps go here; use the pattern observed in conditions)
  sorry
end

end sum_of_first_six_cubes_l60_60820


namespace john_spends_on_burgers_l60_60790

noncomputable def cost_of_burgers 
  (burritos : ℕ) 
  (calories_per_burrito : ℕ) 
  (cost_of_burritos : ℕ) 
  (burgers : ℕ) 
  (calories_per_burger : ℕ) 
  (additional_calories_per_dollar : ℕ) : ℚ :=
  let total_calories_burritos := (burritos * calories_per_burrito : ℕ)
  let cost_per_calorie_burritos := (cost_of_burritos : ℚ) / (total_calories_burritos : ℚ)
  let cost_per_calorie_burgers := cost_per_calorie_burritos + (additional_calories_per_dollar : ℚ) / 1000
  let total_calories_burgers := (burgers * calories_per_burger : ℕ)
  (total_calories_burgers : ℚ) / cost_per_calorie_burgers

theorem john_spends_on_burgers
  (h1: 10) (h2: 120) (h3: 6) (h4: 5) (h5: 400) (h6: 50) :
  cost_of_burgers h1 h2 h3 h4 h5 h6 = 2000 / 0.055 := by
  sorry

end john_spends_on_burgers_l60_60790


namespace triangle_ratios_l60_60691

open Triangle

def is_inside (P : Point) (ABC : Triangle) : Prop :=
  -- Define what it means for a point P to be inside a triangle ABC
  sorry

def intersects (l : Line) (side : Segment) (D : Point) : Prop :=
  -- Define that line l intersects side of the triangle at point D
  sorry

def parallel_through_point (l : Line) (P' : Point) (l' : Line) : Prop :=
  -- Define that line l' is parallel to line l and passes through point P'
  sorry

theorem triangle_ratios
  (A B C P D E F P' D' E' F' : Point)
  (ABC : Triangle)
  (h1: ABC.contains A B C)
  (h2: is_inside P ABC)
  (h3: intersects (Line.mk A P) (Segment.mk B C) D)
  (h4: intersects (Line.mk B P) (Segment.mk C A) E)
  (h5: intersects (Line.mk C P) (Segment.mk A B) F)
  (h6: perimeter_contains_point (Triangle.mk D E F) P')
  (h7: parallel_through_point (Line.mk P D) P' (Line.mk P' D'))
  (h8: parallel_through_point (Line.mk P E) P' (Line.mk P' E'))
  (h9: parallel_through_point (Line.mk P F) P' (Line.mk P' F')) :
  (∃ i j k : ℝ, {i=j+k ∨ j=i+k ∨ k=i+j} ∧ {i = (P'.dist D') / (P.dist D)
                                    ∧ j = (P'.dist E') / (P.dist E)
                                    ∧ k = (P'.dist F') / (P.dist F)}) :=
sorry

end triangle_ratios_l60_60691


namespace sqrt_four_squared_l60_60173

theorem sqrt_four_squared : (Real.sqrt 4) ^ 2 = 4 :=
  by
    sorry

end sqrt_four_squared_l60_60173


namespace count_two_digit_numbers_with_triangular_digit_sum_l60_60732

def is_triangular (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * (m + 1) / 2 

def digits_sum (n : ℕ) : ℕ :=
  n / 10 + n % 10

def two_digit_numbers_with_triangular_digit_sum : ℕ :=
  Finset.card $ (Finset.range 90).filter (λ n, 10 ≤ n + 10 ∧ is_triangular (digits_sum (n + 10)))

theorem count_two_digit_numbers_with_triangular_digit_sum :
  two_digit_numbers_with_triangular_digit_sum = 23 :=
sorry

end count_two_digit_numbers_with_triangular_digit_sum_l60_60732


namespace investor_purchase_price_l60_60950

def dividend_rate : ℝ := 15.5
def face_value : ℝ := 50
def roi : ℝ := 25
def dividend_per_share : ℝ := (dividend_rate / 100) * face_value
def purchase_price : ℝ := 31

theorem investor_purchase_price :
  (dividend_per_share / purchase_price) * 100 = roi :=
by
  sorry

end investor_purchase_price_l60_60950


namespace find_k_l60_60922

theorem find_k (x y k : ℝ) :
  (x - y = k + 2) →
  (x + 3y = k) →
  (x + y = 2) →
  k = 1 :=
by
  intros h1 h2 h3
  -- The proof steps would go here
  sorry

end find_k_l60_60922


namespace program_output_is_201_l60_60741

theorem program_output_is_201 :
  ∃ x S n, x = 3 + 2 * n ∧ S = n^2 + 4 * n ∧ S ≥ 10000 ∧ x = 201 :=
by
  sorry

end program_output_is_201_l60_60741


namespace contradictory_knight_liar_statement_l60_60257

-- Definitions
def Knight (A : Prop) := A
def Liar (A : Prop) := ¬A
def P := Liar (P)
def Q := (2 + 2 = 5)
def A_statement := P ∨ Q

-- Main theorem 
theorem contradictory_knight_liar_statement : ¬(Knight A_statement ∨ Liar A_statement) := by
  sorry

end contradictory_knight_liar_statement_l60_60257


namespace sum_of_altitudes_eq_l60_60493

noncomputable theory

def line_eq (x y: ℝ) := 15 * x + 8 * y = 120

def triangle_vertices : set (ℝ × ℝ) :=
  { v | v = (0, 0) ∨ v = (8, 0) ∨ v = (0, 15) }

def altitude_sum : ℝ :=
  8 + 15 + 120 / Real.sqrt(15^2 + 8^2)

theorem sum_of_altitudes_eq : altitude_sum = 511 / 17 := by
  sorry

end sum_of_altitudes_eq_l60_60493


namespace isosceles_trapezoid_with_inscribed_circle_area_is_20_l60_60296

def isosceles_trapezoid_area (a b c1 c2 h : ℕ) : ℕ :=
  (a + b) * h / 2

theorem isosceles_trapezoid_with_inscribed_circle_area_is_20
  (a b c h : ℕ)
  (ha : a = 2)
  (hb : b = 8)
  (hc : a + b = 2 * c)
  (hh : h ^ 2 = c ^ 2 - ((b - a) / 2) ^ 2) :
  isosceles_trapezoid_area a b c c h = 20 := 
by {
  sorry
}

end isosceles_trapezoid_with_inscribed_circle_area_is_20_l60_60296


namespace reciprocal_of_neg2_l60_60915

-- Define the number
def num : ℤ := -2

-- Define the reciprocal function
def reciprocal (x : ℤ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_neg2 : reciprocal num = -1 / 2 :=
by
  -- Proof is skipped
  sorry

end reciprocal_of_neg2_l60_60915


namespace list_size_l60_60573

theorem list_size (numbers : List ℝ) (n : ℝ) (h_n_in_list : n ∈ numbers) :
  (n = 5 * ((numbers.erase n).sum / (numbers.length - 1))) →
  (n = 0.2 * (numbers.sum)) →
  numbers.length = 21 := 
by
  sorry

end list_size_l60_60573


namespace find_c_of_parabola_l60_60096

theorem find_c_of_parabola 
  (a b c : ℝ)
  (h_eq : ∀ y, -3 = a * (y - 1)^2 + b * (y - 1) - 3)
  (h1 : -1 = a * (3 - 1)^2 + b * (3 - 1) - 3) :
  c = -5/2 := by
  sorry

end find_c_of_parabola_l60_60096


namespace quadrilateral_BD_length_l60_60410

theorem quadrilateral_BD_length :
  ∃ (BD : ℕ), 
    (ABCD.exists
      ∧ AB = 5
      ∧ BC = 17
      ∧ CD = 5
      ∧ DA = 9
      ∧ BD = 13) :=
sorry

end quadrilateral_BD_length_l60_60410


namespace gcd_lcm_product_l60_60491

theorem gcd_lcm_product (a b : ℤ) (h1 : Int.gcd a b = 8) (h2 : Int.lcm a b = 24) : a * b = 192 := by
  sorry

end gcd_lcm_product_l60_60491


namespace metres_sold_is_200_l60_60592

-- Define the conditions
def loss_per_metre : ℕ := 6
def cost_price_per_metre : ℕ := 66
def total_selling_price : ℕ := 12000

-- Define the selling price per metre based on the conditions
def selling_price_per_metre := cost_price_per_metre - loss_per_metre

-- Define the number of metres sold
def metres_sold : ℕ := total_selling_price / selling_price_per_metre

-- Proof statement: Check if the number of metres sold equals 200
theorem metres_sold_is_200 : metres_sold = 200 :=
  by
  sorry

end metres_sold_is_200_l60_60592


namespace bicycle_price_l60_60815

theorem bicycle_price (P : ℝ) (h : 0.2 * P = 200) : P = 1000 := 
by
  sorry

end bicycle_price_l60_60815


namespace speaking_orders_count_l60_60859

open_locale big_operators

/-- The total number of different speaking orders for 4 speakers selected from 
8 candidates (including A and B) with the constraints that at least one of A 
and B must participate, and if both participate, exactly one person must speak 
between them, is 1080. -/
theorem speaking_orders_count :
  ∃ (A B : Type) (candidate_set : Finset A)
    (condition1 : candidate_set.card = 8) -- 8 candidates
    (condition2 : ∃ (subset : Finset A), subset.card = 4 ∧ subset ⊆ candidate_set), -- 4 speakers selected
    let participating_speakers_count (subset : Finset A) : Prop := 
      (A ∈ subset ∨ B ∈ subset) -- at least one of A or B participates
      ∧ (A ∈ subset ∧ B ∈ subset → (∃ x ∈ subset, (x ≠ A ∧ x ≠ B) ∧ ∀ y, y ∈ subset ∧ y ≠ A → y ≠ x ∧ y ≠ B)), -- exactly one person speaks between A and B
  true := sorry

end speaking_orders_count_l60_60859


namespace most_followers_is_sarah_l60_60854

def initial_followers_susy : ℕ := 100
def initial_followers_sarah : ℕ := 50

def susy_week1_new : ℕ := 40
def susy_week2_new := susy_week1_new / 2
def susy_week3_new := susy_week2_new / 2
def susy_total_new := susy_week1_new + susy_week2_new + susy_week3_new
def susy_final_followers := initial_followers_susy + susy_total_new

def sarah_week1_new : ℕ := 90
def sarah_week2_new := sarah_week1_new / 3
def sarah_week3_new := sarah_week2_new / 3
def sarah_total_new := sarah_week1_new + sarah_week2_new + sarah_week3_new
def sarah_final_followers := initial_followers_sarah + sarah_total_new

theorem most_followers_is_sarah : 
    sarah_final_followers ≥ susy_final_followers := by
  sorry

end most_followers_is_sarah_l60_60854


namespace abs_three_minus_pi_l60_60965

theorem abs_three_minus_pi : |(3 : ℝ) - real.pi| = real.pi - 3 := by
  sorry

end abs_three_minus_pi_l60_60965


namespace correct_answer_l60_60161

-- Define the expressions as functions on real numbers
def exprA (x : ℝ) := sqrt (-x - 2)
def exprB (x : ℝ) := sqrt x
def exprC (x : ℝ) := sqrt (x^2 + 2)
def exprD (x : ℝ) := sqrt (x^2 - 2)

-- Define the proposition stating exprC is always a quadratic radical
def isQuadraticRadical (f : ℝ → ℝ) : Prop := 
  ∀ x, 0 ≤ f x

-- Prove exprC is always a quadratic radical
theorem correct_answer : isQuadraticRadical exprC :=
by
  -- To be proved
  sorry

end correct_answer_l60_60161


namespace concurrency_of_lines_l60_60109

noncomputable def incircle_of_triangle (A B C : Point) : Circle :=
  sorry -- Definition of the incircle of a triangle

noncomputable def touches (c : Circle) (a b : Point) : Point :=
  sorry -- Definition of the point where a circle touches a side of the triangle

noncomputable def area (A B C : Point) : ℝ :=
  sorry -- Definition of the area of a triangle

noncomputable def concurrent (l1 l2 l3 : Line) : Prop :=
  sorry -- Definition of concurrency of lines

theorem concurrency_of_lines
  (A_0 B_0 C_0 A B C A_1 B_1 C_1 I : Point)
  (incircle_A0B0C0 : Circle)
  (incircle_ABC : Circle)
  (h1 : incircle_A0B0C0 = incircle_of_triangle A_0 B_0 C_0)
  (h2 : A = touches incircle_A0B0C0 B_0 C_0)
  (h3 : B = touches incircle_A0B0C0 C_0 A_0)
  (h4 : C = touches incircle_A0B0C0 A_0 B_0)
  (h5 : incircle_ABC = incircle_of_triangle A B C)
  (h6 : A_1 = touches incircle_ABC B C)
  (h7 : B_1 = touches incircle_ABC C A)
  (h8 : C_1 = touches incircle_ABC A B)
  (h9 : I = incenter A B C)
  (h10 : area A B C = 2 * area A_1 B_1 C_1) :
  concurrent (line_through A A_0) (line_through B B_0) (line_through I C_1) :=
sorry

end concurrency_of_lines_l60_60109


namespace circle_radius_from_tangents_l60_60678

theorem circle_radius_from_tangents (A : Point) (B C O : Point)
  (h_tangent_AB : tangent_from_to A B O)
  (h_tangent_AC : tangent_from_to A C O)
  (h_AB_len : distance A B = 12)
  (h_AC_len : distance A C = 12)
  (h_BC_len : distance B C = 14.4) :
  distance O B = 9 := 
sorry

end circle_radius_from_tangents_l60_60678


namespace even_function_coeff_l60_60388

theorem even_function_coeff (a : ℝ) (h : ∀ x : ℝ, (a-2)*x^2 + (a-1)*x + 3 = (a-2)*(-x)^2 + (a-1)*(-x) + 3) : a = 1 :=
by {
  -- Proof here
  sorry
}

end even_function_coeff_l60_60388


namespace num_of_nines_1_to_80_l60_60239

theorem num_of_nines_1_to_80 : (Finset.range (80 + 1)).filter (λ n, n % 10 = 9) .card = 8 := by
  sorry

end num_of_nines_1_to_80_l60_60239


namespace sequence_formula_sum_l60_60776

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → (∀ i : ℕ, 0 < i ∧ i ≤ k → a i = 2 * k)

theorem sequence_formula_sum (a : ℕ → ℕ) (b c d : ℤ) :
  sequence a →
  (∀ n : ℕ, n > 0 → a n = b * int.floor (real.sqrt (n + ↑c)) + d) →
  a 1 = 2 →
  b + c + d = 2 :=
by
  sorry

end sequence_formula_sum_l60_60776


namespace cos_pi_plus_alpha_l60_60342

-- Define the angle α and conditions given
variable (α : Real) (h1 : 0 < α) (h2 : α < π/2)

-- Given condition sine of α
variable (h3 : Real.sin α = 4/5)

-- Define the cosine identity to prove the assertion
theorem cos_pi_plus_alpha (h1 : 0 < α) (h2 : α < π/2) (h3 : Real.sin α = 4/5) :
  Real.cos (π + α) = -3/5 :=
sorry

end cos_pi_plus_alpha_l60_60342


namespace num_valid_n_l60_60302

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (Nat.succ n') => Nat.succ n' * factorial n'

def divisible (a b : ℕ) : Prop := b ∣ a

theorem num_valid_n (N : ℕ) :
  N ≤ 30 → 
  ¬ (∃ k, k + 1 ≤ 31 ∧ k + 1 > 1 ∧ (Prime (k + 1)) ∧ ¬ divisible (2 * factorial (k - 1)) (k + 1)) →
  ∃ m : ℕ, m = 20 :=
by
  sorry

end num_valid_n_l60_60302


namespace num_divisors_220_to_6_l60_60374

theorem num_divisors_220_to_6 :
  let n := 220^6
  let perfect_squares := { d | d | n ∧ (∃ a b c, d = 2^(2*a) * 5^(2*b) * 11^(2*c) ∧ 0 ≤ a ∧ 2*a ≤ 12 ∧ 0 ≤ b ∧ 2*b ≤ 6 ∧ 0 ≤ c ∧ 2*c ≤ 6) }
  let perfect_cubes := { d | d | n ∧ (∃ a b c, d = 2^(3*a) * 5^(3*b) * 11^(3*c) ∧ 0 ≤ a ∧ 3*a ≤ 12 ∧ 0 ≤ b ∧ 3*b ≤ 6 ∧ 0 ≤ c ∧ 3*c ≤ 6) }
  let perfect_sixth_powers := { d | d | n ∧ (∃ a b c, d = 2^(6*a) * 5^(6*b) * 11^(6*c) ∧ 0 ≤ a ∧ 6*a ≤ 12 ∧ 0 ≤ b ∧ 6*b ≤ 6 ∧ 0 ≤ c ∧ 6*c ≤ 6) }
  let num_perfect_squares := finset.card perfect_squares
  let num_perfect_cubes := finset.card perfect_cubes
  let num_perfect_sixth_powers := finset.card perfect_sixth_powers
  let num_desired_divisors := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers
  num_desired_divisors = 145 := 
sorry

end num_divisors_220_to_6_l60_60374


namespace graph_passes_through_0_1_l60_60488

theorem graph_passes_through_0_1 (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (0, 1) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^x) } :=
sorry

end graph_passes_through_0_1_l60_60488


namespace sum_distinct_products_641G5073H6_div_by_72_l60_60869

theorem sum_distinct_products_641G5073H6_div_by_72 :
  let is_single_digit (n : ℕ) := n < 10
  let is_divisible_by (n d : ℕ) := n % d = 0
  let satisfies_conditions (G H : ℕ) :=
    is_single_digit G ∧ is_single_digit H ∧
    let num := 6 * 10^9 + 4 * 10^8 + 1 * 10^7 + G * 10^6 + 5 * 10^5 + 0 * 10^4 + 7 * 10^3 + 3 * 10^2 + H * 10 + 6 in
    is_divisible_by num 72
  in (finset.univ.filter (λ GH_pair : ℕ × ℕ, let (G, H) := GH_pair in satisfies_conditions G H)
       .image (λ GH_pair, let (G, H) := GH_pair in G * H)).sum = 42 :=
sorry

end sum_distinct_products_641G5073H6_div_by_72_l60_60869


namespace length_of_train_l60_60552

theorem length_of_train (speed_kmh : ℝ) (cross_time_s : ℝ) (length_m : ℝ) 
  (h_speed : speed_kmh = 90) (h_time : cross_time_s = 15) : length_m = 375 :=
by 
  -- define the speed in m/s
  let speed_ms := speed_kmh * (5 / 18)
  -- the formula for distance
  let length := speed_ms * cross_time_s
  -- simplify the expression
  have h1 : speed_ms = 25 := by calc
     speed_kmh * (5 / 18) = 90 * (5 / 18) : by rw h_speed 
                     ... = 25 : by norm_num
  have h2 : length = 375 := by calc
     speed_ms * cross_time_s = 25 * cross_time_s : by rw h1 
                       ... = 25 * 15 : by rw h_time 
                       ... = 375 : by norm_num
  show length_m = 375, from eq.trans (by rw <-h2; exact eq.refl _) (eq.refl _)

end length_of_train_l60_60552


namespace min_num_triangles_to_cover_l60_60530

-- Define the side lengths of the triangles
def small_triangle_side : ℕ := 2
def large_triangle_side : ℕ := 12

-- The proof problem: Prove that the minimum number of small equilateral triangles to cover the large equilateral triangle is 36
theorem min_num_triangles_to_cover (s_small : ℕ) (s_large : ℕ) (h_small : s_small = small_triangle_side) (h_large : s_large = large_triangle_side) :
  s_small * s_small = (1/6)^2 * s_large * s_large -> 36 = (s_large / s_small)^2 :=
by {
  intros, 
  sorry
}

end min_num_triangles_to_cover_l60_60530


namespace find_p_l60_60054

open Set

theorem find_p
  (U : Set ℕ) (M : Set ℤ) (p : ℤ)
  (hU : U = {1, 2, 3, 4})
  (hM_def : M = {x | x^2 - 5 * x + p = 0})
  (hC_U_M : (U \ M) = {2, 3}) : p = 4 := sorry

end find_p_l60_60054


namespace most_compliant_expression_l60_60945

-- Define the expressions as algebraic terms.
def OptionA : String := "1(1/2)a"
def OptionB : String := "b/a"
def OptionC : String := "3a-1 个"
def OptionD : String := "a * 3"

-- Define a property that represents compliance with standard algebraic notation.
def is_compliant (expr : String) : Prop :=
  expr = OptionB

-- The theorem to prove.
theorem most_compliant_expression :
  is_compliant OptionB :=
by
  sorry

end most_compliant_expression_l60_60945


namespace largest_sum_l60_60944

theorem largest_sum :
  let a := 1 / 4 + 1 / 5,
      b := 1 / 4 + 1 / 6,
      c := 1 / 4 + 1 / 3,
      d := 1 / 4 + 1 / 8,
      e := 1 / 4 + 1 / 7 in
  c > a ∧ c > b ∧ c > d ∧ c > e :=
by
  -- let a := 1 / 4 + 1 / 5
  -- let b := 1 / 4 + 1 / 6
  -- let c := 1 / 4 + 1 / 3
  -- let d := 1 / 4 + 1 / 8
  -- let e := 1 / 4 + 1 / 7
  -- show c > a ∧ c > b ∧ c > d ∧ c > e
  sorry

end largest_sum_l60_60944


namespace combination_lock_code_l60_60481

theorem combination_lock_code :
  ∀ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ (x + y + x * y = 10 * x + y) →
  10 * x + y = 19 ∨ 10 * x + y = 29 ∨ 10 * x + y = 39 ∨ 10 * x + y = 49 ∨
  10 * x + y = 59 ∨ 10 * x + y = 69 ∨ 10 * x + y = 79 ∨ 10 * x + y = 89 ∨
  10 * x + y = 99 :=
by
  sorry

end combination_lock_code_l60_60481


namespace min_blue_beads_78_l60_60987

noncomputable def min_blue_beads_in_necklace : ℕ :=
sorry

theorem min_blue_beads_78 (R B : ℕ) (h_red : R = 100)
(h_condition : ∀ s : finset ℕ, s.card = 16 → 
(s.filter (λ n, n < 10)).card = 10 → 
(s.filter (λ n, n >= 10)).card ≤ 9 →
(s.card - (s.filter (λ n, n < 10)).card) ≥ 7) :
  min_blue_beads_in_necklace = 78 :=
by sorry

end min_blue_beads_78_l60_60987


namespace correct_square_root_operation_l60_60169

theorem correct_square_root_operation : 
  (sqrt 4)^2 = 4 ∧ sqrt 4 ≠ 2 ∨ -2 ∧ sqrt ((-4)^2) ≠ -4 ∧ (-sqrt 4)^2 ≠ -4 :=
by
  have a : (sqrt 4)^2 = 4, from sorry,
  have b : sqrt 4 ≠ 2 ∨ -2, from sorry,
  have c : sqrt ((-4)^2) ≠ -4, from sorry,
  have d : (-sqrt 4)^2 ≠ -4, from sorry,
  exact ⟨a, b, c, d⟩

end correct_square_root_operation_l60_60169


namespace number_of_tiles_per_row_l60_60858

-- Definitions of conditions
def area : ℝ := 320
def length : ℝ := 16
def tile_size : ℝ := 1

-- Theorem statement
theorem number_of_tiles_per_row : (area / length) / tile_size = 20 := by
  sorry

end number_of_tiles_per_row_l60_60858


namespace linear_function_inequality_inverse_function_inequality_l60_60304

-- Part (1) statement in Lean 4
theorem linear_function_inequality (x1 x2 : ℝ) (h : x1 < x2) : 
  let y1 := -2 * x1 + 1;
      y2 := -2 * x2 + 1
  in y1 > y2 :=
by
  sorry

-- Part (2) statement in Lean 4
theorem inverse_function_inequality (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : x2 < 0) :
  let y1 := 2 / x1;
      y2 := 2 / x2
  in y1 > y2 :=
by
  sorry

end linear_function_inequality_inverse_function_inequality_l60_60304


namespace max_sum_mult_table_l60_60114

def isEven (n : ℕ) : Prop := n % 2 = 0
def isOdd (n : ℕ) : Prop := ¬ isEven n
def entries : List ℕ := [3, 4, 6, 8, 9, 12]
def sumOfList (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem max_sum_mult_table :
  ∃ (a b c d e f : ℕ), 
    a ∈ entries ∧ b ∈ entries ∧ c ∈ entries ∧ 
    d ∈ entries ∧ e ∈ entries ∧ f ∈ entries ∧ 
    (isEven a ∧ isEven b ∧ isOdd c ∨ isEven a ∧ isOdd b ∧ isOdd c ∨ isOdd a ∧ isEven b ∧ isEven c ∨ isOdd a ∧ isOdd b ∧ isOdd c ∨ isOdd a ∧ isEven b ∧ isOdd c ∨ isEven a ∧ isOdd b ∧ isEven c) ∧ 
    (isEven d ∧ isEven e ∧ isOdd f ∨ isEven d ∧ isOdd e ∧ isOdd f ∨ isOdd d ∧ isEven e ∧ isEven f ∨ isOdd d ∧ isOdd e ∧ isOdd f ∨ isOdd d ∧ isEven e ∧ isOdd f ∨ isEven d ∧ isOdd e ∧ isEven f) ∧ 
    (sumOfList [a, b, c] * sumOfList [d, e, f] = 425) := 
by
    sorry  -- Skipping the proof as instructed.

end max_sum_mult_table_l60_60114


namespace m_eq_2_sufficient_but_not_necessary_l60_60730

-- Define the two lines l1 and l2
def line1 (m : ℝ) : ℝ × ℝ → ℝ := λ (p : ℝ × ℝ), 2 * p.1 - m * p.2 + 1
def line2 (m : ℝ) : ℝ × ℝ → ℝ := λ (p : ℝ × ℝ), p.1 + (m - 1) * p.2 - 1

-- Define the condition for lines being perpendicular
def perpendicular (l1 l2 : ℝ × ℝ → ℝ) : Prop :=
  (∃ m : ℝ, l1 = line1 m ∧ l2 = line2 m ∧ (2 / m) * (-1 / (m - 1)) = -1)

-- State the main theorem
theorem m_eq_2_sufficient_but_not_necessary (m : ℝ) :
  (m = 2 → perpendicular (line1 m) (line2 m)) ∧
  ∀ m, perpendicular (line1 m) (line2 m) → (m = 2 ∨ m = -1) :=
begin
  sorry
end

end m_eq_2_sufficient_but_not_necessary_l60_60730


namespace min_max_students_intersection_l60_60595

variable {I : Type} (students : Finset I) (A B : Finset I)
variable (hI : students.card = 100) (hA : A.card = 63) (hB : B.card = 75)

theorem min_max_students_intersection :
  38 ≤ (A ∩ B).card ∧ (A ∩ B).card ≤ 63 :=
by
  sorry

end min_max_students_intersection_l60_60595


namespace parabola_intersects_y_axis_l60_60702

theorem parabola_intersects_y_axis (m n : ℝ) :
  (∃ (x y : ℝ), y = x^2 + m * x + n ∧ 
  ((x = -1 ∧ y = -6) ∨ (x = 1 ∧ y = 0))) →
  (0, (-4)) = (0, n) :=
by
  sorry

end parabola_intersects_y_axis_l60_60702


namespace initial_amount_liquid_A_l60_60184

theorem initial_amount_liquid_A (A B : ℝ) (h1 : A / B = 4)
    (h2 : (A / (B + 40)) = 2 / 3) : A = 32 := by
  sorry

end initial_amount_liquid_A_l60_60184


namespace starting_number_divisible_by_3_count_l60_60850

-- Define a predicate for divisibility by 3
def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- Define the main theorem
theorem starting_number_divisible_by_3_count : 
  ∃ n : ℕ, (∀ m, n ≤ m ∧ m ≤ 50 → divisible_by_3 m → ∃ s, (m = n + 3 * s) ∧ s < 13) ∧
           (∀ k : ℕ, (divisible_by_3 k) → n ≤ k ∧ k ≤ 50 → m = 12) :=
sorry

end starting_number_divisible_by_3_count_l60_60850


namespace magnitude_of_vectors_l60_60695

variables (F1 F2 : ℝ × ℝ)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_vectors :
  F1 = (2, 2) ∧ F2 = (-2, 3) →
  vector_magnitude (vector_add F1 F2) = 5 :=
by
  intro h
  rw [h.1, h.2]
  dsimp [vector_add, vector_magnitude]
  norm_num
  -- proof goes here
  sorry

end magnitude_of_vectors_l60_60695


namespace difference_of_bases_l60_60281

def base8_to_base10 (n : ℕ) : ℕ :=
  5 * (8^5) + 4 * (8^4) + 3 * (8^3) + 2 * (8^2) + 1 * (8^1) + 0 * (8^0)

def base5_to_base10 (n : ℕ) : ℕ :=
  4 * (5^4) + 3 * (5^3) + 2 * (5^2) + 1 * (5^1) + 0 * (5^0)

theorem difference_of_bases : 
  base8_to_base10 543210 - base5_to_base10 43210 = 177966 :=
by
  sorry

end difference_of_bases_l60_60281


namespace f_periodic_l60_60037

-- Definitions and conditions
def is_periodic (f: ℝ → ℝ) (p: ℝ) : Prop := ∀ x: ℝ, f(x + p) = f(x)

variable (f : ℝ → ℝ) (a : ℝ)
variable h_cond : ∀ x: ℝ, f(x + a) = 1/2 + real.sqrt (f(x) - f(x)^2)
variable h_pos : a > 0

-- Theorem statement
theorem f_periodic : is_periodic f (2 * a) :=
sorry

-- Example of a non-constant function for a = 1
example : ∃ f: ℝ → ℝ, (∀ x: ℝ, f(x + 1) = 1/2 + real.sqrt (f(x) - f(x)^2)) ∧ (¬ is_periodic f 1) :=
begin
  use (λ x, 1/2 + 1/2 * real.abs (real.cos (real.pi * x/2))),
  split,
  { intro x,
    have h_cos: real.cos (real.pi * (x + 1) / 2) = real.sin (real.pi * x / 2),
    { rw [← real.cos_add_pi_div_two, ← @real.mul_div_cancel_left x 1 two_ne_zero], },
    rw [h_cos, real.abs_sin_eq_sqrt_cos_squared],
    simp only [and_true, eq_self_iff_true, real.mul_div_cancel, one_ne_zero],
  },
  { intro h_periodic,
    have h_period: (λ x, 1/2 + 1/2 * real.abs (real.cos (real.pi * x/2))) 2 = (λ x, 1/2 + 1/2 * real.abs (real.cos (real.pi * x/2))) 0,
    { exact h_periodic 0, },
    calc (λ x, 1/2 + 1/2 * real.abs (real.cos (real.pi * x/2))) 2
        = 1/2 + 1/2 * real.abs (real.cos (real.pi)) : by simp
    ... = 1  : by simp only [real.abs_neg, real.cos_pi, mul_one, neg_self, add_halves]
    ... = 1/2 + 1/2 * real.abs 1 : by simp
    ... = 1/2 + 1/2 * 1 : by simp only [real.abs_one]
    ... = 1    : by ring,
    contradiction }
end

end f_periodic_l60_60037


namespace espresso_job_complete_at_2_20_pm_l60_60577

noncomputable def espresso_completion_time (start_time : ℕ) (partial_completion_time : ℕ) : ℕ :=
let time_interval := partial_completion_time - start_time in
let one_fourth_time := (time_interval / 60) + (time_interval % 60) / 60 in
let total_time := one_fourth_time * 4 in
start_time + total_time

theorem espresso_job_complete_at_2_20_pm :
  espresso_completion_time 540 620 = 860 := by
  -- 540 minutes is 9:00 AM, 620 minutes is 10:20 AM
  -- 860 minutes is 2:20 PM
  sorry

end espresso_job_complete_at_2_20_pm_l60_60577


namespace conference_hall_initial_people_l60_60026

theorem conference_hall_initial_people (x : ℕ)  
  (h1 : 3 ∣ x) 
  (h2 : 4 ∣ (2 * x / 3))
  (h3 : (x / 2) = 27) : 
  x = 54 := 
by 
  sorry

end conference_hall_initial_people_l60_60026


namespace shift_sin_function_l60_60489

-- Define the original function f(x)
def f (x : ℝ) : ℝ := Real.sin (2 * x)

-- The transformation condition
def g (x : ℝ) : ℝ := Real.sin (2 * (x - π / 6))

-- The expected transformed function
def expected_g (x : ℝ) : ℝ := Real.sin (2 * x - π / 3)

-- Prove that g(x) is equivalent to expected_g(x)
theorem shift_sin_function :
  ∀ x : ℝ, g x = expected_g x := 
by 
  sorry

end shift_sin_function_l60_60489


namespace solve_for_k_l60_60221

def sameLine (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem solve_for_k :
  (sameLine (3, 10) (1, k) (-7, 2)) → k = 8.4 :=
by
  sorry

end solve_for_k_l60_60221


namespace altitude_triangle_hypotenuse_l60_60590

noncomputable def altitude_leg_lengths (a b : ℝ) (h_cond1 : a ≥ b) (h_cond2 : a * b = a * 2 * b) : ℝ :=
  ∃ (h_alt : ℝ), h_alt = (2 * a * b / (Real.sqrt (a ^ 2 + 4 * b ^ 2)))

theorem altitude_triangle_hypotenuse (a b : ℝ) (h_cond1 : a ≥ b) (h_cond2 : a * b = (a * b) / 2) :
  altitude_leg_lengths a b h_cond1 h_cond2 = (2 * a * b) / (Real.sqrt (a ^ 2 + 4 * b ^ 2)) :=
sorry

end altitude_triangle_hypotenuse_l60_60590


namespace find_b_l60_60954

theorem find_b (b : ℝ) (n : ℝ) (h1 : n = 2 ^ 0.1) (h2 : n ^ b = 16) : b = 40 := 
by
  sorry

end find_b_l60_60954


namespace second_order_det_example_l60_60599

theorem second_order_det_example : matrix.det ![![2, 1], ![-3, 4]] = 11 := by
  sorry

end second_order_det_example_l60_60599


namespace f_1982_eq_660_l60_60866

def f : ℕ → ℕ := sorry

axiom h1 : ∀ m n : ℕ, f (m + n) - f m - f n = 0 ∨ f (m + n) - f m - f n = 1
axiom h2 : f 2 = 0
axiom h3 : f 3 > 0
axiom h4 : f 9999 = 3333

theorem f_1982_eq_660 : f 1982 = 660 := sorry

end f_1982_eq_660_l60_60866


namespace pieces_given_by_brother_l60_60077

-- Given conditions
def original_pieces : ℕ := 18
def total_pieces_now : ℕ := 62

-- The statement to prove
theorem pieces_given_by_brother : total_pieces_now - original_pieces = 44 := by
  -- Starting with the given conditions
  unfold original_pieces total_pieces_now
  -- Place to insert the proof
  sorry

end pieces_given_by_brother_l60_60077


namespace correct_square_root_operation_l60_60170

theorem correct_square_root_operation : 
  (sqrt 4)^2 = 4 ∧ sqrt 4 ≠ 2 ∨ -2 ∧ sqrt ((-4)^2) ≠ -4 ∧ (-sqrt 4)^2 ≠ -4 :=
by
  have a : (sqrt 4)^2 = 4, from sorry,
  have b : sqrt 4 ≠ 2 ∨ -2, from sorry,
  have c : sqrt ((-4)^2) ≠ -4, from sorry,
  have d : (-sqrt 4)^2 ≠ -4, from sorry,
  exact ⟨a, b, c, d⟩

end correct_square_root_operation_l60_60170


namespace range_of_a_mono_increase_l60_60713

def f (x : ℝ) := (1 / 2) * x^2 + 2 * x - 2 * log x

theorem range_of_a_mono_increase :
  (∀ x > 0, derivative ℝ f x ≥ 0) ↔ (a ≤ 0) := sorry

end range_of_a_mono_increase_l60_60713


namespace circumcircles_meet_at_common_point_l60_60036

open real -- for dealing with geometric definitions

noncomputable def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

-- Define the points A, B, C of an acute triangle
variables (A B C : ℝ × ℝ)
variable (h_acute : -- condition to check if triangle ABC is acute)
variable (h_not_eq : B ≠ C)

-- Define midpoints D, E, F of the sides of triangle ABC
def D := midpoint B C
def E := midpoint C A
def F := midpoint A B

-- Define points M, N as the midpoints of the arcs
variable (M N : ℝ × ℝ)
variable (h_minor_arc : -- M is midpoint of minor arc BC not containing A)
variable (h_major_arc : -- N is midpoint of major arc BAC)

-- Define the points W, X, Y, Z
variable (W : ℝ × ℝ) -- incenter of triangle DEF
variable (X : ℝ × ℝ) -- D-excenter of triangle DEF
variable (Y : ℝ × ℝ) -- E-excenter of triangle DEF
variable (Z : ℝ × ℝ) -- F-excenter of triangle DEF

-- Conclude that the circumcircles of triangles ABC, WNX, YMZ meet at a common point
theorem circumcircles_meet_at_common_point :
  ∃ P : ℝ × ℝ, P ∈ circumcircle A B C ∧ P ∈ circumcircle W N X ∧ P ∈ circumcircle Y M Z :=
sorry

end circumcircles_meet_at_common_point_l60_60036


namespace largest_number_less_than_2_l60_60138

theorem largest_number_less_than_2 (a b c : ℝ) (h_a : a = 0.8) (h_b : b = 1/2) (h_c : c = 0.5) : 
  a < 2 ∧ b < 2 ∧ c < 2 ∧ (∀ x, (x = a ∨ x = b ∨ x = c) → x < 2) → 
  a = 0.8 ∧ 
  (a > b ∧ a > c) ∧ 
  (a < 2) :=
by sorry

end largest_number_less_than_2_l60_60138


namespace different_meal_combinations_l60_60179

-- Defining the conditions explicitly
def items_on_menu : ℕ := 12

-- A function representing possible combinations of choices for Yann and Camille
def meal_combinations (menu_items : ℕ) : ℕ :=
  menu_items * (menu_items - 1)

-- Theorem stating that given 12 items on the menu, the different combinations of meals is 132
theorem different_meal_combinations : meal_combinations items_on_menu = 132 :=
by
  sorry

end different_meal_combinations_l60_60179


namespace club_truncator_equal_wins_losses_l60_60622

noncomputable def probability_equal_wins_losses : ℚ :=
  252 * (1/4)^(10 : ℚ)

theorem club_truncator_equal_wins_losses :
  probability_equal_wins_losses = 63 / 262144 :=
by
  sorry

end club_truncator_equal_wins_losses_l60_60622


namespace returns_to_start_point_after_fourth_passenger_distance_after_last_passenger_total_earnings_l60_60478

noncomputable def driving_distances : List ℤ := [-5, 3, 6, -4, 7, -2]

def fare (distance : ℕ) : ℕ :=
  if distance ≤ 3 then 8 else 8 + 2 * (distance - 3)

theorem returns_to_start_point_after_fourth_passenger :
  List.sum (driving_distances.take 4) = 0 :=
by
  sorry

theorem distance_after_last_passenger :
  List.sum driving_distances = 5 :=
by
  sorry

theorem total_earnings :
  (fare 5 + fare 3 + fare 6 + fare 4 + fare 7 + fare 2) = 68 :=
by
  sorry

end returns_to_start_point_after_fourth_passenger_distance_after_last_passenger_total_earnings_l60_60478


namespace reciprocal_of_neg_two_l60_60904

variable a : ℤ

theorem reciprocal_of_neg_two (h : a = -2) : ∃ b : ℚ, a * b = 1 ∧ b = -1 / 2 := 
by
  exists (-1 / 2)
  rw [h]
  split
  · norm_num
  · refl

end reciprocal_of_neg_two_l60_60904


namespace max_black_squares_2021x2021_l60_60765

-- Define the conditions of the grid and the movement of the mouse
def grid (n : ℕ) := matrix (fin n) (fin n) bool

def mouse_can_escape (n : ℕ) (g : grid n) (x y : fin n) : Prop :=
  (∀ i : fin n, (g x.val i = ff) ∨ (g i y.val = ff))

-- Prove that the maximum number of black squares is 8080 for a 2021x2021 grid
theorem max_black_squares_2021x2021 : 
  ∀ (g : grid 2021), (∀ x y, mouse_can_escape 2021 g x y) → (∑ x y, if g x y then 1 else 0) ≤ 8080 :=
by
  sorry

end max_black_squares_2021x2021_l60_60765


namespace fly_least_distance_l60_60231

noncomputable def leastDistance (r : ℝ) (h : ℝ) (start_dist : ℝ) (end_dist : ℝ) : ℝ := 
  let C := 2 * Real.pi * r
  let R := Real.sqrt (r^2 + h^2)
  let θ := C / R
  let A := (start_dist, 0)
  let B := (Real.cos (θ / 2) * end_dist, Real.sin (θ / 2) * end_dist)
  Real.sqrt ((B.fst - A.fst)^2 + (B.snd - A.snd)^2)

theorem fly_least_distance : 
  leastDistance 600 (200 * Real.sqrt 7) 125 (375 * Real.sqrt 2) = 625 := 
sorry

end fly_least_distance_l60_60231


namespace length_of_string_proof_l60_60216

def length_of_string (circumference loops height : ℝ) : ℝ :=
  let vertical_distance := height / loops
  let hypotenuse := real.sqrt (vertical_distance^2 + circumference^2)
  loops * hypotenuse

theorem length_of_string_proof:
  length_of_string 5 5 15 = 5 * real.sqrt 34 := by
  -- This is the statement we are proving
  sorry

end length_of_string_proof_l60_60216


namespace find_CQ_l60_60561

-- Define the context of the square ABCD with points and conditions.
structure Square (α : Type*) :=
  (A B C D : α)  -- Vertices of the square
  (length : ℝ)  -- Side length of the square
  (point_on_AC : α) -- Point P on the diagonal AC
  (point_on_CD : α) -- Point Q on the side CD
  (perpendicular_PQ_BC : Prop) -- Property that PQ is perpendicular to BC
  (area_of_CPQ : ℝ) -- Area of triangle CPQ

noncomputable def square1 : Square ℝ :=
{
  A := (0, 0),
  B := (1, 0),
  C := (1, 1),
  D := (0, 1),
  length := 1,
  point_on_AC := (1 / 2, 1 / 2),
  point_on_CD := (0, 1 / 2),
  perpendicular_PQ_BC := true, -- PQ is perpendicular to BC by Ptolemy's theorem
  area_of_CPQ := 6 / 25
}

-- The theorem to prove CQ = 3 / 5
theorem find_CQ (sq : Square ℝ) (h1 : sq.length = 1)
  (h2 : sq.area_of_CPQ = 6 / 25) : 
  (dist sq.C sq.point_on_CD) = 3 / 5 := 
begin
  sorry -- Proof not provided
end

end find_CQ_l60_60561


namespace slope_of_tangent_line_at_pi_div_12_l60_60717

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem slope_of_tangent_line_at_pi_div_12 :
  (Real.sin (π / 12) + Real.cos (π / 12))' = (Real.cos (π / 12) - Real.sin (π / 12)) :=
begin
  sorry
end

end slope_of_tangent_line_at_pi_div_12_l60_60717


namespace investment_after_eighteen_months_l60_60253

theorem investment_after_eighteen_months
  (initial_investment : ℝ)
  (first_annual_interest_rate second_annual_interest_rate : ℝ)
  (duration : ℝ)
  (first_interest_rate duration1 : ℝ := (9 : ℕ) / 12 * first_annual_interest_rate : ℝ  := 9 / 12 * 9 / 100 duration2 : ℝ := (9 : ℕ) / 12 * second_annual_interest_rate : ℝ := 9 / 12 * 15 / 100)
  : initial_investment * (1 + first_interest_rate) * (1 + second_interest_rate) = 17814.0625 :=
sorry

end investment_after_eighteen_months_l60_60253


namespace sum_m_n_is_192_l60_60933

def smallest_prime : ℕ := 2

def largest_four_divisors_under_200 : ℕ :=
  -- we assume this as 190 based on the provided problem's solution
  190

theorem sum_m_n_is_192 :
  smallest_prime = 2 →
  largest_four_divisors_under_200 = 190 →
  smallest_prime + largest_four_divisors_under_200 = 192 :=
by
  intros h1 h2
  sorry

end sum_m_n_is_192_l60_60933


namespace perfect_square_trinomial_l60_60390

theorem perfect_square_trinomial (a : ℝ) :
  (∃ m : ℝ, (x^2 + (a-1)*x + 9) = (x + m)^2) → (a = 7 ∨ a = -5) :=
by
  sorry

end perfect_square_trinomial_l60_60390


namespace range_of_x_l60_60159

theorem range_of_x (a : ℝ) (x : ℝ) (h0 : 0 ≤ a) (h1 : a ≤ 2) :
  a * x^2 + (a + 1) * x + 1 - (3 / 2) * a < 0 → -2 < x ∧ x < -1 :=
by
  sorry

end range_of_x_l60_60159


namespace integer_points_on_line_segment_l60_60025

def is_point_on_line (x y : ℤ) (p1 p2 : ℤ × ℤ) : Prop :=
  let m := (p2.2 - p1.2) / (p2.1 - p1.1) in
  y = m * (x - p1.1) + p1.2

def line_segment (p1 p2 : ℤ × ℤ) : set (ℤ × ℤ) :=
  { z | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
    z.1 = round (t * p2.1 + (1 - t) * p1.1) ∧
    z.2 = round (t * p2.2 + (1 - t) * p1.2) }

theorem integer_points_on_line_segment :
  let p1 := (-9, -2) in
  let p2 := (6, 8) in 
  ∃ (pts : set (ℤ × ℤ)), pts = line_segment p1 p2 ∧
  pts.count = 6 := 
sorry

end integer_points_on_line_segment_l60_60025


namespace visitors_correct_l60_60247

def visitors_that_day : ℕ := 92
def visitors_previous_day : ℕ := 419
def total_visitors_before_that_day : ℕ := 522
def visitors_two_days_before : ℕ := total_visitors_before_that_day - visitors_previous_day - visitors_that_day

theorem visitors_correct : visitors_two_days_before = 11 := by
  -- Sorry, proof to be filled in
  sorry

end visitors_correct_l60_60247


namespace hyperbola_properties_l60_60583

-- Define the given ellipse equation and foci
def ellipse_equation (x y : ℝ) : Prop := x^2 / 27 + y^2 / 36 = 1
def foci : set (ℝ × ℝ) := {(0, -6), (0, 6)}

-- Definition of the assumed hyperbola equation with placeholders for parameters
def hyperbola_equation (a² : ℝ) (b² : ℝ) [nontrivial_ordered_ring ℝ] (x y : ℝ) : Prop :=
  y^2 / a² - x^2 / b² = 1

variables (a : ℝ) (c : ℝ)

-- Problem statement conditions
axiom point_on_hyperbola : hyperbola_equation 4 5 (sqrt 15) 4
axiom foci_on_hyperbola : foci = {(0, -6), (0, 6)}

-- Final statement to prove
theorem hyperbola_properties :
  ∃ a² b², a² = 4 ∧ b² = 5 ∧ hyperbola_equation a² b² (sqrt 15) 4 ∧ (
    ∃ e, e = 3 / 2 ∧ (
      ∃ (m : ℝ), m = 2 * sqrt 5 / 5 ∧ (
        ∀ x y, y = m * x ∨ y = -m * x -> hyperbola_equation a² b² x y
      )
    )
  ) := sorry

end hyperbola_properties_l60_60583


namespace general_formula_an_sum_of_bn_l60_60917

open Nat

-- Condition: a1 = 3
def a1 : ℕ := 3

-- Condition: (a_n, a_{n+1}) lies on line y = x + 2
def an (n : ℕ) : ℕ := 2 * n + 1

-- Condition: b_n = a_n * 3^n
def bn (n : ℕ) : ℕ := an n * 3^n

-- Sum of the first n terms of sequence {b_n}, denoted as T_n
def Tn (n : ℕ) : ℕ := ∑ i in range n, bn i

theorem general_formula_an (n : ℕ) : an n = 2 * n + 1 :=
by
  -- Proof is skipped.
  sorry

theorem sum_of_bn (n : ℕ) : Tn n = n * 3^(n+1) :=
by
  -- Proof is skipped.
  sorry

end general_formula_an_sum_of_bn_l60_60917


namespace arithmetic_geometric_sequence_l60_60127

-- Let {a_n} be an arithmetic sequence
-- And let a_1, a_2, a_3 form a geometric sequence
-- Given that a_5 = 1, we aim to prove that a_10 = 1
theorem arithmetic_geometric_sequence (a : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_geom : a 1 * a 3 = (a 2) ^ 2)
  (h_a5 : a 5 = 1) :
  a 10 = 1 :=
sorry

end arithmetic_geometric_sequence_l60_60127


namespace percentage_increase_second_year_l60_60151

theorem percentage_increase_second_year :
  let initial_deposit : ℤ := 1000
  let balance_first_year : ℤ := 1100
  let total_balance_two_years : ℤ := 1320
  let percent_increase_first_year : ℚ := ((balance_first_year - initial_deposit) / initial_deposit) * 100
  let percent_increase_total : ℚ := ((total_balance_two_years - initial_deposit) / initial_deposit) * 100
  let increase_second_year : ℤ := total_balance_two_years - balance_first_year
  let percent_increase_second_year : ℚ := (increase_second_year / balance_first_year) * 100
  percent_increase_first_year = 10 ∧
  percent_increase_total = 32 ∧
  increase_second_year = 220 → 
  percent_increase_second_year = 20 := by
  intros initial_deposit balance_first_year total_balance_two_years percent_increase_first_year
         percent_increase_total increase_second_year percent_increase_second_year
  sorry

end percentage_increase_second_year_l60_60151


namespace reciprocal_of_neg_two_l60_60888

theorem reciprocal_of_neg_two : ∃ y : ℝ, (-2) * y = 1 ∧ y = -1/2 :=
by
  use -1/2
  split
  · -- Prove that (-2) * (-1/2) = 1
    sorry
  · -- Prove that y = -1/2
    rfl

end reciprocal_of_neg_two_l60_60888


namespace median_divides_triangles_into_equal_areas_l60_60111

-- Define a triangle in a Euclidean space
structure Triangle :=
(A B C : Point)  -- Points representing the vertices of the triangle

-- Define the midpoint of a line segment
def midpoint (p1 p2 : Point) : Point := (p1 + p2) / 2

-- Define the median of a triangle from vertex A to the midpoint of BC
def median (T : Triangle) : LineSegment :=
  let K := midpoint T.B T.C
  LineSegment.mk T.A K

-- The theorem statement: Prove that the median divides the triangle into two triangles with equal area
theorem median_divides_triangles_into_equal_areas (T : Triangle) :
  let K := midpoint T.B T.C in
  area (Triangle.mk T.A T.B K) = area (Triangle.mk T.A K T.C) :=
sorry

end median_divides_triangles_into_equal_areas_l60_60111


namespace dice_roll_probability_even_sum_l60_60669

theorem dice_roll_probability_even_sum (d : ℕ → ℕ) (h_dice : ∀ i, 1 ≤ d i ∧ d i ≤ 6)
  (h_odd_product : ∀ i, d i % 2 = 1) : 
  (∃ n, (∑ i in (finRange 5), d i) = 2 * n) :=
by
  sorry

end dice_roll_probability_even_sum_l60_60669


namespace solutions_irrational_l60_60467

theorem solutions_irrational :
  ∃ a b : ℝ, a ≠ b ∧
  (∀ x, (x = a ∨ x = b) → log 10 (x^2 - 18 * x + 24) = 2) ∧
  (irrational a ∧ irrational b) :=
sorry

end solutions_irrational_l60_60467


namespace points_on_circle_120_degrees_l60_60135

theorem points_on_circle_120_degrees (N : ℕ) (points : Fin N → ℝ) (h : ∀ (i j : Fin N), i ≠ j → arc_length (points i) (points j) < 2 * π / 3) : 
  ∃ arc : Set ℝ, ∀ (i : Fin N), points i ∈ arc ∧ arc_length_set arc ≤ 2 * π / 3 :=
by
  sorry

end points_on_circle_120_degrees_l60_60135


namespace solve_y_l60_60085

theorem solve_y :
  ∀ y : ℚ, 6 * (4 * y - 1) - 3 = 3 * (2 - 5 * y) ↔ y = 5 / 13 :=
by
  sorry

end solve_y_l60_60085


namespace espresso_completion_time_l60_60979

theorem espresso_completion_time :
  ∀ (start_time quarter_completed_time total_duration : ℕ), start_time = 9 * 60 + 15 → 
  quarter_completed_time = 12 * 60 + 30 →
  total_duration = 13 * 60 →
  let completion_time := start_time + total_duration in
  completion_time = 22 * 60 + 15 := 
by
  intros start_time quarter_completed_time total_duration h_start_time h_quarter_completed_time h_total_duration completion_time.
  suffices : completion_time = 1335, by simp [this].
  calc completion_time
      = start_time + total_duration     : rfl
  ... = (9 * 60 + 15) + (13 * 60)       : by rw [h_start_time, h_total_duration]
  ... = 540 + 15 + 780                 : by norm_num
  ... = 1335                           : by norm_num

end espresso_completion_time_l60_60979


namespace sequence_sum_abs_8010_l60_60023

noncomputable def sequence (a b n : ℕ) : ℤ := a * 2^n + b * n - 80

/-- Variables and minimum sum condition -/
variables (a b : ℕ)
variables (h_a_pos : 0 < a) (h_b_pos : 0 < b)
variables (S_min : (∑ n in Finset.range 6, sequence a b (n + 1)) < 
            (∑ n in Finset.range 7, sequence a b (n + 1)))

/-- Divisibility condition -/
variables (h_div : 7 ∣ sequence a b 36)

noncomputable def sum_abs_1_12 : ℤ := 
  ∑ n in Finset.range 12, abs (sequence a b (n + 1))

theorem sequence_sum_abs_8010 (h1 : S_min) (h2 : h_div) : 
  sum_abs_1_12 a b = 8010 :=
sorry

end sequence_sum_abs_8010_l60_60023


namespace triangle_area_l60_60782

open Classical

noncomputable def triangle_area_problem : Prop :=
  ∃ (A M C V U : Type) (AM MC : ℝ), 
  (AM = MC) ∧
  (angle A M C = 90) ∧
  (∃ MV CU : ℝ, MV = 10 ∧ CU = 10 ∧ angle M V U = 90) ∧
  (4 * (1/2 * MV * CU) = 200)

theorem triangle_area (A M C V U : Type) (AM MC : ℝ)
  (h_eq : AM = MC) (h_angle : angle A M C = 90)
  (MV CU : ℝ) (h_medians : MV = 10 ∧ CU = 10 ∧ angle M V U = 90) :
  4 * (1 / 2 * MV * CU) = 200 :=
  sorry

end triangle_area_l60_60782


namespace last_triangle_perimeter_l60_60039

-- Definition of the sequence of triangles and their properties
def triangle (a b c : ℕ) := True

-- Initial triangle T1
def T1 := triangle 2021 2022 2023

-- Function to get the next triangle in the sequence
def next_triangle (a b c : ℕ) : (ℕ × ℕ × ℕ) := 
  let ad := (b + c - a) / 2 in
  let be := (a + c - b) / 2 in
  let cf := (a + b - c) / 2 in
  (ad, be, cf)

-- Function to calculate the perimeter of a triangle
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- The final assertion to be proved
theorem last_triangle_perimeter : 
  perimeter (2022 / 2^9) (2022 / 2^9) (2022 / 2^9)
  = 1516.5 / 128 :=
sorry

end last_triangle_perimeter_l60_60039


namespace number_of_positive_terms_up_to_100_l60_60805

noncomputable def a (n : ℕ) : ℝ := if n = 0 then 0 else (1 / n) * Real.sin (n * Real.pi / 25)

noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), a i

theorem number_of_positive_terms_up_to_100 : 
  ∀ n, 1 ≤ n ∧ n ≤ 100 -> S n > 0 := 
by 
  intro n 
  intro h 
  sorry

end number_of_positive_terms_up_to_100_l60_60805


namespace find_highest_number_l60_60456

theorem find_highest_number (A B C D E : ℕ) 
  (h1 : B = 76) 
  (h2 : C = 84) 
  (h3 : D = 88) 
  (h4 : A + B + C + D + E = 408) 
  (h5 : E - A = 24) : 
  E = 92 := 
begin
  sorry
end

end find_highest_number_l60_60456


namespace abs_h_equals_half_l60_60131

noncomputable def equation_has_sum_of_squares (h : ℝ) : Prop :=
  let r := -4 * h in
  let s := -8 : ℝ in
  (r ^ 2 + s ^ 2) = 20

theorem abs_h_equals_half (h : ℝ) : equation_has_sum_of_squares h → |h| = 1 / 2 :=
by
  intro h_condition
  sorry

end abs_h_equals_half_l60_60131


namespace largest_four_digit_integer_divisible_by_75_and_7_l60_60660

theorem largest_four_digit_integer_divisible_by_75_and_7 :
  ∃ m : ℕ, 999 < m ∧ m < 10000 ∧ m % 75 = 0 ∧ m % 7 = 0 ∧ 
          let reversed (k : ℕ) := nat.of_digits 10 (k.digits 10).reverse
          in (reversed m % 75 = 0) ∧ 
          m = 5775 := by
  sorry

end largest_four_digit_integer_divisible_by_75_and_7_l60_60660


namespace second_game_score_count_l60_60518

-- Define the conditions and problem
def total_points (A1 A2 A3 B1 B2 B3 : ℕ) : Prop :=
  A1 + A2 + A3 + B1 + B2 + B3 = 31

def valid_game_1 (A1 B1 : ℕ) : Prop :=
  A1 ≥ 11 ∧ A1 - B1 ≥ 2

def valid_game_2 (A2 B2 : ℕ) : Prop :=
  B2 ≥ 11 ∧ B2 - A2 ≥ 2

def valid_game_3 (A3 B3 : ℕ) : Prop :=
  A3 ≥ 11 ∧ A3 - B3 ≥ 2

def game_sequence (A1 A2 A3 B1 B2 B3 : ℕ) : Prop :=
  valid_game_1 A1 B1 ∧ valid_game_2 A2 B2 ∧ valid_game_3 A3 B3

noncomputable def second_game_score_possibilities : ℕ := 
  8 -- This is derived from calculating the valid scores where B wins the second game.

theorem second_game_score_count (A1 A2 A3 B1 B2 B3 : ℕ) (h_total : total_points A1 A2 A3 B1 B2 B3) (h_sequence : game_sequence A1 A2 A3 B1 B2 B3) :
  second_game_score_possibilities = 8 := sorry

end second_game_score_count_l60_60518


namespace degree_g_eq_5_l60_60473

def f (x : ℝ) : ℝ := -3 * x ^ 5 + 2 * x ^ 4 + x ^ 2 - 6

theorem degree_g_eq_5 (g : ℝ → ℝ) (h : ∀ x, polynomial.degree (f x + g x) = 2) :
  polynomial.degree (g x) = 5 :=
sorry

end degree_g_eq_5_l60_60473


namespace overlapping_sectors_area_l60_60934

noncomputable def sector_area (r : ℝ) (θ : ℝ) : ℝ :=
  (θ / (2 * π)) * (π * r^2)

theorem overlapping_sectors_area :
  let r := 15
  let theta := π / 2 
  let overlap_theta := π / 4 in
  2 * (sector_area r theta) - (sector_area r overlap_theta) = 84.375 * π :=
by
  sorry

end overlapping_sectors_area_l60_60934


namespace find_length_of_train_l60_60557

def train_length (length_train length_platform speed_kph : ℕ) (time_min : ℕ) : Prop :=
  length_train = length_platform ∧ speed_kph = 36 ∧ time_min = 1 → length_train = 300

theorem find_length_of_train (length_train length_platform speed_kph : ℕ) (time_min : ℕ) :
  train_length length_train length_platform speed_kph time_min :=
by
  intro h
  cases h with h1 h2
  cases h2 with h_speed h_time
  sorry

end find_length_of_train_l60_60557


namespace max_black_cells_in_101x101_grid_l60_60969

theorem max_black_cells_in_101x101_grid :
  ∀ (k : ℕ), k ≤ 101 → 2 * k * (101 - k) ≤ 5100 :=
by
  sorry

end max_black_cells_in_101x101_grid_l60_60969


namespace zero_of_f_l60_60134

def f (x : ℝ) : ℝ := 2^x - 8

theorem zero_of_f : f 3 = 0 := by
  sorry

end zero_of_f_l60_60134


namespace biased_coin_probability_l60_60569

theorem biased_coin_probability
  (p : ℝ)
  (h1 : 6.choose 2 * p^2 * (1 - p)^4 = 6.choose 3 * p^3 * (1 - p)^3) :
  let prob_heads_four := 6.choose 4 * (3/7)^4 * (4/7)^2 in
  prob_heads_four = 240 / 1453 ∧ 240 + 1453 = 1693 :=
by
  sorry

end biased_coin_probability_l60_60569


namespace part1_union_part1_inter_complement_part2_empty_intersection_l60_60332

open Set

def A : Set ℝ := { x | -3 < x ∧ x < 4 }
def B (m : ℝ) : Set ℝ := { x | m - 1 < x ∧ x < 3 * m + 3 }
def R : Set ℝ := { x | true }

theorem part1_union (m : ℝ) (h : m = 2) : 
  A ∪ B m = { x | -3 < x ∧ x < 9 } :=
by
  sorry

theorem part1_inter_complement (m : ℝ) (h : m = 2) : 
  A ∩ (R \ B m) = { x | -3 < x ∧ x ≤ 1 } :=
by
  sorry

theorem part2_empty_intersection (m : ℝ) (h : A ∩ B m = ∅) : 
  m ≥ 5 ∨ m ≤ -2 :=
by
  sorry

end part1_union_part1_inter_complement_part2_empty_intersection_l60_60332


namespace simplify_expression_l60_60464

theorem simplify_expression :
  ( (sqrt 3 / sqrt 4 + sqrt 4 / sqrt 5) * (sqrt 5 / sqrt 6) ) = ( (sqrt 10 + 2 * sqrt 2) / 4 ) :=
by
  sorry

end simplify_expression_l60_60464


namespace pam_bags_count_l60_60827

noncomputable def geralds_bag_apples : ℕ := 40

noncomputable def pams_bag_apples := 3 * geralds_bag_apples

noncomputable def pams_total_apples : ℕ := 1200

theorem pam_bags_count : pams_total_apples / pams_bag_apples = 10 := by 
  sorry

end pam_bags_count_l60_60827


namespace remaining_money_exact_l60_60789

def initial_money : ℝ := 20
def spent_on_snacks (money : ℝ) : ℝ := money * 1/5
def spent_on_necessities (money : ℝ) : ℝ := money * 3/4
def spent_on_gift (money : ℝ) : ℝ := money * 1/2
def spent_on_book (money : ℝ) : ℝ := money * 0.20
def gave_to_sister (money : ℝ) : ℝ := money * 1/3

theorem remaining_money_exact : 
  let snacks := spent_on_snacks initial_money in
  let after_snacks := initial_money - snacks in
  let necessities := spent_on_necessities after_snacks in
  let after_necessities := after_snacks - necessities in
  let gift := spent_on_gift after_necessities in
  let after_gift := after_necessities - gift in
  let book := spent_on_book after_gift in
  let after_book := after_gift - book in
  let sister := gave_to_sister after_book in
  let final := after_book - sister in
  final = 1.07 :=
by sorry

end remaining_money_exact_l60_60789


namespace count_quasi_increasing_permutations_7_l60_60616

def quasi_increasing_permutation (n : ℕ) (p : List ℕ) : Prop :=
  ∀ k : ℕ, k < n - 1 → (p[k] ≤ p[k + 1] + 2)

def count_quasi_increasing_permutations (n : ℕ) : ℕ :=
  if n = 3 then 6
  else if n = 4 then 3 * count_quasi_increasing_permutations 3
  else if n = 5 then 3 * count_quasi_increasing_permutations 4
  else if n = 6 then 3 * count_quasi_increasing_permutations 5
  else if n = 7 then 3 * count_quasi_increasing_permutations 6
  else 0

theorem count_quasi_increasing_permutations_7 : count_quasi_increasing_permutations 7 = 486 :=
by
  -- Base case handled by the definition itself
  have base_case : count_quasi_increasing_permutations 3 = 6 := rfl
  rw [count_quasi_increasing_permutations, base_case]
  rw [count_quasi_increasing_permutations, base_case]
  rw [count_quasi_increasing_permutations, base_case]
  rw [count_quasi_increasing_permutations, base_case]
  rw [count_quasi_increasing_permutations, base_case]
  have n4 : count_quasi_increasing_permutations 4 = 18 := by norm_num
  have n5 : count_quasi_increasing_permutations 5 = 54 := by norm_num
  have n6 : count_quasi_increasing_permutations 6 = 162 := by norm_num
  have n7 : count_quasi_increasing_permutations 7 = 486 := by norm_num
  exact n7

end count_quasi_increasing_permutations_7_l60_60616


namespace log_sqrt_defined_range_l60_60314

theorem log_sqrt_defined_range (x: ℝ) : 
  (∃ (y: ℝ), y = (log (5-x) / sqrt (x+2))) ↔ (-2 ≤ x ∧ x < 5) :=
by
  sorry

end log_sqrt_defined_range_l60_60314


namespace m_angle_ECD_eq_150_l60_60394

namespace Geometry

-- Define a constant point structure
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Define the predicate for collinearity
def collinear (A B C : Point) : Prop :=
∃ k : ℝ, C.x = A.x + k * (B.x - A.x) ∧ C.y = A.y + k * (B.y - A.y)

-- Assume the existence of points A, B, C, D, E
variables (A B C D E : Point)

-- Given conditions
axiom AC_eq_BC : dist A C = dist B C
axiom m_angle_ACB_eq_30 : angle A C B = 30
axiom CD_parallel_AB : parallel (Line.mk C D) (Line.mk A B)
axiom E_on_ext_AC : ∃ k > 1, E.x = C.x + k * (A.x - C.x) ∧ E.y = C.y + k * (A.y - C.y)

-- Intended result
theorem m_angle_ECD_eq_150 : angle E C D = 150 :=
sorry

end Geometry

end m_angle_ECD_eq_150_l60_60394


namespace coefficient_of_x2_in_expansion_l60_60413

theorem coefficient_of_x2_in_expansion : 
  (coeff (expand (1 + x) 10 x 2) - coeff (expand (1 - x) 9 x 2)) = 9 := 
  sorry

end coefficient_of_x2_in_expansion_l60_60413


namespace distinct_sets_count_l60_60666

noncomputable def num_distinct_sets : ℕ :=
  let product : ℕ := 11 * 21 * 31 * 41 * 51 * 61
  728

theorem distinct_sets_count : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 11 * 21 * 31 * 41 * 51 * 61 ∧ num_distinct_sets = 728 :=
sorry

end distinct_sets_count_l60_60666


namespace find_x_l60_60379

noncomputable def S (x : ℝ) : ℝ := 1 + 3 * x + 5 * x^2 + 7 * x^3 + ∑' n, (2 * n - 1) * x^n

theorem find_x (x : ℝ) (h : S x = 16) : x = 3/4 :=
sorry

end find_x_l60_60379


namespace katya_sum_greater_than_masha_l60_60060

theorem katya_sum_greater_than_masha (a b c d : ℝ) (h : a + b + c + d = 2021) :
  ((a+1)*(b+1) + (b+1)*(c+1) + (c+1)*(d+1) + (d+1)*(a+1)) - (a*b + b*c + c*d + d*a) = 4046 := by
  sorry

end katya_sum_greater_than_masha_l60_60060


namespace quadrilateral_perimeter_bounds_l60_60099

-- Define the length of an edge of the tetrahedron
def edge_length (a : ℝ) : Prop := a > 0

-- Define what it means for a plane to intersect a tetrahedron and form a quadrilateral
def intersects_as_quadrilateral (plane : set Point) (tetrahedron : Tetrahedron) (quad : Quadrilateral) : Prop :=
  plane ∩ (boundary tetrahedron) = quad

-- Main theorem statement
theorem quadrilateral_perimeter_bounds 
  (a : ℝ) (plane : set Point) (tetrahedron : Tetrahedron) (quad : Quadrilateral)
  (h_edge_length : edge_length a)
  (h_intersection : intersects_as_quadrilateral plane tetrahedron quad) :
  2 * a ≤ quad.perimeter ∧ quad.perimeter ≤ 3 * a :=
by
  sorry

end quadrilateral_perimeter_bounds_l60_60099


namespace reciprocal_of_neg_two_l60_60899

noncomputable theory

def is_reciprocal (a x : ℝ) := a * x = 1

theorem reciprocal_of_neg_two : 
  ∃ x : ℝ, is_reciprocal (-2) x ∧ x = -1 / 2 := 
sorry

end reciprocal_of_neg_two_l60_60899


namespace gain_percentage_is_20_l60_60238

variable (CP SP1 SP2 : ℝ) (loss percentageGain : ℝ)

-- Definitions based on the conditions
def condition1 := SP1 = 170
def condition2 := SP2 = 240
def condition3 := loss = 0.15 * CP
def equation1 := 170 = CP - loss

-- The main theorem: The gain percentage when sold at Rs. 240 is 20%
theorem gain_percentage_is_20 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : equation1) :
  percentageGain = 0.2 := by
  sorry

end gain_percentage_is_20_l60_60238


namespace complete_square_k_value_l60_60006

noncomputable def complete_square_form (x : ℝ) : ℝ := x^2 - 7 * x

theorem complete_square_k_value : ∃ a h k : ℝ, complete_square_form x = a * (x - h)^2 + k ∧ k = -49 / 4 :=
by
  use [1, 7/2, -49/4]
  -- This proof step will establish the relationships and the equality
  sorry

end complete_square_k_value_l60_60006


namespace quadrilateral_with_right_angles_is_rectangle_l60_60542

theorem quadrilateral_with_right_angles_is_rectangle
  (Q : Type)
  [quad : quadrilateral Q]
  (h : ∀ (A B C D : Q), is_right_angle (∠ ABC) ∧ is_right_angle (∠ BCD) ∧ is_right_angle (∠ CDA) ∧ is_right_angle (∠ DAB))
  : is_rectangle Q :=
sorry

end quadrilateral_with_right_angles_is_rectangle_l60_60542


namespace solution_of_inequality_l60_60503

theorem solution_of_inequality :
  { x : ℝ | 2^(x^2 - 5 * x + 5) > 1 / 2 } = { x : ℝ | x < 2 ∨ x > 3 } :=
by
  -- We need to use the properties of exponential functions and solve the inequality
  sorry

end solution_of_inequality_l60_60503


namespace laura_total_owed_l60_60956

variable (P : ℝ) (R : ℝ) (T : ℝ)

def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * R * T

theorem laura_total_owed
    (hP : P = 35)
    (hR : R = 0.07)
    (hT : T = 1) :
  P + simple_interest P R T = 37.45 :=
by
  rw [hP, hR, hT]
  show 35 + simple_interest 35 0.07 1 = 37.45
  simp [simple_interest]
  norm_num
  sorry

end laura_total_owed_l60_60956


namespace sequence_solution_l60_60367

-- Define the sequence x_n
def x (n : ℕ) : ℚ := n / (n + 2016)

-- Given condition: x_2016 = x_m * x_n
theorem sequence_solution (m n : ℕ) (h : x 2016 = x m * x n) : 
  m = 4032 ∧ n = 6048 := 
  by sorry

end sequence_solution_l60_60367


namespace dihedral_angle_PAD_PBC_l60_60021

-- Define the basic geometrical structures
def point : Type := ℝ × ℝ × ℝ
def vector := point
def plane (a b c: point) := { n : vector // n ≠ (0, 0, 0) }

-- Given points A, B, C, D
variables (P A B C D : point)
variables (PAD ABCD PBC: plane P A D) (E: point)

-- Given conditions
variables (hp1 : ∃ PAD ABCD : plane P A D, PAD.n × ABCD.n = (0, 0, 0))
variables (hp2 : ∃ (l : ℝ), 0 < l ∧ l = 2 ∧ equilateral_triangle P A D l)
variables (hp3 : ∃ α β γ δ: ℝ, rhombus A B C D α β γ δ)
variables (hp4 : ∃ α, 0 < α ∧ α = 60 ∧ angle_sum A B D α)

-- Prove the desired dihedral angle
theorem dihedral_angle_PAD_PBC : ∃ θ : ℝ, 0 < θ ∧ θ = 45 ∧ dihedral_angle PAD PBC θ :=
sorry

end dihedral_angle_PAD_PBC_l60_60021


namespace inequality_m_l60_60305

def m (A B C : Point) : ℝ :=
  if collinear A B C then 0
  else min (height A B C) (min (height B A C) (height C A B))

theorem inequality_m {A B C X : Point} :
  m A B C ≤ m A B X + m A X C + m X B C :=
sorry

end inequality_m_l60_60305


namespace first_tv_cost_is_672_l60_60816

-- width and height of the first TV
def width_first_tv : ℕ := 24
def height_first_tv : ℕ := 16
-- width and height of the new TV
def width_new_tv : ℕ := 48
def height_new_tv : ℕ := 32
-- cost of the new TV
def cost_new_tv : ℕ := 1152
-- extra cost per square inch for the first TV
def extra_cost_per_square_inch : ℕ := 1

noncomputable def cost_first_tv : ℕ :=
  let area_first_tv := width_first_tv * height_first_tv
  let area_new_tv := width_new_tv * height_new_tv
  let cost_per_square_inch_new_tv := cost_new_tv / area_new_tv
  let cost_per_square_inch_first_tv := cost_per_square_inch_new_tv + extra_cost_per_square_inch
  cost_per_square_inch_first_tv * area_first_tv

theorem first_tv_cost_is_672 : cost_first_tv = 672 := by
  sorry

end first_tv_cost_is_672_l60_60816


namespace parabola_translation_correct_l60_60514

-- Define the original equation of the parabola
def original_parabola (x : ℝ) : ℝ := 8 * x^2

-- Define the transformation of translating 3 units to the left and 5 units down
def translate_parabola (f : ℝ → ℝ) (h k : ℝ) :=
  λ x, f (x + h) - k

-- Define the expected result after translation
def expected_result (x : ℝ) : ℝ := 8 * (x + 3)^2 - 5

-- The theorem statement proving the expected transformation
theorem parabola_translation_correct :
  ∀ x : ℝ, translate_parabola original_parabola 3 5 x = expected_result x :=
by
  intro x
  sorry

end parabola_translation_correct_l60_60514


namespace axes_of_symmetry_coincide_l60_60777

-- Definitions for the parabolas
def parabola1 : ℝ → ℝ := λ x, p1 * x^2 + q1 * x + r1
def parabola2 : ℝ → ℝ := λ x, p2 * x^2 + q2 * x + r2

-- Conditions
variable (p1 q1 r1 p2 q2 r2 : ℝ)
variable (h_p1_pos : 0 < p1) (h_p2_neg : p2 < 0)
variable (intersects : ∃ A B : ℝ × ℝ, A ≠ B ∧ parabola1 A.1 = A.2 ∧ parabola2 A.1 = A.2 ∧ parabola1 B.1 = B.2 ∧ parabola2 B.1 = B.2)
variable (inscribed_circle : ∀ (A B : ℝ × ℝ), A ≠ B → ∃ (C D : ℝ × ℝ), is_tangent parabola1 A.1 C.1 ∧ is_tangent parabola2 A.1 C.1 ∧ is_tangent parabola1 B.1 D.1 ∧ is_tangent parabola2 B.1 D.1 ∧ quadrilateral_with_inscribed_circle A B C D)

theorem axes_of_symmetry_coincide :
  ∃ c : ℝ, ∀ (x : ℝ), p1 * x + q1 / (2 * p1) = c ∧ p2 * x + q2 / (2 * p2) = c := sorry

end axes_of_symmetry_coincide_l60_60777


namespace sum_of_largest_and_smallest_is_correct_l60_60547

-- Define the set of digits
def digits : Finset ℕ := {2, 0, 4, 1, 5, 8}

-- Define the largest possible number using the digits
def largestNumber : ℕ := 854210

-- Define the smallest possible number using the digits
def smallestNumber : ℕ := 102458

-- Define the sum of largest and smallest possible numbers
def sumOfNumbers : ℕ := largestNumber + smallestNumber

-- Main theorem to prove
theorem sum_of_largest_and_smallest_is_correct : sumOfNumbers = 956668 := by
  sorry

end sum_of_largest_and_smallest_is_correct_l60_60547


namespace hyperbola_standard_equation_l60_60582

theorem hyperbola_standard_equation :
  (∃ C : (ℝ × ℝ → Prop), 
    (∃ e : (ℝ × ℝ → Prop), 
      (e = (λ p, let x := p.1 in let y := p.2 in x^2 / 9 + y^2 / 4 = 1)) ∧ 
      (∃ F1 F2 : ℝ × ℝ, 
        let f_length := (2 : ℝ) * Real.sqrt 5 in 
        ∃ c : ℝ, c = Real.sqrt 5 ∧ 2 * c = f_length ∧ 
        ∃ λ : ℝ, 
          (C = (λ p, let x := p.1 in let y := p.2 in (x^2 / (4 * λ) - y^2 / λ = 1) ∨ (y^2 / λ - x^2 / (4 * λ) = 1))) ∧ 
          x - 2 * y = 0 ∧ 
          (λ > 0 → (4 * λ + λ = 5)) ∧ 
          (λ < 0 → (-λ - 4 * λ = 5) ∧ (λ = -1)) →
    (C = (λ p, let x := p.1 in let y := p.2 in x^2 / 4 - y^2 = 1) ∨ C = (λ p, let x := p.1 in let y := p.2 in y^2 - x^2 / 4 = 1)))))) :=
begin
  sorry
end

end hyperbola_standard_equation_l60_60582


namespace complete_square_k_value_l60_60005

noncomputable def complete_square_form (x : ℝ) : ℝ := x^2 - 7 * x

theorem complete_square_k_value : ∃ a h k : ℝ, complete_square_form x = a * (x - h)^2 + k ∧ k = -49 / 4 :=
by
  use [1, 7/2, -49/4]
  -- This proof step will establish the relationships and the equality
  sorry

end complete_square_k_value_l60_60005


namespace area_of_efcd_l60_60779

-- Definitions for lengths of the sides
variables (AB CD AD BC : ℕ)
-- Definitions for the points dividing the sides into thirds
variables (E F : ℕ)

-- Conditions provided in the problem
def trapezoid_abcd (AB CD AD BC : ℕ) : Prop :=
  AB = 15 ∧ CD = 30 ∧ AD = 3 * (E / 3) ∧ BC = 3 * (F /3) ∧ ((AD + BC) / 2) = 22.5

-- Length of segment EF is determined by the points dividing legs
def segment_ef (AB CD : ℕ) (E F : ℕ) : ℕ :=
  2 * ((AB + CD) / 3)

-- The altitude of EFCD
def altitude_efcd (total_altitude : ℕ) : ℕ :=
  2 * (total_altitude / 3)

-- Area of quadrilateral EFCD
def area_efcd (AB CD : ℕ) (total_altitude : ℕ) : ℕ :=
  altitude_efcd total_altitude * ((segment_ef AB CD (2/3) (3/3)) / 2)

-- Problem statement
theorem area_of_efcd (AB CD AD BC : ℕ) (total_altitude : ℕ) :
  trapezoid_abcd AB CD AD BC →
  altitude_efcd total_altitude = 12 →
  segment_ef AB CD (2/3) (3/3) = 30 →
  area_efcd AB CD total_altitude = 360 :=
by
  sorry

end area_of_efcd_l60_60779


namespace smallest_x_for_f_l60_60982

-- Define the function f with its properties
def f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 4 then 1 - |x - 3| else 0 -- This is simplified

-- Properties of f for all positive real values of x
axiom f_property_pos (x : ℝ) (h : 0 < x) : f (3 * x) = 3 * f x

-- Proof problem statement
theorem smallest_x_for_f (x : ℝ) : f x = f 2004 → x = 729 := by
  sorry

end smallest_x_for_f_l60_60982


namespace define_interval_l60_60307

theorem define_interval (x : ℝ) : 
  (0 < x + 2) → (0 < 5 - x) → (-2 < x ∧ x < 5) :=
by
  intros h1 h2
  sorry

end define_interval_l60_60307


namespace probability_painting_l60_60642

/-- Define the probability that every ball is different in color from exactly half of the other 7 balls,
given each of the 8 balls is painted either black or white with equal probability. -/
noncomputable def probability_diff_half (n : ℕ) : ℚ :=
  if n = 8 then 
    let prob_each = (1 / 2 : ℚ) ^ 8 in
    let num_favorable = Nat.choose 8 4 in
    (num_favorable * prob_each : ℚ)
  else 0

theorem probability_painting (n : ℕ) (hn : n = 8) :
  probability_diff_half n = 35 / 128 :=
by {
  rw [hn, probability_diff_half],
  -- simplify the if condition
  split_ifs,
  -- calculate the values
  have prob_each := (1 / 2 : ℚ) ^ 8,
  have num_favorable := Nat.choose 8 4,
  have num_fav_prob := num_favorable * prob_each,
  norm_num at prob_each num_fav_prob,
  rw [prob_each, num_fav_prob],
  norm_num
}

end probability_painting_l60_60642


namespace lottery_not_guaranteed_to_win_l60_60593

theorem lottery_not_guaranteed_to_win (total_tickets : ℕ) (winning_rate : ℚ) (num_purchased : ℕ) :
  total_tickets = 100000 ∧ winning_rate = 1 / 1000 ∧ num_purchased = 2000 → 
  ∃ (outcome : ℕ), outcome = 0 := by
  sorry

end lottery_not_guaranteed_to_win_l60_60593


namespace factorize_quadratic_trinomial_l60_60282

theorem factorize_quadratic_trinomial (t : ℝ) : t^2 - 10 * t + 25 = (t - 5)^2 :=
by
  sorry

end factorize_quadratic_trinomial_l60_60282


namespace chord_square_length_l60_60621

theorem chord_square_length
    (r1 r2 r3 L1 L2 L3 : ℝ)
    (h1 : r1 = 4) 
    (h2 : r2 = 8) 
    (h3 : r3 = 12) 
    (tangent1 : ∀ x, (L1 - x)^2 + (L2 - x)^2 = (r1 + r2)^2)
    (tangent2 : ∀ x, x^2 + (L3 - x)^2 = (r3 - r2)^2) 
    (tangent3 : ∀ x, x^2 + (L3 - x)^2 = (r3 - r1)^2) : L1^2 = 3584 / 9 :=
by
  sorry

end chord_square_length_l60_60621


namespace ordered_pairs_3430_l60_60116

theorem ordered_pairs_3430 :
  let count_divisors (n : Nat) (p : Nat) := n.gcd p in
  let choices := (count_divisors 3430 2 + 1) * (count_divisors 3430 5 + 1) * (count_divisors 3430 7 + 1) in
  choices = 16 :=
by
  have h1 : count_divisors 3430 2 = 1 := sorry
  have h2 : count_divisors 3430 5 = 1 := sorry
  have h3 : count_divisors 3430 7 = 3 := sorry
  let choices := (h1 + 1) * (h2 + 1) * (h3 + 1)
  have h_choices : choices = 16 := sorry
  exact h_choices

end ordered_pairs_3430_l60_60116


namespace calculate_expression_is_correct_l60_60615

noncomputable def calculate_expression : ℝ :=
  -(-2) + 2 * Real.cos (Real.pi / 3) + (-1 / 8)⁻¹ + (Real.pi - 3.14) ^ 0

theorem calculate_expression_is_correct :
  calculate_expression = -4 :=
by
  -- the conditions as definitions
  have h1 : Real.cos (Real.pi / 3) = 1 / 2 := by sorry
  have h2 : (Real.pi - 3.14) ^ 0 = 1 := by sorry
  -- use these conditions to prove the main statement
  sorry

end calculate_expression_is_correct_l60_60615


namespace least_prime_factor_of_expr_l60_60939

theorem least_prime_factor_of_expr : ∀ n : ℕ, n = 11^5 - 11^2 → (∃ p : ℕ, Nat.Prime p ∧ p ≤ 2 ∧ p ∣ n) :=
by
  intros n h
  -- here will be proof steps, currently skipped
  sorry

end least_prime_factor_of_expr_l60_60939


namespace non_negative_integer_solutions_l60_60087

theorem non_negative_integer_solutions :
  { (x, y, z, t) : ℕ × ℕ × ℕ × ℕ | x + y + z + t = 5 ∧ x + 2y + 5z + 10t = 17 } =
  {(1, 3, 0, 1), (2, 0, 3, 0)} :=
by
  sorry

end non_negative_integer_solutions_l60_60087


namespace proof_problem_l60_60699

theorem proof_problem
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) :
  (x + 1 / y ≥ 2) ∨ (y + 1 / z ≥ 2) ∨ (z + 1 / x ≥ 2) :=
sorry

end proof_problem_l60_60699


namespace reciprocal_of_neg_two_l60_60883

theorem reciprocal_of_neg_two : ∃ x : ℝ, (-2) * x = 1 ∧ x = -1/2 :=
by
  use -1/2
  split
  · exact (by norm_num : (-2) * (-1/2) = 1)
  · rfl

end reciprocal_of_neg_two_l60_60883


namespace relationship_between_y1_y2_y3_l60_60871

-- Define the parabola equation and points
def parabola (x c : ℝ) : ℝ := 2 * (x - 1)^2 + c

-- Define the points
def point1 := -2
def point2 := 0
def point3 := 5 / 3

-- Define the y values at these points
def y1 (c : ℝ) := parabola point1 c
def y2 (c : ℝ) := parabola point2 c
def y3 (c : ℝ) := parabola point3 c

-- Proof statement
theorem relationship_between_y1_y2_y3 (c : ℝ) : 
  y1 c > y2 c ∧ y2 c > y3 c :=
sorry

end relationship_between_y1_y2_y3_l60_60871


namespace reciprocal_of_neg_two_l60_60893

theorem reciprocal_of_neg_two : ∃ y : ℝ, (-2) * y = 1 ∧ y = -1/2 :=
by
  use -1/2
  split
  · -- Prove that (-2) * (-1/2) = 1
    sorry
  · -- Prove that y = -1/2
    rfl

end reciprocal_of_neg_two_l60_60893


namespace average_price_of_cow_l60_60565

theorem average_price_of_cow (total_price_cows_and_goats rs: ℕ) (num_cows num_goats: ℕ)
    (avg_price_goat: ℕ) (total_price: total_price_cows_and_goats = 1400)
    (num_cows_eq: num_cows = 2) (num_goats_eq: num_goats = 8)
    (avg_price_goat_eq: avg_price_goat = 60) :
    let total_price_goats := avg_price_goat * num_goats
    let total_price_cows := total_price_cows_and_goats - total_price_goats
    let avg_price_cow := total_price_cows / num_cows
    avg_price_cow = 460 :=
by
  sorry

end average_price_of_cow_l60_60565


namespace find_m_plus_n_l60_60673

def mean_of_set (s : List ℕ) : ℚ := ((s.foldl (· + ·) 0) : ℚ) / (s.length : ℚ)

def median_of_set (s : List ℕ) : ℚ :=
  if h : s.length % 2 = 0 then
    let mid := s.length / 2
    ((s.nth_le (mid - 1) sorry) + (s.nth_le mid sorry) : ℚ) / 2
  else
    s.nth_le (s.length / 2) sorry

theorem find_m_plus_n (m n p : ℕ) (hm : 0 < m) (hn : 0 < n) (h_cond : m + 15 < n + 5)
  (h_mean : mean_of_set [m, m+5, m+15, n+5, n+6, 2n-1] = p)
  (h_median : median_of_set [m, m+5, m+15, n+5, n+6, 2n-1] = p) :
  m + n = 34 := by
  sorry

end find_m_plus_n_l60_60673


namespace reciprocal_of_neg_two_l60_60881

theorem reciprocal_of_neg_two : ∃ x : ℝ, (-2) * x = 1 ∧ x = -1/2 :=
by
  use -1/2
  split
  · exact (by norm_num : (-2) * (-1/2) = 1)
  · rfl

end reciprocal_of_neg_two_l60_60881


namespace parabola_focus_to_directrix_distance_correct_l60_60102

def parabola_focus_to_directrix_distance (a : ℕ) (y x : ℝ) : Prop :=
  y^2 = 2 * x → a = 2 →  1 = 1

theorem parabola_focus_to_directrix_distance_correct :
  ∀ (a : ℕ) (y x : ℝ), parabola_focus_to_directrix_distance a y x :=
by
  unfold parabola_focus_to_directrix_distance
  intros
  sorry

end parabola_focus_to_directrix_distance_correct_l60_60102


namespace count_sets_without_perfect_square_l60_60802

theorem count_sets_without_perfect_square :
  (∃ (count : ℕ), 
    count = (2000 - 
      (∑ i in finset.range (2000),
       if ∃ n in finset.range (50 * (i + 1)), is_square n then 1 else 0)
    )
  ) = 1507 := by
  sorry

end count_sets_without_perfect_square_l60_60802


namespace find_function_l60_60290

/-- Any function f : ℝ → ℝ satisfying the two given conditions must be of the form f(x) = cx where |c| ≤ 1. -/
theorem find_function (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, x ≠ 0 → x * (f (x + 1) - f x) = f x)
  (h2 : ∀ x y : ℝ, |f x - f y| ≤ |x - y|) :
  ∃ c : ℝ, (∀ x : ℝ, f x = c * x) ∧ |c| ≤ 1 :=
by
  sorry

end find_function_l60_60290


namespace find_common_ratio_l60_60414

-- Definitions based on the conditions given in the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

def sequence_a : ℕ → ℝ
| 0 := 2 / 4 / 4
| 1 := 2 / 4
| 2 := 2
| 3 := 2 * 2
| 4 := 2 * 2 * 2
| 5 := 16 / 2 / 2

-- The final proof problem statement
theorem find_common_ratio
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a 2)
  (h2 : a 2 = 2)
  (h3 : a 5 = 16) :
  2 = 2 :=
by sorry

end find_common_ratio_l60_60414


namespace solution_proof_l60_60574

-- Definitions based on problem conditions
def radius_omega : ℝ := 8
def center_omega (O : Type*) : Prop := is_center O
def radius_Omega : ℝ := (real.sqrt 145) / 2
def center_Omega (B : Type*) : Prop := is_center B
def inscribed_triangle (E N T : Type*) : Prop := inscribed_omega_omega_in_triangle E N T (ω)
def circumscribed_triangle (A C N : Type*) : Prop := circumscribed_Omega_Omega_around_triangle A C N (Ω)
def radius_of_omega_A_angle (O A : Type*) : Prop := angle O A ω = 90
def radius_of_omega_C_angle (O C : Type*) : Prop := angle O C ω = 90

-- Proof Goal 1: Find ON
def find_ON (O N : Type*) : Prop := ON = (real.sqrt 145)

-- Additional Conditions (Part b)
def area_ratio (B T E N T : Type*) : Prop := area B T E / area E N T = 7 / 10
def bisector_length (N L : Type*) : Prop := N L = (5 * (real.sqrt 145)) / 3
def triangle_area (E N T : Type*) : Prop := area E N T = 360

-- Goal 2:
def find_bisector_area (N L : Type*) : Prop := N L = (5 * (real.sqrt 145)) / 3 ∧ area E N T = 360

def proof_goal (O N : Type*) (B T E : Type*) (N L : Type*) (E N T : Type*) : Prop :=
(∃ O, center_omega O ∧ radius_omega = 8) ∧
(∃ B, center_Omega B ∧ radius_Omega = (real.sqrt 145) / 2) ∧
(∃ (E N T : Type*), inscribed_triangle E N T) ∧
(∃ (A C N : Type*), circumscribed_triangle A C N) ∧
(∃ O A , radius_of_omega_A_angle O A) ∧
(∃ O C , radius_of_omega_C_angle O C) ∧
find_ON O N ∧
area_ratio B T E N T ∧
find_bisector_area N L

theorem solution_proof : proof_goal O N B T E N L E N T :=
by sorry

end solution_proof_l60_60574


namespace circumcircle_radius_of_triangle_l60_60785

theorem circumcircle_radius_of_triangle 
  (A B C : ℝ)
  (angle_A : A = π / 3)
  (AB : B = 2)
  (AC : C = 3) :
  ∃ R, R = sqrt 21 / 3 :=
sorry

end circumcircle_radius_of_triangle_l60_60785


namespace pentagon_area_correct_l60_60761

noncomputable def pentagon_area (a b d : ℝ) : ℝ :=
  let r := a / (2 * Real.sin (36 * Real.pi / 180))
  in (5 / 2) * a * r * Real.sin (72 * Real.pi / 180)

theorem pentagon_area_correct (a b d : ℝ) (ha : a = 10) : 
  pentagon_area a b d = 250 * Real.sin (72 * Real.pi / 180) / (2 * Real.sin (36 * Real.pi / 180)) :=
by
  sorry

end pentagon_area_correct_l60_60761


namespace all_terms_perfect_squares_l60_60916

def seq_x : ℕ → ℕ
| 0       => 1
| 1       => 1
| (n + 2) => 14 * seq_x (n + 1) - seq_x n - 4

theorem all_terms_perfect_squares : ∀ n, ∃ k, seq_x n = k^2 :=
by
  sorry

end all_terms_perfect_squares_l60_60916


namespace shaded_area_fraction_is_5_over_8_l60_60992

-- Define the setup with the geometric constraints
def triangle_area_fraction_shaded 
  (triangle : Type) 
  (a b c : triangle) 
  (point_on_side : (triangle → triangle → triangle) → ℚ → triangle) 
  (D := point_on_side a b 0.25) 
  (E := point_on_side b c 0.25) 
  (F := point_on_side c a 0.25) : Prop :=
  -- The fraction of the area of the triangle that is shaded is 5/8
  (shaded_area_fraction : ℚ) = 5 / 8

-- Statement to prove
theorem shaded_area_fraction_is_5_over_8 
  {triangle : Type} 
  {a b c : triangle} 
  {point_on_side : (triangle → triangle → triangle) → ℚ → triangle}
  (D := point_on_side a b 0.25)
  (E := point_on_side b c 0.25)
  (F := point_on_side c a 0.25) :
  triangle_area_fraction_shaded triangle a b c point_on_side :=
  sorry

end shaded_area_fraction_is_5_over_8_l60_60992


namespace correct_condition_l60_60865

-- Definitions based on Conditions
variables {a b : ℝ}

-- Condition 1
def condition1 := a > b ∧ b > 0 → a^2 > b^2

-- Condition 2
def condition2 := a > b ∧ b > 0 ↔ 1/a < 1/b

-- Condition 3
def condition3 := a > b ∧ b > 0 ↔ a^3 > b^3

-- Final statement to prove
theorem correct_condition : 
  (∀ a b, condition1 a b) ∧ ¬( ∀ a b, condition2 a b) ∧ ¬( ∀ a b, condition3 a b) := 
sorry

end correct_condition_l60_60865


namespace binomial_coefficient_example_l60_60624

theorem binomial_coefficient_example :
  2 * (Nat.choose 7 4) = 70 := 
sorry

end binomial_coefficient_example_l60_60624


namespace radius_of_circle_inscribed_in_triangle_is_correct_l60_60532

noncomputable def inradius_of_inscribed_circle (DE DF EF : ℝ) 
  (h_DE : DE = 26)
  (h_DF : DF = 15)
  (h_EF : EF = 17) : ℝ :=
let s := (DE + DF + EF) / 2 in
let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
(K / s)

theorem radius_of_circle_inscribed_in_triangle_is_correct :
  inradius_of_inscribed_circle 26 15 17 26.refl 15.refl 17.refl = (18 * Real.sqrt 34.5) / 29 :=
by
  sorry

end radius_of_circle_inscribed_in_triangle_is_correct_l60_60532


namespace math_problem_l60_60758

noncomputable def students_scoring_above_110 
  (n : ℕ)  -- number of students
  (mu sigma : ℝ)  -- mean and standard deviation of the normal distribution
  (P_90_100 : ℝ)  -- Probability that a student's score is between 90 and 100
  (h_distribution : ∀ x, (λ ξ, P (ξ ≤ x)) = x ↦ (1/2) * (1 + erf (x - mu) / (sigma * sqrt 2))) : ℝ :=
  let P_100_110 := P_90_100 in
  let P_above_110 := 1 - (P_90_100 + P_100_110) in
  n * P_above_110

theorem math_problem 
  (h1 : ∀ x, (λ ξ, P (ξ ≤ x)) = x ↦ (1/2) * (1 + erf (x - 100) / (10 * sqrt 2))) 
  (h2 : P (λ ξ, 90 ≤ ξ ∧ ξ ≤ 100) = 0.3) :
  students_scoring_above_110 50 100 10 0.3 h1 = 10 :=
sorry

end math_problem_l60_60758


namespace cone_to_cylinder_water_height_l60_60255

theorem cone_to_cylinder_water_height :
  let r_cone := 15 -- radius of the cone
  let h_cone := 24 -- height of the cone
  let r_cylinder := 18 -- radius of the cylinder
  let V_cone := (1 / 3: ℝ) * Real.pi * r_cone^2 * h_cone -- volume of the cone
  let h_cylinder := V_cone / (Real.pi * r_cylinder^2) -- height of the water in the cylinder
  h_cylinder = 8.33 := by
  sorry

end cone_to_cylinder_water_height_l60_60255


namespace trapezoid_area_sum_l60_60245

theorem trapezoid_area_sum : 
  let r₁ := 9
      r₂ := 11
      r₃ := 24
      n₁ := 15
      n₂ := 21
  in ⌊r₁ + r₂ + r₃ + n₁ + n₂⌋ = 80 :=
by 
  sorry

end trapezoid_area_sum_l60_60245


namespace sum_of_specific_terms_l60_60055

theorem sum_of_specific_terms 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h1 : S 3 = 9) 
  (h2 : S 6 = 36) 
  (h3 : ∀ n, S n = n * (a 1) + d * n * (n - 1) / 2) :
  a 7 + a 8 + a 9 = 45 := 
sorry

end sum_of_specific_terms_l60_60055


namespace math_expression_equals_2014_l60_60844

-- Define the mapping for each letter
def M : Nat := 1
def A : Nat := 8
def T : Nat := 3
def I : Nat := 9
def K : Nat := 0 -- K corresponds to 'minus', to be used in expression

-- Verification that the expression evaluates to 2014
theorem math_expression_equals_2014 : (100 * M + 10 * A + T) + (1000 * (M + 10 * A + 100 * T) + 100 * M + 10 * A + T + I) - A = 2014 := by
  calc
    (100 * M + 10 * A + T) + (1000 * (M + 10 * A + 100 * T) + 100 * M + 10 * A + T + I) - A
        = (100 * 1 + 10 * 8 + 3) + (1000 * (1 + 10 * 8 + 100 * 3) + 100 * 1 + 10 * 8 + 3 + 9) - 8 : by rfl
    ... = 183 + 1839 - 8 : by rfl
    ... = 2014 : by rfl

end math_expression_equals_2014_l60_60844


namespace germs_per_dish_l60_60190

-- Define the total number of germs as 0.036 * 10^5
def total_germs : ℝ := 0.036 * 10^5

-- Define the total number of petri dishes as 75000 * 10^-3
def total_petri_dishes : ℝ := 75000 * 10 ^ (-3)

-- Prove that the number of germs per dish is 48 given the above definitions
theorem germs_per_dish : (total_germs / total_petri_dishes) = 48 := by 
  sorry

end germs_per_dish_l60_60190


namespace Janet_saves_154_minutes_per_week_l60_60278

-- Definitions for the time spent on each activity daily
def timeLookingForKeys := 8 -- minutes
def timeComplaining := 3 -- minutes
def timeSearchingForPhone := 5 -- minutes
def timeLookingForWallet := 4 -- minutes
def timeSearchingForSunglasses := 2 -- minutes

-- Total time spent daily on these activities
def totalDailyTime := timeLookingForKeys + timeComplaining + timeSearchingForPhone + timeLookingForWallet + timeSearchingForSunglasses
-- Time savings calculation for a week
def weeklySaving := totalDailyTime * 7

-- The proof statement that Janet will save 154 minutes every week
theorem Janet_saves_154_minutes_per_week : weeklySaving = 154 := by
  sorry

end Janet_saves_154_minutes_per_week_l60_60278


namespace slope_of_decreasing_linear_function_l60_60345

theorem slope_of_decreasing_linear_function (m b : ℝ) :
  (∀ x y : ℝ, x < y → mx + b > my + b) → m < 0 :=
by
  intro h
  sorry

end slope_of_decreasing_linear_function_l60_60345


namespace monotonicity_of_f_prove_x1_x2_l60_60358

noncomputable theory

-- Define the function f(x)
def f (x a : ℝ) : ℝ := a * x * exp x - (1 / 2) * x^2 - x

-- Monotonicity of f(x) on (0, +∞)
theorem monotonicity_of_f {a : ℝ} :
  (a ≤ 0 → ∀ x > 0, f x a < f (x + 1) a) ∧
  (a ≥ 1 → ∀ x > 0, f (x + 1) a > f x a) ∧
  (0 < a ∧ a < 1 → ∀ x ∈ (0, -real.log a), f (x + 1) a < f x a ∧ ∀ x > -real.log a, f (x + 1) a > f x a) :=
sorry

-- Prove x1 * x2 > exp (2 - x1 - x2)
theorem prove_x1_x2 {a x1 x2 : ℝ} (h₀ : a > 0) (h₁ : f x1 a = real.log x1 - (1 / 2) * x1^2)
  (h₂ : f x2 a = real.log x2 - (1 / 2) * x2^2) (h₃ : x1 ≠ x2) :
  x1 * x2 > exp (2 - x1 - x2) :=
sorry

end monotonicity_of_f_prove_x1_x2_l60_60358


namespace arun_weight_average_l60_60189

theorem arun_weight_average :
  ∀ (w : ℝ), (w > 61 ∧ w < 72) ∧ (w > 60 ∧ w < 70) ∧ (w ≤ 64) →
  (w = 62 ∨ w = 63) →
  (62 + 63) / 2 = 62.5 :=
by
  intros w h1 h2
  sorry

end arun_weight_average_l60_60189


namespace smallest_possible_integer_l60_60983

theorem smallest_possible_integer :
  ∃ (N : ℕ), (∀ k ∈ (Finset.range 35 ∪ Finset.range' 38 3) \ {35, 36, 37}, N % k = 0) ∧
             (N % 35 ≠ 0) ∧ (N % 36 ≠ 0) ∧ (N % 37 ≠ 0) ∧
             N = 299576986419800 :=
by {
  use 299576986419800,
  -- proof omitted
  sorry
}

end smallest_possible_integer_l60_60983


namespace point_where_incircle_touches_l60_60325

open Set

def hyperbola_foci_and_vertices (F₁ F₂ M N P : Point) :=
  is_hyperbola_foci_and_vertices F₁ F₂ M N ∧ lies_on_hyperbola P F₁ F₂

theorem point_where_incircle_touches (F₁ F₂ M N P G : Point)
  (h : hyperbola_foci_and_vertices F₁ F₂ M N P) :
  touches_incircle (incircle (triangle P F₁ F₂)) G (segment F₁ F₂) →
  G = M ∨ G = N :=
sorry

end point_where_incircle_touches_l60_60325


namespace lana_total_winter_clothing_l60_60794

-- Define the number of boxes, scarves per box, and mittens per box as given in the conditions
def num_boxes : ℕ := 5
def scarves_per_box : ℕ := 7
def mittens_per_box : ℕ := 8

-- The total number of pieces of winter clothing is calculated as total scarves plus total mittens
def total_winter_clothing : ℕ := num_boxes * scarves_per_box + num_boxes * mittens_per_box

-- State the theorem that needs to be proven
theorem lana_total_winter_clothing : total_winter_clothing = 75 := by
  sorry

end lana_total_winter_clothing_l60_60794


namespace cesaro_sum_51_term_l60_60300

/-- Defines the Cesaro sum of a sequence of numbers. -/
def cesaro_sum (n : ℕ) (seq : Fin n → ℝ) : ℝ :=
  (Finset.range n).sum (λ i, seq (i + 1)) / n

theorem cesaro_sum_51_term (b : Fin 50 → ℝ) (h : cesaro_sum 50 b = 600) :
  cesaro_sum 51 (λ i, if i = 0 then 2 else b (i - 1)) = 590.235294 :=
begin
  sorry
end

end cesaro_sum_51_term_l60_60300


namespace percentage_outside_C_is_61_11_l60_60819

def percentage_of_students_outside_C_range (scores : List Nat) : Real :=
  let C_range := (76, 85)
  let outside_C_scores := scores.filter (fun score => score < C_range.fst ∨ score > C_range.snd)
  (outside_C_scores.length.toReal / scores.length.toReal) * 100

theorem percentage_outside_C_is_61_11 :
  percentage_of_students_outside_C_range [98, 73, 55, 100, 76, 93, 88, 72, 77, 65, 82, 79, 68, 85, 91, 56, 81, 89] = 61.11 :=
by
  sorry

end percentage_outside_C_is_61_11_l60_60819


namespace inclusion_probability_of_a_l60_60522

noncomputable def probability_of_inclusion (population : ℕ) (sample_size : ℕ) (individual : ℕ) : ℚ :=
  if population = 10 ∧ sample_size = 3 ∧ individual = 1 then
    (36 : ℚ) / (120 : ℚ) -- Directly using the result 36/120 = 0.3
  else 0

theorem inclusion_probability_of_a (population sample_size individual : ℕ) :
  population = 10 →
  sample_size = 3 →
  individual = 1 →
  probability_of_inclusion population sample_size individual = 0.3 := by
  intros _ _ _
  unfold probability_of_inclusion
  rw if_pos (and.intro (and.intro rfl rfl) rfl)
  norm_num
  sorry

end inclusion_probability_of_a_l60_60522


namespace art_club_students_l60_60139

theorem art_club_students 
    (students artworks_per_student_per_quarter quarters_per_year artworks_in_two_years : ℕ) 
    (h1 : artworks_per_student_per_quarter = 2)
    (h2 : quarters_per_year = 4) 
    (h3 : artworks_in_two_years = 240) 
    (h4 : students * (artworks_per_student_per_quarter * quarters_per_year) * 2 = artworks_in_two_years) :
    students = 15 := 
by
    -- Given conditions for the problem
    sorry

end art_club_students_l60_60139


namespace barbara_removed_114_sheets_l60_60605

/-- Given conditions: -/
def bundles (n : ℕ) := 2 * n
def bunches (n : ℕ) := 4 * n
def heaps (n : ℕ) := 20 * n

/-- Barbara removed certain amounts of paper from the chest of drawers. -/
def total_sheets_removed := bundles 3 + bunches 2 + heaps 5

theorem barbara_removed_114_sheets : total_sheets_removed = 114 := by
  -- proof will be inserted here
  sorry

end barbara_removed_114_sheets_l60_60605


namespace average_reading_time_correct_l60_60144

-- We define total_reading_time as a parameter representing the sum of reading times
noncomputable def total_reading_time : ℝ := sorry

-- We define the number of students as a constant
def number_of_students : ℕ := 50

-- We define the average reading time per student based on the provided data
noncomputable def average_reading_time : ℝ :=
  total_reading_time / number_of_students

-- The theorem we need to prove: that the average reading time per student is correctly calculated
theorem average_reading_time_correct :
  ∃ (total_reading_time : ℝ), average_reading_time = total_reading_time / number_of_students :=
by
  -- since total_reading_time and number_of_students are already defined, we prove the theorem using them
  use total_reading_time
  exact rfl

end average_reading_time_correct_l60_60144


namespace third_vertex_l60_60204

/-- Two vertices of a right triangle are located at (4, 3) and (0, 0).
The third vertex of the triangle lies on the positive branch of the x-axis.
Determine the coordinates of the third vertex if the area of the triangle is 24 square units. -/
theorem third_vertex (x : ℝ) (h : x > 0) : 
  (1 / 2 * |x| * 3 = 24) → (x, 0) = (16, 0) :=
by
  intro h_area
  sorry

end third_vertex_l60_60204


namespace find_a_value_l60_60447

theorem find_a_value 
  (A : Set ℤ := {-1, 0, 1})
  (a : ℤ) 
  (B : Set ℤ := {a, a^2}) 
  (h_union : A ∪ B = A) : 
  a = -1 :=
sorry

end find_a_value_l60_60447


namespace area_of_triangle_QAC_l60_60266

theorem area_of_triangle_QAC (p : ℝ) : 
  let Q := (0 : ℝ, 12 : ℝ)
  let A := (3 : ℝ, 12 : ℝ)
  let C := (0 : ℝ, p - 3)
  ∃ base height : ℝ, base = 3 ∧ height = |15 - p| ∧ 
  (1/2) * base * height = (3/2) * |15 - p| :=
by
  let Q := (0 : ℝ, 12 : ℝ)
  let A := (3 : ℝ, 12 : ℝ)
  let C := (0 : ℝ, p - 3)
  let base := 3
  let height := |15 - p|
  use base, height
  split
  { refl }
  split
  { refl }
  sorry

end area_of_triangle_QAC_l60_60266


namespace compound_interest_difference_l60_60861

variable (P r : ℝ)

theorem compound_interest_difference :
  (P * 9 * r^2 = 360) → (P * r^2 = 40) :=
by
  sorry

end compound_interest_difference_l60_60861


namespace johns_total_spent_l60_60427

def original_price : Float := 20
def discount_rate : Float := 0.15
def number_of_pins : Nat := 10

theorem johns_total_spent : 
  let discount_on_each := original_price * discount_rate
  let discounted_price := original_price - discount_on_each
  let total_cost := discounted_price * (number_of_pins : Float)
  total_cost = 170 := 
by 
  sorry

end johns_total_spent_l60_60427


namespace faster_train_crossing_time_l60_60150

/-- Two trains are moving in the same direction at different speeds. 
    We define the speeds and lengths of the trains, and then prove the time it takes for
    the faster train to cross a man in the slower train given the conditions. -/

def speed_faster_train_kmph : ℝ := 90 -- speed in kmph
def speed_slower_train_kmph : ℝ := 36 -- speed in kmph
def length_faster_train_m : ℝ := 435 -- length in meters
def conversion_factor_kmph_to_mps : ℝ := 1000 / 3600 -- conversion factor from km/h to m/s

def speed_faster_train_mps : ℝ := speed_faster_train_kmph * conversion_factor_kmph_to_mps
def speed_slower_train_mps : ℝ := speed_slower_train_kmph * conversion_factor_kmph_to_mps

def relative_speed_mps : ℝ := speed_faster_train_mps - speed_slower_train_mps

def time_to_cross_seconds : ℝ := length_faster_train_m / relative_speed_mps

theorem faster_train_crossing_time :
  time_to_cross_seconds = 29 := by
  -- conditions given in the problem
  have h1 : speed_faster_train_kmph = 90 := rfl
  have h2 : speed_slower_train_kmph = 36 := rfl
  have h3 : length_faster_train_m = 435 := rfl
  have h4 : conversion_factor_kmph_to_mps = 1000 / 3600 := rfl
  -- speeds converted to m/s
  have h5 : speed_faster_train_mps = 25 := by
    simp [speed_faster_train_mps, h1, h4]
  have h6 : speed_slower_train_mps = 10 := by
    simp [speed_slower_train_mps, h2, h4]
  -- relative speed in m/s
  have h7 : relative_speed_mps = 15 := by
    simp [relative_speed_mps, h5, h6]
  -- time to cross in seconds
  have h8 : time_to_cross_seconds = 29 := by
    simp [time_to_cross_seconds, h3, h7]
  exact h8

end faster_train_crossing_time_l60_60150


namespace ratio_PM_MQ_l60_60471

theorem ratio_PM_MQ (A B C D E M P Q : Type) [Square A B C D] (side_length : Real) 
  (DE_length CE_length AE_length PM_length MQ_length : Real)
  (perpendicular_bisector_point: AE → AD → BC → Prop) :
  side_length = 12 ∧ DE_length = 5 ∧ (CE_length = side_length - DE_length) ∧
  (PM_length * 19 = MQ_length * 5) → PQ_length * 5 = MQ_length :=
by {
  sorry
}

end ratio_PM_MQ_l60_60471


namespace remaining_area_correct_l60_60229

-- Define the dimensions of the large field and the smaller dug-out patch
def field_length (x : ℝ) := x + 8
def field_width (x : ℝ) := x + 6
def dug_out_length (x : ℝ) := 2x - 4
def dug_out_width (x : ℝ) := x - 3

-- Define the area calculations
def field_area (x : ℝ) := field_length x * field_width x
def dug_out_area (x : ℝ) := dug_out_length x * dug_out_width x
def remaining_area (x : ℝ) := field_area x - dug_out_area x

-- Problem statement to prove
theorem remaining_area_correct (x : ℝ) : remaining_area x = -x^2 + 24*x + 36 :=
by sorry

end remaining_area_correct_l60_60229


namespace all_naturals_appear_in_sequence_l60_60264

/-- Define the sequence (a_n) where:
  * a_1 = 1
  * a_2 = 2
  * and each subsequent term is the smallest natural number that
    has not yet appeared in the sequence and is not coprime with 
    the previous term of the sequence. -/
def sequence : ℕ → ℕ
| 0     := 1
| 1     := 2
| (n+2) :=
  let prev := sequence (n+1) in 
  let next := Nat.find (λ m, m > prev ∧ 
                      m ≠ sequence n ∧ 
                      ¬ Nat.coprime m prev ∧ 
                      ∀ k < m, sequence k ≠ m) 
  in next

/-- Prove that every natural number appears in the sequence.-/
theorem all_naturals_appear_in_sequence : ∀ n : ℕ, ∃ k : ℕ, sequence k = n :=
by
  sorry

end all_naturals_appear_in_sequence_l60_60264


namespace complex_conjugate_of_z_l60_60686

theorem complex_conjugate_of_z (a : ℝ) (ha : a > 0) (hz : abs (a + (√3) * complex.I) = 2) : 
  complex.conj (a + (√3) * complex.I) = 1 - (√3) * complex.I :=
by sorry

end complex_conjugate_of_z_l60_60686


namespace delta_equivalence_l60_60633

-- Define the Δ operation
def Delta (a b : ℤ) : ℤ := a^3 - b^2

-- Define the expressions we need
def expr1 : ℤ := Delta 5 14
def expr2 : ℤ := Delta 4 6
def base1 : ℤ := 3 ^ expr1
def base2 : ℤ := 4 ^ expr2

-- The Lean goal statement
theorem delta_equivalence : Delta base1 base2 = -4^56 := by
  sorry

end delta_equivalence_l60_60633


namespace greatest_value_div_by_11_l60_60226

theorem greatest_value_div_by_11 : 
  ∃ (A B C : ℕ), A < 9 ∧ digit A ∧ digit B ∧ digit C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ (10000 * A + 1000 * B + 100 * C + 10 * B + A = 87978) :=
by
  sorry

end greatest_value_div_by_11_l60_60226


namespace find_special_numbers_l60_60176

/--
Consider a positive integer x with digit sequence aₙ, aₙ₋₁, ..., a₁, a₀.
We want to prove that the only x such that 
x = 1 + (aₙ^2 + aₙ₋₁^2 + ... + a₁^2 + a₀^2)
are 35 and 75.
-/
theorem find_special_numbers (x : ℕ) : 
  (∃ (aₙ aₙ₋₁ ... a₁ a₀ : ℕ), 
    aₙ * 10^ₙ + aₙ₋₁ * 10^(ₙ₋₁) + ... + a₁ * 10 + a₀ = x ∧ 
    x = 1 + (aₙ^2 + aₙ₋₁^2 + ... + a₁^2 + a₀^2)) → 
  x = 35 ∨ x = 75 := 
sorry

end find_special_numbers_l60_60176


namespace proposition_truth_l60_60708

noncomputable def propositions_correct : set ℕ :=
  {2, 4}

theorem proposition_truth :
  ∀ P1 P2 P3 P4 : Prop,
  (P1 = ∀ (l1 l2 : line) (π : plane), (l1 ∥ π) ∧ (l2 ∥ π) → (π ∥ π) ∧ (π ∥ π)) →
  (P2 = ∀ (π1 π2 : plane) (l : line), (l ⟂ π2) ∧ (l ∈ π1) → (π1 ⟂ π2)) →
  (P3 = ∀ (l1 l2 l : line), (l1 ⟂ l) ∧ (l2 ⟂ l) → (l1 ∥ l2)) →
  (P4 = ∀ (π1 π2 : plane) (l : line), (π1 ⟂ π2) ∧ (l ∈ π1) ∧ (¬ l ⟂ π1 ∩ π2) → (¬ l ⟂ π2)) →
  {n ∈ {1, 2, 3, 4} | (cond n P1 P2 P3 P4)} = propositions_correct
by 
  sorry

end proposition_truth_l60_60708


namespace reciprocal_neg_half_l60_60123

theorem reciprocal_neg_half : (1 / (- (1 / 2) : ℚ) = -2) :=
by
  sorry

end reciprocal_neg_half_l60_60123


namespace land_cost_is_50_l60_60218

noncomputable def land_cost_per_square_meter (L : ℝ) : Prop :=
  let 
    total_land_cost := 2000 * L
    total_bricks_cost := 10 * 100  -- 10000 bricks, $100 per 1000 bricks
    total_tiles_cost := 500 * 10  -- 500 tiles, $10 per tile
    total_cost := $106000
  in
    total_land_cost + total_bricks_cost + total_tiles_cost = total_cost

theorem land_cost_is_50 : land_cost_per_square_meter 50 :=
  by sorry

end land_cost_is_50_l60_60218


namespace area_formed_by_points_l60_60832

noncomputable def area_of_figure (x y : ℝ) : ℝ :=
if |4 * x| + |3 * y| + |24 - 4 * x - 3 * y| = 24 then area else 0

theorem area_formed_by_points :
  (∀ x y : ℝ, |4 * x| + |3 * y| + |24 - 4 * x - 3 * y| = 24 → x ≥ 0 ∧ y ≥ 0 ∧ 4 * x + 3 * y ≤ 24) →
  (4 * 0 * 0 + 3 * 0 * 0 + 0 * 0 - 48 = 0) →
  ∃ A : ℝ, A = 24 :=
by
  unfold area_of_figure
  sorry

end area_formed_by_points_l60_60832


namespace min_value_of_f_l60_60663

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) + (1 / (x^2 + (1 / x^2)))

theorem min_value_of_f : ∃ (c > 0), (∀ x > 0, f x ≥ f c) ∧ f c = 2.5 := by
  sorry

end min_value_of_f_l60_60663


namespace area_of_enclosed_region_l60_60637

def enclosed_region_area (x y : ℝ) : ℝ := x^2 + y^2 - 5 * |x + y| - 1

noncomputable def solution (x y : ℝ) : ℝ := π * (29 / 8)

theorem area_of_enclosed_region:
  ∀ x y : ℝ, x^2 + y^2 = 5 * |x + y| + 1 → solution x y = π * (29 / 8) :=
by
  intros x y h
  unfold solution
  sorry

end area_of_enclosed_region_l60_60637


namespace convex_pentagon_exists_l60_60137

theorem convex_pentagon_exists 
  (A : Set Point) 
  (hA : A.card = 9) 
  (hInc : ∀ (p1 p2 p3 : Point), p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → ¬Collinear ℝ ({p1, p2, p3} : Set Point)) 
  : ∃ (B : Set Point), B.card = 5 ∧ ConvexHull ℝ B = B :=
sorry

end convex_pentagon_exists_l60_60137


namespace original_number_increased_by_110_l60_60585

-- Define the conditions and the proof statement without the solution steps
theorem original_number_increased_by_110 {x : ℝ} (h : x + 1.10 * x = 1680) : x = 800 :=
by 
  sorry

end original_number_increased_by_110_l60_60585


namespace complex_number_properties_l60_60075

theorem complex_number_properties
  (x y : ℝ)
  (h : (1+complex.I) * x + (1-complex.I) * y = 2) :
  let z := x + y * complex.I in
  z.re = 1 ∧ complex.norm z = √2 ∧ (0 < z.re ∧ 0 < z.im) :=
by sorry

end complex_number_properties_l60_60075


namespace winning_post_distance_l60_60551

theorem winning_post_distance (v x : ℝ) (h₁ : x ≠ 0) (h₂ : v ≠ 0)
  (h₃ : 1.75 * v = v) 
  (h₄ : x = 1.75 * (x - 84)) : 
  x = 196 :=
by 
  sorry

end winning_post_distance_l60_60551


namespace valid_password_count_l60_60597

/-- 
The number of valid 4-digit ATM passwords at Fred's Bank, composed of digits from 0 to 9,
that do not start with the sequence "9,1,1" and do not end with the digit "5",
is 8991.
-/
theorem valid_password_count : 
  let total_passwords : ℕ := 10000
  let start_911 : ℕ := 10
  let end_5 : ℕ := 1000
  let start_911_end_5 : ℕ := 1
  total_passwords - (start_911 + end_5 - start_911_end_5) = 8991 :=
by
  let total_passwords : ℕ := 10000
  let start_911 : ℕ := 10
  let end_5 : ℕ := 1000
  let start_911_end_5 : ℕ := 1
  show total_passwords - (start_911 + end_5 - start_911_end_5) = 8991
  sorry

end valid_password_count_l60_60597


namespace simplify_expression_l60_60646

theorem simplify_expression (x : ℝ) : 24 * (3 * x - 4) - 6 * x = 66 * x - 96 := 
  sorry

end simplify_expression_l60_60646


namespace quadratic_equations_with_common_root_l60_60546

theorem quadratic_equations_with_common_root :
  ∃ (p1 q1 p2 q2 : ℝ),
    p1 ≠ p2 ∧ q1 ≠ q2 ∧
    ∀ x : ℝ,
      (x^2 + p1 * x + q1 = 0 ∧ x^2 + p2 * x + q2 = 0) →
      (x = 2 ∨ (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ ((x = r1 ∧ x == 2) ∨ (x = r2 ∧ x == 2)))) :=
sorry

end quadratic_equations_with_common_root_l60_60546


namespace find_x_given_y_l60_60734

theorem find_x_given_y (x : ℝ) : 
  (∀ y : ℝ, y = 2 → y = 3 / (5 * x + 4)) → 
  x = -1/2 :=
by
  intro h
  apply h 2
  sorry

end find_x_given_y_l60_60734


namespace candy_store_spending_l60_60368

def allowance : ℝ := 2.81
def spent_at_arcade (a : ℝ) : ℝ := (3 / 5) * a
def remaining_after_arcade (a : ℝ) : ℝ := a - spent_at_arcade a
def spent_at_toy_store (r : ℝ) : ℝ := (1 / 3) * r
def remaining_after_toy_store (a : ℝ) : ℝ := remaining_after_arcade a - spent_at_toy_store (remaining_after_arcade a)

theorem candy_store_spending : remaining_after_toy_store allowance ≈ 0.74933 := 
by
  sorry

end candy_store_spending_l60_60368


namespace limit_at_negative_third_l60_60457

theorem limit_at_negative_third:
  ∀ (f : ℝ → ℝ), 
    (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x + (1/3)| ∧ |x + (1/3)| < δ → |f x - (-5/3)| < ε)
  → (∀ x, f x = (6 * x^2 - x - 1) / (3 * x + 1))
  → filter.tendsto f (nhds (-1/3)) (nhds (-5/3)) := 
begin
    sorry
end

end limit_at_negative_third_l60_60457


namespace remaining_area_proof_l60_60548

-- Define the area of a semicircle (general formula)
def area_of_semicircle (r : ℝ) : ℝ :=
  (π / 2) * r^2

-- Define the total area of two semicircles given their diameters
def total_cut_out_area (a b : ℝ) : ℝ :=
  area_of_semicircle (a / 2) + area_of_semicircle (b / 2)

-- The remaining area after cutting out two semicircles from one semicircle
def remaining_area (a b : ℝ) : ℝ :=
  (area_of_semicircle (a + b) - total_cut_out_area a b)

-- Define the geometric mean theorem in our context:
def geom_mean (a b h : ℝ) (h_geom : h^2 = a * b) : ℝ :=
  a * b

-- Given lengths and perpendicular chord
variables (a b : ℝ) (h : ℝ) (h_given : h = 6) (h_geom_def : geom_mean a b h h_given = 36)

-- Statement of the theorem to be proved
theorem remaining_area_proof : remaining_area a b = 9 * π :=
  by sorry

end remaining_area_proof_l60_60548


namespace break_even_machines_l60_60632

def cost_parts : ℤ := 3600
def cost_patent : ℤ := 4500
def selling_price : ℤ := 180

def total_costs : ℤ := cost_parts + cost_patent

def machines_to_break_even : ℤ := total_costs / selling_price

theorem break_even_machines :
  machines_to_break_even = 45 := by
  sorry

end break_even_machines_l60_60632


namespace fraction_displayed_are_sculptures_l60_60251

noncomputable def total_pieces : ℕ := 1800
def not_displayed_pieces := (2 / 3 : ℚ) * total_pieces
def sculptures_not_displayed : ℚ := 800
def paintings_not_displayed := (1 / 3 : ℚ) * not_displayed_pieces
def sculptures_total := sculptures_not_displayed * (3 / 2 : ℚ)
def displayed_pieces := (1 / 3 : ℚ) * total_pieces
def sculptures_displayed := sculptures_total - sculptures_not_displayed

theorem fraction_displayed_are_sculptures :
  (sculptures_displayed / displayed_pieces) = (2 / 3 : ℚ) :=
sorry

end fraction_displayed_are_sculptures_l60_60251


namespace find_Pete_original_number_l60_60455

noncomputable def PeteOriginalNumber (x : ℝ) : Prop :=
  5 * (3 * x + 15) = 200

theorem find_Pete_original_number : ∃ x : ℝ, PeteOriginalNumber x ∧ x = 25 / 3 :=
by
  sorry

end find_Pete_original_number_l60_60455


namespace eccentricity_of_ellipse_l60_60707

theorem eccentricity_of_ellipse 
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (F : ℝ × ℝ) (A : ℝ × ℝ) (P : ℝ × ℝ)
  (hF : F = (-real.sqrt (a ^ 2 - b ^ 2), 0))
  (hA : A = (-a, 0))
  (hP_on_circle : P.1 ^ 2 + P.2 ^ 2 = b ^ 2)
  (constant_ratio : ∃ λ : ℝ, ∀ P, dist A P / dist F P = real.sqrt λ) :
  ∃ e : ℝ, 
    (0 < e ∧ e < 1) ∧ 
    e = (real.sqrt 5 - 1) / 2 :=
sorry

end eccentricity_of_ellipse_l60_60707


namespace circle_radius_circle_equation_l60_60019

-- Define the given conditions
def point_P : ℝ × ℝ := (real.sqrt 3, π / 6)
def line_l (ρ θ : ℝ) : Prop := ρ * real.sin (π / 3 - θ) = real.sqrt 3 / 2
def polar_center : ℝ × ℝ := (1, 0)

-- Define the goals to be proved
theorem circle_radius (P : ℝ × ℝ) (center : ℝ × ℝ)
  (hP : P = (real.sqrt 3, π / 6))
  (hC : center = (1,0))
  : P.dist center = 1 :=
sorry

theorem circle_equation (ρ θ : ℝ)
  (P : ℝ × ℝ) (center : ℝ × ℝ)
  (hP : P = (real.sqrt 3, π / 6))
  (hC : center = (1,0))
  : (ρ^2 - 2 * ρ * real.cos θ = 0) ↔ (ρ = 2 * real.cos θ) :=
sorry

end circle_radius_circle_equation_l60_60019


namespace probability_of_selecting_cooking_l60_60214

-- Define a type representing the courses.
inductive Course
| planting : Course
| cooking : Course
| pottery : Course
| carpentry : Course

-- Define the set of all courses
def all_courses : Finset Course := {Course.planting, Course.cooking, Course.pottery, Course.carpentry}

-- The condition that Xiao Ming randomly selects one of the four courses
def uniform_probability (s : Finset Course) (a : Course) : ℚ := 1 / s.card

-- Prove that the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking : uniform_probability all_courses Course.cooking = 1 / 4 :=
sorry

end probability_of_selecting_cooking_l60_60214


namespace min_value_expression_l60_60808

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (v : ℝ), (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
    8 * x^4 + 12 * y^4 + 18 * z^4 + 25 / (x * y * z) ≥ v) ∧ v = 30 :=
by
  sorry

end min_value_expression_l60_60808


namespace polynomial_classification_l60_60649

noncomputable def is_eventually_constant_mod (P : ℤ[X]) (n : ℕ) : Prop :=
∃ k, ∀ m ≥ k, (polynomial.eval (nat.iterate P m 0) n = polynomial.eval (nat.iterate P k 0) n)

def valid_polynomial (P : ℤ[X]) : Prop :=
∀ n : ℕ, n > 0 → is_eventually_constant_mod P n

theorem polynomial_classification :
  ∀ P : ℤ[X], valid_polynomial P ↔
    (∃ c : ℤ, P = polynomial.C c) ∨
    (∃ Q : ℤ[X], P = polynomial.X * Q) ∨
    (∃ R : ℤ[X], ∃ c : ℤ, P = polynomial.X * (polynomial.X - polynomial.C c) * R + polynomial.C c) ∨
    (∃ S : ℤ[X], ∃ c : ℤ, c ∈ {1, -1, 2, -2} ∧ P = (polynomial.X - polynomial.C c) * (polynomial.X + polynomial.C c) * (polynomial.X * S - polynomial.C (2 / c)) - polynomial.C c) :=
sorry

end polynomial_classification_l60_60649


namespace cost_of_each_box_of_pencils_l60_60234

-- Definitions based on conditions
def cartons_of_pencils : ℕ := 20
def boxes_per_carton_of_pencils : ℕ := 10
def cartons_of_markers : ℕ := 10
def boxes_per_carton_of_markers : ℕ := 5
def cost_per_carton_of_markers : ℕ := 4
def total_spent : ℕ := 600

-- Variable to define cost per box of pencils
variable (P : ℝ)

-- Main theorem to prove
theorem cost_of_each_box_of_pencils :
  cartons_of_pencils * boxes_per_carton_of_pencils * P + 
  cartons_of_markers * cost_per_carton_of_markers = total_spent → 
  P = 2.80 :=
by
  sorry

end cost_of_each_box_of_pencils_l60_60234


namespace problem_l60_60679

noncomputable def a : ℝ := 3

theorem problem : (27^2 = a^6) → a⁻² = 1 / 9 :=
by
  intro h
  sorry

end problem_l60_60679


namespace incenter_inequality_l60_60024

theorem incenter_inequality
  (A B C I A' : Type)
  [has_measure A B C I A']
  (h_triangle : is_triangle A B C)
  (h_incenter : incenter A B C I)
  (h_angle_bisector : angle_bisector A A' B C) :
  dist A I > dist A' I := sorry

end incenter_inequality_l60_60024


namespace sqrt_four_squared_l60_60172

theorem sqrt_four_squared : (Real.sqrt 4) ^ 2 = 4 :=
  by
    sorry

end sqrt_four_squared_l60_60172


namespace find_Z_l60_60141

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def midpoint (p1 p2 : Point3D) : Point3D :=
  Point3D.mk ((p1.x + p2.x) / 2) ((p1.y + p2.y) / 2) ((p1.z + p2.z) / 2)

theorem find_Z (W X Y Z : Point3D)
  (hW : W = ⟨4, 0, 3⟩)
  (hX : X = ⟨2, 3, -3⟩)
  (hY : Y = ⟨0, 2, 3⟩)
  (hWX_YZ : midpoint W Y = midpoint X Z) :
  Z = ⟨2, -1, 9⟩ :=
by
  sorry

end find_Z_l60_60141


namespace quadrilateral_inequality_l60_60482

noncomputable theory
open_locale classical

variables {A B C D M P Q : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M] [metric_space P] [metric_space Q]

def is_convex_quadrilateral (A B C D : Type*) := true  -- Define convex quadrilateral

def circumcenter (X Y Z : Type*) : Type* := sorry  -- Define circumcenter

def intersect (A B C D M : Type*) : Prop := sorry  -- Define intersection of diagonals

theorem quadrilateral_inequality
  (ABCD_convex : is_convex_quadrilateral A B C D)
  (intersect_M : intersect A B C D M)
  (P_center : P = circumcenter A B M)
  (Q_center : Q = circumcenter C D M)
  (AB CD PQ : ℝ) :
  AB + CD < 4 * PQ :=
begin
  sorry, -- The proof is omitted as per instructions.
end

end quadrilateral_inequality_l60_60482


namespace proof_problem_l60_60395

variable {A B C a b c : ℝ}
variable {m n : (ℝ × ℝ)}

def sin_B_plus_C := sin B + sin C
def sin_A_plus_B := sin A + sin B
def sin_B_minus_C := sin B - sin C

def m := (sin_B_plus_C, sin_A_plus_B)
def n := (sin_B_minus_C, sin A)

-- The condition that m is perpendicular to n
def m_perp_n : Prop := m.1 * n.1 + m.2 * n.2 = 0

-- Given C
noncomputable def measure_C : ℝ :=
  if m_perp_n then 2 * π / 3 else 0

-- The range of 2a + b
def range_2a_b (C : ℝ) (c : ℝ) : Set ℝ := 
  {y | ∃ (A B a b : ℝ), 0 < A ∧ A < π / 3 ∧ 0 < B ∧ B < π / 3 ∧ 
    c = sqrt 3 ∧ y = 2 * sqrt 3 * sin (A + π / 6) ∧ 
    sqrt 3 < y ∧ y < 2 * sqrt 3}

-- The theorem statement
theorem proof_problem (h1 : m_perp_n) (h2 : c = sqrt 3) : 
  measure_C = 2 * π / 3 ∧ 
  range_2a_b (2 * π / 3) (sqrt 3) = { y | sqrt 3 < y ∧ y < 2 * sqrt 3 } :=
by 
  sorry

end proof_problem_l60_60395


namespace complete_the_square_k_l60_60008

theorem complete_the_square_k (x : ℝ) : ∃ k : ℝ, (∃ a h: ℝ, (a = 1) ∧ (h = 7 / 2) ∧ (x^2 - 7*x = a * (x - h)^2 + k)) → k = -49 / 4 :=
by
  sorry

end complete_the_square_k_l60_60008


namespace avg_speed_l60_60989

theorem avg_speed (Δt : ℝ) (h : Δt ≠ 0) : 
  let s := λ t : ℝ, t^2 + 3 in 
  (s (3 + Δt) - s 3) / Δt = 6 + Δt :=
by 
  let s := λ t : ℝ, t^2 + 3 
  have s_3 : s 3 = 12 := by norm_num
  have s_3Δt : s (3 + Δt) = 9 + 6 * Δt + Δt^2 + 3 := by simp [s, pow_two]
  calc 
    (s (3 + Δt) - s 3) / Δt
      = (9 + 6 * Δt + Δt^2 + 3 - 12) / Δt : by rw [s_3, s_3Δt]
  ... = (6 * Δt + Δt^2) / Δt           : by norm_num
  ... = 6 + Δt                         : by field_simp [h]

end avg_speed_l60_60989


namespace correct_derivatives_count_l60_60870

theorem correct_derivatives_count :
  let d1 := (fun x : ℝ => (3 : ℝ)^x)
  let d1' := (fun x : ℝ => (3 : ℝ)^x * real.log 3)
  let d1_incorrect := d1' ≠ (fun x : ℝ => (3 : ℝ)^x * real.log 3)
  let d2 := (fun x : ℝ => real.log x / real.log 2)
  let d2' := (fun x : ℝ => 1 / (x * real.log 2))
  let d2_correct := d2' = 1 / (Real.mul x (real.log 2))
  let d3 := (fun x : ℝ => real.exp x)
  let d3_correct := (fun x : ℝ => d3 x) = (fun x : ℝ => real.exp x)
  let d4 := (fun x : ℝ => x * real.exp x)
  let d4' := (fun x : ℝ => real.exp x + 1)
  let d4_incorrect := d4' ≠ (fun x : ℝ => real.exp x + x * real.exp x)
  (d1_incorrect + d2_correct + d3_correct + d4_incorrect) = 2 := 
by sorry

end correct_derivatives_count_l60_60870


namespace simplify_expression_l60_60465

variable (m n : ℝ)

theorem simplify_expression : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end simplify_expression_l60_60465


namespace value_of_x_l60_60382

theorem value_of_x (x : ℝ) : (1 / 8) * (2 : ℝ) ^ 32 = (4 : ℝ) ^ x → x = 29 / 2 :=
by
  sorry

end value_of_x_l60_60382


namespace bar_graph_represents_circle_graph_l60_60978

theorem bar_graph_represents_circle_graph (r b g : ℕ) 
  (h1 : r = g) 
  (h2 : b = 3 * r) : 
  (r = 1 ∧ b = 3 ∧ g = 1) :=
sorry

end bar_graph_represents_circle_graph_l60_60978


namespace range_f_n_l60_60359

noncomputable def f1 (x : ℝ) (hx : x > 0) : ℝ := x / (x + 3)

noncomputable def f_n (n : ℕ) : (ℝ → ℝ) :=
  if h : n = 0 then id else 
    let rec : (ℕ → (ℝ → ℝ)) := λ m, if m = 0 then id else 
      λ x, f1 (rec (m - 1) x) (by sorry) in 
    rec n

theorem range_f_n (n : ℕ) (hn : n > 0) : set.range (f_n n) = set.Ioo (0 : ℝ) (2 / (3 ^ n - 1)) :=
sorry

end range_f_n_l60_60359


namespace sufficient_not_necessary_not_necessary_l60_60560

theorem sufficient_not_necessary (x : ℝ) (h1: x > 2) : x^2 - 3 * x + 2 > 0 :=
sorry

theorem not_necessary (x : ℝ) (h2: x^2 - 3 * x + 2 > 0) : (x > 2 ∨ x < 1) :=
sorry

end sufficient_not_necessary_not_necessary_l60_60560


namespace log_base_5_eq_l60_60129

theorem log_base_5_eq (x : ℝ) : 5 ^ x = 3 → x = log 3 / log 5 :=
by sorry

end log_base_5_eq_l60_60129


namespace correct_operation_l60_60164

theorem correct_operation : 
  (∀ x y : ℝ, (x-y)² ≠ x² + y²) ∧
  (∀ a : ℝ, a² + a² ≠ a⁴) ∧
  (∀ x y : ℝ, -x² * y³ * (2 * x * y²) = -2 * x³ * y⁵) ∧ 
  (∀ x : ℝ, (x³)² ≠ x⁵) :=
by 
  sorry

end correct_operation_l60_60164


namespace pure_alcohol_addition_problem_l60_60550

-- Define the initial conditions
def initial_volume := 6
def initial_concentration := 0.30
def final_concentration := 0.50

-- Define the amount of pure alcohol to be added
def x := 2.4

-- Proof problem statement
theorem pure_alcohol_addition_problem (initial_volume initial_concentration final_concentration x : ℝ) :
  initial_volume * initial_concentration + x = final_concentration * (initial_volume + x) :=
by
  -- Initial condition values definition
  let initial_volume := 6
  let initial_concentration := 0.30
  let final_concentration := 0.50
  let x := 2.4
  -- Skip the proof
  sorry

end pure_alcohol_addition_problem_l60_60550


namespace walk_westward_denotes_negative_l60_60821

-- Define the direction as a set condition
def direction := "east" → "positive"

-- Prove that walking 10m westward is -10m under the given direction condition
theorem walk_westward_denotes_negative (h: direction "east" = "positive") : -10 = -10 :=
by
  sorry

end walk_westward_denotes_negative_l60_60821


namespace inequality_T_l60_60365

-- Define the sequence a_n with the given recurrence relationship and initial condition
def a : ℕ → ℚ
| 1       := 4 / 9
| (n+1) := (n+1) / n * a n - (1 / 3) * (n+1) * (2 / 3)^(n+1)

-- General formula for the sequence a_n
def general_formula (n : ℕ) : ℚ :=
  if h : n > 0 then n * (2 / 3)^(n + 1) else 0

-- Relation for the sequence b_n
def b (n : ℕ) : ℚ :=
  let an := if n > 0 then a n else 0 in
  if h : n > 0 then (n - an) / (3 * n - 2 * an) else 0

-- Sum of the first n terms of the sequence b_n
def T (n : ℕ) : ℚ :=
  (Finset.range n).sum b

-- Main theorem to prove the given inequality for T_n
theorem inequality_T (n : ℕ) (hn : n > 0) :
  (3 * n - 4) / 9 < T n ∧ T n < n / 3 :=
sorry

end inequality_T_l60_60365


namespace intersection_of_A_and_B_l60_60694

-- Definitions
def A := {x : ℤ | x ≤ 0}
def B := {-2, -1, 0, 1, 2}

-- Theorem
theorem intersection_of_A_and_B : A ∩ B = {-2, -1, 0} := 
by
  sorry

end intersection_of_A_and_B_l60_60694


namespace cone_volume_l60_60133

theorem cone_volume (V_cyl : ℝ) (r h : ℝ) (h_cyl : V_cyl = 150 * Real.pi) :
  (1 / 3) * V_cyl = 50 * Real.pi :=
by
  rw [h_cyl]
  ring


end cone_volume_l60_60133


namespace constant_term_of_expansion_l60_60098

-- Define the conditions and constants
def binomial_coefficient_is_largest (n : ℕ) : Prop :=
  let k := 4 in 
  ∀ m ≠ k, (Nat.choose n k > Nat.choose n m)

theorem constant_term_of_expansion : 
  binomial_coefficient_is_largest 8 →
  let expr := (x - 1/x) ^ 8 in
  (∃ c : ℕ, c = 70) :=
by {
  sorry
}

end constant_term_of_expansion_l60_60098


namespace student_failed_by_100_marks_l60_60243

theorem student_failed_by_100_marks 
  (max_marks : ℕ)
  (passing_percentage : ℝ)
  (student_score : ℕ)
  (max_marks_eq : max_marks = 300)
  (passing_percentage_eq : passing_percentage = 0.60)
  (student_score_eq : student_score = 80) :
  (0.60 * 300 - 80) = 100 :=
by
  rw [←passing_percentage_eq, ←max_marks_eq, ←student_score_eq]
  norm_num
  sorry

end student_failed_by_100_marks_l60_60243


namespace mathe_matics_equals_2014_l60_60842

/-- 
Given the following mappings for characters in the word "MATHEMATICS":
M = 1, A = 8, T = 3, E = '+', I = 9, K = '-',
verify that the resulting numerical expression 183 + 1839 - 8 equals 2014.
-/
theorem mathe_matics_equals_2014 :
  183 + 1839 - 8 = 2014 :=
by
  sorry

end mathe_matics_equals_2014_l60_60842


namespace quadratic_roots_difference_l60_60625

theorem quadratic_roots_difference (p q : ℝ) (hp : 0 < p) (hq : 0 < q) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 - x2 = 2 ∧ x1 * x2 = q ∧ x1 + x2 = -p) → p = 2 * Real.sqrt (q + 1) :=
by
  sorry

end quadratic_roots_difference_l60_60625


namespace area_of_isosceles_trapezoid_with_inscribed_circle_l60_60293

-- Definitions
def is_isosceles_trapezoid_with_inscribed_circle (a b c : ℕ) : Prop :=
  a + b = 2 * c

def height_of_trapezoid (c : ℕ) (half_diff_bases : ℕ) : ℕ :=
  (c^2 - half_diff_bases^2).sqrt

noncomputable def area_of_trapezoid (a b h : ℕ) : ℕ :=
  (a + b) * h / 2

-- Given values
def base1 := 2
def base2 := 8
def leg := 5
def height := 4

-- Proof Statement
theorem area_of_isosceles_trapezoid_with_inscribed_circle :
  is_isosceles_trapezoid_with_inscribed_circle base1 base2 leg →
  height_of_trapezoid leg ((base2 - base1) / 2) = height →
  area_of_trapezoid base1 base2 height = 20 :=
by
  intro h1 h2
  sorry

end area_of_isosceles_trapezoid_with_inscribed_circle_l60_60293


namespace richard_sold_booklets_in_15_days_l60_60462

theorem richard_sold_booklets_in_15_days :
  let b : ℕ → ℕ := λ n, 3 * n - 1 in 
  ∑ i in Finset.range 15, b (i + 1) = 345 := by
  sorry

end richard_sold_booklets_in_15_days_l60_60462


namespace geometric_sequence_example_l60_60767

theorem geometric_sequence_example (r s : ℕ) (a_n : ℕ → ℝ) (hr_ne_hs : r ≠ s) 
  (ha_r_eq_ha_s : a_n r = a_n s) (hr_pos : r > 0) (hs_pos : s > 0)
  (arith_seq : ∀ r s, r ≠ s → a_n r = a_n s → ∀ d, ∀ a₀, a_n = λ n, a₀ + n * d → d = 0 → ∃ c, ∀ n, a_n n = c)
  (geom_seq : ∀ r s, r ≠ s → a_n r = a_n s → ∀ q, ∀ a₀, a_n = λ n, a₀ * q ^ n → ¬(q = 1) → q = -1 → 
    ∃ b₀, ∃ b₁, ∀ n, a_n = λ n, if n % 2 = 0 then b₀ else b₁) :
  ∃ b₀ b₁, ∀ n, a_n n = if n % 2 = 0 then b₀ else b₁ :=
begin
  sorry
end

end geometric_sequence_example_l60_60767


namespace correct_operation_among_options_l60_60165

theorem correct_operation_among_options (A B C D : Prop) (cond_A : A = (sqrt 4 = ±2))
  (cond_B : B = (sqrt 4)^2 = 4) (cond_C : C = (sqrt (-4)^2) = -4) (cond_D : D = (-sqrt 4)^2 = -4) :
  B ∧ ¬A ∧ ¬C ∧ ¬D :=
by
  sorry

end correct_operation_among_options_l60_60165


namespace interval_of_monotonic_increase_and_cos2x0_value_l60_60710

theorem interval_of_monotonic_increase_and_cos2x0_value 
  (f : ℝ → ℝ) 
  (x0 : ℝ) 
  (hx0 : 0 ≤ x0 ∧ x0 ≤ π/2) 
  (hf : f = fun x => (sin x + (sqrt 3) * cos x) * (cos x - (sqrt 3) * sin x)) 
  (hf_at_x0 : f x0 = 6/5) :
  (∀ k : ℤ, (π * k - 7 * π / 12 ≤ x0 ∧ x0 ≤ π * k - π / 12)) ∧ (cos (2 * x0) = (4 + 3 * sqrt 3) / 10) :=
by
  sorry

end interval_of_monotonic_increase_and_cos2x0_value_l60_60710


namespace apples_chosen_l60_60817

def total_fruits : ℕ := 12
def bananas : ℕ := 4
def oranges : ℕ := 5
def total_other_fruits := bananas + oranges

theorem apples_chosen : total_fruits - total_other_fruits = 3 :=
by sorry

end apples_chosen_l60_60817


namespace sum_of_squares_l60_60475

def satisfies_conditions (x y z : ℕ) : Prop :=
  x + y + z = 24 ∧
  Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10

theorem sum_of_squares (x y z : ℕ) (h : satisfies_conditions x y z) :
  ∀ (x y z : ℕ), x + y + z = 24 ∧ Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10 →
  x^2 + y^2 + z^2 = 216 :=
sorry

end sum_of_squares_l60_60475


namespace trains_crossing_time_l60_60520

theorem trains_crossing_time :
  let length_first_train := 500
  let length_second_train := 800
  let speed_first_train := 80 * (5/18 : ℚ)  -- convert km/hr to m/s
  let speed_second_train := 100 * (5/18 : ℚ)  -- convert km/hr to m/s
  let relative_speed := speed_first_train + speed_second_train
  let total_distance := length_first_train + length_second_train
  let time_taken := total_distance / relative_speed
  time_taken = 26 :=
by
  sorry

end trains_crossing_time_l60_60520


namespace cost_price_is_700_l60_60186

noncomputable def cost_price_was_700 : Prop :=
  ∃ (CP : ℝ),
    (∀ (SP1 SP2 : ℝ),
      SP1 = CP * 0.84 ∧
        SP2 = CP * 1.04 ∧
        SP2 = SP1 + 140) ∧
    CP = 700

theorem cost_price_is_700 : cost_price_was_700 :=
  sorry

end cost_price_is_700_l60_60186


namespace sector_area_l60_60701

theorem sector_area (r θ l : ℝ) (h1 : θ = 2) (h2: l = 2) (h3: l = r * θ) : 
  let S := (1 / 2) * r^2 * θ
  in S = 1 :=
by
  sorry

end sector_area_l60_60701


namespace reciprocal_of_minus_one_half_l60_60122

theorem reciprocal_of_minus_one_half : (1 / (-1 / 2)) = -2 := 
by sorry

end reciprocal_of_minus_one_half_l60_60122


namespace probability_P8_equals_P_l60_60991

theorem probability_P8_equals_P : 
  let k_even_combinations := [Finset.card (Finset.filter (λ k, k % 2 = 0) (Finset.range 9))],
  let total_sequences := 4 ^ 8,
  P_8_eq_P := (Finset.sum k_even_combinations (λ k, Nat.binomial 8 k ^ 2)),
  P_8_eq_P / total_sequences = 1225 / 16384 :=
by
  let k_even_combinations := List.map (λ k, Nat.choose 8 k * Nat.choose 8 k) [0, 2, 4, 6, 8]
  let valid_sequences := List.sum k_even_combinations
  let total_sequences := 4 ^ 8
  have h : valid_sequences = 6470 := by norm_num -- sum of squared binomials
  have h2 : total_sequences = 65536 := by norm_num -- 4^8
  have P_8_eq_P : (valid_sequences : ℚ) / (total_sequences : ℚ) = 1225 / 16384 := by
    rw [h, h2]
    norm_num
  exact P_8_eq_P

end probability_P8_equals_P_l60_60991


namespace number_of_nintendo_games_to_give_away_l60_60033

-- Define the conditions
def initial_nintendo_games : ℕ := 20
def desired_nintendo_games_left : ℕ := 12

-- Define the proof problem as a Lean theorem
theorem number_of_nintendo_games_to_give_away :
  initial_nintendo_games - desired_nintendo_games_left = 8 :=
by
  sorry

end number_of_nintendo_games_to_give_away_l60_60033


namespace gcd_of_three_numbers_l60_60298

theorem gcd_of_three_numbers (a b c : ℕ) (h1 : a = 15378) (h2 : b = 21333) (h3 : c = 48906) :
  Nat.gcd (Nat.gcd a b) c = 3 :=
by
  rw [h1, h2, h3]
  sorry

end gcd_of_three_numbers_l60_60298


namespace total_sales_is_correct_l60_60764

-- Definitions of prices per pencil type
def price_pencil_eraser : ℝ := 0.8
def price_regular_pencil : ℝ := 0.5
def price_short_pencil : ℝ := 0.4
def price_mechanical_pencil : ℝ := 1.2
def price_novelty_pencil : ℝ := 1.5

-- Definitions of discounts and offers
def discount_for_pencil_eraser (quantity : ℕ) (price : ℝ) : ℝ :=
  if quantity >= 100 then 0.9 * price else price

def total_with_discount (quantity : ℕ) (price_per_unit : ℝ) : ℝ :=
  discount_for_pencil_eraser quantity (quantity * price_per_unit)

def price_mechanical_pencil_with_offer (quantity : ℕ) (price_per_unit : ℝ) : ℝ :=
  let paid_pencils := quantity - quantity / 4 -- since buy 3 get 1 free
  in paid_pencils * price_per_unit

-- Sales quantities
def quantity_pencils_eraser : ℕ := 200
def quantity_regular_pencils : ℕ := 40
def quantity_short_pencils : ℕ := 35
def quantity_mechanical_pencils : ℕ := 25
def quantity_free_mechanical_pencils : ℕ := 5
def quantity_novelty_pencils : ℕ := 15

-- Total sales calculations
def total_sales_pencils_eraser := total_with_discount quantity_pencils_eraser price_pencil_eraser
def total_sales_regular_pencils := quantity_regular_pencils * price_regular_pencil
def total_sales_short_pencils := quantity_short_pencils * price_short_pencil
def total_sales_mechanical_pencils := price_mechanical_pencil_with_offer quantity_mechanical_pencils price_mechanical_pencil
def total_sales_novelty_pencils := quantity_novelty_pencils * price_novelty_pencil

def total_sales := 
  total_sales_pencils_eraser + 
  total_sales_regular_pencils + 
  total_sales_short_pencils + 
  total_sales_mechanical_pencils + 
  total_sales_novelty_pencils

theorem total_sales_is_correct : 
  total_sales = 224.5 :=
by 
  sorry

end total_sales_is_correct_l60_60764


namespace parallelogram_properties_l60_60586

noncomputable def length_adjacent_side_and_area (base height : ℝ) (angle : ℕ) : ℝ × ℝ :=
  let hypotenuse := height / Real.sin (angle * Real.pi / 180)
  let area := base * height
  (hypotenuse, area)

theorem parallelogram_properties :
  ∀ (base height : ℝ) (angle : ℕ),
  base = 12 → height = 6 → angle = 30 →
  length_adjacent_side_and_area base height angle = (12, 72) :=
by
  intros
  sorry

end parallelogram_properties_l60_60586


namespace cube_edge_length_proof_l60_60981

-- Define the edge length of the cube
def edge_length_of_cube := 15

-- Define the volume of the cube
def volume_of_cube (a : ℕ) := a^3

-- Define the volume of the displaced water
def volume_of_displaced_water := 20 * 15 * 11.25

-- The theorem to prove
theorem cube_edge_length_proof : ∃ a : ℕ, volume_of_cube a = 3375 ∧ a = edge_length_of_cube := 
by {
  sorry
}

end cube_edge_length_proof_l60_60981


namespace intersection_eq_l60_60742

noncomputable def A : Set ℝ := {x | ∃ y : ℝ, y^2 = x}
noncomputable def B : Set ℝ := {y | ∃ x : ℝ, y = sin x}

theorem intersection_eq : (A ∩ B) = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end intersection_eq_l60_60742


namespace sufficient_and_necessary_l60_60317

theorem sufficient_and_necessary (a b : ℝ) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end sufficient_and_necessary_l60_60317


namespace minimum_blue_beads_l60_60984

theorem minimum_blue_beads (n : ℕ) (hr : n = 100) (h : ∀ s : finset ℕ, s.card = 10 → ∃ t : finset ℕ, t.card ≥ 7 ∧ t ⊆ {x | x ∉ s}) : ∃ k : ℕ, k = 78 :=
sorry

end minimum_blue_beads_l60_60984


namespace range_of_a_l60_60689

noncomputable def g (x : ℝ) : ℝ := sorry

-- Conditions for g(x)
axiom g_continuous : ∀ x : ℝ, continuous_at g x
axiom g_deriv_pos : ∀ x : ℝ, x > 0 → deriv g x > 0
axiom g_even : ∀ x : ℝ, g x = g (-x)

-- Definition of f(x)
noncomputable def f (x : ℝ) : ℝ :=
if x ∈ [0, sqrt 3] then x^3 - 3*x else
  if x ∈ [-sqrt 3, 0] then -x^3 - 3*sqrt 3*x^2 - 6*x else
    sorry

-- Condition for f(x)
axiom f_property : ∀ x : ℝ, f (sqrt 3 + x) = -f x

-- The inequality condition
axiom g_f_inequality : ∀ x : ℝ, x ∈ [-3, 3] → g (f x) ≤ g (a^2 - a + 2)

-- The goal
theorem range_of_a (a : ℝ) : (a ≤ 0 ∨ a ≥ 1) :=
sorry

end range_of_a_l60_60689


namespace probability_of_one_climber_l60_60210

def enthusiasts : Type := {a : unit // a = unit.star}
def enthusiasts_have_climbed_everest (a : enthusiasts) : Prop :=
  a.val = unit.star ∨ a.val = unit.star

noncomputable def number_of_ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

def count_favorable_events : ℕ :=
  number_of_ways_to_choose 3 1 -- remaining 3 who haven't climbed

def count_total_events : ℕ :=
  number_of_ways_to_choose 5 2

theorem probability_of_one_climber :
  (count_favorable_events * number_of_ways_to_choose 2 1) / count_total_events = 3 / 5 :=
by
  sorry

end probability_of_one_climber_l60_60210


namespace area_formed_by_points_l60_60831

noncomputable def area_of_figure (x y : ℝ) : ℝ :=
if |4 * x| + |3 * y| + |24 - 4 * x - 3 * y| = 24 then area else 0

theorem area_formed_by_points :
  (∀ x y : ℝ, |4 * x| + |3 * y| + |24 - 4 * x - 3 * y| = 24 → x ≥ 0 ∧ y ≥ 0 ∧ 4 * x + 3 * y ≤ 24) →
  (4 * 0 * 0 + 3 * 0 * 0 + 0 * 0 - 48 = 0) →
  ∃ A : ℝ, A = 24 :=
by
  unfold area_of_figure
  sorry

end area_formed_by_points_l60_60831


namespace min_value_of_expression_l60_60698

theorem min_value_of_expression (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y)
    (h3 : 4 * x^2 + 4 * x * y + y^2 + 2 * x + y - 6 = 0) : 
    ∃ (c : ℝ), c = x * (1 - y) ∧ c = - (1 / 8) :=
by
  sorry

end min_value_of_expression_l60_60698


namespace value_of_expression_l60_60728

theorem value_of_expression
  (a b c : ℕ)
  (H1 : {a, b, c} = {0, 1, 2})
  (H2 : (a ≠ 2 ∧ b ≠ 2 ∧ c = 0) ∨ (a = 2 ∧ b = 0 ∧ c ≠ 0) ∨ (a = 0 ∧ b = 2 ∧ c ≠ 0)) :
  10 * a + 2 * b + c = 21 :=
by
  -- Proof goes here
  sorry

end value_of_expression_l60_60728


namespace presidency_meeting_ways_l60_60217

theorem presidency_meeting_ways : 
  ∃ (ways : ℕ), ways = 4 * 6 * 3 * 225 := sorry

end presidency_meeting_ways_l60_60217


namespace smallest_n_l60_60643

theorem smallest_n (o y v : ℕ) (h1 : 18 * o = 21 * y) (h2 : 21 * y = 10 * v) (h3 : 10 * v = 30 * n) : 
  n = 21 := by
  sorry

end smallest_n_l60_60643


namespace tarun_garden_area_l60_60142

theorem tarun_garden_area :
  ∀ (side : ℝ), 
  (1500 / 8 = 4 * side) → 
  (30 * side = 1500) → 
  side^2 = 2197.265625 :=
by
  sorry

end tarun_garden_area_l60_60142


namespace count_incorrect_propositions_l60_60498

-- Define the Propositions
def prop1 (a b : Vector) : Prop := (a ∥ b) → ∃! λ : ℝ, a = λ • b
def prop2 (a b : Vector) (a_nonzero b_nonzero : ¬(a = 0) ∧ ¬(b = 0)) : Prop :=
  (angle a b > (π / 2)) ↔ (dot_product a b < 0)
def prop3 (θ : ℝ) : Prop :=
  (θ ≠ π/3) → (cos θ ≠ 1/2) = ¬((θ = π/3) → (cos θ = 1/2))
def prop4 : Prop :=
  (∃ x : ℝ, x^2 - x + 1 < 0) ↔ ∀ x : ℝ, x^2 - x + 1 > 0
def prop5 (A B C : Point) : Prop :=
  (triangle_is_right A B C) ↔ (cos_angle B = sin_angle A)

-- The main theorem
theorem count_incorrect_propositions : 
  (∃ a b : Vector, ¬(prop1 a b)) ∧ 
  (∃ a b : Vector, ¬(prop2 a b true true)) ∧ 
  (¬prop3 (π/3)) ∧ 
  (¬prop4) ∧ 
  (∃ A B C : Point, ¬(prop5 A B C)) →
  count_incorrect = 4 := by sorry

end count_incorrect_propositions_l60_60498


namespace count_triangles_in_figure_l60_60375

-- Define the structure of the grid with the given properties.
def grid_structure : Prop :=
  ∃ (n1 n2 n3 n4 : ℕ), 
  n1 = 3 ∧  -- First row: 3 small triangles
  n2 = 2 ∧  -- Second row: 2 small triangles
  n3 = 1 ∧  -- Third row: 1 small triangle
  n4 = 1    -- 1 large inverted triangle

-- The problem statement
theorem count_triangles_in_figure (h : grid_structure) : 
  ∃ (total_triangles : ℕ), total_triangles = 9 :=
sorry

end count_triangles_in_figure_l60_60375


namespace area_increase_is_225_percent_l60_60558

namespace CircularField

def radius_ratio := 2 / 5

-- Define the radii of the two fields based on the given ratio
def r1 : ℝ := 1  -- Assume r1 = 1 for simplification
def r2 : ℝ := r1 * (5 / 2)

-- Define the areas of the two fields
def A1 : ℝ := Real.pi * r1^2
def A2 : ℝ := Real.pi * r2^2

-- Calculate the percentage increase in area
def percentage_increase : ℝ := ((A2 - A1) / A1) * 100

theorem area_increase_is_225_percent :
  percentage_increase = 225 := 
sorry

end CircularField

end area_increase_is_225_percent_l60_60558


namespace find_number_l60_60207

theorem find_number (x : ℕ) (h1 : x = 3927) : 9873 + x = 13800 :=
by
  rw [h1]
  norm_num
  sorry

end find_number_l60_60207


namespace substance_volume_proportional_l60_60195

theorem substance_volume_proportional (k : ℝ) (V₁ V₂ : ℝ) (W₁ W₂ : ℝ) 
  (h1 : V₁ = k * W₁) 
  (h2 : V₂ = k * W₂) 
  (h3 : V₁ = 48) 
  (h4 : W₁ = 112) 
  (h5 : W₂ = 84) 
  : V₂ = 36 := 
  sorry

end substance_volume_proportional_l60_60195


namespace max_value_l60_60045

noncomputable def max_sum_reciprocals (x y a b : ℝ) : ℝ :=
  if h : (a > 1) ∧ (b > 1) ∧ (a ^ x = 3) ∧ (b ^ y = 3) ∧ (a + b = 2 * (3)^(1/2)) then
    (1 / x) + (1 / y)
  else 0

theorem max_value :
  ∀ (x y a b : ℝ), (a > 1) ∧ (b > 1) ∧ (a ^ x = 3) ∧ (b ^ y = 3) ∧ (a + b = 2 * (3:ℝ)^(1/2))
  → max_sum_reciprocals x y a b = 1 :=
begin
  intros x y a b h,
  sorry
end

end max_value_l60_60045


namespace height_of_table_l60_60930

variable (h l w h3 : ℝ)

-- Conditions from the problem
def condition1 : Prop := h3 = 4
def configurationA : Prop := l + h - w = 50
def configurationB : Prop := w + h + h3 - l = 44

-- Statement to prove
theorem height_of_table (h l w h3 : ℝ) 
  (cond1 : condition1 h3)
  (confA : configurationA h l w)
  (confB : configurationB h l w h3) : 
  h = 45 := 
by 
  sorry

end height_of_table_l60_60930


namespace train_crosses_platform_in_43_18_seconds_l60_60968

def length_of_train := 140 -- in meters
def length_of_platform := 520 -- in meters
def speed_of_train_kmh := 55 -- in km/hr
def kmh_to_ms_conversion := 1000 / 3600 -- conversion factor from km/hr to m/s

def total_distance := length_of_train + length_of_platform -- total distance in meters
def speed_of_train_ms := speed_of_train_kmh * kmh_to_ms_conversion -- speed in meters/second

def time_to_cross := total_distance / speed_of_train_ms -- time in seconds

theorem train_crosses_platform_in_43_18_seconds :
  abs (time_to_cross - 43.18) < 0.01 := -- allowing for a small rounding difference
by
  sorry

end train_crosses_platform_in_43_18_seconds_l60_60968


namespace find_second_number_l60_60931

theorem find_second_number 
    (lcm : ℕ) (gcf : ℕ) (num1 : ℕ) (num2 : ℕ)
    (h_lcm : lcm = 56) (h_gcf : gcf = 10) (h_num1 : num1 = 14) 
    (h_product : lcm * gcf = num1 * num2) : 
    num2 = 40 :=
by
  sorry

end find_second_number_l60_60931


namespace BD_value_l60_60407

def quadrilateral_ABCD_sides (AB BC CD DA : ℕ) (BD : ℕ) : Prop :=
  AB = 5 ∧ BC = 17 ∧ CD = 5 ∧ DA = 9 ∧ 12 < BD ∧ BD < 14 ∧ BD = 13

theorem BD_value (AB BC CD DA : ℕ) (BD : ℕ) : 
  quadrilateral_ABCD_sides AB BC CD DA BD → BD = 13 :=
by
  sorry

end BD_value_l60_60407


namespace probability_multiple_of_4_or_5_l60_60955

theorem probability_multiple_of_4_or_5 :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 99 →
  let q := (Finset.card (Finset.filter (λ n, (4 ∣ n * (n + 1)) ∨ (5 ∣ n * (n + 1))) (Finset.range 100))) / 99 in
  q = 2/5 :=
by
  intros x hx
  let q := (Finset.card (Finset.filter (λ n, (4 ∣ n * (n + 1)) ∨ (5 ∣ n * (n + 1))) (Finset.range 100))) / 99
  sorry

end probability_multiple_of_4_or_5_l60_60955


namespace correct_operation_among_options_l60_60167

theorem correct_operation_among_options (A B C D : Prop) (cond_A : A = (sqrt 4 = ±2))
  (cond_B : B = (sqrt 4)^2 = 4) (cond_C : C = (sqrt (-4)^2) = -4) (cond_D : D = (-sqrt 4)^2 = -4) :
  B ∧ ¬A ∧ ¬C ∧ ¬D :=
by
  sorry

end correct_operation_among_options_l60_60167
