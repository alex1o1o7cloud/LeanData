import Data.Int.Basic
import Data.Real.Basic
import Mathlib
import Mathlib.Algebra.ArithmeticSeq
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Linear.Basic
import Mathlib.Algebra.Order.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Quadratics
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Monotonicity
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Trigonometry
import Mathlib.Combinatorics.CombinatorialProofs
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Interval
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Probability.ProbabilitySpace
import Mathlib.Tactic
import Mathlib.Tactic.Sorry
import Mathlib.Topology.Basic

namespace min_ratio_bd_l353_353310

noncomputable def triangle_ratio_min (BD : ℝ) : Prop :=
  ∃ K L : ℝ, 
    (K = 1) ∧ 
    (L = 1) ∧ 
    (∀ (x : ℝ), ∀ (y : ℝ), (CD = 2 * BD) → ((x = sqrt(3) - 1) → y = (bd + 1 + 3 / (bd + 1)) ∧ (x + y) = 2 * K * L) ∧ 
    (abs x ≤ (sqrt 3 - 1)))


theorem min_ratio_bd (BD : ℝ) : triangle_ratio_min BD :=
by { sorry }

end min_ratio_bd_l353_353310


namespace expected_winnings_correct_l353_353090

theorem expected_winnings_correct :
  let p6 := 1 / 4.
  let p_other := 3 / 4.
  let p_odd := (3 / 5) * p_other.
  let p_even := (2 / 5) * p_other.
  let winnings_6 := -2.0
  let winnings_odd := 2.0
  let winnings_even := 4.0
  let expected_winnings := p_odd * winnings_odd + p_even * winnings_even + p6 * winnings_6
  expected_winnings = 1.6 := 
by
  sorry

end expected_winnings_correct_l353_353090


namespace least_number_of_pairs_l353_353531

theorem least_number_of_pairs :
  let students := 100
  let messages_per_student := 50
  ∃ (pairs_of_students : ℕ), pairs_of_students = 50 := sorry

end least_number_of_pairs_l353_353531


namespace redesigned_survey_respondents_l353_353493

def survey_count1 := 80
def respondents1 := 7
def survey_count2 := 63
def increased_response_rate := 0.05

def r1 := respondents1 / survey_count1
def r2 := r1 + increased_response_rate

def x := Float.round (r2 * survey_count2)

theorem redesigned_survey_respondents : x = 9 := by
  sorry

end redesigned_survey_respondents_l353_353493


namespace sequence_is_arithmetic_l353_353804

theorem sequence_is_arithmetic {a : ℕ → ℝ} (h_pos : ∀ n, a n > 0)
  (h_second_term : a 2 = 3 * a 1)
  (h_sqrt_seq_arith : ∃ d : ℝ, ∀ n, real.sqrt (∑ i in finset.range (n + 1), a i) = d * n + real.sqrt (a 0)): 
  ∃ d : ℝ, ∀ n, a n = a 0 + d * n := 
by
  sorry

end sequence_is_arithmetic_l353_353804


namespace decimal_to_binary_25_l353_353542

theorem decimal_to_binary_25 : (25 : Nat) = 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by
  sorry

end decimal_to_binary_25_l353_353542


namespace sum_of_leading_digits_l353_353800

-- Define M as a 200-digit number all consisting of 8s
def M : ℕ := 8 * 10^199 + 8 * 10^198 + ... + 8 * 10^0

-- Define the leading digit function g(r). 
-- Note: This is abstractly represented and assumes the existence of such a function
def g (r : ℕ) : ℕ := 
  let root := (M : ℝ)^(1/r)
  -- Floor and conversion to ℕ needed for this leading digit calculation
  let leading_digit := (toString root)[0].toNat
  leading_digit

-- The main theorem to prove
theorem sum_of_leading_digits : g 2 + g 3 + g 4 + g 5 + g 6 = 10 := by
  sorry

end sum_of_leading_digits_l353_353800


namespace sum_lent_is_approx_4083_78_l353_353017

-- Conditions
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

def person_A_total (P : ℝ) : ℝ :=
  compound_interest (compound_interest P 0.04 4) 0.05 4

def person_B_total (P : ℝ) : ℝ :=
  P + simple_interest P 0.06 8

-- Given conditions
def condition := person_A_total - P - (simple_interest P 0.06 8) = -238

-- Prove the initial sum lent by each person is approximately Rs. 4083.78
theorem sum_lent_is_approx_4083_78 (P : ℝ) : P ≈ 4083.78 :=
  sorry

end sum_lent_is_approx_4083_78_l353_353017


namespace base_length_of_isosceles_triangle_l353_353402

-- Defining the conditions of the problem
variables {A B C H : Type} [metric_space A] [metric_space B] [metric_space C]
variables (AH BH AB : ℝ) (CH AC BC : ℝ)
variables (isosceles_triangle : AC = BC)
variables (height_dropped_perpendicularly : CH^2 = AC^2 - AH^2) 
variables (segments_AH_BH : AH = 2) (segments_BH : BH = 1)

-- The theorem to prove
theorem base_length_of_isosceles_triangle 
  (isosceles_triangle : AC = BC)
  (height_dropped_perpendicularly : CH^2 = AC^2 - AH^2)
  (height_dropped_perpendicularly_bis : CH^2 = BC^2 - BH^2)
  (segments_AH_BH : AH = 2) 
  (segments_BH : BH = 1) :
  AB = sqrt (5 + 1) :=
by sorry

end base_length_of_isosceles_triangle_l353_353402


namespace triangle_area_correct_l353_353157

def vec3 := ℝ × ℝ × ℝ

def a : vec3 := (4, 1, 2)
def b : vec3 := (2, 5, 9)
def c : vec3 := (0, 1, 3)

def cross_product (u v : vec3) : vec3 :=
  (u.2 * v.3 - u.3 * v.2,
   u.3 * v.1 - u.1 * v.3,
   u.1 * v.2 - u.2 * v.1)

def magnitude (v : vec3) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def triangle_area (a b c : vec3) : ℝ :=
  (1 / 2) * magnitude (cross_product (b.1 - a.1, b.2 - a.2, b.3 - a.3) (c.1 - a.1, c.2 - a.2, c.3 - a.3))

theorem triangle_area_correct :
  triangle_area a b c = Real.sqrt 1172 / 2 :=
by
  sorry

end triangle_area_correct_l353_353157


namespace distinct_lines_in_equilateral_triangle_l353_353257

-- Define an equilateral triangle with side length 10
def is_equilateral_triangle {α : Type} [MetricSpace α] (Δ : Triangle α) : Prop :=
  (Δ.a = Δ.b ∧ Δ.b = Δ.c) ∧ (dist Δ.a Δ.b = 10 ∧ dist Δ.b Δ.c = 10 ∧ dist Δ.c Δ.a = 10)

-- Prove the number of distinct lines (altitudes, medians, angle bisectors) is 3
theorem distinct_lines_in_equilateral_triangle {α : Type} [MetricSpace α] 
  (Δ : Triangle α) (h : is_equilateral_triangle Δ) : 
  (number_of_distinct_lines Δ) = 3 :=
sorry

end distinct_lines_in_equilateral_triangle_l353_353257


namespace product_of_constants_factoring_quadratic_l353_353162

theorem product_of_constants_factoring_quadratic :
  let p := ∏ t in {t | ∃ a b : ℤ, ab = -24 ∧ t = a + b}, t 
  in p = 5290000 := by
sorry

end product_of_constants_factoring_quadratic_l353_353162


namespace abcd_inequality_l353_353172

noncomputable def max_value (a b c d : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) : Prop :=
  abcd (a + b + c + d) / ((a + b)^2 * (c + d)^2) <= 1 / 4

theorem abcd_inequality (a b c d : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) : max_value a b c d h_a h_b h_c h_d :=
sorry

end abcd_inequality_l353_353172


namespace factors_multiple_of_252_l353_353221

def n : ℕ := 2 ^ 12 * 3 ^ 15 * 7 ^ 9

theorem factors_multiple_of_252 : 
  let k := 252 in 
  let num_factors := 11 * 14 * 9 in
  (∃ m : ℕ, m = num_factors ∧ 
    (∀ d : ℕ, d ∣ n → d % k = 0 → d ∣ m) ∧ 
    (∀ d : ℕ, d ∣ n → d % k = 0 → d ∈ (finset.powerset (finset.range.num_factors)))) := 
sorry

end factors_multiple_of_252_l353_353221


namespace num_correct_solutions_l353_353968

/-- Problem: Given the following equations and their correctness evaluations,
    prove the number of correctly solved problems is 2.
  (1) 6 * (a ^ (2 / 3)) * 7 * (a ^ (1 / 2)) = 42 * (a ^ (7 / 6))
  (2) ((-a) * x) ^ 6 / ((-a) * (x ^ 3)) = a ^ 5 * x ^ 3
  (3) (-1989 ^ 0) ^ 1989 = -1
  (4) ((-3) ^ m) ^ 2 = 3 ^ (2 * m)
-/
theorem num_correct_solutions : ∀ (a x m : ℝ),
  ((6 * (a ^ (2 / 3)) * 7 * (a ^ (1 / 2)) = 42 * (a ^ (7 / 6))) ∧
   ¬ ((-a * x) ^ 6 / (-a * (x ^ 3)) = a ^ 5 * x ^ 3) ∧
   ((-1989 ^ 0) ^ 1989 = -1) ∧
   ¬ (((-3) ^ m) ^ 2 = 3 ^ (2 * m)))
  → 2 := 
sorry

end num_correct_solutions_l353_353968


namespace largest_integer_less_than_100_with_remainder_5_l353_353602

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l353_353602


namespace range_of_a_l353_353661

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ax + x + 4 = 0 → x ∈ Icc (-2 : ℝ) 1) → 
  (a < -5 ∨ a > 1) :=
by
  sorry

end range_of_a_l353_353661


namespace rook_moves_not_equal_l353_353089

-- Define conditions and the theorem.

theorem rook_moves_not_equal (vertical_moves horizontal_moves : ℕ) : 
  ( ∃ moves : Fin 64 → (ℕ × ℕ), 
    ∀ i, 
      (moves i).fst ∈ Fin 8 ∧ (moves i).snd ∈ Fin 8 ∧ 
      (i < 63 → (moves (i + 1)).fst = (moves i).fst ∨ (moves (i + 1)).snd = (moves i).snd) ∧ 
      (moves 0 = moves 63) 
  ) → vertical_moves ≠ horizontal_moves := 
  sorry

end rook_moves_not_equal_l353_353089


namespace number_half_reduction_l353_353417

/-- Define the conditions -/
def percentage_more (percent : Float) (amount : Float) : Float := amount + (percent / 100) * amount

theorem number_half_reduction (x : Float) : percentage_more 30 75 = 97.5 → (x / 2) = 97.5 → x = 195 := by
  intros h1 h2
  sorry

end number_half_reduction_l353_353417


namespace find_g_at_2_l353_353399

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation : ∀ (x y : ℝ), x * g(y) = 2 * y * g(x)
axiom g_at_10 : g(10) = 30

theorem find_g_at_2 : g(2) = 12 :=
by
  sorry

end find_g_at_2_l353_353399


namespace ceil_sqrt_200_eq_15_l353_353536

theorem ceil_sqrt_200_eq_15 (h1 : Real.sqrt 196 = 14) (h2 : Real.sqrt 225 = 15) (h3 : 196 < 200 ∧ 200 < 225) : 
  Real.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l353_353536


namespace smallest_m_divisor_condition_l353_353699

def S := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

theorem smallest_m_divisor_condition (m : ℕ) :
  (∀ (A : Finset ℕ), A ⊆ S → A.card = m →
    ∃ x ∈ A, x ∣ (A.erase x).val.prod id) ↔ m = 26 :=
sorry

end smallest_m_divisor_condition_l353_353699


namespace minimum_BD_value_l353_353304

theorem minimum_BD_value {A B C D : Type*} [Real A] [Real B] [Real C] [Real D] (angle_ADB: A) (AD: A) (CD: A) (BD: A):
  (D ∈ line_segment(B, C)) →
  angle_ADB = 120 → 
  AD = 2 → 
  CD = 2 * BD →
  ∃ BD: A, BD = sqrt(3) - 1 :=
by
  sorry

end minimum_BD_value_l353_353304


namespace g_expression_l353_353355

def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 3

def g (a : ℝ) : ℝ :=
  if a ≥ 2 then 8 * a
  else if 0 ≤ a ∧ a < 2 then a^2 + 4 * a + 4
  else if -2 < a ∧ a < 0 then a^2 - 4 * a + 4
  else -8 * a

theorem g_expression (a : ℝ) : 
  g(a) = 
  if a ≥ 2 then 8 * a 
  else if 0 ≤ a ∧ a < 2 then a^2 + 4 * a + 4 
  else if -2 < a ∧ a < 0 then a^2 - 4 * a + 4 
  else -8 * a := 
by
  unfold g
  sorry

end g_expression_l353_353355


namespace arithmetic_seq_of_pos_and_arithmetic_sqrt_S_l353_353814

variable {a : ℕ → ℝ} (h_pos : ∀ n, a n > 0)
variable (h_a2 : a 2 = 3 * a 1)
variable (S : ℕ → ℝ)
variable (h_S : ∀ n, S n = ∑ i in Finset.range(n+1), a i)
variable (h_sqrt_S_arith : ∃ d, ∀ n, (Sqrt.sqrt (S n) - Sqrt.sqrt (S (n - 1))) = d)

theorem arithmetic_seq_of_pos_and_arithmetic_sqrt_S : 
  ∀ n, a (n+1) - a n = a 1 := 
sorry

end arithmetic_seq_of_pos_and_arithmetic_sqrt_S_l353_353814


namespace exists_shorter_segment_l353_353663

variable {Point : Type}
variable [MetricSpace Point]

structure Tetrahedron (Point : Type) :=
(A B C D : Point)

theorem exists_shorter_segment
  (T : Tetrahedron Point)
  (P : Point)
  (h : P ≠ T.D ∧ (is_on_surface_or_inside T P)) :
  ∃ Q ∈ {T.A, T.B, T.C}, ∃ R ∈ ({T.A, T.B, T.C} \ {Q}), dist P Q < dist T.D R :=
by sorry

-- Helper function (assuming it can be defined)
def is_on_surface_or_inside (T : Tetrahedron Point) (P : Point) : Prop := sorry

end exists_shorter_segment_l353_353663


namespace vote_ratio_proof_l353_353765

theorem vote_ratio_proof (V : ℝ) (q : ℝ) 
  (hV : V > 0) : 
  let X := (7/10) * V,
      counted_votes := (3/10) * V,
      votes_in_favor := (2/5) * counted_votes,
      votes_against := (3/5) * counted_votes,
      abstentions := (1/4) * V
  in ((votes_in_favor + q * X) / (votes_against + (1 - q) * X) = (3/2)) → 
     q = (24/35) :=
by
  sorry

end vote_ratio_proof_l353_353765


namespace tank_filled_fraction_l353_353743

noncomputable def initial_quantity (total_capacity : ℕ) := (3 / 4 : ℚ) * total_capacity

noncomputable def final_quantity (initial : ℚ) (additional : ℚ) := initial + additional

noncomputable def fraction_of_capacity (quantity : ℚ) (total_capacity : ℕ) := quantity / total_capacity

theorem tank_filled_fraction (total_capacity : ℕ) (additional_gas : ℚ)
  (initial_fraction : ℚ) (final_fraction : ℚ) :
  initial_fraction = initial_quantity total_capacity →
  final_fraction = fraction_of_capacity (final_quantity initial_fraction additional_gas) total_capacity →
  total_capacity = 42 →
  additional_gas = 7 →
  initial_fraction = 31.5 →
  final_fraction = (833 / 909 : ℚ) :=
by
  sorry

end tank_filled_fraction_l353_353743


namespace compute_f_e_l353_353683

noncomputable def f (x : ℝ) : ℝ := 4 * x * (f' e) + Real.log x 

theorem compute_f_e : f e = -1 / 3 := 
by 
  sorry

end compute_f_e_l353_353683


namespace find_b_value_l353_353750

noncomputable def h (x : ℝ) (b : ℝ) : ℝ := x^2 - (b-1)*x + b

def weakly_increasing_in (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  (∀ x y ∈ I, x < y → f x ≤ f y) ∧ (∀ x y ∈ I, x < y → f x / x ≥ f y / y)

theorem find_b_value (b : ℝ) :
  weakly_increasing_in (λ x, h x b) (Set.Icc 0 1) → b = 1 :=
by
  sorry

end find_b_value_l353_353750


namespace min_distance_l353_353287

def triangle_ABC (A B C D : Type) (BD : ℝ) :=
∃ (CD : ℝ), CD = 2 * BD ∧ 
  ∃ (AD AC AB : ℝ), AD = 2 ∧ 
    ∃ (α β γ : ℝ), α = 120 ∧ 
      angle_ADB = α ∧ 
        (AC / AB = (sqrt (3) - 1))

theorem min_distance (A B C D : Type) (BD : ℝ) :
  ∃ D ∈ BC, ∃ α β γ AD AC AB, 
    α = 120 ∧ AD = 2 ∧ D ∈ BC ∧ 
      γ = ∠ADB = α ∧ 
        CD = 2 * BD →
          (AC / AB) = (sqrt (3) - 1) := 
sorry

end min_distance_l353_353287


namespace train_passing_time_l353_353047

def train_length : ℝ := 110 -- train length in meters
def train_speed : ℝ := 40 -- train speed in km/h
def man_speed : ℝ := 4 -- man speed in km/h
def distance (t : ℝ) : ℝ := t * (train_speed + man_speed) / 3.6 -- converting speed to m/s and calculating the distance

theorem train_passing_time : ∃ t : ℝ, distance t = train_length ∧ t ≈ 8.99 :=
by
  sorry

end train_passing_time_l353_353047


namespace arithmetic_seq_of_pos_and_arithmetic_sqrt_S_l353_353813

variable {a : ℕ → ℝ} (h_pos : ∀ n, a n > 0)
variable (h_a2 : a 2 = 3 * a 1)
variable (S : ℕ → ℝ)
variable (h_S : ∀ n, S n = ∑ i in Finset.range(n+1), a i)
variable (h_sqrt_S_arith : ∃ d, ∀ n, (Sqrt.sqrt (S n) - Sqrt.sqrt (S (n - 1))) = d)

theorem arithmetic_seq_of_pos_and_arithmetic_sqrt_S : 
  ∀ n, a (n+1) - a n = a 1 := 
sorry

end arithmetic_seq_of_pos_and_arithmetic_sqrt_S_l353_353813


namespace matrix_det_is_neg16_l353_353120

def matrix := Matrix (Fin 2) (Fin 2) ℤ
def given_matrix : matrix := ![![ -7, 5], ![6, -2]]

theorem matrix_det_is_neg16 : Matrix.det given_matrix = -16 := 
by
  sorry

end matrix_det_is_neg16_l353_353120


namespace values_of_a_for_4_zeros_l353_353424

-- Given the transformation of the sine function
def f (x : ℝ) : ℝ := Math.sin (2 * x - Real.pi / 6)

-- Given the piecewise function g(x)
def g (x : ℝ) (a : ℝ) : ℝ :=
  if -11 * Real.pi / 12 ≤ x ∧ x ≤ a then f x
  else 3 * x^2 - 2 * x - 1

-- Prove that the set of values for 'a' when g(x) has 4 zeros is the desired set
theorem values_of_a_for_4_zeros (a : ℝ) :
  (∃ (x : ℝ), -11 * Real.pi / 12 ≤ x ∧ x ≤ a ∧ f x = 0) ∧
  (∃ (x : ℝ), a < x ∧ x ≤ 13 * Real.pi / 12 ∧ 3*x^2 - 2*x - 1 = 0) ∧
  g(x) has 4 zeros → 
  a ∈ Set.Icc (-5 * Real.pi / 12) (-1 / 3) ∪ Set.Ico (Real.pi / 12) 1 ∪ Set.Ico (7 * Real.pi / 12) (13 * Real.pi / 12) :=
sorry

end values_of_a_for_4_zeros_l353_353424


namespace flower_bed_lilies_l353_353004

theorem flower_bed_lilies :
  ∃ (L : ℕ), L = 12 ∧ (82 = 57 + L + 13) :=
by
  use 12
  sorry

end flower_bed_lilies_l353_353004


namespace largest_e_possible_statement_l353_353818

noncomputable def largest_e_possible 
  (P Q : ℝ) (circ : ℝ) (X Y Z : ℝ → ℝ) 
  (diam : ℝ) 
  (midpoint : ℝ) 
  (py_eq : ℝ) 
  (intersects_S : ℝ) 
  (intersects_T : ℝ) 
  : ℝ :=
  sorry

-- Statement of the problem in Lean
theorem largest_e_possible_statement 
  (P Q : ℝ) (circ : ℝ) (X Y Z : ℝ → ℝ) 
  (diam : ℝ) 
  (midpoint : ℝ) 
  (py_eq : ℝ) 
  (intersects_S : ℝ) 
  (intersects_T : ℝ) 
  (circ_diam : circ = 2)
  (X_mid : X midpoint = 1)
  (PY_val : PY_eq = 4/5)
  : largest_e_possible P Q circ X Y Z diam midpoint py_eq intersects_S intersects_T = 25 :=
  sorry

end largest_e_possible_statement_l353_353818


namespace incorrect_table_value_l353_353013

theorem incorrect_table_value (a b c : ℕ) (values : List ℕ) (correct : values = [2051, 2197, 2401, 2601, 2809, 3025, 3249, 3481]) : 
  (2401 ∉ [2051, 2197, 2399, 2601, 2809, 3025, 3249, 3481]) :=
sorry

end incorrect_table_value_l353_353013


namespace edge_length_of_cube_l353_353214

theorem edge_length_of_cube (V_s : ℝ) (a : ℝ) (hV_s : V_s = (32 / 3) * real.pi) :
    ((4 / real.sqrt 3) : ℝ) = (4 * real.sqrt 3) / 3 :=
by
    sorry

end edge_length_of_cube_l353_353214


namespace expand_latin_square_l353_353459

theorem expand_latin_square 
  (n : ℕ) (h : 2 < n) 
  (T : Fin (n-2) → Fin n → Fin n) 
  (uniq_rows : ∀ i : Fin (n-2), Function.Injective (T i))
  (uniq_cols : ∀ j : Fin n, Function.Injective (λ i : Fin (n-2), T i j)) :
  ∃ (S : Fin n → Fin n → Fin n), 
    (∀ i, Function.Injective (S i)) ∧
    (∀ j, Function.Injective (λ i, S i j)) ∧
    (∀ i : Fin (n-2), ∀ j : Fin n, S ⟨i.val, lt_of_lt_of_le i.is_lt (nat.sub_le n 2)⟩ j = T i j) := 
sorry

end expand_latin_square_l353_353459


namespace total_days_equivalent_l353_353065

-- Define constants for the number of days A and B take individually to complete the work
def days_A := 15
def days_B := 10

-- Define the work done per day by A and B
def work_per_day_A := 1 / (days_A : ℝ)
def work_per_day_B := 1 / (days_B : ℝ)
def work_per_day_together := work_per_day_A + work_per_day_B

-- Define the number of days A and B work together
def days_together := 2

-- Calculate the work done in the initial days together
def work_done_together := days_together * work_per_day_together

-- Calculate the remaining work to be done by A alone
def remaining_work := 1 - work_done_together
def days_A_alone := remaining_work / work_per_day_A

-- The total number of days taken to complete the work
def total_days := days_together + days_A_alone

-- The theorem to prove
theorem total_days_equivalent:
  total_days = 12 := by
  sorry

end total_days_equivalent_l353_353065


namespace problem_statement_l353_353824

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition1 : g 2 = 4
axiom g_condition2 : ∀ (x y : ℝ), g (xy + g x) = 2 * x * g y + g x

def n := sorry -- Number of possible values of g(1/2)
def s := sorry -- Sum of all possible values of g(1/2)

theorem problem_statement : n * s = 1 :=
by {
  sorry
}

end problem_statement_l353_353824


namespace projection_of_triangle_l353_353895

-- Definitions of the geometric entities and their properties
inductive ProjectionType where
  | angle
  | strip
  | two_angles_joined_with_continuation
  | triangle
  | angle_with_infinite_figure

open ProjectionType

-- Lean statement of the proof problem
theorem projection_of_triangle (O A B C : Point) (P : Plane) (hO_not_in_plane : ¬ lies_in_plane O ABC) :
  ∃ proj : ProjectionType, proj = angle ∨ proj = strip ∨ proj = two_angles_joined_with_continuation ∨ proj = triangle ∨ proj = angle_with_infinite_figure :=
sorry

end projection_of_triangle_l353_353895


namespace marble_problem_l353_353066

-- Definitions for the problem
def total_marbles := 30
def blue_marbles := 5
def red_white_marbles := total_marbles - blue_marbles
def probability_red_or_white := 5 / 6

-- Define R and W as the numbers of red and white marbles, respectively
variables (R W : ℕ)

-- Conditions stated in the problem
def condition1 := red_white_marbles = R + W
def condition2 := (R + W : ℚ) / total_marbles = probability_red_or_white

-- The proof problem statement
theorem marble_problem (h1 : condition1) (h2 : condition2) : (R + W : ℚ) / total_marbles = probability_red_or_white :=
by sorry

end marble_problem_l353_353066


namespace sin_half_inequality_l353_353854

theorem sin_half_inequality (α β γ : ℝ) : 
  1 - Real.sin (α / 2) ≥ 2 * Real.sin (β / 2) * Real.sin (γ / 2) :=
sorry

end sin_half_inequality_l353_353854


namespace largest_integer_less_than_100_div_8_rem_5_l353_353615

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l353_353615


namespace miniou_circuit_nodes_l353_353064

-- Definition of a Miniou circuit according to the problem's conditions.
structure MiniouCircuit (V E : Type) :=
  (nodes : V)
  (wires : E)
  (incidence : E → V × V)
  (degree : V → ℕ)
  (node_degree : ∀ v, degree v = 3) -- Each node is connected to exactly 3 wires
  (edge_count : ∀ u v, u ≠ v → ((∃! e, incidence e = (u, v) ∨ incidence e = (v, u)) ∨ (¬∃ e, incidence e = (u, v) ∨ incidence e = (v, u))))

-- A proof for calculating the number of nodes given 13788 wires.
theorem miniou_circuit_nodes :
  ∃ n : ℕ, ∀ c : MiniouCircuit, c.wires.size = 13788 → n = 9192 :=
begin
  sorry
end

end miniou_circuit_nodes_l353_353064


namespace parallel_lines_have_equal_slopes_l353_353222

-- Define the two lines l1 and l2 based on the conditions
def l1 (m : ℝ) : ℝ × ℝ → ℝ := λ (p : ℝ × ℝ), (m+1) * p.1 + 2 * p.2 + (2*m) - 2 
def l2 (m : ℝ) : ℝ × ℝ → ℝ := λ (p : ℝ × ℝ), 2 * p.1 + (m-2) * p.2 + 2

-- Define the function to extract the slope of l1 and l2
def slope_l1 (m : ℝ) : ℝ := - (m + 1) / 2
def slope_l2 (m : ℝ) : ℝ := - 2 / (m - 2)

-- Define the theorem statement
theorem parallel_lines_have_equal_slopes {m : ℝ} : 
  slope_l1 m = slope_l2 m → m = -2 :=
sorry

end parallel_lines_have_equal_slopes_l353_353222


namespace seating_arrangement_count_l353_353416

theorem seating_arrangement_count :
  let front_row := 11
  let back_row := 12
  let blocked_seats_front := 3
  let people := 2
  let valid_arrangements := 346
  (∀ positions : list ℕ,
    (length positions = front_row + back_row - blocked_seats_front) → 
    (∀ two_people : list (ℕ × ℕ), 
      (length two_people = people) ∧ 
      (∀ p1 p2, p1 ≠ p2 → abs (p1.1 - p2.1) > 1 ∧ abs (p1.2 - p2.2) > 1)) →
      count_valid_arrangements positions two_people = valid_arrangements) := 
sorry

end seating_arrangement_count_l353_353416


namespace conjugate_of_z_l353_353835

-- Define the imaginary unit.
noncomputable def i := Complex.I

-- Define the complex number z.
def z : ℂ := (5 * i) / (1 - 2 * i)

-- State the theorem that the conjugate of z is -2 - i.
theorem conjugate_of_z :
  conj z = -2 - i :=
sorry

end conjugate_of_z_l353_353835


namespace trillion_value_l353_353776

def ten_thousand : ℕ := 10^4
def million : ℕ := 10^6
def billion : ℕ := ten_thousand * million

theorem trillion_value : (ten_thousand * ten_thousand * billion) = 10^16 :=
by
  sorry

end trillion_value_l353_353776


namespace functions_equivalence_l353_353040

-- Definitions of the given pairs of functions
def f1 (x : ℝ) := |x|
def g1 (t : ℝ) := Real.sqrt (t^2)

def f2 (x : ℝ) := Real.sqrt (x^2)
def g2 (x : ℝ) := (Real.cbrt x)^3

def f3 (x : ℝ) := (x^2 - 1) / (x - 1)
def g3 (x : ℝ) := x + 1

def f4 (x : ℝ) := Real.sqrt (x + 1) * Real.sqrt (x - 1)
def g4 (x : ℝ) := Real.sqrt (x^2 - 1)

-- Theorem to prove the equivalence
theorem functions_equivalence :
  (∀ x : ℝ, f1 x = g1 x) ∧
  ¬ (∀ x : ℝ, f2 x = g2 x) ∧
  ¬ (∀ x : ℝ, f3 x = g3 x) ∧
  ¬ (∀ x : ℝ, f4 x = g4 x) := by
  sorry

end functions_equivalence_l353_353040


namespace product_of_constants_l353_353164

theorem product_of_constants (h_factorized : ∀ t : ℤ, 
    ∃ a b : ℤ, x^2 + t * x - 24 = (x + a) * (x + b) ∧ a * b = -24) : 
    ∏ t in {t : ℤ | ∃ a b : ℤ, a * b = -24 ∧ t = a + b}, t = -10580000 :=
sorry

end product_of_constants_l353_353164


namespace length_of_AB_l353_353057

theorem length_of_AB 
  (O : circle) (A B C D : point) 
  (semicircle_prop : diameter_on_base O A B)
  (tangent_prop : tangent_to_sides O B C D A)
  (BC_eq : length BC = 2)
  (DA_eq : length DA = 3) : 
  length AB = 5 := 
sorry

end length_of_AB_l353_353057


namespace largest_integer_less_than_100_div_8_rem_5_l353_353616

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l353_353616


namespace find_angle_B_find_sin_C_l353_353756

noncomputable section

-- First part, proving B == π/3 given the conditions
theorem find_angle_B (a b c A B C : ℝ)
  (h1 : B ∈ Ioo 0 π)
  (h2 : sin C / (sin A * cos B) = 2 * c / a)
  (h3 : a = (c * sin A) / sin C)
  (h4 : A + B + C = π) :
  B = π / 3 :=
by sorry

-- Second part, proving sin C given that cos A = 1/4
theorem find_sin_C (a b c A B C : ℝ)
  (h1 : B = π / 3)
  (h2 : cos A = 1 / 4)
  (h3 : sin A = sqrt (1 - (cos A) ^ 2))
  (h4 : A + B + C = π) :
  sin C = (sqrt 15 + sqrt 3) / 8 :=
by sorry

end find_angle_B_find_sin_C_l353_353756


namespace find_num_large_envelopes_l353_353105

def numLettersInSmallEnvelopes : Nat := 20
def totalLetters : Nat := 150
def totalLettersInMediumLargeEnvelopes := totalLetters - numLettersInSmallEnvelopes -- 130
def lettersPerLargeEnvelope : Nat := 5
def lettersPerMediumEnvelope : Nat := 3
def numLargeEnvelopes (L : Nat) : Prop := 5 * L + 6 * L = totalLettersInMediumLargeEnvelopes

theorem find_num_large_envelopes : ∃ L : Nat, numLargeEnvelopes L ∧ L = 11 := by
  sorry

end find_num_large_envelopes_l353_353105


namespace even_consecutive_sums_less_1000_l353_353736

theorem even_consecutive_sums_less_1000 :
  {N : ℕ | N % 6 = 0 ∧ N < 1000 ∧ (∃ S: finset ℕ, S.card = 4 ∧ ∀ j ∈ S, ∃ n: ℕ, N = 2 * j * (n + j - 1) ∧ j ≥ 2)}.to_finset.card = 2 := by
    sorry

end even_consecutive_sums_less_1000_l353_353736


namespace probability_divide_event_l353_353915

-- Define the set of integers from 1 to 6
def my_set : finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the total number of ways to choose 2 distinct numbers from the set
def total_pairs : ℕ := nat.choose 6 2

-- Define the set of pairs where (a, b) such that a < b and a divides b
def successful_pairs : finset (ℕ × ℕ) :=
  (my_set.product my_set).filter (λ (ab : ℕ × ℕ), 
    ab.1 < ab.2 ∧ ab.1 ∣ ab.2)

-- Calculate the probability of successful outcome
def probability_success : ℚ :=
  (successful_pairs.card:ℚ) / (total_pairs:ℚ)

theorem probability_divide_event : 
  probability_success = 8 / 15 := 
by sorry

end probability_divide_event_l353_353915


namespace quadratic_sum_roots_l353_353751

-- We define the quadratic function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- The function f passes through points (r, k) and (s, k)
variables (a b c r s k : ℝ)
variable (ha : a ≠ 0)
variable (hr : f a b c r = k)
variable (hs : f a b c s = k)

-- What we want to prove
theorem quadratic_sum_roots :
  f a b c (r + s) = c :=
sorry

end quadratic_sum_roots_l353_353751


namespace largest_integer_less_than_100_div_8_rem_5_l353_353612

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l353_353612


namespace sum_of_solutions_eq_504_pi_l353_353524

noncomputable def sum_of_real_solutions : ℝ :=
  504 * Real.pi

theorem sum_of_solutions_eq_504_pi (x : ℝ) (hx : 0 < x) :
  2 * (Real.cos (2 * x)) * ((Real.cos (2 * x)) - (Real.cos (1006 * Real.pi^2 / x))) = Real.cos (6 * x) - 1 →
  ∃ (xs : list ℝ), (∀ x ∈ xs, 0 < x ∧
    2 * (Real.cos (2 * x)) * ((Real.cos (2 * x)) - (Real.cos (1006 * Real.pi^2 / x))) = Real.cos (6 * x) - 1) ∧
    xs.sum = sum_of_real_solutions := sorry

end sum_of_solutions_eq_504_pi_l353_353524


namespace negation_of_p_l353_353853

theorem negation_of_p : 
  (¬(∀ x : ℝ, |x| < 0)) ↔ (∃ x : ℝ, |x| ≥ 0) :=
by {
  sorry
}

end negation_of_p_l353_353853


namespace largest_integer_lt_100_with_remainder_5_div_8_l353_353596

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l353_353596


namespace range_of_t_l353_353827

noncomputable theory

def has_deriv_on_R (f : ℝ → ℝ) : Prop := ∀ x : ℝ, ∃ f' : ℝ, has_deriv_at f f' x

def even_function (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g x = g (-x)

def monotonic_on_pos (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, x > 0 → ∀ ε > 0, g (x + ε) > g x

variables (f : ℝ → ℝ)
variables [has_deriv_on_R f]
variables (even_f : ∀ x : ℝ, f x - f (-x) = 2 * real.sin x)
variables (monotonic_f : ∀ x : ℝ, x > 0 → f' x > real.cos x)
variables (ineq_f : ∀ t : ℝ, f (real.pi / 2 - t) - f t > real.cos t - real.sin t)

theorem range_of_t :
  ∀ t : ℝ, (f (real.pi / 2 - t) - f t > real.cos t - real.sin t) → t < real.pi / 4 :=
sorry

end range_of_t_l353_353827


namespace round_table_seating_l353_353771

-- Define the number of people
def num_people : Nat := 7

-- Define the factorial function
def factorial : ℕ → ℕ
| 0        := 1
| (n + 1)  := (n + 1) * factorial n

-- Theorem statement
theorem round_table_seating : (factorial num_people / num_people) = factorial (num_people - 1) := by
  sorry

end round_table_seating_l353_353771


namespace function_is_periodic_l353_353059

variable {R : Type*} [LinearOrderedField R]

def periodic_function (f : R → R) (p : R) : Prop := ∀ x, f (x + p) = f x

theorem function_is_periodic 
  (f : R → R) (a1 b1 a2 b2 : R)
  (h1 : a1 + b1 ≠ a2 + b2)
  (h2 : ∀ x, (f (a1 + x) = f (b1 - x) ∧ f (a2 + x) = f (b2 - x)) ∨ 
             (f (a1 + x) = -f (b1 - x) ∧ f (a2 + x) = -f (b2 - x))) :
  periodic_function f (| (a2 + b2) - (a1 + b1) |) :=
by
  sorry

end function_is_periodic_l353_353059


namespace mean_score_of_seniors_l353_353364

theorem mean_score_of_seniors :
  ∀ (n s : ℕ) (m : ℝ), 
  n + s = 120 ∧ (120 * 120) = 14400 ∧ n = 1.8 * s ∧ m = 1.8 * (14400 / (n + s)) → 
  m = 1.8 * (14400 / (154.4)) := 
by
  intros n s m h
  sorry

end mean_score_of_seniors_l353_353364


namespace pairs_a_eq_b_l353_353975

theorem pairs_a_eq_b 
  (n : ℕ) (h_n : ¬ ∃ k : ℕ, k^2 = n) (a b : ℕ) 
  (r : ℝ) (h_r_pos : 0 < r) (h_ra_rational : ∃ q₁ : ℚ, r^a + (n:ℝ)^(1/2) = q₁) 
  (h_rb_rational : ∃ q₂ : ℚ, r^b + (n:ℝ)^(1/2) = q₂) : 
  a = b :=
sorry

end pairs_a_eq_b_l353_353975


namespace prob_both_defective_prob_exactly_one_defective_l353_353758

open_locale classical

-- Definitions based on conditions
def total_bulbs : ℕ := 6
def defective_bulbs : ℕ := 2
def good_bulbs : ℕ := 4
def total_selections := (total_bulbs.choose 2 : ℚ)

-- Question (I): Both selected bulbs are defective
def two_defective_selections := (defective_bulbs.choose 2 : ℚ)
def prob_two_defective := two_defective_selections / total_selections

-- Question (II): Exactly one selected bulb is defective
def one_defective_selections := (defective_bulbs * good_bulbs : ℚ)
def prob_one_defective := one_defective_selections / total_selections

-- Mathematical proof problems
theorem prob_both_defective : prob_two_defective = 1 / 15 :=
by
  sorry

theorem prob_exactly_one_defective : prob_one_defective = 8 / 15 :=
by
  sorry

end prob_both_defective_prob_exactly_one_defective_l353_353758


namespace range_of_g_l353_353131

-- Define the function g(t)
def g (t : ℝ) (k : ℝ) : ℝ := (t^2 + k * t) / (t^2 + 1)

-- Given conditions
def k : ℝ := 1 / 2
noncomputable def t : ℝ := sorry  -- t is any real number

-- The proof problem to show the range of g
theorem range_of_g : range (fun t => g t k) = {1 / 4} :=
by sorry

end range_of_g_l353_353131


namespace arithmetic_mean_pq_l353_353881

variable (p q r : ℝ)

-- Definitions from conditions
def condition1 := (p + q) / 2 = 10
def condition2 := (q + r) / 2 = 26
def condition3 := r - p = 32

-- Theorem statement
theorem arithmetic_mean_pq : condition1 p q → condition2 q r → condition3 p r → (p + q) / 2 = 10 :=
by
  intros h1 h2 h3
  exact h1

end arithmetic_mean_pq_l353_353881


namespace expression_value_l353_353175

theorem expression_value (a b c : ℝ) (h : a * b + b * c + c * a = 3) : 
  (a * (b^2 + 3)) / (a + b) + (b * (c^2 + 3)) / (b + c) + (c * (a^2 + 3)) / (c + a) = 6 := 
by
  sorry

end expression_value_l353_353175


namespace inscribed_triangle_chord_square_l353_353970

theorem inscribed_triangle_chord_square (r : ℝ) (A B C O : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space O] 
  (h_circle : metric_space.sphere O r = {A, B, C}) 
  (h_AB : metric_space.dist A B = r * real.sqrt 2)
  (h_AO : metric_space.dist A O = r)
  (h_BO : metric_space.dist B O = r)
  (h_CO : C ≠ O) :
  (metric_space.dist A C + metric_space.dist B C) ^ 2 = 32 * r ^ 2 :=
sorry

end inscribed_triangle_chord_square_l353_353970


namespace int_solution_l353_353548

theorem int_solution (n : ℕ) (h1 : n ≥ 1) (h2 : n^2 ∣ 2^n + 1) : n = 1 ∨ n = 3 :=
by
  sorry

end int_solution_l353_353548


namespace number_of_subsets_two_elements_set_l353_353890

theorem number_of_subsets_two_elements_set : 
  ∀ (a b : Type), (∃ (S : set Type), S = {a, b}) → set.finite S ∧ set.card S = 4 := sorry

end number_of_subsets_two_elements_set_l353_353890


namespace find_cos_2beta_l353_353674

noncomputable def cos_2beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
  (htan : Real.tan α = 1 / 7) (hcos : Real.cos (α + β) = 2 * Real.sqrt 5 / 5) : Real :=
  2 * (Real.cos β)^2 - 1

theorem find_cos_2beta (α β : ℝ) (h1: 0 < α ∧ α < π / 2) (h2: 0 < β ∧ β < π / 2)
  (htan: Real.tan α = 1 / 7) (hcos: Real.cos (α + β) = 2 * Real.sqrt 5 / 5) :
  cos_2beta α β h1 h2 htan hcos = 4 / 5 := 
sorry

end find_cos_2beta_l353_353674


namespace vector_dot_product_l353_353703

variables (a b : ℝ^3)
variables (norm_a : ∥a∥ = 5) (norm_b : ∥b∥ = 8)
variables (angle_ab : real.angle a b = real.pi / 4 )

theorem vector_dot_product :
  (a + b) • (a - b) = -39 :=
sorry

end vector_dot_product_l353_353703


namespace ratio_of_almonds_to_walnuts_l353_353980

theorem ratio_of_almonds_to_walnuts
  (A W : ℝ)
  (weight_almonds : ℝ)
  (total_weight : ℝ)
  (weight_walnuts : ℝ)
  (ratio : 2 * W = total_weight - weight_almonds)
  (given_almonds : weight_almonds = 107.14285714285714)
  (given_total_weight : total_weight = 150)
  (computed_weight_walnuts : weight_walnuts = 42.85714285714286)
  (proportion : A / (2 * W) = weight_almonds / weight_walnuts) :
  A / W = 5 :=
by
  sorry

end ratio_of_almonds_to_walnuts_l353_353980


namespace xyz_solution_l353_353834

theorem xyz_solution (x y z : ℂ) (h1 : x * y + 5 * y = -20) 
                                 (h2 : y * z + 5 * z = -20) 
                                 (h3 : z * x + 5 * x = -20) :
  x * y * z = 200 / 3 := 
sorry

end xyz_solution_l353_353834


namespace sin_cos_identity_l353_353454

theorem sin_cos_identity (x : ℝ) :
  (sin (2 * x) - sin (3 * x) + sin (8 * x) = cos (7 * x + 3 * Real.pi / 2)) ->
  ∃ (k : ℤ), x = k * Real.pi / 5 :=
by
  sorry

end sin_cos_identity_l353_353454


namespace decimal_to_binary_correct_l353_353545

-- Define the decimal number
def decimal_number : ℕ := 25

-- Define the binary equivalent of 25
def binary_representation : ℕ := 0b11001

-- The condition indicating how the conversion is done
def is_binary_representation (decimal : ℕ) (binary : ℕ) : Prop :=
  -- Check if the binary representation matches the manual decomposition
  decimal = (binary / 2^4) * 2^4 + 
            ((binary % 2^4) / 2^3) * 2^3 + 
            (((binary % 2^4) % 2^3) / 2^2) * 2^2 + 
            ((((binary % 2^4) % 2^3) % 2^2) / 2^1) * 2^1 + 
            (((((binary % 2^4) % 2^3) % 2^2) % 2^1) / 2^0) * 2^0

-- Proof statement
theorem decimal_to_binary_correct : is_binary_representation decimal_number binary_representation :=
  by sorry

end decimal_to_binary_correct_l353_353545


namespace largest_cookies_without_ingredients_l353_353839

noncomputable def largest_number_of_cookies_without_chips_marshmallows_cayenne_coconut_caramel
  (total_cookies : ℕ)
  (contains_chocolate_chips : ℕ)
  (contains_marshmallows : ℕ)
  (contains_cayenne : ℕ)
  (contains_coconut_flakes : ℕ)
  (contains_salted_caramel : ℕ) : ℕ :=
  total_cookies - contains_cayenne

theorem largest_cookies_without_ingredients
  (total_cookies : ℕ := 48)
  (contains_chocolate_chips: ℕ := 24)
  (contains_marshmallows : ℕ := 28)
  (contains_cayenne : ℕ := 32)
  (contains_coconut_flakes : ℕ := 16)
  (contains_salted_caramel : ℕ := 6) :
  largest_number_of_cookies_without_chips_marshmallows_cayenne_coconut_caramel
    total_cookies
    contains_chocolate_chips
    contains_marshmallows
    contains_cayenne
    contains_coconut_flakes
    contains_salted_caramel = 16 :=
begin
  sorry
end

end largest_cookies_without_ingredients_l353_353839


namespace minimize_expression_l353_353450

theorem minimize_expression : ∃ c : ℝ, c = 6 ∧ ∀ x : ℝ, (3 / 4) * (x ^ 2) - 9 * x + 7 ≥ (3 / 4) * (6 ^ 2) - 9 * 6 + 7 :=
by
  sorry

end minimize_expression_l353_353450


namespace inequality_proof_l353_353822

noncomputable def a := 2^(1/2 : ℝ)
noncomputable def b := Real.exp(1/Real.exp 1)
noncomputable def c := 3^(1/3 : ℝ)

theorem inequality_proof : a < c ∧ c < b :=
by
  -- Sorry, placeholder for proof
  sorry

end inequality_proof_l353_353822


namespace sequence_is_arithmetic_l353_353805

theorem sequence_is_arithmetic {a : ℕ → ℝ} (h_pos : ∀ n, a n > 0)
  (h_second_term : a 2 = 3 * a 1)
  (h_sqrt_seq_arith : ∃ d : ℝ, ∀ n, real.sqrt (∑ i in finset.range (n + 1), a i) = d * n + real.sqrt (a 0)): 
  ∃ d : ℝ, ∀ n, a n = a 0 + d * n := 
by
  sorry

end sequence_is_arithmetic_l353_353805


namespace points_per_game_l353_353843

theorem points_per_game (total_points : ℝ) (num_games : ℝ) (h1 : total_points = 120.0) (h2 : num_games = 10.0) : (total_points / num_games) = 12.0 :=
by 
  rw [h1, h2]
  norm_num
  -- sorry


end points_per_game_l353_353843


namespace problem_l353_353701

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def line_equation (P Q : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let a := Q.2 - P.2
  let b := P.1 - Q.1
  let c := a * P.1 + b * P.2
  (a, b, -c)

noncomputable def distance_from_point_to_line 
  (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  (abs (a * P.1 + b * P.2 + c)) / (sqrt (a^2 + b^2))

theorem problem (A B C : ℝ × ℝ)
  (hA : A = (2, 3)) (hB : B = (-1, -2)) (hC : C = (-3, 4)) :
  (∃ a b c, line_equation A (midpoint B C) = (a, b, c) ∧ a = 1 ∧ b = -2 ∧ c = 4) ∧
  (∃ S, S = 14) :=
by
  sorry

end problem_l353_353701


namespace largest_integer_less_than_hundred_with_remainder_five_l353_353587

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l353_353587


namespace doubled_container_volume_l353_353481

theorem doubled_container_volume (v : ℝ) (h₁ : v = 4) (h₂ : ∀ l w h : ℝ, v = l * w * h) : 8 * v = 32 := 
by
  -- The proof will go here, this is just the statement
  sorry

end doubled_container_volume_l353_353481


namespace modulus_calculation_l353_353238

noncomputable theory
open Complex

theorem modulus_calculation (a b : ℝ) (z : ℂ) (h₁ : z = a + b * I) (h₂ : (1 - I) * z = 1) :
  |2 * z - 3| = Real.sqrt 5 :=
by sorry

end modulus_calculation_l353_353238


namespace arithmetic_seq_of_pos_and_arithmetic_sqrt_S_l353_353812

variable {a : ℕ → ℝ} (h_pos : ∀ n, a n > 0)
variable (h_a2 : a 2 = 3 * a 1)
variable (S : ℕ → ℝ)
variable (h_S : ∀ n, S n = ∑ i in Finset.range(n+1), a i)
variable (h_sqrt_S_arith : ∃ d, ∀ n, (Sqrt.sqrt (S n) - Sqrt.sqrt (S (n - 1))) = d)

theorem arithmetic_seq_of_pos_and_arithmetic_sqrt_S : 
  ∀ n, a (n+1) - a n = a 1 := 
sorry

end arithmetic_seq_of_pos_and_arithmetic_sqrt_S_l353_353812


namespace problem_l353_353346

noncomputable def S (k : ℕ → ℕ) (n : ℕ) : ℕ := (List.range n).sum (λ i => 2 ^ (k i))

theorem problem (m n : ℕ) (k : ℕ → ℕ) :
  (∀ i j, i < n → j < n → i ≠ j → k i ≠ k j) →
  (S k n) % (2 ^ m - 1) = 0 →
  n ≥ m :=
by
  sorry

end problem_l353_353346


namespace element_not_in_set_l353_353225

def A := {x : ℝ | x ≤ 4}
def a : ℝ := 3 * Real.sqrt 3

theorem element_not_in_set : a ∉ A :=
by
  sorry

end element_not_in_set_l353_353225


namespace taxi_fare_range_l353_353772

theorem taxi_fare_range (x : ℝ) (h : 12.5 + 2.4 * (x - 3) = 19.7) : 5 < x ∧ x ≤ 6 :=
by
  -- Given conditions and the equation, we need to prove the inequalities.
  have fare_eq : 12.5 + 2.4 * (x - 3) = 19.7 := h
  sorry

end taxi_fare_range_l353_353772


namespace proof_problem_l353_353496

variables {A B C D E I M O : Point} -- Define points as variables
variable {x : ℝ} -- Define variable x as real

-- Given Conditions
def is_rectangle (ABCD: Rectangle) : Prop := true -- stating ABCD is a rectangle
def midpoint_I (I CD: Point) : Prop := true -- I is midpoint of CD
def intersection_M (BI AC M: Point) : Prop := true -- BI meets AC at M
def point_E (AE BE : ℝ) : Prop := AE = BE -- E outside such that AE = BE
def angle_AEB (angle : ℝ) : Prop := angle = 90 -- ∠AEB = 90°
def BE_BC_x (BE BC : ℝ) : Prop := BE = BC = x -- BE = BC = x

-- Lean Statement
theorem proof_problem (rect : is_rectangle ABCD) (mid : midpoint_I I CD) 
    (inter : intersection_M BI AC M) (outside : point_E AE BE) (angle : angle_AEB 90) (length : BE_BC_x BE BC):
    -- Show 1: Line DM passes through the midpoint of BC.
    passes_through_midpoint_of_BC DM ∧
    -- Show 2: EM bisects ∠AMB.
    bisects_angle_AMB EM ∧
    -- Show 3: The area of AEBM in terms of x is x² *(1 / 2 + √2 / 3).
    area_AEBM = x^2 * (1 / 2 + real.sqrt 2 / 3) :=
sorry -- Proof not required

end proof_problem_l353_353496


namespace quadrilateral_diagonal_length_l353_353855

theorem quadrilateral_diagonal_length {A B C D : Type} [convex_quadrilateral A B C D] 
  (obtuse_A : obtuse_angle A) (obtuse_B : obtuse_angle B) (obtuse_C : obtuse_angle C) :
  diagonal_length B D > diagonal_length A C :=
sorry

end quadrilateral_diagonal_length_l353_353855


namespace largest_integer_less_than_100_with_remainder_5_l353_353583

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353583


namespace circle_radius_l353_353406

theorem circle_radius {θ : ℝ} (x y : ℝ) 
  (hx : x = 3 * Math.sin θ + 4 * Math.cos θ) 
  (hy : y = 4 * Math.sin θ - 3 * Math.cos θ) : 
  x^2 + y^2 = 25 := 
by 
  sorry

end circle_radius_l353_353406


namespace smallest_S_same_probability_sum_l353_353035

theorem smallest_S_same_probability_sum (n : ℕ) (d : fin n → ℕ) (S R : ℕ) 
  (h1 : ∀ i, 1 ≤ d i ∧ d i ≤ 6)
  (h2 : R = ∑ i, d i)
  (h3 : S = ∑ i, 7 - d i)
  (h4 : R = 1994) 
  (h5 : n = 333) :
  S = 337 :=
by {
  -- To be proved
  sorry
}

end smallest_S_same_probability_sum_l353_353035


namespace smallest_possible_value_of_a_l353_353821

theorem smallest_possible_value_of_a (P : ℤ[X]) (a : ℤ) (h₁ : a > 0)
  (h₂ : P.eval 1 = a) (h₃ : P.eval 3 = a) (h₄ : P.eval 5 = a) (h₅ : P.eval 7 = a)
  (h₆ : P.eval 2 = -a) (h₇ : P.eval 4 = -a) (h₈ : P.eval 6 = -a) (h₉ : P.eval 8 = -a) :
  a = 315 :=
sorry

end smallest_possible_value_of_a_l353_353821


namespace max_MN_dist_l353_353679

def rho_equation (θ : ℝ) : ℝ := 2 * sin θ

def line_equation (t : ℝ) : ℝ × ℝ := 
  let x := - (3 / 5 : ℝ) * t + 2
  let y := (4 / 5 : ℝ) * t
  (x, y)

def cartesian_curve (x y : ℝ) : Prop := x^2 + y^2 = 2 * y

def parameter_line (x y t : ℝ) : Prop := 
  (x = - (3 / 5 : ℝ) * t + 2) ∧ (y = (4 / 5 : ℝ) * t)

def M_x_coord : ℝ := 2

axiom point_on_curve_C {x y : ℝ} (h : cartesian_curve x y) : Prop

theorem max_MN_dist : ∃ (N : ℝ × ℝ), point_on_curve_C (N.1, N.2) ∧ 
  ∀ (M : ℝ × ℝ), (M.1 = M_x_coord ∧ M.2 = 0) → 
  (∃ (D : ℝ), D = sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) ∧
     D ≤ sqrt 5 + 1 ∧ 
     ∀ (t : ℝ), parameter_line M.1 M.2 t) ∧
     sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = (sqrt 61 / 5) :=
sorry

end max_MN_dist_l353_353679


namespace factor_correct_l353_353146

noncomputable def factor_expression (x : ℝ) : ℝ :=
  66 * x^6 - 231 * x^12

theorem factor_correct (x : ℝ) :
  factor_expression x = 33 * x^6 * (2 - 7 * x^6) :=
by 
  sorry

end factor_correct_l353_353146


namespace unknown_cell_is_red_l353_353434

-- Definitions for the cube and the problem conditions
structure Cube3D :=
  (size: ℕ := 27)
  (block_size: ℕ := 3)
  (visible_colored_cells: ℕ := 8)

-- The main proposition to prove 
theorem unknown_cell_is_red (cube: Cube3D) (h1: cube.size = 27) (h2: cube.block_size = 3) (h3: cube.visible_colored_cells = 8) : 
  ∃ (x: ℕ) (color: char), color = 'R' :=
by 
  sorry

end unknown_cell_is_red_l353_353434


namespace find_x_plus_y_l353_353209

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y = 2009) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 + Real.pi / 2 := 
by
  sorry

end find_x_plus_y_l353_353209


namespace MrFletcher_paid_l353_353359

noncomputable def total_payment (hours_day1 hours_day2 hours_day3 rate_per_hour men : ℕ) : ℕ :=
  let total_hours := hours_day1 + hours_day2 + hours_day3
  let total_man_hours := total_hours * men
  total_man_hours * rate_per_hour

theorem MrFletcher_paid
  (hours_day1 hours_day2 hours_day3 : ℕ)
  (rate_per_hour men : ℕ)
  (h1 : hours_day1 = 10)
  (h2 : hours_day2 = 8)
  (h3 : hours_day3 = 15)
  (h4 : rate_per_hour = 10)
  (h5 : men = 2) :
  total_payment hours_day1 hours_day2 hours_day3 rate_per_hour men = 660 := 
by {
  -- skipped proof details
  sorry
}

end MrFletcher_paid_l353_353359


namespace distance_between_foci_l353_353885

/-- Given the ellipse defined by the equation 
    √((x-4)^2 + (y-3)^2) + √((x+6)^2 + (y-5)^2) = 26,
    the distance between the foci is 2√26. -/
theorem distance_between_foci : 
  ∀ (x y : ℝ),
  (Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 5)^2) = 26)
  → (Real.sqrt ((4 + 6)^2 + (3 - 5)^2) = 2 * Real.sqrt 26) :=
by
  sorry

end distance_between_foci_l353_353885


namespace sum_of_legs_is_104_l353_353886

theorem sum_of_legs_is_104 (x : ℕ) (h₁ : x^2 + (x + 2)^2 = 53^2) : x + (x + 2) = 104 := sorry

end sum_of_legs_is_104_l353_353886


namespace asymptotes_of_hyperbola_l353_353882

theorem asymptotes_of_hyperbola : 
  ∀ x y : ℝ, x ^ 2 - y ^ 2 / 3 = 1 → (y = sqrt 3 * x ∨ y = - sqrt 3 * x) :=
by 
  intros x y h,
  -- The proof would go here
  sorry

end asymptotes_of_hyperbola_l353_353882


namespace gcd_3_666666666_equals_3_l353_353825

theorem gcd_3_666666666_equals_3 :
  Nat.gcd 33333333 666666666 = 3 := by
  sorry

end gcd_3_666666666_equals_3_l353_353825


namespace product_of_constants_l353_353163

theorem product_of_constants (h_factorized : ∀ t : ℤ, 
    ∃ a b : ℤ, x^2 + t * x - 24 = (x + a) * (x + b) ∧ a * b = -24) : 
    ∏ t in {t : ℤ | ∃ a b : ℤ, a * b = -24 ∧ t = a + b}, t = -10580000 :=
sorry

end product_of_constants_l353_353163


namespace arc_length_of_curve_l353_353112

noncomputable def x (t : ℝ) := Real.exp t * (Real.cos t + Real.sin t)
noncomputable def y (t : ℝ) := Real.exp t * (Real.cos t - Real.sin t)

theorem arc_length_of_curve : 
  ∫ t in 0..(3 * Real.pi / 2), Real.sqrt ((2 * Real.exp t * Real.cos t)^2 + (-2 * Real.exp t * Real.sin t)^2) = 
  2 * (Real.exp (3 * Real.pi / 2) - 1) := 
by
  sorry

end arc_length_of_curve_l353_353112


namespace piles_weight_correct_l353_353096

/-- Define constants -/
def initial_weight_kg : ℝ := 4.5          -- Initial weight in kilograms
def weight_eaten_kg : ℝ := 0.85           -- Weight eaten in kilograms
def num_piles : ℕ := 7                    -- Number of piles

/-- Calculate remaining weight -/
def remaining_weight_kg : ℝ := initial_weight_kg - weight_eaten_kg

/-- Weight of each pile as given in conditions -/
def weight_per_pile_kg : ℝ := 0.52

/-- The theorem we want to prove -/
theorem piles_weight_correct :
  (remaining_weight_kg / num_piles).round_to 2 = weight_per_pile_kg :=
by
  sorry

end piles_weight_correct_l353_353096


namespace ceil_sqrt_200_eq_15_l353_353538

theorem ceil_sqrt_200_eq_15 : Int.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l353_353538


namespace no_real_solution_l353_353962

open Real

noncomputable def system_of_equations (x y k d : ℝ) : Prop :=
  x^3 + y^3 = 2 ∧ y = k*x + d

theorem no_real_solution (k d : ℝ) :
  k = -1 ∧ (d < 0 ∨ d > 2) → ∀ x y : ℝ, ¬system_of_equations x y k d :=
begin
  sorry
end

end no_real_solution_l353_353962


namespace ND_is_angle_bisector_l353_353768

-- Definitions for the given conditions
variables (ABC : Type) [Triangle ABC] [acute_scalene_triangle ABC]
variables (A B C : Point ABC) (X N D : Point ABC)
variables (l_b l_c : Line ABC)
variables (Y Z : Point ABC)
variables (tangent_to_circle : Line ABC → Point ABC → Circle ABC → Prop)
variables (external_angle_bisector : Triangle ABC → Angle → Line ABC)

-- Specific conditions
variable (meet_BC : Line ABC → Line ABC → Point ABC)
variable (intersects_at : Line ABC → Line ABC → Point ABC)

-- Prove the final statement
theorem ND_is_angle_bisector :
  tangent_to_circle l_b B (circumcircle ABC) →
  tangent_to_circle l_c C (circumcircle ABC) →
  (line_through X l_b) = Y →
  (line_through X l_c) = Z →
  (intersect_circles (circumcircle AYB) (circumcircle AZC)) = N →
  (intersect_lines l_b l_c) = D →
  bisects_angle D N Y Z :=
by
  sorry

end ND_is_angle_bisector_l353_353768


namespace greatest_unexpressible_sum_l353_353553

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem greatest_unexpressible_sum : 
  ∀ (n : ℕ), (∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n) → n ≤ 11 :=
by
  sorry

end greatest_unexpressible_sum_l353_353553


namespace num_right_triangles_rectangle_l353_353397

def is_right_triangle (A B C : Point) : Prop := 
  -- Definition of a right triangle, e.g., one of the angles is 90 degrees
  sorry

def number_of_right_triangles (points : List Point) : Nat := 
  -- Function to count right triangles among the given points
  sorry

theorem num_right_triangles_rectangle (A P B C R D : Point)
  (h_rectangle : is_rectangle A B C D)
  (h_mid_segment : is_mid_segment P R A D B C) :
  number_of_right_triangles [A, P, B, C, R, D] = 8 :=
  sorry

end num_right_triangles_rectangle_l353_353397


namespace questionnaires_drawn_from_D_l353_353487

theorem questionnaires_drawn_from_D (a b c d : ℕ) (A_s B_s C_s D_s: ℕ) (common_diff: ℕ)
  (h1 : a + b + c + d = 1000)
  (h2 : b = a + common_diff)
  (h3 : c = a + 2 * common_diff)
  (h4 : d = a + 3 * common_diff)
  (h5 : A_s = 30 - common_diff)
  (h6 : B_s = 30)
  (h7 : C_s = 30 + common_diff)
  (h8 : D_s = 30 + 2 * common_diff)
  (h9 : A_s + B_s + C_s + D_s = 150)
  : D_s = 60 := sorry

end questionnaires_drawn_from_D_l353_353487


namespace find_BD_when_AC_over_AB_min_l353_353283

open Real

-- Define the conditions
variables (A B C D : Point)
variables (AD BC BD CD : ℝ)
variables (angle_ADB : ℝ)
variables (AC AB : ℝ)

-- Assign conditions
def conditions : Prop := 
  angle_ADB = 120 ∧
  AD = 2 ∧
  CD = 2 * BD ∧
  AC / AB = minimized

-- Target to prove
theorem find_BD_when_AC_over_AB_min (A B C D : Point) (BD : ℝ)
  (h : conditions A B C D 2 120 BD 2 * BD AC AB) :
  BD = sqrt 3 - 1 :=
sorry

end find_BD_when_AC_over_AB_min_l353_353283


namespace probability_is_43_over_150_l353_353149

-- Define the total number of outcomes
def total_outcomes : ℕ := 15 * 10

-- Define the set of perfect squares within the range of tile and die values
def perfect_squares_set : Finset ℕ := {1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144}.toFinset

-- Define the favorable outcomes
def favorable_outcomes : Finset (ℕ × ℕ) :=
  (Finset.range 15).product (Finset.range 10) |>
  Finset.filter (λ ⟨t, d⟩, ((t + 1) * (d + 1)) ∈ perfect_squares_set)

-- Calculate the probability as the fraction of favorable outcomes over total outcomes
def probability : ℚ := favorable_outcomes.card / total_outcomes

-- The statement to be proved
theorem probability_is_43_over_150 : probability = 43 / 150 := sorry

end probability_is_43_over_150_l353_353149


namespace largest_integer_less_than_100_with_remainder_5_l353_353637

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l353_353637


namespace sum_of_products_l353_353796

open Finset

noncomputable def M (n : ℕ) : Finset ℤ :=
  (range n).map (λ i, -↑(i + 1))

def products (s : Finset ℤ) : Finset ℤ := if h : s.nonempty then {s.prod id} else ∅

theorem sum_of_products (n : ℕ) (h : 1 ≤ n) :
  ∑ t in (M n).powerset.filter (λ s, s.nonempty), (t.prod id) = -1 := by
  sorry

end sum_of_products_l353_353796


namespace dice_probability_l353_353918

def prob_at_least_one_one : ℚ :=
  let total_outcomes := 36
  let no_1_outcomes := 25
  let favorable_outcomes := total_outcomes - no_1_outcomes
  let probability := favorable_outcomes / total_outcomes
  probability

theorem dice_probability :
  prob_at_least_one_one = 11 / 36 :=
by
  sorry

end dice_probability_l353_353918


namespace value_of_expression_l353_353036

theorem value_of_expression (A B C : ℝ) (h1 : 1 / A = -3) (h2 : 2 / B = 4) (h3 : 3 / C = 1 / 2) : 
  6 * A - 8 * B + C = 0 :=
begin
  sorry  -- Proof to be provided later.
end

end value_of_expression_l353_353036


namespace minimum_length_MN_l353_353836

theorem minimum_length_MN (edge_length : ℝ) (x y : ℝ) 
  (M_on_AA1 : ∃ t : ℝ, M = A + t * (A_1 - A))
  (N_on_BC : ∃ t : ℝ, N = B + t * (C - B))
  (MN_intersects_C1D1 : ∃ L : Point, L ∈ MN ∧ L ∈ C1D1) 
  (cube_edge_length : edge_length = 1):
  ∃ M N : Point, ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (A_1M = x) ∧ (BN = y) ∧ distance M N = 3 := 
by
  sorry

end minimum_length_MN_l353_353836


namespace tan_alpha_complex_expression_l353_353671

theorem tan_alpha (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : sin α = 4/5) : tan α = -4/3 :=
by
  sorry

theorem complex_expression (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : sin α = 4/5) :
  (sin (π + α) - 2 * cos (π / 2 + α)) / (-sin (-α) + cos (π - α)) = 4/7 :=
by
  sorry

end tan_alpha_complex_expression_l353_353671


namespace volume_of_water_in_tank_l353_353475

-- Definitions of conditions
def radius : ℝ := 5
def height : ℝ := 10
def water_depth : ℝ := 3

-- Proof problem statement in Lean 4
theorem volume_of_water_in_tank :
  (let r := radius;
       h := height;
       d := water_depth in
       10 * ((132.84 / 360 * π * r^2) - 3 * sqrt 21)) = 102.475 * π - 30 * sqrt 21 :=
  by sorry

end volume_of_water_in_tank_l353_353475


namespace angle_DHC_measure_l353_353753

theorem angle_DHC_measure {α β γ : ℝ} 
  (hα : α = 80) 
  (hβ : β = 60) 
  (hγ : γ = 40) 
  (AH_is_altitude : ∀ (H : Point) (line_AH : Line), Altitude A H line_AH)
  (BD_is_angle_bisector : ∀ (D : Point) (line_BD : Line), AngleBisector B D line_BD) :
  ∠DHC = 20 :=
by
  sorry

end angle_DHC_measure_l353_353753


namespace cone_vertex_angle_proof_l353_353851

noncomputable def vertex_angle_of_cone (r d : ℝ) : ℝ :=
  if r = 4 ∧ d = 5 then
    let tan_phi := r / d in
    if tan_phi = (4 / 5) then
      let alpha := real.atan(1 / tan_phi) in
      let beta := real.acos(3 / 5) in
      if alpha = real.pi / 4 ∨ tan_phi = 1 / real.tan(alpha) then
        2 * alpha
      else
        real.pi / 2
    else
      0
  else
    0

theorem cone_vertex_angle_proof (r d : ℝ) (cond1 : r = 4) (cond2 : d = 5) :
  vertex_angle_of_cone r d = (real.pi / 2) ∨ vertex_angle_of_cone r d = 2 * real.arccot 4 :=
by
  sorry

end cone_vertex_angle_proof_l353_353851


namespace value_two_sd_below_mean_l353_353880

theorem value_two_sd_below_mean (mean : ℝ) (std_dev : ℝ) (h_mean : mean = 17.5) (h_std_dev : std_dev = 2.5) : 
  mean - 2 * std_dev = 12.5 := by
  -- proof omitted
  sorry

end value_two_sd_below_mean_l353_353880


namespace inequality_solution_set_l353_353678

-- Function definitions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) = f(-x)

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, a ≤ x1 → x2 ≤ b → x1 ≤ x2 → f(x1) ≥ f(x2)

-- Hypothesis for the problem
variable (f : ℝ → ℝ)

-- Translated problem statement
theorem inequality_solution_set :
  is_even_function (λ x, f(x + 1)) →
  is_monotonically_decreasing f 1 (λ x, x) →
  ∀ x : ℝ, (f(2x - 1) > f(x + 2)) ↔ (1 / 3 < x ∧ x < 3) :=
sorry

end inequality_solution_set_l353_353678


namespace num_integers_between_sqrt10_sqrt100_l353_353711

theorem num_integers_between_sqrt10_sqrt100 : 
  ∃ n : ℕ, n = 7 ∧ ∀ x : ℤ, (⌊Real.sqrt 10⌋ + 1 <= x) ∧ (x <= ⌈Real.sqrt 100⌉ - 1) ↔ (4 <= x ∧ x <= 10) := 
by 
  sorry

end num_integers_between_sqrt10_sqrt100_l353_353711


namespace pascal_sixth_element_row_20_l353_353443

theorem pascal_sixth_element_row_20 : (Nat.choose 20 5) = 7752 := 
  by
  sorry

end pascal_sixth_element_row_20_l353_353443


namespace polynomial_irreducible_segment_intersect_l353_353466

-- Part (a)
theorem polynomial_irreducible 
  (f : Polynomial ℤ) 
  (h_def : f = Polynomial.C 12 + Polynomial.X * Polynomial.C 9 + Polynomial.X^2 * Polynomial.C 6 + Polynomial.X^3 * Polynomial.C 3 + Polynomial.X^4) : 
  ¬ ∃ (p q : Polynomial ℤ), (Polynomial.degree p = 2) ∧ (Polynomial.degree q = 2) ∧ (f = p * q) :=
sorry

-- Part (b)
theorem segment_intersect 
  (n : ℕ) 
  (segments : Fin (2*n+1) → Set (ℝ × ℝ)) 
  (h_intersect : ∀ i, ∃ n_indices : Finset (Fin (2*n+1)), n_indices.card = n ∧ ∀ j ∈ n_indices, (segments i ∩ segments j).Nonempty) :
  ∃ i, ∀ j, i ≠ j → (segments i ∩ segments j).Nonempty :=
sorry


end polynomial_irreducible_segment_intersect_l353_353466


namespace marcie_loan_difference_l353_353357

theorem marcie_loan_difference :
  let P : ℝ := 20000
  let r1 : ℝ := 0.08
  let n1 : ℕ := 2
  let t1 : ℝ := 10
  let t2 : ℝ := 5
  let r2 : ℝ := 0.09
  let t_total : ℝ := 15

  -- Option 1 calculations (compounded semi-annually)
  let A1 := P * (1 + r1 / n1)^ (n1 * t1)
  let payment1 := A1 / 3
  let remaining1 := A1 - payment1
  let total_after_15_years := remaining1 * (1 + r1 / n1)^ (n1 * t2)
  let total_payment1 := payment1 + total_after_15_years

  -- Option 2 calculations (simple interest)
  let total_payment2 := P + t_total * r2 * P

  -- Difference
  let difference := total_payment1 - total_payment2

  difference ≈ 11520 := by
  sorry

end marcie_loan_difference_l353_353357


namespace sequence_010101_not_appear_l353_353514

theorem sequence_010101_not_appear :
  ∀ a : ℕ → ℕ,
  (a 1 = 1 ∧ a 2 = 0 ∧ a 3 = 1 ∧ a 4 = 0 ∧ a 5 = 1 ∧ a 6 = 0 ∧ a 7 = 2 ∧ a 8 = 3) ∧
  (∀ n, n ≥ 7 → a(n) ≡ (a(n-4) + a(n-3) + a(n-2) + a(n-1)) [MOD 10]) →
  ¬(∃ n, a(n) = 0 ∧ a(n+1) = 1 ∧ a(n+2) = 0 ∧ a(n+3) = 1 ∧ a(n+4) = 0 ∧ a(n+5) = 1) :=
by
  sorry

end sequence_010101_not_appear_l353_353514


namespace Mark_has_23_kangaroos_l353_353840

theorem Mark_has_23_kangaroos :
  ∃ K G : ℕ, G = 3 * K ∧ 2 * K + 4 * G = 322 ∧ K = 23 :=
by
  sorry

end Mark_has_23_kangaroos_l353_353840


namespace selection_two_people_with_at_least_one_boy_l353_353864

theorem selection_two_people_with_at_least_one_boy:
  ∀ (boys girls : ℕ),
  boys = 4 → girls = 2 →
  let total_people := boys + girls in
  let total_ways := (total_people.choose 2) in
  let ways_with_no_boys := (girls.choose 2) in
  total_ways - ways_with_no_boys = 14 :=
by
  intros boys girls h_boys h_girls
  simp [h_boys, h_girls]
  sorry

end selection_two_people_with_at_least_one_boy_l353_353864


namespace evaluate_expression_l353_353541

theorem evaluate_expression :
  (3 + 6 + 9) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 6 + 9) = 5 / 6 :=
by sorry

end evaluate_expression_l353_353541


namespace minimal_n_for_real_root_l353_353695

noncomputable def P (n : ℕ) (coeffs : Fin (2 * n + 1) → ℝ) (x : ℝ) : ℝ :=
  ∑ i in Finset.range (2 * n + 1), coeffs ⟨i, by linarith⟩ * x ^ i

theorem minimal_n_for_real_root :
  ∃ (n : ℕ) (coeffs : Fin (2 * n + 1) → ℝ),
    (∀ i, coeffs i ∈ Set.Icc 100 101) ∧ 
    (∃ x : ℝ, P n coeffs x = 0) ∧
    ∀ m : ℕ, m < n → ¬ (∃ (coeffs' : Fin (2 * m + 1) → ℝ), 
      (∀ i, coeffs' i ∈ Set.Icc 100 101) ∧ 
      (∃ x : ℝ, P m coeffs' x = 0)) :=
  sorry

end minimal_n_for_real_root_l353_353695


namespace inequality_relation_l353_353739

noncomputable def a : ℝ := Real.log 3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 1.2 / Real.log 3
noncomputable def c : ℝ := 0.1^(-0.1)

theorem inequality_relation : a < b ∧ b < c := by
  sorry

end inequality_relation_l353_353739


namespace section_parallel_rect_l353_353969

-- Definitions based on the conditions given
variables {V A B C D E F G : Type*}

-- Tetrahedron with equal edges
def is_regular_tetrahedron (V A B C : Type*) [metric_space A] (d : dist A) :=
∀ (X Y : A), (X ≠ Y) → d X Y = d V A

-- D is a trisection point on AB with ratio 2:1
def trisect_AB (A B D : Type*) [metric_space A] (d : dist A) :=
d A D = 2 * d D B

-- E is a trisection point on AC with ratio 2:1
def trisect_AC (A C E : Type*) [metric_space A] (d : dist A) :=
d A E = 2 * d E C

-- Parallel section passing through D and E and parallel to VA forms a rectangle
theorem section_parallel_rect (V A B C D E F G : Type*) [metric_space A] [is_regular_tetrahedron V A B C (dist A)]
  (trisect_AB A B D (dist A)) (trisect_AC A C E (dist A)) :
  parallelogram (D E F G) → is_rectangle (D E F G) :=
sorry

end section_parallel_rect_l353_353969


namespace part1_part2_l353_353690

section Part1
variable (a x : ℝ)
noncomputable def f (x : ℝ) := x * real.log x - a * x^2

-- Part 1 Problem Statement
theorem part1 (h1 : 1 < x) (h2 : x < 3) :
  (f x + a * x^2 - x + 2) / ((3 - x) * real.exp x) > real.exp (-2) := sorry

end Part1

section Part2
variable (a x : ℝ)

noncomputable def f (x : ℝ) := x * real.log x - a * x^2
noncomputable def F (x : ℝ) := |f x|

-- Part 2 Problem Statement
theorem part2 (h1 : 1 ≤ x) (h2 : x ≤ real.exp 1) :
  ∃ r : set ℝ, (F x).Inf ∈ r ∧
  (r = set.Ioo 0 (real.exp (-1)) ∪ set.Ioo (real.exp (-1)) (1 / 2)) := sorry

end Part2

end part1_part2_l353_353690


namespace doubled_container_volume_l353_353483

theorem doubled_container_volume (original_volume : ℕ) (factor : ℕ) 
  (h1 : original_volume = 4) (h2 : factor = 8) : original_volume * factor = 32 :=
by 
  rw [h1, h2]
  norm_num

end doubled_container_volume_l353_353483


namespace water_level_after_opening_valve_l353_353428

theorem water_level_after_opening_valve (
  h : ℝ,
  ρ_w : ℝ,
  ρ_o : ℝ
) (h_initial : h = 40) (ρ_w_val : ρ_w = 1000) (ρ_o_val : ρ_o = 700) :
  let h_w := (80 * 7) / 17 in
  h_w ≈ 34 := 
by
  sorry

end water_level_after_opening_valve_l353_353428


namespace relationship_between_a_b_c_d_l353_353653

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x)
noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.sin x)

open Real

theorem relationship_between_a_b_c_d :
  ∀ (x : ℝ) (a b c d : ℝ),
  (0 ≤ x ∧ x ≤ π) →
  (∀ x, f x ≤ a ∧ b ≤ f x) →
  (∀ x, g x ≤ c ∧ d ≤ g x) →
  a = sin 1 →
  b = -sin 1 →
  c = 1 →
  d = cos 1 →
  b < d ∧ d < a ∧ a < c := by
  sorry

end relationship_between_a_b_c_d_l353_353653


namespace range_of_m_l353_353396

theorem range_of_m (m : ℝ) : 
    (∀ x : ℝ, mx^2 - 6 * m * x + m + 8 ≥ 0) ↔ (0 ≤ m ∧ m ≤ 1) :=
sorry

end range_of_m_l353_353396


namespace shaded_trapezoids_perimeter_l353_353109

theorem shaded_trapezoids_perimeter :
  let l := 8
  let w := 6
  let half_diagonal_1 := (l^2 + w^2) / 2
  let perimeter := 2 * (w + (half_diagonal_1 / l))
  let total_perimeter := perimeter + perimeter + half_diagonal_1
  total_perimeter = 48 :=
by 
  sorry

end shaded_trapezoids_perimeter_l353_353109


namespace odd_function_a_eq_neg1_fx_geq_2_for_all_x_l353_353184

-- Define the function f(x) = e^x + a * e^(-x)
def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

-- Part 1: Prove that f(x) is odd if and only if a = -1
theorem odd_function_a_eq_neg1 (a : ℝ) : 
    (∀ x : ℝ, f (-x) a = -f x a) ↔ a = -1 := 
sorry

-- Part 2: Prove that f(x) ≥ 2 for all x if and only if a ≥ 1
theorem fx_geq_2_for_all_x (a : ℝ) : 
    (∀ x : ℝ, f x a ≥ 2) ↔ 1 ≤ a := 
sorry

end odd_function_a_eq_neg1_fx_geq_2_for_all_x_l353_353184


namespace vertex_and_maximum_l353_353168

-- Define the quadratic equation
def quadratic (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 9

-- Prove that the vertex of the parabola quadratic is (1, -6) and it is a maximum point
theorem vertex_and_maximum :
  (∃ x y : ℝ, (quadratic x = y) ∧ (x = 1) ∧ (y = -6)) ∧
  (∀ x : ℝ, quadratic x ≤ quadratic 1) :=
sorry

end vertex_and_maximum_l353_353168


namespace min_value_cx_plus_dy_squared_l353_353199

theorem min_value_cx_plus_dy_squared
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (∃ (x y : ℝ), a * x^2 + b * y^2 = 1 ∧ ∀ (x y : ℝ), a * x^2 + b * y^2 = 1 → c * x + d * y^2 ≥ -c / a.sqrt) :=
sorry

end min_value_cx_plus_dy_squared_l353_353199


namespace parallel_line_intersection_l353_353702

variables {Line Plane : Type*} [linear_space Line Plane]

variables (a b : Line) (α β : Plane)

-- Definitions of lines parallel to planes and intersection of planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def intersect_plane (p1 p2 : Plane) (l : Line) : Prop := sorry

theorem parallel_line_intersection :
  parallel_line_plane a α →
  parallel_line_plane a β →
  intersect_plane α β b →
  parallel_line_plane a b :=
sorry

end parallel_line_intersection_l353_353702


namespace min_ratio_bd_l353_353316

noncomputable def triangle_ratio_min (BD : ℝ) : Prop :=
  ∃ K L : ℝ, 
    (K = 1) ∧ 
    (L = 1) ∧ 
    (∀ (x : ℝ), ∀ (y : ℝ), (CD = 2 * BD) → ((x = sqrt(3) - 1) → y = (bd + 1 + 3 / (bd + 1)) ∧ (x + y) = 2 * K * L) ∧ 
    (abs x ≤ (sqrt 3 - 1)))


theorem min_ratio_bd (BD : ℝ) : triangle_ratio_min BD :=
by { sorry }

end min_ratio_bd_l353_353316


namespace minimal_n_for_polynomial_real_root_l353_353697

/-- 
Given a polynomial P(x) = a_{2n} x^{2n} + a_{2n-1} x^{2n-1} + ... + a_1 x + a_0
where each coefficient a_i belongs to the interval [100, 101].
Prove that the minimal n for which the polynomial P(x) can have a real root is 100.
-/
theorem minimal_n_for_polynomial_real_root (P : ℝ[X]) (n : ℕ) (hP : ∀ i, P.coeff i ∈ set.Icc (100 : ℝ) 101) :
  (∃ x : ℝ, P.eval x = 0) ↔ n = 100 :=
sorry

end minimal_n_for_polynomial_real_root_l353_353697


namespace marks_lost_per_wrong_answer_l353_353769

theorem marks_lost_per_wrong_answer (score_per_correct : ℕ) (total_questions : ℕ) 
(total_score : ℕ) (correct_attempts : ℕ) (wrong_attempts : ℕ) (marks_lost_total : ℕ)
(H1 : score_per_correct = 4)
(H2 : total_questions = 75)
(H3 : total_score = 125)
(H4 : correct_attempts = 40)
(H5 : wrong_attempts = total_questions - correct_attempts)
(H6 : marks_lost_total = (correct_attempts * score_per_correct) - total_score)
: (marks_lost_total / wrong_attempts) = 1 := by
  sorry

end marks_lost_per_wrong_answer_l353_353769


namespace largest_int_less_than_100_with_remainder_5_l353_353560

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l353_353560


namespace other_root_of_quadratic_l353_353208

theorem other_root_of_quadratic (a b c : ℚ) (x₁ x₂ : ℚ) :
  a ≠ 0 →
  x₁ = 4 / 9 →
  (a * x₁^2 + b * x₁ + c = 0) →
  (a = 81) →
  (b = -145) →
  (c = 64) →
  x₂ = -16 / 9
:=
sorry

end other_root_of_quadratic_l353_353208


namespace factorial_divisibility_l353_353799

theorem factorial_divisibility {n : ℕ} (h : 2011^(2011) ∣ n!) : 2011^(2012) ∣ n! :=
sorry

end factorial_divisibility_l353_353799


namespace hyperbola_eccentricity_l353_353264

theorem hyperbola_eccentricity (a : ℝ) (h1 : (2, 0) = (2, 0)) 
    (h2 : 1 = 1) : eccentricity ((1 : ℝ)) = (2 : ℝ) :=
by
  sorry

end hyperbola_eccentricity_l353_353264


namespace binom_two_eq_l353_353940

theorem binom_two_eq (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binom_two_eq_l353_353940


namespace total_fish_l353_353391

theorem total_fish (n : ℕ) (t : ℕ) (f : ℕ) :
  n = 32 ∧ t = 1 ∧ f = 31 ∧ ∃ (fish_count_table : ℕ → ℕ), 
  (fish_count_table(t) = 3) ∧ (∀ i, 1 ≤ i ∧ i <= f → fish_count_table(i + t) = 2) → 
  (∑ i in finset.range (t + f), fish_count_table (i + 1)) = 65 :=
by
  sorry

end total_fish_l353_353391


namespace find_BD_when_AC_over_AB_min_l353_353279

open Real

-- Define the conditions
variables (A B C D : Point)
variables (AD BC BD CD : ℝ)
variables (angle_ADB : ℝ)
variables (AC AB : ℝ)

-- Assign conditions
def conditions : Prop := 
  angle_ADB = 120 ∧
  AD = 2 ∧
  CD = 2 * BD ∧
  AC / AB = minimized

-- Target to prove
theorem find_BD_when_AC_over_AB_min (A B C D : Point) (BD : ℝ)
  (h : conditions A B C D 2 120 BD 2 * BD AC AB) :
  BD = sqrt 3 - 1 :=
sorry

end find_BD_when_AC_over_AB_min_l353_353279


namespace lcm_5_6_10_12_l353_353948

open Nat

theorem lcm_5_6_10_12 : lcm (lcm (lcm 5 6) 10) 12 = 60 := by
  sorry

end lcm_5_6_10_12_l353_353948


namespace problem1_domain_of_ln_sin_sin_theta_condition_parallel_lines_correct_propositions_l353_353465

theorem problem1_domain_of_ln_sin (k : ℤ) : domain (λ x, log (2 * sin x - sqrt 2)) = set.Ioo (2 * k * real.pi + real.pi / 4) (2 * k * real.pi + 3 * real.pi / 4) :=
sorry

theorem sin_theta_condition (f : ℝ → ℝ) (θ : ℝ) (k : ℤ) (hx : ∃ x, f x = 0) : θ = k * real.pi + real.pi / 2 :=
sorry

theorem parallel_lines (m : ℝ) : (6 * x + m * y - 1 = 0) ∧ (2 * x - y + 1 = 0) → m = -3 :=
sorry

theorem correct_propositions (α : ℝ) : 
  let f := λ x, 2 * sin (x + real.pi / 4) in
  ∀ α,
    (∃ α ∈ set.Ioo (-real.pi / 2) 0, f α = sqrt 2 = false) ∧
    (∃ α ∈ set.Ioo 0 (real.pi / 2), ∀ x, f (x - α) = f (x + α) = false) ∧
    (∃ α ∈ set.univ, ∀ x, f (x + α) is symmetric about origin) ∧
    (f symmetric about line x = -3 * real.pi / 4) ∧
    (shift f left by real.pi / 4 gives y = -2 * cos x = false) :=
sorry

end problem1_domain_of_ln_sin_sin_theta_condition_parallel_lines_correct_propositions_l353_353465


namespace find_BD_when_AC_over_AB_min_l353_353330

-- Definition of the conditions
variables (A B C D : Type) [metric_space A]
variables (triangle : set A)
variables (on_side_BC : D ∈ segment B C)
variables (angle_ADB : ∠ADB = 120)
variables (AD_len : dist A D = 2)
variables (CD_2BD : dist C D = 2 * dist B D)

noncomputable def minimum_ratio_condition (BD : ℝ) : Prop := 
  let x := dist B D
  let CD := 2 * x
  let AD := 2 in
  let angle_ADB_cos := -1/2 in
  let b_squared := AD^2 + CD^2 - 2 * AD * CD * angle_ADB_cos in
  let c_squared := AD^2 + x^2 + 2 * AD * x * angle_ADB_cos in
  b_squared / c_squared = 4 - (12 / (x + 1 + 3/(x + 1))) → x = sqrt 3 - 1

theorem find_BD_when_AC_over_AB_min (BD : ℝ) : minimum_ratio_condition A B C D 
:= sorry

end find_BD_when_AC_over_AB_min_l353_353330


namespace log_base_2_of_one_fourth_equals_neg_two_l353_353137

theorem log_base_2_of_one_fourth_equals_neg_two :
  (∀ x : ℝ, 2^x = 1 / 4 → x = -2) := 
begin
  intros x hx,
  sorry
end

end log_base_2_of_one_fourth_equals_neg_two_l353_353137


namespace lcm_factor_l353_353401

theorem lcm_factor (A B : ℕ) (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) 
  (hcf_eq : hcf = 15) (factor1_eq : factor1 = 11) (A_eq : A = 225) 
  (hcf_divides_A : hcf ∣ A) (lcm_eq : Nat.lcm A B = hcf * factor1 * factor2) : 
  factor2 = 15 :=
by
  sorry

end lcm_factor_l353_353401


namespace shaded_trapezoid_area_l353_353418

-- Define the conditions in Lean

structure Square :=
  (side_length : ℝ)

-- Given three squares with specified side lengths
def square1 : Square := ⟨3⟩
def square2 : Square := ⟨5⟩
def square3 : Square := ⟨7⟩

-- Define the trapezoid area calculation based on the given conditions
noncomputable def trapezoid_area (b1 b2 h : ℝ) : ℝ :=
  0.5 * (b1 + b2) * h

-- Given the calculated bases and altitude
def base1 : ℝ := 3 * (7 / 15)
def base2 : ℝ := (3 + 5) * (7 / 15)
def height : ℝ := 5

-- Statement of the theorem
theorem shaded_trapezoid_area :
  trapezoid_area base1 base2 height = 12.825 := by
  sorry

end shaded_trapezoid_area_l353_353418


namespace mutually_exclusive_events_l353_353959

noncomputable def exactly_one_head (tosses : list bool) : Prop :=
  (tosses.filter (λ x, x = tt)).length = 1

noncomputable def exactly_two_heads (tosses : list bool) : Prop :=
  (tosses.filter (λ x, x = tt)).length = 2

noncomputable def at_least_one_head (tosses : list bool) : Prop :=
  (tosses.filter (λ x, x = tt)).length ≥ 1

noncomputable def at_most_one_head (tosses : list bool) : Prop :=
  (tosses.filter (λ x, x = tt)).length ≤ 1

theorem mutually_exclusive_events :
  ∀ tosses : list bool,
  (tosses.length = 2 ∧ at_most_one_head tosses ∧ exactly_two_heads tosses) = false :=
by
  intro tosses
  sorry

end mutually_exclusive_events_l353_353959


namespace mutually_exclusive_not_complementary_l353_353135

/-
Given four colored cards (red, yellow, blue, and white) are distributed randomly to four people (A, B, C, and D), 
and each person receives one card, prove that the event "A gets the red card" and the event "D gets the red card" 
are mutually exclusive but not complementary.
-/
theorem mutually_exclusive_not_complementary 
    (cards : Fin 4 → Prop)  -- Cards: 0 -> Red, 1 -> Yellow, 2 -> Blue, 3 -> White
    (assignment : Fin 4 → Fin 4)  -- Assignment: 0 -> A, 1 -> B, 2 -> C, 3 -> D
    (h_random : ∀ i, ∃ j, cards j ∧ assignment i = j) :
    (assignment 0 = 0) ∧ (assignment 3 = 0) → False /\ 
    (assignment 0 = 0 ∨ assignment 3 = 0) ↔ True :=
begin
  sorry -- Proof not provided as per instructions.
end

end mutually_exclusive_not_complementary_l353_353135


namespace minimum_even_integers_is_two_l353_353429

-- Definitions based on conditions:
def two_integers_sum_30 (x y : ℤ) : Prop := x + y = 30
def two_more_integers_sum_20 (a b : ℤ) : Prop := a + b = 20
def final_two_integers_sum_20 (m n : ℤ) : Prop := m + n = 20

-- Question: Minimum number of even integers among the six integers equals 2
theorem minimum_even_integers_is_two (x y a b m n : ℤ) 
  (h1 : two_integers_sum_30 x y)
  (h2 : two_more_integers_sum_20 a b)
  (h3 : final_two_integers_sum_20 m n)
  : x.even + y.even + a.even + b.even + m.even + n.even ≥ 2 := 
sorry

end minimum_even_integers_is_two_l353_353429


namespace minimum_BD_value_l353_353301

theorem minimum_BD_value {A B C D : Type*} [Real A] [Real B] [Real C] [Real D] (angle_ADB: A) (AD: A) (CD: A) (BD: A):
  (D ∈ line_segment(B, C)) →
  angle_ADB = 120 → 
  AD = 2 → 
  CD = 2 * BD →
  ∃ BD: A, BD = sqrt(3) - 1 :=
by
  sorry

end minimum_BD_value_l353_353301


namespace remainder_17_pow_53_mod_7_l353_353441

theorem remainder_17_pow_53_mod_7 : (17 ^ 53) % 7 = 5 := by
  have h1 : 3 ^ 1 % 7 = 3 := sorry
  have h2 : 3 ^ 2 % 7 = 2 := sorry
  have h3 : 3 ^ 3 % 7 = 6 := sorry
  have h4 : 3 ^ 4 % 7 = 4 := sorry
  have h5 : 3 ^ 5 % 7 = 5 := sorry
  have h6 : 3 ^ 6 % 7 = 1 := sorry
  have h7 : 3 ^ 53 % 7 = 5 := by
    calc
      3 ^ 53 % 7 = 3 ^ (6 * 8 + 5) % 7 : by rw Nat.pow_eq_pow (6 * 8 + 5)
               ... = (3 ^ 6) ^ 8 * 3 ^ 5 % 7 : by rw [pow_add, pow_mul]
               ... = 1 ^ 8 * 3 ^ 5 % 7 : by rw [h6]
               ... = 1 * 3 ^ 5 % 7 : by rw [pow_one]
               ... = 3 ^ 5 % 7 : by rw [mul_one]
               ... = 5 : by rw [h5]
  have h8 : 17 % 7 = 3 := by norm_num
  show (17 ^ 53) % 7 = 5 from calc
    (17 ^ 53) % 7 = (3 ^ 53) % 7 : by rw [←h8]
              ... = 5 : by rw [h7]

end remainder_17_pow_53_mod_7_l353_353441


namespace number_of_integers_between_sqrt10_sqrt100_l353_353730

theorem number_of_integers_between_sqrt10_sqrt100 : 
  (set.Ico ⌈Real.sqrt 10⌉₊ (⌊Real.sqrt 100⌋₊ + 1)).card = 7 := by
  sorry

end number_of_integers_between_sqrt10_sqrt100_l353_353730


namespace extended_triangle_area_l353_353247

theorem extended_triangle_area (a b c : ℝ) (Δ : ℝ) (h : Δ = (1/2) * a * (b * c) / (a + b + c)) :
  let extended_area := 7 * Δ in
  extended_area = 7 * Δ :=
by sorry

end extended_triangle_area_l353_353247


namespace f_is_decreasing_on_M_g_range_l353_353218

-- Define the functions and domain
def f (x : ℝ) : ℝ := real.sqrt (1 - x) - real.log (2 + x)
def M : set ℝ := { x : ℝ | -2 < x ∧ x ≤ 1 }

-- Lean 4 statement for question (1)
theorem f_is_decreasing_on_M : ∀ x ∈ M, ∀ y ∈ M, x < y → f x > f y := 
sorry

-- Define the second function
def g (x : ℝ) : ℝ := 4 ^ x - 2 ^ (x + 1)

-- Lean 4 statement for question (2)
theorem g_range : set.range (λ x : ℝ, if (x ∈ M) then g x else 0) = (Icc (-1: ℝ) 0 : set ℝ) :=
sorry

end f_is_decreasing_on_M_g_range_l353_353218


namespace proof_problem_l353_353204

variables (a b c e : ℝ)
variables (B M N : ℝ × ℝ)
variables (k : ℝ)
variables (F1 F2 P O : ℝ × ℝ)

-- Ellipse equation condition
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1

-- The point P such that the dot product equals zero
def orthogonal (P F2 F1 : ℝ × ℝ) : Prop := 
  let v1 := (fst P - fst F2, snd P - snd F2)
  let v2 := (fst F1 - fst F2, snd F1 - snd F2)
  in (fst v1) * (fst v2) + (snd v1) * (snd v2) = 0

-- Distance condition
def distance_condition (b c : ℝ) : Prop := 
  let lhs := b^2 * c / (sqrt (b^4 + 4 * a^2 * c^2))
  let rhs := c / 3
  in lhs = rhs

-- Perpendicular condition
def perpendicular (B M N : ℝ × ℝ) : Prop := 
  let slope_BM := (snd M - snd B) / (fst M - fst B)
  let slope_BN := (snd N - snd B) / (fst N - fst B)
  in slope_BM * slope_BN = -1

-- Final proof problem statement
theorem proof_problem 
  (h₁ : a > 0) 
  (h₂ : b > 0)
  (h₃ : b < a) 
  (h₄ : ellipse a b (fst P) (snd P)) 
  (h₅ : orthogonal P F2 F1)
  (h₆ : distance_condition b c) 
  (h₇ : ∃ k > 0, B = (0, b) ∧ (∃ (x : ℝ), M = (x, k * x + b) ∧ ellipse a b (fst M) (snd M)))
  (h₈ : perpendicular B M N)
  : ∃ k ∈ (Icc (1/4 : ℝ) (1/2 : ℝ)), (complex.abs (snd N - snd B) = 2 * complex.abs (snd M - snd B)) := sorry

end proof_problem_l353_353204


namespace eight_faucets_fill_time_in_seconds_l353_353646

open Nat

-- Definitions under the conditions
def four_faucets_rate (gallons : ℕ) (minutes : ℕ) : ℕ := gallons / minutes

def one_faucet_rate (four_faucets_rate : ℕ) : ℕ := four_faucets_rate / 4

def eight_faucets_rate (one_faucet_rate : ℕ) : ℕ := one_faucet_rate * 8

def time_to_fill (rate : ℕ) (gallons : ℕ) : ℕ := gallons / rate

-- Main theorem to prove 
theorem eight_faucets_fill_time_in_seconds (gallons_tub : ℕ) (four_faucets_time : ℕ) :
    let four_faucets_rate := four_faucets_rate 200 8
    let one_faucet_rate := one_faucet_rate four_faucets_rate
    let rate_eight_faucets := eight_faucets_rate one_faucet_rate
    let time_fill := time_to_fill rate_eight_faucets 50
    gallons_tub = 50 ∧ four_faucets_time = 8 ∧ rate_eight_faucets = 50 -> time_fill * 60 = 60 :=
by
    intros
    sorry

end eight_faucets_fill_time_in_seconds_l353_353646


namespace derivative_of_function_correct_l353_353158

noncomputable def derivative_of_function (x : ℝ) : ℝ := 
  8 * π^2 * x

theorem derivative_of_function_correct (x : ℝ) : 
  has_deriv_at (λ x, (2 * π * x)^2) (derivative_of_function x) x := by
  sorry

end derivative_of_function_correct_l353_353158


namespace largest_integer_less_than_100_with_remainder_5_l353_353632

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l353_353632


namespace count_integers_between_sqrt_10_and_sqrt_100_l353_353722

theorem count_integers_between_sqrt_10_and_sqrt_100 :
  ∃ n : ℕ, (∀ k : ℕ, 4 ≤ k ∧ k ≤ 10 → k ∈ set.Ico (int.floor (sqrt (10 : ℝ))) (int.ceil (sqrt (100 : ℝ)))) ∧ n = 7 :=
sorry

end count_integers_between_sqrt_10_and_sqrt_100_l353_353722


namespace complex_conjugate_quadrant_l353_353266

theorem complex_conjugate_quadrant :
  let Z := (7 + complex.i) / (3 + 4 * complex.i)
  let Z_conj := complex.conj Z
  (0 < Z_conj.re ∧ 0 < Z_conj.im) :=
by
  sorry

end complex_conjugate_quadrant_l353_353266


namespace largest_integer_less_than_100_with_remainder_5_l353_353606

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l353_353606


namespace find_BD_when_AC_over_AB_min_l353_353284

open Real

-- Define the conditions
variables (A B C D : Point)
variables (AD BC BD CD : ℝ)
variables (angle_ADB : ℝ)
variables (AC AB : ℝ)

-- Assign conditions
def conditions : Prop := 
  angle_ADB = 120 ∧
  AD = 2 ∧
  CD = 2 * BD ∧
  AC / AB = minimized

-- Target to prove
theorem find_BD_when_AC_over_AB_min (A B C D : Point) (BD : ℝ)
  (h : conditions A B C D 2 120 BD 2 * BD AC AB) :
  BD = sqrt 3 - 1 :=
sorry

end find_BD_when_AC_over_AB_min_l353_353284


namespace sum_of_possible_digits_l353_353979

theorem sum_of_possible_digits (d : ℕ) : 
  (∀ n : ℕ, (65536 ≤ n) ∧ (n ≤ 1048575) → (log 4 n).ceil = d) → 
  d = 8 ∨ d = 9 ∨ d = 10 → 
  d = 27 := by 
  sorry

end sum_of_possible_digits_l353_353979


namespace max_value_complex_l353_353340

open Complex

noncomputable def max_value_statement (α γ : ℂ) (h₁ : |γ| = 2) (h₂ : γ ≠ 2 * α) : ℝ :=
\max \left| \frac{\gamma - \alpha}{2 - (α̅ * γ)} \right|

theorem max_value_complex (α γ : ℂ) (h₁ : |γ| = 2) (h₂ : γ ≠ 2 * α) :
    max_value_statement α γ h₁ h₂ = 1 :=
sorry

end max_value_complex_l353_353340


namespace probability_equation_solution_exists_l353_353104

theorem probability_equation_solution_exists :
  ∃ p ∈ Icc 0 1, p^4 * (1 - p)^2 = 64 / 10935 :=
by
  sorry

end probability_equation_solution_exists_l353_353104


namespace faster_growth_f_g_l353_353038

noncomputable def f (x : ℝ) : ℝ := x^3
noncomputable def g (x : ℝ) : ℝ := x^2 * Real.log x

theorem faster_growth_f_g (x : ℝ) (h : x > 0) : 
  ∃ x₀ > 0, ∀ x > x₀, f x > g x := 
by
  sorry

end faster_growth_f_g_l353_353038


namespace arithmetic_seq_of_pos_and_arithmetic_sqrt_S_l353_353815

variable {a : ℕ → ℝ} (h_pos : ∀ n, a n > 0)
variable (h_a2 : a 2 = 3 * a 1)
variable (S : ℕ → ℝ)
variable (h_S : ∀ n, S n = ∑ i in Finset.range(n+1), a i)
variable (h_sqrt_S_arith : ∃ d, ∀ n, (Sqrt.sqrt (S n) - Sqrt.sqrt (S (n - 1))) = d)

theorem arithmetic_seq_of_pos_and_arithmetic_sqrt_S : 
  ∀ n, a (n+1) - a n = a 1 := 
sorry

end arithmetic_seq_of_pos_and_arithmetic_sqrt_S_l353_353815


namespace monotonicity_and_bound_for_f_l353_353217

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.ln x - a * x + 1

theorem monotonicity_and_bound_for_f {a : ℝ} :
  ( (a ≤ 0 → ∀ x > 0, 0 < x → (f x a) > (f x 0)) ∧
    (a > 0 → ∀ x > 0, (0 < x ∧ x < 1/a) → (f x a) > (f x a) ∧ ∀ x > 1/a, f x a < f (1/a) a) ∨
    (a = -2 ∧ ∃ m : ℕ, ∀ x > 0, f x (-2) ≤ m * (x + 1) ∧ m = 3)) :=
sorry

end monotonicity_and_bound_for_f_l353_353217


namespace domain_f_x1_l353_353684

noncomputable def f (x : ℝ) : ℝ := Math.log (x / (1 - x))

theorem domain_f_x1 : 
  ∀ x : ℝ, (f (x + 1) ∈ ℝ) ↔ x ∈ set.Ioo (-1 : ℝ) 0 :=
by
  sorry

end domain_f_x1_l353_353684


namespace lattice_points_count_l353_353486

theorem lattice_points_count : 
  let lattice := {p : ℤ × ℤ | ∃ x y : ℤ, p = (x, y)},
      region := {p : ℝ × ℝ | (∃ x y : ℝ, p = (x, y) ∧ y = |x|) ∨ (∃ x y : ℝ, p = (x, y) ∧ y = -x^2 + 8)} in
      let useful_points := {p : ℤ × ℤ | p ∈ lattice ∧ (p.2 = |(p.1 : ℝ)| ∨ p.2 = -(p.1 : ℝ)^2 + 8)} in
      set.finite useful_points ∧ set.card useful_points = 33 :=
by sorry

end lattice_points_count_l353_353486


namespace max_k_value_l353_353700

open Classical

noncomputable def max_k (x y k : ℝ) (h1 : x - 4 * y = k - 1) (h2 : 2 * x + y = k) (h3 : x - y ≤ 0) : ℤ :=
  0

theorem max_k_value :
  ∀ x y k : ℝ, (x - 4 * y = k - 1) → (2 * x + y = k) → (x - y ≤ 0) → 
  max_k x y k (x - 4 * y = k - 1) (2 * x + y = k) (x - y ≤ 0) = 0 :=
by sorry

end max_k_value_l353_353700


namespace log_a_x_minus_one_fixed_point_l353_353971

theorem log_a_x_minus_one_fixed_point (a : ℝ) (x : ℝ) (y : ℝ) (h1: a > 0) (h2: a ≠ 1) (h3: x = 2) (h4: y = log a (x - 1)) :
  (x, y) = (2, 0) :=
sorry

end log_a_x_minus_one_fixed_point_l353_353971


namespace is_same_graph_as_identity_l353_353127

theorem is_same_graph_as_identity :
  (∀ x : ℝ, (∃ y : ℝ, y = (\ln (e ^ x)))) ↔ (∀ x : ℝ, y = x) :=
by
  sorry

end is_same_graph_as_identity_l353_353127


namespace measure_WX_l353_353778

noncomputable def quadrilateral_wxyz
  (WX YZ : ℝ) (WZ YZ_parallel_WX : Prop)
  (angle_Y angle_Z : ℤ)
  (WZ_val YZ_val : ℝ)
  (angle_Z_twice_angle_Y : angle_Z = 2 * angle_Y)
  (WZ_length : WZ = WZ_val) (YZ_length : YZ = YZ_val) : Prop :=
  WX = WZ_val + YZ_val

theorem measure_WX
  (WX YZ : ℝ)
  (WZ YZ_parallel_WX : Prop)
  (angle_Y angle_Z : ℤ)
  (WZ_val YZ_val : ℝ)
  (angle_Z_twice_angle_Y : angle_Z = 2 * angle_Y)
  (WZ_length : WZ = WZ_val)
  (YZ_length : YZ = YZ_val) :
  WX = WZ_val + YZ_val := 
begin
  -- form the quadrilateral and assign properties
  exact quadrilateral_wxyz WX YZ WZ YZ_parallel_WX angle_Y angle_Z WZ_val YZ_val angle_Z_twice_angle_Y WZ_length YZ_length
end

end measure_WX_l353_353778


namespace log_base4_of_8_l353_353140

theorem log_base4_of_8 : log 4 8 = 3 / 2 :=
by
  have h1 : 8 = 2 ^ 3 := by norm_num
  have h2 : 4 = 2 ^ 2 := by norm_num
  rw [h2] -- replace 4 with (2^2)
  rw [←@Real.log_pow 4 (3/2)]
  norm_num
  rw [Real.log_self, mul_one]
  norm_num

end log_base4_of_8_l353_353140


namespace jameson_track_medals_l353_353791

-- Conditions
variable (T : ℕ)
variable (total_medals : ℕ) (track_medals : ℕ) (swimming_medals : ℕ) (badminton_medals : ℕ)

-- Setting values according to conditions
def jameson_conditions :=
  total_medals = 20 ∧
  track_medals = T ∧
  swimming_medals = 2 * T ∧
  badminton_medals = 5

-- Proving the question == answer given conditions
theorem jameson_track_medals : jameson_conditions T total_medals track_medals swimming_medals badminton_medals → T = 5 := 
by 
  intro h
  simp only [jameson_conditions] at h
  obtain ⟨h_total_medals, h_track_medals, h_swimming_medals, h_badminton_medals⟩ := h
  rw [h_total_medals, h_track_medals, h_swimming_medals, h_badminton_medals]
  sorry

end jameson_track_medals_l353_353791


namespace bird_cages_count_l353_353994

/-- 
If each bird cage contains 2 parrots and 2 parakeets,
and the total number of birds is 36,
then the number of bird cages is 9.
-/
theorem bird_cages_count (parrots_per_cage parakeets_per_cage total_birds cages : ℕ)
  (h1 : parrots_per_cage = 2)
  (h2 : parakeets_per_cage = 2)
  (h3 : total_birds = 36)
  (h4 : total_birds = (parrots_per_cage + parakeets_per_cage) * cages) :
  cages = 9 := 
by 
  sorry

end bird_cages_count_l353_353994


namespace largest_integer_less_than_100_div_8_rem_5_l353_353614

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l353_353614


namespace infinite_primes_congruent_one_l353_353435

theorem infinite_primes_congruent_one (r : ℕ) (hr : r ≥ 1) : ∃^∞ p : ℕ, Prime p ∧ p ≡ 1 [MOD 2^r] := 
sorry

end infinite_primes_congruent_one_l353_353435


namespace shaded_area_of_octagon_l353_353270

def side_length := 12
def octagon_area := 288

theorem shaded_area_of_octagon (s : ℕ) (h0 : s = side_length):
  (2 * s * s - 2 * s * s / 2) * 2 / 2 = octagon_area :=
by
  skip
  sorry

end shaded_area_of_octagon_l353_353270


namespace number_of_integers_between_sqrt10_sqrt100_l353_353731

theorem number_of_integers_between_sqrt10_sqrt100 : 
  (set.Ico ⌈Real.sqrt 10⌉₊ (⌊Real.sqrt 100⌋₊ + 1)).card = 7 := by
  sorry

end number_of_integers_between_sqrt10_sqrt100_l353_353731


namespace largest_integer_less_than_100_with_remainder_5_l353_353605

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l353_353605


namespace geom_seq_sum_l353_353360

theorem geom_seq_sum (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < q)
  (h2 : ∀ n, a (n+1) = a n * q)
  (h3 : a 0 + a 1 = 3 / 4)
  (h4 : a 2 + a 3 + a 4 + a 5 = 15) :
  a 6 + a 7 + a 8 = 112 := by
  sorry

end geom_seq_sum_l353_353360


namespace minimum_BD_value_l353_353306

theorem minimum_BD_value {A B C D : Type*} [Real A] [Real B] [Real C] [Real D] (angle_ADB: A) (AD: A) (CD: A) (BD: A):
  (D ∈ line_segment(B, C)) →
  angle_ADB = 120 → 
  AD = 2 → 
  CD = 2 * BD →
  ∃ BD: A, BD = sqrt(3) - 1 :=
by
  sorry

end minimum_BD_value_l353_353306


namespace probability_of_winning_quiz_l353_353086

theorem probability_of_winning_quiz :
  let n := 4 -- number of questions
  let choices := 3 -- number of choices per question
  let probability_correct := 1 / choices -- probability of answering correctly
  let probability_incorrect := 1 - probability_correct -- probability of answering incorrectly
  let probability_all_correct := probability_correct^n -- probability of getting all questions correct
  let probability_exactly_three_correct := 4 * probability_correct^3 * probability_incorrect -- probability of getting exactly 3 questions correct
  probability_all_correct + probability_exactly_three_correct = 1 / 9 :=
by
  sorry

end probability_of_winning_quiz_l353_353086


namespace binom_two_eq_l353_353942

theorem binom_two_eq (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binom_two_eq_l353_353942


namespace op_ø_evaluation_l353_353456

-- Definition of the operation ø
def op_ø (x w : ℕ) : ℕ := (2^x) / (2^w)

-- The theorem we need to prove
theorem op_ø_evaluation : op_ø (op_ø 4 2) 2 = 4 := by
  sorry

end op_ø_evaluation_l353_353456


namespace total_weight_tommy_ordered_l353_353015

theorem total_weight_tommy_ordered :
  let apples := 3
  let oranges := 1
  let grapes := 3
  let strawberries := 3
  apples + oranges + grapes + strawberries = 10 := by
  sorry

end total_weight_tommy_ordered_l353_353015


namespace sum_of_extremes_l353_353517

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

theorem sum_of_extremes :
  (2 ∈ primes) → 
  (47 ∈ primes) → 
  (∀ p ∈ primes, is_prime p ∧ p > 1 ∧ p < 50) →
  (List.head primes + List.get_last primes (by decide) = 49) :=
by
  intros h_min h_max h_primes
  -- proof is omitted
  sorry

end sum_of_extremes_l353_353517


namespace equiangular_iff_regular_polygon_l353_353997

def is_equiangular (polygon : Type) : Prop :=
  ∀ (θ : ℤ), polygon.angles θ = some_value

def is_regular_polygon (polygon : Type) : Prop :=
  polygon.is_equilateral ∧ is_equiangular polygon

theorem equiangular_iff_regular_polygon (P : Type) :
  is_equiangular P ↔ is_regular_polygon P :=
sorry

end equiangular_iff_regular_polygon_l353_353997


namespace fourth_sphere_radius_l353_353180

theorem fourth_sphere_radius
  (r θ : ℝ)
  (θ_val : θ = real.pi / 3)
  (radius_first_spheres : ∀ i ≤ 3, r = 3) : r = 9 - 4 * real.sqrt 2 :=
sorry

end fourth_sphere_radius_l353_353180


namespace max_profit_30000_l353_353985

noncomputable def max_profit (type_A : ℕ) (type_B : ℕ) : ℝ := 
  10000 * type_A + 5000 * type_B

theorem max_profit_30000 :
  ∃ (type_A type_B : ℕ), 
  (4 * type_A + 1 * type_B ≤ 10) ∧
  (18 * type_A + 15 * type_B ≤ 66) ∧
  max_profit type_A type_B = 30000 :=
sorry

end max_profit_30000_l353_353985


namespace concave_number_count_l353_353093

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n ≤ 9

def is_concave_number (a b c : ℕ) : Prop := 
  a > b ∧ b < c ∧ is_digit a ∧ is_digit b ∧ is_digit c

def count_concave_numbers : ℕ :=
  (Finset.range 10).sum (λ b, 
    (Finset.range b).sum (λ a, (Finset.Icc (b+1) 9).sum (λ c, 1)))

theorem concave_number_count : count_concave_numbers = 285 :=
by
  sorry

end concave_number_count_l353_353093


namespace min_value_l353_353741

theorem min_value (x : ℝ) (h : 1 < x) : 
  ∃ y, y = 3 * x + 1 / (x - 1) ∧ y ≥ 2 * real.sqrt 3 + 3 :=
sorry

end min_value_l353_353741


namespace solve_equation_l353_353869

noncomputable def heaviside (x : ℝ) : ℝ :=
if x >= 0 then 1 else 0

theorem solve_equation (x : ℝ) :
  (Real.tan (x * (heaviside (x - 2 * Real.pi) - heaviside (x - 5 * Real.pi))) = (1 / Real.cos x ^ 2 - 1)) →
  (∃ k : ℤ, x = k * Real.pi) ∨ (∃ m : ℤ, m ∈ {2, 3, 4} ∧ x = Real.pi / 4 + m * Real.pi) :=
by
  sorry

end solve_equation_l353_353869


namespace old_edition_pages_l353_353075

theorem old_edition_pages (x : ℕ) 
  (h₁ : 2 * x - 230 = 450) : x = 340 := 
by sorry

end old_edition_pages_l353_353075


namespace proof_problem_l353_353119

def expr_numerator : ℕ → ℕ
| 0     := 3
| (n+1) := 4 * n + 3

def expr_denominator : ℕ → ℕ
| 0     := 4
| (n+1) := 4 * (n+1)

noncomputable def floor_fourth_root (x : ℕ) : ℕ :=
  int.to_nat (int.fourth_root x).to_floor

theorem proof_problem :
  (∏ k in range 504, floor_fourth_root (expr_numerator k)) /
  (∏ k in range 504, floor_fourth_root (expr_denominator (k + 1))) = 5 / 16 :=
by
  sorry

end proof_problem_l353_353119


namespace binom_n_2_l353_353938

theorem binom_n_2 (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_n_2_l353_353938


namespace middle_term_in_arithmetic_sequence_l353_353021

theorem middle_term_in_arithmetic_sequence :
  let a := 3^2 in let c := 3^4 in
  ∃ z : ℤ, (2 * z = a + c) ∧ z = 45 := by
let a := 3^2
let c := 3^4
use (a + c) / 2
split
-- Prove that 2 * ((a + c) / 2) = a + c
sorry
-- Prove that (a + c) / 2 = 45
sorry

end middle_term_in_arithmetic_sequence_l353_353021


namespace number_of_integers_between_sqrt10_sqrt100_l353_353733

theorem number_of_integers_between_sqrt10_sqrt100 : 
  (set.Ico ⌈Real.sqrt 10⌉₊ (⌊Real.sqrt 100⌋₊ + 1)).card = 7 := by
  sorry

end number_of_integers_between_sqrt10_sqrt100_l353_353733


namespace sum_smallest_third_smallest_is_786_l353_353032
open List

noncomputable def permutations_168 : List ℕ := 
(map (λ (l : List ℕ), Nat.ofDigits 10 l) 
  (permutations [1, 6, 8]))

noncomputable def sorted_permutations_168 : List ℕ := 
sorted (<=) permutations_168

def smallest_number : ℕ := head! sorted_permutations_168
def third_smallest_number : ℕ := nthLe sorted_permutations_168 2 (by simp [sorted_permutations_168])

theorem sum_smallest_third_smallest_is_786 :
  smallest_number + third_smallest_number = 786 :=
sorry

end sum_smallest_third_smallest_is_786_l353_353032


namespace eccentricity_hyperbola_l353_353677

-- Conditions
def is_eccentricity_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  let e := (Real.sqrt 2) / 2
  (Real.sqrt (1 - b^2 / a^2) = e)

-- Objective: Find the eccentricity of the given the hyperbola.
theorem eccentricity_hyperbola (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : is_eccentricity_ellipse a b h1 h2) : 
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 3 :=
sorry

end eccentricity_hyperbola_l353_353677


namespace helmet_profit_max_l353_353472

open Real

theorem helmet_profit_max (x : ℝ) (h1 : 60 ≤ x) (h2 : x ≤ 100) :
  let y := -10 * x + 1200 in
  let w := (x - 60) * y in
  w ≤ 9000 ∧ (w = 9000 ↔ x = 90) :=
by
  let y := -10 * x + 1200
  let w := (x - 60) * y
  have : w = -10 * x^2 + 1800 * x - 72000, 
  calc 
    w = (x - 60) * (-10 * x + 1200) : by rw [y]
    ... = -10 * x^2 + 1800 * x - 72000 : by ring
  show 
    -10 * x^2 + 1800 * x - 72000 ≤ 9000 ∧ (-10 * x^2 + 1800 * x - 72000 = 9000 ↔ x = 90)
  sorry

end helmet_profit_max_l353_353472


namespace min_ratio_bd_l353_353315

noncomputable def triangle_ratio_min (BD : ℝ) : Prop :=
  ∃ K L : ℝ, 
    (K = 1) ∧ 
    (L = 1) ∧ 
    (∀ (x : ℝ), ∀ (y : ℝ), (CD = 2 * BD) → ((x = sqrt(3) - 1) → y = (bd + 1 + 3 / (bd + 1)) ∧ (x + y) = 2 * K * L) ∧ 
    (abs x ≤ (sqrt 3 - 1)))


theorem min_ratio_bd (BD : ℝ) : triangle_ratio_min BD :=
by { sorry }

end min_ratio_bd_l353_353315


namespace length_BF_is_1_l353_353847

-- Definitions
def Point := ℝ × ℝ  -- Using tuples to represent points
def is_on_circle (center : Point) (radius : ℝ) (p : Point) : Prop :=
  (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 = radius ^ 2
def is_perpendicular (p1 p2 p3 : Point) : Prop :=
  (p2.1 - p1.1) * (p3.1 - p2.1) + (p2.2 - p1.2) * (p3.2 - p2.2) = 0

-- Variables and Conditions
variables (A B C D E F : Point)
variable (r : ℝ)
variable (center : Point)
variable (radius : ℝ)
variable (AE BF : ℝ)

-- Given conditions
axiom A_on_circle : is_on_circle center radius A
axiom B_on_circle : is_on_circle center radius B
axiom A_opposite_B : A.1 = -B.1 ∧ A.2 = B.2  -- points A and B on opposite arcs of diameter CD
axiom CE_perp_AB : is_perpendicular C E (A, B)
axiom DF_perp_AB : is_perpendicular D F (A, B)
axiom collinear_AEFB : A.1 < E.1 ∧ E.1 < F.1 ∧ F.1 < B.1 -- A, E, F, B are collinear in this order
axiom AE_length : (E.1 - A.1) ^ 2 + (E.2 - A.2) ^ 2 = 1 -- AE = 1

-- Theorem to be proved
theorem length_BF_is_1 : (F.1 - B.1) ^ 2 + (F.2 - B.2) ^ 2 = 1 :=
sorry

end length_BF_is_1_l353_353847


namespace ellipse_tangent_normal_l353_353178

theorem ellipse_tangent_normal :
  let ellipse_eq := (x y : ℝ) → (x^2 / 18 + y^2 / 8 = 1)
  let pt := (3, 2 : ℝ)
  let tangent_eq := (x y : ℝ) → (2 * x + 3 * y - 12 = 0)
  let normal_eq := (x y : ℝ) → (3 * x - 2 * y - 5 = 0)
  (ellipse_eq 3 2) → tangent_eq 3 2 ∧ normal_eq 3 2 :=
sorry

end ellipse_tangent_normal_l353_353178


namespace largest_e_is_23_l353_353820

open Real

-- Define the problem conditions
structure CircleDiameter (P Q X Y Z : ℝ × ℝ) where
  PQ_is_diameter : dist P Q = 2
  X_is_midpoint : dist P Q / 2 = dist P X
  PY_length : dist P Y = 4 / 5
  X_on_semicircle : (X.1 - (P.1 + Q.1) / 2)^2 + (X.2 - (P.2 + Q.2) / 2)^2 = ((P.1 - Q.1)/2)^2 + ((P.2 - Q.2)/2)^2
  Y_on_semicircle : (Y.1 - (P.1 + Q.1) / 2)^2 + (Y.2 - (P.2 + Q.2) / 2)^2 = ((P.1 - Q.1)/2)^2 + ((P.2 - Q.2)/2)^2
  Z_lies_other_semicircle : (Z.1 - (P.1 + Q.1) / 2)^2 + (Z.2 - (P.2 + Q.2) / 2)^2 = ((P.1 - Q.1)/2)^2 + ((P.2 - Q.2)/2)^2

noncomputable def largest_possible_e (P Q X Y Z : ℝ × ℝ) [CircleDiameter P Q X Y Z] : ℝ :=
let V := P -- Intersection point placeholder V
let W := Q -- Intersection point placeholder W
let e := dist V W -- Length segment e
in e 

theorem largest_e_is_23 (P Q X Y Z : ℝ × ℝ) [CircleDiameter P Q X Y Z] :
  largest_possible_e P Q X Y Z = 23 :=
sorry

end largest_e_is_23_l353_353820


namespace find_m_n_l353_353046

theorem find_m_n (m n : ℕ) (h : 26019 * m - 649 * n = 118) : m = 2 ∧ n = 80 :=
by 
  sorry

end find_m_n_l353_353046


namespace log_eight_y_eq_275_l353_353241

def log_base_eight (x : ℝ) : ℝ := Real.log x / Real.log 8

theorem log_eight_y_eq_275 (y : ℝ) (h : log_base_eight y = 2.75) : y = 215 :=
by
  sorry

end log_eight_y_eq_275_l353_353241


namespace value_of_expression_l353_353240

theorem value_of_expression (x y : ℝ) (h1 : x = 1 / 2) (h2 : y = 2) : (1 / 3) * x ^ 8 * y ^ 9 = 2 / 3 :=
by
  -- Proof can be filled in here
  sorry

end value_of_expression_l353_353240


namespace line_slope_l353_353492

theorem line_slope (y_intercept : ℝ) (x1 y1 x2 y2 : ℝ) 
  (h1 : y_intercept = 10) 
  (h2 : (x1, y1) = (0, y_intercept)) 
  (h3 : (x2, y2) = (100, 1000)) : 
  ((y2 - y1) / (x2 - x1)) = 9.9 :=
by {
  rw [h2, h3, h1],
  simp,
  sorry
}

end line_slope_l353_353492


namespace proof_AB_eq_2DM_l353_353754

-- Definitions
variables {A B C D M : Type*}
variables {BC AB : ℝ}

-- Conditions
def is_midpoint (M : Type*) (B C : Type*) (BC : ℝ) := BM = BC / 2 ∧ CM = BC / 2
def is_altitude (AD : Type*) := ∃ A B C, angle(A,B,C) = 90 -- Note: Angle measure is assumed.

-- Main statement
theorem proof_AB_eq_2DM 
    (h_triangle : triangle A B C)
    (h_angle : angle B = 2 * angle C)
    (h_midpoint : is_midpoint M B C BC)
    (h_altitude : is_altitude AD) :
    AB = 2 * DM :=
sorry

end proof_AB_eq_2DM_l353_353754


namespace incident_and_reflected_rays_l353_353087

theorem incident_and_reflected_rays (P A : Point) (a b c k l m : ℝ)
  (hP : P = ⟨2, 3⟩)
  (hA : A = ⟨1, 1⟩)
  (h_line : line_eq a b c k l m)
  (h_reflect : ∃ Q, is_symmetric P Q a b c ∧ line_contains A Q):
  (line_eq 2 (-1) 1) ∧ (line_eq 4 (-5) 1) := 
by {
  -- P(2, 3), A(1, 1), line x + y + 1 = 0
  sorry
}

end incident_and_reflected_rays_l353_353087


namespace flowers_per_bouquet_l353_353987

theorem flowers_per_bouquet (narcissus chrysanthemums bouquets : ℕ) 
  (h1: narcissus = 75) 
  (h2: chrysanthemums = 90) 
  (h3: bouquets = 33) 
  : (narcissus + chrysanthemums) / bouquets = 5 := 
by 
  sorry

end flowers_per_bouquet_l353_353987


namespace unique_line_through_point_with_equal_intercepts_l353_353413

theorem unique_line_through_point_with_equal_intercepts :
  ∃! (L : ℝ → ℝ → Prop), (∀ (x y : ℝ), L x y ↔ x + y = 5) ∧ L 2 3 := 
begin
  sorry
end

end unique_line_through_point_with_equal_intercepts_l353_353413


namespace Dave_ticket_count_l353_353107

variable (T C total : ℕ)

theorem Dave_ticket_count
  (hT1 : T = 12)
  (hC1 : C = 7)
  (hT2 : T = C + 5) :
  total = T + C → total = 19 := by
  sorry

end Dave_ticket_count_l353_353107


namespace avg_marks_l353_353255

theorem avg_marks (P C M B E H G : ℝ) 
  (h1 : C = P + 75)
  (h2 : M = P + 105)
  (h3 : B = P - 15)
  (h4 : E = P - 25)
  (h5 : H = P - 25)
  (h6 : G = P - 25)
  (h7 : P + C + M + B + E + H + G = P + 520) :
  (M + B + H + G) / 4 = 82 :=
by 
  sorry

end avg_marks_l353_353255


namespace tetrahedron_volume_PQRS_l353_353876

noncomputable def tetrahedron_volume (PQ PR PS QR QS RS : ℝ) : ℝ :=
  let a := PQ in
  let b := PR in
  let c := PS in
  let d := QR in
  let e := QS in
  let f := RS in
  let M := λ i j, match (i, j) with
    | (0, 0) => 0   | (0, 1) => 1     | (0, 2) => 1     | (0, 3) => 1     | (0, 4) => 1
    | (1, 0) => 1   | (1, 1) => 0     | (1, 2) => a^2   | (1, 3) => b^2   | (1, 4) => c^2
    | (2, 0) => 1   | (2, 1) => a^2   | (2, 2) => 0     | (2, 3) => d^2   | (2, 4) => e^2
    | (3, 0) => 1   | (3, 1) => b^2   | (3, 2) => d^2   | (3, 3) => 0     | (3, 4) => f^2
    | (4, 0) => 1   | (4, 1) => c^2   | (4, 2) => e^2   | (4, 3) => f^2   | (4, 4) => 0
    | _ => 0
  in
  (Real.sqrt (Matrix.det (Matrix.of (Fin 5) (Fin 5) M))) / 288

theorem tetrahedron_volume_PQRS : tetrahedron_volume 6 4 3 5 7 (Real.sqrt 94) = 2 :=
by
  -- Omitted proof steps
  sorry

end tetrahedron_volume_PQRS_l353_353876


namespace doubled_volume_l353_353476

theorem doubled_volume (V : ℕ) (h : V = 4) : 8 * V = 32 := by
  sorry

end doubled_volume_l353_353476


namespace triangle_problem_l353_353275

noncomputable section -- Use noncomputable section to handle trigonometric calculations

open Real

variables {a b c : ℝ} {A B C : ℝ}

def roots_of_quadratic (x y : ℝ) : Prop :=
  x^2 - 2*sqrt 3 * x + 2 = 0 ∧ y^2 - 2*sqrt 3 * y + 2 = 0

axiom cos_A_plus_B_eq_half : cos (A + B) = 1 / 2

theorem triangle_problem 
  (h1 : roots_of_quadratic a b)
  (h2 : cos_A_plus_B_eq_half)
  (h3 : A + B + C = π) -- Sum of angles in a triangle
  (h4 : 0 < A) (h5 : A < π) (h6 : 0 < B) (h7 : B < π) (h8 : 0 < C) (h9 : C < π):
  C = π / 3 ∧ c = sqrt 6 ∧ (1 / 2 * a * b * sin C = sqrt 3 / 2) :=
sorry

end triangle_problem_l353_353275


namespace find_f_find_g_l353_353156

-- Problem 1: Finding f(x) given f(x+1) = x^2 - 2x
theorem find_f (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 2 * x) :
  ∀ x, f x = x^2 - 4 * x + 3 :=
sorry

-- Problem 2: Finding g(x) given roots and a point
theorem find_g (g : ℝ → ℝ) (h1 : g (-2) = 0) (h2 : g 3 = 0) (h3 : g 0 = -3) :
  ∀ x, g x = (1 / 2) * x^2 - (1 / 2) * x - 3 :=
sorry

end find_f_find_g_l353_353156


namespace manager_salary_l353_353966

theorem manager_salary (avg_salary_20 : ℝ) (increase : ℝ) (num_employees : ℕ) :
  avg_salary_20 = 1700 → increase = 100 → num_employees = 20 → 
  let total_salary_employees := num_employees * avg_salary_20 in
  let new_avg_salary := avg_salary_20 + increase in
  let new_num_people := num_employees + 1 in
  let total_salary_new := new_num_people * new_avg_salary in
  let manager_salary := total_salary_new - total_salary_employees in
  manager_salary = 3800 :=
by
  intros h_avg_salary_20 h_increase h_num_employees
  let total_salary_employees := num_employees * avg_salary_20
  let new_avg_salary := avg_salary_20 + increase
  let new_num_people := num_employees + 1
  let total_salary_new := new_num_people * new_avg_salary
  let manager_salary := total_salary_new - total_salary_employees
  sorry

end manager_salary_l353_353966


namespace hyperbola_eccentricity_l353_353210

theorem hyperbola_eccentricity (a b x y : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : (x:ℝ)^2 / a^2 - (y:ℝ)^2 / b^2 = 1)
  (h4 : ∃ (F : ℝ), y = b^2 / a ∧ F = 0)
  (h5 : ∀ P, ∃ (c : ℝ), x = c ∧ y = b^2 / a ∧ (c - b) / (c + b) = 1 / 3):
  (c : ℝ), ∃ e, e = c / a → e = 2 * sqrt(3) / 3 :=
sorry

end hyperbola_eccentricity_l353_353210


namespace sequence_accumulating_is_arithmetic_l353_353810

noncomputable def arithmetic_sequence {α : Type*} [LinearOrderedField α]
  (a : ℕ → α) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

theorem sequence_accumulating_is_arithmetic
  {α : Type*} [LinearOrderedField α] (a : ℕ → α) (S : ℕ → α)
  (na_gt_zero : ∀ n, a n > 0)
  (ha2 : a 2 = 3 * a 1)
  (hS_arith : arithmetic_sequence (λ n, (S n)^(1/2)))
  (hSn : ∀ n, S n = (∑ i in Finset.range (n+1), a i)) :
  arithmetic_sequence a := 
sorry

end sequence_accumulating_is_arithmetic_l353_353810


namespace problem_l353_353745

theorem problem (x y z : ℝ) 
  (h1 : (x - 4)^2 + (y - 3)^2 + (z - 2)^2 = 0)
  (h2 : 3 * x + 2 * y - z = 12) :
  x + y + z = 9 := 
  sorry

end problem_l353_353745


namespace y_eq_x_exp2n_l353_353837

-- Define sequences x and y

def x : ℕ → ℝ
| 0     := 1
| (n+1) := (x n + 2) / (x n + 1)

def y : ℕ → ℝ
| 0     := 1
| (n+1) := (y n * y n + 2) / (2 * y n)

-- Statement of the proof problem
theorem y_eq_x_exp2n : ∀ n : ℕ, y (n + 1) = x (2^n) := sorry

end y_eq_x_exp2n_l353_353837


namespace initial_eggs_in_the_box_l353_353906

theorem initial_eggs_in_the_box (h1 : ℕ := 5) (h2 : ℕ := 42) :  h1 + h2 = 47 := 
by
  simp
  done

end initial_eggs_in_the_box_l353_353906


namespace count_integers_between_sqrt10_sqrt100_l353_353717

theorem count_integers_between_sqrt10_sqrt100 : 
  ∃ (S : Set ℤ), (∀ n, n ∈ S ↔ (real.sqrt 10 < n ∧ n < real.sqrt 100)) ∧ S.card = 6 := 
by
  sorry

end count_integers_between_sqrt10_sqrt100_l353_353717


namespace force_of_water_pressure_on_plate_l353_353999

noncomputable def force_on_plate_under_water (γ : ℝ) (g : ℝ) (a b : ℝ) : ℝ :=
  γ * g * (b^2 - a^2) / 2

theorem force_of_water_pressure_on_plate :
  let γ : ℝ := 1000 -- kg/m^3
  let g : ℝ := 9.81  -- m/s^2
  let a : ℝ := 0.5   -- top depth
  let b : ℝ := 2.5   -- bottom depth
  force_on_plate_under_water γ g a b = 29430 := sorry

end force_of_water_pressure_on_plate_l353_353999


namespace largest_int_less_than_100_with_remainder_5_l353_353558

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l353_353558


namespace find_BD_when_AC_over_AB_min_l353_353326

-- Definition of the conditions
variables (A B C D : Type) [metric_space A]
variables (triangle : set A)
variables (on_side_BC : D ∈ segment B C)
variables (angle_ADB : ∠ADB = 120)
variables (AD_len : dist A D = 2)
variables (CD_2BD : dist C D = 2 * dist B D)

noncomputable def minimum_ratio_condition (BD : ℝ) : Prop := 
  let x := dist B D
  let CD := 2 * x
  let AD := 2 in
  let angle_ADB_cos := -1/2 in
  let b_squared := AD^2 + CD^2 - 2 * AD * CD * angle_ADB_cos in
  let c_squared := AD^2 + x^2 + 2 * AD * x * angle_ADB_cos in
  b_squared / c_squared = 4 - (12 / (x + 1 + 3/(x + 1))) → x = sqrt 3 - 1

theorem find_BD_when_AC_over_AB_min (BD : ℝ) : minimum_ratio_condition A B C D 
:= sorry

end find_BD_when_AC_over_AB_min_l353_353326


namespace max_of_expression_l353_353171

theorem max_of_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (abcd (a + b + c + d) / ((a + b)^2 * (c + d)^2)) ≤ 1 / 4 :=
sorry

end max_of_expression_l353_353171


namespace min_distance_l353_353289

def triangle_ABC (A B C D : Type) (BD : ℝ) :=
∃ (CD : ℝ), CD = 2 * BD ∧ 
  ∃ (AD AC AB : ℝ), AD = 2 ∧ 
    ∃ (α β γ : ℝ), α = 120 ∧ 
      angle_ADB = α ∧ 
        (AC / AB = (sqrt (3) - 1))

theorem min_distance (A B C D : Type) (BD : ℝ) :
  ∃ D ∈ BC, ∃ α β γ AD AC AB, 
    α = 120 ∧ AD = 2 ∧ D ∈ BC ∧ 
      γ = ∠ADB = α ∧ 
        CD = 2 * BD →
          (AC / AB) = (sqrt (3) - 1) := 
sorry

end min_distance_l353_353289


namespace largest_prime_in_partition_l353_353852


theorem largest_prime_in_partition (P : ℕ) (hP : nat.prime P) (k : ℕ → ℕ) :
  (∃ (k : ℕ → ℕ), (∀ i, nat.prime (k i)) ∧ (∑ i in finset.range 100, k i = 2015) ∧ (∃ j, ∀ i, k i ≤ k j) ∧ k i = P) →
  P = 23 :=
sorry

end largest_prime_in_partition_l353_353852


namespace old_edition_pages_l353_353076

theorem old_edition_pages (x : ℕ) 
  (h₁ : 2 * x - 230 = 450) : x = 340 := 
by sorry

end old_edition_pages_l353_353076


namespace midpoint_AB_product_PA_PB_l353_353261

-- Problem 1 Midpoint
theorem midpoint_AB (α θ t : Real) (hα : α = π / 3) (tx : t = 3) :
  ∃ (x y : Real), (x = 3 + t * (cos α) ∧ y = t * (sin α)) ∧
                  (x^2 - y^2 = 1) ∧
                  (x = 9 / 2 ∧ y = (3 * sqrt 3) / 2) :=
by sorry

-- Problem 2 Distance
theorem product_PA_PB (α : Real) (hα : tan α = 2) :
  (abs (8 / ((cos α)^2 - (sin α)^2)) = 40 / 3) :=
by sorry

end midpoint_AB_product_PA_PB_l353_353261


namespace complex_sum_q_u_l353_353414

theorem complex_sum_q_u (p q r s t u : ℂ) (hp : s = 5) (ht : t = -p - r) (hsum : p + q * complex.I + r + s * complex.I + t + u * complex.I = -6 * complex.I) :
  q + u = -11 :=
by
  sorry

end complex_sum_q_u_l353_353414


namespace tens_digit_of_even_not_divisible_by_10_l353_353829

theorem tens_digit_of_even_not_divisible_by_10 (N : ℕ) (h1 : N % 2 = 0) (h2 : N % 10 ≠ 0) :
  (N ^ 20) % 100 / 10 % 10 = 7 :=
sorry

end tens_digit_of_even_not_divisible_by_10_l353_353829


namespace number_of_boys_exceeds_girls_by_l353_353256

theorem number_of_boys_exceeds_girls_by (girls boys: ℕ) (h1: girls = 34) (h2: boys = 841) : boys - girls = 807 := by
  sorry

end number_of_boys_exceeds_girls_by_l353_353256


namespace tan_alpha_eq_neg_five_twelves_l353_353738

noncomputable def α : ℝ := sorry -- A placeholder for α in the fourth quadrant
def sinα : ℝ := -5 / 13
axiom α_in_fourth_quadrant : 3 * (Real.pi / 2) < α ∧ α < 2 * Real.pi -- Fourth quadrant condition

-- Main statement to prove
theorem tan_alpha_eq_neg_five_twelves : tan α = -5 / 12 :=
by
  have h_sin : sin α = sinα := by sorry
  sorry

end tan_alpha_eq_neg_five_twelves_l353_353738


namespace binom_two_eq_l353_353944

theorem binom_two_eq (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binom_two_eq_l353_353944


namespace find_BD_when_AC_over_AB_min_l353_353285

open Real

-- Define the conditions
variables (A B C D : Point)
variables (AD BC BD CD : ℝ)
variables (angle_ADB : ℝ)
variables (AC AB : ℝ)

-- Assign conditions
def conditions : Prop := 
  angle_ADB = 120 ∧
  AD = 2 ∧
  CD = 2 * BD ∧
  AC / AB = minimized

-- Target to prove
theorem find_BD_when_AC_over_AB_min (A B C D : Point) (BD : ℝ)
  (h : conditions A B C D 2 120 BD 2 * BD AC AB) :
  BD = sqrt 3 - 1 :=
sorry

end find_BD_when_AC_over_AB_min_l353_353285


namespace log_base_2_of_one_fourth_equals_neg_two_l353_353138

theorem log_base_2_of_one_fourth_equals_neg_two :
  (∀ x : ℝ, 2^x = 1 / 4 → x = -2) := 
begin
  intros x hx,
  sorry
end

end log_base_2_of_one_fourth_equals_neg_two_l353_353138


namespace proof_spherical_coordinates_l353_353084

noncomputable def spherical_to_rect (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  (ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

noncomputable def find_spherical_coordinates : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ)
| (x, y, z) :=
  let ρ := real.sqrt (x^2 + y^2 + z^2)
  let sinφ := real.sqrt (x^2 + y^2) / ρ
  let cosφ := z / ρ
  let φ := real.arctan (sinφ / cosφ)
  let sinθ := y / (ρ * sin φ)
  let cosθ := x / (ρ * sin φ)
  let θ := real.arctan (sinθ / cosθ)
  (ρ, θ, φ)

theorem proof_spherical_coordinates (x y z : ℝ) :
    (x, y, z) = spherical_to_rect 4 (5 * (real.pi / 6)) (real.pi / 4) →
    find_spherical_coordinates (x, y, -z) = (2 * real.sqrt 10, 5 * (real.pi / 6), 3 * (real.pi / 4)) :=
by {
  intros h,
  rw [spherical_to_rect, find_spherical_coordinates] at h,
  sorry
}

end proof_spherical_coordinates_l353_353084


namespace nesbitt_inequality_nesbitt_inequality_eq_l353_353830

variable {a b c : ℝ}

theorem nesbitt_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ (3 / 2) :=
sorry

theorem nesbitt_inequality_eq (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ((a / (b + c)) + (b / (a + c)) + (c / (a + b)) = (3 / 2)) ↔ (a = b ∧ b = c) :=
sorry

end nesbitt_inequality_nesbitt_inequality_eq_l353_353830


namespace largest_int_less_than_100_with_remainder_5_l353_353559

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l353_353559


namespace arithmetic_sequence_z_value_l353_353027

theorem arithmetic_sequence_z_value :
  ∃ z : ℤ, (3 ^ 2 = 9 ∧ 3 ^ 4 = 81) ∧ z = (9 + 81) / 2 :=
by
  -- the proof goes here
  sorry

end arithmetic_sequence_z_value_l353_353027


namespace largest_int_with_remainder_5_lt_100_l353_353572

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l353_353572


namespace largest_integer_less_than_100_with_remainder_5_l353_353580

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353580


namespace sum_of_numbers_l353_353878

theorem sum_of_numbers (a b : ℕ) (h_lcm : Nat.lcm a b = 54) (h_ratio : a / b = 2 / 3) : a + b = 45 :=
by
  sorry

end sum_of_numbers_l353_353878


namespace part1_expression_part2_max_profit_part3_profit_constraint_l353_353070

noncomputable def cost_price := 30
noncomputable def sales_quantity (x : ℕ) : ℕ := -x + 60
noncomputable def daily_profit (x : ℕ) : ℝ := (x - cost_price) * (sales_quantity x)

/-- Part 1: Prove the analytical expression of the function w in terms of x --/
theorem part1_expression (x : ℕ) (hx : 30 ≤ x ∧ x ≤ 60) :
  daily_profit x = -x^2 + 90 * x - 1800 :=
sorry

/-- Part 2: Prove that the selling price maximizing the profit is 45 --/
theorem part2_max_profit :
  ∃ x : ℕ, (30 ≤ x ∧ x ≤ 60) ∧ (∀ y : ℕ, 30 ≤ y ∧ y ≤ 60 → daily_profit y ≤ daily_profit 45) ∧ daily_profit 45 = 225 :=
sorry

/-- Part 3: Prove the selling price for a daily profit of 200 yuan with x ≤ 48 is 40 --/
theorem part3_profit_constraint (x : ℕ) (hx : x ≤ 48) :
  daily_profit x = 200 → x = 40 :=
sorry

end part1_expression_part2_max_profit_part3_profit_constraint_l353_353070


namespace minimal_n_for_polynomial_real_root_l353_353698

/-- 
Given a polynomial P(x) = a_{2n} x^{2n} + a_{2n-1} x^{2n-1} + ... + a_1 x + a_0
where each coefficient a_i belongs to the interval [100, 101].
Prove that the minimal n for which the polynomial P(x) can have a real root is 100.
-/
theorem minimal_n_for_polynomial_real_root (P : ℝ[X]) (n : ℕ) (hP : ∀ i, P.coeff i ∈ set.Icc (100 : ℝ) 101) :
  (∃ x : ℝ, P.eval x = 0) ↔ n = 100 :=
sorry

end minimal_n_for_polynomial_real_root_l353_353698


namespace least_common_multiple_5_6_10_12_l353_353946

open Nat

theorem least_common_multiple_5_6_10_12 :
  lcm (lcm 5 6) (lcm 10 12) = 60 :=
by
  sorry

end least_common_multiple_5_6_10_12_l353_353946


namespace doubled_volume_l353_353477

theorem doubled_volume (V : ℕ) (h : V = 4) : 8 * V = 32 := by
  sorry

end doubled_volume_l353_353477


namespace largest_among_four_numbers_l353_353099

theorem largest_among_four_numbers (h : 0 < real.log 2 ∧ real.log 2 < 1) :
  max (max (max (2 * real.log 2) (real.log 2)) ((real.log 2)^2)) (real.log (real.log 2)) = 2 * real.log 2 :=
by sorry

end largest_among_four_numbers_l353_353099


namespace rhombus_area_is_160_l353_353054

-- Define the values of the diagonals
def d1 : ℝ := 16
def d2 : ℝ := 20

-- Define the formula for the area of the rhombus
noncomputable def area_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

-- State the theorem to be proved
theorem rhombus_area_is_160 :
  area_rhombus d1 d2 = 160 :=
by
  sorry

end rhombus_area_is_160_l353_353054


namespace transform_sin_l353_353010

theorem transform_sin :
  ∀ (x : ℝ), (λ x => Real.sin x ∘ λ x => (3 * x - 3 * (π / 4))) = (λ x => Real.sin (3 * x + (π / 4))) :=
by
  sorry

end transform_sin_l353_353010


namespace problem_f_eval_l353_353212

-- Define the function as given in the problem statement
def f (x : ℝ) : ℝ :=
  if x - 2018 >= 0 then sqrt 2 * Real.sin (x - 2018)
  else Real.log (-(x - 2018))

-- State the theorem to prove
theorem problem_f_eval :
  f (2018 + Real.pi / 4) * f (-7982) = 4 :=
by
  -- This is where the proof should be.
  sorry

end problem_f_eval_l353_353212


namespace find_a_l353_353747

open BigOperators

theorem find_a (a : ℝ) 
  (h : (choose 5 2) * a^3 = 10) : 
  a = 1 := 
begin
  sorry
end

end find_a_l353_353747


namespace sum_b_div_3_pow_n_l353_353823

-- Define the sequence b_n
noncomputable def b : ℕ → ℕ
| 0 := 2  -- Note that Lean 4 is 0-indexed
| 1 := 3
| (n + 2) := b (n + 1) + b n

-- Prove that the infinite sum of b_n / 3^(n+1) is 2/5
theorem sum_b_div_3_pow_n :
  ∑' n, (b n) / (3 : ℝ)^(n + 1) = 2 / 5 := by
  sorry

end sum_b_div_3_pow_n_l353_353823


namespace squares_overlap_ratio_l353_353919

theorem squares_overlap_ratio (a b : ℝ) (h1 : 0.52 * a^2 = a^2 - (a^2 - 0.52 * a^2))
                             (h2 : 0.73 * b^2 = b^2 - (b^2 - 0.73 * b^2)) :
                             a / b = 3 / 4 := by
sorry

end squares_overlap_ratio_l353_353919


namespace garys_pool_length_l353_353650

/-- Gary's rectangular pool question translated to Lean 4 statement -/
theorem garys_pool_length :
  ∀ (length width depth quarts_per_cubic_feet chlorine_cost dollors_spent : ℝ),
  let volume := length * width * depth in
  let quarts_bought := dollors_spent / chlorine_cost in
  let total_volume_quarts := quarts_bought * quarts_per_cubic_feet in
  width = 8 ∧ depth = 6 ∧ quarts_per_cubic_feet = 120 ∧ chlorine_cost = 3 ∧ dollors_spent = 12 →
  volume = total_volume_quarts →
  length = 10 :=
by {
  intros length width depth quarts_per_cubic_feet chlorine_cost dollors_spent volume quarts_bought total_volume_quarts,
  intro h,
  cases h with hw h1,
  cases h1 with hd h2,
  cases h2 with hq h3,
  cases h3 with hc hs,
  sorry
}

end garys_pool_length_l353_353650


namespace circle_area_proof_l353_353437

noncomputable def circle_area : Real :=
  let h := (5, 8) -- center of the circle
  let r : Real := 2 * Real.sqrt 6 -- radius of the circle
  let vertices := [(3, 8), (8, 8), (8, 13), (3, 13)] -- vertices of the square
  π * r^2

theorem circle_area_proof : circle_area = 24 * π :=
by
  sorry

end circle_area_proof_l353_353437


namespace shift_graph_l353_353011

theorem shift_graph :
  ∀ x : ℝ, f(x) = 2 * sin (2 * x + π / 6) → (f(x) = y(x + π / 12)) :=
by
  intros x h
  sorry

end shift_graph_l353_353011


namespace campers_afternoon_l353_353976

theorem campers_afternoon (x : ℕ) 
  (h1 : 44 = x + 5) : 
  x = 39 := 
by
  sorry

end campers_afternoon_l353_353976


namespace first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8_l353_353505

noncomputable def first_three_digits_of_decimal_part (x : ℝ) : ℕ :=
  -- here we would have the actual definition
  sorry

theorem first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8 :
  first_three_digits_of_decimal_part ((10^1001 + 1)^((9:ℝ) / 8)) = 125 :=
sorry

end first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8_l353_353505


namespace arithmetic_sequence_l353_353775

theorem arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) (h : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 :=
sorry

end arithmetic_sequence_l353_353775


namespace find_BD_when_AC_over_AB_min_l353_353280

open Real

-- Define the conditions
variables (A B C D : Point)
variables (AD BC BD CD : ℝ)
variables (angle_ADB : ℝ)
variables (AC AB : ℝ)

-- Assign conditions
def conditions : Prop := 
  angle_ADB = 120 ∧
  AD = 2 ∧
  CD = 2 * BD ∧
  AC / AB = minimized

-- Target to prove
theorem find_BD_when_AC_over_AB_min (A B C D : Point) (BD : ℝ)
  (h : conditions A B C D 2 120 BD 2 * BD AC AB) :
  BD = sqrt 3 - 1 :=
sorry

end find_BD_when_AC_over_AB_min_l353_353280


namespace smallest_positive_period_max_min_values_l353_353682

def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi / 6) - 1

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
by
  sorry

theorem max_min_values :
  ∃ max_val min_val : ℝ,
  max_val = 2 ∧ min_val = -1 ∧
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x ≤ max_val ∧ f x ≥ min_val) :=
by
  sorry

end smallest_positive_period_max_min_values_l353_353682


namespace similar_triangles_perimeter_l353_353898

theorem similar_triangles_perimeter (P_small P_large : ℝ) 
  (h_ratio : P_small / P_large = 2 / 3) 
  (h_sum : P_small + P_large = 20) : 
  P_small = 8 := 
sorry

end similar_triangles_perimeter_l353_353898


namespace area_PQR_l353_353362

theorem area_PQR (A B C K L M P Q R: Point) 
  (hAK : isMedian A K B C) 
  (hBL : isMedian B L A C) 
  (hCM : isMedian C M A B)
  (hKP : KP = (1 / 2) * length (A K)) 
  (hLQ : LQ = (1 / 2) * length (B L)) 
  (hMR : MR = (1 / 2) * length (C M)) 
  (h_area_ABC : area_triangle A B C = 1):
  area_triangle P Q R = 25 / 16 :=
sorry

end area_PQR_l353_353362


namespace number_of_integers_between_sqrt10_sqrt100_l353_353734

theorem number_of_integers_between_sqrt10_sqrt100 : 
  (set.Ico ⌈Real.sqrt 10⌉₊ (⌊Real.sqrt 100⌋₊ + 1)).card = 7 := by
  sorry

end number_of_integers_between_sqrt10_sqrt100_l353_353734


namespace find_n_l353_353386

theorem find_n (S : ℕ → ℚ) (hS0 : ∀ n, S n = (∑ i in finset.range n, 1 / (i + 1) / (i + 2)))
  (hS1 : ∀ n, S n * S (n + 1) = 3 / 4) : ∃ n, n = 6 :=
by
  sorry

end find_n_l353_353386


namespace mod_50238_23_l353_353920

theorem mod_50238_23 :
  ∃ n : ℤ, (0 ≤ n ∧ n < 23 ∧ 50238 % 23 = n) := 
begin
  use 19,
  split,
  { norm_num },
  { split, 
    { norm_num },
    { norm_num } 
  }
end

end mod_50238_23_l353_353920


namespace sum_of_valid_two_digit_numbers_is_1631_l353_353956

def isTwoDigitNumber (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99

def digitProductDivisibleBy7 (n : ℕ) : Prop :=
  let a := n / 10
  let b := n % 10
  a * b % 7 = 0

def validTwoDigitNumbers : List ℕ :=
  List.filter (λ n => isTwoDigitNumber n ∧ digitProductDivisibleBy7 n) (List.range 100)

def sumValidTwoDigitNumbers : ℕ :=
  List.sum validTwoDigitNumbers

theorem sum_of_valid_two_digit_numbers_is_1631 : sumValidTwoDigitNumbers = 1631 :=
  sorry

end sum_of_valid_two_digit_numbers_is_1631_l353_353956


namespace find_smallest_mod_l353_353352

-- Definition of the conditions
variables (n p : ℕ)
hypothesis (h_odd_n : n % 2 = 1)
hypothesis (h_p_div_n : p ∣ n)
hypothesis (h_p_proper_div : p < n)

-- The function m expressed in binary form
noncomputable def smallest_m (n p : ℕ) : ℕ := 
  let k := n / p in
  1 + 2^(p) + 2^((k-1)*p)

-- The theorem to verify the conditions of 'm'
theorem find_smallest_mod (n p : ℕ)
  (h_odd_n : n % 2 = 1)
  (h_p_div_n : p ∣ n)
  (h_p_proper_div : p < n) :
  ∃ m : ℕ, (1 + 2^p + 2^(n-p)) * m ≡ 1 [MOD 2^n] ∧
           m = smallest_m n p :=
by {
  sorry
}

end find_smallest_mod_l353_353352


namespace slope_tangent_line_at_x1_l353_353245

def f (x c : ℝ) : ℝ := (x-2)*(x^2 + c)
def f_prime (x c : ℝ) := (x^2 + c) + (x-2) * 2 * x

theorem slope_tangent_line_at_x1 (c : ℝ) (h : f_prime 2 c = 0) : f_prime 1 c = -5 := by
  sorry

end slope_tangent_line_at_x1_l353_353245


namespace necessary_has_7_unique_letters_l353_353902

def word := "necessary".to_list
def unique_letters := ['n', 'e', 'c', 's', 'a', 'r', 'y']

theorem necessary_has_7_unique_letters : (unique_letters.to_finset.card = 7) :=
by simp [unique_letters]; exact rfl

end necessary_has_7_unique_letters_l353_353902


namespace largest_integer_less_than_100_with_remainder_5_l353_353603

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l353_353603


namespace largest_integer_less_than_100_with_remainder_5_l353_353604

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l353_353604


namespace area_of_trapezoid_l353_353274

variables (ABCD : Type) [trapezoid ABCD]   -- Define ABCD as a trapezoid
variables (ω : circle)  -- Define an inscribed circle ω
variables (L : point)  -- Define point of tangency L on CD

-- Given conditions
axiom ratio_CL_LD {CL LD : ℝ} (hCL : CL = 6) (hLD : LD = 24) : CL / LD = 1 / 4
axiom BC_eq_9 : length BC = 9
axiom CD_eq_30 : length CD = 30

-- Additional definitions required for the proof
axiom point_of_tangencies (P Q K : point)
axiom tangency_point_PK : tangency_point ω BC P
axiom tangency_point_QA : tangency_point ω AD Q
axiom tangency_point_KA : tangency_point ω AB K
axiom tangency_point_LD : tangency_point ω CD L

-- Prove that the area of the trapezoid is 972
theorem area_of_trapezoid : area ABCD = 972 :=
sorry

end area_of_trapezoid_l353_353274


namespace multiplication_problem_l353_353271

theorem multiplication_problem : 
  ∃ (X Y : ℕ), 
    X = 12 ∧ Y = 89 ∧ X * Y = 1068 :=
by
  -- Definitions based on conditions
  let X := 12
  let Y := 89
  -- Proof outline (skipped with sorry)
  have h1 : X = 12 := rfl
  have h2 : Y = 89 := rfl
  have h3 : X * Y = 12 * 89 := rfl
  have h4 : 12 * 89 = 1068 := by norm_num
  -- Use the conditions to build the proof
  use [X, Y]
  exact ⟨h1, h2, h4⟩

end multiplication_problem_l353_353271


namespace volume_of_rectangular_box_l353_353905

theorem volume_of_rectangular_box (x y z : ℝ) 
  (h1 : x * y = 15) 
  (h2 : y * z = 20) 
  (h3 : x * z = 12) : 
  x * y * z = 60 := 
sorry

end volume_of_rectangular_box_l353_353905


namespace binom_two_eq_n_choose_2_l353_353932

theorem binom_two_eq_n_choose_2 (n : ℕ) (h : n ≥ 2) :
  (nat.choose n 2) = (n * (n - 1)) / 2 := by
  sorry

end binom_two_eq_n_choose_2_l353_353932


namespace largest_integer_less_than_hundred_with_remainder_five_l353_353589

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l353_353589


namespace largest_integer_less_than_100_with_remainder_5_l353_353622

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353622


namespace solve_inequality_l353_353153

theorem solve_inequality (x : ℝ) : (x+2)/(x-4) ≥ 3 ↔ x ∈ set.Ioo 4 7 ∪ set.Ici 7 :=
by
  split
  sorry

end solve_inequality_l353_353153


namespace part1a_part1b_part2_part3_part4_l353_353740

def floor_ceil (x : Rational) : Int := Int.floor x + 1

theorem part1a : floor_ceil 4.7 = 5 :=
sorry

theorem part1b : floor_ceil (-5.3) = -5 :=
sorry

theorem part2 (a : Rational) (h : floor_ceil a = 2) : 1 < a ∧ a ≤ 2 :=
sorry

theorem part3 (m : Rational) (h : floor_ceil (-2 * m + 7) = -3) : 5 ≤ m ∧ m < 5.5 :=
sorry

theorem part4 (n : Rational) (h : floor_ceil (4.5 * n - 2.5) = 3 * n + 1) : 
  n = 2 ∨ n = 7 / 3 :=
sorry

end part1a_part1b_part2_part3_part4_l353_353740


namespace smallest_n_for_prime_condition_l353_353167

theorem smallest_n_for_prime_condition :
  ∃ n : ℕ, (∀ p : ℕ, nat.prime p → n % (p-1) = 0 → n % p = 0) ∧ 
  (∀ m : ℕ, m < n → (∃ p : ℕ, nat.prime p ∧ m % (p-1) = 0 ∧ m % p ≠ 0)) ∧ 
  n = 1806 :=
sorry

end smallest_n_for_prime_condition_l353_353167


namespace tony_remaining_money_l353_353016

theorem tony_remaining_money :
  let initial_amount := 20
  let ticket_cost := 8
  let hotdog_cost := 3
  initial_amount - ticket_cost - hotdog_cost = 9 :=
by
  let initial_amount := 20
  let ticket_cost := 8
  let hotdog_cost := 3
  show initial_amount - ticket_cost - hotdog_cost = 9
  sorry

end tony_remaining_money_l353_353016


namespace rational_polynomials_with_roots_l353_353151

noncomputable def polynomial_with_rational_roots (a b c : ℚ) : Prop :=
  ∃ (r1 r2 r3 : ℚ), (∀ (f : ℚ → ℚ), f = λ x, x^3 + a * x^2 + b * x + c) ∧
  (f(r1) = 0) ∧ (f(r2) = 0) ∧ (f(r3) = 0)

theorem rational_polynomials_with_roots :
  ∀ (a b c : ℚ), polynomial_with_rational_roots a b c ↔
    ( (a = 1 ∧ b = -2 ∧ c = 0) ∨ (a = 1 ∧ b = -1 ∧ c = -1) ) := by
  sorry

end rational_polynomials_with_roots_l353_353151


namespace player_two_always_wins_l353_353666

theorem player_two_always_wins (n : ℕ) (hn : n > 1) : ∃ strategy : (fin n → ℕ), ∀ (opponent_strategy : fin n → ℕ), (player_two_longest_arc strategy > player_one_longest_arc opponent_strategy) ∨ game_draw strategy opponent_strategy := 
sorry

end player_two_always_wins_l353_353666


namespace pizza_eaten_after_six_trips_l353_353042

theorem pizza_eaten_after_six_trips
  (initial_fraction: ℚ)
  (next_fraction : ℚ -> ℚ)
  (S: ℚ)
  (H0: initial_fraction = 1 / 4)
  (H1: ∀ (n: ℕ), next_fraction n = 1 / 2 ^ (n + 2))
  (H2: S = initial_fraction + (next_fraction 1) + (next_fraction 2) + (next_fraction 3) + (next_fraction 4) + (next_fraction 5)):
  S = 125 / 128 :=
by
  sorry

end pizza_eaten_after_six_trips_l353_353042


namespace binom_eq_fraction_l353_353921

open Nat

theorem binom_eq_fraction (n : ℕ) (h_pos : 0 < n) : choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_eq_fraction_l353_353921


namespace sequence_is_arithmetic_l353_353802

theorem sequence_is_arithmetic {a : ℕ → ℝ} (h_pos : ∀ n, a n > 0)
  (h_second_term : a 2 = 3 * a 1)
  (h_sqrt_seq_arith : ∃ d : ℝ, ∀ n, real.sqrt (∑ i in finset.range (n + 1), a i) = d * n + real.sqrt (a 0)): 
  ∃ d : ℝ, ∀ n, a n = a 0 + d * n := 
by
  sorry

end sequence_is_arithmetic_l353_353802


namespace smaller_triangle_perimeter_l353_353787

theorem smaller_triangle_perimeter (p : ℕ) (h : p * 3 = 120) : p = 40 :=
sorry

end smaller_triangle_perimeter_l353_353787


namespace sum_of_coefficients_l353_353884

theorem sum_of_coefficients (a b c : ℤ) (h : a - b + c = -1) : a + b + c = -1 := sorry

end sum_of_coefficients_l353_353884


namespace correct_barometric_pressure_l353_353252

noncomputable def true_barometric_pressure (p1 p2 v1 v2 T1 T2 observed_pressure_final observed_pressure_initial : ℝ) : ℝ :=
  let combined_gas_law : ℝ := (p1 * v1 * T2) / (v2 * T1)
  observed_pressure_final + combined_gas_law

theorem correct_barometric_pressure :
  true_barometric_pressure 58 56 143 155 288 303 692 704 = 748 :=
by
  sorry

end correct_barometric_pressure_l353_353252


namespace total_fish_count_l353_353393

def number_of_tables : ℕ := 32
def fish_per_table : ℕ := 2
def additional_fish_table : ℕ := 1
def total_fish : ℕ := (number_of_tables * fish_per_table) + additional_fish_table

theorem total_fish_count : total_fish = 65 := by
  sorry

end total_fish_count_l353_353393


namespace min_value_of_g_inequality_f_l353_353686

def f (x m : ℝ) : ℝ := abs (x - m)
def g (x m : ℝ) : ℝ := 2 * f x m - f (x + m) m

theorem min_value_of_g (m : ℝ) (hm : m > 0) (h : ∀ x, g x m ≥ -1) : m = 1 :=
sorry

theorem inequality_f {m a b : ℝ} (hm : m > 0) (ha : abs a < m) (hb : abs b < m) (h0 : a ≠ 0) :
  f (a * b) m > abs a * f (b / a) m :=
sorry

end min_value_of_g_inequality_f_l353_353686


namespace minimum_BD_value_l353_353303

theorem minimum_BD_value {A B C D : Type*} [Real A] [Real B] [Real C] [Real D] (angle_ADB: A) (AD: A) (CD: A) (BD: A):
  (D ∈ line_segment(B, C)) →
  angle_ADB = 120 → 
  AD = 2 → 
  CD = 2 * BD →
  ∃ BD: A, BD = sqrt(3) - 1 :=
by
  sorry

end minimum_BD_value_l353_353303


namespace part1_part2_part3_l353_353369

-- Part 1
theorem part1 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : (x + y) * (y + z) * (z + x) ≥ 8 * x * y * z :=
sorry

-- Part 2
theorem part2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x^2 + y^2 + z^2 ≥ x * y + y * z + z * x :=
sorry

-- Part 3
theorem part3 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x ^ x * y ^ y * z ^ z ≥ (x * y * z) ^ ((x + y + z) / 3) :=
sorry

#print axioms part1
#print axioms part2
#print axioms part3

end part1_part2_part3_l353_353369


namespace trig_inequalities_l353_353183

def a : ℝ := Real.sin (Real.sin (2009 * (Real.pi / 180)))
def b : ℝ := Real.sin (Real.cos (2009 * (Real.pi / 180)))
def c : ℝ := Real.cos (Real.sin (2009 * (Real.pi / 180)))
def d : ℝ := Real.cos (Real.cos (2009 * (Real.pi / 180)))

theorem trig_inequalities : b < a ∧ a < d ∧ d < c := 
by
  sorry

end trig_inequalities_l353_353183


namespace n_is_square_if_m_even_l353_353760

theorem n_is_square_if_m_even
  (n : ℕ)
  (h1 : n ≥ 3)
  (m : ℕ)
  (h2 : m = (1 / 2) * n * (n - 1))
  (h3 : ∀ i j : ℕ, i ≠ j → (a_i + a_j) % m ≠ (a_j + a_k) % m)
  (h4 : even m) :
  ∃ k : ℕ, n = k * k := sorry

end n_is_square_if_m_even_l353_353760


namespace dist_P_to_origin_l353_353263

/-- The coordinates of point P --/
def P : ℝ × ℝ := (3, 4)

/-- The coordinates of the origin O --/
def O : ℝ × ℝ := (0, 0)

/-- Definition of the distance function --/
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

/-- The distance from point P to the origin is 5 --/
theorem dist_P_to_origin : distance P O = 5 := 
  sorry

end dist_P_to_origin_l353_353263


namespace sixth_element_row_20_l353_353444

theorem sixth_element_row_20 : (Nat.choose 20 5) = 15504 := by
  sorry

end sixth_element_row_20_l353_353444


namespace angle_CDB_eq_30_l353_353269

-- Let A, B, and C be points in the plane forming an equilateral triangle ABC
variables (A B C D : Type) [EquilateralTriangle A B C]

-- Given AD = AB
def AD_eq_AB (h1 : (distance A D) = (distance A B))

-- Prove that angle CDB = 30 degrees
theorem angle_CDB_eq_30 (h1 : AD = AB) : angle C D B = 30° := 
  sorry

end angle_CDB_eq_30_l353_353269


namespace decimal_to_binary_25_l353_353543

theorem decimal_to_binary_25 : (25 : Nat) = 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by
  sorry

end decimal_to_binary_25_l353_353543


namespace bill_soaking_time_l353_353111

theorem bill_soaking_time 
  (G M : ℕ) 
  (h₁ : M = G + 7) 
  (h₂ : 3 * G + M = 19) : 
  G = 3 := 
by {
  sorry
}

end bill_soaking_time_l353_353111


namespace ellipse_eq_elliptic_line_exist_l353_353198

-- Definition of ellipse
def ellipse_eq (a b : ℝ) : ℝ × ℝ → Prop :=
λ p => (p.fst ^ 2 / a ^ 2) + (p.snd ^ 2 / b ^ 2) = 1

-- Given conditions
def focal_condition : Prop :=
  let c := 2 in
  let a := 2 * c in
  let b := 2 * Real.sqrt 3 in
  a^2 - b^2 = c^2 ∧ a = 4 ∧ b = 2 * Real.sqrt 3

-- Problem 1
theorem ellipse_eq_elliptic :
  focal_condition →
  ∃ (a b : ℝ), a = 4 ∧ b = 2 * Real.sqrt 3 ∧ ellipse_eq a b = ellipse_eq 4 (2 * Real.sqrt 3) :=
by
  -- Note that the exact proof steps shall be included here when doing the actual proof in Lean
  sorry

-- Condition for line passing through E
def line_through_point (k : ℝ) (x y : ℝ) : Prop :=
  ∃ (l : ℝ → ℝ), l = λ x => k * x - 4 ∧ ∃ (r t : ℝ × ℝ), ellipse_eq 4 (2 * Real.sqrt 3) r ∧ ellipse_eq 4 (2 * Real.sqrt 3) t ∧ r.fst * t.fst + r.snd * t.snd = 16 / 7

-- Problem 2
theorem line_exist :
  ∃ l : ℝ → ℝ, line_through_point 1 0 (-4) ∧ line_through_point (-1) 0 (-4) :=
by
  -- Note that the exact proof steps shall be included here when doing the actual proof in Lean
  -- These represent the equations 'x + y + 4 = 0' and 'x - y - 4 = 0'
  sorry

end ellipse_eq_elliptic_line_exist_l353_353198


namespace largest_int_less_than_100_with_remainder_5_l353_353564

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l353_353564


namespace binom_n_2_l353_353937

theorem binom_n_2 (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_n_2_l353_353937


namespace value_of_expression_l353_353033

theorem value_of_expression : 48^2 - 2 * 48 * 3 + 3^2 = 2025 :=
by
  sorry

end value_of_expression_l353_353033


namespace binom_n_2_l353_353933

theorem binom_n_2 (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_n_2_l353_353933


namespace motorist_total_distance_l353_353455

def total_distance (time: ℕ) (speed1 speed2: ℕ) : ℕ :=
  let half_time := time / 2 in
  let distance1 := speed1 * half_time in
  let distance2 := speed2 * half_time in
  distance1 + distance2

theorem motorist_total_distance :
  total_distance 6 60 48 = 324 := 
by 
  -- cells for human-read  :
  sorry

end motorist_total_distance_l353_353455


namespace expression_value_l353_353176

variables {a b c : ℝ}

theorem expression_value (h : a * b + b * c + c * a = 3) :
  (a * (b^2 + 3) / (a + b)) + (b * (c^2 + 3) / (b + c)) + (c * (a^2 + 3) / (c + a)) = 6 := 
  sorry

end expression_value_l353_353176


namespace modified_expression_range_l353_353643

open Int

theorem modified_expression_range (m : ℤ) :
  ∃ n_min n_max : ℤ, 1 < 4 * n_max + 7 ∧ 4 * n_min + 7 < 60 ∧ (n_max - n_min + 1 = 15) →
  ∃ k_min k_max : ℤ, 1 < m * k_max + 7 ∧ m * k_min + 7 < 60 ∧ (k_max - k_min + 1 ≥ 15) := 
sorry

end modified_expression_range_l353_353643


namespace largest_e_is_23_l353_353819

open Real

-- Define the problem conditions
structure CircleDiameter (P Q X Y Z : ℝ × ℝ) where
  PQ_is_diameter : dist P Q = 2
  X_is_midpoint : dist P Q / 2 = dist P X
  PY_length : dist P Y = 4 / 5
  X_on_semicircle : (X.1 - (P.1 + Q.1) / 2)^2 + (X.2 - (P.2 + Q.2) / 2)^2 = ((P.1 - Q.1)/2)^2 + ((P.2 - Q.2)/2)^2
  Y_on_semicircle : (Y.1 - (P.1 + Q.1) / 2)^2 + (Y.2 - (P.2 + Q.2) / 2)^2 = ((P.1 - Q.1)/2)^2 + ((P.2 - Q.2)/2)^2
  Z_lies_other_semicircle : (Z.1 - (P.1 + Q.1) / 2)^2 + (Z.2 - (P.2 + Q.2) / 2)^2 = ((P.1 - Q.1)/2)^2 + ((P.2 - Q.2)/2)^2

noncomputable def largest_possible_e (P Q X Y Z : ℝ × ℝ) [CircleDiameter P Q X Y Z] : ℝ :=
let V := P -- Intersection point placeholder V
let W := Q -- Intersection point placeholder W
let e := dist V W -- Length segment e
in e 

theorem largest_e_is_23 (P Q X Y Z : ℝ × ℝ) [CircleDiameter P Q X Y Z] :
  largest_possible_e P Q X Y Z = 23 :=
sorry

end largest_e_is_23_l353_353819


namespace locus_of_M_l353_353200

/-- Define the coordinates of points A and B, and given point M(x, y) with the 
    condition x ≠ ±1, ensure the equation of the locus of point M -/
theorem locus_of_M (x y : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) 
  (h3 : (y / (x + 1)) + (y / (x - 1)) = 2) : x^2 - x * y - 1 = 0 := 
sorry

end locus_of_M_l353_353200


namespace sandy_age_l353_353372

variables (S M J : ℕ)

def Q1 : Prop := S = M - 14  -- Sandy is younger than Molly by 14 years
def Q2 : Prop := J = S + 6  -- John is older than Sandy by 6 years
def Q3 : Prop := 7 * M = 9 * S  -- The ratio of Sandy's age to Molly's age is 7:9
def Q4 : Prop := 5 * J = 6 * S  -- The ratio of Sandy's age to John's age is 5:6

theorem sandy_age (h1 : Q1 S M) (h2 : Q2 S J) (h3 : Q3 S M) (h4 : Q4 S J) : S = 49 :=
by sorry

end sandy_age_l353_353372


namespace problem1_problem2_l353_353973

noncomputable def problem1_set_a : Set ℝ := {0, 1/3, -1/2}

theorem problem1 (a : ℝ) (P S : Set ℝ) (hP : P = {x | x^2 + x - 6 = 0}) (hS : S = {x | a*x + 1 = 0}) :
    (S ⊆ P) ↔ a ∈ problem1_set_a :=
by
  sorry

noncomputable def problem2_set_m : Set ℝ := {m | m ≤ 3}

theorem problem2 (m : ℝ) (A B : Set ℝ) (hA : A = {x | -2 ≤ x ∧ x ≤ 5}) (hB : B = {x | (m + 1) ≤ x ∧ x ≤ (2*m - 1)}) :
    (B ⊆ A) ↔ m ∈ problem2_set_m :=
by
  sorry

end problem1_problem2_l353_353973


namespace ratio_of_group_sizes_l353_353907

theorem ratio_of_group_sizes 
  (x : ℝ) 
  (avg1 avg2 avg_total : ℝ) 
  (h1 : avg1 = 15) 
  (h2 : avg2 = 21) 
  (h3 : avg_total = 20) 
  : x / (1 - x) = 1/5 := 
by 
  -- Definition of averages
  have h_avg : avg_total = (avg1 * x + avg2 * (1 - x)) := sorry,
  
  -- Substitution
  rw [h1, h2, h3] at h_avg,
  
  -- Simplify to find x and the ratio
  have h_solve_for_x : x = 1/6 := sorry,
  
  -- Calculate the ratio
  rw [h_solve_for_x],
  norm_num,
  linarith,
  sorry

end ratio_of_group_sizes_l353_353907


namespace increasing_fn_l353_353356

def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x^2 + 2 * m * x - 2 else 1 + Real.log x 

theorem increasing_fn (m : ℝ) (h1 : 1 ≤ m) (h2 : m ≤ 2) :
  ∀ x y : ℝ, x < y → f m x ≤ f m y := by
  sorry

end increasing_fn_l353_353356


namespace binom_two_eq_n_choose_2_l353_353928

theorem binom_two_eq_n_choose_2 (n : ℕ) (h : n ≥ 2) :
  (nat.choose n 2) = (n * (n - 1)) / 2 := by
  sorry

end binom_two_eq_n_choose_2_l353_353928


namespace ways_to_choose_squares_l353_353196
-- Import the entire Mathlib

-- Define the required theorem based on the identified question and conditions
theorem ways_to_choose_squares :
  let even_rows := {2, 4, 6, 8},
      odd_rows := {1, 3, 5, 7},
      even_cols := {2, 4, 6, 8},
      odd_cols := {1, 3, 5, 7},
      choose_p := Nat.choose 4 2,
      factorial_4 := Nat.factorial 4
  in choose_p * choose_p * factorial_4 * factorial_4 = 20736 := by
  sorry

end ways_to_choose_squares_l353_353196


namespace gear_teeth_counts_l353_353020

theorem gear_teeth_counts (k : ℕ) :
  ((9 * k + 3 = 12) ∧ (7 * k - 3 = 4)) :=
by
  -- Initial gear tooth counts
  have h1 : 9 * 1 = 9 := by norm_num
  have h2 : 7 * 1 = 7 := by norm_num

  -- New gear tooth counts after replacement
  have h3 : 9 * 1 + 3 = 12 := by norm_num
  have h4 : 7 * 1 - 3 = 4 := by norm_num

  -- Prove the final statement using the conditions
  exact ⟨h3, h4⟩

end gear_teeth_counts_l353_353020


namespace remainder_is_15x_minus_14_l353_353166

noncomputable def remainder_polynomial_division : Polynomial ℝ :=
  (Polynomial.X ^ 4) % (Polynomial.X ^ 2 - 3 * Polynomial.X + 2)

theorem remainder_is_15x_minus_14 :
  remainder_polynomial_division = 15 * Polynomial.X - 14 :=
by
  sorry

end remainder_is_15x_minus_14_l353_353166


namespace hyperbola_eccentricity_l353_353669

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (F1 F2 P : EuclideanSpace ℝ (Fin 2))
  (hP_on_hyperbola : (P.1 ^ 2 / a^2 - P.2 ^ 2 / b^2 = 1))
  (h_angle_30 : ∠ P F1 F2 = 30)
  (h_angle_120 : ∠ P F2 F1 = 120) :
  let e : ℝ := Eccentricity (Hyperbola a b)
  in e = (sqrt 3 + 1) / 2 :=
  sorry

end hyperbola_eccentricity_l353_353669


namespace range_of_k_l353_353244

theorem range_of_k (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*y^2 = 2 ∧ 
  (∀ e : ℝ, (x^2 / 2 + y^2 / (2 / e) = 1 → (2 / e) > 2))) → 
  0 < k ∧ k < 1 :=
by 
sorry

end range_of_k_l353_353244


namespace gbr_share_and_change_in_share_l353_353522

noncomputable def total_nwf_rubles : ℝ := 794.26

noncomputable def usd_rubles : ℝ := 39.84
noncomputable def eur_rubles : ℝ := 34.72
noncomputable def cny_rubles : ℝ := 110.54
noncomputable def jpy_rubles : ℝ := 600.3
noncomputable def other_rubles : ℝ := 0.31

noncomputable def gbr_rubles_as_of_2021_04_01 : ℝ := total_nwf_rubles - usd_rubles - eur_rubles - cny_rubles - jpy_rubles - other_rubles

noncomputable def alpha_04_gbr : ℝ := gbr_rubles_as_of_2021_04_01 / total_nwf_rubles * 100

noncomputable def alpha_02_gbr : ℝ := 8.2

noncomputable def delta_alpha_gbr : ℝ := alpha_04_gbr - alpha_02_gbr

theorem gbr_share_and_change_in_share :
  alpha_04_gbr ≈ 1.08 ∧ delta_alpha_gbr ≈ -7 :=
sorry

end gbr_share_and_change_in_share_l353_353522


namespace savings_percentage_decrease_l353_353124

-- Definitions based on given conditions
def percentage_decrease (original_amount decreased_amount : ℕ) : ℝ :=
  ((original_amount - decreased_amount : ℝ) / original_amount) * 100

theorem savings_percentage_decrease :
  percentage_decrease 100000 90000 = 10 := 
sorry

end savings_percentage_decrease_l353_353124


namespace sin_alpha_geom_mean_l353_353365

-- Define the right triangle with properties mentioned
variable (α : Real) (b c : Real)
variable (h_angle : ∠ABC = 90)
variable (h_geom_mean : a = Real.sqrt (b * c))
variable (h_sin_alpha : sin α = (Real.sqrt 5 - 1) / 2)

theorem sin_alpha_geom_mean : sin α = (Real.sqrt 5 - 1) / 2 :=
by
  -- problem statement and proof obligation
  sorry

end sin_alpha_geom_mean_l353_353365


namespace slope_ratio_is_functional_domain_and_increasing_interval_l353_353403

-- Define the conditions for the problem
def line_l1_through_M (x y k : ℝ) : Prop := y = k * (x + 1)
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def midpoint (x1 x2 y1 y2 : ℝ) : Prop := (x1 + x2) / 2 = x ∧ (y1 + y2) / 2 = y
def slope (x1 y1 x2 y2 : ℝ) : ℝ := if x1 = x2 then 0 else (y2 - y1) / (x2 - x1)
def slope_ratio (k : ℝ) : ℝ := 1 / (1 - k^2)

-- Theorem stating the relation of slopes as a function of k
theorem slope_ratio_is_functional (k : ℝ) 
  (h_domain : k ∈ Ioo (-1 : ℝ) 0 ∨ k ∈ Ioo 0 1) :
  ∀ x y : ℝ, ∀ x1 x2 y1 y2 : ℝ,
  line_l1_through_M x y k → 
  parabola x y → 
  midpoint x1 x2 y1 y2 → 
  slope_ratio k = slope x (-1) y x (slope x1 y1 x2 y2) :=
sorry

-- Theorem stating the domain and increasing interval of f(k)
theorem domain_and_increasing_interval (k : ℝ) :
  (k ∈ (Ioo (-1 : ℝ) 0 ∨ Ioo 0 1) ∧ 
   Ioo (0 : ℝ) 1 ⊆ { k | derivative (slope_ratio k) > 0 }) :=
sorry

end slope_ratio_is_functional_domain_and_increasing_interval_l353_353403


namespace parabola_and_hyperbola_tangent_l353_353133

theorem parabola_and_hyperbola_tangent (m : ℝ) :
  (∀ (x y : ℝ), (y = x^2 + 6) → (y^2 - m * x^2 = 6) → (m = 12 + 10 * Real.sqrt 6 ∨ m = 12 - 10 * Real.sqrt 6)) :=
sorry

end parabola_and_hyperbola_tangent_l353_353133


namespace min_distance_l353_353294

def triangle_ABC (A B C D : Type) (BD : ℝ) :=
∃ (CD : ℝ), CD = 2 * BD ∧ 
  ∃ (AD AC AB : ℝ), AD = 2 ∧ 
    ∃ (α β γ : ℝ), α = 120 ∧ 
      angle_ADB = α ∧ 
        (AC / AB = (sqrt (3) - 1))

theorem min_distance (A B C D : Type) (BD : ℝ) :
  ∃ D ∈ BC, ∃ α β γ AD AC AB, 
    α = 120 ∧ AD = 2 ∧ D ∈ BC ∧ 
      γ = ∠ADB = α ∧ 
        CD = 2 * BD →
          (AC / AB) = (sqrt (3) - 1) := 
sorry

end min_distance_l353_353294


namespace find_a_2007_l353_353101

noncomputable def a : ℕ → ℝ
| 0     := 1
| 1     := 2
| (n+2) := 6 * a n - a (n+1)

theorem find_a_2007 : a 2007 = 2^2007 := 
sorry

end find_a_2007_l353_353101


namespace imaginary_part_of_inverse_z_plus_a_l353_353748

noncomputable def z (a : ℝ) : ℂ := a^2 - 1 + (a + 1) * complex.I

theorem imaginary_part_of_inverse_z_plus_a (a : ℝ) (ha : a^2 - 1 = 0) (ha_non_zero : a + 1 ≠ 0) :
  (complex.im (1 / (z a + a)) = -2 / 5) :=
by sorry

end imaginary_part_of_inverse_z_plus_a_l353_353748


namespace num_integers_between_sqrt10_sqrt100_l353_353709

theorem num_integers_between_sqrt10_sqrt100 : 
  ∃ n : ℕ, n = 7 ∧ ∀ x : ℤ, (⌊Real.sqrt 10⌋ + 1 <= x) ∧ (x <= ⌈Real.sqrt 100⌉ - 1) ↔ (4 <= x ∧ x <= 10) := 
by 
  sorry

end num_integers_between_sqrt10_sqrt100_l353_353709


namespace decimal_to_binary_correct_l353_353544

-- Define the decimal number
def decimal_number : ℕ := 25

-- Define the binary equivalent of 25
def binary_representation : ℕ := 0b11001

-- The condition indicating how the conversion is done
def is_binary_representation (decimal : ℕ) (binary : ℕ) : Prop :=
  -- Check if the binary representation matches the manual decomposition
  decimal = (binary / 2^4) * 2^4 + 
            ((binary % 2^4) / 2^3) * 2^3 + 
            (((binary % 2^4) % 2^3) / 2^2) * 2^2 + 
            ((((binary % 2^4) % 2^3) % 2^2) / 2^1) * 2^1 + 
            (((((binary % 2^4) % 2^3) % 2^2) % 2^1) / 2^0) * 2^0

-- Proof statement
theorem decimal_to_binary_correct : is_binary_representation decimal_number binary_representation :=
  by sorry

end decimal_to_binary_correct_l353_353544


namespace faucet_fill_time_l353_353648

theorem faucet_fill_time (r : ℝ) (T1 T2 t : ℝ) (F1 F2 : ℕ) (h1 : T1 = 200) (h2 : t = 8) (h3 : F1 = 4) (h4 : F2 = 8) (h5 : T2 = 50) (h6 : r * F1 * t = T1) : 
(F2 * r) * t / (F1 * F2) = T2 -> by sorry := sorry

#check faucet_fill_time

end faucet_fill_time_l353_353648


namespace number_of_paths_from_A_to_C_l353_353470

-- Conditions as definitions:
def points : Type := {A, B, C} -- representation of points A, B, and C
def segments : Type := {aAB, aBC, aAC} -- segments in the hexagonal lattice
def travel_direction (s : segments) : Prop := 
  match s with
  | aAB => true
  | aBC => true
  | aAC => false   -- true if the segment can be traveled in the direction of point C
  end

-- Travel condition functions
def distinct_travel (s : List segments) : Prop := s.nodup -- bug never travels the same segment more than once
def reachable_from_B_to_C (paths_to_C_from_B : Nat) : Prop := paths_to_C_from_B = 2

-- Theorem statement
theorem number_of_paths_from_A_to_C (paths_to_C_from_B : Nat) 
  (h1 : ∀ s, travel_direction s = true)
  (h2 : distinct_travel [aAB, aBC])
  (h3 : reachable_from_B_to_C paths_to_C_from_B) : 
  ∑ (paths : List Nat), (∑ steps_in_path,  path = 4200 := 
begin
  -- Proof goals go here
  sorry
end

end number_of_paths_from_A_to_C_l353_353470


namespace weight_of_b_l353_353389

theorem weight_of_b (A B C : ℝ)
  (h1 : A + B + C = 135)
  (h2 : A + B = 80)
  (h3 : B + C = 94) : 
  B = 39 := 
by 
  sorry

end weight_of_b_l353_353389


namespace triangle_first_side_l353_353639

theorem triangle_first_side (x : ℕ) (h1 : 10 + 15 + x = 32) : x = 7 :=
by
  sorry

end triangle_first_side_l353_353639


namespace min_ratio_bd_l353_353319

noncomputable def triangle_ratio_min (BD : ℝ) : Prop :=
  ∃ K L : ℝ, 
    (K = 1) ∧ 
    (L = 1) ∧ 
    (∀ (x : ℝ), ∀ (y : ℝ), (CD = 2 * BD) → ((x = sqrt(3) - 1) → y = (bd + 1 + 3 / (bd + 1)) ∧ (x + y) = 2 * K * L) ∧ 
    (abs x ≤ (sqrt 3 - 1)))


theorem min_ratio_bd (BD : ℝ) : triangle_ratio_min BD :=
by { sorry }

end min_ratio_bd_l353_353319


namespace find_water_in_sport_formulation_l353_353457

noncomputable def standard_formulation : ℚ × ℚ × ℚ := (1, 12, 30)
noncomputable def sport_flavoring_to_corn : ℚ := 3 * (1 / 12)
noncomputable def sport_flavoring_to_water : ℚ := (1 / 2) * (1 / 30)
noncomputable def sport_formulation (f : ℚ) (c : ℚ) (w : ℚ) : Prop :=
  f / c = sport_flavoring_to_corn ∧ f / w = sport_flavoring_to_water

noncomputable def given_corn_syrup : ℚ := 8

theorem find_water_in_sport_formulation :
  ∀ (f c w : ℚ), sport_formulation f c w → c = given_corn_syrup → w = 120 :=
by
  sorry

end find_water_in_sport_formulation_l353_353457


namespace max_surface_area_of_cylinder_inscribed_in_sphere_l353_353527

noncomputable def max_surface_area_cylinder (R : ℝ) : ℝ :=
  R^2 * π * (1 + Real.sqrt 5)

theorem max_surface_area_of_cylinder_inscribed_in_sphere (R : ℝ) (h r : ℝ) :
  R^2 = r^2 + (h / 2)^2 →
  2 * r * (Real.sqrt(R^2 - r^2)) + 2 * r * (r) = 2 * h * r + 4 * r * (Real.sqrt(R^2 - r^2)) →
  max_surface_area_cylinder R = R^2 * π * (1 + Real.sqrt 5) :=
  sorry

end max_surface_area_of_cylinder_inscribed_in_sphere_l353_353527


namespace largest_integer_less_than_100_with_remainder_5_l353_353627

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353627


namespace part_a_part_b_l353_353788

-- Define the side lengths as powers of 2
def powers_of_2 (n : ℕ) : ℕ := 2^n

-- Define the condition for part (a)
def can_cover_plane_10_times (cover : ℕ → ℕ) : Prop :=
  ∀ n, cover (powers_of_2 n) ≤ 10

-- Define the condition for part (b)
def can_cover_plane_1_time (cover : ℕ → ℕ) : Prop :=
  ∀ n, cover (powers_of_2 n) = 1

-- Representation of the plane coverage given certain constraints
def plane_covered (cover : ℕ → ℕ) (bound : ∀ n, cover (powers_of_2 n) ≤ n) : Prop :=
  sorry   -- Plane coverage representation is complex and left as a placeholder

-- The proof problem statements
theorem part_a : ∃ cover, can_cover_plane_10_times cover → plane_covered cover (can_cover_plane_10_times cover) :=
sorry

theorem part_b : ∃ cover, can_cover_plane_1_time cover → ¬plane_covered cover (can_cover_plane_1_time cover) :=
sorry

end part_a_part_b_l353_353788


namespace find_quarters_l353_353874

-- Define the conditions
def quarters_bounds (q : ℕ) : Prop :=
  8 < q ∧ q < 80

def stacks_mod4 (q : ℕ) : Prop :=
  q % 4 = 2

def stacks_mod6 (q : ℕ) : Prop :=
  q % 6 = 2

def stacks_mod8 (q : ℕ) : Prop :=
  q % 8 = 2

-- The theorem to prove
theorem find_quarters (q : ℕ) (h_bounds : quarters_bounds q) (h4 : stacks_mod4 q) (h6 : stacks_mod6 q) (h8 : stacks_mod8 q) : 
  q = 26 :=
by
  sorry

end find_quarters_l353_353874


namespace otimes_2_3_eq_23_l353_353519

-- Define the new operation
def otimes (a b : ℝ) : ℝ := 4 * a + 5 * b

-- The proof statement
theorem otimes_2_3_eq_23 : otimes 2 3 = 23 := 
  by 
  sorry

end otimes_2_3_eq_23_l353_353519


namespace infinite_even_and_odd_l353_353102
  
def sequence (n : ℕ) : ℕ :=
  nat.rec_on n 2 (λ k a_k, nat.floor (3 * a_k / 2))

def is_even (n : ℕ) : Prop :=
  ∃ k, n = 2 * k
  
def is_odd (n : ℕ) : Prop :=
  ∃ k, n = 2 * k + 1
  
theorem infinite_even_and_odd :
  (∀ n : ℕ, is_even (sequence n) ∨ is_odd (sequence n)) →
  (∃ n1 : ℕ, is_even (sequence n1)) ∧ (∃ n2 : ℕ, is_odd (sequence n2)) → False :=
begin
  intro h1,
  intro h2,
  sorry -- Proof goes here
end

end infinite_even_and_odd_l353_353102


namespace option_A_two_solutions_l353_353249

theorem option_A_two_solutions :
    (∀ (a b : ℝ) (A : ℝ), 
    (a = 3 ∧ b = 4 ∧ A = 45) ∨ 
    (a = 7 ∧ b = 14 ∧ A = 30) ∨ 
    (a = 2 ∧ b = 7 ∧ A = 60) ∨ 
    (a = 8 ∧ b = 5 ∧ A = 135) →
    (∃ a b A : ℝ, a = 3 ∧ b = 4 ∧ A = 45 ∧ 2 = 2)) :=
by
  sorry

end option_A_two_solutions_l353_353249


namespace solve_equation_l353_353379

theorem solve_equation : ∀ x : ℝ, (x - (x + 2) / 2 = (2 * x - 1) / 3 - 1) → (x = 2) :=
by
  intros x h
  sorry

end solve_equation_l353_353379


namespace math_proof_problem_l353_353415

noncomputable def problem := ∃ (a b c d : ℤ), 
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (0 < d) ∧ 
  (∃ x y : ℝ, (x + y = 6) ∧ (4 * x * y = 10) ∧ (x = (a + b * real.sqrt c) / d ∨ x = (a - b * real.sqrt c) / d)) ∧ 
  (a + b + c + d = 31)

theorem math_proof_problem : problem := by 
  sorry

end math_proof_problem_l353_353415


namespace total_fish_l353_353390

theorem total_fish (n : ℕ) (t : ℕ) (f : ℕ) :
  n = 32 ∧ t = 1 ∧ f = 31 ∧ ∃ (fish_count_table : ℕ → ℕ), 
  (fish_count_table(t) = 3) ∧ (∀ i, 1 ≤ i ∧ i <= f → fish_count_table(i + t) = 2) → 
  (∑ i in finset.range (t + f), fish_count_table (i + 1)) = 65 :=
by
  sorry

end total_fish_l353_353390


namespace theta_not_in_first_second_fourth_quadrant_l353_353749

theorem theta_not_in_first_second_fourth_quadrant
  (θ : ℝ)
  (h : 1 + sin θ * sqrt (sin θ ^ 2) + cos θ * sqrt (cos θ ^ 2) = 0) :
  ¬ ((0 < θ ∧ θ < π / 2) ∨ (π / 2 < θ ∧ θ < π) ∨ (3 * π / 2 < θ ∧ θ < 2 * π)) :=
sorry

end theta_not_in_first_second_fourth_quadrant_l353_353749


namespace complement_intersection_l353_353227

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 2, 3}
noncomputable def B : Set ℕ := {3, 4, 5}

theorem complement_intersection (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {3, 4, 5}) :
  U \ (A ∩ B) = {1, 2, 4, 5} :=
by
  sorry

end complement_intersection_l353_353227


namespace xyz_value_l353_353202

variable (x y z : ℝ)

theorem xyz_value :
  (x + y + z) * (x*y + x*z + y*z) = 36 →
  x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 24 →
  x * y * z = 4 :=
by
  intros h1 h2
  sorry

end xyz_value_l353_353202


namespace inequality_solution_set_range_of_a_l353_353219

noncomputable def f (x : ℝ) : ℝ := | x + 1 |

theorem inequality_solution_set :
  {x : ℝ | x * f x > f (x - 2)} = {x : ℝ | sqrt 2 - 1 < x} :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, y = log (f (x - 3) + f x + a)) ↔ a ≤ -3 :=
sorry

end inequality_solution_set_range_of_a_l353_353219


namespace sequence_accumulating_is_arithmetic_l353_353809

noncomputable def arithmetic_sequence {α : Type*} [LinearOrderedField α]
  (a : ℕ → α) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

theorem sequence_accumulating_is_arithmetic
  {α : Type*} [LinearOrderedField α] (a : ℕ → α) (S : ℕ → α)
  (na_gt_zero : ∀ n, a n > 0)
  (ha2 : a 2 = 3 * a 1)
  (hS_arith : arithmetic_sequence (λ n, (S n)^(1/2)))
  (hSn : ∀ n, S n = (∑ i in Finset.range (n+1), a i)) :
  arithmetic_sequence a := 
sorry

end sequence_accumulating_is_arithmetic_l353_353809


namespace coupon_discounts_diff_l353_353490

theorem coupon_discounts_diff {P : ℝ} (hP : P > 100) :
  let discountA := 0.20 * P,
      discountB := 40,
      discountC := 0.30 * (P - 100),
      x := 200,
      y := 300 in
  discountA >= discountB → discountA >= discountC → (y - x) = 100 :=
by
  sorry

end coupon_discounts_diff_l353_353490


namespace largest_e_possible_statement_l353_353817

noncomputable def largest_e_possible 
  (P Q : ℝ) (circ : ℝ) (X Y Z : ℝ → ℝ) 
  (diam : ℝ) 
  (midpoint : ℝ) 
  (py_eq : ℝ) 
  (intersects_S : ℝ) 
  (intersects_T : ℝ) 
  : ℝ :=
  sorry

-- Statement of the problem in Lean
theorem largest_e_possible_statement 
  (P Q : ℝ) (circ : ℝ) (X Y Z : ℝ → ℝ) 
  (diam : ℝ) 
  (midpoint : ℝ) 
  (py_eq : ℝ) 
  (intersects_S : ℝ) 
  (intersects_T : ℝ) 
  (circ_diam : circ = 2)
  (X_mid : X midpoint = 1)
  (PY_val : PY_eq = 4/5)
  : largest_e_possible P Q circ X Y Z diam midpoint py_eq intersects_S intersects_T = 25 :=
  sorry

end largest_e_possible_statement_l353_353817


namespace largest_integer_less_than_100_with_remainder_5_l353_353577

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353577


namespace village_population_equal_in_15_years_l353_353967

theorem village_population_equal_in_15_years :
  ∀ n : ℕ, (72000 - 1200 * n = 42000 + 800 * n) → n = 15 :=
by
  intros n h
  sorry

end village_population_equal_in_15_years_l353_353967


namespace total_papers_drawn_l353_353423

-- Conditions
def n_A : ℕ := 1260
def n_B : ℕ := 720
def n_C : ℕ := 900
def d_C : ℕ := 50

-- Prove the total number of papers drawn
theorem total_papers_drawn :
  let sample_ratio := d_C.to_rat / n_C.to_rat in
  let total_papers := (n_A.to_rat + n_B.to_rat + n_C.to_rat) * sample_ratio in
  total_papers = 160 :=
by
  sorry

end total_papers_drawn_l353_353423


namespace largest_integer_less_than_100_with_remainder_5_l353_353625

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353625


namespace max_value_of_expression_l353_353655

theorem max_value_of_expression (x y : ℝ) (h : 3 * x^2 + y^2 ≤ 3) : 2 * x + 3 * y ≤ Real.sqrt 31 :=
sorry

end max_value_of_expression_l353_353655


namespace sum_of_squares_of_roots_l353_353512

theorem sum_of_squares_of_roots :
  (∀ y : ℝ, y^3 - 8 * y^2 + 9 * y - 3 = 0 → 0 ≤ y)
  → ∑ r in (multiset.of_roots (y^3 - 8 * y^2 + 9 * y - 3)), r^2 = 46 := 
sorry

end sum_of_squares_of_roots_l353_353512


namespace find_BD_when_AC_over_AB_min_l353_353277

open Real

-- Define the conditions
variables (A B C D : Point)
variables (AD BC BD CD : ℝ)
variables (angle_ADB : ℝ)
variables (AC AB : ℝ)

-- Assign conditions
def conditions : Prop := 
  angle_ADB = 120 ∧
  AD = 2 ∧
  CD = 2 * BD ∧
  AC / AB = minimized

-- Target to prove
theorem find_BD_when_AC_over_AB_min (A B C D : Point) (BD : ℝ)
  (h : conditions A B C D 2 120 BD 2 * BD AC AB) :
  BD = sqrt 3 - 1 :=
sorry

end find_BD_when_AC_over_AB_min_l353_353277


namespace elevator_floor_computation_l353_353100

theorem elevator_floor_computation :
  let x := 15
  let y := 9
  let z := 12
  let w := 6
  let v := 10
  x - y + z - w + v = 28 := 
by
  rfl   -- immediate reflection of the arithmetic equation

end elevator_floor_computation_l353_353100


namespace sqrt_a_same_type_l353_353237

theorem sqrt_a_same_type (a : ℝ) (h : (∃ k : ℝ, sqrt a = k * sqrt 3)) : a = 12 :=
by {
    admit -- sorry
}

end sqrt_a_same_type_l353_353237


namespace four_dice_probability_l353_353377

open ProbabilityTheory
open Classical

noncomputable def dice_prob_space : ProbabilitySpace := sorry -- Define the probability space of rolling six 6-sided dice

def condition_no_four_of_a_kind (dice_outcome : Vector ℕ 6) : Prop :=
  ¬∃ n, dice_outcome.count n ≥ 4

def condition_pair_exists (dice_outcome : Vector ℕ 6) : Prop :=
  ∃ n, dice_outcome.count n = 2

def re_rolled_dice (initial_outcome : Vector ℕ 6) (re_roll : Vector ℕ 4) : Vector ℕ 6 :=
  sorry -- Combine initial pair and re-rolled outcomes

def at_least_four_same (dice_outcome : Vector ℕ 6) : Prop :=
  ∃ n, dice_outcome.count n ≥ 4

theorem four_dice_probability :
  ∀ (initial_outcome : Vector ℕ 6)
    (re_roll : Vector ℕ 4),
  (condition_no_four_of_a_kind initial_outcome) →
  (condition_pair_exists initial_outcome) →
  (∃ pr : ℚ, pr = 311 / 648 ∧ 
    (Pr[dice_prob_space, at_least_four_same (re_rolled_dice initial_outcome re_roll)] = pr)) :=
sorry

end four_dice_probability_l353_353377


namespace find_r_l353_353155

theorem find_r (r : ℝ) : log 16 (r + 16) = 5 / 4 ↔ r = 16 :=
by
  sorry

end find_r_l353_353155


namespace card_selection_sum_divisible_by_three_l353_353003

open Nat

theorem card_selection_sum_divisible_by_three :
  (∃ (cards : Finset ℕ), cards.card = 200 ∧
      (∀ card ∈ cards, ∃ k, card = 2 * k + 199) ∧ 
      ∃ ways : ℕ, ways = 437844 ∧ (ways = 
      (∑ multiset.choose (λ (triple : Multiset ℕ), cards.card = 3 ∧ 
      (∑ n, n ∈ triple) % 3 = 0)))) := 
  sorry

end card_selection_sum_divisible_by_three_l353_353003


namespace find_vector_p_l353_353704

noncomputable def a : ℝ × ℝ × ℝ := (2, -2, 4)
noncomputable def b : ℝ × ℝ × ℝ := (1, 6, 1)
noncomputable def p_expected : ℝ × ℝ × ℝ := (59 / 37, 94 / 37, 53 / 37)

theorem find_vector_p (v : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) 
  (h1 : p = v)
  (h2 : ∃ t : ℝ, ∀ i ∈ (Finset.range 3).val, p[i] = a[i] + t * (b[i] - a[i]))
  (h3 : ∀ i ∈ (Finset.range 3).val, (a[i] - b[i]) * p[i] = 0) :
  p = p_expected := 
sorry

end find_vector_p_l353_353704


namespace min_distance_l353_353291

def triangle_ABC (A B C D : Type) (BD : ℝ) :=
∃ (CD : ℝ), CD = 2 * BD ∧ 
  ∃ (AD AC AB : ℝ), AD = 2 ∧ 
    ∃ (α β γ : ℝ), α = 120 ∧ 
      angle_ADB = α ∧ 
        (AC / AB = (sqrt (3) - 1))

theorem min_distance (A B C D : Type) (BD : ℝ) :
  ∃ D ∈ BC, ∃ α β γ AD AC AB, 
    α = 120 ∧ AD = 2 ∧ D ∈ BC ∧ 
      γ = ∠ADB = α ∧ 
        CD = 2 * BD →
          (AC / AB) = (sqrt (3) - 1) := 
sorry

end min_distance_l353_353291


namespace sequence_accumulating_is_arithmetic_l353_353806

noncomputable def arithmetic_sequence {α : Type*} [LinearOrderedField α]
  (a : ℕ → α) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

theorem sequence_accumulating_is_arithmetic
  {α : Type*} [LinearOrderedField α] (a : ℕ → α) (S : ℕ → α)
  (na_gt_zero : ∀ n, a n > 0)
  (ha2 : a 2 = 3 * a 1)
  (hS_arith : arithmetic_sequence (λ n, (S n)^(1/2)))
  (hSn : ∀ n, S n = (∑ i in Finset.range (n+1), a i)) :
  arithmetic_sequence a := 
sorry

end sequence_accumulating_is_arithmetic_l353_353806


namespace range_f_l353_353130

def f (x : ℝ) : ℝ := (1 / 4) ^ x - 3 * (1 / 2) ^ x + 2

theorem range_f : 
  set.range (f ∘ (λ x, x : set.Icc (-2 : ℝ) (2 : ℝ) → ℝ)) = set.Icc (-(1 / 4) : ℝ) (6 : ℝ) := 
sorry

end range_f_l353_353130


namespace limit_proof_l353_353144

noncomputable def lim : ℝ := limit (λ x : ℝ, (x^2 - sin(x)^2) / (x - 1)) 1

theorem limit_proof : lim = 2 - 2 * cos(1) :=
by
  -- Proof goes here
  sorry

end limit_proof_l353_353144


namespace representation_of_1_l353_353154

theorem representation_of_1 (x y z : ℕ) (h : 1 = 1/x + 1/y + 1/z) : 
  (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 3 ∧ y = 3 ∧ z = 3) :=
by
  sorry

end representation_of_1_l353_353154


namespace correct_answer_to_blank_l353_353055

def question := "Do you like ______ here?"

def context := "Oh, yes. The air, the weather, the way of life. Everything is so nice."

def options := ["this", "these", "that", "it"]

def correct_fill (q : String) (ctx : String) (opts : List String) (ans : String) : Prop :=
  ∀ (opt : String), opt ∈ opts → (opt = ans ↔ opt = "it")

theorem correct_answer_to_blank : correct_fill question context options "it" :=
by
  intro opt h
  split
  { intros h1, rw h1 }
  { intro h2, assumption }

#check correct_answer_to_blank

end correct_answer_to_blank_l353_353055


namespace final_price_is_correct_l353_353892

-- Define the original price and the discount rate
variable (a : ℝ)

-- The final price of the product after two 10% discounts
def final_price_after_discounts (a : ℝ) : ℝ :=
  a * (0.9 ^ 2)

-- Theorem stating the final price after two consecutive 10% discounts
theorem final_price_is_correct (a : ℝ) :
  final_price_after_discounts a = a * (0.9 ^ 2) :=
by sorry

end final_price_is_correct_l353_353892


namespace minimize_F_l353_353845

theorem minimize_F : ∃ x1 x2 x3 x4 x5 : ℝ, 
  (-2 * x1 + x2 + x3 = 2) ∧ 
  (x1 - 2 * x2 + x4 = 2) ∧ 
  (x1 + x2 + x5 = 5) ∧ 
  (x1 ≥ 0) ∧ 
  (x2 ≥ 0) ∧ 
  (x2 - x1 = -3) :=
by {
  sorry
}

end minimize_F_l353_353845


namespace exists_Q_R_l353_353186

noncomputable def P (x : ℚ) : ℚ := x^4 + x^3 + x^2 + x + 1

theorem exists_Q_R : ∃ (Q R : Polynomial ℚ), 
  (Q.degree > 0 ∧ R.degree > 0) ∧
  (∀ (y : ℚ), (Q.eval y) * (R.eval y) = P (5 * y^2)) :=
sorry

end exists_Q_R_l353_353186


namespace min_value_expression_l353_353203

theorem min_value_expression (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 3) (hy : 1 ≤ y ∧ y ≤ 4) : 
  ∃ z, z = (x + y) / x ∧ z = 4 / 3 := by
  sorry

end min_value_expression_l353_353203


namespace minimum_BD_value_l353_353300

theorem minimum_BD_value {A B C D : Type*} [Real A] [Real B] [Real C] [Real D] (angle_ADB: A) (AD: A) (CD: A) (BD: A):
  (D ∈ line_segment(B, C)) →
  angle_ADB = 120 → 
  AD = 2 → 
  CD = 2 * BD →
  ∃ BD: A, BD = sqrt(3) - 1 :=
by
  sorry

end minimum_BD_value_l353_353300


namespace pascal_sixth_element_row_20_l353_353442

theorem pascal_sixth_element_row_20 : (Nat.choose 20 5) = 7752 := 
  by
  sorry

end pascal_sixth_element_row_20_l353_353442


namespace find_b_value_l353_353757

def b_value_correct (A B C a : ℝ) (B_is_60 : B = 60) (C_is_75 : C = 75) (a_is_4 : a = 4) : Prop :=
    let b := 4 * Real.sqrt (3/2) in b = 4 * Real.sqrt(3/2)

theorem find_b_value : ∀ (A B C a : ℝ), B = 60 → C = 75 → a = 4 → b_value_correct A B C a :=
by
  intros A B C a B_is_60 C_is_75 a_is_4
  unfold b_value_correct
  sorry

end find_b_value_l353_353757


namespace max_value_of_even_quadratic_l353_353246

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem max_value_of_even_quadratic (a : ℝ) (h1 : b = 0) (h2 : 1 + a = 2 * a) : 
  ∃ M, (∀ x ∈ set.Icc (-1 - a) (2 * a), f a b x ≤ M) ∧ M = 5 := 
by
  sorry

end max_value_of_even_quadratic_l353_353246


namespace value_of_k_l353_353782

theorem value_of_k (k m : ℝ)
    (h1 : m = k / 3)
    (h2 : 2 = k / (3 * m - 1)) :
    k = 2 := by
  sorry

end value_of_k_l353_353782


namespace classify_quadrilateral_l353_353489

theorem classify_quadrilateral (Q : Type) [quadrilateral Q] 
    (equal_sides : ∀ a b, a ≠ b → length a b = 1) 
    (perpendicular_diagonals : ∀ d1 d2, diagonal d1 → diagonal d2 → perp d1 d2) 
    : is_square Q :=
  sorry

end classify_quadrilateral_l353_353489


namespace mismatching_socks_l353_353383

theorem mismatching_socks (total_socks : ℕ) (pairs : ℕ) (socks_per_pair : ℕ) 
  (h1 : total_socks = 25) (h2 : pairs = 4) (h3 : socks_per_pair = 2) : 
  total_socks - (socks_per_pair * pairs) = 17 :=
by
  sorry

end mismatching_socks_l353_353383


namespace find_lambda_collinear_l353_353230

variable (λ : ℝ)

def vector_add := (λ + 2, 2 * λ + 3)
def vector_c := (-4, -7)

theorem find_lambda_collinear :
  -4 * (2 * λ + 3) - (λ + 2) * (-7) = 0 → λ = 2 := 
sorry

end find_lambda_collinear_l353_353230


namespace four_spheres_enclose_point_l353_353789

theorem four_spheres_enclose_point (O : Point) (M : Point) 
  (ABCD : Tetrahedron) (h : is_regular_tetrahedron ABCD ∧ center ABCD = O) :
  ∃ (spheres : List Sphere), List.length spheres = 4 ∧
  (∃ S ∈ spheres, separates S O M) :=
sorry

end four_spheres_enclose_point_l353_353789


namespace doubled_container_volume_l353_353479

theorem doubled_container_volume (v : ℝ) (h₁ : v = 4) (h₂ : ∀ l w h : ℝ, v = l * w * h) : 8 * v = 32 := 
by
  -- The proof will go here, this is just the statement
  sorry

end doubled_container_volume_l353_353479


namespace one_valid_palindrome_year_l353_353516

/-- A year is a palindrome if its representation reads the same forwards and backwards. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

/-- Prime palindromes are palindromes that are also prime numbers. -/
def is_prime_palindrome (n : ℕ) : Prop :=
  is_prime n ∧ is_palindrome n

/-- Checks if a number can be factored into two three-digit palindromes. -/
def can_be_factored_by_three_digit_palindromes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a * b = n ∧ 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 ∧ is_palindrome a ∧ is_palindrome b

/-- Checks if a number can be factored into two three-digit palindromes, 
where one of the factors is a prime palindrome. -/
def can_be_factored_with_prime_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, a * b = n ∧ is_prime_palindrome a ∧ 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 ∧ is_palindrome b

/-- The main theorem stating the proof problem. -/
theorem one_valid_palindrome_year :
  ∃! (y : ℕ), 2000 ≤ y ∧ y < 3000 ∧ is_palindrome y ∧ can_be_factored_with_prime_palindrome y :=
sorry

end one_valid_palindrome_year_l353_353516


namespace cost_of_jeans_and_shirts_l353_353110

theorem cost_of_jeans_and_shirts 
  (S : ℕ) (J : ℕ) (X : ℕ)
  (hS : S = 18)
  (h2J3S : 2 * J + 3 * S = 76)
  (h3J2S : 3 * J + 2 * S = X) :
  X = 69 :=
by
  sorry

end cost_of_jeans_and_shirts_l353_353110


namespace exists_measurable_D_l353_353460

variable {E : Type*} {𝒮 : Type*} [MeasurableSpace E] 
          {μ ν : MeasureTheory.Measure E}

/-- 
  Given measures μ and ν on (E, 𝒮), and a condition that for every δ > 0, 
  there exists a measurable set E_δ such that μ(E_δ) < δ and ν(Eᶜ E_δ) < δ,
  we want to show that there exists a measurable set D with ν(D) = 0 
  and for every measurable set B, μ(B) = μ(B ∩ D). 
-/
theorem exists_measurable_D (μ ν : MeasureTheory.Measure E)
  (h : ∀ δ > 0, ∃ E_δ ∈ 𝒮, μ E_δ < δ ∧ ν (E \ E_δ) < δ) :
  ∃ D ∈ 𝒮, ν D = 0 ∧ ∀ B ∈ 𝒮, μ B = μ (B ∩ D) := by 
  sorry

end exists_measurable_D_l353_353460


namespace find_BD_when_AC_over_AB_min_l353_353324

-- Definition of the conditions
variables (A B C D : Type) [metric_space A]
variables (triangle : set A)
variables (on_side_BC : D ∈ segment B C)
variables (angle_ADB : ∠ADB = 120)
variables (AD_len : dist A D = 2)
variables (CD_2BD : dist C D = 2 * dist B D)

noncomputable def minimum_ratio_condition (BD : ℝ) : Prop := 
  let x := dist B D
  let CD := 2 * x
  let AD := 2 in
  let angle_ADB_cos := -1/2 in
  let b_squared := AD^2 + CD^2 - 2 * AD * CD * angle_ADB_cos in
  let c_squared := AD^2 + x^2 + 2 * AD * x * angle_ADB_cos in
  b_squared / c_squared = 4 - (12 / (x + 1 + 3/(x + 1))) → x = sqrt 3 - 1

theorem find_BD_when_AC_over_AB_min (BD : ℝ) : minimum_ratio_condition A B C D 
:= sorry

end find_BD_when_AC_over_AB_min_l353_353324


namespace max_sides_convex_polygon_max_sides_concave_polygon_l353_353638

/-- A polygon with n sides, two adjacent sides of length 1, and all diagonals of integer length -/
def polygon_has_integer_diagonals (n : ℕ) (polygon : Type) [polygon_has_n_sides : polygon.n_sides = n] : Prop :=
  -- Two adjacent sides are of length 1
  polygon.adjacent_sides_length = 1 → polygon.integer_diagonals → true

/-- Prove that the maximum value of n is 4 for a convex polygon -/
theorem max_sides_convex_polygon : ∀ polygon : Type, 
  polygon_has_integer_diagonals 4 polygon → 
  ∀ n : ℕ, polygon_has_integer_diagonals n polygon → n ≤ 4 :=
sorry

/-- Prove that the maximum value of n is 5 for a concave polygon -/
theorem max_sides_concave_polygon : ∀ polygon : Type, 
  polygon_has_integer_diagonals 5 polygon → 
  ∀ n : ℕ, polygon_has_integer_diagonals n polygon → n ≤ 5 :=
sorry

end max_sides_convex_polygon_max_sides_concave_polygon_l353_353638


namespace proposition_relation_l353_353344

variable (p q : Prop)

-- Propositions Definitions
def propA : Prop := p → q
def propB : Prop := p ↔ q

-- Necessary but not sufficient condition
theorem proposition_relation : (propB → propA) ∧ ¬(propA → propB) := 
by {
  -- Necessary part: propB → propA
  have necessary : propB → propA := by sorry,

  -- Not sufficient part: ¬(propA → propB)
  have not_sufficient : ¬(propA → propB) := by sorry,

  exact ⟨necessary, not_sufficient⟩
}

end proposition_relation_l353_353344


namespace fraction_increase_l353_353960

-- Define the problem conditions and the proof statement
theorem fraction_increase (m n : ℤ) (hnz : n ≠ 0) (hnnz : n ≠ -1) (h : m < n) :
  (m : ℚ) / n < (m + 1 : ℚ) / (n + 1) :=
by sorry

end fraction_increase_l353_353960


namespace min_distance_l353_353293

def triangle_ABC (A B C D : Type) (BD : ℝ) :=
∃ (CD : ℝ), CD = 2 * BD ∧ 
  ∃ (AD AC AB : ℝ), AD = 2 ∧ 
    ∃ (α β γ : ℝ), α = 120 ∧ 
      angle_ADB = α ∧ 
        (AC / AB = (sqrt (3) - 1))

theorem min_distance (A B C D : Type) (BD : ℝ) :
  ∃ D ∈ BC, ∃ α β γ AD AC AB, 
    α = 120 ∧ AD = 2 ∧ D ∈ BC ∧ 
      γ = ∠ADB = α ∧ 
        CD = 2 * BD →
          (AC / AB) = (sqrt (3) - 1) := 
sorry

end min_distance_l353_353293


namespace triangle_area_and_perimeter_l353_353169

-- Given conditions
def vertices (x : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := ((0, 0), (x, x + 4), (x, 0))

-- The statement we need to prove
theorem triangle_area_and_perimeter (x : ℝ) (h₀ : x > 0) (h₁ : (1 / 2) * x * (x + 4) = 50) : 
  x = 8.2 ∧ let p := x + (x + 4) + real.sqrt (x ^ 2 + (x + 4) ^ 2) in p = 35.1 := 
by 
  sorry

end triangle_area_and_perimeter_l353_353169


namespace probability_multiple_of_3_or_7_l353_353891

theorem probability_multiple_of_3_or_7 :
  (∃ (cards : Finset ℕ) (n : ℕ),
    (∀ x ∈ cards, 1 ≤ x ∧ x ≤ 30) ∧ 
    (∀ x ∈ cards, x = 3 ∨ x = 6 ∨ x = 9 ∨ x = 12 ∨ x = 15 ∨ x = 18 ∨ x = 21 ∨ x = 24 ∨ x = 27 ∨ x = 30 ∨
                  x = 7 ∨ x = 14 ∨ x = 21 ∨ x = 28) ∧
    (∃ fav_outcomes, fav_outcomes = (13 : ℕ)) ∧
    (n = 30) ∧
    (fav_outcomes.to_nat / n.to_nat = (13 : ℕ) / (30 : ℕ))) :=
begin
    sorry
end

end probability_multiple_of_3_or_7_l353_353891


namespace largest_integer_less_than_100_with_remainder_5_l353_353628

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353628


namespace area_triangle_CEF_l353_353370

noncomputable def area_of_triangle_CEF : ℝ :=
  let A := (0, 0)
  let B := (6, 0)  -- AB is solved to be 6 units
  let C := (6, 8)
  let D := (0, 8)
  let E := ((0 + 6) / 2, (0 + 0) / 2)  -- Midpoint of AB
  let F := ((6 + 0) / 2, (8 + 8) / 2)  -- Midpoint of CD
  let base : ℝ := E.1 - C.1  -- base = E.x - C.x
  let height : ℝ := F.2 - C.2  -- height = F.y - C.y
  (1/2) * abs (base * height)

theorem area_triangle_CEF :
  let AD := 8
  let Area := 48
  AD = 8 ∧ Area = 48 →
  area_of_triangle_CEF = 12 :=
begin
  sorry
end

end area_triangle_CEF_l353_353370


namespace find_a_l353_353676

theorem find_a (a : ℝ) (h_a_pos : a > 0)
  (M : ℝ × ℝ) (h_M : M = (2, 0))
  (slope : ℝ) (h_slope : slope = Real.sqrt 3)
  (line_eq : (ℝ × ℝ) → ℝ) (h_line : ∀ x y, line_eq (x, y) = y - slope * (x - 2))
  (axis_of_symmetry_eq : ℝ → ℝ) (h_axis : ∀ x, axis_of_symmetry_eq x = -a / 4)
  (B : ℝ × ℝ) (h_B : B = (-a / 4, -sqrt(3) / 4 * a - 2 * sqrt(3)))
  (A : ℝ × ℝ) (h_A : A = (4 + a / 4, sqrt(3) * (a / 4 + 2)))
  (h_midpoint : 2 * (M.1, M.2) = (B.1 + A.1, B.2 + A.2))
  (parabola_eq : ℝ × ℝ → Prop) (h_parabola : ∀ x y, parabola_eq (x, y) = (y^2 = a * x)) :
  a = 8 :=
sorry

end find_a_l353_353676


namespace largest_integer_less_than_100_with_remainder_5_l353_353635

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l353_353635


namespace largest_integer_less_than_100_div_8_rem_5_l353_353617

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l353_353617


namespace count_integers_between_sqrt_10_and_sqrt_100_l353_353718

theorem count_integers_between_sqrt_10_and_sqrt_100 :
  ∃ n : ℕ, (∀ k : ℕ, 4 ≤ k ∧ k ≤ 10 → k ∈ set.Ico (int.floor (sqrt (10 : ℝ))) (int.ceil (sqrt (100 : ℝ)))) ∧ n = 7 :=
sorry

end count_integers_between_sqrt_10_and_sqrt_100_l353_353718


namespace count_integers_between_sqrt10_sqrt100_l353_353714

theorem count_integers_between_sqrt10_sqrt100 : 
  ∃ (S : Set ℤ), (∀ n, n ∈ S ↔ (real.sqrt 10 < n ∧ n < real.sqrt 100)) ∧ S.card = 6 := 
by
  sorry

end count_integers_between_sqrt10_sqrt100_l353_353714


namespace evaluate_expression_l353_353143

noncomputable def cuberoot (x : ℝ) : ℝ := x ^ (1 / 3)

theorem evaluate_expression : 
  cuberoot (1 + 27) * cuberoot (1 + cuberoot 27) = cuberoot 112 := 
by 
  sorry

end evaluate_expression_l353_353143


namespace sin_and_sin_expression_l353_353652

noncomputable def solve_sin_values (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2)) 
  (h3 : Real.cos (α + (Real.pi / 6)) = 1 / 3) : Prop :=
  Real.sin α = (2 * Real.sqrt 6 + 1) / 6 ∧ Real.sin (2 * α + (5 * Real.pi / 6)) = -7 / 9

theorem sin_and_sin_expression (α : ℝ) (h : solve_sin_values α (by linarith) (by linarith) (by norm_num)) :
  solve_sin_values α (by linarith) (by linarith) (by norm_num) :=
sorry

end sin_and_sin_expression_l353_353652


namespace sum_of_n_values_if_continuous_l353_353353

noncomputable def f (x n : ℝ) : ℝ :=
if x < n then x^2 + 2*x + 3 else 3*x + 6

theorem sum_of_n_values_if_continuous :
  (∀ x, f x n = if x < n then x^2 + 2*x + 3 else 3*x + 6) →
  (∃ (n1 n2 : ℝ), 
  (∀ x, (x^2 + 2*x + 3 = 3*x + 6) ↔ (x = n1 ∨ x = n2)) ∧ n1 + n2 = 2) ∧
  (∀ n, (f x n).continuous_at n) :=
by sorry

end sum_of_n_values_if_continuous_l353_353353


namespace n_is_square_l353_353762

theorem n_is_square (n m : ℕ) (h1 : 3 ≤ n) (h2 : m = (n * (n - 1)) / 2) (h3 : ∃ (cards : Finset ℕ), 
  (cards.card = n) ∧ (∀ i ∈ cards, i ∈ Finset.range (m + 1)) ∧ 
  (∀ (i j : ℕ) (hi : i ∈ cards) (hj : j ∈ cards), i ≠ j → 
    ((i + j) % m) ≠ ((i + j) % m))) : 
  ∃ k : ℕ, n = k * k := 
sorry

end n_is_square_l353_353762


namespace pyramid_height_equals_6_point_48_l353_353474

theorem pyramid_height_equals_6_point_48 (V_cube : ℝ) (V_pyramid : ℝ) (h : ℝ) : 
  let edge_length_cube := 6
      base_edge_length_pyramid := 10
  in
  V_cube = edge_length_cube^3 →
  V_pyramid = (1/3) * base_edge_length_pyramid^2 * h →
  V_cube = V_pyramid →
  h = 6.48 :=
by
  -- Proof to be filled in
  sorry

end pyramid_height_equals_6_point_48_l353_353474


namespace find_P2010_l353_353262

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define initial point P1
def P1 : Point := ⟨1, 0⟩

-- Define a function to perform rotation by 30° around the origin
def rotate (p : Point) (deg : ℝ) : Point :=
  let θ := Real.pi * deg / 180
  ⟨p.x * Real.cos θ - p.y * Real.sin θ, p.x * Real.sin θ + p.y * Real.cos θ⟩

-- Define a function to extend the point by a factor
@[simp] def extend (p : Point) (factor : ℝ) : Point :=
  ⟨p.x * factor, p.y * factor⟩

-- Define the iterative process to determine P_k
def P : ℕ → Point
| 1     := P1
| (n+1) := extend (rotate (P n) 30) 2

-- Define the statement to be proved
theorem find_P2010 : (P 2010) = ⟨0, -2^1004⟩ := sorry

end find_P2010_l353_353262


namespace event_A_prob_correct_l353_353122

noncomputable def event_A_prob (n n1 n2 m11 m12 m21 m22 : ℕ) : ℚ :=
  if h : n1 + n2 = n ∧ |m12 - m21| ≤ 1 
  then (nat.choose (n1 - 1) m22 * nat.choose (n2 - 1) m11) / nat.choose n n1
  else 0

theorem event_A_prob_correct (n n1 n2 m11 m12 m21 m22 : ℕ) 
  (h1 : n1 + n2 = n) 
  (h2 : |m12 - m21| ≤ 1) :
  event_A_prob n n1 n2 m11 m12 m21 m22 = 
  (nat.choose (n1 - 1) m22 * nat.choose (n2 - 1) m11) / nat.choose n n1 := 
sorry

end event_A_prob_correct_l353_353122


namespace task1_task2_task3_l353_353858

-- Define the conditions for rotational functions
def is_rotational_function (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 + a2 = 0 ∧ b1 = b2 ∧ c1 + c2 = 0

-- Task 1: Prove the rotational function for y = x^2 - 4x + 3
theorem task1 :
  ∃ (a2 b2 c2 : ℝ),
  is_rotational_function 1 (-4) 3 a2 b2 c2 ∧
  (a2 = -1) ∧ (b2 = -4) ∧ (c2 = -3) :=
by
  sorry

-- Task 2: Prove the value of (m+n)^2023 given the rotational relationship
theorem task2 (m n : ℝ) :
  is_rotational_function 5 (m + 1) n (-5) (-n) (-3) →
  (m + n) ^ 2023 = -1 :=
by
  sorry

-- Task 3: Prove the rotational function relationship for y = 2(x-1)(x+3)
theorem task3 :
  let A := (1, 0 : ℝ)
  let B := (-3, 0 : ℝ)
  let C := (0, -6 : ℝ)
  let A1 := (-1, 0 : ℝ)
  let B1 := (3, 0 : ℝ)
  let C1 := (0, 6 : ℝ)
  ∃ a b c : ℝ,
  (∃ a' b' c' : ℝ,
   is_rotational_function 2 (-8) 6 a' b' c' ∧
   (a' = -2) ∧ (b' = 4) ∧ (c' = 6)) :=
by
  sorry

end task1_task2_task3_l353_353858


namespace well_minimizes_distance_l353_353179

variables (A B C D O : Point)
variables (AC BD : Line)
variable [ConvexQuadrilateral A B C D]
variable [IntersectingDiagonals AC BD O]

theorem well_minimizes_distance : 
  forall (M : Point), M ≠ O → (dist A M + dist B M + dist C M + dist D M > dist A O + dist B O + dist C O + dist D O) :=  
by 
  sorry

end well_minimizes_distance_l353_353179


namespace min_ratio_bd_l353_353309

noncomputable def triangle_ratio_min (BD : ℝ) : Prop :=
  ∃ K L : ℝ, 
    (K = 1) ∧ 
    (L = 1) ∧ 
    (∀ (x : ℝ), ∀ (y : ℝ), (CD = 2 * BD) → ((x = sqrt(3) - 1) → y = (bd + 1 + 3 / (bd + 1)) ∧ (x + y) = 2 * K * L) ∧ 
    (abs x ≤ (sqrt 3 - 1)))


theorem min_ratio_bd (BD : ℝ) : triangle_ratio_min BD :=
by { sorry }

end min_ratio_bd_l353_353309


namespace milk_needed_for_cookies_l353_353009

-- Define the given conditions
def liters_to_cups (liters : ℕ) : ℕ := liters * 4

def milk_per_cookies (cups cookies : ℕ) : ℚ := cups / cookies

-- Define the problem statement
theorem milk_needed_for_cookies (h1 : milk_per_cookies 20 30 = milk_per_cookies x 12) : x = 8 :=
sorry

end milk_needed_for_cookies_l353_353009


namespace sum_real_solutions_eq_l353_353641

theorem sum_real_solutions_eq : 
  let f := λ x : ℝ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 8 * x + 3)
  ∑ x in (set_of f), x = 11 / 2 := sorry

end sum_real_solutions_eq_l353_353641


namespace ceil_sqrt_200_eq_15_l353_353537

theorem ceil_sqrt_200_eq_15 : Int.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l353_353537


namespace doubled_container_volume_l353_353482

theorem doubled_container_volume (original_volume : ℕ) (factor : ℕ) 
  (h1 : original_volume = 4) (h2 : factor = 8) : original_volume * factor = 32 :=
by 
  rw [h1, h2]
  norm_num

end doubled_container_volume_l353_353482


namespace xiaozhi_needs_median_for_top_10_qualification_l353_353779

-- Define a set of scores as a list of integers
def scores : List ℕ := sorry

-- Assume these scores are unique (this is a condition given in the problem)
axiom unique_scores : ∀ (a b : ℕ), a ∈ scores → b ∈ scores → a ≠ b → scores.indexOf a ≠ scores.indexOf b

-- Define the median function (in practice, you would implement this, but we're just outlining it here)
def median (scores: List ℕ) : ℕ := sorry

-- Define the position of Xiao Zhi's score
def xiaozhi_score : ℕ := sorry

-- Given that the top 10 scores are needed to advance
def top_10 (scores: List ℕ) : List ℕ := scores.take 10

-- Proposition that Xiao Zhi needs median to determine his rank in top 10
theorem xiaozhi_needs_median_for_top_10_qualification 
    (scores_median : ℕ) (zs_score : ℕ) : 
    (∀ (s: List ℕ), s = scores → scores_median = median s → zs_score ≤ scores_median → zs_score ∉ top_10 s) ∧ 
    (exists (s: List ℕ), s = scores → zs_score ∉ top_10 s → zs_score ≤ scores_median) := 
sorry

end xiaozhi_needs_median_for_top_10_qualification_l353_353779


namespace binom_n_2_l353_353935

theorem binom_n_2 (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_n_2_l353_353935


namespace even_permutations_exactly_four_transpositions_l353_353993

theorem even_permutations_exactly_four_transpositions :
  let s := {1, 2, 3, 4, 5, 6}
  in (∃ p : Equiv.Perm s, p.is_even ∧ 
      p.transpositions.count = 4 
      ∧ p = Equiv.refl s) →
  (card {p : Equiv.Perm s | p.is_even ∧ p.transpositions.count = 4} = 360) :=
begin
  sorry
end

end even_permutations_exactly_four_transpositions_l353_353993


namespace problem_statement_l353_353497

def scientific_notation_correct (x : ℝ) : Prop :=
  x = 5.642 * 10 ^ 5

theorem problem_statement : scientific_notation_correct 564200 :=
by
  sorry

end problem_statement_l353_353497


namespace height_of_original_triangle_l353_353908

variable (a b c : ℝ)

theorem height_of_original_triangle (a b c : ℝ) : 
  ∃ h : ℝ, h = a + b + c :=
  sorry

end height_of_original_triangle_l353_353908


namespace sum_of_integers_is_eleven_l353_353872

theorem sum_of_integers_is_eleven (p q r s : ℤ) 
  (h1 : p - q + r = 7) 
  (h2 : q - r + s = 8) 
  (h3 : r - s + p = 4) 
  (h4 : s - p + q = 3) : 
  p + q + r + s = 11 := 
by
  sorry

end sum_of_integers_is_eleven_l353_353872


namespace find_number_with_10_questions_l353_353737

theorem find_number_with_10_questions (n : ℕ) (h : n ≤ 1000) : n = 300 :=
by
  sorry

end find_number_with_10_questions_l353_353737


namespace group_average_age_l353_353052

theorem group_average_age (n : ℕ) (h1 : float) (h2 : float) (h3 : float) : n = 9 :=
  assume n h1 h2 h3,
  have h1 : h1 = 15, from sorry,
  have h2 : 35,        -- extra person joining
  have h3 : h2 = 17, from sorry,
  have eq1 : int = (15 * nh2), from sorry, --first equation
  have eq2 : (h1 + h1 + h3) = (17 (n+1)), from sorry, -- second equation
  begin
    sorry
  end

end group_average_age_l353_353052


namespace exists_awesome_2009_l353_353117

-- Define an awesome set of rooks on an infinite chessboard
structure awesome_set (R : Finset ℕ) (C : Finset ℕ) :=
  (no_attack : ∀ (r1 r2 ∈ R) (c1 c2 ∈ C), (r1 ≠ r2 ∧ c1 ≠ c2))
  (full_attack : ∀ (r c : ℕ), r ∉ R ∧ c ∉ C → ∃ r' ∈ R, ∃ c' ∈ C, (r' = r ∨ c' = c))

-- Existence of awesome sets with specific sizes
variable (board : Finset ℕ)
axiom exists_awesome_2008 : ∃ (R C : Finset ℕ), R.card = 2008 ∧ C.card = 2008 ∧ awesome_set R C
axiom exists_awesome_2010 : ∃ (R C : Finset ℕ), R.card = 2010 ∧ C.card = 2010 ∧ awesome_set R C

-- Statement to prove
theorem exists_awesome_2009 : ∃ (R C : Finset ℕ), R.card = 2009 ∧ C.card = 2009 ∧ awesome_set R C := sorry

end exists_awesome_2009_l353_353117


namespace sum_of_coefficients_equals_neg2_l353_353651

theorem sum_of_coefficients_equals_neg2 :
  let p := (x^2 - 3*x + 1)^5
  ∃ (a : ℕ → ℤ), (p = ∑ i in range (11), a i * x^i) ∧
    (∑ i in range (1, 11), a i) = -2 := 
sorry

end sum_of_coefficients_equals_neg2_l353_353651


namespace total_children_is_11_l353_353862

noncomputable def num_of_children (b g : ℕ) := b + g

theorem total_children_is_11 (b g : ℕ) :
  (∃ c : ℕ, b * c + g * (c + 1) = 47) ∧
  (∃ m : ℕ, b * (m + 1) + g * m = 74) → 
  num_of_children b g = 11 :=
by
  -- The proof steps would go here to show that b + g = 11
  sorry

end total_children_is_11_l353_353862


namespace expression_value_l353_353177

variables {a b c : ℝ}

theorem expression_value (h : a * b + b * c + c * a = 3) :
  (a * (b^2 + 3) / (a + b)) + (b * (c^2 + 3) / (b + c)) + (c * (a^2 + 3) / (c + a)) = 6 := 
  sorry

end expression_value_l353_353177


namespace total_fish_count_l353_353392

def number_of_tables : ℕ := 32
def fish_per_table : ℕ := 2
def additional_fish_table : ℕ := 1
def total_fish : ℕ := (number_of_tables * fish_per_table) + additional_fish_table

theorem total_fish_count : total_fish = 65 := by
  sorry

end total_fish_count_l353_353392


namespace binom_two_eq_l353_353941

theorem binom_two_eq (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binom_two_eq_l353_353941


namespace max_n_value_l353_353458

theorem max_n_value (N : ℕ) (h1 : 0 < N) (h2 : N < 1000000)
  (h3 : (14 * N) % 60 = 0) (h4 : ∃ p1 p2 p3 : ℕ, nat.prime p1 ∧ 
  nat.prime p2 ∧ nat.prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ N = p1 * p2 * p3):
  N = 330:= sorry

end max_n_value_l353_353458


namespace ceil_sqrt_200_eq_15_l353_353534

theorem ceil_sqrt_200_eq_15 (h1 : Real.sqrt 196 = 14) (h2 : Real.sqrt 225 = 15) (h3 : 196 < 200 ∧ 200 < 225) : 
  Real.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l353_353534


namespace sequence_accumulating_is_arithmetic_l353_353808

noncomputable def arithmetic_sequence {α : Type*} [LinearOrderedField α]
  (a : ℕ → α) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

theorem sequence_accumulating_is_arithmetic
  {α : Type*} [LinearOrderedField α] (a : ℕ → α) (S : ℕ → α)
  (na_gt_zero : ∀ n, a n > 0)
  (ha2 : a 2 = 3 * a 1)
  (hS_arith : arithmetic_sequence (λ n, (S n)^(1/2)))
  (hSn : ∀ n, S n = (∑ i in Finset.range (n+1), a i)) :
  arithmetic_sequence a := 
sorry

end sequence_accumulating_is_arithmetic_l353_353808


namespace angles_sum_540_l353_353181

theorem angles_sum_540 (p q r s : ℝ) (h1 : ∀ a, a + (180 - a) = 180)
  (h2 : ∀ a b, (180 - a) + (180 - b) = 360 - a - b)
  (h3 : ∀ p q r, (360 - p - q) + (180 - r) = 540 - p - q - r) :
  p + q + r + s = 540 :=
sorry

end angles_sum_540_l353_353181


namespace num_integers_between_sqrt10_sqrt100_l353_353706

theorem num_integers_between_sqrt10_sqrt100 : 
  ∃ n : ℕ, n = 7 ∧ ∀ x : ℤ, (⌊Real.sqrt 10⌋ + 1 <= x) ∧ (x <= ⌈Real.sqrt 100⌉ - 1) ↔ (4 <= x ∧ x <= 10) := 
by 
  sorry

end num_integers_between_sqrt10_sqrt100_l353_353706


namespace exists_x_in_interval_satisfying_sum_l353_353363

theorem exists_x_in_interval_satisfying_sum (a b c d : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) 
    (hb : 0 ≤ b) (hb1 : b ≤ 1)
    (hc : 0 ≤ c) (hc1 : c ≤ 1)
    (hd : 0 ≤ d) (hd1 : d ≤ 1) :
    ∃ (x : ℝ), x ∈ (set.Icc 0 1) ∧ 
    ((1 / |x - a|) + (1 / |x - b|) + (1 / |x - c|) + (1 / |x - d|)) < 40 := 
by
  sorry

end exists_x_in_interval_satisfying_sum_l353_353363


namespace parabola_focus_directrix_distance_is_two_l353_353192

noncomputable theory
open_locale classical

variables (p y₀ : ℝ) (M : ℝ × ℝ) (F : ℝ × ℝ) 
  (C : set (ℝ × ℝ)) (d : ℝ)

def parabola_y2_eq_2px (p : ℝ) : set (ℝ × ℝ) := {xy | xy.2 ^ 2 = 2 * p * xy.1}

def focus (p : ℝ) := (p / 2, 0)

def point_on_parabola (y₀ p : ℝ) : ℝ × ℝ := (y₀ ^ 2 / (2 * p), y₀)

def directrix_distance (p : ℝ) : ℝ := p

def circle_with_diameter_mf (M F : ℝ × ℝ) : set (ℝ × ℝ) :=
{P | ∃ t : ℝ, P ≠ M ∧ P ≠ F ∧ 
  (P.1 - M.1) * (P.1 - F.1) + (P.2 - M.2) * (P.2 - F.2) = 0}

theorem parabola_focus_directrix_distance_is_two :
  (∃ p y₀, 0 < p ∧ M = point_on_parabola y₀ p ∧
        F = focus p ∧
        (M.1 - F.1)^2 + (M.2 - F.2)^2 = 4 ∧
        (0,1) ∈ circle_with_diameter_mf M F) →
  directrix_distance p = 2 := 
sorry

end parabola_focus_directrix_distance_is_two_l353_353192


namespace minimum_BD_value_l353_353308

theorem minimum_BD_value {A B C D : Type*} [Real A] [Real B] [Real C] [Real D] (angle_ADB: A) (AD: A) (CD: A) (BD: A):
  (D ∈ line_segment(B, C)) →
  angle_ADB = 120 → 
  AD = 2 → 
  CD = 2 * BD →
  ∃ BD: A, BD = sqrt(3) - 1 :=
by
  sorry

end minimum_BD_value_l353_353308


namespace find_BD_when_AC_over_AB_min_l353_353321

-- Definition of the conditions
variables (A B C D : Type) [metric_space A]
variables (triangle : set A)
variables (on_side_BC : D ∈ segment B C)
variables (angle_ADB : ∠ADB = 120)
variables (AD_len : dist A D = 2)
variables (CD_2BD : dist C D = 2 * dist B D)

noncomputable def minimum_ratio_condition (BD : ℝ) : Prop := 
  let x := dist B D
  let CD := 2 * x
  let AD := 2 in
  let angle_ADB_cos := -1/2 in
  let b_squared := AD^2 + CD^2 - 2 * AD * CD * angle_ADB_cos in
  let c_squared := AD^2 + x^2 + 2 * AD * x * angle_ADB_cos in
  b_squared / c_squared = 4 - (12 / (x + 1 + 3/(x + 1))) → x = sqrt 3 - 1

theorem find_BD_when_AC_over_AB_min (BD : ℝ) : minimum_ratio_condition A B C D 
:= sorry

end find_BD_when_AC_over_AB_min_l353_353321


namespace cos_QRP_eq_uv_l353_353260

variables {P Q R S : Type} [geometry_space P Q R S]

-- Define the angles
variable (QPR_angle : angle Q P R = 90)
variable (QSP_angle : angle Q S P = 90)
variable (QRS_angle : angle Q R S = 90)

-- Define the sines of the angles
variable (u : ℝ) (v : ℝ)
variable (u_def : u = sin (angle Q P R))
variable (v_def : v = sin (angle Q S P))

theorem cos_QRP_eq_uv (h₁ : angle P Q S = 90) 
  (h₂ : angle P R S = 90) 
  (h₃ : angle Q R S = 90) 
  (u_eq : u = sin (angle Q P R)) 
  (v_eq : v = sin (angle Q S P)) :
  cos (angle Q R P) = u * v := 
sorry

end cos_QRP_eq_uv_l353_353260


namespace largest_integer_lt_100_with_remainder_5_div_8_l353_353599

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l353_353599


namespace count_integers_between_sqrt10_sqrt100_l353_353726

theorem count_integers_between_sqrt10_sqrt100 :
  ∃ n : ℕ, n = 7 ∧ card {x : ℤ | real.sqrt 10 < x ∧ x < real.sqrt 100} = n :=
by
  sorry

end count_integers_between_sqrt10_sqrt100_l353_353726


namespace part1_l353_353972

theorem part1 : (16 : ℝ)^(1/2) + (1 / 81 : ℝ)^(-0.25) - (-1 / 2 : ℝ)^0 = 10 / 3 := by
  sorry

end part1_l353_353972


namespace lines_parallel_l353_353339

open_locale classical

noncomputable theory

variables {Γ₁ Γ₂ : Type*} [circle Γ₁] [circle Γ₂]
variables {A B C D E F : Γ₁}
variables {dA dB : line}
variables (C ∈ Γ₁ ∧ E ∈ Γ₂)
variables (D ∈ Γ₁ ∧ F ∈ Γ₂)

theorem lines_parallel (h : C ∈ dA ∧ E ∈ dA ∧ D ∈ dB ∧ F ∈ dB) : 
  parallel (line_through C D) (line_through E F) :=
begin
  sorry
end

end lines_parallel_l353_353339


namespace boat_speed_is_20_l353_353903

-- Definitions based on conditions from the problem
def boat_speed_still_water (x : ℝ) : Prop := 
  let current_speed := 5
  let downstream_distance := 8.75
  let downstream_time := 21 / 60
  let downstream_speed := x + current_speed
  downstream_speed * downstream_time = downstream_distance

-- The theorem to prove
theorem boat_speed_is_20 : boat_speed_still_water 20 :=
by 
  unfold boat_speed_still_water
  sorry

end boat_speed_is_20_l353_353903


namespace triangle_count_l353_353234

def is_positive_area_triangle (p1 p2 p3 : ℤ × ℤ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  det2 (matrix (fin 2) (fin 2) (λ i j, if j = 0 then (if i = 0 then x2 - x1 else x3 - x1) else (if i = 0 then y2 - y1 else y3 - y1))) ≠ 0

def points : set (ℤ × ℤ) :=
  {p | 1 ≤ p.1 ∧ p.1 ≤ 4 ∧ 1 ≤ p.2 ∧ p.2 ≤ 4}

def count_triangles_with_positive_area (s : set (ℤ × ℤ)) : ℕ :=
  (finset.card ((finset.powerset_len 3 s.to_finset).filter (λ t, ∃ p1 p2 p3 ∈ t, is_positive_area_triangle p1 p2 p3))).toNat

theorem triangle_count :
  count_triangles_with_positive_area points = 508 :=
sorry

end triangle_count_l353_353234


namespace largest_integer_less_than_100_with_remainder_5_l353_353623

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353623


namespace ratio_of_kids_going_to_soccer_camp_to_total_kids_in_camp_l353_353002

def total_kids_in_camp : ℕ := 2000
def kids_afternoon_soccer_camp : ℕ := 750
def morning_fraction_of_soccer_camp : ℚ := 1 / 4
def afternoon_fraction_of_soccer_camp : ℚ := 3 / 4

theorem ratio_of_kids_going_to_soccer_camp_to_total_kids_in_camp 
  (total_kids_in_camp = 2000)
  (kids_afternoon_soccer_camp = 750)
  (morning_fraction_of_soccer_camp = 1 / 4)
  (afternoon_fraction_of_soccer_camp = 3 / 4) :
  let total_kids_soccer := kids_afternoon_soccer_camp / afternoon_fraction_of_soccer_camp in
  let ratio := total_kids_soccer / total_kids_in_camp in
  ratio = 1 / 2 := by
  sorry

end ratio_of_kids_going_to_soccer_camp_to_total_kids_in_camp_l353_353002


namespace optimal_constant_for_triangle_l353_353547

-- Conditions
variables {a b c : ℝ} (S : ℝ)

-- Proof Statement
theorem optimal_constant_for_triangle (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (area : S = Math.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
  S / (a * b + b * c + c * a) ≤ 1 / (4 * Real.sqrt 3) :=
sorry

end optimal_constant_for_triangle_l353_353547


namespace largest_integer_less_than_hundred_with_remainder_five_l353_353588

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l353_353588


namespace billiard_ball_returns_l353_353998

theorem billiard_ball_returns
  (w h : ℕ)
  (launch_angle : ℝ)
  (reflect_angle : ℝ)
  (start_A : ℝ × ℝ)
  (h_w : w = 2021)
  (h_h : h = 4300)
  (h_launch : launch_angle = 45)
  (h_reflect : reflect_angle = 45)
  (h_in_rect : ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 2021 ∧ 0 ≤ y ∧ y ≤ 4300) :
  ∃ (bounces : ℕ), bounces = 294 :=
by
  sorry

end billiard_ball_returns_l353_353998


namespace n_is_square_l353_353763

theorem n_is_square (n m : ℕ) (h1 : 3 ≤ n) (h2 : m = (n * (n - 1)) / 2) (h3 : ∃ (cards : Finset ℕ), 
  (cards.card = n) ∧ (∀ i ∈ cards, i ∈ Finset.range (m + 1)) ∧ 
  (∀ (i j : ℕ) (hi : i ∈ cards) (hj : j ∈ cards), i ≠ j → 
    ((i + j) % m) ≠ ((i + j) % m))) : 
  ∃ k : ℕ, n = k * k := 
sorry

end n_is_square_l353_353763


namespace binom_two_eq_n_choose_2_l353_353927

theorem binom_two_eq_n_choose_2 (n : ℕ) (h : n ≥ 2) :
  (nat.choose n 2) = (n * (n - 1)) / 2 := by
  sorry

end binom_two_eq_n_choose_2_l353_353927


namespace sum_of_written_numbers_l353_353992

theorem sum_of_written_numbers (n : ℕ) : let final_sum := ∑ k in finset.range (2^n), (2^k) in
  final_sum ≥ n * 2^n := 
  sorry

end sum_of_written_numbers_l353_353992


namespace total_volume_formula_l353_353766

-- Define the sequence of radii
def radii : ℕ → ℝ
| 0       := r
| (n + 1) := radii n / real.sqrt 3

-- Define the volume of each sphere
def volumes (n : ℕ) : ℝ := (4 / 3) * real.pi * (radii n)^3

-- Define the total volume
def total_volume : ℝ := ∑' n, volumes n

-- The theorem to be proven
theorem total_volume_formula (r : ℝ) : total_volume = 2 * real.pi * r^3 :=
by
  sorry

end total_volume_formula_l353_353766


namespace prime_arithmetic_progression_difference_divisible_by_6_l353_353752

theorem prime_arithmetic_progression_difference_divisible_by_6
    (p d : ℕ) (h₀ : Prime p) (h₁ : Prime (p - d)) (h₂ : Prime (p + d))
    (p_neq_3 : p ≠ 3) :
    ∃ (k : ℕ), d = 6 * k := by
  sorry

end prime_arithmetic_progression_difference_divisible_by_6_l353_353752


namespace largest_integer_less_than_100_with_remainder_5_l353_353631

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l353_353631


namespace true_false_count_l353_353381

theorem true_false_count :
  ∃ (T F M : ℕ), 
  M = 2 * F ∧ 
  F = T + 7 ∧ 
  T + F + M = 45 ∧ 
  T = 6 :=
by {
  let T := 6,
  let F := T + 7,
  let M := 2 * F,
  have eq1 : M = 2 * F := by simp [M],
  have eq2 : F = T + 7 := by simp [F],
  have eq3 : T + F + M = 45 := by 
    simp [T, F, M],
    norm_num,
  exact ⟨T, F, M, eq1, eq2, eq3, rfl⟩
}

end true_false_count_l353_353381


namespace largest_integer_lt_100_with_remainder_5_div_8_l353_353600

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l353_353600


namespace sin_3theta_l353_353856

theorem sin_3theta (θ : ℝ) :
  sin (3 * θ) = 4 * sin θ * sin (π / 3 + θ) * sin (2 * π / 3 + θ) :=
by
  have h1 : sin (π / 3 + θ) = (sqrt 3 / 2) * cos θ + (1 / 2) * sin θ := sorry
  have h2 : sin (2 * π / 3 + θ) = (sqrt 3 / 2) * cos θ - (1 / 2) * sin θ := sorry
  sorry

end sin_3theta_l353_353856


namespace probability_factor_90_less_than_10_l353_353950

-- Definitions from conditions
def number_factors_90 : ℕ := 12
def factors_90_less_than_10 : ℕ := 6

-- The corresponding proof problem
theorem probability_factor_90_less_than_10 : 
  (factors_90_less_than_10 / number_factors_90 : ℚ) = 1 / 2 :=
by
  sorry  -- proof to be filled in

end probability_factor_90_less_than_10_l353_353950


namespace software_price_l353_353982

theorem software_price (copies total_revenue : ℝ) (P : ℝ) 
  (h1 : copies = 1200)
  (h2 : 0.5 * copies * P + 0.6 * (2 / 3) * (copies - 0.5 * copies) * P + 0.25 * (copies - 0.5 * copies - (2 / 3) * (copies - 0.5 * copies)) * P = total_revenue)
  (h3 : total_revenue = 72000) :
  P = 80.90 :=
by
  sorry

end software_price_l353_353982


namespace coby_travel_time_l353_353118

theorem coby_travel_time :
  let d1 := 640
  let d2 := 400
  let d3 := 250
  let d4 := 380
  let s1 := 80
  let s2 := 65
  let s3 := 75
  let s4 := 50
  let time1 := d1 / s1
  let time2 := d2 / s2
  let time3 := d3 / s3
  let time4 := d4 / s4
  let total_time := time1 + time2 + time3 + time4
  total_time = 25.08 :=
by
  sorry

end coby_travel_time_l353_353118


namespace altitudes_location_property_l353_353879

noncomputable def altitude_location (T : Triangle) : Set Location :=
  if T.is_acute then {Inside}
  else if T.is_right then {Coinside, Inside}
  else if T.is_obtuse then {Outside, Inside}
  else ∅

theorem altitudes_location_property (T : Triangle) :
  altitude_location T = {Inside, Outside, Coinciding} := sorry

end altitudes_location_property_l353_353879


namespace num_integers_between_sqrt10_sqrt100_l353_353710

theorem num_integers_between_sqrt10_sqrt100 : 
  ∃ n : ℕ, n = 7 ∧ ∀ x : ℤ, (⌊Real.sqrt 10⌋ + 1 <= x) ∧ (x <= ⌈Real.sqrt 100⌉ - 1) ↔ (4 <= x ∧ x <= 10) := 
by 
  sorry

end num_integers_between_sqrt10_sqrt100_l353_353710


namespace determine_pump_x_time_l353_353431

noncomputable def pump_x_rate (X : ℝ) : Prop :=
  X > 0

noncomputable def pump_y_rate (Y : ℝ) : Prop :=
  Y > 0 ∧ Y = 1 / 48

noncomputable def combined_rate (X Y : ℝ) : Prop :=
  X - Y = (1 / 6) * X

noncomputable def time_pump_x_alone (T_X : ℝ) : Prop :=
  T_X = 1 / X

theorem determine_pump_x_time (T_X X Y : ℝ)
  (hx : pump_x_rate X)
  (hy : pump_y_rate Y)
  (hc : combined_rate X Y)
  : time_pump_x_alone T_X :=
by
  sorry

end determine_pump_x_time_l353_353431


namespace find_positive_t_l353_353644

theorem find_positive_t (t : ℝ) (ht : 0 < t) : complex.abs (9 + complex.I * t) = 15 ↔ t = 12 := by
  sorry

end find_positive_t_l353_353644


namespace largest_integer_less_than_hundred_with_remainder_five_l353_353592

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l353_353592


namespace sqrt_expression_value_l353_353126

theorem sqrt_expression_value :
  (sqrt ((2 - sin^2 (π / 8)) * (2 - sin^2 (2 * π / 8)) * (2 - sin^2 (3 * π / 8))) = (sqrt 17) / 2) :=
by
  -- Root conditions
  have h1 : polynomial.eval (sin^2 (π / 8)) (polynomial.C 4 * polynomial.X^4 - polynomial.C 8 * polynomial.X^3 + polynomial.C 5 * polynomial.X^2 - polynomial.C 1 * polynomial.X) = 0 := sorry
  have h2 : polynomial.eval (sin^2 (2 * π / 8)) (polynomial.C 4 * polynomial.X^4 - polynomial.C 8 * polynomial.X^3 + polynomial.C 5 * polynomial.X^2 - polynomial.C 1 * polynomial.X) = 0 := sorry
  have h3 : polynomial.eval (sin^2 (3 * π / 8)) (polynomial.C 4 * polynomial.X^4 - polynomial.C 8 * polynomial.X^3 + polynomial.C 5 * polynomial.X^2 - polynomial.C 1 * polynomial.X) = 0 := sorry
  -- Therefore statement
  sorry

end sqrt_expression_value_l353_353126


namespace find_BD_when_AC_over_AB_min_l353_353327

-- Definition of the conditions
variables (A B C D : Type) [metric_space A]
variables (triangle : set A)
variables (on_side_BC : D ∈ segment B C)
variables (angle_ADB : ∠ADB = 120)
variables (AD_len : dist A D = 2)
variables (CD_2BD : dist C D = 2 * dist B D)

noncomputable def minimum_ratio_condition (BD : ℝ) : Prop := 
  let x := dist B D
  let CD := 2 * x
  let AD := 2 in
  let angle_ADB_cos := -1/2 in
  let b_squared := AD^2 + CD^2 - 2 * AD * CD * angle_ADB_cos in
  let c_squared := AD^2 + x^2 + 2 * AD * x * angle_ADB_cos in
  b_squared / c_squared = 4 - (12 / (x + 1 + 3/(x + 1))) → x = sqrt 3 - 1

theorem find_BD_when_AC_over_AB_min (BD : ℝ) : minimum_ratio_condition A B C D 
:= sorry

end find_BD_when_AC_over_AB_min_l353_353327


namespace factorial_coefficient_is_integer_l353_353188

theorem factorial_coefficient_is_integer (p : Polynomial ℝ) (n : ℕ)
  (hp_deg : p.degree = n)
  (hp_int : ∀ m : ℤ, p.eval m ∈ ℤ) :
  ∀ k : ℝ, k ∈ p.coeffs → (n.factorial : ℝ) * k ∈ ℤ :=
by
  sorry

end factorial_coefficient_is_integer_l353_353188


namespace largest_integer_less_than_100_with_remainder_5_l353_353624

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353624


namespace min_distance_l353_353290

def triangle_ABC (A B C D : Type) (BD : ℝ) :=
∃ (CD : ℝ), CD = 2 * BD ∧ 
  ∃ (AD AC AB : ℝ), AD = 2 ∧ 
    ∃ (α β γ : ℝ), α = 120 ∧ 
      angle_ADB = α ∧ 
        (AC / AB = (sqrt (3) - 1))

theorem min_distance (A B C D : Type) (BD : ℝ) :
  ∃ D ∈ BC, ∃ α β γ AD AC AB, 
    α = 120 ∧ AD = 2 ∧ D ∈ BC ∧ 
      γ = ∠ADB = α ∧ 
        CD = 2 * BD →
          (AC / AB) = (sqrt (3) - 1) := 
sorry

end min_distance_l353_353290


namespace count_integers_between_sqrt_10_and_sqrt_100_l353_353720

theorem count_integers_between_sqrt_10_and_sqrt_100 :
  ∃ n : ℕ, (∀ k : ℕ, 4 ≤ k ∧ k ≤ 10 → k ∈ set.Ico (int.floor (sqrt (10 : ℝ))) (int.ceil (sqrt (100 : ℝ)))) ∧ n = 7 :=
sorry

end count_integers_between_sqrt_10_and_sqrt_100_l353_353720


namespace poly_divisible_coeff_sum_eq_one_l353_353526

theorem poly_divisible_coeff_sum_eq_one (C D : ℂ) :
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^100 + C * x^2 + D * x + 1 = 0) →
  C + D = 1 :=
by
  sorry

end poly_divisible_coeff_sum_eq_one_l353_353526


namespace binom_eq_fraction_l353_353924

open Nat

theorem binom_eq_fraction (n : ℕ) (h_pos : 0 < n) : choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_eq_fraction_l353_353924


namespace log_base4_of_8_l353_353139

theorem log_base4_of_8 : log 4 8 = 3 / 2 :=
by
  have h1 : 8 = 2 ^ 3 := by norm_num
  have h2 : 4 = 2 ^ 2 := by norm_num
  rw [h2] -- replace 4 with (2^2)
  rw [←@Real.log_pow 4 (3/2)]
  norm_num
  rw [Real.log_self, mul_one]
  norm_num

end log_base4_of_8_l353_353139


namespace count_integers_between_sqrt10_sqrt100_l353_353725

theorem count_integers_between_sqrt10_sqrt100 :
  ∃ n : ℕ, n = 7 ∧ card {x : ℤ | real.sqrt 10 < x ∧ x < real.sqrt 100} = n :=
by
  sorry

end count_integers_between_sqrt10_sqrt100_l353_353725


namespace puzzle_solution_l353_353051

theorem puzzle_solution :
  (∀ n m k : ℕ, n + m + k = 111 → 9 * (n + m + k) / 3 = 9) ∧
  (∀ n m k : ℕ, n + m + k = 444 → 12 * (n + m + k) / 12 = 12) ∧
  (∀ n m k : ℕ, n + m + k = 777 → (7 * 3 ≠ 15 → (7 * 3 - 6 = 15)) ) →
  ∀ n m k : ℕ, n + m + k = 888 → 8 * (n + m + k / 3) - 6 = 18 :=
by
  intros h n m k h1
  sorry

end puzzle_solution_l353_353051


namespace largest_integer_less_than_100_with_remainder_5_l353_353608

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l353_353608


namespace percent_difference_l353_353964

theorem percent_difference : 
  let a := 60 in
  let b := 45 in
  let c := 40 in
  let d := 35 in
  (b * a / 100 - d * c / 100) = 13 :=
by
  let a := 60 -- 60
  let b := 45 -- 45
  let c := 40 -- 40
  let d := 35 -- 35
  have step1 : b * a / 100 = 27 := by norm_num [b, a]
  have step2 : d * c / 100 = 14 := by norm_num [d, c]
  calc 
    (b * a / 100 - d * c / 100) = 27 - 14 := by rw [step1, step2]
    ... = 13 := by norm_num

end percent_difference_l353_353964


namespace probability_one_boy_one_girl_l353_353894

theorem probability_one_boy_one_girl (boys girls : ℕ) (total_people select_people : ℕ) :
  boys = 2 → girls = 2 → total_people = boys + girls → select_people = 2 →
  (let total_ways := (nat.choose total_people select_people),
       favorable_ways := (nat.choose boys 1) * (nat.choose girls 1)
   in (favorable_ways : ℚ) / (total_ways : ℚ) = 2 / 3) :=
by intros; simp; sorry

end probability_one_boy_one_girl_l353_353894


namespace students_taking_both_l353_353068

def students := 500
def music_students := 30
def art_students := 10
def neither_students := 470

theorem students_taking_both :
  ∃ x, students - neither_students = music_students + art_students - x ∧ x = 10 :=
by
  use 10
  split
  sorry

end students_taking_both_l353_353068


namespace factorize_expression_l353_353147

theorem factorize_expression (x : ℝ) : 2 * x - x^2 = x * (2 - x) := sorry

end factorize_expression_l353_353147


namespace faucet_fill_time_l353_353647

theorem faucet_fill_time (r : ℝ) (T1 T2 t : ℝ) (F1 F2 : ℕ) (h1 : T1 = 200) (h2 : t = 8) (h3 : F1 = 4) (h4 : F2 = 8) (h5 : T2 = 50) (h6 : r * F1 * t = T1) : 
(F2 * r) * t / (F1 * F2) = T2 -> by sorry := sorry

#check faucet_fill_time

end faucet_fill_time_l353_353647


namespace cos_sq_plus_sin_double_l353_353273

-- Definitions and conditions
def α_terminal_side_passes_through_P (α : ℝ) :=
  let x := 2 in
  let y := 1 in
  let r := Real.sqrt (x^2 + y^2) in
  (Real.cos α = x / r) ∧ (Real.sin α = y / r)

-- Main theorem statement
theorem cos_sq_plus_sin_double (α : ℝ) (h : α_terminal_side_passes_through_P α) :
  (Real.cos α)^2 + Real.sin (2 * α) = 8 / 5 :=
sorry

end cos_sq_plus_sin_double_l353_353273


namespace number_mul_five_l353_353044

theorem number_mul_five (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 :=
by
  sorry

end number_mul_five_l353_353044


namespace arithmetic_sequence_general_formula_geometric_sequence_sum_first_n_terms_l353_353197

noncomputable def arithmetic_sequence (a n d : ℝ) : ℝ := 
  a + (n - 1) * d

noncomputable def geometric_sequence_sum (b1 r n : ℝ) : ℝ := 
  b1 * (1 - r^n) / (1 - r)

theorem arithmetic_sequence_general_formula (a1 d : ℝ) (h1 : a1 + 2 * d = 2) (h2 : 3 * a1 + 3 * d = 9 / 2) : 
  ∀ n, arithmetic_sequence a1 n d = (n + 1) / 2 :=
by 
  sorry

theorem geometric_sequence_sum_first_n_terms (a1 d b1 b4 : ℝ) (h1 : a1 + 2 * d = 2) (h2 : 3 * a1 + 3 * d = 9 / 2) 
  (h3 : b1 = a1) (h4 : b4 = arithmetic_sequence a1 15 d) (h5 : b4 = 8) :
  ∀ n, geometric_sequence_sum b1 2 n = 2^n - 1 :=
by 
  sorry

end arithmetic_sequence_general_formula_geometric_sequence_sum_first_n_terms_l353_353197


namespace shift_graph_to_obtain_target_l353_353012

-- Conditions
def initial_function (x : ℝ) : ℝ := Real.sin (Real.pi / 2 + 2 * x)
def target_function (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3)
def shifted_function (x : ℝ) : ℝ := Real.cos (2 * (x - Real.pi / 6))

-- Requirement: Prove that shifting the initial function by π/6 units to the right results in the target function
theorem shift_graph_to_obtain_target :
  (∀ x, initial_function (x - Real.pi / 6) = target_function x) ↔ 
  (shifted_function x = target_function x) :=
begin
  sorry
end

end shift_graph_to_obtain_target_l353_353012


namespace simplify_and_evaluate_l353_353373

theorem simplify_and_evaluate (x y : ℤ) (hx : x = -1) (hy : y = 2) : 
  x^2 - 2 * (3 * y^2 - x * y) + (y^2 - 2 * x * y) = -19 := 
by
  -- Proof will go here, but it's omitted as per instructions
  sorry

end simplify_and_evaluate_l353_353373


namespace decrypted_plaintext_l353_353909

theorem decrypted_plaintext (a b c d : ℕ) : 
  (a + 2 * b = 14) → (2 * b + c = 9) → (2 * c + 3 * d = 23) → (4 * d = 28) → 
  (a = 6 ∧ b = 4 ∧ c = 1 ∧ d = 7) :=
by 
  intros h1 h2 h3 h4
  -- Proof steps go here
  sorry

end decrypted_plaintext_l353_353909


namespace factorize_3m2_minus_12_l353_353148

theorem factorize_3m2_minus_12 (m : ℤ) : 
  3 * m^2 - 12 = 3 * (m - 2) * (m + 2) := 
sorry

end factorize_3m2_minus_12_l353_353148


namespace ceil_sqrt_200_eq_15_l353_353533

theorem ceil_sqrt_200_eq_15 (h1 : Real.sqrt 196 = 14) (h2 : Real.sqrt 225 = 15) (h3 : 196 < 200 ∧ 200 < 225) : 
  Real.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l353_353533


namespace hyperbola_eccentricity_l353_353659

-- Define the conditions
variables (a b : ℝ)
def hyperbola := ∀ x y : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1
def circle := ∀ x y : ℝ, (x^2) + (y - 2)^2 = 3
def asymptote := ∀ x y : ℝ, y = (b / a) * x

-- Main conjecture
theorem hyperbola_eccentricity (ha : a > 0) (hb : b > 0) (tangency_condition : ∀ x y : ℝ, ∃ (dist_cond : Real), 
    dist_cond = abs (-2 * a) / Real.sqrt (a^2 + b^2) ∧ dist_cond = Real.sqrt 3) : 
    ((Real.sqrt (a^2 + b^2)) / a = 2/ Real.sqrt 3) := 
by
  sorry

end hyperbola_eccentricity_l353_353659


namespace seventh_selected_is_320_l353_353473

variable (employees : List ℕ)
          (employees_size : ℕ)
          (num_to_select : ℕ)
          (random_numbers : List ℕ)
          (employees_list : employees = List.range' 1 employees_size.succ)
          (condition : employees_size = 500)
          (selection_order : List (List ℕ) := [
            [7816, 6572, 0802, 6319, 8702, 4369, 9728, 0198], 
            [3204, 9243, 4935, 8200, 3623, 4869, 6938, 7481]
          ])
          (start_row : ℕ)
          (start_column : ℕ)
          (answer : ℕ)

noncomputable def isSelected (n : ℕ) : Bool :=
  n ≤ employees_size

noncomputable def selectNthValid (random_nums : List ℕ) (n : ℕ) : ℕ :=
  (random_nums.filter isSelected).nth n |>.getD (random_nums.headD 0)

#eval selectNthValid [
  6572 % 1000, 0802 % 1000, 6319 % 1000, 8702 % 1000,
  4369 % 1000, 9728 % 1000, 0198 % 1000, 
  3204 % 1000, 9243 % 1000, 4935 % 1000, 
  8200 % 1000, 3623 % 1000, 4869 % 1000, 
  6938 % 1000, 7481 % 1000 
] 6

theorem seventh_selected_is_320 : selectNthValid [
  6572 % 1000, 0802 % 1000, 6319 % 1000, 8702 % 1000,
  4369 % 1000, 9728 % 1000, 0198 % 1000, 
  3204 % 1000, 9243 % 1000, 4935 % 1000, 
  8200 % 1000, 3623 % 1000, 4869 % 1000, 
  6938 % 1000, 7481 % 1000 
] 6 = 320 := 
  by
    sorry

end seventh_selected_is_320_l353_353473


namespace middle_term_in_arithmetic_sequence_l353_353023

theorem middle_term_in_arithmetic_sequence :
  let a := 3^2 in let c := 3^4 in
  ∃ z : ℤ, (2 * z = a + c) ∧ z = 45 := by
let a := 3^2
let c := 3^4
use (a + c) / 2
split
-- Prove that 2 * ((a + c) / 2) = a + c
sorry
-- Prove that (a + c) / 2 = 45
sorry

end middle_term_in_arithmetic_sequence_l353_353023


namespace jack_average_speed_l353_353790

-- Define the given conditions
def total_distance : ℝ := 8
def total_time : ℝ := 1.25
def speed_1 : ℝ := 4
def time_1 : ℝ := 0.5
def speed_2 : ℝ := 6
def time_2 : ℝ := 1 / 3
def speed_3 : ℝ := 3

-- The goal is to prove that the average speed is approximately 3.692 mph
def average_speed : ℝ := total_distance / total_time

theorem jack_average_speed : abs (average_speed - 3.692) < 0.001 :=
by
  sorry

end jack_average_speed_l353_353790


namespace extremum_values_l353_353668

def real_extremum (a b c d e : ℝ) :=
  3 * a + 2 * b - c + 4 * d + Real.sqrt 133 * e = Real.sqrt 133 ∧
  2 * a^2 + 3 * b^2 + 3 * c^2 + d^2 + 6 * e^2 = 60

theorem extremum_values (a b c d e : ℝ) (h : real_extremum a b c d e) :
  e ∈ Icc ((1 - Real.sqrt 19) / 2) ((1 + Real.sqrt 19) / 2) :=
sorry

end extremum_values_l353_353668


namespace number_of_mismatching_socks_l353_353385

-- Define the conditions
def total_socks : Nat := 25
def pairs_of_matching_socks : Nat := 4
def socks_per_pair : Nat := 2
def matching_socks : Nat := pairs_of_matching_socks * socks_per_pair

-- State the theorem
theorem number_of_mismatching_socks : total_socks - matching_socks = 17 :=
by
  -- Skip the proof
  sorry

end number_of_mismatching_socks_l353_353385


namespace old_edition_pages_l353_353082

-- Define the conditions
variables (new_edition : ℕ) (old_edition : ℕ)

-- The conditions given in the problem
axiom new_edition_pages : new_edition = 450
axiom pages_relationship : new_edition = 2 * old_edition - 230

-- Goal: Prove that the old edition Geometry book had 340 pages
theorem old_edition_pages : old_edition = 340 :=
by sorry

end old_edition_pages_l353_353082


namespace number_of_girls_l353_353485

theorem number_of_girls (d c : ℕ) (h1 : c = 2 * (d - 15)) (h2 : d - 15 = 5 * (c - 45)) : d = 40 := 
by
  sorry

end number_of_girls_l353_353485


namespace arithmetic_seq_of_pos_and_arithmetic_sqrt_S_l353_353811

variable {a : ℕ → ℝ} (h_pos : ∀ n, a n > 0)
variable (h_a2 : a 2 = 3 * a 1)
variable (S : ℕ → ℝ)
variable (h_S : ∀ n, S n = ∑ i in Finset.range(n+1), a i)
variable (h_sqrt_S_arith : ∃ d, ∀ n, (Sqrt.sqrt (S n) - Sqrt.sqrt (S (n - 1))) = d)

theorem arithmetic_seq_of_pos_and_arithmetic_sqrt_S : 
  ∀ n, a (n+1) - a n = a 1 := 
sorry

end arithmetic_seq_of_pos_and_arithmetic_sqrt_S_l353_353811


namespace second_interest_rate_l353_353861

def calculate_interest (principal : ℕ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal : ℝ) * rate * time / 100

theorem second_interest_rate (total_amount part1 part2 interest1 total_interest : ℝ) (time : ℝ) :
  total_amount = 2500 ∧
  part1 = 2000 ∧
  interest1 = (part1 * 5 * time) / 100 ∧
  total_interest = 130 ∧
  total_interest = interest1 + calculate_interest part2 r time ∧
  part2 = total_amount - part1 ∧
  time = 1 → 
  r = 6 := 
sorry

end second_interest_rate_l353_353861


namespace find_f_neg_a_l353_353873

def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

theorem find_f_neg_a (a : ℝ) (h : f a = 11) : f (-a) = -9 := 
by
  sorry

end find_f_neg_a_l353_353873


namespace doubled_volume_l353_353478

theorem doubled_volume (V : ℕ) (h : V = 4) : 8 * V = 32 := by
  sorry

end doubled_volume_l353_353478


namespace probability_two_languages_example_l353_353844

noncomputable def probability_two_languages (total : ℕ) (french : ℕ) (spanish : ℕ) (german : ℕ) 
(french_spanish : ℕ) (spanish_german : ℕ) (german_french : ℕ) (all_three : ℕ) : ℚ :=
  let only_french := french - (french_spanish + german_french - all_three) - all_three in
  let only_spanish := spanish - (french_spanish + spanish_german - all_three) - all_three in
  let only_german := german - (spanish_german + german_french - all_three) - all_three in
  let only_fs := french_spanish - all_three in
  let only_sg := spanish_german - all_three in
  let only_gf := german_french - all_three in
  let total_pairs := (total * (total - 1)) / 2 in
  let not_two_languages := (only_french * (only_french - 1)) / 2 
                          + (only_spanish * (only_spanish - 1)) / 2 
                          + (only_german * (only_german - 1)) / 2 in
  1 - (not_two_languages : ℚ) / total_pairs

theorem probability_two_languages_example :
  probability_two_languages 30 20 23 12 8 5 4 3 = 359 / 435 :=
  sorry

end probability_two_languages_example_l353_353844


namespace big_sale_commission_l353_353842

theorem big_sale_commission (avg_increase : ℝ) (new_avg : ℝ) (num_sales : ℕ) 
  (prev_avg := new_avg - avg_increase)
  (total_prev := prev_avg * (num_sales - 1))
  (total_new := new_avg * num_sales)
  (C := total_new - total_prev) :
  avg_increase = 150 → new_avg = 250 → num_sales = 6 → C = 1000 :=
by
  intros 
  sorry

end big_sale_commission_l353_353842


namespace intersection_point_min_ratio_l353_353272

noncomputable def curve_C1 (rho theta : ℝ) := rho * Math.cos (theta + π / 4) = sqrt 2 / 2
noncomputable def curve_C2 (rho theta : ℝ) := (rho = 1) ∧ (0 ≤ theta ∧ theta ≤ π)
noncomputable def curve_C3 (rho theta : ℝ) := 1 / (rho ^ 2) = (Math.cos theta) ^ 2 / 3 + (Math.sin theta) ^ 2
noncomputable def point_M := (1, 0)

theorem intersection_point :
  ∃ theta : ℝ, (curve_C1 (1:ℝ) theta) ∧ (curve_C2 (1:ℝ) theta) ↔ point_M = (1, 0) := 
  sorry

theorem min_ratio :
  ∃ (t1 t2 α : ℝ),
  let MA := abs t1, MB := abs t2,
  let AB := abs (t1 - t2),
  let ratio := (MA * MB) / AB,
  curve_C3 (1:ℝ) α ∧ 0 ≤ α ∧ α ≤ π → 
  ratio = sqrt 6 / 6 :=
  sorry

end intersection_point_min_ratio_l353_353272


namespace problem_l353_353239

theorem problem (n : ℝ) (h : n + 1 / n = 10) : n ^ 2 + 1 / n ^ 2 + 5 = 103 :=
by sorry

end problem_l353_353239


namespace common_ratio_value_l353_353780

variable (a : ℕ → ℝ) -- defining the geometric sequence as a function ℕ → ℝ
variable (q : ℝ) -- defining the common ratio

-- conditions from the problem
def geo_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

axiom h1 : geo_seq a q
axiom h2 : a 2020 = 8 * a 2017

-- main statement to be proved
theorem common_ratio_value : q = 2 :=
sorry

end common_ratio_value_l353_353780


namespace min_value_S_l353_353410

theorem min_value_S (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → 0 ≤ x i) 
  (h2 : ∑ i in finset.range n, x (i + 1) = 1) : 
  ∃ x : ℕ → ℝ, 
    S = 1 - 2^(-(1 : ℝ) / n) ∧ 
    S = max (λ k, x k / (1 + (∑ i in finset.range k, x (i + 1)))) :=
sorry

end min_value_S_l353_353410


namespace ceil_sqrt_200_eq_15_l353_353540

theorem ceil_sqrt_200_eq_15 : Int.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l353_353540


namespace largest_integer_less_than_hundred_with_remainder_five_l353_353591

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l353_353591


namespace train_usual_time_l353_353094

-- Define the conditions
def usual_speed (S : ℝ) := S
def usual_time (T : ℝ) := T
def new_speed (S : ℝ) := (6 / 7) * S
def new_time (T : ℝ) := (7 / 6) * T
def time_late := 15 / 60 -- 15 minutes in hours

-- Prove that usual_time is 3/2 given the conditions
theorem train_usual_time : ∀ (S T : ℝ), 
  new_time T = T + time_late → usual_time T = 3 / 2 :=
by
  intros S T
  intro h
  change (7 / 6) * T = T + 1 / 4 at h
  sorry

end train_usual_time_l353_353094


namespace hanna_gives_roses_l353_353232

-- Conditions as Lean definitions
def initial_budget : ℕ := 300
def price_jenna : ℕ := 2
def price_imma : ℕ := 3
def price_ravi : ℕ := 4
def price_leila : ℕ := 5

def roses_for_jenna (budget : ℕ) : ℕ :=
  budget / price_jenna * 1 / 3

def roses_for_imma (budget : ℕ) : ℕ :=
  budget / price_imma * 1 / 4

def roses_for_ravi (budget : ℕ) : ℕ :=
  budget / price_ravi * 1 / 6

def roses_for_leila (budget : ℕ) : ℕ :=
  budget / price_leila

-- Calculations based on conditions
def roses_jenna : ℕ := Nat.floor (50 * 1/3)
def roses_imma : ℕ := Nat.floor ((100 / price_imma) * 1 / 4)
def roses_ravi : ℕ := Nat.floor ((50 / price_ravi) * 1 / 6)
def roses_leila : ℕ := 50 / price_leila

-- Final statement to be proven
theorem hanna_gives_roses :
  roses_jenna + roses_imma + roses_ravi + roses_leila = 36 := by
  sorry

end hanna_gives_roses_l353_353232


namespace sum_of_primes_dividing_expr_l353_353904

-- Define the expression
def expr := 2^10 - 1

-- Define the prime set that divides the expression
def prime_set := {3, 11, 31}

-- The target sum of the primes
def prime_sum := 45

-- Proven statement
theorem sum_of_primes_dividing_expr :
  (∃ p ∈ prime_set, Nat.Prime p ∧ p ∣ expr) ∧ (prime_set.sum = prime_sum) :=
by
  sorry

end sum_of_primes_dividing_expr_l353_353904


namespace find_BD_when_AC_over_AB_min_l353_353328

-- Definition of the conditions
variables (A B C D : Type) [metric_space A]
variables (triangle : set A)
variables (on_side_BC : D ∈ segment B C)
variables (angle_ADB : ∠ADB = 120)
variables (AD_len : dist A D = 2)
variables (CD_2BD : dist C D = 2 * dist B D)

noncomputable def minimum_ratio_condition (BD : ℝ) : Prop := 
  let x := dist B D
  let CD := 2 * x
  let AD := 2 in
  let angle_ADB_cos := -1/2 in
  let b_squared := AD^2 + CD^2 - 2 * AD * CD * angle_ADB_cos in
  let c_squared := AD^2 + x^2 + 2 * AD * x * angle_ADB_cos in
  b_squared / c_squared = 4 - (12 / (x + 1 + 3/(x + 1))) → x = sqrt 3 - 1

theorem find_BD_when_AC_over_AB_min (BD : ℝ) : minimum_ratio_condition A B C D 
:= sorry

end find_BD_when_AC_over_AB_min_l353_353328


namespace product_is_2008th_power_l353_353419

theorem product_is_2008th_power (a b c : ℕ) (h1 : a = (b + c) / 2) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : a ≠ b) :
  ∃ k : ℕ, (a * b * c) = k^2008 :=
by
  sorry

end product_is_2008th_power_l353_353419


namespace max_of_expression_l353_353170

theorem max_of_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (abcd (a + b + c + d) / ((a + b)^2 * (c + d)^2)) ≤ 1 / 4 :=
sorry

end max_of_expression_l353_353170


namespace part1_solution_part2_solution_l353_353693

-- Define the inequality
def inequality (m x : ℝ) : Prop := (m - 1) * x ^ 2 + (m - 1) * x + 2 > 0

-- Part (1): Prove the solution set for m = 0 is (-2, 1)
theorem part1_solution :
  (∀ x : ℝ, inequality 0 x → (-2 : ℝ) < x ∧ x < 1) := 
by
  sorry

-- Part (2): Prove the range of values for m such that the solution set is R
theorem part2_solution (m : ℝ) :
  (∀ x : ℝ, inequality m x) ↔ (1 ≤ m ∧ m < 9) := 
by
  sorry

end part1_solution_part2_solution_l353_353693


namespace problem1_problem2_l353_353515

def f (a b c x : ℝ) : ℝ := (a * x^2 - 2) / (b * x + c)

theorem problem1 (a b c : ℤ) (h_odd : ∀ x : ℝ, f a b c x + f a b c (-x) = 0)
  (h1 : f a b c 1 = 1) (h2 : f a b c 2 - 4 > 0) : f a b c = f 3 (a - 2) 0 := by
  sorry

theorem problem2 (a : ℤ) (h : 1 ≤ a) (hc : ∀ x : ℝ, 1 < x → (a * x^2 - 2) / x > 1) :
  a ≥ 3 := by
  sorry

end problem1_problem2_l353_353515


namespace inequality_solution_l353_353152

theorem inequality_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, x + a * f y ≤ y + f (f x)) → (a < 0 ∨ a = 1) :=
begin
  sorry
end

end inequality_solution_l353_353152


namespace square_intersection_on_angle_bisector_l353_353850

-- Given data is a triangle ABC, and squares are constructed on sides AB and AC external to ABC.
variables {A B C D E F G : Type} [AffineSegment A B] [AffineSegment A C] [AffineSquare A B D E] [AffineSquare A C F G]

-- Definition that states the sides of the squares opposite to AB and AC intersect.
noncomputable def intersection_on_angle_bisector (A' B' C': Type) [FootPerpendicular A' A C] [FootPerpendicular A' A B] :
  Prop :=
  let X := foot A' A C in
  let Y := foot A' A B in
  
  -- This defines A' to be a point on the angle bisector of ∠BAC
  is_on_angle_bisector (angle A B C) A' 

-- Statement of the problem in Lean: We need to prove that the intersection point lies on the angle bisector.
theorem square_intersection_on_angle_bisector
    (A B C D E F G : Type)
    [AffineSegment A B] [AffineSegment A C] [AffineSquare A B D E] [AffineSquare A C F G]
    (A' : Type) [FootPerpendicular A' A B] [FootPerpendicular A' A C]
    (is_intersection : intersection A' D A' G) :
    intersection_on_angle_bisector A' B' C' :=
by
  -- Proof logic goes here
  sorry

end square_intersection_on_angle_bisector_l353_353850


namespace arithmetic_sequence_middle_term_l353_353024

theorem arithmetic_sequence_middle_term :
  ∀ (a b : ℕ) (z : ℕ), a = 9 → b = 81 → z = (a + b) / 2 → z = 45 :=
by
  intros a b z h_a h_b h_z
  rw [h_a, h_b] at h_z
  exact h_z

end arithmetic_sequence_middle_term_l353_353024


namespace log_sqrt3_sixth_root_a_eq_l353_353150

noncomputable def log_sqrt3_sixth_root_a (a b : ℝ) (hb : log a 27 = b) : ℝ :=
  log (sqrt 3) (a ^ (1 / 6))

theorem log_sqrt3_sixth_root_a_eq (a b : ℝ) (hb : log a 27 = b) : 
  log_sqrt3_sixth_root_a a b hb = 1 / b :=
by sorry

end log_sqrt3_sixth_root_a_eq_l353_353150


namespace largest_prime_of_form_2n_minus_1_5000_l353_353128

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def form_2n_minus_1 (n : ℕ) : ℕ := 2^n - 1

def largest_prime_of_form_2n_minus_1_under (limit : ℕ) : ℕ :=
  if 127 < limit ∧ is_prime 127 ∧ ∀ n : ℕ, n > 1 ∧ is_prime n ∧ form_2n_minus_1 n < limit → form_2n_minus_1 n ≤ 127 then 127 else 0

theorem largest_prime_of_form_2n_minus_1_5000 :
  largest_prime_of_form_2n_minus_1_under 5000 = 127 := 
by 
  sorry

end largest_prime_of_form_2n_minus_1_5000_l353_353128


namespace count_5_primable_l353_353085

-- Define what it means for a number to be n-primable
def is_n_primable (n : ℕ) (p : ℕ) : Prop :=
  (p % n = 0) ∧
  (∀ d in p.digits, d ∈ [2, 3, 5, 7] ∨ (d < 18 ∧ ∃ q, q ∈ [2, 3, 5, 7] ∧ p.digits_adjacent_prime q))

-- Define the specific condition for 5-primable
def is_5_primable (p : ℕ): Prop := is_n_primable 5 p

-- Define criteria for prime digits and two-digit prime numbers.
def prime_digit (d : ℕ) : Prop := d ∈ [2, 3, 5, 7]
def valid_two_digit_prime (d : ℕ) : Prop := d < 18 ∧ (d = 11)

-- Check if p forms valid two-digit primes with adjacent digits
def digits_adjacent_prime (p q : ℕ) : Prop := p * 10 + q = 11

-- Main theorem to prove the count of 5-primable numbers that are four digits or fewer
theorem count_5_primable :
  (∃ fp : Finset ℕ, (∀ x ∈ fp, is_5_primable x) ∧
  fp.card = 40 ∧
  (∀ x ∈ fp, x < 10000)) :=
sorry

end count_5_primable_l353_353085


namespace scientific_notation_of_million_l353_353361

theorem scientific_notation_of_million : 1000000 = 10^6 :=
by
  sorry

end scientific_notation_of_million_l353_353361


namespace vector_cross_product_scaling_l353_353236

-- Given vectors a and b, and the condition for their cross product
variables (a b : ℝ^3)
variable (h : a × b = ⟨5, 4, -7⟩)

-- We want to prove that a × (3 • b) = ⟨15, 12, -21⟩
theorem vector_cross_product_scaling : a × (3 • b) = ⟨15, 12, -21⟩ :=
by
  sorry

end vector_cross_product_scaling_l353_353236


namespace largest_integer_less_than_hundred_with_remainder_five_l353_353584

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l353_353584


namespace largest_integer_lt_100_with_remainder_5_div_8_l353_353597

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l353_353597


namespace largest_int_with_remainder_5_lt_100_l353_353566

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l353_353566


namespace orange_segments_l353_353387

noncomputable def total_segments (H S B : ℕ) : ℕ :=
  H + S + B

theorem orange_segments
  (H S B : ℕ)
  (h1 : H = 2 * S)
  (h2 : S = B / 5)
  (h3 : B = S + 8) :
  total_segments H S B = 16 := by
  -- proof goes here
  sorry

end orange_segments_l353_353387


namespace Leila_weekly_earnings_l353_353018

-- Definitions based on the conditions
def Voltaire_daily_viewers := 50
def Leila_daily_viewers := 2 * Voltaire_daily_viewers
def earnings_per_view := 0.50
def days_in_week := 7

-- Statement to prove Leila's weekly earnings
theorem Leila_weekly_earnings :
  Leila_daily_viewers * earnings_per_view * days_in_week = 350 := by
  sorry

end Leila_weekly_earnings_l353_353018


namespace product_of_intersection_coordinates_l353_353129

noncomputable def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 5)^2 = 1
noncomputable def circle2 (x y : ℝ) : Prop := (x - 5)^2 + (y - 5)^2 = 4

theorem product_of_intersection_coordinates :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y ∧ x * y = 15 :=
by
  sorry

end product_of_intersection_coordinates_l353_353129


namespace max_value_of_abs_z_plus_4_l353_353746

open Complex
noncomputable def max_abs_z_plus_4 {z : ℂ} (h : abs (z + 3 * I) = 5) : ℝ :=
sorry

theorem max_value_of_abs_z_plus_4 (z : ℂ) (h : abs (z + 3 * I) = 5) : abs (z + 4) ≤ 10 :=
sorry

end max_value_of_abs_z_plus_4_l353_353746


namespace region_area_of_decreasing_distance_l353_353828

theorem region_area_of_decreasing_distance
  (A B C : ℝ × ℝ)
  (hAB : dist A B = 1)
  (hC : ∀ C, dist C B < dist A B):
  ↑(real.pi) / 4 := by
  sorry

end region_area_of_decreasing_distance_l353_353828


namespace sets_equal_l353_353243

-- Define sets M and N based on the provided conditions
def M : Set ℝ := {α | ∃ m : ℤ, α = sin ((5 * m - 9) * π / 3)}
def N : Set ℝ := {β | ∃ n : ℤ, β = cos (5 * (9 - 2 * n) * π / 6)}

-- State the theorem to prove M = N
theorem sets_equal : M = N :=
by sorry

end sets_equal_l353_353243


namespace largest_integer_less_than_100_with_remainder_5_l353_353626

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353626


namespace sum_div_l353_353338

variables {n : ℕ} {a : ℕ → ℝ}
def Sn (n : ℕ) (a : ℕ → ℝ) : ℝ := (n / 2) * (a 1 + a n)

theorem sum_div (h : (a 5) / (a 3) = 5 / 9) : Sn 9 a / Sn 5 a = 1 :=
sorry

end sum_div_l353_353338


namespace slope_of_line_determined_by_solutions_l353_353951

theorem slope_of_line_determined_by_solutions (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : 3 / x₁ +  4 / y₁ = 0)
  (h₂ : 3 / x₂ + 4 / y₂ = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -4 / 3 :=
sorry

end slope_of_line_determined_by_solutions_l353_353951


namespace sum_last_digit_8_l353_353784

/--
In the set \( M = \{1, 2, \cdots, 2018\} \), the sum of the elements whose last digit is 8 is 204626.
-/

theorem sum_last_digit_8:
  let M := Finset.range 2019 in
  let S := (M.filter (λ n => n % 10 = 8)).sum (λ n => n) in
  S = 204626 := 
by
  sorry

end sum_last_digit_8_l353_353784


namespace slope_of_line_l353_353446

theorem slope_of_line (z : ℝ) (h_z : z = 3) : 
  let equation := (λ x y z, 3 * y = 4 * x - 9 + 2 * z) in
  let modified_eq := equation x y z in
  let final_eq := 3 * y = 4 * x - 3 in
  equation x y z → 
  final_eq → 
  let slope := 4 / 3 in
  slope = 4 / 3 :=
by 
  sorry

end slope_of_line_l353_353446


namespace multiplier_for_first_part_is_one_l353_353037

-- Define the given number
def x : ℝ := 5.0

-- Define the equation according to the condition
def equation : ℝ := x + 7 * x

-- State the problem as a theorem in Lean
theorem multiplier_for_first_part_is_one : x = 5.0 ∧ (x + 7 * x = 55) → 1 = 1 :=
by
  intros,
  sorry

end multiplier_for_first_part_is_one_l353_353037


namespace largest_integer_less_than_100_with_remainder_5_l353_353634

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l353_353634


namespace product_of_constants_factoring_quadratic_l353_353160

theorem product_of_constants_factoring_quadratic :
  let p := ∏ t in {t | ∃ a b : ℤ, ab = -24 ∧ t = a + b}, t 
  in p = 5290000 := by
sorry

end product_of_constants_factoring_quadratic_l353_353160


namespace candies_problem_l353_353794

theorem candies_problem (x : ℕ) (Nina : ℕ) (Oliver : ℕ) (total_candies : ℕ) (h1 : 4 * x = Mark) (h2 : 2 * Mark = Nina) (h3 : 6 * Nina = Oliver) (h4 : x + Mark + Nina + Oliver = total_candies) :
  x = 360 / 61 :=
by
  sorry

end candies_problem_l353_353794


namespace greatest_unexpressible_sum_l353_353554

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem greatest_unexpressible_sum : 
  ∀ (n : ℕ), (∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n) → n ≤ 11 :=
by
  sorry

end greatest_unexpressible_sum_l353_353554


namespace evaluate_expression_l353_353115

theorem evaluate_expression (a b c : ℕ) (h1 : a = 12) (h2 : b = 8) (h3 : c = 3) :
  (a - b + c - (a - (b + c)) = 6) := by
  sorry

end evaluate_expression_l353_353115


namespace mean_of_other_two_numbers_l353_353500

-- Definitions based on conditions in the problem.
def mean_of_four (numbers : List ℕ) : ℝ := 2187.25
def sum_of_numbers : ℕ := 1924 + 2057 + 2170 + 2229 + 2301 + 2365
def sum_of_four_numbers : ℝ := 4 * 2187.25
def sum_of_two_numbers := sum_of_numbers - sum_of_four_numbers

-- Theorem to assert the mean of the other two numbers.
theorem mean_of_other_two_numbers : (4297 / 2) = 2148.5 := by
  sorry

end mean_of_other_two_numbers_l353_353500


namespace data_summary_correct_median_correct_reward_recipients_correct_l353_353259

-- Define the sales data
def sales_data : List ℚ := [5.9, 9.9, 6.0, 5.2, 8.2, 6.2, 7.6, 9.4, 8.2, 7.8, 
                            5.1, 7.5, 6.1, 6.3, 6.7, 7.9, 8.2, 8.5, 9.2, 9.8]

-- Define the data summary
def frequency (sales_range : ℚ × ℚ) : ℕ :=
  match sales_range with
  | (5, 6)   => 3
  | (6, 7)   => 5
  | (7, 8)   => 4
  | (8, 9)   => 4
  | (9, 10)  => 4
  | _ => 0

-- Calculate total frequency for proof
theorem data_summary_correct :
  let a := 20 - frequency (5, 6) - frequency (6, 7) - frequency (8, 9) - frequency (9, 10) in
  a = 4 :=
by
  let a := 20 - 3 - 5 - 4 - 4
  show a = 4
  sorry

-- Calculate the median sales value for proof
theorem median_correct :
  let sorted_sales := List.sort sales_data in
  let b := (sorted_sales[9] + sorted_sales[10]) / 2 in
  b = 7.7 :=
by
  let sorted_sales := List.sort sales_data
  let b := (sorted_sales[9] + sorted_sales[10]) / 2
  show b = 7.7
  sorry

-- Calculate the number of employees receiving rewards
theorem reward_recipients_correct :
  let target := 7
  let reward_count := sales_data.filter (λ x => x >= target) |>.length in
  reward_count = 12 :=
by
  let target := 7
  let reward_count := sales_data.filter (λ x => x >= target) |>.length
  show reward_count = 12
  sorry

end data_summary_correct_median_correct_reward_recipients_correct_l353_353259


namespace correct_operation_is_d_l353_353452

theorem correct_operation_is_d (a b : ℝ) :
  2 * a^2 * 3 * b^2 = 6 * a^2 * b^2 := 
by
  calc
    2 * a^2 * 3 * b^2 = 2 * 3 * a^2 * b^2 : by rw mul_assoc
                   ... = 6 * a^2 * b^2 : by norm_num

end correct_operation_is_d_l353_353452


namespace cube_surface_area_increase_l353_353958

theorem cube_surface_area_increase (s : ℝ) :
  let original_surface_area := 6 * s^2,
      new_edge_length := 1.3 * s,
      new_surface_area := 6 * (new_edge_length)^2,
      percentage_increase := ((new_surface_area / original_surface_area) - 1) * 100
  in percentage_increase = 69 := 
by
  sorry

end cube_surface_area_increase_l353_353958


namespace find_BD_when_AC_over_AB_min_l353_353286

open Real

-- Define the conditions
variables (A B C D : Point)
variables (AD BC BD CD : ℝ)
variables (angle_ADB : ℝ)
variables (AC AB : ℝ)

-- Assign conditions
def conditions : Prop := 
  angle_ADB = 120 ∧
  AD = 2 ∧
  CD = 2 * BD ∧
  AC / AB = minimized

-- Target to prove
theorem find_BD_when_AC_over_AB_min (A B C D : Point) (BD : ℝ)
  (h : conditions A B C D 2 120 BD 2 * BD AC AB) :
  BD = sqrt 3 - 1 :=
sorry

end find_BD_when_AC_over_AB_min_l353_353286


namespace min_ratio_bd_l353_353312

noncomputable def triangle_ratio_min (BD : ℝ) : Prop :=
  ∃ K L : ℝ, 
    (K = 1) ∧ 
    (L = 1) ∧ 
    (∀ (x : ℝ), ∀ (y : ℝ), (CD = 2 * BD) → ((x = sqrt(3) - 1) → y = (bd + 1 + 3 / (bd + 1)) ∧ (x + y) = 2 * K * L) ∧ 
    (abs x ≤ (sqrt 3 - 1)))


theorem min_ratio_bd (BD : ℝ) : triangle_ratio_min BD :=
by { sorry }

end min_ratio_bd_l353_353312


namespace axis_of_symmetry_shift_l353_353400

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem axis_of_symmetry_shift (f : ℝ → ℝ) (hf : is_even_function f) :
  (∃ a : ℝ, ∀ x : ℝ, f (x + 1) = f (-(x + 1))) :=
sorry

end axis_of_symmetry_shift_l353_353400


namespace arithmetic_sequence_middle_term_l353_353026

theorem arithmetic_sequence_middle_term :
  ∀ (a b : ℕ) (z : ℕ), a = 9 → b = 81 → z = (a + b) / 2 → z = 45 :=
by
  intros a b z h_a h_b h_z
  rw [h_a, h_b] at h_z
  exact h_z

end arithmetic_sequence_middle_term_l353_353026


namespace slope_of_line_determined_by_solutions_l353_353952

theorem slope_of_line_determined_by_solutions (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : 3 / x₁ +  4 / y₁ = 0)
  (h₂ : 3 / x₂ + 4 / y₂ = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -4 / 3 :=
sorry

end slope_of_line_determined_by_solutions_l353_353952


namespace solve_inequality_system_l353_353380

theorem solve_inequality_system (x : ℝ) (h1 : 3 * x - 2 < x) (h2 : (1 / 3) * x < -2) : x < -6 :=
sorry

end solve_inequality_system_l353_353380


namespace Lei_Lei_sheep_count_l353_353795

-- Define the initial average price and number of sheep as parameters
variables (a : ℝ) (x : ℕ)

-- Conditions as hypotheses
def condition1 : Prop := ∀ a x: ℝ,
  60 * x + 2 * (a + 60) = 90 * x + 2 * (a - 90)

-- The main problem stated as a theorem to be proved
theorem Lei_Lei_sheep_count (h : condition1) : x = 10 :=
sorry


end Lei_Lei_sheep_count_l353_353795


namespace find_k_plus_m_l353_353412

-- Define the vertices of the triangle
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 420)
def C : ℝ × ℝ := (560, 0)

-- Define the given point P7
def P7 : ℝ × ℝ := (14, 92)

-- The midpoint function
def midpoint (P L : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + L.1) / 2, (P.2 + L.2) / 2)

-- Define the point P₁ = (k, m)
variable (k m : ℝ)
def P1 : ℝ × ℝ := (k, m)

-- Define the sequence rule as a recursive function (partial definition for illustration)
noncomputable def point_sequence (n : ℕ) : ℝ × ℝ :=
  if n = 1 then P1
  else if n = 2 then midpoint P1 A
  -- this might be expanded as necessary but the exact path is not given in the question
  else sorry

-- Theorem to prove k + m = 344 given the conditions and question
theorem find_k_plus_m (h: point_sequence 7 = P7) : k + m = 344 :=
sorry

end find_k_plus_m_l353_353412


namespace solve_for_x_l353_353870

theorem solve_for_x (x y : ℝ) (h1 : 9^y = x^12) (h2 : y = 6) : x = 3 :=
by
  sorry

end solve_for_x_l353_353870


namespace proof_2m_sub_n_eq_one_l353_353182

variable (m n : ℝ)

theorem proof_2m_sub_n_eq_one (h1 : 9 ^ m = 3 / 2) (h2 : 3 ^ n = 1 / 2) : 2 * m - n = 1 :=
by
  -- Proof goes here
  sorry

end proof_2m_sub_n_eq_one_l353_353182


namespace hawks_total_points_l353_353877

def touchdowns : ℕ := 3
def points_per_touchdown : ℕ := 7
def total_points (t : ℕ) (p : ℕ) : ℕ := t * p

theorem hawks_total_points : total_points touchdowns points_per_touchdown = 21 := 
by 
  sorry

end hawks_total_points_l353_353877


namespace no_bounded_function_exists_l353_353530

theorem no_bounded_function_exists (f : ℝ → ℝ) :
  (∃ M > 0, ∀ x, |f x| ≤ M) →
  (f 1 > 0) →
  (∀ x y, f(x + y)^2 ≥ f(x)^2 + 2*f(x*y) + f(y)^2) →
  false :=
by
  sorry

end no_bounded_function_exists_l353_353530


namespace find_BD_when_AC_over_AB_min_l353_353281

open Real

-- Define the conditions
variables (A B C D : Point)
variables (AD BC BD CD : ℝ)
variables (angle_ADB : ℝ)
variables (AC AB : ℝ)

-- Assign conditions
def conditions : Prop := 
  angle_ADB = 120 ∧
  AD = 2 ∧
  CD = 2 * BD ∧
  AC / AB = minimized

-- Target to prove
theorem find_BD_when_AC_over_AB_min (A B C D : Point) (BD : ℝ)
  (h : conditions A B C D 2 120 BD 2 * BD AC AB) :
  BD = sqrt 3 - 1 :=
sorry

end find_BD_when_AC_over_AB_min_l353_353281


namespace min_ratio_bd_l353_353314

noncomputable def triangle_ratio_min (BD : ℝ) : Prop :=
  ∃ K L : ℝ, 
    (K = 1) ∧ 
    (L = 1) ∧ 
    (∀ (x : ℝ), ∀ (y : ℝ), (CD = 2 * BD) → ((x = sqrt(3) - 1) → y = (bd + 1 + 3 / (bd + 1)) ∧ (x + y) = 2 * K * L) ∧ 
    (abs x ≤ (sqrt 3 - 1)))


theorem min_ratio_bd (BD : ℝ) : triangle_ratio_min BD :=
by { sorry }

end min_ratio_bd_l353_353314


namespace angle_BOC_gt_60_l353_353395

theorem angle_BOC_gt_60 (ABCD : Type) [Trapezoid ABCD] 
  (isosceles : isIsoscelesTrapezoid ABCD) 
  (diagonalsInter : ∃ O, diagonalsIntersectAt ABCD O)
  (inscribed : hasInscribedCircle ABCD) :
  ∃ (O : ABCD), Angle B O C > 60 :=
sorry

end angle_BOC_gt_60_l353_353395


namespace old_edition_pages_l353_353081

-- Define the conditions
variables (new_edition : ℕ) (old_edition : ℕ)

-- The conditions given in the problem
axiom new_edition_pages : new_edition = 450
axiom pages_relationship : new_edition = 2 * old_edition - 230

-- Goal: Prove that the old edition Geometry book had 340 pages
theorem old_edition_pages : old_edition = 340 :=
by sorry

end old_edition_pages_l353_353081


namespace intersection_of_A_and_B_l353_353201

def A : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def B : Set ℝ := { y | 2 ≤ y ∧ y ≤ 5 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | 2 ≤ x ∧ x < 3 } :=
sorry

end intersection_of_A_and_B_l353_353201


namespace consecutive_probability_l353_353499

-- Define the total number of ways to choose 2 episodes out of 6
def total_combinations : ℕ := Nat.choose 6 2

-- Define the number of ways to choose consecutive episodes
def consecutive_combinations : ℕ := 5

-- Define the probability of choosing consecutive episodes
def probability_of_consecutive : ℚ := consecutive_combinations / total_combinations

-- Theorem stating that the calculated probability should equal 1/3
theorem consecutive_probability : probability_of_consecutive = 1 / 3 :=
by
  sorry

end consecutive_probability_l353_353499


namespace smallest_integer_n_l353_353523

theorem smallest_integer_n (n : ℕ) (h₁ : 50 ∣ n^2) (h₂ : 294 ∣ n^3) : n = 210 :=
sorry

end smallest_integer_n_l353_353523


namespace joan_seashells_l353_353331

theorem joan_seashells (initial_seashells : ℕ) (given_to_mike : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 79) (h2 : given_to_mike = 63) : remaining_seashells = 16 :=
by
  have h3 : remaining_seashells = initial_seashells - given_to_mike, from sorry,
  sorry

end joan_seashells_l353_353331


namespace arithmetic_sequence_z_value_l353_353029

theorem arithmetic_sequence_z_value :
  ∃ z : ℤ, (3 ^ 2 = 9 ∧ 3 ^ 4 = 81) ∧ z = (9 + 81) / 2 :=
by
  -- the proof goes here
  sorry

end arithmetic_sequence_z_value_l353_353029


namespace sin_difference_acutes_l353_353670

theorem sin_difference_acutes (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (h1 : sin α = 2 * sin β) (h2 : cos α = 1 / 2 * cos β) :
  sin (α - β) = 3 / 5 :=
sorry

end sin_difference_acutes_l353_353670


namespace estimate_height_l353_353014

-- Definitions directly from conditions
def x̄ : ℝ := 22.5
def ȳ : ℝ := 160
def bhat : ℝ := 4
def regression_intercept (x̄ ȳ bhat : ℝ) : ℝ := ȳ - bhat * x̄

-- The regression equation
def regression_equation (x a b : ℝ) : ℝ := b * x + a

-- Proving the estimated height for x = 24
theorem estimate_height :
  regression_equation 24 (regression_intercept x̄ ȳ bhat) bhat = 166 :=
by
  -- Substitute the values into the regression_equation and regression_intercept
  -- calculations and show that the result is 166
  sorry

end estimate_height_l353_353014


namespace number_of_integers_between_sqrt10_sqrt100_l353_353732

theorem number_of_integers_between_sqrt10_sqrt100 : 
  (set.Ico ⌈Real.sqrt 10⌉₊ (⌊Real.sqrt 100⌋₊ + 1)).card = 7 := by
  sorry

end number_of_integers_between_sqrt10_sqrt100_l353_353732


namespace num_integers_between_sqrt10_sqrt100_l353_353707

theorem num_integers_between_sqrt10_sqrt100 : 
  ∃ n : ℕ, n = 7 ∧ ∀ x : ℤ, (⌊Real.sqrt 10⌋ + 1 <= x) ∧ (x <= ⌈Real.sqrt 100⌉ - 1) ↔ (4 <= x ∧ x <= 10) := 
by 
  sorry

end num_integers_between_sqrt10_sqrt100_l353_353707


namespace train_speed_l353_353095

theorem train_speed (S D : ℝ) (h1: S * 4 = D) (h2: 100 * 2 = D) : S = 50 :=
by
  have h3 : D = 200 := by rw [mul_comm, mul_comm] at h2; exact h2
  rw [←h3] at h1
  have h4 : S * 4 = 200 := h1
  have h5 : S = 200 / 4 := by rw [← h4, mul_div_cancel_left]
  norm_num at h5
  exact h5

end train_speed_l353_353095


namespace ratio_of_division_l353_353755

theorem ratio_of_division 
  {A B C F G E : Type}
  [AffineSpace ℝ A]
  (h1 : divides_in_ratio A C F (2:ℝ) (3:ℝ))
  (G_mid_BF : is_midpoint G B F)
  (E_inter_AG_BC : is_intersection E (line A G) (line B C)) :
  divides_in_ratio B C E (2:ℝ) (5:ℝ) := 
by sorry

end ratio_of_division_l353_353755


namespace adjacent_sum_is_84_l353_353893

def divisors_of_245_except_one : List ℕ := [5, 7, 25, 35, 49, 175, 245]

def adjacent_pairs_condition (circle : List ℕ) : Prop :=
  ∀ i, ∃ j, (circle.nth i = some j) → (i + 1 < circle.length ∀ k, circle.nth (i + 1) = some k →
   gcd j k > 1) ∨ (circle.nth (i + 1 - circle.length) = some k)

theorem adjacent_sum_is_84 :
  ∃ circle : List ℕ, 
    (∀ x, x ∈ circle → x ∈ divisors_of_245_except_one) ∧ 
    List.Nodup circle ∧
    adjacent_pairs_condition circle →
    ((circle.circleIdx 7 + 1) % circle.length).map (circle.get!) + ((circle.circleIdx 7 - 1 + circle.length) % circle.length).map (circle.get!) = 84 :=
by
  sorry

end adjacent_sum_is_84_l353_353893


namespace smaller_circle_radius_l353_353427

theorem smaller_circle_radius
  (P : ℝ × ℝ) (S : ℝ × ℝ) (k : ℝ) (QP QR: ℝ)
  (origin : ℝ × ℝ)
  (hP : P = (8, 6)) 
  (hS : S = (0, k)) 
  (h_origin_center : origin = (0, 0))
  (h_QR : QR = 3) 
  (h_OP : QP = real.sqrt ((P.1 - origin.1)^2 + (P.2 - origin.2)^2)) 
  (h_radius_large_circle : QP = 10) :
  k = 7 := 
sorry

end smaller_circle_radius_l353_353427


namespace largest_int_with_remainder_5_lt_100_l353_353569

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l353_353569


namespace compare_logs_l353_353672

noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.logb 2 3
noncomputable def c : ℝ := Real.logb 5 8

theorem compare_logs : a < c ∧ c < b := by
  sorry

end compare_logs_l353_353672


namespace num_descending_4_digit_even_num_ascending_4_digit_even_l353_353233

/-- 
  The number of four-digit numbers where all digits are even 
  and are in descending order from the set {0, 2, 4, 6, 8} 
  excluding 0 from the first position.
-/
theorem num_descending_4_digit_even : 
  (∃ S : Finset ℤ, S = {2, 4, 6, 8}.image coe ∧ S.card = 4) :=
sorry

/-- 
  The number of four-digit numbers where all digits are even 
  and are in ascending order from the set {0, 2, 4, 6, 8} 
  excluding 0 entirely.
-/
theorem num_ascending_4_digit_even : 
  (↑({2, 4, 6, 8} : Finset ℤ)).card = 1 :=
sorry

end num_descending_4_digit_even_num_ascending_4_digit_even_l353_353233


namespace solution_to_equation_l353_353045

theorem solution_to_equation :
    (∃ k ∈ ℤ, x = (π / 8) * (4 * k - 1)) ∨
    (∃ n ∈ ℤ, x = (1 / 2) * arctan 2 + (π / 2) * n) ↔
    (sin (2 * x))^2 * cos ((3 * π / 2) - 2 * x) +
    3 * sin (2 * x) * (sin ((3 * π / 2) + 2 * x))^2 +
    2 * (cos (2 * x))^3 = 0 :=
sorry

end solution_to_equation_l353_353045


namespace count_integers_between_sqrt10_sqrt100_l353_353712

theorem count_integers_between_sqrt10_sqrt100 : 
  ∃ (S : Set ℤ), (∀ n, n ∈ S ↔ (real.sqrt 10 < n ∧ n < real.sqrt 100)) ∧ S.card = 6 := 
by
  sorry

end count_integers_between_sqrt10_sqrt100_l353_353712


namespace largest_integer_less_than_100_with_remainder_5_l353_353633

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l353_353633


namespace function_passes_fixed_point_l353_353657

theorem function_passes_fixed_point (a : ℝ) (h : 0 < a) (h₀ : a ≠ 1) :
  (a^(1-1) - 1 = 0) :=
by
-- Lean code will automatically handle the trivial proof when given these conditions.
trivial

end function_passes_fixed_point_l353_353657


namespace find_c_l353_353469

theorem find_c (a b c d : ℕ) (h1 : 8 = 4 * a / 100) (h2 : 4 = d * a / 100) (h3 : 8 = d * b / 100) (h4 : c = b / a) : 
  c = 2 := 
by
  sorry

end find_c_l353_353469


namespace simplify_G_to_2F_l353_353337

noncomputable def F (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

theorem simplify_G_to_2F (x : ℝ) :
    F(2*x / (1 + x^2)) = 2 * F x :=
by
    sorry

end simplify_G_to_2F_l353_353337


namespace inclination_angle_range_l353_353896

theorem inclination_angle_range (α : ℝ) : (∃ θ, 0 ≤ θ ∧ θ < π ∧ -1 ≤ -sin α ∧ -sin α ≤ 1 ∧ (θ = atan (-sin α))) → (θ ∈ Icc 0 (π/4) ∨ θ ∈ Icc (3 * π / 4) π) :=
begin
  sorry
end

end inclination_angle_range_l353_353896


namespace num_intersection_points_sine_cosine_l353_353889

theorem num_intersection_points_sine_cosine : 
  {x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin (2 * x) = Real.cos x}.finite.to_finset.card = 4 := 
by sorry

end num_intersection_points_sine_cosine_l353_353889


namespace range_of_expression_l353_353897

theorem range_of_expression (x : ℝ) :
  (2 - x ≥ 0) → (x + 3 ≠ 0) → (x ≤ 2 ∧ x ≠ -3) :=
by
  intro h1 h2
  split
  · sorry -- Proof for x ≤ 2
  · sorry -- Proof for x ≠ -3

end range_of_expression_l353_353897


namespace slope_of_line_l353_353954

theorem slope_of_line (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 3 / x + 4 / y = 0) : 
  ∃ m : ℝ, m = -4 / 3 := 
sorry

end slope_of_line_l353_353954


namespace february_relatively_prime_dates_l353_353088

def relatively_prime_dates_in_february (days_in_february : ℕ) (month_number : ℕ) : ℕ :=
  days_in_february - (days_in_february / month_number)

theorem february_relatively_prime_dates : relatively_prime_dates_in_february 28 2 = 14 :=
by
  have h1 : 28 / 2 = 14 := by norm_num
  have h2 : 28 - 14 = 14 := by norm_num
  rw [relatively_prime_dates_in_february, h1, h2]
  sorry

end february_relatively_prime_dates_l353_353088


namespace discrim_of_quad_l353_353521

-- Definition of the quadratic equation coefficients
def a : ℤ := 5
def b : ℤ := -9
def c : ℤ := 4

-- Definition of the discriminant formula which needs to be proved as 1
def discriminant (a b c : ℤ) : ℤ :=
  b^2 - 4 * a * c

-- The proof problem statement
theorem discrim_of_quad : discriminant a b c = 1 := by
  sorry

end discrim_of_quad_l353_353521


namespace second_merchant_makes_more_profit_l353_353430

-- Let x be the cost price of the goods. 
variable (x : ℝ)

-- The first merchant sells his goods at twice the purchase price.
def selling_price_first_merchant := 2 * x

-- The profit of the first merchant.
def profit_first_merchant := selling_price_first_merchant - x

-- The second merchant increases the price by 60% for the first one-fourth of the goods.
def selling_price_first_portion := 1.6 * x
def profit_first_portion := (1 / 4) * selling_price_first_portion

-- The second merchant increases the price by an additional 40% for the remaining three-fourths of the goods.
def selling_price_second_portion := 2.24 * x
def profit_second_portion := (3 / 4) * selling_price_second_portion

-- The total profit of the second merchant.
def total_profit_second_merchant := profit_first_portion + profit_second_portion - x

-- Prove that the second merchant's profit is greater than the first merchant's profit.
theorem second_merchant_makes_more_profit :
  total_profit_second_merchant x > profit_first_merchant x :=
sorry

end second_merchant_makes_more_profit_l353_353430


namespace parallelogram_area_l353_353513

/-
Given points P, Q, R, S in 3-dimensional space, showing that they form a parallelogram
and computing the area of the parallelogram.
-/

-- Define the points P, Q, R, S as vectors
def P : ℝ × ℝ × ℝ := (1, -2, 3)
def Q : ℝ × ℝ × ℝ := (3, -6, 6)
def R : ℝ × ℝ × ℝ := (2, -1, 1)
def S : ℝ × ℝ × ℝ := (4, -5, 4)

-- Define vectors PQ and RS
def PQ := (Q.1 - P.1, Q.2 - P.2, Q.3 - P.3)
def RS := (S.1 - R.1, S.2 - R.2, S.3 - R.3)

-- Define vector RP
def RP := (R.1 - P.1, R.2 - P.2, R.3 - P.3)

-- Cross product of PQ and RP
def crossProduct (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.2 * b.3 - a.3 * b.2), (a.3 * b.1 - a.1 * b.3), (a.1 * b.2 - a.2 * b.1))

-- Magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Cross product of PQ and RP
def PQ_cross_RP := crossProduct PQ RP

-- Area of the parallelogram
def area : ℝ := magnitude PQ_cross_RP

theorem parallelogram_area :
  (PQ = RS) ∧ (area = real.sqrt 110) :=
by
  unfold PQ RS RP crossProduct magnitude area
  sorry

end parallelogram_area_l353_353513


namespace micah_has_seven_fish_l353_353358

-- Definitions from problem conditions
def micahFish (M : ℕ) : Prop :=
  let kennethFish := 3 * M
  let matthiasFish := kennethFish - 15
  M + kennethFish + matthiasFish = 34

-- Main statement: prove that the number of fish Micah has is 7
theorem micah_has_seven_fish : ∃ M : ℕ, micahFish M ∧ M = 7 :=
by
  sorry

end micah_has_seven_fish_l353_353358


namespace divide_1_2_3_groups_divide_3_groups_of_2_divide_among_A_B_C_l353_353005

-- Define the conditions for the problems
constant books : Finset ℕ
constant h_books_size : books.card = 6

-- Define combinations function
noncomputable def choose (n k : ℕ) : ℕ := Nat.binomial n k

-- Problem 1: Prove the number of ways to divide the books into groups with 1, 2, and 3 books.
theorem divide_1_2_3_groups : 
  choose 6 1 * choose 5 2 * choose 3 3 = 60 := 
by
  sorry

-- Problem 2: Prove the number of ways to divide the books into groups with 2 books each.
theorem divide_3_groups_of_2 : 
  (choose 6 2 * choose 4 2 * choose 2 2) / nat.factorial 3 = 15 := 
by
  sorry

-- Problem 3: Prove the number of ways to divide the books among A, B, and C.
theorem divide_among_A_B_C : 
  choose 6 2 * choose 4 2 * choose 2 2 = 90 := 
by
  sorry

end divide_1_2_3_groups_divide_3_groups_of_2_divide_among_A_B_C_l353_353005


namespace minimum_BD_value_l353_353302

theorem minimum_BD_value {A B C D : Type*} [Real A] [Real B] [Real C] [Real D] (angle_ADB: A) (AD: A) (CD: A) (BD: A):
  (D ∈ line_segment(B, C)) →
  angle_ADB = 120 → 
  AD = 2 → 
  CD = 2 * BD →
  ∃ BD: A, BD = sqrt(3) - 1 :=
by
  sorry

end minimum_BD_value_l353_353302


namespace find_BD_when_AC_over_AB_min_l353_353329

-- Definition of the conditions
variables (A B C D : Type) [metric_space A]
variables (triangle : set A)
variables (on_side_BC : D ∈ segment B C)
variables (angle_ADB : ∠ADB = 120)
variables (AD_len : dist A D = 2)
variables (CD_2BD : dist C D = 2 * dist B D)

noncomputable def minimum_ratio_condition (BD : ℝ) : Prop := 
  let x := dist B D
  let CD := 2 * x
  let AD := 2 in
  let angle_ADB_cos := -1/2 in
  let b_squared := AD^2 + CD^2 - 2 * AD * CD * angle_ADB_cos in
  let c_squared := AD^2 + x^2 + 2 * AD * x * angle_ADB_cos in
  b_squared / c_squared = 4 - (12 / (x + 1 + 3/(x + 1))) → x = sqrt 3 - 1

theorem find_BD_when_AC_over_AB_min (BD : ℝ) : minimum_ratio_condition A B C D 
:= sorry

end find_BD_when_AC_over_AB_min_l353_353329


namespace distinct_star_shaped_pgons_wilsons_theorem_l353_353963

-- Definition and conditions for part (a)
def star_shaped_pgons_count (p : ℕ) [Fact (Nat.Prime p)] [H : Odd p] : ℕ :=
  1 / 2 * (Nat.div ((p - 1)! + 1) p + p - 4)

-- Theorem for part (a)
theorem distinct_star_shaped_pgons (p : ℕ) [Fact (Nat.Prime p)] [H : Odd p] :
  star_shaped_pgons_count p = 1 / 2 * (Nat.div ((p - 1)! + 1) p + p - 4) :=
sorry

-- Theorem for part (b)
theorem wilsons_theorem (p : ℕ) [Fact (Nat.Prime p)] : 
  p ∣ ((p - 1)! + 1) :=
sorry

end distinct_star_shaped_pgons_wilsons_theorem_l353_353963


namespace find_initial_number_l353_353083

theorem find_initial_number (x : ℝ) (h : x + 12.808 - 47.80600000000004 = 3854.002) : x = 3889 := by
  sorry

end find_initial_number_l353_353083


namespace largest_integer_less_than_100_with_remainder_5_l353_353621

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353621


namespace necessary_but_not_sufficient_l353_353464

theorem necessary_but_not_sufficient (x : ℝ) :
  (x < -1) → ((x^2 - 1 > 0) ∧ ¬(x^2 - 1 > 0 → x < -1)) :=
by {
  assume h,
  split,
  { sorry },   -- prove (x^2 - 1 > 0)
  { sorry }    -- prove ¬(x^2 - 1 > 0 → x < -1)
}

end necessary_but_not_sufficient_l353_353464


namespace count_integers_between_sqrt10_sqrt100_l353_353713

theorem count_integers_between_sqrt10_sqrt100 : 
  ∃ (S : Set ℤ), (∀ n, n ∈ S ↔ (real.sqrt 10 < n ∧ n < real.sqrt 100)) ∧ S.card = 6 := 
by
  sorry

end count_integers_between_sqrt10_sqrt100_l353_353713


namespace n_is_square_if_m_even_l353_353761

theorem n_is_square_if_m_even
  (n : ℕ)
  (h1 : n ≥ 3)
  (m : ℕ)
  (h2 : m = (1 / 2) * n * (n - 1))
  (h3 : ∀ i j : ℕ, i ≠ j → (a_i + a_j) % m ≠ (a_j + a_k) % m)
  (h4 : even m) :
  ∃ k : ℕ, n = k * k := sorry

end n_is_square_if_m_even_l353_353761


namespace original_deck_size_l353_353073

noncomputable def initial_red_probability (r b : ℕ) : Prop := r / (r + b) = 1 / 4
noncomputable def added_black_probability (r b : ℕ) : Prop := r / (r + (b + 6)) = 1 / 6

theorem original_deck_size (r b : ℕ) 
  (h1 : initial_red_probability r b) 
  (h2 : added_black_probability r b) : 
  r + b = 12 := 
sorry

end original_deck_size_l353_353073


namespace find_BD_when_AC_over_AB_min_l353_353276

open Real

-- Define the conditions
variables (A B C D : Point)
variables (AD BC BD CD : ℝ)
variables (angle_ADB : ℝ)
variables (AC AB : ℝ)

-- Assign conditions
def conditions : Prop := 
  angle_ADB = 120 ∧
  AD = 2 ∧
  CD = 2 * BD ∧
  AC / AB = minimized

-- Target to prove
theorem find_BD_when_AC_over_AB_min (A B C D : Point) (BD : ℝ)
  (h : conditions A B C D 2 120 BD 2 * BD AC AB) :
  BD = sqrt 3 - 1 :=
sorry

end find_BD_when_AC_over_AB_min_l353_353276


namespace equal_probabilities_l353_353193

theorem equal_probabilities
  (N : ℕ)
  (n : ℕ)
  (p_1 p_2 p_3 : ℝ)
  (h1 : ∀ k : ℕ, k ≤ N → k = n → p_1 = p_2 = p_3)
  (h2 : ∀ k : ℕ, k ≤ N → k = n → p_2 = p_1)
  (h3 : ∀ k : ℕ, k ≤ N → k = n → p_3 = p_1) :
  p_1 = p_2 ∧ p_2 = p_3 ∧ p_3 = p_1 := 
by {
  sorry
}

end equal_probabilities_l353_353193


namespace min_ratio_bd_l353_353317

noncomputable def triangle_ratio_min (BD : ℝ) : Prop :=
  ∃ K L : ℝ, 
    (K = 1) ∧ 
    (L = 1) ∧ 
    (∀ (x : ℝ), ∀ (y : ℝ), (CD = 2 * BD) → ((x = sqrt(3) - 1) → y = (bd + 1 + 3 / (bd + 1)) ∧ (x + y) = 2 * K * L) ∧ 
    (abs x ≤ (sqrt 3 - 1)))


theorem min_ratio_bd (BD : ℝ) : triangle_ratio_min BD :=
by { sorry }

end min_ratio_bd_l353_353317


namespace fraction_of_area_outside_l353_353770

-- Define the isosceles triangle ABC with given properties
def is_isosceles_triangle (A B C : Point) : Prop :=
  distance A B = distance A C ∧ distance B C = 10 ∧ height_from A B C = 12

-- Define the area of a triangle given the base and height
def triangle_area (base height : ℝ) : ℝ :=
  ½ * base * height

-- Calculate the semi-perimeter of the triangle ABC assuming it is isosceles
def semiperimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

-- Define the radius of the inscribed circle in the triangle
def inscribed_circle_radius (A s : ℝ) : ℝ :=
  A / s

-- Calculate the area of a circle given its radius
def circle_area (r : ℝ) : ℝ :=
  π * r^2

-- Calculate the fraction of the area of triangle ABC that lies outside the circle
theorem fraction_of_area_outside (A B C : Point) (r : ℝ) (A_triangle : ℝ)
  (h_isosceles : is_isosceles_triangle A B C)
  (h_radius : r = inscribed_circle_radius A_triangle (semiperimeter 10 (distance A B) (distance A C)))
  : (A_triangle - circle_area r) / A_triangle = 1 - 5 * π / 27 :=
by
  sorry

end fraction_of_area_outside_l353_353770


namespace binom_eq_fraction_l353_353923

open Nat

theorem binom_eq_fraction (n : ℕ) (h_pos : 0 < n) : choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_eq_fraction_l353_353923


namespace particle_intersect_sphere_distance_zero_l353_353258

theorem particle_intersect_sphere_distance_zero :
  let start := (1, 2, 3)
  let end := (-2, -4, -6)
  let line := λ t : ℝ, (1 - 3 * t, 2 - 6 * t, 3 - 9 * t)
  let sphere_eq := λ p : ℝ × ℝ × ℝ, p.1^2 + p.2^2 + p.3^2 = 1
  let intersections := { t : ℝ | sphere_eq (line t) }
  ∀ t1 t2 ∈ intersections, dist (line t1) (line t2) = 0 :=
by
  sorry

end particle_intersect_sphere_distance_zero_l353_353258


namespace minimum_BD_value_l353_353299

theorem minimum_BD_value {A B C D : Type*} [Real A] [Real B] [Real C] [Real D] (angle_ADB: A) (AD: A) (CD: A) (BD: A):
  (D ∈ line_segment(B, C)) →
  angle_ADB = 120 → 
  AD = 2 → 
  CD = 2 * BD →
  ∃ BD: A, BD = sqrt(3) - 1 :=
by
  sorry

end minimum_BD_value_l353_353299


namespace old_edition_pages_l353_353079

theorem old_edition_pages (x : ℕ) (h : 2 * x - 230 = 450) : x = 340 :=
by {
  have eq1 : 2 * x = 450 + 230, from eq_add_of_sub_eq h,
  have eq2 : 2 * x = 680, from eq1,
  have eq3 : x = 680 / 2, from eq_of_mul_eq_mul_right (by norm_num) eq2,
  norm_num at eq3,
  exact eq3,
}

end old_edition_pages_l353_353079


namespace largest_integer_less_than_100_with_remainder_5_l353_353636

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l353_353636


namespace old_edition_pages_l353_353077

theorem old_edition_pages (x : ℕ) (h : 2 * x - 230 = 450) : x = 340 :=
by {
  have eq1 : 2 * x = 450 + 230, from eq_add_of_sub_eq h,
  have eq2 : 2 * x = 680, from eq1,
  have eq3 : x = 680 / 2, from eq_of_mul_eq_mul_right (by norm_num) eq2,
  norm_num at eq3,
  exact eq3,
}

end old_edition_pages_l353_353077


namespace arc_length_sum_limit_l353_353407

theorem arc_length_sum_limit (R : ℝ) : 
  ∀ (n : ℕ), (n > 0) → 
  (let segment_length := R / n in
   let quarter_circle_arc_length := (π * R) / (2 * n) in
   let total_arc_length := n * quarter_circle_arc_length in
   total_arc_length→ n → ∞ = π * R / 2) :=
sorry

end arc_length_sum_limit_l353_353407


namespace rectangle_width_decrease_l353_353887

theorem rectangle_width_decrease {L W : ℝ} (A : ℝ) (hA : A = L * W) (h_new_length : A = 1.25 * L * (W * y)) : y = 0.8 :=
by sorry

end rectangle_width_decrease_l353_353887


namespace number_of_promotional_posters_l353_353421

theorem number_of_promotional_posters (n m : ℕ) (hn : n = 7) (hm : m = 6) : 
  (nat.choose m (m/2)) = 20 := 
by
  have h1 : m / 2 = 3 :=
    by norm_num
  rw h1
  rw nat.choose_eq_factorial_div_factorial (nat.le.intro rfl)
  norm_num
  sorry

end number_of_promotional_posters_l353_353421


namespace prove_interest_rates_equal_l353_353792

noncomputable def interest_rates_equal : Prop :=
  let initial_savings := 1000
  let savings_simple := initial_savings / 2
  let savings_compound := initial_savings / 2
  let simple_interest_earned := 100
  let compound_interest_earned := 105
  let time := 2
  let r_s := simple_interest_earned / (savings_simple * time)
  let r_c := (compound_interest_earned / savings_compound + 1) ^ (1 / time) - 1
  r_s = r_c

theorem prove_interest_rates_equal : interest_rates_equal :=
  sorry

end prove_interest_rates_equal_l353_353792


namespace find_BD_when_AC_over_AB_min_l353_353325

-- Definition of the conditions
variables (A B C D : Type) [metric_space A]
variables (triangle : set A)
variables (on_side_BC : D ∈ segment B C)
variables (angle_ADB : ∠ADB = 120)
variables (AD_len : dist A D = 2)
variables (CD_2BD : dist C D = 2 * dist B D)

noncomputable def minimum_ratio_condition (BD : ℝ) : Prop := 
  let x := dist B D
  let CD := 2 * x
  let AD := 2 in
  let angle_ADB_cos := -1/2 in
  let b_squared := AD^2 + CD^2 - 2 * AD * CD * angle_ADB_cos in
  let c_squared := AD^2 + x^2 + 2 * AD * x * angle_ADB_cos in
  b_squared / c_squared = 4 - (12 / (x + 1 + 3/(x + 1))) → x = sqrt 3 - 1

theorem find_BD_when_AC_over_AB_min (BD : ℝ) : minimum_ratio_condition A B C D 
:= sorry

end find_BD_when_AC_over_AB_min_l353_353325


namespace largest_integer_less_than_100_with_remainder_5_l353_353620

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353620


namespace problem_solution_BE_l353_353367

open EuclideanGeometry

noncomputable def length_BE (F E : Point) (s : ℝ) : ℝ :=
  let A := mkPoint 0 0
  let B := mkPoint s 0
  let C := mkPoint s s
  let D := mkPoint 0 s
  let CF := distance C F
  let CE := distance C E
  let area_ΔCEF := (1 / 2) * CF * CE
  let side_length := Real.sqrt 256
  let area_condition := area_ΔCEF = 200
  let F_on_AD := F.y = side_length ∧ F.x = 0
  let E_on_AB_ext := E.y = 0 ∧ E.x > side_length
  let CF_perpendicular_CE := CF * CE = side_length ^ 2
  let right_angle := E.y == 0 ∧ F.x == 0
  let BE := distance B E

  if F_on_AD ∧ E_on_AB_ext ∧ CF_perpendicular_CE ∧ right_angle ∧ area_condition then
    BE
  else
    sorry

-- Check whether the length of BE is indeed 12 inches
theorem problem_solution_BE : ∃ F E : Point, length_BE F E 16 = 12 :=
by
  exists A (mkPoint 16 (20 * Real.sqrt 10))
  sorry

end problem_solution_BE_l353_353367


namespace part1_increasing_on_interval_part2_min_value_on_range_l353_353685

noncomputable def f (x : Real) (a : Real) : Real := x^2 - a * Real.log x

theorem part1_increasing_on_interval {a : Real} (h : a = 2) :
  ∀ (x : Real), 1 < x → 0 < (derivative (fun x => x^2 - a * Real.log x)) x :=
by
  sorry

theorem part2_min_value_on_range {a : Real} (h1 : 2 < a) (h2 : a < 2 * Real.exp 2) :
  ∃ (x : Real), 1 ≤ x ∧ x ≤ Real.exp 1 ∧ 
    f x a = (a / 2) * (1 - Real.log (a / 2)) :=
by
  sorry

end part1_increasing_on_interval_part2_min_value_on_range_l353_353685


namespace find_BD_when_AC_over_AB_min_l353_353323

-- Definition of the conditions
variables (A B C D : Type) [metric_space A]
variables (triangle : set A)
variables (on_side_BC : D ∈ segment B C)
variables (angle_ADB : ∠ADB = 120)
variables (AD_len : dist A D = 2)
variables (CD_2BD : dist C D = 2 * dist B D)

noncomputable def minimum_ratio_condition (BD : ℝ) : Prop := 
  let x := dist B D
  let CD := 2 * x
  let AD := 2 in
  let angle_ADB_cos := -1/2 in
  let b_squared := AD^2 + CD^2 - 2 * AD * CD * angle_ADB_cos in
  let c_squared := AD^2 + x^2 + 2 * AD * x * angle_ADB_cos in
  b_squared / c_squared = 4 - (12 / (x + 1 + 3/(x + 1))) → x = sqrt 3 - 1

theorem find_BD_when_AC_over_AB_min (BD : ℝ) : minimum_ratio_condition A B C D 
:= sorry

end find_BD_when_AC_over_AB_min_l353_353323


namespace doubled_container_volume_l353_353480

theorem doubled_container_volume (v : ℝ) (h₁ : v = 4) (h₂ : ∀ l w h : ℝ, v = l * w * h) : 8 * v = 32 := 
by
  -- The proof will go here, this is just the statement
  sorry

end doubled_container_volume_l353_353480


namespace ze_age_conditions_l353_353463

theorem ze_age_conditions 
  (z g t : ℕ)
  (h1 : z = 2 * g + 3 * t)
  (h2 : 2 * (z + 15) = 2 * (g + 15) + 3 * (t + 15))
  (h3 : 2 * (g + 15) = 3 * (t + 15)) :
  z = 45 ∧ t = 5 :=
by
  sorry

end ze_age_conditions_l353_353463


namespace largest_prime_factor_of_S_is_11_l353_353986

-- Definitions and conditions translated to Lean
def condition1 (seq : List ℕ) : Prop :=
  ∀ n ∈ seq, 100 ≤ n ∧ n < 1000

def condition2 (seq : List ℕ) : Prop :=
  ∀ (i : ℕ), i < seq.length - 1 →
    (seq[i] / 10 % 10 = seq[i+1] / 100 ∧
    seq[i] % 10 = seq[i+1] / 10 % 10)

def condition3 (seq : List ℕ) : Prop :=
  ∀ (i : ℕ), i < seq.length - 1 →
    (seq[i] % 10 = 9 → seq[i+1] / 100 = 0) ∧
    (seq[i] % 10 ≠ 9 → seq[i+1] / 100 = seq[i] % 10 + 1)

def condition4 (seq : List ℕ) : Prop :=
  (seq.head! / 10 % 10 = seq.last! / 100) ∧
  (seq.head! % 10 = 9 → seq.last! / 100 = 0) ∧
  (seq.head! % 10 ≠ 9 → seq.last! / 100 = seq.head! % 10 + 1)

-- Problem statement
theorem largest_prime_factor_of_S_is_11 (seq : List ℕ)
  (h1 : condition1 seq)
  (h2 : condition2 seq)
  (h3 : condition3 seq)
  (h4 : condition4 seq) : 11 ∣ seq.sum :=
sorry

end largest_prime_factor_of_S_is_11_l353_353986


namespace count_integers_between_sqrt10_sqrt100_l353_353716

theorem count_integers_between_sqrt10_sqrt100 : 
  ∃ (S : Set ℤ), (∀ n, n ∈ S ↔ (real.sqrt 10 < n ∧ n < real.sqrt 100)) ∧ S.card = 6 := 
by
  sorry

end count_integers_between_sqrt10_sqrt100_l353_353716


namespace perpendicular_planes_l353_353342

-- Definitions of lines, planes, and their relations
variable (m n : Type)
variable (α β : Type)
variable [is_line m]
variable [is_line n]
variable [is_plane α]
variable [is_plane β]

-- Conditions
variable (h1 : m ∈ α)
variable (h2 : n ∈ β)
variable (h3 : n ⊥ α)

-- The theorem to be proved
theorem perpendicular_planes (m n : Type) (α β : Type) [is_line m] [is_line n] [is_plane α] [is_plane β]
  (h1 : m ∈ α) (h2 : n ∈ β) (h3 : n ⊥ α) : α ⊥ β :=
sorry

end perpendicular_planes_l353_353342


namespace num_integers_between_sqrt10_sqrt100_l353_353708

theorem num_integers_between_sqrt10_sqrt100 : 
  ∃ n : ℕ, n = 7 ∧ ∀ x : ℤ, (⌊Real.sqrt 10⌋ + 1 <= x) ∧ (x <= ⌈Real.sqrt 100⌉ - 1) ↔ (4 <= x ∧ x <= 10) := 
by 
  sorry

end num_integers_between_sqrt10_sqrt100_l353_353708


namespace math_proof_problem_l353_353350

noncomputable def discriminant (a : ℝ) : ℝ := a^2 - 4 * a + 2

def is_real_roots (a : ℝ) : Prop := discriminant a ≥ 0

def solution_set_a : Set ℝ := { a | is_real_roots a ∧ (a ≤ 2 - Real.sqrt 2 ∨ a ≥ 2 + Real.sqrt 2) }

def f (a : ℝ) : ℝ := -3 * a^2 + 16 * a - 8

def inequality_m (m t : ℝ) : Prop := m^2 + t * m + 4 * Real.sqrt 2 + 6 ≥ f (2 + Real.sqrt 2)

theorem math_proof_problem :
  (∀ a ∈ solution_set_a, ∃ m : ℝ, ∀ t ∈ Set.Icc (-1 : ℝ) (1 : ℝ), inequality_m m t) ∧
  (∀ m t, inequality_m m t → m ≤ -1 ∨ m = 0 ∨ m ≥ 1) :=
by
  sorry

end math_proof_problem_l353_353350


namespace three_consecutive_odd_integers_sum_find_C_ratio_relation_find_A_l353_353409

-- Problem 1: Prove k = 15 given that k is the smallest of 3 consecutive odd integers whose sum is 51.
theorem three_consecutive_odd_integers_sum (k : ℤ) (h : k + (k + 2) + (k + 4) = 51) : k = 15 :=
sorry

-- Problem 2: Prove C = 6 given x² + 6x + k ≡ (x + a)² + C.
theorem find_C (x k a C : ℤ) (h1 : x^2 + 6 * x + k = (x + a)^2 + C) : C = 6 :=
sorry

-- Problem 3: Prove R = 8 given p/q = q/r = r/s = 2.
theorem ratio_relation (p q r s : ℚ) (h1 : p / q = 2) (h2 : q / r = 2) (h3 : r / s = 2) : p / s = 8 :=
sorry

-- Problem 4: Prove A = 729 given A = 3^n * 9^(n + 1) / 27^(n - 1).
theorem find_A (A : ℤ) (n : ℤ) (h : A = 3^n * 9^(n + 1) / 27^(n - 1)) : A = 729 :=
sorry

end three_consecutive_odd_integers_sum_find_C_ratio_relation_find_A_l353_353409


namespace mismatching_socks_l353_353382

theorem mismatching_socks (total_socks : ℕ) (pairs : ℕ) (socks_per_pair : ℕ) 
  (h1 : total_socks = 25) (h2 : pairs = 4) (h3 : socks_per_pair = 2) : 
  total_socks - (socks_per_pair * pairs) = 17 :=
by
  sorry

end mismatching_socks_l353_353382


namespace price_grade6_l353_353508

noncomputable theory
open_locale classical

-- Define the conditions as given in the problem
def price_function (a b x : ℝ) : ℝ := real.exp (a * x + b)

-- Given conditions
def condition1 (a b : ℝ) : Prop :=
  (price_function a b 5) / (price_function a b 1) = 4

def condition2 (a b : ℝ) : Prop :=
  price_function a b 3 = 55

-- Prove that the market selling price of grade 6 cherries is approximately 156 yuan/kg
theorem price_grade6 (a b : ℝ) (h1 : condition1 a b) (h2 : condition2 a b) : 
  ∃ (y : ℝ), abs (price_function a b 6 - y) < 1 ∧ y = 156 :=
  sorry

end price_grade6_l353_353508


namespace postcards_per_day_l353_353422

variable (income_per_card total_income days : ℕ)
variable (x : ℕ)

theorem postcards_per_day
  (h1 : income_per_card = 5)
  (h2 : total_income = 900)
  (h3 : days = 6)
  (h4 : total_income = income_per_card * x * days) :
  x = 30 :=
by
  rw [h1, h2, h3] at h4
  linarith

end postcards_per_day_l353_353422


namespace value_is_three_l353_353049

noncomputable def some_value(m n : ℝ) : ℝ :=
  let lhs := m
  let rhs := (n / 5) - (2 / 5)
  if lhs = rhs then (n + 15)/5 - 2/5 - m else sorry

theorem value_is_three (m n : ℝ) (cond1 : m = (n / 5) - (2 / 5))
                       (cond2 : m + some_value(m,n) = ((n + 15) / 5) - (2 / 5)) :
  some_value(m, n) = 3 :=
by sorry

end value_is_three_l353_353049


namespace largest_integer_less_than_100_with_remainder_5_l353_353607

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l353_353607


namespace no_integers_satisfy_eq_l353_353529

theorem no_integers_satisfy_eq (m n : ℤ) : ¬ (m^2 + 1954 = n^2) := 
by
  sorry

end no_integers_satisfy_eq_l353_353529


namespace g_2002_value_l353_353899

noncomputable def g : ℕ → ℤ := sorry

theorem g_2002_value :
  (∀ a b n : ℕ, a + b = 2^n → g a + g b = n^3) →
  (g 2 + g 46 = 180) →
  g 2002 = 1126 := 
by
  intros h1 h2
  sorry

end g_2002_value_l353_353899


namespace largest_int_with_remainder_5_lt_100_l353_353570

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l353_353570


namespace parabola_and_line_equations_l353_353662

noncomputable def parabola_vertex_origin_focus_xaxis (P : ℝ × ℝ) (d : ℝ) : Prop := 
  let x := P.1 in let y := P.2 in
  let p := 2 * Real.sqrt ((x + d/2)^2) in
  y^2 = 2 * p * x

def midpoint_line_intersection (midpoint : ℝ × ℝ) (l : ℝ → ℝ) : Prop := 
  let x_mid := midpoint.1 in let y_mid := midpoint.2 in
  ∃ a b : ℝ, ∀ x : ℝ, 2 * (a*x + b) = 4 * x_mid ∧ (a*x + b) = y_mid

theorem parabola_and_line_equations
  (P : ℝ × ℝ) (d : ℝ) (midpoint : ℝ × ℝ) (l : ℝ → ℝ)
  (h1 : P = (4, 6) ∧ d = 6)
  (h2 : midpoint = (2, 2))
  (h3 : ∀ x, l x = 2 * x - 2)
  : parabola_vertex_origin_focus_xaxis P d ∧ midpoint_line_intersection midpoint l := 
begin
  sorry
end

end parabola_and_line_equations_l353_353662


namespace largest_int_less_than_100_with_remainder_5_l353_353557

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l353_353557


namespace tangent_parallel_B0C0_l353_353857

variables (A B C D P B0 C0 : Point)
variables (circle : Circle) (line1 : Line) (BC : Line) (circleIsInscribed : InscribedQuadrilateral A B C D circle)
variables (P_on_arc_AD_not_containing_B_C : P ∈ Arc AD (circle) ∧ ¬(B ∈ AD) ∧ ¬(C ∈ AD))
variables (line1_is_perpendicular_to_BC : Perpendicular line1 BC)
variables (B0_on_BP : B0 ∈ LineThrough B P) (C0_on_CP : C0 ∈ LineThrough C P)
variables (B0_on_line1 : B0 ∈ line1) (C0_on_line1 : C0 ∈ line1)

theorem tangent_parallel_B0C0 : 
    ∀ (circleBPC : Circle) (tanPtP : Tangent circleBPC P),
    (Parallel tanPtP (LineThrough B0 C0)) := 
begin
  intros,
  sorry
end

end tangent_parallel_B0C0_l353_353857


namespace colin_first_mile_time_l353_353510

theorem colin_first_mile_time :
  ∃ x : ℕ, (forall n x = 6 
    ∧ (x + 10 + 4 = 20))
    ∧ (x = 6) := 
begin 
    sorry 
end

end colin_first_mile_time_l353_353510


namespace lcm_5_6_10_12_l353_353947

open Nat

theorem lcm_5_6_10_12 : lcm (lcm (lcm 5 6) 10) 12 = 60 := by
  sorry

end lcm_5_6_10_12_l353_353947


namespace variance_defect_rate_l353_353394

noncomputable def defect_rate : ℝ := 0.02
noncomputable def number_of_trials : ℕ := 100
noncomputable def variance_binomial (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem variance_defect_rate :
  variance_binomial number_of_trials defect_rate = 1.96 :=
by
  sorry

end variance_defect_rate_l353_353394


namespace area_of_rectangle_is_588_l353_353071

-- Define the conditions
def radius_of_circle := 7
def width_of_rectangle := 2 * radius_of_circle
def length_to_width_ratio := 3

-- Define the width and length of the rectangle based on the conditions
def width := width_of_rectangle
def length := length_to_width_ratio * width_of_rectangle

-- Define the area of the rectangle
def area_of_rectangle := length * width

-- The theorem to prove
theorem area_of_rectangle_is_588 : area_of_rectangle = 588 :=
by sorry -- Proof is not required

end area_of_rectangle_is_588_l353_353071


namespace log_4_8_l353_353142

theorem log_4_8 : log 4 8 = 3 / 2 :=
by
  -- Sorry is used to skip the proof
  sorry

end log_4_8_l353_353142


namespace count_integers_between_sqrt_10_and_sqrt_100_l353_353721

theorem count_integers_between_sqrt_10_and_sqrt_100 :
  ∃ n : ℕ, (∀ k : ℕ, 4 ≤ k ∧ k ≤ 10 → k ∈ set.Ico (int.floor (sqrt (10 : ℝ))) (int.ceil (sqrt (100 : ℝ)))) ∧ n = 7 :=
sorry

end count_integers_between_sqrt_10_and_sqrt_100_l353_353721


namespace greatest_nat_not_sum_of_two_composites_l353_353556

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

theorem greatest_nat_not_sum_of_two_composites :
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ 11 = a + b ∧
  (∀ n : ℕ, n > 11 → ¬ ∃ x y : ℕ, is_composite x ∧ is_composite y ∧ n = x + y) :=
sorry

end greatest_nat_not_sum_of_two_composites_l353_353556


namespace segments_count_bound_l353_353336

-- Define the overall setup of the problem
variable (n : ℕ) (points : Finset ℕ)

-- The main hypothesis and goal
theorem segments_count_bound (hn : n ≥ 2) (hpoints : points.card = 3 * n) :
  ∃ A B : Finset (ℕ × ℕ), (∀ (i j : ℕ), i ∈ points → j ∈ points → i ≠ j → ((i, j) ∈ A ↔ (i, j) ∉ B)) ∧
  ∀ (X : Finset ℕ) (hX : X.card = n), ∃ C : Finset (ℕ × ℕ), (C ⊆ A) ∧ (X ⊆ points) ∧
  (∃ count : ℕ, count ≥ (n - 1) / 6 ∧ count = C.card ∧ ∀ (a b : ℕ), (a, b) ∈ C → a ∈ X ∧ b ∈ points \ X) := sorry

end segments_count_bound_l353_353336


namespace not_axisymmetric_sqrt_x_l353_353039

theorem not_axisymmetric_sqrt_x :
  ¬ (is_axisymmetric (fun x => sqrt x)) :=
by
  sorry

end not_axisymmetric_sqrt_x_l353_353039


namespace binom_eq_fraction_l353_353922

open Nat

theorem binom_eq_fraction (n : ℕ) (h_pos : 0 < n) : choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_eq_fraction_l353_353922


namespace regular_polygon_sides_l353_353242

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (interior_angle : ℝ) i = 140) : n = 9 := by
  sorry

end regular_polygon_sides_l353_353242


namespace coefficient_of_x_squared_l353_353680

noncomputable def find_a (h : (2 + a)^5 = -1) : ℝ := a

theorem coefficient_of_x_squared 
  (a : ℝ) 
  (h_sum_of_coeff : (2 + a)^5 = -1) 
  : (coeff_of_x_squared ((x + 1/x + a)^5) = -330) := 
sorry

end coefficient_of_x_squared_l353_353680


namespace largest_integer_less_than_100_with_remainder_5_l353_353630

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l353_353630


namespace larger_hexagon_in_ABD_l353_353056

-- Define the conditions in the problem.
variables (A B C D : Point)
  (hex_in_ABD hex_in_CBD : RegularHexagon)
  (inscribed_ABD : hex_in_ABD.inscribed_in (Triangle.mk A B D))
  (inscribed_CBD : hex_in_CBD.inscribed_in (Triangle.mk C B D))

-- Define the property to be proved.
theorem larger_hexagon_in_ABD :
  (area hex_in_ABD) > (area hex_in_CBD)
:= 
sorry

end larger_hexagon_in_ABD_l353_353056


namespace tangent_line_equation_l353_353552

theorem tangent_line_equation :
  (∃ l : ℝ → ℝ, 
   (∀ x, l x = (1 / (4 + 2 * Real.sqrt 3)) * x + (2 + Real.sqrt 3) / 2 ∨ 
         l x = (1 / (4 - 2 * Real.sqrt 3)) * x + (2 - Real.sqrt 3) / 2) ∧ 
   (l 1 = 2) ∧ 
   (∀ x, l x = Real.sqrt x)
  ) →
  (∀ x y, 
   (y = (1 / 4 + Real.sqrt 3) * x + (2 + Real.sqrt 3) / 2 ∨ 
    y = (1 / 4 - Real.sqrt 3) * x + (2 - Real.sqrt 3) / 2) ∨ 
   (x - (4 + 2 * Real.sqrt 3) * y + (7 + 4 * Real.sqrt 3) = 0 ∨ 
    x - (4 - 2 * Real.sqrt 3) * y + (7 - 4 * Real.sqrt 3) = 0)
) :=
sorry

end tangent_line_equation_l353_353552


namespace four_points_convex_quadrilateral_intersection_of_segments_l353_353062

-- Problem (1)
theorem four_points_convex_quadrilateral (points : Fin 5 → ℝ × ℝ) 
  (h_collinear : ∀ (i j k : Fin 5), i ≠ j → j ≠ k → i ≠ k → ¬ collinear [points i, points j, points k]) :
  ∃ (a b c d : Fin 5), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ d ∧ convex_quadrilateral (points a) (points b) (points c) (points d) :=
sorry

-- Problem (2)
theorem intersection_of_segments (n : ℕ) (h : 0 < n) (points : Fin (4 * n + 1) → ℝ × ℝ)
  (h_collinear : ∀ (i j k : Fin (4 * n + 1)), i ≠ j → j ≠ k → i ≠ k → ¬ collinear [points i, points j, points k]) :
  ∃ (pairs : Fin (4 * n) → (Fin (4 * n + 1) × Fin (4 * n + 1))),
  (∀ i j, i ≠ j → (pairs i).1 ≠ (pairs j).1 ∧ (pairs i).2 ≠ (pairs j).2 ∧ (pairs i).1 ≠ (pairs j).2) ∧
  (count_different_intersections (pairs_image points pairs) ≥ n) :=
sorry

end four_points_convex_quadrilateral_intersection_of_segments_l353_353062


namespace min_x2_y2_z2_l353_353654

def min_value (f : ℝ × ℝ × ℝ → ℝ) (cond : ℝ × ℝ × ℝ → Prop) : ℝ :=
  Inf {y | ∃ x, cond x ∧ f x = y}

theorem min_x2_y2_z2 (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : x^2 + y^2 + z^2 ≥ 1 / 14 :=
by
  sorry

example : min_value (λ x, x.1^2 + x.2^2 + x.3^2) (λ x, x.1 + 2 * x.2 + 3 * x.3 = 1) = 1 / 14 :=
by
  sorry

end min_x2_y2_z2_l353_353654


namespace complementary_is_sufficient_but_not_necessary_l353_353917

-- Definitions used:
def mutually_exclusive (A B : Set) := A ∩ B = ∅
def complementary (A B : Set) := A ∪ B = Universe ∧ A ∩ B = ∅

-- Definition of sufficient (but not necessary) condition:
theorem complementary_is_sufficient_but_not_necessary (A B : Set) :
  complementary A B → mutually_exclusive A B ∧ ¬ (mutually_exclusive A B → complementary A B) := 
sorry

end complementary_is_sufficient_but_not_necessary_l353_353917


namespace median_ride_times_correct_l353_353092

def ride_times : List ℕ := [22, 35, 45, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 178, 185, 195, 205, 215]

def median (l : List ℕ) : Float :=
  let sorted := l.qsort (· < ·)
  let n := sorted.length
  if n % 2 = 0 then
    let mid1 := sorted.get! ((n / 2) - 1)
    let mid2 := sorted.get! (n / 2)
    (mid1 + mid2) / 2
  else
    sorted.get! (n / 2)

theorem median_ride_times_correct : median ride_times = 115 := by
  sorry

end median_ride_times_correct_l353_353092


namespace polygon_sides_150_diagonals_l353_353996

theorem polygon_sides_150_diagonals :
  ∃ n : ℕ, 150 = n * (n - 3) / 2 ∧ n = 20 :=
begin
  sorry
end

end polygon_sides_150_diagonals_l353_353996


namespace fraction_area_of_triangles_l353_353426

theorem fraction_area_of_triangles 
  (base_PQR : ℝ) (height_PQR : ℝ)
  (base_XYZ : ℝ) (height_XYZ : ℝ)
  (h_base_PQR : base_PQR = 3)
  (h_height_PQR : height_PQR = 2)
  (h_base_XYZ : base_XYZ = 6)
  (h_height_XYZ : height_XYZ = 3) :
  (1/2 * base_PQR * height_PQR) / (1/2 * base_XYZ * height_XYZ) = 1 / 3 :=
by
  sorry

end fraction_area_of_triangles_l353_353426


namespace complex_plane_region_l353_353034

noncomputable def z (x y : ℝ) : ℂ := x + complex.i * y

theorem complex_plane_region (x y : ℝ) (h : 2 * x * y > 2) : x * y > 1 :=
by
  sorry

end complex_plane_region_l353_353034


namespace total_crayons_l353_353048

theorem total_crayons (crayons_per_child : ℕ) (number_of_children : ℕ) (h1 : crayons_per_child = 5) (h2 : number_of_children = 10) : 
  crayons_per_child * number_of_children = 50 :=
by
  rw [h1, h2]
  norm_num
  done

end total_crayons_l353_353048


namespace domain_of_v_l353_353438

-- Definitions based on conditions
def sqrt_real (x : ℝ) : Prop := x ≥ 0
def sqrt_x_plus_4_real (x : ℝ) : Prop := x + 4 ≥ 0
def denominator_nonzero (x : ℝ) : Prop := sqrt x + sqrt (x + 4) ≠ 0

-- Main proof problem
theorem domain_of_v :
  (∀ x : ℝ, sqrt_real x ∧ sqrt_x_plus_4_real x → 
    denominator_nonzero x →
    (x ∈ set.Ici 0)) :=
by
  intro x
  simp [sqrt_real, sqrt_x_plus_4_real, denominator_nonzero]
  intro hx1 hx2 hx3
  exact hx1

end domain_of_v_l353_353438


namespace eight_faucets_fill_time_in_seconds_l353_353645

open Nat

-- Definitions under the conditions
def four_faucets_rate (gallons : ℕ) (minutes : ℕ) : ℕ := gallons / minutes

def one_faucet_rate (four_faucets_rate : ℕ) : ℕ := four_faucets_rate / 4

def eight_faucets_rate (one_faucet_rate : ℕ) : ℕ := one_faucet_rate * 8

def time_to_fill (rate : ℕ) (gallons : ℕ) : ℕ := gallons / rate

-- Main theorem to prove 
theorem eight_faucets_fill_time_in_seconds (gallons_tub : ℕ) (four_faucets_time : ℕ) :
    let four_faucets_rate := four_faucets_rate 200 8
    let one_faucet_rate := one_faucet_rate four_faucets_rate
    let rate_eight_faucets := eight_faucets_rate one_faucet_rate
    let time_fill := time_to_fill rate_eight_faucets 50
    gallons_tub = 50 ∧ four_faucets_time = 8 ∧ rate_eight_faucets = 50 -> time_fill * 60 = 60 :=
by
    intros
    sorry

end eight_faucets_fill_time_in_seconds_l353_353645


namespace area_code_count_l353_353507

theorem area_code_count : (list.permutations [4, 5, 6]).count (λ l, l.head = some 5 ∨ l.head = some 6) = 8 := 
sorry

end area_code_count_l353_353507


namespace container_alcohol_amount_l353_353983

theorem container_alcohol_amount
  (A : ℚ) -- Amount of alcohol in quarts
  (initial_water : ℚ) -- Initial amount of water in quarts
  (added_water : ℚ) -- Amount of water added in quarts
  (final_ratio_alcohol_to_water : ℚ) -- Final ratio of alcohol to water
  (h_initial_water : initial_water = 4) -- Container initially contains 4 quarts of water.
  (h_added_water : added_water = 8/3) -- 2.666666666666667 quarts of water added.
  (h_final_ratio : final_ratio_alcohol_to_water = 3/5) -- Final ratio is 3 parts alcohol to 5 parts water.
  (h_final_water : initial_water + added_water = 20/3) -- Total final water quarts after addition.
  : A = 4 := 
sorry

end container_alcohol_amount_l353_353983


namespace minimal_n_for_real_root_l353_353696

noncomputable def P (n : ℕ) (coeffs : Fin (2 * n + 1) → ℝ) (x : ℝ) : ℝ :=
  ∑ i in Finset.range (2 * n + 1), coeffs ⟨i, by linarith⟩ * x ^ i

theorem minimal_n_for_real_root :
  ∃ (n : ℕ) (coeffs : Fin (2 * n + 1) → ℝ),
    (∀ i, coeffs i ∈ Set.Icc 100 101) ∧ 
    (∃ x : ℝ, P n coeffs x = 0) ∧
    ∀ m : ℕ, m < n → ¬ (∃ (coeffs' : Fin (2 * m + 1) → ℝ), 
      (∀ i, coeffs' i ∈ Set.Icc 100 101) ∧ 
      (∃ x : ℝ, P m coeffs' x = 0)) :=
  sorry

end minimal_n_for_real_root_l353_353696


namespace sum_of_scores_with_exactly_two_ways_is_correct_l353_353764

noncomputable def sum_of_specific_scores : ℕ :=
  (let conditions (c u i : ℕ) (S : ℕ) :=
    (c + u + i = 30) ∧
    (S = 8 * c + 3 * u) ∧
    (0 ≤ c) ∧ (0 ≤ u) ∧ (0 ≤ i) ∧
    (0 ≤ S ∧ S ≤ 240) in
  let valid_scores := { S | ∃ c u i, conditions c u i S ∧
    (∃! (x, y, z : ℕ), conditions x y z S) } in
  (Finset.univ.filter (λ S, S ∈ valid_scores)).sum id
  )

theorem sum_of_scores_with_exactly_two_ways_is_correct :
  sum_of_specific_scores = ? := sorry

end sum_of_scores_with_exactly_two_ways_is_correct_l353_353764


namespace probability_of_divisibility_l353_353914

noncomputable def probability_smaller_divides_larger : ℚ :=
  let S := {1, 2, 3, 4, 5, 6}
  let pairs := { (x, y) | x ∈ S ∧ y ∈ S ∧ x < y }
  let successful_pairs := { (x, y) ∈ pairs | x ∣ y }
  (successful_pairs.card : ℚ) / pairs.card

theorem probability_of_divisibility :
  probability_smaller_divides_larger = 7 / 15 :=
by
  sorry

end probability_of_divisibility_l353_353914


namespace min_distance_l353_353296

def triangle_ABC (A B C D : Type) (BD : ℝ) :=
∃ (CD : ℝ), CD = 2 * BD ∧ 
  ∃ (AD AC AB : ℝ), AD = 2 ∧ 
    ∃ (α β γ : ℝ), α = 120 ∧ 
      angle_ADB = α ∧ 
        (AC / AB = (sqrt (3) - 1))

theorem min_distance (A B C D : Type) (BD : ℝ) :
  ∃ D ∈ BC, ∃ α β γ AD AC AB, 
    α = 120 ∧ AD = 2 ∧ D ∈ BC ∧ 
      γ = ∠ADB = α ∧ 
        CD = 2 * BD →
          (AC / AB) = (sqrt (3) - 1) := 
sorry

end min_distance_l353_353296


namespace arc_length_calc_l353_353265

-- Defining the conditions
def circle_radius := 12 -- radius OR
def angle_RIP := 30 -- angle in degrees

-- Defining the goal
noncomputable def arc_length_RP := 4 * Real.pi -- length of arc RP

-- The statement to prove
theorem arc_length_calc :
  arc_length_RP = 4 * Real.pi :=
sorry

end arc_length_calc_l353_353265


namespace minimum_BD_value_l353_353305

theorem minimum_BD_value {A B C D : Type*} [Real A] [Real B] [Real C] [Real D] (angle_ADB: A) (AD: A) (CD: A) (BD: A):
  (D ∈ line_segment(B, C)) →
  angle_ADB = 120 → 
  AD = 2 → 
  CD = 2 * BD →
  ∃ BD: A, BD = sqrt(3) - 1 :=
by
  sorry

end minimum_BD_value_l353_353305


namespace old_edition_pages_l353_353080

-- Define the conditions
variables (new_edition : ℕ) (old_edition : ℕ)

-- The conditions given in the problem
axiom new_edition_pages : new_edition = 450
axiom pages_relationship : new_edition = 2 * old_edition - 230

-- Goal: Prove that the old edition Geometry book had 340 pages
theorem old_edition_pages : old_edition = 340 :=
by sorry

end old_edition_pages_l353_353080


namespace laser_path_total_distance_l353_353988

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem laser_path_total_distance :
  let A := (2, 4 : ℝ)
  let B := (0, 4 : ℝ)
  let C := (0, -4 : ℝ)
  let D := (8, 4 : ℝ)
  distance A B + distance B C + distance C D = 10 + 8 * real.sqrt 2 :=
by {
  -- Proof would go here
  sorry
}

end laser_path_total_distance_l353_353988


namespace number_of_integers_between_sqrt10_sqrt100_l353_353735

theorem number_of_integers_between_sqrt10_sqrt100 : 
  (set.Ico ⌈Real.sqrt 10⌉₊ (⌊Real.sqrt 100⌋₊ + 1)).card = 7 := by
  sorry

end number_of_integers_between_sqrt10_sqrt100_l353_353735


namespace monotonicity_of_f_existence_of_local_minimum_and_nonnegativity_l353_353061

-- Part 1: Monotonicity of f(x) = (1 + x + a * x^2) * exp(-x) for a ≥ 0

section
variable (a : ℝ) (h_a : 0 ≤ a)

def f (x : ℝ) : ℝ := (1 + x + a * x^2) * exp(-x)

theorem monotonicity_of_f (x : ℝ) : 
  if a = 0 then (f x : (-∞, 0) → increasing ∧ (0, ∞) → decreasing)
  else if a = 1/2 then (∀ x, decreasing f)
  else if a > 1/2 then (f x : (-∞, 0) ∪ (2 - 1/a, ∞) → decreasing ∧ (0, 2 - 1/a) → increasing)
  else (f x : (2 - 1/a, 0) → increasing ∧ (0, ∞) → decreasing) :=
sorry
end

-- Part 2: Existence of m such that g(x) = exp(x) - m * x - cos x has a local minimum at x = 0 and g(x) ≥ 0 for all x

section
variable (m : ℝ) (h_m : 0 ≤ m)

def g (x : ℝ) : ℝ := exp x - m * x - cos x

theorem existence_of_local_minimum_and_nonnegativity (x : ℝ) :
  (∃ m, g x = 0 ∧ g'(0) = 0) ∧ (∀ x, g x ≥ 0) :=
  let m = 1 in
  ∃ x₀, x₀ = 0 ∧ g x₀ has a local minimum ∧ ∀ x, g x ≥ 0 :=
sorry
end

end monotonicity_of_f_existence_of_local_minimum_and_nonnegativity_l353_353061


namespace triangle_perimeter_l353_353206

theorem triangle_perimeter (a b c : ℝ) (h1 : a = 2) (h2 : (b-2)^2 + |c-3| = 0) : a + b + c = 7 :=
by
  sorry

end triangle_perimeter_l353_353206


namespace probability_same_value_after_reroll_l353_353376

theorem probability_same_value_after_reroll
  (initial_dice : Fin 6 → Fin 6)
  (rerolled_dice : Fin 4 → Fin 6)
  (initial_pair_num : Fin 6)
  (h_initial_no_four_of_a_kind : ∀ (n : Fin 6), (∃ i j : Fin 6, i ≠ j ∧ initial_dice i = n ∧ initial_dice j = n) →
    ∃ (i₁ i₂ i₃ i₄ : Fin 6), i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₃ ≠ i₄ ∧
    initial_dice i₁ ≠ n ∧ initial_dice i₂ ≠ n ∧ initial_dice i₃ ≠ n ∧ initial_dice i₄ ≠ n)
  (h_initial_pair : ∃ i j : Fin 6, i ≠ j ∧ initial_dice i = initial_pair_num ∧ initial_dice j = initial_pair_num) :
  (671 : ℚ) / 1296 = 671 / 1296 :=
by sorry

end probability_same_value_after_reroll_l353_353376


namespace area_ratio_l353_353774

noncomputable def area_of_polygon {α : Type*} [metric_space α] [measurable_space α] (vertices : list α) : ℝ :=
sorry

def regular_hexagon (A B C D E F : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D E ∧ dist D E = dist E F ∧ dist E F = dist F A ∧
  ∀ X Y Z : ℝ × ℝ, X ∈ {A, B, C, D, E, F} ∧ Y ∈ {A, B, C, D, E, F} ∧ Z ∈ {A, B, C, D, E, F} → angle X Y Z = π/3

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

variables {A B C D E F : ℝ × ℝ}
variables (h : regular_hexagon A B C D E F)

def M := midpoint B C
def N := midpoint E F

theorem area_ratio :
  area_of_polygon [A, B, M] / area_of_polygon [C, D, M, N] = 1 / 2 :=
sorry

end area_ratio_l353_353774


namespace Winnie_the_Pooh_stationary_escalator_steps_l353_353961

theorem Winnie_the_Pooh_stationary_escalator_steps
  (u v L : ℝ)
  (cond1 : L * u / (u + v) = 55)
  (cond2 : L * u / (u - v) = 1155) :
  L = 105 := by
  sorry

end Winnie_the_Pooh_stationary_escalator_steps_l353_353961


namespace sum_squares_sum_cubes_sum_n_calculate_expr_l353_353019

-- Define the sum of squares formula
theorem sum_squares (n : ℕ) : ∑ i in Finset.range (n + 1), i^2 = n * (n + 1) * (2 * n + 1) / 6 := sorry

-- Define the sum of cubes formula
theorem sum_cubes (n : ℕ) : ∑ i in Finset.range (n + 1), i^3 = (n * (n + 1) / 2)^2 := sorry

-- Define the sum of first n natural numbers
theorem sum_n (n : ℕ) : ∑ i in Finset.range (n + 1), i = n * (n + 1) / 2 := sorry

-- Main theorem to calculate the given expression
theorem calculate_expr : 
  ∑ k in Finset.range (100), (k^3 + 3 * k^2 + 3 * k) = 25502400 := 
by
  -- Use the sum of cubes
  have sum_cubes := sum_cubes 99,
  
  -- Use the sum of squares
  have sum_squares := sum_squares 99,
  
  -- Use the sum of first n natural numbers
  have sum_n := sum_n 99,
  
  -- Calculate the expression using the sums
  calc 
    ∑ k in Finset.range (100), (k^3 + 3 * k^2 + 3 * k)
      = (∑ k in Finset.range (100), k^3) + 3 * (∑ k in Finset.range (100), k^2) + 3 * (∑ k in Finset.range (100), k) : by apply Finset.sum_add_distrib
    ... = 24502500 + 985050 + 14850 : by {rw [sum_cubes, sum_squares, sum_n], sorry}


end sum_squares_sum_cubes_sum_n_calculate_expr_l353_353019


namespace largest_integer_less_than_100_with_remainder_5_l353_353578

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353578


namespace maximum_m_value_l353_353691

noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 - 2 * x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem maximum_m_value :
  (∀ x > 1, 2 * f' x + x * g x + 3 > m * (x - 1)) → m ≤ 4 :=
by
  sorry

end maximum_m_value_l353_353691


namespace find_line_eq_l353_353190

theorem find_line_eq (l : ℝ → ℝ → Prop) :
  (∃ A B : ℝ × ℝ, l A.fst A.snd ∧ l B.fst B.snd ∧ ((A.fst + 1)^2 + (A.snd - 2)^2 = 100 ∧ (B.fst + 1)^2 + (B.snd - 2)^2 = 100)) ∧
  (∃ M : ℝ × ℝ, M = (-2, 3) ∧ (l M.fst M.snd)) →
  (∀ x y : ℝ, l x y ↔ x - y + 5 = 0) :=
by
  sorry

end find_line_eq_l353_353190


namespace correct_propositions_l353_353681

-- Define the propositions as conditions
def proposition1 (x : ℝ) : Prop := y = 2 * sin (2 * x - π / 3) → x = 5 * π / 12 → y = 1

def proposition2 : Prop := ∀ y, y = tan x → (x, y) = (π / 2, 0) → y symmetry

def proposition3 : Prop := ∀ θ, 0 < θ ∧ θ < π / 2 → sin θ is increasing

def proposition4 : Prop := ∀ x1 x2, sin (2 * x1 - π / 4) = sin (2 * x2 - π / 4) → x1 - x2 = kπ ∨ x1 + x2 = kπ + 3π / 4

-- Provide the lean statement for the given proof problem
theorem correct_propositions : 
  (proposition1 ∧ proposition2) ∧ ¬proposition3 ∧ ¬proposition4 :=
by sorry

end correct_propositions_l353_353681


namespace debate_team_has_11_boys_l353_353900

def debate_team_boys_count (num_groups : Nat) (members_per_group : Nat) (num_girls : Nat) : Nat :=
  let total_members := num_groups * members_per_group
  total_members - num_girls

theorem debate_team_has_11_boys :
  debate_team_boys_count 8 7 45 = 11 :=
by
  sorry

end debate_team_has_11_boys_l353_353900


namespace area_of_quadrilateral_ABCD_l353_353783

-- Define the given conditions and required proof
theorem area_of_quadrilateral_ABCD (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (distance : A → A → ℝ) 
  (HA1 : ∠BCD = π/2) (HA2 : ∠BAD = π/2)
  (HAB : distance A B = 15) (HBC : distance B C = 5)
  (HCD : distance C D = 12) (HAC : distance A C = 13) :
  area_of_quadrilateral A B C D = 127.5 :=
sorry

end area_of_quadrilateral_ABCD_l353_353783


namespace largest_integer_lt_100_with_remainder_5_div_8_l353_353594

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l353_353594


namespace largest_integer_less_than_hundred_with_remainder_five_l353_353586

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l353_353586


namespace problem1_problem2_l353_353116

theorem problem1 : ((1 / 2 - 1 / 4 + 1 / 8) * -24 = -9) := by
  sorry

theorem problem2 : ((-2)^3 * 0.25 + 4 / |(-1 / 8)| - 40 = -10) := by
  sorry

end problem1_problem2_l353_353116


namespace largest_integer_less_than_hundred_with_remainder_five_l353_353585

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l353_353585


namespace unique_positive_solution_l353_353159

theorem unique_positive_solution (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  (cos (arcsin (cot (arccos x)))) = x ↔ x = 1 := 
by sorry

end unique_positive_solution_l353_353159


namespace sequence_is_arithmetic_l353_353801

theorem sequence_is_arithmetic {a : ℕ → ℝ} (h_pos : ∀ n, a n > 0)
  (h_second_term : a 2 = 3 * a 1)
  (h_sqrt_seq_arith : ∃ d : ℝ, ∀ n, real.sqrt (∑ i in finset.range (n + 1), a i) = d * n + real.sqrt (a 0)): 
  ∃ d : ℝ, ∀ n, a n = a 0 + d * n := 
by
  sorry

end sequence_is_arithmetic_l353_353801


namespace problem_l353_353347

noncomputable def S (k : ℕ → ℕ) (n : ℕ) : ℕ := (List.range n).sum (λ i => 2 ^ (k i))

theorem problem (m n : ℕ) (k : ℕ → ℕ) :
  (∀ i j, i < n → j < n → i ≠ j → k i ≠ k j) →
  (S k n) % (2 ^ m - 1) = 0 →
  n ≥ m :=
by
  sorry

end problem_l353_353347


namespace negation_proof_l353_353405

open Real

theorem negation_proof :
  (¬ ∃ x : ℕ, exp x - x - 1 ≤ 0) ↔ (∀ x : ℕ, exp x - x - 1 > 0) :=
by
  sorry

end negation_proof_l353_353405


namespace unknown_towel_rate_l353_353495

theorem unknown_towel_rate (x : ℝ) (towels_total : ℝ → ℝ) (avg_price : ℝ) (num_towels : ℝ) : 
  ( ∀ a b c : ℝ, towels_total a + towels_total b + 2 * x = 10 * avg_price → 
  towels_total 90 100 3 ∧ towels_total 142.5 150 5 ∧ avg_price = 145 ∧ num_towels = 10 )
  = 199.23 := 
begin
  sorry
end

end unknown_towel_rate_l353_353495


namespace distinct_pos_integers_inequality_l353_353833

theorem distinct_pos_integers_inequality (n : Nat) (a : Fin n → ℕ)
  (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j)
  (h_positive : ∀ i : Fin n, 0 < a i) :
  (∑ i, (a i)^7) + (∑ i, (a i)^5) ≥ 2 * (∑ i, (a i)^3)^2 :=
by
  sorry

end distinct_pos_integers_inequality_l353_353833


namespace total_cups_of_mushroom_soup_l353_353866

def cups_team_1 : ℕ := 90
def cups_team_2 : ℕ := 120
def cups_team_3 : ℕ := 70

theorem total_cups_of_mushroom_soup :
  cups_team_1 + cups_team_2 + cups_team_3 = 280 :=
  by sorry

end total_cups_of_mushroom_soup_l353_353866


namespace B_not_in_middle_probability_l353_353420

-- Define the entities involved: three people, A, B, and C
inductive Person
| A | B | C

open Person

-- Define the probability of B not sitting in the middle
def prob_B_not_middle : ℚ := 2 / 3

-- Main statement of the problem
theorem B_not_in_middle_probability :
  let seats := [A, B, C].permutations in
  let middle_seat_B := [A, B, C].permutations.filter (fun l => l.nth 1 = some B) in
  prob_B_not_middle = 1 - ((middle_seat_B.length : ℚ) / (seats.length : ℚ)) :=
by
  sorry

end B_not_in_middle_probability_l353_353420


namespace largest_int_with_remainder_5_lt_100_l353_353567

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l353_353567


namespace find_BD_when_AC_over_AB_min_l353_353278

open Real

-- Define the conditions
variables (A B C D : Point)
variables (AD BC BD CD : ℝ)
variables (angle_ADB : ℝ)
variables (AC AB : ℝ)

-- Assign conditions
def conditions : Prop := 
  angle_ADB = 120 ∧
  AD = 2 ∧
  CD = 2 * BD ∧
  AC / AB = minimized

-- Target to prove
theorem find_BD_when_AC_over_AB_min (A B C D : Point) (BD : ℝ)
  (h : conditions A B C D 2 120 BD 2 * BD AC AB) :
  BD = sqrt 3 - 1 :=
sorry

end find_BD_when_AC_over_AB_min_l353_353278


namespace project_completion_days_l353_353471

open Real

noncomputable def work_rate_A := (1 : ℝ) / 20
noncomputable def work_rate_B := (1 : ℝ) / 30
noncomputable def combined_work_rate := work_rate_A + work_rate_B
noncomputable def B_single_contrib := 5 * work_rate_B
noncomputable def total_days := 15

theorem project_completion_days :
  (∃ x : ℝ, x * combined_work_rate + B_single_contrib = 1 ∧ x + 5 = total_days) :=
by
  use 10
  split
  { 
    calc
      10 * combined_work_rate + B_single_contrib 
        = 10 * (work_rate_A + work_rate_B) + B_single_contrib : by rw [combined_work_rate]
    ... = 10 * (1 / 20 + 1 / 30) + 5 * (1 / 30) : by rw [work_rate_A, work_rate_B]
    ... = 10 * (3 / 60 + 2 / 60) + 5 * (1 / 30) : by norm_num
    ... = 10 * 5 / 60 + 5 * 1 / 30 : by norm_num
    ... = 1 : by norm_num },
  {
    exact rfl
  }

end project_completion_days_l353_353471


namespace proof_f_ff_value_l353_353687

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 1/x - 100

theorem proof_f_ff_value :
  f (f (1/100)) = 1 :=
  sorry

end proof_f_ff_value_l353_353687


namespace y_intercept_of_line_l353_353000

-- Define the line equation
def line_eq (x y : ℚ) : Prop := x - 2 * y - 3 = 0

-- Define the y_intercept function that finds the y-value when x is 0
def y_intercept (L : ℚ → ℚ → Prop) : ℚ :=
  if h : ∃ y, L 0 y then classical.some h else 0

-- Define the theorem to prove the y-intercept equals -3 / 2
theorem y_intercept_of_line : y_intercept line_eq = -3/2 :=
by { sorry }

end y_intercept_of_line_l353_353000


namespace locus_angle_bisector_l353_353949

structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

def distance_point_line (P : ℝ × ℝ) (L : Line) : ℝ :=
  abs (L.A * P.1 + L.B * P.2 + L.C) / sqrt (L.A ^ 2 + L.B ^ 2)

theorem locus_angle_bisector (L1 L2 : Line) (d : ℝ) :
  {P : ℝ × ℝ | abs (distance_point_line P L1 - distance_point_line P L2) = d} =
  {P : ℝ × ℝ | distance_point_line P (Line.mk (L1.A + L2.A / 2) (L1.B + L2.B / 2) (L1.C + L2.C / 2)) = 0} :=
by
  sorry

end locus_angle_bisector_l353_353949


namespace binom_n_2_l353_353936

theorem binom_n_2 (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_n_2_l353_353936


namespace baptiste_socks_problem_l353_353504

def num_red_socks := 6
def num_blue_socks := 13
def num_white_socks := 8

theorem baptiste_socks_problem : ∃ n : ℕ, n = 4 :=
by 
  have h1 : num_red_socks = 6 := rfl
  have h2 : num_blue_socks = 13 := rfl
  have h3 : num_white_socks = 8 := rfl
  use 4
  sorry

end baptiste_socks_problem_l353_353504


namespace sarah_pencils_on_tuesday_l353_353863

theorem sarah_pencils_on_tuesday 
    (x : ℤ)
    (h1 : 20 + x + 3 * x = 92) : 
    x = 18 := 
by 
    sorry

end sarah_pencils_on_tuesday_l353_353863


namespace sum_of_real_numbers_for_conditions_l353_353955

theorem sum_of_real_numbers_for_conditions :
  (∀ x : ℝ, |x^2 - 14 * x + 40| = 3 ∧ x^2 - 14 * x + 45 = 0 → false) :=
begin
  sorry
end

end sum_of_real_numbers_for_conditions_l353_353955


namespace modulus_squared_eq_r_squared_l353_353849

-- Given conditions
variables {z : ℂ} (hz : z.im > 0) (A : ℝ) (H : parallelogram_area 0 z (1/z) (z + 1/z) = 12/13)

-- Required to prove
theorem modulus_squared_eq_r_squared (r : ℝ) : ∃ r : ℝ, |z|^2 = r^2 :=
by 
  sorry -- proof placeholder

end modulus_squared_eq_r_squared_l353_353849


namespace percent_students_at_trip_l353_353965

variable (total_students : ℕ)
variable (students_taking_more_than_100 : ℕ := (14 * total_students) / 100)
variable (students_not_taking_more_than_100 : ℕ := (75 * total_students) / 100)
variable (students_who_went_to_trip := (students_taking_more_than_100 * 100) / 25)

/--
  If 14 percent of the students at a school went to a camping trip and took more than $100,
  and 75 percent of the students who went to the camping trip did not take more than $100,
  then 56 percent of the students at the school went to the camping trip.
-/
theorem percent_students_at_trip :
    (students_who_went_to_trip * 100) / total_students = 56 :=
sorry

end percent_students_at_trip_l353_353965


namespace max_value_NPN_l353_353957

theorem max_value_NPN (M : ℕ) (NPN : ℕ) :
  M < 10 → 
  11 * M * M = NPN →
  (∃ N : ℕ, ∃ P : ℕ, 
    (first_digit NPN = N) ∧ 
    (last_digit NPN = P) ∧ 
    (N = P) ∧ 
    (NPN = M^3)) →
  NPN = 729 :=
by
  sorry

-- Helper functions to find the first and last digits. These can be defined as needed.
def first_digit (n : ℕ) : ℕ := sorry -- Define appropriately
def last_digit (n : ℕ) : ℕ := sorry -- Define appropriately

end max_value_NPN_l353_353957


namespace min_distance_MN_l353_353223

theorem min_distance_MN : 
  ∃ m : ℝ, m > 0 ∧ m = (1:ℝ) / (real.sqrt (3:ℝ)) ∧ 
  let F := fun x : ℝ => x^3 - real.log x in 
  F ((1:ℝ) / (real.sqrt (3:ℝ))) = (1:ℝ / 3) * (1 + real.log 3) :=
sorry

end min_distance_MN_l353_353223


namespace sum_of_sequence_l353_353664

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) + 1 = 2 * (a n + 1)

def function_zero (f : ℝ → ℝ) : Prop :=
  ∀ n, (∃! x, f x = 0)

theorem sum_of_sequence (a : ℕ → ℕ) (f : ℝ → ℝ) :
  sequence a →
  function_zero (λ x, x^4 + a (n + 1) * Real.cos (2 * x) - (2 * a n + 1)) →
  (∀ n, ∑ k in Finset.range n, k * (a k + 1) = (n - 1) * 2^(n + 1) + 2) :=
begin
  intros,
  sorry
end

end sum_of_sequence_l353_353664


namespace periodic_function_proof_l353_353215

theorem periodic_function_proof :
  (∃ f : ℕ → ℚ, f(1) = 2 ∧ (∀ x, f(x + 1) = (1 + f(x)) / (1 - f(x))) ∧ f(2018) + f(2019) = -7 / 2) :=
by
  -- Here, we will construct the function f as a rational function of natural numbers (ℕ → ℚ), based on the provided conditions.
  let f : ℕ → ℚ := sorry   -- Define the sequence appropriately and prove the conditions using mathematical steps.
  existsi f
  split
  -- Prove that f(1) = 2
  { sorry }
  
  split
  -- Prove that ∀ x, f(x + 1) = (1 + f(x)) / (1 - f(x))
  { intro x
    sorry }
  
  -- Prove that f(2018) + f(2019) = -7 / 2
  { sorry }

end periodic_function_proof_l353_353215


namespace largest_integer_less_than_100_div_8_rem_5_l353_353613

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l353_353613


namespace largest_int_less_than_100_with_remainder_5_l353_353561

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l353_353561


namespace find_fayes_age_l353_353528

variable {C D E F : ℕ}

theorem find_fayes_age
  (h1 : D = E - 2)
  (h2 : E = C + 5)
  (h3 : F = C + 4)
  (h4 : D = 15) :
  F = 16 := by
  sorry

end find_fayes_age_l353_353528


namespace ellipse_eccentricity_sum_l353_353398

theorem ellipse_eccentricity_sum :
  let points : List (ℝ × ℝ) := [(-1, 0), (-3, 1), (-3, -1), (-4, Real.sqrt 2), (-4, -Real.sqrt 2)] in
  let a := 2 in
  let b := 1 in
  let c := Real.sqrt (a^2 - b^2) in
  let e := c / a in
  e = Real.sqrt (3 / 4) →
  3 + 4 = 7 := 
by
  intro points a b c e heq
  sorry

end ellipse_eccentricity_sum_l353_353398


namespace golden_triangle_identity_l353_353675

theorem golden_triangle_identity :
  (\cos 10 - \cos 82) * (\sin 10 - \sin 82) ∨ 
  (\sin 173 * \cos 11 - \sin 83 * \cos 101) ∨ 
  (\sqrt{\frac{1 - \sin 54}{2}}) = \frac{\sqrt{5}-1}{4} :=
sorry

end golden_triangle_identity_l353_353675


namespace measure_one_liter_impossible_l353_353041

theorem measure_one_liter_impossible
  (four_liter_jug : ℕ)
  (six_liter_pot : ℕ)
  (big_barrel_w : ℕ)
  (fill : ℕ → ℕ)
  (pour_out : ℕ → ℕ)
  (transfer : ℕ → ℕ → ℕ × ℕ) :
  (∀ (a b : ℕ), (fill a = four_liter_jug ∨ fill a = six_liter_pot) →
  pour_out a = 0 →
  (let (new_a, new_b) := transfer a b in 
    (new_a % 2 = 0 ∧ new_b % 2 = 0)) →
    a % 2 = 0 ∧ b % 2 = 0) →
  (∀ a b, (a = 0 ∧ b = 0) → a ≠ 1 ∧ b ≠ 1) :=
by sorry

end measure_one_liter_impossible_l353_353041


namespace parabola_properties_and_intersection_l353_353694

-- Definition of the parabola C: y^2 = -4x
def parabola_C (x y : ℝ) : Prop := y^2 = -4 * x

-- Focus of the parabola
def focus_C : ℝ × ℝ := (-1, 0)

-- Equation of the directrix
def directrix_C (x: ℝ): Prop := x = 1

-- Distance from the focus to the directrix
def distance_focus_to_directrix : ℝ := 2

-- Line l passing through P(1, 2) with slope k
def line_l (k x y : ℝ) : Prop := y = k * x - k + 2

-- Main theorem statement
theorem parabola_properties_and_intersection (k: ℝ) :
  (focus_C = (-1, 0)) ∧
  (∀ x, directrix_C x ↔ x = 1) ∧
  (distance_focus_to_directrix = 2) ∧
  ((k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2) →
    ∃ x y, parabola_C x y ∧ line_l k x y ∧
    (∀ x' y', parabola_C x' y' ∧ line_l k x' y' → x = x' ∧ y = y')) ∧
  ((1 - Real.sqrt 2 < k ∧ k < 1 + Real.sqrt 2) →
    ∃ x y x' y', x ≠ x' ∧ y ≠ y' ∧
    parabola_C x y ∧ line_l k x y ∧
    parabola_C x' y' ∧ line_l k x' y') ∧
  ((k > 1 + Real.sqrt 2 ∨ k < 1 - Real.sqrt 2) →
    ∀ x y, ¬(parabola_C x y ∧ line_l k x y)) :=
by sorry

end parabola_properties_and_intersection_l353_353694


namespace product_of_four_integers_l353_353268

theorem product_of_four_integers 
  (w x y z : ℕ) 
  (h1 : x * y * z = 280)
  (h2 : w * y * z = 168)
  (h3 : w * x * z = 105)
  (h4 : w * x * y = 120) :
  w * x * y * z = 840 :=
by {
sorry
}

end product_of_four_integers_l353_353268


namespace theorem_incorrect_statement_D_l353_353216

open Real

def incorrect_statement_D (φ : ℝ) (hφ : φ > 0) (x : ℝ) : Prop :=
  cos (2*x + φ) ≠ cos (2*(x - φ/2))

theorem theorem_incorrect_statement_D (φ : ℝ) (hφ : φ > 0) : 
  ∃ x : ℝ, incorrect_statement_D φ hφ x :=
by
  sorry

end theorem_incorrect_statement_D_l353_353216


namespace middle_term_in_arithmetic_sequence_l353_353022

theorem middle_term_in_arithmetic_sequence :
  let a := 3^2 in let c := 3^4 in
  ∃ z : ℤ, (2 * z = a + c) ∧ z = 45 := by
let a := 3^2
let c := 3^4
use (a + c) / 2
split
-- Prove that 2 * ((a + c) / 2) = a + c
sorry
-- Prove that (a + c) / 2 = 45
sorry

end middle_term_in_arithmetic_sequence_l353_353022


namespace min_value_C2_sub_D2_l353_353343

noncomputable def C (u v w : ℝ) : ℝ := 
  real.sqrt (u + 3) + real.sqrt (v + 6) + real.sqrt (w + 15)

noncomputable def D (u v w : ℝ) : ℝ := 
  real.sqrt (u + 2) + real.sqrt (v + 2) + real.sqrt (w + 2)

theorem min_value_C2_sub_D2 (u v w : ℝ) (hu : 0 ≤ u) (hv : 0 ≤ v) (hw : 0 ≤ w) : 
  ∃ (m : ℝ), m = 36 :=
sorry

end min_value_C2_sub_D2_l353_353343


namespace cosine_of_angle_between_vectors_l353_353550

def point := (ℝ × ℝ × ℝ)

def ab : point := (4, 2, -3)
def ac : point := (2, -1, 2)

def dot_product (u v : point) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (v : point) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def cosine_angle (u v : point) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

theorem cosine_of_angle_between_vectors :
  let
    a := (-1, 2, -3) : point
    b := (3, 4, -6) : point
    c := (1, 1, -1) : point
  in
    cosine_angle ab ac = 0 :=
by
  sorry

end cosine_of_angle_between_vectors_l353_353550


namespace ratio_of_x_to_y_l353_353742

variable (x y : ℝ)

theorem ratio_of_x_to_y (h : 3 * x = 0.12 * 250 * y) : x / y = 10 :=
sorry

end ratio_of_x_to_y_l353_353742


namespace laura_annual_income_l353_353785

theorem laura_annual_income (I T : ℝ) (q : ℝ)
  (h1 : I > 50000) 
  (h2 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * 20000 + 0.01 * (q + 5) * (I - 50000))
  (h3 : T = 0.01 * (q + 0.5) * I) : I = 56000 := 
by sorry

end laura_annual_income_l353_353785


namespace largest_integer_less_than_100_with_remainder_5_l353_353582

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353582


namespace arithmetic_sequence_middle_term_l353_353025

theorem arithmetic_sequence_middle_term :
  ∀ (a b : ℕ) (z : ℕ), a = 9 → b = 81 → z = (a + b) / 2 → z = 45 :=
by
  intros a b z h_a h_b h_z
  rw [h_a, h_b] at h_z
  exact h_z

end arithmetic_sequence_middle_term_l353_353025


namespace min_ratio_bd_l353_353318

noncomputable def triangle_ratio_min (BD : ℝ) : Prop :=
  ∃ K L : ℝ, 
    (K = 1) ∧ 
    (L = 1) ∧ 
    (∀ (x : ℝ), ∀ (y : ℝ), (CD = 2 * BD) → ((x = sqrt(3) - 1) → y = (bd + 1 + 3 / (bd + 1)) ∧ (x + y) = 2 * K * L) ∧ 
    (abs x ≤ (sqrt 3 - 1)))


theorem min_ratio_bd (BD : ℝ) : triangle_ratio_min BD :=
by { sorry }

end min_ratio_bd_l353_353318


namespace log_sum_even_l353_353974

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- Define the condition for maximum value at x = 1
def has_max_value_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y : ℝ, f y ≤ f x

-- Main theorem statement: Prove that lg x + lg y is an even function
theorem log_sum_even (A ω φ : ℝ) (hA : 0 < A) (hω : 0 < ω) 
  (hf_max : has_max_value_at (f A ω φ) 1) : 
  ∀ x y : ℝ, Real.log x + Real.log y = Real.log y + Real.log x := by
  sorry

end log_sum_even_l353_353974


namespace percentage_bobby_pins_correct_l353_353106

noncomputable def percentage_bobby_pins : ℕ :=
  let b := 120 in -- barrettes
  let s := b / 2 in -- scrunchies
  let p := 3 * b - 50 in -- bobby pins
  let h := 2 * (p - s) in -- hairbands
  let total := b + s + p + h in -- total hair decorations
  (p * 100) / total -- percentage of bobby pins

theorem percentage_bobby_pins_correct :
  percentage_bobby_pins = 31 :=
by sorry

end percentage_bobby_pins_correct_l353_353106


namespace sqrt_49_times_sqrt_64_l353_353449

theorem sqrt_49_times_sqrt_64 : sqrt (49 * sqrt 64) = 14 * sqrt 2 :=
by
  sorry

end sqrt_49_times_sqrt_64_l353_353449


namespace emily_journey_length_l353_353532

theorem emily_journey_length :
  (total_distance : ℚ) →
  (first_quarter : ℚ := total_distance / 4) →
  (highway_part : ℚ := 24) →
  (remaining_sixth : ℚ := total_distance / 6) →
  ((first_quarter + highway_part + remaining_sixth = total_distance) →
  total_distance = 288 / 7) := 
sorry

end emily_journey_length_l353_353532


namespace cube_volume_and_surface_area_l353_353411

theorem cube_volume_and_surface_area (s : ℝ) (h : 12 * s = 72) :
  s^3 = 216 ∧ 6 * s^2 = 216 :=
by 
  sorry

end cube_volume_and_surface_area_l353_353411


namespace problem_calculations_correct_l353_353451

theorem problem_calculations_correct :
  (∀ (x y : ℝ), √x + √y ≠ √(x + y)) ∧
  (∀ (z1 z2 : ℝ), z1 * √3 - z2 * √3 ≠ 2) ∧
  (∀ a b : ℝ, √(a^2 - b^2) ≠ 1) ∧
  (2 * √5 * √5 = 10) :=
  by sorry

end problem_calculations_correct_l353_353451


namespace expression_value_l353_353174

theorem expression_value (a b c : ℝ) (h : a * b + b * c + c * a = 3) : 
  (a * (b^2 + 3)) / (a + b) + (b * (c^2 + 3)) / (b + c) + (c * (a^2 + 3)) / (c + a) = 6 := 
by
  sorry

end expression_value_l353_353174


namespace P_Q_eq_Q_P_l353_353912

noncomputable def convex_polygon (V : Type) [InnerProductSpace ℝ V] :=  -- Definition of a convex polygon
  (points : List V) (convex_hull : ConvexHull points) 

-- Variables representing the polygons P and Q.
variable {V : Type} [InnerProductSpace ℝ V] 
variable (P Q : convex_polygon V) 

-- Definition of the distance between parallel lines to a side of polygon P squeezing polygon Q
def distance_between_parallel_lines (a : V) (Q : convex_polygon V) : ℝ := sorry

-- Definition of the sum of the product l * h for each side of a convex polygon
def sum_product_lh (P Q : convex_polygon V) : ℝ :=
  ∑ (a in P.points), 
    let l := ‖a‖ in 
    let h := distance_between_parallel_lines a Q in 
    l * h

-- Definition of (P, Q) and (Q, P) as sums
def P_Q (P Q : convex_polygon V) : ℝ := 
  sum_product_lh P Q

def Q_P (Q P : convex_polygon V) : ℝ :=
  sum_product_lh Q P

-- Theorem statement that (P, Q) = (Q, P)
theorem P_Q_eq_Q_P (P Q : convex_polygon V) : P_Q P Q = Q_P Q P := 
  sorry

end P_Q_eq_Q_P_l353_353912


namespace no_solution_for_x_l353_353546

open Real

theorem no_solution_for_x (m : ℝ) : ¬ ∃ x : ℝ, (sin (3 * x) * cos (↑60 - x) + 1) / (sin (↑60 - 7 * x) - cos (↑30 + x) + m) = 0 :=
by
  sorry

end no_solution_for_x_l353_353546


namespace largest_int_with_remainder_5_lt_100_l353_353571

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l353_353571


namespace radius_condition_l353_353462

def X (x y : ℝ) : ℝ := 12 * x
def Y (x y : ℝ) : ℝ := 5 * y

def satisfies_condition (x y : ℝ) : Prop :=
  Real.sin (X x y + Y x y) = Real.sin (X x y) + Real.sin (Y x y)

def no_intersection (R : ℝ) : Prop :=
  ∀ (x y : ℝ), satisfies_condition x y → dist (0, 0) (x, y) ≥ R

theorem radius_condition :
  ∀ R : ℝ, (0 < R ∧ R < Real.pi / 15) →
  no_intersection R :=
sorry

end radius_condition_l353_353462


namespace red_ball_second_given_red_ball_first_l353_353067

noncomputable def probability_of_red_second_given_first : ℚ :=
  let totalBalls := 6
  let redBallsOnFirst := 4
  let whiteBalls := 2
  let redBallsOnSecond := 3
  let remainingBalls := 5

  let P_A := redBallsOnFirst / totalBalls
  let P_AB := (redBallsOnFirst / totalBalls) * (redBallsOnSecond / remainingBalls)
  P_AB / P_A

theorem red_ball_second_given_red_ball_first :
  probability_of_red_second_given_first = 3 / 5 :=
sorry

end red_ball_second_given_red_ball_first_l353_353067


namespace largest_int_with_remainder_5_lt_100_l353_353568

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l353_353568


namespace kanul_cash_spending_percentage_l353_353793

theorem kanul_cash_spending_percentage :
  ∀ (spent_raw_materials spent_machinery total_amount spent_cash : ℝ),
    spent_raw_materials = 500 →
    spent_machinery = 400 →
    total_amount = 1000 →
    spent_cash = total_amount - (spent_raw_materials + spent_machinery) →
    (spent_cash / total_amount) * 100 = 10 :=
by
  intros spent_raw_materials spent_machinery total_amount spent_cash
  intro h1 h2 h3 h4
  sorry

end kanul_cash_spending_percentage_l353_353793


namespace number_of_valid_stackings_l353_353348

def valid_stacking (n : ℕ) (stacking : list (ℕ × ℕ)) : Prop :=
  ∀ k l, (k, l) ∈ stacking → l = k-1 ∨ l = k-2 ∨ l ≥ k

theorem number_of_valid_stackings (n : ℕ) (h : n ≥ 1) :
  ∃ k, k = 3^(n-1) * 2 ∧
  ∃ (stackings : list (ℕ × ℕ)), valid_stacking n stackings := sorry

end number_of_valid_stackings_l353_353348


namespace solution_set_inequality_l353_353185

-- Definitions based on the conditions
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def positive_value (f : ℝ → ℝ) := ∀ x : ℝ, x > 0 → f x = 1

-- Statement to be proved
theorem solution_set_inequality 
  (f : ℝ → ℝ) 
  (h_odd : odd_function f) 
  (h_pos : positive_value f) 
  (h_f0 : f 0 = 0) 
  : { x : ℝ | f (x^2 - x) < f 0 } = set.Ioo 0 1 := 
sorry

end solution_set_inequality_l353_353185


namespace chord_length_of_C1_l353_353656

theorem chord_length_of_C1 (C1 : x^2 + y^2 = 25) (l : 3x - 4y - 15 = 0) : 
  length_chord (C1, l) = 8 :=
sorry

end chord_length_of_C1_l353_353656


namespace binom_two_eq_n_choose_2_l353_353931

theorem binom_two_eq_n_choose_2 (n : ℕ) (h : n ≥ 2) :
  (nat.choose n 2) = (n * (n - 1)) / 2 := by
  sorry

end binom_two_eq_n_choose_2_l353_353931


namespace landscaping_charges_l353_353231

theorem landscaping_charges
    (x : ℕ)
    (h : 63 * x + 9 * 11 + 10 * 9 = 567) :
  x = 6 :=
by
  sorry

end landscaping_charges_l353_353231


namespace max_value_f_l353_353187

noncomputable def max_f (a : Fin 2019 → ℝ) : ℝ :=
  ∑ i, (a i) ^ 3

theorem max_value_f (a : Fin 2019 → ℝ) :
  (∀ i, a i ∈ Set.Icc (-1 : ℝ) 1) → (∑ i, a i = 0) →
  max_f a ≤ 2019 / 4 :=
by
  sorry

end max_value_f_l353_353187


namespace limit_of_series_of_powers_l353_353189

theorem limit_of_series_of_powers (a : ℝ) (h_pos : a > 0) (h_coeff : 6 * a^2 = 3 / 2) :
  (tendsto (λ n, (finset.range n).sum (λ k, a^(k+1))) at_top (𝓝 (1))) :=
by {
  sorry
}

end limit_of_series_of_powers_l353_353189


namespace roots_poly_squared_sum_l353_353121

-- Define the polynomial with roots p, q, r
def poly (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 5 * x + 15

-- Given that p, q, r are roots of poly.
variables {p q r : ℝ}
-- Conditions using Vieta's formulas
def sum_roots : ℝ := p + q + r
def sum_product_roots : ℝ := p*q + q*r + r*p
def product_roots : ℝ := p*q*r

-- Theorem to prove that p^2 + q^2 + r^2 = -26/9 
theorem roots_poly_squared_sum :
  poly p = 0 ∧ poly q = 0 ∧ poly r = 0 →
  sum_roots = 2/3 →
  sum_product_roots = 5/3 →
  (p^2 + q^2 + r^2) = -26/9 :=
begin
  sorry
end

end roots_poly_squared_sum_l353_353121


namespace parabola_directrix_line_parallel_to_OA_l353_353224

noncomputable def parabola_equation (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

def point_in_parabola (p x y : ℝ) : Prop :=
  parabola_equation p x y ∧ x = 1 ∧ y = -2

theorem parabola_directrix (p : ℝ) (h1 : p > 0) (h2 : point_in_parabola p 1 (-2)) : 
  ∃ a b c, (parabola_equation p 1 (-2)) ∧ (a = 4 ∧ b = -1) :=
sorry

theorem line_parallel_to_OA (d : ℝ) (h : d = sqrt(5) / 5) : 
  ∃ l : ℝ → ℝ → Prop, (∀ x y, l x y = (2 * x + y - 1 = 0)) :=
sorry

end parabola_directrix_line_parallel_to_OA_l353_353224


namespace count_integers_between_sqrt_10_and_sqrt_100_l353_353719

theorem count_integers_between_sqrt_10_and_sqrt_100 :
  ∃ n : ℕ, (∀ k : ℕ, 4 ≤ k ∧ k ≤ 10 → k ∈ set.Ico (int.floor (sqrt (10 : ℝ))) (int.ceil (sqrt (100 : ℝ)))) ∧ n = 7 :=
sorry

end count_integers_between_sqrt_10_and_sqrt_100_l353_353719


namespace simplify_vector_expression_l353_353867

variables {V : Type*} [AddGroup V]

noncomputable def vector_expression_simplified {A B C D : V} :
  A - B + C - D = (A + C) - (B + D) := by
sorry

theorem simplify_vector_expression {A B C D : V} :
  (A - B + C - D) = 0 :=
begin
  -- We would include the appropriate logical steps here to justify the conclusion.
  sorry
end

end simplify_vector_expression_l353_353867


namespace minimum_BD_value_l353_353298

theorem minimum_BD_value {A B C D : Type*} [Real A] [Real B] [Real C] [Real D] (angle_ADB: A) (AD: A) (CD: A) (BD: A):
  (D ∈ line_segment(B, C)) →
  angle_ADB = 120 → 
  AD = 2 → 
  CD = 2 * BD →
  ∃ BD: A, BD = sqrt(3) - 1 :=
by
  sorry

end minimum_BD_value_l353_353298


namespace treasures_second_level_l353_353008

-- Defining the conditions
def points_per_treasure : ℕ := 6
def treasures_first_level : ℕ := 3
def total_score : ℕ := 48

-- Defining what we want to prove
theorem treasures_second_level :
  let points_first_level := treasures_first_level * points_per_treasure in
  let points_second_level := total_score - points_first_level in
  points_second_level / points_per_treasure = 5 := 
by
  sorry

end treasures_second_level_l353_353008


namespace angle_BAD_in_quadrilateral_l353_353773

theorem angle_BAD_in_quadrilateral
  (A B C D : Type)
  (AB BD DC : ℝ)
  (h1 : AB = BD)
  (h2 : BD = DC)
  (h3 : ∠ ABC = 60)
  (h4 : ∠ BCD = 160) :
  ∠ BAD = 106.67 :=
sorry

end angle_BAD_in_quadrilateral_l353_353773


namespace total_bottles_per_day_l353_353072

def num_cases_per_day : ℕ := 7200
def bottles_per_case : ℕ := 10

theorem total_bottles_per_day : num_cases_per_day * bottles_per_case = 72000 := by
  sorry

end total_bottles_per_day_l353_353072


namespace rick_gives_miguel_cards_l353_353859

/-- Rick starts with 130 cards, keeps 15 cards for himself, gives 
12 cards each to 8 friends, and gives 3 cards each to his 2 sisters. 
We need to prove that Rick gives 13 cards to Miguel. --/
theorem rick_gives_miguel_cards :
  let initial_cards := 130
  let kept_cards := 15
  let friends := 8
  let cards_per_friend := 12
  let sisters := 2
  let cards_per_sister := 3
  initial_cards - kept_cards - (friends * cards_per_friend) - (sisters * cards_per_sister) = 13 :=
by
  sorry

end rick_gives_miguel_cards_l353_353859


namespace largest_int_less_than_100_with_remainder_5_l353_353562

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l353_353562


namespace binom_n_2_l353_353934

theorem binom_n_2 (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_n_2_l353_353934


namespace main_theorem_l353_353335

-- Definitions: Let n be a positive integer and x be a list of positive real numbers such that their product is 1
def n : ℕ := sorry
def x : list ℝ := sorry
def is_positive (a : ℝ) : Prop := 0 < a

-- Assume the product of elements in x is 1
def product_one (x : list ℝ) : Prop := (x.prod = 1)

-- Define the inequality to be proved
def inequality (n : ℕ) (x : list ℝ) : Prop :=
  ∑ i in (list.range n), (x[i] * real.sqrt ((list.take (i+1) x).sum (λ y, y^2))) ≥ ((n+1)/2) * real.sqrt n

-- The main statement: Prove the inequality given the conditions
theorem main_theorem 
  (h1: n > 0) 
  (h2: ∀ i, i < n → is_positive (x[i]))
  (h3: product_one x) : 
  inequality n x :=
by sorry

end main_theorem_l353_353335


namespace min_value_of_a_plus_b_l353_353660

theorem min_value_of_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) : a + b = 4 :=
sorry

end min_value_of_a_plus_b_l353_353660


namespace find_solutions_l353_353354

-- Given piecewise function definition
def g : ℝ → ℝ :=
  λ x, if x < 0 then 4 * x + 5 else 3 * x - 10

-- Definition of the problem
theorem find_solutions : 
  ∃ x, g x = 2 ∧ (x = -(3 / 4) ∨ x = 4) :=
by
  -- Existence of solutions
  existsi (-3 / 4)
  split
  {
    -- Verifying the equation holds for x = -3/4
    unfold g
    rw [if_pos]
    {
      -- Showing 4 * (-3 / 4) + 5 = 2
      norm_num
    },
    -- Condition that x = -3/4 satisfies x < 0
    norm_num,
  }
  existsi 4
  split
  {
    -- Verifying the equation holds for x = 4
    unfold g
    rw [if_neg]
    {
      -- Showing 3 * 4 - 10 = 2
      norm_num,
    },
    -- Condition that x = 4 satisfies x >= 0
    norm_num,
  }
  -- Combination of valid solutions
  {
    norm_num,
  }
end of code.makedirs

end find_solutions_l353_353354


namespace min_distance_l353_353288

def triangle_ABC (A B C D : Type) (BD : ℝ) :=
∃ (CD : ℝ), CD = 2 * BD ∧ 
  ∃ (AD AC AB : ℝ), AD = 2 ∧ 
    ∃ (α β γ : ℝ), α = 120 ∧ 
      angle_ADB = α ∧ 
        (AC / AB = (sqrt (3) - 1))

theorem min_distance (A B C D : Type) (BD : ℝ) :
  ∃ D ∈ BC, ∃ α β γ AD AC AB, 
    α = 120 ∧ AD = 2 ∧ D ∈ BC ∧ 
      γ = ∠ADB = α ∧ 
        CD = 2 * BD →
          (AC / AB) = (sqrt (3) - 1) := 
sorry

end min_distance_l353_353288


namespace find_correct_representation_l353_353098

noncomputable def check_representation (m n : ℝ) (h_m : m ≠ 0) (h_n : n ≠ 0) : Prop :=
  ∀ (A B C D : Prop), 
    (m * x + n * y^2 = 0 → m * x^2 + n * y^2 = 1 → A) →
    (m * x + n * y^2 = 0 → m * x^2 + n * y^2 = 1 → B) →
    (m * x + n * y^2 = 0 → m * x^2 + n * y^2 = 1 → C) →
    (m * x + n * y^2 = 0 → m * x^2 + n * y^2 = 1 → D) →
    (A ∨ B ∨ C ∨ D)

theorem find_correct_representation (m n : ℝ) (h_m : m ≠ 0) (h_n : n ≠ 0) :
  check_representation m n h_m h_n (A) :=
sorry

end find_correct_representation_l353_353098


namespace find_equations_and_incircle_l353_353989

-- Defining points P, Q, and M
def P : (ℝ × ℝ) := (-2, 4 * Real.sqrt 3)
def Q : (ℝ × ℝ) := (2, 0)
def M : (ℝ × ℝ) := (0, -6)

-- Line equation template
def line_eq (m b x : ℝ) := m * x + b
-- Line l equation: x = my - 2√3, we need to transform y into form dependent on x
def line_l (m : ℝ) (x : ℝ) : ℝ := (x + 2 * Real.sqrt 3) / m

theorem find_equations_and_incircle :
  ∃ (l₁ l₂ : ℝ → ℝ) (m : ℝ) (r t : ℝ),
    l₁ = fun x => -Real.sqrt 3 * (x - 2) ∧
    l₂ = fun x => Real.sqrt 3 * (x - 2) ∧
    l₂ - l₁ = fun x => 2 * Real.sqrt 3 * x ∧
    t = 2 ∧
    r = 1 ∧ -- Radius has to be related to incircle
    ∀ x y, (x - 2)^2 + (y - t)^2 = r^2 :=
begin
  -- The proof would go here
  sorry
end

end find_equations_and_incircle_l353_353989


namespace divides_8x_7y_l353_353467

theorem divides_8x_7y (x y : ℤ) (h : 5 ∣ (x + 9 * y)) : 5 ∣ (8 * x + 7 * y) :=
sorry

end divides_8x_7y_l353_353467


namespace translate_to_left_find_ϕ_l353_353910

noncomputable def original_function (x : ℝ) : ℝ := 5 * (Real.sin (2 * x + π / 4))

noncomputable def translated_function (x ϕ : ℝ) : ℝ := original_function (x + ϕ)

theorem translate_to_left_find_ϕ (ϕ : ℝ) (h : 0 < ϕ ∧ ϕ < π / 2) :
  (translated_function x ϕ = original_function (-(x + ϕ))) → ϕ = π / 8 :=
sorry

end translate_to_left_find_ϕ_l353_353910


namespace reciprocal_inequality_l353_353207

open Real

theorem reciprocal_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (1 / a) + (1 / b) > 1 / (a + b) :=
sorry

end reciprocal_inequality_l353_353207


namespace count_integers_between_sqrt10_sqrt100_l353_353724

theorem count_integers_between_sqrt10_sqrt100 :
  ∃ n : ℕ, n = 7 ∧ card {x : ℤ | real.sqrt 10 < x ∧ x < real.sqrt 100} = n :=
by
  sorry

end count_integers_between_sqrt10_sqrt100_l353_353724


namespace smallest_odd_digits_multiple_of_9_l353_353447

theorem smallest_odd_digits_multiple_of_9 :
  ∃ n : ℕ, n < 10000 ∧ 
           (∀ d : ℕ, d ∈ [n.digits 10] → d % 2 = 1) ∧ 
           (∑ d in n.digits 10) % 9 = 0 ∧ 
           (∀ m : ℕ, m < n ∧ m < 10000 ∧ 
           (∀ d : ℕ, d ∈ [m.digits 10] → d % 2 = 1) ∧ 
           (∑ d in m.digits 10) % 9 = 0 → m ≥ n) :=
exists.intro 1117 sorry

end smallest_odd_digits_multiple_of_9_l353_353447


namespace coefficient_of_x6_in_expansion_l353_353520

theorem coefficient_of_x6_in_expansion :
  let general_term (k : ℕ) := (binom 7 k) * ((1 : ℤ)^(7 - k)) * ((-3)^(k)) in
  (∃ (coeff : ℤ), coeff = (general_term 3) ∧ (3 : ℕ) * 2 = 6 ∧ coeff = -945) :=
by
  sorry

end coefficient_of_x6_in_expansion_l353_353520


namespace sum_squared_distances_eq_2p_times_r2_plus_d2_l353_353461

theorem sum_squared_distances_eq_2p_times_r2_plus_d2 
  (n : ℕ) (p r d : ℝ)
  (A : fin n → ℝ) -- A_i are the sides of the polygon
  (B : fin n → ℝ) -- B_i are the distances MB_i
  (h_perimeter : ∑ i, A i = 2 * p) -- Perimeter condition
  (h_distance : ∀ i, B i^2 * A i = (2 * p * (r^2 + d^2))) : 
  ∑ i, (B i^2 * A i) = 2 * p * (r^2 + d^2) :=
sorry

end sum_squared_distances_eq_2p_times_r2_plus_d2_l353_353461


namespace common_difference_is_neg_4_maximum_value_of_Sn_is_78_maximum_n_when_Sn_positive_is_12_l353_353901

-- Conditions
variables {a : ℕ → ℝ}
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a n = a 1 + (n - 1) * d

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * a 1 + (n * (n - 1) * (-4))/2 -- Simplified based on given sequence and common difference

-- Theorem for first problem
theorem common_difference_is_neg_4 : a 1 = 23 ∧ a 6 = 3 → ∃ d, arithmetic_sequence a d ∧ d = -4 := by
  sorry

-- Theorem for second problem
theorem maximum_value_of_Sn_is_78 : a 1 = 23 ∧ a 6 = 3 → ∃ n, Sn a n = 78 := by
  sorry

-- Theorem for third problem
theorem maximum_n_when_Sn_positive_is_12 : a 1 = 23 ∧ a 6 = 3 → ∃ n, Sn a n > 0 ∧ n = 12 := by
  sorry

end common_difference_is_neg_4_maximum_value_of_Sn_is_78_maximum_n_when_Sn_positive_is_12_l353_353901


namespace cos_F_eq_l353_353786

noncomputable def angle_D := 90
noncomputable def sin_E := 3 / 5

theorem cos_F_eq :
  angle_D = 90 → sin_E = 3 / 5 → ∃ F, cos F = 3 / 5 :=
by
  intros hD hE
  use 90 - E
  have hCompl : E + F = 90,
  sorry
  have hCos := sin E,
  sorry
  exact hCos
  sorry

end cos_F_eq_l353_353786


namespace question1_solution_set_question2_proof_l353_353688

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 3)

theorem question1_solution_set : 
  { x : ℝ | f x < 7 } = { x : ℝ | -2 < x ∧ x < 5 } := 
by 
  sorry

theorem question2_proof (x : ℝ) : 
  f x - abs (2 * x - 7) < x^2 - 2 * x + real.sqrt 26 := 
by 
  sorry

end question1_solution_set_question2_proof_l353_353688


namespace slope_of_line_l353_353953

theorem slope_of_line (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 3 / x + 4 / y = 0) : 
  ∃ m : ℝ, m = -4 / 3 := 
sorry

end slope_of_line_l353_353953


namespace tennis_tournament_rounds_l353_353797

theorem tennis_tournament_rounds (N : ℕ) (h : N ≥ 2) :
  (N = 2 → ∃ M, M = 2 ∧ ∀ (pairing: Fin (2*N) → Fin N), ∀ (m : ℕ), 
      (m = M) -> ∀ k ∈ (range (N + 1)).erase 0, 
      (function.MoveCourt pairing k m).court ≠ function.initialCourt k) ∧
  (N ≥ 3 → ∃ M, M = N + 1 ∧ ∀ (pairing: Fin (2*N) → Fin N), ∀ (m : ℕ), 
      (m = M) -> ∀ k ∈ (range (N + 1)).erase 0, 
      (function.MoveCourt pairing k m).court ≠ function.initialCourt k) :=
by
  sorry

end tennis_tournament_rounds_l353_353797


namespace children_are_sorted_after_operations_l353_353491

noncomputable def children_arrangement(N : ℕ) : Prop :=
  ∀ (children : list ℕ), (children.length = N) →
  (∀ (i j : ℕ), i < j → i < N ∧ j < N → children[i] ≠ children[j]) →
  let operation := λ (l : list ℕ), 
    let groups := l.splitWhen (λ (a b : ℕ), a >= b) in
    (groups.bind list.reverse) in
  ∃ (result : list ℕ), 
  (result = (list.iterate operation (N - 1)) children) ∧ 
  (∀ (k : ℕ), k < N-1 → ∀ (i : ℕ), i < (l.length - 1) →
    result[i] > result[i+1])

theorem children_are_sorted_after_operations :
  ∀ (N : ℕ), children_arrangement N :=
sorry

end children_are_sorted_after_operations_l353_353491


namespace fourth_person_height_l353_353006

theorem fourth_person_height (h : ℝ)
  (h2 : h + 2 = h₂)
  (h3 : h + 4 = h₃)
  (h4 : h + 10 = h₄)
  (average_height : (h + h₂ + h₃ + h₄) / 4 = 77) :
  h₄ = 83 :=
by
  sorry

end fourth_person_height_l353_353006


namespace count_integers_between_sqrt10_sqrt100_l353_353727

theorem count_integers_between_sqrt10_sqrt100 :
  ∃ n : ℕ, n = 7 ∧ card {x : ℤ | real.sqrt 10 < x ∧ x < real.sqrt 100} = n :=
by
  sorry

end count_integers_between_sqrt10_sqrt100_l353_353727


namespace length_CG_length_AE_l353_353351

noncomputable section

open_locale classical

variables (A B C D G E : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space G] [metric_space E]
variables (distance : ∀ X Y : A, ℝ) -- assuming distance is defined for any pair of points

-- Conditions
axiom cond1 : ∀ (A G : A), ¬ collinear A G C
axiom cond2 : ∀ (A G : A), collinear A G D
axiom cond3 : distance C D = 10
axiom cond4 : distance A B = 5
axiom cond5 : distance A G = 7
axiom cond6 : ∀ (A G : A), E = midpoint C D

-- Questions to prove
theorem length_CG : ∃ CG : ℝ, 5 * CG = 70 := sorry

theorem length_AE : ∃ AE : ℝ, (AE^2 = 2^2 + 5^2) := sorry

end length_CG_length_AE_l353_353351


namespace sin_pow_cos_pow_sum_l353_353744

namespace ProofProblem

-- Define the condition
def trig_condition (x : ℝ) : Prop :=
  3 * (Real.sin x)^3 + (Real.cos x)^3 = 3

-- State the theorem
theorem sin_pow_cos_pow_sum (x : ℝ) (h : trig_condition x) : Real.sin x ^ 2018 + Real.cos x ^ 2018 = 1 :=
by
  sorry

end ProofProblem

end sin_pow_cos_pow_sum_l353_353744


namespace water_transfer_possible_iff_power_of_two_l353_353468

theorem water_transfer_possible_iff_power_of_two (n : ℕ) (h : n > 0) : (∃ k : ℕ, n = 2^k) ↔ (∃ (cups : Fin n → ℕ), (∀ i : Fin n, cups i = 13 * 60 / n) ∧ ∃ j : Fin n, (∀ k ≠ j, cups k = 0) ∧ (∑ i, cups i = 13 * 60)) :=
by
  sorry

end water_transfer_possible_iff_power_of_two_l353_353468


namespace least_common_multiple_5_6_10_12_l353_353945

open Nat

theorem least_common_multiple_5_6_10_12 :
  lcm (lcm 5 6) (lcm 10 12) = 60 :=
by
  sorry

end least_common_multiple_5_6_10_12_l353_353945


namespace quotient_polynomial_division_l353_353031

noncomputable def P (x : ℝ) : ℝ := 6*x^3 + 12*x^2 - 5*x + 3

noncomputable def D (x : ℝ) : ℝ := 3*x + 4

theorem quotient_polynomial_division :
  ∀ x : ℝ, (P(x) / D(x)) = 2*x^2 + (4/3)*x - (31/9) := 
sorry

end quotient_polynomial_division_l353_353031


namespace joes_mean_score_is_88_83_l353_353332

def joesQuizScores : List ℕ := [88, 92, 95, 81, 90, 87]

noncomputable def mean (lst : List ℕ) : ℝ := (lst.sum : ℝ) / lst.length

theorem joes_mean_score_is_88_83 :
  mean joesQuizScores = 88.83 := 
sorry

end joes_mean_score_is_88_83_l353_353332


namespace coeff_x3y3_in_expansion_l353_353125

section
variable {R : Type*} [CommRing R]
open BigOperators

noncomputable def coefficient_x3y3 (x y : R) : R :=
  (range(6)).sum (λ r, (binom 5 r) * ((x^2 + x)^r) * (y^(5 - r)))

theorem coeff_x3y3_in_expansion : coefficient_x3y3 x y = 20 :=
sorry
end

end coeff_x3y3_in_expansion_l353_353125


namespace intersect_range_of_f_l353_353883

open Real

def f (x : ℝ) : ℝ := sin x + 2 * abs (sin x)

theorem intersect_range_of_f : 
  ∀ k : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * π → f x ≠ k) → (1 < k ∧ k < 3) := 
by
  sorry

end intersect_range_of_f_l353_353883


namespace translate_parabola_l353_353425

theorem translate_parabola :
  ∀ (x y : ℝ), y = -5*x^2 + 1 → y = -5*(x + 1)^2 - 1 := by
  sorry

end translate_parabola_l353_353425


namespace calculate_ff2_l353_353341

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x + 4

theorem calculate_ff2 : f (f 2) = 5450 := by
  sorry

end calculate_ff2_l353_353341


namespace nut_swapping_l353_353091

-- Define the problem
theorem nut_swapping (n : ℕ) (h_n : n = 2021) (nuts : Fin n → ℕ)
    (moves : Fin n → Fin n × Fin n)
    (moves_correct : ∀ k : Fin n, 
      moves k = (⟨(k.1 + n - 1) % n, (sorry : k.1 + 1) % n⟩)) :
    ∃ k : Fin n, ∃ a b : Fin n, (a, b) = moves k ∧ a.1 < k.1 ∧ k.1 < b.1 := 
  sorry

end nut_swapping_l353_353091


namespace find_angle_QRC_l353_353408

theorem find_angle_QRC (P Q R C : Point)
  (h_incircle : incircle_of_center P Q R C)
  (angle_PQR : angle P Q R = 63)
  (angle_QPR : angle Q P R = 59) :
  angle Q R C = 29 := sorry

end find_angle_QRC_l353_353408


namespace product_of_constants_l353_353165

theorem product_of_constants (h_factorized : ∀ t : ℤ, 
    ∃ a b : ℤ, x^2 + t * x - 24 = (x + a) * (x + b) ∧ a * b = -24) : 
    ∏ t in {t : ℤ | ∃ a b : ℤ, a * b = -24 ∧ t = a + b}, t = -10580000 :=
sorry

end product_of_constants_l353_353165


namespace largest_int_with_remainder_5_lt_100_l353_353573

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l353_353573


namespace total_bananas_bought_l353_353860

-- Define the conditions
def went_to_store_times : ℕ := 2
def bananas_per_trip : ℕ := 10

-- State the theorem/question and provide the answer
theorem total_bananas_bought : (went_to_store_times * bananas_per_trip) = 20 :=
by
  -- Proof here
  sorry

end total_bananas_bought_l353_353860


namespace parallel_lines_slope_l353_353990

theorem parallel_lines_slope (m : ℝ) :
  (∀ (L1 L2 : AffineSpace ℝ ℝ), 
    L1 = {p : ℝ × ℝ | p.2 = - (5 / 4) * p.1 + 5} ∧ 
    ∃ (p1 p2 : ℝ × ℝ), p1 = (1, 4) ∧ p2 = (m, -3) ∧ 
    L2 = LineThrough p1 p2 ∧ Parallel L1 L2) → m = 33 / 5 :=
by
  sorry

end parallel_lines_slope_l353_353990


namespace doubled_container_volume_l353_353484

theorem doubled_container_volume (original_volume : ℕ) (factor : ℕ) 
  (h1 : original_volume = 4) (h2 : factor = 8) : original_volume * factor = 32 :=
by 
  rw [h1, h2]
  norm_num

end doubled_container_volume_l353_353484


namespace largest_integer_less_than_hundred_with_remainder_five_l353_353590

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l353_353590


namespace binom_two_eq_n_choose_2_l353_353929

theorem binom_two_eq_n_choose_2 (n : ℕ) (h : n ≥ 2) :
  (nat.choose n 2) = (n * (n - 1)) / 2 := by
  sorry

end binom_two_eq_n_choose_2_l353_353929


namespace linear_function_range_l353_353191

theorem linear_function_range (a : ℝ) : (2 + a > 0) ∧ (5 - a > 0) ↔ (-2 < a ∧ a < 5) :=
by 
  split 
  all_goals { intro h; sorry }

end linear_function_range_l353_353191


namespace product_of_roots_l353_353349

theorem product_of_roots (p q r : ℝ) (hp : 3*p^3 - 9*p^2 + 5*p - 15 = 0) 
  (hq : 3*q^3 - 9*q^2 + 5*q - 15 = 0) (hr : 3*r^3 - 9*r^2 + 5*r - 15 = 0) :
  p * q * r = 5 :=
sorry

end product_of_roots_l353_353349


namespace binom_two_eq_n_choose_2_l353_353930

theorem binom_two_eq_n_choose_2 (n : ℕ) (h : n ≥ 2) :
  (nat.choose n 2) = (n * (n - 1)) / 2 := by
  sorry

end binom_two_eq_n_choose_2_l353_353930


namespace sheets_in_stack_l353_353488

theorem sheets_in_stack (n : ℕ) (h1 : 400 = n) : 
  (6 : ℝ) / (4 : ℝ / n : ℝ) = 600 :=
by
  sorry

end sheets_in_stack_l353_353488


namespace longer_side_of_rectangle_l353_353981

-- Define the given conditions and problem
def radius : ℝ := 3
def area_circle : ℝ := π * radius^2
def area_rectangle : ℝ := 3 * area_circle
def shorter_side : ℝ := 2 * radius
def longer_side : ℝ := area_rectangle / shorter_side

-- Proof goal: Prove that the length of the longer side of the rectangle is 4.5π cm.
theorem longer_side_of_rectangle : longer_side = 4.5 * π :=
by
  sorry

end longer_side_of_rectangle_l353_353981


namespace largest_int_less_than_100_with_remainder_5_l353_353563

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l353_353563


namespace find_BD_when_AC_over_AB_min_l353_353320

-- Definition of the conditions
variables (A B C D : Type) [metric_space A]
variables (triangle : set A)
variables (on_side_BC : D ∈ segment B C)
variables (angle_ADB : ∠ADB = 120)
variables (AD_len : dist A D = 2)
variables (CD_2BD : dist C D = 2 * dist B D)

noncomputable def minimum_ratio_condition (BD : ℝ) : Prop := 
  let x := dist B D
  let CD := 2 * x
  let AD := 2 in
  let angle_ADB_cos := -1/2 in
  let b_squared := AD^2 + CD^2 - 2 * AD * CD * angle_ADB_cos in
  let c_squared := AD^2 + x^2 + 2 * AD * x * angle_ADB_cos in
  b_squared / c_squared = 4 - (12 / (x + 1 + 3/(x + 1))) → x = sqrt 3 - 1

theorem find_BD_when_AC_over_AB_min (BD : ℝ) : minimum_ratio_condition A B C D 
:= sorry

end find_BD_when_AC_over_AB_min_l353_353320


namespace arithmetic_sequence_a1_l353_353781

theorem arithmetic_sequence_a1 (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_inc : d > 0)
  (h_a3 : a 3 = 1)
  (h_a2a4 : a 2 * a 4 = 3 / 4) : 
  a 1 = 0 :=
sorry

end arithmetic_sequence_a1_l353_353781


namespace polynomial_inequality_l353_353345

-- Define P(x) as a polynomial with non-negative coefficients
def isNonNegativePolynomial (P : Polynomial ℝ) : Prop :=
  ∀ i, P.coeff i ≥ 0

-- The main theorem, which states that for any polynomial P with non-negative coefficients,
-- if P(1) * P(1) ≥ 1, then P(x) * P(1/x) ≥ 1 for all positive x.
theorem polynomial_inequality (P : Polynomial ℝ) (hP : isNonNegativePolynomial P) (hP1 : P.eval 1 * P.eval 1 ≥ 1) :
  ∀ x : ℝ, 0 < x → P.eval x * P.eval (1 / x) ≥ 1 :=
by {
  sorry
}

end polynomial_inequality_l353_353345


namespace last_digit_final_result_l353_353439

-- Conditions in the problem
def mod_exp (a b n : ℕ) : ℕ := a^b % n

lemma last_digit_periodicity (n : ℕ) : ∃ k, n = 2 + 4 * k :=
begin
  use (n - 2) / 4,
  ring,
end

lemma mod_1989_4 : 1989 % 4 = 1 := 
by norm_num

lemma mod_exp_8_10 : 2^8 % 10 = 6 :=
by norm_num

theorem last_digit_final_result : mod_exp (2^2^1989 + 1) 1 10 = 7 :=
sorry

end last_digit_final_result_l353_353439


namespace probability_point_in_circle_l353_353767

theorem probability_point_in_circle (r : ℝ) (h: r = 2) :
  let side_length := 2 * r
  let area_square := side_length ^ 2
  let area_circle := Real.pi * r ^ 2
  (area_circle / area_square) = Real.pi / 4 :=
by
  sorry

end probability_point_in_circle_l353_353767


namespace ones_digit_sum_l353_353440

theorem ones_digit_sum {n : ℕ} (h : n = 2010) : 
  (Finset.range n).sum (λ k, (k + 1) ^ n % 10) % 10 = 5 :=
by
  sorry

end ones_digit_sum_l353_353440


namespace min_distance_l353_353295

def triangle_ABC (A B C D : Type) (BD : ℝ) :=
∃ (CD : ℝ), CD = 2 * BD ∧ 
  ∃ (AD AC AB : ℝ), AD = 2 ∧ 
    ∃ (α β γ : ℝ), α = 120 ∧ 
      angle_ADB = α ∧ 
        (AC / AB = (sqrt (3) - 1))

theorem min_distance (A B C D : Type) (BD : ℝ) :
  ∃ D ∈ BC, ∃ α β γ AD AC AB, 
    α = 120 ∧ AD = 2 ∧ D ∈ BC ∧ 
      γ = ∠ADB = α ∧ 
        CD = 2 * BD →
          (AC / AB) = (sqrt (3) - 1) := 
sorry

end min_distance_l353_353295


namespace largest_integer_less_than_100_with_remainder_5_l353_353576

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353576


namespace fractions_are_integers_l353_353832

theorem fractions_are_integers
  (a b c : ℤ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h : ∃ k : ℤ, \left(\frac{a * b}{c} + \frac{b * c}{a} + \frac{c * a}{b}\right) = k) :
  \left(\frac{a * b}{c}\right).is_integer ∧ \left(\frac{b * c}{a}\right).is_integer ∧ \left(\frac{c * a}{b}\right).is_integer := by
sory

end fractions_are_integers_l353_353832


namespace percent_of_g_is_a_l353_353053

theorem percent_of_g_is_a (a b c d e f g : ℤ) (h1 : (a + b + c + d + e + f + g) / 7 = 9)
: (a / g) * 100 = 50 := 
sorry

end percent_of_g_is_a_l353_353053


namespace a_100_value_l353_353194

variables (S : ℕ → ℚ) (a : ℕ → ℚ)

def S_n (n : ℕ) : ℚ := S n
def a_n (n : ℕ) : ℚ := a n

axiom a1_eq_3 : a 1 = 3
axiom a_n_formula (n : ℕ) (hn : n ≥ 2) : a n = (3 * S n ^ 2) / (3 * S n - 2)

theorem a_100_value : a 100 = -3 / 88401 :=
sorry

end a_100_value_l353_353194


namespace equilateral_triangle_sum_l353_353501

theorem equilateral_triangle_sum (x y : ℕ) (h1 : x + 5 = 14) (h2 : y + 11 = 14) : x + y = 12 :=
by
  sorry

end equilateral_triangle_sum_l353_353501


namespace eligibility_conditions_met_l353_353253

-- Define the necessary structures and conditions
-- Assume the existence of individuals and the ability to specify neighbors via radii R and r
def individual := { height : ℝ, position : ℝ }

-- Define neighbors for police and military based on radii R and r
def is_police_neighbor (R : ℝ) (pos1 pos2 : ℝ) : Bool :=
  abs (pos1 - pos2) < R

def is_military_neighbor (r : ℝ) (pos1 pos2 : ℝ) : Bool :=
  abs (pos1 - pos2) < r

-- Prove that there exists a setup where eligibility conditions are met for either police or military
theorem eligibility_conditions_met :
  ∃ (individuals : List individual) (R r : ℝ), 
  (∀ ind₁ ∈ individuals, ∃ neighbors_police : List individual,
    neighbors_police.length > 0 ∧
    (∀ ind₂ ∈ neighbors_police, is_police_neighbor R ind₁.position ind₂.position) ∧
    ((1/5)*neighbors_police.length ≤ (neighbors_police.filter (λ n, n.height < ind₁.height)).length)) ∨
  (∀ ind₁ ∈ individuals, ∃ neighbors_military : List individual,
    neighbors_military.length > 0 ∧
    (∀ ind₂ ∈ neighbors_military, is_military_neighbor r ind₁.position ind₂.position) ∧
    ((1/5)*neighbors_military.length ≤ (neighbors_military.filter (λ n, n.height > ind₁.height)).length)) :=
sorry

end eligibility_conditions_met_l353_353253


namespace average_of_second_and_third_smallest_is_3_5_l353_353388

theorem average_of_second_and_third_smallest_is_3_5 
  (a1 a2 a3 a4 a5 : ℕ) (h1 : a1 ≠ a2 ∧ a2 ≠ a3 ∧ a3 ≠ a4 ∧ a4 ≠ a5 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a3 ≠ a5)
  (h2 : a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0 ∧ a5 > 0)
  (h3 : (a1 + a2 + a3 + a4 + a5) = 25)
  (maximize_diff : a5 - a1 = max (a5 - a1) (a5 - a2) (a5 - a3) (a5 - a4) (a4 - a1) (a4 - a2) (a4 - a3) (a3 - a1) (a3 - a2) (a2 - a1))
  (sorted : a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5) :
  (a2 + a3) / 2 = 7 / 2 :=
by {
    sorry
}

end average_of_second_and_third_smallest_is_3_5_l353_353388


namespace average_number_of_candies_l353_353333

theorem average_number_of_candies (candies_per_bag : List ℕ) (h_length : candies_per_bag.length = 9)
    (h_bags : candies_per_bag = [9, 11, 13, 16, 18, 19, 21, 23, 25]) :
  (List.sum candies_per_bag : ℝ) / candies_per_bag.length = 17.22 :=
  by
  sorry

end average_number_of_candies_l353_353333


namespace hours_worked_l353_353334

theorem hours_worked (w e : ℝ) (hw : w = 6.75) (he : e = 67.5) 
  : e / w = 10 := by
  sorry

end hours_worked_l353_353334


namespace average_six_consecutive_integers_starting_with_d_l353_353868

theorem average_six_consecutive_integers_starting_with_d (c : ℝ) (d : ℝ)
  (h₁ : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5)) / 6 = c + 5 :=
by
  sorry -- Proof to be completed

end average_six_consecutive_integers_starting_with_d_l353_353868


namespace problem_statement_l353_353667

variables (a : ℝ)

def line1 := ∀ x y : ℝ, sqrt 3 * x + y - 1 = 0
def line2 := ∀ x y : ℝ, a * x + y = 1
def perpendicular (m1 m2 : ℝ) := m1 * m2 = -1

def angle_of_inclination_of_l1 : Prop := 
  let m := - sqrt 3 in 
  real.atan m = 2 * real.pi / 3

def distance_from_origin_to_l2 : Prop :=
  abs (0 * 0 + 1 * 0 - 1) / sqrt ((-a)^2 + 1^2) = sqrt 3 / 2

theorem problem_statement (a : ℝ) 
  (h1 : line1) 
  (h2 : line2) 
  (h3 : perpendicular (- sqrt 3) a) : 
  angle_of_inclination_of_l1 ∧ distance_from_origin_to_l2 :=
by sorry

end problem_statement_l353_353667


namespace adam_remaining_loads_l353_353498

-- Define the initial conditions
def total_loads : ℕ := 25
def washed_loads : ℕ := 6

-- Define the remaining loads as the total loads minus the washed loads
def remaining_loads (total_loads washed_loads : ℕ) : ℕ := total_loads - washed_loads

-- State the theorem to be proved
theorem adam_remaining_loads : remaining_loads total_loads washed_loads = 19 := by
  sorry

end adam_remaining_loads_l353_353498


namespace min_ratio_bd_l353_353311

noncomputable def triangle_ratio_min (BD : ℝ) : Prop :=
  ∃ K L : ℝ, 
    (K = 1) ∧ 
    (L = 1) ∧ 
    (∀ (x : ℝ), ∀ (y : ℝ), (CD = 2 * BD) → ((x = sqrt(3) - 1) → y = (bd + 1 + 3 / (bd + 1)) ∧ (x + y) = 2 * K * L) ∧ 
    (abs x ≤ (sqrt 3 - 1)))


theorem min_ratio_bd (BD : ℝ) : triangle_ratio_min BD :=
by { sorry }

end min_ratio_bd_l353_353311


namespace lockers_coloring_ways_l353_353875

def color := ℕ → ℕ -- 1, 2, 3 represent the colors blue, red, green respectively

def valid_coloring (c : color) : Prop :=
  ∀ m n : ℕ, m > 0 → n > 0 → m ≤ 10 → n ≤ 10 → (m - n) % 2 = 1 → c m ≠ c n

def one_of_three_colors (c : color) : Prop :=
  ∀ n : ℕ, n > 0 → n ≤ 10 → (c n = 1 ∨ c n = 2 ∨ c n = 3)

theorem lockers_coloring_ways :
  ∃ c : color, valid_coloring c ∧ one_of_three_colors c ∧
    ∃ (list_colorings : list color), (∀ c' : color, c' ∈ list_colorings → valid_coloring c' ∧ one_of_three_colors c') →
    list.length list_colorings = 186 :=
by
  sorry

end lockers_coloring_ways_l353_353875


namespace actual_length_of_road_l353_353848

-- Define the conditions
def scale_factor : ℝ := 2500000
def length_on_map : ℝ := 6
def cm_to_km : ℝ := 100000

-- State the theorem
theorem actual_length_of_road : (length_on_map * scale_factor) / cm_to_km = 150 := by
  sorry

end actual_length_of_road_l353_353848


namespace olivia_nigel_remaining_money_l353_353846

theorem olivia_nigel_remaining_money :
  let olivia_money := 112
  let nigel_money := 139
  let ticket_count := 6
  let ticket_price := 28
  let total_money := olivia_money + nigel_money
  let total_cost := ticket_count * ticket_price
  total_money - total_cost = 83 := 
by 
  sorry

end olivia_nigel_remaining_money_l353_353846


namespace probability_same_value_after_reroll_l353_353375

theorem probability_same_value_after_reroll
  (initial_dice : Fin 6 → Fin 6)
  (rerolled_dice : Fin 4 → Fin 6)
  (initial_pair_num : Fin 6)
  (h_initial_no_four_of_a_kind : ∀ (n : Fin 6), (∃ i j : Fin 6, i ≠ j ∧ initial_dice i = n ∧ initial_dice j = n) →
    ∃ (i₁ i₂ i₃ i₄ : Fin 6), i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₃ ≠ i₄ ∧
    initial_dice i₁ ≠ n ∧ initial_dice i₂ ≠ n ∧ initial_dice i₃ ≠ n ∧ initial_dice i₄ ≠ n)
  (h_initial_pair : ∃ i j : Fin 6, i ≠ j ∧ initial_dice i = initial_pair_num ∧ initial_dice j = initial_pair_num) :
  (671 : ℚ) / 1296 = 671 / 1296 :=
by sorry

end probability_same_value_after_reroll_l353_353375


namespace minors_of_D_l353_353453

-- Define the given matrix D
def D := ![![(-1 : ℤ), 2, 0], ![3, 7, -1], ![5, 4, 2]]

-- Calculating the minors as functions
def minor11 : ℤ := D[1][1] * D[2][2] - D[1][2] * D[2][1]
def minor12 : ℤ := D[1][0] * D[2][2] - D[1][2] * D[2][0]
def minor13 : ℤ := D[1][0] * D[1][1] - D[1][1] * D[2][0]

def minor21 : ℤ := D[0][1] * D[2][2] - D[0][2] * D[2][1]
def minor22 : ℤ := D[0][0] * D[2][2] - D[0][2] * D[2][0]
def minor23 : ℤ := D[0][0] * D[1][1] - D[0][1] * D[1][0]

def minor31 : ℤ := D[0][1] * D[1][2] - D[0][2] * D[1][1]
def minor32 : ℤ := D[0][0] * D[1][2] - D[0][2] * D[1][0]
def minor33 : ℤ := D[0][0] * D[0][1] - D[0][1] * D[1][0]

-- The goal statement
theorem minors_of_D :
    minor11 D = 18 ∧ minor12 D = 11 ∧ minor13 D = -23 ∧
    minor21 D = 4 ∧ minor22 D = -2 ∧ minor23 D = -14 ∧
    minor31 D = -2 ∧ minor32 D = 1 ∧ minor33 D = -13 := 
by {
    -- Here should be the proof but we are adding sorry to skip it
    sorry
}

end minors_of_D_l353_353453


namespace collinear_points_l353_353213

open EuclideanGeometry

theorem collinear_points (A B C A' B' C' D E F L M N P Q R: Point) :
  -- Given triangle ABC
  Triangle A B C →
  -- and its corresponding medians intersecting nine-point circle at D, E, and F 
  Median A B C A' ∧ OnNinePointCircle D A' →
  Median B A C B' ∧ OnNinePointCircle E B' →
  Median C A B C' ∧ OnNinePointCircle F C' →
  -- Given feet of perpendiculars are L, M, N
  FootOfPerpendicular A B C L →
  FootOfPerpendicular B C A M →
  FootOfPerpendicular C A B N →
  -- Tangents at D, E, F intersecting lines MN, LN, LM at P, Q, R
  TangentAt D E F P ∧ Intersect mn P →
  TangentAt E D F Q ∧ Intersect ln Q →
  TangentAt F D E R ∧ Intersect lm R →
  -- Prove that P, Q, R are collinear
  Collinear [P, Q, R] :=
sorry

end collinear_points_l353_353213


namespace count_integers_between_sqrt_10_and_sqrt_100_l353_353723

theorem count_integers_between_sqrt_10_and_sqrt_100 :
  ∃ n : ℕ, (∀ k : ℕ, 4 ≤ k ∧ k ≤ 10 → k ∈ set.Ico (int.floor (sqrt (10 : ℝ))) (int.ceil (sqrt (100 : ℝ)))) ∧ n = 7 :=
sorry

end count_integers_between_sqrt_10_and_sqrt_100_l353_353723


namespace equilateral_triangles_count_in_grid_of_side_4_l353_353705

-- Define a function to calculate the number of equilateral triangles in a triangular grid of side length n
def countEquilateralTriangles (n : ℕ) : ℕ :=
  (n * (n + 1) * (n + 2) * (n + 3)) / 24

-- Define the problem statement for n = 4
theorem equilateral_triangles_count_in_grid_of_side_4 :
  countEquilateralTriangles 4 = 35 := by
  sorry

end equilateral_triangles_count_in_grid_of_side_4_l353_353705


namespace binom_two_eq_l353_353939

theorem binom_two_eq (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binom_two_eq_l353_353939


namespace sum_of_divisors_45_l353_353448

theorem sum_of_divisors_45 : 
  let n := 45 in 
  let prime_factors := [(3, 2), (5, 1)] in 
  (∑ d in divisors n, d) = 78 :=
by
  let n := 45
  let prime_factors := [(3, 2), (5, 1)]
  sorry

end sum_of_divisors_45_l353_353448


namespace arithmetic_sequence_z_value_l353_353028

theorem arithmetic_sequence_z_value :
  ∃ z : ℤ, (3 ^ 2 = 9 ∧ 3 ^ 4 = 81) ∧ z = (9 + 81) / 2 :=
by
  -- the proof goes here
  sorry

end arithmetic_sequence_z_value_l353_353028


namespace checkerboard_squares_containing_at_least_7_black_squares_l353_353063

def checkerboard := List (List Bool)

def is_alternating (board : checkerboard) : Prop :=
  board.length = 10 ∧ (∀ row, row ∈ board → row.length = 10) ∧ 
  (∀ i j, i < 10 → j < 10 → (board[i][j] = (i + j) % 2 = 0))

def contains_at_least_7_black_squares (board : checkerboard) (x y size : ℕ) : Prop :=
  let sub_board := List.get (List.map (List.get ·) (List.drop y board) (List.take size board))
                             (List.drop x board)
  (sum (List.map sum (List.map (λ row, List.filter id row) sub_board)) ≥ 7)

def count_squares_with_7_black_squares (board : checkerboard) : ℕ :=
  ∑ size in [1..10], ∑ y in [0..(10 - size)], ∑ x in [0..(10 - size)],
    if contains_at_least_7_black_squares board x y size then 1 else 0

theorem checkerboard_squares_containing_at_least_7_black_squares 
  (board : checkerboard) (h : is_alternating board) : 
  count_squares_with_7_black_squares board = 115 := sorry

end checkerboard_squares_containing_at_least_7_black_squares_l353_353063


namespace money_weed_eating_l353_353366

-- Define the amounts and conditions
def money_mowing : ℕ := 68
def money_per_week : ℕ := 9
def weeks : ℕ := 9
def total_money : ℕ := money_per_week * weeks

-- Define the proof that the money made weed eating is 13 dollars
theorem money_weed_eating :
  total_money - money_mowing = 13 := sorry

end money_weed_eating_l353_353366


namespace ellipse_area_l353_353665

theorem ellipse_area (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1 = -5) (h2 : y1 = 0) (h3 : x2 = 15) (h4 : y2 = 0)
  (h5 : x3 = 14) (h6 : y3 = 6)
  (passes_through : ∃ b : ℝ, (x3 - (x1 + x2) / 2)^2 / (20 / 2)^2 + y3^2 / b^2 = 1) :
  ∃ b : ℝ, 36 / b^2 = 1 - 81 / 100 ∧
  (20 / 2) * (b) = 10 * (sqrt (3600 / 19)) ∧
  ∃ π, area =  (π * 600 / (sqrt 19)) :=
by
  sorry

end ellipse_area_l353_353665


namespace prob1_part1_prob1_part2_find_integers_x_prob3_min_value_prob4_l353_353871

-- Problem 1
theorem prob1_part1 : |-4 + 6| = 2 := sorry
theorem prob1_part2 : |-2 - 4| = 6 := sorry

-- Problem 2
theorem find_integers_x :
  {x : ℤ | |x + 2| + |x - 1| = 3} = {-2, -1, 0, 1} :=
sorry

-- Problem 3
theorem prob3 (a : ℤ) (h : -4 ≤ a ∧ a ≤ 6) : |a + 4| + |a - 6| = 10 :=
sorry

-- Problem 4
theorem min_value_prob4 :
  ∃ (a : ℤ), |a - 1| + |a + 5| + |a - 4| = 9 ∧ ∀ (b : ℤ), |b - 1| + |b + 5| + |b - 4| ≥ 9 :=
sorry

end prob1_part1_prob1_part2_find_integers_x_prob3_min_value_prob4_l353_353871


namespace pirate_loot_sum_base_10_l353_353995

theorem pirate_loot_sum_base_10 :
  let silverware := 42135,
      tiaras := 31015,
      scarves := 2025,
      silverware_base_10 := 3 * 5^0 + 1 * 5^1 + 2 * 5^2 + 4 * 5^3,
      tiaras_base_10 := 1 * 5^0 + 0 * 5^1 + 1 * 5^2 + 3 * 5^3,
      scarves_base_10 := 2 * 5^0 + 0 * 5^1 + 2 * 5^2
  in silverware_base_10 + tiaras_base_10 + scarves_base_10 = 1011 := by
  sorry

end pirate_loot_sum_base_10_l353_353995


namespace count_integers_between_sqrt10_sqrt100_l353_353728

theorem count_integers_between_sqrt10_sqrt100 :
  ∃ n : ℕ, n = 7 ∧ card {x : ℤ | real.sqrt 10 < x ∧ x < real.sqrt 100} = n :=
by
  sorry

end count_integers_between_sqrt10_sqrt100_l353_353728


namespace count_integers_between_sqrt10_sqrt100_l353_353729

theorem count_integers_between_sqrt10_sqrt100 :
  ∃ n : ℕ, n = 7 ∧ card {x : ℤ | real.sqrt 10 < x ∧ x < real.sqrt 100} = n :=
by
  sorry

end count_integers_between_sqrt10_sqrt100_l353_353729


namespace solution_l353_353228

def coprime (a b : ℕ) : Prop :=
  ∃ x y : ℤ, x * a + y * b = 1

def admissible_set (a b : ℕ) (S : set ℕ) : Prop :=
  (0 ∈ S) ∧ ∀ k ∈ S, ((k + a) ∈ S) ∧ ((k + b) ∈ S)

noncomputable def f (a b : ℕ) : ℕ :=
  (Nat.choose (a + b) a) / (a + b)

theorem solution (a b : ℕ) (h : coprime a b) : 
  ∃ S : set ℕ, admissible_set a b S → f(a, b) = (Nat.choose (a + b) a) / (a + b) := 
sorry

end solution_l353_353228


namespace sixth_element_row_20_l353_353445

theorem sixth_element_row_20 : (Nat.choose 20 5) = 15504 := by
  sorry

end sixth_element_row_20_l353_353445


namespace largest_integer_less_than_100_div_8_rem_5_l353_353619

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l353_353619


namespace binom_eq_fraction_l353_353926

open Nat

theorem binom_eq_fraction (n : ℕ) (h_pos : 0 < n) : choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_eq_fraction_l353_353926


namespace find_BD_when_AC_over_AB_min_l353_353282

open Real

-- Define the conditions
variables (A B C D : Point)
variables (AD BC BD CD : ℝ)
variables (angle_ADB : ℝ)
variables (AC AB : ℝ)

-- Assign conditions
def conditions : Prop := 
  angle_ADB = 120 ∧
  AD = 2 ∧
  CD = 2 * BD ∧
  AC / AB = minimized

-- Target to prove
theorem find_BD_when_AC_over_AB_min (A B C D : Point) (BD : ℝ)
  (h : conditions A B C D 2 120 BD 2 * BD AC AB) :
  BD = sqrt 3 - 1 :=
sorry

end find_BD_when_AC_over_AB_min_l353_353282


namespace hyperbola_eccentricity_l353_353007

theorem hyperbola_eccentricity
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_hyperbola : b * b = c * c - a * a)
  (h_AB_CD : 3 * (2 * b * b / a) = 2 * (2 * b * c / a)) :
  let e := c / a in
  e = 3 * real.sqrt 5 / 5 := 
begin
  sorry
end

end hyperbola_eccentricity_l353_353007


namespace old_edition_pages_l353_353078

theorem old_edition_pages (x : ℕ) (h : 2 * x - 230 = 450) : x = 340 :=
by {
  have eq1 : 2 * x = 450 + 230, from eq_add_of_sub_eq h,
  have eq2 : 2 * x = 680, from eq1,
  have eq3 : x = 680 / 2, from eq_of_mul_eq_mul_right (by norm_num) eq2,
  norm_num at eq3,
  exact eq3,
}

end old_edition_pages_l353_353078


namespace min_value_of_c_l353_353250

variables {a b c S : ℝ} -- Defining the variables

theorem min_value_of_c (h1 : c * cos B = a + 1/2 * b) (h2 : S = sqrt 3 / 12 * c) :
  c ≥ 1 :=
sorry

end min_value_of_c_l353_353250


namespace ratio_of_roots_l353_353640

variable (a b c x1 x2 : ℝ)

theorem ratio_of_roots :
  (a ≠ 0) → (x1 = 4 * x2) → (a * x1^2 + b * x1 + c = 0) → (a * x2^2 + b * x2 + c = 0) → 
  (16 * b^2 / (a * c) = 100) :=
by
  intros ha hx1 root1 root2
  sorry

end ratio_of_roots_l353_353640


namespace smallest_nonprime_with_large_primes_l353_353831

theorem smallest_nonprime_with_large_primes
  (n : ℕ)
  (h1 : n > 1)
  (h2 : ¬ Prime n)
  (h3 : ∀ p : ℕ, Prime p → p ∣ n → p ≥ 20) :
  660 < n ∧ n ≤ 670 :=
sorry

end smallest_nonprime_with_large_primes_l353_353831


namespace part_I_part_II_l353_353248

-- Given conditions:
section
variables {A B C : ℝ} {a b c m : ℝ} [real]

-- condition: m = 5/4 and b = 1
variables (h1 : sin A + sin C = m * sin B)
variables (h2 : a * c = (1 / 4) * b ^ 2)

-- Additional conditions for Part (I)
variables (m_val : m = 5 / 4) (b_val : b = 1)

-- Part (I) Specific Result: find values of a and c
theorem part_I : (a = 1 ∧ c = 1 / 4) ∨ (a = 1 / 4 ∧ c = 1) := sorry

end

-- Part (II) Given B is acute
section
variables {B : ℝ} [real]

-- Additional conditions and resulting range for m
theorem part_II (h3 : 0 < cos B ∧ cos B < 1) : 
  (sqrt 6 / 2 < m ∧ m < sqrt 2) := sorry
end

end part_I_part_II_l353_353248


namespace largest_integer_less_than_100_div_8_rem_5_l353_353611

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l353_353611


namespace geom_seq_inequality_l353_353211

open Real

-- Definition of a geometric sequence with first term a1 and common ratio q
def geom_seq (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^(n-1)

-- Sum of the first n terms of a geometric sequence
def geom_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

theorem geom_seq_inequality
  (a1 q : ℝ) (hq : q < 0) :
  let a9 := geom_seq a1 q 9 in
  let S8 := geom_sum a1 q 8 in
  let a8 := geom_seq a1 q 8 in
  let S9 := geom_sum a1 q 9 in
  a9 * S8 > a8 * S9 :=
by
  sorry

end geom_seq_inequality_l353_353211


namespace dakota_total_medical_bill_l353_353518

theorem dakota_total_medical_bill : 
    let days_in_hospital := 3
    let hospital_bed_charge_per_day := 900
    let specialist_charge_per_hour := 250
    let specialists_count := 2
    let specialist_visit_time_per_specialist_per_day_in_hours := 0.25
    let ambulance_cost := 1800
    let surgery_duration_in_hours := 2
    let surgeon_charge_per_hour := 1500
    let assistant_charge_per_hour := 800
    let physical_therapy_charge_per_hour := 300 
    let physical_therapy_time_per_day_in_hours := 1
    let medication_A_cost_per_pill := 20
    let medication_A_dosage_per_day := 3
    let medication_B_cost_per_pill := 45
    let medication_B_dosage_per_day := 3
    let medication_C_cost_per_hour := 80
    let medication_C_infusion_time_per_day_in_hours := 2
    in (hospital_bed_charge_per_day * days_in_hospital 
        + specialist_charge_per_hour * specialist_visit_time_per_specialist_per_day_in_hours * specialists_count * days_in_hospital 
        + ambulance_cost 
        + (surgeon_charge_per_hour + assistant_charge_per_hour) * surgery_duration_in_hours 
        + physical_therapy_charge_per_hour * physical_therapy_time_per_day_in_hours * days_in_hospital 
        + medication_A_cost_per_pill * medication_A_dosage_per_day * days_in_hospital 
        + medication_B_cost_per_pill * medication_B_dosage_per_day * days_in_hospital 
        + medication_C_cost_per_hour * medication_C_infusion_time_per_day_in_hours * days_in_hospital 
    ) = 12190 := 
by 
    sorry

end dakota_total_medical_bill_l353_353518


namespace measure_angle_BCD_l353_353254

-- Define the geometry premises
variables (circle : Type) [circle.basic]
variables (F B D C A : circle.point)
variables (FB DC AB FD : circle.segment)
variables {angle_AFB angle_ABF angle_BCD : ℝ}

-- Define the conditions
def geometry_conditions :=
  FB.is_diameter ∧
  FB.is_parallel_to DC ∧
  AB.is_parallel_to FD ∧
  angle_AFB / angle_ABF = 3 / 4

-- Define the theorem to be proven
theorem measure_angle_BCD (h : geometry_conditions circle F B D C A FB DC AB FD angle_AFB angle_ABF) :
  angle_BCD = 52 :=
sorry

end measure_angle_BCD_l353_353254


namespace angle_BAC_is_60_l353_353060

theorem angle_BAC_is_60
    (A B C K M P : Point)
    (triangle_ABC : Triangle A B C)
    (K_on_AB : K ∈ Segment A B)
    (M_on_AC : M ∈ Segment A C)
    (intersection_P : IntersectAt P (Line B M) (Line C K))
    (angle_APB : angle A P B = 120°)
    (angle_BPC : angle B P C = 120°)
    (angle_CPA : angle C P A = 120°)
    (area_AKPM : Area (Quadrilateral A K P M))
    (area_BPC : Area (Triangle B P C))
    (area_equality : AreaA = AreaB)
    : angle A B C = 60° := 
begin
    sorry
end

end angle_BAC_is_60_l353_353060


namespace greatest_nat_not_sum_of_two_composites_l353_353555

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

theorem greatest_nat_not_sum_of_two_composites :
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ 11 = a + b ∧
  (∀ n : ℕ, n > 11 → ¬ ∃ x y : ℕ, is_composite x ∧ is_composite y ∧ n = x + y) :=
sorry

end greatest_nat_not_sum_of_two_composites_l353_353555


namespace problem_1_problem_2_l353_353503

-- Given these two conditions and the corresponding questions.
theorem problem_1 (O_1 O_2 : Type) [circle O_1] [circle O_2] 
  (A B C: Point) (M N P Q: Point)
  (B_1 C_1: Point) (MN: Line)
  (tangent_circles: are_tangent_externally O_1 O_2 A)
  (diameter_AB: diameter O_1 A B)
  (diameter_AC: diameter O_2 A C)
  (midpoint_P: is_midpoint P M N)
  (tangent_line_MN: is_tangent_line_between M N)
  (circle_O_through_BPC: passes_through_circle B P C -> some O)
  (O1_intersection_B1: intersects_again O_1 B_1)
  (O2_intersection_C1: intersects_again O_2 C_1)
  (intersection_line_BB1_CC1: intersects_lines BB_1 CC_1 Q) :
  (6 * (distance P Q) = length MN) :=
sorry

theorem problem_2 (O_1 O_2 : Type) [circle O_1] [circle O_2] 
  (A B C: Point) (M N P Q: Point)
  (B_1 C_1: Point) (MN: Line)
  (tangent_circles: are_tangent_externally O_1 O_2 A)
  (diameter_AB: diameter O_1 A B)
  (diameter_AC: diameter O_2 A C)
  (midpoint_P: is_midpoint P M N)
  (tangent_line_MN: is_tangent_line_between M N)
  (circle_O_through_BPC: passes_through_circle B P C -> some O)
  (O1_intersection_B1: intersects_again O_1 B_1)
  (O2_intersection_C1: intersects_again O_2 C_1) 
  (intersection_line_BB1_CC1: intersects_lines BB_1 CC_1 Q) :
  cyclic_quad M N C_1 B_1 :=
sorry

end problem_1_problem_2_l353_353503


namespace number_of_correct_propositions_is_zero_l353_353649

-- Define the propositions as conditions
def P1 : Prop := ∀ (C : Type) [cylinder C], all_congruent_rectangular_faces C → right_cylinder C
def P2 : Prop := ∀ (H : Type) [hexahedron H], diagonal_faces_congruent_rectangles H → rectangular_parallelepiped H
def P3 : Prop := ∀ (C : Type) [cylinder C], two_lateral_faces_perpendicular_to_base C → right_cylinder C
def P4 : Prop := ∀ (R : Type) [rectangular_parallelepiped R], right_square_prism R

-- The problem statement as a theorem to be proved
theorem number_of_correct_propositions_is_zero :
  ¬P1 ∧ ¬P2 ∧ ¬P3 ∧ ¬P4 :=
by
  sorry

end number_of_correct_propositions_is_zero_l353_353649


namespace binom_eq_fraction_l353_353925

open Nat

theorem binom_eq_fraction (n : ℕ) (h_pos : 0 < n) : choose n 2 = n * (n - 1) / 2 :=
by
  sorry

end binom_eq_fraction_l353_353925


namespace find_BD_when_AC_over_AB_min_l353_353322

-- Definition of the conditions
variables (A B C D : Type) [metric_space A]
variables (triangle : set A)
variables (on_side_BC : D ∈ segment B C)
variables (angle_ADB : ∠ADB = 120)
variables (AD_len : dist A D = 2)
variables (CD_2BD : dist C D = 2 * dist B D)

noncomputable def minimum_ratio_condition (BD : ℝ) : Prop := 
  let x := dist B D
  let CD := 2 * x
  let AD := 2 in
  let angle_ADB_cos := -1/2 in
  let b_squared := AD^2 + CD^2 - 2 * AD * CD * angle_ADB_cos in
  let c_squared := AD^2 + x^2 + 2 * AD * x * angle_ADB_cos in
  b_squared / c_squared = 4 - (12 / (x + 1 + 3/(x + 1))) → x = sqrt 3 - 1

theorem find_BD_when_AC_over_AB_min (BD : ℝ) : minimum_ratio_condition A B C D 
:= sorry

end find_BD_when_AC_over_AB_min_l353_353322


namespace calculate_altitude_l353_353991

-- Define the conditions
def Speed_up : ℕ := 18
def Speed_down : ℕ := 24
def Avg_speed : ℝ := 20.571428571428573

-- Define what we want to prove
theorem calculate_altitude : 
  2 * Speed_up * Speed_down / (Speed_up + Speed_down) = Avg_speed →
  (864 : ℝ) / 2 = 432 :=
by
  sorry

end calculate_altitude_l353_353991


namespace evaluate_divisor_sum_l353_353123

noncomputable def num_divisors (n : ℕ) : ℕ :=
  (n.factors.foldr (λ p card a, (card + 1) * a) 1)

theorem evaluate_divisor_sum (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hneq_pq : p ≠ q) (hneq_pr : p ≠ r) (hneq_qr : q ≠ r) :
  let a := p^4
  let b := q * r
  let k := a^5
  let m := b^2
  num_divisors k + num_divisors m = 30 := 
by
  sorry

end evaluate_divisor_sum_l353_353123


namespace number_of_mismatching_socks_l353_353384

-- Define the conditions
def total_socks : Nat := 25
def pairs_of_matching_socks : Nat := 4
def socks_per_pair : Nat := 2
def matching_socks : Nat := pairs_of_matching_socks * socks_per_pair

-- State the theorem
theorem number_of_mismatching_socks : total_socks - matching_socks = 17 :=
by
  -- Skip the proof
  sorry

end number_of_mismatching_socks_l353_353384


namespace evaluate_expression_l353_353136

theorem evaluate_expression :
  ⌈5 * (8 - (3 / 4)) - 2.5⌉ = 34 := 
by
  sorry

end evaluate_expression_l353_353136


namespace ratio_proof_l353_353103

variable (c x y : ℝ)

def loss_eq (c : ℝ) := x = 0.80 * c

def profit_eq (c : ℝ) := y = 1.25 * c

theorem ratio_proof (h1 : x = 0.80 * c) (h2 : y = 1.25 * c) : y / x = 25 / 16 :=
by
  rw [h1, h2]
  have hc_ne_zero : c ≠ 0 := sorry
  field_simp [hc_ne_zero]
  norm_num

end ratio_proof_l353_353103


namespace largest_integer_less_than_100_with_remainder_5_l353_353610

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l353_353610


namespace mary_investment_amount_l353_353841

theorem mary_investment_amount
  (A : ℝ := 100000) -- Future value in dollars
  (r : ℝ := 0.08) -- Annual interest rate
  (n : ℕ := 12) -- Compounded monthly
  (t : ℝ := 10) -- Time in years
  : (⌈A / (1 + r / n) ^ (n * t)⌉₊ = 45045) :=
by
  sorry

end mary_investment_amount_l353_353841


namespace integral_solution_l353_353058

noncomputable def integral_problem (f : ℝ → ℝ := λ x, tan x * log (cos x)): ℝ → ℝ := 
  -((log (cos f))^2) / 2 + const

theorem integral_solution (C : ℝ):
  ∫ (tan x * log (cos x)) dx = -((log (cos x))^2) / 2 + C :=
sorry

end integral_solution_l353_353058


namespace log_4_8_l353_353141

theorem log_4_8 : log 4 8 = 3 / 2 :=
by
  -- Sorry is used to skip the proof
  sorry

end log_4_8_l353_353141


namespace polynomial_factor_determines_c_l353_353132

theorem polynomial_factor_determines_c (q k c : ℚ) : (k = 8 ∧ q = 8 / 3) → c = -32 / 3 :=
by
  intro h
  cases h with h1 h2
  rw [h1, h2]
  sorry

end polynomial_factor_determines_c_l353_353132


namespace largest_integer_less_than_100_div_8_rem_5_l353_353618

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l353_353618


namespace soccer_ball_problem_l353_353069

-- Definitions of conditions
def price_eqs (x y : ℕ) : Prop :=
  x + 2 * y = 800 ∧ 3 * x + 2 * y = 1200

def total_cost_constraint (m : ℕ) : Prop :=
  200 * m + 300 * (20 - m) ≤ 5000 ∧ 1 ≤ m ∧ m ≤ 19

def store_discounts (x y : ℕ) (m : ℕ) : Prop :=
  200 * m + (3 / 5) * 300 * (20 - m) = (200 * m + (3 / 5) * 300 * (20 - m))

-- Main problem statement
theorem soccer_ball_problem :
  ∃ (x y m : ℕ), price_eqs x y ∧ total_cost_constraint m ∧ store_discounts x y m :=
sorry

end soccer_ball_problem_l353_353069


namespace find_k_value_l353_353525

theorem find_k_value (k : ℝ) : 
    (∀ x : ℂ, 5 * x^2 + 7 * x + k = 0 → x = (-7 + complex.I * real.sqrt 399) / 10 ∨ x = (-7 - complex.I * real.sqrt 399) / 10) →
    k = 22.4 :=
by
  sorry

end find_k_value_l353_353525


namespace total_number_of_apples_l353_353865

namespace Apples

def red_apples : ℕ := 7
def green_apples : ℕ := 2
def total_apples : ℕ := red_apples + green_apples

theorem total_number_of_apples : total_apples = 9 := by
  -- Definition of total_apples is used directly from conditions.
  -- Conditions state there are 7 red apples and 2 green apples.
  -- Therefore, total_apples = 7 + 2 = 9.
  sorry

end Apples

end total_number_of_apples_l353_353865


namespace find_length_AP_l353_353267

def square_side_length (s : ℝ) : Prop := s = 6
def rect_side_lengths : Prop := ZY = 10 ∧ XY = 6
def perpendicular (l1 l2 : ℝ) : Prop := l1 = 6 ∧ l2 = 10  -- Here AD is perpendicular to WX
def shaded_area_half : Prop := shaded_area = (10 * 6) / 2  -- The shaded area is half of WXYZ

theorem find_length_AP (AP AD PD : ℝ) (s : ℝ) (ZY XY shaded_area : ℝ) :
  square_side_length s → 
  rect_side_lengths → 
  perpendicular AD ZY →
  shaded_area_half → 
  AD = s →
  DC = s →
  shaded_area = PD * DC →
  AP = AD - PD :=
begin
  intros h_square h_rect h_perp h_shaded h_AD h_DC h_area,
  have PD_eq : PD = shaded_area / DC,
  { rw h_area,
    rw h_DC,
    rw h_shaded,
    norm_num,
    },
  rw PD_eq,
  rw h_AD,
  rw h_DC,
  norm_num,
end

end find_length_AP_l353_353267


namespace parallel_lines_l353_353673

variables (line : Type) (plane : Type)
variables (m n : line) (α β γ : plane)

def parallel_line_to_plane (l : line) (π : plane) : Prop := sorry
def parallel_planes (π₁ π₂ : plane) : Prop := sorry
def intersection_line_plane (π₁ π₂ : plane) (l : line) : Prop := sorry

axiom non_coincident_lines : ¬(m = n)
axiom pairwise_non_coincident_planes : (α ≠ β) ∧ (α ≠ γ) ∧ (β ≠ γ)

theorem parallel_lines (h_parallel_planes : parallel_planes α β)
(h_alpha_intersection : intersection_line_plane α γ m)
(h_beta_intersection : intersection_line_plane β γ n) :
parallel_line_to_plane m n :=
sorry

end parallel_lines_l353_353673


namespace sum_k_pow_k_div_101_eq_29_l353_353511

def sum_k_pow_k_mod_101 : ℕ :=
  ∑ k in Finset.range (30303 + 1), k ^ k % 101

theorem sum_k_pow_k_div_101_eq_29 :
  sum_k_pow_k_mod_101 % 101 = 29 :=
sorry

end sum_k_pow_k_div_101_eq_29_l353_353511


namespace largest_integer_less_than_100_with_remainder_5_l353_353575

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353575


namespace largest_integer_lt_100_with_remainder_5_div_8_l353_353601

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l353_353601


namespace min_distance_l353_353297

def triangle_ABC (A B C D : Type) (BD : ℝ) :=
∃ (CD : ℝ), CD = 2 * BD ∧ 
  ∃ (AD AC AB : ℝ), AD = 2 ∧ 
    ∃ (α β γ : ℝ), α = 120 ∧ 
      angle_ADB = α ∧ 
        (AC / AB = (sqrt (3) - 1))

theorem min_distance (A B C D : Type) (BD : ℝ) :
  ∃ D ∈ BC, ∃ α β γ AD AC AB, 
    α = 120 ∧ AD = 2 ∧ D ∈ BC ∧ 
      γ = ∠ADB = α ∧ 
        CD = 2 * BD →
          (AC / AB) = (sqrt (3) - 1) := 
sorry

end min_distance_l353_353297


namespace min_ratio_bd_l353_353313

noncomputable def triangle_ratio_min (BD : ℝ) : Prop :=
  ∃ K L : ℝ, 
    (K = 1) ∧ 
    (L = 1) ∧ 
    (∀ (x : ℝ), ∀ (y : ℝ), (CD = 2 * BD) → ((x = sqrt(3) - 1) → y = (bd + 1 + 3 / (bd + 1)) ∧ (x + y) = 2 * K * L) ∧ 
    (abs x ≤ (sqrt 3 - 1)))


theorem min_ratio_bd (BD : ℝ) : triangle_ratio_min BD :=
by { sorry }

end min_ratio_bd_l353_353313


namespace cyclic_quadrilaterals_count_l353_353432

noncomputable def num_cyclic_quadrilaterals (n : ℕ) : ℕ :=
  if n = 32 then 568 else 0 -- encapsulating the problem's answer

theorem cyclic_quadrilaterals_count :
  num_cyclic_quadrilaterals 32 = 568 :=
sorry

end cyclic_quadrilaterals_count_l353_353432


namespace money_has_48_l353_353050

-- Definitions derived from conditions:
def money (p : ℝ) := 
  p = (1/3 * p) + 32

-- The main theorem statement
theorem money_has_48 (p : ℝ) : money p → p = 48 := by
  intro h
  -- Skipping the proof
  sorry

end money_has_48_l353_353050


namespace quotient_polynomial_division_l353_353030

noncomputable def polynomial_division (p q : Polynomial ℤ) : Polynomial ℤ × Polynomial ℤ :=
  Polynomial.div_mod_by_monic p q

theorem quotient_polynomial_division :
  (polynomial_division (Polynomial.C (12 : ℤ) * X^4 - Polynomial.C (9 : ℤ) * X^3 + Polynomial.C (6 : ℤ) * X^2 + Polynomial.C (11 : ℤ) * X - Polynomial.C (3 : ℤ))
                       (Polynomial.C (3 : ℤ) * X - Polynomial.C (2 : ℤ))).1
  = Polynomial.C (4 : ℤ) * X^3 + Polynomial.C (1 : ℤ) * X^2 + Polynomial.C (2 : ℤ) * X + Polynomial.C (3 : ℤ) :=
begin
  sorry
end

end quotient_polynomial_division_l353_353030


namespace largest_integer_less_than_100_with_remainder_5_l353_353581

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353581


namespace trig_identity_example_l353_353134

theorem trig_identity_example :
  sin (15 * Real.pi / 180) * sin (105 * Real.pi / 180) - cos (15 * Real.pi / 180) * cos (105 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trig_identity_example_l353_353134


namespace product_sign_and_units_digit_l353_353436
-- Import necessary libraries

-- Define the sequence of odd negative integers strictly greater than -2005
def odd_neg_integers := {x : ℤ | x < 0 ∧ x % 2 ≠ 0 ∧ -2005 < x}

-- The count of this sequence
def count := 1002

-- The theorem statement
theorem product_sign_and_units_digit:
  (∏ x in odd_neg_integers, x) > 0 ∧ (∏ x in odd_neg_integers, x) % 10 = 5 :=
sorry

end product_sign_and_units_digit_l353_353436


namespace minimum_val_BE_DE_CD_l353_353251

/-- 
Given a triangle BAC with ∠BAC = 50°, AB = 8, and AC = 7. Points D and E lie on AB, AC respectively. 
Prove that the minimum possible value of BE + DE + CD is equal to √(113 + 56√3).
-/
theorem minimum_val_BE_DE_CD :
  ∀ (BAC : Triangle) (D E : Point), 
  (BAC.angle BAC.vertex = 50) ∧ 
  (BAC.side_length BAC.edge1 = 8) ∧ 
  (BAC.side_length BAC.edge2 = 7) ∧ 
  (D ∈ BAC.edge1) ∧ 
  (E ∈ BAC.edge2) → 
  BE + DE + CD = real.sqrt (113 + 56 * real.sqrt 3) :=
by
  sorry

end minimum_val_BE_DE_CD_l353_353251


namespace solution_set_l353_353838

open Real

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the differentiable function f

axiom differentiable_f : Differentiable ℝ f
axiom condition_f : ∀ x, f x > 0 ∧ x * (deriv (deriv (deriv f))) x > 0

theorem solution_set :
  {x : ℝ | 1 ≤ x ∧ x < 2} =
    {x : ℝ | f (sqrt (x + 1)) > sqrt (x - 1) * f (sqrt (x ^ 2 - 1))} :=
sorry

end solution_set_l353_353838


namespace ben_hits_7_l353_353374

def scores : List (String × Nat) :=
  [("Alice", 18), ("Ben", 9), ("Cindy", 15), ("Dave", 14), ("Ellen", 19), ("Frank", 8)]

def total_score (name : String) : Nat :=
  (scores.find_x? (fun x => x.1 = name)).getD ("", 0) |>.2

theorem ben_hits_7 :
  ∃ p1 p2 : Nat, p1 ≠ p2 ∧ p1 + p2 = total_score "Ben" ∧ (p1 = 7 ∨ p2 = 7) ∧
  ∀ name ∈ ["Alice", "Cindy", "Dave", "Ellen", "Frank"], 
    ∃ q1 q2 : Nat, q1 ≠ q2 ∧ q1 + q2 = total_score name ∧ q1 ≠ 7 ∧ q2 ≠ 7 := 
by
  sorry

end ben_hits_7_l353_353374


namespace tangent_expression_l353_353205

open Real

theorem tangent_expression
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (geom_seq : ∀ n m, a (n + m) = a n * a m) 
  (arith_seq : ∀ n, b (n + 1) = b n + (b 2 - b 1))
  (cond1 : a 1 * a 6 * a 11 = -3 * sqrt 3)
  (cond2 : b 1 + b 6 + b 11 = 7 * pi) :
  tan ( (b 3 + b 9) / (1 - a 4 * a 8) ) = -sqrt 3 :=
sorry

end tangent_expression_l353_353205


namespace positive_integer_count_l353_353642

/-
  Prove that the number of positive integers \( n \) for which \( \frac{n(n+1)}{2} \) divides \( 30n \) is 11.
-/

theorem positive_integer_count (n : ℕ) :
  (∃ k : ℕ, k > 0 ∧ k ≤ 11 ∧ (2 * 30 * n) % (n * (n + 1)) = 0) :=
sorry

end positive_integer_count_l353_353642


namespace largest_integer_less_than_100_with_remainder_5_l353_353629

theorem largest_integer_less_than_100_with_remainder_5 (x : ℕ) : 
  x < 100 ∧ x % 8 = 5 → x = 93 :=
by {
  intros,
  sorry
}

end largest_integer_less_than_100_with_remainder_5_l353_353629


namespace sequence_is_arithmetic_l353_353803

theorem sequence_is_arithmetic {a : ℕ → ℝ} (h_pos : ∀ n, a n > 0)
  (h_second_term : a 2 = 3 * a 1)
  (h_sqrt_seq_arith : ∃ d : ℝ, ∀ n, real.sqrt (∑ i in finset.range (n + 1), a i) = d * n + real.sqrt (a 0)): 
  ∃ d : ℝ, ∀ n, a n = a 0 + d * n := 
by
  sorry

end sequence_is_arithmetic_l353_353803


namespace total_amount_divided_l353_353978

theorem total_amount_divided (A B C : ℝ) (h1 : A = 2/3 * (B + C)) (h2 : B = 2/3 * (A + C)) (h3 : A = 80) : 
  A + B + C = 200 :=
by
  sorry

end total_amount_divided_l353_353978


namespace allan_balloons_l353_353097

theorem allan_balloons (a j t : ℕ) (h1 : t = 6) (h2 : j = 4) (h3 : t = a + j) : a = 2 := by
  sorry

end allan_balloons_l353_353097


namespace largest_integer_less_than_100_with_remainder_5_l353_353609

theorem largest_integer_less_than_100_with_remainder_5 (n : ℕ) : n < 100 ∧ n % 8 = 5 → n ≤ 93 := 
  begin
    sorry
  end

end largest_integer_less_than_100_with_remainder_5_l353_353609


namespace rectangle_perimeter_l353_353911

-- Conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def area_of_triangle (base height : ℕ) : ℕ :=
  (base * height) / 2

def area_of_rectangle (length width : ℕ) : ℕ :=
  length * width

def perimeter_of_rectangle (length width : ℕ) : ℕ :=
  2 * (length + width)

-- Given conditions from the problem
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15
def width_of_rectangle : ℕ := 6

-- Main theorem
theorem rectangle_perimeter :
  is_right_triangle a b c →
  area_of_triangle a b = area_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle →
  perimeter_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle = 30 :=
by
  sorry

end rectangle_perimeter_l353_353911


namespace expand_expression_l353_353145

variable {R : Type} [CommRing R]
variable (a b x : R)

theorem expand_expression (a b x : R) :
  (a * x^2 + b) * (5 * x^3) = 35 * x^5 + (-15) * x^3 :=
by
  -- The proof goes here
  sorry

end expand_expression_l353_353145


namespace largest_integer_less_than_100_with_remainder_5_l353_353579

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l353_353579


namespace sum_of_base5_numbers_l353_353506

-- Definitions for the numbers in base 5
def n1_base5 := (1 * 5^2 + 3 * 5^1 + 2 * 5^0 : ℕ)
def n2_base5 := (2 * 5^2 + 1 * 5^1 + 4 * 5^0 : ℕ)
def n3_base5 := (3 * 5^2 + 4 * 5^1 + 1 * 5^0 : ℕ)

-- Sum the numbers in base 10
def sum_base10 := n1_base5 + n2_base5 + n3_base5

-- Define the base 5 value of the sum
def sum_base5 := 
  -- Convert the sum to base 5
  1 * 5^3 + 2 * 5^2 + 4 * 5^1 + 2 * 5^0

-- The theorem we want to prove
theorem sum_of_base5_numbers :
    (132 + 214 + 341 : ℕ) = 1242 := by
    sorry

end sum_of_base5_numbers_l353_353506


namespace hyperbola_eccentricity_l353_353658

-- Define the hyperbola, the conditions and prove that the eccentricity e is (sqrt(5) + 1)/2
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (F1 F2 O P Q B : ℝ × ℝ)
  (h_hyperbola : ∀ x y, (x, y) ∈ set_of (λ p: ℝ × ℝ, p.1^2 / a^2 - p.2^2 / b^2 = 1))
  (h_circle : x^2 + y^2 = (a^2 + b^2))
  (h_anglePOF2_eq_angleQOB : ∠ P O F2 = ∠ Q O B) :
  let e := (sqrt 5 + 1) / 2 in
  eccentricity (a, b) = e :=
sorry

end hyperbola_eccentricity_l353_353658


namespace largest_integer_lt_100_with_remainder_5_div_8_l353_353595

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l353_353595


namespace min_distance_l353_353292

def triangle_ABC (A B C D : Type) (BD : ℝ) :=
∃ (CD : ℝ), CD = 2 * BD ∧ 
  ∃ (AD AC AB : ℝ), AD = 2 ∧ 
    ∃ (α β γ : ℝ), α = 120 ∧ 
      angle_ADB = α ∧ 
        (AC / AB = (sqrt (3) - 1))

theorem min_distance (A B C D : Type) (BD : ℝ) :
  ∃ D ∈ BC, ∃ α β γ AD AC AB, 
    α = 120 ∧ AD = 2 ∧ D ∈ BC ∧ 
      γ = ∠ADB = α ∧ 
        CD = 2 * BD →
          (AC / AB) = (sqrt (3) - 1) := 
sorry

end min_distance_l353_353292


namespace convex_quadrilateral_perimeter_l353_353984

noncomputable def perimeter (ABCD : ConvexQuadrilateral) (P: Point)
                              (h1 : area ABCD = 2601)
                              (h2 : distance P A = 25)
                              (h3 : distance P B = 35)
                              (h4 : distance P C = 30)
                              (h5 : distance P D = 50)
                              (h6 : orthogonal_diagonals_intersect_at_P ABCD P) : Real :=
  sqrt 1850 + sqrt 2125 + sqrt 3400 + sqrt 3125

theorem convex_quadrilateral_perimeter (ABCD : ConvexQuadrilateral) (P: Point)
                                       (h1 : area ABCD = 2601)
                                       (h2 : distance P A = 25)
                                       (h3 : distance P B = 35)
                                       (h4 : distance P C = 30)
                                       (h5 : distance P D = 50)
                                       (h6 : orthogonal_diagonals_intersect_at_P ABCD P) :
  perimeter ABCD P h1 h2 h3 h4 h5 h6 = sqrt 1850 + sqrt 2125 + sqrt 3400 + sqrt 3125 := 
by sorry

end convex_quadrilateral_perimeter_l353_353984


namespace sequence_accumulating_is_arithmetic_l353_353807

noncomputable def arithmetic_sequence {α : Type*} [LinearOrderedField α]
  (a : ℕ → α) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

theorem sequence_accumulating_is_arithmetic
  {α : Type*} [LinearOrderedField α] (a : ℕ → α) (S : ℕ → α)
  (na_gt_zero : ∀ n, a n > 0)
  (ha2 : a 2 = 3 * a 1)
  (hS_arith : arithmetic_sequence (λ n, (S n)^(1/2)))
  (hSn : ∀ n, S n = (∑ i in Finset.range (n+1), a i)) :
  arithmetic_sequence a := 
sorry

end sequence_accumulating_is_arithmetic_l353_353807


namespace angies_monthly_salary_l353_353502

theorem angies_monthly_salary 
    (necessities_expense : ℕ)
    (taxes_expense : ℕ)
    (left_over : ℕ)
    (monthly_salary : ℕ) :
  necessities_expense = 42 → 
  taxes_expense = 20 → 
  left_over = 18 → 
  monthly_salary = necessities_expense + taxes_expense + left_over → 
  monthly_salary = 80 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end angies_monthly_salary_l353_353502


namespace baseball_cards_start_count_l353_353043

theorem baseball_cards_start_count (X : ℝ) 
  (h1 : ∃ (x : ℝ), x = (X + 1) / 2)
  (h2 : ∃ (x' : ℝ), x' = X - ((X + 1) / 2) - 1)
  (h3 : ∃ (y : ℝ), y = 3 * (X - ((X + 1) / 2) - 1))
  (h4 : ∃ (z : ℝ), z = 18) : 
  X = 15 :=
by
  sorry

end baseball_cards_start_count_l353_353043


namespace gifts_left_l353_353509

variable (initial_gifts : ℕ)
variable (gifts_sent : ℕ)

theorem gifts_left (h_initial : initial_gifts = 77) (h_sent : gifts_sent = 66) : initial_gifts - gifts_sent = 11 := by
  sorry

end gifts_left_l353_353509


namespace probability_of_divisibility_l353_353913

noncomputable def probability_smaller_divides_larger : ℚ :=
  let S := {1, 2, 3, 4, 5, 6}
  let pairs := { (x, y) | x ∈ S ∧ y ∈ S ∧ x < y }
  let successful_pairs := { (x, y) ∈ pairs | x ∣ y }
  (successful_pairs.card : ℚ) / pairs.card

theorem probability_of_divisibility :
  probability_smaller_divides_larger = 7 / 15 :=
by
  sorry

end probability_of_divisibility_l353_353913


namespace four_dice_probability_l353_353378

open ProbabilityTheory
open Classical

noncomputable def dice_prob_space : ProbabilitySpace := sorry -- Define the probability space of rolling six 6-sided dice

def condition_no_four_of_a_kind (dice_outcome : Vector ℕ 6) : Prop :=
  ¬∃ n, dice_outcome.count n ≥ 4

def condition_pair_exists (dice_outcome : Vector ℕ 6) : Prop :=
  ∃ n, dice_outcome.count n = 2

def re_rolled_dice (initial_outcome : Vector ℕ 6) (re_roll : Vector ℕ 4) : Vector ℕ 6 :=
  sorry -- Combine initial pair and re-rolled outcomes

def at_least_four_same (dice_outcome : Vector ℕ 6) : Prop :=
  ∃ n, dice_outcome.count n ≥ 4

theorem four_dice_probability :
  ∀ (initial_outcome : Vector ℕ 6)
    (re_roll : Vector ℕ 4),
  (condition_no_four_of_a_kind initial_outcome) →
  (condition_pair_exists initial_outcome) →
  (∃ pr : ℚ, pr = 311 / 648 ∧ 
    (Pr[dice_prob_space, at_least_four_same (re_rolled_dice initial_outcome re_roll)] = pr)) :=
sorry

end four_dice_probability_l353_353378


namespace unique_element_set_l353_353226

noncomputable def unique_element_condition (a : ℝ) : Prop :=
  ∃ x, (ax^2 + 2 * x + 1 = 0) ∧ ∀ y, (ay^2 + 2 * y + 1 = 0) → y = x

theorem unique_element_set (a : ℝ) : unique_element_condition a ↔ (a = 0 ∨ a = 1) := by
  sorry

end unique_element_set_l353_353226


namespace largest_integer_lt_100_with_remainder_5_div_8_l353_353593

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l353_353593


namespace hyperbola_eq_and_line_eq_l353_353692

noncomputable def hyperbola (a b : ℝ) : set (ℝ × ℝ) := 
  {p | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

theorem hyperbola_eq_and_line_eq (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (F₁ F₂ : ℝ × ℝ) (h_F₁ : F₁ = (-2, 0)) (h_F₂ : F₂ = (2, 0))
  (P : ℝ × ℝ) (h_P : P = (3, real.sqrt 7)) 
  (Q : ℝ × ℝ) (h_Q : Q = (0, 2)) 
  (O : ℝ × ℝ) (h_O : O = (0, 0))
  (h_area_OEF : ∃ E F, 
    E ∈ hyperbola a b ∧ F ∈ hyperbola a b ∧ 
    ∃ l : ℝ → ℝ, (∀ x, l x = x) ∧ l(Q.1) = Q.2 ∧
    (1/2) * abs ((E.1 * F.2) - (E.2 * F.1)) = 2 * real.sqrt 2) : 
  (a = real.sqrt 2 ∧ b = real.sqrt 2 ∧ ∀ x y, 
    \frac{x^2}{2} - \frac{y^2}{2} = 1) ∧ 
  (∀ k, k = real.sqrt 2 ∨ k = -real.sqrt 2 ∧ 
    (∀ x, 2 + k * x = Q.2)) :=
sorry

end hyperbola_eq_and_line_eq_l353_353692


namespace intersection_point_of_curve_and_line_l353_353549

theorem intersection_point_of_curve_and_line :
  (∃ θ : ℝ, sin θ = -1 ∧ sin θ * sin θ = 1) ∧ (-1 + 2 = 1) :=
by
  sorry

end intersection_point_of_curve_and_line_l353_353549


namespace minimum_BD_value_l353_353307

theorem minimum_BD_value {A B C D : Type*} [Real A] [Real B] [Real C] [Real D] (angle_ADB: A) (AD: A) (CD: A) (BD: A):
  (D ∈ line_segment(B, C)) →
  angle_ADB = 120 → 
  AD = 2 → 
  CD = 2 * BD →
  ∃ BD: A, BD = sqrt(3) - 1 :=
by
  sorry

end minimum_BD_value_l353_353307


namespace largest_integer_lt_100_with_remainder_5_div_8_l353_353598

theorem largest_integer_lt_100_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
begin
  use 93,
  split,
  { norm_num }, -- n < 100
  split,
  { norm_num }, -- n % 8 = 5
  { intros m hm1 hm2,
    linarith }, -- For all m, m < 100 and m % 8 = 5, m <= n
end

end largest_integer_lt_100_with_remainder_5_div_8_l353_353598


namespace soccer_playing_time_configurations_l353_353108

theorem soccer_playing_time_configurations :
  ∃ (C1 : ℕ) (C2 : ℕ) (C3 : ℕ),
  C1 = nat.choose 32 3 * nat.choose 2 2 ∧
  C2 = nat.choose 19 3 * nat.choose 9 2 ∧
  C3 = nat.choose 6 3 * nat.choose 16 2 ∧
  (C1 + C2 + C3 = 42244) :=
by
  let C1 := nat.choose 32 3 * nat.choose 2 2
  let C2 := nat.choose 19 3 * nat.choose 9 2
  let C3 := nat.choose 6 3 * nat.choose 16 2
  exact ⟨C1, C2, C3, rfl, rfl, rfl, rfl⟩

end soccer_playing_time_configurations_l353_353108


namespace count_integers_between_sqrt10_sqrt100_l353_353715

theorem count_integers_between_sqrt10_sqrt100 : 
  ∃ (S : Set ℤ), (∀ n, n ∈ S ↔ (real.sqrt 10 < n ∧ n < real.sqrt 100)) ∧ S.card = 6 := 
by
  sorry

end count_integers_between_sqrt10_sqrt100_l353_353715


namespace complex_expression_evaluation_l353_353816

-- Define the complex number ω
def ω : ℂ := Complex.cos (6 * Real.pi / 11) + Complex.sin (6 * Real.pi / 11) * Complex.I

-- Prove the given expression equals -2
theorem complex_expression_evaluation : 
  (ω / (1 + ω^3) + ω^2 / (1 + ω^6) + ω^3 / (1 + ω^9) = -2) :=
by
  sorry

end complex_expression_evaluation_l353_353816


namespace range_of_m_l353_353229

theorem range_of_m (m : ℝ) : 
  (¬(-2 ≤ 1 - (x - 1) / 3 ∧ (1 - (x - 1) / 3 ≤ 2)) → (∀ x, m > 0 → x^2 - 2*x + 1 - m^2 > 0)) → 
  (40 ≤ m ∧ m < 50) :=
by
  sorry

end range_of_m_l353_353229


namespace distance_between_points_is_sqrt_59_l353_353551

-- Definitions for the points and distance formula
def point1 : ℝ × ℝ × ℝ := (2, 1, -4)
def point2 : ℝ × ℝ × ℝ := (5, 8, -3)

-- Definition of the Euclidean distance in 3D space
def euclidean_distance_3d (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2).sqrt

-- Statement to prove
theorem distance_between_points_is_sqrt_59 :
  euclidean_distance_3d point1 point2 = Real.sqrt 59 :=
by
  sorry

end distance_between_points_is_sqrt_59_l353_353551


namespace product_of_constants_factoring_quadratic_l353_353161

theorem product_of_constants_factoring_quadratic :
  let p := ∏ t in {t | ∃ a b : ℤ, ab = -24 ∧ t = a + b}, t 
  in p = 5290000 := by
sorry

end product_of_constants_factoring_quadratic_l353_353161


namespace minimum_seats_occupied_l353_353001

theorem minimum_seats_occupied (total_seats : ℕ) (h : total_seats = 180) : 
  ∃ occupied_seats : ℕ, occupied_seats = 45 ∧ 
  ∀ additional_person,
    (∀ i : ℕ, i < total_seats → 
     (occupied_seats ≤ i → i < occupied_seats + 1 ∨ i > occupied_seats + 1)) →
    additional_person = occupied_seats + 1  :=
by
  sorry

end minimum_seats_occupied_l353_353001


namespace probability_divide_event_l353_353916

-- Define the set of integers from 1 to 6
def my_set : finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the total number of ways to choose 2 distinct numbers from the set
def total_pairs : ℕ := nat.choose 6 2

-- Define the set of pairs where (a, b) such that a < b and a divides b
def successful_pairs : finset (ℕ × ℕ) :=
  (my_set.product my_set).filter (λ (ab : ℕ × ℕ), 
    ab.1 < ab.2 ∧ ab.1 ∣ ab.2)

-- Calculate the probability of successful outcome
def probability_success : ℚ :=
  (successful_pairs.card:ℚ) / (total_pairs:ℚ)

theorem probability_divide_event : 
  probability_success = 8 / 15 := 
by sorry

end probability_divide_event_l353_353916


namespace largest_int_with_remainder_5_lt_100_l353_353574

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l353_353574


namespace div_rule_2701_is_37_or_73_l353_353368

theorem div_rule_2701_is_37_or_73 (a b x : ℕ) (h1 : 10 * a + b = x) (h2 : a^2 + b^2 = 58) : 
  (x = 37 ∨ x = 73) ↔ 2701 % x = 0 :=
by
  sorry

end div_rule_2701_is_37_or_73_l353_353368


namespace parallel_lines_l353_353888

theorem parallel_lines (a : ℝ) :
  (∀ x y, x + a^2 * y + 6 = 0 → (a - 2) * x + 3 * a * y + 2 * a = 0) ↔ (a = 0 ∨ a = -1) :=
by
  sorry

end parallel_lines_l353_353888


namespace residents_ticket_price_l353_353977

theorem residents_ticket_price
  (total_attendees : ℕ)
  (resident_count : ℕ)
  (non_resident_price : ℝ)
  (total_revenue : ℝ)
  (R : ℝ)
  (h1 : total_attendees = 586)
  (h2 : resident_count = 219)
  (h3 : non_resident_price = 17.95)
  (h4 : total_revenue = 9423.70)
  (total_residents_pay : ℝ := resident_count * R)
  (total_non_residents_pay : ℝ := (total_attendees - resident_count) * non_resident_price)
  (h5 : total_revenue = total_residents_pay + total_non_residents_pay) :
  R = 12.95 := by
  sorry

end residents_ticket_price_l353_353977


namespace abcd_inequality_l353_353173

noncomputable def max_value (a b c d : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) : Prop :=
  abcd (a + b + c + d) / ((a + b)^2 * (c + d)^2) <= 1 / 4

theorem abcd_inequality (a b c d : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) : max_value a b c d h_a h_b h_c h_d :=
sorry

end abcd_inequality_l353_353173


namespace ceil_sqrt_200_eq_15_l353_353535

theorem ceil_sqrt_200_eq_15 (h1 : Real.sqrt 196 = 14) (h2 : Real.sqrt 225 = 15) (h3 : 196 < 200 ∧ 200 < 225) : 
  Real.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l353_353535


namespace find_length_PQ_l353_353777

noncomputable def length_PQ : ℝ :=
  let RS := 15
  let tan_S := 2
  let tan_Q := 3
  let PR := RS * tan_S
  let PQ := PR / tan_Q
  PQ

theorem find_length_PQ (RS : ℝ) (tan_S : ℝ) (tan_Q : ℝ) :
  RS = 15 → tan_S = 2 → tan_Q = 3 → length_PQ = 10 := by
  intros hRS htS htQ
  unfold length_PQ
  rw [hRS, htS, htQ]
  norm_num
  exact rfl

end find_length_PQ_l353_353777


namespace largest_int_less_than_100_with_remainder_5_l353_353565

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l353_353565


namespace old_edition_pages_l353_353074

theorem old_edition_pages (x : ℕ) 
  (h₁ : 2 * x - 230 = 450) : x = 340 := 
by sorry

end old_edition_pages_l353_353074


namespace variance_of_data_set_is_two_l353_353195

-- Define the data set
def data_set : List ℝ := [8, 10, 9, 12, 11]

-- Define the function to calculate the mean
def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

-- Define the function to calculate the variance
def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x - μ) ^ 2)).sum / data.length

-- The theorem statement
theorem variance_of_data_set_is_two :
  variance data_set = 2 := by
  sorry -- proof to be provided

end variance_of_data_set_is_two_l353_353195


namespace range_of_a_for_function_min_max_l353_353689

theorem range_of_a_for_function_min_max 
  (a : ℝ) 
  (h_min : ∀ x ∈ [-1, 1], x = -1 → x^2 + a * x + 3 ≤ y) 
  (h_max : ∀ x ∈ [-1, 1], x = 1 → x^2 + a * x + 3 ≥ y) : 
  2 ≤ a := 
sorry

end range_of_a_for_function_min_max_l353_353689


namespace angle_in_third_or_fourth_quadrant_l353_353235

theorem angle_in_third_or_fourth_quadrant (α : ℝ) (h : cos α * tan α < 0) : 
  (π < α ∧ α < 3 * π / 2) ∨ (3 * π / 2 < α ∧ α < 2 * π) :=
sorry

end angle_in_third_or_fourth_quadrant_l353_353235


namespace max_value_f_l353_353404

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + Real.log x

theorem max_value_f : 
  ∃ (x : ℝ), x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1 ∧ 
             (∀ y ∈ Set.Icc (1 / Real.exp 1) Real.exp 1, f y ≤ f x) ∧ 
             f x = - (1 / 2) := 
sorry

end max_value_f_l353_353404


namespace book_pages_total_l353_353371

-- Definitions based on conditions
def pages_first_three_days: ℕ := 3 * 28
def pages_next_three_days: ℕ := 3 * 35
def pages_following_three_days: ℕ := 3 * 42
def pages_last_day: ℕ := 15

-- Total pages read calculated from above conditions
def total_pages_read: ℕ :=
  pages_first_three_days + pages_next_three_days + pages_following_three_days + pages_last_day

-- Proof problem statement: prove that the total pages read equal 330
theorem book_pages_total:
  total_pages_read = 330 :=
by
  sorry

end book_pages_total_l353_353371


namespace sum_cubes_squared_l353_353114

theorem sum_cubes_squared :
  let pos_sum := ∑ n in Finset.range 50, (n + 1)^3
  let neg_sum := ∑ n in Finset.range 50, (-(n + 1))^3
  (pos_sum + neg_sum) ^ 2 = 0 :=
by
  sorry

end sum_cubes_squared_l353_353114


namespace derivative_of_y_l353_353826

variable {x : ℝ}

def y : ℝ := x * Real.cos x

theorem derivative_of_y : deriv y x = Real.cos x - x * Real.sin x :=
by 
  sorry

end derivative_of_y_l353_353826


namespace third_vertex_of_right_triangle_l353_353433

theorem third_vertex_of_right_triangle (x : ℝ) :
  let A := (0 : ℝ, 0 : ℝ),
      B := (3 : ℝ, 3 : ℝ),
      C := (x, 0) in
  (0 < x) ∧ (1 / 2 * 3 * x = 18) → C = (12, 0) :=
sorry

end third_vertex_of_right_triangle_l353_353433


namespace binom_two_eq_l353_353943

theorem binom_two_eq (n : ℕ) (h : 0 < n) : nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binom_two_eq_l353_353943


namespace distance_from_center_to_point_l353_353113

open Real

def circle_center (h : ℝ) (k : ℝ) := (h, k)
def distance (p1 p2 : ℝ × ℝ) : ℝ := sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_from_center_to_point :
  ∀ (x y : ℝ),
  (∀ (h k : ℝ), x^2 - 4 * x + y^2 - 6 * y = -3 → circle_center h k = (2, 3)) →
  distance (circle_center 2 3) (10, 3) = 8 :=
by
  sorry

end distance_from_center_to_point_l353_353113


namespace polynomial_factorization_l353_353798

noncomputable def rational_polynomials (x y z : ℚ) := ℚ[x, y, z]

theorem polynomial_factorization :
  ∀ P : rational_polynomials, P ≠ 0 → ∃ Q R : rational_polynomials, Q ≠ 0 ∧ R ≠ 0 ∧
  (∀ x y z : ℚ, R (x^2 * y) (y^2 * z) (z^2 * x) = P x y z * Q x y z) :=
by sorry

end polynomial_factorization_l353_353798


namespace length_of_train_is_155_96_meters_l353_353494

def train_speed_kmh := 45 -- Speed of the train in km/h
def train_time_seconds := 40 -- Time taken to cross the bridge in seconds
def bridge_length_meters := 344.04 -- Length of the bridge in meters

def kmh_to_mps (kmh : Float) : Float := kmh * 1000 / 3600

def total_distance_covered (speed_mps : Float) (time_seconds : Float) : Float :=
  speed_mps * time_seconds

def length_of_train (total_distance : Float) (bridge_length : Float) : Float :=
  total_distance - bridge_length

theorem length_of_train_is_155_96_meters :
  length_of_train
    (total_distance_covered (kmh_to_mps train_speed_kmh) train_time_seconds)
    bridge_length_meters = 155.96 :=
by
  sorry

end length_of_train_is_155_96_meters_l353_353494


namespace chess_tournament_games_l353_353759

theorem chess_tournament_games (P : ℕ) (TotalGames : ℕ) (hP : P = 21) (hTotalGames : TotalGames = 210) : 
  ∃ G : ℕ, G = 20 ∧ TotalGames = (P * (P - 1)) / 2 :=
by
  sorry

end chess_tournament_games_l353_353759


namespace ceil_sqrt_200_eq_15_l353_353539

theorem ceil_sqrt_200_eq_15 : Int.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l353_353539


namespace range_of_a_l353_353220

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → (differentiable_at ℝ (λ x, x^2 - 4 * a * x + 1) x ∧ 
    deriv (λ x, x^2 - 4 * a * x + 1) x ≥ 0)) → a ≤ 1 / 2 :=
sorry

end range_of_a_l353_353220
