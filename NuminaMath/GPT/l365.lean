import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Floor
import Mathlib.Algebra.Function.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Log
import Mathlib.Algebra.Order.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Order.Floor
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Base
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.NumberTheory.Prime
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace problem_I_problem_II_l365_365393

noncomputable def parabola_focus (C : XReal ** YReal -> Prop) : XReal * YReal := (1,0)
noncomputable def x_axis_intersection (d : XReal -> Option (XReal * YReal)) : XReal * YReal := (-1,0)
noncomputable def trajectory_eqn (P : XReal ** YReal) : Prop := P.2 ^ 2 = 4 * P.1 - 12
noncomputable def line_eqn_min_area (l : XReal ** YReal -> Prop) : Prop := ∀ P : XReal * YReal, (d d P).1 = 1

theorem problem_I (C : XReal -> XReal -> Prop) (l : XReal ** YReal -> Prop) (F : XReal ** YReal := parabola_focus C) 
(E : XReal ** YReal := x_axis_intersection d) (P : XReal ** YReal) : 
C P.1 P.2 = (∃ A B, l A ∧ l B ∧ (∇(E,P) = (∇(E,B) + ∇(E,A)))) → trajectory_eqn P :=
sorry

theorem problem_II (C : XReal ** YReal -> Prop) (l : XReal ** YReal -> Prop) (F : XReal ** YReal := parabola_focus C) 
(E : XReal ** YReal := x_axis_intersection d) : 
(∀ P : XReal ** YReal, C P.1 P.2 = (∇(E,P) = (∇(E,B) + ∇(E,A)))) → 
line_eqn_min_area l :=
sorry

end problem_I_problem_II_l365_365393


namespace original_number_l365_365860

theorem original_number (x : ℝ) (hx : 1000 * x = 9 * (1 / x)) : 
  x = 3 * (Real.sqrt 10) / 100 :=
by
  sorry

end original_number_l365_365860


namespace justin_run_time_l365_365720

theorem justin_run_time : 
  let flat_ground_rate := 2 / 2 -- Justin runs 2 blocks in 2 minutes on flat ground
  let uphill_rate := 2 / 3 -- Justin runs 2 blocks in 3 minutes uphill
  let total_blocks := 10 -- Justin is 10 blocks from home
  let uphill_blocks := 6 -- 6 of those blocks are uphill
  let flat_ground_blocks := total_blocks - uphill_blocks -- Remainder are flat ground
  let flat_ground_time := flat_ground_blocks * flat_ground_rate
  let uphill_time := uphill_blocks * uphill_rate
  let total_time := flat_ground_time + uphill_time
  total_time = 13 := 
by 
  sorry

end justin_run_time_l365_365720


namespace polygon_area_is_correct_l365_365175

-- Define vertices as terms
def vertices := [(2, 1), (4, 3), (7, 1), (4, 6)]

-- Define a function to calculate the area of a polygon using the Shoelace Theorem
def polygon_area (verts : List (ℕ × ℕ)) : ℚ :=
  let n := verts.length
  let mut sum1 := 0
  let mut sum2 := 0
  for i in [0:n] do
    if i < n - 1 then
      sum1 := sum1 + verts[i].fst * verts[i+1].snd
      sum2 := sum2 + verts[i].snd * verts[i+1].fst
    else
      sum1 := sum1 + verts[i].fst * verts[0].snd
      sum2 := sum2 + verts[i].snd * verts[0].fst
  0.5 *  ((sum1 - sum2).natAbs : ℚ)

-- Statement to prove
theorem polygon_area_is_correct : polygon_area vertices = 7.5 :=
by
  sorry

end polygon_area_is_correct_l365_365175


namespace cylinder_new_volume_l365_365819

-- Define the original volume condition
def original_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Define the new volume condition
def new_volume (r h : ℝ) : ℝ := Real.pi * (3 * r)^2 * (2 * h)

-- Theorem statement to prove the new volume is 360 cubic feet
theorem cylinder_new_volume (r h : ℝ) (h1 : original_volume r h = 20) : new_volume r h = 360 :=
sorry

end cylinder_new_volume_l365_365819


namespace quadrilateral_relation_proof_l365_365391

variables (ad dc ab bc ak ck al cl bk dk bl dl : ℝ)
variables (K L : Type)

-- Problem statement in Lean
theorem quadrilateral_relation_proof 
    (h1 : ad + dc = ab + bc 
    ∨ ak + ck = al + cl 
    ∨ bk + dk = bl + dl) : 
    (ad + dc = ab + bc 
    ∧ ak + ck = al + cl 
    ∧ bk + dk = bl + dl) :=
begin
  sorry,
end

end quadrilateral_relation_proof_l365_365391


namespace integral_result_l365_365933

theorem integral_result (a k : ℝ) (h_a : 0 < a) (h_k : 0 < k) :
  ∫ x in 0..∞, (x * sin (a * x)) / (x^2 + k^2) = (π / 2) * real.exp (-a * k) :=
sorry

end integral_result_l365_365933


namespace range_of_m_l365_365978

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x + |x - 1| > m) → m < 1 :=
by
  sorry

end range_of_m_l365_365978


namespace smallest_nat_mul_47_last_four_digits_l365_365183

theorem smallest_nat_mul_47_last_four_digits (N : ℕ) :
  (47 * N) % 10000 = 1969 ↔ N = 8127 :=
sorry

end smallest_nat_mul_47_last_four_digits_l365_365183


namespace probability_valid_n_is_1_over_5_l365_365244

def satisfies_equation (n m : ℤ) : Prop := n * m - 6 * n - 3 * m = 3

def valid_n (n : ℤ) : Prop := ∃ m : ℤ, satisfies_equation n m

def n_in_range (n : ℤ) : Prop := 1 ≤ n ∧ n ≤ 15

theorem probability_valid_n_is_1_over_5 :
  (∑ n in Finset.filter n_in_range (Finset.range 16), if valid_n n then 1 else 0 : ℤ)/(Finset.card (Finset.filter n_in_range (Finset.range 16))) = 1/5 := sorry

end probability_valid_n_is_1_over_5_l365_365244


namespace pq_relation_l365_365643

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := sin (2 * x + φ)

def p : ℝ := f (2 * π / 3) (π / 18)
def q : ℝ := f (5 * π / 6) (π / 18)
def r : ℝ := f (7 * π / 6) (π / 18)

theorem pq_relation :
  p < q ∧ q < r :=
sorry

end pq_relation_l365_365643


namespace find_k_l365_365976

theorem find_k (k n m : ℕ) (hk : k > 0) (hn : n > 0) (hm : m > 0) 
  (h : (1 / (n ^ 2 : ℝ) + 1 / (m ^ 2 : ℝ)) = (k : ℝ) / (n ^ 2 + m ^ 2)) : k = 4 :=
sorry

end find_k_l365_365976


namespace find_triples_l365_365722

theorem find_triples (x y p : ℤ) (prime_p : Prime p) :
  x^2 - 3 * x * y + p^2 * y^2 = 12 * p ↔ 
  (p = 3 ∧ ( (x = 6 ∧ y = 0) ∨ (x = -6 ∧ y = 0) ∨ (x = 4 ∧ y = 2) ∨ (x = -2 ∧ y = 2) ∨ (x = 2 ∧ y = -2) ∨ (x = -4 ∧ y = -2) ) ) := 
by
  sorry

end find_triples_l365_365722


namespace decreasing_exponential_quadratic_l365_365640

theorem decreasing_exponential_quadratic {f : ℝ → ℝ} (a : ℝ) 
    (h : ∀ x y ∈ Ioo 0 1, x < y → f x ≥ f y) :
    a ≥ 2 :=
begin
    sorry
end

end decreasing_exponential_quadratic_l365_365640


namespace candy_unclaimed_after_second_round_l365_365507

theorem candy_unclaimed_after_second_round :
  ∀ (x : ℝ), x - ((4 / 10) * x + (3 / 10) * x + (2 / 10) * x + (1 / 10) * x) = 0 :=
by
  intro x
  simp
  done

end candy_unclaimed_after_second_round_l365_365507


namespace find_x_six_l365_365118

noncomputable def positive_real : Type := { x : ℝ // 0 < x }

theorem find_x_six (x : positive_real)
  (h : (1 - x.val ^ 3) ^ (1/3) + (1 + x.val ^ 3) ^ (1/3) = 1) :
  x.val ^ 6 = 28 / 27 := 
sorry

end find_x_six_l365_365118


namespace first_term_of_geometric_sequence_l365_365018

noncomputable def geometric_sequence_first_term
  (t₅ t₆ : ℚ) (h₅ : t₅ = 48) (h₆ : t₆ = 72) (r : ℚ) (a : ℚ) : Prop :=
  t₆ = r * t₅ ∧ t₅ = a * r^4 ∧ a = 256/27

theorem first_term_of_geometric_sequence
  (t₅ t₆ : ℚ) (h₅ : t₅ = 48) (h₆ : t₆ = 72) : geometric_sequence_first_term t₅ t₆ h₅ h₆ (3/2) (256/27) :=
by
  unfold geometric_sequence_first_term
  split
  simp
  split
  simp
  admit

end first_term_of_geometric_sequence_l365_365018


namespace optionA_optionB_optionD_l365_365109

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, f(x) + f(4 + x) = 0
axiom condition2 : ∀ x : ℝ, f(2 + 2x) = f(2 - 2x)
axiom condition3 : f(1) = 1

theorem optionA : ∀ x : ℝ, f(-x) = -f(x) :=
by
  intros
  sorry

theorem optionB : f(2023) = -1 :=
by
  sorry

theorem optionD : ∑ k in finset.range 100, (k+1) * f(2*(k+1) - 1) = -100 :=
by
  sorry

end optionA_optionB_optionD_l365_365109


namespace susie_earnings_l365_365380

/-- Given the conditions:
  1. Susie babysits every day for 3 hours a day at the rate of $10 per hour.
  2. She spent 3/10 of the money she earned from last week to buy a make-up set.
  3. She then spent 2/5 of her money on her skincare products.
  Prove that Susie has 88.2 dollars left from her earnings last week.
-/
theorem susie_earnings (h1 : ∀ d, d = 7 → (3 * 10) * d = 210)
                       (h2 : ∀ t, t = 210 → (3/10 * t) = 63)
                       (h3 : ∀ r, r = (210 - 63) → (2/5 * r) = 58.8)
                       (h4 : ∀ l, l = (147 - 58.8) → l = 88.2) :
  88.2 = 88.2 :=
begin
  sorry
end

end susie_earnings_l365_365380


namespace solve_for_s_l365_365374

theorem solve_for_s (s : ℝ) (t : ℝ) (h1 : t = 8 * s^2) (h2 : t = 4.8) : s = Real.sqrt 0.6 ∨ s = -Real.sqrt 0.6 := by
  sorry

end solve_for_s_l365_365374


namespace investment_equality_l365_365895

noncomputable def average_interest_rate (total : ℝ) (rate1 rate2 : ℝ) (investment1 investment2 : ℝ) : ℝ :=
(let interest1 := rate1 * investment1 in
 let interest2 := rate2 * investment2 in
 (interest1 + interest2) / total)

theorem investment_equality (total x : ℝ) : 
  let rate1 := 0.03
  let rate2 := 0.07
  let investment1 := 6000 - x
  let investment2 := x
  let interest1 := rate1 * investment1
  let interest2 := rate2 * investment2
  interest1 = interest2 → 
  average_interest_rate 6000 0.03 0.07 (6000 - x) x = 0.042 :=
by
  sorry

end investment_equality_l365_365895


namespace divide_number_l365_365159

theorem divide_number (x : ℝ) (h : 0.3 * x = 0.2 * (80 - x) + 10) : min x (80 - x) = 28 := 
by 
  sorry

end divide_number_l365_365159


namespace total_cups_used_l365_365035

theorem total_cups_used (butter flour sugar : ℕ) (h1 : 2 * sugar = 3 * butter) (h2 : 5 * sugar = 3 * flour) (h3 : sugar = 12) : butter + flour + sugar = 40 :=
by
  sorry

end total_cups_used_l365_365035


namespace sue_cost_l365_365377

def cost_of_car : ℝ := 2100
def total_days_in_week : ℝ := 7
def sue_days : ℝ := 3

theorem sue_cost : (cost_of_car * (sue_days / total_days_in_week)) = 899.99 :=
by
  sorry

end sue_cost_l365_365377


namespace min_cubes_required_l365_365891

/--
A lady builds a box with dimensions 10 cm length, 18 cm width, and 4 cm height using 12 cubic cm cubes. Prove that the minimum number of cubes required to build the box is 60.
-/
def min_cubes_for_box (length width height volume_cube : ℕ) : ℕ :=
  (length * width * height) / volume_cube

theorem min_cubes_required :
  min_cubes_for_box 10 18 4 12 = 60 :=
by
  -- The proof details are omitted.
  sorry

end min_cubes_required_l365_365891


namespace plus_sign_segment_sum_455_l365_365875

theorem plus_sign_segment_sum_455 :
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 100 ∧
           (let cell_sum := 7 * x in
            cell_sum = 455 ∧ x % 10 ≠ 0 ∧ x % 10 ≠ 1 ∧ div x 10 ≠ 0 ∧ div x 10 ≠ 9) :=
by
  sorry

end plus_sign_segment_sum_455_l365_365875


namespace compound_interest_rate_l365_365816

theorem compound_interest_rate 
  (PV FV : ℝ) (n : ℕ) (r : ℝ) 
  (hPV : PV = 200) 
  (hFV : FV = 242) 
  (hn : n = 2) 
  (h_eq : PV = FV / (1 + r) ^ n) : 
  r = 0.1 := 
by
  sorry

end compound_interest_rate_l365_365816


namespace number_of_correct_statements_l365_365803

-- Given Conditions:
def statement1 : Prop := ∀ x : ℝ, irrational x → ¬ ∃ y : ℝ, x = y^2
def statement2 : Prop := ∀ x : ℝ, irrational x ↔ infinite (decimal x) ∧ non_repeating (decimal x)
def statement3 : Prop := irrational 0
def statement4 : Prop := ∀ x : ℝ, irrational x → ∃ y : ℝ, x = point_on_number_line y

-- Define the correctness of each statement
def is_correct (s : Prop) : Prop := s

-- The math proof problem
theorem number_of_correct_statements :
  (is_correct statement1 = false ∧ is_correct statement2 = true ∧ is_correct statement3 = false ∧ is_correct statement4 = true) →
  number_of_true [statement1, statement2, statement3, statement4] = 2 :=
sorry

end number_of_correct_statements_l365_365803


namespace temperature_difference_product_l365_365926

theorem temperature_difference_product (N : ℤ) (L : ℤ) (M : ℤ)
  (h1 : M = L + N)
  (h2 : M - 10 = (L + N) - 10)
  (h3 : (L + 5) = L + 5)
  (h4 : abs ((L + N - 10) - (L + 5)) = 6) :
  ∃ N1 N2, N1 * N2 = 189 :=
by
  let N1 := 21
  let N2 := 9
  existsi N1
  existsi N2
  show N1 * N2 = 189, from rfl

#check temperature_difference_product

end temperature_difference_product_l365_365926


namespace min_magnitude_roots_le_one_l365_365653

open Complex

theorem min_magnitude_roots_le_one 
  (z1 z2 z3 ω1 ω2 : ℂ) 
  (mag_z1 : |z1| ≤ 1) 
  (mag_z2 : |z2| ≤ 1) 
  (mag_z3 : |z3| ≤ 1)
  (root_eq : ∀ z, (z - z1) * (z - z2) + (z - z2) * (z - z3) + (z - z3) * (z - z1) = 3 * (z - ω1) * (z - ω2)) :
  (∀ j, j ∈ {1, 2, 3} → (min (|z j - ω1|) (|z_j - ω2|)) ≤ 1) :=
sorry

end min_magnitude_roots_le_one_l365_365653


namespace positive_integer_divisors_of_sum_l365_365986

theorem positive_integer_divisors_of_sum (n : ℕ) :
  (∃ n_values : Finset ℕ, 
    (∀ n ∈ n_values, n > 0 
      ∧ (n * (n + 1)) ∣ (2 * 10 * n)) 
      ∧ n_values.card = 5) :=
by
  sorry

end positive_integer_divisors_of_sum_l365_365986


namespace inverse_proportion_quadrants_l365_365020

theorem inverse_proportion_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∃ x y : ℝ, x = -2 ∧ y = 3 ∧ y = k / x) →
  (∀ x : ℝ, (x < 0 → k / x > 0) ∧ (x > 0 → k / x < 0)) :=
sorry

end inverse_proportion_quadrants_l365_365020


namespace simplify_sum_l365_365002

theorem simplify_sum (n : ℕ) (x : ℝ) :
  ∑ k in Finset.range (n + 1), (cos (3^k * x) + 3 * cos (3^(k-1) * x)) / (3^(k-1) * sin (3^k * x)) = 
  1 / 2 * (3 * (Real.cot x) - (Real.cot (3^n * x)) / (3^(n-1))) :=
by
  sorry

end simplify_sum_l365_365002


namespace triangle_angle_A_triangle_side_a_l365_365707

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (h1 : (a - c)*(a + c)*sin C = c * (b - c) * sin B)
variable (area : ℝ)
variable (h2 : area = sqrt 3)
variable (h3 : sin B * sin C = 1/4)

theorem triangle_angle_A (h1 : (a - c)*(a + c)*sin C = c * (b - c) * sin B) : A = π / 3 := 
sorry

theorem triangle_side_a (h1 : (a - c)*(a + c)*sin C = c * (b - c) * sin B)
                        (h2 : area = sqrt 3)
                        (h3 : sin B * sin C = 1/4) : a = 2 * sqrt 3 := 
sorry

end triangle_angle_A_triangle_side_a_l365_365707


namespace simple_interest_years_l365_365403

variable (P : ℝ) (r1 r2 : ℝ) (n : ℕ) (CI_half SI P_simple t : ℝ) 

-- Given conditions
def conditions (CI_half = 420) : Prop :=
  (SI = P_simple * r1 / 100 * t) ∧
  (SI = CI_half) ∧
  (P_simple = 1750) ∧
  (r1 = 8) ∧
  (CI_half = CI / 2) ∧
  (CI = P * (1 + r2 / 100)^n - P) ∧
  (P = 4000) ∧
  (n = 2) ∧
  (r2 = 10)

-- Prove that the number of years t is 3
theorem simple_interest_years : conditions CI_half SI P_simple t -> t = 3 :=
by
  intro h
  sorry

end simple_interest_years_l365_365403


namespace combination_sum_l365_365584

theorem combination_sum :
  (∑ n in Finset.range 9 \ Finset.range 2, Nat.choose (n+2) 2) = Nat.choose 11 3 :=
by
  sorry

end combination_sum_l365_365584


namespace percentage_problem_l365_365260

theorem percentage_problem (P : ℕ) : (P / 100 * 400 = 20 / 100 * 700) → P = 35 :=
by
  intro h
  sorry

end percentage_problem_l365_365260


namespace voting_probability_l365_365054

theorem voting_probability (n m : ℕ) (h : n > m) : 
  (m + n > 0) → 
  let total_sequences := Nat.choose (n + m) n in
  let valid_sequences := (n - m) * total_sequences / (n + m) in
  valid_sequences / total_sequences = (n - m : ℚ) / (n + m : ℚ) := 
sorry

end voting_probability_l365_365054


namespace area_of_union_of_square_and_circle_l365_365501

-- Definitions based on problem conditions
def side_length : ℕ := 8
def radius : ℝ := 12
def square_area := side_length ^ 2
def circle_area := Real.pi * radius ^ 2
def overlapping_area := 1 / 4 * circle_area
def union_area := square_area + circle_area - overlapping_area

-- The problem statement in Lean format
theorem area_of_union_of_square_and_circle :
  union_area = 64 + 108 * Real.pi :=
by
  sorry

end area_of_union_of_square_and_circle_l365_365501


namespace ratio_x_y_l365_365679

theorem ratio_x_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : x / y = 13 / 2 :=
by
  sorry

end ratio_x_y_l365_365679


namespace AE_AF_identity_l365_365404

-- Definition for the given conditions
def point := ℝ × ℝ

-- Coordinates for points A, B, C, and D based on condition (1)
def A : point := (0, 1)
def B : point := (1, 1)
def C : point := (1, 0)
def D : point := (0, 0)

-- Definition for point E on side CD
def E (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : point := (x, 0)

-- Definition for distance function
def dist (P Q : point) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Function to calculate |AE| and |AF|
def AE (x hx) := dist (A) (E x hx)
def AF : ℝ := dist (A) (C)

-- The main statement to be proved
theorem AE_AF_identity (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : 
  (1 / (AE x hx) ^ 2) + (1 / (AF)^ 2) = 1 := by 
  sorry

end AE_AF_identity_l365_365404


namespace square_diagonal_length_l365_365059
-- Import the complete Mathlib library

-- Problem conditions translated to Lean
def area_of_square (s : ℝ) : ℝ := s^2
def diagonal_of_square (s : ℝ) : ℝ := s * Real.sqrt 2

-- Define the specific conditions given in the problem
def square_side (A : ℝ) : ℝ := Real.sqrt A
def specific_square_side := square_side 128
def specific_diagonal := diagonal_of_square specific_square_side

-- The proof goal
theorem square_diagonal_length (A : ℝ) (is_area_128 : A = 128) : specific_diagonal = 16 := by
  -- Proof structure placeholder, 'sorry' to indicate unfinished proof
  sorry

end square_diagonal_length_l365_365059


namespace sum_double_series_l365_365534

theorem sum_double_series :
  (∑ n from 2 to ∞, ∑ k from 1 to (n-1), k / 3^(n+k)) = 9 / 136 :=
by
  sorry

end sum_double_series_l365_365534


namespace infinite_m_eq_1989_l365_365328

-- Define f(m) as the number of factors of 2 in m!
def f (m : ℕ) : ℕ := ∑ k in finset.range (m + 1), m / (2 ^ k)

theorem infinite_m_eq_1989 : ∀ n : ℕ, ∃ m : ℕ, m - f m = 1989 :=
sorry

end infinite_m_eq_1989_l365_365328


namespace S_13_eq_3136_l365_365322

-- Define the set of sequences following the condition
def S : ℕ → ℕ
| 0     := 0    -- Base case for n = 0
| 1     := 2
| 2     := 4
| 3     := 7
| (n+4) := S (n+3) + S (n+2) + S (n+1)

theorem S_13_eq_3136 : S 13 = 3136 :=
by simp [S]; 
sorry -- Proof skipped for framework setup

end S_13_eq_3136_l365_365322


namespace volume_of_cube_is_correct_l365_365089

-- Define necessary constants and conditions
def cost_in_paise : ℕ := 34398
def rate_per_sq_cm : ℕ := 13
def surface_area : ℕ := cost_in_paise / rate_per_sq_cm
def face_area : ℕ := surface_area / 6
def side_length : ℕ := Nat.sqrt face_area
def volume : ℕ := side_length ^ 3

-- Prove the volume of the cube
theorem volume_of_cube_is_correct : volume = 9261 := by
  -- Using given conditions and basic arithmetic 
  sorry

end volume_of_cube_is_correct_l365_365089


namespace usual_time_72_l365_365458

namespace TypicalTimeProof

variables (S T : ℝ) 

theorem usual_time_72 (h : T ≠ 0) (h2 : 0.75 * S ≠ 0) (h3 : 4 * T = 3 * (T + 24)) : T = 72 := by
  sorry

end TypicalTimeProof

end usual_time_72_l365_365458


namespace positive_integers_dividing_10n_l365_365983

-- Define the sum of the first n positive integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the predicate that checks if the sum of the first n positive integers evenly divides 10n
def evenly_divides_10n (n : ℕ) : Prop :=
  10 * n % sum_first_n n = 0

-- Define the proof statement that there are exactly 5 positive integers n where the sum evenly divides 10n
theorem positive_integers_dividing_10n : (finset.range 20).filter (λ n, n > 0 ∧ evenly_divides_10n n).card = 5 :=
by
  sorry

end positive_integers_dividing_10n_l365_365983


namespace sqrt_meaningful_iff_l365_365032

theorem sqrt_meaningful_iff (x : ℝ) : (∃ y, y = sqrt(x + 1)) ↔ x ≥ -1 :=
by
  sorry

end sqrt_meaningful_iff_l365_365032


namespace necessary_but_not_sufficient_l365_365336

-- Define sets M and N
def M (x : ℝ) : Prop := x < 5
def N (x : ℝ) : Prop := x > 3

-- Define the union and intersection of M and N
def M_union_N (x : ℝ) : Prop := M x ∨ N x
def M_inter_N (x : ℝ) : Prop := M x ∧ N x

-- Theorem statement: Prove the necessity but not sufficiency
theorem necessary_but_not_sufficient (x : ℝ) :
  M_inter_N x → M_union_N x ∧ ¬(M_union_N x → M_inter_N x) := 
sorry

end necessary_but_not_sufficient_l365_365336


namespace next_working_day_together_l365_365661

theorem next_working_day_together : 
  let greta_days := 5
  let henry_days := 3
  let linda_days := 9
  let sam_days := 8
  ∃ n : ℕ, n = Nat.lcm (Nat.lcm (Nat.lcm greta_days henry_days) linda_days) sam_days ∧ n = 360 :=
by
  sorry

end next_working_day_together_l365_365661


namespace find_angle_ABC_cosine_l365_365281

-- Definitions based on conditions
variables (A B C D P Q : Type)
variables [affine_space A] [metric_space A] [affine_geometry P Q A]
variables {S : ℝ} (ABCD : P ∈ A → B ∈ A → C ∈ A → D ∈ A)
variables (AC_perp_AB : ∀ A B C : P, (AC ∥ AB) → angle B AC AB = π / 2)
variables (circle_touch_BC_P : circle P ↔ touching_line P BC)
variables (circle_touch_lineAB_A : circle A ↔ touching_line A AB)
variables (PQ_perp_AB : ∃ Q : P, P ∈ AB ∧ angle P Q AB = π / 2)
variables (area_ABCD : area A B C D = 1/2)
variables (area_QPCDA : area Q P C D A = S)

-- Problem statement
theorem find_angle_ABC_cosine :
  cos (angle A B C) = sqrt (2 - 4 * S) :=
sorry

end find_angle_ABC_cosine_l365_365281


namespace solution_to_equation_l365_365944

theorem solution_to_equation :
  (∃ x : ℝ, sqrt (3 * x - 2) + 9 / sqrt (3 * x - 2) = 6) ↔
  (∃ x : ℝ, x = 11 / 3) := by
  sorry

end solution_to_equation_l365_365944


namespace problem_statement_l365_365570

noncomputable def i : ℂ := Complex.I

theorem problem_statement : i^10 + i^20 + i^(-30) = -1 :=
by
  have h1 : i^2 = -1 := by norm_num [i]
  have h2 : i^4 = 1 := by norm_num [i]
  sorry

end problem_statement_l365_365570


namespace probability_Alex_Mel_Chelsea_l365_365275

variables (game_rounds : ℕ) 
          (prob_Alex : ℚ)
          (prob_Mel : ℚ)
          (prob_Chelsea : ℚ)
          (A_wins : ℕ)
          (M_wins : ℕ)
          (C_wins : ℕ)

def condition_probability_sum : Prop :=
  prob_Alex + prob_Mel + prob_Chelsea = 1

def condition_Mel_twice_Chelsea : Prop :=
  prob_Mel = 2 * prob_Chelsea

theorem probability_Alex_Mel_Chelsea :
  game_rounds = 8 →
  prob_Alex = 1 / 4 →
  condition_Mel_twice_Chelsea prob_Mel prob_Chelsea →
  condition_probability_sum prob_Alex prob_Mel prob_Chelsea →
  A_wins = 2 →
  M_wins = 3 →
  C_wins = 3 →
  (nat.choose 8 2) * (nat.choose (8 - 2) 3) * (nat.choose (8 - 2 - 3) 3) * (prob_Alex ^ 2) * (prob_Mel ^ 3) * (prob_Chelsea ^ 3) = 35 / 512 :=
sorry

end probability_Alex_Mel_Chelsea_l365_365275


namespace y_coordinate_of_P_l365_365319

theorem y_coordinate_of_P :
  let A := (-4, 0)
  let B := (-3, 2)
  let C := (3, 2)
  let D := (4, 0)
  ∃ P : ℝ × ℝ, (dist P A + dist P D = 10 ∧ dist P B + dist P C = 10) ∧ P.snd = 6 / 7 :=
begin
  sorry
end

end y_coordinate_of_P_l365_365319


namespace find_value_in_table_l365_365438

theorem find_value_in_table :
  let W := 'W'
  let L := 'L'
  let Q := 'Q'
  let table := [
    [W, '?', Q],
    [L, Q, W],
    [Q, W, L]
  ]
  table[0][1] = L :=
by
  sorry

end find_value_in_table_l365_365438


namespace clothing_weight_removed_l365_365033

/-- 
In a suitcase, the initial ratio of books to clothes to electronics, by weight measured in pounds, 
is 7:4:3. The electronics weight 9 pounds. Someone removes some pounds of clothing, doubling the ratio of books to clothes. 
This theorem verifies the weight of clothing removed is 1.5 pounds.
-/
theorem clothing_weight_removed 
  (B C E : ℕ) 
  (initial_ratio : B / 7 = C / 4 ∧ C / 4 = E / 3)
  (E_val : E = 9)
  (new_ratio : ∃ x : ℝ, B / (C - x) = 2) : 
  ∃ x : ℝ, x = 1.5 := 
sorry

end clothing_weight_removed_l365_365033


namespace simplify_expression_l365_365522

theorem simplify_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + 2 * Real.sqrt(x * y) + y) / (Real.sqrt x + Real.sqrt y) - (Real.sqrt(x * y) + Real.sqrt x) * Real.sqrt(1 / x) = Real.sqrt x - 1 := 
by
  -- proof goes here
  sorry

end simplify_expression_l365_365522


namespace correct_option_is_B_l365_365864

noncomputable def correct_calculation (x : ℝ) : Prop :=
  (x ≠ 1) → (x ≠ 0) → (x ≠ -1) → (-2 / (2 * x - 2) = 1 / (1 - x))

theorem correct_option_is_B (x : ℝ) : correct_calculation x := by
  intros hx1 hx2 hx3
  sorry

end correct_option_is_B_l365_365864


namespace tangent_to_conic_l365_365179

variables {a b c d e f x y x₀ y₀ : ℝ}

def on_conic (x₀ y₀ : ℝ) : Prop := 
  a * x₀^2 + b * x₀ * y₀ + c * y₀^2 + d * x₀ + e * y₀ + f = 0

def tangent_line_eq (x y x₀ y₀ : ℝ) : ℝ :=
  (2 * a * x₀ + b * y₀ + d) * x + (b * x₀ + 2 * c * y₀ + e) * y + d * x₀ + e * y₀ + 2 * f

theorem tangent_to_conic {x₀ y₀ : ℝ} 
  (h : on_conic x₀ y₀) : 
  tangent_line_eq x y x₀ y₀ = 
    (2 * a * x₀ + b * y₀ + d) * x + (b * x₀ + 2 * c * y₀ + e) * y + d * x₀ + e * y₀ + 2 * f :=
sorry

end tangent_to_conic_l365_365179


namespace rhombus_perimeter_is_52_l365_365796

-- Define the problem setup with the measurements of the diagonals.
def diagonals : ℝ × ℝ := (24, 10)

-- Define a function to compute the perimeter of a rhombus given its diagonals.
noncomputable def rhombus_perimeter (d1 d2 : ℝ) : ℝ :=
  let side := Math.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side

-- The theorem we want to prove.
theorem rhombus_perimeter_is_52 : rhombus_perimeter 24 10 = 52 := by
  sorry

end rhombus_perimeter_is_52_l365_365796


namespace percentage_decrease_visitors_l365_365690

-- Definitions based on conditions
def original_ticket_price (P : ℝ) : ℝ := P
def original_number_of_visitors (V : ℝ) : ℝ := V
def original_revenue (R : ℝ) (P : ℝ) (V : ℝ) : ℝ := P * V

def new_ticket_price (P : ℝ) : ℝ := 1.5 * P
def new_revenue (R : ℝ) : ℝ := 1.2 * R

def percentage_decrease (V V' : ℝ) : ℝ := 100 * (V - V') / V

-- Problem statement in Lean 4 
theorem percentage_decrease_visitors (P V R : ℝ) (hR : R = original_revenue R P V)
    (h1 : new_revenue R = 1.2 * original_revenue R P V)
    (h2 : new_ticket_price P = 1.5 * P) : 
    percentage_decrease V (0.8 * V) = 20 := 
by
  sorry

end percentage_decrease_visitors_l365_365690


namespace min_value_quadratic_l365_365441

theorem min_value_quadratic (x : ℝ) : ∃ x, x = -7 ∧ (x^2 + 14 * x + 24 = -25) := sorry

end min_value_quadratic_l365_365441


namespace ball_removed_solution_l365_365410

theorem ball_removed_solution (n m : ℕ) 
  (h1 : ∑ i in finset.range (n + 1), i = n * (n + 1) / 2) 
  (h2 : ∑ i in finset.range (n + 1) \ {m} = 5048) : 
  m = 2 := 
sorry

end ball_removed_solution_l365_365410


namespace fourth_number_is_two_l365_365055

   def sequence_of_medians : List ℚ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 5]

   def first_number : ℤ := 1

   theorem fourth_number_is_two :
     ∃ numbers : List ℤ, numbers.length ≥ 4 ∧ numbers.head = first_number ∧ ∀ n ≤ 3, (numbers.take (n+1)).med = sequence_of_medians.nth! n → numbers.nth 3 = 2 := sorry
   
end fourth_number_is_two_l365_365055


namespace volume_of_A_is_2800_l365_365483

-- Define the dimensions of the fishbowl and water heights
def fishbowl_side_length : ℝ := 20
def height_with_A : ℝ := 16
def height_without_A : ℝ := 9

-- Compute the volume of water with and without object (A)
def volume_with_A : ℝ := fishbowl_side_length ^ 2 * height_with_A
def volume_without_A : ℝ := fishbowl_side_length ^ 2 * height_without_A

-- The volume of object (A)
def volume_A : ℝ := volume_with_A - volume_without_A

-- Prove that this volume is 2800 cubic centimeters
theorem volume_of_A_is_2800 :
  volume_A = 2800 := by
  sorry

end volume_of_A_is_2800_l365_365483


namespace green_to_purple_ratio_l365_365786

def number_of_blue_horses : ℕ := 3
def number_of_purple_horses : ℕ := 3 * number_of_blue_horses
def number_of_green_horses : ℕ
def number_of_gold_horses := number_of_green_horses / 6

axiom total_horses : number_of_blue_horses + number_of_purple_horses + number_of_green_horses + number_of_gold_horses = 33

theorem green_to_purple_ratio : number_of_green_horses / number_of_purple_horses = 2 :=
by
  -- Proof steps will go here, but for now, we put
  sorry

end green_to_purple_ratio_l365_365786


namespace polynomial_roots_sum_reciprocal_l365_365551

open Polynomial

theorem polynomial_roots_sum_reciprocal (a b c : ℝ) (h : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1) :
    (40 * a^3 - 70 * a^2 + 32 * a - 3 = 0) ∧
    (40 * b^3 - 70 * b^2 + 32 * b - 3 = 0) ∧
    (40 * c^3 - 70 * c^2 + 32 * c - 3 = 0) →
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) = 3 :=
by
  sorry

end polynomial_roots_sum_reciprocal_l365_365551


namespace greatest_integer_difference_l365_365669

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x) (hx2 : x < 6) (hy : 6 < y) (hy2 : y < 10) :
  ∃ d : ℤ, d = y - x ∧ d = 5 :=
by
  sorry

end greatest_integer_difference_l365_365669


namespace max_ab_l365_365654

theorem max_ab {a b : ℝ} (h : a + b = 3) : ab ≤ (9 / 4) :=
begin
  have amgm : (a * b) ≤ ((a + b) / 2) ^ 2,
  { exact real.geom_mean_le_iff_le_arith_mean_sq.mpr (abs_nonneg (a + b)).le },
  rw [h, div_pow],
  norm_num,
  exact amgm,
end

end max_ab_l365_365654


namespace system_solution_l365_365376
-- importing the Mathlib library

-- define the problem with necessary conditions
theorem system_solution (x y : ℝ → ℝ) (x0 y0 : ℝ) 
    (h1 : ∀ t, deriv x t = y t) 
    (h2 : ∀ t, deriv y t = -x t) 
    (h3 : x 0 = x0)
    (h4 : y 0 = y0):
    (∀ t, x t = x0 * Real.cos t + y0 * Real.sin t) ∧ (∀ t, y t = -x0 * Real.sin t + y0 * Real.cos t) ∧ (∀ t, (x t)^2 + (y t)^2 = x0^2 + y0^2) := 
by 
    sorry

end system_solution_l365_365376


namespace sheena_sewing_hours_l365_365768

theorem sheena_sewing_hours
  (h : ℕ)
  (hours_per_week : ℕ)
  (weeks : ℕ)
  (bridesmaids : ℕ)
  (total_hours : ℕ)
  (hours_per_week = 4)
  (weeks = 15)
  (bridesmaids = 5)
  (total_hours = hours_per_week * weeks)
  (total_hours = bridesmaids * h) :
  h = 12 :=
by
  sorry

end sheena_sewing_hours_l365_365768


namespace proof_problem_l365_365613

variable (a b : ℝ) -- Semi-major and semi-minor axes
variable (a_gt_b : a > b)
variable (b_pos : b > 0)

--- Ellipse with given foci
def ellipse1 : Prop := ∀ x y : ℝ, (x^2 / 5) + (y^2 / 2) = 1

-- Ellipse C's equation
def ellipseC : Prop := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

-- Same foci condition
def same_foci (a b : ℝ) : Prop := 
  let c := Real.sqrt (a^2 - b^2) in 
  (c * c = a^2 - b^2)

-- Chord length condition
def chord_length (a b : ℝ) : Prop := 
  let c := Real.sqrt (a^2 - b^2) in 
  ∃ x : ℝ, x = (a * a) / b ^ 2 = 1

-- Equation of ellipse C
def equation_ellipseC (x y : ℝ) : Prop := 
  x^2 / 4 + y^2 = 1

-- Line equation and intersection points
def line_l (m : ℝ) : Prop := ∀ x y : ℝ, y = x + m

def intersects (a b m : ℝ) : Prop := 
  ∃ x1 x2 : ℝ, 
  let y1 y2 := x1 + m, x2 + m in
  line_l m ∧ (x1 + x2) = -8 * m / 5 ∧ (x1 * x2) = (4 * m ^ 2 - 4) / 5 ∧ 
  (Real.sqrt (1 + m * m) * Real.sqrt ((x1 + x2)^2 - 4 * x1 * x2))/5 = 8/5

theorem proof_problem :
  same_foci a b → chord_length a b →
  (∃ m : ℝ, intersects 4 1 m) → 
  ∃ m, m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := 
sorry

end proof_problem_l365_365613


namespace find_angle_y_l365_365289

noncomputable def angle_ABC := 70
noncomputable def angle_BAC := 50
noncomputable def angle_CDE := 30

theorem find_angle_y (h1 : angle_ABC = 70) (h2 : angle_BAC = 50) (h3 : angle_CDE = 30) : 
  ∠ CDE = 30 := by sorry

end find_angle_y_l365_365289


namespace ratio_volumes_eq_l365_365436

def volume_cube (s : ℝ) := s ^ 3
def volume_sphere (r : ℝ) := (4 / 3) * Real.pi * r ^ 3

theorem ratio_volumes_eq :
  let s := 8
  let d := 12
  let r := d / 2
  volume_cube s / volume_sphere r = 16 / (9 * Real.pi) := by
  sorry

end ratio_volumes_eq_l365_365436


namespace negate_condition_l365_365612

theorem negate_condition {x : ℝ}
  (p : | x + 1 | ≤ 4)
  (q : 2 < x ∧ x < 3) :
  (¬(2 < x ∧ x < 3) → ¬(| x + 1 | ≤ 4)) ∧ ¬(¬(| x + 1 | ≤ 4) → ¬(2 < x ∧ x < 3)) :=
by {
  sorry
}

end negate_condition_l365_365612


namespace manager_salary_3700_l365_365088

theorem manager_salary_3700
  (salary_20_employees_avg : ℕ)
  (salary_increase : ℕ)
  (total_employees : ℕ)
  (manager_salary : ℕ)
  (h_avg : salary_20_employees_avg = 1600)
  (h_increase : salary_increase = 100)
  (h_total_employees : total_employees = 20)
  (h_manager_salary : manager_salary = 21 * (salary_20_employees_avg + salary_increase) - 20 * salary_20_employees_avg) :
  manager_salary = 3700 :=
by
  sorry

end manager_salary_3700_l365_365088


namespace find_scalars_l365_365406

-- defining vectors
def a : ℝ^3 := ⟨1, 2, 2⟩
def b : ℝ^3 := ⟨0, 3, -4⟩
def c : ℝ^3 := ⟨0, 5, 0⟩
def v : ℝ^3 := ⟨6, 10, -8⟩

-- proving orthogonality
def orthogonal (u v : ℝ^3) : Prop := 
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0

-- main theorem statement
theorem find_scalars : ∃ x y z : ℝ, v = x • a + y • b + z • c ∧ 
                        orthogonal a b ∧ orthogonal a c ∧ orthogonal b c :=
  sorry

end find_scalars_l365_365406


namespace equal_circumradii_l365_365360

-- Define the points and triangles involved
variable (A B C M : Type*) 

-- The circumcircle radius of a triangle is at least R
variable (R R1 R2 R3 : ℝ)

-- Hypotheses: the given conditions
variable (hR1 : R1 ≥ R)
variable (hR2 : R2 ≥ R)
variable (hR3 : R3 ≥ R)

-- The goal: to show that all four radii are equal
theorem equal_circumradii {A B C M : Type*} (R R1 R2 R3 : ℝ) 
    (hR1 : R1 ≥ R) 
    (hR2 : R2 ≥ R) 
    (hR3 : R3 ≥ R): 
    R1 = R ∧ R2 = R ∧ R3 = R := 
by 
  sorry

end equal_circumradii_l365_365360


namespace percentage_of_singles_is_correct_l365_365126

-- Define the problem conditions
def total_hits : ℕ := 45
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 7

-- Define the target percentage
def target_percentage : ℝ := 73.33

-- Formulate the proof problem in Lean 4
theorem percentage_of_singles_is_correct :
  let non_single_hits := home_runs + triples + doubles in
  let singles := total_hits - non_single_hits in
  (singles:ℝ) / (total_hits:ℝ) * 100 = target_percentage :=
by
  sorry  -- Proof is omitted

end percentage_of_singles_is_correct_l365_365126


namespace max_species_in_110x110_array_l365_365823

def max_distinct_species_in_row_or_col : ℕ :=
  110

theorem max_species_in_110x110_array :
  ∃ n : ℕ, n = 11 ∧
    ∀ (arr : fin 110 → fin 110 → fin 110),
      (∃ r : fin 110, fintype.card (finset.univ.image (λ c, arr r c)) ≥ n) ∨
      (∃ c : fin 110, fintype.card (finset.univ.image (λ r, arr r c)) ≥ n) :=
by { use 11, sorry }

end max_species_in_110x110_array_l365_365823


namespace proper_subset_count_l365_365815

theorem proper_subset_count (S : Set (Fin 5)) : (card {M | M ⊂ S}) = 31 := by
  sorry

end proper_subset_count_l365_365815


namespace polynomial_inequality_l365_365315

theorem polynomial_inequality (b c : ℝ) (x₁ x₂ x₃ : ℝ) 
  (hx₁ : 0 < x₁ ∧ x₁ < 1)
  (hx₂ : 0 < x₂ ∧ x₂ < 1)
  (hx₃ : 0 < x₃ ∧ x₃ < 1)
  (h_roots : x₁ + x₂ + x₃ = 2 ∧ x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = b ∧ x₁ * x₂ * x₃ = -c) :
  8 * b + 9 * c ≤ 8 :=
begin
  sorry,
end

end polynomial_inequality_l365_365315


namespace possible_values_of_alpha_l365_365977

noncomputable def α := ℝ
def a_seq (α : α) : ℕ → ℕ := λ n, ⌊ α^n ⌋

theorem possible_values_of_alpha (α : ℝ) (hα : α > 1) :
  ( ∀ m : ℕ, ∃ k : ℕ, ∃ i : Fin k → ℕ, StrictMono i ∧ m = ∑ j, a_seq α (i j) ) ↔ (α ∈ Set.Ioo 1 (Real.root 4 13) ∪ {2} : Set ℝ) :=
sorry

end possible_values_of_alpha_l365_365977


namespace dispatch_plans_count_l365_365121

def num_dispatch_plans (teachers : Finset ℕ) (A B C : ℕ) : ℕ :=
  if H : ({A, B} ⊆ teachers) 
  then 0 
  else if H' : ({A} ∈ teachers ↔ {C} ∈ teachers)
  then Nat.choose (teachers.erase A).erase B).card 3 * fact 4 
  else 0

theorem dispatch_plans_count :
  let teachers := Finset.range 8
  let A := 0
  let B := 1
  let C := 2
  num_dispatch_plans teachers A B C = 600 := by
  sorry

end dispatch_plans_count_l365_365121


namespace number_of_distinct_real_roots_l365_365199

noncomputable def f (x : ℝ) (a b c : ℝ) := x^3 + a * x^2 + b * x + c

def f' (x a b : ℝ) := 3 * x^2 + 2 * a * x + b

theorem number_of_distinct_real_roots 
(a b c x1 x2 : ℝ) 
(h1 : x1 < x2)
(h2 : f' x1 a b = 0)
(h3 : f' x2 a b = 0): 
  ∃ n, n = 3 ∧ 
    ∀ x, 3 * (f x a b c)^2 + 2 * a * (f x a b c) + b = 0 → 
    count_distinct_real_roots (3 * (f x a b c)^2 + 2 * a * (f x a b c) + b = 0) n :=
sorry

end number_of_distinct_real_roots_l365_365199


namespace compute_H_five_times_l365_365896

-- Define the function H based on observed points from the graph
def H : ℤ → ℤ
| 2  := -2
| (-2) := 3
| 3  := 3
| _  := 0 -- default case, to ensure totality of function

-- The theorem to prove
theorem compute_H_five_times : H(H(H(H(H(2))))) = 3 :=
by
  -- placeholder for the actual proof
  sorry

end compute_H_five_times_l365_365896


namespace shifted_sine_symmetry_l365_365261

theorem shifted_sine_symmetry :
  ∀ x : ℝ, sin (2 * (x + π / 6)) = sin (2 * ((π / 12) - (x - π / 12))) :=
by sorry

end shifted_sine_symmetry_l365_365261


namespace P_eval_sum_l365_365321

noncomputable def P (x : ℕ) : ℕ := 0 -- Define P as a placeholder

variable (k : ℕ)

-- Define conditions
axiom P_at_0 : P 0 = k
axiom P_at_1 : P 1 = 2 * k
axiom P_at_neg1 : P (-1) = 3 * k

-- The theorem statement
theorem P_eval_sum (P : ℤ → ℤ) (k : ℤ) (h0 : P 0 = k) (h1 : P 1 = 2 * k) (h_neg1 : P (-1) = 3 * k) : P 2 + P (-2) = 14 * k :=
sorry 

end P_eval_sum_l365_365321


namespace triangle_orthocenter_tan_proof_l365_365701

noncomputable def triangle_ABC_orthocenter_tanA_tanB (HF HC tan_A tan_B : ℝ) : Prop :=
  let CF := HF + HC in
  HF = 4 ∧ HC = 12 ∧ CF = 16 → tan_A * tan_B = 4

theorem triangle_orthocenter_tan_proof :
  ∀ (ABC : Type) [triangle ABC] (HF HC : ℝ) (tan_A tan_B : ℝ),
  triangle_ABC_orthocenter_tanA_tanB HF HC tan_A tan_B :=
by 
  intros ABC _ HF HC tan_A tan_B h,
  cases' h with h1,
  sorry

end triangle_orthocenter_tan_proof_l365_365701


namespace quadratic_roots_squared_sum_l365_365212

theorem quadratic_roots_squared_sum (m n : ℝ) (h1 : m^2 - 2 * m - 1 = 0) (h2 : n^2 - 2 * n - 1 = 0) : m^2 + n^2 = 6 :=
sorry

end quadratic_roots_squared_sum_l365_365212


namespace sum_double_series_l365_365533

theorem sum_double_series :
  (∑ n from 2 to ∞, ∑ k from 1 to (n-1), k / 3^(n+k)) = 9 / 136 :=
by
  sorry

end sum_double_series_l365_365533


namespace dilution_problem_l365_365029
-- Definitions of the conditions
def volume_initial : ℝ := 15
def concentration_initial : ℝ := 0.60
def concentration_final : ℝ := 0.40
def amount_alcohol_initial : ℝ := volume_initial * concentration_initial

-- Proof problem statement in Lean 4
theorem dilution_problem : 
  ∃ (x : ℝ), x = 7.5 ∧ 
              amount_alcohol_initial = concentration_final * (volume_initial + x) :=
sorry

end dilution_problem_l365_365029


namespace count_valid_three_digit_numbers_l365_365510

def digits := {1, 2, 3, 4, 5}

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def valid_three_digit_numbers : Finset ℕ := 
  (Finset.range 1000).filter (λ n, 
    ∀ d ∈ n.digits, d ∈ digits ∧
    n.digits.sum = 9)

theorem count_valid_three_digit_numbers : valid_three_digit_numbers.card = 19 :=
sorry

end count_valid_three_digit_numbers_l365_365510


namespace triangle_angle_a_value_triangle_side_a_value_l365_365705

open Real

theorem triangle_angle_a_value (a b c A B C : ℝ) 
  (h1 : (a - c) * (a + c) * sin C = c * (b - c) * sin B)
  (h2 : (1/2) * b * c * sin A = sqrt 3)
  (h3 : sin B * sin C = 1/4) :
  A = π / 3 :=
sorry

theorem triangle_side_a_value (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : (a - c) * (a + c) * sin C = c * (b - c) * sin B)
  (h2 : (1/2) * b * c * sin(A) = sqrt 3)
  (h3 : sin B * sin C = 1/4)
  (h4 : A = π / 3) :
  a = 2 * sqrt 3 :=
sorry

end triangle_angle_a_value_triangle_side_a_value_l365_365705


namespace triple_solution_exists_and_unique_l365_365964

theorem triple_solution_exists_and_unique:
  ∀ (x y z : ℝ), (1 + x^4 ≤ 2 * (y - z) ^ 2) ∧ (1 + y^4 ≤ 2 * (z - x) ^ 2) ∧ (1 + z^4 ≤ 2 * (x - y) ^ 2)
  → (x = 1 ∧ y = 0 ∧ z = -1) :=
by
  sorry

end triple_solution_exists_and_unique_l365_365964


namespace smallest_positive_integer_congruence_l365_365437

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 5 * x ≡ 18 [MOD 31] ∧ 0 < x ∧ x < 31 ∧ x = 16 := 
by sorry

end smallest_positive_integer_congruence_l365_365437


namespace sqrt_of_fraction_l365_365142

theorem sqrt_of_fraction (a b : ℕ) (h₁ : a = 4) (h₂ : b = 9) : (Real.sqrt (a / b) = 2 / 3) :=
by
  have h₃ : (a / b : ℝ) = (2 / 3)^2 := by
    rw [h₁, h₂]
    norm_num
  rw Real.sqrt_eq_rpow
  rw h₃
  norm_num
  sorry

end sqrt_of_fraction_l365_365142


namespace compute_fractional_part_l365_365316

noncomputable def floor (x : ℝ) : ℤ := x.to_int
noncomputable def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

-- main problem: prove that the fractional part of 'n' equals to 2014/2015 under the given conditions
theorem compute_fractional_part (n : ℝ) (h : floor(n) / n = 2015 / 2016) :
  fractional_part(n) = 2014 / 2015 := sorry

end compute_fractional_part_l365_365316


namespace smallest_phi_abs_proof_l365_365233

noncomputable def smallest_phi_abs (φ : ℝ) : ℝ :=
  let original_func := λ x : ℝ, 2 * sin (3 * x + φ)
  let shifted_func  := λ x : ℝ, 2 * sin (3 * x - 3 * π / 4 + φ)
  let symmetry_point := (π / 3, 0)
  φ -- Placeholder for the actual value of φ we aim to find

theorem smallest_phi_abs_proof : 
  smallest_phi_abs φ = π / 4 := 
begin
  sorry -- Actual proof would go here
end

end smallest_phi_abs_proof_l365_365233


namespace Zeljko_total_distance_l365_365452

-- Conditions
def speed1 : ℝ := 30  -- speed in km/h
def speed2 : ℝ := 20  -- speed in km/h
def time1 : ℝ := 20 / 60  -- time in hours
def time2 : ℝ := 30 / 60  -- time in hours

-- Question converted into a goal
theorem Zeljko_total_distance :
  speed1 * time1 + speed2 * time2 = 20 :=
by
  sorry

end Zeljko_total_distance_l365_365452


namespace concentric_circles_color_rotation_l365_365838

theorem concentric_circles_color_rotation 
  (k : ℕ)
  (sectors_inner sectors_outer : Fin (2 * k) → Bool)
  (hw : ∃ w, (Finset.univ.filter (λ i, sectors_inner i = tt)).card = w ∧ (Finset.univ.filter (λ i, sectors_outer i = tt)).card = w) :
  ∃ m : ℕ, m < 2 * k ∧ (Finset.range k).card ≤ (Finset.range (2 * k)).filter (λ i, sectors_inner i ≠ sectors_outer ((i + m) % (2 * k))).card :=
begin
  sorry
end

end concentric_circles_color_rotation_l365_365838


namespace ratio_of_votes_l365_365697

theorem ratio_of_votes (randy_votes : ℕ) (shaun_votes : ℕ) (eliot_votes : ℕ)
  (h1 : randy_votes = 16)
  (h2 : shaun_votes = 5 * randy_votes)
  (h3 : eliot_votes = 160) :
  eliot_votes / shaun_votes = 2 :=
by
  sorry

end ratio_of_votes_l365_365697


namespace correct_correlation_coefficient_statement_l365_365077

variable {α : Type} [Field α]

/-- Definition of a correlation coefficient -/
def is_correlation_coefficient (r : α) : Prop :=
  r ≥ -1 ∧ r ≤ 1

/-- The closer |r| is to 1, the stronger the linear correlation between two variables -/
theorem correct_correlation_coefficient_statement (r : α) (hr : is_correlation_coefficient r) : 
  (C : ∀ r, r = r → (|r| → r ≠ r → r = 1)) :=
sorry

end correct_correlation_coefficient_statement_l365_365077


namespace parabola_symmetry_l365_365247

theorem parabola_symmetry :
  ∃ (m : ℝ), (∀ x : ℝ, x ∈ {0, 2} → 5 = 2 * (x - m)^2 + 3) ↔ m = 1 :=
by
  sorry

end parabola_symmetry_l365_365247


namespace expression_value_l365_365562

noncomputable def real_cube_root (x : ℝ) := x ^ (1/3 : ℝ)

theorem expression_value :
  real_cube_root (-8) + (π ^ 0) + (Real.log 4 / Real.log 10 + Real.log 25 / Real.log 10) = 1 :=
by
  -- Using real cube root function defined as x^(1/3), and considering only real part for negative numbers.
  have h1 : real_cube_root (-8) = -2 := by sorry,
  -- Any non-zero number raised to 0 is 1
  have h2 : π ^ 0 = 1 := by sorry,
  -- log_10(4) + log_10(25) = log_10(4*25) = log_10(100) = 2
  have h3 : Real.log 4 / Real.log 10 + Real.log 25 / Real.log 10 = 2 := by sorry,
  calc
    real_cube_root (-8) + (π ^ 0) + (Real.log 4 / Real.log 10 + Real.log 25 / Real.log 10)
        = -2 + 1 + 2 : by rw [h1, h2, h3]
    ... = 1 : by ring

end expression_value_l365_365562


namespace prop_f_symmetric_l365_365227

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

theorem prop_f_symmetric : (∀ x₁ x₂, f(x₁) = -f(x₂) → x₁ = 1 + (1 - x₂)) :=
sorry

end prop_f_symmetric_l365_365227


namespace solve_abc_l365_365329

def f (x a b c : ℤ) : ℤ := x^3 + a*x^2 + b*x + c

theorem solve_abc (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_fa : f a a b c = a^3) (h_fb : f b b a c = b^3) : 
  a = -2 ∧ b = 4 ∧ c = 16 := 
sorry

end solve_abc_l365_365329


namespace Arianna_distance_when_Ethan_finishes_l365_365160

theorem Arianna_distance_when_Ethan_finishes :
  ∀ (total_distance distance_apart : ℕ), 
  total_distance = 1000 → distance_apart = 816 → (total_distance - distance_apart) = 184 :=
by
  intros total_distance distance_apart h_total h_apart
  rw [h_total, h_apart]
  exact rfl

end Arianna_distance_when_Ethan_finishes_l365_365160


namespace range_of_a_l365_365337

noncomputable def f (a x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x ≠ y ∧ f a' (x' x) a = 0 ∧ f a' (y' y) a = 0 ∧ ∀ z, f a' z a ≥ x → z = x ∨ z = y ∧ ∀ w, f a' w a ≤ y → w = x ∨ w = y ) ↔ (a > 6 ∨ a < -3) :=
begin
  sorry
end

end range_of_a_l365_365337


namespace percent_republicans_voting_A_l365_365273

-- Definitions based on conditions
def total_voters : ℝ := sorry

def percent_democrats : ℝ := 0.60
def percent_republicans : ℝ := 0.40
def percent_democrats_voting_A : ℝ := 0.65
def percent_total_voting_A : ℝ := 0.47

-- Question translated as a theorem statement
theorem percent_republicans_voting_A :
  let democrats := percent_democrats * total_voters,
      republicans := percent_republicans * total_voters,
      democrat_votes_A := percent_democrats_voting_A * democrats,
      total_votes_A := percent_total_voting_A * total_voters,
      republican_votes_A := total_votes_A - democrat_votes_A in
  ((republican_votes_A / republicans) * 100) = 20 :=
sorry

end percent_republicans_voting_A_l365_365273


namespace trailing_zeros_50_factorial_l365_365521

def factorial_trailing_zeros (n : Nat) : Nat :=
  n / 5 + n / 25 -- Count the number of trailing zeros given the algorithm used in solution steps

theorem trailing_zeros_50_factorial : factorial_trailing_zeros 50 = 12 :=
by 
  -- Proof goes here
  sorry

end trailing_zeros_50_factorial_l365_365521


namespace quadratic_vertex_property_l365_365628

variable {a b c x0 y0 m n : ℝ}

-- Condition 1: (x0, y0) is a fixed point on the graph of the quadratic function y = ax^2 + bx + c
axiom fixed_point_on_graph : y0 = a * x0^2 + b * x0 + c

-- Condition 2: (m, n) is a moving point on the graph of the quadratic function
axiom moving_point_on_graph : n = a * m^2 + b * m + c

-- Condition 3: For any real number m, a(y0 - n) ≤ 0
axiom inequality_condition : ∀ m : ℝ, a * (y0 - (a * m^2 + b * m + c)) ≤ 0

-- Statement to prove
theorem quadratic_vertex_property : 2 * a * x0 + b = 0 := 
sorry

end quadratic_vertex_property_l365_365628


namespace compare_values_l365_365229

noncomputable def f (x : ℝ) : ℝ := real.logb 0.2 (x^2 - x + 1)

def a : ℝ := real.log 2 3
def b : ℝ := real.log 3 2
def c : ℝ := real.log 3 (real.sqrt 2)

theorem compare_values : f a < f c ∧ f c < f b := 
by
  sorry

end compare_values_l365_365229


namespace quadrilateral_area_is_correct_l365_365176

noncomputable def area_quadrilateral (A B C D : ℝ) (AB BC AD DC : ℝ) (angle_ABC : ℝ) (circle_radius : ℝ) : ℝ :=
  if (AB = 3 * real.sqrt 3 ∧ BC = 3 * real.sqrt 3 ∧ AD = real.sqrt 13 ∧ DC = real.sqrt 13 ∧ angle_ABC = 60 ∧ circle_radius = 2) then
    3 * real.sqrt 3
  else
    0

theorem quadrilateral_area_is_correct :
  let A B C D : ℝ := 0 -- Define the vertices positions arbitrarily as they don't affect the calculable area using given lengths
  in area_quadrilateral A B C D (3 * real.sqrt 3) (3 * real.sqrt 3) (real.sqrt 13) (real.sqrt 13) 60 2 = 3 * real.sqrt 3 :=
sorry

end quadrilateral_area_is_correct_l365_365176


namespace line_graph_displays_trend_l365_365146

-- Define the types of statistical graphs
inductive StatisticalGraph : Type
| barGraph : StatisticalGraph
| lineGraph : StatisticalGraph
| pieChart : StatisticalGraph
| histogram : StatisticalGraph

-- Define the property of displaying trends over time
def displaysTrend (g : StatisticalGraph) : Prop := 
  g = StatisticalGraph.lineGraph

-- Theorem to prove that the type of statistical graph that displays the trend of data is the line graph
theorem line_graph_displays_trend : displaysTrend StatisticalGraph.lineGraph :=
sorry

end line_graph_displays_trend_l365_365146


namespace two_KL_eq_DE_l365_365806

variables {ABC : Type*} [metric_space ABC] -- Assuming the triangle exists in some metric space
variables {A B C D E P K L : ABC}
variables {incircle : set ABC}
variables {touches_BC_at_D touches_AC_at_E : Prop}
variables {on_shorter_arc_DE : Prop}
variables {angle_APE_eq_angle_DPB : Prop}
variables {AP_BP_meet_DE_at_KL : Prop}

-- Assuming necessary properties of the configuration
axiom touches_BC_at_D_def : touches_BC_at_D ↔ is_tangent incircle (line[BC]) D
axiom touches_AC_at_E_def : touches_AC_at_E ↔ is_tangent incircle (line[AC]) E
axiom on_shorter_arc_DE_def : on_shorter_arc_DE ↔ (P ∈ incircle) ∧ (arc_length shorter P D E)
axiom angle_APE_eq_angle_DPB_def : angle_APE_eq_angle_DPB ↔ angle A P E = angle D P B
axiom AP_BP_meet_DE_at_K_def : AP_BP_meet_DE_at_KL ↔ (is_intersection (line[AP]) (line[DE]) K) ∧ (is_intersection (line[BP]) (line[DE]) L)

-- Statement of the proof problem
theorem two_KL_eq_DE (h1 : touches_BC_at_D) (h2 : touches_AC_at_E) (h3 : on_shorter_arc_DE)
  (h4 : angle_APE_eq_angle_DPB) (h5 : AP_BP_meet_DE_at_KL) : 
  2 * distance K L = distance D E := 
sorry

end two_KL_eq_DE_l365_365806


namespace exists_m_n_l365_365090

def valid_triple (x y z m : ℤ) (n : ℤ) : Prop :=
  xyz = x + y + z + m ∧ max (|x|) (max (|y|) (|z|)) ≤ n

def f (m n : ℤ) : ℤ :=
  ∑ x y z, if valid_triple x y z m n then 1 else 0

theorem exists_m_n : ∃ (m n : ℕ), f m n = 2018 := by 
  use 2 
  use 168 
  sorry

end exists_m_n_l365_365090


namespace degree_of_resulting_polynomial_l365_365558

noncomputable def p (x : ℝ) : ℝ := (3*x^4 + 4*x^3 + 2*x - 7)*(3*x^10 - 9*x^7 + 9*x^4 + 30)
noncomputable def q (x : ℝ) : ℝ := (x^2 + 5)^8
noncomputable def r (x : ℝ) : ℝ := p(x) - q(x)

theorem degree_of_resulting_polynomial : polynomial.degree (polynomial.of_real r) = 16 := 
sorry

end degree_of_resulting_polynomial_l365_365558


namespace largest_class_students_l365_365086

theorem largest_class_students (x : ℕ) (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 105) : x = 25 :=
by {
  sorry
}

end largest_class_students_l365_365086


namespace modulo_residue_l365_365849

theorem modulo_residue : (325 + 6.5 * 22 + 9 * 121 + 3^2 * 33) % 11 = 0 := by
  sorry

end modulo_residue_l365_365849


namespace count_real_roots_log_sin_eq_l365_365934

-- Definitions of the functions involved
def log_abs (x : ℝ) : ℝ := Real.log (Real.abs x)
def sin_func (x : ℝ) : ℝ := Real.sin x

-- The theorem stating the problem
theorem count_real_roots_log_sin_eq : 
  (∃ x1 x2 : list ℝ, list.of_fn log_abs = list.of_fn sin_func 
  ∧ list.length x1 + list.length x2 = 10) :=
by
  sorry

end count_real_roots_log_sin_eq_l365_365934


namespace max_value_of_y_l365_365234

theorem max_value_of_y (a x : ℝ) (h1 : 19 < a) (h2 : a < 96) (h3 : a ≤ x) (h4 : x ≤ 96) :
  let y := |x - a| + |x + 19| + |x - a - 96| in y ≤ 211 :=
by 
  sorry

end max_value_of_y_l365_365234


namespace sum_absolute_minus_absolute_sum_leq_l365_365317

theorem sum_absolute_minus_absolute_sum_leq (n : ℕ) (x : ℕ → ℝ)
  (h_n : n ≥ 1)
  (h_abs_diff : ∀ k, 0 < k ∧ k < n → |x k.succ - x k| ≤ 1) :
  (∑ k in Finset.range n, |x k|) - |∑ k in Finset.range n, x k| ≤ (n^2 - 1) / 4 := 
sorry

end sum_absolute_minus_absolute_sum_leq_l365_365317


namespace range_of_values_for_a_l365_365638

theorem range_of_values_for_a (f : ℝ → ℝ) (a : ℝ) (h_monotone : ∀ x y, 0 < x ∧ x < 1 → 0 < y ∧ y < 1 → x < y → f x ≥ f y) :
  a ∈ set.Ici 2 :=
by
  sorry

end range_of_values_for_a_l365_365638


namespace a_salary_is_3000_l365_365082

theorem a_salary_is_3000 {a b : ℝ} 
  (h1 : a + b = 4000)
  (h2 : 0.05 * a = 0.15 * b) :
  a = 3000 := 
begin
  sorry,
end

end a_salary_is_3000_l365_365082


namespace solve_system_eq_l365_365005

theorem solve_system_eq (x y z : ℝ) :
  (x * y * z / (x + y) = 6 / 5) ∧
  (x * y * z / (y + z) = 2) ∧
  (x * y * z / (z + x) = 3 / 2) ↔
  ((x = 3 ∧ y = 2 ∧ z = 1) ∨ (x = -3 ∧ y = -2 ∧ z = -1)) := 
by
  -- proof to be provided
  sorry

end solve_system_eq_l365_365005


namespace perimeter_of_rhombus_l365_365788

def rhombus_perimeter (d1 d2 : ℝ) : ℝ :=
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2))

theorem perimeter_of_rhombus (h1 : ∀ (d1 d2 : ℝ), d1 = 24 ∧ d2 = 10) : rhombus_perimeter 24 10 = 52 := 
by
  sorry

end perimeter_of_rhombus_l365_365788


namespace equal_angles_not_necessarily_vertical_l365_365817

-- Define what it means for angles to be vertical
def is_vertical_angle (a b : ℝ) : Prop :=
∃ l1 l2 : ℝ, a = 180 - b ∧ (l1 + l2 == 180 ∨ l1 == 0 ∨ l2 == 0)

-- Define what it means for angles to be equal
def are_equal_angles (a b : ℝ) : Prop := a = b

-- Proposition to be proved
theorem equal_angles_not_necessarily_vertical (a b : ℝ) (h : are_equal_angles a b) : ¬ is_vertical_angle a b :=
by
  sorry

end equal_angles_not_necessarily_vertical_l365_365817


namespace felix_chopped_at_least_91_trees_l365_365958

def cost_to_sharpen := 5
def total_spent := 35
def trees_per_sharpen := 13

theorem felix_chopped_at_least_91_trees :
  (total_spent / cost_to_sharpen) * trees_per_sharpen = 91 := by
  sorry

end felix_chopped_at_least_91_trees_l365_365958


namespace corner_cells_different_colors_l365_365566

theorem corner_cells_different_colors 
  (colors : Fin 4 → Prop)
  (painted : (Fin 100 × Fin 100) → Fin 4)
  (h : ∀ (i j : Fin 99), 
    ∃ f g h k, 
      f ≠ g ∧ f ≠ h ∧ f ≠ k ∧
      g ≠ h ∧ g ≠ k ∧ 
      h ≠ k ∧ 
      painted (i, j) = f ∧ 
      painted (i.succ, j) = g ∧ 
      painted (i, j.succ) = h ∧ 
      painted (i.succ, j.succ) = k) :
  painted (0, 0) ≠ painted (99, 0) ∧
  painted (0, 0) ≠ painted (0, 99) ∧
  painted (0, 0) ≠ painted (99, 99) ∧
  painted (99, 0) ≠ painted (0, 99) ∧
  painted (99, 0) ≠ painted (99, 99) ∧
  painted (0, 99) ≠ painted (99, 99) :=
  sorry

end corner_cells_different_colors_l365_365566


namespace smallest_number_l365_365509

theorem smallest_number : Min 2 (-2.5) 0 (-3) = -3 := 
sorry

end smallest_number_l365_365509


namespace sum_of_reciprocals_of_distances_l365_365026

-- Define points M (Town Hall), C (Catholic cathedral), P (Protestant cathedral), and E (School)
variables (M C P E : Point)

-- Define streets
variables (MS GDS CE : Line)

-- Conditions
axiom main_street_is_perpendicular : MS ⊥ GDS
axiom MS_intersects_GDS_at_M : M ∈ MS ∧ M ∈ GDS
axiom C_on_MS : C ∈ MS
axiom P_on_GDS : P ∈ GDS
axiom E_on_line_between_C_and_P : E ∈ Line.through C P
axiom E_distance_to_MS_GDS : (distance E MS = 500) ∧ (distance E GDS = 500)
axiom streets_are_straight : straight_line MS ∧ straight_line GDS ∧ straight_line (Line.through C P)

-- Define distances
variables (d_MC d_MP : ℝ)

axiom MC : distance M C = d_MC
axiom MP : distance M P = d_MP

-- Goal
theorem sum_of_reciprocals_of_distances
  (main_street_is_perpendicular : MS ⊥ GDS)
  (MS_intersects_GDS_at_M : M ∈ MS ∧ M ∈ GDS)
  (C_on_MS : C ∈ MS)
  (P_on_GDS : P ∈ GDS)
  (E_on_line_between_C_and_P : E ∈ Line.through C P)
  (E_distance_to_MS_GDS : (distance E MS = 500) ∧ (distance E GDS = 500))
  (streets_are_straight : straight_line MS ∧ straight_line GDS ∧ straight_line (Line.through C P))
  (MC : distance M C = d_MC)
  (MP : distance M P = d_MP) :
  (1 / d_MC + 1 / d_MP = 0.002) := 
begin
  sorry
end

end sum_of_reciprocals_of_distances_l365_365026


namespace unique_solutions_xy_l365_365960

theorem unique_solutions_xy (x y : ℝ) : 
  x^3 + y^3 = 1 ∧ x^4 + y^4 = 1 ↔ (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0) :=
by sorry

end unique_solutions_xy_l365_365960


namespace simplify_radicals_l365_365370

theorem simplify_radicals (y : ℝ) : 
  real.sqrt (50 * y) * real.sqrt (18 * y) * real.sqrt (32 * y) = 30 * y * real.sqrt (2 * y) :=
by sorry

end simplify_radicals_l365_365370


namespace capsule_cost_difference_l365_365871

theorem capsule_cost_difference :
  let cost_per_capsule_r := 6.25 / 250
  let cost_per_capsule_t := 3.00 / 100
  cost_per_capsule_t - cost_per_capsule_r = 0.005 := by
  sorry

end capsule_cost_difference_l365_365871


namespace pascal_theorem_l365_365763

def Point : Type := sorry
def Line : Type := sorry
def Conic : Type := sorry

def inscribed_hexagon (A B C D E F : Point) (conic : Conic) : Prop := sorry
def intersection (l1 l2 : Line) : Point := sorry
def collinear (p1 p2 p3 : Point) : Prop := sorry
def line_through (p1 p2 : Point) : Line := sorry
def line (p1 p2 : Point) : Line := sorry

theorem pascal_theorem (A B C D E F : Point) (conic : Conic)
  (h_inscribed : inscribed_hexagon A B C D E F conic) :
  let G := intersection (line A B) (line D E),
      H := intersection (line B C) (line E F),
      K := intersection (line C D) (line F A)
  in collinear G H K :=
sorry

end pascal_theorem_l365_365763


namespace polar_equation_of_curve_range_of_OM_ON_l365_365284

/-- Given the parametric equations of a curve C in the Cartesian coordinate system -/
variable (t : ℝ)
def x (t : ℝ) := 1 + cos t
def y (t : ℝ) := Real.sqrt 3 + sin t

/-- The polar equation of the curve C is given by -/
theorem polar_equation_of_curve :
  ∃ θ : ℝ, (x θ)^2 + (y θ)^2 - 2 * (cos θ + Real.sqrt 3 * sin θ) * sqrt ((x θ)^2 + (y θ)^2) + 3 = 0 :=
sorry

/-- The range of values for |OM| + |ON|, given that the ray θ = α intersects C at two distinct points M and N -/
theorem range_of_OM_ON (α : ℝ) :
  (π / 6 < α ∧ α < π / 2) → 2 * Real.sqrt 3 < 4 * sin (α + (π / 6)) ∧ 4 * sin (α + (π / 6)) ≤ 4 :=
sorry

end polar_equation_of_curve_range_of_OM_ON_l365_365284


namespace pq_zero_l365_365725

-- Define universal set U
def U : Set ℕ := {1, 2}

-- Define the set A in terms of a quadratic equation
def A (p q : ℤ) : Set ℤ := {x | x * x + p * x + q = 0}

-- Define the complement of set A with respect to U
def complement_U_A (p q : ℤ) : Set ℕ := U \ (A p q)

-- Given conditions
variable (p q : ℤ)
hypotheses (h1 : complement_U_A p q = {1})

-- Prove p + q = 0
theorem pq_zero (p q : ℤ) (h1 : complement_U_A p q = {1}) : p + q = 0 := 
by 
  sorry

end pq_zero_l365_365725


namespace renovation_services_are_credence_goods_and_choice_arguments_l365_365460

-- Define what credence goods are and the concept of information asymmetry
structure CredenceGood where
  information_asymmetry : Prop
  unobservable_quality  : Prop

-- Define renovation service as an instance of CredenceGood
def RenovationService : CredenceGood := {
  information_asymmetry := true,
  unobservable_quality := true
}

-- Primary conditions for choosing between construction company and private repair crew
structure ChoiceArgument where
  information_availability     : Prop
  warranty_and_accountability  : Prop
  higher_costs                 : Prop
  potential_bias_in_reviews    : Prop

-- Arguments for using construction company
def ConstructionCompanyArguments : ChoiceArgument := {
  information_availability := true,
  warranty_and_accountability := true,
  higher_costs := true,
  potential_bias_in_reviews := true
}

-- Arguments against using construction company
def PrivateRepairCrewArguments : ChoiceArgument := {
  information_availability := false,
  warranty_and_accountability := false,
  higher_costs := true,
  potential_bias_in_reviews := true
}

-- Proof statement to show renovation services are credence goods and economically reasoned arguments for/against
theorem renovation_services_are_credence_goods_and_choice_arguments:
  RenovationService = {
    information_asymmetry := true,
    unobservable_quality := true
  } ∧
  (ConstructionCompanyArguments.information_availability = true ∧
   ConstructionCompanyArguments.warranty_and_accountability = true) ∧
  (ConstructionCompanyArguments.higher_costs = true ∧
   ConstructionCompanyArguments.potential_bias_in_reviews = true) ∧
  (PrivateRepairCrewArguments.higher_costs = true ∧
   PrivateRepairCrewArguments.potential_bias_in_reviews = true) :=
by sorry

end renovation_services_are_credence_goods_and_choice_arguments_l365_365460


namespace distance_point_to_line_l365_365390

theorem distance_point_to_line :
  let x0 := 1
  let y0 := 2
  let A := -1
  let B := 1
  let C := 2
  let distance := (|A * x0 + B * y0 + C| : ℝ) / Real.sqrt (A ^ 2 + B ^ 2)
  distance = 3 * Real.sqrt 2 / 2 :=
by
  let x0 := 1
  let y0 := 2
  let A := -1
  let B := 1
  let C := 2
  have h1 : A * x0 + B * y0 + C = 3 := by rfl
  have h2 : A ^ 2 + B ^ 2 = 2 := by rfl
  have h3 : Real.sqrt 2 * 3 = 3 * Real.sqrt 2 := by sorry
  show distance = _ := by sorry

end distance_point_to_line_l365_365390


namespace positive_integers_dividing_10n_l365_365984

-- Define the sum of the first n positive integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the predicate that checks if the sum of the first n positive integers evenly divides 10n
def evenly_divides_10n (n : ℕ) : Prop :=
  10 * n % sum_first_n n = 0

-- Define the proof statement that there are exactly 5 positive integers n where the sum evenly divides 10n
theorem positive_integers_dividing_10n : (finset.range 20).filter (λ n, n > 0 ∧ evenly_divides_10n n).card = 5 :=
by
  sorry

end positive_integers_dividing_10n_l365_365984


namespace f_increasing_l365_365366

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 6 * x
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 6

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by
  have derivative_positive : ∀ x : ℝ, f' x > 0 := by
    intro x
    calc
      f' x = 3 * (x^2 - 2 * x + 2) : by ring
      _ = 3 * ((x - 1)^2 + 1) : by ring
      _ ≥ 3 * 1 : by apply mul_le_mul_of_nonneg_left; linarith
      _ > 0 : by linarith
  sorry

end f_increasing_l365_365366


namespace max_value_of_k_l365_365626

noncomputable def f (x : ℝ) : ℝ :=
  Real.log x / x

theorem max_value_of_k (k : ℝ) : 
  (∀ x > 0, kx = Real.log x → k = f x) → 
  ∃ x > 0, f(x) = 1 / Real.exp 1 :=
begin
  sorry
end

end max_value_of_k_l365_365626


namespace cube_division_problem_l365_365091

def volume (a : ℕ) : ℕ := a^3
def lateral_surface_area (a : ℕ) : ℕ := 4 * a^2
def edge_length (a : ℕ) : ℕ := 12 * a

theorem cube_division_problem : 
  ∃ (S S' : finset ℕ), 
    S ∪ S' = {1, 2, ..., 16} ∧ 
    S ∩ S' = ∅ ∧ 
    S.card = S'.card ∧
    (S.sum volume = S'.sum volume) ∧ 
    (S.sum lateral_surface_area = S'.sum lateral_surface_area) ∧ 
    (S.sum edge_length = S'.sum edge_length) :=
by
  sorry

end cube_division_problem_l365_365091


namespace equilateral_triangle_properties_l365_365039

-- Definitions
variables {R : Type*} [LinearOrderedField R] 
variables (a b c : R) (n : ℕ)

-- r is the shortest distance and S is the total sum of node values
def shortest_distance (a b c : R) (n : ℕ) : R :=
  if a ≠ b ∧ b ≠ c ∧ a ≠ c then 1
  else if a = b ∨ b = c ∨ a = c then 
    if n % 2 = 0 then R.sqrt 3 / 2
    else 1 / (2 * n) * R.sqrt (3 * n ^ 2 + 1)
  else 0

def total_sum (a b c : R) (n : ℕ) : R :=
  1 / 6 * (n + 1) * (n + 2) * (a + b + c)

-- Lean statement for proof
theorem equilateral_triangle_properties :
  ∀ (a b c : R) (n : ℕ),
    0 < n →
    let r := shortest_distance a b c n in
    let S := total_sum a b c n in
    (r = if a ≠ b ∧ b ≠ c ∧ a ≠ c then 1
         else if a = b ∨ b = c ∨ a = c then 
           if n % 2 = 0 then R.sqrt 3 / 2
           else 1 / (2 * n) * R.sqrt (3 * n ^ 2 + 1)
         else 0) ∧
    (S = 1 / 6 * (n + 1) * (n + 2) * (a + b + c)) :=
by
  intro a b c n h
  let r := shortest_distance a b c n
  let S := total_sum a b c n
  have hr : r = if a ≠ b ∧ b ≠ c ∧ a ≠ c then 1
                 else if a = b ∨ b = c ∨ a = c then 
                   if n % 2 = 0 then R.sqrt 3 / 2
                   else 1 / (2 * n) * R.sqrt (3 * n ^ 2 + 1)
                 else 0 := by sorry
  have hs : S = 1 / 6 * (n + 1) * (n + 2) * (a + b + c) := by sorry
  exact ⟨hr, hs⟩

end equilateral_triangle_properties_l365_365039


namespace fraction_evaluation_l365_365954

theorem fraction_evaluation : (1 - (1 / 4)) / (1 - (1 / 3)) = (9 / 8) :=
by
  sorry

end fraction_evaluation_l365_365954


namespace planted_fraction_l365_365167

noncomputable def isPlantedArea (hypotenuse : ℝ) (triangleArea : ℝ) (plant : ℝ) := plant / triangleArea

theorem planted_fraction : 
  let a := 5
  let b := 12
  let c := (real.sqrt $ ((a:ℝ) ^ 2 + (b:ℝ) ^ 2)) -- Calculation of hypotenuse (c = 13)
  let S := 3 -- The shortest distance from the square to the hypotenuse is 3 units
  let triangleArea := (1/2 : ℝ) * a * b
  let squareArea := (S ^ 2) / 9 -- Square side length derived as (3 * 3)
  let plantedArea := triangleArea - squareArea 
  (isPlantedArea c triangleArea plantedArea = 7/9)
  := sorry

end planted_fraction_l365_365167


namespace cos_compare_l365_365149

theorem cos_compare (θ₁ θ₂ : ℝ) (h₁ : θ₁ = -508) (h₂ : θ₂ = -144)
  (hcos1 : ∀ n : ℤ, cos (θ₁ + 360 * n) = cos θ₁)
  (hcos2 : ∀ θ : ℝ, cos (-θ) = cos θ)
  (hdec : ∀ θ₁ θ₂ : ℝ, 0 ≤ θ₁ ∧ θ₁ ≤ θ₂ ∧ θ₂ ≤ 180 → cos θ₁ ≥ cos θ₂) :
  cos (-508) < cos (-144) := by
  sorry

end cos_compare_l365_365149


namespace sequence_not_riemann_integrable_l365_365365

noncomputable def fn (n : ℕ) (x : ℝ) : ℝ :=
  if x ∈ {q_i : ℚ | (q_i : ℝ) ∈ Icc 0 1 ∧ q_i ≤ n} then 1 else 0

noncomputable def lim_f (x : ℝ) : ℝ :=
  if x ∈ {q_i : ℚ | (q_i : ℝ) ∈ Icc 0 1 } then 1 else 0

theorem sequence_not_riemann_integrable :
  ∃ f_n : ℕ → ℝ → ℝ, 
    (∀ n x, |f_n n x| ≤ 1) ∧
    (∀ x, tendsto (λ n, f_n n x) at_top (nhds (lim_f x))) ∧
    ¬ is_riemann_integrable_on (lim_f) (Icc 0 1) :=
by {
  let f_n := λ n x, fn n x,
  let f := lim_f,
  use f_n,
  split,
  { intros n x, simp [fn], split_ifs; norm_num },
  split,
  { intros x, dsimp, sorry },
  { sorry }
}

end sequence_not_riemann_integrable_l365_365365


namespace find_complex_number_purely_imaginary_l365_365959

theorem find_complex_number_purely_imaginary :
  ∃ z : ℂ, (∃ b : ℝ, b ≠ 0 ∧ z = 1 + b * I) ∧ (∀ a b : ℝ, z = a + b * I → a^2 - b^2 + 3 = 0) :=
by
  -- Proof will go here
  sorry

end find_complex_number_purely_imaginary_l365_365959


namespace parallel_lines_slope_eq_l365_365071

theorem parallel_lines_slope_eq (k : ℚ) :
  (5 = 3 * k) → k = 5 / 3 :=
by
  intros h
  sorry

end parallel_lines_slope_eq_l365_365071


namespace full_price_tickets_revenue_l365_365116

-- Define the conditions and then prove the statement
theorem full_price_tickets_revenue (f d p : ℕ) (h1 : f + d = 200) (h2 : f * p + d * (p / 3) = 3000) : f * p = 1500 := by
  sorry

end full_price_tickets_revenue_l365_365116


namespace circle_properties_l365_365223

-- Define the circle M by its equation
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 1 = 0

-- Prove the radius and symmetry properties of the circle
theorem circle_properties :
  (∀ x y : ℝ, circle_eqn x y ↔ (x - 2)^2 + y^2 = 5) ∧
  (let center := (2, 0) in
   ∀ x y : ℝ, (x + 3 * y - 2 = 0) ↔ (x, y) = center) :=
by
  -- The proof is omitted
  sorry

end circle_properties_l365_365223


namespace monotonic_decreasing_interval_l365_365395

noncomputable def f (x a b : ℝ) : ℝ := x^3 - 3 * a * x + b

theorem monotonic_decreasing_interval 
  (a b : ℝ) (h_a : a > 0) 
  (h_max : f (-1 : ℝ) a b = 6) 
  (h_min : f (1 : ℝ) a b = 2) : 
  ∀ x : ℝ, (-1 < x ∧ x < 1) → (f' x a b < 0) := 
by
  -- Proof goes here
  sorry

end monotonic_decreasing_interval_l365_365395


namespace ratio_of_red_to_white_toys_l365_365042

def toys_in_box (total_toys red_toys_removed remaining_red_toys white_toys : ℕ) : Prop :=
  total_toys = 134 ∧
  red_toys_removed = 2 ∧
  remaining_red_toys = 88 ∧
  (remaining_red_toys + red_toys_removed) + white_toys = total_toys

theorem ratio_of_red_to_white_toys :
  toys_in_box 134 2 88 44 →
  88 / 44 = 2 :=
by
  intro h,
  sorry

end ratio_of_red_to_white_toys_l365_365042


namespace find_AN_in_ABC_l365_365682

theorem find_AN_in_ABC 
  (A B C N : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  [MetricSpace N]
  (AB BC AC : ℝ)
  (h1 : AB = 30)
  (h2 : BC = 30)
  (h3 : AC = 28)
  (N_midpoint : N = midpoint B C) : 
  dist A N = 15 * real.sqrt 3 :=
sorry

end find_AN_in_ABC_l365_365682


namespace solve_for_x_l365_365004

theorem solve_for_x {x : ℝ} (h₁ : 2^(x + 6) = 64^(x - 1)) (h₂ : 64 = 2^6) : x = 2.4 :=
sorry

end solve_for_x_l365_365004


namespace curve_left_of_line_l365_365675

theorem curve_left_of_line (x y : ℝ) : x^3 + 2*y^2 = 8 → x ≤ 2 := 
sorry

end curve_left_of_line_l365_365675


namespace exists_nat_square_starting_with_digits_l365_365770

theorem exists_nat_square_starting_with_digits (S : ℕ) : 
  ∃ (N k : ℕ), S * 10^k ≤ N^2 ∧ N^2 < (S + 1) * 10^k := 
by {
  sorry
}

end exists_nat_square_starting_with_digits_l365_365770


namespace circle_intersections_l365_365197

theorem circle_intersections {P₁ P₂ P₃ : Point} (h₁ : P₁ ≠ P₂) (h₂ : P₃ ∈ segment P₁ P₂) 
  (n : ℕ) (hn : 0 < n) :
  ∃ c : ℝ, c > 0 ∧ ∃ (r₁ r₂ r₃ : ℕ → ℝ),
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → r₁ i ≠ r₂ i ∧ r₁ i ≠ r₃ i ∧ r₂ i ≠ r₃ i) ∧
    (∑ k, (number of points on exactly three circles r₁ k, r₂ k, r₃ k) ≥ c * n^2) := 
sorry

end circle_intersections_l365_365197


namespace triangle_angle_relations_l365_365734

variables {A B C H H_A H_B H_C : Type} -- declaring the variables to be of a general type (e.g., points)
-- declare the angles as real numbers
variables (angle_A angle_B angle_C : ℝ)

-- conditions as assumptions
def is_orthocenter_of (H : Type) (A B C : Type) : Prop := sorry
def is_foot_of_altitude (H_X X A B C : Type) : Prop := sorry
def is_acute_angle (angle : ℝ) : Prop := angle > 0 ∧ angle < π/2

-- statement of the theorem
theorem triangle_angle_relations
(ortho : is_orthocenter_of H A B C)
(foot_A : is_foot_of_altitude H_A A B C)
(foot_B : is_foot_of_altitude H_B B A C)
(foot_C : is_foot_of_altitude H_C C A B)
(acute_A : is_acute_angle angle_A)
(acute_B : is_acute_angle angle_B)
(acute_C : is_acute_angle angle_C)
: 
(angle_A H_B H_C = angle_B) ∧
(angle_B H_C H_A = angle_C) ∧
(angle_C H_A H_B = angle_A) ∧
(angle_H_A H_B H_C = 180 - 2 * angle_B) ∧
(angle_H_B H_C H_A = 180 - 2 * angle_C) ∧
(angle_H_C H_A H_B = 180 - 2 * angle_A) :=
sorry -- Proof not provided

end triangle_angle_relations_l365_365734


namespace first_year_more_rabbits_than_squirrels_l365_365287

noncomputable def squirrel_population : ℕ → ℕ
| 0     := 1
| (k+1) := 2 * squirrel_population k + 2019

noncomputable def rabbit_population : ℕ → ℕ
| 0     := 1
| (k+1) := 4 * rabbit_population k - 2

theorem first_year_more_rabbits_than_squirrels :
  ∃ (k : ℕ), k = 13 ∧ rabbit_population k > squirrel_population k ∧ 
              ∀ (j : ℕ), j < k → rabbit_population j ≤ squirrel_population j :=
by
  sorry

end first_year_more_rabbits_than_squirrels_l365_365287


namespace binary_multiplication_l365_365935

/-- 
Calculate the product of two binary numbers and validate the result.
Given:
  a = 1101 in base 2,
  b = 111 in base 2,
Prove:
  a * b = 1011110 in base 2. 
-/
theorem binary_multiplication : 
  let a := 0b1101
  let b := 0b111
  a * b = 0b1011110 :=
by
  sorry

end binary_multiplication_l365_365935


namespace mary_fruits_left_l365_365346

theorem mary_fruits_left (apples_initial : ℕ) (oranges_initial : ℕ) (blueberries_initial : ℕ)
                         (ate_apples : ℕ) (ate_oranges : ℕ) (ate_blueberries : ℕ) :
  apples_initial = 14 → oranges_initial = 9 → blueberries_initial = 6 → 
  ate_apples = 1 → ate_oranges = 1 → ate_blueberries = 1 → 
  (apples_initial - ate_apples) + (oranges_initial - ate_oranges) + (blueberries_initial - ate_blueberries) = 26 :=
by
  intros
  simp [*]
  sorry

end mary_fruits_left_l365_365346


namespace triangle_qr_length_l365_365295

noncomputable def length_of_qr (PQ PR PN : ℝ) (N_midpoint : Prop) : ℝ :=
  if N_midpoint then sqrt 38 else 0

theorem triangle_qr_length (P Q R N : Type) 
  (PQ PR PN : ℝ) 
  (hPQ : PQ = 5) (hPR : PR = 10) 
  (hPN : PN = 6) 
  (N_midpoint : (N = (P + R) / 2)) : length_of_qr PQ PR PN N_midpoint = sqrt 38 :=
by
  sorry

end triangle_qr_length_l365_365295


namespace parallel_lines_slope_eq_l365_365070

theorem parallel_lines_slope_eq (k : ℚ) :
  (5 = 3 * k) → k = 5 / 3 :=
by
  intros h
  sorry

end parallel_lines_slope_eq_l365_365070


namespace function_neither_odd_nor_even_l365_365991

def f (x: ℝ) : ℝ := x^2 - 2

theorem function_neither_odd_nor_even (d : set ℝ) (hdom : d = set.Ioc (-5) 5) :
  ¬(∀ x ∈ d, f (-x) = f x) ∧ ¬(∀ x ∈ d, f (-x) = -f x) :=
by
  sorry

end function_neither_odd_nor_even_l365_365991


namespace cost_of_meal_l365_365517

   -- Define the problem conditions as Lean definitions
   def tax_rate : ℝ := 9.6 / 100
   def tip_rate : ℝ := 18 / 100
   def service_charge : ℝ := 5
   def total_spent : ℝ := 40

   -- Define the solution theorem
   theorem cost_of_meal : 
     ∃ x : ℝ, (x + tax_rate * x + tip_rate * x + service_charge = total_spent) ∧ x = 27.43 :=
   by
     -- Skip the proof
     sorry
   
end cost_of_meal_l365_365517


namespace failed_students_l365_365824

variable (n_t : ℕ) (n_a n_bc n_f : ℕ)

-- Conditions
def total_students : Prop := n_t = 32
def students_received_A : Prop := n_a = 0.25 * n_t
def students_B_or_C : Prop := n_bc = 0.25 * (n_t - n_a)

-- Proof statement
theorem failed_students : total_students ∧ students_received_A ∧ students_B_or_C → n_f = n_t - n_a - n_bc → n_f = 18 :=
by
  sorry

end failed_students_l365_365824


namespace find_min_max_value_l365_365181

open Real

theorem find_min_max_value (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) (h_det : b^2 - 4 * a * c < 0) :
  ∃ (min_val max_val: ℝ),
    min_val = (2 * d * sqrt (a * c)) / (b + 2 * sqrt (a * c)) ∧ 
    max_val = (2 * d * sqrt (a * c)) / (b - 2 * sqrt (a * c)) ∧
    (∀ x y : ℝ, a * x^2 + c * y^2 ≥ min_val ∧ a * x^2 + c * y^2 ≤ max_val) :=
by
  -- Proof goes here
  sorry

end find_min_max_value_l365_365181


namespace negation_of_proposition_l365_365468

theorem negation_of_proposition (x : ℝ) :
  ¬ (∃ x > -1, x^2 + x - 2018 > 0) ↔ ∀ x > -1, x^2 + x - 2018 ≤ 0 := sorry

end negation_of_proposition_l365_365468


namespace smallest_integer_value_l365_365949

theorem smallest_integer_value (n : ℤ) : ∃ (n : ℤ), n = 5 ∧ n^2 - 11*n + 28 < 0 :=
by
  use 5
  sorry

end smallest_integer_value_l365_365949


namespace find_P_Q_R_l365_365174

theorem find_P_Q_R :
  ∃ P Q R : ℝ, (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 → 
    (5 * x / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2)) 
    ∧ P = 5 ∧ Q = -5 ∧ R = -5 :=
by
  sorry

end find_P_Q_R_l365_365174


namespace scientific_notation_of_600000_l365_365166

theorem scientific_notation_of_600000 :
  600000 = 6 * 10^5 :=
sorry

end scientific_notation_of_600000_l365_365166


namespace rhombus_perimeter_l365_365793

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 24) (h2 : d2 = 10) : 4 * (Int.sqrt (d1/2 * d1/2 + d2/2 * d2/2)) = 52 :=
by {
  rw [h1, h2],
  have h_diag1 : (24 / 2 : ℤ) = 12 := by norm_num,
  have h_diag2 : (10 / 2 : ℤ) = 5 := by norm_num,
  have h_sq : (12 * 12 + 5 * 5 : ℤ) = 169 := by norm_num,
  have h_sqrt : Int.sqrt 169 = 13 := by norm_num [Int.sqrt_eq],
  rw [← h_sqrt, h_sq],
  norm_num
}

end rhombus_perimeter_l365_365793


namespace trig_equation_solution_l365_365775

open Real

noncomputable def solve_trig_equation (x : ℝ) : Prop :=
  ∀ (k n : ℤ), (
    (x = -atan (1 / 3) + π * k) ∨
    (x = π / 4 + π * n)
  )

theorem trig_equation_solution (x : ℝ) :
  (4 * sin ((π / 6) + x) * sin ((5 * π / 6) + x)) / (cos x^2) + 2 * tan x = 0 →
  solve_trig_equation x :=
by
  sorry

end trig_equation_solution_l365_365775


namespace students_play_neither_l365_365873

theorem students_play_neither (N F T F_inter_T : ℕ) (hN : N = 39) (hF : F = 26) (hT : T = 20) (hF_inter_T : F_inter_T = 17) :
  N - (F + T - F_inter_T) = 10 :=
by
  rw [hN, hF, hT, hF_inter_T]
  calc 39 - (26 + 20 - 17) = 39 - 29 : by rw [add_sub_assoc 26 20 17]
                       ... = 10     : by rw [sub_self 29]

end students_play_neither_l365_365873


namespace minimum_flour_cost_l365_365052

-- Definitions based on conditions provided
def loaves : ℕ := 12
def flour_per_loaf : ℕ := 4
def flour_needed : ℕ := loaves * flour_per_loaf

def ten_pound_bag_weight : ℕ := 10
def ten_pound_bag_cost : ℕ := 10

def twelve_pound_bag_weight : ℕ := 12
def twelve_pound_bag_cost : ℕ := 13

def cost_10_pound_bags : ℕ := (flour_needed + ten_pound_bag_weight - 1) / ten_pound_bag_weight * ten_pound_bag_cost
def cost_12_pound_bags : ℕ := (flour_needed + twelve_pound_bag_weight - 1) / twelve_pound_bag_weight * twelve_pound_bag_cost

theorem minimum_flour_cost : min cost_10_pound_bags cost_12_pound_bags = 50 := by
  sorry

end minimum_flour_cost_l365_365052


namespace influenza_probability_conditional_probability_A_given_flu_l365_365921

-- Definitions based on the conditions:
def flu_probability_A : ℝ := 0.06
def flu_probability_B : ℝ := 0.05
def flu_probability_C : ℝ := 0.04

def population_ratio_A : ℝ := 3.0
def population_ratio_B : ℝ := 5.0
def population_ratio_C : ℝ := 2.0

-- Calculating weighted probabilities:
def weighted_flu_probability : ℝ :=
  flu_probability_A * (population_ratio_A / (population_ratio_A + population_ratio_B + population_ratio_C)) +
  flu_probability_B * (population_ratio_B / (population_ratio_A + population_ratio_B + population_ratio_C)) +
  flu_probability_C * (population_ratio_C / (population_ratio_A + population_ratio_B + population_ratio_C))

-- Definition of the problem:
theorem influenza_probability :
  weighted_flu_probability = 0.051 :=
sorry

-- Calculating the conditional probability
def prob_A_given_flu : ℝ :=
  (flu_probability_A * (population_ratio_A / (population_ratio_A + population_ratio_B + population_ratio_C))) / weighted_flu_probability

-- Definition of the problem:
theorem conditional_probability_A_given_flu :
  prob_A_given_flu = 18 / 51 :=
sorry

end influenza_probability_conditional_probability_A_given_flu_l365_365921


namespace prop_p_iff_prop_q_iff_not_or_p_q_l365_365334

theorem prop_p_iff (m : ℝ) :
  (∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + (2 + m) = 0) ↔ (m ≤ -1 ∨ m ≥ 2) :=
sorry

theorem prop_q_iff (m : ℝ) :
  (∃ x y : ℝ, (x^2)/(1 - 2*m) + (y^2)/(m + 2) = 1) ↔ (m < -2 ∨ m > 1/2) :=
sorry

theorem not_or_p_q (m : ℝ) :
  ¬(∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + (2 + m) = 0) ∧
  ¬(∃ x y : ℝ, (x^2)/(1 - 2*m) + (y^2)/(m + 2) = 1) ↔
  (-1 < m ∧ m ≤ 1/2) :=
sorry

end prop_p_iff_prop_q_iff_not_or_p_q_l365_365334


namespace translation_of_cosine_graph_l365_365805

theorem translation_of_cosine_graph :
  ∀ x : ℝ, 3 * cos (3 * (x + (π / 6))) = 3 * cos (3 * x + (π / 2)) :=
by
  intro x
  sorry

end translation_of_cosine_graph_l365_365805


namespace range_of_k_l365_365226

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x
def e_val : ℝ := 2.7128
def interval : Set ℝ := {x | 0 < x ∧ x ≤ Real.pi / 2}

theorem range_of_k (k : ℝ) : (∀ x ∈ interval, f(x) ≥ k * x) ↔ k ∈ Set.Iic (Real.exp (Real.pi / 2)) := by
  sorry

end range_of_k_l365_365226


namespace base6_divisible_by_13_l365_365190

theorem base6_divisible_by_13 (d : ℕ) (h : d < 6) : 13 ∣ (435 + 42 * d) ↔ d = 5 := 
by
  -- Proof implementation will go here, but is currently omitted
  sorry

end base6_divisible_by_13_l365_365190


namespace mary_fruits_left_l365_365342

-- Conditions as definitions:
def mary_bought_apples : ℕ := 14
def mary_bought_oranges : ℕ := 9
def mary_bought_blueberries : ℕ := 6

def mary_ate_apples : ℕ := 1
def mary_ate_oranges : ℕ := 1
def mary_ate_blueberries : ℕ := 1

-- The problem statement:
theorem mary_fruits_left : 
  (mary_bought_apples - mary_ate_apples) + 
  (mary_bought_oranges - mary_ate_oranges) + 
  (mary_bought_blueberries - mary_ate_blueberries) = 26 := by
  sorry

end mary_fruits_left_l365_365342


namespace solve_for_p_l365_365373

theorem solve_for_p (q p : ℝ) (h : p^2 * q = p * q + p^2) : 
  p = 0 ∨ p = q / (q - 1) :=
by
  sorry

end solve_for_p_l365_365373


namespace domain_of_f_3x_minus_1_domain_of_f_l365_365093

-- Problem (1): Domain of f(3x - 1)
theorem domain_of_f_3x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -2 ≤ f x ∧ f x ≤ 1) →
  (∀ x, -1 / 3 ≤ x ∧ x ≤ 2 / 3) :=
by
  intro h
  sorry

-- Problem (2): Domain of f(x)
theorem domain_of_f (f : ℝ → ℝ) :
  (∀ x, -1 ≤ 2*x + 5 ∧ 2*x + 5 ≤ 4) →
  (∀ y, 3 ≤ y ∧ y ≤ 13) :=
by
  intro h
  sorry

end domain_of_f_3x_minus_1_domain_of_f_l365_365093


namespace geometric_sum_eq_l365_365739

theorem geometric_sum_eq {a : ℕ → ℝ} {n : ℕ} (h : ∀ k, 0 < k ≤ n → a (k + 1) = a k * a 2 / a 1)
  (S : ℝ) (hn : S = a 1 * (1 - (a 2 / a 1) ^ n) / (1 - a 2 / a 1)) :
  S = a 1 * a n * (∑ i in Finset.range n, 1 / a (i + 1)) :=
sorry

end geometric_sum_eq_l365_365739


namespace triangle_area_l365_365060

theorem triangle_area (A B C : ℝ × ℝ)
  (hA : A = (2, 3))
  (hB : B = (8, 3))
  (hC : C = (5, 10)) :
  let base := B.1 - A.1,
      height := C.2 - A.2,
      area := (1 / 2) * base * height
  in area = 21 := 
sorry

end triangle_area_l365_365060


namespace tan_17_plus_tan_28_plus_tan_17_tan_28_l365_365585

lemma tan_sum_formula 
  (A B : ℝ) :
  tan (A + B) = (tan A + tan B) / (1 - tan A * tan B) := sorry

theorem tan_17_plus_tan_28_plus_tan_17_tan_28 :
  tan 17 + tan 28 + (tan 17 * tan 28) = 1 := by
  have h : tan 45 = 1 := by sorry
  have tan_17_28 : tan 45 = (tan 17 + tan 28) / (1 - tan 17 * tan 28) := 
    tan_sum_formula 17 28
  rw [h] at tan_17_28
  sorry

end tan_17_plus_tan_28_plus_tan_17_tan_28_l365_365585


namespace true_propositions_l365_365469

open Real

-- Definitions of hypotheses
def prop1 (x : ℝ) := (x > 2) → (x > 0)
def prop2 := ∀ a : ℝ, (0 < a) → StrictMono (λ x : ℝ, a^x)
def prop3 := (∃ k : ℕ, (k > 0) ∧ ∀ x, sin (x + k * π) = sin x)
def prop4 (x y : ℝ) := ((x^2 + y^2 = 0) → (x = 0 ∧ y = 0)) ∧ ((x = 0 ∧ y = 0) → (x^2 + y^2 = 0))

theorem true_propositions :
  ¬ prop1 ∧ ¬ prop2 ∧ prop3 ∧ ¬ prop4 :=
sorry

end true_propositions_l365_365469


namespace problem_solution_l365_365439

theorem problem_solution :
  (3012 - 2933)^2 / 196 = 32 := sorry

end problem_solution_l365_365439


namespace shelves_used_l365_365908

def coloring_books := 87
def sold_books := 33
def books_per_shelf := 6

theorem shelves_used (h1: coloring_books - sold_books = 54) : 54 / books_per_shelf = 9 :=
by
  sorry

end shelves_used_l365_365908


namespace parabola_property_l365_365605

-- Define the conditions of the problem in Lean
variable (a b : ℝ)
variable (h1 : (a, b) ∈ {p : ℝ × ℝ | p.1^2 = 20 * p.2}) -- P lies on the parabola x^2 = 20y
variable (h2 : dist (a, b) (0, 5) = 25) -- Distance from P to focus F

theorem parabola_property : |a * b| = 400 := by
  sorry

end parabola_property_l365_365605


namespace generalized_inequality_l365_365756

theorem generalized_inequality (n : ℕ) (h : 1 ≤ n) :
  (finset.range(n).sum (λ i, real.sqrt ((i+1) * (i+2)))) < (n * (n + 2) / 2) :=
sorry

end generalized_inequality_l365_365756


namespace minimum_value_of_expression_l365_365615

theorem minimum_value_of_expression {x y : ℝ} (h : x^2 + y^2 = 4) :
  ∃ (m : ℝ), (∀ xy_value, 
    (h : xy_value = x * y → ∃ xy_sum, xy_sum = x + y - 2 → 
    (∀ expr_value, expr_value = xy_value / xy_sum → expr_value ≥ m))) ∧ m = 1 - Real.sqrt 2 :=
sorry

end minimum_value_of_expression_l365_365615


namespace range_of_a_l365_365394

open Real

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem range_of_a (a : ℝ) (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) :
  (f (a * sin θ) + f (1 - a) > 0) → a ≤ 1 :=
sorry

end range_of_a_l365_365394


namespace interval_of_monotonic_increase_parallel_vectors_tan_x_perpendicular_vectors_value_x_l365_365656

-- Conditions from the problem
def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
def f (x : ℝ) : ℝ := 2 * ((Real.sin x * (Real.sqrt 3 * Real.cos x)) + (Real.cos x * Real.cos x)) - 1

-- Definitions of the corresponding questions
-- Question 1: Interval of monotonic increase
theorem interval_of_monotonic_increase (k : ℤ) :
  interval_of_monotonic_increase (f x) = set.Icc (k * Real.pi - (Real.pi / 3)) (k * Real.pi + (Real.pi / 6)) :=
sorry

-- Question 2: Parallel vectors result in a specific value of tan x
theorem parallel_vectors_tan_x (h : ∀ x : ℝ, a x ∥ b x) :
  (∃ x: ℝ, Real.tan x = Real.sqrt 3) :=
sorry

-- Question 3: Perpendicular vectors result in a specific value of x
theorem perpendicular_vectors_value_x (h : ∀ x : ℝ, a x ⊥ b x) :
  (∃ x: ℝ, x = 5 * Real.pi / 6) :=
sorry

end interval_of_monotonic_increase_parallel_vectors_tan_x_perpendicular_vectors_value_x_l365_365656


namespace sum_double_series_l365_365531

theorem sum_double_series :
  (∑ n from 2 to ∞, ∑ k from 1 to (n-1), k / 3^(n+k)) = 9 / 136 :=
by
  sorry

end sum_double_series_l365_365531


namespace cinema_total_seats_l365_365688

/-- 
In the "Triangle" cinema, seats are arranged in the form of a triangle: in the first row, there is one seat numbered 1; 
in the second row, there are seats numbered 2 and 3; in the third row, seats are numbered 4, 5, 6, and so on. 
The best seat in the cinema hall is the one located at the center of the height drawn from the vertex of the triangle 
corresponding to seat number 1. If the best seat is numbered 265, 
then the problem is to prove that the total number of seats in the cinema hall is 1035.
-/
theorem cinema_total_seats (rows : ℕ) (total_seats : ℕ) (best_seat : ℕ) 
  (h_best_seat : best_seat = 265) (h_odd_rows : rows % 2 = 1) 
  (h_rows_best_seat : (rows = 2 * (floor (real.sqrt (2 * best_seat + 1) / 2) : ℕ) + 1)) :
  total_seats = 1035 := by
  sorry

end cinema_total_seats_l365_365688


namespace log_865_bounds_l365_365041

theorem log_865_bounds : 
  let log_100 := log 100
  let log_1000 := log 1000
  (∀ x : ℝ, ∀ y : ℝ, (1 < x) → (x < y) → log x < log y) →
  log_100 = 2 → log_1000 = 3 → 2 < log 865 ∧ log 865 < 3 → 5 = 2 + 3 :=
by
  intro log_100 log_1000 log_increasing log_100_val log_1000_val log_bounds
  sorry

end log_865_bounds_l365_365041


namespace total_distance_l365_365749

theorem total_distance (x y : ℝ) (h1 : x * y = 18) :
  let D2 := (y - 1) * (x + 1)
  let D3 := 15
  let D_total := 18 + D2 + D3
  D_total = y * x + y - x + 32 :=
by
  let D2 := (y - 1) * (x + 1)
  let D3 := 15
  let D_total := 18 + D2 + D3
  sorry

end total_distance_l365_365749


namespace max_circles_packed_correctly_l365_365358

-- Diameter of each circle
def circle_diameter : ℝ := 1

-- Side length of the equilateral triangle formed by centers of tangent circles
def side_length : ℝ := circle_diameter

-- Height of an equilateral triangle with side length 1
def triangle_height : ℝ := (Real.sqrt 3) / 2

-- Given dimensions of the square
def square_side_length : ℝ := 100

-- Number of original circles
def original_circle_count : ℕ := 10000

-- Number of circles that can fit in one row with the given conditions
def row_circle_count : ℕ := 100

-- Number of double rows that fit in the given height using hexagonal packing
def double_row_count : ℝ := square_side_length / Real.sqrt 3

-- Number of circles in one set of double rows (hexagonal packing: 100 + 99)
def double_row_circle_count : ℕ := 199

-- Calculate the new total number of circles in hexagonal packing
def new_circle_count : ℕ := (double_row_count.toNat * double_row_circle_count) + row_circle_count

-- Calculate the maximum additional circles that can be packed
def max_additional_circles : ℕ := new_circle_count - original_circle_count

theorem max_circles_packed_correctly : max_additional_circles = 1443 := by
  sorry

end max_circles_packed_correctly_l365_365358


namespace minyoung_money_l365_365027

theorem minyoung_money (A M : ℕ) (h1 : M = 90 * A) (h2 : M = 60 * A + 270) : M = 810 :=
by 
  sorry

end minyoung_money_l365_365027


namespace value_40th_number_l365_365801

def starts_with (n : ℕ) : ℕ :=
  3 * n

def num_elements (n : ℕ) : ℕ :=
  2 * n

noncomputable def cumulative_elements (n : ℕ) : ℕ :=
  (finset.range (n + 1)).sum num_elements

theorem value_40th_number :
  ∃ (n : ℕ), cumulative_elements (n - 1) < 40 ∧ 40 ≤ cumulative_elements n ∧ starts_with n = 18 :=
begin
  sorry
end

end value_40th_number_l365_365801


namespace value_of_a_plus_b_squared_l365_365158

-- Define the quadratic equation
def quadratic_eq (x : ℂ) : Prop :=
  5 * x^2 - 4 * x + 15 = 0

-- Define the values a and b based on the given solution
def a : ℝ := 2 / 5
def b : ℝ := Real.sqrt 284 / 10

-- State the main theorem to prove
theorem value_of_a_plus_b_squared :
  (∀ x : ℂ, quadratic_eq x → x = a + b * Complex.I ∨ x = a - b * Complex.I) →
  a + b^2 = 81 / 25 :=
by
  sorry

end value_of_a_plus_b_squared_l365_365158


namespace line_graph_displays_trend_l365_365145

-- Define the types of statistical graphs
inductive StatisticalGraph : Type
| barGraph : StatisticalGraph
| lineGraph : StatisticalGraph
| pieChart : StatisticalGraph
| histogram : StatisticalGraph

-- Define the property of displaying trends over time
def displaysTrend (g : StatisticalGraph) : Prop := 
  g = StatisticalGraph.lineGraph

-- Theorem to prove that the type of statistical graph that displays the trend of data is the line graph
theorem line_graph_displays_trend : displaysTrend StatisticalGraph.lineGraph :=
sorry

end line_graph_displays_trend_l365_365145


namespace sum_of_arithmetic_sequence_zero_l365_365009

noncomputable def arithmetic_sequence_sum (S : ℕ → ℤ) : Prop :=
S 20 = S 40

theorem sum_of_arithmetic_sequence_zero {S : ℕ → ℤ} (h : arithmetic_sequence_sum S) : 
  S 60 = 0 :=
sorry

end sum_of_arithmetic_sequence_zero_l365_365009


namespace fans_received_all_items_l365_365565

theorem fans_received_all_items (total_fans : ℕ) 
    (h_total_fans : total_fans = 5000) 
    (tshirt_interval : ℕ) (h_tshirt_interval : tshirt_interval = 90)
    (cap_interval : ℕ) (h_cap_interval : cap_interval = 45)
    (scarf_interval : ℕ) (h_scarf_interval : scarf_interval = 60) :
    let lcm := Nat.lcm (Nat.lcm tshirt_interval cap_interval) scarf_interval in
    total_fans / lcm = 27 := 
by
  let lcm := Nat.lcm (Nat.lcm tshirt_interval cap_interval) scarf_interval
  have h_lcm : lcm = 180 := by sorry
  have h_div : total_fans / lcm = 27 := by sorry
  assumption

end fans_received_all_items_l365_365565


namespace number_of_correct_conclusions_l365_365400

def quadratic_function := ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ t : ℝ, (y : ℝ) = a * t ^ 2 + b * t + c) ∧
  (y (-2) = m) ∧
  (y (-1) = 3) ∧
  (y 1 = 4) ∧
  (y 2 = 3) ∧
  (y t = n)

def conclusions_correct (a b c : ℝ) (m n : ℝ) (t : ℝ) : Prop :=
  (a * b * c < 0) ∧    -- Conclusion ①
  (x = 1) = false ∧         -- Conclusion ② (False as per solution explanation)
  (0 and 1 are roots of (a * x ^ 2 + b * x + c - 4 = 0)) = false ∧ -- Conclusion ③ (False)
  (t > 3 → m > n) -- Conclusion ④

theorem number_of_correct_conclusions :
  ∀ (a b c m n : ℝ) (t : ℝ), 
  quadratic_function a b c m n t → conclusions_correct a b c m n t →
  number_of_correct_conclusions = 2 :=
begin
  sorry
end

end number_of_correct_conclusions_l365_365400


namespace sequence_sum_S_2018_l365_365747

noncomputable def a : ℕ → ℚ
| 0       := 4 / 5
| (n + 1) := if a n ≤ 1 / 2 then 2 * a n else 2 * a n - 1

def S : ℕ → ℚ
| 0       := a 0
| (n + 1) := S n + a (n + 1)

theorem sequence_sum_S_2018:
  S 2018 = 5047 / 5 := 
sorry

end sequence_sum_S_2018_l365_365747


namespace water_bottle_capacity_l365_365506

theorem water_bottle_capacity:
  let capacity_250ml := 250
  let capacity_600ml := 600
  let pours_250ml := 20
  let pours_600ml := 13
  let total_ml := (capacity_250ml * pours_250ml) + (capacity_600ml * pours_600ml)
  in (total_ml / 1000 : ℝ) = 12.8 :=
by 
  sorry

end water_bottle_capacity_l365_365506


namespace nested_sum_equals_fraction_l365_365529

noncomputable def nested_sum : ℝ := ∑' (n : ℕ) in Ico 2 (⊤), ∑' (k : ℕ) in Ico 1 n, (k : ℝ) / 3^(n + k)

theorem nested_sum_equals_fraction :
  nested_sum = 3 / 64 := sorry

end nested_sum_equals_fraction_l365_365529


namespace triangle_ratio_l365_365134

theorem triangle_ratio (A B C D E O : Point)
  (h₁ : D ∈ segment A C) (h₂ : E ∈ segment A B)
  (h₃ : O ∈ line B D) (h₄ : O ∈ line C E)
  (area_OBE : ℝ) (area_OBC : ℝ) (area_OCD : ℝ)
  (h₅ : area_OBE = 15) (h₆ : area_OBC = 30) (h₇ : area_OCD = 24) :
  (segment_ratio A E A B = 2) :=
sorry

end triangle_ratio_l365_365134


namespace problem_solution_l365_365078

lemma factor_def (m n : ℕ) : n ∣ m ↔ ∃ k, m = n * k := by sorry

def is_true_A : Prop := 4 ∣ 24
def is_true_B : Prop := 19 ∣ 209 ∧ ¬ (19 ∣ 63)
def is_true_C : Prop := ¬ (30 ∣ 90) ∧ ¬ (30 ∣ 65)
def is_true_D : Prop := 11 ∣ 33 ∧ ¬ (11 ∣ 77)
def is_true_E : Prop := 9 ∣ 180

theorem problem_solution : (is_true_A ∧ is_true_B ∧ is_true_E) ∧ ¬(is_true_C) ∧ ¬(is_true_D) :=
  by sorry

end problem_solution_l365_365078


namespace legs_in_room_l365_365829

def total_legs_in_room (tables4 : Nat) (sofa : Nat) (chairs4 : Nat) (tables3 : Nat) (table1 : Nat) (rocking_chair2 : Nat) : Nat :=
  (tables4 * 4) + (sofa * 4) + (chairs4 * 4) + (tables3 * 3) + (table1 * 1) + (rocking_chair2 * 2)

theorem legs_in_room :
  total_legs_in_room 4 1 2 3 1 1 = 40 :=
by
  -- Skipping proof steps
  sorry

end legs_in_room_l365_365829


namespace solve_for_n_l365_365773

theorem solve_for_n : ∃ n : ℚ, 3^(2 * n + 1) = (1 / 81) :=
by
  use -5 / 2
  -- Proof steps would go here, but are omitted
  sorry

end solve_for_n_l365_365773


namespace vector_length_add_eq_sqrt_3_l365_365239

variables {V : Type*} [inner_product_space ℝ V] {a b : V}
noncomputable def vector_length (v : V) := real.sqrt (inner_product_space.norm_sq v)

theorem vector_length_add_eq_sqrt_3 (ha : vector_length a = 1) (hb : vector_length b = 1) (h_perp : inner_product_space.dot_product a (a - 2*b) = 0) :
  vector_length (a + b) = real.sqrt 3 :=
begin
  -- proof goes here
  sorry
end

end vector_length_add_eq_sqrt_3_l365_365239


namespace average_is_50x_l365_365010

theorem average_is_50x (x : ℝ) (h : (∑ i in finset.range 150, i + x) / 150 = 50 * x) : 
  x = 11175 / 7499 :=
by
  -- sum of the numbers from 1 to 149 is known
  have sum_1_to_149 : ∑ i in finset.range 149, i = 11175 := sorry,
  -- setting up the average equation and solving for x
  sorry

end average_is_50x_l365_365010


namespace value_of_expression_l365_365854

theorem value_of_expression : 3^2 * 5 * 7^2 * 11 = 24255 := by
  have h1 : 3^2 = 9 := by norm_num
  have h2 : 7^2 = 49 := by norm_num
  calc
    3^2 * 5 * 7^2 * 11
        = 9 * 5 * 7^2 * 11 : by rw h1
    ... = 9 * 5 * 49 * 11  : by rw h2
    ... = 24255            : by norm_num

end value_of_expression_l365_365854


namespace logarithmic_expression_floor_l365_365065

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithmic_expression_floor :
  (Int.floor (log_base 3 (1006 : ℝ))) = 6 :=
begin
  -- no proof required
  sorry
end

end logarithmic_expression_floor_l365_365065


namespace liam_cleans_fraction_l365_365255

theorem liam_cleans_fraction (total_time clean_time : ℕ) (H : total_time = 60) (H2 : clean_time = 15) :
  (clean_time : ℚ) / total_time = 1 / 4 := 
by
  simp [H, H2]
  norm_num
  sorry

end liam_cleans_fraction_l365_365255


namespace positive_integer_divisors_of_sum_l365_365985

theorem positive_integer_divisors_of_sum (n : ℕ) :
  (∃ n_values : Finset ℕ, 
    (∀ n ∈ n_values, n > 0 
      ∧ (n * (n + 1)) ∣ (2 * 10 * n)) 
      ∧ n_values.card = 5) :=
by
  sorry

end positive_integer_divisors_of_sum_l365_365985


namespace true_propositions_l365_365802

theorem true_propositions :
  ¬(from_uniform_prod_line_sampling_is_stratified_sampling) ∧
  (regression_line_property holds_for_correlated_vars x y) ∧
  (regression_line_eq_increase_by_0_point_2_unit holds_for_explanatory_var_increase) ∧
  ¬(weaker_corr_coeff_closer_to_1 holds_for_random_vars) ↔ Correct_Answer = Prop_D :=
by {
  sorry
}

end true_propositions_l365_365802


namespace finite_inf_n_rephinado_primes_l365_365433

noncomputable def delta := (1 + Real.sqrt 5) / 2 + 1

def is_nth_residue_mod (p n : ℕ) (a : ℕ) : Prop :=
  ∃ x : ℕ, x^n % p = a % p

def is_n_rephinado (p n : ℕ) : Prop :=
  n ∣ (p - 1) ∧ ∀ a, 1 ≤ a ∧ a ≤ Nat.floor (Real.sqrt (p ^ (1 / delta))) → is_nth_residue_mod p n a

theorem finite_inf_n_rephinado_primes :
  ∀ n : ℕ, ¬(∃ S : Set ℕ, S.infinite ∧ ∀ p ∈ S, is_n_rephinado p n) :=
sorry

end finite_inf_n_rephinado_primes_l365_365433


namespace profit_at_least_1300_l365_365888

theorem profit_at_least_1300 (x : ℝ) 
  (P : ℝ := 160 - 2 * x) 
  (R : ℝ := 500 + 30 * x) 
  (profit : ℝ := (160 - 2 * x) * x - (500 + 30 * x)) :
  profit ≥ 1300 → 20 ≤ x ∧ x ≤ 45 :=
by
  intros h
  have eq_profit : profit = -2 * x^2 + 130 * x - 500, linarith,
  rw eq_profit at h,
  have h₁ : -2 * x^2 + 130 * x - 500 ≥ 1300 := h,
  linarith,
  sorry

end profit_at_least_1300_l365_365888


namespace decreasing_exponential_quadratic_l365_365639

theorem decreasing_exponential_quadratic {f : ℝ → ℝ} (a : ℝ) 
    (h : ∀ x y ∈ Ioo 0 1, x < y → f x ≥ f y) :
    a ≥ 2 :=
begin
    sorry
end

end decreasing_exponential_quadratic_l365_365639


namespace domain_of_f_l365_365946

def f (x : ℝ) : ℝ := 1 / Real.log (3 * x + 1)

theorem domain_of_f :
  {x : ℝ | 3 * x + 1 > 0 ∧ 3 * x + 1 ≠ 1} = {x : ℝ | x > -1/3 ∧ x ≠ 0} :=
by
  sorry

end domain_of_f_l365_365946


namespace Tim_total_expenditure_l365_365804

theorem Tim_total_expenditure 
  (appetizer_price : ℝ) (main_course_price : ℝ) (dessert_price : ℝ)
  (appetizer_tip_percentage : ℝ) (main_course_tip_percentage : ℝ) (dessert_tip_percentage : ℝ) :
  appetizer_price = 12.35 →
  main_course_price = 27.50 →
  dessert_price = 9.95 →
  appetizer_tip_percentage = 0.18 →
  main_course_tip_percentage = 0.20 →
  dessert_tip_percentage = 0.15 →
  appetizer_price * (1 + appetizer_tip_percentage) + 
  main_course_price * (1 + main_course_tip_percentage) + 
  dessert_price * (1 + dessert_tip_percentage) = 12.35 * 1.18 + 27.50 * 1.20 + 9.95 * 1.15 :=
  by sorry

end Tim_total_expenditure_l365_365804


namespace four_pow_three_letter_is_L_l365_365516

theorem four_pow_three_letter_is_L :
  let alphabet : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  in alphabet.nth ((64 % 26) - 1) = some 'L' := 
by
  sorry

end four_pow_three_letter_is_L_l365_365516


namespace integer_approx_sum_diff_l365_365996

noncomputable def find_integer_approx (n : ℕ) (a : Fin n → ℝ) : Fin n → ℤ :=
  sorry

theorem integer_approx_sum_diff (n : ℕ) (a : Fin n → ℝ) :
  ∃ b : Fin n → ℤ, (∀ i, |a i - b i| < 1) ∧ (∀ S : Finset (Fin n), |S.sum (λ i, a i) - S.sum (λ i, b i)| ≤ (n + 1) / 4) :=
sorry

end integer_approx_sum_diff_l365_365996


namespace mistaken_multiplier_is_34_l365_365890

-- Define the main conditions
def correct_number : ℕ := 135
def correct_multiplier : ℕ := 43
def difference : ℕ := 1215

-- Define what we need to prove
theorem mistaken_multiplier_is_34 :
  (correct_number * correct_multiplier - correct_number * x = difference) →
  x = 34 :=
by
  sorry

end mistaken_multiplier_is_34_l365_365890


namespace ball_hits_ground_time_l365_365798

theorem ball_hits_ground_time :
  ∃ t : ℚ, -20 * t^2 + 30 * t + 50 = 0 ∧ t = 5 / 2 :=
sorry

end ball_hits_ground_time_l365_365798


namespace triangle_third_side_l365_365354

theorem triangle_third_side (A B C : ℝ) (a b c : ℝ) 
  (h1 : A = 3 * C) (h2 : b = 6) (h3 : c = 18) 
  (law_of_cosines : cos C = (a^2 + c^2 - b^2) / (2 * a * c))
  (law_of_sines : sin C = (3 * sin C - 4 * (sin C)^3)) :
  a = 72 := 
sorry

end triangle_third_side_l365_365354


namespace number_of_distinct_integers_l365_365398

theorem number_of_distinct_integers : 
  let primes := [2, 3, 5, 7] in
  let num_one_digit := primes.length in
  let num_two_digit := primes.length * (primes.length - 1) in
  let num_three_digit := primes.length * (primes.length - 1) * (primes.length - 2) in
  let num_four_digit := primes.length * (primes.length - 1) * (primes.length - 2) * (primes.length - 3) in
  num_one_digit + num_two_digit + num_three_digit + num_four_digit = 64 :=
by
  sorry

end number_of_distinct_integers_l365_365398


namespace option_one_cost_option_two_cost_cost_effectiveness_l365_365476

-- Definition of costs based on conditions
def price_of_suit : ℕ := 500
def price_of_tie : ℕ := 60
def discount_option_one (x : ℕ) : ℕ := 60 * x + 8800
def discount_option_two (x : ℕ) : ℕ := 54 * x + 9000

-- Theorem statements
theorem option_one_cost (x : ℕ) (hx : x > 20) : discount_option_one x = 60 * x + 8800 :=
by sorry

theorem option_two_cost (x : ℕ) (hx : x > 20) : discount_option_two x = 54 * x + 9000 :=
by sorry

theorem cost_effectiveness (x : ℕ) (hx : x = 30) : discount_option_one x < discount_option_two x :=
by sorry

end option_one_cost_option_two_cost_cost_effectiveness_l365_365476


namespace greg_experienced_less_rain_l365_365865

theorem greg_experienced_less_rain (rain_day1 rain_day2 rain_day3 rain_house : ℕ) 
  (h1 : rain_day1 = 3) 
  (h2 : rain_day2 = 6) 
  (h3 : rain_day3 = 5) 
  (h4 : rain_house = 26) :
  rain_house - (rain_day1 + rain_day2 + rain_day3) = 12 :=
by
  sorry

end greg_experienced_less_rain_l365_365865


namespace given_expression_simplifies_to_l365_365453

-- Given conditions: a ≠ ±1, a ≠ 0, b ≠ -1, b ≠ 0
variable (a b : ℝ)
variable (ha1 : a ≠ 1)
variable (ha2 : a ≠ -1)
variable (ha3 : a ≠ 0)
variable (hb1 : b ≠ 0)
variable (hb2 : b ≠ -1)

theorem given_expression_simplifies_to (h1 : a ≠ 1) (h2 : a ≠ -1) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : b ≠ -1) :
    (a * b^(2/3) - b^(2/3) - a + 1) / ((1 - a^(1/3)) * ((a^(1/3) + 1)^2 - a^(1/3)) * (b^(1/3) + 1))
  + (a * b)^(1/3) * (1/a^(1/3) + 1/b^(1/3)) = 1 + a^(1/3) := by
  sorry

end given_expression_simplifies_to_l365_365453


namespace half_angle_quadrant_l365_365621

theorem half_angle_quadrant (θ : ℝ) (h_cos : |cos θ| = cos θ) (h_tan : |tan θ| = -tan θ) : 
  let θ_div_2 := θ / 2 in
  (π/2 < θ_div_2 ∧ θ_div_2 < π) ∨ (3 * π / 2 < θ_div_2 ∧ θ_div_2 ≤ 2 * π) :=
sorry

end half_angle_quadrant_l365_365621


namespace sum_eq_9_div_64_l365_365548

noncomputable def double_sum : ℝ := ∑' (n : ℕ) in (set.Ici 2 : set ℕ), ∑' (k : ℕ) in set.Ico 1 n, (k : ℝ) / 3^(n + k)

theorem sum_eq_9_div_64 : double_sum = 9 / 64 := 
by 
sorry

end sum_eq_9_div_64_l365_365548


namespace exist_xy_l365_365730

-- Define the smallest integer greater than the square root of n
def smallest_integer_gt_sqrt (n : ℕ) : ℕ :=
  if h : n > 0 then (⌈Real.sqrt n⌉.toNat) else 0

-- Main theorem to be proved
theorem exist_xy (n : ℕ) (e : ℕ) (a : ℕ) (h_n : n > 1) (h_e : e = smallest_integer_gt_sqrt n) (h_coprime : Nat.gcd a n = 1) : 
  ∃ (x y : ℕ), (x ≤ e - 1) ∧ (y ≤ e - 1) ∧ (a * y ≡ x [MOD n] ∨ a * y ≡ -x [MOD n]) :=
sorry

end exist_xy_l365_365730


namespace greater_combined_area_l365_365665

noncomputable def area_of_rectangle (length : ℝ) (width : ℝ) : ℝ :=
  length * width

noncomputable def combined_area (length : ℝ) (width : ℝ) : ℝ :=
  2 * (area_of_rectangle length width)

theorem greater_combined_area 
  (length1 width1 length2 width2 : ℝ)
  (h1 : length1 = 11) (h2 : width1 = 13)
  (h3 : length2 = 6.5) (h4 : width2 = 11) :
  combined_area length1 width1 - combined_area length2 width2 = 143 :=
by
  rw [h1, h2, h3, h4]
  sorry

end greater_combined_area_l365_365665


namespace product_value_l365_365856

-- Definitions of each term
def term (n : Nat) : Rat :=
  1 + 1 / (n^2 : ℚ)

-- Define the product of these terms
def product : Rat :=
  term 1 * term 2 * term 3 * term 4 * term 5 * term 6

-- The proof problem statement that needs to be verified
theorem product_value :
  product = 16661 / 3240 :=
sorry

end product_value_l365_365856


namespace sum_of_tangencies_l365_365552

noncomputable def f (x : ℝ) : ℝ := max (-7 * x - 23) (max (2 * x + 5) (5 * x + 17))

noncomputable def q (x : ℝ) : ℝ := sorry  -- since the exact form of q is not specified, we use sorry here

-- Define the tangency condition
def is_tangent (q f : ℝ → ℝ) (x : ℝ) : Prop := (q x = f x) ∧ (deriv q x = deriv f x)

-- Define the three points of tangency
variable {x₄ x₅ x₆ : ℝ}

-- q(x) is tangent to f(x) at points x₄, x₅, x₆
axiom tangent_x₄ : is_tangent q f x₄
axiom tangent_x₅ : is_tangent q f x₅
axiom tangent_x₆ : is_tangent q f x₆

-- Now state the theorem
theorem sum_of_tangencies : x₄ + x₅ + x₆ = -70 / 9 :=
sorry

end sum_of_tangencies_l365_365552


namespace simplify_and_evaluate_evaluate_at_zero_l365_365771

noncomputable def simplified_expression (x : ℤ) : ℚ :=
  (1 - 1/(x-1)) / ((x^2 - 4*x + 4) / (x^2 - 1))

theorem simplify_and_evaluate (x : ℤ) (h : x ≠ 1 ∧ x ≠ 2 ∧ x ≠ -1) : 
  simplified_expression x = (x+1)/(x-2) :=
by
  sorry

theorem evaluate_at_zero : simplified_expression 0 = -1/2 :=
by
  sorry

end simplify_and_evaluate_evaluate_at_zero_l365_365771


namespace probability_two_red_balls_randomly_picked_l365_365471

theorem probability_two_red_balls_randomly_picked :
  (3/9) * (2/8) = 1/12 :=
by sorry

end probability_two_red_balls_randomly_picked_l365_365471


namespace min_mn_proof_l365_365618

def min_mn_value (m n : ℝ) (λ μ : ℝ) : ℝ :=
  m + n

theorem min_mn_proof :
  ∀ (m n λ μ : ℝ), 
    0 ≤ λ ∧ λ ≤ 1 ∧ 1 ≤ μ ∧ μ ≤ 2 ∧
    m > 0 ∧ n > 0 ∧
    (λ + μ) ≤ m ∧ μ ≤ n ∧ 
    (∀ x y : ℝ, 
      (x = λ + μ) ∧ 
      (y = μ) ∧ 
      (2 : ℝ) = (x / m + y / n)) →
      (min_mn_value m n λ μ) = (5 / 2 + Real.sqrt 6) := 
by 
  sorry

end min_mn_proof_l365_365618


namespace part1_part2_l365_365631

variable {a : ℕ → ℕ}

-- Condition: The sequence is increasing and geometric.
def increasing_geometric_sequence (a : ℕ → ℕ) (q : ℕ) :=
  ∀ n, a (n+1) = a n * q

-- Given Conditions
def condition1 := a 1 + a 4 = 9
def condition2 := a 2 * a 3 = 8

-- General formula for the sequence
theorem part1 (h : increasing_geometric_sequence a 2) :
  condition1 → condition2 → (∀ n, a n = 2^(n-1)) :=
sorry

-- Sum of the first n terms and related sequence
noncomputable def S (n : ℕ) := ∑ i in (Finset.range n), a (i + 1)
noncomputable def b (n : ℕ) := a (n + 1) / (S n * S (n + 1))

-- Sum of the first n terms of sequence {b_n}
theorem part2 (h : increasing_geometric_sequence a 2) :
  condition1 → condition2 → 
  (∀ n, (Finset.range n).sum b = 1 - 1 / (2^(n+1) - 1)) :=
sorry

end part1_part2_l365_365631


namespace imaginary_part_of_z_l365_365596

open Complex

-- Define the context
variables (z : ℂ) (a b : ℂ)

-- Define the condition
def condition := (1 - 2*I) * z = 5 * I

-- Lean 4 statement to prove the imaginary part of z 
theorem imaginary_part_of_z (h : condition z) : z.im = 1 :=
sorry

end imaginary_part_of_z_l365_365596


namespace smallest_section_area_l365_365687

-- Assuming the rectangle ABCD coordinates and the line OY conditions
variables {A B C D O Y : ℝ}
variables {AB_CD_area : ℝ} -- The total area of the rectangle ABCD
variables {k : ℝ} -- ratio k = 1/4 for segment division

-- The line OY divides side AB and CD in the ratio 1:3 starting from vertices B and D
def divides_in_ratio (f : ℝ → ℝ) (l : ℝ) : Prop := f(l) = 1 / 4 * l

-- Smallest section's area is 3/16 of the total area of the rectangle
theorem smallest_section_area (AB_eq_CD : divides_in_ratio (λ xy, xy) (side length AB or CD))
  (OY_divides_AB : divides_in_ratio (λ x, x) AB)
  (OY_divides_CD : divides_in_ratio (λ y, y) CD) :
  ∃ smallest_area_fraction : ℝ, smallest_area_fraction = 3 / 16 :=
by
  -- Construct the geometric setup
  sorry

end smallest_section_area_l365_365687


namespace total_drink_volume_l365_365108

variable (T : ℝ)

theorem total_drink_volume :
  (0.15 * T + 0.60 * T + 0.25 * T = 35) → T = 140 :=
by
  intros h
  have h1 : (0.25 * T) = 35 := by sorry
  have h2 : T = 140 := by sorry
  exact h2

end total_drink_volume_l365_365108


namespace rectangular_field_area_l365_365087

theorem rectangular_field_area (a c : ℝ) (h_a : a = 13) (h_c : c = 17) :
  ∃ b : ℝ, (b = 2 * Real.sqrt 30) ∧ (a * b = 26 * Real.sqrt 30) :=
by
  sorry

end rectangular_field_area_l365_365087


namespace marty_combinations_l365_365751

theorem marty_combinations : 
  ∃ n : ℕ, n = 5 * 4 ∧ n = 20 :=
by
  sorry

end marty_combinations_l365_365751


namespace solution_of_quartic_l365_365966

theorem solution_of_quartic (z : ℝ) : z^4 - 8 * z^2 + 12 = 0 → (z = -√6 ∨ z = -√2 ∨ z = √2 ∨ z = √6) :=
by
  intro h
  sorry

end solution_of_quartic_l365_365966


namespace max_negative_pairs_l365_365431

theorem max_negative_pairs (n : ℕ) (hn : 0 < n) : 
  let x := if n % 2 = 0 then n / 2 else (n - 1) / 2 in
  let y := n - x in
  (x = n / 2 ∧ y = n / 2 ∨ x = (n - 1) / 2 ∧ y = (n + 1) / 2 ∨ x = (n + 1) / 2 ∧ y = (n - 1) / 2) :=
sorry

end max_negative_pairs_l365_365431


namespace min_value_quadratic_l365_365442

theorem min_value_quadratic (x : ℝ) : ∃ x, x = -7 ∧ (x^2 + 14 * x + 24 = -25) := sorry

end min_value_quadratic_l365_365442


namespace prob_one_transformed_in_R_l365_365901

noncomputable def in_region (z : ℂ) : Prop :=
  let x := z.re
  let y := z.im
  -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2

noncomputable def transform (z : ℂ) : ℂ :=
  (1/2 + 1/2 * complex.I) * z

theorem prob_one_transformed_in_R (z : ℂ) (hz : in_region z) :
  in_region (transform z) :=
by sorry

end prob_one_transformed_in_R_l365_365901


namespace cartesian_eq_of_curve_C_midpoint_polar_coordinates_l365_365689

-- Define the equations and conditions
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def line_l (x : ℝ) : ℝ :=
  Real.sqrt 3 * x

-- Given the polar equation of the curve C
def curve_C_polar (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ)^2 = 8 * Real.sin θ

-- Cartesian equation derived from the polar equation of curve C
def curve_C_cartesian (x y : ℝ) : Prop :=
  x^2 = 8 * y

-- Given line l's intersection with curve C at points O and P
def intersection_O : ℝ × ℝ := (0, 0)
def intersection_P : ℝ × ℝ := (16 * Real.sqrt 3, Real.pi / 3)

-- Midpoint N of line segment OP
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Prove the Cartesian equation of the curve C
theorem cartesian_eq_of_curve_C (ρ θ : ℝ) :
  curve_C_polar ρ θ → curve_C_cartesian (ρ * Real.cos θ) (ρ * Real.sin θ) :=
sorry

-- Prove the polar coordinates of the midpoint N
theorem midpoint_polar_coordinates :
  midpoint intersection_O intersection_P = (8 * Real.sqrt 3, Real.pi / 3) :=
sorry

end cartesian_eq_of_curve_C_midpoint_polar_coordinates_l365_365689


namespace find_log_base_l365_365171

theorem find_log_base 
  (x : ℝ) 
  (h : log x 216 = -1 / 3) : 
  x = 216 := 
by 
  sorry

end find_log_base_l365_365171


namespace four_digit_numbers_l365_365662

theorem four_digit_numbers : 
  ((finset.range 10).erase 2.card * 2 + (finset.range 10).erase 2.card) = 27 :=
by {
  sorry
}

end four_digit_numbers_l365_365662


namespace mixed_number_arithmetic_l365_365138

theorem mixed_number_arithmetic :
  26 * (2 + 4 / 7 - (3 + 1 / 3)) + (3 + 1 / 5 + (2 + 3 / 7)) = -14 - 223 / 735 :=
by
  sorry

end mixed_number_arithmetic_l365_365138


namespace sum_f_eq_14_l365_365588

noncomputable def f (n : ℕ) : ℝ :=
if (Real.log n / Real.log 4).den = 1 then Real.log n / Real.log 4 else 0

theorem sum_f_eq_14 : (Finset.sum (Finset.range 256) (fun n => f n) = 14) :=
by
  sorry

end sum_f_eq_14_l365_365588


namespace mailman_should_give_junk_mail_l365_365043

-- Definitions from the conditions
def houses_in_block := 20
def junk_mail_per_house := 32

-- The mathematical equivalent proof problem statement in Lean 4
theorem mailman_should_give_junk_mail : 
  junk_mail_per_house * houses_in_block = 640 :=
  by sorry

end mailman_should_give_junk_mail_l365_365043


namespace will_net_calorie_intake_is_600_l365_365866

-- Given conditions translated into Lean definitions and assumptions
def breakfast_calories : ℕ := 900
def jogging_time_minutes : ℕ := 30
def calories_burned_per_minute : ℕ := 10

-- Proof statement in Lean
theorem will_net_calorie_intake_is_600 :
  breakfast_calories - (jogging_time_minutes * calories_burned_per_minute) = 600 :=
by
  sorry

end will_net_calorie_intake_is_600_l365_365866


namespace problem_1_problem_2_problem_3_l365_365594

section Problem

-- Initial conditions
variable (a : ℕ → ℝ) (t m : ℝ)
def a_1 : ℝ := 3
def a_n (n : ℕ) (h : 2 ≤ n) : ℝ := 2 * a (n - 1) + (t + 1) * 2^n + 3 * m + t

-- Problem 1:
theorem problem_1 (h : t = 0) (h' : m = 0) :
  ∃ d, ∀ n, 2 ≤ n → (a n / 2^n) = (a (n - 1) / 2^(n-1)) + d := sorry

-- Problem 2:
theorem problem_2 (h : t = -1) (h' : m = 4/3) :
  ∃ r, ∀ n, 2 ≤ n → a n + 3 = r * (a (n - 1) + 3) := sorry

-- Problem 3:
theorem problem_3 (h : t = 0) (h' : m = 1) :
  (∀ n, 1 ≤ n → a n = (n + 2) * 2^n - 3) ∧
  (∃ S : ℕ → ℝ, ∀ n, S n = (n + 1) * 2^(n + 1) - 2 - 3 * n) := sorry

end Problem

end problem_1_problem_2_problem_3_l365_365594


namespace range_of_first_term_l365_365219

-- Define the arithmetic sequence and its common difference.
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Define the sum of the first n terms of the sequence.
def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a + (n - 1) * d)) / 2

-- Prove the range of the first term a1 given the conditions.
theorem range_of_first_term (a d : ℤ) (S : ℕ → ℤ) (h1 : d = -2)
  (h2 : ∀ n, S n = sum_of_first_n_terms a d n)
  (h3 : S 7 = S 7)
  (h4 : ∀ n, n ≠ 7 → S n < S 7) :
  12 < a ∧ a < 14 :=
by
  sorry

end range_of_first_term_l365_365219


namespace thabo_hardcover_nonfiction_is_1450_l365_365784

noncomputable theory

def thabo_books_conditions 
  (total_books : ℕ) 
  (paperback_fiction_books paperback_nonfiction_books hardcover_nonfiction_books hardcover_fiction_books : ℕ) 
  (x : ℕ) :=
  total_books = 10_000 ∧
  paperback_nonfiction_books = hardcover_nonfiction_books + 100 ∧
  5 * x = paperback_fiction_books ∧
  3 * x = hardcover_fiction_books ∧
  hardcover_fiction_books = 12 * total_books / 100 ∧
  (hardcover_nonfiction_books + paperback_nonfiction_books) = 30 * total_books / 100

theorem thabo_hardcover_nonfiction_is_1450
  (H_nf : ℕ)
  (h : ∃ total_books paperback_fiction_books paperback_nonfiction_books hardcover_nonfiction_books hardcover_fiction_books x, 
    thabo_books_conditions total_books paperback_fiction_books paperback_nonfiction_books hardcover_nonfiction_books hardcover_fiction_books x ∧ 
    hardcover_nonfiction_books = H_nf) :
  H_nf = 1450 := 
sorry

end thabo_hardcover_nonfiction_is_1450_l365_365784


namespace sum_eq_9_div_64_l365_365546

noncomputable def double_sum : ℝ := ∑' (n : ℕ) in (set.Ici 2 : set ℕ), ∑' (k : ℕ) in set.Ico 1 n, (k : ℝ) / 3^(n + k)

theorem sum_eq_9_div_64 : double_sum = 9 / 64 := 
by 
sorry

end sum_eq_9_div_64_l365_365546


namespace pentagon_rectangle_ratio_l365_365902

theorem pentagon_rectangle_ratio (h_pentagon_perimeter : ∀ s : ℝ, 5 * s = 100)
                                 (h_rectangle_perimeter : ∀ w : ℝ, 2 * (2 * w + w) = 100) :
    (∃ s w : ℝ, h_pentagon_perimeter s ∧ h_rectangle_perimeter w ∧ s / w = 6 / 5) :=
begin
  sorry
end

end pentagon_rectangle_ratio_l365_365902


namespace sequence_not_periodic_thousandth_digit_one_ten_thousandth_one_at_position_21328_positions_formula_l365_365555

def sequence : ℕ → ℕ := λn, (if n.mod 2 = 0 then 0 else 1) 

theorem sequence_not_periodic : ∀ (n : ℕ), ¬ (∀ (m : ℕ), sequence (n + m) = sequence m) := 
sorry

theorem thousandth_digit_one : sequence 1000 = 1 := 
sorry

theorem ten_thousandth_one_at_position_21328 : ∃ (n : ℕ), sequence n = 1 ∧ (∑ i in list.range n, if sequence i = 1 then 1 else 0) = 10000 := 
sorry

theorem positions_formula : 
  ∀ (n : ℕ), 
    (sequence n = 1 → ∃ (k : ℕ), n = nat.floor ((2 + real.sqrt 2) * k)) ∧ 
    (sequence n = 0 → ∃ (m : ℕ), n = nat.floor (real.sqrt 2 * m)) := 
sorry

end sequence_not_periodic_thousandth_digit_one_ten_thousandth_one_at_position_21328_positions_formula_l365_365555


namespace rhombus_perimeter_l365_365792

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 24) (h2 : d2 = 10) : 4 * (Int.sqrt (d1/2 * d1/2 + d2/2 * d2/2)) = 52 :=
by {
  rw [h1, h2],
  have h_diag1 : (24 / 2 : ℤ) = 12 := by norm_num,
  have h_diag2 : (10 / 2 : ℤ) = 5 := by norm_num,
  have h_sq : (12 * 12 + 5 * 5 : ℤ) = 169 := by norm_num,
  have h_sqrt : Int.sqrt 169 = 13 := by norm_num [Int.sqrt_eq],
  rw [← h_sqrt, h_sq],
  norm_num
}

end rhombus_perimeter_l365_365792


namespace largest_integer_less_than_log3_sum_l365_365062

theorem largest_integer_less_than_log3_sum : 
  let log_sum := ∑ k in (range 1005).map (λ x, x + 2) \ (range 1005).map (λ x, x + 1)
  have hlog_sum : log_sum = log 1006 3,
  largest_integer_less_than_log3_sum < 7 :=
sorry

end largest_integer_less_than_log3_sum_l365_365062


namespace ratio_of_segments_l365_365513

open_locale classical 

theorem ratio_of_segments (squares : ℕ) (B P C N F : ℝ) (BPC_line : B + P + C = 3 * P)
(BNF_line : B + N + F = 3 * N) (left_area : ℝ) (right_area : ℝ) 
(h : left_area = 2 * right_area) :
  MN / NP = 1 / 5 :=
by 
  let a := 1 -- Define side length of each square as 'a'
  let side_length := a
  let AP := side_length -- Following the proportion derived in the problem
  let FC := 2.5 * side_length -- Following the calculations in the solution
  let NP := 1.25 * side_length -- NP derived previously as 1.25 * side_length
  let MN := 0.25 * side_length -- MN derived previously as 0.25 * side_length
  sorry

end ratio_of_segments_l365_365513


namespace angle_BMC_l365_365314

open EuclideanGeometry

noncomputable theory

def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def equilateral_conditions (A B C M D N : Point) : Prop :=
  equilateral_triangle A B C ∧
  dist M A ^ 2 = dist M B ^ 2 + dist M C ^ 2 ∧
  ∃ D, equilateral_triangle A C D ∧ D ≠ B ∧
  ∃ N, equilateral_triangle A M N

theorem angle_BMC (A B C M D N : Point) 
  (h : equilateral_conditions A B C M D N) : 
  ∠ (B -ᵥ M) (C -ᵥ M) = 150 :=
sorry

end angle_BMC_l365_365314


namespace prop_p_l365_365595

open Real

noncomputable def f (x : ℝ) : ℝ := -x + sin x

theorem prop_p : ∀ x ∈ Ioo (0 : ℝ) (π / 2), f x < 0 :=
by
  intro x hx
  sorry

end prop_p_l365_365595


namespace hexagon_area_correct_l365_365952

noncomputable def area_of_hexagon : ℝ :=
let side_length := 2 in
let height := side_length * (Real.sqrt 3) / 2 in
let larger_triangle_side := side_length + 2 * height in
let larger_triangle_area := (Real.sqrt 3) / 4 * larger_triangle_side^2 in
let smaller_triangle_area := (Real.sqrt 3 / 4) * side_length^2 in
let hexagon_area := larger_triangle_area - 3 * smaller_triangle_area in
hexagon_area

theorem hexagon_area_correct :
  area_of_hexagon = 4 * Real.sqrt 3 - 3 :=
sorry

end hexagon_area_correct_l365_365952


namespace cyclic_quadrilateral_equivalence_l365_365721

-- Define the cyclic quadrilateral with mechanisms to check angles
variables {A B C D X Y : Type*}
variable [euclidean_geometry A B C D X Y]


-- The problem statement as a Lean 4 theorem
theorem cyclic_quadrilateral_equivalence
  (h1 : cyclic_quadrilateral A B C D)
  (hx1 : X ∈ segment B D)
  (hx2 : angle BAC = angle XAD)
  (hx3 : angle BCA = angle XCD)
  (hy1 : Y ∈ segment A C)
  (hy2 : angle CBD = angle YBA)
  (hy3 : angle CDB = angle YDA) :
  (∃ X, X ∈ segment B D ∧ angle BAC = angle XAD ∧ angle BCA = angle XCD) ↔
  (∃ Y, Y ∈ segment A C ∧ angle CBD = angle YBA ∧ angle CDB = angle YDA) :=
sorry  -- Proof is not required, hence omitted

end cyclic_quadrilateral_equivalence_l365_365721


namespace dracula_story_inconsistencies_l365_365698

structure DraculaStory :=
  (use_of_yes : Bool)
  (higher_nobility_representation : Bool)
  (magical_assertion : Bool)

def identify_inconsistencies (story : DraculaStory) : Bool :=
  story.use_of_yes &&
  story.higher_nobility_representation &&
  story.magical_assertion

theorem dracula_story_inconsistencies (story : DraculaStory) (h1 : story.use_of_yes = true) (h2 : story.higher_nobility_representation = false) (h3 : story.magical_assertion = false) : identify_inconsistencies story = true :=
by
  rw [identify_inconsistencies, h1, h2, h3]
  simp
  rfl

end dracula_story_inconsistencies_l365_365698


namespace Frank_mowing_lawns_l365_365988

theorem Frank_mowing_lawns (M : ℕ) :
  let money_weed_eating := 58
  let weekly_spending := 7
  let weeks := 9
  let total_money := weeks * weekly_spending
  M + money_weed_eating = total_money → M = 5 :=
by
  intros money_weed_eating weekly_spending weeks total_money h
  have h1 : money_weed_eating = 58 := rfl
  have h2 : weekly_spending = 7 := rfl
  have h3 : weeks = 9 := rfl
  have h4 : total_money = 63 := by calc
    total_money = weeks * weekly_spending : rfl
              ... = 9 * 7                 : by rw [h3, h2]
              ... = 63                    : rfl
  rw [h4] at h
  rw [h1] at h
  sorry

end Frank_mowing_lawns_l365_365988


namespace find_r2_l365_365154

noncomputable def r2 : ℚ :=
  let r1 := (1 : ℚ) / 729
  let x := (1 : ℚ) / 3
  let q1 := x^5 + (1/3)*x^4 + (1/9)*x^3 + (1/27)*x^2 + (1/81)*x + 1/243
  (q1.evaluate x : ℚ)

theorem find_r2 :
  (let r1 := (1 : ℚ) / 729
   let x := (1 : ℚ) / 3
   let q1 := x^5 + (1/3)*x^4 + (1/9)*x^3 + (1/27)*x^2 + (1/81)*x + 1/243
   (q1.evaluate x : ℚ)) = 2 / 81 :=
sorry

end find_r2_l365_365154


namespace number_of_girls_in_basketball_club_l365_365472

-- Define the number of members in the basketball club
def total_members : ℕ := 30

-- Define the number of members who attended the practice session
def attended : ℕ := 18

-- Define the unknowns: number of boys (B) and number of girls (G)
variables (B G : ℕ)

-- Define the conditions provided in the problem
def condition1 : Prop := B + G = total_members
def condition2 : Prop := B + (1 / 3) * G = attended

-- Define the theorem to prove
theorem number_of_girls_in_basketball_club (B G : ℕ) (h1 : condition1 B G) (h2 : condition2 B G) : G = 18 :=
sorry

end number_of_girls_in_basketball_club_l365_365472


namespace mary_fruits_left_l365_365344

theorem mary_fruits_left (apples_initial oranges_initial blueberries_initial : ℕ)
  (apples_eaten oranges_eaten blueberries_eaten : ℕ) :
  apples_initial = 14 →
  oranges_initial = 9 →
  blueberries_initial = 6 →
  apples_eaten = 1 →
  oranges_eaten = 1 →
  blueberries_eaten = 1 →
  (apples_initial - apples_eaten) + (oranges_initial - oranges_eaten) + (blueberries_initial - blueberries_eaten) = 26 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end mary_fruits_left_l365_365344


namespace odd_function_solution_set_l365_365627

noncomputable def f : ℝ → ℝ
| x := if x ≤ 0 then x^2 + 4 * x else - x^2 + 4 * x

theorem odd_function (x : ℝ) : f (-x) = -f x :=
by
  unfold f
  split_ifs
  { ring }
  { ring }

theorem solution_set : {x : ℝ | f x > 3} = {x : ℝ | 1 < x ∧ x < 3} ∪ {x : ℝ | x < -2 - real.sqrt 7} :=
by
  sorry

end odd_function_solution_set_l365_365627


namespace jerry_total_income_l365_365713

/-- Jerry charges $20 to pierce someone's nose and 50% more to pierce their ears. 
    If he pierces 6 noses and 9 ears, prove that he makes $390. -/
theorem jerry_total_income :
  let nose_piercing_cost := 20 in
  let ear_piercing_cost := nose_piercing_cost + 0.5 * nose_piercing_cost in
  let total_noses := 6 in
  let total_ears := 9 in
  let total_income := (nose_piercing_cost * total_noses) + (ear_piercing_cost * total_ears) in
  total_income = 390 :=
by 
  sorry

end jerry_total_income_l365_365713


namespace unique_triple_solution_l365_365963

theorem unique_triple_solution (x y z : ℝ) :
  (1 + x^4 ≤ 2 * (y - z)^2) →
  (1 + y^4 ≤ 2 * (z - x)^2) →
  (1 + z^4 ≤ 2 * (x - y)^2) →
  (x = 1 ∧ y = 0 ∧ z = -1) :=
sorry

end unique_triple_solution_l365_365963


namespace perimeter_of_rhombus_l365_365789

def rhombus_perimeter (d1 d2 : ℝ) : ℝ :=
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2))

theorem perimeter_of_rhombus (h1 : ∀ (d1 d2 : ℝ), d1 = 24 ∧ d2 = 10) : rhombus_perimeter 24 10 = 52 := 
by
  sorry

end perimeter_of_rhombus_l365_365789


namespace sum_floor_is_1650_l365_365647

-- Define the function satisfying the condition
def satisfies_condition (f : ℝ → ℝ) : Prop := 
  ∀ x y z : ℝ, (1 / 3) * f (x * y) + (1 / 3) * f (x * z) - f x * f (y * z) ≥ 1 / 9

noncomputable def f_value (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f x = 1 / 3)

theorem sum_floor_is_1650 (f : ℝ → ℝ) (h_condition : satisfies_condition f) (h_value : f_value f) :
  ∑ i in Finset.range 100 (λ i, Nat.floor ((i + 1) * f (i + 1))) = 1650 :=
sorry

end sum_floor_is_1650_l365_365647


namespace shorter_side_of_quilt_l365_365837

theorem shorter_side_of_quilt :
  ∀ (x : ℕ), (∃ y : ℕ, 24 * y = 144) -> x = 6 :=
by
  intros x h
  sorry

end shorter_side_of_quilt_l365_365837


namespace sum_of_parts_l365_365096

variable (x y : ℤ)
variable (h1 : x + y = 60)
variable (h2 : y = 45)

theorem sum_of_parts : 10 * x + 22 * y = 1140 :=
by
  sorry

end sum_of_parts_l365_365096


namespace nested_sum_equals_fraction_l365_365527

noncomputable def nested_sum : ℝ := ∑' (n : ℕ) in Ico 2 (⊤), ∑' (k : ℕ) in Ico 1 n, (k : ℝ) / 3^(n + k)

theorem nested_sum_equals_fraction :
  nested_sum = 3 / 64 := sorry

end nested_sum_equals_fraction_l365_365527


namespace number_of_n_le_100_with_f50_eq_16_l365_365980

-- Define the functions and conditions given in the problem
def number_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

def f1 (n : ℕ) : ℕ :=
  2 * number_of_divisors n

def fj : ℕ → ℕ → ℕ
| 0 m := m
| (n+1) m := f1 (fj n m)

-- Define the main statement that needs to be proved
theorem number_of_n_le_100_with_f50_eq_16 : 
  (Finset.range 101).filter (λ n, fj 50 n = 16).card = 3 :=
sorry

end number_of_n_le_100_with_f50_eq_16_l365_365980


namespace lowest_score_zero_l365_365384

theorem lowest_score_zero (scores : List ℝ)
  (h_len : scores.length = 15)
  (h_mean : scores.sum / 15 = 85)
  (h_highest : scores.maximum = 105)
  (h_new_mean : (scores.erase 105).erase (scores.minimum).sum / 13 = 90) :
  scores.minimum = 0 := 
sorry

end lowest_score_zero_l365_365384


namespace unique_triangle_exists_l365_365153

-- Given conditions of the triangle
variables (BC AC e f γ : ℝ)
variables (C1 A B C : Point) -- Points in the triangle
variables (median_CC1 : median C C1)

-- Mathematical equivalences
axiom sum_eq_e : BC + AC = e
axiom diff_eq_f : BC - AC = f
axiom angle_eq_gamma : angle B C A = γ

-- Our goal is to prove existence and uniqueness of the triangle
theorem unique_triangle_exists (h1 : BC ≥ AC) (h2 : median C A B C1):
  ∃! (triangle : Triangle), 
  (triangle.BC = BC)
  ∧ (triangle.AC = AC)
  ∧ (triangle.angle = γ)
  ∧ (triangle.median C CC1) :=
sorry

end unique_triangle_exists_l365_365153


namespace sum_f_2_to_2n_l365_365327

noncomputable def f (x : ℝ) := 2 * x + 1

theorem sum_f_2_to_2n (n : ℕ) :
  (∑ i in finset.range n, f (2 * (i + 1))) = 2 * n^2 + 3 * n := 
  sorry

end sum_f_2_to_2n_l365_365327


namespace prism_perimeter_is_26_l365_365906

noncomputable def equilateral_prism_midpoints_perimeter 
    (height : ℝ) (sidelength : ℝ) (X Y Z : ℝ × ℝ) 
    (hX : X = (sidelength/2, 0)) 
    (hY : Y = (sidelength, -sidelength * (√3)/2))
    (hZ : Z = (sidelength/2, -height/2)) : ℝ :=
  ∥X - Z∥ + ∥Z - Y∥ + ∥Y - X∥

theorem prism_perimeter_is_26 
    (height : ℝ) (sidelength : ℝ) 
    (h_height : height = 16) 
    (h_sidelength : sidelength = 12) : 
  equilateral_prism_midpoints_perimeter height sidelength (sidelength/2, 0) (sidelength, -sidelength * (√3)/2) (sidelength/2, -height/2) = 26 :=
by
  rw [h_height, h_sidelength]
  sorry

end prism_perimeter_is_26_l365_365906


namespace pool_volume_diameter_20_depth_6_to_3_l365_365563

theorem pool_volume_diameter_20_depth_6_to_3 :
  let diameter := 20
  let max_depth := 6
  let min_depth := 3
  let radius := diameter / 2
  let avg_depth := (max_depth + min_depth) / 2
  let volume := Real.pi * radius^2 * avg_depth
  volume = 450 * Real.pi := by
  let diameter := 20
  let max_depth := 6
  let min_depth := 3
  let radius := diameter / 2
  let avg_depth := (max_depth + min_depth) / 2
  let volume := Real.pi * radius^2 * avg_depth
  have radius_eq : radius = 10 := by
    unfold radius diameter
    norm_num
  have avg_depth_eq : avg_depth = 4.5 := by
    unfold avg_depth max_depth min_depth
    norm_num
  have volume_eq : volume = 450 * π := by
    unfold volume
    rw [radius_eq, avg_depth_eq]
    norm_num
    rw mul_div_cancel (10:ℝ)

  exact volume_eq

end pool_volume_diameter_20_depth_6_to_3_l365_365563


namespace total_upwards_distance_l365_365132

-- Define the coordinates of Annie, Barbara, and Charlie
structure Coordinates where
  x : ℝ
  y : ℝ

-- Define the given locations
def Annie : Coordinates := { x := 8, y := -15 }
def Barbara : Coordinates := { x := 2, y := 10 }
def Charlie : Coordinates := { x := 5, y := 3 }

-- Define the midpoint calculation
def midpoint (A B : Coordinates) : Coordinates :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

-- Define the vertical distance calculation
def vertical_distance (A B : Coordinates) : ℝ :=
  B.y - A.y

-- Proof statement
theorem total_upwards_distance :
  vertical_distance (midpoint Annie Barbara) Charlie = 5.5 :=
by
  -- This 'sorry' is a placeholder for the actual proof
  sorry

end total_upwards_distance_l365_365132


namespace eleven_billion_in_scientific_notation_l365_365048

-- Definition: "Billion" is 10^9
def billion : ℝ := 10^9

-- Theorem: 11 billion can be represented as 1.1 * 10^10
theorem eleven_billion_in_scientific_notation : 11 * billion = 1.1 * 10^10 := by
  sorry

end eleven_billion_in_scientific_notation_l365_365048


namespace smallest_x_for_max_f_l365_365151

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 12)

theorem smallest_x_for_max_f : ∃ x > 0, f x = 2 ∧ ∀ y > 0, (f y = 2 → y ≥ x) :=
sorry

end smallest_x_for_max_f_l365_365151


namespace cube_edge_skew_probability_l365_365600

theorem cube_edge_skew_probability : 
  let edges := 12 
  let remaining_edges := edges - 1
  let parallel_edges := 3
  let perpendicular_edges := 4
  let skew_edges := 4
  let probability := (skew_edges : ℚ) / (remaining_edges : ℚ)
  in probability = 4 / 11 := 
by 
  sorry

end cube_edge_skew_probability_l365_365600


namespace area_union_square_circle_l365_365498

theorem area_union_square_circle :
  let side_length_square := 8
  let radius_circle := 12
  let area_square := side_length_square ^ 2
  let area_circle := Real.pi * (radius_circle ^ 2)
  let overlap_area := (1 / 4) * area_circle
  let union_area := area_square + area_circle - overlap_area
  union_area = 64 + 108 * Real.pi := by
begin
  sorry
end

end area_union_square_circle_l365_365498


namespace constant_term_in_expansion_l365_365674

theorem constant_term_in_expansion (n : ℕ) (hn : n = 10) :
  let term := λ r : ℕ, binomial n r * (-2)^r * x^((n - r) / 2 - 2 * r)
  in term 5 = (480 : ℕ)  :=
by
  sorry

end constant_term_in_expansion_l365_365674


namespace tetrahedron_area_theorem_l365_365382

noncomputable def tetrahedron_faces_areas_and_angles
  (a b c d : ℝ) (α β γ : ℝ) : Prop :=
  d^2 = a^2 + b^2 + c^2 - 2 * a * b * Real.cos γ - 2 * b * c * Real.cos α - 2 * c * a * Real.cos β

theorem tetrahedron_area_theorem
  (a b c d : ℝ) (α β γ : ℝ) :
  tetrahedron_faces_areas_and_angles a b c d α β γ :=
sorry

end tetrahedron_area_theorem_l365_365382


namespace point_on_x_axis_coordinates_l365_365672

theorem point_on_x_axis_coordinates (a : ℝ) (P : ℝ × ℝ) (h : P = (a - 1, a + 2)) (hx : P.2 = 0) : P = (-3, 0) :=
by
  -- Replace this with the full proof
  sorry

end point_on_x_axis_coordinates_l365_365672


namespace geometric_construction_l365_365999

open Set

structure Point := (x : ℝ) (y : ℝ)

structure Line := (a b c : ℝ)

def is_on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

def on_opposite_sides (A B : Point) (l : Line) : Prop :=
  (l.a * A.x + l.b * A.y + l.c) * (l.a * B.x + l.b * B.y + l.c) < 0

def tangent_point (A : Point) (l : Line) : Point := sorry -- Point of tangency K

def circle_center (A : Point) (radius : ℝ) (l : Line) : Point := sorry -- Circle center at A with given radius

def tangent_from_point (B : Point) (circle_center : Point) : Point := sorry -- Tangent from B to the circle with center A touches at N

def intersection_with_line (from : Point) (to : Point) (l : Line) : Point := sorry -- Intersection point of BN with line l

def angle (P Q R : Point) : ℝ := sorry -- Angle PQS in degrees or radians; to be defined

theorem geometric_construction (l : Line) (A B : Point)
(h1 : on_opposite_sides A B l) : 
∃ M : Point,
  let K := tangent_point A l in
  let N := tangent_from_point B (circle_center A (sqrt (l.a ^ 2 + l.b ^ 2)) l) in
  M = intersection_with_line B N l ∧
  (angle A M K) = (1 / 2) * (angle B M K):=
sorry


end geometric_construction_l365_365999


namespace expression_positive_intervals_l365_365560

theorem expression_positive_intervals {x : ℝ} :
  (x + 2) * (x - 2) * (x + 1) > 0 ↔ (x ∈ Ioo (-2 : ℝ) (-1) ∨ x ∈ Ioi 2) :=
by
  sorry

end expression_positive_intervals_l365_365560


namespace num_real_values_x_l365_365987

theorem num_real_values_x : 
  (∃ x : ℝ, abs (1 - ↑x * complex.I) = 2) ∧ 
  (∃ x₁ x₂ : ℝ, abs (1 - ↑x₁ * complex.I) = 2 ∧ abs (1 - ↑x₂ * complex.I) = 2 ∧ x₁ ≠ x₂) :=
  sorry

end num_real_values_x_l365_365987


namespace area_of_union_of_square_and_circle_l365_365499

-- Definitions based on problem conditions
def side_length : ℕ := 8
def radius : ℝ := 12
def square_area := side_length ^ 2
def circle_area := Real.pi * radius ^ 2
def overlapping_area := 1 / 4 * circle_area
def union_area := square_area + circle_area - overlapping_area

-- The problem statement in Lean format
theorem area_of_union_of_square_and_circle :
  union_area = 64 + 108 * Real.pi :=
by
  sorry

end area_of_union_of_square_and_circle_l365_365499


namespace determinant_of_triangle_angles_l365_365736

theorem determinant_of_triangle_angles (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Matrix.det ![
    ![Real.tan α, Real.sin α * Real.cos α, 1],
    ![Real.tan β, Real.sin β * Real.cos β, 1],
    ![Real.tan γ, Real.sin γ * Real.cos γ, 1]
  ] = 0 :=
by
  -- Proof statement goes here
  sorry

end determinant_of_triangle_angles_l365_365736


namespace max_value_of_expression_l365_365379

noncomputable def maximum_value {x y : ℝ} (hx : 0 < x) (hy : 0 < y) 
  (h : x^2 - x*y + 2*y^2 = 8) : ℝ :=
  x^2 + x*y + 2*y^2

theorem max_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x^2 - x*y + 2*y^2 = 8) : maximum_value hx hy h = (72 + 32 * Real.sqrt 2) / 7 :=
by
  sorry

end max_value_of_expression_l365_365379


namespace probability_of_divisibility_by_15_is_zero_l365_365429

def digits : List ℕ := [1, 2, 3, 7, 7, 9]

def sum_digits (lst : List ℕ) : ℕ := lst.sum

def is_divisible_by (n : ℕ) (d : ℕ) : Prop := d ∣ n

theorem probability_of_divisibility_by_15_is_zero :
  (sum_digits digits % 3 ≠ 0) →
  (∀ perm, perm ∈ List.permutations digits → is_divisible_by (nat.of_digits 10 perm) 15 = false) :=
by
  sorry

end probability_of_divisibility_by_15_is_zero_l365_365429


namespace sq_of_gt_and_pos_l365_365195

-- Define the variables with their respective conditions
variables (a b : ℝ) 
variable (h : a > b > 0)

-- State the theorem
theorem sq_of_gt_and_pos (h : a > b ∧ b > 0) : a^2 > b^2 := by
  sorry

end sq_of_gt_and_pos_l365_365195


namespace radius_length_l365_365678

theorem radius_length {r : ℝ} (h : r = 2) : ∀ (s : ℝ), s = r → s = 2 :=
by
  intros s hs
  rw hs
  exact h

end radius_length_l365_365678


namespace bobs_improvement_percentage_l365_365456

-- Define the conditions
def bobs_time_minutes := 10
def bobs_time_seconds := 40
def sisters_time_minutes := 10
def sisters_time_seconds := 8

-- Convert minutes and seconds to total seconds
def bobs_total_time_seconds := bobs_time_minutes * 60 + bobs_time_seconds
def sisters_total_time_seconds := sisters_time_minutes * 60 + sisters_time_seconds

-- Define the improvement needed and calculate the percentage improvement
def improvement_needed := bobs_total_time_seconds - sisters_total_time_seconds
def percentage_improvement := (improvement_needed / bobs_total_time_seconds) * 100

-- The lean statement to prove
theorem bobs_improvement_percentage : percentage_improvement = 5 := by
  sorry

end bobs_improvement_percentage_l365_365456


namespace ratio_a_c_l365_365401

variable (a b c d : ℕ)

/-- The given conditions -/
axiom ratio_a_b : a / b = 5 / 2
axiom ratio_c_d : c / d = 4 / 1
axiom ratio_d_b : d / b = 1 / 3

/-- The proof problem -/
theorem ratio_a_c : a / c = 15 / 8 := by
  sorry

end ratio_a_c_l365_365401


namespace proof_correct_statements_l365_365641

-- Definitions
def f (x : ℝ) := (1/2) * Real.sin (2 * x - (Real.pi / 6))

-- Hypotheses
def critical_points (x : ℝ) := ∃ k : ℤ, x = Real.pi / 3 + k * Real.pi / 2
def graph_shift (x : ℝ) := f x = (1/2) * Real.sin (2 * (x - Real.pi / 12))

theorem proof_correct_statements : (∀ x, critical_points x) ∧ (∀ x, graph_shift x) := 
by
  sorry

end proof_correct_statements_l365_365641


namespace volume_ratio_l365_365243

def volume_of_cube (side_length : ℕ) : ℕ :=
  side_length * side_length * side_length

theorem volume_ratio 
  (hyungjin_side_length_cm : ℕ)
  (kyujun_side_length_m : ℕ)
  (h1 : hyungjin_side_length_cm = 100)
  (h2 : kyujun_side_length_m = 2) :
  volume_of_cube (kyujun_side_length_m * 100) = 8 * volume_of_cube hyungjin_side_length_cm :=
by
  sorry

end volume_ratio_l365_365243


namespace lucys_mother_age_twice_her_age_in_2040_l365_365351

theorem lucys_mother_age_twice_her_age_in_2040:
  (∀ (lucy_age_2010 mother_age_2010 : ℕ), 
    lucy_age_2010 = 10 →
    mother_age_2010 = 5 * lucy_age_2010 →
    year = 2040 →
    mother_age_2040 = 2 * lucy_age_2040) :=
by
  intros lucy_age_2010 mother_age_2010 h1 h2 year mother_age_2040 lucy_age_2040
  have h3 : lucy_age_2010 + 30 = 40, from sorry
  have h4 : mother_age_2010 + 30 = 80, from sorry
  sorry

end lucys_mother_age_twice_her_age_in_2040_l365_365351


namespace present_age_ratio_l365_365760

theorem present_age_ratio (D J : ℕ) (h1 : Dan = 24) (h2 : James = 20) : Dan / James = 6 / 5 := by
  sorry

end present_age_ratio_l365_365760


namespace total_items_proof_l365_365267

noncomputable def totalItemsBought (budget : ℕ) (sandwichCost : ℕ) 
  (pastryCost : ℕ) (maxSandwiches : ℕ) : ℕ :=
  let s := min (budget / sandwichCost) maxSandwiches
  let remainingMoney := budget - s * sandwichCost
  let p := remainingMoney / pastryCost
  s + p

theorem total_items_proof : totalItemsBought 50 6 2 7 = 11 := by
  sorry

end total_items_proof_l365_365267


namespace problem_I_problem_II_problem_III_l365_365644

noncomputable def f (x a b : ℝ) : ℝ := Real.exp x - (1/2) * b * x^2 + a * x

theorem problem_I (a : ℝ) (ha : a > -1) : ∀ x : ℝ, f x a 1 = deriv (f x a 1) x ≥ 0 :=
by sorry

theorem problem_II (a : ℝ) (ha : a < 1 - Real.exp 1) : ∃ x ∈ Icc 1 +∞, f x a 1 < 1/2 :=
by sorry

theorem problem_III (h : ∀ x : ℝ, deriv (f x 1 1) x ≥ 0) : ∀ a b, f (a*b) a b = Real.exp (a*b) - a*b - (1/2) * b * (a*b)^2 ≥ -Real.exp (1) :=
by sorry

end problem_I_problem_II_problem_III_l365_365644


namespace power_function_value_l365_365021

theorem power_function_value (f : ℝ → ℝ) 
  (h1 : ∃ n : ℝ, ∀ x : ℝ, f(x) = x^n) 
  (h2 : f(4) = 1/2) : 
  f(1/4) = 2 :=
sorry

end power_function_value_l365_365021


namespace prove_necessary_but_not_sufficient_l365_365067

noncomputable def necessary_but_not_sufficient_condition (m : ℝ) :=
  (∀ x : ℝ, x^2 + 2*x + m > 0) → (m > 0) ∧ ¬ (∀ x : ℝ, x^2 + 2*x + m > 0 → m <= 1)

theorem prove_necessary_but_not_sufficient
    (m : ℝ) :
    necessary_but_not_sufficient_condition m :=
by
  sorry

end prove_necessary_but_not_sufficient_l365_365067


namespace max_value_a_l365_365892

noncomputable theory

-- Definitions for the problem conditions
def is_lattice_point (x y : ℤ) : Prop :=
  ∃ m : ℚ, y = m * x + 3

def avoids_lattice_points (m : ℚ) : Prop :=
  ∀ (x : ℤ), 0 < x ∧ x ≤ 200 → ¬ is_lattice_point x ((m * (x : ℚ)) + 3)

-- The main theorem statement
theorem max_value_a (a : ℚ) : (1 / 3 : ℚ) < (68 / 200) ∧ (68 / 200) < a ∧ (∀ m : ℚ, (1 / 3 : ℚ) < m ∧ m < a → avoids_lattice_points m) :=
by sorry

end max_value_a_l365_365892


namespace jerry_hose_pumping_ratio_l365_365303

theorem jerry_hose_pumping_ratio :
  let pond_capacity := 200 -- in gallons
  let normal_pump_speed := 6 -- in gallons per minute
  let time_to_fill_pond := 50 -- in minutes
  let current_pump_speed := pond_capacity / time_to_fill_pond
  (current_pump_speed : ℚ) / normal_pump_speed = 2 / 3 :=
by
  let pond_capacity := 200
  let normal_pump_speed := 6
  let time_to_fill_pond := 50
  let current_pump_speed := pond_capacity / time_to_fill_pond
  have current_pump_speed_2 : current_pump_speed = 4 := by sorry
  show (current_pump_speed : ℚ) / normal_pump_speed = 2 / 3 from sorry

end jerry_hose_pumping_ratio_l365_365303


namespace additional_savings_zero_l365_365502

noncomputable def windows_savings (purchase_price : ℕ) (free_windows : ℕ) (paid_windows : ℕ)
  (dave_needs : ℕ) (doug_needs : ℕ) : ℕ := sorry

theorem additional_savings_zero :
  windows_savings 100 2 5 12 10 = 0 := sorry

end additional_savings_zero_l365_365502


namespace ratio_wy_l365_365034

-- Define the variables and conditions
variables (w x y z : ℚ)
def ratio_wx := w / x = 5 / 4
def ratio_yz := y / z = 7 / 5
def ratio_zx := z / x = 1 / 8

-- Statement to prove
theorem ratio_wy (hwx : ratio_wx w x) (hyz : ratio_yz y z) (hzx : ratio_zx z x) : w / y = 25 / 7 :=
by
  sorry  -- Proof not needed

end ratio_wy_l365_365034


namespace decreased_and_divided_l365_365262

theorem decreased_and_divided (x : ℝ) (h : (x - 5) / 7 = 7) : (x - 14) / 10 = 4 := by
  sorry

end decreased_and_divided_l365_365262


namespace tangent_line_y_intercept_l365_365186

theorem tangent_line_y_intercept : 
  (let P := (1:ℝ, 2:ℝ) in 
   let f := λ x : ℝ, x^3 + 2 * x - 1 in 
   let y' := λ (x : ℝ), 3 * x^2 + 2 in
   let k := y' (fst P) in 
   let tangent_line := λ x : ℝ, (2:ℝ) + k * (x - (1:ℝ)) in
   tangent_line 0 = (-3 : ℝ)) := 
begin
  sorry
end

end tangent_line_y_intercept_l365_365186


namespace division_problem_l365_365254

theorem division_problem : (2994 / 14.5 = 173) :=
by
  have h : 29.94 / 1.45 = 17.3 := by sorry
  calc
    2994 / 14.5 = (29.94 * 100) / (1.45 * 10) : by sorry
              ... = (29.94 / 1.45) * (100 / 10) : by sorry
              ... = 17.3 * 10 : by sorry
              ... = 173 : by sorry

end division_problem_l365_365254


namespace min_distance_vertex_to_path_l365_365120

theorem min_distance_vertex_to_path {r l : ℝ} (P : ℝ) :
  r = 1 →
  l = 3 →
  P ∈ set.Icc 0 (2 * Real.pi) →
  ∃ (d : ℝ), d = (3 * Real.sqrt 3) / 2 :=
by
  intros hr hl hP
  use (3 * Real.sqrt 3) / 2
  sorry

end min_distance_vertex_to_path_l365_365120


namespace count_of_integer_values_not_satisfying_inequality_l365_365589

theorem count_of_integer_values_not_satisfying_inequality :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, (3 * x^2 + 11 * x + 10 ≤ 17) ↔ (x = -7 ∨ x = -6 ∨ x = -5 ∨ x = -4 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0) :=
by sorry

end count_of_integer_values_not_satisfying_inequality_l365_365589


namespace quadrilateral_divisible_into_5_equal_triangles_l365_365081

theorem quadrilateral_divisible_into_5_equal_triangles :
  ∃ (quadrilateral : Type) (divide : quadrilateral → list triangle), 
    quadrilateral ≠ ∅ ∧ length (divide quadrilateral) = 5 ∧ 
    ∀ (t1 t2 : triangle), t1 ∈ (divide quadrilateral) → t2 ∈ (divide quadrilateral) → area t1 = area t2 :=
sorry

end quadrilateral_divisible_into_5_equal_triangles_l365_365081


namespace cosine_identity_l365_365993

theorem cosine_identity (α : ℝ) (h : sin (α + π / 6) = 1 / 3) : cos (2 * π / 3 - 2 * α) = -7 / 9 := 
  sorry

end cosine_identity_l365_365993


namespace g_2_equals_5_l365_365330

-- Definition of the function g and the condition it satisfies
def g (x : ℝ) : ℝ := sorry

-- Given condition
axiom condition_g (x y : ℝ) : g(x) * g(y) - g(x * y) = x^2 + y^2

-- Theorem stating the question and the correct answer
theorem g_2_equals_5 : g 2 = 5 :=
by
  -- Proof steps yet to be filled
  sorry

end g_2_equals_5_l365_365330


namespace find_marias_salary_l365_365750

variable (S : ℝ) -- Maria's monthly salary
variable (tax_rate : ℝ := 0.20)  -- 20% tax rate
variable (insurance_rate : ℝ := 0.05)  -- 5% insurance rate
variable (utility_ratio : ℝ := 1 / 4.0)  -- A quarter ratio for utility bills
variable (remaining_amount : ℝ := 1125)  -- Amount Maria has after deductions and utility bill payments

theorem find_marias_salary (S : ℝ) (tax_rate : ℝ := 0.20) (insurance_rate : ℝ := 0.05) 
    (utility_ratio : ℝ := 1 / 4) (remaining_amount : ℝ := 1125) :
  let tax = tax_rate * S in
  let insurance = insurance_rate * S in
  let total_deductions = tax + insurance in
  let amount_after_deductions = S - total_deductions in
  let utility_bills = utility_ratio * amount_after_deductions in
  let final_amount = amount_after_deductions - utility_bills in
  final_amount = remaining_amount → S = 2000 :=
by
  sorry

end find_marias_salary_l365_365750


namespace length_BE_is_sqrt4_3_l365_365692

-- Definitions of conditions
structure Rhombus (ABCD : Type) :=
(side_length : ℝ)
(diagonal1 : ℝ)
(diagonal2 : ℝ)
(area : ℝ)

structure Rectangle (EBCF : Type) :=
(side1 : ℝ)
(side2 : ℝ)

-- Given conditions
def rhombus : Rhombus ℝ :=
{ side_length := 2,
  diagonal1 := 2 * Real.sqrt 3,
  diagonal2 := 2,
  area := 2 * Real.sqrt 3 }

def congruent_rectangles (EBCF JKHG : Rectangle ℝ) := 
  EBCF.side1 = JKHG.side1 ∧ EBCF.side2 = JKHG.side2

def rectangle_EBCF : Rectangle ℝ :=
{ side1 := Real.sqrt (Real.sqrt 3),
  side2 := Real.sqrt (Real.sqrt 3) }

noncomputable def length_BE := rectangle_EBCF.side1

-- Final proof statement
theorem length_BE_is_sqrt4_3 : length_BE = Real.sqrt (Real.sqrt 3) :=
sorry

end length_BE_is_sqrt4_3_l365_365692


namespace count_valid_n_l365_365188

def is_valid_n (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 200 ∧ gcd 21 n = 3

theorem count_valid_n : (finset.filter is_valid_n (finset.range 201)).card = 57 := 
by
  sorry

end count_valid_n_l365_365188


namespace collinear_M_N_T_l365_365426

open_locale classical

noncomputable def midpoint (A B : Point) : Point := 
{ x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2 }

variables {O1 O2 : Circle}
variables {P Q A B C D M N T : Point}

axiom intersect_points (O1 O2 : Circle) : ∃ (P Q : Point), 
  P ∈ O1 ∧ P ∈ O2 ∧ Q ∈ O1 ∧ Q ∈ O2 ∧ P ≠ Q

axiom eq_radii (O1 O2 : Circle) : O1.radius = O2.radius
axiom midpoint_T : midpoint P Q = T
axiom secants_through_P (P : Point) (A B C D : Point) : 
  incidence (P, A) ∧ incidence (P, B) ∧ incidence (P, C) ∧ incidence (P, D)

axiom points_on_circles (A : Point) (B : Point) (C : Point) (D : Point) :
  A ∈ circle O1 ∧ C ∈ circle O1 ∧ B ∈ circle O2 ∧ D ∈ circle O2

axiom midpoints (M N : Point) (A D B C : Point) : 
  midpoint A D = M ∧ midpoint B C = N

axiom centers_not_in_common_region (O1 O2 : Point) : 
  O1 ∉ region (circle O1 ∩ circle O2) ∧ O2 ∉ region (circle O1 ∩ circle O2)

theorem collinear_M_N_T (O1 O2 : Circle) (P Q A B C D M N T : Point)
  (h_intersect : intersect_points O1 O2)
  (h_equal_radii : eq_radii O1 O2)
  (h_midT : midpoint_T)
  (h_secants : secants_through_P P A B C D)
  (h_on_circles : points_on_circles A B C D)
  (h_midpoints : midpoints M N A D B C) :
  collinear M N T :=
sorry

end collinear_M_N_T_l365_365426


namespace math_problem_l365_365232

def g (x : ℝ) : ℝ := (4^x - n) / (2^x)
def f (x : ℝ) : ℝ := log 4 (4^x + 1) + m * x
def h (x : ℝ) : ℝ := f x + (1 / 2) * x

theorem math_problem 
  {n m : ℝ}
  (h_odd_g : ∀ x : ℝ, g (-x) = -g x)
  (h_even_f : ∀ x : ℝ, f (-x) = f x) :
  (m + n = -1/2) ∧ ( ∀ x ≥ 1, g x > h (log 4 (2 * a + 1)) → -1/2 < a ∧ a < 3) :=
sorry

end math_problem_l365_365232


namespace willie_currency_exchange_l365_365448

theorem willie_currency_exchange :
  let euro_amount := 70
  let pound_amount := 50
  let franc_amount := 30

  let euro_to_dollar := 1.2
  let pound_to_dollar := 1.5
  let franc_to_dollar := 1.1

  let airport_euro_rate := 5 / 7
  let airport_pound_rate := 3 / 4
  let airport_franc_rate := 9 / 10

  let flat_fee := 5

  let official_euro_dollars := euro_amount * euro_to_dollar
  let official_pound_dollars := pound_amount * pound_to_dollar
  let official_franc_dollars := franc_amount * franc_to_dollar

  let airport_euro_dollars := official_euro_dollars * airport_euro_rate
  let airport_pound_dollars := official_pound_dollars * airport_pound_rate
  let airport_franc_dollars := official_franc_dollars * airport_franc_rate

  let final_euro_dollars := airport_euro_dollars - flat_fee
  let final_pound_dollars := airport_pound_dollars - flat_fee
  let final_franc_dollars := airport_franc_dollars - flat_fee

  let total_dollars := final_euro_dollars + final_pound_dollars + final_franc_dollars

  total_dollars = 130.95 :=
by
  sorry

end willie_currency_exchange_l365_365448


namespace rugby_tournament_n_count_l365_365122

noncomputable def valid_n_count : ℕ :=
  (10 to 2017).count (λ n,
    (n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5)) % (2^5 * 3^2 * 5 * 7) = 0
  )

theorem rugby_tournament_n_count : valid_n_count = 562 :=
by {
  sorry
}

end rugby_tournament_n_count_l365_365122


namespace Hexagon_Point_Intersection_l365_365625

variable {P Q R S T U : Type*}

structure InscribedHexagon (P Q R S T U : Type*) :=
(circle : Subtype {c : Circle | P ∈ c ∧ Q ∈ c ∧ R ∈ c ∧ S ∈ c ∧ T ∈ c ∧ U ∈ c})
(eq1 : dist P Q = dist Q R)
(eq2 : dist R S = dist S T)
(eq3 : dist T U = dist U P)

theorem Hexagon_Point_Intersection
  (hex : InscribedHexagon P Q R S T U) :
  concurrent (seg P R) (seg Q S) (seg T U) :=
sorry

end Hexagon_Point_Intersection_l365_365625


namespace time_to_ride_escalator_without_walking_l365_365712

variables (c s d : ℝ)

-- Conditions based on the problem statement
axiom condition_1 : d = 120 * c
axiom condition_2 : d = 48 * (c + s)

-- The main statement to prove
theorem time_to_ride_escalator_without_walking (c_nonzero : c ≠ 0) :
  (d / s = 80) :=
by
  -- Use the conditions to derive the solution
  have h1 : d = 120 * c := condition_1,
  have h2 : d = 48 * (c + s) := condition_2,
  -- Solve for s
  have h3 : 120 * c = 48 * (c + s), from by rw [h1, h2],
  have h4 : 120 * c = 48 * c + 48 * s, from by rw [mul_add, h3],
  have h5 : 72 * c = 48 * s, from by linarith,
  have s_eq : s = (3 / 2) * c, from by field_simp [c_nonzero] at *,
  -- Calculate time t when Clea stands
  have h6 : d / s = 80, from by
    -- We know d = 120c and s = 3c/2
    calc
      d / s = (120 * c) / ((3 / 2) * c) : by rw [h1, s_eq]
      ...   = 80 : by field_simp [c_nonzero],
  exact h6

end time_to_ride_escalator_without_walking_l365_365712


namespace part_a_part_b_part_c_part_d_l365_365807

section problem_conditions

variables {a b c d : ℕ}

-- Define the range of values
def in_range (x : ℕ) : Prop := x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the product
def P (a b c d : ℕ) : ℕ := a * b * c * d

end problem_conditions

-- Part (a): Proof that P can be a multiple of 216
theorem part_a : ∃ a b c d, in_range a ∧ in_range b ∧ in_range c ∧ in_range d ∧ (P a b c d) % 216 = 0 := sorry

-- Part (b): Proof that P cannot be a multiple of 2000
theorem part_b : ¬ ∃ a b c d, in_range a ∧ in_range b ∧ in_range c ∧ in_range d ∧ (P a b c d) % 2000 = 0 := sorry

-- Part (c): Proof that the number of different possible values of P that are divisible by 128 but not by 1024 is 18
noncomputable def count_divisible_128_not_1024 : ℕ :=
  Set.card {P a b c d | ∃ a b c d, in_range a ∧ in_range b ∧ in_range c ∧ in_range d ∧ (P a b c d) % 128 = 0 ∧ (P a b c d) % 1024 ≠ 0 }

theorem part_c : count_divisible_128_not_1024 = 18 := sorry

-- Part (d): Proof that the number of ordered quadruples (a, b, c, d) such that P is 98 less than a multiple of 100 is 4
noncomputable def count_mod_100_98 : ℕ :=
  Set.card {⟨a, b, c, d⟩ | in_range a ∧ in_range b ∧ in_range c ∧ in_range d ∧ ((P a b c d + 98) % 100 = 0)}

theorem part_d : count_mod_100_98 = 4 := sorry

end part_a_part_b_part_c_part_d_l365_365807


namespace tan_value_l365_365222

theorem tan_value (θ : ℝ) (h1 : 0 < θ ∧ θ < π) (h2 : sin (2*θ) = 2 - 2*(cos θ)^2) : tan θ = (1/2) :=
by
  sorry

end tan_value_l365_365222


namespace perp_a_beta_l365_365614

noncomputable def line : Type := sorry
noncomputable def plane : Type := sorry
noncomputable def Incident (l : line) (p : plane) : Prop := sorry
noncomputable def Perpendicular (l1 l2 : line) : Prop := sorry
noncomputable def Parallel (l1 l2 : line) : Prop := sorry

variables {α β : plane} {a AB : line}

-- Conditions extracted from the problem
axiom condition1 : Perpendicular α β
axiom condition2 : Incident AB β ∧ Incident AB α
axiom condition3 : Parallel a α
axiom condition4 : Perpendicular a AB

-- The statement that needs to be proved
theorem perp_a_beta : Perpendicular a β :=
  sorry

end perp_a_beta_l365_365614


namespace mercury_radius_scientific_notation_l365_365031

-- Defining the radius of Mercury
def radius_mercury : ℝ := 2_440_000

-- The mathematical proof problem statement
theorem mercury_radius_scientific_notation : radius_mercury = 2.44 * 10^6 := 
by 
  sorry

end mercury_radius_scientific_notation_l365_365031


namespace diana_reading_hours_l365_365950

theorem diana_reading_hours :
  ∃ H : ℕ, let old_reward := 30 
           let raise := 0.20
           let new_reward := old_reward + (raise * old_reward)
           let additional_minutes := 72
           let increase_per_hour := new_reward - old_reward
           in additional_minutes = increase_per_hour * H ∧ H = 12 := 
by 
   -- sorry states that we'll skip the actual proof details
   sorry

end diana_reading_hours_l365_365950


namespace find_constants_l365_365967

theorem find_constants :
  ∃ A B C D : ℚ,
    (∀ x : ℚ,
      x ≠ 2 → x ≠ 3 → x ≠ 5 → x ≠ -1 →
      (x^2 - 9) / ((x - 2) * (x - 3) * (x - 5) * (x + 1)) =
      A / (x - 2) + B / (x - 3) + C / (x - 5) + D / (x + 1)) ∧
  A = -5/9 ∧ B = 0 ∧ C = 4/9 ∧ D = -1/9 :=
by
  sorry

end find_constants_l365_365967


namespace g_subtraction_l365_365667

noncomputable def g : ℝ → ℝ := λ x, 3 * x^3 + 4 * x^2 - 3 * x + 2

theorem g_subtraction (x h : ℝ) : g (x + h) - g x = h * (9 * x^2 + 8 * x + 9 * x * h + 4 * h + 3 * h^2 - 3) :=
by
  sorry

end g_subtraction_l365_365667


namespace topmost_triangle_is_multiple_of_5_l365_365290

-- Definitions of the given constants and conditions
def bottom_values := [12, a, b, c, d, 3]

-- The condition that the sum of the numbers in neighboring triangles of each gray triangle must be divisible by 5
def condition (neighbors : List ℤ) : Prop :=
  sum neighbors % 5 = 0

-- Using the given conditions to format the Lean proof problem
theorem topmost_triangle_is_multiple_of_5 (a b c d : ℤ) :
  (∃ k : ℤ, -(5 + 5 * a + 5 * b + 5 * c + 5 * d) = 5 * k) :=
sorry

end topmost_triangle_is_multiple_of_5_l365_365290


namespace dragon_legs_l365_365882

variable {x y n : ℤ}

theorem dragon_legs :
  (x = 40) ∧
  (y = 9) ∧
  (220 = 40 * x + n * y) →
  n = 4 :=
by
  sorry

end dragon_legs_l365_365882


namespace smallest_prime_number_conditions_l365_365852

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |> List.sum -- Summing the digits in base 10

def is_prime (n : ℕ) : Prop := Nat.Prime n

def smallest_prime_number (n : ℕ) : Prop :=
  is_prime n ∧ sum_of_digits n = 17 ∧ n > 200 ∧
  (∀ m : ℕ, is_prime m ∧ sum_of_digits m = 17 ∧ m > 200 → n ≤ m)

theorem smallest_prime_number_conditions (p : ℕ) : 
  smallest_prime_number p ↔ p = 197 :=
by
  sorry

end smallest_prime_number_conditions_l365_365852


namespace consecutive_days_probability_l365_365051

theorem consecutive_days_probability :
  let n := 5 in
  let total_pairs := Nat.choose n 2 in
  let consecutive_pairs := 4 in
  (consecutive_pairs : ℚ) / total_pairs = 2 / 5 :=
by
  sorry

end consecutive_days_probability_l365_365051


namespace triangle_area_correct_l365_365874

theorem triangle_area_correct (b h : ℝ) (b_val : b = 8.4) (h_val : h = 5.8) :
  (b * h) / 2 = 24.36 := by
  rw [b_val, h_val]
  norm_num
  sorry

end triangle_area_correct_l365_365874


namespace anand_investment_l365_365922

theorem anand_investment
  (D : ℝ) (P_D : ℝ) (P_T : ℝ)
  (hD : D = 3200)
  (hPD : P_D = 810.28)
  (hPT : P_T = 1380)
  (hProp : ∀ A, P_T = P_D + ( (P_T - P_D) * (D / 3200)))
  (hRat : ∀ A, (P_T - P_D) / P_D = A / D)
  : ∃ A : ℝ, A = 2250.24 :=
begin
  have hPA : P_T - P_D = 569.72, by
  {
    calc
      P_T - P_D = 1380 - 810.28 : by rw [hPT, hPD]
             ... = 569.72 : by norm_num
  },

  have hA : A = (P_T - P_D) / P_D * D, by
  {
    refine (hRat (569.72 / 810.28)) (by (have := 3200; assumption)).elim,
    rw [hD, hPD, hPA],
  },
  use 2250.24,
  exact 2250.24,
  sorry
end

end anand_investment_l365_365922


namespace log_tan_cot_sum_zero_l365_365569

open BigOperators Real

theorem log_tan_cot_sum_zero : 
  ∑ x in Finset.range 44, (log (10 : ℝ) (tan (Real.pi * (x + 1) / 180)) + log (10 : ℝ) (cot (Real.pi * (x + 1) / 180))) = 0 :=
by
  have h_cot_tan : ∀ x : ℝ, log 10 (cot x) = -log 10 (tan x) := 
    by sorry
  rw Finset.sum_congr rfl (λ x hx, h_cot_tan _)
  simp
  sorry

end log_tan_cot_sum_zero_l365_365569


namespace largest_among_abc_l365_365251

variable {a b c : ℝ}

theorem largest_among_abc 
  (hn1 : a < 0) 
  (hn2 : b < 0) 
  (hn3 : c < 0) 
  (h : (c / (a + b)) < (a / (b + c)) ∧ (a / (b + c)) < (b / (c + a))) : c > a ∧ c > b :=
by
  sorry

end largest_among_abc_l365_365251


namespace wet_surface_area_l365_365083

def cistern_length := 8 -- in meters
def cistern_width := 6 -- in meters
def water_height := 1.85 -- in meters

theorem wet_surface_area : (cistern_length * cistern_width +
                          2 * (cistern_length * water_height) +
                          2 * (cistern_width * water_height)) = 99.8 :=
by
  sorry

end wet_surface_area_l365_365083


namespace min_value_at_neg7_l365_365444

noncomputable def f (x : ℝ) : ℝ := x^2 + 14 * x + 24

theorem min_value_at_neg7 : ∀ x : ℝ, f (-7) ≤ f x :=
by
  sorry

end min_value_at_neg7_l365_365444


namespace area_of_triangle_ABC_given_conditions_l365_365268

theorem area_of_triangle_ABC_given_conditions
  (BC : ℝ) (AC : ℝ) (B : ℝ)
  (h1 : BC = 2)
  (h2 : AC = real.sqrt 7)
  (h3 : B = 2 * real.pi / 3) :
  let A := real.arcsin ((2 * real.sqrt 7) / 7)
  let C := B + A 
  sin A := real.sqrt 21 / 7  →
  cos A := 2 * real.sqrt 7 / 7 → 
  sin C := sin(A + B)  →
  (1/2) * AC * BC * sin C = (real.sqrt 3) / 2 :=
sorry

end area_of_triangle_ABC_given_conditions_l365_365268


namespace tonya_hamburgers_to_beat_winner_l365_365422

-- Given conditions
def ounces_per_hamburger : ℕ := 4
def ounces_eaten_last_year : ℕ := 84

-- Calculate the number of hamburgers eaten last year
def hamburgers_eaten_last_year : ℕ := ounces_eaten_last_year / ounces_per_hamburger

-- Prove the number of hamburgers Tonya needs to eat to beat last year's winner
theorem tonya_hamburgers_to_beat_winner : 
  hamburgers_eaten_last_year + 1 = 22 :=
by
  -- It remains to be proven
  sorry

end tonya_hamburgers_to_beat_winner_l365_365422


namespace find_parabola_equation_l365_365602

def parabola_condition (x y p : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ p > 0

def hyperbola_condition (x y : ℝ) : Prop :=
  (x^2 / 3) - y^2 = 1

def common_focus_condition (p : ℝ) : Prop :=
  ∃ f : ℝ, f > 0 ∧ (f = 2) ∧ (p / 2 = f)

theorem find_parabola_equation (p : ℝ) :
  (∃ p, parabola_condition x y p ∧ common_focus_condition p) →
  ∃ k, parabola_condition x y k ∧ k = 4 :=
begin
  intro hp,
  cases hp with p hp_cond,
  use 4,
  unfold parabola_condition,
  sorry
end

end find_parabola_equation_l365_365602


namespace impossible_tiling_13x13_with_1x4_excluding_center_l365_365842

theorem impossible_tiling_13x13_with_1x4_excluding_center :
  ¬ (∃ f : (Fin 42) → Finset (Fin 13 × Fin 13),
        (∀ i, (f i).card = 4) ∧ 
        Pairwise (Disjoint on f) ∧ 
        (⋃ i, f i) = univ \ {(Fin.mk 6 (by norm_num), Fin.mk 6 (by norm_num))})
    :=
by
  -- We need to prove this statement
  sorry

end impossible_tiling_13x13_with_1x4_excluding_center_l365_365842


namespace simplify_neg_pos_neg_l365_365001

theorem simplify_neg_pos_neg : -(+(-6)) = 6 := 
by 
  sorry

end simplify_neg_pos_neg_l365_365001


namespace roots_of_quadratic_l365_365668

variables (k : ℝ) 

noncomputable def has_two_positive_roots (k : ℝ) : Prop :=
  let discriminant := (4 * k + 1)^2 - 4 * 2 * (2 * k^2 - 1) in
  let delta := discriminant > 0 in
  let sum_roots := (4 * k + 1) / 2 > 0 in
  let product_roots := (2 * k^2 - 1) / 2 > 0 in
  discriminant > 0 ∧ sum_roots ∧ product_roots

theorem roots_of_quadratic (k : ℝ) (h : k > 1) : has_two_positive_roots k := by
  sorry

end roots_of_quadratic_l365_365668


namespace ratio_of_p_to_q_l365_365440

theorem ratio_of_p_to_q (p q : ℝ) (h₁ : (p + q) / (p - q) = 4 / 3) (h₂ : p / q = r) : r = 7 :=
sorry

end ratio_of_p_to_q_l365_365440


namespace lambda_range_l365_365608

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {λ : ℝ}

/-- Given the sum of the first n terms of sequence a represented as Sn -/
def Sn (n : ℕ) := (-1)^n * a n - 1/(2^n)

/-- Define bn as 8 times a2 times 2^(n-1) -/
def bn (n : ℕ) := 8 * a 2 * 2^(n-1)

/-- Theorem stating the range for λ satisfying the given conditions -/
theorem lambda_range (hS : ∀ n, S n = Sn n)
                     (hb : ∀ n, ∀ bn > 0, λ bn - 1 > 0)
                     (n_pos : ∀ n, n > 0) : (λ ∈ (1/2, +∞)) :=
sorry

end lambda_range_l365_365608


namespace original_number_satisfies_equation_l365_365859

theorem original_number_satisfies_equation 
  (x : ℝ) 
  (h : 1000 * x = 9 * (1 / x)) : 
  x = 3 * (√10 / 100) :=
by
  sorry

end original_number_satisfies_equation_l365_365859


namespace range_of_x_l365_365556

variable (x y : ℝ)

def op (x y : ℝ) := x * (1 - y)

theorem range_of_x (h : op (x - 1) (x + 2) < 0) : x < -1 ∨ 1 < x :=
by
  dsimp [op] at h
  sorry

end range_of_x_l365_365556


namespace sum_of_coefficients_in_expansion_l365_365822

theorem sum_of_coefficients_in_expansion : 
  (∑ k in finset.range 11, (nat.choose 10 k) * (-2)^k) = 1 :=
by
  sorry

end sum_of_coefficients_in_expansion_l365_365822


namespace angle_ABM_obtuse_l365_365700

theorem angle_ABM_obtuse
  (A B C M : Type) [h1: scalene_triangle A B C]
  (AC_longest: AC > AB ∧ AC > BC)
  (h2: segment_on_extension_AC A C M)
  (h3: CM = BC) : angle A B M > 90 :=
sorry

end angle_ABM_obtuse_l365_365700


namespace min_value_inequality_l365_365323

theorem min_value_inequality (θ φ : ℝ) : 
  (3 * Real.cos θ + 4 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 4 * Real.cos φ - 20)^2 ≥ 549 - 140 * Real.sqrt 5 := 
by
  sorry

end min_value_inequality_l365_365323


namespace probability_correct_phone_number_l365_365349

theorem probability_correct_phone_number :
  let first_digits := {293, 296}
  let last_digits := {0, 2, 5, 8}
  let total_choices := 2 * (4!)
  1 / total_choices = 1 / 48 :=
by
  let first_digits := {293, 296}
  let last_digits := {0, 2, 5, 8}
  let total_choices := 2 * (4!)
  have h1 : ∀ n : ℕ, n = 48 ↔ 1 / n = 1 / 48 := by sorry
  exact (h1 48).2 rfl

end probability_correct_phone_number_l365_365349


namespace strategy_for_antonio_l365_365133

-- We define the concept of 'winning' and 'losing' positions
def winning_position (m n : ℕ) : Prop :=
  ¬ (m % 2 = 0 ∧ n % 2 = 0)

-- Now create the main theorem
theorem strategy_for_antonio (m n : ℕ) : winning_position m n ↔ 
  (¬(m % 2 = 0 ∧ n % 2 = 0)) :=
by
  unfold winning_position
  sorry

end strategy_for_antonio_l365_365133


namespace distance_center_of_cylinder_l365_365480

def radius_cylinder : ℝ := 3 -- inches, since the diameter is 6 inches
def R1 : ℝ := 120
def R2 : ℝ := 90
def R3 : ℝ := 100
def R4 : ℝ := 70

def adjusted_R1 : ℝ := R1 - radius_cylinder -- 117 inches
def adjusted_R2 : ℝ := R2 + radius_cylinder -- 93 inches
def adjusted_R3 : ℝ := R3 - radius_cylinder -- 97 inches
def adjusted_R4 : ℝ := R4 + radius_cylinder -- 73 inches

def distance_arc (r : ℝ) : ℝ := π * r

def total_distance : ℝ := distance_arc adjusted_R1 + distance_arc adjusted_R2 + distance_arc adjusted_R3 + distance_arc adjusted_R4

theorem distance_center_of_cylinder :
  total_distance = 380 * π := by sorry

end distance_center_of_cylinder_l365_365480


namespace tonya_needs_to_eat_more_l365_365420

-- Define the conditions in the problem
def ounces_per_hamburger : ℕ := 4
def ounces_eaten_last_year : ℕ := 84

-- Define a function to calculate hamburgers given ounces
def hamburgers_eaten (ounces : ℕ) (ounce_per_hamburger : ℕ) : ℕ :=
  ounces / ounce_per_hamburger

-- State the theorem
theorem tonya_needs_to_eat_more (ounces_per_hamburger ounces_eaten_last_year : ℕ) :
  hamburgers_eaten ounces_eaten_last_year ounces_per_hamburger + 1 = 22 := by
  sorry

end tonya_needs_to_eat_more_l365_365420


namespace area_of_union_l365_365495

noncomputable theory

-- Definitions based on the given conditions
def side_length := 8
def radius := 12
def area_square := side_length ^ 2
def area_circle := Real.pi * radius ^ 2
def overlap := (1 / 4) * area_circle
def area_union := area_square + area_circle - overlap

-- The theorem stating the desired proof
theorem area_of_union (side_length radius : ℝ) (h_side : side_length = 8) (h_radius : radius = 12) :
  (side_length ^ 2 + Real.pi * radius ^ 2 - (1 / 4) * Real.pi * radius ^ 2) = 64 + 108 * Real.pi :=
by
  rw [h_side, h_radius]
  simp [area_square, area_circle, overlap, area_union]
  sorry

end area_of_union_l365_365495


namespace symmetry_center_of_g_l365_365423

noncomputable def g (x : ℝ) : ℝ := 2 * real.cos (2 * x - 2 * real.pi / 3) - 1

theorem symmetry_center_of_g : ∃ k : ℤ, g (k * real.pi / 2 + real.pi / 12) = -1 :=
by
  sorry

end symmetry_center_of_g_l365_365423


namespace simplify_f_find_f_given_cos_find_f_given_alpha_l365_365211

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.tan (-α - Real.pi)) /
    (Real.tan (-α) * Real.sin (-Real.pi - α))

-- Statement 1: Simplify f(α) == cos(α)
theorem simplify_f (α : ℝ) (h : 0 < (α % (2 * Real.pi)) ∧ (α % (2 * Real.pi)) < Real.pi) : 
  f(α) = Real.cos α :=
sorry

-- Statement 2: Given cos(α - 3/2 * π) = 1/5, find the value of f(α) 
theorem find_f_given_cos (α : ℝ) (h : Real.cos (α - 3 / 2 * Real.pi) = 1 / 5) : 
  f(α) = - 2 * Real.sqrt 6 / 5 :=
sorry

-- Statement 3: Given α = -1860°, find the value of f(α)
theorem find_f_given_alpha (α : ℝ) (h : α = -1860 * (π / 180)) : 
  f(α) = 1 / 2 :=
sorry

end simplify_f_find_f_given_cos_find_f_given_alpha_l365_365211


namespace intersection_A_B_l365_365616

def setA : Set ℝ := {x : ℝ | x > -1}
def setB : Set ℝ := {x : ℝ | x < 3}
def setIntersection : Set ℝ := {x : ℝ | x > -1 ∧ x < 3}

theorem intersection_A_B :
  setA ∩ setB = setIntersection :=
by sorry

end intersection_A_B_l365_365616


namespace problem_intersection_l365_365218

theorem problem_intersection (a b : ℝ) 
    (h1 : b = - 2 / a) 
    (h2 : b = a + 3) 
    : 1 / a - 1 / b = -3 / 2 :=
by
  sorry

end problem_intersection_l365_365218


namespace min_value_expr_l365_365673

theorem min_value_expr (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 1) : 
  (b / (3 * a)) + (3 / b) ≥ 5 := 
sorry

end min_value_expr_l365_365673


namespace vector_subtraction_l365_365990

-- Define the vectors c and d
def c : ℝ × ℝ × ℝ := (5, -3, 2)
def d : ℝ × ℝ × ℝ := (-2, 1, 5)

-- Define the scalar multiplication function
def scalar_mul (a : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (a * v.1, a * v.2, a * v.3)

-- Define vector subtraction function
def vec_sub (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

-- State the theorem
theorem vector_subtraction : vec_sub c (scalar_mul 4 d) = (13, -7, -18) :=
by
  sorry

end vector_subtraction_l365_365990


namespace triangle_angle_A_triangle_side_a_l365_365706

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (h1 : (a - c)*(a + c)*sin C = c * (b - c) * sin B)
variable (area : ℝ)
variable (h2 : area = sqrt 3)
variable (h3 : sin B * sin C = 1/4)

theorem triangle_angle_A (h1 : (a - c)*(a + c)*sin C = c * (b - c) * sin B) : A = π / 3 := 
sorry

theorem triangle_side_a (h1 : (a - c)*(a + c)*sin C = c * (b - c) * sin B)
                        (h2 : area = sqrt 3)
                        (h3 : sin B * sin C = 1/4) : a = 2 * sqrt 3 := 
sorry

end triangle_angle_A_triangle_side_a_l365_365706


namespace large_beaker_fraction_filled_l365_365492

def capacity_small (C : ℝ) := C
def capacity_large (C : ℝ) := 5 * C
def capacity_third (C : ℝ) := (5 / 2) * C

def volume_salt_water (C : ℝ) := (1 / 2) * C
def volume_fresh_water (C : ℝ) := C
def volume_liquid_solution (C : ℝ) := (15 / 8) * C

def total_volume (C : ℝ) := volume_fresh_water C + volume_salt_water C + volume_liquid_solution C
def fraction_filled (C : ℝ) := total_volume C / capacity_large C

theorem large_beaker_fraction_filled (C : ℝ) :
  fraction_filled C = 27 / 40 :=
by
  sorry

end large_beaker_fraction_filled_l365_365492


namespace greta_hourly_wage_is_12_l365_365660

-- Define constants
def greta_hours : ℕ := 40
def lisa_hourly_wage : ℕ := 15
def lisa_hours : ℕ := 32

-- Define the total earnings of Greta and Lisa
def greta_earnings (G : ℕ) : ℕ := greta_hours * G
def lisa_earnings : ℕ := lisa_hours * lisa_hourly_wage

-- Main theorem statement
theorem greta_hourly_wage_is_12 (G : ℕ) (h : greta_earnings G = lisa_earnings) : G = 12 :=
by
  sorry

end greta_hourly_wage_is_12_l365_365660


namespace solve_for_x_l365_365857

theorem solve_for_x (x : ℝ) (h : 4 * x - 5 = 3) : x = 2 :=
by sorry

end solve_for_x_l365_365857


namespace average_of_remaining_two_students_l365_365372

theorem average_of_remaining_two_students
  (h: ∀ (scores : Fin 6 → ℕ), scores = ![59, 67, 97, 103, 109, 113] → 
       ∀ (subset_scores: Fin 4 → ℕ), (94 * 4 = ∑ x, subset_scores x)) : 
  ∃ (remaining_avg: ℕ), remaining_avg = 86 :=
by sorry

end average_of_remaining_two_students_l365_365372


namespace equal_areas_of_triangles_l365_365057

-- Definition for a regular pentagon and inscribed rectangle
structure RegularPentagon (A B C D E : Point) : Prop :=
  (pentagon_property : is_regular_pentagon A B C D E)

structure InscribedRectangle (A B G F : Point) (p : RegularPentagon A B C D E) : Prop :=
  (rectangle_property : is_rectangle A B G F)
  (on_perimeter_F : is_on_perimeter F p)
  (on_perimeter_G : is_on_perimeter G p)

-- Statement to prove equality of triangle areas
theorem equal_areas_of_triangles
  (A B C D E F G : Point)
  (h1 : RegularPentagon A B C D E)
  (h2 : InscribedRectangle A B G F h1) :
  area_of_triangle A E G = area_of_triangle D F G := 
sorry

end equal_areas_of_triangles_l365_365057


namespace no_real_roots_ffx_l365_365876

noncomputable def quadratic_f (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem no_real_roots_ffx (a b c : ℝ) (h : (b - 1)^2 < 4 * a * c) :
  ∀ x : ℝ, quadratic_f a b c (quadratic_f a b c x) ≠ x :=
by
  sorry

end no_real_roots_ffx_l365_365876


namespace photo_arrangements_l365_365470

theorem photo_arrangements (students : Fin 5 -> Prop) (A B : Fin 5) 
  (A_next_to_B : ∃ n, students n ∧ students (n+1) ∧ (n+1 < 5) ∨ (students n ∧ students (n-1) ∧ (n > 0)))
  (A_not_end : ∃ n, students n ∧ (1 ≤ n ∧ n ≤ 3)) :
  (∃! arr : list (Fin 5), 
    A_next_to_B arr ∧ A_not_end arr ∧ (list.distinct arr)) →
    card (list.permutations 5 (subset_of_students A B)) = 36 := 
sorry

end photo_arrangements_l365_365470


namespace problem_II_proof_l365_365213

def set_M (q : ℕ) : Set ℕ := { x | x < q }

def set_A (q n : ℕ) (M : Set ℕ) : Set ℕ := 
  { x | ∃ (xs : Fin n → ℕ), (∀ i, xs i ∈ M) ∧ x = Finset.sum (Finset.finRange n) (λ i, xs i * q ^ (i : ℕ)) }

theorem problem_II_proof (q n : ℕ) (hq : q > 1) (hn : n > 1)
  (s t : ℕ) (M : Set ℕ) (hs : s ∈ set_A q n M) (ht : t ∈ set_A q n M)
  (a b : Fin n → ℕ) (ha : (∀ i, a i ∈ M) ∧ s = Finset.sum (Finset.finRange n) (λ i,  a i * q ^ (i : ℕ)))
  (hb : (∀ i, b i ∈ M) ∧ t = Finset.sum (Finset.finRange n) (λ i, b i * q ^ (i : ℕ))) 
  (h_ineq : a (Fin.last n) < b (Fin.last n)) : 
  s < t :=
sorry

end problem_II_proof_l365_365213


namespace Megan_book_total_l365_365464

theorem Megan_book_total (mystery_shelves picture_shelves books_per_shelf : ℕ)
  (h1 : mystery_shelves = 8)
  (h2 : picture_shelves = 2)
  (h3 : books_per_shelf = 7) :
  (mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf) = 70 :=
by
  rw [h1, h2, h3]
  norm_num
  exact rfl

end Megan_book_total_l365_365464


namespace equal_after_gifting_l365_365187

theorem equal_after_gifting (k : ℕ) (hk : k ≠ 0) : 
  let x := 6 * k
  let y := 4 * k
  let z := 5 * k
  in (2 * x / 3 + z / 5 = 3 * y / 4 ∧ 4 * z / 5 + y / 4 = 2 * x / 3 + z / 5) :=
by 
  let x := 6 * k
  let y := 4 * k
  let z := 5 * k
  have hk_ne0 : (k : ℝ) ≠ 0 := by exact_mod_cast hk
  let a : ℝ := 2 * (x : ℝ) / 3 + z / 5
  let b : ℝ := 3 * (y : ℝ) / 4 + x / 6 
  let c : ℝ := 4 * (z : ℝ) / 5 + y / 4
  have ha : a = b := sorry -- from given conditions and numerical value
  have hb : b = c := sorry -- from given conditions and numerical value
  exact ⟨ha, hb⟩

end equal_after_gifting_l365_365187


namespace base_conversion_unique_solution_l365_365449

theorem base_conversion_unique_solution :
  ∃ (x y z b : ℕ), 
  1989 = x * b^2 + y * b + z ∧
  x + y + z = 27 ∧
  13 ≤ b ∧ b ≤ 44 ∧
  x = 5 ∧ y = 9 ∧ z = 13 ∧ b = 19 :=
by {
  use [5, 9, 13, 19],
  split,
  { exact calc
      1989 = 5 * 19^2 + 9 * 19 + 13 : by norm_num },
  split,
  { exact calc 
      5 + 9 + 13 = 27 : by norm_num },
  split,
  { exact calc 
      13 ≤ 19 : by norm_num },
  split,
  { exact calc 
      19 ≤ 44 : by norm_num },
  repeat { split; try { refl } }
}

end base_conversion_unique_solution_l365_365449


namespace caitlin_age_l365_365514

theorem caitlin_age (aunt_anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ) 
  (h1 : aunt_anna_age = 60)
  (h2 : brianna_age = aunt_anna_age / 3)
  (h3 : caitlin_age = brianna_age - 7)
  : caitlin_age = 13 :=
by
  sorry

end caitlin_age_l365_365514


namespace pythagorean_diagonal_l365_365007

variable (m : ℕ) (h_m : m ≥ 3)

theorem pythagorean_diagonal (h : (2 * m)^2 + a^2 = (a + 2)^2) :
  (a + 2) = m^2 + 1 :=
by
  sorry

end pythagorean_diagonal_l365_365007


namespace sum_of_marked_angles_l365_365853

theorem sum_of_marked_angles (sum_of_angles_around_vertex : ℕ := 360) 
    (vertices : ℕ := 7) (triangles : ℕ := 3) 
    (sum_of_interior_angles_triangle : ℕ := 180) :
    (vertices * sum_of_angles_around_vertex - triangles * sum_of_interior_angles_triangle) = 1980 :=
by
  sorry

end sum_of_marked_angles_l365_365853


namespace perimeter_C_l365_365169

def is_square (n : ℕ) : Prop := n > 0 ∧ ∃ s : ℕ, s * s = n

variable (A B C : ℕ) -- Defining the squares
variable (sA sB sC : ℕ) -- Defining the side lengths

-- Conditions as definitions
axiom square_figures : is_square A ∧ is_square B ∧ is_square C 
axiom perimeter_A : 4 * sA = 20
axiom perimeter_B : 4 * sB = 40
axiom side_length_C : sC = 2 * (sA + sB)

-- The equivalent proof problem statement
theorem perimeter_C : 4 * sC = 120 :=
by
  -- Proof will go here
  sorry

end perimeter_C_l365_365169


namespace circle_reflection_l365_365386

-- Definitions provided in conditions
def initial_center : ℝ × ℝ := (6, -5)
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.snd, p.fst)
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.fst, p.snd)

-- The final statement we need to prove
theorem circle_reflection :
  reflect_y_axis (reflect_y_eq_x initial_center) = (5, 6) :=
by
  -- By reflecting the point (6, -5) over y = x and then over the y-axis, we should get (5, 6)
  sorry

end circle_reflection_l365_365386


namespace product_of_primes_is_582_l365_365066

-- Define the relevant primes based on the conditions.
def smallest_one_digit_prime_1 := 2
def smallest_one_digit_prime_2 := 3
def largest_two_digit_prime := 97

-- Define the product of these primes as stated in the problem.
def product_of_primes := smallest_one_digit_prime_1 * smallest_one_digit_prime_2 * largest_two_digit_prime

-- Prove that this product equals to 582.
theorem product_of_primes_is_582 : product_of_primes = 582 :=
by {
  sorry
}

end product_of_primes_is_582_l365_365066


namespace problem_solution_l365_365352

theorem problem_solution (a b : ℝ) (ha : |a| = 5) (hb : b = -3) :
  a + b = 2 ∨ a + b = -8 :=
by sorry

end problem_solution_l365_365352


namespace max_d_value_l365_365172

theorem max_d_value : ∀ (d e : ℕ), (d < 10) → (e < 10) → (5 * 10^5 + d * 10^4 + 5 * 10^3 + 2 * 10^2 + 2 * 10 + e ≡ 0 [MOD 22]) → (e % 2 = 0) → (d + e = 10) → d ≤ 8 :=
by
  intros d e h1 h2 h3 h4 h5
  sorry

end max_d_value_l365_365172


namespace count_valid_numbers_l365_365663

-- Define the set of allowed digits and the range
def allowed_digits : set ℕ := {0, 1, 7, 8, 9}
def valid_digit (d : ℕ) : Prop := d ∈ allowed_digits
def check_digits (n : ℕ) : Prop := (∀ d ∈ int.digits 10 n, valid_digit d)

theorem count_valid_numbers : ∃ n, n = 1874 ∧ ∀ x, (1 ≤ x ∧ x ≤ 19999) → check_digits x ↔ x ∈ finset.range 19999 \ finset.filter (λ d, ∃ y ∈ int.digits 10 d, ¬ valid_digit y) (finset.range 19999) :=
by {
  sorry
}

end count_valid_numbers_l365_365663


namespace problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l365_365150

theorem problem_1 :
  (sqrt 8 - sqrt (1 / 2)) / sqrt 2 = 3 / 2 :=
sorry

theorem problem_2 :
  2 * sqrt 3 * (sqrt 12 - 3 * sqrt 75 + (1 / 3) * sqrt 108) = -66 :=
sorry

def a := 3 + 2 * sqrt 2
def b := 3 - 2 * sqrt 2

theorem problem_3 :
  a^2 - 3 * a * b + b^2 = 31 :=
sorry

theorem problem_4 (x : Real) (h : (2 * x - 1)^2 = x * (3 * x + 2) - 7) :
  x = 2 ∨ x = 4 :=
sorry

theorem problem_5 (x : Real) (h : 2 * x^2 - 3 * x + 1 / 2 = 0) :
  x = (3 + sqrt 5) / 4 ∨ x = (3 - sqrt 5) / 4 :=
sorry

variables {a b : Real}
axiom roots : a^2 - a - 1 = 0 ∧ b^2 - b - 1 = 0

theorem problem_6 (h : a^2 - a - 1 = 0) (h2 : b^2 - b - 1 = 0) :
  b / a + a / b = -3 :=
sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l365_365150


namespace jerry_total_income_l365_365716

-- Define the base charge for piercing noses and the additional rate for ears
def noseCharge := 20
def earExtraRate := 0.5
def earCharge := noseCharge + (noseCharge * earExtraRate)

-- Define the number of piercings for noses and ears
def numNoses := 6
def numEars := 9

-- Calculate the total money Jerry makes and state the theorem
def totalMoney := (numNoses * noseCharge) + (numEars * earCharge)

theorem jerry_total_income : totalMoney = 390 := by
  sorry

end jerry_total_income_l365_365716


namespace DE_proportion_l365_365269

-- Define the basic geometric setup
structure Triangle (α : Type*) :=
(a b c : α)  -- sides opposite to the angles A, B, and C

variable {α : Type*} [LinearOrderedField α]

def AD_bisects_angle_A (t : Triangle α) (A B C : Point) : Prop :=
  ∃ D E : Point, 
  Between A D C ∧ Between D B C ∧ Midpoint E A D ∧ AngleBisector t.a t.b t.c

-- The main theorem: Given the conditions, the ratio holds
theorem DE_proportion (t : Triangle α) (a b c : α)
  (AD_bisects_angle_A t A B C) (E Midpoint AD) : 
  DE / a = 1 / 2 := 
sorry

end DE_proportion_l365_365269


namespace pigeon_count_correct_l365_365880

def initial_pigeon_count : ℕ := 1
def new_pigeon_count : ℕ := 1
def total_pigeon_count : ℕ := 2

theorem pigeon_count_correct : initial_pigeon_count + new_pigeon_count = total_pigeon_count :=
by
  sorry

end pigeon_count_correct_l365_365880


namespace count_6_digit_palindromes_with_even_middle_digits_l365_365242

theorem count_6_digit_palindromes_with_even_middle_digits :
  let a_values := 9
  let b_even_values := 5
  let c_values := 10
  a_values * b_even_values * c_values = 450 :=
by {
  sorry
}

end count_6_digit_palindromes_with_even_middle_digits_l365_365242


namespace solve_diophantine_eq_l365_365945

theorem solve_diophantine_eq (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  a^2 = b * (b + 7) ↔ (a = 12 ∧ b = 9) ∨ (a = 0 ∧ b = 0) := 
by 
  sorry

end solve_diophantine_eq_l365_365945


namespace inequality_solution_set_when_a_is_5_determining_integer_a_for_A_set_l365_365636

noncomputable theory

def f (a x : ℝ) : ℝ := |x - 1| - |2 * x - a|

/-
  Problem (1)
-/
theorem inequality_solution_set_when_a_is_5 :
  {x : ℝ | f 5 x ≥ 0} = Icc 2 4 :=
sorry

/-
  Problem (2)
-/
theorem determining_integer_a_for_A_set :
  ∃ a : ℤ, {5} ⊆ {x : ℝ | f a x ≥ 3} ∧ ¬ (6 ∈ {x : ℝ | f a x ≥ 3}) ∧ a = 9 :=
sorry

end inequality_solution_set_when_a_is_5_determining_integer_a_for_A_set_l365_365636


namespace total_number_of_games_played_l365_365951

theorem total_number_of_games_played
  (win_rate_first_100_games : 0.85)
  (win_first_100_games : ℕ)
  (total_first_100_games : ℕ)
  (home_win_rate_remaining : 0.6)
  (away_win_rate_remaining : 0.45)
  (overall_win_rate : 0.7)
  (streak_of_wins : ℕ)
  (remaining_games : ℕ)
  (total_games : ℕ)
  (equal_home_away_games : remaining_games / 2 = remaining_games / 2)
  (calculated_first_100_games : win_first_100_games = 85)
  (calculated_total_first_100_games : total_first_100_games = 100)
  (calculated_remaining_games : remaining_games = total_games - total_first_100_games)
  (total_game_wins: ℕ)
  (calculated_total_game_wins: total_game_wins = total_games * overall_win_rate)
  (home_win: ℕ)
  (away_win: ℕ)
  (calculated_home_win: home_win = home_win_rate_remaining * (remaining_games / 2))
  (calculated_away_win: away_win = away_win_rate_remaining * (remaining_games / 2))
  (calculated_remaining_game_wins: (win_first_100_games + home_win + away_win) = total_game_wins)
  : total_games = 186 :=
by
  sorry

end total_number_of_games_played_l365_365951


namespace altitude_bisects_l365_365686

structure Triangle (V : Type) [inner_product_space ℝ V] :=
(A B C A₁ B₁ L : V)
(hABC_acute : angle A B C < π / 2 ∧ angle B C A < π / 2 ∧ angle C A B < π / 2)
(hA₁ : ∃ H : V, collinear [A, A₁, H] ∧ angle B A A₁ = π / 2 ∧ angle C A A₁ = π / 2)
(hB₁ : ∃ H : V, collinear [B, B₁, H] ∧ angle A B B₁ = π / 2 ∧ angle C B B₁ = π / 2)
(hL : ∃ circABC : set V, circumcircle circABC [A, B, C] ∧ L ∈ circABC ∧ angle L C B = π / 2 ∧ LC = CB)
(hBLB₁ : angle B L B₁ = π / 2)

theorem altitude_bisects 
  {V : Type} [inner_product_space ℝ V] 
  (Δ : Triangle V) : 
  bisects Δ.A Δ.A₁ Δ.B Δ.B₁ :=
sorry

end altitude_bisects_l365_365686


namespace find_num_apples_l365_365867

def num_apples (A P : ℕ) : Prop :=
  P = (3 * A) / 5 ∧ A + P = 240

theorem find_num_apples (A : ℕ) (P : ℕ) :
  num_apples A P → A = 150 :=
by
  intros h
  -- sorry for proof
  sorry

end find_num_apples_l365_365867


namespace arithmetic_sequence_properties_l365_365610

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n + 1) * (a 0 + a n) / 2

theorem arithmetic_sequence_properties :
  ∃ (a : ℕ → ℤ) (S : ℕ → ℤ),
  is_arithmetic_sequence a ∧
  sum_of_first_n_terms a S ∧
  a 1 = 5 ∧
  S 8 = 99 ∧
  (∀ n : ℕ, a n = 2 * n + 1) ∧
  (∀ n : ℕ, T n = 1 - 1 / (n + 1)) ∧
  (∀ n : ℕ, 1 / 2 ≤ T n ∧ T n < 1) :=
begin
  sorry
end

end arithmetic_sequence_properties_l365_365610


namespace fraction_to_terminating_decimal_l365_365571

-- Define the fraction as given
def frac := 19 / (2^2 * 5^3)

-- Define the condition that it can be written as a / 10^n
def can_be_written_as_term_dec (x : ℝ) : Prop :=
  ∃ (a n : ℤ), x = a / 10^n

-- The final theorem to be proved
theorem fraction_to_terminating_decimal : frac = 0.038 :=
by
  sorry

end fraction_to_terminating_decimal_l365_365571


namespace prob_students_on_both_days_l365_365191
noncomputable def probability_event_on_both_days: ℚ := by
  let total_days := 2
  let total_students := 4
  let prob_single_day := (1 / total_days : ℚ) ^ total_students
  let prob_all_same_day := 2 * prob_single_day
  let prob_both_days := 1 - prob_all_same_day
  exact prob_both_days

theorem prob_students_on_both_days : probability_event_on_both_days = 7 / 8 :=
by
  exact sorry

end prob_students_on_both_days_l365_365191


namespace total_volume_of_sand_l365_365911

def r1 : ℝ := 6
def h1 : ℝ := 7.2
def V1 : ℝ := (π * r1^2 * h1) / 3

def r2 : ℝ := 3
def h2 : ℝ := 3.6
def V2 : ℝ := (π * r2^2 * h2) / 3

theorem total_volume_of_sand :
  V1 + V2 = 97.2 * π := by
  sorry

end total_volume_of_sand_l365_365911


namespace students_behind_yoongi_l365_365831

theorem students_behind_yoongi (n k : ℕ) (hn : n = 30) (hk : k = 20) : n - (k + 1) = 9 := by
  sorry

end students_behind_yoongi_l365_365831


namespace isosceles_triangle_A₀B₀C₀_l365_365709

-- Definition of the geometric entities and conditions given in the problem
variables (A B C A' B' C' M A₀ B₀ C₀ : Type)
variables [InTriangle A B C] [MediansOf A A' B B' C C'] [Circumcircle A₀ B₀ C₀]
variables [Centroid M]

-- The property that the centroid M bisects AA₀
axiom M_bisects_AA₀ : Bisects M A₀ A

-- Main theorem statement
theorem isosceles_triangle_A₀B₀C₀ : IsIsoscelesTriangle A₀ B₀ C₀ :=
sorry

end isosceles_triangle_A₀B₀C₀_l365_365709


namespace rides_on_roller_coaster_l365_365809

-- Definitions based on the conditions given.
def roller_coaster_cost : ℕ := 17
def total_tickets : ℕ := 255
def tickets_spent_on_other_activities : ℕ := 78

-- The proof statement.
theorem rides_on_roller_coaster : (total_tickets - tickets_spent_on_other_activities) / roller_coaster_cost = 10 :=
by 
  sorry

end rides_on_roller_coaster_l365_365809


namespace train_length_l365_365910

theorem train_length (v : ℝ) (t : ℝ) (b : ℝ) : v = 72 ∧ t = 13.998880089592832 ∧ b = 170 →
  let v_ms := v * 1000 / 3600 in
  let d := v_ms * t in
  let l := d - b in
  l = 109.97760179185664 := 
sorry

end train_length_l365_365910


namespace longest_gp_within_bounds_l365_365180

-- Define the sequence and the geometric properties
def geometric_sequence (seq : List ℕ) (r : ℕ) : Prop :=
  ∀ i, i < seq.length - 1 → seq.get i * r = seq.get (i + 1)

-- Define the conditions for the set and the sequence
def is_within_bounds (seq : List ℕ) (lower upper : ℕ) : Prop :=
  ∀ i, i < seq.length → lower ≤ seq.get i ∧ seq.get i ≤ upper

-- Define the main hypothesis for the problem
theorem longest_gp_within_bounds :
  ∃ seq : List ℕ, 
    geometric_sequence seq 3/2 ∧ 
    is_within_bounds seq 100 1000 ∧ 
    seq.length = 6 :=
sorry

end longest_gp_within_bounds_l365_365180


namespace base_of_exponent_l365_365258

theorem base_of_exponent (b x y : ℕ) (h1 : x - y = 12) (h2 : x = 12) (h3 : b^x * 4^y = 531441) : b = 3 :=
by
  sorry

end base_of_exponent_l365_365258


namespace sum_digits_joeys_age_l365_365717

theorem sum_digits_joeys_age (C J Z n : ℕ) (h1 : J = C + 2) (h2 : Z = 2) 
    (h3 : ∀ k, k ∈ {0, 1, 2, 3, 4, 5, 6, 7} → (C + k) % (Z + k) = 0) :
    ∃ n, (J + n) % (Z + n) = 0 ∧ ((J + n) / 10 + (J + n) % 10) = 9 :=
begin
  sorry
end

end sum_digits_joeys_age_l365_365717


namespace optionA_not_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_right_triangle_l365_365508
-- Import necessary libraries

-- Define each of the conditions as Lean definitions
def OptionA (a b c : ℝ) : Prop := a = 1.5 ∧ b = 2 ∧ c = 3
def OptionB (a b c : ℝ) : Prop := a = 7 ∧ b = 24 ∧ c = 25
def OptionC (a b c : ℝ) : Prop := ∃ k : ℕ, a = (3 : ℝ)*k ∧ b = (4 : ℝ)*k ∧ c = (5 : ℝ)*k
def OptionD (a b c : ℝ) : Prop := a = 9 ∧ b = 12 ∧ c = 15

-- Define the Pythagorean theorem predicate
def Pythagorean (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- State the theorem to prove Option A cannot form a right triangle
theorem optionA_not_right_triangle : ¬ Pythagorean 1.5 2 3 := by sorry

-- State the remaining options can form a right triangle
theorem optionB_right_triangle : Pythagorean 7 24 25 := by sorry
theorem optionC_right_triangle (k : ℕ) : Pythagorean (3 * k) (4 * k) (5 * k) := by sorry
theorem optionD_right_triangle : Pythagorean 9 12 15 := by sorry

end optionA_not_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_right_triangle_l365_365508


namespace aligned_point_correct_l365_365907

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def fold_line (p1 p2 : ℝ × ℝ) : ℝ × ℝ → ℝ :=
  let mid := midpoint p1 p2
  λ (p : ℝ × ℝ), (mid.2 - mid.1) * (p.1 - mid.1) + mid.2

def symmetric_point (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ × ℝ :=
  let d := (p.2 - l(p)).abs / real.sqrt 2
  (p.2, p.1) 

theorem aligned_point_correct :
  let p1 := (0, 4) 
  let p2 := (4, 0)
  let aligned_point := (8, 6)
  let (m, n) := symmetric_point aligned_point (fold_line p1 p2)
  m + n = 14 :=
by 
  sorry

end aligned_point_correct_l365_365907


namespace roots_of_polynomial_l365_365173

def poly (x : ℝ) : ℝ := x^3 - 3 * x^2 - 4 * x + 12

theorem roots_of_polynomial : 
  (poly 2 = 0) ∧ (poly (-2) = 0) ∧ (poly 3 = 0) ∧ 
  (∀ x, poly x = 0 → x = 2 ∨ x = -2 ∨ x = 3) :=
by
  sorry

end roots_of_polynomial_l365_365173


namespace sum_base6_l365_365914

theorem sum_base6 (a b : ℕ) (h₁ : a = 5) (h₂ : b = 23) : 
  let sum := Nat.ofDigits 6 [2, 3] + Nat.ofDigits 6 [5]
  in Nat.digits 6 sum = [2, 3] :=
by
  sorry

end sum_base6_l365_365914


namespace koschei_coins_l365_365312

theorem koschei_coins :
  ∃ a : ℕ, a % 10 = 7 ∧ a % 12 = 9 ∧ 300 ≤ a ∧ a ≤ 400 ∧ a = 357 :=
by
  sorry

end koschei_coins_l365_365312


namespace superprimes_less_than_15_l365_365898

def is_superprime (p : ℕ) : Prop :=
  is_prime p ∧ is_prime (2 * p - 1)

def number_of_superprimes_less_than_15 : ℕ :=
  (List.range 15).filter is_superprime).length

theorem superprimes_less_than_15 :
  number_of_superprimes_less_than_15 = 3 :=
by sorry

end superprimes_less_than_15_l365_365898


namespace power_zero_any_nonzero_l365_365141

theorem power_zero_any_nonzero (a : ℕ) (h : a ≠ 0) : a^0 = 1 := by 
  -- This proof typically follows from basic properties of exponentiation, but we don't need to prove it here
  sorry

example : 2023^0 = 1 := by
  apply power_zero_any_nonzero,
  -- Non-zero condition is trivially true in this case, justifying assumption
  dec_trivial

end power_zero_any_nonzero_l365_365141


namespace mary_fruits_left_l365_365345

theorem mary_fruits_left (apples_initial oranges_initial blueberries_initial : ℕ)
  (apples_eaten oranges_eaten blueberries_eaten : ℕ) :
  apples_initial = 14 →
  oranges_initial = 9 →
  blueberries_initial = 6 →
  apples_eaten = 1 →
  oranges_eaten = 1 →
  blueberries_eaten = 1 →
  (apples_initial - apples_eaten) + (oranges_initial - oranges_eaten) + (blueberries_initial - blueberries_eaten) = 26 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end mary_fruits_left_l365_365345


namespace tank_capacity_l365_365881

-- Define the miles per gallon of the car
def miles_per_gallon_before_mod := 33

-- Define the improvement in fuel efficiency
def efficiency_factor := 0.75

-- Define the extra miles the car can travel after the modification
def extra_miles_after_mod := 176

-- Define the modified miles per gallon
def miles_per_gallon_after_mod := miles_per_gallon_before_mod * (1 / efficiency_factor)

-- Define the equation to find the tank capacity x in gallons
theorem tank_capacity : ∃ x : ℝ, miles_per_gallon_after_mod * x = miles_per_gallon_before_mod * x + extra_miles_after_mod :=
by
  sorry

end tank_capacity_l365_365881


namespace girls_equal_barefoot_children_l365_365411

variables (B_b G_s B_g : ℕ)

theorem girls_equal_barefoot_children (h : B_b = G_s) : 
  G_s + B_g = B_b + B_g :=
by
  rw [← h]
  exact rfl

end girls_equal_barefoot_children_l365_365411


namespace total_inflation_time_l365_365127

-- Define the conditions
def time_to_inflate_alexia := 18
def time_to_inflate_ermias := 25
def alexia_balls := 36
def ermias_balls := alexia_balls + 8

-- Statement of the theorem
theorem total_inflation_time :
  let total_alexia_time := alexia_balls * time_to_inflate_alexia,
  let total_ermias_time := ermias_balls * time_to_inflate_ermias,
  total_alexia_time + total_ermias_time = 1748 := by
  sorry

end total_inflation_time_l365_365127


namespace tan_alpha_plus_beta_tan_alpha_plus_2beta_l365_365194

noncomputable def tan (x : ℝ) : ℝ := Mathlib.tan x

variables (α β : ℝ)
hypothesis (hα : tan α = 1 / 7) (hβ : tan β = 1 / 3)

theorem tan_alpha_plus_beta :
  tan (α + β) = 1 / 2 :=
sorry

theorem tan_alpha_plus_2beta :
  tan (α + 2 * β) = 1 :=
sorry

end tan_alpha_plus_beta_tan_alpha_plus_2beta_l365_365194


namespace number_of_possible_integral_values_l365_365733

theorem number_of_possible_integral_values (x : ℤ) (hx : 7 < x) (hx2 : x < 21) :
  finset.card (finset.Ico 8 21) = 13 :=
by
  sorry

end number_of_possible_integral_values_l365_365733


namespace product_of_segments_is_constant_l365_365878

variable (R d : ℝ)
variable (O K : Point)
variable (circle : Circle O R)
variable (chord_passes_through_K : ∀ (A B : Point), A ∈ circle → B ∈ circle → A ≠ B → K ∈ Segment A B)

theorem product_of_segments_is_constant (A B : Point) (hA : A ∈ circle) (hB : B ∈ circle) (hAB : A ≠ B) (hK : K ∈ Segment A B) :
  AK * KB = R^2 - d^2 :=
sorry

end product_of_segments_is_constant_l365_365878


namespace AE_distance_proof_l365_365283

-- Define the setup for the problem: rectangle ABCD with specified side lengths, point E on CD, and angle CBE.

noncomputable def rectangle_ABC (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] := 
  (ab_bc_len : AB.dist (30) ∧ BC.dist (15)) ∧ 
  (E_on_CD : ∃ E, E ∈ line[CD]) ∧
  (angle_CBE : angle (C B E) = 30°)

-- Define the distance from A to E

noncomputable def AE_dist (A E : Type) [MetricSpace A] [MetricSpace E] := 
  15 * real.sqrt 5

-- Main theorem statement

theorem AE_distance_proof (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (h : rectangle_ABC A B C D E) : AE_dist A E := by 
  sorry

end AE_distance_proof_l365_365283


namespace rational_power_sum_l365_365487

theorem rational_power_sum (a b : ℚ) (ha : a = 1 / a) (hb : b = - b) : a ^ 2007 + b ^ 2007 = 1 ∨ a ^ 2007 + b ^ 2007 = -1 := by
  sorry

end rational_power_sum_l365_365487


namespace calculate_expression_l365_365143

theorem calculate_expression : |1 - Real.sqrt 3| - (Real.sqrt 3 - 1)^0 = Real.sqrt 3 - 2 := by
  sorry

end calculate_expression_l365_365143


namespace quad_area_inscribed_circumscribed_l365_365886

noncomputable def area_of_quad (R : ℝ) : ℝ :=
  8 * R ^ 2 / 5

theorem quad_area_inscribed_circumscribed (AB CD BC AD : ℝ) (R : ℝ) :
  -- Conditions
  AB = 2 * BC →
  AB + CD = AD + BC →
  ∃ O : ℝ × ℝ, dist O (0, 0) = R ∧ 
    ∀ A B C D : ℝ × ℝ, collinear {A, B, C, D} →
    ∃ K M : ℝ × ℝ, K = midpoint (A, D) ∧ M = midpoint (C, D) →
    -- Diagonals of the quadrilateral are mutually perpendicular
    (let AC : ℝ × ℝ := A - C in let BD : ℝ × ℝ := B - D in dot_product AC BD = 0) →
    -- Prove the area of quadrilateral ABCD
    area_of_quad R = 8 * R ^ 2 / 5 :=
begin
  sorry
end

end quad_area_inscribed_circumscribed_l365_365886


namespace time_to_cross_platform_l365_365484

-- Definitions
def length_of_train : ℝ := 270.0416
def length_of_platform : ℝ := 250
def speed_of_train_kmh : ℕ := 72

-- Core calculation (converted units and total length)
def total_distance_covered : ℝ := length_of_train + length_of_platform
def speed_of_train_ms : ℝ := (speed_of_train_kmh * 1000) / 3600

-- Statement to prove
theorem time_to_cross_platform : total_distance_covered / speed_of_train_ms = 26.00208 :=
by sorry

end time_to_cross_platform_l365_365484


namespace gallery_has_pieces_of_art_l365_365455

variable (A : ℕ)
variable (h_disp : 1 / 3 A)
variable (h_scul_disp : 1 / 6 (1 / 3 A))
variable (h_not_disp : 2 / 3 A)
variable (h_scul_not_disp : 2 / 3 (2 / 3 A) = 400)

theorem gallery_has_pieces_of_art (h_disp : 1 / 3 A) 
                                  (h_scul_disp : 1 / 6 (1 / 3 A))
                                  (h_not_disp : 2 / 3 A) 
                                  (h_scul_not_disp : 2 / 3 (2 / 3 A) = 400) :
  A = 900 :=
sorry

end gallery_has_pieces_of_art_l365_365455


namespace parallel_lines_slope_eq_l365_365072

variable (k : ℝ)

theorem parallel_lines_slope_eq (h : 5 = 3 * k) : k = 5 / 3 :=
by
  sorry

end parallel_lines_slope_eq_l365_365072


namespace graphs_intersect_at_one_point_l365_365561

theorem graphs_intersect_at_one_point (b : ℝ) :
  (∃! x : ℝ, bx^2 - 5x + 3 = 7x - 6) ↔ b = 4 :=
by
  sorry

end graphs_intersect_at_one_point_l365_365561


namespace geometric_seq_a3_equals_2_l365_365324

-- Definitions of the conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q
def common_ratio_negative (q : ℝ) := q < 0
def initial_term := a 1 = 2
def condition := a 3 - 4 = a 2

-- Lean 4 statement to prove
theorem geometric_seq_a3_equals_2
  (a : ℕ → ℝ)
  (q : ℝ)
  (hq : common_ratio_negative q)
  (hg : geometric_sequence a q)
  (h1 : initial_term)
  (h2 : condition) :
  a 3 = 2 :=
by
  sorry

end geometric_seq_a3_equals_2_l365_365324


namespace circle_inscribed_in_rectangle_l365_365099

theorem circle_inscribed_in_rectangle (a b : ℝ) (h1 : a = 9) (h2 : b = 12) :
  let d := Real.sqrt (a^2 + b^2) in
  let C := π * d in
  let A := π * (d / 2)^2 in
  C = 15 * π ∧ A = (225 / 4) * π :=
by
  sorry

end circle_inscribed_in_rectangle_l365_365099


namespace logarithmic_expression_floor_l365_365064

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithmic_expression_floor :
  (Int.floor (log_base 3 (1006 : ℝ))) = 6 :=
begin
  -- no proof required
  sorry
end

end logarithmic_expression_floor_l365_365064


namespace cost_of_each_shirt_l365_365974

theorem cost_of_each_shirt
  (x : ℝ) 
  (h : 3 * x + 2 * 20 = 85) : x = 15 :=
sorry

end cost_of_each_shirt_l365_365974


namespace closest_value_to_sin_2023_l365_365814

theorem closest_value_to_sin_2023 :
  (|sin 2023 - (- √2 / 2)| < |sin 2023 - 1 / 2| ∧
   |sin 2023 - (- √2 / 2)| < |sin 2023 - √2 / 2| ∧
   |sin 2023 - (- √2 / 2)| < |sin 2023 - (- 1 / 2)|) :=
sorry

end closest_value_to_sin_2023_l365_365814


namespace interval_of_monotonic_increase_l365_365677

-- Definition of the conditions
def power_function (a : ℝ) (x : ℝ) : ℝ := x ^ a
def passes_through_point (a : ℝ) : Prop := power_function a 3 = 9

-- The main statement to prove
theorem interval_of_monotonic_increase (a : ℝ) (h : passes_through_point a) : 
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → power_function a x ≤ power_function a y) :=
sorry

end interval_of_monotonic_increase_l365_365677


namespace least_squares_minimizes_squared_errors_l365_365609

variables {n : ℕ} {x y : Fin n → ℝ} {a b : ℝ}

theorem least_squares_minimizes_squared_errors :
  ∃ (a b : ℝ), (∀ x y : Fin n → ℝ, ∃ (squared_errors_sum : ℝ), 
  squared_errors_sum = ∑ i, (y i - (a + b * x i))^2) := 
sorry

end least_squares_minimizes_squared_errors_l365_365609


namespace problem_statement_l365_365740

noncomputable def a (k : ℕ) : ℝ := 2^k / (3^(2^k) + 1)
noncomputable def A : ℝ := (Finset.range 10).sum (λ k => a k)
noncomputable def B : ℝ := (Finset.range 10).prod (λ k => a k)

theorem problem_statement : A / B = (3^(2^10) - 1) / 2^47 - 1 / 2^36 := 
by
  sorry

end problem_statement_l365_365740


namespace remaining_balloons_l365_365591

theorem remaining_balloons : 
  let fred_balloons := 10.0 in
  let sam_balloons := 46.0 in
  let destroyed_balloons := 16.0 in
  fred_balloons + sam_balloons - destroyed_balloons = 40.0 :=
by
  let fred_balloons := 10.0
  let sam_balloons := 46.0
  let destroyed_balloons := 16.0
  have h : fred_balloons + sam_balloons - destroyed_balloons = 40.0 := by
    simp [fred_balloons, sam_balloons, destroyed_balloons]
    sorry
  exact h

end remaining_balloons_l365_365591


namespace math_problem_l365_365265

noncomputable theory
open Real

theorem math_problem (a b c : ℝ) 
  (h_b_eq_a : b = a) (h_c_eq_neg2a : c = -2 * a) (h_a_lt_0 : a < 0) :
  (b < 0) ∧ (c > 0) ∧ (a - b + c > 0) ∧ ¬(a + b + c > 0) ∧ 
  {x : ℝ | -2 < x ∧ x < 1} = {x : ℝ | ax^2 + bx + c > 0} :=
by {
  sorry
}

end math_problem_l365_365265


namespace find_a_and_monotonicity_range_of_k_l365_365998

noncomputable def f (x : ℝ) : ℝ := (- 2 ^ x + 1) / (2 ^ x + 1)

theorem find_a_and_monotonicity (a : ℝ) :
  (∀ x : ℝ, f x = (a * 2^x + 1) / (2^x + 1)) →
  (∀ x : ℝ, f (-x) = - f x) ∧
  (∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2) :=
sorry

theorem range_of_k (k : ℝ) :
  (∃ t : ℝ, t ∈ set.Icc 1 2 ∧ f (t^2 - 2*t) + f (2*t^2 - k) > 0) →
  k ∈ set.Ioi 1 :=
sorry

end find_a_and_monotonicity_range_of_k_l365_365998


namespace ten_unique_positive_odd_integers_equality_l365_365407

theorem ten_unique_positive_odd_integers_equality {x : ℕ} (h1: x = 3):
  ∃ S : Finset ℕ, S.card = 10 ∧ 
    (∀ n ∈ S, n < 100 ∧ n % 2 = 1 ∧ 
      ∃ k : ℕ, k % 2 = 1 ∧ n = k * x) :=
by
  sorry

end ten_unique_positive_odd_integers_equality_l365_365407


namespace expected_pulls_l365_365381

/-- Define the expected number of pulls needed to crash the helicopter when 
    it is currently at an angle n degrees with the ground. -/
def E : ℕ → ℝ
| n := if n >= 90 then 0 
       else (E (n + x) + E (n + x + 1)) / 2 + 1

/-- Given the conditions, prove the expected number of pulls needed to crash 
    the helicopter starting from 0 degrees is 269/32. -/
theorem expected_pulls (x : ℕ) : E 0 = 269 / 32 :=
sorry

end expected_pulls_l365_365381


namespace number_of_ordered_triples_l365_365580

theorem number_of_ordered_triples (x y z : ℝ) (hx : x + y = 3) (hy : xy - z^2 = 4)
  (hnn : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) : 
  ∃! (x y z : ℝ), (x + y = 3) ∧ (xy - z^2 = 4) ∧ (0 ≤ x) ∧ (0 ≤ y) ∧ (0 ≤ z) :=
sorry

end number_of_ordered_triples_l365_365580


namespace phi_is_pi_over_6_l365_365646

theorem phi_is_pi_over_6 (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π)
  (h_even : ∀ x : ℝ, sin (2 * (x - π / 3) + φ) = sin (2 * (-x - π / 3) + φ)) :
  φ = π / 6 := 
sorry

end phi_is_pi_over_6_l365_365646


namespace max_distance_P_to_PBC_l365_365135

open EuclideanGeometry

-- Conditions
variable (A B C D P : Point)
variable (ABC PBD PBC : Plane)
variable (AB AC : ℝ)

-- Given conditions encoded as hypotheses
axiom H1 : ∠ABC = 120
axiom H2 : AB = 2
axiom H3 : AC = 2
axiom H4 : D ∈ Line·segment A C

-- Additional structure
axiom H5 : P ∉ Plane⟨ABC⟩
axiom H6 : Plane⟨P B D⟩ ⊥ Plane⟨A B C⟩

-- Proof problem
theorem max_distance_P_to_PBC : ∀ (P : Point), max_distance_to_plane (P : Point) (PBC : Plane) = 2 := 
by 
  sorry

end max_distance_P_to_PBC_l365_365135


namespace annie_journey_time_l365_365923

noncomputable def total_time_journey (walk_speed1 bus_speed train_speed walk_speed2 blocks_walk1 blocks_bus blocks_train blocks_walk2 : ℝ) : ℝ :=
  let time_walk1 := blocks_walk1 / walk_speed1
  let time_bus := blocks_bus / bus_speed
  let time_train := blocks_train / train_speed
  let time_walk2 := blocks_walk2 / walk_speed2
  let time_back := time_walk2
  time_walk1 + time_bus + time_train + time_walk2 + time_back + time_train + time_bus + time_walk1

theorem annie_journey_time :
  total_time_journey 2 4 5 2 5 7 10 4 = 16.5 := by 
  sorry

end annie_journey_time_l365_365923


namespace double_sum_eq_l365_365543

theorem double_sum_eq : 
  (∑ n in (finset.Ico 2 (⊤ : ℕ)), ∑ k in (finset.Ico 1 n), k / (3 : ℝ)^(n + k)) = (9 / 128 : ℝ) :=
sorry

end double_sum_eq_l365_365543


namespace repeating_decimal_product_l365_365435

theorem repeating_decimal_product :
  let x := 0.016 in
  let frac := 16 / 999 in
  (nat.lcm (nat.gcd (16, 999))) = 15984 :=
by
  sorry

end repeating_decimal_product_l365_365435


namespace sequence_bounds_l365_365607

section sequences

-- Definition of the sequences a_n
def a (n : ℕ) : ℕ :=
if n = 0 then 0 -- For natural number indexing convenience
else 2^(n-1)

-- Sum of the first n terms of sequence a_n
def S (n : ℕ) : ℕ :=
∑ i in Finset.range n, a (i + 1)

-- Definition of the sequence b_n
def b (n : ℕ) : ℕ :=
2 * n - 1

-- Definition of c_n and T_n
def c (n : ℕ) : ℚ :=
1 / ((b n : ℚ) * (b (n + 1) : ℚ))

def T (n : ℕ) : ℚ :=
∑ i in Finset.range (n + 1), c (i + 1)

-- The main theorem to prove the bounds on T_n
theorem sequence_bounds (n : ℕ) (hn : 1 ≤ n) : 
  1 / 3 ≤ T n ∧ T n < 1 / 2 :=
sorry

end sequences

end sequence_bounds_l365_365607


namespace value_increase_factor_l365_365418

theorem value_increase_factor (P S : ℝ) (frac F : ℝ) (hP : P = 200) (hS : S = 240) (hfrac : frac = 0.40) :
  frac * (P * F) = S -> F = 3 := by
  sorry

end value_increase_factor_l365_365418


namespace num_zeros_1_div_15_pow_15_l365_365155

theorem num_zeros_1_div_15_pow_15 : 
  (nat.find_greatest_zero_prefix_length (1 / (15:ℚ)^15) = 22) := 
sorry

end num_zeros_1_div_15_pow_15_l365_365155


namespace product_of_invertible_numbers_mod_120_l365_365742

-- Definition of factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- The main theorem to prove
theorem product_of_invertible_numbers_mod_120 :
  let n := factorial 5 in
  n = 120 →
  let product := (∏ i in finset.filter (λ x, nat.gcd x n = 1) (finset.range n), i) in
  product % n = 1 :=
by
  intro n
  intro hn
  intro product
  sorry

end product_of_invertible_numbers_mod_120_l365_365742


namespace cos_sum_inequality_l365_365981

theorem cos_sum_inequality (n : ℕ) (x : ℝ) :
  (Finset.range (n + 1)).sum (λ k, |cos (2^k * x)|) ≥ n / (2 * Real.sqrt 2) :=
sorry

end cos_sum_inequality_l365_365981


namespace find_distance_l365_365259

theorem find_distance (T D : ℝ) 
  (h1 : D = 5 * (T + 0.2)) 
  (h2 : D = 6 * (T - 0.25)) : 
  D = 13.5 :=
by
  sorry

end find_distance_l365_365259


namespace line_through_points_l365_365799

/-- The line passing through points A(1, 1) and B(2, 3) satisfies the equation 2x - y - 1 = 0. -/
theorem line_through_points (x y : ℝ) :
  (∃ k : ℝ, k * (y - 1) = 2 * (x - 1)) → 2 * x - y - 1 = 0 :=
by
  sorry

end line_through_points_l365_365799


namespace stirling_duality_l365_365364

def StirlingSecondKind (N n : ℕ) : ℕ := sorry
def StirlingFirstKind (n M : ℕ) : ℕ := sorry

def KroneckerDelta (a b : ℕ) : ℕ :=
  if a = b then 1 else 0

theorem stirling_duality (N M : ℕ) :
  (∑ n in Finset.range (N + 1), StirlingSecondKind N n * StirlingFirstKind n M) = KroneckerDelta N M := 
sorry

end stirling_duality_l365_365364


namespace triangle_third_side_l365_365353

theorem triangle_third_side (A B C : ℝ) (a b c : ℝ) 
  (h1 : A = 3 * C) (h2 : b = 6) (h3 : c = 18) 
  (law_of_cosines : cos C = (a^2 + c^2 - b^2) / (2 * a * c))
  (law_of_sines : sin C = (3 * sin C - 4 * (sin C)^3)) :
  a = 72 := 
sorry

end triangle_third_side_l365_365353


namespace arithmetic_sequence_second_term_l365_365040

theorem arithmetic_sequence_second_term (a d : ℤ)
  (h1 : a + 11 * d = 11)
  (h2 : a + 12 * d = 14) :
  a + d = -19 :=
sorry

end arithmetic_sequence_second_term_l365_365040


namespace directrix_of_parabola_l365_365017

theorem directrix_of_parabola :
  (x : ℝ) → (p : ℝ) → (h1 : p = 4) →
  ∃ (d : ℝ), d = -1/16 ∧ ∀ y, y = p * x^2 → directrix y = d  :=
by
  sorry

end directrix_of_parabola_l365_365017


namespace annika_total_east_hike_distance_l365_365512

def annika_flat_rate : ℝ := 10 -- minutes per kilometer on flat terrain
def annika_initial_distance : ℝ := 2.75 -- kilometers already hiked east
def total_time : ℝ := 45 -- minutes
def uphill_rate : ℝ := 15 -- minutes per kilometer on uphill
def downhill_rate : ℝ := 5 -- minutes per kilometer on downhill
def uphill_distance : ℝ := 0.5 -- kilometer of uphill section
def downhill_distance : ℝ := 0.5 -- kilometer of downhill section

theorem annika_total_east_hike_distance :
  let total_uphill_time := uphill_distance * uphill_rate
  let total_downhill_time := downhill_distance * downhill_rate
  let time_for_uphill_and_downhill := total_uphill_time + total_downhill_time
  let time_available_for_outward_hike := total_time / 2
  let remaining_time_after_up_down := time_available_for_outward_hike - time_for_uphill_and_downhill
  let additional_flat_distance := remaining_time_after_up_down / annika_flat_rate
  (annika_initial_distance + additional_flat_distance) = 4 :=
by
  sorry

end annika_total_east_hike_distance_l365_365512


namespace complex_exponentiation_problem_l365_365598

theorem complex_exponentiation_problem (x y : ℝ) (i : ℂ) (h_i : i^2 = -1) (h_cond : y * i - x = -1 + i) :
  (1 - i) ^ (x + y) = -2 * i := by
  sorry

end complex_exponentiation_problem_l365_365598


namespace probability_not_rel_prime_50_l365_365850

theorem probability_not_rel_prime_50 : 
  let n := 50;
  let non_rel_primes_count := n - Nat.totient 50;
  let total_count := n;
  let probability := (non_rel_primes_count : ℚ) / (total_count : ℚ);
  probability = 3 / 5 :=
by
  sorry

end probability_not_rel_prime_50_l365_365850


namespace min_questions_to_find_phone_number_min_questions_to_find_phone_number_is_17_l365_365451

theorem min_questions_to_find_phone_number : 
  ∃ n : ℕ, ∀ (N : ℕ), (N = 100000 → 2 ^ n ≥ N) ∧ (2 ^ (n - 1) < N) := sorry

-- In simpler form, since log_2(100000) ≈ 16.60965, we have:
theorem min_questions_to_find_phone_number_is_17 : 
  ∀ (N : ℕ), (N = 100000 → 17 = Nat.ceil (Real.logb 2 100000)) := sorry

end min_questions_to_find_phone_number_min_questions_to_find_phone_number_is_17_l365_365451


namespace tv_selection_count_l365_365766

/-- Total number of different ways to select 3 TV sets from 4 Type A and 5 Type B TV sets,
    such that at least one TV of each type is included, is 70. -/
theorem tv_selection_count :
  (∑ (i : ℕ) in finset.range 4, (nat.choose 4 i) * (nat.choose 5 (3 - i))) - (nat.choose 4 3 + nat.choose 5 3) = 70 :=
by sorry

end tv_selection_count_l365_365766


namespace domain_of_f_l365_365178

open Set

def f (x : ℝ) : ℝ := (x + 1) / (x^3 + 3 * x^2 - 4 * x)

theorem domain_of_f :
  {x : ℝ | f x = (x + 1) / (x^3 + 3 * x^2 - 4 * x)} =
  {x : ℝ | x < -4} ∪ {x : ℝ | -4 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x < 1} ∪ {x : ℝ | 1 < x} :=
by
  sorry

end domain_of_f_l365_365178


namespace ratio_of_new_marks_l365_365385

theorem ratio_of_new_marks (avg1 avg2 : ℕ) (n : ℕ) (R : ℝ)
    (h1 : avg1 = 36) 
    (h2 : avg2 = 72) 
    (h3 : n = 11)
    (h4 : R = (avg2 : ℝ) / avg1) :
    (R = 2) :=
by
  rw [h1, h2, h3] at h4
  rw [h4]
  norm_num
  sorry

end ratio_of_new_marks_l365_365385


namespace consecutive_composite_l365_365524

theorem consecutive_composite (n : ℕ) (h : n ≥ 2) : 
  ∃ seq : ℕ → ℕ, (∀ k : ℕ, 2 ≤ k ∧ k ≤ n → ¬ nat.prime (seq k)) ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ n → seq k = n! + k) :=
by
  sorry

end consecutive_composite_l365_365524


namespace triangle_area_is_9sqrt2_l365_365577

noncomputable def triangle_area_with_given_medians_and_angle (CM BN : ℝ) (angle_BKM : ℝ) : ℝ :=
  let centroid_division_ratio := (2.0 / 3.0)
  let BK := centroid_division_ratio * BN
  let MK := (1.0 / 3.0) * CM
  let area_BKM := (1.0 / 2.0) * BK * MK * Real.sin angle_BKM
  6.0 * area_BKM

theorem triangle_area_is_9sqrt2 :
  triangle_area_with_given_medians_and_angle 6 4.5 (Real.pi / 4) = 9 * Real.sqrt 2 :=
by
  sorry

end triangle_area_is_9sqrt2_l365_365577


namespace g_675_eq_42_l365_365744

-- Define the function g on positive integers
def g : ℕ → ℕ := sorry

-- State the conditions
axiom g_multiplicative : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g_15 : g 15 = 18
axiom g_45 : g 45 = 24

-- The theorem we want to prove
theorem g_675_eq_42 : g 675 = 42 := 
by 
  sorry

end g_675_eq_42_l365_365744


namespace number_of_arrangements_l365_365684

def students := ["male", "female", "female", "female"]
def adjacent_different (s1 s2 : String) : Prop :=
  s1 ≠ s2

def A := "male"
def B := "female"

theorem number_of_arrangements : 
  (∃ (arr : List String), 
    ∀ i, (i < arr.length - 1) → adjacent_different (arr.nth i).getD "" (arr.nth (i + 1)).getD "") ∧ 
    (∃ j, (arr.nth j).getD "" = A ∧ (arr.nth (j + 1)).getD "" = B ∨ (arr.nth j).getD "" = B ∧ (arr.nth (j + 1)).getD "" = A) ∧
    ∀ k, (k = 0 ∨ k = arr.length - 1) → (arr.nth k).getD "" ≠ A ∧ (arr.nth k).getD "" ≠ B
  ) = 16 := 
sorry

end number_of_arrangements_l365_365684


namespace mans_rate_in_still_water_l365_365869

theorem mans_rate_in_still_water (Vm Vs : ℝ) (h1 : Vm + Vs = 14) (h2 : Vm - Vs = 4) : Vm = 9 :=
by
  sorry

end mans_rate_in_still_water_l365_365869


namespace words_per_hour_and_total_days_l365_365894

/-- Given that a novel contains 50,000 words, the author completed it in 100 hours, and she wrote for 5 hours each day, 
prove that the author wrote on average 500 words per hour and it took a total of 20 days. -/
theorem words_per_hour_and_total_days :
  (total_words : ℕ) (total_hours : ℕ) (hours_per_day : ℕ)
  (h1 : total_words = 50000)
  (h2 : total_hours = 100)
  (h3 : hours_per_day = 5) :
  (words_per_hour : ℕ) (total_days : ℕ)
  (h4 : words_per_hour = total_words / total_hours)
  (h5 : total_days = total_hours / hours_per_day)
  (h4' : words_per_hour = 500)
  (h5' : total_days = 20) :=
by {
  have h4 : 50000 / 100 = 500 := by norm_num,
  have h5 : 100 / 5 = 20 := by norm_num,
  exact ⟨h4, h5⟩,
  sorry
}

end words_per_hour_and_total_days_l365_365894


namespace find_a_l365_365593

variable (a : ℝ) (h_pos : a > 0) (h_integral : ∫ x in 0..a, (2 * x - 2) = 3)

theorem find_a : a = 3 :=
by sorry

end find_a_l365_365593


namespace average_seq_13_to_52_l365_365845

-- Define the sequence of natural numbers from 13 to 52
def seq : List ℕ := List.range' 13 52

-- Define the average of a list of natural numbers
def average (xs : List ℕ) : ℚ := (xs.sum : ℚ) / xs.length

-- Define the specific set of numbers and their average
theorem average_seq_13_to_52 : average seq = 32.5 := 
by 
  sorry

end average_seq_13_to_52_l365_365845


namespace ratio_of_speeds_l365_365465

theorem ratio_of_speeds (v_A v_B : ℝ)
  (h₀ : 4 * v_A = abs (600 - 4 * v_B))
  (h₁ : 9 * v_A = abs (600 - 9 * v_B)) :
  v_A / v_B = 2 / 3 :=
sorry

end ratio_of_speeds_l365_365465


namespace sum_of_possible_values_l365_365743

noncomputable def problem (x y : ℝ) : Prop :=
  xy - (3 * x) / (y^2) - (3 * y) / (x^2) = 7

theorem sum_of_possible_values (x y : ℝ) (h : problem x y) :
  (x - 2) * (y - 2) = 1 :=
sorry

end sum_of_possible_values_l365_365743


namespace min_value_on_interval_l365_365623

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ x

theorem min_value_on_interval (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) (h_max : ∀ x ∈ set.Icc (-2 : ℝ) 1, f a x ≤ 4 ∧ ∃ y ∈ set.Icc (-2 : ℝ) 1, f a y = 4) :
  ∃ m : ℝ, (m = (1 / 16) ∧ 1 < a) ∨ (m = (1 / 2) ∧ 0 < a ∧ a < 1) :=
by
  sorry


end min_value_on_interval_l365_365623


namespace original_number_l365_365861

theorem original_number (x : ℝ) (hx : 1000 * x = 9 * (1 / x)) : 
  x = 3 * (Real.sqrt 10) / 100 :=
by
  sorry

end original_number_l365_365861


namespace minimum_guests_l365_365357

theorem minimum_guests (x : ℕ) : (120 + 18 * x > 250 + 15 * x) → (x ≥ 44) := by
  intro h
  sorry

end minimum_guests_l365_365357


namespace geometric_sequence_abc_target_l365_365236

variable (a b c : ℝ)

def is_geometric_sequence (a b c : ℝ) (u v : ℝ) : Prop :=
  u * v = a * a ∧ b^2 = u * v ∧ c = (u * v) ^ (1/2)

theorem geometric_sequence_abc_target :
  is_geometric_sequence a b c (-1) (-2) ∧ a * c = 2 ∧ b^2 = 2 
  → a * b * c = -2 * (2)^(1/2) := by
  intro h,
  sorry

end geometric_sequence_abc_target_l365_365236


namespace rectangle_perimeter_l365_365119

theorem rectangle_perimeter (breadth length : ℝ) (h1 : length = 3 * breadth) (h2 : length * breadth = 147) : 2 * length + 2 * breadth = 56 :=
by
  sorry

end rectangle_perimeter_l365_365119


namespace problem_l365_365655

noncomputable theory
open_locale classical

def sequences (a b : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, (∀ x, x^2 - b n * x + 2^n = 0) → (x = a n ∨ x = a (n+1))

theorem problem (a b : ℕ → ℝ) (h : sequences a b) : b 10 = 4 :=
sorry

end problem_l365_365655


namespace rhombus_perimeter_is_52_l365_365794

-- Define the problem setup with the measurements of the diagonals.
def diagonals : ℝ × ℝ := (24, 10)

-- Define a function to compute the perimeter of a rhombus given its diagonals.
noncomputable def rhombus_perimeter (d1 d2 : ℝ) : ℝ :=
  let side := Math.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side

-- The theorem we want to prove.
theorem rhombus_perimeter_is_52 : rhombus_perimeter 24 10 = 52 := by
  sorry

end rhombus_perimeter_is_52_l365_365794


namespace tom_PIN_permutations_l365_365836

theorem tom_PIN_permutations : 
  ∀ (digits : Finset ℕ), digits = {3, 5, 7, 9} → digits.card = 4 → 
  (digits = {3, 5, 7, 9} →
  (∃ n, n = digits.card ∧ n.factorial = 24)) :=
begin
  intros digits h1 h2 h3,
  dsimp at h2,
  use digits.card,
  split,
  { exact h2, },
  { norm_num, },
end

end tom_PIN_permutations_l365_365836


namespace largest_possible_value_l365_365425

theorem largest_possible_value (a : ℕ → ℕ) (h : ∀ n, a n ≠ a (n + 12))
  (distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (neighbors : ∀ n, (a (n + 1) = a n + 10 ∨ a (n + 1) = a n + 7)) :
  ∃ m, m ≤ 58 ∧ ∀ n, a n ≤ m :=
begin
  sorry
end

end largest_possible_value_l365_365425


namespace koschei_coins_l365_365310

theorem koschei_coins :
  ∃ a : ℕ, (a % 10 = 7) ∧ (a % 12 = 9) ∧ (300 ≤ a) ∧ (a ≤ 400) ∧ a = 357 :=
by 
  sorry

end koschei_coins_l365_365310


namespace sum_of_solutions_f_eq_1_l365_365333

def f (x : ℝ) : ℝ :=
if x ≤ 2 then 2 * x + 4
else x / 3 - 1

theorem sum_of_solutions_f_eq_1 : (∑ x in {x | f x = 1}.toFinset, x) = 9 / 2 := by
  sorry

end sum_of_solutions_f_eq_1_l365_365333


namespace triangle_area_correct_l365_365969

def vertexA : ℝ × ℝ × ℝ := (1, 8, 11)
def vertexB : ℝ × ℝ × ℝ := (0, 7, 7)
def vertexC : ℝ × ℝ × ℝ := (-4, 12, 7)

noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def area (a b c : ℝ) (s : ℝ) : ℝ :=
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def ABC_area : ℝ :=
  let AB := distance vertexA vertexB
  let AC := distance vertexA vertexC
  let BC := distance vertexB vertexC
  in area AB AC BC (semiperimeter AB AC BC)

theorem triangle_area_correct :
  ABC_area = Real.sqrt (
    let s := (3 * Real.sqrt 2 + Real.sqrt 41 + Real.sqrt 57) / 2
    s * (s - 3 * Real.sqrt 2) * (s - Real.sqrt 41) * (s - Real.sqrt 57)
  ) := sorry

end triangle_area_correct_l365_365969


namespace unique_triple_solution_l365_365962

theorem unique_triple_solution (x y z : ℝ) :
  (1 + x^4 ≤ 2 * (y - z)^2) →
  (1 + y^4 ≤ 2 * (z - x)^2) →
  (1 + z^4 ≤ 2 * (x - y)^2) →
  (x = 1 ∧ y = 0 ∧ z = -1) :=
sorry

end unique_triple_solution_l365_365962


namespace books_sold_on_thursday_l365_365304

-- Define the given conditions as constants
def initial_stock : ℕ := 1100
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def friday_sales : ℕ := 135
def percentage_not_sold : ℚ := 63.45

-- Define the calculated number of books not sold
def books_not_sold : ℕ := (percentage_not_sold * initial_stock : ℚ).toNat

-- Define the total number of books sold
def total_books_sold : ℕ := initial_stock - books_not_sold

-- Define the number of books sold from Monday to Wednesday and Friday
def other_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + friday_sales

-- Main theorem statement: proving the books sold on Thursday
theorem books_sold_on_thursday : total_books_sold - other_sales = 78 :=
by
  -- Insert detailed proof here
  sorry

end books_sold_on_thursday_l365_365304


namespace mary_fruits_left_l365_365340

-- Conditions as definitions:
def mary_bought_apples : ℕ := 14
def mary_bought_oranges : ℕ := 9
def mary_bought_blueberries : ℕ := 6

def mary_ate_apples : ℕ := 1
def mary_ate_oranges : ℕ := 1
def mary_ate_blueberries : ℕ := 1

-- The problem statement:
theorem mary_fruits_left : 
  (mary_bought_apples - mary_ate_apples) + 
  (mary_bought_oranges - mary_ate_oranges) + 
  (mary_bought_blueberries - mary_ate_blueberries) = 26 := by
  sorry

end mary_fruits_left_l365_365340


namespace box_a_defective_probability_box_b_good_probability_after_transfer_l365_365518

theorem box_a_defective_probability : 
  let total_selections := Nat.choose 8 2,
      defective_selections := Nat.choose 3 2 in
  (defective_selections / total_selections : ℚ) = 3 / 28 :=
by sorry

theorem box_b_good_probability_after_transfer :
  let total_selections := Nat.choose 8 2,
      good_selections_from_a := Nat.choose 5 2,
      mixed_selections_from_a := 5 * 3,
      defective_selections_from_a := Nat.choose 3 2,

      P_B1 := (good_selections_from_a / total_selections : ℚ),
      P_B2 := (mixed_selections_from_a / total_selections : ℚ),
      P_B3 := (defective_selections_from_a / total_selections : ℚ),

      P_A_given_B1 := 2 / 3,
      P_A_given_B2 := 5 / 9,
      P_A_given_B3 := 4 / 9,

      P_A := P_B1 * P_A_given_B1 + P_B2 * P_A_given_B2 + P_B3 * P_A_given_B3 in
  P_A = 7 / 12 :=
by sorry

end box_a_defective_probability_box_b_good_probability_after_transfer_l365_365518


namespace students_in_fourth_section_l365_365904

/-- 
A school has 65 students in one section of chemistry in class X, 35 students in the second section, 
45 students in the third section, and some students in the fourth section. The mean marks obtained 
in the chemistry test are 50, 60, 55, and 45 respectively for the 4 sections. The overall average 
of marks per student is 51.95. Prove that the number of students in the fourth section is 42. 
-/
theorem students_in_fourth_section :
  ∃ x : ℕ, (65 + 35 + 45 + x) ≠ 0 ∧
  (65 * 50 + 35 * 60 + 45 * 55 + x * 45) / (65 + 35 + 45 + x) = 51.95 ∧
  x = 42 := by
  sorry

end students_in_fourth_section_l365_365904


namespace sum_powers_of_7_mod_5_l365_365851

theorem sum_powers_of_7_mod_5 : 
  (Finset.range 11).sum (λ n, 7^n % 5) % 5 = 2 := 
by sorry

end sum_powers_of_7_mod_5_l365_365851


namespace premium_amount_l365_365504

def original_value : ℝ := 87500
def insured_rate : ℝ := 4/5
def premium_rate : ℝ := 0.013

theorem premium_amount (V : ℝ) (IR : ℝ) (PR : ℝ) : V = original_value → IR = insured_rate → PR = premium_rate → PR * (IR * V) = 910 :=
by
  intros hV hIR hPR
  -- The proof of the theorem goes here
  sorry

end premium_amount_l365_365504


namespace maxwell_walking_speed_l365_365752

theorem maxwell_walking_speed :
  ∃ v : ℝ, (∀ (d home_distance maxwell_time brad_speed brad_time : ℝ),
    home_distance = 14 ∧ brad_speed = 6 ∧ maxwell_time = 2 ∧ brad_time = 1 →
    2 * v + brad_speed * brad_time = home_distance) ∧ v = 4 :=
begin
  sorry
end

end maxwell_walking_speed_l365_365752


namespace mary_fruits_left_l365_365343

theorem mary_fruits_left (apples_initial oranges_initial blueberries_initial : ℕ)
  (apples_eaten oranges_eaten blueberries_eaten : ℕ) :
  apples_initial = 14 →
  oranges_initial = 9 →
  blueberries_initial = 6 →
  apples_eaten = 1 →
  oranges_eaten = 1 →
  blueberries_eaten = 1 →
  (apples_initial - apples_eaten) + (oranges_initial - oranges_eaten) + (blueberries_initial - blueberries_eaten) = 26 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end mary_fruits_left_l365_365343


namespace kolakoski_13_to_20_l365_365024

noncomputable def kolakoski : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 2
| (n+3) :=
  if H : (kolakoski n = 1) then
    if HH : (kolakoski (n+1) = kolakoski (n+2)) then
      if HHH : (kolakoski (n+2) = 1) then 2 else 1
    else if HHH : (kolakoski (n+1) = 1 ∧ kolakoski (n+2) = 2) then 2 else 1
  else if HH : (kolakoski n = 2) then
    if HHH : (kolakoski (n+1) = kolakoski (n+2)) then
      if HHHH : (kolakoski (n+2) = 1) then 2 else 1
    else if HHHH : (kolakoski (n+1) = 2 ∧ kolakoski (n+2) = 1) then 1 else 2
  else 0 -- should not reach

theorem kolakoski_13_to_20 :
  (kolakoski 12 = 2 ∧ 
   kolakoski 13 = 2 ∧ 
   kolakoski 14 = 1 ∧ 
   kolakoski 15 = 1 ∧ 
   kolakoski 16 = 2 ∧ 
   kolakoski 17 = 1 ∧ 
   kolakoski 18 = 1 ∧ 
   kolakoski 19 = 2) := 
sorry

end kolakoski_13_to_20_l365_365024


namespace product_of_squares_gt_half_l365_365332

theorem product_of_squares_gt_half (n : ℕ) (p : Fin n → ℕ) (h_distinct : Function.Injective p) (h_gt_one : ∀ i, p i > 1) :
  (∏ i, (1 - 1 / (p i)^2)) > (1 / 2) :=
sorry

end product_of_squares_gt_half_l365_365332


namespace area_of_rhombus_is_140_l365_365013

def d1 : ℝ := 14
def d2 : ℝ := 20
def area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem area_of_rhombus_is_140 :
  area d1 d2 = 140 := by
  sorry

end area_of_rhombus_is_140_l365_365013


namespace tower_count_with_blue_top_l365_365885

/-- A child has 2 red cubes, 4 blue cubes, and 3 green cubes. The task is to find the number of 
different towers with a height of 8 cubes, where a blue cube is always at the top. --/
def num_towers_with_blue_top (red blue green total height top : ℕ) : ℕ :=
  if top > 0 then 
    let remaining := height - 1
    in nat.factorial remaining / (nat.factorial (blue - 1) * nat.factorial red * nat.factorial green)
  else 0

theorem tower_count_with_blue_top 
  (red blue green total height top : ℕ) 
  (h_red : red = 2)
  (h_blue : blue = 4)
  (h_green : green = 3)
  (h_total : total = red + blue + green)
  (h_height : height = 8)
  (h_top : top = 1) :
  num_towers_with_blue_top red blue green total height top = 210 :=
begin
  -- Definitions according to the conditions
  rw [h_red, h_blue, h_green, h_total, h_height, h_top],
  -- Substitution in the definition
  unfold num_towers_with_blue_top,
  -- Calculations as per the multinomial coefficient
  have : 7.factorial = 5040 := rfl,
  have : (3.factorial * 2.factorial * 2.factorial) = 24 := rfl,
  rw [this],
  -- Final calculation
  norm_num,
end

end tower_count_with_blue_top_l365_365885


namespace sampling_method_is_systematic_l365_365270

def conveyor_belt_sampling (interval: ℕ) (product_picking: ℕ → ℕ) : Prop :=
  ∀ (n: ℕ), product_picking n = n * interval

theorem sampling_method_is_systematic
  (interval: ℕ)
  (product_picking: ℕ → ℕ)
  (h: conveyor_belt_sampling interval product_picking) :
  interval = 30 → product_picking = systematic_sampling := 
sorry

end sampling_method_is_systematic_l365_365270


namespace euler_product_identity_l365_365953

noncomputable def z1 := complex.exp (complex.I * (real.pi / 3))
noncomputable def z2 := complex.exp (complex.I * (real.pi / 6))

theorem euler_product_identity : z1 * z2 = complex.I := by
  sorry -- Proof to be filled in

end euler_product_identity_l365_365953


namespace sum_of_cubes_mod_5_l365_365583

theorem sum_of_cubes_mod_5 : 
  ((∑ n in Finset.range 151, n^3) % 5) = 0 := 
  sorry

end sum_of_cubes_mod_5_l365_365583


namespace roots_of_polynomial_l365_365972

theorem roots_of_polynomial :
  ∀ x : ℝ, (2 * x^3 - 3 * x^2 - 13 * x + 10) * (x - 1) = 0 → x = 1 :=
by
  sorry

end roots_of_polynomial_l365_365972


namespace arithmetic_sequence_properties_sum_first_n_b1_sum_first_n_b2_sum_first_n_b3_l365_365279

variable (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ)

noncomputable def a (n : ℕ) : ℕ := 2n + 1

noncomputable def S (n : ℕ) : ℕ := n^2 + 2 * n

theorem arithmetic_sequence_properties (n : ℕ) :
  a 4 * a 4 = a 1 * (a 7 + 12) ∧
  S 3 = 15 :=
sorry

theorem sum_first_n_b1 (n : ℕ) (h : ∀ m, b m = (S m) / m + 2 ^ (a m)) : 
  T n = (n^2 + 5 * n) / 2 + (2 * 4^(n + 1) - 8) / 3 :=
sorry

theorem sum_first_n_b2 (n : ℕ) (h : ∀ m, b m = 1 / (S m)) : 
  T n = 3 / 4 - (2 * n + 3) / (2 * n^2 + 6 * n + 4) :=
sorry

theorem sum_first_n_b3 (n : ℕ) (h : ∀ m, b m = (a m - 1) * 2 ^ (m - 1)) : 
  T n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end arithmetic_sequence_properties_sum_first_n_b1_sum_first_n_b2_sum_first_n_b3_l365_365279


namespace circle_through_I_B_and_C_l365_365735

theorem circle_through_I_B_and_C (ABC : Triangle) (I : Point) (AI : Line)
  (D : Point) (circABC : Circle) (circD : Circle) (J : Point) :
  I = incenter ABC →
  AI = angle_bisector (angle BAC) →
  D ∈ circABC ∧ AI.intersects_def D →
  circD.center = D ∧ I ∈ circD →
  J ∈ AI ∧ I ∈ circABC ∧ B ∈ circD ∧ C ∈ circD →
  (J = excenter_center ABC A) := 
sorry

end circle_through_I_B_and_C_l365_365735


namespace train_speed_45_kmph_l365_365909

variable (length_train length_bridge time_passed : ℕ)

def total_distance (length_train length_bridge : ℕ) : ℕ :=
  length_train + length_bridge

def speed_m_per_s (length_train length_bridge time_passed : ℕ) : ℚ :=
  (total_distance length_train length_bridge) / time_passed

def speed_km_per_h (length_train length_bridge time_passed : ℕ) : ℚ :=
  (speed_m_per_s length_train length_bridge time_passed) * 3.6

theorem train_speed_45_kmph :
  length_train = 360 → length_bridge = 140 → time_passed = 40 → speed_km_per_h length_train length_bridge time_passed = 45 := 
by
  sorry

end train_speed_45_kmph_l365_365909


namespace koschei_coin_count_l365_365307

theorem koschei_coin_count (a : ℕ) :
  (a % 10 = 7) ∧
  (a % 12 = 9) ∧
  (300 ≤ a ∧ a ≤ 400) →
  a = 357 :=
sorry

end koschei_coin_count_l365_365307


namespace arc_length_of_given_function_l365_365877

noncomputable def arc_length 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h : a ≤ b) : ℝ :=
∫ x in a..b, sqrt (1 + (deriv f x)^2)

noncomputable def given_function (x : ℝ) : ℝ :=
  2 + arcsin(sqrt(x)) + sqrt(x - x^2)

theorem arc_length_of_given_function : 
  arc_length given_function (1/4) 1 (by norm_num) = 1 := 
sorry

end arc_length_of_given_function_l365_365877


namespace max_sum_of_progression_l365_365378

-- Define the arithmetic progression sequence
def arithmetic_progression (a d n : ℤ) : ℤ := a + (n - 1) * d

-- Define the specific sequence given a1, a2, a3, a4, a5 with a3 = 13
def a_seq (d : ℤ) (n : ℕ) : ℤ :=
  let a_1 := 13 - 2 * d in
  arithmetic_progression a_1 d n 

-- The goal is to show the maximum value given the conditions
theorem max_sum_of_progression : 
  ∃ d : ℤ, (2 * d < 13 ∧ d > 0) → 
    a_seq d (a_seq d 1) + a_seq d (a_seq d 2) + a_seq d (a_seq d 3) + a_seq d (a_seq d 4) + a_seq d (a_seq d 5) = 365 :=
by {
  -- Proof goes here
  sorry
}

end max_sum_of_progression_l365_365378


namespace sum_eq_9_div_64_l365_365549

noncomputable def double_sum : ℝ := ∑' (n : ℕ) in (set.Ici 2 : set ℕ), ∑' (k : ℕ) in set.Ico 1 n, (k : ℝ) / 3^(n + k)

theorem sum_eq_9_div_64 : double_sum = 9 / 64 := 
by 
sorry

end sum_eq_9_div_64_l365_365549


namespace trig_identity_l365_365592

theorem trig_identity (θ : ℝ) (h : sin θ + 2 * cos θ = 0) : (1 + sin (2 * θ)) / (cos θ)^2 = 1 := 
by
  -- Proof goes here
  sorry

end trig_identity_l365_365592


namespace rhombus_perimeter_is_52_l365_365795

-- Define the problem setup with the measurements of the diagonals.
def diagonals : ℝ × ℝ := (24, 10)

-- Define a function to compute the perimeter of a rhombus given its diagonals.
noncomputable def rhombus_perimeter (d1 d2 : ℝ) : ℝ :=
  let side := Math.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side

-- The theorem we want to prove.
theorem rhombus_perimeter_is_52 : rhombus_perimeter 24 10 = 52 := by
  sorry

end rhombus_perimeter_is_52_l365_365795


namespace constant_sum_of_reciprocals_l365_365729

theorem constant_sum_of_reciprocals (c : ℝ) (D : ℝ × ℝ) (A B : ℝ × ℝ)
  (hD : D = (0, c + 1))
  (h_parabola_A : A.2 = A.1 ^ 2)
  (h_parabola_B : B.2 = B.1 ^ 2)
  (h_line_through_D : ∀ (m : ℝ), ∃ (x : ℝ), D.2 = m * D.1 + c + 1)
  (h_AD_BD : ∀ (A B : ℝ × ℝ), ∃ (AD BD : ℝ), AD = real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) ∧ BD = real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)) : 
  (1 / real.sqrt(A.1^2 + (A.2 - D.2)^2) + 1 / real.sqrt(B.1^2 + (B.2 - D.2)^2)) = 4 :=
by
  sorry

end constant_sum_of_reciprocals_l365_365729


namespace final_weight_after_drying_l365_365680

/-- Conditions of the initial problem --/
def initialWeight : ℕ := 1000 -- in kg
def initialWaterContent : ℕ := 99 -- as a percentage
def waterContentDecrease : ℕ := 4 -- as a percentage

/-- Theorem stating the final weight of the tomatoes after drying --/
theorem final_weight_after_drying :
  let dryMass := (initialWeight * (100 - initialWaterContent)) / 100 in
  let finalWaterContent := initialWaterContent - waterContentDecrease in
  let finalTotalWeight := (100 * dryMass) / (100 - finalWaterContent) in
  finalTotalWeight = 200 :=
by
  sorry

end final_weight_after_drying_l365_365680


namespace range_m_l365_365207

variable {m : ℝ}

def p : Prop := ∀ x, (x^2 + m * x + 1 = 0) → (x < 0)
def q : Prop := ∀ x,  ¬ (4 * x^2 + 4 * (m - 2) * x + 1 = 0)

theorem range_m (h : (p ∨ q) ∧ ¬(p ∧ q)) :
  m ∈ set.Ioc 1 2 ∪ set.Ici 3 :=
  sorry

end range_m_l365_365207


namespace total_blocks_to_ride_l365_365658

-- Constants representing the problem conditions
def rotations_per_block : ℕ := 200
def initial_rotations : ℕ := 600
def additional_rotations : ℕ := 1000

-- Main statement asserting the total number of blocks Greg wants to ride
theorem total_blocks_to_ride : 
  (initial_rotations / rotations_per_block) + (additional_rotations / rotations_per_block) = 8 := 
  by 
    sorry

end total_blocks_to_ride_l365_365658


namespace functional_relationship_l365_365214

-- Define the conditions
def directlyProportional (y x k : ℝ) : Prop :=
  y + 6 = k * (x + 1)

def specificCondition1 (x y : ℝ) : Prop :=
  x = 3 ∧ y = 2

-- State the theorem
theorem functional_relationship (k : ℝ) :
  (∀ x y, directlyProportional y x k) →
  specificCondition1 3 2 →
  ∀ x, ∃ y, y = 2 * x - 4 :=
by
  intro directProp
  intro specCond
  sorry

end functional_relationship_l365_365214


namespace modulus_of_complex_l365_365579

-- Definition of the complex number z = 2.2i / (1 + i)
def z : ℂ := (2.2 * complex.I) / (1 + complex.I)

-- Theorem to prove the modulus of the complex number z is sqrt(2)
theorem modulus_of_complex : complex.abs z = real.sqrt 2 := 
by
  -- This proof is omitted 
  sorry

end modulus_of_complex_l365_365579


namespace part1_part2_l365_365748

-- Definitions and conditions for the first problem.
def A (a : ℝ) : set ℝ := { x | abs (x - 2) < a }

theorem part1 (a : ℕ)
  (h1 : (3 / 2) ∈ A a)
  (h2 : (-1 / 2) ∉ A a)
  (h3 : ∀ x : ℝ, abs (x - 1) + abs (x - 3) ≥ a^2 + a) :
  a = 1 :=
sorry

-- Definitions and conditions for the second problem.
theorem part2 (a b : ℝ) 
  (h1 : a + b = 1) 
  (h2 : 0 < a) 
  (h3 : 0 < b) :
  ∃ a' b', 
  a' = (3 - Real.sqrt 3) / 2 ∧ 
  b' = (Real.sqrt 3 - 1) / 2 ∧ 
  (∀ a b : ℝ, a + b = 1 → 0 < a → 0 < b → (1 / (3 * b) + b / a) ≥ (1 + 2 * Real.sqrt 3) / 3) :=
sorry

end part1_part2_l365_365748


namespace quotient_surface_area_l365_365825

def radius_larger : ℝ := 6
def radius_smaller : ℝ := 3
def surface_area (r : ℝ) : ℝ := 4 * π * r^2

theorem quotient_surface_area :
  surface_area radius_larger / surface_area radius_smaller = 4 :=
by
  sorry

end quotient_surface_area_l365_365825


namespace double_sum_evaluation_l365_365539

theorem double_sum_evaluation : 
  (∑ n from 2 to ∞, ∑ k from 1 to (n - 1), k / (3 ^ (n + k))) = (6 / 25) :=
by sorry

end double_sum_evaluation_l365_365539


namespace log_product_log_expression_l365_365929

theorem log_product (h1 : log 2 25 = log 2 5 ^ 2) 
                    (h2 : log 3 4 = 2 * log 3 2)
                    (h3 : log 5 9 = log 5 3 ^ 2)
                    (h4 : log 2 5 * log 5 3 * log 3 2 = 1) :
  log 2 25 * log 3 4 * log 5 9 = 8 :=
by sorry

theorem log_expression (h1 : lg (32 / 49) = lg 32 - lg 49) 
                       (h2 : lg 32 = 5 * lg 2)
                       (h3 : lg 49 = 2 * lg 7)
                       (h4 : lg (sqrt 8) = (1/2) * lg 8)
                       (h5 : lg 8 = 3 * lg 2)
                       (h6 : lg (sqrt 245) = (1/2) * lg 245)
                       (h7 : lg 245 = lg 5 + 2 * lg 7)
                       (h8 : (1/2) * lg 2 + (1/2) * lg 5 = (1/2) * (lg 2 + lg 5)) :
  (1/2) * (lg (32 / 49)) - (4/3) * (lg (sqrt 8)) + lg (sqrt 245) = (1/2) :=
by sorry

end log_product_log_expression_l365_365929


namespace arithmetic_sequence_S9_l365_365691

theorem arithmetic_sequence_S9 (a_n : ℕ → ℕ) (n : ℕ) 
  (h_arithmetic_sequence : ∀ i j, a_n (i + j + 1) = a_n i + a_n j)
  (h_sum_mem_4_to_8 : a_n 3 + a_n 4 + a_n 5 + a_n 6 + a_n 7 = 25) : S 9 = 45 := 
sorry

end arithmetic_sequence_S9_l365_365691


namespace area_triangle_ABD_l365_365294

theorem area_triangle_ABD {A B C D : Type*}
  (h_triangle : Triangle A B C)
  (angle_ABC : angle A B C = 60)
  (AB : length A B = 120)
  (BC : length B C = y)
  (AC : length A C = 3 * y - 18)
  (AD_bisector : angle_bisector D A B C)
  : area (triangle A B D) = 30 * sqrt 3 * length B D :=
by sorry

end area_triangle_ABD_l365_365294


namespace Karlsson_can_repair_propeller_l365_365462

def blade_cost : ℕ := 120
def screw_cost : ℕ := 9
def discount_rate : ℝ := 0.2
def discount_threshold : ℕ := 250
def total_budget : ℕ := 360

theorem Karlsson_can_repair_propeller :
  let total_cost := (2 * blade_cost + 2 * screw_cost) + (blade_cost - (blade_cost * discount_rate).to_nat)
  in total_cost ≤ total_budget :=
by
  sorry

end Karlsson_can_repair_propeller_l365_365462


namespace area_locus_highest_points_l365_365940

noncomputable def area_of_locus_highest_points 
  (v g: ℝ) : ℝ :=
  let h : ℝ := v^2 / (8 * g)
  in (3 * π / 32) * (v^4 / g^2)

-- The problem statement in Lean
theorem area_locus_highest_points 
  (v g: ℝ) : area_of_locus_highest_points v g = (3 * π / 32) * (v^4 / g^2) :=
sorry

end area_locus_highest_points_l365_365940


namespace range_of_a_l365_365320

def A := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) := {x : ℝ | x^2 + 2*(a+1)*x + (a^2 -1) = 0}

theorem range_of_a (a : ℝ) :
  (A ∩ B a = B a) → (a = 1 ∨ a ≤ -1) :=
by
  sorry

end range_of_a_l365_365320


namespace trig_expression_equals_one_l365_365525

-- Definitions of the conditions
def sin_75 : ℝ := Real.sin (75 * Real.pi / 180)
def cos_15 : ℝ := Real.cos (15 * Real.pi / 180)
def sin_15 : ℝ := Real.sin (15 * Real.pi / 180)
def cos_75 : ℝ := Real.cos (75 * Real.pi / 180)

-- Stating the given conditions
axiom sin_75_eq_cos_15 : sin_75 = cos_15
axiom cos_75_eq_sin_15 : cos_75 = sin_15

-- The theorem to prove
theorem trig_expression_equals_one :
  (1 - 1 / cos_15) * (1 + 1 / sin_75) * (1 - 1 / sin_15) * (1 + 1 / cos_75) = 1 :=
by
  -- Proof omitted
  sorry

end trig_expression_equals_one_l365_365525


namespace remainder_of_number_divisor_l365_365486

-- Define the interesting number and the divisor
def number := 2519
def divisor := 9
def expected_remainder := 8

-- State the theorem to prove the remainder condition
theorem remainder_of_number_divisor :
  number % divisor = expected_remainder := by
  sorry

end remainder_of_number_divisor_l365_365486


namespace max_area_of_rectangle_with_perimeter_40_l365_365995

theorem max_area_of_rectangle_with_perimeter_40 :
  ∃ (A : ℝ), (A = 100) ∧ (∀ (length width : ℝ), 2 * (length + width) = 40 → length * width ≤ A) :=
by
  sorry

end max_area_of_rectangle_with_perimeter_40_l365_365995


namespace digits_to_left_condition_count_l365_365389

theorem digits_to_left_condition_count :
  let six_digits := {a | a ∈ {1, 2, 3, 4, 5, 6}}
  ∃! n : ℕ, n = 180 ∧ (∀ l : list ℕ, l.perm six_digits → 
                         list.nodup l ∧ 
                         list.length l = 6 ∧ 
                         list.index_of 1 l < list.index_of 2 l ∧ 
                         list.index_of 3 l < list.index_of 4 l → n)
:= sorry

end digits_to_left_condition_count_l365_365389


namespace koschei_coin_count_l365_365306

theorem koschei_coin_count (a : ℕ) :
  (a % 10 = 7) ∧
  (a % 12 = 9) ∧
  (300 ≤ a ∧ a ≤ 400) →
  a = 357 :=
sorry

end koschei_coin_count_l365_365306


namespace double_sum_eq_l365_365541

theorem double_sum_eq : 
  (∑ n in (finset.Ico 2 (⊤ : ℕ)), ∑ k in (finset.Ico 1 n), k / (3 : ℝ)^(n + k)) = (9 / 128 : ℝ) :=
sorry

end double_sum_eq_l365_365541


namespace find_number_l365_365872

theorem find_number (x : ℝ) (h: x - (3 / 5) * x = 58) : x = 145 :=
by {
  sorry
}

end find_number_l365_365872


namespace total_legs_in_room_l365_365826

def count_legs : Nat :=
  let tables_4_legs := 4 * 4
  let sofas_legs := 1 * 4
  let chairs_4_legs := 2 * 4
  let tables_3_legs := 3 * 3
  let tables_1_leg := 1 * 1
  let rocking_chair_legs := 1 * 2
  tables_4_legs + sofas_legs + chairs_4_legs + tables_3_legs + tables_1_leg + rocking_chair_legs

theorem total_legs_in_room : count_legs = 40 := by
  sorry

end total_legs_in_room_l365_365826


namespace type_of_graph_displays_trend_l365_365147

theorem type_of_graph_displays_trend :
  (∃ graph_type : Type, graph_type = "line graph") :=
sorry

end type_of_graph_displays_trend_l365_365147


namespace equal_lengths_of_incircle_touchpoints_l365_365189

theorem equal_lengths_of_incircle_touchpoints
  {A B C D E F G H : Point}
  (hABCD : convex_quadrilateral A B C D)
  (hAB_BC_CD_DA : AB A B > AB B C ∧ AB B C > AB C D ∧ AB C D > AB D A)
  (hE_touch : incircle_touch_point E (triangle A B D) (diagonal B D))
  (hF_touch : incircle_touch_point F (triangle B C D) (diagonal B D))
  (hH_touch : incircle_touch_point H (triangle A B C) (diagonal A C))
  (hG_touch : incircle_touch_point G (triangle A C D) (diagonal A C)) :
  distance E F = distance G H :=
sorry

end equal_lengths_of_incircle_touchpoints_l365_365189


namespace symmetric_graph_l365_365676

noncomputable def f (x : ℝ) : ℝ := 2 - Real.log x

theorem symmetric_graph (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x y : ℝ, g y = x ↔ y = f x) → (f = fun x => 2 - Real.log x) :=
by
  intro hyp
  have h1 : ∀ y : ℝ, g y = e^(2 - y) :=
  begin
    -- Here we transform the condition to the specific form g y = exp (2 - y)
    sorry
  end
  have h2 : ∀ y : ℝ, x = e^(2 - y) → y = 2 - Real.log x,
  from λ y, (fun x => by simpa [←Real.exp_eq_two_sub_log] using Real.inv_fun_eq' h1),
  -- Using this, we prove f x = 2 - Real.log x
  sorry

#check symmetric_graph

end symmetric_graph_l365_365676


namespace determinant_sum_is_34_l365_365937

-- Define matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![5, -2],
  ![3, 4]
]

def B : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1, 3],
  ![-1, 2]
]

-- Prove the determinant of the sum of A and B is 34
theorem determinant_sum_is_34 : Matrix.det (A + B) = 34 := by
  sorry

end determinant_sum_is_34_l365_365937


namespace average_monthly_balance_l365_365023

theorem average_monthly_balance :
  let january_balance := 200
  let february_balance := january_balance + 100
  let march_balance := february_balance - 50
  let april_balance := march_balance + 100
  let may_balance := april_balance
  let june_balance := may_balance - 100
  let total_balance := january_balance + february_balance + march_balance + april_balance + may_balance + june_balance
  let months := 6
  let average_balance := total_balance / months
  average_balance = 283.33 := by
  let january_balance := 200
  let february_balance := january_balance + 100
  let march_balance := february_balance - 50
  let april_balance := march_balance + 100
  let may_balance := april_balance
  let june_balance := may_balance - 100
  let total_balance := january_balance + february_balance + march_balance + april_balance + may_balance + june_balance
  let months := 6
  let average_balance := total_balance / months
  show average_balance = 283.33 from sorry

end average_monthly_balance_l365_365023


namespace felix_trees_chopped_l365_365956

theorem felix_trees_chopped (trees_per_sharpen : ℕ) (cost_per_sharpen : ℕ) (total_spent : ℕ) (trees_chopped : ℕ) :
  trees_per_sharpen = 13 →
  cost_per_sharpen = 5 →
  total_spent = 35 →
  trees_chopped = (total_spent / cost_per_sharpen) * trees_per_sharpen →
  trees_chopped ≥ 91 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  have : (35 / 5) * 13 = 91 := sorry
  rw this at h4
  exact le_of_eq h4

end felix_trees_chopped_l365_365956


namespace sum_of_squares_of_sequence_l365_365685

theorem sum_of_squares_of_sequence (n : ℕ) (a : ℕ → ℕ) (h : ∀ n, ∑ i in finset.range (n + 1), a i = 2^(n + 1) - 1) : 
  ∑ i in finset.range (n + 1), (a i)^2 = (4^(n + 1) - 1) / 3 := 
by
  sorry

end sum_of_squares_of_sequence_l365_365685


namespace significant_digits_side_length_of_area_l365_365030
noncomputable theory

def area : ℝ := 2.4896

def side_length (a : ℝ) : ℝ := Real.sqrt a

def significant_digits (x : ℝ) : ℕ :=
  x.toString.filter (λ c, c ≠ '.' ∧ c ≠ '0').length

theorem significant_digits_side_length_of_area :
  significant_digits (side_length area) = 5 :=
sorry

end significant_digits_side_length_of_area_l365_365030


namespace type_of_graph_displays_trend_l365_365148

theorem type_of_graph_displays_trend :
  (∃ graph_type : Type, graph_type = "line graph") :=
sorry

end type_of_graph_displays_trend_l365_365148


namespace pizza_slices_l365_365780

theorem pizza_slices (S : ℕ) (h1 : 2 * S - 0.25 * 2 * S - 0.5 * (2 * S - 0.25 * 2 * S) = 9) : S = 12 := 
sorry

end pizza_slices_l365_365780


namespace equal_roots_B_expression_l365_365130

theorem equal_roots_B_expression (k : ℝ) (B : ℝ) (h : k = 0.4444444444444444) :
  (∃ x : ℝ, (2*k*x^2 + B*x + 2 = 0) ∧ (B^2 - 4*2*k*2 = 0)) → B = 4*sqrt(k) :=
by
  sorry

end equal_roots_B_expression_l365_365130


namespace triangle_inequalities_l365_365361

-- Define the conditions and the statement to be proven
theorem triangle_inequalities (A B C D : ℝ) (x y z : ℝ)
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ D) (h4 : x = B - A) (h5 : y = C - A) (h6 : z = D - A)
  (h7 : ∀ (u v w : ℝ), u + v > w ∧ v + w > u ∧ w + u > v) :
  x < z / 2 ∧ y < x + z / 2 :=
begin
  sorry
end

end triangle_inequalities_l365_365361


namespace locus_C2_max_area_OAB_l365_365285

-- Define the conditions for part 1
def curve_C1 (ρ θ : ℝ) : Prop := ρ * (Real.cos θ) = 4
def OP (ρ ρ1 θ: ℝ) := ρ 
def OM (ρ1 θ: ℝ) := ρ1 * (Real.cos θ)
def condition_OM_OP (ρ ρ1 θ: ℝ) := (ρ * (4 / (Real.cos θ))) = 16
noncomputable def locus_P (ρ θ: ℝ) := 4 * (Real.cos θ)

-- Prove that the rectangular coordinate equation for C2
theorem locus_C2 (x y: ℝ) : (∃ ρ θ, curve_C1 ρ θ ∧ locus_P ρ θ ∧ ρ^2 = x^2 + y^2 ∧ ρ * (Real.cos θ) = x ∧ x ≠ 0 → (x - 2)^2 + y^2 = 4):=
  sorry

-- Define the conditions for part 2
def polar_point (ρ θ: ℝ) : ℝ×ℝ := (ρ, θ)
def point_B (ρ α: ℝ) (C2_on_B : locus_P ρ α) := (ρ, α)

-- Prove the maximum area of triangle OAB
theorem max_area_OAB : (∃ α, α = -π/12 ∧ ∀ A B: ℝ×ℝ, polar_point 2 (π/3) = A ∧ B = point_B (4 * (Real.cos α)) α C2_on_B → 2 + ℓt3/2) :=
  sorry

end locus_C2_max_area_OAB_l365_365285


namespace ratio_of_second_bidder_to_harry_first_bid_l365_365241

theorem ratio_of_second_bidder_to_harry_first_bid :
  let auction_start := 300 in
  let harry_first_bid_add := 200 in
  let harry_first_bid := auction_start + harry_first_bid_add in
  let third_bidder_bid := 3 * harry_first_bid in
  let harry_final_bid := 4000 in
  let difference := 1500 in
  let second_bidder_bid := third_bidder_bid - harry_first_bid in
  harry_final_bid - difference = third_bidder_bid →
  (second_bidder_bid = 1000 ∧ harry_first_bid = 500) →
  second_bidder_bid / harry_first_bid = 2 := 
by
  intros h1 h2,
  have h3 : harry_first_bid = 500 := by sorry,
  have h4 : second_bidder_bid = 1000 := by sorry,
  rw [h4, h3],
  norm_num,

end ratio_of_second_bidder_to_harry_first_bid_l365_365241


namespace tonya_needs_to_eat_more_l365_365419

-- Define the conditions in the problem
def ounces_per_hamburger : ℕ := 4
def ounces_eaten_last_year : ℕ := 84

-- Define a function to calculate hamburgers given ounces
def hamburgers_eaten (ounces : ℕ) (ounce_per_hamburger : ℕ) : ℕ :=
  ounces / ounce_per_hamburger

-- State the theorem
theorem tonya_needs_to_eat_more (ounces_per_hamburger ounces_eaten_last_year : ℕ) :
  hamburgers_eaten ounces_eaten_last_year ounces_per_hamburger + 1 = 22 := by
  sorry

end tonya_needs_to_eat_more_l365_365419


namespace largest_number_in_set_l365_365919

theorem largest_number_in_set :
  ∀ (a b c d : ℤ), (a ∈ [0, 2, -1, -2]) → (b ∈ [0, 2, -1, -2]) → (c ∈ [0, 2, -1, -2]) → (d ∈ [0, 2, -1, -2])
  → (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  → max (max a b) (max c d) = 2
  := 
by
  sorry

end largest_number_in_set_l365_365919


namespace sum_base6_l365_365915

theorem sum_base6 (a b : ℕ) (h₁ : a = 5) (h₂ : b = 23) : 
  let sum := Nat.ofDigits 6 [2, 3] + Nat.ofDigits 6 [5]
  in Nat.digits 6 sum = [2, 3] :=
by
  sorry

end sum_base6_l365_365915


namespace interval_monotonic_decrease_f_monotonic_decrease_g_min_value_m_l365_365225

-- Definitions of f(x) and g(x)
def f (x : ℝ) := x / real.log x
def g (x : ℝ) (m : ℝ) := f x - m * x

-- Part (I): Prove the intervals of monotonic decrease for function f(x)
theorem interval_monotonic_decrease_f :
  Ioo 0 1 ⊆ { x : ℝ | ∀ x, differentiable_at ℝ f x ∧ f' x < 0 } ∧ 
  Ioo 1 real.exp 1 ⊆ { x : ℝ | ∀ x, differentiable_at ℝ f x ∧ f' x < 0 } := 
sorry

-- Part (II): Prove that m ≥ 1/4 when g(x) is monotonically decreasing on (1, +∞)
theorem monotonic_decrease_g (m : ℝ) :
  (∀ x : ℝ, 1 < x → differentiable_at ℝ (g x m) x ∧ (g' x m) ≤ 0) → m ≥ 1/4 :=
sorry

-- Part (III): Prove that the minimum value of m is 1/2 - 1/(4e^2)
theorem min_value_m (x1 x2 : ℝ) (hx1 : x1 ∈ Icc (real.exp 1) (real.exp 2)) (hx2 : x2 ∈ Icc (real.exp 1) (real.exp 2)) :
  (m : ℝ) (m ≥ (g x1 m - diff g' x2 m)) → m ≥ 1/2 - 1/(4 * (real.exp 2)^2) :=
sorry

end interval_monotonic_decrease_f_monotonic_decrease_g_min_value_m_l365_365225


namespace remove_parentheses_correct_l365_365075

variable {a b c : ℝ}

theorem remove_parentheses_correct :
  -(a - b) = -a + b :=
by sorry

end remove_parentheses_correct_l365_365075


namespace initial_boys_count_l365_365037

theorem initial_boys_count (b : ℕ) (h1 : b + 10 - 4 - 3 = 17) : b = 14 :=
by
  sorry

end initial_boys_count_l365_365037


namespace ratio_of_areas_l365_365038

noncomputable def side_length_C := 24 -- cm
noncomputable def side_length_D := 54 -- cm
noncomputable def ratio_areas := (side_length_C / side_length_D) ^ 2

theorem ratio_of_areas : ratio_areas = 16 / 81 := sorry

end ratio_of_areas_l365_365038


namespace find_n_l365_365572

theorem find_n (n : ℕ)
  (h1 : n ≥ 2)
  (a : Fin (2*n) → ℕ)
  (h2 : ∀ i, 1 ≤ a i ∧ a i ≤ 2*n ∧ (∀ j, i ≠ j → a i ≠ a j))
  (h3 : (∑ i in Finset.range n, a(2*i) * a(2*i+1)) = a(2*n-2) * a(2*n-1)) :
  n = 2 := by
  sorry

end find_n_l365_365572


namespace proof_of_distance_l365_365416

noncomputable def distance_between_parallel_lines 
  (r : ℝ) (d : ℝ) (AB_length : ℝ) (CD_length : ℝ) (EF_length : ℝ) : Prop :=
  AB_length = 40 ∧ CD_length = 40 ∧ EF_length = 36 ∧
  (r^2 * 20 + r^2 * 20 = 40 * ((d / 2)^2 + 20^2)) ∧
  (r^2 * 18 + r^2 * 18 = 36 * ((3 * d / 2)^2 + 18^2)) ∧
  (d ≈ 1.46)

theorem proof_of_distance :
  ∀ r d, distance_between_parallel_lines r d 40 40 36 :=
sorry

end proof_of_distance_l365_365416


namespace equation_solution_l365_365774

noncomputable def solve_equation : Set ℝ := {x : ℝ | (3 * x + 2) / (x ^ 2 + 5 * x + 6) = 3 * x / (x - 1)
                                             ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ 1}

theorem equation_solution (r : ℝ) (h : r ∈ solve_equation) : 3 * r ^ 3 + 12 * r ^ 2 + 19 * r + 2 = 0 :=
sorry

end equation_solution_l365_365774


namespace sum_binom_squared_eq_binom_double_l365_365326

theorem sum_binom_squared_eq_binom_double (n : ℕ) : 
  (∑ k in Finset.range (n + 1), (Nat.choose n k)^2) = Nat.choose (2 * n) n := 
by 
  sorry

end sum_binom_squared_eq_binom_double_l365_365326


namespace sum_of_primes_l365_365629

theorem sum_of_primes (p1 p2 p3 : ℕ) (hp1 : Nat.Prime p1) (hp2 : Nat.Prime p2) (hp3 : Nat.Prime p3) 
    (h : p1 * p2 * p3 = 31 * (p1 + p2 + p3)) :
    p1 + p2 + p3 = 51 := by
  sorry

end sum_of_primes_l365_365629


namespace area_of_union_l365_365494

noncomputable theory

-- Definitions based on the given conditions
def side_length := 8
def radius := 12
def area_square := side_length ^ 2
def area_circle := Real.pi * radius ^ 2
def overlap := (1 / 4) * area_circle
def area_union := area_square + area_circle - overlap

-- The theorem stating the desired proof
theorem area_of_union (side_length radius : ℝ) (h_side : side_length = 8) (h_radius : radius = 12) :
  (side_length ^ 2 + Real.pi * radius ^ 2 - (1 / 4) * Real.pi * radius ^ 2) = 64 + 108 * Real.pi :=
by
  rw [h_side, h_radius]
  simp [area_square, area_circle, overlap, area_union]
  sorry

end area_of_union_l365_365494


namespace triangle_areas_ratio_l365_365703

variables {x y z : ℝ}

theorem triangle_areas_ratio :
  x + y + z = 3 / 4 →
  x^2 + y^2 + z^2 = 3 / 8 →
  (1 - (x * (1 - z) + y * (1 - x) + z * (1 - y))) = 3 / 32 :=
by
  intros h1 h2
  have h3 : (x + y + z)^2 = (3 / 4)^2 := by sorry
  have h4 : 2 * (x * y + y * z + z * x) = 3 / 16 := by sorry
  have h5 : x * y + y * z + z * x = 3 / 32 := by sorry
  calc
    1 - (x * (1 - z) + y * (1 - x) + z * (1 - y))
        = 1 - (x + y + z - (x * z + y * x + z * y)) : by sorry
    ... = 1 - (3 / 4 - 3 / 32) : by sorry
    ... = 1 - 3 / 4 + 3 / 32 : by sorry
    ... = 3 / 32 : by sorry

end triangle_areas_ratio_l365_365703


namespace solve_system_l365_365574

theorem solve_system (x y : ℚ) (h1 : 6 * x = -9 - 3 * y) (h2 : 4 * x = 5 * y - 34) : x = 1/2 ∧ y = -4 :=
by
  sorry

end solve_system_l365_365574


namespace common_tangent_intersects_x_axis_at_point_A_l365_365224

-- Define the ellipses using their equations
def ellipse_C1 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def ellipse_C2 (x y : ℝ) : Prop := (x - 2)^2 + 4 * y^2 = 1

-- The theorem stating the coordinates of the point where the common tangent intersects the x-axis
theorem common_tangent_intersects_x_axis_at_point_A :
  (∃ x : ℝ, (ellipse_C1 x 0 ∧ ellipse_C2 x 0) ↔ x = 4) :=
sorry

end common_tangent_intersects_x_axis_at_point_A_l365_365224


namespace cos_of_triangle_l365_365708

theorem cos_of_triangle (A B C : ℝ) (a b c : ℝ) (h1 : sin A = 2 * sin C) 
  (h2 : a * a = b * c) : cos A = - (Real.sqrt 2) / 4 :=
sorry

end cos_of_triangle_l365_365708


namespace ab_range_l365_365206

variable (a b : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variable (h_eq : a * b = a + b + 8)

theorem ab_range (h : a * b = a + b + 8) : 16 ≤ a * b :=
by sorry

end ab_range_l365_365206


namespace eccentricity_of_ellipse_l365_365941

theorem eccentricity_of_ellipse (p q : ℕ) (hp : Nat.Coprime p q) (z : ℂ) :
  ((z - 2) * (z^2 + 3 * z + 5) * (z^2 + 5 * z + 8) = 0) →
  (∃ p q : ℕ, Nat.Coprime p q ∧ (∃ e : ℝ, e^2 = p / q ∧ p + q = 16)) :=
by
  sorry

end eccentricity_of_ellipse_l365_365941


namespace koschei_coins_l365_365313

theorem koschei_coins :
  ∃ a : ℕ, a % 10 = 7 ∧ a % 12 = 9 ∧ 300 ≤ a ∧ a ≤ 400 ∧ a = 357 :=
by
  sorry

end koschei_coins_l365_365313


namespace solve_for_n_l365_365246

theorem solve_for_n : ∀ (n : ℕ), 4 ^ 8 = 16 ^ n → n = 4 :=
by
  intros n h,
  sorry

end solve_for_n_l365_365246


namespace binomial_identity_l365_365769

theorem binomial_identity (n k r s : ℤ) :
  Nat.choose n k * Nat.choose k r * Nat.choose r s = Nat.choose n s * Nat.choose (n - s) (r - s) * Nat.choose (n - r) (k - r) := 
sorry

end binomial_identity_l365_365769


namespace isosceles_triangle_perpendicular_sum_eq_altitude_l365_365363

theorem isosceles_triangle_perpendicular_sum_eq_altitude
  (A B C D E F : Point)
  (h₁ : is_isosceles_triangle A B C)
  (h₂ : is_base BC A)
  (h₃ : is_point_on_line D BC)
  (h₄ : is_perpendicular DE D AB)
  (h₅ : is_perpendicular DF D AC)
  : distance DE + distance DF = altitude A BC := 
sorry

end isosceles_triangle_perpendicular_sum_eq_altitude_l365_365363


namespace structure_of_S_l365_365724

def set_S (x y : ℝ) : Prop :=
  (5 >= x + 1 ∧ 5 >= y - 5) ∨
  (x + 1 >= 5 ∧ x + 1 >= y - 5) ∨
  (y - 5 >= 5 ∧ y - 5 >= x + 1)

theorem structure_of_S :
  ∃ (a b c : ℝ), set_S x y ↔ (y <= x + 6) ∧ (x <= 4) ∧ (y <= 10) 
:= sorry

end structure_of_S_l365_365724


namespace oranges_in_bin_l365_365463

theorem oranges_in_bin (initial_oranges : ℕ) (oranges_thrown_away : ℕ) (oranges_added : ℕ) 
  (h1 : initial_oranges = 50) (h2 : oranges_thrown_away = 40) (h3 : oranges_added = 24) 
  : initial_oranges - oranges_thrown_away + oranges_added = 34 := 
by
  -- Simplification and calculation here
  sorry

end oranges_in_bin_l365_365463


namespace subset_B_A_l365_365238

open Set

-- Define sets A and B based on the given conditions.
def A := { x : ℝ | -1 < x ∧ x < 2 }
def B := { x : ℝ | -1 < x ∧ x < 1 }

-- The proof problem (statement only, no proof required):
theorem subset_B_A : B ⊆ A :=
by
  sorry

end subset_B_A_l365_365238


namespace work_completed_by_a_l365_365670

theorem work_completed_by_a (a b : ℕ) (work_in_30_days : a + b = 4 * 30) (a_eq_3b : a = 3 * b) : (120 / a) = 40 :=
by
  -- Given a + b = 120 and a = 3 * b, prove that 120 / a = 40
  sorry

end work_completed_by_a_l365_365670


namespace sqrt_floor_sum_inequality_l365_365461

theorem sqrt_floor_sum_inequality
  (a : Fin 25 → ℕ) 
  (k : ℕ) 
  (h1 : ∀ i, 0 ≤ a i) 
  (h2 : ∀ i, k ≤ a i)
  (h3 : ∃ j, k = a j):
  ∑ i, Int.floor (Real.sqrt (a i : ℝ)) ≥ Int.floor (Real.sqrt (∑ i, a i + 200 * k)) :=
by
  sorry

end sqrt_floor_sum_inequality_l365_365461


namespace probability_perpendicular_probability_magnitude_l365_365215

-- Define the set of possible outcomes for a dice roll.
def outcomes := {1, 2, 3, 4, 5, 6}

-- Define vectors a and b in the plane.
def a (m n : ℕ) := (m, n)
def b := (1, -3)

-- Question 1: Prove the probability that a is perpendicular to b.
theorem probability_perpendicular (m n : ℕ) (hm : m ∈ outcomes) (hn : n ∈ outcomes) :
  real := (if m = 3 * n ∨ n = 3 * m then 1 else 0) / 36 =
  1 / 18 := sorry

-- Question 2: Prove the probability that the magnitude of a is less than or equal to the magnitude of b.
theorem probability_magnitude (m n : ℕ) (hm : m ∈ outcomes) (hn : n ∈ outcomes) :
  real := (if m^2 + n^2 ≤ 10 then 1 else 0) / 36 =
  1 / 6 := sorry

end probability_perpendicular_probability_magnitude_l365_365215


namespace find_f_log2_20_l365_365943

noncomputable def f (x : ℝ) : ℝ :=
if -1 < x ∧ x < 0 then 2^x + 1 else sorry

lemma f_periodic (x : ℝ) : f (x - 2) = f (x + 2) :=
sorry

lemma f_odd (x : ℝ) : f (-x) = -f (x) :=
sorry

theorem find_f_log2_20 : f (Real.log 20 / Real.log 2) = -1 :=
sorry

end find_f_log2_20_l365_365943


namespace union_of_A_and_B_l365_365335

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem union_of_A_and_B :
  (A ∪ B) = {1, 2, 3, 4, 5, 7} := 
by
  sorry

end union_of_A_and_B_l365_365335


namespace largest_integer_less_than_log3_sum_l365_365063

theorem largest_integer_less_than_log3_sum : 
  let log_sum := ∑ k in (range 1005).map (λ x, x + 2) \ (range 1005).map (λ x, x + 1)
  have hlog_sum : log_sum = log 1006 3,
  largest_integer_less_than_log3_sum < 7 :=
sorry

end largest_integer_less_than_log3_sum_l365_365063


namespace intersection_eq_l365_365209

def A : Set ℝ := {x : ℝ | (x - 2) / (x + 3) ≤ 0 }
def B : Set ℝ := {x : ℝ | x ≤ 1 }

theorem intersection_eq : A ∩ B = {x : ℝ | -3 < x ∧ x ≤ 1 } :=
sorry

end intersection_eq_l365_365209


namespace attendees_wearing_glasses_l365_365924

theorem attendees_wearing_glasses
  (total_attendees : ℕ)
  (women_percentage : ℚ)
  (men_percentage : ℚ)
  (women_glasses_percentage : ℚ)
  (men_glasses_percentage : ℚ)
  (H1 : total_attendees = 3000)
  (H2 : women_percentage = 0.40)
  (H3 : men_percentage = 1 - women_percentage)
  (H4 : women_glasses_percentage = 0.15)
  (H5 : men_glasses_percentage = 0.12) :
  let women_attendees := total_attendees * women_percentage
  let men_attendees := total_attendees * men_percentage
  let women_wearing_glasses := women_attendees * women_glasses_percentage
  let men_wearing_glasses := men_attendees * men_glasses_percentage
  in women_wearing_glasses + men_wearing_glasses = 396 := 
by 
  sorry

end attendees_wearing_glasses_l365_365924


namespace mary_fruits_left_l365_365347

theorem mary_fruits_left (apples_initial : ℕ) (oranges_initial : ℕ) (blueberries_initial : ℕ)
                         (ate_apples : ℕ) (ate_oranges : ℕ) (ate_blueberries : ℕ) :
  apples_initial = 14 → oranges_initial = 9 → blueberries_initial = 6 → 
  ate_apples = 1 → ate_oranges = 1 → ate_blueberries = 1 → 
  (apples_initial - ate_apples) + (oranges_initial - ate_oranges) + (blueberries_initial - ate_blueberries) = 26 :=
by
  intros
  simp [*]
  sorry

end mary_fruits_left_l365_365347


namespace flashlight_distance_difference_l365_365430

/--
Veronica's flashlight can be seen from 1000 feet. Freddie's flashlight can be seen from a distance
three times that of Veronica's flashlight. Velma's flashlight can be seen from a distance 2000 feet
less than 5 times Freddie's flashlight distance. We want to prove that Velma's flashlight can be seen 
12000 feet farther than Veronica's flashlight.
-/
theorem flashlight_distance_difference :
  let v_d := 1000
  let f_d := 3 * v_d
  let V_d := 5 * f_d - 2000
  V_d - v_d = 12000 := by
    sorry

end flashlight_distance_difference_l365_365430


namespace ratio_of_areas_l365_365723

open_locale big_operators

noncomputable def AreaRatios (A B C P : Point) : ℝ := sorry

theorem ratio_of_areas (A B C P : Point) (h : Vector PA + 3 • Vector PB + 4 • Vector PC = 0) :
  AreaRatios A B C P = 4 := sorry

end ratio_of_areas_l365_365723


namespace find_lambda_l365_365617

variables {A P B : Type} [AddCommGroup A] [AddCommGroup P] [AddCommGroup B]
           [Module ℝ A] [Module ℝ P] [Module ℝ B]

variables (AP PB AB BP : A) (λ : ℝ)

-- Condition 1: AP = 1/2 * PB
axiom h1 : AP = (1 / 2) • PB

-- Condition 2: AB = (λ + 1) • BP
axiom h2 : AB = (λ + 1) • BP

-- We need to prove λ = -5/2
theorem find_lambda : λ = - (5 / 2) :=
by
  sorry

end find_lambda_l365_365617


namespace correct_pairing_l365_365489

-- Definitions based on conditions of the problem
def num_students := 500
def blood_type_o := 200
def blood_type_a := 125
def blood_type_b := 125
def blood_type_ab := 50
def sample_size := 20
def soccer_team_size := 11
def players_sample := 2

-- Methods of sampling
inductive SamplingMethod
| Random
| Systematic
| Stratified

open SamplingMethod

-- Define the theorem to state the correct pairing
theorem correct_pairing :
  (① Stratified ② Random) :=
sorry

end correct_pairing_l365_365489


namespace intersection_of_A_and_B_l365_365989

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {2, 4, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := sorry

end intersection_of_A_and_B_l365_365989


namespace merchant_profit_percentage_l365_365113

noncomputable def cost_price : ℝ := 100
noncomputable def marked_up_price : ℝ := cost_price + (0.75 * cost_price)
noncomputable def discount : ℝ := 0.30 * marked_up_price
noncomputable def selling_price : ℝ := marked_up_price - discount
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percentage : ℝ := (profit / cost_price) * 100

theorem merchant_profit_percentage :
  profit_percentage = 22.5 :=
by
  sorry

end merchant_profit_percentage_l365_365113


namespace total_rain_expected_l365_365125

def prob_rain_first_week : ℕ → ℝ
| 0 := 0.3 -- so sun
| 2 := 0.4 -- 2 inches rain
| 8 := 0.3 -- 8 inches rain
| _ := 0 -- all other cases

def prob_rain_second_week : ℕ → ℝ
| 0 := 0.2 -- so sun
| 6 := 0.3 -- 6 inches rain
| 12 := 0.5 -- 12 inches rain
| _ := 0 -- all other cases

def expected_rain (prob : ℕ → ℝ) : ℝ :=
[0, 2, 8].sum (λ x, prob x * x)

def expected_rain_total : ℝ :=
let first_week := 5 * expected_rain prob_rain_first_week in
let second_week := 5 * expected_rain prob_rain_second_week in
first_week + second_week

theorem total_rain_expected : expected_rain_total = 55.0 := by
  sorry

end total_rain_expected_l365_365125


namespace Razorback_shop_total_revenue_l365_365008

theorem Razorback_shop_total_revenue :
  let Tshirt_price := 62
  let Jersey_price := 99
  let Hat_price := 45
  let Keychain_price := 25
  let Tshirt_sold := 183
  let Jersey_sold := 31
  let Hat_sold := 142
  let Keychain_sold := 215
  let revenue := (Tshirt_price * Tshirt_sold) + (Jersey_price * Jersey_sold) + (Hat_price * Hat_sold) + (Keychain_price * Keychain_sold)
  revenue = 26180 :=
by
  sorry

end Razorback_shop_total_revenue_l365_365008


namespace num_black_balls_l365_365280

-- Definitions based on problem conditions
def num_red_balls : ℕ := 3
def num_white_balls : ℕ := 5
def probability_white_ball := 0.25

-- Theorem statement
theorem num_black_balls (x : ℕ) : (5 / (num_red_balls + num_white_balls + x) = probability_white_ball) → x = 12 :=
sorry

end num_black_balls_l365_365280


namespace suitable_high_jump_athlete_l365_365883

structure Athlete where
  average : ℕ
  variance : ℝ

def A : Athlete := ⟨169, 6.0⟩
def B : Athlete := ⟨168, 17.3⟩
def C : Athlete := ⟨169, 5.0⟩
def D : Athlete := ⟨168, 19.5⟩

def isSuitableCandidate (athlete: Athlete) (average_threshold: ℕ) : Prop :=
  athlete.average = average_threshold

theorem suitable_high_jump_athlete : isSuitableCandidate C 169 ∧
  (∀ a, isSuitableCandidate a 169 → a.variance ≥ C.variance) := by
  sorry

end suitable_high_jump_athlete_l365_365883


namespace range_of_b_minus_2_over_a_minus_1_l365_365587

theorem range_of_b_minus_2_over_a_minus_1
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = (1/3)*x^3 + (1/2)*a*x^2 + 2*b*x + c)
  (h2 : ∃ (x ∈ Ioo 0 1), f'' x < 0)
  (h3 : ∃ (x ∈ Ioo 1 2), f'' x > 0) :
  (1/4 : ℝ) < (b - 2) / (a - 1) ∧ (b - 2) / (a - 1) < 1 := 
sorry

end range_of_b_minus_2_over_a_minus_1_l365_365587


namespace find_parameters_l365_365810

theorem find_parameters (s h : ℝ) :
  (∀ (x y t : ℝ), (x = s + 3 * t) ∧ (y = 2 + h * t) ∧ (y = 5 * x - 7)) → (s = 9 / 5 ∧ h = 15) :=
by
  sorry

end find_parameters_l365_365810


namespace eight_odot_six_eq_ten_l365_365399

-- Define the operation ⊙ as given in the problem statement
def operation (a b : ℕ) : ℕ := a + (3 * a) / (2 * b)

-- State the theorem to prove
theorem eight_odot_six_eq_ten : operation 8 6 = 10 :=
by
  -- Here you will provide the proof, but we skip it with sorry
  sorry

end eight_odot_six_eq_ten_l365_365399


namespace double_sum_evaluation_l365_365540

theorem double_sum_evaluation : 
  (∑ n from 2 to ∞, ∑ k from 1 to (n - 1), k / (3 ^ (n + k))) = (6 / 25) :=
by sorry

end double_sum_evaluation_l365_365540


namespace triangle_angle_a_value_triangle_side_a_value_l365_365704

open Real

theorem triangle_angle_a_value (a b c A B C : ℝ) 
  (h1 : (a - c) * (a + c) * sin C = c * (b - c) * sin B)
  (h2 : (1/2) * b * c * sin A = sqrt 3)
  (h3 : sin B * sin C = 1/4) :
  A = π / 3 :=
sorry

theorem triangle_side_a_value (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : (a - c) * (a + c) * sin C = c * (b - c) * sin B)
  (h2 : (1/2) * b * c * sin(A) = sqrt 3)
  (h3 : sin B * sin C = 1/4)
  (h4 : A = π / 3) :
  a = 2 * sqrt 3 :=
sorry

end triangle_angle_a_value_triangle_side_a_value_l365_365704


namespace blue_bird_chess_team_arrangement_l365_365006

theorem blue_bird_chess_team_arrangement : 
  let boys := 3
  let girls := 3
  let end_arrangements := boys * (boys - 1) -- choosing 2 out of 3 and arranging
  let mid_arrangements := 1               -- only 1 way to place in middle
  let girl_arrangements := Mathlib.factorial girls -- arranging girls
  end_arrangements * mid_arrangements * girl_arrangements = 36 := 
by {
  sorry
}

end blue_bird_chess_team_arrangement_l365_365006


namespace run_program_l365_365862

theorem run_program : ∀ (a b : ℕ), a = 2 → b = 3 → (if a > b then a else b) = 3 :=
begin
  intros a b ha hb,
  rw [ha, hb],
  simp [if_neg, lt_irrefl]
end

end run_program_l365_365862


namespace simplify_and_evaluate_expression_l365_365772

theorem simplify_and_evaluate_expression (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  ( (2 * x - 3) / (x - 2) - 1 ) / ( (x^2 - 2 * x + 1) / (x - 2) ) = 1 / 2 :=
by {
  sorry
}

end simplify_and_evaluate_expression_l365_365772


namespace sum_non_empty_proper_subset_not_zero_l365_365331

open Complex

noncomputable def is_primitive_root (n : ℕ) (z : ℂ) : Prop :=
z ^ n = 1 ∧ ∀ m : ℕ, 0 < m ∧ m < n → z ^ m ≠ 1

theorem sum_non_empty_proper_subset_not_zero (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  let ε := exp (2 * π * I / p)
  let M := {x | ∃ (i : ℕ), 1 ≤ i ∧ i ≤ p ∧ x = ε ^ i}
  is_primitive_root p ε →
  ∀ S : Set ℂ, S ⊆ M ∧ S ≠ ∅ ∧ S ≠ M → (∑ x in S, x) ≠ 0 := by
  sorry

end sum_non_empty_proper_subset_not_zero_l365_365331


namespace symmetry_implies_p_eq_s_l365_365811

variables (p q r s : ℝ)
variables (p_nonzero : p ≠ 0) (q_nonzero : q ≠ 0) (r_nonzero : r ≠ 0) (s_nonzero : s ≠ 0)
variables (h_symmetry : ∀ a b : ℝ, (b ≠ 0 → (a, b)) ∈ set_of (λ (x, y), y = (px + q) / (rx + s)) →
                                  (a ≠ 0 → (-b, -a)) ∈ set_of (λ (x, y), y = (px + q) / (rx + s)))

-- We need to prove that p = s
theorem symmetry_implies_p_eq_s : p = s :=
sorry

end symmetry_implies_p_eq_s_l365_365811


namespace duty_schedules_count_l365_365832

theorem duty_schedules_count :
  let A, B, C := ("A", "B", "C") in
  let days := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"] in
  (∀ day, day ∈ days → (A ≠ day ∨ B ≠ day)) → 
  (∀ day, day ∈ days → (day ≠ "Monday" ∨ A ≠ "Monday")) → 
  (∀ day, day ∈ days → (day ≠ "Saturday" ∨ B ≠ "Saturday")) → 
  (∑ p in days.powerset.filter (λ s, s.card = 2), 
    ∑ q in (days \ p).powerset.filter (λ t, t.card = 2), 
      if (¬ ("Monday" ∈ p ∧ "Saturday" ∈ q)) then 1 else 0) = 42 := 
by sorry

end duty_schedules_count_l365_365832


namespace BC_total_750_l365_365505

theorem BC_total_750 (A B C : ℤ) 
  (h1 : A + B + C = 900) 
  (h2 : A + C = 400) 
  (h3 : C = 250) : 
  B + C = 750 := 
by 
  sorry

end BC_total_750_l365_365505


namespace proof_problem_l365_365069

variables (a b : ℝ)

noncomputable def expr := (2 * a⁻¹ + (a⁻¹ / b)) / a

theorem proof_problem (h1 : a = 1/3) (h2 : b = 3) : expr a b = 21 :=
by
  sorry

end proof_problem_l365_365069


namespace find_x_l365_365256

theorem find_x 
  (x : ℕ)
  (h : 3^x = 3^(20) * 3^(20) * 3^(18) + 3^(19) * 3^(20) * 3^(19) + 3^(18) * 3^(21) * 3^(19)) :
  x = 59 :=
sorry

end find_x_l365_365256


namespace find_a_squared_l365_365488

-- Defining the conditions for the problem
structure RectangleConditions :=
  (a : ℝ) 
  (side_length : ℝ := 36)
  (hinges_vertex : Bool := true)
  (hinges_midpoint : Bool := true)
  (pressed_distance : ℝ := 24)
  (hexagon_area_equiv : Bool := true)

-- Stating the theorem
theorem find_a_squared (cond : RectangleConditions) (ha : 36 * cond.a = 
  (24 * cond.a) + 2 * 15 * Real.sqrt (cond.a^2 - 36)) : 
  cond.a^2 = 720 :=
sorry

end find_a_squared_l365_365488


namespace evaluate_expression_l365_365164

def ceil (x : ℚ) : ℤ := sorry -- Implement the ceiling function for rational numbers as needed

theorem evaluate_expression :
  (ceil ((23 : ℚ) / 9 - ceil ((35 : ℚ) / 23))) 
  / (ceil ((35 : ℚ) / 9 + ceil ((9 * 23 : ℚ) / 35))) = (1 / 10 : ℚ) :=
by
  intros
  -- Proof goes here
  sorry

end evaluate_expression_l365_365164


namespace add_in_base6_l365_365913

def add_base6 (a b : ℕ) : ℕ := (a + b) % 6 + (((a + b) / 6) * 10)

theorem add_in_base6 (x y : ℕ) (h1 : x = 5) (h2 : y = 23) : add_base6 x y = 32 :=
by
  rw [h1, h2]
  -- Explanation: here add_base6 interprets numbers as base 6 and then performs addition,
  -- taking care of the base conversion automatically. This avoids directly involving steps of the given solution.
  sorry

end add_in_base6_l365_365913


namespace solution_set_l365_365746

def f (x : ℝ) : ℝ :=
  if x >= 0 then x^2 - 4*x + 6 else x + 6

theorem solution_set (x : ℝ) : f(x) > f(1) ↔ ((-3 < x ∧ x < 1) ∨ (3 < x)) :=
by
  sorry

end solution_set_l365_365746


namespace locus_C2_max_area_OAB_l365_365286

-- Define the conditions for part 1
def curve_C1 (ρ θ : ℝ) : Prop := ρ * (Real.cos θ) = 4
def OP (ρ ρ1 θ: ℝ) := ρ 
def OM (ρ1 θ: ℝ) := ρ1 * (Real.cos θ)
def condition_OM_OP (ρ ρ1 θ: ℝ) := (ρ * (4 / (Real.cos θ))) = 16
noncomputable def locus_P (ρ θ: ℝ) := 4 * (Real.cos θ)

-- Prove that the rectangular coordinate equation for C2
theorem locus_C2 (x y: ℝ) : (∃ ρ θ, curve_C1 ρ θ ∧ locus_P ρ θ ∧ ρ^2 = x^2 + y^2 ∧ ρ * (Real.cos θ) = x ∧ x ≠ 0 → (x - 2)^2 + y^2 = 4):=
  sorry

-- Define the conditions for part 2
def polar_point (ρ θ: ℝ) : ℝ×ℝ := (ρ, θ)
def point_B (ρ α: ℝ) (C2_on_B : locus_P ρ α) := (ρ, α)

-- Prove the maximum area of triangle OAB
theorem max_area_OAB : (∃ α, α = -π/12 ∧ ∀ A B: ℝ×ℝ, polar_point 2 (π/3) = A ∧ B = point_B (4 * (Real.cos α)) α C2_on_B → 2 + ℓt3/2) :=
  sorry

end locus_C2_max_area_OAB_l365_365286


namespace part1_part2_l365_365657

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

def two_a : ℝ × ℝ := (2 * a.1, 2 * a.2)
def four_b : ℝ × ℝ := (4 * b.1, 4 * b.2)

def two_a_minus_four_b : ℝ × ℝ := (two_a.1 - four_b.1, two_a.2 - four_b.2)
def magnitude (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

noncomputable def magnitude_two_a_minus_four_b : ℝ := magnitude two_a_minus_four_b.1 two_a_minus_four_b.2

def ka (k : ℝ) : ℝ × ℝ := (k * a.1, k * a.2)
def ka_plus_two_b (k : ℝ) : ℝ × ℝ := (ka k.1 + 2 * b.1, ka k.2 + 2 * b.2)
def is_parallel (v w : ℝ × ℝ) : Prop := ∃ m : ℝ, (v.1, v.2) = (m * w.1, m * w.2)

theorem part1 : magnitude_two_a_minus_four_b = 2 * Real.sqrt 53 := sorry

theorem part2 : is_parallel (ka_plus_two_b (-1)) two_a_minus_four_b := sorry

end part1_part2_l365_365657


namespace unit_cubes_with_no_more_than_two_shared_vertices_l365_365106

theorem unit_cubes_with_no_more_than_two_shared_vertices (n : ℕ) : 
  ∑ 1 ≤ k ≤ n, 2 * (k * ((n ^ 4) - (7 * k) + (6 * (n ^ 2)))) / 2 = n^6 :=
by
  sorry

end unit_cubes_with_no_more_than_two_shared_vertices_l365_365106


namespace fourth_term_expansion_l365_365559

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem fourth_term_expansion (x : ℝ) (n : ℕ) (h : binomial_coefficient n 2 = binomial_coefficient n 6) :
  n = 8 → (binomial_coefficient 8 3 * (1 / x)^3 * x^5 = 56 * x^2) :=
by
  intro h_n
  rw h_n
  sorry

end fourth_term_expansion_l365_365559


namespace find_a2016_l365_365338

-- Given conditions
axiom pos_seq {α : Type*} (a : ℕ → ℕ) : ∀ n : ℕ, a n > 0
axiom sum_first_n {α : Type*} (a : ℕ → ℕ) (S : ℕ → ℕ) : ∀ n : ℕ, S n = ∑ i in finset.range (n + 1), a i
axiom seq_relation {α : Type*} (a : ℕ → ℕ) (S : ℕ → ℕ) : ∀ n : ℕ, 4 * S n = a n ^ 2 + 2 * a n - 3

-- Statement to prove
theorem find_a2016 {α : Type*} (a : ℕ → ℕ) (S : ℕ → ℕ) [pos_seq a] [sum_first_n a S] [seq_relation a S] :
  a 2016 = 4033 :=
sorry

end find_a2016_l365_365338


namespace impossible_to_have_all_piles_with_2023_tokens_l365_365408

def is_prime (n : ℕ) : Prop := nat.prime n

def initial_piles : list ℕ := finite (2023.list.map nat.prime)

def split_pile (pile : ℕ) : (ℕ × ℕ) :=
(pile / 2, pile / 2 + 1)

def merge_piles (pile1 pile2 : ℕ) : ℕ :=
pile1 + pile2 + 1

theorem impossible_to_have_all_piles_with_2023_tokens (piles : list ℕ)
  (h_initial : piles = initial_piles)
  (h_final : ∀ p ∈ piles, p = 2023)
  (h_operations : ∀ piles : list ℕ, (∃ p ∈ piles, p = split_pile p₁) ∨ (∃ p₁ p₂ ∈ piles, merge_piles p₁ p₂ = p)) :
  false :=
begin
  sorry
end

end impossible_to_have_all_piles_with_2023_tokens_l365_365408


namespace camp_children_current_count_l365_365291

theorem camp_children_current_count :
  ∀ (C : ℕ) (B : ℕ) (G : ℕ),
    (0.85 * C).nat_abs = B →
    (0.15 * C).nat_abs = G →
    ∃ (C' : ℕ), C' = C + 50 ∧ 
    ∃ (B' : ℕ), B' = B + 50 ∧ 
    ((0.05 * C').nat_abs = G) → 
    C = 25 :=
by
  intros C B G hB hG hC' hB' H
  sorry

end camp_children_current_count_l365_365291


namespace mary_fruits_left_l365_365341

-- Conditions as definitions:
def mary_bought_apples : ℕ := 14
def mary_bought_oranges : ℕ := 9
def mary_bought_blueberries : ℕ := 6

def mary_ate_apples : ℕ := 1
def mary_ate_oranges : ℕ := 1
def mary_ate_blueberries : ℕ := 1

-- The problem statement:
theorem mary_fruits_left : 
  (mary_bought_apples - mary_ate_apples) + 
  (mary_bought_oranges - mary_ate_oranges) + 
  (mary_bought_blueberries - mary_ate_blueberries) = 26 := by
  sorry

end mary_fruits_left_l365_365341


namespace distance_between_parallel_lines_l365_365797

theorem distance_between_parallel_lines :
  ∀ {x y : ℝ}, 
  (3 * x - 4 * y + 1 = 0) → (3 * x - 4 * y + 7 = 0) → 
  ∃ d, d = (6 : ℝ) / 5 :=
by 
  sorry

end distance_between_parallel_lines_l365_365797


namespace sum_eq_9_div_64_l365_365547

noncomputable def double_sum : ℝ := ∑' (n : ℕ) in (set.Ici 2 : set ℕ), ∑' (k : ℕ) in set.Ico 1 n, (k : ℝ) / 3^(n + k)

theorem sum_eq_9_div_64 : double_sum = 9 / 64 := 
by 
sorry

end sum_eq_9_div_64_l365_365547


namespace sin_eq_cos_example_l365_365578

theorem sin_eq_cos_example 
  (n : ℤ) (h_range : -180 ≤ n ∧ n ≤ 180)
  (h_eq : Real.sin (n * Real.pi / 180) = Real.cos (682 * Real.pi / 180)) :
  n = 128 :=
sorry

end sin_eq_cos_example_l365_365578


namespace difference_of_roots_l365_365732

noncomputable def r_and_s (r s : ℝ) : Prop :=
(∃ (r s : ℝ), (r, s) ≠ (s, r) ∧ r > s ∧ (5 * r - 15) / (r ^ 2 + 3 * r - 18) = r + 3
  ∧ (5 * s - 15) / (s ^ 2 + 3 * s - 18) = s + 3)

theorem difference_of_roots (r s : ℝ) (h : r_and_s r s) : r - s = Real.sqrt 29 := by
  sorry

end difference_of_roots_l365_365732


namespace perimeter_of_rhombus_l365_365790

def rhombus_perimeter (d1 d2 : ℝ) : ℝ :=
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2))

theorem perimeter_of_rhombus (h1 : ∀ (d1 d2 : ℝ), d1 = 24 ∧ d2 = 10) : rhombus_perimeter 24 10 = 52 := 
by
  sorry

end perimeter_of_rhombus_l365_365790


namespace max_value_of_b_over_a_squared_l365_365727

variables {a b x y : ℝ}

def triangle_is_right (a b x y : ℝ) : Prop :=
  (a - x)^2 + (b - y)^2 = a^2 + b^2

theorem max_value_of_b_over_a_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b)
    (h4 : ∃ x y, a^2 + y^2 = b^2 + x^2 
                 ∧ b^2 + x^2 = (a - x)^2 + (b - y)^2
                 ∧ 0 ≤ x ∧ x < a 
                 ∧ 0 ≤ y ∧ y < b 
                 ∧ triangle_is_right a b x y) 
    : (b / a)^2 = 4 / 3 :=
sorry

end max_value_of_b_over_a_squared_l365_365727


namespace apple_problem_l365_365820

def initial_apples : ℕ := 23
def apples_used_for_slices : ℕ := 15
def apples_bought : ℕ := 6
def donation_percentage : ℝ := 35 / 100
def total_fruits_donated : ℕ := 120
def percentage_for_pies : ℝ := 12.5 / 100

def apples_remaining_after_all_operations : ℕ :=
  let apples_after_slices := initial_apples - apples_used_for_slices in
  let apples_after_buying := apples_after_slices + apples_bought in
  let apples_donated := (donation_percentage * total_fruits_donated).to_nat in
  let apples_total := apples_after_buying + apples_donated in
  let apples_for_pies := (percentage_for_pies * apples_total).to_nat in
  apples_total - apples_for_pies

theorem apple_problem : apples_remaining_after_all_operations = 49 := 
by
  sorry

end apple_problem_l365_365820


namespace find_f_2_l365_365645

noncomputable def f (a b x : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 :=
by
  sorry

end find_f_2_l365_365645


namespace solve_equation_l365_365776

theorem solve_equation (x y : ℝ) : 
    ((16 * x^2 + 1) * (y^2 + 1) = 16 * x * y) ↔ 
        ((x = 1/4 ∧ y = 1) ∨ (x = -1/4 ∧ y = -1)) := 
by
  sorry

end solve_equation_l365_365776


namespace select_representatives_l365_365409

theorem select_representatives
  (female_count : ℕ) (male_count : ℕ)
  (female_count_eq : female_count = 4)
  (male_count_eq : male_count = 6) :
  female_count * male_count = 24 := by
  sorry

end select_representatives_l365_365409


namespace sum_palindromic_primes_upto_50_eq_109_l365_365359

def is_palindromic_prime (n : ℕ) : Prop :=
  let reversed := (n % 10) * 10 + (n / 10)
  prime n ∧ prime reversed

def sum_palindromic_primes_upto_50 : ℕ :=
  (11 :: 13 :: 17 :: 19 :: 23 :: 29 :: 31 :: 37 :: 41 :: 43 :: 47 :: []).filter is_palindromic_prime |>.sum

theorem sum_palindromic_primes_upto_50_eq_109 :
  sum_palindromic_primes_upto_50 = 109 := by
  sorry

end sum_palindromic_primes_upto_50_eq_109_l365_365359


namespace dance_problem_l365_365402

theorem dance_problem :
  ∃ (G : ℝ) (B T : ℝ),
    B / G = 3 / 4 ∧
    T = 0.20 * B ∧
    B + G + T = 114 ∧
    G = 60 :=
by
  sorry

end dance_problem_l365_365402


namespace min_distance_from_C_to_circle_l365_365759

theorem min_distance_from_C_to_circle
  (R : ℝ) (AC : ℝ) (CB : ℝ) (C M : ℝ)
  (hR : R = 6) (hAC : AC = 4) (hCB : CB = 5)
  (hCM_eq : C = 12 - M) :
  C * M = 20 → (M < 6) → M = 2 := 
sorry

end min_distance_from_C_to_circle_l365_365759


namespace sum_of_series_l365_365165

theorem sum_of_series : (∑ n in Finset.range 10, (1 : ℚ) / ((n + 1) * (n + 2) * (n + 3))) = 1 / 24 :=
by
  sorry

end sum_of_series_l365_365165


namespace min_width_for_fence_area_least_200_l365_365765

theorem min_width_for_fence_area_least_200 (w : ℝ) (h : w * (w + 20) ≥ 200) : w ≥ 10 :=
sorry

end min_width_for_fence_area_least_200_l365_365765


namespace price_per_pound_salt_is_50_l365_365114

-- Given conditions
def totalWeight : ℕ := 60
def weightSalt1 : ℕ := 20
def priceSalt2 : ℕ := 35
def weightSalt2 : ℕ := 40
def sellingPricePerPound : ℕ := 48
def desiredProfitRate : ℚ := 0.20

-- Mathematical definitions derived from conditions
def costSalt1 (priceSalt1 : ℕ) : ℕ := weightSalt1 * priceSalt1
def costSalt2 : ℕ := weightSalt2 * priceSalt2
def totalCost (priceSalt1 : ℕ) : ℕ := costSalt1 priceSalt1 + costSalt2
def totalRevenue : ℕ := totalWeight * sellingPricePerPound
def profit (priceSalt1 : ℕ) : ℚ := desiredProfitRate * totalCost priceSalt1
def totalProfit (priceSalt1 : ℕ) : ℚ := totalCost priceSalt1 + profit priceSalt1

-- Proof statement
theorem price_per_pound_salt_is_50 : ∃ (priceSalt1 : ℕ), totalRevenue = totalProfit priceSalt1 ∧ priceSalt1 = 50 := by
  -- We provide the prove structure, exact proof steps are skipped with sorry
  sorry

end price_per_pound_salt_is_50_l365_365114


namespace members_count_l365_365454

theorem members_count (n : ℕ) (h : n * n = 2025) : n = 45 :=
sorry

end members_count_l365_365454


namespace pencils_per_row_l365_365168

-- Problem: Faye was placing 12 pencils equally into 3 rows.
-- Prove that she placed 4 pencils in each row.

theorem pencils_per_row {total_pencils rows pencils_per_row : ℕ} 
  (h1 : total_pencils = 12)
  (h2 : rows = 3)
  (h3 : total_pencils % rows = 0): 
  pencils_per_row = total_pencils / rows :=
by
  -- The correct answer should be 4 pencils per row
  have : pencils_per_row = 4, from sorry
  exact this

end pencils_per_row_l365_365168


namespace least_faces_of_two_dice_l365_365427

theorem least_faces_of_two_dice (a b : ℕ) (h : 6 ≤ a ∧ 6 ≤ b) 
  (prob_sum8 : 6 * (1 / (a * b)) = (2 / 3) * (sum_ways_of 11 a b * (1 / (a * b))))
  (prob_sum13 : ∃ n, n = 1 / 13 ∧ sum_ways_of 13 a b = n * (a * b))
  (distinct : ∀ m n, faces m a → faces n b → m ≠ n) :
  a + b = 22 :=
by
  sorry

def sum_ways_of (n a b : ℕ) : ℕ :=
  sorry

def faces (n max : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ max

#check least_faces_of_two_dice

end least_faces_of_two_dice_l365_365427


namespace area_of_union_of_square_and_circle_l365_365500

-- Definitions based on problem conditions
def side_length : ℕ := 8
def radius : ℝ := 12
def square_area := side_length ^ 2
def circle_area := Real.pi * radius ^ 2
def overlapping_area := 1 / 4 * circle_area
def union_area := square_area + circle_area - overlapping_area

-- The problem statement in Lean format
theorem area_of_union_of_square_and_circle :
  union_area = 64 + 108 * Real.pi :=
by
  sorry

end area_of_union_of_square_and_circle_l365_365500


namespace sqrt_eq_condition_l365_365253

theorem sqrt_eq_condition (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (√ (x - y / z.to_rat) = x * √ (y / z.to_rat)) ↔ (z = (y * (x^2 + 1)) / x) := sorry

end sqrt_eq_condition_l365_365253


namespace probability_cheryl_same_color_l365_365102

-- Define the marbles and the conditions given in the problem
def total_marbles : ℕ := 3  -- Each of red, green, yellow, and blue
def total_colors : ℕ := 4  -- Number of colors
def total_draws : ℕ := 12 -- Total marbles in the box

-- Define the draws by Carol, Claudia, and Cheryl
def carol_draws : ℕ := 3
def claudia_draws : ℕ := 3
def cheryl_draws : ℕ := 3

-- Defining the conditions in terms of binomial coefficients
def total_ways : ℕ := Nat.choose total_draws carol_draws * Nat.choose (total_draws - carol_draws) claudia_draws * Nat.choose (total_draws - carol_draws - claudia_draws) cheryl_draws

def favorable_claudia_outcomes : ℕ := Nat.choose (total_draws - carol_draws) claudia_draws - total_colors
def favorable_cheryl_outcomes : ℕ := total_colors - 1  -- Claudia did not draw any complete color, so Cheryl has 3 left to form a set of 3

def favorable_ways : ℕ := favorable_claudia_outcomes * Nat.choose carol_draws (carol_draws/total_colors)

theorem probability_cheryl_same_color (condition_claudia : favorable_claudia_outcomes) :
  (favorable_ways : ℚ) / (total_ways : ℚ) =  Rational.mk(55, 1540) :=
by 
-- Proof not required as per the problem statement
sorry

end probability_cheryl_same_color_l365_365102


namespace sqrt_product_simplified_l365_365930

theorem sqrt_product_simplified (q : ℝ) : 
  (√(15 * q) * √(7 * q^3) * √(8 * q^5)) = 210 * q^4 * √q :=
by sorry

end sqrt_product_simplified_l365_365930


namespace superprimes_less_than_15_l365_365897

def is_superprime (p : ℕ) : Prop :=
  is_prime p ∧ is_prime (2 * p - 1)

def number_of_superprimes_less_than_15 : ℕ :=
  (List.range 15).filter is_superprime).length

theorem superprimes_less_than_15 :
  number_of_superprimes_less_than_15 = 3 :=
by sorry

end superprimes_less_than_15_l365_365897


namespace dice_sum_25_probability_l365_365481

/--
A fair twenty-faced die has 19 of its faces numbered from 1 through 18 and one blank face.
Another fair twenty-faced die has 19 of its faces numbered from 2 through 9 and 11 through 21 and one blank face.
When the two dice are rolled, the probability that the sum of the two numbers facing up will be 25 is 3/80.
-/
theorem dice_sum_25_probability :
  (∃ die1 die2 : ℕ, (die1 ∈ Finset.range(19) ∧ die2 ∈ Finset.range(20) \ {10} ∧ die1 + die2 = 25) → 3 / 80) := sorry

end dice_sum_25_probability_l365_365481


namespace sum_slope_yintercept_half_area_l365_365424

def Point := (ℝ × ℝ)

def P : Point := (0, 10)
def Q : Point := (3, 0)
def R : Point := (9, 0)

noncomputable def midpoint (A B : Point) : Point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def slope (A B : Point) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

noncomputable def line_eq (A : Point) (m : ℝ) : (ℝ × ℝ) :=
  (m, A.2 - m * A.1)  -- y = mx + b => b = A.2 - m * A.1

theorem sum_slope_yintercept_half_area :
  let M := midpoint P R
  let m_QM := slope Q M
  let line := line_eq Q m_QM
  line.1 + line.2 = -20 / 3 :=
by
  sorry

end sum_slope_yintercept_half_area_l365_365424


namespace range_of_a_l365_365217

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, sin x * sin x - 4 * sin x + 1 - a = 0) ↔ -2 ≤ a ∧ a ≤ 6 :=
by sorry

end range_of_a_l365_365217


namespace speed_of_man_in_still_water_l365_365870

variable (v_m v_s : ℝ)

-- Conditions as definitions 
def downstream_distance_eq : Prop :=
  36 = (v_m + v_s) * 3

def upstream_distance_eq : Prop :=
  18 = (v_m - v_s) * 3

theorem speed_of_man_in_still_water (h1 : downstream_distance_eq v_m v_s) (h2 : upstream_distance_eq v_m v_s) : v_m = 9 := 
  by
  sorry

end speed_of_man_in_still_water_l365_365870


namespace range_of_values_for_a_l365_365637

theorem range_of_values_for_a (f : ℝ → ℝ) (a : ℝ) (h_monotone : ∀ x y, 0 < x ∧ x < 1 → 0 < y ∧ y < 1 → x < y → f x ≥ f y) :
  a ∈ set.Ici 2 :=
by
  sorry

end range_of_values_for_a_l365_365637


namespace number_of_good_permutations_correct_l365_365205

noncomputable def number_of_good_permutations (m n : ℕ) (h1 : 2 ≤ m) (h2 : m ≤ n) : ℕ :=
  (Nat.factorial n) / (Nat.factorial (n - m)) - Nat.choose (Nat.floor ((m + n) / 2)) m

theorem number_of_good_permutations_correct (m n : ℕ) (h1 : 2 ≤ m) (h2 : m ≤ n) :
  let good_permutations := (Nat.factorial n) / (Nat.factorial (n - m)) - Nat.choose (Nat.floor ((m + n) / 2)) m
  in number_of_good_permutations m n h1 h2 = good_permutations :=
by sorry

end number_of_good_permutations_correct_l365_365205


namespace max_length_AB_l365_365413

theorem max_length_AB : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 3 → ∃ M, M = 81 / 8 ∧ ∀ t, -2 * (t - 3/4)^2 + 81 / 8 = M :=
by sorry

end max_length_AB_l365_365413


namespace circle_equation_l365_365477

theorem circle_equation (a : ℝ) (x y : ℝ) (h1 : a < 0) (h2 : (x - a)^2 + y^2 = 4) (h3 : (0 - a)^2 + 0^2 = 4) :
  ∃ a, (x + 2)^2 + y^2 = 4 := 
by
  sorry

end circle_equation_l365_365477


namespace ratio_laptops_to_total_l365_365755

theorem ratio_laptops_to_total (total_computers : ℕ)
  (one_third_netbooks : ℕ)
  (desktop_computers : ℕ)
  (h_total : total_computers = 72)
  (h_one_third_netbooks : one_third_netbooks = total_computers / 3)
  (h_desktops : desktop_computers = 12) :
  let laptops := total_computers - (one_third_netbooks + desktop_computers) in
  ∃ (r : ℕ × ℕ), r = (laptops / (Nat.gcd laptops total_computers), total_computers / (Nat.gcd laptops total_computers)) ∧ r = (1, 2) :=
by
  sorry

end ratio_laptops_to_total_l365_365755


namespace profit_from_ad_l365_365301

def advertising_cost : ℝ := 1000
def customers : ℕ := 100
def purchase_rate : ℝ := 0.8
def purchase_price : ℝ := 25

theorem profit_from_ad (advertising_cost customers purchase_rate purchase_price : ℝ) : 
  (customers * purchase_rate * purchase_price - advertising_cost) = 1000 :=
by
  -- assumptions as conditions
  let bought_customers := (customers : ℝ) * purchase_rate
  let revenue := bought_customers * purchase_price
  let profit := revenue - advertising_cost
  -- state the proof goal
  have goal : profit = 1000 :=
    sorry
  exact goal

end profit_from_ad_l365_365301


namespace expected_sides_quadrilateral_expected_sides_n_sided_l365_365841

-- Part (a)
theorem expected_sides_quadrilateral :
  let k := 3600
  let initial_sides := 4
  let total_sides := initial_sides + 4 * k
  let num_polygons := k + 1
  let expected_sides := (initial_sides + 4 * k) / (k + 1)
  in expected_sides = 4 :=
by
  let k := 3600
  let initial_sides := 4
  let total_sides := initial_sides + 4 * k
  let num_polygons := k + 1
  let expected_sides := (initial_sides + 4 * k) / (k + 1)
  sorry

-- Part (b)
theorem expected_sides_n_sided (n : ℕ) :
  let k := 3600
  let total_sides := n + 4 * k
  let num_polygons := k + 1
  let expected_sides := (n + 14400) / (3601)
  in expected_sides = (n + 14400) / 3601 :=
by
  let k := 3600
  let total_sides := n + 4 * k
  let num_polygons := k + 1
  let expected_sides := (n + 14400) / 3601
  sorry

end expected_sides_quadrilateral_expected_sides_n_sided_l365_365841


namespace johns_investments_l365_365718

theorem johns_investments 
  (total_interest : ℝ := 1282)
  (A1 A2 A3 A4 : ℝ := 4000, 8200, 5000, 6000)
  (r1 r2 r3 r4 : ℝ)
  (H1 : r2 = r1 + 1.5)
  (H2 : r3 = 2 * r1)
  (H3 : r4 = r3 - 0.5) 
  (H_sum : A1 * (r1/100) + A2 * (r2/100) + A3 * (r3/100) + A4 * (r4/100) = total_interest) :
  r1 = 2.64 ∧ r2 = 4.14 ∧ r3 = 5.28 ∧ r4 = 4.78 :=
by sorry

end johns_investments_l365_365718


namespace mutually_exclusive_event_C_l365_365863

def event_A := {ω | (∃ k : ℕ, ω.count_tt = k ∧ 1 ≤ k ≤ 1) }
def event_B := {ω | ω.count_tt ≤ 1 ∧ ω.count_tt = 2 }
def event_C := {ω | ω.count_tt ≤ 1 ∧ ω.count_tt ≥ 2 }
def event_D := {ω | (∃ k : ℕ, ω.count_tt = k ∧ 2 ≤ k ≤ 2) ∧ (∃ k : ℕ, ω.count_tt = k ∧ k = 1) }

theorem mutually_exclusive_event_C :
  ∀ ω : finset (bool × bool × bool), ω ∈ event_C → ω ∉ event_D := by
  sorry

end mutually_exclusive_event_C_l365_365863


namespace evaluate_ceiling_expression_l365_365161

theorem evaluate_ceiling_expression:
  (Int.ceil ((23 : ℚ) / 9 - Int.ceil ((35 : ℚ) / 23)))
  / (Int.ceil ((35 : ℚ) / 9 + Int.ceil ((9 * 23 : ℚ) / 35))) = 1 / 12 := by
  sorry

end evaluate_ceiling_expression_l365_365161


namespace arithmetic_mean_of_sixty_integers_starting_from_3_l365_365139

def arithmetic_mean_of_sequence (a d n : ℕ) : ℚ :=
  let a_n := a + (n - 1) * d
  let S_n := n * (a + a_n) / 2
  S_n / n

theorem arithmetic_mean_of_sixty_integers_starting_from_3 : arithmetic_mean_of_sequence 3 1 60 = 32.5 :=
by 
  sorry

end arithmetic_mean_of_sixty_integers_starting_from_3_l365_365139


namespace arithmetic_sequence_sum_first_nine_terms_l365_365264

variable (a_n : ℕ → ℤ)
variable (S_n : ℕ → ℤ)
variable (d : ℤ)

-- The sequence {a_n} is an arithmetic sequence.
def arithmetic_sequence := ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- The sum of the first n terms of the sequence.
def sum_first_n_terms := ∀ n : ℕ, S_n n = (n * (a_n 1 + a_n n)) / 2

-- Given condition: a_2 = 3 * a_4 - 6
def given_condition := a_n 2 = 3 * a_n 4 - 6

-- The main theorem to prove S_9 = 27
theorem arithmetic_sequence_sum_first_nine_terms (h_arith : arithmetic_sequence a_n d) (h_sum : sum_first_n_terms a_n S_n) (h_condition : given_condition a_n) : 
  S_n 9 = 27 := 
by
  sorry

end arithmetic_sequence_sum_first_nine_terms_l365_365264


namespace area_parallelogram_l365_365604

-- Define the given conditions
variables (ABCD : Type) [parallelogram ABCD] 
variables (C K L : ABCD → Point)
variables (p q : ℝ)

-- Define the conditions
variables (area_KBC : ℝ) (area_CDL : ℝ)
hypothesis h1 : area_KBC = p
hypothesis h2 : area_CDL = q

-- State the theorem
theorem area_parallelogram (hC_K : line_through C K) (hC_L : line_through C L) (hK_AB : ∃ A B, line_through A B ∧ K ∈ line A B) (hL_AD : ∃ A D, line_through A D ∧ L ∈ line A D): 
  parallelogram_area ABCD = 2 * real.sqrt (p * q) := 
sorry

end area_parallelogram_l365_365604


namespace pyramid_volume_eq_l365_365415

open Real

def volume_of_pyramid_cones (l : ℝ) (α β : ℝ) : ℝ :=
  let o1o2 := l * sin (2 * α)
  let bo2 := l * (sin α * cos α)
  let ao3 := l * cos β
  let bc := l * cos α * sqrt (1 / sqrt 2 * (1 + cos (π / 4)))
  1 / 3 * o1o2 * ao3 * bo2 * bc

theorem pyramid_volume_eq :
  let l : ℝ := 6
  let α : ℝ := π / 8
  let β : ℝ := π / 4
  volume_of_pyramid_cones l α β = 9 * sqrt (sqrt 2 + 1) :=
by
  sorry

end pyramid_volume_eq_l365_365415


namespace hiking_days_calculation_l365_365112

noncomputable def hike_days (pack_weight resupply_percentage weight_per_mile speed hours_per_day target_weight : ℕ) : ℕ :=
  let initial_miles := pack_weight / weight_per_mile in
  let resupply_weight := resupply_percentage * pack_weight / 100 in
  let resupply_miles := resupply_weight / weight_per_mile in
  let total_miles := initial_miles + resupply_miles in
  let total_hours := total_miles / speed in
  total_hours / hours_per_day

theorem hiking_days_calculation : 
  hike_days 40 25 0.5 2.5 8 40 = 5 :=
begin
  sorry
end

end hiking_days_calculation_l365_365112


namespace number_of_cities_l365_365414

theorem number_of_cities (n : ℕ) (h : n * (n - 1) / 2 = 15) : n = 6 :=
sorry

end number_of_cities_l365_365414


namespace distance_from_P_to_AB_l365_365053

variables {A B C P : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]

-- Defining the given conditions and question
def altitude_triangle_C_to_AB (h : ℝ) : Prop := h = 2

def area_ratio (area_1 area_2 : ℝ) : Prop := area_1 / area_2 = 1/3

def distance_P_to_AB (dist : ℝ) : Prop := dist = -1 + sqrt 3

-- The ultimate goal to prove
theorem distance_from_P_to_AB 
  (h : ℝ) 
  (area_1 area_2 : ℝ) 
  (dist : ℝ) 
  (alt_cond : altitude_triangle_C_to_AB h)
  (area_cond : area_ratio area_1 area_2) : 
  distance_P_to_AB dist :=
begin
  sorry,
end

end distance_from_P_to_AB_l365_365053


namespace identify_roles_l365_365050

-- Definitions
def person := Type
variables (Yaroslav Kirill Andrey : person)

def says (p : person) (s : person → Prop) : Prop := sorry

-- Conditions
def tells_truth (p : person) : Prop := sorry
def always_lies (p : person) : Prop := sorry
def cunning (p : person) : Prop := sorry

-- Statements made by the friends
def statement_Yaroslav := says Yaroslav (λ p, p = Kirill → always_lies Kirill)
def statement_Kirill := says Kirill (λ p, Kirill = Kirill → cunning Kirill)
def statement_Andrey := says Andrey (λ p, p = Kirill → tells_truth Kirill)

-- Main theorem
theorem identify_roles :
  (tells_truth Yaroslav ∧ cunning Kirill ∧ always_lies Andrey) ∨
  (tells_truth Kirill ∧ cunning Andrey ∧ always_lies Yaroslav) ∨
  (tells_truth Andrey ∧ cunning Yaroslav ∧ always_lies Kirill) :=
sorry

end identify_roles_l365_365050


namespace closest_integer_to_2_plus_sqrt_15_l365_365445

theorem closest_integer_to_2_plus_sqrt_15 :
  (3 < Real.sqrt 15) ∧ (Real.sqrt 15 < 4) → Int.closest (2 + Real.sqrt 15) = 6 :=
by 
  intro h
  sorry

end closest_integer_to_2_plus_sqrt_15_l365_365445


namespace maximize_y_l365_365635

noncomputable def z (θ : ℝ) : ℂ := 3 * Complex.cos θ + Complex.i * 2 * Complex.sin θ

def upper_half_plane (θ : ℝ) : Prop :=
  0 < θ ∧ θ < Real.pi / 2

def y (θ : ℝ) : ℝ := θ - Complex.arg (z θ)

theorem maximize_y :
  ∃ θ : ℝ, upper_half_plane θ ∧ θ = Real.arctan (Real.sqrt 6 / 2) :=
sorry

end maximize_y_l365_365635


namespace total_students_in_college_l365_365457

theorem total_students_in_college (B G : ℕ) (h_ratio: 8 * G = 5 * B) (h_girls: G = 175) :
  B + G = 455 := 
  sorry

end total_students_in_college_l365_365457


namespace smallest_four_digit_number_divisible_by_8_with_three_odd_one_even_is_1032_l365_365068

theorem smallest_four_digit_number_divisible_by_8_with_three_odd_one_even_is_1032 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 8 = 0) ∧
  (∀ d ∈ [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10], d % 2 = 1) ∧
  (∃ d ∈ [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10], d % 2 = 0) ∧
  (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ (m % 8 = 0) ∧
  (∀ d ∈ [m / 1000 % 10, m / 100 % 10, m / 10 % 10, m % 10], d % 2 = 1) ∧
  (∃ d ∈ [m / 1000 % 10, m / 100 % 10, m / 10 % 10, m % 10], d % 2 = 0) →
  1032 ≤ m) :=
begin
  sorry
end

end smallest_four_digit_number_divisible_by_8_with_three_odd_one_even_is_1032_l365_365068


namespace quadrilateral_area_is_nine_l365_365058

structure Point where
  x : ℕ
  y : ℕ

def A : Point := {x := 6, y := 1}
def B : Point := {x := 1, y := 6}
def C : Point := {x := 4, y := 3}
def D : Point := {x := 8, y := 8}

def shoelace_area (p1 p2 p3 p4 : Point) : ℤ :=
  Int.ofNat (1 / 2 * ((p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) -
                     (p1.y * p2.x + p2.y * p3.x + p3.y * p4.x + p4.y * p1.x)))

theorem quadrilateral_area_is_nine
  (p1 p2 p3 p4 : Point) :
  shoelace_area p1 p2 p3 p4 = 9 :=
by
  -- Definition of the points based on the problem statement
  let A := {x := 6, y := 1}
  let B := {x := 1, y := 6}
  let C := {x := 4, y := 3}
  let D := {x := 8, y := 8}
  -- Apply Shoelace Theorem formula and assert
  sorry

end quadrilateral_area_is_nine_l365_365058


namespace smile_shaped_region_area_l365_365274

theorem smile_shaped_region_area :
  (let r1 := 2 in
  let r2 := 3 in
  let theta := Real.arctan (2 / 3) in
  let semicircle_area := (π * r1^2) / 2 in
  let sector1_area := (π * r2^2) / 2 in
  let sector2_area := π * r2^2 * (theta / π) in
  let smile_area := (semicircle_area + sector1_area - sector2_area) in
  smile_area = (9 * π / 2) - (27 * theta / π)) :=
sorry

end smile_shaped_region_area_l365_365274


namespace find_quadratic_function_l365_365582

open Function

-- Define the quadratic function g(x) with parameters c and d
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c * x + d

-- State the main theorem
theorem find_quadratic_function :
  ∃ (c d : ℝ), (∀ x : ℝ, (g c d (g c d x + x)) / (g c d x) = x^2 + 120 * x + 360) ∧ c = 119 ∧ d = 240 :=
by
  sorry

end find_quadratic_function_l365_365582


namespace football_basketball_problem_l365_365115

theorem football_basketball_problem :
  ∃ (football_cost basketball_cost : ℕ),
    (3 * football_cost + basketball_cost = 230) ∧
    (2 * football_cost + 3 * basketball_cost = 340) ∧
    football_cost = 50 ∧
    basketball_cost = 80 ∧
    ∃ (basketballs footballs : ℕ),
      (basketballs + footballs = 20) ∧
      (footballs < basketballs) ∧
      (80 * basketballs + 50 * footballs ≤ 1400) ∧
      ((basketballs = 11 ∧ footballs = 9) ∨
       (basketballs = 12 ∧ footballs = 8) ∨
       (basketballs = 13 ∧ footballs = 7)) :=
by
  sorry

end football_basketball_problem_l365_365115


namespace koschei_coin_count_l365_365305

theorem koschei_coin_count (a : ℕ) :
  (a % 10 = 7) ∧
  (a % 12 = 9) ∧
  (300 ≤ a ∧ a ≤ 400) →
  a = 357 :=
sorry

end koschei_coin_count_l365_365305


namespace bruno_initial_books_l365_365519

theorem bruno_initial_books (X : ℝ)
  (h1 : X - 4.5 + 10.25 = 39.75) :
  X = 34 := by
  sorry

end bruno_initial_books_l365_365519


namespace paint_chessboard_l365_365840

variables (m n : ℕ)

def valid_colors (m n : ℕ) : Prop :=
  (m >= 2) ∧ (n >= 2)

theorem paint_chessboard (h : valid_colors m n) :
  (∃ a_n : ℕ, a_n = (m-1)^n + (-1)^n * (m-1)) :=
sorry

end paint_chessboard_l365_365840


namespace smallest_natural_greater_than_1_with_divisors_properties_l365_365182

theorem smallest_natural_greater_than_1_with_divisors_properties :
  ∃ (N : ℕ), N > 1 ∧ (∀ p : ℕ, prime p ∧ p ∣ N → N ≥ 600 * p) ∧ 
  (∀ M : ℕ, M > 1 ∧ (∀ q : ℕ, prime q ∧ q ∣ M → M ≥ 600 * q) → N ≤ M) ∧ N = 1944 :=
by
  sorry

end smallest_natural_greater_than_1_with_divisors_properties_l365_365182


namespace estimate_red_balls_is_correct_l365_365080

noncomputable def estimated_red_balls (total_balls : ℕ) (red_ball_frequency : ℝ) : ℕ :=
  (total_balls : ℕ) * red_ball_frequency.toNNReal.to_nN

theorem estimate_red_balls_is_correct :
  estimated_red_balls 1000 0.2 = 200 := sorry

end estimate_red_balls_is_correct_l365_365080


namespace y_intercept_tangent_line_l365_365104

noncomputable def tangent_line_y_intercept (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (htangent: Prop) : ℝ :=
  if r1 = 3 ∧ r2 = 2 ∧ c1 = (3, 0) ∧ c2 = (8, 0) ∧ htangent = true then 6 * Real.sqrt 6 else 0

theorem y_intercept_tangent_line (h : tangent_line_y_intercept 3 2 (3, 0) (8, 0) true = 6 * Real.sqrt 6) :
  tangent_line_y_intercept 3 2 (3, 0) (8, 0) true = 6 * Real.sqrt 6 :=
by
  exact h

end y_intercept_tangent_line_l365_365104


namespace f_f_neg3_eq_1_l365_365601

-- Define the piecewise function f(x)
def f : ℝ → ℝ
| x := if x ≤ 0 then f (x + 1) else -x^2 + 2*x

-- State the theorem that needs to be proved
theorem f_f_neg3_eq_1 : f (f (-3)) = 1 :=
sorry

end f_f_neg3_eq_1_l365_365601


namespace black_area_fraction_after_three_changes_l365_365920

theorem black_area_fraction_after_three_changes (initial_area : ℝ) :
  ∀ (changes : ℕ), changes = 3 → 
  ((8 / 9) * (8 / 9) * (8 / 9) * initial_area) / initial_area = 512 / 729 :=
begin
  intros changes h,
  rw h,
  have frac_eqn : (8 / 9 : ℝ) ^ 3 = 512 / 729,
  { norm_num },
  rw frac_eqn,
  field_simp,
  ring,
end

end black_area_fraction_after_three_changes_l365_365920


namespace binomial_expansion_equivalence_l365_365979

theorem binomial_expansion_equivalence 
  (x : ℝ)
  (a : ℕ → ℝ)
  (h : (x - 1)^11 = ∑ k in Finset.range 12, a k * (x - 3)^k) :
  a 9 = 220 →
  (a 1 + a 3 + a 5 + a 7 + a 11) / a 9 = (3^11 - 441) / 440 :=
by
  intro h_a9
  sorry

end binomial_expansion_equivalence_l365_365979


namespace sum_double_series_l365_365535

theorem sum_double_series :
  (∑ n from 2 to ∞, ∑ k from 1 to (n-1), k / 3^(n+k)) = 9 / 136 :=
by
  sorry

end sum_double_series_l365_365535


namespace convex_polygon_symmetry_iff_sum_of_segments_l365_365362

theorem convex_polygon_symmetry_iff_sum_of_segments
  (P : Type*) [convex_polygon P] [has_center_of_symmetry P] :
  (∃ segments : list (line_segment P), P = segments.sum) ↔ (∃ O : point P, P.has_center_of_symmetry O) := 
sorry

end convex_polygon_symmetry_iff_sum_of_segments_l365_365362


namespace centroid_area_l365_365779

-- Definitions of the conditions
def isSquare (ABCD : quadrilateral) (s : ℝ) : Prop :=
 ∀ (A B C D : point), dist A B = s ∧ dist B C = s ∧ dist C D = s ∧ dist D A = s

def isInside (P : point) (A B C D : point) : Prop :=
 dist A P < dist A B ∧ dist B P < dist B C ∧ dist C P < dist C D ∧ dist D P < dist D A

def formConvexQuadrilateral (Gs: list point) : Prop :=
 ∃ (G1 G2 G3 G4 : point), Gs = [G1, G2, G3, G4] ∧ convex_quadrilateral G1 G2 G3 G4

-- Main theorem statement
theorem centroid_area
  {ABCD : quadrilateral}
  {P A B C D : point}
  {AP BP : ℝ}
  (hSquare : isSquare ABCD 30)
  (hInside : isInside P A B C D)
  (hAP : dist A P = 12)
  (hBP : dist B P = 26)
  (G1 G2 G3 G4 : point)
  (hCentroids : formConvexQuadrilateral [G1, G2, G3, G4]) :
  area [G1, G2, G3, G4] = 200 :=
sorry

end centroid_area_l365_365779


namespace solveExpression_l365_365879

noncomputable def evaluateExpression : ℝ := (Real.sqrt 3) / Real.sin (Real.pi / 9) - 1 / Real.sin (7 * Real.pi / 18)

theorem solveExpression : evaluateExpression = 4 :=
by sorry

end solveExpression_l365_365879


namespace length_of_paper_l365_365123

-- Definitions
def paper_width : ℝ := 4   -- width of the paper in cm
def wraps : ℕ := 800      -- number of wraps
def final_diameter : ℝ := 14  -- final diameter in cm
def initial_diameter : ℝ := 1.5  -- initial diameter in cm
def half_initial_diameter : ℝ := initial_diameter / 2  -- initial radius in cm
def half_final_diameter : ℝ := final_diameter / 2  -- final radius in cm
def increase_in_radius_per_wrap : ℝ := paper_width / 2  -- increase in radius per wrap in cm

-- Theorem statement
theorem length_of_paper :
  (half_final_diameter + increase_in_radius_per_wrap * (wraps - 1)) * π = 62 * π := by
  sorry

end length_of_paper_l365_365123


namespace smallest_N_for_right_triangle_l365_365973

theorem smallest_N_for_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  ∃ N, ∀ (a b c : ℝ), (a^2 + b^2 = c^2) → (a^2 + b^2 - c^2) / (a^2 + b^2 + c^2) < N ∧ N = 0 :=
begin
  existsi 0,
  intros a b c h,
  rw [h],
  have h1 : (a^2 + b^2 - c^2) / (a^2 + b^2 + c^2) = 0,
  { rw h,
    calc (a^2 + b^2 - c^2) / (a^2 + b^2 + c^2)
        = (a^2 + b^2 - (a^2 + b^2)) / (a^2 + b^2 + (a^2 + b^2)) : by rw h
    ... = 0 / (a^2 + b^2 + a^2 + b^2) : by ring
    ... = 0 : by simp },
  exact lt_of_le_of_ne (le_of_eq h1) (ne_of_lt (zero_lt_one)),
end

end smallest_N_for_right_triangle_l365_365973


namespace no_centrally_symmetric_polygon_with_one_line_of_symmetry_l365_365297

theorem no_centrally_symmetric_polygon_with_one_line_of_symmetry :
  ¬ ∃ (S : Type) [polygon S] (O : point) (t : line), centrally_symmetric S O ∧ (symmetry_axes S = {t}) := 
sorry

end no_centrally_symmetric_polygon_with_one_line_of_symmetry_l365_365297


namespace polynomial_roots_and_divisibility_l365_365318

-- Definition of a commutative field
variables {K : Type*} [Field K]

-- Definitions of polynomials P and Q with their constraints
variable {d : ℕ}
variable {P Q : K[X]}
variable (hQ_nonconst : Q.degree > 0)

-- Theorem statement
theorem polynomial_roots_and_divisibility
  (hP_degree : P.degree = degree_nat d) 
  (hQ_div : P ∣ Q^2) :
  (∀ x : K, multiplicity x P ≤ d) ∧ (Q ∣ P) ∧ (Q ∣ P.derivative)
:= 
sorry

end polynomial_roots_and_divisibility_l365_365318


namespace correct_options_incorrect_options_l365_365994

-- Given definitions and conditions
section
variable (z1 z2 : ℂ)

-- Defining the conditions for option B
def condition_B : Prop := z1 * z2 = 0 → z1 = 0 ∨ z2 = 0

-- Defining the conditions for option D
def condition_D : Prop := (|z1| = 1) → (|z2| = 1) → (z1 + z2 = 1) → (|z1 - z2| = Real.sqrt 3)

-- Math proof problem stating that options B and D are correct.
theorem correct_options : (condition_B z1 z2) ∧ (condition_D z1 z2) := by
  sorry
end

-- Incorrect options A and C (optional)
-- We do not actually need to write proofs for these in the Lean statement; just for completeness:
section
-- Defining the false conditions for option A and C
def condition_A : Prop := (z1 ^ 2 + 1 = 0) → (z1 = Complex.I)
def condition_C : Prop := (|z1| = |z2|) → (z1 ^ 2 = z2 ^ 2)

-- Math proof problem stating that options A and C are incorrect.
theorem incorrect_options : ¬(condition_A z1) ∧ ¬(condition_C z1 z2) := by
  sorry
end

end correct_options_incorrect_options_l365_365994


namespace symmetric_points_x_axis_sum_l365_365204

theorem symmetric_points_x_axis_sum (x y : ℝ) (h₁ : P = (x, -3)) (h₂ : Q = (4, y)) (h_symm : ∀ P Q, P.y = -Q.y ∧ P.x = Q.x) : x + y = 7 :=
by
  sorry

end symmetric_points_x_axis_sum_l365_365204


namespace tie_rate_correct_l365_365277

-- Define the fractions indicating win rates for Amy, Lily, and John
def AmyWinRate : ℚ := 4 / 9
def LilyWinRate : ℚ := 1 / 3
def JohnWinRate : ℚ := 1 / 6

-- Define the fraction they tie
def TieRate : ℚ := 1 / 18

-- The theorem for proving the tie rate
theorem tie_rate_correct : AmyWinRate + LilyWinRate + JohnWinRate = 17 / 18 → (1 : ℚ) - (17 / 18) = TieRate :=
by
  sorry -- Proof is omitted

-- Define the win rate sums and tie rate equivalence
example : (AmyWinRate + LilyWinRate + JohnWinRate = 17 / 18) ∧ (TieRate = 1 - 17 / 18) :=
by
  sorry -- Proof is omitted

end tie_rate_correct_l365_365277


namespace angles_GAC_EAC_are_equal_l365_365282

-- Define the quadrilateral with given properties
structure Quadrilateral (ℝ : Type*) :=
(A B C D E F G : ℝ × ℝ)
(AC_bisects_∠BAD : sorry)
(E_on_CD : E ∈ line_through C D)
(BE_intersects_AC_at_F : line_through B E ∩ line_through A C = {F})
(DF_intersects_BC_at_G : line_through D F ∩ line_through B C = {G})

-- Formalize the problem statement in Lean
theorem angles_GAC_EAC_are_equal {ℝ : Type*} [linear_ordered_field ℝ]
  (A B C D E F G : ℝ × ℝ)
  (q : Quadrilateral ℝ)
  (h1 : q.AC_bisects_∠BAD)
  (h2 : q.E_on_CD)
  (h3 : q.BE_intersects_AC_at_F)
  (h4 : q.DF_intersects_BC_at_G) :
  angle q.G q.A q.C = angle q.E q.A q.C :=
sorry

end angles_GAC_EAC_are_equal_l365_365282


namespace minimum_stages_l365_365047

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem minimum_stages {length_track length_stage : ℕ} (h1 : length_track = 660) 
  (h2 : length_stage = 150) : 
  ∃ n, length_stage * n = lcm length_track length_stage ∧ 
       n = 22 := 
by
  have h3 : lcm 660 150 = 3300 := by sorry
  use 22
  split
  case left =>
    show 150 * 22 = 3300 by sorry
  case right =>
    show 22 = 22 by rfl

end minimum_stages_l365_365047


namespace geometric_sequence_general_formula_no_arithmetic_sequence_l365_365405

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Condition: Sum of the first n terms of the sequence {a_n} is S_n
-- and S_n = 2a_n - n for n \in \mathbb{N}^*.
axiom sum_condition (n : ℕ) (h : n > 0) : S n = 2 * a n - n

-- Question 1: Prove that the sequence {a_n + 1} forms a geometric sequence.
theorem geometric_sequence (n : ℕ) (h : n > 0) : ∃ r, r ≠ 0 ∧ ∀ m, m > 0 → a (m + 1) + 1 = r * (a m + 1) := 
sorry

-- Question 2: Find the general formula for the sequence {a_n}.
theorem general_formula (n : ℕ) (h : n > 0) : a n = 2 ^ n - 1 := 
sorry

-- Question 3: Prove that there do not exist three consecutive terms in the sequence {a_n} that can form an arithmetic sequence.
theorem no_arithmetic_sequence (k : ℕ) (h : k > 0) : ¬ ∃ k, k > 0 ∧ a k = (a (k + 1) + a (k + 2)) / 2 := 
sorry

end geometric_sequence_general_formula_no_arithmetic_sequence_l365_365405


namespace problem_solution_l365_365573

noncomputable def solveProblem (x : ℝ) : Prop :=
  x > 0 ∧ x * sqrt (12 - x) + sqrt (12 * x - x^3) >= 12 ↔ x = 3

theorem problem_solution :
  ∀ x : ℝ, x > 0 → (x * sqrt (12 - x) + sqrt (12 * x - x^3) >= 12 → x = 3) :=
by
  intro x hx hineq
  have hcs := CauchySchwarzInequality _ _
  sorry

end problem_solution_l365_365573


namespace tangent_dot_product_is_three_half_l365_365203

noncomputable def circle_tangent_dot_product : ℝ :=
  let P := (1 : ℝ, real.sqrt 3) in
  let f : ℝ × ℝ → bool := λ ⟨x, y⟩, x^2 + y^2 = 1 in
  -- Assume A and B are the points of contact of the tangents from P
  let A := (1 : ℝ, 0) in -- placeholder values for points of contact
  let B := (-1 : ℝ, 0) in -- placeholder values for points of contact
  -- Calculate the dot product
  let PA := (A.1 - P.1, A.2 - P.2) in
  let PB := (B.1 - P.1, B.2 - P.2) in
  PA.1 * PB.1 + PA.2 * PB.2

theorem tangent_dot_product_is_three_half (hP : (1 : ℝ)^2 + (real.sqrt 3)^2 = 1) 
(hA: (1 : ℝ)^2 = 1)
(hB: (-1 : ℝ)^2 = 1) 
: circle_tangent_dot_product = 3 / 2 :=
by sorry

end tangent_dot_product_is_three_half_l365_365203


namespace union_sets_l365_365649

open Set

variable {α : Type*}

def A : Set ℝ := {x | -2 < x ∧ x < 2}

def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = 2^x}

theorem union_sets : A ∪ B = {z | -2 < z ∧ z < 4} :=
by sorry

end union_sets_l365_365649


namespace probability_four_green_marbles_l365_365916

theorem probability_four_green_marbles :
  let number_of_green := 10
  let number_of_purple := 5
  let total_marbles := number_of_green + number_of_purple
  let trials := 8
  let desired_green_marbles := 4
  (finset.card (finset.range(total_marbles).filter (λ x, x < number_of_green)) / total_marbles)^desired_green_marbles * 
  (finset.card (finset.range(total_marbles).filter (λ x, x >= number_of_green)) / total_marbles)^(trials - desired_green_marbles) *
  nat.choose(trials, desired_green_marbles) = 0.171 :=
by
  let number_of_green := 10
  let number_of_purple := 5
  let total_marbles := number_of_green + number_of_purple
  let trials := 8
  let desired_green_marbles := 4
  have h1 : (number_of_green / total_marbles : ℝ)^desired_green_marbles * 
            (number_of_purple / total_marbles : ℝ)^(trials - desired_green_marbles) * 
            nat.choose(trials, desired_green_marbles) = 0.171 := sorry
  exact h1

end probability_four_green_marbles_l365_365916


namespace luxury_class_adults_l365_365369

def total_passengers : ℕ := 300
def adult_percentage : ℝ := 0.70
def luxury_percentage : ℝ := 0.15

def total_adults (p : ℕ) : ℕ := (p * 70) / 100
def adults_in_luxury (a : ℕ) : ℕ := (a * 15) / 100

theorem luxury_class_adults :
  adults_in_luxury (total_adults total_passengers) = 31 :=
by
  sorry

end luxury_class_adults_l365_365369


namespace distance_between_foci_l365_365177

-- Define the ellipse
def ellipse_eq (x y : ℝ) := 9 * x^2 + 36 * y^2 = 1296

-- Define the semi-major and semi-minor axes
def semi_major_axis := 12
def semi_minor_axis := 6

-- Distance between the foci of the ellipse
theorem distance_between_foci : 
  (∃ x y : ℝ, ellipse_eq x y) → 2 * Real.sqrt (semi_major_axis^2 - semi_minor_axis^2) = 12 * Real.sqrt 3 :=
by
  sorry

end distance_between_foci_l365_365177


namespace jerry_total_income_l365_365714

/-- Jerry charges $20 to pierce someone's nose and 50% more to pierce their ears. 
    If he pierces 6 noses and 9 ears, prove that he makes $390. -/
theorem jerry_total_income :
  let nose_piercing_cost := 20 in
  let ear_piercing_cost := nose_piercing_cost + 0.5 * nose_piercing_cost in
  let total_noses := 6 in
  let total_ears := 9 in
  let total_income := (nose_piercing_cost * total_noses) + (ear_piercing_cost * total_ears) in
  total_income = 390 :=
by 
  sorry

end jerry_total_income_l365_365714


namespace f_divides_f_next_l365_365975

noncomputable def f (d : ℕ) : ℕ :=
  if h : d > 0 then Nat.find (Exists.min ⟨Nat.factorial d, by simp [Nat.factorial, Nat.divisors]⟩ h)
  else 0

theorem f_divides_f_next (k : ℕ) : f (2^k) ∣ f (2^(k+1)) :=
by
  sorry

end f_divides_f_next_l365_365975


namespace inclination_angle_l365_365843

theorem inclination_angle (x y : ℝ) (h : x + sqrt 3 * y + 5 = 0) : 
  ∃ θ : ℝ, θ = 150 ∧ tan θ = -1 / sqrt 3 :=
begin
  sorry
end

end inclination_angle_l365_365843


namespace stratified_sampling_third_grade_l365_365111

theorem stratified_sampling_third_grade :
  let total_students := 1000
    let first_grade_students := 380
    let male_students_second_grade := 180
    let prob_female_second_grade := 0.19
    let female_students_second_grade := total_students * prob_female_second_grade
    let total_second_grade_students := male_students_second_grade + female_students_second_grade
    let total_third_grade_students := total_students - total_second_grade_students - first_grade_students
    let selected_students_third_grade := (total_third_grade_students / total_students) * 100
  in selected_students_third_grade = 25 :=
by
  -- This is where the proof would go
  sorry

end stratified_sampling_third_grade_l365_365111


namespace correct_sum_p_q_l365_365022

noncomputable def p (x : ℝ) : ℝ := sorry
noncomputable def q (x : ℝ) : ℝ := sorry

theorem correct_sum_p_q :
  (∀ x, (abs x) > 0 → (p(x) / q(x)) → 0) ∧
  (∀ x, x = 2 → (abs q(x) → ∞)) ∧
  (∃ a b c, q(x) = a * x^3 + b * x^2 + c * x + d ∧ a ≠ 0) ∧
  (p(3) = 2) ∧
  (q(3) = 4) → 
  (p(x) + q(x) = (2/5)*x^3 - (4/5)*x^2 + (2/5)*x + (8/5)) :=
by
  sorry

end correct_sum_p_q_l365_365022


namespace walkways_area_correct_l365_365783

def garden : Type :=
  { flower_beds : Nat → (Nat × Nat), walkways_width : Nat }

def garden_conditions (g : garden) : Prop :=
  g.flower_beds = (λ n, if n < 6 then (8, 3) else (0, 0)) ∧ g.walkways_width = 1

def total_area (g : garden) : Nat :=
  let total_width := 2 * 8 + 4 * g.walkways_width
  let total_height := 3 * 3 + 4 * g.walkways_width
  total_width * total_height

def flower_beds_area (g : garden) : Nat :=
  6 * 8 * 3

def walkways_area (g : garden) : Nat :=
  total_area g - flower_beds_area g

theorem walkways_area_correct :
  ∀ g : garden,
    garden_conditions g →
    walkways_area g = 116 := by
  sorry

end walkways_area_correct_l365_365783


namespace bottle_caps_per_box_l365_365302

theorem bottle_caps_per_box (total_caps : ℕ) (total_boxes : ℕ) (caps_per_box : ℕ) (h1 : total_caps = 316) (h2 : total_boxes = 79) : caps_per_box = 4 :=
by
  have h3 : caps_per_box = total_caps / total_boxes, from sorry
  rw [h1, h2] at h3
  exact h3

end bottle_caps_per_box_l365_365302


namespace min_value_l365_365597

theorem min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ (1 / x + 4 / y) ≥ 9) :=
by
  sorry

end min_value_l365_365597


namespace angle_between_vectors_alpha_beta_values_l365_365240

-- Define vector a and b
variables (α β : ℝ) (a b c : ℝ × ℝ)
noncomputable def va : ℝ × ℝ := (Real.cos α, Real.sin α)
noncomputable def vb : ℝ × ℝ := (Real.cos β, Real.sin β)
noncomputable def vc : ℝ × ℝ := (0, 1)

-- Define the magnitude of a vector
noncomputable def vector_mag (v : ℝ × ℝ) : ℝ := 
  Real.sqrt (v.1^2 + v.2^2)

-- Define the subtraction of vectors
noncomputable def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := 
  (v1.1 - v2.1, v1.2 - v2.2)

-- The angle between two vectors
noncomputable def vector_angle (v1 v2 : ℝ × ℝ) : ℝ := 
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (vector_mag v1 * vector_mag v2))

-- Define the conditions for the first proof
def condition1 := 0 < β ∧ β < α ∧ α < π ∧ vector_mag (vector_sub va vb) = Real.sqrt 2

-- Prove that the angle between a and b is pi / 2
theorem angle_between_vectors (h : condition1) : vector_angle va vb = π / 2 := sorry

-- Define the conditions for the second proof
def condition2 := 0 < β ∧ β < α ∧ α < π ∧ (va.1 + vb.1 = 0) ∧ (va.2 + vb.2 = 1)

-- Prove the values of alpha and beta
theorem alpha_beta_values (h : condition2) : α = 5 * π / 6 ∧ β = π / 6 := sorry

end angle_between_vectors_alpha_beta_values_l365_365240


namespace cost_per_package_l365_365350

theorem cost_per_package
  (parents : ℕ)
  (brothers : ℕ)
  (spouses_per_brother : ℕ)
  (children_per_brother : ℕ)
  (total_cost : ℕ)
  (num_packages : ℕ)
  (h1 : parents = 2)
  (h2 : brothers = 3)
  (h3 : spouses_per_brother = 1)
  (h4 : children_per_brother = 2)
  (h5 : total_cost = 70)
  (h6 : num_packages = parents + brothers + brothers * spouses_per_brother + brothers * children_per_brother) :
  total_cost / num_packages = 5 :=
by
  -- Proof goes here
  sorry

end cost_per_package_l365_365350


namespace felix_chopped_at_least_91_trees_l365_365957

def cost_to_sharpen := 5
def total_spent := 35
def trees_per_sharpen := 13

theorem felix_chopped_at_least_91_trees :
  (total_spent / cost_to_sharpen) * trees_per_sharpen = 91 := by
  sorry

end felix_chopped_at_least_91_trees_l365_365957


namespace cos_tan_neg_iff_quadrant_l365_365193

open Real

-- We define the quadrants using Lean structures/enums if necessary
inductive Quadrant
| First
| Second
| Third
| Fourth

def quadrant (θ : ℝ) : Quadrant :=
  if 0 ≤ θ ∧ θ < π / 2 then Quadrant.First
  else if π / 2 ≤ θ ∧ θ < π then Quadrant.Second
  else if π ≤ θ ∧ θ < 3 * π / 2 then Quadrant.Third
  else Quadrant.Fourth

theorem cos_tan_neg_iff_quadrant (θ : ℝ) (h : cos θ * tan θ < 0) :
  quadrant θ = Quadrant.Third ∨ quadrant θ = Quadrant.Fourth :=
  sorry

end cos_tan_neg_iff_quadrant_l365_365193


namespace find_sequences_and_max_n_l365_365633

variables {a b : ℕ → ℝ}

-- Given conditions
def S (n : ℕ) : ℝ := a n + n^2 - 1
def b_condition (n : ℕ) : ℝ := (n + 1) * a (n + 1) - n * a n
def b_1 : ℝ := 3

-- The statements to be proven
theorem find_sequences_and_max_n :
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, b (n + 1) = (4 * n + 3) / (3^n)) ∧
  (∃ n, n ≤ 3 ∧ T n < 7) :=
sorry

end find_sequences_and_max_n_l365_365633


namespace quadratic_roots_solution_l365_365210

theorem quadratic_roots_solution:
  ∃ (a b p q: ℝ), 
  (∀ x: ℂ, x^2 + (p:ℂ)*x + (q:ℂ) = 0 ↔ x = 2 + complex.I*a ∨ x = b + complex.I) ∧ 
  a = -1 ∧ b = 2 ∧ p = -4 ∧ q = 5 ∧ 
  (a + complex.I * b) / (p + complex.I * q) = 3/41 + complex.I * (6/41) :=
by {
  sorry
}

end quadratic_roots_solution_l365_365210


namespace renovation_services_are_credence_goods_and_choice_arguments_l365_365459

-- Define what credence goods are and the concept of information asymmetry
structure CredenceGood where
  information_asymmetry : Prop
  unobservable_quality  : Prop

-- Define renovation service as an instance of CredenceGood
def RenovationService : CredenceGood := {
  information_asymmetry := true,
  unobservable_quality := true
}

-- Primary conditions for choosing between construction company and private repair crew
structure ChoiceArgument where
  information_availability     : Prop
  warranty_and_accountability  : Prop
  higher_costs                 : Prop
  potential_bias_in_reviews    : Prop

-- Arguments for using construction company
def ConstructionCompanyArguments : ChoiceArgument := {
  information_availability := true,
  warranty_and_accountability := true,
  higher_costs := true,
  potential_bias_in_reviews := true
}

-- Arguments against using construction company
def PrivateRepairCrewArguments : ChoiceArgument := {
  information_availability := false,
  warranty_and_accountability := false,
  higher_costs := true,
  potential_bias_in_reviews := true
}

-- Proof statement to show renovation services are credence goods and economically reasoned arguments for/against
theorem renovation_services_are_credence_goods_and_choice_arguments:
  RenovationService = {
    information_asymmetry := true,
    unobservable_quality := true
  } ∧
  (ConstructionCompanyArguments.information_availability = true ∧
   ConstructionCompanyArguments.warranty_and_accountability = true) ∧
  (ConstructionCompanyArguments.higher_costs = true ∧
   ConstructionCompanyArguments.potential_bias_in_reviews = true) ∧
  (PrivateRepairCrewArguments.higher_costs = true ∧
   PrivateRepairCrewArguments.potential_bias_in_reviews = true) :=
by sorry

end renovation_services_are_credence_goods_and_choice_arguments_l365_365459


namespace median_of_scores_l365_365767

def scores : List ℕ := [35, 38, 40, 40, 42, 42, 45]

theorem median_of_scores : median scores = 40 :=
by
  sorry

end median_of_scores_l365_365767


namespace probability_AC_lt_7_is_one_third_l365_365511

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def probability_AC_lt_7 : ℝ :=
  if h : ∀ A B C : ℝ × ℝ, distance A B = 8 ∧ distance B C = 5 ∧ ∀ α ∈ (0, π), true ∧ distance A C < 7
  then 1 / 3
  else 0

theorem probability_AC_lt_7_is_one_third :
  probability_AC_lt_7 = 1 / 3 :=
  by sorry

end probability_AC_lt_7_is_one_third_l365_365511


namespace rational_terms_binomial_expansion_l365_365632

theorem rational_terms_binomial_expansion (x : ℝ) (n : ℕ) 
    (h1 : (2^(n-1) = 512)) 
    : n = 10 ∧ (∀ r ∈ {0, 6}, ∀ k, k = 5 - r / 6 → 
                  (∃ c : ℝ, c ≡ if r = 0 then x^5 else 
                                   if r = 6 then real.coe_binom 10 4 * x^4 else 0)) := by
  sorry

end rational_terms_binomial_expansion_l365_365632


namespace explicit_formula_of_f_tangent_line_at_A_l365_365745

noncomputable def f (x a : ℝ) := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

theorem explicit_formula_of_f :
  ∃ a : ℝ, (∀ x : ℝ, f x a = 2 * x^3 - 12 * x^2 + 18 * x + 8) :=
by
  let a := 3
  have h : f'(3) = 0 := sorry
  use a
  intros x
  calc
    f x a = 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8 := rfl
       ... = 2 * x^3 - 12 * x^2 + 18 * x + 8 := sorry

theorem tangent_line_at_A :
  ∀ x : ℝ, (f x 3 = 2 * x^3 - 12 * x^2 + 18 * x + 8) → ∀ y : ℝ, (f 1 3 = 16) → y = 16 :=
by
  intros x fx_eq y hy
  have tangent : f'(1) = 0 := sorry
  show y = 16
  ... := sorry

end explicit_formula_of_f_tangent_line_at_A_l365_365745


namespace area_of_union_l365_365493

noncomputable theory

-- Definitions based on the given conditions
def side_length := 8
def radius := 12
def area_square := side_length ^ 2
def area_circle := Real.pi * radius ^ 2
def overlap := (1 / 4) * area_circle
def area_union := area_square + area_circle - overlap

-- The theorem stating the desired proof
theorem area_of_union (side_length radius : ℝ) (h_side : side_length = 8) (h_radius : radius = 12) :
  (side_length ^ 2 + Real.pi * radius ^ 2 - (1 / 4) * Real.pi * radius ^ 2) = 64 + 108 * Real.pi :=
by
  rw [h_side, h_radius]
  simp [area_square, area_circle, overlap, area_union]
  sorry

end area_of_union_l365_365493


namespace bakery_baguettes_l365_365011

theorem bakery_baguettes : 
  ∃ B : ℕ, 
  (∃ B : ℕ, 3 * B - 138 = 6) ∧ 
  B = 48 :=
by
  sorry

end bakery_baguettes_l365_365011


namespace car_traverse_possible_l365_365757

-- Define the main condition in the problem
noncomputable def fuel_station_symmetric (n : ℕ) : Prop :=
  -- On a spherical planet, there are 2n fuel stations symmetrically positioned with respect to the center of the sphere
  ∃ s : set (ℝ × ℝ × ℝ), s.card = 2 * n ∧ ∀ x ∈ s, ∃ y ∈ s, x ≠ y ∧ x + y = (0, 0, 0)

-- Define the condition that the car can traverse all stations
noncomputable def car_can_traverse (n : ℕ) : Prop :=
  (fuel_station_symmetric n) ∧ n ∈ {1, 2, 3}

-- Lean theorem statement
theorem car_traverse_possible : ∀ (n : ℕ), car_can_traverse n ↔ n = 1 ∨ n = 2 ∨ n = 3 :=
begin
  sorry -- Proof not required
end

end car_traverse_possible_l365_365757


namespace mode_of_given_data_set_l365_365813

namespace mode_proof

def data_set : List Int := [-1, 0, 2, -1, 3]

def mode (lst : List Int) : Int :=
(lst.map (λ x => (x, lst.count x))).maxBy (λ x => x.2).1

theorem mode_of_given_data_set : mode data_set = -1 :=
by
  sorry -- Proof is omitted for now, we are only writing the statement.

end mode_proof

end mode_of_given_data_set_l365_365813


namespace monkeys_banana_distribution_l365_365417

theorem monkeys_banana_distribution
    (a b c : ℕ) -- The number of bananas initially taken by monkeys A, B, and C, respectively
    (h1 : a + b + c = 540) -- Total bananas being 540
    (h2 : (∃ k : ℕ, k * 270 + k * 162 + k * 108 = 540)) -- Given the ratio 5:3:2 
    (h3 : (1 / 2) * a ∈ ℤ) -- A keeps half of a 
    (h4 : (1 / 3) * b ∈ ℤ) -- B keeps third of b
    (h5 : (1 / 4) * c ∈ ℤ) -- C keeps quarter of c
    : 5 * 54 = 270 ∧ 3 * 54 = 162 ∧ 2 * 54 = 108 := sorry

end monkeys_banana_distribution_l365_365417


namespace total_legs_in_room_l365_365827

def count_legs : Nat :=
  let tables_4_legs := 4 * 4
  let sofas_legs := 1 * 4
  let chairs_4_legs := 2 * 4
  let tables_3_legs := 3 * 3
  let tables_1_leg := 1 * 1
  let rocking_chair_legs := 1 * 2
  tables_4_legs + sofas_legs + chairs_4_legs + tables_3_legs + tables_1_leg + rocking_chair_legs

theorem total_legs_in_room : count_legs = 40 := by
  sorry

end total_legs_in_room_l365_365827


namespace girls_first_half_l365_365049

theorem girls_first_half (total_students boys_first_half girls_first_half boys_second_half girls_second_half boys_whole_year : ℕ)
  (h1: total_students = 56)
  (h2: boys_first_half = 25)
  (h3: girls_first_half = 15)
  (h4: boys_second_half = 26)
  (h5: girls_second_half = 25)
  (h6: boys_whole_year = 23) : 
  ∃ girls_first_half_only : ℕ, girls_first_half_only = 3 :=
by {
  sorry
}

end girls_first_half_l365_365049


namespace edward_original_amount_l365_365568

-- Given conditions
def spent : ℝ := 16
def remaining : ℝ := 6

-- Question: How much did Edward have before he spent his money?
-- Correct answer: 22
theorem edward_original_amount : (spent + remaining) = 22 :=
by sorry

end edward_original_amount_l365_365568


namespace general_term_formula_l365_365202

-- Conditions
def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n, a (n + 1) = a n + d

-- Arithmetic sequence properties
variables (a : ℕ → ℤ) (d : ℤ) (m : ℕ)

-- Given conditions
axiom odd_m : is_odd m
axiom sum_first_m_terms : ∑ i in finset.range m, a i = 77
axiom sum_even_terms : ∑ i in (finset.range m).filter (λ n, n % 2 = 1), a i = 33
axiom diff_first_last : a 0 - a (m - 1) = 18

-- General term to prove
def general_term (a : ℕ → ℤ) (n : ℕ) : ℤ := 20 + (n - 1) * (-3)

-- Theorem to prove
theorem general_term_formula : is_arithmetic_sequence a (-3) → (∀ n, a n = -3 * n + 23) := by
  sorry

end general_term_formula_l365_365202


namespace felix_trees_chopped_l365_365955

theorem felix_trees_chopped (trees_per_sharpen : ℕ) (cost_per_sharpen : ℕ) (total_spent : ℕ) (trees_chopped : ℕ) :
  trees_per_sharpen = 13 →
  cost_per_sharpen = 5 →
  total_spent = 35 →
  trees_chopped = (total_spent / cost_per_sharpen) * trees_per_sharpen →
  trees_chopped ≥ 91 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  have : (35 / 5) * 13 = 91 := sorry
  rw this at h4
  exact le_of_eq h4

end felix_trees_chopped_l365_365955


namespace largest_nine_element_subset_l365_365056

-- Definition of the triangle property for a subset
def triangle_property (S : Set ℕ) : Prop :=
  ∀ x y z: ℕ, x < y → y < z → z ∈ S → x ∈ S → y ∈ S → z < x + y

-- The main statement to prove
theorem largest_nine_element_subset (a : ℕ) (h : a > 3) (n : ℕ) :
  (∀ (S : Set ℕ), (∃ (S' : Finset ℕ), S'.card = 9 ∧ ∀ s ∈ S', s ≥ a ∧ s ≤ n) → triangle_property S) → 
  n = 224 :=
begin
  sorry
end

end largest_nine_element_subset_l365_365056


namespace minimize_PA_plus_PB_l365_365025

-- Given conditions
def line_intersect_x_axis (l : ℝ → ℝ → ℝ) : ℝ × ℝ :=
  let x := 6 / 2 in (x, 0)

def line_intersect_y_axis (l : ℝ → ℝ → ℝ) : ℝ × ℝ :=
  let y := 6 / 3 in (0, y)

def reflection_across_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p in (-y, -x)

-- Problem statement
theorem minimize_PA_plus_PB {A B B' P : ℝ × ℝ} :
  A = (3, 0) →
  B = (0, 2) →
  B' = reflection_across_y_eq_neg_x B → 
  (P = (0, 0)) ↔ (P minimizes (λ P, abs (dist P A + dist P B))) :=
by
  intros hA hB hB'
  sorry

end minimize_PA_plus_PB_l365_365025


namespace exist_2x2_square_covered_by_two_dominoes_l365_365671

theorem exist_2x2_square_covered_by_two_dominoes :
  ∀ (grid : matrix (fin 8) (fin 8) bool) (cover : ∀ (i j : fin 8), bool),
  (∀ (i j : fin 8), cover i j ∨ cover i (j + 1) ∨ cover (i + 1) j) →
  ∃ (a b c d : fin 8), a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧
  (grid a b = tt ∧ grid c b = tt ∧ grid a d = tt ∧ grid c d = tt) :=
by
  sorry

end exist_2x2_square_covered_by_two_dominoes_l365_365671


namespace real_root_intervals_l365_365554

noncomputable def f (x : ℝ) := if x > -2 then exp (x + 1) - 2 else exp (-(x + 1)) - 2

theorem real_root_intervals :
  ∃ k : ℤ, ∀ x : ℝ, f x = 0 → (k - 1 : ℤ) < x ∧ x < (k : ℤ) →
    k = -3 ∨ k = 0 :=
by
  sorry

end real_root_intervals_l365_365554


namespace exists_number_with_digits_l365_365737

theorem exists_number_with_digits (a b : ℕ) (n : ℕ) (ha : odd a) (hb : even b) (n_pos : 0 < n) :
  ∃ N : ℕ, (2^n ∣ N) ∧ (∀ d ∈ (N.digits 10), d = a ∨ d = b) :=
sorry

end exists_number_with_digits_l365_365737


namespace area_AQC_is_9_l365_365296

-- Define the initial state of the problem
variables (A B C P Q R : Type) [add_comm_group A] [module ℝ A]
variables [add_comm_group B] [module ℝ B]
variables [add_comm_group C] [module ℝ C]
variables [add_comm_group P] [module ℝ P]
variables [add_comm_group Q] [module ℝ Q]
variables [add_comm_group R] [module ℝ R]

-- Define the segments AP and PB on AB
variable (AP PB : ℝ)
-- Area of triangle ABC
variable (area_ABC : ℝ := 15)

-- Define ratios
axiom AP_PB: AP = 3 ∧ PB = 2
axiom equal_areas: ∀ A C Q : A, ((area_Δ A Q C : ℝ)/5 * (PB + AP) =  area_Δ A Q C)

-- Define the area of triangle AQC
noncomputable def area_AQC : ℝ := (3 / 5) * area_ABC

-- Prove the area of triangle AQC
theorem area_AQC_is_9 : area_AQC = 9 :=
by {
  dsimp [area_AQC, area_ABC],
  norm_num,
}

end area_AQC_is_9_l365_365296


namespace arithmetic_sequence_nth_term_l365_365019

theorem arithmetic_sequence_nth_term (a₁ a₂ a₃ : ℤ) (x : ℤ) (n : ℕ)
  (h₁ : a₁ = 3 * x - 4)
  (h₂ : a₂ = 6 * x - 14)
  (h₃ : a₃ = 4 * x + 3)
  (h₄ : ∀ k : ℕ, a₁ + (k - 1) * ((a₂ - a₁) + (a₃ - a₂) / 2) = 3012) :
  n = 247 :=
by {
  -- Proof to be provided
  sorry
}

end arithmetic_sequence_nth_term_l365_365019


namespace ny_sales_tax_l365_365447

theorem ny_sales_tax {x : ℝ} 
  (h1 : 100 + x * 1 + 6/100 * (100 + x * 1) = 110) : 
  x = 3.77 :=
by
  sorry

end ny_sales_tax_l365_365447


namespace distance_calculation_l365_365846

def circle_center_distance : ℝ :=
  let circle_eq : (ℝ × ℝ) → Prop := λ p, (p.1)^2 + (p.2)^2 = 4 * p.1 + 2 * p.2 + 6
  let point := (10, 3)
  sorry

theorem distance_calculation :
  circle_center_distance = Real.sqrt 68 := sorry

end distance_calculation_l365_365846


namespace lattice_points_origin_center_lattice_points_shifted_center_approximation_of_pi_from_lattice_points_l365_365432

noncomputable def sphere_lattice_points_origin (r : ℕ) := 
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 ≤ r^2}

noncomputable def sphere_lattice_points_shifted (r : ℕ) := 
  {p : ℤ × ℤ × ℤ // (p.1 - 1/2)^2 + (p.2.1 - 1/2)^2 + p.2.2^2 ≤ r^2}

theorem lattice_points_origin_center (r : ℕ) (h : r = 10) :
  |sphere_lattice_points_origin r| = 4169 :=
sorry

theorem lattice_points_shifted_center (r : ℕ) (h : r = 10) :
  |sphere_lattice_points_shifted r| = 4196 :=
sorry

theorem approximation_of_pi_from_lattice_points:
  let pi_from_origin := (3 * (4169 : ℚ) / 4000)
  let pi_from_shifted := (3 * (4196 : ℚ) / 4000)
  pi_from_origin ≈ 3.12675 ∧ pi_from_shifted ≈ 3.147 :=
sorry

end lattice_points_origin_center_lattice_points_shifted_center_approximation_of_pi_from_lattice_points_l365_365432


namespace simplify_fraction_l365_365466

theorem simplify_fraction : (2 / (1 - (2 / 3))) = 6 :=
by
  sorry

end simplify_fraction_l365_365466


namespace number_of_superprimes_lt_15_l365_365900

/--
A prime number is called a "Superprime" if doubling it, and then subtracting 1,
results in another prime number.
-/
def is_superprime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (2 * p - 1)

/--
There are 3 Superprimes less than 15.
-/
theorem number_of_superprimes_lt_15 : Finset.card (Finset.filter is_superprime (Finset.filter Nat.Prime (Finset.range 15))) = 3 :=
by
  sorry

end number_of_superprimes_lt_15_l365_365900


namespace gcf_of_24_and_16_l365_365428

theorem gcf_of_24_and_16 :
  let n := 24
  let lcm := 48
  gcd n 16 = 8 :=
by
  sorry

end gcf_of_24_and_16_l365_365428


namespace slip_2_5_into_X_l365_365128

theorem slip_2_5_into_X 
  (slips : Set ℝ := {1.5, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 5})
  (cups : Set (Set ℝ) := {X, Y, Z, W : Set ℝ})
  (perfect_squares : Set ℝ := {4, 9, 16, 25})
  (hY : {3} ⊆ Y)
  (hW : {4} ⊆ W)
  (assignment : ∀ cup ∈ cups, ∃ s ∈ perfect_squares, ∑ slip in cup, slip = s) :
  {2.5} ⊆ X :=
begin
  sorry
end

end slip_2_5_into_X_l365_365128


namespace minimal_value_of_function_l365_365812

theorem minimal_value_of_function (x : ℝ) (hx : x > 1 / 2) :
  (x = 1 → (x^2 + 1) / x = 2) ∧
  (∀ y, (∀ z, z > 1 / 2 → y ≤ (z^2 + 1) / z) → y = 2) :=
by {
  sorry
}

end minimal_value_of_function_l365_365812


namespace koschei_coins_l365_365311

theorem koschei_coins :
  ∃ a : ℕ, a % 10 = 7 ∧ a % 12 = 9 ∧ 300 ≤ a ∧ a ≤ 400 ∧ a = 357 :=
by
  sorry

end koschei_coins_l365_365311


namespace find_f_x_l365_365992

def f : ℝ → ℝ := sorry

theorem find_f_x (x : ℝ) (h : f (x-1) = x^2 + 4x - 5) : f(x) = x^2 + 6x :=
sorry

end find_f_x_l365_365992


namespace min_value_at_neg7_l365_365443

noncomputable def f (x : ℝ) : ℝ := x^2 + 14 * x + 24

theorem min_value_at_neg7 : ∀ x : ℝ, f (-7) ≤ f x :=
by
  sorry

end min_value_at_neg7_l365_365443


namespace find_f1_l365_365192

def f : ℝ → ℝ
| x := if x > 2 then real.log (x + 1) / real.log 2 else f (x + 1)

theorem find_f1 : f 1 = 2 := by
  sorry

end find_f1_l365_365192


namespace double_sum_eq_l365_365544

theorem double_sum_eq : 
  (∑ n in (finset.Ico 2 (⊤ : ℕ)), ∑ k in (finset.Ico 1 n), k / (3 : ℝ)^(n + k)) = (9 / 128 : ℝ) :=
sorry

end double_sum_eq_l365_365544


namespace age_difference_is_51_l365_365753

def Milena_age : ℕ := 7
def Grandmother_age : ℕ := 9 * Milena_age
def Grandfather_age : ℕ := Grandmother_age + 2
def Cousin_age : ℕ := 2 * Milena_age
def Age_difference : ℕ := Grandfather_age - Cousin_age

theorem age_difference_is_51 : Age_difference = 51 := by
  sorry

end age_difference_is_51_l365_365753


namespace rationalize_and_min_val_l365_365764

noncomputable theory

-- Definitions based on the problem's conditions
def sqrt_50 := real.sqrt 50
def sqrt_25 := real.sqrt 25
def sqrt_5 := real.sqrt 5
def expr := sqrt_50 / (sqrt_25 - sqrt_5)
def rationalized := (5 * real.sqrt 2 + real.sqrt 10) / 4
def A := 5
def B := 2
def C := 1
def D := 4
def min_val := A + B + C + D

-- The main theorem statement
theorem rationalize_and_min_val :
  expr = rationalized ∧ min_val = 12 :=
by
  sorry

end rationalize_and_min_val_l365_365764


namespace sum_eq_9_div_64_l365_365550

noncomputable def double_sum : ℝ := ∑' (n : ℕ) in (set.Ici 2 : set ℕ), ∑' (k : ℕ) in set.Ico 1 n, (k : ℝ) / 3^(n + k)

theorem sum_eq_9_div_64 : double_sum = 9 / 64 := 
by 
sorry

end sum_eq_9_div_64_l365_365550


namespace surface_area_of_revolution_l365_365761

variable {a b : ℝ} (f : ℝ → ℝ)

theorem surface_area_of_revolution (h1 : ∀ x, x ∈ set.Icc a b → 0 < f x) 
  (h2 : continuous_on (deriv f) (set.Icc a b)) :
  2 * π * ∫ x in a..b, f x * sqrt (1 + (deriv f x)^2) = 2 * π * ∫ x in a..b, f x * sqrt (1 + (f' x)^2) x :=
sorry

end surface_area_of_revolution_l365_365761


namespace pyramid_soda_cases_l365_365339

theorem pyramid_soda_cases (n : ℕ) (h : n = 4) : 
  let level_cases (l : ℕ) := l * l in
  (level_cases 1 + level_cases 2 + level_cases 3 + level_cases 4) = 30 :=
by
  intros
  sorry

end pyramid_soda_cases_l365_365339


namespace least_froods_l365_365292

theorem least_froods (n : ℕ) :
  (∃ n, n ≥ 1 ∧ (n * (n + 1)) / 2 > 20 * n) → (∃ n, n = 40) :=
by {
  sorry
}

end least_froods_l365_365292


namespace complex_quadrant_l365_365619

open Complex

theorem complex_quadrant
  (z1 z2 z : ℂ) (h1 : z1 = 2 + I) (h2 : z2 = 1 - I) (h3 : z = z1 / z2) :
  0 < z.re ∧ 0 < z.im :=
by
  -- sorry to skip the proof steps
  sorry

end complex_quadrant_l365_365619


namespace nested_sum_equals_fraction_l365_365528

noncomputable def nested_sum : ℝ := ∑' (n : ℕ) in Ico 2 (⊤), ∑' (k : ℕ) in Ico 1 n, (k : ℝ) / 3^(n + k)

theorem nested_sum_equals_fraction :
  nested_sum = 3 / 64 := sorry

end nested_sum_equals_fraction_l365_365528


namespace correct_conclusions_l365_365634

def pos_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

def sum_of_n_terms (S a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) * S (n+1) = 9

def second_term_less_than_3 (a S : ℕ → ℝ) : Prop :=
  a 1 < 3

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

def exists_term_less_than_1_over_100 (a : ℕ → ℝ) : Prop :=
  ∃ n : ℕ, a n < 1/100

theorem correct_conclusions (a S : ℕ → ℝ) :
  pos_sequence a → sum_of_n_terms S a →
  second_term_less_than_3 a S ∧ (¬(∀ q : ℝ, ∃ r : ℝ, ∀ n : ℕ, a n = r * q ^ n)) ∧ is_decreasing_sequence a ∧ exists_term_less_than_1_over_100 a :=
sorry

end correct_conclusions_l365_365634


namespace polynomial_f_is_x_l365_365971

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := sorry

axiom functional_eqn (a : ℝ) : f(a + 1) = f(a) + f(1)
axiom cyclic_values (k : ℝ) (n : ℕ) (hk : k ≠ 0) (k_vals : Fin (n + 1) → ℝ) :
  (f k_vals[0] = k_vals[1] ∧ f k_vals[n] = k_vals[0])

-- The statement to prove
theorem polynomial_f_is_x : f x = x := sorry

end polynomial_f_is_x_l365_365971


namespace constant_term_expansion_l365_365196

-- Define a as the integral
def a : ℝ := ∫ (x : ℝ) in 0..(Real.pi / 6), Real.cos x

-- Define the main theorem stating the constant term in the expansion
theorem constant_term_expansion : (x : ℝ) →
  x * (x - (1 / a) * x) ^ 7 = -128 :=
by
  sorry

end constant_term_expansion_l365_365196


namespace double_sum_evaluation_l365_365538

theorem double_sum_evaluation : 
  (∑ n from 2 to ∞, ∑ k from 1 to (n - 1), k / (3 ^ (n + k))) = (6 / 25) :=
by sorry

end double_sum_evaluation_l365_365538


namespace part1_monotonic_increasing_interval_part2_min_diff_l365_365230

def f (x : ℝ) : ℝ := sqrt 2 * sin x * cos x - sqrt 2 * cos x * cos x + sqrt 2 / 2

def g (x : ℝ) : ℝ := f(x) + f(x + π/4) - f(x) * f(x + π/4)

theorem part1_monotonic_increasing_interval :
  ∃ (k : ℤ), ∀ (x : ℝ), 
    ( -π / 8 + k * π ≤ x ∧ x ≤ 3 * π / 8 + k * π) ∧
    monotonic_increasing (λ y, f y) :=
sorry

theorem part2_min_diff :
  ∃ (x1 x2 : ℝ), 
    (∀ (x : ℝ), g x1 ≤ g x ∧ g x ≤ g x2) →
    abs (x1 - x2) = 3 * π / 8 :=
sorry

end part1_monotonic_increasing_interval_part2_min_diff_l365_365230


namespace number_of_superprimes_lt_15_l365_365899

/--
A prime number is called a "Superprime" if doubling it, and then subtracting 1,
results in another prime number.
-/
def is_superprime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (2 * p - 1)

/--
There are 3 Superprimes less than 15.
-/
theorem number_of_superprimes_lt_15 : Finset.card (Finset.filter is_superprime (Finset.filter Nat.Prime (Finset.range 15))) = 3 :=
by
  sorry

end number_of_superprimes_lt_15_l365_365899


namespace prob_of_winning_both_prob_of_winning_exactly_once_prob_of_winning_at_least_once_l365_365905

/-
Define the events A and B and their probabilities under the condition of independence.
-/

variables {Ω : Type} [Nonempty Ω] [ProbabilitySpace Ω]

constant A B : Event Ω
constant pA pB : ℝ
constant independent_events : IndependentEvents A B

axiom probA : ℙ[A] = 0.05
axiom probB : ℙ[B] = 0.05

theorem prob_of_winning_both : ℙ[A ∩ B] = 0.0025 :=
by sorry

theorem prob_of_winning_exactly_once : ℙ[(A ∩ Bᶜ) ∪ (Aᶜ ∩ B)] = 0.095 :=
by sorry

theorem prob_of_winning_at_least_once : ℙ[A ∪ B] = 0.0975 :=
by sorry

end prob_of_winning_both_prob_of_winning_exactly_once_prob_of_winning_at_least_once_l365_365905


namespace general_term_l365_365368

noncomputable def sequence (n : ℕ) : ℚ :=
  if h : n = 0 then 1 else
    sqrt ((sequence (n-1))^2 + sequence (n-1)) + sequence (n-1)

theorem general_term (n : ℕ) :
  sequence (n+1) = 1 / (real.sqrt (2 ^ ((n:ℚ) / (nat.pow 2 n - 1))) - 1) :=
sorry

end general_term_l365_365368


namespace almonds_walnuts_ratio_l365_365474

-- Define the given weights and parts
def w_a : ℝ := 107.14285714285714
def w_m : ℝ := 150
def p_a : ℝ := 5

-- Now we will formulate the statement to prove the ratio of almonds to walnuts
theorem almonds_walnuts_ratio : 
  ∃ (p_w : ℝ), p_a / p_w = 5 / 2 :=
by
  -- It is given that p_a / p_w = 5 / 2, we need to find p_w
  sorry

end almonds_walnuts_ratio_l365_365474


namespace monica_read_books_l365_365754

theorem monica_read_books (x : ℕ) 
    (h1 : 2 * (2 * x) + 5 = 69) : 
    x = 16 :=
by 
  sorry

end monica_read_books_l365_365754


namespace juice_spilled_l365_365101

def initial_amount := 1.0
def Youngin_drank := 0.1
def Narin_drank := Youngin_drank + 0.2
def remaining_amount := 0.3

theorem juice_spilled :
  initial_amount - (Youngin_drank + Narin_drank) - remaining_amount = 0.3 :=
by
  sorry

end juice_spilled_l365_365101


namespace parallelogram_as_analogy_l365_365446

def spatial_rectangular_parallelepiped := 
  { faces : List (List ℝ) // faces.length = 6 ∧ ∀ face ∈ faces, (∀ side ∈ face, side == face.head)}

theorem parallelogram_as_analogy :
  ∀ (p : spatial_rectangular_parallelepiped), 
    ∃ (analog : Type), 
      analog = parallelogram :=
by sorry

end parallelogram_as_analogy_l365_365446


namespace sequence_arithmetic_l365_365710

theorem sequence_arithmetic (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h : ∀ n, S n = 2 * n^2 - 3 * n)
  (h₀ : S 0 = 0) 
  (h₁ : ∀ n, S (n+1) = S n + a (n+1)) :
  ∀ n, a n = 4 * n - 1 := sorry

end sequence_arithmetic_l365_365710


namespace total_tables_made_l365_365103

theorem total_tables_made (x : ℕ → ℕ) (sum_x : ℕ) (styles : ℕ) (h_sum : sum_x = 100)
  (h_styles : styles = 10) (hx_per_style : ∀ i, x i this_month + (x i last_month) = (2 * (x i)) - 3) :
  sum (λ i, (x i this_month) + (x i last_month)) (range styles) = 170 :=
sorry

end total_tables_made_l365_365103


namespace find_a_l365_365245

def integral_value (a : ℝ) : ℝ := ∫ x in 1..a, (2 * x + 1 / x) 

theorem find_a (a : ℝ) (h1 : integral_value a = 3 + real.log 2) (h2 : a > 1) : a = 2 :=
by
  -- proof will go here
  sorry

end find_a_l365_365245


namespace evaluate_ceiling_expression_l365_365162

theorem evaluate_ceiling_expression:
  (Int.ceil ((23 : ℚ) / 9 - Int.ceil ((35 : ℚ) / 23)))
  / (Int.ceil ((35 : ℚ) / 9 + Int.ceil ((9 * 23 : ℚ) / 35))) = 1 / 12 := by
  sorry

end evaluate_ceiling_expression_l365_365162


namespace train_crosses_platform_in_39_sec_l365_365098

noncomputable def train_speed (train_length : ℕ) (time_signal_pole : ℕ) : ℚ :=
  train_length / time_signal_pole

noncomputable def time_to_cross_platform (train_length platform_length : ℕ) (train_speed : ℚ) : ℚ :=
  (train_length + platform_length) / train_speed

theorem train_crosses_platform_in_39_sec :
  ∀ (train_length platform_length time_signal_pole : ℕ),
  train_length = 600 → platform_length = 700 → time_signal_pole = 18 →
  time_to_cross_platform train_length platform_length (train_speed train_length time_signal_pole) = 39 := 
by
  intros train_length platform_length time_signal_pole h1 h2 h3
  rw [h1, h2, h3]
  simp [train_speed, time_to_cross_platform]
  norm_num
  sorry

end train_crosses_platform_in_39_sec_l365_365098


namespace longest_segment_in_cylinder_l365_365107

noncomputable def length_of_longest_segment (r h : ℝ) : ℝ :=
  real.sqrt (h^2 + (2 * r)^2)

theorem longest_segment_in_cylinder :
  length_of_longest_segment 5 10 = 10 * real.sqrt 2 :=
by
  sorry

end longest_segment_in_cylinder_l365_365107


namespace min_occupied_seats_l365_365136

theorem min_occupied_seats (n : ℕ) (h_n : n = 150) : 
  ∃ k : ℕ, k = 37 ∧ ∀ (occupied : Finset ℕ), 
    occupied.card < k → ∃ i : ℕ, i ∉ occupied ∧ ∀ j : ℕ, j ∈ occupied → j + 1 ≠ i ∧ j - 1 ≠ i :=
by
  sorry

end min_occupied_seats_l365_365136


namespace three_digit_sum_of_factorials_eq_l365_365800

-- Define the property of a three-digit number
def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000

-- Define the function to calculate the sum of factorials of the digits of a number
def sum_of_factorials_of_digits (n : ℕ) : ℕ :=
  let x := n / 100 in
  let y := (n % 100) / 10 in
  let z := n % 10 in
  Nat.factorial x + Nat.factorial y + Nat.factorial z

-- State the theorem to prove
theorem three_digit_sum_of_factorials_eq (n : ℕ) (h : is_three_digit n) :
  sum_of_factorials_of_digits n = n ↔ n = 145 :=
sorry

end three_digit_sum_of_factorials_eq_l365_365800


namespace part_1_part_2_l365_365200

variable {a x x1 x2 : ℝ}

-- Assume the function f(x) and its conditions
def f (x : ℝ) : ℝ := a / x + Real.log x - 3
axiom hx1 : f x1 = 0
axiom hx2 : f x2 = 0
axiom h_order : x1 < x2

-- Prove 0 < a < e^2
theorem part_1 : 0 < a ∧ a < Real.exp 2 :=
sorry

-- Prove x1 + x2 > 2a
theorem part_2 : x1 + x2 > 2 * a :=
sorry

end part_1_part_2_l365_365200


namespace father_gave_8_candies_to_Billy_l365_365523

theorem father_gave_8_candies_to_Billy (candies_Billy : ℕ) (candies_Caleb : ℕ) (candies_Andy : ℕ) (candies_father : ℕ) 
  (candies_given_to_Caleb : ℕ) (candies_more_than_Caleb : ℕ) (candies_given_by_father_total : ℕ) :
  (candies_given_to_Caleb = 11) →
  (candies_Caleb = 11) →
  (candies_Andy = 9) →
  (candies_father = 36) →
  (candies_Andy = candies_Caleb + 4) →
  (candies_given_by_father_total = candies_given_to_Caleb + (candies_Andy - 9)) →
  (candies_father - candies_given_by_father_total = 8) →
  candies_Billy = 8 := 
by
  intros
  sorry

end father_gave_8_candies_to_Billy_l365_365523


namespace expression_value_l365_365140

theorem expression_value :
  (1 / (3 - (1 / (3 + (1 / (3 - (1 / 3))))))) = (27 / 73) :=
by 
  sorry

end expression_value_l365_365140


namespace rachel_drinks_amount_l365_365564

theorem rachel_drinks_amount (don_milk : ℚ) (cat_milk_ratio : ℚ) (rachel_drink_ratio : ℚ) : 
  don_milk = 3 / 4 →
  cat_milk_ratio = 1 / 8 →
  rachel_drink_ratio = 1 / 2 →
  let remaining_milk := don_milk - (cat_milk_ratio * don_milk) in
  let rachel_drinks := rachel_drink_ratio * remaining_milk in
  rachel_drinks = 21 / 64 :=
by
  intros _ _ _
  simp
  sorry

end rachel_drinks_amount_l365_365564


namespace not_necessarily_same_digit_sum_l365_365758

def sum_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem not_necessarily_same_digit_sum (M N : ℕ) (h : sum_digits (N + M) = sum_digits N) :
  ¬ ∀ k : ℕ, sum_digits (N + k * M) = sum_digits N :=
by
  sorry

end not_necessarily_same_digit_sum_l365_365758


namespace ratio_of_ladies_l365_365475

-- Define the number of ladies in each group
variables (L1 L2 : ℕ)

-- Define the first condition
def first_group_work_done (L1 : ℕ) : ℕ := L1 * 12

-- Define the second condition
def second_group_work_done (L2 : ℕ) : ℕ := L2 * 3

-- Given conditions
def condition1 : Prop := second_group_work_done L2 = (1 / 2 : ℝ) * first_group_work_done L1

-- Prove the ratio of L2 to L1 is 2
theorem ratio_of_ladies (L1 L2 : ℕ) (h : condition1) : L2 / L1 = 2 :=
by 
  sorry

end ratio_of_ladies_l365_365475


namespace B_pow4_vec_l365_365726

variable (B : Matrix (Fin 2) (Fin 2) ℝ)
variable (v : Vector (Fin 2) ℝ := ![4, -1])

theorem B_pow4_vec :
  B.mulVec v = ![12, -3] →
  (B ^ 4).mulVec v = ![324, -81] :=
by sorry

end B_pow4_vec_l365_365726


namespace arith_seq_sum_of_first_8_terms_geom_seq_sum_l365_365467

noncomputable def arithmetic_sequence (a1 a3 a5 a4 : ℤ) : ℤ := 
  let d := a4 - a3 in
  let a1_prime := a3 - 2 * d in
  8 * a1_prime + (8 * d)

theorem arith_seq_sum_of_first_8_terms 
  (a1 a3 a5 a4 : ℤ)
  (h1 : a1 + a3 + a5 = 21)
  (h2 : a4 = 9)
  (a3_eq : 3 * a3 = 21) :
  arithmetic_sequence a1 a3 a5 a4 = 80 :=
  sorry

noncomputable def geometric_sequence_sum (a1 q an : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geom_seq_sum
  (a1 q an : ℝ)
  (h1 : a1 = -2.7)
  (h2 : q = -1/3)
  (h3 : an = 1/90)
  (n_eq : n = 6) :
  geometric_sequence_sum a1 q an n = -(91/45) :=
  sorry

end arith_seq_sum_of_first_8_terms_geom_seq_sum_l365_365467


namespace graph_shift_equiv_l365_365835

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)
def g (x : ℝ) : ℝ := Real.cos (2 * x)

theorem graph_shift_equiv : f (x - Real.pi / 12) = g x := sorry

end graph_shift_equiv_l365_365835


namespace fraction_of_widgets_second_shift_l365_365925

theorem fraction_of_widgets_second_shift (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let first_shift_widgets := x * y
  let second_shift_widgets := (2 / 3) * x * (4 / 3) * y
  let total_widgets := first_shift_widgets + second_shift_widgets
  let fraction_second_shift := second_shift_widgets / total_widgets
  fraction_second_shift = 8 / 17 :=
by
  sorry

end fraction_of_widgets_second_shift_l365_365925


namespace nested_sum_equals_fraction_l365_365526

noncomputable def nested_sum : ℝ := ∑' (n : ℕ) in Ico 2 (⊤), ∑' (k : ℕ) in Ico 1 n, (k : ℝ) / 3^(n + k)

theorem nested_sum_equals_fraction :
  nested_sum = 3 / 64 := sorry

end nested_sum_equals_fraction_l365_365526


namespace hyperbola_asymptotes_l365_365016

theorem hyperbola_asymptotes:
  (∀ x y : Real, (x^2 / 16 - y^2 / 9 = 1) → (y = 3 / 4 * x ∨ y = -3 / 4 * x)) :=
by {
  sorry
}

end hyperbola_asymptotes_l365_365016


namespace continuous_value_at_neg_one_l365_365834

theorem continuous_value_at_neg_one : 
  filter.tendsto (λ x : ℝ, (x^2 - x + 1) / (x - 1)) (nhds (-1)) (nhds (-3 / 2)) :=
by
sorry

end continuous_value_at_neg_one_l365_365834


namespace solve_nine_sections_bamboo_problem_l365_365785

-- Define the bamboo stick problem
noncomputable def nine_sections_bamboo_problem : Prop :=
∃ (a : ℕ → ℝ) (d : ℝ),
  (∀ n, a (n + 1) = a n + d) ∧ -- Arithmetic sequence
  (a 1 + a 2 + a 3 + a 4 = 3) ∧ -- Top 4 sections' total volume
  (a 7 + a 8 + a 9 = 4) ∧ -- Bottom 3 sections' total volume
  (a 5 = 67 / 66) -- Volume of the 5th section

theorem solve_nine_sections_bamboo_problem : nine_sections_bamboo_problem :=
sorry

end solve_nine_sections_bamboo_problem_l365_365785


namespace solve_log_equation_l365_365375

theorem solve_log_equation (x : ℝ) (h : log 2 x + log 8 x = 5) : x = 2^3.75 :=
sorry

end solve_log_equation_l365_365375


namespace ed_more_marbles_l365_365567

def initial_difference : ℕ := 22 -- Ed had 22 more marbles than Doug initially
def lost_marbles : ℕ := 8       -- Doug lost 8 marbles
def found_marbles : ℕ := 5      -- Susan found 5 of the lost marbles

-- Let D be the initial number of marbles Doug had
variables (D : ℕ)

theorem ed_more_marbles :
  let ed_initial := D + initial_difference in
  let ed_after := ed_initial + found_marbles in
  let doug_after := D - lost_marbles in
  ed_after - doug_after = 35 :=
by {
  sorry,
}

end ed_more_marbles_l365_365567


namespace third_side_of_triangle_l365_365356

theorem third_side_of_triangle
  (A B C : Type)
  (a b c : ℝ)
  (ha : a = 6)
  (hb : b = 18)
  (angle_B angle_C : ℝ)
  (hangle : angle_B = 3 * angle_C)
  (cos_C : ℝ)
  (hcos_C : cos_C = Real.sqrt(2 / 3))
  (sin_C : ℝ)
  (hsin_C : sin_C = Real.sqrt(3) / 3)
  (hcos_eq : cos_C = (a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b)) :
  c = 33 * Real.sqrt(6) :=
sorry

end third_side_of_triangle_l365_365356


namespace greatest_two_digit_multiple_of_11_l365_365848

-- Definitions based on conditions in a)
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_11 (n : ℕ) : Prop := n % 11 = 0

-- The main statement to be proven
theorem greatest_two_digit_multiple_of_11 : ∃ n, is_two_digit n ∧ is_multiple_of_11 n ∧ ∀ m, is_two_digit m ∧ is_multiple_of_11 m → m ≤ n :=
by
  have h : is_two_digit 99 ∧ is_multiple_of_11 99,
  { exact ⟨⟨le_refl 99, le_of_lt (by norm_num)⟩, by norm_num⟩ },
  use 99,
  simp,
  intros m h1 h2,
  have : m ≤ 99 := by linarith,
  exact this

end greatest_two_digit_multiple_of_11_l365_365848


namespace prob_fourth_term_integer_l365_365917

def sequence (a : ℕ) (step : ℕ → ℕ) (tail : ℕ → ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0     => a
  | n + 1 => if n % 2 = 0 then step (sequence a step tail n) else tail (sequence a step tail n)

def heads (a : ℕ) : ℕ := 3 * a - 2
def tails (a : ℕ) : ℕ := a / 3 - 2

theorem prob_fourth_term_integer :
  let a := 7 in
  let a4 := sequence a heads tails 3 in
  (a4 = 163 ∨ a4 = 16) → (a4 = 163) = true :=
by sorry

end prob_fourth_term_integer_l365_365917


namespace area_union_square_circle_l365_365496

theorem area_union_square_circle :
  let side_length_square := 8
  let radius_circle := 12
  let area_square := side_length_square ^ 2
  let area_circle := Real.pi * (radius_circle ^ 2)
  let overlap_area := (1 / 4) * area_circle
  let union_area := area_square + area_circle - overlap_area
  union_area = 64 + 108 * Real.pi := by
begin
  sorry
end

end area_union_square_circle_l365_365496


namespace change_received_l365_365479

noncomputable def cost_of_tickets := sorry

theorem change_received : 
  let regular_ticket := 129
  let discount_below_12 := 15
  let discount_12_to_15 := 7
  let additional_family_discount := 10
  let amount_given := 900

  let ages := [6, 10, 13, 8]
  let under_12 := ages.filter (λ age => age < 12)
  let over_12 := ages.filter (λ age => 12 ≤ age ∧ age <= 15)
  
  let cost_under_12 := under_12.map (λ _ => regular_ticket - discount_below_12)
  let cost_over_12 := over_12.map (λ _ => regular_ticket - discount_12_to_15)
  
  let total_cost := cost_under_12.sum + cost_over_12.sum
  let total_cost_with_additional_discount := if under_12.length > 2 then total_cost - additional_family_discount else total_cost

  let parents_cost := 2 * regular_ticket
  
  let final_cost := total_cost_with_additional_discount + parents_cost
  let change := amount_given - final_cost

  change = 188 :=
by 
  intros
  sorry

end change_received_l365_365479


namespace shaded_area_proof_l365_365014

noncomputable def problem_statement : Prop :=
  ∃ (P Q R S T U O : ℝ × ℝ),
  distance P O = 2 ∧ 
  distance Q O = 2 ∧ 
  distance R O = 2 ∧ 
  distance S O = 2 ∧ 
  distance T O = 2 ∧ 
  distance U O = 2 ∧ 
  distance O ((0,0) : ℝ × ℝ) = 0 ∧
  distance (0,0) ((0,1) : ℝ × ℝ) = 1 ∧
  (∀ x : ℝ, distance ((0,1) : ℝ × ℝ) ((cos (x), sin (x)) : ℝ × ℝ) = 1) ∧
  -- Calculating area of sector and quadrilateral
  ∀ x y z w v : ℝ, 
  (distance (cos (x), sin (x)) (0,0) = 2) ∧ 
  (distance (cos (y), sin (y)) (0,0) = 2) ∧ 
  angle_vectors (cos (x), sin (x)) (cos (y), sin (y)) = 120 ∧
  angle_vectors (cos (z), sin (z)) (cos (w), sin (w)) = 60 ∧ 
  is_triangle_area := | cos (x) - cos (y) | * | sin (x) - sin (y) | * sin (v) / 2,
  -- total area of shaded region is 2 * pi + 3
  (3 * (total_inner_sector + total_area_quadri)) = ((3 * 2 * π / 3) + 3),
  2 * π + 3

theorem shaded_area_proof : problem_statement :=
  sorry

end shaded_area_proof_l365_365014


namespace inscribed_circumscribed_quadrilateral_l365_365762

theorem inscribed_circumscribed_quadrilateral
  (R r d : ℝ)
  (ABCD : set (Euclidean_Space ℝ 2))
  (h1 : ∃ c₁ : Euclidean_Space ℝ 2, ∀ P ∈ ABCD, dist P c₁ = R)
  (h2 : ∃ c₂ : Euclidean_Space ℝ 2, ∀ P ∈ ABCD, dist P c₂ = r)
  (h3 : ∃ c₁ c₂ : Euclidean_Space ℝ 2, dist c₁ c₂ = d)
  : 
  (1 / (R + d)^2) + (1 / (R - d)^2) = 1 / r^2 ∧
  ∃ (pts : set (Euclidean_Space ℝ 2)), ∀ pt ∈ pts, 
    ∃ ABCD' ⊆ pts, ∃ (c₁ c₂ : Euclidean_Space ℝ 2), 
      (∀ P ∈ ABCD', dist P c₁ = R) ∧ (∀ P ∈ ABCD', dist P c₂ = r) ∧
      dist c₁ c₂ = d :=
sorry

end inscribed_circumscribed_quadrilateral_l365_365762


namespace solution_set_of_inequality_l365_365216

variable {f : ℝ → ℝ}
variable (f' : ℝ → ℝ)
variable (domain : ∀ x, x ∈ set.Ioo (-5 : ℝ) 5 → differentiable_at ℝ f x)
variable (ineq : ∀ x, x ∈ set.Ioo (-5 : ℝ) 5 → f x + x * f' x > 2)

theorem solution_set_of_inequality : {x : ℝ | (2 < x) ∧ (x < 4)} = {x : ℝ | (2x-3) * f (2x-3) - (x-1) * f (x-1) > 2x-4} :=
by {
  sorry
}

end solution_set_of_inequality_l365_365216


namespace average_nat_series_l365_365844

theorem average_nat_series : 
  let a := 12  -- first term
  let l := 53  -- last term
  let n := (l - a) / 1 + 1  -- number of terms
  let sum := n / 2 * (a + l)  -- sum of the arithmetic series
  let average := sum / n  -- average of the series
  average = 32.5 :=
by
  let a := 12
  let l := 53
  let n := (l - a) / 1 + 1
  let sum := n / 2 * (a + l)
  let average := sum / n
  sorry

end average_nat_series_l365_365844


namespace third_side_of_triangle_l365_365355

theorem third_side_of_triangle
  (A B C : Type)
  (a b c : ℝ)
  (ha : a = 6)
  (hb : b = 18)
  (angle_B angle_C : ℝ)
  (hangle : angle_B = 3 * angle_C)
  (cos_C : ℝ)
  (hcos_C : cos_C = Real.sqrt(2 / 3))
  (sin_C : ℝ)
  (hsin_C : sin_C = Real.sqrt(3) / 3)
  (hcos_eq : cos_C = (a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b)) :
  c = 33 * Real.sqrt(6) :=
sorry

end third_side_of_triangle_l365_365355


namespace sequence_elements_are_integers_l365_365821

theorem sequence_elements_are_integers 
  (seq : ℕ → ℝ) 
  (h : ∀ (n : ℕ), 1 ≤ n → n ≤ 2000 → (∑ i in range n, seq i) ^ 3 = (∑ i in range n, (seq i) ^ 3)) :
  ∀ (n : ℕ), 1 ≤ n → n ≤ 2000 → ∀ (i : ℕ), i < n → ∃ (k : ℤ), seq i = k :=
by
  sorry

end sequence_elements_are_integers_l365_365821


namespace add_in_base6_l365_365912

def add_base6 (a b : ℕ) : ℕ := (a + b) % 6 + (((a + b) / 6) * 10)

theorem add_in_base6 (x y : ℕ) (h1 : x = 5) (h2 : y = 23) : add_base6 x y = 32 :=
by
  rw [h1, h2]
  -- Explanation: here add_base6 interprets numbers as base 6 and then performs addition,
  -- taking care of the base conversion automatically. This avoids directly involving steps of the given solution.
  sorry

end add_in_base6_l365_365912


namespace interest_rate_other_fund_l365_365124

theorem interest_rate_other_fund
    (total_investment : ℕ)
    (interest_first_fund : ℝ)
    (total_annual_interest : ℝ)
    (invested_first_fund : ℕ)
    (invested_second_fund : ℕ)
    (interest_first_fund_rate : ℝ)
    (annual_interest_first_fund : ℝ)
    (annual_interest_second_fund : ℝ)
    (interest_rate_other_fund : ℝ)
    (principle_amount_other_fund : ℝ) :
    total_investment = 50000 →
    interest_first_fund = 0.08 →
    total_annual_interest = 4120 →
    invested_first_fund = 26000 →
    invested_second_fund = total_investment - invested_first_fund →
    annual_interest_first_fund = invested_first_fund * interest_first_fund →
    annual_interest_second_fund = total_annual_interest - annual_interest_first_fund →
    annual_interest_second_fund = principle_amount_other_fund * interest_rate_other_fund →
    principle_amount_other_fund = (total_investment - invested_first_fund) →
    interest_rate_other_fund = 0.085 :=
begin
    intros,
    sorry -- Proof not required.
end

end interest_rate_other_fund_l365_365124


namespace a_1000_value_l365_365683

open Nat

theorem a_1000_value :
  ∃ (a : ℕ → ℤ), (a 1 = 1010) ∧ (a 2 = 1011) ∧ 
  (∀ n ≥ 1, a n + a (n+1) + a (n+2) = 2 * n) ∧ 
  (a 1000 = 1676) :=
sorry

end a_1000_value_l365_365683


namespace speed_increase_impossible_l365_365520

theorem speed_increase_impossible (v : ℝ) : v = 60 → (¬ ∃ v', (1 / (v' / 60) = 0)) :=
by sorry

end speed_increase_impossible_l365_365520


namespace part1_part2_l365_365235

-- Part (1)
theorem part1 (m : ℝ) (h : ∀ x ∈ set.Icc (-1 : ℝ) 2, x^2 - 2*x - m ≤ 0) : 3 ≤ m :=
sorry

-- Part (2)
theorem part2 (a : ℝ)
  (h : ∀ x ∈ set.Icc (2 * a : ℝ) (a + 1), x ∈ set.Icc (3 : ℝ) (⊤ : ℝ)) :
  2 ≤ a :=
sorry

end part1_part2_l365_365235


namespace bertha_initial_balls_l365_365515

-- Definitions of the conditions
def wears_out_balls (total_games : ℕ) : ℕ := total_games / 10
def lost_balls (total_games : ℕ) : ℕ := total_games / 5
def bought_balls (total_games : ℕ) : ℕ := (total_games / 4) * 3
def total_change (total_games : ℕ) : ℕ :=
  bought_balls(total_games) - wears_out_balls(total_games) - lost_balls(total_games)
def initial_balls_after_20_games (final_balls : ℕ, total_games : ℕ) : ℕ :=
  (final_balls : int) - (total_change(total_games) : int)

-- Proof statement
theorem bertha_initial_balls (final_balls : ℕ) (total_games : ℕ) (partner_balls_given : ℕ) : 
  final_balls = 10 → total_games = 20 → partner_balls_given = 1 → 
  initial_balls_after_20_games final_balls total_games + partner_balls_given = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  unfold initial_balls_after_20_games
  unfold total_change
  unfold bought_balls
  unfold wears_out_balls
  unfold lost_balls
  sorry

end bertha_initial_balls_l365_365515


namespace nested_sum_equals_fraction_l365_365530

noncomputable def nested_sum : ℝ := ∑' (n : ℕ) in Ico 2 (⊤), ∑' (k : ℕ) in Ico 1 n, (k : ℝ) / 3^(n + k)

theorem nested_sum_equals_fraction :
  nested_sum = 3 / 64 := sorry

end nested_sum_equals_fraction_l365_365530


namespace sin_alpha_value_l365_365248

-- Define the variables and conditions
variables (α : ℝ) 
def in_third_quadrant (α : ℝ) : Prop := π < α ∧ α < 3 * π / 2
def tan_alpha := 3

-- State the theorem
theorem sin_alpha_value (h1 : in_third_quadrant α) (h2 : Real.tan α = tan_alpha) :
  Real.sin α = - (3 * Real.sqrt 10 / 10) := 
sorry

end sin_alpha_value_l365_365248


namespace double_sum_evaluation_l365_365537

theorem double_sum_evaluation : 
  (∑ n from 2 to ∞, ∑ k from 1 to (n - 1), k / (3 ^ (n + k))) = (6 / 25) :=
by sorry

end double_sum_evaluation_l365_365537


namespace solve_system_l365_365778

theorem solve_system :
  ∃ (x y z u v : ℤ), 
    (x - y + z = 1) ∧ 
    (y - z + u = 2) ∧ 
    (z - u + v = 3) ∧ 
    (u - v + x = 4) ∧ 
    (v - x + y = 5) ∧ 
    (x, y, z, u, v) = (0, 6, 7, 3, -1) :=
begin
  sorry
end

end solve_system_l365_365778


namespace range_of_t_l365_365731

noncomputable def condition (t : ℝ) : Prop :=
  ∃ x, 1 < x ∧ x < 5 / 2 ∧ (t * x^2 + 2 * x - 2 > 0)

theorem range_of_t (t : ℝ) : ¬¬ condition t → t > - 1 / 2 :=
by
  intros h
  -- The actual proof should be here
  sorry

end range_of_t_l365_365731


namespace probability_fujian_l365_365045

def companies : List (String × String) :=
  [("A", "Liaoning"), ("B", "Fujian"), ("C", "Fujian"), 
   ("D", "Henan"), ("E", "Henan"), ("F", "Henan")]

def pairs (l : List α) : List (α × α) :=
  List.bind l (λ x, List.map (λ y, (x, y)) l).filter (λ p, p.fst ≠ p.snd)

def is_from_fujian (c : (String × String)) : Bool :=
  c.snd = "Fujian"

def count_pairs_with_fujian (ps : List ((String × String) × (String × String))) : Nat :=
  ps.filter (λ p, is_from_fujian p.fst || is_from_fujian p.snd).length

theorem probability_fujian :
  let ps := pairs companies
  count_pairs_with_fujian ps / ps.length = (3 : ℝ) / (5 : ℝ) :=
by
  sorry

end probability_fujian_l365_365045


namespace tonya_hamburgers_to_beat_winner_l365_365421

-- Given conditions
def ounces_per_hamburger : ℕ := 4
def ounces_eaten_last_year : ℕ := 84

-- Calculate the number of hamburgers eaten last year
def hamburgers_eaten_last_year : ℕ := ounces_eaten_last_year / ounces_per_hamburger

-- Prove the number of hamburgers Tonya needs to eat to beat last year's winner
theorem tonya_hamburgers_to_beat_winner : 
  hamburgers_eaten_last_year + 1 = 22 :=
by
  -- It remains to be proven
  sorry

end tonya_hamburgers_to_beat_winner_l365_365421


namespace sine_of_angle_from_point_l365_365221

theorem sine_of_angle_from_point (x y : ℤ) (r : ℝ) (h : r = Real.sqrt ((x : ℝ)^2 + (y : ℝ)^2)) (hx : x = -12) (hy : y = 5) :
  Real.sin (Real.arctan (y / x)) = y / r := 
by
  sorry

end sine_of_angle_from_point_l365_365221


namespace calculate_principal_sum_l365_365084

def simple_interest (P R T : ℚ) : ℚ := (P * R * T) / 100

theorem calculate_principal_sum :
  ∀ (SI P R T : ℚ), SI = 1000 ∧ R = 10 ∧ T = 4 ∧ SI = simple_interest P R T → P = 2500 := 
by
  intros SI P R T h
  cases h with h1 h2,
  cases h2 with hR h3,
  cases h3 with hT hSI_eq,
  rw [simple_interest, hR, hT, h1] at hSI_eq,
  sorry

end calculate_principal_sum_l365_365084


namespace legs_in_room_l365_365828

def total_legs_in_room (tables4 : Nat) (sofa : Nat) (chairs4 : Nat) (tables3 : Nat) (table1 : Nat) (rocking_chair2 : Nat) : Nat :=
  (tables4 * 4) + (sofa * 4) + (chairs4 * 4) + (tables3 * 3) + (table1 * 1) + (rocking_chair2 * 2)

theorem legs_in_room :
  total_legs_in_room 4 1 2 3 1 1 = 40 :=
by
  -- Skipping proof steps
  sorry

end legs_in_room_l365_365828


namespace stratified_sample_count_l365_365884

theorem stratified_sample_count (total_population elderly_population middle_aged_population young_population sample_size : ℕ)
  (h_total_population : total_population = 162)
  (h_elderly_population : elderly_population = 27)
  (h_middle_aged_population : middle_aged_population = 54)
  (h_young_population : young_population = 81)
  (h_sample_size : sample_size = 36) :
  let elderly_sample := (elderly_population * sample_size) / total_population,
      middle_aged_sample := (middle_aged_population * sample_size) / total_population,
      young_sample := (young_population * sample_size) / total_population in
  elderly_sample = 6 ∧ middle_aged_sample = 12 ∧ young_sample = 18 :=
by sorry

end stratified_sample_count_l365_365884


namespace julia_monday_kids_l365_365719

theorem julia_monday_kids (x : ℕ) (h1 : x + 14 = 16) : x = 2 := 
by
  sorry

end julia_monday_kids_l365_365719


namespace real_part_max_l365_365781

-- Given conditions
variables {z w : ℂ}
variables (a b c d : ℝ)
variables (ha : |z| = 1) (hb : |w| = 1)
variables (hc : 2 * (z * w.conj + z.conj * w) = 2)
variables (hd : |z + w| = 2)

-- Prove the maximum value of the real part of z + w is 2.
theorem real_part_max {z w : ℂ} 
  (ha : |z| = 1) (hb : |w| = 1)
  (hc : 2 * (z * w.conj + z.conj * w) = 2)
  (hd : |z + w| = 2) : 
  ∃ (a c b d : ℝ), 
    max (a + c) = 2 ∧
    z = a + b * complex.I ∧
    w = c + d * complex.I ∧
    a^2 + b^2 = 1 ∧ c^2 + d^2 = 1 ∧ 2 * (a * c + b * d) = 1 ∧
    (a + c)^2 + (b + d)^2 = 4 := 
sorry

end real_part_max_l365_365781


namespace problem_solution_l365_365711

def is_prime (n : ℕ) : Prop := ∀ m, m > 1 → m < n → n % m ≠ 0

theorem problem_solution :
  ∃ a, (∃ B, (\overline (B : ℕ) = 41)  ∧ is_prime (\overline (B : ℕ))) ∧
       (∃ c, (\overline (c : ℕ) = 47) ∧ is_prime (\overline (c : ℕ))) ∧
       (is_prime (11 * a - 1) ∧ (11 * a - 1) = 43) :=
by
  use 4
  split
  { use 41
    split
    { exact rfl }
    { sorry } } -- proof that 41 is prime skipped
  split
  { use 47
    split
    { exact rfl }
    { sorry } } -- proof that 47 is prime skipped
  { split
    { sorry } -- proof that 43 is prime skipped
    { exact rfl } }

end problem_solution_l365_365711


namespace parabola_equation_l365_365603

theorem parabola_equation (a : ℝ) :
  (∀ x, (x + 1) * (x - 3) = 0 ↔ x = -1 ∨ x = 3) →
  (∀ y, y = a * (0 + 1) * (0 - 3) → y = 3) →
  a = -1 → 
  (∀ x, y = a * (x + 1) * (x - 3) → y = -x^2 + 2 * x + 3) :=
by
  intros h₁ h₂ ha
  sorry

end parabola_equation_l365_365603


namespace double_sum_eq_l365_365542

theorem double_sum_eq : 
  (∑ n in (finset.Ico 2 (⊤ : ℕ)), ∑ k in (finset.Ico 1 n), k / (3 : ℝ)^(n + k)) = (9 / 128 : ℝ) :=
sorry

end double_sum_eq_l365_365542


namespace part1_part2_part3_l365_365198

section CircleLine

-- Given: Circle C with equation x^2 + y^2 - 2x - 2y + 1 = 0
-- Tangent to line l intersecting the x-axis at A and the y-axis at B
variable (a b : ℝ) (ha : a > 2) (hb : b > 2)

-- Ⅰ. Prove that (a - 2)(b - 2) = 2
theorem part1 : (a - 2) * (b - 2) = 2 :=
sorry

-- Ⅱ. Find the equation of the trajectory of the midpoint of segment AB
theorem part2 (x y : ℝ) (hx : x > 1) (hy : y > 1) : (x - 1) * (y - 1) = 1 :=
sorry

-- Ⅲ. Find the minimum value of the area of triangle AOB
theorem part3 : ∃ (area : ℝ), area = 6 :=
sorry

end CircleLine

end part1_part2_part3_l365_365198


namespace max_distinct_prime_factors_a_l365_365782

theorem max_distinct_prime_factors_a
  (a b c : ℕ)
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_gcd : Multiset.card (Multiset.toFinset (Multiset.pmap Prime.uniq_factor_of a (λ a, h_pos_a))) = 10)
  (h_lcm : Multiset.card (Multiset.toFinset (Multiset.pmap Prime.uniq_factor_of (a * b) (λ s, h_pos_a))) = 40)
  (h_less : Multiset.card (Multiset.toFinset (Multiset.pmap Prime.uniq_factor_of a (λ a, h_pos_a))) <
            Multiset.card (Multiset.toFinset (Multiset.pmap Prime.uniq_factor_of b (λ b, h_pos_b))) ∧
            Multiset.card (Multiset.toFinset (Multiset.pmap Prime.uniq_factor_of a (λ a, h_pos_a))) <
            Multiset.card (Multiset.toFinset (Multiset.pmap Prime.uniq_factor_of c (λ c, h_pos_c)))) :
  Multiset.card (Multiset.toFinset (Multiset.pmap Prime.uniq_factor_of a (λ a, h_pos_a))) ≤ 24 := 
sorry

end max_distinct_prime_factors_a_l365_365782


namespace cos_pi_plus_alpha_l365_365249

theorem cos_pi_plus_alpha (α : ℝ) (h₁ : Real.sin α = Real.log 8 (1/4))
    (h₂ : α ∈ Set.Ioo (-Real.pi/2) 0) :
    Real.cos (Real.pi + α) = -Real.sqrt 5 / 3 :=
sorry

end cos_pi_plus_alpha_l365_365249


namespace double_sum_eq_l365_365545

theorem double_sum_eq : 
  (∑ n in (finset.Ico 2 (⊤ : ℕ)), ∑ k in (finset.Ico 1 n), k / (3 : ℝ)^(n + k)) = (9 / 128 : ℝ) :=
sorry

end double_sum_eq_l365_365545


namespace koschei_coins_l365_365309

theorem koschei_coins :
  ∃ a : ℕ, (a % 10 = 7) ∧ (a % 12 = 9) ∧ (300 ≤ a) ∧ (a ≤ 400) ∧ a = 357 :=
by 
  sorry

end koschei_coins_l365_365309


namespace rhombus_perimeter_l365_365388

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * (Nat.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 68 :=
by
  sorry

end rhombus_perimeter_l365_365388


namespace system_solution_unique_l365_365777

noncomputable def solve_system (n : ℕ) (x : Fin n → ℝ) : Prop :=
(∀ k : ℕ, k > 0 ∧ k ≤ n → ∑ i in Finset.univ, (x i) ^ k = n)

theorem system_solution_unique (n : ℕ) (x : Fin n → ℝ) :
  solve_system n x → (∀ i : Fin n, x i = 1) :=
by
  intro h
  -- Here would come the proof
  sorry

end system_solution_unique_l365_365777


namespace modulus_of_z_l365_365622

variable {z : ℂ}

theorem modulus_of_z (h : (1 + complex.I) * z = 3 + complex.I) : complex.abs z = real.sqrt 5 :=
sorry

end modulus_of_z_l365_365622


namespace sum_floor_log2_eq_18445_l365_365932

-- Define the sum function for the given range and floor(log2(N))
def floor_log2_sum : ℕ :=
  (∑ N in Finset.range (2048 + 1), Int.floor (Real.log N / Real.log 2))

-- The statement of the problem
theorem sum_floor_log2_eq_18445 : floor_log2_sum = 18445 := 
  sorry

end sum_floor_log2_eq_18445_l365_365932


namespace volunteer_arrangement_l365_365490

open Finset

theorem volunteer_arrangement {A B C D E : Type} :
  let students := {A, B, C, D, E} in
  let roles := {athletics, swimming, ball_games} in
  let no_swimming_choice := students.erase A in
  let arrangements := no_swimming_choice.card * (students.erase A).card.succ.pred in
  arrangements = 48 :=
by
  sorry

end volunteer_arrangement_l365_365490


namespace find_DZ_l365_365117

-- Define the elements of the parallelepiped and given conditions
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {A A1 B B1 C C1 D D1 X Y Z : V}

-- Define the conditions
def is_parallelepiped (A A1 B B1 C C1 D D1 : V) : Prop := sorry
def on_edge (X : V) (A1 D1 : V) : Prop := sorry
def on_edge' (Y : V) (B C : V) : Prop := sorry
def plane_intersection (C1 X Y : V) (DA : set V) (Z : V) : Prop := sorry

-- Provide the numerical values
def A1X_eq_5 := (dist A1 X) = 5
def BY_eq_3 := (dist B Y) = 3
def B1C1_eq_14 := (dist B1 C1) = 14

-- The proof problem in Lean statement
theorem find_DZ 
  (piped : is_parallelepiped A A1 B B1 C C1 D D1)
  (hx : on_edge X A1 D1)
  (hy : on_edge' Y B C)
  (hplane : plane_intersection C1 X Y (ray DA) Z)
  (h1 : A1X_eq_5)
  (h2 : BY_eq_3)
  (h3 : B1C1_eq_14) : 
  dist D Z = 20 :=
sorry

end find_DZ_l365_365117


namespace largest_number_with_7_and_3_l365_365434

/-
  Given the digits 7 and 3, each used only once, prove that the largest number
  that can be formed is 73.
-/

theorem largest_number_with_7_and_3 : ∀ (digits : List ℕ), digits = [7, 3] → (digits = [7, 3] ∨ digits = [3, 7]) → largest_number(digits) = 73 :=
by
  intros digits h1 h2
  sorry

noncomputable def largest_number : List ℕ → ℕ :=
by
  intro ds
  exact max (10 * ds.head! + ds.tail!.head!) (10 * ds.tail!.head! + ds.head!)

end largest_number_with_7_and_3_l365_365434


namespace double_sum_evaluation_l365_365536

theorem double_sum_evaluation : 
  (∑ n from 2 to ∞, ∑ k from 1 to (n - 1), k / (3 ^ (n + k))) = (6 / 25) :=
by sorry

end double_sum_evaluation_l365_365536


namespace area_union_square_circle_l365_365497

theorem area_union_square_circle :
  let side_length_square := 8
  let radius_circle := 12
  let area_square := side_length_square ^ 2
  let area_circle := Real.pi * (radius_circle ^ 2)
  let overlap_area := (1 / 4) * area_circle
  let union_area := area_square + area_circle - overlap_area
  union_area = 64 + 108 * Real.pi := by
begin
  sorry
end

end area_union_square_circle_l365_365497


namespace perimeter_of_region_l365_365696

noncomputable def region_perimeter (height : ℝ) (width : ℝ) : ℝ :=
  width + 2 * height + 4 + 8 * 1

theorem perimeter_of_region : 
  ∀ (height : ℝ) (width : ℝ), 
  (width = 12) → 
  (12 * height - 8 = 69) → 
  region_perimeter height width = 30 := 
by
  intros height width hw area_condition
  rw hw at area_condition
  have h : height = 77 / 12 := by
    linarith
  rw h
  apply calc ...
  sorry

end perimeter_of_region_l365_365696


namespace correct_M_min_t_for_inequality_l365_365220

-- Define the set M
def M : Set ℝ := {a | 0 ≤ a ∧ a < 4}

-- Prove that M is correct given ax^2 + ax + 2 > 0 for all x ∈ ℝ implies 0 ≤ a < 4
theorem correct_M (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + 2 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

-- Prove the minimum value of t given t > 0 and the inequality holds for all a ∈ M
theorem min_t_for_inequality (t : ℝ) (h : 0 < t) : 
  (∀ a ∈ M, (a^2 - 2 * a) * t ≤ t^2 + 3 * t - 46) ↔ 46 ≤ t :=
sorry

end correct_M_min_t_for_inequality_l365_365220


namespace average_of_first_12_even_is_13_l365_365061

-- Define the first 12 even numbers
def first_12_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

-- Define the sum of the first 12 even numbers
def sum_first_12_even : ℕ := first_12_even_numbers.sum

-- Define the number of values
def num_vals : ℕ := first_12_even_numbers.length

-- Define the average calculation
def average_first_12_even : ℕ := sum_first_12_even / num_vals

-- The theorem we want to prove
theorem average_of_first_12_even_is_13 : average_first_12_even = 13 := by
  sorry

end average_of_first_12_even_is_13_l365_365061


namespace cylindrical_to_rectangular_correct_l365_365942

def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_correct :
  cylindrical_to_rectangular 3 (Real.pi / 3) 8 = (1.5, (3 * Real.sqrt 3) / 2, 8) :=
by
  -- Definitions and steps are not required; only the statement is necessary.
  sorry

end cylindrical_to_rectangular_correct_l365_365942


namespace trapezium_side_length_l365_365576

variable (length1 length2 height area : ℕ)

theorem trapezium_side_length
  (h1 : length1 = 20)
  (h2 : height = 15)
  (h3 : area = 270)
  (h4 : area = (length1 + length2) * height / 2) :
  length2 = 16 :=
by
  sorry

end trapezium_side_length_l365_365576


namespace exists_real_root_in_interval_l365_365397

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 3

theorem exists_real_root_in_interval (f : ℝ → ℝ)
  (h_mono : ∀ x y, x < y → f x < f y)
  (h1 : f 1 < 0)
  (h2 : f 2 > 0) : 
  ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 := 
sorry

end exists_real_root_in_interval_l365_365397


namespace valid_paths_from_A_to_B_l365_365939

def valid_paths_6x3_grid : ℕ :=
  6 * 3 -- This is a placeholder for the actual computation of paths.

theorem valid_paths_from_A_to_B (A B : ℕ):
  (A = (0, 3 ∨ 0, 0)) ∧ (B = (6, 0)) →
  valid_paths_6x3_grid = 48 :=
by
  sorry

end valid_paths_from_A_to_B_l365_365939


namespace total_cost_l365_365387

def cost(M R F : ℝ) := 10 * M = 24 * R ∧ 6 * F = 2 * R ∧ F = 23

theorem total_cost (M R F : ℝ) (h : cost M R F) : 
  4 * M + 3 * R + 5 * F = 984.40 :=
by
  sorry

end total_cost_l365_365387


namespace probability_outside_inner_circle_l365_365085

theorem probability_outside_inner_circle 
  (r : ℝ) (r_pos : 0 < r) (h : ∀ (x y : ℝ), concentric x y) :
  let A_Y := real.pi * (r ^ 2)
  let A_X := real.pi * (6 * r) ^ 2
  let A_outside_Y := A_X - A_Y
  let P := A_outside_Y / A_X
  P = 35 / 36 :=
by 
  sorry

end probability_outside_inner_circle_l365_365085


namespace function_must_pass_through_l365_365129

theorem function_must_pass_through :
  ∀ (a : ℝ), (a > 0 ∧ a ≠ 1) → (∃ (x y : ℝ), x = 2 ∧ y = 1 ∧ y = log a (x - 1) + 1) :=
by
  intro a ha
  use [2, 1]
  simp [*, Real.log]
  sorry

end function_must_pass_through_l365_365129


namespace problem_l365_365250

theorem problem (a b : ℝ) (h : sqrt (a + 1) + sqrt (b - 1) = 0) : a ^ 1011 + b ^ 1011 = 0 := 
sorry

end problem_l365_365250


namespace intersecting_lines_properties_l365_365074

theorem intersecting_lines_properties (l1 l2 l3 : Line) (h : ¬ (parallel l1 l2)) :
  ¬ (alternateInteriorAnglesEqual l1 l2 l3) ∧
  ¬ (correspondingAnglesEqual l1 l2 l3) ∧
  ¬ (consecutiveInteriorAnglesSupplementary l1 l2 l3) :=
by
  sorry

end intersecting_lines_properties_l365_365074


namespace parallel_lines_slope_eq_l365_365073

variable (k : ℝ)

theorem parallel_lines_slope_eq (h : 5 = 3 * k) : k = 5 / 3 :=
by
  sorry

end parallel_lines_slope_eq_l365_365073


namespace part1_part2_l365_365681

variable {A B C : ℝ} -- Declare variables for the angles in the triangle

-- Defining the condition provided in the problem
def given_condition : Prop :=
  (sin A + sin B + sin C) * (sin B + sin C - sin A) = 3 * sin B * sin C

-- Statement for the first part of the problem
theorem part1 (h : given_condition) : A = π / 3 := sorry

-- Define the maximum value expression to be maximized under given constraints
def expr (B C : ℝ) : ℝ := sqrt 3 * sin B - cos C

-- Statement for the second part of the problem
theorem part2 (h : A = π / 3) : 
  ∃ B C : ℝ, (B + C = 2 * π / 3 ∧ (expr B C) ≤ 1) := sorry

end part1_part2_l365_365681


namespace problem1_l365_365094

theorem problem1 (x : ℝ) : ∀ (y : ℝ), y = (1 + 5 * x)^3 → deriv (λ x, y) = 15 * (1 + 5 * x)^2 :=
by
  intros y hy
  sorry

end problem1_l365_365094


namespace butterfly_theorem_extended_l365_365928

/-- Butterfly theorem on circles -/
theorem butterfly_theorem_extended (O1 O2 : Circle) (A B : Point) (O : Point) (C D E F P Q : Point) :
  circle_eq_radius O1 O2 → 
  point_on_circle A O1 → 
  point_on_circle A O2 → 
  point_on_circle B O1 → 
  point_on_circle B O2 → 
  midpoint O A B → 
  chord_through_midpoint C D O1 O → 
  chord_intersects_other_circle_at P O1 O2 C D → 
  chord_through_midpoint E F O2 O → 
  chord_intersects_other_circle_at Q O2 O1 E F → 
  concurrent (line_through_points A B) (line_through_points C Q) (line_through_points E P) := 
begin
  sorry  
end

end butterfly_theorem_extended_l365_365928


namespace value_of_expression_l365_365855

theorem value_of_expression : 3^2 * 5 * 7^2 * 11 = 24255 := by
  have h1 : 3^2 = 9 := by norm_num
  have h2 : 7^2 = 49 := by norm_num
  calc
    3^2 * 5 * 7^2 * 11
        = 9 * 5 * 7^2 * 11 : by rw h1
    ... = 9 * 5 * 49 * 11  : by rw h2
    ... = 24255            : by norm_num

end value_of_expression_l365_365855


namespace infinite_series_eq_15_l365_365184

theorem infinite_series_eq_15 (x : ℝ) :
  (∑' (n : ℕ), (5 + n * x) / 3^n) = 15 ↔ x = 10 :=
by
  sorry

end infinite_series_eq_15_l365_365184


namespace MethodC_is_not_systematic_l365_365076

def MethodA : Prop := ∃ k : ℕ, k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} ∧ 
                      ∀ n : ℕ, (n ∈ {k+5, k+10} ∨ (k+5 > 15 ∧ n = k+5 - 15) ∨ (k+10 > 15 ∧ n = k+10 - 15))

def MethodB : Prop := ∃ m : ℕ, m ∈ {0, 5..}

def MethodC : Prop := ∃ n : ℕ, n ∈ ℕ ∧ true

def MethodD : Prop := true

def systematic_sampling (M : Prop) : Prop := 
∃ k : ℕ, ∀ j : ℕ, j ∈ {k+5, k+10} ∨ (k+5 > 15 ∧ j = k+5 - 15) ∨ (k+10 > 15 ∧ j = k+10 - 15)

theorem MethodC_is_not_systematic : ¬ systematic_sampling MethodC := 
  sorry

end MethodC_is_not_systematic_l365_365076


namespace triangle_ratio_l365_365702

theorem triangle_ratio 
  (A B C D F Q : Type)
  (hQ1 : AQ / QD = 5 / 2)
  (hQ2 : BQ / QF = 3 / 4)
  (on_line_D : D ∈ line B C)
  (on_line_F : F ∈ line A B)
  (intersect_Q : Q = intersection (line A D) (line B F)) :
  AF / FB = 1 / 2 := by 
  sorry

end triangle_ratio_l365_365702


namespace profit_function_maximize_profit_l365_365503

def cost_per_item : ℝ := 80
def purchase_quantity : ℝ := 1000
def selling_price_initial : ℝ := 100
def price_increase_per_item : ℝ := 1
def sales_decrease_per_yuan : ℝ := 10
def selling_price (x : ℕ) : ℝ := selling_price_initial + x
def profit (x : ℕ) : ℝ := (selling_price x - cost_per_item) * (purchase_quantity - sales_decrease_per_yuan * x)

theorem profit_function (x : ℕ) (h : 0 ≤ x ∧ x ≤ 100) : 
  profit x = -10 * (x : ℝ)^2 + 800 * (x : ℝ) + 20000 :=
by sorry

theorem maximize_profit :
  ∃ max_x, (0 ≤ max_x ∧ max_x ≤ 100) ∧ 
  (∀ x : ℕ, (0 ≤ x ∧ x ≤ 100) → profit x ≤ profit max_x) ∧ 
  max_x = 40 ∧ 
  profit max_x = 36000 :=
by sorry

end profit_function_maximize_profit_l365_365503


namespace partI_partII_l365_365231

def f (x a : ℝ) : ℝ := |x - 2| - |2 * x - a|

-- Part I: Prove that f(x) > 0 when a = 3 if and only if 1 < x < 5/3
theorem partI (x : ℝ) : (f x 3 > 0) ↔ (1 < x ∧ x < (5:ℝ)/3) := 
by
  sorry

-- Part II: Prove that f(x) < 0 always holds when x ∈ (-∞, 2) if and only if a ≥ 4
theorem partII (a : ℝ) : (∀ x ∈ set.Iio 2, f x a < 0) ↔ (4 ≤ a) := 
by
  sorry

end partI_partII_l365_365231


namespace crystal_total_money_l365_365553

-- Define the basic prices and reductions
def original_price_cupcake := 3.00
def original_price_cookie := 2.00

def reduced_price_cupcake := original_price_cupcake / 2
def reduced_price_cookie := original_price_cookie / 2

-- Define the number of cupcakes and cookies sold
def number_of_cupcakes_sold := 16
def number_of_cookies_sold := 8

-- Define the total money made from cupcakes and cookies
def total_money_from_cupcakes := number_of_cupcakes_sold * reduced_price_cupcake
def total_money_from_cookies := number_of_cookies_sold * reduced_price_cookie

-- The total money made from selling all pastries
def total_money_made := total_money_from_cupcakes + total_money_from_cookies

-- Theorem statement
theorem crystal_total_money : total_money_made = 32.00 := sorry

end crystal_total_money_l365_365553


namespace num_valid_k_l365_365947

theorem num_valid_k : (∃ n : ℕ, n = 6) :=
begin
  sorry
end

end num_valid_k_l365_365947


namespace min_x_log_sqrt_defined_l365_365157

theorem min_x_log_sqrt_defined : ∀ x : ℝ, x > 3001 ^ 3002 → 
  (log 3004 (log 3003 (log 3002 (log 3001 x))) + sqrt (x - 3000) > 3) := 
begin
  sorry
end

end min_x_log_sqrt_defined_l365_365157


namespace find_length_BO_l365_365887

-- Define the given geometric configuration and conditions
variables {O M N B K T A : Type*} {r a : ℝ}
variables [MetricSpace O] [MetricSpace M] [MetricSpace N] [MetricSpace B] [MetricSpace K] [MetricSpace T] [MetricSpace A]
variables (circle : Metric.ball O r)
variables (touches_BA : ∃ M : O, Metric.Metric.ball O r ∩ Metric.line_segment B A = {M})
variables (touches_BC : ∃ N : O, Metric.Metric.ball O r ∩ Metric.line_segment B C = {N})
variables (MK_parallel_BC : Metric.parallel M K B C)
variables (K_on_BO : Metric.belongs K B O)
variables (T_on_MN : ∃ T : O, Metric.belongs T (Metric.line_segment M N))
variables (half_angle_condition : ∀ (ABC : ℝ), ∃ T : O, ∃ (half_angle : ℝ), half_angle = 1/2 * ABC)

noncomputable def length_BO {O B : Type*} [MetricSpace O] [MetricSpace B] (r a : ℝ) : ℝ :=
  sqrt (r * (a + r))

theorem find_length_BO 
  (circle : Metric.ball O r)
  (touches_BA : ∃ M : O, Metric.Metric.ball O r ∩ Metric.line_segment B A = {M})
  (touches_BC : ∃ N : O, Metric.Metric.ball O r ∩ Metric.line_segment B C = {N})
  (MK_parallel_BC : Metric.parallel M K B C)
  (K_on_BO : Metric.belongs K B O)
  (T_on_MN : ∃ T : O, Metric.belongs T (Metric.line_segment M N))
  (half_angle_condition : ∀ (ABC : ℝ), ∃ T : O, ∃ (half_angle : ℝ), half_angle = 1/2 * ABC)
  (KT_eq_a : ∃ T : O, Metric.distance K T = a) :
  Metric.distance O B = sqrt (r * (a + r)) := 
  sorry

end find_length_BO_l365_365887


namespace profit_from_ad_l365_365300

def advertising_cost : ℝ := 1000
def customers : ℕ := 100
def purchase_rate : ℝ := 0.8
def purchase_price : ℝ := 25

theorem profit_from_ad (advertising_cost customers purchase_rate purchase_price : ℝ) : 
  (customers * purchase_rate * purchase_price - advertising_cost) = 1000 :=
by
  -- assumptions as conditions
  let bought_customers := (customers : ℝ) * purchase_rate
  let revenue := bought_customers * purchase_price
  let profit := revenue - advertising_cost
  -- state the proof goal
  have goal : profit = 1000 :=
    sorry
  exact goal

end profit_from_ad_l365_365300


namespace geom_seq_general_term_l365_365396

noncomputable def a_n (n : ℕ) : ℝ := 8 * (1/2)^(n-1)

theorem geom_seq_general_term:
  ∃ q (a_1 : ℝ), 
    (q ≠ 0 ∧ q ≠ 1) ∧ 
    (a_1 + a_1 * q^2 = 10) ∧ 
    (a_1 * q + a_1 * q^3 = 5) ∧ 
    (∀ n : ℕ, a_n n = 8 * (1/2)^(n-1)) :=
begin
  sorry
end

end geom_seq_general_term_l365_365396


namespace enclosed_area_eq_32_over_3_l365_365575

def line (x : ℝ) : ℝ := 2 * x + 3
def parabola (x : ℝ) : ℝ := x^2

theorem enclosed_area_eq_32_over_3 :
  ∫ x in (-(1:ℝ))..(3:ℝ), (line x - parabola x) = 32 / 3 :=
by
  sorry

end enclosed_area_eq_32_over_3_l365_365575


namespace triangle_ABK_properties_l365_365699

theorem triangle_ABK_properties :
  ∀ (A B C K : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ K] 
  (AB : ℝ) (BC : ℝ) (AC : ℝ) (AB_eq : AB = 15) (BC_eq : BC = 13) (AC_eq : AC = 14)
  (h : ∃ K, is_perpendicular (C - K) (A - B)),
  let BK := (65 : ℝ) / 3 in
  let CK := (56 : ℝ) / 3 in
  dist B K = BK ∧ dist C K = CK :=
by
  sorry

end triangle_ABK_properties_l365_365699


namespace P_2007_gt_P_2008_l365_365903

-- Define the probability function P
def P : ℕ → ℝ := sorry

-- Define the initial condition
axiom P_initial : P 0 = 1

-- Define the difference equation
axiom P_diff (k : ℕ) : P k = (1 / 2008) * (∑ i in Finset.range 2008, P (k - i))

-- Theorem: P_2007 > P_2008
theorem P_2007_gt_P_2008 : P 2007 > P 2008 :=
by sorry

end P_2007_gt_P_2008_l365_365903


namespace solve_trig_expression_l365_365586

-- Definitions based on the conditions
def expr (θ φ ψ: ℝ) :=
  (sin θ * sin θ + cos θ * sin φ - tan (ψ/2) * tan (ψ/2)) / (3 * tan (ψ / 2))

def θ := 38 * real.pi / 180  -- convert degrees to radians
def φ := 52 * real.pi / 180  -- convert degrees to radians
def ψ := 30 * real.pi / 180  -- convert degrees to radians

-- The theorem to prove
theorem solve_trig_expression :
  expr θ φ ψ = 2 * sqrt 3 / 3 :=
sorry

end solve_trig_expression_l365_365586


namespace board_filling_ways_l365_365664

theorem board_filling_ways : 
  ∃ (fillings : fin 4 → fin 4 → ℕ), 
    (∀ i, (∑ j, fillings i j) = 3) ∧ 
    (∀ j, (∑ i, fillings i j) = 3) ∧ 
    (card {fillings // (∀ i, (∑ j, fillings i j) = 3) ∧ (∀ j, (∑ i, fillings i j) = 3)}) = 2008 :=
sorry

end board_filling_ways_l365_365664


namespace sodium_hydride_reacts_to_form_NaOH_l365_365970

theorem sodium_hydride_reacts_to_form_NaOH (mNaH : ℝ):
  (∃ (nNaOH : ℝ), mNaH + 1 = nNaOH + 1 ∧ nNaOH = 1) → mNaH = 1 :=
by
  assume h,
  let ⟨nNaOH, h_condition, h_produced⟩ := h,
  calc
    mNaH = nNaOH : by sorry
    ...   = 1     : by exact h_produced

end sodium_hydride_reacts_to_form_NaOH_l365_365970


namespace sum_double_series_l365_365532

theorem sum_double_series :
  (∑ n from 2 to ∞, ∑ k from 1 to (n-1), k / 3^(n+k)) = 9 / 136 :=
by
  sorry

end sum_double_series_l365_365532


namespace general_formula_arithmetic_sequence_sum_of_reciprocals_lt_two_l365_365611

-- Definitions based on the given conditions
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n : ℕ, b (n + 1) = b n * q

-- Given conditions
def a : ℕ → ℕ := λ n, 3 * n - 2
def b : ℕ → ℕ := λ n, 2 ^ (n - 1)
def T (n : ℕ) : ℕ := 2^n - 1

axiom condition_a1 : a 1 = 1
axiom condition_b1 : b 1 = 1
axiom condition_a2_b3 : a 2 = b 3
axiom condition_a6_b5 : a 6 = b 5

-- Mathematical equivalent proof problems
theorem general_formula_arithmetic_sequence :
  ∀ n : ℕ, a n = 3 * n - 2 :=
sorry

theorem sum_of_reciprocals_lt_two (n : ℕ) :
  (finset.range n).sum (λ i, 1 / T (i + 1) : ℝ) < 2 :=
sorry

end general_formula_arithmetic_sequence_sum_of_reciprocals_lt_two_l365_365611


namespace pentagon_angle_E_l365_365293

theorem pentagon_angle_E 
    (A B C D E : Type)
    [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
    (AB BC CD DE : ℝ)
    (angle_B angle_C angle_D : ℝ)
    (h1 : AB = BC)
    (h2 : BC = CD)
    (h3 : CD = DE)
    (h4 : angle_B = 96)
    (h5 : angle_C = 108)
    (h6 : angle_D = 108) :
    ∃ angle_E : ℝ, angle_E = 102 := 
by
  sorry

end pentagon_angle_E_l365_365293


namespace jackson_vacuums_l365_365299

-- Definitions from the given conditions
constant earnings_per_hour : ℕ := 5
constant time_vacuuming_per_instance : ℕ := 2
constant time_washing_dishes : ℝ := 0.5
constant time_cleaning_bathroom : ℝ := 3 * time_washing_dishes
constant total_earnings : ℕ := 30

-- Statement of the problem to prove
theorem jackson_vacuums (V : ℕ) (T : ℝ) :
  T = (2 * V) + time_washing_dishes + time_cleaning_bathroom → (earnings_per_hour * T = total_earnings) → V = 2 :=
by
  intros hT hE.
  have h1 : T = 2 * V + 2, from by
  {
    rw [hT, time_washing_dishes, time_cleaning_bathroom], 
    norm_num,
  },
  rw [h1] at hE,
  have h2 : 10 * V + 10 = 30, from hE,
  norm_num at h2,
  sorry


end jackson_vacuums_l365_365299


namespace charles_remaining_skittles_l365_365144

def c : ℕ := 25
def d : ℕ := 7
def remaining_skittles : ℕ := c - d

theorem charles_remaining_skittles : remaining_skittles = 18 := by
  sorry

end charles_remaining_skittles_l365_365144


namespace find_m_l365_365648

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (2, m)
def vector_b : ℝ × ℝ := (1, 1)
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2
def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_m (m : ℝ) :
  dot_product (vector_a m) vector_b = magnitude (vector_sub (vector_a m) vector_b) → 
  m = -1 / 3 := by
  sorry

end find_m_l365_365648


namespace vaccine_codes_l365_365271

theorem vaccine_codes (vaccines : List ℕ) :
  vaccines = [785, 567, 199, 507, 175] :=
  by
  sorry

end vaccine_codes_l365_365271


namespace part1_part2_part3_l365_365228

open Real

noncomputable def f (a x : ℝ) : ℝ := x / log x + a * x

theorem part1 (a : ℝ) : (∀ x > 1, deriv (f a) x ≤ 0) → a ≤ -1/4 :=
sorry

theorem part2 : f 2 (exp (1/2)) = 4 * exp (1/2) :=
sorry

noncomputable def g (x : ℝ) : ℝ := x / log x + 2 * x

theorem part3 (m : ℝ) :
  (∃ x1 x2 ∈ Ioo(1, exp 1), 
  x1 ≠ x2 ∧ (2 * x1 - m) * log x1 + x1 = 0 ∧ (2 * x2 - m) * log x2 + x2 = 0) → m ∈ Ioo(4 * exp (1/2), 3 * exp 1) :=
sorry

end part1_part2_part3_l365_365228


namespace square_area_l365_365968

theorem square_area (side_length : ℕ) (h : side_length = 3) : side_length * side_length = 9 :=
by {
  rw h,  -- replace side_length with 3
  norm_num, -- simplify the expression 3 * 3
  exact rfl,
}

end square_area_l365_365968


namespace segment_DB_length_l365_365693

theorem segment_DB_length (AC AD : ℝ) (right_angle_ABC : ∠ABC = 90) (right_angle_ADB : ∠ADB = 90) (AC_eq : AC = 20) (AD_eq : AD = 8) :
  let DC := AC - AD in
  let triangles_similar := ∠ADB = ∠CDB in
  DC = 12 → triangles_similar →
  DB = 4 * real.sqrt 6 :=
by
  intros
  let DC := AC - AD
  have DC_eq : DC = 12 := by sorry
  have simi : triangles_similar := by sorry
  let DB := 4 * real.sqrt 6
  sorry

end segment_DB_length_l365_365693


namespace koschei_coins_l365_365308

theorem koschei_coins :
  ∃ a : ℕ, (a % 10 = 7) ∧ (a % 12 = 9) ∧ (300 ≤ a) ∧ (a ≤ 400) ∧ a = 357 :=
by 
  sorry

end koschei_coins_l365_365308


namespace ratio_DE_DF_l365_365606

variable (A B C D E F E' F' : Point)
variable (λ μ : ℝ)
variable (AB : Line)
variable (AD : Line)
variable (incircle : Circle)

-- Assuming conditions as variables
variable (is_rhombus : rhombus A B C D)
variable (on_extension_AB : E ∈ Line.extension AB B ∧ F ∈ Line.extension AB B)
variable (tangents_from_E : tangent_to_circle incircle E ∧ tangent_meets_line_at E AD E')
variable (tangents_from_F : tangent_to_circle incircle F ∧ tangent_meets_line_at F AD F')
variable (ratio_BE_BF : ratio BE BF = λ / μ)

theorem ratio_DE_DF :
  ratio DE' DF' = μ / λ :=
by
  sorry

end ratio_DE_DF_l365_365606


namespace count_ordered_pairs_l365_365557

theorem count_ordered_pairs (x y : ℕ) (px : 0 < x) (py : 0 < y) (h : 2310 = 2 * 3 * 5 * 7 * 11) :
  (x * y = 2310 → ∃ n : ℕ, n = 32) :=
by
  sorry

end count_ordered_pairs_l365_365557


namespace ideal_gas_temperature_l365_365931

-- Given values
def p := 9 * 10^5 -- Pressure in Pa
def V := 95 * 10^{-5} -- Volume in m^3
def n := 0.5 -- Number of moles
def R := 8.31 -- Ideal gas constant in J/(mol·K)

-- Calculate temperature using ideal gas law
def T : Real := (p * V) / (n * R)

-- Proof that T is approximately 206 K
theorem ideal_gas_temperature :
  abs (T - 206) < 1 := by
  sorry

end ideal_gas_temperature_l365_365931


namespace num_children_attended_show_l365_365028

def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := 13
def num_adults : ℕ := 183
def total_revenue : ℕ := 5122

theorem num_children_attended_show : ∃ C : ℕ, (num_adults * ticket_price_adult + C * ticket_price_child = total_revenue) ∧ C = 28 :=
by
  sorry

end num_children_attended_show_l365_365028


namespace XY_sym_diff_l365_365590

-- The sets X and Y
def X : Set ℤ := {1, 3, 5, 7}
def Y : Set ℤ := { x | x < 4 ∧ x ∈ Set.univ }

-- Definition of set operation (A - B)
def set_sub (A B : Set ℤ) : Set ℤ := { x | x ∈ A ∧ x ∉ B }

-- Definition of set operation (A * B)
def set_sym_diff (A B : Set ℤ) : Set ℤ := (set_sub A B) ∪ (set_sub B A)

-- Prove that X * Y = {-3, -2, -1, 0, 2, 5, 7}
theorem XY_sym_diff : set_sym_diff X Y = {-3, -2, -1, 0, 2, 5, 7} :=
by
  sorry

end XY_sym_diff_l365_365590


namespace luca_lost_more_weight_l365_365137

theorem luca_lost_more_weight (barbi_kg_month : ℝ) (luca_kg_year : ℝ) (months_in_year : ℕ) (years : ℕ) 
(h_barbi : barbi_kg_month = 1.5) (h_luca : luca_kg_year = 9) (h_months_in_year : months_in_year = 12) (h_years : years = 11) : 
  (luca_kg_year * years) - (barbi_kg_month * months_in_year * (years / 11)) = 81 := 
by 
  sorry

end luca_lost_more_weight_l365_365137


namespace integer_values_of_n_l365_365982

theorem integer_values_of_n :
  (∃ (n : ℤ), ∀ (m : ℤ), 5000 * ((5 / 2) ^ n) = m → -3 ≤ n ∧ n ≤ 3) → 
    fintype.card {n : ℤ | -3 ≤ n ∧ n ≤ 3} = 7 :=
by
  sorry

end integer_values_of_n_l365_365982


namespace mary_fruits_left_l365_365348

theorem mary_fruits_left (apples_initial : ℕ) (oranges_initial : ℕ) (blueberries_initial : ℕ)
                         (ate_apples : ℕ) (ate_oranges : ℕ) (ate_blueberries : ℕ) :
  apples_initial = 14 → oranges_initial = 9 → blueberries_initial = 6 → 
  ate_apples = 1 → ate_oranges = 1 → ate_blueberries = 1 → 
  (apples_initial - ate_apples) + (oranges_initial - ate_oranges) + (blueberries_initial - ate_blueberries) = 26 :=
by
  intros
  simp [*]
  sorry

end mary_fruits_left_l365_365348


namespace constant_polynomial_l365_365961

noncomputable def polynomials_constant (P : Polynomial ℝ → Polynomial ℝ → Polynomial ℝ) : Prop := 
  ∀ (x y : ℝ), P (x + y) (y - x) = P x y

theorem constant_polynomial (P : Polynomial ℝ → Polynomial ℝ → Polynomial ℝ) (h : polynomials_constant P) :
  ∃ c : ℝ, ∀ x y : ℝ, P x y = c :=
sorry

end constant_polynomial_l365_365961


namespace number_of_acceptable_outfits_l365_365450

-- Definitions based on conditions
def total_shirts := 5
def total_pants := 4
def restricted_shirts := 2
def restricted_pants := 1

-- Defining the problem statement
theorem number_of_acceptable_outfits : 
  (total_shirts * total_pants - restricted_shirts * restricted_pants + restricted_shirts * (total_pants - restricted_pants)) = 18 :=
by sorry

end number_of_acceptable_outfits_l365_365450


namespace coin_probability_l365_365478

theorem coin_probability :
  ∃ p : ℝ, (1 / 2 < p) ∧ (p^3 * (1 - p)^2 = 1 / 100) ∧ (p = (6 + sqrt (6 * sqrt 6 + 2)) / 12) :=
by sorry

end coin_probability_l365_365478


namespace count_valid_n_l365_365581

theorem count_valid_n : 
  ∃ (count : ℕ), count = 88 ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 2000 ∧ 
   (∃ (a b : ℤ), a + b = -2 ∧ a * b = -n) ↔ 
   ∃ m, 1 ≤ m ∧ m ≤ 2000 ∧ (∃ a, a * (a + 2) = m)) := 
sorry

end count_valid_n_l365_365581


namespace cows_ran_away_l365_365868

theorem cows_ran_away (total_cows days food_consumed : ℕ) (H1 : total_cows = 1000) (H2 : days = 50) (H3 : food_consumed = 40_000) : 
  ∃ x : ℕ, x = 200 ∧ food_consumed = (total_cows - x) * days - total_cows * 10 := 
by
  sorry

end cows_ran_away_l365_365868


namespace triangle_area_ab_l365_365252

theorem triangle_area_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0.5 * (12 / a) * (12 / b) = 12) : a * b = 6 :=
by
  sorry

end triangle_area_ab_l365_365252


namespace largest_possible_value_is_one_l365_365325

open Complex

noncomputable def largest_possible_value {a b c z : ℂ}
  (h1 : |a| = 2 * |b|)
  (h2 : |b| = |c|)
  (h3 : a ≠ 0)
  (h4 : a * z^2 + b * z + c = 0) : ℝ :=
  1

theorem largest_possible_value_is_one {a b c z : ℂ}
  (h1 : |a| = 2 * |b|)
  (h2 : |b| = |c|)
  (h3 : a ≠ 0)
  (h4 : a * z^2 + b * z + c = 0) :
  largest_possible_value h1 h2 h3 h4 = 1 := 
begin
  -- proof skipped
  sorry
end

end largest_possible_value_is_one_l365_365325


namespace SUCCESSOR_arrangement_count_l365_365948

theorem SUCCESSOR_arrangement_count :
  (Nat.factorial 9) / (Nat.factorial 3 * Nat.factorial 2) = 30240 :=
by
  sorry

end SUCCESSOR_arrangement_count_l365_365948


namespace first_player_does_not_necessarily_win_l365_365046

theorem first_player_does_not_necessarily_win :
  ∀ (cube : Type) (edges : set (cube → cube)), 
  (∀ (color : cube → cube → Prop), 
  ∃ (strategy1 strategy2 : list (set (cube → cube))), 
  (length strategy1 = length strategy2) ∧
  (set.pairwise_disjoint strategy1 edges) ∧ 
  (set.pairwise_disjoint strategy2 edges) ∧ 
  ¬ (∃ (face : set (cube → cube)), 
  (∀ (e ∈ face), color e = (∃ i : ℕ, e ∈ strategy1 i))) ∧ 
  ¬ (∃ (face : set (cube → cube)), 
  (∀ (e ∈ face), color e = (∃ i : ℕ, e ∈ strategy2 i)))) → 
  ¬ ∃ (optimal_strategy : list (set (cube → cube))), 
  ∀ (color : cube → cube → Prop), 
  color ∈ optimal_strategy → 
  (∃ (face : set (cube → cube)), (∀ (e ∈ face), color e)) :=
sorry

end first_player_does_not_necessarily_win_l365_365046


namespace exists_xyz_prime_expression_l365_365092

theorem exists_xyz_prime_expression (a b c p : ℤ) (h_prime : Prime p)
    (h_div : p ∣ (a^2 + b^2 + c^2 - ab - bc - ca))
    (h_gcd : Int.gcd p ((a^2 + b^2 + c^2 - ab - bc - ca) / p) = 1) :
    ∃ x y z : ℤ, p = x^2 + y^2 + z^2 - xy - yz - zx := by
  sorry

end exists_xyz_prime_expression_l365_365092


namespace train_length_l365_365110

theorem train_length (initial_speed_kmh : ℝ) (platform_length_m : ℝ) (crossing_time_s : ℝ) (acceleration_m_s2 : ℝ) : 
  initial_speed_kmh = 72 → 
  platform_length_m = 270 → 
  crossing_time_s = 26 → 
  acceleration_m_s2 = 0.1 → 
  ∃ length_of_train_m : ℝ, length_of_train_m = 283.8 :=
by
  -- Convert initial speed from km/h to m/s
  let initial_speed_m_s := 20
  -- Using the kinematic equation: s = ut + (1/2)at^2
  let distance_covered := (initial_speed_m_s * crossing_time_s) + (1 / 2 * acceleration_m_s2 * crossing_time_s^2)
  -- The distance covered is the sum of the length of the train and the length of the platform
  let length_of_train := distance_covered - platform_length_m
  existsi length_of_train
  sorry

end train_length_l365_365110


namespace correct_option_B_l365_365079

noncomputable def P : Prop := ∃ x ∈ set.Ioo (0:ℝ) real.pi, x + 1 / real.sin x ≤ 2

theorem correct_option_B : ¬P :=
by
  -- Here we state that Option B is correct, indicating that P is false, hence ¬P is true
  sorry

end correct_option_B_l365_365079


namespace inscribed_sphere_volume_l365_365630

noncomputable def volume_of_inscribed_sphere_in_cone
  (r : ℝ) (A : ℝ) (H_A : A = 8 * Real.pi) (H_r : r = 2) : ℝ :=
let l := 4 in
let h := 2 * Real.sqrt 3 in
let R := (2 * Real.sqrt 3) / 3 in
(4 * Real.pi * R^3) / 3

theorem inscribed_sphere_volume :
  volume_of_inscribed_sphere_in_cone 2 8*Real.pi (by norm_num) (by norm_num) = (32 * Real.sqrt 3 / 27) * Real.pi :=
sorry

end inscribed_sphere_volume_l365_365630


namespace focus_of_parabola_l365_365012

-- Define the given parabola equation
def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the parameter p for this specific parabola
def p : ℝ := 1 / 2

-- Define the expected coordinates of the focus
def focus : ℝ × ℝ := (1 / 4, 0)

-- Statement to prove that the coordinates of the focus of the parabola y^2 = x are (1/4, 0).
theorem focus_of_parabola : ∃ F : ℝ × ℝ, (F = (1 / 4, 0)) ∧ (focus = F) :=
by
  sorry

end focus_of_parabola_l365_365012


namespace parabola_intersection_probability_l365_365839

noncomputable def intersection_probability : ℚ :=
  let die_outcomes := {1, 2, 3, 4, 5, 6}
  let total_outcomes := 6 * 6 * 6 * 6 -- Total ways to choose a, b, c, d
  let favorable_outcomes := 
        ((die_outcomes.product die_outcomes).product (die_outcomes.product die_outcomes)).count
        (λ (((a, b), c), d), a ≠ c ∨ b = d)
  favorable_outcomes / total_outcomes

theorem parabola_intersection_probability :
  intersection_probability = 31 / 36 := 
  sorry

end parabola_intersection_probability_l365_365839


namespace equation_represents_circle_and_line_l365_365015

-- Definitions for the conditions
def equation (ρ θ : ℝ) : Prop :=
  ρ^2 * cos θ + ρ - 3 * ρ * cos θ - 3 = 0

-- Statement of the theorem
theorem equation_represents_circle_and_line :
  forall (ρ θ : ℝ), equation ρ θ ->
  (∃ (x y : ℝ), (x^2 + y^2 = 9 ∧ x = ρ * cos θ) ∨ (x = -1)) :=
begin
  -- Proof will be provided here
  sorry
end

end equation_represents_circle_and_line_l365_365015


namespace most_stable_shape_l365_365918

variables (Rectangle Trapezoid Parallelogram Triangle : Type)

axiom stability : Type → ℝ

-- Assuming that stability is a property measured by a real number
-- where higher values mean more stability, we state the following conditions:
axiom stability_Rectangle : stability Rectangle = 1
axiom stability_Trapezoid : stability Trapezoid = 2
axiom stability_Parallelogram : stability Parallelogram = 3
axiom stability_Triangle : stability Triangle = 4

theorem most_stable_shape : ∀ (shapes : list Type), Triangle ∈ shapes → Rectangle ∈ shapes → Trapezoid ∈ shapes → Parallelogram ∈ shapes → 
  (∀ shape ∈ shapes, stability shape ≤ stability Triangle) :=
by 
  intros shapes hT hR hTr hP shape hshapes
  cases shape
  case Rectangle:
    rw [hR, stability_Rectangle]
    linarith [stability_Triangle]
  case Trapezoid:
    rw [hTr, stability_Trapezoid]
    linarith [stability_Triangle]
  case Parallelogram:
    rw [hP, stability_Parallelogram]
    linarith [stability_Triangle]
  case Triangle:
    rw [hT]
    exact le_refl (stability Triangle)

end most_stable_shape_l365_365918


namespace goods_train_pass_time_l365_365485

theorem goods_train_pass_time 
  (speed_mans_train_kmph : ℝ) (speed_goods_train_kmph : ℝ) (length_goods_train_m : ℝ) :
  speed_mans_train_kmph = 20 → 
  speed_goods_train_kmph = 92 → 
  length_goods_train_m = 280 → 
  abs ((length_goods_train_m / ((speed_mans_train_kmph + speed_goods_train_kmph) * 1000 / 3600)) - 8.99) < 0.01 :=
by
  sorry

end goods_train_pass_time_l365_365485


namespace discount_percentage_l365_365833

theorem discount_percentage (original_price sale_price : ℕ) (h₁ : original_price = 1200) (h₂ : sale_price = 1020) : 
  ((original_price - sale_price) * 100 / original_price : ℝ) = 15 :=
by
  sorry

end discount_percentage_l365_365833


namespace find_xyz_l365_365152

theorem find_xyz (x y z : ℤ) 
  (h1 : y = 3) 
  (h2 : y^4 + 2 * z^2 ≡ 2 [MOD 3])
  (h3 : x = 5) 
  (h4 : 3 * x^4 + z^2 ≡ 1 [MOD 5]) : 
  (x, y, z) = (5, 3, 19) := 
begin
  sorry
end

end find_xyz_l365_365152


namespace f_100_eq_11_l365_365741
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

def f (n : ℕ) : ℕ := sum_of_digits (n^2 + 1)

def f_iter : ℕ → ℕ → ℕ
| 0,    n => f n
| k+1,  n => f (f_iter k n)

theorem f_100_eq_11 (n : ℕ) (h : n = 1990) : f_iter 100 n = 11 := by
  sorry

end f_100_eq_11_l365_365741


namespace count_pairs_l365_365237

theorem count_pairs : {p : ℤ × ℝ // (∃ (x y : ℝ), p = (x, y) ∧ (x * x / 4 + y * y / 3 = 1) ∧ x ∈ Int.filter (λ x, 0 ≤ x))}.size = 5 :=
sorry

end count_pairs_l365_365237


namespace number_of_subsets_of_A_l365_365650

-- Definition of set A based on the given condition
def A : Set ℕ := {x | -1 ≤ x ∧ x < 4 ∧ x ∈ ℕ}

-- Problem Statement: Prove that the number of subsets of set A is 16
theorem number_of_subsets_of_A : (Set.powerset A).card = 16 := sorry

end number_of_subsets_of_A_l365_365650


namespace cylinder_surface_area_is_128pi_l365_365889

noncomputable def cylinder_total_surface_area (h r : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

theorem cylinder_surface_area_is_128pi :
  cylinder_total_surface_area 12 4 = 128 * Real.pi :=
by
  sorry

end cylinder_surface_area_is_128pi_l365_365889


namespace number_of_workers_l365_365298

theorem number_of_workers 
  (N : ℕ)
  (h1 : N = 2 + 6)
  (h2 : (2 : ℝ) = 2)
  (h3 : (fact N / (fact 2 * fact (N - 2))) = 28)
  (h4 : (1 / ((fact N) / (fact 2 * fact (N - 2): ℝ))) = 0.03571428571428571) :
  N = 8 :=
by
  sorry

end number_of_workers_l365_365298


namespace six_nickels_around_one_l365_365893

theorem six_nickels_around_one :
  ∃ n, (∀ (C : set (euclidean_affine_space ℝ 2)), 
          (∀ x ∈ C, is_circle x) ∧ 
          (∀ x ∈ C, ∃! y ∈ C, tangent x y) ∧ 
          (∀ x ∈ C, ∃! y z ∈ C, y ≠ z → tangent x y ∧ tangent x z ∧ tangent y z) ∧
          (∃ c ∈ C, ∀ x ∈ C, x ≠ c → tangent c x)) → n = 6 := 
by 
  sorry

end six_nickels_around_one_l365_365893


namespace fruit_basket_problem_l365_365288

noncomputable def num_oranges : ℕ := 8
noncomputable def num_apples : ℕ := 10
noncomputable def num_bananas : ℕ := 10
noncomputable def total_fruits : ℕ := num_oranges + num_apples + num_bananas

theorem fruit_basket_problem :
  (total_fruits = 28) ∧
  (num_oranges = 8) ∧
  (num_apples = 10) ∧
  (num_bananas = 10) ∧
  (5 ≤ num_oranges) ∧
  (3 ≤ num_apples) ∧
  (num_oranges - 5 = 3) ∧
  (num_apples - 3 = 7) ∧
  ((num_oranges.to_rat / total_fruits.to_rat) = 2 / 7) :=
by
  sorry

end fruit_basket_problem_l365_365288


namespace sufficient_but_not_necessary_l365_365620

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 / x > 1) : x < 1 := by
  sorry

end sufficient_but_not_necessary_l365_365620


namespace bacon_suggestion_count_l365_365003

theorem bacon_suggestion_count (B : ℕ) (h1 : 408 = B + 366) : B = 42 :=
by
  sorry

end bacon_suggestion_count_l365_365003


namespace ladder_rungs_count_l365_365482

theorem ladder_rungs_count :
  ∃ (n : ℕ), ∀ (start mid : ℕ),
    start = n / 2 →
    mid = ((start + 5 - 7) + 8 + 7) →
    mid = n →
    n = 27 :=
by
  sorry

end ladder_rungs_count_l365_365482


namespace lattice_paths_properties_l365_365808

-- Define the function f(n) for the number of walks.
def f : ℕ → ℕ 
| 1 := 4
| 2 := 12
| 3 := 36
| 4 := 100
| _ := 0 -- For simplicity, define 0 for n > 4

-- Properties about the function f.

-- The specific values for n = 1, 2, 3, 4.
def property_values : Prop :=
  f 1 = 4 ∧ f 2 = 12 ∧ f 3 = 36 ∧ f 4 = 100

-- The general inequality property for any n.
def property_inequalities (n : ℕ) : Prop :=
  2^n < f n ∧ f n ≤ 4 * 3^(n-1)

-- The main theorem combining both properties
theorem lattice_paths_properties (n : ℕ) : 
  (property_values) ∧ (∀ n, property_inequalities n) :=
by
  sorry

end lattice_paths_properties_l365_365808


namespace evaluate_expression_l365_365163

def ceil (x : ℚ) : ℤ := sorry -- Implement the ceiling function for rational numbers as needed

theorem evaluate_expression :
  (ceil ((23 : ℚ) / 9 - ceil ((35 : ℚ) / 23))) 
  / (ceil ((35 : ℚ) / 9 + ceil ((9 * 23 : ℚ) / 35))) = (1 / 10 : ℚ) :=
by
  intros
  -- Proof goes here
  sorry

end evaluate_expression_l365_365163


namespace total_fruit_count_l365_365044

theorem total_fruit_count :
  let crates := 25
  let oranges_per_crate := 270
  let boxes := 38
  let nectarines_per_box := 50
  let baskets := 15
  let apples_per_basket := 90
  let total_oranges := crates * oranges_per_crate
  let total_nectarines := boxes * nectarines_per_box
  let total_apples := baskets * apples_per_basket
  total_oranges + total_nectarines + total_apples = 10000
:= by
  let crates := 25
  let oranges_per_crate := 270
  let boxes := 38
  let nectarines_per_box := 50
  let baskets := 15
  let apples_per_basket := 90
  let total_oranges := crates * oranges_per_crate
  let total_nectarines := boxes * nectarines_per_box
  let total_apples := baskets * apples_per_basket
  have h1 : total_oranges = crates * oranges_per_crate := rfl
  have h2 : total_nectarines = boxes * nectarines_per_box := rfl
  have h3 : total_apples = baskets * apples_per_basket := rfl
  have h4 : total_oranges + total_nectarines + total_apples = 6750 + 1900 + 1350 := by
{ rw [h1, h2, h3] }
  have h5 : 6750 + 1900 + 1350 = 10000 := by norm_num
  exact eq.trans h4 h5

end total_fruit_count_l365_365044


namespace seven_pow_mod_hundred_l365_365847

theorem seven_pow_mod_hundred :
  ∃ e : ℕ, e = 4 ∧ 7^e % 100 = 1 :=
by {
  use 4,
  split,
  { refl, },
  { norm_num, },
  sorry
}

end seven_pow_mod_hundred_l365_365847


namespace general_term_a_find_k_l365_365201

noncomputable theory

def S (n : ℕ) : ℤ := n * n - 9 * n

def a (n : ℕ) : ℤ := 2 * n - 10

theorem general_term_a (n : ℕ) : a n = 2 * n - 10 := 
by 
  sorry

theorem find_k (k : ℕ) (hk1 : 5 < a k) (hk2 : a k < 8) : k = 8 :=
by
  sorry

end general_term_a_find_k_l365_365201


namespace hyperbola_vertex_to_asymptote_distance_focus_of_parabola_distance_to_asymptote_of_hyperbola_l365_365787

noncomputable def parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) : ℝ × ℝ :=
(1 / (4 * a), 0)

theorem hyperbola_vertex_to_asymptote_distance : ℝ :=
  (real.sqrt 30) / 5

theorem focus_of_parabola (a : ℝ) (h : a ≠ 0) :
  parabola_focus_coordinates a h = (1 / (4 * a), 0) :=
sorry

theorem distance_to_asymptote_of_hyperbola :
  hyperbola_vertex_to_asymptote_distance = (real.sqrt 30) / 5 :=
sorry

end hyperbola_vertex_to_asymptote_distance_focus_of_parabola_distance_to_asymptote_of_hyperbola_l365_365787


namespace triple_solution_exists_and_unique_l365_365965

theorem triple_solution_exists_and_unique:
  ∀ (x y z : ℝ), (1 + x^4 ≤ 2 * (y - z) ^ 2) ∧ (1 + y^4 ≤ 2 * (z - x) ^ 2) ∧ (1 + z^4 ≤ 2 * (x - y) ^ 2)
  → (x = 1 ∧ y = 0 ∧ z = -1) :=
by
  sorry

end triple_solution_exists_and_unique_l365_365965


namespace halves_left_l365_365659

theorem halves_left (whole_cookies : ℕ) (greg_halves : ℕ) (brad_halves : ℕ) (total_halves_made : whole_cookies * 2):
  (total_halves_made - (greg_halves + brad_halves) = 18) :=
by
  assume (whole_cookies : ℕ) (greg_halves : ℕ) (brad_halves : ℕ)
  assume total_halves_made : whole_cookies * 2
  have total_halves_made = 14 * 2
  have greg_halves = 4
  have brad_halves = 6
  show (total_halves_made - (greg_halves + brad_halves)) = 18, from sorry

end halves_left_l365_365659


namespace passing_probability_l365_365278

def probability_of_passing (p : ℝ) : ℝ :=
  p^3 + p^2 * (1 - p) + (1 - p) * p^2

theorem passing_probability :
  probability_of_passing 0.6 = 0.504 :=
by {
  sorry
}

end passing_probability_l365_365278


namespace least_possible_n_l365_365105

theorem least_possible_n (N : ℕ) (hN : ∀ k ∈ (Finset.range 28).image (λ x, x + 1), k ∣ N) (h_inc_29 : ¬ (29 ∣ N)) (h_inc_30 : ¬ (30 ∣ N)) : 80313433200 ≤ N := 
sorry

end least_possible_n_l365_365105


namespace trajectory_circle_l365_365695

-- Define the problem conditions
variables {A B O : EuclideanSpace ℝ (fin 2)} (lambda mu : ℝ)
def angle_AOB : real.angle := 60 -- 60 degrees angle in radians
def OA : EuclideanSpace ℝ (fin 2) := @euclideanSpace.mk ℝ _ _ [(1/2), (math.sqrt 3 / 2)]
def OB : EuclideanSpace ℝ (fin 2) := @euclideanSpace.mk ℝ _ _ [(1), (0)]
def OC : EuclideanSpace ℝ (fin 2) := (λ λ mu, λ • OA + mu • OB)

-- Main theorem to prove the trajectory is a circle
theorem trajectory_circle 
  (h1 : ∥OA∥ = 1) 
  (h2 : ∥OB∥ = 1)
  (h3 : ∠ O A B = angle_AOB)
  (h4 : λ^2 + λ * μ + μ^2 = 1) : 
  ∥OC∥ = 1 :=
sorry

end trajectory_circle_l365_365695


namespace conjugate_complex_number_integral_abs_value_l365_365936

-- Problem 1: Conjugate of a complex number
theorem conjugate_complex_number (z : ℂ) (h : z = 1 / (1 - complex.i)) : conj z = 1/2 - 1/2 * complex.i :=
by sorry

-- Problem 2: Definite integral of absolute value function
theorem integral_abs_value : ∫ x in 0..2, |1 - x| = 1 :=
by sorry

end conjugate_complex_number_integral_abs_value_l365_365936


namespace ceil_inequality_range_x_solve_eq_l365_365728

-- Definition of the mathematical ceiling function to comply with the condition a).
def ceil (a : ℚ) : ℤ := ⌈a⌉

-- Condition 1: Relationship between m and ⌈m⌉.
theorem ceil_inequality (m : ℚ) : m ≤ ceil m ∧ ceil m < m + 1 :=
sorry

-- Part 2.1: Range of x given {3x + 2} = 8.
theorem range_x (x : ℚ) (h : ceil (3 * x + 2) = 8) : 5 / 3 < x ∧ x ≤ 2 :=
sorry

-- Part 2.2: Solving {3x - 2} = 2x + 1/2
theorem solve_eq (x : ℚ) (h : ceil (3 * x - 2) = 2 * x + 1 / 2) : x = 7 / 4 ∨ x = 9 / 4 :=
sorry

end ceil_inequality_range_x_solve_eq_l365_365728


namespace ratio_unit_price_l365_365927

variables (v p : ℝ)

def unit_price_brand_x := (0.7 * p) / (1.2 * v)
def unit_price_brand_y := p / v

theorem ratio_unit_price (v_pos : v > 0) (p_pos : p > 0) : 
  (unit_price_brand_x v p) / (unit_price_brand_y v p) = 7 / 12 :=
by
  unfold unit_price_brand_x unit_price_brand_y
  sorry

end ratio_unit_price_l365_365927


namespace problem_1_problem_2_problem_3_l365_365651

-- Conditions
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x < m}
def U (m : ℝ) : Set ℝ := A ∪ B m

-- Proof statements
theorem problem_1 : A ∩ (U 3ᶜ B 3) = {x | 3 ≤ x ∧ x < 4} := sorry
theorem problem_2 : ∀ m, A ∩ B m = ∅ ↔ m ≤ -2 := sorry
theorem problem_3 : ∀ m, A ∩ B m = A ↔ m ≥ 4 := sorry

end problem_1_problem_2_problem_3_l365_365651


namespace prob_both_successful_prob_at_least_one_successful_l365_365100

variables (P_A P_B : ℚ)
variables (h1 : P_A = 1 / 2)
variables (h2 : P_B = 2 / 5)

/-- Prove that the probability that both A and B score in one shot each is 1 / 5. -/
theorem prob_both_successful (P_A P_B : ℚ) (h1 : P_A = 1 / 2) (h2 : P_B = 2 / 5) :
  P_A * P_B = 1 / 5 :=
by sorry

variables (P_A_miss P_B_miss : ℚ)
variables (h3 : P_A_miss = 1 / 2)
variables (h4 : P_B_miss = 3 / 5)

/-- Prove that the probability that at least one shot is successful is 7 / 10. -/
theorem prob_at_least_one_successful (P_A_miss P_B_miss : ℚ) (h3 : P_A_miss = 1 / 2) (h4 : P_B_miss = 3 / 5) :
  1 - P_A_miss * P_B_miss = 7 / 10 :=
by sorry

end prob_both_successful_prob_at_least_one_successful_l365_365100


namespace percentage_of_children_prefer_corn_l365_365095

theorem percentage_of_children_prefer_corn
    (kids_prefer_peas : ℕ)
    (kids_prefer_carrots : ℕ)
    (kids_prefer_corn : ℕ)
    (total_kids : ℕ)
    (h1 : kids_prefer_peas = 6)
    (h2 : kids_prefer_carrots = 9)
    (h3 : kids_prefer_corn = 5)
    (h4 : total_kids = kids_prefer_peas + kids_prefer_carrots + kids_prefer_corn) :
    (kids_prefer_corn.to_rat / total_kids.to_rat) * 100 = 25 :=
by
  sorry

end percentage_of_children_prefer_corn_l365_365095


namespace find_number_l365_365185

theorem find_number (x : ℝ) (h : x = 12) : ( ( 17.28 / x ) / ( 3.6 * 0.2 ) ) = 2 := 
by
  -- Proof will be here
  sorry

end find_number_l365_365185


namespace dinners_alone_l365_365666

theorem dinners_alone :
  ∃ m n o (d : ℕ),
  (m = 6) ∧ 
  (n = 14) ∧
  (d = 1) ∧
  ((∀ k ∈ finset.range 6, k ≠ 6 → ∃ s ∈ finset.powerset_len k (finset.range 6), s.card * (7 - k) * d = n - 1 * m ∧  s.card * (7 + k) * d = 7) ∧
  d = n - (m + 6)) ∧ d = 1 :=
begin
  sorry
end

end dinners_alone_l365_365666


namespace geometric_seq_a4_a7_l365_365694

variable {a : ℕ → ℝ}

def is_geometric (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

theorem geometric_seq_a4_a7
  (h_geom : is_geometric a)
  (h_roots : ∃ a_1 a_10 : ℝ, (a 1 = a_1 ∧ a 10 = a_10) ∧ (2 * a_1 ^ 2 + 5 * a_1 + 1 = 0) ∧ (2 * a_10 ^ 2 + 5 * a_10 + 1 = 0)):
  a 4 * a 7 = 1 / 2 :=
by
  sorry

end geometric_seq_a4_a7_l365_365694


namespace cardinality_of_M_is_zero_l365_365599

noncomputable def M := {x : ℝ | x^4 + 4 * x^2 - 12 * x + 8 = 0 ∧ 0 < x}

theorem cardinality_of_M_is_zero : Fintype.card M = 0 :=
sorry

end cardinality_of_M_is_zero_l365_365599


namespace amount_saved_l365_365156

theorem amount_saved (list_price : ℝ) (tech_deals_discount : ℝ) (electro_bargains_discount : ℝ)
    (tech_deals_price : ℝ) (electro_bargains_price : ℝ) (amount_saved : ℝ) :
  tech_deals_discount = 0.15 →
  list_price = 120 →
  tech_deals_price = list_price * (1 - tech_deals_discount) →
  electro_bargains_discount = 20 →
  electro_bargains_price = list_price - electro_bargains_discount →
  amount_saved = tech_deals_price - electro_bargains_price →
  amount_saved = 2 :=
by
  -- proof steps would go here
  sorry

end amount_saved_l365_365156


namespace monomial_2023_l365_365491

def monomial (n : ℕ) : ℤ × ℕ :=
  ((-1)^n * (n + 1), n)

theorem monomial_2023 :
  monomial 2023 = (-2024, 2023) :=
by
  sorry

end monomial_2023_l365_365491


namespace num_ways_to_tile_3x5_is_40_l365_365938

-- Definition of the problem
def numTilings (tiles : List (ℕ × ℕ)) (m n : ℕ) : ℕ :=
  sorry -- Placeholder for actual tiling computation

-- Condition specific to this problem
def specificTiles : List (ℕ × ℕ) :=
  [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]

-- Problem statement in Lean 4
theorem num_ways_to_tile_3x5_is_40 :
  numTilings specificTiles 3 5 = 40 :=
sorry

end num_ways_to_tile_3x5_is_40_l365_365938


namespace initial_plants_count_l365_365412

theorem initial_plants_count (p : ℕ) 
    (h1 : p - 20 > 0)
    (h2 : (p - 20) / 2 > 0)
    (h3 : ((p - 20) / 2) - 1 > 0)
    (h4 : ((p - 20) / 2) - 1 = 4) : 
    p = 30 :=
by
  sorry

end initial_plants_count_l365_365412


namespace original_number_satisfies_equation_l365_365858

theorem original_number_satisfies_equation 
  (x : ℝ) 
  (h : 1000 * x = 9 * (1 / x)) : 
  x = 3 * (√10 / 100) :=
by
  sorry

end original_number_satisfies_equation_l365_365858


namespace sum_2018_l365_365624

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := (x - 1) ^ α

def a_n (n : ℕ) (f : ℝ → ℝ) : ℝ := 1 / (f (n + 1) + f n)

noncomputable def S_n (n : ℕ) (f : ℝ → ℝ) : ℝ :=
  ∑ i in Finset.range n, a_n (i + 1) f

theorem sum_2018 (α : ℝ) (h : 3 = 9 ^ α) :
  S_n 2018 (λ x, (x - 1) ^ (1/2)) = Real.sqrt 2018 :=
sorry

end sum_2018_l365_365624


namespace find_hours_l365_365830

theorem find_hours (x : ℕ) (h : (14 + 10 + 13 + 9 + 12 + 11 + x) / 7 = 12) : x = 15 :=
by
  -- The proof is omitted
  sorry

end find_hours_l365_365830


namespace KL_perpendicular_BC_l365_365997

-- Definitions and conditions
variables {A B C D P K L : Type}
variables [cyclic_quadrilateral A B C D]
variables (hP: on_side P B C)
variables (hPAB: ∠ PAB = 90) (hPDC: ∠ PDC = 90)
variables (medianK_from_A : medians_intersection A P B K) (medianK_from_D : medians_intersection D P C K)
variables (bisectorL_from_PAB : angle_bisector_intersection P A B L) (bisectorL_from_PDC : angle_bisector_intersection P D C L)

-- Theorem statement
theorem KL_perpendicular_BC : ⊥(K, L) (B, C) :=
by
  sorry

end KL_perpendicular_BC_l365_365997


namespace rhombus_perimeter_l365_365791

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 24) (h2 : d2 = 10) : 4 * (Int.sqrt (d1/2 * d1/2 + d2/2 * d2/2)) = 52 :=
by {
  rw [h1, h2],
  have h_diag1 : (24 / 2 : ℤ) = 12 := by norm_num,
  have h_diag2 : (10 / 2 : ℤ) = 5 := by norm_num,
  have h_sq : (12 * 12 + 5 * 5 : ℤ) = 169 := by norm_num,
  have h_sqrt : Int.sqrt 169 = 13 := by norm_num [Int.sqrt_eq],
  rw [← h_sqrt, h_sq],
  norm_num
}

end rhombus_perimeter_l365_365791


namespace oakwood_math_team_l365_365392

theorem oakwood_math_team : ∃ (n : ℕ), n = 4.choose 3 * 6.choose 2 ∧ n = 60 :=
by
  use 4.choose 3 * 6.choose 2
  have h1 : 4.choose 3 = 4 := by simp [Nat.choose]
  have h2 : 6.choose 2 = 15 := by simp [Nat.choose]
  simp [h1, h2]
  sorry

end oakwood_math_team_l365_365392


namespace simplify_fraction_l365_365371

noncomputable def expr_simplify := 
  let sqrt48 := sqrt (48 : ℝ) 
  let sqrt75 := sqrt (75 : ℝ) 
  let sqrt27 := sqrt (27 : ℝ) 
  let term1 := (4:ℝ) * sqrt (3 : ℝ)
  let term2 := (15:ℝ) * sqrt (3 : ℝ)
  let term3 := (15:ℝ) * sqrt (3 : ℝ)
  let denom := term1 + term2 + term3
  let frac := (5:ℝ) / denom
  let result := (5:ℝ) * sqrt (3 : ℝ) / (102 : ℝ)
  frac = result

theorem simplify_fraction : expr_simplify = true := by
  sorry

end simplify_fraction_l365_365371


namespace min_value_expression_l365_365263

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (H : 1 / a + 1 / b = 1) :
  ∃ c : ℝ, (∀ (a b : ℝ), 0 < a → 0 < b → 1 / a + 1 / b = 1 → c ≤ 4 / (a - 1) + 9 / (b - 1)) ∧ (c = 6) :=
by
  sorry

end min_value_expression_l365_365263


namespace possible_values_for_ceil_x_sq_l365_365257

theorem possible_values_for_ceil_x_sq (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ S : set ℤ, (∀ n ∈ S, 122 ≤ n ∧ n ≤ 143) ∧ S.card = 22 :=
by
  let S := { n : ℤ | 122 ≤ n ∧ n ≤ 143 }
  use S
  split
  · intros n hn
    exact hn
  · have h1 : S.card = 143 - 122 + 1 := by sorry
    exact h1

end possible_values_for_ceil_x_sq_l365_365257


namespace find_integer_x_l365_365131

theorem find_integer_x (x : ℤ) :
  1 < x ∧ x < 9 ∧ 
  2 < x ∧ x < 15 ∧ 
  0 < x ∧ x < 7 ∧ 
  0 < x ∧ x < 4 ∧ 
  x + 1 < 5 
  → x = 3 :=
by
  intros h
  sorry

end find_integer_x_l365_365131


namespace red_pairs_count_l365_365276

theorem red_pairs_count (blue_shirts red_shirts total_pairs blue_blue_pairs : ℕ)
  (h1 : blue_shirts = 63) 
  (h2 : red_shirts = 81) 
  (h3 : total_pairs = 72) 
  (h4 : blue_blue_pairs = 21)
  : (red_shirts - (blue_shirts - blue_blue_pairs * 2)) / 2 = 30 :=
by
  sorry

end red_pairs_count_l365_365276


namespace area_of_triangle_AFD_l365_365208

-- Define the problem conditions
structure Rectangle :=
(A B C D F : Type)
(side_AB : ℝ)
(side_BC : ℝ)
(perpendicular_F : ∀ F, F ∈ segment (B, AC) ∧ is_perpendicular B F AC)

-- Define the Lean statement to be proved
theorem area_of_triangle_AFD:
  ∀ (r : Rectangle),
    r.side_AB = 5 → r.side_BC = 12 →
    ∃ (area_AFD : ℝ), area_AFD = 20 :=
begin
  -- Conditions
  intros r h_AB h_BC,
  -- Prove the area
  sorry
end

end area_of_triangle_AFD_l365_365208


namespace min_value_proof_l365_365738

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  9 / a + 16 / b + 25 / (c ^ 2)

theorem min_value_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 5) :
  minimum_value a b c ≥ 50 :=
sorry

end min_value_proof_l365_365738


namespace number_of_valid_rectangles_l365_365652

open Real EuclideanGeometry

axiom Point : Type
variable (P1 P2 P3 : Point)
variable [EuclideanGeometry Point]
variable h_distinct : P1 ≠ P2 ∧ P2 ≠ P3 ∧ P1 ≠ P3

-- Define a rectangle
structure Rectangle (A B C D : Point) : Prop :=
  (angles : {angle_AB := (∠ A B C), angle_BC := (∠ B C D), 
             angle_CD := (∠ C D A), angle_DA := (∠ D A B)}
   (angles_proper : angle_AB = 90 ∧ angle_BC = 90 ∧ angle_CD = 90 ∧ angle_DA = 90))

-- Define the problem: constructing rectangular solution
def valid_rectangles (A B C D : Point) : Prop :=
  Rectangle A B C D ∧ (A = P1 ∨ A = P2 ∨ A = P3) ∧
  (B ∈ line_through A P2 ∨ B ∈ line_through A P3) ∧
  (C ∈ line_through A P2 ∨ C ∈ line_through A P3) ∧
  ((∠ A C B = 30 ∨ ∠ A C B = 60) ∨ (∠ A C B = 120 ∨ ∠ A C B = 150))

theorem number_of_valid_rectangles : 
  ∃ num_rectangles : ℕ, num_rectangles = 60 :=
  sorry

end number_of_valid_rectangles_l365_365652


namespace ladder_distance_from_wall_l365_365097

theorem ladder_distance_from_wall (h a b : ℕ) (h_hyp : h = 13) (h_wall : a = 12) :
  a^2 + b^2 = h^2 → b = 5 :=
by
  intros h_eq
  sorry

end ladder_distance_from_wall_l365_365097


namespace number_used_twice_l365_365170

-- Define the grid and the constraints
def magic_constant : ℕ := 15

def is_magic_square (square : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  (∀ i : Fin 3, (∑ j, square i j) = magic_constant) ∧
  (∀ j : Fin 3, (∑ i, square i j) = magic_constant) ∧
  (∑ i, square i i) = magic_constant ∧
  (∑ i, square i (2 - i)) = magic_constant ∧
  (∀ i j, 1 ≤ square i j ∧ square i j ≤ 9)

def partially_filled : Matrix (Fin 3) (Fin 3) (Option ℕ) :=
  λ i j, match (i, j) with
         | (0, 0) => some 8
         | (1, 1) => some 4
         | _      => none

def fill_partial (partial : Matrix (Fin 3) (Fin 3) (Option ℕ)) : Matrix (Fin 3) (Fin 3) ℕ :=
  λ i j, partial i j.getD 0

theorem number_used_twice : ∃ (square : Matrix (Fin 3) (Fin 3) ℕ), 
  is_magic_square square ∧ (fill_partial partially_filled = square) ∧ (∃! x, ∀ i j, square i j = x → x = 8) := sorry

end number_used_twice_l365_365170


namespace delivery_cost_l365_365473

variable (x : ℝ) (h : x > 2)

def base_charge : ℝ := 10
def additional_charge_per_kg : ℝ := 2

def cost (x : ℝ) : ℝ := base_charge + additional_charge_per_kg * (x - 2)

theorem delivery_cost (x : ℝ) (h : x > 2) : cost x = 2 * x + 6 :=
by
  unfold cost base_charge additional_charge_per_kg
  rw [← add_assoc, add_comm 10, add_assoc, ← sub_add, sub_self, zero_add]
  sorry

end delivery_cost_l365_365473


namespace profit_calculation_l365_365367

def totalProfit (totalMoney part1 interest1 interest2 time : ℕ) : ℕ :=
  let part2 := totalMoney - part1
  let interestFromPart1 := part1 * interest1 / 100 * time
  let interestFromPart2 := part2 * interest2 / 100 * time
  interestFromPart1 + interestFromPart2

theorem profit_calculation : 
  totalProfit 80000 70000 10 20 1 = 9000 :=
  by 
    -- Rather than providing a full proof, we insert 'sorry' as per the instruction.
    sorry

end profit_calculation_l365_365367


namespace length_of_rectangle_l365_365036

theorem length_of_rectangle (L : ℝ) (W : ℝ) (A_triangle : ℝ) (hW : W = 4) (hA_triangle : A_triangle = 60)
  (hRatio : (L * W) / A_triangle = 2 / 5) : L = 6 :=
by
  sorry

end length_of_rectangle_l365_365036


namespace range_of_f_solution_set_of_f_lt_1_l365_365642

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2 else x ^ 2

theorem range_of_f : ∀ y ∈ set.range f, y < 4 :=
sorry

theorem solution_set_of_f_lt_1 : {x : ℝ | f x < 1} = (set.Iio (-1)) ∪ (set.Ioo (-1) 1) :=
sorry

end range_of_f_solution_set_of_f_lt_1_l365_365642


namespace cylinder_new_volume_l365_365818

-- Define the original volume condition
def original_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Define the new volume condition
def new_volume (r h : ℝ) : ℝ := Real.pi * (3 * r)^2 * (2 * h)

-- Theorem statement to prove the new volume is 360 cubic feet
theorem cylinder_new_volume (r h : ℝ) (h1 : original_volume r h = 20) : new_volume r h = 360 :=
sorry

end cylinder_new_volume_l365_365818


namespace find_lowest_score_l365_365383

-- Definitions based on the conditions
variable (scores : Fin 15 → ℝ)
variable (mean15 : ℝ) (mean13 : ℝ) (maxScore : ℝ)

-- Assuming the conditions as hypotheses
axiom mean15_is_85 : mean15 = 85
axiom mean13_is_88 : mean13 = 88
axiom maxScore_is_99 : maxScore = 99 

-- Define the sum of scores
def totalSum : ℝ := ∑ i, scores i

-- Define the sum of the remaining 13 scores after removing the highest and lowest scores
def sumOfRemaining (removed_high removed_low : ℝ) : ℝ := totalSum - removed_high - removed_low

-- Calculate the mean of 15 scores
axiom mean15_def : totalSum / 15 = mean15

-- Calculate the mean of the remaining 13 scores
axiom mean13_def (removed_high removed_low : ℝ) : sumOfRemaining removed_high removed_low / 13 = mean13

-- The main theorem that needs to be proved
theorem find_lowest_score : ∃ (lowest: ℝ), mean15 = 85 → mean13 = 88 → maxScore = 99 → totalSum / 15 = 85 → 
  (∑ i, scores i) - (totalSum - (maxScore + lowest)) - maxScore = 131 → lowest = 32 := by 
  sorry

end find_lowest_score_l365_365383


namespace jerry_total_income_l365_365715

-- Define the base charge for piercing noses and the additional rate for ears
def noseCharge := 20
def earExtraRate := 0.5
def earCharge := noseCharge + (noseCharge * earExtraRate)

-- Define the number of piercings for noses and ears
def numNoses := 6
def numEars := 9

-- Calculate the total money Jerry makes and state the theorem
def totalMoney := (numNoses * noseCharge) + (numEars * earCharge)

theorem jerry_total_income : totalMoney = 390 := by
  sorry

end jerry_total_income_l365_365715


namespace simplify_expression_l365_365000

def expression1 (x : ℝ) : ℝ :=
  3 * x^3 + 4 * x^2 + 2 * x + 5 - (2 * x^3 - 5 * x^2 + x - 3) + (x^3 - 2 * x^2 - 4 * x + 6)

def expression2 (x : ℝ) : ℝ :=
  2 * x^3 + 7 * x^2 - 3 * x + 14

theorem simplify_expression (x : ℝ) : expression1 x = expression2 x :=
by 
  sorry

end simplify_expression_l365_365000


namespace odd_square_minus_one_div_by_eight_l365_365266

theorem odd_square_minus_one_div_by_eight (n : ℤ) : ∃ k : ℤ, (2 * n + 1) ^ 2 - 1 = 8 * k :=
by
  sorry

end odd_square_minus_one_div_by_eight_l365_365266


namespace distance_between_trees_correct_l365_365272

def yard_length : ℝ := 320
def number_of_trees : ℕ := 47

def distance_between_trees (L : ℝ) (T : ℕ) : ℝ :=
  L / (T - 1)

theorem distance_between_trees_correct :
  distance_between_trees yard_length number_of_trees = 320 / (47 - 1) :=
by
  repeat {sorry}

end distance_between_trees_correct_l365_365272
