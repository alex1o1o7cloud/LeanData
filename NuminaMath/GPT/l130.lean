import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.Group
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Coloring
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.NumberTheory.GCD
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Conditional
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith

namespace least_possible_value_of_a_plus_b_plus_c_l130_130589

noncomputable def least_possible_abc_sum : ℕ :=
  let a := 3
  let b := 3
  let c := 4
  a + b + c

theorem least_possible_value_of_a_plus_b_plus_c :
  ∃ h : ℕ, (225 ∣ h) ∧ (216 ∣ h) ∧ (∃ a b c : ℕ, h = (2^a) * (3^b) * (5^c) ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ least_possible_abc_sum = 10) :=
begin
  sorry
end

end least_possible_value_of_a_plus_b_plus_c_l130_130589


namespace composite_prob_from_50_nat_numbers_l130_130029

theorem composite_prob_from_50_nat_numbers : 
  (∃ n, 1 ≤ n ∧ n ≤ 50 ∧ ∑ i in finset.range 50, if (¬is_prime (i + 1) ∧ (i + 1) ≠ 1) then 1 else 0 = 34 ∧ (34: ℝ) / 50 = 0.68) :=
by {
  sorry,
}

end composite_prob_from_50_nat_numbers_l130_130029


namespace divide_by_repeating_decimal_l130_130177

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l130_130177


namespace eight_div_repeating_three_l130_130156

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l130_130156


namespace compare_powers_l130_130738

theorem compare_powers (x : ℝ) (m n : ℝ) (hx_pos : x > 0) (hx_ne_one : x ≠ 1) (hmn : m > n > 0) : 
  x^m + x^(-m) > x^n + x^(-n) :=
sorry

end compare_powers_l130_130738


namespace probability_longer_piece_l130_130646

theorem probability_longer_piece {x y : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) :
  (∃ (p : ℝ), p = 2 / (x * y + 1)) :=
by
  sorry

end probability_longer_piece_l130_130646


namespace keith_missed_games_l130_130989

-- Define the total number of football games
def total_games : ℕ := 8

-- Define the number of games Keith attended
def attended_games : ℕ := 4

-- Define the number of games played at night (although it is not directly necessary for the proof)
def night_games : ℕ := 4

-- Define the number of games Keith missed
def missed_games : ℕ := total_games - attended_games

-- Prove that the number of games Keith missed is 4
theorem keith_missed_games : missed_games = 4 := by
  sorry

end keith_missed_games_l130_130989


namespace division_of_repeating_decimal_l130_130139

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l130_130139


namespace divide_by_repeating_decimal_l130_130170

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l130_130170


namespace find_z_l130_130850

-- Definitions from conditions
def x : ℕ := 22
def y : ℕ := 13
def total_boys_who_went_down_slide : ℕ := x + y
def ratio_slide_to_watch := 5 / 3

-- Statement we need to prove
theorem find_z : ∃ z : ℕ, (5 / 3 = total_boys_who_went_down_slide / z) ∧ z = 21 :=
by
  use 21
  sorry

end find_z_l130_130850


namespace solve_equation_l130_130005

noncomputable def solutions (x : ℝ) : Prop :=
  (∃ n: ℕ, x = (Real.pi / 2) + 2 * n * Real.pi) ∨ x = 10

theorem solve_equation (x : ℝ) : 
  (Real.sqrt (Real.sin x ^ 2 + (Real.log10 x) ^ 2 - 1) = Real.sin x + Real.log10 x - 1) 
  ↔ solutions x :=
by sorry

end solve_equation_l130_130005


namespace compare_a_b_l130_130690

noncomputable def a : ℝ := 2^0.6
noncomputable def b : ℝ := 0.6^2

theorem compare_a_b : a > b := 
by
  -- Sorry to skip the actual proof
  sorry

end compare_a_b_l130_130690


namespace min_S_n_T_n_formula_l130_130790

-- Definitions of sequence terms and their sums
def S (n : ℕ) : ℤ := 2 * (n : ℤ)^2 - 19 * (n : ℤ) + 1
def a (n : ℕ) : ℤ := 
  if n = 1 then S 1
  else S n - S (n - 1)

-- Proof statements
theorem min_S_n : ∃ n : ℕ, S n = -44 ∧ ∀ m : ℕ, S m ≥ -44 := 
  sorry

theorem T_n_formula (n : ℕ) : 
    (T n = 
      if 1 ≤ n ∧ n ≤ 5 then -2 * n^2 + 19 * n - 1
      else if n ≥ 6 then 2 * n^2 - 19 * n + 89) := 
  sorry

-- Definitions of T_n
def T (n : ℕ) : ℤ := 
  if 1 ≤ n ∧ n ≤ 5 then -2 * (n : ℤ)^2 + 19 * (n : ℤ) - 1
  else if n ≥ 6 then 2 * (n : ℤ)^2 - 19 * (n : ℤ) + 89
  else 0  -- This covers the case where n < 1 which typically should not be included in the domain

end min_S_n_T_n_formula_l130_130790


namespace largest_three_digit_number_l130_130721

def divisible_by_each_digit (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  ∀ d ∈ digits, d ≠ 0 ∧ n % d = 0

def sum_of_digits_divisible_by (n : ℕ) (k : ℕ) : Prop :=
  let sum := (n / 100) + ((n / 10) % 10) + (n % 10)
  sum % k = 0

theorem largest_three_digit_number : ∃ n : ℕ, n = 936 ∧
  n >= 100 ∧ n < 1000 ∧
  divisible_by_each_digit n ∧
  sum_of_digits_divisible_by n 6 :=
by
  -- Proof details are omitted
  sorry

end largest_three_digit_number_l130_130721


namespace polar_eq_of_line_segment_l130_130840

-- Define the polar coordinate equation of the line segment
theorem polar_eq_of_line_segment :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ y = 1 - x →
  ∃ (θ ρ : ℝ), (0 ≤ θ ∧ θ ≤ π / 2) ∧ (y = ρ * sin θ) ∧ (x = ρ * cos θ) ∧ (ρ = 1 / (cos θ + sin θ)) :=
by
  sorry

end polar_eq_of_line_segment_l130_130840


namespace harmonic_sum_is_integer_iff_l130_130330

theorem harmonic_sum_is_integer_iff (m n : ℤ) (h : m ≤ n) :
  ∑ k in Finset.range (n - m + 1), 1 / (m + k) ∈ ℤ ↔ (m = 1 ∧ n = 1) :=
by sorry

end harmonic_sum_is_integer_iff_l130_130330


namespace find_a_l130_130397

theorem find_a (a : ℝ) :
  (∃ x y : ℝ, x - 2 * y + 1 = 0 ∧ x + 3 * y - 1 = 0 ∧ ¬(∀ x y : ℝ, ax + 2 * y - 3 = 0)) →
  (∃ p q : ℝ, ax + 2 * q - 3 = 0 ∧ (a = -1 ∨ a = 2 / 3)) :=
by {
  sorry
}

end find_a_l130_130397


namespace correct_calculation_l130_130230

variable (a b : ℝ)

theorem correct_calculation : ((-a^2)^3 = -a^6) :=
by sorry

end correct_calculation_l130_130230


namespace geometric_sum_S5_l130_130968

-- Given conditions
variable (a_1 q : ℝ)
variable (a_n : ℕ → ℝ)
variable [IsGeometricSeq a_n a_1 q]

-- Definition of geometric sequence terms
def a_2 := a_1 * q
def a_3 := a_1 * q^2
def a_4 := a_1 * q^3
def a_5 := a_1 * q^4
def a_7 := a_1 * q^6

-- Condition (1): a_2 * a_5 = 2 * a_3
axiom cond1 : a_2 a_5 = 2 * a_3

-- Condition (2): Arithmetic mean of a_4 and 2 * a_7 is 5/4
axiom cond2 : (a_4 + 2 * a_7) / 2 = 5 / 4

-- Target value for the sum S_5
def S_5 := a_1 * (1 - q^5) / (1 - q)

-- Prove that given the conditions, S_5 = 31
theorem geometric_sum_S5 : S_5 a_1 q = 31 :=
by
  sorry

end geometric_sum_S5_l130_130968


namespace evaluate_i11_plus_i111_l130_130326

def i_pow : ℤ → ℂ
| n := match n % 4 with
  | 0 := 1
  | 1 := complex.I
  | 2 := -1
  | 3 := -complex.I
  | _ := 0  -- this case will never be hit, since n % 4 is between 0 and 3.

theorem evaluate_i11_plus_i111 : 
  i_pow 11 + i_pow 111 = -2 * complex.I :=
begin
  sorry
end

end evaluate_i11_plus_i111_l130_130326


namespace division_of_repeating_decimal_l130_130140

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l130_130140


namespace find_cubic_sum_l130_130735

theorem find_cubic_sum
  {a b : ℝ}
  (h1 : a^5 - a^4 * b - a^4 + a - b - 1 = 0)
  (h2 : 2 * a - 3 * b = 1) :
  a^3 + b^3 = 9 :=
by
  sorry

end find_cubic_sum_l130_130735


namespace exists_root_in_interval_l130_130529

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

theorem exists_root_in_interval : ∃ c ∈ set.Ioo 1 2, f c = 0 :=
by {
  have h0 : f 1 = -1 := by norm_num,
  have h2 : f 2 = 5 := by norm_num,
  -- Use Intermediate Value Theorem here (proof omitted)
  sorry
}

end exists_root_in_interval_l130_130529


namespace division_by_repeating_decimal_l130_130220

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l130_130220


namespace point_on_line_has_correct_m_l130_130769

theorem point_on_line_has_correct_m (m : ℝ) : (∃ y, y = 2 * 3 - 1 ∧ y = m) → m = 5 :=
by
  intro ⟨y, hy1, hy2⟩
  rw hy1 at hy2
  simp at hy2 
  exact hy2

end point_on_line_has_correct_m_l130_130769


namespace sum_of_alternating_sums_n_eq_7_l130_130343

open Finset

-- Define the alternating sum of a nonempty subset
section
variable {α : Type*} [LinearOrder α] [AddCommGroup α]

def alternatingSum (s : Finset α) : α :=
  let l := s.sort (· ≥ ·) in
  l.enum.foldl (λ acc ⟨i, a⟩, if i % 2 = 0 then acc + a else acc - a) 0
end

-- Proving the sum of all alternating sums for n = 7
theorem sum_of_alternating_sums_n_eq_7 : 
  let S : Finset ℕ := range 8 \ {0} in -- This gives us {1, 2, ..., 7}
  ∑ T in S.powerset, alternatingSum T = 448 :=
sorry

end sum_of_alternating_sums_n_eq_7_l130_130343


namespace values_of_neg_cos_2theta_l130_130456

variable {x y r : ℝ}
variable {θ : ℝ}
def s := Real.sin θ
def c := Real.cos θ

theorem values_of_neg_cos_2theta (h_circle : x^2 + y^2 = r^2) :
  -1 ≤ s^2 - c^2 ∧ s^2 - c^2 ≤ 1 :=
sorry

end values_of_neg_cos_2theta_l130_130456


namespace combined_mpg_l130_130912

def ray_mpg := 50
def tom_mpg := 20
def ray_miles := 100
def tom_miles := 200

theorem combined_mpg : 
  let ray_gallons := ray_miles / ray_mpg
  let tom_gallons := tom_miles / tom_mpg
  let total_gallons := ray_gallons + tom_gallons
  let total_miles := ray_miles + tom_miles
  total_miles / total_gallons = 25 :=
by
  sorry

end combined_mpg_l130_130912


namespace tickets_per_box_l130_130401

-- Definitions
def boxes (G: Type) : ℕ := 9
def total_tickets (G: Type) : ℕ := 45

-- Theorem statement
theorem tickets_per_box (G: Type) : total_tickets G / boxes G = 5 :=
by
  sorry

end tickets_per_box_l130_130401


namespace rationalize_sqrt_three_sub_one_l130_130493

theorem rationalize_sqrt_three_sub_one :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_sqrt_three_sub_one_l130_130493


namespace ratio_of_segments_equal_areas_l130_130277

open EuclideanGeometry

noncomputable def midpoint (A B : Point): Point := sorry  -- Midpoint definition (noncomputable for real numbers)

variable {A B C D P Q M N : Point}

-- Conditions: midpoints of diagonals and intersecting line through midpoints
axiom midpoints_of_diagonals (A B C D : Point) (P Q : Point) :
  midpoint A C = P ∧ midpoint B D = Q

axiom line_through_midpoints (P Q : Point) (M N : Point) :
  LiesOnLine M (LineThroughPoints P Q) ∧ LiesOnLine N (LineThroughPoints P Q)

-- Questions to prove
theorem ratio_of_segments (A B M C D N : Point)
  (midpoints : midpoint A C = P ∧ midpoint B D = Q)
  (line_intersects : LiesOnLine M (LineThroughPoints P Q) ∧ LiesOnLine N (LineThroughPoints P Q)) :
  ((SegmentRatio A M B) = (SegmentRatio C N D)) :=
sorry

theorem equal_areas (A B N C D M : Point)
  (midpoints : midpoint A C = P ∧ midpoint B D = Q)
  (line_intersects : LiesOnLine M (LineThroughPoints P Q) ∧ LiesOnLine N (LineThroughPoints P Q)) :
  (TriangleArea A B N) = (TriangleArea C D M) :=
sorry

end ratio_of_segments_equal_areas_l130_130277


namespace sum_f_factorials_l130_130885

theorem sum_f_factorials {f : ℝ → ℝ} (h_add : ∀ x y, f(x + y) = f(x) + f(y)) (h_one : f(1) = 100) :
  ∑ k in Finset.range 10, f (Nat.factorial (k + 1)) = 403791300 := by
  sorry

end sum_f_factorials_l130_130885


namespace eight_div_repeat_three_l130_130120

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l130_130120


namespace column_for_975_l130_130668

/- Define the columns A, B, C, D, E, F as an enumeration -/
inductive Column
| A | B | C | D | E | F

open Column

/- Define a function to determine the column given a number and its row position -/
def column_of_number (n : ℕ) : Column :=
  let row_num := (n - 2) / 6 + 1 in
  let position_in_row := (n - 2) % 6 + 1 in
  if row_num % 2 = 1 then
    match position_in_row with
    | 1 => A | 2 => B | 3 => C | 4 => D | 5 => E | 6 => F
    | _ => A -- This case will never happen as position_in_row is in {1, ..., 6}
  else
    match position_in_row with
    | 1 => F | 2 => E | 3 => D | 4 => C | 5 => B | 6 => A
    | _ => F -- This case will never happen as position_in_row is in {1, ..., 6}

/- The main theorem to prove -/
theorem column_for_975 : column_of_number 975 = B :=
by
  sorry

end column_for_975_l130_130668


namespace divide_by_repeating_decimal_l130_130095

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l130_130095


namespace shaded_areas_I_and_III_equal_l130_130552

def area_shaded_square_I : ℚ := 1 / 4
def area_shaded_square_II : ℚ := 1 / 2
def area_shaded_square_III : ℚ := 1 / 4

theorem shaded_areas_I_and_III_equal :
  area_shaded_square_I = area_shaded_square_III ∧
   area_shaded_square_I ≠ area_shaded_square_II ∧
   area_shaded_square_III ≠ area_shaded_square_II :=
by {
  sorry
}

end shaded_areas_I_and_III_equal_l130_130552


namespace length_of_EF_l130_130997

-- Definitions for the problem conditions
variables (AB CD EF : ℕ) (BC : ℕ)
variables (h_parallel1 : AB ∥ CD) (h_parallel2 : CD ∥ EF)
variables (h_AB : AB = 150) (h_BC : BC = 100)

-- Theorem statement
theorem length_of_EF : EF = 60 :=
by
  sorry

end length_of_EF_l130_130997


namespace evaluate_expression_at_minus_one_l130_130893

def h (x : ℝ) : ℝ :=
  (2 * x^2 + 3 * x + 7) / (x^2 + 2 * x + 5)

def k (x : ℝ) : ℝ :=
  x + 2

theorem evaluate_expression_at_minus_one : h (k (-1)) + k (h (-1)) = 5 :=
by
  sorry

end evaluate_expression_at_minus_one_l130_130893


namespace cos_arithmetic_sequence_result_l130_130374

-- Define an arithmetic sequence as a function
def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem cos_arithmetic_sequence_result (a d : ℝ) 
  (h : arithmetic_seq a d 1 + arithmetic_seq a d 5 + arithmetic_seq a d 9 = 8 * Real.pi) :
  Real.cos (arithmetic_seq a d 3 + arithmetic_seq a d 7) = -1 / 2 := by
  sorry

end cos_arithmetic_sequence_result_l130_130374


namespace Eight_div_by_repeating_decimal_0_3_l130_130157

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l130_130157


namespace B_share_is_correct_l130_130639

open Real

noncomputable def total_money : ℝ := 10800
noncomputable def ratio_A : ℝ := 0.5
noncomputable def ratio_B : ℝ := 1.5
noncomputable def ratio_C : ℝ := 2.25
noncomputable def ratio_D : ℝ := 3.5
noncomputable def ratio_E : ℝ := 4.25
noncomputable def total_ratio : ℝ := ratio_A + ratio_B + ratio_C + ratio_D + ratio_E
noncomputable def value_per_part : ℝ := total_money / total_ratio
noncomputable def B_share : ℝ := ratio_B * value_per_part

theorem B_share_is_correct : B_share = 1350 := by 
  sorry

end B_share_is_correct_l130_130639


namespace expected_value_unfair_die_l130_130468

open ProbabilityTheory

noncomputable def expected_value (pmf : pmf ℕ) : ℝ :=
∑' i, (i:ℝ) * pmf i

theorem expected_value_unfair_die :
  let pmf := @pmf.of_finite_support _ _ _ _ ⟨
    { val := ![
      (1, 2/21:ℝ),
      (2, 2/21:ℝ),
      (3, 2/21:ℝ),
      (4, 2/21:ℝ),
      (5, 2/21:ℝ),
      (6, 2/21:ℝ),
      (7, 2/21:ℝ),
      (8, 1/3:ℝ)
    ], nodup := by simp [finset.nodup_map_on prod.fst_injective (finset.range 8).nodup] }⟩ in
  expected_value pmf = 16 / 3 :=
begin
  sorry
end

end expected_value_unfair_die_l130_130468


namespace persimmons_count_l130_130381

def apples : ℕ := 18
def fruits : ℕ := 33

theorem persimmons_count : ∀ (persimmons : ℕ), apples + persimmons = fruits → persimmons = 15 :=
by
  intros P h
  rw [←nat.add_sub_of_le (le_of_eq h), add_comm apples P, h]
  rw [sub_eq_zero_of_le (le_add_of_nonneg_left (le_refl 0))]
  rfl

end persimmons_count_l130_130381


namespace cows_in_group_l130_130858

theorem cows_in_group (c h : ℕ) (h_condition : 4 * c + 2 * h = 2 * (c + h) + 16) : c = 8 :=
sorry

end cows_in_group_l130_130858


namespace division_of_decimal_l130_130115

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l130_130115


namespace hex_inscribed_eq_triangle_l130_130942

theorem hex_inscribed_eq_triangle {O A B C D E F K L M : Type*}
    [EuclideanGeometry O] {R : ℝ}
    (h_hex : InscribedHexagon O A B C D E F)
    (h_radius : distance O A = R)
    (h_eq_sides : distance A B = R ∧ distance C D = R ∧ distance E F = R)
    (h_circ1 : Circumcircle O B C)
    (h_circ2 : Circumcircle O D E)
    (h_circ3 : Circumcircle O F A)
    (h_int1 : Intersection K h_circ1 h_circ3)
    (h_int2 : Intersection L h_circ1 h_circ2)
    (h_int3 : Intersection M h_circ2 h_circ3)
    (h_diff : K ≠ O ∧ L ≠ O ∧ M ≠ O) :
    EquilateralTriangle K L M ∧ side_length K L M = R := by
  sorry

end hex_inscribed_eq_triangle_l130_130942


namespace parallel_AC_l130_130527

theorem parallel_AC'_bisector
  (ABC : Triangle)
  (h_right_angle : ABC.angle B = 90)
  (I : Point) -- Incenter of the triangle
  (excircle : Excircle) -- Excircle opposite B
  (A1 A2 A' C' : Point)
  (h_A1_touch_BC : excircle.touches ABC.side BC A1)
  (h_A2_touch_AC : excircle.touches ABC.side AC A2)
  (h_A'_intersect_incircle : intersects_first (line_through A1 A2) (incircle ABC) A')
  (h_C'_intersect_incircle : intersects_first (line_through A1 A2) (incircle ABC) C') :
  parallel ABC.side AC (line_through A' C') :=
sorry

end parallel_AC_l130_130527


namespace sum_of_acute_angles_l130_130062

theorem sum_of_acute_angles (α β : ℝ) (h1 : 0 < α ∧ α < π / 2)
                            (h2 : 0 < β ∧ β < π / 2)
                            (h3 : sin α ^ 2 + sin β ^ 2 = sin (α + β)) :
  α + β = π / 2 :=
sorry

end sum_of_acute_angles_l130_130062


namespace eventually_constant_bn_l130_130446

open BigOperators

noncomputable def eventually_constant_series (a : ℕ → ℤ) :=
  ∃ N c, ∀ n, N ≤ n → a n = c

theorem eventually_constant_bn (a : ℕ → ℤ) (h1 : ∀ n ≥ 2016, n^2 ∣ ∑ i in finset.range(n + 1), a i)
                               (h2 : ∀ n ≥ 2016, a n ≤ (n + 2016)^2) :
    eventually_constant_series (λ n, a (n + 1) - a n) :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end eventually_constant_bn_l130_130446


namespace winning_candidate_percentage_l130_130256

theorem winning_candidate_percentage
  (votes_candidate1 : ℕ) (votes_candidate2 : ℕ) (votes_candidate3 : ℕ)
  (total_votes : ℕ) (winning_votes : ℕ) (percentage : ℚ)
  (h1 : votes_candidate1 = 1000)
  (h2 : votes_candidate2 = 2000)
  (h3 : votes_candidate3 = 4000)
  (h4 : total_votes = votes_candidate1 + votes_candidate2 + votes_candidate3)
  (h5 : winning_votes = votes_candidate3)
  (h6 : percentage = (winning_votes : ℚ) / total_votes * 100) :
  percentage = 57.14 := 
sorry

end winning_candidate_percentage_l130_130256


namespace at_least_two_solutions_l130_130751

theorem at_least_two_solutions (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (∃ x, (x - a) * (x - b) = x - c) ∨ (∃ x, (x - b) * (x - c) = x - a) ∨ (∃ x, (x - c) * (x - a) = x - b) ∨
    (((x - a) * (x - b) = x - c) ∧ ((x - b) * (x - c) = x - a)) ∨ 
    (((x - b) * (x + c) = x - a) ∧ ((x - c) * (x - a) = x - b)) ∨ 
    (((x - c) * (x - a) = x - b) ∧ ((x - a) * (x - b) = x - c)) :=
sorry

end at_least_two_solutions_l130_130751


namespace sum_of_squares_of_consecutive_integers_l130_130965

theorem sum_of_squares_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x^2 + (x + 1)^2 = 1625 := by
  sorry

end sum_of_squares_of_consecutive_integers_l130_130965


namespace total_deviation_correct_average_weight_per_bag_correct_l130_130606

-- Define standard weight per bag
def standard_weight : ℕ := 150

-- Define the deviations for each bag
def deviations : List ℤ := [-6, -3, -1, 7, 3, 4, -3, -2, -2, 1]

-- Define the number of bags
def num_bags : ℕ := 10

-- Define a function to calculate the total deviation
def total_deviation (devs : List ℤ) : ℤ := devs.sum

-- Define the total weight deviation
def total_weight_deviation := total_deviation deviations

-- Calculate the total weight
def total_weight := (standard_weight * num_bags : ℤ) + total_weight_deviation

-- Calculate the average weight per bag
def average_weight_per_bag := total_weight / (num_bags : ℤ)

-- Verify the total deviation
theorem total_deviation_correct : total_deviation deviations = -2 :=
by
  -- skipping the proof
  sorry

-- Verify the average weight per bag
theorem average_weight_per_bag_correct : average_weight_per_bag = 149.8 :=
by
  -- skipping the proof
  sorry

end total_deviation_correct_average_weight_per_bag_correct_l130_130606


namespace Eight_div_by_repeating_decimal_0_3_l130_130164

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l130_130164


namespace bob_age_l130_130685

theorem bob_age (a b : ℝ) 
    (h1 : b = 3 * a - 20)
    (h2 : b + a = 70) : 
    b = 47.5 := by
    sorry

end bob_age_l130_130685


namespace base_conversion_l130_130530

theorem base_conversion (c d : ℕ) (h₁ : 65_c = 56_d) (h₂ : 6 * c - 5 * d = 1) (hc : 0 < c) (hd : 0 < d) : c + d = 13 :=
by sorry

end base_conversion_l130_130530


namespace probability_exactly_one_win_l130_130531

theorem probability_exactly_one_win :
  let P_win_Jp := 2 / 3
  let P_win_Us := 2 / 5
  let P_exactly_one_win := P_win_Jp * (1 - P_win_Us) + (1 - P_win_Jp) * P_win_Us
  P_exactly_one_win = 8 / 15 :=
by
  let P_win_Jp := 2 / 3
  let P_win_Us := 2 / 5
  let P_exactly_one_win := P_win_Jp * (1 - P_win_Us) + (1 - P_win_Jp) * P_win_Us
  have h1 : P_exactly_one_win = 8 / 15 := sorry
  exact h1

end probability_exactly_one_win_l130_130531


namespace max_k_for_inequality_l130_130518

theorem max_k_for_inequality :
  (∀ x : ℝ, x > 2 → x * (1 + Real.log x) + 2 * 4 > 4 * x) :=
begin
  sorry
end

end max_k_for_inequality_l130_130518


namespace PlaneEquationCorrect_l130_130431

-- Definitions based on given conditions
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure Vector3D :=
  (i : ℝ)
  (j : ℝ)
  (k : ℝ)

def NormalPlaneEquation
  (P : Point3D) 
  (n : Vector3D) : Prop :=
  n.i * (x - P.x) + n.j * (y - P.y) + n.k * (z - P.z) = 0

-- Point P and Normal vector n based on conditions
def P : Point3D := {x := 1, y := 2, z := -1}
def n : Vector3D := {i := -2, j := 3, k := 1}

-- The theorem we need to prove
theorem PlaneEquationCorrect :
  NormalPlaneEquation P n :=
by 
  sorry

end PlaneEquationCorrect_l130_130431


namespace total_area_of_forest_and_fields_l130_130974

theorem total_area_of_forest_and_fields (r p k : ℝ) (h1 : k = 12) 
  (h2 : r^2 + 4 * p^2 + 45 = 12 * k) :
  (r^2 + 4 * p^2 + 12 * k = 135) :=
by
  -- Proof goes here
  sorry

end total_area_of_forest_and_fields_l130_130974


namespace divide_by_repeating_decimal_l130_130102

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l130_130102


namespace Zain_coins_total_l130_130234

theorem Zain_coins_total :
  ∀ (quarters dimes nickels : ℕ),
  quarters = 6 →
  dimes = 7 →
  nickels = 5 →
  Zain_coins = quarters + 10 + (dimes + 10) + (nickels + 10) →
  Zain_coins = 48 :=
by intros quarters dimes nickels hq hd hn Zain_coins
   sorry

end Zain_coins_total_l130_130234


namespace eight_div_repeat_three_l130_130128

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l130_130128


namespace conditional_probability_l130_130372

open ProbabilityTheory

variables {Ω : Type*} {P : MeasureTheory.Measure Ω}
variables {A B : Set Ω}

/-- Given that P(B) = 1/4 and P(A ∩ B) = 1/8, prove that P(A|B) = 1/2. -/
theorem conditional_probability (h1 : MeasureTheory.Probability (P B) = 1/4)
                                (h2 : MeasureTheory.Probability (P (A ∩ B)) = 1/8) :
  MeasureTheory.Probability (MeasureTheory.ProbabilityTheory.condP B A) = 1/2 := 
sorry

end conditional_probability_l130_130372


namespace zain_coin_total_l130_130237

def zain_coins (q d n : ℕ) := q + d + n
def emerie_quarters : ℕ := 6
def emerie_dimes : ℕ := 7
def emerie_nickels : ℕ := 5
def zain_quarters : ℕ := emerie_quarters + 10
def zain_dimes : ℕ := emerie_dimes + 10
def zain_nickels : ℕ := emerie_nickels + 10

theorem zain_coin_total : zain_coins zain_quarters zain_dimes zain_nickels = 48 := 
by
  unfold zain_coins zain_quarters zain_dimes zain_nickels emerie_quarters emerie_dimes emerie_nickels
  rfl

end zain_coin_total_l130_130237


namespace area_of_square_field_l130_130595

-- Define the side length of the square
def side_length : ℝ := 15

-- Define the area of the square based on the side length
def square_area (side : ℝ) : ℝ := side * side

-- The theorem stating the area of a square with side length 15 meters
theorem area_of_square_field : square_area side_length = 225 := 
by 
  sorry

end area_of_square_field_l130_130595


namespace composite_probability_matches_l130_130033

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, m > 1 ∧ m < n ∧ n % m = 0

def count_composites_up_to (n : ℕ) : ℕ :=
  (Finset.range n).filter is_composite |>.card

theorem composite_probability_matches :
  (count_composites_up_to 51) / 50 = 0.68 := 
sorry

end composite_probability_matches_l130_130033


namespace simplify_trig_expression_l130_130918

variables {x : Real}
def tan (x : Real) := sin x / cos x
def cot (x : Real) := cos x / sin x
def sec (x : Real) := 1 / cos x
def csc (x : Real) := 1 / sin x

theorem simplify_trig_expression :
  (tan x) / (1 + cot x) + (1 + cot x) / (tan x) = sec^2 x + csc^2 x :=
by
  sorry

end simplify_trig_expression_l130_130918


namespace eight_div_repeating_three_l130_130185

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l130_130185


namespace rationalize_sqrt_three_sub_one_l130_130495

theorem rationalize_sqrt_three_sub_one :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_sqrt_three_sub_one_l130_130495


namespace eight_div_repeating_three_l130_130079

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l130_130079


namespace triangle_area_given_conditions_l130_130869

theorem triangle_area_given_conditions :
  ∀ (P : ℝ × ℝ) (t : ℝ),
  let l_parametric_x := 1 + t,
      l_parametric_y := 1 - t,
      rho := sqrt (P.1^2 + P.2^2),
      theta := real.arctan (P.2 / P.1)
  in (ρ*ρ*(5 - 4 * real.cos (2 * theta) = 9) → 
  let rect_x := -1,
      rect_y := 1,
      distance_from_P_to_l := sqrt 2,
      AB_length := (6 * real.sqrt 3) / 5,
      area := (3 * real.sqrt 6) / 5
  in area = (3 * sqrt 6) / 5) := 
sorry

end triangle_area_given_conditions_l130_130869


namespace determine_hyperbola_equation_l130_130432

-- Definitions for the problem
def is_asymptote (H : ℝ × ℝ → Prop) (L : ℝ → ℝ) : Prop :=
  ∀ x y, H (x, y) → (y = L x)

def hyperbola_equation (x y : ℝ) (λ : ℝ) : Prop :=
  x^2 - y^2 / 4 = λ

def parabola_focus : ℝ × ℝ := (1, 0)

-- Statement of the problem
theorem determine_hyperbola_equation :
  ∃ λ : ℝ, λ ≠ 0 ∧ 
    (∀ x y, hyperbola_equation x y λ → (is_asymptote (hyperbola_equation x y λ) (λ x)) ∨ (is_asymptote (hyperbola_equation x y λ) (λ (-x)))) ∧
    (hyperbola_equation 1 0 λ) :=
  sorry

end determine_hyperbola_equation_l130_130432


namespace simplify_trig_expression_l130_130922

theorem simplify_trig_expression (α : ℝ) : 
  (1 + cos α + cos (2 * α) + cos (3 * α)) / (cos α + 2 * cos α^2 - 1) = 2 * cos α := 
by
  sorry

end simplify_trig_expression_l130_130922


namespace perfect_square_factorial_l130_130231

theorem perfect_square_factorial :
  ∃ (n : ℕ), (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5) →
  (n = 1 → ¬ perfect_square (factorial 97 * factorial 98)) ∧
  (n = 2 → ¬ perfect_square (factorial 97 * factorial 99)) ∧
  (n = 3 → ¬ perfect_square (factorial 98 * factorial 99)) ∧
  (n = 4 → ¬ perfect_square (factorial 98 * factorial 100)) ∧
  (n = 5 → perfect_square (factorial 99 * factorial 100)) :=
begin
  sorry
end

end perfect_square_factorial_l130_130231


namespace number_of_integers_satisfying_inequality_l130_130815

theorem number_of_integers_satisfying_inequality :
  set.count {n : ℤ | (n + 2) * (n - 8) ≤ 0} = 11 :=
sorry

end number_of_integers_satisfying_inequality_l130_130815


namespace division_of_decimal_l130_130110

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l130_130110


namespace probability_of_no_rain_l130_130956

theorem probability_of_no_rain (prob_rain : ℚ) (prob_no_rain : ℚ) (days : ℕ) (h : prob_rain = 2/3) (h_prob_no_rain : prob_no_rain = 1 - prob_rain) :
  (prob_no_rain ^ days) = 1/243 :=
by 
  sorry

end probability_of_no_rain_l130_130956


namespace compound_interest_principal_l130_130931

theorem compound_interest_principal (I : ℝ) (r : ℝ) (t : ℝ) (n : ℕ) (P : ℝ) :
  I = 662 →
  r = 0.10 →
  t = 3.0211480362537766 →
  n = 1 →
  P = I / ((1 + r/n)^(n*t) - 1) →
  P ≈ 2000 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end compound_interest_principal_l130_130931


namespace division_of_repeating_decimal_l130_130066

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l130_130066


namespace cos_sum_of_sins_l130_130403

theorem cos_sum_of_sins (α β : ℝ) (h : sin α * sin β = 1) : cos (α + β) = -1 :=
by
  sorry

end cos_sum_of_sins_l130_130403


namespace least_x_satisfying_cos_l130_130896

theorem least_x_satisfying_cos (x : ℝ) (h1 : x > 2) (h2 : cos (x * pi / 180) = cos ((x^2 + 10) * pi / 180)) : x = 7 :=
sorry

end least_x_satisfying_cos_l130_130896


namespace initial_amount_of_money_l130_130903

theorem initial_amount_of_money (dollars_left : ℕ) (dollars_spent : ℕ)
  (h_left : dollars_left = 29) (h_spent : dollars_spent = 25) : 
  dollars_left + dollars_spent = 54 :=
by 
  rw [h_left, h_spent]
  rfl

end initial_amount_of_money_l130_130903


namespace vector_combination_bound_l130_130465

-- Define the context of vectors and their norms
variables {n : ℕ} (a : fin n → EuclideanSpace ℝ (fin 3))
-- Length condition of the vectors
variable (ha : ∀ i, ∥a i∥ ≤ 1)

theorem vector_combination_bound :
  ∃ s : fin n → bool, ∥∑ i in finset.univ, (if s i then 1 else -1 ) • (a i : EuclideanSpace ℝ (fin 3))∥ ≤ √2 :=
sorry

end vector_combination_bound_l130_130465


namespace battery_charging_time_l130_130640

theorem battery_charging_time (t_phone t_tablet t_laptop : ℕ) (p_phone p_tablet p_laptop : ℕ) :
  t_phone = 26 →
  t_tablet = 53 →
  t_laptop = 80 →
  p_phone = 75 →
  p_tablet = 100 →
  p_laptop = 50 →
  (t_tablet + (p_phone * t_phone / 100) + (p_laptop * t_laptop / 100) = 112.5) := sorry

end battery_charging_time_l130_130640


namespace probability_of_concave_number_is_one_third_l130_130655

-- Definitions based on the conditions in the problem
def is_concave (a b c : ℕ) := (a > b) ∧ (b < c)

def is_valid_digit (n : ℕ) := n ∈ ({1, 2, 3, 4} : Finset ℕ)

def are_distinct (a b c : ℕ) := a ≠ b ∧ b ≠ c ∧ a ≠ c

def all_three_digit_numbers : Finset (ℕ × ℕ × ℕ) :=
  { x | let (a, b, c) := x in is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ are_distinct a b c }

def concave_numbers : Finset (ℕ × ℕ × ℕ) :=
  { x | let (a, b, c) := x in is_concave a b c ∧ x ∈ all_three_digit_numbers }

def probability_concave : ℚ :=
  (concave_numbers.card : ℚ) / (all_three_digit_numbers.card : ℚ)

theorem probability_of_concave_number_is_one_third :
  probability_concave = 1 / 3 :=
by
  sorry

end probability_of_concave_number_is_one_third_l130_130655


namespace proof_median_mean_modes_median_l130_130310

-- Define the given data set
def occurrences : ℕ → ℕ
| n := if n = 29 then 12
       else if n = 30 then 11
       else if n = 31 then 7
       else if n ≥ 1 ∧ n ≤ 28 then 12
       else 0

-- Define the median
noncomputable def median (occurrences : ℕ → ℕ) : ℚ :=
let values_list := List.range' 1 32
  in let all_values := values_list.bind (λ n, List.replicate (occurrences n) n)
    in (all_values[(366 / 2) - 1] + all_values[366 / 2]) / 2

-- Define the mean
noncomputable def mean (occurrences : ℕ → ℕ) : ℚ :=
let numerator := ∑ n in List.range' 1 32, occurrences n * n
    in numerator / 366

-- Define the median of the modes
noncomputable def modes_median (occurrences : ℕ → ℕ) : ℚ :=
let modes := List.Icc 1 28
    in (1 + 29) / 2

-- Define the proof statement
theorem proof_median_mean_modes_median :
  let μ := mean occurrences
  let M := median occurrences
  let d := modes_median occurrences
  in d < μ ∧ μ < M :=
by
  sorry

end proof_median_mean_modes_median_l130_130310


namespace distance_travelled_downstream_l130_130539

theorem distance_travelled_downstream :
  let speed_boat_still_water := 42 -- km/hr
  let rate_current := 7 -- km/hr
  let time_travelled_min := 44 -- minutes
  let time_travelled_hrs := time_travelled_min / 60.0 -- converting minutes to hours
  let effective_speed_downstream := speed_boat_still_water + rate_current -- km/hr
  let distance_downstream := effective_speed_downstream * time_travelled_hrs
  distance_downstream = 35.93 :=
by
  -- Proof will go here
  sorry

end distance_travelled_downstream_l130_130539


namespace sum_possible_values_of_n_l130_130915

variable {T : ℕ} -- Total number of players
variable {n : ℕ} -- Score of each remaining player

-- Conditions based on the problem
axiom simon_score : ∀ n T : ℕ, ∃ S : ℕ, S = 8 -- Simon scores 8 points
axiom garfunkle_score : ∀ n T : ℕ, ∃ G : ℕ, G = 8 -- Garfunkle scores 8 points
axiom total_points : ∀ n T : ℕ, 16 + n * (T - 2) = T * (T - 1) / 2 -- Total points equation

theorem sum_possible_values_of_n (n T : ℕ) (h_valid_total : T ≠ 0 ∧ T ≠ 1) :
  (n = 17) :=
begin
  sorry
end

end sum_possible_values_of_n_l130_130915


namespace arithmetic_sequence_ratio_l130_130367

noncomputable def sum_first_n_terms (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_ratio (d : ℚ) (h : d ≠ 0) :
  let a₁ := 8 * d
  let S₅ := sum_first_n_terms a₁ d 5
  let S₇ := sum_first_n_terms a₁ d 7
  (7 * S₅) / (5 * S₇) = 10 / 11 :=
by 
  let a₁ := 8 * d
  let S₅ := sum_first_n_terms a₁ d 5
  let S₇ := sum_first_n_terms a₁ d 7
  sorry

end arithmetic_sequence_ratio_l130_130367


namespace largest_divisor_of_n5_minus_n_l130_130720

theorem largest_divisor_of_n5_minus_n (n : ℤ) : 
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n^5 - n)) ∧ d = 30 :=
sorry

end largest_divisor_of_n5_minus_n_l130_130720


namespace find_value_of_P_l130_130254

variable {a b c : ℝ}
variable (h1 : a^2 + b^2 = c^2 + a * b)
variable (h2 : c = sqrt (a^2 + b^2 - a * b))
variable (h3 : ∠ C = 60)

theorem find_value_of_P (h : ∠ C = 60) (h4 : (a / (b + c)) + (b / (a + c)) = P) : P = 1 :=
by
  sorry

end find_value_of_P_l130_130254


namespace smallest_positive_angle_l130_130307

theorem smallest_positive_angle :
  ∃ (x : ℝ), (x > 0) ∧ (tan (3 * x * (real.pi / 180)) = (cos (x * (real.pi / 180)) - sin (x * (real.pi / 180))) / (cos (x * (real.pi / 180)) + sin (x * (real.pi / 180)))) ∧ x = 11.25 :=
by
  sorry

end smallest_positive_angle_l130_130307


namespace count_integers_satisfying_inequality_l130_130803

theorem count_integers_satisfying_inequality :
  {n : ℤ | (n + 2) * (n - 8) ≤ 0}.to_finset.card = 11 :=
sorry

end count_integers_satisfying_inequality_l130_130803


namespace no_integer_solutions_for_inequality_l130_130729

open Int

theorem no_integer_solutions_for_inequality : ∀ x : ℤ, (x - 4) * (x - 5) < 0 → False :=
by
  sorry

end no_integer_solutions_for_inequality_l130_130729


namespace john_piano_lessons_l130_130442

theorem john_piano_lessons (total_cost piano_cost original_price_per_lesson discount : ℕ) 
    (total_spent : ℕ) : 
    total_spent = piano_cost + ((total_cost - piano_cost) / (original_price_per_lesson - discount)) → 
    total_cost = 1100 ∧ piano_cost = 500 ∧ original_price_per_lesson = 40 ∧ discount = 10 → 
    (total_cost - piano_cost) / (original_price_per_lesson - discount) = 20 :=
by
  intros h1 h2
  sorry

end john_piano_lessons_l130_130442


namespace variance_le_second_moment_l130_130483

noncomputable def variance (X : ℝ → ℝ) (MX : ℝ) : ℝ :=
  sorry -- Assume defined as M[(X - MX)^2]

noncomputable def second_moment (X : ℝ → ℝ) (C : ℝ) : ℝ :=
  sorry -- Assume defined as M[(X - C)^2]

theorem variance_le_second_moment (X : ℝ → ℝ) :
  ∀ C : ℝ, C ≠ MX → variance X MX ≤ second_moment X C := 
by
  sorry

end variance_le_second_moment_l130_130483


namespace eight_div_repeating_three_l130_130190

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l130_130190


namespace sum_of_solutions_l130_130726

theorem sum_of_solutions (x : ℝ) (hx : x ∈ Set.Icc 0 (2 * Real.pi)) (h_eq : Real.tan x ^ 2 - 9 * Real.tan x + 1 = 0) :
  ∑ x in Set.toFinset {x | x ∈ Set.Icc 0 (2 * Real.pi) ∧ Real.tan x ^ 2 - 9 * Real.tan x + 1 = 0}, x = 3 * Real.pi := 
sorry

end sum_of_solutions_l130_130726


namespace value_of_a_plus_b_minus_c_l130_130450

def a : ℤ := 1 -- smallest positive integer
def b : ℤ := 0 -- number with the smallest absolute value
def c : ℤ := -1 -- largest negative integer

theorem value_of_a_plus_b_minus_c : a + b - c = 2 := by
  -- skipping the proof
  sorry

end value_of_a_plus_b_minus_c_l130_130450


namespace describe_shape_l130_130425

variable {ρ : ℝ} {θ φ a b : ℝ}

noncomputable def shape (a b : ℝ) (ρ : ℝ) (θ : ℝ) (φ : ℝ) : String :=
  if 0 <= a ∧ a < b ∧ b <= π ∧ ρ > 0 ∧ 0 <= θ ∧ θ < 2 * π then
    if a = 0 ∧ b = π then "Cone"
    else "Conical Frustum"
  else
    "Undefined"

theorem describe_shape (a b : ℝ) (ρ : ℝ) (θ : ℝ) (φ : ℝ) :
  0 <= a →
  a < b →
  b <= π →
  ρ > 0 →
  0 <= θ →
  θ < 2 * π →
  (φ ∈ Set.Icc a b) →
  shape a b ρ θ φ = if a = 0 ∧ b = π then "Cone" else "Conical Frustum" :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  unfold shape
  split_ifs
  { sorry }
  { sorry }
  { contradiction }

end describe_shape_l130_130425


namespace division_of_repeating_decimal_l130_130070

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l130_130070


namespace length_of_diagonal_and_BD_l130_130533

theorem length_of_diagonal_and_BD' (α β : ℝ) (h1 : 0 < α) (h2 : 0 < β) 
  (h3 : α ≤ β) (h4 : α + β < π / 2) : 
  ∃ BD B'D, BD = real.sin (α + β) ∨ BD = real.sin (β - α) ∧ 
  (B'D = real.sin (β - α) ∨ B'D = 0) := 
sorry

end length_of_diagonal_and_BD_l130_130533


namespace employee_pay_l130_130247

theorem employee_pay (y : ℝ) (x : ℝ) (h1 : x = 1.2 * y) (h2 : x + y = 700) : y = 318.18 :=
by
  sorry

end employee_pay_l130_130247


namespace find_n_l130_130382

variable (P s k m : ℝ)

theorem find_n (n : ℝ) : 
  P = s / (1 + k + m) ^ n → 
  n = log (s / P) / log (1 + k + m) :=
by
  intro h
  sorry

end find_n_l130_130382


namespace divide_by_repeating_decimal_l130_130096

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l130_130096


namespace smallest_n_condition_l130_130875

theorem smallest_n_condition (n : ℕ) : 25 * n - 3 ≡ 0 [MOD 16] → n ≡ 11 [MOD 16] :=
by
  sorry

end smallest_n_condition_l130_130875


namespace student_correct_answers_l130_130423

noncomputable def correct_answers : ℕ := 58

theorem student_correct_answers (C W : ℕ) (h1 : C + W = 100) (h2 : 5 * C - 2 * W = 210) : C = correct_answers :=
by {
  -- placeholder for actual proof
  sorry
}

end student_correct_answers_l130_130423


namespace trapezoid_area_ratio_l130_130990

theorem trapezoid_area_ratio (AD AO OB BC AB DO OC : ℝ) (h_eq1 : AD = 15) (h_eq2 : AO = 15) (h_eq3 : OB = 15) (h_eq4 : BC = 15)
  (h_eq5 : AB = 20) (h_eq6 : DO = 20) (h_eq7 : OC = 20) (is_trapezoid : true) (OP_perp_to_AB : true) 
  (X_mid_AD : true) (Y_mid_BC : true) : (5 + 7 = 12) :=
by
  sorry

end trapezoid_area_ratio_l130_130990


namespace man_monthly_salary_l130_130635

theorem man_monthly_salary (S E : ℝ) (h1 : 0.20 * S = S - 1.20 * E) (h2 : E = 0.80 * S) :
  S = 6000 :=
by
  sorry

end man_monthly_salary_l130_130635


namespace possible_orders_count_l130_130693

open List

-- Condition definitions
def initial_array := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

/--
This examines the number of possible orders after performing the given action
any number of times and determines that there are 20 possible orders.
-/
theorem possible_orders_count : 
  let actions : List ℕ → List (List ℕ) :=
    λ arr, [arr] ++ (List.range (length arr)).map (λ i, 
      let n := i + 1 in
      let first_part := reverse (arr.take n) in
      let second_part := reverse (arr.drop n) in
      (first_part ++ second_part)) in
  ∃ l : List (List ℕ), (∀ arr ∈ l, arr.permutations = initial_array.permutations ∨ arr.permutations = (reverse initial_array).permutations) 
                       ∧ l.length = 20 := 
begin
  -- The proof steps can be added here
  sorry
end

end possible_orders_count_l130_130693


namespace evaluate_expression_l130_130707

theorem evaluate_expression : 8 - 5 * (9 - (4 - 2)^2) * 2 = -42 := by
  sorry

end evaluate_expression_l130_130707


namespace spherical_straight_lines_l130_130573

noncomputable def is_sphere (S : Type*) [metric_space S] : Prop := sorry
noncomputable def is_plane (P : Type*) [metric_space P] : Prop := sorry
noncomputable def is_geodesic (M : Type*) [metric_space M] (γ : M → M → M) : Prop := sorry
noncomputable def is_straight_line (P : Type*) [metric_space P] (line : P → P → P) : Prop := 
  ∀ (p1 p2 : P), ∃! (γ : P → P → P), γ = line p1 p2 ∧ is_geodesic P γ

noncomputable def great_circle (S : Type*) [metric_space S] (gc : S → S → S) : Prop := 
  ∀ (p1 p2 : S), gc p1 p2 = (shortest_path_on_sphere S p1 p2)

theorem spherical_straight_lines (S : Type*) [metric_space S] [is_sphere S] :
  ∀ (gc : S → S → S), (is_geodesic S gc) → (great_circle S gc) := sorry

end spherical_straight_lines_l130_130573


namespace even_function_has_a_equal_2_l130_130824

noncomputable def f (a x : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_has_a_equal_2 (a : ℝ) :
  (∀ x : ℝ, f a (-x) = f a x) → a = 2 :=
sorry

end even_function_has_a_equal_2_l130_130824


namespace exactly_one_true_l130_130780

-- Define the propositions
def proposition1 (Q : Type) [Quadrilateral Q] : Prop := 
  ∀ q : Q, equal_diagonals q → is_rectangle q

def proposition2 (P : Type) [Parallelogram P] : Prop :=
  ∀ p : P, bisects_diagonal p → is_rhombus p

def proposition3 (Q : Type) [Quadrilateral Q] : Prop :=
  ∀ q : Q, (has_one_pair_parallel q ∧ has_one_pair_equal_sides q) → is_parallelogram q

-- Theorem to be proved: Exactly one of the propositions is true.
theorem exactly_one_true (Q P : Type) [Quadrilateral Q] [Parallelogram P] :
  (proposition1 Q → false) ∧
  proposition2 P ∧
  (proposition3 Q → false) :=
begin
  sorry,
end

end exactly_one_true_l130_130780


namespace Eight_div_by_repeating_decimal_0_3_l130_130159

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l130_130159


namespace distances_inequality_l130_130882

theorem distances_inequality (x y z : ℝ) (h : x + y + z = 1): x^2 + y^2 + z^2 ≥ x^3 + y^3 + z^3 + 6 * x * y * z :=
by
  sorry

end distances_inequality_l130_130882


namespace rope_length_before_decorating_l130_130437

/-
It takes 0.84 meters (m) of shiny string to decorate one tree.
There are 50 centimeters (cm) of rope left after decorating 10 trees.
Prove that the length of the rope before decorating the trees is 8.9 meters.
-/
theorem rope_length_before_decorating : 
  ∀ (rope_needed_per_tree : ℝ) (rope_left_after_decorating : ℝ) (number_of_trees : ℕ) (initial_rope_length : ℝ),  
  rope_needed_per_tree = 0.84 →
  rope_left_after_decorating = 0.5 →
  number_of_trees = 10 →
  initial_rope_length = rope_needed_per_tree * number_of_trees + rope_left_after_decorating →
  initial_rope_length = 8.9 :=
by 
  intros rope_needed_per_tree rope_left_after_decorating number_of_trees initial_rope_length
  rintros rfl rfl rfl rfl
  sorry

end rope_length_before_decorating_l130_130437


namespace isosceles_triangle_base_length_l130_130865

theorem isosceles_triangle_base_length
  (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [LE A] [LE B] [LE C]
  (isosceles_triangle : ∃ (a b c : ℝ), a ≠ b ∧ b = c ∧ c ≠ a ∧ angle A B C = 120)
  (AB_eq_1 : dist A B = 1) :
  dist B C = Real.sqrt 3 :=
sorry

end isosceles_triangle_base_length_l130_130865


namespace neither_sufficient_nor_necessary_l130_130898

noncomputable def condition_check (a : ℝ) : Prop :=
  (a = 4) → ¬(∀ b : ℝ, (a ≠ b) → ¬((ax + 8y - 8 = 0) ∧ (2x + ay - a = 0) → (a = 4)))

theorem neither_sufficient_nor_necessary (a : ℝ) : condition_check a :=
sorry

end neither_sufficient_nor_necessary_l130_130898


namespace number_of_integers_satisfying_inequality_l130_130813

theorem number_of_integers_satisfying_inequality : 
  {n : ℤ | (n + 2) * (n - 8) ≤ 0}.finite.card = 11 := by
  sorry

end number_of_integers_satisfying_inequality_l130_130813


namespace maria_oatmeal_cookies_l130_130901

/-- Maria was making baggies of cookies with 8 cookies in each bag. 
    She had 5 chocolate chip cookies and she could make 3 baggies. 
    Show that she had 19 oatmeal cookies. -/
theorem maria_oatmeal_cookies : 
  ∀ (bags : ℕ) (cookies_per_bag : ℕ) (chocolate_chip_cookies : ℕ), 
  bags = 3 → cookies_per_bag = 8 → chocolate_chip_cookies = 5 → 
  (bags * cookies_per_bag) - chocolate_chip_cookies = 19 :=
by
  intros bags cookies_per_bag chocolate_chip_cookies h_bags h_cookies_per_bag h_chocolate_chip_cookies
  rw [h_bags, h_cookies_per_bag, h_chocolate_chip_cookies]
  exact Nat.zero_add 19

end maria_oatmeal_cookies_l130_130901


namespace eight_div_repeating_three_l130_130152

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l130_130152


namespace division_of_repeating_decimal_l130_130075

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l130_130075


namespace median_length_of_isosceles_triangle_l130_130994

theorem median_length_of_isosceles_triangle
  (A B C : Type*)
  [inner_product_space ℝ A]
  (AB AC : A)
  (AM : A)
  (BC_length : ℝ)
  (h1 : ∥AB∥ = 8)
  (h2 : ∥AC∥ = 8)
  (h3 : BC_length = 10)
  (M : A)
  (midpoint_M : (B + C) / 2 = M)
  : ∥AM∥ = real.sqrt 39 :=
sorry

end median_length_of_isosceles_triangle_l130_130994


namespace number_of_small_jars_l130_130050

theorem number_of_small_jars (S L : ℕ) (h1 : S + L = 100) (h2 : 3 * S + 5 * L = 376) : S = 62 :=
by
  sorry

end number_of_small_jars_l130_130050


namespace orchids_now_l130_130988

def initial_roses : ℕ := 5
def initial_orchids : ℕ := 3
def current_roses : ℕ := 12
def rose_orchid_difference : ℕ := 10

theorem orchids_now :
  ∃ (O : ℕ), current_roses = O + rose_orchid_difference ∧ O = 2 :=
by
  use 2
  split
  · exact Nat.add_comm 10 2 ▸ rfl
  · rfl

end orchids_now_l130_130988


namespace Bob_age_is_47_l130_130681

variable (Bob_age Alice_age : ℝ)

def equations_holds : Prop := 
  Bob_age = 3 * Alice_age - 20 ∧ Bob_age + Alice_age = 70

theorem Bob_age_is_47.5 (h: equations_holds Bob_age Alice_age) : Bob_age = 47.5 := 
by sorry

end Bob_age_is_47_l130_130681


namespace length_AD_l130_130870

theorem length_AD {A B C D : Type} [linear_ordered_field D] 
  (BD BC AC : D)
  (h1 : BD = 36) 
  (h2 : BC = 45) 
  (h3 : AC = 40) 
  (h4 : ∃ D : A, ∀ A B C, A ≠ B ∧ A ≠ C ∧ angle_bisector_in_corners A B C D) : 
  AD = 68 := 
by 
  sorry
  
-- Definitions needed for the theorem
def angle_bisector_in_corners (A B C : Type) (D : A) : Prop :=
  ∃ AD1 AD2 : D, AD1 = AD2  -- Simplified for example

end length_AD_l130_130870


namespace total_number_of_letters_l130_130855

theorem total_number_of_letters
  (D S T : ℕ)
  (D_inter_S : ℕ := 20)
  (S_minus_D_inter_S : ℕ := 36)
  (D_minus_D_inter_S : ℕ := 4)
  (D_union_S_equals_T : D ∪ S = T)
  (D_and_S_equation : D = D_minus_D_inter_S + D_inter_S)
  (S_and_D_equation : S = S_minus_D_inter_S + D_inter_S) :
  T = 60 := by
  sorry

end total_number_of_letters_l130_130855


namespace divide_by_repeating_decimal_l130_130099

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l130_130099


namespace max_pq_rs_l130_130705

theorem max_pq_rs : 
  ∃ (p q r s : ℕ), {p, q, r, s} = {1, 2, 3, 4} ∧ p^q + r^s = 83 :=
by
  sorry

end max_pq_rs_l130_130705


namespace shortest_tree_height_is_correct_l130_130549

-- Definitions of the tree heights
def tallest_tree_height : ℕ := 150
def middle_tree_height : ℕ := (2 * tallest_tree_height) / 3
def shortest_tree_height : ℕ := middle_tree_height / 2

-- Theorem statement
theorem shortest_tree_height_is_correct :
  shortest_tree_height = 50 :=
by
  sorry

end shortest_tree_height_is_correct_l130_130549


namespace total_animal_eyes_l130_130421

theorem total_animal_eyes (frogs crocodiles : ℕ) (eyes_per_frog eyes_per_crocodile : ℕ) :
  frogs = 20 →
  crocodiles = 6 →
  eyes_per_frog = 2 →
  eyes_per_crocodile = 2 →
  (frogs * eyes_per_frog + crocodiles * eyes_per_crocodile) = 52 :=
by
  intros h_frogs h_crocodiles h_eyes_per_frog h_eyes_per_crocodile
  rw [h_frogs, h_crocodiles, h_eyes_per_frog, h_eyes_per_crocodile]
  norm_num
  sorry

end total_animal_eyes_l130_130421


namespace vertex_of_parabola_l130_130016

theorem vertex_of_parabola : 
  ∀ x, (3 * (x - 1)^2 + 2) = ((x - 1)^2 * 3 + 2) := 
by {
  -- The proof steps would go here
  sorry -- Placeholder to signify the proof steps are omitted
}

end vertex_of_parabola_l130_130016


namespace power_function_value_l130_130789

noncomputable def f (x : ℝ) : ℝ := x ^ (1/2)

-- Given the condition
axiom passes_through_point : f 3 = Real.sqrt 3

-- Prove that f(9) = 3
theorem power_function_value : f 9 = 3 := by
  sorry

end power_function_value_l130_130789


namespace least_positive_integer_satisfies_l130_130722

noncomputable def least_positive_integer : ℕ :=
  1

theorem least_positive_integer_satisfies :
  (∑ k in finset.range 80, (1 / (Real.sin (15 + 2 * k) * Real.sin (15 + 2 * k + 1)))) = (1 / (Real.sin least_positive_integer)) := 
by
  sorry

end least_positive_integer_satisfies_l130_130722


namespace distance_MN_is_2R_l130_130418

-- Definitions for the problem conditions
variable (R : ℝ) (A B C M N : ℝ) (alpha : ℝ)
variable (AC AB : ℝ)

-- Assumptions based on the problem statement
axiom circle_radius (r : ℝ) : r = R
axiom chord_length_AC (ch_AC : ℝ) : ch_AC = AC
axiom chord_length_AB (ch_AB : ℝ) : ch_AB = AB
axiom distance_M_to_AC (d_M_AC : ℝ) : d_M_AC = AC
axiom distance_N_to_AB (d_N_AB : ℝ) : d_N_AB = AB
axiom angle_BAC (ang_BAC : ℝ) : ang_BAC = alpha

-- To prove: the distance between M and N is 2R
theorem distance_MN_is_2R : |MN| = 2 * R := sorry

end distance_MN_is_2R_l130_130418


namespace max_value_f_l130_130762

theorem max_value_f (x y z : ℝ) (h : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) : 
  ∃ M, M = (sqrt 5) / 2 ∧ ∀ x y z : ℝ, (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) → (f x y z =  (xy + 2 * yz) / (x^2 + y^2 + z^2)) ≤ M :=
begin
  sorry
end

end max_value_f_l130_130762


namespace division_of_repeating_decimal_l130_130142

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l130_130142


namespace sum_of_dimensions_l130_130281

-- Define A, B, C as real numbers
variables (A B C : ℝ)

-- Given conditions
def cond1 : Prop := A * B = 40
def cond2 : Prop := A * C = 90
def cond3 : Prop := B * C = 100

-- Prove that A + B + C = 83/3
theorem sum_of_dimensions (h1 : cond1 A B C) (h2 : cond2 A B C) (h3 : cond3 A B C) : A + B + C = 83 / 3 := 
sorry

end sum_of_dimensions_l130_130281


namespace percentage_import_tax_l130_130636

theorem percentage_import_tax (total_value import_paid excess_amount taxable_amount : ℝ) 
  (h1 : total_value = 2570) 
  (h2 : import_paid = 109.90) 
  (h3 : excess_amount = 1000) 
  (h4 : taxable_amount = total_value - excess_amount) : 
  taxable_amount = 1570 →
  (import_paid / taxable_amount) * 100 = 7 := 
by
  intros h_taxable_amount
  simp [h1, h2, h3, h4, h_taxable_amount]
  sorry -- Proof goes here

end percentage_import_tax_l130_130636


namespace A1M_perpendicular_BC_l130_130873

variables (A B C A1 B1 C1 M : Type)
variables [incircle_touches_sides : ∀ (x : Type), (A B C : x) → Prop]
variables [median_intersects_segment : ∀ (x : Type), (A B C : x) → Prop]
variables [triangle : (A B C A1 B1 C1 : Type) → triangle]
variables [median : (A B C : Type) → median]
variables [perpendicular : (A B C A1 B1 C1 M : Type) → Prop]

theorem A1M_perpendicular_BC 
  (h₁ : incircle_touches_sides A B C A1 B1 C1)
  (h₂ : median_intersects_segment A B C A1 B1 C1 M) :
  perpendicular A1 M B C :=
sorry

end A1M_perpendicular_BC_l130_130873


namespace complement_intersection_l130_130900

variable U : Set Int := {-1, -2, -3, -4, 0}
variable A : Set Int := {-1, -2, 0}
variable B : Set Int := {-3, -4, 0}

theorem complement_intersection :
  (U \ A) ∩ B = {-3, -4} :=
by
  sorry

end complement_intersection_l130_130900


namespace min_packs_to_buy_exactly_120_cans_l130_130511

theorem min_packs_to_buy_exactly_120_cans : ∀ (packs : List ℕ), packs = [8, 16, 32] → 
  (∃ (n : ℕ), ∃ (a b c : ℕ), n = a + b + c ∧ 32 * a + 16 * b + 8 * c = 120 ∧ n = 5) :=
begin
  sorry
end

end min_packs_to_buy_exactly_120_cans_l130_130511


namespace segments_in_proportion_l130_130841

theorem segments_in_proportion (a b c d : ℝ) (ha : a = 1) (hb : b = 4) (hc : c = 2) (h : a / b = c / d) : d = 8 := 
by 
  sorry

end segments_in_proportion_l130_130841


namespace prob_odd_product_l130_130009

-- Definition of the range of integers from 1 to 25 inclusive.
def intRange := {n : ℤ | 1 ≤ n ∧ n ≤ 25}

-- Number of odd integers in the given range.
def oddIntRange := {n : ℤ | 1 ≤ n ∧ n ≤ 25 ∧ n % 2 = 1}

-- Total number of ways to choose two elements from the range.
def totalWays := nat.choose 25 2

-- Number of ways to choose two odd elements from the range.
def oddWays := nat.choose 13 2

-- Probability that the product of two distinct integers chosen from the range is odd.
theorem prob_odd_product : 
  (oddWays : ℚ) / (totalWays : ℚ) = 13 / 50 := 
by
  sorry

end prob_odd_product_l130_130009


namespace sum_powers_seventh_l130_130472

/-- Given the sequence values for sums of powers of 'a' and 'b', prove the value of the sum of the 7th powers. -/
theorem sum_powers_seventh (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^7 + b^7 = 29 := 
  sorry

end sum_powers_seventh_l130_130472


namespace t_minus_s_l130_130647

-- definitions
def num_students : ℕ := 120
def num_teachers : ℕ := 6
def enrollments : list ℕ := [40, 30, 30, 10, 5, 5]

-- calculating t (average number of students per teacher)
def t : ℕ := (enrollments.sum) / num_teachers

-- calculating s (average number of students per student)
def class_weights : list ℕ := enrollments.map (λ n, n * n)
def s : ℚ := (class_weights.sum : ℚ) / num_students

-- proving t - s = -9.58
theorem t_minus_s : (t : ℚ) - s = -9.58 := sorry

end t_minus_s_l130_130647


namespace least_positive_angle_l130_130334

theorem least_positive_angle:
  let θ := (15: ℝ) in
  let cos_15 := (Real.cos (Real.pi / 12)) in
  let sin_45 := (Real.sin (Real.pi / 4)) in
  sin_45 + Real.sin (θ * Real.pi / 180) = cos_15 → θ = 15 :=
by sorry

end least_positive_angle_l130_130334


namespace total_area_of_forest_and_fields_l130_130971

-- Define the problem in Lean 4
theorem total_area_of_forest_and_fields (r p : ℝ) (A_square A_rect A_forest : ℝ) (q : ℝ) :
  q = 4 * p ∧
  A_square = r^2 ∧
  A_rect = p * q ∧
  A_forest = 12 * 12 ∧
  A_forest = (A_square + A_rect + 45) →
  A_forest + A_square + A_rect = 135 :=
by
  intros h
  cases h with h1 h
  cases h with h2 h
  cases h with h3 h
  cases h with h4 h5
  sorry -- Proof step skipped

end total_area_of_forest_and_fields_l130_130971


namespace evaluate_expression_l130_130709

theorem evaluate_expression : - (16 / 4 * 8 - 70 + 4^2 * 7) = -74 := by
  sorry

end evaluate_expression_l130_130709


namespace count_integers_in_interval_l130_130809

theorem count_integers_in_interval : 
  ∃ (n : ℕ), (∀ (x : ℤ), (-2 ≤ x ∧ x ≤ 8 → ∃ (k : ℕ), k < n ∧ x = -2 + k)) ∧ n = 11 := 
by
  sorry

end count_integers_in_interval_l130_130809


namespace solve_quadratic_eq_l130_130924

theorem solve_quadratic_eq :
  ∃ (x : ℝ), 2 * x^2 - 5 * x + 2 = 0 ∧ (x = 2 ∨ x = 1 / 2) :=
by
  existsi 2
  simp
  { linarith }
  existsi 1 / 2
  simp
  { linarith }
#align solve_quadratic_eq lean_example_2

end solve_quadratic_eq_l130_130924


namespace inequality_transformation_l130_130837

theorem inequality_transformation (x y a : ℝ) (hxy : x < y) (ha : a < 1) : x + a < y + 1 := by
  sorry

end inequality_transformation_l130_130837


namespace rectangular_prism_volume_dependency_l130_130945

theorem rectangular_prism_volume_dependency (a : ℝ) (V : ℝ) (h : a > 2) :
  V = a * 2 * 1 → (∀ a₀ > 2, a ≠ a₀ → V ≠ a₀ * 2 * 1) :=
by
  sorry

end rectangular_prism_volume_dependency_l130_130945


namespace division_by_repeating_decimal_l130_130214

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l130_130214


namespace division_by_repeating_decimal_l130_130196

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l130_130196


namespace sams_charge_per_sheet_l130_130651

theorem sams_charge_per_sheet :
  ∃ x : ℝ, x = 1.5 ∧ (12 * 2.75 + 125) = (12 * x + 140) :=
by
  use 1.5
  split
  sorry

end sams_charge_per_sheet_l130_130651


namespace isosceles_triangle_perimeter_l130_130297

theorem isosceles_triangle_perimeter (a b : ℕ) (a = 6 ∧ b = 10) ∧ 
  (isosceles_triangle.has_side a ∧ isosceles_triangle.has_side b) : 
  (∃ c, isosceles_triangle.has_side c ∧ (c = 22 ∨ c = 26)) :=
    sorry

end isosceles_triangle_perimeter_l130_130297


namespace division_of_repeating_decimal_l130_130135

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l130_130135


namespace compound_interest_correct_l130_130717

-- Definitions for conditions from part a)
def initial_amount : ℕ := 6160
def interest_rate_year1 : ℕ := 10
def interest_rate_year2 : ℕ := 12

-- The proof statement showing that the final amount after 2 years is as expected
theorem compound_interest_correct :
  let A1 := initial_amount + (initial_amount * interest_rate_year1 / 100)
  let A2 := A1 + (A1 * interest_rate_year2 / 100)
  A2 = 7589.12 :=
by
  sorry

end compound_interest_correct_l130_130717


namespace eight_div_repeat_three_l130_130121

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l130_130121


namespace probability_even_sum_97_l130_130848

-- You don't need to include numbers since they are directly available in Lean's library
-- This will help to ensure broader compatibility and avoid namespace issues

theorem probability_even_sum_97 (m n : ℕ) (hmn : Nat.gcd m n = 1) 
  (hprob : (224 : ℚ) / 455 = m / n) : 
  m + n = 97 :=
sorry

end probability_even_sum_97_l130_130848


namespace total_coins_Zain_l130_130240

variable (quartersEmerie dimesEmerie nickelsEmerie : Nat)
variable (additionalCoins : Nat)

theorem total_coins_Zain (h_q : quartersEmerie = 6)
                         (h_d : dimesEmerie = 7)
                         (h_n : nickelsEmerie = 5)
                         (h_add : additionalCoins = 10) :
    let quartersZain := quartersEmerie + additionalCoins
    let dimesZain := dimesEmerie + additionalCoins
    let nickelsZain := nickelsEmerie + additionalCoins
    quartersZain + dimesZain + nickelsZain = 48 := by
  sorry

end total_coins_Zain_l130_130240


namespace find_initial_fee_l130_130799

noncomputable def initial_fee (F d: ℝ) : Prop :=
  (F + 0.40 * d = 0.60 * d)

theorem find_initial_fee : initial_fee 65 325 := 
by
  unfold initial_fee
  have h1 : 0.60 * 325 = 0.60 * 325 := rfl -- Just to show steps, Lean already knows this
  calc
    65 + 0.40 * 325 = 65 + 130 : by norm_num
    ... = 195 : by norm_num
    ... = 0.60 * 325 : by norm_num

end find_initial_fee_l130_130799


namespace rationalize_denominator_l130_130488

theorem rationalize_denominator : (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_denominator_l130_130488


namespace division_of_decimal_l130_130114

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l130_130114


namespace union_sets_example_l130_130791

theorem union_sets_example : ({0, 1} ∪ {2} : Set ℕ) = {0, 1, 2} := by 
  sorry

end union_sets_example_l130_130791


namespace largest_five_digit_integer_with_digit_product_proof_l130_130561

noncomputable def largest_five_digit_integer_with_digit_product : ℕ :=
  98752

theorem largest_five_digit_integer_with_digit_product_proof :
  ∃ n : ℕ, (n >= 10000) ∧ (n < 100000) ∧ 
           (∃ (digits : list ℕ), (n = digits.foldl (λ acc d, acc * 10 + d) 0) ∧
           (digits.prod = (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) ∧
           (n = 98752)) :=
by {
    let answer := 98752,
    use answer,
    split, { norm_num },
    split, { norm_num },
    use [9, 8, 7, 5, 2],
    split, { refl },
    split, {
        rw [list.prod_cons, list.prod_cons, list.prod_cons, list.prod_cons, list.prod_nil],
        norm_num,
    },
    refl,
}

end largest_five_digit_integer_with_digit_product_proof_l130_130561


namespace maximum_PC_length_l130_130793

theorem maximum_PC_length (PA PB PC : EuclideanSpace ℝ (Fin 2)) 
  (h1 : PA.dot PA + PB.dot PB = 4)
  (h2 : PA.dot PB = 0)
  (h3 : PC = (1 / 3) • PA + (2 / 3) • PB) : 
  ∥PC∥ ≤ 4 / 3 := 
begin
  sorry
end

end maximum_PC_length_l130_130793


namespace probability_no_rain_next_five_days_eq_1_over_243_l130_130963

theorem probability_no_rain_next_five_days_eq_1_over_243 :
  let p_rain : ℚ := 2 / 3 in
  let p_no_rain : ℚ := 1 - p_rain in
  let probability_no_rain_five_days : ℚ := p_no_rain ^ 5 in
  probability_no_rain_five_days = 1 / 243 :=
by
  sorry

end probability_no_rain_next_five_days_eq_1_over_243_l130_130963


namespace probability_of_at_least_6_consecutive_heads_l130_130267

-- Define the conditions
def flip_options : finset (fin 9 → bool) := 
  finset.univ

def at_least_6_consecutive_heads (seq : fin 9 → bool) : bool :=
  (seq 0 && seq 1 && seq 2 && seq 3 && seq 4 && seq 5) ||
  (seq 1 && seq 2 && seq 3 && seq 4 && seq 5 && seq 6) ||
  (seq 2 && seq 3 && seq 4 && seq 5 && seq 6 && seq 7) ||
  (seq 3 && seq 4 && seq 5 && seq 6 && seq 7 && seq 8)

-- Define the theorem to prove the probability
theorem probability_of_at_least_6_consecutive_heads :
  (flip_options.filter at_least_6_consecutive_heads).card = 11 / 512 :=
by
  sorry

end probability_of_at_least_6_consecutive_heads_l130_130267


namespace starting_number_is_10_l130_130982

axiom between_nums_divisible_by_10 (n : ℕ) : 
  (∃ start : ℕ, start ≤ n ∧ n ≤ 76 ∧ 
  ∀ m, start ≤ m ∧ m ≤ n → m % 10 = 0 ∧ 
  (¬ (76 % 10 = 0) → start = 10) ∧ 
  ((76 - (76 % 10)) / 10 = 6) )

theorem starting_number_is_10 
  (start : ℕ) 
  (h1 : ∃ n, (start ≤ n ∧ n ≤ 76 ∧ 
             ∀ m, start ≤ m ∧ m ≤ n → m % 10 = 0 ∧ 
             (n - start) / 10 = 6)):
  start = 10 :=
sorry

end starting_number_is_10_l130_130982


namespace line_through_focus_l130_130749

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 18) + (y^2 / 9) = 1

-- Define the focus
def focus : ℝ × ℝ := (3, 0)

-- Define the line intersecting the ellipse
def intersects_ellipse (l : ℝ → ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
    ellipse x1 y1 ∧ ellipse x2 y2 ∧
    y1 = l x1 ∧ y2 = l x2

-- Define the midpoint condition
def midpoint_inclination (l : ℝ → ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
    ellipse x1 y1 ∧ ellipse x2 y2 ∧
    y1 = l x1 ∧ y2 = l x2 ∧
    let M := ((x1 + x2) / 2, (y1 + y2) / 2) in
    M.2 / M.1 = -1

-- Prove the equation of the line given the conditions
theorem line_through_focus (l : ℝ → ℝ) :
  (intersects_ellipse l ∧ midpoint_inclination l) →
  ∃ a b c : ℝ, (a = 1) ∧ (b = -2) ∧ (c = -3) ∧ ∀ x y : ℝ, y = l x → (a * x + b * y + c = 0) :=
by
  sorry

end line_through_focus_l130_130749


namespace distance_between_trees_l130_130857

-- The conditions given
def trees_on_yard := 26
def yard_length := 500
def trees_at_ends := true

-- Theorem stating the proof
theorem distance_between_trees (h1 : trees_on_yard = 26) 
                               (h2 : yard_length = 500) 
                               (h3 : trees_at_ends = true) : 
  500 / (26 - 1) = 20 :=
by
  sorry

end distance_between_trees_l130_130857


namespace chicks_problem_solution_l130_130599

noncomputable def num_chicks_with_beaks_open_two_weeks_ago : ℕ := 11
noncomputable def num_chicks_growing_feathers_next_week : ℕ := 15
noncomputable def product_of_numbers : ℕ := num_chicks_with_beaks_open_two_weeks_ago * num_chicks_growing_feathers_next_week

theorem chicks_problem_solution :
    num_chicks_with_beaks_open_two_weeks_ago = 11 ∧
    num_chicks_growing_feathers_next_week = 15 ∧
    product_of_numbers = 165 := 
by
  unfold num_chicks_with_beaks_open_two_weeks_ago num_chicks_growing_feathers_next_week product_of_numbers
  simp
  split
  . rfl
  split 
  . rfl
  . rfl

end chicks_problem_solution_l130_130599


namespace division_by_repeating_decimal_l130_130216

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l130_130216


namespace ticket_cost_l130_130440

theorem ticket_cost (total_amount_spent number_of_tickets : ℕ) (h1 : total_amount_spent = 308) (h2 : number_of_tickets = 7) : total_amount_spent / number_of_tickets = 44 :=
by
  rw [h1, h2]
  exact rfl

end ticket_cost_l130_130440


namespace longest_segment_inside_cylinder_l130_130620

theorem longest_segment_inside_cylinder :
  ∀ (r h : ℝ), r = 5 → h = 12 → ∃ (d : ℝ), d = 2 * Real.sqrt 61 ∧ d = Real.sqrt ((2 * r) ^ 2 + h ^ 2) :=
by
  intros r h hr hh
  have : 2 * r = 10 := by rw [←hr]; norm_num
  have : h = 12 := by rw [←hh]
  use 2 * Real.sqrt 61
  simp [this]
  sorry

end longest_segment_inside_cylinder_l130_130620


namespace probability_neither_prime_nor_composite_l130_130244

theorem probability_neither_prime_nor_composite :
  let nums := (1: Nat) :: List.range (94) in
  let count_neither_prime_nor_composite := nums.count (λ n, n == 1) in
  let count_total := List.length nums in
  (count_neither_prime_nor_composite : ℝ) / (count_total : ℝ) = 1 / 95 :=
by
  sorry

end probability_neither_prime_nor_composite_l130_130244


namespace eight_div_repeating_three_l130_130149

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l130_130149


namespace cricket_runs_l130_130419

variable (A B C D E : ℕ)

theorem cricket_runs
  (h1 : (A + B + C + D + E) = 180)
  (h2 : D = E + 5)
  (h3 : A = E + 8)
  (h4 : B = D + E)
  (h5 : B + C = 107) :
  E = 20 := by
  sorry

end cricket_runs_l130_130419


namespace pump_fill_time_without_leak_l130_130280

-- Definitions used directly from conditions
def combined_fill_time : ℝ := 2 + 4/5  -- 2 and 4/5 hours, which is 2.8 hours
def leak_drain_time : ℝ := 7          -- The leak empties the tank in 7 hours

-- Question: Prove that the pump fills the tank in 2 hours
theorem pump_fill_time_without_leak (P : ℝ) :
  (1 / P) - (1 / leak_drain_time) = (1 / combined_fill_time) → P = 2 := by
  sorry

end pump_fill_time_without_leak_l130_130280


namespace eight_div_repeat_three_l130_130130

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l130_130130


namespace cafe_working_days_l130_130930

theorem cafe_working_days :
  (∀ (d : ℕ), d ∈ {i | 1 ≤ i ∧ i ≤ 27} → d % 7 ≠ 1 → True) →
  (({i | 1 ≤ i ∧ i ≤ 20}.card - {i | 1 ≤ i ∧ i ≤ 20 ∧ i % 7 = 1}.card) ≠ 18 ∨
   ({i | 10 ≤ i ∧ i ≤ 30}.card - {i | 10 ≤ i ∧ i ≤ 30 ∧ i % 7 = 1}.card) = 18) →
  ({i | 1 ≤ i ∧ i ≤ 27}.card - {i | 1 ≤ i ∧ i ≤ 27 ∧ i % 7 = 1}.card) = 23 :=
by
  -- proof logic here
  sorry

end cafe_working_days_l130_130930


namespace triangle_area_perpendicular_distances_l130_130020

/-- The area of the triangle formed by a point on the ellipse
such that the point's distances to the foci are perpendicular -/
theorem triangle_area_perpendicular_distances
  (a b c : ℝ)
  (h_ellipse : a = 5)
  (h_ellipse' : b = 3)
  (h_focal_distance : c = Real.sqrt (a ^ 2 - b ^ 2))
  (h_f1 : (F_1 : ℝ × ℝ) = (-c, 0))
  (h_f2 : (F_2 : ℝ × ℝ) = (c, 0))
  (P : ℝ × ℝ)
  (h_perpendicular : PF_1 ⊥ PF_2)
  (h_sum_distances : PF_1 + PF_2 = 2 * a) :
  let d1 := (P - F_1).norm,
      d2 := (P - F_2).norm in
  1 / 2 * d1 * d2 = 9 := sorry

end triangle_area_perpendicular_distances_l130_130020


namespace perimeter_of_congruent_rectangle_l130_130528

namespace RectangleProblem

-- Definitions based on the conditions
variables (y x : ℝ)
-- Assuming y and x are real numbers with x <= y as a condition for the problem
hypothesis (h : x ≤ y)

-- Definition of a perimeter of a rectangle given its dimensions
def rect_perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

-- Main problem statement
theorem perimeter_of_congruent_rectangle : rect_perimeter y x = 2 * y + x :=
  by
    sorry

end RectangleProblem

end perimeter_of_congruent_rectangle_l130_130528


namespace polynomial_g_value_at_1_l130_130461

theorem polynomial_g_value_at_1 (g : ℝ[X])
  (h_non_const : ¬ is_constant g)
  (h_eq : ∀ x : ℝ, x ≠ 0 → g.eval (x - 2) + g.eval x + g.eval (x + 2) = (g.eval x) ^ 2 / (4026 * x)) :
  g.eval 1 = 12078 :=
sorry

end polynomial_g_value_at_1_l130_130461


namespace rationalize_sqrt_three_sub_one_l130_130494

theorem rationalize_sqrt_three_sub_one :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_sqrt_three_sub_one_l130_130494


namespace triangle_APQ_is_isosceles_right_triangle_l130_130479

variables {A B C D E F G H K P Q : Type} [EuclideanGeometry]

-- Conditions for the problem
definition is_square_BCDE (BCDE : Type) : Prop := is_square BCDE B C D E
definition is_square_ACFG (ACFG : Type) : Prop := is_square ACFG A C F G
definition is_square_BAHK (BAHK : Type) : Prop := is_square BAHK B A H K
definition is_parallelogram_FCDQ (FCDQ : Type) : Prop := is_parallelogram FCDQ F C D Q
definition is_parallelogram_EBKP (EBKP : Type) : Prop := is_parallelogram EBKP E B K P

-- The theorem to prove 
theorem triangle_APQ_is_isosceles_right_triangle 
  (h1 : is_square_BCDE BCDE)
  (h2 : is_square_ACFG ACFG)
  (h3 : is_square_BAHK BAHK)
  (h4 : is_parallelogram_FCDQ FCDQ)
  (h5 : is_parallelogram_EBKP EBKP) : 
  is_isosceles_right_triangle A P Q :=
sorry

end triangle_APQ_is_isosceles_right_triangle_l130_130479


namespace quadratic_real_roots_l130_130730

theorem quadratic_real_roots (k : ℝ) : (x^2 - 3 * x + k = 0 → ℝ) → k ≤ 9 / 4 :=
by
  intros h_eq
  have discriminant := (-3)^2 - 4 * (1 : ℝ) * k
  have d_nonneg := discriminant ≥ 0
  rw [pow_two, mul_one] at discriminant
  linarith

end quadratic_real_roots_l130_130730


namespace cistern_fill_time_l130_130909

theorem cistern_fill_time (A B : ℝ) (hA : A = 1/60) (hB : B = 1/45) : (|A - B|)⁻¹ = 180 := by
  sorry

end cistern_fill_time_l130_130909


namespace eight_div_repeating_three_l130_130085

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l130_130085


namespace find_a_l130_130826

noncomputable def f (a x : ℝ) := (x - 1)^2 + a * x + Real.cos x

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) → 
  a = 2 :=
by
  sorry

end find_a_l130_130826


namespace magician_strategy_l130_130278

-- Definitions for boxes and their sets
inductive Box
| a_1 | a_2 | a_3 | a_4 | b_1 | b_2 | b_3 | b_4 | c_1 | c_2 | c_3 | c_4

open Box

def A := {a_1, a_2, a_3, a_4}
def B := {b_1, b_2, b_3, b_4}
def C := {c_1, c_2, c_3, c_4}

-- Conditions
def condition_1 (coins : Box × Box) : Prop :=
  ∀ b ∈ A ∪ B ∪ C, b ≠ coins.1 ∧ b ≠ coins.2

def condition_2 (message : Box) (coins : Box × Box) : Prop :=
  match message with
  | c_1 => coins.1 ∈ A ∧ coins.2 ∈ B ∧ coins.2 ≠ b_2 ∧ coins.2 ≠ b_4
  | c_2 => coins.1 ∈ A ∧ coins.2 ∈ B ∧ coins.2 ≠ b_1 ∧ coins.2 ≠ b_3
  | c_3 => coins.1 ∈ A ∧ coins.2 ∈ B ∧ coins.2 ≠ b_2 ∧ coins.2 ≠ b_3
  | c_4 => coins.1 ∈ A ∧ coins.2 ∈ B ∧ coins.2 ≠ b_1 ∧ coins.2 ≠ b_4
  | _ => False

-- Theorem statement
theorem magician_strategy :
  ∃ strategy : Box → Box × Box → Box,
    ∀ coins : Box × Box, coins.1 ≠ coins.2 →
    condition_1 coins →
    (∃ message : Box, condition_2 message coins ∧ strategy message coins = coins.1 ∧ strategy message coins = coins.2) :=
by {
 sorry
}

end magician_strategy_l130_130278


namespace division_of_decimal_l130_130108

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l130_130108


namespace integral_solution_l130_130302

noncomputable def integral_problem : Prop :=
  ∫ (x : ℝ) in 0..5, exp (sqrt ((5 - x) / (5 + x))) * (1 / ((5 + x) * sqrt (25 - x ^ 2))) = (1 - exp 1) / 5

theorem integral_solution : integral_problem := by
  sorry

end integral_solution_l130_130302


namespace initial_girls_l130_130859

theorem initial_girls (G : ℕ) (h : G + 682 = 1414) : G = 732 := 
by
  sorry

end initial_girls_l130_130859


namespace division_by_repeating_decimal_l130_130201

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l130_130201


namespace product_of_repeating_decimal_l130_130306

noncomputable def repeating_decimal := 1357 / 9999
def product_with_7 (x : ℚ) := 7 * x

theorem product_of_repeating_decimal :
  product_with_7 repeating_decimal = 9499 / 9999 :=
by sorry

end product_of_repeating_decimal_l130_130306


namespace regression_slope_l130_130604

variable {F : Type} [Field F]

theorem regression_slope (x y : Fin 8 → F) (h1 : ∑ i, x i = 6) (h2 : ∑ i, y i = 3) :
  let x_avg := (∑ i, x i) / 8
      y_avg := (∑ i, y i) / 8 in
  let a := y_avg - (1 / 3) * x_avg in
  a = (1 / 8) :=
by
  let x_avg := (∑ i, x i) / 8
  let y_avg := (∑ i, y i) / 8
  let a := y_avg - (1 / 3) * x_avg
  have h_avg_x : x_avg = 3 / 4 := by rw [h1, sum_of_nat_cast, sum_const, nat_cast_div]
  have h_avg_y : y_avg = 3 / 8 := by rw [h2, sum_of_nat_cast, sum_const, nat_cast_div]
  sorry

end regression_slope_l130_130604


namespace sum_of_coefficients_l130_130452

noncomputable def integral : ℝ :=
∫ x in 0..(Real.pi / 2), 3 * Real.sin x

theorem sum_of_coefficients :
  integral = 3 → 
  (x : ℝ) → (x = 1) →
  let n := integral in
  let expression := (x + (2 / x)) * (x - (2 / x)) ^ n in
  (expression = -3) :=
begin
  intro h,
  intro hx,
  simp [hx, h],
  sorry
end

end sum_of_coefficients_l130_130452


namespace least_positive_angle_l130_130333

theorem least_positive_angle (θ : ℝ) :
  (∃ θ, 0 < θ ∧ θ ≤ 360 ∧ cos (15 * real.pi / 180) = sin (45 * real.pi / 180) + sin (θ * real.pi / 180)) ↔ θ = 195 :=
by
  sorry

end least_positive_angle_l130_130333


namespace length_P_to_F₂_l130_130347

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)
def P : ℝ × ℝ := (-2, 3 / 2)

theorem length_P_to_F₂ : 
  let |PF₂| := sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)
  |PF₂| = 5 / 2 := 
by
  sorry

end length_P_to_F₂_l130_130347


namespace range_of_m_l130_130411

theorem range_of_m (x m : ℝ) (h1 : x + 3 = 3 * x - m) (h2 : x ≥ 0) : m ≥ -3 := by
  sorry

end range_of_m_l130_130411


namespace boys_without_calculators_l130_130471

theorem boys_without_calculators 
  (total_students : ℕ)
  (total_boys : ℕ)
  (students_with_calculators : ℕ)
  (girls_with_calculators : ℕ)
  (H_total_students : total_students = 30)
  (H_total_boys : total_boys = 20)
  (H_students_with_calculators : students_with_calculators = 25)
  (H_girls_with_calculators : girls_with_calculators = 18) :
  total_boys - (students_with_calculators - girls_with_calculators) = 13 :=
by
  sorry

end boys_without_calculators_l130_130471


namespace smallest_possible_value_of_c_l130_130891

/-- 
Given three integers \(a, b, c\) with \(a < b < c\), 
such that they form an arithmetic progression (AP) with the property that \(2b = a + c\), 
and form a geometric progression (GP) with the property that \(c^2 = ab\), 
prove that \(c = 2\) is the smallest possible value of \(c\).
-/
theorem smallest_possible_value_of_c :
  ∃ a b c : ℤ, a < b ∧ b < c ∧ 2 * b = a + c ∧ c^2 = a * b ∧ c = 2 :=
by
  sorry

end smallest_possible_value_of_c_l130_130891


namespace tan_690_eq_l130_130341

-- Conditions for the angles and their trigonometric properties.
def angle1 : ℝ := 690
def reducible_angle : ℕ := 360
def reduced_angle1 : ℝ := angle1 - reducible_angle -- One iteration of reduction
def reference_angle1 : ℝ := 30

-- Known value of tan 30 degrees
def tan_30 : ℝ := real.sqrt 3 / 3

-- The value of tan(angle) in the fourth quadrant
def tan_330 : ℝ := -tan_30

-- Statement to be proven
theorem tan_690_eq : real.tan (angle1 * real.pi / 180) = tan_330 :=
by
  -- This is the statement, so we use sorry to skip the proof.
  sorry

end tan_690_eq_l130_130341


namespace simplify_and_evaluate_expression_l130_130003

-- Define the expression
def simplified_expression (x : Real) : Real :=
  ((2 * x^2 + 2 * x) / (x^2 - 1) - (x^2 - x) / (x^2 - 2 * x + 1)) / (x / (x + 1))

-- Define the given x value
def given_x : Real := 1 - Real.sqrt 3

-- The theorem to be proved. 
theorem simplify_and_evaluate_expression :
  simplified_expression given_x = - (2 * Real.sqrt 3) / 3 + 1 :=
by 
  sorry

end simplify_and_evaluate_expression_l130_130003


namespace sqrt_400_div_2_l130_130224

theorem sqrt_400_div_2 : (Nat.sqrt 400) / 2 = 10 := by
  sorry

end sqrt_400_div_2_l130_130224


namespace Mike_initial_amount_eq_90_l130_130688

-- Definitions for initial amounts and weekly savings for Carol and Mike
def Carol_initial := 60
def Carol_savings_per_week := 9
def Mike_savings_per_week := 3

-- Invariant that states Carol and Mike have the same amount of money after 5 weeks
theorem Mike_initial_amount_eq_90 : 
  ∃ Mike_initial, 
    (Carol_initial + 5 * Carol_savings_per_week) = (Mike_initial + 5 * Mike_savings_per_week) 
    ↔ Mike_initial = 90 :=
begin
  -- Skipping the proof details, directly giving the theorem condition with 'sorry'
  sorry
end

end Mike_initial_amount_eq_90_l130_130688


namespace bowling_ball_remaining_volume_l130_130260

noncomputable def remaining_volume_after_drilling (ball_diameter : ℝ) (hole1_diameter : ℝ) (hole2_diameter : ℝ) (hole3_diameter : ℝ) (hole_depth : ℝ) : ℝ :=
  let r := ball_diameter / 2
  let V_sphere := (4 / 3) * π * r^3
  let V_cyl1 := π * (hole1_diameter / 2)^2 * hole_depth
  let V_cyl2 := π * (hole2_diameter / 2)^2 * hole_depth
  let V_cyl3 := π * (hole3_diameter / 2)^2 * hole_depth
  V_sphere - (V_cyl1 + V_cyl2 + V_cyl3)

theorem bowling_ball_remaining_volume :
  remaining_volume_after_drilling 24 3 3 4 10 = 2219 * π :=
by
  sorry

end bowling_ball_remaining_volume_l130_130260


namespace number_of_noncongruent_triangles_l130_130911

-- Define the points and properties using Lean's type system
section
variable {α : Type} [AffineSpace α]
variable (A B C M N O : α)

-- Conditions definitions
def right_triangle_with_midpoints (A B C M N O : α) :=
  let is_right_triangle := ∃ (angle : ℝ), angle = 90 ∧ (angle of BAC = angle)
  ∧ (M = midpoint A B ∧ N = midpoint B C ∧ O = midpoint C A)

-- The main statement to prove
theorem number_of_noncongruent_triangles (h : right_triangle_with_midpoints A B C M N O) :
    ∃! (n : ℕ), n = 5 :=
by sorry
end

end number_of_noncongruent_triangles_l130_130911


namespace division_by_repeating_decimal_l130_130217

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l130_130217


namespace circle_second_x_intercept_l130_130522

open Real

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem circle_second_x_intercept :
  ∀ (A B : ℝ × ℝ), 
  A = (0, 0) → B = (10, 8) → 
  ∃ x2 : ℝ, x2 = 10 ∧ (x2, 0) ≠ A :=
by
  intro A B hA hB
  let C := midpoint A B
  have hC : C = (5, 4) := sorry
  let r := distance A C
  have hr : r = sqrt 41 := sorry
  let eq_circle := λ x y, (x - C.1) ^ 2 + (y - C.2) ^ 2 = r ^ 2
  have h_eq_circle : eq_circle = λ x y, (x - 5) ^ 2 + (y - 4) ^ 2 = 41 := sorry
  have intercepts : ∃ x1 x2 : ℝ, eq_circle x1 0 ∧ eq_circle x2 0 ∧ x1 ≠ x2 ∧ (0, 0) = (x1, 0) :=
    by {
      use [0, 10],
      dsimp [eq_circle],
      split; {
        norm_num }
    }
  use 10,
  exact intercepts

end circle_second_x_intercept_l130_130522


namespace factorization_problem_l130_130019

theorem factorization_problem (a b c : ℤ)
  (h1 : ∀ x : ℝ, x^2 + 7 * x + 12 = (x + a) * (x + b))
  (h2 : ∀ x : ℝ, x^2 - 8 * x - 20 = (x - b) * (x - c)) :
  a - b + c = -9 :=
sorry

end factorization_problem_l130_130019


namespace meal_choice_combinations_l130_130416

-- Definitions for the conditions
def dishes : List String := ["Steamed Buns", "Baozi", "Noodles", "Fried Rice with Eggs"]
def students : List String := ["A", "B", "C", "D", "E"]

-- Given conditions
def at_least_one_dish (choices : List String) : Prop :=
  ∀ dish ∈ dishes, ∃ student ∈ students, choices.nth student = some dish

def baozis_limit (choices : List String) : Prop :=
  (choices.filter (λ x => x = "Baozi")).length ≤ 1

def student_A_condition (choices : List String) : Prop :=
  choices.nth 0 ≠ some "Fried Rice with Eggs"

-- Main statement to prove
theorem meal_choice_combinations :
  ∃ choices : List String, choices.length = 5 ∧
    at_least_one_dish choices ∧
    baozis_limit choices ∧
    student_A_condition choices ∧
    (number_of_combinations choices = 132) :=
sorry

end meal_choice_combinations_l130_130416


namespace largest_factorial_at_least_61_l130_130540

theorem largest_factorial_at_least_61 (k : ℕ) : 
  (fact (k - 2) + fact (k - 1) + fact k) % 61 = 0 → k ≥ 61 :=
by 
  sorry

end largest_factorial_at_least_61_l130_130540


namespace cards_playing_count_l130_130630

theorem cards_playing_count (total_cards : ℕ) (cards_kept_away : ℕ) (std_deck : total_cards = 52) 
  (kept_away : cards_kept_away = 2) :
  total_cards - cards_kept_away = 50 :=
by
  rw [std_deck, kept_away]
  norm_num

end cards_playing_count_l130_130630


namespace eight_div_repeating_three_l130_130147

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l130_130147


namespace geometric_progression_sum_lt_three_halves_l130_130699

noncomputable section

variable {a : ℕ → ℝ} (h1 : a 1 = 2) (h2 : a 2 = 3)
variable (H : ∀ n : ℕ, a (n + 2) = 3 * a (n + 1) - 2 * a n)

def d (n : ℕ) := a (n + 1) - a n

-- Question 1
theorem geometric_progression : ∀ n : ℕ, ∃ q : ℝ, d (n + 1) = q * d n :=
sorry

-- Question 2
def S (n : ℕ) := ∑ i in Finset.range n, 1 / a (i + 1)

theorem sum_lt_three_halves : ∀ n : ℕ, S n < 3 / 2 :=
sorry

end geometric_progression_sum_lt_three_halves_l130_130699


namespace Zain_coins_total_l130_130233

theorem Zain_coins_total :
  ∀ (quarters dimes nickels : ℕ),
  quarters = 6 →
  dimes = 7 →
  nickels = 5 →
  Zain_coins = quarters + 10 + (dimes + 10) + (nickels + 10) →
  Zain_coins = 48 :=
by intros quarters dimes nickels hq hd hn Zain_coins
   sorry

end Zain_coins_total_l130_130233


namespace problem_I_problem_II_l130_130737

open Real
open Complex

noncomputable def f (x : ℝ) : ℝ := sin x + sqrt 3 * cos x

theorem problem_I (x : ℝ) : ∃ T, (∀ y, f (y + T) = f y) ∧ T = 2 * π ∧ (∀ y, f y ≤ 2) ∧ (∃ y, f y = 2) :=
  by
  sorry

theorem problem_II (A : ℝ) (h : f (A + π / 6) = 2 / 3) : cos (2 * A) = -7 / 9 :=
  by
  sorry

end problem_I_problem_II_l130_130737


namespace rotation_transform_l130_130252

theorem rotation_transform (x y α : ℝ) :
    let x' := x * Real.cos α - y * Real.sin α
    let y' := x * Real.sin α + y * Real.cos α
    (x', y') = (x * Real.cos α - y * Real.sin α, x * Real.sin α + y * Real.cos α) := by
  sorry

end rotation_transform_l130_130252


namespace division_of_decimal_l130_130113

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l130_130113


namespace sum_of_divisors_equals_self_l130_130883

theorem sum_of_divisors_equals_self (S : Set ℕ) (hS : ∀ n, n ∈ S ↔ ∑ d in (Finset.filter (λ d, d < n ∧ d ∈ S) (Finset.divisors n)), d ≤ n) :
  ∀ k : ℕ, ∀ p : ℕ, (Nat.prime p ∧ ¬ 2 ∣ p) →
    (∑ d in Finset.filter (λ d, d < (2^k * p) ∧ d ∈ S) (Finset.divisors (2^k * p)), d = 2^k * p) ↔
    (∃ a x : ℕ, p = 2^(a + 1) - 1 ∧ k = x * (a + 1) - 1) :=
by
  sorry

end sum_of_divisors_equals_self_l130_130883


namespace treaty_signed_on_wednesday_l130_130010

-- This function calculates the weekday after a given number of days since a known weekday.
def weekday_after (start_day: ℕ) (days: ℕ) : ℕ :=
  (start_day + days) % 7

-- Given the problem conditions:
-- The war started on a Friday: 5th day of the week (considering Sunday as 0)
def war_start_day_of_week : ℕ := 5

-- The number of days after which the treaty was signed
def days_until_treaty : ℕ := 926

-- Expected final day (Wednesday): 3rd day of the week (considering Sunday as 0)
def treaty_day_of_week : ℕ := 3

-- The theorem to be proved:
theorem treaty_signed_on_wednesday :
  weekday_after war_start_day_of_week days_until_treaty = treaty_day_of_week :=
by
  sorry

end treaty_signed_on_wednesday_l130_130010


namespace quadratic_permutation_n_l130_130695

theorem quadratic_permutation_n (n : ℕ) : 
  (∀ (a : Fin n.succ → ℕ), 
    (∃ k : ℕ, k ≤ n ∧ ∃ m : ℕ, (∑ i in Finset.range (k+1), a ⟨i, _⟩) = m * m)) 
  ↔ ∃ k : ℕ, n = ( (3 + 2 * Real.sqrt 2)^k + (3 - 2 * Real.sqrt 2)^k - 2 ) / 4 := 
by sorry

end quadratic_permutation_n_l130_130695


namespace runners_meet_l130_130057

theorem runners_meet (T : ℕ) 
  (h1 : T > 4) 
  (h2 : Nat.lcm 2 (Nat.lcm 4 T) = 44) : 
  T = 11 := 
sorry

end runners_meet_l130_130057


namespace find_largest_k_l130_130970

def tower_function : ℕ → ℕ
| 0 := 3  -- T(1) according to the indexing in Lean
| (n + 1) := 3 ^ tower_function n

def T (n : ℕ) : ℕ := tower_function (n - 1)  -- Adjusted for indexing

noncomputable def A : ℕ := (T 5) ^ (T 5)

noncomputable def B : ℕ := (T 5) ^ A

def log_base (b x : ℕ) : ℕ :=
  if h : 1 < b ∧ b ^ 0 ≤ x then
    Nat.find (λ y, x < b ^ (y + 1))
  else
    0

def largest_k (B : ℕ) : ℕ :=
  Nat.find (λ k, (List.foldr (λ _ y, log_base 3 y) B (List.range k) = 0))

theorem find_largest_k :
  largest_k B = 6 :=
  sorry

end find_largest_k_l130_130970


namespace division_by_repeating_decimal_l130_130204

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l130_130204


namespace inequality_solution_l130_130331

theorem inequality_solution (x : ℝ) :
    \frac{2 * x + 1}{x - 2} + \frac{x - 3}{3 * x} ≥ 4 ↔ x ∈ set.Ioo (-2/5) 0 ∪ set.Ioo 2 3 ∪ set.Icc 3 3 :=
sorry

end inequality_solution_l130_130331


namespace divide_by_repeating_decimal_l130_130100

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l130_130100


namespace locus_of_intersection_l130_130378

def parabola (x y : ℝ) (p : ℝ) : Prop := y^2 = 2 * p * x

def on_parabola (p : ℝ) (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 p ∧ parabola B.1 B.2 p

def perpendicular_tangents (p : ℝ) (A B : ℝ × ℝ) : Prop :=
  let slope_tangent_A := p / A.2 in
  let slope_tangent_B := p / B.2 in
  slope_tangent_A * slope_tangent_B = -1

theorem locus_of_intersection (A B C : ℝ × ℝ) (p : ℝ) (h : p = 4)
  (h1 : on_parabola p A B) 
  (h2 : perpendicular_tangents p A B) :
  C.1 = -1 :=
sorry

end locus_of_intersection_l130_130378


namespace number_of_outfits_l130_130232

def shirts : ℕ := 5
def hats : ℕ := 3

theorem number_of_outfits : shirts * hats = 15 :=
by 
  -- This part intentionally left blank since no proof required.
  sorry

end number_of_outfits_l130_130232


namespace find_m_parallel_vectors_l130_130795

theorem find_m_parallel_vectors 
  (m : ℝ)
  (a b : ℝ × ℝ)
  (h_a : a = (m, 2))
  (h_b : b = (2, -3))
  (parallel : ∃ k : ℝ, ((m+2, -1) = k • (m-2, 5))) : m = -4/3 :=
begin
  sorry,
end

end find_m_parallel_vectors_l130_130795


namespace least_positive_angle_l130_130335

theorem least_positive_angle:
  let θ := (15: ℝ) in
  let cos_15 := (Real.cos (Real.pi / 12)) in
  let sin_45 := (Real.sin (Real.pi / 4)) in
  sin_45 + Real.sin (θ * Real.pi / 180) = cos_15 → θ = 15 :=
by sorry

end least_positive_angle_l130_130335


namespace rationalize_denominator_l130_130489

theorem rationalize_denominator : (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_denominator_l130_130489


namespace time_to_fill_tank_l130_130593

-- Define the rates of the pipes
def rateA : ℝ := 1 / 6
def rateB : ℝ := 1 / 4
def rateC : ℝ := -1 / 12

-- Define the combined rate
def combinedRate : ℝ := rateA + rateB + rateC

-- Define the proof problem
theorem time_to_fill_tank : combinedRate = 1 / 3 ∧ (1 / combinedRate) = 3 :=
by {
  have rateA_def : rateA = 1 / 6 := rfl,
  have rateB_def : rateB = 1 / 4 := rfl,
  have rateC_def : rateC = -1 / 12 := rfl,
  have combined_def : combinedRate = rateA + rateB + rateC := rfl,
  rw [rateA_def, rateB_def, rateC_def] at combined_def,
  sorry
}

end time_to_fill_tank_l130_130593


namespace albert_maturity_amount_l130_130294

noncomputable def compound_interest_maturity (P : ℝ) (years : ℕ) (rates : ℕ → ℝ) : ℝ :=
  (List.range years).foldl (λ acc year, acc * (1 + rates year)) P

theorem albert_maturity_amount :
  compound_interest_maturity 6500 5 (λ n, [0.065, 0.07, 0.06, 0.075, 0.08].nth! n) = 9113.43 :=
by
  sorry

end albert_maturity_amount_l130_130294


namespace triangle_angle_relationship_l130_130359

theorem triangle_angle_relationship
  (A B C D E : Type)
  [triangle ABC]
  (h1 : AB = BC)
  (h2 : angle BCD = 30)
  (h3 : bisects CE C)
  (h4 : angle BCE = x) :
  x = 50 ∧ angle A = 100 :=
sorry

end triangle_angle_relationship_l130_130359


namespace sum_of_p_factors_l130_130229

theorem sum_of_p_factors (p q : ℕ) (hpq : p * q = 75) : 
  ∃ (S : Finset ℕ), S = {1, 3, 5, 15, 25, 75} ∧ S.sum = 124 :=
begin
  sorry
end

end sum_of_p_factors_l130_130229


namespace range_of_k_monotonic_range_of_k_negative_l130_130742

-- Define the quadratic function and its properties
variables (a b k : ℝ) (f g : ℝ → ℝ)
variables (h_positive_a : a > 0)
variables (h_f_def : ∀ x, f x = a * x^2 + b * x + 1)
variables (h_f_neg1 : f (-1) = 0)
variables (h_f_nonneg : ∀ x, f x ≥ 0)
variables (g_def : ∀ x, g x = f x - k * x)

-- Theorem 1: g(x) is monotonic on [-2, 2]
theorem range_of_k_monotonic :
  (∀ x ∈ set.Icc (-2 : ℝ) 2, (x:ℝ).has_deriv_at g x) → 
  ({ k | ∃ (f : ℝ → ℝ) (a b : ℝ), 
   (h_positive_a : a > 0) ∧ 
   (h_f_def : ∀ x, f x = a * x^2 + b * x + 1) ∧ 
   (h_f_neg1 : f (-1) = 0) ∧ 
   (h_f_nonneg : ∀ x, f x ≥ 0) ∧ 
   (g_def : ∀ x, g x = f x - k * x) ∧ 
   (∀ x ∈ set.Icc (-2:ℝ) 2, (x:ℝ).has_deriv_at g x) } = { k | k ≤ -2 ∨ k ≥ 6 }) :=
sorry

-- Theorem 2: g(x) < 0 on [1, 2]
theorem range_of_k_negative :
  (∀ x ∈ set.Icc (1 : ℝ) 2, g x < 0) → 
  ({ k | ∃ (f : ℝ → ℝ) (a b : ℝ), 
   (h_positive_a : a > 0) ∧ 
   (h_f_def : ∀ x, f x = a * x^2 + b * x + 1) ∧ 
   (h_f_neg1 : f (-1) = 0) ∧ 
   (h_f_nonneg : ∀ x, f x ≥ 0) ∧ 
   (g_def : ∀ x, g x = f x - k * x) ∧ 
   (∀ x ∈ set.Icc (1:ℝ) 2, g x < 0) } = { k | k > 9 / 2 }) :=
sorry

end range_of_k_monotonic_range_of_k_negative_l130_130742


namespace range_of_a_l130_130358

def f (a x : ℝ) : ℝ := 3 * a * x - 2 * a + 1

theorem range_of_a (a : ℝ) :
  (∃ x0 : ℝ, -1 < x0 ∧ x0 < 1 ∧ f a x0 = 0) →
  (a ∈ set.Ioo (-∞) (-1) ∪ set.Ioo (1/5) (∞)) := by
  sorry

end range_of_a_l130_130358


namespace max_gcd_of_sequence_l130_130535

def term (n : ℕ) : ℕ := 100 + n^2

def gcd_terms (n : ℕ) : ℕ := Int.gcd (term n) (term (n + 1))

theorem max_gcd_of_sequence : ∃ n : ℕ, ∀ m : ℕ, gcd_terms m ≤ 401 ∧ gcd_terms 200 = 401 :=
by
  sorry

end max_gcd_of_sequence_l130_130535


namespace price_per_slice_is_five_l130_130279

-- Definitions based on the given conditions
def pies_sold := 9
def slices_per_pie := 4
def total_revenue := 180

-- Definition derived from given conditions
def total_slices := pies_sold * slices_per_pie

-- The theorem to prove
theorem price_per_slice_is_five :
  total_revenue / total_slices = 5 :=
by
  sorry

end price_per_slice_is_five_l130_130279


namespace determine_k_and_a_n_and_T_n_l130_130395

noncomputable def S_n (n : ℕ) (k : ℝ) : ℝ := -0.5 * n^2 + k * n

/-- Given the sequence S_n with sum of the first n terms S_n := -1/2 n^2 + k*n,
where k is a positive natural number. The maximum value of S_n is 8. -/
theorem determine_k_and_a_n_and_T_n (k : ℝ) (h : k = 4) :
  (∀ n : ℕ, S_n n k ≤ 8) ∧ 
  (∀ n : ℕ, ∃ a : ℝ, a = 9/2 - n) ∧
  (∀ n : ℕ, ∃ T : ℝ, T = 4 - (n + 2)/2^(n-1)) :=
by
  sorry

end determine_k_and_a_n_and_T_n_l130_130395


namespace cos_tan_sin_identity_sin_cos_identity_l130_130255

section problem1
  theorem cos_tan_sin_identity :
    cos (9 * π / 4) + tan (-π / 4) + sin (21 * π) = (real.sqrt 2 / 2) - 1 :=
  by sorry
end problem1

section problem2
  variables (θ : ℝ)
  
  theorem sin_cos_identity (h : sin θ = 2 * cos θ) :
    (sin θ ^ 2 + 2 * sin θ * cos θ) / (2 * sin θ ^ 2 - cos θ ^ 2) = 8 / 7 :=
  by sorry
end problem2

end cos_tan_sin_identity_sin_cos_identity_l130_130255


namespace total_area_of_forest_and_fields_l130_130972

-- Define the problem in Lean 4
theorem total_area_of_forest_and_fields (r p : ℝ) (A_square A_rect A_forest : ℝ) (q : ℝ) :
  q = 4 * p ∧
  A_square = r^2 ∧
  A_rect = p * q ∧
  A_forest = 12 * 12 ∧
  A_forest = (A_square + A_rect + 45) →
  A_forest + A_square + A_rect = 135 :=
by
  intros h
  cases h with h1 h
  cases h with h2 h
  cases h with h3 h
  cases h with h4 h5
  sorry -- Proof step skipped

end total_area_of_forest_and_fields_l130_130972


namespace factorization_a_minus_b_l130_130937

theorem factorization_a_minus_b (a b : ℤ) (h1 : 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) : a - b = -7 :=
by
  sorry

end factorization_a_minus_b_l130_130937


namespace binomial_square_coefficients_l130_130404

noncomputable def a : ℝ := 13.5
noncomputable def b : ℝ := 18

theorem binomial_square_coefficients (c d : ℝ) :
  (∀ x : ℝ, 6 * x ^ 2 + 18 * x + a = (c * x + d) ^ 2) ∧ 
  (∀ x : ℝ, 3 * x ^ 2 + b * x + 4 = (c * x + d) ^ 2)  → 
  a = 13.5 ∧ b = 18 := sorry

end binomial_square_coefficients_l130_130404


namespace orange_profit_loss_l130_130634

variable (C : ℝ) -- Cost price of one orange in rupees

-- Conditions as hypotheses
theorem orange_profit_loss :
  (1 / 16 - C) / C * 100 = 4 :=
by
  have h1 : 1.28 * C = 1 / 12 := sorry
  have h2 : C = 1 / (12 * 1.28) := sorry
  have h3 : C = 1 / 15.36 := sorry
  have h4 : (1/16 - C) = 1 / 384 := sorry
  -- Proof of main statement here
  sorry

end orange_profit_loss_l130_130634


namespace probability_at_least_6_heads_in_9_flips_l130_130270

def fair_coin_flips := 9
def total_outcomes := 512

/-- The probability of obtaining at least 'k' consecutive heads in 'n' flips of a fair coin. -/
def at_least_k_consecutive_heads_probability (k n : ℕ) : ℚ :=
∑ i in ((list.range (n - k + 1)).map (λ start_pos, 
  let end_pos := start_pos + k - 1 in 
  ((2 : ℚ)^(start_pos) + (n - end_pos - 1)))) 
  , (1 / (2^n : ℚ))

theorem probability_at_least_6_heads_in_9_flips :
  at_least_k_consecutive_heads_probability 6 9 = 49 / 512 := sorry

end probability_at_least_6_heads_in_9_flips_l130_130270


namespace units_digit_of_17_pow_549_l130_130572

theorem units_digit_of_17_pow_549 : (17 ^ 549) % 10 = 7 :=
by {
  -- Provide the necessary steps or strategies to prove the theorem
  sorry
}

end units_digit_of_17_pow_549_l130_130572


namespace ratio_of_areas_l130_130925

-- Define the squares and their side lengths
def Square (side_length : ℝ) := side_length * side_length

-- Define the side lengths of Square C and Square D
def side_C (x : ℝ) : ℝ := x
def side_D (x : ℝ) : ℝ := 3 * x

-- Define their areas
def area_C (x : ℝ) : ℝ := Square (side_C x)
def area_D (x : ℝ) : ℝ := Square (side_D x)

-- The statement to prove
theorem ratio_of_areas (x : ℝ) (hx : x ≠ 0) : area_C x / area_D x = 1 / 9 := by
  sorry

end ratio_of_areas_l130_130925


namespace division_by_repeating_decimal_l130_130210

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l130_130210


namespace equation_simplification_isosceles_triangle_perimeter_tiling_feasibility_rotational_symmetry_l130_130021

-- 1. Equation simplification
theorem equation_simplification (x : ℝ) :
  (3 * (x + 1)) = (12 * x - (5 * x - 1)) := sorry

-- 2. Isosceles triangle perimeter
theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 5) (h2 : b = 9) :
  ¬ (a + a + b = 23 ∨ a + b + b = 23) := sorry

-- 3. Tiling feasibility
theorem tiling_feasibility :
  ¬ (tiles ("equilateral triangle") ∧ tiles ("octagon")) := sorry

-- 4. Rotational symmetry
theorem rotational_symmetry :
  symmetric_under_rotation ("equilateral triangle") ∧ symmetric_under_rotation ("line segment") := sorry

end equation_simplification_isosceles_triangle_perimeter_tiling_feasibility_rotational_symmetry_l130_130021


namespace standard_parabola_with_symmetry_axis_eq_1_l130_130340

-- Define the condition that the axis of symmetry is x = 1
def axis_of_symmetry_x_eq_one (x : ℝ) : Prop :=
  x = 1

-- Define the standard equation of the parabola y^2 = -4x
def standard_parabola_eq (y x : ℝ) : Prop :=
  y^2 = -4 * x

-- Theorem: Prove that given the axis of symmetry of the parabola is x = 1,
-- the standard equation of the parabola is y^2 = -4x.
theorem standard_parabola_with_symmetry_axis_eq_1 : ∀ (x y : ℝ),
  axis_of_symmetry_x_eq_one x → standard_parabola_eq y x :=
by
  intros
  sorry

end standard_parabola_with_symmetry_axis_eq_1_l130_130340


namespace number_of_boys_l130_130975

-- Definitions
def total_students : ℕ := 12
def fraction_boys : ℚ := 2 / 3

-- Statement to prove
theorem number_of_boys (S : ℕ) (F : ℚ) (H : S = total_students ∧ F = fraction_boys) : ℕ :=
  (S * (F : ℚ)).natAbs

#eval number_of_boys total_students fraction_boys ⟨rfl, rfl⟩ -- Expected: 8

end number_of_boys_l130_130975


namespace bob_mother_twice_age_2040_l130_130473

theorem bob_mother_twice_age_2040 :
  ∀ (bob_age_2010 mother_age_2010 : ℕ), 
  bob_age_2010 = 10 ∧ mother_age_2010 = 50 →
  ∃ (x : ℕ), (mother_age_2010 + x = 2 * (bob_age_2010 + x)) ∧ (2010 + x = 2040) :=
by
  sorry

end bob_mother_twice_age_2040_l130_130473


namespace sum_of_11378_and_121_is_odd_l130_130967

theorem sum_of_11378_and_121_is_odd (h1 : Even 11378) (h2 : Odd 121) : Odd (11378 + 121) :=
by
  sorry

end sum_of_11378_and_121_is_odd_l130_130967


namespace projective_transformation_l130_130362

-- Definitions of the initial objects: line, circle, point M
variable (l : Line) (C : Circle) (M : Point)
  (hM_on_circle : M ∈ C)
  (hM_not_on_line : ¬ M ∈ l)

-- Define the projection P_M
def P_M (X : Point) : Point := sorry -- Projection logic from X to the circle

-- Define the motion R preserving the circle
def R (pt : Point) : Point := sorry -- Motion logic, either rotation or reflection

-- Define the inverse of the projection P_M
def P_M_inv (pt : Point) : Point := sorry

-- Prove the transformation is projective
theorem projective_transformation :
  is_projective (P_M_inv ∘ R ∘ P_M) :=
sorry

end projective_transformation_l130_130362


namespace total_value_after_3_years_l130_130263

noncomputable def value_after_years (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

def machine1_initial_value : ℝ := 2500
def machine1_depreciation_rate : ℝ := 0.05
def machine2_initial_value : ℝ := 3500
def machine2_depreciation_rate : ℝ := 0.07
def machine3_initial_value : ℝ := 4500
def machine3_depreciation_rate : ℝ := 0.04
def years : ℕ := 3

theorem total_value_after_3_years :
  value_after_years machine1_initial_value machine1_depreciation_rate years +
  value_after_years machine2_initial_value machine2_depreciation_rate years +
  value_after_years machine3_initial_value machine3_depreciation_rate years = 8940 :=
by
  sorry

end total_value_after_3_years_l130_130263


namespace even_function_has_a_equal_2_l130_130822

noncomputable def f (a x : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_has_a_equal_2 (a : ℝ) :
  (∀ x : ℝ, f a (-x) = f a x) → a = 2 :=
sorry

end even_function_has_a_equal_2_l130_130822


namespace det_divisible_by_k_pow_n_minus_one_l130_130927

variable {n : ℕ}
variable {a : Fin n → ℝ}
variable {k : ℝ}

def m (i j : Fin n) : ℝ :=
  if i = j then a i * a i + k else a i * a j

theorem det_divisible_by_k_pow_n_minus_one :
  let M := λ i j : Fin n, m i j
  ∃ f : ℝ, Matrix.det (Matrix.of fun (i j : Fin n) => M i j) = k^(n-1) * f ∧
           f = (k + ∑ i, (a i)^2) :=
by
  sorry

end det_divisible_by_k_pow_n_minus_one_l130_130927


namespace find_f_log2_3_l130_130784

def f : ℝ → ℝ :=
by
  intro x
  exact if x ≥ 1 then 2^x else sorry -- function recursive definition f(x + 1)

theorem find_f_log2_3 : f (Real.log 3 / Real.log 2) = 3 :=
sorry

end find_f_log2_3_l130_130784


namespace find_a_if_even_function_l130_130835

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem find_a_if_even_function :
  (∃ a : ℝ, is_even_function (f a)) → ∃ a : ℝ, a = 2 :=
by
  intros h
  sorry

end find_a_if_even_function_l130_130835


namespace non_shaded_perimeter_6_l130_130521

theorem non_shaded_perimeter_6 
  (area_shaded : ℝ) (area_large_rect : ℝ) (area_extension : ℝ) (total_area : ℝ)
  (non_shaded_area : ℝ) (perimeter : ℝ) :
  area_shaded = 104 → 
  area_large_rect = 12 * 8 → 
  area_extension = 5 * 2 → 
  total_area = area_large_rect + area_extension → 
  non_shaded_area = total_area - area_shaded → 
  non_shaded_area = 2 → 
  perimeter = 2 * (2 + 1) → 
  perimeter = 6 := 
by 
  sorry

end non_shaded_perimeter_6_l130_130521


namespace fixed_point_l130_130932

theorem fixed_point (x y : ℝ) (hx : x > 0) (h1 : x ≠ 1) (f : ℝ → ℝ) (hf : ∀ x, f x = Real.log (3 * x - 2) / Real.log x + 2) :
  f 1 = 2 :=
by
  have : 3 * 1 - 2 = 1 := by norm_num
  rw [hf, this, Real.log_one, div_zero, zero_add]
  exact rfl

end fixed_point_l130_130932


namespace min_tiles_to_change_l130_130312

def letter_frequency (s : String) : List (Char × Nat) :=
  (s.foldr (fun c m => m.update c (m.findD c 0 + 1)) Std.RBMap.empty).toList

def difference_in_frequencies (freq1 freq2 : List (Char × Nat)) : Nat :=
  let freq_map1 := Std.RBMap.ofList freq1
  let freq_map2 := Std.RBMap.ofList freq2
  freq_map1.fold (fun k v acc => (acc + if freq_map2.contains k then abs (v - freq_map2.findD k 0) else v)) 0

def total_changes (s1 s2 : String) : Nat :=
  difference_in_frequencies (letter_frequency s1) (letter_frequency s2)

theorem min_tiles_to_change
  (s1 : "Central Michigan University")
  (s2 : "Carnegie Mellon University") :
  total_changes s1 s2 = 9 :=
sorry

end min_tiles_to_change_l130_130312


namespace division_by_repeating_decimal_l130_130199

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l130_130199


namespace cos_thm_l130_130352

variable (θ : ℝ)

-- Conditions
def condition1 : Prop := 3 * Real.sin (2 * θ) = 4 * Real.tan θ
def condition2 : Prop := ∀ k : ℤ, θ ≠ k * Real.pi

-- Prove that cos 2θ = 1/3 given the conditions
theorem cos_thm (h1 : condition1 θ) (h2 : condition2 θ) : Real.cos (2 * θ) = 1 / 3 :=
by
  sorry

end cos_thm_l130_130352


namespace books_left_over_l130_130265

theorem books_left_over (boxes : ℕ) (books_per_box : ℕ) (new_box_capacity : ℕ)
  (h1 : boxes = 2020) (h2 : books_per_box = 42) (h3 : new_box_capacity = 45) : 
  (boxes * books_per_box) % new_box_capacity = 30 :=
by
  rw [h1, h2, h3]
  have total_books : ℕ := 2020 * 42
  have remainder : ℕ := total_books % 45
  show remainder = 30
  sorry

end books_left_over_l130_130265


namespace Eight_div_by_repeating_decimal_0_3_l130_130160

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l130_130160


namespace min_tickets_required_l130_130996
-- Importing the necessary library

-- Defining the problem
theorem min_tickets_required (n : ℕ) (k : ℕ) (total : ℕ) : 
  (total = 49) ∧ (n = 6) ∧ (k = 8) → ∃ (tickets: finset (finset ℕ)), 
  (tickets.card = k) ∧ (∀ (draw : finset ℕ), draw.card = n → ∃ t ∈ tickets, t ∩ draw ≠ ∅) := 
by
  sorry

end min_tickets_required_l130_130996


namespace median_of_set_is_eight_l130_130365

theorem median_of_set_is_eight (x y : ℝ) :
  (7 + 8 + 9 + x + y) / 5 = 8 →
  (∀ (s : Finset ℝ), s = {7, 8, 9, x, y} → s.median = 8) := 
by 
  sorry

end median_of_set_is_eight_l130_130365


namespace train_speed_l130_130291

theorem train_speed :
  ∃ V : ℝ,
    (∃ L : ℝ, L = V * 18) ∧ 
    (∃ L : ℝ, L + 260 = V * 31) ∧ 
    V * 3.6 = 72 := by
  sorry

end train_speed_l130_130291


namespace total_rattlesnakes_l130_130047

-- Definitions based on the problem's conditions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def other_snakes : ℕ := total_snakes - (pythons + boa_constrictors)

-- Statement to be proved
theorem total_rattlesnakes : other_snakes = 40 := 
by 
  -- Skipping the proof
  sorry

end total_rattlesnakes_l130_130047


namespace area_of_quadrilateral_ABCD_l130_130475

-- The definitions are based on the conditions provided
variable (A B C D : Type)
variable [MetricSpace A]
variable (dist : A → A → ℝ) -- distance function 

def cyclist_speed_asphalt := 15   -- cyclist's speed on asphalt roads in km/h
def travel_time := 2             -- travel time on asphalt roads in hours
def distance_BX (X : A) : ℝ := cyclist_speed_asphalt * travel_time -- BX distance for X in {A, C, D}

-- Introduce the distances based on the provided conditions
def BD := 30     -- Distance BD is 30 km

-- Given that point B travels equidistantly to points A, C, and D
def is_dirt_road (X Y : A) : Prop := dist X Y = 30

-- Main statement to be proved in Lean
theorem area_of_quadrilateral_ABCD
  (h_AB : is_dirt_road B A)
  (h_BC : is_dirt_road B C)
  (h_CD : is_dirt_road C D)
  (h_DA : is_dirt_road D A)
  (h_BD : dist B D = BD)
  : quadrilateral_area A B C D = 450 := 
sorry  -- proof to be derived later

end area_of_quadrilateral_ABCD_l130_130475


namespace division_of_decimal_l130_130105

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l130_130105


namespace shortest_distance_point_to_parabola_l130_130338

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ :=
  (x, x^2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem shortest_distance_point_to_parabola :
  distance (7, 15) (3, 9) = 2 * real.sqrt 13 :=
by
  sorry

end shortest_distance_point_to_parabola_l130_130338


namespace area_of_shaded_region_l130_130643

theorem area_of_shaded_region (s : ℝ) (r : ℝ) (h1 : s = 3) (h2 : r = 1.5) :
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s ^ 2 in
  let semicircle_area := 0.5 * Real.pi * r ^ 2 in
  let total_semicircle_area := 3 * semicircle_area in
  let quartercircle_area := 0.25 * Real.pi * r ^ 2 in
  let total_quartercircle_area := 3 * quartercircle_area in
  let total_circle_area := total_semicircle_area + total_quartercircle_area in
  let shaded_area := hexagon_area - total_circle_area in
  shaded_area = 13.5 * Real.sqrt 3 - 5.0625 * Real.pi :=
by
  intros
  rw [h1, h2]
  sorry

end area_of_shaded_region_l130_130643


namespace vertex_of_parabola_l130_130015

theorem vertex_of_parabola : 
  ∀ x, (3 * (x - 1)^2 + 2) = ((x - 1)^2 * 3 + 2) := 
by {
  -- The proof steps would go here
  sorry -- Placeholder to signify the proof steps are omitted
}

end vertex_of_parabola_l130_130015


namespace max_value_fraction_l130_130753

theorem max_value_fraction (a b : ℝ) (h1 : ab = 1) (h2 : a > b) (h3 : b ≥ 2/3) :
  ∃ C, C = 30 / 97 ∧ (∀ x y : ℝ, (xy = 1) → (x > y) → (y ≥ 2/3) → (x - y) / (x^2 + y^2) ≤ C) :=
sorry

end max_value_fraction_l130_130753


namespace find_cubic_sum_l130_130736

theorem find_cubic_sum
  {a b : ℝ}
  (h1 : a^5 - a^4 * b - a^4 + a - b - 1 = 0)
  (h2 : 2 * a - 3 * b = 1) :
  a^3 + b^3 = 9 :=
by
  sorry

end find_cubic_sum_l130_130736


namespace division_of_repeating_decimal_l130_130077

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l130_130077


namespace fraction_equivalent_to_repeating_decimal_l130_130719

/- Definitions of the given conditions -/
def s : ℝ := 36 / 99
def a : ℝ := 36 / 100
def r : ℝ := 1 / 100

/- Lean statement of the problem -/
theorem fraction_equivalent_to_repeating_decimal : s = 4 / 11 := 
sorry

end fraction_equivalent_to_repeating_decimal_l130_130719


namespace number_of_integer_roots_of_32_is_2_l130_130351

theorem number_of_integer_roots_of_32_is_2 :
  (count (λ n : ℕ, n > 0 ∧ is_integer (2^(5/n))) {1, 2, 3, ...}) = 2 := by
sorry

end number_of_integer_roots_of_32_is_2_l130_130351


namespace distance_C_to_D_is_8_point_9_l130_130538

-- Define the conditions for the smaller and larger squares
def smaller_square_perimeter : ℝ := 8
def larger_square_area : ℝ := 36

-- Define a theorem to prove the distance from C to D
theorem distance_C_to_D_is_8_point_9 :
  let side_s := smaller_square_perimeter / 4 in
  let side_L := Real.sqrt larger_square_area in
  let horizontal_distance := side_s + side_L in
  let vertical_distance := side_L - side_s in
  let hypotenuse := Real.sqrt (horizontal_distance^2 + vertical_distance^2) in
  Real.approx hypotenuse 8.9 0.1 := -- the hypotenuse is approximately 8.9 to the nearest tenth
sorry

end distance_C_to_D_is_8_point_9_l130_130538


namespace inverse_function_point_l130_130843

theorem inverse_function_point (f : ℝ → ℝ) (hf : Function.Bijective f) :
  Function.inv_fun f (1 : ℝ) = (5 : ℝ) → f (5 : ℝ) = (1 : ℝ) :=
by
  intro h
  have h1 := Function.left_inverse_inv_fun hf
  rw h1 at h
  exact h

end inverse_function_point_l130_130843


namespace average_speed_comparison_l130_130689

variables (u v : ℝ) (hu : u > 0) (hv : v > 0)

theorem average_speed_comparison (x y : ℝ) 
  (hx : x = 2 * u * v / (u + v)) 
  (hy : y = (u + v) / 2) : x ≤ y := 
sorry

end average_speed_comparison_l130_130689


namespace find_angle_A_l130_130748

-- Definitions
variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
def given_triangle (a b c A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 0 < A ∧ A < 180 ∧ 0 < B ∧ B < 180 ∧ 0 < C ∧ C < 180 ∧ A + B + C = 180

def given_equation (a b c A B C : ℝ) : Prop :=
  b * cos C + c * cos B = a * sin A

-- Proof statement
theorem find_angle_A (h₁ : given_triangle a b c A B C) (h₂ : given_equation a b c A B C) : 
  A = 90 :=
sorry

end find_angle_A_l130_130748


namespace Eight_div_by_repeating_decimal_0_3_l130_130161

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l130_130161


namespace binomial_coeff_coprime_l130_130462

def binom (a b : ℕ) : ℕ := Nat.factorial a / (Nat.factorial b * Nat.factorial (a - b))

theorem binomial_coeff_coprime (p a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hp : Nat.Prime p) 
  (hbase_p_a : ∀ i, (a / p^i % p) ≥ (b / p^i % p)) 
  : Nat.gcd (binom a b) p = 1 :=
by sorry

end binomial_coeff_coprime_l130_130462


namespace number_of_children_l130_130944

theorem number_of_children (n : ℕ) :
  (∃ (A P T : ℕ), A = 270 - n * (270 / n) ∧ P = 180 - n * (180 / n) ∧ T = 235 - n * (235 / n) ∧ (A : P : T) = (3 : 2 : 1)) → n = 29 :=
by
sorry

end number_of_children_l130_130944


namespace divide_by_repeating_decimal_l130_130174

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l130_130174


namespace parabola_vertex_l130_130933

theorem parabola_vertex :
  ∃ h k : ℝ, (∀ x : ℝ, -2 * (x - 3)^2 - 4 = -2 * (x - h)^2 + k) ∧ h = 3 ∧ k = -4 :=
by
  use 3, -4
  -- We need to show -2 * (x - 3)^2 - 4 = -2 * (x - 3)^2 - 4 for all x and verify the coordinates (3,-4)
  intros x
  simp
  exact ⟨rfl, rfl⟩

end parabola_vertex_l130_130933


namespace problem_correct_transformation_l130_130579

def correctTransformation (A B C D : Prop) : Prop :=
  D

theorem problem_correct_transformation :
  let A := sqrt (1 + 7 / 9) = ±4 / 3 
  let B := sqrt 3 27 = ±3
  let C := sqrt ((-4) ^ 2) =  -4
  let D := ± sqrt 121 = ± 11
  correctTransformation A B C D :=
by sorry

end problem_correct_transformation_l130_130579


namespace frustum_lateral_surface_area_original_cone_height_l130_130694

-- Define the given conditions and parameters
def r1 : ℝ := 4
def r2 : ℝ := 8
def h : ℝ := 6

-- Define the slant height s
def s : ℝ := Real.sqrt ((r2 - r1) ^ 2 + h ^ 2)

-- Prove the lateral surface area of the frustum
theorem frustum_lateral_surface_area : 
  (Real.pi * (r1 + r2) * s) = 24 * Real.pi * Real.sqrt 13 :=
by
  sorry

-- Prove the height of the original cone
theorem original_cone_height : 
  (h + s * (r1 / r2)) = 6 + Real.sqrt 13 :=
by
  sorry

end frustum_lateral_surface_area_original_cone_height_l130_130694


namespace sum_of_possible_values_of_d_l130_130612

def base_digits (n : ℕ) (b : ℕ) : ℕ := 
  if n = 0 then 1 else Nat.log (n + 1) b

theorem sum_of_possible_values_of_d :
  let min_val_7 := 1 * 7^3
  let max_val_7 := 6 * 7^3 + 6 * 7^2 + 6 * 7^1 + 6 * 7^0
  let min_val_10 := 343
  let max_val_10 := 2400
  let d1 := base_digits min_val_10 3
  let d2 := base_digits max_val_10 3
  d1 + d2 = 13 := sorry

end sum_of_possible_values_of_d_l130_130612


namespace angle_OQP_is_90_l130_130463

noncomputable theory
open_locale classical

variables {A B C D O P Q : Type}

-- Assume given conditions
variables [InCircle A B C D O] [IntersectionPointDiagonals P A B C D] [SecondIntersectionCircumcircles Q A P D B P C]

-- Prove the statement
theorem angle_OQP_is_90 (A B C D O P Q : Point) 
  (h1 : InCircle A B C D O) 
  (h2 : IntersectionPointDiagonals P A B C D) 
  (h3 : SecondIntersectionCircumcircles Q A P D B P C) : 
  angle O Q P = 90 :=
sorry

end angle_OQP_is_90_l130_130463


namespace simon_practice_hours_l130_130509

theorem simon_practice_hours (x : ℕ) (h : (12 + 16 + 14 + x) / 4 ≥ 15) : x = 18 := 
by {
  -- placeholder for the proof
  sorry
}

end simon_practice_hours_l130_130509


namespace strength_coeff_all_greater_zero_impossible_strength_coeff_all_less_zero_impossible_l130_130251

-- Define structures for participants and games
structure Participant :=
  (id : ℕ) -- unique identifier

structure Game :=
  (p1 p2 : Participant) -- participants

-- Define points for wins, losses, and draws
inductive Result
| win
| loss
| draw

-- Function to get points based on the result
def points : Result → ℝ
| Result.win := 1
| Result.loss := 0
| Result.draw := 0.5

-- Function to calculate strength coefficient
def strength_coefficient (p : Participant) (games : list Game) (results : Game → Result) : ℝ :=
  let opponents := games.filter (λ g, g.p1 = p ∨ g.p2 = p) in
  let defeated := opponents.filter (λ g, results g = Result.win ∧ g.p1 = p ∨ results g = Result.loss ∧ g.p2 = p) in
  let lost_to := opponents.filter (λ g, results g = Result.loss ∧ g.p1 = p ∨ results g = Result.win ∧ g.p2 = p) in
  (defeated.sum (λ g, points (results g))) - (lost_to.sum (λ g, points (results g)))

-- Main theorem statement for part a
theorem strength_coeff_all_greater_zero_impossible (participants : list Participant) (games : list Game) (results : Game → Result) :
  ¬∀ p ∈ participants, strength_coefficient p games results > 0 :=
sorry

-- Main theorem statement for part b
theorem strength_coeff_all_less_zero_impossible (participants : list Participant) (games : list Game) (results : Game → Result) :
  ¬∀ p ∈ participants, strength_coefficient p games results < 0 :=
sorry

end strength_coeff_all_greater_zero_impossible_strength_coeff_all_less_zero_impossible_l130_130251


namespace total_number_of_boys_in_camp_l130_130245

variables (T : ℕ)
def boys_from_school_a := 0.20 * T
def boys_not_studying_science := 0.70 * (boys_from_school_a T)
def given_boys_not_studying_science := 56

theorem total_number_of_boys_in_camp : T = 400 :=
begin
  have h1 : boys_not_studying_science T = given_boys_not_studying_science,
  { -- by definition, 0.70 * (0.20 * T) = 56
    unfold boys_not_studying_science,
    rw mul_assoc,
    rw mul_comm (0.70 * 0.20) T,
    rw ←mul_assoc,
    norm_num,
    exact eq.refl given_boys_not_studying_science,
  },
  -- Solving for T in h1
  rw h1 at *,
  norm_num at h1,
  exact sorry, -- proof steps go here
end

end total_number_of_boys_in_camp_l130_130245


namespace find_a_l130_130827

noncomputable def f (a x : ℝ) := (x - 1)^2 + a * x + Real.cos x

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) → 
  a = 2 :=
by
  sorry

end find_a_l130_130827


namespace no_rain_five_days_l130_130959

-- Define the problem conditions and the required result.
def prob_rain := (2 / 3)
def prob_no_rain := (1 - prob_rain)
def prob_no_rain_five_days := prob_no_rain^5

theorem no_rain_five_days : 
  prob_no_rain_five_days = (1 / 243) :=
by
  sorry

end no_rain_five_days_l130_130959


namespace area_ratio_inequality_l130_130012

variable (ABC A1B1C1 : Type)
variable [triangle_ABC_class ABC] [triangle_A1B1C1_class A1B1C1]
variable (a b c a1 b1 c1 k : ℝ)
variable (S S1 : ℝ)
variable (not_obtuse_ABC : ¬ obtuse_triangle ABC)
variable (max_ratio : max (a1/a) (max (b1/b) (c1/c)) = k)

noncomputable def area_ABC := area_of_triangle ABC
noncomputable def area_A1B1C1 := area_of_triangle A1B1C1

theorem area_ratio_inequality :
  S1 ≤ k^2 * S :=
sorry

end area_ratio_inequality_l130_130012


namespace division_of_repeating_decimal_l130_130138

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l130_130138


namespace division_of_repeating_decimal_l130_130137

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l130_130137


namespace divide_by_repeating_decimal_l130_130103

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l130_130103


namespace green_balls_l130_130049

theorem green_balls (total_balls : ℕ) (h1 : (1 / 3 : ℚ) * total_balls ∈ ℕ) (h2 : (2 / 7 : ℚ) * total_balls ∈ ℕ) :
  let red_balls := (1 / 3 : ℚ) * total_balls,
      blue_balls := (2 / 7 : ℚ) * total_balls,
      green_balls := total_balls - red_balls - blue_balls
  in green_balls = 2 * blue_balls - 8 → green_balls = 16 :=
by
  sorry

end green_balls_l130_130049


namespace total_rectangles_in_drawing_l130_130470

/-- 
Masha's drawing conditions:
1. There are seven rectangles: one big one and six small ones.
2. There are various middle-sized rectangles as well.
-/
theorem total_rectangles_in_drawing 
  (big: ℕ) (small: ℕ) (middle1: ℕ) (middle2: ℕ) (middle3: ℕ)
  (total: ℕ)
  (h_big: big = 1)
  (h_small: small = 6)
  (h_middle1: middle1 = 2)
  (h_middle2: middle2 = 4)
  (h_middle3: middle3 = 2)
  (h_middle4: middle3 = 3)
  : total = 18 := 
by 
  have h_total : total = big + small + middle1 + middle2 + middle3 + h_middle4,
    simp [h_big, h_small, h_middle1, h_middle2, h_middle3, h_middle4],
  exact h_total


end total_rectangles_in_drawing_l130_130470


namespace find_a8_l130_130746

variable (a : ℕ+ → ℕ)

theorem find_a8 (h : ∀ m n : ℕ+, a (m * n) = a m * a n) (h2 : a 2 = 3) : a 8 = 27 := 
by
  sorry

end find_a8_l130_130746


namespace projection_incorrect_l130_130578

open Real

def is_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

def magnitude_projection (a b : ℝ × ℝ) : ℝ :=
  let num := a.1 * b.1 + a.2 * b.2
  let den := sqrt (b.1 ^ 2 + b.2 ^ 2)
  abs (num / den)

theorem projection_incorrect (a b : ℝ × ℝ)
  (h : is_parallel a b) :
  magnitude_projection a b ≠ |a| := by
  sorry

end projection_incorrect_l130_130578


namespace reach_2_12_22_impossible_12_12_12_l130_130052

-- Initial conditions
def initial_state : ℕ × ℕ × ℕ := (19, 8, 9)

-- Define the operation
def operation (a b c : ℕ) : ℕ × ℕ × ℕ :=
  if a > 0 ∧ b > 0 then (a - 1, b - 1, c + 2) else (a, b, c)

-- Question 1: Is it possible to make the piles (2, 12, 22) after a series of operations?
theorem reach_2_12_22 : 
  ∃ n : ℕ, 
  (19, 8, 9) ->* (2, 12, 22) :=
sorry

-- Question 2: Is it possible to make all piles (12, 12, 12) after a series of operations?
theorem impossible_12_12_12 : 
  ¬ ∃ n : ℕ, 
  (19, 8, 9) ->* (12, 12, 12) :=
sorry

end reach_2_12_22_impossible_12_12_12_l130_130052


namespace eight_div_repeating_three_l130_130080

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l130_130080


namespace coord_relationship_M_l130_130371

theorem coord_relationship_M (x y z : ℝ) (A B : ℝ × ℝ × ℝ)
  (hA : A = (1, 2, -1)) (hB : B = (2, 0, 2))
  (hM : ∃ M : ℝ × ℝ × ℝ, M = (x, y, z) ∧ y = 0 ∧ |(1 - x)^2 + 2^2 + (-1 - z)^2| = |(2 - x)^2 + (0 - z)^2|) :
  x + 3 * z - 1 = 0 ∧ y = 0 := 
sorry

end coord_relationship_M_l130_130371


namespace eight_div_repeating_three_l130_130153

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l130_130153


namespace division_by_repeating_decimal_l130_130211

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l130_130211


namespace rhombus_circle_radius_l130_130744

variable (a α : ℝ)
hypothesis (α : ℝ) (hα0 : 0 < α) (hαpi : α < Real.pi / 2) (a : ℝ) (ha : 0 < a)

theorem rhombus_circle_radius :
  radius (circle_through_vertices_tangent_opposite (side_length a) (acute_angle α)) =
  (a * (1 + 4 * Real.sin α ^ 2)) / (8 * Real.sin α) :=
sorry

end rhombus_circle_radius_l130_130744


namespace concave_probability_l130_130654

def is_concave_number (a b c : ℕ) : Prop := a > b ∧ b < c

def valid_digits (a b c : ℕ) : Prop := 
  a ∈ ({1, 2, 3, 4} : Finset ℕ) ∧ 
  b ∈ ({1, 2, 3, 4} : Finset ℕ) ∧ 
  c ∈ ({1, 2, 3, 4} : Finset ℕ) ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem concave_probability : 
  (Finset.card { n : ℕ | ∃ a b c, valid_digits a b c ∧ is_concave_number a b c ∧ n = 100 * a + 10 * b + c }) = 
  (1/3 : ℚ) * 
  (Finset.card { n : ℕ | ∃ a b c, valid_digits a b c ∧ n = 100 * a + 10 * b + c }) := 
sorry

end concave_probability_l130_130654


namespace division_of_repeating_decimal_l130_130071

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l130_130071


namespace simplify_trig_expression_l130_130919

variables {x : Real}
def tan (x : Real) := sin x / cos x
def cot (x : Real) := cos x / sin x
def sec (x : Real) := 1 / cos x
def csc (x : Real) := 1 / sin x

theorem simplify_trig_expression :
  (tan x) / (1 + cot x) + (1 + cot x) / (tan x) = sec^2 x + csc^2 x :=
by
  sorry

end simplify_trig_expression_l130_130919


namespace longest_segment_in_cylinder_l130_130619

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 12) : 
  (∃ d : ℝ, d = 2 * r) ∧ (∃ hyp : ℝ, hyp = Real.sqrt(h^2 + (2 * r)^2) ∧ hyp = Real.sqrt(244)) :=
by
  sorry

end longest_segment_in_cylinder_l130_130619


namespace pencil_notebook_cost_l130_130286

variable {p n : ℝ}

theorem pencil_notebook_cost (hp1 : 9 * p + 11 * n = 6.05) (hp2 : 6 * p + 4 * n = 2.68) :
  18 * p + 13 * n = 8.45 :=
sorry

end pencil_notebook_cost_l130_130286


namespace minimum_number_of_rooks_l130_130667

theorem minimum_number_of_rooks (n : ℕ) : 
  ∃ (num_rooks : ℕ), (∀ (cells_colored : ℕ), cells_colored = n^2 → num_rooks = n) :=
by sorry

end minimum_number_of_rooks_l130_130667


namespace parabola_relative_position_l130_130698

theorem parabola_relative_position :
  let h₁ := - (2 / 3) / (2 * 1)
  let k₁ := 1 - (2 / 3) ^ 2 / (4 * 1)
  let vertex₁ := (h₁, k₁)
  let h₂ := - (- 2 / 3) / (2 * 1)
  let k₂ := 3 - (-2 / 3) ^ 2 / (4 * 1)
  let vertex₂ := (h₂, k₂)
  vertex₁ = (-1/3 : ℝ, 8/9 : ℝ) ∧ vertex₂ = (1/3 : ℝ, 26/9 : ℝ) ∧ h₁ < h₂ ∧ k₁ < k₂ :=
by
  -- Prove vertex calculations
  have h₁_calc : h₁ = - (2 / 3) / (2 * 1) := rfl
  have k₁_calc : k₁ = 1 - (2 / 3)^2 / (4 * 1) := rfl
  have vertex₁_pos : vertex₁ = (-1/3 : ℝ, 8/9 : ℝ) := by
    simp [h₁_calc, k₁_calc]
    sorry

  have h₂_calc : h₂ = - (- 2 / 3) / (2 * 1) := rfl
  have k₂_calc : k₂ = 3 - (- 2 / 3)^2 / (4 * 1) := rfl
  have vertex₂_pos : vertex₂ = (1/3 : ℝ, 26/9 : ℝ) := by
    simp [h₂_calc, k₂_calc]
    sorry
  
  -- Prove comparisons
  have h_relation : h₁ < h₂ := by
    rw [h₁_calc, h₂_calc]
    simp
    sorry

  have k_relation : k₁ < k₂ := by
    rw [k₁_calc, k₂_calc]
    simp
    sorry

  exact ⟨vertex₁_pos, vertex₂_pos, h_relation, k_relation⟩

end parabola_relative_position_l130_130698


namespace division_by_repeating_decimal_l130_130219

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l130_130219


namespace total_sum_of_different_calculations_eq_600_l130_130711

theorem total_sum_of_different_calculations_eq_600 :
  let numbers := [25, 9, 7, 5, 3, 1] in 
  (∀ s : List Bool, -- For each configuration of signs (+ or -)
    let sum := numbers.zip s.sum (fun n sign => if sign then n else -n) -- Calculate sum based on signs
    (0 ≤ sum) ∧ (sum ≤ 50)) → -- Sum is within the range 0 to 50
  (Sum_of_all_possible_unique_sums numbers = 600) := sorry

end total_sum_of_different_calculations_eq_600_l130_130711


namespace find_a_if_f_is_even_l130_130829

def f (a : ℝ) (x : ℝ) : ℝ := (x-1)^2 + a*x + Real.sin(x + Real.pi / 2)

theorem find_a_if_f_is_even (a : ℝ) (h : ∀ x : ℝ, f(a, x) = f(a, -x)) : a = 2 := by
  sorry

end find_a_if_f_is_even_l130_130829


namespace constant_term_expansion_max_coeff_term_expansion_l130_130773

noncomputable def n : ℕ := 5

theorem constant_term_expansion :
  let a := 8064 in
  ∑ k in Finset.range (2 * n + 1), (Nat.choose (2 * n) k) * (2 ^ (2 * n - k)) * (let x := (1 : ℝ) in x^(2 * n - 2 * k)) = a :=
by
  let exp := (2*x + (1/x))^(2*n)
  have h : 2^(2*n) = 32 * 2^n  -- given condition
  {
    rw ← pow_mul 2,
    exact pow_eq_pow_of_square_eq_pow (2*n) (n + 5),  -- n = 5
  }
  sorry

theorem max_coeff_term_expansion :
  let a := 15360 in
  let r := 3 in
  ∑ k in Finset.range (2 * n + 1), (Nat.choose (2 * n) k) * (2 ^ (2 * n - k)) * (let x := (1 : ℝ) in x^(2 * n - 2 * k)) = a * (x : ℝ)^(4) :=
by
  let exp := (2*x + (1/x))^(2*n)
  have h₁ : nat.gcd (10) (3) = 1 -- C_{10}^{3} is maximum coefficient
  {
    sorry
  }
  have h₂: 2^(2*n) = 32 * 2^n  -- given condition
  {
    rw ← pow_mul 2,
    exact pow_eq_pow_of_square_eq_pow (2*n) (n + 5),  -- n = 5
  }
  sorry

end constant_term_expansion_max_coeff_term_expansion_l130_130773


namespace correct_geometric_statement_l130_130577

theorem correct_geometric_statement :
  (∃ (A B C D : Prop),
    (A ↔ (∀ x y : ℕ, x = y → are_vertical_angles x y)) ∧
    (B ↔ (∀ l m : set ℕ, corresponding_angles l m ↔ are_parallel l m)) ∧
    (C ↔ (∃ l p : set ℕ, p ∉ l ∧ ∀ m : set ℕ, (p ∈ m ∧ is_parallel m l))) ∧
    (D ↔ (∀ θ : ℝ, is_supplement θ → is_obtuse θ)) ∧
    C) :=
sorry

noncomputable def are_vertical_angles (x y : ℕ) : Prop := sorry
noncomputable def corresponding_angles (l m : set ℕ) : Prop := sorry
noncomputable def are_parallel (l m : set ℕ) : Prop := sorry
noncomputable def is_supplement (θ : ℝ) : Prop := sorry
noncomputable def is_obtuse (θ : ℝ) : Prop := sorry

end correct_geometric_statement_l130_130577


namespace find_triangle_height_l130_130373

theorem find_triangle_height 
  (C : ℝ) (b : ℝ) (A : ℝ) 
  (h : ℝ) 
  (hC : C = π / 3)
  (hb : b = 4)
  (hA : A = 2 * sqrt 3) :
  A = 1 / 2 * b * h * sin C → h = 2 * sqrt 3 :=
sorry

end find_triangle_height_l130_130373


namespace time_to_cut_womans_hair_l130_130438

theorem time_to_cut_womans_hair 
  (WL : ℕ) (WM : ℕ) (WK : ℕ) (total_time : ℕ) 
  (num_women : ℕ) (num_men : ℕ) (num_kids : ℕ) 
  (men_haircut_time : ℕ) (kids_haircut_time : ℕ) 
  (overall_time : ℕ) :
  men_haircut_time = 15 →
  kids_haircut_time = 25 →
  num_women = 3 →
  num_men = 2 →
  num_kids = 3 →
  overall_time = 255 →
  overall_time = (num_women * WL + num_men * men_haircut_time + num_kids * kids_haircut_time) →
  WL = 50 :=
by
  sorry

end time_to_cut_womans_hair_l130_130438


namespace solve_quadratic_equation_l130_130006

theorem solve_quadratic_equation (x : ℝ) :
  (x^2 + 2 * x - 15 = 0) ↔ (x = 3 ∨ x = -5) :=
by
  sorry -- proof omitted

end solve_quadratic_equation_l130_130006


namespace division_by_repeating_decimal_l130_130207

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l130_130207


namespace selling_price_correctness_l130_130672

noncomputable def lower_selling_price 
  (selling_price_350 : ℝ) 
  (gain_percent : ℝ) 
  (cost_price : ℝ) 
  : ℝ := 
  cost_price + (selling_price_350 - cost_price) / (1 + gain_percent / 100)

theorem selling_price_correctness :
  lower_selling_price 350 5 200 ≈ 342.86 := 
sorry

end selling_price_correctness_l130_130672


namespace least_possible_N_l130_130629

theorem least_possible_N : ∃ (N : ℕ), (∀ k, 1 ≤ k ∧ k ≤ 28 → k ∣ N) ∧ ¬ (29 ∣ N) ∧ ¬ (30 ∣ N) ∧ N = 5348882400 :=
by
  use 5348882400
  split
  { intro k
    rintro ⟨h1, h2⟩
    cases k
    iterate 28 { exact dvd_refl _ }
    { exfalso; linarith } }
  split
  { norm_num }
  split
  { norm_num }
  rfl
  sorry

end least_possible_N_l130_130629


namespace c_finishes_job_in_60_days_l130_130242

-- Definitions of work rates
variables {A B C : ℝ}

-- Defining the problem
def work_rates_conditions (h1 : A + B = 1/15) (h2 : A + B + C = 1/12) : Prop :=
  1 / C = 60

-- Statement of problem in Lean
theorem c_finishes_job_in_60_days {A B C : ℝ} (h1 : A + B = 1/15) (h2 : A + B + C = 1/12) : 
  work_rates_conditions h1 h2 :=
begin
  sorry
end

end c_finishes_job_in_60_days_l130_130242


namespace pi_bounds_l130_130507

theorem pi_bounds :
  3 < Real.pi ∧ Real.pi < 4 :=
by
  sorry

end pi_bounds_l130_130507


namespace cost_of_gravelling_path_l130_130587

-- Define the problem conditions
def rectangular_grassy_plot_length : ℝ := 110
def rectangular_grassy_plot_width : ℝ := 65
def gravel_path_width : ℝ := 2.5
def cost_per_square_meter_in_rs : ℝ := 0.80

-- Formulate the theorem to prove the cost of gravelling the path
theorem cost_of_gravelling_path :
  let total_area := rectangular_grassy_plot_length * rectangular_grassy_plot_width,
      inner_length := rectangular_grassy_plot_length - 2 * gravel_path_width,
      inner_width := rectangular_grassy_plot_width - 2 * gravel_path_width,
      inner_area := inner_length * inner_width,
      path_area := total_area - inner_area,
      cost := path_area * cost_per_square_meter_in_rs in
  cost = 680 :=
by
  -- Lean proof would go here
  sorry

end cost_of_gravelling_path_l130_130587


namespace largest_n_l130_130317

theorem largest_n (x y z n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12 → n ≤ 6 :=
by
  sorry

end largest_n_l130_130317


namespace minimum_abs_sum_l130_130453

def matrix_squared_condition (p q r s : ℤ) : Prop :=
  (p * p + q * r = 9) ∧ 
  (q * r + s * s = 9) ∧ 
  (p * q + q * s = 0) ∧ 
  (r * p + r * s = 0)

def abs_sum (p q r s : ℤ) : ℤ :=
  |p| + |q| + |r| + |s|

theorem minimum_abs_sum (p q r s : ℤ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0) 
  (h5 : matrix_squared_condition p q r s) : abs_sum p q r s = 8 :=
by 
  sorry

end minimum_abs_sum_l130_130453


namespace divide_by_repeating_decimal_l130_130094

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l130_130094


namespace total_coins_Zain_l130_130241

variable (quartersEmerie dimesEmerie nickelsEmerie : Nat)
variable (additionalCoins : Nat)

theorem total_coins_Zain (h_q : quartersEmerie = 6)
                         (h_d : dimesEmerie = 7)
                         (h_n : nickelsEmerie = 5)
                         (h_add : additionalCoins = 10) :
    let quartersZain := quartersEmerie + additionalCoins
    let dimesZain := dimesEmerie + additionalCoins
    let nickelsZain := nickelsEmerie + additionalCoins
    quartersZain + dimesZain + nickelsZain = 48 := by
  sorry

end total_coins_Zain_l130_130241


namespace division_of_repeating_decimal_l130_130143

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l130_130143


namespace cucumbers_count_l130_130986

theorem cucumbers_count (c : ℕ) (n : ℕ) (additional : ℕ) (initial_cucumbers : ℕ) (total_cucumbers : ℕ) :
  c = 4 → n = 10 → additional = 2 → initial_cucumbers = n - c → total_cucumbers = initial_cucumbers + additional → total_cucumbers = 8 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  simp at h4
  rw [h4, h3] at h5
  simp at h5
  exact h5

end cucumbers_count_l130_130986


namespace find_a_l130_130741

theorem find_a (a : ℝ) :
  (∃ x : ℝ, (a + 1) * x^2 - x + a^2 - 2*a - 2 = 0 ∧ x = 1) → a = 2 :=
by
  sorry

end find_a_l130_130741


namespace division_of_repeating_decimal_l130_130141

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l130_130141


namespace eight_div_repeating_three_l130_130146

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l130_130146


namespace dreams_ratio_l130_130733

variable (N : ℕ) (D_total : ℕ) (D_per_day : ℕ)

-- Conditions
def days_per_year : Prop := N = 365
def dreams_per_day : Prop := D_per_day = 4
def total_dreams : Prop := D_total = 4380

-- Derived definitions
def dreams_this_year := D_per_day * N
def dreams_last_year := D_total - dreams_this_year

-- Theorem to prove
theorem dreams_ratio 
  (h1 : days_per_year N)
  (h2 : dreams_per_day D_per_day)
  (h3 : total_dreams D_total)
  : dreams_last_year N D_total D_per_day / dreams_this_year N D_per_day = 2 :=
by
  sorry

end dreams_ratio_l130_130733


namespace problem1_problem2_problem3_l130_130755

noncomputable def f (x a: ℝ) := Real.logb 2 ((1/x) + a)
noncomputable def g (x a: ℝ) := Real.logb 2 ((a - 4) * x + 2 * a - 5)

theorem problem1 (a: ℝ) (x: ℝ) : 
  g x a > 0 ↔ 
  (a > 4 ∧ x ∈ Ioi (Real.div (6 - 2 * a) (a - 4)) ∨ 
   a = 4 ∧ True ∨ 
   a < 4 ∧ x ∈ Iic (Real.div (6 - 2 * a) (a - 4))) := sorry

theorem problem2 (a: ℝ) :
  (∀ x, f x a = g x a → y = f x a) ↔ (1 < a ∧ a ≤ 2 ∨ a = 3 ∨ a = 4) := sorry

theorem problem3 (a: ℝ) (t x₁ x₂: ℝ) :
  (a > 0 ∧ t ∈ Icc (1/2) 1 ∧ x₁ ∈ Icc t (t + 1) ∧ x₂ ∈ Icc t (t + 1) ∧ 
  |f x₁ a - f x₂ a| ≤ 1) ↔ (2/3 ≤ a) := sorry

end problem1_problem2_problem3_l130_130755


namespace find_digits_sum_l130_130228

theorem find_digits_sum (A B : ℕ) (h1 : A < 10) (h2 : B < 10) 
  (h3 : (A = 6) ∧ (B = 6))
  (h4 : (100 * A + 44610 + B) % 72 = 0) : A + B = 12 := 
by
  sorry

end find_digits_sum_l130_130228


namespace extreme_points_l130_130391

noncomputable def f (x : ℝ) := x * exp (2 * x) - x^2 - x - 1 / 4

theorem extreme_points :
  (∃ x : ℝ, f' x = 0 ∧ (x = -1/2 ∨ x = 0)) ∧
  ¬(∀ x ∈ Icc (-1/2 : ℝ) (0 : ℝ), f' x ≥ 0) ∧
  f (-1/2) = -1/(2*exp 1) ∧
  f 0 = -1/4 :=
sorry

end extreme_points_l130_130391


namespace infinite_common_representations_l130_130322

theorem infinite_common_representations (A B : ℕ) (hA : 0 < A) (hB : 0 < B) (hAB : A ≠ B) :
  ∃ᶠ n in at_top, ∃ x1 y1 x2 y2 : ℕ,
    Nat.coprime x1 y1 ∧ Nat.coprime x2 y2 ∧
    x1 ^ 2 + A * y1 ^ 2 = n ∧ x2 ^ 2 + B * y2 ^ 2 = n := 
by
  sorry

end infinite_common_representations_l130_130322


namespace probability_no_rain_five_days_l130_130950

noncomputable def probability_of_no_rain (rain_prob : ℚ) (days : ℕ) :=
  (1 - rain_prob) ^ days

theorem probability_no_rain_five_days :
  probability_of_no_rain (2/3) 5 = 1/243 :=
by sorry

end probability_no_rain_five_days_l130_130950


namespace division_by_repeating_decimal_l130_130212

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l130_130212


namespace max_t_value_l130_130400

variables {V : Type*} [inner_product_space ℝ V]

theorem max_t_value (a b : V) (t : ℝ) (h1 : ∥a + t • b∥ = 3) (h2 : ⟪a, b⟫ = 2) : t ≤ 9 / 8 :=
begin
  sorry
end

#check max_t_value

end max_t_value_l130_130400


namespace tetrahedron_volume_inequality_l130_130600

theorem tetrahedron_volume_inequality {V R : ℝ} (hR_pos : 0 < R) :
  ∃ V : ℝ, V ≤ (8 / (9 * real.sqrt 3)) * R^3 :=
begin
  sorry
end

end tetrahedron_volume_inequality_l130_130600


namespace locus_of_points_sum_distances_to_two_lines_l130_130318

-- Lean 4 statement for the problem
theorem locus_of_points_sum_distances_to_two_lines (c : ℝ) (e f : set (ℝ × ℝ)) : 
  is_locus_of_points_with_sum_distances_eq_c e f c :=
sorry

end locus_of_points_sum_distances_to_two_lines_l130_130318


namespace boys_without_notebooks_l130_130853

theorem boys_without_notebooks
  (total_boys : ℕ) (students_with_notebooks : ℕ) (girls_with_notebooks : ℕ)
  (h1 : total_boys = 24) (h2 : students_with_notebooks = 30) (h3 : girls_with_notebooks = 17) :
  total_boys - (students_with_notebooks - girls_with_notebooks) = 11 :=
by
  sorry

end boys_without_notebooks_l130_130853


namespace eight_div_repeating_three_l130_130192

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l130_130192


namespace find_a_l130_130011

-- Define the equations of the parabola and hyperbola
def parabola (a : ℝ) : set (ℝ × ℝ) := { p | p.2^2 = a * p.1 }
def hyperbola : set (ℝ × ℝ) := { p | p.1^2 / 8 - p.2^2 / 4 = 1 }

-- Define the latus rectum of the parabola
def latus_rectum (a : ℝ) : set (ℝ × ℝ) := { p | p.1 = -a / 4 }

-- Define the asymptotes of the hyperbola
def asymptotes_up (p : ℝ × ℝ) := p.2 = (sqrt 2 / 2) * p.1
def asymptotes_down (p : ℝ × ℝ) := p.2 = -(sqrt 2 / 2) * p.1

-- The goal theorem to prove
theorem find_a
  (a : ℝ)
  (ha : 0 < a)
  (intersect1 : ( -a / 4 , sqrt 2 / 8 * a ) ∈ latus_rectum a ∩ hyperbola ∩ { p | asymptotes_up p })
  (intersect2 : ( -a / 4 , -sqrt 2 / 8 * a ) ∈ latus_rectum a ∩ hyperbola ∩ { p | asymptotes_down p })
  (area_eq : (1/2) * (a / 4) * (sqrt 2 / 4 * a) = 2 * sqrt 2) :
  a = 8 := 
  sorry

end find_a_l130_130011


namespace total_miles_traveled_l130_130466

/-- Linda's travel conditions for five days -/
def travel_conditions (day: ℕ) : ℕ := 3 * day

def total_distance (f : ℕ → ℕ) (d : ℕ) : ℕ :=
list.sum (list.map f (list.range d))

theorem total_miles_traveled :
  total_distance (λ day, 90 / travel_conditions (day + 1)) 5 = 66 :=
by
  sorry

end total_miles_traveled_l130_130466


namespace min_n_for_constant_term_l130_130412

theorem min_n_for_constant_term :
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, 3 * n = 5 * r) → ∃ n : ℕ, n = 5 :=
by
  intros h
  sorry

end min_n_for_constant_term_l130_130412


namespace shopkeeper_loss_percentages_l130_130625

theorem shopkeeper_loss_percentages 
  (TypeA : Type) (TypeB : Type) (TypeC : Type)
  (theft_percentage_A : ℝ) (theft_percentage_B : ℝ) (theft_percentage_C : ℝ)
  (hA : theft_percentage_A = 0.20)
  (hB : theft_percentage_B = 0.25)
  (hC : theft_percentage_C = 0.30)
  :
  (theft_percentage_A = 0.20 ∧ theft_percentage_B = 0.25 ∧ theft_percentage_C = 0.30) ∧
  ((theft_percentage_A + theft_percentage_B + theft_percentage_C) / 3 = 0.25) :=
by
  sorry

end shopkeeper_loss_percentages_l130_130625


namespace divide_by_repeating_decimal_l130_130176

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l130_130176


namespace select_half_boxes_l130_130588

theorem select_half_boxes (boxes : Fin 99 → Nat × Nat) :
  ∃ (selection : Finset (Fin 99)), selection.card = 50 ∧
  (∑ i in selection, (boxes i).1) ≥ (∑ i, (boxes i).1) / 2 ∧
  (∑ i in selection, (boxes i).2) ≥ (∑ i, (boxes i).2) / 2 :=
sorry

end select_half_boxes_l130_130588


namespace wheat_rate_is_correct_l130_130674

noncomputable def wheat_rate_proof_problem : Prop := 
  let cost1 := 30 * 11.50
  let cost2 (x : ℝ) := 20 * x
  let total_cost (x : ℝ) := cost1 + cost2 x
  let selling_price := 15.75 * 50
  let desired_cost := selling_price / 1.25 in
  ∃ x : ℝ, total_cost x = desired_cost ∧ x = 14.25

theorem wheat_rate_is_correct : wheat_rate_proof_problem := 
by sorry

end wheat_rate_is_correct_l130_130674


namespace largest_angle_in_triangle_l130_130928

theorem largest_angle_in_triangle {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (ha : a = k / 9) (hb : b = k / 12) (hc : c = k / 18) (h : 9*a = 12*b ∧ 12*b = 18*c) :
  ∃ (θ : ℝ), (θ = Real.arccos (-1/4)) :=
begin
  sorry
end

end largest_angle_in_triangle_l130_130928


namespace inversely_proportional_x_y_l130_130516

-- Define the problem statement
theorem inversely_proportional_x_y (x y k : ℝ)
  (h1 : x * y = k)
  (h2 : x = 5)
  (h3 : y = 15) :
  let y' := -30 in
  let x' := -5/2 in
  (x' * y' = k) :=
by 
  sorry

#check inversely_proportional_x_y

end inversely_proportional_x_y_l130_130516


namespace area_shaded_region_is_correct_l130_130428

noncomputable def radius_of_larger_circle : ℝ := 8
noncomputable def radius_of_smaller_circle := radius_of_larger_circle / 2

-- Define areas
noncomputable def area_of_larger_circle := Real.pi * radius_of_larger_circle ^ 2
noncomputable def area_of_smaller_circle := Real.pi * radius_of_smaller_circle ^ 2
noncomputable def total_area_of_smaller_circles := 2 * area_of_smaller_circle
noncomputable def area_of_shaded_region := area_of_larger_circle - total_area_of_smaller_circles

-- Prove that the area of the shaded region is 32π
theorem area_shaded_region_is_correct : area_of_shaded_region = 32 * Real.pi := by
  sorry

end area_shaded_region_is_correct_l130_130428


namespace division_by_repeating_decimal_l130_130200

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l130_130200


namespace count_valid_permutations_l130_130349

theorem count_valid_permutations : 
  let digits := {1, 2, 3, 4, 5}
  let is_even (n : ℕ) := n % 2 = 0
  let is_odd (n : ℕ) := n % 2 = 1
  let valid_perm (p : List ℕ) := ∑ i in p.toFinset, i = 15 ∧ List.Nodup p ∧
    ∃ i, 0 < i ∧ i < 4 ∧ is_even (p.nth_le i sorry) ∧ 
        is_odd (p.nth_le (i - 1) sorry) ∧ is_odd (p.nth_le (i + 1) sorry)
  { p : List ℕ | valid_perm p }.card = 48 :=
sorry

end count_valid_permutations_l130_130349


namespace find_a_if_even_function_l130_130834

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem find_a_if_even_function :
  (∃ a : ℝ, is_even_function (f a)) → ∃ a : ℝ, a = 2 :=
by
  intros h
  sorry

end find_a_if_even_function_l130_130834


namespace f_behavior_l130_130820

def f (x : ℝ) : ℝ := Real.cos x - 1 + (x^2 / 2)

theorem f_behavior :
  (∀ x < 0, derivative f x < 0) ∧ 
  (∀ x > 0, derivative f x > 0) ∧ 
  (derivative f 0 = 0) :=
sorry

end f_behavior_l130_130820


namespace minimized_value_theorem_l130_130348

noncomputable def question (m : ℝ) : ℝ := (m + 1)^2 - 2 * (m - 1)
noncomputable def minimized_value : ℝ := ∀ m, (0 ≤ m) ∧ ((question m) = 3) ∧ (∀ n, (question n) ≥ (question m))

theorem minimized_value_theorem : minimized_value := sorry

end minimized_value_theorem_l130_130348


namespace parametric_equation_tan_l130_130311

theorem parametric_equation_tan (t : ℝ) (n : ℤ) (h : t ≠ (π / 2) + n * π) :
  (∃ x y : ℝ, x = tan t ∧ y = 1 / (tan t) ∧ x * y = 1) :=
by
  sorry

end parametric_equation_tan_l130_130311


namespace bisection_interval_division_l130_130739

-- Definitions of conditions
def is_continuous (f : ℝ → ℝ) : Prop :=
∀ x ε > 0, ∃ δ > 0, ∀ x', abs (x' - x) < δ → abs (f x' - f x) < ε

def has_unique_zero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∃! x, a < x ∧ x < b ∧ f x = 0

noncomputable def bisection_interval_length (a b : ℝ) (n : ℕ) : ℝ :=
(b - a) / 2^n

def within_accuracy (interval_length accuracy : ℝ) : Prop :=
interval_length < accuracy

-- Theorem statement based on the problem
theorem bisection_interval_division (f : ℝ → ℝ) (a b : ℝ)
  (h_continuous : is_continuous f)
  (h_unique_zero : has_unique_zero f a b)
  (h_interval : b - a = 0.1) :
  ∃ n, n ≤ 10 ∧ within_accuracy (bisection_interval_length a b n) 0.0001 :=
begin
  sorry
end

end bisection_interval_division_l130_130739


namespace subtract_complex_eq_l130_130225

noncomputable def subtract_complex (a b : ℂ) : ℂ := a - b

theorem subtract_complex_eq (i : ℂ) (h_i : i^2 = -1) :
  subtract_complex (5 - 3 * i) (7 - 7 * i) = -2 + 4 * i :=
by
  sorry

end subtract_complex_eq_l130_130225


namespace monotonically_increasing_on_interval_l130_130316

noncomputable def f (a x : ℝ) : ℝ := a * x / (x^2 + 1)

theorem monotonically_increasing_on_interval (a : ℝ) (h : 0 < a) :
  ∀ x, -1 < x → x < 1 → deriv (λ x, f a x) x > 0 :=
sorry

end monotonically_increasing_on_interval_l130_130316


namespace eight_div_repeating_three_l130_130150

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l130_130150


namespace max_height_l130_130258
open Real

def height (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 50 + 5

theorem max_height : ∃ t_max : ℝ, height t_max = 135 ∧ ∀ t : ℝ, height t ≤ 135 :=
by
  sorry

end max_height_l130_130258


namespace probability_no_rain_five_days_l130_130952

noncomputable def probability_of_no_rain (rain_prob : ℚ) (days : ℕ) :=
  (1 - rain_prob) ^ days

theorem probability_no_rain_five_days :
  probability_of_no_rain (2/3) 5 = 1/243 :=
by sorry

end probability_no_rain_five_days_l130_130952


namespace probability_unlock_combination_lock_l130_130652

theorem probability_unlock_combination_lock :
  ∀ (n : Nat) (h : n = 10), 
  (1 / ↑n) = (1 / 10) := by
  intros
  rw h
  sorry

end probability_unlock_combination_lock_l130_130652


namespace eight_div_repeating_three_l130_130194

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l130_130194


namespace eight_div_repeating_three_l130_130189

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l130_130189


namespace sin_14_cos_16_plus_cos_14_sin_16_equals_half_l130_130319

theorem sin_14_cos_16_plus_cos_14_sin_16_equals_half :
  sin (14 * Real.pi / 180) * cos (16 * Real.pi / 180) + cos (14 * Real.pi / 180) * sin (16 * Real.pi / 180) = 1 / 2 := 
by 
  sorry

end sin_14_cos_16_plus_cos_14_sin_16_equals_half_l130_130319


namespace equivalent_fraction_power_multiplication_l130_130222

theorem equivalent_fraction_power_multiplication : 
  (8 / 9) ^ 2 * (1 / 3) ^ 2 * (2 / 5) = (128 / 3645) := 
by 
  sorry

end equivalent_fraction_power_multiplication_l130_130222


namespace dandelions_two_days_ago_yellow_turned_white_l130_130243

theorem dandelions_two_days_ago_yellow_turned_white 
  (turned_white_yesterday turned_white_today : ℕ) 
  (h1 : turned_white_yesterday = 14) 
  (h2 : turned_white_today = 11) : 
  turned_white_yesterday + turned_white_today = 25 := 
by 
  rw [h1, h2] 
  norm_num
  -- proof omitted
  sorry

end dandelions_two_days_ago_yellow_turned_white_l130_130243


namespace prism_diagonal_l130_130282

def diagonal_length_prism (length width height : ℝ) : ℝ :=
  Real.sqrt (length^2 + width^2 + height^2)

theorem prism_diagonal :
  ∀ (a b c : ℝ), a = 6 → b = 8 → c = 15 →
  diagonal_length_prism a b c = Real.sqrt 325 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  unfold diagonal_length_prism
  repeat {rw Real.rpow_two}
  sorry

end prism_diagonal_l130_130282


namespace eight_div_repeating_three_l130_130090

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l130_130090


namespace min_red_points_proof_l130_130886

noncomputable def min_red_points (n : ℕ) : ℕ := 2021 * n^(2021 - 1)

theorem min_red_points_proof (n : ℕ) (N : ℕ) (hN : N = n^(2021)) :
  ∃ red_points : ℕ -> ℕ, (∀ i : ℕ, i < 2021 -> red_points i ≤ N) ∧
    ∀ points_on_each_circle (chosen : Fin 2021 -> ℕ), 
    (∀ i, chosen i < N) -> 
    (∃ θ, ∀ i, points_on_each_circle i = (chosen i + θ) % N ->
    red_points i = points_on_each_circle i) -> 
    (∑ i in Finset.range 2021, red_points i ≥ min_red_points n) :=
sorry

end min_red_points_proof_l130_130886


namespace baby_photo_matching_probability_l130_130632

-- Define the total number of photos, famous people, and baby photos.
def total_photos : Nat := 6
def famous_photos : Nat := 3
def baby_photos : Nat := 3

-- Define the total permutations of the baby photos.
def total_permutations : Nat := Nat.factorial baby_photos

-- Define the number of favorable outcomes (correct matching).
def favorable_outcomes : Nat := 1

-- Define the probability of a correct matching.
def correct_matching_probability : ℚ := favorable_outcomes / total_permutations

-- The final theorem we need to prove.
theorem baby_photo_matching_probability (total_photos = 6) 
  (famous_photos = 3) 
  (baby_photos = 3) 
  (total_permutations = 6) 
  (favorable_outcomes = 1) : 
  correct_matching_probability = 1 / 6 := 
sorry

end baby_photo_matching_probability_l130_130632


namespace part1_part2_l130_130786

variables {a m n : ℝ} 

-- Definition of the function f(x)
def f (x : ℝ) := log x - (a * (x - 1) / (x + 1))

-- Part 1: If the function f(x) is monotonically increasing on (0, +∞), then a ≤ 2.
theorem part1 (h : ∀ x y : ℝ, 0 < x → x < y → y < ∞ → f x ≤ f y) : a ≤ 2 :=
sorry

-- Part 2: For m ≠ n, prove that (m - n) / (log m - log n) < (m + n) / 2
theorem part2 (hmn : m ≠ n) : (m - n) / (log m - log n) < (m + n) / 2 :=
sorry

end part1_part2_l130_130786


namespace longest_segment_inside_cylinder_l130_130622

theorem longest_segment_inside_cylinder :
  ∀ (r h : ℝ), r = 5 → h = 12 → ∃ (d : ℝ), d = 2 * Real.sqrt 61 ∧ d = Real.sqrt ((2 * r) ^ 2 + h ^ 2) :=
by
  intros r h hr hh
  have : 2 * r = 10 := by rw [←hr]; norm_num
  have : h = 12 := by rw [←hh]
  use 2 * Real.sqrt 61
  simp [this]
  sorry

end longest_segment_inside_cylinder_l130_130622


namespace division_by_repeating_decimal_l130_130221

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l130_130221


namespace arithmetic_sequence_fraction_l130_130366

noncomputable theory
open_locale classical

-- Definitions of the conditions in plain Lean
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

def common_difference_nonzero (d : ℝ) : Prop := d ≠ 0

def given_condition (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 2 + a 3 = a 6

-- The theorem that needs to be proven
theorem arithmetic_sequence_fraction {a : ℕ → ℝ} {d : ℝ}
  (h1 : is_arithmetic_sequence a d)
  (h2 : common_difference_nonzero d)
  (h3 : given_condition a d) :
  (a 1 + a 2) / (a 3 + a 4 + a 5) = 1 / 3 :=
sorry

end arithmetic_sequence_fraction_l130_130366


namespace gcd_polynomials_l130_130761

theorem gcd_polynomials (a : ℤ) (h : Odd (a / 7889)) : 
  Nat.gcd (eval (6 * X^2 + 55 * X + 126) a).natAbs (eval (2 * X + 11) a).natAbs = 1 :=
by 
  -- Proof omitted
  sorry

end gcd_polynomials_l130_130761


namespace eight_div_repeating_three_l130_130155

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l130_130155


namespace shortest_tree_height_is_correct_l130_130548

-- Definitions of the tree heights
def tallest_tree_height : ℕ := 150
def middle_tree_height : ℕ := (2 * tallest_tree_height) / 3
def shortest_tree_height : ℕ := middle_tree_height / 2

-- Theorem statement
theorem shortest_tree_height_is_correct :
  shortest_tree_height = 50 :=
by
  sorry

end shortest_tree_height_is_correct_l130_130548


namespace gasoline_distribution_possible_l130_130249

-- Defining the initial conditions
def Barrel : Type := ℕ
def Bucket : Type := ℕ
def Scoop : Type := ℕ

noncomputable def initial_barrel : Barrel := 28
noncomputable def bucket_capacity : Bucket := 7
noncomputable def scoop_capacity : Scoop := 4

-- Theorem statement
theorem gasoline_distribution_possible :
  ∃ (final_bucket1 final_bucket2 : Bucket) (final_barrel : Barrel),
    final_bucket1 = 6 ∧ final_bucket2 = 6 ∧ final_barrel + final_bucket1 + final_bucket2 = initial_barrel :=
  sorry

end gasoline_distribution_possible_l130_130249


namespace division_by_repeating_decimal_l130_130218

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l130_130218


namespace collinear_probability_proof_l130_130696

noncomputable def collinear_probability_5x5 : ℚ :=
  7 / 6325

theorem collinear_probability_proof :
  ∀ (S : finset (fin 25)),
    S.card = 4 →
    (S ∈ ({(0, 1, 2, 3, 4) | ((0, 1, 2, 3, 4) ∈ finset.univ \ {3, 3, 3, 3, 3})} ∪
          {(5, 6, 7, 8, 9) | ((5, 6, 7, 8, 9) ∈ finset.univ \ {8, 8, 8, 8, 8})} ∪
          {(10, 11, 12, 13, 14) | ((10, 11, 12, 13, 14) ∈ finset.univ \ {13, 13, 13, 13, 13})} ∪
          {(15, 16, 17, 18, 19) | ((15, 16, 17, 18, 19) ∈ finset.univ \ {18, 18, 18, 18, 18})} ∪
          {(20, 21, 22, 23, 24) | ((20, 21, 22, 23, 24) ∈ finset.univ \ {23, 23, 23, 23, 23})} ∪
          {(0, 5, 10, 15, 20) | ((0, 5, 10, 15, 20) ∈ finset.univ \ {15, 15, 15, 15, 20})} ∪
          {(1, 6, 11, 16, 21) | ((1, 6, 11, 16, 21) ∈ finset.univ \ {16, 16, 16, 16, 21})} ∪
          {(2, 7, 12, 17, 22) | ((2, 7, 12, 17, 22) ∈ finset.univ \ {17, 17, 17, 17, 22})} ∪
          {(3, 8, 13, 18, 23) | ((3, 8, 13, 18, 23) ∈ finset.univ \ {18, 18, 18, 18, 23})} ∪
          {(4, 9, 14, 19, 24) | ((4, 9, 14, 19, 24) ∈ finset.univ \ {19, 19, 19, 19, 24})})) →
  ∃ (prob : ℚ), prob = collinear_probability_5x5 :=
sorry

end collinear_probability_proof_l130_130696


namespace problem_1_10th_number_problem_1_50th_number_problem_2_520_l130_130055

noncomputable def pattern : (ℕ → ℝ)
| (0) := 1
| (1) := -1
| (2) := real.sqrt 2
| (3) := -real.sqrt 2
| (4) := real.sqrt 3
| (5) := -real.sqrt 3
| (n + 6) := pattern n

theorem problem_1_10th_number : pattern 9 = -real.sqrt 2 := by
  sorry

theorem problem_1_50th_number : pattern 49 = -1 := by
  sorry

noncomputable def sum_of_squares (n : ℕ) : ℝ :=
match n with
| 0     => 0
| (n+1) => (pattern n)^2 + sum_of_squares n

theorem problem_2_520 : ∃ n, n = 261 ∧ sum_of_squares 261 = 520 := by
  sorry

end problem_1_10th_number_problem_1_50th_number_problem_2_520_l130_130055


namespace longest_segment_in_cylinder_l130_130617

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 12) : 
  (∃ d : ℝ, d = 2 * r) ∧ (∃ hyp : ℝ, hyp = Real.sqrt(h^2 + (2 * r)^2) ∧ hyp = Real.sqrt(244)) :=
by
  sorry

end longest_segment_in_cylinder_l130_130617


namespace range_of_a_l130_130752

theorem range_of_a (a : ℝ) (in_fourth_quadrant : (a+2 > 0) ∧ (a-3 < 0)) : -2 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l130_130752


namespace students_not_make_cut_l130_130987

theorem students_not_make_cut (girls boys called_back : ℕ) 
  (h_girls : girls = 42) (h_boys : boys = 80)
  (h_called_back : called_back = 25) : 
  (girls + boys - called_back = 97) := by
  sorry

end students_not_make_cut_l130_130987


namespace eight_div_repeating_three_l130_130186

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l130_130186


namespace euler_totient_has_second_solution_l130_130508

theorem euler_totient_has_second_solution {n : ℕ} (x₁ : ℕ) (h₁ : Euler_totient x₁ = n) :
  ∃ (x₂ : ℕ), x₂ ≠ x₁ ∧ Euler_totient x₂ = n :=
sorry

end euler_totient_has_second_solution_l130_130508


namespace total_coins_Zain_l130_130239

variable (quartersEmerie dimesEmerie nickelsEmerie : Nat)
variable (additionalCoins : Nat)

theorem total_coins_Zain (h_q : quartersEmerie = 6)
                         (h_d : dimesEmerie = 7)
                         (h_n : nickelsEmerie = 5)
                         (h_add : additionalCoins = 10) :
    let quartersZain := quartersEmerie + additionalCoins
    let dimesZain := dimesEmerie + additionalCoins
    let nickelsZain := nickelsEmerie + additionalCoins
    quartersZain + dimesZain + nickelsZain = 48 := by
  sorry

end total_coins_Zain_l130_130239


namespace divide_by_repeating_decimal_l130_130101

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l130_130101


namespace trig_identity_l130_130544

theorem trig_identity : sin (10 * real.pi / 180) * cos (50 * real.pi / 180) + cos (10 * real.pi / 180) * sin (50 * real.pi / 180) = sqrt 3 / 2 :=
by 
  sorry

end trig_identity_l130_130544


namespace division_of_repeating_decimal_l130_130131

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l130_130131


namespace four_people_know_each_other_in_group_of_nine_l130_130979

def knows (P Q : ℕ) : Prop := sorry
def group_of_friends (group : Finset ℕ) : Prop :=
  ∀ P Q ∈ group, P ≠ Q → knows P Q

theorem four_people_know_each_other_in_group_of_nine (people : Finset ℕ) 
  (h_card : people.card = 9)
  (h : ∀ (t : Finset ℕ), t.card = 3 → ∃ P Q ∈ t, P ≠ Q ∧ knows P Q) :
  ∃ (group : Finset ℕ), group.card = 4 ∧ group_of_friends group := 
sorry

end four_people_know_each_other_in_group_of_nine_l130_130979


namespace sqrt_eq_solution_l130_130716

theorem sqrt_eq_solution :
  ∀ (x : ℝ), x ≥ 2 → (√(x + 5 - 6 * √(x - 2)) + √(x + 10 - 8 * √(x - 2)) = 3 ↔ x = 32.25 ∨ x = 8.25) := by
  sorry

end sqrt_eq_solution_l130_130716


namespace find_expression_l130_130768

noncomputable def symm_func (f : ℝ → ℝ) : Prop :=
  ∀ x, f (f x) = x

theorem find_expression (f : ℝ → ℝ):
  symm_func f ∧ (∀ x, f f x = x) → ∀ x : ℝ, f x = 2^x - 1 :=
begin
  sorry
end

end find_expression_l130_130768


namespace bob_total_distance_traveled_over_six_days_l130_130686

theorem bob_total_distance_traveled_over_six_days (x : ℤ) (hx1 : 3 ≤ x) (hx2 : x % 3 = 0):
  (90 / x + 90 / (x + 3) + 90 / (x + 6) + 90 / (x + 9) + 90 / (x + 12) + 90 / (x + 15) : ℝ) = 73.5 :=
by
  sorry

end bob_total_distance_traveled_over_six_days_l130_130686


namespace probability_same_color_l130_130586

/-
Problem statement:
Given a bag contains 6 green balls and 7 white balls,
if two balls are drawn simultaneously, prove that the probability 
that both balls are the same color is 6/13.
-/

theorem probability_same_color
  (total_balls : ℕ := 6 + 7)
  (green_balls : ℕ := 6)
  (white_balls : ℕ := 7)
  (two_balls_drawn_simultaneously : Prop := true) :
  ((green_balls / total_balls) * ((green_balls - 1) / (total_balls - 1))) +
  ((white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))) = 6 / 13 :=
sorry

end probability_same_color_l130_130586


namespace simplify_and_evaluate_l130_130920

theorem simplify_and_evaluate (a : ℝ) (h : a = real.sqrt 3 - 1) :
  ((1 - 1 / a) / ((a^2 - 1) / a)) = real.sqrt 3 / 3 :=
by {
  sorry
}

end simplify_and_evaluate_l130_130920


namespace harrison_annual_croissant_expenditure_l130_130800

-- Define the different costs and frequency of croissants.
def cost_regular_croissant : ℝ := 3.50
def cost_almond_croissant : ℝ := 5.50
def cost_chocolate_croissant : ℝ := 4.50
def cost_ham_cheese_croissant : ℝ := 6.00

def frequency_regular_croissant : ℕ := 52
def frequency_almond_croissant : ℕ := 52
def frequency_chocolate_croissant : ℕ := 52
def frequency_ham_cheese_croissant : ℕ := 26

-- Calculate annual expenditure for each type of croissant.
def annual_expenditure (cost : ℝ) (frequency : ℕ) : ℝ :=
  cost * frequency

-- Total annual expenditure on croissants.
def total_annual_expenditure : ℝ :=
  annual_expenditure cost_regular_croissant frequency_regular_croissant +
  annual_expenditure cost_almond_croissant frequency_almond_croissant +
  annual_expenditure cost_chocolate_croissant frequency_chocolate_croissant +
  annual_expenditure cost_ham_cheese_croissant frequency_ham_cheese_croissant

-- The theorem to prove.
theorem harrison_annual_croissant_expenditure :
  total_annual_expenditure = 858 := by
  sorry

end harrison_annual_croissant_expenditure_l130_130800


namespace divide_by_repeating_decimal_l130_130093

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l130_130093


namespace divide_by_repeating_decimal_l130_130173

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l130_130173


namespace acute_angles_sine_cosine_l130_130770

variable {f : ℝ → ℝ}
variable {α β : ℝ}

-- Given conditions
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def monotonically_decreasing_on (f : ℝ → ℝ) (I : set ℝ) := ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x 

-- The main theorem
theorem acute_angles_sine_cosine (h1 : odd_function f)
    (h2 : monotonically_decreasing_on f (set.Icc (-1 : ℝ) (0 : ℝ)))
    (h3 : α > 0) (h4 : α < π / 2) (h5 : β > 0) (h6 : β < π / 2)
    (h7 : cos α > cos β) : f (sin α) < f (cos β) :=
sorry

end acute_angles_sine_cosine_l130_130770


namespace find_f_3_l130_130783

-- Define function f
def f (a b c x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 3

-- Define our conditions
def cond1 {a b c : ℝ} : (f a b c (-3) = 7) := sorry

-- Prove that f(3) = -13 under the given conditions
theorem find_f_3 (a b c : ℝ) (h : cond1) : f a b c 3 = -13 := 
by 
  sorry

end find_f_3_l130_130783


namespace roots_of_my_func_extrema_of_my_func_l130_130384

noncomputable def my_func (x : ℝ) : ℝ :=
  (real.sqrt 3) * (real.sin x)^2 + (real.sin x) * (real.cos x)

theorem roots_of_my_func : ∀ x ∈ set.Icc (real.pi / 2) real.pi,
  my_func x = 0 ↔ (x = 5 * real.pi / 6 ∨ x = real.pi) :=
begin
  sorry
end

theorem extrema_of_my_func :
  ∃ x_max x_min ∈ set.Icc (real.pi / 2) real.pi,
    (my_func x_max = real.sqrt 3) ∧ (my_func x_min = -1 + real.sqrt 3 / 2) :=
begin
  sorry
end

end roots_of_my_func_extrema_of_my_func_l130_130384


namespace correct_judgment_f_l130_130315

def f (x : ℝ) : ℝ := (2 * x - x^2) * Real.exp x

theorem correct_judgment_f (x : ℝ) :
  ((0 < x ∧ x < 2) → f(x) > 0) ∧
  (f (-Real.sqrt 2) = min f ∧ f (Real.sqrt 2) = max f) ∧
  ¬(∀ x, f(x) < 0 ∨ ¬(f (-Real.sqrt 2) = min f ∧ f (Real.sqrt 2) = max f)) :=
by
  sorry

end correct_judgment_f_l130_130315


namespace range_of_a_correct_l130_130782

-- Define the function f(x)
def f (x : ℝ) : ℝ :=
if x ≤ 0 then 2^(1-x)
else 1 - Real.log2 x

-- Define the condition for the range
def condition (a : ℝ) : Prop :=
|f a| ≥ 2

-- Define the range of a that we need to prove
def range_of_a (a : ℝ) : Prop :=
a ≤ (1/2) ∨ a ≥ 8

-- Statement of the theorem
theorem range_of_a_correct (a : ℝ) (h : condition a) : range_of_a a :=
sorry

end range_of_a_correct_l130_130782


namespace division_of_repeating_decimal_l130_130073

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l130_130073


namespace rationalize_denominator_correct_l130_130505

noncomputable def rationalize_denominator : Prop :=
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l130_130505


namespace conference_handshakes_l130_130301

def Group3_shakes (n3 n1 n2 : ℕ) : ℕ := n3 * (n1 + n2)
def Group2_shakes (n2 n1 n3 : ℕ) : ℕ := n2 * (n1 + n3)
def total_unique_handshakes (g3 g2 : ℕ) : ℕ := (g3 + g2) / 2

theorem conference_handshakes :
  let n1 := 25 in
  let n2 := 10 in
  let n3 := 5 in
  total_unique_handshakes (Group3_shakes n3 n1 n2) (Group2_shakes n2 n1 n3) = 237 :=
by
  sorry

end conference_handshakes_l130_130301


namespace f_2017_plus_f_2016_l130_130750

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom f_even_shift : ∀ x : ℝ, f (x + 2) = f (-x + 2)
axiom f_at_neg1 : f (-1) = -1

theorem f_2017_plus_f_2016 : f 2017 + f 2016 = 1 :=
by
  sorry

end f_2017_plus_f_2016_l130_130750


namespace bob_is_47_5_l130_130677

def bob_age (a b : ℝ) := b = 3 * a - 20
def sum_of_ages (a b : ℝ) := b + a = 70

theorem bob_is_47_5 (a b : ℝ) (h1 : bob_age a b) (h2 : sum_of_ages a b) : b = 47.5 :=
by
  sorry

end bob_is_47_5_l130_130677


namespace simplify_fraction_sum_l130_130000

theorem simplify_fraction_sum :
  (3 / 462) + (17 / 42) + (1 / 11) = 116 / 231 := 
by
  sorry

end simplify_fraction_sum_l130_130000


namespace probability_even_sum_is_correct_l130_130627

noncomputable def probability_even_sum : ℚ :=
  let p_even_first := (2 : ℚ) / 5
  let p_odd_first := (3 : ℚ) / 5
  let p_even_second := (1 : ℚ) / 4
  let p_odd_second := (3 : ℚ) / 4

  let p_both_even := p_even_first * p_even_second
  let p_both_odd := p_odd_first * p_odd_second

  p_both_even + p_both_odd

theorem probability_even_sum_is_correct : probability_even_sum = 11 / 20 := by
  sorry

end probability_even_sum_is_correct_l130_130627


namespace min_domain_size_of_f_l130_130063

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem min_domain_size_of_f : {n : ℕ // ∀ a b : ℕ, f(a) = b → f(b) = 34 → n ≥ 15} :=
sorry

end min_domain_size_of_f_l130_130063


namespace only_set_C_forms_triangle_l130_130669

def triangle_inequality (a b c : ℝ) : Prop := 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem only_set_C_forms_triangle : 
  (¬ triangle_inequality 1 2 3) ∧ 
  (¬ triangle_inequality 2 3 6) ∧ 
  triangle_inequality 4 6 8 ∧ 
  (¬ triangle_inequality 5 6 12) := 
by 
  sorry

end only_set_C_forms_triangle_l130_130669


namespace parabola_intersection_angles_l130_130718

def parabola (x : ℝ) : ℝ := x^2 - x
def derivative (x : ℝ) : ℝ := 2 * x - 1
def angle_atan (slope : ℝ) : ℝ := Real.arctan slope

theorem parabola_intersection_angles :
  (angle_atan (derivative 0) = (3 * Real.pi / 4))
  ∧ (angle_atan (derivative 1) = (Real.pi / 4)) :=
by
  sorry

end parabola_intersection_angles_l130_130718


namespace cone_height_l130_130845

theorem cone_height (a : ℝ) (h l r : ℝ)
  (hcircumference : ℝ) (hslant_height : l = a) (hlateral_surface : hcircumference = π * a)
  (hradius_base : hcircumference = 2 * π * r) : h = (sqrt 3 / 2) * a :=
by
  -- define the necessary conditions
  sorry

end cone_height_l130_130845


namespace soybeans_in_jar_l130_130275

theorem soybeans_in_jar
  (totalRedBeans : ℕ)
  (sampleSize : ℕ)
  (sampleRedBeans : ℕ)
  (totalBeans : ℕ)
  (proportion : sampleRedBeans / sampleSize = totalRedBeans / totalBeans)
  (h1 : totalRedBeans = 200)
  (h2 : sampleSize = 60)
  (h3 : sampleRedBeans = 5) :
  totalBeans = 2400 :=
by
  sorry

end soybeans_in_jar_l130_130275


namespace division_by_repeating_decimal_l130_130205

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l130_130205


namespace divide_by_repeating_decimal_l130_130171

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l130_130171


namespace new_ratio_milk_water_after_adding_milk_l130_130854

variable (initial_volume : ℕ) (initial_milk_ratio : ℕ) (initial_water_ratio : ℕ)
variable (added_milk_volume : ℕ)

def ratio_of_mix_after_addition (initial_volume : ℕ) (initial_milk_ratio : ℕ) (initial_water_ratio : ℕ) 
  (added_milk_volume : ℕ) : ℕ × ℕ :=
  let total_parts := initial_milk_ratio + initial_water_ratio
  let part_volume := initial_volume / total_parts
  let initial_milk_volume := initial_milk_ratio * part_volume
  let initial_water_volume := initial_water_ratio * part_volume
  let new_milk_volume := initial_milk_volume + added_milk_volume
  (new_milk_volume / initial_water_volume, 1)

theorem new_ratio_milk_water_after_adding_milk 
  (h_initial_volume : initial_volume = 20)
  (h_initial_milk_ratio : initial_milk_ratio = 3)
  (h_initial_water_ratio : initial_water_ratio = 1)
  (h_added_milk_volume : added_milk_volume = 5) : 
  ratio_of_mix_after_addition initial_volume initial_milk_ratio initial_water_ratio added_milk_volume = (4, 1) :=
  by
    sorry

end new_ratio_milk_water_after_adding_milk_l130_130854


namespace minimum_lines_to_determine_point_in_square_l130_130597

noncomputable def point := ℝ × ℝ -- Define a point as a tuple of real numbers.
noncomputable def line := { l : point × point // l.1 ≠ l.2 } -- Define a line as a pair of distinct points.

structure Square :=
(A B C D : point) -- Define a square by its four vertices.

noncomputable def lies_on_line (P : point) (l : line) : bool := sorry -- This is a placeholder for the actual function defining whether a point lies on a given line.

theorem minimum_lines_to_determine_point_in_square (sq : Square) (P : point) :
  (∃ (l1 l2 l3 : line), 
    (lies_on_line P l1) ∨ (¬ lies_on_line P l1) ∧
    (lies_on_line P l2) ∨ (¬ lies_on_line P l2) ∧
    (lies_on_line P l3) ∨ (¬ lies_on_line P l3)) →
  (P.fst ≥ sq.A.fst ∧ P.fst ≤ sq.C.fst ∧ -- x-coordinate bounds
   P.snd ≥ sq.A.snd ∧ P.snd ≤ sq.C.snd) -- y-coordinate bounds 
:= sorry

end minimum_lines_to_determine_point_in_square_l130_130597


namespace brie_to_salami_ratio_l130_130926

theorem brie_to_salami_ratio :
  ∀ (sandwich_price salami_price olives_price_per_pound feta_price_per_pound bread_price total_spent : ℝ)
  (sandwiches : ℕ) (olives_weight feta_weight : ℝ),
  sandwiches = 2 →
  sandwich_price = 7.75 →
  salami_price = 4.00 →
  olives_price_per_pound = 10.00 →
  olives_weight = 0.25 →
  feta_price_per_pound = 8.00 →
  feta_weight = 0.50 →
  bread_price = 2.00 →
  total_spent = 40.00 →
  let total_cost := sandwiches * sandwich_price + salami_price + (olives_price_per_pound * olives_weight) + 
                    (feta_price_per_pound * feta_weight) + bread_price in
  total_spent - total_cost = 12.00 →
  (total_spent - total_cost) / salami_price = 3 :=
begin
  intros,
  simp,
  sorry
end

end brie_to_salami_ratio_l130_130926


namespace diogenes_club_invariant_participants_l130_130981

-- Define the graph structure and properties
structure Graph :=
  (V : Type) -- Vertices
  (adj : V → V → Prop) -- Adjacency relation
  (symm : ∀ (u v : V), adj u v → adj v u)

-- Define degree of a vertex
def degree (G : Graph) (v : G.V) : ℕ :=
  finset.card { u | G.adj v u }

-- Define the goal in terms of the graph operations
theorem diogenes_club_invariant_participants (G : Graph) :
  ∀ (G₀ : finset G.V) (H₀ : ∀ v ∈ G₀, ∀ u ∈ G₀, ¬ G.adj u v),
  ∃ (k : ℕ), ∀ (G₁ : finset G.V) (H₁ : ∀ v ∈ G₁, ∀ u ∈ G₁, ¬ G.adj u v),
  (∃ S ⊆ G₀, G₁ = G₀ \ S) → finset.card G₁ = k :=
sorry

end diogenes_club_invariant_participants_l130_130981


namespace rationalize_denominator_correct_l130_130504

noncomputable def rationalize_denominator : Prop :=
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l130_130504


namespace max_value_of_expression_l130_130357

theorem max_value_of_expression (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 2 * a + b = 1) :
  2 * real.sqrt (a * b) - 4 * a^2 - b^2 ≤ (real.sqrt 2 - 1) / 2 :=
sorry

end max_value_of_expression_l130_130357


namespace original_price_of_tennis_racket_l130_130878

theorem original_price_of_tennis_racket
  (sneaker_cost : ℝ) (outfit_cost : ℝ) (discount_rate : ℝ) (total_spent : ℝ)
  (price_of_tennis_racket : ℝ) :
  sneaker_cost = 200 → 
  outfit_cost = 250 → 
  discount_rate = 0.20 → 
  total_spent = 750 → 
  price_of_tennis_racket = 289.77 :=
by
  intros hs ho hd ht
  have ht := ht.symm   -- To rearrange the equation
  sorry

end original_price_of_tennis_racket_l130_130878


namespace fruit_salad_mixture_l130_130272

theorem fruit_salad_mixture :
  ∃ (A P G : ℝ), A / P = 12 / 8 ∧ A / G = 12 / 7 ∧ P / G = 8 / 7 ∧ A = G + 10 ∧ A + P + G = 54 :=
by
  sorry

end fruit_salad_mixture_l130_130272


namespace count_integers_satisfying_inequality_l130_130802

theorem count_integers_satisfying_inequality :
  {n : ℤ | (n + 2) * (n - 8) ≤ 0}.to_finset.card = 11 :=
sorry

end count_integers_satisfying_inequality_l130_130802


namespace prove_divisibility_l130_130884

-- Given the conditions:
variables (a b r s : ℕ)
variables (pos_a : a > 0) (pos_b : b > 0) (pos_r : r > 0) (pos_s : s > 0)
variables (a_le_two : a ≤ 2)
variables (no_common_prime_factor : (gcd a b) = 1)
variables (divisibility_condition : (a ^ s + b ^ s) ∣ (a ^ r + b ^ r))

-- We aim to prove that:
theorem prove_divisibility : s ∣ r := 
sorry

end prove_divisibility_l130_130884


namespace constant_length_O₁O₂_l130_130868

-- Given trapezoid ABCD with AD parallel to BC
variables {A B C D E : Type*} [trapezoid ABCD] (hAD_parallel_BC : AD ∥ BC)
-- E is a point on AB
(hE_on_AB : E ∈ AB)
-- O1 is the circumcenter of triangle AED
(O₁ : circumcenter(△ AED))
-- O2 is the circumcenter of triangle BEC
(O₂ : circumcenter(△ BEC))

-- Goal: Prove that the length of O₁O₂ is constant
theorem constant_length_O₁O₂ (E : Point) (hE_on_AB : E ∈ AB) 
  (O₁ : Circumcenter(△ AED)) (O₂ : Circumcenter(△ BEC))
  : ∃ k : ℝ, ∀ E ∈ AB, dist O₁ O₂ = k :=
sorry

end constant_length_O₁O₂_l130_130868


namespace division_of_repeating_decimal_l130_130136

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l130_130136


namespace eight_div_repeating_three_l130_130091

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l130_130091


namespace lisa_marbles_l130_130692

def ConnieMarbles : ℕ := 323
def JuanMarbles (ConnieMarbles : ℕ) : ℕ := ConnieMarbles + 175
def MarkMarbles (JuanMarbles : ℕ) : ℕ := 3 * JuanMarbles
def LisaMarbles (MarkMarbles : ℕ) : ℕ := MarkMarbles / 2 - 200

theorem lisa_marbles :
  LisaMarbles (MarkMarbles (JuanMarbles ConnieMarbles)) = 547 := by
  sorry

end lisa_marbles_l130_130692


namespace regular_octagon_from_circles_l130_130064

noncomputable def diagonal_length (c : ℝ) : ℝ :=
  c * Real.sqrt 2

noncomputable def circle_radius (c : ℝ) : ℝ :=
  (diagonal_length c) / 2

noncomputable def intersection_points_form_regular_octagon (c : ℝ) : Prop :=
  let r := circle_radius c in
  -- Points P1, P2, ..., P8 are the intersection points
  -- Proving that they form a regular octagon:
  ∀ P1 P2 P3 P4 P5 P6 P7 P8 : ℝ×ℝ,
    -- Points intersection on the square sides
    (dist P1 P2 = dist P2 P3) ∧ 
    (dist P3 P4 = dist P4 P5) ∧ 
    (dist P5 P6 = dist P6 P7) ∧ 
    (dist P7 P8 = dist P8 P1) ∧ 
    (dist P1 P3 = dist P3 P5) ∧ 
    (dist P5 P7 = dist P7 P1) ∧ 
    (dist P2 P4 = dist P4 P6) ∧ 
    (dist P6 P8 = dist P8 P2)

theorem regular_octagon_from_circles (c : ℝ) :
  ∃ P1 P2 P3 P4 P5 P6 P7 P8 : ℝ×ℝ,
    intersection_points_form_regular_octagon c := by
  sorry

end regular_octagon_from_circles_l130_130064


namespace multiplicative_inverse_123_mod_455_l130_130691

theorem multiplicative_inverse_123_mod_455 :
    ∃ b : ℤ, 0 ≤ b ∧ b < 455 ∧ (123 * b) % 455 = 1 := 
by
  use 223
  split; try {norm_num}
  sorry

end multiplicative_inverse_123_mod_455_l130_130691


namespace division_by_repeating_decimal_l130_130208

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l130_130208


namespace fraction_division_l130_130894

def smallest_multiple_75_with_100_divisors : ℕ := 3^3 * 5^2 * 2^24

theorem fraction_division (n : ℕ) (h1 : n = smallest_multiple_75_with_100_divisors) 
(h2 : ∃ k, n = 75 * k ∧ (∀ m, (∀ k1, m = 75 * k1 → m = m → n = n)
→ n = n))
: n / 75 = 150994944 :=
begin
  sorry
end

end fraction_division_l130_130894


namespace ellipse_eccentricity_proof_l130_130368

theorem ellipse_eccentricity_proof (a b c : ℝ) 
  (ha_gt_hb : a > b) (hb_gt_zero : b > 0) (hc_gt_zero : c > 0)
  (h_ellipse : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h_r : ∃ r : ℝ, r = (Real.sqrt 2 / 6) * c) :
  (Real.sqrt (1 - b^2 / a^2)) = (2 * Real.sqrt 5 / 5) := by {
  sorry
}

end ellipse_eccentricity_proof_l130_130368


namespace chocolates_difference_l130_130913

theorem chocolates_difference (R N A : ℕ) (hR : R = 10) (hN : N = 5) (hA : A = 15) : (R + N) - A = 0 :=
by 
  have h1 : R + N = 15, by rw [hR, hN]; norm_num,
  have h2 : A = 15, by exact hA,
  rw [h1, h2],
  norm_num,
  done

end chocolates_difference_l130_130913


namespace probability_product_divisible_by_5_l130_130880

def probability_divisible_by_5 : ℚ := 144495 / 262144

theorem probability_product_divisible_by_5 :
  (let dice_sides := (1 : ℕ)..8 in
   let roll_count := 6 in
   let valid_rolls := dice_sides.erase 5 in
   let valid_probability := (7 / 8)^roll_count in
   1 - valid_probability = probability_divisible_by_5) :=
by
  sorry

end probability_product_divisible_by_5_l130_130880


namespace avg_marks_chem_math_l130_130969

variable (P C M : ℝ)

theorem avg_marks_chem_math (h : P + C + M = P + 140) : (C + M) / 2 = 70 :=
by
  -- skip the proof, just provide the statement
  sorry

end avg_marks_chem_math_l130_130969


namespace conjugate_of_z_l130_130776

noncomputable def z : ℂ := 2 * complex.I * (3 + complex.I)
def conjugate_z : ℂ := complex.conj z

theorem conjugate_of_z : conjugate_z = -2 - 6 * complex.I :=
by
  unfold z
  unfold conjugate_z
  sorry

end conjugate_of_z_l130_130776


namespace divide_by_repeating_decimal_l130_130172

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l130_130172


namespace row_speed_with_stream_l130_130633

theorem row_speed_with_stream (v : ℝ) (s : ℝ) (h1 : s = 2) (h2 : v - s = 12) : v + s = 16 := by
  -- Placeholder for the proof
  sorry

end row_speed_with_stream_l130_130633


namespace number_of_integers_satisfying_inequality_l130_130810

theorem number_of_integers_satisfying_inequality : 
  {n : ℤ | (n + 2) * (n - 8) ≤ 0}.finite.card = 11 := by
  sorry

end number_of_integers_satisfying_inequality_l130_130810


namespace probability_of_concave_number_is_one_third_l130_130656

-- Definitions based on the conditions in the problem
def is_concave (a b c : ℕ) := (a > b) ∧ (b < c)

def is_valid_digit (n : ℕ) := n ∈ ({1, 2, 3, 4} : Finset ℕ)

def are_distinct (a b c : ℕ) := a ≠ b ∧ b ≠ c ∧ a ≠ c

def all_three_digit_numbers : Finset (ℕ × ℕ × ℕ) :=
  { x | let (a, b, c) := x in is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ are_distinct a b c }

def concave_numbers : Finset (ℕ × ℕ × ℕ) :=
  { x | let (a, b, c) := x in is_concave a b c ∧ x ∈ all_three_digit_numbers }

def probability_concave : ℚ :=
  (concave_numbers.card : ℚ) / (all_three_digit_numbers.card : ℚ)

theorem probability_of_concave_number_is_one_third :
  probability_concave = 1 / 3 :=
by
  sorry

end probability_of_concave_number_is_one_third_l130_130656


namespace probability_of_no_rain_l130_130953

theorem probability_of_no_rain (prob_rain : ℚ) (prob_no_rain : ℚ) (days : ℕ) (h : prob_rain = 2/3) (h_prob_no_rain : prob_no_rain = 1 - prob_rain) :
  (prob_no_rain ^ days) = 1/243 :=
by 
  sorry

end probability_of_no_rain_l130_130953


namespace solve_equation_l130_130513

theorem solve_equation (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x - y = x / y + (x * x) / (y * y) + (x * x * x) / (y * y * y) ↔ 
  (x = 28 ∧ y = 14) ∨ (x = 112 ∧ y = 28) :=
begin
  sorry
end

end solve_equation_l130_130513


namespace number_of_trees_in_park_l130_130292

/-- A wildlife team is monitoring the number of birds in a park with 3 blackbirds in each tree 
    and 13 magpies roaming around. The total number of birds in the park is 34. 
    This statement proves that the number of trees in the park is 7. -/
theorem number_of_trees_in_park : 
  ∃ T : ℕ, (3 * T + 13 = 34) ∧ T = 7 :=
by
  use 7
  split
  · -- prove 3 * T + 13 = 34
    sorry
  · -- prove T = 7
    rfl

end number_of_trees_in_park_l130_130292


namespace hens_to_roosters_multiplier_l130_130626

def totalChickens : ℕ := 75
def numHens : ℕ := 67

-- Given the total number of chickens and a certain relationship
theorem hens_to_roosters_multiplier
  (numRoosters : ℕ) (multiplier : ℕ)
  (h1 : totalChickens = numHens + numRoosters)
  (h2 : numHens = multiplier * numRoosters - 5) :
  multiplier = 9 :=
by sorry

end hens_to_roosters_multiplier_l130_130626


namespace cylinder_and_cone_l130_130645

noncomputable def cylinder_cone_relationship (m: ℝ) : Prop :=
  let r := (m * (Real.sqrt 21 - 4) / 10)
  let R := (m * ((3 * Real.sqrt 7) - (4 * Real.sqrt 3)) / 10)
  let V := ((m ^ 3) * π / 100 * (37 - 8 * Real.sqrt 21))
  let F := ((m ^ 2) * π / 50 * (2 * Real.sqrt 21 - 3))
  (V = π * r^2 * m) ∧ 
  (V = (1 / 3) * π * R^2 * m) ∧ 
  (F = 2 * π * r^2 + 2 * π * r * m) ∧ 
  (F = 2 * π * R * Real.sqrt (R^2 + (m^2 / 4))) ∧ 
  (r = m * (Real.sqrt 21 - 4) / 10) ∧
  (R = m * ((3 * Real.sqrt 7) - (4 * Real.sqrt 3)) / 10) ∧
  (V = m^3 * π * (37 - 8 * Real.sqrt 21) / 100) ∧
  (F = m^2 * π * (2 * Real.sqrt 21 - 3) / 50)

theorem cylinder_and_cone (m: ℝ) : cylinder_cone_relationship m :=
  sorry

end cylinder_and_cone_l130_130645


namespace eight_div_repeat_three_l130_130127

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l130_130127


namespace find_integer_mul_a_l130_130839

noncomputable def integer_mul_a (a b : ℤ) (n : ℤ) : Prop :=
  n * a * (-8 * b) + a * b = 89 ∧ n < 0 ∧ n * a < 0 ∧ -8 * b < 0

theorem find_integer_mul_a (a b : ℤ) (n : ℤ) (h : integer_mul_a a b n) : n = -11 :=
  sorry

end find_integer_mul_a_l130_130839


namespace simplify_and_evaluate_l130_130002

theorem simplify_and_evaluate :
  let x := (abs (sqrt 3 - 2) + (1/2) ^ (-1) - ((π - 3.14) ^ 0) - (3 : ℝ) ^ (1/3) + 1 : ℝ)
  ( ( 2*x^2 + 2*x)/(x^2 - 1) - ( x^2 - x)/(x^2 - 2*x + 1) ) / ( x/(x + 1) )
  = -((2 * sqrt 3) / 3) + 1 := by 
  sorry

end simplify_and_evaluate_l130_130002


namespace tangent_line_circle_l130_130393

noncomputable section

def line_eqn (a : ℝ) : ℝ → ℝ → Prop := λ x y, 3 * x + 4 * y + a = 0
def circle_eqn : ℝ → ℝ → Prop := λ x y, (x - 1)^2 + y^2 = 1

theorem tangent_line_circle (a : ℝ) :
  (a = 2 ∨ a = -8) → ∀ x y, circle_eqn x y → line_eqn a x y →
    dist (1, 0) (x, y) = 1 :=
by 
  intro ha x y h_circle h_line
  sorry

end tangent_line_circle_l130_130393


namespace term_40_is_284_l130_130042

def contains_digit2 (n : ℕ) : Prop :=
  n.digits 10 any (=2)

def sequence : ℕ → ℕ 
| n := Nat.find (λ m, contains_digit2 m ∧ (m % 4 = 0 ∧ (sequence (n - 1) < m)))

noncomputable def term_40 : ℤ := sequence 40

theorem term_40_is_284 : term_40 = 284 := 
by
  sorry

end term_40_is_284_l130_130042


namespace probability_of_at_least_6_consecutive_heads_l130_130268

-- Define the conditions
def flip_options : finset (fin 9 → bool) := 
  finset.univ

def at_least_6_consecutive_heads (seq : fin 9 → bool) : bool :=
  (seq 0 && seq 1 && seq 2 && seq 3 && seq 4 && seq 5) ||
  (seq 1 && seq 2 && seq 3 && seq 4 && seq 5 && seq 6) ||
  (seq 2 && seq 3 && seq 4 && seq 5 && seq 6 && seq 7) ||
  (seq 3 && seq 4 && seq 5 && seq 6 && seq 7 && seq 8)

-- Define the theorem to prove the probability
theorem probability_of_at_least_6_consecutive_heads :
  (flip_options.filter at_least_6_consecutive_heads).card = 11 / 512 :=
by
  sorry

end probability_of_at_least_6_consecutive_heads_l130_130268


namespace division_of_repeating_decimal_l130_130069

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l130_130069


namespace beach_event_savings_l130_130324

noncomputable def discount_price (price : ℝ) (discount : ℝ) : ℝ :=
  price - (price * discount)

noncomputable def total_cost (items : List ℝ) : ℝ :=
  items.sum

theorem beach_event_savings :
  let price := 60
  let discounts := [0, 0.3, 0.5, 0.6, 0.6]
  let original_total := 5 * price
  let discounted_prices := List.map (λ d, discount_price price d) discounts
  let discounted_total := total_cost discounted_prices
  let savings := original_total - discounted_total
  let percentage_savings := (savings / original_total) * 100
  percentage_savings = 40 :=
by
  sorry

end beach_event_savings_l130_130324


namespace circle_equation_l130_130525

noncomputable theory

def center : ℝ × ℝ := (-2, 3)
def radius : ℝ := 2

theorem circle_equation :
  ∀ x y : ℝ, (x + 2) ^ 2 + (y - 3) ^ 2 = 4 ↔ (x - fst center) ^ 2 + (y - snd center) ^ 2 = radius ^ 2 :=
by
  sorry

end circle_equation_l130_130525


namespace range_f_l130_130723

noncomputable def f (x : ℝ) : ℝ := 2 * x / (x^2 - x + 1)

theorem range_f : set.Icc (-2/3 : ℝ) 2 = set.range f := 
by sorry

end range_f_l130_130723


namespace no_rain_five_days_l130_130958

-- Define the problem conditions and the required result.
def prob_rain := (2 / 3)
def prob_no_rain := (1 - prob_rain)
def prob_no_rain_five_days := prob_no_rain^5

theorem no_rain_five_days : 
  prob_no_rain_five_days = (1 / 243) :=
by
  sorry

end no_rain_five_days_l130_130958


namespace incenters_of_ACD_and_BCD_l130_130364

-- Define the context and assumptions
variables {A B C O M K L D : Type*} [right_triangle A B C] (h1 : right_angle C) (h2 : incenter O A B C) 
           (h3 : incircle_touches_hypotenuse M A B C) (h4 : circle_center_M_passes_through_O M O) 
           (h5 : circle_intersects_angle_bisectors K L M O (angle_bisector A) (angle_bisector B))
           (h6 : altitude_perpendicular CD AB)

-- State the theorem
theorem incenters_of_ACD_and_BCD : incenter K A C D ∧ incenter L B C D :=
by
  sorry -- Proof is omitted, only the statement is required.

end incenters_of_ACD_and_BCD_l130_130364


namespace stephanie_gas_payment_l130_130515

variables (electricity_bill : ℕ) (gas_bill : ℕ) (water_bill : ℕ) (internet_bill : ℕ)
variables (electricity_paid : ℕ) (gas_paid_fraction : ℚ) (water_paid_fraction : ℚ) (internet_paid : ℕ)
variables (additional_gas_payment : ℕ) (remaining_payment : ℕ) (expected_remaining : ℕ)

def stephanie_budget : Prop :=
  electricity_bill = 60 ∧
  electricity_paid = 60 ∧
  gas_bill = 40 ∧
  gas_paid_fraction = 3/4 ∧
  water_bill = 40 ∧
  water_paid_fraction = 1/2 ∧
  internet_bill = 25 ∧
  internet_paid = 4 * 5 ∧
  remaining_payment = 30 ∧
  expected_remaining = 
    (gas_bill - gas_paid_fraction * gas_bill) +
    (water_bill - water_paid_fraction * water_bill) + 
    (internet_bill - internet_paid) - 
    additional_gas_payment ∧
  expected_remaining = remaining_payment

theorem stephanie_gas_payment : additional_gas_payment = 5 :=
by sorry

end stephanie_gas_payment_l130_130515


namespace area_of_quadrilateral_ABCD_l130_130477

theorem area_of_quadrilateral_ABCD
    (A B C D : Type)
    (is_on_shore : A ∧ B ∧ C ∧ D)
    (ac_divides_island : ∀ (p : Type), (p ∈ A ∨ p ∈ C) → (p ∉ B ∨ p ∉ D))
    (bd_shorter_than_ac : ∀ (ac bd : ℝ), bd < ac)
    (cyclist_speed_asphalt : ∀ (asphalt_speed : ℝ), asphalt_speed = 15)
    (cyclist_time_travel : ∀ (p : in A ∨ in C ∨ in D) (t : ℝ), t = 2): 
    ∃ area,
      area = 450 :=
sorry

end area_of_quadrilateral_ABCD_l130_130477


namespace constant_term_in_binomial_expansion_l130_130764

theorem constant_term_in_binomial_expansion :
  (∃ x n : ℕ, (∀ r : ℕ, (n = 6) → (binomial n r) * (x ^ ((6 - 3 * r) / 2)) * (-2)^r = 60) → r = 2 → x^0) := 
sorry

end constant_term_in_binomial_expansion_l130_130764


namespace eight_div_repeating_three_l130_130089

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l130_130089


namespace sum_prime_factors_396_l130_130567

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factors (n : ℕ) : Finset ℕ := 
  (Finset.filter is_prime (Finset.range (n + 1))).filter (λ p, p ∣ n)

def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  (prime_factors n).sum

theorem sum_prime_factors_396 : sum_of_distinct_prime_factors 396 = 16 :=
  sorry

end sum_prime_factors_396_l130_130567


namespace Eight_div_by_repeating_decimal_0_3_l130_130166

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l130_130166


namespace composite_probability_l130_130035

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

theorem composite_probability :
  let total_numbers := 50 in
  let composite_count := total_numbers - 1 - 15 in -- 1 is neither prime nor composite; 15 primes under 50.
  (composite_count : ℝ) / total_numbers = 0.68 :=
by
  sorry

end composite_probability_l130_130035


namespace division_of_decimal_l130_130116

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l130_130116


namespace problem1_problem2_l130_130687

-- Proof Problem for (1)
theorem problem1 : -15 - (-5) + 6 = -4 := sorry

-- Proof Problem for (2)
theorem problem2 : 81 / (-9 / 5) * (5 / 9) = -25 := sorry

end problem1_problem2_l130_130687


namespace rectangle_perimeter_change_l130_130983

theorem rectangle_perimeter_change :
  ∀ (a b : ℝ), 
  (2 * (a + b) = 2 * (1.3 * a + 0.8 * b)) →
  ((2 * (0.8 * a + 1.95 * b) - 2 * (a + b)) / (2 * (a + b)) = 0.1) :=
by
  intros a b h
  sorry

end rectangle_perimeter_change_l130_130983


namespace always_in_range_l130_130731

noncomputable def g (x k : ℝ) : ℝ := x^2 + 2 * k * x + 1

theorem always_in_range (k : ℝ) : 
  ∃ x : ℝ, g x k = 3 :=
by
  sorry

end always_in_range_l130_130731


namespace sum_x_y_eq_8_l130_130849

theorem sum_x_y_eq_8 (x y S : ℝ) (h1 : x + y = S) (h2 : y - 3 * x = 7) (h3 : y - x = 7.5) : S = 8 :=
by
  sorry

end sum_x_y_eq_8_l130_130849


namespace number_of_integers_satisfying_inequality_l130_130811

theorem number_of_integers_satisfying_inequality : 
  {n : ℤ | (n + 2) * (n - 8) ≤ 0}.finite.card = 11 := by
  sorry

end number_of_integers_satisfying_inequality_l130_130811


namespace part1_part2_l130_130392

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := log x + (1 / 2) * x^2 - (b - 1) * x

theorem part1 (b : ℝ) :
  (∃ x : ℝ, x > 0 ∧ ∃ x₀ : ℝ, x₀ > 0 ∧ ∀ y : ℝ, y > 0 → g y b < g x₀ b) →
  b > 3 :=
sorry

theorem part2 (b : ℝ) (h_b : b ≥ 7 / 2) :
  let x1 := (- (1 / 2) * (b - 1) - sqrt ((1 / 4) * (b - 1)^2 - 1)) in
  let x2 := (- (1 / 2) * (b - 1) + sqrt ((1 / 4) * (b - 1)^2 - 1)) in
  x1 < x2 →
  g x1 b - g x2 b = (15 / 8) - 2 * log 2 :=
sorry

end part1_part2_l130_130392


namespace pairs_m_n_l130_130715

theorem pairs_m_n (m n : ℤ) (h : m ≠ n) : m^n = n^m ↔ (m = 2 ∧ n = 4) ∨ (m = 4 ∧ n = 2) ∨ (m = -2 ∧ n = -4) ∨ (m = -4 ∧ n = -2) :=
by
  sorry

end pairs_m_n_l130_130715


namespace age_group_caloric_allowance_l130_130851

theorem age_group_caloric_allowance
  (average_daily_allowance : ℕ)
  (daily_reduction : ℕ)
  (reduced_weekly_allowance : ℕ)
  (week_days : ℕ)
  (h1 : daily_reduction = 500)
  (h2 : week_days = 7)
  (h3 : reduced_weekly_allowance = 10500)
  (h4 : reduced_weekly_allowance = (average_daily_allowance - daily_reduction) * week_days) :
  average_daily_allowance = 2000 :=
sorry

end age_group_caloric_allowance_l130_130851


namespace range_of_m_l130_130948

theorem range_of_m {x m : ℤ} : 
  (∀ x : ℤ, 1 ≤ x ∧ x ≤ 4 → 3 * x - 3 * m ≤ -2 * m) ↔ 12 ≤ m ∧ m < 15 :=
by
  sorry

end range_of_m_l130_130948


namespace max_bonus_students_l130_130417

-- Define the number of students in the class
def number_of_students : ℕ := 150

-- Define the scores of the students
def scores (i : ℕ) : ℕ :=
  if i < 149 then 10 else 0

-- Define the total score
def total_score : ℕ :=
  Finset.sum (Finset.range 150) scores

-- Define the class average
noncomputable def class_average : ℚ :=
  total_score / number_of_students

-- Define predicate to check if a student receives bonus points
def receives_bonus (i : ℕ) : Prop :=
  scores i > class_average

-- Define the proof problem
theorem max_bonus_students : (Finset.filter receives_bonus (Finset.range number_of_students)).card = 149 :=
by
  sorry

end max_bonus_students_l130_130417


namespace polygon_angle_accuracy_l130_130743

def radius : ℝ := OA
def polygon_side_length : ℝ := radius / 16 -- calculation based on DE being approximated

noncomputable def alpha := 2 * arcsin (3 * real.sqrt 3 / 32)
noncomputable def total_angle := 19 * alpha
noncomputable def beta := 2 * π - total_angle

theorem polygon_angle_accuracy : 
  ∀ (n : ℕ), n = 19 →
  let DE := polygon_side_length,
      α := 18.6861112 * (π / 180),  -- α in radians based on 18 degrees 41.2 minutes
      total_angle := n * α,
      β := 2 * π - total_angle
  in 
  β = 4.95 * (π / 180) :=  -- 4 degrees 57 minutes in radians
begin
  intros n hn,
  rw hn,
  let DE : ℝ := 3 * real.sqrt 3 / 32 * radius,
  let α := 2 * real.arcsin (DE / (2 * radius)),
  let total_angle := 19 * α,
  let β := 2 * π - total_angle,
  exact β = 4.95 * (π / 180), -- Expected β in radians.
  sorry
end

end polygon_angle_accuracy_l130_130743


namespace ensure_winning_at_least_once_l130_130532

noncomputable def minimum_tickets (p : ℝ) (P : ℝ) : ℝ :=
if P ≥ 0.95 then 3 / p else 0

theorem ensure_winning_at_least_once (p : ℝ) (P : ℝ) (n : ℝ) :
  p = 0.01 →
  P ≥ 0.95 →
  n ≥ 3 / p :=
  by
  intros h1 h2
  rw h1
  norm_num at h2
  norm_num
  linarith

end ensure_winning_at_least_once_l130_130532


namespace symmetric_point_origin_l130_130523

-- Define the notion of symmetry with respect to the origin
def symmetric_with_origin (p : ℤ × ℤ) : ℤ × ℤ :=
  (-p.1, -p.2)

-- Define the given point
def given_point : ℤ × ℤ :=
  (-2, 5)

-- State the theorem to be proven
theorem symmetric_point_origin : 
  symmetric_with_origin given_point = (2, -5) :=
by 
  -- The proof will go here, use sorry for now
  sorry

end symmetric_point_origin_l130_130523


namespace perimeter_of_floor_l130_130984

-- Define the side length of the room's floor
def side_length : ℕ := 5

-- Define the formula for the perimeter of a square
def perimeter_of_square (side : ℕ) : ℕ := 4 * side

-- State the theorem: the perimeter of the floor of the room is 20 meters
theorem perimeter_of_floor : perimeter_of_square side_length = 20 :=
by
  sorry

end perimeter_of_floor_l130_130984


namespace president_vice_president_opposite_genders_l130_130906

theorem president_vice_president_opposite_genders (total_members boys girls : ℕ) :
  total_members = 30 → boys = 18 → girls = 12 →
  (boys * girls + girls * boys) = 432 :=
by
  intros h_total h_boys h_girls
  rw [h_boys, h_girls]
  norm_num
  sorry

end president_vice_president_opposite_genders_l130_130906


namespace transformed_fn_fixed_point_l130_130941

-- Define the exponential function
def exp_fn (a : ℝ) (x : ℝ) : ℝ := a ^ x

-- Define the transformed function
def transformed_fn (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1) + 1

-- Define the fixed points
def fixed_point_f : Prop := ∀ a : ℝ, exp_fn a 0 = 1

-- Define the final theorem
theorem transformed_fn_fixed_point (a : ℝ) (h : fixed_point_f a) : transformed_fn a 1 = 2 :=
by
  sorry

end transformed_fn_fixed_point_l130_130941


namespace morse_code_count_l130_130415

noncomputable def morse_code_sequences : Nat :=
  let case_1 := 2            -- 1 dot or dash
  let case_2 := 2 * 2        -- 2 dots or dashes
  let case_3 := 2 * 2 * 2    -- 3 dots or dashes
  let case_4 := 2 * 2 * 2 * 2-- 4 dots or dashes
  let case_5 := 2 * 2 * 2 * 2 * 2 -- 5 dots or dashes
  case_1 + case_2 + case_3 + case_4 + case_5

theorem morse_code_count : morse_code_sequences = 62 := by
  sorry

end morse_code_count_l130_130415


namespace buses_fewer_than_cars_l130_130039

-- Define the conditions
def ratio_buses_to_cars : ℚ := 1 / 3
def number_of_cars : ℕ := 60

-- Define the function to calculate the number of buses from the number of cars
def number_of_buses (cars : ℕ) (ratio : ℚ) : ℕ :=
  (cars * ratio.num : ℚ / (ratio.denom : ℚ)).toNat

-- Define the condition that there are fewer buses than cars
def fewer_buses (cars : ℕ) (buses : ℕ) : ℕ :=
  cars - buses

-- Theorem statement
theorem buses_fewer_than_cars : 
  fewer_buses number_of_cars (number_of_buses number_of_cars ratio_buses_to_cars) = 40 := 
sorry

end buses_fewer_than_cars_l130_130039


namespace division_of_repeating_decimal_l130_130072

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l130_130072


namespace intersection_points_eq_10_l130_130409

noncomputable def f (x : ℝ) : ℝ := |x| -- assuming f is defined suitably with given properties

theorem intersection_points_eq_10 :
  (∀ x : ℝ, f(x + 2) = f(x)) →
  (∀ x : Ioo (-1 : ℝ) 1, f x = |x|) →
  (∃! t, t ∈ ( ⋂ (x : ℝ), {x | f x = log 4 |x|} ) ∧ (#intersections t = 10)) :=
sorry

end intersection_points_eq_10_l130_130409


namespace range_of_t_max_radius_circle_eq_l130_130778

-- Definitions based on conditions
def circle_equation (x y t : ℝ) := x^2 + y^2 - 2 * x + t^2 = 0

-- Statement for the range of values of t
theorem range_of_t (t : ℝ) (h : ∃ x y : ℝ, circle_equation x y t) : -1 < t ∧ t < 1 := sorry

-- Statement for the equation of the circle when t = 0
theorem max_radius_circle_eq (x y : ℝ) (h : circle_equation x y 0) : (x - 1)^2 + y^2 = 1 := sorry

end range_of_t_max_radius_circle_eq_l130_130778


namespace magnitude_complex_div_l130_130759

theorem magnitude_complex_div (i : ℂ) (hi : i = complex.I) : complex.abs(i / (2 - i)) = real.sqrt(5) / 5 :=
by
  -- Proof goes here
  sorry

end magnitude_complex_div_l130_130759


namespace rationalize_denominator_correct_l130_130502

noncomputable def rationalize_denominator : Prop :=
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l130_130502


namespace range_of_m_l130_130910

theorem range_of_m (m y1 y2 k : ℝ) (h1 : y1 = -2 * (m - 2) ^ 2 + k) (h2 : y2 = -2 * (m - 1) ^ 2 + k) (h3 : y1 > y2) : m > 3 / 2 := 
sorry

end range_of_m_l130_130910


namespace sequence_a_n_formula_sequence_b_n_range_l130_130345

theorem sequence_a_n_formula (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) :
  (∀ n, S_n n > 0 ∧ S_n n ^ 2 - (n^2 + n - 1) * S_n n - (n^2 + n) = 0) →
  a_n 1 = 2 ∧ (∀ n, n ≥ 2 → a_n n = S_n n - S_n (n - 1)) →
  (∀ n, a_n n = 2 * n) :=
by
  sorry

theorem sequence_b_n_range (a_n b_n : ℕ → ℝ) (T_n : ℕ → ℝ) (m : ℝ) :
  (∀ n, a_n n = 2 * n) →
  (∀ n, (n + 2)^2 * b_n n = (n + 1) / (a_n n)^2) →
  (∀ n, T_n n = ∑ i in finset.range n, b_n i) →
  (∀ n, T_n n < m^2 / 5) →
  m ∈ Set.Icc (-5.0 / 8) (5.0 / 8) ∨ m ∈ Set.Iio (-5.0 / 8) ∨ m ∈ Set.Ioi (5.0 / 8) :=
by
  sorry

end sequence_a_n_formula_sequence_b_n_range_l130_130345


namespace replace_half_black_cubes_l130_130257

-- Define the conditions
def is_black_cube (n : ℕ) (i j k : ℕ) : Prop :=
-- Mock-up property for black cubes
sorry

def valid_subprism (n : ℕ) (axis : ℕ) (index : ℕ) : Prop :=
-- Mock-up property for subprisms containing exactly two black cubes
-- and the two black cubes are separated by an even number of white cubes
sorry

-- Define the cube's property for the X and Y cubes
def is_X_cube (i j k : ℕ) : Prop :=
(i % 2 + j % 2 + k % 2) % 2 = 0

def is_Y_cube (i j k : ℕ) : Prop :=
(i % 2 + j % 2 + k % 2) % 2 = 1

-- Prove that we can replace half of the black cubes such that the condition is met
theorem replace_half_black_cubes
  (n : ℕ)
  (h1 : ∀ i j k, i < n ∧ j < n ∧ k < n → is_black_cube n i j k)
  (h2 : ∀ axis index, index < n → valid_subprism n axis index) :
  ∃ f : ℕ × ℕ × ℕ → Prop,
    (∀ i j k, i < n ∧ j < n ∧ k < n → f (i, j, k) = is_Y_cube i j k) ∧
    (∀ axis index, index < n →
      ∃! k_i, 0 ≤ k_i < n ∧ is_black_cube n k_i (index axis i) ∧ f (k_i, index axis i)) :=
sorry

end replace_half_black_cubes_l130_130257


namespace alpha_value_l130_130457

theorem alpha_value (α : ℝ) (h1 : α ∈ {-2, -1/2, 2/3, 3})
  (h2 : ∀ x : ℝ, y = x^α → y = (-x)^α)
  (h3 : ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → x₁^α < x₂^α) : α = 2/3 := sorry

end alpha_value_l130_130457


namespace perpendicular_iff_collinear_l130_130675

theorem perpendicular_iff_collinear
  (O O1 O2 M N S T : Point)
  (h1 : Circle O1 M N)
  (h2 : Circle O2 M N)
  (h3 : Circle O S)
  (h4 : Circle O T)
  (tangent_O1_O : Tangent O1 O S)
  (tangent_O2_O : Tangent O2 O T) :
  (Perpendicular O M N) ↔ Collinear S N T := by
  sorry

end perpendicular_iff_collinear_l130_130675


namespace algebraic_expression_zero_iff_x_eq_2_l130_130226

theorem algebraic_expression_zero_iff_x_eq_2 (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (1 / (x - 1) + 3 / (1 - x^2) = 0) ↔ (x = 2) :=
by
  sorry

end algebraic_expression_zero_iff_x_eq_2_l130_130226


namespace min_max_m_l130_130398

theorem min_max_m
  (a b c : ℝ)
  (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c)
  (h3 : 3 * a + 2 * b + c = 5)
  (h4 : 2 * a + b - 3 * c = 1)
  (hc_min : c = 3/7)
  (hc_max : c = 7/11) :
  (m_min : ℝ := 3 * a + b - 7 * c) ∧ (m_min = -5/7) ∧ 
  (m_max : ℝ := 3 * a + b - 7 * c) ∧ (m_max = -1/11) :=
by
  sorry

end min_max_m_l130_130398


namespace C_share_l130_130246

theorem C_share (a b c : ℕ) (h1 : a + b + c = 1010)
                (h2 : ∃ k : ℕ, a = 3 * k + 25 ∧ b = 2 * k + 10 ∧ c = 5 * k + 15) : c = 495 :=
by
  -- Sorry is used to skip the proof
  sorry

end C_share_l130_130246


namespace ratio_of_x_y_l130_130353

theorem ratio_of_x_y (a x y : ℝ) (h1 : log a ((x - y) / 2) = (log a x + log a y) / 2) 
  (hx_pos : x > 0) (hy_pos : y > 0) (hx_gt_hy : x > y) : 
  (x / y = 3 + 2 * Real.sqrt 2) ∨ (x / y = 3 - 2 * Real.sqrt 2) :=
sorry

end ratio_of_x_y_l130_130353


namespace induction_step_method_l130_130993

theorem induction_step_method (x y : ℤ) (P : ℕ → Prop) (k : ℕ) (odd_k : k % 2 = 1) :
  (P k → P (k + 2)) → (∀ n, n % 2 = 1 → P n) :=
  by
    intro H
    intro n
    intro odd_n
    obtain ⟨m, rfl⟩ : ∃ m, n = 2 * m + 1 :=
      by sorry -- Placeholder for proof of n being an odd number
    induction m
    case succ m ih =>
      apply H
      exact ih
    case zero => sorry -- Placeholder for proving the base case

end induction_step_method_l130_130993


namespace sector_properties_l130_130285

-- Define the given conditions
def arc_length : ℝ := 6
def radius : ℝ := 3
def alpha := arc_length / radius

-- Define the statements to prove
theorem sector_properties :
    (1/2 * radius^2 * alpha = 9) ∧ (let c := 2 * radius * sin(1) in c = 6 * sin(1)) :=
by
  have h_alpha : alpha = 2 := by sorry
  have h_area : (1/2 * radius^2 * alpha = 9) := by sorry
  have h_chord : (let c := 2 * radius * sin(1) in c = 6 * sin(1)) := by sorry
  exact ⟨h_area, h_chord⟩

end sector_properties_l130_130285


namespace rattlesnakes_count_l130_130046

-- Definitions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def rattlesnakes : ℕ := total_snakes - (boa_constrictors + pythons)

-- Theorem to prove
theorem rattlesnakes_count : rattlesnakes = 40 := by
  -- provide proof here
  sorry

end rattlesnakes_count_l130_130046


namespace Eight_div_by_repeating_decimal_0_3_l130_130169

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l130_130169


namespace log_comparison_l130_130757

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4
noncomputable def log6 (x : ℝ) : ℝ := Real.log x / Real.log 6

theorem log_comparison :
  let a := log2 6
  let b := log4 12
  let c := log6 18
  a > b ∧ b > c :=
by 
  sorry

end log_comparison_l130_130757


namespace polynomial_quotient_correct_l130_130337

noncomputable def polynomial_division_quotient : Polynomial ℝ :=
  (Polynomial.C 1 * Polynomial.X^6 + Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 8) / (Polynomial.X - Polynomial.C 1)

-- Math proof statement
theorem polynomial_quotient_correct :
  polynomial_division_quotient = Polynomial.C 1 * Polynomial.X^5 + Polynomial.C 1 * Polynomial.X^4 
                                 + Polynomial.C 1 * Polynomial.X^3 + Polynomial.C 1 * Polynomial.X^2 
                                 + Polynomial.C 3 * Polynomial.X + Polynomial.C 3 :=
by
  sorry

end polynomial_quotient_correct_l130_130337


namespace valid_arrangements_count_l130_130051

-- Definitions of the problem
def products : Type := { A, B, C, D, E }

def AB_together (arrangement : list products) : Prop :=
  ∃ n, n + 1 < arrangement.length ∧ (arrangement.get_or_else n A) = A ∧ (arrangement.get_or_else (n + 1) B) = B

def CD_not_together (arrangement : list products) : Prop :=
  ∀ n, n + 1 < arrangement.length → ¬((arrangement.get_or_else n C) = C ∧ (arrangement.get_or_else (n + 1) D) = D)

-- Proving the total number of valid arrangements
theorem valid_arrangements_count : 
  let arrangements := { l : list products | multiset.card (multiset.of_list l) = list.length l ∧ 
                                       l.length = 5 ∧ AB_together l ∧ CD_not_together l }
  in arrangements.card = 24 := 
sorry

end valid_arrangements_count_l130_130051


namespace zain_coin_total_l130_130236

def zain_coins (q d n : ℕ) := q + d + n
def emerie_quarters : ℕ := 6
def emerie_dimes : ℕ := 7
def emerie_nickels : ℕ := 5
def zain_quarters : ℕ := emerie_quarters + 10
def zain_dimes : ℕ := emerie_dimes + 10
def zain_nickels : ℕ := emerie_nickels + 10

theorem zain_coin_total : zain_coins zain_quarters zain_dimes zain_nickels = 48 := 
by
  unfold zain_coins zain_quarters zain_dimes zain_nickels emerie_quarters emerie_dimes emerie_nickels
  rfl

end zain_coin_total_l130_130236


namespace eight_div_repeating_three_l130_130195

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l130_130195


namespace area_enclosed_l130_130557

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (x - Real.pi / 3)
noncomputable def area_between (a b : ℝ) (f g : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, (g x - f x)

theorem area_enclosed (h₀ : 0 ≤ 2 * Real.pi) (h₁ : 2 * Real.pi ≤ 2 * Real.pi) :
  area_between (2 * Real.pi / 3) (5 * Real.pi / 3) f g = 2 :=
by 
  sorry

end area_enclosed_l130_130557


namespace number_of_integers_satisfying_inequality_l130_130812

theorem number_of_integers_satisfying_inequality : 
  {n : ℤ | (n + 2) * (n - 8) ≤ 0}.finite.card = 11 := by
  sorry

end number_of_integers_satisfying_inequality_l130_130812


namespace greatest_possible_ratio_l130_130702

noncomputable def max_ratio_AB_CD_over_EF : ℝ :=
  10 * Real.sqrt 2

-- Definitions of points given their properties.
structure Point where
  x : ℤ
  y : ℤ
  property : x ^ 2 + y ^ 2 = 64

-- Distinct points on the circle with integer coordinates.
variables (A B C D E F : Point)

-- Conditions
axiom distinct_points : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
                        B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
                        C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
                        D ≠ E ∧ D ≠ F ∧
                        E ≠ F

def distance (P Q : Point) : ℝ :=
  Real.sqrt (Int.toReal ((P.x - Q.x)^2 + (P.y - Q.y)^2))

-- Distances between points are irrational
axiom irrational_distances :
  Irrational (distance A B) ∧
  Irrational (distance C D) ∧
  Irrational (distance E F)

-- Problem statement in Lean
theorem greatest_possible_ratio :
  ∃ (A B C D E F : Point),
  distinct_points A B C D E F ∧
  irrational_distances A B C D E F ∧
  max_ratio_AB_CD_over_EF = (distance A B * distance C D) / (distance E F) :=
sorry

end greatest_possible_ratio_l130_130702


namespace minimum_number_of_rooks_l130_130666

theorem minimum_number_of_rooks (n : ℕ) : 
  ∃ (num_rooks : ℕ), (∀ (cells_colored : ℕ), cells_colored = n^2 → num_rooks = n) :=
by sorry

end minimum_number_of_rooks_l130_130666


namespace sams_charge_per_sheet_is_1_5_l130_130649

variable (x : ℝ)
variable (a : ℝ) -- John's Photo World's charge per sheet
variable (b : ℝ) -- Sam's Picture Emporium's one-time sitting fee
variable (c : ℝ) -- John's Photo World's one-time sitting fee
variable (n : ℕ) -- Number of sheets

def johnsCost (n : ℕ) (a c : ℝ) := n * a + c
def samsCost (n : ℕ) (x b : ℝ) := n * x + b

theorem sams_charge_per_sheet_is_1_5 :
  ∀ (a b c : ℝ) (n : ℕ), a = 2.75 → b = 140 → c = 125 → n = 12 →
  johnsCost n a c = samsCost n x b → x = 1.50 := by
  intros a b c n ha hb hc hn h
  sorry

end sams_charge_per_sheet_is_1_5_l130_130649


namespace ant_cube_visits_l130_130295

theorem ant_cube_visits
  (vertices : Fin 8) -- the vertices of the cube
  (checkerboard : vertices → Bool) -- the checkerboard labeling
  (visits : vertices → Nat) -- the number of visits for each vertex
  (no_backtracking : ∀ (u v : vertices), visits u ≠ visits u.succ := if v = u.succ then visits u.succ ≠ visits u else visits v ≠ visits u) :
  (∃ (v : vertices), visits v = 25 ∧ (∀ (u : vertices), u ≠ v → visits u = 20)) → False :=
by
  intro h
  sorry

end ant_cube_visits_l130_130295


namespace eight_div_repeating_three_l130_130087

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l130_130087


namespace find_a_if_even_function_l130_130833

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem find_a_if_even_function :
  (∃ a : ℝ, is_even_function (f a)) → ∃ a : ℝ, a = 2 :=
by
  intros h
  sorry

end find_a_if_even_function_l130_130833


namespace platform_length_correct_l130_130631

/-
Conditions:
1. speed of train (120 kmph)
2. time to pass a man (22 seconds)
3. time to cross a platform (45 seconds)

Conclusion:
Length of platform = 766.59 meters
-/

def speed_kmph : ℝ := 120
def time_to_pass_man : ℝ := 22
def time_to_cross_platform : ℝ := 45
def speed_mps : ℝ := speed_kmph * 1000 / 3600
def distance_passing_man : ℝ := speed_mps * time_to_pass_man
def distance_crossing_platform : ℝ := speed_mps * time_to_cross_platform

theorem platform_length_correct :
  distance_crossing_platform - distance_passing_man = 766.59 := by
  sorry

end platform_length_correct_l130_130631


namespace initial_fee_of_first_plan_equals_65_l130_130796

theorem initial_fee_of_first_plan_equals_65 (miles : ℕ) (F : ℝ) 
  (h_cost_eq : ∀ miles, F + 0.40 * miles = 0.60 * miles) : F = 65 :=
by
  have key_eq : F + 0.40 * (325 : ℝ) = 0.60 * (325 : ℝ) := h_cost_eq 325
  rw [show 0.40 * 325 = 130, by norm_num] at key_eq
  rw [show 0.60 * 325 = 195, by norm_num] at key_eq
  linarith

end initial_fee_of_first_plan_equals_65_l130_130796


namespace division_of_repeating_decimal_l130_130078

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l130_130078


namespace sum_abs_diff_eq_n_squared_l130_130543

theorem sum_abs_diff_eq_n_squared (n : ℕ)
  (a : Fin n → ℕ) (b : Fin n → ℕ)
  (h1 : ∀ i j, i < j → a i > a j)
  (h2 : ∀ i j, i < j → b i < b j)
  (h3 : ∀ x, x ∈ (Finset.univ.image a ∪ Finset.univ.image b) ↔ 1 ≤ x ∧ x ≤ 2*n) :
  ∑ i in Finset.range n, |a i - b i| = n^2 :=
by sorry

end sum_abs_diff_eq_n_squared_l130_130543


namespace division_by_repeating_decimal_l130_130215

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l130_130215


namespace xiao_ming_kite_payment_l130_130583

/-- Xiao Ming has multiple 1 yuan, 2 yuan, and 5 yuan banknotes. 
    He wants to buy a kite priced at 18 yuan using no more than 10 of these banknotes
    and must use at least two different denominations.
    Prove that there are exactly 11 different ways he can pay. -/
theorem xiao_ming_kite_payment : 
  ∃ (combinations : Nat), 
    (∀ (c1 c2 c5 : Nat), (c1 * 1 + c2 * 2 + c5 * 5 = 18) → 
    (c1 + c2 + c5 ≤ 10) → 
    ((c1 > 0 ∧ c2 > 0) ∨ (c1 > 0 ∧ c5 > 0) ∨ (c2 > 0 ∧ c5 > 0)) →
    combinations = 11) :=
sorry

end xiao_ming_kite_payment_l130_130583


namespace final_position_point_A_l130_130054

theorem final_position_point_A :
  let initial_position := -3
  let movement := 4.5
  let final_position := initial_position + movement
  final_position = 1.5 :=
by
  simp only [*]
  exact rfl

end final_position_point_A_l130_130054


namespace Dan_speed_must_exceed_45_mph_l130_130018

theorem Dan_speed_must_exceed_45_mph : 
  ∀ (distance speed_Cara time_lag time_required speed_Dan : ℝ),
    distance = 180 →
    speed_Cara = 30 →
    time_lag = 2 →
    time_required = 4 →
    (distance / speed_Cara) = 6 →
    (∀ t, t = distance / speed_Dan → t < time_required) →
    speed_Dan > 45 :=
by
  intro distance speed_Cara time_lag time_required speed_Dan
  intro h1 h2 h3 h4 h5 h6
  sorry

end Dan_speed_must_exceed_45_mph_l130_130018


namespace eight_div_repeating_three_l130_130144

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l130_130144


namespace division_by_repeating_decimal_l130_130203

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l130_130203


namespace problem_solution_l130_130747

noncomputable def S (n : ℕ) (a : ℕ → ℕ) : ℕ := 2 * a n - 2
noncomputable def a (n : ℕ) : ℕ := 2 ^ n
noncomputable def b (n : ℕ) : ℝ :=
  if n % 2 = 1 then (Real.log2 (a n)) / (n^2 * (n + 2))
  else 2 * n / (a n)

noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range (2 * n), b i

theorem problem_solution (n : ℕ) : 
  T n = (n : ℝ) / (2 * n + 1) + 16 / 9 - (4 + 3 * (n : ℝ)) / (9 * 4 ^ (n - 1)) :=
sorry

end problem_solution_l130_130747


namespace xiao_gang_steps_l130_130439

theorem xiao_gang_steps (x : ℕ) (H1 : 9000 / x = 13500 / (x + 15)) : x = 30 :=
by
  sorry

end xiao_gang_steps_l130_130439


namespace sum_first_10_terms_l130_130426

-- Definitions for the arithmetic sequence and its sum
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

def sum_first_n_terms (a d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions
variable (a d : ℝ)
variable (a3_condition : arithmetic_sequence a d 3 = 2)
variable (a6_a10_condition : arithmetic_sequence a d 6 + arithmetic_sequence a d 10 = 20)

-- Statement to prove
theorem sum_first_10_terms : sum_first_n_terms a d 10 = 60 :=
sorry

end sum_first_10_terms_l130_130426


namespace zain_coin_total_l130_130238

def zain_coins (q d n : ℕ) := q + d + n
def emerie_quarters : ℕ := 6
def emerie_dimes : ℕ := 7
def emerie_nickels : ℕ := 5
def zain_quarters : ℕ := emerie_quarters + 10
def zain_dimes : ℕ := emerie_dimes + 10
def zain_nickels : ℕ := emerie_nickels + 10

theorem zain_coin_total : zain_coins zain_quarters zain_dimes zain_nickels = 48 := 
by
  unfold zain_coins zain_quarters zain_dimes zain_nickels emerie_quarters emerie_dimes emerie_nickels
  rfl

end zain_coin_total_l130_130238


namespace find_a_l130_130708

-- Definitions based on conditions
def integral_formula (a : ℝ) : Prop :=
  ∫ x in 0..(Real.pi / 2), (Real.sin x + a * Real.cos x) = 2

-- Lean statement to prove
theorem find_a (a : ℝ) (h : integral_formula a) : a = 1 :=
sorry

end find_a_l130_130708


namespace discount_percentage_is_correct_l130_130469

noncomputable def cost_prices := [540, 660, 780]
noncomputable def markup_percentages := [0.15, 0.20, 0.25]
noncomputable def selling_prices := [496.80, 600, 750]

noncomputable def marked_price (cost : ℝ) (markup : ℝ) : ℝ := cost + (markup * cost)

noncomputable def total_marked_price : ℝ := 
  (marked_price 540 0.15) + (marked_price 660 0.20) + (marked_price 780 0.25)

noncomputable def total_selling_price : ℝ := 496.80 + 600 + 750

noncomputable def overall_discount_percentage : ℝ :=
  ((total_marked_price - total_selling_price) / total_marked_price) * 100

theorem discount_percentage_is_correct : overall_discount_percentage = 22.65 :=
by
  sorry

end discount_percentage_is_correct_l130_130469


namespace yoongi_class_combination_l130_130584

theorem yoongi_class_combination : (Nat.choose 10 3 = 120) := by
  sorry

end yoongi_class_combination_l130_130584


namespace computation_check_l130_130387

def f (x : ℝ) : ℝ :=
  if x >= 3 then real.sqrt (x + 1) else -2 * x + 8

theorem computation_check : f (f (-2)) = real.sqrt 13 :=
by
  sorry

end computation_check_l130_130387


namespace log_inequality_l130_130356

noncomputable def a : ℝ := Real.log 3.6 / Real.log 2
noncomputable def b : ℝ := Real.log 3.2 / Real.log 4
noncomputable def c : ℝ := Real.log 3.6 / Real.log 4

theorem log_inequality : a > c ∧ c > b :=
by {
  -- Proof goes here
  sorry
}

end log_inequality_l130_130356


namespace polynomial_existence_l130_130445

theorem polynomial_existence :
  ∃ f : ℝ[X],
  (f.coeff 0 = 0) ∧
  (∀ a b : ℤ, a ≠ b → ∃ k : ℤ, f.eval a - f.eval b = (a - b) * ↑k) ∧
  ¬ ∀ n : ℕ, is_integer (f.coeff n) :=
by
  let f : ℝ[X] := (1/2) * (X^4 + X^2),
  use f,
  sorry

end polynomial_existence_l130_130445


namespace passing_probability_l130_130863

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem passing_probability :
  let total_questions := 5
  let questions_to_select := 3
  let known_questions := 3
  let unknown_questions := total_questions - known_questions
  let passing_threshold := 2
  let scenario1 := combination 3 3 / combination total_questions questions_to_select
  let scenario2 := combination 3 2 * combination 2 1 / combination total_questions questions_to_select
  (scenario1 + scenario2) = 0.7 :=
by
  sorry

end passing_probability_l130_130863


namespace valid_number_of_knights_l130_130547

inductive Inhabitant
| knight
| liar

def inhabitants : List Inhabitant := 
  [Inhabitant.knight, Inhabitant.liar, Inhabitant.liar, Inhabitant.liar, Inhabitant.liar, Inhabitant.liar]

def statement_correct {self : Inhabitant} (others: List Inhabitant) :=
  (self = Inhabitant.knight → (count others Inhabitant.liar = 4)) ∧ 
  (self = Inhabitant.liar → (count others Inhabitant.liar ≠ 4))

theorem valid_number_of_knights (n : ℕ) (h₀ : n = 0 ∨ n = 2) :
  (∀ i, i < 6 → statement_correct Inhabitant.knight (inhabitants.remove_nth i)) ∨ 
  (∀ i, i < 6 → statement_correct Inhabitant.liar (inhabitants.remove_nth i)) :=
sorry

end valid_number_of_knights_l130_130547


namespace friendly_point_pairs_l130_130377

def friendly_points (k : ℝ) (a : ℝ) (A B : ℝ × ℝ) : Prop :=
  A = (a, -1 / a) ∧ B = (-a, 1 / a) ∧
  B.2 = k * B.1 + 1 + k

theorem friendly_point_pairs : ∀ (k : ℝ), k ≥ 0 → 
  ∃ n, (n = 1 ∨ n = 2) ∧
  (∀ a : ℝ, a > 0 →
    friendly_points k a (a, -1 / a) (-a, 1 / a))
:= by
  sorry

end friendly_point_pairs_l130_130377


namespace rationalize_denominator_l130_130490

theorem rationalize_denominator : (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_denominator_l130_130490


namespace average_speed_of_train_l130_130657

-- Definitions based on the conditions
def distance1 : ℝ := 325
def distance2 : ℝ := 470
def time1 : ℝ := 3.5
def time2 : ℝ := 4

-- Proof statement
theorem average_speed_of_train :
  (distance1 + distance2) / (time1 + time2) = 106 := 
by 
  sorry

end average_speed_of_train_l130_130657


namespace Eight_div_by_repeating_decimal_0_3_l130_130165

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l130_130165


namespace XY_squared_l130_130424

-- Define the parallelogram and related conditions
variables (A B C D X Y : Type)
variable  [isParallelogram : parallelogram A B C D]
variable  [length_AB : length A B = 15]
variable  [length_BC : length B C = 15]
variable  [length_CD : length C D = 20]
variable  [length_DA : length D A = 20]
variable  [angle_D : angle D = 45]
variable  [midpoint_X : isMidpoint X B C]
variable  [midpoint_Y : isMidpoint Y D A]

-- Define proof problem statement
theorem XY_squared (XY : length X Y) : XY^2 = 1250 := sorry

end XY_squared_l130_130424


namespace sequence_contains_exactly_3_integers_l130_130041

def sequence (a : ℕ) : ℕ → ℕ
| 0 => a
| (n + 1) => sequence n / 4

theorem sequence_contains_exactly_3_integers (a : ℕ) (h : a = 7200) :
  ∃ n : ℕ, n = 2 ∧ ∀ m : ℕ, (m <= 2) → ((sequence a m) ∈ ℕ) :=
sorry

end sequence_contains_exactly_3_integers_l130_130041


namespace divide_by_repeating_decimal_l130_130179

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l130_130179


namespace divide_by_repeating_decimal_l130_130182

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l130_130182


namespace erdos_mordell_l130_130481

-- Definitions for vertices, volume, and radius assumed from conditions.
variables {A B C D : Type} [euclidean_space A B C D]

-- Hypotheses concerning the tetrahedron and its properties
def V (A B C D : Type) : ℝ := sorry  -- Volume of the tetrahedron
def R (A B C D : Type) : ℝ := sorry  -- Radius of the circumscribed sphere

-- Statement of the theorem
theorem erdos_mordell (A B C D : Type) [euclidean_space A B C D] :
  ∃ (T : Type) [triangle T], 
  (∀ (x1 x2 x3 : Type), 
    (opposite_edge_length_product A B C D x1 x2 x3 = 
    side_length T x1 x2 ∧ 
    side_length T x2 x3 ∧ 
    side_length T x3 x1)) ∧
  (area T = 6 * V A B C D * R A B C D) :=
sorry

-- Placeholder for opposite edge length function
def opposite_edge_length_product (A B C D : Type) : 
  Type → Type → Type → ℝ := sorry

-- Placeholder for side length of a triangle function
def side_length (T : Type) : 
  Type → Type → ℝ := sorry

-- Placeholder for area of a triangle
def area (T : Type) : ℝ := sorry

end erdos_mordell_l130_130481


namespace relationship_is_uncertain_l130_130402

open ProbabilityTheory MeasureTheory

-- Variables defining the events and probabilities
variables {Ω : Type*} {P : MeasureTheory.ProbabilityMeasure Ω}

-- Events A and B
variables (A B : Set Ω)

-- Conditions
axiom cond : P (A ∪ B) = P A + P B

-- Statement to prove
theorem relationship_is_uncertain : (A ∩ B = ∅ ∧ P A + P B = 1) ∨ (¬(A ∩ B = ∅) ∧ P A + P B = 1) :=
sorry

end relationship_is_uncertain_l130_130402


namespace not_center_of_symmetry_l130_130576

theorem not_center_of_symmetry (x : ℝ) :
  ¬ (∃ k : ℤ, x = (k * Real.pi) / 4 + Real.pi / 8) :=
by
  intro h
  cases h with k hk
  sorry -- proof is omitted

end not_center_of_symmetry_l130_130576


namespace probability_no_rain_next_five_days_eq_1_over_243_l130_130961

theorem probability_no_rain_next_five_days_eq_1_over_243 :
  let p_rain : ℚ := 2 / 3 in
  let p_no_rain : ℚ := 1 - p_rain in
  let probability_no_rain_five_days : ℚ := p_no_rain ^ 5 in
  probability_no_rain_five_days = 1 / 243 :=
by
  sorry

end probability_no_rain_next_five_days_eq_1_over_243_l130_130961


namespace number_of_integers_satisfying_inequality_l130_130816

theorem number_of_integers_satisfying_inequality :
  set.count {n : ℤ | (n + 2) * (n - 8) ≤ 0} = 11 :=
sorry

end number_of_integers_satisfying_inequality_l130_130816


namespace simplify_tan_cot_l130_130917

theorem simplify_tan_cot (x : ℝ) (h1 : tan x ≠ 0) (h2 : 1 + 1 / tan x ≠ 0) :
    (tan x / (1 + (1 / tan x)) + (1 + 1 / tan x) / tan x) = (sec x) ^ 2 + (csc x) ^ 2 :=
by 
  sorry

end simplify_tan_cot_l130_130917


namespace sphere_in_cone_volume_l130_130283

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem sphere_in_cone_volume :
  let d := 12
  let θ := 45
  let r := 3 * Real.sqrt 2
  let V := volume_of_sphere r
  d = 12 → θ = 45 → V = 72 * Real.sqrt 2 * Real.pi := by
  intros h1 h2
  sorry

end sphere_in_cone_volume_l130_130283


namespace dennis_loose_coins_l130_130662

theorem dennis_loose_coins (initial_money shirts_cost : ℕ) (ten_bills : ℕ) (change : ℕ) (loose_coins : ℕ) :
  initial_money = 50 ∧ shirts_cost = 27 ∧ ten_bills = 2 ∧ change = initial_money - shirts_cost ∧ loose_coins = change - (ten_bills * 10) →
  loose_coins = 3 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  cases h6 with h7 h8
  rw [h1, h3] at h7
  have h9 : change = 23 := h7
  have h10 : loose_coins = 23 - 20 := by rwa [h9, ←h8]
  norm_num at h10
  exact h10

end dennis_loose_coins_l130_130662


namespace number_of_people_in_group_is_three_l130_130274

noncomputable def total_bill : ℝ := 139.00
noncomputable def tip_percentage : ℝ := 0.10
noncomputable def share_per_person : ℝ := 50.97

theorem number_of_people_in_group_is_three : 
  let total_amount_paid := total_bill + (tip_percentage * total_bill) in
  let number_of_people := total_amount_paid / share_per_person in
  number_of_people ≈ 3 :=
by
  have total_amount_paid := total_bill + (tip_percentage * total_bill)
  have number_of_people := total_amount_paid / share_per_person
  have h : round number_of_people = 3 := sorry
  exact h

end number_of_people_in_group_is_three_l130_130274


namespace intersection_complement_correct_l130_130396

def I : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4, 5}
def B : Set ℕ := {1, 4}
def C_I (s : Set ℕ) := I \ s  -- set complement

theorem intersection_complement_correct: A ∩ C_I B = {3, 5} := by
  -- proof steps go here
  sorry

end intersection_complement_correct_l130_130396


namespace friends_who_dont_eat_meat_l130_130663

-- Definitions based on conditions
def number_of_friends : Nat := 10
def burgers_per_friend : Nat := 3
def buns_per_pack : Nat := 8
def packs_of_buns : Nat := 3
def friends_dont_eat_meat : Nat := 1
def friends_dont_eat_bread : Nat := 1

-- Total number of buns Alex plans to buy
def total_buns : Nat := buns_per_pack * packs_of_buns

-- Calculation of friends needing buns
def friends_needing_buns : Nat := number_of_friends - friends_dont_eat_meat - friends_dont_eat_bread

-- Total buns needed
def buns_needed : Nat := friends_needing_buns * burgers_per_friend

theorem friends_who_dont_eat_meat :
  buns_needed = total_buns -> friends_dont_eat_meat = 1 := by
  sorry

end friends_who_dont_eat_meat_l130_130663


namespace multiple_of_1984_exists_l130_130727

theorem multiple_of_1984_exists (a : Fin 97 → ℕ) (h_distinct: Function.Injective a) :
  ∃ i j k l : Fin 97, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ 
  1984 ∣ (a i - a j) * (a k - a l) :=
by
  sorry

end multiple_of_1984_exists_l130_130727


namespace maria_remaining_towels_l130_130580

-- Define the number of green towels Maria bought
def greenTowels : ℕ := 58

-- Define the number of white towels Maria bought
def whiteTowels : ℕ := 43

-- Define the total number of towels Maria bought
def totalTowels : ℕ := greenTowels + whiteTowels

-- Define the number of towels Maria gave to her mother
def towelsGiven : ℕ := 87

-- Define the resulting number of towels Maria has
def remainingTowels : ℕ := totalTowels - towelsGiven

-- Prove that the remaining number of towels is 14
theorem maria_remaining_towels : remainingTowels = 14 :=
by
  sorry

end maria_remaining_towels_l130_130580


namespace probability_maria_given_win_l130_130422

-- Define the problem conditions in Lean
namespace Race

-- Each car's lap time uniformly distributed between 150 and 155 seconds
noncomputable def LapTimes := {i | 150 ≤ i ∧ i ≤ 155 ∧ ∃ n : ℕ, i = 150 + n ∧ n ≤ 5}.to_finset

-- Maria's lap time
def maria_lap_time : ℕ := 152
def maria_wins (maria_lap_time : ℕ) := ∀ (other_lap_time : ℕ), other_lap_time ∈ LapTimes → other_lap_time > maria_lap_time

-- Probability definition
noncomputable def uniform_prob (val : ℕ) (S : finset ℕ) : ℝ :=
  if val ∈ S then 1 / S.card else 0

#check (LapTimes.card : ℤ) -- ensure LapTimes is of size 6

lemma probability_maria_lap_time_152 : uniform_prob 152 LapTimes = 1 / 6 := by
  sorry

lemma probability_maria_wins : ∀ (t : ℕ), t ∈ LapTimes → t > 152 → (LapTimes) = 27 / 216 := by
  sorry

lemma probability_maria_wins_given_lap_time_152 : ∀ (t : ℕ), maria_wins 152 → (LapTimes) = 1 / 8 := by
  sorry

lemma probability_maria_wins_overall : (LapTimes) = 1 / 4 := by
  sorry

theorem probability_maria_given_win : 
  uniform_prob 152 LapTimes * probability_maria_wins_given_lap_time_152 / probability_maria_wins_overall = 1 / 3  
:= by lem probability_maria_lap_time_152; lem probability_maria_wins; sorry

end Race

end probability_maria_given_win_l130_130422


namespace division_of_repeating_decimal_l130_130134

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l130_130134


namespace divide_by_repeating_decimal_l130_130178

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l130_130178


namespace julia_download_songs_l130_130443

-- Basic definitions based on conditions
def internet_speed_MBps : ℕ := 20
def song_size_MB : ℕ := 5
def half_hour_seconds : ℕ := 30 * 60

-- Statement of the proof problem
theorem julia_download_songs : 
  (internet_speed_MBps * half_hour_seconds) / song_size_MB = 7200 :=
by
  sorry

end julia_download_songs_l130_130443


namespace find_slope_of_line_l130_130375

open Set

def ellipse (x y : ℝ) : Prop := (x^2) / 20 + (y^2) / 16 = 1

def is_midpoint (P A B : ℝ × ℝ) : Prop := P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

def slope (A B : ℝ × ℝ) : ℝ := (A.2 - B.2) / (A.1 - B.1)

theorem find_slope_of_line (P A B : ℝ × ℝ) (hP : P = (3, -2))
  (hPA : ellipse A.1 A.2)
  (hPB : ellipse B.1 B.2)
  (hM : is_midpoint P A B) :
  slope A B = 6 / 5 :=
by sorry

end find_slope_of_line_l130_130375


namespace polygon_representation_l130_130935

-- Define the discrete random variable X and its probabilities
def X : List ℤ := [1, 3, 6, 8]
def p : List ℝ := [0.2, 0.1, 0.4, 0.3]

-- The theorem statement to prove that the constructed polygon represents the distribution of X
theorem polygon_representation :
  ∃ points : List (ℤ × ℝ), points = [(1, 0.2), (3, 0.1), (6, 0.4), (8, 0.3)] ∧ 
  is_polygon_of_distribution X p points :=
sorry

end polygon_representation_l130_130935


namespace eight_div_repeat_three_l130_130125

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l130_130125


namespace find_lambda_l130_130399

def vector := ℝ × ℝ

def dot_product (v₁ v₂ : vector) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

def vector_a : vector := (4, 3)
def vector_b : vector := (-1, 2)

def vector_m (λ : ℝ) : vector :=
  (vector_a.1 - λ * vector_b.1, vector_a.2 - λ * vector_b.2)

def vector_n : vector :=
  (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)

theorem find_lambda (λ : ℝ) (h : dot_product (vector_m λ) vector_n = 0) : λ = 27 / 7 :=
by sorry

end find_lambda_l130_130399


namespace max_expr_value_l130_130027

noncomputable def expr (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_expr_value : 
  ∃ (a b c d : ℝ),
    a ∈ Set.Icc (-5 : ℝ) 5 ∧
    b ∈ Set.Icc (-5 : ℝ) 5 ∧
    c ∈ Set.Icc (-5 : ℝ) 5 ∧
    d ∈ Set.Icc (-5 : ℝ) 5 ∧
    expr a b c d = 110 :=
by
  -- Proof omitted
  sorry

end max_expr_value_l130_130027


namespace jessica_balloons_l130_130441

-- Given conditions
def joan_balloons : Nat := 9
def sally_balloons : Nat := 5
def total_balloons : Nat := 16

-- The theorem to prove the number of balloons Jessica has
theorem jessica_balloons : (total_balloons - (joan_balloons + sally_balloons) = 2) :=
by
  -- Proof goes here
  sorry

end jessica_balloons_l130_130441


namespace equilateral_triangle_dot_product_l130_130369

/-- If ABC is an equilateral triangle with a side length of 6,
    and BC = 3 BE, AD = DC, then BD · AE = -18. -/
theorem equilateral_triangle_dot_product :
  ∀ {A B C D E : ℝ^2},
    ∥A - B∥ = 6 →
    ∥B - C∥ = 6 →
    ∥C - A∥ = 6 →
    B - C = 3 • (B - E) →
    A - D = D - C →
    (B - D) • (A - E) = -18 := by
  sorry

end equilateral_triangle_dot_product_l130_130369


namespace tony_money_left_l130_130059

theorem tony_money_left 
  (initial_money : ℕ)
  (ticket_cost : ℕ)
  (hotdog_cost : ℕ)
  (drink_cost : ℕ)
  (cap_cost : ℕ)
  (final_money : ℕ)
  (initial_hyp : initial_money = 50)
  (ticket_hyp : ticket_cost = 16)
  (hotdog_hyp : hotdog_cost = 5)
  (drink_hyp : drink_cost = 4)
  (cap_hyp : cap_cost = 12)
  (final_hyp : final_money = 13) :
  final_money = initial_money - ticket_cost - hotdog_cost - drink_cost - cap_cost :=
by 
  rw [initial_hyp, ticket_hyp, hotdog_hyp, drink_hyp, cap_hyp]
  (* The full proof would go here, but we'll put a placeholder for now. *)
  sorry

end tony_money_left_l130_130059


namespace roots_of_transformed_quadratic_l130_130771

variables {a b c r s : ℝ}

noncomputable def quadratic_with_roots (a b c : ℝ) := (y : ℝ) → y^2 - b * y + 4 * a * c

theorem roots_of_transformed_quadratic :
  (r s : ℝ) (hr : r + s = -b / a) (hprod : r * s = c / a) :
  has_roots (quadratic_with_roots a b c) (2 * a * r + b) (2 * a * s + b) :=
sorry

end roots_of_transformed_quadratic_l130_130771


namespace tree_height_increase_l130_130227

theorem tree_height_increase
  (initial_height : ℝ)
  (height_increase : ℝ)
  (h6 : ℝ) :
  initial_height = 4 →
  (0 ≤ height_increase) →
  height_increase * 6 + initial_height = (height_increase * 4 + initial_height) + 1 / 7 * (height_increase * 4 + initial_height) →
  height_increase = 2 / 5 :=
by
  intro h_initial h_nonneg h_eq
  sorry

end tree_height_increase_l130_130227


namespace longest_segment_in_cylinder_l130_130618

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 12) : 
  (∃ d : ℝ, d = 2 * r) ∧ (∃ hyp : ℝ, hyp = Real.sqrt(h^2 + (2 * r)^2) ∧ hyp = Real.sqrt(244)) :=
by
  sorry

end longest_segment_in_cylinder_l130_130618


namespace part_a_part_b_part_c_l130_130582

-- Part (a)
theorem part_a : (7 * (2 / 3) + 16 * (5 / 12)) = (34 / 3) :=
by
  sorry

-- Part (b)
theorem part_b : (5 - (2 / (5 / 3))) = (19 / 5) :=
by
  sorry

-- Part (c)
theorem part_c : (1 + (2 / (1 + (3 / (1 + 4))))) = (9 / 4) :=
by
  sorry

end part_a_part_b_part_c_l130_130582


namespace units_digit_fraction_l130_130570

open Nat

theorem units_digit_fraction : 
  (15 * 16 * 17 * 18 * 19 * 20) % 500 % 10 = 2 := by
  sorry

end units_digit_fraction_l130_130570


namespace rattlesnakes_count_l130_130045

-- Definitions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def rattlesnakes : ℕ := total_snakes - (boa_constrictors + pythons)

-- Theorem to prove
theorem rattlesnakes_count : rattlesnakes = 40 := by
  -- provide proof here
  sorry

end rattlesnakes_count_l130_130045


namespace simplify_tan_cot_l130_130916

theorem simplify_tan_cot (x : ℝ) (h1 : tan x ≠ 0) (h2 : 1 + 1 / tan x ≠ 0) :
    (tan x / (1 + (1 / tan x)) + (1 + 1 / tan x) / tan x) = (sec x) ^ 2 + (csc x) ^ 2 :=
by 
  sorry

end simplify_tan_cot_l130_130916


namespace min_value_of_expression_l130_130754

open Real

theorem min_value_of_expression (x y z : ℝ) (h₁ : x + y + z = 1) (h₂ : x > 0) (h₃ : y > 0) (h₄ : z > 0) :
  (∃ a, (∀ x y z, a ≤ (1 / (x + y) + (x + y) / z)) ∧ a = 3) :=
by
  sorry

end min_value_of_expression_l130_130754


namespace vertex_of_parabola_l130_130014

theorem vertex_of_parabola : 
  ∀ x, (3 * (x - 1)^2 + 2) = ((x - 1)^2 * 3 + 2) := 
by {
  -- The proof steps would go here
  sorry -- Placeholder to signify the proof steps are omitted
}

end vertex_of_parabola_l130_130014


namespace eight_div_repeating_three_l130_130081

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l130_130081


namespace min_value_of_expression_l130_130842

theorem min_value_of_expression {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 3) : 
  ∃ (m : ℝ), ∀ x y, (x > 0) → (y > 0) → (2 * x + y = 3) → (2 / x + 1 / y ≥ m) ∧ m = 3 :=
begin
  -- Proof skipped.
  sorry
end

end min_value_of_expression_l130_130842


namespace three_people_on_staircase_l130_130553

theorem three_people_on_staircase (A B C : Type) (steps : Finset ℕ) (h1 : steps.card = 7) 
  (h2 : ∀ step ∈ steps, step ≤ 2) : 
  ∃ (total_ways : ℕ), total_ways = 336 :=
by {
  sorry
}

end three_people_on_staircase_l130_130553


namespace solve_inequality_l130_130514

noncomputable def polynomial (x : ℝ) := -3 * x^3 + 5 * x^2 - 2 * x + 1

theorem solve_inequality (x : ℝ) : 
  (polynomial x > 0) ↔ (x ∈ Set.Ioo (-1 : ℝ) (1 / 3) ∪ Set.Ioo 1 (+∞)) :=
sorry

end solve_inequality_l130_130514


namespace remaining_erasers_is_nine_l130_130978

def total_erasers : ℕ := 28
def yeonju_received (n : ℕ) : ℕ := (1/4 : ℚ) * n
def minji_received (n : ℕ) : ℕ := (3/7 : ℚ) * n
def total_received (n : ℕ) : ℕ := yeonju_received n + minji_received n
def remaining_erasers (n : ℕ) : ℕ := n - total_received n

theorem remaining_erasers_is_nine : remaining_erasers total_erasers = 9 := by
  sorry

end remaining_erasers_is_nine_l130_130978


namespace sum_prime_factors_396_l130_130566

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factors (n : ℕ) : Finset ℕ := 
  (Finset.filter is_prime (Finset.range (n + 1))).filter (λ p, p ∣ n)

def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  (prime_factors n).sum

theorem sum_prime_factors_396 : sum_of_distinct_prime_factors 396 = 16 :=
  sorry

end sum_prime_factors_396_l130_130566


namespace not_sit_next_probability_l130_130261

theorem not_sit_next_probability (n : ℕ) (h : n = 9) :
  let total_ways := Nat.choose 9 2,
  let adjacent_pairs := 8,
  let probability_adjacent := (adjacent_pairs : ℚ) / total_ways,
  let probability_not_adjacent := 1 - probability_adjacent
  in probability_not_adjacent = 7 / 9 := 
by
  sorry

end not_sit_next_probability_l130_130261


namespace range_of_m_l130_130895

-- Definitions of p and q
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, m * x₀^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 > 0

-- Prove the main statement
theorem range_of_m (m : ℝ) : (p m ∨ q m) → m ∈ Iio 2 :=
by
  sorry

end range_of_m_l130_130895


namespace division_of_repeating_decimal_l130_130132

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l130_130132


namespace problem_statement_l130_130542

variable {A B C : ℝ}
variable {A₁ B₁ C₁ : ℝ}
variable (AA₁ BB₁ CC₁ : ℝ)

namespace triangle_geom

-- The triangle ABC is inscribed in a unit circle
def unit_circle (A B C : ℝ) : Prop := 
  is_unit_circle A B C

-- Internal angle bisectors intersect circle again at points A₁, B₁, C₁
def angle_bisectors_intersect (A B C A₁ B₁ C₁ : ℝ) : Prop := 
  internal_bisectors_intersect_again A B C A₁ B₁ C₁

-- Main statement to prove
theorem problem_statement
  (h1 : unit_circle A B C)
  (h2 : angle_bisectors_intersect A B C A₁ B₁ C₁)
  (hAA₁ : AA₁ = 2 * real.cos ((B - C) / 2))
  (hBB₁ : BB₁ = 2 * real.cos ((C - A) / 2))
  (hCC₁ : CC₁ = 2 * real.cos ((A - B) / 2)) :
  (AA₁ * real.cos (A / 2) + BB₁ * real.cos (B / 2) + CC₁ * real.cos (C / 2)) / (real.sin A + real.sin B + real.sin C) = 2 := 
  sorry

end triangle_geom

end problem_statement_l130_130542


namespace cone_height_l130_130847

theorem cone_height (a : ℝ) (h : ℝ) :
    (h^2 = a^2 - (a/2)^2) → (h = (√3 / 2) * a) :=
by
  sorry

end cone_height_l130_130847


namespace rationalization_correct_l130_130500

noncomputable def rationalize_denominator (a b : ℝ) : ℝ :=
  a / (b + 1)

theorem rationalization_correct :
  rationalize_denominator 1 (sqrt 3 - 1) = (sqrt 3 + 1) / 2 :=
by
  sorry

end rationalization_correct_l130_130500


namespace eight_div_repeat_three_l130_130124

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l130_130124


namespace Zain_coins_total_l130_130235

theorem Zain_coins_total :
  ∀ (quarters dimes nickels : ℕ),
  quarters = 6 →
  dimes = 7 →
  nickels = 5 →
  Zain_coins = quarters + 10 + (dimes + 10) + (nickels + 10) →
  Zain_coins = 48 :=
by intros quarters dimes nickels hq hd hn Zain_coins
   sorry

end Zain_coins_total_l130_130235


namespace determinant_problem_l130_130890

/-- Given distinct real numbers a, b, and c such that the determinant of the following matrix is 0:
    | 2  5  12  |
    | 4  a  b  |
    | 4  c  a  |,
    prove that a + b + c = 68.4. -/
theorem determinant_problem
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_det : det ![
      ![2, 5, 12],
      ![4, a, b],
      ![4, c, a]
    ] = 0) :
  a + b + c = 68.4 :=
begin
  sorry
end

end determinant_problem_l130_130890


namespace product_of_three_consecutive_integers_l130_130038

theorem product_of_three_consecutive_integers (x : ℕ) (h1 : x * (x + 1) = 740)
    (x1 : ℕ := x - 1) (x2 : ℕ := x) (x3 : ℕ := x + 1) :
    x1 * x2 * x3 = 17550 :=
by
  sorry

end product_of_three_consecutive_integers_l130_130038


namespace integer_solutions_l130_130714

theorem integer_solutions :
  { (x, y) : ℤ × ℤ |
       y^2 + y = x^4 + x^3 + x^2 + x } =
  { (-1, -1), (-1, 0), (0, -1), (0, 0), (2, 5), (2, -6) } :=
by
  sorry

end integer_solutions_l130_130714


namespace largest_five_digit_number_with_product_l130_130562

theorem largest_five_digit_number_with_product :
  ∃ (x : ℕ), (x = 98752) ∧ (∀ (d : List ℕ), (x.digits 10 = d) → (d.prod = 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) ∧ (x < 100000) ∧ (x ≥ 10000) :=
by
  sorry

end largest_five_digit_number_with_product_l130_130562


namespace rationalization_correct_l130_130499

noncomputable def rationalize_denominator (a b : ℝ) : ℝ :=
  a / (b + 1)

theorem rationalization_correct :
  rationalize_denominator 1 (sqrt 3 - 1) = (sqrt 3 + 1) / 2 :=
by
  sorry

end rationalization_correct_l130_130499


namespace find_a_if_even_function_l130_130836

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem find_a_if_even_function :
  (∃ a : ℝ, is_even_function (f a)) → ∃ a : ℝ, a = 2 :=
by
  intros h
  sorry

end find_a_if_even_function_l130_130836


namespace meeting_time_l130_130991

theorem meeting_time (x : ℝ) :
  (1/6) * x + (1/4) * (x - 1) = 1 :=
sorry

end meeting_time_l130_130991


namespace total_distance_l130_130053

noncomputable def total_distance_covered 
  (radius1 radius2 radius3 : ℝ) 
  (rev1 rev2 rev3 : ℕ) : ℝ :=
  let π := Real.pi
  let circumference r := 2 * π * r
  let distance r rev := circumference r * rev
  distance radius1 rev1 + distance radius2 rev2 + distance radius3 rev3

theorem total_distance
  (h1 : radius1 = 20.4) 
  (h2 : radius2 = 15.3) 
  (h3 : radius3 = 25.6) 
  (h4 : rev1 = 400) 
  (h5 : rev2 = 320) 
  (h6 : rev3 = 500) :
  total_distance_covered 20.4 15.3 25.6 400 320 500 = 162436.6848 := 
sorry

end total_distance_l130_130053


namespace min_distance_to_line_l130_130564

theorem min_distance_to_line : 
  let A := 5
  let B := -3
  let C := 4
  let d (x₀ y₀ : ℤ) := (abs (A * x₀ + B * y₀ + C) : ℝ) / (Real.sqrt (A ^ 2 + B ^ 2))
  ∃ (x₀ y₀ : ℤ), d x₀ y₀ = Real.sqrt 34 / 85 := 
by 
  sorry

end min_distance_to_line_l130_130564


namespace min_value_of_f_value_of_a_l130_130788

-- Definition of the function f
def f (x : ℝ) : ℝ := abs (x + 2) + 2 * abs (x - 1)

-- Problem: Prove that the minimum value of f(x) is 3
theorem min_value_of_f : ∃ x : ℝ, f x = 3 := sorry

-- Additional definitions for the second part of the problem
def g (x a : ℝ) : ℝ := f x + x - a

-- Problem: Given that the solution set of g(x,a) < 0 is (m, n) and n - m = 6, prove that a = 8
theorem value_of_a (a : ℝ) (m n : ℝ) (h : ∀ x : ℝ, g x a < 0 ↔ m < x ∧ x < n) (h_interval : n - m = 6) : a = 8 := sorry

end min_value_of_f_value_of_a_l130_130788


namespace count_integers_in_interval_l130_130808

theorem count_integers_in_interval : 
  ∃ (n : ℕ), (∀ (x : ℤ), (-2 ≤ x ∧ x ≤ 8 → ∃ (k : ℕ), k < n ∧ x = -2 + k)) ∧ n = 11 := 
by
  sorry

end count_integers_in_interval_l130_130808


namespace eight_div_repeating_three_l130_130086

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l130_130086


namespace evaluate_i11_plus_i111_l130_130325

def i_pow : ℤ → ℂ
| n := match n % 4 with
  | 0 := 1
  | 1 := complex.I
  | 2 := -1
  | 3 := -complex.I
  | _ := 0  -- this case will never be hit, since n % 4 is between 0 and 3.

theorem evaluate_i11_plus_i111 : 
  i_pow 11 + i_pow 111 = -2 * complex.I :=
begin
  sorry
end

end evaluate_i11_plus_i111_l130_130325


namespace ravi_prakash_work_together_l130_130594

theorem ravi_prakash_work_together (ravi_time prakash_time : ℕ) 
  (hravi : ravi_time = 24) (hprakash : prakash_time = 40) : 
  1 / ((1 : ℝ) / ravi_time + (1 : ℝ) / prakash_time) = 15 := 
by
  rw [hravi, hprakash]
  norm_num
  sorry -- here the proof should follow

-- This statement can be converted to the following:
-- theorem ravi_prakash_work_together : 
--   1 / ((1 : ℝ) / 24 + (1 : ℝ) / 40) = 15 := by 
--   norm_num
--   sorry

end ravi_prakash_work_together_l130_130594


namespace cone_central_angle_l130_130613

/-- Proof Problem Statement: Given the radius of the base circle of a cone (r) and the slant height of the cone (l),
    prove that the central angle (θ) of the unfolded diagram of the lateral surface of this cone is 120 degrees. -/
theorem cone_central_angle (r l : ℝ) (h_r : r = 10) (h_l : l = 30) : (360 * r) / l = 120 :=
by
  -- The proof steps are omitted
  sorry

end cone_central_angle_l130_130613


namespace eight_div_repeating_three_l130_130084

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l130_130084


namespace trajectory_is_line_segment_l130_130370

-- Defining distance function
def distance (P Q : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Variables and points
variable (P : ℝ × ℝ)
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (5, 0)

-- Condition: |PA| + |PB| = 5
def condition1 : Prop := distance P A + distance P B = 5

-- The proof statement
theorem trajectory_is_line_segment (h : condition1 P) : P.2 = 0 ∧ P.1 ∈ set.Icc 0 5 :=
sorry

end trajectory_is_line_segment_l130_130370


namespace eight_div_repeating_three_l130_130183

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l130_130183


namespace Eight_div_by_repeating_decimal_0_3_l130_130168

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l130_130168


namespace count_integers_satisfying_inequality_l130_130805

theorem count_integers_satisfying_inequality :
  {n : ℤ | (n + 2) * (n - 8) ≤ 0}.to_finset.card = 11 :=
sorry

end count_integers_satisfying_inequality_l130_130805


namespace divide_by_repeating_decimal_l130_130098

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l130_130098


namespace eight_div_repeat_three_l130_130126

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l130_130126


namespace simplify_trigonometric_expression_l130_130510

theorem simplify_trigonometric_expression (y z : ℝ) :
  sin y ^ 2 + sin (y + z) ^ 2 - 2 * sin y * sin z * sin (y + z) = sin y ^ 2 + sin z ^ 2 * cos y ^ 2 :=
by
  sorry

end simplify_trigonometric_expression_l130_130510


namespace equal_area_sums_of_strip_parts_l130_130380

theorem equal_area_sums_of_strip_parts
  (ABCD : Type)
  (n : ℕ)
  (lengths : Fin n → ℝ)
  (sum_odd_lengths_eq_sum_even_lengths : (∑ i in (Finset.filter (λ i, i % 2 = 1) Finset.univ), lengths i) = (∑ i in (Finset.filter (λ i, i % 2 = 0) Finset.univ), lengths i))
  (S1 S2 : ℝ)
  (areas : Fin n → ℝ)
  (left_parts_odd : S1 = ∑ i in (Finset.filter (λ i, i % 2 = 1) Finset.univ), areas i)
  (right_parts_even : S2 = ∑ i in (Finset.filter (λ i, i % 2 = 0) Finset.univ), areas i) :
  S1 = S2 :=
sorry

end equal_area_sums_of_strip_parts_l130_130380


namespace find_a_l130_130828

noncomputable def f (a x : ℝ) := (x - 1)^2 + a * x + Real.cos x

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) → 
  a = 2 :=
by
  sorry

end find_a_l130_130828


namespace sin_cos_identity_l130_130379

theorem sin_cos_identity (α : ℝ) (hα_cos : Real.cos α = 3/5) (hα_sin : Real.sin α = 4/5) : Real.sin α + 2 * Real.cos α = 2 :=
by
  -- Proof omitted
  sorry

end sin_cos_identity_l130_130379


namespace total_combined_rainfall_l130_130874

theorem total_combined_rainfall :
  let monday_hours := 5
  let monday_rate := 1
  let tuesday_hours := 3
  let tuesday_rate := 1.5
  let wednesday_hours := 4
  let wednesday_rate := 2 * monday_rate
  let thursday_hours := 6
  let thursday_rate := tuesday_rate / 2
  let friday_hours := 2
  let friday_rate := 1.5 * wednesday_rate
  let monday_rain := monday_hours * monday_rate
  let tuesday_rain := tuesday_hours * tuesday_rate
  let wednesday_rain := wednesday_hours * wednesday_rate
  let thursday_rain := thursday_hours * thursday_rate
  let friday_rain := friday_hours * friday_rate
  monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain = 28 := by
  sorry

end total_combined_rainfall_l130_130874


namespace lowest_dropped_score_l130_130877

theorem lowest_dropped_score (A B C D : ℕ)
  (h1 : (A + B + C + D) / 4 = 50)
  (h2 : (A + B + C) / 3 = 55) :
  D = 35 :=
by
  sorry

end lowest_dropped_score_l130_130877


namespace pyramid_volume_correct_l130_130056

noncomputable def volume_of_pyramid : ℝ :=
let l := sqrt 8 in
let α := π / 6 in
let β := π / 4 in
let AO1 := l * cos(α) in
let AO2 := l * cos(α) in
let AO3 := l * cos(β) in
let O1O2 := l * sin(α + β) in
let O1O3 := O1O2 in
let O2O3 := O1O2 in
let AH := AO3 in
let base_area := (O1O2 ^ 2) * (sqrt 3 / 4) in
(sqrt ((48 * (sqrt 3 + 1)) / 16 : ℝ)).sqrt

theorem pyramid_volume_correct :
  let V := volume_of_pyramid in
  V = sqrt (sqrt 3 + 1) :=
by {
  sorry
}

end pyramid_volume_correct_l130_130056


namespace total_runs_by_opponents_l130_130609

theorem total_runs_by_opponents :
  let team_scores := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let lost_scores := [1, 3, 5, 7, 9, 11]
  let won_scores := [2, 4, 6, 8, 10, 12]
  let opponents_in_lost_games := [3, 5, 7, 9, 11, 13]
  let opponents_in_won_games := [1, 1, 2, 3, 3, 4]
  (list.sum opponents_in_lost_games) + (list.sum opponents_in_won_games) = 62 :=
by
  let team_scores := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let lost_scores := [1, 3, 5, 7, 9, 11]
  let won_scores := [2, 4, 6, 8, 10, 12]
  let opponents_in_lost_games := [3, 5, 7, 9, 11, 13]
  let opponents_in_won_games := [1, 1, 2, 3, 3, 4]
  have h_lost := list.sum opponents_in_lost_games,
  have h_won := list.sum opponents_in_won_games,
  have h_total := h_lost + h_won,
  sorry

end total_runs_by_opponents_l130_130609


namespace exists_n0_l130_130250

noncomputable def x : ℕ → ℤ
| 0     := 2
| 1     := 6
| (n+2) := 2 * x (n+1) + x n

noncomputable def y : ℕ → ℤ
| 0     := 3
| 1     := 9
| (n+2) := y (n+1) + 2 * y n

theorem exists_n0 : ∃ n0 : ℕ, ∀ n : ℕ, n > n0 → x n > y n :=
sorry

end exists_n0_l130_130250


namespace amount_after_two_years_l130_130329

def present_value : ℝ := 62000
def rate_of_increase : ℝ := 0.125
def time_period : ℕ := 2

theorem amount_after_two_years:
  let amount_after_n_years (pv : ℝ) (r : ℝ) (n : ℕ) := pv * (1 + r)^n
  amount_after_n_years present_value rate_of_increase time_period = 78468.75 := 
  by 
    -- This is where your proof would go
    sorry

end amount_after_two_years_l130_130329


namespace selling_price_correct_l130_130485

-- Define the conditions
def cost_per_cupcake : ℝ := 0.75
def total_cupcakes_burnt : ℕ := 24
def total_eaten_first : ℕ := 5
def total_eaten_later : ℕ := 4
def net_profit : ℝ := 24
def total_cupcakes_made : ℕ := 72
def total_cost : ℝ := total_cupcakes_made * cost_per_cupcake
def total_eaten : ℕ := total_eaten_first + total_eaten_later
def total_sold : ℕ := total_cupcakes_made - total_eaten
def revenue (P : ℝ) : ℝ := total_sold * P

-- Prove the correctness of the selling price P
theorem selling_price_correct : 
  ∃ P : ℝ, revenue P - total_cost = net_profit ∧ (P = 1.24) :=
by
  sorry

end selling_price_correct_l130_130485


namespace eight_div_repeat_three_l130_130129

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l130_130129


namespace even_function_has_a_equal_2_l130_130823

noncomputable def f (a x : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_has_a_equal_2 (a : ℝ) :
  (∀ x : ℝ, f a (-x) = f a x) → a = 2 :=
sorry

end even_function_has_a_equal_2_l130_130823


namespace solve_for_x_l130_130321

theorem solve_for_x (x : ℝ) : 
  (5 * (x^2) ^ 2 + 3 * x^2 + 2 = 4 * (4 * x^2 + x^2 + 1)) → 
  (x = real.sqrt((17 + real.sqrt 329) / 10) ∨ x = -real.sqrt((17 + real.sqrt 329) / 10)) :=
  sorry

end solve_for_x_l130_130321


namespace ratio_cube_sphere_surface_area_l130_130966

theorem ratio_cube_sphere_surface_area (R : ℝ) (h1 : R > 0) :
  let Scube := 24 * R^2
  let Ssphere := 4 * Real.pi * R^2
  (Scube / Ssphere) = (6 / Real.pi) :=
by
  sorry

end ratio_cube_sphere_surface_area_l130_130966


namespace shortest_tree_height_l130_130551

theorem shortest_tree_height :
  (tallest_tree_height = 150) →
  (middle_tree_height = (2 / 3) * tallest_tree_height) →
  (shortest_tree_height = (1 / 2) * middle_tree_height) →
  shortest_tree_height = 50 :=
by
  intros h1 h2 h3
  sorry

end shortest_tree_height_l130_130551


namespace number_of_integers_satisfying_inequality_l130_130817

theorem number_of_integers_satisfying_inequality :
  set.count {n : ℤ | (n + 2) * (n - 8) ≤ 0} = 11 :=
sorry

end number_of_integers_satisfying_inequality_l130_130817


namespace proof_problem_l130_130536

def sequence : Nat → Rat
| 0 => 2000000
| (n + 1) => sequence n / 2

theorem proof_problem :
  (∀ n, ((sequence n).den = 1) → n < 7) ∧ 
  (sequence 7 = 15625) ∧ 
  (sequence 7 - 3 = 15622) :=
by
  sorry

end proof_problem_l130_130536


namespace coupon_discounts_l130_130288

theorem coupon_discounts (P : ℝ) (hP : P > 120) :
  let A_savings := 0.20 * P
  let B_savings := 40
  let C_savings := 0.30 * (P - 120)
  let x := 200
  let y := 360
  A_savings ≥ B_savings ∧ A_savings ≥ C_savings →
  y - x = 160 := by
  -- Define savings for each coupon
  let A_savings := 0.20 * P
  let B_savings := 40
  let C_savings := 0.30 * (P - 120)
  -- Define price ranges
  let x := 200
  let y := 360
  -- Given conditions
  unfold A_savings
  unfold B_savings
  unfold C_savings
  have A_vs_B : A_savings ≥ B_savings :=
    by sorry
  have A_vs_C : A_savings ≥ C_savings :=
    by sorry
  -- Simplify result
  have result : y - x = 160 := by
    dsimp [y, x]
    exact rfl
  exact result

end coupon_discounts_l130_130288


namespace parabola_distance_x_coord_l130_130725

theorem parabola_distance_x_coord
  (M : ℝ × ℝ) 
  (hM : M.2^2 = 4 * M.1)
  (hMF : (M.1 - 1)^2 + M.2^2 = 4^2)
  : M.1 = 3 :=
sorry

end parabola_distance_x_coord_l130_130725


namespace lollipop_distribution_l130_130728

theorem lollipop_distribution 
  (P1 P2 P_total L x : ℕ) 
  (h1 : P1 = 45) 
  (h2 : P2 = 15) 
  (h3 : L = 12) 
  (h4 : P_total = P1 + P2) 
  (h5 : P_total = 60) : 
  x = 5 := 
by 
  sorry

end lollipop_distribution_l130_130728


namespace problem_part1_29_13_problem_part2_mn_problem_part3_k_36_problem_part4_min_val_l130_130947

def is_perfect_number (n : ℕ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

theorem problem_part1_29_13 : is_perfect_number 29 ∧ is_perfect_number 13 := by
  sorry

theorem problem_part2_mn : 
  ∃ m n : ℤ, (∀ a : ℤ, a^2 - 4 * a + 8 = (a - m)^2 + n^2) ∧ (m * n = 4 ∨ m * n = -4) := by
  sorry

theorem problem_part3_k_36 (a b : ℤ) : 
  ∃ k : ℤ, (∀ k : ℤ, a^2 + 4*a*b + 5*b^2 - 12*b + k = (a + 2*b)^2 + (b-6)^2) ∧ k = 36 := by
  sorry

theorem problem_part4_min_val (a b : ℝ) : 
  (∀ (a b : ℝ), -a^2 + 5*a + b - 7 = 0 → ∃ a' b', (a + b = (a'-2)^2 + 3) ∧ a' + b' = 3) := by
  sorry

end problem_part1_29_13_problem_part2_mn_problem_part3_k_36_problem_part4_min_val_l130_130947


namespace num_five_digit_integers_l130_130801

theorem num_five_digit_integers
  (total_digits : ℕ := 8)
  (repeat_3 : ℕ := 2)
  (repeat_6 : ℕ := 3)
  (repeat_8 : ℕ := 2)
  (arrangements : ℕ := Nat.factorial total_digits / (Nat.factorial repeat_3 * Nat.factorial repeat_6 * Nat.factorial repeat_8)) :
  arrangements = 1680 := by
  sorry

end num_five_digit_integers_l130_130801


namespace prime_mod_30_not_composite_l130_130303

theorem prime_mod_30_not_composite (p : ℕ) (h_prime : Prime p) (h_gt_30 : p > 30) : 
  ¬ ∃ (x : ℕ), (x > 1 ∧ ∃ (a b : ℕ), x = a * b ∧ a > 1 ∧ b > 1) ∧ (0 < x ∧ x < 30 ∧ ∃ (k : ℕ), p = 30 * k + x) :=
by
  sorry

end prime_mod_30_not_composite_l130_130303


namespace midpoint_PQ_is_incenter_l130_130860

-- Define the isosceles triangle and its properties
variables {α : Type*} [euclidean_space α] {A B C O P Q: α}
variables (h_iso : dist A B = dist A C)
variables (circumcircle_tangent: ∀ (P Q : α), circle_internally_tangent (A B C) (P Q))

-- Define the midpoint and incenter
def midpoint (P Q : α) : α := (P + Q) / 2
def incenter_triangle (A B C : α) : α := sorry -- Define incenter of the triangle

-- Prove that the midpoint of PQ is the incenter
theorem midpoint_PQ_is_incenter :
  (midpoint P Q = incenter_triangle A B C) :=
begin
  sorry
end

end midpoint_PQ_is_incenter_l130_130860


namespace sum_of_possible_remainders_eq_98_l130_130934

/-
  The sum of the possible remainders when a positive integer m, which has digits as four consecutive integers in increasing order, is divided by 23 equals 98.
-/

theorem sum_of_possible_remainders_eq_98 :
  (∑ n in Finset.range 7, (1111 * n + 123) % 23) = 98 := 
by
  sorry

end sum_of_possible_remainders_eq_98_l130_130934


namespace sum_47_neg27_eq_20_l130_130565

def sum_of_numbers (a b : Int) : Int :=
  a + b

theorem sum_47_neg27_eq_20 : sum_of_numbers 47 (-27) = 20 := 
by 
  simp [sum_of_numbers]
  sorry

end sum_47_neg27_eq_20_l130_130565


namespace rationalize_sqrt_three_sub_one_l130_130491

theorem rationalize_sqrt_three_sub_one :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_sqrt_three_sub_one_l130_130491


namespace area_uncovered_by_semicircles_is_zero_l130_130659

noncomputable def isosceles_right_triangle_hypotenuse_area_uncovered_by_semicircles : ℝ :=
  let AB := 16 in
  let AC := 8 / Real.sqrt 2 in
  let BC := 8 / Real.sqrt 2 in
  let r_large := 8 in
  let r_small := 8 / Real.sqrt 2 in
  let area_triangle := (1 / 2) * AB * AC in
  let area_large_semi := (1 / 2) * Real.pi * r_large^2 in
  let area_small_semi := (1 / 2) * Real.pi * r_small^2 in
  area_triangle - (2 * area_small_semi - area_large_semi)

theorem area_uncovered_by_semicircles_is_zero :
  isosceles_right_triangle_hypotenuse_area_uncovered_by_semicircles = 0 :=
by
  sorry

end area_uncovered_by_semicircles_is_zero_l130_130659


namespace periodic_sequence_sum_l130_130344

def is_periodic (a : ℕ → ℕ) (T : ℕ) : Prop :=
  ∀ n : ℕ, a (n + T) = a n

variables (a : ℕ → ℕ) (T m q r : ℕ)
variables (S_m S_T S_r : ℕ)

def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).sum (λ i, a i)

theorem periodic_sequence_sum (hT : ∃ T : ℕ, is_periodic a T) 
  (hm : m = q * T + r)
  (hS_m : S_m = sum_first_n_terms a m)
  (hS_T : S_T = sum_first_n_terms a T)
  (hS_r : S_r = sum_first_n_terms a r) : 
  S_m = q * S_T + S_r :=
sorry

end periodic_sequence_sum_l130_130344


namespace flour_cups_l130_130641

theorem flour_cups (f : ℚ) (h : f = 4 + 3/4) : (1/3) * f = 1 + 7/12 := by
  sorry

end flour_cups_l130_130641


namespace systematic_sampling_unique_set_l130_130977

theorem systematic_sampling_unique_set (students : Finset ℕ) (h_students : students = Finset.range 21 \ {0}) 
  (selected : Finset ℕ) (h_selected : selected.card = 4) 
  (sampling_method : ∀ x ∈ selected, ∀ y ∈ selected, x < y → y = x + 5) 
  : selected = {3, 8, 13, 18} :=
sorry

end systematic_sampling_unique_set_l130_130977


namespace employee_hourly_pay_l130_130615

-- Definitions based on conditions
def initial_employees := 500
def daily_hours := 10
def weekly_days := 5
def monthly_weeks := 4
def additional_employees := 200
def total_payment := 1680000
def total_employees := initial_employees + additional_employees
def monthly_hours_per_employee := daily_hours * weekly_days * monthly_weeks
def total_monthly_hours := total_employees * monthly_hours_per_employee

-- Lean 4 statement proving the hourly pay per employee
theorem employee_hourly_pay : total_payment / total_monthly_hours = 12 := by sorry

end employee_hourly_pay_l130_130615


namespace provisions_last_days_after_reinforcement_l130_130273

-- Definitions based on the conditions
def initial_men := 2000
def initial_days := 40
def reinforcement_men := 2000
def days_passed := 20

-- Calculate the total provisions initially
def total_provisions := initial_men * initial_days

-- Calculate the remaining provisions after some days passed
def remaining_provisions := total_provisions - (initial_men * days_passed)

-- Total number of men after reinforcement
def total_men := initial_men + reinforcement_men

-- The Lean statement proving the duration the remaining provisions will last
theorem provisions_last_days_after_reinforcement :
  remaining_provisions / total_men = 10 := by
  sorry

end provisions_last_days_after_reinforcement_l130_130273


namespace complex_number_properties_l130_130361

theorem complex_number_properties (z : ℂ) (h : (complex.i - 1) * z = 2 * complex.i) : 
  abs z = real.sqrt 2 ∧ (polynomial.eval z (polynomial.C 1 * polynomial.X^2 - polynomial.C 2 * polynomial.X + polynomial.C 2) = 0) :=
by
  sorry

end complex_number_properties_l130_130361


namespace exists_infinite_subset_with_gcd_l130_130458

/-- A set of natural numbers where each number is a product of at most 1987 primes -/
def is_bounded_product_set (A : Set ℕ) (k : ℕ) : Prop :=
  ∀ a ∈ A, ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ a = S.prod id ∧ S.card ≤ k

/-- Prove the existence of an infinite subset and a common gcd for any pair of its elements -/
theorem exists_infinite_subset_with_gcd (A : Set ℕ) (k : ℕ) (hk : k = 1987)
  (hA : is_bounded_product_set A k) (h_inf : Set.Infinite A) :
  ∃ (B : Set ℕ) (b : ℕ), Set.Subset B A ∧ Set.Infinite B ∧ ∀ (x y : ℕ), x ∈ B → y ∈ B → x ≠ y → Nat.gcd x y = b := 
sorry

end exists_infinite_subset_with_gcd_l130_130458


namespace magnitude_difference_l130_130794

noncomputable def vector_a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
noncomputable def vector_b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))

theorem magnitude_difference (a b : ℝ × ℝ) 
  (ha : a = (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180)))
  (hb : b = (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))) :
  (Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2)) = Real.sqrt 3 :=
by
  sorry

end magnitude_difference_l130_130794


namespace expected_winnings_correct_l130_130591

noncomputable def first_team_odd := 1.28
noncomputable def second_team_odd := 5.23
noncomputable def third_team_odd := 3.25
noncomputable def fourth_team_odd := 2.05
noncomputable def bet_amount := 5.00

theorem expected_winnings_correct :
  let total_odds := first_team_odd * second_team_odd * third_team_odd * fourth_team_odd,
      total_payout := total_odds * bet_amount,
      expected_winnings := total_payout - bet_amount
  in expected_winnings = 211.4035 := by
  -- proof not required
  sorry

end expected_winnings_correct_l130_130591


namespace bob_is_47_5_l130_130678

def bob_age (a b : ℝ) := b = 3 * a - 20
def sum_of_ages (a b : ℝ) := b + a = 70

theorem bob_is_47_5 (a b : ℝ) (h1 : bob_age a b) (h2 : sum_of_ages a b) : b = 47.5 :=
by
  sorry

end bob_is_47_5_l130_130678


namespace eight_div_repeating_three_l130_130088

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l130_130088


namespace root_interval_01_l130_130943

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2^x + x - 2

-- Prove that there is a root in the interval (0,1)
theorem root_interval_01 : 
  f 0 < 0 → f 1 > 0 → (∃ c ∈ Ioo 0 1, f c = 0) :=
begin
  intros h0 h1,
  -- Proof is to be provided
  sorry
end

end root_interval_01_l130_130943


namespace min_n_to_prevent_T_l130_130862

def grid := fin 8 × fin 8

def T_shaped (squares : set grid) : Prop :=
  ∃ (c : grid), squares = {c,
                           (fin.mk (c.1.val + 1) sorry, c.2),
                           (fin.mk (c.1.val - 1) sorry, c.2),
                           (c.1, fin.mk (c.2.val + 1) sorry),
                           (c.1, fin.mk (c.2.val - 1) sorry)}

def min_squares_to_prevent_T : ℕ := 32

theorem min_n_to_prevent_T :
  ∀ (squaresToRemove : set grid), 
  (∀ gridSet : set grid, T_shaped gridSet → gridSet ⊆ squaresToRemove) 
  ↔ ∃ (n : ℕ), n = min_squares_to_prevent_T :=
by
  sorry

end min_n_to_prevent_T_l130_130862


namespace division_of_repeating_decimal_l130_130067

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l130_130067


namespace number_of_machines_sold_l130_130284

-- Define the parameters and conditions given in the problem
def commission_of_first_150 (sale_price : ℕ) : ℕ := 150 * (sale_price * 3 / 100)
def commission_of_next_100 (sale_price : ℕ) : ℕ := 100 * (sale_price * 4 / 100)
def commission_of_after_250 (sale_price : ℕ) (x : ℕ) : ℕ := x * (sale_price * 5 / 100)

-- Define the total commission using these commissions
def total_commission (x : ℕ) : ℕ :=
  commission_of_first_150 10000 + 
  commission_of_next_100 9500 + 
  commission_of_after_250 9000 x

-- The main statement we want to prove
theorem number_of_machines_sold (x : ℕ) (total_commission : ℕ) : x = 398 ↔ total_commission = 150000 :=
by
  sorry

end number_of_machines_sold_l130_130284


namespace dress_design_combinations_l130_130624

theorem dress_design_combinations (colors patterns sleeve_types : ℕ)
  (h_colors : colors = 5)
  (h_patterns : patterns = 4)
  (h_sleeve_types : sleeve_types = 3) : 
  colors * patterns * sleeve_types = 60 :=
by
  rw [h_colors, h_patterns, h_sleeve_types]
  norm_num

end dress_design_combinations_l130_130624


namespace division_by_repeating_decimal_l130_130213

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l130_130213


namespace volunteers_same_team_probability_l130_130866

theorem volunteers_same_team_probability : 
  let teams := 3 in
  let total_outcomes := teams * teams in
  let favorable_outcomes := teams in
  let probability := favorable_outcomes.toRat / total_outcomes.toRat in
  probability = 1 / 3 := 
by
  sorry

end volunteers_same_team_probability_l130_130866


namespace sum_of_intersection_coordinates_eq_zero_l130_130309

theorem sum_of_intersection_coordinates_eq_zero :
  let parabola1 := fun x => (x - 2) ^ 2
  let parabola2 := fun y => (y + 2) ^ 2 - 1
  -- Define the sets of x and y coordinates of the intersection points
  ∃ (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ), 
  (∀ xi ∈ {x1, x2, x3, x4}, parabola1 xi = (parabola2 (parabola1 xi) + 2)^2 - 1) ∧
  (∀ yi ∈ {y1, y2, y3, y4}, parabola2 yi = (parabola1 (parabola2 yi) - 2)^2) ∧ 
  (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 0) := 
sorry

end sum_of_intersection_coordinates_eq_zero_l130_130309


namespace final_range_a_l130_130389

open Real

noncomputable def f (a x : ℝ) : ℝ := log x + x^2 - a * x

lemma increasing_function_range_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0) :
  a ≤ 2 * sqrt 2 :=
sorry

lemma condition_range_a (a : ℝ) (h1 : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0)
  (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x ≤ 1/2 * (3 * x^2 + 1 / x^2 - 6 * x)) :
  2 ≤ a :=
sorry

theorem final_range_a (a : ℝ)
  (h1 : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0)
  (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x ≤ 1/2 * (3 * x^2 + 1 / x^2 - 6 * x)) :
  2 ≤ a ∧ a ≤ 2 * sqrt 2 :=
sorry

end final_range_a_l130_130389


namespace count_integers_satisfying_inequality_l130_130804

theorem count_integers_satisfying_inequality :
  {n : ℤ | (n + 2) * (n - 8) ≤ 0}.to_finset.card = 11 :=
sorry

end count_integers_satisfying_inequality_l130_130804


namespace find_p_plus_q_l130_130449

noncomputable def center_ω1 : (ℝ × ℝ) := (-6, 10)
noncomputable def radius_ω1 : ℝ := 16
noncomputable def center_ω2 : (ℝ × ℝ) := (6, 10)
noncomputable def radius_ω2 : ℝ := 8

theorem find_p_plus_q :
  let a := sqrt (160 / 99) in
  let m := sqrt (160 / 99) in
  let p := 160,
  let q := 99 in
  m^2 = 160 / 99 ∧ ↑p + ↑q = 259 :=
by
  sorry

end find_p_plus_q_l130_130449


namespace arccos_cos_10_l130_130305

noncomputable def arccos_periodic (x : ℝ) : Prop :=
  ∀ k : ℤ, arccos (cos (x + k * 2 * Real.pi)) = arccos (cos x)

theorem arccos_cos_10 :
  arccos_cos_10 = 0.283185307 :=
by 
  have key : ∀ (x : ℝ), arccos (cos x) = arccos (cos (x - 4 * Real.pi)) :=
    by sorry  -- Placeholder for the periodicity argument
  rw key 10
  norm_num
  sorry

end arccos_cos_10_l130_130305


namespace find_a_if_f_is_even_l130_130830

def f (a : ℝ) (x : ℝ) : ℝ := (x-1)^2 + a*x + Real.sin(x + Real.pi / 2)

theorem find_a_if_f_is_even (a : ℝ) (h : ∀ x : ℝ, f(a, x) = f(a, -x)) : a = 2 := by
  sorry

end find_a_if_f_is_even_l130_130830


namespace composite_probability_l130_130037

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

theorem composite_probability :
  let total_numbers := 50 in
  let composite_count := total_numbers - 1 - 15 in -- 1 is neither prime nor composite; 15 primes under 50.
  (composite_count : ℝ) / total_numbers = 0.68 :=
by
  sorry

end composite_probability_l130_130037


namespace area_of_triangle_eq_420_l130_130413

variables {A B C : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]

def AB := 29
def BC := 29
def AC := 42

theorem area_of_triangle_eq_420 (h₁ : dist A B = AB) (h₂ : dist B C = BC) (h₃ : dist A C = AC) : 
∃ (Δ : Triangle A B C), area Δ = 420 :=
by {
  -- The proof is omitted.
  sorry
}

end area_of_triangle_eq_420_l130_130413


namespace division_of_decimal_l130_130111

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l130_130111


namespace area_of_quadrilateral_ABCD_l130_130476

-- The definitions are based on the conditions provided
variable (A B C D : Type)
variable [MetricSpace A]
variable (dist : A → A → ℝ) -- distance function 

def cyclist_speed_asphalt := 15   -- cyclist's speed on asphalt roads in km/h
def travel_time := 2             -- travel time on asphalt roads in hours
def distance_BX (X : A) : ℝ := cyclist_speed_asphalt * travel_time -- BX distance for X in {A, C, D}

-- Introduce the distances based on the provided conditions
def BD := 30     -- Distance BD is 30 km

-- Given that point B travels equidistantly to points A, C, and D
def is_dirt_road (X Y : A) : Prop := dist X Y = 30

-- Main statement to be proved in Lean
theorem area_of_quadrilateral_ABCD
  (h_AB : is_dirt_road B A)
  (h_BC : is_dirt_road B C)
  (h_CD : is_dirt_road C D)
  (h_DA : is_dirt_road D A)
  (h_BD : dist B D = BD)
  : quadrilateral_area A B C D = 450 := 
sorry  -- proof to be derived later

end area_of_quadrilateral_ABCD_l130_130476


namespace division_of_repeating_decimal_l130_130133

theorem division_of_repeating_decimal :
  (8 : ℝ) / (0.333333... : ℝ) = 24 :=
by
  -- It is known that 0.333333... = 1/3
  have h : (0.333333... : ℝ) = (1 / 3 : ℝ) :=
    by sorry
  -- Thus, 8 / (0.333333...) = 8 / (1 / 3) = 8 * 3
  calc
    (8 : ℝ) / (0.333333... : ℝ)
        = (8 : ℝ) / (1 / 3 : ℝ) : by rw h
    ... = (8 : ℝ) * (3 : ℝ) : by norm_num
    ... = 24 : by norm_num

end division_of_repeating_decimal_l130_130133


namespace rationalization_correct_l130_130496

noncomputable def rationalize_denominator (a b : ℝ) : ℝ :=
  a / (b + 1)

theorem rationalization_correct :
  rationalize_denominator 1 (sqrt 3 - 1) = (sqrt 3 + 1) / 2 :=
by
  sorry

end rationalization_correct_l130_130496


namespace conjugate_of_z_l130_130775

noncomputable def z : ℂ := 2 / (1 + complex.I)

theorem conjugate_of_z : complex.conj z = 1 + complex.I :=
by
  sorry

end conjugate_of_z_l130_130775


namespace eight_div_repeat_three_l130_130118

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l130_130118


namespace mirror_wall_area_ratio_l130_130287

theorem mirror_wall_area_ratio :
  let mirror_side := 18
  let wall_width := 32
  let wall_length := 20.25
  let mirror_area := mirror_side * mirror_side
  let wall_area := wall_width * wall_length
  mirror_area / wall_area = 1 / 2 :=
by
  -- Define the values
  let mirror_side := 18
  let wall_width := 32
  let wall_length := 20.25
  let mirror_area := mirror_side * mirror_side
  let wall_area := wall_width * wall_length
  -- Calculate and simplify the ratio
  have h1 : mirror_area = 324 := rfl
  have h2 : wall_area = 648 := rfl
  calc
    324 / 648 = 1 / 2 : by norm_num

end mirror_wall_area_ratio_l130_130287


namespace division_of_decimal_l130_130109

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l130_130109


namespace male_students_outnumber_female_students_l130_130296

-- Define the given conditions
def total_students : ℕ := 928
def male_students : ℕ := 713
def female_students : ℕ := total_students - male_students

-- The theorem to be proven
theorem male_students_outnumber_female_students :
  male_students - female_students = 498 :=
by
  sorry

end male_students_outnumber_female_students_l130_130296


namespace find_a_if_f_is_even_l130_130832

def f (a : ℝ) (x : ℝ) : ℝ := (x-1)^2 + a*x + Real.sin(x + Real.pi / 2)

theorem find_a_if_f_is_even (a : ℝ) (h : ∀ x : ℝ, f(a, x) = f(a, -x)) : a = 2 := by
  sorry

end find_a_if_f_is_even_l130_130832


namespace nina_expected_tomato_harvest_l130_130902

noncomputable def expected_tomato_harvest 
  (garden_length : ℝ) (garden_width : ℝ) 
  (plants_per_sq_ft : ℝ) (tomatoes_per_plant : ℝ) : ℝ :=
  garden_length * garden_width * plants_per_sq_ft * tomatoes_per_plant

theorem nina_expected_tomato_harvest : 
  expected_tomato_harvest 10 20 5 10 = 10000 :=
by
  -- Proof would go here
  sorry

end nina_expected_tomato_harvest_l130_130902


namespace concurrency_of_lines_l130_130430

variables {α : Type*} [euclidean_space α] -- Define a type for Euclidean space elements

open_locale euclidean_geometry -- Open locale for Euclidean geometry

noncomputable def problem (A B C D O : α) (P O1 O2 O3 O4 : α) : Prop :=
  cyclic_quadrilateral A B C D ∧
  circle O A B C D ∧
  intersects_at AC BD P ∧
  circumcenter O1 A B P ∧
  circumcenter O2 B C D ∧
  circumcenter O3 C D P ∧
  circumcenter O4 D A P ∧
  concurrent OP O1O3 O2O4

-- Statement of the theorem to be proved
theorem concurrency_of_lines :
  ∀ (A B C D O P O1 O2 O3 O4 : α),
  problem A B C D O P O1 O2 O3 O4 →
  concurrent OP O1O3 O2O4 :=
sorry

end concurrency_of_lines_l130_130430


namespace division_by_repeating_decimal_l130_130197

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l130_130197


namespace min_value_a_squared_ab_b_squared_l130_130889

theorem min_value_a_squared_ab_b_squared {a b t p : ℝ} (h1 : a + b = t) (h2 : ab = p) :
  a^2 + ab + b^2 ≥ 3 * t^2 / 4 := by
  sorry

end min_value_a_squared_ab_b_squared_l130_130889


namespace eight_div_repeating_three_l130_130191

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l130_130191


namespace selling_price_correctness_l130_130673

noncomputable def lower_selling_price 
  (selling_price_350 : ℝ) 
  (gain_percent : ℝ) 
  (cost_price : ℝ) 
  : ℝ := 
  cost_price + (selling_price_350 - cost_price) / (1 + gain_percent / 100)

theorem selling_price_correctness :
  lower_selling_price 350 5 200 ≈ 342.86 := 
sorry

end selling_price_correctness_l130_130673


namespace minimum_value_is_25_over_6_l130_130756

def find_minimum_value (a b : ℝ) (h1 : 2 * a + 3 * b = 6) (h2 : a > 0) (h3 : b > 0) : ℝ :=
  if a > 0 ∧ b > 0 then (2 / a + 3 / b) else 0

theorem minimum_value_is_25_over_6 (a b : ℝ)
  (h1 : 2 * a + 3 * b = 6) (h2 : a > 0) (h3 : b > 0) :
  find_minimum_value a b h1 h2 h3 = 25 / 6 :=
  sorry

end minimum_value_is_25_over_6_l130_130756


namespace evaluate_Q_at_2_and_neg2_l130_130887

-- Define the polynomial Q and the conditions
variable {Q : ℤ → ℤ}
variable {m : ℤ}

-- The given conditions
axiom cond1 : Q 0 = m
axiom cond2 : Q 1 = 3 * m
axiom cond3 : Q (-1) = 4 * m

-- The proof goal
theorem evaluate_Q_at_2_and_neg2 : Q 2 + Q (-2) = 22 * m :=
sorry

end evaluate_Q_at_2_and_neg2_l130_130887


namespace functions_symmetric_about_y_eq_x_l130_130940

theorem functions_symmetric_about_y_eq_x :
  ∀ (f g : ℝ → ℝ), (∀ x, f x = 2 * x) → (∀ x, g x = log x / log 2) → 
  (∀ x, f (g x) = x) ∧ (∀ x, g (f x) = x) := 
by 
  intros f g hf hg
  sorry

end functions_symmetric_about_y_eq_x_l130_130940


namespace tangent_line_equation_parallel_to_given_line_l130_130017

theorem tangent_line_equation_parallel_to_given_line :
  ∃ (x y : ℝ),  y = x^3 - 3 * x^2
    ∧ (3 * x^2 - 6 * x = -3)
    ∧ (y = -2)
    ∧ (3 * x + y - 1 = 0) :=
sorry

end tangent_line_equation_parallel_to_given_line_l130_130017


namespace problem_statement_l130_130390

noncomputable def f (a b x : ℝ) : ℝ :=
  a * Real.sin x + b * Real.cos x

theorem problem_statement (a b : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∀ x, f a b x ≤ f a b (π / 3)) :
  ∃ (k : ℤ), ∀ x, f a b (x + π / 3) = (Real.sqrt (a^2 + b^2)) * Real.cos x ∧
  (∀ x, f a b (x + π / 3) = f a b (-x + π / 3)) ∧
  set.eq_on (λ x, f a b (x + π / 3)) (λ x, f a b (π + 3π/2 - x)) (set.Icc (π/3) (5π/3)) :=
by sorry

end problem_statement_l130_130390


namespace B_joined_with_54000_l130_130259

theorem B_joined_with_54000 :
  ∀ (x : ℕ),
    (36000 * 12) / (x * 4) = 2 → x = 54000 :=
by 
  intro x h
  sorry

end B_joined_with_54000_l130_130259


namespace tangent_line_equation_at_point_l130_130526

theorem tangent_line_equation_at_point {x y : ℝ} (h_curve : y = x * (3 * Real.log x + 1))
  (h_point : (1, 1)) :
  ∃ m b, m = 4 ∧ b = -3 ∧ (∀ x y, y = m * x + b) := by
  sorry

end tangent_line_equation_at_point_l130_130526


namespace longest_segment_inside_cylinder_l130_130621

theorem longest_segment_inside_cylinder :
  ∀ (r h : ℝ), r = 5 → h = 12 → ∃ (d : ℝ), d = 2 * Real.sqrt 61 ∧ d = Real.sqrt ((2 * r) ^ 2 + h ^ 2) :=
by
  intros r h hr hh
  have : 2 * r = 10 := by rw [←hr]; norm_num
  have : h = 12 := by rw [←hh]
  use 2 * Real.sqrt 61
  simp [this]
  sorry

end longest_segment_inside_cylinder_l130_130621


namespace sufficient_but_not_necessary_condition_l130_130346

noncomputable theory

open_locale classical

variables {V : Type*} [inner_product_space ℝ V]

def vectors_are_nonzero (a b : V) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def vectors_sum_to_zero (a b : V) : Prop :=
  a + b = 0

def vectors_are_parallel (a b : V) : Prop :=
  ∃ k : ℝ, a = k • b

theorem sufficient_but_not_necessary_condition
  (a b : V)
  (h_nonzero : vectors_are_nonzero a b) :
  vectors_sum_to_zero a b → vectors_are_parallel a b ∧ ¬(vectors_are_parallel a b → vectors_sum_to_zero a b) :=
sorry

end sufficient_but_not_necessary_condition_l130_130346


namespace find_a_l130_130825

noncomputable def f (a x : ℝ) := (x - 1)^2 + a * x + Real.cos x

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) → 
  a = 2 :=
by
  sorry

end find_a_l130_130825


namespace compare_sqrts_l130_130405

theorem compare_sqrts (a b c : ℝ) (h1 : a = 2 * Real.sqrt 7) (h2 : b = 3 * Real.sqrt 5) (h3 : c = 5 * Real.sqrt 2):
  c > b ∧ b > a :=
by
  sorry

end compare_sqrts_l130_130405


namespace divide_by_repeating_decimal_l130_130175

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l130_130175


namespace evaluate_expression_l130_130313

def star (A B : ℚ) : ℚ := (A + B) / 3

theorem evaluate_expression : star (star 7 15) 10 = 52 / 9 := by
  sorry

end evaluate_expression_l130_130313


namespace absolute_value_distance_l130_130661

theorem absolute_value_distance (a b : ℤ) (h : Int.abs (a - b) = 4) : (a = 2 ∧ b = -2) ∨ (a = -2 ∧ b = 2) :=
sorry

end absolute_value_distance_l130_130661


namespace rationalize_denominator_correct_l130_130501

noncomputable def rationalize_denominator : Prop :=
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l130_130501


namespace students_per_class_l130_130907

theorem students_per_class (teachers principal classes total_people : ℕ) (h1 : teachers = 48) (h2 : principal = 1) (h3 : classes = 15) (h4 : total_people = 349) :
  (total_people - teachers - principal) / classes = 20 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end students_per_class_l130_130907


namespace cone_height_l130_130844

theorem cone_height (a : ℝ) (h l r : ℝ)
  (hcircumference : ℝ) (hslant_height : l = a) (hlateral_surface : hcircumference = π * a)
  (hradius_base : hcircumference = 2 * π * r) : h = (sqrt 3 / 2) * a :=
by
  -- define the necessary conditions
  sorry

end cone_height_l130_130844


namespace probability_of_no_rain_l130_130954

theorem probability_of_no_rain (prob_rain : ℚ) (prob_no_rain : ℚ) (days : ℕ) (h : prob_rain = 2/3) (h_prob_no_rain : prob_no_rain = 1 - prob_rain) :
  (prob_no_rain ^ days) = 1/243 :=
by 
  sorry

end probability_of_no_rain_l130_130954


namespace eight_div_repeating_three_l130_130188

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l130_130188


namespace paul_books_sale_l130_130480

variable (books_initial books_given books_left books_sold : Nat)

def paul_books_sale_conditions (books_initial books_given books_left : Nat) : Prop :=
  books_initial = 108 ∧ books_given = 35 ∧ books_left = 62

theorem paul_books_sale (h : paul_books_sale_conditions 108 35 62) : books_sold = 11 := by
  cases h with
  | intro h1 h2 h3 =>
  -- skipping the proof as assuming conditions to be non-trivial and requiring proper context
  sorry

end paul_books_sale_l130_130480


namespace min_rooks_to_color_grid_l130_130664

theorem min_rooks_to_color_grid (n : ℕ) (numbering : fin n.succ × fin n.succ → ℕ)
  (h_numbering : ∀ i j, 1 ≤ numbering (i, j) ∧ numbering (i, j) ≤ n * n)
  (h_distinct : ∀ i₁ j₁ i₂ j₂, numbering (i₁, j₁) = numbering (i₂, j₂) → (i₁ = i₂ ∧ j₁ = j₂)) :
  ∃ (rooks : fin n.succ → fin n.succ), ∀ i j, ∃ k, rooks k = (i, j) ∨
    (∃ i' j', (rooks k = (i', j') ∧ numbering (i', j') < numbering (i, j)) ∧
      (i' = i ∨ j' = j ∧ (∀ i'' j'', numbering (i'', j'') = numbering (i, j) → ((i'' = i ∧ j'' ≠ j) ∨ (i'' ≠ i ∧ j'' = j))))) ∧
  ∀ m, (∀ i j, ∃ k, rooks k = (i, j) ∨
    (∃ i' j', (rooks k = (i', j') ∧ numbering (i', j') < numbering (i, j)) ∧
      (i' = i ∨ j' = j ∧ (∀ i'' j'', numbering (i'', j'') = numbering (i, j) → ((i'' = i ∧ j'' ≠ j) ∨ (i'' ≠ i ∧ j'' = j))))) → m ≥ n :=
sorry

end min_rooks_to_color_grid_l130_130664


namespace rationalize_denominator_l130_130486

theorem rationalize_denominator : (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_denominator_l130_130486


namespace sum_areas_of_inscribed_circles_l130_130614

-- Define the semi-perimeter function.
def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Given conditions and question conversion
theorem sum_areas_of_inscribed_circles (a b c r : ℝ) (s := semi_perimeter a b c) :
    a > 0 → b > 0 → c > 0 → r > 0 → (s = (a + b + c) / 2) →
    let sum := π * r^2 * ((a^2 + b^2 + c^2) / s^2) in
    sum = π * r^2 * (1 + (b / s)^2 + (c / s)^2 - 2 * (b / s + c / s) + 1) :=
by
  intros
  rw [←semiperimeter_eq_sum_compl]
  sorry

end sum_areas_of_inscribed_circles_l130_130614


namespace eight_div_repeating_three_l130_130184

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l130_130184


namespace train_passes_jogger_in_37_seconds_l130_130276

-- Definitions based on the given conditions
def jogger_speed_kmh : ℝ := 9
def initial_distance_m : ℝ := 250
def train_length_m : ℝ := 120
def train_speed_kmh : ℝ := 45

-- Converted speeds from km/hr to m/s
def jogger_speed_ms : ℝ := jogger_speed_kmh * (1000 / 3600)
def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)

-- Relative speed of the train with respect to the jogger
def relative_speed_ms : ℝ := train_speed_ms - jogger_speed_ms

-- Total distance to be covered by the train to pass the jogger
def total_distance_m : ℝ := initial_distance_m + train_length_m

-- Time taken by the train to pass the jogger
def time_to_pass_jogger_s : ℝ := total_distance_m / relative_speed_ms

-- Proof problem statement
theorem train_passes_jogger_in_37_seconds :
  time_to_pass_jogger_s = 37 := 
sorry

end train_passes_jogger_in_37_seconds_l130_130276


namespace division_by_repeating_decimal_l130_130202

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l130_130202


namespace max_running_speed_l130_130637

-- Constants
def distance_AB : ℝ := sorry  -- Distance between stations A and B. We can assume 'x' in the problem.
def train_speed : ℝ := 30  -- The speed of the train in km/h.
def distance_CA : ℝ := distance_AB / 3  -- 1/3 of the distance between A and B.

-- Definition for the point where the man spots the train, running either towards A or B
def max_speed_from_C_to_A (v : ℝ) : Prop :=
  distance_CA / v = distance_AB / train_speed

-- The proof: The maximum speed at which the man can run to catch the train 
theorem max_running_speed : ∃ v : ℝ, max_speed_from_C_to_A v ∧ v = 10 :=
by
  use 10
  sorry  -- proof steps would go here

end max_running_speed_l130_130637


namespace eight_div_repeating_three_l130_130193

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l130_130193


namespace simplify_and_evaluate_expression_l130_130004

-- Define the expression
def simplified_expression (x : Real) : Real :=
  ((2 * x^2 + 2 * x) / (x^2 - 1) - (x^2 - x) / (x^2 - 2 * x + 1)) / (x / (x + 1))

-- Define the given x value
def given_x : Real := 1 - Real.sqrt 3

-- The theorem to be proved. 
theorem simplify_and_evaluate_expression :
  simplified_expression given_x = - (2 * Real.sqrt 3) / 3 + 1 :=
by 
  sorry

end simplify_and_evaluate_expression_l130_130004


namespace at_least_two_rectangles_congruent_l130_130603

-- Definitions representing the conditions of the problem
def num_rectangles : Nat := 10
def num_points : Nat := 40
def equal_arcs (num_points : Nat) : Prop := 
  ∀ i j : Fin num_points, i ≠ j → arc_length i j = arc_length j i

-- Assumption: The circle is divided into 40 equal arcs
axiom equal_arc_division : equal_arcs num_points

-- Main theorem to prove
theorem at_least_two_rectangles_congruent
  (h : equal_arcs num_points) :
  ∃ r1 r2 : Rectangle, r1 ≠ r2 ∧ are_congruent r1 r2 :=
sorry

end at_least_two_rectangles_congruent_l130_130603


namespace find_a_minus_b_l130_130938

-- Definitions based on conditions
def eq1 (a b : Int) : Prop := 2 * b + a = 5
def eq2 (a b : Int) : Prop := a * b = -12

-- Statement of the problem
theorem find_a_minus_b (a b : Int) (h1 : eq1 a b) (h2 : eq2 a b) : a - b = -7 := 
sorry

end find_a_minus_b_l130_130938


namespace bob_is_47_5_l130_130679

def bob_age (a b : ℝ) := b = 3 * a - 20
def sum_of_ages (a b : ℝ) := b + a = 70

theorem bob_is_47_5 (a b : ℝ) (h1 : bob_age a b) (h2 : sum_of_ages a b) : b = 47.5 :=
by
  sorry

end bob_is_47_5_l130_130679


namespace third_grade_total_trees_l130_130044

-- Defining the conditions
def total_students : ℕ := 100
def students_in_grades_three_and_four_are_equal : Prop := ∃ x, x + x + y = total_students
def planted_trees_by_third_grade_student : ℕ := 4
def planted_trees_by_fourth_grade_student : ℕ := 5
def planted_trees_by_fifth_grade_student : ℝ := 6.5
def total_planted_trees : ℝ := 566
def students_in_grade_three (x : ℕ) := x
def students_in_grade_four (x : ℕ) := x
def students_in_grade_five (y : ℕ) := y

theorem third_grade_total_trees (x : ℕ) (y : ℕ) 
  (h1 : 2 * x + y = total_students)
  (h2 : 4 * x + 5 * x + 6.5 * y = total_planted_trees) :
  4 * x = 84 :=
sorry

end third_grade_total_trees_l130_130044


namespace maximum_sum_of_digits_difference_l130_130460

-- Definition of the sum of the digits of a number
-- For the purpose of this statement, we'll assume the existence of a function sum_of_digits

def sum_of_digits (n : ℕ) : ℕ :=
  sorry -- Assume the function is defined elsewhere

-- Statement of the problem
theorem maximum_sum_of_digits_difference :
  ∃ x : ℕ, sum_of_digits (x + 2019) - sum_of_digits x = 12 :=
sorry

end maximum_sum_of_digits_difference_l130_130460


namespace divide_by_repeating_decimal_l130_130181

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l130_130181


namespace concurrency_of_perpendiculars_l130_130447

/-- Let ABC be an acute-angled triangle and D, E, F be the feet of the altitudes 
from A, B, and C respectively. Let P (respectively Q, respectively R) 
be the foot of the perpendicular from A (respectively B, respectively C) to EF 
(respectively FD, respectively DE). Show that the lines (AP), (BQ), and (CR) 
are concurrent. -/
theorem concurrency_of_perpendiculars (ABC : Type) [triangle ABC]
  (A B C D E F P Q R : ABC)
  (hD : altitude A B C D) 
  (hE : altitude B A C E) 
  (hF : altitude C A B F)
  (hP : perpendicular A E F P)
  (hQ : perpendicular B F D Q)
  (hR : perpendicular C D E R) :
  concurrent (line A P) (line B Q) (line C R) := 
sorry

end concurrency_of_perpendiculars_l130_130447


namespace percentage_discount_total_amount_paid_l130_130008

variable (P Q : ℝ)

theorem percentage_discount (h₁ : P > Q) (h₂ : Q > 0) :
  100 * ((P - Q) / P) = 100 * (P - Q) / P :=
sorry

theorem total_amount_paid (h₁ : P > Q) (h₂ : Q > 0) :
  10 * Q = 10 * Q :=
sorry

end percentage_discount_total_amount_paid_l130_130008


namespace combined_weight_of_new_persons_l130_130013

variable (W1 W2 : ℝ)

theorem combined_weight_of_new_persons 
    (avg_weight_increase : ℝ)
    (current_weight_sum : ℝ) 
    (replaced_weight : ℝ) :
    avg_weight_increase = 5.2 →
    current_weight_sum = 15 * avg_weight_increase →
    replaced_weight = 68 + 70 →
    W1 + W2 = current_weight_sum + replaced_weight → 
    W1 + W2 = 216 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

#print combined_weight_of_new_persons

end combined_weight_of_new_persons_l130_130013


namespace inner_tetrahedron_volume_l130_130616

def volume_of_inner_tetrahedron(cube_side : ℕ) : ℚ :=
  let base_area := (cube_side * cube_side) / 2
  let height := cube_side
  let original_tetra_volume := (1 / 3) * base_area * height
  let inner_tetra_volume := original_tetra_volume / 8
  inner_tetra_volume

theorem inner_tetrahedron_volume {cube_side : ℕ} (h : cube_side = 2) : 
  volume_of_inner_tetrahedron cube_side = 1 / 6 := 
by
  rw [h]
  unfold volume_of_inner_tetrahedron 
  norm_num
  sorry

end inner_tetrahedron_volume_l130_130616


namespace find_unit_prices_calculate_total_expenditure_l130_130554

/-
Definitions based on given conditions.
-/
variables (p_A p_B : ℕ)   -- unit prices for type A and type B books
variables (n_A n_B: ℕ)   -- number of type A and type B books purchased

-- Spending conditions
def spend_A : ℕ := 3000
def spend_B : ℕ := 1600

-- Unit price relationship
def unit_price_relation : Prop := p_A = 3 * p_B / 2

-- Quantity relationship
def quantity_relation : Prop := n_A = n_B + 20

-- Total expenditure on World Book Day (20 books of type A and 25 of type B at 20% discount)
def total_expenditure_world_book_day : ℕ :=
  let discount := 4 / 5 in
  (20 * (p_A * discount)) + (25 * (p_B * discount))

-- Correct answers
def correct_unit_price_A := 30
def correct_unit_price_B := 20
def correct_total_expenditure := 880

/- 
Proof problems.
-/
theorem find_unit_prices (h1: spend_A = p_A * n_A) (h2: spend_B = p_B * n_B) 
  (h3: unit_price_relation) (h4: quantity_relation) :
  p_A = correct_unit_price_A ∧ p_B = correct_unit_price_B := 
sorry

theorem calculate_total_expenditure (h1: spend_A = p_A * n_A) (h2: spend_B = p_B * n_B) 
  (h3: unit_price_relation) (h4: quantity_relation) :
  total_expenditure_world_book_day p_A p_B = correct_total_expenditure := 
sorry

end find_unit_prices_calculate_total_expenditure_l130_130554


namespace equilateral_triangle_sum_l130_130464

theorem equilateral_triangle_sum (a u v w : ℝ)
  (h1: u^2 + v^2 = w^2):
  w^2 + Real.sqrt 3 * u * v = a^2 := 
sorry

end equilateral_triangle_sum_l130_130464


namespace minimum_value_fraction_l130_130024

-- Conditions
variables (a : ℝ) (m n : ℝ)
-- Constraints
variables (a_pos : a > 0) (a_not_one : a ≠ 1) (m_pos : m > 0) (n_pos : n > 0)
-- The point A lies on both the function and the line
variables (A_on_graph : (-2 : ℝ), -1 = log a (1 : ℝ) - 1)
variables (A_on_line : -2 * m - n + 2 = 0)

-- Conclusion
theorem minimum_value_fraction (a_pos : a > 0)
                              (a_not_one : a ≠ 1)
                              (m_pos : m > 0)
                              (n_pos : n > 0)
                              (A_on_graph : (-2 : ℝ), -1 = log a (1 : ℝ) - 1)
                              (A_on_line : -2 * m - n + 2 = 0) :
  ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ -2 * m - n + 2 = 0 → (2 / m + 1 / n) = 9 / 2 :=
by
safety_skip := sorry

end minimum_value_fraction_l130_130024


namespace general_term_a_n_sum_T_n_l130_130745

open BigOperators

-- Definition of the first sequence {a_n} 
def a_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else 1 / 2^(n-1)

-- The sum of the first n terms S_n
def S_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a_n (i+1)

-- First part: Proving the general term formula of {a_n}
theorem general_term_a_n (n : ℕ) : a_n n = 1 / 2^(n-1) :=
  by sorry

-- Definition of the second sequence {b_n}
def b_n (n : ℕ) : ℝ := n * a_n n

-- The sum of the first n terms T_n for {b_n}
def T_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b_n (i+1)

-- Second part: Proving the sum of the first n terms of {b_n} 
from the given formula of {a_n}
theorem sum_T_n (n : ℕ) : T_n n = 4 - (n + 2) / 2^(n-1) :=
  by sorry

end general_term_a_n_sum_T_n_l130_130745


namespace non_intersecting_chords_20_points_l130_130607

noncomputable def a (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | n + 1 => ∑ k in Finset.range (n + 1), (a k) * (a (n - k))

theorem non_intersecting_chords_20_points :
  a 10 = 16796 :=
by
  sorry

end non_intersecting_chords_20_points_l130_130607


namespace tan_theta_dilation_rotation_l130_130888

theorem tan_theta_dilation_rotation
  (k θ : ℝ) (hk : k > 0)
  (hRD : (matrix.std_basis_matrix 2 2 0 ((λ x, if x = 0 then k else 0)) 
              ⬝
              λ x y, if (x, y) = (0,0) then (real.cos θ) else if(y,x)= (1,0) then (real.sin θ) else if  (x,0)= y then  if (1,0)= y then -(real.sin θ) else (real.cos θ) else  0) ) = 
            ![![12, -5], ![5, 12]]) :
  real.tan θ = 5 / 12 :=
begin
  sorry
end

end tan_theta_dilation_rotation_l130_130888


namespace division_by_repeating_decimal_l130_130209

theorem division_by_repeating_decimal :
  (8 : ℚ) / (0.3333333333333333 : ℚ) = 24 :=
by {
  have h : (0.3333333333333333 : ℚ) = 1/3 :=
    by {
      sorry
    },
  rw h,
  field_simp,
  norm_num
}

end division_by_repeating_decimal_l130_130209


namespace percentage_of_copper_in_second_alloy_l130_130298

theorem percentage_of_copper_in_second_alloy
  (w₁ w₂ w_total : ℝ)
  (p₁ p_total : ℝ)
  (h₁ : w₁ = 66)
  (h₂ : p₁ = 0.10)
  (h₃ : w_total = 121)
  (h₄ : p_total = 0.15) :
  (w_total - w₁) * 0.21 = w_total * p_total - w₁ * p₁ := 
  sorry

end percentage_of_copper_in_second_alloy_l130_130298


namespace eight_div_repeat_three_l130_130119

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l130_130119


namespace smallest_integer_n_l130_130998

theorem smallest_integer_n (n : ℕ) : (1 / 2 : ℝ) < n / 9 ↔ n ≥ 5 := 
sorry

end smallest_integer_n_l130_130998


namespace normal_distribution_symmetry_l130_130363

theorem normal_distribution_symmetry {σ : ℝ} (hσ : 0 < σ) :
  P(ξ < 2016) = 0.5 :=
by 
  sorry

end normal_distribution_symmetry_l130_130363


namespace composite_prob_from_50_nat_numbers_l130_130031

theorem composite_prob_from_50_nat_numbers : 
  (∃ n, 1 ≤ n ∧ n ≤ 50 ∧ ∑ i in finset.range 50, if (¬is_prime (i + 1) ∧ (i + 1) ≠ 1) then 1 else 0 = 34 ∧ (34: ℝ) / 50 = 0.68) :=
by {
  sorry,
}

end composite_prob_from_50_nat_numbers_l130_130031


namespace greatest_possible_z_l130_130590

theorem greatest_possible_z (x y z : ℕ) (hx_prime : Nat.Prime x) (hy_prime : Nat.Prime y) (hz_prime : Nat.Prime z)
  (hx_cond : 7 < x) (hy_cond : y < 15) (hx_lt_y : x < y) (hz_gt_zero : z > 0) 
  (hy_sub_x_div_z : (y - x) % z = 0) : z = 2 := 
sorry

end greatest_possible_z_l130_130590


namespace diminish_value_l130_130985

theorem diminish_value (a b : ℕ) (h1 : a = 1015) (h2 : b = 12) (h3 : b = 16) (h4 : b = 18) (h5 : b = 21) (h6 : b = 28) :
  ∃ k, a - k = lcm (lcm (lcm b b) (lcm b b)) (lcm b b) ∧ k = 7 :=
sorry

end diminish_value_l130_130985


namespace composite_probability_matches_l130_130032

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, m > 1 ∧ m < n ∧ n % m = 0

def count_composites_up_to (n : ℕ) : ℕ :=
  (Finset.range n).filter is_composite |>.card

theorem composite_probability_matches :
  (count_composites_up_to 51) / 50 = 0.68 := 
sorry

end composite_probability_matches_l130_130032


namespace rectangle_perimeter_l130_130929

variables (L B P : ℝ)

theorem rectangle_perimeter (h1 : B = 0.60 * L) (h2 : L * B = 37500) : P = 800 :=
by
  sorry

end rectangle_perimeter_l130_130929


namespace probability_no_rain_five_days_l130_130951

noncomputable def probability_of_no_rain (rain_prob : ℚ) (days : ℕ) :=
  (1 - rain_prob) ^ days

theorem probability_no_rain_five_days :
  probability_of_no_rain (2/3) 5 = 1/243 :=
by sorry

end probability_no_rain_five_days_l130_130951


namespace reduced_coffee_per_week_l130_130304

-- Definitions for the problem conditions
def thermos_volume : ℕ := 20

def daily_fill_amount (day : Nat) : ℕ :=
  if day = 1 ∨ day = 3 then 2 * thermos_volume else
  if day = 2 ∨ day = 4 then 3 * thermos_volume else
  if day = 5 then thermos_volume else 0

def daily_reduced_amount (day : Nat) : ℕ :=
  if day = 1 ∨ day = 2 ∨ day = 4 then daily_fill_amount(day) / 4 else
  if day = 3 ∨ day = 5 then daily_fill_amount(day) / 2 else 0

def weekly_reduced_amount : ℕ :=
  let mondays := daily_reduced_amount 1
  let tuesdays := daily_reduced_amount 2
  let wednesdays := daily_reduced_amount 3
  let thursdays := daily_reduced_amount 4
  let fridays := daily_reduced_amount 5
  mondays + tuesdays + wednesdays + thursdays + fridays

-- Theorem to be proven
theorem reduced_coffee_per_week : weekly_reduced_amount = 70 := by sorry

end reduced_coffee_per_week_l130_130304


namespace sum_of_m_values_l130_130660

theorem sum_of_m_values :
  let A := (0,0) in
  let B := (4,4) in
  let C := (10 * m, 0) in
  (∃ m : ℝ, ∀ A B C, (y = m * x) divides triangle A B C into two equal areas) →
  (sum(all_possible_m_values) = -0.4) :=
sorry

end sum_of_m_values_l130_130660


namespace domain_of_f_2x_minus_3_l130_130385

noncomputable def f (x : ℝ) := 2 * x + 1

theorem domain_of_f_2x_minus_3 :
  (∀ x, 1 ≤ 2 * x - 3 ∧ 2 * x - 3 ≤ 5 → (2 ≤ x ∧ x ≤ 4)) :=
by
  sorry

end domain_of_f_2x_minus_3_l130_130385


namespace number_of_integers_satisfying_inequality_l130_130814

theorem number_of_integers_satisfying_inequality :
  set.count {n : ℤ | (n + 2) * (n - 8) ≤ 0} = 11 :=
sorry

end number_of_integers_satisfying_inequality_l130_130814


namespace roots_of_polynomial_l130_130724

noncomputable def polynomial := (x : ℝ) => x^4 - 4 * x^3 + 3 * x^2 + 2 * x - 6

theorem roots_of_polynomial : { x : ℝ // polynomial x = 0 } = { -1, 3 } := by
  sorry

end roots_of_polynomial_l130_130724


namespace divide_by_repeating_decimal_l130_130180

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l130_130180


namespace bob_age_l130_130683

theorem bob_age (a b : ℝ) 
    (h1 : b = 3 * a - 20)
    (h2 : b + a = 70) : 
    b = 47.5 := by
    sorry

end bob_age_l130_130683


namespace pow_100_mod_18_l130_130608

theorem pow_100_mod_18 : (5 ^ 100) % 18 = 13 := by
  -- Define the conditions
  have h1 : (5 ^ 1) % 18 = 5 := by norm_num
  have h2 : (5 ^ 2) % 18 = 7 := by norm_num
  have h3 : (5 ^ 3) % 18 = 17 := by norm_num
  have h4 : (5 ^ 4) % 18 = 13 := by norm_num
  have h5 : (5 ^ 5) % 18 = 11 := by norm_num
  have h6 : (5 ^ 6) % 18 = 1 := by norm_num
  
  -- The required theorem is based on the conditions mentioned
  sorry

end pow_100_mod_18_l130_130608


namespace range_of_k_l130_130779

theorem range_of_k (n : ℕ) (k : ℝ) :
  (∃ x y : ℝ, (2n - 1 < x ∧ x ≤ 2n + 1) ∧ (2n - 1 < y ∧ y ≤ 2n + 1) ∧ x ≠ y
  ∧ |x - 2n| = k * real.sqrt x ∧ |y - 2n| = k * real.sqrt y) ↔
  0 < k ∧ k ≤ 1 / real.sqrt (2n + 1) := 
sorry

end range_of_k_l130_130779


namespace sams_charge_per_sheet_is_1_5_l130_130648

variable (x : ℝ)
variable (a : ℝ) -- John's Photo World's charge per sheet
variable (b : ℝ) -- Sam's Picture Emporium's one-time sitting fee
variable (c : ℝ) -- John's Photo World's one-time sitting fee
variable (n : ℕ) -- Number of sheets

def johnsCost (n : ℕ) (a c : ℝ) := n * a + c
def samsCost (n : ℕ) (x b : ℝ) := n * x + b

theorem sams_charge_per_sheet_is_1_5 :
  ∀ (a b c : ℝ) (n : ℕ), a = 2.75 → b = 140 → c = 125 → n = 12 →
  johnsCost n a c = samsCost n x b → x = 1.50 := by
  intros a b c n ha hb hc hn h
  sorry

end sams_charge_per_sheet_is_1_5_l130_130648


namespace odd_function_property_f_nonnegative_expression_l130_130767

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then -x * real.log (2 - x) else -x * real.log (2 + x)

theorem odd_function_property (x : ℝ) (h : x < 0) : f (-x) = -f x := by
  sorry

theorem f_nonnegative_expression {x : ℝ} (h : 0 ≤ x) : f x = -x * real.log (2 + x) :=
  by sorry

end odd_function_property_f_nonnegative_expression_l130_130767


namespace area_of_quadrilateral_ABCD_l130_130478

theorem area_of_quadrilateral_ABCD
    (A B C D : Type)
    (is_on_shore : A ∧ B ∧ C ∧ D)
    (ac_divides_island : ∀ (p : Type), (p ∈ A ∨ p ∈ C) → (p ∉ B ∨ p ∉ D))
    (bd_shorter_than_ac : ∀ (ac bd : ℝ), bd < ac)
    (cyclist_speed_asphalt : ∀ (asphalt_speed : ℝ), asphalt_speed = 15)
    (cyclist_time_travel : ∀ (p : in A ∨ in C ∨ in D) (t : ℝ), t = 2): 
    ∃ area,
      area = 450 :=
sorry

end area_of_quadrilateral_ABCD_l130_130478


namespace evaluate_i_powers_l130_130327

theorem evaluate_i_powers :
  (complex.I ^ 11 + complex.I ^ 111) = -2 * complex.I :=
by sorry

end evaluate_i_powers_l130_130327


namespace length_of_PS_is_7_5_l130_130060

variables (P Q R T S : Type) 
variable [inhabited P]

-- Triangle PQR is isosceles with PQ = QR.
variable (PQ QR : ℝ) (PQR_is_isosceles : PQ = QR)

-- Point S is the midpoint of both QR and PT.
variable (S_is_midpoint_QR : (QR / 2) = 1/2 * QR)
variable (S_is_midpoint_PT: ℝ) -- PT will be inferred

-- RT is 15 units long.
variable (RT : ℝ) (RT_is_15 : RT = 15)

noncomputable def length_of_PS : ℝ := QR / 2

-- The theorem to prove
theorem length_of_PS_is_7_5 : length_of_PS QR = 7.5 :=
by
  linarith [PQR_is_isosceles, RT_is_15]

end length_of_PS_is_7_5_l130_130060


namespace find_functions_satisfying_condition_l130_130713

noncomputable def function_satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 → a * b * c * d = 1 →
  (f a + f b) * (f c + f d) = (a + b) * (c + d)

theorem find_functions_satisfying_condition :
  ∀ f : ℝ → ℝ, function_satisfies_condition f →
    (∀ x : ℝ, x > 0 → f x = x) ∨ (∀ x : ℝ, x > 0 → f x = 1 / x) :=
sorry

end find_functions_satisfying_condition_l130_130713


namespace Eight_div_by_repeating_decimal_0_3_l130_130167

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l130_130167


namespace sams_charge_per_sheet_l130_130650

theorem sams_charge_per_sheet :
  ∃ x : ℝ, x = 1.5 ∧ (12 * 2.75 + 125) = (12 * x + 140) :=
by
  use 1.5
  split
  sorry

end sams_charge_per_sheet_l130_130650


namespace division_of_decimal_l130_130107

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l130_130107


namespace min_value_of_sum_of_sides_proof_l130_130436

noncomputable def min_value_of_sum_of_sides (a b c : ℝ) (angleC : ℝ) : ℝ :=
  if (angleC = 60 * (Real.pi / 180)) ∧ ((a + b)^2 - c^2 = 4) then 4 * Real.sqrt 3 / 3 
  else 0

theorem min_value_of_sum_of_sides_proof (a b c : ℝ) (angleC : ℝ) 
  (h1 : angleC = 60 * (Real.pi / 180)) 
  (h2 : (a + b)^2 - c^2 = 4) 
  : min_value_of_sum_of_sides a b c angleC = 4 * Real.sqrt 3 / 3 := 
by
  sorry

end min_value_of_sum_of_sides_proof_l130_130436


namespace combined_bills_weighted_average_bd_l130_130519

structure Bill where
  pw : ℝ
  td : ℝ

def banker_discount (b : Bill) : ℝ :=
  b.td + (b.td^2 / b.pw)

noncomputable def weighted_average_banker_discount (bills : List Bill) : ℝ :=
  let total_pw := (bills.map (·.pw)).sum
  let weighted_bd := (bills.map (λ b => banker_discount b * b.pw)).sum
  weighted_bd / total_pw

theorem combined_bills_weighted_average_bd :
  let bills := [
    { pw := 8000, td := 360 },
    { pw := 10000, td := 450 },
    { pw := 12000, td := 480 },
    { pw := 15000, td := 500 }
  ]
  weighted_average_banker_discount bills = 476.72 :=
by
  sorry

end combined_bills_weighted_average_bd_l130_130519


namespace mark_hours_left_l130_130676

theorem mark_hours_left (sick_days_per_year vacation_days_per_year half_usage days_per_year hours_per_day : ℕ) : 
  sick_days_per_year = 10 ∧ vacation_days_per_year = 10 ∧ half_usage = 5 ∧ days_per_year = 10 ∧ hours_per_day = 8 →
  (10 - 5) * 8 + (10 - 5) * 8 = 80 := 
by 
  intros h
  cases h with h_sick h_tr
  cases h_tr with h_vac h_tr
  cases h_tr with h_half h_tr
  cases h_tr with h_days h_hours
  rw [h_sick, h_vac, h_half, h_days, h_hours]
  simp
  sorry

end mark_hours_left_l130_130676


namespace least_positive_angle_l130_130332

theorem least_positive_angle (θ : ℝ) :
  (∃ θ, 0 < θ ∧ θ ≤ 360 ∧ cos (15 * real.pi / 180) = sin (45 * real.pi / 180) + sin (θ * real.pi / 180)) ↔ θ = 195 :=
by
  sorry

end least_positive_angle_l130_130332


namespace find_omega_l130_130787

theorem find_omega (ω : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x : ℝ, f x = sin (2 * ω * x) - (sqrt 3) * cos (2 * ω * x))
    ∧ (∃ d : ℝ, d = π / 3 ∧ d = π / |ω|)) → ω = 3 / 2 ∨ ω = -3 / 2 :=
by
  sorry

end find_omega_l130_130787


namespace inversely_proportional_x_y_l130_130517

-- Define the problem statement
theorem inversely_proportional_x_y (x y k : ℝ)
  (h1 : x * y = k)
  (h2 : x = 5)
  (h3 : y = 15) :
  let y' := -30 in
  let x' := -5/2 in
  (x' * y' = k) :=
by 
  sorry

#check inversely_proportional_x_y

end inversely_proportional_x_y_l130_130517


namespace infinite_subsequence_exists_l130_130598

theorem infinite_subsequence_exists {α : Type*} (h : α → ℕ) (s : ℕ → α)
  (hinf : ∀ n, ∃ m > n, true) :
  ∃ (f : ℕ → ℕ), (∀ n, f n < f (n + 1)) ∨ ∃ (f : ℕ → ℕ), (∀ n, f n > f (n + 1)) :=
  sorry

end infinite_subsequence_exists_l130_130598


namespace solve_abs_equation_l130_130342

theorem solve_abs_equation (x : ℝ) (h : abs (x - 20) + abs (x - 18) = abs (2 * x - 36)) : x = 19 :=
sorry

end solve_abs_equation_l130_130342


namespace unique_n_with_arith_prog_divisors_l130_130524

theorem unique_n_with_arith_prog_divisors (n : ℕ) 
  (h1 : ∃ k > 3, (∀ i j : ℕ, i < j → j = i + 1 → (d i) < (d j) ∧ (d j) ≤ n))
  (h2 : ∀ i : ℕ, i ≥ 1 → (d i + 1) - (d i) = (d i) - (d i - 1)) 
  : n = 10 :=
by
  sorry

end unique_n_with_arith_prog_divisors_l130_130524


namespace cos_sub_sin_value_l130_130355

theorem cos_sub_sin_value (x : ℝ) 
  (h₀ : sin x + cos x = 1/5)
  (h₁ : 0 < x ∧ x < π) :
  cos x - sin x = -7/5 :=
sorry

end cos_sub_sin_value_l130_130355


namespace least_fib_angle_l130_130520

open Nat

-- Define a predicate to check if a number is in the Fibonacci sequence
def is_fib : ℕ → Prop
| 0 => True
| 1 => True
| n => ∃ (a b : ℕ), is_fib a ∧ is_fib b ∧ (a + b = n)

-- The main statement that needs to be proven
theorem least_fib_angle (a b : ℕ) (ha : is_fib a) (hb : is_fib b) (h : a + b = 90) (h1 : a > b) : b = 1 :=
by 
  sorry

end least_fib_angle_l130_130520


namespace final_punch_percentage_l130_130467

theorem final_punch_percentage
    (initial_volume : ℚ)
    (initial_percentage : ℚ)
    (added_volume : ℚ)
    (initial_volume = 2)
    (initial_percentage = 0.10)
    (added_volume = 0.4) :
    let final_volume := initial_volume + added_volume
    let final_pure_juice := (initial_percentage * initial_volume) + added_volume
    (final_pure_juice / final_volume) * 100 = 25 := 
by
  sorry

end final_punch_percentage_l130_130467


namespace rationalization_correct_l130_130498

noncomputable def rationalize_denominator (a b : ℝ) : ℝ :=
  a / (b + 1)

theorem rationalization_correct :
  rationalize_denominator 1 (sqrt 3 - 1) = (sqrt 3 + 1) / 2 :=
by
  sorry

end rationalization_correct_l130_130498


namespace rationalization_correct_l130_130497

noncomputable def rationalize_denominator (a b : ℝ) : ℝ :=
  a / (b + 1)

theorem rationalization_correct :
  rationalize_denominator 1 (sqrt 3 - 1) = (sqrt 3 + 1) / 2 :=
by
  sorry

end rationalization_correct_l130_130497


namespace probability_no_rain_five_days_l130_130949

noncomputable def probability_of_no_rain (rain_prob : ℚ) (days : ℕ) :=
  (1 - rain_prob) ^ days

theorem probability_no_rain_five_days :
  probability_of_no_rain (2/3) 5 = 1/243 :=
by sorry

end probability_no_rain_five_days_l130_130949


namespace divide_by_repeating_decimal_l130_130092

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l130_130092


namespace paving_rate_correct_l130_130946

-- Define the constants
def length (L : ℝ) := L = 5.5
def width (W : ℝ) := W = 4
def cost (C : ℝ) := C = 15400
def area (A : ℝ) := A = 22

-- Given the definitions above, prove the rate per sq. meter
theorem paving_rate_correct (L W C A : ℝ) (hL : length L) (hW : width W) (hC : cost C) (hA : area A) :
  C / A = 700 := 
sorry

end paving_rate_correct_l130_130946


namespace find_a_if_f_is_even_l130_130831

def f (a : ℝ) (x : ℝ) : ℝ := (x-1)^2 + a*x + Real.sin(x + Real.pi / 2)

theorem find_a_if_f_is_even (a : ℝ) (h : ∀ x : ℝ, f(a, x) = f(a, -x)) : a = 2 := by
  sorry

end find_a_if_f_is_even_l130_130831


namespace ω_range_l130_130007

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  sin (ω * x + (real.pi / 3))

theorem ω_range (ω : ℝ) (h_extr : ∃! t ∈ Ioo 0 real.pi, (deriv (f ω) t = 0)) (h_zeros : ∃! t ∈ Ioo 0 real.pi, (f ω t = 0)) : 
  13 / 6 < ω ∧ ω ≤ 8 / 3 :=
begin
  sorry
end

end ω_range_l130_130007


namespace unique_ordered_pairs_l130_130914

def people_sitting_at_round_table (n : ℕ) := { i : ℕ // i < n }

def configurations (n : ℕ) := { config : Finset (people_sitting_at_round_table n) // 
  ∀ i ∈ config, ∃ j, j ∈ config ∧ (adjacent i j n) }

noncomputable def count_unique_pairs (configs : Finset (Finset (people_sitting_at_round_table 7))) : ℕ :=
Finset.card configs

theorem unique_ordered_pairs : count_unique_pairs (configurations 7) = 4 :=
by
  -- elaborate the proof here based on the problem and solution information
  sorry

variables (i j : people_sitting_at_round_table 7)

-- Helper definition for adjacency in a round table
def adjacent (i j : people_sitting_at_round_table 7) (n : ℕ) : Prop :=
(j.val = (i.val + 1) % n) ∨ (j.val = (i.val + n - 1) % n)


end unique_ordered_pairs_l130_130914


namespace twenty_fifth_is_monday_l130_130856

-- Definitions based on conditions
def is_saturday (date : ℕ) : Prop := ... -- A definition to check if a particular date is a Saturday
def is_even (date : ℕ) : Prop := date % 2 = 0
def month_has_three_even_saturdays (dates : List ℕ) : Prop := 
  dates.filter (λ d, is_even d ∧ is_saturday d).length = 3

-- Given a month with certain dates, prove the 25th is a Monday
theorem twenty_fifth_is_monday (dates : List ℕ) (h : month_has_three_even_saturdays dates) : 
  weekday_of 25 = "Monday" := 
sorry

end twenty_fifth_is_monday_l130_130856


namespace eight_div_repeating_three_l130_130083

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l130_130083


namespace q_range_l130_130454

-- Define the function q based on the given conditions
def q (x : ℝ) : ℝ :=
  if prime (⌊x⌋) then
    x + 2
  else
    let y := greatest_prime_factor (⌊x⌋) in
    q y + (x + 2 - ⌊x⌋)

-- Prove that the range of the function q in the interval 3 ≤ x ≤ 15
-- is equal to [5, 10) ∪ [11, 11] ∪ [13, 16)
theorem q_range :
  set.range (q) = set.union (set.union (set.Ico 5 10) (set.Icc 11 11)) (set.Ico 13 16) :=
sorry

end q_range_l130_130454


namespace division_of_repeating_decimal_l130_130074

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l130_130074


namespace count_integers_in_interval_l130_130806

theorem count_integers_in_interval : 
  ∃ (n : ℕ), (∀ (x : ℤ), (-2 ≤ x ∧ x ≤ 8 → ∃ (k : ℕ), k < n ∧ x = -2 + k)) ∧ n = 11 := 
by
  sorry

end count_integers_in_interval_l130_130806


namespace compound_proposition_truth_l130_130410

theorem compound_proposition_truth (p q : Prop) (h1 : ¬p ∨ ¬q = False) : (p ∧ q) ∧ (p ∨ q) :=
by
  sorry

end compound_proposition_truth_l130_130410


namespace max_distance_from_circle_to_line_l130_130394

noncomputable def circle : ℝ → Prop := 
  λ P, ∃ x y : ℝ, x ^ 2 + y ^ 2 - 4 * x + 3 = 0 ∧ P = x + y

noncomputable def line (m : ℝ) : ℝ × ℝ → Prop := 
  λ P, ∃ x y : ℝ, x + m * y + 1 = 0 ∧ P = (x, y)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ := 
  real.sqrt ((P.fst - Q.fst) ^ 2 + (P.snd - Q.snd) ^ 2)

theorem max_distance_from_circle_to_line 
  (m : ℝ) 
  (circle_condition : ∃ x y, x^2 + y^2 - 4*x + 3 = 0) 
  (line_condition : ∃ x y, x + m*y + 1 = 0) 
  (P : ℝ × ℝ) 
  (point_on_circle : circle P.fst + P.snd)
  (line_P : ℝ × ℝ)
  (point_on_line : line m line_P) : 
  ∃ d : ℝ, d = 4 := 
by 
  sorry

end max_distance_from_circle_to_line_l130_130394


namespace find_x_values_l130_130892

def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem find_x_values :
  {x : ℝ | f (f x) = f x} = {0, 4, 5, -1} :=
by
  sorry

end find_x_values_l130_130892


namespace sum_of_distinct_prime_factors_of_396_l130_130569

theorem sum_of_distinct_prime_factors_of_396 : 
  (∑ p in (Nat.factors 396).toFinset, p) = 16 := 
by
  sorry

end sum_of_distinct_prime_factors_of_396_l130_130569


namespace peter_total_books_is_20_l130_130908

noncomputable def total_books_peter_has (B : ℝ) : Prop :=
  let Peter_Books_Read := 0.40 * B
  let Brother_Books_Read := 0.10 * B
  Peter_Books_Read = Brother_Books_Read + 6

theorem peter_total_books_is_20 :
  ∃ B : ℝ, total_books_peter_has B ∧ B = 20 := 
by
  sorry

end peter_total_books_is_20_l130_130908


namespace ticket_cost_is_4_l130_130061

-- Define the number of tickets required for each ride
def rollercoaster_tickets (x : ℕ) : ℕ := 3 * x
def catapult_tickets (x : ℕ) : ℕ := 2 * x
def ferris_wheel_tickets : ℕ := 1

-- Define the total number of tickets
def total_tickets (x : ℕ) : ℕ := rollercoaster_tickets x + catapult_tickets x + ferris_wheel_tickets

-- The total tickets Turner needs is 21
axiom total_tickets_needed : ℕ := 21

-- Prove that the cost per ride for the rollercoaster and the Catapult is 4
theorem ticket_cost_is_4 : ∃ x : ℕ, total_tickets x = total_tickets_needed ∧ x = 4 :=
by
  sorry

end ticket_cost_is_4_l130_130061


namespace total_rattlesnakes_l130_130048

-- Definitions based on the problem's conditions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def other_snakes : ℕ := total_snakes - (pythons + boa_constrictors)

-- Statement to be proved
theorem total_rattlesnakes : other_snakes = 40 := 
by 
  -- Skipping the proof
  sorry

end total_rattlesnakes_l130_130048


namespace cos_alpha_plus_2pi_over_3_l130_130354

theorem cos_alpha_plus_2pi_over_3
  (α : ℝ)
  (h1 : sin (α + π / 3) + sin α = -4 * real.sqrt 3 / 5)
  (h2 : -π / 2 < α ∧ α < 0) :
  cos (α + 2 * π / 3) = 4 / 5 :=
sorry

end cos_alpha_plus_2pi_over_3_l130_130354


namespace lower_selling_price_l130_130671

theorem lower_selling_price (cost_price selling_price : ℝ) (profit_percentage : ℝ) (lower_price lower_profit : ℝ) :
  selling_price = 350 →
  cost_price = 200 →
  profit_percentage = 0.05 →
  lower_profit + profit_percentage * lower_profit = selling_price - cost_price →
  lower_price = cost_price + lower_profit →
  lower_price ≈ 343 :=
by
  sorry

end lower_selling_price_l130_130671


namespace composite_probability_matches_l130_130034

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, m > 1 ∧ m < n ∧ n % m = 0

def count_composites_up_to (n : ℕ) : ℕ :=
  (Finset.range n).filter is_composite |>.card

theorem composite_probability_matches :
  (count_composites_up_to 51) / 50 = 0.68 := 
sorry

end composite_probability_matches_l130_130034


namespace proof_l130_130406

noncomputable def a : ℝ := Real.logb 2 (2 * Real.sqrt 2)
noncomputable def b : ℝ := by {
  have h : 2 * Real.sqrt 2 = 8 ^ (1 / 2) := by sorry,
  exact Real.logb 8 (2 * Real.sqrt 2)
}

theorem proof : a + b = 2 := by {
  have ha : a = Real.logb 2 (2 * Real.sqrt 2) := rfl,
  have hb : b = Real.logb 8 (2 * Real.sqrt 2) := rfl,
  sorry
}

end proof_l130_130406


namespace eight_div_repeating_three_l130_130145

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l130_130145


namespace number_of_correct_propositions_zero_l130_130383

-- Define conditions based on given propositions and definition of a regular pyramid.
def is_regular_polygon (P : Type) [polygon P] : Prop := 
sorry -- definition of a regular polygon

def apex_projection (P : Type) [polyhedron P] : P -> Prop :=
sorry -- checks if projection of apex of pyramid onto the base is the center of the base polygon

def is_regular_pyramid (P : Type) [polyhedron P] : Prop :=
is_regular_polygon (base P) ∧ apex_projection P (apex P)

-- Given propositions conditions
def prop1 (P : Type) [polyhedron P] : Prop :=
is_regular_polygon (base P) -> is_regular_pyramid P

def prop2 (P : Type) [polyhedron P] : Prop :=
(lateral_edges_equal P) -> is_regular_pyramid P

def prop3 (P : Type) [polyhedron P] : Prop :=
(lateral_edges_equal_angle_base P) -> is_regular_pyramid P

def prop4 (P : Type) [polyhedron P] : Prop :=
(dihedral_angles_equal P) -> is_regular_pyramid P

-- Main theorem to prove that the number of correct propositions is 0
theorem number_of_correct_propositions_zero
  {P : Type} [polyhedron P] :
  ¬ (prop1 P ∨ prop2 P ∨ prop3 P ∨ prop4 P) :=
sorry  -- actual proof needed

end number_of_correct_propositions_zero_l130_130383


namespace eight_div_repeat_three_l130_130123

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l130_130123


namespace exists_negative_fraction_lt_four_l130_130581

theorem exists_negative_fraction_lt_four : 
  ∃ (x : ℚ), x < 0 ∧ |x| < 4 := 
sorry

end exists_negative_fraction_lt_four_l130_130581


namespace probability_at_least_6_heads_in_9_flips_l130_130269

def fair_coin_flips := 9
def total_outcomes := 512

/-- The probability of obtaining at least 'k' consecutive heads in 'n' flips of a fair coin. -/
def at_least_k_consecutive_heads_probability (k n : ℕ) : ℚ :=
∑ i in ((list.range (n - k + 1)).map (λ start_pos, 
  let end_pos := start_pos + k - 1 in 
  ((2 : ℚ)^(start_pos) + (n - end_pos - 1)))) 
  , (1 / (2^n : ℚ))

theorem probability_at_least_6_heads_in_9_flips :
  at_least_k_consecutive_heads_probability 6 9 = 49 / 512 := sorry

end probability_at_least_6_heads_in_9_flips_l130_130269


namespace arithmetic_sequence_a1_a7_a3_a5_l130_130427

noncomputable def arithmetic_sequence_property (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_a1_a7_a3_a5 (a : ℕ → ℝ) (h_arith : arithmetic_sequence_property a)
  (h_cond : a 1 + a 7 = 10) : a 3 + a 5 = 10 :=
by
  sorry

end arithmetic_sequence_a1_a7_a3_a5_l130_130427


namespace solve_quadratic_eq_solve_inequality_system_l130_130253

theorem solve_quadratic_eq (x: ℝ) :
  x^2 - 4*x + 1 = 0 ↔ x = 2 + sqrt 3 ∨ x = 2 - sqrt 3 := by
sorry

theorem solve_inequality_system (x: ℝ) :
  (3*x + 5 ≥ 2) ∧ ((x - 1) / 2 < (x + 1) / 4) ↔ (-1 ≤ x ∧ x < 3) := by
sorry

end solve_quadratic_eq_solve_inequality_system_l130_130253


namespace select_television_sets_l130_130350

theorem select_television_sets :
  (choose 9 3) - (choose 4 3) - (choose 5 3) = 70 := by
  sorry

end select_television_sets_l130_130350


namespace find_four_numbers_l130_130546

theorem find_four_numbers (a b c d : ℕ) : 
  a + b + c + d = 45 ∧ (∃ k : ℕ, a + 2 = k ∧ b - 2 = k ∧ 2 * c = k ∧ d / 2 = k) → (a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20) :=
by
  sorry

end find_four_numbers_l130_130546


namespace largest_measure_event_l130_130697

-- Define the probability of each event on a fair die
def P (a : ℕ) : ℚ := 1 / 6

-- Define the "measure" of an event
def f (p : ℚ) : ℚ := log (p + 1 / p)

-- Define the events as the sum of probabilities for face values
def P_A : ℚ := P 2
def P_B : ℚ := P 1 + P 2
def P_C : ℚ := P 1 + P 3 + P 5
def P_D : ℚ := P 3 + P 4 + P 5 + P 6

-- Prove that the event with the largest measure is event A
theorem largest_measure_event : (max (max (f P_A) (f P_B)) (max (f P_C) (f P_D))) = f P_A :=
by sorry

end largest_measure_event_l130_130697


namespace find_people_got_off_at_first_stop_l130_130040

def total_seats (rows : ℕ) (seats_per_row : ℕ) : ℕ :=
  rows * seats_per_row

def occupied_seats (total_seats : ℕ) (initial_people : ℕ) : ℕ :=
  total_seats - initial_people

def occupied_seats_after_first_stop (initial_people : ℕ) (boarded_first_stop : ℕ) (got_off_first_stop : ℕ) : ℕ :=
  (initial_people + boarded_first_stop) - got_off_first_stop

def occupied_seats_after_second_stop (occupied_after_first_stop : ℕ) (boarded_second_stop : ℕ) (got_off_second_stop : ℕ) : ℕ :=
  (occupied_after_first_stop + boarded_second_stop) - got_off_second_stop

theorem find_people_got_off_at_first_stop
  (initial_people : ℕ := 16)
  (boarded_first_stop : ℕ := 15)
  (total_rows : ℕ := 23)
  (seats_per_row : ℕ := 4)
  (boarded_second_stop : ℕ := 17)
  (got_off_second_stop : ℕ := 10)
  (empty_seats_after_second_stop : ℕ := 57)
  : ∃ x, (occupied_seats_after_second_stop (occupied_seats_after_first_stop initial_people boarded_first_stop x) boarded_second_stop got_off_second_stop) = total_seats total_rows seats_per_row - empty_seats_after_second_stop :=
by
  sorry

end find_people_got_off_at_first_stop_l130_130040


namespace number_of_students_in_section_B_l130_130976

theorem number_of_students_in_section_B (
    avg_weight_A : ℝ := 50,
    num_students_A : ℕ := 40,
    avg_weight_B : ℝ := 40,
    avg_weight_class : ℝ := 46.67) 
    (x : ℝ) :
  ((40 * 50 + x * 40) / (40 + x) = 46.67) → x = 20 := 
by 
  sorry

end number_of_students_in_section_B_l130_130976


namespace simplify_expression_eq_l130_130921

noncomputable def simplified_expression (b : ℝ) : ℝ :=
  (Real.rpow (Real.rpow (b ^ 16) (1 / 8)) (1 / 4)) ^ 3 *
  (Real.rpow (Real.rpow (b ^ 16) (1 / 4)) (1 / 8)) ^ 3

theorem simplify_expression_eq (b : ℝ) (hb : 0 < b) :
  simplified_expression b = b ^ 3 :=
by sorry

end simplify_expression_eq_l130_130921


namespace tyler_total_saltwater_animals_l130_130992

theorem tyler_total_saltwater_animals (num_aquariums : ℕ) (animals_per_aquarium : ℕ) (total_animals : ℕ) : 
  num_aquariums = 8 → animals_per_aquarium = 64 → total_animals = num_aquariums * animals_per_aquarium → total_animals = 512 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3
  sorry

end tyler_total_saltwater_animals_l130_130992


namespace eccentricity_of_hyperbola_l130_130434

-- Define the hyperbola and its properties
def hyperbola (a b x y : ℝ) (h₁ : a > 0) (h₂ : b > 0) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the left and right foci of the hyperbola
def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (a^2 + b^2) in
  ((-c, 0), (c, 0))

-- Define the distances from a point P on the hyperbola to the foci
def distances (a b x y : ℝ) (h : hyperbola a b x y sorry sorry) : (ℝ × ℝ) :=
  let ((l₁x, l₁y), (r₁x, r₁y)) := foci a b in
  let PF₁ := Real.sqrt ((x - l₁x)^2 + (y - l₁y)^2) in
  let PF₂ := Real.sqrt ((x - r₁x)^2 + (y - r₁y)^2) in
  (PF₁, PF₂)

-- Condition given in problem
def given_condition (a PF₁ PF₂ : ℝ) : Prop := 3 * PF₁ = 4 * PF₂

-- Define the eccentricity
def eccentricity (c a : ℝ) : ℝ := c / a

-- Prove the final statement: the eccentricity is 5 given the conditions
theorem eccentricity_of_hyperbola (a b x y : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (hx : hyperbola a b x y h₁ h₂)
  (h_distances : distances a b x y hx = (8 * a, 6 * a)) :
  let c := Real.sqrt (a^2 + b^2) in
  eccentricity c a = 5 := by
  -- Assuming the nontrivial calculation and verification steps
  sorry

end eccentricity_of_hyperbola_l130_130434


namespace eight_div_repeating_three_l130_130154

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l130_130154


namespace area_of_quadrilateral_is_7_5_l130_130223

open Real

theorem area_of_quadrilateral_is_7_5 (A B C D : ℝ × ℝ) (hA : A = (2, 1)) (hB : B = (5, 3)) (hC : C = (7, 1)) (hD : D = (4, 6)) :
  let area := (1 / 2) * |(A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * A.2) -
                         (A.2 * B.1 + B.2 * C.1 + C.2 * D.1 + D.2 * A.1)|
  in area = 7.5 :=
by
  sorry

end area_of_quadrilateral_is_7_5_l130_130223


namespace rationalize_denominator_correct_l130_130503

noncomputable def rationalize_denominator : Prop :=
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l130_130503


namespace composite_probability_l130_130036

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

theorem composite_probability :
  let total_numbers := 50 in
  let composite_count := total_numbers - 1 - 15 in -- 1 is neither prime nor composite; 15 primes under 50.
  (composite_count : ℝ) / total_numbers = 0.68 :=
by
  sorry

end composite_probability_l130_130036


namespace decreasing_interval_iff_a_range_l130_130781

def f (x a : ℝ) : ℝ := log (x ^ 2 - 2 * a * x + 3)

theorem decreasing_interval_iff_a_range (a : ℝ) :
  (∀ x1 x2, (1 / 2 < x1) → (x1 < 1) → (1 / 2 < x2) → (x2 < 1) → x1 < x2 → f x2 a < f x1 a)
  ↔ (1 ≤ a ∧ a ≤ 2) := 
sorry

end decreasing_interval_iff_a_range_l130_130781


namespace probability_at_least_6_heads_in_9_flips_l130_130271

def fair_coin_flips := 9
def total_outcomes := 512

/-- The probability of obtaining at least 'k' consecutive heads in 'n' flips of a fair coin. -/
def at_least_k_consecutive_heads_probability (k n : ℕ) : ℚ :=
∑ i in ((list.range (n - k + 1)).map (λ start_pos, 
  let end_pos := start_pos + k - 1 in 
  ((2 : ℚ)^(start_pos) + (n - end_pos - 1)))) 
  , (1 / (2^n : ℚ))

theorem probability_at_least_6_heads_in_9_flips :
  at_least_k_consecutive_heads_probability 6 9 = 49 / 512 := sorry

end probability_at_least_6_heads_in_9_flips_l130_130271


namespace division_by_repeating_decimal_l130_130206

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l130_130206


namespace min_translated_point_l130_130023

theorem min_translated_point :
  let original_graph := λ x : ℝ, 2 * |x| - 4
  let translation_x := 4
  let translation_y := 2
  let min_point_original := (0, original_graph 0)
  let min_point_translated := (min_point_original.1 + translation_x, min_point_original.2 + translation_y)
  min_point_translated = (4, -2) :=
by
  sorry

end min_translated_point_l130_130023


namespace even_function_has_a_equal_2_l130_130821

noncomputable def f (a x : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_has_a_equal_2 (a : ℝ) :
  (∀ x : ℝ, f a (-x) = f a x) → a = 2 :=
sorry

end even_function_has_a_equal_2_l130_130821


namespace vacant_student_seats_given_to_parents_l130_130545

-- Definitions of the conditions
def total_seats : Nat := 150

def awardees_seats : Nat := 15
def admins_teachers_seats : Nat := 45
def students_seats : Nat := 60
def parents_seats : Nat := 30

def awardees_occupied_seats : Nat := 15
def admins_teachers_occupied_seats : Nat := 9 * admins_teachers_seats / 10
def students_occupied_seats : Nat := 4 * students_seats / 5
def parents_occupied_seats : Nat := 7 * parents_seats / 10

-- Vacant seats calculation
def awardees_vacant_seats : Nat := awardees_seats - awardees_occupied_seats
def admins_teachers_vacant_seats : Nat := admins_teachers_seats - admins_teachers_occupied_seats
def students_vacant_seats : Nat := students_seats - students_occupied_seats
def parents_vacant_seats : Nat := parents_seats - parents_occupied_seats

-- Theorem statement
theorem vacant_student_seats_given_to_parents :
  students_vacant_seats = 12 →
  parents_vacant_seats = 9 →
  9 ≤ students_vacant_seats ∧ 9 ≤ parents_vacant_seats :=
by
  sorry

end vacant_student_seats_given_to_parents_l130_130545


namespace bridge_length_approx_l130_130290

def length_train : ℝ := 100
def time_seconds : ℝ := 14.284571519992687
def speed_kmph : ℝ := 63
def speed_mps : ℝ := 17.5 -- 63 kmph converted to m/s

def distance_train : ℝ := speed_mps * time_seconds
def length_bridge : ℝ := distance_train - length_train

theorem bridge_length_approx :
  length_bridge ≈ 149.98 :=
by 
  sorry

end bridge_length_approx_l130_130290


namespace odd_function_property_f_nonnegative_expression_l130_130766

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then -x * real.log (2 - x) else -x * real.log (2 + x)

theorem odd_function_property (x : ℝ) (h : x < 0) : f (-x) = -f x := by
  sorry

theorem f_nonnegative_expression {x : ℝ} (h : 0 ≤ x) : f x = -x * real.log (2 + x) :=
  by sorry

end odd_function_property_f_nonnegative_expression_l130_130766


namespace ratio_of_areas_l130_130602

-- Definitions and Preparations
structure Circle := (radius : ℝ)
structure Point := (x : ℝ) (y : ℝ)
structure Triangle := (A B C : Point)

-- We assume there are functions that return the areas of the triangles formed by the three points.
def area (T : Triangle) : ℝ := sorry

-- Given Data
variable (ω1 ω2 : Circle)
variable (a b c : Line)
variable (A1 B1 C1 A2 B2 C2 : Point)

-- Non-overlapping circles with their tangents touching at the specified points
noncomputable def ω1 := Circle.mk r1
noncomputable def ω2 := Circle.mk r2

-- Triangle created by tangent points on ω1 and ω2 respectively
noncomputable def T1 := Triangle.mk A1 B1 C1
noncomputable def T2 := Triangle.mk A2 B2 C2

-- Theorem statement
theorem ratio_of_areas (h1 : Triangle) (h2 : Triangle) : 
  area h1 / area h2 = ω1.radius / ω2.radius := 
sorry

end ratio_of_areas_l130_130602


namespace coprime_count_eq_12_l130_130700

def positive_integers_less_than_30_coprime_to_20 (n : ℕ) : Prop :=
  n < 30 ∧ Nat.gcd 20 n = 1

theorem coprime_count_eq_12 :
  ({a : ℕ | positive_integers_less_than_30_coprime_to_20 a}.card = 12) :=
by
  sorry

end coprime_count_eq_12_l130_130700


namespace find_a6_l130_130772

-- Define the arithmetic sequence properties
variables (a : ℕ → ℤ) (d : ℤ)

-- Define the initial conditions
axiom h1 : a 4 = 1
axiom h2 : a 7 = 16
axiom h_arith_seq : ∀ n, a (n + 1) - a n = d

-- Statement to prove
theorem find_a6 : a 6 = 11 :=
by
  sorry

end find_a6_l130_130772


namespace my_op_eq_l130_130407

-- Define the custom operation
def my_op (m n : ℝ) : ℝ := m * n * (m - n)

-- State the theorem
theorem my_op_eq :
  ∀ (a b : ℝ), my_op (a + b) a = a^2 * b + a * b^2 :=
by intros a b; sorry

end my_op_eq_l130_130407


namespace survival_probability_l130_130601

theorem survival_probability : 
  let v1 := (486 : ℚ) / 630
  let v2 := (540 : ℚ) / 675 in
  (v1 * v2 = 108 / 175) ∧
  (1 - v1 = 8 / 35) ∧
  (1 - v2 = 1 / 5) ∧
  ((1 - v1) * (1 - v2) = 8 / 175) ∧
  ((1 - v1) * v2 = 32 / 175) ∧
  (v1 * (1 - v2) = 27 / 175) := 
by {
  unfold v1 v2,
  split, { sorry },
  split, { sorry },
  split, { sorry },
  split, { sorry },
  split, { sorry },
  sorry
}

end survival_probability_l130_130601


namespace product_inequality_l130_130763

theorem product_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∏ k in finset.range 2015, (a ^ (k + 1) + b ^ (k + 1))) ≥ (∏ m in finset.range 403, (a ^ (m + 807) + b ^ (m + 807)) ^ 5) :=
by
  sorry

end product_inequality_l130_130763


namespace divide_by_repeating_decimal_l130_130097

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l130_130097


namespace probability_B_does_not_lose_l130_130558

def prob_A_wins : ℝ := 0.3
def prob_draw : ℝ := 0.5

-- Theorem: the probability that B does not lose is 70%.
theorem probability_B_does_not_lose : prob_A_wins + prob_draw ≤ 1 → 1 - prob_A_wins - (1 - prob_draw - prob_A_wins) = 0.7 := by
  sorry

end probability_B_does_not_lose_l130_130558


namespace smallest_p_l130_130897

theorem smallest_p 
  (p q : ℕ) 
  (h1 : (5 : ℚ) / 8 < p / (q : ℚ) ∧ p / (q : ℚ) < 7 / 8)
  (h2 : p + q = 2005) : p = 772 :=
sorry

end smallest_p_l130_130897


namespace probability_no_disease_after_continuous_smoke_l130_130923

noncomputable section

namespace SmokingProblem

-- Definitions according to the conditions
def P_A : ℝ := 0.98   -- Probability of not inducing disease after 5 cigarettes
def P_B : ℝ := 0.84   -- Probability of not inducing disease after 10 cigarettes

-- The theorem we need to prove
theorem probability_no_disease_after_continuous_smoke :
  P_B / P_A = 6 / 7 := by
  sorry

end SmokingProblem

end probability_no_disease_after_continuous_smoke_l130_130923


namespace Eight_div_by_repeating_decimal_0_3_l130_130163

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l130_130163


namespace proof_incorrect_solution1_proof_expression_part2_l130_130506

-- Define the incorrect solution
def incorrect_solution1 := (1 / 60) / ((1 / 3) - (1 / 4) + (1 / 12)) = (1 / 60) / (1 / 3) - (1 / 60) / (1 / 4) + (1 / 60) / (1 / 12)

-- The actual correct answer for the given expression
def correct_expression := (1 / 60) / ((1 / 3) - (1 / 4) + (1 / 12)) = 1 / 10

-- Prove that Solution 1 is incorrect
theorem proof_incorrect_solution1 : ¬ incorrect_solution1 :=
by
  intros h
  rw [← correct_expression] at h
  contradiction

-- Define the mathematical expression from part 2
def expression_part2 := - (1 / 42) / ((3 / 7) - (5 / 14) + (2 / 3) - (1 / 6))

-- The correct final answer for the given expression in part 2
def correct_answer2 := expression_part2 = - (1 / 24)

-- Prove that the calculated answer for part 2 is correct
theorem proof_expression_part2 : correct_answer2 :=
by
  rw [expression_part2]
  sorry

end proof_incorrect_solution1_proof_expression_part2_l130_130506


namespace seq_no_squares_l130_130537

-- Define the sequence
def seq (n : ℕ) : ℕ :=
  if n = 0 then 1 else 1 + (Finset.range n).sum (λ k, (seq k)^2)

-- Main theorem: for all n > 1, seq(n) is not a perfect square
theorem seq_no_squares (n : ℕ) (h : n > 0) : ∀ k : ℕ, seq n ≠ k^2 := by
  induction n with m hm
  case zero => 
    -- n = 0, trivial case
    intro k
    simp only [seq, false_or, if_true_eq_iff, nat.not_succ_le_zero, nat.one_ne_zero] with arith
  case succ m =>
    intro k
    -- Prove the main statement here, but for now we use sorry
    sorry

end seq_no_squares_l130_130537


namespace hyperbola_eccentricity_l130_130433

theorem hyperbola_eccentricity 
  (center_origin : ∃ x y : ℝ, x = 0 ∧ y = 0)
  (focus_on_x_axis : ∃ c : ℝ, c > 0)
  (asymptote_eq : ∀ x y : ℝ, (4 + 3 * y = 0) ∨ (4 - 3 * y = 0)) :
  ∃ e : ℝ, e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l130_130433


namespace smallest_number_of_blocks_needed_l130_130610

/--
Given:
  A wall with the following properties:
  1. The wall is 100 feet long and 7 feet high.
  2. Blocks used are 1 foot high and either 1 foot or 2 feet long.
  3. Blocks cannot be cut.
  4. Vertical joins in the blocks must be staggered.
  5. The wall must be even on the ends.
Prove:
  The smallest number of blocks needed to build this wall is 353.
-/
theorem smallest_number_of_blocks_needed :
  let length := 100
  let height := 7
  let block_height := 1
  (∀ b : ℕ, b = 1 ∨ b = 2) →
  ∃ (blocks_needed : ℕ), blocks_needed = 353 :=
by sorry

end smallest_number_of_blocks_needed_l130_130610


namespace find_y_l130_130320

theorem find_y (y : ℝ) (h : 10^(2*y) * 100^y = 1000^3 * 10^y) : y = 3 :=
sorry

end find_y_l130_130320


namespace triangle_largest_angle_cosine_l130_130414

theorem triangle_largest_angle_cosine (a b c : ℝ) (h₁ : a = 3) (h₂ : b = sqrt 2) (h₃ : c = 2 * sqrt 2) :
  ∃ A B C : ℝ, A = acos ( (b^2 + c^2 - a^2) / (2 * b * c) ) ∧
  (a > b ∧ a > c ∧ cos A =  1/8) :=
by
  use acos ((b ^ 2 + c ^ 2 - a ^ 2) / (2 * b * c))
  split
  · sorry

end triangle_largest_angle_cosine_l130_130414


namespace smallest_period_f_min_value_f_l130_130388

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := cos(x)^4 - 2 * sin(x) * cos(x) - sin(x)^4

-- Smallest positive period of the function f
theorem smallest_period_f : (∀ x, f (x + π) = f x) ∧ (∀ T, T > 0 ∧ ∀ x, f (x + T) = f x → T ≥ π) := by
  sorry

-- Minimum value of the function f over [0, π/2]
theorem min_value_f : (minValue : ℝ) ∧ (minValue = -sqrt 2) ∧ (∀ x ∈ set.Icc 0 (π / 2), f x ≥ minValue) ∧ (f (3 * π / 8) = minValue) := by
  sorry

end smallest_period_f_min_value_f_l130_130388


namespace hyperbola_equation_l130_130765

variables (a b c : ℝ)

def hyperbola_center_at_origin (x y : ℝ) : Prop :=
  (a^2 - b^2 = 27) ∧ (c = 9) ∧ (a^2 = 27) ∧ (b^2 = 54) ∧ (c / a = sqrt 3)

theorem hyperbola_equation (x y : ℝ) :
  hyperbola_center_at_origin x y →
  x^2 / 27 - y^2 / 54 = 1 :=
by
  intro H
  sorry

end hyperbola_equation_l130_130765


namespace new_price_of_sugar_l130_130028

theorem new_price_of_sugar (C : ℝ) (H : 10 * C = P * (0.7692307692307693 * C)) : P = 13 := by
  sorry

end new_price_of_sugar_l130_130028


namespace hyperbola_focus_eq_parabola_focus_l130_130376

theorem hyperbola_focus_eq_parabola_focus 
    (a : ℝ) (h : a > 0)
    (focus_parabola : (Real.pow 2 2 = a^2 + 3)) :
  a = 1 :=
by
  sorry

end hyperbola_focus_eq_parabola_focus_l130_130376


namespace eight_div_repeating_three_l130_130187

theorem eight_div_repeating_three : 
  ∀ (x : ℝ), x = 1 / 3 → 8 / x = 24 :=
by
  intro x h
  rw h
  norm_num
  done

end eight_div_repeating_three_l130_130187


namespace comparison_of_prices_l130_130774

theorem comparison_of_prices:
  ∀ (x y : ℝ), (6 * x + 3 * y > 24) → (4 * x + 5 * y < 22) → (2 * x > 3 * y) :=
by
  intros x y h1 h2
  sorry

end comparison_of_prices_l130_130774


namespace scrabble_score_l130_130876

-- Definitions derived from conditions
def value_first_and_third : ℕ := 1
def value_middle : ℕ := 8
def multiplier : ℕ := 3

-- Prove the total points earned by Jeremy
theorem scrabble_score : (value_first_and_third * 2 + value_middle) * multiplier = 30 :=
by
  sorry

end scrabble_score_l130_130876


namespace no_rain_five_days_l130_130960

-- Define the problem conditions and the required result.
def prob_rain := (2 / 3)
def prob_no_rain := (1 - prob_rain)
def prob_no_rain_five_days := prob_no_rain^5

theorem no_rain_five_days : 
  prob_no_rain_five_days = (1 / 243) :=
by
  sorry

end no_rain_five_days_l130_130960


namespace volume_of_intersection_l130_130448

noncomputable def symmetric_polyhedron_intersection_volume (e π sqrt5 : ℝ) : ℝ :=
  if h : 0 ≤ e ∧ e ≤ 6 ∧ 0 ≤ π ∧ π ≤ 6 ∧ 0 ≤ sqrt5 ∧ sqrt5 ≤ 6 then
    6^3 - 6^2
  else
    0

theorem volume_of_intersection (e π sqrt5 : ℝ) (h : 0 ≤ e ∧ e ≤ 6 ∧ 0 ≤ π ∧ π ≤ 6 ∧ 0 ≤ sqrt5 ∧ sqrt5 ≤ 6) :
  symmetric_polyhedron_intersection_volume e π sqrt5 = 180 :=
by {
  unfold symmetric_polyhedron_intersection_volume,
  rw [if_pos h],
  norm_num
}

end volume_of_intersection_l130_130448


namespace eight_div_repeating_three_l130_130148

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l130_130148


namespace sum_of_distinct_prime_factors_of_396_l130_130568

theorem sum_of_distinct_prime_factors_of_396 : 
  (∑ p in (Nat.factors 396).toFinset, p) = 16 := 
by
  sorry

end sum_of_distinct_prime_factors_of_396_l130_130568


namespace binomial_identity_l130_130058

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the summation of k * C(n, k) for k from 1 to n
def sum_binomial_mul_k (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), k * binomial n k

-- The theorem statement
theorem binomial_identity (n : ℕ) (hn : n > 0) :
  sum_binomial_mul_k n = n * 2^(n-1) :=
by
  sorry

end binomial_identity_l130_130058


namespace instantaneous_velocity_at_3_l130_130534

def displacement (t : ℝ) : ℝ :=
  1.5 * t - 0.1 * t^2

theorem instantaneous_velocity_at_3 (t := 3) :
  let v := (deriv displacement) t
  in v = 0.9 :=
by
  sorry

end instantaneous_velocity_at_3_l130_130534


namespace division_of_repeating_decimal_l130_130076

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l130_130076


namespace division_of_decimal_l130_130106

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l130_130106


namespace competition_scores_order_l130_130420

theorem competition_scores_order (A B C D : ℕ) (h1 : A + B = C + D) (h2 : C + A > D + B) (h3 : B > A + D) : (B > A) ∧ (A > C) ∧ (C > D) := 
by 
  sorry

end competition_scores_order_l130_130420


namespace wheel_speed_l130_130455

def circumference_feet : ℝ := 11
def wheel_circumference_miles : ℝ := circumference_feet / 5280
def original_speed (r t : ℝ) : Prop := r * t = wheel_circumference_miles * 3600
def new_speed (r t : ℝ) : Prop :=
  (r + 5) * (t - 1 / 14400) = wheel_circumference_miles * 3600

theorem wheel_speed
  (r t : ℝ)
  (h1 : original_speed r t)
  (h2 : new_speed r t) : 
  r = 10 :=
sorry

end wheel_speed_l130_130455


namespace evaluate_modulus_l130_130706

def omega : ℂ := 8 + complex.I

theorem evaluate_modulus : complex.abs (omega^2 - 4 * omega + 13) = 4 * real.sqrt 130 :=
by
  sorry

end evaluate_modulus_l130_130706


namespace composite_prob_from_50_nat_numbers_l130_130030

theorem composite_prob_from_50_nat_numbers : 
  (∃ n, 1 ≤ n ∧ n ≤ 50 ∧ ∑ i in finset.range 50, if (¬is_prime (i + 1) ∧ (i + 1) ≠ 1) then 1 else 0 = 34 ∧ (34: ℝ) / 50 = 0.68) :=
by {
  sorry,
}

end composite_prob_from_50_nat_numbers_l130_130030


namespace smallest_prime_solution_l130_130339
open Nat

noncomputable def smallest_prime_lcm_minus_1 : ℕ :=
  let lcm_values := List.lcm [823, 618, 3648, 60, 3917, 4203, 1543, 2971]
  lcm_values - 1

theorem smallest_prime_solution : Prime (smallest_prime_lcm_minus_1) ∧ 
  ∀ x ∈ [823, 618, 3648, 60, 3917, 4203, 1543, 2971], (smallest_prime_lcm_minus_1 + 1) % x = 0 := by
  sorry

end smallest_prime_solution_l130_130339


namespace g_neither_even_nor_odd_l130_130701

def g (x : ℝ) := log (x - sqrt (1 + x^2))

theorem g_neither_even_nor_odd : 
  ¬ (∀ x : ℝ, g (-x) = g x) ∧ ¬ (∀ x : ℝ, g (-x) = -g x) :=
sorry

end g_neither_even_nor_odd_l130_130701


namespace european_stamps_cost_l130_130605

def prices : String → ℕ 
| "Italy"   => 7
| "Japan"   => 7
| "Germany" => 5
| "China"   => 5
| _ => 0

def stamps_1950s : String → ℕ 
| "Italy"   => 5
| "Germany" => 8
| "China"   => 10
| "Japan"   => 6
| _ => 0

def stamps_1960s : String → ℕ 
| "Italy"   => 9
| "Germany" => 12
| "China"   => 5
| "Japan"   => 10
| _ => 0

def total_cost (stamps : String → ℕ) (price : String → ℕ) : ℕ :=
  (stamps "Italy" * price "Italy" +
   stamps "Germany" * price "Germany") 

theorem european_stamps_cost : total_cost stamps_1950s prices + total_cost stamps_1960s prices = 198 :=
by
  sorry

end european_stamps_cost_l130_130605


namespace constant_term_expansion_l130_130451

def f (a x : ℝ) : ℝ :=
if x > 0 then log x / log 10 else x + ∫ t in 0..a, 3 * t^2

theorem constant_term_expansion (a : ℝ) (h : f a (f a 1) = 1) : 
  let a := 1 in 
  let k := a + 5 in 
  let expand := (4^x - 2^(-x))^k in
  true := by
  let a := 1;
  have k := a + 5;
  have expand_eq_const := (binomial_expansion_term _ _ k _ 4 (-2) _ _ 6 4 (-1)^4);
  reserve 4;
  exact True.intro

end constant_term_expansion_l130_130451


namespace min_rooks_to_color_grid_l130_130665

theorem min_rooks_to_color_grid (n : ℕ) (numbering : fin n.succ × fin n.succ → ℕ)
  (h_numbering : ∀ i j, 1 ≤ numbering (i, j) ∧ numbering (i, j) ≤ n * n)
  (h_distinct : ∀ i₁ j₁ i₂ j₂, numbering (i₁, j₁) = numbering (i₂, j₂) → (i₁ = i₂ ∧ j₁ = j₂)) :
  ∃ (rooks : fin n.succ → fin n.succ), ∀ i j, ∃ k, rooks k = (i, j) ∨
    (∃ i' j', (rooks k = (i', j') ∧ numbering (i', j') < numbering (i, j)) ∧
      (i' = i ∨ j' = j ∧ (∀ i'' j'', numbering (i'', j'') = numbering (i, j) → ((i'' = i ∧ j'' ≠ j) ∨ (i'' ≠ i ∧ j'' = j))))) ∧
  ∀ m, (∀ i j, ∃ k, rooks k = (i, j) ∨
    (∃ i' j', (rooks k = (i', j') ∧ numbering (i', j') < numbering (i, j)) ∧
      (i' = i ∨ j' = j ∧ (∀ i'' j'', numbering (i'', j'') = numbering (i, j) → ((i'' = i ∧ j'' ≠ j) ∨ (i'' ≠ i ∧ j'' = j))))) → m ≥ n :=
sorry

end min_rooks_to_color_grid_l130_130665


namespace jonah_needs_15_pounds_of_meat_l130_130879

theorem jonah_needs_15_pounds_of_meat
  (meat_per_10_burgers : ℝ)
  (meat_used : meat_per_10_burgers = 5)
  (burgers_made : 10) :
  meat_needed_for_30 : ℝ := by
    let meat_per_burger := meat_per_10_burgers / burgers_made
    let meat_needed_for_30 := meat_per_burger * 30
    have h : meat_needed_for_30 = 15 := by sorry
    exact h

end jonah_needs_15_pounds_of_meat_l130_130879


namespace exists_rationals_leq_l130_130308

theorem exists_rationals_leq (f : ℚ → ℤ) : ∃ a b : ℚ, (f a + f b) / 2 ≤ f (a + b) / 2 :=
by
  sorry

end exists_rationals_leq_l130_130308


namespace Eight_div_by_repeating_decimal_0_3_l130_130162

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l130_130162


namespace findTwoHeaviestStonesWith35Weighings_l130_130065

-- Define the problem with conditions
def canFindTwoHeaviestStones (stones : Fin 32 → ℝ) (weighings : ℕ) : Prop :=
  ∀ (balanceScale : (Fin 32 × Fin 32) → Bool), weighings ≤ 35 → 
  ∃ (heaviest : Fin 32) (secondHeaviest : Fin 32), 
  (heaviest ≠ secondHeaviest) ∧ 
  (∀ i : Fin 32, stones heaviest ≥ stones i) ∧ 
  (∀ j : Fin 32, j ≠ heaviest → stones secondHeaviest ≥ stones j)

-- Formally state the theorem
theorem findTwoHeaviestStonesWith35Weighings (stones : Fin 32 → ℝ) :
  canFindTwoHeaviestStones stones 35 :=
sorry -- Proof is omitted

end findTwoHeaviestStonesWith35Weighings_l130_130065


namespace ratio_EG_GD_l130_130429

theorem ratio_EG_GD (a EG GD : ℝ)
  (h1 : EG = 4 * GD)
  (gcd_1 : Int.gcd 4 1 = 1) :
  4 + 1 = 5 := by
  sorry

end ratio_EG_GD_l130_130429


namespace eight_div_repeating_three_l130_130082

theorem eight_div_repeating_three : 8 / (1 / 3) = 24 :=
by
  have q : ℝ := 1 / 3
  calc
    8 / q = 8 * 3 : by simp [q]  -- since q = 1 / 3
        ... = 24 : by ring

end eight_div_repeating_three_l130_130082


namespace train_speed_is_180_kmh_l130_130658

-- Defining the conditions
def train_length : ℕ := 1500  -- meters
def platform_length : ℕ := 1500  -- meters
def crossing_time : ℕ := 1  -- minute

-- Function to compute the speed in km/hr
def speed_in_km_per_hr (length : ℕ) (time : ℕ) : ℕ :=
  let distance := length + length
  let speed_m_per_min := distance / time
  let speed_km_per_hr := speed_m_per_min * 60 / 1000
  speed_km_per_hr

-- The main theorem we need to prove
theorem train_speed_is_180_kmh :
  speed_in_km_per_hr train_length crossing_time = 180 :=
by
  sorry

end train_speed_is_180_kmh_l130_130658


namespace expected_number_of_cards_turned_up_l130_130623

noncomputable def expected_cards_to_second_ace (n : ℕ) : ℝ :=
(n + 1) / 2

theorem expected_number_of_cards_turned_up (n : ℕ) (h : n ≥ 3) :
  let E := expected_cards_to_second_ace n
  in E = (n + 1) / 2 := 
sorry

end expected_number_of_cards_turned_up_l130_130623


namespace division_of_repeating_decimal_l130_130068

theorem division_of_repeating_decimal :
  let q : ℝ := 0.3333 -- This should be interpreted as q = 0.\overline{3}
  in 8 / q = 24 :=
by
  let q : ℝ := 1 / 3 -- equivalent to 0.\overline{3}
  show 8 / q = 24
  sorry

end division_of_repeating_decimal_l130_130068


namespace shortest_distance_ln_x_to_line_theorem_l130_130043

noncomputable def shortest_distance_ln_x_to_line : Prop :=
  let curve := fun x => Math.log x
  let point_on_curve := (1:ℝ, Math.log (1:ℝ))
  let line := fun x => x + 1
  let distance := fun (p1 p2 : ℝ × ℝ) => Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  distance point_on_curve (1, 2) = Real.sqrt 2

-- The proposition stating that the shortest distance from a point on the logarithmic curve to the line is sqrt 2.
def shortest_distance_proposition : Prop :=
  ∃ (x : ℝ), x > 0 ∧ (distance (x, Math.log x) (x, x + 1)) = Real.sqrt 2

-- Lean statement to prove the proposition
theorem shortest_distance_ln_x_to_line_theorem : shortest_distance_proposition :=
sorry

end shortest_distance_ln_x_to_line_theorem_l130_130043


namespace minimum_employees_l130_130264

theorem minimum_employees (forest_conservation wildlife_conservation both_conservation : ℕ) (h₁ : forest_conservation = 100) (h₂ : wildlife_conservation = 90) (h₃ : both_conservation = 40) :
  (forest_conservation + wildlife_conservation - both_conservation) = 150 :=
by
  -- Given conditions
  rw [h₁, h₂, h₃]
  -- Proof to follow
  sorry

end minimum_employees_l130_130264


namespace largest_five_digit_integer_with_digit_product_proof_l130_130560

noncomputable def largest_five_digit_integer_with_digit_product : ℕ :=
  98752

theorem largest_five_digit_integer_with_digit_product_proof :
  ∃ n : ℕ, (n >= 10000) ∧ (n < 100000) ∧ 
           (∃ (digits : list ℕ), (n = digits.foldl (λ acc d, acc * 10 + d) 0) ∧
           (digits.prod = (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) ∧
           (n = 98752)) :=
by {
    let answer := 98752,
    use answer,
    split, { norm_num },
    split, { norm_num },
    use [9, 8, 7, 5, 2],
    split, { refl },
    split, {
        rw [list.prod_cons, list.prod_cons, list.prod_cons, list.prod_cons, list.prod_nil],
        norm_num,
    },
    refl,
}

end largest_five_digit_integer_with_digit_product_proof_l130_130560


namespace z6_distance_is_291_l130_130314

-- Define the sequence of complex numbers.
def complex_sequence : ℕ → ℂ
| 0       := 1
| (n + 1) := complex_sequence n ^ 2 - 1 + complex.I

-- Define the 6th term in the series.
def z6 := complex_sequence 5

-- The distance from the origin for z_6.
def distance_from_origin (z : ℂ) : ℝ := abs z

-- Prove the distance from the origin for z_6 is 291.
theorem z6_distance_is_291 :
  distance_from_origin z6 = 291 := by
  sorry

end z6_distance_is_291_l130_130314


namespace sum_of_radii_eq_FA_l130_130541

variables {S F A P Q : Point}
-- Given conditions
variables (triangle_SFA_right_angle : ∃ F : Point, ∠SFA = 90)
variables (P_on_SF_between_S_and_F : ∃ P : Point, line_on_P : P ∈ SF ∧ S ∈ SF ∧ F ∈ SF ∧ S-P-F)
variables (Q_on_SF_with_F_midpoint_PQ : ∃ Q : Point, line_on_Q : Q ∈ SF ∧ F ∈ SF ∧ F ∈ midpoint P Q)

-- Define circles' radii
variables (r1 r2 : ℝ)

-- Circle properties
variables (circle_k1_inc : incircle T_1 k_1)
variables (circle_k2_ex : excircle T_2 k_2)

-- Main theorem
theorem sum_of_radii_eq_FA : 
  r1 + r2 = dist F A := sorry

end sum_of_radii_eq_FA_l130_130541


namespace Bob_age_is_47_l130_130682

variable (Bob_age Alice_age : ℝ)

def equations_holds : Prop := 
  Bob_age = 3 * Alice_age - 20 ∧ Bob_age + Alice_age = 70

theorem Bob_age_is_47.5 (h: equations_holds Bob_age Alice_age) : Bob_age = 47.5 := 
by sorry

end Bob_age_is_47_l130_130682


namespace shortest_tree_height_l130_130550

theorem shortest_tree_height :
  (tallest_tree_height = 150) →
  (middle_tree_height = (2 / 3) * tallest_tree_height) →
  (shortest_tree_height = (1 / 2) * middle_tree_height) →
  shortest_tree_height = 50 :=
by
  intros h1 h2 h3
  sorry

end shortest_tree_height_l130_130550


namespace rectangle_circle_circumference_l130_130642

theorem rectangle_circle_circumference
  (a b : ℝ)
  (h1 : a = 9)
  (h2 : b = 12)
  (h3 : a^2 + b^2 = 225) 
  : ∃ (C : ℝ), C = 15 * Real.pi :=
by 
  -- Prove the diagonal of the rectangle which is inscribed in a circle
  have d := Mathlib.sqrt (a^2 + b^2),
  have d_eq_15 : d = 15, 
  { 
    rw [h1, h2],
    calc sqrt (9^2 + 12^2) = sqrt 225 : by sorry -- skipping the actual proof calculation.
    ... = 15 : by sorry -- skipping the actual proof calculation.
  },
  
  -- circumference C = π * d 
  use 15 * Real.pi,
  calc 15 * Real.pi = C : by sorry

end rectangle_circle_circumference_l130_130642


namespace incorrect_statement_C_l130_130838

variable (x : ℝ) (b : ℝ)

noncomputable def y : ℝ := log b (x^2 + 1)

theorem incorrect_statement_C (hb : b > 1) : ¬ (y = log b 2) := 
sorry

end incorrect_statement_C_l130_130838


namespace distribution_of_balls_l130_130819

theorem distribution_of_balls (n k : ℕ) (h_n : n = 7) (h_k : k = 3) : (nat.choose (n + k - 1) (k - 1)) = 36 :=
by
  rw [h_n, h_k]
  show (nat.choose (7 + 3 - 1) (3 - 1)) = 36
  simp
  sorry

end distribution_of_balls_l130_130819


namespace min_degree_g_l130_130777

open Polynomial

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := sorry

-- Conditions
axiom cond1 : 5 • f + 7 • g = h
axiom cond2 : natDegree f = 10
axiom cond3 : natDegree h = 12

-- Question: Minimum degree of g
theorem min_degree_g : natDegree g = 12 :=
sorry

end min_degree_g_l130_130777


namespace triangles_similar_ratio_eg_ef_l130_130596

-- Definitions and conditions based on the problem statement
variables {A B C D E M F G : Type} [Point A] [Point B] [Point C] [Point D] [Point E] [Point M] [Point F] [Point G]
variables [circular_point A B C D] -- Assuming all points are on the same circle
variables [intersection_point E (segment A B) (segment C D)] -- E is where [A B] and [C D] intersect
variables [on_segment M B E] -- M is on segment [B E], distinct from B and E
variables [tangent_at E (circle D E M)] -- Tangent at E to the circle passing through D, E, and M
variables [intersection_point F (line B C) (tangent_at E (circle D E M))] -- F is the intersection of the tangent and line (B C)
variables [intersection_point G (line A C) (tangent_at E (circle D E M))] -- G is the intersection of the tangent and line (A C)

-- Part (a): To show triangles GCE and DBM are similar
theorem triangles_similar : similar (triangle G C E) (triangle D B M) :=
sorry

-- Part (b): To calculate the ratio EG/EF as a function of t
variables {t : Type} [ratio t (segment A M) (segment A B)] -- t = AM/AB
theorem ratio_eg_ef : (EG / EF) = t / (1 - t) :=
sorry

end triangles_similar_ratio_eg_ef_l130_130596


namespace minimal_ratio_l130_130248

noncomputable def S₁ (α : ℝ) : ℝ := 4 - 2 * Real.sqrt 2 / Real.cos α

noncomputable def S₂ (α : ℝ) : ℝ :=
  let numerator := (Real.sqrt 2 * (Real.sin α + Real.cos α) - 1) ^ 2
  numerator / (2 * Real.sin α * Real.cos α)

theorem minimal_ratio : 
  (∃ α₁ α₂, α₁ = 0 ∧ α₂ = π / 4 ∧ S₁ 0 = 4 - 2 * Real.sqrt 2 / Real.cos (0 : ℝ) ∧ S₂ (π / 4) = 1 ∧ (∀ α : ℝ, 0 ≤ α ∧ α ≤ π / 12 → S₁(α) < S₁(π / 12) ∧ (π / 12) < α ∧ α ≤ 5 * π / 12 → S₂(α) ≥ 1) →
  (S₁ α / S₂ α) ≥ 1 / 7) :=
sorry

end minimal_ratio_l130_130248


namespace no_integer_solutions_l130_130512

theorem no_integer_solutions : ¬∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := 
by
  sorry

end no_integer_solutions_l130_130512


namespace find_a_range_l130_130386

def f (a x : ℝ) : ℝ :=
  if x < 2 then abs (2^x - a) else x^2 - 3 * a * x + 2 * a^2

def has_two_zeros (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0

theorem find_a_range :
  { a : ℝ | has_two_zeros a } = { a : ℝ | 1 ≤ a ∧ a < 2 ∨ a ≥ 4 } :=
sorry

end find_a_range_l130_130386


namespace sum_two_smallest_prime_factors_of_180_eq_5_l130_130999

theorem sum_two_smallest_prime_factors_of_180_eq_5 :
  (prime_factors_of 180).sorted.head + (prime_factors_of 180).sorted.nth(1) = 5 :=
by
  sorry

end sum_two_smallest_prime_factors_of_180_eq_5_l130_130999


namespace mod_inverse_97_101_l130_130712

theorem mod_inverse_97_101 :
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 100 ∧ (97 * x ≡ 1 [MOD 101]) :=
begin
  use 25,
  split; try {norm_num},
  split; try {norm_num},
  apply nat.modeq.modeq_of_dvd,
  use (-24),
  norm_num,
end

end mod_inverse_97_101_l130_130712


namespace train_distance_difference_l130_130559

theorem train_distance_difference (t : ℝ) (D₁ D₂ : ℝ)
(h_speed1 : D₁ = 20 * t)
(h_speed2 : D₂ = 25 * t)
(h_total_dist : D₁ + D₂ = 540) :
  D₂ - D₁ = 60 :=
by {
  -- These are the conditions as stated in step c)
  sorry
}

end train_distance_difference_l130_130559


namespace exists_point_M_l130_130444

variables {A B C D E M : Point}
variables (S : Point → Point → Point → ℝ)

-- Conditions given in the problem
def convex_pentagon (A B C D E : Point) : Prop := 
  convex (polygon A B C D E) 

def area_condition := 
  S A B C = S B C D ∧ S B C D = S C D E ∧ S C D E = S D E A ∧ S D E A = S E A B

def equal_areas := 
  S M A B = S M B C ∧ S M B C = S M C D ∧ S M C D = S M D E ∧ S M D E = S M E A

-- Theorem statement
theorem exists_point_M 
  (h_convex: convex_pentagon A B C D E)
  (h_area: area_condition S A B C D E) :
  ∃ M : Point, equal_areas S M A B C D E :=
sorry

end exists_point_M_l130_130444


namespace bryden_total_money_l130_130262

def rate := 15
def face_value_of_quarter := 0.25
def num_quarters := 6
def total_money := rate * (face_value_of_quarter * num_quarters)

theorem bryden_total_money :
  total_money = 22.5 := 
sorry

end bryden_total_money_l130_130262


namespace total_area_of_forest_and_fields_l130_130973

theorem total_area_of_forest_and_fields (r p k : ℝ) (h1 : k = 12) 
  (h2 : r^2 + 4 * p^2 + 45 = 12 * k) :
  (r^2 + 4 * p^2 + 12 * k = 135) :=
by
  -- Proof goes here
  sorry

end total_area_of_forest_and_fields_l130_130973


namespace cone_height_l130_130846

theorem cone_height (a : ℝ) (h : ℝ) :
    (h^2 = a^2 - (a/2)^2) → (h = (√3 / 2) * a) :=
by
  sorry

end cone_height_l130_130846


namespace Bob_age_is_47_l130_130680

variable (Bob_age Alice_age : ℝ)

def equations_holds : Prop := 
  Bob_age = 3 * Alice_age - 20 ∧ Bob_age + Alice_age = 70

theorem Bob_age_is_47.5 (h: equations_holds Bob_age Alice_age) : Bob_age = 47.5 := 
by sorry

end Bob_age_is_47_l130_130680


namespace distance_to_Rock_Mist_Mountains_l130_130293

theorem distance_to_Rock_Mist_Mountains
  (d_Sky_Falls : ℕ) (d_Sky_Falls_eq : d_Sky_Falls = 8)
  (d_Rock_Mist : ℕ) (d_Rock_Mist_eq : d_Rock_Mist = 50 * d_Sky_Falls)
  (detour_Thunder_Pass : ℕ) (detour_Thunder_Pass_eq : detour_Thunder_Pass = 25) :
  d_Rock_Mist + detour_Thunder_Pass = 425 := by
  sorry

end distance_to_Rock_Mist_Mountains_l130_130293


namespace largest_binomial_coefficient_term_sum_of_coefficients_alternating_binomial_sum_l130_130899

-- Part (1)① 
theorem largest_binomial_coefficient_term {y : ℝ} (hy : y > 0) :
  let f (x y : ℝ) := (1 - 2 / y) ^ x in
  f 4 y = (1 - 2 / y) ^ 4 →
  ∀ m : ℝ, m = 2 →
  (1 - 2 / y) ^ 4 = ∑ i in finset.range(5), (binom 4 i) * (1 ^ (4 - i)) * ((-2 / y) ^ i) →
  let term_with_largest_binom := (binom 4 2) * (1 ^ (4 - 2)) * ((-2 / y) ^ 2) in
  term_with_largest_binom = 24 / y^2 :=
by
  sorry

-- Part (1)②
theorem sum_of_coefficients {a : ℕ → ℝ} {m : ℝ} (hm : m = 2) (h : a 1 = -12) :
  ∑ i in finset.range(1, 7), a i = 126 :=
by
  sorry

-- Part (2)
theorem alternating_binomial_sum (n : ℕ) (hn : n ≥ 1) :
  ∑ k in finset.range(1, n + 1), (-1) ^ k * k ^ 2 * (binom n k) = 0 :=
by
  sorry

end largest_binomial_coefficient_term_sum_of_coefficients_alternating_binomial_sum_l130_130899


namespace choose_3_from_10_l130_130864

-- Definitions based on the conditions
def num_people : ℕ := 10
def num_chosen : ℕ := 3

-- Theorem stating the solution
theorem choose_3_from_10 : nat.choose num_people num_chosen = 120 :=
by {
  sorry
}

end choose_3_from_10_l130_130864


namespace find_x_l130_130867

variables (A B C D E : Point)
variables (x angle_ACE angle_ABC angle_ADC : ℝ)
variables (AB // DC : Prop) (ACE_straight : line_ACE)

-- Conditions
def given_conditions : Prop := 
  AB // DC ∧ 
  ACE_straight ∧ 
  angle_ACE = 105 ∧ 
  angle_ABC = 75 ∧ 
  angle_ADC = 115

-- The theorem statement
theorem find_x : given_conditions → x = 35 :=
by sorry

end find_x_l130_130867


namespace percentage_books_not_sold_approx_l130_130592

def initial_stock : ℕ := 1400
def books_sold_monday : ℕ := 62
def books_sold_tuesday : ℕ := 62
def books_sold_wednesday : ℕ := 60
def books_sold_thursday : ℕ := 48
def books_sold_friday : ℕ := 40

def total_books_sold : ℕ :=
  books_sold_monday + books_sold_tuesday + books_sold_wednesday + books_sold_thursday + books_sold_friday

def books_not_sold : ℕ :=
  initial_stock - total_books_sold

def percentage_books_not_sold : ℝ :=
  (books_not_sold / initial_stock) * 100

theorem percentage_books_not_sold_approx : percentage_books_not_sold ≈ 80.57 := by
  sorry

end percentage_books_not_sold_approx_l130_130592


namespace eight_div_repeating_three_l130_130151

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end eight_div_repeating_three_l130_130151


namespace vector_Q_proof_l130_130871

namespace VectorProof

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C G H Q : V)
variables (u v w : ℝ)

-- Conditions
def condition_G : G = (3/4 : ℝ) • A + (1/4 : ℝ) • B := sorry
def condition_H : H = (1/3 : ℝ) • B + (2/3 : ℝ) • C := sorry
def condition_Q : Q = (u : ℝ) • A + (v : ℝ) • B + (w : ℝ) • C := sorry
def sum_to_one : u + v + w = 1 := sorry

-- Theorem to Prove
theorem vector_Q_proof (hG : condition_G G A B) (hH : condition_H H B C) (hQ : condition_Q Q A B C u v w) (hsum : sum_to_one u v w) :
  Q = (3/7 : ℝ) • A + (2/7 : ℝ) • B + (2/7 : ℝ) • C :=
sorry

end VectorProof

end vector_Q_proof_l130_130871


namespace westminster_bridge_proof_l130_130638

noncomputable def westminster_bridge_times : (ℕ × ℕ) :=
  let x : ℚ := 23 + 71/143
  let y : ℚ := 41 + 137/143
  (x, y)

theorem westminster_bridge_proof:
  let x : ℚ := 23 + 71/143
  let y : ℚ := 41 + 137/143
  (240 + 0.5 * x = 6 * y) ∧ (120 + 0.5 * y = 6 * x) :=
by {
  sorry
}

end westminster_bridge_proof_l130_130638


namespace cannot_be_sum_of_four_consecutive_even_integers_l130_130575

-- Define what it means to be the sum of four consecutive even integers
def sum_of_four_consecutive_even_integers (n : ℤ) : Prop :=
  ∃ m : ℤ, n = 4 * m + 12 ∧ m % 2 = 0

-- State the problem in Lean 4
theorem cannot_be_sum_of_four_consecutive_even_integers :
  ¬ sum_of_four_consecutive_even_integers 32 ∧
  ¬ sum_of_four_consecutive_even_integers 80 ∧
  ¬ sum_of_four_consecutive_even_integers 104 ∧
  ¬ sum_of_four_consecutive_even_integers 200 :=
by
  sorry

end cannot_be_sum_of_four_consecutive_even_integers_l130_130575


namespace point_line_real_assoc_l130_130710

theorem point_line_real_assoc : 
  ∀ (p : ℝ), ∃! (r : ℝ), p = r := 
by 
  sorry

end point_line_real_assoc_l130_130710


namespace lower_selling_price_l130_130670

theorem lower_selling_price (cost_price selling_price : ℝ) (profit_percentage : ℝ) (lower_price lower_profit : ℝ) :
  selling_price = 350 →
  cost_price = 200 →
  profit_percentage = 0.05 →
  lower_profit + profit_percentage * lower_profit = selling_price - cost_price →
  lower_price = cost_price + lower_profit →
  lower_price ≈ 343 :=
by
  sorry

end lower_selling_price_l130_130670


namespace area_ratios_equal_l130_130299

theorem area_ratios_equal
  (O : Type) [metric_space O] {P Q: Type} [metric_space P] [metric_space Q]
  {A B C D E F G H I : Type} [metric_space A] [metric_space B] 
  [metric_space C] [metric_space D] [metric_space E]
  [metric_space F] [metric_space G] [metric_space H]
  [metric_space I]
  (incircle_condition : ∀ (O : metric_space), is_incircle O (triangle ABC))
  (tangency_points : D ∈ line BC ∧ E ∈ line CA ∧ F ∈ line AB)
  (DG_perpendicular_EF : ⊥(line DG) ∧ EF ⊥ (line DG) ∧ G ∈ line EF)
  (EH_perpendicular_DF : ⊥(line EH) ∧ DF ⊥ (line EH) ∧ H ∈ line DF)
  (FI_perpendicular_DE : ⊥(line FI) ∧ DE ⊥ (line FI) ∧ I ∈ line DE) :
  S_triangle G_B H / S_triangle G_C I = S_triangle G_F H / S_triangle G_E I := by
  sorry

end area_ratios_equal_l130_130299


namespace independent_of_joint_density_l130_130482

variable {X Y : Type} [MeasureSpace X] [MeasureSpace Y]
variable {f : X → Y → ℝ}
variable {φ : X → ℝ}
variable {ψ : Y → ℝ}

def independent (f : X → Y → ℝ) (fX : X → ℝ) (fY : Y → ℝ) :=
∀ x y, f (x, y) = fX x * fY y

theorem independent_of_joint_density (h : ∀ x y, f x y = φ x * ψ y) :
  ∃ fX fY, independent f fX fY :=
sorry

end independent_of_joint_density_l130_130482


namespace probability_no_rain_next_five_days_eq_1_over_243_l130_130962

theorem probability_no_rain_next_five_days_eq_1_over_243 :
  let p_rain : ℚ := 2 / 3 in
  let p_no_rain : ℚ := 1 - p_rain in
  let probability_no_rain_five_days : ℚ := p_no_rain ^ 5 in
  probability_no_rain_five_days = 1 / 243 :=
by
  sorry

end probability_no_rain_next_five_days_eq_1_over_243_l130_130962


namespace required_run_rate_l130_130704

def cricket_chase (target_runs : ℕ) (runs_scored_1st11_overs: ℕ) (remaining_overs : ℕ) : ℕ :=
  target_runs - runs_scored_1st11_overs / remaining_overs

theorem required_run_rate
  (run_rate_powerplay : ℕ) 
  (overs_powerplay : ℕ)
  (run_rate_non_powerplay : ℕ)
  (non_powerplay_overs : ℕ)
  (total_overs : ℕ)
  (wickets_lost : ℕ)
  (wickets_in_hand : ℕ)
  (target_runs : ℕ)
  (runs_scored : ℕ)
  (remaining_overs : ℕ) :
  (overs_powerplay * run_rate_powerplay + non_powerplay_overs * run_rate_non_powerplay = runs_scored) →
  (total_overs - (overs_powerplay + non_powerplay_overs) = remaining_overs) →
  wickets_lost = 3 →
  wickets_in_hand = 7 →
  target_runs = 360 →
  run_rate_powerplay = 3.5 →
  overs_powerplay = 10 →
  run_rate_non_powerplay = 7.5 →
  non_powerplay_overs = 2 →
  total_overs = 50 →
  remaining_overs = 38 →
  runs_scored = 50 →
  cricket_chase target_runs runs_scored remaining_overs = 8.1579 :=
by
  intros
  sorry

end required_run_rate_l130_130704


namespace probability_no_rain_next_five_days_eq_1_over_243_l130_130964

theorem probability_no_rain_next_five_days_eq_1_over_243 :
  let p_rain : ℚ := 2 / 3 in
  let p_no_rain : ℚ := 1 - p_rain in
  let probability_no_rain_five_days : ℚ := p_no_rain ^ 5 in
  probability_no_rain_five_days = 1 / 243 :=
by
  sorry

end probability_no_rain_next_five_days_eq_1_over_243_l130_130964


namespace total_apples_in_stack_l130_130628

theorem total_apples_in_stack:
  let base_layer := 6 * 9
  let layer_2 := 5 * 8
  let layer_3 := 4 * 7
  let layer_4 := 3 * 6
  let layer_5 := 2 * 5
  let layer_6 := 1 * 4
  let top_layer := 2
  base_layer + layer_2 + layer_3 + layer_4 + layer_5 + layer_6 + top_layer = 156 :=
by sorry

end total_apples_in_stack_l130_130628


namespace division_of_decimal_l130_130112

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l130_130112


namespace probability_neither_red_nor_purple_l130_130611

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 18
def yellow_balls : ℕ := 8
def red_balls : ℕ := 5
def purple_balls : ℕ := 7

theorem probability_neither_red_nor_purple : 
  (total_balls - (red_balls + purple_balls)) / total_balls = 4 / 5 :=
by sorry

end probability_neither_red_nor_purple_l130_130611


namespace determine_ab_l130_130336

theorem determine_ab : ∃ a b : ℤ, (real.sqrt (16 - 12 * real.sin (real.pi / 4.5)) = a + b * real.csc (real.pi / 4.5)) ∧ a = 4 ∧ b = real.sqrt 3 :=
  by
    sorry

end determine_ab_l130_130336


namespace count_integers_in_interval_l130_130807

theorem count_integers_in_interval : 
  ∃ (n : ℕ), (∀ (x : ℤ), (-2 ≤ x ∧ x ≤ 8 → ∃ (k : ℕ), k < n ∧ x = -2 + k)) ∧ n = 11 := 
by
  sorry

end count_integers_in_interval_l130_130807


namespace union_complements_eq_l130_130792

-- Definitions for the universal set U and subsets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5, 7}
def B : Set ℕ := {3, 4, 5}

-- Definition of the complements of A and B with respect to U
def complement_U_A : Set ℕ := {x ∈ U | x ∉ A}
def complement_U_B : Set ℕ := {x ∈ U | x ∉ B}

-- The union of the two complements
def union_complements : Set ℕ := complement_U_A ∪ complement_U_B

-- The target proof statement
theorem union_complements_eq : union_complements = {1, 2, 3, 6, 7} := by
  sorry

end union_complements_eq_l130_130792


namespace Eight_div_by_repeating_decimal_0_3_l130_130158

theorem Eight_div_by_repeating_decimal_0_3 : (8 : ℝ) / (0.3333333333333333 : ℝ) = 24 := by
  have h : 0.3333333333333333 = (1 : ℝ) / 3 := by sorry
  rw [h]
  exact (8 * 3 = 24 : ℝ)

end Eight_div_by_repeating_decimal_0_3_l130_130158


namespace combinatorial_sum_l130_130484

theorem combinatorial_sum (n m : ℕ) : 
  (∑ k in finset.range (n + 1), (-1)^(k + 1) * nat.choose n k * nat.choose (k * n) n) 
  = (-1)^(n + 1) * n^n :=
by 
  sorry

end combinatorial_sum_l130_130484


namespace length_of_segment_AB_l130_130408

noncomputable def circle_1 := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 = 5}
noncomputable def circle_2 (m : ℝ) := {p : ℝ × ℝ | (p.1 + m)^2 + (p.2)^2 = 20}

theorem length_of_segment_AB (m : ℝ)
  (h1 : ∃ A B, A ∈ circle_1 ∧ A ∈ circle_2 m ∧ B ∈ circle_1 ∧ B ∈ circle_2 m)
  (h2 : ∀ A B, (A ∈ circle_1 ∧ A ∈ circle_2 m ∧ B ∈ circle_1 ∧ B ∈ circle_2 m) →
                let p := complex.abs (complex.A - complex.O1) in 
                let q := complex.abs (complex.A - complex.O2) in 
                p * q = 0 → m = 5 ∨ m = -5) :
  ∃ A B, (A ∈ circle_1 ∧ A ∈ circle_2 m ∧ B ∈ circle_1 ∧ B ∈ circle_2 m) → 
         dist A B = 4 := 
sorry

end length_of_segment_AB_l130_130408


namespace range_of_a_plus_c_l130_130861

theorem range_of_a_plus_c 
  (A B C a b c : ℝ) 
  (h1 : triangle_acute A B C)
  (h2 : b = 2 * sqrt 3)
  (h3 : cos A / a + cos B / b = (2 * sqrt 3 * sin C) / (3 * a)) :
  a + c ∈ Ioo 6 (4 * sqrt 3) :=
sorry

end range_of_a_plus_c_l130_130861


namespace factorization_a_minus_b_l130_130936

theorem factorization_a_minus_b (a b : ℤ) (h1 : 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) : a - b = -7 :=
by
  sorry

end factorization_a_minus_b_l130_130936


namespace triangle_area_ratio_l130_130459

theorem triangle_area_ratio
  {A B C P : Type} [AddCommGroup A] [Module ℝ A]
  (point_in_triangle : ∃ P, ∃ A B C : A, True) 
  (vec_eq : ∀ (P A B C : A), P - A + 3 • (P - B) + 2 • (P - C) = 0) :
  ∃ (ratio: ℝ), 
  ratio = 3 := 
sorry

end triangle_area_ratio_l130_130459


namespace bob_age_l130_130684

theorem bob_age (a b : ℝ) 
    (h1 : b = 3 * a - 20)
    (h2 : b + a = 70) : 
    b = 47.5 := by
    sorry

end bob_age_l130_130684


namespace ellipse_foci_distance_l130_130732

def midpoint (p1 p2 : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def fourth_endpoint (p1 p3 : ℝ × ℝ) : ℝ × ℝ :=
  let mid := midpoint p1 p3
  in (mid.1, 2 * mid.2 - -1)

def distance_between_foci (a b : ℝ) : ℝ :=
  2 * real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  let p1 := (1, 3)
      p2 := (4, -1)
      p3 := (10, 3)
      fourth := fourth_endpoint p1 p3
      a := distance p1 p3 / 2
      b := distance p2 fourth / 2
  in fourth = (5.5, 7) ∧ distance_between_foci a b = 4.12 :=
by 
  sorry

end ellipse_foci_distance_l130_130732


namespace largest_five_digit_number_with_product_l130_130563

theorem largest_five_digit_number_with_product :
  ∃ (x : ℕ), (x = 98752) ∧ (∀ (d : List ℕ), (x.digits 10 = d) → (d.prod = 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) ∧ (x < 100000) ∧ (x ≥ 10000) :=
by
  sorry

end largest_five_digit_number_with_product_l130_130563


namespace probability_of_no_rain_l130_130955

theorem probability_of_no_rain (prob_rain : ℚ) (prob_no_rain : ℚ) (days : ℕ) (h : prob_rain = 2/3) (h_prob_no_rain : prob_no_rain = 1 - prob_rain) :
  (prob_no_rain ^ days) = 1/243 :=
by 
  sorry

end probability_of_no_rain_l130_130955


namespace initial_fee_of_first_plan_equals_65_l130_130797

theorem initial_fee_of_first_plan_equals_65 (miles : ℕ) (F : ℝ) 
  (h_cost_eq : ∀ miles, F + 0.40 * miles = 0.60 * miles) : F = 65 :=
by
  have key_eq : F + 0.40 * (325 : ℝ) = 0.60 * (325 : ℝ) := h_cost_eq 325
  rw [show 0.40 * 325 = 130, by norm_num] at key_eq
  rw [show 0.60 * 325 = 195, by norm_num] at key_eq
  linarith

end initial_fee_of_first_plan_equals_65_l130_130797


namespace tan_pi_over_4_plus_a_l130_130734

noncomputable def condition_cos (a : ℝ) : Prop :=
  Real.cos (π / 2 + a) = 2 * Real.cos (π - a)

theorem tan_pi_over_4_plus_a (a : ℝ) (h : condition_cos a) :
  Real.tan (π / 4 + a) = -3 := 
sorry

end tan_pi_over_4_plus_a_l130_130734


namespace find_a_plus_b_l130_130022

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def piecewise_function (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then x - 1 else if x < 0 then x + b else a

theorem find_a_plus_b (a b : ℝ) :
  is_odd (piecewise_function a b) → a + b = 1 :=
by
  intro h
  have h0 : piecewise_function a b 0 = 0 := by
    exact h 0
  have ha : a = 0 := by
    simp [piecewise_function] at h0
    exact h0

  have h1 : piecewise_function a b 1 = -piecewise_function a b (-1) := by
    exact h 1

  have hb : b = 1 := by
    simp [piecewise_function] at h1
    rw [ha, sub_eq_add_neg, neg_add', neg_neg] at h1
    simp at h1
    assumption

  rw [ha, hb]
  exact add_zero 1

#eval find_a_plus_b

end find_a_plus_b_l130_130022


namespace employees_trained_in_family_buffet_l130_130300

theorem employees_trained_in_family_buffet :
  ∀ (total emp_d emp_s two_rest three_rest : ℕ),
    total = 39 → emp_d = 18 → emp_s = 12 → two_rest = 4 → three_rest = 3 →
    ∃ F : ℕ, F + emp_d + emp_s - two_rest - 2 * three_rest + three_rest = total ∧ F = 20 := by
  intro total emp_d emp_s two_rest three_rest ht hd hs htwo hthree
  use 20
  constructor
  · rw [ht, hd, hs, htwo, hthree]
    norm_num
  · rfl

end employees_trained_in_family_buffet_l130_130300


namespace units_digit_fraction_l130_130571

open Nat

theorem units_digit_fraction : 
  (15 * 16 * 17 * 18 * 19 * 20) % 500 % 10 = 2 := by
  sorry

end units_digit_fraction_l130_130571


namespace division_by_repeating_decimal_l130_130198

theorem division_by_repeating_decimal: 8 / (0 + (list.repeat 3 (0 + 1)) - 3) = 24 :=
by sorry

end division_by_repeating_decimal_l130_130198


namespace find_theta_l130_130026

variable {t θ α : Real}

-- Define the parametric equations of the line.
def line_x (t θ : Real) := t * Real.cos θ
def line_y (t θ : Real) := t * Real.sin θ

-- Define the parametric equations of the circle.
def circle_x (α : Real) := 4 + 2 * Real.cos α
def circle_y (α : Real) := 2 * Real.sin α

-- Define the function to compute the distance from point (a, b) to the line y = mx.
def distance_from_point_to_line (a b m : Real) := Real.abs (a * m - b) / Real.sqrt (1 + m^2)

theorem find_theta (hθ : 0 < θ ∧ θ < π) (h_tangent : ∃ t α, line_x t θ = circle_x α ∧ line_y t θ = circle_y α) :
  θ = π / 6 ∨ θ = 5 * π / 6 :=
by
  sorry

end find_theta_l130_130026


namespace measure_of_AED_l130_130872

-- Importing the necessary modules for handling angles and geometry
variables {A B C D E : Type}
noncomputable def angle (p q r : Type) : ℝ := sorry -- Definition to represent angles in general

-- Given conditions
variables
  (hD_on_AC : D ∈ line_segment A C)
  (hE_on_BC : E ∈ line_segment B C)
  (h_angle_ABD : angle A B D = 30)
  (h_angle_BAE : angle B A E = 60)
  (h_angle_CAE : angle C A E = 20)
  (h_angle_CBD : angle C B D = 30)

-- The goal to prove
theorem measure_of_AED :
  angle A E D = 20 :=
by
  -- Proof details will go here
  sorry

end measure_of_AED_l130_130872


namespace valid_numbers_count_l130_130818

def valid_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 3 ∨ d = 4 ∨ d = 7 ∨ d = 8 ∨ d = 9

def valid_number (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, valid_digit d

theorem valid_numbers_count :
  (Finset.filter valid_number (Finset.range 10000)).card = 1295 :=
by
  sorry

end valid_numbers_count_l130_130818


namespace probability_of_sum_being_9_l130_130980

noncomputable def weightSet : Finset ℝ := {1, 2, 2, 3, 5} -- Set of weights as a finite set of reals.

noncomputable def numWaysChooseThree : ℝ := Finset.card (Finset.powersetLen 3 weightSet) -- Number of ways to choose 3 weights.

noncomputable def favorableOutcomes : Finset (Finset ℝ) := {s ∈ Finset.powersetLen 3 weightSet | s.sum = 9} -- Outcomes where the sum is 9g.

theorem probability_of_sum_being_9 :
  (Finset.card favorableOutcomes : ℝ) / numWaysChooseThree = 1 / 5 :=
  by sorry

end probability_of_sum_being_9_l130_130980


namespace tony_initial_money_l130_130555

theorem tony_initial_money (ticket_cost hotdog_cost money_left initial_money : ℕ) 
  (h_ticket : ticket_cost = 8)
  (h_hotdog : hotdog_cost = 3) 
  (h_left : money_left = 9)
  (h_spent : initial_money = ticket_cost + hotdog_cost + money_left) :
  initial_money = 20 := 
by 
  sorry

end tony_initial_money_l130_130555


namespace eight_div_repeat_three_l130_130122

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l130_130122


namespace rhombus_compression_problem_l130_130644

def rhombus_diagonal_lengths (side longer_diagonal : ℝ) (compression : ℝ) : ℝ × ℝ :=
  let new_longer_diagonal := longer_diagonal - compression
  let new_shorter_diagonal := 1.2 * compression + 24
  (new_longer_diagonal, new_shorter_diagonal)

theorem rhombus_compression_problem :
  let side := 20
  let longer_diagonal := 32
  let compression := 2.62
  rhombus_diagonal_lengths side longer_diagonal compression = (29.38, 27.14) :=
by sorry

end rhombus_compression_problem_l130_130644


namespace nods_per_kilometer_l130_130852

theorem nods_per_kilometer
  (p q r s t u : ℕ)
  (h1 : p * q = q * p)
  (h2 : r * s = s * r)
  (h3 : t * u = u * t) : 
  (1 : ℕ) = qts/pru :=
by
  sorry

end nods_per_kilometer_l130_130852


namespace islander_real_name_l130_130585

-- Definition of types of people on the island
inductive IslanderType
| Knight   -- Always tells the truth
| Liar     -- Always lies
| Normal   -- Can lie or tell the truth

-- The possible names of the islander
inductive Name
| Edwin
| Edward

-- Condition: You met the islander who can be Edwin or Edward
def possible_names : List Name := [Name.Edwin, Name.Edward]

-- Condition: The islander said their name is Edward
def islander_statement : Name := Name.Edward

-- Condition: The islander is a Liar (as per the solution interpretation)
def islander_type : IslanderType := IslanderType.Liar

-- The proof problem: Prove the islander's real name is Edwin
theorem islander_real_name : islander_type = IslanderType.Liar ∧ islander_statement = Name.Edward → ∃ n : Name, n = Name.Edwin :=
by
  sorry

end islander_real_name_l130_130585


namespace simplify_and_evaluate_l130_130001

theorem simplify_and_evaluate :
  let x := (abs (sqrt 3 - 2) + (1/2) ^ (-1) - ((π - 3.14) ^ 0) - (3 : ℝ) ^ (1/3) + 1 : ℝ)
  ( ( 2*x^2 + 2*x)/(x^2 - 1) - ( x^2 - x)/(x^2 - 2*x + 1) ) / ( x/(x + 1) )
  = -((2 * sqrt 3) / 3) + 1 := by 
  sorry

end simplify_and_evaluate_l130_130001


namespace min_value_x_plus_2y_l130_130760

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 1 / y = 4) : x + 2 * y = 2 :=
sorry

end min_value_x_plus_2y_l130_130760


namespace no_rain_five_days_l130_130957

-- Define the problem conditions and the required result.
def prob_rain := (2 / 3)
def prob_no_rain := (1 - prob_rain)
def prob_no_rain_five_days := prob_no_rain^5

theorem no_rain_five_days : 
  prob_no_rain_five_days = (1 / 243) :=
by
  sorry

end no_rain_five_days_l130_130957


namespace divide_by_repeating_decimal_l130_130104

theorem divide_by_repeating_decimal :
  (8 : ℝ) / (0.333333333333333... : ℝ) = 24 :=
by
  have h : (0.333333333333333... : ℝ) = (1 : ℝ) / (3 : ℝ) := sorry
  rw [h]
  calc
    (8 : ℝ) / ((1 : ℝ) / (3 : ℝ)) = (8 : ℝ) * (3 : ℝ) : by field_simp
                        ...          = 24             : by norm_num

end divide_by_repeating_decimal_l130_130104


namespace completing_the_square_l130_130574

theorem completing_the_square (x : ℝ) :
  x^2 - 6 * x + 2 = 0 →
  (x - 3)^2 = 7 :=
by sorry

end completing_the_square_l130_130574


namespace probability_of_at_least_6_consecutive_heads_l130_130266

-- Define the conditions
def flip_options : finset (fin 9 → bool) := 
  finset.univ

def at_least_6_consecutive_heads (seq : fin 9 → bool) : bool :=
  (seq 0 && seq 1 && seq 2 && seq 3 && seq 4 && seq 5) ||
  (seq 1 && seq 2 && seq 3 && seq 4 && seq 5 && seq 6) ||
  (seq 2 && seq 3 && seq 4 && seq 5 && seq 6 && seq 7) ||
  (seq 3 && seq 4 && seq 5 && seq 6 && seq 7 && seq 8)

-- Define the theorem to prove the probability
theorem probability_of_at_least_6_consecutive_heads :
  (flip_options.filter at_least_6_consecutive_heads).card = 11 / 512 :=
by
  sorry

end probability_of_at_least_6_consecutive_heads_l130_130266


namespace find_initial_fee_l130_130798

noncomputable def initial_fee (F d: ℝ) : Prop :=
  (F + 0.40 * d = 0.60 * d)

theorem find_initial_fee : initial_fee 65 325 := 
by
  unfold initial_fee
  have h1 : 0.60 * 325 = 0.60 * 325 := rfl -- Just to show steps, Lean already knows this
  calc
    65 + 0.40 * 325 = 65 + 130 : by norm_num
    ... = 195 : by norm_num
    ... = 0.60 * 325 : by norm_num

end find_initial_fee_l130_130798


namespace evaluate_i_powers_l130_130328

theorem evaluate_i_powers :
  (complex.I ^ 11 + complex.I ^ 111) = -2 * complex.I :=
by sorry

end evaluate_i_powers_l130_130328


namespace find_a_minus_b_l130_130939

-- Definitions based on conditions
def eq1 (a b : Int) : Prop := 2 * b + a = 5
def eq2 (a b : Int) : Prop := a * b = -12

-- Statement of the problem
theorem find_a_minus_b (a b : Int) (h1 : eq1 a b) (h2 : eq2 a b) : a - b = -7 := 
sorry

end find_a_minus_b_l130_130939


namespace inscribed_circle_center_of_tangent_circles_l130_130474

open EuclideanGeometry

noncomputable def center_of_inscribed_circle (A B C D I : Point) (ω1 ω2 : Circle) : Prop :=
  is_diameter A B ω1 ∧ is_diameter C D ω2 ∧
  externally_tangent ω1 ω2 ∧
  convex_cyclic_quadrilateral A B C D ∧
  is_inscribed_circle_center I A B C D

theorem inscribed_circle_center_of_tangent_circles (A B C D I : Point) (ω1 ω2 : Circle) 
  (h1: is_diameter A B ω1) (h2: is_diameter C D ω2)
  (h3: externally_tangent ω1 ω2) (h4: convex_cyclic_quadrilateral A B C D)
  (h5: is_inscribed_circle_center I A B C D) :
  I = point_of_tangency ω1 ω2 :=
sorry

end inscribed_circle_center_of_tangent_circles_l130_130474


namespace rationalize_sqrt_three_sub_one_l130_130492

theorem rationalize_sqrt_three_sub_one :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_sqrt_three_sub_one_l130_130492


namespace parametric_to_standard_line_parametric_to_standard_ellipse_l130_130556

theorem parametric_to_standard_line (t : ℝ) (x y : ℝ) 
  (h₁ : x = 1 - 3 * t)
  (h₂ : y = 4 * t) :
  4 * x + 3 * y - 4 = 0 := by
sorry

theorem parametric_to_standard_ellipse (θ x y : ℝ) 
  (h₁ : x = 5 * Real.cos θ)
  (h₂ : y = 4 * Real.sin θ) :
  (x^2 / 25) + (y^2 / 16) = 1 := by
sorry

end parametric_to_standard_line_parametric_to_standard_ellipse_l130_130556


namespace exists_parallel_segment_l130_130360

-- Define the problem context with the given circle, square, and line.

structure Circle (O : Type) :=
  (center : O)
  (radius : ℝ)

structure Square (K : Type) :=
  (center : K)
  (side_length : ℝ)

structure Line (L : Type) :=
  (point : L)
  (slope : ℝ)

variables {O K L : Type}

-- The formal Lean statement declaring the existence of the segment.
theorem exists_parallel_segment (circle : Circle O) (square : Square K) (line : Line L) (length : ℝ) :
  ∃ (seg : O × K), 
    (parallel_to line seg) ∧
    (length_of seg = length) ∧
    (lies_on_circle seg.1 circle) ∧
    (lies_on_square seg.2 square) :=
sorry

-- Definitions for parallelism, segment length, and point containment:
def lies_on_circle (point : O) (circle : Circle O) : Prop := sorry
def lies_on_square (point : K) (square : Square K) : Prop := sorry
def parallel_to (line : Line L) (seg : O × K) : Prop := sorry
def length_of (seg : O × K) : ℝ := sorry

end exists_parallel_segment_l130_130360


namespace no_quadratic_trinomials_l130_130323

theorem no_quadratic_trinomials :
  ¬ ∃ (P Q R : ℤ → ℤ), (quadratic_trinomial P ∧ quadratic_trinomial Q ∧ quadratic_trinomial R) ∧
  ∀ (x y : ℤ), ∃ (z : ℤ), P x + Q y = R z :=
by
  sorry

-- A helper definition to check if a given function f is a quadratic trinomial.
def quadratic_trinomial (f : ℤ → ℤ) : Prop :=
  ∃ (a b c : ℤ), ∀ x, f x = a * x ^ 2 + b * x + c

end no_quadratic_trinomials_l130_130323


namespace oxygen_content_function_fish_growth_slows_down_suitable_time_to_sign_contract_l130_130881

-- Question (1) Prove the relationship model function of the oxygen content
def oxygen_content_model (k a : ℝ) (x : ℕ) : ℝ :=
  k * (a ^ x)

def first_year_condition (k : ℝ) (a : ℝ) : Prop :=
  8 = k * a

def third_year_condition (k : ℝ) (a : ℝ) : Prop :=
  4.5 = k * (a ^ 3)

theorem oxygen_content_function :
  ∃ k a : ℝ, k > 0 ∧ 0 < a ∧ a < 1 ∧ first_year_condition k a ∧ third_year_condition k a ∧
  ∀ x : ℕ, oxygen_content_model k a x = 8 * (3/4)^(x-1) :=
sorry

-- Question (2) Determine in which year the fish growth will slow down
def oxygen_content_threshold (k : ℝ) (a : ℝ) (x : ℕ) : Prop :=
  oxygen_content_model k a x < (81 / 32)

theorem fish_growth_slows_down (k a : ℝ) (h_k : k > 0) (h_a : 0 < a ∧ a < 1)
  (h_first_year : first_year_condition k a) (h_third_year : third_year_condition k a) :
  ∃ y : ℕ, y = 6 ∧ ∀ x ≥ y, oxygen_content_threshold k a x :=
sorry

-- Question (3) Prove the relationship formula and determine how many years to sign the contract
def weight_growth (k : ℝ) (a : ℝ) (n : ℕ) (an : ℝ) : ℝ :=
  if n ≤ 5 then (1 + 1 / 2^(n-2)) * an
  else (1 + 1 / 2^(n-2)) * an * 0.95

theorem suitable_time_to_sign_contract (a6 a7 : ℝ) :
  ∃ ani : ℕ, ani = 7 :=
sorry

end oxygen_content_function_fish_growth_slows_down_suitable_time_to_sign_contract_l130_130881


namespace new_cylinder_volume_l130_130289

theorem new_cylinder_volume (V : ℝ) (r h : ℝ) (original_volume : V = π * r^2 * h) (new_volume : V' = π * (3 * r)^2 * (2 * h)) : V' = 54 := 
by
  have : V = 3 := by
    calc
      V = π * r^2 * h  := original_volume
        ... = 3        := sorry  -- Given original volume is 3 gallons
  calc
    V' = π * (3 * r)^2 * (2 * h) := new_volume
       ... = 18 * V              := by sorry
       ... = 18 * 3              := by sorry
       ... = 54                  := by sorry

end new_cylinder_volume_l130_130289


namespace rationalize_denominator_l130_130487

theorem rationalize_denominator : (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_denominator_l130_130487


namespace minimum_x_coord_of_midpoint_l130_130740

noncomputable def min_x_coord_midpoint (a b m : ℝ) (h : m > 2 * b^2 / a) : ℝ :=
  a * (m + 2 * a) / (2 * Real.sqrt (a^2 + b^2))

theorem minimum_x_coord_of_midpoint
  (a b m : ℝ)
  (h_hyperbola : ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_m : m > 2 * b^2 / a) :
  ∃ (M : ℝ), M = min_x_coord_midpoint a b m h :=
begin
  use min_x_coord_midpoint a b m h,
  sorry
end

end minimum_x_coord_of_midpoint_l130_130740


namespace integral_identity_1_integral_identity_2_l130_130703

-- Define the greatest integer function
def floor (x : ℝ) : ℤ := int.floor x

-- Problem 1: Prove the integral identity given
theorem integral_identity_1 (f : ℝ → ℝ) (k : ℝ) (hk : k > 1) :
  ∫ (x : ℝ) in 1..k, floor x * deriv f x = floor k * f k - ∑ n in finset.Ico 1 (floor k + 1), f n :=
sorry

-- Problem 2: Prove the similar integral identity
theorem integral_identity_2 (f : ℝ → ℝ) (k : ℝ) (hk : k > 1) :
  ∫ (x : ℝ) in 1..k, floor (x^2) * deriv f x = floor (k^2) * f k - ∑ n in finset.Ico 1 (floor (k^2) + 1), f (real.sqrt n) :=
sorry

end integral_identity_1_integral_identity_2_l130_130703


namespace weeds_cannot_spread_to_all_cells_l130_130904

-- Define the grid and conditions
def grid_size : ℕ := 10
def initial_weeds (grid : Fin grid_size × Fin grid_size → Prop) : Prop :=
  (∃ cells : List (Fin grid_size × Fin grid_size),
    cells.length = 9 ∧
    ∀ cell, cell ∈ cells → grid cell)

-- Define the weed propagation rule
noncomputable def weed_propagation (grid : Fin grid_size × Fin grid_size → Prop) : Prop :=
  ∀ cell, grid cell → 
  (∃ neighbors, neighbors.length = 2 ∧ 
  ∀ neighbor, neighbor ∈ neighbors → grid neighbor)

-- Define the main theorem to be proved
theorem weeds_cannot_spread_to_all_cells (grid : Fin grid_size × Fin grid_size → Prop)
  (h1 : initial_weeds grid)
  (h2 : weed_propagation grid) :
  ¬ (∀ cell, grid cell) :=
by sorry

end weeds_cannot_spread_to_all_cells_l130_130904


namespace max_followers_l130_130905

/-- 
On an island, there are 2018 inhabitants comprising knights, liars, and followers. 
Each was asked to answer "Yes" or "No" to the question "Are there more knights than liars on the island?"
Knights tell the truth, liars lie, and followers mimic the majority of prior answers or answer arbitrarily if there is no majority.
There are exactly 1009 "Yes" answers. 

Prove that the maximum number of followers on the island equals 1009.
-/
theorem max_followers (total_inhabitants : ℕ)
  (yes_answers : ℕ)
  (K : Type) (L : Type) (F : Type)
  [inhabited K] [inhabited L] [inhabited F]
  (is_knight : K → Prop)
  (is_liar : L → Prop)
  (is_follower : F → Prop)
  (truth_told : K → bool → bool)
  (lie_told : L → bool → bool)
  (mimic_majority : F → bool → bool → bool)
  (total_inhabitants = 2018)
  (yes_answers = 1009)
  : ∃ m : ℕ, m = 1009 :=
sorry

end max_followers_l130_130905


namespace division_of_decimal_l130_130117

theorem division_of_decimal :
  8 / (1 / 3) = 24 :=
by
  linarith

end division_of_decimal_l130_130117


namespace part1_part2_l130_130785

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x + a * x ^ 2 - 3 * x

theorem part1 (a : ℝ) (h_tangent : deriv (f 1 a) = 0) : 
  ∃ x : ℝ, (0 < x ∧ x < 1) ∧ (∀ y : ℝ, 0 < y ∧ y < 1 → f y a ≥ f x a) ∧ f x a = - log 2 - 5 / 4 := sorry

theorem part2 (m : ℝ) (x1 x2 : ℝ) (h_a : ∀ x1 x2 ∈ (set.Icc 1 2), x1 < x2 → f x1 1 - f x2 1 > m * (x2 - x1) / (x1 * x2)) : 
  m ≤ -6 := sorry

end part1_part2_l130_130785


namespace f_4_eq_24_l130_130758

noncomputable def f : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * f n

theorem f_4_eq_24 : f 4 = 24 := by
  sorry

end f_4_eq_24_l130_130758


namespace sequence_general_term_l130_130435

open Nat

theorem sequence_general_term (a : ℕ → ℚ) 
  (h₀ : a 1 = 1) 
  (h₁ : ∀ n, a (n + 1) = 3 * a n + 2 * n - 1) :
  ∀ n, a n = (2 / 3 : ℚ) * 3 ^ n - n := 
sorry

end sequence_general_term_l130_130435


namespace imaginary_part_of_z_l130_130025

def z : ℂ := 1 - 2 * complex.i

theorem imaginary_part_of_z : complex.im z = -2 :=
by
satisfy run

end imaginary_part_of_z_l130_130025


namespace concave_probability_l130_130653

def is_concave_number (a b c : ℕ) : Prop := a > b ∧ b < c

def valid_digits (a b c : ℕ) : Prop := 
  a ∈ ({1, 2, 3, 4} : Finset ℕ) ∧ 
  b ∈ ({1, 2, 3, 4} : Finset ℕ) ∧ 
  c ∈ ({1, 2, 3, 4} : Finset ℕ) ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem concave_probability : 
  (Finset.card { n : ℕ | ∃ a b c, valid_digits a b c ∧ is_concave_number a b c ∧ n = 100 * a + 10 * b + c }) = 
  (1/3 : ℚ) * 
  (Finset.card { n : ℕ | ∃ a b c, valid_digits a b c ∧ n = 100 * a + 10 * b + c }) := 
sorry

end concave_probability_l130_130653


namespace gcd_of_256_450_720_is_18_l130_130995

-- Defining the constants based on the conditions
def a : ℕ := 256
def b : ℕ := 450
def c : ℕ := 720

-- The problem proof statement in Lean 4
theorem gcd_of_256_450_720_is_18 : Nat.gcd (Nat.gcd a b) c = 18 :=
by
  -- We declare the structure here, proof to be filled later
  sorry

end gcd_of_256_450_720_is_18_l130_130995
