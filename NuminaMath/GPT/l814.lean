import Complex
import Mathlib
import Mathlib.
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.FunctionalAnalysis.Banach
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.GroupWithZero.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Module.Submodule.Basic
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Order.ArithmeticMean
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.VectorSpace
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Combinatorial
import Mathlib.Combinatorics.CombinatorialGame
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Matrix.Polynomial
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Polynomial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.FiniteDimensional
import Mathlib.Tactic

namespace calculate_expression_l814_814186

theorem calculate_expression :
  (Real.log 5) ^ 2 - (Real.log 2) ^ 2 + 8 ^ (2 / 3) * Real.log (sqrt 2) - 0.6 ^ 0 + 0.2 ^ (-1) = 5 :=
by
  sorry

end calculate_expression_l814_814186


namespace right_angled_iff_2R_eq_s_minus_r_l814_814900

-- Definitions for the semi-perimeter, circumradius, and inradius of a triangle.
def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2
def circumradius (a b c K : ℝ) : ℝ := (a * b * c) / (4 * K)
def inradius (K s : ℝ) : ℝ := K / s

-- Definition for a right-angled triangle
def is_right_angled (a b c : ℝ) : Prop := c^2 = a^2 + b^2

-- The main statement to prove
theorem right_angled_iff_2R_eq_s_minus_r 
  (a b c K s r R : ℝ)
  (h_s : s = semi_perimeter a b c)
  (h_R : R = circumradius a b c K)
  (h_r : r = inradius K s)
  (h_area : K = (1/2) * a * b) : 
  is_right_angled a b c ↔ 2 * R = s - r :=
sorry

end right_angled_iff_2R_eq_s_minus_r_l814_814900


namespace wire_length_10_loops_9440_2_m_l814_814598

def length_wire (L W : ℕ) (t : ℕ) (n : ℕ) : ℚ :=
  let P := 2 * (L + W)
  let L_total := n * P
  let P_add := 0.02
  let L_add_total := n * P_add
  L_total + L_add_total

theorem wire_length_10_loops_9440_2_m :
  ∀ (L W : ℕ) (t : ℕ) (n : ℕ), 
    L = 194 → W = 278 → t = 5 → n = 10 →
    length_wire L W t n = 9440.2 := 
by
  intros L W t n hL hW ht hn
  rw [hL, hW, ht, hn]
  sorry

end wire_length_10_loops_9440_2_m_l814_814598


namespace dominos_balance_l814_814195

theorem dominos_balance (b n : ℕ) :
  (∃ (b_dom n_dom : ℕ),
    b_dom + n_dom = 32 ∧
    b_dom = b ∧
    n_dom = n ∧
    ∀ i ∈ finset.range(8),
      ∃ (bi ni : ℕ),
        bi = b_dom / 8 ∧
        ni = n_dom / 8 ∧
        bi = ni) →
  b = n :=
by 
sorry

end dominos_balance_l814_814195


namespace remainder_of_sum_div_6_l814_814466

def sequence_term (n : ℕ) : ℕ := 6 * n - 5

theorem remainder_of_sum_div_6 : 
  let S := finset.sum (finset.range 68) sequence_term in
  S % 6 = 2
:= 
sorry

end remainder_of_sum_div_6_l814_814466


namespace array_element_count_l814_814176

theorem array_element_count (A : Finset ℕ) 
  (h1 : ∀ n ∈ A, n ≠ 1 → (∃ a ∈ [2, 3, 5], a ∣ n)) 
  (h2 : ∀ n ∈ A, (2 * n ∈ A ∨ 3 * n ∈ A ∨ 5 * n ∈ A) ↔ (n ∈ A ∧ 2 * n ∈ A ∧ 3 * n ∈ A ∧ 5 * n ∈ A)) 
  (card_A_range : 300 ≤ A.card ∧ A.card ≤ 400) : 
  A.card = 364 := 
sorry

end array_element_count_l814_814176


namespace angle_between_two_vectors_l814_814267

noncomputable theory
open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]

def is_unit_vector (v : V) : Prop := ∥v∥ = 1

theorem angle_between_two_vectors (a b : V) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (h : ∥a - (2 : ℝ) • b∥ = real.sqrt 3) :
  real.angle a b = real.pi / 3 :=
sorry

end angle_between_two_vectors_l814_814267


namespace terry_age_proof_l814_814698

-- Condition 1: In 10 years, Terry will be 4 times the age that Nora is currently.
-- Condition 2: Nora is currently 10 years old.
-- We need to prove that Terry's current age is 30 years old.

variable (Terry_now Terry_in_10 Nora_now : ℕ)

theorem terry_age_proof (h1: Terry_in_10 = 4 * Nora_now) (h2: Nora_now = 10) (h3: Terry_in_10 = Terry_now + 10) : Terry_now = 30 := 
by
  sorry

end terry_age_proof_l814_814698


namespace find_angle_A_l814_814333

theorem find_angle_A (a b : ℝ) (sin_B : ℝ) (ha : a = 3) (hb : b = 4) (hsinB : sin_B = 2/3) :
  ∃ A : ℝ, A = π / 6 :=
by
  sorry

end find_angle_A_l814_814333


namespace average_percentage_revenue_fall_l814_814825

theorem average_percentage_revenue_fall
  (initial_revenue_A final_revenue_A : ℝ)
  (initial_revenue_B final_revenue_B : ℝ) (exchange_rate_B : ℝ)
  (initial_revenue_C final_revenue_C : ℝ) (exchange_rate_C : ℝ) :
  initial_revenue_A = 72.0 →
  final_revenue_A = 48.0 →
  initial_revenue_B = 20.0 →
  final_revenue_B = 15.0 →
  exchange_rate_B = 1.30 →
  initial_revenue_C = 6000.0 →
  final_revenue_C = 5500.0 →
  exchange_rate_C = 0.0091 →
  (33.33 + 25 + 8.33) / 3 = 22.22 :=
by
  sorry

end average_percentage_revenue_fall_l814_814825


namespace ratio_of_square_side_to_distance_l814_814191

theorem ratio_of_square_side_to_distance (s : ℝ) (P : ℝ × ℝ) (A B C D : ℝ × ℝ) (O : ℝ × ℝ)
  (h₁ : B ≠ A ∧ C ≠ B ∧ D ≠ C ∧ A ≠ D) (area_APB area_BPC area_CPD area_DPA : ℝ)
  (h₂ : area_APB = 1 ∧ area_BPC = 2 ∧ area_CPD = 3 ∧ area_DPA = 4)
  (h₃ : IsSquare AB := ((D.x - A.x) = s ∧ (D.y - A.y) = s) ∧ centroid O := (A.x + B.x)/2,
  (center_of_square := (A.x + C.x)/2, (A.y + C.y)/2) :
  (distance O P := (P.x - O.x)^2 + (P.y - O.y)^2) : 10 = s * s;
  (h₄ := sqrt (distance_from_center P) = 2;
  h₅ := sqrt(2) = 1/2 → : 2;
:= sorry

end ratio_of_square_side_to_distance_l814_814191


namespace guessing_probability_l814_814847

theorem guessing_probability :
  let P_correct : ℚ := 1 - (5/6) ^ 6
  P_correct = 31031 / 46656 :=
by sorry

end guessing_probability_l814_814847


namespace sum_of_solutions_eq_zero_l814_814102

theorem sum_of_solutions_eq_zero (x : ℝ) (h : 6 * x / 30 = 7 / x) :
  (∃ x₁ x₂ : ℝ, x₁^2 = 35 ∧ x₂^2 = 35 ∧ x₁ + x₂ = 0) :=
sorry

end sum_of_solutions_eq_zero_l814_814102


namespace stamp_bijection_l814_814702

theorem stamp_bijection (n : ℕ) (hn : 0 < n) : 
  (∀ l_o, stamp_value_Oddland l_o = n → valid_stamp_Oddland l_o →
         ∃ l_s, stamp_value_Squareland l_s = n ∧ valid_stamp_Squareland l_s) ∧
  (∀ l_s, stamp_value_Squareland l_s = n → valid_stamp_Squareland l_s →
         ∃ l_o, stamp_value_Oddland l_o = n ∧ valid_stamp_Oddland l_o) := 
sorry

-- Definitions for conditions:

-- In Oddland
def odd_stamps : list ℕ := list.filter (λ x, x % 2 = 1) (list.nats (n+1))
def stamp_value_Oddland (l : list ℕ) : ℕ := l.sum
def valid_stamp_Oddland (l : list ℕ) : Prop := 
  ∀ i j, i < j → (l.nth i = none ∨ l.nth j = none ∨ l.nth i ≥ l.nth j)

-- In Squareland
def square_stamps : list ℕ := list.map (λ x, x * x) (list.nats (n).succ)
def stamp_value_Squareland (l : list ℕ) : ℕ := l.sum
def valid_stamp_Squareland (l : list ℕ) : Prop := 
  list.all (λ x, x ∈ square_stamps) l

-- Note: The custom definitions provided here are illustrative; in practice, you might need more sophisticated data structures or logic to accurately represent the problem's stamping rules and constraints.

end stamp_bijection_l814_814702


namespace smallest_x_y_sum_l814_814250

theorem smallest_x_y_sum :
  ∃ x y : ℕ,
    0 < x ∧ 0 < y ∧ x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 15) ∧ (x + y = 64) := 
by
  sorry

end smallest_x_y_sum_l814_814250


namespace value_of_x_add_y_not_integer_l814_814373

theorem value_of_x_add_y_not_integer (x y: ℝ) (h1: y = 3 * ⌊x⌋ + 4) (h2: y = 2 * ⌊x - 3⌋ + 7) (h3: ¬ ∃ n: ℤ, x = n): -8 < x + y ∧ x + y < -7 := 
sorry

end value_of_x_add_y_not_integer_l814_814373


namespace problem_statement_l814_814352

theorem problem_statement
  (AB AC : ℝ)
  (BC : ℝ)
  (radiusP : ℝ)
  (radiusQ_form : ℝ → ℝ)
  (m n k : ℕ)
  (k_prime_distinct : k = 41) -- Given k is a product of distinct primes
  (radiusQ : radiusQ_form = λ r : ℝ, 58 - 8 * Real.sqrt k)
  (condition1 : AB = 130)
  (condition2 : AC = 130)
  (condition3 : BC = 150)
  (condition4 : radiusP = 20) :
  m + n * k = 386 := by
  sorry

end problem_statement_l814_814352


namespace right_triangle_projection_theorem_l814_814249

theorem right_triangle_projection_theorem (A B C D : Type)
  [inhabited A] [inhabited B] [inhabited C] [inhabited D]
  (h_triangle: ∀ (a b c : Type), (a ≠ b) → (b ≠ c) → (a ≠ c))
  (height_from_hypotenuse: ∀ (a b d : Type), (a ≠ b) → (b ≠ d) → (d ≠ a))
  (CD : Type) (h_CD_4 : CD = 4) :
  AD * BD = 16 := by
  sorry

end right_triangle_projection_theorem_l814_814249


namespace correct_transformation_l814_814343

theorem correct_transformation :
  ∀ (x y : ℝ), 
    (∀ (x' y' : ℝ),
      (x' = x + 0 ∧ y' = y + 2) ∨
      (x' = x * 2 ∧ y' = y * 2) ∨
      (x' = x ∧ y' = y) ∨
      (x' = x * 2 ∧ y' = y) →
      (x - x' = 0 ∧ y - y' = 2) ∨
      (2 * (x - x') = 0 ∧ 2 * (y - y') = 0) ∨
      (x = x' ∧ y = y') ∨
      (2 * (x - x') = 0 ∧ (y - y') = 0)) :=
begin
  sorry
end

end correct_transformation_l814_814343


namespace total_amount_distributed_l814_814910

theorem total_amount_distributed :
  let num_persons := 22
  let amount_per_person := 1950
  num_persons * amount_per_person = 42900 := 
by
  let num_persons := 22
  let amount_per_person := 1950
  show num_persons * amount_per_person = 42900 from sorry

end total_amount_distributed_l814_814910


namespace garden_width_l814_814366

theorem garden_width (w : ℕ) (h_area : w * (w + 10) ≥ 150) : w = 10 :=
sorry

end garden_width_l814_814366


namespace smaller_solution_quadratic_smaller_solution_is_neg12_quadratic_has_smaller_solution_l814_814841

theorem smaller_solution_quadratic : ∀ x : ℝ, x^2 + 15 * x + 36 = 0 → x = -12 ∨ x = -3 :=
begin
  intro x,
  intro h,
  have h1 : (x + 12) * (x + 3) = 0,
  { 
    calc (x + 12) * (x + 3) 
        = x^2 + 3 * x + 12 * x + 36 : by ring
    ... = x^2 + 15 * x + 36         : by linarith },
  cases (em (x + 12 = 0)) with h2 h2,
  { left, linarith },
  { right, linarith },
  linarith,
end

theorem smaller_solution_is_neg12 : ∀ x1 x2 : ℝ, (x1 = -12 ∧ x2 = -3 ∨ x1 = -3 ∧ x2 = -12) → x1 = -12 :=
begin
  intros x1 x2 h,
  cases h,
  { exact h.1 },
  { exact h.2 },
end

theorem quadratic_has_smaller_solution : ∃ x1 x2 : ℝ, (x1^2 + 15 * x1 + 36 = 0 ∧ x2^2 + 15 * x2 + 36 = 0 ∧ x1 < x2) ∧ x1 = -12 :=
begin
  use [-12, -3],
  split,
  { split,
    { linarith },
    { linarith },
    { linarith } },
  linarith,
end

end smaller_solution_quadratic_smaller_solution_is_neg12_quadratic_has_smaller_solution_l814_814841


namespace problem1_a2_a3_a4_problem1_general_formula_problem2_i_problem2_ii_l814_814011

def seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n ^ 2 - n * a n + 1

noncomputable def a_n : ℕ → ℝ
| 1     := 2
| (n+1) := a_n n ^ 2 - n * a_n n + 1

theorem problem1_a2_a3_a4 :
  seq a_n →
  a_n 1 = 2 →
  a_n 2 = 3 ∧ a_n 3 = 4 ∧ a_n 4 = 5 :=
by
  intros h_seq h_a1

  -- Proof parts would go here.
  sorry

theorem problem1_general_formula (a : ℕ → ℝ) :
  seq a →
  a 1 = 2 →
  ∀ n, a n = n + 1 :=
by
  intros h_seq h_a1
  
  -- Proof parts would go here.
  sorry

theorem problem2_i (a : ℕ → ℝ) :
  seq a →
  a 1 ≥ 3 →
  ∀ n, a n ≥ n + 2 :=
by
  intros h_seq h_a1
  
  -- Proof parts would go here.
  sorry

theorem problem2_ii (a : ℕ → ℝ) :
  seq a →
  a 1 ≥ 3 →
  ∑ k in (range n).map (λ i, 1 / (1 + a (i + 1))) ≤ 1 / 2 :=
by
  intros h_seq h_a1
  
  -- Proof parts would go here.
  sorry

end problem1_a2_a3_a4_problem1_general_formula_problem2_i_problem2_ii_l814_814011


namespace equation_of_line_AB_equation_of_line_P_l814_814127

/-- Prove that the line passing through points A(3, 2) and B(-2, 0) has the equation 2x - 5y + 4 = 0 --/
theorem equation_of_line_AB (A B : ℝ × ℝ) (hA : A = (3, 2)) (hB : B = (-2, 0)) :
  ∃ (a b c : ℝ), a * (fst A) + b * (snd A) + c = 0 ∧ a * (fst B) + b * (snd B) + c = 0 ∧ (a = 2 ∧ b = -5 ∧ c = 4) :=
by
  sorry

/-- Prove that the line passing through point P(-1, 3) and having equal intercepts on both axes has the equation x + y = 2 --/
theorem equation_of_line_P (P : ℝ × ℝ) (hP : P = (-1, 3)) :
  ∃ (a b c : ℝ), (∀ x y : ℝ, a * x + b * y = c → x + y = 2) ∧ (a = 1 ∧ b = 1 ∧ c = 2) :=
by
  sorry

end equation_of_line_AB_equation_of_line_P_l814_814127


namespace number_of_ways_to_choose_chairs_l814_814787

def choose_chairs_equivalent (chairs : Nat) (students : Nat) (professors : Nat) : Nat :=
  let positions := (chairs - 2)  -- exclude first and last chair
  Nat.choose positions professors * Nat.factorial professors

theorem number_of_ways_to_choose_chairs : choose_chairs_equivalent 10 5 4 = 1680 :=
by
  -- The positions for professors are available from chairs 2 through 9 which are 8 positions.
  /- Calculation for choosing 4 positions out of these 8:
     C(8,4) * 4! = 70 * 24 = 1680 -/
  sorry

end number_of_ways_to_choose_chairs_l814_814787


namespace impossible_to_arrange_segments_l814_814725

-- Define a line segment as a pair of points
structure Point where
  x : ℝ
  y : ℝ

structure Segment where
  p1 : Point
  p2 : Point

-- Define the problem statement
theorem impossible_to_arrange_segments : 
  ¬ ∃ (segments : Fin 1000 → Segment), ∀ i, (∃ j, i ≠ j ∧ (segments i).p1 ∈ (interior_of_segment (segments j)) ∧ (segments i).p2 ∈ (interior_of_segment (segments j))) :=
  sorry

-- Additional definitions to support the theorem statement
def interior_of_segment (s : Segment) : Set Point := 
  { p | ∃ t : ℝ, 0 < t ∧ t < 1 ∧ p = ⟨s.p1.x + t * (s.p2.x - s.p1.x), s.p1.y + t * (s.p2.y - s.p1.y)⟩ }

end impossible_to_arrange_segments_l814_814725


namespace find_angle_l814_814233

-- Definitions based on the problem's conditions
def acute_angle (a : ℝ) : Prop := 0 < a ∧ a < π / 2
noncomputable def sin_value := real.sin

-- Theorem statement based on the question and correct answer
theorem find_angle (a : ℝ) : sin_value a = (real.sqrt 3) / 2 ∧ acute_angle a → a = real.pi / 3 := 
by
  sorry

end find_angle_l814_814233


namespace right_triangle_geom_l814_814350

theorem right_triangle_geom {ABC DEF : Type} [Triangle ABC] [Arc DEF]  
  (AB BC : ℝ) (A_center : Prop) 
  (AD : ℝ) (x : ℝ):
  (AB = 1) -> (BC = 1) -> (AD = x / (Real.sqrt Real.pi)) -> 
  (ShadedAreaEqual : (shadedArea BDE = shadedArea CEF)) -> 
  (x = 2) :=
by
  sorry

end right_triangle_geom_l814_814350


namespace perpendicular_vector_l814_814996

variables {a b : ℝ}

/-- Given that the angle between unit vectors $\overrightarrow{a}$ and $\overrightarrow{b}$ 
is $60^{\circ}$, prove that $2\overrightarrow{a} - \overrightarrow{b}$ is perpendicular to 
$\overrightarrow{b}$. -/
theorem perpendicular_vector 
  (a b : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hab : a * b = 0.5) :
  (2 * a - b) * b = 0 :=
begin
  sorry,
end

end perpendicular_vector_l814_814996


namespace sequence_length_16_l814_814192

noncomputable def c : ℕ → ℕ
| 0 := 0
| 1 := 0
| n := if n - 1 = 0 then 0 else d (n - 1)

noncomputable def d : ℕ → ℕ
| 0 := 0
| 1 := 0
| 2 := 1
| n := if n - 2 = 0 then 1 else c (n - 2) + (if n - 2 = 1 then 0 else d (n - 2))

theorem sequence_length_16 : c 16 + d 16 = 55 :=
by { sorry }

end sequence_length_16_l814_814192


namespace distinct_three_digit_numbers_with_8_l814_814664

theorem distinct_three_digit_numbers_with_8 :
  let digits := {5, 6, 7, 8}
  in ∃ n : Nat, n = 18 ∧
       ∀ number : Finset (Finset Nat),
         (∀ num ∈ number, ∃ d ∈ digits, num = d) →
         number.card = 3 →
         8 ∈ number →
         (∀ d ∈ number, (d : Nat) ∈ digits) →
         ∃! n = number,
        -- The proof goes here.
        sorry

end distinct_three_digit_numbers_with_8_l814_814664


namespace largest_number_with_digits_sum_14_l814_814834

theorem largest_number_with_digits_sum_14 : 
  ∃ n : ℕ, (∀ d ∈ (list_of_digits n), d = 2 ∨ d = 3) ∧ (list_of_digits n).sum = 14 ∧ n = 3222233 :=
by
  sorry

end largest_number_with_digits_sum_14_l814_814834


namespace domain_f_monotonicity_f_inequality_solution_l814_814287

noncomputable def f (x: ℝ) := Real.log ((1 - x) / (1 + x))

variable {x : ℝ}

theorem domain_f : ∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 -> Set.Ioo (-1 : ℝ) 1 := sorry

theorem monotonicity_f : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f y < f x := sorry

theorem inequality_solution :
  {x : ℝ | f (2 * x - 1) < 0} = {x | x > 1 / 2 ∧ x < 1} := sorry

end domain_f_monotonicity_f_inequality_solution_l814_814287


namespace largest_m_factorial_product_l814_814230

theorem largest_m_factorial_product (k : ℕ) (h : k = 2022) :
  ∃ m : ℕ, m = k! - 1 ∧ m! * k! = (k!)! :=
by
  use k! - 1
  sorry

end largest_m_factorial_product_l814_814230


namespace solve_equation_l814_814956

noncomputable def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, x = Real.arcsin (3/4) + 2 * k * Real.pi ∨ x = Real.pi - Real.arcsin (3/4) + 2 * k * Real.pi

theorem solve_equation (x : ℝ) :
  (5 * Real.sin x = 4 + 2 * Real.cos (2 * x)) ↔ solution_set x := 
sorry

end solve_equation_l814_814956


namespace minimum_product_value_l814_814761

-- Problem conditions
def total_stones : ℕ := 40
def b_min : ℕ := 20
def b_max : ℕ := 32

-- Define the product function
def P (b : ℕ) : ℕ := b * (total_stones - b)

-- Goal: Prove the minimum value of P(b) for b in [20, 32] is 256
theorem minimum_product_value : ∃ (b : ℕ), b_min ≤ b ∧ b ≤ b_max ∧ P b = 256 := by
  sorry

end minimum_product_value_l814_814761


namespace unique_mag_of_quadratic_eq_l814_814679

open Complex

theorem unique_mag_of_quadratic_eq (z : ℂ) : (z^2 - 12 * z + 157 = 0) → (∃ (c : ℕ), c = 1 ∧ ∀ w, (w^2 - 12 * w + 157 = 0) → abs w = abs z) := sorry

end unique_mag_of_quadratic_eq_l814_814679


namespace quarters_to_dimes_ratio_l814_814545

variables {pennies nickels dimes quarters : ℕ}

-- Conditions
def conditions (pennies nickels dimes quarters : ℕ) : Prop :=
  (pennies = 3 * nickels) ∧
  (nickels = 5 * dimes) ∧
  (pennies = 120) ∧
  ((120 * 1 + (nickels * 5) + (dimes * 10) + (quarters * 25)) / 100 = 8)

-- Theorem to prove the ratio of quarters to dimes is 2:1
theorem quarters_to_dimes_ratio (pennies nickels dimes quarters : ℕ) 
  (h : conditions pennies nickels dimes quarters) : quarters / dimes = 2 :=
begin
  sorry
end

end quarters_to_dimes_ratio_l814_814545


namespace terry_age_proof_l814_814697

-- Condition 1: In 10 years, Terry will be 4 times the age that Nora is currently.
-- Condition 2: Nora is currently 10 years old.
-- We need to prove that Terry's current age is 30 years old.

variable (Terry_now Terry_in_10 Nora_now : ℕ)

theorem terry_age_proof (h1: Terry_in_10 = 4 * Nora_now) (h2: Nora_now = 10) (h3: Terry_in_10 = Terry_now + 10) : Terry_now = 30 := 
by
  sorry

end terry_age_proof_l814_814697


namespace pudding_cups_minimum_addition_l814_814816

theorem pudding_cups_minimum_addition (P S m n : ℕ) (hP : P = 4752) (hS : S = 3019) (hm : m = 2) (p : ∀ d : ℕ, d ∣ 3019 → d = 1 ∨ d = 3019) :
  (n = (S * m - P)) → n = 1286 :=
by
  rw [hP, hS, hm]
  norm_num
  have n := λ n, (3019 * 2 - 4752)
  rw ←n
  norm_num
  sorry

end pudding_cups_minimum_addition_l814_814816


namespace at_least_one_angle_not_less_than_sixty_l814_814032

theorem at_least_one_angle_not_less_than_sixty (A B C : ℝ)
  (hABC_sum : A + B + C = 180)
  (hA : A < 60)
  (hB : B < 60)
  (hC : C < 60) : false :=
by
  sorry

end at_least_one_angle_not_less_than_sixty_l814_814032


namespace relationship_among_abc_l814_814235

noncomputable def a : ℝ := 3^0.1
noncomputable def b : ℝ := 2^0.1
noncomputable def c : ℝ := 0.2^1.3

theorem relationship_among_abc : a > b ∧ b > c := by
  sorry

end relationship_among_abc_l814_814235


namespace distance_to_yaxis_eq_nine_l814_814275

-- Define the parabola and its properties.
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the focus of the parabola y^2 = 4x
def focus : ℝ × ℝ := (1, 0)

-- Define the distance function between two points in ℝ²
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Point M lies on the parabola and its distance from the focus is 10
variables (M : ℝ × ℝ) (hM_parabola : parabola M.1 M.2) (hM_focus : distance M focus = 10)

-- We need to prove that the distance from M to the y-axis is 9
theorem distance_to_yaxis_eq_nine :
  distance (M.1, M.2) (0, M.2) = 9 := by
  sorry

end distance_to_yaxis_eq_nine_l814_814275


namespace dogs_in_shelter_l814_814703

theorem dogs_in_shelter (D C : ℕ) (h1 : D * 7 = 15 * C) (h2 : D * 11 = 15 * (C + 8)) :
  D = 30 :=
sorry

end dogs_in_shelter_l814_814703


namespace terry_current_age_l814_814699

theorem terry_current_age (T : ℕ) (nora_current_age : ℕ) (h1 : nora_current_age = 10)
  (h2 : T + 10 = 4 * nora_current_age) : T = 30 :=
by
  sorry

end terry_current_age_l814_814699


namespace part1_part2_l814_814621

-- Define vector a
def a : ℝ × ℝ := (3, 4)

-- Define vector b
def b : ℝ × ℝ := (-2, 1)

-- First problem: assert c coordinates
theorem part1 (c : ℝ × ℝ) (h_parallel : ∃ μ : ℝ, c = (-μ, 6 * μ)) (h_norm : ∥c∥ = √37) : 
  c = (-1, 6) ∨ c = (1, -6) :=
sorry

-- Second problem: assert range of λ for acute angle
theorem part2 (λ : ℝ) (h_acute : 0 < a.1 * (a.1 + λ * b.1) + a.2 * (a.2 + λ * b.2)) : 
  λ ∈ Set.Ico (-∞) 0 ∪ Set.Ico 0 (25 / 2) :=
sorry

end part1_part2_l814_814621


namespace tony_puzzles_l814_814828

theorem tony_puzzles (x : ℕ) (warm_up_time : ℕ) (subsequent_time : ℕ)
  (total_time : ℕ) (h_warm_up : warm_up_time = 10) 
  (h_subsequent : subsequent_time = 30) 
  (h_total : total_time = 70) :
  10 + x * 30 = 70 → x = 2 :=
by {
  intro h,
  linarith,
}

end tony_puzzles_l814_814828


namespace min_value_eight_l814_814986

def min_value (x y : ℝ) : ℝ :=
  y / x + 4 / y

theorem min_value_eight (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : 
  min_value x y = 8 :=
sorry

end min_value_eight_l814_814986


namespace congruent_triangles_angle_BDE_l814_814784

theorem congruent_triangles_angle_BDE
  (A B C D E : Type)
  [triangle ABC : triangle A B C]
  [triangle ADE : triangle A D E]
  (h1 : cong A B = cong A C)
  (h2 : cong A D = cong A E)
  (h3 : ∠ B A C = 30)
  : ∠ B D E = 15 :=

sorry

end congruent_triangles_angle_BDE_l814_814784


namespace find_unknown_towel_rate_l814_814533

theorem find_unknown_towel_rate 
    (cost_known1 : ℕ := 300)
    (cost_known2 : ℕ := 750)
    (total_towels : ℕ := 10)
    (average_price : ℕ := 150)
    (total_cost : ℕ := total_towels * average_price) :
  let total_cost_known := cost_known1 + cost_known2
  let cost_unknown := 2 * x
  300 + 750 + 2 * x = total_cost → x = 225 :=
by
  sorry

end find_unknown_towel_rate_l814_814533


namespace curve_circle_intersection_l814_814636

theorem curve_circle_intersection (k : ℝ) :
    let C := {p : ℝ × ℝ | (p.2 - k * p.1) * (p.2 - k * p.1 + 7 / 4) = 0}
    let M := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 1}
    if k < 0 ∨ k > 35 / 12 then
        C ∩ M = ∅
    else if 0 < k ∧ k < 3 / 4 then
        ∃ A B ∈ C ∩ M, A ≠ B ∧ ∀ C ∈ C ∩ M, C = A ∨ C = B
    else if k = 3 / 4 then
        ∃ A ∈ C ∩ M, ∀ B ∈ C ∩ M, B = A
    else if 3 / 4 < k ∧ k < 4 / 3 then
        ∃ A B C D ∈ C ∩ M, A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ 
        ∀ E ∈ C ∩ M, E = A ∨ E = B ∨ E = C ∨ E = D
    else
        false :=
by
  sorry

end curve_circle_intersection_l814_814636


namespace solve_fraction_l814_814843

theorem solve_fraction :
  (144^2 - 100^2) / 22 = 488 := 
by 
  sorry

end solve_fraction_l814_814843


namespace solution_set_of_inequality_l814_814809

theorem solution_set_of_inequality (x : ℝ) : 
  (2 * x - 1) / (x + 2) > 1 ↔ x < -2 ∨ x > 3 :=
by
  sorry

end solution_set_of_inequality_l814_814809


namespace original_faculty_number_l814_814529

theorem original_faculty_number (x : ℝ) (reduction : ℝ) (remaining : ℝ) 
  (hx : reduction = 0.23) (hr : remaining = 195) (h_eq : remaining = x * (1 - reduction)) : 
  x ≈ 253 := 
by
  sorry

end original_faculty_number_l814_814529


namespace fraction_is_terminating_decimal_l814_814577

noncomputable def fraction_to_decimal : ℚ :=
  58 / 160

theorem fraction_is_terminating_decimal : fraction_to_decimal = 3625 / 10000 :=
by
  sorry

end fraction_is_terminating_decimal_l814_814577


namespace asymptotes_of_hyperbola_l814_814686

-- Given conditions
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a^2 = m ∧ b^2 = 1 ∧ (c^2 = a^2 + b^2 ∧ 2 * c = 4 ∧ c = 2)

-- The proof statement
theorem asymptotes_of_hyperbola (m : ℝ) (h : is_hyperbola m) :
  (m = -3) ∧ (∀ (x : ℝ), y = (λ x, +((1 / (sqrt 3)) * x)) ∨ y = (λ x, -((1 / (sqrt 3)) * x))) :=
sorry

end asymptotes_of_hyperbola_l814_814686


namespace fraction_simplification_l814_814928

theorem fraction_simplification : 
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = (1 / 3) := 
sorry

end fraction_simplification_l814_814928


namespace remainder_5_7_9_6_3_5_mod_7_l814_814100

theorem remainder_5_7_9_6_3_5_mod_7 : (5^7 + 9^6 + 3^5) % 7 = 5 :=
by sorry

end remainder_5_7_9_6_3_5_mod_7_l814_814100


namespace solve_for_x_logarithm_l814_814406

theorem solve_for_x_logarithm :
  ∀ x : ℝ, log 3 (4 * x + 12) / (6 * x - 4) + log 3 (6 * x - 4) / (2 * x - 3) = 3 → x = 93 / 50 :=
by
  intro x
  sorry

end solve_for_x_logarithm_l814_814406


namespace campers_afternoon_l814_814129

theorem campers_afternoon (campers_morning campers_evening campers_afternoon campers_difference : ℕ)
  (h_campers_morning : campers_morning = 33)
  (h_campers_evening : campers_evening = 10)
  (h_campers_difference : campers_difference = 24)
  (h_relation : campers_afternoon = campers_evening + campers_difference) :
  campers_afternoon = 34 :=
by
  rw [h_campers_evening, h_campers_difference, h_relation]
  norm_num

end campers_afternoon_l814_814129


namespace continuous_piecewise_function_l814_814383

theorem continuous_piecewise_function (a c : ℝ) (h1 : 2 * a * 2 + 6 = 3 * 2 - 2) (h2 : 4 * (-2) + 2 * c = 3 * (-2) - 2) : 
  a + c = -1/2 := 
sorry

end continuous_piecewise_function_l814_814383


namespace largest_number_digits_sum_14_l814_814836

noncomputable def largest_number_with_sum_14 : ℕ :=
  333322

theorem largest_number_digits_sum_14:
  ∀ (ds : List ℕ), (∀ d ∈ ds, d = 2 ∨ d = 3) ∧ ds.sum = 14 →
    nat.join ds = 333322 :=
sorry

end largest_number_digits_sum_14_l814_814836


namespace circle_area_l814_814460

theorem circle_area (x y : ℝ) (h : x^2 + y^2 - 6 * x + 8 * y - 12 = 0) : 
  let area := 37 * Real.pi in 
  area = 37 * Real.pi := 
sorry

end circle_area_l814_814460


namespace mod_remainder_1287_1499_l814_814465

theorem mod_remainder_1287_1499 : (1287 * 1499) % 300 = 213 := 
by 
  sorry

end mod_remainder_1287_1499_l814_814465


namespace length_of_cube_side_l814_814074

theorem length_of_cube_side (SA : ℝ) (h₀ : SA = 600) (h₁ : SA = 6 * a^2) : a = 10 := by
  sorry

end length_of_cube_side_l814_814074


namespace find_pq_sum_l814_814268

theorem find_pq_sum : 
  let p q : ℕ in 
  (∑ k in Finset.range 36 + 1, Real.sin (4 * k)) = Real.tan (p / q * Real.pi / 180) ∧
  p.gcd q = 1 ∧ p / q < 90 → p + q = 73 :=
by sorry

end find_pq_sum_l814_814268


namespace max_value_alpha_minus_beta_l814_814269

theorem max_value_alpha_minus_beta 
  (α β : ℝ) 
  (h1 : tan α = 3 * tan β)
  (h2 : 0 ≤ β)
  (h3 : β < α)
  (h4 : α ≤ π / 2) 
  : α - β ≤ π / 6 := 
sorry

end max_value_alpha_minus_beta_l814_814269


namespace find_coefficient_a_l814_814289

noncomputable theory
open real

def parabola_vertex (a b c : ℝ) (vertex_x vertex_y : ℝ) : Prop :=
  vertex_y = a * vertex_x ^ 2 + b * vertex_x + c
  ∧ vertex_x = -b / (2 * a)

def passes_through (a b c : ℝ) (x y : ℝ) : Prop :=
  y = a * x ^ 2 + b * x + c

theorem find_coefficient_a :
  ∃ a : ℝ, ∃ b c : ℝ,
    let vertex_x := -3
    let vertex_y := 2
    let point_x := 2
    let point_y := -43
    in parabola_vertex a b c vertex_x vertex_y ∧ passes_through a b c point_x point_y ∧ a = - (9 / 5) :=
begin
  sorry
end

end find_coefficient_a_l814_814289


namespace probability_of_forming_triangle_l814_814946

noncomputable def sticks : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_triangle_combinations : List (ℕ × ℕ × ℕ) :=
  (sticks.choose 3).filter (λ t, satisfies_triangle_inequality t.1 t.2.fst t.2.snd)

theorem probability_of_forming_triangle :
  (valid_triangle_combinations.length : ℚ) / (sticks.choose 3).length = 9 / 28 :=
by
  sorry

end probability_of_forming_triangle_l814_814946


namespace piravena_total_distance_l814_814027

theorem piravena_total_distance (A B C : Point) (h_right_triangle : ∃ R : RightAngleTriangle, R.A = A ∧ R.B = B ∧ R.C = C)
  (h_AC : dist A C = 4000) (h_AB : dist A B = 4250) :
  let BC := dist B C in
  let AB := dist A B in
  let CA := dist C A in
  BC = 1582 ∧ AB = 4250 ∧ CA = 4000 ∧ (AB + BC + CA) = 9832 := by
sorry

end piravena_total_distance_l814_814027


namespace total_distance_driven_l814_814047

-- Definition for miles driven each day
def daily_mileage : ℕ := 250

-- Definition for days of travel
def total_days : ℕ := 5

-- Calculate the total miles
def total_miles : ℕ := daily_mileage * total_days 

-- The theorem to prove
theorem total_distance_driven :
  daily_mileage * total_days = 1250 :=
by
  simp [daily_mileage, total_days]
  exact eq.refl 1250

end total_distance_driven_l814_814047


namespace train_length_in_terms_of_james_cycle_l814_814727

/-- Define the mathematical entities involved: L (train length), J (James's cycle length), T (train length per cycle) -/
theorem train_length_in_terms_of_james_cycle 
  (L J T : ℝ) 
  (h1 : 130 * J = L + 130 * T) 
  (h2 : 26 * J = L - 26 * T) 
    : L = 58 * J := 
by 
  sorry

end train_length_in_terms_of_james_cycle_l814_814727


namespace length_of_CP_l814_814779

theorem length_of_CP
  (ABCD_is_square : ∀ (A B C D : ℝ), A = 5 ∧ B = 5 ∧ C = 5 ∧ D = 5) 
  (CP_bisects_AC : ∀ (A C P : ℝ × ℝ), P = ((A + C) / 2, (A + C) / 2)) 
  (CQ_bisects_CD : ∀ (C D Q : ℝ × ℝ), Q = (C, (D + D) / 2)) :
  ∀ (C P : ℝ × ℝ), dist C P = (5 * Real.sqrt 2) / 2 :=
by
  sorry

end length_of_CP_l814_814779


namespace find_correct_function_l814_814174

noncomputable def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

noncomputable def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f x = f (x + T)

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x < f y

noncomputable def func_A : ℝ → ℝ := λ x, Real.sin x
noncomputable def func_B : ℝ → ℝ := λ x, Real.sin (2 * |x|)
noncomputable def func_C : ℝ → ℝ := λ x, -Real.cos (2 * x)
noncomputable def func_D : ℝ → ℝ := λ x, Real.cos (2 * x)

theorem find_correct_function :
  is_even func_C ∧ is_periodic func_C (π) ∧ is_monotonically_increasing func_C {x | 0 < x ∧ x < π / 4} :=
begin
  sorry
end

end find_correct_function_l814_814174


namespace significant_digits_of_square_side_length_l814_814802

theorem significant_digits_of_square_side_length :
  ∀ (A : ℝ), A = 3.2416 → (number_of_significant_digits (√A) = 4) :=
by
  intros A hA
  -- Definitions and assumptions (skipped)
  sorry

end significant_digits_of_square_side_length_l814_814802


namespace boys_meet_time_is_correct_l814_814456

structure TrackMeetProblem where
  (track_length : ℕ) -- Track length in meters
  (speed_first_boy_kmh : ℚ) -- Speed of the first boy in km/hr
  (speed_second_boy_kmh : ℚ) -- Speed of the second boy in km/hr

noncomputable def time_to_meet (p : TrackMeetProblem) : ℚ :=
  let speed_first_boy_ms := (p.speed_first_boy_kmh * 1000) / 3600
  let speed_second_boy_ms := (p.speed_second_boy_kmh * 1000) / 3600
  let relative_speed := speed_first_boy_ms + speed_second_boy_ms
  (p.track_length : ℚ) / relative_speed

theorem boys_meet_time_is_correct (p : TrackMeetProblem) : 
  p.track_length = 4800 → 
  p.speed_first_boy_kmh = 61.3 → 
  p.speed_second_boy_kmh = 97.5 → 
  time_to_meet p = 108.8 := by
  intros
  sorry  

end boys_meet_time_is_correct_l814_814456


namespace sum_of_squares_of_medians_is_correct_l814_814842

-- Defining the triangle with given side lengths and condition of being Isosceles
variable (A B C : Type*)
variable [MetricSpace A]
variable [MetricSpace B]
variable [MetricSpace C]

def isosceles_triangle (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] :=
  dist A B = 10 ∧ dist A C = 10 ∧ dist B C = 12

-- Statement: Prove the sum of the squares of the lengths of the medians
theorem sum_of_squares_of_medians_is_correct (T : Type) [MetricSpace T] (A B C : T)
  (h : isosceles_triangle A B C) : 
  let D : T := midpoint ℝ A C,
      E : T := midpoint ℝ A B,
      F : T := midpoint ℝ B C in
  let AD_m := dist A D
      BE_m := dist B E
      CF_m := dist C F in
  (AD_m^2 + BE_m^2 + CF_m^2 = 258) := 
sorry -- Proof of the theorem is omitted for brevity

end sum_of_squares_of_medians_is_correct_l814_814842


namespace A_share_is_correct_l814_814171

-- Define initial conditions and investments
variables (x : ℝ) -- A's initial investment
variables (total_profit : ℝ) (A_share : ℝ) (B_share : ℝ) (C_share : ℝ)
variable (profit_share : ℝ)

-- Given conditions
def investment_at_start (x : ℝ) := x
def investment_after_6_months (x : ℝ) := 2 * x
def investment_after_8_months (x : ℝ) := 3 * x
def annual_profit := 18900.0

-- Time periods for which the investments are made
def time_period_a : ℝ := 12
def time_period_b : ℝ := 6
def time_period_c : ℝ := 4

-- Calculated shares based on contributions
def share_of_A (x : ℝ) : ℝ := x * time_period_a
def share_of_B (x : ℝ) : ℝ := (2 * x) * time_period_b
def share_of_C (x : ℝ) : ℝ := (3 * x) * time_period_c

-- Total contributions
def total_contributions (x : ℝ) : ℝ := share_of_A x + share_of_B x + share_of_C x

-- Fraction of the profits corresponding to each contribution
def profit_fraction (x : ℝ) : ℝ := total_profit / total_contributions x

-- A's share of the total profit
def A_share_from_profit (x : ℝ) : ℝ := share_of_A x * profit_fraction x

-- The final proof statement we need to ensure
theorem A_share_is_correct {x : ℝ} (h : total_profit = annual_profit) :
  A_share_from_profit x = 6300 := by
  sorry

end A_share_is_correct_l814_814171


namespace number_of_distinct_right_angled_triangles_l814_814198

-- Definitions and conditions based on the problem
def isIntegerLength (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def hasArea999TimesPerimeter (a b c : ℕ) : Prop :=
  2 * a * b = 999 * (a + b + c)

-- Core theorem statement
theorem number_of_distinct_right_angled_triangles : ∃ n : ℕ, n = 90 ∧ (
  ∀ (a b c : ℕ), (isIntegerLength a b c ∧ hasArea999TimesPerimeter a b c) → 
  (a = b → a = c)) 
:= 
begin
  -- Proof skipped
  sorry
end

end number_of_distinct_right_angled_triangles_l814_814198


namespace part1_daily_sales_profit_part2_maximum_daily_profit_l814_814413

-- Definitions of initial conditions
def original_price : ℝ := 30
def original_sales_volume : ℝ := 60
def cost_price : ℝ := 15
def price_reduction_effect : ℝ := 10

-- Part 1: Prove the daily sales profit if the price is reduced by 2 yuan
def new_price_after_reduction (reduction : ℝ) : ℝ := original_price - reduction
def new_sales_volume (reduction : ℝ) : ℝ := original_sales_volume + reduction * price_reduction_effect
def profit_per_kg (selling_price : ℝ) : ℝ := selling_price - cost_price
def daily_sales_profit (reduction : ℝ) : ℝ := profit_per_kg (new_price_after_reduction reduction) * new_sales_volume reduction

theorem part1_daily_sales_profit : daily_sales_profit 2 = 1040 := by sorry

-- Part 2: Prove the selling price for maximum profit and the maximum profit
def profit_function (x : ℝ) : ℝ := (x - cost_price) * (original_sales_volume + (original_price - x) * price_reduction_effect)

theorem part2_maximum_daily_profit : 
  ∃ x, profit_function x = 1102.5 ∧ x = 51 / 2 := by sorry

end part1_daily_sales_profit_part2_maximum_daily_profit_l814_814413


namespace value_depletion_rate_l814_814138

-- Define the given conditions
constant InitialValue : ℝ := 800
constant FinalValue : ℝ := 648
constant Time : ℕ := 2

-- The equation modeling the exponential decay
def decay_equation (r : ℝ) : Prop :=
  FinalValue = InitialValue * (1 - r)^Time

-- The theorem to be proven
theorem value_depletion_rate : ∃ r : ℝ, decay_equation r ∧ r = 0.1 :=
by 
  sorry

end value_depletion_rate_l814_814138


namespace sum_first_2017_b_l814_814975

noncomputable def geometric_sequence := ℕ → ℝ

-- Define the geometric sequence a_n with conditions
def a (n : ℕ) : ℝ := sorry

-- The definition of the b_n sequence
def b (n : ℕ) : ℝ := log (sqrt 3) (3 * a n)

theorem sum_first_2017_b :
  let S (n : ℕ) := ∑ i in finset.range n, b (i + 1)
  S 2017 = 2017 * 1011 := sorry

end sum_first_2017_b_l814_814975


namespace lcm_mod_100_example_l814_814119

noncomputable def lcm (a b : ℕ) : ℕ :=
(a * b) / Nat.gcd a b

theorem lcm_mod_100_example :
    let p := 2^4 * 3^2
    let q := 2^12
    let r := 2^2 * 3^4
    let s := 2^9 * 3^3
    let l1 := lcm p q
    let l2 := lcm r s
    let result := lcm l1 l2
    (result % 100) = 24 := by
    sorry

end lcm_mod_100_example_l814_814119


namespace eq_root_condition_l814_814684

theorem eq_root_condition (m : ℚ) : (5^2 - 4 * 2 * m = 0) → (m = 25/8) :=
by
  intro h
  have : 25 - 8 * m = 0 := by exact h
  sorry

end eq_root_condition_l814_814684


namespace A_shoots_second_P_n_explicit_P_n_l814_814089

variables (P : ℕ → ℚ)
variables (pA1 pB1 pASuccess pBSuccess : ℚ)
variables (n : ℕ)

-- Given initial probabilities
def pA1 : ℚ := 2 / 3
def pB1 : ℚ := 1 / 3
def pASuccess : ℚ := 7 / 10
def pBSuccess : ℚ := 4 / 5

-- Given the problem's derived probability for second shot by A
theorem A_shoots_second : 
  ((pA1 * pASuccess) + (pB1 * (1 - pBSuccess)) = 8 / 15) := 
sorry

-- Recursive relationship
theorem P_n (Pn Pn1 : ℚ):
  P (n+1) = 1/2 * P n + 1/5 := 
sorry

-- Explicit formula solution
theorem explicit_P_n :
  P n = 2 / 5 + (1 / 15) * (1 / 2)^(n - 3) := 
sorry

end

end A_shoots_second_P_n_explicit_P_n_l814_814089


namespace inequality_with_xy_l814_814606

theorem inequality_with_xy
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) :
  (1 / (x + 3)) + (1 / (y + 3)) ≤ 2 / 5 :=
sorry

end inequality_with_xy_l814_814606


namespace candle_burn_at_11am_l814_814457

noncomputable def candle_burn_time :=
  ∀ {ℓ1 ℓ2 : ℝ},
    let burn_rate1 := ℓ1 / 300 in
    let burn_rate2 := ℓ2 / 360 in
    let t := 1800 in
    ℓ2 - burn_rate2 * t = 3 * (ℓ1 - burn_rate1 * t) →
    t = 1800

-- Statement:
theorem candle_burn_at_11am (ℓ1 ℓ2 : ℝ) :
  let burn_rate1 := ℓ1 / 300 in
  let burn_rate2 := ℓ2 / 360 in
  let time_lit_minutes := 11 * 60 in -- Convert 11:00 AM to minutes.
  let t := 6 * 60 in -- Time at 5:00 PM in minutes.
  let elapsed_time := t + 6 * 60 in -- Time elapsed from 11:00 AM to 5:00 PM.
  elapsed_time = 1800 →
  ℓ2 - burn_rate2 * elapsed_time = 3 * (ℓ1 - burn_rate1 * elapsed_time) :=
sorry

end candle_burn_at_11am_l814_814457


namespace solve_for_star_l814_814118

theorem solve_for_star : ∀ (star : ℝ), (45 - (28 - (37 - (15 - star))) = 54) → star = 15 := by
  intros star h
  sorry

end solve_for_star_l814_814118


namespace abundant_numbers_less_than_50_eq_9_l814_814305

def proper_factors (n : ℕ) : List ℕ :=
  List.filter (λ m, m < n ∧ n % m = 0) (List.range n)

def sum_proper_factors (n : ℕ) : ℕ :=
  (proper_factors n).sum

def is_abundant (n : ℕ) : Prop :=
  sum_proper_factors n > n

def abundant_numbers_count_below (m : ℕ) : ℕ :=
  (List.range m).filter is_abundant |>.length

theorem abundant_numbers_less_than_50_eq_9 :
  abundant_numbers_count_below 50 = 9 :=
by
  sorry

end abundant_numbers_less_than_50_eq_9_l814_814305


namespace dealer_profit_percentage_is_correct_l814_814885

noncomputable def dealer_net_profit_percentage (cost_price weight_percentage impurity_percentage : ℝ) : ℝ :=
  let real_quantity := weight_percentage * (1 - impurity_percentage) in
  ((cost_price - real_quantity * cost_price) / (real_quantity * cost_price) * 100)

theorem dealer_profit_percentage_is_correct :
  dealer_net_profit_percentage 100 0.8 0.25 = 66.67 := 
by
  sorry

end dealer_profit_percentage_is_correct_l814_814885


namespace local_value_proof_l814_814463

-- Definitions based on the conditions
def face_value_7 : ℕ := 7
def local_value_6_in_7098060 : ℕ := 6000
def product_of_face_value_and_local_value : ℕ := face_value_7 * local_value_6_in_7098060
def local_value_6_in_product : ℕ := 6000

-- Theorem statement
theorem local_value_proof : local_value_6_in_product = 6000 :=
by
  -- Direct restatement of the condition in Lean
  sorry

end local_value_proof_l814_814463


namespace area_of_given_trapezium_l814_814853

def area_of_trapezium (a b h : ℕ) : ℕ :=
  (1 / 2) * (a + b) * h

theorem area_of_given_trapezium :
  area_of_trapezium 20 18 25 = 475 :=
by
  sorry

end area_of_given_trapezium_l814_814853


namespace distance_between_parallel_lines_l814_814651

theorem distance_between_parallel_lines :
  ∀ (x y : ℝ) (a : ℝ),
      (l1 : x + a * y = 1) → (l2 : a * x + y = 1) → 
      (a = -1) → 
      (distance l1 l2 = √2) :=
by
  -- Assumptions and conditions 
  sorry

end distance_between_parallel_lines_l814_814651


namespace fibers_length_less_than_20_l814_814454

def fibers_length_distribution : ℝ → ℝ
| x := if x < 20 then 0.006 else 0 -- Simplified example, real histogram should be more complex

theorem fibers_length_less_than_20 (n : ℕ) (len_intvl : ℝ) (freqs : list ℝ) :
  n = 100 →
  len_intvl = 5 →
  freqs = [0.001, 0.001, 0.004] →
  (len_intvl * n * (freqs.sum)) = 30 :=
by
  intros hn hintvl hfreqs
  rw [hn, hintvl, hfreqs]
  norm_num
  sorry

end fibers_length_less_than_20_l814_814454


namespace third_function_symmetric_l814_814297

variable {α : Type} [LinearOrderedField α]

/-- Given a function ϕ and its inverse ϕ⁻¹, the third function, 
    which is symmetric to ϕ⁻¹ about the line x + y = 0, is -ϕ⁻¹(-x). -/
theorem third_function_symmetric (ϕ : α → α) (h_ϕ_inverse : Function.RightInverse ϕ⁻¹ ϕ) :
  ∀ x : α, ∃ y : α, y = -ϕ⁻¹(-x) := by
  sorry

end third_function_symmetric_l814_814297


namespace smallest_n_int_l814_814190

def x_seq : ℕ → ℝ
| 0       := real.cbrt 3
| (n + 1) := (x_seq n) ^ (real.cbrt 3)

theorem smallest_n_int (n : ℕ) : x_seq n = 3 := sorry

end smallest_n_int_l814_814190


namespace not_possible_to_get_105_single_stone_piles_l814_814821

noncomputable def piles : List Nat := [51, 49, 5]
def combine (a b : Nat) : Nat := a + b
def split (a : Nat) : List Nat := if a % 2 = 0 then [a / 2, a / 2] else [a]

theorem not_possible_to_get_105_single_stone_piles 
  (initial_piles : List Nat := piles) 
  (combine : Nat → Nat → Nat := combine) 
  (split : Nat → List Nat := split) :
  ¬ ∃ (final_piles : List Nat), final_piles.length = 105 ∧ (∀ n ∈ final_piles, n = 1) :=
by
  sorry

end not_possible_to_get_105_single_stone_piles_l814_814821


namespace family_reunion_cost_l814_814224

theorem family_reunion_cost :
  let people := 5 * 12
  let cans_per_person := 2
  let cans_needed := people * cans_per_person
  let cans_per_box := 10
  let boxes_needed := cans_needed / cans_per_box
  let cost_per_box := 2
  let total_cost := boxes_needed * cost_per_box
  let family_members := 6
  let cost_per_member := total_cost / family_members
  cost_per_member = 4 := by
{
  let people := 5 * 12
  let cans_per_person := 2
  let cans_needed := people * cans_per_person
  let cans_per_box := 10
  let boxes_needed := cans_needed / cans_per_box
  let cost_per_box := 2
  let total_cost := boxes_needed * cost_per_box
  let family_members := 6
  let cost_per_member := total_cost / family_members
  show cost_per_member = 4, from
    sorry
}

end family_reunion_cost_l814_814224


namespace sequence_eval_l814_814152

def seq (i : ℕ) : ℕ :=
if i = 1 then 1 else if i = 2 then 2 else (finset.prod (finset.range (i - 1)) seq) - 1

theorem sequence_eval :
  let prod := finset.prod (finset.range 100) seq
  let sum_sq := finset.sum (finset.range 100) (λ i, (seq i) ^ 2)
  prod - sum_sq = -101 :=
by sorry

end sequence_eval_l814_814152


namespace find_a4_l814_814436

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

axiom hyp1 : is_arithmetic_sequence a d
axiom hyp2 : a 5 = 9
axiom hyp3 : a 7 + a 8 = 28

-- Goal
theorem find_a4 : a 4 = 7 :=
by
  sorry

end find_a4_l814_814436


namespace theater_total_seats_l814_814854

theorem theater_total_seats :
  (∃ n : ℕ, (15 + (n - 1) * 2 = 53) ∧ (∑ k in finset.range n, (15 + k * 2)) = 680) :=
begin
  sorry
end

end theater_total_seats_l814_814854


namespace find_min_value_l814_814953

theorem find_min_value : ∀ (x : Real), 0 < x ∧ x < π / 2 → (tan x + cot x)^2 + (sin x + cos x)^2 = 6 :=
by
  sorry

end find_min_value_l814_814953


namespace salary_C_more_than_A_ratio_salary_E_to_A_and_B_l814_814807

variable (x : ℝ)
variables (salary_A salary_B salary_C salary_D salary_E combined_salary_BCD : ℝ)

-- Conditions
def conditions : Prop :=
  salary_B = 2 * salary_A ∧
  salary_C = 3 * salary_A ∧
  salary_D = 4 * salary_A ∧
  salary_E = 5 * salary_A ∧
  combined_salary_BCD = 15000 ∧
  combined_salary_BCD = salary_B + salary_C + salary_D

-- Statements to prove
theorem salary_C_more_than_A
  (cond : conditions salary_A salary_B salary_C salary_D salary_E combined_salary_BCD) :
  (salary_C - salary_A) / salary_A * 100 = 200 := by
  sorry

theorem ratio_salary_E_to_A_and_B
  (cond : conditions salary_A salary_B salary_C salary_D salary_E combined_salary_BCD) :
  salary_E / (salary_A + salary_B) = 5 / 3 := by
  sorry

end salary_C_more_than_A_ratio_salary_E_to_A_and_B_l814_814807


namespace find_g_l814_814619

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x^2 - 3^(-x) else g x + 1

theorem find_g (g : ℝ → ℝ) (x : ℝ) (h_odd : ∀ x, f (-x) = -f (x)) (h_def : f x = if x < 0 then x^2 - 3^(-x) else g x + 1) :
  x > 0 → g x = -x^2 + 3^x - 1 :=
sorry

end find_g_l814_814619


namespace plane_split_into_four_regions_l814_814562

theorem plane_split_into_four_regions (x y : ℝ) : 
  (y = 3 * x ∨ y = (1 / 3) * x) → 
  let lines_intersect_and_divide_plane_into_four_regions := 
    ∀ (p : ℝ × ℝ), 
      (p.1 = 0 ∧ p.2 ≠ 0) ∨ 
      (p.2 = 3 * p.1) ∨ 
      (p.2 = (1 / 3) * p.1) ∨ 
      (p.1 ≠ 0 ∧ p.2 ≠ 3 * p.1 ∧ p.2 ≠ (1 / 3) * p.1) in
    lines_intersect_and_divide_plane_into_four_regions

end plane_split_into_four_regions_l814_814562


namespace salt_amount_evaporation_l814_814359

-- Define the conditions as constants
def total_volume : ℕ := 2 -- 2 liters
def salt_concentration : ℝ := 0.2 -- 20%

-- The volume conversion factor from liters to milliliters.
def liter_to_ml : ℕ := 1000

-- Define the statement to prove
theorem salt_amount_evaporation : total_volume * (salt_concentration * liter_to_ml) = 400 := 
by 
  -- We'll skip the proof steps here
  sorry

end salt_amount_evaporation_l814_814359


namespace oil_output_per_capita_l814_814866

theorem oil_output_per_capita 
  (total_oil_output_russia : ℝ := 13737.1 * 100 / 9)
  (population_russia : ℝ := 147)
  (population_non_west : ℝ := 6.9)
  (oil_output_non_west : ℝ := 1480.689)
  : 
  (55.084 : ℝ) = 55.084 ∧ 
    (214.59 : ℝ) = (1480.689 / 6.9) ∧ 
    (1038.33 : ℝ) = (total_oil_output_russia / population_russia) :=
by
  sorry

end oil_output_per_capita_l814_814866


namespace repeating_decimal_sum_l814_814181

theorem repeating_decimal_sum :
  let a := (2 : ℚ) / 3
  let b := (2 : ℚ) / 9
  let c := (4 : ℚ) / 9
  a + b - c = (4 : ℚ) / 9 :=
by
  sorry

end repeating_decimal_sum_l814_814181


namespace sin_sum_positive_l814_814768

theorem sin_sum_positive (α : ℝ) (h1 : 0 < α) (h2 : α < π) : 
  sin α + (1/2) * sin (2 * α) + (1/3) * sin (3 * α) > 0 :=
sorry

end sin_sum_positive_l814_814768


namespace problem_b_capital_contribution_l814_814479

theorem problem_b_capital_contribution 
  (months_to_b_joined : ℕ) (a_initial_investment : ℝ) (total_months : ℕ)
  (profit_ratio : ℝ) (months_b_invested : ℕ) :
  months_to_b_joined = 5 → a_initial_investment = 3500 →
  total_months = 12 → profit_ratio = 2/3 →
  months_b_invested = total_months - months_to_b_joined →
  let a_investment := a_initial_investment * total_months;
      a_ratio := 2;
      b_ratio := 3 in
  ∃ (b_contribution : ℝ), b_contribution * months_b_invested = (a_investment * b_ratio) / a_ratio ∧ b_contribution = 9000 :=
by
  sorry

end problem_b_capital_contribution_l814_814479


namespace find_three_digit_number_l814_814584

theorem find_three_digit_number : 
  ∃ x : ℕ, (x >= 100 ∧ x < 1000) ∧ (2 * x = 3 * x - 108) :=
by
  have h : ∀ x : ℕ, 100 ≤ x → x < 1000 → 2 * x = 3 * x - 108 → x = 108 := sorry
  exact ⟨108, by sorry⟩

end find_three_digit_number_l814_814584


namespace min_value_expr_l814_814212

theorem min_value_expr : ∀ x : ℝ, (x^2 + 8) / Real.sqrt (x^2 + 4) ≥ 4 ∧ ((x = 0) → (x^2 + 8)/Real.sqrt (x^2 + 4) = 4) := by
  intro x
  split
  sorry
  sorry

end min_value_expr_l814_814212


namespace correct_calculation_is_B_l814_814475

theorem correct_calculation_is_B :
  (- (- (1 : ℚ) / 3) ≠ - (1 : ℚ) / 3) ∧
  (-3 - 2 = -5) ∧
  ((- (3 : ℚ) / 2) / (4 / 3) ≠ -2) ∧
  (2 ^ 3 ≠ 6) :=
by
  split
  · simp [neg_div, div_neg, one_div] -- proof of A
  split
  · exact (rfl : -3 - 2 = -5) -- proof of B
  split
  · simp [div_eq_mul_inv, neg_div, one_div] -- proof of C
  · simp -- proof of D

end correct_calculation_is_B_l814_814475


namespace count_abundant_numbers_less_than_50_l814_814317

open_locale classical

def is_abundant (n : ℕ) : Prop :=
  ∑ m in finset.filter (λ m, m ∣ n ∧ m < n) (finset.range n), m > n

theorem count_abundant_numbers_less_than_50 :
  (finset.filter (λ n, is_abundant n) (finset.range 50)).card = 9 :=
by {
  sorry
}

end count_abundant_numbers_less_than_50_l814_814317


namespace greatest_distance_is__l814_814950

noncomputable def distance (t : ℝ) : ℝ :=
  Real.sqrt (25 + (Real.sin t - Real.cos t)^2)

theorem greatest_distance_is_√29 :
  ∃ t : ℝ, distance t = Real.sqrt 29 :=
  sorry

end greatest_distance_is__l814_814950


namespace sum_of_g_79_l814_814743

def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 1
def g (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem sum_of_g_79 :
  let x1 := (-3 + sqrt 1289) / 8
  let x2 := (-3 - sqrt 1289) / 8
  g f x1 + g f x2 = g 79 :=
sorry

end sum_of_g_79_l814_814743


namespace gift_items_l814_814459

theorem gift_items (x y z : ℕ) : 
  x + y + z = 20 ∧ 60 * x + 50 * y + 10 * z = 720 ↔ 
  ((x = 4 ∧ y = 8 ∧ z = 8) ∨ (x = 8 ∧ y = 3 ∧ z = 9)) :=
by sorry

end gift_items_l814_814459


namespace correct_average_l814_814120

theorem correct_average (n : ℕ) (average incorrect correct : ℕ) (h1 : n = 10) (h2 : average = 15) 
(h3 : incorrect = 26) (h4 : correct = 36) :
  (n * average - incorrect + correct) / n = 16 :=
  sorry

end correct_average_l814_814120


namespace valid_number_of_beads_l814_814167

def Shape := ℕ -- simple definition for the shapes: cubes, pyramids
def Color := ℕ -- simple definition for the colors: green, blue, red

-- Definitions for our specific beads
def Kg : Shape × Color := (1, 1) -- green cube
def Kb : Shape × Color := (1, 2) -- blue cube
def Pr : Shape × Color := (2, 3) -- red pyramid
def Pb : Shape × Color := (2, 2) -- blue pyramid

-- Conditions given in the problem
def is_valid_necklace (necklace : List (Shape × Color)) : Prop :=
  ∀ i, (necklace.nth i ≠ none) →
      -- different shapes and different colors condition
      let ⟨(s1, c1)⟩ := necklace.get (i % necklace.length);
      let ⟨(s2, c2)⟩ := necklace.get ((i + 1) % necklace.length);
      s1 ≠ s2 ∧ c1 ≠ c2

-- Prove that the necklace must have a valid even count of beads and specifically 8 or 10.
theorem valid_number_of_beads : 
  ∀ (necklace : List (Shape × Color)), 
  is_valid_necklace necklace →
  (necklace.length = 8 ∨ necklace.length = 10) :=
by
  intros necklace h
  sorry

end valid_number_of_beads_l814_814167


namespace find_fraction_l814_814321

def f (x : ℤ) : ℤ := 3 * x + 4
def g (x : ℤ) : ℤ := 4 * x - 3

theorem find_fraction :
  (f (g (f 2)):ℚ) / (g (f (g 2)):ℚ) = 115 / 73 := by
  sorry

end find_fraction_l814_814321


namespace distance_center_circle_to_line_l814_814424

theorem distance_center_circle_to_line : 
  let center_of_circle := (0,0)
  let line_a := 1
  let line_b := 2
  let line_c := -5
  let distance := Real.abs (line_a * center_of_circle.1 + line_b * center_of_circle.2 + line_c) / Real.sqrt (line_a^2 + line_b^2)
  distance = Real.sqrt 5 :=
by
  let center_of_circle := (0,0)
  let line_a := 1
  let line_b := 2
  let line_c := -5
  let distance := Real.abs (line_a * center_of_circle.1 + line_b * center_of_circle.2 + line_c) / Real.sqrt (line_a^2 + line_b^2)
  have calc_distance : distance = Real.sqrt 5 := sorry
  exact calc_distance

end distance_center_circle_to_line_l814_814424


namespace constant_d_satisfaction_l814_814935

theorem constant_d_satisfaction {m : ℝ} :
  let d := 1 / 4
  ∀ (x₁ x₂ : ℝ), 
    (x₁^4 - m * x₁ - d = 0) ∧ 
    (x₂^4 - m * x₂ - d = 0) →
    let A := (x₁, m * x₁ + d)
    let B := (x₂, m * x₂ + d)
    let AC² := x₁^2 + (m * x₁)^2
    let BC² := x₂^2 + (m * x₂)^2
    (1 / AC²) + (1 / BC²) = 0
by
  sorry

end constant_d_satisfaction_l814_814935


namespace general_term_formula_sum_of_transformed_sequence_l814_814246

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a d : α) (n : ℕ) : α := a + d * (n - 1)

noncomputable def a1 : α := 1
noncomputable def d : α := 2

theorem general_term_formula :
  ∀ n : ℕ, arithmetic_sequence a1 d n = 2 * n - 1 :=
by sorry

def transformed_sequence (a d : α) (n : ℕ) : α :=
  (arithmetic_sequence a d (n + 1) - 2 * arithmetic_sequence a d n) / (2 ^ (n + 1))

theorem sum_of_transformed_sequence (n : ℕ) :
  let T (n : ℕ) := ∑ i in finset.range n, transformed_sequence a1 d i in
    T n = (2 * n + 1) / 2^(n + 1) - 1/2 :=
by sorry

end general_term_formula_sum_of_transformed_sequence_l814_814246


namespace zoo_problem_l814_814902

theorem zoo_problem :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := monkeys - 35
  elephants - zebras = 3 :=
by
  sorry

end zoo_problem_l814_814902


namespace coefficient_x2_in_expansion_l814_814197

theorem coefficient_x2_in_expansion (c : ℝ) :
  let p := (cx^3 + 4*x^2 - 3*x + 5) * (2*x^3 - 5*x^2 + 6*x - 7)
  in p.coeff 2 = -71 :=
by
  sorry

end coefficient_x2_in_expansion_l814_814197


namespace find_m_l814_814892

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (p q : V)

theorem find_m (hpq : p ≠ q) : ∃ m : ℝ, m = 1/3 ∧ (m • p + (2/3) • q) ∈ line_through p q :=
by {
  let m := 1/3,
  use m,
  split,
  { norm_num },
  { sorry }
}

end find_m_l814_814892


namespace total_fish_combined_l814_814760

variable {x : ℝ}
variables (Harry Joe Sam Max : ℝ)

-- Conditions translated as Lean definitions
def conditions : Prop :=
  Joe = 9.5 * Sam ∧
  Sam = 3.25 * Max ∧
  Max = x ∧
  Harry = 5.5 * Joe

-- Theorem translating the question and the correct answer
theorem total_fish_combined (h : conditions Harry Joe Sam Max) :
  Harry + Joe + Sam + Max = 204.9375 * x :=
by
  obtain ⟨hJoeSam, hSamMax, hMax, hHarryJoe⟩ := h
  sorry

end total_fish_combined_l814_814760


namespace div_conditions_l814_814736

theorem div_conditions (p q n : ℤ) (hp : Prime p) (hq : Prime q) (hp_odd : p % 2 = 1) (hq_odd : q % 2 = 1) (h_diff : p ≠ q)
  (h1 : p * q ∣ n ^ (p * q) + 1) (h2 : p ^ 3 * q ^ 3 ∣ n ^ (p * q) + 1) :
  p ^ 2 ∣ n + 1 ∨ q ^ 2 ∣ n + 1 := 
sorry

end div_conditions_l814_814736


namespace tripod_problem_l814_814901

theorem tripod_problem :
  let m := 180
  let n := 1585
  let h := m / Real.sqrt n
  in ⌊ m + Real.sqrt n ⌋ = 219 :=
by 
  let m := 180
  let n := 1585
  let h := m / Real.sqrt n
  have sqrt_n_floor : ⌊Real.sqrt n⌋ = 39 := sorry
  calc
    ⌊ m + Real.sqrt n ⌋ 
      = ⌊180 + Real.sqrt 1585⌋ : by refl
  ... = 180 + 39 : by rw sqrt_n_floor
  ... = 219    : by norm_num

end tripod_problem_l814_814901


namespace cars_meet_time_l814_814858

theorem cars_meet_time 
  (L : ℕ) (v1 v2 : ℕ) (t : ℕ)
  (H1 : L = 333)
  (H2 : v1 = 54)
  (H3 : v2 = 57)
  (H4 : v1 * t + v2 * t = L) : 
  t = 3 :=
by
  -- Insert proof here
  sorry

end cars_meet_time_l814_814858


namespace abs_neg_three_l814_814871

theorem abs_neg_three : abs (-3) = 3 :=
by 
  sorry

end abs_neg_three_l814_814871


namespace continued_fraction_l814_814683

theorem continued_fraction {w x y : ℕ} (hw : 0 < w) (hx : 0 < x) (hy : 0 < y)
  (h_eq : (97:ℚ) / 19 = w + 1 / (x + 1 / y)) : w + x + y = 16 :=
sorry

end continued_fraction_l814_814683


namespace sum_of_second_largest_and_smallest_is_22_l814_814076

def numbers : List ℕ := [10, 11, 12, 13]

def smallest (l : List ℕ) : ℕ := List.minimum l
def second_largest (l : List ℕ) : ℕ := l.erase (List.maximum l) |> List.maximum

theorem sum_of_second_largest_and_smallest_is_22 :
  smallest numbers + second_largest numbers = 22 :=
by
  sorry

end sum_of_second_largest_and_smallest_is_22_l814_814076


namespace probability_three_white_balls_l814_814499

noncomputable def probability_all_white (white black total_drawn : ℕ) : ℚ :=
  (nat.choose white total_drawn : ℚ) / (nat.choose (white + black) total_drawn : ℚ)

theorem probability_three_white_balls :
  probability_all_white 7 8 3 = 1 / 13 :=
by 
  sorry

end probability_three_white_balls_l814_814499


namespace min_sum_xy_l814_814262

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (hcond : ↑(1 / x) + ↑(1 / y) = 1 / 15) : x + y = 64 :=
sorry

end min_sum_xy_l814_814262


namespace problem_parabola_line_intersection_l814_814291

theorem problem_parabola_line_intersection 
  (focus : ℝ × ℝ)
  (parabola : ℝ → ℝ)
  (line : ℝ → ℝ)
  (A B : ℝ × ℝ)
  (h1 : focus = (0, 1))
  (h2 : parabola x = x^2 / 4)
  (h3 : ∀ y, line y = √3 * (y - 1))
  (h4 : ∃ (x1 y1 : ℝ), A = (x1, y1) ∧ y1 = 3)
  (h5 : ∃ (x2 y2 : ℝ), B = (x2, y2) ∧ y2 = 1 / 3)
  (h6 : |A.1|^2 + (A.2 - 1)^2 = (A.2 + 1)^2)
  (h7 : |B.1|^2 + (B.2 - 1)^2 = (B.2 + 1)^2) :
  |A.2 - 1| / |B.2 - 1| = 3 :=
by
  sorry

end problem_parabola_line_intersection_l814_814291


namespace constant_term_in_expansion_l814_814969

theorem constant_term_in_expansion :
  (let n := ∫ x in 0..2, x^3 in
  ∑ r in Finset.range (n + 1), binomial n r * (-2)^r * (1 : ℝ)^(n - r) = -32) :=
by
  let n := (∫ x in (0:ℝ)..(2:ℝ), x^3).toReal
  calc
    n = 4 : by sorry
  ∑ r in Finset.range (n + 1), binomial n r * (-2 : ℝ)^r * 1^(n - r) = (-2)^3 * binomial n 3 : by sorry
  (-2)^3 * binomial 4 3 = -8 * 4 : by sorry
  -8 * 4 = -32 : by sorry

end constant_term_in_expansion_l814_814969


namespace a_n_inequality_l814_814068

-- Definition of the sequence a_n
def a : ℕ → ℕ
| 0     := 1  -- Note: Lean sequences typically start from index 0, adjust accordingly
| (n+1) := if n+1 ∈  Set.range a then a n + 3 else a n + 2

-- Theorem to prove the inequality for all positive integers
theorem a_n_inequality (n : ℕ) (h_pos : n > 0) : a n < n * (1 + Real.sqrt 2) :=
by
  sorry

end a_n_inequality_l814_814068


namespace pool_one_quarter_capacity_at_6_l814_814446

-- Variables and parameters
variables (volume : ℕ → ℝ) (T : ℕ)

-- Conditions
def doubles_every_hour : Prop :=
  ∀ t, volume (t + 1) = 2 * volume t

def full_capacity_at_8 : Prop :=
  volume 8 = T

def one_quarter_capacity (t : ℕ) : Prop :=
  volume t = T / 4

-- Theorem to prove
theorem pool_one_quarter_capacity_at_6 (h1 : doubles_every_hour volume) (h2 : full_capacity_at_8 volume T) : one_quarter_capacity volume T 6 :=
sorry

end pool_one_quarter_capacity_at_6_l814_814446


namespace travel_agency_choice_l814_814200

noncomputable def y₁ (x : ℝ) : ℝ := 350 * x + 1000

noncomputable def y₂ (x : ℝ) : ℝ := 400 * x + 800

theorem travel_agency_choice (x : ℝ) (h : 0 < x) :
  (x < 4 → y₁ x > y₂ x) ∧ 
  (x = 4 → y₁ x = y₂ x) ∧ 
  (x > 4 → y₁ x < y₂ x) :=
by {
  sorry
}

end travel_agency_choice_l814_814200


namespace polynomial_divisible_by_roots_of_unity_l814_814940

theorem polynomial_divisible_by_roots_of_unity (C D E : ℤ) 
  (h1 : C = 1) (h2 : D = -2) (h3 : E = 1) :
  ∃ (C D E : ℤ), (C + D + E = 0) ∧ (∀ α : ℂ, α^2 - α + 1 = 0 → (α^103 + C * α^2 + D * α + E = 0)) :=
by
  use [1, -2, 1],
  split,
  { exact add_add_of_eq_add (show 1 + -2 + 1 = 0, by norm_num) },
  { intros α hα,
    have hα_cubed : α^3 = 1 := by
      rw [← add_left_eq_self (show α^2 - α + 1 = 0 from hα), pow_succ, pow_two, ← sub_eq_of_eq_add' (show α * α = α * (α - 1) + 1 from _), mul_sub, mul_one, add_one, add_neg_self],
    rw [pow_add, pow_mul (_+ 1)],
    simp only [hα_cubed, one_pow, mul_one, pow_one],
    have hα_sq : α^2 = α - 1 := by
      rw [← hα, sub_add_eq_zero],
    rw [hα_sq, mul_add, ← mul_sub_left_distrib, sub_add],
    ring }
sorry

end polynomial_divisible_by_roots_of_unity_l814_814940


namespace variance_correct_l814_814245

def data_set : List ℝ := [8, 12, 10, 11, 9]

def mean (l : List ℝ) : ℝ := l.sum / l.length

def variance (l : List ℝ) : ℝ := 
  let μ := mean l
  (l.map (λ x => (x - μ) * (x - μ))).sum / l.length

theorem variance_correct : variance data_set = 2 := by
  sorry

end variance_correct_l814_814245


namespace Morse_code_distinct_symbols_count_l814_814341

theorem Morse_code_distinct_symbols_count :
  let count (n : ℕ) := 2 ^ n
  count 1 + count 2 + count 3 + count 4 + count 5 = 62 :=
by
  sorry

end Morse_code_distinct_symbols_count_l814_814341


namespace situationD_not_represented_l814_814846

def situationA := -2 + 10 = 8

def situationB := -2 + 10 = 8

def situationC := 10 - 2 = 8 ∧ -2 + 10 = 8

def situationD := |10 - (-2)| = 12

theorem situationD_not_represented : ¬ (|10 - (-2)| = -2 + 10) := 
by
  sorry

end situationD_not_represented_l814_814846


namespace balanced_numbers1_balanced_numbers2_l814_814566

-- Part 1: Balanced numbers proof
theorem balanced_numbers1 (a b n : ℤ) (h : a = -6) (h' : b = 8) : a + b = 2 := by
    -- Apply the conditions of the problem
    rw [h, h']
    -- Showing that they add up to 2
    exact rfl

-- Part 2: Balanced numbers quadratic proof
theorem balanced_numbers2 (a b : ℤ → ℤ) (k n : ℤ) (h : ∀ x, a x = 6 * x^2 - 4 * k * x + 8) (h' : ∀ x, b x = -2 * (3 * x^2 - 2 * x + k)) (hk : k = 1) : (∀ x, a x + b x = n) → n = 6 := by
    -- Substitute the function definitions
    intro hn
    -- Apply the condition of k
    rw [hk]
    -- Simplify the left-hand side expression
    funext x
    calc (6 * x^2 - 4 * 1 * x + 8) + (-2 * (3 * x^2 - 2 * x + 1)) = _ := by {
        simp [h, h']
        ring_nf
    }
    sorry -- Further simplification and proof that it always equals 6

end balanced_numbers1_balanced_numbers2_l814_814566


namespace eight_girls_circle_least_distance_l814_814943

theorem eight_girls_circle_least_distance :
  let r := 50
  let num_girls := 8
  let total_distance := (8 * (3 * (r * Real.sqrt 2) + 2 * (2 * r)))
  total_distance = 1200 * Real.sqrt 2 + 1600 :=
by
  sorry

end eight_girls_circle_least_distance_l814_814943


namespace men_science_majors_percentage_l814_814491

def percentage_of_men_science_majors 
    (total_class_percent : ℕ) 
    (percent_women_sci_majors : ℕ) 
    (percent_nonscience_majors : ℕ) 
    (percent_men : ℕ) : ℕ :=
  let percent_women := total_class_percent - percent_men
  let women_sci_majors := (percent_women_sci_majors * percent_women) / total_class_percent
  let total_sci_majors := total_class_percent - percent_nonscience_majors
  let men_sci_majors := total_sci_majors - women_sci_majors
  men_sci_majors

theorem men_science_majors_percentage 
    (h1 : 20% of the women in a college class are science majors)
    (h2 : Non-science majors make up 60% of the class)
    (h3 : 40% of the class are men) : 
    percentage_of_men_science_majors 100 20 60 40 = 28 := 
by
  sorry

end men_science_majors_percentage_l814_814491


namespace matrix_det_nonzero_l814_814384

theorem matrix_det_nonzero (n : ℕ) (a : ℤ) (h : a > 1) : 
  let A := matrix (fin n) (fin n) (λ (i j : fin n), a ^ ((i.1 + 1) * (j.1 + 1))) in
  matrix.det A ≠ 0 := 
sorry

end matrix_det_nonzero_l814_814384


namespace unique_untouchable_temple_l814_814021

-- Conditions and Definitions
def Temple (n : ℕ) : Type := { T : fin (2 * n) → ℕ // function.injective T }

-- The main statement to prove
theorem unique_untouchable_temple (n : ℕ) (T : Temple n) : 
  ∃! t : fin n, ∀ e : fin (2 * n), 
    (e.val < n ∧ T.val e < T.val (fin.of_nat (e.val + 1)) ∧ T.val e < T.val (fin.of_nat (e.val - 1))) → 
    ¬ ∃ e', (T.val e' > T.val e ∧ e.val = e'.val + 1) :=
sorry

end unique_untouchable_temple_l814_814021


namespace num_moles_CaO_combined_l814_814214

-- Define the molarity and chemical substances involved
def moles (substance : Type) : Type := ℝ

constant CaO : Type
constant H2O : Type
constant CaOH₂ : Type

constant reaction : moles CaO → moles H2O → moles CaOH₂ → Prop
constant one_mole_CaOH₂ : moles CaOH₂
constant one_mole_H2O : moles H2O

axiom balanced_reaction : ∀ (n_moles_CaO : moles CaO), 
  reaction n_moles_CaO one_mole_H2O one_mole_CaOH₂ →

theorem num_moles_CaO_combined : reaction 1 1 1 :=
by
  sorry

end num_moles_CaO_combined_l814_814214


namespace sum_of_N_even_l814_814001

def N (n : ℕ) : ℕ := (Finset.range n).filter (λ d, n % (d + 1) = 0).card + 1

theorem sum_of_N_even : 
  (Finset.range 1989).sum (λ n, N (n + 1)) % 2 = 0 :=
by
  sorry

end sum_of_N_even_l814_814001


namespace trains_crossing_time_l814_814091

theorem trains_crossing_time :
  ∀ (len1 len2 : ℝ) (speed1 speed2 : ℝ), 
    (len1 = 140) →
    (len2 = 160) →
    (speed1 = 60 * 1000 / 3600) →
    (speed2 = 40 * 1000 / 3600) →
    (len1 + len2) / ((speed1 + speed2)) ≈ 10.8 := 
by {
  intros len1 len2 speed1 speed2 hlen1 hlen2 hspeed1 hspeed2,
  sorry
}

end trains_crossing_time_l814_814091


namespace max_distance_circle_to_point_A_l814_814211

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (-1, 3)

noncomputable def max_distance (d : ℝ) : Prop :=
  ∃ x y, circle_eq x y ∧ d = Real.sqrt ((2 + 1)^2 + (0 - 3)^2) + Real.sqrt 2 

theorem max_distance_circle_to_point_A : max_distance (4 * Real.sqrt 2) :=
sorry

end max_distance_circle_to_point_A_l814_814211


namespace distance_between_parallel_lines_l814_814649

-- Definitions of the lines
def line1 (x y : ℝ) (a : ℝ) : Prop := x + a * y = 1
def line2 (x y : ℝ) (a : ℝ) : Prop := a * x + y = 1

-- Definition of parallel lines
def are_parallel (a : ℝ) : Prop := - (1 / a) = -a 

-- Given condition: parallel lines, a = -1, under these conditions 
-- we need to prove the distance between the lines is √2.
theorem distance_between_parallel_lines (a : ℝ) 
  (ha1 : are_parallel a)
  (ha2 : a = -1) : 
  ∃ d : ℝ, d = Real.sqrt 2 ∧ 
           (∃ x1 y1 x2 y2 : ℝ, line1 x1 y1 a ∧ line2 x2 y2 a ∧ 
            (d = (Real.abs (-1 - 1)) / Real.sqrt (1^2 + (-1)^2))) :=
begin
  sorry
end

end distance_between_parallel_lines_l814_814649


namespace probability_three_white_balls_l814_814504

open Nat

def totalWaysToDrawThreeBalls : ℕ := choose 15 3
def waysToDrawThreeWhiteBalls : ℕ := choose 7 3

theorem probability_three_white_balls :
  (waysToDrawThreeWhiteBalls : ℚ) / (totalWaysToDrawThreeBalls : ℚ) = 1 / 13 := 
sorry

end probability_three_white_balls_l814_814504


namespace winner_votes_percentage_l814_814713

-- Define the total votes as V
def total_votes (winner_votes : ℕ) (winning_margin : ℕ) : ℕ :=
  winner_votes + (winner_votes - winning_margin)

-- Define the percentage function
def percentage_of_votes (part : ℕ) (total : ℕ) : ℕ :=
  (part * 100) / total

-- Lean statement to prove the result
theorem winner_votes_percentage
  (winner_votes : ℕ)
  (winning_margin : ℕ)
  (H_winner_votes : winner_votes = 550)
  (H_winning_margin : winning_margin = 100) :
  percentage_of_votes winner_votes (total_votes winner_votes winning_margin) = 55 := by
  sorry

end winner_votes_percentage_l814_814713


namespace counting_colorings_l814_814732

-- Defining the problem conditions as given:
def number_of_colorings (n k p : ℕ) : ℕ :=
  if 2 ≤ k ∧ k ≤ n/(p + 1) then
    (n - kp - 1).choose (k - 1) / k
  else
    0

-- Statement asserting the answer to the coloring problem
theorem counting_colorings (n k p : ℕ) (hn : n > 0) (hk : 2 ≤ k) (hp : k ≤ n/(p + 1)) :
  number_of_colorings n k p = (n - kp - 1).choose (k - 1) / k := 
sorry

end counting_colorings_l814_814732


namespace trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13_l814_814404

theorem trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13 :
  (Real.cos (58 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) +
   Real.sin (58 * Real.pi / 180) * Real.sin (13 * Real.pi / 180) =
   Real.cos (45 * Real.pi / 180)) :=
sorry

end trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13_l814_814404


namespace standard_equation_of_ellipse_l814_814217

theorem standard_equation_of_ellipse
  (focus₁ focus₂ : ℝ × ℝ) (sum_dist : ℝ)
  (P : ℝ × ℝ) (std_eq : ℝ → ℝ → Prop)
  (foci : focus₁ = (0, -4) ∧ focus₂ = (0, 4))
  (sum_dist_condition : sum_dist = 10)
  (passes_through_P : P = (-3, 2))
  (equiv_foci_eq : std_eq = λ x y, x^2 / 9 + y^2 / 4 = 1) :
  ∀ (x y : ℝ), (x^2 / 15 + y^2 / 10 = 1) :=
by
  sorry

end standard_equation_of_ellipse_l814_814217


namespace probability_of_drawing_balls_l814_814877

theorem probability_of_drawing_balls :
  let total_balls := 18
  let draw_balls := 7
  let white_balls := 7
  let red_balls := 3
  let total_ways := Nat.choose 18 7
  let ways_white := Nat.choose 7 5
  let ways_red := Nat.choose 3 2
  let favorable_ways := ways_white * ways_red
  let probability := (favorable_ways : ℚ) / (total_ways : ℚ)
  probability = 63 / 31824 :=
by
  sorry

end probability_of_drawing_balls_l814_814877


namespace find_x_l814_814495

variable (P T S : Point)
variable (angle_PTS angle_TSR x : ℝ)
variable (reflector : Point)

-- Given conditions
axiom angle_PTS_is_90 : angle_PTS = 90
axiom angle_TSR_is_26 : angle_TSR = 26

-- Proof problem
theorem find_x : x = 32 := by
  sorry

end find_x_l814_814495


namespace sum_sequence_evaluation_l814_814203

theorem sum_sequence_evaluation : ∑ n in Finset.range 10 \ fun i => i < 2, (n + 3) * (1 - (1 / (n + 3)) + 1) = 140 :=
by
  -- This will translate the goal directly into: 
  -- Have the specific sum of each term, considering n ranges from 3 to 12 inclusive.
  sorry

end sum_sequence_evaluation_l814_814203


namespace sum_of_digits_of_palindrome_l814_814131

def is_palindrome (n : ℕ) : Prop :=
  n = nat.of_digits (nat.base_digits 10 n).reverse

def sum_of_digits (n : ℕ) : ℕ :=
  (nat.base_digits 10 n).sum

theorem sum_of_digits_of_palindrome (y : ℕ) (hy1 : 100 ≤ y ∧ y < 1000)
  (hy2 : is_palindrome y) (hy3 : is_palindrome (y + 50)) :
  sum_of_digits y = 15 ∨ sum_of_digits y = 16 :=
sorry

end sum_of_digits_of_palindrome_l814_814131


namespace Collin_total_petals_l814_814554

-- Definitions of the conditions
def initial_flowers_Collin : ℕ := 25
def flowers_Ingrid : ℕ := 33
def petals_per_flower : ℕ := 4
def third_of_flowers_Ingrid : ℕ := flowers_Ingrid / 3

-- Total number of flowers Collin has after receiving from Ingrid
def total_flowers_Collin : ℕ := initial_flowers_Collin + third_of_flowers_Ingrid

-- Total number of petals Collin has
def total_petals_Collin : ℕ := total_flowers_Collin * petals_per_flower

-- The theorem to be proved
theorem Collin_total_petals : total_petals_Collin = 144 := by
  -- Proof goes here
  sorry

end Collin_total_petals_l814_814554


namespace number_of_distinguishable_colorings_l814_814942
open Classical

-- Define the concept of a face being colored in one of the four colors
inductive Color
| red
| white
| blue
| green

-- Define a structure for a Tetrahedron Coloring
structure TetrahedronColoring :=
  (face1 : Color)
  (face2 : Color)
  (face3 : Color)
  (face4 : Color)

-- Function that determines when two colorings are equivalent based on rotations
noncomputable def equivalent_colorings (c1 c2 : TetrahedronColoring) : Prop := sorry

-- Define the set of all possible colorings of a tetrahedron
noncomputable def all_colorings : set TetrahedronColoring := 
  { t | true }

-- Define the set of distinguishable colorings
noncomputable def distinguishable_colorings : set TetrahedronColoring :=
  { t ∈ all_colorings | ∀ t', t' ∈ all_colorings → equivalent_colorings t t' → t = t' }

-- The theorem to prove the number of distinguishable colorings is 58
theorem number_of_distinguishable_colorings : (distinguishable_colorings.to_finset.card = 58) := sorry

end number_of_distinguishable_colorings_l814_814942


namespace winning_candidate_percentage_is_31_3_l814_814706

-- Define initial votes
def initial_votes : List Nat := [15236, 20689, 12359, 30682, 25213, 18492]

-- Define the votes of candidates who withdrew.
def withdrawn_votes : List Nat := [12359, 18492]

-- Define the redistribution percentages in a tuple (60%, 25%, 15%)
def redistribution_percentages : List (ℚ) := [0.60, 0.25, 0.15]

-- Define vote redistribution functions
def redistributed_votes (percent : ℚ) (votes : ℚ) : ℚ := percent * votes

-- Calculate the redistributed votes
def redistributed_15236 := redistributed_votes redistribution_percentages[0] withdrawn_votes[0] + 
                            redistributed_votes redistribution_percentages[0] withdrawn_votes[1]

def redistributed_30682 := redistributed_votes redistribution_percentages[1] withdrawn_votes[0] + 
                            redistributed_votes redistribution_percentages[1] withdrawn_votes[1]

def redistributed_20689 := redistributed_votes redistribution_percentages[2] withdrawn_votes[0] + 
                            redistributed_votes redistribution_percentages[2] withdrawn_votes[1]

-- Total votes for each remaining candidate after redistribution
def final_votes_15236 := initial_votes[0] + redistributed_15236
def final_votes_20689 := initial_votes[1] + redistributed_20689
def final_votes_30682 := initial_votes[3] + redistributed_30682
def final_votes_25213 := initial_votes[4] -- no redistribution for 25213

-- Calculate the total number of votes in the final tally
def total_votes_final := final_votes_15236 + final_votes_20689 + final_votes_30682 + initial_votes[4]

-- Define the percentage calculation
def percentage (part : ℚ) (whole : ℚ) : ℚ := (part / whole) * 100

-- Assert the final percentage
theorem winning_candidate_percentage_is_31_3 :
  percentage final_votes_30682 total_votes_final ≈ 31.3 := sorry

end winning_candidate_percentage_is_31_3_l814_814706


namespace total_weight_of_2m_portions_l814_814453

-- Define the given conditions
def bag_length := 5
def bag_weight := 29 / 8
def chair_length := 4
def chair_weight := 2.8
def portion_length := 2

-- Define the weights per meter
def bag_weight_per_meter := bag_weight / bag_length
def chair_weight_per_meter := chair_weight / chair_length

-- Define the weights of 2-meter portions
def bag_2m_weight := portion_length * bag_weight_per_meter
def chair_2m_weight := portion_length * chair_weight_per_meter

-- Calculate the expected total weight
def expected_total_weight := bag_2m_weight + chair_2m_weight

-- The theorem statement in Lean
theorem total_weight_of_2m_portions :
  expected_total_weight = 2.85 := by
  -- The actual proof is omitted for now
  sorry

end total_weight_of_2m_portions_l814_814453


namespace bela_wins_always_l814_814920

-- Define conditions for the game
variable (n : ℕ) (h : n > 10)

-- Main theorem statement: Bela will always win the game
theorem bela_wins_always (n : ℕ) (h : n > 10) : 
  ∀ (optimal_strategy : (ℝ → Prop)), 
  (optimal_strategy ∈ set.Icc 0 n) → 
  (∀ x y : ℝ, optimal_strategy x → optimal_strategy y → abs (x - y) > 2) → 
  ∃ bela_wins : (ℕ > 10) → Prop, bela_wins = true := 
by 
  intro optimal_strategy
  intro h_interval
  intro h_spacing
  sorry

end bela_wins_always_l814_814920


namespace valid_number_of_beads_l814_814168

def Shape := ℕ -- simple definition for the shapes: cubes, pyramids
def Color := ℕ -- simple definition for the colors: green, blue, red

-- Definitions for our specific beads
def Kg : Shape × Color := (1, 1) -- green cube
def Kb : Shape × Color := (1, 2) -- blue cube
def Pr : Shape × Color := (2, 3) -- red pyramid
def Pb : Shape × Color := (2, 2) -- blue pyramid

-- Conditions given in the problem
def is_valid_necklace (necklace : List (Shape × Color)) : Prop :=
  ∀ i, (necklace.nth i ≠ none) →
      -- different shapes and different colors condition
      let ⟨(s1, c1)⟩ := necklace.get (i % necklace.length);
      let ⟨(s2, c2)⟩ := necklace.get ((i + 1) % necklace.length);
      s1 ≠ s2 ∧ c1 ≠ c2

-- Prove that the necklace must have a valid even count of beads and specifically 8 or 10.
theorem valid_number_of_beads : 
  ∀ (necklace : List (Shape × Color)), 
  is_valid_necklace necklace →
  (necklace.length = 8 ∨ necklace.length = 10) :=
by
  intros necklace h
  sorry

end valid_number_of_beads_l814_814168


namespace oil_output_per_capita_l814_814867

theorem oil_output_per_capita 
  (total_oil_output_russia : ℝ := 13737.1 * 100 / 9)
  (population_russia : ℝ := 147)
  (population_non_west : ℝ := 6.9)
  (oil_output_non_west : ℝ := 1480.689)
  : 
  (55.084 : ℝ) = 55.084 ∧ 
    (214.59 : ℝ) = (1480.689 / 6.9) ∧ 
    (1038.33 : ℝ) = (total_oil_output_russia / population_russia) :=
by
  sorry

end oil_output_per_capita_l814_814867


namespace general_term_formula_l814_814293

noncomputable def a : ℕ → ℤ
| 0     := 0  -- This is for the natural numbers starting from 0, not used in the problem
| (n+1) := 2 * a n + 4

theorem general_term_formula (n : ℕ) (h : n > 0) :
  a (n - 1 + 1) = 5 * 2^((n - 1)ℕ) - 4 :=
by
  sorry

end general_term_formula_l814_814293


namespace elements_smaller_than_bound_l814_814124

def P : polynomial ℤ := sorry -- P is any monic integer coefficient polynomial of degree n with no integer roots

-- Conditions
variables (n : ℕ) (P : polynomial ℤ)
#check polynomial.degree P  -- ensure P is a polynomial
#check polynomial.monic P   -- ensure P is monic

noncomputable def A : finset ℕ := {v | ∃ m : ℤ, (padic_val_rat 2 (P.eval m)).nat_abs = v ∧ v ≥ 1}

axiom P_monic : P.monic
axiom P_no_int_roots : ∀ m : ℤ, P.eval m ≠ 0
axiom P_degree : polynomial.degree P = n
axiom A_size : A.card = n

-- Proof objective
theorem elements_smaller_than_bound : ∀ a ∈ A, a < (3 / 2) * n^2 :=
by
  intro a ha
  sorry -- Proof to be filled in

end elements_smaller_than_bound_l814_814124


namespace intersection_is_3_l814_814296

def is_solution (x : ℝ) : Prop := x^2 - 3 * x = 0
def M : set ℝ := { x | is_solution x }
def N : set ℝ := {-1, 1, 3}
def correct_intersection : set ℝ := {3}

theorem intersection_is_3 : M ∩ N = correct_intersection := 
  by
    sorry

end intersection_is_3_l814_814296


namespace polynomial_descending_order_a_l814_814433

theorem polynomial_descending_order_a : 
  let p : polynomial ℚ := 2 * (X ^ 4) + 4 * (X ^ 3) * (Y ^ 4) - 5 * (X ^ 2) * Y + 2 * X
  in (p.coeff (4, 0, 0) > p.coeff (3, 4, 0) ∧ p.coeff (3, 4, 0) > p.coeff (2, 1, 0) ∧ p.coeff (2, 1, 0) > p.coeff (1, 0, 0)) :=
by
  sorry

end polynomial_descending_order_a_l814_814433


namespace C_share_of_rent_l814_814111

def oxen_months (oxen : Nat) (months : Nat) : Nat :=
  oxen * months

def total_rent : Nat := 245

theorem C_share_of_rent :
  let a_oxen := 10 in
  let a_months := 7 in
  let b_oxen := 12 in
  let b_months := 5 in
  let c_oxen := 15 in
  let c_months := 3 in
  let A_share := oxen_months a_oxen a_months in
  let B_share := oxen_months b_oxen b_months in
  let C_share := oxen_months c_oxen c_months in
  let total_ox_months := A_share + B_share + C_share in
  (C_share * total_rent) / total_ox_months = 63 :=
by
  sorry

end C_share_of_rent_l814_814111


namespace checkerboard_L_tile_coverable_l814_814004

-- Definitions based on the conditions given in the problem
def L_tile (tile : List (Int × Int)) : Prop :=
    tile.length = 3 ∧
    (∃ (a b c : Int × Int), tile = [a, b, c] ∧ ∃ (dx dy dz : Int), 
        (b.1 = a.1 + dx ∧ b.2 = a.2 + dy) ∧ 
        (c.1 = a.1 + dz ∧ c.2 = a.2 + dz))

def is_coverable_by_L_tiles (n : Int) : Prop :=
    ∀ (board : List (Int × Int)), board.length = 2^n * 2^n - 1 → 
    (∃ (cover : List (List (Int × Int))),
     ∀ tile ∈ cover, L_tile tile ∧ 
     cover.join = board)

theorem checkerboard_L_tile_coverable (n : Int) (hn : 0 < n) :
  is_coverable_by_L_tiles n :=
by
  -- Proof omitted, placeholder for actual implementation
  sorry

end checkerboard_L_tile_coverable_l814_814004


namespace triangle_area_is_32_l814_814588

-- Definitions according to conditions
def point := ℝ × ℝ

structure triangle :=
  (D E F : point)
  (right_angle : ∠ D E F = π / 2)
  (angle_DFE : ∠ D F E = π / 4)
  (angle_DEF : ∠ D E F = π / 4)
  (DE_length : dist D E = 8)

-- The statement to prove the area of the triangle is 32 square units
theorem triangle_area_is_32 (t : triangle) : 
  let D := t.D
      E := t.E
      F := t.F
  in 1 / 2 * (dist D E) * (dist D F) = 32 :=
by 
  -- This is to be proven
  sorry

end triangle_area_is_32_l814_814588


namespace log_problem_solution_l814_814548

def log_problem_statement : ℝ :=
  log 5 * (log 20 / (log (sqrt 10))) + (log 2 ^ sqrt 2)^2 + Real.exp (log pi)

theorem log_problem_solution : log_problem_statement = 2 + Real.pi :=
  sorry

end log_problem_solution_l814_814548


namespace equation_of_line_m_l814_814941

theorem equation_of_line_m 
  (P : ℝ×ℝ) (P'' : ℝ×ℝ) (hP : P = (-3, 2)) (hP'' : P'' = (2, -1)) (hℓ : ∀ x y : ℝ, 3 * x - y = 0 ↔ (x, y) ∈ {p : ℝ×ℝ | 3 * p.1 - p.2 = 0}) :
  ∃ (m : ℝ → ℝ→ Prop), (∀ x y, m x y ↔ x + 3 * y = 0) → P'' = 2 * P + _ := 
sorry

end equation_of_line_m_l814_814941


namespace sum_of_squares_PV_l814_814202

noncomputable def equilateral_triangle_PQR := 
  {P Q R : ℝ → ℝ → Prop // ∃ l : ℝ, l = 10 
   ∧ ∀ (x y : ℝ → ℝ), x y ∈ {P Q R} → dist x y = l} 

noncomputable def congruent_triangles :=
  {U_1 U_2 V_1 V_2 V_3 V_4: ℝ → ℝ → Prop // 
   (∀ (x y : ℝ → ℝ), dist x y = 3 → (x = Q ∧ y = U_1) ∨ (x = Q ∧ y = U_2))
   ∧ ∀ (t : (ℝ → ℝ) → (ℝ → ℝ) → (ℝ → ℝ)) (i : ℕ), i < 4 → 
       congruent t (equilateral_triangle_PQR) ∧ (t = (U_1 V_1)) ∨ (t = (U_1 V_2)) ∨ (t = (U_2 V_3)) ∨ (t = (U_2 V_4)) }

theorem sum_of_squares_PV (P Q R : ℝ → ℝ) (U_1 U_2 V_1 V_2 V_3 V_4 : ℝ → ℝ) : 
  ∃ (P Q R : ℝ → ℝ) (U_1 U_2 V_1 V_2 V_3 V_4 : ℝ → ℝ), 
    equilateral_triangle_PQR ∧ congruent_triangles ∧ 
    (∑ k : fin 4, dist (P) (V k).val ^ 2) = 800 :=
sorry

end sum_of_squares_PV_l814_814202


namespace inradius_circumradius_inequality_l814_814733

variables {A B C : Type} [triangle A B C] (a b c : ℝ)
variables (angle_A : ℝ) (r R : ℝ)
variables [h_A : angle_A > 90]
variables [h_r : r = inradius A] [h_R : R = circumradius A]

theorem inradius_circumradius_inequality (angle_A : ℝ) [h_A : angle_A > 90]
    (a b c : ℝ) (r R : ℝ)
    [h_r : r = inradius A B C]
    [h_R : R = circumradius A B C] :
  (r / R) ≤ (a * sin angle_A) / (a + b + c) :=
sorry

end inradius_circumradius_inequality_l814_814733


namespace stock_worth_l814_814852

theorem stock_worth (profit_part loss_part total_loss : ℝ) 
  (h1 : profit_part = 0.10) 
  (h2 : loss_part = 0.90) 
  (h3 : total_loss = 400) 
  (profit_rate : ℝ := 0.20) 
  (loss_rate : ℝ := 0.05)
  (profit_value := profit_rate * profit_part)
  (loss_value := loss_rate * loss_part)
  (overall_loss := total_loss)
  (h4 : loss_value - profit_value = overall_loss) :
  ∃ X : ℝ, X = 16000 :=
by
  sorry

end stock_worth_l814_814852


namespace steve_reading_pages_l814_814780

theorem steve_reading_pages (total_pages: ℕ) (weeks: ℕ) (reading_days_per_week: ℕ) 
  (reads_on_monday: ℕ) (reads_on_wednesday: ℕ) (reads_on_friday: ℕ) :
  total_pages = 2100 → weeks = 7 → reading_days_per_week = 3 → 
  (reads_on_monday = reads_on_wednesday ∧ reads_on_wednesday = reads_on_friday) → 
  ((weeks * reading_days_per_week) > 0) → 
  (total_pages / (weeks * reading_days_per_week)) = reads_on_monday :=
by
  intro h_total_pages h_weeks h_reading_days_per_week h_reads_on_days h_nonzero
  sorry

end steve_reading_pages_l814_814780


namespace min_value_f_range_of_m_l814_814620

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x - m * Real.log x - (m - 1) / x
noncomputable def g (x : ℝ) : ℝ := (1/2) * x^2 + Real.exp x - x * Real.exp x

theorem min_value_f (m : ℝ) (hm1 : m ≤ 2 ∨ m ≥ Real.exp 1 + 1 ∨ (2 < m ∧ m < Real.exp 1 + 1)) :
  ∃ x ∈ Set.Icc 1 (Real.exp 1), 
    f x m = if m ≤ 2 then 2 - m 
            else if m ≥ Real.exp 1 + 1 then Real.exp 1 - m - (m - 1) / Real.exp 1 
            else m - 2 - m * Real.log (m - 1) :=
sorry

theorem range_of_m (m : ℝ) (hm2 : m ≤ 2) (hx1 : ∃ x₁ ∈ Set.Icc (Real.exp 1) (Real.exp 1 ^ 2), ∀ x₂ ∈ Set.Icc (-2 : ℝ) 0, f x₁ m ≤ g x₂) :
  ∃ m ∈ Set.Icc (Real.exp 1 ^ 2 - Real.exp 1 + 1 / (Real.exp 1 + 1)) 2 :=
sorry

end min_value_f_range_of_m_l814_814620


namespace fraction_decomposition_l814_814427

noncomputable def p (n : ℕ) : ℚ :=
  (n + 1) / 2

noncomputable def q (n : ℕ) : ℚ :=
  n * p n

theorem fraction_decomposition (n : ℕ) (h : ∃ k : ℕ, n = 5 + 2*k) :
  (2 / n : ℚ) = (1 / p n) + (1 / q n) :=
by
  sorry

end fraction_decomposition_l814_814427


namespace wrapping_paper_solution_l814_814550

variable (P1 P2 P3 : ℝ)

def wrapping_paper_problem : Prop :=
  P1 = 2 ∧
  P3 = P1 + P2 ∧
  P1 + P2 + P3 = 7 →
  (P2 / P1) = 3 / 4

theorem wrapping_paper_solution : wrapping_paper_problem P1 P2 P3 :=
by
  sorry

end wrapping_paper_solution_l814_814550


namespace scientific_notation_l814_814726

theorem scientific_notation (x : ℝ) (h : x = 0.000005) : x = 5 * 10^(-6) := 
by
  -- we skip the proof
  sorry

end scientific_notation_l814_814726


namespace solve_ordered_pair_l814_814939

theorem solve_ordered_pair : ∃ (x y : ℚ), 3*x - 24*y = 3 ∧ x - 3*y = 4 ∧ x = 29/5 ∧ y = 3/5 := by
  sorry

end solve_ordered_pair_l814_814939


namespace cities_real_distance_l814_814763

def map_scale := 7 -- number of centimeters representing 35 kilometers
def real_distance_equiv := 35 -- number of kilometers that corresponds to map_scale

def centimeters_per_kilometer := real_distance_equiv / map_scale -- kilometers per centimeter

def distance_on_map := 49 -- number of centimeters cities are separated by on the map

theorem cities_real_distance : distance_on_map * centimeters_per_kilometer = 245 :=
by
  sorry

end cities_real_distance_l814_814763


namespace center_of_hyperbola_is_3_4_l814_814589

noncomputable def hyperbola_center : Prop :=
  ∃ c : ℝ × ℝ, 
    let x := c.1 in
    let y := c.2 in
    9 * x^2 - 54 * x - 16 * y^2 + 128 * y - 400 = 0 ∧
    c = (3, 4)

theorem center_of_hyperbola_is_3_4 : hyperbola_center :=
begin
  sorry
end

end center_of_hyperbola_is_3_4_l814_814589


namespace population_growth_l814_814701

theorem population_growth (p q r : ℕ) 
  (h1 : p^2 > 0)
  (h2 : p^2 + 100 = q^2 + 9)
  (h3 : p^2 + 275 = r^2) :
  ((r^2 - p^2 : ℚ) / p^2) * 100 ≈ 9.09 :=
by
  sorry

end population_growth_l814_814701


namespace domain_of_sqrt_expression_l814_814936

theorem domain_of_sqrt_expression :
  {x : ℝ | x^2 - 5 * x - 6 ≥ 0} = {x : ℝ | x ≤ -1} ∪ {x : ℝ | x ≥ 6} := by
sorry

end domain_of_sqrt_expression_l814_814936


namespace intersection_of_lines_l814_814937

theorem intersection_of_lines
    (x y : ℚ) 
    (h1 : y = 3 * x - 1)
    (h2 : y + 4 = -6 * x) :
    x = -1 / 3 ∧ y = -2 := 
sorry

end intersection_of_lines_l814_814937


namespace volleyball_club_lineups_l814_814023
-- Import the required Lean library

-- Define the main problem
theorem volleyball_club_lineups :
  let total_players := 18
  let quadruplets := 4
  let starters := 6
  let eligible_lineups := Nat.choose 18 6 - Nat.choose 14 2 - Nat.choose 14 6
  eligible_lineups = 15470 :=
by
  sorry

end volleyball_club_lineups_l814_814023


namespace initial_pen_count_is_30_l814_814477

def pen_count (initial_pens : ℕ) : ℕ :=
  let after_mike := initial_pens + 20
  let after_cindy := 2 * after_mike
  let after_sharon := after_cindy - 10
  after_sharon

theorem initial_pen_count_is_30 : pen_count 30 = 30 :=
by
  sorry

end initial_pen_count_is_30_l814_814477


namespace tim_total_payment_l814_814081

-- Conditions given in the problem
def doc_visit_cost : ℕ := 300
def insurance_coverage_pct : ℕ := 75
def cat_visit_cost : ℕ := 120
def pet_insurance_coverage : ℕ := 60

-- Formulation of the final proof problem
theorem tim_total_payment : 
  let doc_insurance_coverage := (insurance_coverage_pct * doc_visit_cost) / 100,
      doc_out_of_pocket := doc_visit_cost - doc_insurance_coverage,
      cat_out_of_pocket := cat_visit_cost - pet_insurance_coverage,
      total_payment := doc_out_of_pocket + cat_out_of_pocket
  in total_payment = 135 :=
by
  let doc_insurance_coverage := (insurance_coverage_pct * doc_visit_cost) / 100
  let doc_out_of_pocket := doc_visit_cost - doc_insurance_coverage
  let cat_out_of_pocket := cat_visit_cost - pet_insurance_coverage
  let total_payment := doc_out_of_pocket + cat_out_of_pocket
  have h1 : total_payment = 75 + 60 := by rfl
  have h2 : 75 + 60 = 135 := by rfl
  exact Eq.trans h1 h2

end tim_total_payment_l814_814081


namespace max_lambda_of_inequality_l814_814238

theorem max_lambda_of_inequality :
  (λ λ, λ > 0 → (∀ x : ℝ, (0 < x) → (Real.exp (x / λ) - λ * Real.log x) ≥ 0) → λ ≤ Real.exp 1) :=
sorry

end max_lambda_of_inequality_l814_814238


namespace triangle_shape_l814_814696

variable {A B C a b c : ℝ}

def triangle (A B C : ℝ) : Prop := (A + B + C = 180) ∧ (A > 0) ∧ (B > 0) ∧ (C > 0)

theorem triangle_shape (h : triangle A B C) (h_cos : cos A / cos B = c / a) : 
  (A = B ∨ A + B = 90) → 
  (((A = B) ∨ (A + B = 90)) → is_isosceles_or_right_triangle A B C) := 
sorry

def is_isosceles_or_right_triangle (A B C : ℝ) : Prop := (A = B ∨ A = C ∨ B = C) ∨ 
  ((A = 90) ∨ (B = 90) ∨ (C = 90))

end triangle_shape_l814_814696


namespace equation_of_line_through_A_and_B_l814_814794

-- Definitions for the points A and B
def A : ℝ × ℝ := (0, -5)
def B : ℝ × ℝ := (1, 0)

-- The slope calculation
def slope (p1 p2 : ℝ × ℝ) := (p2.2 - p1.2) / (p2.1 - p1.1)

-- The point-slope form of the line
def line_through_points (p1 p2 : ℝ × ℝ) (x : ℝ) := slope p1 p2 * (x - p1.1) + p1.2

-- The given condition
def line_ab (x : ℝ) : ℝ := 5 * x - 5

-- The theorem we need to prove
theorem equation_of_line_through_A_and_B : 
  ∀ x, line_through_points A B x = line_ab x :=
by 
  -- This is where the proof would go.
  sorry

end equation_of_line_through_A_and_B_l814_814794


namespace find_bloom_day_l814_814822

def days := {d : Fin 7 // 1 ≤ d.val ∧ d.val ≤ 7}

def sunflowers_bloom (d : days) : Prop :=
¬ (d.val = 2 ∨ d.val = 4 ∨ d.val = 7)

def lilies_bloom (d : days) : Prop :=
¬ (d.val = 4 ∨ d.val = 6)

def magnolias_bloom (d : days) : Prop :=
¬ (d.val = 7)

def all_bloom_together (d : days) : Prop :=
sunflowers_bloom d ∧ lilies_bloom d ∧ magnolias_bloom d

def blooms_simultaneously (d : days) : Prop :=
∀ d1 d2 d3 : days, (d1 = d ∧ d2 = d ∧ d3 = d) →
(all_bloom_together d1 ∧ all_bloom_together d2 ∧ all_bloom_together d3)

theorem find_bloom_day :
  ∃ d : days, blooms_simultaneously d :=
sorry

end find_bloom_day_l814_814822


namespace value_of_f_f_3_l814_814286

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 2^x else x - 1

theorem value_of_f_f_3 : f (f 3) = 4 := by
  sorry

end value_of_f_f_3_l814_814286


namespace problem_1_problem_2_l814_814988

/-- Problem 1: If A ⊆ B, find the range of values for a --/
theorem problem_1 (a : ℝ) :
  (∀ x : ℝ, (0 < a * x + 1 ∧ a * x + 1 ≤ 5) → (-1/2 < x ∧ x ≤ 2)) →
  (a ∈ set.Ioo (-∞) (-8) ∪ set.Icc 2 2) :=
begin
  sorry
end

/-- Problem 2: If B ⊆ A, find the range of values for a --/
theorem problem_2 (a : ℝ) :
  (∀ x : ℝ, (-1/2 < x ∧ x ≤ 2) → (0 < a * x + 1 ∧ a * x + 1 ≤ 5)) →
  (a ∈ set.Icc 0 0 ∪ set.Ioo 0 2) :=
begin
  sorry
end

end problem_1_problem_2_l814_814988


namespace clock_ticks_six_times_l814_814915

-- Define the conditions
def time_between_ticks (ticks : Nat) : Nat :=
  ticks - 1

def interval_duration (total_time : Nat) (ticks : Nat) : Nat :=
  total_time / time_between_ticks ticks

def number_of_ticks (total_time : Nat) (interval_time : Nat) : Nat :=
  total_time / interval_time + 1

-- Given conditions
def specific_time_intervals : Nat := 30
def eight_oclock_intervals : Nat := 42

-- Proven result
theorem clock_ticks_six_times : number_of_ticks specific_time_intervals (interval_duration eight_oclock_intervals 8) = 6 := 
sorry

end clock_ticks_six_times_l814_814915


namespace chord_length_is_correct_tangent_lines_equations_l814_814062

-- Defining the circle and the line
def circle (x y : ℝ) := (x - 1)^2 + (y - 3)^2 = 9
def line (x y : ℝ) := x - y = 0

-- Given points and the center
def point_P : ℝ × ℝ := (4, -3)
def center : ℝ × ℝ := (1, 3)
def radius : ℝ := 3

-- Distance formula from center to the line
def distance_from_center_to_line : ℝ :=
  abs (1 * 1 + (-1) * 3 + 0) / real.sqrt ((1:ℝ)^2 + (-1:ℝ)^2)

-- Length of chord
def chord_length : ℝ :=
  2 * real.sqrt (radius ^ 2 - distance_from_center_to_line ^ 2)

-- Theorems to state the conclusions to prove

-- Part 1: Proving the length of the chord is 2sqrt(7)
theorem chord_length_is_correct :
  chord_length = 2 * real.sqrt 7 :=
sorry

-- Part 2: Proving the equations of the tangent lines
theorem tangent_lines_equations :
  (point_P.1 = 4 ∧ (3 * point_P.1 + 4 * point_P.2 = 0 ∨ 3 * point_P.1 + 4 * point_P.2 = 0)) :=
sorry

end chord_length_is_correct_tangent_lines_equations_l814_814062


namespace find_x_such_that_sqrt_x_minus_six_eq_ten_l814_814218

theorem find_x_such_that_sqrt_x_minus_six_eq_ten : 
  ∀ x : ℝ, (sqrt (x - 6) = 10) → (x = 106) :=
by
  intro x
  intro h
  sorry

end find_x_such_that_sqrt_x_minus_six_eq_ten_l814_814218


namespace value_range_func_interval_l814_814443

noncomputable def func (x : ℝ) : ℝ :=
  1 / 2 * Real.exp x * (Real.sin x + Real.cos x)

theorem value_range_func_interval :
  set.image func (set.Icc 0 (Real.pi / 2)) = set.Icc (1 / 2) (1 / 2 * Real.exp (Real.pi / 2)) :=
by
  sorry

end value_range_func_interval_l814_814443


namespace count_abundant_numbers_less_than_50_l814_814315

open_locale classical

def is_abundant (n : ℕ) : Prop :=
  ∑ m in finset.filter (λ m, m ∣ n ∧ m < n) (finset.range n), m > n

theorem count_abundant_numbers_less_than_50 :
  (finset.filter (λ n, is_abundant n) (finset.range 50)).card = 9 :=
by {
  sorry
}

end count_abundant_numbers_less_than_50_l814_814315


namespace abs_neg_three_l814_814870

theorem abs_neg_three : abs (-3) = 3 :=
by 
  sorry

end abs_neg_three_l814_814870


namespace valid_number_of_beads_l814_814169

def Shape := ℕ -- simple definition for the shapes: cubes, pyramids
def Color := ℕ -- simple definition for the colors: green, blue, red

-- Definitions for our specific beads
def Kg : Shape × Color := (1, 1) -- green cube
def Kb : Shape × Color := (1, 2) -- blue cube
def Pr : Shape × Color := (2, 3) -- red pyramid
def Pb : Shape × Color := (2, 2) -- blue pyramid

-- Conditions given in the problem
def is_valid_necklace (necklace : List (Shape × Color)) : Prop :=
  ∀ i, (necklace.nth i ≠ none) →
      -- different shapes and different colors condition
      let ⟨(s1, c1)⟩ := necklace.get (i % necklace.length);
      let ⟨(s2, c2)⟩ := necklace.get ((i + 1) % necklace.length);
      s1 ≠ s2 ∧ c1 ≠ c2

-- Prove that the necklace must have a valid even count of beads and specifically 8 or 10.
theorem valid_number_of_beads : 
  ∀ (necklace : List (Shape × Color)), 
  is_valid_necklace necklace →
  (necklace.length = 8 ∨ necklace.length = 10) :=
by
  intros necklace h
  sorry

end valid_number_of_beads_l814_814169


namespace people_not_in_any_club_l814_814448

def num_people_company := 120
def num_people_club_A := 25
def num_people_club_B := 34
def num_people_club_C := 21
def num_people_club_D := 16
def num_people_club_E := 10
def overlap_C_D := 8
def overlap_D_E := 4

theorem people_not_in_any_club :
  num_people_company - 
  (num_people_club_A + num_people_club_B + 
  (num_people_club_C + (num_people_club_D - overlap_C_D) + (num_people_club_E - overlap_D_E))) = 26 :=
by
  unfold num_people_company num_people_club_A num_people_club_B num_people_club_C num_people_club_D num_people_club_E overlap_C_D overlap_D_E
  sorry

end people_not_in_any_club_l814_814448


namespace probability_of_triangle_formation_l814_814944

def sticks : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

noncomputable def valid_triangle_combinations (s : List ℕ) : ℕ :=
  (s.erase 2).erase 3.length -- This is simplified, for illustration. Properly computing valid set count is required.

def total_combinations : ℕ := (List.choose sticks 3).length

def probability_triangle : ℚ := valid_triangle_combinations sticks / total_combinations

theorem probability_of_triangle_formation : probability_triangle = 1 / 4 := by
  sorry

end probability_of_triangle_formation_l814_814944


namespace roots_cosines_of_triangle_l814_814066

-- Condition: polynomial p(x) has three positive real roots
variables {a b c : ℝ}

-- Definition of the polynomial
def p (x : ℝ) := x^3 + a*x^2 + b*x + c

theorem roots_cosines_of_triangle (h_pos_roots : ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ p x₁ = 0 ∧ p x₂ = 0 ∧ p x₃ = 0) :
  ∃ A B C : ℝ, 
    (A + B + C = π) ∧ 
    (a^2 - 2*b - 2*c = 1) :=
sorry

end roots_cosines_of_triangle_l814_814066


namespace g_at_6_l814_814746

def g (x : ℝ) : ℝ := 2 * x^4 - 13 * x^3 + 28 * x^2 - 32 * x - 48

theorem g_at_6 : g 6 = 552 :=
by sorry

end g_at_6_l814_814746


namespace correct_result_l814_814474

theorem correct_result (x : ℕ) (h: (325 - x) * 5 = 1500) : 325 - x * 5 = 200 := 
by
  -- placeholder for proof
  sorry

end correct_result_l814_814474


namespace find_natural_number_A_l814_814882

theorem find_natural_number_A :
  ∃ A : ℕ, (∑ k in finset.range A, A % (k + 1)) = A ∧ A = 10 :=
by
  use 10
  sorry -- Proof is omitted as instructed

end find_natural_number_A_l814_814882


namespace induction_proof_l814_814033

theorem induction_proof (n : ℕ) (h : n > 0) :
    (∑ i in range n, (2 * i + 1)^2 - (2 * (i + 1))^2) = -n * (2 * n + 1) := by
  -- We can assume the induction hypothesis and proceed with the proof,
  sorry

end induction_proof_l814_814033


namespace max_consecutive_semi_prime_l814_814549

-- Define a semi-prime number as the sum of two distinct primes greater than 25
def is_semi_prime (n : ℕ) : Prop := 
  n > 25 ∧ ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n = p + q

-- Define the statement about the maximum number of consecutive semi-prime numbers
theorem max_consecutive_semi_prime : 
  ∃ k, (∀ n, is_semi_prime ((2*n)+1) ∧ is_semi_prime ((2*n)+3) ∧ is_semi_prime ((2*n)+5) → k ≤ 5) := 
begin
  -- Proof here
  sorry
end

end max_consecutive_semi_prime_l814_814549


namespace smallest_x_y_sum_l814_814251

theorem smallest_x_y_sum :
  ∃ x y : ℕ,
    0 < x ∧ 0 < y ∧ x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 15) ∧ (x + y = 64) := 
by
  sorry

end smallest_x_y_sum_l814_814251


namespace Blaine_wins_l814_814552

-- Define the probabilities for Amelia, Blaine and Charlie
def Amelia_heads_prob : ℚ := 1 / 4
def Blaine_heads_prob : ℚ := 1 / 3
def Charlie_heads_prob : ℚ := 1 / 5

-- Define the probability that Blaine wins the game
def Blaine_wins_prob : ℚ :=
  let first_cycle_win := (3 / 4) * (1 / 3)
  let subsequent_cycle_win := (2 / 5) * (1 / 3)
  first_cycle_win + ∑' n, (2 / 5)^n * subsequent_cycle_win

-- Main statement to prove
theorem Blaine_wins (p q : ℕ) (h_coprime : Nat.coprime p q) (h_prob : Blaine_wins_prob = p / q) : q - p = 11 :=
by
  -- Hypotheses
  have h1 : Blaine_wins_prob = 25 / 36 := sorry
  have h2 : Nat.coprime 25 36 := sorry
  -- Proof
  have h3 : 36 - 25 = 11 := sorry
  exact h3

end Blaine_wins_l814_814552


namespace find_t_l814_814985

open Real

def collinear {A : Point ℝ} {B : Point ℝ} {C : Point ℝ} : Prop :=
  ∃ k : ℝ, k • (B - A) = (C - A)

theorem find_t (t : ℝ) : collinear (1, 2) (-3, 4) (2, t) → t = 3 / 2 := by
  sorry

end find_t_l814_814985


namespace part1_daily_sales_profit_part2_maximum_daily_profit_l814_814415

-- Definitions of initial conditions
def original_price : ℝ := 30
def original_sales_volume : ℝ := 60
def cost_price : ℝ := 15
def price_reduction_effect : ℝ := 10

-- Part 1: Prove the daily sales profit if the price is reduced by 2 yuan
def new_price_after_reduction (reduction : ℝ) : ℝ := original_price - reduction
def new_sales_volume (reduction : ℝ) : ℝ := original_sales_volume + reduction * price_reduction_effect
def profit_per_kg (selling_price : ℝ) : ℝ := selling_price - cost_price
def daily_sales_profit (reduction : ℝ) : ℝ := profit_per_kg (new_price_after_reduction reduction) * new_sales_volume reduction

theorem part1_daily_sales_profit : daily_sales_profit 2 = 1040 := by sorry

-- Part 2: Prove the selling price for maximum profit and the maximum profit
def profit_function (x : ℝ) : ℝ := (x - cost_price) * (original_sales_volume + (original_price - x) * price_reduction_effect)

theorem part2_maximum_daily_profit : 
  ∃ x, profit_function x = 1102.5 ∧ x = 51 / 2 := by sorry

end part1_daily_sales_profit_part2_maximum_daily_profit_l814_814415


namespace determine_omega_l814_814645

theorem determine_omega (ω : ℝ) : 
  (∀ x, f(x) = sin (ω * x + (π / 3))) ∧ (∀ d, is_adjacent_symmetry_axes_distance f d ∧ d = π) → ω = 1 := 
by 
  -- This is the problem statement, the proof steps would follow here
  sorry

end determine_omega_l814_814645


namespace find_base_of_isosceles_triangle_l814_814419

variables (α S : Real)

def isosceles_triangle_base (α S : Real) : Real :=
  (2 * Real.sqrt (3 * S * Real.sin (α / 2))) / 3

theorem find_base_of_isosceles_triangle (α S : Real) :
  isosceles_triangle_base α S = (2 * Real.sqrt (3 * S * Real.sin (α / 2))) / 3 :=
by
  sorry

end find_base_of_isosceles_triangle_l814_814419


namespace find_a_l814_814737

-- Definitions of the sets based on given conditions
def A : Set ℝ := { x | x^2 - 4 ≤ 0 }
def B (a : ℝ) : Set ℝ := { x | 2 * x + a ≤ 0 }
def intersectionAB : Set ℝ := { x | -2 ≤ x ∧ x ≤ 1 }

-- The theorem to prove
theorem find_a (a : ℝ) (h : A ∩ B a = intersectionAB) : a = -2 := by
  sorry

end find_a_l814_814737


namespace find_functions_l814_814207

variable (f : ℝ → ℝ)

theorem find_functions (h : ∀ x y : ℝ, f (x + f y) = f x + f y ^ 2 + 2 * x * f y) :
  ∃ c : ℝ, (∀ x, f x = x ^ 2 + c) ∨ (∀ x, f x = 0) :=
by
  sorry

end find_functions_l814_814207


namespace probability_of_drawing_three_white_balls_l814_814511

theorem probability_of_drawing_three_white_balls
  (total_balls white_balls black_balls: ℕ)
  (h_total: total_balls = 15)
  (h_white: white_balls = 7)
  (h_black: black_balls = 8)
  (draws: ℕ)
  (h_draws: draws = 3) :
  (Nat.choose white_balls draws / Nat.choose total_balls draws) = (7 / 91) :=
by sorry

end probability_of_drawing_three_white_balls_l814_814511


namespace intervals_of_monotonicity_and_range_of_c_l814_814647

def f (x c : ℝ) : ℝ := x^3 - (1/2) * x^2 - 2 * x + c

theorem intervals_of_monotonicity_and_range_of_c (c : ℝ) :
  (∀ x : ℝ, 
    (has_deriv_at (λ x : ℝ, x^3 - (1/2) * x^2 - 2 * x + c) (3 * x^2 - x - 2) x)) ∧ 
  ((∀ x : ℝ, 
    (x < -2/3 → monotone_incr (λ x : ℝ, x^3 - (1/2) * x^2 - 2 * x + c x)) ∧
    (-2/3 < x ∧ x < 1 → monotone_decr (λ x : ℝ, x^3 - (1/2) * x^2 - 2 * x + c x)) ∧
    (x > 1 → monotone_incr (λ x : ℝ, x^3 - (1/2) * x^2 - 2 * x + c x)))) ∧
  (∀ x ∈ Icc (-1 : ℝ) 2, f x c < c^2 → c < -1 ∨ c > 2) :=
by sorry

end intervals_of_monotonicity_and_range_of_c_l814_814647


namespace area_FDBG_l814_814085

open Real

variables {A B C D E F G : Point}

/-- Conditions: 
    - Triangle ABC with AB = 60, AC = 20, and area = 240
    - D is the midpoint of AB
    - E is the midpoint of AC
    - Angle bisector of ∠BAC intersects DE at F and BC at G
--/
def conditions (A B C D E F G : Point) :=
  distance A B = 60 ∧
  distance A C = 20 ∧
  area A B C = 240 ∧
  midpoint D A B ∧
  midpoint E A C ∧
  angle_bisector_intersects F A B C D E G

/-- Proof problem: The area of quadrilateral FDBG is 180 --/
theorem area_FDBG (A B C D E F G : Point) (h : conditions A B C D E F G) :
  area F D B G = 180 :=
sorry

end area_FDBG_l814_814085


namespace sum_of_roots_l814_814958

noncomputable def poly : Polynomial ℝ :=
  Polynomial.sum $ List.ofFn (λ k, Polynomial.monomial (k + 1) (1 : ℝ) * Polynomial.X ^ (2008 - k))

theorem sum_of_roots :
  let p := (0, Polynomial.sum $ List.mapWithIndex (λ k coeff, Polynomial.monomial coeff (Polynomial.X - (k + 1)) ^ (2008 - k)) $ List.range 2008)
  Vieta.sumRoots p.2 = 2006 :=
by sorry

end sum_of_roots_l814_814958


namespace log_inequality_range_l814_814319

theorem log_inequality_range (a : ℝ) : log a (2/3) < 1 → (0 < a ∧ a < 2/3) ∨ (1 < a) :=
by
  sorry

end log_inequality_range_l814_814319


namespace haley_trees_abs_diff_l814_814659

/-- Haley had 167 trees initially. After a storm, 84 trees died. Prove that the absolute difference
between the number of trees that survived and the number of trees that died is 1. -/
theorem haley_trees_abs_diff (initial_trees died_trees : ℕ) (h_initial : initial_trees = 167) 
  (h_died : died_trees = 84) : abs ((initial_trees - died_trees) - died_trees) = 1 := by
  sorry

end haley_trees_abs_diff_l814_814659


namespace abundant_numbers_less_than_50_eq_9_l814_814304

def proper_factors (n : ℕ) : List ℕ :=
  List.filter (λ m, m < n ∧ n % m = 0) (List.range n)

def sum_proper_factors (n : ℕ) : ℕ :=
  (proper_factors n).sum

def is_abundant (n : ℕ) : Prop :=
  sum_proper_factors n > n

def abundant_numbers_count_below (m : ℕ) : ℕ :=
  (List.range m).filter is_abundant |>.length

theorem abundant_numbers_less_than_50_eq_9 :
  abundant_numbers_count_below 50 = 9 :=
by
  sorry

end abundant_numbers_less_than_50_eq_9_l814_814304


namespace find_scalars_l814_814444

noncomputable def vector_a : ℝ × ℝ × ℝ := (1, 1, 1)
noncomputable def vector_b : ℝ × ℝ × ℝ := (2, -3, 1)
noncomputable def vector_c : ℝ × ℝ × ℝ := (4, 1, -5)
noncomputable def target_vector : ℝ × ℝ × ℝ := (5, -3, 2)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_scalars :
  ∃ s t u : ℝ, target_vector = (s * vector_a.1 + t * vector_b.1 + u * vector_c.1,
                                 s * vector_a.2 + t * vector_b.2 + u * vector_c.2,
                                 s * vector_a.3 + t * vector_b.3 + u * vector_c.3) ∧
             s = 4 / 3 ∧ t = 10 / 7 ∧ u = -3 / 7 :=
begin
  use [4 / 3, 10 / 7, -3 / 7],
  split,
  { split,
    { simp [vector_a, vector_b, vector_c, target_vector] },
    { split,
      { exact rfl },
      { split,
        { exact rfl },
        { exact rfl } } } }
end

end find_scalars_l814_814444


namespace range_of_b_l814_814754

noncomputable def f (a x : ℝ) : ℝ := 
  Real.log x - (1 / 2) * a * x^2 - 2 * x

theorem range_of_b (a : ℝ) (b : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → a ∈ Set.Ico (-1 : ℝ) (0 : ℝ) → f a x < b) ↔ b > -3 / 2 :=
by
  sorry

end range_of_b_l814_814754


namespace memorable_telephone_numbers_l814_814925

theorem memorable_telephone_numbers : ∃ (P : ℕ → Prop), 
  (∀ t : ℕ, P t ↔ 0 ≤ t ∧ t < 10) ∧
  (∃ d_1 d_2 d_3 d_4 d_5 d_6 d_7 : ℕ,
    0 ≤ d_1 ∧ d_1 < 10 ∧
    0 ≤ d_2 ∧ d_2 < 10 ∧
    0 ≤ d_3 ∧ d_3 < 10 ∧
    0 ≤ d_4 ∧ d_4 < 10 ∧
    0 ≤ d_5 ∧ d_5 < 10 ∧
    0 ≤ d_6 ∧ d_6 < 10 ∧
    0 ≤ d_7 ∧ d_7 < 10 ∧
    ((d_1 = d_4 ∧ d_2 = d_5 ∧ d_3 = d_6) ∨ 
    (d_1 = d_5 ∧ d_2 = d_6 ∧ d_3 = d_7))) ∧
  (19990 = (
    let telephone_num := 10 * 10 * 10 * 10 in 
    let memorable_non_overlap := telephone_num * 2 in
    memorable_non_overlap - 10)) :=
begin
  sorry
end

end memorable_telephone_numbers_l814_814925


namespace correct_statements_about_task_assignments_l814_814225

theorem correct_statements_about_task_assignments :
  let students := ["A", "B", "C", "D", "E"]
  let tasks := ["translation", "tour guide", "etiquette", "driver"]
  let num_assignments := 4^5 -- This corresponds to option A.
  let task_constraints_valid  := 
    (students |>.erase "A" |>.erase "B" ) = ["C", "D", "E"] -- A and B can't drive but can do other tasks.
    ∧ (A_tasks := 3) ∧ (driver_tasks := 4) -- 3 tasks for A and driver task.
    ∧ (assignments_case_1 := comb 3 1 * comb 4 2 * (3^3)) -- Selecting 1 driver from C, D, E.
    ∧ (assignments_case_2 := comb 3 2 * (3^3)) -- Selecting 2 drivers from C, D, E.
    ∧ (total_assignments_D := assign_case_1 + assign_case_2) 
  ( (num_assignments == 4^5) ∧ (total_assignments_D == comb 3 1 * comb 4 2 * 3^3 + comb 3 2 * 3^3) ) :=
  sorry

end correct_statements_about_task_assignments_l814_814225


namespace ramu_profit_percent_correct_l814_814771

noncomputable def cost_of_car : ℝ := 36400
noncomputable def repair_costs : ℝ := 8000
noncomputable def taxes : ℝ := 4500
noncomputable def insurance : ℝ := 2500
noncomputable def selling_price : ℝ := 68400

def total_cost := cost_of_car + repair_costs + taxes + insurance
def profit := selling_price - total_cost

def profit_percent := (profit / total_cost) * 100

theorem ramu_profit_percent_correct : profit_percent = 33.07 := by
  sorry

end ramu_profit_percent_correct_l814_814771


namespace arithmetic_sequence_n_terms_l814_814602

theorem arithmetic_sequence_n_terms:
  ∀ (a₁ d aₙ n: ℕ), 
  a₁ = 6 → d = 3 → aₙ = 300 → aₙ = a₁ + (n - 1) * d → n = 99 :=
by
  intros a₁ d aₙ n h1 h2 h3 h4
  sorry

end arithmetic_sequence_n_terms_l814_814602


namespace goose_eggs_laid_l814_814914

theorem goose_eggs_laid (E : ℕ) 
    (H1 : ∃ h, h = (2 / 5) * E)
    (H2 : ∃ m, m = (11 / 15) * h)
    (H3 : ∃ s, s = (1 / 4) * m)
    (H4 : ∃ y, y = (2 / 7) * s)
    (H5 : y = 150) : 
    E = 7160 := 
sorry

end goose_eggs_laid_l814_814914


namespace smallest_n_for_conditions_l814_814569

theorem smallest_n_for_conditions :
  ∃ (n : ℕ), ∃ (x : Fin n → ℝ), (∀ i : Fin n, 0 < x i) ∧ 
             (∑ i : Fin n, x i = 500) ∧ 
             (∑ i : Fin n, (x i)^3 = 125000) ∧
             n = 64 :=
by
  sorry

end smallest_n_for_conditions_l814_814569


namespace exponent_subtraction_l814_814613

theorem exponent_subtraction (a : ℝ) (m n : ℕ) (h1 : a^m = 6) (h2 : a^n = 2) : a^(m - n) = 3 := by
  sorry

end exponent_subtraction_l814_814613


namespace sandy_jacket_price_l814_814402

noncomputable def discounted_shirt_price (initial_shirt_price discount_percentage : ℝ) : ℝ :=
  initial_shirt_price - (initial_shirt_price * discount_percentage / 100)

noncomputable def money_left (initial_money additional_money discounted_price : ℝ) : ℝ :=
  initial_money + additional_money - discounted_price

noncomputable def jacket_price_before_tax (remaining_money tax_percentage : ℝ) : ℝ :=
  remaining_money / (1 + tax_percentage / 100)

theorem sandy_jacket_price :
  let initial_money := 13.99
  let initial_shirt_price := 12.14
  let discount_percentage := 5.0
  let additional_money := 7.43
  let tax_percentage := 10.0
  
  let discounted_price := discounted_shirt_price initial_shirt_price discount_percentage
  let remaining_money := money_left initial_money additional_money discounted_price
  
  jacket_price_before_tax remaining_money tax_percentage = 8.99 := sorry

end sandy_jacket_price_l814_814402


namespace compound_percentage_increase_l814_814730

noncomputable def weeklyEarningsAfterRaises (initial : ℝ) (raises : List ℝ) : ℝ :=
  raises.foldl (λ sal raise_rate => sal * (1 + raise_rate / 100)) initial

theorem compound_percentage_increase :
  let initial := 60
  let raises := [10, 15, 12, 8]
  weeklyEarningsAfterRaises initial raises = 91.80864 ∧
  ((weeklyEarningsAfterRaises initial raises - initial) / initial * 100 = 53.0144) :=
by
  sorry

end compound_percentage_increase_l814_814730


namespace snake_count_l814_814394

def neighborhood : Type := {n : ℕ // n = 200}

def percentage (total : ℕ) (percent : ℕ) : ℕ := total * percent / 100

def owns_only_dogs (total : ℕ) : ℕ := percentage total 13
def owns_only_cats (total : ℕ) : ℕ := percentage total 10
def owns_only_snakes (total : ℕ) : ℕ := percentage total 5
def owns_only_rabbits (total : ℕ) : ℕ := percentage total 7
def owns_only_birds (total : ℕ) : ℕ := percentage total 3
def owns_only_exotic (total : ℕ) : ℕ := percentage total 6
def owns_dogs_and_cats (total : ℕ) : ℕ := percentage total 8
def owns_dogs_cats_exotic (total : ℕ) : ℕ := percentage total 9
def owns_cats_and_snakes (total : ℕ) : ℕ := percentage total 4
def owns_cats_and_birds (total : ℕ) : ℕ := percentage total 2
def owns_snakes_and_rabbits (total : ℕ) : ℕ := percentage total 5
def owns_snakes_and_birds (total : ℕ) : ℕ := percentage total 3
def owns_rabbits_and_birds (total : ℕ) : ℕ := percentage total 1
def owns_all_except_snakes (total : ℕ) : ℕ := percentage total 2
def owns_all_except_birds (total : ℕ) : ℕ := percentage total 1
def owns_three_with_exotic (total : ℕ) : ℕ := percentage total 11
def owns_only_chameleons (total : ℕ) : ℕ := percentage total 3
def owns_only_hedgehogs (total : ℕ) : ℕ := percentage total 2

def exotic_pet_owners (total : ℕ) : ℕ :=
  owns_only_exotic total + owns_dogs_cats_exotic total + owns_all_except_snakes total +
  owns_all_except_birds total + owns_three_with_exotic total + owns_only_chameleons total +
  owns_only_hedgehogs total

def exotic_pet_owners_with_snakes (total : ℕ) : ℕ :=
  percentage (exotic_pet_owners total) 25

def total_snake_owners (total : ℕ) : ℕ :=
  owns_only_snakes total + owns_cats_and_snakes total +
  owns_snakes_and_rabbits total + owns_snakes_and_birds total +
  exotic_pet_owners_with_snakes total

theorem snake_count (nh : neighborhood) : total_snake_owners (nh.val) = 51 :=
by
  sorry

end snake_count_l814_814394


namespace probability_of_shortest_diagonal_15sided_polygon_eq_one_sixth_l814_814099

-- Define a regular 15-sided polygon
def is_regular_15sided_polygon (n : ℕ) := n = 15

-- Define the formula to calculate the number of diagonals of a n-sided polygon
def number_of_diagonals (n : ℕ) := n * (n - 3) / 2

-- Define the number of shortest diagonals for a regular 15-sided polygon
def number_of_shortest_diagonals (n : ℕ) := n

-- Define the probability of selecting one of the shortest diagonals
def probability_of_shortest_diagonal (n : ℕ) : ℚ :=
  (number_of_shortest_diagonals n) / (number_of_diagonals n)

-- The main theorem to prove
theorem probability_of_shortest_diagonal_15sided_polygon_eq_one_sixth :
  ∀ n, is_regular_15sided_polygon n → probability_of_shortest_diagonal n = 1 / 6 :=
by
  intros n h
  rw [is_regular_15sided_polygon, number_of_diagonals, number_of_shortest_diagonals, probability_of_shortest_diagonal] at h
  sorry

end probability_of_shortest_diagonal_15sided_polygon_eq_one_sixth_l814_814099


namespace Запад_oil_output_per_capita_Не_Запад_oil_output_per_capita_Россия_oil_output_per_capita_l814_814864

noncomputable def oil_output_per_capita (total_output : ℝ) (population : ℝ) : ℝ := total_output / population

theorem Запад_oil_output_per_capita :
  oil_output_per_capita 55.084 1 = 55.084 :=
by
  sorry

theorem Не_Запад_oil_output_per_capita :
  oil_output_per_capita 1480.689 6.9 = 214.59 :=
by
  sorry

theorem Россия_oil_output_per_capita :
  oil_output_per_capita (13737.1 * 100 / 9) 147 = 1038.33 :=
by
  sorry

end Запад_oil_output_per_capita_Не_Запад_oil_output_per_capita_Россия_oil_output_per_capita_l814_814864


namespace length_AD_l814_814631

noncomputable def angle_between_vectors (m n : ℝ) : ℝ := Real.angle m n

theorem length_AD
  (m n : ℝ → ℝ → ℝ)
  (H1 : angle_between_vectors m n = π / 6)
  (H2 : ∥m∥ = Real.sqrt 3)
  (H3 : ∥n∥ = 2)
  (AB AC : ℝ → ℝ → ℝ)
  (H4 : AB = 2 • m + 2 • n)
  (H5 : AC = 2 • m - 6 • n)
  (AD D BC : ℝ → ℝ → ℝ)
  (H6 : D = midpoint B C)
  (H7 : AD = AB + (1 / 2) • BC) :
  ∥AD∥ = 4 := 
sorry

end length_AD_l814_814631


namespace child_height_at_age_10_l814_814893

theorem child_height_at_age_10 (x y : ℝ) (h : y = 7.19 * x + 73.93) (hx : x = 10) : abs (y - 145.83) < 1 :=
by {
  sorry
}

end child_height_at_age_10_l814_814893


namespace max_reverse_sum_l814_814932

theorem max_reverse_sum (a b : ℕ) (ha1: 100 ≤ a) (ha2: a < 1000) (hb1: 100 ≤ b) (hb2: b < 1000) (h₁: (a % 10 ≠ 0)) (h₂: (b % 10 ≠ 0)) (h : a + b = 1372) :
  let reverse := λ n, (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100) in
  reverse a + reverse b = 1372 :=
by
  sorry

end max_reverse_sum_l814_814932


namespace quadrilateral_parallel_l814_814172

open Classical

variables {α : Type} [NormedField α] [NormedSpace ℝ α]

structure Point (α) : Type :=
(x y : α)

def parallel_lines (A B C D : Point α) : Prop :=
∃ k : α, (B.y - A.y) = k * (B.x - A.x) ∧ (D.y - C.y) = k * (D.x - C.x)

theorem quadrilateral_parallel
  (A B C D P : Point α)
  (hBCAD : parallel_lines B C A D)
  (hP_on_AB : ∃ r : α, P.x = A.x + r * (B.x - A.x) ∧ P.y = A.y + r * (B.y - A.y)) :
  ∃ E : Point α,
    (∃ k1 : α, E.y = A.y + k1 * (P.y - C.y) ∧ E.x = A.x + k1 * (P.x - C.x)) ∧
    (∃ k2 : α, E.y = B.y + k2 * (P.y - D.y) ∧ E.x = B.x + k2 * (P.x - D.x)) :=
sorry

end quadrilateral_parallel_l814_814172


namespace al_barb_rest_days_l814_814906

theorem al_barb_rest_days:
  let al_cycle := 6
  let barb_cycle := 6
  let common_cycle := Nat.lcm al_cycle barb_cycle
  let total_days := 1000
  let cycles := total_days / common_cycle
  in (166 : ℕ) = 1 * cycles :=
  by 
  sorry

end al_barb_rest_days_l814_814906


namespace greatest_mean_weight_l814_814125

variable (X Y Z : Type) [Group X] [Group Y] [Group Z]

theorem greatest_mean_weight 
  (mean_X : ℝ) (mean_Y : ℝ) (mean_XY : ℝ) (mean_XZ : ℝ)
  (hX : mean_X = 30)
  (hY : mean_Y = 70)
  (hXY : mean_XY = 50)
  (hXZ : mean_XZ = 40) :
  ∃ k : ℝ, k = 70 :=
by {
  sorry
}

end greatest_mean_weight_l814_814125


namespace probability_three_white_balls_l814_814506

theorem probability_three_white_balls (total_balls: ℕ) (white_balls: ℕ) (black_balls: ℕ) (drawn_balls: ℕ) 
    (h_total: total_balls = 15) (h_white: white_balls = 7) (h_black: black_balls = 8) (h_drawn: drawn_balls = 3) : 
    ((choose white_balls drawn_balls) / (choose total_balls drawn_balls) : ℚ) = 1 / 13 := 
by {
    -- Definitions and conditions come from part (a)
    -- The lean code should be able to be built successfully
    sorry
} 

end probability_three_white_balls_l814_814506


namespace expected_stand_ups_expected_no_stand_l814_814020

theorem expected_stand_ups (n : ℕ) : (∑ i in finset.range (n - 1), i)/2 = n * (n - 1) / 4 := 
sorry

theorem expected_no_stand (n : ℕ) : 
  ∑ k in finset.range n, 1/(k+1) = 1 + ∑ k in finset.range (n - 1), 1/(k + 2) :=
sorry

end expected_stand_ups_expected_no_stand_l814_814020


namespace shadow_length_l814_814888

theorem shadow_length (h_flagstaff : ℝ) (s_flagstaff : ℝ) (h_building : ℝ) : 
  (h_flagstaff = 17.5) → (s_flagstaff = 40.25) → (h_building = 12.5) → 
  ∃ s_building : ℝ, (h_flagstaff / s_flagstaff = h_building / s_building) ∧ s_building = 28.75 :=
by
  intros hf sf hb
  use 28.75
  split
  sorry -- Prove the ratio equality here
  exact rfl -- Prove that s_building is indeed 28.75

end shadow_length_l814_814888


namespace point_not_on_line_l814_814322

theorem point_not_on_line (m b : ℝ) (h1 : m > 2) (h2 : m * b > 0) : ¬ (b = -2023) :=
by
  sorry

end point_not_on_line_l814_814322


namespace tank_filled_time_l814_814116

-- Definitions for the rates
def rate_first_tap : ℝ := 1 / 20
def rate_second_tap : ℝ := 1 / 60

-- Combined rate with both taps open
def combined_rate : ℝ := rate_first_tap + rate_second_tap

-- Amount filled in the first 10 minutes
def amount_filled_10_min : ℝ := 10 * combined_rate

-- Remaining fraction of the tank
def remaining_tank : ℝ := 1 - amount_filled_10_min

-- Define the total remaining time needed by the second tap to fill the rest of the tank
def remaining_time_for_tank (remaining_fraction : ℝ) : ℝ := remaining_fraction / rate_second_tap

theorem tank_filled_time :
  remaining_time_for_tank remaining_tank = 20 :=
by
  sorry

end tank_filled_time_l814_814116


namespace problem1_problem2_l814_814547

-- Define problem 1 conditions and expected result
theorem problem1:
  ( ((25 / 9) ^ (1 / 2)) + (0.1 ^ (-2)) + ((27 / 64) ^ (2 / 3)) - 3 * (Real.pi ^ 0) + (37 / 48) ) = 100 :=
by sorry
  
-- Define problem 2 conditions and expected result
theorem problem2:
  (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) + (Real.log 5) ^ 2 +
    Real.log 5 * Real.log 20 + (1 / 2) * Real.log 16 - (2 ^ (Real.log 3 / Real.log 2)) = 1 :=
by sorry

end problem1_problem2_l814_814547


namespace parabola_intersections_l814_814800

theorem parabola_intersections (p : ℝ → ℝ) (h_eq : p = λ x, 2 * x^2 - 4 * x + 1) :
    ∃ y_intersection x_intersections,
      y_intersection = 1 ∧
      x_intersections = { (2 + Real.sqrt 2) / 2, (2 - Real.sqrt 2) / 2 } ∧
      1 + 2 = 3 :=
by {
  sorry
}

end parabola_intersections_l814_814800


namespace shortest_midpoint_segment_l814_814638

variable (A B C D J K E O Z X : ℝ)
variable (AB CD AC BC AD BD : ℝ)

-- Conditions
axiom h1 : AB > CD
axiom h2 : CD > AC
axiom h3 : AC > BC
axiom h4 : BC > AD
axiom h5 : AD > BD

-- Define segments connecting midpoints of opposite edges
def midpoint_segment (x y : ℝ) := x / 2

noncomputable def KJ := midpoint_segment AC BD
noncomputable def others := [midpoint_segment AB CD, midpoint_segment AC BD, midpoint_segment AB AC, midpoint_segment AB BC]

-- Define the shortest segment connecting midpoints of opposite edges
theorem shortest_midpoint_segment : KJ = List.minimum others :=
  sorry

end shortest_midpoint_segment_l814_814638


namespace center_of_hyperbola_is_3_4_l814_814590

noncomputable def hyperbola_center : Prop :=
  ∃ c : ℝ × ℝ, 
    let x := c.1 in
    let y := c.2 in
    9 * x^2 - 54 * x - 16 * y^2 + 128 * y - 400 = 0 ∧
    c = (3, 4)

theorem center_of_hyperbola_is_3_4 : hyperbola_center :=
begin
  sorry
end

end center_of_hyperbola_is_3_4_l814_814590


namespace least_pos_base10_num_with_eight_binary_digits_l814_814468

theorem least_pos_base10_num_with_eight_binary_digits :
  ∃ n : ℕ, (n ≥ 1) ∧ (nat.binary_bits n = 8) ∧ n = 128 := by
  sorry

end least_pos_base10_num_with_eight_binary_digits_l814_814468


namespace equal_area_division_l814_814146

-- Define the setup of quarter circle and semicircle
structure Geometry (α : Type*) [LinearOrder α] :=
  (O A B : α)
  (n : ℕ)
  (quarter_circle : α)
  (semicircle : α)

-- Define the division of the diameter and perpendiculars
def divide_and_perpendiculars {α : Type*} [LinearOrder α] (geo : Geometry α) : Prop :=
  ∃ (D : ℕ → α) (D' : ℕ → α), 
  ∀ i : ℕ, i < geo.n → D i ∈ geo.quarter_circle ∧ D' i ∈ geo.semicircle ∧ 
           perpendicular (geo.A + i * (geo.B - geo.A) / geo.n) (D i) (D' i)

-- Prove the areas are divided equally
theorem equal_area_division {α : Type*} [LinearOrder α] (geo : Geometry α) 
  (h : divide_and_perpendiculars geo) 
  : ∀ i : ℕ, i < geo.n → 
    area (crescent_slice geo i) = area (crescent_slice geo (i + 1)) :=
begin
  sorry
end

end equal_area_division_l814_814146


namespace quadrilateral_equal_parts_l814_814826

-- Define the quadrilateral and its properties
structure ConvexQuadrilateral (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Assume points and basic properties in Euclidean space
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (quad : ConvexQuadrilateral A B C D)
variables (M N O : Point2 ℝ) -- midpoints and intersection point

-- Assuming the convex quadrilateral and required conditions:
axiom diag_midpoints_parallel : 
  (Midpoint(A, C) = M) ∧ 
  (Midpoint(B, D) = N) ∧ 
  (Parallel(M, Line(B, D))) ∧ 
  (Parallel(N, Line(A, C))) ∧ 
  (Intersection(Point(M, Parallel(N))), O)

-- Define the proof problem
theorem quadrilateral_equal_parts (quad : ConvexQuadrilateral A B C D) 
  (M N O : Point2 ℝ) :
  Segment(O, Midpoint(A, B)) ∧ 
  Segment(O, Midpoint(B, C)) ∧ 
  Segment(O, Midpoint(C, D)) ∧ 
  Segment(O, Midpoint(D, A)) 
  ⟹ Segments divide quadrilateral into equal areas :=
begin
  sorry
end

end quadrilateral_equal_parts_l814_814826


namespace find_angle_BFC_l814_814514

-- Definitions based on conditions
structure Triangle :=
  (A B C D E I_A K L F : Point)
  (touches_AB : ∃ inscribed_circle : Circle, touches inscribed_circle AB D)
  (touches_AC : ∃ inscribed_circle : Circle, touches inscribed_circle AC E)
  (is_excenter : ∃ excircle : Circle, touches_excircle opposite I_A BC)
  (is_midpoint_K : K = midpoint D I_A)
  (is_midpoint_L : L = midpoint E I_A)
  (intersection_BK_CL : intersects_at BK CL F)
  (angle_BAC : ∠ A B C = 50°)

-- Theorem statement
theorem find_angle_BFC (T : Triangle) : ∠ B F C = 130° :=
sorry

end find_angle_BFC_l814_814514


namespace quadratic_root_k_l814_814691

theorem quadratic_root_k (k : ℝ) : (∃ x : ℝ, x^2 - 2 * x + k = 0 ∧ x = 1) → k = 1 :=
by
  sorry

end quadratic_root_k_l814_814691


namespace max_a_value_in_fourth_quadrant_l814_814283

-- Define the complex number z
def z (a : ℤ) : ℂ := (2 + a * complex.I) / (1 + 2 * complex.I)

-- Define the real and imaginary parts of z
def realPart (a : ℤ) : ℝ := (2 + 2 * a) / 5
def imaginaryPart : ℝ := -4 / 5

-- Condition that z is in the fourth quadrant
def inFourthQuadrant (a : ℤ) : Prop :=
  realPart a > 0 ∧ imaginaryPart < 0

-- The proof problem statement
theorem max_a_value_in_fourth_quadrant :
  ∃ (a : ℤ), inFourthQuadrant a ∧ ∀ (b : ℤ), inFourthQuadrant b → b ≤ a :=
sorry

end max_a_value_in_fourth_quadrant_l814_814283


namespace math_expression_evaluation_l814_814927

theorem math_expression_evaluation :
  |1 - Real.sqrt 3| + 3 * Real.tan (Real.pi / 6) - (1/2)⁻¹ + (3 - Real.pi)^0 = 3.732 + Real.sqrt 3 := by
  sorry

end math_expression_evaluation_l814_814927


namespace remainder_x_2023_l814_814216

open Polynomial

-- Conditions
lemma poly_expansion_1 (x : ℚ) : (X^2 + 1)*(X - 1) = X^3 - X^2 + X - 1 := 
by ring

lemma poly_expansion_2 (x : ℚ) : (X^2 + 1)*(X - 1)*(X + 1) = X^4 - 1 := 
by ring

-- The Problem
theorem remainder_x_2023 (x : ℚ) : 
  mod_by_monic (X^2023) ((X^2 + 1)*(X - 1)) = X^3 :=
sorry

end remainder_x_2023_l814_814216


namespace transformed_sum_l814_814528

theorem transformed_sum (n : ℕ) (s : ℝ) (x : ℕ → ℝ) (h : ∑ i in finset.range n, x i = s) : 
  (∑ i in finset.range n, (3 * x i + 35) / 2) = (3 * s) / 2 + (35 * n) / 2 := 
by 
  sorry

end transformed_sum_l814_814528


namespace smallest_n_l814_814399

theorem smallest_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : 15 < n) : n = 52 :=
by
  sorry

end smallest_n_l814_814399


namespace Запад_oil_output_per_capita_Не_Запад_oil_output_per_capita_Россия_oil_output_per_capita_l814_814862

noncomputable def oil_output_per_capita (total_output : ℝ) (population : ℝ) : ℝ := total_output / population

theorem Запад_oil_output_per_capita :
  oil_output_per_capita 55.084 1 = 55.084 :=
by
  sorry

theorem Не_Запад_oil_output_per_capita :
  oil_output_per_capita 1480.689 6.9 = 214.59 :=
by
  sorry

theorem Россия_oil_output_per_capita :
  oil_output_per_capita (13737.1 * 100 / 9) 147 = 1038.33 :=
by
  sorry

end Запад_oil_output_per_capita_Не_Запад_oil_output_per_capita_Россия_oil_output_per_capita_l814_814862


namespace cos_x_in_terms_of_a_b_l814_814669

theorem cos_x_in_terms_of_a_b (a b x : ℝ)
  (h1 : tan x = (a^2 + b^2) / (2 * a * b))
  (h2 : a > b)
  (h3 : b > 0)
  (h4 : π/4 < x ∧ x < π/2) :
  cos x = (2 * a * b) / real.sqrt(a^4 + 6 * a^2 * b^2 + b^4) :=
sorry

end cos_x_in_terms_of_a_b_l814_814669


namespace min_n_value_l814_814753

noncomputable def S : Set ℕ := {1, 2, 3, 4}

def sequence_condition (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ B ⊆ S, B ≠ ∅ → ∃ i, B = (Finset.image a (Finset.range (B.to_finset.card))).subset_repeat

theorem min_n_value : ∃ n, sequence_condition (λ a, true) n ∧ ∀ m < n, ¬sequence_condition (λ a, true) m :=
sorry

end min_n_value_l814_814753


namespace probability_of_insight_l814_814519

noncomputable def students_in_both_classes : ℕ := 30 + 35 - 40

noncomputable def only_mandarin : ℕ := 30 - students_in_both_classes
noncomputable def only_german : ℕ := 35 - students_in_both_classes
noncomputable def total_ways_to_choose_2_students : ℕ := (Nat.choose 40 2)
noncomputable def ways_to_choose_2_only_mandarin : ℕ := (Nat.choose only_mandarin 2)
noncomputable def ways_to_choose_2_only_german : ℕ := (Nat.choose only_german 2)

theorem probability_of_insight : 
  (1 - ((ways_to_choose_2_only_mandarin + ways_to_choose_2_only_german) / total_ways_to_choose_2_students)) = (145 / 156) := 
by
  have h1 : students_in_both_classes = 25 := by sorry
  have h2 : only_mandarin = 5 := by sorry
  have h3 : only_german = 10 := by sorry
  have h4 : total_ways_to_choose_2_students = 780 := by sorry
  have h5 : ways_to_choose_2_only_mandarin = 10 := by sorry
  have h6 : ways_to_choose_2_only_german = 45 := by sorry
  sorry

end probability_of_insight_l814_814519


namespace percentage_divisible_by_7_l814_814844

-- Define the total integers and the condition for being divisible by 7
def total_ints := 140
def divisible_by_7 (n : ℕ) : Prop := n % 7 = 0

-- Calculate the number of integers between 1 and 140 that are divisible by 7
def count_divisible_by_7 : ℕ := Nat.succ (140 / 7)

-- The theorem to prove
theorem percentage_divisible_by_7 : (count_divisible_by_7 / total_ints : ℚ) * 100 = 14.29 := by
  sorry

end percentage_divisible_by_7_l814_814844


namespace candidate_X_expected_to_win_by_16_percent_l814_814335

theorem candidate_X_expected_to_win_by_16_percent :
  (∀ (R : ℝ), R > 0 → 
    let republicans := 3 * R in
    let democrats := 2 * R in
    let total_voters := republicans + democrats in
    let votes_for_X := (0.8 * republicans) + (0.25 * democrats) in
    let votes_for_Y := (0.2 * republicans) + (0.75 * democrats) in
    let difference := votes_for_X - votes_for_Y in
    (difference / total_voters) * 100 = 16) :=
by
  intros R hR
  let republicans := 3 * R
  let democrats := 2 * R
  let total_voters := republicans + democrats
  let votes_for_X := (0.8 * republicans) + (0.25 * democrats)
  let votes_for_Y := (0.2 * republicans) + (0.75 * democrats)
  let difference := votes_for_X - votes_for_Y
  have h1 : total_voters = 5 * R := by 
    sorry
  have h2 : votes_for_X = 2.9 * R := by 
    sorry
  have h3 : votes_for_Y = 2.1 * R := by 
    sorry
  have h4 : difference = 0.8 * R := by 
    sorry
  show (difference / total_voters) * 100 = 16
  by
    sorry

end candidate_X_expected_to_win_by_16_percent_l814_814335


namespace continuous_on_uniform_continuous_l814_814029

variable {α : Type*} {β : Type*}
variable [MetricSpace α] [LinearOrder α] [OrderTopology α]
variable [MetricSpace β]
variable {a b : α}
variable {f : α → β}

theorem continuous_on_uniform_continuous (h_cont : ContinuousOn f (set.Icc a b)) :
  UniformContinuousOn f (set.Icc a b) :=
by
  sorry

end continuous_on_uniform_continuous_l814_814029


namespace salt_amount_evaporation_l814_814358

-- Define the conditions as constants
def total_volume : ℕ := 2 -- 2 liters
def salt_concentration : ℝ := 0.2 -- 20%

-- The volume conversion factor from liters to milliliters.
def liter_to_ml : ℕ := 1000

-- Define the statement to prove
theorem salt_amount_evaporation : total_volume * (salt_concentration * liter_to_ml) = 400 := 
by 
  -- We'll skip the proof steps here
  sorry

end salt_amount_evaporation_l814_814358


namespace Q_moves_clockwise_with_2omega_l814_814690

noncomputable def pointP (ω t : ℝ) : ℝ × ℝ :=
  (Real.cos (ω * t), Real.sin (ω * t))

noncomputable def pointQ (x y : ℝ) : ℝ × ℝ :=
  (-2 * x * y, y ^ 2 - x ^ 2)

theorem Q_moves_clockwise_with_2omega (ω t : ℝ) :
  let (x, y) := pointP ω t
  let (qx, qy) := pointQ x y
  qx = Real.cos (2 * ω * t + π / 2) ∧ qy = Real.sin (2 * ω * t + π / 2) :=
by
  intro x y
  cases pointP ω t with xp yp
  have : (x, y) = (xp, yp) by sorry
  rw this
  cases pointQ x y with qx qy
  have : (qx, qy) = (-Real.sin (2 * ω * t), -Real.cos (2 * ω * t)) by sorry
  rw this
  split
  {
    have : Real.cos (2 * ω * t + π / 2) = -Real.sin (2 * ω * t) by sorry
    exact this
  }
  {
    have : Real.sin (2 * ω * t + π / 2) = -Real.cos (2 * ω * t) by sorry
    exact this
  }

end Q_moves_clockwise_with_2omega_l814_814690


namespace domain_of_inverse_l814_814425

def f(x : ℝ) : ℝ := 3^x

theorem domain_of_inverse :
  Set.Ioc 1 9 = (Set.range (fun x => f x)) ∩ Set.Ioc 0 2 := sorry

end domain_of_inverse_l814_814425


namespace logarithmic_equation_solution_l814_814110

theorem logarithmic_equation_solution (x : ℝ) (hx : 0 < x) :
  (log 2 x) * (log 3 x) = (log 3 (x^3)) + (log 2 (x^2)) - 6 ↔ x = 8 ∨ x = 9 := 
sorry

end logarithmic_equation_solution_l814_814110


namespace sum_of_roots_l814_814959

noncomputable def poly : Polynomial ℝ :=
  Polynomial.sum $ List.ofFn (λ k, Polynomial.monomial (k + 1) (1 : ℝ) * Polynomial.X ^ (2008 - k))

theorem sum_of_roots :
  let p := (0, Polynomial.sum $ List.mapWithIndex (λ k coeff, Polynomial.monomial coeff (Polynomial.X - (k + 1)) ^ (2008 - k)) $ List.range 2008)
  Vieta.sumRoots p.2 = 2006 :=
by sorry

end sum_of_roots_l814_814959


namespace perpendicular_vector_l814_814992

variables {a b : ℝ}

/-- Given that the angle between unit vectors $\overrightarrow{a}$ and $\overrightarrow{b}$ 
is $60^{\circ}$, prove that $2\overrightarrow{a} - \overrightarrow{b}$ is perpendicular to 
$\overrightarrow{b}$. -/
theorem perpendicular_vector 
  (a b : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hab : a * b = 0.5) :
  (2 * a - b) * b = 0 :=
begin
  sorry,
end

end perpendicular_vector_l814_814992


namespace g_increasing_on_interval_l814_814644

noncomputable def f (x : ℝ) : ℝ := sin (1/4 * x + π/6)
noncomputable def g (x : ℝ) : ℝ := sin (1/4 * (x - 8*π/3) + π/6)

theorem g_increasing_on_interval : ∀ x y, π ≤ x → x < y → y ≤ 2*π → g x < g y := by
  sorry

end g_increasing_on_interval_l814_814644


namespace pyramid_division_ratio_l814_814898

/-- Proof problem for pyramid division -/
theorem pyramid_division_ratio (V : ℝ) (A : ℝ) (x : ℝ) (V_orig : ℝ) (A_orig : ℝ)
  (h_V : V_orig = V + (48 - V))
  (h_A : A_orig = A + (60 - A))
  (h_V_ratio : V / (48 - V) = A / (60 - A)) :
  let k := A / (60 - A) in
  k = 91 / 149 ∧ 91 + 149 = 240 :=
by
  -- Proof omitted
  sorry

end pyramid_division_ratio_l814_814898


namespace sum_infinite_series_l814_814185

theorem sum_infinite_series : (∑' n : ℕ, (n + 1) / 8^(n + 1)) = 8 / 49 := sorry

end sum_infinite_series_l814_814185


namespace collin_total_petals_l814_814555

variable (collin_flowers initially given_flowers received_flowers each_flower_petals total_petals : ℕ)

-- Conditions as definitions in Lean
def collin_initial_flowers := 25
def ingrid_total_flowers := 33 / 3
def each_flower_petals := 4

-- Collin receives 11 flowers from Ingrid
def received_flowers := ingrid_total_flowers

-- Total flowers Collin has
def collin_flowers := 25 + received_flowers

-- Total petals Collin has
def total_petals := collin_flowers * each_flower_petals

-- Proof that Collin has 144 petals in total
theorem collin_total_petals : total_petals = 144 := by
  sorry

end collin_total_petals_l814_814555


namespace find_ABC_sum_l814_814061

theorem find_ABC_sum (A B C : ℤ) (h : ∀ x : ℤ, x = -3 ∨ x = 0 ∨ x = 4 → x^3 + A * x^2 + B * x + C = 0) : 
  A + B + C = -13 := 
by 
  sorry

end find_ABC_sum_l814_814061


namespace range_of_a_l814_814974

noncomputable def piecewise_f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 - 2 * a * x - 2 else x + (36 / x) - 6 * a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, piecewise_f a x ≥ piecewise_f a 2) ↔ (2 ≤ a ∧ a ≤ 5) :=
by
  sorry

end range_of_a_l814_814974


namespace greatest_third_term_arithmetic_seq_l814_814072

theorem greatest_third_term_arithmetic_seq (a d : ℤ) (h1: a > 0) (h2: d ≥ 0) (h3: 5 * a + 10 * d = 65) : 
  a + 2 * d = 13 := 
by 
  sorry

end greatest_third_term_arithmetic_seq_l814_814072


namespace weight_of_replaced_person_l814_814421

theorem weight_of_replaced_person 
  (avg_increase : ℝ)
  (num_persons : ℕ)
  (new_person_weight : ℝ)
  (weight_increase : ℝ)
  (new_person_might_be_90_kg : new_person_weight = 90)
  (average_increase_by_3_5_kg : avg_increase = 3.5)
  (group_of_8_persons : num_persons = 8)
  (total_weight_increase_formula : weight_increase = num_persons * avg_increase)
  (weight_of_replaced_person : ℝ)
  (weight_difference_formula : weight_of_replaced_person = new_person_weight - weight_increase) :
  weight_of_replaced_person = 62 :=
sorry

end weight_of_replaced_person_l814_814421


namespace perpendicular_lines_condition_l814_814298

theorem perpendicular_lines_condition (A1 B1 C1 A2 B2 C2 : ℝ) :
  (A1 * A2 + B1 * B2 = 0) ↔ (A1 * A2) / (B1 * B2) = -1 := sorry

end perpendicular_lines_condition_l814_814298


namespace value_of_a_l814_814488

theorem value_of_a (a : ℝ) (M N : set ℝ) :
  (M = {x | x = a}) →
  (N = {x | a * x = 1}) →
  (M ∩ N = N) →
  a = 0 ∨ a = 1 ∨ a = -1 :=
by
  intro hM hN hMN
  sorry

end value_of_a_l814_814488


namespace ones_digit_of_3_pow_53_l814_814838

-- Define the cycle of ones digits of powers of 3
def cycle_of_ones_digits : List ℕ := [3, 9, 7, 1]

-- Define a function to get the remainder
def remainder (a b : ℕ) : ℕ := a % b

-- State the theorem about the ones digit of 3^53
theorem ones_digit_of_3_pow_53 : remainder 53 4 = 1 → cycle_of_ones_digits.nth 0 = some 3 :=
by sorry

end ones_digit_of_3_pow_53_l814_814838


namespace Miguel_jump_is_longer_l814_814026

theorem Miguel_jump_is_longer
  (peter_strides_per_gap : ℕ)
  (miguel_jumps_per_gap : ℕ)
  (total_distance : ℕ)
  (poles_count : ℕ)
  (peter_total_strides : ℕ)
  (miguel_total_jumps : ℕ)
  (peter_stride_length : ℕ)
  (miguel_jump_length : ℕ)
  (peter_strides_per_gap = 66)
  (miguel_jumps_per_gap = 18)
  (total_distance = 3960)
  (poles_count = 31)
  (peter_total_strides = 1980)
  (miguel_total_jumps = 540)
  (peter_stride_length = total_distance / peter_total_strides)
  (miguel_jump_length = total_distance / miguel_total_jumps) :
  miguel_jump_length - peter_stride_length = 5.333 := by
  sorry

end Miguel_jump_is_longer_l814_814026


namespace johns_speed_l814_814729

theorem johns_speed :
  ∀ (v : ℝ), 
    (∀ (t : ℝ), 24 = 30 * (t + 4 / 60) → 24 = v * (t - 8 / 60)) → 
    v = 40 :=
by
  intros
  sorry

end johns_speed_l814_814729


namespace max_common_initial_segment_length_l814_814431

/-- 
Given the periods of two sequences, m and n, are coprime,
we aim to prove that their maximum common initial segment length is m + n - 2.
-/
theorem max_common_initial_segment_length (m n : ℕ) (hmn_coprime : Nat.coprime m n) :
  ∃ L, L = m + n - 2 := 
sorry

end max_common_initial_segment_length_l814_814431


namespace cream_cheese_cost_l814_814227

theorem cream_cheese_cost:
  ∃ (B C : ℝ), (2 * B + 3 * C = 12) ∧ (4 * B + 2 * C = 14) ∧ (C = 2.5) :=
by
  sorry

end cream_cheese_cost_l814_814227


namespace arithmetic_sequence_a7_l814_814719

variable {a : ℕ → ℚ}

def isArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (h_arith : isArithmeticSequence a) (h_a1 : a 1 = 2) (h_a3_a5 : a 3 + a 5 = 8) :
  a 7 = 6 :=
sorry

end arithmetic_sequence_a7_l814_814719


namespace find_third_number_l814_814680

-- Define the given conditions
def proportion_condition (x y : ℝ) : Prop :=
  (0.75 / x) = (y / 8)

-- The main statement to be proven
theorem find_third_number (x y : ℝ) (hx : x = 1.2) (h_proportion : proportion_condition x y) : y = 5 :=
by
  -- Using the assumptions and the definition provided.
  sorry

end find_third_number_l814_814680


namespace find_n_l814_814332

open Real

def f (x : ℝ) : ℝ := log x / log 4 + x - 7

theorem find_n (h₁ : ∀ x, f x = 0 → x ∈ Ioo (5 : ℝ) 6) : ∃ n : ℕ, 1 ≤ n ∧ x ∈ Ioo (n, n + 1) := by
  sorry

end find_n_l814_814332


namespace number_of_valid_grids_l814_814875

-- Define the problem context as a function in Lean
def valid_grid (grid : list (list ℕ)) : Prop :=
  grid.length = 3 ∧ (∀ row, row ∈ grid → row.length = 3) ∧ 
  (∀ row, row ∈ grid → 0 ∈ row ∧ 1 ∈ row) ∧ 
  (∀ i, i < 3 → 0 ∈ [grid[0][i], grid[1][i], grid[2][i]] ∧ 
         1 ∈ [grid[0][i], grid[1][i], grid[2][i]])

-- Theorem stating the number of valid configurations
theorem number_of_valid_grids :
  ∃ grids : finset (list (list ℕ)), 
    (∀ grid, grid ∈ grids → valid_grid grid) ∧ 
    finset.card grids = 102 :=
by
  sorry

end number_of_valid_grids_l814_814875


namespace order_of_numbers_l814_814064

noncomputable def log_3_4 := Real.logBase 3 4
noncomputable def log_1_1_0_9 := Real.logBase 1.1 0.9
def zero_dot_3_pow_4 := 0.3 ^ 4

theorem order_of_numbers :
  log_3_4 > zero_dot_3_pow_4 ∧ zero_dot_3_pow_4 > log_1_1_0_9 :=
by
  sorry

end order_of_numbers_l814_814064


namespace arithmetic_sequence_root_sum_l814_814712

theorem arithmetic_sequence_root_sum (a : ℕ → ℝ) (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) 
    (h_roots : (a 3) * (a 8) + 3 * (a 3) + 3 * (a 8) - 18 = 0) : a 5 + a 6 = 3 := by
  sorry

end arithmetic_sequence_root_sum_l814_814712


namespace min_expression_value_l814_814955

open Real

-- Define the conditions given in the problem: x, y, z are positive reals and their product is 32
variables {x y z : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 32)

-- Define the expression that we want to find the minimum for: x^2 + 4xy + 4y^2 + 2z^2
def expression (x y z : ℝ) : ℝ := x^2 + 4 * x * y + 4 * y^2 + 2 * z^2

-- State the theorem: proving that the minimum value of the expression given the conditions is 96
theorem min_expression_value : ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y * z = 32 ∧ expression x y z = 96 :=
sorry

end min_expression_value_l814_814955


namespace valid_necklace_length_l814_814163

-- Define the shapes and colors as types
inductive Shape | cube | pyramid
inductive Color | green | blue | red

-- Each bead is characterized by its shape and color
structure Bead :=
  (shape : Shape)
  (color : Color)

-- Definition to check adjacent beads for different shape and color
def valid_pair (b1 b2 : Bead) :=
  (b1.shape ≠ b2.shape) ∧ (b1.color ≠ b2.color)

-- Definition of a valid necklace
def valid_necklace (beads : List Bead) :=
  (List.length beads) % 2 = 0 ∧
  (∀ i : Fin (List.length beads), valid_pair (beads.get i) (beads.get ((i + 1) % (List.length beads))))

-- The main proposition
theorem valid_necklace_length (n : Nat) :
  n = 8 ∨ n = 10 ↔
  ∃ beads : List Bead,
    valid_necklace beads ∧
    List.length beads = n :=
by
  sorry

end valid_necklace_length_l814_814163


namespace family_cost_l814_814221

noncomputable def cost_per_family_member (n_persons_per_dozen : ℕ) (n_dozens : ℕ) (cans_per_person : ℕ) 
  (cost_per_box : ℝ) (cans_per_box : ℕ) (n_family_members : ℕ) : ℝ :=
  let total_persons := n_persons_per_dozen * n_dozens in
  let total_cans := total_persons * cans_per_person in
  let total_boxes := total_cans / cans_per_box in
  let total_cost := total_boxes * cost_per_box in
    total_cost / n_family_members

theorem family_cost (n_persons_per_dozen n_dozens cans_per_person cost_per_box cans_per_box n_family_members : ℕ)
  (h1 : n_persons_per_dozen = 12) 
  (h2 : n_dozens = 5) 
  (h3 : cans_per_person = 2) 
  (h4 : cost_per_box = 2) 
  (h5 : cans_per_box = 10) 
  (h6 : n_family_members = 6) :
  cost_per_family_member n_persons_per_dozen n_dozens cans_per_person cost_per_box cans_per_box n_family_members = 4 := 
  by
    sorry

end family_cost_l814_814221


namespace solve_equation_l814_814407

-- We need to define the conditions
theorem solve_equation (x : ℝ) (h : 81 = 6 * 16^(x - 2)) : 
  x = real.log 13.5 / real.log 16 + 2 :=
by
  -- Here we will usually provide the proof
  sorry

end solve_equation_l814_814407


namespace value_of_n_l814_814673

theorem value_of_n (n : ℕ) (h : sqrt (10 + n) = 9) : n = 71 :=
by
  sorry

end value_of_n_l814_814673


namespace increasing_on_interval_l814_814428

open Real

noncomputable def f (x : ℝ) : ℝ := exp (2 * x) + 2 * cos x - 4

theorem increasing_on_interval :
  ∀ x ∈ (Icc 0 (2 * π)), (2 * exp (2 * x) - 2 * sin x) ≥ 0 :=
by
  intros x hx
  sorry

end increasing_on_interval_l814_814428


namespace number_of_integers_l814_814922

theorem number_of_integers (n : ℕ) (h : n = 1199) : (#finset.filter (λ x, (¬ (x % 8 = 0) ∧ ¬ (x % 7 = 0))) (finset.range n)) = 900 :=
by
  sorry

end number_of_integers_l814_814922


namespace find_second_number_l814_814324

variable (n : ℕ)

theorem find_second_number (h : 8000 * n = 480 * 10^5) : n = 6000 :=
by
  sorry

end find_second_number_l814_814324


namespace num_ordered_triples_l814_814601

theorem num_ordered_triples :
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ∣ b ∧ a ∣ c ∧ a + b + c = 100) :=
  sorry

end num_ordered_triples_l814_814601


namespace pencil_count_l814_814856

variable (P L : ℕ)

theorem pencil_count 
  (h1 : Ratio P L = Rat.mk 5 6)
  (h2 : L = P + 8) :
  L = 48 := 
sorry

end pencil_count_l814_814856


namespace jeff_injured_cats_l814_814362

theorem jeff_injured_cats:
  ∀ (Initial_cats Monday_kittens Adopted_cats Final_cats Injured_cats : ℕ),
    Initial_cats = 20 →
    Monday_kittens = 2 →
    Adopted_cats = 6 →
    Final_cats = 17 →
    Initial_cats + Monday_kittens + Injured_cats - Adopted_cats = Final_cats →
    Injured_cats = 1 :=
by
  intros Initial_cats Monday_kittens Adopted_cats Final_cats Injured_cats hInit hMon hAdopt hFinal hEq
  rw [hInit, hMon, hAdopt, hFinal] at hEq
  linarith

end jeff_injured_cats_l814_814362


namespace james_paid_with_l814_814361

variable (candy_packs : ℕ) (cost_per_pack : ℕ) (change_received : ℕ)

theorem james_paid_with (h1 : candy_packs = 3) (h2 : cost_per_pack = 3) (h3 : change_received = 11) :
  let total_cost := candy_packs * cost_per_pack
  let amount_paid := total_cost + change_received
  amount_paid = 20 :=
by
  sorry

end james_paid_with_l814_814361


namespace abs_neg_three_l814_814872

theorem abs_neg_three : |(-3 : ℝ)| = 3 := 
by
  -- The proof would go here, but we skip it for this exercise.
  sorry

end abs_neg_three_l814_814872


namespace part1_daily_sales_profit_final_max_daily_sales_profit_l814_814416

-- Conditions from part (a)
def original_selling_price : ℚ := 30
def cost_price : ℚ := 15
def original_sales_volume : ℚ := 60
def sales_increase_per_yuan : ℚ := 10

-- Part (1): Daily sales profit if the price is reduced by 2 yuan
def new_selling_price1 : ℚ := original_selling_price - 2
def new_sales_volume1 : ℚ := original_sales_volume + (2 * sales_increase_per_yuan)
def profit_per_kilogram1 : ℚ := new_selling_price1 - cost_price
def daily_sales_profit1 : ℚ := profit_per_kilogram1 * new_sales_volume1

theorem part1_daily_sales_profit : daily_sales_profit1 = 1040 := by
  sorry

-- Part (2): Maximum daily sales profit and corresponding selling price
def selling_price_at_max_profit : ℚ := 51 / 2

def daily_profit (x : ℚ) : ℚ :=
  (x - cost_price) * (original_sales_volume + (original_selling_price - x) * sales_increase_per_yuan)

theorem final_max_daily_sales_profit :
  (∀ x : ℚ, daily_profit x ≤ daily_profit selling_price_at_max_profit) ∧ daily_profit selling_price_at_max_profit = 1102.5 := by
  sorry

end part1_daily_sales_profit_final_max_daily_sales_profit_l814_814416


namespace class_performance_comparison_l814_814411

noncomputable def class_a_avg := 106.8
noncomputable def class_b_avg := 107
noncomputable def class_a_var := 6
noncomputable def class_b_var := 14

theorem class_performance_comparison :
  (class_b_avg > class_a_avg) ∧ (class_a_var < class_b_var) ∧ (class_b_var > class_a_var) :=
by
  sorry

end class_performance_comparison_l814_814411


namespace arithmetic_sequence_problem_l814_814347

theorem arithmetic_sequence_problem 
    (a : ℕ → ℝ)  -- Define the arithmetic sequence as a function from natural numbers to reals
    (a1 : ℝ)  -- Represent a₁ as a1
    (a8 : ℝ)  -- Represent a₈ as a8
    (a9 : ℝ)  -- Represent a₉ as a9
    (a10 : ℝ)  -- Represent a₁₀ as a10
    (a15 : ℝ)  -- Represent a₁₅ as a15
    (h1 : a 1 = a1)  -- Hypothesis that a(1) is represented by a1
    (h8 : a 8 = a8)  -- Hypothesis that a(8) is represented by a8
    (h9 : a 9 = a9)  -- Hypothesis that a(9) is represented by a9
    (h10 : a 10 = a10)  -- Hypothesis that a(10) is represented by a10
    (h15 : a 15 = a15)  -- Hypothesis that a(15) is represented by a15
    (h_condition : a1 + 2 * a8 + a15 = 96)  -- Condition of the problem
    : 2 * a9 - a10 = 24 := 
sorry

end arithmetic_sequence_problem_l814_814347


namespace product_sum_of_roots_l814_814747

theorem product_sum_of_roots
  {p q r : ℝ}
  (h : (∀ x : ℝ, (4 * x^3 - 8 * x^2 + 16 * x - 12) = 0 → (x = p ∨ x = q ∨ x = r))) :
  p * q + q * r + r * p = 4 := 
sorry

end product_sum_of_roots_l814_814747


namespace quadratic_function_expression_point_not_on_graph_l814_814277

theorem quadratic_function_expression :
  (∃ a b c : ℝ, ∀ x : ℝ, (0, 3) ∧ (x, y) = (-2, -1) → y = (x + 2)^2 - 1) :=
by sorry

theorem point_not_on_graph :
  ¬ ((1, 7) ∈ { p : (ℝ × ℝ) | ∃ y : ℝ, y = (p.1 + 2)^2 - 1 }) :=
by sorry

end quadratic_function_expression_point_not_on_graph_l814_814277


namespace immovability_of_strings_l814_814180

-- Define the structure of the box and the points N and P
structure Box (α : Type) :=
  (width height depth : α)

variables {α : Type} [OrderedField α]

-- Define the gluing of strings that implies immovability
def strong_gluing (b : Box α) (N P : Box α → α × α × α) (right_angle : Prop) : Prop :=
  ∀(x y : α), N b = (x, y, b.height) ∧ P b = (x, y, 0) ∧ right_angle → false

-- Prove the immovability of the strings due to strong gluing
theorem immovability_of_strings (b : Box α) (N P : Box α → α × α × α) :
  strong_gluing b N P (right_angle := true) :=
sorry

end immovability_of_strings_l814_814180


namespace probability_units_digit_4_l814_814220

open Finset

theorem probability_units_digit_4 :
  let S := (range 100).map (λ n, n + 1)
  let count := (card (filter (λ (a : ℕ), (2^a % 10 + 5^5 % 10) % 10 = 4) S))
  --> The probability calculation
  (count : ℚ) / (100 * 100) = 1 / 4 :=
by
  let S := (range 100).map (λ n, n + 1)
  let count := (card (filter (λ (a : ℕ), (2^a % 10 + 5 % 10) % 10 = 4) S))
  have : count = 25,
  {
    -- Explanation here why the count is 25
    sorry
  }
  have total_outcomes : 100 * 100 = 10000,
  {
    sorry
  }
  have probability : (count : ℚ) / 10000 = 1 / 4,
  {
    rw [this, total_outcomes],
    norm_num
  }
  exact probability

end probability_units_digit_4_l814_814220


namespace smallest_x_y_sum_l814_814252

theorem smallest_x_y_sum :
  ∃ x y : ℕ,
    0 < x ∧ 0 < y ∧ x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 15) ∧ (x + y = 64) := 
by
  sorry

end smallest_x_y_sum_l814_814252


namespace increase_in_average_age_l814_814048

-- Define the ages of the two replaced men
def age1 : Nat := 21
def age2 : Nat := 23

-- Define the average age of the two new men
def average_age_new_men : ℕ := 37

-- Define the total number of men
def num_men : ℕ := 15

-- Theorem to prove
theorem increase_in_average_age : 
  let total_age_new_men := 2 * average_age_new_men in
  let total_age_old_men := age1 + age2 in
  let increase_in_total_age := total_age_new_men - total_age_old_men in
  let increase_in_average := increase_in_total_age / num_men in
  increase_in_average = 2 := sorry

end increase_in_average_age_l814_814048


namespace sum_of_arithmetic_sequences_l814_814748

theorem sum_of_arithmetic_sequences (n : ℕ) (h : n ≠ 0) :
  (2 * n * (n + 3) = n * (n + 12)) → (n = 6) :=
by
  intro h_eq
  have h_nonzero : n ≠ 0 := h
  sorry

end sum_of_arithmetic_sequences_l814_814748


namespace fraction_is_terminating_decimal_l814_814575

noncomputable def fraction_to_decimal : ℚ :=
  58 / 160

theorem fraction_is_terminating_decimal : fraction_to_decimal = 3625 / 10000 :=
by
  sorry

end fraction_is_terminating_decimal_l814_814575


namespace equal_tuesdays_thursdays_l814_814141

-- A definition to represent the seven days of the week
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday
deriving DecidableEq, Inhabited

-- A Proof Problem Statement
theorem equal_tuesdays_thursdays (days_in_month : ℕ) (first_day : DayOfWeek) (num_weeks : ℕ) :
  days_in_month = 31 ∧ num_weeks = 4 + 1  -- 4 full weeks + 3 extra days
  → (∃ days: fin 7 → fin 4, first_day = days 0)   -- Only three possible starting days lead to same number of Tuesdays and Thursdays
    → ∃ (days_count : ℕ), days_count = 3 :=
by
  sorry

end equal_tuesdays_thursdays_l814_814141


namespace abundant_numbers_less_than_50_eq_9_l814_814306

def proper_factors (n : ℕ) : List ℕ :=
  List.filter (λ m, m < n ∧ n % m = 0) (List.range n)

def sum_proper_factors (n : ℕ) : ℕ :=
  (proper_factors n).sum

def is_abundant (n : ℕ) : Prop :=
  sum_proper_factors n > n

def abundant_numbers_count_below (m : ℕ) : ℕ :=
  (List.range m).filter is_abundant |>.length

theorem abundant_numbers_less_than_50_eq_9 :
  abundant_numbers_count_below 50 = 9 :=
by
  sorry

end abundant_numbers_less_than_50_eq_9_l814_814306


namespace lines_connecting_intersections_l814_814667

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem lines_connecting_intersections (n : ℕ) (h : n ≥ 2) :
  let N := binomial n 2
  binomial N 2 = (n * n * (n - 1) * (n - 1) - 2 * n * (n - 1)) / 8 :=
by {
  sorry
}

end lines_connecting_intersections_l814_814667


namespace harry_terry_difference_l814_814661

theorem harry_terry_difference :
  let H := 12 - (3 * 4)
  let T := 12 - (3 * 4) -- Correcting Terry's mistake
  H - T = 0 := by
  sorry

end harry_terry_difference_l814_814661


namespace tangent_line_at_1_l814_814075

noncomputable theory

open Real

def f (x : ℝ) : ℝ := exp x * log x

def derivative_f (x : ℝ) : ℝ := exp x * log x + exp x / x

theorem tangent_line_at_1 :
  let x₀ := 1,
      y₀ := f x₀,
      m := derivative_f x₀
  in y₀ = 0 ∧ m = exp 1 ∧ (∀ x, (f x₀ = y₀ ∧ m * (x - x₀) = exp 1 * (x - 1)) → (y = exp 1 * (x - 1))) :=
begin
  -- Proof steps would go here, currently skipped
  sorry
end

end tangent_line_at_1_l814_814075


namespace sum_of_distances_eq_l814_814752

theorem sum_of_distances_eq (n : ℕ) (h1 : n % 2 = 1) (A : Fin n → ℝ × ℝ) (M : ℝ × ℝ)
  (circumcircle : set (ℝ × ℝ)) (hA : ∀ i j, dist (A i) (A (i + 1 % n)) = dist (A j) (A ((j + 1) % n)))
  (hM : M ∈ circumcircle) :
  let dist_to_M (i : Fin n) := dist M (A i) in
  ∑ i in Finset.filter (λ k, k.1 % 2 = 1) Finset.univ dist_to_M =
  ∑ i in Finset.filter (λ k, k.1 % 2 = 0) Finset.univ dist_to_M := sorry

end sum_of_distances_eq_l814_814752


namespace expected_value_two_X_plus_three_l814_814792

def X : Type := ℕ

def P : X → ℝ
| 1 := 0.2
| 2 := 0.4
| 3 := 0.4
| _ := 0

noncomputable def expected_value (f : X → ℝ) : ℝ :=
Σ (x : X), f x * P x

theorem expected_value_two_X_plus_three : expected_value (λ x, 2 * x + 3) = 7.4 := by
  sorry

end expected_value_two_X_plus_three_l814_814792


namespace sum_of_roots_l814_814961

noncomputable def poly := (x - 1) ^ 2008 + 2 * (x - 2) ^ 2007 + 3 * (x - 3) ^ 2006 +
  (Finset.range 2007).sum (λ k, (↑k + 1) * (x - (↑k + 1)) ^ (2008 - (k + 1)))

theorem sum_of_roots : (sum_of_roots poly) = 2006 :=
by 
  sorry

end sum_of_roots_l814_814961


namespace permutations_multiple_of_2010_l814_814896

def is_perfect_square (x : ℕ) : Prop := ∃ m : ℕ, m * m = x

def valid_permutation (n : ℕ) (a : Fin n → Fin n) : Prop :=
  ∀ k : Fin n, is_perfect_square (k * (a k))

def count_valid_permutations (n : ℕ) : ℕ :=
  -- Assume there is a function P(n) which counts the number of valid permutations
  Finset.card {a : Fin n → Fin n | valid_permutation n a}

theorem permutations_multiple_of_2010 (n : ℕ) : 
  (count_valid_permutations n % 2010 = 0) ↔ (n ≥ 4489) :=
sorry

end permutations_multiple_of_2010_l814_814896


namespace sum_of_adjacent_to_14_l814_814803

/-!
# Problem Statement
The positive integer divisors of 294, except 1, are arranged around a circle so that every pair of adjacent integers has a common factor greater than 1. Prove that the sum of the two integers adjacent to 14 is 140.
-/

noncomputable def divisors_except_one (n : ℕ) : List ℕ :=
  (List.range (n+1)).filter (λ x, x > 1 ∧ n % x = 0)

noncomputable def arranged_circle (n : ℕ) : List ℕ := 
  [2, 3, 7, 6, 14, 21, 42, 49, 98, 147, 294] -- manually derived

theorem sum_of_adjacent_to_14 : 
  let divisors := divisors_except_one 294 in
  let adjacent_to_14 := List.filter (λ x, Nat.gcd 14 x > 1) (arranged_circle 294) in
  adjacent_to_14 = [42, 98] →
  adjacent_to_14.sum = 140 :=
by
  intro divisors adjacent_to_14 H
  have h1 : adjacent_to_14 = [42, 98] := H
  rw h1
  norm_num
  rfl

example : sum_of_adjacent_to_14 :=
by
  unfold sum_of_adjacent_to_14
  rw List.filter_eq_of_sublist
  sorry

end sum_of_adjacent_to_14_l814_814803


namespace value_of_a_l814_814685

theorem value_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 2 * x + 1) 
  (h_max : ∀ x ∈ set.Icc a (a + 2), f x ≤ 4) (h_maxValue : f (a + 2) = 4 ∨ f a = 4) :
  a = 1 ∨ a = -1 := 
by {
  sorry
}

end value_of_a_l814_814685


namespace bridge_length_is_1500_l814_814522

noncomputable def length_of_bridge (speed_kmph : ℝ) (time_minutes : ℕ) : ℝ :=
  let speed_kmpmin := speed_kmph / 60 -- converting speed from km/hr to km/min
  speed_kmpmin * time_minutes * 1000 -- calculating distance in meters

theorem bridge_length_is_1500 (speed : ℝ) (time : ℕ) (h1 : speed = 6) (h2 : time = 15) :
  length_of_bridge speed time = 1500 :=
by {
  rw [h1, h2],
  -- calculate the length
  have speed_kmpmin := (6 : ℝ) / 60,
  have length_km := speed_kmpmin * (15 : ℝ),
  have length_m := length_km * 1000,
  rw [show speed_kmpmin = 0.1, by norm_num],
  rw [show length_km = 1.5, by norm_num],
  rw [show length_m = 1500, by norm_num],
  exact rfl,
  sorry -- skip proof steps
}

end bridge_length_is_1500_l814_814522


namespace determine_a_l814_814641

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * (x + a) / x

-- Define the property of the function being odd
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

-- The main theorem statement
theorem determine_a (a : ℝ) : 
  (is_odd_function (λ x, f x a)) → a = -1 :=
by 
  sorry

end determine_a_l814_814641


namespace B_finishes_in_45_days_l814_814134

-- Definitions of the given conditions
def work_rate_A : ℚ := 1 / 40
def work_rate_B : ℚ := 1 / 60
def days_A_worked : ℚ := 10
def total_work : ℚ := 1 -- Assuming the total work is 1 unit

-- A's contribution in 10 days
def A_contribution := work_rate_A * days_A_worked
def remaining_work := total_work - A_contribution

-- The statement to prove
theorem B_finishes_in_45_days : 
  remaining_work * (60 : ℚ) = 3 / 4 :=
by
  have h1 : remaining_work = 3 / 4 := by sorry
  show work_rate_B * (45 : ℚ) = remaining_work
  sorry -- proof omitted

end B_finishes_in_45_days_l814_814134


namespace triangle_inequality_half_perimeter_l814_814397

theorem triangle_inequality_half_perimeter 
  (a b c : ℝ)
  (h_a : a < b + c)
  (h_b : b < a + c)
  (h_c : c < a + b) :
  a < (a + b + c) / 2 ∧ b < (a + b + c) / 2 ∧ c < (a + b + c) / 2 := 
sorry

end triangle_inequality_half_perimeter_l814_814397


namespace area_of_rectangle_l814_814931

variables (m k a b c d : ℝ)
variables (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
variables (hm : m > 0) (hk : k > 0)
variables (hkm : km ≠ 1)

theorem area_of_rectangle :
  let xm_pos := (ka + c) / (1 - km) in
  let xm_neg := (ka - c) / (1 - km) in
  let ym_pos := m * xm_pos + a in
  let ym_neg := m * xm_neg - a in
  let delta_y := (ym_pos - ym_neg) in
  let delta_y := (2 * mc) / (1 - km) in
  let delta_x := (2 * ka) / (1 - km) in
  let area := delta_x * delta_y in
  area = (4 * m * c * k * a) / (1 - k * m)^2 :=
sorry

end area_of_rectangle_l814_814931


namespace hypotenuse_squared_l814_814557

noncomputable def polynomial_zeros_hypotenuse_squared (a b c : ℂ) (s t : ℂ) (h₁ : a^3 + s * a + t = 0) (h₂ : b^3 + s * b + t = 0) (h₃ : c^3 + s * c + t = 0) (h₄ : a + b + c = 0) (h₅ : abs a^2 + abs b^2 + abs c^2 = 350) : Prop :=
|a - c|^2 + |b - c|^2 = 612.5

theorem hypotenuse_squared (a b c : ℂ) (s t : ℂ)
  (h₁ : a^3 + s * a + t = 0)
  (h₂ : b^3 + s * b + t = 0)
  (h₃ : c^3 + s * c + t = 0)
  (h₄ : a + b + c = 0)
  (h₅ : abs a^2 + abs b^2 + abs c^2 = 350)
  (h₆ : ∃ k : ℝ, k² + (1 - k)² = 1)
  : abs c ^ 2 = 612.5 :=
sorry

end hypotenuse_squared_l814_814557


namespace perfect_square_adjacent_smaller_l814_814323

noncomputable def is_perfect_square (n : ℕ) : Prop := 
    ∃ k : ℕ, k * k = n

theorem perfect_square_adjacent_smaller (m : ℕ) (hm : is_perfect_square m) : 
    ∃ k : ℕ, (k * k = m ∧ (k - 1) * (k - 1) = m - 2 * k + 1) := 
by 
  sorry

end perfect_square_adjacent_smaller_l814_814323


namespace probability_three_white_balls_l814_814505

theorem probability_three_white_balls (total_balls: ℕ) (white_balls: ℕ) (black_balls: ℕ) (drawn_balls: ℕ) 
    (h_total: total_balls = 15) (h_white: white_balls = 7) (h_black: black_balls = 8) (h_drawn: drawn_balls = 3) : 
    ((choose white_balls drawn_balls) / (choose total_balls drawn_balls) : ℚ) = 1 / 13 := 
by {
    -- Definitions and conditions come from part (a)
    -- The lean code should be able to be built successfully
    sorry
} 

end probability_three_white_balls_l814_814505


namespace average_weight_A_l814_814449

-- Definitions of conditions
def number_of_students_A : ℕ := 26
def number_of_students_B : ℕ := 34
def average_weight_B : ℝ := 30.0
def total_students : ℕ := 60
def average_weight_class : ℝ := 38.67

-- Statement of the goal
theorem average_weight_A :
  let W_A := (60 * 38.67 - 34 * 30) / 26 in
  W_A = 50.01 :=
by
  sorry

end average_weight_A_l814_814449


namespace triangle_area_tangent_line_l814_814420

noncomputable def f : ℝ → ℝ := λ x, x * Real.log x
def P : ℝ × ℝ := (1, 0)
def tangent_line_at_P : ℝ → ℝ := λ x, x - 1

theorem triangle_area_tangent_line :
  let area := 1/2 * 1 * 1 in
  area = 1/2 :=
by
  sorry

end triangle_area_tangent_line_l814_814420


namespace sum_of_sequence_b_l814_814151

theorem sum_of_sequence_b {n : ℕ} (h1 : ∀ n ≥ 1, a n = 2 * n + 1) :
    let b (n : ℕ) := 1 / ((2 * n + 1) * (2 * n + 3)) in
    let T (n : ℕ) := ∑ k in Finset.range n, b k in
    T n = n / (6 * n + 9) := 
begin
  sorry
end

end sum_of_sequence_b_l814_814151


namespace ways_to_choose_3_qualified_prob_exactly_1_defective_prob_at_least_2_non_defective_l814_814450

-- Define the given conditions
def total_products : ℕ := 20
def qualified_products : ℕ := 17
def defective_products : ℕ := 3

-- Define the required statements to prove
theorem ways_to_choose_3_qualified : nat.choose qualified_products 3 = 680 :=
by sorry

theorem prob_exactly_1_defective : 
  (nat.choose qualified_products 2 * nat.choose defective_products 1) / nat.choose total_products 3 = 34 / 95 :=
by sorry

theorem prob_at_least_2_non_defective : 
  (nat.choose qualified_products 3 / nat.choose total_products 3) + 
  (nat.choose qualified_products 2 * nat.choose defective_products 1 / nat.choose total_products 3) = 272 / 285 :=
by sorry

end ways_to_choose_3_qualified_prob_exactly_1_defective_prob_at_least_2_non_defective_l814_814450


namespace mark_sprint_time_l814_814015

theorem mark_sprint_time : ∀ (distance speed : ℕ), distance = 24 → speed = 6 → distance / speed = 4 :=
by
  intros distance speed h_distance h_speed
  rw [h_distance, h_speed]
  exact Nat.div_eq_of_eq_mul_right (by decide) rfl

-- Alternatively using def to explicitly state the variables:
def mark_sprint_time (distance speed : ℕ) (h_distance : distance = 24) (h_speed : speed = 6) : distance / speed = 4 :=
by
  rw [h_distance, h_speed]
  exact Nat.div_eq_of_eq_mul_right (by decide) rfl

end mark_sprint_time_l814_814015


namespace measure_angle_BAC_l814_814724

theorem measure_angle_BAC (A B C D E : Type*) [EuclideanGeometry] 
  (triangleABC : EuclideanTriangle A B C)
  (angleBAC_lt_90 : triangleABC.angle_ABC < 90)
  (D_perpendicular : perpendicular C B D)
  (E_perpendicular : perpendicular B C E)
  (DE_eq_BC : dist D E = dist B C) : 
  triangleABC.angle_ABC = 60 :=
by
  sorry

end measure_angle_BAC_l814_814724


namespace part1_part2_l814_814971

noncomputable section

-- Definitions for conditions
def g (m : ℕ) (x : ℝ) : ℝ := 3^m * Real.exp x

def f1 (x : ℝ) : ℝ := 3 * x + 2

def f : ℕ → ℝ → ℝ
| 0, x   => x
| (n+1), x => f1 (f n x)

-- Part 1: Prove the summation
theorem part1 (x : ℝ) (hx : x ≥ 2) :
  (∑ i in Finset.range 2014, 1 / (g (i + 1) x - f (i + 1) x)) < 1 / 2 := 
sorry

-- Part 2: Existence and uniqueness of δ_k
theorem part2 (k : ℕ) (hk : k ≥ 7) :
  ∃! δ ∈ Finset.range 2014, 2014 ∣ f k δ := 
sorry

end part1_part2_l814_814971


namespace minimum_obtuse_edges_at_least_3_l814_814980

noncomputable def minimum_obtuse_edges (A B C D P : Point) (h : is_internal P A B C D) : ℕ :=
  if is_tetrahedron A B C D then
    let obtuse_edges := count_obtuse_edges A B C D P in
    if obtuse_edges ≤ 3 then obtuse_edges else sorry
  else sorry

theorem minimum_obtuse_edges_at_least_3 (A B C D P : Point) (h : is_internal P A B C D) :
  minimum_obtuse_edges A B C D P = 3 := sorry

end minimum_obtuse_edges_at_least_3_l814_814980


namespace oil_output_per_capita_correctness_l814_814859

variable (population_west : ℝ := 1)
variable (output_west : ℝ := 55.084)
variable (population_non_west : ℝ := 6.9)
variable (output_non_west : ℝ := 1480.689)
variable (output_russia_9_percent : ℝ := 13737.1)
variable (percentage : ℝ := 9)
variable (total_population_russia : ℝ := 147)

def west_output_per_capita : ℝ :=
  output_west / population_west

def non_west_output_per_capita : ℝ :=
  output_non_west / population_non_west

def total_output_russia : ℝ :=
  (output_russia_9_percent * 100) / percentage

def russia_output_per_capita : ℝ :=
  total_output_russia / total_population_russia

theorem oil_output_per_capita_correctness :
  west_output_per_capita = 55.084 ∧
  non_west_output_per_capita = 214.59 ∧
  total_output_russia = 152634.44 ∧
  russia_output_per_capita = 1038.33 :=
by
  sorry

end oil_output_per_capita_correctness_l814_814859


namespace valid_bead_count_l814_814159

structure Bead :=
  (shape : String)   -- "cube" or "pyramid"
  (color : String)   -- colors: "green", "blue", "red"

inductive Shape 
| Cube
| Pyramid 

inductive Color 
| Green
| Blue
| Red

def is_solution (n : ℕ) : Prop :=
  ∃ (b : fin n → Bead), 
    (∀ i, b (i + 1) % n).shape ≠ (b i).shape ∧ (b (i + 1) % n).color ≠ (b i).color) ∧
    set.of_list (list.of_fn b).to_finset = { Bead.mk Shape.Cube Color.Green, Bead.mk Shape.Cube Color.Blue, Bead.mk Shape.Pyramid Color.Red, Bead.mk Shape.Pyramid Color.Blue }

theorem valid_bead_count :
  ∀ n : ℕ, is_solution n → (n = 8 ∨ n = 10) :=
begin
  sorry
end

end valid_bead_count_l814_814159


namespace book_total_pages_l814_814850

theorem book_total_pages (x : ℕ) (h1 : x * (3 / 5) * (3 / 8) = 36) : x = 120 := 
by
  -- Proof should be supplied here, but we only need the statement
  sorry

end book_total_pages_l814_814850


namespace part1_daily_sales_profit_final_max_daily_sales_profit_l814_814417

-- Conditions from part (a)
def original_selling_price : ℚ := 30
def cost_price : ℚ := 15
def original_sales_volume : ℚ := 60
def sales_increase_per_yuan : ℚ := 10

-- Part (1): Daily sales profit if the price is reduced by 2 yuan
def new_selling_price1 : ℚ := original_selling_price - 2
def new_sales_volume1 : ℚ := original_sales_volume + (2 * sales_increase_per_yuan)
def profit_per_kilogram1 : ℚ := new_selling_price1 - cost_price
def daily_sales_profit1 : ℚ := profit_per_kilogram1 * new_sales_volume1

theorem part1_daily_sales_profit : daily_sales_profit1 = 1040 := by
  sorry

-- Part (2): Maximum daily sales profit and corresponding selling price
def selling_price_at_max_profit : ℚ := 51 / 2

def daily_profit (x : ℚ) : ℚ :=
  (x - cost_price) * (original_sales_volume + (original_selling_price - x) * sales_increase_per_yuan)

theorem final_max_daily_sales_profit :
  (∀ x : ℚ, daily_profit x ≤ daily_profit selling_price_at_max_profit) ∧ daily_profit selling_price_at_max_profit = 1102.5 := by
  sorry

end part1_daily_sales_profit_final_max_daily_sales_profit_l814_814417


namespace right_triangle_hypotenuse_l814_814464

theorem right_triangle_hypotenuse (a b : ℕ) (h₁ : a = 60) (h₂ : b = 100) : (∃ d, d = 20 * Real.sqrt 34) :=
by
  have A1: a ^ 2 = 3600 := by rw [h₁, Nat.pow]
  have A2: b ^ 2 = 10000 := by rw [h₂, Nat.pow]
  have S1: a ^ 2 + b ^ 2 = 13600 := by rw [A1, A2, Nat.add]
  have E1: Real.sqrt (a ^ 2 + b ^ 2) = Real.sqrt 13600 := by rw S1
  have E2: Real.sqrt 13600 = 20 * Real.sqrt 34 := by sorry
  exact ⟨20 * Real.sqrt 34, E2⟩

end right_triangle_hypotenuse_l814_814464


namespace jack_salt_evaporation_l814_814356

/-- Calculate the volume of salt in milliliters from a given volume of seawater and salt concentration. --/
def volume_of_salt_in_ml (seawater_volume : ℝ) (salt_percentage : ℝ) : ℝ :=
  seawater_volume * salt_percentage * 1000

theorem jack_salt_evaporation :
  volume_of_salt_in_ml 2 0.20 = 400 :=
by 
sory

end jack_salt_evaporation_l814_814356


namespace problem_part1_problem_part2_l814_814717

noncomputable def probability_of_B (p : ℝ) (A0 A1 A : events) : ℝ :=
  let PA := p
  let PA0 := p * p
  let PA1 := 2 * p * (1 - p)
  PA0 * PA + PA1 * (1 - PA)

noncomputable def expectation_of_xi (p : ℝ) : ℝ :=
  let PA0 := p * p
  let PA1 := 2 * p * (1 - p)
  let PA2 := (1 - p) * (1 - p)
  let PB := PA0 * p + PA1 * (1 - p)
  let Pxi_0 := PA2 * p
  let Pxi_2 := PB
  let Pxi_3 := PA0 * (1 - p)
  let Pxi_1 := 1 - (Pxi_0 + Pxi_2 + Pxi_3)
  1 * Pxi_1 + 2 * Pxi_2 + 3 * Pxi_3

theorem problem_part1 : 
  let p := 0.6
  let A0 := ∅ -- Dummy event A0
  let A1 := ∅ -- Dummy event A1
  let A := ∅ -- Dummy event A
  probability_of_B p A0 A1 A = 0.352 :=
by {
  sorry
}

theorem problem_part2 : 
  let p := 0.6
  expectation_of_xi p = 1.4 :=
by {
  sorry
}

end problem_part1_problem_part2_l814_814717


namespace parabola_equation_line_AB_fixed_point_l814_814009

section ParabolaProblem

variable {p : ℝ} (hp : p > 0) (sin_theta : Real.sin θ = sqrt(3) / 3)
variable (hAB_len : |A0B0_len| = 6) 

-- First part: proving the equation of the parabola
theorem parabola_equation : 
  (∃ p : ℝ, p > 0 → (2*p == 4) → 
  (∀ x y, x^2 = 2*p*y )) : Prop :=
sorry

-- Second part: proving line AB passes through a fixed point
theorem line_AB_fixed_point : 
  (∀ (x0 y0 : ℝ), (y0 = x0 - 4) → 
  (∃ A B : ℝ × ℝ, line_1_tangent_condition ∧ line_2_tangent_condition → 
  (∀ (x y : ℝ), x0 * x - 2 * (y0 + y) = 0) → 
  (2, 4))) :=
sorry

end ParabolaProblem

end parabola_equation_line_AB_fixed_point_l814_814009


namespace max_ab_min_3x_4y_max_f_l814_814126

-- Proof Problem 1
theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 4 * a + b = 1) : ab <= 1/16 :=
  sorry

-- Proof Problem 2
theorem min_3x_4y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) : 3 * x + 4 * y >= 5 :=
  sorry

-- Proof Problem 3
theorem max_f (x : ℝ) (h1 : x < 5/4) : 4 * x - 2 + 1 / (4 * x - 5) <= 1 :=
  sorry

end max_ab_min_3x_4y_max_f_l814_814126


namespace jumping_contest_total_jumps_l814_814662

theorem jumping_contest_total_jumps : 
  let H_1 := 180
  let L_1 := 180 * (3 / 4) * 1.2
  let H_2 := 180 * (2 / 3) + 15
  let L_2 := (180 * (2 / 3) + 15) + 50
  let H_3 := (180 * (2 / 3) + 15) * (4 / 3) * 0.9
  let L_3 := (180 * (3 / 4) * 1.2) * (4 / 5) * 1.1
  in (H_1 + H_2 + H_3) + (L_1 + L_2 + L_3) = 967 :=
by {
  sorry
}

end jumping_contest_total_jumps_l814_814662


namespace average_monthly_growth_rate_equation_l814_814135

variable (x : ℝ)

-- Conditions and definitions
def P_April : ℝ := 50
def P_total : ℝ := 182
def P_May : ℝ := 50 * (1 + x)
def P_June : ℝ := 50 * (1 + x)^2

-- The proposition to prove
theorem average_monthly_growth_rate_equation :
  P_April + P_May + P_June = P_total := sorry

end average_monthly_growth_rate_equation_l814_814135


namespace avg_adults_proof_l814_814789

variable (n_total : ℕ) (n_girls : ℕ) (n_boys : ℕ) (n_adults : ℕ)
variable (avg_total : ℕ) (avg_girls : ℕ) (avg_boys : ℕ)

def avg_age_adults (n_total n_girls n_boys n_adults avg_total avg_girls avg_boys : ℕ) : ℕ :=
  let sum_total := n_total * avg_total
  let sum_girls := n_girls * avg_girls
  let sum_boys := n_boys * avg_boys
  let sum_adults := sum_total - sum_girls - sum_boys
  sum_adults / n_adults

theorem avg_adults_proof :
  avg_age_adults 50 25 20 5 21 18 20 = 40 := 
by
  -- Proof will go here
  sorry

end avg_adults_proof_l814_814789


namespace twentieth_digit_of_half_power_twenty_l814_814096

/-- Define the function to get the 20th digit after the decimal point in the decimal expansion of a number. -/
noncomputable def twentieth_decimal_digit (x : ℝ) : ℕ :=
  let frac_part : ℝ := x - x.floor
  (frac_part * 10^20).floor.to_nat % 10

theorem twentieth_digit_of_half_power_twenty :
  twentieth_decimal_digit ((1/2 : ℝ)^20) = 1 :=
sorry

end twentieth_digit_of_half_power_twenty_l814_814096


namespace least_x_l814_814121

noncomputable def satisfies_conditions (x : ℕ) : Prop :=
  let y := (4 * x - 12) / 3 in
  x ∈ finset.range (16) ∧ (x ≥ 15) ∧ (4 * x - 12) % 3 = 0 ∧ (6 - x) * 21 < (7 - x) * 16 ∧ (4 * x - 12) % 3 = 0

theorem least_x :
  ∃ x ∈ finset.range (16), (x ≥ 15) ∧ satisfies_conditions x ∧ ∀ z ∈ finset.range (16), (z ≥ 15) → satisfies_conditions z → x ≤ z :=
begin
  sorry
end

end least_x_l814_814121


namespace solve_for_m_l814_814242

theorem solve_for_m 
  (m : ℝ) 
  (h : (m - 1) * 2^2 + 3 * 2 - 5 * m + 4 = 0) : m = 6 :=
sorry

end solve_for_m_l814_814242


namespace math_problem_l814_814236

def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 2)
def g (x : ℝ) : ℝ := Real.cos (x - Real.pi / 2)

theorem math_problem :
  (∀ x, f(x) * g(x) = (1 / 2) * Real.sin (2 * x)) ∧
  (∀ x, (1 / 2) * Real.sin (2 * (x + Real.pi)) = (1 / 2) * Real.sin (2 * x)) ∧
  Real.sin(2 * 0) / 2 ≤ 1 / 2 ∧
  (∀ x, f (x + Real.pi / 2) = g x) :=
by { sorry }

end math_problem_l814_814236


namespace count_of_abundant_numbers_less_than_50_l814_814309

-- Define the proper divisors of a number
def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d => d < n ∧ n % d = 0)

-- Define what it means for a number to be abundant
def is_abundant (n : ℕ) : Prop :=
  (proper_divisors n).sum > n

-- Define the set of numbers less than 50 which are abundant
def abundant_numbers_less_than_50 : List ℕ :=
  (List.range 50).filter is_abundant

-- The theorem stating the number of abundant numbers less than 50
theorem count_of_abundant_numbers_less_than_50 : abundant_numbers_less_than_50.length = 9 := 
  sorry

end count_of_abundant_numbers_less_than_50_l814_814309


namespace find_coordinates_of_M_l814_814247

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*sqrt(3)*y + 3 = 0

def point_on_circle (M : ℝ × ℝ) : Prop :=
  circle_eq M.1 M.2

def symmetric_about_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

def orthogonal_vectors (A M B : ℝ × ℝ) : Prop :=
  let AM := (M.1 - A.1, M.2 - A.2)
  let BM := (M.1 - B.1, M.2 - B.2)
  AM.1 * BM.1 + AM.2 * BM.2 = 0

theorem find_coordinates_of_M (A M B : ℝ × ℝ) (m : ℝ) (h₀ : m > 0) (h₁ : A = (0, m))
  (h₂ : symmetric_about_x_axis A B) (h₃ : point_on_circle M) (h₄ : orthogonal_vectors A M B)
  (h₅ : ∀ (x y : ℝ), circle_eq x y ↔ (x - 1)^2 + (y - sqrt(3))^2 = 1)
  (h₆ : m = 3) :
  M = (3 / 2, 3 * sqrt(3) / 2) :=
sorry

end find_coordinates_of_M_l814_814247


namespace yulia_expenses_l814_814109
-- Import the entire Mathlib library for comprehensive support

-- Define the problem with the given conditions and then assert the conclusion
theorem yulia_expenses (
  (net_profit : ℕ) (net_profit_eq : net_profit = 44) 
  (gross_revenue : ℕ) (gross_revenue_eq : gross_revenue = 47) 
  (babysitting_earnings : ℕ) (babysitting_earnings_eq : babysitting_earnings = 31) 
) : 
  (gross_revenue + babysitting_earnings - net_profit) = 34 := 
by
  -- Skipping the proof with sorry
  sorry

end yulia_expenses_l814_814109


namespace transformed_variance_l814_814282

variables (x : ℕ → ℝ) (n : ℕ)

def variance (data : ℕ → ℝ) (N : ℕ) : ℝ :=
  let mean := (finset.range N).sum (λ i, data i) / N in
  (finset.range N).sum (λ i, (data i - mean) ^ 2) / N

theorem transformed_variance (h : variance x 10 = 3) :
  variance (λ i, 2 * x i + 3) 10 = 12 :=
begin
  sorry
end

end transformed_variance_l814_814282


namespace last_five_digits_of_sequence_l814_814069

theorem last_five_digits_of_sequence (seq : Fin 36 → Fin 2) 
  (h0 : seq 0 = 0) (h1 : seq 1 = 0) (h2 : seq 2 = 0) (h3 : seq 3 = 0) (h4 : seq 4 = 0)
  (unique_combos : ∀ (combo: Fin 32 → Fin 2), 
    ∃ (start_index : Fin 32), ∀ (i : Fin 5),
      combo i = seq ((start_index + i) % 36)) :
  seq 31 = 1 ∧ seq 32 = 1 ∧ seq 33 = 1 ∧ seq 34 = 0 ∧ seq 35 = 1 :=
by
  sorry

end last_five_digits_of_sequence_l814_814069


namespace fraction_to_decimal_l814_814572

-- We define the fraction and its simplified form
def fraction : ℚ := 58 / 160
def simplified_fraction : ℚ := 29 / 80

-- We state that the fraction simplifies correctly
lemma simplify_fraction : fraction = simplified_fraction := by
  sorry

-- Define the factorization of the denominator
def denominator_factorization : ℕ := 2^4 * 5

-- Verify the fraction when multiplied by 125/125
def equalized_fraction : ℚ := 29 * 125 / 10000

-- State the final result as a decimal
theorem fraction_to_decimal : fraction = 0.3625 := by
  sorry

end fraction_to_decimal_l814_814572


namespace increasing_interval_of_f_decreasing_interval_of_f_range_of_f_l814_814648

noncomputable def f (x : ℝ) : ℝ := 3^(x^2 - 2*x + 3)

theorem increasing_interval_of_f :
  ∀ x, x ∈ Icc 1 ∞ → deriv f x > 0 := sorry

theorem decreasing_interval_of_f :
  ∀ x, x ∈ Icc ∞ 1 → deriv f x < 0 := sorry

theorem range_of_f :
  ∀ y, ∃ x, y = f x ∧ y ∈ Icc 9 ∞ := sorry

end increasing_interval_of_f_decreasing_interval_of_f_range_of_f_l814_814648


namespace vehicle_speed_increase_l814_814832

theorem vehicle_speed_increase
  (vA vB vC : ℝ)
  (dAB dAC : ℝ)
  (speed_limit : ℝ)
  (hAB : vA = 60)
  (hBB : vB = 50)
  (hBC : vC = 70)
  (hDAB : dAB = 100)
  (hDAC : dAC = 500)
  (hSL : speed_limit = 80) :
  ∃ ΔvA : ℝ, ΔvA = 20 :=
begin
  sorry
end

end vehicle_speed_increase_l814_814832


namespace find_set_abc_l814_814970

theorem find_set_abc (a b c : ℝ) 
  (h1: {a, b, c} = {-a^3, abc - b^3, 2 * abc - c^3}) 
  : {a, b, c} = {-1, 0, 1} := 
by 
  sorry

end find_set_abc_l814_814970


namespace parallel_vectors_sin_cos_l814_814656

theorem parallel_vectors_sin_cos (θ : ℝ) (a := (6, 3)) (b := (Real.sin θ, Real.cos θ))
  (h : (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2)) :
  Real.sin (2 * θ) - 2 * (Real.cos θ)^2 = 2 / 5 :=
by
  sorry

end parallel_vectors_sin_cos_l814_814656


namespace circumcenter_BOD_lies_on_AC_l814_814806

-- Define the geometrical entities and their properties
variables {A B C D O : Type} [Incircle A B C D O]
variable (pointNotOnDiagonals : ¬(liesOnDiagonal O A B C D))
variable (circumcenterAOC : Line BD contains Circumcenter (Triangle A O C))

theorem circumcenter_BOD_lies_on_AC (h: ByCircumcenter (A O C) (B D) ): Line AC contains Circumcenter (Triangle B O D) := sorry

end circumcenter_BOD_lies_on_AC_l814_814806


namespace sum_of_fundamental_terms_divisible_by_4_l814_814823

theorem sum_of_fundamental_terms_divisible_by_4
  (n : ℕ) (hn : n ≤ 4)
  (grid : Fin n → Fin n → ℤ)
  (hgrid : ∀ i j, grid i j = 1 ∨ grid i j = -1) :
  let fundamental_terms := { f : Fin n → Fin n // function.bijective f }
  let term_prod := λ f : { f : Fin n → Fin n // function.bijective f }, ∏ i, grid i (f.val i)
  let S := ∑ f in set.to_finset fundamental_terms, term_prod f
  in 4 ∣ S :=
sorry

end sum_of_fundamental_terms_divisible_by_4_l814_814823


namespace modulus_of_complex_number_l814_814328

noncomputable def z : ℂ := (1 - complex.i)^2 / (1 + complex.i)

theorem modulus_of_complex_number : complex.abs z = real.sqrt 2 :=
by
  sorry

end modulus_of_complex_number_l814_814328


namespace parabola_properties_l814_814292

variable {x y c: ℝ}
def parabola (x y : ℝ) := (x^2 = 4 * y)

noncomputable def focus := (0, 1)

noncomputable def point_P := {P : (ℝ × ℝ) // P ∈ {(2, 1), (-2, 1)}}

noncomputable def max_area (A B P : (ℝ × ℝ)) := 4 * real.sqrt 2

theorem parabola_properties :
  (∀ P : (ℝ × ℝ), parabola P.1 P.2 ∧ |P - focus| = 2 → P ∈ {(2, 1), (-2, 1)}) ∧
  (∀ A B P: (ℝ × ℝ), parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ P ∈ arc A B → area P A B ≤ max_area A B P) :=
by sorry

end parabola_properties_l814_814292


namespace value_of_expression_l814_814857

theorem value_of_expression : (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) = 470 :=
by
  sorry

end value_of_expression_l814_814857


namespace remainder_div_3005_95_l814_814840

theorem remainder_div_3005_95 : 3005 % 95 = 60 := 
by {
  sorry
}

end remainder_div_3005_95_l814_814840


namespace find_highest_score_l814_814049

noncomputable def highest_score (A : ℕ) (I : ℕ) (diff_HL : ℕ) (A' : ℕ) (I' : ℕ) : ℕ :=
let total_runs := A * I in
let total_runs_excl := A' * I' in
let sum_HL := total_runs - total_runs_excl in
let H := (sum_HL + diff_HL) / 2 in
H

theorem find_highest_score :
  let A := 62
  let I := 46
  let diff_HL := 150
  let A' := 58
  let I' := 44 in
  highest_score A I diff_HL A' I' = 225 :=
by
  sorry

end find_highest_score_l814_814049


namespace circle_through_points_l814_814595

-- Definitions of the points
def O : (ℝ × ℝ) := (0, 0)
def M1 : (ℝ × ℝ) := (1, 1)
def M2 : (ℝ × ℝ) := (4, 2)

-- Definition of the center and radius of the circle
def center : (ℝ × ℝ) := (4, -3)
def radius : ℝ := 5

-- The circle equation function
def circle_eq (x y : ℝ) (c : ℝ × ℝ) (r : ℝ) : Prop :=
  (x - c.1)^2 + (y + c.2)^2 = r^2

theorem circle_through_points :
  circle_eq 0 0 center radius ∧ circle_eq 1 1 center radius ∧ circle_eq 4 2 center radius :=
by
  -- This is where the proof would go
  sorry

end circle_through_points_l814_814595


namespace planes_parallel_or_intersect_l814_814622

variables {Plane : Type} {Line : Type}
variables (α β : Plane) (a b : Line)

-- Conditions
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def not_parallel (l1 l2 : Line) : Prop := sorry

-- Given conditions
axiom h₁ : line_in_plane a α
axiom h₂ : line_in_plane b β
axiom h₃ : not_parallel a b

-- The theorem statement
theorem planes_parallel_or_intersect : (exists l : Line, line_in_plane l α ∧ line_in_plane l β) ∨ (α = β) :=
sorry

end planes_parallel_or_intersect_l814_814622


namespace no_friendly_triplet_in_range_l814_814095

open Nat

def isFriendly (a b c : ℕ) : Prop :=
  (a ∣ (b * c) ∨ b ∣ (a * c) ∨ c ∣ (a * b))

theorem no_friendly_triplet_in_range (n : ℕ) (a b c : ℕ) :
  n^2 < a ∧ a < n^2 + n → n^2 < b ∧ b < n^2 + n → n^2 < c ∧ c < n^2 + n → a ≠ b → b ≠ c → a ≠ c →
  ¬ isFriendly a b c :=
by sorry

end no_friendly_triplet_in_range_l814_814095


namespace problem1_problem2_l814_814345

variables {n m θ : ℝ}
variables (P : ℝ × ℝ)
noncomputable def A := (n, 2 / n)
noncomputable def B := (8 - m, 8 - 3 / 2 * m)
noncomputable def C := (7 - m, 0)
noncomputable def A' := (2, -4)
noncomputable def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.1 - p1.1) * (p3.2 - p1.2) = (p3.1 - p1.1) * (p2.2 - p1.2)

-- (1) Coordinates of P
noncomputable def P := (10 / 3, 0)

theorem problem1 (h1 : collinear A' P B) (h_m : m = 4) :
  P = (10 / 3, 0) :=
sorry

-- (2) Range of values for m
noncomputable def Aθ := (sin θ, 2 / sin θ)
noncomputable def CA := (sin θ + m - 7, 2 / sin θ)
noncomputable def CB := (1, 8 - 3 / 2 * m)
noncomputable def angle_between (v1 v2 : ℝ × ℝ) : ℝ :=
  real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (real.sqrt ((v1.1 ^ 2 + v1.2 ^ 2) * (v2.1 ^ 2 + v2.2 ^ 2))))

theorem problem2 (h_n : n = sin θ) (h_theta : θ ∈ Ioo 0 π) (h_alpha : angle_between CA CB ∈ Icc 0 (π / 2)) :
  m < 5 :=
sorry

end problem1_problem2_l814_814345


namespace evaluate_expression_l814_814382

theorem evaluate_expression :
  let a := Real.sqrt 2 ^ 2 + Real.sqrt 3 + Real.sqrt 5
  let b := - Real.sqrt 2 ^ 2 + Real.sqrt 3 + Real.sqrt 5
  let c := Real.sqrt 2 ^ 2 - Real.sqrt 3 + Real.sqrt 5
  let d := - Real.sqrt 2 ^ 2 - Real.sqrt 3 + Real.sqrt 5
  (1/a + 1/b + 1/c + 1/d)^2 = 5 :=
by
  sorry

end evaluate_expression_l814_814382


namespace median_baskets_l814_814715

-- Define the data set as a list of natural numbers
def baskets : List ℕ := [2, 3, 4, 4, 3, 5]

-- Define the problem statement that asserts the median of the data set
theorem median_baskets : List.median baskets = 3.5 :=
by
  sorry

end median_baskets_l814_814715


namespace unique_line_parallel_l814_814476

variables {α β : Type} [plane α] [plane β]
variables {a : line α} {M : point β}

-- Definition indicating parallel planes
def planes_parallel (α β : Type) [plane α] [plane β] : Prop :=
  ∀ (p : point), (∃ (a : line α), p ∈ a) ↔ (∃ (b : line β), p ∈ b)

-- Definition indicating points and lines on a plane
def point_on_plane (p : point) (α : Type) [plane α] : Prop :=
  ∃ (a : line α), p ∈ a

def line_on_plane (l : line α) (α : Type) [plane α] : Prop :=
  ∀ (p : point), p ∈ l → p ∈ α

-- Main Lean 4 statement
theorem unique_line_parallel (h1 : planes_parallel α β)
                             (h2 : line_on_plane a α)
                             (h3 : M ∈ β)
                             : ∃! (b : line β), M ∈ b ∧ ∀ (p : point), p ∈ b ↔ p ∈ a := sorry

end unique_line_parallel_l814_814476


namespace soccer_balls_per_class_l814_814145

-- Definitions for all conditions in the problem
def elementary_classes_per_school : ℕ := 4
def middle_school_classes_per_school : ℕ := 5
def number_of_schools : ℕ := 2
def total_soccer_balls_donated : ℕ := 90

-- The total number of classes in one school
def classes_per_school : ℕ := elementary_classes_per_school + middle_school_classes_per_school

-- The total number of classes in both schools
def total_classes : ℕ := classes_per_school * number_of_schools

-- Prove that the number of soccer balls donated per class is 5
theorem soccer_balls_per_class : total_soccer_balls_donated / total_classes = 5 :=
  by sorry

end soccer_balls_per_class_l814_814145


namespace range_of_a_l814_814329

noncomputable def f (a x : ℝ) : ℝ := (a^2 - 2 * a - 3) * x^2 + (a - 3) * x + 1

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, f a x ∈ set.univ) ∧ (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ a = -1 :=
by
  -- proof goes here
  sorry

end range_of_a_l814_814329


namespace minimum_sum_labels_l814_814071

open Matrix

-- Define the labeling of the chessboard
def label (i j : ℕ) : ℝ := 1 / (↑(i + j - 1)^2)

-- Define the condition on the chosen squares: Exactly one square from each row and column, no adjacent squares
def valid_selection (selection : Fin 9 → Fin 9) : Prop :=
  ∀ i j : Fin 9, i ≠ j → selection i ≠ selection j ∧ (selection i.succ = selection i + 1)

-- The desired minimum sum of the labels
def min_sum (selection : Fin 9 → Fin 9) : ℝ :=
  ∑ i : Fin 9, label (selection i).succ i.succ

theorem minimum_sum_labels : ∃ selection : Fin 9 → Fin 9, min_sum selection = 1 / 969 :=
by
  sorry

end minimum_sum_labels_l814_814071


namespace analytical_expression_of_f_l814_814274

theorem analytical_expression_of_f (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f (x + 1 / x) = x^2 + 1 / x^2) →
  (∀ y : ℝ, (y ≥ 2 ∨ y ≤ -2) → f y = y^2 - 2) :=
by
  intro h1 y hy
  sorry

end analytical_expression_of_f_l814_814274


namespace probability_grade_A_l814_814351

theorem probability_grade_A (P_B : ℝ) (P_C : ℝ) (hB : P_B = 0.03) (hC : P_C = 0.01) : 
  1 - (P_B + P_C) = 0.96 :=
by
  -- Use the facts P_B = 0.03 and P_C = 0.01 to show the conclusion
  rw [hB, hC]
  -- Simplify the arithmetic to get the final result
  norm_num
  -- To finish, we state the target goal and not solve it entirely here
  sorry

end probability_grade_A_l814_814351


namespace probability_two_points_one_unit_apart_l814_814455

def twelve_points_probability : ℚ := 2 / 11

/-- Twelve points are spaced around at intervals of one unit around a \(3 \times 3\) square.
    Two of the 12 points are chosen at random.
    Prove that the probability that the two points are one unit apart is \(\frac{2}{11}\). -/
theorem probability_two_points_one_unit_apart :
  let total_points := 12
  let total_combinations := (total_points * (total_points - 1)) / 2
  let favorable_pairs := 12
  (favorable_pairs : ℚ) / total_combinations = twelve_points_probability := by
  sorry

end probability_two_points_one_unit_apart_l814_814455


namespace inclusion_M_P_N_l814_814079

def M : set (ℝ × ℝ) := {p : ℝ × ℝ | abs p.1 + abs p.2 < 1}

def N : set (ℝ × ℝ) := {p : ℝ × ℝ | real.sqrt ((p.1 - 0.5) ^ 2 + (p.2 + 0.5) ^ 2) + real.sqrt ((p.1 + 0.5) ^ 2 + (p.2 - 0.5) ^ 2) < 2 * real.sqrt 2}

def P : set (ℝ × ℝ) := {
  p : ℝ × ℝ | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1
}

theorem inclusion_M_P_N : M ⊆ P ∧ P ⊆ N := by
sorry

end inclusion_M_P_N_l814_814079


namespace area_of_figure_points_P_l814_814616

theorem area_of_figure_points_P {x y x₀ y₀ : ℝ}
  (h1 : (x - x₀)^2 + (y - y₀)^2 = 16)
  (h2 : x₀^2 + y₀^2 = 4) :
  ∀ (a : set (ℝ × ℝ)), (∀ (P : ℝ × ℝ), P ∈ a ↔ ∃ (x y : ℝ), (P = (x, y) ∧ (x - x₀)^2 + (y - y₀)^2 = 16 ∧ x₀^2 + y₀^2 = 4)) → 
  ∃ (r_inner r_outer : ℝ), r_inner = 2 ∧ r_outer = 6 ∧ 
  (π * (r_outer^2 - r_inner^2) = 32 * π) :=
by sorry

end area_of_figure_points_P_l814_814616


namespace rect_area_162_l814_814148

def rectangle_field_area (w l : ℝ) (A : ℝ) : Prop :=
  w = (1/2) * l ∧ 2 * (w + l) = 54 ∧ A = w * l

theorem rect_area_162 {w l A : ℝ} :
  rectangle_field_area w l A → A = 162 :=
by
  intro h
  sorry

end rect_area_162_l814_814148


namespace find_m_l814_814330

theorem find_m (m : ℝ) : 
  (m^2 + 3 * m + 3 ≠ 0) ∧ (m^2 + 2 * m - 3 ≠ 0) ∧ 
  (m^2 + 3 * m + 3 = 1) → m = -2 := 
by
  sorry

end find_m_l814_814330


namespace arithmetic_sequence_variance_d_squared_l814_814981

-- Define the arithmetic sequence and variance properties
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def variance (s : fin 5 → ℝ) : ℝ :=
  let m := (s 0 + s 1 + s 2 + s 3 + s 4) / 5 in
  (1 / 5) * ((s 0 - m) ^ 2 + (s 1 - m) ^ 2 + (s 2 - m) ^ 2 + (s 3 - m) ^ 2 + (s 4 - m) ^ 2)

theorem arithmetic_sequence_variance_d_squared (a : ℕ → ℝ) (d : ℝ)
  (h1 : is_arithmetic_sequence(a, d))
  (h2 : variance (λ i : fin 5, a i + d * i) = 8) :
  d^2 = 4 :=
by
  sorry

end arithmetic_sequence_variance_d_squared_l814_814981


namespace total_price_of_purchases_l814_814108

def price_of_refrigerator := 4275
def price_difference := 1490
def price_of_washing_machine := price_of_refrigerator - price_difference
def total_price := price_of_refrigerator + price_of_washing_machine

theorem total_price_of_purchases : total_price = 7060 :=
by
  rfl  -- This is just a placeholder; you need to solve the proof.

end total_price_of_purchases_l814_814108


namespace range_a_for_false_proposition_l814_814692

theorem range_a_for_false_proposition :
  {a : ℝ | ¬ ∃ x : ℝ, x^2 + (1 - a) * x < 0} = {1} :=
sorry

end range_a_for_false_proposition_l814_814692


namespace probability_of_twice_event_l814_814839

def digits := {1, 2, 3, 4}

noncomputable def total_events : ℕ := 4 * 4

noncomputable def specific_events : ℕ := 4 -- (pairs: (1,2), (2,1), (2,4), (4,2))

noncomputable def desired_probability : ℚ := specific_events / total_events

theorem probability_of_twice_event :
  desired_probability = 1 / 4 := 
sorry

end probability_of_twice_event_l814_814839


namespace quadratic_polynomial_with_root_and_coefficient_l814_814215

theorem quadratic_polynomial_with_root_and_coefficient :
  ∃ (p : ℝ[X]), (p.coeff 2 = 3) ∧ (p.eval (4 + complex.i) = 0) ∧ (p = 3 * (X - 4)^2 + 51 * X - 24) :=
by sorry

end quadratic_polynomial_with_root_and_coefficient_l814_814215


namespace punger_bought_60_packs_l814_814398

theorem punger_bought_60_packs
  (packs_cards : ℕ)
  (pages : ℕ)
  (cards_per_pack : ℕ)
  (cards_per_page : ℕ)
  (total_pages : ℕ)
  (total_cards : packs_cards * cards_per_pack = pages * cards_per_page) :
  total_pages = 42 → cards_per_pack = 7 → cards_per_page = 10 → packs_cards = 60 :=
by
  intros h_total_pages h_cards_per_pack h_cards_per_page,
  have h1 : total_cards = packs_cards * cards_per_pack := by assumption,
  have h2 : total_cards = 420 := by
    rw [←h_total_pages, ←h_cards_per_page],
    norm_num,
  have h3 : packs_cards * cards_per_pack = 420 := by
    rw [h1, h2],
    norm_num,
  rw h_cards_per_pack at h3,
  exact nat.eq_of_mul_eq_mul_right (show 7 > 0, by norm_num) h3,
  sorry -- Since full proof is not needed

end punger_bought_60_packs_l814_814398


namespace mildred_total_oranges_l814_814758

-- Conditions
def initial_oranges : ℕ := 77
def additional_oranges : ℕ := 2

-- Question/Goal
theorem mildred_total_oranges : initial_oranges + additional_oranges = 79 := by
  sorry

end mildred_total_oranges_l814_814758


namespace find_tan_alpha_l814_814270

-- Given conditions as definitions
variable {α β : ℝ}
def tan_sum_condition : ℝ := (3/5) -- represents tan(α + β) = 3/5
def tan_beta : ℝ := (1/3) -- represents tan(β) = 1/3

-- The mathematically equivalent proof statement
theorem find_tan_alpha (h1 : Real.tan (α + β) = tan_sum_condition) (h2 : Real.tan β = tan_beta) : Real.tan α = (2/9) :=
by sorry

end find_tan_alpha_l814_814270


namespace circle_intersection_product_l814_814199

theorem circle_intersection_product :
  let first_circle := {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 + 3)^2 = 1},
      second_circle := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 3)^2 = 4},
      intersection_points := first_circle ∩ second_circle,
      products := intersection_points.map (λ p => p.1 * p.2)
  in products = {-1.5} :=
by
  sorry

end circle_intersection_product_l814_814199


namespace find_n_l814_814676

theorem find_n (n : ℤ) (h : Real.sqrt (10 + n) = 9) : n = 71 :=
sorry

end find_n_l814_814676


namespace part1_daily_sales_profit_part2_maximum_daily_profit_l814_814414

-- Definitions of initial conditions
def original_price : ℝ := 30
def original_sales_volume : ℝ := 60
def cost_price : ℝ := 15
def price_reduction_effect : ℝ := 10

-- Part 1: Prove the daily sales profit if the price is reduced by 2 yuan
def new_price_after_reduction (reduction : ℝ) : ℝ := original_price - reduction
def new_sales_volume (reduction : ℝ) : ℝ := original_sales_volume + reduction * price_reduction_effect
def profit_per_kg (selling_price : ℝ) : ℝ := selling_price - cost_price
def daily_sales_profit (reduction : ℝ) : ℝ := profit_per_kg (new_price_after_reduction reduction) * new_sales_volume reduction

theorem part1_daily_sales_profit : daily_sales_profit 2 = 1040 := by sorry

-- Part 2: Prove the selling price for maximum profit and the maximum profit
def profit_function (x : ℝ) : ℝ := (x - cost_price) * (original_sales_volume + (original_price - x) * price_reduction_effect)

theorem part2_maximum_daily_profit : 
  ∃ x, profit_function x = 1102.5 ∧ x = 51 / 2 := by sorry

end part1_daily_sales_profit_part2_maximum_daily_profit_l814_814414


namespace least_positive_base_ten_number_with_eight_binary_digits_l814_814471

theorem least_positive_base_ten_number_with_eight_binary_digits : 
  ∃ n : ℕ, (n = 128) ∧ (binary_digits n = 8) :=
begin
  sorry
end

noncomputable def binary_digits (n : ℕ) : ℕ :=
  nat.log n 2 + 1

end least_positive_base_ten_number_with_eight_binary_digits_l814_814471


namespace find_f_f_neg2_l814_814640

def f (x : ℝ) : ℝ := if x < 0 then x + 2 else 3^(x + 1)

theorem find_f_f_neg2 : f (f (-2)) = 3 := 
  sorry

end find_f_f_neg2_l814_814640


namespace price_increase_is_12_percent_l814_814536

theorem price_increase_is_12_percent
    (P : ℝ) (d : ℝ) (P' : ℝ) (sale_price : ℝ) (increase : ℝ) (percentage_increase : ℝ) :
    P = 470 → d = 0.16 → P' = 442.18 → 
    sale_price = P - P * d →
    increase = P' - sale_price →
    percentage_increase = (increase / sale_price) * 100 →
    percentage_increase = 12 :=
  by
  sorry

end price_increase_is_12_percent_l814_814536


namespace find_number_l814_814874

theorem find_number (x : ℕ) (h1 : x - 13 = 31) : x + 11 = 55 :=
  sorry

end find_number_l814_814874


namespace probability_of_triangle_formation_l814_814945

def sticks : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

noncomputable def valid_triangle_combinations (s : List ℕ) : ℕ :=
  (s.erase 2).erase 3.length -- This is simplified, for illustration. Properly computing valid set count is required.

def total_combinations : ℕ := (List.choose sticks 3).length

def probability_triangle : ℚ := valid_triangle_combinations sticks / total_combinations

theorem probability_of_triangle_formation : probability_triangle = 1 / 4 := by
  sorry

end probability_of_triangle_formation_l814_814945


namespace probability_of_drawing_three_white_balls_l814_814509

theorem probability_of_drawing_three_white_balls
  (total_balls white_balls black_balls: ℕ)
  (h_total: total_balls = 15)
  (h_white: white_balls = 7)
  (h_black: black_balls = 8)
  (draws: ℕ)
  (h_draws: draws = 3) :
  (Nat.choose white_balls draws / Nat.choose total_balls draws) = (7 / 91) :=
by sorry

end probability_of_drawing_three_white_balls_l814_814509


namespace distance_school_house_l814_814114

def speed_to_school : ℝ := 6
def speed_from_school : ℝ := 4
def total_time : ℝ := 10

theorem distance_school_house : 
  ∃ D : ℝ, (D / speed_to_school + D / speed_from_school = total_time) ∧ (D = 24) :=
sorry

end distance_school_house_l814_814114


namespace count_4x4_increasing_arrays_l814_814056

-- Define the notion of a 4x4 grid that satisfies the given conditions
def isInIncreasingOrder (matrix : (Fin 4) → (Fin 4) → Nat) : Prop :=
  (∀ i j : Fin 4, i < 3 -> matrix i j < matrix (i+1) j) ∧
  (∀ i j : Fin 4, j < 3 -> matrix i j < matrix i (j+1))

def validGrid (matrix : (Fin 4) → (Fin 4) → Nat) : Prop :=
  (∀ i j : Fin 4, 1 ≤ matrix i j ∧ matrix i j ≤ 16) ∧ isInIncreasingOrder matrix

noncomputable def countValidGrids : ℕ :=
  sorry

theorem count_4x4_increasing_arrays : countValidGrids = 13824 :=
  sorry

end count_4x4_increasing_arrays_l814_814056


namespace ratio_of_d_to_s_l814_814929

theorem ratio_of_d_to_s (s d : ℝ) (n : ℕ) (h1 : n = 15) (h2 : (n^2 * s^2) / ((n * s + 2 * n * d)^2) = 0.75) :
  d / s = 1 / 13 :=
by
  sorry

end ratio_of_d_to_s_l814_814929


namespace triangle_construction_l814_814092

theorem triangle_construction
  (A B C M N K Q : Type)
  (AM : A → M)
  (BN : B → N)
  (CK : C → K)
  (line_l : set (Type))
  (angle_bisector_CK : CK ∈ line_l)
  (intersection_Q : Q = (AM ∩ BN)) :
  ∃ (triangle_ABC : Type),
    -- Further statements about the construction steps that ensure triangle ABC
sorry

end triangle_construction_l814_814092


namespace probability_of_same_color_is_correct_l814_814201

def probability_same_color (blue_balls yellow_balls : ℕ) : ℚ :=
  let total_balls := blue_balls + yellow_balls
  let prob_blue := (blue_balls / total_balls : ℚ)
  let prob_yellow := (yellow_balls / total_balls : ℚ)
  (prob_blue ^ 2) + (prob_yellow ^ 2)

theorem probability_of_same_color_is_correct :
  probability_same_color 8 5 = 89 / 169 :=
by 
  sorry

end probability_of_same_color_is_correct_l814_814201


namespace contrapositive_l814_814625

theorem contrapositive (q p : Prop) (h : q → p) : ¬p → ¬q :=
by
  -- Proof will be filled in later.
  sorry

end contrapositive_l814_814625


namespace game_terminates_if_n_lt_1994_game_does_not_terminate_if_n_eq_1994_l814_814123

-- Definitions and conditions for the problem
def num_girls : ℕ := 1994
def tokens (n : ℕ) := n

-- Main theorem statements
theorem game_terminates_if_n_lt_1994 (n : ℕ) (h : n < num_girls) :
  ∃ (S : ℕ) (invariant : ℕ) (steps : ℕ), (∀ j : ℕ, 1 ≤ j ∧ j ≤ num_girls → (tokens n % num_girls) ≤ 1) :=
by
  sorry

theorem game_does_not_terminate_if_n_eq_1994 :
  ∃ (S : ℕ) (invariant : ℕ) (steps : ℕ), (tokens 1994 % num_girls = 0) :=
by
  sorry

end game_terminates_if_n_lt_1994_game_does_not_terminate_if_n_eq_1994_l814_814123


namespace average_of_second_and_fourth_is_8_l814_814063

theorem average_of_second_and_fourth_is_8 (l : List ℤ)
  (h1 : l ~ [ -3, 2, 5, 8, 11 ])  --l is a permutation of [-3, 2, 5, 8, 11]
  (h2 : ∃ i, 0 ≤ i ∧ i < l.length ∧ l.nth i = some 11 ∧ 2 ≤ i)
  (h3 : ∃ i, 0 ≤ i ∧ i < l.length ∧ l.nth i = some 11 ∧ i < 4)
  (h4 : ∃ i, 0 ≤ i ∧ i < l.length ∧ l.nth i = some (-3) ∧ 1 ≤ i ∧ i < 3)
  (h5 : ∃ i, 0 ≤ i ∧ i < l.length ∧ l.nth i = some (-3) ∧ i < 3)
  (h6 : ∃ i, 0 ≤ i ∧ i < l.length ∧ l.nth i = some 5 ∧ 1 ≤ i ∧ i < 3)
  (h7 : ∃ j, 0 ≤ j ∧ j < l.length ∧ l.nth j = some (-3) ∧ j < l.length)
  (h8 : ∃ k, j+n ∃ k, 0 ≤ k ∧ k < l.length ∧ l.nth k = (l.nth j-1))
  (h9 : ∃ i, 0 ≤ i ∧ i < l.length ∧ l.head l ≠ 11)
  (h10 : ∃ i, 0 ≤ i ∧ i < l.length ∧ l.lphone l ≠ 5)
: (l !! 1 + l !! 3) / 2 = 8 := sorry

end average_of_second_and_fourth_is_8_l814_814063


namespace cos_neg_30_eq_sqrt_3_div_2_l814_814442

theorem cos_neg_30_eq_sqrt_3_div_2 : 
  Real.cos (-30 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_neg_30_eq_sqrt_3_div_2_l814_814442


namespace circle_equation_tangent_y_axis_center_on_line_chord_length_condition_l814_814240

theorem circle_equation_tangent_y_axis_center_on_line_chord_length_condition :
  ∃ (x₀ y₀ r : ℝ), 
  (x₀ - 3 * y₀ = 0) ∧ 
  (r = |3 * y₀|) ∧ 
  ((x₀ + 3)^2 + (y₀ - 1)^2 = r^2 ∨ (x₀ - 3)^2 + (y₀ + 1)^2 = r^2) :=
sorry

end circle_equation_tangent_y_axis_center_on_line_chord_length_condition_l814_814240


namespace trapezoid_sides_l814_814531

noncomputable def trapezoid_side_lengths (x y z : ℝ) : Prop :=
  let area := 3600.0 in
  let ratio := 5 / 4 in
  let eq_triangle_area := (area * 5) / 9 in
  let gen_triangle_area := (area * 4) / 9 in
  let a := 40 * Real.sqrt (5 / Real.sqrt (3)) in
  let b := 32 * Real.sqrt (5 / Real.sqrt (3)) in
  let c := 8 * Real.sqrt (105 / Real.sqrt (3)) in
  area = 3600 ∧ ratio = 5 / 4 ∧
  eq_triangle_area = (Real.sqrt 3 / 4) * a^2 ∧
  gen_triangle_area = (1 / 2) * b * c ∧
  x = a ∧ y = b ∧ z = c

theorem trapezoid_sides : ∃ (x y z : ℝ), trapezoid_side_lengths x y z :=
sorry

end trapezoid_sides_l814_814531


namespace students_at_1544_l814_814179

noncomputable def students_in_lab : Nat := 44

theorem students_at_1544 :
  let initial_students := 20
  let enter_interval := 3
  let enter_students := 4
  let leave_interval := 10
  let leave_students := 8

  ∃ students : Nat,
    students = initial_students
    + (34 / enter_interval) * enter_students
    - (34 / leave_interval) * leave_students
    ∧ students = students_in_lab :=
by
  let initial_students := 20
  let enter_interval := 3
  let enter_students := 4
  let leave_interval := 10
  let leave_students := 8
  use 20 + (34 / 3) * 4 - (34 / 10) * 8
  sorry

end students_at_1544_l814_814179


namespace john_jane_pass_each_other_distance_from_top_l814_814363

/-- John and Jane's race conditions -/
def john_uphill_speed : ℝ := 10 -- in km/hr
def john_downhill_speed : ℝ := 15 -- in km/hr
def john_head_start : ℝ := 12 / 60 -- in hours (12 minutes converted to hours)

def jane_uphill_speed : ℝ := 12 -- in km/hr
def jane_downhill_speed : ℝ := 18 -- in km/hr

def start_distance_uphill : ℝ := 8 -- km
def start_distance_downhill : ℝ := 6 -- km

theorem john_jane_pass_each_other_distance_from_top :
  let
    t_john_uphill := start_distance_uphill / john_uphill_speed,
    t_jane_uphill := start_distance_uphill / jane_uphill_speed,
    x := 22.4 / 27,
    jane_position_when_meet := 12 * (x - john_head_start),
    distance_from_top := start_distance_uphill - jane_position_when_meet in
  distance_from_top = 24 / 27 :=
by
  sorry

end john_jane_pass_each_other_distance_from_top_l814_814363


namespace smallest_sum_l814_814256

theorem smallest_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) 
  (h_fraction : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) : x + y = 64 :=
sorry

end smallest_sum_l814_814256


namespace max_value_of_xyz_l814_814751

theorem max_value_of_xyz (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 2) :
  x^3 * y^2 * z^4 ≤ 13824 / 40353607 :=
begin
  sorry,
end

end max_value_of_xyz_l814_814751


namespace who_is_who_l814_814393

-- Definitions of the brothers
def Pervy_is_Tralyalya := ∀ (Pervy : Prop), Pervy = Tralyalya
def Vtory_if_True_then_Trulalya := ∀ (Pervy Vtory : Prop), (Pervy = Tralyalya) → (Vtory = Trulalya)

-- The lean statement that formalizes the proof problem
theorem who_is_who (Pervy Vtory : Prop) (Pervy_is_Tralyalya : Pervy = Tralyalya) 
                    (Vtory_if_True_then_Trulalya : (Pervy = Tralyalya) → (Vtory = Trulalya)) 
                    (weekday_meeting : True) : 
    (Pervy = Trulalya) ∧ (Vtory = Tralyalya) :=
by {
    -- Proof would go here
    sorry
}

end who_is_who_l814_814393


namespace unique_path_contest_l814_814013

theorem unique_path_contest : 
  ∀ (grid : list (list char)),
  grid = [
    ['C'],
    ['C', 'O'],
    ['C', 'O', 'N'],
    ['C', 'O', 'N', 'T'],
    ['C', 'O', 'N', 'T', 'E'],
    ['C', 'O', 'N', 'T', 'E', 'S'],
    ['C', 'O', 'N', 'T', 'E', 'S', 'T']
  ] → 
  (∃! path : list (nat × nat), 
    path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)] ∈ grid →
      ∀ i < 6, 
        let (x1, y1) := path[i] in
        let (x2, y2) := path[i + 1] in 
        (x2 = x1 + 1 ∧ y2 = y1 + 1 ∨ y2 = y1 ∧ x2 = x1 + 1) :=
by
  sorry

end unique_path_contest_l814_814013


namespace sum_floor_305_div_503_l814_814184

theorem sum_floor_305_div_503 :
  ∑ n in finset.range 503, (floor ((305 * n) / 503 : ℚ)) = 76304 :=
by
  sorry

end sum_floor_305_div_503_l814_814184


namespace square_free_even_less_than_200_count_l814_814666

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m^2 ∣ n → m = 1

def is_multiple_of_2 (n : ℕ) : Prop := n % 2 = 0

theorem square_free_even_less_than_200_count : ∃ (count : ℕ), count = 38 ∧ (∀ n : ℕ, n < 200 ∧ is_multiple_of_2 n ∧ is_square_free n → count = 38) :=
by
  sorry

end square_free_even_less_than_200_count_l814_814666


namespace jacob_needs_to_catch_26_more_fish_l814_814916

noncomputable def Jacob_needs_to_catch (jacob_initial_fish alex_lost_fish : ℕ) :=
  let jacob_final_fish := 34 - jacob_initial_fish
  in jacob_final_fish

theorem jacob_needs_to_catch_26_more_fish :
  let jacob_initial_fish := 8 in
  let alex_factor := 7 in
  let alex_initial_fish := alex_factor * jacob_initial_fish in
  let alex_lost_fish := 23 in
  let alex_final_fish := alex_initial_fish - alex_lost_fish in
  let jacob_final_fish := alex_final_fish + 1 in
  Jacob_needs_to_catch jacob_initial_fish alex_lost_fish = 26 :=
by
  let jacob_initial_fish := 8
  let alex_initial_fish := 7 * jacob_initial_fish
  let alex_lost_fish := 23
  let alex_final_fish := alex_initial_fish - alex_lost_fish
  let jacob_final_fish := alex_final_fish + 1
  let needed := jacob_final_fish - jacob_initial_fish
  have : needed = 26 := by sorry
  exact this

end jacob_needs_to_catch_26_more_fish_l814_814916


namespace randy_kept_value_l814_814036

theorem randy_kept_value (initial_amount : ℕ) (gift_amount : ℕ) (given_amount : ℕ) (final_amount : ℕ) 
  (h1 : initial_amount = 3000) 
  (h2 : gift_amount = 200) 
  (h3 : given_amount = 1200) 
  : final_amount = 2000 :=
begin
  have total_amount := initial_amount + gift_amount,
  have remaining_amount := total_amount - given_amount,
  calc 
    final_amount = remaining_amount : by rw [remaining_amount]
    ... = initial_amount + gift_amount - given_amount : by rw [total_amount]
    ... = 3000 + 200 - 1200 : by rw [h1, h2, h3]
    ... = 2000 : by norm_num
end

end randy_kept_value_l814_814036


namespace shortest_side_length_inscribed_circle_l814_814515

theorem shortest_side_length_inscribed_circle
  (A B C D : ℝ)
  (AB : ℝ) (AD : ℝ := 7) (DB : ℝ := 9)
  (r : ℝ := 5) (angleC : ℝ := 60)
  (s : ℝ) (Δ : ℝ) (x : ℝ)
  (BC CA : ℝ := 2 * x) :
  AB = AD + DB ∧ angleC = 60 ∧ r = 5 ∧
  s = (AB + BC + CA) / 2 ∧ Δ = 1 / 2 * AB * BC * real.sin (angleC * real.pi / 180) ∧
  r = Δ / s ∧ x = 10 * (2 + real.sqrt 3) →
  BC = 20 * (2 + real.sqrt 3) :=
by
  intro h
  obtain ⟨hAB, hangleC, hr, hs, hΔ, hratio, hx⟩ := h
  sorry

end shortest_side_length_inscribed_circle_l814_814515


namespace smallest_sum_l814_814261

theorem smallest_sum (x y : ℕ) (hx : x ≠ y) (hxy : (1/x:ℚ) + (1/y:ℚ) = 1/15) : x + y = 64 :=
sorry

end smallest_sum_l814_814261


namespace oil_output_per_capita_correctness_l814_814861

variable (population_west : ℝ := 1)
variable (output_west : ℝ := 55.084)
variable (population_non_west : ℝ := 6.9)
variable (output_non_west : ℝ := 1480.689)
variable (output_russia_9_percent : ℝ := 13737.1)
variable (percentage : ℝ := 9)
variable (total_population_russia : ℝ := 147)

def west_output_per_capita : ℝ :=
  output_west / population_west

def non_west_output_per_capita : ℝ :=
  output_non_west / population_non_west

def total_output_russia : ℝ :=
  (output_russia_9_percent * 100) / percentage

def russia_output_per_capita : ℝ :=
  total_output_russia / total_population_russia

theorem oil_output_per_capita_correctness :
  west_output_per_capita = 55.084 ∧
  non_west_output_per_capita = 214.59 ∧
  total_output_russia = 152634.44 ∧
  russia_output_per_capita = 1038.33 :=
by
  sorry

end oil_output_per_capita_correctness_l814_814861


namespace complement_of_B_l814_814294

def set_B := {x : ℝ | x^2 ≤ 4}

def complement_R (B : set ℝ) := {x : ℝ | x ∉ B}

theorem complement_of_B :
  complement_R set_B = {x : ℝ | x < -2 ∨ x > 2} :=
by 
-- The proof goes here, but is not required for this task.
sorry

end complement_of_B_l814_814294


namespace minimize_expression_l814_814439

theorem minimize_expression (n : ℕ) (h : 0 < n) : 
  (n = 10) ↔ (∀ m : ℕ, 0 < m → ((n / 2) + (50 / n) ≤ (m / 2) + (50 / m))) :=
sorry

end minimize_expression_l814_814439


namespace highest_points_Ana_can_earn_l814_814912

theorem highest_points_Ana_can_earn (n : ℕ) : 
  ∃ k : ℕ, k = n / 2 ∧
    (∀ aCards bCards : List ℕ, 
      aCards = List.range' 1 (2 * n - 1) ∧ 
      bCards = List.range' 2 (2 * n) ∧
      (∀ turn : ℕ, turn < n → 
        (turn % 2 = 0 → ∃ b : ℕ, b ∈ bCards ∧ b > List.nthLe aCards turn sorry) ∧
        (turn % 2 = 1 → ∃ a : ℕ, a ∈ aCards ∧ a > List.nthLe bCards turn sorry)) 
    → Ana points = k) :=
begin
  sorry 
end

end highest_points_Ana_can_earn_l814_814912


namespace P_at_2018_l814_814379

open Polynomial

noncomputable def P : Polynomial ℤ :=
-- This represents a monic polynomial of degree 2017 satisfying the given conditions.
sorry

theorem P_at_2018 :
  ∀ (P : Polynomial ℤ), 
  (∀ n, 1 ≤ n ∧ n ≤ 2017 → P.eval n = n) ∧
  (P.degree = 2017) → 
  P.eval 2018 = nat.factorial 2017 + 2018 :=
begin
  -- Since the proof is complex, place a placeholder for the actual proof here.
  sorry
end

end P_at_2018_l814_814379


namespace no_integers_satisfy_equation_l814_814368

theorem no_integers_satisfy_equation (p : ℕ) (h : ∃ k : ℕ, p = k * (k + 1) ∧ k > 1) :
  (∀ (x : fin p → ℤ), 
     ∑ i, (x i) ^ 2 - 4 / (4 * p + 1 : ℚ) * (∑ i, x i)^2 ≠ 1) 
  ∨ (p = 6 ∨ p = 2) :=
sorry

end no_integers_satisfy_equation_l814_814368


namespace find_all_possible_f_l814_814205

-- Noncomputability is needed here since we cannot construct a function 
-- like f deterministically via computation due to the nature of the problem.
noncomputable def functional_equation_solution (f : ℕ → ℕ) := 
  (∀ a b : ℕ, f a + f b ∣ 2 * (a + b - 1)) → 
  (∀ x : ℕ, f x = 1) ∨ (∀ x : ℕ, f x = 2 * x - 1)

-- Statement of the mathematically equivalent proof problem.
theorem find_all_possible_f (f : ℕ → ℕ) : functional_equation_solution f := 
sorry

end find_all_possible_f_l814_814205


namespace john_new_weekly_earnings_l814_814365

theorem john_new_weekly_earnings :
  ∀ (original_earnings : ℤ) (percentage_increase : ℝ),
  original_earnings = 60 →
  percentage_increase = 66.67 →
  (original_earnings + (percentage_increase / 100 * original_earnings)) = 100 := 
by
  intros original_earnings percentage_increase h_earnings h_percentage
  rw [h_earnings, h_percentage]
  norm_num
  sorry

end john_new_weekly_earnings_l814_814365


namespace gauss_algorithm_sum_l814_814965

def f (x : Nat) (m : Nat) : Rat := x / (3 * m + 6054)

theorem gauss_algorithm_sum (m : Nat) :
  (Finset.sum (Finset.range (m + 2017 + 1)) (λ x => f x m)) = (m + 2017) / 6 := by
sorry

end gauss_algorithm_sum_l814_814965


namespace contrapositive_if_even_then_forall_neg_eq_contrapositive_statement_l814_814053

variable {D : Type*} [h: Nonempty D]
variable (f : D → ℝ)

def is_even_function (f : D → ℝ) := ∀ x ∈ D, f(-x) = f(x)
def not_even_function (f : D → ℝ) := ∃ x ∈ D, f(-x) != f(x)

theorem contrapositive_if_even_then_forall_neg_eq :
  (∀ x ∈ D, f x = f (- x)) → is_even_function f :=
  sorry

theorem contrapositive_statement :
  (¬ is_even_function f) → (not_even_function f) := by
  intro h1
  unfold is_even_function at h
  unfold not_even_function
  sorry

end contrapositive_if_even_then_forall_neg_eq_contrapositive_statement_l814_814053


namespace sum_of_squares_of_distances_l814_814695

/-- In triangle ABC with AB = 17, AC = 25, and BC = 28,
let M and N be the midpoints of AB and AC, respectively.
Let P be a point on BC, and Q be the second intersection point of the circumcircles
of triangles BMP and CNP.
As P moves along BC, let PQ always pass through a fixed point X.
Prove that the sum of the squares of the distances from X to each of A, B, and C is 710. -/
theorem sum_of_squares_of_distances {A B C M N P Q X : Type}
  (h1 : dist A B = 17)
  (h2 : dist A C = 25)
  (h3 : dist B C = 28)
  (h4 : is_midpoint M A B)
  (h5 : is_midpoint N A C)
  (h6 : P ∈ line_segment B C)
  (h7 : second_intersection_point (circumcircle (triangle B M P)) (circumcircle (triangle C N P)) = Q)
  (h8 : ∀ P, P ∈ line_segment B C → line (P, Q) passes_through X) :
  (dist_sq X A + dist_sq X B + dist_sq X C) = 710 :=
sorry

end sum_of_squares_of_distances_l814_814695


namespace f_3_eq_8_f_n_eq_n_squared_minus_n_plus_2_l814_814337

noncomputable def f : ℕ → ℕ 
| n := n * (n - 1) + 2 

theorem f_3_eq_8 : f 3 = 8 := by
  sorry

theorem f_n_eq_n_squared_minus_n_plus_2 (n : ℕ) : f n = n^2 - n + 2 := by
  sorry

end f_3_eq_8_f_n_eq_n_squared_minus_n_plus_2_l814_814337


namespace curve_circle_no_common_points_curve_circle_one_common_point_curve_circle_two_common_points_curve_circle_four_common_points_l814_814634

noncomputable def curveC (k : ℝ) : ℝ → ℝ → Prop := λ x y, (y - k * x) * (y - k * x + 7 / 4) = 0

noncomputable def circleM : ℝ → ℝ → Prop := λ x y, (x - 2) ^ 2 + (y - 1) ^ 2 = 1

theorem curve_circle_no_common_points (k : ℝ) :
  (k < 0 ∨ k > 35 / 12) → ∀ x y : ℝ, ¬(curveC k x y ∧ circleM x y) :=
sorry

theorem curve_circle_one_common_point (k : ℝ) :
  k = 3 / 4 → ∃! x y : ℝ, curveC k x y ∧ circleM x y :=
sorry

theorem curve_circle_two_common_points (k : ℝ) :
  (0 < k ∧ k < 3 / 4) → ∃ x1 y1 x2 y2 : ℝ, (x1 ≠ x2 ∨ y1 ≠ y2) ∧ curveC k x1 y1 ∧ circleM x1 y1 ∧ curveC k x2 y2 ∧ circleM x2 y2 :=
sorry

theorem curve_circle_four_common_points (k : ℝ) :
  (3 / 4 < k ∧ k < 4 / 3) → ∃ x1 y1 x2 y2 x3 y3 x4 y4 : ℝ, (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧ curveC k x1 y1 ∧ circleM x1 y1 ∧ curveC k x2 y2 ∧ circleM x2 y2 ∧ curveC k x3 y3 ∧ circleM x3 y3 ∧ curveC k x4 y4 ∧ circleM x4 y4 :=
sorry

end curve_circle_no_common_points_curve_circle_one_common_point_curve_circle_two_common_points_curve_circle_four_common_points_l814_814634


namespace mike_remaining_amount_l814_814804

variables (mike johnson : ℝ)
variables (ratio_mike ratio_johnson : ℝ)
variables (johnson_share_mike_spent : ℝ)
variables (one_part mike_share mike_total_output : ℝ)

definition profit_ratio (ratio_mike : ℝ) (ratio_johnson : ℝ) := ratio_mike / ratio_johnson
definition mike_share_calculation (one_part : ℝ) (ratio_mike : ℝ) := one_part * ratio_mike
definition remaining_amount (mike_total_output mike_spent : ℝ) := mike_total_output - mike_spent

theorem mike_remaining_amount :
  ratio_mike = 2 → ratio_johnson = 5 →
  johnson_share_mike_spent = 2500 / 5 → 
  mike_total_output = johnson_share_mike_spent * 2 →
  remaining_amount mike_total_output 200 = 800 :=
by
  sorry

end mike_remaining_amount_l814_814804


namespace sum_k_minus_1_p_k_eq_zero_l814_814005

theorem sum_k_minus_1_p_k_eq_zero (n : ℕ) (h : n > 1) 
(p : ℕ → ℕ) (hp0 : p 0 + p 1 + p 2 + p 3 = n * (n - 1) * (n * (n - 1)) / 2)
(hp1 : (p 1 + 2 * p 2 + 3 * p 3 = 6 * (n * (n - 1) * ((n * (n - 1)) / 12)))) :
  ∑ k in (Finset.range 4), (k - 1) * p k = 0 := by
  sorry

end sum_k_minus_1_p_k_eq_zero_l814_814005


namespace trigonometric_relationship_l814_814741

noncomputable def a : ℝ := Real.sin (393 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (55 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (50 * Real.pi / 180)

theorem trigonometric_relationship : a < b ∧ b < c := by
  sorry

end trigonometric_relationship_l814_814741


namespace fraction_to_decimal_l814_814579

theorem fraction_to_decimal : (58 : ℚ) / 160 = 0.3625 := 
by sorry

end fraction_to_decimal_l814_814579


namespace petya_wins_prize_at_least_one_person_wins_l814_814766

noncomputable def probability_petya_wins (n : ℕ) : ℝ :=
  (5 / 6) ^ n

theorem petya_wins_prize (n : ℕ) (hn : n = 9) : probability_petya_wins n = 0.194 := by
  suffices : probability_petya_wins 9 = (5 / 6) ^ 9
  · rw [hn] at this
    norm_num at this
    exact this
  sorry

noncomputable def probability_at_least_one_wins (n : ℕ) : ℝ :=
  let single_win_prob := (5 / 6) ^ 9
  let union_prob := 10 * single_win_prob - 
                    45 * (5/6 * (4/6)^8) +
                    120 * (5/6 * 4/6 * (3/6)^7) - 
                    210 * (5/6 * 4/6 * 3/6 * (2/6)^6) + 
                    252 * (5/6 * 4/6 * 3/6 * 2/6 * (1/6)^5)
  in union_prob

theorem at_least_one_person_wins (n : ℕ) (hn : n = 10) : probability_at_least_one_wins n = 0.919 := by
  apply sorry

end petya_wins_prize_at_least_one_person_wins_l814_814766


namespace deepak_present_age_l814_814434

-- Define the variables R and D
variables (R D : ℕ)

-- The conditions:
-- 1. After 4 years, Rahul's age will be 32 years.
-- 2. The ratio between Rahul and Deepak's ages is 4:3.
def rahul_age_after_4 : Prop := R + 4 = 32
def age_ratio : Prop := R / D = 4 / 3

-- The statement we want to prove:
theorem deepak_present_age (h1 : rahul_age_after_4 R) (h2 : age_ratio R D) : D = 21 :=
by sorry

end deepak_present_age_l814_814434


namespace cubic_eq_solutions_l814_814487

theorem cubic_eq_solutions (x : ℝ) :
  x^3 - 4 * x = 0 ↔ x = 0 ∨ x = -2 ∨ x = 2 := by
  sorry

end cubic_eq_solutions_l814_814487


namespace abundant_numbers_less_than_50_number_of_abundant_numbers_less_than_50_l814_814311

def sum_of_proper_factors (n : Nat) : Nat :=
  (List.range n).filter (λ i => i > 0 ∧ n % i = 0).sum

def is_abundant (n : Nat) : Prop :=
  sum_of_proper_factors n > n

def abundant_numbers_below (n : Nat) : List Nat :=
  (List.range n).filter is_abundant

theorem abundant_numbers_less_than_50 : 
  abundant_numbers_below 50 = [12, 18, 20, 24, 30, 36, 40, 42, 48] := 
by sorry

theorem number_of_abundant_numbers_less_than_50 : 
  List.length (abundant_numbers_below 50) = 9 := 
by sorry

end abundant_numbers_less_than_50_number_of_abundant_numbers_less_than_50_l814_814311


namespace find_polynomial_l814_814603

noncomputable def p : ℝ → ℝ := sorry

theorem find_polynomial (p : ℝ → ℝ) :
  (p(-1) = 2) ∧ (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 1) → ∀ x : ℝ, p x = 2 := 
sorry

end find_polynomial_l814_814603


namespace number_of_valid_5digit_numbers_l814_814663

-- Define the conditions
def valid_digit (a b c d e : ℕ) : Prop := 
  a ∈ {1, 2, ..., 9} ∧ 
  b = a + c ∧ 
  d = c + e ∧ 
  b ∈ {0, 1, ..., 9} ∧ 
  d ∈ {0, 1, ..., 9} ∧ 
  c ∈ {0, 1, ..., 9} ∧ 
  e ∈ {0, 1, ..., 9}

theorem number_of_valid_5digit_numbers : 
  {n : ℕ // ∃ (a b c d e : ℕ), valid_digit a b c d e ∧ n = 330}.val = 330 :=
by
  sorry

end number_of_valid_5digit_numbers_l814_814663


namespace regular_n_solid_angle_has_symmetry_axis_l814_814034

noncomputable def regular_n_solid_angle {n : ℕ} (solid_angle : Type*) : Prop :=
  -- Definition to be provided depending on formalization of solid_angle
  sorry

theorem regular_n_solid_angle_has_symmetry_axis (solid_angle : Type*) (n : ℕ) 
  (h : regular_n_solid_angle solid_angle) :
  ∃ (axis : solid_angle → solid_angle) (θ : ℝ), θ = 2 * real.pi / n ∧ 
    ∀ (p : solid_angle), axis (rotate θ p) = p :=
  sorry

end regular_n_solid_angle_has_symmetry_axis_l814_814034


namespace delta_maximum_success_ratio_l814_814608

theorem delta_maximum_success_ratio (x y z w : ℕ) (h1 : 0 < x ∧ x * 5 < y * 3)
    (h2 : 0 < z ∧ z * 5 < w * 3) (h3 : y + w = 600) :
    (x + z) / 600 ≤ 359 / 600 :=
by
  sorry

end delta_maximum_success_ratio_l814_814608


namespace centroid_divides_medians_l814_814610

-- Definitions based on the problem conditions
structure Point := (x : ℝ) (y : ℝ)

def is_midpoint (M : Point) (A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_median (A B C : Point) (M : Point) : Prop :=
  is_midpoint M B C

def is_centroid (G A B C M N : Point) : Prop :=
  is_median A B C M ∧ is_median B A C N ∧ 
  G.x = (A.x + B.x + C.x) / 3 ∧ G.y = (A.y + B.y + C.y) / 3

-- Mathematically equivalent proof problem statement
theorem centroid_divides_medians (A B C G M N : Point) :
  is_midpoint M B C → 
  is_median A B C M → 
  is_centroid G A B C M N → 
  dist A G = 2 * dist G M :=
by sorry

end centroid_divides_medians_l814_814610


namespace valid_necklace_length_l814_814157

def bead_shape := {cube, pyramid}
def bead_color := {green, blue, red}

structure bead :=
(shape : bead_shape)
(color : bead_color)

def is_valid_necklace (necklace : list bead) : Prop :=
all_different_adjacent_shapes_and_colors necklace ∧
necklace.count_in_necklace bead (shape part shape) (color part color)

theorem valid_necklace_length (n : ℕ):
    (n = 8 ∨ n = 10) ↔ valid_necklace necklace :=
begin
  sorry -- proof of the theorem
end

end valid_necklace_length_l814_814157


namespace order_of_a_b_c_l814_814271

noncomputable def a : ℝ := real.exp (real.log 2 / 3)
noncomputable def b : ℝ := real.log (2 / 3) / real.log 3
noncomputable def c : ℝ := real.log 3 / real.log 2

theorem order_of_a_b_c : c > a ∧ a > b := by
  sorry

end order_of_a_b_c_l814_814271


namespace sum_of_cubes_l814_814782

theorem sum_of_cubes (x y z : ℝ) (h1 : x + y + z = 7) (h2 : xy + xz + yz = 9) (h3 : xyz = -18) :
  x^3 + y^3 + z^3 = 100 :=
by
  sorry

end sum_of_cubes_l814_814782


namespace part1_part2_l814_814490

open Set

-- Definitions from conditions in a)
def R : Set ℝ := univ
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Question part (1)
theorem part1 (a : ℝ) (h : a = 1) :
  (compl A) ∪ B a = {x | x ≤ -2 ∨ x > 1} :=
by 
  simp [h]
  sorry

-- Question part (2)
theorem part2 (a : ℝ) :
  A ⊆ B a → a ≤ -2 :=
by 
  sorry

end part1_part2_l814_814490


namespace find_f_l814_814206

theorem find_f (f : ℤ → ℤ) (h : ∀ n : ℤ, n^2 + 4 * (f n) = (f (f n))^2) :
  (∀ x : ℤ, f x = 1 + x) ∨
  (∃ a : ℤ, (∀ x ≤ a, f x = 1 - x) ∧ (∀ x > a, f x = 1 + x)) ∨
  (f 0 = 0 ∧ (∀ x < 0, f x = 1 - x) ∧ (∀ x > 0, f x = 1 + x)) :=
sorry

end find_f_l814_814206


namespace simplify_cubed_sum_l814_814403
-- Import the complete Mathlib

-- Define the problem in Lean
theorem simplify_cubed_sum :
  (∛(40^3 + 60^3 + 80^3 + 100^3) : ℝ) = 20 * (∛224 : ℝ) :=
by
  -- Prove the statement
  sorry

end simplify_cubed_sum_l814_814403


namespace valid_bead_count_l814_814161

structure Bead :=
  (shape : String)   -- "cube" or "pyramid"
  (color : String)   -- colors: "green", "blue", "red"

inductive Shape 
| Cube
| Pyramid 

inductive Color 
| Green
| Blue
| Red

def is_solution (n : ℕ) : Prop :=
  ∃ (b : fin n → Bead), 
    (∀ i, b (i + 1) % n).shape ≠ (b i).shape ∧ (b (i + 1) % n).color ≠ (b i).color) ∧
    set.of_list (list.of_fn b).to_finset = { Bead.mk Shape.Cube Color.Green, Bead.mk Shape.Cube Color.Blue, Bead.mk Shape.Pyramid Color.Red, Bead.mk Shape.Pyramid Color.Blue }

theorem valid_bead_count :
  ∀ n : ℕ, is_solution n → (n = 8 ∨ n = 10) :=
begin
  sorry
end

end valid_bead_count_l814_814161


namespace quadratic_inequality_solution_l814_814437

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 2 * x - 3 < 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end quadratic_inequality_solution_l814_814437


namespace scientific_notation_of_122254_l814_814391

theorem scientific_notation_of_122254 :
  122254 = 1.22254 * 10^5 :=
sorry

end scientific_notation_of_122254_l814_814391


namespace total_price_of_purchases_l814_814107

def price_of_refrigerator := 4275
def price_difference := 1490
def price_of_washing_machine := price_of_refrigerator - price_difference
def total_price := price_of_refrigerator + price_of_washing_machine

theorem total_price_of_purchases : total_price = 7060 :=
by
  rfl  -- This is just a placeholder; you need to solve the proof.

end total_price_of_purchases_l814_814107


namespace find_f_2023_div_2_l814_814744

noncomputable def f (x : ℝ) : ℝ := 
  if 1 < x ∧ x < 2 then log x / log 2 + 1 else sorry

theorem find_f_2023_div_2 (h_odd : ∀ x : ℝ, f (-x) = -f x)
                          (h_period : ∀ x : ℝ, f (x + 2) = -f x)
                          (h_interval : ∀ x : ℝ, 1 < x → x < 2 → f x = log x / log 2 + 1) :
  f (2023 / 2) = - log 3 / log 2 :=
begin
  sorry
end

end find_f_2023_div_2_l814_814744


namespace tangent_line_eq_correct_l814_814597

noncomputable def tangentLineEquation (x : ℝ) : ℝ := 2 * x + 1

theorem tangent_line_eq_correct (x : ℝ) :
  let y := λ x : ℝ, Real.exp (2 * x)
  ∃ (m b : ℝ), m = 2 ∧ b = 1 ∧ (∀ x, y x = Real.exp (2 * x)) ∧ (∀ x, y 0 = 1) ∧ tangentLineEquation x = 2 * x + 1 :=
by
  sorry

end tangent_line_eq_correct_l814_814597


namespace acute_triangle_cos_sin_condition_l814_814630

theorem acute_triangle_cos_sin_condition
  {A B C a b c : ℝ}
  (h_acute : A + B + C = π)  -- Acute means the angles sum up to π radians (180 degrees)
  (h_angles : 0 < A ∧ 0 < B ∧ 0 < C ∧ A < π/2 ∧ B < π/2 ∧ C < π/2)  -- All angles are less than π/2
  (h_sides : 0 < a ∧ 0 < b ∧ 0 < c)  -- Sides of the triangle are positive
  (h_cos_sin : (cos B) / b + (cos C) / c = (sin A) / (sqrt 3 * sin C)) :
  b = sqrt 3 :=
by sorry

end acute_triangle_cos_sin_condition_l814_814630


namespace frog_jump_coordinates_l814_814344

-- Define the starting point A
def A := (-1, 0)

-- Define the movement to the right
def move_right (point : ℝ × ℝ) (distance : ℝ) : ℝ × ℝ :=
  (point.1 + distance, point.2)

-- Define the movement upwards
def move_up (point : ℝ × ℝ) (distance : ℝ) : ℝ × ℝ :=
  (point.1, point.2 + distance)

-- Define the final point A' after the frog makes its jumps
def A' := move_up (move_right A 2) 2

-- The statement to prove
theorem frog_jump_coordinates :
  A' = (1, 2) :=
sorry

end frog_jump_coordinates_l814_814344


namespace find_point_C_l814_814984

structure Point where
  x : ℝ 
  y : ℝ 

def A : Point := { x := 0, y := 1 }

def vector_ab : Point := { x := -4, y := -3 }

def vector_bc : Point := { x := -7, y := -4 }

theorem find_point_C : 
  ∃ C : Point, 
  C.x = vector_ab.x + vector_bc.x ∧ 
  C.y = vector_ab.y + vector_bc.y :=
by 
  use { x := -11, y := -6 }
  split
  . rfl
  . rfl
-- Because we know the point C will be at the coordinates -11 and -6 based on the problem conditions.

end find_point_C_l814_814984


namespace planB_greater_than_planA_planB_greater_than_planC_l814_814883

variable (a b : ℝ)
variable (h : a > b) (hb : b > 0)

def planA_salary_increase := (1 + a / 100) * (1 + b / 100)
def planB_salary_increase := (1 + (a + b) / 200) * (1 + (a + b) / 200)
def planC_salary_increase := (1 + (Real.sqrt (a * b)) / 100) * (1 + (Real.sqrt (a * b)) / 100)

theorem planB_greater_than_planA : planB_salary_increase a b > planA_salary_increase a b := sorry

theorem planB_greater_than_planC : planB_salary_increase a b > planC_salary_increase a b := sorry

end planB_greater_than_planA_planB_greater_than_planC_l814_814883


namespace debby_candy_problem_l814_814226

theorem debby_candy_problem (D : ℕ) (sister_candy : ℕ) (eaten : ℕ) (remaining : ℕ) 
  (h1 : sister_candy = 42) (h2 : eaten = 35) (h3 : remaining = 39) :
  D + sister_candy - eaten = remaining ↔ D = 32 :=
by
  sorry

end debby_candy_problem_l814_814226


namespace multiple_of_r_capital_l814_814325

theorem multiple_of_r_capital (Cp Cq Cr profit : ℝ) (H1 : 4 * Cp = 6 * Cq) (H2 : 6 * Cq = Cr * 2.4) (H3 : (2.4 / (10 + 2.4)) * profit = 900) (total_profit : profit = 4650) : 
  ∃ k, k = 2.4 := 
by
  sorry

end multiple_of_r_capital_l814_814325


namespace most_stable_performance_l814_814485

theorem most_stable_performance :
  let S_甲 := 0.035
  let S_乙 := 0.015
  let S_丙 := 0.025
  let S_丁 := 0.027
  S_乙 < S_甲 ∧ S_乙 < S_丙 ∧ S_乙 < S_丁 :=
by
  let S_甲 := 0.035
  let S_乙 := 0.015
  let S_丙 := 0.025
  let S_丁 := 0.027
  sorry

end most_stable_performance_l814_814485


namespace max_height_of_tetrahedron_inscribed_in_sphere_l814_814979

theorem max_height_of_tetrahedron_inscribed_in_sphere :
  ∀ (O : Type) (d : ℝ) (Vmax : ℝ), 
  d = 12 ∧ 
  (∃ a h, 0 < h ∧ h < 8 ∧ a^2 = 24*h - 2*h^2 ∧ Vmax = (2/3)*(12*h^2 - h^3)) → 
  ∃ hmax, hmax = 8 :=
by 
  intros O d Vmax h a
  assume (d_eq : d = 12)
  assume (a_h_eq : ∃ a h, 0 < h ∧ h < 8 ∧ a^2 = 24*h - 2*h^2)
  assume (Vmax_eq : Vmax = (2/3)*(12*h^2 - h^3))
  sorry

end max_height_of_tetrahedron_inscribed_in_sphere_l814_814979


namespace width_of_room_l814_814797

-- Define the givens
def length_of_room : ℝ := 5.5
def total_cost : ℝ := 20625
def rate_per_sq_meter : ℝ := 1000

-- Define the required proof statement
theorem width_of_room : (total_cost / rate_per_sq_meter) / length_of_room = 3.75 :=
by
  sorry

end width_of_room_l814_814797


namespace purple_cells_possible_counts_l814_814496

theorem purple_cells_possible_counts (w h : ℕ) (p : ℕ → ℕ → Prop) (is_purple : ℕ → Prop) :
  w = 2022 ∧ h = 2022 ∧ 
  (∀ n₁ n₂, p n₁ n₂ = n₁ * n₂) ∧ 
  (∀ n, is_purple n ↔ (n = 2022 * 2020 ∨ n = 2021 * 2020)) →
  is_purple (p w h) :=
sorry

end purple_cells_possible_counts_l814_814496


namespace find_number_l814_814962

theorem find_number (x : ℕ) (h : x = 4) : x + 1 = 5 :=
by
  sorry

end find_number_l814_814962


namespace find_polynomials_l814_814229

noncomputable def omega (n : ℕ) : ℕ :=
  if n = 0 then 0 else (Nat.primeFactors n).to_finset.card

theorem find_polynomials (P : ℤ → ℤ) : (∃ (m : ℕ), ∀ (n : ℕ), omega n > 2023 ^ 2023 → omega n ≥ omega (P n) ∧ P n > 0) →
  ∃ k : ℕ, P = (fun x => x^k) ∨ ∃ c : ℕ, P = (fun _ => c) ∧ omega c ≤ 2023 ^ 2023 + 1 :=
by
  intros h
  sorry

end find_polynomials_l814_814229


namespace g_inv_f_five_l814_814044

-- Declare the existence of functions f and g and their inverses
variables (f g : ℝ → ℝ)

-- Given condition from the problem
axiom inv_cond : ∀ x, f⁻¹ (g x) = 4 * x - 1

-- Define the specific problem to solve
theorem g_inv_f_five : g⁻¹ (f 5) = 3 / 2 :=
by
  sorry

end g_inv_f_five_l814_814044


namespace perpendicular_vector_condition_l814_814999

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

noncomputable def dot_product : ℝ := ⟪a, b⟫

theorem perpendicular_vector_condition 
  (h_unit_a : ∥a∥ = 1) 
  (h_unit_b : ∥b∥ = 1) 
  (h_angle : real.angle_of _ a b = real.pi / 3) :
  ⟪2 • a - b, b⟫ = 0 := 
begin
  sorry
end

end perpendicular_vector_condition_l814_814999


namespace reflection_of_P_across_y_axis_l814_814423

-- Define the initial point P as a tuple
def P : ℝ × ℝ := (1, -2)

-- Define the reflection across the y-axis function
def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

-- State the theorem that we want to prove
theorem reflection_of_P_across_y_axis :
  reflect_y_axis P = (-1, -2) :=
by
  -- placeholder for the proof steps
  sorry

end reflection_of_P_across_y_axis_l814_814423


namespace kids_wearing_socks_l814_814451

-- Define the constants as given in the conditions
def total_kids : ℕ := 22
def kids_wearing_shoes : ℕ := 8
def kids_wearing_both : ℕ := 6
def kids_barefoot : ℕ := 8

-- Define the proposition to be proved
theorem kids_wearing_socks : total_kids - kids_barefoot - (kids_wearing_shoes - kids_wearing_both) = 12 :=
by 
  calc
    total_kids - kids_barefoot - (kids_wearing_shoes - kids_wearing_both)
        = 22 - 8 - (8 - 6) : by sorry
    ... = 12               : by sorry

end kids_wearing_socks_l814_814451


namespace volume_relation_l814_814814

noncomputable def A (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3
noncomputable def M (r : ℝ) : ℝ := 2 * Real.pi * r^3
noncomputable def C (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_relation (r : ℝ) : A r - M r + C r = 0 :=
by
  sorry

end volume_relation_l814_814814


namespace min_sum_xy_l814_814265

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (hcond : ↑(1 / x) + ↑(1 / y) = 1 / 15) : x + y = 64 :=
sorry

end min_sum_xy_l814_814265


namespace sum_of_midpoint_coordinates_l814_814055

theorem sum_of_midpoint_coordinates :
  let (x1, y1) := (3 : ℝ, 4 : ℝ)
  let (x2, y2) := (10 : ℝ, 20 : ℝ)
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  midpoint.1 + midpoint.2 = 18.5 :=
by
  sorry

end sum_of_midpoint_coordinates_l814_814055


namespace equilateral_triangle_colored_area_l814_814538

theorem equilateral_triangle_colored_area {A B C : Type*} [triangle A B C] 
  (h_equilateral : equilateral A B C) (h_area : area A B C = 210) : 
  exists S, 
  (inside_colored_area S A B C) 
  ∧ (area S = 70) :=
sorry

end equilateral_triangle_colored_area_l814_814538


namespace exists_abcd_for_n_gt_one_l814_814396

theorem exists_abcd_for_n_gt_one (n : Nat) (h : n > 1) :
  ∃ a b c d : Nat, a + b = 4 * n ∧ c + d = 4 * n ∧ a * b - c * d = 4 * n := 
by
  sorry

end exists_abcd_for_n_gt_one_l814_814396


namespace num_outfits_l814_814773

-- Define the number of trousers, shirts, and jackets available
def num_trousers : Nat := 5
def num_shirts : Nat := 6
def num_jackets : Nat := 4

-- Define the main theorem
theorem num_outfits (t : Nat) (s : Nat) (j : Nat) (ht : t = num_trousers) (hs : s = num_shirts) (hj : j = num_jackets) :
  t * s * j = 120 :=
by 
  rw [ht, hs, hj]
  exact rfl

end num_outfits_l814_814773


namespace max_marked_points_l814_814897

noncomputable def polynomial := Polynomial ℤ

theorem max_marked_points (P : polynomial) (h_degree : P.degree = 20) :
  ∃ S : Finset (ℤ × ℤ), (∀ x y, (x, y) ∈ S → y = P.eval x ∧ 0 ≤ y ∧ y ≤ 10) ∧ S.card ≤ 20 :=
sorry

end max_marked_points_l814_814897


namespace distance_between_parallel_lines_l814_814652

theorem distance_between_parallel_lines :
  ∀ (x y : ℝ) (a : ℝ),
      (l1 : x + a * y = 1) → (l2 : a * x + y = 1) → 
      (a = -1) → 
      (distance l1 l2 = √2) :=
by
  -- Assumptions and conditions 
  sorry

end distance_between_parallel_lines_l814_814652


namespace find_speed_of_man_in_still_water_l814_814139

def speed_of_man_in_still_water (t1 t2 d1 d2: ℝ) (v_m v_s: ℝ) : Prop :=
  d1 / t1 = v_m + v_s ∧ d2 / t2 = v_m - v_s

theorem find_speed_of_man_in_still_water :
  ∃ v_m : ℝ, ∃ v_s : ℝ, speed_of_man_in_still_water 2 2 16 10 v_m v_s ∧ v_m = 6.5 :=
by
  sorry

end find_speed_of_man_in_still_water_l814_814139


namespace count_abundant_numbers_less_than_50_l814_814318

open_locale classical

def is_abundant (n : ℕ) : Prop :=
  ∑ m in finset.filter (λ m, m ∣ n ∧ m < n) (finset.range n), m > n

theorem count_abundant_numbers_less_than_50 :
  (finset.filter (λ n, is_abundant n) (finset.range 50)).card = 9 :=
by {
  sorry
}

end count_abundant_numbers_less_than_50_l814_814318


namespace trains_cross_time_l814_814494

/-
  Problem: Prove that the time it takes for two trains to cross each other is 9 seconds given the following conditions:
  - The first train is 250 m long.
  - The speed of the first train is 120 km/hr.
  - The second train runs in the opposite direction.
  - The speed of the second train is 80 km/hr.
  - The length of the second train is 250.04 m.
-/

-- Definitions based on conditions:
def length_train1 := 250 -- in meters
def speed_train1 := 120 * 1000 / 3600 -- converted to m/s

def length_train2 := 250.04 -- in meters
def speed_train2 := 80 * 1000 / 3600 -- converted to m/s

def relative_speed := speed_train1 + speed_train2
def total_distance := length_train1 + length_train2

-- Theorem to prove:
theorem trains_cross_time : 
  total_distance / relative_speed = 9 := by
  -- The solution was provided in the initial steps, but as per instruction, the proof will be skipped.
  sorry

end trains_cross_time_l814_814494


namespace find_speed_of_P_l814_814087

noncomputable def walking_speeds (v_P v_Q : ℝ) : Prop :=
  let distance_XY := 90
  let distance_meet_from_Y := 15
  let distance_P := distance_XY - distance_meet_from_Y
  let distance_Q := distance_XY + distance_meet_from_Y
  (v_Q = v_P + 3) ∧
  (distance_P / v_P = distance_Q / v_Q)

theorem find_speed_of_P : ∃ v_P : ℝ, walking_speeds v_P (v_P + 3) ∧ v_P = 7.5 :=
by
  sorry

end find_speed_of_P_l814_814087


namespace abundant_numbers_less_than_50_eq_9_l814_814303

def proper_factors (n : ℕ) : List ℕ :=
  List.filter (λ m, m < n ∧ n % m = 0) (List.range n)

def sum_proper_factors (n : ℕ) : ℕ :=
  (proper_factors n).sum

def is_abundant (n : ℕ) : Prop :=
  sum_proper_factors n > n

def abundant_numbers_count_below (m : ℕ) : ℕ :=
  (List.range m).filter is_abundant |>.length

theorem abundant_numbers_less_than_50_eq_9 :
  abundant_numbers_count_below 50 = 9 :=
by
  sorry

end abundant_numbers_less_than_50_eq_9_l814_814303


namespace complex_magnitude_l814_814385

theorem complex_magnitude (z : ℂ) (i : ℂ) (h1 : i * i = -1) (h2 : (z + i) * (1 + i) = 1 - i) : |z| = 2 := 
by
  sorry

end complex_magnitude_l814_814385


namespace m_value_l814_814655

-- Vector definitions
def vector_a : Prod ℤ ℤ := (-1, 1)
def vector_b (m : ℤ) : Prod ℤ ℤ := (3, m)
def vector_c (m : ℤ) : Prod ℤ ℤ := (2, m + 1)

-- Condition for parallel vectors
def is_parallel (a b : Prod ℤ ℤ) : Prop :=
  ∃ k : ℚ, a = (k * b.1, k * b.2)

-- Main theorem
theorem m_value : ∀ (m : ℤ), is_parallel vector_a (vector_c m) → m = -3 := by
  intros m h
  -- Proof goes here
  sorry

end m_value_l814_814655


namespace circle_radius_l814_814829

theorem circle_radius :
  ∃ (r : ℝ) (x : ℝ), (x, 0) ∈ set_of (λ c : ℝ × ℝ, dist c (0, 5) = r ∧ dist c (2, 3) = r) ∧ r = real.sqrt 34 :=
by
  sorry

end circle_radius_l814_814829


namespace fin_solutions_l814_814380

theorem fin_solutions (u : ℕ) (hu : u > 0) :
  ∃ N : ℕ, ∀ n a b : ℕ, n > N → ¬ (n! = u^a - u^b) :=
sorry

end fin_solutions_l814_814380


namespace percentage_of_blue_chips_is_10_l814_814078

-- Definitions of the conditions
variable (T : ℕ) -- Total number of chips
variable (number_of_blue_chips : ℕ := 3)
variable (percentage_white_chips : ℝ := 0.5)
variable (percentage_non_white_chips : ℝ := 1 - percentage_white_chips)
variable (number_of_green_chips : ℕ := 12)

-- Proof statement
theorem percentage_of_blue_chips_is_10 :
  (number_of_blue_chips.toReal / T) * 100 = 10 :=
by
  -- Given conditions
  have h1 : T - percentage_white_chips * T = number_of_blue_chips + number_of_green_chips, from sorry,
  have h2 : T = 30, from sorry,
  have h3 : number_of_blue_chips = 3, from rfl,
  sorry

end percentage_of_blue_chips_is_10_l814_814078


namespace alpha_is_five_l814_814582

def distinct_non_zero_digits (s : set ℕ) : Prop :=
s = {1, 2, 3, 4, 5, 6, 7, 8, 9}

def correct_arithmetic_expression (grid : list (list ℕ)) : Prop :=
-- Placeholder for the condition that each row and column form correct arithmetic expressions.
sorry

theorem alpha_is_five (grid : list (list ℕ)) (s : set ℕ) : 
  distinct_non_zero_digits s → 
  correct_arithmetic_expression grid → 
  α(grid) = 5 :=
by {
  -- original solution shows alpha(grid) = 5
  sorry
}

end alpha_is_five_l814_814582


namespace count_of_abundant_numbers_less_than_50_l814_814310

-- Define the proper divisors of a number
def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d => d < n ∧ n % d = 0)

-- Define what it means for a number to be abundant
def is_abundant (n : ℕ) : Prop :=
  (proper_divisors n).sum > n

-- Define the set of numbers less than 50 which are abundant
def abundant_numbers_less_than_50 : List ℕ :=
  (List.range 50).filter is_abundant

-- The theorem stating the number of abundant numbers less than 50
theorem count_of_abundant_numbers_less_than_50 : abundant_numbers_less_than_50.length = 9 := 
  sorry

end count_of_abundant_numbers_less_than_50_l814_814310


namespace predict_height_at_age_10_l814_814142

def regression_line := fun (x : ℝ) => 7.19 * x + 73.93

theorem predict_height_at_age_10 :
  regression_line 10 = 145.83 :=
by
  sorry

end predict_height_at_age_10_l814_814142


namespace fixed_point_f_l814_814795

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  a^(x - 1) - 1

-- Define the conditions for the parameter a
variables (a : ℝ) (h : a > 0) (h_ne_one : a ≠ 1)

-- The theorem we need to prove
theorem fixed_point_f : f a 1 = 0 :=
by
  -- Proof goes here
  sorry

end fixed_point_f_l814_814795


namespace participation_schemes_correct_l814_814175

noncomputable def total_participation_schemes : ℕ :=
  let choose := Nat.choose
  let perm := Nat.factorial
  in (choose 4 2) * (perm 3) - perm 3

theorem participation_schemes_correct :
  total_participation_schemes = 30 :=
by
  sorry

end participation_schemes_correct_l814_814175


namespace fraction_to_decimal_l814_814580

theorem fraction_to_decimal : (58 : ℚ) / 160 = 0.3625 := 
by sorry

end fraction_to_decimal_l814_814580


namespace find_solution_l814_814586

theorem find_solution (x y : ℕ) (h1 : y ∣ (x^2 + 1)) (h2 : x^2 ∣ (y^3 + 1)) : (x = 1 ∧ y = 1) :=
sorry

end find_solution_l814_814586


namespace odd_mul_abs_even_is_odd_l814_814654

def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def even_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = g x
def is_odd (h : ℝ → ℝ) : Prop := ∀ x : ℝ, h (-x) = -h x

theorem odd_mul_abs_even_is_odd
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : odd_function f) (hg : even_function g) :
  is_odd (λ x, f x * |g x|) :=
sorry

end odd_mul_abs_even_is_odd_l814_814654


namespace volume_of_pyramid_correct_l814_814526

-- Given conditions
def area_square_ABCD : ℝ := 256
def area_triangle_ABE : ℝ := 128
def area_triangle_CDE : ℝ := 96

-- We want to prove the volume of the pyramid is 1194.666...
theorem volume_of_pyramid_correct : 
  ∀ (s : ℝ) (height_ABE : ℝ) (height_CDE : ℝ) (h : ℝ), 
  s = real.sqrt area_square_ABCD ∧
  height_ABE = (2 * area_triangle_ABE) / s ∧
  height_CDE = (2 * area_triangle_CDE) / s ∧
  h = (height_ABE + height_CDE) / 2 →
  volume = (1/3) * area_square_ABCD * h →
  volume = 1194.6666 :=
by
  sorry

end volume_of_pyramid_correct_l814_814526


namespace circles_intersect_l814_814568

noncomputable def circleO1 : {x : ℝ × ℝ // (x.1)^2 + (x.2)^2 - 2 * x.1 = 0} :=
  sorry

noncomputable def circleO2 : {x : ℝ × ℝ // (x.1)^2 + (x.2)^2 - 6 * x.2 = 0} :=
  sorry

theorem circles_intersect : 
  (∃ P₁ P₂ ∈ circleO1, P₁ = P₂ ∧ dist P₁ P₂ ≤ 4 ∧ dist P₁ P₂ ≥ 2) ∨ 
  (∃ Q₁ Q₂ ∈ circleO2, Q₁ = Q₂ ∧ dist Q₁ Q₂ ≤ 4 ∧ dist Q₁ Q₂ ≥ 2) :=
  sorry

end circles_intersect_l814_814568


namespace triangle_inequality_l814_814769

theorem triangle_inequality (a b c s_a s_b s_c R : ℝ) 
  (h1 : is_triangle a b c) 
  (h2 : is_median a s_a) 
  (h3 : is_median b s_b) 
  (h4 : is_median c s_c) 
  (h5 : is_circumradius a b c R) :
  2 * R * (s_a + s_b + s_c) ≥ a^2 + b^2 + c^2 := sorry

end triangle_inequality_l814_814769


namespace hyperbola_and_line_properties_l814_814976

open Real

def hyperbola (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def asymptote1 (x y : ℝ) : Prop := y = sqrt 3 * x
def asymptote2 (x y : ℝ) : Prop := y = -sqrt 3 * x
def line (x y t : ℝ) : Prop := y = x + t

theorem hyperbola_and_line_properties :
  ∃ a b t : ℝ,
  a > 0 ∧ b > 0 ∧ a = 1 ∧ b^2 = 3 ∧
  (∀ x y, hyperbola x y a b ↔ (x^2 - y^2 / 3 = 1)) ∧
  (∀ x y, asymptote1 x y ↔ y = sqrt 3 * x) ∧
  (∀ x y, asymptote2 x y ↔ y = -sqrt 3 * x) ∧
  (∀ x y, (line x y t ↔ (y = x + sqrt 3) ∨ (y = x - sqrt 3))) := sorry

end hyperbola_and_line_properties_l814_814976


namespace maximum_value_of_f_l814_814599

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * Real.sqrt x

theorem maximum_value_of_f :
  ∃ x_max : ℝ, x_max > 0 ∧ (∀ x : ℝ, x > 0 → f x ≤ f x_max) ∧ f x_max = -2 :=
by
  sorry

end maximum_value_of_f_l814_814599


namespace algebraic_expression_value_l814_814968

noncomputable def algebraic_expression (a b : ℝ) : ℝ := a^2 * b + 2 * a^2 * b^2 + a * b^3

theorem algebraic_expression_value :
  ∀ (a b : ℝ), a + b = 3 → a * b = 2 → algebraic_expression a b = 18 :=
by
  intros a b h1 h2
  have h3 : (a^2 * b + 2 * a^2 * b^2 + a * b^3) = (a * b) * (a^2 + 2 * a * b + b^2), sorry
  -- Using h1 and h2 to substitute values and show the final result
  sorry

end algebraic_expression_value_l814_814968


namespace jacob_needs_to_catch_l814_814919

def initial_fish_jacob : ℕ := 8
def factor_alex_catches_more_than_jacob : ℕ := 7
def fish_lost_by_alex : ℕ := 23

theorem jacob_needs_to_catch : ∃ (n : ℕ), n = 26 :=
  let initial_fish_alex := factor_alex_catches_more_than_jacob * initial_fish_jacob
  let remaining_fish_alex := initial_fish_alex - fish_lost_by_alex
  let target_fish_jacob := remaining_fish_alex + 1
  let additional_fish_needed := target_fish_jacob - initial_fish_jacob
  begin
    use additional_fish_needed,
    calc additional_fish_needed
          = target_fish_jacob - initial_fish_jacob : rfl
      ... = (remaining_fish_alex + 1) - initial_fish_jacob : rfl
      ... = ((factor_alex_catches_more_than_jacob * initial_fish_jacob - fish_lost_by_alex) + 1) - initial_fish_jacob : rfl
      ... = (56 - 23) + 1 - 8 : by norm_num
      ... = 33 + 1 - 8 : rfl
      ... = 34 - 8 : rfl
      ... = 26 : rfl
  end.

end jacob_needs_to_catch_l814_814919


namespace tom_must_work_31_hours_l814_814084

-- Define the given costs and hourly wage
def item_cost1 : ℝ := 25.35
def item_cost2 : ℝ := 70.69
def item_cost3 : ℝ := 85.96
def hourly_wage : ℝ := 6.50
def percentage_for_other_purposes : ℝ := 0.10

-- Compute the total cost and the effective hourly wage after setting aside 10%
def total_cost : ℝ := item_cost1 + item_cost2 + item_cost3
def effective_hourly_wage : ℝ := hourly_wage * (1 - percentage_for_other_purposes)

-- Determine the required hours rounded to the nearest whole number
def required_hours : ℝ := total_cost / effective_hourly_wage
def rounded_hours : ℕ := round required_hours

-- Statement of the theorem
theorem tom_must_work_31_hours : rounded_hours = 31 := by
  sorry

end tom_must_work_31_hours_l814_814084


namespace modified_short_bingo_unique_arrangements_l814_814707

theorem modified_short_bingo_unique_arrangements : 
  ∃ (S : Finset ℕ), S.card = 15 ∧ (∃ (subS : Finset ℕ), subS ⊆ S ∧ subS.card = 7 ∧ (multiset.prod (subS.val.attach.map (λ i, (S.erase i).card))) = 1816214400) :=
by
  sorry

end modified_short_bingo_unique_arrangements_l814_814707


namespace sphere_radius_geometric_mean_l814_814770

-- Definitions from conditions
variable (r R ρ : ℝ)
variable (r_nonneg : 0 ≤ r)
variable (R_relation : R = 3 * r)
variable (ρ_relation : ρ = Real.sqrt 3 * r)

-- Problem statement
theorem sphere_radius_geometric_mean (tetrahedron : Prop):
  ρ * ρ = R * r :=
by
  sorry

end sphere_radius_geometric_mean_l814_814770


namespace magnitude_of_z_l814_814614

-- Definition of the given complex number z
def z : ℂ := (1 - complex.I) / (1 + complex.I) + 2 * complex.I

-- Statement to prove the magnitude of z
theorem magnitude_of_z : complex.abs z = 1 :=
by
  -- The proof will go here when implemented
  sorry

end magnitude_of_z_l814_814614


namespace determine_tuples_l814_814196

theorem determine_tuples (x y z t : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) :
  x * y * z = t! ∧ (x + 1) * (y + 1) * (z + 1) = (t + 1)! ↔ 
  (x, y, z, t) ∈ [
    (8, 3, 3, 1), (5, 4, 3, 1), (3, 2, 2, 2), (7, 6, 2, 1), (9, 5, 2, 1), (15, 4, 2, 1),
    (1, 1, 1, 7), (2, 1, 1, 5), (3, 2, 1, 3), (8, 3, 1, 2), (5, 4, 1, 2)
  ].to_set :=
by sorry

end determine_tuples_l814_814196


namespace length_of_AP_l814_814721

noncomputable def square_side_length : ℝ := 8
noncomputable def rect_width : ℝ := 12
noncomputable def rect_height : ℝ := 8

axiom AD_perpendicular_WX : true
axiom shaded_area_half_WXYZ : true

theorem length_of_AP (AP : ℝ) (shaded_area : ℝ)
  (h1 : shaded_area = (rect_width * rect_height) / 2)
  (h2 : shaded_area = (square_side_length - AP) * square_side_length)
  : AP = 2 := by
  sorry

end length_of_AP_l814_814721


namespace perpendicular_vector_l814_814993

variables {a b : ℝ}

/-- Given that the angle between unit vectors $\overrightarrow{a}$ and $\overrightarrow{b}$ 
is $60^{\circ}$, prove that $2\overrightarrow{a} - \overrightarrow{b}$ is perpendicular to 
$\overrightarrow{b}$. -/
theorem perpendicular_vector 
  (a b : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hab : a * b = 0.5) :
  (2 * a - b) * b = 0 :=
begin
  sorry,
end

end perpendicular_vector_l814_814993


namespace gcd_1978_2017_l814_814461

theorem gcd_1978_2017 : Int.gcd 1978 2017 = 1 :=
sorry

end gcd_1978_2017_l814_814461


namespace hyperbola_center_l814_814591

theorem hyperbola_center : ∃ (h k : ℝ), (h, k) = (3, 4) ∧ 
  (∀ x y : ℝ, 9*x^2 - 54*x - 16*y^2 + 128*y - 400 = 0 ↔ ((x - h)^2 / 25) - ((y - k)^2 / 14.0625) = 1) :=
begin
  use [3, 4],
  split,
  { simp, },
  { sorry }
end

end hyperbola_center_l814_814591


namespace log_comparison_l814_814611

-- Define the conditions
def a : ℝ := Real.log 3 / Real.log 4
def b : ℝ := Real.log 4 / Real.log 5
def c : ℝ := Real.exp (-0.2)

-- The proof statement
theorem log_comparison : b > c ∧ c > a :=
by {
  -- Proof content would go here, but we leave it with 'sorry' as instructed.
  sorry
}

end log_comparison_l814_814611


namespace valid_necklace_length_l814_814156

def bead_shape := {cube, pyramid}
def bead_color := {green, blue, red}

structure bead :=
(shape : bead_shape)
(color : bead_color)

def is_valid_necklace (necklace : list bead) : Prop :=
all_different_adjacent_shapes_and_colors necklace ∧
necklace.count_in_necklace bead (shape part shape) (color part color)

theorem valid_necklace_length (n : ℕ):
    (n = 8 ∨ n = 10) ↔ valid_necklace necklace :=
begin
  sorry -- proof of the theorem
end

end valid_necklace_length_l814_814156


namespace value_of_3W5_l814_814331

def W (a b : ℕ) : ℕ := b + 7 * a - a ^ 2

theorem value_of_3W5 : W 3 5 = 17 := by 
  sorry

end value_of_3W5_l814_814331


namespace problem_statement_l814_814783

-- Define the arithmetic sequence and the given condition
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- The condition that a_{a_{20}} = 17
def condition (a d : ℤ) : Prop := arithmetic_sequence a d (arithmetic_sequence a d 20).to_nat = 17

-- Prove that a_{2017} = 4013 given the condition
theorem problem_statement (a d : ℤ) (h : condition a d) : arithmetic_sequence a d 2017 = 4013 := 
sorry

end problem_statement_l814_814783


namespace max_quarters_l814_814041

theorem max_quarters (total_value : ℝ) (n_quarters n_nickels n_dimes : ℕ) 
  (h1 : n_nickels = n_quarters) 
  (h2 : n_dimes = 2 * n_quarters)
  (h3 : 0.25 * n_quarters + 0.05 * n_nickels + 0.10 * n_dimes = total_value)
  (h4 : total_value = 3.80) : 
  n_quarters = 7 := 
by
  sorry

end max_quarters_l814_814041


namespace equilateral_triangle_perimeter_l814_814714

theorem equilateral_triangle_perimeter
  (r : ℝ) (h1 : r = 4)
  (h2 : ∀ (A B C : ℝ), A = B ∧ B = C ∧ C = A)
  (h3 : ∀ x y z : ℝ, x = 8 ∧ y = 8 ∧ z = 8):
  let s := 3 * 16 in
  s = 48 := 
sorry

end equilateral_triangle_perimeter_l814_814714


namespace probability_symmetric_line_l814_814709

open_locale classical

theorem probability_symmetric_line 
  (grid_points : finset (ℕ × ℕ))
  (R : ℕ × ℕ)
  (S : ℕ × ℕ → Prop)
  (count_S : 99 = finset.card (grid_points \ {R}))
  (symmetric_lines : set (ℕ × ℕ))
  (count_symmetric_lines : 16 = symmetric_lines.card)
  (S_in_symmetric_lines : ∀ (s : ℕ × ℕ), s ∈ grid_points \ {R} → (s ∈ symmetric_lines ↔ S s)
  ) : (finset.card (grid_points \ {R})) = 99 → (symmetric_lines.card) = 16 → (∀ (s : ℕ × ℕ), S s → s ∈ symmetric_lines) → 
  ( ∀ (s : ℕ × ℕ), s ∈ grid_points \ {R} → S s → s ∈ symmetric_lines ) → 
  (count_S : 99 = finset.card (grid_points \ {R})) → 
  (count_symmetric_lines : 16 = symmetric_lines.card) →
  ∃ P : ℚ, P = 16 / 99 := 
by {
  sorry
}

end probability_symmetric_line_l814_814709


namespace maximize_area_l814_814938

-- Define the variables and constants
variables {x y p : ℝ}

-- Define the conditions
def perimeter (x y p : ℝ) := (2 * x + 2 * y = p)
def area (x y : ℝ) := x * y

-- The theorem statement with conditions
theorem maximize_area (h : perimeter x y p) : x = y → x = p / 4 :=
by
  sorry

end maximize_area_l814_814938


namespace option_A_option_C_option_D_l814_814284

def circle_equation (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 4 * y + a = 0

def circle_center : ℝ × ℝ := (1, -2)

def circle_radius (a : ℝ) : ℝ := Real.sqrt (5 - a)

theorem option_A (a : ℝ) : circle_radius a = 1 ↔ a = 4 := by sorry

theorem option_C (a : ℝ) : a = 0 ↔ circle_radius a = Real.sqrt 5 := by sorry

theorem option_D (a : ℝ) : a < 5 ↔ (∃ r : ℝ, r > 0 ∧ circle_radius a = r) := by sorry

end option_A_option_C_option_D_l814_814284


namespace john_candies_on_fourth_day_l814_814364

theorem john_candies_on_fourth_day (c : ℕ) (h1 : 5 * c + 80 = 150) : c + 24 = 38 :=
by 
  -- Placeholder for proof
  sorry

end john_candies_on_fourth_day_l814_814364


namespace total_spent_is_64_l814_814093

def deck_price : ℕ := 8
def victors_decks : ℕ := 6
def friends_decks : ℕ := 2

def victors_spending : ℕ := victors_decks * deck_price
def friends_spending : ℕ := friends_decks * deck_price
def total_spending : ℕ := victors_spending + friends_spending

theorem total_spent_is_64 : total_spending = 64 := by
  sorry

end total_spent_is_64_l814_814093


namespace f_odd_f_increasing_on_1_inf_max_min_values_on_interval_l814_814642

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + a) / x

-- Given condition
def a_constant : ℝ := 1

-- Lean statement to prove the function is odd
theorem f_odd : ∀ x : ℝ, f (-x) a_constant = -f x a_constant := sorry

-- Lean statement to prove the function is increasing on (1, +∞)
theorem f_increasing_on_1_inf : ∀ x : ℝ, 1 < x → f' x a_constant > 0 := sorry

-- Calculate the value of f(2) and f(5)
def f_at_2 : ℝ := f 2 a_constant
def f_at_5 : ℝ := f 5 a_constant

-- Lean statement to declare the max and min values on [2,5]
theorem max_min_values_on_interval : 
  (f_at_2, f_at_5) = (5 / 2, 26 / 5) := sorry

end f_odd_f_increasing_on_1_inf_max_min_values_on_interval_l814_814642


namespace arithmetic_sequence_problems_l814_814618

-- Define conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d, d ≠ 0 ∧ ∃ a₁, ∀ n, a n = a₁ + d * (n - 1)

def general_term_formula : Prop :=
  ∀ n, (∃ (d : ℤ) (a : ℕ → ℤ), is_arithmetic_sequence a ∧ a 1 = 2 ∧ 
      (a 1, a 2, a 5 are_in_geometric_sequence) ∧
      (∀ n, a n = 4 * n - 2))

def smallest_positive_n : Prop :=
  ∃ n, n > 0 ∧ ∀ m, m > 0 → (2 * m^2 > 60 * m + 800 ↔ m ≥ n)

-- Define the problem
theorem arithmetic_sequence_problems : 
  general_term_formula ∧ smallest_positive_n :=
by
  -- Provide the proof here if required
  sorry

end arithmetic_sequence_problems_l814_814618


namespace min_value_expr_l814_814213

theorem min_value_expr : ∀ x : ℝ, (x^2 + 8) / Real.sqrt (x^2 + 4) ≥ 4 ∧ ((x = 0) → (x^2 + 8)/Real.sqrt (x^2 + 4) = 4) := by
  intro x
  split
  sorry
  sorry

end min_value_expr_l814_814213


namespace one_by_k_tileable_iff_even_two_by_n_not_tileable_iff_not_div_by_four_l814_814775

noncomputable def is_tileable (m n : ℕ) : Prop := 
  ∃ (tiles : list (ℕ × ℕ × ℕ × ℕ)), 
    ∀ t ∈ tiles, (t.1 = 1 ∧ t.3 = 1) ∧ (t.2 = 2 ∧ t.4 = 1) 
    ∧ (∀ x1 y1 x2 y2, (x1, y1, x2, y2) ∈ tiles → (x1 < m ∧ y1 < n ∧ x2 < m ∧ y2 < n))
    ∧ (∀ x y, (∃ t ∈ tiles, (x, y) = (t.1, t.2) ∨ (x, y) = (t.3, t.4))) 
    ∧ (∀ t1 t2 ∈ tiles, t1 ≠ t2 → (t1.1 ≠ t2.1 ∨ t1.2 ≠ t2.2) ∧ (t1.3 ≠ t2.3 ∨ t1.4 ≠ t2.4))

theorem one_by_k_tileable_iff_even (k : ℕ) : is_tileable 1 k ↔ (k % 2 = 0) :=
  by sorry

theorem two_by_n_not_tileable_iff_not_div_by_four (n : ℕ) (h : ¬(4 ∣ n)) : ¬is_tileable 2 n :=
  by sorry

end one_by_k_tileable_iff_even_two_by_n_not_tileable_iff_not_div_by_four_l814_814775


namespace minimize_expression_l814_814438

theorem minimize_expression (n : ℕ) (h : 0 < n) : 
  (n = 10) ↔ (∀ m : ℕ, 0 < m → ((n / 2) + (50 / n) ≤ (m / 2) + (50 / m))) :=
sorry

end minimize_expression_l814_814438


namespace last_digit_of_large_exponents_l814_814796

theorem last_digit_of_large_exponents :
  let a := 3^1991 in
  let b := 1991^3 in
  ((a + b) % 10) = 8 :=
by
  sorry

end last_digit_of_large_exponents_l814_814796


namespace valid_number_of_beads_l814_814170

def Shape := ℕ -- simple definition for the shapes: cubes, pyramids
def Color := ℕ -- simple definition for the colors: green, blue, red

-- Definitions for our specific beads
def Kg : Shape × Color := (1, 1) -- green cube
def Kb : Shape × Color := (1, 2) -- blue cube
def Pr : Shape × Color := (2, 3) -- red pyramid
def Pb : Shape × Color := (2, 2) -- blue pyramid

-- Conditions given in the problem
def is_valid_necklace (necklace : List (Shape × Color)) : Prop :=
  ∀ i, (necklace.nth i ≠ none) →
      -- different shapes and different colors condition
      let ⟨(s1, c1)⟩ := necklace.get (i % necklace.length);
      let ⟨(s2, c2)⟩ := necklace.get ((i + 1) % necklace.length);
      s1 ≠ s2 ∧ c1 ≠ c2

-- Prove that the necklace must have a valid even count of beads and specifically 8 or 10.
theorem valid_number_of_beads : 
  ∀ (necklace : List (Shape × Color)), 
  is_valid_necklace necklace →
  (necklace.length = 8 ∨ necklace.length = 10) :=
by
  intros necklace h
  sorry

end valid_number_of_beads_l814_814170


namespace valid_necklace_length_l814_814158

def bead_shape := {cube, pyramid}
def bead_color := {green, blue, red}

structure bead :=
(shape : bead_shape)
(color : bead_color)

def is_valid_necklace (necklace : list bead) : Prop :=
all_different_adjacent_shapes_and_colors necklace ∧
necklace.count_in_necklace bead (shape part shape) (color part color)

theorem valid_necklace_length (n : ℕ):
    (n = 8 ∨ n = 10) ↔ valid_necklace necklace :=
begin
  sorry -- proof of the theorem
end

end valid_necklace_length_l814_814158


namespace area_of_triangle_AEP_l814_814130

noncomputable def unitSquare : set (ℝ × ℝ) := 
  {p | (p.1 = 0 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) ∨ 
       (p.1 = 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) ∨ 
       (p.2 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ 1) ∨ 
       (p.2 = 1 ∧ 0 ≤ p.1 ∧ p.1 ≤ 1)}

def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (1, 0)
def D : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (0, 1/4)
def E : ℝ × ℝ := (1/3, 1/3)

theorem area_of_triangle_AEP : 
  (unitSquare {A, B, C, D}) → 
  (P.2 = 1/4) → 
  (E = (1/3, 1/3) ∧ (E ∈ line_through B P ∧ E ∈ line_through A C)) → 
  (area_triangle A E P = 1/8) := 
by
  sorry

end area_of_triangle_AEP_l814_814130


namespace ellipse_properties_l814_814278

theorem ellipse_properties :
  (∃ a b : ℝ,
    a > b ∧ b > 0 ∧
    (∀ x y : ℝ, (x = 2 ∧ y = 0) → (x / a) ^ 2 + (y / b) ^ 2 = 1) ∧
    (a = 2 ∧ b = √3 ∧
     (∀ x y : ℝ, (1 / 4) * x^2 + (1 / 3) * y^2 = 1) ∧
     ∃ x1 x2 : ℝ, 
     (x = 1 ∧ y = 0) → (7 * x^2 - 8 * x - 8 = 0) ∧ 
     x1 + x2 = 8/7 ∧ x1 * x2 = -8/7 ∧
     (x' = 4 * k^2 / (4 * k^2 + 3) ) ∧ (y' = k * (x' - 1) = -3 * k / (4 * k^2 + 3)) ∧
     (y0 = y' /\ -√3/12 ≤ y0 ∧ y0 ≤ √3/12))) :=
   sorry

end ellipse_properties_l814_814278


namespace find_value_of_a_l814_814628

noncomputable def z1 : ℂ := 3 - 2 * complex.I
noncomputable def z2 (a : ℝ) : ℂ := 1 + a * complex.I

theorem find_value_of_a (a : ℝ) (h : z1 * z2 a ∈ ℝ) : a = 2 / 3 := sorry

end find_value_of_a_l814_814628


namespace elliot_book_pages_l814_814077

theorem elliot_book_pages : 
  ∀ (initial_pages read_per_day days_in_week remaining_pages total_pages: ℕ), 
    initial_pages = 149 → 
    read_per_day = 20 → 
    days_in_week = 7 → 
    remaining_pages = 92 → 
    total_pages = initial_pages + (read_per_day * days_in_week) + remaining_pages → 
    total_pages = 381 :=
by
  intros initial_pages read_per_day days_in_week remaining_pages total_pages
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  simp at h5
  assumption

end elliot_book_pages_l814_814077


namespace smallest_sum_l814_814255

theorem smallest_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) 
  (h_fraction : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) : x + y = 64 :=
sorry

end smallest_sum_l814_814255


namespace blocks_eaten_correct_l814_814042

def initial_blocks : ℕ := 55
def remaining_blocks : ℕ := 26

-- How many blocks were eaten by the hippopotamus?
def blocks_eaten_by_hippopotamus : ℕ := initial_blocks - remaining_blocks

theorem blocks_eaten_correct :
  blocks_eaten_by_hippopotamus = 29 := by
  sorry

end blocks_eaten_correct_l814_814042


namespace pyramid_section_through_points_l814_814933

noncomputable section

-- Define the points and pyramid
variables (A B C D K L M : Type) 
           [point A] [point B] [point C] [point D]
           [on_face K A B D] [on_face L B C D] [on_face M A C D]

-- Define the proof statement
theorem pyramid_section_through_points : 
  ∃ (P Q : points_in_space),
  (line_through P Q) ⊆ plane_intersection (plane_through K L M) (pyramid_faces_intersection A B C D) := 
sorry

end pyramid_section_through_points_l814_814933


namespace sum_of_extremes_90_l814_814810

noncomputable def find_sum_of_extremes : Prop :=
  ∃ (nums : Fin 1928 → ℕ), (∑ i, nums i = 2016) ∧ (∏ i, nums i = 1001) ∧ ((Finset.max' (Finset.univ.image nums) Finset.univ_nonempty) + (Finset.min' (Finset.univ.image nums) Finset.univ_nonempty) = 90)

theorem sum_of_extremes_90 : find_sum_of_extremes :=
sorry

end sum_of_extremes_90_l814_814810


namespace part1_part2_part3_l814_814756

-- Given function f is defined as an odd function.
def f (a k : ℝ) (x : ℝ) : ℝ := a^x - (k - 1) * a^(-x)

-- 1. Prove k = 2 given that f(x) is odd and a > 0, a ≠ 1.
theorem part1 (a : ℝ) (h_pos : 0 < a) (h_ne1 : a ≠ 1) (hf_odd : ∀ x : ℝ, f a k x = -f a k (-x)) : k = 2 := by
  sorry
  
-- 2. Prove the range of t is (-3, 5) given f(x² + tx) + f(4 - x) < 0 and a ∈ (0, 1).
theorem part2 (a : ℝ) (h_bound_a : 0 < a ∧ a < 1) (ht : ∀ x : ℝ, f a 2 (x^2 + t*x) < -f a 2 (4 - x)) : -3 < t ∧ t < 5 := by
  sorry

-- 3. Prove m = 2 given f(1) = 3/2 and g(x) has minimum value -2 on [1, +∞).
def g (a : ℝ) (m : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2*m*(f a 2 x)

theorem part3 (a : ℝ) (h_f1 : f a 2 1 = 3/2) (hmin : ∀ x ∈ set.Ici 1, g a m x ≥ -2) : m = 2 := by
  sorry

end part1_part2_part3_l814_814756


namespace train_speed_proof_l814_814132

variable (L1 : ℝ) (V1_kmph : ℝ) (L2 : ℝ) (t : ℝ) (V2_kmph : ℝ)
variable (L1_val : L1 = 300)
variable (V1_kmph_val : V1_kmph = 120)
variable (L2_val : L2 = 200.04)
variable (t_val : t = 9)
variable (V2_kmph_val : V2_kmph = 80)

noncomputable def V1_mps := V1_kmph * 1000 / 3600

noncomputable def total_distance := L1 + L2

noncomputable def relative_speed_mps := total_distance / t

noncomputable def V2_mps := relative_speed_mps - V1_mps

noncomputable def V2_kmph_calc := V2_mps * 3600 / 1000

theorem train_speed_proof :
  L1 = 300 →
  V1_kmph = 120 →
  L2 = 200.04 →
  t = 9 →
  V2_kmph_calc = 80 :=
by
  intros L1_val V1_kmph_val L2_val t_val
  simp [L1_val, V1_kmph_val, L2_val, t_val, V1_mps, total_distance, relative_speed_mps, V2_mps, V2_kmph_calc]
  sorry

end train_speed_proof_l814_814132


namespace find_article_cost_l814_814115

noncomputable def original_cost_price (C S : ℝ) :=
  (S = 1.25 * C) ∧
  (S - 6.30 = 1.04 * C)

theorem find_article_cost (C S : ℝ) (h : original_cost_price C S) : C = 30 :=
by sorry

end find_article_cost_l814_814115


namespace perpendicular_vector_condition_l814_814997

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

noncomputable def dot_product : ℝ := ⟪a, b⟫

theorem perpendicular_vector_condition 
  (h_unit_a : ∥a∥ = 1) 
  (h_unit_b : ∥b∥ = 1) 
  (h_angle : real.angle_of _ a b = real.pi / 3) :
  ⟪2 • a - b, b⟫ = 0 := 
begin
  sorry
end

end perpendicular_vector_condition_l814_814997


namespace seeds_germinated_percentage_l814_814228

theorem seeds_germinated_percentage 
  (n1 n2 : ℕ) 
  (p1 p2 : ℝ) 
  (h1 : n1 = 300)
  (h2 : n2 = 200)
  (h3 : p1 = 0.15)
  (h4 : p2 = 0.35) : 
  ( ( p1 * n1 + p2 * n2 ) / ( n1 + n2 ) ) * 100 = 23 :=
by
  -- Mathematical proof goes here.
  sorry

end seeds_germinated_percentage_l814_814228


namespace man_walking_speed_l814_814523

theorem man_walking_speed (length_of_bridge : ℝ) (time_to_cross : ℝ) 
  (h1 : length_of_bridge = 1250) (h2 : time_to_cross = 15) : 
  (length_of_bridge / time_to_cross) * (60 / 1000) = 5 := 
sorry

end man_walking_speed_l814_814523


namespace evaluate_expression_l814_814570

theorem evaluate_expression (x : ℝ) (h : x < 2) :
  sqrt ((x - 2) / (1 - (x - 3) / (x - 2))) = (2 - x) / sqrt 3 :=
by
  sorry

end evaluate_expression_l814_814570


namespace percent_answered_second_correctly_l814_814326

theorem percent_answered_second_correctly
  (p1_percentage : ℕ)
  (p_neither_percentage : ℕ)
  (p_both_percentage : ℕ)
  (p_first_correctly : p1_percentage = 85)
  (p_neither_correctly : p_neither_percentage = 5)
  (p_both_correctly : p_both_percentage = 55) :
  let p_second_percentage :=
    100 - p_neither_percentage - (p1_percentage - p_both_percentage) - p_both_percentage
  in
  p_second_percentage = 65 := 
by
  let p_only_first := p1_percentage - p_both_percentage
  let p_at_least_one := 100 - p_neither_percentage
  let p_only_second := p_at_least_one - p_only_first - p_both_percentage
  let p_second_percentage := p_only_second + p_both_percentage
  
  have h1 : p_only_first = p1_percentage - p_both_percentage := rfl
  have h2 : p_at_least_one = 100 - p_neither_percentage := rfl
  have h3 : p_only_second = p_at_least_one - p_only_first - p_both_percentage := rfl
  have h4 : p_second_percentage = p_only_second + p_both_percentage := rfl
  
  have h5 : p_only_first = 85 - 55 := by rw [p_first_correctly, p_both_correctly]
  have h6 : p_at_least_one = 95 := by rw p_neither_correctly
  have h7 : p_only_second = 95 - 30 - 55 := by rw [h6, h5, p_both_correctly]
  have h8 : p_second_percentage = 10 + 55 := by rw [←h7]
  
  exact eq.trans h8 (eq.refl 65)

end percent_answered_second_correctly_l814_814326


namespace largest_number_digits_sum_14_l814_814835

noncomputable def largest_number_with_sum_14 : ℕ :=
  333322

theorem largest_number_digits_sum_14:
  ∀ (ds : List ℕ), (∀ d ∈ ds, d = 2 ∨ d = 3) ∧ ds.sum = 14 →
    nat.join ds = 333322 :=
sorry

end largest_number_digits_sum_14_l814_814835


namespace probability_proof_l814_814144

-- Define the problem domain
def domain : set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 7}

-- Define the event we're interested in
def event : set (ℝ × ℝ) := {p | p.1 + p.2 ≤ 5}

-- Calculate the area of the region of interest
def area_region_of_interest : ℝ := (1 / 2) * 4 * 5

-- Calculate the total area
def total_area : ℝ := 4 * 7

-- Probability of the event given the domain
def probability_of_event := area_region_of_interest / total_area

noncomputable def proof_problem : Prop :=
  probability_of_event = 5 / 14

-- Proof (stubbed)
theorem probability_proof : proof_problem := 
by {
  sorry
}

end probability_proof_l814_814144


namespace find_lambda_l814_814978
/-
Import the complete Mathlib library to provide all necessary functions.
-/

/-
Define the original line, the translation operation, and the circle.
-/
def line (x y λ : ℝ) : Prop := x - 2 * y + λ = 0

def translate (x y λ : ℝ) : Prop := x - 2 * y + λ - 3 = 0

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y = 0

/-
State the correct proofs as a Lean theorem.
-/
theorem find_lambda (λ : ℝ) :
  (∀ x y : ℝ, circle_eq x y → translate x y λ → (abs (-1 - 2 * (2) + λ - 3) = sqrt 5)) → 
  λ = 3 ∨ λ = 13 :=
by
  sorry

end find_lambda_l814_814978


namespace gain_percentage_l814_814481

theorem gain_percentage (selling_price gain : ℝ) (h_selling : selling_price = 90) (h_gain : gain = 15) : 
  (gain / (selling_price - gain)) * 100 = 20 := 
by
  sorry

end gain_percentage_l814_814481


namespace symmetry_axis_l814_814567

theorem symmetry_axis (x : ℝ) : 
  (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3) → 
  y = sin (1/2 * x + Real.pi / 3) :=
sorry

end symmetry_axis_l814_814567


namespace length_of_PQ_l814_814370

-- Define the points and lines
def R : ℝ × ℝ := (10, 7)

-- Line equations
def line1 (P : ℝ × ℝ) : Prop := 6 * P.2 = 11 * P.1
def line2 (Q : ℝ × ℝ) : Prop := 7 * Q.2 = 2 * Q.1

-- Midpoint condition
def midpoint (P Q R : ℝ × ℝ) : Prop :=
  (R.1 = (P.1 + Q.1) / 2) ∧ (R.2 = (P.2 + Q.2) / 2)

-- Distance formula
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Prove that the length of PQ is 337824/1365
theorem length_of_PQ (P Q : ℝ × ℝ) (hP : line1 P) (hQ : line2 Q) (hmid : midpoint P Q R) :
  distance P Q = 337824 / 1365 :=
by
  sorry

end length_of_PQ_l814_814370


namespace probability_of_drawing_three_white_balls_l814_814512

theorem probability_of_drawing_three_white_balls
  (total_balls white_balls black_balls: ℕ)
  (h_total: total_balls = 15)
  (h_white: white_balls = 7)
  (h_black: black_balls = 8)
  (draws: ℕ)
  (h_draws: draws = 3) :
  (Nat.choose white_balls draws / Nat.choose total_balls draws) = (7 / 91) :=
by sorry

end probability_of_drawing_three_white_balls_l814_814512


namespace rationalize_denominator_min_sum_l814_814772

theorem rationalize_denominator_min_sum :
  ∃ A B C D : ℤ, 
    A * (Real.sqrt B) + C == (D * (Real.sqrt 32)) / (Real.sqrt 16 - Real.sqrt 2) ∧
    0 < D ∧ 
    (∃ p : ℕ, p.prime ∧ ¬(∀ n : ℕ, B ≠ p ^ 2)) ∧ 
    (A + B + C + D) = 21 :=
sorry

end rationalize_denominator_min_sum_l814_814772


namespace bananas_total_days_l814_814521

theorem bananas_total_days 
  (total_bananas : ℕ)
  (first_day : ℕ)
  (daily_increase : ℕ)
  (sum_days : ℕ → ℤ → ℕ ↔ ℤ)
  (days : ℕ)
  (h_total : total_bananas = 100)
  (h_first : first_day = 8)
  (h_increase : daily_increase = 6)
  (h_sum : ∀ n a d, sum_days n a d = a * n + d * (n * (n - 1)) / 2) :
  (days = 5) :=
by
  have h_eq1 : sum_days 5 8 6 = 100 := by sorry
  have h_eq2 : total_bananas = sum_days 5 8 6 := by sorry
  exact sorry

end bananas_total_days_l814_814521


namespace batch_production_equation_l814_814517

theorem batch_production_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 20) :
  (500 / x) = (300 / (x - 20)) :=
sorry

end batch_production_equation_l814_814517


namespace a_nine_val_l814_814694

def sequence_a (n : ℕ) : ℕ → ℝ
| 1 := 1
| (k + 1) := 1 / (2 / (sequence_a k) + 1)

theorem a_nine_val : sequence_a 9 = 1 / (2^9 - 1) := 
sorry

end a_nine_val_l814_814694


namespace perpendicular_vector_condition_l814_814998

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

noncomputable def dot_product : ℝ := ⟪a, b⟫

theorem perpendicular_vector_condition 
  (h_unit_a : ∥a∥ = 1) 
  (h_unit_b : ∥b∥ = 1) 
  (h_angle : real.angle_of _ a b = real.pi / 3) :
  ⟪2 • a - b, b⟫ = 0 := 
begin
  sorry
end

end perpendicular_vector_condition_l814_814998


namespace three_digit_numbers_with_properties_l814_814799

noncomputable def valid_numbers_with_properties : List Nat :=
  [179, 239, 299, 359, 419, 479, 539, 599, 659, 719, 779, 839, 899, 959]

theorem three_digit_numbers_with_properties (N : ℕ) :
  N >= 100 ∧ N < 1000 ∧ 
  N ≡ 1 [MOD 2] ∧
  N ≡ 2 [MOD 3] ∧
  N ≡ 3 [MOD 4] ∧
  N ≡ 4 [MOD 5] ∧
  N ≡ 5 [MOD 6] ↔ N ∈ valid_numbers_with_properties :=
by
  sorry

end three_digit_numbers_with_properties_l814_814799


namespace base_length_of_isosceles_triangle_l814_814052

theorem base_length_of_isosceles_triangle (a b : ℕ) (h1 : a = 8) (h2 : b + 2 * a = 26) : b = 10 :=
by
  have h3 : 2 * 8 = 16 := by norm_num
  rw [h1] at h2
  rw [h3] at h2
  linarith

end base_length_of_isosceles_triangle_l814_814052


namespace abs_diff_between_sequences_l814_814086

def sequence_C (n : ℕ) : ℤ := 50 + 12 * (n - 1)
def sequence_D (n : ℕ) : ℤ := 50 + (-8) * (n - 1)

theorem abs_diff_between_sequences :
  |sequence_C 31 - sequence_D 31| = 600 :=
by
  sorry

end abs_diff_between_sequences_l814_814086


namespace total_cars_produced_l814_814878

def cars_produced_north_america : ℕ := 3884
def cars_produced_europe : ℕ := 2871
def cars_produced_asia : ℕ := 5273
def cars_produced_south_america : ℕ := 1945

theorem total_cars_produced : cars_produced_north_america + cars_produced_europe + cars_produced_asia + cars_produced_south_america = 13973 := by
  sorry

end total_cars_produced_l814_814878


namespace equation_of_chord_midpoint_l814_814639

theorem equation_of_chord_midpoint (x y : ℝ) : 
  (∃ (x_1 y_1 x_2 y_2 : ℝ), (x_1 + x_2 = 4) ∧ (y_1 + y_2 = 3) ∧ 
  (x_1, y_1, x_2, y_2 ∈ ℝ) ∧ 
  (x_1 ^ 2 / 16 + y_1 ^ 2 / 9 = 1) ∧ 
  (x_2 ^ 2 / 16 + y_2 ^ 2 / 9 = 1)) → 
  3 * x + 4 * y - 12 = 0 :=
begin
  sorry,
end

end equation_of_chord_midpoint_l814_814639


namespace find_circle_l814_814658

def circle := ℕ
def triangle := ℕ

axiom h1 : ∀ (c t : ℕ), c + c + c + c = t + t + c
axiom h2 : ∀ (t : ℕ), t = 63

theorem find_circle (c t : ℕ) (h1 : c + c + c + c = t + t + c) (h2 : t = 63) : c = 42 :=
by sorry

end find_circle_l814_814658


namespace hyperbola_eccentricity_condition_l814_814290

theorem hyperbola_eccentricity_condition (m : ℝ) (h : m > 0) : 
  (∃ e : ℝ, e = Real.sqrt (1 + m) ∧ e > Real.sqrt 2) → m > 1 :=
by
  sorry

end hyperbola_eccentricity_condition_l814_814290


namespace cevian_sum_less_than_largest_side_l814_814410

variable {α : Type}
variable [EuclideanGeometry α]

-- Defining the triangle and the given points
variables (A B C O A' B' C' : α)

-- Hypotheses based on the given conditions
hypothesis (tri : is_triangle A B C)
hypothesis (largest_side : bc_side_length A B C > ab_side_length A B C . ∧ bc_side_length A B C > ac_side_length A B C)
hypothesis (inside_point : in_interior O (triangle A B C))
hypothesis (intersects_opposite_sides :
  on_line A' (line_through A O) ∧ on_line A' (opposite_line A (triangle A B C)) ∧ 
  on_line B' (line_through B O) ∧ on_line B' (opposite_line B (triangle A B C)) ∧ 
  on_line C' (line_through C O) ∧ on_line C' (opposite_line C (triangle A B C)))

-- The theorem to prove the required inequality
theorem cevian_sum_less_than_largest_side :
  distance O A' + distance O B' + distance O C' < distance B C := 
sorry

end cevian_sum_less_than_largest_side_l814_814410


namespace sum_of_roots_l814_814960

noncomputable def poly := (x - 1) ^ 2008 + 2 * (x - 2) ^ 2007 + 3 * (x - 3) ^ 2006 +
  (Finset.range 2007).sum (λ k, (↑k + 1) * (x - (↑k + 1)) ^ (2008 - (k + 1)))

theorem sum_of_roots : (sum_of_roots poly) = 2006 :=
by 
  sorry

end sum_of_roots_l814_814960


namespace find_valid_pairs_l814_814563

theorem find_valid_pairs :
  (∃ n : ℕ, ∃ m : ℕ, 1 ≤ m ∧ m ≤ 2314 ∧ 5^n < 2^m ∧ 2^{m + 1} < 5^{n + 1}) → 
  ∑ k in range 1000, 1 = 1000 :=
by sorry

end find_valid_pairs_l814_814563


namespace trigonometric_eq_has_root_l814_814030

theorem trigonometric_eq_has_root
  (n : ℕ) (h : 0 < n)
  (a b : Fin n → ℝ) :
  ∃ x : ℝ, ∑ k in Finset.range n, a k * Real.sin ((k + 1) * x) = ∑ k in Finset.range n, b k * Real.cos ((k + 1) * x) :=
by
  sorry

end trigonometric_eq_has_root_l814_814030


namespace find_matrix_M_transformed_curve_equation_l814_814045

open Matrix

noncomputable def M1 : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 2], ![1, 1]]
noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 2], ![3, 4]]
noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ := ![![7, 10], ![4, 6]]

theorem find_matrix_M :
  M1 ⬝ A = B :=
by sorry

noncomputable def curve_transformation (x y : ℚ) : ℚ × ℚ :=
((1 * x + 2 * y), (1 * x + 1 * y))

def transformed_curve (x' y' : ℚ) : Prop :=
x' ^ 2 + 2 * y' ^ 2 = 1

theorem transformed_curve_equation :
  ∀ x y, 3 * x ^ 2 + 8 * x * y + 6 * y ^ 2 = 1 →
    transformed_curve (curve_transformation x y).1 (curve_transformation x y).2 :=
by sorry

end find_matrix_M_transformed_curve_equation_l814_814045


namespace probability_three_white_balls_l814_814497

noncomputable def probability_all_white (white black total_drawn : ℕ) : ℚ :=
  (nat.choose white total_drawn : ℚ) / (nat.choose (white + black) total_drawn : ℚ)

theorem probability_three_white_balls :
  probability_all_white 7 8 3 = 1 / 13 :=
by 
  sorry

end probability_three_white_balls_l814_814497


namespace coeff_x6_zero_l814_814593

noncomputable def coeff_in_expansion : ℚ :=
  let expr := (3⁻¹ * x^3 - 3 * x⁻²) ^ 9
  coeff x 6 expr

theorem coeff_x6_zero : coeff_in_expansion = 0 :=
sorry

end coeff_x6_zero_l814_814593


namespace cosine_angle_PA_CF_l814_814781

-- We start by defining the points and lines in the coordinate system.

def P := (0, 0, Real.sqrt 2)
def A := (1, -1, 0)
def B := (1, 1, 0)
def C := (-1, 1, 0)
def D := (-1, -1, 0)
def F := (1, 1, -2)

-- Define the vectors PA and CF.
def vector_PA := (1 - 0, (-1) - 0, 0 - Real.sqrt 2)
def vector_CF := (1 - (-1), 1 - 1, -2 - 0)

-- Calculate the norms of vectors PA and CF.
def norm_PA := Real.sqrt (1^2 + (-1)^2 + (-Real.sqrt 2)^2)
def norm_CF := Real.sqrt (2^2 + 0^2 + (-2)^2)

-- Calculate the dot product of vectors PA and CF.
def dot_product_PA_CF := (1 * 2) + (-1 * 0) + (-Real.sqrt 2 * -2)

-- Define the correct answer.
def correct_cos_theta := (2 + Real.sqrt 2) / (2 * 2 * Real.sqrt 2)

-- The theorem to be proved: the cosine of the angle is equal to the given correct answer.
theorem cosine_angle_PA_CF : 
    (dot_product_PA_CF / (norm_PA * norm_CF)) = correct_cos_theta := 
by
    sorry

end cosine_angle_PA_CF_l814_814781


namespace number_of_real_pairs_l814_814801

theorem number_of_real_pairs :
  ∃! (x y : ℝ), 11 * x^2 + 2 * x * y + 9 * y^2 + 8 * x - 12 * y + 6 = 0 :=
sorry

end number_of_real_pairs_l814_814801


namespace value_of_expression_l814_814678

theorem value_of_expression (a b : ℝ) (h : a + b = 4) : a^2 + 2 * a * b + b^2 = 16 := by
  sorry

end value_of_expression_l814_814678


namespace abundant_numbers_less_than_50_number_of_abundant_numbers_less_than_50_l814_814313

def sum_of_proper_factors (n : Nat) : Nat :=
  (List.range n).filter (λ i => i > 0 ∧ n % i = 0).sum

def is_abundant (n : Nat) : Prop :=
  sum_of_proper_factors n > n

def abundant_numbers_below (n : Nat) : List Nat :=
  (List.range n).filter is_abundant

theorem abundant_numbers_less_than_50 : 
  abundant_numbers_below 50 = [12, 18, 20, 24, 30, 36, 40, 42, 48] := 
by sorry

theorem number_of_abundant_numbers_less_than_50 : 
  List.length (abundant_numbers_below 50) = 9 := 
by sorry

end abundant_numbers_less_than_50_number_of_abundant_numbers_less_than_50_l814_814313


namespace half_pow_sub_l814_814239

theorem half_pow_sub (x y : ℝ) (h₀ : x > y) (h₁ : y > 0) : (1 / 2)^x - (1 / 2)^y < 0 :=
by
  sorry

end half_pow_sub_l814_814239


namespace maximum_m_value_l814_814234

theorem maximum_m_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  ∃ m, m = 4 ∧ (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → (1 / a + 1 / b) ≥ m) :=
sorry

end maximum_m_value_l814_814234


namespace selection_and_ranking_methods_arrangement_methods_l814_814447

-- Problem 1
theorem selection_and_ranking_methods :
  let products := 10
  let excluded := 2
  let remaining := products - excluded
  (∑ (i : ℕ) in finset.range remaining, (∏ (j : ℕ) in finset.range 4, i - j)) == 1680 :=
by sorry

-- Problem 2
theorem arrangement_methods :
  let products := 10
  let gold_medal := 2
  let positions := 6
  let remaining_products := products - gold_medal
  (∑ (i : ℕ) in finset.range positions, (∏ (j : ℕ) in finset.range 2, i - j))
  * (∑ (k : ℕ) in finset.range remaining_products,
      (∏ (l : ℕ) in finset.range 4, k - l)) == 50400 :=
by sorry

end selection_and_ranking_methods_arrangement_methods_l814_814447


namespace find_linear_function_and_triangle_area_l814_814520

theorem find_linear_function_and_triangle_area :
  ∃ (k b : ℝ), (b = 2) ∧ (k + b = 3) ∧
  (triangle_area (line_eq k b) [(−2, 0), (0, 2)]) = 2 :=
by
  sorry

/-- Helper definitions for clarity -/
def line_eq (k b : ℝ) : ℝ → ℝ := λ x, k * x + b

noncomputable def triangle_area (f : ℝ → ℝ) (points : List (ℝ × ℝ)) : ℝ := 
  -- Assume that points contains exactly the two points of intercepts
  let [(x₁, y₁), (x₂, y₂)] := points
  0.5 * |x₁ * y₂ - x₂ * y₁|

end find_linear_function_and_triangle_area_l814_814520


namespace max_a_plus_b_l814_814963

def z1 (θ : ℝ) : ℂ := (8 + (1 : ℂ)) * Complex.sin θ + (7 + 4 * (1 : ℂ)) * Complex.cos θ
def z2 (θ : ℝ) : ℂ := (1 + 8 * (1 : ℂ)) * Complex.sin θ + (4 + 7 * (1 : ℂ)) * Complex.cos θ

theorem max_a_plus_b (θ : ℝ) : ∃ (a b : ℝ), (a + b = 125) → 
  let z := z1 θ * z2 θ in a = z.re ∧ b = z.im :=
sorry

end max_a_plus_b_l814_814963


namespace smallest_a1_l814_814742

theorem smallest_a1 (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_recurrence : ∀ n > 1, a n = 7 * a (n - 1) - 2 * n) :
  ∃ a1, (∀ a1' < a1, ¬(a1 = a1')) ∧ a 1 = 1 :=
by
  sorry

end smallest_a1_l814_814742


namespace possible_values_l814_814299

theorem possible_values (a b : ℕ → ℕ) (h1 : ∀ n, a n < (a (n + 1)))
  (h2 : ∀ n, b n < (b (n + 1)))
  (h3 : a 10 = b 10)
  (h4 : a 10 < 2017)
  (h5 : ∀ n, a (n + 2) = a (n + 1) + a n)
  (h6 : ∀ n, b (n + 1) = 2 * b n) :
  ∃ (a1 b1 : ℕ), (a 1 = a1) ∧ (b 1 = b1) ∧ (a1 + b1 = 13 ∨ a1 + b1 = 20) := sorry

end possible_values_l814_814299


namespace Q_moves_clockwise_l814_814687

-- Conditions
def on_unit_circle (P : ℝ → ℝ × ℝ) (ω : ℝ) : Prop :=
  ∀ t, P t = (Real.cos (ω * t), Real.sin (ω * t))

def Q_coordinates (P : ℝ → ℝ × ℝ) (Q : ℝ → ℝ × ℝ) : Prop :=
  ∀ t, let (x, y) := P t in Q t = (-2 * x * y, y^2 - x^2)

-- The Proof Statement
theorem Q_moves_clockwise (P Q : ℝ → ℝ × ℝ) (ω : ℝ) :
  on_unit_circle P ω →
  Q_coordinates P Q →
  ∀ t, Q t = (Real.cos (-(2 * ω) * t), Real.sin (-(2 * ω) * t)) :=
by
  intro h1 h2
  sorry

end Q_moves_clockwise_l814_814687


namespace maria_savings_percentage_is_33_l814_814541

noncomputable def regular_price : ℝ := 60
noncomputable def second_pair_price : ℝ := regular_price - (0.4 * regular_price)
noncomputable def third_pair_price : ℝ := regular_price - (0.6 * regular_price)
noncomputable def total_regular_price : ℝ := 3 * regular_price
noncomputable def total_discounted_price : ℝ := regular_price + second_pair_price + third_pair_price
noncomputable def savings : ℝ := total_regular_price - total_discounted_price
noncomputable def savings_percentage : ℝ := (savings / total_regular_price) * 100

theorem maria_savings_percentage_is_33 :
  savings_percentage = 33 :=
by
  sorry

end maria_savings_percentage_is_33_l814_814541


namespace total_masks_correct_l814_814815

-- Define the conditions
def boxes := 18
def capacity_per_box := 15
def deficiency_per_box := 3
def masks_per_box := capacity_per_box - deficiency_per_box
def total_masks := boxes * masks_per_box

-- The theorem statement we need to prove
theorem total_masks_correct : total_masks = 216 := by
  unfold total_masks boxes masks_per_box capacity_per_box deficiency_per_box
  sorry

end total_masks_correct_l814_814815


namespace largest_angle_between_a_and_c_l814_814374

variables (a b c : ℝ^3)

-- Defining the conditions given in the problem
def norm_a : ∥a∥ = 2 := sorry
def norm_b : ∥b∥ = 2 := sorry
def norm_c : ∥c∥ = 3 := sorry
def vector_triple_product (a b c : ℝ^3) : a × (b × c) + 2 • b = 0 := sorry

-- Statement of the proof
theorem largest_angle_between_a_and_c (a b c : ℝ^3)
  (h1 : ∥a∥ = 2)
  (h2 : ∥b∥ = 2)
  (h3 : ∥c∥ = 3)
  (h4 : a × (b × c) + 2 • b = 0) : 
  ∃ θ : ℝ, real.arccos (-1/3) = θ ∧ θ ≈ 109.47 := 
by sorry

end largest_angle_between_a_and_c_l814_814374


namespace find_winner_votes_l814_814824

-- Define the conditions
variables (V : ℝ) (winner_votes second_votes : ℝ)
def election_conditions :=
  winner_votes = 0.468 * V ∧
  second_votes = 0.326 * V ∧
  winner_votes - second_votes = 752

-- State the theorem
theorem find_winner_votes (h : election_conditions V winner_votes second_votes) :
  winner_votes = 2479 :=
sorry

end find_winner_votes_l814_814824


namespace distinct_digit_S_problem_l814_814346

theorem distinct_digit_S_problem :
  ∃! (S : ℕ), S < 10 ∧ 
  ∃ (P Q R : ℕ), P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ S ∧ R ≠ S ∧ 
  P < 10 ∧ Q < 10 ∧ R < 10 ∧
  ((P + Q = S) ∨ (P + Q = S + 10)) ∧
  (R = 0) :=
sorry

end distinct_digit_S_problem_l814_814346


namespace child_support_amount_l814_814019

-- Definitions
def base_salary_1_3 := 30000
def base_salary_4_7 := 36000
def bonus_1 := 2000
def bonus_2 := 3000
def bonus_3 := 4000
def bonus_4 := 5000
def bonus_5 := 6000
def bonus_6 := 7000
def bonus_7 := 8000
def child_support_1_5 := 30 / 100
def child_support_6_7 := 25 / 100
def paid_total := 1200

-- Total Income per year
def income_year_1 := base_salary_1_3 + bonus_1
def income_year_2 := base_salary_1_3 + bonus_2
def income_year_3 := base_salary_1_3 + bonus_3
def income_year_4 := base_salary_4_7 + bonus_4
def income_year_5 := base_salary_4_7 + bonus_5
def income_year_6 := base_salary_4_7 + bonus_6
def income_year_7 := base_salary_4_7 + bonus_7

-- Child Support per year
def support_year_1 := child_support_1_5 * income_year_1
def support_year_2 := child_support_1_5 * income_year_2
def support_year_3 := child_support_1_5 * income_year_3
def support_year_4 := child_support_1_5 * income_year_4
def support_year_5 := child_support_1_5 * income_year_5
def support_year_6 := child_support_6_7 * income_year_6
def support_year_7 := child_support_6_7 * income_year_7

-- Total Support calculation
def total_owed := support_year_1 + support_year_2 + support_year_3 + 
                  support_year_4 + support_year_5 +
                  support_year_6 + support_year_7

-- Final amount owed
def amount_owed := total_owed - paid_total

-- Theorem statement
theorem child_support_amount :
  amount_owed = 75150 :=
sorry

end child_support_amount_l814_814019


namespace find_sin_tan_l814_814627

variable {α : Real} (h1 : cos α = - (4/5)) (h2 : π < α ∧ α < 3 * π / 2)

theorem find_sin_tan : sin α = - (3/5) ∧ tan α = 3/4 :=
  by
    sorry

end find_sin_tan_l814_814627


namespace shirts_sold_l814_814530

theorem shirts_sold (initial final : ℕ) (h : initial = 49) (h1 : final = 28) : initial - final = 21 :=
sorry

end shirts_sold_l814_814530


namespace balance_relationship_l814_814340

theorem balance_relationship (x : ℕ) (hx : 0 ≤ x ∧ x ≤ 5) : 
  ∃ y : ℝ, y = 200 - 36 * x := 
sorry

end balance_relationship_l814_814340


namespace find_cost_price_l814_814177

noncomputable def cost_price (SP : ℝ) (profit_margin : ℝ) : ℝ :=
  SP / profit_margin

theorem find_cost_price : 
  let SP := 69.44
  let profit_margin := 1.25
  cost_price SP profit_margin = 55.552
:= by
  have SP_eq : SP = 69.44 := rfl
  have profit_margin_eq : profit_margin = 1.25 := rfl
  rw [SP_eq, profit_margin_eq]
  show 69.44 / 1.25 = 55.552
  simp
  sorry

end find_cost_price_l814_814177


namespace abc_sub_c_minus_2023_eq_2023_l814_814677

theorem abc_sub_c_minus_2023_eq_2023 (a b c : ℝ) (h : a * b = 1) : 
  a * b * c - (c - 2023) = 2023 := 
by sorry

end abc_sub_c_minus_2023_eq_2023_l814_814677


namespace trihedral_angle_inequality_l814_814734

variables {O A B C P P1 P2 P3 : Point}
variable {α β γ : ℝ}
variables {OP PP1 PP2 PP3 : ℝ}

-- Define the conditions
def conditions (O A B C P P1 P2 P3 : Point) (α β γ : ℝ) : Prop :=
  angle O B C = α ∧
  angle O C A = β ∧
  angle O A B = γ ∧
  α + β + γ = Real.pi ∧
  is_interior_point P (triangle O A B C) ∧
  projections P P1 P2 P3 (faces O A B C)

-- Theorem statement with the given conditions and the inequality to prove
theorem trihedral_angle_inequality (h : conditions O A B C P P1 P2 P3 α β γ) : OP ≥ PP1 + PP2 + PP3 :=
sorry

end trihedral_angle_inequality_l814_814734


namespace number_of_valid_pairs_l814_814189

-- Definition of the factorial 10!
def ten_factorial : ℕ := 10.factorial

-- Definition for the problem conditions
def valid_pair (m n : ℕ) : Prop :=
m ∣ ten_factorial ∧ n ∣ ten_factorial ∧ (2 ^ m - 1) * (2 ^ n - 1) ∣ (2 ^ ten_factorial - 1)

-- Main theorem to be proven
theorem number_of_valid_pairs : 
  (finset.univ.filter (λ (i : ℕ × ℕ), valid_pair i.fst i.snd)).card = 5509 :=
sorry

end number_of_valid_pairs_l814_814189


namespace BD_not_both_win_l814_814607

-- Define the statements as Boolean variables.
variables (win_A win_B win_C win_D : Prop)

-- Define the statements made by A, B, C, D
def statement_A : Prop := win_B ∨ win_C → ¬win_A
def statement_B : Prop := win_A ∧ win_C
def statement_C : Prop := ¬win_C
def statement_D : Prop := win_B

-- Define the condition: only two of these statements are true
def condition : Prop :=
  (statement_A ∧ statement_B ∧ ¬statement_C ∧ ¬statement_D) ∨
  (statement_A ∧ ¬statement_B ∧ statement_C ∧ ¬statement_D) ∨
  (statement_A ∧ ¬statement_B ∧ ¬statement_C ∧ statement_D) ∨
  (¬statement_A ∧ statement_B ∧ statement_C ∧ ¬statement_D) ∨
  (¬statement_A ∧ statement_B ∧ ¬statement_C ∧ statement_D) ∨
  (¬statement_A ∧ ¬statement_B ∧ statement_C ∧ statement_D)

-- The statement to be proven: B and D cannot both win
theorem BD_not_both_win (h : condition) : ¬ (win_B ∧ win_D) :=
by sorry

end BD_not_both_win_l814_814607


namespace resulting_polygon_sides_l814_814194

/-
Problem statement: 

Construct a regular pentagon on one side of a regular heptagon.
On one non-adjacent side of the pentagon, construct a regular hexagon.
On a non-adjacent side of the hexagon, construct an octagon.
Continue to construct regular polygons in the same way, until you construct a nonagon.
How many sides does the resulting polygon have?

Given facts:
1. Start with a heptagon (7 sides).
2. Construct a pentagon (5 sides) on one side of the heptagon.
3. Construct a hexagon (6 sides) on a non-adjacent side of the pentagon.
4. Construct an octagon (8 sides) on a non-adjacent side of the hexagon.
5. Construct a nonagon (9 sides) on a non-adjacent side of the octagon.
-/

def heptagon_sides : ℕ := 7
def pentagon_sides : ℕ := 5
def hexagon_sides : ℕ := 6
def octagon_sides : ℕ := 8
def nonagon_sides : ℕ := 9

theorem resulting_polygon_sides : 
  (heptagon_sides + nonagon_sides - 2 * 1) + (pentagon_sides + hexagon_sides + octagon_sides - 3 * 2) = 27 := by
  sorry

end resulting_polygon_sides_l814_814194


namespace valid_necklace_length_l814_814164

-- Define the shapes and colors as types
inductive Shape | cube | pyramid
inductive Color | green | blue | red

-- Each bead is characterized by its shape and color
structure Bead :=
  (shape : Shape)
  (color : Color)

-- Definition to check adjacent beads for different shape and color
def valid_pair (b1 b2 : Bead) :=
  (b1.shape ≠ b2.shape) ∧ (b1.color ≠ b2.color)

-- Definition of a valid necklace
def valid_necklace (beads : List Bead) :=
  (List.length beads) % 2 = 0 ∧
  (∀ i : Fin (List.length beads), valid_pair (beads.get i) (beads.get ((i + 1) % (List.length beads))))

-- The main proposition
theorem valid_necklace_length (n : Nat) :
  n = 8 ∨ n = 10 ↔
  ∃ beads : List Bead,
    valid_necklace beads ∧
    List.length beads = n :=
by
  sorry

end valid_necklace_length_l814_814164


namespace range_of_a_l814_814371

theorem range_of_a (U : Set ℝ) (A B : Set ℝ) (a : ℝ) 
  (hU : U = Set.univ) 
  (hA : A = {x | x ≥ 1}) 
  (hB : B = {x | x > a}) 
  (h : Set.compl A ∪ B = Set.univ) : 
  a ∈ (-∞, 1) :=
sorry

end range_of_a_l814_814371


namespace probability_three_white_balls_l814_814501

open Nat

def totalWaysToDrawThreeBalls : ℕ := choose 15 3
def waysToDrawThreeWhiteBalls : ℕ := choose 7 3

theorem probability_three_white_balls :
  (waysToDrawThreeWhiteBalls : ℚ) / (totalWaysToDrawThreeBalls : ℚ) = 1 / 13 := 
sorry

end probability_three_white_balls_l814_814501


namespace problem_statement_l814_814786

variable (a : ℝ)

theorem problem_statement (h : 5 = a + a⁻¹) : a^4 + (a⁻¹)^4 = 527 := 
by 
  sorry

end problem_statement_l814_814786


namespace probability_three_white_balls_l814_814503

open Nat

def totalWaysToDrawThreeBalls : ℕ := choose 15 3
def waysToDrawThreeWhiteBalls : ℕ := choose 7 3

theorem probability_three_white_balls :
  (waysToDrawThreeWhiteBalls : ℚ) / (totalWaysToDrawThreeBalls : ℚ) = 1 / 13 := 
sorry

end probability_three_white_balls_l814_814503


namespace distance_PQ_l814_814372

theorem distance_PQ :
  let W' := (0 : ℝ, 0 : ℝ, 12 : ℝ)
  let X' := (2 : ℝ, 0 : ℝ, 10 : ℝ)
  let Y' := (2 : ℝ, 3 : ℝ, 16 : ℝ)
  let Z' := (0 : ℝ, 3 : ℝ, 20 : ℝ)
  let P := ((W'.1 + Y'.1) / 2, (W'.2 + Y'.2) / 2, (W'.3 + Y'.3) / 2)
  let Q := ((X'.1 + Z'.1) / 2, (X'.2 + Z'.2) / 2, (X'.3 + Z'.3) / 2)
  dist P Q = 1 :=
by
  sorry

end distance_PQ_l814_814372


namespace problem_statement_l814_814409

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem problem_statement : avg3 (avg3 (-1) 2 3) (avg2 2 3) 1 = 29 / 18 := 
by 
  sorry

end problem_statement_l814_814409


namespace train_lengths_l814_814831

noncomputable def length_of_train_A : ℝ :=
  let L_A := 400 in
  L_A

noncomputable def length_of_train_B : ℝ :=
  let L_B := 800 / 3 in
  L_B

theorem train_lengths (bridge_length : ℝ) (time_A_bridge : ℝ) (time_B_bridge : ℝ) (time_A_lamp : ℝ) (time_B_lamp : ℝ) :
  bridge_length = 800 →
  time_A_bridge = 45 →
  time_B_bridge = 40 →
  time_A_lamp = 15 →
  time_B_lamp = 10 →
  ∃ (L_A L_B : ℝ), 
    L_A = length_of_train_A ∧ 
    L_B = length_of_train_B :=
by 
  intros h_bridge h_timeA_bridge h_timeB_bridge h_timeA_lamp h_timeB_lamp
  use length_of_train_A, length_of_train_B
  constructor
  . exact rfl
  . exact rfl

end train_lengths_l814_814831


namespace factor_expression_l814_814949

variable (x : ℝ)

-- Mathematically define the expression e
def e : ℝ := 4 * x * (x + 2) + 10 * (x + 2) + 2 * (x + 2)

-- State that e is equivalent to the factored form
theorem factor_expression : e x = (x + 2) * (4 * x + 12) :=
by
  sorry

end factor_expression_l814_814949


namespace equilateral_triangle_inscribed_circle_area_l814_814813

theorem equilateral_triangle_inscribed_circle_area (r : ℝ) (h : r = 2) :
  let PQR_area := 3 * real.sqrt 3 in
  ∃ P Q R : ℝ × ℝ, ∃ O : ℝ × ℝ, 
    dist O P = r ∧ dist O Q = r ∧ dist O R = r ∧ 
    triangle.is_equilateral P Q R ∧
    triangle.area P Q R = PQR_area :=
begin
  sorry
end

end equilateral_triangle_inscribed_circle_area_l814_814813


namespace valid_necklace_length_l814_814166

-- Define the shapes and colors as types
inductive Shape | cube | pyramid
inductive Color | green | blue | red

-- Each bead is characterized by its shape and color
structure Bead :=
  (shape : Shape)
  (color : Color)

-- Definition to check adjacent beads for different shape and color
def valid_pair (b1 b2 : Bead) :=
  (b1.shape ≠ b2.shape) ∧ (b1.color ≠ b2.color)

-- Definition of a valid necklace
def valid_necklace (beads : List Bead) :=
  (List.length beads) % 2 = 0 ∧
  (∀ i : Fin (List.length beads), valid_pair (beads.get i) (beads.get ((i + 1) % (List.length beads))))

-- The main proposition
theorem valid_necklace_length (n : Nat) :
  n = 8 ∨ n = 10 ↔
  ∃ beads : List Bead,
    valid_necklace beads ∧
    List.length beads = n :=
by
  sorry

end valid_necklace_length_l814_814166


namespace factory_maximum_profit_l814_814880

-- Definitions based on conditions
def fixed_cost := 2.5
def selling_price_per_unit := 0.05

def additional_cost (x : ℝ) : ℝ :=
  if x < 80 then (1 / 3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

def revenue (x : ℝ) : ℝ :=
  selling_price_per_unit * 1000 * x

def profit (x : ℝ) : ℝ :=
  revenue(x) - additional_cost(x) - fixed_cost

-- Define the function L(x)
def L (x : ℝ) : ℝ :=
  if x > 0 ∧ x < 80 then
    - (1 / 3) * x^2 + 40 * x - 250
  else
    1200 - (x + 10000 / x)

-- Lean theorem statement for maximum profit
theorem factory_maximum_profit : ∀ x : ℝ, 0 < x → 
  (L 100 = max (L x) for x ∈ set.Ioi 0) :=
by
  sorry

end factory_maximum_profit_l814_814880


namespace point_on_x_axis_l814_814991

theorem point_on_x_axis (a : ℝ) (h₁ : 1 - a = 0) : (3 * a - 6, 1 - a) = (-3, 0) :=
by
  sorry

end point_on_x_axis_l814_814991


namespace largest_unique_k_l814_814098

theorem largest_unique_k (n : ℕ) :
  (∀ k : ℤ, (8:ℚ)/15 < n / (n + k) ∧ n / (n + k) < 7/13 → False) ∧
  (∃ k : ℤ, (8:ℚ)/15 < n / (n + k) ∧ n / (n + k) < 7/13) → n = 112 :=
by sorry

end largest_unique_k_l814_814098


namespace average_birth_rate_l814_814338

theorem average_birth_rate 
  (death_rate : ℕ) 
  (net_increase_one_day : ℕ) 
  (intervals_per_day : ℕ) 
  (net_increase_per_two_seconds : ℕ) 
  (B : ℕ):
  death_rate = 6 →
  net_increase_one_day = 86400 →
  intervals_per_day = 43200 →
  net_increase_per_two_seconds = net_increase_per_two_seconds →
  (net_increase_per_two_seconds = 86400) →
  B = ((net_increase_one_day + (death_rate * intervals_per_day)) / intervals_per_day) →
  B = 8 :=
by
  intro h_death_rate h_net_increase h_intervals h_net_per_two_seconds h_net_eq
  have : net_increase_per_two_seconds = 43200 * (B - death_rate), by sorry
  have : net_increase_per_two_seconds = net_increase_one_day, by sorry
  have : B = (net_increase_one_day + (death_rate * intervals_per_day)) / intervals_per_day, by sorry
  exact h_net_eq

end average_birth_rate_l814_814338


namespace percent_increase_decrease_l814_814153

theorem percent_increase_decrease (P y : ℝ) (h : (P * (1 + y / 100) * (1 - y / 100) = 0.90 * P)) :
    y = 31.6 :=
by
  sorry

end percent_increase_decrease_l814_814153


namespace sum_values_of_x_l814_814006

def f (x : ℝ) : ℝ :=
if x ≤ 3 then 7 * x + 10 else 3 * x - 9

theorem sum_values_of_x (x1 x2 : ℝ) (h1 : f x1 = 3) (h2 : f x2 = 3) :
  x1 + x2 = 3 :=
begin
  sorry
end

end sum_values_of_x_l814_814006


namespace distance_AB_l814_814718

open Real

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 - t, 2 * t)

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ := (2 + 2 * cos θ, 2 * sin θ)

noncomputable def curve_C2 (θ : ℝ) : ℝ := 4 * sin θ

def polar_eq_curve_C1 (ρ θ : ℝ) : Prop := ρ = 4 * cos θ
def polar_eq_line_l (ρ θ : ℝ) : Prop := 2 * ρ * cos θ + ρ * sin θ - 4 = 0

def given_conditions (α ρ : ℝ) : Prop :=
  (α > 0 ∧ π / 4 < α ∧ α < π / 2) ∧
  (∃ ρ_A ρ_B, 
    ρ_A = 4 * cos α ∧ ρ_B = 4 * sin α ∧ 
    (2 * (sin α + cos α), α) ∈ {p : ℝ × ℝ | polar_eq_line_l p.1 p.2})

def find_AB_distance (α : ℝ) : ℝ :=
  4 * abs (sin α - cos α)

theorem distance_AB (α : ℝ) (h : given_conditions α (4 * abs (sin α - cos α))) : 
  find_AB_distance α = (4 * sqrt 10) / 5 :=
sorry

end distance_AB_l814_814718


namespace circumcircle_diameter_l814_814967

variables (A B C : Type) [inner_product_space ℝ A] 

-- position vectors of the triangle vertices
variables (a b c : A) 
noncomputable def length_a : ℝ := 2
noncomputable def angle_A : ℝ := real.pi / 3

-- required diameter
def diameter : ℝ := 4 * real.sqrt (3) / 3

-- theorem statement
theorem circumcircle_diameter (a b: A)
  (length_a : ∥a - b∥ = 2)
  (angle_A : ∡a b c = real.pi / 3) :
  diameter a b c = 4 * real.sqrt(3) / 3 :=
sorry

end circumcircle_diameter_l814_814967


namespace intersection_cardinality_l814_814624

open Set

/-- Given sets A and B, prove the intersection of A and B has 2015 elements. -/
theorem intersection_cardinality :
  let A := { x : ℝ | log 3 x > 2 }
  let B := { x : ℕ | x ≤ 2024 }
  let intersection := { x : ℕ | 10 ≤ x ∧ x ≤ 2024 }
  ∃ (s : Finset ℕ), (∀ x, x ∈ s ↔ x ∈ intersection) ∧ Finset.card s = 2015 :=
by
  sorry

end intersection_cardinality_l814_814624


namespace constants_inequality_value_l814_814740

theorem constants_inequality_value
  (a b c d : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : ∀ x, (1 ≤ x ∧ x ≤ 5) ∨ (24 ≤ x ∧ x ≤ 26) ∨ x < -4 ↔ (x - a) * (x - b) * (x - c) / (x - d) ≤ 0) :
  a + 3 * b + 3 * c + 4 * d = 72 :=
sorry

end constants_inequality_value_l814_814740


namespace lines_intersect_at_common_point_l814_814764

-- Define the conditions
def is_equilateral (A B C : Point) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A

variables {α β γ : ℝ} (A B C A1 B1 C1 : Point)

-- Assume we have equilateral triangles built on triangle ABC
axiom equilateral_triangle_ABC1 : is_equilateral A B C1
axiom equilateral_triangle_AB1C : is_equilateral A B1 C
axiom equilateral_triangle_A1BC : is_equilateral A1 B C

-- Include trilinear coordinates definitions for the intersection points
def trilinear_coordinates (α β γ : ℝ) : ℝ × ℝ × ℝ :=
   ( 1 / sin(α ± π / 3), 1 / sin(β ± π / 3), 1 / sin(γ ± π / 3))

-- Theorem statement
theorem lines_intersect_at_common_point : 
  intersect (Line.mk A A1) (Line.mk B B1) (Line.mk C C1) 
  (trilinear_coordinates α β γ) :=
sorry

end lines_intersect_at_common_point_l814_814764


namespace B_150_eq_I_l814_814367

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
     #[0, 1, 0],
     #[0, 0, 1],
     #[1, 0, 0]
   ]

theorem B_150_eq_I : B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by sorry

end B_150_eq_I_l814_814367


namespace find_x_weeks_l814_814539

-- Definition of the problem conditions:
def archibald_first_two_weeks_apples : Nat := 14
def archibald_next_x_weeks_apples (x : Nat) : Nat := 14
def archibald_last_two_weeks_apples : Nat := 42
def total_weeks : Nat := 7
def weekly_average : Nat := 10

-- Statement of the theorem to prove that x = 2 given the conditions
theorem find_x_weeks :
  ∃ x : Nat, (archibald_first_two_weeks_apples + archibald_next_x_weeks_apples x + archibald_last_two_weeks_apples = total_weeks * weekly_average) 
  ∧ (archibald_next_x_weeks_apples x / x = 7) 
  → x = 2 :=
by
  sorry

end find_x_weeks_l814_814539


namespace building_shadow_length_l814_814886

-- Given conditions
def h_f : ℝ := 17.5               -- Height of the flagstaff
def s_f : ℝ := 40.25              -- Length of the flagstaff's shadow
def h_b : ℝ := 12.5               -- Height of the building

-- The theorem that needs to be proven
theorem building_shadow_length :
  ∃ (s_b : ℝ), s_b = 28.75 ∧ (h_f / s_f = h_b / s_b) :=
begin
  use 28.75,
  split,
  { refl },
  { -- The following steps directly solve for the ratios equality condition
    rw [h_f, s_f, h_b],
    norm_num,
  }
end

end building_shadow_length_l814_814886


namespace terry_current_age_l814_814700

theorem terry_current_age (T : ℕ) (nora_current_age : ℕ) (h1 : nora_current_age = 10)
  (h2 : T + 10 = 4 * nora_current_age) : T = 30 :=
by
  sorry

end terry_current_age_l814_814700


namespace total_points_scored_l814_814040

theorem total_points_scored (n : ℕ) (points_5_avg : ℕ) (points_remaining_avg : ℕ) (remainder : ℕ) :
  n = 9 → points_5_avg = 50 → points_remaining_avg = 5 → remainder = 4 →
  (5 * points_5_avg + remainder * points_remaining_avg) = 270 := 
by 
  intros hn hpa hra hr
  rw [hn, hpa, hra, hr]
  norm_num
  sorry

end total_points_scored_l814_814040


namespace part1_monotonic_increasing_part2_range_of_k_l814_814643

def f (x : ℝ) : ℝ := ln x - x^2

def y (x : ℝ) : ℝ := f x + x * (1/x - 2*x)

def g (x : ℝ) (b : ℝ) : ℝ := ln x + (1/2)*x^2 - (1 + b) * x

theorem part1_monotonic_increasing :
  (∀ x > 0, ((1 - 6 * x^2) / x > 0 ↔ 0 < x ∧ x < sqrt(6) / 6)) :=
begin
  intro x,
  split,
  { intro h,
    split; linarith },
  { intro h,
    linarith["inequalities about x"],
end

theorem part2_range_of_k
  (b : ℝ) (x1 x2 : ℝ) (h1 : b ≥ (exp(2) + 1) / exp(1) - 1)
  (h2 : (x1 + 1 / x1 = 1 + b) ∧ (x1 < x2) ∧ (0 < x1) ∧ (x1 ≤ 1 / exp(1)))
  (h3 : g x1 b - g x2 b ≥ k) :
  k ≤ (exp 2 / 2) - (1 / (2 * exp 2)) - 2 :=
begin
  sorry,
end

end part1_monotonic_increasing_part2_range_of_k_l814_814643


namespace find_number_eq_150_l814_814493

variable {x : ℝ}

theorem find_number_eq_150 (h : 0.60 * x - 40 = 50) : x = 150 :=
sorry

end find_number_eq_150_l814_814493


namespace total_books_in_bookcase_l814_814668

def num_bookshelves := 8
def num_layers_per_bookshelf := 5
def books_per_layer := 85

theorem total_books_in_bookcase : 
  (num_bookshelves * num_layers_per_bookshelf * books_per_layer) = 3400 := by
  sorry

end total_books_in_bookcase_l814_814668


namespace total_points_correct_l814_814389

def marks_team_2_pointers := 25
def marks_team_3_pointers := 8
def marks_team_free_throws := 10

def opponents_2_pointers := 2 * marks_team_2_pointers
def opponents_3_pointers := 0.5 * marks_team_3_pointers
def opponents_free_throws := 0.5 * marks_team_free_throws

noncomputable def total_points_mark_team := 
  (marks_team_2_pointers * 2) + 
  (marks_team_3_pointers * 3) + 
  (marks_team_free_throws * 1)

noncomputable def total_points_opponents := 
  (opponents_2_pointers * 2) + 
  (opponents_3_pointers * 3) + 
  (opponents_free_throws * 1)

noncomputable def total_points_both_teams := 
  total_points_mark_team + total_points_opponents

theorem total_points_correct : total_points_both_teams = 201 :=
  by sorry

end total_points_correct_l814_814389


namespace shadow_length_l814_814889

theorem shadow_length (h_flagstaff : ℝ) (s_flagstaff : ℝ) (h_building : ℝ) : 
  (h_flagstaff = 17.5) → (s_flagstaff = 40.25) → (h_building = 12.5) → 
  ∃ s_building : ℝ, (h_flagstaff / s_flagstaff = h_building / s_building) ∧ s_building = 28.75 :=
by
  intros hf sf hb
  use 28.75
  split
  sorry -- Prove the ratio equality here
  exact rfl -- Prove that s_building is indeed 28.75

end shadow_length_l814_814889


namespace girls_in_wind_band_not_string_band_l814_814150

def M_G : ℕ := 100
def F_G : ℕ := 80
def M_O : ℕ := 80
def F_O : ℕ := 100
def total_students : ℕ := 230
def boys_in_both : ℕ := 60

theorem girls_in_wind_band_not_string_band : (F_G - (total_students - (M_G + F_G + M_O + F_O - boys_in_both - boys_in_both))) = 10 :=
by
  sorry

end girls_in_wind_band_not_string_band_l814_814150


namespace radius_of_inscribed_semicircle_l814_814716

-- Definitions of the elements in the problem
variables (X Y Z : Type) [Triangle X Y Z]
variables (XZ YZ XY : ℝ)
variables (angleZ : ℝ)
variables (semiperimeter area r : ℝ)

-- Conditions
axiom XZ_eq_15 : XZ = 15
axiom YZ_eq_8 : YZ = 8
axiom angle_Z_right : angleZ = 90
axiom XY_eq_17 : XY = Real.sqrt (XZ^2 + YZ^2)
axiom area_XYY' : area = 120
axiom semiperimeter_XYY' : semiperimeter = 25
axiom radius_of_semicircle : r = area / semiperimeter

-- Theorem stating the problem
theorem radius_of_inscribed_semicircle : r = 24 / 5 :=
by 
  rw [←radius_of_semicircle]
  rw [semiperimeter_XYY', area_XYY']
  linarith

end radius_of_inscribed_semicircle_l814_814716


namespace rabbit_probability_l814_814147

def cube_vertices : ℕ := 8
def cube_edges : ℕ := 12
def moves : ℕ := 11
def paths_after_11_moves : ℕ := 3 ^ moves
def favorable_paths : ℕ := 24

theorem rabbit_probability :
  (favorable_paths : ℚ) / paths_after_11_moves = 24 / 177147 := by
  sorry

end rabbit_probability_l814_814147


namespace inradius_circumradius_le_height_l814_814755

theorem inradius_circumradius_le_height
    {α β γ : ℝ}
    (hα : 0 < α ∧ α ≤ 90)
    (hβ : 0 < β ∧ β ≤ 90)
    (hγ : 0 < γ ∧ γ ≤ 90)
    (α_ge_β : α ≥ β)
    (β_ge_γ : β ≥ γ)
    {r R h : ℝ} :
  r + R ≤ h := 
sorry

end inradius_circumradius_le_height_l814_814755


namespace problem_solution_l814_814281

noncomputable def α : ℝ :=
  let x := -1
  let y := 2
  real.arctan2 y x

theorem problem_solution (hx : -1 < 0) (hy : 2 > 0) (hpoint : (x, y) = (-1, 2)) :
  (real.tan (α / 2) = (1 + real.sqrt 5) / 2) ∧ 
  (real.pi / 2 < α ∧ α < 2 * real.pi / 3) :=
by
  sorry

end problem_solution_l814_814281


namespace unique_students_count_l814_814452

theorem unique_students_count 
  (O B C OB OC BC OBC : ℕ)
  (hO : O = 25)
  (hB : B = 40)
  (hC : C = 30)
  (hOB : OB = 5)
  (hOC : OC = 6)
  (hBC : BC = 4)
  (hOBC : OBC = 2) :
  O + B + C - (OB + OC + BC) + OBC = 82 :=
by
  rw [hO, hB, hC, hOB, hOC, hBC, hOBC]
  by norm_num
  sorry

end unique_students_count_l814_814452


namespace six_transform_1_12_eq_neg2_15_six_transform_1_17_eq_2_14_or_4_14_l814_814133

def transformPair (a b : ℤ) : ℕ → (ℤ × ℤ)
| 0       := (a, b)
| (n + 1) :=
  let (a', b') := transformPair a b n in
  match (a' + b') % 4 with
  | 0 => (a' + 1, b')
  | 1 => (a', b' + 1)
  | 2 => (a' - 1, b')
  | 3 => (a', b' - 1)
  | _ => (a', b') -- This case won't happen

theorem six_transform_1_12_eq_neg2_15 :
  transformPair 1 12 6 = (-2, 15) := 
sorry

theorem six_transform_1_17_eq_2_14_or_4_14 (p : ℤ × ℤ) :
  transformPair p.1 p.2 6 = (1, 17) → p = (2, 14) ∨ p = (4, 14) :=
sorry

end six_transform_1_12_eq_neg2_15_six_transform_1_17_eq_2_14_or_4_14_l814_814133


namespace sum_of_0_l814_814022

theorem sum_of_0.57_and_0.23 : 0.57 + 0.23 = 0.8 :=
by
  sorry

end sum_of_0_l814_814022


namespace time_spent_on_type_A_problems_l814_814334

def examination_duration : ℕ := 180
def total_questions : ℕ := 200
def type_A_problems : ℕ := 10
def type_A_time_multiplier : ℕ := 2

theorem time_spent_on_type_A_problems (x : ℚ) (x_pos : 0 < x) :
  20 * x = (120 : ℚ) / 7 :=
by
  have total_type_B_problems : ℕ := total_questions - type_A_problems
  have total_time_eq : 20 * x + total_type_B_problems * x = examination_duration := by
    linarith
  have total_time_eq' : (20 + total_type_B_problems) * x = examination_duration := by
    linarith
  have x_eq : x = 180 / 210 := by
    rw [total_time_eq']
    field_simp
  field_simp [x_eq]
  linarith

end time_spent_on_type_A_problems_l814_814334


namespace wilma_red_flowers_l814_814848

theorem wilma_red_flowers :
  ∀ (total_rows total_flowers_per_row yellow_flowers : ℕ),
  (∃ (green_factor : ℕ), green_factor = 2) →
  total_rows = 6 →
  total_flowers_per_row = 13 →
  yellow_flowers = 12 →
  let total_flowers := total_rows * total_flowers_per_row in
  let green_flowers := green_factor * yellow_flowers in
  let yellow_and_green_flowers := yellow_flowers + green_flowers in
  let red_flowers := total_flowers - yellow_and_green_flowers in
  red_flowers = 42 :=
by
  intros total_rows total_flowers_per_row yellow_flowers
  intro green_factor_exists
  intro total_rows_eq
  intro total_flowers_per_row_eq
  intro yellow_flowers_eq
  let total_flowers := total_rows * total_flowers_per_row
  let green_factor := 2
  let green_flowers := green_factor * yellow_flowers
  let yellow_and_green_flowers := yellow_flowers + green_flowers
  let red_flowers := total_flowers - yellow_and_green_flowers
  have red_flowers_eq : red_flowers = total_flowers - (yellow_flowers + green_flowers) := rfl
  show red_flowers = 42 from sorry

end wilma_red_flowers_l814_814848


namespace Q_moves_clockwise_l814_814688

-- Conditions
def on_unit_circle (P : ℝ → ℝ × ℝ) (ω : ℝ) : Prop :=
  ∀ t, P t = (Real.cos (ω * t), Real.sin (ω * t))

def Q_coordinates (P : ℝ → ℝ × ℝ) (Q : ℝ → ℝ × ℝ) : Prop :=
  ∀ t, let (x, y) := P t in Q t = (-2 * x * y, y^2 - x^2)

-- The Proof Statement
theorem Q_moves_clockwise (P Q : ℝ → ℝ × ℝ) (ω : ℝ) :
  on_unit_circle P ω →
  Q_coordinates P Q →
  ∀ t, Q t = (Real.cos (-(2 * ω) * t), Real.sin (-(2 * ω) * t)) :=
by
  intro h1 h2
  sorry

end Q_moves_clockwise_l814_814688


namespace distance_between_parallel_lines_l814_814650

-- Definitions of the lines
def line1 (x y : ℝ) (a : ℝ) : Prop := x + a * y = 1
def line2 (x y : ℝ) (a : ℝ) : Prop := a * x + y = 1

-- Definition of parallel lines
def are_parallel (a : ℝ) : Prop := - (1 / a) = -a 

-- Given condition: parallel lines, a = -1, under these conditions 
-- we need to prove the distance between the lines is √2.
theorem distance_between_parallel_lines (a : ℝ) 
  (ha1 : are_parallel a)
  (ha2 : a = -1) : 
  ∃ d : ℝ, d = Real.sqrt 2 ∧ 
           (∃ x1 y1 x2 y2 : ℝ, line1 x1 y1 a ∧ line2 x2 y2 a ∧ 
            (d = (Real.abs (-1 - 1)) / Real.sqrt (1^2 + (-1)^2))) :=
begin
  sorry
end

end distance_between_parallel_lines_l814_814650


namespace coloring_change_impossible_l814_814876

-- Define the initial coloring sequence of the 99-sided polygon
def initial_coloring : Fin 99 → Color 
| i => if i % 2 = 0 then red else if i.val = 98 then yellow else blue

-- Define the target coloring sequence
def target_coloring : Fin 99 → Color 
| i => if i % 2 = 0 then red else if i.val = 98 then blue else if i.val = 97 then yellow else blue

-- Define the proof theorem
theorem coloring_change_impossible : 
  ¬ ∃ f : (Fin 99 → Color) → (Fin 99 → Color), 
    ∀ col : Fin 99 → Color, 
    (∀ i : Fin 99, col (Fin.succ i) ≠ col i) →
    (∃ N : ℕ, (f^[N] initial_coloring) = target_coloring) := 
sorry

end coloring_change_impossible_l814_814876


namespace percentage_increase_area_of_rectangle_l814_814855

theorem percentage_increase_area_of_rectangle (L W : ℝ) :
  let L_new := 1.10 * L,
      W_new := 1.10 * W,
      A_original := L * W,
      A_new := L_new * W_new,
      percentage_increase := ((A_new - A_original) / A_original) * 100
  in percentage_increase = 21 := by
  sorry

end percentage_increase_area_of_rectangle_l814_814855


namespace jacob_needs_to_catch_l814_814918

def initial_fish_jacob : ℕ := 8
def factor_alex_catches_more_than_jacob : ℕ := 7
def fish_lost_by_alex : ℕ := 23

theorem jacob_needs_to_catch : ∃ (n : ℕ), n = 26 :=
  let initial_fish_alex := factor_alex_catches_more_than_jacob * initial_fish_jacob
  let remaining_fish_alex := initial_fish_alex - fish_lost_by_alex
  let target_fish_jacob := remaining_fish_alex + 1
  let additional_fish_needed := target_fish_jacob - initial_fish_jacob
  begin
    use additional_fish_needed,
    calc additional_fish_needed
          = target_fish_jacob - initial_fish_jacob : rfl
      ... = (remaining_fish_alex + 1) - initial_fish_jacob : rfl
      ... = ((factor_alex_catches_more_than_jacob * initial_fish_jacob - fish_lost_by_alex) + 1) - initial_fish_jacob : rfl
      ... = (56 - 23) + 1 - 8 : by norm_num
      ... = 33 + 1 - 8 : rfl
      ... = 34 - 8 : rfl
      ... = 26 : rfl
  end.

end jacob_needs_to_catch_l814_814918


namespace john_ate_steak_l814_814731

def original_steak_weight : ℝ := 30
def burned_factor : ℝ := 0.5
def eaten_percentage : ℝ := 0.80

theorem john_ate_steak (weight : ℝ) (burned : ℝ) (eaten_percent : ℝ) : 
  (weight = original_steak_weight) → 
  (burned = burned_factor) → 
  (eaten_percent = eaten_percentage) → 
  (eaten_percent * (weight * (1 - burned)) = 12) :=
begin
  sorry
end

end john_ate_steak_l814_814731


namespace sums_of_columns_contradiction_l814_814720

theorem sums_of_columns_contradiction :
  ∀ (S : Fin 10 → ℕ),
  (∀ i : Fin 9, (S i).abs_diff (S (i + 1)) = 1) →
  (∑ i, S i = 5050) →
  False := by
  intros S h_diff h_sum
  sorry

end sums_of_columns_contradiction_l814_814720


namespace christina_snakes_length_l814_814926

theorem christina_snakes_length :
  let snake1_feet := 2 in
  let snake2_inches := 16 in
  let snake3_inches := 10 in
  let feet_to_inches := 12 in
  (snake1_feet * feet_to_inches) + snake2_inches + snake3_inches = 50 := 
by 
  -- snake1_feet is 2 feet, converted to inches
  let snake1_inches := 2 * 12
  -- calculate the total length in inches
  calc 
    snake1_inches + snake2_inches + snake3_inches = 24 + 16 + 10 : by rfl
    ... = 50 : by rfl
  sorry

end christina_snakes_length_l814_814926


namespace equal_charges_at_hours_l814_814765

theorem equal_charges_at_hours :
  ∀ (h : ℕ), (55 + 35 * h = 75 + 30 * h) → h = 4 :=
begin
  assume h,
  assume eq_charges,
  sorry
end

end equal_charges_at_hours_l814_814765


namespace reduced_rates_start_l814_814879

theorem reduced_rates_start (reduced_fraction : ℝ) (total_hours : ℝ) (weekend_hours : ℝ) (weekday_hours : ℝ) 
  (start_time : ℝ) (end_time : ℝ) : 
  reduced_fraction = 0.6428571428571429 → 
  total_hours = 168 → 
  weekend_hours = 48 → 
  weekday_hours = 60 - weekend_hours → 
  end_time = 8 → 
  start_time = end_time - weekday_hours → 
  start_time = 20 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end reduced_rates_start_l814_814879


namespace find_omega_minimum_g_l814_814646

noncomputable def f (ω x : ℝ) : ℝ := 
  sin (ω * x - π / 6) + sin (ω * x - π / 2)

noncomputable def g (x : ℝ) : ℝ := 
  sqrt 3 * sin (x - π / 12)

theorem find_omega
  (h0 : 0 < ω)
  (h1 : ω < 3)
  (h2 : f ω (π / 6) = 0) :
  ω = 2 := 
sorry

theorem minimum_g
  (h0 : ∀ x ∈ Icc (-π / 4) (3 * π / 4), g x >= - (sqrt 3 / 2)) :
  ∃ x ∈ Icc (-π / 4) (3 * π / 4), g x = - (sqrt 3 / 2) :=
sorry

end find_omega_minimum_g_l814_814646


namespace total_population_eq_51b_over_40_l814_814705

variable (b g t : Nat)

-- Conditions
def boys_eq_four_times_girls (b g : Nat) : Prop := b = 4 * g
def girls_eq_ten_times_teachers (g t : Nat) : Prop := g = 10 * t

-- Statement to prove
theorem total_population_eq_51b_over_40 (b g t : Nat) 
  (h1 : boys_eq_four_times_girls b g) 
  (h2 : girls_eq_ten_times_teachers g t) : 
  b + g + t = (51 * b) / 40 := 
sorry

end total_population_eq_51b_over_40_l814_814705


namespace liz_three_pointers_l814_814757

-- Define the points scored by Liz's team in the final quarter.
def points_scored_by_liz (free_throws jump_shots three_pointers : ℕ) : ℕ :=
  free_throws * 1 + jump_shots * 2 + three_pointers * 3

-- Define the points needed to tie the game.
def points_needed_to_tie (initial_deficit points_lost other_team_points : ℕ) : ℕ :=
  points_lost + (initial_deficit - points_lost) + other_team_points

-- The total points scored by Liz from free throws and jump shots.
def liz_regular_points (free_throws jump_shots : ℕ) : ℕ :=
  free_throws * 1 + jump_shots * 2

theorem liz_three_pointers :
  ∀ (free_throws jump_shots liz_team_deficit_final quarter_deficit other_team_points liz_team_deficit_end final_deficit : ℕ),
    liz_team_deficit_final = 20 →
    free_throws = 5 →
    jump_shots = 4 →
    other_team_points = 10 →
    liz_team_deficit_end = 8 →
    final_deficit = liz_team_deficit_final - liz_team_deficit_end →
    (free_throws * 1 + jump_shots * 2 + 3 * final_deficit) = 
      points_needed_to_tie 20 other_team_points 8 →
    (3 * final_deficit) = 9 →
    final_deficit = 3 →
    final_deficit = 3 :=
by
  intros 
  try sorry

end liz_three_pointers_l814_814757


namespace unique_five_digit_integers_l814_814302

-- Define the problem conditions
def digits := [2, 2, 3, 9, 9]
def total_spots := 5
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- Compute the number of five-digit integers that can be formed
noncomputable def num_unique_permutations : Nat :=
  factorial total_spots / (factorial 2 * factorial 1 * factorial 2)

-- Proof statement
theorem unique_five_digit_integers : num_unique_permutations = 30 := by
  sorry

end unique_five_digit_integers_l814_814302


namespace S_eq_T_l814_814670

variable {ℤ : Type} [IntegralDomain ℤ]

def S : Set ℤ := {x | ∃ n : ℤ, x = 2 * n + 1}
def T : Set ℤ := {x | ∃ n : ℤ, x = 4 * n + 1 ∨ x = 4 * n - 1}

theorem S_eq_T : S = T := 
by sorry

end S_eq_T_l814_814670


namespace cube_vertex_to_diagonal_distance_l814_814301

-- Define the vertices of the cube
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def G : ℝ × ℝ × ℝ := (1, 1, 1)

-- Function to calculate the distance from a point to a line in 3D space
noncomputable def distance_point_to_line (P L1 L2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x0, y0, z0) := P in
  let (x1, y1, z1) := L1 in
  let (x2, y2, z2) := L2 in
  let num := abs ((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1) + (z1 - z0) * (z2 - z1)) in
  let denom := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2) in
  num / denom

-- Define edge length and space diagonal conditions
def edge_length := 1
def space_diagonal := distance_point_to_line (0,0,0) (0,0,0) (1,1,1)

-- Define the theorem
theorem cube_vertex_to_diagonal_distance :
  ∀ (v : ℝ × ℝ × ℝ), 
  (v = (0,0,0) ∨ v = (1,0,0) ∨ v = (1,1,0) ∨ v = (0,1,0) ∨ 
   v = (0,0,1) ∨ v = (1,0,1) ∨ v = (1,1,1) ∨ v = (0,1,1)) → 
  distance_point_to_line v A G = real.sqrt(2) / real.sqrt(3) :=
sorry

end cube_vertex_to_diagonal_distance_l814_814301


namespace initial_width_of_river_eq_50_yards_l814_814774

-- Define the conditions
def width_increase_per_meter : ℝ := 2 / 10
def rowing_speed_m_s : ℝ := 5
def time_seconds : ℝ := 30
def final_width_yards : ℝ := 80
def distance_covered_m : ℝ := rowing_speed_m_s * time_seconds

-- Lean statement to prove the width at the starting point
theorem initial_width_of_river_eq_50_yards (W : ℝ) 
  (h_eq : final_width_yards = W + width_increase_per_meter * distance_covered_m) : 
  W = 50 := by
  sorry

end initial_width_of_river_eq_50_yards_l814_814774


namespace fair_division_of_jackfruits_fair_division_of_jackfruits_proof_l814_814037

theorem fair_division_of_jackfruits 
(Renato_has_30_watermelons: ℕ)
(Leandro_has_18_watermelons: ℕ)
(Marcelo_has_24_jackfruits: ℕ)
(Marcelo_dislikes_jackfruits: Prop)
(Marcelo_shares_jackfruits: Prop)
(Renato_Leandro_Watermelons_Divided_Equally: ℕ) 
: Prop := 
Renato_has_30_watermelons = 30 → 
Leandro_has_18_watermelons = 18 → 
Marcelo_has_24_jackfruits = 24 → 
Marcelo_dislikes_jackfruits → 
Marcelo_shares_jackfruits → 
Renato_Leandro_Watermelons_Divided_Equally = 16 →
let total_watermelons := Renato_has_30_watermelons + Leandro_has_18_watermelons,
    each_person_gets := total_watermelons / 3 in
(Renato_receives_jackfruits : ℕ) (Leandro_receives_jackfruits : ℕ) :=
Renato_receives_jackfruits = Renato_has_30_watermelons * Marcelo_has_24_jackfruits / total_watermelons ∧
Leandro_receives_jackfruits = Leandro_has_18_watermelons * Marcelo_has_24_jackfruits / total_watermelons

theorem fair_division_of_jackfruits_proof 
: fair_division_of_jackfruits 30 18 24 True True 16 :=
begin
  intros,
  exact ⟨15, 9⟩
end

end fair_division_of_jackfruits_fair_division_of_jackfruits_proof_l814_814037


namespace perfect_square_of_d_l814_814031

theorem perfect_square_of_d (a b c d : ℤ) (h : d = (a + (2:ℝ)^(1/3) * b + (4:ℝ)^(1/3) * c)^2) : ∃ k : ℤ, d = k^2 :=
by
  sorry

end perfect_square_of_d_l814_814031


namespace ellipse_equation_and_slope_range_l814_814983

theorem ellipse_equation_and_slope_range
  (a b c : ℝ) (M : ℝ × ℝ) (H1 : a > 0) (H2 : b > 0) (H3 : a > b)
  (eccentricity : ℝ) (H4 : eccentricity = √3 / 2)
  (perimeter : ℝ) (H5 : perimeter = 4 + 2 * √3)
  (H6 : M = (0, 2))
  (ellipse_eq : Prop) (slope_range : Set ℝ) :
  (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ ellipse_eq = (x^2 / 4 + y^2 = 1)) ∧
  (∀ l : ℝ × ℝ → Prop, ∃ k : ℝ, (M.2 = k * M.1 + 2) ∧
  ∀ x1 x2 : ℝ, ∃ y1 y2 : ℝ, y1 = k * x1 + 2 ∧ y2 = k * x2 + 2 ∧
  (x1 * x2 + y1 * y2 > 0) ↔ (k ∈ slope_range) = 
  slope_range = {x : ℝ | (-2 < x ∧ x < -√3 / 2) ∨ (√3 / 2 < x ∧ x < 2)}) :=
sorry

end ellipse_equation_and_slope_range_l814_814983


namespace printers_finish_tasks_l814_814525

theorem printers_finish_tasks :
  ∀ (start_time_1 finish_half_time_1 start_time_2 : ℕ) (half_task_duration full_task_duration second_task_duration : ℕ),
    start_time_1 = 9 * 60 ∧
    finish_half_time_1 = 12 * 60 + 30 ∧
    half_task_duration = finish_half_time_1 - start_time_1 ∧
    full_task_duration = 2 * half_task_duration ∧
    start_time_2 = 13 * 60 ∧
    second_task_duration = 2 * 60 ∧
    start_time_1 + full_task_duration = 4 * 60 ∧
    start_time_2 + second_task_duration = 15 * 60 →
  max (start_time_1 + full_task_duration) (start_time_2 + second_task_duration) = 16 * 60 := 
by
  intros start_time_1 finish_half_time_1 start_time_2 half_task_duration full_task_duration second_task_duration
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩
  sorry

end printers_finish_tasks_l814_814525


namespace smallest_sum_l814_814260

theorem smallest_sum (x y : ℕ) (hx : x ≠ y) (hxy : (1/x:ℚ) + (1/y:ℚ) = 1/15) : x + y = 64 :=
sorry

end smallest_sum_l814_814260


namespace sum_of_cubes_lt_one_l814_814868

noncomputable def seq (x : ℝ) : ℕ → ℝ
| 0       := x
| (n + 1) := seq x n - (seq x n)^2

theorem sum_of_cubes_lt_one {x : ℝ} (h : 0 < x ∧ x < 1) :
  let s := seq x in
  ∑ i in Finset.range 99, (s i)^3 < 1 :=
sorry

end sum_of_cubes_lt_one_l814_814868


namespace minimum_diagonals_l814_814623

variables {m n r : ℕ}
def generalized_diagonal (m n : ℕ) :=
  {cells : finset (fin m × fin n) // ∀ (c1 c2 ∈ cells), c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2}

open_locale classical

theorem minimum_diagonals (m n r : ℕ) (h1 : 1 ≤ r) (h2 : r ≤ m) (h3 : m ≤ n) 
  (h4 : ∀ (i : fin m), (∃ j, (i, j) ∈ red_cells) →
                        (∀ i, ∃ j, (j, i) ∈ red_cells) →
                        ∀ i, ∃ j, (j ∈ (finset.univ : finset (fin m ∷ fin n))) :=
begin
  -- The proof will be placed here explaining why a = n (omitted for brevity)
  sorry
end

end minimum_diagonals_l814_814623


namespace triangle_calculation_l814_814711

theorem triangle_calculation (DEF : Triangle) (acute_DEF : acute DEF)
  (DR ES : Line) (T : Point) (DR_alt : is_altitude DR DEF)
  (ES_alt : is_altitude ES DEF) (T_intersect : T ∈ DR ∩ ES)
  (TR TS : ℝ) (TR_eq : TR = 3) (TS_eq : TS = 6) :
  (ER : ℝ) (RD : ℝ) (DS : ℝ) (SE : ℝ) :
  (ER * RD) - (DS * SE) = 0 := 
  sorry

end triangle_calculation_l814_814711


namespace total_pears_l814_814173

theorem total_pears (S P C : ℕ) (hS : S = 20) (hP : P = (S - S / 2)) (hC : C = (P + P / 5)) : S + P + C = 42 :=
by
  -- We state the theorem with the given conditions and the goal of proving S + P + C = 42.
  sorry

end total_pears_l814_814173


namespace transformed_triangle_area_l814_814785

theorem transformed_triangle_area
  (a b c : ℝ)
  (g : ℝ → ℝ)
  (h_domain : ∀ x, x ∈ {a, b, c} → g x ≠ 0)
  (original_triangle_area : ℝ)
  (h_area : original_triangle_area = 54)
  (h_transformed_points : ∀ x ∈ {a, b, c}, (3 * g (1.5 * x) ≠ 0)) :
  let transformed_area := 54 * 3 * (2/3)
  in transformed_area = 108 := sorry

end transformed_triangle_area_l814_814785


namespace no_mapping_D_l814_814104

def setA1 := { x : ℝ | x ≥ 0 }
def setB1 := { y : ℝ | y ≥ 0 }
def f1 (x : ℝ) : ℝ := x^2

def setA2 := { x : ℝ | x > 0 ∨ x < 0 }
def setB2 := { 1 }
def f2 (x : ℝ) : ℝ := x

def setA3 := ℝ
def setB3 := ℝ
def f3 (x : ℝ) : ℝ := 2^x

def setA4 := { 2, 3 }
def setB4 := { 4, 9 }
def f4 (x : ℕ) : ℕ := x /- Given in the problem with additional constraints -/

theorem no_mapping_D :
  ∀ (A : set ℝ) (B : set ℕ) (f : ℕ → ℕ),
    (A = {2, 3}) →
    (B = {4, 9}) →
    (f = λ x, x) →
    ¬ (∀ x ∈ A, f x ∈ B) :=
by {
    intro A B f,
    intro hA hB hf,
    rw [hA, hB, hf],
    simp,
    sorry
}

end no_mapping_D_l814_814104


namespace probability_one_from_each_l814_814535

-- Define the total number of cards
def total_cards : ℕ := 10

-- Define the number of cards from Amelia's name
def amelia_cards : ℕ := 6

-- Define the number of cards from Lucas's name
def lucas_cards : ℕ := 4

-- Define the probability that one letter is from each person's name
theorem probability_one_from_each : (amelia_cards / total_cards) * (lucas_cards / (total_cards - 1)) +
                                    (lucas_cards / total_cards) * (amelia_cards / (total_cards - 1)) = 8 / 15 :=
by
  sorry

end probability_one_from_each_l814_814535


namespace simplify_and_sum_coefficients_l814_814561

theorem simplify_and_sum_coefficients :
  let p := λ x : ℝ, x^3 + 3 * x^2 - 4 * x - 12,
      q := λ x : ℝ, x - 1,
      simplified_fun := λ x : ℝ, x^2 + 4 * x + 12,
      A := 1,
      B := 4,
      C := 12,
      D := 1
  in p[1] = 0 ∧ (∀ x, (x ≠ 1) → (p x / q x) = simplified_fun x) ∧ (A + B + C + D = 18) :=
by
  sorry

end simplify_and_sum_coefficients_l814_814561


namespace probability_of_drawing_three_white_balls_l814_814510

theorem probability_of_drawing_three_white_balls
  (total_balls white_balls black_balls: ℕ)
  (h_total: total_balls = 15)
  (h_white: white_balls = 7)
  (h_black: black_balls = 8)
  (draws: ℕ)
  (h_draws: draws = 3) :
  (Nat.choose white_balls draws / Nat.choose total_balls draws) = (7 / 91) :=
by sorry

end probability_of_drawing_three_white_balls_l814_814510


namespace event_A_plus_B_event_not_A_plus_C_event_A_and_C_event_A_and_B_plus_C_l814_814911

-- Define the events A, B, and C
def A : Prop := "A first-grade part is drawn"
def B : Prop := "A second-grade part is drawn"
def C : Prop := "A third-grade part is drawn"

-- Define the composite events in Lean

theorem event_A_plus_B :
  A ∨ B ↔ ("A first-grade part is drawn" ∨ "A second-grade part is drawn") :=
by sorry

theorem event_not_A_plus_C :
  ¬ (A ∨ C) ↔ ("A second-grade part is drawn") :=
by sorry

theorem event_A_and_C :
  A ∧ C ↔ false :=
by sorry

theorem event_A_and_B_plus_C :
  (A ∧ B) ∨ C ↔ ("A third-grade part is drawn") :=
by sorry

end event_A_plus_B_event_not_A_plus_C_event_A_and_C_event_A_and_B_plus_C_l814_814911


namespace intersection_of_sets_l814_814010

open Set

theorem intersection_of_sets :
  let A := {1, 2, 3, 4}
  let B := {x : ℝ | 1 < x ∧ x ≤ 4}
  A ∩ B = {2, 3, 4} :=
by
  let A := {1, 2, 3, 4}
  let B := {x : ℝ | 1 < x ∧ x ≤ 4}
  sorry

end intersection_of_sets_l814_814010


namespace project_hours_l814_814483

variable (K P M : ℕ)

theorem project_hours
  (h1 : P + K + M = 144)
  (h2 : P = 2 * K)
  (h3 : P = M / 3) :
  M - K = 80 :=
sorry

end project_hours_l814_814483


namespace least_positive_base_ten_number_with_eight_binary_digits_l814_814470

theorem least_positive_base_ten_number_with_eight_binary_digits : 
  ∃ n : ℕ, (n = 128) ∧ (binary_digits n = 8) :=
begin
  sorry
end

noncomputable def binary_digits (n : ℕ) : ℕ :=
  nat.log n 2 + 1

end least_positive_base_ten_number_with_eight_binary_digits_l814_814470


namespace family_reunion_cost_l814_814223

theorem family_reunion_cost :
  let people := 5 * 12
  let cans_per_person := 2
  let cans_needed := people * cans_per_person
  let cans_per_box := 10
  let boxes_needed := cans_needed / cans_per_box
  let cost_per_box := 2
  let total_cost := boxes_needed * cost_per_box
  let family_members := 6
  let cost_per_member := total_cost / family_members
  cost_per_member = 4 := by
{
  let people := 5 * 12
  let cans_per_person := 2
  let cans_needed := people * cans_per_person
  let cans_per_box := 10
  let boxes_needed := cans_needed / cans_per_box
  let cost_per_box := 2
  let total_cost := boxes_needed * cost_per_box
  let family_members := 6
  let cost_per_member := total_cost / family_members
  show cost_per_member = 4, from
    sorry
}

end family_reunion_cost_l814_814223


namespace probability_three_white_balls_l814_814508

theorem probability_three_white_balls (total_balls: ℕ) (white_balls: ℕ) (black_balls: ℕ) (drawn_balls: ℕ) 
    (h_total: total_balls = 15) (h_white: white_balls = 7) (h_black: black_balls = 8) (h_drawn: drawn_balls = 3) : 
    ((choose white_balls drawn_balls) / (choose total_balls drawn_balls) : ℚ) = 1 / 13 := 
by {
    -- Definitions and conditions come from part (a)
    -- The lean code should be able to be built successfully
    sorry
} 

end probability_three_white_balls_l814_814508


namespace fraction_is_terminating_decimal_l814_814576

noncomputable def fraction_to_decimal : ℚ :=
  58 / 160

theorem fraction_is_terminating_decimal : fraction_to_decimal = 3625 / 10000 :=
by
  sorry

end fraction_is_terminating_decimal_l814_814576


namespace sum_of_reciprocals_of_squares_roots_eq_14_3125_l814_814793

theorem sum_of_reciprocals_of_squares_roots_eq_14_3125
  (α β γ : ℝ)
  (h1 : α + β + γ = 15)
  (h2 : α * β + β * γ + γ * α = 26)
  (h3 : α * β * γ = -8) :
  (1 / α^2) + (1 / β^2) + (1 / γ^2) = 14.3125 := 
by
  sorry

end sum_of_reciprocals_of_squares_roots_eq_14_3125_l814_814793


namespace cube_volume_from_surface_area_l814_814122

theorem cube_volume_from_surface_area (A : ℝ) (h : A = 54) :
  ∃ V : ℝ, V = 27 := by
  sorry

end cube_volume_from_surface_area_l814_814122


namespace jacob_needs_to_catch_26_more_fish_l814_814917

noncomputable def Jacob_needs_to_catch (jacob_initial_fish alex_lost_fish : ℕ) :=
  let jacob_final_fish := 34 - jacob_initial_fish
  in jacob_final_fish

theorem jacob_needs_to_catch_26_more_fish :
  let jacob_initial_fish := 8 in
  let alex_factor := 7 in
  let alex_initial_fish := alex_factor * jacob_initial_fish in
  let alex_lost_fish := 23 in
  let alex_final_fish := alex_initial_fish - alex_lost_fish in
  let jacob_final_fish := alex_final_fish + 1 in
  Jacob_needs_to_catch jacob_initial_fish alex_lost_fish = 26 :=
by
  let jacob_initial_fish := 8
  let alex_initial_fish := 7 * jacob_initial_fish
  let alex_lost_fish := 23
  let alex_final_fish := alex_initial_fish - alex_lost_fish
  let jacob_final_fish := alex_final_fish + 1
  let needed := jacob_final_fish - jacob_initial_fish
  have : needed = 26 := by sorry
  exact this

end jacob_needs_to_catch_26_more_fish_l814_814917


namespace count_abundant_numbers_less_than_50_l814_814316

open_locale classical

def is_abundant (n : ℕ) : Prop :=
  ∑ m in finset.filter (λ m, m ∣ n ∧ m < n) (finset.range n), m > n

theorem count_abundant_numbers_less_than_50 :
  (finset.filter (λ n, is_abundant n) (finset.range 50)).card = 9 :=
by {
  sorry
}

end count_abundant_numbers_less_than_50_l814_814316


namespace collinear_A1B1C1_l814_814395

-- Declare the points A, B, C, A1, B1, C1
variables (A B C A1 B1 C1 : Type)

-- Conditions
-- Assume A, B, C are collinear
axiom collinear_ABC : collinear A B C

-- Lines paralllel relationships
axiom parallel_AB1_BA1 : ∥AB1∥ ∥BA1∥
axiom parallel_AC1_CA1 : ∥AC1∥ ∥CA1∥
axiom parallel_BC1_CB1 : ∥BC1∥ ∥CB1∥

-- The goal is to prove that A1, B1, and C1 are collinear
theorem collinear_A1B1C1 : collinear A1 B1 C1 :=
sorry

end collinear_A1B1C1_l814_814395


namespace least_positive_base_ten_number_with_eight_binary_digits_l814_814472

theorem least_positive_base_ten_number_with_eight_binary_digits : 
  ∃ n : ℕ, (n = 128) ∧ (binary_digits n = 8) :=
begin
  sorry
end

noncomputable def binary_digits (n : ℕ) : ℕ :=
  nat.log n 2 + 1

end least_positive_base_ten_number_with_eight_binary_digits_l814_814472


namespace probability_nan_kai_l814_814923

theorem probability_nan_kai :
  let total_outcomes := Nat.choose 6 4
  let successful_outcomes := Nat.choose 4 4
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 1 / 15 :=
by
  sorry

end probability_nan_kai_l814_814923


namespace average_speed_problem_l814_814513

noncomputable def average_speed (d₁ d₂ d₃ d₄ t₁ t₂ t₃ t₄ : ℝ) : ℝ :=
  (d₁ + d₂ + d₃ + d₄) / (t₁ + t₂ + t₃ + t₄)

theorem average_speed_problem :
  average_speed 30 40 37.5 7 (30 / 35) (40 / 55) 0.5 (10 / 60) = 51 :=
by
  -- skip the proof
  sorry

end average_speed_problem_l814_814513


namespace length_of_DB_l814_814339

theorem length_of_DB (angle_ABC_right : ∀ A B C : Type, angle ABC = 90)
  (angle_ADB_right : ∀ A D B : Type, angle ADB = 90)
  (AC : ℝ) (AD : ℝ) (DB : ℝ) (h1 : AC = 25) (h2 : AD = 7) :
  DB = 3 * Real.sqrt 14 :=
sorry

end length_of_DB_l814_814339


namespace jack_salt_evaporation_l814_814357

/-- Calculate the volume of salt in milliliters from a given volume of seawater and salt concentration. --/
def volume_of_salt_in_ml (seawater_volume : ℝ) (salt_percentage : ℝ) : ℝ :=
  seawater_volume * salt_percentage * 1000

theorem jack_salt_evaporation :
  volume_of_salt_in_ml 2 0.20 = 400 :=
by 
sory

end jack_salt_evaporation_l814_814357


namespace sin_half_angle_product_less_than_quarter_l814_814767

theorem sin_half_angle_product_less_than_quarter 
  (A B C : ℝ) 
  (h₁ : A + B + C = π) 
  (h₂ : 0 < A) 
  (h₃ : 0 < B) 
  (h₄ : 0 < C) : 
  sin (A / 2) * sin (B / 2) * sin (C / 2) < 1 / 4 :=
by
  sorry

end sin_half_angle_product_less_than_quarter_l814_814767


namespace f_at_pi_over_2_eq_1_l814_814745

noncomputable def f := λ x ω b, sin (ω * x + π / 4) + b

theorem f_at_pi_over_2_eq_1 :
  ∀ (ω b T : ℝ),
  (0 < ω) →
  (T = (2 * π) / ω) →
  (2 * π / 3 < T ∧ T < π) →
  (∀ x, f x ω 2 = f (3 * π / 2 - x) ω 2) → 
  f (π / 2) ω 2 = 1 := 
by
  intros ω b T hω hT hTbounds hsym
  sorry

end f_at_pi_over_2_eq_1_l814_814745


namespace value_of_n_l814_814674

theorem value_of_n (n : ℕ) (h : sqrt (10 + n) = 9) : n = 71 :=
by
  sorry

end value_of_n_l814_814674


namespace degree_poly_is_22_l814_814182

-- Definition of the polynomial (x^2 + y)^5 * (x^3 + y)^4
def poly := (x^2 + y)^5 * (x^3 + y)^4

-- Definition to extract the degree of a polynomial in terms of x
noncomputable def degree_in_x (p : ℚ[X]) : ℕ :=
  p.natDegree

-- Statement to be proved
theorem degree_poly_is_22 : degree_in_x poly = 22 :=
sorry

end degree_poly_is_22_l814_814182


namespace basic_printer_total_price_l814_814811

theorem basic_printer_total_price (C P : ℝ) (hC : C = 1500) (hP : P = (1/3) * (C + 500 + P)) : C + P = 2500 := 
by
  sorry

end basic_printer_total_price_l814_814811


namespace volume_rectangular_solid_l814_814445

theorem volume_rectangular_solid
  (a b c : ℝ) 
  (h1 : a * b = 12)
  (h2 : b * c = 8)
  (h3 : a * c = 6) :
  a * b * c = 24 :=
sorry

end volume_rectangular_solid_l814_814445


namespace clock_strikes_10_seconds_l814_814355

theorem clock_strikes_10_seconds:
  ∀ t : ℕ, ∀ n : ℕ, ∀ intervals : ℕ,
    (t = 7 ∧ n = 8 ∧ intervals = n - 1 ∧ t = intervals * 1)
    → (intervals = 9 ∧ n = 10 ∧ t = intervals * 1)
    → t = 9 := 
by
  assume (t : ℕ) (n : ℕ) (intervals : ℕ)
  intro h1 h2
  sorry

end clock_strikes_10_seconds_l814_814355


namespace added_number_is_five_l814_814895

def original_number := 19
def final_resultant := 129
def doubling_expression (x : ℕ) (y : ℕ) := 3 * (2 * x + y)

theorem added_number_is_five:
  ∃ y, doubling_expression original_number y = final_resultant ↔ y = 5 :=
sorry

end added_number_is_five_l814_814895


namespace roots_equal_and_real_l814_814904

noncomputable def quadratic_roots (a c : ℝ) : ℝ × ℝ :=
  let b := -2 * real.sqrt 2
  let Δ := b * b - 4 * a * c
  let root := (-b + real.sqrt Δ) / (2 * a)
  (root, root)

theorem roots_equal_and_real (a c : ℝ) (h : 4 * a * c = 8) : 
  let (x1, x2) := quadratic_roots a c in x1 = x2 ∧ (∃ r : ℝ, x1 = r) :=
by
  sorry

end roots_equal_and_real_l814_814904


namespace degree_of_polynomial_product_l814_814046

-- Conditions that we have as per the problem
variables {f g : polynomial ℝ}
hypothesis h_f : f.degree = 4
hypothesis h_g : g.degree = 5

-- We need to state the main theorem to be proved.
theorem degree_of_polynomial_product (h_f : f.degree = 4) (h_g : g.degree = 5) : 
    (polynomial.degree (f.comp (X ^ 2)) + polynomial.degree (g.comp (X ^ 3))) = 23 :=
sorry

end degree_of_polynomial_product_l814_814046


namespace find_y_l814_814349

theorem find_y (EF GH : ℝ)
  (h1 : line EF ∧ line GH)
  (h2 : ∠EPF = 180)
  (h3 : ∠EPQ = 70)
  (h4 : ∠QPR = 40)
  (h5 : ∠GQH = 150) :
  ∠PQR = 80 := by sorry

end find_y_l814_814349


namespace balance_balls_l814_814392

theorem balance_balls :
  ∀ (R B O P : ℝ),
  (R = 2 * B) →
  (O = 2.5 * B) →
  (P = (4 / 3) * B) →
  (5 * R + 3 * O + 4 * P = (68.5 / 3) * B) :=
begin
  intros R B O P hR hO hP,
  calc
    5 * R + 3 * O + 4 * P = 5 * (2 * B) + 3 * (2.5 * B) + 4 * ((4 / 3) * B) : by rw [hR, hO, hP]
    ... = 10 * B + 7.5 * B + 4 * (4 / 3 * B) : by ring
    ... = 10 * B + 7.5 * B + 16 / 3 * B : by ring
    ... = ((10 + 7.5) + 16 / 3) * B : by ring
    ... = ((17.5) + 16 / 3) * B : by ring
    ... = (52.5 / 3 + 16 / 3) * B : by field_simp [52.5]
    ... = (68.5 / 3) * B : by ring
end

end balance_balls_l814_814392


namespace minimum_value_of_f_pos_reals_l814_814600

def f (x : ℝ) : ℝ :=
  x + x / (x^2 + 1) + x * (x + 4) / (x^2 + 2) + 2 * (x + 2) / (x * (x^2 + 2))

theorem minimum_value_of_f_pos_reals (x : ℝ) (hx : 0 < x) : 
  ∃ c : ℝ, (∀ y : ℝ, (y > 0) → f y ≥ c) ∧ c = 5 :=
sorry

end minimum_value_of_f_pos_reals_l814_814600


namespace curve_circle_intersection_l814_814637

theorem curve_circle_intersection (k : ℝ) :
    let C := {p : ℝ × ℝ | (p.2 - k * p.1) * (p.2 - k * p.1 + 7 / 4) = 0}
    let M := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 1}
    if k < 0 ∨ k > 35 / 12 then
        C ∩ M = ∅
    else if 0 < k ∧ k < 3 / 4 then
        ∃ A B ∈ C ∩ M, A ≠ B ∧ ∀ C ∈ C ∩ M, C = A ∨ C = B
    else if k = 3 / 4 then
        ∃ A ∈ C ∩ M, ∀ B ∈ C ∩ M, B = A
    else if 3 / 4 < k ∧ k < 4 / 3 then
        ∃ A B C D ∈ C ∩ M, A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ 
        ∀ E ∈ C ∩ M, E = A ∨ E = B ∨ E = C ∨ E = D
    else
        false :=
by
  sorry

end curve_circle_intersection_l814_814637


namespace find_bounds_of_f_l814_814952

-- Define the function f
def f (x y : ℝ) : ℝ := y - 2 * x

-- Conditions given in the problem
def condition1 (x y : ℝ) : Prop := x ≠ y ∧ x ≥ 0 ∧ y ≥ 0
def condition2 (x y : ℝ) : Prop := (x^2 + y^2) / (x + y) ≤ 4

theorem find_bounds_of_f : 
  ∃ (a b : ℝ), (∀ (x y : ℝ), condition1 x y → condition2 x y → f x y ≥ a ∧ f x y ≤ b) ∧ 
  a = -2 - 2 * √10 ∧ b = -2 + 2 * √10 := 
by
  sorry

end find_bounds_of_f_l814_814952


namespace a_n_correct_T_n_correct_l814_814280

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b (n : ℕ) : ℕ := (2 * n - 1) * 2 ^ n

noncomputable def T (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ (n + 1) + 6

axiom Sn (n : ℕ) : ℕ
axiom h1 : ∀ n, 4 * Sn n = (2 * n - 1) * a (n + 1) + 1
axiom h2 : a 1 = 1

theorem a_n_correct : ∀ n, a n = 2 * n - 1 :=
sorry

theorem T_n_correct : ∀ n, (∑ i in Finset.range n, b (i + 1)) = T n :=
sorry

end a_n_correct_T_n_correct_l814_814280


namespace probability_three_white_balls_l814_814507

theorem probability_three_white_balls (total_balls: ℕ) (white_balls: ℕ) (black_balls: ℕ) (drawn_balls: ℕ) 
    (h_total: total_balls = 15) (h_white: white_balls = 7) (h_black: black_balls = 8) (h_drawn: drawn_balls = 3) : 
    ((choose white_balls drawn_balls) / (choose total_balls drawn_balls) : ℚ) = 1 / 13 := 
by {
    -- Definitions and conditions come from part (a)
    -- The lean code should be able to be built successfully
    sorry
} 

end probability_three_white_balls_l814_814507


namespace non_zero_real_values_unique_solution_l814_814473

theorem non_zero_real_values_unique_solution 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h_eq : (5 * x) ^ 10 = (10 * y) ^ 5 - 25 * x) : 
  x = 1 / 5 ∧ y = 1 := 
sorry

end non_zero_real_values_unique_solution_l814_814473


namespace problem_sin_cos_l814_814633

theorem problem_sin_cos (a : ℝ) :
  (∀ x, f x = 2 * a * sin x * cos x + cos (2 * x)) → 
  f (π / 4) = 1 →
  a = 1 ∧ ∀ x, 0 < x → x < π → f (x + 2 * π) = f x ∧ 
  ∃ I, I = Set.Ico (π / 8) (5 * π / 8) ∧ 
  ∀ x, x ∈ I → f' x < 0 :=
by sorry

end problem_sin_cos_l814_814633


namespace part1_daily_sales_profit_final_max_daily_sales_profit_l814_814418

-- Conditions from part (a)
def original_selling_price : ℚ := 30
def cost_price : ℚ := 15
def original_sales_volume : ℚ := 60
def sales_increase_per_yuan : ℚ := 10

-- Part (1): Daily sales profit if the price is reduced by 2 yuan
def new_selling_price1 : ℚ := original_selling_price - 2
def new_sales_volume1 : ℚ := original_sales_volume + (2 * sales_increase_per_yuan)
def profit_per_kilogram1 : ℚ := new_selling_price1 - cost_price
def daily_sales_profit1 : ℚ := profit_per_kilogram1 * new_sales_volume1

theorem part1_daily_sales_profit : daily_sales_profit1 = 1040 := by
  sorry

-- Part (2): Maximum daily sales profit and corresponding selling price
def selling_price_at_max_profit : ℚ := 51 / 2

def daily_profit (x : ℚ) : ℚ :=
  (x - cost_price) * (original_sales_volume + (original_selling_price - x) * sales_increase_per_yuan)

theorem final_max_daily_sales_profit :
  (∀ x : ℚ, daily_profit x ≤ daily_profit selling_price_at_max_profit) ∧ daily_profit selling_price_at_max_profit = 1102.5 := by
  sorry

end part1_daily_sales_profit_final_max_daily_sales_profit_l814_814418


namespace no_rain_either_day_l814_814708

noncomputable def P_A := 0.62
noncomputable def P_B := 0.54
noncomputable def P_A_and_B := 0.44
noncomputable def P_A_or_B := P_A + P_B - P_A_and_B -- Applying Inclusion-Exclusion principle.
noncomputable def P_A_and_B_complement := 1 - P_A_or_B -- Complement of P(A ∪ B).

theorem no_rain_either_day :
  P_A_and_B_complement = 0.28 :=
by
  unfold P_A_and_B_complement P_A_or_B
  unfold P_A P_B P_A_and_B
  simp
  sorry

end no_rain_either_day_l814_814708


namespace problem_c_l814_814375

noncomputable def M (a b : ℝ) := (a^4 + b^4) * (a^2 + b^2)
noncomputable def N (a b : ℝ) := (a^3 + b^3) ^ 2

theorem problem_c (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_neq : a ≠ b) : M a b > N a b := 
by
  -- Proof goes here
  sorry

end problem_c_l814_814375


namespace volume_less_than_1200_l814_814106

def volume (x : ℕ) : ℕ := (x + 3) * (x^2 - 1) * (x^2 + 9)

theorem volume_less_than_1200 {x : ℕ} : 
  ∃ s, s = { x // 1 ≤ x ∧ x ≤ 3 } ∧ ∀ y ∈ s, volume (y) < 1200 :=
by sorry

end volume_less_than_1200_l814_814106


namespace valid_bead_count_l814_814162

structure Bead :=
  (shape : String)   -- "cube" or "pyramid"
  (color : String)   -- colors: "green", "blue", "red"

inductive Shape 
| Cube
| Pyramid 

inductive Color 
| Green
| Blue
| Red

def is_solution (n : ℕ) : Prop :=
  ∃ (b : fin n → Bead), 
    (∀ i, b (i + 1) % n).shape ≠ (b i).shape ∧ (b (i + 1) % n).color ≠ (b i).color) ∧
    set.of_list (list.of_fn b).to_finset = { Bead.mk Shape.Cube Color.Green, Bead.mk Shape.Cube Color.Blue, Bead.mk Shape.Pyramid Color.Red, Bead.mk Shape.Pyramid Color.Blue }

theorem valid_bead_count :
  ∀ n : ℕ, is_solution n → (n = 8 ∨ n = 10) :=
begin
  sorry
end

end valid_bead_count_l814_814162


namespace inequality_true_l814_814320

theorem inequality_true (a b : ℝ) (h : a > b) (h_pos : b > 0) : 1.5^a > 1.5^b := by
  sorry

end inequality_true_l814_814320


namespace min_xy_min_x_plus_y_l814_814028

theorem min_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : xy ≥ 36 :=
sorry  

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : x + y ≥ 16 :=
sorry

end min_xy_min_x_plus_y_l814_814028


namespace max_dot_product_value_l814_814966

noncomputable def max_dot_product (OA OB OC OP : ℝ × ℝ × ℝ)
  (hOA_unit : |OA| = 1)
  (hOB_unit : |OB| = 1)
  (hOC_unit : |OC| = 1)
  (hOA_OB_orth : OA.1 * OB.1 + OA.2 * OB.2 + OA.3 * OB.3 = 0)
  (hOA_OC_orth : OA.1 * OC.1 + OA.2 * OC.2 + OA.3 * OC.3 = 0)
  (hOB_OC_angle : acos ((OB.1 * OC.1 + OB.2 * OC.2 + OB.3 * OC.3) / (1 * 1)) = real.pi / 3)
  (hOP_unit : sqrt (OP.1^2 + OP.2^2 + OP.3^2) = 1)
  (ineq1 : abs (OP.1 * OC.1 + OP.2 * OC.2 + OP.3 * OC.3) ≤ abs (OP.1 * OB.1 + OP.2 * OB.2 + OP.3 * OB.3))
  (ineq2 : abs (OP.1 * OB.1 + OP.2 * OB.2 + OP.3 * OB.3) ≤ abs (OP.1 * OA.1 + OP.2 * OA.2 + OP.3 * OA.3)) :
  ℝ :=
  sorry

theorem max_dot_product_value (OA OB OC OP : ℝ × ℝ × ℝ)
  (hOA_unit : |OA| = 1)
  (hOB_unit : |OB| = 1)
  (hOC_unit : |OC| = 1)
  (hOA_OB_orth : OA.1 * OB.1 + OA.2 * OB.2 + OA.3 * OB.3 = 0)
  (hOA_OC_orth : OA.1 * OC.1 + OA.2 * OC.2 + OA.3 * OC.3 = 0)
  (hOB_OC_angle : acos ((OB.1 * OC.1 + OB.2 * OC.2 + OB.3 * OC.3) / (1 * 1)) = real.pi / 3)
  (hOP_unit : sqrt (OP.1^2 + OP.2^2 + OP.3^2) = 1)
  (ineq1 : abs (OP.1 * OC.1 + OP.2 * OC.2 + OP.3 * OC.3) ≤ abs (OP.1 * OB.1 + OP.2 * OB.2 + OP.3 * OB.3))
  (ineq2 : abs (OP.1 * OB.1 + OP.2 * OB.2 + OP.3 * OB.3) ≤ abs (OP.1 * OA.1 + OP.2 * OA.2 + OP.3 * OA.3)) :
  max_dot_product OA OB OC OP hOA_unit hOB_unit hOC_unit hOA_OB_orth hOA_OC_orth hOB_OC_angle hOP_unit ineq1 ineq2 = (sqrt 21) / 7 := sorry

end max_dot_product_value_l814_814966


namespace intersection_points_of_C_and_circle_l814_814629

theorem intersection_points_of_C_and_circle :
  let C := {p : ℝ × ℝ | p.snd ^ 2 = 8 * p.fst} ∪ {p : ℝ × ℝ | p.fst = -2}
  let circle := {p : ℝ × ℝ | (p.fst + 3) ^ 2 + p.snd ^ 2 = 16}
  (C ∩ circle).card = 4 :=
by
  sorry

end intersection_points_of_C_and_circle_l814_814629


namespace speed_in_still_water_l814_814140

variables {v_m v_s : ℝ}

-- Conditions from the problem
def downstream_condition : Prop := (v_m + v_s = 15)
def upstream_condition : Prop := (v_m - v_s = 6)

theorem speed_in_still_water :
  downstream_condition ∧ upstream_condition → v_m = 10.5 :=
by
  sorry

end speed_in_still_water_l814_814140


namespace valid_necklace_length_l814_814155

def bead_shape := {cube, pyramid}
def bead_color := {green, blue, red}

structure bead :=
(shape : bead_shape)
(color : bead_color)

def is_valid_necklace (necklace : list bead) : Prop :=
all_different_adjacent_shapes_and_colors necklace ∧
necklace.count_in_necklace bead (shape part shape) (color part color)

theorem valid_necklace_length (n : ℕ):
    (n = 8 ∨ n = 10) ↔ valid_necklace necklace :=
begin
  sorry -- proof of the theorem
end

end valid_necklace_length_l814_814155


namespace total_stories_l814_814360

theorem total_stories (h1: ∀ n < 10, height n = 12) 
                      (h2: ∀ n ≥ 10, height n = 15) 
                      (total_height: (Σ n : Fin 20, height n) = 270) : 
                      10 + ((270 - (10 * 12)) / 15) = 20 :=
by
  sorry

end total_stories_l814_814360


namespace barium_hydroxide_formation_l814_814682

theorem barium_hydroxide_formation 
  (moles_H₂O : ℕ) (moles_BaO : ℕ) 
  (h_stoich : ∀ (n : ℕ), (n ≤ moles_H₂O) ∧ (n ≤ moles_BaO) → n = 3) :
  (moles_H₂O = 5) → 
  (moles_BaO = 3) → 
  ∃ (moles_Ba(OH)₂ : ℕ), (moles_Ba(OH)₂ = 3) ∧ (moles_BaO ≤ moles_H₂O) := 
sorry

end barium_hydroxide_formation_l814_814682


namespace equal_perimeters_of_sub_triangles_l814_814354

theorem equal_perimeters_of_sub_triangles (perimeter_large_triangle : ℝ) (n : ℕ) (equal_small_triangles : ∀ (i : ℕ), i < n → ℝ) :
  (perimeter_large_triangle = 120) →
  (n = 9) →
  (∀ (i j : ℕ), i < n → j < n → equal_small_triangles i = equal_small_triangles j) →
  equal_small_triangles 0 = 40 :=
by
  intros h1 h2 h3
  -- placeholder for the actual proof
  sorry

end equal_perimeters_of_sub_triangles_l814_814354


namespace three_segments_form_triangle_l814_814909

def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem three_segments_form_triangle :
  ¬ can_form_triangle 2 3 6 ∧
  ¬ can_form_triangle 2 4 6 ∧
  ¬ can_form_triangle 2 2 4 ∧
    can_form_triangle 6 6 6 :=
by
  repeat {sorry}

end three_segments_form_triangle_l814_814909


namespace repeating_decimal_to_fraction_l814_814571

theorem repeating_decimal_to_fraction : (∃ x : ℚ, x = 0.3 + x * (1 / 10^3 - 1)) ∧ (0.3 : ℚ = 3 / 10) → 0.3 ≈ 0.3\overline{045} = 383 / 1110 := by
  sorry

end repeating_decimal_to_fraction_l814_814571


namespace loss_percentage_is_25_l814_814054

variables (C S : ℝ)
variables (h : 30 * C = 40 * S)

theorem loss_percentage_is_25 (h : 30 * C = 40 * S) : ((C - S) / C) * 100 = 25 :=
by
  -- proof skipped
  sorry

end loss_percentage_is_25_l814_814054


namespace max_tg_ctg_48_l814_814105

open Real

theorem max_tg_ctg_48 (a b c d : ℝ) :
  (a = tan 48 * π / 180 + cot 48 * π / 180) →
  (b = sin 48 * π / 180 + cos 48 * π / 180) →
  (c = tan 48 * π / 180 + cos 48 * π / 180) →
  (d = cot 48 * π / 180 + sin 48 * π / 180) →
  (a > b ∧ a > c ∧ a > d) :=
by
  intro ha hb hc hd
  exact sorry

end max_tg_ctg_48_l814_814105


namespace sufficient_but_not_necessary_condition_l814_814750

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition (h : x > 1) : x > 0 :=
by
  sorry

end sufficient_but_not_necessary_condition_l814_814750


namespace fantasy_birthday_problem_l814_814704

theorem fantasy_birthday_problem : 
  ∀ (people : ℕ) (months : ℕ), people = 60 ∧ months = 10 → 
  ∃ n, (∀ distribution : (fin people → fin months), 
  ∃ (m : fin months), (∃ (b : fin people → fin people.succ), 
  b m = m → b m = (λ d, 6))) := 
  sorry

end fantasy_birthday_problem_l814_814704


namespace calculate_fraction_l814_814546

theorem calculate_fraction : (1 / (1 + 1 / (4 + 1 / 5))) = (21 / 26) :=
by
  sorry

end calculate_fraction_l814_814546


namespace count_of_abundant_numbers_less_than_50_l814_814307

-- Define the proper divisors of a number
def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d => d < n ∧ n % d = 0)

-- Define what it means for a number to be abundant
def is_abundant (n : ℕ) : Prop :=
  (proper_divisors n).sum > n

-- Define the set of numbers less than 50 which are abundant
def abundant_numbers_less_than_50 : List ℕ :=
  (List.range 50).filter is_abundant

-- The theorem stating the number of abundant numbers less than 50
theorem count_of_abundant_numbers_less_than_50 : abundant_numbers_less_than_50.length = 9 := 
  sorry

end count_of_abundant_numbers_less_than_50_l814_814307


namespace pattyCoinsValue_l814_814025

def totalCoins (q d : ℕ) : Prop := q + d = 30
def originalValue (q d : ℕ) : ℝ := 0.25 * q + 0.10 * d
def swappedValue (q d : ℕ) : ℝ := 0.10 * q + 0.25 * d
def valueIncrease (q : ℕ) : Prop := swappedValue q (30 - q) - originalValue q (30 - q) = 1.20

theorem pattyCoinsValue (q d : ℕ) (h1 : totalCoins q d) (h2 : valueIncrease q) : originalValue q d = 4.65 := 
by
  sorry

end pattyCoinsValue_l814_814025


namespace gcd_ten_factorial_twelve_factorial_l814_814605

/-- Define factorials -/
def factorial : Nat → Nat
| 0     => 1
| (n+1) => (n+1) * factorial n

/-- Define gcd function (could use existing gcd definition from Mathlib) -/
def gcd (a b : Nat) : Nat :=
sorry -- (Mathlib already has a gcd definition, we'll use it.)

theorem gcd_ten_factorial_twelve_factorial :
  gcd (factorial 10) (factorial 12) = factorial 10 :=
by
  sorry

end gcd_ten_factorial_twelve_factorial_l814_814605


namespace lcm_fractions_l814_814462

theorem lcm_fractions (x : ℕ) (hx : x > 0) :
  lcm (1 / (2 * x)) (lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (12 * x)))) = 1 / (12 * x) :=
sorry

end lcm_fractions_l814_814462


namespace find_m_n_l814_814348

-- Definitions for the vectors and magnitudes
variables (OA OB OC : ℝ → vector ℝ) (m n : ℝ)

-- Given conditions
def OA_mag : Prop := ‖OA 1‖ = 2
def OB_mag : Prop := ‖OB 1‖ = 3
def OC_mag : Prop := ‖OC 1‖ = 2 * sqrt 5
def tan_AOC : Prop := real.tan (vector.angle (OC 1) (OA 1)) = 3
def angle_BOC : Prop := vector.angle (OB 1) (OC 1) = real.pi / 3

-- Statement that OC can be expressed as a linear combination of OA and OB
def OC_as_linear_combination (m n : ℝ) : Prop :=
  OC 1 = m • (OA 1) + n • (OB 1)

theorem find_m_n :
  OA_mag OA → OB_mag OB → OC_mag OC → tan_AOC OC OA → angle_BOC OB OC →
  ∃ m n, OC_as_linear_combination m n ∧ m = 9 / 8 ∧ n = - sqrt 10 / 3 :=
by
  sorry

end find_m_n_l814_814348


namespace die_top_face_points_l814_814136

theorem die_top_face_points (initial_top_points : ℕ) (opposite_sum : ℕ) (final_top_points : ℕ) :
  initial_top_points = 2 → opposite_sum = 7 → final_top_points = 5 :=
by
  intros h1 h2
  -- Definitions based on the conditions
  let bottom_face_points := opposite_sum - initial_top_points
  have h3 : bottom_face_points = 5 := by simp [h1, h2]
  -- Final conclusion based on the reverse rolling logic
  have h4 : final_top_points = bottom_face_points := by simp [h3]
  exact h4
  sorry

end die_top_face_points_l814_814136


namespace abundant_numbers_less_than_50_number_of_abundant_numbers_less_than_50_l814_814312

def sum_of_proper_factors (n : Nat) : Nat :=
  (List.range n).filter (λ i => i > 0 ∧ n % i = 0).sum

def is_abundant (n : Nat) : Prop :=
  sum_of_proper_factors n > n

def abundant_numbers_below (n : Nat) : List Nat :=
  (List.range n).filter is_abundant

theorem abundant_numbers_less_than_50 : 
  abundant_numbers_below 50 = [12, 18, 20, 24, 30, 36, 40, 42, 48] := 
by sorry

theorem number_of_abundant_numbers_less_than_50 : 
  List.length (abundant_numbers_below 50) = 9 := 
by sorry

end abundant_numbers_less_than_50_number_of_abundant_numbers_less_than_50_l814_814312


namespace last_three_digits_of_5_power_odd_l814_814429

theorem last_three_digits_of_5_power_odd (n : ℕ) (h : n % 2 = 1) : (5 ^ n) % 1000 = 125 :=
sorry

end last_three_digits_of_5_power_odd_l814_814429


namespace probability_three_white_balls_l814_814502

open Nat

def totalWaysToDrawThreeBalls : ℕ := choose 15 3
def waysToDrawThreeWhiteBalls : ℕ := choose 7 3

theorem probability_three_white_balls :
  (waysToDrawThreeWhiteBalls : ℚ) / (totalWaysToDrawThreeBalls : ℚ) = 1 / 13 := 
sorry

end probability_three_white_balls_l814_814502


namespace chocolate_bar_winning_for_Alice_l814_814907

theorem chocolate_bar_winning_for_Alice (m n : ℕ) (hm : m ≥ 2) : 
  ∃ (winning_strategy : ℕ × ℕ → Prop), winning_strategy (1, 1) :=
by
  -- Assume the conditions
  assume m n : ℕ,
  assume hm : m ≥ 2,
  have h1 : true := trivial, -- Adding a trivial true condition to avoid unused variable warning
  sorry

end chocolate_bar_winning_for_Alice_l814_814907


namespace segment_triangle_inequality_l814_814154

theorem segment_triangle_inequality (a : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1 / 10) :
  (∀ l : Fin 11 → ℝ, (∀ i, 0 ≤ l i) ∧ (∀ i, l i ≤ a) ∧ (∑ i, l i = 1) →
  (∀ (x y z : Fin 11), l x + l y > l z ∧ l y + l z > l x ∧ l z + l x > l y))
  ∧ (1 / 11 ≤ a ∧ a < 1 / 10) := 
sorry

end segment_triangle_inequality_l814_814154


namespace company_spent_on_supplies_l814_814891

noncomputable def lawnmower_blades_cost (quantity price_per_unit : ℝ) : ℝ := quantity * price_per_unit
noncomputable def weed_eater_string_cost (quantity price_per_unit : ℝ) : ℝ := quantity * price_per_unit
noncomputable def fuel_cost (quantity price_per_unit : ℝ) : ℝ := quantity * price_per_unit
noncomputable def trash_bags_cost (quantity price_per_unit : ℝ) : ℝ := quantity * price_per_unit
noncomputable def gloves_cost (quantity price_per_unit : ℝ) : ℝ := quantity * price_per_unit
noncomputable def goggles_cost (quantity price_per_unit : ℝ) : ℝ := quantity * price_per_unit

noncomputable def total_cost (lawnmower_blades weed_eater_string fuel trash_bags gloves goggles : ℝ) : ℝ :=
  lawnmower_blades + weed_eater_string + fuel + trash_bags + gloves + goggles

noncomputable def apply_discount (initial_cost discount_percent : ℝ) : ℝ :=
  initial_cost * (1 - discount_percent / 100)

noncomputable def apply_sales_tax (amount tax_percent : ℝ) : ℝ :=
  amount * (1 + tax_percent / 100)

theorem company_spent_on_supplies : 
  let lawnmower_blades_initial := lawnmower_blades_cost 4 8 in
  let weed_eater_string_initial := weed_eater_string_cost 2 7 in
  let fuel_initial := fuel_cost 1 4 in
  let trash_bags_initial := trash_bags_cost 1 5 in
  let gloves_initial := gloves_cost 5 3 in
  let goggles_initial := goggles_cost 3 12 in
  
  let total_initial := total_cost lawnmower_blades_initial weed_eater_string_initial fuel_initial trash_bags_initial gloves_initial goggles_initial in
  let total_after_general_discount := apply_discount total_initial 10 in
  let lawnmower_blades_discount := if lawnmower_blades_initial > 25 then 15 else 0 in
  let lawnmower_blades_final := apply_discount lawnmower_blades_initial lawnmower_blades_discount in
  let goggles_final := goggles_cost 2 12 in
  
  let specific_discounts := (lawnmower_blades_initial - lawnmower_blades_final) + (goggles_initial - goggles_final) in
  let new_total_after_discounts := total_after_general_discount - specific_discounts in
  let final_total := apply_sales_tax new_total_after_discounts 6 in
  
  final_total = 83.32 :=
by
  sorry

end company_spent_on_supplies_l814_814891


namespace number_of_teenagers_l814_814890

theorem number_of_teenagers (total_bill: ℝ) (cost_per_person: ℝ) (gratuity: ℝ) (H1: total_bill = 840) 
  (H2: cost_per_person = 100) (H3: gratuity = 0.20) : ℕ :=
let n := (total_bill/((1 + gratuity) * cost_per_person)) in nat_ceil n

end number_of_teenagers_l814_814890


namespace sum_f_from_1_to_2015_l814_814973

noncomputable def f (x : ℝ) : ℝ :=
if -3 ≤ x ∧ x < -1 then -(x + 2)^2 
else if -1 ≤ x ∧ x < 3 then x 
else sorry -- periodicity and further evaluation not directly encompassed here

theorem sum_f_from_1_to_2015 :
  (∑ i in (finset.range 2015).filter (λ i, 1 ≤ i + 1), f (i + 1)) = 336 :=
sorry

end sum_f_from_1_to_2015_l814_814973


namespace trays_from_first_table_l814_814039

theorem trays_from_first_table (trays_per_trip trips trays_from_second: ℕ) (h1: trays_per_trip = 4) (h2: trips = 3) (h3: trays_from_second = 2) :
  (trays_per_trip * trips) - trays_from_second = 10 :=
by
  rw [h1, h2]
  rw [nat.cast_zero, nat.cast_one, nat.cast_mul, nat.cast_tsub, nat.cast_add]
  rw [show 4 * 3 = 12, by norm_num]
  rw [show 12 - 2 = 10, by norm_num]
  sorry

end trays_from_first_table_l814_814039


namespace oil_output_per_capita_l814_814865

theorem oil_output_per_capita 
  (total_oil_output_russia : ℝ := 13737.1 * 100 / 9)
  (population_russia : ℝ := 147)
  (population_non_west : ℝ := 6.9)
  (oil_output_non_west : ℝ := 1480.689)
  : 
  (55.084 : ℝ) = 55.084 ∧ 
    (214.59 : ℝ) = (1480.689 / 6.9) ∧ 
    (1038.33 : ℝ) = (total_oil_output_russia / population_russia) :=
by
  sorry

end oil_output_per_capita_l814_814865


namespace prove_x_is_90_l814_814430

open scoped Classical

def data_set (x : ℝ) : List ℝ :=
  [70, 110, x, 40, 50, 210, 100, 90, x]

def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

def is_mean (x : ℝ) :=
  mean (data_set x) = x

def median (xs : List ℝ) : ℝ :=
  let sorted_xs := xs.qsort (≤)
  sorted_xs[(sorted_xs.length / 2)]

def is_median (x : ℝ) :=
  median (data_set x) = x

def mode (xs : List ℝ) : ℝ :=
  let counts := List.map (λ a, (a, xs.count a)) (xs.eraseDups)
  (counts.maximumBy (λ a b => a.snd ≤ b.snd) []).fst

def is_mode (x : ℝ) :=
  mode (data_set x) = x

theorem prove_x_is_90 :
  ∀ x : ℝ, is_mean x ∧ is_median x ∧ is_mode x → x = 90 :=
by
  sorry

end prove_x_is_90_l814_814430


namespace abundant_numbers_less_than_50_number_of_abundant_numbers_less_than_50_l814_814314

def sum_of_proper_factors (n : Nat) : Nat :=
  (List.range n).filter (λ i => i > 0 ∧ n % i = 0).sum

def is_abundant (n : Nat) : Prop :=
  sum_of_proper_factors n > n

def abundant_numbers_below (n : Nat) : List Nat :=
  (List.range n).filter is_abundant

theorem abundant_numbers_less_than_50 : 
  abundant_numbers_below 50 = [12, 18, 20, 24, 30, 36, 40, 42, 48] := 
by sorry

theorem number_of_abundant_numbers_less_than_50 : 
  List.length (abundant_numbers_below 50) = 9 := 
by sorry

end abundant_numbers_less_than_50_number_of_abundant_numbers_less_than_50_l814_814314


namespace largest_number_with_digits_sum_14_l814_814833

theorem largest_number_with_digits_sum_14 : 
  ∃ n : ℕ, (∀ d ∈ (list_of_digits n), d = 2 ∨ d = 3) ∧ (list_of_digits n).sum = 14 ∧ n = 3222233 :=
by
  sorry

end largest_number_with_digits_sum_14_l814_814833


namespace quadratic_satisfies_l814_814738

noncomputable def omega : ℂ := Classical.choice (Complex.exists_root_unit 9) -- Let's assume some root of unity

def alpha (ω : ℂ) : ℂ := ω + ω^2 + ω^4 + ω^7
def beta (ω : ℂ) : ℂ := ω^3 + ω^5 + ω^6 + ω^8

theorem quadratic_satisfies (ω : ℂ) (h_ω : ω^9 = 1) (h_ω_ne : ω ≠ 1) :
  ∃ (a b : ℝ), alpha ω ^ 2 + alpha ω * beta ω + beta ω ^ 2 = a * alpha ω + b :=
sorry

end quadratic_satisfies_l814_814738


namespace school_cannot_buy_both_sets_with_500_yuan_max_revolutionary_books_with_300_yuan_after_price_changes_l814_814435

theorem school_cannot_buy_both_sets_with_500_yuan :
  let x := 270 -- price for one set of revolutionary story series
  let y := 240 -- price for one set of popular science books for teenagers
  in 500 < x + y :=
by { -- sorry to skip the proof.
  sorry
}

theorem max_revolutionary_books_with_300_yuan_after_price_changes :
  let book_price := (270 / 18 : ℝ) * 1.2
  let discount_price := 240 * 0.9
  in 300 ≥ 216 + 21.6 * 4 ∧ 4 ≤ 300 // assert the correctness
  -- find the maximum 'n' such that book_price * n + discount_price ≤ 300
  ∀ (n : ℕ), 21.6 * n + discount_price ≤ 300 → n ≤ 4 :=
by { -- sorry to skip the proof.
  sorry
}

end school_cannot_buy_both_sets_with_500_yuan_max_revolutionary_books_with_300_yuan_after_price_changes_l814_814435


namespace max_value_sin_expression_l814_814723

theorem max_value_sin_expression (A B C : ℝ) (h_sum : A + B + C = Real.pi) : 
  ∃ (m : ℝ), m = sin A + sin B + 2 * Real.sqrt 7 * sin C ∧ m ≤ 27 / 4 := 
by 
  use sin A + sin B + 2 * Real.sqrt 7 * sin C
  have h_max : sin A + sin B + 2 * Real.sqrt 7 * sin C ≤ 27 / 4 := sorry
  exact ⟨_, h_max⟩
  sorry

end max_value_sin_expression_l814_814723


namespace number_of_divisors_is_power_of_two_l814_814035

-- Define the sum of divisors function σ(n)
def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (Finset.range (n + 1)).filter (λ d, n % d = 0), d

-- Define the power of two predicate
def is_power_of_two (m : ℕ) : Prop :=
  ∃ k : ℕ, m = 2^k

-- Define the number of divisors function τ(n)
def number_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

-- The statement to be proved
theorem number_of_divisors_is_power_of_two
  (n : ℕ) (hn : 0 < n) (hσ : is_power_of_two (sum_of_divisors n)) :
  is_power_of_two (number_of_divisors n) :=
sorry

end number_of_divisors_is_power_of_two_l814_814035


namespace solve_ineq1_solve_ineq2_l814_814777

-- (1) |x-2| + |2x-3| < 4 ⇔ 1/3 < x < 3
theorem solve_ineq1 (x : ℝ) : |x-2| + |2x-3| < 4 ↔ (1 / 3 < x ∧ x < 3) :=
sorry

-- (2) (x^2 - 3x) / (x^2 - x - 2) ≤ x ⇔ x ∈ (-1, 0] ∪ {1} ∪ (2, ∞)
theorem solve_ineq2 (x : ℝ) : (x^2 - 3 * x) / (x^2 - x - 2) ≤ x ↔ x ∈ Set.Icc (-1) 0 ∨ x = 1 ∨ x ∈ Set.Ioi 2 :=
sorry

end solve_ineq1_solve_ineq2_l814_814777


namespace f_of_f_five_l814_814060

noncomputable def f : ℝ → ℝ := sorry

axiom f_periodicity (x : ℝ) : f (x + 2) = 1 / f x
axiom f_initial_value : f 1 = -5

theorem f_of_f_five : f (f 5) = -1 / 5 :=
by sorry

end f_of_f_five_l814_814060


namespace binomial_expansion_coeff_l814_814386

theorem binomial_expansion_coeff (a : ℝ) (h_a : a > 0) :
  coeff_x_5 (x + a / x)^9 = 144 → a = 2 := by
  sorry

end binomial_expansion_coeff_l814_814386


namespace sum_of_integers_satisfying_inequality_l814_814957

theorem sum_of_integers_satisfying_inequality : 
  (∑ x in finset.range 36, if 33 ≤ x then x else 0) = 102 := by
  sorry

end sum_of_integers_satisfying_inequality_l814_814957


namespace congruent_triangles_l814_814231

-- Define points and condition 
variables {A B C D : Type} [plane A B C D] (no_collinear : ¬ collinear A B C ∧ ¬ collinear A B D ∧ ¬ collinear A C D ∧ ¬ collinear B C D)

-- Define inradii equality condition
variables (rABC rABD rBCD rACD : ℝ) (h_inradii : rABC = rABD ∧ rABD = rBCD ∧ rBCD = rACD)

-- Main Theorem Statement
theorem congruent_triangles 
    (h_no_collinear : no_collinear)
    (h_inradii : rABC = rABD ∧ rABD = rBCD ∧ rBCD = rACD) : 
    congruent (triangle A B C) (triangle A B D) ∧
    congruent (triangle A B D) (triangle B C D) ∧
    congruent (triangle B C D) (triangle A C D) := 
sorry

end congruent_triangles_l814_814231


namespace general_term_a_n_sum_of_b_n_l814_814617

-- Proof Problem 1: General term of sequence {a_n}
theorem general_term_a_n (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 2) (h2 : a 2 = 4) 
    (h3 : ∀ n ≥ 2, a (n+1) - a n = 2) : 
    ∀ n, a n = 2 * n :=
by
  sorry

-- Proof Problem 2: Sum of the first n terms of sequence {b_n}
theorem sum_of_b_n (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ)
    (h : ∀ n, (1 / (a n ^ 2 - 1) : ℝ) + b n = 2^n) :
    T n = 2^(n+1) - n / (2*n + 1) :=
by
  sorry

end general_term_a_n_sum_of_b_n_l814_814617


namespace tim_will_attend_game_probability_l814_814827

theorem tim_will_attend_game_probability :
  let P_rain := 0.60
  let P_sunny := 1 - P_rain
  let P_attends_given_rain := 0.25
  let P_attends_given_sunny := 0.70
  let P_rain_and_attends := P_rain * P_attends_given_rain
  let P_sunny_and_attends := P_sunny * P_attends_given_sunny
  (P_rain_and_attends + P_sunny_and_attends) = 0.43 :=
by
  sorry

end tim_will_attend_game_probability_l814_814827


namespace other_number_when_one_is_thirtyfive_by_twentyfour_never_written_l814_814820

theorem other_number_when_one_is (a b : ℚ) (h : a * b = 2) (a_eq : a = (941664:ℚ) / 665857) : 
  b = (665857:ℚ) / 470832 := by
  sorry

theorem thirtyfive_by_twentyfour_never_written : 
  ∀ (a b : ℚ), (a = 1 ∧ b = 2) ∨ (∃ n : ℕ, (a, b) = (arithmetic_mean a_n b_n, harmonic_mean a_n b_n)) → 
               (a ≠ (35:ℚ) / 24 ∧ b ≠ (35:ℚ) / 24) := by
  sorry

end other_number_when_one_is_thirtyfive_by_twentyfour_never_written_l814_814820


namespace probability_both_info_l814_814016

open_locale classical

noncomputable def probability_information (num_students : ℕ) 
                     (students_japanese : ℕ) 
                     (students_german : ℕ) : ℚ := 
  let students_both := students_japanese + students_german - num_students in
  let students_only_jap := students_japanese - students_both in
  let students_only_ger := students_german - students_both in
  let total_ways := nat.choose num_students 2 in
  let same_language_ways := nat.choose students_only_jap 2 + nat.choose students_only_ger 2 in
  1 - (same_language_ways / total_ways)

theorem probability_both_info : 
  probability_information 30 22 24 = 56 / 62 :=
by
  sorry

end probability_both_info_l814_814016


namespace hyperbola_center_l814_814592

theorem hyperbola_center : ∃ (h k : ℝ), (h, k) = (3, 4) ∧ 
  (∀ x y : ℝ, 9*x^2 - 54*x - 16*y^2 + 128*y - 400 = 0 ↔ ((x - h)^2 / 25) - ((y - k)^2 / 14.0625) = 1) :=
begin
  use [3, 4],
  split,
  { simp, },
  { sorry }
end

end hyperbola_center_l814_814592


namespace value_of_f_ln2_l814_814749

noncomputable def f : ℝ → ℝ := sorry  -- we assume the existence of such a function for now

axiom f_monotone : ∀ x y : ℝ, x ≤ y → f x ≤ f y
axiom f_condition : ∀ x : ℝ, f (f x - real.exp x) = real.e + 1

theorem value_of_f_ln2 : f (real.log 2) = 3 :=
by
  sorry

end value_of_f_ln2_l814_814749


namespace sum_of_k_eq_one_l814_814739

theorem sum_of_k_eq_one (a b c k : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
   (h4 : a^2 / (1 - b) = k) (h5 : b^2 / (1 - c) = k) (h6 : c^2 / (1 - a) = k) 
   : k^2 - k + 1 = 0 → k^1.5 - 1 = 0 -> ∑ x in ({1 : ℝ} : finset ℝ), x = 1 :=
by
  sorry

end sum_of_k_eq_one_l814_814739


namespace part_a_part_b_l814_814480

-- Part (a)
theorem part_a (a b : set (ℝ × ℝ)) (α : set (ℝ × ℝ)) (P : ℝ × ℝ):
  (∀ x ∈ a, x ∈ α) →
  (∀ y ∈ b, y ∈ α) →
  (∃ S, S ∈ a ∧ S ∈ b ∧ S ∉ accessible) →
  P ∉ α →
  (l : set (ℝ × ℝ)) →
  (β : set (ℝ × ℝ)) →
  (∀ x ∈ a, x ∈ β) →
  P ∈ β →
  (γ : set (ℝ × ℝ)) →
  (∀ y ∈ b, y ∈ γ) →
  P ∈ γ →
  ∀ x, x ∈ l ↔ x ∈ β ∧ x ∈ γ :=
sorry

-- Part (b)
theorem part_b (a b : set (ℝ × ℝ)) (α : set (ℝ × ℝ)) (P : ℝ × ℝ):
  (∀ x ∈ a, x ∈ α) →
  (∀ y ∈ b, y ∈ α) →
  (∃ S, S ∈ a ∧ S ∈ b ∧ S ∉ accessible) →
  P ∈ α →
  (A B : set (ℝ × ℝ)) →
  (A' B' : set (ℝ × ℝ)) →
  (∀ x ∈ A, x ∈ a) →
  (∀ y ∈ B, y ∈ a) →
  (∀ x' ∈ A', x' ∈ b) →
  (∀ y' ∈ B', y' ∈ b) →
  A ≠ B →
  A' ≠ B' →
  ∃ PP', ∀ S, PP' ∈ { l : set (ℝ × ℝ) | S ∈ l ⋀ P ∈ l } ∧ S ∉ accessible :=
sorry

end part_a_part_b_l814_814480


namespace sum_coefficients_equals_1024_l814_814300

theorem sum_coefficients_equals_1024 (a : ℕ → ℝ) :
  (∀ x : ℝ, (x + 2) ^ 8 = ∑ i in Finset.range 9, a i * (x + 1) ^ i) →
  a 1 + 2 * a 2 + 3 * a 3 + 4 * a 4 + 5 * a 5 + 6 * a 6 + 7 * a 7 + 8 * a 8 = 1024 :=
by
  intro h
  sorry

end sum_coefficients_equals_1024_l814_814300


namespace apples_used_l814_814905

def initial_apples : ℕ := 43
def apples_left : ℕ := 2

theorem apples_used : initial_apples - apples_left = 41 :=
by sorry

end apples_used_l814_814905


namespace combinations_of_eight_choose_four_l814_814336

theorem combinations_of_eight_choose_four :
  (nat.choose 8 4) = 70 :=
by
  sorry

end combinations_of_eight_choose_four_l814_814336


namespace volume_change_l814_814183

theorem volume_change (L B H : ℝ):
  let V_original := L * B * H in
  let L_new := 1.5 * L in
  let B_new := 0.7 * B in
  let H_new := 1.2 * H in
  let V_new := L_new * B_new * H_new in
  V_new = 1.26 * V_original :=
by
  -- sorry will be used to indicate skipped proof
  sorry

end volume_change_l814_814183


namespace farmer_bob_water_percentage_l814_814551

/-- 
  Farmer Bob grows:
  - 3 acres of corn
  - 9 acres of cotton
  - 12 acres of beans
  - 4 acres of wheat
  
  Farmer Brenda grows:
  - 6 acres of corn
  - 7 acres of cotton
  - 14 acres of beans
  - 5 acres of wheat
  
  Farmer Bernie grows:
  - 2 acres of corn
  - 12 acres of cotton
  - 3 acres of wheat
  
  Water requirements per acre:
  - Corn: 20 gallons
  - Cotton: 65 gallons
  - Beans: 60 gallons (3 times water for corn)
  - Wheat: 45 gallons

  We need to prove that approximately 37.32% of the total water used will go to Farmer Bob's farm.
-/
theorem farmer_bob_water_percentage :
  let bob_corn := 3 * 20,
      bob_cotton := 9 * 65,
      bob_beans := 12 * 60,
      bob_wheat := 4 * 45,
      brenda_corn := 6 * 20,
      brenda_cotton := 7 * 65,
      brenda_beans := 14 * 60,
      brenda_wheat := 5 * 45,
      bernie_corn := 2 * 20,
      bernie_cotton := 12 * 65,
      bernie_wheat := 3 * 45 in
  let total_bob := bob_corn + bob_cotton + bob_beans + bob_wheat,
      total_brenda := brenda_corn + brenda_cotton + brenda_beans + brenda_wheat,
      total_bernie := bernie_corn + bernie_cotton + bernie_wheat,
      total_water := total_bob + total_brenda + total_bernie in
  let percentage_bob := (total_bob / total_water.toFloat) * 100 in
  abs (percentage_bob - 37.32) < 0.01 :=
sorry

end farmer_bob_water_percentage_l814_814551


namespace slope_of_line_m_l814_814482

-- Define the condition of the midpoint
def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

-- Define the condition of a line given a point and a y-intercept
def line_slope (midpoint y_intercept : ℝ × ℝ) : ℝ :=
  (y_intercept.2 - midpoint.2) / (y_intercept.1 - midpoint.1)

theorem slope_of_line_m :
  let A : ℝ × ℝ := (2, 8)
      B : ℝ × ℝ := (14, 4)
      M := midpoint A B
      y_intercept := (0, -2)
  in line_slope M y_intercept = 1 :=
by
  let A : ℝ × ℝ := (2, 8)
  let B : ℝ × ℝ := (14, 4)
  let M := midpoint A B
  let y_intercept := (0, -2)
  -- The proof is to be provided, here we skip it with sorry
  sorry

end slope_of_line_m_l814_814482


namespace problem_statement_l814_814560

noncomputable def G (x : ℝ) : ℝ := ((x + 1) ^ 2) / 2 - 4

theorem problem_statement : G (G (G 0)) = -3.9921875 :=
by
  sorry

end problem_statement_l814_814560


namespace perpendicular_vector_l814_814994

variables {a b : ℝ}

/-- Given that the angle between unit vectors $\overrightarrow{a}$ and $\overrightarrow{b}$ 
is $60^{\circ}$, prove that $2\overrightarrow{a} - \overrightarrow{b}$ is perpendicular to 
$\overrightarrow{b}$. -/
theorem perpendicular_vector 
  (a b : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hab : a * b = 0.5) :
  (2 * a - b) * b = 0 :=
begin
  sorry,
end

end perpendicular_vector_l814_814994


namespace determine_shift_l814_814272

noncomputable def f (x : ℝ) : ℝ := Math.sin (3 * x) - Math.cos (3 * x)
noncomputable def g (x : ℝ) : ℝ := -Math.sin (3 * x) + Math.cos (3 * x)

theorem determine_shift :
  ∃ (ϕ : ℝ), ϕ > 0 ∧ (∀ x, g x = Math.sin (3 * (x + ϕ)) - Math.cos (3 * (x + ϕ))) ↔ ϕ = Real.pi :=
begin
  sorry
end

end determine_shift_l814_814272


namespace commutative_associative_laws_multiplication_l814_814869

theorem commutative_associative_laws_multiplication (a b c : ℕ) 
  (h1 : a = 125) (h2 : b = 32) (h3 : c = 8) :
  (a * (b * c) = 32 * 1000) ∧ ((b * c) * a = b * (a * c)) :=
by
  -- First, use provided equalities to substitute a, b, and c with their values
  have h4 : a = 125 := h1,
  have h5 : b = 32 := h2,
  have h6 : c = 8 := h3,
  
  -- Now, perform the calculations
  -- 125 * (32 * 8) should be simplified to 32 * 1000 (using relevant laws)
  have h7 : a * (b * c) = 32 * 1000,
    sorry,

  -- Check that both sides of the associative law holds with the respective commutative changes
  have h8 : (b * c) * a = b * (a * c),
    sorry,

  -- Combining both results
  exact ⟨h7, h8⟩

end commutative_associative_laws_multiplication_l814_814869


namespace mike_remaining_amount_l814_814805

variables (mike johnson : ℝ)
variables (ratio_mike ratio_johnson : ℝ)
variables (johnson_share_mike_spent : ℝ)
variables (one_part mike_share mike_total_output : ℝ)

definition profit_ratio (ratio_mike : ℝ) (ratio_johnson : ℝ) := ratio_mike / ratio_johnson
definition mike_share_calculation (one_part : ℝ) (ratio_mike : ℝ) := one_part * ratio_mike
definition remaining_amount (mike_total_output mike_spent : ℝ) := mike_total_output - mike_spent

theorem mike_remaining_amount :
  ratio_mike = 2 → ratio_johnson = 5 →
  johnson_share_mike_spent = 2500 / 5 → 
  mike_total_output = johnson_share_mike_spent * 2 →
  remaining_amount mike_total_output 200 = 800 :=
by
  sorry

end mike_remaining_amount_l814_814805


namespace find_b_squared_l814_814426

noncomputable def b_squared : ℚ :=
  let a_ellipse := 25
  let a_hyperbola := 4
  let b_hyperbola := 121 / 49
  let c_hyperbola := sqrt (a_hyperbola + b_hyperbola)
  a_ellipse - c_hyperbola^2

theorem find_b_squared :
  let foci_ellipse_coincide_with_foci_hyperbola := 
    (sqrt (196/49 + 121/49)) / 7 = (sqrt (25 - (b_squared)^2)) / a_ellipse
  b_squared = 908 / 49 :=
sorry

end find_b_squared_l814_814426


namespace line_parallel_or_skew_l814_814990

open set

variables {Point : Type*} [affine_space Point]

-- Definitions
def line (l : set Point) := aff_indep l ∧ ∃ p1 p2, p1 ≠ p2 ∧ l = affine_span {p1, p2}
def plane (π : set Point) := ∃ p1 p2 p3, aff_indep {p1, p2, p3} ∧ π = affine_span {p1, p2, p3}

-- Conditions
variables (a b : set Point) (α : set Point)
variable [λ a α, line a ∧ plane α ∧ ¬∃ p, p ∈ a ∧ p ∈ α]
variable [λ b α, line b ∧ b ⊆ α]

-- Theorem Statement
theorem line_parallel_or_skew (a b : set Point) (α : set Point) :
    (line a ∧ plane α ∧ ¬∃ p, p ∈ a ∧ p ∈ α) ∧ (line b ∧ b ⊆ α) →
    (parallel a b ∨ skew a b) :=
    sorry

end line_parallel_or_skew_l814_814990


namespace ellipse_equation_product_of_slopes_l814_814982

-- Given conditions and requirements for Part (I)
theorem ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) 
    (focal_length : ℝ) (focal_length_pos : focal_length = 2)
    (point_on_ellipse : (ℝ × ℝ)) (h_point : point_on_ellipse = (1, 3 / 2)) 
: (a = 2 ∧ b = sqrt 3) → (∀ x y, (x = 1 ∧ y = 3 / 2) → (x ^ 2 / 4 + y ^ 2 / 3 = 1)) := 
sorry

-- Given conditions and requirements for Part (II)
theorem product_of_slopes (A B P : ℝ × ℝ) (x1 y1 x2 y2 : ℝ)  
    (hA : A = (x1, y1)) (hB : B = (-x1, -y1)) (hP : P = (x2, y2))  
    (ellipse_cond : (x1 ^ 2 / 4 + y1 ^ 2 / 3 = 1) ∧ (x2 ^ 2 / 4 + y2 ^ 2 / 3 = 1))
    (not_equal_cond : x1 ≠ x2)
: ((y1 * (x1 - x2) / (x1 * (y1 - y2))) * (y1 * (x1 - x2) / (x2 * (y1 - y2)))) = 1 := 
sorry

end ellipse_equation_product_of_slopes_l814_814982


namespace fried_hop_edges_in_three_hops_l814_814014

noncomputable def fried_hop_probability : ℚ :=
  let moves : List (Int × Int) := [(-1, 0), (1, 0), (0, -1), (0, 1)]
  let center := (2, 2)
  let edges := [(1, 2), (1, 3), (2, 1), (2, 4), (3, 1), (3, 4), (4, 2), (4, 3)]
  -- Since the exact steps of solution calculation are complex,
  -- we assume the correct probability as per our given solution.
  5 / 8

theorem fried_hop_edges_in_three_hops :
  let p := fried_hop_probability
  p = 5 / 8 := by
  sorry

end fried_hop_edges_in_three_hops_l814_814014


namespace minimum_value_attained_at_n_10_l814_814440

theorem minimum_value_attained_at_n_10 : ∀ (n : ℕ) (h : 0 < n), 
  (n = 10) → (n / 2 + 50 / n = 10) :=
begin
  intros n h hn,
  have h1 : n / 2 = 50 / n, by sorry,
  have h2 : 2 * sqrt (n / 2 * 50 / n) = 10, by sorry,
  exact eq.trans (add_eq_of_eq_of_eq h1 h2) hn,
end

end minimum_value_attained_at_n_10_l814_814440


namespace line_equation_of_intersecting_ellipse_and_centroid_l814_814626

theorem line_equation_of_intersecting_ellipse_and_centroid:
  (∀ (A B M : ℝ × ℝ) (F : ℝ × ℝ),
    A ∈ ellipse 20 16 →
    B ∈ ellipse 20 16 →
    M = (0, 4) →
    F = (2, 0) →
    centroid (M, A, B) = F →
    line_through A B = { p : ℝ × ℝ | 6 * p.1 - 5 * p.2 - 28 = 0 }) :=
by
  sorry

end line_equation_of_intersecting_ellipse_and_centroid_l814_814626


namespace smallest_sum_l814_814259

theorem smallest_sum (x y : ℕ) (hx : x ≠ y) (hxy : (1/x:ℚ) + (1/y:ℚ) = 1/15) : x + y = 64 :=
sorry

end smallest_sum_l814_814259


namespace sin_600_eq_neg_sqrt_3_div_2_l814_814043

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- proof to be provided here
  sorry

end sin_600_eq_neg_sqrt_3_div_2_l814_814043


namespace measure_of_angle_A_l814_814008

-- Define the angles in the triangle
variables (A B C : ℝ)

-- Define the conditions
def cond1 : Prop := B = 30
def cond2 : Prop := C = 3 * B
def angle_sum : Prop := A + B + C = 180

-- State the main theorem
theorem measure_of_angle_A (h1: cond1) (h2: cond2) (h3: angle_sum) : A = 60 :=
sorry

end measure_of_angle_A_l814_814008


namespace birch_count_l814_814128

theorem birch_count (T : ℕ) (B L : ℕ) (h1 : T = 130) 
  (h2 : B + L = T) (h3 : L + 1 = B) (h5 : 2 * B - 1 = T) : 
  B = 87 :=
by 
  rw [h1] at h5
  rw [h1] at h2
  simp at h5
  simp at h2
  sorry

end birch_count_l814_814128


namespace sum_floor_eq_half_prod_l814_814381

theorem sum_floor_eq_half_prod (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  ∑ r in Finset.range (b - 1), (r * a / b) = (a - 1) * (b - 1) / 2 := sorry

end sum_floor_eq_half_prod_l814_814381


namespace circumcircle_radius_half_hypotenuse_l814_814050

theorem circumcircle_radius_half_hypotenuse 
  (A B C M N K : Point) 
  (h_right_angle : is_right_angle A B C) 
  (h_midpoints_MN : midpoint A B M ∧ midpoint B C N) 
  (h_midpoint_K : midpoint A C K) :
  radius_of_circumcircle_passing_through B M K = dist A C / 2 :=
by
  sorry

end circumcircle_radius_half_hypotenuse_l814_814050


namespace sum_of_A_H_l814_814722

theorem sum_of_A_H (A B C D E F G H : ℝ) (h1 : C = 10) 
  (h2 : A + B + C = 40) (h3 : B + C + D = 40) (h4 : C + D + E = 40) 
  (h5 : D + E + F = 40) (h6 : E + F + G = 40) (h7 : F + G + H = 40) :
  A + H = 30 := 
sorry

end sum_of_A_H_l814_814722


namespace trigonometric_identity_l814_814188

theorem trigonometric_identity :
  (1 / Real.cos (40 * Real.pi / 180) - 2 * Real.sqrt 3 / Real.sin (40 * Real.pi / 180)) = -4 * Real.tan (20 * Real.pi / 180) := 
sorry

end trigonometric_identity_l814_814188


namespace smallest_n_subset_contains_five_pairwise_rel_prime_numbers_l814_814735

theorem smallest_n_subset_contains_five_pairwise_rel_prime_numbers :
  let S := {1, 2, 3, ..., 280}
  in ∃ n, (∀ T, T ⊆ S ∧ card T = n → ∃ U, U ⊆ T ∧ card U = 5 ∧ (∀ a b ∈ U, gcd a b = 1)) 
  → n = 217 :=
by
  sorry

end smallest_n_subset_contains_five_pairwise_rel_prime_numbers_l814_814735


namespace tangent_line_eq_range_of_a_f_prime_sum_l814_814388

-- Problem definition:
def f (a : ℝ) (x : ℝ) : ℝ := 0.5 * a * x^2 - 1 - log x

-- Given part 1:
theorem tangent_line_eq (a : ℝ) (h : a = 0) :
  let f0 := f 0 in
  let tangent_line := λ x, -1/e * x - 1 in
  let T := (e, -1 - log e) in
  T.2 = f0 e ∧ 
  ((λ y : ℝ, y + 1 + log e = -1/e * (y - e)).1 0 = -1) ∧ 
  (tangent_line 0  = -1) ∧
  tangent_line = λ x, -1/e * x - 1
:= 
  sorry

-- Given part 2: The function f(x) has two zeros x1 and x2
-- 2(i) Range of values for a
theorem range_of_a (a : ℝ) (x1 x2 : ℝ) (hx1 : f a x1 = 0) (hx2 : f a x2 = 0) : 
  0 < a ∧ a < exp 1
:= 
  sorry

-- 2(ii) Proof that f'(x1) + f'(x2) < 0
theorem f_prime_sum (a : ℝ) (x1 x2 : ℝ) (hx1 : f a x1 = 0) (hx2 : f a x2 = 0) : 
  ((a * x1 - 1/x1) + (a * x2 - 1/x2)) < 0
:=
  sorry

end tangent_line_eq_range_of_a_f_prime_sum_l814_814388


namespace smallest_x_2_abs_eq_24_l814_814604

theorem smallest_x_2_abs_eq_24 : ∃ x : ℝ, (2 * |x - 10| = 24) ∧ (∀ y : ℝ, (2 * |y - 10| = 24) -> x ≤ y) := 
sorry

end smallest_x_2_abs_eq_24_l814_814604


namespace probability_of_closer_to_center_l814_814112

theorem probability_of_closer_to_center {R : ℝ} (hR : R = 1) : 
  let larger_area := Real.pi * R ^ 2 in
  let smaller_radius := R / 2 in
  let smaller_area := Real.pi * (smaller_radius) ^ 2 in
  (smaller_area / larger_area) = 1 / 4 :=
by
  sorry

end probability_of_closer_to_center_l814_814112


namespace num_physical_signals_given_4_zeros_4_ones_l814_814559

theorem num_physical_signals_given_4_zeros_4_ones : 
  let zeros := 4
  let ones := 4
  let total_digits := zeros + ones
  (Nat.fact total_digits) / ((Nat.fact zeros) * (Nat.fact ones)) = 70 :=
by
  let zeros := 4
  let ones := 4
  let total_digits := zeros + ones
  sorry

end num_physical_signals_given_4_zeros_4_ones_l814_814559


namespace grasshopper_min_jumps_l814_814137

/-- A grasshopper can jump exactly 50 cm at a time. 
    Each side of the grid cell is 10 cm. 
    There are 8 points on the grid that need to be visited by the grasshopper.
    The grasshopper can start and finish at any points.
    It is allowed to visit other points on the plane, including points that are not grid nodes.
    Prove that the minimum number of jumps required for the grasshopper to visit all 8 points is 8. -/
theorem grasshopper_min_jumps
  (points : Fin 8 → Prop) 
  (jump_distance : ℝ := 50) 
  (grid_size : ℝ := 10) 
  (start_finish_freedom : ∀ {p1 p2 : Prop}, p1 ∈ points → p2 ∈ points → True)
  (allowed_other_points : ∀ (p : Prop), p ∉ points → True) :
  ∃ (n : ℕ), n = 8 := 
by
  sorry

end grasshopper_min_jumps_l814_814137


namespace cone_height_relation_l814_814090

variables {r1 r2 h1 h2 : ℝ}

theorem cone_height_relation (h : r2 = 1.1 * r1) (hv : (1 / 3) * π * r1^2 * h1 = (1 / 3) * π * r2^2 * h2) :
  h1 = 1.21 * h2 :=
begin
  -- Since we are only required to state the theorem, we omit the proof.
  sorry,
end

end cone_height_relation_l814_814090


namespace valid_necklace_length_l814_814165

-- Define the shapes and colors as types
inductive Shape | cube | pyramid
inductive Color | green | blue | red

-- Each bead is characterized by its shape and color
structure Bead :=
  (shape : Shape)
  (color : Color)

-- Definition to check adjacent beads for different shape and color
def valid_pair (b1 b2 : Bead) :=
  (b1.shape ≠ b2.shape) ∧ (b1.color ≠ b2.color)

-- Definition of a valid necklace
def valid_necklace (beads : List Bead) :=
  (List.length beads) % 2 = 0 ∧
  (∀ i : Fin (List.length beads), valid_pair (beads.get i) (beads.get ((i + 1) % (List.length beads))))

-- The main proposition
theorem valid_necklace_length (n : Nat) :
  n = 8 ∨ n = 10 ↔
  ∃ beads : List Bead,
    valid_necklace beads ∧
    List.length beads = n :=
by
  sorry

end valid_necklace_length_l814_814165


namespace proofSmallestM_l814_814376

def LeanProb (a b c d e f : ℕ) : Prop :=
  a + b + c + d + e + f = 2512 →
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (0 < d) ∧ (0 < e) ∧ (0 < f) →
  ∃ M, (M = 1005) ∧ (M = max (a+b) (max (b+c) (max (c+d) (max (d+e) (e+f)))))

theorem proofSmallestM (a b c d e f : ℕ) (h1 : a + b + c + d + e + f = 2512) 
(h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) (h5 : 0 < d) (h6 : 0 < e) (h7 : 0 < f) : 
  ∃ M, (M = 1005) ∧ (M = max (a+b) (max (b+c) (max (c+d) (max (d+e) (e+f))))):=
by
  sorry

end proofSmallestM_l814_814376


namespace value_range_of_f_l814_814812

open Set

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_range_of_f : {y : ℝ | ∃ x ∈ Icc (-2 : ℝ) (2 : ℝ), f x = y} = Icc (-1 : ℝ) 8 := 
by
  sorry

end value_range_of_f_l814_814812


namespace isosceles_triangle_angle_l814_814400

theorem isosceles_triangle_angle 
  (ABCD : Type) [rectangle ABCD]
  (A B C D N : Point) 
  (h1 : dist A B = 8) 
  (h2 : dist B C = 4) 
  (h3 : N ∈ segment B C) 
  (h4 : dist A N = dist N D) :
  angle A N D = 56 := 
by
  sorry

end isosceles_triangle_angle_l814_814400


namespace university_major_preference_l814_814532

noncomputable def number_of_ways (total_majors : ℕ) (choose_majors : ℕ) (a : ℕ) (b : ℕ) : ℕ :=
  let total_combinations := Nat.choose total_majors choose_majors * Nat.factorial choose_majors
  let invalid_combinations := Nat.choose (total_majors - 2) (choose_majors - 2) * Nat.factorial choose_majors
  total_combinations - invalid_combinations

theorem university_major_preference :
  number_of_ways 10 3 1 2 = 672 :=
by
  have h1 : number_of_ways 10 3 1 2 = 
              ((Nat.choose 10 3) * Nat.factorial 3) - 
              ((Nat.choose 8 1) * (Nat.choose 2 2) * Nat.factorial 3),
  { 
    rw [number_of_ways],
    simp,
  }
  rw [h1],
  simp,
  exact rfl

end university_major_preference_l814_814532


namespace perpendicular_vector_l814_814995

variables {a b : ℝ}

/-- Given that the angle between unit vectors $\overrightarrow{a}$ and $\overrightarrow{b}$ 
is $60^{\circ}$, prove that $2\overrightarrow{a} - \overrightarrow{b}$ is perpendicular to 
$\overrightarrow{b}$. -/
theorem perpendicular_vector 
  (a b : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hab : a * b = 0.5) :
  (2 * a - b) * b = 0 :=
begin
  sorry,
end

end perpendicular_vector_l814_814995


namespace angle_between_vectors_l814_814657

variables {V : Type*} [inner_product_space ℝ V]

theorem angle_between_vectors 
  {a b : V} 
  (h1 : (a + 2 • b) ⬝ (5 • a - 4 • b) = 0) 
  (h2 : ∥a∥ = 1) 
  (h3 : ∥b∥ = 1) : 
  real.angle (a ⬝ b) = real.angle (cos (π / 3)) :=
sorry

end angle_between_vectors_l814_814657


namespace min_sum_xy_l814_814263

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (hcond : ↑(1 / x) + ↑(1 / y) = 1 / 15) : x + y = 64 :=
sorry

end min_sum_xy_l814_814263


namespace proof_problem_l814_814987

variable (a b : ℝ)

def p : Prop := (a > b) ↔ (2^a > 2^b)
def q : Prop := ¬ ∃ x : ℝ, exp x < log x

theorem proof_problem : p ∨ q := by
  sorry

end proof_problem_l814_814987


namespace polynomial_pattern_1_polynomial_pattern_2_polynomial_calculation_polynomial_factorization_l814_814849

theorem polynomial_pattern_1 (a b : ℝ) : (a + b) * (a ^ 2 - a * b + b ^ 2) = a ^ 3 + b ^ 3 :=
sorry

theorem polynomial_pattern_2 (a b : ℝ) : (a - b) * (a ^ 2 + a * b + b ^ 2) = a ^ 3 - b ^ 3 :=
sorry

theorem polynomial_calculation (a b : ℝ) : (a + 2 * b) * (a ^ 2 - 2 * a * b + 4 * b ^ 2) = a ^ 3 + 8 * b ^ 3 :=
sorry

theorem polynomial_factorization (a : ℝ) : a ^ 3 - 8 = (a - 2) * (a ^ 2 + 2 * a + 4) :=
sorry

end polynomial_pattern_1_polynomial_pattern_2_polynomial_calculation_polynomial_factorization_l814_814849


namespace max_distance_S_to_origin_l814_814369

open Complex

-- Definitions based on the conditions
def P (z : ℂ) := z
def Q (z : ℂ) := (2 + I) * z
def R (z : ℂ) := (1 - I) * conj z
def S (z : ℂ) := (1 + I) * z + (1 - I) * conj z

-- The condition that |z| = 1
def unit_circle (z : ℂ) := abs z = 1

-- The statement to prove
theorem max_distance_S_to_origin (z : ℂ) (hz : unit_circle z) : abs (S z) ≤ 2 := 
sorry

end max_distance_S_to_origin_l814_814369


namespace graph_inverse_point_sum_l814_814412

theorem graph_inverse_point_sum 
  (f : ℝ → ℝ) (f_inv : ℝ → ℝ) 
  (h1 : ∀ x, f_inv (f x) = x) 
  (h2 : ∀ x, f (f_inv x) = x) 
  (h3 : f 2 = 6) 
  (h4 : (2, 3) ∈ {p : ℝ × ℝ | p.snd = f p.fst / 2}) :
  (6, 1) ∈ {p : ℝ × ℝ | p.snd = f_inv p.fst / 2} ∧ (6 + 1 = 7) :=
by
  sorry

end graph_inverse_point_sum_l814_814412


namespace second_number_is_90_l814_814759

theorem second_number_is_90 (x y z : ℕ) 
  (h1 : z = 4 * y)
  (h2 : y = 2 * x)
  (h3 : (x + y + z) / 3 = 165) : 
  y = 90 := 
begin
  sorry
end

end second_number_is_90_l814_814759


namespace necessary_and_sufficient_condition_l814_814051

variable {R : Type*} [LinearOrderedField R]
variable (f : R × R → R)
variable (x₀ y₀ : R)

theorem necessary_and_sufficient_condition :
  (f (x₀, y₀) = 0) ↔ ((x₀, y₀) ∈ {p : R × R | f p = 0}) :=
by
  sorry

end necessary_and_sufficient_condition_l814_814051


namespace coefficient_of_q_is_correct_l814_814285

theorem coefficient_of_q_is_correct (q' : ℕ → ℕ) : 
  (∀ q : ℕ, q' q = 3 * q - 3) ∧  q' (q' 7) = 306 → ∃ a : ℕ, (∀ q : ℕ, q' q = a * q - 3) ∧ a = 17 :=
by
  sorry

end coefficient_of_q_is_correct_l814_814285


namespace selling_prices_max_units_model_A_profit_goal_l814_814899

theorem selling_prices (x y : ℝ) : 
  3 * x + 2 * y = 1120 ∧ 4 * x + 3 * y = 1560 → 
  x = 240 ∧ y = 200 :=
begin
  sorry
end

theorem max_units_model_A (a : ℝ) : 
  (a : ℝ) ≥ 0 ∧ a ≤ 35 ∧ 180 * a + 160 * (35 - a) ≤ 6000 → 
  a ≤ 20 :=
begin
  sorry
end

theorem profit_goal (a b : ℕ) :
  a + b = 35 ∧ 180 * a + 160 * b ≤ 6000 ∧ 
  60 * a + 40 * b > 1750 → 
  60 * 20 + 40 * 15 = 1800 :=
begin
  sorry
end

end selling_prices_max_units_model_A_profit_goal_l814_814899


namespace number_of_students_selected_from_grade_12_correct_l814_814083

variable (total_students : ℕ) (students10 : ℕ) (r1 r2 r3 : ℕ)

def number_of_students_selected_from_grade_12 : ℕ :=
  total_students * r3 / (r1 + r2 + r3)

theorem number_of_students_selected_from_grade_12_correct
  (h_total : total_students = 1200)
  (h_students10 : students10 = 240)
  (h_ratio10 : ∀ k : ℕ, r1 = k ∧ r2 = 5 ∧ r3 = 3 →
    students10 * (r1 + r2 + r3) / total_students = k) :
  number_of_students_selected_from_grade_12 total_students 240 2 5 3 = 360 :=
by
  sorry

end number_of_students_selected_from_grade_12_correct_l814_814083


namespace principle_calculation_l814_814117

noncomputable def calculate_principal (A : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  A / (1 + (R * T))

theorem principle_calculation :
  calculate_principal 1456 0.05 2.4 = 1300 :=
by
  sorry

end principle_calculation_l814_814117


namespace total_amount_returned_l814_814408

noncomputable def continuous_compounding_interest : ℝ :=
  let P : ℝ := 325 / (Real.exp 0.12 - 1)
  let A1 : ℝ := P * Real.exp 0.04
  let A2 : ℝ := A1 * Real.exp 0.05
  let A3 : ℝ := A2 * Real.exp 0.03
  let total_interest : ℝ := 325
  let total_amount : ℝ := P + total_interest
  total_amount

theorem total_amount_returned :
  continuous_compounding_interest = 2874.02 :=
by
  sorry

end total_amount_returned_l814_814408


namespace third_discount_is_five_percent_l814_814881

theorem third_discount_is_five_percent (P F : ℝ) (D : ℝ)
  (h1: P = 9356.725146198829)
  (h2: F = 6400)
  (h3: F = (1 - D / 100) * (0.9 * (0.8 * P))) : 
  D = 5 := by
  sorry

end third_discount_is_five_percent_l814_814881


namespace smallest_sum_l814_814258

theorem smallest_sum (x y : ℕ) (hx : x ≠ y) (hxy : (1/x:ℚ) + (1/y:ℚ) = 1/15) : x + y = 64 :=
sorry

end smallest_sum_l814_814258


namespace count_n_less_than_150_satisfying_conditions_l814_814665

def satisfies_conditions (n m : ℕ) : Prop :=
  ∃ r : ℕ, 
    n = 2 * r + 2 ∧ 
    m = r * (r + 2) ∧ 
    m % 4 = 0

theorem count_n_less_than_150_satisfying_conditions : 
  set.count {n | ∃ m, satisfies_conditions n m ∧ n < 150 ∧ n > 0} = 37 :=
by sorry

end count_n_less_than_150_satisfying_conditions_l814_814665


namespace sum_of_binom_coeffs_l814_814972

theorem sum_of_binom_coeffs (a : Fin 11 → ℤ) :
  (∀ x : ℤ, (1 - 2*x)^10 = ∑ i in Finset.range 11, a i * x^i) →
  ∑ i in Finset.range 10, a (i + 1) = 0 :=
by
  intro h
  -- Proof steps would go here
  sorry

end sum_of_binom_coeffs_l814_814972


namespace exists_nonidentical_subset_with_common_elements_l814_814378

open Finset

theorem exists_nonidentical_subset_with_common_elements {k n : ℕ} (X : Finset (Fin k)) (A : Finset (Finset (Fin k)))
  (hA : A.card = n)
  (hX : ∀ (A1 A2 : Finset (Fin k)), A1 ∈ A → A2 ∈ A → A1 ≠ A2 → (A1 ∩ A2).nonempty)
  (hn : n < 2 ^ (k - 1)) :
  ∃ (C : Finset (Fin k)), C ∉ A ∧ ∀ (A_i : Finset (Fin k)), A_i ∈ A → (C ∩ A_i).nonempty := sorry

end exists_nonidentical_subset_with_common_elements_l814_814378


namespace tan_half_angle_computation_l814_814067

theorem tan_half_angle_computation :
  ∃ (e f g h : ℕ), (e ≥ f ∧ f ≥ g ∧ g ≥ h ∧
  tan (22.5 * (real.pi / 180)) = real.sqrt e + f - real.sqrt g - h ∧
  e + f + g + h = 4) :=
begin
  sorry
end

end tan_half_angle_computation_l814_814067


namespace arithmetic_sequence_smallest_value_l814_814232

theorem arithmetic_sequence_smallest_value:
  ∃ a : ℕ, (7 * a + 63) % 11 = 0 ∧ (a - 9) % 11 = 4 := sorry

end arithmetic_sequence_smallest_value_l814_814232


namespace largest_c_for_range_of_f_l814_814210

theorem largest_c_for_range_of_f :
  ∃ (c : ℝ), (∀ (x : ℝ), (f x = x^2 - 6 * x + c ∧ 3 ∈ set.range f) → c ≤ 12) :=
sorry

end largest_c_for_range_of_f_l814_814210


namespace find_d_l814_814377

noncomputable def f (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem find_d (a b c d : ℝ) (roots_negative_integers : ∀ x, f x a b c d = 0 → x < 0) (sum_is_2023 : a + b + c + d = 2023) :
  d = 17020 :=
sorry

end find_d_l814_814377


namespace abs_neg_three_l814_814873

theorem abs_neg_three : |(-3 : ℝ)| = 3 := 
by
  -- The proof would go here, but we skip it for this exercise.
  sorry

end abs_neg_three_l814_814873


namespace find_constant_term_l814_814219

theorem find_constant_term (c : ℤ) (y : ℤ) (h1 : y = 2) (h2 : 5 * y^2 - 8 * y + c = 59) : c = 55 :=
by
  sorry

end find_constant_term_l814_814219


namespace relationship_among_abc_l814_814612

noncomputable def a := Real.log 2 / Real.log (1/5)
noncomputable def b := 3 ^ (3/5)
noncomputable def c := 4 ^ (1/5)

theorem relationship_among_abc : a < c ∧ c < b := 
by
  sorry

end relationship_among_abc_l814_814612


namespace average_runs_in_32_matches_l814_814790

theorem average_runs_in_32_matches (avg_25: ℝ) (matches_25: ℝ) (avg_7: ℝ) (matches_7: ℝ)
  (h_avg_25: avg_25 = 45) (h_matches_25: matches_25 = 25) (h_avg_7: avg_7 = 15) (h_matches_7: matches_7 = 7) :
  (avg_25 * matches_25 + avg_7 * matches_7) / (matches_25 + matches_7) = 38.44 :=
by
  have total_runs_25 := avg_25 * matches_25
  have total_runs_7 := avg_7 * matches_7
  have total_runs_32 := total_runs_25 + total_runs_7
  have matches_32 := matches_25 + matches_7
  have avg_32 := total_runs_32 / matches_32
  have : avg_32 = 38.4375 := by sorry
  show avg_32 = 38.44 from by sorry

end average_runs_in_32_matches_l814_814790


namespace cotton_equals_iron_l814_814489

theorem cotton_equals_iron (cotton_weight : ℝ) (iron_weight : ℝ)
  (h_cotton : cotton_weight = 1)
  (h_iron : iron_weight = 4) :
  (4 / 5) * cotton_weight = (1 / 5) * iron_weight :=
by
  rw [h_cotton, h_iron]
  simp
  sorry

end cotton_equals_iron_l814_814489


namespace tangent_line_at_x1_l814_814057

def f (x : ℝ) : ℝ := x^3 + x

def tangent_line_equation := 4 * (1 : ℝ) - (f 1) - 2 = 0

theorem tangent_line_at_x1 : tangent_line_equation :=
begin
  sorry
end

end tangent_line_at_x1_l814_814057


namespace janet_daily_search_time_l814_814948

-- Define the conditions
def minutes_looking_for_keys_per_day (x : ℕ) := 
  let total_time_per_day := x + 3
  let total_time_per_week := 7 * total_time_per_day
  total_time_per_week = 77

-- State the theorem
theorem janet_daily_search_time : 
  ∃ x : ℕ, minutes_looking_for_keys_per_day x ∧ x = 8 := by
  sorry

end janet_daily_search_time_l814_814948


namespace discriminant_of_cubic_eq_1_l814_814594

-- Define the discriminant for a cubic equation
def discriminant_cubic (a b c d : ℝ) : ℝ :=
  -27 * a ^ 2 * d ^ 2 + 18 * a * b * c * d - 4 * b ^ 3 * d + b ^ 2 * c ^ 2 - 4 * a * c ^ 3

-- Define the specific cubic equation with its coefficients
def cubic_eq_1_coefs : ℝ × ℝ × ℝ × ℝ := (1, -2, 5, 2)

-- Extracting the coefficients
def a : ℝ := cubic_eq_1_coefs.1
def b : ℝ := cubic_eq_1_coefs.2
def c : ℝ := cubic_eq_1_coefs.3
def d : ℝ := cubic_eq_1_coefs.4

-- Statement to prove the discriminant of the cubic equation
theorem discriminant_of_cubic_eq_1 : discriminant_cubic a b c d = -640 := by
  -- Proof would go here
  sorry

end discriminant_of_cubic_eq_1_l814_814594


namespace eiffel_tower_model_height_l814_814537

theorem eiffel_tower_model_height :
  let scale_ratio := 25
  let actual_height := 984
  let model_height := (actual_height : ℚ) / scale_ratio
  let rounded_model_height := Int.round model_height
  rounded_model_height = 39 :=
by
  sorry

end eiffel_tower_model_height_l814_814537


namespace malfunctioning_clock_fraction_correct_l814_814516

noncomputable def malfunctioning_clock_correct_time_fraction : ℚ := 5 / 8

theorem malfunctioning_clock_fraction_correct :
  malfunctioning_clock_correct_time_fraction = 5 / 8 := 
by
  sorry

end malfunctioning_clock_fraction_correct_l814_814516


namespace find_David_marks_in_Physics_l814_814564

theorem find_David_marks_in_Physics
  (english_marks : ℕ) (math_marks : ℕ) (chem_marks : ℕ) (biology_marks : ℕ)
  (avg_marks : ℕ) (num_subjects : ℕ)
  (h_english : english_marks = 76)
  (h_math : math_marks = 65)
  (h_chem : chem_marks = 67)
  (h_bio : biology_marks = 85)
  (h_avg : avg_marks = 75) 
  (h_num_subjects : num_subjects = 5) :
  english_marks + math_marks + chem_marks + biology_marks + physics_marks = avg_marks * num_subjects → physics_marks = 82 := 
  sorry

end find_David_marks_in_Physics_l814_814564


namespace log_base_increasing_incorrect_l814_814632

theorem log_base_increasing_incorrect (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  ¬ (∀ a, log a x (a : ℝ) x > 0 → 1 < a → strict_mono (log a x)) :=
by {
  intro h,
  have ha : 0 < (1 / 3) ∧ (1 / 3) ≠ 1 := by { split; norm_num, },
  specialize h ((1 / 3)),
  contradiction,
}

end log_base_increasing_incorrect_l814_814632


namespace terms_are_coprime_l814_814808

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0     => 2
  | (n+1) => (sequence n) ^ 2 - (sequence n) + 1

theorem terms_are_coprime (n m : ℕ) (h : n ≠ m) : Nat.gcd (sequence n) (sequence m) = 1 := 
sorry

end terms_are_coprime_l814_814808


namespace original_list_size_l814_814103

-- Define the problem conditions in Lean

variables {n m : ℕ} -- n: number of integers in the original list, m: mean of these integers

theorem original_list_size 
  (h1 : (m + 3) * (n + 1) = m * n + 20) 
  (h2 : (m + 1) * (n + 2) = m * n + 21) :
  n = 3 :=
begin
  -- We start by translating the conditions into a system of equations
  -- The actual proof would follow
  sorry
end

end original_list_size_l814_814103


namespace shaded_rectangle_ratio_l814_814930

/-- Define conditions involved in the problem -/
def side_length_large_square : ℕ := 50
def num_rows_cols_grid : ℕ := 5
def rows_spanned_rect : ℕ := 2
def cols_spanned_rect : ℕ := 3

/-- Calculate the side length of a small square in the grid -/
def side_length_small_square := side_length_large_square / num_rows_cols_grid

/-- Calculate the area of the large square -/
def area_large_square := side_length_large_square * side_length_large_square

/-- Calculate the area of the shaded rectangle -/
def area_shaded_rectangle :=
  (rows_spanned_rect * side_length_small_square) *
  (cols_spanned_rect * side_length_small_square)

/-- Prove the ratio of the shaded rectangle's area to the large square's area -/
theorem shaded_rectangle_ratio : 
  (area_shaded_rectangle : ℚ) / area_large_square = 6/25 := by
  sorry

end shaded_rectangle_ratio_l814_814930


namespace min_positive_period_cos_l814_814798

theorem min_positive_period_cos : 
  (∃ T > 0, ∀ x : ℝ, 
    cos (π / 3 - 2 / 5 * (x + T)) = cos (π / 3 - 2 / 5 * x)) →
  5 * π :=
by
  intro h
  sorry

end min_positive_period_cos_l814_814798


namespace logarithmic_condition_solution_l814_814581

noncomputable def logarithmic_points_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | log (p.1^3 + (1 / 3) * p.2^3 + (1 / 9)) = log p.1 + log p.2}

theorem logarithmic_condition_solution :
  logarithmic_points_set = { (⟨ real.cbrt 3 / 3, real.cbrt 9 / 3 ⟩) } := 
by 
  sorry

end logarithmic_condition_solution_l814_814581


namespace number_of_n_is_5_l814_814964

noncomputable def arithmetic_sequences (S T : ℕ → ℕ) (P : ℕ → Prop) :=
  (∀ n, S (n + 1) = S n + a_n) ∧ 
  (∀ n, T (n + 1) = T n + b_n) ∧ 
  (∀ n, S n ≠ 0 ∧ T n ≠ 0) ∧
  (∀ n, (S n).toRat / (T n).toRat = (7 * n + 45 : ℕ) / (n - 3 : ℕ))

theorem number_of_n_is_5 (S T : ℕ → ℕ) (P : ℕ → Prop)
  (h : arithmetic_sequences S T P) : 
  {n : ℕ | P n}.card = 5 :=
sorry

end number_of_n_is_5_l814_814964


namespace difference_x_l814_814693

def range (s : Set ℕ) : ℕ := s.sup' - s.inf'

theorem difference_x (x : ℕ) (h : range {4, 314, 710, x} = 12) :
  abs ((max x (714 - 12)) - (min x (4 + 12))) = 682 :=
by
  sorry

end difference_x_l814_814693


namespace Запад_oil_output_per_capita_Не_Запад_oil_output_per_capita_Россия_oil_output_per_capita_l814_814863

noncomputable def oil_output_per_capita (total_output : ℝ) (population : ℝ) : ℝ := total_output / population

theorem Запад_oil_output_per_capita :
  oil_output_per_capita 55.084 1 = 55.084 :=
by
  sorry

theorem Не_Запад_oil_output_per_capita :
  oil_output_per_capita 1480.689 6.9 = 214.59 :=
by
  sorry

theorem Россия_oil_output_per_capita :
  oil_output_per_capita (13737.1 * 100 / 9) 147 = 1038.33 :=
by
  sorry

end Запад_oil_output_per_capita_Не_Запад_oil_output_per_capita_Россия_oil_output_per_capita_l814_814863


namespace cube_root_floor_div_l814_814558

theorem cube_root_floor_div (L R : ℕ → ℕ) (hL : ∀ n, L n = Int.floor (Real.cbrt (↑n : ℝ)))
  (hR : ∀ n, R n = Int.floor (Real.cbrt (↑(n + 1) : ℝ))) :
  (∏ n in Ico 1 220, L (2 * n - 1)) / (∏ n in Ico 1 220, R (2 * n)) = 1 / 6 :=
by {
  sorry
}

end cube_root_floor_div_l814_814558


namespace oil_output_per_capita_correctness_l814_814860

variable (population_west : ℝ := 1)
variable (output_west : ℝ := 55.084)
variable (population_non_west : ℝ := 6.9)
variable (output_non_west : ℝ := 1480.689)
variable (output_russia_9_percent : ℝ := 13737.1)
variable (percentage : ℝ := 9)
variable (total_population_russia : ℝ := 147)

def west_output_per_capita : ℝ :=
  output_west / population_west

def non_west_output_per_capita : ℝ :=
  output_non_west / population_non_west

def total_output_russia : ℝ :=
  (output_russia_9_percent * 100) / percentage

def russia_output_per_capita : ℝ :=
  total_output_russia / total_population_russia

theorem oil_output_per_capita_correctness :
  west_output_per_capita = 55.084 ∧
  non_west_output_per_capita = 214.59 ∧
  total_output_russia = 152634.44 ∧
  russia_output_per_capita = 1038.33 :=
by
  sorry

end oil_output_per_capita_correctness_l814_814860


namespace volume_pyramid_eq_l814_814422

variable (d1 d2 Q : ℝ) (h_d1_greater_d2 : d1 > d2)

def volume_pyramid_with_rhombus_base (d1 d2 Q : ℝ) (h_d1_greater_d2 : d1 > d2) : ℝ :=
  d1 / 12 * Real.sqrt (16 * Q ^ 2 - d1 ^ 2 * d2 ^ 2)

theorem volume_pyramid_eq :
  ∀ (d1 d2 Q : ℝ) (h_d1_greater_d2 : d1 > d2),
    volume_pyramid_with_rhombus_base d1 d2 Q h_d1_greater_d2 =
      d1 / 12 * Real.sqrt (16 * Q ^ 2 - d1 ^ 2 * d2 ^ 2) :=
by
  intros
  unfold volume_pyramid_with_rhombus_base
  sorry

end volume_pyramid_eq_l814_814422


namespace sum_digits_of_series_is_300_l814_814193

def sumSeries (n : ℕ) : ℕ :=
  let rec sumDigits (acc : ℕ) : ℕ :=
    if n = 0 then acc
    else sumDigits ((acc + (10 ^ n - 2)) % 10)
  sumDigits 0

theorem sum_digits_of_series_is_300 :
  sumSeries 150 = 300 :=
sorry

end sum_digits_of_series_is_300_l814_814193


namespace line_equation_from_angle_line_equation_from_intercepts_l814_814241

-- Define the line λ passing through point (3, 2)
def passes_through (x y: ℝ) (P: ℝ × ℝ) : Prop := P = (3, 2)

-- Define the angle of inclination property
def angle_of_inclination_twice (λ: ℝ × ℝ → ℝ) (ρ: ℝ × ℝ → ℝ) : Prop :=
  let α := atan (λ (1, 0))
  let β := atan (ρ (1, 0))
  β = 2 * α

-- Define the line equation
def line_equation (a b c: ℝ) (P: ℝ × ℝ) : Prop := 
  a * P.1 + b * P.2 + c = 0

-- Problem statement when the angle of inclination is twice
theorem line_equation_from_angle:
  (∀ P, passes_through P.1 P.2 P) ∧ angle_of_inclination_twice
  (λ P, (P.1 - 3) / (P.2 - 2)) (λ P, (P.1 - 3) / (P.2 - 8 / 15 * (P.1 - 3))) 
  → line_equation (-8) 15 (-6) (3, 2) :=
by sorry

-- Problem statement when the intercepts on the two coordinate axes are equal
theorem line_equation_from_intercepts:
  (∀ P, passes_through P.1 P.2 P) ∧ 
  (line_equation 1 1 (-5) (3, 2) ∨ line_equation 2 (-3) 0 (3, 2)) :=
by sorry

end line_equation_from_angle_line_equation_from_intercepts_l814_814241


namespace fraction_to_decimal_l814_814574

-- We define the fraction and its simplified form
def fraction : ℚ := 58 / 160
def simplified_fraction : ℚ := 29 / 80

-- We state that the fraction simplifies correctly
lemma simplify_fraction : fraction = simplified_fraction := by
  sorry

-- Define the factorization of the denominator
def denominator_factorization : ℕ := 2^4 * 5

-- Verify the fraction when multiplied by 125/125
def equalized_fraction : ℚ := 29 * 125 / 10000

-- State the final result as a decimal
theorem fraction_to_decimal : fraction = 0.3625 := by
  sorry

end fraction_to_decimal_l814_814574


namespace total_games_played_l814_814818

theorem total_games_played :
  let teams := 50
  let regular_season_games := (teams * (teams - 1)) / 2
  let rivalry_groups := 10
  let teams_per_group := 5
  let additional_games_per_group := (teams_per_group * 2)
  let total_additional_games := rivalry_groups * additional_games_per_group
  regular_season_games + total_additional_games = 1325 :=
by
  let teams := 50
  let regular_season_games := (teams * (teams - 1)) / 2
  let rivalry_groups := 10
  let teams_per_group := 5
  let additional_games_per_group := (teams_per_group * 2)
  let total_additional_games := rivalry_groups * additional_games_per_group
  show regular_season_games + total_additional_games = 1325
  from sorry

end total_games_played_l814_814818


namespace modulus_of_complex_l814_814989

-- Given conditions
def i : ℂ := complex.I
variables {x y : ℝ}

-- The theorem to be proved
theorem modulus_of_complex (h : (x + 2 * i) * i = y - i) : complex.abs (x - y * i) = real.sqrt 5 :=
sorry

end modulus_of_complex_l814_814989


namespace angle_between_two_vectors_l814_814266

noncomputable theory
open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]

def is_unit_vector (v : V) : Prop := ∥v∥ = 1

theorem angle_between_two_vectors (a b : V) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (h : ∥a - (2 : ℝ) • b∥ = real.sqrt 3) :
  real.angle a b = real.pi / 3 :=
sorry

end angle_between_two_vectors_l814_814266


namespace ratio_of_perimeters_l814_814484

theorem ratio_of_perimeters (s : ℝ) : 
  (let d := s * Real.sqrt 2 in
  let S := 4 * s in
  let P := 4 * s in
  let P' := 4 * S in
  (P' / P)) = 4 :=
by
  let d := s * Real.sqrt 2
  let S := 4 * s
  let P := 4 * s
  let P' := 4 * S
  have h : P' / P = 4 := sorry
  exact h

end ratio_of_perimeters_l814_814484


namespace total_pieces_l814_814038

def gum_packages : ℕ := 28
def candy_packages : ℕ := 14
def pieces_per_package : ℕ := 6

theorem total_pieces : (gum_packages * pieces_per_package) + (candy_packages * pieces_per_package) = 252 :=
by
  sorry

end total_pieces_l814_814038


namespace april_plant_arrangements_l814_814913

/-- April has five different basil plants and three different tomato plants. Prove that the number of ways to arrange
    the plants in a row, given all the basil plants must be next to each other and one specific basil plant must always
    be at one end of the basil group, is 576. -/
theorem april_plant_arrangements : 
  let basil_plants := 5
  let tomato_plants := 3
  let total_plants := basil_plants + tomato_plants - 4
  let factorial (n : ℕ) := if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_plants) * (factorial 4) = 576 :=
by sorry

end april_plant_arrangements_l814_814913


namespace total_diamonds_G10_l814_814058

-- Condition Definitions
def initial_diamonds : ℕ := 1
def added_diamonds (n : ℕ) : ℕ := if n = 1 then 0 else 8 * (n-1) * n

-- Main Statement: Total number of diamonds in G10
theorem total_diamonds_G10 : (∑ k in Finset.range 10, added_diamonds k) = 721 :=
by
  sorry

end total_diamonds_G10_l814_814058


namespace min_value_of_func_l814_814954

theorem min_value_of_func : 
  let y := λ x : ℝ, 2 * x^3 - 3 * x^2 - 12 * x + 5 in
  ∃ c ∈ set.Icc (0 : ℝ) 3, ∀ x ∈ set.Icc (0 : ℝ) 3, y c ≤ y x :=
begin
  let y := λ x : ℝ, 2 * x^3 - 3 * x^2 - 12 * x + 5,
  use 2,
  split,
  { split; linarith },
  { intro x,
    intro hx,
    have h1 : x = 0 ∨ x = 3 ∨ (0 < x ∧ x < 3),
    { by_cases hx0 : x = 0; by_cases hx3 : x = 3; tauto },
    cases h1 with h0 h1; cases h1 with h3 h1,
    { rw h0, linarith },
    { rw h3, linarith },
    { cases h1 with hxl hxr,
      have h_deriv : deriv y x = 6 * x^2 - 6 * x - 12,
      { dsimp [y],
        norm_num [has_deriv_at, deriv],
        simp only [mul_add, add_mul, mul_sub], },
      have := polynomial.ext_iff.1 h_deriv,
      rw [mlocalext, sub_eq_zero] at this,
      have hy' : deriv y x = (6 * (x - 2) * (x + 1)),
      { convert this using 2,
        ring },
      replace hy' := eq.symm hy',
      simp only [mul_eq_zero, sub_eq_zero, add_eq_zero_iff_eq_neg] at hy',
      cases hy',
      { exfalso, linarith },
      { cases hy',
        { exfalso, linarith },
        { rw hy', linarith } } } }
end

end min_value_of_func_l814_814954


namespace collinearity_of_centers_and_point_O_l814_814080

-- Given Definitions
variables {A B C O : Point}
variable {r : Real}
variables {c₁ c₂ c₃ : Circle}

-- Conditions
axiom common_point_O : O ∈ c₁ ∧ O ∈ c₂ ∧ O ∈ c₃
axiom equal_radii : c₁.radius = r ∧ c₂.radius = r ∧ c₃.radius = r
axiom touches_sides {c : Circle} {p1 p2 : Point} : c ∈ {p1, p2} → ∃ d, d ∈ c ∧ d ∈ Line(p1, p2)

-- Definitions of circumcenter and incenter
def incenter (t : Triangle) : Point := sorry -- definition of incenter
def circumcenter (t : Triangle) : Point := sorry -- definition of circumcenter

theorem collinearity_of_centers_and_point_O :
  let I := incenter (Triangle.mk A B C),
      C := circumcenter (Triangle.mk A B C) in
  collinear I C O :=
sorry

end collinearity_of_centers_and_point_O_l814_814080


namespace total_profit_l814_814903

variable (A_s B_s C_s : ℝ)
variable (A_p : ℝ := 14700)
variable (P : ℝ)

theorem total_profit
  (h1 : A_s + B_s + C_s = 50000)
  (h2 : A_s = B_s + 4000)
  (h3 : B_s = C_s + 5000)
  (h4 : A_p = 14700) :
  P = 35000 :=
sorry

end total_profit_l814_814903


namespace correct_answers_l814_814237

open Set Finset

noncomputable def problem (teams_A teams_B : Finset ℕ) (hpart : teams_A ∪ teams_B = {0, 1, 2, 3, 4, 5, 6, 7}) (hw : ∀ t ∈ teams_A, t < 4 ∨ t >= 4 → t < 3 ∨ t >= 3 ) : ℚ :=
  (calc_probability teams_A teams_B)/2

theorem correct_answers:
  ∀ (teams : Finset ℕ) (h_div_in_half : (|teams| = 8) -> ( (teams = teams_A ∪ teams_B) ∧ (|teams_A| = 4) ∧ (|teams_B| = 4))),
  (Probability({ one_group_exactly_two_weak teams_A teams_B }) = 6 / 7) ∧ (Probability({ at_least_two_weak teams_A }) = 1 / 2) :=
begin
  sorry
end

end correct_answers_l814_814237


namespace arithmetic_mean_l814_814097

theorem arithmetic_mean (a b : ℚ) (h1 : a = 3/7) (h2 : b = 6/11) :
  (a + b) / 2 = 75 / 154 :=
by
  sorry

end arithmetic_mean_l814_814097


namespace total_wolves_l814_814524

theorem total_wolves (x y : ℕ) :
  (x + 2 * y = 20) →
  (4 * x + 3 * y = 55) →
  (x + y = 15) :=
by
  intro h1 h2
  sorry

end total_wolves_l814_814524


namespace total_attendance_l814_814534

theorem total_attendance (A C : ℕ) (adult_ticket_price child_ticket_price total_revenue : ℕ) 
(h1 : adult_ticket_price = 11) (h2 : child_ticket_price = 10) (h3 : total_revenue = 246) 
(h4 : C = 7) (h5 : adult_ticket_price * A + child_ticket_price * C = total_revenue) : 
A + C = 23 :=
by {
  sorry
}

end total_attendance_l814_814534


namespace measure_of_B_l814_814353

variables {R : Type} [LinearOrderedField R]

-- Define the condition relating the sides and angle measures
def condition (a b c : R) : Prop :=
  (a - b + c) * (a + b + c) = 3 * a * c

-- State the theorem to prove that B = π / 3 under the condition
theorem measure_of_B 
  {a b c : R} -- real values for sides a, b, c
  (h : condition a b c) -- given condition
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :  -- ensure sides are positive lengths
  ∠B = π / 3 := 
sorry

end measure_of_B_l814_814353


namespace find_x_l814_814204

theorem find_x (x : ℤ) (h : 3^7 * 3^x = 81) : x = -3 :=
by
  sorry

end find_x_l814_814204


namespace tangent_line_intersecting_line_l814_814977

noncomputable def point := (2 : ℝ, 2 : ℝ)
noncomputable def circle_center := (3 : ℝ, 0 : ℝ)
noncomputable def radius := 1 : ℝ

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 8 = 0

def tangent_line_eq1 : Prop := ∀ x y : ℝ, x = 2
def tangent_line_eq2 : Prop := ∀ x y : ℝ, 3 * x + 4 * y - 14 = 0

def intersecting_line_eq1 : Prop := ∀ x y : ℝ, x + y - 4 = 0
def intersecting_line_eq2 : Prop := ∀ x y : ℝ, 7 * x + y - 16 = 0

theorem tangent_line (x y : ℝ) (h : circle_eq x y) :
    (tangent_line_eq1 x y ∨ tangent_line_eq2 x y) := sorry

theorem intersecting_line (x y : ℝ) (h : circle_eq x y) :
    (intersecting_line_eq1 x y ∨ intersecting_line_eq2 x y) := sorry

end tangent_line_intersecting_line_l814_814977


namespace moving_west_denotation_l814_814327

/-- 
If moving east for 3 meters is denoted as +3m,
then moving west for 4 meters is denoted as -4m.
-/
theorem moving_west_denotation : (moving_east (3 : ℤ) = 3) → (moving_west (4 : ℤ) = -4) := 
by 
  intro h
  sorry

-- Definitions used in the conditions
def moving_east (n : ℤ) : ℤ := n

def moving_west (n : ℤ) : ℤ := -n

end moving_west_denotation_l814_814327


namespace running_between_wickets_percentage_l814_814851

theorem running_between_wickets_percentage : 
    ∀ (total_runs boundaries sixes : ℕ),
    total_runs = 120 →
    boundaries = 5 →
    sixes = 5 →
    let runs_from_boundaries := boundaries * 4
    let runs_from_sixes := sixes * 6
    let runs_from_boundaries_and_sixes := runs_from_boundaries + runs_from_sixes
    let runs_between_wickets := total_runs - runs_from_boundaries_and_sixes
    (runs_between_wickets * 100 / total_runs) ≈ 58.33 :=
by
    intros total_runs boundaries sixes h_total_runs h_boundaries h_sixes
    let runs_from_boundaries := boundaries * 4
    let runs_from_sixes := sixes * 6
    let runs_from_boundaries_and_sixes := runs_from_boundaries + runs_from_sixes
    let runs_between_wickets := total_runs - runs_from_boundaries_and_sixes
    have h1 : runs_from_boundaries = 20 := by simp [h_boundaries]
    have h2 : runs_from_sixes = 30 := by simp [h_sixes]
    have h3 : runs_from_boundaries_and_sixes = 50 := by simp [h1, h2]
    have h4 : runs_between_wickets = 70 := by simp [h_total_runs, h3]
    have h5 : (runs_between_wickets * 100 / total_runs : ℚ) = 58.33 := sorry
    exact h5

end running_between_wickets_percentage_l814_814851


namespace bella_bracelets_l814_814543

theorem bella_bracelets (h_beads_per_bracelet : Nat)
  (h_initial_beads : Nat) 
  (h_additional_beads : Nat) 
  (h_friends : Nat):
  h_beads_per_bracelet = 8 →
  h_initial_beads = 36 →
  h_additional_beads = 12 →
  h_friends = (h_initial_beads + h_additional_beads) / h_beads_per_bracelet →
  h_friends = 6 :=
by
  intros h_beads_per_bracelet_eq h_initial_beads_eq h_additional_beads_eq h_friends_eq
  subst_vars
  sorry

end bella_bracelets_l814_814543


namespace least_pos_base10_num_with_eight_binary_digits_l814_814467

theorem least_pos_base10_num_with_eight_binary_digits :
  ∃ n : ℕ, (n ≥ 1) ∧ (nat.binary_bits n = 8) ∧ n = 128 := by
  sorry

end least_pos_base10_num_with_eight_binary_digits_l814_814467


namespace theta_in_fourth_quadrant_l814_814671

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.tan θ < 0) : 
  (π < θ ∧ θ < 2 * π) :=
by
  sorry

end theta_in_fourth_quadrant_l814_814671


namespace fraction_to_decimal_l814_814578

theorem fraction_to_decimal : (58 : ℚ) / 160 = 0.3625 := 
by sorry

end fraction_to_decimal_l814_814578


namespace minimum_distance_l814_814248

-- Define curve C1 with parametric equations
def C1 (t : ℝ) : ℝ × ℝ := (-4 + Real.cos t, 3 + Real.sin t)

-- Define curve C2 with general and parametric equations
def C2 : ℝ × ℝ := (64, 9)
def parametric_C2 (θ : ℝ) : ℝ × ℝ := (8 * Real.cos θ, 3 * Real.sin θ)

-- Define line C3
def C3 (x y : ℝ) : ℝ := x - 2 * y - 7

-- Define distance function
def distance_to_line (x y : ℝ) : ℝ := Real.abs (C3 x y) / Real.sqrt (1^2 + (-2)^2)

-- Given conditions
def P : ℝ × ℝ := C1 (Real.pi / 2)
def Q (θ : ℝ) : ℝ × ℝ := parametric_C2 θ
def M (θ : ℝ) : ℝ × ℝ := 
  let P := P
  let Q := Q θ
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- The main theorem to prove
theorem minimum_distance : ∃ θ : ℝ, distance_to_line ((M θ).1) ((M θ).2) = 8 * Real.sqrt 5 / 5 := sorry

end minimum_distance_l814_814248


namespace count_of_abundant_numbers_less_than_50_l814_814308

-- Define the proper divisors of a number
def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d => d < n ∧ n % d = 0)

-- Define what it means for a number to be abundant
def is_abundant (n : ℕ) : Prop :=
  (proper_divisors n).sum > n

-- Define the set of numbers less than 50 which are abundant
def abundant_numbers_less_than_50 : List ℕ :=
  (List.range 50).filter is_abundant

-- The theorem stating the number of abundant numbers less than 50
theorem count_of_abundant_numbers_less_than_50 : abundant_numbers_less_than_50.length = 9 := 
  sorry

end count_of_abundant_numbers_less_than_50_l814_814308


namespace number_of_true_propositions_l814_814007

-- Defining types for lines and planes
variables (l m n : Line) (α β γ : Plane)

-- Conditions provided in the problem
axiom prop1 : l ⊥ α → m ⊥ l → m ⊥ β → α ⊥ β
axiom prop2 : m ∈ β → n = projection_of_line l β → m ⊥ n → m ⊥ l
axiom prop3 : α ⊥ β → α ⊥ γ → α ∥ β

-- Theorem stating the number of true propositions 
theorem number_of_true_propositions (l m n : Line) (α β γ : Plane) : 
  nat := 2 :=
sorry

end number_of_true_propositions_l814_814007


namespace find_n_l814_814675

theorem find_n (n : ℤ) (h : Real.sqrt (10 + n) = 9) : n = 71 :=
sorry

end find_n_l814_814675


namespace inequality_solution_l814_814208

open Real

theorem inequality_solution (x : ℝ) :
  (1 / (x^2 + 1) < 5 / x + 21 / 10) ↔ (x ∈ set.Ioo (-∞) (-1 / 2) ∪ set.Ioo 0 ∞) := 
sorry

end inequality_solution_l814_814208


namespace sum_of_inverse_geometric_sequence_l814_814279

variable {a_n : ℕ → ℝ}  -- sequence a_n
variable {S_n : ℕ → ℝ}  -- sum of first n terms of sequence a_n

noncomputable def a_n {first_term : ℝ} {q : ℝ} (n : ℕ) : ℝ := first_term * q^n
noncomputable def S_n {a_n : ℕ → ℝ} (n : ℕ) : ℝ := (list.range n).map a_n |>.sum

theorem sum_of_inverse_geometric_sequence:
    ∀ {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} (q : ℝ),
    (a_n 0 = 1) →
    (∀ n, a_n (n+1) = a_n n * q) →
    (9 * S_n 3 = S_n 6) →
    let b_n := λ n, (1 : ℝ) / a_n n in
    (S_n 5 = 31 / 16) :=
by
  intros a_n S_n q h_first_term h_geometric h_sum_eq
  have := sorry
  exact this

end sum_of_inverse_geometric_sequence_l814_814279


namespace eccentricity_range_of_ellipse_l814_814000

theorem eccentricity_range_of_ellipse 
  (a b c e : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (hc2 : c^2 + b^2 = a^2)
  (he : e = c / a)
  (hP : ∃ m : ℝ, ∃ x : ℝ, x = (a^2 / c) ∧ (x, m) ∈ line x (a^2 / c))
  (h_perpendicular_bisector : ∃ P : ℝ × ℝ, exists F1 : ℝ × ℝ, exists F2 : ℝ × ℝ, F1 = (-c, 0) ∧ F2 = (c, 0) ∧ ∃ m, P = (a^2 / c, m) ∧ (midpoint P F1 ∉ perpendicular_bisector F1 F2))
  : ∃ e : ℝ, (e >= (sqrt 3) / 3) ∧ e < 1 :=
sorry

end eccentricity_range_of_ellipse_l814_814000


namespace integral_sin_pi_div_2_l814_814924

theorem integral_sin_pi_div_2 : ∫ x in 0..(Real.pi / 2), Real.sin x = 1 := 
by
  -- proof omitted
  sorry

end integral_sin_pi_div_2_l814_814924


namespace find_p_l814_814187

theorem find_p
  (a r1 b r2 p: ℝ)
  (h1 : 0 < p)
  (intersection : (8:ℝ, 4:ℝ) ∈ set.inter (sphere (a,r1) r1) (sphere (b,r2) r2))
  (product_radii : r1 * r2 = 77)
  (tangent_xaxis : (sphere (a,r1) r1).cinter (sphere (a,0) r1))
  (tangent_line1 : tangent (sphere (a,r1) r1) (line (2:ℝ, 3:ℝ)))
  (tangent_vertical : tangent (sphere (a,r1) r1) (line ((p:ℝ), 0:ℝ))) :
  p = 12 :=
sorry

end find_p_l814_814187


namespace smallest_even_integer_sum_l814_814073

theorem smallest_even_integer_sum :
  let S := (Finset.range 30).sum (λ n, 2 * (n + 1))
  let sum_consec_evens (n : ℕ) := n + (n + 2) + (n + 4) + (n + 6) + (n + 8)
  ∃ n, sum_consec_evens n = S ∧ n = 182 :=
by
  let S := (Finset.range 30).sum (λ n, 2 * (n + 1))
  let sum_consec_evens (n : ℕ) := n + (n + 2) + (n + 4) + (n + 6) + (n + 8)
  existsi 182
  sorry

end smallest_even_integer_sum_l814_814073


namespace Collin_total_petals_l814_814553

-- Definitions of the conditions
def initial_flowers_Collin : ℕ := 25
def flowers_Ingrid : ℕ := 33
def petals_per_flower : ℕ := 4
def third_of_flowers_Ingrid : ℕ := flowers_Ingrid / 3

-- Total number of flowers Collin has after receiving from Ingrid
def total_flowers_Collin : ℕ := initial_flowers_Collin + third_of_flowers_Ingrid

-- Total number of petals Collin has
def total_petals_Collin : ℕ := total_flowers_Collin * petals_per_flower

-- The theorem to be proved
theorem Collin_total_petals : total_petals_Collin = 144 := by
  -- Proof goes here
  sorry

end Collin_total_petals_l814_814553


namespace hannah_monday_run_l814_814660

-- Definitions of the conditions
def ran_on_wednesday : ℕ := 4816
def ran_on_friday : ℕ := 2095
def extra_on_monday : ℕ := 2089

-- Translations to set the total combined distance and the distance ran on Monday
def combined_distance := ran_on_wednesday + ran_on_friday
def ran_on_monday := combined_distance + extra_on_monday

-- A statement to show she ran 9 kilometers on Monday
theorem hannah_monday_run :
  ran_on_monday = 9000 / 1000 * 1000 := sorry

end hannah_monday_run_l814_814660


namespace objects_meet_at_10_seconds_l814_814088

theorem objects_meet_at_10_seconds (t : ℕ) (d1 d2 : ℝ) :
  let d1 := 3 * t * (t + 1)
  let d2 := 0.012 * (t - 5)
  (d1 + d2 = 390) → t = 10 :=
by
  intros
  sorry

end objects_meet_at_10_seconds_l814_814088


namespace product_first_11_terms_eq_2_pow_11_l814_814276

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_first_11_terms_eq_2_pow_11 
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h5_6_7 : a 5 * a 6 * a 7 = 8) :
  (finset.range 11).prod a = 2 ^ 11 :=
begin
  sorry
end

end product_first_11_terms_eq_2_pow_11_l814_814276


namespace dog_weight_ratio_l814_814788

theorem dog_weight_ratio :
  ∀ (brown black white grey : ℕ),
    brown = 4 →
    black = brown + 1 →
    grey = black - 2 →
    (brown + black + white + grey) / 4 = 5 →
    white / brown = 2 :=
by
  intros brown black white grey h_brown h_black h_grey h_avg
  sorry

end dog_weight_ratio_l814_814788


namespace virginia_final_eggs_l814_814094

theorem virginia_final_eggs : 
  ∀ (initial_eggs : ℕ) (taken_away : ℕ), 
  initial_eggs = 96 ∧ taken_away = 3 → 
  initial_eggs - taken_away = 93 :=
by
  intros initial_eggs taken_away h,
  cases h with h_initial h_taken,
  rw [h_initial, h_taken],
  norm_num

end virginia_final_eggs_l814_814094


namespace probability_three_white_balls_l814_814500

noncomputable def probability_all_white (white black total_drawn : ℕ) : ℚ :=
  (nat.choose white total_drawn : ℚ) / (nat.choose (white + black) total_drawn : ℚ)

theorem probability_three_white_balls :
  probability_all_white 7 8 3 = 1 / 13 :=
by 
  sorry

end probability_three_white_balls_l814_814500


namespace find_f_l814_814003

noncomputable def f (x : ℝ) : ℝ :=
  let a b c : ℝ in (c / (a - b)) * x + (c / (a + b))

theorem find_f (a b c : ℝ) (x : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ a ≠ -b) :
  a * f (x - 1) + b * f (1 - x) = c * x :=
by
  sorry

end find_f_l814_814003


namespace sum_reciprocal_squares_lt_two_l814_814776

theorem sum_reciprocal_squares_lt_two (n : ℕ) (hn : n > 0) : 
  (∑ k in Finset.range n.succ, 1 / (k * k : ℝ)) < 2 := sorry

end sum_reciprocal_squares_lt_two_l814_814776


namespace min_sum_xy_l814_814264

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (hcond : ↑(1 / x) + ↑(1 / y) = 1 / 15) : x + y = 64 :=
sorry

end min_sum_xy_l814_814264


namespace tennis_players_count_l814_814710

theorem tennis_players_count (total_members members_no_sport members_badminton members_both : ℕ) (h1 : total_members = 40) (h2 : members_no_sport = 5) (h3 : members_badminton = 20) (h4 : members_both = 3) : 
  (total_members - members_no_sport) = (members_badminton - members_both) + (total_members - members_no_sport - (members_badminton - members_both + members_both)) + members_both ∧ 
  (18 : ℕ) = 15 + members_both :=
by 
  split;
  sorry

end tennis_players_count_l814_814710


namespace line_through_point_segment_circle_l814_814596

theorem line_through_point_segment_circle (x y : ℝ) : 
  (∃ l : ℝ, (l = x ∧ x = 3) ∨ l = (3*x - 4*y + 15) = 0) ↔ 
  (x = 3 ∧ y = 6) ∧ (x^2 + y^2 = 25) ∧ (6*x*y - 15) = 8 := 
              sorry

end line_through_point_segment_circle_l814_814596


namespace curve_circle_no_common_points_curve_circle_one_common_point_curve_circle_two_common_points_curve_circle_four_common_points_l814_814635

noncomputable def curveC (k : ℝ) : ℝ → ℝ → Prop := λ x y, (y - k * x) * (y - k * x + 7 / 4) = 0

noncomputable def circleM : ℝ → ℝ → Prop := λ x y, (x - 2) ^ 2 + (y - 1) ^ 2 = 1

theorem curve_circle_no_common_points (k : ℝ) :
  (k < 0 ∨ k > 35 / 12) → ∀ x y : ℝ, ¬(curveC k x y ∧ circleM x y) :=
sorry

theorem curve_circle_one_common_point (k : ℝ) :
  k = 3 / 4 → ∃! x y : ℝ, curveC k x y ∧ circleM x y :=
sorry

theorem curve_circle_two_common_points (k : ℝ) :
  (0 < k ∧ k < 3 / 4) → ∃ x1 y1 x2 y2 : ℝ, (x1 ≠ x2 ∨ y1 ≠ y2) ∧ curveC k x1 y1 ∧ circleM x1 y1 ∧ curveC k x2 y2 ∧ circleM x2 y2 :=
sorry

theorem curve_circle_four_common_points (k : ℝ) :
  (3 / 4 < k ∧ k < 4 / 3) → ∃ x1 y1 x2 y2 x3 y3 x4 y4 : ℝ, (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧ curveC k x1 y1 ∧ circleM x1 y1 ∧ curveC k x2 y2 ∧ circleM x2 y2 ∧ curveC k x3 y3 ∧ circleM x3 y3 ∧ curveC k x4 y4 ∧ circleM x4 y4 :=
sorry

end curve_circle_no_common_points_curve_circle_one_common_point_curve_circle_two_common_points_curve_circle_four_common_points_l814_814635


namespace average_fixed_points_is_one_l814_814143

noncomputable def fixed_points (σ : Perm ℕ) (n : ℕ) : Finset ℕ :=
  Finset.filter (λ k => σ k = k) (Finset.range n)

noncomputable def average_fixed_points (n : ℕ) : ℚ :=
  (Finset.sum (Equiv.perm_fin n) (λ σ => (fixed_points σ n).card)) / nat.factorial n

theorem average_fixed_points_is_one (n : ℕ) : average_fixed_points n = 1 := by
  sorry

end average_fixed_points_is_one_l814_814143


namespace fraction_to_decimal_l814_814573

-- We define the fraction and its simplified form
def fraction : ℚ := 58 / 160
def simplified_fraction : ℚ := 29 / 80

-- We state that the fraction simplifies correctly
lemma simplify_fraction : fraction = simplified_fraction := by
  sorry

-- Define the factorization of the denominator
def denominator_factorization : ℕ := 2^4 * 5

-- Verify the fraction when multiplied by 125/125
def equalized_fraction : ℚ := 29 * 125 / 10000

-- State the final result as a decimal
theorem fraction_to_decimal : fraction = 0.3625 := by
  sorry

end fraction_to_decimal_l814_814573


namespace find_specific_number_l814_814791

theorem find_specific_number :
  ∃ (n : ℕ), (400 < n ∧ n < 500) ∧ 
             (let (x, y, z) := (n / 100, (n / 10) % 10, n % 10) in x + y + z = 9) ∧ 
             (let (x, y, z) := (n / 100, (n / 10) % 10, n % 10) in 36 * (100 * x + 10 * y + z) = 47 * (100 * z + 10 * y + x)) ∧
             n = 423 := 
sorry

end find_specific_number_l814_814791


namespace sum_identity_l814_814012

theorem sum_identity (S a b B : ℕ → ℝ) (n : ℕ) 
  (h : ∀ i, B i = (finset.range (i+1)).sum (λ j, b j))
  (S_eq : S n = (finset.range n).sum (λ i, a i * b i)) :
  S n = (finset.range (n-1)).sum (λ i, (a i - a (i + 1)) * B (i+1)) + a (n-1) * B n :=
by
  sorry

end sum_identity_l814_814012


namespace periodicity_and_sum_l814_814518

variable {f : ℝ → ℝ}

-- Given conditions
axiom cond1 : ∀ x : ℝ, f (2 + x) + f (2 - x) = 4
axiom cond2 : ∀ x : ℝ, f (x + 1) = f (1 - x + 1)

-- Proving the statements
theorem periodicity_and_sum :
  (∀ x : ℝ, f (x + 4) = f x) ∧
  (f 0 + f 1 + f 2 + ⋯ + f 2023 = 4048) :=
sorry

end periodicity_and_sum_l814_814518


namespace greatest_possible_difference_l814_814681

theorem greatest_possible_difference (x y : ℤ) (h1 : 3 < x) (h2 : x < 6) (h3 : 6 < y) (h4 : y < 8) :
  ∃ (d : ℤ), d = y - x ∧ d = 3 :=
begin
  sorry
end

end greatest_possible_difference_l814_814681


namespace distance_from_desk_to_fountain_l814_814018

-- Problem definitions with given conditions
def total_distance : ℕ := 120
def trips : ℕ := 4

-- Formulate the proof problem as a Lean theorem statement
theorem distance_from_desk_to_fountain :
  total_distance / trips = 30 :=
by
  sorry

end distance_from_desk_to_fountain_l814_814018


namespace necessary_not_sufficient_l814_814653

open Set

noncomputable def M := { x : ℝ | 0 < x ∧ x < 1 }
noncomputable def N := { x : ℝ | -2 < x ∧ x < 1 }

theorem necessary_not_sufficient (a : ℝ) :
  (∀ a, a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) :=
by
  split
  -- Proof of necessity
  sorry
  -- Proof of not sufficient condition
  sorry

end necessary_not_sufficient_l814_814653


namespace closest_point_on_parabola_to_line_is_l814_814432

-- Definitions of the parabola and the line
def parabola (x : ℝ) : ℝ := 4 * x^2
def line (x : ℝ) : ℝ := 4 * x - 5

-- Prove that the point on the parabola that is closest to the line is (1/2, 1)
theorem closest_point_on_parabola_to_line_is (x y : ℝ) :
  parabola x = y ∧ (∀ (x' y' : ℝ), parabola x' = y' -> (line x - y)^2 >= (line x' - y')^2) ->
  (x, y) = (1/2, 1) :=
by
  sorry

end closest_point_on_parabola_to_line_is_l814_814432


namespace total_weight_on_scale_l814_814401

def weight_blue_ball : ℝ := 6
def weight_brown_ball : ℝ := 3.12

theorem total_weight_on_scale :
  weight_blue_ball + weight_brown_ball = 9.12 :=
by sorry

end total_weight_on_scale_l814_814401


namespace parallelogram_complex_number_l814_814615

theorem parallelogram_complex_number (B : ℂ) :
  let O : ℂ := 0
  let A : ℂ := 3 + 2 * complex.i
  let C : ℂ := -2 + 4 * complex.i
  OABC_parallelogram : true
  B = A + C :=
begin
  sorry
end

end parallelogram_complex_number_l814_814615


namespace collin_total_petals_l814_814556

variable (collin_flowers initially given_flowers received_flowers each_flower_petals total_petals : ℕ)

-- Conditions as definitions in Lean
def collin_initial_flowers := 25
def ingrid_total_flowers := 33 / 3
def each_flower_petals := 4

-- Collin receives 11 flowers from Ingrid
def received_flowers := ingrid_total_flowers

-- Total flowers Collin has
def collin_flowers := 25 + received_flowers

-- Total petals Collin has
def total_petals := collin_flowers * each_flower_petals

-- Proof that Collin has 144 petals in total
theorem collin_total_petals : total_petals = 144 := by
  sorry

end collin_total_petals_l814_814556


namespace squirrel_calories_l814_814544

def rabbits_caught_per_hour := 2
def rabbits_calories := 800
def squirrels_caught_per_hour := 6
def extra_calories_squirrels := 200

theorem squirrel_calories : 
  ∀ (S : ℕ), 
  (6 * S = (2 * 800) + 200) → S = 300 := by
  intros S h
  sorry

end squirrel_calories_l814_814544


namespace least_pos_base10_num_with_eight_binary_digits_l814_814469

theorem least_pos_base10_num_with_eight_binary_digits :
  ∃ n : ℕ, (n ≥ 1) ∧ (nat.binary_bits n = 8) ∧ n = 128 := by
  sorry

end least_pos_base10_num_with_eight_binary_digits_l814_814469


namespace cannot_make_120_cents_with_6_coins_l814_814405

def Coin := ℕ → ℕ -- represents a number of each type of coin

noncomputable def coin_value (c : Coin) : ℕ :=
  c 0 * 1 + c 1 * 5 + c 2 * 10 + c 3 * 25

def total_coins (c : Coin) : ℕ :=
  c 0 + c 1 + c 2 + c 3

theorem cannot_make_120_cents_with_6_coins (c : Coin) (h1 : total_coins c = 6) :
  coin_value c ≠ 120 :=
sorry

end cannot_make_120_cents_with_6_coins_l814_814405


namespace identifier_count_l814_814178

-- Implementing the equivalent proof problem statement in Lean 4
theorem identifier_count (first_char choices : ℕ) (digit_count : ℕ) : 
  first_char = 26 → choices = 26 + 10 → digit_count = digit_count →
  let num_identifiers := first_char + (first_char * choices) + (first_char * choices ^ 2) in
  num_identifiers = 34658 := 
by
  intro h1 h2 h3
  let first_char := 26
  let choices := 36
  let num_identifiers := first_char + (first_char * choices) + (first_char * choices ^ 2)
  have h4 : num_identifiers = 34658 := by sorry
  exact h4

end identifier_count_l814_814178


namespace rocks_placed_l814_814390

theorem rocks_placed (total_rocks remaining_rocks : ℕ) (h : total_rocks = 189 ∧ remaining_rocks = 64) : total_rocks - remaining_rocks = 125 :=
by
  cases h with h_total h_remaining
  rw [h_total, h_remaining]
  exact rfl

end rocks_placed_l814_814390


namespace cyclic_pentagon_area_l814_814486

theorem cyclic_pentagon_area :
  ∃ (a b c : ℤ), (c > 0) ∧ (nat.gcd a c = 1) ∧ 
  (a = 27) ∧ (b = 15) ∧ (c = 2) ∧ 
  (100 * a + 10 * b + c = 2752) ∧ 
  ∀ (A B C D E X Y : ℝ × ℝ), 
    (cyclic A B C D E) ∧ (convex A B C D E) ∧
    (AC = BD) ∧ (BD = CE) ∧ 
    (dist A B = 6) ∧ (dist X Y = 4) ∧ (dist Y E = 7) 
      → 
    (area A B C D E = (27 * sqrt 15) / 2) := 
begin
  sorry
end

end cyclic_pentagon_area_l814_814486


namespace sec_150_eq_neg_2_sqrt_3_div_3_csc_150_eq_2_l814_814921

noncomputable def sec (x : ℝ) := 1 / Real.cos x
noncomputable def csc (x : ℝ) := 1 / Real.sin x

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := by
  sorry

theorem csc_150_eq_2 : csc (150 * Real.pi / 180) = 2 := by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_csc_150_eq_2_l814_814921


namespace vertical_asymptotes_sum_l814_814209

theorem vertical_asymptotes_sum : 
  let f (x : ℝ) := (8 * x^2 - 12) / (4 * x^2 + 8 * x + 3)
  in ∑ x ∈ (Finset.mk {x | 4 * x^2 + 8 * x + 3 = 0} sorry), x = -2 :=
by sorry

end vertical_asymptotes_sum_l814_814209


namespace Q_moves_clockwise_with_2omega_l814_814689

noncomputable def pointP (ω t : ℝ) : ℝ × ℝ :=
  (Real.cos (ω * t), Real.sin (ω * t))

noncomputable def pointQ (x y : ℝ) : ℝ × ℝ :=
  (-2 * x * y, y ^ 2 - x ^ 2)

theorem Q_moves_clockwise_with_2omega (ω t : ℝ) :
  let (x, y) := pointP ω t
  let (qx, qy) := pointQ x y
  qx = Real.cos (2 * ω * t + π / 2) ∧ qy = Real.sin (2 * ω * t + π / 2) :=
by
  intro x y
  cases pointP ω t with xp yp
  have : (x, y) = (xp, yp) by sorry
  rw this
  cases pointQ x y with qx qy
  have : (qx, qy) = (-Real.sin (2 * ω * t), -Real.cos (2 * ω * t)) by sorry
  rw this
  split
  {
    have : Real.cos (2 * ω * t + π / 2) = -Real.sin (2 * ω * t) by sorry
    exact this
  }
  {
    have : Real.sin (2 * ω * t + π / 2) = -Real.cos (2 * ω * t) by sorry
    exact this
  }

end Q_moves_clockwise_with_2omega_l814_814689


namespace only_possible_b_l814_814587

theorem only_possible_b (b : ℕ) (h : ∃ a k l : ℕ, k ≠ l ∧ (b > 0) ∧ (a > 0) ∧ (b ^ (k + l)) ∣ (a ^ k + b ^ l) ∧ (b ^ (k + l)) ∣ (a ^ l + b ^ k)) : 
  b = 1 :=
sorry

end only_possible_b_l814_814587


namespace perpendicular_distance_from_D_to_plane_l814_814762

noncomputable def perpendicular_distance_to_plane : ℝ :=
  let A := (6 : ℝ, 0 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 5 : ℝ, 0 : ℝ)
  let C := (0 : ℝ, 0 : ℝ, 4 : ℝ)
  let D := (0 : ℝ, 0 : ℝ, 0 : ℝ)
  let V := 6 * 5 * 4 -- Volume of the parallelepiped
  let area_ABC := 1/2 * 6 * 5 -- Area of triangle ABC
  V / area_ABC * 3 -- Height or perpendicular distance from D to the plane

theorem perpendicular_distance_from_D_to_plane :
  perpendicular_distance_to_plane = 24 := by
  sorry

end perpendicular_distance_from_D_to_plane_l814_814762


namespace sum_of_3rd_and_2nd_largest_l814_814819

theorem sum_of_3rd_and_2nd_largest : 
  let numbers := [10, 11, 12, 13, 14]
  let sorted_numbers := List.sort numbers
  (sorted_numbers.get 2 + sorted_numbers.get 3) = 25 :=
by
  let numbers := [10, 11, 12, 13, 14]
  let sorted_numbers := List.sort numbers
  have h : sorted_numbers = [10, 11, 12, 13, 14] := by sorry
  rw [h]
  show 12 + 13 = 25
  sorry

end sum_of_3rd_and_2nd_largest_l814_814819


namespace domino_tiling_possible_l814_814884

theorem domino_tiling_possible (n : ℕ) : n % 2 = 0 ↔ 
  ∃ (covering : (ℕ × ℕ) → bool), 
    (∀ (x y : ℕ), x < 2 * n + 1 ∧ y < 2 * n + 1 ∧ (x, y) ≠ (0, 0) → covering (x, y) ∧ covering (x + 1, y) (x + 2, y) ∨ covering (x, y) ∧ covering (x, y + 1) (x, y + 2)) ∧
    (∃ (horizontals : ℕ), horizontals = (2 * n^2 + 1) / 2) ∧
    (∃ (verticals : ℕ), verticals = (2 * n^2 + 1) / 2) :=
begin
  sorry
end

end domino_tiling_possible_l814_814884


namespace ava_distance_to_finish_l814_814542

theorem ava_distance_to_finish (total_distance ava_covered : ℕ) (h1 : total_distance = 1000) (h2 : ava_covered = 833) :
  total_distance - ava_covered = 167 := 
by 
  rw [h1, h2]
  norm_num

end ava_distance_to_finish_l814_814542


namespace Joan_seashells_l814_814728

theorem Joan_seashells : 
  ∀ (total_seashells : ℝ) (percentage_given : ℝ), 
  total_seashells = 79.5 → 
  percentage_given = 0.45 → 
  (total_seashells - (percentage_given * total_seashells)) = 43.725 :=
by
  intros total_seashells percentage_given h1 h2
  rw [h1, h2]
  norm_num
  sorry

end Joan_seashells_l814_814728


namespace employee_count_end_of_year_l814_814894

theorem employee_count_end_of_year :
  let part_time_initial := 2047
  let full_time_initial := 63109
  let contractors_initial := 1500
  let interns_initial := 333
  let consultants_initial := 918
  let full_time_attrition := 3 / 100 * full_time_initial
  let part_time_increase := (5 / 100) * part_time_initial
  let contractors_decrease := (10 / 100) * contractors_initial
  let interns_hired := 100
  let consultants_non_renewed := 200

  let full_time_end_year := full_time_initial - full_time_attrition
  let part_time_end_year := part_time_initial + part_time_increase
  let contractors_end_year := contractors_initial - contractors_decrease
  let interns_end_year := interns_initial + interns_hired
  let consultants_end_year := consultants_initial - consultants_non_renewed

  in full_time_end_year.to_nat + part_time_end_year.to_nat + contractors_end_year.to_nat +
     interns_end_year.to_nat + consultants_end_year.to_nat = 65866 := 
by
  sorry

end employee_count_end_of_year_l814_814894


namespace solve_equation_l814_814934

theorem solve_equation (x : ℝ) :
  16^x - (5/2) * (2^(2*x + 1)) + 4 = 0 ↔ x = 0 ∨ x = 1 :=
sorry

end solve_equation_l814_814934


namespace fraction_computation_l814_814609

theorem fraction_computation (p q s u : ℚ)
  (hpq : p / q = 5 / 2)
  (hsu : s / u = 7 / 11) :
  (5 * p * s - 3 * q * u) / (7 * q * u - 4 * p * s) = 109 / 14 := 
by
  sorry

end fraction_computation_l814_814609


namespace prove_problem_statement_l814_814478

noncomputable def problem_statement : Prop :=
  ∃ (k : ℤ), ∃ (x y : ℝ), (cos x)^2 + (cos y)^2 = 0.25 ∧ x + y = 5 * π / 6 ∧
    ((x = π / 2 * (2 * k + 1) ∧ y = π / 3 * (1 - 3 * k)) ∨
     (x = π / 3 * (3 * k + 1) ∧ y = π / 2 * (1 - 2 * k)))

theorem prove_problem_statement : problem_statement :=
  sorry

end prove_problem_statement_l814_814478


namespace greatest_teams_l814_814817

-- Define the number of girls and boys as constants
def numGirls : ℕ := 40
def numBoys : ℕ := 32

-- Define the greatest number of teams possible with equal number of girls and boys as teams.
theorem greatest_teams : Nat.gcd numGirls numBoys = 8 := sorry

end greatest_teams_l814_814817


namespace sum_of_d_e_f_g_l814_814951

theorem sum_of_d_e_f_g :
  ∃ (d e f g : ℕ), 
  3 * Real.sqrt (Real.cbrt 7 + Real.cbrt 6) = Real.cbrt d - Real.cbrt e + Real.cbrt f + Real.cbrt g ∧
  d + e + f + g = 96 :=
by 
  use [42, 6, 42, 6]
  split
  sorry
  sorry

end sum_of_d_e_f_g_l814_814951


namespace smallest_x_y_sum_l814_814253

theorem smallest_x_y_sum :
  ∃ x y : ℕ,
    0 < x ∧ 0 < y ∧ x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 15) ∧ (x + y = 64) := 
by
  sorry

end smallest_x_y_sum_l814_814253


namespace seniors_selected_correct_l814_814149

-- Definitions based on the conditions problem
def total_freshmen : ℕ := 210
def total_sophomores : ℕ := 270
def total_seniors : ℕ := 300
def selected_freshmen : ℕ := 7

-- Problem statement to prove
theorem seniors_selected_correct : 
  (total_seniors / (total_freshmen / selected_freshmen)) = 10 := 
by 
  sorry

end seniors_selected_correct_l814_814149


namespace S_30_is_5702886_l814_814527

-- Define the sequence {A_n} with the given conditions.
def A : ℕ → ℕ
| 0     := 1
| 1     := 1
| 2     := 2
| (n+3) := A (n+2) + A (n+1)

-- Define the sum sequence S_n = A_0 + A_1 + ... + A_n.
def S (n : ℕ) : ℕ := (Finset.range (n + 1)).sum A

-- Statement to prove: S_{30} = 5702886
theorem S_30_is_5702886 : S 30 = 5702886 :=
by {
    -- This is where the proof would go
    sorry
}

end S_30_is_5702886_l814_814527


namespace sin_eleven_pi_over_three_l814_814583

theorem sin_eleven_pi_over_three : Real.sin (11 * Real.pi / 3) = -((Real.sqrt 3) / 2) :=
by
  -- Conversion factor between radians and degrees
  -- periodicity of sine function: sin theta = sin (theta + n * 360 degrees) for any integer n
  -- the sine function is odd: sin (-theta) = -sin theta
  -- sin 60 degrees = sqrt(3)/2
  sorry

end sin_eleven_pi_over_three_l814_814583


namespace valid_bead_count_l814_814160

structure Bead :=
  (shape : String)   -- "cube" or "pyramid"
  (color : String)   -- colors: "green", "blue", "red"

inductive Shape 
| Cube
| Pyramid 

inductive Color 
| Green
| Blue
| Red

def is_solution (n : ℕ) : Prop :=
  ∃ (b : fin n → Bead), 
    (∀ i, b (i + 1) % n).shape ≠ (b i).shape ∧ (b (i + 1) % n).color ≠ (b i).color) ∧
    set.of_list (list.of_fn b).to_finset = { Bead.mk Shape.Cube Color.Green, Bead.mk Shape.Cube Color.Blue, Bead.mk Shape.Pyramid Color.Red, Bead.mk Shape.Pyramid Color.Blue }

theorem valid_bead_count :
  ∀ n : ℕ, is_solution n → (n = 8 ∨ n = 10) :=
begin
  sorry
end

end valid_bead_count_l814_814160


namespace determine_a_l814_814295

def A := {x : ℝ | x^2 + 3*x + 2 = 0}
def B (a : ℝ) := {x : ℝ | x^2 + a*x + 4 = 0}

theorem determine_a (a : ℝ) (h₁ : B(a) ≠ ∅) (h₂ : B(a) ⊆ A) : a = 4 := by
  sorry

end determine_a_l814_814295


namespace cindy_marbles_l814_814024

theorem cindy_marbles :
  let total_marbles := 1000
  let friends := 6
  let marbles_per_friend := 120
  let marbles_given_away := friends * marbles_per_friend
  let marbles_remaining := total_marbles - marbles_given_away
  7 * marbles_remaining = 1960 := by 
  let total_marbles := 1000
  let friends := 6
  let marbles_per_friend := 120
  let marbles_given_away := friends * marbles_per_friend
  let marbles_remaining := total_marbles - marbles_given_away
  calc
  7 * marbles_remaining 
  = 7 * (1000 - (6 * 120)) : by have h : marbles_remaining = total_marbles - marbles_given_away := rfl; rw [h]
  ... = 7 * 280 : by rfl
  ... = 1960 : by rfl

end cindy_marbles_l814_814024


namespace sum_of_squares_consecutive_integers_l814_814585

theorem sum_of_squares_consecutive_integers (m : ℤ) (h : m > 1) :
  ∃ (n : ℤ), m^3 = ∑ i in finset.range m, (n + i + 1)^2 ↔ m = 47 :=
by
  sorry

end sum_of_squares_consecutive_integers_l814_814585


namespace quadratic_real_roots_discriminant_quadratic_real_roots_sum_of_squares_l814_814243

theorem quadratic_real_roots_discriminant (m : ℝ) :
  (2 * (m + 1))^2 - 4 * m * (m - 1) > 0 ↔ (m > -1/2 ∧ m ≠ 0) := 
sorry

theorem quadratic_real_roots_sum_of_squares (m x1 x2 : ℝ) 
  (h1 : m > -1/2 ∧ m ≠ 0)
  (h2 : x1 + x2 = -2 * (m + 1) / m)
  (h3 : x1 * x2 = (m - 1) / m)
  (h4 : x1^2 + x2^2 = 8) : 
  m = (6 + 2 * Real.sqrt 33) / 8 := 
sorry

end quadratic_real_roots_discriminant_quadratic_real_roots_sum_of_squares_l814_814243


namespace center_of_circle_is_midpoint_of_diameter_l814_814065

-- Define the points as given in the conditions
def pointA : ℝ × ℝ := (7, -8)
def pointB : ℝ × ℝ := (-5, 4)

-- Define the function to find the midpoint of two points
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Statement to prove that the midpoint of the given points is (1, -2)
theorem center_of_circle_is_midpoint_of_diameter :
  midpoint pointA pointB = (1, -2) :=
by
  sorry

end center_of_circle_is_midpoint_of_diameter_l814_814065


namespace blue_hat_cost_l814_814458

variable (B : ℕ)
variable (totalHats : ℕ := 85)
variable (greenHatCost : ℕ := 7)
variable (greenHatsBought : ℕ := 38)
variable (totalCost : ℕ := 548)

theorem blue_hat_cost 
(h1 : greenHatsBought = 38) 
(h2 : totalHats = 85) 
(h3 : greenHatCost = 7)
(h4 : totalCost = 548) :
  let totalGreenHatCost := greenHatCost * greenHatsBought
  let totalBlueHatCost := totalCost - totalGreenHatCost
  let totalBlueHatsBought := totalHats - greenHatsBought
  B = totalBlueHatCost / totalBlueHatsBought := by
  sorry

end blue_hat_cost_l814_814458


namespace smallest_sum_l814_814254

theorem smallest_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) 
  (h_fraction : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) : x + y = 64 :=
sorry

end smallest_sum_l814_814254


namespace minimum_value_attained_at_n_10_l814_814441

theorem minimum_value_attained_at_n_10 : ∀ (n : ℕ) (h : 0 < n), 
  (n = 10) → (n / 2 + 50 / n = 10) :=
begin
  intros n h hn,
  have h1 : n / 2 = 50 / n, by sorry,
  have h2 : 2 * sqrt (n / 2 * 50 / n) = 10, by sorry,
  exact eq.trans (add_eq_of_eq_of_eq h1 h2) hn,
end

end minimum_value_attained_at_n_10_l814_814441


namespace height_of_shorter_pot_is_20_l814_814830

-- Define the conditions as given
def height_of_taller_pot := 40
def shadow_of_taller_pot := 20
def shadow_of_shorter_pot := 10

-- Define the height of the shorter pot to be determined
def height_of_shorter_pot (h : ℝ) := h

-- Define the relationship using the concept of similar triangles
theorem height_of_shorter_pot_is_20 (h : ℝ) :
  (height_of_taller_pot / shadow_of_taller_pot = height_of_shorter_pot h / shadow_of_shorter_pot) → h = 20 :=
by
  intros
  sorry

end height_of_shorter_pot_is_20_l814_814830


namespace range_of_a_minimum_value_of_b_l814_814565

def is_fixed_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop := f x₀ = x₀

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (2 * b - 1) * x + b - 2
noncomputable def g (a x : ℝ) : ℝ := -x + a / (3 * a^2 - 2 * a + 1)

theorem range_of_a (h : ∀ b : ℝ, ∃ x1 x2 : ℝ, is_fixed_point (f a b) x1 ∧ is_fixed_point (f a b) x2) : 0 < a ∧ a < 4 :=
sorry

theorem minimum_value_of_b (hx1 : is_fixed_point (f a b) x₁) (hx2 : is_fixed_point (f a b) x₂)
  (hm : g a ((x₁ + x₂) / 2) = (x₁ + x₂) / 2) (ha : 0 < a ∧ a < 4) : b ≥ 3/4 :=
sorry

end range_of_a_minimum_value_of_b_l814_814565


namespace f_inv_2016_l814_814288

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * Real.log x + b * Real.log2 x + 1

-- Assume the condition
axiom h : f a b 2016 = 3

theorem f_inv_2016 : f a b (1 / 2016) = -1 :=
by
  -- Provide a temporary proof using 'sorry'
  sorry

end f_inv_2016_l814_814288


namespace number_of_kabadi_players_l814_814778

-- Define the conditions and variables
def Kho_only := 35
def Both := 5
def Total_players := 45

-- Define the proof problem
theorem number_of_kabadi_players : ∃ K, K + Kho_only - Both = Total_players ∧ K = 15 :=
by {
  use 15,
  split,
  calc
    15 + Kho_only - Both = 15 + 35 - 5 : rfl
                ... = 15 + 30         : by rfl
                ... = 45              : by rfl,
  rfl,
}

end number_of_kabadi_players_l814_814778


namespace family_cost_l814_814222

noncomputable def cost_per_family_member (n_persons_per_dozen : ℕ) (n_dozens : ℕ) (cans_per_person : ℕ) 
  (cost_per_box : ℝ) (cans_per_box : ℕ) (n_family_members : ℕ) : ℝ :=
  let total_persons := n_persons_per_dozen * n_dozens in
  let total_cans := total_persons * cans_per_person in
  let total_boxes := total_cans / cans_per_box in
  let total_cost := total_boxes * cost_per_box in
    total_cost / n_family_members

theorem family_cost (n_persons_per_dozen n_dozens cans_per_person cost_per_box cans_per_box n_family_members : ℕ)
  (h1 : n_persons_per_dozen = 12) 
  (h2 : n_dozens = 5) 
  (h3 : cans_per_person = 2) 
  (h4 : cost_per_box = 2) 
  (h5 : cans_per_box = 10) 
  (h6 : n_family_members = 6) :
  cost_per_family_member n_persons_per_dozen n_dozens cans_per_person cost_per_box cans_per_box n_family_members = 4 := 
  by
    sorry

end family_cost_l814_814222


namespace smallest_sum_l814_814257

theorem smallest_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) 
  (h_fraction : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) : x + y = 64 :=
sorry

end smallest_sum_l814_814257


namespace probability_three_white_balls_l814_814498

noncomputable def probability_all_white (white black total_drawn : ℕ) : ℚ :=
  (nat.choose white total_drawn : ℚ) / (nat.choose (white + black) total_drawn : ℚ)

theorem probability_three_white_balls :
  probability_all_white 7 8 3 = 1 / 13 :=
by 
  sorry

end probability_three_white_balls_l814_814498


namespace probability_of_forming_triangle_l814_814947

noncomputable def sticks : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_triangle_combinations : List (ℕ × ℕ × ℕ) :=
  (sticks.choose 3).filter (λ t, satisfies_triangle_inequality t.1 t.2.fst t.2.snd)

theorem probability_of_forming_triangle :
  (valid_triangle_combinations.length : ℚ) / (sticks.choose 3).length = 9 / 28 :=
by
  sorry

end probability_of_forming_triangle_l814_814947


namespace complex_fraction_a_plus_b_l814_814672

noncomputable theory

def complex : Type := ℂ

theorem complex_fraction_a_plus_b :
  ∀ (a b : ℝ) (i : complex), 
  i * i = -1 →
  (a + b = 3) →
  (3 + b * i) / (1 - i) = a + b * i → a + b = 3 :=
by
  intros a b i H H_eq H'.
  sorry

end complex_fraction_a_plus_b_l814_814672


namespace number_of_correct_expressions_l814_814908

def empty_set : Set Nat := ∅
def set_with_zero : Set Nat := {0}
def el_zero : Nat := 0
def set_with_1_2_3 : Set Nat := {1, 2, 3}
def set_with_1 : Set Nat := {1}
def set_with_1_2 : Set Nat := {1, 2}
def set_ab : Set (Nat × Nat) := {⟨1, 2⟩ , ⟨2, 1⟩}

theorem number_of_correct_expressions :
  (∅ ≠ {0}) ∧
  (∅ ⊆ {0}) ∧
  (∅ ∉ {0}) ∧
  (0 ≠ {0}) ∧
  (0 ∈ {0}) ∧
  ({1} ∉ {1, 2, 3}) ∧
  ({1, 2} ⊆ {1, 2, 3}) ∧
  ({⟨1, 2⟩ , ⟨2, 1⟩} = {⟨2, 1⟩ , ⟨1, 2⟩}) → 
  4 = 4 :=
by {
  sorry
}

end number_of_correct_expressions_l814_814908


namespace range_of_m_l814_814273

def f (x : ℝ) : ℝ := 
  if x ∈ set.Icc (-2 : ℝ) (2 : ℝ) then 
    if x > 0 then 2^x - 1 else 
      if x < 0 then 1 - 2^(-x) else 0
  else 0

def g (x m : ℝ) : ℝ := x^2 - 2 * x + m

theorem range_of_m :
  ∀ m : ℝ,
  (∀ x1 ∈ set.Icc (-2 : ℝ) (2 : ℝ), ∃ x2 ∈ set.Icc (-2 : ℝ) (2 : ℝ), g x2 m = f x1) ↔ m ∈ set.Icc (-5 : ℝ) (-2 : ℝ) :=
by {
  sorry
}

end range_of_m_l814_814273


namespace pq_product_l814_814540

theorem pq_product (p q : ℝ) (root1 root2 : ℂ)
  (h1 : root1 = 2 + complex.i)
  (h2 : root2 = 2 - complex.i)
  (h3 : ∀ z : ℂ, z^2 + (p : ℂ) * z + (q : ℂ) = 0 ↔ (z = root1 ∨ z = root2)) :
  p * q = -20 := by
  -- Proof goes here
  sorry

end pq_product_l814_814540


namespace solution_set_of_inequality_l814_814070

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / (3 - x) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} := sorry

end solution_set_of_inequality_l814_814070


namespace parallelogram_angles_equal_l814_814342

/-- Given a parallelogram ABCD with points E and F on segments AD and CD
respectively, such that ∠BCE = ∠BAF. Points K and L lie on segments AD and
CD such that AK = ED and CL = FD. Prove that ∠BKD = ∠BLD. -/
theorem parallelogram_angles_equal
  {A B C D E F K L : Type*}
  [parallelogram A B C D] 
  (hEA : point_on_segment E A D)
  (hFA : point_on_segment F C D)
  (h_angles : ∠ B C E = ∠ B A F)
  (hKA : point_on_segment K A D)
  (hLA : point_on_segment L C D)
  (h_lengths1 : segment_length A K = segment_length E D)
  (h_lengths2 : segment_length C L = segment_length F D) :
  ∠ B K D = ∠ B L D :=
sorry

end parallelogram_angles_equal_l814_814342


namespace sugar_in_lollipop_l814_814017

-- Definitions based on problem conditions
def chocolate_bars := 14
def sugar_per_bar := 10
def total_sugar := 177

-- The theorem to prove
theorem sugar_in_lollipop : total_sugar - (chocolate_bars * sugar_per_bar) = 37 :=
by
  -- we are not providing the proof, hence using sorry
  sorry

end sugar_in_lollipop_l814_814017


namespace cistern_wet_surface_area_l814_814113

def length : ℝ := 6
def width : ℝ := 4
def height : ℝ := 1.25

def area_bottom := length * width
def area_longer_sides := 2 * (length * height)
def area_shorter_sides := 2 * (width * height)
def total_wet_surface_area := area_bottom + area_longer_sides + area_shorter_sides

theorem cistern_wet_surface_area : total_wet_surface_area = 49 := 
by
  sorry

end cistern_wet_surface_area_l814_814113


namespace quadrilateral_area_l814_814244

-- Definition of the problem in Lean

variables (P : Type) [affine_space P]
def regular_pentagon (p1 p2 p3 p4 p5 : P) (s : ℝ) : Prop :=
  -- Conditions for being a regular pentagon
  -- Includes equidistant vertices, equal sides of length s, and equal angles
  sorry -- Actual geometric conditions go here

def midpoint (p1 p2 : P) : P :=
  -- Function to find the midpoint of two points
  sorry -- Midpoint calculation

def apothem {P : Type} [metric_space P] [normed_add_torsor ℝ P] (p1 p2 p3 p4 p5 center : P) : ℝ :=
  -- Function to find the apothem, distance from center to midpoint of one side
  sorry -- Apothem calculation

def area (q1 q2 q3 q4 : P) : ℝ :=
  -- Function to find the area of the quadrilateral
  sorry -- Area calculation

theorem quadrilateral_area {P : Type} [affine_space P] [metric_space P] [normed_add_torsor ℝ P]
  (p1 p2 p3 p4 p5 q1 q2 q3 q4 center : P) (s a : ℝ) 
  (h_reg_pentagon : regular_pentagon p1 p2 p3 p4 p5 s)
  (h_apothem : apothem p1 p2 p3 p4 p5 center = 3)
  (h_q1 : q1 = midpoint p1 p2)
  (h_q2 : q2 = midpoint p2 p3)
  (h_q3 : q3 = midpoint p3 p4)
  (h_q4 : q4 = midpoint p4 p5) :
  area q1 q2 q3 q4 = 9 * real.tan (36 * real.pi / 180) :=
sorry

end quadrilateral_area_l814_814244


namespace find_r_l814_814002

def f (x r : ℝ) : ℝ := 3 * x ^ 4 + 2 * x ^ 3 - x ^ 2 - 5 * x + r

theorem find_r (r : ℝ) : f 3 r = 0 → r = -273 := 
by
  intros h,
  sorry

end find_r_l814_814002


namespace smallest_integer_solution_l814_814101

theorem smallest_integer_solution (x : ℤ) : (x < 3 * x - 7) → x ≥ 4 :=
begin
  sorry
end

end smallest_integer_solution_l814_814101


namespace time_3339_minutes_after_midnight_l814_814845

def minutes_since_midnight (minutes : ℕ) : ℕ × ℕ :=
  let hours := minutes / 60
  let remaining_minutes := minutes % 60
  (hours, remaining_minutes)

def time_after_midnight (start_time : ℕ × ℕ) (hours : ℕ) (minutes : ℕ) : ℕ × ℕ :=
  let (start_hours, start_minutes) := start_time
  let total_minutes := start_hours * 60 + start_minutes + hours * 60 + minutes
  let end_hours := total_minutes / 60
  let end_minutes := total_minutes % 60
  (end_hours, end_minutes)

theorem time_3339_minutes_after_midnight :
  time_after_midnight (0, 0) 55 39 = (7, 39) :=
by
  sorry

end time_3339_minutes_after_midnight_l814_814845


namespace molecular_weight_is_122_l814_814837

noncomputable def molecular_weight_of_compound := 
  let atomic_weight_C := 12.01
  let atomic_weight_H := 1.008
  let atomic_weight_O := 16.00
  7 * atomic_weight_C + 6 * atomic_weight_H + 2 * atomic_weight_O

theorem molecular_weight_is_122 :
  molecular_weight_of_compound = 122 := by
  sorry

end molecular_weight_is_122_l814_814837


namespace smallest_n_l814_814492

noncomputable def exists_two_potential_winners (num_rounds num_participants : ℕ) 
(score : ℕ → ℕ) : Prop :=
  ∃ (A B : fin num_participants),
  A ≠ B ∧ 
  let points_A := (∑ i in finset.range (num_rounds - 1), score i) + score (num_rounds - 1),
      points_B := (∑ i in finset.range (num_rounds - 1), score i) + score (num_rounds - 1) 
  in points_A ≠ points_B

theorem smallest_n : ∃ n, (∀ (ranking : fin 12 → fin n), exists_two_potential_winners 12 n ranking) ∧ n = 13 := 
sorry

end smallest_n_l814_814492


namespace exactly_one_correct_l814_814059

def stmt1 (a b : ℤ) : Prop := a + b = 0 → a = -b
def stmt2 : Prop := 3^2 = 6
def stmt3 (a : ℚ) : Prop := a > -a
def stmt4 (a b : ℤ) : Prop := abs a = abs b → a = b

theorem exactly_one_correct : 
  (∃ a b : ℤ, stmt1 a b) ∧ ¬stmt2 ∧ ¬(∃ a : ℚ, stmt3 a) ∧ ¬(∃ a b : ℤ, stmt4 a b) :=
begin
  sorry
end

end exactly_one_correct_l814_814059


namespace realize_ancient_dreams_only_C_l814_814082

-- Define the available options
inductive Options
| A : Options
| B : Options
| C : Options
| D : Options

-- Define the ancient dreams condition
def realize_ancient_dreams (o : Options) : Prop :=
  o = Options.C

-- The theorem states that only Geographic Information Technology (option C) can realize the ancient dreams
theorem realize_ancient_dreams_only_C :
  realize_ancient_dreams Options.C :=
by
  -- skip the exact proof
  sorry

end realize_ancient_dreams_only_C_l814_814082


namespace building_shadow_length_l814_814887

-- Given conditions
def h_f : ℝ := 17.5               -- Height of the flagstaff
def s_f : ℝ := 40.25              -- Length of the flagstaff's shadow
def h_b : ℝ := 12.5               -- Height of the building

-- The theorem that needs to be proven
theorem building_shadow_length :
  ∃ (s_b : ℝ), s_b = 28.75 ∧ (h_f / s_f = h_b / s_b) :=
begin
  use 28.75,
  split,
  { refl },
  { -- The following steps directly solve for the ratios equality condition
    rw [h_f, s_f, h_b],
    norm_num,
  }
end

end building_shadow_length_l814_814887


namespace range_of_x_l814_814387

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x - 4| + |x - a|

theorem range_of_x (a : ℝ) (h1 : a > 1) (h2 : ∀ x : ℝ, f x a ≥ |a - 4|) (h3 : |a - 4| = 3) :
  { x : ℝ | f x a ≤ 5 } = { x : ℝ | 3 ≤ x ∧ x ≤ 8 } := 
sorry

end range_of_x_l814_814387
